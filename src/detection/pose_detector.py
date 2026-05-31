import cv2
from ultralytics import YOLO
from src.utils.math_utils import calculate_angle

def run_pose_estimation(video_path=0, mode="squat", app_state=None):
    """
    종료 신호 감지 시 루프를 완전히 이탈하여 카메라 장치를 해제(release)합니다.
    """
    model = YOLO('yolov8n-pose.pt')

    if video_path == 0: 
        cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)  
    else:
        cap = cv2.VideoCapture(video_path)

    counter = 0    
    stage = "ready" 
    feedback_msg = "START EXERCISE"

    if mode == "squat":
        p_indices = [12, 14, 16]
        DOWN_ANGLE_LIMIT = 95
        UP_ANGLE_LIMIT   = 160
        DOWN_DIST_LIMIT  = 30   
        UP_DIST_LIMIT    = 100
    else:
        p_indices = [6, 8, 10]
        DOWN_ANGLE_LIMIT = 85   
        DOWN_DIST_LIMIT  = 60   
        UP_ANGLE_LIMIT   = 150  
        UP_DIST_LIMIT    = 90   

    try:
        while cap.isOpened():
            # 웹 인터페이스에서 정지(stopped)를 요청하면 루프를 부수고 나갑니다.
            if app_state and app_state.state.stream_state == "stopped":
                print(">>> [SharpEyes] 정지 신호 수신. 루프를 이탈합니다.")
                break

            success, frame = cap.read()
            if not success: 
                break

            annotated_frame = frame
            results = model(frame, stream=True)

            for r in results:
                annotated_frame = r.plot()
                if r.keypoints is not None and len(r.keypoints.xy) > 0:
                    keypoints = r.keypoints.xy[0] 
                    if len(keypoints) > 16:
                        p1 = keypoints[p_indices[0]].tolist() 
                        p2 = keypoints[p_indices[1]].tolist() 
                        p3 = keypoints[p_indices[2]].tolist() 

                        if all(p[0] > 0 for p in [p1, p2, p3]):
                            angle = calculate_angle(p1, p2, p3)
                            if mode == "squat":
                                vertical_dist = abs(p2[1] - p1[1])
                            else:
                                vertical_dist = abs(p3[1] - p1[1])

                            if stage == "ready" or stage == "return":  
                                if (angle < DOWN_ANGLE_LIMIT) or (vertical_dist < DOWN_DIST_LIMIT):
                                    stage = "action"
                                    feedback_msg = "DEPTH OK!" if mode == "squat" else "PULL OK!"

                            elif stage == "action":
                                if (angle > UP_ANGLE_LIMIT) and (vertical_dist > UP_DIST_LIMIT):
                                    stage = "return"
                                    counter += 1
                                    feedback_msg = f"SUCCESS! COUNT: {counter}"

                            cv2.putText(annotated_frame, f"COUNT: {counter}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            status_color = (0, 0, 255) if stage == "action" else (255, 255, 0)
                            cv2.putText(annotated_frame, f"STAGE: {stage.upper()}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                            cv2.putText(annotated_frame, f"A: {int(angle)} / D: {int(vertical_dist)}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                            cv2.putText(annotated_frame, feedback_msg, (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 153, 51), 2)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        # 루프가 깨지면 무조건 카메라 장치를 OS에 반납합니다. (불빛 꺼짐)
        print(">>> [SharpEyes] 카메라 하드웨어 연결을 종료합니다.")
        cap.release()