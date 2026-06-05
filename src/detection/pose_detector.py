import cv2
from ultralytics import YOLO
from src.utils.math_utils import calculate_angle

def run_pose_estimation(video_path=0, mode="squat", app_state=None):
    """
    YOLOv8 기반의 실시간 자세 인식 및 운동 카운팅 제너레이터 함수입니다.  
        Args:
- video_path: 0 (웹캠) 또는 비디오 파일 경로
- mode: "squat" 또는 "pullup"
- app_state: FastAPI 전역 상태 객체
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
            if app_state and app_state.state.stream_state == "stopped":
                print(">>> [SharpEyes] 정지 신호 수신. 루프를 이탈합니다.")
                break

            success, frame = cap.read()
            if not success: 
                break

            annotated_frame = frame
            results = model(frame, stream=True)

            for r in results:
                # 영상에는 오직 깔끔한 YOLOv8 스켈레톤 라인만 그립니다.
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

                            # --- [하이브리드 카운팅 알고리즘] ---
                            if stage == "ready" or stage == "return":  
                                if (angle < DOWN_ANGLE_LIMIT) or (vertical_dist < DOWN_DIST_LIMIT):
                                    stage = "action"
                                    feedback_msg = "DEPTH OK!" if mode == "squat" else "PULL OK!"

                            elif stage == "action":
                                if (angle > UP_ANGLE_LIMIT) and (vertical_dist > UP_DIST_LIMIT):
                                    stage = "return"
                                    counter += 1
                                    feedback_msg = f"SUCCESS! COUNT: {counter}"

                            #  [핵심 연동] 계산된 실시간 변수들을 FastAPI 전역 state에 실시간 저장
                            if app_state:
                                app_state.state.count = counter
                                app_state.state.stage = stage
                                app_state.state.feedback = feedback_msg

            # cv2.putText() 파트 완전 삭제 (영상의 순수화 및 CPU 부하 경감)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        print(">>> [SharpEyes] 카메라 하드웨어 연결을 종료합니다.")
        cap.release()