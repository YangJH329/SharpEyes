import cv2
from ultralytics import YOLO
from src.utils.math_utils import calculate_angle

def run_pose_estimation(video_path=0):
    # 1. 모델 로드
    model = YOLO('yolov8n-pose.pt')

    # 2. 영상 소스 불러오기
    cap = cv2.VideoCapture(video_path)

    print("프로그램 시작: ESC를 누르면 종료됩니다.")

    # --- [카운팅 변수 및 임계값 초기화] ---
    counter = 0    
    stage = "ready"   # 초기 상태에는 '준비' 상태로 시작 

    ## [하이브리드 카운팅 알고리즘을 위한 임계값]
    # 환경에 따라 아래 수치들을 미세 조정할것.
    # 각도 (골반-무릎-발목), 주로 측면에서의 자세 판단에 활용됨 
    DOWN_ANGLE_LIMIT = 95    # 이 각도보다 작아지면 DOWN
    UP_ANGLE_LIMIT   = 160   # 이 각도보다 커지면 UP

    # 수직 거리 (골반-무릎), 주로 정면에서의 깊이 판단에 활용됨
    DOWN_DIST_LIMIT  = 30    # 이 수직 거리보다 작아지면 DOWN
    UP_DIST_LIMIT    = 100   # 이 수직 거리보다 커지면 UP

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 3. 모델로 포즈 추론
        results = model(frame, stream=True)

        for r in results:
            annotated_frame = r.plot()

            # 관절 데이터 추출 및 카운팅 로직
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                keypoints = r.keypoints.xy[0] 

                # 인덱스 : 골반 (12), 무릎 (14), 발목 (16)
                if len(keypoints) > 16:
                    hip = keypoints[12].tolist()   
                    knee = keypoints[14].tolist()
                    ankle = keypoints[16].tolist()

                    if all(p[0] > 0 for p in [hip, knee, ankle]):
                        # 데이터 계산
                        angle = calculate_angle(hip, knee, ankle)
                        vertical_dist = abs(knee[1] - hip[1]) # 골반-무릎 수직 거리

                        # --- [하이브리드 카운팅 알고리즘] ---
                        
                        # 1. 하강 판정 (UP -> DOWN)
                        # 측면(각도) 혹은 정면(거리) 중 하나라도 만족하면 상태 변경
                        # 초기에는 '대기' 상태에서 시작, UP 상태에서만 DOWN으로 넘어갈 수 있도록 설계
                        if stage == "up" or stage == "wait":  
                            if (angle < DOWN_ANGLE_LIMIT) or (vertical_dist < DOWN_DIST_LIMIT):
                                stage = "down"
                                print(">>> DOWN 감지")

                        # 2. 상승 판정 및 카운트 (DOWN -> UP)
                        # 노이즈 방지를 위해 두 지표가 모두 회복되었을 때만 카운트
                        elif stage == "down":
                            if (angle > UP_ANGLE_LIMIT) and (vertical_dist > UP_DIST_LIMIT):
                                stage = "up"
                                counter += 1
                                print(f"★ 스쿼트 성공! 현재 개수: {counter} ★")
                        
                        # ----------------------------

                        # 텍스트 시각화 (화면 표시)
                        # 1. 스쿼트 개수 (초록색)
                        cv2.putText(annotated_frame, f"COUNT: {counter}", 
                                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        
                        # 2. 현재 상태 표시 (내려갔을 때 빨간색)
                        status_color = (0, 0, 255) if stage == "down" else (255, 255, 0)
                        cv2.putText(annotated_frame, f"STAGE: {stage.upper()}", 
                                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                        # 3. 디버깅 데이터 (A: 각도, D: 수직거리)
                        # 이 숫자를 보고 위 임계값(LIMIT)을 조정하세요
                        cv2.putText(annotated_frame, f"A: {int(angle)} / D: {int(vertical_dist)}", 
                                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        if stage == "down":
                            cv2.putText(annotated_frame, "DEPTH OK!", (30, 190), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 실시간 화면 출력
            cv2.imshow("SharpEyes - Squat Counter", annotated_frame)

        # 'ESC' 키 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "data/raw/squat.mp4" 
    run_pose_estimation(video_path)