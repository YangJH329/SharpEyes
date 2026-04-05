import cv2
from ultralytics import YOLO
from src.utils.math_utils import calculate_angle

def run_pose_estimation(video_path=0, mode="squat"):
    # 1. 모델 로드
    model = YOLO('yolov8n-pose.pt')

    # 2. 영상 소스 불러오기
    cap = cv2.VideoCapture(video_path)

    print(f"[{mode.upper()}] 모드 시작: ESC를 누르면 종료됩니다.")

    # --- [카운팅 변수 및 임계값 초기화] ---
    counter = 0    
    stage = "ready" 

    # [운동별 설정 데이터]
    if mode == "squat":
        p_indices = [12, 14, 16] # 골반, 무릎, 발목
        DOWN_ANGLE_LIMIT = 95
        UP_ANGLE_LIMIT   = 160
        DOWN_DIST_LIMIT  = 30   # 골반-무릎 수직 거리
        UP_DIST_LIMIT    = 100
    else: # pullup
        p_indices = [6, 8, 10]  # 어깨, 팔꿈치, 손목
        # 정점 도달(수축) 임계값
        DOWN_ANGLE_LIMIT = 85   # 170 -> 85 미만 시 정점
        DOWN_DIST_LIMIT  = 60   # 110 -> 60 미만 시 정점 (어깨-손목)
        # 시작자세 복귀(이완) 임계값 - 너무 높으면 카운트가 안 되므로 넉넉히 설정
        UP_ANGLE_LIMIT   = 150  # 다시 150도 이상 펴지면 복귀
        UP_DIST_LIMIT    = 90   # 다시 90px 이상 멀어지면 복귀

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = model(frame, stream=True)

        for r in results:
            annotated_frame = r.plot()

            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                keypoints = r.keypoints.xy[0] 

                if len(keypoints) > 16:
                    # 관절 추출
                    p1 = keypoints[p_indices[0]].tolist() # Squat: Hip / Pullup: Shoulder
                    p2 = keypoints[p_indices[1]].tolist() # Squat: Knee / Pullup: Elbow
                    p3 = keypoints[p_indices[2]].tolist() # Squat: Ankle / Pullup: Wrist

                    if all(p[0] > 0 for p in [p1, p2, p3]):
                        # 1. 공통 데이터 계산 (각도는 1-2-3 사잇각)
                        angle = calculate_angle(p1, p2, p3)
                        
                        # 2. 수직 거리 계산 (JH님 요청 반영: 모드별 대상 분기)
                        if mode == "squat":
                            vertical_dist = abs(p2[1] - p1[1]) # 골반-무릎 (y축)
                        else:
                            vertical_dist = abs(p3[1] - p1[1]) # 어깨-손목 (y축)

                        # --- [하이브리드 카운팅 알고리즘] ---

                        # STEP 1: 정점(Action) 판정 (READY/RETURN -> ACTION)
                        if stage == "ready" or stage == "return":  
                            # 각도가 좁아지거나(측면), 거리가 짧아지면(정면) 정점 인정
                            if (angle < DOWN_ANGLE_LIMIT) or (vertical_dist < DOWN_DIST_LIMIT):
                                stage = "action"
                                print(f">>> {mode.upper()} 정점(Action) 진입")

                        # STEP 2: 복귀 및 카운트 (ACTION -> RETURN)
                        elif stage == "action":
                            # 각도가 커지고(펴짐) AND 거리도 충분히 멀어져야 카운트 (노이즈 방지)
                            if (angle > UP_ANGLE_LIMIT) and (vertical_dist > UP_DIST_LIMIT):
                                stage = "return"
                                counter += 1
                                print(f"★ {mode.upper()} 성공! COUNT: {counter} ★")

                        # ----------------------------------------------

                        # 시각화 (UI)
                        cv2.putText(annotated_frame, f"COUNT: {counter}", 
                                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        
                        # 상태 표시 (Action 상태일 때 빨간색으로 경고)
                        status_color = (0, 0, 255) if stage == "action" else (255, 255, 0)
                        cv2.putText(annotated_frame, f"STAGE: {stage.upper()}", 
                                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                        # 디버깅 데이터 출력 (중요: 이 수치를 보고 LIMIT를 조정하세요)
                        cv2.putText(annotated_frame, f"A: {int(angle)} / D: {int(vertical_dist)}", 
                                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        if stage == "action":
                            msg = "DEPTH OK!" if mode == "squat" else "PULL OK!"
                            cv2.putText(annotated_frame, msg, (30, 190), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(f"SharpEyes - {mode.capitalize()} Counter", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # "squat" 또는 "pullup"으로 테스트해보세요.
    run_pose_estimation(video_path="data/raw/squat.mp4", mode="squat")