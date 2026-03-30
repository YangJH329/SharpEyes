import cv2
from ultralytics import YOLO
# import os # 파일 존재 여부 확인용
from src.utils.math_utils import calculate_angle

def run_pose_estimation(video_path=0):
    # # 파일이 진짜 있는지 검사하는 코드 추가 # 파일 경로 찾기용
    # if isinstance(video_path, str) and not os.path.exists(video_path):
    #     print(f" 에러: '{video_path}' 파일을 찾을 수 없습니다!")
    #     print(f"현재 위치: {os.getcwd()}")
    # 1. 모델 로드
    model = YOLO('yolov8n-pose.pt')

    # 2. 영상 소스 불러오기
    cap = cv2.VideoCapture(video_path)

    print("프로그램 시작: ESC를 누르면 종료됩니다.")

    # --- [카운팅 변수 초기화] ---
    counter = 0    # 스쿼트 개수
    stage = None   # 현재 상태 ('down': 내려감, 'up': 올라옴)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 3. 모델로 포즈 추론
        results = model(frame, stream=True)

        for r in results:
            # 기본 뼈대 그리기
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
                        # 각도 계산
                        angle = calculate_angle(hip, knee, ankle)
                        
                        # --- [카운팅 알고리즘 적용] ---
                        # 1. 내려가는 동작 인식 (기준: 90도 미만)
                        if angle < 90:
                            stage = "down"
                        
                        # 2. 올라오는 동작 인식 및 카운트 (기준: 160도 이상이고 이전에 내려갔어야 함)
                        if angle > 160 and stage == "down":
                            stage = "up"
                            counter += 1
                            print(f"스쿼트 성공! 현재 개수: {counter}")
                        # ----------------------------

                        # 텍스트 시각화 (화면 표시)
                        # 1. 실시간 무릎 각도 (노란색)
                        cv2.putText(annotated_frame, f"Knee Angle: {int(angle)}deg", 
                                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        
                        # 2. 스쿼트 개수 (초록색, 크게 표시)
                        cv2.putText(annotated_frame, f"COUNT: {counter}", 
                                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                        # 3. 상태 피드백 (깊게 내려갔을 때 표시)
                        if stage == "down":
                            cv2.putText(annotated_frame, "DEPTH OK!", (30, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 화면에 출력 (루프 안에서 호출되어야 실시간으로 보임)
            cv2.imshow("SharpEyes - Squat Counter", annotated_frame)

        # 'ESC' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 데이터 폴더 내 영상 경로
    video_path = "data/raw/squat.mp4" 
    run_pose_estimation(video_path)