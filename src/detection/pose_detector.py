import cv2
from ultralytics import YOLO
# import os # 파일 존재 여부 확인하느라 쓴 모듈, 필요하면 주석 해제

def run_pose_estimation(video_path=0):
    # # 파일이 진짜 있는지 검사하는 코드 추가 # 파일 경로 찾기용
    # if isinstance(video_path, str) and not os.path.exists(video_path):
    #     print(f" 에러: '{video_path}' 파일을 찾을 수 없습니다!")
    #     print(f"현재 위치: {os.getcwd()}")
    #     return

    # 1. 모델 로드 (가장 가벼운 'yolov8n-pose' 모델 사용)
    # 처음 실행 시 모델 파일(.pt)을 자동으로 다운로드합니다.
    model = YOLO('yolov8n-pose.pt')

    # 2. 영상 소스 불러오기 (0은 웹캠, 파일 경로를 넣으면 비디오 파일)
    cap = cv2.VideoCapture(video_path)

    print("프로그램 시작: ESC를 누르면 종료됩니다.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 3. 모델로 포즈 추론 (stream=True는 실시간 처리에 최적화)
        results = model(frame, stream=True)

        for r in results:
            annotated_frame = r.plot()

            # 관절 데이터가 있는지 확인
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                # 0번 사람의 모든 관절 좌표 가져오기
                keypoints = r.keypoints.xy[0] 

                # 오른쪽 무릎(인덱스 14) 좌표 추출
                # keypoints[14]는 [x, y] 형태의 리스트입니다.
                if len(keypoints) > 14:
                    rk_x, rk_y = keypoints[14]

                    # 좌표가 0, 0이 아닐 때만(인식되었을 때만) 화면에 표시
                    if rk_x > 0 and rk_y > 0:
                        text = f"Right Knee: ({int(rk_x)}, {int(rk_y)})"
                        
                        # 화면 좌측 상단에 텍스트 그리기 (OpenCV 함수)
                        cv2.putText(annotated_frame, text, (30, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # 터미널에도 실시간 출력
                        print(f"오른쪽 무릎 위치 -> X: {rk_x:.1f}, Y: {rk_y:.1f}")

        # 화면에 출력
        cv2.imshow("SharpEyes - Pose Test", annotated_frame)

        # 'ESC' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 테스트하고 싶은 영상 파일 경로를 넣으세요. 
    # 웹캠으로 하려면 0을 넣으면 됩니다.
    video_path = "data/raw/test.mp4"  # 웹캠 사용
    run_pose_estimation(video_path)