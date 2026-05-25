import cv2
from ultralytics import YOLO
from src.utils.math_utils import calculate_angle

def run_pose_estimation(video_path=0, mode="squat", app_state=None):
    """
    운동 자세 분석 및 카운팅을 수행하고, 웹 스트리밍을 위한 바이너리 프레임을 yield합니다.
    - try...finally 구문을 통해 어떤 상황에서도 카메라 자원 해제(release)를 보장합니다.
    """
    # 1. 모델 로드 (가벼운 YOLOv8n-pose 사용)
    model = YOLO('yolov8n-pose.pt')

    # 2. 영상 소스 불러오기 (Windows 환경에서의 안정성을 위한 백엔드 설정)
    if video_path == 0: 
        cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)  
    else:
        cap = cv2.VideoCapture(video_path)

    # 카운팅 변수 및 상태 머신 임계값 초기화
    counter = 0    
    stage = "ready" 
    feedback_msg = "START EXERCISE"

    # [운동별 임계값 및 관절 인덱스 설정 데이터]
    if mode == "squat":
        p_indices = [12, 14, 16] # 골반, 무릎, 발목
        DOWN_ANGLE_LIMIT = 95
        UP_ANGLE_LIMIT   = 160
        DOWN_DIST_LIMIT  = 30   # 골반-무릎 수직 거리
        UP_DIST_LIMIT    = 100
    else: # pullup
        p_indices = [6, 8, 10]  # 어깨, 팔꿈치, 손목
        DOWN_ANGLE_LIMIT = 85   
        DOWN_DIST_LIMIT  = 60   # 어깨-손목 수직 거리
        UP_ANGLE_LIMIT   = 150  
        UP_DIST_LIMIT    = 90   

    # 예외 발생 시에도 안전하게 자원을 해제하기 위한 try 블록
    try:
        while cap.isOpened():
            # 핵심 안전 장치 : main.py에서 이탈 플래그가 False로 설정되면 즉시 루프 탈출하여 자원 해제
            if app_state and not app_state.state.is_running:
                print(">>> [SharpEyes] 웹 인터페이스로부터 종료 신호를 수신하여 스트리밍 루프를 안전하게 종료합니다.")
                break
            
            # 카메라 프레임 읽기
            success, frame = cap.read()
            if not success: 
                break
            
            # [안전 장치] 사람이 없어도 최소한 원본 화면(frame)은 전송되도록 초기화 (엑스박스 방지)
            annotated_frame = frame

            # YOLO 포즈 추정 (stream=True로 메모리 효율화)
            results = model(frame, stream=True)

            for r in results:
                # 감지된 사람의 뼈대 프레임 그리기
                annotated_frame = r.plot()

                if r.keypoints is not None and len(r.keypoints.xy) > 0:
                    keypoints = r.keypoints.xy[0] 

                    # 필요한 핵심 관절 포인트들이 신뢰도 높게 잡혔는지 확인
                    if len(keypoints) > 16:
                        # 모드별 관절 좌표 추출
                        p1 = keypoints[p_indices[0]].tolist() 
                        p2 = keypoints[p_indices[1]].tolist() 
                        p3 = keypoints[p_indices[2]].tolist() 

                        # 화면 왜곡이나 미감지 등으로 인한 제로 좌표 필터링
                        if all(p[0] > 0 for p in [p1, p2, p3]):
                            # 1. 하이브리드 교차 검증 데이터 계산 (사잇각 및 수직 변위)
                            angle = calculate_angle(p1, p2, p3)
                            
                            if mode == "squat":
                                vertical_dist = abs(p2[1] - p1[1]) # 골반-무릎 (y축 변위)
                            else:
                                vertical_dist = abs(p3[1] - p1[1]) # 어깨-손목 (y축 변위)

                            # --- [하이브리드 카운팅 상태 머신] ---
                            
                            # STEP 1: 정점(Action) 판정
                            if stage == "ready" or stage == "return":  
                                if (angle < DOWN_ANGLE_LIMIT) or (vertical_dist < DOWN_DIST_LIMIT):
                                    stage = "action"
                                    feedback_msg = "DEPTH OK!" if mode == "squat" else "PULL OK!"

                            # STEP 2: 시작자세 복귀 및 카운트 증가 (노이즈 방지를 위해 AND 조건 처리)
                            elif stage == "action":
                                if (angle > UP_ANGLE_LIMIT) and (vertical_dist > UP_DIST_LIMIT):
                                    stage = "return"
                                    counter += 1
                                    feedback_msg = f"SUCCESS! COUNT: {counter}"
                                    
                            # --- [예외/피드백 확장용 뼈대] ---
                            # 가동범위 미달이나 불안정한 프레임에 대한 가이드는 추후 여기에 살을 붙입니다.

                            # ----------------------------------------------
                            # 실시간 대시보드 그래픽 오버레이 (UI Overlay)
                            cv2.putText(annotated_frame, f"COUNT: {counter}", 
                                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            
                            status_color = (0, 0, 255) if stage == "action" else (255, 255, 0)
                            cv2.putText(annotated_frame, f"STAGE: {stage.upper()}", 
                                        (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                            cv2.putText(annotated_frame, f"A: {int(angle)} / D: {int(vertical_dist)}", 
                                        (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                            # 실시간 사용자 가이드 피드백 텍스트 출력
                            cv2.putText(annotated_frame, feedback_msg, (30, 190), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 153, 51), 2)

            # 웹 브라우저 전송을 위한 JPEG 인코딩 및 바이너리 스트림 변환 (MJPEG 표준 규격)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            
            # 멀티파트 규격 헤더에 맞춰 프레임 바이트를 순차적으로 밀어줌
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        # 브라우저 종료, 예외 발생 등 어떤 상황에서도 카메라 장치 자원 반납을 무조건 보장
        print(">>> [SharpEyes] 비디오 스트리밍 루프를 탈출하여 카메라 자원을 안전하게 해제합니다.")
        cap.release()