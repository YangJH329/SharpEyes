# 🦅 SharpEyes: Intelligent Home-Training Safety System

**SharpEyes**는 YOLOv8 기반의 **Pose Estimation(자세 추정)**과 **Instance Segmentation(영역 분할)** 기술을 결합하여, 사용자의 운동 자세를 교정함과 동시에 지정된 안전 구역 이탈을 실시간으로 감지하는 지능형 홈 트레이닝 보조 시스템입니다.

---

##  프로젝트 개요 (Project Overview)
본 프로젝트는 일반적인 웹캠 환경에서 사용자의 신체 데이터와 물리적 운동 공간을 동시에 분석하는 것을 목표로 합니다. 
1. **Pose Estimation**: 17개 주요 관절 좌표를 추출하여 스쿼트 등 운동 횟수를 정밀하게 카운팅합니다.
2. **Image Segmentation**: 사용자의 운동 구역(예: 요가 매트)을 픽셀 단위로 분할하여 인식합니다.
3. **Safety Analysis**: 추출된 관절 좌표가 인식된 매트 구역을 벗어날 경우(Out-of-Bounds) 실시간 경고를 제공하여 운동 중 부상을 방지합니다.

---

## 🛠 기술 스택 (Tech Stack)
* **AI Model**: Ultralytics YOLOv8 (Pose & Segmentation)
* **Computer Vision**: OpenCV, NumPy
* **Language**: Python 3.11+
* **Version Control**: Git / GitHub (Feature Branch Strategy)
* **Environment**: Virtual Environment (venv)

---

##  핵심 로직 (Core Logic)

### 1. 지능형 스쿼트 카운팅 (State Machine)
단순 각도 측정이 아닌 **상태 머신(READY - DOWN - UP)** 구조를 설계하여 중복 카운팅 노이즈를 제거했습니다.
* **Multi-Modal Analysis**: 측면(각도 기반)과 정면(수직 거리 기반) 데이터를 결합한 하이브리드 판정 알고리즘을 적용하여 다양한 카메라 각도에 대응합니다.

### 2. 구역 이탈 판정 (Spatial Boundary Detection)
* **Segmentation Mask**: YOLOv8-seg 모델이 바닥의 운동 매트 영역을 바이너리 마스크(Binary Mask)로 실시간 생성합니다.
* **Coordinate Mapping**: 추출된 사용자의 발목(Ankle) 및 발(Foot) 좌표가 해당 마스크 영역(픽셀 값 1) 내부에 존재하는지 기하학적으로 연산합니다.
* **Real-time Feedback**: 이탈 감지 시 시각적/청각적 피드백을 통해 사용자의 위치 수정을 유도합니다.

---

## 📂 프로젝트 구조 (Project Structure)
```text
SharpEyes/
├── data/               # 테스트용 영상 및 이미지 데이터
├── src/
│   ├── detection/      # Pose & Segmentation 추론 및 판정 로직
│   └── utils/          # 각도 계산 및 기하학 연산 유틸리티
├── venv/               # 가상환경
├── requirements.txt    # 의존성 패키지 목록
└── README.md           # 프로젝트 정의서