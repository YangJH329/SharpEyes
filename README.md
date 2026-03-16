# SharpEyes 🦅
**AI 기반 실시간 스포츠 구역 이탈 판정 시스템**

## 1. 프로젝트 개요
- **배경**: 고가의 장비 없이 웹캠 한 대와 AI를 활용한 저비용 스포츠 판독 시스템

- **주요 기능**: 
  - 사용자 정의 관심 영역(ROI) 설정
  - YOLOv8/11 기반 객체(사람/공) 실시간 추적
  - 다각형 영역 내 존재 여부 자동 판정 (In/Out)

## 2. 기술 스택
- Language: Python 3.10+
- Library: OpenCV, Ultralytics(YOLO), MediaPipe, NumPy

## 3. 실행 방법
1. 필수 라이브러리 설치: `pip install -r requirements.txt`
2. 메인 프로그램 실행: `python src/main.py`

## 4. 개발 일정 (예정.)
- 1~8주차: 핵심 알고리즘 개발 및 중간 보고
- 9~15주차: UI 고도화 및 최종 프로토타입 완성
