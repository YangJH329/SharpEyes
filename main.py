from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import src.detection.pose_detector as pdf

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 1. 글로벌 시스템 상태 및 운동 데이터 보관소 초기화
app.state.stream_state = "running"
app.state.count = 0
app.state.stage = "ready"
app.state.feedback = "START EXERCISE"

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # 페이지 새로고침 시 데이터 초기화
    app.state.stream_state = "running"
    app.state.count = 0
    app.state.stage = "ready"
    app.state.feedback = "START EXERCISE"
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        pdf.run_pose_estimation(video_path=0, mode="squat", app_state=app),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/toggle")
def toggle_stream():
    if app.state.stream_state == "running":
        app.state.stream_state = "stopped"
        print(">>> [System] 카메라 가동 정지 (Camera OFF)")
    else:
        app.state.stream_state = "running"
        print(">>> [System] 카메라 가동 재개 (Camera ON)")
    return {"status": app.state.stream_state}

#  2. 자바스크립트가 0.1초마다 요청할 실시간 데이터 전송 라우터 신설
@app.get("/get_data")
def get_data():
    """우측 사이드바 대시보드 갱신용 JSON 데이터 반환 API"""
    return {
        "count": app.state.count,
        "stage": app.state.stage.upper(),
        "feedback": app.state.feedback
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)