from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import src.detection.pose_detector as pdf

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 글로벌 시스템 상태 및 운동 데이터 보관소 초기화
app.state.stream_state = "running"
app.state.current_mode = "squat"  # 기본값: squat
app.state.count = 0
app.state.stage = "ready"
app.state.feedback = "START EXERCISE"

# 모드 변경 요청을 받기 위한 데이터 모델 선언
class ModeRequest(BaseModel):
    mode: str

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    app.state.stream_state = "running"
    app.state.count = 0
    app.state.stage = "ready"
    app.state.feedback = "START EXERCISE"
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
def video_feed():
    # 현재 설정된 전역 모드(squat 또는 pullup)를 분석 엔진에 동적으로 주입
    return StreamingResponse(
        pdf.run_pose_estimation(video_path=0, mode=app.state.current_mode, app_state=app),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/toggle")
def toggle_stream():
    if app.state.stream_state == "running":
        app.state.stream_state = "stopped"
    else:
        app.state.stream_state = "running"
    return {"status": app.state.stream_state}

# 웹 페이지에서 운동 종류를 바꿀 때 호출할 API 신설
@app.post("/set_mode")
def set_mode(data: ModeRequest):
    """프론트엔드에서 선택한 운동 모드로 서버 전역 설정을 변경하고 상태를 초기화합니다."""
    app.state.current_mode = data.mode
    app.state.count = 0
    app.state.stage = "ready"
    app.state.feedback = "START EXERCISE"
    print(f">>> [System] 운동 모드 변경 완료: {data.mode.upper()}")
    return {"status": "success", "current_mode": app.state.current_mode}

@app.get("/get_data")
def get_data():
    return {
        "count": app.state.count,
        "stage": app.state.stage.upper(),
        "feedback": app.state.feedback
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)