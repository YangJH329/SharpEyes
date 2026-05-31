from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import src.detection.pose_detector as pdf

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 글로벌 가동 상태 플래그 ("running" 또는 "stopped")
app.state.stream_state = "running"

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    app.state.stream_state = "running"
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
def video_feed():
    """
    브라우저가 이 엔드포인트를 호출할 때마다 
    pose_detector 내부에서 cv2.VideoCapture가 새로 생성(카메라 ON)됩니다.
    """
    return StreamingResponse(
        pdf.run_pose_estimation(video_path=0, mode="squat", app_state=app),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/toggle")
def toggle_stream():
    """상태를 반전시키는 엔드포인트"""
    if app.state.stream_state == "running":
        app.state.stream_state = "stopped"
        print(">>> [System] 카메라 가동 정지 (Camera OFF)")
    else:
        app.state.stream_state = "running"
        print(">>> [System] 카메라 가동 재개 (Camera ON)")
    return {"status": app.state.stream_state}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)