from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import src.detection.pose_detector as pdf

app = FastAPI()

# 핵심: FastAPI 내장 state 구조에 전역 변수 등록
app.state.is_running = True

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # 페이지를 새로 열 때 상태를 다시 True로 초기화
    app.state.is_running = True 
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
def video_feed():
    # app 객체 통째로 넘겨주기
    return StreamingResponse(
        pdf.run_pose_estimation(video_path=0, mode="squat", app_state=app),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/stop")
def stop_streaming():
    """웹의 종료 버튼을 누르면 호출되는 엔드포인트"""
    #  버튼 누르면 state 내부 값을 False로 변경
    app.state.is_running = False
    return {"status": "stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)