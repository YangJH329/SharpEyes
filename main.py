from fastapi import FastAPI, Request, Response, Cookie, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import sqlite3
import src.detection.pose_detector as pdf
import hashlib

# Fast API 애플리케이션 인스턴스 생성 및 템플릿 디렉토리 설정
app = FastAPI()
templates = Jinja2Templates(directory="templates")

def hash_password(password : str) -> str:
    """비밀번호를 SHA-256 해시값(64자리 문자열)으로 리턴 ."""
    return hashlib.sha256(password.encode()).hexdigest()

# 글로벌 시스템 상태 및 운동 데이터 보관소 초기화
app.state.stream_state = "running"
app.state.current_mode = "squat"
app.state.count = 0
app.state.stage = "ready"
app.state.feedback = "START EXERCISE"

# 프론트엔드 통신용 데이터 모델 정의
class ModeRequest(BaseModel):
    mode: str

class AuthSchema(BaseModel):
    username: str
    password: str

# 프론트엔드가 보낼 운동 결과 데이터 구조 정의
class WorkoutLogSchema(BaseModel) : 
    exercise_mode: str
    total_count: int
    total_time: str

# -------------------------------------------------------------
# 1. 인증 및 페이지 렌더링 라우터
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """메인 대시보드 화면을 렌더링하고 상태를 초기화합니다."""
    app.state.stream_state = "running"
    app.state.count = 0
    app.state.stage = "ready"
    app.state.feedback = "START EXERCISE"
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/register")
def register_user(data: AuthSchema):
    """새로운 사용자를 데이터베이스에 등록합니다."""
    conn = sqlite3.connect("sharpeyes.db")
    cursor = conn.cursor()
    
    try:
        # 아이디 중복 검사
        cursor.execute("SELECT id FROM users WHERE username = ?", (data.username,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")
        
        hashed_pw = hash_password(data.password)
        # 중복이 없는 경우 회원 정보 삽입
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (data.username, hashed_pw)
        )
        conn.commit()
        return {"status": "success", "message": "회원가입이 완료되었습니다."}
        
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"데이터베이스 오류: {str(e)}")
    finally:
        conn.close()

@app.post("/login")
def login_user(data: AuthSchema, response: Response):
    """사용자 인증을 처리하고 세션 쿠키를 발급합니다."""
    conn = sqlite3.connect("sharpeyes.db")
    cursor = conn.cursor()
    
    try:
        # 데이터베이스 회원 정보 조회
        cursor.execute("SELECT id, username, password FROM users WHERE username = ?", (data.username,))
        user = cursor.fetchone()
        
        # 자격 증명 검증 (비밀번호 단순 텍스트 비교)
        if not user or user[2] != hash_password(data.password):
            raise HTTPException(status_code=400, detail="아이디 또는 비밀번호가 일치하지 않습니다.")
        
        # 로그인 성공 시 클라이언트에 보안 쿠키 심기
        response.set_cookie(key="user_id", value=str(user[0]), httponly=True)
        response.set_cookie(key="username", value=user[1], httponly=True)
        
        return {"status": "success", "message": f"{user[1]}님 환영합니다."}
        
    finally:
        conn.close()

@app.post("/logout")
def logout_user(response: Response):
    """인증 쿠키를 삭제하여 로그아웃 처리합니다."""
    response.delete_cookie(key="user_id")
    response.delete_cookie(key="username")
    return {"status": "success", "message": "로그아웃 되었습니다."}

# -------------------------------------------------------------
# 2. 실시간 비전 스트리밍 및 비디오 제어 라우터
# -------------------------------------------------------------
@app.get("/video_feed")
def video_feed():
    """현재 설정된 전역 모드를 분석 엔진에 동적으로 주입하여 스트리밍합니다."""
    return StreamingResponse(
        pdf.run_pose_estimation(video_path=0, mode=app.state.current_mode, app_state=app),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/toggle")
def toggle_stream():
    """웹캠 영상 분석 루프의 가동 및 일시정지 상태를 토글합니다."""
    if app.state.stream_state == "running":
        app.state.stream_state = "stopped"
    else:
        app.state.stream_state = "running"
    return {"status": app.state.stream_state}

# -------------------------------------------------------------
# 3. 데이터 송수신 및 환경 제어 라우터
# -------------------------------------------------------------
@app.post("/set_mode")
def set_mode(data: ModeRequest):
    """운동 종류 모드를 스위칭하고 관련 상태 지표를 초기화합니다."""
    app.state.current_mode = data.mode
    app.state.count = 0
    app.state.stage = "ready"
    app.state.feedback = "START EXERCISE"
    print(f">>> [System] 운동 모드 변경 완료: {data.mode.upper()}")
    return {"status": "success", "current_mode": app.state.current_mode}

@app.get("/get_data")
def get_data():
    """프론트엔드 폴링 주기에 맞춰 실시간 분석 지표를 전송합니다."""
    return {
        "count": app.state.count,
        "stage": app.state.stage.upper(),
        "feedback": app.state.feedback
    }

# -------------------------------------------------------------
# 4. 운동 기록 저장 API (/save_workout)
# -------------------------------------------------------------
@app.post("/save_workout")
def save_workout(data : WorkoutLogSchema, user_id : str = Cookie(None)):
    """운동 기록을 데이터베이스에 저장합니다."""

    # 만약 쿠키에 user_id가 없다면 인증되지 않은 요청으로 간주하여 에러 반환
    if not user_id :
        raise HTTPException(status_code = 401, detail="로그인이 필요한 서비스 입니다.")
    
    conn = sqlite3.connect("sharpeyes.db")
    cursor = conn.cursor()
    
    try:
        # workout_logs 테이블에 운동 기록 삽입
        cursor.execute("""
            INSERT INTO workout_logs (user_id, exercise_mode, total_count, total_time)
            VALUES (?, ?, ?, ?)
        """, (int(user_id), data.exercise_mode, data.total_count, data.total_time))
        
        conn.commit()
        return {"status": "success", "message": "운동 기록이 저장되었습니다."}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"데이터베이스 오류: {str(e)}")
    finally:
        conn.close()
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)