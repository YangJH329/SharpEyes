import sqlite3

def init_database():
    # 프로젝트 폴더 내에 sharpeyes.db 파일을 생성하고 연결합니다.
    # 파일이 이미 존재하면 연결만 하고, 없으면 새로 생성합니다.
    conn = sqlite3.connect("sharpeyes.db")
    cursor = conn.cursor()

    # 1. 사용자 정보가 담길 users 테이블을 생성합니다.
    # id: 고유 식별 번호 (자동으로 1씩 증가)
    # username: 중복을 허용하지 않는 사용자 아이디
    # password: 로그인 검증을 위한 비밀번호
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)

    # 2. 운동 데이터가 누적될 workout_logs 테이블을 생성합니다.
    # user_id: users 테이블의 id 번호와 매핑되어 누가 운동했는지 식별하는 외래키
    # exercise_mode: squat 또는 pullup 문자열 저장
    # total_count: 최종 성공 세트 개수
    # total_time: MM:SS 포맷의 운동 경과 시간
    # timestamp: 데이터가 삽입된 날짜와 시간 자동 기록
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workout_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            exercise_mode TEXT NOT NULL,
            total_count INTEGER NOT NULL,
            total_time TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # 테이블 구조 변경 사항을 최종 물리 저장소에 반영합니다.
    conn.commit()
    
    # 데이터베이스 연결을 안전하게 닫아줍니다.
    conn.close()
    print("Database and Tables initialized successfully!")

if __name__ == "__main__":
    init_database()