import sqlite3
import os

def view_users():
    # 스크립트 파일의 현재 위치(__file__)를 기준으로 한 단계 상위 폴더(루트)에 있는 sharpeyes.db 경로 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "..", "sharpeyes.db")

    # 계산된 절대 경로를 통해 데이터베이스 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("--- [users] 테이블 전체 데이터 조회 ---")
    try:
        # users 테이블의 모든 데이터 가져오기
        cursor.execute("SELECT id, username, password FROM users")
        rows = cursor.fetchall()
        
        # 터미널에 줄 맞춰 출력하기 위한 가독성 헤더 설정
        header = f"{'ID':<5} | {'USERNAME':<15} | {'PASSWORD':<15}"
        print(header)
        print("-" * len(header))
        
        # 각 행(Row) 반복 출력 (비밀번호는 SHA-256 해시값으로 출력됩니다)
        for row in rows:
            print(f"{row[0]:<5} | {row[1]:<15} | {row[2]:<15}")
            
        if not rows:
            print("아직 가입된 회원이 없습니다.")
            
    except sqlite3.Error as e:
        print(f"데이터 조회 중 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    view_users()