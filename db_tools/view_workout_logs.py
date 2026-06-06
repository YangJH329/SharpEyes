import sqlite3
import os

def view_logs():
    # 스크립트 파일의 현재 위치를 기준으로 한 단계 상위 폴더(루트)에 있는 sharpeyes.db 경로 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "..", "sharpeyes.db")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("--- [workout_logs] 운동 기록 데이터 조회 (최신순) ---")
    try:
        # workout_logs 테이블과 users 테이블을 JOIN하여 user_id 대신 username을 가져옵니다.
        cursor.execute("""
            SELECT 
                w.id, 
                u.username, 
                w.exercise_mode, 
                w.total_count, 
                w.total_time, 
                w.timestamp
            FROM workout_logs w
            JOIN users u ON w.user_id = u.id
            ORDER BY w.timestamp DESC
        """)
        rows = cursor.fetchall()
        
        # 터미널에 줄 맞춰 출력하기 위한 가독성 헤더 설정
        header = f"{'LOG ID':<8} | {'USERNAME':<12} | {'MODE':<10} | {'COUNT':<6} | {'TIME':<8} | {'TIMESTAMP':<20}"
        print(header)
        print("-" * len(header))
        
        # 각 행(Row) 반복 출력
        for row in rows:
            print(f"{row[0]:<8} | {row[1]:<12} | {row[2].upper():<10} | {row[3]:<6} | {row[4]:<8} | {row[5]:<20}")
            
        if not rows:
            print("아직 저장된 운동 기록이 없습니다.")
            
    except sqlite3.Error as e:
        print(f"데이터 조회 중 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    view_logs()