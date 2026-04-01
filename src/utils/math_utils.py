import numpy as np

def calculate_angle(a, b, c):
    """
    세 점 a, b, c 사이의 각도를 계산합니다 (b가 중심점).
    a: [x, y] - 골반
    b: [x, y] - 무릎 (중심)
    c: [x, y] - 발목
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # 라디안 값 계산 (atan2 사용)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # 180도를 넘어가면 보정 (안쪽 각도를 구하기 위함)
    if angle > 180.0:
        angle = 360 - angle

    return angle