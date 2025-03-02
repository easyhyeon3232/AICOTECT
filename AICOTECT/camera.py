import cv2

# 카메라 객체 생성
cap = cv2.VideoCapture(0)

# 카메라 연결 확인
if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    # 카메라에서 프레임 읽기
    ret, frame = cap.read()

    # 프레임 출력
    cv2.imshow('Camera', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# 저장할 비디오 파일 생성
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

while True:
    # 카메라에서 프레임 읽기
    ret, frame = cap.read()

    # 프레임 출력
    cv2.imshow('Camera', frame)

    # 프레임 저장
    out.write(frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
