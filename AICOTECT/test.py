import cv2

# 비디오 불러오기
cap = cv2.VideoCapture("gwu_fastapi_mysite-main/gwu_fastapi_mysite/data/r.mp4")

if cap.isOpened() == False:
    print("동영상을 열지 못했습니다.")
    exit(1)

# 반복횟수 O : for(게시글) / 반복횟수 X : while(키오스크)
while True:
    # ret(True, False), frame(영상 프레임)
    ret, frame = cap.read()
    if not ret:
        print("더 이상 가져올 프레임이 없어요")
        break

    cv2.imshow("video", frame)

    if cv2.waitKey(25) == ord("q"):
        print("사용자 입력에 의해 종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()