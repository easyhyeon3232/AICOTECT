import os
import cv2
import dlib
import numpy as np
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
from starlette.responses import HTMLResponse
from fastapi.responses import RedirectResponse
import uvicorn
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
import time
from imutils import face_utils

from sqlalchemy.orm import Session
from db.session import get_db
from models.notice import NoticeORM
from schemas.notice import NoticeDTO
from typing import List
from sqlalchemy import desc
from fastapi.responses import HTMLResponse

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the data file
predictor_path = os.path.join(script_dir, './data/shape_predictor_68_face_landmarks.dat')
font_path = os.path.join(script_dir, './data/NanumSquare_acB.ttf')
model_path = os.path.join(script_dir, './models/2024_06_01_23_22_37.keras')

# Ensure the paths are correct
if not os.path.exists(predictor_path):
    raise Exception(f"Predictor file not found: {predictor_path}")
if not os.path.exists(font_path):
    raise Exception(f"Font file not found: {font_path}")
if not os.path.exists(model_path):
    raise Exception(f"Model file not found: {model_path}")

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load the Keras model
model = load_model(model_path)

# Set up the font
font = ImageFont.truetype(font_path, 40)

# FastAPI setup
app = FastAPI()
templates = Jinja2Templates(directory="templates/")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

IMG_SIZE = (34, 26)

def crop_eye(img, eye_points, gray):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect

def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("카메라를 열 수 없습니다.")

    eye_closed_start_time = None
    eye_closed_duration = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                shapes = predictor(gray, face)
                shapes = face_utils.shape_to_np(shapes)

                eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42], gray=gray)
                eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48], gray=gray)

                eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
                eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
                eye_img_r = cv2.flip(eye_img_r, flipCode=1)

                eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
                eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

                pred_l = model.predict(eye_input_l)
                pred_r = model.predict(eye_input_r)

                state_l = 'O %.1f' % pred_l if pred_l > 0.5 else '- %.1f' % pred_l
                state_r = 'O %.1f' % pred_r if pred_r > 0.5 else '- %.1f' % pred_r

                cv2.rectangle(frame, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
                cv2.rectangle(frame, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)

                cv2.putText(frame, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if pred_l < 0.5 and pred_r < 0.5:
                    if eye_closed_start_time is None:
                        eye_closed_start_time = time.time()
                    eye_closed_duration = time.time() - eye_closed_start_time
                    print(f"Eyes closed for {eye_closed_duration:.2f} seconds")
                else:
                    eye_closed_start_time = None
                    eye_closed_duration = 0

                if eye_closed_duration >= 2:
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((frame.shape[1] // 2 - 150, frame.shape[0] // 2), "졸음 운전의 위험이 있습니다.", font=font, fill=(0, 0, 255, 255))
                    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except GeneratorExit:
        cap.release()
        print("Generator was closed")
    except Exception as e:
        cap.release()
        print(f"An error occurred: {e}")
        raise e
    finally:
        cap.release()

@app.get("/camera")
async def camera():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/")
async def welcome(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})


@app.get("/noticeWrite.do")  # 매핑한 사이트
async def notice_write_form(request: Request) :
    return templates.TemplateResponse("notice_write.html",
                                      {"request": request})

@app.get("/notices", response_model=List[NoticeDTO])
async def get_notices(request: Request, db: Session = Depends(get_db)):
    notices = db.query(NoticeORM).order_by(desc(NoticeORM.dates)).all()
    return templates.TemplateResponse("notice.html", {"request": request, "noticeList": notices})

@app.get("/notice.do", response_class=HTMLResponse)  # 매핑한 사이트
async def get_notices(request: Request, db: Session = Depends(get_db)):
    try:
        notices = db.query(NoticeORM).all()
        return templates.TemplateResponse("notice.html", {"request": request, "noticeList": notices})
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/noticeDetail.do")  # 매핑한 사이트
async def welcome(request: Request) :
    return templates.TemplateResponse("notice_select.html",
                                      {"request": request})

@app.get("/home.do")
async def home(request: Request):
    return templates.TemplateResponse("kkg.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=2239)
