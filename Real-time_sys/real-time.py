from unittest import result
import cv2
import time
from ultralytics.models import YOLO
import easyocr
import numpy as np

#Config
reader = easyocr.Reader(["en"],gpu=True)
model = YOLO('../src/runs/detect/yolo11_vntraffic_merged_32classes_dataset-4/weights/best.pt')

SPEED_LIMIT_CLASS=24
VALID_SPEEDS= {"30", "40", "50", "60", "70", "80", "100", "120"}
vid = cv2.VideoCapture("../Real-time_sys/Testing-Video/Testing-video.mp4")

prev_time = 0
frame_count = 0
OCR_INTERVAL = 5
speed_cache = {}

PANEL_WIDTH = 200
SIGN_SIZE = 100
MAX_SIGN = 5

#Ve label
def draw_label(img, text, x1, y1, x2, y2, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if y1 - h - 8 < 0:
        text_y = y2 + h + 4
        bg_y1, bg_y2 = y2, y2 + h + 8
    else:
        text_y = y1 - 4
        bg_y1, bg_y2 = y1 - h - 8, y1

    cv2.rectangle(img, (x1, bg_y1), (x1 + w + 4, bg_y2), (0, 0, 0), -1)
    cv2.putText(img, text, (x1 + 2, text_y), font, font_scale, color, thickness)

#Doc so tren bien
def read_speed(crop):
    orc_results = reader.readtext(crop,detail=0)
    for text in orc_results:
        text = text.strip().replace(" ","")
        if text in VALID_SPEEDS:
            return text
    return None

#Hien thi side pannel
def side_pannel(frame,detected_signs):
    h = frame.shape[0]
    panel = np.zeros((h,PANEL_WIDTH,3),dtype=np.uint8)
    panel[:] = (40,40,40)

    cv2.putText(panel,"Bien bao",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    y_offset = 40

    for crop, label in detected_signs[:MAX_SIGN]:
        if crop.size == 0:
            continue

        sign_img = cv2.resize(crop,(SIGN_SIZE,SIGN_SIZE))
        x1 = (PANEL_WIDTH - SIGN_SIZE)//2
        panel[y_offset:y_offset+SIGN_SIZE,x1:x1+SIGN_SIZE] = sign_img

        cv2.putText(panel,label[:18],(5,y_offset+SIGN_SIZE+15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)
        y_offset += SIGN_SIZE+25
        if y_offset +SIGN_SIZE > h:
            break
    return panel



#Vong lap chinh
while vid.isOpened():
    success, frame = vid.read()
    if not success:
        print("Camera not found or video ended!")
        break
    detected_sign = []
    frame_count += 1
    run_orc = (frame_count%OCR_INTERVAL == 0)
    results = model.predict(source=frame,conf=0.65,stream=True,device=0,verbose=False)

    new_cache = {}
    for result in results:
        boxes = result.boxes
        for i,box in enumerate(boxes):
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls_id]

            if cls_id == SPEED_LIMIT_CLASS:
                color = (0,0,255)

                if run_orc:
                    crop = frame[y1:y2,x1:x2]
                    speed = read_speed(crop)
                    new_cache[i] = speed
                else:
                    speed = speed_cache.get(i)
                display_label = f"Max {speed} km/h" if speed else "Toc do (?)"
            else:
                color = (0,255,0)
                display_label = f"{label[:20]} {conf:.2f}"

            crop = frame[y1:y2,x1:x2].copy()
            detected_sign.append((crop,display_label))

            cv2.rectangle(frame, (x1,y1), (x2,y2),color,2)
            draw_label(frame,display_label,x1,y1,x2,y2,color)

    if run_orc:
        speed_cache = new_cache

    cur_time = time.time()
    fps = 1 / (cur_time-prev_time)
    prev_time = cur_time
    cv2.putText(frame,f"FPS: {int(fps)}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    panel = side_pannel(frame,detected_sign)
    display_frame = np.hstack([panel,frame])
    cv2.imshow("Traffic Sign Detection - Realtime",display_frame)
    time.sleep(0.0001)

    if cv2.waitKey(1) & 0xFF == ord("k"):
        break
vid.release()
cv2.destroyAllWindows()