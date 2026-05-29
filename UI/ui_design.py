import sys, os
import threading

import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

from ultralytics import YOLO
import torch
import easyocr

from PyQt5.QtWidgets import (
    QApplication,QMainWindow,QWidget,QLabel,QPushButton,
    QSlider,QVBoxLayout,QHBoxLayout,QFrame,QScrollArea,
    QSizePolicy,QSpacerItem,QFileDialog
)
from PyQt5.QtCore import Qt, QThread,pyqtSignal,pyqtSlot,QTimer
from PyQt5.QtGui import QImage,QPixmap,QFont,QColor

#Path
_BASE = getattr(sys,"_MEIPASS",os.path.dirname(os.path.abspath(__file__)))
_OCR_DIR = os.path.join(_BASE,"easyocr_models")
_CAPTURE_DIR = os.path.join(os.path.expanduser("~"),"VN_Traffic_Captures")
os.makedirs(_CAPTURE_DIR,exist_ok=True)
#Config
MODEL_PATH = os.path.join(_BASE,"models","best.pt")
VIDEO_SOURCE = 0 # 0=webcam/videosource
# VIDEO_SOURCE    = "../Real-time_sys/Testing-Video/Testing-video.mp4"
CONF       = 0.75
CAM_ID     = 0
CAM_W,CAM_H= 1280, 720

#OCR config
SPEED_CLASS_ID  = 24
VALID_SPEEDS    = {"5","10","15","20","30","40","50","60","70","80","90","100","110","120"}
OCR_INTERVAL    = 5 #OCR only after OCR_INTERVAL frames

#Config color
"#000000"
C_BG        = "#0b0f18" #background
C_SURFACE   = "#131929" #surface
C_CARD      = "#1a2236" #log and dectected sign
C_BORDER    = "#243048" #border
C_ACCENT    = "#00d4aa"
C_WARN      = "#ffaa00" #warning
C_DANGER    = "#f85149" #danger
C_TEXT      = "#e2e8f7" #text
C_DIM       = "#ffffff" #small content
C_FPS       = "#64b5f6" #FPS
C_SPEED     = "#ff6b6b" #speed sign

CV_GREEN  = (0,212,170) #Color for normal sign, conf >= 0.65
CV_ORANGE = (0,170,255) #Detection's conf >= 0.5
CV_BLUE    = (68,68,255) #Speed Sign
CV_RED   = (60,60,220) #Detection's conf < 0.5

#OCR
def read_speed(reader,crop:np.ndarray) -> str | None:
    if crop is None or crop.size == 0:
        return None
    results = reader.readtext(crop,detail=0)
    for text in results:
        text = text.strip().replace(" ","")
        if text in VALID_SPEEDS:
            return text
    return None

def _cv_color(conf:float, is_speed: bool = False) -> tuple:
    if is_speed:
        return CV_RED
    if conf >= 0.65:
        return CV_GREEN
    if conf >= 0.5:
        return CV_ORANGE
    return CV_BLUE

def draw_detections(frame: np.ndarray,dets:list) -> np.ndarray:
    for d in dets:
        x1,y1,x2,y2 = d["bbox"]
        is_speed    = (d["class_id"] == SPEED_CLASS_ID)
        col         = _cv_color(d["conf"], is_speed)
        clen        = 18

        ov = frame.copy()
        cv2.rectangle(ov,(x1,y1),(x2,y2),col,-1)
        frame = cv2.addWeighted(ov,0.07,frame,0.93,0)

        for (cx,cy,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(cx,cy),(cx+dx*clen,cy),col,2,cv2.LINE_AA)
            cv2.line(frame,(cx,cy), (cx, cy+dy*clen), col, 2, cv2.LINE_AA)

        if is_speed:
            speed_val = d.get("ocr_text")
            if speed_val:
                display = f"Gioi han toc do {speed_val} km/h"
            else:
                display = "Toc do (?)"
        else:
            display = f"{d['class_name']} {d['conf']:.2f}"
        label = f"{display}"
        scale = 0.52
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,scale,1)
        lx1,ly1 = x1,y1-th-10
        if ly1<0:
            ly1 = y2
        cv2.rectangle(frame,(lx1,ly1),(lx1+tw+4,ly1+th+8),col,-1)
        cv2.putText(frame,label,(lx1+2,ly1+th+2),cv2.FONT_HERSHEY_SIMPLEX,scale,(10,15,25),1,cv2.LINE_AA)
    return frame

#Webcam Frame Grabber
class FrameGrabber(QThread):
    def __init__(self,source):
        super().__init__()
        self.source = source
        self._running = True
        self._frame = None
        self._lock = threading.Lock()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        while self._running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self._lock:
                    self._frame = frame
        cap.release()

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        self.wait()

#Inference Thead
"""
Webcam/Video + YOLO + EasyOCR // UI.
"""
class InferenceThread(QThread):
    frame_ready = pyqtSignal(np.ndarray,list,float)
    status_changed = pyqtSignal(str,str)

    def __init__(self,model_path:str, conf:float=CONF):
        super().__init__()
        self.model_path     =   model_path
        self.conf           =   conf
        self._running       =   False
        self._paused        =   False
        self.model          =   None
        self._ocr_reader    =   None
        self._device        =   "cpu"
        self._fps_buf       =   deque(maxlen=30)
        self._frame_count   =   0
        self._speed_cache   =   {}

    def run(self):
        self.status_changed.emit("Loading Model...","warn")
        try:
            self.model = YOLO(self.model_path)
            self._device = 0 if torch.cuda.is_available() else "cpu"
            dev_name = "GPU" if self._device == 0 else "CPU"
            self.status_changed.emit(f"Model loaded ({dev_name})","ok")
        except Exception as exc:
            self.status_changed.emit(f"Model Error: {exc}","error")
            return

        self.status_changed.emit("Loading EasyOCR...","warn")
        try:
            gpu_ocr = (self._device==0)
            self._ocr_reader = easyocr.Reader(["en"],gpu=gpu_ocr,model_storage_directory=_OCR_DIR)
            self.status_changed.emit("EasyOCR Loaded","ok")
        except Exception as exc:
            self.status_changed.emit(f"Error EasyOCR: {exc}","error")
            return

        grabber = None
        if isinstance(VIDEO_SOURCE,int):
            grabber = FrameGrabber(VIDEO_SOURCE)
            grabber.start()
            for _ in range (50):
                if grabber.get_frame() is not None:
                    break
                time.sleep(0.1)
            if grabber.get_frame() is None:
                self.status_changed.emit("Camera not found!","error")
                grabber.stop()
                return
            cap = None
        else:
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            if not cap.isOpened():
                self.status_changed.emit("Video not found!","error")
                return

        self._running = True
        self.status_changed.emit("Live","ok")

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()
            if grabber:
                frame = grabber.get_frame()
                if frame is None:
                    continue
            else:
                assert cap is not None
                ret,frame = cap.read()
                if not ret:
                    self.status_changed.emit("Video Ended!","warn")
                    continue

            self._frame_count += 1
            run_ocr = (self._frame_count % OCR_INTERVAL == 0)

            results = self.model.predict(
                source=frame, conf=self.conf,
                verbose=False, device=self._device
            )

            dets = []
            new_cache = {}

            for result in results:
                for i,box in enumerate(result.boxes):
                    cid         = int(box.cls[0])
                    cconf       = float(box.conf[0])
                    x1,y1,x2,y2 = map(int,box.xyxy[0])
                    name        = self.model.names.get(cid,f"Class{cid}")
                    crop        = frame[y1:y2,x1:x2].copy()

                    ocr_text = None
                    if cid == SPEED_CLASS_ID:
                        if run_ocr:
                            ocr_text=read_speed(self._ocr_reader,crop)
                            new_cache[i]=ocr_text
                        else:
                            ocr_text = self._speed_cache.get(i)
                    dets.append(dict(class_id=cid,class_name=name,conf=cconf,bbox=(x1,y1,x2,y2),ocr_text=ocr_text,crop=crop))

            if run_ocr:
                self._speed_cache=new_cache

            fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
            self._fps_buf.append(fps)
            avg_fps = sum(self._fps_buf) / len(self._fps_buf)

            self.frame_ready.emit(frame, dets,avg_fps)
        if grabber:
            grabber.stop()
        elif cap:
            cap.release()

    def stop(self):
        self._running=False
        self.wait()

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        return self._paused

    def set_conf(self,v:float):
        self.conf=v

#Widgets
class StatCard(QFrame):
    def __init__(self,title: str, unit: str = '',parent=None):
        super().__init__(parent)
        self.setObjectName("StatCrad")
        root = QVBoxLayout(self)
        root.setContentsMargins(12,8,12,8)
        root.setSpacing(2)
        lbl = QLabel(title.upper())
        lbl.setObjectName("StatLabel")
        row = QHBoxLayout()
        self._val_lbl=QLabel('-')
        self._val_lbl.setObjectName("StatValue")
        unit_lbl = QLabel(unit)
        unit_lbl.setObjectName("StatUnit")
        row.addWidget(self._val_lbl)
        row.addWidget(unit_lbl)
        row.addStretch()
        root.addWidget(lbl)
        root.addLayout(row)

    def set_value(self,v):
        self._val_lbl.setText(str(v))


class DetectionItem(QFrame):
    def __init__(self,cls_name:str, conf:float,ocr_text:str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("DetItem")
        row = QHBoxLayout(self)
        row.setContentsMargins(10,5,10,5)
        row.setSpacing(8)

        dot = QLabel(".")
        dot.setFixedWidth(12)
        col = C_ACCENT if conf >= 0.65 else (C_WARN if conf >= 0.5 else C_DANGER)
        dot.setStyleSheet(f"color: {col}; font-size: 7px;")

        if ocr_text:
            dis = f"Gioi han toc do {ocr_text} km/h"
            name_lbl = QLabel(dis)
            name_lbl.setObjectName("detNameSpeed")
        else:
            name_lbl = QLabel(cls_name)
            name_lbl.setObjectName("detName")

        conf_lbl = QLabel(f"{conf:.2f}")
        conf_lbl.setObjectName("detConf")

        row.addWidget(dot)
        row.addWidget(name_lbl,1)
        row.addWidget(conf_lbl)

class DetectionLog(QScrollArea):
    MAX_ITEMS = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DetLog")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._container = QWidget()
        self._layout    = QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(1)
        self._layout.addStretch()
        self.setWidget(self._container)

    def push(self, cls_name: str, conf: float, ocr_text: str = None):
        item = DetectionItem(cls_name, conf, ocr_text)
        self._layout.insertWidget(0, item)
        while self._layout.count() - 1 > self.MAX_ITEMS:
            layout_item = self._layout.takeAt(self._layout.count() - 2)
            if layout_item is not None:
                widget = layout_item.widget()
                if widget is not None:
                    widget.deleteLater()
    def clear_log(self):
        while self._layout.count() >1:
            layout_item = self._layout.takeAt(0)
            if layout_item is not None:
                widget = layout_item.widget()
                if widget is not None:
                    widget.deleteLater()

"""
Hien thi anh crop bien + label
"""
class SignThumnail(QFrame):
    THUMB_SIZE=72

    def __init__(self,crop:np.ndarray,label:str,is_speed:bool = False,parent=None):
        super().__init__(parent)
        self.setObjectName("SpeedThumb" if is_speed else "SignThumb")
        self.setFixedWidth(self.THUMB_SIZE+8)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4,4,4,4)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        img_lbl = QLabel()
        img_lbl.setFixedSize(self.THUMB_SIZE,self.THUMB_SIZE)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if crop is not None and crop.size >0:
            rgb = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
            h,w = rgb.shape[:2]
            qi = QImage(rgb.data,w,h,w*3,QImage.Format_RGB888)
            pix = QPixmap.fromImage(qi).scaled(self.THUMB_SIZE,self.THUMB_SIZE,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
            img_lbl.setPixmap(pix)
        else:
            img_lbl.setText("?")

        txt_lbl = QLabel(label[:16])
        txt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        txt_lbl.setWordWrap(True)
        txt_lbl.setObjectName("SpeedLabel" if is_speed else "ThumbLabel")

        layout.addWidget(img_lbl)
        layout.addWidget(txt_lbl)

"""Hien thi cac bien bao nhan dang duoc"""
class ThumbnailStrip(QScrollArea):
    MAX_THUMB = 8

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName("ThumbStrip")
        self.setFixedHeight(116)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)

        self._container = QWidget()
        self._layout = QHBoxLayout(self._container)
        self._layout.setContentsMargins(6,4,6,4)
        self._layout.setSpacing(6)
        self._layout.addStretch()
        self.setWidget(self._container)

    def update_signs(self,dets:list):
        while self._layout.count()>1:
            layout_item = self._layout.takeAt(0)
            if layout_item is not None:
                widget = layout_item.widget()
                if widget is not None:
                    widget.deleteLater()
        shown = 0
        for det in dets:
            if shown >= self.MAX_THUMB:
                break
            is_speed = (det["class_id"] == SPEED_CLASS_ID)
            if is_speed:
                speed = det.get("ocr_text")
                label = f"Gioi han toc do {speed} km/h" if speed else "Toc do (?)"
            else:
                label = det["class_name"]
            thumb = SignThumnail(det.get("crop"),label, is_speed)
            self._layout.insertWidget(self._layout.count()-1,thumb)
            shown += 1

class PulsingDot(QLabel):
    def __init__(self,parent=None):
        super().__init__(". Live",parent)
        self.setObjectName("LiveDot")
        self._on = True
        self._timer=QTimer(self)
        self._timer.timeout.connect(self._blink)
        self._timer.start(700)

    def _blink(self):
        self._on = not self._on
        col = C_ACCENT if self._on else C_DIM
        self.setStyleSheet(f"color:{col}; font-family:Consolas,monospace;"
                           f"font-size:12px;font-weight:700")

    def set_stopped(self):
        self._timer.stop()
        self.setText("!! STOPPED")

    def set_paused(self):
        self._timer.stop()
        self.setText("|| PAUSED")
        self.setStyleSheet(f"color:{C_WARN}l font-family:Consolas,monospace;"
                           f"font-size:12px; font-weight:700")

    def set_live(self):
        if not self._timer.isActive():
            self._timer.start(700)
        self.setText("|> LIVE")

#Main Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VN Traffic Sign Detection - YOLOv11 + OCR")
        self.setMinimumSize(1100,720)
        self.resize(1400,860)

        self._thread = None
        self._paused = False
        self._conf = CONF
        self._last_frame = None

        self._build_ui()
        self._apply_stylesheet()
        self._connect_signals()
        self._start_inference()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0,0,0,0)
        vbox.setSpacing(0)

        #header
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(52)
        hrow = QHBoxLayout(header)
        hrow.setContentsMargins(20,0,20,0)

        title = QLabel("VN TRAFFIC SIGN DETECTION")
        title.setObjectName("AppTitle")
        self.live_dot = PulsingDot()
        model_meta = QLabel(f"MODEL: {Path(MODEL_PATH).name.upper()}")
        model_meta.setObjectName("HeaderMeta")
        hw_meta = QLabel("GPU: RTX4050  |  OCR: EasyOCR")
        hw_meta.setObjectName("HeaderMeta")

        hrow.addWidget(title)
        hrow.addStretch()
        hrow.addWidget(model_meta)
        hrow.addSpacing(20)
        hrow.addWidget(hw_meta)
        hrow.addSpacing(20)
        hrow.addWidget(self.live_dot)

        #content row
        content = QWidget()
        crow = QHBoxLayout(content)
        crow.setContentsMargins(12,12,12,0)
        crow.setSpacing(12)

        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0,0,0,0)
        left_v.setSpacing(0)

        video_frame = QFrame()
        video_frame.setObjectName("VideoPanel")
        vf_layout = QVBoxLayout(video_frame)
        vf_layout.setContentsMargins(0,0,0,0)

        self.video_lbl = QLabel()
        self.video_lbl.setObjectName("VideoFeed")
        self.video_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_lbl.setMinimumSize(640,400)
        self._show_placeholder()

        vf_layout.addWidget(self.video_lbl)

        strip_header = QLabel("DETECTED SIGNS")
        strip_header.setObjectName("SectionTitle")

        self.thumb_strip = ThumbnailStrip()

        left_v.addWidget(video_frame,1)
        left_v.addWidget(strip_header)
        left_v.addWidget(self.thumb_strip)

        side = QWidget()
        side.setFixedWidth(270)
        side_v = QVBoxLayout(side)
        side_v.setContentsMargins(0,0,0,0)
        side_v.setSpacing(8)

        sec1 = QLabel("SYS STATUS")
        sec1.setObjectName("SectionTitle")

        stats_box = QFrame()
        stats_box.setObjectName("StatsBox")
        sb_layout = QVBoxLayout(stats_box)
        sb_layout.setContentsMargins(6,6,6,6)
        sb_layout.setSpacing(4)

        self.stat_fps = StatCard("FPS","fps")
        self.stat_det = StatCard("Detections","")
        self.stat_conf = StatCard("Avg Conf", "")
        self.stat_speed = StatCard("Speed Limit","km/h")

        sb_layout.addWidget(self.stat_fps)
        sb_layout.addWidget(self.stat_det)
        sb_layout.addWidget(self.stat_conf)
        sb_layout.addWidget(self.stat_speed)

        sec2 = QLabel("DETECTION LOG")
        sec2.setObjectName("SectionTitle")

        self.det_log = DetectionLog()

        #Control
        ctrl = QFrame()
        ctrl.setObjectName("ControlFrame")
        cv_layout = QVBoxLayout(ctrl)
        cv_layout.setContentsMargins(12,10,12,10)
        cv_layout.setSpacing(6)

        thresh_row = QHBoxLayout()
        thresh_lbl = QLabel("CONFIDENCE")
        thresh_lbl.setObjectName("ControlLabel")
        self.thread_val = QLabel(f"{self._conf:.2f}")
        self.thread_val.setObjectName("ControlValue")
        thresh_row.addWidget(thresh_lbl)
        thresh_row.addStretch()
        thresh_row.addWidget(self.thread_val)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(10,90)
        self.slider.setValue(int(CONF*100))
        self.slider.setObjectName("ConfSlider")

        ocr_info = QLabel(f"OCR interval: every {OCR_INTERVAL} frames")
        ocr_info.setObjectName("OCRInfo")

        btn_row = QHBoxLayout()
        self.btn_pause = QPushButton("|| Pause")
        self.btn_capture = QPushButton("|O| Capture")
        self.btn_pause.setObjectName("BtnSecondary")
        self.btn_capture.setObjectName("BtnPrimary")
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_capture)

        self.btn_open_video = QPushButton("|V| Open Video")
        self.btn_open_video.setObjectName("BtnSecondary")
        cv_layout.addWidget(self.btn_open_video)

        self.btn_clear = QPushButton("|X| CLEAR LOG")
        self.btn_clear.setObjectName("BtnGhost")

        cv_layout.addLayout(thresh_row)
        cv_layout.addWidget(self.slider)
        cv_layout.addWidget(ocr_info)
        cv_layout.addLayout(btn_row)
        cv_layout.addWidget(self.btn_clear)

        self.status_lbl = QLabel("Initialization...")
        self.status_lbl.setObjectName("StatusBar")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_lbl.setWordWrap(True)

        side_v.addWidget(sec1)
        side_v.addWidget(stats_box)
        side_v.addWidget(sec2)
        side_v.addWidget(self.det_log,1)
        side_v.addWidget(ctrl)
        side_v.addWidget(self.status_lbl)

        crow.addWidget(left,1)
        crow.addWidget(side)

        bottom_pad = QWidget()
        bottom_pad.setFixedHeight(10)

        vbox.addWidget(header)
        vbox.addWidget(content,1)
        vbox.addWidget(bottom_pad)

    #Signals
    def _connect_signals(self):
        self.slider.valueChanged.connect(self._on_conf_change)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_capture.clicked.connect(self._on_capture)
        self.btn_clear.clicked.connect(lambda: self.det_log.clear_log())
        self.btn_open_video.clicked.connect(self._on_open_video)

    #Inference
    def _start_inference(self):
        self._thread = InferenceThread(MODEL_PATH,self._conf)
        self._thread.frame_ready.connect(self._on_frame)
        self._thread.status_changed.connect(self._on_status)
        self._thread.start()

    #Slots
    @pyqtSlot(np.ndarray,list,float)
    def _on_frame(self,frame:np.ndarray,dets:list,fps:float):
        annotated = draw_detections(frame.copy(),dets)
        self._last_frame = annotated
        self._show_frame(annotated)

        #StatCards
        self.stat_fps.set_value(f"{fps:.1f}")
        self.stat_det.set_value(len(dets))

        if dets:
            avg_c = sum(det['conf'] for det in dets) / len(dets)
            self.stat_conf.set_value(f"{avg_c:.2f}")

        #Speed card
        speed_dets = [det for det in dets if det["class_id"] == SPEED_CLASS_ID]
        if speed_dets:
            speed_val = speed_dets[0].get("ocr_text")
            self.stat_speed.set_value(speed_val if speed_val else "?")
        else:
            self.stat_speed.set_value("-")

        #Detection Log
        for det in dets:
            self.det_log.push(det["class_name"],det["conf"],det.get("ocr_text"))

        #Thunbnail strip
        self.thumb_strip.update_signs(dets)

    @pyqtSlot(str,str)
    def _on_status(self,msg:str,level:str):
        cols = {"ok": C_ACCENT,"warm":C_WARN,"error":C_DANGER}
        col = cols.get(level,C_TEXT)
        self.status_lbl.setText(msg)
        self.status_lbl.setStyleSheet(f"color:{col};font-family:Consolas,monospace;font-size:12px;"
                                      f"background:{C_SURFACE};border:1px solid{C_BORDER};"
                                      f"border-radius:4px;padding:5px;")
        if level == "error":
            self.live_dot.set_stopped()

    def _on_conf_change(self,val:int):
        self._conf = val /100.0
        self.thread_val.setText(f"{self._conf:.2f}")
        if self._thread:
            self._thread.set_conf(self._conf)

    def _on_pause(self):
        if not self._thread:
            return
        self._paused = self._thread.toggle_pause()
        if self._paused:
            self.btn_pause.setText("|> Resume")
            self.live_dot.set_paused()
        else:
            self.btn_pause.setText("|| Pause")
            self.live_dot.set_live()

    def _on_capture(self):
        if self._last_frame is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_CAPTURE_DIR,f"capture_{ts}.jpg")
        cv2.imwrite(path,self._last_frame)
        self._on_status(f"Saved: {path}","ok")

    def _on_open_video(self):
        path,_ = QFileDialog.getOpenFileName(
            self,"Choose Video",
            os.path.expanduser("~"), "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        if not path:
            return

        if self._thread and self._thread.isRunning():
            self._thread.frame_ready.disconnect()
            self._thread.status_changed.disconnect()
            self._thread.stop()

        global VIDEO_SOURCE
        VIDEO_SOURCE = path
        self.det_log.clear_log()
        self._on_status(f"{Path(path).name}","ok")
        self._start_inference()

    #Helper
    def _show_frame(self,frame:np.ndarray):
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w = rgb.shape[:2]
        qi = QImage(rgb.data,w,h,w*3,QImage.Format_RGB888)
        lw,lh = self.video_lbl.width(),self.video_lbl.height()
        pix = QPixmap.fromImage(qi).scaled(
            lw,lh,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation
        )
        self.video_lbl.setPixmap(pix)

    def _show_placeholder(self):
        w,h = 960,540
        ph = np.zeros((h,w,3),dtype=np.uint8)
        ph[:] = (11,15,24)
        for x in range(0,w,60):
            cv2.line(ph,(x,0),(x,h),(18,36,58),1)
        for y in range(0,h,60):
            cv2.line(ph,(0,y),(w,y),(18,36,58),1)
        text = "LOADING MODEL + OCR ..."
        (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.85,2)
        cv2.putText(ph,text,((w-tw)//2,h//2),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,120,90),2,cv2.LINE_AA)
        sub = "YOLOv11 + EasyOCR - VN Traffic Sign Detection"
        (sw,_),_=cv2.getTextSize(sub,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
        cv2.putText(ph,sub,((w-sw)//2,h//2+38),cv2.FONT_HERSHEY_SIMPLEX,0.42,(50,90,120),1,cv2.LINE_AA)
        self._show_frame(ph)

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
        event.accept()

    #StyleSheet
    def _apply_stylesheet(self):
        self.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background: {C_BG};
            color: {C_TEXT};
            font-family: "Segoe UI", "SF Pro Display", sans-serif;
            font-size: 13px;
        }}
        QFrame#Header {{
            background: {C_SURFACE};
            border-bottom: 1px solid {C_BORDER};
        }}
        QLabel#AppTitle {{
            font-family: "Consolas", monospace;
            font-size: 14px; font-weight: 700;
            color: {C_ACCENT}; letter-spacing: 2px;
        }}
        QLabel#HeaderMeta {{
            font-family: "Consolas", monospace;
            font-size: 10px; color: {C_DIM}; letter-spacing: 1px;
        }}
        QFrame#VideoPanel {{
            background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 6px;
        }}
        QLabel#VideoFeed {{
            background: #070b12; border-radius: 4px;
        }}
        QLabel#SectionTitle {{
            font-family: "Consolas", monospace; font-size: 9px;
            font-weight: 700; color: {C_DIM}; letter-spacing: 2px; padding: 2px 0;
        }}
        QFrame#StatsBox {{
            background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 6px;
        }}
        QFrame#StatCard {{
            background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 4px;
        }}
        QLabel#StatLabel {{
            font-family: "Consolas", monospace; font-size: 9px;
            color: {C_DIM}; letter-spacing: 1.5px;
        }}
        QLabel#StatValue {{
            font-family: "Consolas", monospace; font-size: 22px;
            font-weight: 700; color: {C_FPS};
        }}
        QLabel#StatUnit {{
            font-family: "Consolas", monospace; font-size: 9px;
            color: {C_DIM}; padding-top: 8px;
        }}
        QScrollArea#DetLog {{
            background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 6px;
        }}
        QScrollArea#DetLog QWidget {{ background: {C_CARD}; }}
        QFrame#DetItem {{
            background: transparent; border-bottom: 1px solid {C_BORDER};
        }}
        QFrame#DetItem:hover {{ background: {C_SURFACE}; }}
        QLabel#detName    {{ font-size: 12px; color: {C_TEXT}; }}
        QLabel#detNameSpeed {{ font-size: 12px; color: {C_SPEED}; font-weight: 600; }}
        QLabel#detConf    {{ font-family: "Consolas", monospace; font-size: 11px; color: {C_ACCENT}; }}

        /* Thumbnail strip */
        QScrollArea#ThumbStrip {{
            background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 6px;
        }}
        QScrollArea#ThumbStrip QWidget {{ background: {C_CARD}; }}
        QFrame#SignThumb {{
            background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 4px;
        }}
        QFrame#SpeedThumb {{
            background: {C_SURFACE}; border: 2px solid {C_SPEED}; border-radius: 4px;
        }}
        QLabel#ThumbLabel {{ font-size: 9px; color: {C_DIM}; }}
        QLabel#SpeedLabel {{ font-size: 9px; color: {C_SPEED}; font-weight: 700; }}

        /* Controls */
        QFrame#ControlFrame {{
            background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 6px;
        }}
        QLabel#ControlLabel {{
            font-family: "Consolas", monospace; font-size: 9px;
            color: {C_DIM}; letter-spacing: 1.5px;
        }}
        QLabel#ControlValue {{
            font-family: "Consolas", monospace; font-size: 14px;
            font-weight: 700; color: {C_ACCENT};
        }}
        QLabel#OcrInfo {{
            font-family: "Consolas", monospace; font-size: 9px; color: {C_DIM};
            border-top: 1px solid {C_BORDER}; padding-top: 4px;
        }}
        QSlider#ConfSlider::groove:horizontal {{
            height: 4px; background: {C_BORDER}; border-radius: 2px;
        }}
        QSlider#ConfSlider::handle:horizontal {{
            width: 14px; height: 14px; background: {C_ACCENT};
            border-radius: 7px; margin: -5px 0;
        }}
        QSlider#ConfSlider::sub-page:horizontal {{
            background: {C_ACCENT}; border-radius: 2px;
        }}
        QPushButton#BtnPrimary {{
            background: {C_ACCENT}; color: #050c12; font-weight: 700;
            font-size: 11px; border: none; border-radius: 4px; padding: 7px 12px;
        }}
        QPushButton#BtnPrimary:hover   {{ background: #00f0c0; }}
        QPushButton#BtnPrimary:pressed {{ background: #009975; }}
        QPushButton#BtnSecondary {{
            background: transparent; color: {C_TEXT}; font-size: 11px;
            border: 1px solid {C_BORDER}; border-radius: 4px; padding: 7px 12px;
        }}
        QPushButton#BtnSecondary:hover {{ background: {C_SURFACE}; }}
        QPushButton#BtnGhost {{
            background: transparent; color: {C_DIM}; font-size: 11px;
            border: none; padding: 4px 12px;
        }}
        QPushButton#BtnGhost:hover {{ color: {C_DANGER}; }}
        QLabel#StatusBar {{
            font-family: "Consolas", monospace; font-size: 11px; color: {C_ACCENT};
            background: {C_SURFACE}; border: 1px solid {C_BORDER};
            border-radius: 4px; padding: 5px;
        }}
        QScrollBar:vertical {{ width: 5px; background: transparent; }}
        QScrollBar::handle:vertical {{
            background: {C_BORDER}; border-radius: 2px; min-height: 30px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar:horizontal {{ height: 5px; background: transparent; }}
        QScrollBar::handle:horizontal {{
            background: {C_BORDER}; border-radius: 2px; min-width: 30px;
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        """)
#EntryPoint
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VietNam Traffic Sign Detection")
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling,True)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()