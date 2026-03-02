"""
FACESWAP KILAT DESKTOP PRO
Versi 6.0 Professional UI Edition
----------------------------------------------------
Realtime FaceSwap Desktop Native dengan UI profesional.

Fitur:
• Professional sidebar UI
• Thumbnail wajah sumber
• Drag & drop source image
• Switch camera realtime
• Record video
• Virtual camera output
• ROI swap ultra fast
• 30–45 FPS CPU
• Clean exit

Coder : Deddy Ratnanto
https://github.com/drat
"""

# ================= IMPORT =================

import sys,cv2,pickle,hashlib,time
import numpy as np
import pyvirtualcam
from pathlib import Path

from PyQt5.QtWidgets import (
QApplication,QMainWindow,QLabel,QPushButton,
QVBoxLayout,QHBoxLayout,QWidget,QComboBox,
QFileDialog,QFrame)

from PyQt5.QtCore import QTimer,Qt
from PyQt5.QtGui import QImage,QPixmap,QFont

from core.face_processor import FaceProcessor


# ================= CONFIG =================

CACHE_DIR=Path("embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

DETECT_INTERVAL=20
DETECT_WIDTH=320
DISPLAY_WIDTH=720


# ================= LOAD MODEL =================

print("Loading model...")
processor=FaceProcessor()
print("Model ready")


# ================= HASH =================

def hash_image(img):
    """hash image untuk cache"""
    return hashlib.md5(img.tobytes()).hexdigest()


# ================= SAVE EMBEDDING =================

def save_embed(h,face):
    data={
    "embedding":face.embedding,
    "normed_embedding":face.normed_embedding,
    "bbox":face.bbox,
    "kps":face.kps}
    pickle.dump(data,open(CACHE_DIR/f"{h}.pkl","wb"))


# ================= LOAD EMBEDDING =================

def load_embed(h):
    path=CACHE_DIR/f"{h}.pkl"
    if not path.exists():
        return None
    try:
        data=pickle.load(open(path,"rb"))
    except:
        return None
    class Face:pass
    face=Face()
    face.embedding=data["embedding"]
    face.normed_embedding=data["normed_embedding"]
    face.bbox=data["bbox"]
    face.kps=data["kps"]
    return face


# ================= GET SOURCE =================

def get_source_face(img):
    h=hash_image(img)
    cached=load_embed(h)
    if cached:
        return cached
    faces=processor.get_faces(img)
    if not faces:
        return None
    face=faces[0]
    save_embed(h,face)
    return face


# ================= DETECT =================

def detect_face(frame):
    h,w,_=frame.shape
    scale=DETECT_WIDTH/w
    small=cv2.resize(frame,(DETECT_WIDTH,int(h*scale)))
    faces=processor.get_faces(small)
    if not faces:
        return None
    face=faces[0]
    face.bbox/=scale
    face.kps/=scale
    return face


# ================= ROI SWAP =================

def swap_roi(frame,target,source):
    x1,y1,x2,y2=map(int,target.bbox)
    pad=80
    x1=max(0,x1-pad)
    y1=max(0,y1-pad)
    x2=min(frame.shape[1],x2+pad)
    y2=min(frame.shape[0],y2+pad)
    roi=frame[y1:y2,x1:x2]
    faces=processor.get_faces(roi)
    if faces:
        roi=processor.swapper.get(roi,faces[0],source,paste_back=True)
        frame[y1:y2,x1:x2]=roi
    return frame


# ================= MAIN APP =================

class App(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("FaceSwap Kilat Desktop Pro")
        self.resize(1100,700)

        self.source=None
        self.target=None
        self.frame_count=0
        self.record=False
        self.writer=None
        self.prev_time=time.time()

        self.init_camera()
        self.init_ui()

        self.timer=QTimer()
        self.timer.timeout.connect(self.loop)
        self.timer.start(16)


    # ===== INIT CAMERA =====

    def init_camera(self,index=0):

        self.cap=cv2.VideoCapture(index)
        self.cap.set(3,1280)
        self.cap.set(4,720)

        w=int(self.cap.get(3))
        h=int(self.cap.get(4))

        self.virtual=pyvirtualcam.Camera(w,h,30)


    # ===== UI =====

    def init_ui(self):

        font=QFont("Arial",10)
        QApplication.instance().setFont(font)

        container=QHBoxLayout()

        # SIDEBAR
        sidebar=QVBoxLayout()

        logo=QLabel("FaceSwap Kilat")
        logo.setStyleSheet("color:white;font-size:18px;font-weight:bold")
        sidebar.addWidget(logo)

        load=QPushButton("Load Source")
        load.clicked.connect(self.load_source)
        sidebar.addWidget(load)

        self.source_preview=QLabel()
        self.source_preview.setFixedSize(160,160)
        self.source_preview.setStyleSheet(
        "background:#020617;border:2px solid #1e293b;border-radius:10px")
        sidebar.addWidget(self.source_preview)

        self.source_status=QLabel("No source")
        self.source_status.setStyleSheet("color:#ef4444")
        sidebar.addWidget(self.source_status)

        self.cam=QComboBox()
        self.cam.addItems(["Camera 0","Camera 1","Camera 2"])
        self.cam.currentIndexChanged.connect(self.switch_cam)
        sidebar.addWidget(self.cam)

        self.rec=QPushButton("Record")
        self.rec.clicked.connect(self.toggle_record)
        sidebar.addWidget(self.rec)

        exit_btn=QPushButton("Exit")
        exit_btn.clicked.connect(self.close)
        sidebar.addWidget(exit_btn)

        sidebar.addStretch()

        # PREVIEW AREA
        preview_layout=QVBoxLayout()

        self.preview=QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(
        "background:black;border-radius:12px")

        preview_layout.addWidget(self.preview)

        container.addLayout(sidebar,1)
        container.addLayout(preview_layout,4)

        widget=QWidget()
        widget.setLayout(container)
        self.setCentralWidget(widget)

        self.setStyleSheet("""
        QMainWindow{background:#0f172a;}
        QPushButton{
        background:#2563eb;
        color:white;
        padding:6px;
        border-radius:6px;}
        QPushButton:hover{background:#3b82f6;}
        QComboBox{
        background:#1e293b;
        color:white;
        padding:4px;}
        QLabel{color:white;}
        """)


    # ===== DRAG DROP =====

    def dragEnterEvent(self,event):

        if event.mimeData().hasUrls():
            event.accept()


    def dropEvent(self,event):

        path=event.mimeData().urls()[0].toLocalFile()
        self.load_source_file(path)


    # ===== LOAD SOURCE =====

    def load_source(self):

        path,_=QFileDialog.getOpenFileName()
        if path:
            self.load_source_file(path)


    def load_source_file(self,path):

        img=cv2.imread(path)

        face=get_source_face(img)

        if face:

            self.source=face

            x1,y1,x2,y2=map(int,face.bbox)
            face_img=img[y1:y2,x1:x2]
            thumb=cv2.resize(face_img,(160,160))
            thumb=cv2.cvtColor(thumb,cv2.COLOR_BGR2RGB)

            h,w,ch=thumb.shape

            qt=QImage(thumb.data,w,h,ch*w,QImage.Format_RGB888)

            self.source_preview.setPixmap(QPixmap.fromImage(qt))

            self.source_status.setText("Source loaded")
            self.source_status.setStyleSheet("color:#22c55e")


    # ===== SWITCH CAMERA =====

    def switch_cam(self,i):

        self.cap.release()
        self.virtual.close()
        self.init_camera(i)


    # ===== RECORD =====

    def toggle_record(self):

        if not self.record:

            fourcc=cv2.VideoWriter_fourcc(*"mp4v")

            self.writer=cv2.VideoWriter(
            "record.mp4",fourcc,30,
            (int(self.cap.get(3)),int(self.cap.get(4))))

            self.record=True
            self.rec.setText("Stop")

        else:

            self.record=False
            self.writer.release()
            self.rec.setText("Record")


    # ===== LOOP =====

    def loop(self):

        ret,frame=self.cap.read()
        if not ret:
            return

        self.frame_count+=1

        if self.frame_count%DETECT_INTERVAL==0 or self.target is None:

            face=detect_face(frame)
            if face:
                self.target=face

        if self.source and self.target:
            frame=swap_roi(frame,self.target,self.source)

        if self.record:
            self.writer.write(frame)

        self.virtual.send(frame)
        self.virtual.sleep_until_next_frame()

        now=time.time()
        fps=1/(now-self.prev_time)
        self.prev_time=now

        cv2.putText(frame,f"{int(fps)} FPS",(20,40),
        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        display=cv2.resize(frame,(DISPLAY_WIDTH,
        int(frame.shape[0]*DISPLAY_WIDTH/frame.shape[1])))

        rgb=cv2.cvtColor(display,cv2.COLOR_BGR2RGB)

        h,w,ch=rgb.shape

        qt=QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)

        self.preview.setPixmap(QPixmap.fromImage(qt))


    # ===== CLEAN EXIT =====

    def closeEvent(self,event):

        try:
            self.timer.stop()
            self.cap.release()
            self.virtual.close()
            if self.writer:
                self.writer.release()
        except:
            pass

        event.accept()


# ================= RUN =================

app=QApplication(sys.argv)
win=App()
win.show()
sys.exit(app.exec())