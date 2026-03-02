"""
FACESWAP KILAT LIVE EDITION
Versi 2.1 (Bright Edition)
----------------------------------------------------
FaceSwap Kilat LIVE adalah aplikasi realtime face swap
berbasis InsightFace yang dioptimalkan untuk CPU.

Fitur:
• Realtime webcam face swap
• Cache embedding wajah ke disk (anti-corrupt)
• Load model hanya sekali per session
• AVX2 / OpenVINO acceleration auto-detect
• CPU usage monitor realtime
• Preview deteksi wajah
• GUI sidebar profesional
• Tema cerah modern

Semua berjalan lokal tanpa cloud/API.

----------------------------------------------------
Coder : Deddy Ratnanto
GitHub : https://github.com/drat
----------------------------------------------------
"""

# =========================================================
# IMPORT
# =========================================================

import gradio as gr
import cv2
import numpy as np
import os
import pickle
import hashlib
import psutil
import warnings
import logging

from pathlib import Path

from core.face_processor import FaceProcessor
from core.config import ensure_directories


# =========================================================
# KONFIGURASI GLOBAL
# =========================================================

warnings.filterwarnings("ignore")
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "embedding_cache"
CACHE_DIR.mkdir(exist_ok=True)

cv2.setUseOptimized(True)
cv2.setNumThreads(os.cpu_count())


# =========================================================
# LOAD MODEL SEKALI
# =========================================================

print("\nMemuat model FaceSwap Kilat LIVE...")

FACE_PROCESSOR = FaceProcessor()

print("Model siap digunakan\n")


# =========================================================
# DETEKSI ACCELERATION
# =========================================================

def detect_acceleration():

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()

        if "OpenVINOExecutionProvider" in providers:
            return "OpenVINO aktif"

    except:
        pass

    return "CPU AVX2 aktif"


ACCEL_INFO = detect_acceleration()


# =========================================================
# CPU MONITOR
# =========================================================

def cpu_info():

    return f"""
🧠 CPU Usage : {psutil.cpu_percent()}%  
⚡ Acceleration : {ACCEL_INFO}
"""


# =========================================================
# EMBEDDING CACHE (ANTI CORRUPT)
# =========================================================

def image_hash(img):

    return hashlib.md5(img.tobytes()).hexdigest()


def save_embedding(hash_id, face):

    if face is None:
        return

    data = {
        "embedding": face.embedding,
        "normed_embedding": face.normed_embedding,
        "bbox": face.bbox
    }

    path = CACHE_DIR / f"{hash_id}.pkl"
    temp = path.with_suffix(".tmp")

    with open(temp, "wb") as f:
        pickle.dump(data, f)

    temp.replace(path)


def load_embedding(hash_id):

    path = CACHE_DIR / f"{hash_id}.pkl"

    if not path.exists():
        return None

    if path.stat().st_size == 0:
        path.unlink()
        return None

    try:

        with open(path, "rb") as f:
            return pickle.load(f)

    except:
        path.unlink()
        return None


def get_source_face_cached(img):

    h = image_hash(img)

    cached = load_embedding(h)

    if cached is not None:

        class CachedFace:
            pass

        face = CachedFace()

        face.embedding = cached["embedding"]
        face.normed_embedding = cached["normed_embedding"]
        face.bbox = cached["bbox"]

        return face


    faces = FACE_PROCESSOR.get_faces(img)

    if not faces:
        return None

    face = faces[0]

    save_embedding(h, face)

    return face


# =========================================================
# PREVIEW DETEKSI WAJAH
# =========================================================

def preview_faces(img):

    if img is None:
        return None

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    faces = FACE_PROCESSOR.get_faces(bgr)

    for f in faces:

        x1,y1,x2,y2 = map(int,f.bbox)

        cv2.rectangle(
            bgr,
            (x1,y1),
            (x2,y2),
            (0,200,255),
            3
        )

    return cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )


# =========================================================
# REALTIME WEBCAM SWAP
# =========================================================

SOURCE_FACE = None


def set_source(img):

    global SOURCE_FACE

    if img is None:
        return None

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    SOURCE_FACE = get_source_face_cached(bgr)

    print("Source embedding loaded:", SOURCE_FACE.normed_embedding.shape)

    if SOURCE_FACE is None:
        raise gr.Error("Tidak ada wajah terdeteksi")

    return preview_faces(img)


FRAME_COUNT = 0
CACHED_TARGET = None

def webcam_swap(frame):

    global SOURCE_FACE, FRAME_COUNT, CACHED_TARGET

    if SOURCE_FACE is None:
        return frame

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    FRAME_COUNT += 1

    if FRAME_COUNT % 3 == 0 or CACHED_TARGET is None:

        faces = FACE_PROCESSOR.get_faces(bgr)

        if faces:
            CACHED_TARGET = faces[0]

    if CACHED_TARGET is None:
        return frame

    result = FACE_PROCESSOR.swapper.get(
        bgr,
        CACHED_TARGET,
        SOURCE_FACE,
        paste_back=True
    )

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


# =========================================================
# CSS TEMA CERAH MODERN
# =========================================================

CSS = """

body {
background: linear-gradient(135deg,#e3f2fd,#f8fbff);
}

.header {
text-align:center;
padding:18px;
background:linear-gradient(90deg,#2196f3,#00e5ff);
color:white;
border-radius:12px;
margin-bottom:15px;
box-shadow:0 4px 20px rgba(0,0,0,0.1);
}

.sidebar {
background:white;
padding:15px;
border-radius:12px;
box-shadow:0 4px 12px rgba(0,0,0,0.1);
}

.footer {
text-align:center;
color:#444;
font-size:12px;
padding:12px;
}

"""


# =========================================================
# UI
# =========================================================

with gr.Blocks(
    css=CSS,
    title="FaceSwap Kilat LIVE Bright"
) as app:


    gr.HTML("""
    <div class="header">
    <h2>FaceSwap Kilat LIVE</h2>
    Realtime FaceSwap • Bright Edition
    </div>
    """)


    cpu_label = gr.Markdown(cpu_info())


    with gr.Row():

        with gr.Column(scale=1):

            gr.Markdown("### Kontrol")

            source = gr.Image(
                label="Pilih wajah sumber"
            )

            preview = gr.Image(
                label="Preview deteksi"
            )


        with gr.Column(scale=2):

            webcam = gr.Image(
                sources="webcam",
                streaming=True,
                label="Webcam"
            )

            output = gr.Image(
                label="Output"
            )


    source.change(
        set_source,
        source,
        preview
    )


    webcam.stream(
        webcam_swap,
        webcam,
        output
    )


    timer = gr.Timer(2)

    timer.tick(
        cpu_info,
        outputs=cpu_label
    )


    gr.HTML("""
    <div class="footer">
    FaceSwap Kilat LIVE | code by Deddy Ratnanto |
    https://github.com/drat<br>
    Gunakan secara etis dan bertanggung jawab.
    </div>
    """)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    ensure_directories()

    app.queue()

    app.launch(
        server_name="127.0.0.1",
        server_port=7861
    )