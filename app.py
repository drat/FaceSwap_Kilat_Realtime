"""
FACESWAP KILAT
Versi 1.2
----------------------------------------------------
FaceSwap Kilat adalah aplikasi penukar wajah berbasis
InsightFace yang dioptimalkan untuk CPU dan batch processing.

Fitur utama:
• Batch FaceSwap multi-image
• Progress bar realtime
• Preview deteksi wajah
• Download hasil otomatis (ZIP)
• Monitoring CPU realtime
• Dark theme professional UI

Aplikasi ini berjalan sepenuhnya lokal tanpa API eksternal.

----------------------------------------------------
Coder : Deddy Ratnanto
GitHub : https://github.com/drat
----------------------------------------------------
"""

# ============================================================
# IMPORT LIBRARY
# ============================================================

import gradio as gr
import cv2
import numpy as np
import os
import warnings
import logging
import psutil
import zipfile
import tempfile

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# core faceswap
from core.face_processor import FaceProcessor
from core.config import ensure_directories


# ============================================================
# KONFIGURASI GLOBAL
# ============================================================

# nonaktifkan warning agar console bersih
warnings.filterwarnings("ignore")
logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# optimasi OpenCV untuk CPU
cv2.setUseOptimized(True)
cv2.setNumThreads(os.cpu_count())

print("\n===================================")
print(" FACESWAP KILAT v1.2 STARTING")
print("===================================\n")

# inisialisasi processor AI
face_processor = FaceProcessor()

print("Model berhasil dimuat\n")


# ============================================================
# FUNGSI MONITOR CPU REALTIME
# ============================================================

def get_cpu_usage():
    """
    Mengambil persentase penggunaan CPU saat ini
    """
    usage = psutil.cpu_percent()
    return f"🧠 Penggunaan CPU: {usage}%"


# ============================================================
# PREVIEW DETEKSI WAJAH
# ============================================================

def preview_detected_faces(image):
    """
    Menampilkan bounding box wajah pada gambar sumber
    agar user dapat mengetahui jumlah wajah
    """

    if image is None:
        return None

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    faces = face_processor.get_faces(bgr)

    preview = bgr.copy()

    for face in faces:

        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    return preview


# ============================================================
# MEMBUAT FILE ZIP OTOMATIS
# ============================================================

def create_zip(images):
    """
    Menggabungkan semua hasil menjadi ZIP
    """

    if not images:
        return None

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".zip"
    )

    with zipfile.ZipFile(temp.name, "w") as zipf:

        for i, img in enumerate(images):

            filename = f"faceswap_{i+1}.png"

            cv2.imwrite(
                filename,
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )

            zipf.write(filename)

            os.remove(filename)

    return temp.name


# ============================================================
# PROSES FACESWAP BATCH
# ============================================================

def process_faceswap(
    source_image,
    target_files,
    face_index_str,
    enhance,
    progress=gr.Progress()
):
    """
    Fungsi utama faceswap batch
    """

    if source_image is None:
        raise gr.Error("Pilih gambar sumber")

    if not target_files:
        raise gr.Error("Pilih minimal satu gambar target")

    face_index = int(face_index_str)

    source_bgr = cv2.cvtColor(
        source_image,
        cv2.COLOR_RGB2BGR
    )

    total = len(target_files)

    results = []

    print(f"\nBatch dimulai ({total} file)\n")

    # worker parallel
    def worker(index, file_path):

        filename = Path(file_path).name

        print(
            f"Proses [{index}/{total}] : {filename}"
        )

        target = cv2.imread(file_path)

        result = face_processor.swap_face(
            source_bgr,
            target,
            face_index,
            enhance_result=enhance
        )

        if result is None:
            return None, filename

        rgb = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )

        return rgb, filename


    # parallel execution
    with ThreadPoolExecutor(
        max_workers=max(1, os.cpu_count()-1)
    ) as executor:

        futures = [

            executor.submit(
                worker,
                i+1,
                file
            )

            for i, file in enumerate(target_files)

        ]


        for i, future in enumerate(futures):

            img, name = future.result()

            progress(
                (i+1)/total,
                desc=f"⚙️ Memproses {name}"
            )

            if img is not None:
                results.append(img)

            yield (
                results,
                f"Memproses [{i+1}/{total}] {name}",
                create_zip(results)
            )


    print("\nBatch selesai\n")

    yield (
        results,
        "✅ Batch selesai",
        create_zip(results)
    )


# ============================================================
# RESET BATCH
# ============================================================

def reset_batch():

    print("Batch direset")

    return [], None, None, "Siap"


# ============================================================
# CSS DARK THEME PROFESSIONAL
# ============================================================

CSS = """

body {
    background: #0b0f17;
}

.header {
    text-align:center;
    padding:20px;
    background: linear-gradient(90deg,#141e30,#243b55);
    border-radius:12px;
    margin-bottom:15px;
}

.footer {
    text-align:center;
    color:#8b949e;
    font-size:12px;
    padding:15px;
}

"""


# ============================================================
# UI GRADIO
# ============================================================

with gr.Blocks(
    title="FaceSwap Kilat",
    css=CSS
) as app:


    # HEADER
    gr.HTML(
        """
        <div class="header">
        <h1>⚡ FaceSwap Kilat</h1>
        <p>Batch FaceSwap Profesional • CPU Optimized</p>
        </div>
        """
    )


    # CPU monitor
    cpu_label = gr.Markdown(
        get_cpu_usage()
    )


    with gr.Row():

        with gr.Column():

            source_image = gr.Image(
                label="👤 Gambar Sumber",
                type="numpy"
            )

            preview = gr.Image(
                label="🔍 Preview Deteksi Wajah"
            )

            face_index = gr.Textbox(
                value="0",
                label="Indeks wajah"
            )

            enhance = gr.Checkbox(
                value=True,
                label="Enhance kualitas"
            )


        with gr.Column():

            target_files = gr.File(
                file_count="multiple",
                label="🖼️ Gambar Target"
            )

            gallery = gr.Gallery(
                label="📁 Hasil"
            )

            zip_output = gr.File(
                label="⬇️ Download ZIP"
            )


    status = gr.Markdown("Siap")


    with gr.Row():

        run_button = gr.Button(
            "🚀 Mulai Proses",
            variant="primary"
        )

        reset_button = gr.Button(
            "Reset"
        )


    # EVENT PREVIEW
    source_image.change(
        preview_detected_faces,
        source_image,
        preview
    )


    # EVENT PROCESS
    run_button.click(
        process_faceswap,
        inputs=[
            source_image,
            target_files,
            face_index,
            enhance
        ],
        outputs=[
            gallery,
            status,
            zip_output
        ]
    )


    # EVENT RESET
    reset_button.click(
        reset_batch,
        outputs=[
            gallery,
            target_files,
            zip_output,
            status
        ]
    )


    # TIMER CPU
    cpu_timer = gr.Timer(2)

    cpu_timer.tick(
        get_cpu_usage,
        outputs=cpu_label
    )


    # FOOTER
    gr.HTML(
        """
        <div class="footer">
        FaceSwap Kilat | code by Deddy Ratnanto |
        https://github.com/drat
        <br><br>
        ⚠️ Gunakan alat ini secara bertanggung jawab.
        Jangan gunakan untuk penipuan, pelanggaran privasi,
        atau pelanggaran hak cipta.
        </div>
        """
    )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    ensure_directories()

    app.queue()

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )