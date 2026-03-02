import gradio as gr
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Import pemroses wajah dari core
from core.face_processor import FaceProcessor
from core.config import ensure_directories

# --- Inisialisasi model ---
face_processor = None
try:
    print("🚀 Sedang menginisialisasi model AI, harap tunggu...")
    face_processor = FaceProcessor()
    print("✅ Model AI berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    print("   Jika melihat error unduhan, lihat panduan di README.md untuk mengunduh manual.")
    face_processor = None

# --- Fungsi utama untuk Gradio (batch process) ---
def swap_face_gradio(source_image, target_files, face_index_str, enhance_quality):
    """
    Fungsi utama yang dipanggil dari antarmuka Gradio.
    - source_image: gambar sumber (numpy array RGB)
    - target_files: daftar path file gambar target (list of str) atau None
    - face_index_str: indeks wajah pada sumber (string)
    - enhance_quality: boolean apakah meningkatkan kualitas
    Mengembalikan daftar gambar hasil dalam format RGB (list of numpy arrays) untuk ditampilkan di galeri.
    """
    if face_processor is None:
        raise gr.Error("Model AI gagal dimuat. Tidak dapat memproses permintaan. Periksa log di belakang.")

    print(f"ℹ️ Memproses... Indeks wajah sumber: {face_index_str}, Tingkatkan kualitas: {enhance_quality}")

    try:
        # Validasi input
        if source_image is None:
            raise gr.Error("Harap unggah gambar sumber.")
        if not target_files or len(target_files) == 0:
            raise gr.Error("Harap unggah setidaknya satu gambar target.")

        # Konversi indeks wajah ke integer
        try:
            face_index = int(face_index_str)
            if face_index < 0:
                raise ValueError
        except (ValueError, TypeError):
            raise gr.Error("Indeks wajah harus berupa angka >= 0.")

        # Konversi gambar sumber RGB ke BGR (model menggunakan BGR)
        source_bgr = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)

        # Daftar untuk menyimpan hasil
        results_rgb = []

        # Proses setiap file target
        for file_path in target_files:
            try:
                # Baca gambar target dari file
                target_pil = Image.open(file_path).convert("RGB")
                target_np = np.array(target_pil)
                target_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)

                # Panggil fungsi swap dengan opsi enhancement
                result_bgr = face_processor.swap_face(
                    source_bgr,
                    target_bgr,
                    face_index,
                    enhance_result=enhance_quality
                )

                if result_bgr is None:
                    print(f"⚠️ Penukaran gagal untuk {file_path}, mungkin tidak ada wajah.")
                    # Tambahkan placeholder atau lewati? Untuk konsistensi, kita bisa tambahkan gambar hitam atau abaikan.
                    # Di sini kita lewati, tetapi perlu diingat bahwa jumlah hasil bisa kurang dari jumlah target.
                    continue

                # Konversi ke RGB untuk ditampilkan
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                results_rgb.append(result_rgb)

            except Exception as e:
                print(f"❌ Error memproses {file_path}: {e}")
                # Lewati file yang error
                continue

        if not results_rgb:
            raise gr.Error("Tidak ada gambar target yang berhasil diproses. Pastikan setiap gambar target memiliki wajah yang terdeteksi.")

        print(f"✅ Proses selesai! {len(results_rgb)} gambar berhasil.")
        return results_rgb

    except ValueError as ve:
        print(f"❌ Error: {ve}")
        raise gr.Error(str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error tak terduga: {e}")
        raise gr.Error(f"Terjadi kesalahan: {str(e)}")

# --- Bangun antarmuka Gradio ---
with gr.Blocks(
    title="Penukar Wajah by DAVID888",
    theme=gr.themes.Soft(),
    css=".gradio-container {max-width: 1200px !important}"
) as demo:
    
    gr.HTML("""
        <div style="text-align: center; font-family: 'Arial', sans-serif; color: #333;">
            <h1 style="color: #2c3e50;">Penukar Wajah Batch (Face Swap)</h1>
            <p style="font-size: 1.1em;">Unggah <b>satu gambar sumber</b> (wajah Anda) dan <b>satu atau banyak gambar target</b> (yang akan ditempeli).</p>
            <p style="font-size: 0.9em; color: #7f8c8d;">Jika gambar sumber memiliki banyak wajah, tentukan indeks wajah yang ingin digunakan (mulai dari 0).</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            source_image = gr.Image(
                label="Gambar Sumber (wajah Anda)",
                type="numpy",
                height=300
            )
            face_index_input = gr.Textbox(
                label="Indeks Wajah Sumber",
                value="0",
                placeholder="Jika banyak wajah, tulis indeks (0,1,2,...)",
                info="0 = wajah pertama, 1 = wajah kedua, dst."
            )
            # Checkbox untuk peningkatan kualitas
            enhance_checkbox = gr.Checkbox(
                label="Tingkatkan kualitas hasil",
                value=True,
                info="Aktifkan untuk memperhalus dan mempertajam wajah hasil (disarankan)"
            )

        with gr.Column(scale=1):
            target_files = gr.File(
                label="Gambar Target (bisa banyak, pilih beberapa file)",
                file_count="multiple",
                type="filepath",  # Mengembalikan daftar path file
                height=300
            )

    # Tombol proses
    swap_button = gr.Button(
        "🚀 Mulai Tukar Wajah Batch",
        variant="primary",
        size="lg"
    )

    # Tempat hasil (Gallery agar bisa zoom dan melihat banyak gambar)
    result_gallery = gr.Gallery(
        label="Hasil Penukaran",
        columns=3,          # Jumlah kolom dalam galeri
        height=500,
        object_fit="contain",
        preview=True        # Memungkinkan zoom saat diklik
    )

    # Contoh sederhana (opsional, bisa dihilangkan jika tidak ada contoh)
    # Karena template sudah dihapus, contoh mungkin perlu disesuaikan.
    # Kita bisa tetap menampilkan contoh jika ada file step01.jpg dan step02.jpg
    # Tapi karena target sekarang adalah file, contoh harus berupa daftar file.
    # Untuk kesederhanaan, kita tidak menampilkan contoh.

    # Hubungkan tombol dengan fungsi
    swap_button.click(
        fn=swap_face_gradio,
        inputs=[
            source_image,
            target_files,
            face_index_input,
            enhance_checkbox
        ],
        outputs=result_gallery,
        api_name="face_swap_batch"
    )

# --- Jalankan aplikasi dengan antrian (queue) untuk batch process ---
if __name__ == "__main__":
    # Pastikan direktori yang diperlukan ada
    ensure_directories()
    # Aktifkan queue untuk menangani banyak permintaan
    demo.queue()
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=True
    )