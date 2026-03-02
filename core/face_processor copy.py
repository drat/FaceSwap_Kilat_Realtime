"""
Modul inti pemrosesan wajah
"""
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import onnxruntime
from pathlib import Path
import urllib.request
import os

# Import konfigurasi
from core.config import (
    MODEL_CONFIG, MODELS_DIR, BASE_DIR, 
    ENHANCE_CONFIG, ensure_directories
)

class FaceProcessor:
    """Kelas pemroses wajah yang membungkus semua model AI"""
    
    def __init__(self):
        self.swapper = None
        self.face_analyzer = None
        self._initialize_models()

    def _ensure_model_downloaded(self, model_name: str, url: str):
        """
        Memastikan file model ada di direktori models.
        Jika tidak ada, unduh dari URL yang diberikan.
        """
        model_path = MODELS_DIR / model_name
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if model_path.exists():
            print(f"INFO:core.face_processor:Model lokal ditemukan: {model_path}, lewati unduhan.")
            return

        print(f"INFO:core.face_processor:Model tidak ditemukan, mengunduh dari:")
        print(f"   -> URL: {url}")
        print(f"   -> Tujuan: {model_path}")

        try:
            response = urllib.request.urlopen(url)
            total_length = response.getheader('content-length')
            
            with open(model_path, 'wb') as f:
                if total_length is None:
                    f.write(response.read())
                    print("INFO:core.face_processor:Unduhan selesai.")
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response:
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        print(f"\r   [ {'=' * done}{' ' * (50-done)} ] {dl * 100 / total_length:.2f}%", end='')
            print("\nINFO:core.face_processor:Unduhan selesai.")

        except Exception as e:
            print(f"\nERROR:core.face_processor:Gagal mengunduh model.")
            if model_path.exists():
                os.remove(model_path)
            print(f"   Periksa koneksi internet Anda, atau unduh manual dari:")
            print(f"   {url}")
            print(f"   dan letakkan di folder '{model_path.parent}'")
            raise RuntimeError(f"Gagal mengunduh model: {url}") from e

    def _initialize_models(self):
        """Inisialisasi semua model yang diperlukan dengan optimasi CPU"""
        try:
            print("INFO:core.face_processor:Memulai inisialisasi model AI...")
            # Redam log ONNX Runtime
            onnxruntime.set_default_logger_severity(3)
            
            # Pastikan direktori models ada
            ensure_directories()

            # Tentukan provider berdasarkan ketersediaan dan konfigurasi
            ctx_id = MODEL_CONFIG.get("CTX_ID", 0)
            if ctx_id >= 0:
                # Coba gunakan GPU jika tersedia
                providers = onnxruntime.get_available_providers()
                if 'CUDAExecutionProvider' in providers:
                    print("INFO:core.face_processor:Terdeteksi CUDA, akan menggunakan GPU.")
                else:
                    print("INFO:core.face_processor:CUDA tidak tersedia, beralih ke CPU.")
                    ctx_id = -1
                    providers = ['CPUExecutionProvider']
            else:
                # Paksa CPU
                print("INFO:core.face_processor:Konfigurasi memaksa penggunaan CPU.")
                providers = ['CPUExecutionProvider']

            # Optimasi untuk CPU: set jumlah thread
            if providers == ['CPUExecutionProvider'] or 'CPUExecutionProvider' in providers:
                # Insightface menggunakan ONNX dengan provider yang diberikan, tetapi tidak secara langsung menerima SessionOptions.
                # Namun kita bisa set environment variable untuk mengontrol thread.
                num_threads = MODEL_CONFIG.get("CPU_THREADS", 4)
                os.environ["OMP_NUM_THREADS"] = str(num_threads)
                os.environ["MKL_NUM_THREADS"] = str(num_threads)

            # --- Muat model analisis wajah (buffalo_l) ---
            model_name = MODEL_CONFIG["FACE_ANALYSIS_MODEL"]
            print(f"INFO:core.face_processor:Memeriksa model analisis wajah '{model_name}'...")
            model_path = MODELS_DIR / model_name
            if not model_path.is_dir() or not any(model_path.iterdir()):
                print(f"PERINGATAN:core.face_processor:Folder model '{model_name}' tidak ditemukan atau kosong.")
                print("   -> Program akan mencoba mengunduh dari internet. Jika ingin manual, letakkan folder hasil ekstraksi di 'models'.")
            else:
                print(f"INFO:core.face_processor:Folder model '{model_name}' ditemukan.")

            # Inisialisasi FaceAnalysis dengan root = BASE_DIR (agar mencari di folder models)
            self.face_analyzer = FaceAnalysis(
                name=model_name, 
                root=str(BASE_DIR), 
                providers=providers
            )
            det_size = MODEL_CONFIG.get("DETECTION_SIZE", (640, 640))
            # ctx_id disesuaikan: jika ctx_id asli >=0 dan GPU tidak tersedia, kita sudah set ctx_id=-1
            self.face_analyzer.prepare(ctx_id=ctx_id, det_size=det_size)

            # --- Muat model penukaran wajah (inswapper_128.onnx) ---
            swap_model_name = MODEL_CONFIG["FACE_SWAP_MODEL"]
            swap_model_url = "https://huggingface.co/xingren23/comfyflow-models/resolve/976de8449674de379b02c144d0b3cfa2b61482f2/insightface/inswapper_128.onnx?download=true"
            
            self._ensure_model_downloaded(swap_model_name, swap_model_url)

            swap_model_path = MODELS_DIR / swap_model_name
            print(f"INFO:core.face_processor:Memuat model dari: {swap_model_path}")
            
            self.swapper = insightface.model_zoo.get_model(
                str(swap_model_path),
                download=False,
                download_zip=False
            )
            print("INFO:core.face_processor:Semua model AI berhasil dimuat!")

        except Exception as e:
            print(f"ERROR:core.face_processor:Inisialisasi model gagal: {e}")
            raise RuntimeError(f"Tidak dapat menginisialisasi model AI: {e}") from e

    def get_faces(self, image):
        """Mendeteksi semua wajah dalam gambar"""
        try:
            faces = self.face_analyzer.get(image)
            if faces:
                print(f"INFO:core.face_processor:Terdeteksi {len(faces)} wajah")
            else:
                print("PERINGATAN:core.face_processor:Tidak ada wajah terdeteksi.")
            return faces
        except Exception as e:
            print(f"ERROR:core.face_processor:Gagal mendeteksi wajah: {e}")
            return []

    def enhance_face(self, image):
        """
        Meningkatkan kualitas gambar wajah (post-processing).
        image: citra dalam format BGR.
        Mengembalikan citra BGR yang sudah ditingkatkan kualitasnya.
        """
        if not ENHANCE_CONFIG.get("ENABLE_ENHANCEMENT", True):
            return image

        method = ENHANCE_CONFIG.get("METHOD", "simple")
        if method == "simple":
            # 1. Sharpening (penajaman)
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # 2. CLAHE (peningkatan kontras lokal) pada channel L
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced_lab = cv2.merge((l, a, b))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_bgr
        elif method == "gfpgan":
            # Di sini bisa ditambahkan integrasi GFPGAN jika tersedia
            # Untuk sementara, fallback ke simple
            print("PERINGATAN:core.face_processor:Metode GFPGAN belum diimplementasikan, menggunakan simple.")
            return self.enhance_face(image)  # recursive dengan method simple
        else:
            return image

    def swap_face(self, source_img, target_img, face_index=0, enhance_result=False):
        """
        Menukar wajah dari gambar sumber ke gambar target.
        
        Parameter:
        - source_img: gambar sumber (BGR) yang berisi wajah yang akan dipindahkan
        - target_img: gambar target (BGR) tempat wajah akan ditempel
        - face_index: indeks wajah pada sumber yang akan digunakan (0 = wajah pertama)
        - enhance_result: jika True, lakukan peningkatan kualitas pada hasil akhir
        
        Mengembalikan gambar hasil penukaran dalam format BGR.
        """
        try:
            # Deteksi wajah di sumber
            source_faces = self.get_faces(source_img)
            if not source_faces:
                raise ValueError("Tidak ditemukan wajah pada gambar sumber.")
            
            if face_index >= len(source_faces):
                raise ValueError(f"Indeks wajah {face_index} melebihi jumlah wajah terdeteksi ({len(source_faces)}).")

            # Deteksi wajah di target
            target_faces = self.get_faces(target_img)
            if not target_faces:
                raise ValueError("Tidak ditemukan wajah pada gambar target. Pastikan target memiliki setidaknya satu wajah.")

            source_face = source_faces[face_index]
            target_face = target_faces[0]  # Gunakan wajah pertama di target (bisa dikembangkan untuk pilih)

            # Lakukan penukaran
            result_img = self.swapper.get(target_img, target_face, source_face, paste_back=True)
            
            # Tingkatkan kualitas jika diminta
            if enhance_result:
                result_img = self.enhance_face(result_img)
                print("INFO:core.face_processor:Peningkatan kualitas selesai.")

            print("INFO:core.face_processor:Penukaran wajah berhasil.")
            return result_img

        except ValueError as ve:
            print(f"ERROR:core.face_processor:{ve}")
            raise
        except Exception as e:
            print(f"ERROR:core.face_processor:Proses penukaran gagal: {e}")
            raise RuntimeError("Terjadi kesalahan saat menukar wajah.") from e