"""
File konfigurasi aplikasi
"""
import os
from pathlib import Path

# Direktori dasar (root proyek)
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"   # Mungkin tidak digunakan oleh Gradio

# Konfigurasi model AI
MODEL_CONFIG = {
    "FACE_ANALYSIS_MODEL": "buffalo_l",      # Model untuk deteksi dan analisis wajah
    "FACE_SWAP_MODEL": "inswapper_128.onnx", # Model untuk penukaran wajah
    "DETECTION_SIZE": (640, 640),            # Ukuran deteksi (lebar, tinggi)
    "CTX_ID": 0,                              # -1 untuk CPU, 0 untuk GPU pertama (jika ada)
    "CPU_THREADS": 4,                         # Jumlah thread untuk inferensi CPU (optimasi)
}

# Konfigurasi template gambar
TEMPLATE_CONFIG = {
    "TEMPLATES": {
        "1": {
            "id": "1",
            "name": "Template 1",
            "description": "Template default",
            "path": "./models/templates/step01.jpg"
        },
        "2": {
            "id": "2",
            "name": "Template 2",
            "description": "Template default",
            "path": "./models/templates/step02.jpg"
        },
        "3": {
            "id": "3",
            "name": "Template 3",
            "description": "Template default",
            "path": "./models/templates/step03.jpg"
        },
        "4": {
            "id": "4",
            "name": "Template 4",
            "description": "Template default",
            "path": "./models/templates/step04.jpg"
        },
        "5": {
            "id": "5",
            "name": "Template 5",
            "description": "Template default",
            "path": "./models/templates/step05.jpg"
        },
        "6": {
            "id": "6",
            "name": "Template 6",
            "description": "Template default",
            "path": "./models/templates/step06.jpg"
        }
    }
}

# Konfigurasi peningkatan kualitas (enhancement)
ENHANCE_CONFIG = {
    "ENABLE_ENHANCEMENT": True,           # Aktifkan peningkatan kualitas secara default
    "METHOD": "simple",                    # Metode: "simple" (cepat) atau "gfpgan" (jika tersedia)
    "GFPGAN_MODEL_PATH": None,              # Path ke model GFPGAN (jika digunakan)
}

def get_model_path(model_name: str) -> Path:
    """Mendapatkan path lengkap file model"""
    local_path = MODELS_DIR / model_name
    return local_path  # Jika tidak ada, nanti akan di-download

def get_template_path(template_id: str) -> Path:
    """Mendapatkan path lengkap file template berdasarkan ID"""
    template = TEMPLATE_CONFIG["TEMPLATES"].get(template_id)
    if not template:
        raise ValueError(f"Template dengan ID {template_id} tidak ditemukan")
    
    path_str = template["path"]
    if path_str.startswith("./"):
        return BASE_DIR / path_str[2:]
    return Path(path_str)

def get_all_template_paths() -> list:
    """Mengembalikan daftar path absolut semua template yang ada"""
    paths = []
    for tpl in TEMPLATE_CONFIG["TEMPLATES"].values():
        path_str = tpl["path"]
        if path_str.startswith("./"):
            full_path = BASE_DIR / path_str[2:]
        else:
            full_path = Path(path_str)
        if full_path.exists():
            paths.append(str(full_path))
    return paths

def ensure_directories():
    """Memastikan direktori yang diperlukan ada"""
    (MODELS_DIR / "templates").mkdir(parents=True, exist_ok=True)