# ⚡ FaceSwap Kilat
<img width="1536" height="1024" alt="ChatGPT Image 3 Mar 2026, 03 08 45" src="https://github.com/user-attachments/assets/99c83600-a82f-435b-90e2-e65569dfdfae" />

# UBAH WAJAH DENGAN KECEPATAN ⚡ KILAT!
**FaceSwap Kilat** adalah toolkit FaceSwap profesional berbasis **InsightFace + ONNXRuntime** yang dioptimalkan untuk performa tinggi pada CPU. Proyek ini menyediakan dua mode aplikasi:
        
1. **FaceSwap Kilat Batch & GUI Web (Gradio)** — untuk pemrosesan batch dan penggunaan cepat melalui browser.
2. **FaceSwap Kilat Webcam GUI web (Gradio)** — aplikasi browser realtime dengan virtual camera dan performa rendah latency.
3. **FaceSwap Kilat Webcam Desktop (PyQt5)** — aplikasi desktop native realtime dengan virtual camera dan performa rendah latency.

**Tampilan Batch & GUI Web**
<img width="2772" height="3666" alt="FaceSwap-Kilat-03-03-2026_04_09_AM" src="https://github.com/user-attachments/assets/90f9c70b-ea40-4698-a743-eb3ee76290fc" />

**Tampilan Webcam GUI Web**

![demo_fskilat (1)](https://github.com/user-attachments/assets/2a1c0cc5-0910-496b-8970-6532734fc186)

**Tampilan Webcam Desktop**

![fskilat_desk (1)](https://github.com/user-attachments/assets/91fa11ff-8d9a-4fe8-96de-cebd2ed35b89)

Dirancang untuk berjalan sepenuhnya **lokal tanpa cloud**, dengan fokus pada:

* Performa tinggi pada CPU
* Latency rendah
* Arsitektur modular
* UI profesional
* Stabilitas produksi

---

# 🧠 Arsitektur

FaceSwap Kilat menggunakan pipeline berikut:

```
Camera / Image Input
        │
        ▼
InsightFace Detection (buffalo_l)
        │
        ▼
Embedding Extraction
        │
        ▼
InSwapper ONNX Model
        │
        ▼
ROI Composite
        │
        ▼
Display / Save / Virtual Camera
```

Optimasi utama:

* Embedding cache ke disk
* Region‑of‑Interest swap
* Adaptive detection interval
* CPU vectorization (AVX2)
* ONNXRuntime inference

---

# 📦 Struktur Proyek

```
faceswap-kilat/
│
├── app.py
├── app2.py
├── app_desktop.py
│
├── core/
│   ├── face_processor.py
│   └── config.py
│
├── models/
│   ├── buffalo_l/
│   └── inswapper_128.onnx
│
├── embedding_cache/
│
└── README.md
```

---

# 🌐 Aplikasi 1 — FaceSwap Kilat (Web / Batch)

## Fitur

* FaceSwap batch multi‑file
* UI modern berbasis Gradio
* Progress realtime
* Preview wajah
* Download hasil otomatis
* CPU optimized

## Jalankan

```
python app.py
```

atau

```
python app2.py
```

akses melalui:

```
http://127.0.0.1:7860
```

## Use Case

* Batch processing dataset
* Konten kreatif
* Eksperimen model

---

# 🖥 Aplikasi 2 — FaceSwap Kilat Desktop Pro

Aplikasi desktop realtime dengan performa tinggi.

## Fitur utama

* Realtime FaceSwap 30–45 FPS CPU
* GUI profesional
* Drag & Drop source image
* Thumbnail wajah sumber
* Switch kamera realtime
* Record video
* Virtual camera output (Zoom, OBS, Meet)
* Embedding cache
* ROI swap optimization

## Jalankan

```
python app_desktop.py
```

---

# 🎥 Virtual Camera

FaceSwap Kilat dapat digunakan sebagai kamera virtual di:

* Zoom
* OBS Studio
* Google Meet
* Microsoft Teams
* Discord

Pilih device:

```
pyvirtualcam
```

atau

```
OBS Virtual Camera
```

---

# ⚙️ Requirements

Python 3.10 direkomendasikan.

## Install dependencies

```
pip install numpy==1.26.4
pip install insightface==0.7.3
pip install onnxruntime
pip install opencv-contrib-python
pip install pyqt5
pip install pyvirtualcam
pip install gradio
```

---

# 🤖 Model

Menggunakan:

**InsightFace buffalo_l**

untuk:

* face detection
* landmark
* recognition

serta:

**inswapper_128.onnx**

untuk face swap.

---

# ⚡ Performa

Performa tipikal:

| Hardware | FPS   |
| -------- | ----- |
| Intel i7 | 25–35 |
| Apple M1 | 30–45 |
| Ryzen 7  | 30–45 |

---

# 🔒 Privasi

Semua proses berjalan **lokal**.

Tidak ada:

* upload
* cloud
* telemetry

---

# 🧩 Optimasi yang digunakan

* ONNXRuntime inference
* embedding cache
* ROI swap
* adaptive detection
* multithread safe pipeline

---

# 📸 Screenshot (opsional)

Tambahkan screenshot di sini.

---

# ⚠️ Disclaimer

Perangkat lunak ini disediakan untuk:

* riset
* edukasi
* penggunaan kreatif yang sah

Pengguna bertanggung jawab atas penggunaan sesuai hukum dan etika.

---

# 👨‍💻 Author

**Deddy Ratnanto**
Musician and Coder.

GitHub:

[https://github.com/drat](https://github.com/drat)

---

# 📄 License

Private / Custom License

Tentukan sesuai kebutuhan distribusi.

---

# 🚀 Roadmap

Planned:

* GPU acceleration
* macOS .app bundle
* Windows executable
* multi‑face swap
* plugin OBS

---

# ⭐ FaceSwap Kilat

Toolkit FaceSwap lokal profesional dengan fokus pada performa, stabilitas, dan kontrol penuh.
