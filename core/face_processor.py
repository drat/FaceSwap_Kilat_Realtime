"""
Face Processor - CPU Optimized Version
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import onnxruntime
import urllib.request
import os
import hashlib

from pathlib import Path

from core.config import (
    MODEL_CONFIG,
    MODELS_DIR,
    BASE_DIR,
    ENHANCE_CONFIG,
    ensure_directories
)


# ============================================
# GLOBAL CPU OPTIMIZATION
# ============================================

cv2.setUseOptimized(True)
cv2.setNumThreads(os.cpu_count())

onnxruntime.set_default_logger_severity(3)


class FaceProcessor:


    def __init__(self):

        self.swapper = None
        self.face_analyzer = None

        # cache source face
        self._source_hash = None
        self._source_face = None

        # optional target cache
        self._target_cache = {}

        self._initialize_models()


    # ============================================
    # HASH IMAGE (FAST)
    # ============================================

    def _hash_image(self, image):

        return hashlib.md5(
            image.tobytes()
        ).hexdigest()


    # ============================================
    # MODEL DOWNLOAD
    # ============================================

    def _ensure_model_downloaded(
        self,
        model_name,
        url
    ):

        path = MODELS_DIR / model_name

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        if path.exists():

            print(f"Model ditemukan: {path}")

            return


        print("Download model:")

        response = urllib.request.urlopen(url)

        total = int(
            response.getheader(
                'content-length',
                0
            )
        )

        downloaded = 0

        with open(path, "wb") as f:

            while True:

                chunk = response.read(8192)

                if not chunk:
                    break

                f.write(chunk)

                downloaded += len(chunk)

                percent = downloaded * 100 / total

                print(
                    f"\r{percent:.1f}%",
                    end=""
                )

        print("\nDownload selesai")


    # ============================================
    # INITIALIZE MODELS
    # ============================================

    def _initialize_models(self):

        print("Init InsightFace...")

        ensure_directories()

        cpu_threads = MODEL_CONFIG.get(
            "CPU_THREADS",
            os.cpu_count()
        )

        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)


        providers = ['CPUExecutionProvider']


        model_name = MODEL_CONFIG[
            "FACE_ANALYSIS_MODEL"
        ]


        self.face_analyzer = FaceAnalysis(

            name=model_name,
            root=str(BASE_DIR),
            providers=providers

        )


        self.face_analyzer.prepare(

            ctx_id=-1,
            det_size=MODEL_CONFIG.get(
                "DETECTION_SIZE",
                (640, 640)
            )

        )


        swap_model = MODEL_CONFIG[
            "FACE_SWAP_MODEL"
        ]


        url = (
            "https://huggingface.co/"
            "xingren23/comfyflow-models/"
            "resolve/main/"
            "insightface/inswapper_128.onnx"
        )


        self._ensure_model_downloaded(
            swap_model,
            url
        )


        model_path = MODELS_DIR / swap_model


        self.swapper = insightface.model_zoo.get_model(

            str(model_path),
            download=False,
            download_zip=False

        )


        print("Warmup model...")


        dummy = np.zeros(
            (640, 640, 3),
            dtype=np.uint8
        )

        self.face_analyzer.get(dummy)


        print("Model ready")


    # ============================================
    # GET FACES
    # ============================================

    def get_faces(
        self,
        image
    ):

        try:

            return self.face_analyzer.get(
                image
            )

        except:

            return []


    # ============================================
    # SOURCE CACHE
    # ============================================

    def get_source_face_cached(

        self,
        source_img,
        face_index

    ):

        h = self._hash_image(
            source_img
        )


        if h == self._source_hash:

            return self._source_face


        faces = self.get_faces(
            source_img
        )


        if not faces:

            raise ValueError(
                "Tidak ada wajah sumber"
            )


        if face_index >= len(faces):

            raise ValueError(
                "Face index invalid"
            )


        self._source_hash = h

        self._source_face = faces[
            face_index
        ]


        return self._source_face


    # ============================================
    # ENHANCEMENT (FACE ROI ONLY)
    # ============================================

    def enhance_face(
        self,
        image
    ):

        if not ENHANCE_CONFIG.get(
            "ENABLE_ENHANCEMENT",
            True
        ):

            return image


        kernel = np.array([

            [0,-1,0],
            [-1,5,-1],
            [0,-1,0]

        ])


        sharpened = cv2.filter2D(

            image,
            -1,
            kernel

        )


        lab = cv2.cvtColor(

            sharpened,
            cv2.COLOR_BGR2LAB

        )


        l,a,b = cv2.split(lab)


        clahe = cv2.createCLAHE(

            clipLimit=2.0,
            tileGridSize=(8,8)

        )


        l = clahe.apply(l)


        lab = cv2.merge((l,a,b))


        return cv2.cvtColor(

            lab,
            cv2.COLOR_LAB2BGR

        )


    # ============================================
    # SWAP FACE
    # ============================================

    def swap_face(

        self,
        source_img,
        target_img,
        face_index=0,
        enhance_result=False

    ):

        source_face = self.get_source_face_cached(

            source_img,
            face_index

        )


        target_faces = self.get_faces(

            target_img

        )


        if not target_faces:

            return None


        target_face = target_faces[0]


        result = self.swapper.get(

            target_img,
            target_face,
            source_face,
            paste_back=True

        )


        if enhance_result:

            x1,y1,x2,y2 = map(

                int,
                target_face.bbox

            )


            x1 = max(x1,0)
            y1 = max(y1,0)

            roi = result[
                y1:y2,
                x1:x2
            ]


            if roi.size > 0:

                enhanced = self.enhance_face(

                    roi

                )


                result[
                    y1:y2,
                    x1:x2
                ] = enhanced


        return result