import cv2
import numpy as np
from pathlib import Path

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class SmartCropper:
    """
    Smart crop 9:16 che segue il soggetto principale nel tempo.
    Ispirato alla pipeline di Clipper (campionamento frame + vision validator).
    """

    def __init__(self, target_w=1080, target_h=1920, num_samples=8):
        self.target_w = target_w
        self.target_h = target_h
        self.target_aspect = target_w / target_h
        self.num_samples = num_samples

        # Rilevatore volti (MediaPipe)
        self.face_detector = None
        if MEDIAPIPE_AVAILABLE and mp is not None:
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )

        # YOLO per rilevamento persone (opzionale, piu' robusto su corpi interi)
        self.yolo = None
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO('yolo11n.pt')
            except Exception:
                try:
                    self.yolo = YOLO('yolov8n.pt')
                except Exception:
                    self.yolo = None

    def _detect_cx(self, frame):
        """
        Ritorna il centro X del soggetto principale in valore normalizzato 0..1.
        """
        h, w = frame.shape[:2]

        if self.yolo is not None:
            try:
                results = self.yolo(frame, verbose=False)
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls == 0:  # classe 'person' in COCO
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cx = ((x1 + x2) / 2) / w
                            return cx
            except Exception:
                pass

        if self.face_detector is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.face_detector.process(rgb)
                if res.detections:
                    best = max(
                        res.detections,
                        key=lambda d: d.location_data.relative_bounding_box.width
                                      * d.location_data.relative_bounding_box.height
                    )
                    bbox = best.location_data.relative_bounding_box
                    cx = bbox.xmin + bbox.width / 2
                    return cx
            except Exception:
                pass

        return 0.5

    def sample_positions(self, video_path):
        """
        Campiona frame distribuiti uniformemente nel video e ritorna
        lista di (tempo_secondi, cx_normalizzato) + durata totale.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], 0.0

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'[SmartCrop] Video {width}x{height}, {fps} fps, durata {duration:.2f}s')

        positions = []
        if self.num_samples <= 1:
            times = [0.0]
        else:
            times = [duration * i / (self.num_samples - 1) for i in range(self.num_samples)]

        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            cx = self._detect_cx(frame)
            positions.append((t, cx))
            print(f'[SmartCrop] t={t:.2f}s cx={cx:.3f}')

        cap.release()
        return positions, duration

    def generate_filter(self, positions, duration, video_w, video_h):
        """
        Genera la stringa -vf per FFmpeg con crop dinamico a gradini.
        L'espressione x viene racchiusa tra virgolette singole per proteggere
        le virgole interne dall'essere interpretate come separatori di parametri.
        """
        target_w = self.target_w
        target_h = self.target_h

        # Larghezza del crop verticale per mantenere 9:16
        crop_w = int(video_h * self.target_aspect)
        if crop_w > video_w:
            crop_w = video_w
        crop_h = video_h

        max_x = video_w - crop_w
        if max_x < 0:
            max_x = 0

        # Nessun soggetto rilevato -> crop centrato statico
        if not positions:
            x = max(0, min(video_w // 2 - crop_w // 2, max_x))
            print(f'[SmartCrop] Nessun soggetto, crop statico x={x}')
            return f"crop={crop_w}:{crop_h}:{x}:0,scale={target_w}:{target_h}"

        # Ordina per tempo
        positions = sorted(positions, key=lambda p: p[0])

        # Converte cx normalizzato (0..1) in pixel x, clamped al range valido
        pixel_positions = []
        for t, cx_norm in positions:
            x_px = int(cx_norm * video_w - crop_w // 2)
            x_px = max(0, min(x_px, max_x))
            pixel_positions.append((t, x_px))

        # Un solo punto -> crop statico
        if len(pixel_positions) == 1:
            x = pixel_positions[0][1]
            print(f'[SmartCrop] Crop statico x={x}')
            return f"crop={crop_w}:{crop_h}:{x}:0,scale={target_w}:{target_h}"

        # Costruisci espressione FFmpeg a gradini:
        # if(lte(t,t1),x1,if(lte(t,t2),x2,...,xn))
        parts = []
        n = len(pixel_positions)
        for i, (t, x) in enumerate(pixel_positions):
            if i == n - 1:
                parts.append(str(x))
            else:
                parts.append(f"if(lte(t,{t:.3f}),{x},")
        expr = "".join(parts) + ")" * (n - 1)

        # RACCHIUDI l'espressione x tra virgolette singole: FFmpeg non
        # confondira' le virgole dentro l'espressione con separatori di filtro.
        filter_str = f"crop={crop_w}:{crop_h}:'{expr}':0,scale={target_w}:{target_h}"
        print(f'[SmartCrop] Filtro dinamico: {filter_str}')
        return filter_str
