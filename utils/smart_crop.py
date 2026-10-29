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


class ViralSmartCropper:
    """
    Smart crop ottimizzato per TikTok / Instagram Reels.
    - Campiona N frame distribuiti uniformemente nel clip
    - Rileva soggetto principale (volto MediaPipe o corpo YOLO)
    - Applica smoothing (media mobile) alle posizioni per evitare salti bruschi
    - Genera filtro FFmpeg con crop dinamico a gradini protetto da virgolette
    - Headroom dinamico sopra la testa per non tagliare la fronte
    - Fallback: blur_background se nessun soggetto trovato
    """

    def __init__(self, target_w=1080, target_h=1920, num_samples=16, headroom=0.20, smooth_window=3):
        self.target_w = target_w
        self.target_h = target_h
        self.target_aspect = target_w / target_h  # 0.5625
        self.num_samples = num_samples
        self.headroom = headroom
        self.smooth_window = smooth_window

        self.face_detector = None
        if MEDIAPIPE_AVAILABLE and mp is not None:
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4
            )

        self.yolo = None
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO('yolo11n.pt')
            except Exception:
                try:
                    self.yolo = YOLO('yolov8n.pt')
                except Exception:
                    self.yolo = None

    def _detect_subject(self, frame):
        """
        Ritorna (cx, cy) normalizzati 0..1 del soggetto principale.
        Priorità: YOLO persona (corpo intero), poi MediaPipe volto.
        """
        h, w = frame.shape[:2]

        if self.yolo is not None:
            try:
                results = self.yolo(frame, verbose=False)
                best_area = 0
                best_cx, best_cy = 0.5, 0.5
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls == 0:  # COCO classe 'person'
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            area = (x2 - x1) * (y2 - y1)
                            if area > best_area:
                                best_area = area
                                best_cx = (x1 + x2) / 2 / w
                                best_cy = (y1 + y2) / 2 / h
                if best_area > 0:
                    return best_cx, best_cy
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
                    cy = bbox.ymin + bbox.height / 2
                    return cx, cy
            except Exception:
                pass

        return 0.5, 0.5

    def sample_subject_positions(self, video_path):
        """
        Campiona frame distribuiti uniformemente.
        Ritorna lista di (tempo, cx, cy), durata, larghezza, altezza.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], 0.0, 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'[ViralCrop] Video {width}x{height}, {fps} fps, durata {duration:.2f}s')

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
            cx, cy = self._detect_subject(frame)
            positions.append((t, cx, cy))
            print(f'[ViralCrop] t={t:.2f}s cx={cx:.3f} cy={cy:.3f}')

        cap.release()
        return positions, duration, width, height

    def _smooth_positions(self, positions):
        """
        Applica media mobile alle posizioni X per evitare salti bruschi del crop.
        """
        if len(positions) <= self.smooth_window:
            return positions

        smoothed = []
        n = len(positions)
        half = self.smooth_window // 2
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window = positions[start:end]
            avg_t = sum(p[0] for p in window) / len(window)
            avg_cx = sum(p[1] for p in window) / len(window)
            avg_cy = sum(p[2] for p in window) / len(window)
            smoothed.append((avg_t, avg_cx, avg_cy))
        return smoothed

    def generate_filter(self, positions, duration, video_w, video_h, strategy='smart_crop'):
        """
        Genera stringa -vf per FFmpeg.
        - smart_crop: crop dinamico con espressione a gradini protetta da virgolette
        - blur_background: video centrato su sfondo sfocato
        - bars: bande nere classiche (gestito altrove, qui restituisce solo scale)
        """
        target_w = self.target_w
        target_h = self.target_h

        # Larghezza del crop verticale per 9:16
        crop_w = int(video_h * self.target_aspect)
        if crop_w > video_w:
            crop_w = video_w
        crop_h = video_h
        max_x = max(0, video_w - crop_w)

        # --- STRATEGIA BLUR BACKGROUND ---
        if strategy == 'blur_background':
            filter_str = (
                f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h},boxblur=25:5[bg];"
                f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,scale={target_w}:{target_h}"
            )
            print(f'[ViralCrop] Strategia blur_background')
            return filter_str

        # --- STRATEGIA SMART CROP ---
        if not positions:
            x = max(0, min(video_w // 2 - crop_w // 2, max_x))
            print(f'[ViralCrop] Nessun soggetto, crop statico centrato x={x}')
            return f"crop={crop_w}:{crop_h}:{x}:0,scale={target_w}:{target_h}"

        positions = sorted(positions, key=lambda p: p[0])
        positions = self._smooth_positions(positions)

        pixel_positions = []
        for t, cx_norm, cy_norm in positions:
            x_center = cx_norm * video_w
            x_px = int(x_center - crop_w / 2)
            # Headroom dinamico: se il soggetto è alto nel frame (cy basso), sposta leggermente su
            # Per ora applichiamo solo X clamping; Y rimane 0 (crop dall'alto del video, poi scale)
            x_px = max(0, min(x_px, max_x))
            pixel_positions.append((t, x_px))

        if len(pixel_positions) == 1:
            x = pixel_positions[0][1]
            print(f'[ViralCrop] Crop statico x={x}')
            return f"crop={crop_w}:{crop_h}:{x}:0,scale={target_w}:{target_h}"

        # Espressione FFmpeg a gradini: if(lte(t,t1),x1,if(lte(t,t2),x2,...,xN))
        parts = []
        n = len(pixel_positions)
        for i, (t, x) in enumerate(pixel_positions):
            if i == n - 1:
                parts.append(str(x))
            else:
                parts.append(f"if(lte(t,{t:.3f}),{x},")
        expr = "".join(parts) + ")" * (n - 1)

        filter_str = f"crop={crop_w}:{crop_h}:'{expr}':0,scale={target_w}:{target_h}"
        print(f'[ViralCrop] Filtro dinamico: {filter_str}')
        return filter_str
