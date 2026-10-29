import numpy as np
from pathlib import Path
import time

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    cv2 = None
    OPENCV_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False


class FaceTracker:
    def __init__(self):
        self.mp_face_detection = None
        self.face_cascade = None
        if OPENCV_AVAILABLE:
            if MEDIAPIPE_AVAILABLE and mp is not None:
                self.mp_face_detection = mp.solutions.face_detection
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5
                )
                print('[FaceTracker] MediaPipe FaceDetection inizializzato')
            else:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                print('[FaceTracker] OpenCV Haar Cascade inizializzato')
        else:
            print('[FaceTracker] OpenCV NON installato: face tracking disattivato')

    def get_crop_regions(self, video_path, target_width=1080, target_height=1920, sample_interval=15):
        if not OPENCV_AVAILABLE:
            target_aspect = target_width / target_height
            crop_x = max(0, int((1920 - 1920 * target_aspect) // 2))
            print('[FaceTracker] OpenCV non disponibile, uso crop centrato (fallback)')
            return [
                {'frame': 0, 'time': 0, 'crop_x': crop_x},
                {'frame': 1, 'time': 0.04, 'crop_x': crop_x}
            ], 1920, 1080, 30

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f'[FaceTracker] Impossibile aprire {video_path}, uso crop centrato')
            target_aspect = target_width / target_height
            crop_x = max(0, int((1080 - 1080 * target_aspect) // 2))
            return [
                {'frame': 0, 'time': 0, 'crop_x': crop_x},
                {'frame': 1, 'time': 0.04, 'crop_x': crop_x}
            ], 1920, 1080, 30

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'[FaceTracker] Video: {width}x{height}, {fps} fps, {total_frames} frame totali')

        regions = []
        frame_idx = 0
        start_time = time.time()
        max_frames_to_read = 300

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                crop_x = self._find_best_crop(frame, width, height, target_width, target_height)
                regions.append({
                    'frame': frame_idx,
                    'time': frame_idx / fps if fps > 0 else frame_idx / 30,
                    'crop_x': crop_x
                })

            frame_idx += 1
            if frame_idx > max_frames_to_read:
                print(f'[FaceTracker] Limite {max_frames_to_read} frame raggiunto, stop campionamento')
                break

        cap.release()
        elapsed = time.time() - start_time
        print(f'[FaceTracker] Analizzati {frame_idx} frame in {elapsed:.2f}s, {len(regions)} regioni crop')

        if not regions:
            regions.append({'frame': 0, 'time': 0, 'crop_x': max(0, (width - int(height * target_width / target_height)) // 2)})
        return regions, width, height, fps

    def _find_best_crop(self, frame, width, height, target_width, target_height):
        target_aspect = target_width / target_height
        crop_width = int(height * target_aspect)
        if crop_width > width:
            crop_width = width
        center_x = width // 2

        if MEDIAPIPE_AVAILABLE and self.mp_face_detection:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_face_detection.process(rgb_frame)
                if results and results.detections:
                    best_det = None
                    best_area = 0
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        area = bbox.width * bbox.height
                        if area > best_area:
                            best_area = area
                            best_det = det
                    if best_det:
                        bbox = best_det.location_data.relative_bounding_box
                        cx = int((bbox.xmin + bbox.width / 2) * width)
                        center_x = cx
            except Exception as e:
                print(f'[FaceTracker] Errore MediaPipe: {e}')
        elif self.face_cascade:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w_face, _ = largest
                    center_x = x + w_face // 2
            except Exception as e:
                print(f'[FaceTracker] Errore Haar Cascade: {e}')

        crop_x = center_x - crop_width // 2
        if crop_x < 0:
            crop_x = 0
        if crop_x + crop_width > width:
            crop_x = width - crop_width
        return crop_x

    def generate_smart_crop_filter(self, regions, duration, width, height):
        target_aspect = 1080 / 1920
        crop_w = int(height * target_aspect)
        if crop_w > width:
            crop_w = width

        if not OPENCV_AVAILABLE:
            crop_x = max(0, (width - crop_w) // 2)
            print(f'[FaceTracker] Fallback crop centrato (no OpenCV): x={crop_x}')
            return f"crop={crop_w}:{height}:{crop_x}:0,scale=1080:1920"

        if not regions:
            crop_x = max(0, (width - crop_w) // 2)
            print(f'[FaceTracker] Nessuna regione, crop centrato: x={crop_x}')
            return f"crop={crop_w}:{height}:{crop_x}:0,scale=1080:1920"

        # Clamp tutte le x al range valido [0, width - crop_w]
        valid_x = []
        for r in regions:
            cx = max(0, min(r['crop_x'], width - crop_w))
            valid_x.append(cx)

        if len(valid_x) >= 2:
            x1 = valid_x[0]
            x2 = valid_x[-1]
            if x1 == x2:
                # Nessun movimento: crop statico, senza espressione ridondante
                print(f'[FaceTracker] Smart crop STATICO: x={x1}')
                return f"crop={crop_w}:{height}:{x1}:0,scale=1080:1920"
            else:
                # Crop dinamico: x(t) = x1 + (x2-x1)*t/duration
                # FFmpeg accetta espressioni come 3° parametro posizionale di crop (senza "x=")
                expr = f"{x1}+({x2}-{x1})*t/{duration}"
                filter_str = f"crop={crop_w}:{height}:{expr}:0,scale=1080:1920"
                print(f'[FaceTracker] Smart crop DINAMICO: x1={x1} -> x2={x2} su {duration}s')
                return filter_str

        # Una sola regione: crop statico
        crop_x = valid_x[0]
        print(f'[FaceTracker] Smart crop STATICO: x={crop_x}')
        return f"crop={crop_w}:{height}:{crop_x}:0,scale=1080:1920"
