import statistics

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
    Campiona il SOLO segmento da elaborare (start_time -> end_time)
    così il crop segue davvero il soggetto dentro la clip.

    Supporto multi-soggetto:
    - 1 soggetto  -> crop dinamico che lo segue (come prima)
    - 2 soggetti  -> schermo diviso in due riquadri impilati
    - 3 soggetti  -> due riquadri piccoli sopra + il soggetto principale grande sotto
    - 4 soggetti  -> griglia 2x2
    """

    MAX_SUBJECTS = 4
    # Sotto questa copertura verticale dell'unica inquadratura di gruppo
    # (frame residual fill <= 55%), e' meglio lo SPLIT: bandone' sfocate in
    # quarti del frame sono peggio dei riquadri impilati zoomati
    GROUP_COVER_MIN = 0.55

    def __init__(self, target_w=1080, target_h=1920, num_samples=10, headroom=0.20, smooth_window=3):
        self.target_w = target_w
        self.target_h = target_h
        self.target_aspect = target_w / target_h
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

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def _detect_all_subjects(self, frame):
        """
        Rileva TUTTI i soggetti nel frame.
        Ritorna lista di dict {cx, cy, width} normalizzati, ordinata da sinistra a destra.
        """
        h, w = frame.shape[:2]
        faces = []

        if self.yolo is not None:
            try:
                results = self.yolo(frame, verbose=False)
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:  # persona
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            faces.append({
                                'cx': (x1 + x2) / 2 / w,
                                'cy': (y1 + y2) / 2 / h,
                                'width': (x2 - x1) / w
                            })
                if faces:
                    return sorted(faces, key=lambda f: f['cx'])
            except Exception:
                pass

        if self.face_detector is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.face_detector.process(rgb)
                if res.detections:
                    for d in res.detections:
                        bb = d.location_data.relative_bounding_box
                        faces.append({
                            'cx': bb.xmin + bb.width / 2,
                            'cy': bb.ymin + bb.height / 2,
                            'width': bb.width
                        })
            except Exception:
                pass

        return sorted(faces, key=lambda f: f['cx'])

    def _detect_subject(self, frame):
        """Rileva il soggetto principale (retro-compatibile)."""
        faces = self._detect_all_subjects(frame)
        if not faces:
            return 0.5, 0.5
        best = max(faces, key=lambda f: f['width'])
        return best['cx'], best['cy']

    # ------------------------------------------------------------------ #
    # Campionamento singolo soggetto (Crop dinamico classico)
    # ------------------------------------------------------------------ #
    def sample_subject_positions(self, video_path, start_time=0.0, end_time=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], 0.0, 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if end_time is None or end_time <= start_time:
            end_time = start_time + 30

        clip_duration = end_time - start_time
        print(f'[ViralCrop] Campionamento segmento {start_time:.2f}s -> {end_time:.2f}s '
              f'(durata {clip_duration:.2f}s) su {width}x{height}')

        positions = []
        if self.num_samples <= 1:
            times = [0.0]
        else:
            times = [clip_duration * i / (self.num_samples - 1) for i in range(self.num_samples)]

        for t_rel in times:
            t_abs = start_time + t_rel
            cap.set(cv2.CAP_PROP_POS_MSEC, t_abs * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            cx, cy = self._detect_subject(frame)
            positions.append((t_rel, cx, cy))
            print(f'[ViralCrop] t_rel={t_rel:.2f}s cx={cx:.3f} cy={cy:.3f}')

        cap.release()
        return positions, clip_duration, width, height

    # ------------------------------------------------------------------ #
    # Campionamento MULTI-SOGGETTO (tracking Nearest-Neighbor simultaneo)
    # ------------------------------------------------------------------ #
    def _dedup_faces(self, faces):
        """
        Elimina le DOPPIE rilevazioni dello stesso volto: se due box hanno i
        centri a meno del 60% della media larghezze e'sono lo stesso volto,
        tengo il piu' grande. Senza questo un doppio rilevamento inventa un
        "secondo soggetto" e attiva uno split sbagliato.
        """
        result = []
        for f in sorted(faces, key=lambda x: -x['width']):
            if all(abs(f['cx'] - g['cx']) > 0.6 * (f['width'] + g['width']) / 2
                   for g in result):
                result.append(f)
        return result

    def _assign_tracks(self, rows, n, seeds=None):
        """
        Matching GLOBALO volti <-> soggetti: per ogni frame ogni volto va a
        UN solo soggetto (il piu' vicino alla sua ultima posizione nota).
        Prima il matching era sequenziale e con un solo volto nel frame le
        due tracce lo prendevano ENTRAMBE -> stessa persona nei due riquadri.
        Ritorna (tracks, widths, hits) dove hits[k] = campioni con volto
        reale assegnato al soggetto k.
        """
        tracks = {i: [] for i in range(n)}
        widths = {i: [] for i in range(n)}
        hits = [0] * n
        last_pos = [tuple(s) if s else (0.5, 0.5) for s in (seeds or [()] * n)]

        for t_rel, faces in rows:
            pairs = sorted(
                (abs(f['cx'] - last_pos[i][0]), i, j)
                for i in range(n) for j, f in enumerate(faces)
            )
            used_t, used_f = set(), set()
            for _, i, j in pairs:
                if i in used_t or j in used_f:
                    continue
                f = faces[j]
                tracks[i].append((t_rel, f['cx'], f['cy']))
                widths[i].append(f['width'])
                hits[i] += 1
                last_pos[i] = (f['cx'], f['cy'])
                used_t.add(i)
                used_f.add(j)
            for i in range(n):
                if i not in used_t:
                    # nessun volto libero per questo soggetto: tiene la posizione
                    tracks[i].append((t_rel, last_pos[i][0], last_pos[i][1]))
                    widths[i].append(0.0)
        return tracks, widths, hits

    def sample_subjects(self, video_path, start_time=0.0, end_time=None):
        """
        Campiona il segmento e tiene traccia di OGNI soggetto nel tempo.
        Ritorna (num_subjects, tracks, avg_widths, clip_duration, video_w, video_h)
        - tracks: {idx_soggetto: [(t_rel, cx_norm, cy_norm), ...]}
        - avg_widths: {idx_soggetto: larghezza_mediana_normalizzata}
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 1, {0: [(0.0, 0.5, 0.5)]}, {0: 0.2}, 0.0, 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if end_time is None or end_time <= start_time:
            end_time = start_time + 30

        clip_duration = end_time - start_time
        print(f'[ViralCrop] Campionamento multi-soggetto {start_time:.2f}s -> {end_time:.2f}s '
              f'(durata {clip_duration:.2f}s) su {width}x{height}')

        if self.num_samples <= 1:
            times = [0.0]
        else:
            times = [clip_duration * i / (self.num_samples - 1) for i in range(self.num_samples)]

        rows = []  # (t_rel, faces)
        for t_rel in times:
            t_abs = start_time + t_rel
            cap.set(cv2.CAP_PROP_POS_MSEC, t_abs * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            faces = self._dedup_faces(self._detect_all_subjects(frame))
            rows.append((t_rel, faces))
            print(f'[ViralCrop] t_rel={t_rel:.2f}s rilevati={len(faces)} '
                  + ' '.join(f"cx={f['cx']:.2f}" for f in faces))
        cap.release()

        counts = [len(f) for _, f in rows]
        if not rows or max(counts) == 0:
            print('[ViralCrop] Nessun soggetto rilevato: crop statico centrato, nessuno split')
            return 1, {0: [(0.0, 0.5, 0.5)]}, {0: 0.2}, clip_duration, width, height

        frames_any = [c for c in counts if c > 0]
        multi_frames = [c for c in counts if c >= 2]

        # -------------------------------------------------------------- #
        # MISURA DELLA DISTANZA TRA SOGGETTI                              #
        # Distanza tra volti adiacenti normalizzata in "larghezze di volto" #
        # (< 3.0 => attaccati, stessa inquadratura; >= 3.0 => staccati)    #
        # -------------------------------------------------------------- #
        ATTACH_THRESHOLD = 3.0

        group_track = []     # (t, cx_media, cy_media, span_norm) - caso attaccati
        multi_gap_norms = [] # distanze normalizzate minime per frame con 2+ soggetti
        all_widths = []
        for t_rel, faces in rows:
            if faces:
                cxm = sum(f['cx'] for f in faces) / len(faces)
                cym = sum(f['cy'] for f in faces) / len(faces)
                # estensione orizzontale del gruppo (bounding box dei volti):
                # serve per dimensionare la finestra che li contiene TUTTI
                left = min(f['cx'] - f['width'] / 2 for f in faces)
                right = max(f['cx'] + f['width'] / 2 for f in faces)
                span = max(0.05, right - left)
                group_track.append((t_rel, cxm, cym, span))
                all_widths.extend(f['width'] for f in faces if f['width'] > 0)
            if len(faces) >= 2:
                gaps = []
                for i in range(len(faces) - 1):
                    avg_w = (faces[i]['width'] + faces[i + 1]['width']) / 2
                    if avg_w > 0:
                        gaps.append((faces[i + 1]['cx'] - faces[i]['cx']) / avg_w)
                if gaps:
                    multi_gap_norms.append(min(gaps))

        # Trigger multi-soggetto: i frame con 2+ soggetti devono essere
        # almeno il 25% dei campioni in cui c'e' almeno un soggetto
        # (ignora apparizioni sporche singole)
        num_subjects = 1
        if frames_any and len(multi_frames) >= max(1, int(round(0.25 * len(frames_any)))):
            med_gap = statistics.median(multi_gap_norms) if multi_gap_norms else 999.0
            if med_gap < ATTACH_THRESHOLD:
                # Soggetti TROPPO ATTACCATI: normalmente nessuno split, un
                # UNICO crop dinamico che contiene il gruppo. MA se il gruppo
                # e' troppo largo per stare in 9:16 (l'unica inquadratura
                # coprirebbe meno del 55% del raccordo verticale, il resto
                # sarebbe sfocato), lo split impilato con zoom su ogni
                # soggetto rende decisamente meglio.
                num_subjects = 1
                merged_width = statistics.median(all_widths) if all_widths else 0.2
                span_px = max(p[3] for p in group_track) * width
                crop_w_guess = min(width, max(span_px * 1.2, 1))
                crop_h_guess = min(height, crop_w_guess * self.target_h / self.target_w)
                cov_h = min((crop_h_guess * self.target_w)
                            / (crop_w_guess * self.target_h), 1.0)
                if cov_h < self.GROUP_COVER_MIN:
                    num_subjects = max(2, min(self.MAX_SUBJECTS,
                                              int(statistics.mode(multi_frames))))
                    print(f'[ViralCrop] Soggetti ATTACCATI (distanza {med_gap:.1f}) '
                          f'ma gruppo TROPPO largo: un\'inquadratura coprirebbe '
                          f'solo il {cov_h * 100:.0f}% del verticale -> '
                          f'SPLIT impilato con {num_subjects} riquadri')
                else:
                    print(f'[ViralCrop] Soggetti ATTACCATI (distanza {med_gap:.1f} < '
                          f'{ATTACH_THRESHOLD} volti, copertura {cov_h * 100:.0f}%): '
                          f'nessuno split, un\'inquadratura unica che contiene il gruppo')
            else:
                # SOGGETTI STACCATI: split con N riquadri = soggetti contemporanei
                num_subjects = max(2, min(self.MAX_SUBJECTS, int(statistics.mode(multi_frames))))
                print(f'[ViralCrop] MULTI-SOGGETTO STACCATI (distanza {med_gap:.1f} >= {ATTACH_THRESHOLD} volti): '
                      f'{num_subjects} riquadri '
                      f'({len(multi_frames)}/{len(frames_any)} campioni con 2+ soggetti)')
        else:
            num_subjects = 1
            print(f'[ViralCrop] Un solo soggetto dominante '
                  f'({len(multi_frames)}/{len(frames_any)} campioni con 2+ soggetti, sotto soglia): nessuno split')

        # --- Tracking per soggetto: matching globale 1 volto = 1 soggetto ---
        if num_subjects == 1 and group_track:
            # Caso attaccati: una sola traccia = finestra del gruppo
            tracks = {0: group_track}
            widths = {0: all_widths or [0.2]}
        else:
            tracks, widths, hits = self._assign_tracks(rows, num_subjects)
            n_rows = len(rows)

            # Soggetti SPORADICI: se una traccia riceve un volto reale in
            # meno del 30% dei campioni, non e' un soggetto stabile della
            # scena -> lo scarto e riceco solo i soggetti stabili
            stable = [i for i in range(num_subjects)
                      if hits[i] >= max(2, 0.3 * n_rows)]
            if 0 < len(stable) < num_subjects:
                seeds = []
                for i in stable:
                    pts = tracks[i]
                    cxm = statistics.median([p[1] for p in pts])
                    cym = statistics.median([p[2] for p in pts])
                    seeds.append((cxm, cym))
                num_subjects = len(stable)
                tracks, widths, hits = self._assign_tracks(rows, num_subjects, seeds)
                print(f'[ViralCrop] Soggetti sporadici scartati, '
                      f'restano {num_subjects} soggetti stabili')

            # PROTEZIONE ANTI-DOPPIO: se due tracce sono PRATICAMENTE
            # sovrapposte (< 0.6 larghezze di volto) e' la STESSA persona
            # tracciata due volte -> un'unica inquadratura del gruppo.
            # Soglia bassa perche' con il matching globale 1 volto = 1
            # soggetto la duplicazione e' gia' impossibile: restiamo prudenziali
            if num_subjects >= 2:
                med_w = statistics.median(
                    [wd for ws in widths.values() for wd in ws if wd > 0]
                ) or 0.2
                overlaps = []
                for a in range(num_subjects):
                    for b in range(a + 1, num_subjects):
                        gaps = [
                            abs(tracks[a][r][1] - tracks[b][r][1]) / med_w
                            for r in range(n_rows)
                        ]
                        if gaps:
                            overlaps.append(statistics.median(gaps))
                if overlaps and statistics.median(overlaps) < 0.6:
                    print(f'[ViralCrop] PROTEZIONE: tracce quasi coincidenti '
                          f'(dist. {statistics.median(overlaps):.1f} < 0.6 volti) '
                          f'-> nessuno split, un\'inquadratura del gruppo')
                    num_subjects = 1
                    tracks = {0: group_track}
                    widths = {0: all_widths or [0.2]}

        avg_widths = {i: statistics.median([wd for wd in widths[i] if wd > 0]) if any(w > 0 for w in widths[i]) else 0.1
                      for i in range(num_subjects)}
        return num_subjects, tracks, avg_widths, clip_duration, width, height

    # ------------------------------------------------------------------ #
    # Smoothing
    # ------------------------------------------------------------------ #
    def _smooth_positions(self, positions):
        if len(positions) <= self.smooth_window:
            return positions
        smoothed = []
        n = len(positions)
        half = self.smooth_window // 2
        n_fields = min(len(p) for p in positions)
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window = positions[start:end]
            # media di ogni campo (t, cx, cy, [span, ...])
            avg = [sum(p[k] for p in window) / len(window) for k in range(n_fields)]
            smoothed.append(tuple(avg))
        return smoothed

    def _step_expr(self, values, fallback=0):
        """
        Converte lista ordinata [(t, val)] in espressione FFmpeg a gradini.
        Liste piatte (tutti uguali o singolo punto) -> costante.
        """
        if not values:
            return str(fallback)
        vals = [v for _, v in values]
        if len(vals) == 1 or all(abs(v - vals[0]) < 1e-6 for v in vals):
            return str(vals[0])
        parts = []
        n = len(values)
        for i, (t, v) in enumerate(values):
            if i == n - 1:
                parts.append(str(v))
            else:
                parts.append(f"if(lte(t,{t:.3f}),{v},")
        return "".join(parts) + ")" * (n - 1)

    def _window_expr(self, points, crop_w, crop_h, video_w, video_h, anchor_y=0.38):
        """
        Conversione track -> espressioni x(t) e y(t) finestra di crop zoomata.
        anchor_y: porzione della finestra sopra il centro del volto (headroom).
        """
        max_x = max(0, video_w - crop_w)
        max_y = max(0, video_h - crop_h)

        xs, ys = [], []
        for t, cx_norm, cy_norm in points:
            x = int(cx_norm * video_w - crop_w / 2)
            y = int(cy_norm * video_h - anchor_y * crop_h)
            xs.append((t, max(0, min(x, max_x))))
            ys.append((t, max(0, min(y, max_y))))
        xs.sort(key=lambda p: p[0])
        ys.sort(key=lambda p: p[0])

        return self._step_expr(xs, fallback=max_x // 2), self._step_expr(ys, fallback=max_y // 2)

    def _stepped_expr(self, points, crop_w, video_w):
        """
        Converte track [(t_rel, cx_norm, cy_norm)] in espressione FFmpeg x(t)
        a gradini (o x statica se un solo punto). Full-height: y=0.
        """
        max_x = max(0, video_w - crop_w)
        pix = []
        for t, cx_norm, _cy in points:
            x_center = cx_norm * video_w
            x_px = int(x_center - crop_w / 2)
            x_px = max(0, min(x_px, max_x))
            pix.append((t, x_px))
        pix.sort(key=lambda p: p[0])
        return self._step_expr(pix, fallback=max_x // 2)

    # ------------------------------------------------------------------ #
    # Filtro crop dinamico SINGOLO soggetto (come prima)
    # ------------------------------------------------------------------ #
    def generate_filter(self, positions, duration, video_w, video_h, strategy='smart_crop'):
        target_w = self.target_w
        target_h = self.target_h

        # Caso "soggetti ATTACCATI": la traccia contiene l'estensione del
        # gruppo (t, cx, cy, span) -> finestra unica che li contiene tutti
        if positions and any(len(p) >= 4 for p in positions):
            return self._generate_group_filter(positions, video_w, video_h)

        crop_w = int(video_h * self.target_aspect)
        if crop_w > video_w:
            crop_w = video_w
        crop_h = video_h
        max_x = max(0, video_w - crop_w)

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

    def _generate_group_filter(self, positions, video_w, video_h):
        """
        Caso soggetti ATTACCATI: UN'unica inquadratura che contiene TUTTI i
        soggetti. La finestra di crop viene dimensionata sul bounding box
        del gruppo (span massima + margini), segue il gruppo in x e y, e
        l'output e' riempito con sfondo sfocato quando l'aspect della
        finestra non puo' essere 9:16 (gruppi larghi non ci stanno in un
        frame verticale): cosi' NON si taglia mai ameta' un soggetto.
        """
        target_w, target_h = self.target_w, self.target_h
        margin_factor = 0.2  # span * 1.20: aria moderata ai lati del gruppo

        positions = sorted(positions, key=lambda p: p[0])
        positions = self._smooth_positions(positions)

        span_px = max(p[3] for p in positions) * video_w
        min_w = min(video_w, int(video_h * self.target_aspect))
        crop_w = int(min(video_w, max(span_px * (1 + margin_factor), min_w)))
        crop_h = min(video_h, int(round(crop_w * target_h / target_w)))
        max_x = max(0, video_w - crop_w)
        max_y = max(0, video_h - crop_h)

        xs, ys = [], []
        for t, cx, cy, _span in positions:
            x = max(0, min(int(cx * video_w - crop_w / 2), max_x))
            y = max(0, min(int(cy * video_h - 0.38 * crop_h), max_y))
            xs.append((t, x))
            ys.append((t, y))
        x_expr = self._step_expr(xs, fallback=max_x // 2)
        y_expr = self._step_expr(ys, fallback=max_y // 2)

        print(f'[ViralCrop] Inquadratura di GRUPPO: finestra {crop_w}x{crop_h} '
              f'su {video_w}x{video_h} (segue il gruppo, sfondo sfocato dove '
              f'la finestra non copre tutto il 9:16)')

        filter_str = (
            f"[0:v]crop={crop_w}:{crop_h}:'{x_expr}':'{y_expr}'[gwin];"
            f"[gwin]split=2[gblur][gfg];"
            f"[gblur]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h},boxblur=25:5[bg];"
            f"[gfg]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,scale={target_w}:{target_h}"
        )
        print(f'[ViralCrop] Filtro gruppo: {filter_str[:180]}...')
        return filter_str

    # ------------------------------------------------------------------ #
    # Filtro Split Screen multi-soggetto (grafo filter_complex)
    # ------------------------------------------------------------------ #
    def generate_split_filter(self, num_subjects, tracks, avg_widths, video_w, video_h, ass_path=None):
        """
        Costruisce un grafo -filter_complex che divide lo schermo in riquadri,
        uno per soggetto, ciascuno con crop dinamico ZOOMATO che segue il
        soggetto in orizzontale E in verticale. La finestra di crop copre solo
        ~75% dell'altezza del source, quindi ogni riquadro mostra UN soggetto
        (prima la finestra era larga quanto quasi tutto il frame e le tile
        ripetevano gli stessi soggetti).

        Layout:
        - 2 soggetti: due riquadri grandi (1080x960) impilati verticalmente
        - 3 soggetti: due piccoli (540x960) sopra + il principale grande (1080x960) sotto
        - 4 soggetti: griglia 2x2 di riquadri 540x960
        """
        if num_subjects <= 1:
            labels = list(tracks.keys())
            positions = tracks[labels[0]]
            return None  # caller usa generate_filter

        # Ordina per dimensione: il soggetto più grande/principale per ultima (grande / riga bassa)
        order = sorted(range(num_subjects),
                       key=lambda i: -avg_widths.get(i, 0.1))
        print(f'[ViralSplit] layout={num_subjects} soggetti, ordine per dimensione: {order}')

        # Finestre di crop ZOOMATE (75% dell'altezza sorgente),
        # stesso aspect del riquadro di destinazione
        zoom_h = min(video_h - 2, int(round(video_h * 0.75)))
        big_crop_w = min(video_w, int(round(zoom_h * 1080 / 960)))   # ~911 su 1080p
        small_crop_w = min(video_w, int(round(zoom_h * 540 / 960)))  # ~456 su 1080p

        def _chain(subj, crop_w, scale_w, scale_h, label):
            pts = self._smooth_positions(tracks[subj])
            x_expr, y_expr = self._window_expr(pts, crop_w, zoom_h, video_w, video_h)
            return (
                f"[0:v]crop={crop_w}:{zoom_h}:'{x_expr}':'{y_expr}',"
                f"scale={scale_w}:{scale_h}[{label}];"
            )

        chains = []

        if num_subjects == 2:
            # due riquadri grandi impilati: il più grande sotto
            subj_top, subj_bottom = order[1], order[0]
            chains.append(_chain(subj_top, big_crop_w, 1080, 960, 't0'))
            chains.append(_chain(subj_bottom, big_crop_w, 1080, 960, 't1'))
            compose = "[t0][t1]vstack"
            print(f'[ViralSplit] 2 soggetti: riquadro grande sopra {subj_top}, '
                  f'riquadro grande (principale) sotto {subj_bottom}')
        elif num_subjects == 3:
            # il principale grande in basso, gli altri due piccoli sopra (ordine sx->dx)
            subj_main = order[0]
            others = sorted(order[1:], key=lambda i: tracks[i][0][1])
            chains.append(_chain(others[0], small_crop_w, 540, 960, f'tk{others[0]}'))
            chains.append(_chain(others[1], small_crop_w, 540, 960, f'tk{others[1]}'))
            chains.append(_chain(subj_main, big_crop_w, 1080, 960, 'tm'))
            compose = f"[tk{others[0]}][tk{others[1]}]hstack[top];[top][tm]vstack"
            print(f'[ViralSplit] 3 soggetti: piccoli sopra {others}, principale grande sotto {subj_main}')
        else:
            # 4 (o più, troncati): griglia 2x2; riga bassa con i 2 principali
            chosen = order[:4]
            # riga bassa: i 2 principali; dentro ogni riga ordina da sx a dx
            top = sorted([chosen[2], chosen[3]], key=lambda i: tracks[i][0][1])
            bottom = sorted([chosen[0], chosen[1]], key=lambda i: tracks[i][0][1])
            for subj in chosen:
                chains.append(_chain(subj, small_crop_w, 540, 960, f'ts{subj}'))
            compose = (f"[ts{top[0]}][ts{top[1]}]hstack[top];"
                       f"[ts{bottom[0]}][ts{bottom[1]}]hstack[bot];"
                       f"[top][bot]vstack")
            print(f'[ViralSplit] 4 soggetti: griglia 2x2, riga alta {top}, riga bassa {bottom}')

        graph = "".join(chains) + compose + "[vout]"

        if ass_path:
            ass_str = str(ass_path).replace('\\', '/')
            graph += f";[vout]ass='{ass_str}'[vass]"
            print(f'[ViralSplit] Sottotitoli karaoke sul composite: {ass_str}')

        print(f'[ViralSplit] Inizio filter_complex: {graph[:160]}...')
        return graph
