import numpy as np
from pathlib import Path

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    LIBROSA_AVAILABLE = False


class AudioAnalyzer:
    """
    Analisi audio con librosa per ottimizzare il taglio virale.
    Rileva:
    - Picchi di energia (colpi, risate, momenti alti)
    - Silenzi (punti di taglio sicuri)
    - Segmenti di parlato continuo vs musica
    - Punti di hook consigliati (inizio con energia alta)
    """

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def analyze(self, audio_path):
        """
        Ritorna dizionario con:
        - energy_profile: lista di (tempo, energia_RMS) normalizzata 0-1
        - silent_ranges: lista di (start, end) dei silenzi
        - peak_times: tempi dei picchi di energia più alti
        - speech_segments: segmenti di parlato stimati (energia sopra soglia)
        - duration: durata totale
        """
        if not LIBROSA_AVAILABLE:
            return self._fallback_analysis(audio_path)

        try:
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            # RMS energy per frame (frame_length=2048, hop=512)
            hop_length = 512
            frame_length = 2048
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

            # Normalizza energia 0-1
            if rms.max() > 0:
                rms_norm = rms / rms.max()
            else:
                rms_norm = rms

            # Soglia dinamica per silenzio (20% del max)
            silence_threshold = 0.15
            silent_mask = rms_norm < silence_threshold

            # Trova range di silenzio consecutivi
            silent_ranges = []
            in_silence = False
            silence_start = 0
            for i, is_silent in enumerate(silent_mask):
                if is_silent and not in_silence:
                    in_silence = True
                    silence_start = times[i]
                elif not is_silent and in_silence:
                    in_silence = False
                    silent_ranges.append((round(silence_start, 2), round(times[i], 2)))
            if in_silence:
                silent_ranges.append((round(silence_start, 2), round(times[-1], 2)))

            # Filtra silenzi troppo brevi (< 0.3s)
            silent_ranges = [(s, e) for s, e in silent_ranges if (e - s) > 0.3]

            # Picchi di energia: punti locali sopra 70% del max
            peak_threshold = 0.70
            peak_times = []
            for i in range(1, len(rms_norm) - 1):
                if (rms_norm[i] > peak_threshold and
                        rms_norm[i] >= rms_norm[i - 1] and
                        rms_norm[i] >= rms_norm[i + 1]):
                    peak_times.append(round(float(times[i]), 2))

            # Mantieni solo i picchi più significativi (max 20, spaziati almeno 1s)
            peak_times = self._space_peaks(peak_times, min_gap=1.0)

            # Segmenti di parlato stimati (energia sopra 30% del max)
            speech_threshold = 0.30
            speech_mask = rms_norm > speech_threshold
            speech_segments = []
            in_speech = False
            speech_start = 0
            for i, is_speech in enumerate(speech_mask):
                if is_speech and not in_speech:
                    in_speech = True
                    speech_start = times[i]
                elif not is_speech and in_speech:
                    in_speech = False
                    speech_segments.append((round(speech_start, 2), round(times[i], 2)))
            if in_speech:
                speech_segments.append((round(speech_start, 2), round(times[-1], 2)))

            return {
                'duration': round(duration, 2),
                'energy_profile': list(zip(times.tolist(), rms_norm.tolist())),
                'silent_ranges': silent_ranges,
                'peak_times': peak_times,
                'speech_segments': speech_segments,
                'sample_rate': sr
            }

        except Exception as e:
            print(f'[AudioAnalyzer] Errore librosa: {e}')
            return self._fallback_analysis(audio_path)

    def _space_peaks(self, peak_times, min_gap=1.0):
        """Mantiene picchi spaziati almeno min_gap secondi, prende i più alti."""
        if not peak_times:
            return []
        result = [peak_times[0]]
        for t in peak_times[1:]:
            if t - result[-1] >= min_gap:
                result.append(t)
        return result[:20]

    def _fallback_analysis(self, audio_path):
        """Analisi minimale senza librosa."""
        from pathlib import Path
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(audio_path)],
                capture_output=True, timeout=15
            )
            import json
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
        except Exception:
            duration = 60.0
        return {
            'duration': round(duration, 2),
            'energy_profile': [],
            'silent_ranges': [],
            'peak_times': [],
            'speech_segments': [(0.0, duration)],
            'sample_rate': 22050
        }

    def find_best_cut_points(self, analysis, num_clips=3, min_duration=30, max_duration=60):
        """
        Trova i migliori punti di inizio per i clip basandosi su:
        - Inizio vicino a un picco di energia (hook)
        - Fine vicino a un silenzio o fine parlato
        - Durata entro min/max
        """
        duration = analysis['duration']
        peaks = analysis['peak_times']
        silences = analysis['silent_ranges']
        speech = analysis['speech_segments']

        # Candidati di inizio: picchi di energia o inizio segmenti di parlato
        start_candidates = []
        for p in peaks:
            if p < duration - min_duration:
                start_candidates.append(p)
        for s, e in speech:
            if s < duration - min_duration:
                start_candidates.append(round(s, 2))
        if not start_candidates:
            start_candidates = [0.0, duration / 2]

        # Rimuovi duplicati e ordina
        start_candidates = sorted(set(start_candidates))

        # Per ogni candidato, trova la fine migliore (silencio più vicino dopo min_duration)
        candidates = []
        for start in start_candidates:
            ideal_end = start + min_duration
            max_end = min(start + max_duration, duration)

            # Cerca silenzio dopo ideal_end
            best_end = max_end
            for s, e in silences:
                if ideal_end <= s <= max_end:
                    best_end = round(s, 2)
                    break
            if best_end < ideal_end:
                best_end = ideal_end
            if best_end - start < min_duration:
                best_end = min(start + min_duration, duration)

            if best_end - start >= min_duration:
                candidates.append({
                    'start': round(start, 2),
                    'end': round(best_end, 2),
                    'duration': round(best_end - start, 2)
                })

        # Se non abbastanza candidati, aggiungi fissi
        if len(candidates) < num_clips:
            for i in range(num_clips):
                s = i * min_duration
                e = min(s + min_duration, duration)
                if e - s >= min_duration and not any(abs(c['start'] - s) < 1 for c in candidates):
                    candidates.append({'start': round(s, 2), 'end': round(e, 2), 'duration': round(e - s, 2)})

        candidates.sort(key=lambda c: c['start'])
        return candidates[:num_clips]
