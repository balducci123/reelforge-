import os
from pathlib import Path

try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False


class Transcriber:
    def __init__(self, model_size="base"):
        self.model = None
        self.model_size = model_size
        if WHISPER_AVAILABLE:
            try:
                self.model = whisper.load_model(model_size)
            except Exception:
                self.model = None
    
    def transcribe(self, audio_path):
        if not WHISPER_AVAILABLE or self.model is None:
            return []
        try:
            result = self.model.transcribe(str(audio_path), language='it')
            segments = []
            for seg in result.get('segments', []):
                segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text']
                })
            return segments
        except Exception:
            return []
    
    def get_smart_cut_points(self, segments, target_duration=30, num_clips=3):
        if not segments:
            return []
        cut_points = []
        total_duration = segments[-1]['end'] if segments else target_duration * num_clips
        interval = total_duration / num_clips if num_clips > 0 else target_duration
        for i in range(num_clips):
            target_time = i * interval
            best_seg = None
            min_diff = float('inf')
            for seg in segments:
                diff = abs(seg['start'] - target_time)
                if diff < min_diff:
                    min_diff = diff
                    best_seg = seg
            if best_seg:
                cut_points.append(max(0, best_seg['start']))
            else:
                cut_points.append(i * target_duration)
        return cut_points
