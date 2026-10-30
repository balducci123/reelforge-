import os
import json
import zipfile
import urllib.request
from pathlib import Path

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    WhisperModel = None
    FASTER_WHISPER_AVAILABLE = False

try:
    import vosk
    VOSK_AVAILABLE = True
except Exception:
    vosk = None
    VOSK_AVAILABLE = False


class Transcriber:
    """
    Trascrizione locale con fallback multi-backend:
    1. faster-whisper (più accurato, richiede build C++)
    2. vosk (leggero, wheel precompilata, modello scaricato automaticamente)
    3. Placeholder temporali (sempre funzionante, mostra i tempi sul video)
    """

    HOOK_WORDS = {
        'ecco', 'guarda', 'incredibile', 'segreto', 'trucco', 'consiglio',
        'non', 'mai', 'sempre', 'perché', 'come', 'cosa', 'quando', 'dove',
        'wow', 'pazzesco', 'assurdo', 'finale', 'aspetta', 'ascolta',
        'risultato', 'scopri', 'impara', 'errore', 'evita', 'atento',
        'fermo', 'stop', 'leggi', 'sapere', 'voglio', 'ascolta bene',
        'wait', 'look', 'incredible', 'secret', 'trick', 'tip',
        'never', 'always', 'why', 'how', 'what', 'when', 'where',
        'listen', 'result', 'discover', 'learn', 'mistake', 'avoid',
        'attention', 'crazy', 'insane', 'finally'
    }

    VOSK_MODELS = {
        'it': 'https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip',
        'en': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
    }

    def __init__(self, model_size="base", device="cpu", compute_type="int8", lang='it'):
        self.model = None
        self.backend = None
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.lang = lang
        self.models_dir = Path('models') / 'vosk'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 1) Prova faster-whisper
        if FASTER_WHISPER_AVAILABLE and WhisperModel is not None:
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
                self.backend = 'faster-whisper'
                print(f'[Transcriber] faster-whisper {model_size} caricato su {device} ({compute_type})')
                return
            except Exception as e:
                print(f'[Transcriber] faster-whisper non disponibile: {e}')

        # 2) Prova Vosk
        if VOSK_AVAILABLE and vosk is not None:
            try:
                model_path = self._ensure_vosk_model()
                if model_path:
                    self.model = vosk.Model(str(model_path))
                    self.backend = 'vosk'
                    print(f'[Transcriber] Vosk caricato da {model_path}')
                    return
            except Exception as e:
                print(f'[Transcriber] Vosk non disponibile: {e}')

        # 3) Fallback: nessun backend
        print('[Transcriber] Nessun backend di trascrizione disponibile. '
              'I sottotitoli saranno generati come placeholder temporali.')
        self.backend = None

    def _ensure_vosk_model(self):
        """Scarica il modello Vosk se non presente."""
        model_name = f'vosk-model-small-{self.lang}-0.22' if self.lang == 'it' else f'vosk-model-small-en-us-0.15'
        model_dir = self.models_dir / model_name
        if model_dir.exists():
            return model_dir

        zip_path = self.models_dir / f'{model_name}.zip'
        url = self.VOSK_MODELS.get('it' if self.lang == 'it' else 'en')

        print(f'[Transcriber] Download modello Vosk {model_name}...')
        try:
            urllib.request.urlretrieve(url, str(zip_path))
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(str(self.models_dir))
            zip_path.unlink()
            if model_dir.exists():
                print(f'[Transcriber] Modello Vosk installato in {model_dir}')
                return model_dir
        except Exception as e:
            print(f'[Transcriber] Download modello Vosk fallito: {e}')
        return None

    def transcribe_words(self, audio_path):
        """
        Ritorna lista di dizionari: {word, start, end, probability}
        """
        if self.backend == 'faster-whisper':
            return self._transcribe_faster_whisper(audio_path)
        if self.backend == 'vosk':
            return self._transcribe_vosk(audio_path)
        return self._placeholder_words(audio_path)

    def _transcribe_faster_whisper(self, audio_path):
        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language=self.lang,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400)
            )
            words = []
            for seg in segments:
                if hasattr(seg, 'words') and seg.words:
                    for w in seg.words:
                        words.append({
                            'word': w.word.strip(),
                            'start': float(w.start),
                            'end': float(w.end),
                            'probability': float(getattr(w, 'probability', 0.0))
                        })
            print(f'[Transcriber] faster-whisper: {len(words)} parole')
            return words
        except Exception as e:
            print(f'[Transcriber] Errore faster-whisper: {e}')
            return self._placeholder_words(audio_path)

    def _transcribe_vosk(self, audio_path):
        import wave
        import json as _json
        try:
            wf = wave.open(str(audio_path), "rb")
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in (8000, 16000, 32000, 44100, 48000):
                # Vosk richiede WAV mono PCM 16-bit; se non lo è, fallback
                wf.close()
                return self._placeholder_words(audio_path)

            rec = vosk.KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)

            words = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = _json.loads(rec.Result())
                    for w in res.get('result', []):
                        words.append({
                            'word': w.get('word', ''),
                            'start': float(w.get('start', 0)),
                            'end': float(w.get('end', 0)),
                            'probability': float(w.get('conf', 0.0))
                        })
            # Ultimo risultato
            res = _json.loads(rec.FinalResult())
            for w in res.get('result', []):
                words.append({
                    'word': w.get('word', ''),
                    'start': float(w.get('start', 0)),
                    'end': float(w.get('end', 0)),
                    'probability': float(w.get('conf', 0.0))
                })
            wf.close()
            print(f'[Transcriber] Vosk: {len(words)} parole')
            return words
        except Exception as e:
            print(f'[Transcriber] Errore Vosk: {e}')
            return self._placeholder_words(audio_path)

    def _placeholder_words(self, audio_path):
        """Genera parole placeholder ogni 2 secondi basate sulla durata del file."""
        from pathlib import Path as _Path
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(audio_path)],
                capture_output=True, timeout=15
            )
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
        except Exception:
            duration = 60.0

        words = []
        t = 0.0
        while t < duration:
            words.append({
                'word': f'[{t:.0f}s]',
                'start': round(t, 2),
                'end': round(min(t + 2.0, duration), 2),
                'probability': 1.0
            })
            t += 2.0
        print(f'[Transcriber] Placeholder: {len(words)} eventi temporali')
        return words

    def get_smart_segments(self, words, min_clip_duration=30, max_clip_duration=60):
        if not words:
            return []

        # 1) Raggruppa parole separate da pause > 0.4s
        segments = []
        current = []
        for i, w in enumerate(words):
            current.append(w)
            if i + 1 < len(words):
                gap = words[i + 1]['start'] - w['end']
                if gap > 0.4:
                    segments.append(current)
                    current = []
        if current:
            segments.append(current)

        # 2) Costruisci candidati
        candidates = []
        i = 0
        while i < len(segments):
            start = segments[i][0]['start']
            end = segments[i][-1]['end']
            text = ' '.join(w['word'] for w in segments[i])
            has_hook = any(hw in text.lower() for hw in self.HOOK_WORDS)

            j = i + 1
            while j < len(segments) and (end - start) < min_clip_duration:
                next_seg = segments[j]
                end = next_seg[-1]['end']
                text = text + ' ' + ' '.join(w['word'] for w in next_seg)
                if any(hw in next_seg[-1]['word'].lower() for hw in self.HOOK_WORDS):
                    has_hook = True
                j += 1

            if end - start > max_clip_duration:
                acc = 0
                cut_words = []
                for w in segments[i]:
                    if acc + (w['end'] - w['start']) <= max_clip_duration:
                        cut_words.append(w)
                        acc = w['end'] - start
                    else:
                        break
                if cut_words:
                    end = cut_words[-1]['end']
                    text = ' '.join(w['word'] for w in cut_words)
                else:
                    end = min(start + max_clip_duration, words[-1]['end'])

            candidates.append({
                'start': round(start, 2),
                'end': round(end, 2),
                'duration': round(end - start, 2),
                'text': text,
                'has_hook': has_hook,
                'words': [w for w in words if start <= w['start'] and w['end'] <= end]
            })
            i = j

        candidates.sort(key=lambda c: (not c['has_hook'], -c['duration']))
        return candidates[:10]
