from pathlib import Path

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    WhisperModel = None
    FASTER_WHISPER_AVAILABLE = False


class Transcriber:
    """
    Trascrizione locale con faster-whisper (molto più veloce e leggero di openai-whisper).
    Estrae timestamp a livello di PAROLA per sottotitoli karaoke e taglio intelligente.
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

    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        self.model = None
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        if FASTER_WHISPER_AVAILABLE and WhisperModel is not None:
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
                print(f'[Transcriber] faster-whisper {model_size} caricato su {device} ({compute_type})')
            except Exception as e:
                print(f'[Transcriber] Errore caricamento modello: {e}')
                self.model = None
        else:
            print('[Transcriber] faster-whisper non disponibile. Installa: pip install faster-whisper')

    def transcribe_words(self, audio_path):
        """
        Ritorna lista di dizionari: {word, start, end, probability}
        """
        if not FASTER_WHISPER_AVAILABLE or self.model is None:
            return []
        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language='it',
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
            print(f'[Transcriber] Estratte {len(words)} parole con timestamp')
            return words
        except Exception as e:
            print(f'[Transcriber] Errore trascrizione: {e}')
            return []

    def get_smart_segments(self, words, min_clip_duration=30, max_clip_duration=60):
        """
        Raggruppa parole in segmenti di parlato continuo e restituisce candidati clip
        che iniziano e finiscono su confini di frase/parola.
        """
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

        # 2) Costruisci candidati unendo segmenti adiacenti per raggiungere durata target
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

            # Taglia se supera max_clip_duration (taglia all'ultimo confine di parola)
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

        # Ordina per: hook prima, poi durata
        candidates.sort(key=lambda c: (not c['has_hook'], -c['duration']))
        return candidates[:10]
