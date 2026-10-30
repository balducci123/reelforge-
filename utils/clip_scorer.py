from typing import List, Dict


class ClipScorer:
    """
    Valuta ogni candidato clip e assegna un punteggio 0-100 basato su:
    - Hook: le prime 3 secondi contengono parole chiave ad alto impatto?
    - Energia: l'energia audio è alta durante il clip?
    - Ritenzione: pochi silenzi, parlato continuo
    - Soggetto: volto/corpo presente costantemente (smart crop efficace)
    - Durata: nella gamma ottimale 30-60s
    """

    # Parole chiave hook in italiano/inglese (inizio video)
    HOOK_WORDS_IT = [
        'ecco', 'guarda', 'incredibile', 'segreto', 'trucco', 'consiglio',
        'non', 'mai', 'sempre', 'perché', 'come', 'cosa', 'quando', 'dove',
        'wow', 'pazzesco', 'assurdo', 'finale', 'aspetta', 'ascolta',
        'risultato', 'scopri', 'impara', 'errore', 'evita', 'atento',
        'ascolta bene', 'fermo', 'stop', 'leggi', 'sapere', 'voglio'
    ]
    HOOK_WORDS_EN = [
        'wait', 'look', 'incredible', 'secret', 'trick', 'tip',
        'never', 'always', 'why', 'how', 'what', 'when', 'where',
        'wow', 'crazy', 'insane', 'finally', 'listen', 'result',
        'discover', 'learn', 'mistake', 'avoid', 'attention', 'stop'
    ]

    def __init__(self):
        self.hook_words = set(w.lower() for w in self.HOOK_WORDS_IT + self.HOOK_WORDS_EN)

    def score(self, candidate: Dict, audio_analysis: Dict = None, face_present: bool = True) -> Dict:
        """
        Ritorna dizionario con punteggio totale e sotto-punteggi.
        candidate: {start, end, duration, text, words, has_hook}
        audio_analysis: output di AudioAnalyzer.analyze()
        face_present: se il volto è stato rilevato nella maggioranza dei frame campionati
        """
        scores = {}

        # 1) Hook score (0-30): parole chiave all'inizio
        text = candidate.get('text', '').lower()
        first_words = text.split()[:5]
        hook_hits = sum(1 for w in first_words if any(hw in w for hw in self.hook_words))
        has_hook_flag = candidate.get('has_hook', False)
        hook_score = min(30, (hook_hits * 10) + (15 if has_hook_flag else 0))
        scores['hook'] = hook_score

        # 2) Energia audio (0-25): presenza di picchi nel range del clip
        energy_score = 0
        if audio_analysis and audio_analysis.get('peak_times'):
            clip_start = candidate.get('start', candidate.get('start_time', 0.0))
            clip_end = candidate.get('end', clip_start + candidate.get('duration', 0.0))
            peaks_in_clip = [p for p in audio_analysis['peak_times'] if clip_start <= p <= clip_end]
            # Normalizza: fino a 5 picchi = 25 punti
            energy_score = min(25, len(peaks_in_clip) * 5)
        scores['energy'] = energy_score

        # 3) Ritenzione (0-25): pochi silenzi, parlato continuo
        retention_score = 25
        if audio_analysis and audio_analysis.get('silent_ranges'):
            clip_start = candidate.get('start', candidate.get('start_time', 0.0))
            clip_end = candidate.get('end', clip_start + candidate.get('duration', 0.0))
            silence_in_clip = 0
            for s, e in audio_analysis['silent_ranges']:
                overlap_start = max(s, clip_start)
                overlap_end = min(e, clip_end)
                if overlap_end > overlap_start:
                    silence_in_clip += (overlap_end - overlap_start)
            clip_duration = candidate['duration']
            if clip_duration > 0:
                silence_ratio = silence_in_clip / clip_duration
                retention_score = max(0, 25 - int(silence_ratio * 50))  # ogni 2% silenzio = -1 punto
        scores['retention'] = retention_score

        # 4) Presenza soggetto (0-10): utile per smart crop
        face_score = 10 if face_present else 2
        scores['subject'] = face_score

        # 5) Durata ottimale (0-10): 30-45s è il sweet spot per TikTok/Reels
        dur = candidate['duration']
        if 30 <= dur <= 45:
            duration_score = 10
        elif 45 < dur <= 60:
            duration_score = 8
        elif 25 <= dur < 30:
            duration_score = 6
        else:
            duration_score = 4
        scores['duration'] = duration_score

        total = sum(scores.values())
        scores['total'] = min(100, total)
        scores['grade'] = self._grade(total)
        return scores

    def _grade(self, score: int) -> str:
        if score >= 90: return 'S'
        if score >= 80: return 'A'
        if score >= 70: return 'B'
        if score >= 60: return 'C'
        if score >= 50: return 'D'
        return 'E'

    def rank_candidates(self, candidates: List[Dict], audio_analysis: Dict = None,
                        face_present: bool = True, top_n: int = 5) -> List[Dict]:
        """
        Aggiunge punteggio a ogni candidato e restituisce i migliori ordinati per score discendente.
        """
        scored = []
        for c in candidates:
            s = self.score(c, audio_analysis=audio_analysis, face_present=face_present)
            c_copy = dict(c)
            c_copy['score'] = s
            scored.append(c_copy)

        scored.sort(key=lambda x: x['score']['total'], reverse=True)
        return scored[:top_n]
