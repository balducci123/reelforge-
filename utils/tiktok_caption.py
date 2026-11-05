"""
Generatore di caption TikTok per ReelForge.

Estrae dalla trascrizione del clip (gia' prodotta da faster-whisper) la
frase forte e la accompagna con hashtag pertinenti. Nessuna API esterna:
funziona offline e senza costi — se un domani si vogliano LLM si puo'
sostituire solo questa funzione mantenendo l'interfaccia.
"""
import re

# Hashtag ben noti su TikTok, per tema
_THEME_TAGS = {
    'calcio': ['#calcio', '#seriea', '#perte'],
    'sport': ['#sport', '#allenamento', '#perte'],
    'food': ['#food', '#ricetta', '#cucina'],
    'tech': ['#tech', '#tecnologia', '#impararesutiktok'],
    'auto': ['#auto', '#automobili', '#perte'],
    'money': ['#soldi', '#lavoro', '#money'],
    'viaggio': ['#viaggio', '#viaggi', '#italia'],
    'comedy': ['#comedy', '#neiperte', '#ridere'],
}

_DEFAULT_TAGS = ['#perte', '#neiperte', '#viral']

_HOOK_EMOJI = {
    'ecco': '👀', 'guarda': '👀', 'incredibile': '🔥', 'segreto': '🤫',
    'trucco': '🤫', 'consiglio': '💡', 'mai': '🚫', 'sempre': '✅',
    'errore': '❌', 'evita': '⚠️', 'scopri': '🔍', 'wow': '🤯',
}


def _themes_from_text(text):
    """Temi dominanti in base a keyword semplici (it + en)."""
    t = ' ' + (text or '').lower() + ' '
    themes = []
    if any(k in t for k in ['calcio', 'squadra', 'partita', ' gol ', 'campo ',
                            'allenamento', 'portiere', 'tiro ', 'maglia']):
        themes.append('calcio')
    if any(k in t for k in ['sport', 'corsa', 'palestra', 'esercizio', 'workout']):
        themes.append('sport')
    if any(k in t for k in ['ricetta', 'cucina', 'ingredienti', 'gusto',
                            'pranzo', 'cena', 'pasta', 'dessert', 'forno']):
        themes.append('food')
    if any(k in t for k in [' pc ', ' app ', 'software', 'programma', 'codice',
                            ' intelligenza ', 'gadget', 'applicazione']):
        themes.append('tech')
    if any(k in t for k in [' auto ', 'macchina', 'motore', 'gara ', 'km ']):
        themes.append('auto')
    if any(k in t for k in ['soldi', ' euro ', 'lavoro', 'guadagn', 'investi']):
        themes.append('money')
    if any(k in t for k in ['viaggio', 'vacanza', 'citta', 'paese', 'hotel', 'aereo']):
        themes.append('viaggio')
    if any(k in t for k in ['ridere', 'comico', 'pazzo', 'imbarazzo', 'risata']):
        themes.append('comedy')
    return themes


def _pick_sentence(text, hook_words):
    """Restituisce la frase piu' forte: prima con hook, altrimenti la prima lunga."""
    if not text:
        return ''
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip(' .,') for s in sentences if len(s.strip()) >= 10]
    if not sentences:
        return text.strip()[:60]
    for s in sentences:
        low = s.lower()
        if any(hw in low for hw in hook_words):
            return s
    return max(sentences, key=len)


def generate_caption(clip_text, max_len=90, extra_tags=None):
    """
    Caption per un clip a partire dal suo testo trascritto.

    TikTok limita il titolo del Direct Post: tagliamo caption+tag alla
    soglia, mantenendo sempre in coda almeno un tag principale.
    """
    text = (clip_text or '').strip()
    hooks = ['ecco', 'guarda', 'incredibile', 'segreto', 'trucco', 'consiglio',
             'mai', 'sempre', 'errore', 'evita', 'scopri', 'wow',
             'wait', 'listen', 'secret', 'trick', 'avoid', 'discover',
             'never', 'always']

    body = _pick_sentence(text, hooks)
    if body:
        low = body.lower()
        for hw, emo in _HOOK_EMOJI.items():
            if hw in low:
                body = f'{emo} {body}'
                break
        body = body[0].upper() + body[1:]

    tags = list(_DEFAULT_TAGS)
    for th in _themes_from_text(text):
        tags.extend(_THEME_TAGS[th])
    if extra_tags:
        tags = list(extra_tags) + tags
    seen, uniq = set(), []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)

    if body:
        caption = f'{body}  {" ".join(uniq)}'
    else:
        caption = '✂ Nuovo reel • ReelForge  ' + ' '.join(uniq)

    if len(caption) > max_len:
        caption = caption[:max_len - 1].rstrip() + '…'
    return caption


if __name__ == '__main__':
    demo = [
        'ecco il trucco che nessuno ti dice su questo allenamento ascolta bene',
        'la mia squadra ha fatto un gol incredibile in campo, non ci credo ancora',
        'questa ricetta si fa in dieci minuti e vi piace davvero, guardate il risultato finale',
        '',
    ]
    for d in demo:
        cap = generate_caption(d)
        print(f'({len(cap)}c) {cap}')
