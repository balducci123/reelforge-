"""
Publisher TikTok per ReelForge via API ufficiale (Content Posting API v2:
Direct Post). OAuth2 PKCE, upload a blocchi FILE_UPLOAD, polling di stato.

Credenziali e token restano SOLO nel disco locale:
  data/tiktok_config.json  -> client_key + client_secret + redirect_uri
  data/tiktok_tokens.json  -> access_token, refresh_token, open_id ...
(entrambi git-ignorati)

Limiti noti dell'API (rif. documentazione TikTok, 2026):
- 6 richieste di publish al minuto per access token (sequentialmente ok)
- Client NON ancora auditato (sandbox/appIntermediate): solo SELF_ONLY
  (il video finisce privato nell'account). administrators dell'app devono
  passare l'audit per pubblicare pubblici.
"""
import base64
import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

BASE = 'https://open.tiktokapis.com'
AUTH_BASE = 'https://www.tiktok.com/v2/auth/authorize/'
SCOPE = 'user.info.basic,video.publish'

DATA_DIR = Path('data')
CONFIG_FILE = DATA_DIR / 'tiktok_config.json'
TOKENS_FILE = DATA_DIR / 'tiktok_tokens.json'

CHUNK_SIZE = 32 * 1024 * 1024          # 32 MB per blocco (max API: 64 MB)
TITLE_MAX_LEN = 90                      # prudenziale sull'attributo title
STATUS_POLL_TRIES = 6
STATUS_POLL_INTERVAL = 3.0


def _load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path, data):
    DATA_DIR.mkdir(exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    try:
        os.chmod(path, 0o600)  # minimizza permessi su sistemi POSIX
    except OSError:
        pass


class TikTokPublisher:
    """Client OAuth2 + Direct Post per TikTok."""

    def __init__(self):
        # stato OAuth transiente (login -> callback); persiste su file perche'
        # il reloader di debug puo' riciclare il processo tra le due richieste
        self._oauth_pending_file = DATA_DIR / 'tiktok_oauth_pending.json'

    # ------------------------------------------------------------------ #
    # Configurazione (client_key/secret del desarrollador portale)
    # ------------------------------------------------------------------ #
    def configured(self):
        cfg = _load_json(CONFIG_FILE)
        return bool(cfg.get('client_key') and cfg.get('client_secret'))

    def save_config(self, client_key, client_secret, redirect_uri=None):
        cfg = _load_json(CONFIG_FILE)
        if client_key:
            cfg['client_key'] = client_key.strip()
        if client_secret:
            cfg['client_secret'] = client_secret.strip()
        if redirect_uri:
            cfg['redirect_uri'] = redirect_uri.strip()
        cfg.setdefault('redirect_uri', 'https://balducci123.github.io/reelforge-/callback.html')
        _save_json(CONFIG_FILE, cfg)
        return cfg

    def get_config_safe(self):
        cfg = _load_json(CONFIG_FILE)
        key = cfg.get('client_key', '')
        return {
            'configured': self.configured(),
            'client_key': (key[:4] + '••••' + key[-3:]) if len(key) > 7 else (key if key else ''),
            'redirect_uri': cfg.get('redirect_uri', 'https://balducci123.github.io/reelforge-/callback.html'),
        }

    # ------------------------------------------------------------------ #
    # Token / OAuth2 PKCE
    # ------------------------------------------------------------------ #
    def connected(self):
        tok = _load_json(TOKENS_FILE)
        return bool(tok.get('access_token'))

    def _save_tokens(self, payload):
        payload['expires_at'] = time.time() + int(payload.get('expires_in', 7200))
        _save_json(TOKENS_FILE, payload)

    def _access_token(self):
        tok = _load_json(TOKENS_FILE)
        if not tok.get('access_token'):
            raise RuntimeError('Nessun account TikTok collegato. Vai su "Collega TikTok".')
        # refresh se manca < 90s a scadere
        if time.time() > tok.get('expires_at', 0) - 90:
            tok = self._refresh_token(tok)
        return tok

    def _refresh_token(self, tok):
        cfg = _load_json(CONFIG_FILE)
        r = requests.post(f'{BASE}/v2/oauth/token/', data={
            'client_key': cfg['client_key'],
            'client_secret': cfg['client_secret'],
            'grant_type': 'refresh_token',
            'refresh_token': tok['refresh_token'],
        }, timeout=30)
        data = r.json()
        if 'access_token' not in data:
            print(f'[TikTok] Refresh token fallito: {data}')
            raise RuntimeError("Refresh token TikTok fallito: ricollega l'account.")
        print('[TikTok] Access token rinfrescato')
        self._save_tokens(data)
        return _load_json(TOKENS_FILE)

    def get_authorize_url(self):
        if not self.configured():
            raise RuntimeError('Client key/secret TikTok non configurati.')
        cfg = _load_json(CONFIG_FILE)
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
        state = secrets.token_urlsafe(16)
        _save_json(self._oauth_pending_file, {'verifier': verifier, 'state': state})
        qs = urlencode({
            'client_key': cfg['client_key'],
            'scope': SCOPE,
            'response_type': 'code',
            'redirect_uri': cfg.get('redirect_uri', 'https://balducci123.github.io/reelforge-/callback.html'),
            'state': state,
            'code_challenge': challenge,
            'code_challenge_method': 'S256',
        })
        return f'{AUTH_BASE}?{qs}'

    def exchange_code(self, code, state):
        try:
            pending = _load_json(self._oauth_pending_file)
            if not pending or pending.get('state') != state:
                return False, 'Stato OAuth non valido: rilanciare il collegamento.'
            cfg = _load_json(CONFIG_FILE)
            r = requests.post(f'{BASE}/v2/oauth/token/', data={
                'client_key': cfg['client_key'],
                'client_secret': cfg['client_secret'],
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': cfg.get('redirect_uri'),
                'code_verifier': pending['verifier'],
            }, timeout=30)
            data = r.json()
            if 'access_token' not in data:
                print(f'[TikTok] Scambio codice fallito: {data}')
                return False, data.get('error_description') or data.get('message') or 'OAuth fallito'
            self._save_tokens(data)
            print(f"[TikTok] Connesso: open_id={data.get('open_id')}")
            self._oauth_pending_file.unlink(missing_ok=True)
            return True, ''
        finally:
            if self._oauth_pending_file.exists():
                try:
                    self._oauth_pending_file.unlink()
                except OSError:
                    pass

    def disconnect(self):
        TOKENS_FILE.unlink(missing_ok=True)
        print('[TikTok] Account scollegato (tokens rimossi)')

    def get_user_info(self):
        tok = self._access_token()
        r = requests.get(f'{BASE}/v2/user/info/',
                         headers={'Authorization': f"Bearer {tok['access_token']}"},
                         params={'fields': 'open_id,display_name,avatar_url'},
                         timeout=20)
        data = r.json().get('data', {})
        user = (data or {}).get('user', {})
        if not user:
            return {}
        return {'display_name': user.get('display_name') or user.get('username') or 'TikTok',
                'avatar_url': user.get('avatar_url') or ''}

    # ------------------------------------------------------------------ #
    # Publish (Direct Post FILE_UPLOAD)
    # ------------------------------------------------------------------ #
    def publish_video(self, file_path, caption, privacy_level='SELF_ONLY',
                      progress_cb=None):
        """
        Pubblica un file video locale. Torna dict con esito/ stato.

        progress_cb(pct:int, phase:str) per la barra di avanzamento.
        """
        path = Path(file_path)
        if not path.exists():
            return {'ok': False, 'error': f'File non trovato: {path.name}'}

        caption = (caption or '').strip()[:TITLE_MAX_LEN]
        tok = self._access_token()
        auth = {'Authorization': f"Bearer {tok['access_token']}"}

        def cb(pct, phase):
            if progress_cb:
                try:
                    progress_cb(max(0, min(100, int(pct))), phase)
                except Exception:
                    pass

        size = path.stat().st_size
        if size <= 0:
            return {'ok': False, 'error': 'File video vuoto'}
        total_chunks = max(1, -(-size // CHUNK_SIZE))

        # 1) init
        cb(2, 'inizializzazione')
        init_body = {
            'post_info': {
                'title': caption,
                'privacy_level': privacy_level,
            },
            'source_info': {
                'source': 'FILE_UPLOAD',
                'video_size': size,
                'chunk_size': CHUNK_SIZE,
                'total_chunk_count': total_chunks,
            },
        }
        r = requests.post(f'{BASE}/v2/post/publish/video/init/',
                          headers={**auth, 'Content-Type': 'application/json; charset=UTF-8'},
                          data=json.dumps(init_body), timeout=30)
        data = r.json().get('data', {})
        err = r.json().get('error', {})
        if err.get('code') not in ('ok', ''):
            msg = err.get('message') or err.get('code') or 'init fallito'
            print(f"[TikTok] init errore [{err.get('code')}]: {msg}")
            # Client non auditato -> solo account privato: riduco a SELF_ONLY
            if err.get('code') == 'unaudited_client_can_only_post_to_private_accounts' \
                    and privacy_level != 'SELF_ONLY':
                print('[TikTok] Client non auditato: riprovo con SELF_ONLY (video privato)')
                return self.publish_video(file_path, caption, 'SELF_ONLY', progress_cb)
            return {'ok': False, 'error': msg}
        publish_id = data.get('publish_id')
        upload_url = data.get('upload_url')
        if not publish_id or not upload_url:
            return {'ok': False, 'error': 'Risposta init senza publish_id'}

        # 2) upload a blocchi
        sent = 0
        with open(path, 'rb') as f:
            for i in range(total_chunks):
                chunk = f.read(CHUNK_SIZE)
                end = sent + len(chunk) - 1
                headers = {
                    **auth,
                    'Content-Type': 'video/mp4',
                    'Content-Length': str(len(chunk)),
                    'Content-Range': f'bytes {sent}-{end}/{size}',
                }
                r = requests.put(upload_url, headers=headers, data=chunk, timeout=600)
                if r.status_code not in (200, 201, 206):
                    return {'ok': False,
                            'error': f'Upload chunk {i + 1}/{total_chunks} HTTP {r.status_code}: {r.text[:200]}'}
                sent += len(chunk)
                cb(2 + 93 * (sent / size), f'caricamento {sent // (1024 * 1024)} MB')

        # 3) polling di stato
        cb(97, 'elaborazione TikTok')
        status = 'SENT_TO_TIKTOK'
        for _ in range(STATUS_POLL_TRIES):
            time.sleep(STATUS_POLL_INTERVAL)
            r = requests.post(f'{BASE}/v2/post/publish/status/fetch/',
                              headers={**auth, 'Content-Type': 'application/json'},
                              data=json.dumps({'publish_id': publish_id}), timeout=30)
            sd = r.json().get('data', {})
            status = sd.get('status') or status
            if status in ('PUBLISHED', 'FAILED'):
                break
        cb(100, 'completato')
        ok = status == 'PUBLISHED'
        print(f'[TikTok] {path.name}: {status} (publish_id={publish_id})')
        return {'ok': ok, 'status': status, 'publish_id': publish_id,
                'privacy': privacy_level,
                'error': None if ok else f'TikTok: stato finale {status}'}


if __name__ == '__main__':
    tp = TikTokPublisher()
    print('configurato:', tp.configured(), '| connesso:', tp.connected())
