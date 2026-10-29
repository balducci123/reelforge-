# ReelForge

Piattaforma all-in-one per la creazione di video verticali 9:16 da YouTube, costruita con Flask, FFmpeg, OpenCV, MediaPipe, Whisper e Socket.IO.

## Funzionalità

- **Smart Auto-Reframe & Face Tracking**: rilevamento del soggetto principale con OpenCV/MediaPipe (o YOLO) e crop dinamico 9:16 che segue chi parla
- **Trascrizione AI e Taglio Intelligente**: OpenAI Whisper per individuare momenti di parlato continuo e tagliare a frasi sensate
- **Coda Asincrona e WebSockets**: elaborazione in background con Flask-Executor + Socket.IO e barra di avanzamento in tempo reale per ogni clip
- **UI/UX di livello superiore**: dark mode, glassmorphism, dashboard interattiva, anteprima parametri in tempo reale
- **Archiviazione Intelligente e Cache**: cartelle dedicate `outputs/video_<ID>/` e controllo preventivo per riusare clip già esistenti
- **Clip Controls**: selezione numero di clip (1-10) e durata per clip (30-60 secondi)

## Requisiti

- Python 3.10+
- FFmpeg e ffprobe nel PATH
- yt-dlp
- OpenCV, MediaPipe, torch, torchaudio, openai-whisper, flask-socketio

## Installazione

```bash
# Clona la repository
git clone https://github.com/TUO-USERNAME/reelforge.git
cd reelforge

# Crea e attiva l'ambiente virtuale
python -m venv .venv
.\.venv\Scripts\activate        # Windows
source .venv/bin/activate       # Linux/Mac

# Installa le dipendenze
pip install -r requirements.txt
```

## Avvio

```bash
python app.py
```

Apri il browser su `http://127.0.0.1:5000`.

## Struttura

```
app.py                      # Applicazione Flask + Socket.IO
utils/
  video_processor.py        # Elaborazione video, download, taglio, FFmpeg
  smart_crop.py             # Smart crop dinamico con face tracking
  transcriber.py            # Trascrizione Whisper e punti di taglio intelligenti
  face_tracker.py           # (legacy) Face tracking OpenCV/MediaPipe
templates/
  index.html                # Interfaccia utente glassmorphism dark
uploads/                    # File temporanei upload/download
outputs/video_<ID>/         # Cartelle dedicate per ogni video YouTube
```

## Licenza

MIT
