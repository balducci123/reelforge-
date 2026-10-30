import os
import subprocess
import uuid
import json
import re
from pathlib import Path
from .smart_crop import ViralSmartCropper
from .transcriber import Transcriber
from .audio_analyzer import AudioAnalyzer
from .clip_scorer import ClipScorer


class VideoProcessor:
    def __init__(self, upload_dir='uploads', output_dir='outputs'):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.smart_cropper = ViralSmartCropper()
        self.transcriber = Transcriber(model_size="base", device="cpu", compute_type="int8")
        self.audio_analyzer = AudioAnalyzer()
        self.clip_scorer = ClipScorer()

    # ------------------------------------------------------------------ #
    # YouTube ID & folder management
    # ------------------------------------------------------------------ #
    def get_youtube_id(self, youtube_url):
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        return str(uuid.uuid4())[:11]

    def get_video_folder(self, youtube_url):
        video_id = self.get_youtube_id(youtube_url)
        folder = self.output_dir / f"video_{video_id}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder, video_id

    # ------------------------------------------------------------------ #
    # Duration
    # ------------------------------------------------------------------ #
    def get_video_duration(self, video_path):
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            print(f'[VideoProcessor] Durata video: {duration:.2f}s')
            return duration
        except Exception as e:
            raise Exception(f"Failed to get video duration: {str(e)}")

    def is_vertical(self, video_path):
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', str(video_path)
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, timeout=15)
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    w = stream.get('width', 0)
                    h = stream.get('height', 0)
                    if w and h and h > w:
                        return True
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #
    def download_video(self, youtube_url, proxy_url=None, local_video_path=None):
        video_id = str(uuid.uuid4())
        output_template = self.upload_dir / f"{video_id}.%(ext)s"

        if local_video_path:
            local_file = Path(local_video_path)
            if local_file.exists():
                print(f'[yt-dlp] Uso file locale: {local_file.name}')
                return local_file, local_file.stem

        base_args = ['yt-dlp', '--no-playlist', '--restrict-filenames', '-o', str(output_template), youtube_url]
        proxy_args = ['--proxy', proxy_url] if proxy_url else []

        attempts = [
            base_args + ['--js-runtimes', 'node', '--remote-components', 'ejs:github'] + proxy_args,
            base_args + proxy_args,
        ]

        last_error = None
        for attempt_index, cmd in enumerate(attempts, 1):
            print(f'[yt-dlp] Attempt {attempt_index}/{len(attempts)}: {" ".join(cmd[:5])} ...')
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=300)
                last_error = None
                break
            except subprocess.CalledProcessError as e:
                last_error = f"yt-dlp failed: {e.stderr.decode() if e.stderr else 'unknown'}"
                print(f'[yt-dlp] Attempt fallito: {last_error}')
                continue
            except subprocess.TimeoutExpired:
                raise Exception("Download timeout")
            except FileNotFoundError as e:
                raise Exception(f"yt-dlp non trovato in PATH. pip install yt-dlp. Errore: {e}")

        if last_error:
            raise Exception(
                "Failed to download video. "
                f"Last error: {last_error}. "
                "Suggerimento: installa EJS con `pip install ejs ytdlp-ejs` e Node.js, "
                "oppure carica un file video locale."
            )

        for file in self.upload_dir.glob(f"{video_id}.*"):
            if file.suffix in ['.mp4', '.mkv', '.webm']:
                if file.suffix != '.mp4':
                    mp4_file = file.with_suffix('.mp4')
                    conv_cmd = ['ffmpeg', '-i', str(file), '-c', 'copy', '-y', str(mp4_file)]
                    subprocess.run(conv_cmd, check=True, capture_output=True)
                    self.cleanup(file)
                    return mp4_file, video_id
                return file, video_id

        raise Exception("Downloaded file not found")

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    def check_cached_clips(self, video_folder, num_clips, clip_duration):
        if not video_folder.exists():
            return []
        clips = []
        for i in range(1, num_clips + 1):
            pattern = f"clip_{i}_*.mp4"
            matches = list(video_folder.glob(pattern))
            if not matches:
                return []
            clips.append(matches[0])
        return clips

    # ------------------------------------------------------------------ #
    # ASS Subtitle generator (karaoke parola per parola)
    # ------------------------------------------------------------------ #
    def generate_ass_subtitles(self, words, output_path, style_name='Default'):
        if not words:
            output_path.write_text(
                '[Script Info]\nTitle: ReelForge\nScriptType: v4.00+\n'
                '[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, '
                'OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, '
                'Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, '
                'MarginV, Encoding\n'
                'Style: Default,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,'
                '100,100,0,0,1,3,0,2,10,10,30,1\n'
                '[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, '
                'Effect, Text\n',
                encoding='utf-8'
            )
            return output_path

        events = []
        for w in words:
            start = self._format_ass_time(w['start'])
            end = self._format_ass_time(w['end'])
            word_text = w['word'].replace('{', '\\{').replace('}', '\\}')
            events.append(
                f"Dialogue: 0,{start},{end},{style_name},,0,0,0,,"
                f"{{\\c&H00E1FF&\\b1\\fs66\\3c&H000000&\\4c&H000000&\\3a&H60&\\4a&H60&}}"
                f"{word_text}"
            )

        ass_content = (
            '[Script Info]\nTitle: ReelForge Karaoke\nScriptType: v4.00+\n'
            '[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, '
            'OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, '
            'Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, '
            'MarginV, Encoding\n'
            'Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,'
            '100,100,0,0,1,2,0,2,10,10,40,1\n'
            '[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, '
            'Effect, Text\n'
            + '\n'.join(events)
            + '\n'
        )
        output_path.write_text(ass_content, encoding='utf-8')
        print(f'[Subtitles] Generato ASS con {len(words)} eventi karaoke')
        return output_path

    def _format_ass_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds - int(seconds)) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    # ------------------------------------------------------------------ #
    # Smart split con Whisper + Audio Analyzer + Clip Scorer
    # ------------------------------------------------------------------ #
    def split_by_duration(self, input_path, clip_duration, num_clips=None, top_text="", bottom_text="",
                          max_workers=1, output_folder=None, crop_strategy='smart_crop',
                          use_whisper=False, use_karaoke=False,
                          progress_callback=None, auto_select=True, min_score=60):
        clip_duration = int(round(float(clip_duration)))
        if clip_duration < 30 or clip_duration > 60:
            raise Exception("Clip duration must be between 30 and 60 seconds")

        duration = self.get_video_duration(input_path)
        is_vid_vertical = self.is_vertical(input_path)

        if num_clips is not None:
            num_clips = int(num_clips)
            if num_clips < 1:
                num_clips = 1
            if num_clips > 10:
                num_clips = 10

        print(f'[VideoProcessor] Video verticale: {is_vid_vertical} | Durata: {duration:.2f}s | '
              f'Richeste: {num_clips} clip da {clip_duration}s | crop={crop_strategy}')

        target_folder = Path(output_folder) if output_folder else self.output_dir
        target_folder.mkdir(parents=True, exist_ok=True)

        # --- Fase 1: Analisi audio ---
        audio_analysis = self.audio_analyzer.analyze(input_path)
        print(f'[AudioAnalyzer] Picchi: {audio_analysis["peak_times"][:5]}... | Silenzi: {len(audio_analysis["silent_ranges"])}')

        # --- Fase 2: Candidati iniziali ---
        start_times = None
        all_words = []
        if use_whisper:
            try:
                print('[VideoProcessor] Avvio trascrizione Whisper...')
                words = self.transcriber.transcribe_words(input_path)
                all_words = words
                segments = self.transcriber.get_smart_segments(
                    words, min_clip_duration=clip_duration, max_clip_duration=clip_duration
                )
                start_times = [s['start'] for s in segments]
                print(f'[VideoProcessor] Cut points Whisper: {start_times[:num_clips]}')
            except Exception as e:
                print(f'[VideoProcessor] Whisper fallito, uso audio analyzer: {e}')
                start_times = None

        if not start_times or len(start_times) < num_clips:
            auto_cuts = self.audio_analyzer.find_best_cut_points(
                audio_analysis, num_clips=num_clips,
                min_duration=clip_duration, max_duration=clip_duration
            )
            start_times = [c['start'] for c in auto_cuts]
            print(f'[AudioAnalyzer] Cut points automatici: {start_times[:num_clips]}')

        while len(start_times) < num_clips:
            s = len(start_times) * clip_duration
            if s >= duration:
                break
            start_times.append(round(s, 2))

        # --- Fase 3: Costruisci task ---
        raw_tasks = []
        for i in range(num_clips):
            start_time = start_times[i] if i < len(start_times) else i * clip_duration
            actual_duration = clip_duration
            if start_time + actual_duration > duration:
                actual_duration = duration - start_time
            if actual_duration < 5:
                continue

            clip_id = str(uuid.uuid4())
            output_filename = f"clip_{i+1}_{clip_id}.mp4"
            clip_words = [w for w in all_words if start_time <= w['start'] and w['end'] <= start_time + actual_duration]
            clip_text = ' '.join(w['word'] for w in clip_words)

            # Se karaoke attivo ma nessuna parola (Whisper fallito/disattivato),
            # genera sottotitoli temporali vuoti ogni 2 secondi
            if use_karaoke and not clip_words:
                t = start_time
                while t < start_time + actual_duration:
                    clip_words.append({
                        'word': '',
                        'start': round(t, 2),
                        'end': round(min(t + 2.0, start_time + actual_duration), 2),
                        'probability': 1.0
                    })
                    t += 2.0

            raw_tasks.append({
                'index': len(raw_tasks) + 1,
                'start_time': round(start_time, 2),
                'duration': int(actual_duration),
                'output_filename': output_filename,
                'words': clip_words,
                'text': clip_text,
                'has_hook': any(hw in clip_text.lower() for hw in self.transcriber.HOOK_WORDS)
            })

        # Applica Clip Scorer
        if auto_select and raw_tasks:
            scored = self.clip_scorer.rank_candidates(
                raw_tasks, audio_analysis=audio_analysis,
                face_present=True, top_n=num_clips
            )
            tasks = []
            for new_idx, c in enumerate(scored, 1):
                c['index'] = new_idx
                tasks.append(c)
        else:
            tasks = raw_tasks

        total_tasks = len(tasks)
        print(f'[VideoProcessor] Totale task da eseguire: {total_tasks}')

        # --- Fase 4: Esecuzione ---
        results = []
        for idx, task in enumerate(tasks):
            if progress_callback:
                try:
                    pct = int((idx / max(1, total_tasks)) * 100)
                    progress_callback(pct, f'Clip {task["index"]}/{total_tasks}: conversione')
                except Exception:
                    pass

            print(f'[VideoProcessor] Clip {task["index"]}/{total_tasks}: '
                  f'start={task["start_time"]}s dur={task["duration"]}s '
                  f'crop={crop_strategy} karaoke={use_karaoke}')

            try:
                output_path = self._convert_clip_segment_viral(
                    input_path,
                    task['start_time'],
                    task['duration'],
                    task['output_filename'],
                    top_text,
                    bottom_text,
                    target_folder,
                    crop_strategy=crop_strategy,
                    use_karaoke=use_karaoke,
                    words=task['words'],
                    is_vertical=is_vid_vertical,
                    clip_start=task['start_time'],
                    clip_end=task['start_time'] + task['duration']
                )
                print(f'[VideoProcessor] Clip {task["index"]} completata: {output_path.name}')
                results.append({
                    'index': task['index'],
                    'success': True,
                    'download_url': f'/download/{target_folder.name}/{output_path.name}',
                    'filename': output_path.name,
                    'duration': task['duration'],
                    'cached': False,
                    'score': task.get('score', {}),
                    'text': task['text'][:80]
                })
            except Exception as e:
                print(f'[VideoProcessor] Clip {task["index"]} FALLITA: {e}')
                results.append({
                    'index': task['index'],
                    'success': False,
                    'error': str(e)
                })

        if progress_callback:
            try:
                progress_callback(100, 'Completato')
            except Exception:
                pass

        results.sort(key=lambda x: x['index'])
        successful = sum(1 for r in results if r['success'])
        print(f'[VideoProcessor] Fine: {successful}/{len(results)} clip riuscite')
        return results

    # ------------------------------------------------------------------ #
    # Conversione singola clip
    # ------------------------------------------------------------------ #
    def _convert_clip_segment_viral(self, input_path, start_time, duration, output_filename,
                                    top_text, bottom_text, target_folder=None,
                                    crop_strategy='smart_crop',
                                    use_karaoke=False, words=None, is_vertical=False,
                                    clip_start=None, clip_end=None):
        folder = Path(target_folder) if target_folder else self.output_dir
        folder.mkdir(parents=True, exist_ok=True)
        output_path = folder / output_filename

        print(f'[FFmpeg] Clip: start={start_time}s dur={duration}s output={output_path.name} '
              f'crop={crop_strategy} karaoke={use_karaoke} vertical_input={is_vertical}')

        # --- Filtro video ---
        video_filter_parts = []

        if crop_strategy == 'smart_crop':
            try:
                # CAMPIONA SOLO IL SEGMENTO DEL CLIP, non tutto il video
                positions, vid_duration, w, h = self.smart_cropper.sample_subject_positions(
                    input_path, start_time=start_time, end_time=(clip_end or start_time + duration)
                )
                crop_filter = self.smart_cropper.generate_filter(
                    positions, vid_duration, w, h, strategy='smart_crop'
                )
                video_filter_parts.append(crop_filter)
                print(f'[SmartCrop] Filtro applicato: {crop_filter}')
            except Exception as e:
                print(f'[SmartCrop] Errore, fallback a bande nere: {e}')
                video_filter_parts.append(self._build_filter(top_text, bottom_text))

        elif crop_strategy == 'blur_background':
            try:
                positions, vid_duration, w, h = self.smart_cropper.sample_subject_positions(
                    input_path, start_time=start_time, end_time=(clip_end or start_time + duration)
                )
                blur_filter = self.smart_cropper.generate_filter(
                    positions, vid_duration, w, h, strategy='blur_background'
                )
                video_filter_parts.append(blur_filter)
                print(f'[BlurBG] Filtro applicato: {blur_filter}')
            except Exception as e:
                print(f'[BlurBG] Errore, fallback a bande nere: {e}')
                video_filter_parts.append(self._build_filter(top_text, bottom_text))

        else:
            video_filter_parts.append(self._build_filter(top_text, bottom_text))
            print(f'[Bars] Filtro bande nere applicato')

        # --- Sottotitoli karaoke (ASS) ---
        ass_path = None
        if use_karaoke and words:
            ass_filename = output_path.stem + '.ass'
            ass_path = folder / ass_filename
            self.generate_ass_subtitles(words, ass_path)
            ass_path_str = str(ass_path).replace('\\', '/')
            video_filter_parts.append(f"ass='{ass_path_str}'")
            print(f'[Subtitles] Aggiunto overlay ASS: {ass_path_str}')

        filter_complex = ','.join(video_filter_parts)
        print(f'[FFmpeg] Filter completo: {filter_complex}')

        # --- Filtro audio: normalizzazione loudnorm ---
        audio_filter = "loudnorm=I=-16:TP=-1.5:LRA=11"

        cmd = [
            'ffmpeg',
            '-ss', str(float(start_time)),
            '-i', str(input_path),
            '-t', str(float(duration)),
            '-vf', filter_complex,
            '-af', audio_filter,
            '-threads', '2',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '20',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', '44100',
            '-ac', '2',
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',
            '-y',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            print(f'[FFmpeg] Clip {output_path.name} completata con successo')
            if ass_path and ass_path.exists():
                try:
                    ass_path.unlink()
                except Exception:
                    pass
            return output_path
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode() if e.stderr else 'unknown'
            print(f'[FFmpeg] ERRORE conversione: {err}')
            raise Exception(f"FFmpeg conversion failed: {err}")
        except subprocess.TimeoutExpired:
            print(f'[FFmpeg] TIMEOUT conversione clip {output_path.name}')
            raise Exception("Conversion timeout")

    # ------------------------------------------------------------------ #
    # Filtro bande nere classico
    # ------------------------------------------------------------------ #
    def _build_filter(self, top_text, bottom_text):
        base_filter = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        filters = [base_filter]

        if top_text and top_text.strip():
            escaped_top = self._escape_text(top_text.strip())
            top_drawtext = (
                f"drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{escaped_top}':"
                f"fontcolor=white:fontsize=56:box=1:boxcolor=black@0.4:boxborderw=10:"
                f"x=(w-text_w)/2:y=(h*0.16)-(text_h/2):shadowcolor=black@0.9:shadowx=2:shadowy=2"
            )
            filters.append(top_drawtext)

        if bottom_text and bottom_text.strip():
            escaped_bottom = self._escape_text(bottom_text.strip())
            bottom_drawtext = (
                f"drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{escaped_bottom}':"
                f"fontcolor=white:fontsize=56:box=1:boxcolor=black@0.4:boxborderw=10:"
                f"wrap_width=w*0.9:x=(w-text_w)/2:y=h*0.88:shadowcolor=black@0.9:shadowx=2:shadowy=2"
            )
            filters.append(bottom_drawtext)

        return ','.join(filters)

    def _escape_text(self, text):
        text = str(text)
        text = text.replace('\\', '\\\\')
        text = text.replace(':', '\\:')
        text = text.replace("'", "\\'")
        text = text.replace('%', '\\%')
        text = text.replace(',', '\\,')
        text = text.replace(';', '\\;')
        return text

    def cleanup(self, file_path):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
