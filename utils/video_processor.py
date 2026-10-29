import os
import subprocess
import uuid
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .smart_crop import SmartCropper
from .transcriber import Transcriber


class VideoProcessor:
    def __init__(self, upload_dir='uploads', output_dir='outputs'):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.smart_cropper = SmartCropper()
        self.transcriber = Transcriber()

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
                "Suggerimento: installa EJS con `pip install ejs ytdlp-ejs` e Node.js con `winget install OpenJS.NodeJS.LTS`, "
                "oppure usa l'upload locale."
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

    def split_by_duration(self, input_path, clip_duration, num_clips=None, top_text="", bottom_text="", max_workers=1, output_folder=None, mode='bars', use_whisper=False, progress_callback=None):
        clip_duration = int(round(float(clip_duration)))
        if clip_duration < 30 or clip_duration > 60:
            raise Exception("Clip duration must be between 30 and 60 seconds")

        duration = self.get_video_duration(input_path)

        if num_clips is not None:
            num_clips = int(num_clips)
            if num_clips < 1:
                num_clips = 1
            if num_clips > 10:
                num_clips = 10
            segment_duration = clip_duration
            max_possible = int(duration // segment_duration)
            if max_possible > 0 and num_clips > max_possible:
                num_clips = max_possible
        else:
            segment_duration = clip_duration
            num_clips = int(duration // segment_duration)
            if num_clips < 1:
                num_clips = 1

        print(f'[VideoProcessor] Pianificati {num_clips} clip da {segment_duration}s ciascuna su video di {duration:.2f}s')

        target_folder = Path(output_folder) if output_folder else self.output_dir
        target_folder.mkdir(parents=True, exist_ok=True)

        results = []
        tasks = []

        start_times = None
        if use_whisper:
            try:
                print('[VideoProcessor] Avvio trascrizione Whisper...')
                segments = self.transcriber.transcribe(input_path)
                start_times = self.transcriber.get_smart_cut_points(segments, target_duration=segment_duration, num_clips=num_clips)
                print(f'[VideoProcessor] Cut points Whisper: {start_times}')
            except Exception as e:
                print(f'[VideoProcessor] Whisper fallito, uso tagli fissi: {e}')
                start_times = None

        for i in range(num_clips):
            if start_times and i < len(start_times):
                start_time = start_times[i]
            else:
                start_time = i * segment_duration
            actual_duration = segment_duration
            if start_time + actual_duration > duration:
                actual_duration = duration - start_time
            if actual_duration < 5:
                continue

            clip_id = str(uuid.uuid4())
            output_filename = f"clip_{i+1}_{clip_id}.mp4"
            tasks.append({
                'index': len(tasks) + 1,
                'start_time': start_time,
                'duration': int(actual_duration),
                'output_filename': output_filename
            })

        total_tasks = len(tasks)
        print(f'[VideoProcessor] Totale task da eseguire: {total_tasks}')

        for idx, task in enumerate(tasks):
            if progress_callback:
                try:
                    pct = int((idx / max(1, total_tasks)) * 100)
                    progress_callback(pct, f'Clip {task["index"]}/{total_tasks}: inizio conversione')
                except Exception:
                    pass

            print(f'[VideoProcessor] Inizio clip {task["index"]}/{total_tasks}: start={task["start_time"]}s dur={task["duration"]}s mode={mode}')

            try:
                output_path = self._convert_clip_segment(
                    input_path,
                    task['start_time'],
                    task['duration'],
                    task['output_filename'],
                    top_text,
                    bottom_text,
                    target_folder,
                    mode=mode,
                    use_smart_crop=(mode == 'smart_crop')
                )
                print(f'[VideoProcessor] Clip {task["index"]} completata: {output_path.name}')
                results.append({
                    'index': task['index'],
                    'success': True,
                    'download_url': f'/download/{target_folder.name}/{output_path.name}',
                    'filename': output_path.name,
                    'duration': task['duration']
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
        print(f'[VideoProcessor] Fine: {sum(1 for r in results if r["success"])}/{len(results)} clip riuscite')
        return results

    def _convert_clip_segment(self, input_path, start_time, duration, output_filename, top_text, bottom_text, target_folder=None, mode='bars', use_smart_crop=False):
        folder = Path(target_folder) if target_folder else self.output_dir
        folder.mkdir(parents=True, exist_ok=True)
        output_path = folder / output_filename

        print(f'[FFmpeg] Conversione clip: start={start_time}s dur={duration}s output={output_path.name} mode={mode}')

        if use_smart_crop or mode == 'smart_crop':
            try:
                import cv2
                use_smart_crop_mode = True
            except ImportError:
                print('[SmartCrop] OpenCV non installato, fallback a bande nere')
                use_smart_crop_mode = False

            if use_smart_crop_mode:
                try:
                    print('[SmartCrop] Campionamento frame e rilevamento soggetto...')
                    positions, vid_duration = self.smart_cropper.sample_positions(input_path)
                    w = int(cv2.VideoCapture(str(input_path)).get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cv2.VideoCapture(str(input_path)).get(cv2.CAP_PROP_FRAME_HEIGHT))
                    filter_complex = self.smart_cropper.generate_filter(positions, vid_duration, w, h)
                    if top_text and top_text.strip():
                        escaped_top = self._escape_text(top_text.strip())
                        filter_complex += f",drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{escaped_top}':fontcolor=white:fontsize=56:box=1:boxcolor=black@0.4:boxborderw=10:x=(w-text_w)/2:y=h*0.08:shadowcolor=black@0.9:shadowx=2:shadowy=2"
                    if bottom_text and bottom_text.strip():
                        escaped_bottom = self._escape_text(bottom_text.strip())
                        filter_complex += f",drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{escaped_bottom}':fontcolor=white:fontsize=56:box=1:boxcolor=black@0.4:boxborderw=10:wrap_width=w*0.9:x=(w-text_w)/2:y=h*0.88:shadowcolor=black@0.9:shadowx=2:shadowy=2"
                except Exception as e:
                    print(f'[SmartCrop] Errore, fallback a bande nere: {e}')
                    filter_complex = self._build_filter(top_text, bottom_text)
            else:
                filter_complex = self._build_filter(top_text, bottom_text)
        else:
            filter_complex = self._build_filter(top_text, bottom_text)

        print(f'[FFmpeg] Filter: {filter_complex}')

        cmd = [
            'ffmpeg',
            '-ss', str(float(start_time)),
            '-i', str(input_path),
            '-t', str(float(duration)),
            '-vf', filter_complex,
            '-threads', '2',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',
            '-y',
            str(output_path)
        ]

        print(f'[FFmpeg] Comando: {" ".join(cmd[:6])} ...')

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            print(f'[FFmpeg] Clip {output_path.name} completata con successo')
            return output_path
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode() if e.stderr else 'unknown'
            print(f'[FFmpeg] ERRORE conversione: {err}')
            raise Exception(f"FFmpeg conversion failed: {err}")
        except subprocess.TimeoutExpired:
            print(f'[FFmpeg] TIMEOUT conversione clip {output_path.name}')
            raise Exception("Conversion timeout")

    def _build_filter(self, top_text, bottom_text):
        base_filter = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        filters = [base_filter]

        if top_text and top_text.strip():
            escaped_top = self._escape_text(top_text.strip())
            top_drawtext = f"drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{escaped_top}':fontcolor=white:fontsize=56:box=0:x=(w-text_w)/2:y=(h*0.16)-(text_h/2):shadowcolor=black@0.9:shadowx=2:shadowy=2"
            filters.append(top_drawtext)

        if bottom_text and bottom_text.strip():
            escaped_bottom = self._escape_text(bottom_text.strip())
            bottom_drawtext = f"drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{escaped_bottom}':fontcolor=white:fontsize=56:box=0:wrap_width=w*0.9:x=(w-text_w)/2:y=(h*0.88)-(text_h/2):shadowcolor=black@0.9:shadowx=2:shadowy=2"
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
