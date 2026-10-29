import os
import re
from flask import Flask, render_template, request, jsonify, send_file
from utils.video_processor import VideoProcessor
from pathlib import Path
from threading import Thread
from flask_socketio import SocketIO, emit
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'reelforge-secret'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
processor = VideoProcessor()
jobs = {}


@app.route('/')
def index():
    return render_template('index.html')


def process_job(job_id, data):
    try:
        youtube_url = data.get('youtube_url', '').strip()
        top_text = data.get('top_text', '').strip()
        bottom_text = data.get('bottom_text', '').strip()
        clip_duration = int(float(data.get('clip_duration') or 30))
        num_clips = data.get('num_clips')
        crop_strategy = data.get('crop_strategy', 'smart_crop')  # smart_crop | blur_background | bars
        use_whisper = bool(data.get('use_whisper', False))
        use_karaoke = bool(data.get('use_karaoke', False))
        auto_select = bool(data.get('auto_select', True))
        min_score = int(data.get('min_score', 60))
        proxy_url = data.get('proxy_url', '').strip()
        local_video_path = data.get('local_video_path')

        print(f'[ReelForge][{job_id}] Avvio: {youtube_url or local_video_path} | '
              f'clip={clip_duration}s n={num_clips} crop={crop_strategy} '
              f'whisper={use_whisper} karaoke={use_karaoke} auto_select={auto_select} min_score={min_score}')

        if num_clips is not None:
            num_clips = int(num_clips)
        else:
            num_clips = 3

        video_folder, video_id = processor.get_video_folder(youtube_url) if youtube_url else processor.get_video_folder(str(Path(local_video_path).stem))
        print(f'[ReelForge][{job_id}] Cartella: {video_folder.name}')

        cached = processor.check_cached_clips(video_folder, num_clips, clip_duration)
        if cached:
            print(f'[ReelForge][{job_id}] Cache trovata: {len(cached)} clip')
            results = []
            for i, clip_path in enumerate(cached, 1):
                results.append({
                    'index': i, 'success': True,
                    'download_url': f'/download/{video_folder.name}/{clip_path.name}',
                    'filename': clip_path.name, 'duration': clip_duration, 'cached': True
                })
            socketio.emit('job_complete', {'job_id': job_id, 'success': True, 'clips': results, 'cached': True}, namespace='/')
            return

        socketio.emit('job_progress', {'job_id': job_id, 'progress': 10, 'status': 'Downloading video...' if youtube_url else 'Loading local video...'}, namespace='/')
        print(f'[ReelForge][{job_id}] {"Download video da YouTube..." if youtube_url else "Usando video locale..."}')

        downloaded_file, video_id = processor.download_video(youtube_url, proxy_url=proxy_url, local_video_path=local_video_path)
        print(f'[ReelForge][{job_id}] Video pronto: {downloaded_file.name}')

        socketio.emit('job_progress', {'job_id': job_id, 'progress': 30, 'status': 'Processing clips...'}, namespace='/')

        def progress_cb(pct, status):
            socket_pct = 30 + int(pct * 0.7)
            print(f'[ReelForge][{job_id}] Progress: {socket_pct}% - {status}')
            socketio.emit('job_progress', {'job_id': job_id, 'progress': socket_pct, 'status': status}, namespace='/')

        results = processor.split_by_duration(
            downloaded_file,
            clip_duration=clip_duration, num_clips=num_clips,
            top_text=top_text, bottom_text=bottom_text,
            max_workers=1, output_folder=video_folder,
            crop_strategy=crop_strategy,
            use_whisper=use_whisper,
            use_karaoke=use_karaoke,
            auto_select=auto_select,
            min_score=min_score,
            progress_callback=progress_cb
        )

        processor.cleanup(downloaded_file)
        successful = sum(1 for r in results if r['success'])
        print(f'[ReelForge][{job_id}] Completato: {successful}/{len(results)} clip')
        socketio.emit('job_complete', {'job_id': job_id, 'success': True, 'clips': results, 'cached': False}, namespace='/')

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'[ReelForge][{job_id}] ERRORE: {e}')
        socketio.emit('job_error', {'job_id': job_id, 'error': str(e)}, namespace='/')


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        file = request.files['video']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        safe_name = re.sub(r'[^\w\-]', '_', file.filename)[:60] or 'video.mp4'
        target = processor.upload_dir / safe_name
        file.save(target)
        print(f'[ReelForge] Video locale salvato: {target.name}')
        return jsonify({'success': True, 'path': str(target), 'filename': target.name})
    except Exception as e:
        print(f'[ReelForge] Errore upload: {e}')
        return jsonify({'error': str(e)}), 500


@socketio.on('start_processing')
def on_start_processing(data):
    try:
        data = data or {}
        youtube_url = (data.get('youtube_url') or '').strip()
        local_video_path = data.get('local_video_path')
        if not youtube_url and not local_video_path:
            emit('job_error', {'error': 'YouTube URL or local video path is required'})
            return

        clip_duration = int(float(data.get('clip_duration') or 30))
        if clip_duration < 30 or clip_duration > 60:
            emit('job_error', {'error': 'Clip duration must be between 30 and 60 seconds'})
            return

        num_clips = data.get('num_clips')
        if num_clips is not None:
            num_clips = int(num_clips)
            if num_clips < 1 or num_clips > 10:
                emit('job_error', {'error': 'Number of clips must be between 1 and 10'})
                return
        else:
            num_clips = 3
        data['num_clips'] = num_clips

        if youtube_url:
            video_folder, _ = processor.get_video_folder(youtube_url)
            cached = processor.check_cached_clips(video_folder, num_clips, clip_duration)
            if cached:
                results = []
                for i, clip_path in enumerate(cached, 1):
                    results.append({
                        'index': i, 'success': True,
                        'download_url': f'/download/{video_folder.name}/{clip_path.name}',
                        'filename': clip_path.name, 'duration': clip_duration, 'cached': True
                    })
                emit('job_complete', {'success': True, 'clips': results, 'cached': True})
                return

        job_id = str(uuid.uuid4())
        jobs[job_id] = data
        thread = Thread(target=process_job, args=(job_id, data))
        thread.daemon = True
        thread.start()
        print(f'[ReelForge] Socket.IO start_processing -> job {job_id} lanciato')
    except Exception as e:
        import traceback
        traceback.print_exc()
        emit('job_error', {'error': str(e)})


@app.route('/convert', methods=['POST'])
def convert():
    try:
        data = request.get_json(silent=True) or {}
        on_start_processing(data)
        return jsonify({'success': True, 'async': True, 'note': 'Use Socket.IO start_processing instead'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<folder>/<filename>')
@app.route('/download/<filename>')
def download(folder=None, filename=None):
    try:
        if folder and filename:
            file_path = processor.output_dir / folder / filename
        else:
            file_path = processor.output_dir / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        return send_file(file_path, as_attachment=True, download_name=f'ReelForge_{filename}')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    emit('connected', {'status': 'ok'})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
