"""
AI Face Analysis System - Flask Application
Real-time face detection, tracking, recognition, and emotion analysis web interface
"""

import sys

# Windows consoles often default to a legacy codepage (e.g. cp1252) that
# can't encode the checkmark/cross characters used in the log messages
# below; reconfigure to UTF-8 so startup doesn't crash on such consoles.
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import secrets
import time
import uuid
import os

from dotenv import load_dotenv

load_dotenv()

from flask import (
    Flask, render_template, Response, jsonify, request,
    session, redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename
import cv2
import numpy as np

from face_analyzer import FaceAnalyzer, VideoAnalyzer
from face_recognizer import FaceRecognizer, PEOPLE_DIR
from auth import check_password, login_required_page, login_required_api

app = Flask(__name__)

app.secret_key = os.environ.get('FACE_AI_SECRET_KEY')
if not app.secret_key:
    print(
        "WARNING: FACE_AI_SECRET_KEY is not set. Using a randomly generated "
        "key for this run only, so sessions won't survive a restart. Set "
        "FACE_AI_SECRET_KEY in your .env for stable sessions."
    )
    app.secret_key = secrets.token_hex(32)

app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB upload cap

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}


def has_allowed_extension(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


# One shared recognizer/embeddings-index instance so the live video pipeline
# and the image-upload/enrollment routes always see the same enrolled people.
shared_recognizer = FaceRecognizer()
video_analyzer = VideoAnalyzer(recognizer=shared_recognizer)
image_analyzer = FaceAnalyzer(recognizer=shared_recognizer)

# Saved snapshots/analyzed images live outside static/ so they're only
# reachable through the authenticated /media/<filename> route below.
MEDIA_FOLDER = os.path.join('data', 'media')
os.makedirs(MEDIA_FOLDER, exist_ok=True)


def generate_frames():
    """Generator function for video streaming"""
    while video_analyzer.is_running:
        frame, analysis = video_analyzer.get_frame()

        if frame is None:
            # Capture/inference threads haven't produced a frame yet
            time.sleep(0.03)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/login', methods=['GET', 'POST'])
def login():
    next_url = request.args.get('next') or request.form.get('next') or url_for('index')
    error = None

    if request.method == 'POST':
        password = request.form.get('password', '')
        if check_password(password):
            session.clear()
            session['authenticated'] = True
            return redirect(next_url)
        error = 'Incorrect password'

    return render_template('login.html', error=error, next=next_url)


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@login_required_page
def index():
    """Render main interface"""
    return render_template('index.html')


@app.route('/video_feed')
@login_required_api
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/start_camera', methods=['POST'])
@login_required_api
def start_camera():
    """Start camera feed"""
    try:
        camera_id = request.json.get('camera_id', 0)
        success = video_analyzer.start_camera(camera_id)

        if success:
            return jsonify({
                'success': True,
                'message': 'Camera started successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start camera'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/stop_camera', methods=['POST'])
@login_required_api
def stop_camera():
    """Stop camera feed"""
    try:
        video_analyzer.stop_camera()
        return jsonify({
            'success': True,
            'message': 'Camera stopped'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/get_analysis')
@login_required_api
def get_analysis():
    """Get current analysis data"""
    current_analysis = video_analyzer.get_latest_analysis()

    response = {
        'num_faces': current_analysis.get('num_faces', 0),
        'faces': []
    }

    for face in current_analysis.get('faces', []):
        response['faces'].append({
            'track_id': face.get('track_id'),
            'name': face.get('name'),
            'match_score': face.get('match_score', 0),
            'dominant_emotion': face.get('dominant_emotion', 'neutral'),
            'confidence': face.get('confidence', 0),
            'emotion_scores': face.get('emotion_scores', {})
        })

    # Calculate overall emotion distribution
    if response['num_faces'] > 0:
        emotion_dist = {}
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
            total = sum(face['emotion_scores'].get(emotion, 0) for face in response['faces'])
            emotion_dist[emotion] = total / response['num_faces']

        response['emotion_distribution'] = emotion_dist
    else:
        response['emotion_distribution'] = {e: 0 for e in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}

    return jsonify(response)


@app.route('/capture_snapshot', methods=['POST'])
@login_required_api
def capture_snapshot():
    """Capture and save current frame"""
    try:
        frame, analysis = video_analyzer.get_frame()

        if frame is None:
            return jsonify({
                'success': False,
                'error': 'No frame available'
            }), 400

        filename = f'snapshot_{uuid.uuid4().hex}.jpg'
        filepath = os.path.join(MEDIA_FOLDER, filename)

        success = video_analyzer.save_snapshot(frame, filepath)

        if success:
            return jsonify({
                'success': True,
                'filename': filename,
                'path': f'/media/{filename}',
                'analysis': {
                    'num_faces': analysis['num_faces'],
                    'faces': [
                        {
                            'name': face.get('name'),
                            'emotion': face['dominant_emotion'],
                            'confidence': face['confidence']
                        }
                        for face in analysis.get('faces', [])
                    ]
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save snapshot'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/analyze_image', methods=['POST'])
@login_required_api
def analyze_image():
    """Analyze uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not has_allowed_extension(file.filename):
            return jsonify({
                'success': False,
                'error': 'Unsupported file type (use jpg, png, or webp)'
            }), 400

        # Read image — the decode check below is the real content
        # validation; an extension check alone is trivially spoofable.
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image file'
            }), 400

        # Analyze image (shared analyzer instance — avoids reloading the
        # detection/recognition models on every request)
        analysis = image_analyzer.analyze_image(image)

        # Draw annotations
        annotated_image = image_analyzer.draw_analysis(image, analysis)

        # Save result
        filename = f'analyzed_{uuid.uuid4().hex}.jpg'
        filepath = os.path.join(MEDIA_FOLDER, filename)
        cv2.imwrite(filepath, annotated_image)

        return jsonify({
            'success': True,
            'result_path': f'/media/{filename}',
            'analysis': {
                'num_faces': analysis['num_faces'],
                'faces': [
                    {
                        'face_id': face['face_id'],
                        'name': face.get('name'),
                        'match_score': face.get('match_score', 0),
                        'emotion': face['dominant_emotion'],
                        'confidence': face['confidence'],
                        'emotion_scores': face['emotion_scores']
                    }
                    for face in analysis.get('faces', [])
                ]
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/camera_status')
@login_required_api
def camera_status():
    """Get camera status"""
    return jsonify({
        'is_running': video_analyzer.is_running
    })


@app.route('/get_snapshots')
@login_required_api
def get_snapshots():
    """Get list of saved snapshots"""
    try:
        files = os.listdir(MEDIA_FOLDER)
        snapshots = [f for f in files if f.startswith('snapshot_') or f.startswith('analyzed_')]
        snapshots.sort(
            key=lambda f: os.path.getmtime(os.path.join(MEDIA_FOLDER, f)),
            reverse=True
        )

        return jsonify({
            'success': True,
            'snapshots': [
                {
                    'filename': f,
                    'path': f'/media/{f}',
                }
                for f in snapshots[:10]  # Last 10 snapshots
            ]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/media/<filename>')
@login_required_api
def media(filename):
    """Serve a saved snapshot/analyzed image (gated behind login)"""
    safe_filename = secure_filename(filename)
    return send_from_directory(MEDIA_FOLDER, safe_filename)


@app.route('/people', methods=['GET'])
@login_required_api
def list_people():
    """List enrolled people"""
    try:
        return jsonify({
            'success': True,
            'people': shared_recognizer.list_people()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/people', methods=['POST'])
@login_required_api
def add_person():
    """Enroll a person from one or more uploaded photos ('name' is the training label)"""
    try:
        name = request.form.get('name', '')
        photo_files = request.files.getlist('photos')

        if not name.strip():
            return jsonify({'success': False, 'error': 'Name is required'}), 400

        if not photo_files:
            return jsonify({'success': False, 'error': 'At least one photo is required'}), 400

        decoded_photos = []
        for photo_file in photo_files:
            if not photo_file.filename or not has_allowed_extension(photo_file.filename):
                continue
            file_bytes = np.frombuffer(photo_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                decoded_photos.append(image)

        if not decoded_photos:
            return jsonify({'success': False, 'error': 'No valid image files provided'}), 400

        result = shared_recognizer.enroll_person(name, decoded_photos)
        return jsonify({'success': True, **result})

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/people/<name>', methods=['DELETE'])
@login_required_api
def delete_person(name):
    """Remove an enrolled person and their stored embeddings/photos"""
    try:
        existed = shared_recognizer.delete_person(name)
        if not existed:
            return jsonify({'success': False, 'error': 'Person not found'}), 404
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/people/<name>/photo/<filename>')
@login_required_api
def person_photo(name, filename):
    """Serve an enrolled person's thumbnail photo"""
    safe_name = secure_filename(name)
    safe_filename = secure_filename(filename)
    person_dir = os.path.join(PEOPLE_DIR, safe_name)

    return send_from_directory(person_dir, safe_filename)


if __name__ == '__main__':
    if not os.environ.get('FACE_AI_PASSWORD'):
        print(
            "WARNING: FACE_AI_PASSWORD is not set — no password will work "
            "and login is impossible until you set it (see .env.example)."
        )

    host = os.environ.get('FACE_AI_HOST', '127.0.0.1')
    port = int(os.environ.get('FACE_AI_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'

    print("=" * 60)
    print("AI Face Analysis System")
    print("=" * 60)
    print("Features:")
    print("  - Real-time face detection (YuNet)")
    print("  - Face tracking across frames (IOU tracker)")
    print("  - Face recognition / identification (SFace)")
    print("  - Emotion analysis (7 emotions)")
    print("  - Webcam support")
    print("  - 100% Open-source & Offline")
    print("=" * 60)
    print("\nStarting server...")
    print(f"Open http://{host}:{port} in your browser")
    print("=" * 60 + "\n")

    app.run(debug=debug, host=host, port=port, threaded=True)
