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

import logging
import secrets
import time
import uuid
import os
from datetime import timedelta

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
from face_recognizer import FaceRecognizer, PEOPLE_DIR, MATCH_THRESHOLD
from auth import (
    check_password, login_required_page, login_required_api,
    is_login_rate_limited, record_failed_login, clear_login_attempts,
    get_csrf_token, csrf_token_valid, CSRF_HEADER_NAME, CSRF_FORM_FIELD,
    CSRF_SAFE_METHODS,
)

logging.basicConfig(
    level=logging.DEBUG if os.environ.get('FLASK_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
)
logger = logging.getLogger('face_ai')

# Separate audit trail (who logged in/out, enrolled/removed people, ran the
# camera) kept in its own file rather than mixed into the general console
# log, so it survives independently and stays easy to review.
os.makedirs('data', exist_ok=True)
audit_logger = logging.getLogger('face_ai.audit')
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False
_audit_handler = logging.FileHandler(os.path.join('data', 'audit.log'), encoding='utf-8')
_audit_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
audit_logger.addHandler(_audit_handler)


def audit(action: str, detail: str = ''):
    ip = request.remote_addr or 'unknown'
    audit_logger.info(f"{action} ip={ip} {detail}".rstrip())


app = Flask(__name__)

app.secret_key = os.environ.get('FACE_AI_SECRET_KEY')
if not app.secret_key:
    logger.warning(
        "FACE_AI_SECRET_KEY is not set. Using a randomly generated key for "
        "this run only, so sessions won't survive a restart. Set "
        "FACE_AI_SECRET_KEY in your .env for stable sessions."
    )
    app.secret_key = secrets.token_hex(32)

app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
# Set FACE_AI_HTTPS=1 once the app is served over TLS (e.g. behind a reverse
# proxy) so session cookies are never sent over plain HTTP.
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FACE_AI_HTTPS', '0') == '1'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB upload cap
# Sessions expire after this many hours of being issued, instead of lasting
# indefinitely as long as the browser stays open.
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(
    hours=float(os.environ.get('FACE_AI_SESSION_HOURS', '12'))
)

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


@app.context_processor
def inject_csrf_token():
    return {'csrf_token': get_csrf_token}


@app.before_request
def enforce_csrf():
    if request.method in CSRF_SAFE_METHODS:
        return None

    submitted = request.form.get(CSRF_FORM_FIELD) or request.headers.get(CSRF_HEADER_NAME)
    if not csrf_token_valid(submitted):
        if request.path == '/login':
            # No session yet on a first-ever visit; render the login page's
            # error state instead of a bare JSON 400.
            return render_template('login.html', error='Session expired, please try again', next=request.form.get('next', '')), 400
        return jsonify({'success': False, 'error': 'Invalid or missing CSRF token'}), 400


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
        client_ip = request.remote_addr or 'unknown'
        if is_login_rate_limited(client_ip):
            error = 'Too many attempts. Please wait a few minutes and try again.'
            return render_template('login.html', error=error, next=next_url), 429

        password = request.form.get('password', '')
        if check_password(password):
            clear_login_attempts(client_ip)
            session.clear()
            session['authenticated'] = True
            session.permanent = True
            audit('login_success')
            return redirect(next_url)

        record_failed_login(client_ip)
        audit('login_failed')
        error = 'Incorrect password'

    return render_template('login.html', error=error, next=next_url)


@app.route('/logout', methods=['POST'])
def logout():
    audit('logout')
    session.clear()
    return redirect(url_for('login'))


@app.route('/healthz')
def healthz():
    """Unauthenticated liveness check for uptime monitoring / container orchestration"""
    return jsonify({'status': 'ok'})


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


@app.route('/cameras')
@login_required_api
def list_cameras():
    """
    Probe a handful of device indices for available cameras. Skipped while
    a camera is already running to avoid fighting it for the device.
    """
    if video_analyzer.is_running:
        return jsonify({'success': True, 'cameras': [], 'busy': True})

    available = []
    for camera_id in range(4):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            available.append(camera_id)
        cap.release()

    return jsonify({'success': True, 'cameras': available, 'busy': False})


@app.route('/start_camera', methods=['POST'])
@login_required_api
def start_camera():
    """Start camera feed"""
    try:
        camera_id = request.json.get('camera_id', 0)
        success = video_analyzer.start_camera(camera_id)

        if success:
            audit('camera_start', f'camera_id={camera_id}')
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
        audit('camera_stop')
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
            'emotion_scores': face.get('emotion_scores', {}),
            'recognized_once': face.get('recognized_once', False),
            'liveness_status': face.get('liveness_status', 'checking'),
            'liveness_score': face.get('liveness_score'),
            'liveness_progress': face.get('liveness_progress', 0)
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

    response['fps'] = current_analysis.get('fps', 0)
    response['processing_ms'] = current_analysis.get('processing_ms', 0)
    response['liveness_summary'] = {
        'live': sum(1 for f in response['faces'] if f['liveness_status'] == 'live'),
        'low_motion': sum(1 for f in response['faces'] if f['liveness_status'] == 'low_motion'),
        'checking': sum(1 for f in response['faces'] if f['liveness_status'] == 'checking'),
    }

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


@app.route('/settings', methods=['GET'])
@login_required_api
def get_settings():
    """Current runtime-adjustable settings"""
    return jsonify({
        'success': True,
        'match_threshold': shared_recognizer.match_threshold,
        'default_match_threshold': MATCH_THRESHOLD,
    })


@app.route('/settings', methods=['POST'])
@login_required_api
def update_settings():
    """Adjust the recognition match-confidence threshold for this run (not persisted across restarts)"""
    try:
        threshold = float(request.json.get('match_threshold'))
    except (TypeError, ValueError, AttributeError):
        return jsonify({'success': False, 'error': 'match_threshold must be a number'}), 400

    # SFace's cosine similarity score is bounded to roughly this range in
    # practice; clamp rather than trust the client blindly.
    if not (0.0 <= threshold <= 1.0):
        return jsonify({'success': False, 'error': 'match_threshold must be between 0 and 1'}), 400

    shared_recognizer.match_threshold = threshold
    logger.info(f"Match threshold changed to {threshold:.3f}")
    return jsonify({'success': True, 'match_threshold': threshold})


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


@app.route('/media/<filename>', methods=['DELETE'])
@login_required_api
def delete_media(filename):
    """Delete a saved snapshot/analyzed image"""
    safe_filename = secure_filename(filename)
    # Only ever delete files this app itself generated into MEDIA_FOLDER —
    # same prefix check get_snapshots() uses to list them.
    if not (safe_filename.startswith('snapshot_') or safe_filename.startswith('analyzed_')):
        return jsonify({'success': False, 'error': 'Not a deletable file'}), 400

    filepath = os.path.join(MEDIA_FOLDER, safe_filename)
    if not os.path.isfile(filepath):
        return jsonify({'success': False, 'error': 'File not found'}), 404

    os.remove(filepath)
    return jsonify({'success': True})


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
        audit('person_enrolled', f"name={result['name']} photos={result['photos_enrolled']}")
        return jsonify({'success': True, **result})

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

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
        audit('person_deleted', f'name={name}')
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


@app.route('/people/<name>/photos', methods=['GET'])
@login_required_api
def list_person_photos(name):
    """List an enrolled person's individual photos, for the per-photo management UI"""
    try:
        return jsonify({'success': True, 'photos': shared_recognizer.get_photos(name)})
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/people/<name>/photos/<filename>', methods=['DELETE'])
@login_required_api
def delete_person_photo(name, filename):
    """Remove a single enrolled photo (and its embedding) from a person"""
    try:
        result = shared_recognizer.delete_photo(name, filename)
        if result is None:
            return jsonify({'success': False, 'error': 'Photo not found'}), 404
        audit('photo_deleted', f'name={name} person_removed={not result}')
        return jsonify({'success': True, 'person_removed': not result})
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    if not os.environ.get('FACE_AI_PASSWORD'):
        logger.warning(
            "FACE_AI_PASSWORD is not set — no password will work and login "
            "is impossible until you set it (see .env.example)."
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
