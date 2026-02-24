"""
AI Face Analysis System - Flask Application
Real-time face detection and emotion analysis web interface
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from face_analyzer import FaceAnalyzer, VideoAnalyzer
import json
from datetime import datetime
import os
import base64

app = Flask(__name__)

# Global video analyzer
video_analyzer = VideoAnalyzer()
current_analysis = {'num_faces': 0, 'faces': []}

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def generate_frames():
    """Generator function for video streaming"""
    while True:
        frame, analysis = video_analyzer.get_frame()
        
        if frame is None:
            break
        
        # Update global analysis
        global current_analysis
        current_analysis = analysis
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        
        if not ret:
            break
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Render main interface"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/start_camera', methods=['POST'])
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
def get_analysis():
    """Get current analysis data"""
    global current_analysis
    
    # Format response
    response = {
        'num_faces': current_analysis['num_faces'],
        'faces': []
    }
    
    for face in current_analysis.get('faces', []):
        response['faces'].append({
            'face_id': face.get('face_id', 0),
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
def capture_snapshot():
    """Capture and save current frame"""
    try:
        frame, analysis = video_analyzer.get_frame()
        
        if frame is None:
            return jsonify({
                'success': False,
                'error': 'No frame available'
            }), 400
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'snapshot_{timestamp}.jpg'
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        # Save frame
        success = video_analyzer.save_snapshot(frame, filepath)
        
        if success:
            return jsonify({
                'success': True,
                'filename': filename,
                'path': f'/static/images/{filename}',
                'analysis': {
                    'num_faces': analysis['num_faces'],
                    'faces': [
                        {
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
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image file'
            }), 400
        
        # Analyze image
        face_analyzer = FaceAnalyzer()
        analysis = face_analyzer.analyze_image(image)
        
        # Draw annotations
        annotated_image = face_analyzer.draw_analysis(image, analysis)
        
        # Save result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'analyzed_{timestamp}.jpg'
        filepath = os.path.join(RESULTS_FOLDER, filename)
        cv2.imwrite(filepath, annotated_image)
        
        return jsonify({
            'success': True,
            'result_path': f'/static/images/{filename}',
            'analysis': {
                'num_faces': analysis['num_faces'],
                'faces': [
                    {
                        'face_id': face['face_id'],
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
def camera_status():
    """Get camera status"""
    return jsonify({
        'is_running': video_analyzer.is_running
    })


@app.route('/get_snapshots')
def get_snapshots():
    """Get list of saved snapshots"""
    try:
        files = os.listdir(RESULTS_FOLDER)
        snapshots = [f for f in files if f.startswith('snapshot_') or f.startswith('analyzed_')]
        snapshots.sort(reverse=True)
        
        return jsonify({
            'success': True,
            'snapshots': [
                {
                    'filename': f,
                    'path': f'/static/images/{f}',
                    'timestamp': f.split('_')[1].split('.')[0] if '_' in f else ''
                }
                for f in snapshots[:10]  # Last 10 snapshots
            ]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("="*60)
    print("AI Face Analysis System")
    print("="*60)
    print("Features:")
    print("  ✓ Real-time face detection")
    print("  ✓ Emotion analysis (7 emotions)")
    print("  ✓ Multiple face tracking")
    print("  ✓ Webcam support")
    print("  ✓ 100% Open-source & Offline")
    print("="*60)
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
