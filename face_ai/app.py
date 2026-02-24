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


