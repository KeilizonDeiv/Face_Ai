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
