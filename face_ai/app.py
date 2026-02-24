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

