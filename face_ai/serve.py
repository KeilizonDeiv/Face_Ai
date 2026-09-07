"""
Production entrypoint: serves the Flask app with waitress instead of
Flask's development server (app.run in app.py), which isn't designed to
handle concurrent connections safely or efficiently.

Usage:
    python serve.py

Reads the same FACE_AI_HOST / FACE_AI_PORT env vars as app.py. If you're
putting this behind a TLS-terminating reverse proxy, also set
FACE_AI_HTTPS=1 so session cookies get the Secure flag.
"""

import os

from waitress import serve

from app import app, logger

if __name__ == '__main__':
    host = os.environ.get('FACE_AI_HOST', '127.0.0.1')
    port = int(os.environ.get('FACE_AI_PORT', 5000))

    if not os.environ.get('FACE_AI_PASSWORD'):
        logger.warning(
            "FACE_AI_PASSWORD is not set — no password will work and login "
            "is impossible until you set it (see .env.example)."
        )

    logger.info(f"Serving with waitress on http://{host}:{port}")
    serve(app, host=host, port=port, threads=8)
