"""
Minimal session-based auth: a single shared password gates the whole app.

Not a multi-user system by design (see plan) — one password, checked with a
constant-time comparison, stored server-side only as a session flag.
"""

import hmac
import os
from functools import wraps

from flask import jsonify, redirect, request, session, url_for


def password_is_configured() -> bool:
    return bool(os.environ.get('FACE_AI_PASSWORD'))


def check_password(password: str) -> bool:
    expected = os.environ.get('FACE_AI_PASSWORD', '')
    if not expected:
        # No password configured: fail closed rather than silently allow-all.
        return False
    return hmac.compare_digest(password, expected)


def login_required_page(view):
    """For browser-navigated pages: redirect to /login if not authenticated."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get('authenticated'):
            return view(*args, **kwargs)
        return redirect(url_for('login', next=request.path))

    return wrapped


def login_required_api(view):
    """For JSON/API/image routes: respond 401 instead of redirecting."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get('authenticated'):
            return view(*args, **kwargs)
        return jsonify({'success': False, 'error': 'Authentication required'}), 401

    return wrapped
