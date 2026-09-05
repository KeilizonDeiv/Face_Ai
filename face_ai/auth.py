"""
Minimal session-based auth: a single shared password gates the whole app.

Not a multi-user system by design (see plan) — one password, checked with a
constant-time comparison, stored server-side only as a session flag.
"""

import hmac
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Dict

from flask import jsonify, redirect, request, session, url_for


def password_is_configured() -> bool:
    return bool(os.environ.get('FACE_AI_PASSWORD'))


def check_password(password: str) -> bool:
    expected = os.environ.get('FACE_AI_PASSWORD', '')
    if not expected:
        # No password configured: fail closed rather than silently allow-all.
        return False
    return hmac.compare_digest(password, expected)


# ---------- login rate limiting ----------
#
# Hand-rolled in-memory limiter rather than a dependency: this app is a
# single Flask process (see app.run in app.py), so a module-level dict is
# sufficient and avoids pulling in Flask-Limiter for one endpoint.

LOGIN_MAX_ATTEMPTS = 8
LOGIN_WINDOW_SECONDS = 300  # 5 minutes

_login_attempts: Dict[str, "deque[float]"] = defaultdict(deque)
_login_attempts_lock = threading.Lock()


def _prune_locked(ip: str, now: float):
    attempts = _login_attempts[ip]
    while attempts and now - attempts[0] > LOGIN_WINDOW_SECONDS:
        attempts.popleft()


def is_login_rate_limited(ip: str) -> bool:
    now = time.monotonic()
    with _login_attempts_lock:
        _prune_locked(ip, now)
        return len(_login_attempts[ip]) >= LOGIN_MAX_ATTEMPTS


def record_failed_login(ip: str):
    now = time.monotonic()
    with _login_attempts_lock:
        _prune_locked(ip, now)
        _login_attempts[ip].append(now)


def clear_login_attempts(ip: str):
    with _login_attempts_lock:
        _login_attempts.pop(ip, None)


# ---------- CSRF ----------
#
# Hand-rolled session-bound token (synchronizer token pattern) rather than
# Flask-WTF, again to avoid a new dependency for one mechanism. A random
# token is minted into the session on first render of a page that has a
# form, and every unsafe request must echo it back via a form field or the
# X-CSRFToken header.

CSRF_SESSION_KEY = '_csrf_token'
CSRF_HEADER_NAME = 'X-CSRFToken'
CSRF_FORM_FIELD = 'csrf_token'
CSRF_SAFE_METHODS = ('GET', 'HEAD', 'OPTIONS')


def get_csrf_token() -> str:
    """Return this session's CSRF token, minting one if it doesn't have one yet."""
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_hex(32)
        session[CSRF_SESSION_KEY] = token
    return token


def csrf_token_valid(submitted: str) -> bool:
    expected = session.get(CSRF_SESSION_KEY)
    return bool(expected) and bool(submitted) and hmac.compare_digest(expected, submitted)


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
