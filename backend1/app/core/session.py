"""
Session management for Backend 1.
"""
import uuid
import time

_sessions: dict[str, float] = {}


def generate_session_id() -> str:
    sid = str(uuid.uuid4())
    _sessions[sid] = time.time()
    return sid


def touch_session(session_id: str) -> None:
    _sessions[session_id] = time.time()


def validate_session(session_id: str, timeout_seconds: int = 3600) -> bool:
    if session_id not in _sessions:
        return False
    age = time.time() - _sessions[session_id]
    return age <= timeout_seconds


def cleanup_expired_sessions(timeout_seconds: int = 3600) -> int:
    now = time.time()
    expired = [sid for sid, ts in _sessions.items()
               if now - ts > timeout_seconds]
    for sid in expired:
        del _sessions[sid]
    return len(expired)


def get_active_session_count() -> int:
    return len(_sessions)
