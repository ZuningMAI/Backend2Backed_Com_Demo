"""Health check router."""
import time
from fastapi import APIRouter

from app.config import settings
from app.services.tdengine_client import tdengine_client
from app.services.backend2_client import backend2_client
from app.core.session import get_active_session_count

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    td_health = tdengine_client.health_check()
    b2_health = await backend2_client.health_check()

    return {
        "status": "ok",
        "service": f"{settings.APP_NAME} v{settings.APP_VERSION}",
        "timestamp_ms": int(time.time() * 1000),
        "dependencies": {
            **td_health,
            **b2_health,
        },
        "active_sessions": get_active_session_count(),
    }
