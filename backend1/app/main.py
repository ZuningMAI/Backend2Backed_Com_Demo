"""
Backend 1 -- FastAPI application entry point.
Serves as the proxy/gateway between frontend and Backend 2 (Qt6/C++).
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health, energy
from app.services.tdengine_client import tdengine_client
from app.services.backend2_client import backend2_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"TDengine target: {settings.TDENGINE_HOST}:{settings.TDENGINE_PORT}")
    logger.info(f"Backend2 target: {settings.BACKEND2_URL}")

    if not tdengine_client.connect():
        logger.warning("TDengine connection failed -- telemetry writes will be skipped")
    else:
        logger.info("TDengine connected successfully")

    yield

    # Shutdown
    logger.info("Shutting down...")
    tdengine_client.close()
    await backend2_client.close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(energy.router)


@app.get("/")
async def root():
    return {
        "service": f"{settings.APP_NAME} v{settings.APP_VERSION}",
        "docs": "/docs",
        "health": "/health",
        "timestamp_ms": int(time.time() * 1000),
    }
