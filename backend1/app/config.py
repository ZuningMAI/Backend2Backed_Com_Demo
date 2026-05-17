"""
Centralized configuration for Backend 1.
All values can be overridden via environment variables.
"""
import os


class Settings:
    APP_NAME: str = "Backend1"
    APP_VERSION: str = "0.1.0"
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("BACKEND1_PORT", "8000"))

    # TDengine connection
    TDENGINE_HOST: str = os.getenv("TDENGINE_HOST", "localhost")
    TDENGINE_PORT: int = int(os.getenv("TDENGINE_PORT", "6041"))
    TDENGINE_USER: str = os.getenv("TDENGINE_USER", "root")
    TDENGINE_PASSWORD: str = os.getenv("TDENGINE_PASSWORD", "taosdata")
    TDENGINE_DATABASE: str = os.getenv("TDENGINE_DATABASE", "energy_mgmt")

    # Backend 2 connection
    BACKEND2_URL: str = os.getenv("BACKEND2_URL", "http://localhost:9000")

    # Session
    SESSION_TIMEOUT_SECONDS: int = 3600
    DEFAULT_DATA_FREQUENCY_HZ: float = 1.0

    # CORS
    CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
