"""
HTTP client for communicating with Backend 2 (Qt6/C++ server on port 9000).
"""
import logging
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class Backend2Client:

    def __init__(self):
        self._base_url = settings.BACKEND2_URL
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(10.0),
                proxy=None,
            )
        return self._client

    async def health_check(self) -> dict:
        try:
            client = await self._get_client()
            resp = await client.get("/health")
            resp.raise_for_status()
            return {"backend2": "ok", "detail": resp.json()}
        except Exception as e:
            logger.warning(f"Backend2 health check failed: {e}")
            return {"backend2": "unreachable", "error": str(e)}

    async def calc_energy(self, request_data: dict) -> dict:
        client = await self._get_client()
        resp = await client.post("/internal/calc/energy", json=request_data)
        resp.raise_for_status()
        return resp.json()

    async def predict_math(self, request_data: dict) -> dict:
        client = await self._get_client()
        resp = await client.post("/internal/predict/math", json=request_data)
        resp.raise_for_status()
        return resp.json()

    async def reset_session(self, session_id: str) -> dict:
        client = await self._get_client()
        resp = await client.post("/internal/session/reset",
                                 json={"session_id": session_id})
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


backend2_client = Backend2Client()
