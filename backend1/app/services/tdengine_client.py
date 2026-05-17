"""
TDengine client service for Backend 1.
Uses TDengine REST API (taosAdapter on port 6041) via httpx.
"""
import logging
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class TDengineClient:

    def __init__(self):
        self._base_url = f"http://{settings.TDENGINE_HOST}:{settings.TDENGINE_PORT}"
        self._auth = (settings.TDENGINE_USER, settings.TDENGINE_PASSWORD)
        self._client: Optional[httpx.Client] = None
        self._connected = False

    def connect(self) -> bool:
        try:
            self._client = httpx.Client(timeout=httpx.Timeout(10.0),
                                        proxy=None)
            resp = self._client.post(
                f"{self._base_url}/rest/sql",
                auth=self._auth,
                content=f"USE {settings.TDENGINE_DATABASE}",
                headers={"Content-Type": "text/plain"},
            )
            result = resp.json()
            if result.get("code") == 0:
                self._connected = True
                logger.info(f"TDengine connected: {settings.TDENGINE_HOST}:{settings.TDENGINE_PORT}")
                return True
            else:
                logger.error(f"TDengine USE failed: {result}")
                return False
        except Exception as e:
            logger.error(f"TDengine connection failed: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        if not self._client or not self._connected:
            return False
        try:
            resp = self._client.post(
                f"{self._base_url}/rest/sql",
                auth=self._auth,
                content="SELECT 1",
                headers={"Content-Type": "text/plain"},
            )
            return resp.json().get("code") == 0
        except Exception:
            return False

    def _execute(self, sql: str) -> dict:
        """Execute SQL via REST API, return parsed JSON response."""
        if not self._client:
            raise RuntimeError("TDengine client not connected")
        resp = self._client.post(
            f"{self._base_url}/rest/sql",
            auth=self._auth,
            content=sql,
            headers={"Content-Type": "text/plain"},
        )
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> dict:
        status = "ok" if self.is_connected else "disconnected"
        return {
            "tdengine": status,
            "host": f"{settings.TDENGINE_HOST}:{settings.TDENGINE_PORT}",
            "database": settings.TDENGINE_DATABASE,
        }

    def ensure_subtable(self, session_id: str, vehicle_id: str = "default") -> bool:
        if not self._client:
            return False
        safe_sid = session_id.replace("-", "_").replace(" ", "_")
        table_name = f"telemetry_{safe_sid}"
        try:
            sql = (
                f"CREATE TABLE IF NOT EXISTS {settings.TDENGINE_DATABASE}.{table_name} "
                f"USING {settings.TDENGINE_DATABASE}.vehicle_telemetry "
                f"TAGS ('{session_id}', '{vehicle_id}')"
            )
            result = self._execute(sql)
            if result.get("code") == 0:
                logger.info(f"Sub-table ensured: {table_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to create sub-table {table_name}: {e}")
            return False

    def insert_telemetry(self, session_id: str, data_point: "DataPoint", energy: float = 0.0) -> bool:
        if not self._client:
            return False
        safe_sid = session_id.replace("-", "_").replace(" ", "_")
        table_name = f"telemetry_{safe_sid}"
        try:
            sql = (
                f"INSERT INTO {settings.TDENGINE_DATABASE}.{table_name} VALUES "
                f"({data_point.time}, {data_point.speed}, {data_point.tractive_force}, "
                f"{data_point.electric_brake_force}, {data_point.battery_power}, "
                f"{data_point.soc}, {energy})"
            )
            self._execute(sql)
            return True
        except Exception as e:
            logger.error(f"Insert telemetry failed: {e}")
            return False

    def query_history(self, session_id: str, limit: int = 1000) -> list[dict]:
        if not self._client:
            return []
        safe_sid = session_id.replace("-", "_").replace(" ", "_")
        table_name = f"telemetry_{safe_sid}"
        try:
            sql = (
                f"SELECT * FROM {settings.TDENGINE_DATABASE}.{table_name} "
                f"ORDER BY ts DESC LIMIT {limit}"
            )
            result = self._execute(sql)
            rows = []
            if result.get("code") == 0 and result.get("data"):
                for row in result["data"]:
                    # TDengine REST returns timestamps as ISO strings; convert to ms
                    ts_val = row[0]
                    if isinstance(ts_val, str):
                        from datetime import datetime
                        try:
                            ts_val = int(datetime.fromisoformat(ts_val.replace("Z", "+00:00")).timestamp() * 1000)
                        except Exception:
                            ts_val = 0
                    rows.append({
                        "ts": ts_val,
                        "speed": float(row[1]) if len(row) > 1 else 0.0,
                        "tractive_force": float(row[2]) if len(row) > 2 else 0.0,
                        "electric_brake_force": float(row[3]) if len(row) > 3 else 0.0,
                        "battery_power": float(row[4]) if len(row) > 4 else 0.0,
                        "soc": float(row[5]) if len(row) > 5 else 0.0,
                        "energy": float(row[6]) if len(row) > 6 else 0.0,
                    })
            return rows
        except Exception as e:
            logger.error(f"Query history failed: {e}")
            return []

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            self._connected = False


tdengine_client = TDengineClient()
