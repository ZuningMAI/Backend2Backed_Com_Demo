"""
Runtime scheduler: dual 100ms timers.
  Timer A (calc):   from t=0ms,  query 100 rows,  call /internal/calc/energy
  Timer B (predict): from t=800ms, query 800 rows, call /internal/predict/train
"""
import time
import threading
import logging
import httpx

logger = logging.getLogger(__name__)

TD_HOST = "http://localhost:6041"
TD_AUTH = ("root", "taosdata")
B2_URL = "http://localhost:9000"
TABLE = "energy_mgmt.sec_1_2"  # direct sub-table query (faster than super table)

def _td_query(sql: str) -> list[dict]:
    with httpx.Client(timeout=httpx.Timeout(10.0)) as cli:
        resp = cli.post(f"{TD_HOST}/rest/sql", auth=TD_AUTH,
                        content=sql, headers={"Content-Type": "text/plain"})
        resp.raise_for_status()
        result = resp.json()
    if result.get("code") != 0:
        return []
    rows = []
    for row in result.get("data", []):
        # Query columns: gtm(0), pos(1), spd(2), frc(3), opr(4), trf(5), ebf(6), bpw(7), soc(8), rte(9)
        rows.append({
            "time": int(round(float(row[0]))) if row[0] else 0,
            "position": float(row[1]) if len(row) > 1 else 0.0,
            "speed": float(row[2]) if len(row) > 2 else 0.0,
            "force": float(row[3]) if len(row) > 3 else 0.0,
            "mode": int(float(row[4])) if len(row) > 4 else 0,
            "tractive_force": float(row[5]) if len(row) > 5 else 0.0,
            "electric_brake_force": float(row[6]) if len(row) > 6 else 0.0,
            "battery_power": float(row[7]) if len(row) > 7 else 0.0,
            "soc": float(row[8]) if len(row) > 8 else 0.0,
            "real_time_energy": float(row[9]) if len(row) > 9 else 0.0,
        })
    return rows


class Session:
    def __init__(self, sid: str, total_ms: int):
        self.sid = sid
        self.total_ms = total_ms
        self.t_ms = 0
        self.state = "running"
        self.energy_data = {}
        self.actual_curve = []
        self.predicted_curve = []
        self._last_net_e = 0.0
        self._lock = threading.Lock()

    def progress(self) -> dict:
        pct = round(self.t_ms / max(1, self.total_ms) * 100, 1)
        return {"current_ms": self.t_ms, "total_ms": self.total_ms,
                "percent": min(pct, 100.0), "state": self.state}

    def start_timers(self):
        threading.Thread(target=self._run_calc, daemon=True).start()
        threading.Thread(target=self._delayed_predict, daemon=True).start()

    def _delayed_predict(self):
        time.sleep(0.8)
        if self.state != "running": return
        threading.Thread(target=self._run_predict, daemon=True).start()

    def _run_calc(self):
        logger.info(f"Calc thread started for {self.sid}")
        while self.state == "running":
            t = self.t_ms
            try:
                rows = _td_query(
                    f"SELECT gtm,pos,spd,frc,opr,trf,ebf,bpw,soc,rte "
                    f"FROM {TABLE} WHERE gtm >= {t} AND gtm < {t + 100} "
                    f"ORDER BY gtm ASC")
                if not rows:
                    if self.t_ms < 200:
                        logger.warning(f"Calc t={t}: no rows from TDengine (table={TABLE})")
                    time.sleep(0.1)
                    with self._lock:
                        self.t_ms += 100
                        if self.t_ms >= self.total_ms:
                            self.state = "completed"
                    continue

                data_points = [{
                    "time": r["time"], "tractive_force": r["tractive_force"],
                    "electric_brake_force": r["electric_brake_force"],
                    "speed": r["speed"], "battery_power": r["battery_power"],
                    "soc": r["soc"],
                } for r in rows]

                with httpx.Client(timeout=httpx.Timeout(10.0)) as cli:
                    resp = cli.post(f"{B2_URL}/internal/calc/energy", json={
                        "session_id": self.sid, "data_points": data_points,
                        "sample_interval": 0.001})
                    resp.raise_for_status()
                    with self._lock:
                        self.energy_data = resp.json().get("data", {})

                # Build actual_curve: interpolate net_energy across sampled positions
                with self._lock:
                    net_e = self.energy_data.get("net_energy", 0)
                    if net_e > 0:
                        batch_samples = list(range(0, len(rows), 10))  # every 10th row
                        n_samples = len(batch_samples)
                        for j, i in enumerate(batch_samples):
                            pos_km = rows[i]["position"] / 1000.0
                            # Linearly interpolate energy between previous and current net_e
                            frac = (j + 1) / n_samples if n_samples > 0 else 1.0
                            interp_e = self._last_net_e + (net_e - self._last_net_e) * frac
                            self.actual_curve.append({"position": pos_km, "energy": interp_e})
                        self._last_net_e = net_e
                        # Deduplicate consecutive same-position points
                        dedup = []
                        for p in self.actual_curve:
                            if not dedup or abs(p["position"] - dedup[-1]["position"]) > 0.00001:
                                dedup.append(p)
                        self.actual_curve = dedup[-2000:]

            except Exception as e:
                logger.error(f"Calc error t={t}: {e}")

            with self._lock:
                self.t_ms += 100
                if self.t_ms >= self.total_ms:
                    self.state = "completed"
            time.sleep(0.1)

    def _run_predict(self):
        while self.state == "running":
            t = self.t_ms
            try:
                start_t = max(0, t - 800)
                rows = _td_query(
                    f"SELECT gtm,pos,spd,frc,opr,trf,ebf,bpw,soc,rte "
                    f"FROM {TABLE} WHERE gtm >= {start_t} AND gtm < {t} "
                    f"ORDER BY gtm ASC")
                if len(rows) < 100:
                    time.sleep(0.1)
                    continue

                history_data = [{
                    "time": r["time"], "speed": r["speed"], "force": r["force"],
                    "mode": r["mode"], "tractive_force": r["tractive_force"],
                    "electric_brake_force": r["electric_brake_force"],
                    "battery_power": r["battery_power"], "soc": r["soc"],
                    "real_time_energy": r["real_time_energy"],
                    "position": r["position"],  # meters, for predict start point
                } for r in rows]

                with self._lock:
                    net_e = self.energy_data.get("net_energy", 0)
                with httpx.Client(timeout=httpx.Timeout(10.0)) as cli:
                    resp = cli.post(f"{B2_URL}/internal/predict/train", json={
                        "session_id": self.sid, "history_data": history_data,
                        "cumulative_energy": net_e})
                    resp.raise_for_status()
                    with self._lock:
                        self.predicted_curve = resp.json().get("data", {}).get("predicted_curve", [])

            except Exception as e:
                logger.error(f"Predict error t={t}: {e}")
            time.sleep(0.1)


_sessions: dict[str, Session] = {}

def create_session(total_ms: int) -> str:
    import uuid
    sid = str(uuid.uuid4())
    sess = Session(sid, total_ms)
    _sessions[sid] = sess
    sess.start_timers()
    logger.info(f"Session {sid} started, total={total_ms}ms")
    return sid

def get_session(sid: str) -> Session | None:
    return _sessions.get(sid)

def get_energy_result(sid: str) -> dict:
    sess = _sessions.get(sid)
    if not sess:
        return {"status": 1, "message": "session not found", "data": {}, "progress": {}, "actual_curve": []}
    with sess._lock:
        return {
            "status": 0, "message": "success",
            "data": sess.energy_data,
            "progress": sess.progress(),
            "actual_curve": list(sess.actual_curve),
        }

def get_predict_result(sid: str) -> dict:
    sess = _sessions.get(sid)
    if not sess:
        return {"status": 1, "message": "session not found", "data": {}, "progress": {}}
    with sess._lock:
        predicted = list(sess.predicted_curve)
        # Anchor predicted curve to last actual point for visual continuity
        if sess.actual_curve and predicted:
            anchor = dict(sess.actual_curve[-1])
            # Only anchor if predicted curve doesn't already start from actual endpoint
            if predicted[0].get("position", 0) != anchor.get("position", -1):
                predicted.insert(0, anchor)
        return {
            "status": 0, "message": "success",
            "data": {
                "actual_curve": sess.actual_curve[-1:] if sess.actual_curve else [],
                "predicted_curve": predicted,
            },
            "progress": sess.progress(),
        }
