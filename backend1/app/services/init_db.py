"""
One-time initialization: load CSV → 1ms interpolate → derive → batch insert into TDengine.
Run once at startup if TDengine table is empty.
"""
import csv
import re
import time
import logging
from pathlib import Path
import numpy as np
import httpx

logger = logging.getLogger(__name__)

# ── Train params ──
ETA = 0.85
BATTERY_CAPACITY_KWH = 200.0
BAT_RATIO_TRACTION = 0.30
BAT_RATIO_REGEN = 0.50
AUX_POWER_KW = 120.0
SOC_INITIAL = 80.0

# TDengine REST config
TD_HOST = "http://localhost:6041"
TD_AUTH = ("root", "taosdata")
TD_DB = "energy_mgmt"
TD_TABLE = "line_telemetry_1ms"

# Column mapping: code_name → TDengine column
COL_MAP = {
    "global_time_ms": "gtm", "position": "pos", "speed": "spd",
    "force": "frc", "mode": "opr", "tractive_force": "trf",
    "electric_brake_force": "ebf", "battery_power": "bpw",
    "soc": "soc", "real_time_energy": "rte"
}

# Sections in order: (section_id, run_time_s)
SECTIONS = [
    ("1-2", 86), ("2-3", 145), ("3-4", 165), ("4-5", 92),
    ("5-6", 455), ("6-7", 130), ("7-8", 110), ("8-9", 87),
    ("9-10", 120), ("10-11", 131), ("11-12", 95), ("12-13", 244), ("13-14", 173),
]
DWELL_TIME_S = 60

# ── CSV loading ──

def find_csv_files(data_dir: str) -> dict:
    """Scan data/ for OptReslog CSV files, map by section."""
    data_dir = Path(data_dir)
    files = {}
    for f in data_dir.glob("OptReslog*.csv"):
        name = f.name
        m = re.search(r"FZ(\d+)-(\d+)-(\d+)-(\d+)", name)
        if m:
            section = f"{m.group(2)}-{m.group(3)}"
            run_time = int(m.group(4))
            files[section] = (str(f), run_time)
    return files


def load_csv(filepath: str) -> tuple[dict, dict]:
    """Load CSV, return (meta, arrays)."""
    data = {"relative_time": [], "position": [], "speed": [],
            "force": [], "mode": []}
    with open(filepath) as f:
        meta_line = f.readline().strip()
        meta = {"raw": meta_line}
        f.readline()
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            data["relative_time"].append(float(row[0]))
            data["position"].append(float(row[1]))
            data["speed"].append(float(row[2]))
            data["force"].append(float(row[3]))
            data["mode"].append(int(row[4]))
    for k in data:
        data[k] = np.array(data[k], dtype=np.float64)
    return meta, data


# ── 1ms interpolation ──

def interpolate_1ms(data: dict, run_time_s: float) -> dict:
    """Linear interpolate to uniform 1ms grid of length run_time_s * 1000."""
    n_points = int(run_time_s * 1000)
    t_raw = data["relative_time"]
    t_grid = np.arange(0, n_points) / 1000.0

    interp = {}
    for key in ["position", "speed", "force"]:
        interp[key] = np.interp(t_grid, t_raw, data[key])

    interp["mode"] = np.zeros(n_points, dtype=int)
    for i, tg in enumerate(t_grid):
        idx = np.searchsorted(t_raw, tg)
        idx = min(idx, len(data["mode"]) - 1)
        interp["mode"][i] = int(data["mode"][idx])

    interp["global_time_ms"] = np.arange(n_points, dtype=np.int64)
    return interp


# ── Derive fields ──

def derive_fields(interp: dict, initial_soc: float) -> dict:
    """Derive tractive_force, electric_brake_force, battery_power, soc, real_time_energy."""
    n = len(interp["global_time_ms"])
    trf = np.zeros(n)
    ebf = np.zeros(n)
    bpw = np.zeros(n)
    soc_arr = np.zeros(n)
    rte = np.zeros(n)

    soc_val = initial_soc
    dt_h = 0.001 / 3600.0  # 1ms in hours
    bat_cap = BATTERY_CAPACITY_KWH

    for i in range(n):
        force = interp["force"][i]
        spd_kmh = interp["speed"][i]
        spd_ms = spd_kmh / 3.6

        # Split force
        if force > 0:
            trf[i] = force; ebf[i] = 0.0
        elif force < 0:
            trf[i] = 0.0; ebf[i] = abs(force)
        else:
            trf[i] = 0.0; ebf[i] = 0.0

        # Mechanical power
        p_mech = force * spd_ms
        if p_mech > 0:
            p_dc = p_mech / ETA
        elif p_mech < 0:
            p_dc = p_mech * ETA
        else:
            p_dc = 0.0

        # Battery power
        if p_dc > 0:
            bat = p_dc * BAT_RATIO_TRACTION
        elif p_dc < 0:
            bat = p_dc * BAT_RATIO_REGEN
        else:
            bat = -AUX_POWER_KW

        if bat < 0 and soc_val >= 99.0:
            bat = 0.0
        if bat > 0 and soc_val <= 10.0:
            bat = 0.0
        bpw[i] = bat

        # SOC
        soc_val -= (bat * dt_h) / bat_cap * 100.0
        soc_val = max(0.0, min(100.0, soc_val))
        soc_arr[i] = soc_val

        # Real-time energy
        p_consume = max(p_dc - bat, 0.0) + max(bat, 0.0)
        rte[i] = p_consume / (spd_kmh + 0.01) if spd_kmh > 0.01 else 0.0

    return {"tractive_force": trf, "electric_brake_force": ebf,
            "battery_power": bpw, "soc": soc_arr, "real_time_energy": rte}


# ── Dwell section ──

def make_dwell_data(start_ms: int, duration_s: int = 60) -> dict:
    """Generate 60s station dwell data (speed=0, force=0, auxiliary power only)."""
    n = duration_s * 1000
    dt_h = 0.001 / 3600.0
    soc_val = SOC_INITIAL  # will be overwritten during sequential processing

    return {
        "global_time_ms": np.arange(start_ms, start_ms + n, dtype=np.int64),
        "position": np.full(n, 0.0),
        "speed": np.zeros(n),
        "force": np.zeros(n),
        "mode": np.full(n, 4, dtype=int),
        "tractive_force": np.zeros(n),
        "electric_brake_force": np.zeros(n),
        "battery_power": np.full(n, -AUX_POWER_KW),
        "soc": np.zeros(n),  # placeholder
        "real_time_energy": np.zeros(n),
    }


# ── TDengine insert ──

def td_execute(client: httpx.Client, sql: str) -> dict:
    resp = client.post(f"{TD_HOST}/rest/sql", auth=TD_AUTH,
                       content=sql, headers={"Content-Type": "text/plain"})
    resp.raise_for_status()
    return resp.json()


def insert_section(client: httpx.Client, section_id: str, order: int,
                   interp: dict, derived: dict, start_ms: int,
                   dwell_soc: float = None) -> float:
    """Insert one section's data into TDengine. Returns final SOC."""
    n = len(interp["global_time_ms"])
    safe_sid = section_id.replace("-", "_")

    # Create sub-table
    td_execute(client,
        f"CREATE TABLE IF NOT EXISTS {TD_DB}.sec_{safe_sid} "
        f"USING {TD_DB}.{TD_TABLE} TAGS ('FZ602', '{section_id}', {order})")

    # Update SOC in derived arrays with carry-over
    if dwell_soc is not None:
        dt_h = 0.001 / 3600.0
        soc_val = dwell_soc
        for i in range(n):
            derived["soc"][i] = soc_val
            bat = derived["battery_power"][i]
            soc_val -= (bat * dt_h) / BATTERY_CAPACITY_KWH * 100.0
            soc_val = max(0.0, min(100.0, soc_val))

    # Batch insert (200 rows per batch, multi-row VALUES)
    batch_size = 200
    ts_base = int(time.time() * 1000)  # current wall time as base

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        rows = []
        for i in range(start, end):
            ts = ts_base + start_ms + interp["global_time_ms"][i]
            gtm = int(start_ms + interp["global_time_ms"][i])
            pos = interp["position"][i]
            spd = interp["speed"][i]
            frc = interp["force"][i]
            opr = int(interp["mode"][i])
            trf = derived["tractive_force"][i]
            ebf = derived["electric_brake_force"][i]
            bpw = derived["battery_power"][i]
            soc = derived["soc"][i]
            rte = derived["real_time_energy"][i]
            rows.append(f"({ts}, {gtm}, {pos}, {spd}, {frc}, {opr}, {trf}, {ebf}, {bpw}, {soc}, {rte})")
        sql = f"INSERT INTO {TD_DB}.sec_{safe_sid} VALUES " + " ".join(rows)
        result = td_execute(client, sql)
        if result.get("code") != 0:
            logger.error(f"Insert failed at batch {start}-{end}: {result.get('desc', 'unknown')}")
        if start % 10000 == 0:
            logger.info(f"  sec_{safe_sid}: {end}/{n}")

    return soc_val if dwell_soc is not None else derived["soc"][-1]


# ── Main ──

def init_database(data_dir: str = "data"):
    """Full initialization: CSV → TDengine."""
    logger.info("Starting TDengine initialization...")
    files = find_csv_files(data_dir)
    logger.info(f"Found {len(files)} CSV files")

    client = httpx.Client(timeout=httpx.Timeout(30.0))
    soc_carry = SOC_INITIAL
    global_ms = 0

    for order, (section_id, run_time_s) in enumerate(SECTIONS, 1):
        if section_id not in files:
            logger.warning(f"CSV not found for section {section_id}, skipping")
            continue

        filepath, csv_run_time = files[section_id]
        logger.info(f"Section {section_id} (order={order}, {run_time_s}s): {filepath}")

        # Load & interpolate
        meta, raw = load_csv(filepath)
        interp = interpolate_1ms(raw, run_time_s)

        # Update global time
        interp["global_time_ms"] = np.arange(global_ms, global_ms + len(interp["global_time_ms"]), dtype=np.int64)

        # Derive
        derived = derive_fields(interp, soc_carry)

        # Insert
        final_soc = insert_section(client, section_id, order, interp, derived, 0, dwell_soc=soc_carry)
        soc_carry = derived["soc"][-1]
        global_ms += len(interp["global_time_ms"])
        logger.info(f"  → {len(interp['global_time_ms'])} rows, final_soc={soc_carry:.2f}%, global_ms={global_ms}")

        # Insert dwell after each section (except last)
        if order < len(SECTIONS):
            dwell = make_dwell_data(global_ms, DWELL_TIME_S)
            # Update SOC for dwell: constant discharge
            dt_h = 0.001 / 3600.0
            for i in range(len(dwell["global_time_ms"])):
                dwell["soc"][i] = soc_carry
                soc_carry -= (AUX_POWER_KW * dt_h) / BATTERY_CAPACITY_KWH * 100.0
                soc_carry = max(0.0, min(100.0, soc_carry))

            # Insert dwell as a separate sub-table with section_id like "dwell_1"
            td_execute(client,
                f"CREATE TABLE IF NOT EXISTS {TD_DB}.dwell_{order} "
                f"USING {TD_DB}.{TD_TABLE} TAGS ('FZ602', 'dwell', {order})")
            global_ms += len(dwell["global_time_ms"])
            logger.info(f"  → {len(dwell['global_time_ms'])} dwell rows, global_ms={global_ms}")

    client.close()
    logger.info(f"Initialization complete: {global_ms}ms total ({global_ms/1000:.0f}s)")
    return global_ms


# ── CLI ──

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    init_database("data")
