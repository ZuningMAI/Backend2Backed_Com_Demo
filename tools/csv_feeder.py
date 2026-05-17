#!/usr/bin/env python3
"""
CSV Data Feeder: reads OptReslog CSV, interpolates to 100ms, derives
missing telemetry fields (battery_power, soc), and feeds to Backend 1.

Usage:
  python tools/csv_feeder.py data/OptReslog.*.csv --interval 0.1 --backend http://localhost:8000

Columns in CSV (after header line 2):
  relative_time(s), position(m), speed(km/h), force(kN), operation_mode(0/1/2/3)
"""

import csv
import json
import time
import argparse
import sys
import math
from pathlib import Path

import httpx
import numpy as np


# ── Train parameters (from metrodas das_db_if / das_alg_if) ──
class TrainParams:
    MASS_TONNES = 350.0          # 车重 (t)
    EFFICIENCY = 0.85            # 传动效率 η
    BATTERY_CAPACITY_KWH = 200.0 # 电池额定能量 (kWh)
    BAT_RATIO_TRACTION = 0.30    # 牵引时电池供电占比
    BAT_RATIO_REGEN = 0.50       # 再生时电池吸收占比
    AUX_POWER_KW = 30.0          # 辅助系统功耗 (kW)
    SOC_INITIAL = 80.0           # 初始 SOC (%)
    MAX_CAT_POWER = 1500.0       # 接触网最大供电 (kW)


# ── CSV Loading ──

def load_csv(filepath: str) -> dict:
    """Load an OptReslog CSV, return metadata + data arrays."""
    data = {"relative_time": [], "position": [], "speed": [],
            "force": [], "mode": []}
    with open(filepath) as f:
        meta_line = f.readline().strip()
        parts = meta_line.split(",")
        meta = {
            "line": parts[0].split("：")[1] if "线路" in parts[0] else "",
            "section": parts[1].split("：")[1] if "优化区间" in parts[1] else "",
            "target_time_s": float(parts[2].split("：")[1]) if "设定时间" in parts[2] else 0,
            "actual_time_s": float(parts[3].split("：")[1]) if "实际时间" in parts[3] else 0,
            "energy_kwh": float(parts[4].split("：")[1]) if "能耗" in parts[4] else 0,
        }
        f.readline()  # skip header
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


# ── 100ms Interpolation ──

def interpolate_100ms(data: dict, dt: float = 0.1) -> dict:
    """Linear interpolation to uniform dt grid."""
    t_raw = data["relative_time"]
    T_max = t_raw[-1]
    t_grid = np.arange(0.0, T_max + dt / 2, dt)

    interp = {}
    for key in ["position", "speed", "force"]:
        interp[key] = np.interp(t_grid, t_raw, data[key])

    # mode: nearest-neighbour
    interp["mode"] = np.zeros(len(t_grid), dtype=int)
    for i, tg in enumerate(t_grid):
        idx = np.searchsorted(t_raw, tg)
        idx = min(idx, len(data["mode"]) - 1)
        interp["mode"][i] = int(data["mode"][idx])

    interp["relative_time"] = t_grid
    return interp


# ── Derived Telemetry ──

def derive_telemetry(interp: dict, params: TrainParams, dt: float = 0.1):
    """
    From (time, pos, speed, force, mode) derive:
      tractive_force, electric_brake_force, battery_power, soc
    Returns list of dicts ready for Backend 1 POST.
    """
    n = len(interp["relative_time"])
    tractive_force = np.zeros(n)
    electric_brake_force = np.zeros(n)
    battery_power = np.zeros(n)
    soc = np.zeros(n)

    soc_val = params.SOC_INITIAL
    eta = params.EFFICIENCY
    bat_cap = params.BATTERY_CAPACITY_KWH

    for i in range(n):
        force = interp["force"][i]
        speed_kmh = interp["speed"][i]
        speed_ms = speed_kmh / 3.6

        # 1. Split force
        if force > 0:
            tractive_force[i] = force
            electric_brake_force[i] = 0.0
        elif force < 0:
            tractive_force[i] = 0.0
            electric_brake_force[i] = abs(force)
        else:
            tractive_force[i] = 0.0
            electric_brake_force[i] = 0.0

        # 2. Mechanical power
        p_mech = force * speed_ms  # kW

        # 3. DC bus power
        if p_mech > 0:
            p_dc = p_mech / eta
        elif p_mech < 0:
            p_dc = p_mech * eta
        else:
            p_dc = 0.0

        # 4. Battery power (simplified model)
        if p_dc > 0:
            bat_power = p_dc * params.BAT_RATIO_TRACTION
        elif p_dc < 0:
            bat_power = p_dc * params.BAT_RATIO_REGEN  # negative = charging
        else:
            bat_power = -params.AUX_POWER_KW  # auxiliary only

        # SOC-based limits
        if bat_power < 0 and soc_val >= 99.0:
            bat_power = 0.0
        if bat_power > 0 and soc_val <= 10.0:
            bat_power = 0.0

        battery_power[i] = bat_power

        # 5. SOC update
        soc_val -= (bat_power * dt / 3600.0) / bat_cap * 100.0
        soc_val = max(0.0, min(100.0, soc_val))
        soc[i] = soc_val

    return {
        "tractive_force": tractive_force,
        "electric_brake_force": electric_brake_force,
        "battery_power": battery_power,
        "soc": soc,
    }


# ── Feed to Backend ──

def feed_to_backend(interp: dict, derived: dict, backend_url: str,
                    start_time_ms: int, dry_run: bool = False, speed_factor: float = 1.0,
                    sim_duration_s: float = 200.0):
    """
    Feed interpolated telemetry to Backend 1 at 100ms intervals.
    speed_factor > 1 speeds up playback (e.g. 10 = 10x realtime).
    """
    t = interp["relative_time"]
    n = len(t)
    dt_total = t[-1] - t[0]
    session_id = None
    total_req = 0
    success_req = 0

    print(f"Feeding {n} points over {dt_total:.1f}s (speed {speed_factor}x)")
    print(f"Backend: {backend_url}")
    print()

    with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
        t_start_wall = time.time()

        for i in range(n):
            # Build request
            req = {
                "session_id": session_id,
                "tractive_force": round(float(derived["tractive_force"][i]), 2),
                "electric_brake_force": round(float(derived["electric_brake_force"][i]), 2),
                "speed": round(float(interp["speed"][i]), 3),
                "battery_power": round(float(derived["battery_power"][i]), 2),
                "soc": round(float(derived["soc"][i]), 2),
                "sample_interval": 0.1,
                "start_time": start_time_ms,
                "end_time": start_time_ms + int(sim_duration_s * 1000) + 10000,
            }

            try:
                resp = client.post(
                    f"{backend_url}/vehicle/energy/result",
                    json=req,
                )
                resp.raise_for_status()
                body = resp.json()
                success_req += 1

                # Extract session_id from first response
                if session_id is None and body.get("message"):
                    import re
                    m = re.search(r"session:\s*(\S+)", body["message"])
                    if m:
                        session_id = m.group(1).rstrip(")")

                if not dry_run and i % 500 == 0:
                    data = body.get("data", {})
                    print(f"  [{i}/{n}] net_energy={data.get('net_energy', 0):.3f} "
                          f"regen={data.get('regenerative_energy', 0):.3f} "
                          f"soc={derived['soc'][i]:.1f}%")

            except Exception as e:
                print(f"  [{i}/{n}] ERROR: {e}")
                if not dry_run and total_req > 10:
                    print("  Too many errors, stopping")
                    break

            total_req += 1

            # Real-time pacing
            if not dry_run and i < n - 1:
                elapsed_wall = time.time() - t_start_wall
                elapsed_sim = t[i] / speed_factor
                sleep_t = elapsed_sim - elapsed_wall
                if sleep_t > 0:
                    time.sleep(min(sleep_t, 0.5))  # cap sleep at 500ms

    print(f"\nDone: {success_req}/{total_req} requests ({100*success_req/max(1,total_req):.1f}%)")
    return session_id


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="CSV telemetry feeder for Backend 1")
    parser.add_argument("csv_file", help="Path to OptReslog CSV")
    parser.add_argument("--interval", type=float, default=0.1, help="Resample interval in seconds")
    parser.add_argument("--backend", default="http://localhost:8000", help="Backend 1 URL")
    parser.add_argument("--dry-run", action="store_true", help="Print without sending")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (10=10x)")
    parser.add_argument("--start-offset-sec", type=int, default=28800,
                        help="Start time as seconds from midnight (28800=8:00AM)")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    print(f"Loading: {csv_path.name}")
    meta, raw = load_csv(str(csv_path))
    print(f"  Section: {meta['section']}, target: {meta['target_time_s']}s, "
          f"actual: {meta['actual_time_s']:.1f}s, optimal energy: {meta['energy_kwh']:.3f} kWh")
    print(f"  {len(raw['relative_time'])} raw points")

    print(f"Interpolating to {args.interval}s grid...")
    interp = interpolate_100ms(raw, dt=args.interval)
    print(f"  {len(interp['relative_time'])} interpolated points")

    print("Deriving telemetry...")
    params = TrainParams()
    derived = derive_telemetry(interp, params, dt=args.interval)

    # Time window: use current wall time for alignment with backend's now_ms
    # dt_total = interp["relative_time"][-1] from the interpolation
    sim_duration_s = float(interp["relative_time"][-1])
    wall_now = int(time.time() * 1000)
    start_time_ms = wall_now  # window starts now

    session_id = feed_to_backend(interp, derived, args.backend,
                                  start_time_ms, dry_run=args.dry_run,
                                  speed_factor=args.speed,
                                  sim_duration_s=sim_duration_s)

    if session_id:
        print(f"Session: {session_id}")
        print(f"\nPredict test:")
        try:
            with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
                resp = client.post(
                    f"{args.backend}/vehicle/energy/time_predict",
                    json={
                        "session_id": session_id,
                        "lookback_window": 1000,
                        "forecast_horizon": 200,
                        "model_type": "math_only",
                    },
                )
                body = resp.json()
                data = body.get("data", {})
                actual = data.get("actual_curve", [])
                predicted = data.get("predicted_curve", [])
                print(f"  actual_curve: {len(actual)} points")
                print(f"  predicted_curve: {len(predicted)} points")
                if predicted:
                    print(f"  first: pos={predicted[0]['position']:.3f} energy={predicted[0]['energy']:.3f}")
                    print(f"  last:  pos={predicted[-1]['position']:.3f} energy={predicted[-1]['energy']:.3f}")
        except Exception as e:
            print(f"  Predict test failed: {e}")


if __name__ == "__main__":
    main()
