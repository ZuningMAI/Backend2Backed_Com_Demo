"""
Synthetic train run data generator.
Simulates a complete train journey with acceleration, cruising, coasting, and braking phases.
Output: JSON array of telemetry data points with energy labels.
"""
import json
import math
import random
import argparse
import sys
from dataclasses import dataclass, field


@dataclass
class TrainParams:
    max_speed: float = 80.0       # km/h
    acceleration: float = 0.5     # m/s^2
    brake_deceleration: float = 0.8  # m/s^2
    initial_soc: float = 85.0     # %
    eta: float = 0.85             # powertrain efficiency
    mass: float = 120.0           # tonnes
    max_tractive_force: float = 300.0  # kN
    max_brake_force: float = 200.0     # kN


@dataclass
class RunConfig:
    total_distance: float = 80.0  # km
    sample_interval: float = 1.0  # seconds
    noise_std: float = 0.02       # 2% sensor noise
    seed: int = 42


def generate_speed_profile(params: TrainParams, config: RunConfig) -> list[float]:
    """Generate a speed profile (m/s) with accel → cruise → coast → brake phases."""
    max_speed_ms = params.max_speed / 3.6
    accel_time = max_speed_ms / params.acceleration
    brake_time = max_speed_ms / params.brake_deceleration

    # Phase durations (randomized slightly)
    accel_dur = accel_time * random.uniform(0.8, 1.2)
    cruise_dur = random.uniform(600, 1800)
    coast_dur = random.uniform(30, 120)
    brake_dur = brake_time * random.uniform(0.8, 1.2)

    speeds = []
    dt = config.sample_interval
    t = 0.0

    # Acceleration phase
    while t < accel_dur:
        v = params.acceleration * t
        speeds.append(min(v, max_speed_ms))
        t += dt

    cruise_speed = speeds[-1] if speeds else max_speed_ms

    # Cruise phase
    cruise_start_t = t
    while t < cruise_start_t + cruise_dur:
        speeds.append(cruise_speed + random.uniform(-0.5, 0.5))
        t += dt

    # Coast phase (no tractive force)
    coast_start_t = t
    coast_start_speed = speeds[-1]
    while t < coast_start_t + coast_dur:
        frac = (t - coast_start_t) / coast_dur
        v = coast_start_speed * (1.0 - 0.3 * frac)
        speeds.append(max(v, 0.0))
        t += dt

    # Braking phase
    brake_start_t = t
    brake_start_speed = speeds[-1]
    while t < brake_start_t + brake_dur and speeds[-1] > 0:
        frac = (t - brake_start_t) / brake_dur
        v = brake_start_speed * (1.0 - frac)
        speeds.append(max(v, 0.0))
        t += dt

    # Final zero
    speeds[-1] = 0.0

    return speeds


def generate_run(params: TrainParams = None, config: RunConfig = None,
                 run_id: int = 0) -> list[dict]:
    """Generate a complete train run with telemetry and energy labels."""
    if params is None:
        params = TrainParams()
    if config is None:
        config = RunConfig()

    random.seed(config.seed + run_id)
    speeds = generate_speed_profile(params, config)

    data_points = []
    cumulative_position = 0.0  # m
    cumulative_energy = 0.0    # kWh
    soc = params.initial_soc
    soc_energy_capacity = 1000.0  # kWh (battery rated energy)
    eta = params.eta
    dt = config.sample_interval / 3600.0  # hours

    for i, speed_ms in enumerate(speeds):
        time_ms = int(i * config.sample_interval * 1000)

        # Determine phase based on acceleration
        if i < len(speeds) - 1:
            accel = (speeds[i + 1] - speed_ms) / config.sample_interval
        else:
            accel = 0.0

        # Tractive force / brake force
        tractive_force = 0.0
        electric_brake_force = 0.0

        if accel > 0.01 and speed_ms > 0.1:
            # Acceleration: tractive force
            tractive_force = min(params.mass * accel + 20.0, params.max_tractive_force)
        elif accel < -0.01 and speed_ms > 0.1:
            # Braking: electric brake force
            electric_brake_force = min(params.mass * abs(accel), params.max_brake_force)

        # Battery power: discharge in traction, charge in braking
        mech_power = tractive_force * speed_ms - electric_brake_force * speed_ms
        if mech_power > 0:
            dc_power = mech_power / eta
            battery_power = dc_power * 0.7  # battery provides ~70% of DC power
        elif mech_power < 0:
            dc_power = mech_power * eta
            battery_power = dc_power * 0.7  # battery absorbs ~70% of regen
        else:
            battery_power = random.uniform(-10, 30)  # auxiliary load only

        # SOC-based limits
        if battery_power > 0 and soc <= 10.0:
            battery_power = 0.0
        if battery_power < 0 and soc >= 99.0:
            battery_power = 0.0

        # Energy accumulation
        cat_power = dc_power - battery_power if 'dc_power' in dir() else 0
        p_consume = max(cat_power, 0) + max(battery_power, 0)
        p_regen = abs(min(cat_power, 0)) + abs(min(battery_power, 0))
        net_power = cat_power + battery_power

        cumulative_energy += net_power * dt

        # SOC update
        soc -= (battery_power * dt / soc_energy_capacity) * 100.0
        soc = max(0.0, min(100.0, soc))

        # Position update
        cumulative_position += speed_ms * config.sample_interval

        # Add sensor noise
        speed_noisy = speed_ms + random.gauss(0, params.max_speed / 3.6 * config.noise_std)
        tractive_noisy = tractive_force + random.gauss(0, 20 * config.noise_std)
        brake_noisy = electric_brake_force + random.gauss(0, 10 * config.noise_std)
        bat_noisy = battery_power + random.gauss(0, 20 * config.noise_std)

        data_points.append({
            "time": time_ms,
            "speed": round(max(0, speed_noisy), 3),
            "tractive_force": round(max(0, tractive_noisy), 2),
            "electric_brake_force": round(max(0, brake_noisy), 2),
            "battery_power": round(bat_noisy, 2),
            "soc": round(soc, 2),
            "position": round(cumulative_position / 1000.0, 6),  # km
            "energy": round(max(0, cumulative_energy), 6),       # kWh cumulative
        })

    return data_points


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic train run data")
    parser.add_argument("--runs", type=int, default=100, help="Number of runs to generate")
    parser.add_argument("--output", type=str, default="synthetic_data.json", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    all_runs = []
    for i in range(args.runs):
        params = TrainParams(
            max_speed=random.uniform(60, 140),
            acceleration=random.uniform(0.3, 0.8),
            brake_deceleration=random.uniform(0.5, 1.0),
            initial_soc=random.uniform(50, 100),
        )
        config = RunConfig(
            total_distance=random.uniform(50, 200),
            noise_std=random.uniform(0.01, 0.03),
            seed=args.seed + i,
        )
        data = generate_run(params, config, run_id=i)
        all_runs.append({
            "run_id": i,
            "params": {
                "max_speed_kmh": params.max_speed,
                "acceleration": params.acceleration,
                "initial_soc": params.initial_soc,
            },
            "data": data,
        })

    with open(args.output, "w") as f:
        json.dump(all_runs, f, indent=2)

    total_points = sum(len(r["data"]) for r in all_runs)
    print(f"Generated {args.runs} runs, {total_points} total data points → {args.output}")


if __name__ == "__main__":
    main()
