"""
Online energy prediction algorithm verification.
Simulates the 100ms communication cycle using real CSV data.

Flow:
  - CSV → 1ms interpolation → derive fields
  - Every 100ms: receive 100 1ms data points
  - After 800ms: start online training + prediction
  - Each prediction: 800 history pts → MLP train → predict 200 future pts
  - Save stage-by-stage plots showing the process
"""
import csv
import re
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Train params ──
ETA = 0.85
BAT_CAP = 200.0
BAT_TR = 0.30
BAT_RE = 0.50
AUX = 120.0
SOC0 = 80.0

# ── MLP (same as Backend 2, in numpy for verification) ──
class TinyMLP:
    def __init__(self):
        rng = np.random.RandomState(42)
        scale1 = np.sqrt(6.0 / (30 + 128))
        scale2 = np.sqrt(6.0 / (128 + 64))
        scale3 = np.sqrt(6.0 / (64 + 200))
        self.W1 = rng.uniform(-scale1, scale1, (128, 30))
        self.b1 = np.zeros(128)
        self.W2 = rng.uniform(-scale2, scale2, (64, 128))
        self.b2 = np.zeros(64)
        self.W3 = rng.uniform(-scale3, scale3, (200, 64))
        self.b3 = np.zeros(200)

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        self.x = np.array(x, dtype=np.float64).reshape(30)
        self.z1 = self.x @ self.W1.T + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2.T + self.b2
        self.a2 = self._relu(self.z2)
        self.z3 = self.a2 @ self.W3.T + self.b3
        self.a3 = self.z3
        v = self.a3.copy()
        v[np.isnan(v) | np.isinf(v)] = 0.0
        return v

    def train_step(self, x, y, lr=0.0001):
        pred = self.forward(x)
        y = np.array(y, dtype=np.float64)
        diff = pred - y
        mse = float(np.mean(diff * diff))

        if np.isnan(mse) or np.isinf(mse):
            return 1e9

        dL = diff * (2.0 / 200.0)
        dW3 = np.outer(dL, self.a2)
        db3 = dL
        d2 = (dL @ self.W3) * (self.z2 > 0)
        dW2 = np.outer(d2, self.a1)
        db2 = d2
        d1 = (d2 @ self.W2) * (self.z1 > 0)
        dW1 = np.outer(d1, self.x)
        db1 = d1

        self.W3 -= lr * dW3; self.b3 -= lr * db3
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        return mse


# ── Feature extraction (same 30-dim as Backend 2) ──
def extract_features(history):
    n = len(history)
    spd = np.array([p["speed"] for p in history])
    frc = np.array([p["force"] for p in history])
    rte = np.array([p["real_time_energy"] for p in history])
    soc = np.array([p["soc"] for p in history])
    bpw = np.array([p["battery_power"] for p in history])
    mode = np.array([p["mode"] for p in history])

    def sl(v): return (v[-1]-v[0])/(n-1) if n>1 else 0
    def ratio(m): return np.sum(mode==m)/n

    f = np.zeros(30)
    f[0]=np.mean(spd); f[1]=np.std(spd); f[2]=spd[-1]; f[3]=spd[0]
    f[4]=sl(spd); f[5]=np.mean(frc); f[6]=np.std(frc); f[7]=frc[-1]
    f[8]=np.max(frc); f[9]=np.min(frc); f[10]=sl(frc)
    f[11]=np.mean(rte); f[12]=np.std(rte); f[13]=rte[-1]
    f[14]=np.mean(soc); f[15]=np.std(soc); f[16]=soc[-1]
    f[17]=np.mean(bpw); f[18]=np.std(bpw); f[19]=np.max(bpw); f[20]=np.min(bpw)
    f[21]=ratio(0); f[22]=ratio(2); f[23]=ratio(3); f[24]=ratio(1)
    f[25]=spd[0]; f[26]=spd[-1]; f[27]=rte[0]; f[28]=rte[-1]; f[29]=sl(rte)
    return f


# ── CSV loading + 1ms interpolation + derivation ──
def load_csv(filepath):
    data = {"relative_time":[],"position":[],"speed":[],"force":[],"mode":[]}
    with open(filepath) as f:
        meta = f.readline().strip()
        f.readline()
        for row in csv.reader(f):
            if len(row)<5: continue
            data["relative_time"].append(float(row[0]))
            data["position"].append(float(row[1]))
            data["speed"].append(float(row[2]))
            data["force"].append(float(row[3]))
            data["mode"].append(int(row[4]))
    for k in data: data[k] = np.array(data[k])
    return meta, data


def interpolate_1ms(data, duration_s):
    n = int(duration_s * 1000)
    t_raw = data["relative_time"]
    t_grid = np.arange(n) / 1000.0
    interp = {}
    for key in ["position","speed","force"]:
        interp[key] = np.interp(t_grid, t_raw, data[key])
    interp["mode"] = np.zeros(n, dtype=int)
    for i, tg in enumerate(t_grid):
        idx = np.searchsorted(t_raw, tg)
        interp["mode"][i] = int(data["mode"][min(idx, len(data["mode"])-1)])
    return interp


def derive(interp, soc0=SOC0):
    n = len(interp["speed"])
    trf = np.zeros(n); ebf = np.zeros(n); bpw = np.zeros(n)
    soc = np.zeros(n); rte = np.zeros(n)
    sv = soc0; dt_h = 0.001/3600.0

    for i in range(n):
        f = interp["force"][i]; s = interp["speed"][i]; sms = s/3.6
        if f>0: trf[i]=f; ebf[i]=0
        elif f<0: trf[i]=0; ebf[i]=abs(f)
        else: trf[i]=0; ebf[i]=0
        pm = f*sms
        pdc = pm/ETA if pm>0 else (pm*ETA if pm<0 else 0)
        bat = pdc*BAT_TR if pdc>0 else (pdc*BAT_RE if pdc<0 else -AUX)
        if bat<0 and sv>=99: bat=0
        if bat>0 and sv<=10: bat=0
        bpw[i]=bat
        sv -= (bat*dt_h)/BAT_CAP*100; sv=max(0,min(100,sv)); soc[i]=sv
        pc = max(pdc-bat,0)+max(bat,0)
        rte[i] = pc/(s+0.01) if s>0.01 else 0
    return {"tractive_force":trf,"electric_brake_force":ebf,"battery_power":bpw,"soc":soc,"real_time_energy":rte}


# ── Main ──
def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "/home/maksuning/Proj/API_COM/verify_prediction/OptReslog.2026-05-09_20-10-54-FZ602-1-2-86.csv"
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    # Clear old plots
    for f in out_dir.glob("*.png"): f.unlink()

    print(f"Loading: {csv_file}")
    meta, raw = load_csv(csv_file)
    dur = float(re.search(r'设定时间：(\d+)', meta).group(1))
    print(f"  Duration: {dur}s, points: {len(raw['relative_time'])}")

    print("Interpolating to 1ms...")
    interp = interpolate_1ms(raw, dur)
    d = derive(interp)
    total = len(interp["speed"])
    print(f"  {total} points ({total/1000:.0f}s)")

    # Build combined data list
    pts = []
    for i in range(total):
        pts.append({
            "time_ms": i,
            "position": float(interp["position"][i]),
            "speed": float(interp["speed"][i]),
            "force": float(interp["force"][i]),
            "mode": int(interp["mode"][i]),
            "tractive_force": float(d["tractive_force"][i]),
            "electric_brake_force": float(d["electric_brake_force"][i]),
            "battery_power": float(d["battery_power"][i]),
            "soc": float(d["soc"][i]),
            "real_time_energy": float(d["real_time_energy"][i]),
        })

    # ═══════════════════════════════════════════════════════════════
    # Plot 1: Position-Energy curve (full section)
    # ═══════════════════════════════════════════════════════════════
    print("Generating position-energy curve...")
    pos_km = [p["position"] / 1000 for p in pts]
    cum_energy = []
    ce = 0.0
    for p in pts:
        ce += p["real_time_energy"] * max(p["speed"], 0.1) * 0.001 / 3600.0
        cum_energy.append(ce)

    fig1, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(pos_km, cum_energy, 'b-', lw=1.2)
    ax1.set_xlabel('Position (km)')
    ax1.set_ylabel('Cumulative Energy (kWh)')
    ax1.set_title(f'Position-Energy Curve (full section, {dur}s)')
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(out_dir / "pos_energy_curve.png", dpi=150)
    plt.close(fig1)
    print(f"  Saved: pos_energy_curve.png")

    # ═══════════════════════════════════════════════════════════════
    # Plot 2: Energy-Time-Position 3D plot
    # ═══════════════════════════════════════════════════════════════
    print("Generating 3D energy-time-position plot...")
    fig2 = plt.figure(figsize=(14, 9))
    ax3d = fig2.add_subplot(111, projection='3d')

    t_sec = [p["time_ms"] / 1000 for p in pts]
    # Downsample for 3D clarity (every 100th point)
    step = max(1, len(pts) // 5000)
    t_ds = t_sec[::step]
    p_ds = pos_km[::step]
    e_ds = cum_energy[::step]

    # Color by speed
    spd_ds = [pts[i]["speed"] for i in range(0, len(pts), step)]

    scatter = ax3d.scatter(t_ds, p_ds, e_ds, c=spd_ds, cmap='viridis',
                           s=1, alpha=0.8)
    ax3d.plot(t_ds, p_ds, e_ds, 'k-', lw=0.3, alpha=0.3)
    ax3d.set_xlabel('Time (s)')
    ax3d.set_ylabel('Position (km)')
    ax3d.set_zlabel('Cumulative Energy (kWh)')
    ax3d.set_title(f'Energy-Time-Position 3D View (full section, {dur}s)')
    cbar = fig2.colorbar(scatter, ax=ax3d, shrink=0.5, pad=0.1)
    cbar.set_label('Speed (km/h)')

    fig2.tight_layout()
    fig2.savefig(out_dir / "energy_3d.png", dpi=150)
    plt.close(fig2)
    print(f"  Saved: energy_3d.png")

    # ═══════════════════════════════════════════════════════════════
    # Plot 3: Sliding window pos-energy curves WITH prediction overlay
    #   Window [t, t+800] actual + predict [t+800, t+1000]
    # ═══════════════════════════════════════════════════════════════
    print("Generating sliding window curves with prediction overlay...")
    window_dir = out_dir / "sliding_windows_com"
    window_dir.mkdir(exist_ok=True)
    for f in window_dir.glob("*.png"): f.unlink()

    win_size = 800
    win_times = list(range(0, 500, 100))
    win_times += list(range(1000, total - win_size + 1, 2000))
    if win_times[-1] != total - win_size:
        win_times.append(total - win_size)

    # Run online MLP across all windows to get predictions
    mlp = TinyMLP()
    hist_buf = []

    # Pre-compute predictions for all windows
    predictions = {}  # t_start -> (predicted positions, predicted energies)
    for t_cur in range(0, total, 100):
        batch = pts[t_cur:t_cur+100]
        hist_buf.extend(batch)
        if len(hist_buf) > 2000:
            hist_buf = hist_buf[-2000:]

        if t_cur >= 800 and len(hist_buf) >= 800:
            td = hist_buf[-800:]
            feats = extract_features(td)
            tgt_start = t_cur + 100
            tgt_end = tgt_start + 200
            if tgt_end <= total:
                tgt = [pts[i]["real_time_energy"] for i in range(tgt_start, min(tgt_end, total))]
                if len(tgt) < 200:
                    tgt.extend([tgt[-1]] * (200 - len(tgt)))
            else:
                tgt = [td[-1]["real_time_energy"]] * 200
            mlp.train_step(feats, tgt, 0.0001)

            # Store prediction for this t_cur
            if t_cur in win_times:
                pred_rte = mlp.forward(feats)
                last_p = td[-1]["position"] / 1000
                last_s = td[-1]["speed"]
                # cumulative energy at end of window
                win_pts = hist_buf[-800:]
                ce = 0.0
                for p in win_pts:
                    ce += p["real_time_energy"] * max(p["speed"], 0.1) * 0.001 / 3600.0

                pc_pos = [last_p]
                pc_ene = [ce]
                cp = last_p
                ce2 = ce
                for i in range(200):
                    r = pred_rte[i] if i < len(pred_rte) else 0
                    r = max(0, r) if not np.isnan(r) else 0
                    cp += max(last_s, 0.1) * 0.001 / 3.6
                    ce2 += r * max(last_s, 0.1) * 0.001 / 3600.0
                    pc_pos.append(cp)
                    pc_ene.append(ce2)
                predictions[t_cur] = (pc_pos, pc_ene, float(ce))

    # Now plot each window with prediction overlay
    for t_start in win_times:
        t_end = t_start + win_size
        w_pts = pts[t_start:t_end]

        pos_w = [p["position"] / 1000 for p in w_pts]
        ce = 0.0
        ene_w = []
        for p in w_pts:
            ce += p["real_time_energy"] * max(p["speed"], 0.1) * 0.001 / 3600.0
            ene_w.append(ce)

        fig_w, ax_w = plt.subplots(figsize=(12, 5))

        # Actual curve
        ax_w.plot(pos_w, ene_w, 'b-', lw=1.5, label='Actual')
        last_actual_pos = pos_w[-1] if pos_w else 0
        last_actual_ene = ene_w[-1] if ene_w else 0

        # Prediction overlay
        if t_start in predictions:
            pc_pos, pc_ene, _ = predictions[t_start]
            ax_w.plot(pc_pos, pc_ene, '--', color='orange', lw=1.8, label=f'Predicted [{t_end}, {t_end+200}]')
            ax_w.axvline(x=last_actual_pos, color='gray', ls=':', alpha=0.6)
            ax_w.annotate('Predict→', xy=(last_actual_pos, last_actual_ene),
                         xytext=(last_actual_pos + 0.0005, last_actual_ene * 1.02),
                         fontsize=8, color='gray')

        ax_w.set_xlabel('Position (km)')
        ax_w.set_ylabel('Cumulative Energy (kWh)')
        ax_w.set_title(f'Window [{t_start}, {t_end}]ms → predict [{t_end}, {t_end+200}]ms')
        ax_w.legend(loc='upper left')
        ax_w.grid(True, alpha=0.3)
        fig_w.tight_layout()
        idx = win_times.index(t_start) + 1
        fig_w.savefig(window_dir / f"win_{idx:02d}_t{t_start}_{t_end}ms.png", dpi=120)
        plt.close(fig_w)

    print(f"  Saved {len(list(window_dir.glob('*.png')))} sliding window plots with prediction to {window_dir}/")

    # ═══════════════════════════════════════════════════════════════
    # Online prediction stage snapshots (separate timeline plots)
    # ═══════════════════════════════════════════════════════════════
    history_buffer = []

    # Snapshot stages (in ms): at 800ms, 2000ms, 5000ms, 10000ms, ...
    snapshot_times = [800, 2000, 5000, 10000, 20000, 40000, 80000, 120000, 186000]
    snapshot_times = [t for t in snapshot_times if t < total]
    snapshot_idx = 0
    plots_saved = 0

    # Simulate 100ms communication cycles
    for t_batch in range(0, total, 100):
        batch = pts[t_batch:t_batch+100]
        history_buffer.extend(batch)
        if len(history_buffer) > 2000:
            history_buffer = history_buffer[-2000:]

        # Prediction: every 100ms after 800ms cold start
        if t_batch >= 800 and len(history_buffer) >= 800:
            # Use latest 800 points for training+prediction
            train_data = history_buffer[-800:]
            features = extract_features(train_data)

            # Target: next 200ms of real_time_energy
            target_start = t_batch + 100
            target_end = target_start + 200
            if target_end <= total:
                target = [pts[i]["real_time_energy"] for i in range(target_start, min(target_end, total))]
                if len(target) < 200:
                    target.extend([target[-1]] * (200 - len(target)))
            else:
                target = [train_data[-1]["real_time_energy"]] * 200

            loss = mlp.train_step(features, target, 0.0001)

        # Generate snapshot at designated times
        if snapshot_idx < len(snapshot_times) and t_batch >= snapshot_times[snapshot_idx]:
            t_ms = snapshot_times[snapshot_idx]
            snapshot_idx += 1

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            # ── Top: speed & force over time ──
            hist = history_buffer[-min(1000, len(history_buffer)):]
            t_hist = [p["time_ms"]/1000 for p in hist]
            ax1b = ax1.twinx()
            ax1.plot(t_hist, [p["speed"] for p in hist], 'b-', lw=1, alpha=0.7, label='Speed (km/h)')
            ax1b.plot(t_hist, [p["force"] for p in hist], 'r-', lw=1, alpha=0.5, label='Force (kN)')
            ax1.set_ylabel('Speed (km/h)', color='b')
            ax1b.set_ylabel('Force (kN)', color='r')
            ax1.set_title(f'Speed & Force (t={t_ms}ms)')
            ax1.grid(True, alpha=0.3)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1b.get_legend_handles_labels()
            ax1.legend(lines1+lines2, labels1+labels2, loc='upper right')

            # ── Bottom: actual vs predicted energy-position curve ──
            # Actual curve (from history)
            ac_pos = [p["position"]/1000 for p in history_buffer]
            ac_ene = []
            cum = 0.0
            for p in history_buffer:
                cum += p["real_time_energy"] * max(p["speed"], 0.1) * 0.001 / 3600.0
                ac_ene.append(cum)

            ax2.plot(ac_pos, ac_ene, 'b-', lw=2, label='Actual energy curve')

            # Predicted curve (MLP output)
            if t_ms >= 800:
                train_data2 = history_buffer[-800:]
                features2 = extract_features(train_data2)
                pred_rte = mlp.forward(features2)
                last_pos = train_data2[-1]["position"] / 1000
                last_spd = train_data2[-1]["speed"]
                last_ene = ac_ene[-1] if ac_ene else 0
                cum_pos = last_pos
                cum_ene = last_ene
                pc_pos = [cum_pos]
                pc_ene = [cum_ene]
                for i in range(200):
                    r = pred_rte[i] if i<len(pred_rte) else 0
                    r = max(0, r) if not np.isnan(r) else 0
                    cum_pos += max(last_spd, 0.1) * 0.001 / 3.6
                    cum_ene += r * max(last_spd, 0.1) * 0.001 / 3600.0
                    pc_pos.append(cum_pos)
                    pc_ene.append(cum_ene)
                ax2.plot(pc_pos, pc_ene, '--', color='orange', lw=2, label='Predicted (MLP)')

                # Mark prediction start point
                ax2.axvline(x=last_pos, color='gray', ls=':', alpha=0.5)
                ax2.annotate('Predict starts', xy=(last_pos, last_ene),
                            xytext=(last_pos+0.002, last_ene*1.1),
                            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)

            ax2.set_xlabel('Position (km)')
            ax2.set_ylabel('Cumulative Energy (kWh)')
            ax2.set_title(f'Energy Curve @ t={t_ms}ms ({t_ms/1000:.1f}s)')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            fname = out_dir / f"stage_{plots_saved+1:02d}_t{t_ms}ms.png"
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            plots_saved += 1
            print(f"  Saved: {fname}")

    print(f"\nDone: {plots_saved} plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
