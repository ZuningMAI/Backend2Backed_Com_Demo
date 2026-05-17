"""
LSTM-based online energy prediction verification.
Input:  800ms time-series (10 features per 1ms step)
Output: 200ms future time-series (speed + real_time_energy per step)
        → position derived from speed integration
        → energy derived from rte * speed integration

Model: 1-layer LSTM (128 hidden) → Linear(128, 2)
Online: each 800ms window, do 1-step gradient update, then predict.
"""
import csv, re, sys, math, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ── Train params ──
ETA = 0.85; BAT_CAP = 200.0; BAT_TR = 0.30; BAT_RE = 0.50; AUX = 120.0; SOC0 = 80.0

# ── CSV → 1ms → derive ──
def load_csv(filepath):
    data = {"rt":[],"pos":[],"spd":[],"frc":[],"mod":[]}
    with open(filepath) as f:
        meta = f.readline().strip(); f.readline()
        for row in csv.reader(f):
            if len(row)<5: continue
            data["rt"].append(float(row[0])); data["pos"].append(float(row[1]))
            data["spd"].append(float(row[2])); data["frc"].append(float(row[3]))
            data["mod"].append(int(row[4]))
    for k in data: data[k]=np.array(data[k])
    return meta, data

def interp_1ms(data, dur_s):
    n=int(dur_s*1000); t_raw=data["rt"]; tg=np.arange(n)/1000.0
    out={}
    for k in ["pos","spd","frc"]: out[k]=np.interp(tg, t_raw, data[k])
    out["mod"]=np.zeros(n,dtype=int)
    for i,g in enumerate(tg):
        idx=np.searchsorted(t_raw,g); out["mod"][i]=int(data["mod"][min(idx,len(data["mod"])-1)])
    out["time_ms"]=np.arange(n)
    return out

def derive(interp, soc0=SOC0):
    n=len(interp["spd"]); trf=np.zeros(n); ebf=np.zeros(n); bpw=np.zeros(n)
    soc=np.zeros(n); rte=np.zeros(n); sv=soc0; dt_h=0.001/3600.0
    for i in range(n):
        f=interp["frc"][i]; s=interp["spd"][i]; sms=s/3.6
        if f>0: trf[i]=f; ebf[i]=0
        elif f<0: trf[i]=0; ebf[i]=abs(f)
        else: trf[i]=0; ebf[i]=0
        pm=f*sms; pdc=pm/ETA if pm>0 else (pm*ETA if pm<0 else 0)
        bat=pdc*BAT_TR if pdc>0 else (pdc*BAT_RE if pdc<0 else -AUX)
        if bat<0 and sv>=99: bat=0
        if bat>0 and sv<=10: bat=0
        bpw[i]=bat; sv-=(bat*dt_h)/BAT_CAP*100; sv=max(0,min(100,sv)); soc[i]=sv
        pc=max(pdc-bat,0)+max(bat,0); rte[i]=pc/(s+0.01) if s>0.01 else 0
    return {"trf":trf,"ebf":ebf,"bpw":bpw,"soc":soc,"rte":rte}

# ── LSTM Model ──
class EnergyLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden=128, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, 1, batch_first=True)
        self.fc = nn.Linear(hidden, output_dim)

    def forward(self, x):
        # x: (batch, 800, input_dim)
        out, _ = self.lstm(x)
        # Take last hidden state for each time step to predict each future step
        # Actually decode each time step's hidden state
        out = self.fc(out)  # (batch, 800, 2)
        # For prediction: use last hidden state → project to 200 steps
        # During training we use teacher forcing; for prediction we roll out
        return out

class SeqPredictor(nn.Module):
    """LSTM encoder → autoregressive decoder for 200-step prediction."""
    def __init__(self, input_dim=10, hidden=128, output_dim=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden, 1, batch_first=True)
        self.decoder = nn.LSTMCell(output_dim, hidden)
        self.fc = nn.Linear(hidden, output_dim)
        self.hidden = hidden

    def encode(self, x):
        # x: (batch, 800, input_dim) → (h_n, c_n)
        _, (h_n, c_n) = self.encoder(x)
        return h_n.squeeze(0), c_n.squeeze(0)  # (batch, hidden)

    def forward(self, x, targets=None, teacher_forcing=0.5):
        # x: (batch, 800, input_dim), targets: (batch, 200, output_dim) or None
        batch = x.size(0)
        h, c = self.encode(x)
        outputs = []
        # First decoder input: last target value or last input's speed+rte
        inp = x[:, -1, 1:3] if targets is None else targets[:, 0, :]  # (batch, 2)
        if inp.dim() == 1: inp = inp.unsqueeze(0)

        for t in range(200):
            h, c = self.decoder(inp, (h, c))
            out = self.fc(h)  # (batch, 2)
            outputs.append(out)
            if targets is not None and torch.rand(1).item() < teacher_forcing:
                inp = targets[:, t, :]
            else:
                inp = out
        return torch.stack(outputs, dim=1)  # (batch, 200, 2)


# ── Features per timestep ──
def make_features(pts_slice):
    """
    pts_slice: list of dicts with keys: time_ms, position, speed, force, mode,
               tractive_force, electric_brake_force, battery_power, soc, real_time_energy
    Returns: (N, 10) numpy array
    """
    feats = []
    for p in pts_slice:
        feats.append([
            p["position"] / 1000.0,  # km
            p["speed"] / 100.0,      # normalize
            p["force"] / 300.0,      # normalize
            p["mode"] / 4.0,
            p["tractive_force"] / 300.0,
            p["electric_brake_force"] / 300.0,
            p["battery_power"] / 500.0,
            p["soc"] / 100.0,
            p["real_time_energy"] / 100.0,
            p["time_ms"] / 100000.0,
        ])
    return np.array(feats, dtype=np.float32)

def make_targets(pts_slice):
    """pts_slice: 200 future points. Returns (200, 2): speed, real_time_energy"""
    tgt = []
    for p in pts_slice:
        tgt.append([
            p["speed"] / 100.0,
            p["real_time_energy"] / 100.0,
        ])
    return np.array(tgt, dtype=np.float32)

def compute_curve(pts, start_pos_km=0, start_energy=0):
    """Convert list of dicts to (position_km, cumulative_energy) arrays."""
    pos = [start_pos_km]
    ene = [start_energy]
    cp, ce = start_pos_km, start_energy
    for p in pts:
        s_ms = p["speed"] / 3.6
        cp += s_ms * 0.001 / 1000.0  # m/s * 1ms → km
        ce += p["real_time_energy"] * max(p["speed"], 0.1) * 0.001 / 3600.0
        pos.append(cp)
        ene.append(ce)
    return pos, ene

def predict_curve_from_output(output, last_pt, last_pos_km, last_energy):
    """
    output: (200, 2) predicted [speed_norm, rte_norm]
    Returns: (positions_km, energies) arrays of length 201 (start + 200 steps)
    """
    cp = last_pos_km
    ce = last_energy
    pos = [cp]
    ene = [ce]
    last_s = last_pt["speed"]
    for i in range(200):
        spd = output[i, 0] * 100.0  # denorm
        rte = output[i, 1] * 100.0  # denorm
        spd = max(0.1, min(spd, 200.0))
        rte = max(0.0, min(rte, 200.0))
        cp += spd / 3.6 * 0.001 / 1000.0
        ce += rte * spd * 0.001 / 3600.0
        pos.append(cp)
        ene.append(ce)
    return pos, ene


# ── Main ──
def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "/home/maksuning/Proj/API_COM/verify_prediction/OptReslog.2026-05-09_20-10-54-FZ602-1-2-86.csv"
    out_dir = Path("plots_lstm")
    out_dir.mkdir(exist_ok=True)
    win_dir = out_dir / "sliding_windows"
    win_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*.png"): f.unlink()
    for f in win_dir.glob("*.png"): f.unlink()

    print(f"Loading: {csv_file}")
    meta, raw = load_csv(csv_file)
    dur = float(re.search(r'设定时间：(\d+)', meta).group(1))
    print(f"  {dur}s, {len(raw['rt'])} raw points")

    interp = interp_1ms(raw, dur)
    d = derive(interp)
    total = len(interp["spd"])
    print(f"  {total} 1ms points")

    # Build point list
    pts = []
    for i in range(total):
        pts.append({
            "time_ms": i, "position": float(interp["pos"][i]),
            "speed": float(interp["spd"][i]), "force": float(interp["frc"][i]),
            "mode": int(interp["mod"][i]),
            "tractive_force": float(d["trf"][i]), "electric_brake_force": float(d["ebf"][i]),
            "battery_power": float(d["bpw"][i]), "soc": float(d["soc"][i]),
            "real_time_energy": float(d["rte"][i]),
        })

    # ── LSTM model ──
    device = torch.device("cpu")
    model = SeqPredictor(input_dim=10, hidden=128, output_dim=2).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Online LSTM training + prediction...")

    win_size = 800
    pred_size = 200
    win_times = list(range(0, 500, 100))
    win_times += list(range(1000, total - win_size + 1, 2000))
    if win_times[-1] != total - win_size:
        win_times.append(total - win_size)

    predictions = {}  # t_start → (pred_pos, pred_ene, pred_ene_raw, actual_curve_points)
    losses = []

    # Online loop: 100ms batches
    for t_cur in range(0, total - pred_size, 100):
        if t_cur + win_size + pred_size > total:
            break

        # Training: use [t_cur, t_cur+800] to predict [t_cur+800, t_cur+1000]
        hist = pts[t_cur:t_cur + win_size]
        future = pts[t_cur + win_size:t_cur + win_size + pred_size]

        if len(hist) != win_size or len(future) != pred_size:
            continue

        X = torch.tensor(make_features(hist)).unsqueeze(0).to(device)   # (1, 800, 10)
        Y = torch.tensor(make_targets(future)).unsqueeze(0).to(device)  # (1, 200, 2)

        optimizer.zero_grad()
        pred = model(X, Y, teacher_forcing=0.5)
        loss = nn.MSELoss()(pred, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        # Store prediction for plotting
        if t_cur in win_times:
            model.eval()
            with torch.no_grad():
                pred_out = model(X, None).squeeze(0).cpu().numpy()  # (200, 2)
            model.train()

            last_pt = hist[-1]
            # Actual curve position and energy within window
            actual_pos, actual_ene = compute_curve(hist, start_pos_km=hist[0]["position"]/1000, start_energy=0)

            # Prediction curve
            pred_pos, pred_ene = predict_curve_from_output(
                pred_out, last_pt, actual_pos[-1], actual_ene[-1])

            # Actual future curve [t+800, t+1000] for comparison
            future_pos, future_ene = compute_curve(
                future, start_pos_km=actual_pos[-1], start_energy=actual_ene[-1])

            predictions[t_cur] = (actual_pos, actual_ene, pred_pos, pred_ene, future_pos, future_ene)

    print(f"  Avg loss (last 100): {np.mean(losses[-100:]):.6f}" if len(losses)>=100 else f"  Total steps: {len(losses)}")

    # ── Plot sliding windows + stats ──
    print("Generating sliding window plots with actual comparison...")

    pos_errors = []   # final position error (m)
    ene_errors = []   # final energy error (kWh)
    ene_mae = []      # per-point energy MAE

    for t_start in win_times:
        if t_start not in predictions:
            continue
        actual_pos, actual_ene, pred_pos, pred_ene, future_pos, future_ene = predictions[t_start]
        t_end = t_start + win_size

        # ── Statistics ──
        # Position error at end of prediction (in meters)
        pos_err = abs(pred_pos[-1] - future_pos[-1]) * 1000
        pos_errors.append(pos_err)
        # Energy error at end of prediction
        ene_err = abs(pred_ene[-1] - future_ene[-1])
        ene_errors.append(ene_err)
        # Per-point energy MAE over 200 prediction points
        n_comp = min(len(pred_ene), len(future_ene))
        mae = np.mean([abs(pred_ene[i] - future_ene[i]) for i in range(n_comp)])
        ene_mae.append(mae)

        # ── Plot ──
        fig, ax = plt.subplots(figsize=(12, 5))
        # Actual history
        ax.plot(actual_pos, actual_ene, 'b-', lw=1.5, label=f'Actual [{t_start}, {t_end}]')
        # Actual future (ground truth for comparison)
        ax.plot(future_pos, future_ene, '-', color='green', lw=1.2, alpha=0.7,
                label=f'Actual [{t_end}, {t_end+pred_size}]')
        # Predicted future
        ax.plot(pred_pos, pred_ene, '--', color='orange', lw=1.8,
                label=f'Predicted [{t_end}, {t_end+pred_size}]')
        # Prediction start line
        ax.axvline(x=actual_pos[-1], color='gray', ls=':', alpha=0.5)

        ax.set_xlabel('Position (km)'); ax.set_ylabel('Cumulative Energy (kWh)')
        ax.set_title(f'LSTM: [{t_start},{t_end}]ms → [{t_end},{t_end+pred_size}]ms  (pos_err={pos_err:.1f}m, ene_err={ene_err:.4f}kWh)')
        ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        idx = win_times.index(t_start) + 1
        fig.savefig(win_dir / f"win_{idx:02d}_t{t_start}_{t_end}ms.png", dpi=120)
        plt.close(fig)

    n_plots = len(list(win_dir.glob("*.png")))
    print(f"  Saved {n_plots} plots to {win_dir}/")

    # ── Statistics summary ──
    print(f"\n{'='*60}")
    print(f"Prediction Statistics ({n_plots} windows)")
    print(f"{'='*60}")
    print(f"  Position error (m):  mean={np.mean(pos_errors):.2f}, max={np.max(pos_errors):.2f}, median={np.median(pos_errors):.2f}")
    print(f"  Energy error (kWh):  mean={np.mean(ene_errors):.6f}, max={np.max(ene_errors):.6f}, median={np.median(ene_errors):.6f}")
    print(f"  Energy MAE  (kWh):   mean={np.mean(ene_mae):.6f}, max={np.max(ene_mae):.6f}, median={np.median(ene_mae):.6f}")
    print(f"  Loss (final 100):    {np.mean(losses[-100:]):.8f}" if len(losses)>=100 else f"  Steps: {len(losses)}")

    # Save stats to file
    with open(out_dir / "stats.txt", "w") as f:
        f.write(f"windows: {n_plots}\n")
        f.write(f"pos_err_mean_m: {np.mean(pos_errors):.2f}\n")
        f.write(f"pos_err_max_m: {np.max(pos_errors):.2f}\n")
        f.write(f"ene_err_mean_kwh: {np.mean(ene_errors):.6f}\n")
        f.write(f"ene_err_max_kwh: {np.max(ene_errors):.6f}\n")
        f.write(f"ene_mae_mean_kwh: {np.mean(ene_mae):.6f}\n")

    # ── Loss curve ──
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(losses, lw=0.5, alpha=0.7)
    ax.set_xlabel('Training step'); ax.set_ylabel('MSE Loss')
    ax.set_title('Online LSTM Training Loss')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=120)
    plt.close(fig)
    print(f"  Saved: loss_curve.png")


if __name__ == "__main__":
    main()
