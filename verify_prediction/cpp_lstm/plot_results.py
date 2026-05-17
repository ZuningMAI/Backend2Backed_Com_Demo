"""Plot C++ LSTM results from prediction_results.json"""
import json, sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    jf = sys.argv[1] if len(sys.argv)>1 else "prediction_results.json"
    with open(jf) as f: data = json.load(f)

    out = Path("plots_cpp")
    out.mkdir(exist_ok=True)
    for f in out.glob("*.png"): f.unlink()

    windows = data["windows"]
    total_ms = data.get("total_time_ms",0)
    avg_loss = data.get("avg_loss",0)

    print(f"Windows: {len(windows)}, total_time: {total_ms}ms, avg_loss: {avg_loss:.6f}")
    print(f"Per window: {total_ms/max(1,len(windows)):.1f}ms")

    # ── Statistics ──
    pos_errs = [w["pos_err_m"] for w in windows]
    ene_errs = [w["ene_err"] for w in windows]
    ene_maes = [w["ene_mae"] for w in windows]
    losses = [w["loss"] for w in windows]

    print(f"\n{'='*50}")
    print(f"Prediction Statistics ({len(windows)} windows)")
    print(f"{'='*50}")
    print(f"  Time: {total_ms}ms total, {total_ms/max(1,len(windows)):.1f}ms/win")
    print(f"  Loss: mean={np.mean(losses):.6f}, final={losses[-1]:.6f}")
    print(f"  Pos err (m):  mean={np.mean(pos_errs):.2f}, max={np.max(pos_errs):.2f}")
    print(f"  Ene err (kWh): mean={np.mean(ene_errs):.6f}, max={np.max(ene_errs):.6f}")
    print(f"  Ene MAE (kWh): mean={np.mean(ene_maes):.6f}")

    # ── Plot each window ──
    for i, w in enumerate(windows):
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(w["actual_pos"], w["actual_ene"], 'b-', lw=1.5,
                label=f'Actual [{w[\"t_start\"]},{w[\"t_end\"]}]')
        ax.plot(w["future_pos"], w["future_ene"], '-', color='green', lw=1.2, alpha=0.7,
                label=f'Actual [{w[\"pred_start\"]},{w[\"pred_end\"]}]')
        ax.plot(w["pred_pos"], w["pred_ene"], '--', color='orange', lw=1.8,
                label=f'Predicted [{w[\"pred_start\"]},{w[\"pred_end\"]}]')
        ax.axvline(x=w["actual_pos"][-1], color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Position (km)'); ax.set_ylabel('Cumulative Energy (kWh)')
        ax.set_title(f'C++ LSTM: [{w[\"t_start\"]},{w[\"t_end\"]}]ms ({w[\"pos_err_m\"]:.1f}m, {w[\"ene_err\"]:.4f}kWh)')
        ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out/f"win_{i+1:02d}_t{w['t_start']}_{w['t_end']}ms.png", dpi=120)
        plt.close(fig)

    # ── Loss curve ──
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(losses, lw=0.5, alpha=0.7)
    ax.set_xlabel('Step'); ax.set_ylabel('MSE Loss')
    ax.set_title('C++ LSTM Online Training Loss')
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out/"loss_curve.png", dpi=120); plt.close(fig)

    # ── Error distribution ──
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    ax1.hist(pos_errs, bins=20, color='steelblue', edgecolor='white')
    ax1.set_xlabel('Position Error (m)'); ax1.set_ylabel('Count')
    ax1.set_title(f'Pos Error (mean={np.mean(pos_errs):.2f}m)')
    ax2.hist(ene_errs, bins=20, color='orange', edgecolor='white')
    ax2.set_xlabel('Energy Error (kWh)'); ax2.set_title(f'Ene Error (mean={np.mean(ene_errs):.6f}kWh)')
    fig.tight_layout(); fig.savefig(out/"error_dist.png", dpi=120); plt.close(fig)

    print(f"  Saved {len(windows)} plots + loss_curve + error_dist to {out}/")

    # ── Comparison with Python ──
    with open(out/"comparison.txt","w") as f:
        f.write(f"C++ LSTM ({len(windows)} windows)\n")
        f.write(f"  total_time_ms: {total_ms}\n")
        f.write(f"  ms_per_win: {total_ms/max(1,len(windows)):.1f}\n")
        f.write(f"  avg_loss: {avg_loss:.6f}\n")
        f.write(f"  pos_err_mean_m: {np.mean(pos_errs):.2f}\n")
        f.write(f"  pos_err_max_m: {np.max(pos_errs):.2f}\n")
        f.write(f"  ene_err_mean_kwh: {np.mean(ene_errs):.6f}\n")
        f.write(f"  ene_mae_mean_kwh: {np.mean(ene_maes):.6f}\n")

if __name__=="__main__": main()
