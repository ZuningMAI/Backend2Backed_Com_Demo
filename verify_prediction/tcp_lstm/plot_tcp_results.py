"""Plot TCP LSTM results from tcp_results.json"""
import json, sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    jf = sys.argv[1] if len(sys.argv)>1 else "tcp_results.json"
    with open(jf) as f: data = json.load(f)

    out = Path("plots_tcp")
    out.mkdir(exist_ok=True)
    for f in out.glob("*.png"): f.unlink()

    results = data.get("results", [])
    elapsed = data.get("elapsed_s", 0)
    n = len(results)

    pos_errs = [r["pos_err_m"] for r in results]
    ene_errs = [r["ene_err"] for r in results]
    losses = [r.get("loss", 0) for r in results]

    print(f"TCP LSTM: {n} windows, {elapsed:.1f}s ({elapsed/n*1000:.0f}ms/win)")
    print(f"  Pos err (m):  mean={np.mean(pos_errs):.2f} max={np.max(pos_errs):.2f}")
    print(f"  Ene err (kWh): mean={np.mean(ene_errs):.6f} max={np.max(ene_errs):.6f}")

    # ── Each window plot ──
    for i, r in enumerate(results):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(r["actual_pos"], r["actual_ene"], 'b-', lw=1.5, label='Actual (history)')
        ax.plot(r["future_pos"], r["future_ene"], '-', color='green', lw=1.2, alpha=0.7, label='Actual (future)')
        ax.plot(r["pred_pos"], r["pred_ene"], '--', color='orange', lw=1.8, label='Predicted')
        ax.axvline(x=r["actual_pos"][-1], color='gray', ls=':', alpha=0.5)
        tid = r["window_id"]
        ax.set_xlabel('Position (km)'); ax.set_ylabel('Cumulative Energy (kWh)')
        pe = r["pos_err_m"]; ee = r["ene_err"]
        ax.set_title(f'TCP LSTM: window {tid}ms  (err={pe:.1f}m, {ee:.4f}kWh)')
        ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"win_{i+1:02d}_t{tid}ms.png", dpi=120)
        plt.close(fig)

    # ── Loss curve ──
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(losses, lw=0.5, alpha=0.7)
    ax.set_xlabel('Step'); ax.set_ylabel('MSE Loss'); ax.grid(True, alpha=0.3)
    ax.set_title('TCP LSTM Online Training Loss')
    fig.tight_layout(); fig.savefig(out / "loss_curve.png", dpi=120); plt.close(fig)

    # ── Error distribution ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(pos_errs, bins=20, color='steelblue', edgecolor='white')
    ax1.set_xlabel('Position Error (m)'); ax1.set_title(f'Pos Error (mean={np.mean(pos_errs):.2f}m)')
    ax2.hist(ene_errs, bins=20, color='orange', edgecolor='white')
    ax2.set_xlabel('Energy Error (kWh)'); ax2.set_title(f'Ene Error (mean={np.mean(ene_errs):.6f}kWh)')
    fig.tight_layout(); fig.savefig(out / "error_dist.png", dpi=120); plt.close(fig)

    print(f"  Saved {len(results)} plots + loss + error_dist to {out}/")


if __name__ == "__main__":
    main()
