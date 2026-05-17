"""
Compare data-driven prediction methods (no DL, no physics):
  A: k-NN lookup — find most similar 800ms window → use its future
  B: Online polynomial fit — fit poly to last 200ms → extrapolate 200ms
  C: Template match — match current phase (accel/cruise/coast/brake) → use template
All methods: zero training, pure data-driven from CSV.
"""
import csv, re, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ETA=0.85; BAT_CAP=200.0; BAT_TR=0.30; BAT_RE=0.50; AUX=120.0; SOC0=80.0

def load_csv(fp):
    d={"rt":[],"pos":[],"spd":[],"frc":[],"mod":[]}
    with open(fp) as f:
        meta=f.readline().strip(); f.readline()
        for r in csv.reader(f):
            if len(r)<5: continue
            d["rt"].append(float(r[0])); d["pos"].append(float(r[1]))
            d["spd"].append(float(r[2])); d["frc"].append(float(r[3])); d["mod"].append(int(r[4]))
    for k in d: d[k]=np.array(d[k])
    return meta,d

def interp_1ms(data,dur):
    n=int(dur*1000); tr=data["rt"]; tg=np.arange(n)/1000.0
    o={}; o["mod"]=np.zeros(n,dtype=int)
    for k in ["pos","spd","frc"]: o[k]=np.interp(tg,tr,data[k])
    for i,g in enumerate(tg):
        idx=np.searchsorted(tr,g); o["mod"][i]=int(data["mod"][min(idx,len(data["mod"])-1)])
    return o

def derive(ip,soc0=SOC0):
    n=len(ip["spd"]); trf=np.zeros(n); ebf=np.zeros(n); bpw=np.zeros(n)
    soc=np.zeros(n); rte=np.zeros(n); sv=soc0; dt_h=0.001/3600.0
    for i in range(n):
        f=ip["frc"][i]; s=ip["spd"][i]; sms=s/3.6
        if f>0: trf[i]=f; ebf[i]=0
        elif f<0: trf[i]=0; ebf[i]=abs(f)
        else: trf[i]=0; ebf[i]=0
        pm=f*sms; pdc=pm/ETA if pm>0 else (pm*ETA if pm<0 else 0)
        bat=pdc*BAT_TR if pdc>0 else (pdc*BAT_RE if pdc<0 else -AUX)
        if bat<0 and sv>=99: bat=0
        if bat>0 and sv<=10: bat=0
        bpw[i]=bat; sv-=(bat*dt_h)/BAT_CAP*100; sv=max(0,min(100,sv)); soc[i]=sv
        pc=max(pdc-bat,0)+max(bat,0); rte[i]=pc/(s+0.01) if s>0.01 else 0
    return trf,ebf,bpw,soc,rte

def compute_curve(pts, spk=0, sek=0):
    pos=[spk]; ene=[sek]; cp=spk; ce=sek
    for p in pts:
        cp+=p["speed"]/3.6*0.001/1000.0; ce+=p["real_time_energy"]*max(p["speed"],0.1)*0.001/3600.0
        pos.append(cp); ene.append(ce)
    return pos,ene

# ═══════════════════════════════════════════════════════════
# Method A: k-NN lookup
# ═══════════════════════════════════════════════════════════
class KNNPredictor:
    """Find most similar 800ms window in entire dataset, use its future as prediction."""
    def __init__(self, all_pts):
        # Build index: every possible 800ms window
        self.windows = []
        self.futures = []
        for t in range(0, len(all_pts)-1000, 100):
            w = all_pts[t:t+800]
            f = all_pts[t+800:t+1000]
            if len(w)==800 and len(f)==200:
                # Feature: downsampled speed + force (every 50ms)
                feat = np.array([[w[i]["speed"]/100, w[i]["force"]/300] for i in range(0,800,50)])
                self.windows.append(feat.ravel())  # (32,) vector
                self.futures.append(f)
        self.windows = np.array(self.windows)

    def predict(self, hist):
        feat = np.array([[hist[i]["speed"]/100, hist[i]["force"]/300] for i in range(0,800,50)]).ravel()
        dists = np.sum((self.windows - feat)**2, axis=1)
        best_idx = np.argmin(dists)
        return self.futures[best_idx]

# ═══════════════════════════════════════════════════════════
# Method B: Online polynomial fit
# ═══════════════════════════════════════════════════════════
def poly_predict(hist, degree=3, pred_steps=200):
    """Fit polynomial to last 200ms of speed and RTE, extrapolate."""
    # Use last 200ms as fitting data
    fit_len = 200
    t_fit = np.arange(fit_len)
    spd_fit = np.array([p["speed"] for p in hist[-fit_len:]])
    rte_fit = np.array([p["real_time_energy"] for p in hist[-fit_len:]])

    # Fit polynomials
    spd_coef = np.polyfit(t_fit, spd_fit, min(degree, 3))
    rte_coef = np.polyfit(t_fit, rte_fit, min(degree, 2))

    # Extrapolate
    t_pred = np.arange(fit_len, fit_len + pred_steps)
    spd_pred = np.polyval(spd_coef, t_pred)
    rte_pred = np.polyval(rte_coef, t_pred)

    # Build future points
    result = []
    for i in range(pred_steps):
        result.append({
            "speed": max(0.1, float(spd_pred[i])),
            "real_time_energy": max(0.0, float(rte_pred[i])),
        })
    return result

# ═══════════════════════════════════════════════════════════
# Method C: Template match by operation mode
# ═══════════════════════════════════════════════════════════
class TemplatePredictor:
    """Match current mode sequence against historical patterns, use template continuation."""
    def __init__(self, all_pts):
        # Build template library: (mode_sequence_800) → future_200
        self.templates = {}  # mode_pattern → list of futures
        for t in range(0, len(all_pts)-1000, 200):
            w = all_pts[t:t+800]
            f = all_pts[t+800:t+1000]
            if len(w)==800 and len(f)==200:
                # Mode pattern: dominant modes in 100ms chunks
                pattern = tuple(int(np.mean([w[i]["mode"] for i in range(j,j+100)])) for j in range(0,800,100))
                if pattern not in self.templates:
                    self.templates[pattern] = []
                self.templates[pattern].append(f)

    def predict(self, hist):
        pattern = tuple(int(np.mean([hist[i]["mode"] for i in range(j,j+100)])) for j in range(0,800,100))
        # Find closest pattern
        if not self.templates:
            return None
        best_pat = min(self.templates.keys(), key=lambda p: sum(a!=b for a,b in zip(p,pattern)) if len(p)==len(pattern) else 99)
        futures = self.templates[best_pat]
        # Return median future
        if len(futures) > 1:
            speeds = np.median([[f[i]["speed"] for i in range(200)] for f in futures], axis=0)
            rtes = np.median([[f[i]["real_time_energy"] for i in range(200)] for f in futures], axis=0)
            return [{"speed":float(s), "real_time_energy":float(r)} for s,r in zip(speeds,rtes)]
        return futures[0] if futures else None

# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════
def eval_method(name, predictor, pts, save_times, out_dir, is_knn=False, is_template=False):
    errors = []
    for t_start in save_times:
        hist = pts[t_start:t_start+800]
        fut = pts[t_start+800:t_start+1000]
        if len(hist)!=800 or len(fut)!=200:
            continue

        # Predict
        if is_knn or is_template:
            pred_fut = predictor.predict(hist)
        else:
            pred_fut = predictor(hist)  # poly_predict

        if pred_fut is None:
            continue

        # Compute curves
        ap, ae = compute_curve(hist, spk=hist[0]["position"]/1000)
        fp_curve, fe = compute_curve(fut, spk=ap[-1], sek=ae[-1])

        # Predicted curve
        pp, pe = ap[-1], ae[-1]
        pp_list = [pp]; pe_list = [pe]
        for p in pred_fut:
            pp += p["speed"]/3.6*0.001/1000.0
            pe += p["real_time_energy"]*max(p["speed"],0.1)*0.001/3600.0
            pp_list.append(pp); pe_list.append(pe)

        # Per-point mean error over all 200 prediction steps
        n_pts = min(len(pp_list), len(fp_curve))
        pos_diffs = [abs(pp_list[i] - fp_curve[i]) * 1000 for i in range(1, n_pts)]  # m
        ene_diffs = [abs(pe_list[i] - fe[i]) for i in range(1, min(len(pe_list), len(fe)))]
        pe_err = np.mean(ene_diffs)  # mean energy MAE over window
        pm_err = np.mean(pos_diffs)  # mean position MAE over window (m)
        errors.append((pe_err, pm_err, pp_list, pe_list, ap, ae, fp_curve, fe, t_start))

    ene_errs = [e[0] for e in errors]
    pos_errs = [e[1] for e in errors]
    print(f"\n{name}:")
    print(f"  Energy err: mean={np.mean(ene_errs):.4f} max={np.max(ene_errs):.4f} kWh")
    print(f"  Position err: mean={np.mean(pos_errs):.1f} max={np.max(pos_errs):.1f} m")

    # Save plots (every 5th)
    for i, (ee, pm, pp_list, pe_list, ap, ae, fp_curve, fe, ts) in enumerate(errors):
        if i % 5 != 0 and ts not in (save_times[0], save_times[-1]):
            continue
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(ap, ae, 'b-', lw=1.2, label='Actual')
        ax.plot(pp_list, pe_list, '--', color='orange', lw=1.5, label='Predicted')
        ax.axvline(x=ap[-1], color='gray', ls=':')
        ax.set_title(f'{name}: window {ts}ms (ene_err={ee:.4f}kWh)')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir/f"{name.replace(' ','_')}_{i}.png", dpi=100)
        plt.close(fig)

    return np.mean(ene_errs), np.mean(pos_errs)


def main():
    csv_file = sys.argv[1] if len(sys.argv)>1 else "/home/maksuning/Proj/API_COM/verify_prediction/OptReslog.2026-05-09_20-10-54-FZ602-1-2-86.csv"
    out_dir = Path("plots_methods")
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*.png"): f.unlink()

    meta,raw = load_csv(csv_file)
    dur = float(re.search(r'设定时间：(\d+)', meta).group(1))
    interp = interp_1ms(raw,dur)
    trf,ebf,bpw,soc,rte = derive(interp)
    total = len(interp["spd"])

    pts = []
    for i in range(total):
        pts.append({"time_ms":i, "position":float(interp["pos"][i]),
            "speed":float(interp["spd"][i]), "force":float(interp["frc"][i]),
            "mode":int(interp["mod"][i]), "tractive_force":float(trf[i]),
            "electric_brake_force":float(ebf[i]), "battery_power":float(bpw[i]),
            "soc":float(soc[i]), "real_time_energy":float(rte[i])})

    win_times = list(range(0,500,100)) + list(range(1000,total-800,2000))
    if total-800 not in win_times: win_times.append(total-800)

    results = {}

    # Method A: k-NN
    t0=time.time()
    knn = KNNPredictor(pts)
    print(f"k-NN index built in {time.time()-t0:.1f}s ({len(knn.windows)} windows)")
    results["A: k-NN lookup"] = eval_method("A_kNN", knn, pts, win_times, out_dir, is_knn=True)

    # Method B: Polynomial fit
    results["B: Polynomial fit"] = eval_method("B_Poly", poly_predict, pts, win_times, out_dir)

    # Method C: Template match
    t0=time.time()
    tmp = TemplatePredictor(pts)
    print(f"Template index built in {time.time()-t0:.1f}s ({len(tmp.templates)} patterns)")
    results["C: Template"] = eval_method("C_Template", tmp, pts, win_times, out_dir, is_template=True)

    # ── Summary bar chart ──
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    names = list(results.keys())
    ene_vals = [results[n][0] for n in names]
    pos_vals = [results[n][1] for n in names]
    ax1.bar(names, ene_vals, color=['steelblue','orange','green'])
    ax1.set_ylabel('Mean Energy Error (kWh)'); ax1.set_title('Energy Prediction Error')
    ax1.tick_params(axis='x', rotation=15)
    ax2.bar(names, pos_vals, color=['steelblue','orange','green'])
    ax2.set_ylabel('Mean Position Error (m)'); ax2.set_title('Position Prediction Error')
    ax2.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir/"comparison.png", dpi=120)
    plt.close(fig)

    print(f"\n  Saved comparison to {out_dir}/")
    for n, (ene, pos) in results.items():
        print(f"  {n:25s}: ene={ene:.4f}kWh  pos={pos:.1f}m")


if __name__=="__main__": main()
