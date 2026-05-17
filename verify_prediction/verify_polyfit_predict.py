"""
Polynomial fit prediction verification — same style as verify_lstm_predict.py.
Zero training: fit 3rd-order poly to speed, 2nd-order to RTE over last 200ms.
"""
import csv, re, sys, numpy as np
from pathlib import Path
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

def polyfit_predict(hist, fit_len=200, pred_len=200, spd_deg=3, rte_deg=2):
    """Fit polynomials to recent speed and RTE, extrapolate forward."""
    n = len(hist)
    fit_start = max(0, n - fit_len)
    t_fit = np.arange(fit_len)
    spd_fit = np.array([hist[fit_start+i]["speed"] for i in range(fit_len)])
    rte_fit = np.array([hist[fit_start+i]["real_time_energy"] for i in range(fit_len)])

    spd_coef = np.polyfit(t_fit, spd_fit, spd_deg)
    rte_coef = np.polyfit(t_fit, rte_fit, rte_deg)

    t_pred = np.arange(fit_len, fit_len + pred_len)
    spd_pred = np.polyval(spd_coef, t_pred)
    rte_pred = np.polyval(rte_coef, t_pred)

    result = []
    for i in range(pred_len):
        result.append({"speed": max(0.1, float(spd_pred[i])),
                       "real_time_energy": max(0.0, float(rte_pred[i]))})
    return result


def main():
    csv_file = sys.argv[1] if len(sys.argv)>1 else "/home/maksuning/Proj/API_COM/verify_prediction/OptReslog.2026-05-09_20-10-54-FZ602-1-2-86.csv"
    out_dir = Path("plots_polyfit")
    out_dir.mkdir(exist_ok=True)
    wd = out_dir / "sliding_windows"
    wd.mkdir(exist_ok=True)
    for f in out_dir.glob("*.png"): f.unlink()
    for f in wd.glob("*.png"): f.unlink()

    meta,raw = load_csv(csv_file)
    dur = float(re.search(r'设定时间：(\d+)', meta).group(1))
    interp = interp_1ms(raw,dur)
    trf,ebf,bpw,soc,rte = derive(interp)
    total = len(interp["spd"])
    print(f"Loaded: {dur}s, {total} 1ms points")

    pts = []
    for i in range(total):
        pts.append({"time_ms":i, "position":float(interp["pos"][i]),
            "speed":float(interp["spd"][i]), "force":float(interp["frc"][i]),
            "mode":int(interp["mod"][i]), "tractive_force":float(trf[i]),
            "electric_brake_force":float(ebf[i]), "battery_power":float(bpw[i]),
            "soc":float(soc[i]), "real_time_energy":float(rte[i])})

    WIN, PRED = 800, 200
    win_times = list(range(0, 500, 100)) + list(range(1000, total-WIN+1, 2000))
    if total-WIN not in win_times: win_times.append(total-WIN)

    pos_errs = []; ene_errs = []; ene_maes = []

    print(f"Running polynomial fit on {len(win_times)} windows...")
    for t_start in win_times:
        hist = pts[t_start:t_start+WIN]
        fut = pts[t_start+WIN:t_start+WIN+PRED]
        if len(hist)!=WIN or len(fut)!=PRED: continue

        pred_fut = polyfit_predict(hist)

        ap, ae = compute_curve(hist, spk=hist[0]["position"]/1000)
        fp, fe = compute_curve(fut, spk=ap[-1], sek=ae[-1])

        pp, pe = ap[-1], ae[-1]
        pp_list, pe_list = [pp], [pe]
        for p in pred_fut:
            pp += p["speed"]/3.6*0.001/1000.0
            pe += p["real_time_energy"]*max(p["speed"],0.1)*0.001/3600.0
            pp_list.append(pp); pe_list.append(pe)

        # Per-point errors
        nc = min(len(pp_list), len(fp))
        pos_errs.append(np.mean([abs(pp_list[i]-fp[i])*1000 for i in range(1,nc)]))
        ene_errs.append(abs(pe_list[-1]-fe[-1]))
        ene_maes.append(np.mean([abs(pe_list[i]-fe[i]) for i in range(1,min(len(pe_list),len(fe)))]))

        # Plot
        te = t_start + WIN
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ap, ae, 'b-', lw=1.5, label=f'Actual [{t_start},{te}]')
        ax.plot(fp, fe, '-', color='green', lw=1.2, alpha=0.7, label=f'Actual [{te},{te+PRED}]')
        ax.plot(pp_list, pe_list, '--', color='orange', lw=1.8, label=f'Predicted [{te},{te+PRED}]')
        ax.axvline(x=ap[-1], color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Position (km)'); ax.set_ylabel('Cumulative Energy (kWh)')
        ax.set_title(f'Polyfit: [{t_start},{te}]ms → [{te},{te+PRED}]ms  (pos_mae={pos_errs[-1]:.2f}m, ene_err={ene_errs[-1]:.4f}kWh)')
        ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        idx = win_times.index(t_start) + 1
        fig.savefig(wd / f"win_{idx:02d}_t{t_start}_{te}ms.png", dpi=120)
        plt.close(fig)

    n_plots = len(pos_errs)
    print(f"\n{'='*50}")
    print(f"Polynomial Fit Prediction ({n_plots} windows)")
    print(f"{'='*50}")
    print(f"  Position MAE (m):   mean={np.mean(pos_errs):.3f}  max={np.max(pos_errs):.3f}  median={np.median(pos_errs):.3f}")
    print(f"  Energy endpoint (kWh): mean={np.mean(ene_errs):.6f}  max={np.max(ene_errs):.6f}")
    print(f"  Energy MAE (kWh):  mean={np.mean(ene_maes):.6f}  max={np.max(ene_maes):.6f}")
    print(f"  Saved {n_plots} plots to {wd}/")

    with open(out_dir/"stats.txt","w") as f:
        f.write(f"windows: {n_plots}\n")
        f.write(f"pos_mae_mean_m: {np.mean(pos_errs):.3f}\n")
        f.write(f"pos_mae_max_m: {np.max(pos_errs):.3f}\n")
        f.write(f"ene_mae_mean_kwh: {np.mean(ene_maes):.6f}\n")
        f.write(f"ene_endpoint_mean_kwh: {np.mean(ene_errs):.6f}\n")

if __name__=="__main__": main()
