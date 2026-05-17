"""
Create animation from polynomial fit sliding windows.
Each frame = one window [t, t+800]ms actual + [t+800, t+1000]ms predicted.
"""
import csv, re, sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def polyfit_predict(hist, fit_len=200, pred_len=200):
    n=len(hist); fs=max(0,n-fit_len)
    t_fit=np.arange(fit_len)
    spd_fit=np.array([hist[fs+i]["speed"] for i in range(fit_len)])
    rte_fit=np.array([hist[fs+i]["real_time_energy"] for i in range(fit_len)])
    spd_c=np.polyfit(t_fit,spd_fit,3); rte_c=np.polyfit(t_fit,rte_fit,2)
    tp=np.arange(fit_len,fit_len+pred_len)
    spd_p=np.polyval(spd_c,tp); rte_p=np.polyval(rte_c,tp)
    return [{"speed":max(0.1,float(spd_p[i])),"real_time_energy":max(0.0,float(rte_p[i]))} for i in range(pred_len)]


def main():
    csv_file = sys.argv[1] if len(sys.argv)>1 else "/home/maksuning/Proj/API_COM/verify_prediction/OptReslog.2026-05-09_20-10-54-FZ602-1-2-86.csv"
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

    WIN, PRED = 800, 200
    win_times = list(range(0, total-WIN+1, 100))  # every 100ms for smooth animation
    # Downsample for manageable frame count (~100 frames)
    step = max(1, len(win_times) // 100)
    win_times = win_times[::step]
    if win_times[-1] != total-WIN: win_times.append(total-WIN)
    print(f"Generating {len(win_times)} animation frames...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle('Polynomial Fit Online Prediction', fontsize=14, fontweight='bold')

    frames_data = []
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

        # Speed over time (top plot data)
        t_hist = [p["time_ms"]/1000 for p in hist]
        spd_hist = [p["speed"] for p in hist]
        t_pred_abs = [(WIN + i)/1000 for i in range(PRED)]
        spd_pred_abs = [p["speed"] for p in pred_fut]
        t_fut_abs = [(WIN + i)/1000 for i in range(PRED)]
        spd_fut_abs = [p["speed"] for p in fut]

        frames_data.append({
            "t_start": t_start, "t_end": t_start+WIN,
            "t_hist": t_hist, "spd_hist": spd_hist,
            "t_pred": t_pred_abs, "spd_pred": spd_pred_abs,
            "t_fut": t_fut_abs, "spd_fut": spd_fut_abs,
            "ap": ap, "ae": ae, "pp": pp_list, "pe": pe_list,
            "fp": fp, "fe": fe,
        })

    def animate(i):
        fd = frames_data[i]
        ax1.clear(); ax2.clear()

        # Top: speed vs time
        ax1.plot(fd["t_hist"], fd["spd_hist"], 'b-', lw=1, alpha=0.7)
        ax1.plot(fd["t_fut"], fd["spd_fut"], '-', color='green', lw=1, alpha=0.5)
        ax1.plot(fd["t_pred"], fd["spd_pred"], '--', color='orange', lw=1.5)
        ax1.axvline(x=fd["t_hist"][-1], color='gray', ls=':', alpha=0.4)
        ax1.set_ylabel('Speed (km/h)')
        ax1.set_title(f'Window [{fd["t_start"]},{fd["t_end"]}]ms  ({i+1}/{len(frames_data)})')
        ax1.grid(True, alpha=0.2)
        ax1.legend(['History', 'Actual future', 'Predicted'], loc='upper right', fontsize=8)

        # Bottom: position-energy curve
        ax2.plot(fd["ap"], fd["ae"], 'b-', lw=1.5, label='Actual (history)')
        ax2.plot(fd["fp"], fd["fe"], '-', color='green', lw=1.2, alpha=0.7, label='Actual (future)')
        ax2.plot(fd["pp"], fd["pe"], '--', color='orange', lw=1.8, label='Predicted')
        ax2.axvline(x=fd["ap"][-1], color='gray', ls=':', alpha=0.5)
        ax2.set_xlabel('Position (km)'); ax2.set_ylabel('Energy (kWh)')
        ax2.grid(True, alpha=0.2)
        ax2.legend(loc='upper left', fontsize=8)

        return [ax1, ax2]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames_data),
                                   interval=200, blit=False, repeat=True)
    out_path = "polyfit_animation.gif"
    ani.save(out_path, writer='pillow', fps=5, dpi=100)
    print(f"Saved: {out_path}")

    # Also save as MP4
    try:
        ani.save("polyfit_animation.mp4", writer='ffmpeg', fps=8, dpi=120)
        print(f"Saved: polyfit_animation.mp4")
    except:
        print("MP4 skipped (ffmpeg not available)")

    plt.close(fig)

if __name__=="__main__": main()
