"""Export preprocessed 1ms telemetry data for C++ LSTM verification."""
import csv, re, sys, struct, numpy as np
from pathlib import Path

ETA=0.85; BAT_CAP=200.0; BAT_TR=0.30; BAT_RE=0.50; AUX=120.0; SOC0=80.0

def load_csv(fp):
    d={"rt":[],"pos":[],"spd":[],"frc":[],"mod":[]}
    with open(fp) as f:
        meta=f.readline().strip(); f.readline()
        for r in csv.reader(f):
            if len(r)<5: continue
            d["rt"].append(float(r[0])); d["pos"].append(float(r[1]))
            d["spd"].append(float(r[2])); d["frc"].append(float(r[3]))
            d["mod"].append(int(r[4]))
    for k in d: d[k]=np.array(d[k])
    return meta,d

def interp_1ms(data,dur):
    n=int(dur*1000); tr=data["rt"]; tg=np.arange(n)/1000.0
    o={}
    for k in ["pos","spd","frc"]: o[k]=np.interp(tg,tr,data[k])
    o["mod"]=np.zeros(n,dtype=int)
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

def main():
    csv_file = sys.argv[1] if len(sys.argv)>1 else "../OptReslog.2026-05-09_20-10-54-FZ602-1-2-86.csv"
    meta,raw = load_csv(csv_file)
    dur = float(re.search(r'设定时间：(\d+)',meta).group(1))
    interp = interp_1ms(raw,dur)
    trf,ebf,bpw,soc,rte = derive(interp)
    n = len(interp["spd"])

    # Write binary: [n] [10 floats per point] (float32)
    out = Path("telemetry.bin")
    with open(out,"wb") as f:
        f.write(struct.pack("i",n))
        for i in range(n):
            feats = np.array([
                interp["pos"][i]/1000.0,       # 0 position km
                interp["spd"][i]/100.0,         # 1 speed normalized
                interp["frc"][i]/300.0,         # 2 force normalized
                interp["mod"][i]/4.0,           # 3 mode normalized
                trf[i]/300.0,                   # 4 tractive_force
                ebf[i]/300.0,                   # 5 electric_brake_force
                bpw[i]/500.0,                   # 6 battery_power
                soc[i]/100.0,                   # 7 soc
                rte[i]/100.0,                   # 8 real_time_energy
                i/100000.0,                     # 9 time_ms normalized
            ], dtype=np.float32)
            f.write(feats.tobytes())
        # Also write raw speed and rte + position for curve computation
        raw_data = np.column_stack([
            interp["spd"], rte, interp["pos"]/1000.0
        ]).astype(np.float32)
        f.write(raw_data.tobytes())

    print(f"Exported {n} points to {out}")
    print(f"  Total size: {out.stat().st_size/1024:.0f} KB")

if __name__=="__main__": main()
