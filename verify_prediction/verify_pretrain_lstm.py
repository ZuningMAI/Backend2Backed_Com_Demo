"""
Pre-trained LSTM: train once offline on full CSV data, then online inference only.
No weight updates during online prediction — just fast forward pass.

Comparison with online-training LSTM:
  - Online training: each 100ms step does BPTT + Adam update (~10ms per step)
  - Pre-trained: each 100ms step does only encoder+decoder forward (~0.5ms)
"""
import csv, re, sys, math, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Same data pipeline as before ──
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
    o={};
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
    return {"trf":trf,"ebf":ebf,"bpw":bpw,"soc":soc,"rte":rte}

def make_features(pts):
    feats=[]
    for p in pts:
        feats.append([p["position"]/1000.0, p["speed"]/100.0, p["force"]/300.0,
            p["mode"]/4.0, p["tractive_force"]/300.0, p["electric_brake_force"]/300.0,
            p["battery_power"]/500.0, p["soc"]/100.0, p["real_time_energy"]/100.0,
            p["time_ms"]/100000.0])
    return np.array(feats,dtype=np.float32)

def make_targets(pts):
    return np.array([[p["speed"]/100.0, p["real_time_energy"]/100.0] for p in pts],dtype=np.float32)

def compute_curve(pts, spk=0, sek=0):
    pos=[spk]; ene=[sek]; cp=spk; ce=sek
    for p in pts:
        cp+=p["speed"]/3.6*0.001/1000.0
        ce+=p["real_time_energy"]*max(p["speed"],0.1)*0.001/3600.0
        pos.append(cp); ene.append(ce)
    return pos,ene

def pred_curve_from_output(out, last_pt, lpk, lek):
    cp=lpk; ce=lek; pos=[cp]; ene=[ce]
    for i in range(len(out)):
        spd=out[i,0]*100.0; rte=out[i,1]*100.0
        spd=max(0.1,min(spd,200.0)); rte=max(0.0,min(rte,200.0))
        cp+=spd/3.6*0.001/1000.0; ce+=rte*spd*0.001/3600.0
        pos.append(cp); ene.append(ce)
    return pos,ene

# ── Model (same architecture) ──
class SeqPredictor(nn.Module):
    def __init__(self, in_dim=10, hid=128, out_dim=2):
        super().__init__()
        self.encoder=nn.LSTM(in_dim,hid,1,batch_first=True)
        self.decoder=nn.LSTMCell(out_dim,hid)
        self.fc=nn.Linear(hid,out_dim); self.hid=hid

    def encode(self,x):
        _,(hn,cn)=self.encoder(x); return hn.squeeze(0),cn.squeeze(0)

    def forward(self,x, targets=None, tf=0.5):
        h,c=self.encode(x)
        outputs=[]
        inp=x[:,-1,1:3] if targets is None else targets[:,0,:]
        if inp.dim()==1: inp=inp.unsqueeze(0)
        for t in range(200):
            h,c=self.decoder(inp,(h,c)); out=self.fc(h)
            outputs.append(out)
            if targets is not None and torch.rand(1).item()<tf: inp=targets[:,t,:]
            else: inp=out
        return torch.stack(outputs,dim=1)


def main():
    csv_file = sys.argv[1] if len(sys.argv)>1 else "/home/maksuning/Proj/API_COM/verify_prediction/OptReslog.2026-05-09_20-10-54-FZ602-1-2-86.csv"
    out_dir = Path("plots_pretrain")
    out_dir.mkdir(exist_ok=True); wd=out_dir/"sliding_windows"; wd.mkdir(exist_ok=True)
    for f in out_dir.glob("*.png"): f.unlink()
    for f in wd.glob("*.png"): f.unlink()

    print(f"Loading: {csv_file}")
    meta,raw=load_csv(csv_file)
    dur=float(re.search(r'设定时间：(\d+)',meta).group(1))

    interp=interp_1ms(raw,dur); d=derive(interp)
    total=len(interp["spd"])
    pts=[]
    for i in range(total):
        pts.append({"time_ms":i,"position":float(interp["pos"][i]),"speed":float(interp["spd"][i]),
            "force":float(interp["frc"][i]),"mode":int(interp["mod"][i]),
            "tractive_force":float(d["trf"][i]),"electric_brake_force":float(d["ebf"][i]),
            "battery_power":float(d["bpw"][i]),"soc":float(d["soc"][i]),
            "real_time_energy":float(d["rte"][i])})

    # ═══ Phase 1: Offline pre-training ═══
    print("Phase 1: Offline pre-training...")
    device=torch.device("cpu")
    model=SeqPredictor().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    loss_fn=nn.MSELoss()

    win_sz=800; pred_sz=200; stride=500  # fewer samples for faster training
    X_list=[]; Y_list=[]
    for t in range(0,total-win_sz-pred_sz,stride):
        hist=pts[t:t+win_sz]; fut=pts[t+win_sz:t+win_sz+pred_sz]
        if len(hist)!=win_sz or len(fut)!=pred_sz: continue
        X_list.append(torch.tensor(make_features(hist)))
        Y_list.append(torch.tensor(make_targets(fut)))
    X_all=torch.stack(X_list).to(device); Y_all=torch.stack(Y_list).to(device)
    print(f"  Training samples: {len(X_all)}")

    ds=TensorDataset(X_all,Y_all); dl=DataLoader(ds,batch_size=32,shuffle=True)
    best_loss=float("inf")
    t0=time.time()
    for epoch in range(10):
        model.train(); tl=0.0
        for xb,yb in dl:
            opt.zero_grad()
            pred=model(xb,yb,0.5); loss=loss_fn(pred,yb)
            loss.backward(); opt.step(); tl+=loss.item()
        tl/=len(dl)
        if tl<best_loss: best_loss=tl; torch.save(model.state_dict(),str(out_dir/"pretrained.pt"))
        if (epoch+1)%5==0 or epoch==0: print(f"  Epoch {epoch+1}/10, loss={tl:.6f}")
    t_train=time.time()-t0
    print(f"  Pre-training done in {t_train:.1f}s, best_loss={best_loss:.6f}")

    # ═══ Phase 2: Online inference only (no training) ═══
    print("Phase 2: Online inference (forward pass only)...")
    model.load_state_dict(torch.load(str(out_dir/"pretrained.pt"),weights_only=True))
    model.eval()

    win_times=list(range(0,500,100))+list(range(1000,total-win_sz+1,2000))
    if win_times[-1]!=total-win_sz: win_times.append(total-win_sz)

    predictions={}
    pos_errs=[]; ene_errs=[]; ene_maes=[]
    t_infer=0.0

    for t_start in win_times:
        hist=pts[t_start:t_start+win_sz]
        fut=pts[t_start+win_sz:t_start+win_sz+pred_sz]
        if len(hist)!=win_sz or len(fut)!=pred_sz: continue

        # Inference only
        X=torch.tensor(make_features(hist)).unsqueeze(0).to(device)
        t0=time.time()
        with torch.no_grad():
            pred_out=model(X,None).squeeze(0).cpu().numpy()
        t_infer+=time.time()-t0

        actual_pos,actual_ene=compute_curve(hist,spk=hist[0]["position"]/1000)
        pred_pos,pred_ene=pred_curve_from_output(pred_out,hist[-1],actual_pos[-1],actual_ene[-1])
        future_pos,future_ene=compute_curve(fut,spk=actual_pos[-1],sek=actual_ene[-1])
        predictions[t_start]=(actual_pos,actual_ene,pred_pos,pred_ene,future_pos,future_ene)

        pos_errs.append(abs(pred_pos[-1]-future_pos[-1])*1000)
        ene_errs.append(abs(pred_ene[-1]-future_ene[-1]))
        nc=min(len(pred_ene),len(future_ene))
        ene_maes.append(np.mean([abs(pred_ene[i]-future_ene[i]) for i in range(nc)]))

    # ── Plots ──
    print("Generating plots...")
    for t_start in win_times:
        if t_start not in predictions: continue
        ap,ae,pp,pe,fp,fe=predictions[t_start]; te=t_start+win_sz
        pe_m=abs(pp[-1]-fp[-1])*1000; ee=abs(pe[-1]-fe[-1])
        fig,ax=plt.subplots(figsize=(12,5))
        ax.plot(ap,ae,'b-',lw=1.5,label=f'Actual [{t_start},{te}]')
        ax.plot(fp,fe,'-',color='green',lw=1.2,alpha=0.7,label=f'Actual [{te},{te+pred_sz}]')
        ax.plot(pp,pe,'--',color='orange',lw=1.8,label=f'Predicted [{te},{te+pred_sz}]')
        ax.axvline(x=ap[-1],color='gray',ls=':',alpha=0.5)
        ax.set_xlabel('Position (km)'); ax.set_ylabel('Cumulative Energy (kWh)')
        ax.set_title(f'Pre-trained LSTM: [{t_start},{te}]ms → [{te},{te+pred_sz}]ms  (err={pe_m:.1f}m,{ee:.4f}kWh)')
        ax.legend(loc='upper left'); ax.grid(True,alpha=0.3)
        fig.tight_layout()
        idx=win_times.index(t_start)+1
        fig.savefig(wd/f"win_{idx:02d}_t{t_start}_{te}ms.png",dpi=120); plt.close(fig)

    n_plots=len(list(wd.glob("*.png")))

    # ── Comparison ──
    avg_infer=t_infer/max(n_plots,1)*1000
    print(f"\n{'='*60}")
    print(f"Pre-trained LSTM vs Online-training LSTM")
    print(f"{'='*60}")
    print(f"  Pre-training time:       {t_train:.1f}s (one-time offline)")
    print(f"  Inference per window:    {avg_infer:.2f}ms (forward only, no backprop)")
    print(f"")
    print(f"  Prediction accuracy ({n_plots} windows):")
    print(f"    Position error (m):    mean={np.mean(pos_errs):.2f}, max={np.max(pos_errs):.2f}")
    print(f"    Energy error (kWh):    mean={np.mean(ene_errs):.6f}, max={np.max(ene_errs):.6f}")
    print(f"    Energy MAE  (kWh):     mean={np.mean(ene_maes):.6f}")
    print(f"  Saved {n_plots} plots to {wd}/")

    with open(out_dir/"stats.txt","w") as f:
        f.write(f"pretrain_time_s: {t_train:.1f}\n")
        f.write(f"infer_time_ms: {avg_infer:.2f}\n")
        f.write(f"windows: {n_plots}\n")
        f.write(f"pos_err_mean_m: {np.mean(pos_errs):.2f}\n")
        f.write(f"pos_err_max_m: {np.max(pos_errs):.2f}\n")
        f.write(f"ene_err_mean_kwh: {np.mean(ene_errs):.6f}\n")
        f.write(f"ene_mae_mean_kwh: {np.mean(ene_maes):.6f}\n")


if __name__=="__main__": main()
