"""
TCP LSTM Client — simulates Backend 2 feeding data to the Python LSTM server.
Loads CSV, interpolates, sends 100ms batches, receives predictions, saves all results.
"""
import csv, re, sys, json, socket, time, struct
import numpy as np
from pathlib import Path

# ── Same pipeline ──
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


class TCPClient:
    def __init__(self, host="127.0.0.1", port=9900, timeout=10):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect((host, port))
        self.buf = b""

    def send_msg(self, msg):
        data = json.dumps(msg).encode() + b"\n"
        self.sock.sendall(data)

    def recv_msg(self):
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("Connection closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line.decode())

    def close(self):
        self.sock.close()


def main():
    csv_file = sys.argv[1] if len(sys.argv)>1 else "/home/maksuning/Proj/API_COM/verify_prediction/OptReslog.2026-05-09_20-10-54-FZ602-1-2-86.csv"
    meta, raw = load_csv(csv_file)
    dur = float(re.search(r'设定时间：(\d+)', meta).group(1))
    interp = interp_1ms(raw, dur)
    trf, ebf, bpw, soc, rte = derive(interp)
    total = len(interp["spd"])
    print(f"Loaded: {dur}s, {total} 1ms points")

    # Build data with features + raw columns
    all_data = []
    for i in range(total):
        feats = [
            interp["pos"][i]/1000.0, interp["spd"][i]/100.0, interp["frc"][i]/300.0,
            interp["mod"][i]/4.0, trf[i]/300.0, ebf[i]/300.0,
            bpw[i]/500.0, soc[i]/100.0, rte[i]/100.0, i/100000.0,
        ]
        raw_cols = [interp["spd"][i], rte[i], interp["pos"][i]/1000.0]
        all_data.append(feats + raw_cols)

    WIN, PRED, STRIDE = 800, 200, 100

    # Window times to save
    save_times = set()
    for t in range(0, 500, STRIDE): save_times.add(t)
    for t in range(1000, total-WIN+1, 2000): save_times.add(t)
    if total-WIN not in save_times: save_times.add(total-WIN)

    print(f"Connecting to LSTM server...")
    cli = TCPClient()
    cli.send_msg({"type": "reset"})
    _ = cli.recv_msg()
    print(f"Connected. Training on ~{total//STRIDE} windows, saving {len(save_times)}...")

    results = []
    t0 = time.time()

    train_count = 0
    for t_cur in range(0, total-WIN-PRED, STRIDE):
        hist = all_data[t_cur:t_cur+WIN]
        fut = all_data[t_cur+WIN:t_cur+WIN+PRED]
        if len(hist) != WIN or len(fut) != PRED:
            continue

        is_save = t_cur in save_times
        msg = {"type": "train_predict", "window_id": t_cur, "hist": hist, "future": fut}
        cli.send_msg(msg)
        resp = cli.recv_msg()
        train_count += 1
        if is_save:
            results.append(resp)
        if train_count % 100 == 0:
            print(f"  [{train_count} steps] loss={resp.get('loss',0):.6f}")

    elapsed = time.time() - t0

    # Get final stats
    cli.send_msg({"type": "save_stats"})
    stats = cli.recv_msg()
    cli.close()

    stats["elapsed_s"] = elapsed
    stats["train_steps"] = train_count
    stats["data_file"] = csv_file
    stats["duration_s"] = dur

    out_file = "tcp_results.json"
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*50}")
    print(f"TCP LSTM Results ({len(results)} windows, {elapsed:.1f}s)")
    print(f"{'='*50}")
    if results:
        pos_errs = [r["pos_err_m"] for r in results]
        ene_errs = [r["ene_err"] for r in results]
        print(f"  Pos err (m):  mean={np.mean(pos_errs):.2f}, max={np.max(pos_errs):.2f}")
        print(f"  Ene err (kWh): mean={np.mean(ene_errs):.6f}, max={np.max(ene_errs):.6f}")
        print(f"  Loss (final):  {results[-1].get('loss',0):.6f}")
    print(f"  Saved: {out_file}")


if __name__ == "__main__":
    main()
