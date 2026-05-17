"""
Pipe-based LSTM server — reads JSON from stdin, writes JSON to stdout.
Each line: {"type":"train_predict","window_id":N,"hist":[...],"future":[...]}
Response:  {"window_id":N,"pred_pos":[...],...}
On EOF or "exit" → save stats and quit.
"""
import sys, json, traceback
import numpy as np
import torch, torch.nn as nn

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class SeqPredictor(nn.Module):
    def __init__(self, in_dim=10, hid=128, out_dim=2):
        super().__init__()
        self.encoder = nn.LSTM(in_dim, hid, 1, batch_first=True)
        self.decoder = nn.LSTMCell(out_dim, hid)
        self.fc = nn.Linear(hid, out_dim)
    def encode(self, x):
        _, (hn, cn) = self.encoder(x)
        return hn.squeeze(0), cn.squeeze(0)
    def forward(self, x, targets=None, tf=0.5):
        h, c = self.encode(x)
        outputs = []
        inp = x[:, -1, 1:3] if targets is None else targets[:, 0, :]
        if inp.dim() == 1: inp = inp.unsqueeze(0)
        for t in range(200):
            h, c = self.decoder(inp, (h, c))
            out = self.fc(h)
            outputs.append(out)
            if targets is not None and torch.rand(1).item() < tf:
                inp = targets[:, t, :]
            else:
                inp = out
        return torch.stack(outputs, dim=1)


def main():
    device = torch.device("cpu")
    model = SeqPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    losses = []
    results = []
    count = 0

    sys.stderr.write("LSTM Pipe Server ready\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line or line == "exit":
            break

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if msg.get("type") == "train_predict":
            hist = np.array(msg["hist"], dtype=np.float32)
            fut = np.array(msg["future"], dtype=np.float32)
            wid = msg["window_id"]
            is_save = msg.get("save", False)

            X = torch.tensor(hist[:, :10]).unsqueeze(0).to(device)
            Y = torch.tensor(fut[:, [1, 8]]).unsqueeze(0).to(device)

            # Train step
            model.train()
            optimizer.zero_grad()
            pred = model(X, Y, 0.5)
            loss = loss_fn(pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_val = float(loss.item())
            losses.append(loss_val)
            count += 1

            if not is_save:
                # Quick response for training-only windows
                resp = {"window_id": wid, "loss": float(loss_val), "step": count}
            else:
                # Full response with curves
                model.eval()
                with torch.no_grad():
                    pred_out = model(X, None).squeeze(0).cpu().numpy()

                raw = hist[:, 10:13]
                cp = float(raw[0, 2]); ce = 0.0
                ap, ae = [cp], [ce]
                for i in range(len(raw)):
                    spd = max(raw[i,0], 0.1); rte = max(raw[i,1], 0.0)
                    cp += spd/3.6*0.001/1000.0; ce += rte*spd*0.001/3600.0
                    ap.append(float(cp)); ae.append(float(ce))

                pp, pe = cp, ce
                pp_list, pe_list = [pp], [pe]
                for i in range(200):
                    spd = max(float(pred_out[i,0])*100.0, 0.1); rte = max(float(pred_out[i,1])*100.0, 0.0)
                    spd = min(spd, 200.0); rte = min(rte, 200.0)
                    pp += spd/3.6*0.001/1000.0; pe += rte*spd*0.001/3600.0
                    pp_list.append(float(pp)); pe_list.append(float(pe))

                # Compute actual future curve for error comparison
                fut_raw = fut[:, 10:13]
                fp, fe = cp, ce
                fp_list, fe_list = [fp], [fe]
                for i in range(len(fut_raw)):
                    spd = max(fut_raw[i,0], 0.1); rte = max(fut_raw[i,1], 0.0)
                    fp += spd/3.6*0.001/1000.0; fe += rte*spd*0.001/3600.0
                    fp_list.append(float(fp)); fe_list.append(float(fe))

                resp = {
                    "window_id": wid, "loss": float(loss_val),
                    "pos_err_m": float(abs(pp_list[-1]-fp_list[-1])*1000),
                    "ene_err": float(abs(pe_list[-1]-fe_list[-1])),
                    "actual_pos": ap, "actual_ene": ae,
                    "pred_pos": pp_list, "pred_ene": pe_list,
                    "future_pos": fp_list, "future_ene": fe_list,
                    "step": count,
                }
                results.append(resp)

            sys.stdout.write(json.dumps(resp, cls=NumpyEncoder) + "\n")
            sys.stdout.flush()

        elif msg.get("type") == "stats":
            if results:
                pe = [r["pos_err_m"] for r in results]
                ee = [r["ene_err"] for r in results]
                resp = {
                    "total": len(results),
                    "mean_loss": float(np.mean(losses[-100:])) if len(losses)>=100 else float(np.mean(losses)),
                    "pos_err_mean": float(np.mean(pe)),
                    "ene_err_mean": float(np.mean(ee)),
                }
            else:
                resp = {"error": "no results"}
            sys.stdout.write(json.dumps(resp, cls=NumpyEncoder) + "\n")
            sys.stdout.flush()

        elif msg.get("type") == "reset":
            model = SeqPredictor().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            losses = []; results = []; count = 0
            sys.stdout.write('{"status":"reset"}\n')
            sys.stdout.flush()

    # Save on exit
    with open("pipe_results.json", "w") as f:
        json.dump({"results": results, "total_steps": count,
                   "mean_loss": float(np.mean(losses[-100:])) if len(losses)>=100 else float(np.mean(losses))}, f, indent=2)
    sys.stderr.write("Done\n")


if __name__ == "__main__":
    main()
