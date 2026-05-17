"""
TCP LSTM Server — receives training data, does online LSTM train+predict, returns results.
Protocol: newline-delimited JSON over TCP.
  Request:  {"type":"train_predict","window_id":N,"hist":[...],"future":[...]}
  Response: {"window_id":N,"pred_pos":[...],"pred_ene":[...],"future_pos":[...],...}
  Command:  {"type":"save_stats"} → returns stats dict
  Command:  {"type":"reset"} → reinitialize model
"""
import json, struct, socket, sys, time, threading, traceback
import numpy as np
import torch, torch.nn as nn

# ── LSTM Model (same as verify_lstm_predict.py) ──
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


class LSTMServer:
    def __init__(self, host="127.0.0.1", port=9900):
        self.host = host
        self.port = port
        self.device = torch.device("cpu")
        self.model = SeqPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.losses = []
        self.results = []
        self.lock = threading.Lock()

    def handle_train_predict(self, msg):
        """Process a train_predict request, return response dict."""
        hist_data = np.array(msg["hist"], dtype=np.float32)
        future_data = np.array(msg["future"], dtype=np.float32)
        win_id = msg["window_id"]

        # hist: (800, 10), future: (200, 10) — last 2 cols are raw speed, rte
        # First 10 cols are normalized features
        X = torch.tensor(hist_data[:, :10]).unsqueeze(0).to(self.device)    # (1, 800, 10)
        # Targets: speed_n, rte_n (cols 1, 8 of future or use actual targets)
        # Use positions 1 and 8 from future as targets
        Y = torch.tensor(future_data[:, [1, 8]]).unsqueeze(0).to(self.device)  # (1, 200, 2)

        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(X, Y, 0.5)
        loss = self.loss_fn(pred, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        loss_val = float(loss.item())

        # Inference for prediction curve
        self.model.eval()
        with torch.no_grad():
            pred_out = self.model(X, None).squeeze(0).cpu().numpy()  # (200, 2)

        # Compute curves
        # hist raw data: speed (raw col 10), rte (raw col 11), pos_km (raw col 12)
        raw_cols = hist_data[:, 10:13]  # speed_raw, rte_raw, pos_raw
        future_raw = future_data[:, 10:13]

        # Actual curve from history
        cp = float(raw_cols[0, 2])  # pos_km
        ce = 0.0
        actual_pos = [cp]
        actual_ene = [ce]
        for i in range(len(raw_cols)):
            spd = max(raw_cols[i, 0], 0.1)
            rte = max(raw_cols[i, 1], 0.0)
            cp += spd / 3.6 * 0.001 / 1000.0
            ce += rte * spd * 0.001 / 3600.0
            actual_pos.append(float(cp))
            actual_ene.append(float(ce))

        # Predicted curve
        pp, pe = cp, ce
        pred_pos = [pp]
        pred_ene = [pe]
        for i in range(200):
            spd = max(float(pred_out[i, 0]) * 100.0, 0.1)
            rte = max(float(pred_out[i, 1]) * 100.0, 0.0)
            spd = min(spd, 200.0)
            rte = min(rte, 200.0)
            pp += spd / 3.6 * 0.001 / 1000.0
            pe += rte * spd * 0.001 / 3600.0
            pred_pos.append(float(pp))
            pred_ene.append(float(pe))

        # Future actual curve
        fp, fe = cp, ce
        future_pos = [fp]
        future_ene = [fe]
        for i in range(len(future_raw)):
            spd = max(future_raw[i, 0], 0.1)
            rte = max(future_raw[i, 1], 0.0)
            fp += spd / 3.6 * 0.001 / 1000.0
            fe += rte * spd * 0.001 / 3600.0
            future_pos.append(float(fp))
            future_ene.append(float(fe))

        n_comp = min(len(pred_ene), len(future_ene))
        ene_mae = float(np.mean([abs(pred_ene[i] - future_ene[i]) for i in range(n_comp)]))

        resp = {
            "window_id": win_id,
            "loss": float(loss_val),
            "pos_err_m": float(abs(pred_pos[-1] - future_pos[-1]) * 1000),
            "ene_err": float(abs(pred_ene[-1] - future_ene[-1])),
            "ene_mae": float(ene_mae),
            "actual_pos": [float(x) for x in actual_pos],
            "actual_ene": [float(x) for x in actual_ene],
            "pred_pos": [float(x) for x in pred_pos],
            "pred_ene": [float(x) for x in pred_ene],
            "future_pos": [float(x) for x in future_pos],
            "future_ene": [float(x) for x in future_ene],
        }

        with self.lock:
            self.losses.append(loss_val)
            self.results.append(resp)

        return resp

    def get_stats(self):
        with self.lock:
            if not self.results:
                return {"error": "no results yet"}
            pos_errs = [r["pos_err_m"] for r in self.results]
            ene_errs = [r["ene_err"] for r in self.results]
            ene_maes = [r["ene_mae"] for r in self.results]
            return {
                "total_windows": len(self.results),
                "mean_loss": float(np.mean(self.losses[-100:])) if len(self.losses) >= 100 else float(np.mean(self.losses)),
                "pos_err_mean_m": float(np.mean(pos_errs)),
                "pos_err_max_m": float(np.max(pos_errs)),
                "ene_err_mean": float(np.mean(ene_errs)),
                "ene_mae_mean": float(np.mean(ene_maes)),
                "results": self.results,
            }

    def reset(self):
        with self.lock:
            self.model = SeqPredictor().to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.losses = []
            self.results = []
        return {"status": "reset"}

    def _recv_msg(self, conn):
        """Receive a newline-delimited JSON message."""
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                return None
            data += chunk
            if b"\n" in data:
                line, rest = data.split(b"\n", 1)
                return json.loads(line.decode())

    def _send_msg(self, conn, msg):
        """Send a newline-delimited JSON message."""
        data = json.dumps(msg).encode() + b"\n"
        conn.sendall(data)

    def handle_client(self, conn, addr):
        print(f"[connect] {addr}")
        try:
            while True:
                msg = self._recv_msg(conn)
                if msg is None:
                    break
                mtype = msg.get("type", "")
                if mtype == "train_predict":
                    resp = self.handle_train_predict(msg)
                    self._send_msg(conn, resp)
                elif mtype == "save_stats":
                    resp = self.get_stats()
                    self._send_msg(conn, resp)
                elif mtype == "reset":
                    resp = self.reset()
                    self._send_msg(conn, resp)
                else:
                    self._send_msg(conn, {"error": f"unknown type: {mtype}"})
        except Exception as e:
            print(f"[error] {addr}: {e}")
            traceback.print_exc()
        finally:
            conn.close()
            print(f"[disconnect] {addr}")

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(5)
        print(f"LSTM Server listening on {self.host}:{self.port}")
        try:
            while True:
                conn, addr = sock.accept()
                t = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
                t.start()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            sock.close()


if __name__ == "__main__":
    server = LSTMServer()
    server.start()
