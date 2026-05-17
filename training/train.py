"""
LSTM model training for energy curve prediction.
Input: (position, energy) sequence → predict future energy at future positions.
"""
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Dataset ----------

class EnergyCurveDataset(Dataset):
    """Create (input_seq, target_seq) pairs from energy curves."""

    def __init__(self, data_file: str, lookback: int = 100, forecast: int = 50,
                 stride: int = 20):
        self.samples = []
        with open(data_file) as f:
            runs = json.load(f)

        for run in runs:
            data = run["data"]
            # Extract (position, energy) sequences
            positions = np.array([d["position"] for d in data], dtype=np.float32)
            energies = np.array([d["energy"] for d in data], dtype=np.float32)

            # Create sliding windows
            for i in range(0, len(data) - lookback - forecast, stride):
                x_pos = positions[i:i + lookback]
                x_energy = energies[i:i + lookback]
                y_pos = positions[i + lookback:i + lookback + forecast]
                y_energy = energies[i + lookback:i + lookback + forecast]

                # Stack position and energy as 2 features
                x = np.stack([x_pos, x_energy], axis=1)
                y = np.stack([y_pos, y_energy], axis=1)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


# ---------- Model ----------

class EnergyLSTM(nn.Module):
    """LSTM-based sequence-to-sequence energy curve predictor."""

    def __init__(self, input_size: int = 2, hidden_size: int = 128,
                 num_layers: int = 2, forecast_len: int = 50, dropout: float = 0.2):
        super().__init__()
        self.forecast_len = forecast_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, forecast_len * 2)  # 2 = position + energy

    def forward(self, x):
        # x: (batch, seq_len, 2)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        out = self.fc(last_hidden)        # (batch, forecast_len * 2)
        return out.view(-1, self.forecast_len, 2)


# ---------- Training ----------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = EnergyCurveDataset(args.data, lookback=args.lookback,
                                  forecast=args.forecast, stride=args.stride)
    print(f"Dataset: {len(dataset)} samples")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = EnergyLSTM(
        input_size=2,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        forecast_len=args.forecast,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += criterion(pred, y).item()
        test_loss /= len(test_loader)

        scheduler.step(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), args.model_output)
            print(f"  → Best model saved (loss={best_loss:.6f})")

    print(f"\nTraining complete. Best test loss: {best_loss:.6f}")
    print(f"Model saved to: {args.model_output}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM energy curve predictor")
    parser.add_argument("--data", type=str, default="synthetic_data.json",
                        help="Training data file")
    parser.add_argument("--model-output", type=str, default="energy_lstm.pt",
                        help="Model output file")
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--forecast", type=int, default=50)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
