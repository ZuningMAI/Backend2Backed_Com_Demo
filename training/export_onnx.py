"""
Export trained PyTorch model to ONNX format for Qt6/ONNX Runtime inference.
"""
import argparse
import torch

from train import EnergyLSTM


def export(model_path: str, onnx_path: str, lookback: int = 100,
           forecast: int = 50, hidden_size: int = 128, num_layers: int = 2):
    device = torch.device("cpu")

    # Load model
    model = EnergyLSTM(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        forecast_len=forecast,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Create dummy input: (batch=1, seq_len=lookback, features=2)
    dummy_input = torch.randn(1, lookback, 2, device=device)

    # Export to ONNX (use older export path for compatibility)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"ONNX model exported to: {onnx_path}")

    # Verify
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification: OK")

    # Test inference consistency
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {"input": dummy_input.numpy()})[0]
    torch_out = model(dummy_input).detach().numpy()
    diff = float(((onnx_out - torch_out) ** 2).mean())
    print(f"ONNX vs PyTorch MSE: {diff:.10f} (should be ~0)")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", type=str, default="energy_lstm.pt",
                        help="Trained PyTorch model path")
    parser.add_argument("--onnx", type=str, default="energy_lstm.onnx",
                        help="Output ONNX model path")
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--forecast", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()

    export(args.model, args.onnx, args.lookback, args.forecast,
           args.hidden_size, args.num_layers)


if __name__ == "__main__":
    main()
