"""
Example: quick batch inference on a single CSV folder (same schema).
Saves predictions vs actual CSV and a scatter plot.

Usage:
  python scripts/predict_lstm.py /path/to/new_csv_folder
"""
import sys
from pathlib import Path
import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from battery_voltage.io import load_and_merge
from battery_voltage.features import add_basic_features
from battery_voltage.sequence import make_sequences_per_battery


def main(data_dir: str, cfg_path: str = "configs/default.yaml", artifacts: str = "artifacts"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    feature_cols = cfg["feature_cols"]
    target_col = cfg["target_col"]
    seq_len = int(cfg["sequence_length"])

    # Load artifacts
    model = load_model(Path(artifacts) / "lstm_voltage.keras")
    scaler_X = joblib.load(Path(artifacts) / "scaler_X.joblib")
    scaler_y = joblib.load(Path(artifacts) / "scaler_y.joblib")

    # Load data
    df = load_and_merge(data_dir, required_cols=cfg["required_features"])
    df = add_basic_features(df)

    # Build sequences
    X_seq, y_seq = make_sequences_per_battery(
        df, feature_cols, target_col, seq_len, scaler_X, scaler_y
    )

    y_pred_s = model.predict(X_seq).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_seq.reshape(-1, 1)).ravel()

    # Save outputs
    out_dir = Path(artifacts) / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out_dir / "predictions.csv", index=False)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle="--")
    plt.xlabel("Actual (Volts)")
    plt.ylabel("Predicted (Volts)")
    plt.title("Inference: Actual vs Predicted Voltage")
    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_pred.png")
    plt.close()

    print(f"Saved predictions to: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_lstm.py /path/to/new_csv_folder")
        sys.exit(1)
    main(sys.argv[1])
