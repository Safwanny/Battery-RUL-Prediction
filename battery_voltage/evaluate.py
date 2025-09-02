# battery_voltage/evaluate.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _to_1d_numeric(a):
    """Convert input to a 1-D float64 numpy array."""
    a = np.asarray(a)
    if a.ndim > 1:
        a = a.reshape(-1)
    return a.astype(np.float64, copy=False)


def compute_metrics(y_true, y_pred) -> dict:
    """
    Robust, version-agnostic metrics:
    - enforce 1-D float arrays
    - drop NaN/inf pairs
    - compute RMSE = sqrt(MSE) (no 'squared' kwarg)
    """
    y_true = _to_1d_numeric(y_true)
    y_pred = _to_1d_numeric(y_pred)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        raise ValueError("No finite values to score.")
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.shape[0] != y_pred.shape[0]:
        n = min(y_true.shape[0], y_pred.shape[0])
        y_true, y_pred = y_true[:n], y_pred[:n]

    mse  = mean_squared_error(y_true, y_pred)     # no 'squared' kwarg (works on old sklearn)
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def plot_actual_vs_pred(y_true, y_pred, out_path: Path):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims, linestyle="--")
    plt.xlabel("Actual (Volts)")
    plt.ylabel("Predicted (Volts)")
    plt.title("Actual vs Predicted Voltage")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_metrics_json(metrics: dict, out_path: Path):
    out_path.write_text(json.dumps(metrics, indent=2))
