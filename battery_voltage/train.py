from __future__ import annotations
from pathlib import Path
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .io import load_and_merge
from .features import add_basic_features, fit_clip_params, apply_clip
from .split import group_aware_split, time_order_split
from .sequence import make_sequences_per_battery
from .model import build_lstm
from .evaluate import compute_metrics, plot_actual_vs_pred, save_metrics_json
from .utils import ensure_dir, set_seed


def train(cfg_path: str = "configs/default.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    set_seed(cfg.get("random_state", 42))

    data_dir = cfg["data_dir"]
    required = cfg["required_features"]
    feature_cols = cfg["feature_cols"]
    target_col = cfg["target_col"]
    seq_len = int(cfg["sequence_length"])
    test_ratio = float(cfg["test_ratio"])
    split_mode = cfg.get("split_mode", "group")

    art_dir = Path(cfg["artifacts_dir"])
    ensure_dir(art_dir)

    # 1) Load
    df = load_and_merge(data_dir, required_cols=required)
    # Make per-file battery groups. Already assigned in io.py.

    # 2) Basic features
    df = add_basic_features(df)

    # 3) Split (BEFORE fitting clipping/scalers to avoid leakage)
    if split_mode == "group":
        train_df, test_df = group_aware_split(
            df, group_col="Battery_ID",
            test_size=test_ratio,
            random_state=cfg.get("random_state", 42)
        )
    else:
        train_df, test_df = time_order_split(df, test_ratio=test_ratio)

    # 4) Fit clipping on TRAIN only, then apply to both
    clip_cfg = cfg.get("clip", {"enabled": False})
    clip_params = None
    if clip_cfg.get("enabled", False):
        cols_to_clip = [c for c in clip_cfg["columns"] if c in train_df.columns]
        clip_params = fit_clip_params(
            train_df, cols=cols_to_clip,
            lower_q=float(clip_cfg["lower_q"]),
            upper_q=float(clip_cfg["upper_q"])
        )
        train_df = apply_clip(train_df, clip_params)
        test_df = apply_clip(test_df, clip_params)

    # 5) Prepare arrays & scalers (FIT ONLY ON TRAIN)
    X_train = train_df[feature_cols].values
    y_train = train_df[[target_col]].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[[target_col]].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(X_train)
    scaler_y.fit(y_train)

    # 6) Build sequences (per battery, no cross-boundaries)
    Xtr_seq, ytr_seq = make_sequences_per_battery(
        train_df, feature_cols, target_col, seq_len, scaler_X, scaler_y
    )
    Xte_seq, yte_seq = make_sequences_per_battery(
        test_df, feature_cols, target_col, seq_len, scaler_X, scaler_y
    )

    # 7) Model
    model = build_lstm(input_shape=(seq_len, Xtr_seq.shape[-1]), cfg=cfg)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=int(cfg["patience"]), restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]

    # 8) Train (validation from TRAIN; test is untouched)
    history = model.fit(
        Xtr_seq, ytr_seq,
        validation_split=float(cfg["validation_split"]),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        callbacks=callbacks,
        verbose=1
    )

    # 9) Evaluate on TEST (convert back to Volts)
    y_pred_s = model.predict(Xte_seq).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(yte_seq.reshape(-1, 1)).ravel()

    metrics = compute_metrics(y_true, y_pred)
    print("Test metrics:", metrics)

    # 10) Save artifacts
    ensure_dir(art_dir)
    model.save(art_dir / "lstm_voltage.keras")
    joblib.dump(scaler_X, art_dir / "scaler_X.joblib")
    joblib.dump(scaler_y, art_dir / "scaler_y.joblib")

    # Persist clipping bounds (if used)
    if clip_params is not None:
        import json
        (art_dir / "clip_bounds.json").write_text(json.dumps(clip_params.bounds, indent=2))

    # Metrics + plot
    save_metrics_json(metrics, art_dir / "metrics.json")
    plot_actual_vs_pred(y_true, y_pred, art_dir / "actual_vs_pred.png")

    # Optional: save training loss curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(art_dir / "training_curve.png")
    plt.close()

    return metrics
