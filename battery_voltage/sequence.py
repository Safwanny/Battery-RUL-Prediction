from __future__ import annotations
import numpy as np
import pandas as pd


def make_sequences_per_battery(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    scaler_X,
    scaler_y
):
    """
    Builds sliding-window sequences per Battery_ID to avoid crossing boundaries.
    Returns X (N, seq_len, F), y (N,)
    """
    Xs, ys = [], []
    for bid, g in df.groupby("Battery_ID"):
        g = g.sort_values("Time")  # ensure time order
        Xg = g[feature_cols].values
        yg = g[[target_col]].values  # shape (len, 1)

        Xg_s = scaler_X.transform(Xg)
        yg_s = scaler_y.transform(yg).ravel()

        # build sequences
        for i in range(len(g) - seq_len):
            Xs.append(Xg_s[i:i+seq_len])
            ys.append(yg_s[i+seq_len])

    if not Xs:
        raise ValueError("No sequences could be constructed. Check seq_len vs data length.")

    return np.array(Xs), np.array(ys)
