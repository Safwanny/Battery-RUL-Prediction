from __future__ import annotations
import pandas as pd
from dataclasses import dataclass


@dataclass
class ClipParams:
    bounds: dict  # {col: (low, high)}


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple derived features and normalizes sign conventions.
    """
    df = df.copy()
    # 1 if charging else 0; keep for EDA if you want, not necessary as model input
    df["Charge_Discharge"] = (df["Current_measured"] > 0).astype(int)
    # Absolute current magnitude
    df["Current_measured"] = df["Current_measured"].abs()
    return df


def fit_clip_params(df: pd.DataFrame, cols: list[str], lower_q: float, upper_q: float) -> ClipParams:
    bounds = {}
    for col in cols:
        lo, hi = df[col].quantile([lower_q, upper_q]).tolist()
        bounds[col] = (lo, hi)
    return ClipParams(bounds=bounds)


def apply_clip(df: pd.DataFrame, params: ClipParams) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in params.bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df
