from __future__ import annotations
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_aware_split(df: pd.DataFrame, group_col: str, test_size: float, random_state: int):
    """
    Splits by groups so that entire batteries are either in train or test.
    """
    groups = df[group_col]
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def time_order_split(df: pd.DataFrame, test_ratio: float):
    """
    Splits by row order (already sorted by Time within each battery when loaded).
    Use when you only have 1 battery or explicit time series per file.
    """
    n = len(df)
    cut = int((1 - test_ratio) * n)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
