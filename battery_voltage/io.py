from pathlib import Path
import glob
import pandas as pd


def load_and_merge(directory: str, required_cols: list[str]) -> pd.DataFrame:
    """
    Loads all CSV files that contain the required columns.
    Adds a Battery_ID based on file order (Battery_1, Battery_2, ...).
    Sorts within each battery by Time and concatenates.
    """
    directory = Path(directory)
    files = sorted(glob.glob(str(directory / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    dfs = []
    for i, f in enumerate(files, start=1):
        # Quick header read to check columns
        head = pd.read_csv(f, nrows=1)
        if not set(required_cols).issubset(set(head.columns)):
            # Skip files that don't have all required columns
            continue
        df = pd.read_csv(f, usecols=required_cols)
        df["Battery_ID"] = f"Battery_{i}"
        # Ensure numeric Time sort; if Time is not numeric, try to coerce
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df = df.sort_values(["Time"]).reset_index(drop=True)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid CSVs contained the required columns.")

    final_df = pd.concat(dfs, ignore_index=True)
    # Ensure column order: required + Battery_ID
    cols = [c for c in required_cols] + ["Battery_ID"]
    final_df = final_df[cols]
    return final_df
