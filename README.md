# Battery Voltage Forecasting (LSTM)

Forecast **battery voltage** from time-series sensor data (a stepping stone toward RUL prediction) using a leakage-safe, modular pipeline and an LSTM model.

---

## What’s been done & goal achieved
- **Goal:** Predict near-term **Voltage_measured** from multi-sensor time series; establish a solid base for future **RUL** modeling.
- **Built:**
  - Loader that ingests multiple CSVs, tags each as a unique **Battery_ID**, and sorts by **Time**.
  - Minimal feature step (+ optional quantile clipping **fit on train only**).
  - **Group-aware split** (entire batteries held out for test) to avoid leakage.
  - **Per-battery sliding-window sequences** for the LSTM (no sequence crosses batteries).
  - Training with early stopping; evaluation on an unseen test set; saved **model, scalers, metrics, plots**.

---

## How it works (brief)
1. **Load & prepare** → `battery_voltage/io.py`, `features.py`  
   Required CSV columns:  
   `Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, Time`
2. **Split without leakage** → `split.py`  
   Group split by `Battery_ID` (default) or time-ordered split.
3. **Scale & sequence** → `sequence.py`  
   Fit scalers on **train only**; build sequences of length `sequence_length`.
4. **Train LSTM** → `model.py`, `train.py`  
   `LSTM(64) → Dropout → LSTM(32) → Dense(16) → Dense(1)`, MSE loss, early stopping.
5. **Evaluate & save** → `evaluate.py`  
   Saves **RMSE/MAE/R²**, **Actual vs Predicted** plot, and training curve to `artifacts/`.

---

## Folder layout
```text
Battery-RUL-Prediction/
├─ battery_voltage/              # pipeline modules
│  ├─ io.py                      # load & merge CSVs, assign Battery_ID, sort by Time
│  ├─ features.py                # basic features + (optional) quantile clipping
│  ├─ split.py                   # group-aware or time-ordered split
│  ├─ sequence.py                # per-battery sequences (no boundary crossing)
│  ├─ model.py                   # LSTM architecture & compile
│  ├─ evaluate.py                # metrics + plots + JSON saving
│  ├─ utils.py                   # seeds, helpers
│  └─ train.py                   # end-to-end training pipeline
├─ configs/
│  └─ default.yaml               # paths + hyperparameters
├─ scripts/
│  ├─ train_lstm.py              # train CLI
│  └─ predict_lstm.py            # batch inference CLI for a new CSV folder
├─ requirements.txt
└─ artifacts/                    # (created at runtime) model, scalers, metrics, plots
```
---

## How to use & run

1) Set up environment
cd /ABSOLUTE/PATH/TO/Battery-RUL-Prediction
python -m venv .venv
macOS/Linux:
```bash
source .venv/bin/activate
```
Windows PowerShell:
```bash
.\.venv\Scripts\Activate.ps1
```
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

2) Configure paths & options
Edit configs/default.yaml:
```yaml
data_dir: "/ABSOLUTE/PATH/TO/YOUR/CSV/FOLDER"
artifacts_dir: "artifacts"
sequence_length: 30
epochs: 10          # small for a quick run; increase later
batch_size: 128
split_mode: "group" # or "time"
```

3) Train (from project root)
```bash
python -m scripts.train_lstm
```

Outputs → artifacts/
lstm_voltage.keras (model), scaler_X.joblib, scaler_y.joblib
metrics.json (RMSE/MAE/R²), actual_vs_pred.png, training_curve.png

Stop training: press Ctrl+C (terminal) or click Stop (PyCharm).
By default, artifacts are saved at the end of training.

change to location of dataset in default.yaml`/ABSOLUTE/PATH/TO/NEW/CSV_FOLDER`
Outputs → artifacts/inference/: predictions.csv, actual_vs_pred.png


## Notes
CSVs must include: Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, Time.
The pipeline is leakage-safe: clipping/scaling are fit only on train, test is untouched until final evaluation.
dataset is NASA battery dataset from kaggle
