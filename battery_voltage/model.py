from __future__ import annotations
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm(input_shape, cfg):
    """
    input_shape = (seq_len, n_features)
    """
    mcfg = cfg["model"]
    model = Sequential([
        LSTM(mcfg["lstm1"], return_sequences=True, input_shape=input_shape),
        Dropout(mcfg["dropout"]),
        LSTM(mcfg["lstm2"], return_sequences=False),
        Dense(mcfg["dense"], activation="relu"),
        Dense(1)
    ])
    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=cfg["learning_rate"]),
        metrics=["mae"]
    )
    return model
