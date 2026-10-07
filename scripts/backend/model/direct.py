# scripts/backend/model/direct.py
"""
Direct 24-hour forecaster.

Instead of predicting 60 minutes and feeding its own guesses back in 24 times
(which lets errors pile up), this model reads the last 24 hours once and
predicts all 24 upcoming hourly averages in a single pass.

Input : last 24h of GOES flux, averaged into 96 x 15-minute steps
        (log10 long flux, log10 short flux)
Output: next 24 hourly averages of log10 long flux
"""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT       = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "regressors" / "flux_lstm_direct24h.pt"

WINDOW      = 1440   # minutes of history the model reads
BIN         = 15     # minutes per input step
N_IN        = WINDOW // BIN   # 96 input steps
N_OUT       = 24     # hourly outputs
FLOOR       = 1e-9   # flux floor before log (below GOES noise level)

CLASS_THRESH = [(1e-4, "X"), (1e-5, "M"), (1e-6, "C"), (1e-7, "B"), (0.0, "A")]

def flux_to_class(f):
    for thr, label in CLASS_THRESH:
        if f >= thr:
            return label
    return "A"


class DirectLSTM(nn.Module):
    """LSTM encoder + linear head that predicts the change from the last hour."""
    def __init__(self, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, N_OUT))

    def forward(self, x):
        # x: (batch, N_IN, 2), already normalized
        out, _ = self.lstm(x)
        delta = self.head(out[:, -1, :])
        # Predict relative to the last hour of input, so "no change" is the easy default
        last_hour = x[:, -4:, 0].mean(dim=1, keepdim=True)
        return last_hour + delta


def to_log_bins(minute_vals: np.ndarray) -> np.ndarray:
    """(WINDOW, 2) raw flux -> (N_IN, 2) log10 15-minute means."""
    logv = np.log10(np.maximum(minute_vals, FLOOR))
    return logv.reshape(N_IN, BIN, 2).mean(axis=1)


def clean_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted 1-minute grid of long/short flux with gaps filled."""
    df = df[["timestamp", "long_flux", "short_flux"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for c in ("long_flux", "short_flux"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df.loc[df["long_flux"] <= 0, "long_flux"] = np.nan
    df.loc[df["short_flux"] <= 0, "short_flux"] = np.nan
    df = (df.drop_duplicates("timestamp").sort_values("timestamp")
            .set_index("timestamp").asfreq("1min")
            .interpolate(method="time").ffill().bfill()
            .reset_index())
    return df


@lru_cache(maxsize=1)
def _load():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = DirectLSTM(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, np.array(ckpt["mu"]), np.array(ckpt["sd"])


def predict_hourly(seed_df: pd.DataFrame) -> np.ndarray:
    """Return 24 predicted hourly mean long-flux values (linear units)."""
    model, mu, sd = _load()
    seed = clean_minutes(seed_df).iloc[-WINDOW:]
    if len(seed) < WINDOW:
        raise RuntimeError(f"Need {WINDOW} seed minutes; got {len(seed)}")
    x = (to_log_bins(seed[["long_flux", "short_flux"]].to_numpy(float)) - mu) / sd
    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).numpy()[0]
    return 10.0 ** (y * sd[0] + mu[0])


def predict_from_seed_df(seed_df: pd.DataFrame, horizon: int = 1440) -> pd.DataFrame:
    """
    Same interface the API used before: minute-level rows
    ['timestamp', 'long_flux_pred', 'goes_class_pred'].
    Minute values are a smooth curve through the 24 hourly predictions.
    """
    hourly = predict_hourly(seed_df)
    t0 = pd.to_datetime(seed_df["timestamp"], utc=True).max() + pd.Timedelta(minutes=1)
    minutes = np.arange(horizon)
    centers = np.arange(N_OUT) * 60 + 30
    log_min = np.interp(minutes, centers, np.log10(hourly))
    out = pd.DataFrame({
        "timestamp": pd.date_range(t0, periods=horizon, freq="1min"),
        "long_flux_pred": 10.0 ** log_min,
    })
    out["goes_class_pred"] = out["long_flux_pred"].apply(flux_to_class)
    return out
