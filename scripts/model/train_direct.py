"""
Train the direct 24-hour LSTM forecaster.

Run from the project root:
    python scripts/model/train_direct.py --days 90

Steps:
  1. Download the last N days of GOES X-ray flux (cached to data/processed/train_direct.csv)
  2. Build samples: 24h of history -> next 24 hourly averages
  3. Train on the older 85% of days, validate on the newest 15%
  4. Compare against a "tomorrow looks like the last hour" baseline
  5. Save the model to models/regressors/flux_lstm_direct24h.pt
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "backend"))
from model.direct import (DirectLSTM, MODEL_PATH, WINDOW, N_OUT, FLOOR,  # noqa: E402
                          to_log_bins, clean_minutes, flux_to_class)

CACHE = ROOT / "data" / "processed" / "train_direct.csv"


def download(days: int) -> pd.DataFrame:
    from collect.fetch import fetch_range_minute
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    parts, cur = [], start
    while cur < end:  # 10-day chunks keep the archive from timing out
        nxt = min(cur + timedelta(days=10), end)
        print(f"Fetching {cur:%Y-%m-%d} to {nxt:%Y-%m-%d} ...")
        try:
            parts.append(fetch_range_minute(cur, nxt - timedelta(minutes=1)))
        except Exception as e:
            print(f"  skipped chunk: {e}")
        cur = nxt
    df = pd.concat(parts, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def build_samples(df: pd.DataFrame, stride: int = 60):
    vals = df[["long_flux", "short_flux"]].to_numpy(float)
    log_long = np.log10(np.maximum(vals[:, 0], FLOOR))
    X, Y, starts = [], [], []
    for i in range(WINDOW, len(df) - N_OUT * 60 + 1, stride):
        X.append(to_log_bins(vals[i - WINDOW:i]))
        Y.append(log_long[i:i + N_OUT * 60].reshape(N_OUT, 60).mean(axis=1))
        starts.append(i)
    return np.array(X), np.array(Y), np.array(starts)


def class_acc(y_log_pred, y_log_true):
    p = np.vectorize(flux_to_class)(10 ** y_log_pred)
    t = np.vectorize(flux_to_class)(10 ** y_log_true)
    return (p == t).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--csv", help="use an existing minute CSV instead of downloading")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    if args.csv:
        raw = pd.read_csv(args.csv)
    elif CACHE.exists():
        print(f"Using cached data: {CACHE} (delete it to re-download)")
        raw = pd.read_csv(CACHE)
    else:
        raw = download(args.days)
    df = clean_minutes(raw)
    print(f"{len(df):,} minutes ({len(df) / 1440:.1f} days)")

    X, Y, starts = build_samples(df)
    split = int(len(X) * 0.85)
    # Gap of one day between train and validation so they share no data
    gap = 24
    Xtr, Ytr = X[:split - gap], Y[:split - gap]
    Xva, Yva = X[split:], Y[split:]
    print(f"{len(Xtr)} training samples, {len(Xva)} validation samples")

    mu = Xtr.reshape(-1, 2).mean(axis=0)
    sd = Xtr.reshape(-1, 2).std(axis=0) + 1e-6
    norm_x = lambda a: (a - mu) / sd
    norm_y = lambda a: (a - mu[0]) / sd[0]

    Xtr_t = torch.tensor(norm_x(Xtr), dtype=torch.float32)
    Ytr_t = torch.tensor(norm_y(Ytr), dtype=torch.float32)
    Xva_t = torch.tensor(norm_x(Xva), dtype=torch.float32)

    config = dict(hidden=64, layers=2, dropout=0.2)
    torch.manual_seed(0)
    model = DirectLSTM(**config)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss()

    best, best_state, wait = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for b in range(0, len(perm), 64):
            idx = perm[b:b + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), Ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t).numpy() * sd[0] + mu[0]
        val_mae = np.abs(pred_va - Yva).mean()
        sched.step(val_mae)
        print(f"Epoch {epoch + 1:2d}  val error (log10): {val_mae:.3f}")
        if val_mae < best - 1e-4:
            best, best_state, wait = val_mae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 8:
                print("Stopping early, validation stopped improving")
                break

    model.load_state_dict(best_state)
    with torch.no_grad():
        pred_va = model(Xva_t).numpy() * sd[0] + mu[0]
    baseline = np.repeat(Xva[:, -4:, 0].mean(axis=1, keepdims=True), N_OUT, axis=1)

    print("\nValidation results (newest 15% of days, never seen in training)")
    print(f"  Hourly class accuracy  model: {class_acc(pred_va, Yva):.1%}   baseline: {class_acc(baseline, Yva):.1%}")
    print(f"  Avg error (log10 flux) model: {np.abs(pred_va - Yva).mean():.3f}   baseline: {np.abs(baseline - Yva).mean():.3f}")
    err_by_hour = np.abs(pred_va - Yva).mean(axis=0)
    print(f"  Error at hour 1: {err_by_hour[0]:.3f}   hour 12: {err_by_hour[11]:.3f}   hour 24: {err_by_hour[23]:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": config,
                "mu": mu.tolist(), "sd": sd.tolist()}, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
