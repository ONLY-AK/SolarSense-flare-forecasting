# SolarSense

**Forecasting solar flare activity 24 hours ahead from live NASA/NOAA satellite data.**

SolarSense pulls minute-by-minute X-ray readings from the GOES satellites, trains two machine learning models on them, and predicts the next day of solar activity. Each forecast is converted into the official flare classes (A, B, C, M, X) and shown in a web dashboard next to what actually happened.

![PyTorch LSTM forecast vs. actual](visuals/forecast_validation_pytorch.png)

## Results

Tested on a full 24-hour day the models had never seen (1,440 minute-level predictions):

| Model | Hourly flare-class accuracy | Minute-level accuracy | Strength |
|---|---|---|---|
| Gradient boosting (scikit-learn) | ~96% | ~63% | Stable, reliable class predictions |
| LSTM neural network (PyTorch) | ~96% | ~64% | Follows the real shape of the flux curve |

The LSTM is the one used in the live web app because it tracks real trends instead of flattening them out.

## How it works

1. **Collect data.** Two weeks of historical GOES readings come from NASA's archive (via SunPy), and the latest 7 days come from NOAA's live Space Weather API. The two are merged, duplicates removed and overlapping readings averaged.
2. **Prepare it.** X-ray flux values are tiny (around 10⁻⁶), so they are log-transformed and normalized before training so the models can learn from them reliably.
3. **Train two models.**
   - *Gradient boosting:* looks at the last 12 hours and predicts the next minute.
   - *LSTM (2 layers, 256 units):* looks at the last 24 hours and predicts the next 60 minutes at once.
4. **Forecast recursively.** Each prediction is fed back in as input until a full 24 hours is covered. Results are converted back to real flux values and labelled with a flare class.
5. **Serve it.** A FastAPI backend runs the whole pipeline for any date on request. A React dashboard charts predicted vs. actual activity hour by hour and minute by minute, and shows NASA's SDO daily sun video for that day.

## Tech stack

**Machine learning:** PyTorch, scikit-learn, pandas, NumPy
**Data sources:** SunPy (NASA GOES archive), NOAA SWPC API, NASA SDO
**Backend:** FastAPI, Uvicorn
**Frontend:** React, Tailwind CSS, Recharts, Webpack
**Visualization:** Matplotlib

## Project structure

```
SolarSense/
├── scripts/
│   ├── collect/fetch.py         # Download and clean GOES data
│   ├── model/                   # Train and predict (scikit-learn + PyTorch)
│   └── backend/                 # FastAPI server used by the web app
├── test/                        # Accuracy checks against real data
├── visuals/                     # Plot scripts and result charts
├── models/regressors/           # Trained models and scalers
├── data/processed/              # Sample training, seed and forecast data
└── react-tailwind-vanilla/      # React dashboard
```

## Running it

Requires Python 3.10+ and Node.js 18+.

```bash
pip install -r requirements.txt
```

### Web app

```bash
# Terminal 1: backend (from scripts/backend)
cd scripts/backend
uvicorn api:app --reload --port 8000

# Terminal 2: frontend (from react-tailwind-vanilla)
cd react-tailwind-vanilla
npm install
npm start
```

Open http://localhost:8080 and use the arrow keys to move between days. Each new day takes about 1 to 2 minutes while it fetches data and runs the forecast.

### Full training pipeline

Run from the project root. Dates and output file names are set at the top of each script.

```bash
python scripts/collect/fetch.py          # run 3 times: training data, seed day, actual day
python scripts/model/train_pytorch.py    # or train_sklearn.py
python scripts/model/predict_pytorch.py  # or predict_sklearn.py
python test/process_prediction.py        # optional: average to hourly
python test/test_prediction.py           # accuracy vs. actual
python visuals/graph_pytorch.py          # or graph_sklearn.py
```

## Limitations and next steps

- **Rare big flares.** Most real data is A, B and C class, so the models see few M and X class events. Weighting or oversampling rare events is the next improvement.
- **LSTM overfitting.** Too many epochs on a small window overfits. Learning rate scheduling helps, but more and longer training data would help more.
- **Archive rate limits.** NASA's archive sometimes throttles large downloads, so fetching uses retries with backoff.

## Team

Built by **Amanuel Kassa**, **William Phan** and **Gurmukh Kharod** as a group project for CMPT 310 (Artificial Intelligence) at Simon Fraser University.

- Amanuel: data collection and cleaning, flare classification, recursive forecasting, React dashboard
- William: PyTorch LSTM, training optimization, forecasting, visualizations
- Gurmukh: scikit-learn model, flare classification, validation, React dashboard
