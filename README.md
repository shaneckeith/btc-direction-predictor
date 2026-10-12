# Bitcoin Daily Direction Predictor

A small machine learning project that predicts, every day, whether Bitcoin's next daily close will be higher or lower than today's, logs the prediction before the day finishes, and grades itself in public.

**Live page:** https://shanekeith.com/btc/

## What it does

Each night a scheduled job fetches the newest finished day of Bitcoin prices, trains a model on all finished days, and predicts the direction of the next day. The prediction is saved to a log that is only ever added to. Once that day closes, the next run compares the prediction with what actually happened, and the page shows the running record next to a simple baseline: always guessing "up."

## Does it work?

No edge has been found as of yet. A walk-forward backtest retrained the model every day on only the days before the one being predicted. It covered 1,452 predictions from 2022-10-19 to 2026-10-09.

| Strategy | Accuracy |
|---|---|
| Model | 50.3% |
| Always guess up | 50.0% |
| Repeat yesterday's direction | 47.9% |

The standard error at that sample size is about 1.3 points, so the model and the baseline are statistically indistinguishable. Daily Bitcoin direction is close to a coin flip, and these six simple features don't change that. The project is built so that result can be checked and published, not hidden.

An earlier single 90-day test scored the model at 55.6%. Across the full walk-forward run, that gap disappeared, which is a good reminder of how much small samples can mislead.

## How it works

- **Data:** daily BTC-USD candles from Coinbase's public Exchange API, five years back from 2021-10-12. Days are UTC days, and only finished days are stored.
- **Storage:** a DuckDB file, `data/crypto.duckdb`, with a `btc_daily` price table and an append-only `predictions` table.
- **Features** (all look backward only):
  - Returns over the last 1, 3, and 7 days
  - 7-day volatility, the standard deviation of daily returns
  - Volume relative to its 7-day average
  - Day of week, one-hot encoded
- **Target:** 1 if the next day's close is above today's close, otherwise 0.
- **Model:** logistic regression with standardized features (scikit-learn), retrained from scratch on every run.
- **Honest grading:** the `predictions` table uses the date as its primary key and inserts with `INSERT OR IGNORE`, so a prediction can't be overwritten. Results are never written back to the log. The page works out each result by comparing the logged prediction with the stored closes when it is built.
- **Automation:** a GitHub Actions workflow runs shortly after 00:00 UTC, fetches new prices, logs the new prediction, rebuilds the page, and commits the database and `docs/index.html`. GitHub Pages serves the `docs/` folder.

## Files

| File | Purpose |
|---|---|
| `update_btc.py` | Fetches only the missing finished days from Coinbase into `btc_daily` |
| `daily_btc.py` | The nightly job: update prices, train, predict, log, rebuild the page |
| `walkforward_btc.py` | The backtest that retrains daily and writes `walkforward_results.csv` |
| `explore_btc.py` | Single train/test split with baselines, used for exploration |
| `fetch_btc.py` | The first data pull, kept as a record of how the data was checked |
| `docs/index.html` | The generated page |

## Run it yourself

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 walkforward_btc.py
```

The backtest is safe to run anywhere. Running `daily_btc.py` logs a real prediction into your local copy of the database, so use a separate clone for experiments.

## Limits

- Built for learning, not financial advice.
- The log starts with the prediction for 2026-10-12. A test prediction made before launch was cleared, so it is not part of the record.
- The model uses six simple price and volume features. It does not use news, order books, or other markets.
