import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

con = duckdb.connect("data/crypto.duckdb", read_only=True)
df = con.execute("SELECT * FROM btc_daily ORDER BY date").df()
con.close()

# Features: look backward only
df["ret_1d"] = df["close"].pct_change(1)
df["ret_3d"] = df["close"].pct_change(3)
df["ret_7d"] = df["close"].pct_change(7)
df["vol_7d"] = df["ret_1d"].rolling(7).std()
df["volume_ratio"] = df["volume"] / df["volume"].rolling(7).mean()

df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
for i in range(7):
    df[f"dow_{i}"] = (df["day_of_week"] == i).astype(int)

# Answer key: the one place we look forward
df["next_close"] = df["close"].shift(-1)
df["up_tomorrow"] = (df["next_close"] > df["close"]).astype(int)

FEATURES = ["ret_1d", "ret_3d", "ret_7d", "vol_7d", "volume_ratio"] + [f"dow_{i}" for i in range(7)]
model_df = df.dropna(subset=FEATURES + ["next_close"]).reset_index(drop=True)

# Walk-forward: retrain every day on all earlier days, predict one day
MIN_TRAIN = 365
n = len(model_df)
print(f"Usable rows: {n}. Graded predictions: {n - MIN_TRAIN}")
print(f"First graded day: {model_df['date'].iloc[MIN_TRAIN].date()}\n")

records = []
for i in range(MIN_TRAIN, n):
    train = model_df.iloc[:i]
    today = model_df.iloc[i:i + 1]

    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(train[FEATURES], train["up_tomorrow"])

    records.append({
        "date": today["date"].iloc[0],
        "pred": int(model.predict(today[FEATURES])[0]),
        "prob_up": float(model.predict_proba(today[FEATURES])[0, 1]),
        "actual": int(today["up_tomorrow"].iloc[0]),
        "majority": int(train["up_tomorrow"].mean() >= 0.5),
        "repeat": int(today["ret_1d"].iloc[0] > 0),
    })

    done = i - MIN_TRAIN + 1
    if done % 100 == 0:
        so_far = pd.DataFrame(records)
        acc = (so_far["pred"] == so_far["actual"]).mean() * 100
        print(f"  {done} graded... model so far: {acc:.1f}%")

res = pd.DataFrame(records)
res["year"] = pd.to_datetime(res["date"]).dt.year
res["model_ok"] = res["pred"] == res["actual"]
res["up_ok"] = res["actual"] == 1
res["majority_ok"] = res["majority"] == res["actual"]
res["repeat_ok"] = res["repeat"] == res["actual"]

N = len(res)
se = np.sqrt(0.25 / N) * 100

print("\n--- Overall ---")
print("Graded predictions:", N)
print(f"Model:              {res['model_ok'].mean() * 100:5.1f} %")
print(f"Always up:          {res['up_ok'].mean() * 100:5.1f} %")
print(f"Past majority call: {res['majority_ok'].mean() * 100:5.1f} %")
print(f"Repeat today:       {res['repeat_ok'].mean() * 100:5.1f} %")
print(f"Model guessed up on {(res['pred'] == 1).mean() * 100:.1f} % of days")
print(f"Rough standard error: +/- {se:.1f} points")

print("\n--- By year (accuracy %) ---")
cols = ["model_ok", "up_ok", "majority_ok", "repeat_ok"]
by_year = res.groupby("year")[cols].mean().mul(100).round(1)
by_year.insert(0, "days", res.groupby("year").size())
print(by_year)

res.to_csv("walkforward_results.csv", index=False)
print("\nSaved every graded prediction to walkforward_results.csv")
