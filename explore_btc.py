import duckdb
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

# Day of week, one-hot coded (dow_0 = Monday ... dow_6 = Sunday)
df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
for i in range(7):
    df[f"dow_{i}"] = (df["day_of_week"] == i).astype(int)

# Answer key: the one place we look forward
df["next_close"] = df["close"].shift(-1)
df["up_tomorrow"] = (df["next_close"] > df["close"]).astype(int)

FEATURES = ["ret_1d", "ret_3d", "ret_7d", "vol_7d", "volume_ratio"] + [f"dow_{i}" for i in range(7)]

model_df = df.dropna(subset=FEATURES + ["next_close"]).reset_index(drop=True)

# Chronological split: learn from the past, get graded on the most recent days
TEST_DAYS = 90
train = model_df.iloc[:-TEST_DAYS]
test = model_df.iloc[-TEST_DAYS:]

print("Usable rows:", len(model_df))
print("Train:", len(train), train["date"].min(), "->", train["date"].max())
print("Test: ", len(test), test["date"].min(), "->", test["date"].max())

# Baselines on the test period
print("\n--- Baselines on the test period ---")
majority = int(train["up_tomorrow"].mean() >= 0.5)
print("Always guess up:         ", round((test["up_tomorrow"] == 1).mean() * 100, 1), "%")
print("Always guess down:       ", round((test["up_tomorrow"] == 0).mean() * 100, 1), "%")
print("Train's most common call:", round((test["up_tomorrow"] == majority).mean() * 100, 1), "%")
print("Repeat today's direction:", round(((test["ret_1d"] > 0).astype(int) == test["up_tomorrow"]).mean() * 100, 1), "%")

# The model
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(train[FEATURES], train["up_tomorrow"])

preds = model.predict(test[FEATURES])
print("\n--- Model on the test period ---")
print("Accuracy:", round((preds == test["up_tomorrow"]).mean() * 100, 1), "%")
print("Share of test days it guessed up:", round((preds == 1).mean() * 100, 1), "%")

print("\nWhat it learned (positive = pushes toward 'up'):")
coefs = model.named_steps["logisticregression"].coef_[0]
for name, c in zip(FEATURES, coefs):
    print(f"  {name:14s} {c:+.3f}")
