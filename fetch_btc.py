import time
import requests
import pandas as pd

URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
CHUNK_DAYS = 150
DAYS_BACK = 365

def fetch_range(start, end):
    params = {
        "granularity": 86400,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    response = requests.get(URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

today_utc = pd.Timestamp.now(tz="UTC").normalize()
window_start = today_utc - pd.Timedelta(days=DAYS_BACK)

rows = []
chunk_start = window_start
while chunk_start < today_utc:
    chunk_end = min(chunk_start + pd.Timedelta(days=CHUNK_DAYS), today_utc)
    batch = fetch_range(chunk_start, chunk_end)
    print(f"{chunk_start.date()} -> {chunk_end.date()}: {len(batch)} candles")
    rows.extend(batch)
    chunk_start = chunk_end
    time.sleep(0.5)

df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
df = df.drop_duplicates(subset="time")
df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.date
df = df.sort_values("date").reset_index(drop=True)
df = df[df["date"] < today_utc.date()].reset_index(drop=True)

print(df[["date", "open", "high", "low", "close", "volume"]].head(3))
print(df[["date", "open", "high", "low", "close", "volume"]].tail(3))
print("Rows:", len(df))

dates = pd.to_datetime(df["date"])
print("\n--- Integrity checks ---")
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate dates:", int(df["date"].duplicated().sum()))
print("Gaps between days (should be only 1):", dates.diff().dropna().dt.days.unique())
print("Days with zero volume:", int((df["volume"] == 0).sum()))
