import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
DB_PATH = "data/crypto.duckdb"
CHUNK_DAYS = 150
DAYS_BACK = 1825


def fetch_range(start, end):
    params = {
        "granularity": 86400,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    response = requests.get(URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


Path("data").mkdir(exist_ok=True)
con = duckdb.connect(DB_PATH)
con.execute("""
    CREATE TABLE IF NOT EXISTS btc_daily (
        date   DATE PRIMARY KEY,
        open   DOUBLE,
        high   DOUBLE,
        low    DOUBLE,
        close  DOUBLE,
        volume DOUBLE
    )
""")

today_utc = pd.Timestamp.now(tz="UTC").normalize()
last_date = con.execute("SELECT MAX(date) FROM btc_daily").fetchone()[0]

if last_date is None:
    start = today_utc - pd.Timedelta(days=DAYS_BACK)
    print(f"Table is empty: backfilling {DAYS_BACK} days.")
else:
    start = pd.Timestamp(last_date).tz_localize("UTC") + pd.Timedelta(days=1)
    print(f"Newest day on file: {last_date}")

if start >= today_utc:
    print("Already up to date. Nothing to fetch.")
else:
    rows = []
    chunk_start = start
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
    df = df[df["date"] < today_utc.date()]
    df = df[["date", "open", "high", "low", "close", "volume"]]

    con.register("new_rows", df)
    con.execute("INSERT OR IGNORE INTO btc_daily SELECT * FROM new_rows")
    print(f"Fetched {len(df)} finished days.")

print(con.execute(
    "SELECT COUNT(*) AS days, MIN(date) AS first_day, MAX(date) AS last_day FROM btc_daily"
).df())
con.close()
