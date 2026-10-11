import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DB_PATH = "data/crypto.duckdb"
OUT_PATH = Path("docs/index.html")

# 1) Bring the price table up to date (fetches only missing finished days)
subprocess.run([sys.executable, "update_btc.py"], check=True)

con = duckdb.connect(DB_PATH)
con.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        as_of_date DATE PRIMARY KEY,   -- the close the prediction was made from
        made_at    TIMESTAMP,          -- when it was logged (UTC)
        pred       INTEGER,            -- 1 = UP, 0 = DOWN, for the following day
        prob_up    DOUBLE
    )
""")

# 2) Features (look backward only) and answer key
df = con.execute("SELECT * FROM btc_daily ORDER BY date").df()

df["ret_1d"] = df["close"].pct_change(1)
df["ret_3d"] = df["close"].pct_change(3)
df["ret_7d"] = df["close"].pct_change(7)
df["vol_7d"] = df["ret_1d"].rolling(7).std()
df["volume_ratio"] = df["volume"] / df["volume"].rolling(7).mean()
df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
for i in range(7):
    df[f"dow_{i}"] = (df["day_of_week"] == i).astype(int)

df["next_close"] = df["close"].shift(-1)
df["up_tomorrow"] = (df["next_close"] > df["close"]).astype(int)

FEATURES = ["ret_1d", "ret_3d", "ret_7d", "vol_7d", "volume_ratio"] + [f"dow_{i}" for i in range(7)]
feat_df = df.dropna(subset=FEATURES).reset_index(drop=True)
train = feat_df.dropna(subset=["next_close"])   # only days whose answer is known
live = feat_df.iloc[[-1]]                       # newest finished day: no answer yet
assert live["date"].iloc[0] == df["date"].iloc[-1]

# 3) Train on everything known, predict the day after the newest close
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(train[FEATURES], train["up_tomorrow"])
prob_up = float(model.predict_proba(live[FEATURES])[0, 1])
pred = int(model.predict(live[FEATURES])[0])

as_of = live["date"].iloc[0].date()
made_at = datetime.now(timezone.utc).replace(tzinfo=None)
before = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
con.execute("INSERT OR IGNORE INTO predictions VALUES (?, ?, ?, ?)", [as_of, made_at, pred, prob_up])
after = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
if after > before:
    print(f"Logged prediction made from the {as_of} close: {'UP' if pred else 'DOWN'} ({prob_up * 100:.1f}% up)")
else:
    print(f"A prediction from the {as_of} close is already logged; keeping the original.")

# 4) Read the log back, grading each prediction against what actually happened
log = con.execute("""
    SELECT p.as_of_date, p.pred, p.prob_up, d0.close AS close_d0, d1.close AS close_d1
    FROM predictions p
    JOIN btc_daily d0 ON d0.date = p.as_of_date
    LEFT JOIN btc_daily d1 ON d1.date = CAST(p.as_of_date + INTERVAL 1 DAY AS DATE)
    ORDER BY p.as_of_date
""").df()
con.close()

log["actual"] = log.apply(
    lambda r: None if pd.isna(r["close_d1"]) else int(r["close_d1"] > r["close_d0"]), axis=1)
graded = log[log["actual"].notna()]


def word(x):
    return "UP" if int(x) == 1 else "DOWN"


def cls(x):
    return "up" if int(x) == 1 else "down"


def target_day(as_of_date):
    return (pd.Timestamp(as_of_date) + pd.Timedelta(days=1)).date()


# Backtest summary from the walk-forward run (history, not live)
backtest = ""
csv_path = Path("walkforward_results.csv")
if csv_path.exists():
    r = pd.read_csv(csv_path)
    backtest = (
        f"<p><b>Backtest</b> ({str(r['date'].min())[:10]} to {str(r['date'].max())[:10]}, "
        f"{len(r):,} days, retraining every day): model {(r['pred'] == r['actual']).mean() * 100:.1f}%, "
        f"always guessing up {(r['actual'] == 1).mean() * 100:.1f}%. That gap is well within chance, "
        f"so expect today's call to be no better than a coin flip.</p>"
    )

# Cards: newest prediction and latest graded result
latest = log.iloc[-1]
card_pred = f"""<div class="card">
  <div class="label">Prediction for {target_day(latest['as_of_date'])} (UTC)</div>
  <div class="big {cls(latest['pred'])}">{word(latest['pred'])}</div>
  <div class="label">Model's chance of up: {latest['prob_up'] * 100:.1f}%. Reference: the {str(latest['as_of_date'])[:10]} close of ${latest['close_d0']:,.0f}.</div>
</div>"""

if len(graded):
    g = graded.iloc[-1]
    ok = int(g["pred"]) == int(g["actual"])
    card_result = f"""<div class="card">
  <div class="label">Latest result: {target_day(g['as_of_date'])}</div>
  <div class="big">{'&#10003; Right' if ok else '&#10007; Wrong'}</div>
  <div class="label">Predicted {word(g['pred'])}, actual {word(g['actual'])}.</div>
</div>"""
    n = len(graded)
    record = (f"<p><b>Live record:</b> {n} graded prediction{'s' if n != 1 else ''}. "
              f"Model {(graded['pred'] == graded['actual']).mean() * 100:.1f}% correct; "
              f"always guessing up would have scored {(graded['actual'] == 1).mean() * 100:.1f}%.</p>")
else:
    card_result = """<div class="card">
  <div class="label">Latest result</div>
  <div class="big">Pending</div>
  <div class="label">The first live prediction is graded once its day closes at midnight UTC.</div>
</div>"""
    record = "<p><b>Live record:</b> no graded predictions yet.</p>"

rows = ""
for _, r in log.tail(14).iloc[::-1].iterrows():
    if pd.isna(r["actual"]):
        actual_txt, result = "pending", "&hellip;"
    else:
        actual_txt = word(r["actual"])
        result = "&#10003;" if int(r["pred"]) == int(r["actual"]) else "&#10007;"
    rows += (f"<tr><td>{target_day(r['as_of_date'])}</td><td>{word(r['pred'])}</td>"
             f"<td>{r['prob_up'] * 100:.1f}%</td><td>{actual_txt}</td><td>{result}</td></tr>")

CSS = """
:root { --bg:#ffffff; --fg:#1a1a1a; --muted:#666; --card:#f4f4f5; --up:#12805c; --down:#c0392b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#111; --fg:#eee; --muted:#999; --card:#1e1e20; --up:#3ddc97; --down:#ff6b5e; }
}
body { font-family: system-ui, sans-serif; background:var(--bg); color:var(--fg);
       max-width:720px; margin:0 auto; padding:24px 16px; line-height:1.5; }
h1 { font-size:1.5rem; margin-bottom:4px; }
.sub { color:var(--muted); margin-top:0; }
.cards { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:20px 0; }
@media (max-width:560px) { .cards { grid-template-columns:1fr; } }
.card { background:var(--card); border-radius:12px; padding:16px; }
.label { color:var(--muted); font-size:.85rem; }
.big { font-size:2.2rem; font-weight:700; }
.up { color:var(--up); } .down { color:var(--down); }
table { width:100%; border-collapse:collapse; margin-top:8px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--card); font-size:.95rem; }
.note { color:var(--muted); font-size:.85rem; }
"""

html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bitcoin Daily Direction Predictor</title>
<style>{CSS}</style>
</head><body>
<h1>Bitcoin Daily Direction Predictor</h1>
<p class="sub">Will tomorrow's close be higher or lower than today's? Data through the {as_of} close (UTC).</p>
<div class="cards">{card_pred}{card_result}</div>
{record}
{backtest}
<h2>Recent predictions</h2>
<table>
<tr><th>Day</th><th>Predicted</th><th>Chance of up</th><th>Actual</th><th>Result</th></tr>
{rows}
</table>
<p class="note">"Up" means that day's UTC close was higher than the previous day's. Each prediction is logged
before its day finishes and is never edited. Only finished days are used. Built for learning, not financial advice.</p>
<p class="note">Updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC.</p>
</body></html>"""

OUT_PATH.parent.mkdir(exist_ok=True)
OUT_PATH.write_text(html, encoding="utf-8")
print(f"Wrote {OUT_PATH}")
