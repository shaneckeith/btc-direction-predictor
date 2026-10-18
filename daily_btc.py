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
README_URL = "https://github.com/shaneckeith/btc-direction-predictor/blob/main/README.md"

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
    SELECT p.as_of_date, p.made_at, p.pred, p.prob_up, d0.close AS close_d0, d1.close AS close_d1
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
bt_df = None
csv_path = Path("walkforward_results.csv")
if csv_path.exists():
    bt_df = pd.read_csv(csv_path)
    brier_txt = ""
    if "prob_up" in bt_df.columns:
        brier = ((bt_df["prob_up"] - bt_df["actual"]) ** 2).mean()
        brier_txt = (f" <b>Brier score: {brier:.3f}</b> (lower is better; always saying 50% scores 0.250; "
                     f"<a href='https://en.wikipedia.org/wiki/Brier_score' target='_blank' rel='noopener'>what's a Brier score?</a>).")
    backtest = (
        f"<p><b>Backtest</b> ({str(bt_df['date'].min())[:10]} to {str(bt_df['date'].max())[:10]}, "
        f"{len(bt_df):,} days, retraining every day): model {(bt_df['pred'] == bt_df['actual']).mean() * 100:.1f}%, "
        f"always guessing up {(bt_df['actual'] == 1).mean() * 100:.1f}%. That gap is well within chance, "
        f"so expect today's call to be no better than a coin flip.{brier_txt}</p>"
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
    logged = pd.Timestamp(r["made_at"]).strftime("%Y-%m-%d %H:%M")
    rows += (f"<tr><td>{target_day(r['as_of_date'])}</td><td>{word(r['pred'])}</td>"
             f"<td>{r['prob_up'] * 100:.1f}%</td><td>{actual_txt}</td><td>{result}</td>"
             f"<td>{logged}</td></tr>")


# Chart: running total of the model's lead over "always guess up"
def lead_steps(pred_col, actual_col):
    # +1 when the model was right and always-up was wrong, -1 the reverse, 0 when they agree
    return (pred_col == actual_col).astype(int) - (actual_col == 1).astype(int)


bt_pts, live_pts, base = [], [], 0
if bt_df is not None and len(bt_df):
    steps = lead_steps(bt_df["pred"].astype(int), bt_df["actual"].astype(int)).cumsum()
    days = pd.to_datetime(bt_df["date"]) + pd.Timedelta(days=1)
    bt_pts = list(zip(days, steps))
    base = int(steps.iloc[-1])
if len(graded):
    steps = lead_steps(graded["pred"].astype(int), graded["actual"].astype(int)).cumsum() + base
    days = pd.to_datetime(graded["as_of_date"]) + pd.Timedelta(days=1)
    live_pts = list(zip(days, steps))


def advantage_chart(bt_points, live_points):
    pts = bt_points + live_points
    if not pts:
        return ""
    xs = [p[0].toordinal() for p in pts]
    ys = [float(p[1]) for p in pts] + [0.0]
    x0, x1 = min(xs), max(xs)
    lo, hi = min(ys), max(ys)
    if hi == lo:
        hi = lo + 1
    pad = (hi - lo) * 0.12
    lo, hi = lo - pad, hi + pad
    W, H, L, R, T, B = 640, 220, 44, 12, 12, 28

    def sx(d):
        return L + (d - x0) / max(x1 - x0, 1) * (W - L - R)

    def sy(v):
        return T + (hi - v) / (hi - lo) * (H - T - B)

    def poly(points):
        return " ".join(f"{sx(p[0].toordinal()):.1f},{sy(float(p[1])):.1f}" for p in points)

    zero_y = sy(0)
    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Running total of the model lead over always guessing up">']
    parts.append(f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{H - B}"/>')
    parts.append(f'<line class="zero" x1="{L}" y1="{zero_y:.1f}" x2="{W - R}" y2="{zero_y:.1f}"/>')
    parts.append(f'<text class="tick" x="{L - 6}" y="{zero_y + 4:.1f}" text-anchor="end">0</text>')
    for v in (round(max(ys)), round(min(ys))):
        if abs(sy(v) - zero_y) > 16:
            parts.append(f'<text class="tick" x="{L - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:+d}</text>')
    parts.append(f'<text class="tick" x="{L}" y="{H - 8}">{pts[0][0].date()}</text>')
    parts.append(f'<text class="tick" x="{W - R}" y="{H - 8}" text-anchor="end">{pts[-1][0].date()}</text>')
    if bt_points:
        parts.append(f'<polyline class="bt" points="{poly(bt_points)}"/>')
    if live_points:
        # The live line starts at its own first point and is not joined to the backtest line
        parts.append(f'<polyline class="lv" points="{poly(live_points)}"/>')
        last = live_points[-1]
        parts.append(f'<circle class="lvdot" cx="{sx(last[0].toordinal()):.1f}" '
                     f'cy="{sy(float(last[1])):.1f}" r="4.5"/>')
        lx = sx(live_points[0][0].toordinal())
        parts.append(f'<line class="axis" x1="{lx:.1f}" y1="{T}" x2="{lx:.1f}" y2="{H - B}" stroke-dasharray="2 3"/>')
        parts.append(f'<text class="livelabel" x="{lx - 4:.1f}" y="{T + 10}" text-anchor="end">live &rarr;</text>')
    parts.append("</svg>")
    return "".join(parts)


gap_note = ""
if bt_pts and live_pts:
    missing = (live_pts[0][0] - bt_pts[-1][0]).days - 1
    if missing > 0:
        first_missing = (bt_pts[-1][0] + pd.Timedelta(days=1)).date()
        gap_note = (f" No predictions were logged for {missing} day{'s' if missing != 1 else ''} "
                    f"between them (starting {first_missing}), so the two lines are not joined.")

chart_html = ""
chart_svg = advantage_chart(bt_pts, live_pts)
if chart_svg:
    chart_html = f"""<h2>Model vs always-up</h2>
{chart_svg}
<p class="note">Running total of the model's lead over always guessing up. Each graded day moves the line
up one step when the model was right and always-up was wrong, down one step when it was the other way around,
and not at all when they agreed. Above the dashed zero line, the model is ahead. Blue is the backtest and
orange is the live record.{gap_note}</p>"""

how_html = f"""<h2>How it works</h2>
<p>Each night a logistic regression model is retrained from scratch on every finished day of Bitcoin history
(daily candles from Coinbase, UTC days, back to October 2021). It sees six backward-looking facts about each day:
returns over 1, 3, and 7 days, 7-day volatility, volume relative to its 7-day average, and the day of the week.
It predicts whether the next day's close will be higher than today's. The backtest retrains on every day using
only the days before it. <a href="{README_URL}" target="_blank" rel="noopener">Code and full write-up on GitHub.</a></p>"""

COIN_SVG = """<svg class="coin" viewBox="0 0 48 48" width="40" height="40" role="img" aria-label="A coin">
<circle class="coin-face" cx="24" cy="24" r="22"/>
<circle class="coin-ring" cx="24" cy="24" r="17"/>
<text class="coin-b" x="24" y="31" text-anchor="middle">B</text>
</svg>"""

CSS = """
:root { --bg:#ffffff; --fg:#1a1a1a; --muted:#666; --card:#f4f4f5; --up:#12805c; --down:#c0392b;
        --bt:#2563eb; --live:#d97706; --coin:#fbbf24; --coin-edge:#b45309; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#111; --fg:#eee; --muted:#999; --card:#1e1e20; --up:#3ddc97; --down:#ff6b5e;
          --bt:#6ea8ff; --live:#fbbf24; }
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
.tablewrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; margin-top:8px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--card); font-size:.95rem; white-space:nowrap; }
.note { color:var(--muted); font-size:.85rem; }
.chart { width:100%; height:auto; display:block; margin-top:8px; }
.axis { stroke:var(--muted); stroke-width:1; opacity:.5; }
.zero { stroke:var(--muted); stroke-width:1; stroke-dasharray:4 3; }
.bt { fill:none; stroke:var(--bt); stroke-width:1.75; }
.lv { fill:none; stroke:var(--live); stroke-width:2.5; }
.lvdot { fill:var(--live); stroke:var(--bg); stroke-width:1.5; }
.tick { fill:var(--muted); font-size:11px; }
.livelabel { fill:var(--live); font-size:11px; font-weight:600; }
a { color:var(--bt); }
.back { margin:0 0 12px; font-size:.9rem; }
.titlerow { display:flex; align-items:center; gap:12px; margin-bottom:4px; }
.titlerow h1 { margin:0; }
.coin { flex:none; }
.coin-face { fill:var(--coin); stroke:var(--coin-edge); stroke-width:2; }
.coin-ring { fill:none; stroke:var(--coin-edge); stroke-width:1.5; opacity:.6; }
.coin-b { fill:var(--coin-edge); font:700 22px system-ui, sans-serif; }
"""

html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bitcoin Flip</title>
<style>{CSS}</style>
</head><body>
<p class="back"><a href="https://shanekeith.com/" target="_top">&larr; shanekeith.com</a></p>
<div class="titlerow">{COIN_SVG}<h1>Bitcoin Flip</h1></div>
<p class="sub">Bitcoin Daily Direction Predictor</p>
<p>Will tomorrow's close be higher or lower than today's? Data through the {as_of} close (UTC).</p>
<div class="cards">{card_pred}{card_result}</div>
{record}
{backtest}
{chart_html}
<h2>Recent predictions</h2>
<div class="tablewrap">
<table>
<tr><th>Day</th><th>Predicted</th><th>Chance of up</th><th>Actual</th><th>Result</th><th>Logged (UTC)</th></tr>
{rows}
</table>
</div>
{how_html}
<p class="note">"Up" means that day's UTC close was higher than the previous day's. Each prediction is logged
before its day finishes, and the code never overwrites a logged prediction. Every nightly update is also a public
commit in the GitHub repository, so the logging times can be checked against its history. Only finished days are
used. Built for learning, not financial advice.</p>
<p class="note">Updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC.</p>
</body></html>"""

OUT_PATH.parent.mkdir(exist_ok=True)
OUT_PATH.write_text(html, encoding="utf-8")
print(f"Wrote {OUT_PATH}")
