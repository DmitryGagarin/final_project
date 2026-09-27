"""
Build a fresh evaluation gold set from Finnhub news headlines.

Uses DIFFERENT tickers than build_corpus.py, so no row here overlaps with
the training corpus. Headlines are human-written abstractive summaries,
which makes them a legitimate reference standard.
"""
import time
from datetime import datetime, timedelta

import finnhub
import pandas as pd

FINNHUB_KEY = "daikevpr01qqjcj45d60daikevpr01qqjcj45d6g"

# Completely different from the training tickers in build_corpus.py
GOLD_TICKERS = [
    "ORCL", "CRM", "ADBE", "AVGO", "QCOM", "TXN", "MU", "IBM",
    "CSCO", "UBER", "ABNB", "PYPL", "SHOP", "COIN",
    "GS", "MS", "WFC", "C", "SCHW", "BLK",
    "LLY", "UNH", "MRK", "ABBV", "TMO",
    "HON", "CAT", "DE", "LMT", "RTX",
]

TARGET_ROWS = 250  # cap the gold set at ~250 examples

client = finnhub.Client(api_key=FINNHUB_KEY)

end = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

rows = []
for ticker in GOLD_TICKERS:
    if len(rows) >= TARGET_ROWS:
        break
    try:
        news = client.company_news(ticker, _from=start, to=end)
    except Exception as e:
        print(f"{ticker}: {e}")
        continue

    kept = 0
    for art in news:
        headline = (art.get("headline") or "").strip()
        content = (art.get("summary") or "").strip()
        # Require real abstractive references:
        if len(content) < 100 or len(headline) < 25:
            continue
        # A summary should be shorter than the text
        if len(headline) >= len(content):
            continue
        rows.append({
            "cleaned_text": content,
            "summary": headline,
            "ticker": ticker,
            "published": art.get("datetime", ""),
            "source": art.get("source", ""),
        })
        kept += 1
        if len(rows) >= TARGET_ROWS:
            break
    print(f"{ticker}: +{kept} (total {len(rows)})")
    time.sleep(1.1)  # 60 req/min

if not rows:
    print("No gold rows collected. Check the Finnhub key.")
    raise SystemExit(2)

df = pd.DataFrame(rows)
df.to_csv("gold_set_v2.csv", sep=";", index=False)
print(f"Wrote {len(df)} gold examples → gold_set_v2.csv")
print(f"Avg text len:    {df['cleaned_text'].str.len().mean():.0f} chars")
print(f"Avg summary len: {df['summary'].str.len().mean():.0f} chars")
