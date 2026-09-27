import time
from datetime import datetime, timedelta

import finnhub
import pandas as pd

FINNHUB_KEY = "daikevpr01qqjcj45d60daikevpr01qqjcj45d6g"

TICKERS = [
    # Tech
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
    "AMD", "NFLX", "INTC", "ORCL", "CRM", "ADBE", "AVGO", "QCOM",
    "TXN", "MU", "IBM", "CSCO", "UBER", "ABNB", "PYPL", "SHOP",
    "COIN", "SNOW", "PLTR", "SQ", "ZM", "DOCU", "ROKU",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "SCHW", "BLK", "AXP",
    "USB", "PNC", "TFC",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB",
    # Healthcare
    "JNJ", "PFE", "LLY", "UNH", "MRK", "ABBV", "TMO", "DHR",
    "ABT", "BMY", "GILD", "AMGN",
    # Industrial
    "HON", "CAT", "DE", "LMT", "RTX", "MMM", "UPS", "FDX",
    "BA", "GE",
    # Consumer
    "WMT", "PG", "KO", "PEP", "MCD", "SBUX", "NKE", "HD", "LOW",
    "TGT", "COST", "CVS", "WBA", "DIS",
]

DAYS_BACK = 365  # was 180

client = finnhub.Client(api_key=FINNHUB_KEY)

end = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")

rows = []
for ticker in TICKERS:
    try:
        news = client.company_news(ticker, _from=start, to=end)
    except Exception as e:
        print(f"{ticker}: {e}")
        continue

    kept = 0
    for art in news:
        headline = art.get("headline", "") or ""
        content = art.get("summary", "") or ""
        if len(content) < 80:
            continue
        rows.append({
            "ticker": ticker,
            "headline": headline,
            "content": content,
            "published": art.get("datetime", ""),
            "source": art.get("source", ""),
            "url": art.get("url", ""),
        })
        kept += 1
    print(f"{ticker}: {kept} articles kept (of {len(news)} returned)")
    time.sleep(1.1)  # 60 req/min limit

if not rows:
    print("No articles retrieved. Check your Finnhub key.")
    raise SystemExit(2)

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["headline", "content"]).reset_index(drop=True)
df.to_csv("eulerpool_corpus.csv", sep=";", index=False)
print(f"Pulled {len(df)} articles → eulerpool_corpus.csv")
print(f"Unique tickers: {df['ticker'].nunique()}")
print(f"Avg content length:  {df['content'].str.len().mean():.0f} chars")
print(f"Avg headline length: {df['headline'].str.len().mean():.0f} chars")
