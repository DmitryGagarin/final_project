"""
RSS-feed corpus builder.

Pulls headlines + body text from free financial RSS feeds (no API key
required). Writes `rss_corpus.csv` with the same schema as
`eulerpool_corpus.csv` so it can be merged into the training pipeline.

Each feed has a hard per-request timeout (default 8s). Feeds that fail or
time out are skipped and reported, so the script never hangs.

Run:
    python build_rss_corpus.py
    python build_rss_corpus.py --timeout 5     # override per-feed timeout
"""
import argparse
import re
import time

import feedparser
import pandas as pd
import requests

GLOBAL_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "https://www.investing.com/rss/news.rss",
    "https://www.investing.com/rss/news_25.rss",
    "https://seekingalpha.com/market_currents.xml",
    "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
    "https://www.nasdaq.com/feed/rssoutbound?category=Stocks",
]

YAHOO_TICKERS = [
    # Tech
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
    "AMD", "NFLX", "INTC", "ORCL", "CRM", "ADBE", "AVGO", "QCOM",
    "TXN", "MU", "IBM", "CSCO", "UBER", "ABNB", "PYPL", "SHOP",
    "COIN",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "SCHW", "BLK", "AXP",
    # Energy / industrial
    "XOM", "CVX", "COP", "HON", "CAT", "DE", "LMT", "RTX", "BA", "GE",
    # Healthcare
    "JNJ", "PFE", "LLY", "UNH", "MRK", "ABBV", "TMO",
    # Consumer
    "WMT", "DIS", "PG", "KO", "PEP", "MCD", "SBUX", "NKE", "HD", "COST",
]

MIN_CONTENT_CHARS = 80
MIN_HEADLINE_CHARS = 20

# User-Agent helps with feeds that block default Python clients
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

DEFAULT_TIMEOUT = 8  # seconds per feed


def _clean_html(s):
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_entries(feed_url, ticker="", timeout=DEFAULT_TIMEOUT):
    """
    Fetch feed with a hard timeout, then parse. Returns a list of row dicts.
    Never raises: any error is reported and treated as an empty result.
    """
    out = []
    try:
        # Two-stage timeout: (connect_timeout, read_timeout)
        resp = requests.get(
            feed_url,
            headers=HEADERS,
            timeout=(timeout, timeout),
            allow_redirects=True,
        )
    except requests.exceptions.Timeout:
        print(f"timeout after {timeout}s")
        return out
    except requests.exceptions.RequestException as e:
        print(f"request failed: {type(e).__name__}: {e}")
        return out

    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}")
        return out

    try:
        feed = feedparser.parse(resp.content)
    except Exception as e:
        print(f"parse failed: {type(e).__name__}: {e}")
        return out

    if feed.bozo and not feed.entries:
        print(f"malformed feed: {getattr(feed, 'bozo_exception', '?')}")
        return out

    for entry in feed.entries:
        headline = _clean_html(entry.get("title", ""))
        content = _clean_html(
            entry.get("summary", "") or entry.get("description", "")
        )
        if len(headline) < MIN_HEADLINE_CHARS:
            continue
        if len(content) < MIN_CONTENT_CHARS:
            continue
        out.append({
            "ticker": ticker,
            "headline": headline,
            "content": content,
            "published": entry.get("published", "") or entry.get("updated", ""),
            "source": "rss:" + feed_url.split("/")[2],
            "url": entry.get("link", ""),
        })
    return out


def _host(url):
    try:
        return url.split("/")[2]
    except IndexError:
        return url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"Per-feed timeout in seconds (default {DEFAULT_TIMEOUT})")
    ap.add_argument("--out", default="rss_corpus.csv",
                    help="Output CSV path (default rss_corpus.csv)")
    args = ap.parse_args()

    rows = []

    print(f"Global feeds ({len(GLOBAL_FEEDS)}) — timeout {args.timeout}s each:")
    for url in GLOBAL_FEEDS:
        print(f"→ {_host(url)} ", end="", flush=True)
        ents = _extract_entries(url, ticker="", timeout=args.timeout)
        print(f"({len(ents)} entries)")
        rows.extend(ents)
        time.sleep(0.3)

    print(f"Per-ticker Yahoo Finance feeds ({len(YAHOO_TICKERS)}):")
    total = 0
    for t in YAHOO_TICKERS:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
        ents = _extract_entries(url, ticker=t, timeout=args.timeout)
        total += len(ents)
        if ents:
            print(f"{t}: +{len(ents)}")
        else:
            print(f"·  {t}: 0")
        rows.extend(ents)
        time.sleep(0.2)
    print(f"└─ per-ticker total: {total}")

    if not rows:
        print("No entries collected. Check your internet connection "
              "or try a larger --timeout.")
        raise SystemExit(2)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["headline", "content"]).reset_index(drop=True)
    df.to_csv(args.out, sep=";", index=False)

    print(f"Wrote {len(df)} RSS entries → {args.out}")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    print(f"Avg content len:  {df['content'].str.len().mean():.0f} chars")
    print(f"Avg headline len: {df['headline'].str.len().mean():.0f} chars")


if __name__ == "__main__":
    main()
