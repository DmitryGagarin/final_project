"""
Train the DQN trading agent using corpus-based signals.

Sources (both must exist; missing ones are skipped):
    eulerpool_corpus.csv   -- Finnhub news, 80 tickers x 365 days
    rss_corpus.csv         -- RSS feed headlines + bodies

Signal generation:
    action  <- extract_signal(headline)         (buy / sell / hold)
    ticker  <- CSV 'ticker' column, fallback to extract_signal
    signal  <- composite(analyst_signed, finbert_signed) in {-1, 0, +1}

NOTE: signals are keyed by (date, ticker), not by date alone. Using just
the date caused later articles on the same day to overwrite earlier ones,
which collapsed the effective signal set to a handful of days.
"""
import json
import os
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from RL.backtest import run_backtest
from RL.rl_agent import DQNAgent
from RL.rl_env import TradingEnv
from market_data import get_market_data, compute_features
from summarizer import extract_signal

try:
    import finbert_signal

    _FINBERT_OK = True
except Exception as e:
    print(f"[finbert] not available: {e}")
    _FINBERT_OK = False

CONFIG = {
    'corpus_paths': ['eulerpool_corpus.csv', 'rss_corpus.csv'],
    'start_date': (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d'),
    'end_date': datetime.now().strftime('%Y-%m-%d'),
    'train_split': 0.8,
    'episodes': 30,
    'batch_size': 32,
    'lr': 1e-3,
    'gamma': 0.99,
    'epsilon_decay': 0.995,
    'target_update': 10,
}


def _analyst_to_signed(action):
    return {'buy': 1, 'sell': -1}.get(action.lower(), 0)


def _parse_published(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = int(value)
        if ts > 10_000_000_000:
            ts = ts / 1000
        return datetime.fromtimestamp(ts).date()
    except (ValueError, TypeError):
        pass
    try:
        return pd.to_datetime(value, utc=True).date()
    except Exception:
        return None


def _pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_all_corpora():
    frames = []
    for path in CONFIG['corpus_paths']:
        if not os.path.exists(path):
            print(f"{path} not found, skipping")
            continue
        df = pd.read_csv(path, sep=';')
        if df.empty:
            continue
        print(f"{path}: {len(df)} rows")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def train_rl():
    print("=" * 60)
    print("Training RL Agent (corpus-based signals)")
    print("=" * 60)

    print("Loading corpora...")
    df = _load_all_corpora()
    if df.empty:
        print("No corpus data. Run build_corpus.py and/or build_rss_corpus.py.")
        return
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    headline_col = _pick_column(df, ['headline', 'title', 'summary'])
    ticker_col = _pick_column(df, ['ticker', 'symbol'])
    date_col = _pick_column(df, ['published', 'datetime', 'date'])
    body_col = _pick_column(df, ['content', 'body', 'text', 'cleaned_text'])

    if headline_col is None or date_col is None:
        print(f"Missing headline or date column. Have: {list(df.columns)}")
        return

    print(f"headline='{headline_col}', ticker='{ticker_col}', "
          f"date='{date_col}', body='{body_col}'")

    if _FINBERT_OK:
        print("FinBERT sentiment enabled — building composite signal.")

    # --- Generate signals keyed by (date, ticker) ---
    signals = {}  # (date, ticker) -> action_val
    n_from_meta, n_from_signal = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating signals"):
        date = _parse_published(row.get(date_col))
        if date is None:
            continue

        headline = str(row.get(headline_col, "") or "").strip()
        if not headline:
            continue

        _, action = extract_signal(headline)

        ticker = None
        if ticker_col is not None:
            ticker = str(row.get(ticker_col, "") or "").strip().upper() or None

        if ticker is not None:
            n_from_meta += 1
        else:
            ticker, action_from_headline = extract_signal(headline)
            if ticker is None and body_col is not None:
                body_text = str(row.get(body_col, "") or "")
                ticker, action_from_body = extract_signal(body_text)
                if action == "hold":
                    action = action_from_body
            if ticker is not None:
                n_from_signal += 1

        if ticker is None:
            continue

        analyst_signed = _analyst_to_signed(action)

        finbert_signed = 0
        if _FINBERT_OK:
            try:
                _, finbert_signed, _ = finbert_signal.score(headline)
            except Exception:
                finbert_signed = 0

        composite = int(round((analyst_signed + finbert_signed) / 2))
        composite = max(-1, min(1, composite))
        action_val = {-1: 2, 0: 0, 1: 1}[composite]

        # KEYED BY (date, ticker) - multiple tickers per day are preserved
        signals[(date, ticker)] = action_val

    n_dates = len(set(d for d, _ in signals.keys()))
    n_pairs = len(signals)
    print(f"Generated {n_pairs} (date, ticker) signal pairs across {n_dates} "
          f"dates ({n_from_meta} from CSV ticker, {n_from_signal} from NLP).")

    if not signals:
        print("No signals produced. Aborting.")
        return

    ticker_counts = {}
    for (_, ticker), _ in signals.items():
        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
    print(f"Top tickers: {sorted(ticker_counts.items(), key=lambda x: -x[1])[:8]}")
    main_ticker = max(ticker_counts, key=ticker_counts.get)
    print(f"Most frequent ticker: {main_ticker}")

    # --- Market data ---
    df_market = get_market_data(main_ticker, CONFIG['start_date'], CONFIG['end_date'])
    if df_market is None or df_market.empty:
        print(f"No market data for {main_ticker}. Aborting.")
        return
    df_feat = compute_features(df_market)

    # Align only main_ticker signals to trading days
    signal_series = []
    for date in df_feat.index:
        d = date.date()
        signal_series.append(signals.get((d, main_ticker), 0))
    df_feat['signal'] = signal_series

    n_signal = sum(1 for s in signal_series if s != 0)
    n_hold = len(signal_series) - n_signal
    print(f"Market data: {df_feat.shape}  "
          f"({n_signal} signal-days, {n_hold} hold-days) for {main_ticker}")

    # --- Split ---
    split_idx = int(len(df_feat) * CONFIG['train_split'])
    train_df = df_feat.iloc[:split_idx].copy()
    test_df = df_feat.iloc[split_idx:].copy()
    train_df.to_pickle('rl_train_set.pkl')
    test_df.to_pickle('rl_test_set.pkl')
    with open('rl_main_ticker.txt', 'w') as f:
        f.write(main_ticker)
    print(f"Saved splits  ({len(train_df)}/{len(test_df)} bars)")

    # --- Train ---
    agent = DQNAgent(
        state_dim=7, action_dim=3,
        lr=CONFIG['lr'], gamma=CONFIG['gamma'],
        epsilon_decay=CONFIG['epsilon_decay'],
        batch_size=CONFIG['batch_size'],
        target_update=CONFIG['target_update'],
    )

    env = TradingEnv(train_df, train_df['signal'], initial_cash=10000, shares=100)
    print(f"\nTraining for {CONFIG['episodes']} episodes...")
    for ep in range(CONFIG['episodes']):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Episode {ep + 1}: reward={total_reward:.4f}")

    agent.save('dqn_model.pth')
    print("\nAgent saved to dqn_model.pth")

    print("Backtesting...")
    results = run_backtest(agent, test_df, test_df['signal'], initial_cash=10000)
    print(f"Results: {results}")
    with open('rl_training_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    train_rl()
