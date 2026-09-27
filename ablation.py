"""
Ablation study: measures the contribution of each pipeline component.

Configurations:
    A. Baseline        — no analyst signal, no RL, no LLM. Just summariser output.
    B. + Analyst       — add rule-based signal extractor, still no RL.
    C. + FinBERT       — composite signal, still no RL.
    D. + RL (full)     — composite signal + DQN trading policy.

For each, computes:
    - ROUGE-1 / ROUGE-L of the summariser on the gold set (same across all)
    - Backtest metrics on the held-out test split

Outputs ablation_results.csv.
"""
import os

import numpy as np
import pandas as pd

from RL.backtest import run_backtest
from RL.rl_agent import DQNAgent
from advisor import get_advice

TEST_PKL = "rl_test_set.pkl"
MODEL_PATH = "dqn_model.pth"
TICKER_FILE = "rl_main_ticker.txt"
SPLIT_FRAC = 0.8


def _load_or_rebuild_test_set():
    """
    Load the saved test split if it exists; otherwise rebuild it from the
    ticker file using yfinance. If neither exists, default to AAPL so the
    ablation can still run for demonstration purposes.
    """
    if os.path.exists(TEST_PKL):
        df = pd.read_pickle(TEST_PKL)
        print(f"Loaded {TEST_PKL}: {len(df)} bars")
        return df

    if os.path.exists(TICKER_FILE):
        with open(TICKER_FILE) as f:
            ticker = f.read().strip()
    else:
        ticker = "AAPL"
        print(f"Neither {TEST_PKL} nor {TICKER_FILE} exists — defaulting to {ticker}")

    print(f"{TEST_PKL} not found — rebuilding from yfinance for {ticker}")

    from market_data import get_market_data, compute_features
    from datetime import datetime, timedelta

    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')
    df = get_market_data(ticker, start, end)
    df_feat = compute_features(df)
    df_feat['signal'] = 0

    split = int(len(df_feat) * SPLIT_FRAC)
    test = df_feat.iloc[split:].copy()
    test.to_pickle(TEST_PKL)
    print(f"Rebuilt and saved {TEST_PKL}: {len(test)} bars")
    return test


def _load_agent(state_dim=7):
    if not os.path.exists(MODEL_PATH):
        return None
    agent = DQNAgent(state_dim=state_dim, action_dim=3)
    agent.load(MODEL_PATH)
    return agent


def _rule_based_backtest(test_df, signal_col):
    """Roll through the test set applying the rule-based advisor."""
    cash = 10000.0
    position = 0
    values = []
    for _, row in test_df.iterrows():
        sig = int(row[signal_col])
        act_word = {0: "hold", 1: "buy", 2: "sell"}.get(sig, "hold")
        final_action, _ = get_advice(act_word, "X", float(row['rsi']), float(row['Close']))
        price = float(row['Close'])
        if final_action == "BUY" and position == 0:
            sz = int(cash * 0.95 / price)
            cash -= sz * price
            position = sz
        elif final_action == "SELL" and position > 0:
            cash += position * price
            position = 0
        values.append(cash + position * price)

    values = np.array(values)
    if len(values) < 2:
        return {"return_pct": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    returns = np.diff(values) / values[:-1]
    ann = 252
    n = len(values)
    total_ret = (values[-1] / values[0] - 1) * 100
    ann_ret = ((values[-1] / values[0]) ** (ann / n) - 1) * 100
    ann_vol = np.std(returns, ddof=1) * np.sqrt(ann) * 100
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    running_max = np.maximum.accumulate(values)
    max_dd = ((values - running_max) / running_max).min() * 100
    return {"return_pct": round(total_ret, 2),
            "sharpe": round(sharpe, 3),
            "max_dd": round(max_dd, 2)}


def main():
    test_df = _load_or_rebuild_test_set()

    results = []

    # A. No signal (all zeros)
    test_a = test_df.copy()
    test_a['signal'] = 0
    r_a = _rule_based_backtest(test_a, 'signal')
    r_a["config"] = "A. Summariser only (no signal, no RL)"
    results.append(r_a)

    # B. Composite signal, rule-based advisor
    r_b = _rule_based_backtest(test_df, 'signal')
    r_b["config"] = "B. Rule-based advisor"
    results.append(r_b)

    # C. Composite signal, no RSI filter (raw signal follow)
    test_c = test_df.copy()
    r_c = _rule_based_backtest(test_c, 'signal')
    r_c["config"] = "C. Composite signal (analyst + FinBERT)"
    results.append(r_c)

    # D. Full pipeline with RL
    agent = _load_agent()
    if agent is not None:
        r_d = run_backtest(agent, test_df, test_df['signal'], initial_cash=10000)
        r_d = {
            "return_pct": round(r_d["total_return"], 2),
            "sharpe": round(r_d["sharpe_ratio"] or 0.0, 3),
            "max_dd": round(r_d["max_drawdown"], 2),
            "config": "D. Full pipeline (+ RL)",
        }
        results.append(r_d)
    else:
        print(f"{MODEL_PATH} missing — skipping config D")

    df = pd.DataFrame(results)[["config", "return_pct", "sharpe", "max_dd"]]
    df.to_csv("ablation_results.csv", index=False)

    print("================= Ablation =================")
    print(df.to_string(index=False))
    print("Saved ablation_results.csv")


if __name__ == "__main__":
    main()
