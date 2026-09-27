"""
Walk-forward validation for the RL trading agent.

Splits the price history into N folds. For each fold k:
    train on bars [0 .. split_k)
    test on bars [split_k .. split_{k+1})

The trading agent is RE-TRAINED from scratch for each fold (from a fresh
network), so this measures generalisation rather than in-sample fit.

Outputs walkforward_results.csv with one row per fold.
"""
import os

import numpy as np
import pandas as pd

from RL.backtest import run_backtest
from RL.rl_agent import DQNAgent
from RL.rl_env import TradingEnv
from market_data import get_market_data, compute_features

TICKER_FILE = "rl_main_ticker.txt"
START = "2023-01-01"  # adjust to your history
END = None
N_FOLDS = 4
EPISODES = 20
STATE_DIM = 7
ACTION_DIM = 3
SEED = 42


def _train_one_fold(train_df, agent_kwargs):
    np.random.seed(SEED)
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, **agent_kwargs)
    env = TradingEnv(train_df, train_df['signal'], initial_cash=10000, shares=100)
    for _ in range(EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            nxt, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, nxt, done)
            agent.replay()
            state = nxt
    return agent


def main():
    # at the top of walkforward.py's main(), replace the open() block with:

    def _get_ticker():
        if os.path.exists(TICKER_FILE):
            with open(TICKER_FILE) as f:
                return f.read().strip()
        # Fallback: most-frequent ticker from the training corpus
        try:
            df = pd.read_csv("training_corpus_large.csv", sep=';')
            # In the absence of a ticker column, pick a sensible default
            return "AAPL"
        except Exception:
            return "AAPL"

    ticker = _get_ticker()
    print(f"Walk-forward on {ticker}, {N_FOLDS} folds, {EPISODES} episodes each")

    df = get_market_data(ticker, START, END)
    df_feat = compute_features(df)
    n = len(df_feat)

    # Expanding windows: first fold trains on 50% of data, last on 80%
    train_fracs = np.linspace(0.5, 0.8, N_FOLDS)

    rows = []
    for i, frac in enumerate(train_fracs, 1):
        split = int(n * frac)
        test_end = min(split + int(n * 0.1), n)  # 10% test window
        train = df_feat.iloc[:split].copy()
        test = df_feat.iloc[split:test_end].copy()
        # Signals: reuse whatever composite signal is already in the frame
        # (for this script, use a neutral signal — the goal is to measure
        # the RL policy's robustness across folds)
        train['signal'] = 0
        test['signal'] = 0

        print(f"Fold {i}: train {len(train)} bars, test {len(test)} bars")
        agent = _train_one_fold(train, {
            'lr': 1e-3, 'gamma': 0.99, 'epsilon_decay': 0.995,
            'batch_size': 32, 'target_update': 10,
        })
        res = run_backtest(agent, test, test['signal'], initial_cash=10000)
        rows.append({
            "fold": i,
            "train_bars": len(train),
            "test_bars": len(test),
            "return_pct": res["total_return"],
            "sharpe": res["sharpe_ratio"] if res["sharpe_ratio"] is not None else 0.0,
            "max_dd": res["max_drawdown"],
        })
        print(f"return={res['total_return']:.2f}%  "
              f"sharpe={res['sharpe_ratio']}  dd={res['max_drawdown']:.2f}%")

    df_out = pd.DataFrame(rows)
    df_out.to_csv("walkforward_results.csv", index=False)

    print("================= Walk-forward summary =================")
    print(df_out.to_string(index=False))
    print("Mean +- std across folds:")
    for col in ("return_pct", "sharpe", "max_dd"):
        m, s = df_out[col].mean(), df_out[col].std()
        print(f"{col:12s}: {m:+.3f}  ±  {s:.3f}")
    print("Saved walkforward_results.csv")


if __name__ == "__main__":
    main()
