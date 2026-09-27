"""
Compare the trained RL agent against four baselines on the held-out test set.

Strategies:
  1. RL (DQN)
  2. Buy & Hold
  3. Rule-Based advisor (RSI-filtered)
  4. Signal-Only (follow analyst signal directly)
  5. Random policy

Outputs:
  - rl_comparison.csv     (metrics table)
  - rl_equity_curves.png  (equity curves for the report)

Run from project root:
    python evaluate_rl.py
"""
import os
import random

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advisor import get_advice


def load_test_data():
    if not os.path.exists("rl_test_set.pkl"):
        raise FileNotFoundError(
            "rl_test_set.pkl not found. Re-run RL training (from the UI) "
            "so the split is saved."
        )
    return pd.read_pickle("rl_test_set.pkl")


def load_agent():
    if not os.path.exists("dqn_model.pth"):
        raise FileNotFoundError("dqn_model.pth not found. Train the RL agent first.")
    from RL.rl_agent import DQNAgent
    agent = DQNAgent(state_dim=7, action_dim=3)
    agent.load("dqn_model.pth")
    return agent


# ----------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------
class EquityCurve(bt.Analyzer):
    def start(self):
        self.values = []
        self.dates = []
        self.trades = 0

    def next(self):
        self.values.append(self.strategy.broker.getvalue())
        self.dates.append(self.strategy.datas[0].datetime.date(0))

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades += 1

    def get_analysis(self):
        return {"values": self.values, "dates": self.dates, "n_trades": self.trades}


class DQNStrategy(bt.Strategy):
    params = (("agent", None), ("signal_series", None), ("features_df", None))

    def __init__(self):
        self.agent = self.params.agent
        self.signal_series = self.params.signal_series.reset_index(drop=True)
        self.features_df = self.params.features_df.reset_index(drop=True)
        self.idx = 0

    def _size(self):
        cash = self.broker.getcash()
        price = float(self.data.close[0])
        return max(int(cash * 0.95 / price), 0)

    def next(self):
        if self.idx >= len(self.features_df):
            return
        row = self.features_df.iloc[self.idx]
        state = np.array([
            float(row['returns']),
            float(row['rsi']) / 100.0,
            float(row['macd']),
            (float(row['Close']) - float(row['sma_20'])) / float(row['sma_20']),
            (float(row['Close']) - float(row['sma_50'])) / float(row['sma_50']),
            float(row['volume_ratio']),
            float(self.signal_series.iloc[self.idx]) / 2.0,
        ], dtype=np.float32)
        action = self.agent.act(state, training=False)
        if action == 1 and not self.position:
            sz = self._size()
            if sz > 0:
                self.buy(size=sz)
        elif action == 2 and self.position:
            self.sell(size=self.position.size)
        self.idx += 1


class BuyAndHold(bt.Strategy):
    def __init__(self):
        self.bought = False

    def next(self):
        if not self.bought:
            cash = self.broker.getcash()
            price = float(self.data.close[0])
            sz = int(cash * 0.99 / price)
            if sz > 0:
                self.buy(size=sz)
                self.bought = True


class RuleBased(bt.Strategy):
    params = (("signal_series", None), ("features_df", None))

    def __init__(self):
        self.signal_series = self.params.signal_series.reset_index(drop=True)
        self.features_df = self.params.features_df.reset_index(drop=True)
        self.idx = 0

    def _size(self):
        cash = self.broker.getcash()
        price = float(self.data.close[0])
        return max(int(cash * 0.95 / price), 0)

    def next(self):
        if self.idx >= len(self.features_df):
            return
        row = self.features_df.iloc[self.idx]
        sig = int(self.signal_series.iloc[self.idx])
        act_word = {0: "hold", 1: "buy", 2: "sell"}.get(sig, "hold")
        final_action, _ = get_advice(act_word, "X", float(row['rsi']), float(row['Close']))
        if final_action == "BUY" and not self.position:
            sz = self._size()
            if sz > 0:
                self.buy(size=sz)
        elif final_action == "SELL" and self.position:
            self.sell(size=self.position.size)
        self.idx += 1


class SignalOnly(bt.Strategy):
    params = (("signal_series", None),)

    def __init__(self):
        self.signal_series = self.params.signal_series.reset_index(drop=True)
        self.idx = 0

    def _size(self):
        cash = self.broker.getcash()
        price = float(self.data.close[0])
        return max(int(cash * 0.95 / price), 0)

    def next(self):
        if self.idx >= len(self.signal_series):
            return
        sig = int(self.signal_series.iloc[self.idx])
        if sig == 1 and not self.position:
            sz = self._size()
            if sz > 0:
                self.buy(size=sz)
        elif sig == 2 and self.position:
            self.sell(size=self.position.size)
        self.idx += 1


class RandomPolicy(bt.Strategy):
    params = (("seed", 42),)

    def __init__(self):
        random.seed(self.params.seed)
        np.random.seed(self.params.seed)

    def _size(self):
        cash = self.broker.getcash()
        price = float(self.data.close[0])
        return max(int(cash * 0.95 / price), 0)

    def next(self):
        a = np.random.randint(0, 3)
        if a == 1 and not self.position:
            sz = self._size()
            if sz > 0:
                self.buy(size=sz)
        elif a == 2 and self.position:
            self.sell(size=self.position.size)


# ----------------------------------------------------------------------
# Runner + metrics
# ----------------------------------------------------------------------
def run_one(strategy_cls, market_df, kwargs, initial_cash=10000, commission=0.001):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.adddata(bt.feeds.PandasData(dataname=market_df))
    cerebro.addstrategy(strategy_cls, **kwargs)
    cerebro.addanalyzer(EquityCurve, _name="eq")
    results = cerebro.run()
    return results[0].analyzers.eq.get_analysis()


def compute_metrics(eq):
    values = np.array(eq["values"], dtype=float)
    if len(values) < 2:
        return {}
    returns = np.diff(values) / values[:-1]
    n = len(values)
    ann = 252

    total_ret = (values[-1] / values[0] - 1) * 100
    ann_ret = ((values[-1] / values[0]) ** (ann / n) - 1) * 100
    ann_vol = np.std(returns, ddof=1) * np.sqrt(ann) * 100
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = returns[returns < 0]
    down_vol = np.std(downside, ddof=1) * np.sqrt(ann) * 100 if len(downside) > 1 else 0.0
    sortino = ann_ret / down_vol if down_vol > 0 else 0.0

    running_max = np.maximum.accumulate(values)
    dd = (values - running_max) / running_max
    max_dd = dd.min() * 100
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    wins = int(np.sum(returns > 0))
    win_rate = wins / len(returns) * 100 if len(returns) else 0.0

    return {
        "Total Return %": round(total_ret, 2),
        "Ann. Return %": round(ann_ret, 2),
        "Ann. Vol %": round(ann_vol, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Max DD %": round(max_dd, 2),
        "Calmar": round(calmar, 3),
        "Win Rate %": round(win_rate, 2),
        "N Trades": eq["n_trades"],
        "Final Value": round(values[-1], 2),
    }


def main():
    print("Loading test set...")
    test_df = load_test_data()
    print(f"Test set: {len(test_df)} bars "
          f"({test_df.index[0].date()} -> {test_df.index[-1].date()})")

    print("Loading RL agent...")
    agent = load_agent()

    market_df = test_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    signal = test_df["signal"].copy()

    strategies = {
        "RL (DQN)": (DQNStrategy, {"agent": agent, "signal_series": signal, "features_df": test_df}),
        "Buy & Hold": (BuyAndHold, {}),
        "Rule-Based": (RuleBased, {"signal_series": signal, "features_df": test_df}),
        "Signal-Only": (SignalOnly, {"signal_series": signal}),
        "Random": (RandomPolicy, {}),
    }

    rows, curves = [], {}
    for name, (cls, kw) in strategies.items():
        print(f"Running {name}...")
        eq = run_one(cls, market_df, kw)
        m = compute_metrics(eq)
        m["Strategy"] = name
        rows.append(m)
        curves[name] = (eq["dates"], eq["values"])

    df = pd.DataFrame(rows).set_index("Strategy")
    df = df[["Total Return %", "Ann. Return %", "Ann. Vol %", "Sharpe",
             "Sortino", "Max DD %", "Calmar", "Win Rate %", "N Trades", "Final Value"]]
    print("================== Comparison ==================")
    print(df.to_string())
    df.to_csv("rl_comparison.csv")
    print("Saved rl_comparison.csv")

    plt.figure(figsize=(12, 6))
    for name, (dates, values) in curves.items():
        plt.plot(dates, values, label=name, linewidth=1.8)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Strategy Equity Curves on Held-Out Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rl_equity_curves.png", dpi=150)
    print("Saved rl_equity_curves.png")


if __name__ == "__main__":
    main()
