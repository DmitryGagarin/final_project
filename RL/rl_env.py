import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _to_float(x):
    """Convert numpy scalar / single-element Series / list to a Python float."""
    if hasattr(x, "item"):
        return float(x.item())
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    return float(x)


class TradingEnv(gym.Env):
    """
    Single-asset trading environment.
    State: [returns, rsi, macd, sma_20_diff, sma_50_diff, volume_ratio, signal_encoded]
    signal_encoded: 0=hold, 1=buy, 2=sell (from analyst)
    """

    def __init__(self, df_features, signal_series, initial_cash=10000, shares=100):
        super(TradingEnv, self).__init__()
        self.df = df_features.copy()
        self.signal_series = signal_series
        self.initial_cash = initial_cash
        self.shares = shares
        self.current_step = 0
        self.cash = initial_cash
        self.position = 0
        self.previous_price = None
        self.prev_portfolio_value = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0
        self.previous_price = None
        self.prev_portfolio_value = None
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]

        close = _to_float(row['Close'])
        sma20 = _to_float(row['sma_20'])
        sma50 = _to_float(row['sma_50'])
        rsi = _to_float(row['rsi'])
        sig = _to_float(self.signal_series.iloc[self.current_step])

        features = [
            _to_float(row['returns']),
            rsi / 100.0,
            _to_float(row['macd']),
            (close - sma20) / sma20 if sma20 != 0 else 0.0,
            (close - sma50) / sma50 if sma50 != 0 else 0.0,
            _to_float(row['volume_ratio']),
            sig / 2.0,
        ]
        return np.array(features, dtype=np.float32)

    def step(self, action):
        price = _to_float(self.df.iloc[self.current_step]['Close'])

        # Buy/sell fixed number of shares (position >= 0, no shorting)
        if action == 1:  # buy
            cost = self.shares * price
            if self.cash >= cost:
                self.position += self.shares
                self.cash -= cost
        elif action == 2:  # sell
            if self.position >= self.shares:
                self.position -= self.shares
                self.cash += self.shares * price

        portfolio_value = self.cash + self.position * price

        if self.prev_portfolio_value is None:
            self.prev_portfolio_value = portfolio_value

        if self.prev_portfolio_value != 0:
            reward = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        else:
            reward = 0.0
        self.prev_portfolio_value = portfolio_value

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        self.previous_price = price

        return self._get_obs(), reward, done, False, {}

    def render(self, mode='human'):
        pass
