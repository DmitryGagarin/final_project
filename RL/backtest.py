import backtrader as bt
import numpy as np


class DQNStrategy(bt.Strategy):
    params = (
        ('agent', None),
        ('signal_series', None),
        ('features_df', None),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        # We'll store previous portfolio value for reward? Not needed for backtest.
        self.order = None
        self.agent = self.params.agent
        self.signal_series = self.params.signal_series
        self.features_df = self.params.features_df
        self.idx = 0  # index in features_df

    def next(self):
        if self.idx >= len(self.features_df):
            return
        # Get current state
        row = self.features_df.iloc[self.idx]
        state = [
            row['returns'],
            row['rsi'] / 100.0,
            row['macd'],
            (row['Close'] - row['sma_20']) / row['sma_20'],
            (row['Close'] - row['sma_50']) / row['sma_50'],
            row['volume_ratio'],
            self.signal_series.iloc[self.idx] / 2.0
        ]
        state = np.array(state, dtype=np.float32)
        # Agent acts (no exploration)
        action = self.agent.act(state, training=False)
        # Execute
        if action == 1:  # buy
            if not self.position:
                self.buy(size=100)
        elif action == 2:  # sell
            if self.position:
                self.sell(size=100)
        # else hold
        self.idx += 1


def run_backtest(agent, market_df, signal_series, initial_cash=10000):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1%

    # Add data
    data = bt.feeds.PandasData(dataname=market_df)
    cerebro.adddata(data)

    # Add strategy with agent and signal
    cerebro.addstrategy(DQNStrategy, agent=agent, signal_series=signal_series, features_df=market_df)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print('Final Portfolio Value: %.2f' % final_value)

    # Get metrics
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown.max.drawdown if drawdown else 0

    return {
        'final_value': final_value,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'total_return': (final_value - initial_cash) / initial_cash * 100
    }
