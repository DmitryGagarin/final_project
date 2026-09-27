from datetime import datetime

import yfinance as yf


def get_market_data(ticker, start_date, end_date=None):
    """Fetch OHLCV data for ticker between start_date and end_date."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Newer yfinance returns MultiIndex columns like ('Close', 'C').
    # Flatten to single-level 'Close', 'Open', ...
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    return df


def compute_features(df):
    """
    Add technical indicators to the DataFrame.
    Returns DataFrame with columns: returns, rsi, macd, sma_20, sma_50, volume_ratio.
    """
    df = df.copy()
    # Make sure Close/Volume are 1-D numeric Series
    for col in ("Close", "Volume"):
        if col in df.columns:
            s = df[col]
            if hasattr(s, "squeeze"):
                s = s.squeeze()
            df[col] = s.astype(float)

    df['returns'] = df['Close'].pct_change()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2

    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df = get_market_data("SBER.ME", "2023-01-01")
    df_feat = compute_features(df)
    print(df_feat.tail())
