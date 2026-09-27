def get_advice(summary_action, ticker, rsi, price):
    """
    Returns (action, explanation) based on summary_action and RSI.
    """
    if summary_action == 'buy':
        if rsi < 70:
            action = 'BUY'
            reason = f"bullish signal and RSI ({rsi:.1f}) is not overbought"
        else:
            action = 'HOLD'
            reason = f"bullish signal but RSI ({rsi:.1f}) suggests overbought conditions"
    elif summary_action == 'sell':
        if rsi > 30:
            action = 'SELL'
            reason = f"bearish signal and RSI ({rsi:.1f}) is not oversold"
        else:
            action = 'HOLD'
            reason = f"bearish signal but RSI ({rsi:.1f}) suggests oversold conditions"
    else:
        action = 'HOLD'
        reason = "no clear signal from the analyst"

    explanation = f"We recommend {action} for {ticker} because {reason}. Current price: {price:.2f}"
    return action, explanation
