import os

import pandas as pd
import requests
import yfinance as yf


class MarketDataProvider:
    """Base class for market data providers."""

    def get_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_quote(self, ticker: str) -> dict:
        raise NotImplementedError


class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance data provider (free, no API key)."""

    def get_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        return yf.download(ticker, start=start_date, end=end_date, progress=False)

    def get_quote(self, ticker: str) -> dict:
        ticker_obj = yf.Ticker(ticker)
        return ticker_obj.info


class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage API provider (requires free API key)."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
        self.base_url = "https://www.alphavantage.co/query"

    def get_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": self.api_key,
            "datatype": "json"
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Alpha Vantage API error: {response.status_code}")

        data = response.json()
        if "Time Series (Daily)" not in data:
            raise Exception(f"Alpha Vantage error: {data.get('Note', 'Unknown error')}")

        # Parse to DataFrame
        ts = data["Time Series (Daily)"]
        df_data = []
        for date, values in ts.items():
            if start_date <= date <= end_date:
                df_data.append({
                    "Date": date,
                    "Open": float(values["1. open"]),
                    "High": float(values["2. high"]),
                    "Low": float(values["3. low"]),
                    "Close": float(values["5. adjusted close"]),
                    "Volume": int(values["6. volume"])
                })

        df = pd.DataFrame(df_data)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df.sort_index()

    def get_quote(self, ticker: str) -> dict:
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Alpha Vantage API error: {response.status_code}")

        data = response.json()
        if "Global Quote" not in data:
            raise Exception(f"Alpha Vantage error: {data.get('Note', 'Unknown error')}")

        quote = data["Global Quote"]
        return {
            "price": float(quote.get("05. price", 0)),
            "change": float(quote.get("09. change", 0)),
            "change_percent": float(quote.get("10. change percent", "0%").replace("%", ""))
        }


class TwelveDataProvider(MarketDataProvider):
    """Twelve Data API provider (free tier)."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("TWELVE_DATA_API_KEY")
        self.base_url = "https://api.twelvedata.com"

    def get_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.api_key:
            raise Exception("Twelve Data API key not configured")

        params = {
            "symbol": ticker,
            "interval": "1day",
            "start_date": start_date,
            "end_date": end_date,
            "apikey": self.api_key,
            "outputsize": 1000
        }
        response = requests.get(f"{self.base_url}/time_series", params=params)
        if response.status_code != 200:
            raise Exception(f"Twelve Data API error: {response.status_code}")

        data = response.json()
        if "values" not in data:
            raise Exception(f"Twelve Data error: {data.get('status', 'Unknown error')}")

        # Parse to DataFrame
        df_data = []
        for entry in data["values"]:
            df_data.append({
                "Date": entry["datetime"],
                "Open": float(entry["open"]),
                "High": float(entry["high"]),
                "Low": float(entry["low"]),
                "Close": float(entry["close"]),
                "Volume": int(entry["volume"])
            })

        df = pd.DataFrame(df_data)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df.sort_index()

    def get_quote(self, ticker: str) -> dict:
        if not self.api_key:
            raise Exception("Twelve Data API key not configured")

        params = {
            "symbol": ticker,
            "apikey": self.api_key
        }
        response = requests.get(f"{self.base_url}/price", params=params)
        if response.status_code != 200:
            raise Exception(f"Twelve Data API error: {response.status_code}")

        data = response.json()
        return {"price": float(data.get("price", 0))}


def get_provider(provider_name: str = "yfinance"):
    """Factory function to get the appropriate provider."""
    providers = {
        "yfinance": YahooFinanceProvider,
        "alphavantage": AlphaVantageProvider,
        "twelvedata": TwelveDataProvider,
    }
    provider_class = providers.get(provider_name.lower())
    if provider_class is None:
        raise ValueError(f"Unknown provider: {provider_name}")
    return provider_class()


def get_market_data_with_provider(ticker: str, start_date: str, end_date: str, provider: str = "yfinance"):
    """Get market data using the specified provider."""
    provider_instance = get_provider(provider)
    try:
        return provider_instance.get_stock_data(ticker, start_date, end_date)
    except Exception as e:
        print(f"Provider {provider} failed: {e}")
        # Fallback to yfinance
        if provider != "yfinance":
            print("Falling back to yfinance...")
            return YahooFinanceProvider().get_stock_data(ticker, start_date, end_date)
        raise
