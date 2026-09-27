import os
import re
import threading

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils.utils import clean_text

# ======================================================================
# Signal extraction
# ======================================================================
# Company names → ticker. Extend as needed.
_COMPANY_TO_TICKER = {
    "apple": "AAPL", "iphone": "AAPL", "ipad": "AAPL", "macbook": "AAPL",
    "microsoft": "MSFT", "windows": "MSFT", "azure": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "android": "GOOGL",
    "amazon": "AMZN", "aws": "AMZN",
    "nvidia": "NVDA", "geforce": "NVDA",
    "tesla": "TSLA", "musk": "TSLA",
    "meta": "META", "facebook": "META", "instagram": "META", "whatsapp": "META",
    "netflix": "NFLX",
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "eth": "ETH",
    "solana": "SOL", "ripple": "XRP", "xrp": "XRP",
    "citigroup": "C", "citibank": "C",
    "visa": "V", "mastercard": "MA", "paypal": "PYPL",
    "ford": "F", "general motors": "GM",
    "at&t": "T", "verizon": "VZ",
    "boeing": "BA", "lockheed": "LMT",
    "exxon": "XOM", "chevron": "CVX",
    "pfizer": "PFE", "moderna": "MRNA", "johnson": "JNJ",
    "walmart": "WMT", "target": "TGT", "costco": "COST",
    "disney": "DIS", "nike": "NKE", "starbucks": "SBUX",
}

_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
    "ADBE", "CRM", "ORCL", "IBM", "INTC", "CSCO", "AMD", "QCOM",
    "TXN", "AVGO", "MU", "NXP",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "PYPL",
    "AXP", "COF", "PNC", "USB", "TFC",
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "AMGN", "MDT", "ISRG",
    "TMO", "DHR", "ABT", "LLY", "BMY", "GILD",
    "WMT", "PG", "KO", "PEP", "DIS", "NKE", "MCD", "SBUX",
    "HD", "LOW", "TGT", "COST", "CVS", "WBA",
    "GE", "BA", "CAT", "DE", "HON", "UPS", "FDX", "MMM",
    "XOM", "CVX", "COP", "EOG", "SLB",
    "BTC", "ETH", "SOL", "ADA", "DOT", "AVAX",
]

# Only tickers with length >= 2 are safe for substring matching
_SAFE_TICKERS = [t for t in _TICKERS if len(t) >= 2]
_SAFE_TICKER_SET = set(_SAFE_TICKERS)

_BUY_WORDS = [
    "buy", "long", "bullish", "upside", "breakout",
    "accumulate", "buying", "rally", "outperform", "overweight",
    "покупать", "покупка", "купить", "лонг",
]
_SELL_WORDS = [
    "sell", "short", "bearish", "downside", "breakdown",
    "distribute", "selling", "drop", "underperform", "underweight",
    "продавать", "продажа", "продать", "шорт",
]


def extract_signal(summary):
    """
    Extract ticker and action (buy/sell/hold) from a summary.

    Priority:
      1. Explicit $TICKER or #TICKER notation (most reliable)
      2. Word-boundary match against a whitelist of >= 2-char tickers
      3. Company-name -> ticker dictionary
    """
    text = summary or ""
    upper = text.upper()
    lower = text.lower()

    found_ticker = None

    # 1. $TICKER / #TICKER notation
    m = re.search(r"[$#]([A-Z]{2,5})\b", upper)
    if m:
        candidate = m.group(1)
        if candidate in _SAFE_TICKER_SET or (2 <= len(candidate) <= 5):
            found_ticker = candidate

    # 2. Word-boundary match on whitelist
    if found_ticker is None:
        for t in _SAFE_TICKERS:
            if re.search(rf"\b{re.escape(t)}\b", upper):
                found_ticker = t
                break

    # 3. Company-name fallback
    if found_ticker is None:
        for name, ticker in _COMPANY_TO_TICKER.items():
            if re.search(rf"\b{re.escape(name)}\b", lower):
                found_ticker = ticker
                break

    # 4. Action from keyword lists
    action = "hold"
    if any(re.search(rf"\b{re.escape(w)}\b", lower) for w in _BUY_WORDS):
        action = "buy"
    elif any(re.search(rf"\b{re.escape(w)}\b", lower) for w in _SELL_WORDS):
        action = "sell"

    return found_ticker, action


# ======================================================================
# Summariser model management
# ======================================================================
def get_default_model_path():
    """Find the most recent merged model in ./models/ directory."""
    models_dir = "./models"
    if os.path.exists(models_dir):
        dirs = [os.path.join(models_dir, d) for d in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, d))]
        if dirs:
            dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            for d in dirs:
                merged = os.path.join(d, "merged")
                if os.path.exists(merged):
                    return merged
    fallback = "./models/english_lora_20260908_010129/merged"
    if os.path.exists(fallback):
        return fallback
    return None


_MODEL_PATH = get_default_model_path()
if _MODEL_PATH is None:
    raise RuntimeError(
        "No trained summarisation model found. Please train a model first."
    )

_tokenizer = None
_model = None
_summarizer_lock = threading.RLock()


def set_model_path(path):
    """Idempotent model path setter - no-op if same path is already loaded."""
    global _MODEL_PATH, _tokenizer, _model
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path does not exist: {path}")

    with _summarizer_lock:
        if _MODEL_PATH == path and _tokenizer is not None and _model is not None:
            return
        print(f"[summarizer] Setting model path to: {path}")
        _MODEL_PATH = path
        _tokenizer = None
        _model = None


def load_summarizer():
    """Load tokenizer and model from _MODEL_PATH. Thread-safe."""
    global _tokenizer, _model
    with _summarizer_lock:
        if _tokenizer is not None and _model is not None:
            return
        print(f"[summarizer] Loading summarizer from {_MODEL_PATH}")
        try:
            tok = AutoTokenizer.from_pretrained(_MODEL_PATH)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_PATH)
            if torch.backends.mps.is_available():
                mdl.to("mps")
            else:
                mdl.to("cpu")
            mdl.eval()
            _tokenizer = tok
            _model = mdl
            print("[summarizer] Load successful.")
        except Exception as e:
            _tokenizer = None
            _model = None
            raise RuntimeError(
                f"Failed to load summarisation model from {_MODEL_PATH}: {e}"
            )


def summarize_post(post_text):
    """Summarise a post using the currently loaded model."""
    load_summarizer()

    tokenizer = _tokenizer
    model = _model
    if tokenizer is None or model is None:
        raise RuntimeError("Summariser not properly initialised.")

    cleaned = clean_text(post_text)
    input_text = "summarize: " + cleaned
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=384)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.5,
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def get_summary_for_model(post_text, model_path):
    """Load a specific model and summarise a post (isolated from globals)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    if torch.backends.mps.is_available():
        model.to("mps")
    else:
        model.to("cpu")
    model.eval()

    cleaned = clean_text(post_text)
    input_text = "summarize: " + cleaned
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=384)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.5,
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
