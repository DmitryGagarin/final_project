"""
Natural-language explanation generator using a local Ollama model.

Turns structured advisor output (summary, ticker, action, RSI, RL Q-values)
into a short prose explanation for a non-technical user. This is the
"tool-using agent" layer described in the project brief.

If Ollama is not running on localhost:11434, generate_explanation()
falls back to a deterministic template so the UI never breaks.

Environment variables:
    OLLAMA_URL   default http://localhost:11434
    OLLAMA_MODEL default llama3.2:1b   (small enough for an 8GB Mac)
"""
import json
import os
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

_ollama_available = None  # cached after first probe


def _check_ollama(timeout=2):
    """Probe once and cache the result."""
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:
            _ollama_available = (r.status == 200)
    except Exception:
        _ollama_available = False
    return _ollama_available


def is_available():
    """Public check used by the UI."""
    return _check_ollama()


def list_models():
    """Return the list of locally available Ollama models."""
    if not _check_ollama():
        return []
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _template_explanation(summary, ticker, analyst_action, rsi, price, final_action):
    """Deterministic fallback so the UI always has something to show."""
    direction = {"BUY": "a bullish", "SELL": "a bearish", "HOLD": "a neutral"}
    rsi_note = ""
    if rsi >= 70:
        rsi_note = f"RSI at {rsi:.1f} flags overbought conditions, "
    elif rsi <= 30:
        rsi_note = f"RSI at {rsi:.1f} flags oversold conditions, "
    else:
        rsi_note = f"RSI at {rsi:.1f} is neutral, "

    return (
        f"{ticker} shows {direction.get(final_action, 'a neutral')} setup. "
        f"{rsi_note}with the analyst signal reading '{analyst_action}'. "
        f"At ${price:.2f}, the recommended action is {final_action}. "
        f"Summary context: {summary[:180]}..."
    )


def generate_explanation(summary, ticker, analyst_action, rsi, price,
                         final_action, q_str=None, model=None, timeout=60):
    """
    Generate a natural-language explanation.

    Returns a string. Uses Ollama if available, template fallback otherwise.
    """
    if not _check_ollama():
        return _template_explanation(summary, ticker, analyst_action, rsi, price, final_action)

    model = model or OLLAMA_MODEL

    q_line = f"Reinforcement-learning Q-values: {q_str}." if q_str else ""
    prompt = (
        "You are a financial advisor bot explaining a recommendation to a "
        "non-technical user. Write 2-3 sentences. Be concrete, no hedging, "
        "no disclaimers, no markdown. Do not invent numbers or comparisons.\n\n"
        f"Ticker: {ticker}\n"
        f"News summary: {summary}\n"
        f"Analyst signal: {analyst_action}\n"
        f"RSI reading: {'overbought' if rsi >= 70 else 'oversold' if rsi <= 30 else 'neutral'}\n"
        f"Final action: {final_action}\n"
        f"{q_line}\n"
        "Explanation:"
    )

    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 120},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        text = (data.get("response") or "").strip()
        if text:
            return text
        return _template_explanation(summary, ticker, analyst_action, rsi, price, final_action)
    except Exception as e:
        return _template_explanation(summary, ticker, analyst_action, rsi, price, final_action) + \
            f"\n\n[LLM unavailable: {type(e).__name__}]"
