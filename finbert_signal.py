"""
Financial sentiment classifier using ProsusAI/finbert.

Returns a sentiment in {-1, 0, +1} (bearish / neutral / bullish) with a
confidence score, so callers can use it as a numeric feature.
"""
import threading

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_MODEL_NAME = "ProsusAI/finbert"
_tokenizer = None
_model = None
_lock = threading.RLock()


def _load():
    global _tokenizer, _model
    with _lock:
        if _tokenizer is not None and _model is not None:
            return
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        if torch.backends.mps.is_available():
            _model.to("mps")
        else:
            _model.to("cpu")
        _model.eval()


def score(text, max_length=256):
    """
    Return (label, signed_value, confidence).

    label      : 'positive' | 'negative' | 'neutral'
    signed_val : +1 / -1 / 0 (useful as an RL feature)
    confidence : probability of the winning class (0..1)
    """
    if not text or not isinstance(text, str):
        return "neutral", 0, 0.0
    _load()
    tok = _tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    device = next(_model.parameters()).device
    tok = {k: v.to(device) for k, v in tok.items()}
    with torch.no_grad():
        logits = _model(**tok).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    idx = int(probs.argmax())
    label = _model.config.id2label[idx].lower()
    conf = float(probs[idx])
    signed = {"positive": 1, "negative": -1}.get(label, 0)
    return label, signed, conf


def is_available():
    try:
        _load()
        return True
    except Exception:
        return False
