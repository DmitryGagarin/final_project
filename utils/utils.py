import json
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s,.!?$%€£@#\-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_training_ids(channel):
    ids = set()
    csv_path = Path('../telegram_legacy/csvs/english_channels_sample_30.csv')
    if csv_path.exists():
        df = pd.read_csv(csv_path, sep=';')
        if 'id' in df.columns:
            ids.update(df['id'].dropna().astype(int).tolist())
    return ids


def get_unseen_posts(channel, limit=3):
    conn = sqlite3.connect('../telegram_legacy/telegram_posts.db')
    cursor = conn.cursor()
    training_ids = get_training_ids(channel)
    if training_ids:
        placeholders = ','.join(['?'] * len(training_ids))
        query = (f"SELECT date, text FROM posts WHERE channel=? "
                 f"AND id NOT IN ({placeholders}) ORDER BY date DESC LIMIT ?")
        params = (channel,) + tuple(training_ids) + (limit,)
    else:
        query = "SELECT date, text FROM posts WHERE channel=? ORDER BY date DESC LIMIT ?"
        params = (channel, limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_available_models(models_dir="./models"):
    """Return list of merged model directories (paths only). Legacy helper."""
    available = []
    if os.path.exists(models_dir):
        for d in os.listdir(models_dir):
            merged_path = os.path.join(models_dir, d, "merged")
            if os.path.exists(merged_path):
                available.append(merged_path)
    default_model = "./rut5_financial_merged_v3"
    if os.path.exists(default_model):
        available.append(default_model)
    return available


def get_named_models(models_dir="./models"):
    """
    Return list of dicts:
        [{"name": ..., "path": ..., "registered": bool, "meta": {...}}, ...]

    Registered models come first (newest first), then unregistered merged
    directories (whose "name" falls back to the directory basename).
    """
    try:
        import model_registry
        registry = model_registry.list_models()
    except Exception:
        registry = {}

    out = []
    seen_paths = set()

    # Registered models first, newest first
    for name, meta in sorted(registry.items(),
                             key=lambda kv: kv[1].get("created", ""),
                             reverse=True):
        path = meta.get("path", "")
        if not os.path.exists(path):
            continue
        out.append({
            "name": name,
            "path": path,
            "registered": True,
            "meta": meta,
        })
        seen_paths.add(os.path.abspath(path))

    # Then any unregistered merged dirs on disk
    if os.path.exists(models_dir):
        for d in sorted(os.listdir(models_dir), reverse=True):
            merged = os.path.join(models_dir, d, "merged")
            if not os.path.exists(merged):
                continue
            if os.path.abspath(merged) in seen_paths:
                continue
            out.append({
                "name": d,  # e.g. english_lora_20260913_205244
                "path": merged,
                "registered": False,
                "meta": {},
            })

    return out


def get_model_metrics(model_path):
    """Load metrics.json from the model's parent directory."""
    parent_dir = os.path.dirname(model_path)
    metrics_file = os.path.join(parent_dir, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None
