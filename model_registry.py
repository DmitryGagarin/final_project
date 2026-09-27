"""
Named model registry.

Maps a user-chosen name (e.g. "expanded_v1") to the on-disk model path
plus the hyperparameters and metrics from the training run.

Stored as JSON at models/registry.json. Every read/write goes through
this module so the file format stays consistent.
"""
import json
import os
import shutil
from datetime import datetime

REGISTRY_PATH = os.path.join("models", "registry.json")


def _load():
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, REGISTRY_PATH)


def list_models():
    """Return the whole registry: {name: metadata}."""
    return _load()


def get(name):
    """Return metadata for a single entry, or None."""
    return _load().get(name)


def exists(name):
    return name in _load()


def register(name, path, hyperparams, metrics=None, notes="", overwrite=False):
    """
    Register a model. `path` should point at the merged model directory.

    Returns (ok, message).
    """
    if not name or not isinstance(name, str):
        return False, "Name must be a non-empty string"
    if not os.path.exists(path):
        return False, f"Path does not exist: {path}"

    data = _load()
    if name in data and not overwrite:
        return False, f"Name '{name}' is already taken. Pick another or set overwrite=True."

    data[name] = {
        "path": os.path.abspath(path),
        "created": datetime.now().isoformat(),
        "hyperparams": hyperparams or {},
        "metrics": metrics or {},
        "notes": notes,
    }
    _save(data)
    return True, f"Registered as '{name}'"


def update_metrics(name, metrics):
    data = _load()
    if name not in data:
        return False
    data[name]["metrics"] = metrics
    data[name]["updated"] = datetime.now().isoformat()
    _save(data)
    return True


def rename(old, new):
    data = _load()
    if old not in data or not new or new in data:
        return False
    data[new] = data.pop(old)
    _save(data)
    return True


def remove(name, delete_files=False):
    """
    Remove an entry. If delete_files=True, also removes the on-disk run
    directory (parent of `merged/`). Returns True on success.
    """
    data = _load()
    entry = data.pop(name, None)
    if entry is None:
        return False
    _save(data)

    if delete_files:
        merged_path = entry.get("path", "")
        run_dir = os.path.dirname(merged_path)
        models_root = os.path.abspath("./models")
        # Safety: only delete inside ./models
        if run_dir and os.path.abspath(run_dir).startswith(models_root):
            shutil.rmtree(run_dir, ignore_errors=True)
    return True


def summary_rows():
    """
    Return a list of dicts suitable for rendering in a Streamlit table.
    Sorted newest-first.
    """
    data = _load()
    rows = []
    for name, meta in data.items():
        hp = meta.get("hyperparams", {}) or {}
        m = meta.get("metrics", {}) or {}
        rows.append({
            "Name": name,
            "Created": meta.get("created", "")[:19].replace("T", " "),
            "r": hp.get("r", "-"),
            "alpha": hp.get("alpha", "-"),
            "dropout": hp.get("dropout", "-"),
            "epochs": hp.get("num_epochs", "-"),
            "batch": hp.get("batch_size", "-"),
            "LR": hp.get("learning_rate", "-"),
            "ROUGE-1": m.get("rouge1", "-"),
            "ROUGE-L": m.get("rougeL", "-"),
            "Notes": meta.get("notes", ""),
        })
    rows.sort(key=lambda r: r["Created"], reverse=True)
    return rows
