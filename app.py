"""Financial Advisor Bot — Streamlit front-end.

Wires the trained summariser, signal extractor, RL agent, and explanation
layer into a single web UI. Long-running work (training, scraping,
evaluation) runs in a subprocess or a background thread so the Streamlit
event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch

import model_registry
from market_data_providers import get_market_data_with_provider
from summarizer import summarize_post, extract_signal, set_model_path
from train_model import train_english_lora
from training_logger import TrainingLogger
from utils.utils import get_named_models


# ======================================================================
# Constants
# ======================================================================

CHANNELS = [
    "Bloomberg",
    "TraderTVLive",
    "TheFinancialExpressOnline",
    "vanillafinancenews",
    "TradingGainX",
]

FLAG_STALE_SECONDS = 3 * 60 * 60  # 3 hours
FLAG_DIR = "logs"

RL_AGENT_PATH = "dqn_model.pth"
RL_LOG_PATH = "logs/rl_training_output.log"
DEFAULT_TRAINING_LOG = "logs/training_output.log"
DEFAULT_CORPUS = "training_corpus_large.csv"
DEFAULT_GOLD_SET = "gold_set_v2.csv"

GOLD_SET_CANDIDATES = ("gold_set_v2.csv", "filtered_telegram_gold.csv")

CORPUS_SOURCES = {
    "Training corpus (merged)": {
        "path": "training_corpus_large.csv",
        "text_col": "cleaned_text",
        "summary_col": "summary",
    },
    "HF Telegram corpus (mxlcw)": {
        "path": "hf_telegram_corpus.csv",
        "text_col": "cleaned_text",
        "summary_col": "summary",
    },
    "Finnhub news corpus": {
        "path": "eulerpool_corpus.csv",
        "text_col": "content",
        "summary_col": "headline",
    },
    "RSS news corpus": {
        "path": "rss_corpus.csv",
        "text_col": "content",
        "summary_col": "headline",
    },
    "Gold set v2 (Finnhub, held out)": {
        "path": "gold_set_v2.csv",
        "text_col": "cleaned_text",
        "summary_col": "summary",
    },
    "Legacy gold set": {
        "path": "filtered_telegram_gold.csv",
        "text_col": "cleaned_text",
        "summary_col": "summary",
    },
    "Telegram DB (legacy)": None,
}

SESSION_DEFAULTS = {
    "scraping": False,
    "selected_model": None,
    "selected_model_name": None,
    "current_post": None,
    "rl_training": False,
    "rl_training_log_file": RL_LOG_PATH,
    "_loaded_model_path": None,
}


@dataclass
class AppConfig:
    """Configuration produced by the sidebar and consumed by the main body."""
    data_provider: str
    recommendation_mode: str
    confidence_threshold: float
    use_llm: bool
    llm_model: Optional[str]
    rl_available: bool


# ======================================================================
# Training-flag helpers (survive page refreshes)
# ======================================================================

def _flag_path(log_file: str) -> str:
    return log_file + ".flag"


def _write_flag(log_file: str, kind: str) -> None:
    """Write a marker file saying a training job is in progress."""
    try:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        with open(_flag_path(log_file), "w") as f:
            json.dump({"kind": kind, "started": datetime.now().isoformat()}, f)
    except Exception:
        pass


def _remove_flag(log_file: str) -> None:
    try:
        path = _flag_path(log_file)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _find_active_flags() -> list[tuple[str, dict]]:
    """Return [(log_file, flag_data), ...] for every live training marker."""
    out: list[tuple[str, dict]] = []
    if not os.path.exists(FLAG_DIR):
        return out

    for fname in os.listdir(FLAG_DIR):
        if not fname.endswith(".flag"):
            continue
        flag_path = os.path.join(FLAG_DIR, fname)
        try:
            age = time.time() - os.path.getmtime(flag_path)
        except OSError:
            continue
        if age > FLAG_STALE_SECONDS:
            try:
                os.remove(flag_path)
            except OSError:
                pass
            continue

        log_file = flag_path[: -len(".flag")]
        try:
            with open(flag_path) as f:
                data = json.load(f)
        except Exception:
            data = {"kind": "summariser"}
        out.append((log_file, data))
    return out


def _reconstruct_training_state() -> None:
    """Rebuild st.session_state from flags on disk after a page refresh."""
    for log_file, data in _find_active_flags():
        kind = data.get("kind", "summariser")
        if kind == "rl":
            if not st.session_state.get("rl_training", False):
                st.session_state.rl_training = True
                st.session_state.rl_training_log_file = log_file
        else:
            if not st.session_state.get("training", False):
                st.session_state.training = True
                st.session_state.training_log_file = log_file


# ======================================================================
# Corpus helpers
# ======================================================================

def get_random_corpus_row(source_name: str) -> Optional[dict]:
    """Pull one random example from the named corpus source."""
    spec = CORPUS_SOURCES.get(source_name)

    # Legacy Telegram DB path (source name maps to None).
    if spec is None:
        try:
            conn = sqlite3.connect("telegram_legacy/telegram_posts.db")
            cur = conn.cursor()
            placeholders = ",".join(["?"] * len(CHANNELS))
            cur.execute(
                f"SELECT channel, date, text, NULL FROM posts "
                f"WHERE channel IN ({placeholders}) "
                f"ORDER BY RANDOM() LIMIT 1",
                tuple(CHANNELS),
            )
            row = cur.fetchone()
            conn.close()
        except Exception:
            return None
        if not row:
            return None
        return {"channel": row[0], "date": row[1], "text": row[2], "reference": None}

    path = spec["path"]
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return None
    if df.empty:
        return None

    tcol, scol = spec["text_col"], spec["summary_col"]
    if tcol not in df.columns or scol not in df.columns:
        return None

    row = df.sample(1).iloc[0]
    return {
        "channel": spec["path"],
        "date": "",
        "text": str(row[tcol]),
        "reference": str(row[scol]),
    }


def corpus_stats(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        return None
    if df.empty:
        return None
    return {
        "rows": len(df),
        "cols": list(df.columns),
        "avg_text": int(df.iloc[:, 0].astype(str).str.len().mean()),
    }


def clean_corpus_file(path: str) -> tuple[int, int]:
    """In-place corpus cleaning. Returns (rows_before, rows_after)."""
    df = pd.read_csv(path, sep=";")
    n0 = len(df)

    url_only = df["cleaned_text"].str.match(r"^\s*https?://\S+\s*$", na=False)
    df = df[~url_only]

    def is_english_ish(s):
        if not isinstance(s, str) or not s:
            return False
        ascii_letters = sum(1 for c in s if c.isascii() and c.isalpha())
        total_letters = sum(1 for c in s if c.isalpha())
        if total_letters == 0:
            return False
        return (ascii_letters / total_letters) > 0.60

    mask = df["cleaned_text"].apply(is_english_ish) & df["summary"].apply(is_english_ish)
    df = df[mask]

    bad_summary = df["summary"].str.contains(r"https?://|@\w+", regex=True, na=False)
    df = df[~bad_summary]

    df = df[
        (df["cleaned_text"].str.len().between(100, 2000))
        & (df["summary"].str.len().between(20, 400))
    ]
    df = df.drop_duplicates(subset=["cleaned_text"])

    df.to_csv(path, sep=";", index=False)
    return n0, len(df)


# ======================================================================
# Subprocess helper
# ======================================================================

def _run_python_script(script: str, timeout: int = 3600) -> tuple[bool, str]:
    """Run a script as a subprocess. Returns (ok, error_message).

    stdout is discarded — the scripts write their results to CSV files
    which the UI reads directly. stderr is returned on failure.
    """
    if not os.path.exists(script):
        return False, f"Script not found: {script}"

    # Prefer the real `python3` on PATH; sys.executable can be a shim on macOS.
    python_exe = shutil.which("python3") or sys.executable

    try:
        r = subprocess.run(
            [python_exe, script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            cwd=os.getcwd(),
            text=True,
        )
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    if r.returncode == 0:
        return True, ""

    err = (r.stderr or "").strip() or f"Script exited with code {r.returncode} (no stderr)"
    return False, err[-3000:]


# ======================================================================
# Background worker threads
# ======================================================================

def run_training_thread(csv_path, log_file_path, hyperparams, model_name, notes):
    """Train a LoRA summariser in a thread; redirect stdout/stderr to a log."""
    orig_out, orig_err = sys.stdout, sys.stderr
    _write_flag(log_file_path, "summariser")
    try:
        os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)
        with open(log_file_path, "w") as f:
            sys.stdout = f
            sys.stderr = f
            try:
                model_path = train_english_lora(
                    csv_path, hyperparams, model_name=model_name, notes=notes,
                )
                st.session_state.training_result = {
                    "status": "success",
                    "model_path": model_path,
                    "model_name": model_name,
                }
            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                st.session_state.training_result = {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            finally:
                sys.stdout, sys.stderr = orig_out, orig_err
    except Exception as e:
        st.session_state.training_result = {
            "status": "error",
            "error": f"Failed to start: {e}",
        }
    finally:
        _remove_flag(log_file_path)
    st.session_state.training = False
    st.rerun()


def run_scraper_thread():
    """Scrape Telegram channels in a thread. Import is lazy so app boots without scraper.py."""
    try:
        from telegram_legacy.scraper import scrape_all_channels
        st.session_state.scrape_results = asyncio.run(scrape_all_channels())
    except Exception as e:
        st.session_state.scrape_results = {"error": str(e)}
    st.session_state.scraping = False
    st.rerun()


def run_rl_training_thread(log_file_path, model_path):
    """Train the RL agent in a thread; redirect stdout/stderr to a log."""
    set_model_path(model_path)
    orig_out, orig_err = sys.stdout, sys.stderr
    _write_flag(log_file_path, "rl")
    try:
        os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)
        with open(log_file_path, "w") as f:
            sys.stdout = f
            sys.stderr = f
            try:
                from train_rl_agent import train_rl
                train_rl()
                st.session_state.rl_training_result = {
                    "status": "success",
                    "message": "RL agent trained successfully!",
                }
            except Exception as e:
                print(f"ERROR in RL training: {e}")
                traceback.print_exc()
                st.session_state.rl_training_result = {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            finally:
                sys.stdout, sys.stderr = orig_out, orig_err
    except Exception as e:
        st.session_state.rl_training_result = {
            "status": "error",
            "error": f"Failed to start RL training: {e}",
        }
    finally:
        _remove_flag(log_file_path)
    st.session_state.rl_training = False
    st.rerun()


# ======================================================================
# Gold-set evaluation + bootstrap
# ======================================================================

def _evaluate_predictions(preds: list[str], refs: list[str]) -> dict:
    import evaluate
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return {k: v * 100 for k, v in scores.items()}


def evaluate_on_gold(model_path: str, gold_path: str, max_examples: Optional[int] = None):
    """Run the summariser on the gold set and return (scores, rows, preds, refs)."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    gold = pd.read_csv(gold_path, sep=";")
    gold.columns = gold.columns.str.strip()

    if "cleaned_text" not in gold.columns:
        for cand in ("content", "text", "body"):
            if cand in gold.columns:
                gold = gold.rename(columns={cand: "cleaned_text"})
                break
    if "summary" not in gold.columns:
        for cand in ("headline", "title", "highlights"):
            if cand in gold.columns:
                gold = gold.rename(columns={cand: "summary"})
                break
    if "cleaned_text" not in gold.columns or "summary" not in gold.columns:
        raise ValueError(f"Gold CSV must have text/summary columns: {list(gold.columns)}")

    if max_examples is not None:
        gold = gold.head(max_examples)

    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    if torch.backends.mps.is_available():
        mdl.to("mps")
    mdl.eval()
    device = next(mdl.parameters()).device

    preds, refs, rows = [], [], []
    for _, row in gold.iterrows():
        inp = tok("summarize: " + str(row["cleaned_text"]),
                  return_tensors="pt", truncation=True, max_length=384)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = mdl.generate(
                **inp, max_new_tokens=64, num_beams=5, early_stopping=True,
                no_repeat_ngram_size=4, repetition_penalty=1.5,
            )
        pred = tok.decode(out[0], skip_special_tokens=True)
        ref = str(row["summary"])
        preds.append(pred)
        refs.append(ref)
        rows.append({
            "reference": ref,
            "prediction": pred,
            "input": str(row["cleaned_text"])[:200] + "...",
        })

    scores = {k: round(v, 2) for k, v in _evaluate_predictions(preds, refs).items()}
    return scores, rows, preds, refs


def bootstrap_ci(preds, refs, n_boot: int = 200, alpha: float = 0.05, seed: int = 42) -> Optional[dict]:
    rng = np.random.default_rng(seed)
    n = len(preds)
    if n < 5:
        return None

    boot = {"rouge1": [], "rouge2": [], "rougeL": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bp = [preds[i] for i in idx]
        br = [refs[i] for i in idx]
        try:
            sc = _evaluate_predictions(bp, br)
            for k in boot:
                if k in sc:
                    boot[k].append(sc[k])
        except Exception:
            continue

    lo_q = 100 * alpha / 2
    hi_q = 100 * (1 - alpha / 2)
    return {
        k: {
            "mean": float(np.mean(v)),
            "lo": float(np.percentile(v, lo_q)),
            "hi": float(np.percentile(v, hi_q)),
            "n": len(v),
        }
        for k, v in boot.items()
        if v
    }


# ======================================================================
# RL helpers
# ======================================================================

_rl_agent = None
_rl_agent_loaded = False


def get_rl_agent():
    global _rl_agent, _rl_agent_loaded
    if not _rl_agent_loaded:
        if os.path.exists(RL_AGENT_PATH):
            try:
                from RL.rl_agent import DQNAgent
                agent = DQNAgent(state_dim=7, action_dim=3)
                agent.load(RL_AGENT_PATH)
                _rl_agent = agent
            except Exception as e:
                st.error(f"Failed to load RL agent: {e}")
                _rl_agent = None
        else:
            _rl_agent = None
        _rl_agent_loaded = True
    return _rl_agent


def get_rl_recommendation(df_feat, signal_action: str, confidence_threshold: float = 0.05):
    """Ask the RL agent for an action.

    Returns (final_action | None, q_str, confident: bool).
    """
    agent = get_rl_agent()
    if agent is None:
        return None, "RL agent not available.", True

    try:
        from RL.rl_env import TradingEnv

        action_val = {"buy": 1, "sell": 2}.get(signal_action, 0)
        signal_series = pd.Series([0] * len(df_feat))
        signal_series.iloc[-1] = action_val

        env = TradingEnv(df_feat, signal_series)
        state, _ = env.reset()
        for _ in range(len(df_feat) - 1):
            env.step(0)

        rl_action = agent.act(state, training=False)
        final_action = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(rl_action, "HOLD")

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q_values = agent.q_net(state_tensor)
        q = q_values[0].detach().cpu().numpy()

        q_str = f"Hold: {q[0]:.2f}, Buy: {q[1]:.2f}, Sell: {q[2]:.2f}"
        sorted_q = np.sort(q)[::-1]
        margin = float(sorted_q[0] - sorted_q[1])

        if margin < confidence_threshold:
            return None, f"{q_str} (margin {margin:.3f} below threshold)", False
        return final_action, q_str, True
    except Exception as e:
        return None, str(e), False


# ======================================================================
# Sidebar — split into focused sub-renderers
# ======================================================================

def render_sidebar() -> AppConfig:
    with st.sidebar:
        st.subheader("Configuration")
        st.caption("Market data provider: **Yahoo Finance** (no API key required).")
        data_provider = "yfinance"

        rl_available = os.path.exists(RL_AGENT_PATH)
        recommendation_mode, confidence_threshold = _render_recommendation_controls(rl_available)

        st.subheader("Explanation Generator")
        use_llm, llm_model = _render_llm_controls()

        st.subheader("RL Agent Training")
        _render_rl_training_controls(rl_available)

        st.divider()
        _render_corpus_inventory()

    return AppConfig(
        data_provider=data_provider,
        recommendation_mode=recommendation_mode,
        confidence_threshold=confidence_threshold,
        use_llm=use_llm,
        llm_model=llm_model,
        rl_available=rl_available,
    )


def _render_recommendation_controls(rl_available: bool) -> tuple[str, float]:
    if not rl_available:
        st.info("RL agent not trained. Using Rule-Based mode.")
        return "Rule-Based (Explainable)", 0.05

    mode = st.radio(
        "Recommendation Engine",
        ["Rule-Based (Explainable)", "Reinforcement Learning (RL)"],
        index=1,
    )
    threshold = st.slider(
        "RL confidence threshold", 0.0, 0.5, 0.05, 0.01,
        help="Fall back to rule-based if top two Q-values are closer than this.",
    )
    return mode, threshold


def _render_llm_controls() -> tuple[bool, Optional[str]]:
    try:
        from llm_advisor import is_available as llm_available, list_models as llm_list_models
        llm_ok = llm_available()
    except Exception:
        llm_ok = False
        llm_list_models = lambda: []

    if not llm_ok:
        st.caption("Ollama not running — using template explanations.")
        return False, None

    st.success("Ollama detected")
    models = llm_list_models()
    if models:
        llm_model = st.selectbox("LLM model", models, index=0)
    else:
        llm_model = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
    use_llm = st.checkbox("Use LLM for explanations", value=True)
    return use_llm, llm_model


def _render_rl_training_controls(rl_available: bool) -> None:
    if rl_available:
        _render_rl_ready_panel()
    else:
        _render_rl_train_panel()

    if st.session_state.get("rl_training", False):
        _render_rl_training_progress()

    result = st.session_state.get("rl_training_result")
    if not result:
        return

    if result["status"] == "success":
        st.success(result["message"])
        _reset_rl_agent_cache()
        st.rerun()
    else:
        st.error(f"RL training failed: {result['error']}")
        if "traceback" in result:
            with st.expander("Show error details"):
                st.code(result["traceback"])


def _render_rl_ready_panel() -> None:
    st.success("RL agent is ready!")
    if st.button("Delete RL agent"):
        try:
            for f in (RL_AGENT_PATH, "rl_training_metrics.json",
                      "rl_test_set.pkl", "rl_train_set.pkl", "rl_main_ticker.txt"):
                if os.path.exists(f):
                    os.remove(f)
            _reset_rl_agent_cache()
            st.success("RL agent deleted.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to delete: {e}")


def _render_rl_train_panel() -> None:
    st.warning("RL agent not found.")
    if not st.button("Train RL Agent (background)"):
        return
    if st.session_state.get("rl_training", False):
        return

    sel_model_path = st.session_state.get("selected_model")
    if sel_model_path is None:
        st.warning("Please select a summarisation model first.")
        return

    st.session_state.rl_training = True
    st.session_state.rl_training_result = None
    threading.Thread(
        target=run_rl_training_thread,
        args=(RL_LOG_PATH, sel_model_path),
    ).start()


def _render_rl_training_progress() -> None:
    st.info("RL training in progress...")
    rl_log = st.session_state.get("rl_training_log_file", RL_LOG_PATH)

    if os.path.exists(rl_log):
        with open(rl_log) as f:
            content = f.read()
        mtime = datetime.fromtimestamp(os.path.getmtime(rl_log)).strftime("%H:%M:%S")
        st.caption(f"Last updated: {mtime}")
        st.text_area("RL Training Logs", value=content, height=200)
    else:
        st.text("Waiting for logs...")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Refresh RL Logs", key="refresh_rl_logs"):
            st.rerun()
    with c2:
        if st.button("Force reset", key="force_reset_rl"):
            _remove_flag(rl_log)
            st.session_state.rl_training = False
            st.rerun()


def _render_corpus_inventory() -> None:
    st.subheader("Corpus Inventory")
    for label, spec in CORPUS_SOURCES.items():
        if spec is None:
            continue
        stats = corpus_stats(spec["path"])
        if stats:
            st.caption(f"**{label}** — {stats['rows']} rows")
        else:
            st.caption(f"**{label}** — missing")


def _reset_rl_agent_cache() -> None:
    global _rl_agent, _rl_agent_loaded
    _rl_agent = None
    _rl_agent_loaded = False


# ======================================================================
# Model registry
# ======================================================================

def render_model_registry() -> None:
    with st.expander("Model Registry", expanded=False):
        st.caption(
            "Named models with their hyperparameters and validation metrics. "
            "Names are created when you start training (see 'Train New Model' below)."
        )

        rows = model_registry.summary_rows()
        if not rows:
            st.info("No models registered yet. Train one below with a name to get started.")
            return

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        _render_registry_management(rows)


def _render_registry_management(rows: list[dict]) -> None:
    st.markdown("**Manage a model**")
    names = [r["Name"] for r in rows]
    sel = st.selectbox("Select model", names, key="registry_manage_sel")

    c1, c2, c3 = st.columns(3)
    with c1:
        new_name = st.text_input("Rename to", value="", key="registry_rename_input")
        if st.button("Rename", key="registry_rename_btn") and new_name:
            if model_registry.rename(sel, new_name):
                st.success(f"Renamed '{sel}' → '{new_name}'")
                st.rerun()
            else:
                st.error("Rename failed (name in use or invalid).")
    with c2:
        if st.button("Delete entry (keep files)", key="registry_del_btn"):
            if model_registry.remove(sel, delete_files=False):
                st.success(f"Removed '{sel}' from registry.")
                st.rerun()
    with c3:
        if st.button("Delete entry AND files", key="registry_del_files_btn"):
            if model_registry.remove(sel, delete_files=True):
                st.success(f"Removed '{sel}' and deleted its run directory.")
                st.rerun()

    entry = model_registry.get(sel)
    if entry:
        st.markdown(f"**Details for `{sel}`**")
        st.json({
            "path": entry.get("path"),
            "created": entry.get("created"),
            "hyperparams": entry.get("hyperparams"),
            "metrics": entry.get("metrics"),
            "notes": entry.get("notes"),
        })


# ======================================================================
# Step 1 — model selection
# ======================================================================

def render_model_selection() -> None:
    named_models = get_named_models()

    st.subheader("Step 1: Choose a primary summarisation model")
    if not named_models:
        st.warning("No models found. Train one below.")
        return

    cols = st.columns(min(3, len(named_models)))
    for idx, m in enumerate(named_models):
        label = m["name"] if m["registered"] else f"{m['name']} (unregistered)"
        with cols[idx % 3]:
            if st.button(label, key=f"model_{idx}"):
                st.session_state.selected_model = m["path"]
                st.session_state.selected_model_name = m["name"]
                st.rerun()

    if not st.session_state.get("selected_model"):
        st.info("Please select a model to proceed.")
        return

    sel_path = st.session_state.selected_model
    sel_name = st.session_state.get("selected_model_name", "")
    st.success(f"Selected: **{sel_name}**")

    entry = model_registry.get(sel_name) if sel_name else None
    if entry:
        _render_selected_model_metrics(entry)

    if st.session_state.get("_loaded_model_path") != sel_path:
        set_model_path(sel_path)
        st.session_state["_loaded_model_path"] = sel_path


def _render_selected_model_metrics(entry: dict) -> None:
    hp = entry.get("hyperparams", {})
    mm = entry.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("r", hp.get("r", "-"))
    c2.metric("dropout", hp.get("dropout", "-"))
    c3.metric("epochs", hp.get("num_epochs", "-"))
    c4.metric("val ROUGE-L", mm.get("rougeL", "-"))
    if entry.get("notes"):
        st.caption(f"Notes: {entry['notes']}")


# ======================================================================
# Step 1b — gold-set evaluation
# ======================================================================

def render_gold_evaluation() -> None:
    st.subheader("Step 1b: Evaluate selected model on a gold set")

    gold_path = _pick_gold_set_path()
    col_a, col_b = st.columns(2)
    with col_a:
        run_point = st.button("Run point-estimate ROUGE")
    with col_b:
        run_boot = st.button("Run with bootstrap 95% CIs (slower)")

    if run_point or run_boot:
        _run_gold_evaluation(gold_path)
    if run_boot and st.session_state.get("gold_preds"):
        _run_bootstrap()

    if st.session_state.get("gold_scores"):
        _render_gold_results()


def _pick_gold_set_path() -> str:
    options = [p for p in GOLD_SET_CANDIDATES if os.path.exists(p)]
    if not options:
        st.warning("No gold set found. Run `python build_gold_set.py` first.")
        return st.text_input("Or paste a path:", DEFAULT_GOLD_SET)
    return st.selectbox("Gold CSV", options, index=0)


def _run_gold_evaluation(gold_path: str) -> None:
    if not st.session_state.get("selected_model"):
        st.error("Select a summarisation model in Step 1 first.")
        return
    if not os.path.exists(gold_path):
        st.error(f"{gold_path} not found.")
        return

    with st.spinner("Evaluating on gold set..."):
        try:
            scores, rows, preds, refs = evaluate_on_gold(
                st.session_state.selected_model, gold_path,
            )
            st.session_state.update({
                "gold_scores": scores,
                "gold_rows": rows,
                "gold_path": gold_path,
                "gold_preds": preds,
                "gold_refs": refs,
            })
            sel_name = st.session_state.get("selected_model_name")
            if sel_name:
                model_registry.update_metrics(sel_name, {
                    "rouge1": scores.get("rouge1"),
                    "rouge2": scores.get("rouge2"),
                    "rougeL": scores.get("rougeL"),
                    "gold_eval_path": gold_path,
                })
        except Exception as e:
            st.error(f"Evaluation failed: {e}")


def _run_bootstrap() -> None:
    with st.spinner("Bootstrapping 95% confidence intervals..."):
        try:
            st.session_state["gold_ci"] = bootstrap_ci(
                st.session_state["gold_preds"],
                st.session_state["gold_refs"],
                n_boot=200,
            )
        except Exception as e:
            st.error(f"Bootstrap failed: {e}")


def _render_gold_results() -> None:
    st.caption(f"Evaluated on: `{st.session_state.get('gold_path', '?')}`")
    scores = st.session_state["gold_scores"]
    ci = st.session_state.get("gold_ci")

    cols = st.columns(len(scores))
    for i, (k, v) in enumerate(scores.items()):
        if ci and k in ci:
            c = ci[k]
            cols[i].metric(k.upper(), f"{v:.2f}",
                           delta=f"95% CI [{c['lo']:.1f}, {c['hi']:.1f}]")
        else:
            cols[i].metric(k.upper(), f"{v:.2f}")

    with st.expander("Show individual predictions (sample)"):
        df = pd.DataFrame(st.session_state["gold_rows"]).head(15)
        for i, row in df.iterrows():
            st.markdown(f"**Example {i + 1}**")
            st.markdown(f"*Input:* {row['input']}")
            st.markdown(f"*Reference:* {row['reference']}")
            st.markdown(f"*Prediction:* {row['prediction']}")
            st.divider()


# ======================================================================
# Evaluation Suite (ablation / walk-forward / RL comparison)
# ======================================================================

def render_evaluation_suite(rl_available: bool) -> None:
    st.subheader("Evaluation Suite")
    tab_abl, tab_wf, tab_rl = st.tabs(["Ablation", "Walk-forward", "RL comparison"])

    with tab_abl:
        _render_script_tab(
            caption="Runs `ablation.py`. Fast (~10s).",
            button_label="Run ablation",
            button_key="run_abl",
            script="ablation.py",
            spinner_msg="Running ablation.py...",
            timeout=600,
            results_csv="ablation_results.csv",
            state_prefix="abl",
        )

    with tab_wf:
        _render_script_tab(
            caption="Runs `walkforward.py`. Slow: ~10 min (retrains 4×).",
            button_label="Run walk-forward",
            button_key="run_wf",
            script="walkforward.py",
            spinner_msg="Running walkforward.py...",
            timeout=3600,
            results_csv="walkforward_results.csv",
            state_prefix="wf",
            summary_renderer=_render_walkforward_summary,
        )

    with tab_rl:
        _render_rl_comparison_tab(rl_available)


def _render_script_tab(caption, button_label, button_key, script, spinner_msg,
                       timeout, results_csv, state_prefix, summary_renderer=None):
    st.caption(caption)
    if st.button(button_label, key=button_key):
        with st.spinner(spinner_msg):
            ok, err = _run_python_script(script, timeout=timeout)
        st.session_state[f"{state_prefix}_ok"] = ok
        st.session_state[f"{state_prefix}_err"] = err
        st.rerun()

    if f"{state_prefix}_ok" in st.session_state:
        if st.session_state[f"{state_prefix}_ok"]:
            st.success(f"{button_label} completed")
        else:
            st.error(f"{button_label} failed")
            if st.session_state.get(f"{state_prefix}_err"):
                st.code(st.session_state[f"{state_prefix}_err"])

    if os.path.exists(results_csv):
        st.markdown("**Latest results**")
        df = pd.read_csv(results_csv)
        st.dataframe(df, use_container_width=True)
        if summary_renderer:
            summary_renderer(df)


def _render_walkforward_summary(df: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean return %", f"{df['return_pct'].mean():+.2f}",
              delta=f"± {df['return_pct'].std():.2f}")
    c2.metric("Mean Sharpe", f"{df['sharpe'].mean():+.3f}",
              delta=f"± {df['sharpe'].std():.3f}")
    c3.metric("Mean max DD %", f"{df['max_dd'].mean():.2f}",
              delta=f"± {df['max_dd'].std():.2f}")


def _render_rl_comparison_tab(rl_available: bool) -> None:
    st.caption("Runs `evaluate_rl.py`.")
    can_run = rl_available and os.path.exists("rl_test_set.pkl")
    if not can_run:
        st.warning("Train the RL agent first.")

    if st.button("Run RL comparison", key="run_rl_eval", disabled=not can_run):
        with st.spinner("Running evaluate_rl.py..."):
            ok, err = _run_python_script("evaluate_rl.py", timeout=600)
        st.session_state["rl_eval_ok"] = ok
        st.session_state["rl_eval_err"] = err
        st.rerun()

    if "rl_eval_ok" in st.session_state:
        if st.session_state["rl_eval_ok"]:
            st.success("RL comparison completed")
        else:
            st.error("RL comparison failed")
            if st.session_state.get("rl_eval_err"):
                st.code(st.session_state["rl_eval_err"])

    if os.path.exists("rl_comparison.csv"):
        st.dataframe(pd.read_csv("rl_comparison.csv", index_col="Strategy"),
                     use_container_width=True)
    if os.path.exists("rl_equity_curves.png"):
        # Bug fix: use_column_width is deprecated in recent Streamlit.
        st.image("rl_equity_curves.png", use_container_width=True)


# ======================================================================
# Corpus maintenance
# ======================================================================

def render_corpus_maintenance() -> None:
    with st.expander("Corpus Maintenance"):
        corpus_path = st.text_input("Corpus path", DEFAULT_CORPUS, key="clean_path")

        _render_clean_corpus_button(corpus_path)
        _render_corpus_preview(corpus_path)

        st.divider()
        _render_subset_creator(corpus_path)


def _render_clean_corpus_button(corpus_path: str) -> None:
    if not st.button("Clean corpus (in place)"):
        return
    if not os.path.exists(corpus_path):
        st.error(f"{corpus_path} not found.")
        return

    backup = corpus_path + ".backup"
    if not os.path.exists(backup):
        pd.read_csv(corpus_path, sep=";").to_csv(backup, sep=";", index=False)
        st.info(f"Backup saved to {backup}")

    with st.spinner("Cleaning..."):
        n0, n1 = clean_corpus_file(corpus_path)
    st.success(f"Dropped {n0 - n1} rows. Kept {n1} / {n0}.")


def _render_corpus_preview(corpus_path: str) -> None:
    if not os.path.exists(corpus_path):
        return
    try:
        df = pd.read_csv(corpus_path, sep=";")
        st.write(
            f"Rows: **{len(df)}** | "
            f"Avg text: **{df['cleaned_text'].str.len().mean():.0f}** | "
            f"Avg summary: **{df['summary'].str.len().mean():.0f}**"
        )
    except Exception as e:
        st.warning(f"Could not preview corpus: {e}")


def _render_subset_creator(corpus_path: str) -> None:
    sub_size = st.number_input("Subset size (rows)", 200, 5000, 2000, 200,
                               key="sub_size")
    if not st.button("Create subset → training_corpus_small.csv"):
        return
    if not os.path.exists(corpus_path):
        st.error(f"{corpus_path} not found.")
        return

    df_full = pd.read_csv(corpus_path, sep=";")
    n = min(sub_size, len(df_full))
    df_small = df_full.sample(n=n, random_state=42).reset_index(drop=True)
    df_small.to_csv("training_corpus_small.csv", sep=";", index=False)
    st.success(f"Wrote {n} rows → training_corpus_small.csv")


# ======================================================================
# Train new model
# ======================================================================

def render_training_form() -> None:
    with st.expander("Train New Model", expanded=False):
        st.markdown(
            "Train a LoRA-T5 summariser. A **name is required** so the model "
            "can be found later in the Model Registry."
        )

        model_name = st.text_input(
            "Model name (e.g. expanded_v1, lora_r16_3ep)", value="",
            key="train_model_name",
            help="Must be unique. Existing names are rejected.",
        )

        name_taken = bool(model_name) and model_registry.exists(model_name)
        if name_taken:
            st.error(f"Name '{model_name}' already exists. Pick another.")

        notes = st.text_input("Notes (optional)", value="", key="train_notes")

        st.divider()
        csv_path = st.text_input("Path to training CSV", DEFAULT_CORPUS)
        log_file = st.text_input("Log file path", DEFAULT_TRAINING_LOG)
        hyperparams = _render_hyperparameter_inputs(model_name)

        _render_training_trigger(model_name, name_taken, csv_path, log_file, hyperparams, notes)
        _render_training_progress()
        _render_training_result()


def _render_hyperparameter_inputs(model_name: str) -> dict:
    col1, col2, col3 = st.columns(3)
    with col1:
        lora_r = st.number_input("r (rank)", 4, 64, 16, 4)
        lora_alpha = st.number_input("alpha", 8, 128, 32, 8)
    with col2:
        lora_dropout = st.slider("dropout", 0.0, 0.5, 0.10, 0.05)
        learning_rate = st.number_input("learning rate", 1e-5, 1e-3, 5e-4,
                                        format="%.0e", step=1e-5)
    with col3:
        num_epochs = st.number_input("epochs", 1, 50, 3, 1)
        batch_size = st.number_input("batch size", 1, 8, 2, 1)

    hyperparams = {
        "r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout,
        "learning_rate": learning_rate, "num_epochs": num_epochs,
        "batch_size": batch_size,
    }

    st.caption(
        f"Will register as **{model_name or '—'}** with "
        f"r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, "
        f"epochs={num_epochs}, batch={batch_size}, lr={learning_rate:.0e}"
    )
    return hyperparams


def _render_training_trigger(model_name, name_taken, csv_path, log_file, hyperparams, notes):
    disabled = (not model_name) or name_taken or st.session_state.get("training", False)
    if not st.button("Start Training", key="train_button", disabled=disabled):
        return
    if st.session_state.get("training", False):
        st.warning("Training already in progress.")
        return
    if not os.path.exists(csv_path):
        st.error("CSV file not found.")
        return

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    st.session_state.training = True
    st.session_state.training_result = None
    st.session_state.training_log_file = log_file
    threading.Thread(
        target=run_training_thread,
        args=(csv_path, log_file, hyperparams, model_name, notes),
    ).start()


def _render_training_progress() -> None:
    if not st.session_state.get("training", False):
        return

    st.info("Training in progress...")
    log_path = st.session_state.get("training_log_file", DEFAULT_TRAINING_LOG)

    if os.path.exists(log_path):
        with open(log_path) as f:
            content = f.read()
        mtime = datetime.fromtimestamp(os.path.getmtime(log_path)).strftime("%H:%M:%S")
        st.caption(f"Last updated: {mtime}")
        st.text_area("Training Logs", value=content, height=300)
    else:
        st.text("Waiting for logs...")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Refresh Logs", key="refresh_train_logs"):
            st.rerun()
    with c2:
        if st.button("Force reset (unstick UI)", key="force_reset_train"):
            _remove_flag(log_path)
            st.session_state.training = False
            st.rerun()


def _render_training_result() -> None:
    result = st.session_state.get("training_result")
    if not result:
        return
    if result["status"] == "success":
        st.success(
            f"Training completed! Model: "
            f"{result.get('model_name') or result['model_path']}"
        )
        st.rerun()
    else:
        st.error(f"Training failed: {result['error']}")


# ======================================================================
# Training history dashboard
# ======================================================================

def render_training_history() -> None:
    with st.expander("Training History Dashboard"):
        all_runs = TrainingLogger().get_all_runs()
        if not all_runs:
            st.info("No training runs found.")
            return
        runs_df = pd.DataFrame(all_runs)
        display_cols = [
            c for c in ["run_id", "hyperparameters", "duration_seconds",
                        "total_epochs", "status"]
            if c in runs_df.columns
        ]
        st.dataframe(runs_df[display_cols])


# ======================================================================
# Telegram scraper panel (optional)
# ======================================================================

def render_scraper() -> None:
    with st.expander("Scrape Telegram posts (optional)"):
        if st.button("Scrape New Posts"):
            if not st.session_state.get("scraping", False):
                st.session_state.scraping = True
                threading.Thread(target=run_scraper_thread).start()

        if st.session_state.get("scraping", False):
            st.info("Scraping in progress...")

        results = st.session_state.pop("scrape_results", None)
        if results is None:
            return

        if "error" in results:
            st.error(f"Scraping error: {results['error']}")
        else:
            for ch, data in results.items():
                if data["error"]:
                    st.error(f"{ch}: {data['error']}")
                else:
                    st.success(f"{ch}: {data['saved']} new posts")
        st.session_state.scraping = False


# ======================================================================
# Step 2 — analyse a random sample
# ======================================================================

def render_analysis(config: AppConfig) -> None:
    if not st.session_state.get("selected_model"):
        st.info("Select a model in Step 1 to analyse posts.")
        return

    st.subheader("Step 2: Analyse a random corpus sample")
    source_name = st.selectbox(
        "Sample source", list(CORPUS_SOURCES.keys()), index=0, key="sample_source",
    )

    if st.button("Load Random Sample"):
        row = get_random_corpus_row(source_name)
        if row is None:
            st.warning(f"No rows available from '{source_name}'.")
            st.session_state.current_post = None
        else:
            st.session_state.current_post = row
            st.rerun()

    post = st.session_state.get("current_post")
    if post:
        _render_analysis_for_post(post, config)


def _render_analysis_for_post(post: dict, config: AppConfig) -> None:
    st.write(f"**Source:** {post.get('channel', '?')}")
    st.write(f"**Original text:** {post['text']}")

    with st.spinner("Summarising..."):
        summary = summarize_post(post["text"])
    st.subheader("Generated summary")
    st.write(summary)

    if post.get("reference"):
        st.markdown(f"**Reference summary:** {post['reference']}")

    ticker, action = extract_signal(summary)
    st.write(f"**Ticker:** {ticker} | **Analyst action:** {action}")

    if ticker is None:
        st.warning("No ticker found in summary.")
        return

    _render_recommendation(ticker, action, summary, config)


def _render_recommendation(ticker, action, summary, config: AppConfig) -> None:
    df_feat = _fetch_features(ticker, config.data_provider)
    if df_feat is None:
        return

    latest = df_feat.iloc[-1]
    rsi = float(latest["rsi"])
    price = float(latest["Close"])

    final_action, q_str = _decide_action(config, df_feat, action, ticker, rsi, price)
    explanation = _generate_explanation(
        config, summary, ticker, action, rsi, price, final_action, q_str,
    )

    st.subheader("Advisor recommendation")
    st.write(f"**Action:** {final_action}")
    st.write(f"**Explanation:** {explanation}")


def _fetch_features(ticker: str, provider: str) -> Optional[pd.DataFrame]:
    with st.spinner(f"Fetching market data ({provider})..."):
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        try:
            df = get_market_data_with_provider(
                ticker, start_date, end_date, provider=provider,
            )
        except Exception as e:
            st.error(f"Data fetch error: {e}")
            return None

    if df is None or df.empty:
        st.warning(f"No market data for {ticker}.")
        return None

    from market_data import compute_features
    return compute_features(df)


def _decide_action(config: AppConfig, df_feat, action, ticker, rsi, price) -> tuple[str, Optional[str]]:
    """Return (final_action, q_str). Falls back to rule-based when RL is unconfident."""
    from advisor import get_advice

    if config.recommendation_mode == "Rule-Based (Explainable)":
        final_action, _ = get_advice(action, ticker, rsi, price)
        st.caption("Engine: Rule-Based")
        return final_action, None

    # Bug fix: original call passed 4 positional args but the function
    # only accepts 3 (df_feat, signal_action, confidence_threshold).
    rl_action, q_str, confident = get_rl_recommendation(
        df_feat, action, config.confidence_threshold,
    )
    if rl_action is not None and confident:
        st.caption("Engine: RL (confident)")
        return rl_action, q_str

    reason = "low-confidence" if q_str and "margin" in q_str else "failed"
    st.caption(f"Engine: Rule-Based (RL {reason}: {q_str})")
    final_action, _ = get_advice(action, ticker, rsi, price)
    return final_action, q_str


def _generate_explanation(config: AppConfig, summary, ticker, action,
                          rsi, price, final_action, q_str) -> str:
    from llm_advisor import generate_explanation

    if config.use_llm:
        with st.spinner("Generating LLM explanation..."):
            text = generate_explanation(
                summary=summary, ticker=ticker, analyst_action=action,
                rsi=rsi, price=price, final_action=final_action,
                q_str=q_str, model=config.llm_model,
            )
        st.caption("Explanation source: Ollama LLM")
    else:
        text = generate_explanation(
            summary=summary, ticker=ticker, analyst_action=action,
            rsi=rsi, price=price, final_action=final_action,
            q_str=q_str, model=None,
        )
        st.caption("Explanation source: template")
    return text


# ======================================================================
# Entry point
# ======================================================================

def main() -> None:
    st.set_page_config(page_title="Financial Advisor Bot", layout="wide")
    st.title("Financial Advisor Bot")

    config = render_sidebar()
    render_model_registry()
    render_model_selection()
    render_gold_evaluation()
    render_evaluation_suite(config.rl_available)
    render_corpus_maintenance()
    render_training_form()
    render_training_history()
    render_scraper()
    render_analysis(config)


def _init_session_state() -> None:
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


if __name__ == "__main__":
    _init_session_state()
    _reconstruct_training_state()  # Rebuild in-progress state from disk.
    main()