import os

import pandas as pd
from datasets import load_dataset

# Datasets to try.
DATASETS = [
    "mxlcw/telegram-financial-sentiment-summarization",
    "ZombitX64/telegram-financial-signalsv2",
    "cnn_dailymail",  # add for real summaries
]

# Candidate column names - ORDER MATTERS. Put real summary columns first.
TEXT_CANDIDATES = [
    "text", "cleaned_text", "content", "body", "post", "message", "article",
]
SUMMARY_CANDIDATES = [
    # Real summary candidates FIRST - never put "label" or "target" here
    "summarized_text", "summary", "highlights", "headline",
    "title", "abstract", "short_text",
]

# A summary should be reasonably shorter than the text.
MAX_SUMMARY_TO_TEXT_RATIO = 0.8
MIN_SUMMARY_CHARS = 15
MIN_TEXT_CHARS = 80


def pick_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def looks_like_summary(text_series, summary_series):
    """Sanity-check: are summary strings actually short-ish text, not labels?"""
    t = text_series.astype(str)
    s = summary_series.astype(str)

    avg_t = t.str.len().mean()
    avg_s = s.str.len().mean()
    if avg_s == 0 or avg_t == 0:
        return False, f"avg len text={avg_t:.0f}, summary={avg_s:.0f}"

    ratio = avg_s / avg_t
    if ratio > MAX_SUMMARY_TO_TEXT_RATIO:
        return False, f"summary too long (avg {avg_s:.0f} vs text {avg_t:.0f})"

    # Also reject when the "summary" is a tiny vocabulary (labels!)
    unique_ratio = s.nunique() / max(len(s), 1)
    if unique_ratio < 0.01:
        return False, f"only {s.nunique()} unique values out of {len(s)} - looks like labels"

    return True, f"avg len text={avg_t:.0f}, summary={avg_s:.0f}"


def process_dataset(name, out_dir="hf_raw"):
    print(f"=== {name} ===")
    try:
        ds = load_dataset(name)
    except Exception as e:
        print(f"Could not load: {e}")
        return None

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    print(f"Split: {split_name} | rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    text_col = pick_column(df.columns, TEXT_CANDIDATES)
    summary_col = pick_column(df.columns, SUMMARY_CANDIDATES)

    if text_col is None or summary_col is None:
        print(f"No real summary column found "
              f"(text={text_col}, summary={summary_col}). Skipping.")
        return None
    print(f"Using text='{text_col}', summary='{summary_col}'")

    out = df[[text_col, summary_col]].rename(
        columns={text_col: "cleaned_text", summary_col: "summary"}
    )

    out = out.dropna()
    out["cleaned_text"] = out["cleaned_text"].astype(str).str.strip()
    out["summary"] = out["summary"].astype(str).str.strip()
    out = out[(out["cleaned_text"].str.len() > MIN_TEXT_CHARS) &
              (out["summary"].str.len() > MIN_SUMMARY_CHARS)]

    if out.empty:
        print(f"No rows after basic filtering. Skipping.")
        return None

    ok, msg = looks_like_summary(out["cleaned_text"], out["summary"])
    print(f"Sanity check: {'good' if ok else 'bad'} {msg}")
    if not ok:
        print(f"Summary column does not look like real summaries. Skipping.")
        return None

    os.makedirs(out_dir, exist_ok=True)
    safe_name = name.replace("/", "__")
    per_file = os.path.join(out_dir, f"{safe_name}.csv")
    out.to_csv(per_file, sep=";", index=False)
    print(f"Saved {len(out)} rows → {per_file}")
    return out


def main():
    frames = []
    for name in DATASETS:
        df = process_dataset(name)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        print("No datasets produced usable rows.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["cleaned_text"])
    combined.to_csv("hf_telegram_corpus.csv", sep=";", index=False)
    print(f"Combined HuggingFace corpus: ")
    print(f"{len(combined)} rows → hf_telegram_corpus.csv")


if __name__ == "__main__":
    main()
