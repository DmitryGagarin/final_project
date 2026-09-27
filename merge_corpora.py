import os

import pandas as pd

SOURCES = {
    "eulerpool_corpus.csv": {
        "text": ["content", "body", "cleaned_text", "text"],
        "summary": ["headline", "title", "summary"],
    },
    "rss_corpus.csv": {
        "text": ["content", "body", "cleaned_text", "text"],
        "summary": ["headline", "title", "summary"],
    },
    "hf_telegram_corpus.csv": {
        "text": ["cleaned_text", "text"],
        "summary": ["summary"],
    },
    "cnn_dailymail_corpus.csv": {
        "text": ["article", "cleaned_text", "text"],
        "summary": ["highlights", "summary"],
    },
}


def pick(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def load_one(path, spec):
    if not os.path.exists(path):
        print(f"{path} not found, skipping")
        return None

    try:
        df = pd.read_csv(path, sep=";")
    except pd.errors.EmptyDataError:
        print(f"{path} is empty (header only). Skipping")
        return None

    if df.empty:
        print(f"{path} has 0 rows. Skipping")
        return None

    tcol = pick(df, spec["text"])
    scol = pick(df, spec["summary"])
    if tcol is None or scol is None:
        print(f"{path}: cannot find text/summary columns "
              f"(have {list(df.columns)})")
        return None

    out = df[[tcol, scol]].rename(columns={tcol: "cleaned_text",
                                           scol: "summary"})
    print(f"{path}: {len(out)} rows (text={tcol}, summary={scol})")
    return out


def main():
    frames = []
    for path, spec in SOURCES.items():
        df = load_one(path, spec)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        print("No usable source files. Nothing to merge.")
        return

    train = pd.concat(frames, ignore_index=True)
    train["cleaned_text"] = train["cleaned_text"].astype(str).str.strip()
    train["summary"] = train["summary"].astype(str).str.strip()
    train = train[(train["cleaned_text"].str.len() > 50) &
                  (train["summary"].str.len() > 10)]
    train = train.drop_duplicates(subset=["cleaned_text"])

    train.to_csv("training_corpus_large.csv", sep=";", index=False)
    print(f"Training corpus: {len(train)} examples "
          f"→ training_corpus_large.csv")


if __name__ == "__main__":
    main()
