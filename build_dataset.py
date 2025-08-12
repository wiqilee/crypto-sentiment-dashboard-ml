# ============================================
# Crypto Sentiment Dashboard Machine Learning
# Author: wiqilee
# License: MIT
# ============================================
# build_dataset.py — score sentimen dan hasilkan dataset.csv yang dipakai UI

# SPDX-License-Identifier: MIT
# build_dataset.py — score sentiment and produce dataset.csv used by the UI

import argparse
import warnings
from typing import List
import re

import numpy as np
import pandas as pd

from config import cfg
from sentiment_backends import get_scorer


def _debug(msg: str, on: bool):
    if on:
        print(f"[DEBUG] {msg}")


def _ensure_text_columns(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series of text to be scored with priority:
      content > (title + ' ' + description) > title > description
    """
    cols = {c.lower(): c for c in df.columns}
    get = lambda *names: next((cols[n] for n in names if n in cols), None)

    c_content = get("content")
    c_title = get("title", "headline")
    c_desc = get("description", "summary", "desc")

    if c_content and c_content in df:
        return df[c_content].fillna("").astype(str)

    if c_title and c_desc and c_title in df and c_desc in df:
        return (df[c_title].fillna("") + " " + df[c_desc].fillna("")).astype(str)

    if c_title and c_title in df:
        return df[c_title].fillna("").astype(str)

    if c_desc and c_desc in df:
        return df[c_desc].fillna("").astype(str)

    # fallback: everything empty -> empty strings (so the scorer won’t error)
    return pd.Series([""] * len(df), index=df.index)


def _pick_backend(args_backend: str | None) -> str:
    b = (args_backend or cfg.get("model", {}).get("sentiment_backend", "vader")).lower()
    return b


def _get_model_cfg(backend: str):
    m = cfg.get("model")
    if isinstance(m, dict):
        return m.get(backend)
    return None


def _score_in_chunks(texts: pd.Series, scorer, chunk: int = 32) -> List[float]:
    out: List[float] = []
    n = len(texts)
    for i in range(0, n, chunk):
        part = texts.iloc[i:i+chunk].tolist()
        try:
            scores = scorer(part)
        except Exception as e:
            # if scoring fails, fill zeros so the output still has a compound column
            warnings.warn(f"Scoring chunk failed: {e}")
            scores = [0.0] * len(part)
        if isinstance(scores, (list, tuple, np.ndarray)):
            out.extend([float(x) for x in scores])
        else:
            # if a scorer returns a single value (unusual), replicate it
            out.extend([float(scores)] * len(part))
    return out


# ------------ Label inference (BTC/ETH/SOL) -------------
_BTC_RE = re.compile(r"\b(bitcoin|\$?btc)\b", flags=re.IGNORECASE)
_ETH_RE = re.compile(r"\b(ethereum|\$?eth)\b", flags=re.IGNORECASE)
_SOL_RE = re.compile(r"\b(solana|\$?sol)\b", flags=re.IGNORECASE)

def _infer_label_from_text(txt: str) -> str:
    if not isinstance(txt, str) or not txt:
        return "OTHER"
    b = len(_BTC_RE.findall(txt))
    e = len(_ETH_RE.findall(txt))
    s = len(_SOL_RE.findall(txt))
    if (b, e, s) == (0, 0, 0):
        return "OTHER"
    # choose the highest count; tie-break: BTC > ETH > SOL
    if b >= e and b >= s:
        return "BTC"
    if e >= b and e >= s:
        return "ETH"
    return "SOL"

def _infer_labels(df: pd.DataFrame) -> pd.Series:
    cols = {c.lower(): c for c in df.columns}
    title = df[cols["title"]].fillna("").astype(str) if "title" in cols else ""
    desc  = df[cols["description"]].fillna("").astype(str) if "description" in cols else ""
    content = df[cols["content"]].fillna("").astype(str) if "content" in cols else ""
    text = (title + " " + desc + " " + content)
    return text.apply(_infer_label_from_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CSV input (raw)")
    ap.add_argument("--out", dest="out", required=True, help="CSV output (scored)")
    ap.add_argument("--backend", dest="backend", default=None, help="vader | finbert | roberta")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    _debug(f"Config loaded: {cfg}", args.debug)

    # 1) Load
    df = pd.read_csv(args.inp)
    _debug(f"Loaded {len(df)} rows", args.debug)

    # 2) Prepare text
    texts = _ensure_text_columns(df)
    _debug("Prepared text column for scoring", args.debug)

    # 3) Build scorer
    backend = _pick_backend(args.backend)
    model_cfg = _get_model_cfg(backend)

    dev = cfg.get("hardware", {}).get("device", "auto")
    try:
        if model_cfg and isinstance(model_cfg, dict):
            scorer = get_scorer(model_cfg, device=dev)
        else:
            scorer = get_scorer(backend, device=dev)
        _debug(f"Using backend: {backend} (device={dev})", args.debug)
    except Exception as e:
        warnings.warn(f"Failed to create backend {backend}, falling back to VADER. ({e})")
        scorer = get_scorer("vader")

    # 4) Score to 'compound'
    scores = _score_in_chunks(texts, scorer, chunk=32)
    df["compound"] = pd.Series(scores, index=df.index).clip(-1, 1)

    # 5) Write output (ensure minimal columns required by the UI)
    # normalize several important column names to keep the UI safe
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_pub = pick("publishedat", "published", "publishdate", "date", "datetime", "time")
    if c_pub and c_pub != "publishedat":
        df = df.rename(columns={c_pub: "publishedat"})
    if "publishedat" in df.columns:
        df["publishedat"] = pd.to_datetime(df["publishedat"], errors="coerce")

    c_src = pick("source", "domain", "site")
    if c_src and c_src != "source":
        df = df.rename(columns={c_src: "source"})
    if "source" not in df.columns:
        df["source"] = ""

    c_chn = pick("channel", "provider")
    if c_chn and c_chn != "channel":
        df = df.rename(columns={c_chn: "channel"})
    if "channel" not in df.columns:
        df["channel"] = "newsapi"

    # label (can be empty; we will fill it)
    if "label" not in df.columns:
        df["label"] = ""

    # ---- NEW: infer BTC/ETH/SOL label for rows with empty labels ----
    try:
        empty_mask = df["label"].astype(str).str.strip().eq("") | df["label"].isna()
    except Exception:
        empty_mask = pd.Series([True] * len(df), index=df.index)
    if empty_mask.any():
        inferred = _infer_labels(df.loc[empty_mask].copy())
        df.loc[empty_mask, "label"] = inferred.values

    # final label normalization
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df.loc[~df["label"].isin(["BTC", "ETH", "SOL", "OTHER"]), "label"] = "OTHER"

    # url/title/description are optional
    for want in ["title", "description", "url"]:
        if want not in df.columns:
            df[want] = ""

    # 6) Save
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
