# ============================================
# Crypto Sentiment Dashboard Machine Learning
# Author: wiqilee
# License: MIT
# ============================================
# build_dataset_multi.py — build a multi-backend dataset: VADER, FinBERT, RoBERTa

from __future__ import annotations
import argparse
from datetime import timezone
import pandas as pd
import numpy as np

from config import cfg
from sentiment_backends import get_scorer

# ---------- keyword heuristics for labels ----------
_DEFAULT_KW = {
    "BTC": ["bitcoin","btc","satoshi","ordinals","inscriptions","taproot","lightning","halving"],
    "ETH": ["ethereum","eth","vitalik","beacon","merge","rollup","l2","optimism","arbitrum",
            "starknet","zksync","blob","dencun","base"],
    "SOL": ["solana","sol","raydium","jupiter","pump.fun","phantom","saga phone","helium","marinade"]
}
def _kw() -> dict[str, list[str]]:
    raw = (cfg.get("keywords") or {})
    out = {}
    for k in ("BTC","ETH","SOL"):
        base = _DEFAULT_KW.get(k, [])
        ext  = [str(x).lower() for x in (raw.get(k) or [])]
        out[k] = sorted(list(dict.fromkeys([*base, *ext])))
    return out
KW = _kw()

def _norm_label(x: str) -> str:
    """Normalize raw label to {BTC, ETH, SOL, OTHER}."""
    if not isinstance(x, str) or not x.strip(): return "OTHER"
    u = x.strip().upper()
    return u if u in {"BTC","ETH","SOL","OTHER"} else "OTHER"

def _guess_label(text: str) -> str:
    """Guess label from simple keyword counts in the provided text."""
    if not isinstance(text, str) or not text.strip(): return "OTHER"
    t = text.lower()
    scores = {sym: sum(1 for w in words if w and w in t) for sym, words in KW.items()}
    scores = {k:v for k,v in scores.items() if v}
    if not scores: return "OTHER"
    # Tie-break on a simple priority (BTC > ETH > SOL)
    priority = {"BTC":3,"ETH":2,"SOL":1}
    best = sorted(scores.items(), key=lambda kv: (kv[1], priority[kv[0]]), reverse=True)[0][0]
    return best

# ---------- helpers ----------
def _first_nonempty(*vals):
    """Return the first non-empty string among vals, otherwise empty string."""
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
    return ""

def _load_input(path: str) -> pd.DataFrame:
    """Load raw CSV and normalize core columns required downstream."""
    df = pd.read_csv(path)
    # normalize timestamp column -> 'publishedat' (UTC ISO)
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("publishedat") or cols.get("published") or cols.get("date")
    if ts_col:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_convert(timezone.utc).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        ts = pd.Series([""], index=df.index)
    out = pd.DataFrame({
        "publishedat": ts,
        "channel": df.get("channel","newsapi"),
        "source": df.get("source",""),
        "title": df.get("title",""),
        "description": df.get("description",""),
        "url": df.get("url",""),
    })
    # labels: use existing if provided; if empty/OTHER, guess from title+source
    if "label" in df.columns:
        out["label"] = df["label"].map(_norm_label)
    else:
        out["label"] = "OTHER"
    mask_other = out["label"].eq("OTHER")
    guess_text = (out["title"].fillna("") + "  " + out["source"].fillna(""))
    out.loc[mask_other, "label"] = [ _guess_label(t) for t in guess_text[mask_other] ]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Raw CSV input (e.g., news_raw.csv)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV (dataset_multi.csv)")
    ap.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = ap.parse_args()

    df = _load_input(args.inp)

    # prepare scorers
    mcfg = cfg.get("model", {})
    vader = get_scorer(mcfg.get("vader", "vader"), device="cpu")
    finbert_cfg = mcfg.get("finbert") or {"type":"transformers","pretrained_model":"Shaivn/Financial-Sentiment-Analysis"}
    finbert = get_scorer(finbert_cfg, device="cpu")
    roberta_cfg = mcfg.get("roberta") or {"type":"transformers","pretrained_model":"mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"}
    roberta = get_scorer(roberta_cfg, device="cpu")

    # text to score (title > description > source)
    text = [
        _first_nonempty(t, d, s)
        for t, d, s in zip(df["title"].astype(str), df["description"].astype(str), df["source"].astype(str))
    ]

    # scores
    df["compound_vader"]   = pd.to_numeric(vader(text), errors="coerce")
    df["compound_finbert"] = pd.to_numeric(finbert(text), errors="coerce")
    df["compound_roberta"] = pd.to_numeric(roberta(text), errors="coerce")

    # select/order & save
    cols = ["publishedat","channel","source","label","title","url","compound_vader","compound_finbert","compound_roberta"]
    df = df[cols]
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
