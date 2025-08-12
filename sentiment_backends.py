# ============================================
# Crypto Sentiment Dashboard Machine Learning
# Author: wiqilee
# License: MIT
# ============================================
# sentiment_backends.py — pluggable sentiment scorers (VADER / FinBERT / RoBERTa)

from __future__ import annotations
from typing import Callable, List, Union, Optional, Dict

TextIn = Union[str, List[str]]
ScoresOut = Union[float, List[float]]

def _ensure_list(x: TextIn) -> List[str]:
    return [x] if isinstance(x, str) else list(x)

# -------- device resolver (avoid 'auto' issues) --------
def _resolve_device(device: Optional[str]) -> str:
    if not device:
        return "cpu"
    d = str(device).lower().strip()
    if d in {"auto", "default"}:
        return "cpu"
    # allow 'cpu', 'cuda:0', 'mps'
    if d.startswith(("cpu", "cuda", "mps")):
        return d
    return "cpu"

# ---------------- VADER ----------------
def _vader_scorer() -> Callable[[TextIn], ScoresOut]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    s = SentimentIntensityAnalyzer()
    def run(texts: TextIn) -> ScoresOut:
        arr = _ensure_list(texts)
        out = [s.polarity_scores(t or "")["compound"] for t in arr]
        return out[0] if isinstance(texts, str) else out
    return run

# ---------------- HF helpers ----------------
def _hf_pipeline(model_id: str, device: str = "cpu"):
    from transformers import pipeline
    dev = _resolve_device(device)
    return pipeline(
        task="text-classification",
        model=model_id,
        tokenizer=model_id,
        device=dev,            # 'cpu' | 'cuda:0' | 'mps'
        truncation=True
    )

def _hf_to_compound(label: str, score: float) -> float:
    l = (label or "").lower()
    if "pos" in l:
        return +float(score)
    if "neg" in l:
        return -float(score)
    return 0.0

# ---------------- FinBERT ----------------
def _finbert_scorer(model_id: str, device: str = "cpu") -> Callable[[TextIn], ScoresOut]:
    nlp = _hf_pipeline(model_id, device=device)
    def run(texts: TextIn) -> ScoresOut:
        arr = _ensure_list(texts)
        out = nlp(arr)
        def one(o):
            if isinstance(o, list): o = o[0]
            return _hf_to_compound(o.get("label", ""), o.get("score", 0.0))
        res = [one(o) for o in out]
        return res[0] if isinstance(texts, str) else res
    return run

# ---------------- RoBERTa (financial) ----------------
def _roberta_scorer(model_id: str, device: str = "cpu") -> Callable[[TextIn], ScoresOut]:
    nlp = _hf_pipeline(model_id, device=device)
    def run(texts: TextIn) -> ScoresOut:
        arr = _ensure_list(texts)
        out = nlp(arr)
        def one(o):
            if isinstance(o, list): o = o[0]
            return _hf_to_compound(o.get("label", ""), o.get("score", 0.0))
        res = [one(o) for o in out]
        return res[0] if isinstance(texts, str) else res
    return run

# ---------------- Public API ----------------
def get_scorer(name_or_cfg: Union[str, Dict], device: Optional[str] = "cpu") -> Callable[[TextIn], ScoresOut]:
    """
    Build a callable scorer(texts) -> list[compound] (or single float for single string).

    Parameters
    ----------
    name_or_cfg:
      - string: "vader" | "finbert" | "roberta"
      - dict: e.g. {"type":"transformers","pretrained_model":"Shaivn/Financial-Sentiment-Analysis"}
    device:
      - "cpu" | "cuda:0" | "mps" | "auto" (mapped to "cpu")

    Returns
    -------
    Callable that accepts str | list[str] and returns float | list[float]
    """
    dev = _resolve_device(device)

    if isinstance(name_or_cfg, dict):
        t = (name_or_cfg.get("type") or "").lower()
        model_id = name_or_cfg.get("pretrained_model") or name_or_cfg.get("model") or ""
        nm = (name_or_cfg.get("name") or model_id).lower()
        if t == "lexicon" or "vader" in nm:
            return _vader_scorer()
        if not model_id:
            raise ValueError("Missing 'pretrained_model' for transformers backend.")
        if "finbert" in nm or "financial-sentiment" in model_id.lower():
            return _finbert_scorer(model_id, device=dev)
        return _roberta_scorer(model_id, device=dev)

    key = (name_or_cfg or "").lower()
    if key == "vader":
        return _vader_scorer()
    if key == "finbert":
        # safe default (safetensors)
        return _finbert_scorer("Shaivn/Financial-Sentiment-Analysis", device=dev)
    if key == "roberta":
        return _roberta_scorer("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", device=dev)
    raise ValueError(f"Unknown backend: {name_or_cfg!r}")
