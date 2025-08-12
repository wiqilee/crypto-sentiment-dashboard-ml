# ============================================
# Crypto Sentiment Dashboard Machine Learning
# Author: wiqilee
# License: MIT
# ============================================
# SPDX-License-Identifier: MIT
# fetchers.py — Fetch News via NewsAPI (key from [sources] or NEWSAPI_KEY env)
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pandas as pd
import requests
from config import cfg

UA = "CryptoSentimentDashboard/1.0 (+https://github.com/wiqilee)"
DEFAULT_TIMEOUT = 20
RETRY_STATUS = {429, 500, 502, 503, 504}

def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 8.0) -> None:
    """Exponential backoff sleep helper."""
    time.sleep(min(cap, base * (2 ** (attempt - 1))))

def _requests_get(url: str, *, params=None, timeout=DEFAULT_TIMEOUT, max_retries=3) -> requests.Response:
    """GET with simple retry/backoff on transient errors."""
    attempt = 1
    while True:
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": UA})
        except requests.RequestException:
            if attempt >= max_retries: raise
            _sleep_backoff(attempt); attempt += 1; continue
        if r.status_code in RETRY_STATUS and attempt < max_retries:
            _sleep_backoff(attempt); attempt += 1; continue
        return r

def _to_iso_utc(dt: datetime) -> str:
    """Format a datetime as ISO 8601 in UTC (YYYY-MM-DDTHH:MM:SSZ)."""
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    else: dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_newsapi(api_key: str, query: str, *, days=7, language="en", page_size=50,
                  search_in="title,description", debug=False) -> pd.DataFrame:
    """
    Fetch news from NewsAPI /v2/everything with simple pagination & rate-limit handling.
    Returns a DataFrame with core fields expected by downstream scripts.
    """
    if not api_key:
        raise RuntimeError("NEWSAPI key is empty. Set [sources].newsapi_key or export NEWSAPI_KEY.")
    url = "https://newsapi.org/v2/everything"
    MAX_TOTAL = 100
    now_utc = datetime.now(timezone.utc)
    from_date = now_utc - timedelta(days=int(days or 7))
    rows, fetched, page = [], 0, 1
    while fetched < MAX_TOTAL:
        size = min(int(page_size), MAX_TOTAL - fetched)
        params = {
            "q": query, "from": _to_iso_utc(from_date), "to": _to_iso_utc(now_utc),
            "language": language, "sortBy": "publishedAt", "searchIn": search_in,
            "pageSize": size, "page": page, "apiKey": api_key,
        }
        r = _requests_get(url, params=params)
        if r.status_code != 200:
            if r.status_code == 426:
                if debug: print("[DEBUG] NewsAPI 426 - upgrade required; stop paging.")
                break
            raise RuntimeError(f"Error fetching NewsAPI: {r.status_code} - {r.text}")
        arts = (r.json().get("articles") or [])
        for a in arts:
            rows.append({
                "source": (a.get("source") or {}).get("name"),
                "author": a.get("author"),
                "title": a.get("title") or "",
                "description": a.get("description") or "",
                "url": a.get("url"),
                "publishedAt": a.get("publishedAt"),
                "channel": "newsapi",
            })
        fetched += len(arts)
        if debug: print(f"[DEBUG] page={page} got={len(arts)} fetched_total={fetched}")
        if len(arts) < size: break
        page += 1; time.sleep(1.0)
    df = pd.DataFrame(rows)
    if not df.empty and "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df

# (optional) CoinGecko utilities can be copied from the legacy version if needed.

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output CSV path (e.g., data/news_raw.csv)")
    p.add_argument("--debug", action="store_true", help="Enable verbose logs")
    args = p.parse_args()

    SRC = cfg.get("sources", {}) or {}
    use_newsapi = bool(SRC.get("use_newsapi", cfg.get("use_newsapi", True)))
    api_key = os.environ.get("NEWSAPI_KEY") or str(SRC.get("newsapi_key", cfg.get("newsapi_key", "")))
    query = str(SRC.get("query", cfg.get("query", "")))
    language = str(SRC.get("language", cfg.get("language", "en")))
    days = int(SRC.get("days", cfg.get("days", 7)))
    page_size = int(SRC.get("page_size", cfg.get("page_size", 50)))

    if args.debug:
        print("Fetching started...")
        print("Config loaded:", cfg)
        print("NewsAPI key present?", bool(api_key))

    df_all = pd.DataFrame()
    if use_newsapi:
        df_news = fetch_newsapi(api_key=api_key, query=query, days=days,
                                language=language, page_size=page_size, debug=args.debug)
        df_all = pd.concat([df_all, df_news], ignore_index=True)

    for col in ["source","title","description","publishedAt","url","channel"]:
        if col not in df_all.columns: df_all[col] = ""

    df_all.to_csv(args.out, index=False)
    print(f"Saved {len(df_all)} rows to {args.out}")

if __name__ == "__main__":
    main()
