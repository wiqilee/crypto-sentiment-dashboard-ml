# ============================================
# Crypto Sentiment Dashboard Machine Learning
# Author: wiqilee
# License: MIT
# ============================================
# config.py — loader TOML yang kompatibel dengan modul lama & baru

# SPDX-License-Identifier: MIT
# config.py — TOML loader compatible with old & new modules
from __future__ import annotations
import os

# --- load TOML (Py3.11+ = tomllib; else fall back to toml) ---
try:
    import tomllib  # Python 3.11+
    def _read_toml():
        with open("config.toml", "rb") as f:
            return tomllib.load(f)
except Exception:
    import toml
    def _read_toml():
        return toml.load("config.toml")

_raw = _read_toml() if os.path.exists("config.toml") else {}

# --- flatten: move [sources] keys to top-level so fetchers.py & older scripts keep working ---
cfg = dict(_raw) if isinstance(_raw, dict) else {}
sources = (cfg.pop("sources", {}) or {})
for k, v in sources.items():
    # top-level wins if there is a duplicate
    cfg.setdefault(k, v)

# --- override NEWSAPI_KEY from environment if present ---
env_key = os.getenv("NEWSAPI_KEY") or os.getenv("NEWS_API_KEY")
if env_key:
    cfg["newsapi_key"] = env_key

# Ensure commonly accessed blocks exist
cfg.setdefault("data", {})
cfg.setdefault("model", {})
cfg.setdefault("keywords", _raw.get("keywords", {}))
cfg.setdefault("coingecko_ids", _raw.get("coingecko_ids", {}))
