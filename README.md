# ============================================
# Crypto Sentiment Dashboard Machine Learning
# Author: wiqilee
# License: MIT
# ============================================
# Crypto Sentiment Dashboard

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
[![Made by wiqilee](https://img.shields.io/badge/made%20by-wiqilee-000000.svg?logo=github)](https://github.com/wiqilee)

A Streamlit dashboard to analyze crypto news sentiment for **BTC / ETH / SOL / OTHER**.

- **Single mode (VADER)** and **Compare mode (VADER + FinBERT + RoBERTa)**
- Interactive charts (Plotly), export-ready PNGs (RGB-safe), and **PDF reports that mirror the UI** (filters, table preview, charts, metrics, glossary & methodology)

> **Disclaimer:** For research and monitoring only. This is **not** financial advice.

---

## Features

- **Two views**
  - **Single (VADER)** — fast lexicon scoring; labels synchronized with Compare.
  - **Compare (VADER + FinBERT + RoBERTa)** — side-by-side scores, agreement rate, and correlations.
- **Beginner-friendly explanations** in the UI for all metrics and charts.
- **PDF “same as UI”**
  - Header + active filters
  - Filtered table preview (truncated for print)
  - All core charts
  - Key metrics + glossary & methodology
- **High-quality exports** — Plotly → PNG with large margins and **forced RGB** (avoids faint lines in PDFs).
- Robust data loading: flexible column names and safe fallbacks.

---

## Quick Start

```bash
# 1) (optional) create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt

# 3) run the app
streamlit run streamlit_app.py
```

You can fetch & build data from the **left sidebar** (Fetch News, Build Single/Compare).  
Or do it from the CLI:

```bash
# Fetch raw news (NewsAPI) to data/news_raw.csv
python fetchers.py --out data/news_raw.csv --debug

# Build Single dataset (choose a backend: vader|finbert|roberta)
python build_dataset.py --in data/news_raw.csv --out data/dataset.csv --backend vader --debug

# Build Compare dataset (VADER + FinBERT + RoBERTa)
python build_dataset_multi.py --in data/news_raw.csv --out data/dataset_multi.csv --debug
```

---

## Configuration

Use **`config.example.ml.toml`** as a template and create your local `config.toml` (do **not** commit real keys).

Key fields:

- `newsapi_key` (or export `NEWSAPI_KEY` in your shell)
- `query`, `language`, `days`, `page_size`
- `[data]` paths (optional): `dataset_csv`, `dataset_multi_csv`, `raw_news_csv`, `chart_dir`
- `[model]` backends for `vader`, `finbert`, `roberta`
- `[keywords]` (optional) to extend BTC/ETH/SOL heuristics in builders

---

## Data Schema

### Single (`data/dataset.csv`)

Required (the app normalizes common variants):

- `publishedAt` / `published` / `date` → normalized to `publishedat` (UTC)
- `source` — outlet name/domain
- `compound` — sentiment in `[-1, +1]`

Optional: `channel`, `title`, `description`, `url`, `label`  
If `label` is empty, the builder infers **BTC/ETH/SOL** from text (otherwise **OTHER**).

**Example**
```csv
publishedat,source,compound,title,url,label
2025-08-01T10:30:00Z,CoinDesk,0.21,"BTC breaks range","https://example.com/article",BTC
```

### Compare (`data/dataset_multi.csv`)
```
publishedat,channel,source,label,title,url,
compound_vader,compound_finbert,compound_roberta
```
Each `compound_*` is in `[-1, +1]`.

---

## Charts & Reports

- **By Source** and **By Label** bar charts (Single)
- **Daily time-series** by label (Single) / by backend (Compare)
- **Source correlation heatmap** (Single)
- **Compare extras**: pairwise scatter, mean by backend, agreement rate, Pearson r & MAD table

**PDF export**

- **Single**: *Build PDF report (single)* → matches the UI (filters, table preview, charts, metrics, glossary).
- **Compare**: *Build PDF report (compare)* → table + all charts + metrics + glossary & methodology.
- Chart PNGs are saved to `charts/` with large margins and forced RGB.

---

## How to Read the Numbers (for non-experts)

- **Compound (VADER)**: −1 (negative) to +1 (positive); around 0 is neutral.  
- **Thresholds** (configurable): default `positive > +0.05`, `negative < −0.05`, else neutral.
- **By Source / By Label**: average sentiment per group.
- **Time-series**: daily averages; lines above 0 indicate net-positive sentiment.
- **Agreement (Compare)**: share of articles where all three backends agree (Pos/Neu/Neg). Higher is better.

---

## Project Structure

```
.
├─ streamlit_app.py           # Streamlit UI (Single & Compare) + PDF triggers
├─ build_dataset.py           # Single: score & infer labels (BTC/ETH/SOL/OTHER)
├─ build_dataset_multi.py     # Compare: VADER/FinBERT/RoBERTa (3 scores)
├─ fetchers.py                # NewsAPI fetcher (pagination + backoff)
├─ sentiment.py               # Loader, crypto takeaways, PDF helpers
├─ config.py                  # TOML loader with env overrides
├─ config.example.ml.toml     # example config (no secrets)
├─ config.toml                # local config (gitignored)
├─ requirements.txt
├─ data/                      # local datasets (gitignored)
├─ charts/                    # exported PNGs (gitignored)
└─ LICENSE
```

---

## Troubleshooting

- **Plotly export fails** → `pip install --upgrade kaleido`
- **PDF lines look faint** → handled by forced-RGB PNGs (ensure `Pillow` is installed).
- **Pandas “FutureWarning: downcasting in replace”** → harmless; the code uses a safe pattern. If you see this elsewhere, prefer `series.replace(...).infer_objects(copy=False)`.

---

## Contributing

Pull requests are welcome. Please include a clear description and screenshots for UI changes.

---

## License

**MIT** © 2025–present **wiqilee** — see [`LICENSE`](./LICENSE).
