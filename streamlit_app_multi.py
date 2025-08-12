# ============================================
# Crypto Sentiment Dashboard Machine Learning
# Author: wiqilee
# License: MIT
# ============================================
# streamlit_app.py — Single (VADER) & Compare (VADER+FinBERT+RoBERTa)

from __future__ import annotations
import os, sys, subprocess, math
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from sentiment import (
    load_data as core_load_data,
    derive_crypto_takeaways,
    build_pdf_report as core_build_pdf,
)

try:
    import fetchers
except Exception:
    fetchers = None

def load_cfg():
    try:
        import tomllib
        with open("config.toml", "rb") as f:
            return tomllib.load(f)
    except Exception:
        try:
            import toml
            return toml.load("config.toml")
        except Exception:
            return {}

CFG = load_cfg()
DATA_CFG = (CFG.get("data", {}) or {})
SRC_CFG  = (CFG.get("sources", {}) or {})
DATA_CSV   = DATA_CFG.get("dataset_csv", "data/dataset.csv")
DATA_MULTI = DATA_CFG.get("dataset_multi_csv", "data/dataset_multi.csv")
RAW_CSV    = DATA_CFG.get("raw_news_csv", "data/news_raw.csv")
CHART_DIR  = DATA_CFG.get("chart_dir", "charts")
os.makedirs(os.path.dirname(RAW_CSV) or ".", exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

st.set_page_config(
    page_title="Crypto Sentiment Dashboard",
    layout="wide",
    page_icon=":chart_with_upwards_trend:",
    initial_sidebar_state="expanded",
)

LABEL_COLORS = {"BTC":"#22c55e","ETH":"#ffffff","SOL":"#fde047","OTHER":"#94a3b8"}
LABEL_ORDER = ["BTC","ETH","SOL","OTHER"]
SOURCE_COLORS = px.colors.qualitative.Set3 + px.colors.qualitative.Set2
LEGEND_TOP = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, bgcolor="rgba(0,0,0,0)")

def to_utc_naive(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize("UTC")
    except Exception:
        dt = pd.to_datetime(series, errors="coerce").dt.tz_localize("UTC")
    return dt.dt.tz_convert("UTC").dt.tz_localize(None)

def fmtdt(ts) -> str:
    if pd.isna(ts): return "-"
    try: return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception: return str(ts)

def chip(label: str) -> str:
    color = LABEL_COLORS.get(label, "#9ca3af")
    dot = f"<span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:{color};margin-right:6px;vertical-align:middle'></span>"
    return f"{dot}{label}"

def _pad_range(xmin: float, xmax: float) -> tuple[float, float]:
    span = xmax - xmin if np.isfinite(xmax - xmin) else 2.0
    pad = max(0.08, 0.06 * span)
    return xmin - pad, xmax + pad

def _force_rgb_png(path_png: str):
    try:
        im = Image.open(path_png)
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(path_png, format="PNG", optimize=True)
    except Exception:
        pass

def ensure_png(fig: go.Figure, path_png: str, scale: float = 3.0):
    try:
        f = go.Figure(fig)
        f.update_layout(width=1300, height=780, margin=dict(l=130,r=110,t=110,b=110),
                        paper_bgcolor="white", plot_bgcolor="white")
        f.update_traces(cliponaxis=False, opacity=1.0)
        f.write_image(path_png, scale=scale)
        _force_rgb_png(path_png)
        return True
    except Exception:
        return False

def polarity_of(x: float, pos_thr: float, neg_thr: float) -> str | float:
    if pd.isna(x): return np.nan
    if x > pos_thr: return "positive"
    if x < neg_thr: return "negative"
    return "neutral"

def polarity_counts(series: pd.Series, pos_thr: float, neg_thr: float) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    return dict(
        positive=int((s >  pos_thr).sum()),
        neutral =int(((s >= neg_thr) & (s <= pos_thr)).sum()),
        negative=int((s <  neg_thr).sum()),
    )

def percent(x: int, n: int) -> float:
    return 0.0 if n == 0 else 100.0 * x / n

def _cfg_get(key: str, default=None):
    if key in SRC_CFG: return SRC_CFG.get(key)
    return CFG.get(key, default)

def run_fetch_news(api_key: str, query: str, language: str, days: int, page_size: int):
    if fetchers is None:
        st.error("Modul `fetchers.py` tidak ditemukan.")
        return
    if not api_key:
        st.error("NEWSAPI key kosong.")
        return
    with st.spinner("Fetching NewsAPI..."):
        df = fetchers.fetch_newsapi(
            api_key=api_key, query=query,
            days=days, language=language, page_size=page_size, debug=True
        )
        df.to_csv(RAW_CSV, index=False)
    st.success(f"Saved {len(df)} rows to {RAW_CSV}")

def run_build_single(backend: str):
    with st.spinner(f"Building dataset (single={backend}) ..."):
        cmd = [sys.executable, "build_dataset.py", "--in", RAW_CSV, "--out", DATA_CSV, "--backend", backend, "--debug"]
        res = subprocess.run(cmd, capture_output=True, text=True)
    st.code(res.stdout or "", language="bash")
    if res.returncode != 0:
        st.error(res.stderr or "build_dataset.py failed")
    else:
        st.success(f"Built → {DATA_CSV}")

def run_build_multi():
    with st.spinner("Building dataset (compare: VADER/FinBERT/RoBERTa) ..."):
        cmd = [sys.executable, "build_dataset_multi.py", "--in", RAW_CSV, "--out", DATA_MULTI, "--debug"]
        res = subprocess.run(cmd, capture_output=True, text=True)
    st.code(res.stdout or "", language="bash")
    if res.returncode != 0:
        st.error(res.stderr or "build_dataset_multi.py failed")
    else:
        st.success(f"Built → {DATA_MULTI}")

@st.cache_data(show_spinner=False)
def load_df_single(path: str) -> pd.DataFrame:
    df = core_load_data(path)
    df = df.copy()
    df["publishedat_norm"] = to_utc_naive(df["publishedat"])
    return df

def bar_sentiment_by_source(dff: pd.DataFrame):
    agg = dff.groupby("source", dropna=False)["compound"].mean().reset_index().sort_values("compound")
    cmap = {src: SOURCE_COLORS[i % len(SOURCE_COLORS)] for i, src in enumerate(agg["source"])}
    fig = px.bar(
        agg, x="compound", y="source", orientation="h",
        labels={"compound": "Avg VADER compound", "source": "Source"},
        text=agg["compound"].round(3).astype(str),
        color="source", color_discrete_map=cmap
    )
    fig.update_traces(textposition="outside", textfont_size=12, marker_line_color="#0f172a",
                      marker_line_width=0.6, showlegend=False, cliponaxis=False)
    if len(agg):
        xmin, xmax = float(agg["compound"].min()), float(agg["compound"].max())
        xmin, xmax = _pad_range(xmin, xmax)
        fig.update_xaxes(range=[xmin, xmax], automargin=True, zeroline=True, zerolinewidth=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title_standoff=18), yaxis=dict(title_standoff=22))
    return fig

def bar_sentiment_by_label(dff: pd.DataFrame):
    agg = dff.groupby("label", dropna=False)["compound"].mean().reset_index()
    agg["__ord"] = agg["label"].apply(lambda x: LABEL_ORDER.index(x) if x in LABEL_ORDER else 999)
    agg = agg.sort_values("__ord").drop(columns="__ord")
    vals = agg["compound"].astype(float).values
    text_pos = ["inside" if v < 0 else "outside" for v in vals]
    text_colors = ["#111111" if (v < 0 and lab in {"SOL", "OTHER"}) else None
                   for lab, v in zip(agg["label"], vals)]
    fig = px.bar(
        agg, x="compound", y="label", orientation="h",
        labels={"compound": "Avg VADER compound", "label": "Label"},
        text=agg["compound"].round(3).astype(str),
        color="label", color_discrete_map=LABEL_COLORS,
    )
    fig.update_traces(textposition=text_pos, textfont_size=12, marker_line_color="#0f172a",
                      marker_line_width=1.0, showlegend=False, cliponaxis=False)
    for i, c in enumerate(text_colors):
        if c:
            fig.data[i].textfont = dict(color=c, size=12)
    xmin, xmax = (float(np.nanmin(vals)) if len(vals) else -1.0,
                  float(np.nanmax(vals)) if len(vals) else  1.0)
    xmin, xmax = _pad_range(xmin, xmax)
    fig.update_xaxes(range=[xmin, xmax], automargin=True, zeroline=True, zerolinewidth=1)
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title_standoff=18), yaxis=dict(title_standoff=22))
    return fig, agg

def _apply_explicit_colors(fig: go.Figure, line_colors: dict, theme_for_pdf: bool):
    for tr in fig.data:
        name = getattr(tr, "name", None)
        if not name:
            continue
        col = line_colors.get(name, "#222222")
        dash = "dot" if name == "ETH" else "solid"
        outline = "#000000" if theme_for_pdf else "#0f172a"
        tr.update(mode="lines+markers",
                  line=dict(color=col, width=4.5, dash=dash),
                  marker=dict(color=col, size=9, line=dict(width=1.2, color=outline)),
                  opacity=1.0)

def timeseries_by_label(dff: pd.DataFrame, theme_for_pdf: bool=False, show_end_labels: bool=False):
    if dff.empty:
        return px.line()
    ts = dff.copy()
    if "publishedat_norm" not in ts.columns:
        ts["publishedat_norm"] = to_utc_naive(ts["publishedat"])
    ts["date"] = pd.to_datetime(ts["publishedat_norm"]).dt.floor("D")
    g = ts.groupby(["date", "label"], dropna=False)["compound"].mean().reset_index()

    line_colors = LABEL_COLORS.copy()
    if theme_for_pdf:
        line_colors["ETH"] = "#111111"

    fig = px.line(g, x="date", y="compound", color="label",
                  color_discrete_map=line_colors, markers=True,
                  labels={"date":"Date","compound":"Avg VADER compound","label":"Label"})
    _apply_explicit_colors(fig, line_colors, theme_for_pdf)

    if show_end_labels:
        last_vals = g.sort_values("date").groupby("label").tail(1)
        yshift = {"BTC": 16, "ETH": -18, "SOL": 16, "OTHER": -10}
        for _, row in last_vals.iterrows():
            fig.add_annotation(x=row["date"], y=row["compound"],
                               text=f"{row['label']} {row['compound']:.3f}",
                               showarrow=False, yshift=yshift.get(row["label"], 0),
                               font=dict(color="#e5e7eb" if not theme_for_pdf else "#111111", size=12))
    if not g.empty:
        dmin, dmax = pd.to_datetime(g["date"].min()), pd.to_datetime(g["date"].max())
        span = (dmax - dmin) if pd.notna(dmax) and pd.notna(dmin) else pd.Timedelta(days=1)
        pad = max(pd.Timedelta(hours=6), span * 0.03)
        if theme_for_pdf:
            rmin, rmax = dmin - pad, dmax + pad
            fig.update_xaxes(type="date", range=[rmin, rmax], automargin=True,
                             title_standoff=20, tickformat="%b %d, %Y")
        else:
            fig.update_xaxes(type="date", range=[dmin, dmax], constrain="domain", automargin=True,
                             title_standoff=20, tickformat="%b %d, %Y")
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=10), legend=LEGEND_TOP,
                      yaxis=dict(range=[-1.05,1.05], title_standoff=26),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def heatmap_corr_by_source(dff: pd.DataFrame, theme_for_pdf: bool=False):
    tmp = dff.copy()
    if "publishedat_norm" not in tmp.columns:
        tmp["publishedat_norm"] = to_utc_naive(tmp["publishedat"])
    tmp["date"] = pd.to_datetime(tmp["publishedat_norm"]).dt.date
    daily = tmp.groupby(["date", "source"], dropna=False)["compound"].mean().reset_index()
    pivot = daily.pivot(index="date", columns="source", values="compound")

    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        text_color = "#e5e7eb" if not theme_for_pdf else "#111111"
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                           text="Not enough data for correlation\n(≥3 days & ≥2 sources)",
                           showarrow=False, font=dict(color=text_color, size=16), align="center",
                           bgcolor="rgba(0,0,0,0)" if not theme_for_pdf else "rgba(255,255,255,0)")
        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        fig.update_layout(height=420, margin=dict(l=80, r=80, t=60, b=80),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    corr = pivot.corr(min_periods=3)
    fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    labels=dict(color="corr"), aspect="auto")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title_standoff=18), yaxis=dict(title_standoff=18),
                      legend=LEGEND_TOP)
    return fig

def backend_correlations(dff: pd.DataFrame) -> pd.DataFrame:
    nums = dff[["compound_vader", "compound_finbert", "compound_roberta"]].dropna()
    if nums.empty:
        return pd.DataFrame(columns=["pair", "pearson_r", "mad"])
    r_vf = nums["compound_vader"].corr(nums["compound_finbert"])
    r_vr = nums["compound_vader"].corr(nums["compound_roberta"])
    r_fr = nums["compound_finbert"].corr(nums["compound_roberta"])
    mad_vf = (nums["compound_vader"] - nums["compound_finbert"]).abs().mean()
    mad_vr = (nums["compound_vader"] - nums["compound_roberta"]).abs().mean()
    mad_fr = (nums["compound_finbert"] - nums["compound_roberta"]).abs().mean()
    return pd.DataFrame([
        {"pair":"VADER–FinBERT", "pearson_r": r_vf, "mad": mad_vf},
        {"pair":"VADER–RoBERTa", "pearson_r": r_vr, "mad": mad_vr},
        {"pair":"FinBERT–RoBERTa", "pearson_r": r_fr, "mad": mad_fr},
    ])

def agreement_rate(dff: pd.DataFrame, pos_thr: float, neg_thr: float) -> dict:
    tmp = dff.dropna(subset=["compound_vader","compound_finbert","compound_roberta"]).copy()
    if tmp.empty:
        return dict(agree_n=0, total=0, rate=0.0)
    pv = tmp["compound_vader"].apply(lambda x: polarity_of(x, pos_thr, neg_thr))
    pf = tmp["compound_finbert"].apply(lambda x: polarity_of(x, pos_thr, neg_thr))
    pr = tmp["compound_roberta"].apply(lambda x: polarity_of(x, pos_thr, neg_thr))
    agree_mask = (pv == pf) & (pf == pr)
    total = len(tmp)
    agree_n = int(agree_mask.sum())
    return dict(agree_n=agree_n, total=total, rate=percent(agree_n, total))

def combined_score(row, w_v=0.2, w_f=0.4, w_r=0.4):
    vals, ws = [], []
    if not pd.isna(row.get("compound_vader")):   vals.append(row["compound_vader"]);   ws.append(w_v)
    if not pd.isna(row.get("compound_finbert")): vals.append(row["compound_finbert"]); ws.append(w_f)
    if not pd.isna(row.get("compound_roberta")): vals.append(row["compound_roberta"]); ws.append(w_r)
    if not vals: return np.nan
    wsum = sum(ws) if sum(ws) > 0 else 1.0
    return float(np.dot(vals, ws) / wsum)

with st.sidebar:
    st.header("Mode")
    mode = st.radio("View", ["Single (VADER)", "Compare (VADER + FinBERT + RoBERTa)"], index=0)

    st.header("Data")
    st.caption(f"Raw CSV: `{RAW_CSV}`")
    default_key = (os.getenv("NEWSAPI_KEY") or SRC_CFG.get("newsapi_key", "")).strip()
    api_key = st.text_input("NewsAPI key", type="password", value=default_key)
    query_in = st.text_input("Query", value=str(_cfg_get("query", "(bitcoin OR ethereum OR crypto OR blockchain OR BTC OR ETH OR SOL)")))
    lang_in  = st.selectbox("Language", ["en","id","es","de","fr","it","pt","ru","ja","zh"], index=0)
    days_in  = st.slider("Days back", 1, 30, int(_cfg_get("days", 7)))
    psize_in = st.slider("Page size", 20, 100, int(_cfg_get("page_size", 50)))

    colA, colB = st.columns(2)
    with colA:
        if st.button("Fetch News (NewsAPI)", use_container_width=True):
            run_fetch_news(api_key, query_in, lang_in, days_in, psize_in)
    with colB:
        if st.button("Open raw CSV", use_container_width=True):
            if os.path.exists(RAW_CSV):
                st.dataframe(pd.read_csv(RAW_CSV).head(50))
            else:
                st.warning("Raw CSV belum ada. Klik Fetch dulu.")

    st.caption(f"Single dataset: `{DATA_CSV}` | Compare dataset: `{DATA_MULTI}`")
    backend_sel = st.selectbox("Backend (single)", ["vader", "finbert", "roberta"], index=2)
    colC, colD = st.columns(2)
    with colC:
        if st.button("Build dataset (Single)", use_container_width=True):
            run_build_single(backend_sel)
    with colD:
        if st.button("Build dataset (Compare)", use_container_width=True):
            run_build_multi()

    st.header("Polarity thresholds")
    neg_thr = st.slider("Negative threshold", -0.40, -0.00, -0.05, step=0.01)
    pos_thr = st.slider("Positive threshold",  0.00,  0.40,  0.05, step=0.01)
    if pos_thr <= 0 or neg_thr >= 0:
        st.warning("Rekomendasi: neg_thr < 0 < pos_thr (mis. −0.05 dan +0.05).")

    st.header("Actions")
    if st.button("Refresh UI cache", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

# =============== SINGLE MODE ===============
if mode.startswith("Single"):
    df_raw = load_df_single(DATA_CSV)

    with st.sidebar.expander("Filters (single)", expanded=True):
        labels_all = LABEL_ORDER
        labels_sel = st.multiselect("Filter by label", labels_all, default=labels_all)
        sources_all = sorted(df_raw["source"].astype(str).unique().tolist())
        sources_sel = st.multiselect("Filter by source", sources_all, default=sources_all)
        min_dt = df_raw["publishedat_norm"].min()
        max_dt = df_raw["publishedat_norm"].max()
        date_from, date_to = st.date_input("Date range (UTC)", value=(min_dt.date(), max_dt.date()))
        if isinstance(date_from, tuple):
            date_from, date_to = date_from
        show_end_ann = st.checkbox("Show end annotations (time-series)", value=True)
        show_index   = st.checkbox("Show table row index (UI)", value=False)
        add_polarity = st.checkbox("Add polarity column to table", value=True)

    cA, cB, cC = st.columns([0.7, 0.2, 0.1])
    with cA: st.title("Crypto Sentiment Dashboard — Single (VADER)")
    with cB: st.caption(f"Rows: {len(df_raw)} | File: `{DATA_CSV}`")
    with cC: st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    dff = df_raw.copy()
    if labels_sel:  dff = dff[dff["label"].isin(labels_sel)]
    if sources_sel: dff = dff[dff["source"].isin(sources_sel)]
    if date_from and date_to:
        pub_utc = pd.to_datetime(dff["publishedat_norm"])
        mask_dt = (pub_utc.dt.date >= pd.to_datetime(date_from).date()) & \
                  (pub_utc.dt.date <= pd.to_datetime(date_to).date())
        dff = dff[mask_dt]

    rng_min = dff["publishedat_norm"].min() if not dff.empty else None
    rng_max = dff["publishedat_norm"].max() if not dff.empty else None
    st.caption(f"{len(dff)} results after filters (UTC {fmtdt(rng_min)} ➜ {fmtdt(rng_max)})")

    reading_single = (
        "- **compound**: VADER sentiment score in the range −1..+1 (negative..positive). 0 ≈ neutral.\n"
        "- **label**: inferred using simple keywords (BTC/ETH/SOL). If none match, it falls back to OTHER.\n"
        f"- **Positive/Neutral/Negative** use thresholds: negative < {neg_thr:.2f}, neutral in [{neg_thr:.2f}..{pos_thr:.2f}], positive > {pos_thr:.2f}.\n"
        "- **by Source** = average score per outlet; **by Label** = average per label.\n"
        "- **Time-series** = daily means (above 0 skews positive).\n"
        "- **Heatmap** = correlation across sources (needs ≥3 days & ≥2 sources)."
    )
    with st.expander("ℹ️ How to read the data (Single)"):
        st.markdown(reading_single)

    st.subheader("📰 Latest Crypto News")
    show_cols = ["publishedat", "channel", "source", "label", "title", "url", "compound"]
    dshow = dff[show_cols].copy()
    if add_polarity:
        dshow["polarity"] = dshow["compound"].apply(lambda x: polarity_of(x, pos_thr, neg_thr))
    try: dshow["label"] = dshow["label"].apply(chip)
    except Exception: pass
    dshow["url"] = dshow["url"].apply(lambda u: f"<a href='{u}' target='_blank'>open</a>" if isinstance(u, str) and u.startswith("http") else "")
    styler = (dshow.style.format(na_rep="").set_table_styles([
        dict(selector="th", props=[("text-align","left")]),
        dict(selector="td", props=[("text-align","left"), ("vertical-align","top")]),
    ]))
    try: styler = styler.hide_index()
    except Exception:
        try: styler = styler.hide(axis="index")
        except Exception: pass
    st.markdown(styler.to_html(escape=False), unsafe_allow_html=True)

    st.subheader("🙂 Polarity breakdown (VADER)")
    pc = polarity_counts(dff["compound"], pos_thr, neg_thr)
    m1, m2, m3 = st.columns(3)
    m1.metric("Positive", pc["positive"]); m2.metric("Neutral",  pc["neutral"]); m3.metric("Negative", pc["negative"])

    st.subheader("📊 Sentiment Score by News Source")
    fig_src = bar_sentiment_by_source(dff); st.plotly_chart(fig_src, use_container_width=True)
    ensure_png(fig_src, os.path.join(CHART_DIR, "single_sentiment_by_source.png"))

    st.subheader("📊 Sentiment by Label (colored)")
    fig_lbl, agg_lbl = bar_sentiment_by_label(dff); st.plotly_chart(fig_lbl, use_container_width=True)
    ensure_png(fig_lbl, os.path.join(CHART_DIR, "single_sentiment_by_label.png"))

    st.subheader("⏱️ Time-series by Label (daily mean)")
    fig_ts_ui = timeseries_by_label(dff, theme_for_pdf=False, show_end_labels=st.checkbox("Show end labels", value=True))
    st.plotly_chart(fig_ts_ui, use_container_width=True)
    fig_ts_pdf = timeseries_by_label(dff, theme_for_pdf=True, show_end_labels=True)
    ensure_png(fig_ts_pdf, os.path.join(CHART_DIR, "single_timeseries_by_label.png"))

    st.subheader("🧩 Source Correlation Heatmap (daily mean)")
    fig_hm_ui = heatmap_corr_by_source(dff, theme_for_pdf=False); st.plotly_chart(fig_hm_ui, use_container_width=True)
    fig_hm_pdf = heatmap_corr_by_source(dff, theme_for_pdf=True)
    ensure_png(fig_hm_pdf, os.path.join(CHART_DIR, "single_source_corr_heatmap.png"))

    st.subheader("🧾 Summary (Research-ready)")
    n = len(dff)
    avg = float(dff["compound"].mean()) if n else float("nan")
    median = float(dff["compound"].median()) if n else float("nan")
    std = float(dff["compound"].std()) if n else float("nan")
    pos_n = int((dff["compound"] >  pos_thr).sum())
    neu_n = int(((dff["compound"] >= neg_thr) & (dff["compound"] <= pos_thr)).sum())
    neg_n = int((dff["compound"] <  neg_thr).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles (n)", f"{n}")
    c2.metric("Avg / Median", f"{avg:.3f} / {median:.3f}" if n else "nan / nan")
    c3.metric("Std dev", f"{std:.3f}" if n else "nan")
    c4.metric("Positive / Neutral / Negative", f"{pos_n} / {neu_n} / {neg_n}",
              help=f"Thresholds: pos > {pos_thr:.2f}, neg < {neg_thr:.2f}")

    agg_tbl = dff.groupby("label", dropna=False)["compound"].mean().reset_index().rename(columns={"compound":"avg_compound"})
    agg_tbl["avg_compound"] = agg_tbl["avg_compound"].round(3)
    st.markdown("**Per-label (mean compound)**")
    st.dataframe(agg_tbl.sort_values("label"), use_container_width=True, hide_index=True)

    take = derive_crypto_takeaways(dff)
    expert_single = (
        "- **Use sentiment as context**, not as a standalone trading signal.\n"
        "- **Strong positive spikes** for a label (e.g., BTC) should be cross-checked against price/volume and on-chain data.\n"
        "- **Source bias**: some outlets tend to be more bullish/bearish — see the by-Source chart."
    )
    st.subheader("🧠 Expert view (single)"); st.markdown(expert_single)
    st.subheader("🔎 Crypto analysis"); st.markdown(take["analysis_md"])
    st.subheader("🧭 Recommendations"); st.markdown(take["recommendations_md"])
    st.subheader("✅ Conclusion");     st.write(take["conclusion_md"])

    st.subheader("⬇️ Downloads")
    csv_bytes = dff.drop(columns=["publishedat_norm"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv_bytes, "news_filtered.csv", "text/csv", use_container_width=True)

    meta = {
        "n": n, "avg": avg, "median": median, "std": std,
        "pos_n": pos_n, "pos_pct": percent(pos_n,n),
        "neu_n": neu_n, "neu_pct": percent(neu_n,n),
        "neg_n": neg_n, "neg_pct": percent(neg_n,n),
        "pos_thr": float(pos_thr), "neg_thr": float(neg_thr),
        "data_file": DATA_CSV,
        "date_range": f"{fmtdt(rng_min)} – {fmtdt(rng_max)}" if n else "-",
        "insight_text": take.get("insight_text", ""),
        "analysis_md": take.get("analysis_md", ""),
        "recommendations_md": take.get("recommendations_md", ""),
        "conclusion_md": take.get("conclusion_md", ""),
        "reading_md": reading_single,
    }
    pngs_single = {
        "by_source": os.path.join(CHART_DIR, "single_sentiment_by_source.png"),
        "by_label":  os.path.join(CHART_DIR, "single_sentiment_by_label.png"),
        "ts_label":  os.path.join(CHART_DIR, "single_timeseries_by_label.png"),
        "heatmap":   os.path.join(CHART_DIR, "single_source_corr_heatmap.png"),
    }
    if st.button("Build PDF report (single)", type="primary", use_container_width=True):
        try:
            pdf_bytes = core_build_pdf(dff.drop(columns=["publishedat_norm"], errors="ignore"),
                                       agg_tbl, pngs_single, meta)
            st.download_button("Download PDF (single)", pdf_bytes,
                               "crypto_sentiment_report.pdf", "application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"PDF build failed: {e}")

# =============== COMPARE MODE ===============
else:
    st.title("Crypto Sentiment Dashboard — Compare (VADER + FinBERT + RoBERTa)")
    if not os.path.exists(DATA_MULTI):
        st.error(f"File not found: {DATA_MULTI}. Bangun dulu dengan tombol Build dataset (Compare) di sidebar.")
        st.stop()

    dfm = pd.read_csv(DATA_MULTI)
    for c in ["publishedat","source","label","title","url"]:
        if c not in dfm.columns: dfm[c] = ""
    for c in ["compound_vader","compound_finbert","compound_roberta"]:
        if c not in dfm.columns: dfm[c] = None
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

    with st.sidebar.expander("Filters (compare)", expanded=True):
        src_all = sorted(dfm["source"].astype(str).unique().tolist())
        src_sel = st.multiselect("Source", src_all, default=src_all)
        lbl_all = sorted([x for x in dfm["label"].dropna().astype(str).unique().tolist() if x])
        lbl_sel = st.multiselect("Label", lbl_all, default=lbl_all) if lbl_all else []
        pub_all = pd.to_datetime(dfm["publishedat"], errors="coerce", utc=True)
        try:
            dt_min = pub_all.min().date(); dt_max = pub_all.max().date()
        except Exception:
            today = datetime.utcnow().date()
            dt_min = dt_max = today
        date_from, date_to = st.date_input("Date range (UTC)", value=(dt_min, dt_max))
        only_complete = st.checkbox("Only rows with all 3 scores", value=False)
        add_polarity_cols = st.checkbox("Add per-backend polarity columns to table", value=True)
        drop_nan_scores = st.checkbox("Drop rows with any NaN score", value=False)
        show_combined = st.checkbox("Show combined score (0.2*VADER + 0.4*FinBERT + 0.4*RoBERTa)", value=True)

    dff = dfm.copy()
    if src_sel: dff = dff[dff["source"].isin(src_sel)]
    if lbl_sel: dff = dff[dff["label"].isin(lbl_sel)]
    pub = pd.to_datetime(dff["publishedat"], errors="coerce", utc=True)
    mask = (pub.dt.date >= pd.to_datetime(date_from).date()) & (pub.dt.date <= pd.to_datetime(date_to).date())
    dff = dff[mask]
    if only_complete:
        dff = dff.dropna(subset=["compound_vader", "compound_finbert", "compound_roberta"])
    if drop_nan_scores:
        dff = dff.dropna(subset=["compound_vader", "compound_finbert", "compound_roberta"])

    st.caption(f"{len(dff)} results • file: `{DATA_MULTI}`")

    reading_compare = (
        "- **We compare 3 backends** on the same set of articles:\n"
        "  - **VADER**: lexicon-based — fast, good for short headlines; can fail on negation/irony.\n"
        "  - **FinBERT**: BERT fine-tuned on financial news — stable in finance contexts.\n"
        "  - **RoBERTa (financial)**: usually similar to FinBERT, sometimes more aggressive.\n"
        "- **NaN** = missing score (e.g., build interrupted). Check *Only rows with all 3 scores* to keep complete rows only.\n"
        f"- **Positive/Neutral/Negative** use thresholds: negative < {neg_thr:.2f}, neutral in [{neg_thr:.2f}..{pos_thr:.2f}], positive > {pos_thr:.2f}.\n"
        "- **Correlation** & **agreement** are proxies for consistency: higher correlation and higher agreement → more reliable."
    )
    with st.expander("ℹ️ How to read the data (Compare)"):
        st.markdown(reading_compare)

    st.subheader("🙂 Polarity breakdown per backend")
    c_v = polarity_counts(dff["compound_vader"], pos_thr, neg_thr)
    c_f = polarity_counts(dff["compound_finbert"], pos_thr, neg_thr)
    c_r = polarity_counts(dff["compound_roberta"], pos_thr, neg_thr)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("**VADER**")
        st.metric("Positive", c_v["positive"]); st.metric("Neutral", c_v["neutral"]); st.metric("Negative", c_v["negative"])
    with m2:
        st.markdown("**FinBERT**")
        st.metric("Positive", c_f["positive"]); st.metric("Neutral", c_f["neutral"]); st.metric("Negative", c_f["negative"])
    with m3:
        st.markdown("**RoBERTa**")
        st.metric("Positive", c_r["positive"]); st.metric("Neutral", c_r["neutral"]); st.metric("Negative", c_r["negative"])

    cnt = pd.DataFrame([
        {"backend":"VADER",   **c_v},
        {"backend":"FinBERT", **c_f},
        {"backend":"RoBERTa", **c_r},
    ])
    melt = cnt.melt(id_vars="backend", value_vars=["positive","neutral","negative"], var_name="polarity", value_name="n")
    fig_stack = px.bar(melt, x="backend", y="n", color="polarity", barmode="stack",
                       color_discrete_map={"positive":"#22c55e","neutral":"#a3a3a3","negative":"#ef4444"},
                       title="Polarity counts (by thresholds)")
    fig_stack.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_stack, use_container_width=True)

    st.subheader("📰 Table (all backends)")
    tcols = ["publishedat","source","label","title","url","compound_vader","compound_finbert","compound_roberta"]
    tbl = dff[tcols].copy()
    if add_polarity_cols:
        tbl["pol_vader"]   = tbl["compound_vader"].apply(lambda x: polarity_of(x, pos_thr, neg_thr))
        tbl["pol_finbert"] = tbl["compound_finbert"].apply(lambda x: polarity_of(x, pos_thr, neg_thr))
        tbl["pol_roberta"] = tbl["compound_roberta"].apply(lambda x: polarity_of(x, pos_thr, neg_thr))
    if show_combined:
        tbl["combined_0.2V_0.4F_0.4R"] = dff.apply(combined_score, axis=1)
        tbl["combined_polarity"] = tbl["combined_0.2V_0.4F_0.4R"].apply(lambda x: polarity_of(x, pos_thr, neg_thr))
    tbl["url"] = tbl["url"].apply(lambda u: f"<a href='{u}' target='_blank'>open</a>" if isinstance(u, str) and u.startswith("http") else "")
    st.markdown(tbl.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.caption("Note: NaN = rows missing a score for a given backend.")

    st.subheader("📊 Mean sentiment by backend")
    agg = pd.DataFrame({
        "backend": ["vader", "finbert", "roberta"],
        "mean": [
            dff["compound_vader"].mean(),
            dff["compound_finbert"].mean(),
            dff["compound_roberta"].mean(),
        ],
    }).round(4)
    fig_bar = px.bar(agg, x="backend", y="mean", text="mean",
                     color="backend",
                     color_discrete_map={"vader": "#60a5fa", "finbert": "#34d399", "roberta": "#fbbf24"},
                     labels={"backend": "Backend", "mean": "Mean compound"})
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(yaxis=dict(range=[-1,1]), height=420, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("🔎 Pairwise comparisons")
    c1, c2 = st.columns(2)
    with c1:
        fig_pair_FR = px.scatter(dff, x="compound_finbert", y="compound_roberta",
                                 labels={"compound_finbert":"FinBERT", "compound_roberta":"RoBERTa"},
                                 opacity=0.75)
        fig_pair_FR.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1, line=dict(width=1, dash="dot", color="#444"))
        fig_pair_FR.update_xaxes(range=[-1, 1]); fig_pair_FR.update_yaxes(range=[-1, 1])
        fig_pair_FR.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pair_FR, use_container_width=True)
    with c2:
        fig_pair_VR = px.scatter(dff, x="compound_vader", y="compound_roberta",
                                 color="source", opacity=0.75,
                                 labels={"compound_vader":"VADER", "compound_roberta":"RoBERTa"})
        fig_pair_VR.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1, line=dict(width=1, dash="dot", color="#444"))
        fig_pair_VR.update_xaxes(range=[-1, 1]); fig_pair_VR.update_yaxes(range=[-1, 1])
        fig_pair_VR.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20),
                                  legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_pair_VR, use_container_width=True)

    st.subheader("⏱️ Daily mean by backend")
    tmp = dff.copy()
    tmp["date"] = to_utc_naive(tmp["publishedat"]).dt.date
    daily = pd.DataFrame({
        "date": tmp["date"],
        "VADER": tmp["compound_vader"],
        "FinBERT": tmp["compound_finbert"],
        "RoBERTa": tmp["compound_roberta"],
    }).groupby("date", as_index=False).mean(numeric_only=True)
    fig_ts = go.Figure()
    for col, color in [("VADER","#60a5fa"),("FinBERT","#34d399"),("RoBERTa","#fbbf24")]:
        fig_ts.add_trace(go.Scatter(x=daily["date"], y=daily[col], mode="lines+markers", name=col, line=dict(width=3, color=color)))
    fig_ts.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20),
                         yaxis=dict(range=[-1, 1], title="Mean compound"),
                         xaxis_title="Date", legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("🧪 Evaluasi ketepatan (proxy)")
    corr_tbl = backend_correlations(dff).round(3)
    agree = agreement_rate(dff, pos_thr, neg_thr)
    colx, coly = st.columns([0.55, 0.45])
    with colx:
        st.markdown("**Correlation & MAD (mean absolute difference)** — higher r & lower MAD → more consistent.")
        st.dataframe(corr_tbl, hide_index=True, use_container_width=True)
    with coly:
        st.metric("Agreement (same polarity across all backends)", f"{agree['agree_n']} / {agree['total']}")
        st.metric("Agreement rate", f"{agree['rate']:.1f}%")

    expert_compare = (
        "- **FinBERT & RoBERTa** are generally more suitable for financial-news context.\n"
        "- **VADER** is lightweight and fast; great for short headlines but sensitive to negation/irony.\n"
        "- **Practice**: consider a **weighted blend** (0.2·VADER + 0.4·FinBERT + 0.4·RoBERTa). "
        "If any is NaN, average the available ones.\n"
        "- Low **disagreement** = stronger signal; high disagreement → read the article (it may be ambiguous or out-of-domain)."
    )
    st.subheader("🧠 Expert view (compare)")
    st.markdown(expert_compare)

    st.subheader("⬇️ Downloads")
    csv_bytes = dff.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV (multi)", csv_bytes, "news_multi_filtered.csv", "text/csv", use_container_width=True)

    # ===== Save PNGs for PDF =====
    pngs_multi = {
        "mean_bar":             os.path.join(CHART_DIR, "multi_mean_by_backend.png"),
        "pair_finbert_roberta": os.path.join(CHART_DIR, "multi_pair_finbert_roberta.png"),
        "pair_vader_roberta":   os.path.join(CHART_DIR, "multi_pair_vader_roberta.png"),
        "daily_mean":           os.path.join(CHART_DIR, "multi_daily_mean.png"),
    }
    ensure_png(fig_bar,      pngs_multi["mean_bar"])
    ensure_png(fig_pair_FR,  pngs_multi["pair_finbert_roberta"])
    ensure_png(fig_pair_VR,  pngs_multi["pair_vader_roberta"])
    ensure_png(fig_ts,       pngs_multi["daily_mean"])

    # ===== PDF builder (compare) — with reading & analysis =====
    def build_pdf_multi(png_paths: dict, meta: dict) -> bytes:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        from reportlab.lib import colors
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.4*cm, bottomMargin=1.2*cm)
        ss = getSampleStyleSheet()
        H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceAfter=8)
        H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceAfter=6)
        P  = ParagraphStyle("P", parent=ss["BodyText"], fontSize=9, leading=12)
        Small = ParagraphStyle("Small", parent=P, fontSize=8, textColor=colors.gray)

        s = []
        s.append(Paragraph("Crypto Sentiment — Multi-backend Report", H1))
        s.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
            f"Data file: {meta.get('data_file','-')}<br/>"
            f"Rows (filtered): {meta.get('n',0)} | Date range: {meta.get('date_range','-')}",
            Small
        ))
        s.append(Spacer(1, 6))

        # How to read
        if meta.get("reading_md"):
            s.append(Paragraph("<b>How to read</b>", H2))
            s.append(Paragraph(meta["reading_md"].replace("\n","<br/>"), P))
            s.append(Spacer(1, 6))

        # Summary (means + agreement)
        k = [
            ["Mean VADER",   f"{meta.get('mean_vader','—'):.3f}" if not math.isnan(meta.get('mean_vader', float('nan'))) else "—"],
            ["Mean FinBERT", f"{meta.get('mean_finbert','—'):.3f}" if not math.isnan(meta.get('mean_finbert', float('nan'))) else "—"],
            ["Mean RoBERTa", f"{meta.get('mean_roberta','—'):.3f}" if not math.isnan(meta.get('mean_roberta', float('nan'))) else "—"],
            ["Agreement rate", f"{meta.get('agree_rate','—'):.1f}%"],
        ]
        t = Table(k, colWidths=[6.0*cm, 8.3*cm])
        t.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
            ("ALIGN",(0,0),(0,-1),"RIGHT"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        s.append(t); s.append(Spacer(1, 6))

        # Thresholds & counts
        k2 = [
            ["Polarity thresholds", f"negative < {meta.get('neg_thr',-0.05):.2f}, neutral in [{meta.get('neg_thr',-0.05):.2f}..{meta.get('pos_thr',0.05):.2f}], positive > {meta.get('pos_thr',0.05):.2f}"],
            ["VADER  (pos/neu/neg)", f"{meta.get('cnt_v_pos',0)} / {meta.get('cnt_v_neu',0)} / {meta.get('cnt_v_neg',0)}"],
            ["FinBERT(pos/neu/neg)", f"{meta.get('cnt_f_pos',0)} / {meta.get('cnt_f_neu',0)} / {meta.get('cnt_f_neg',0)}"],
            ["RoBERTa(pos/neu/neg)", f"{meta.get('cnt_r_pos',0)} / {meta.get('cnt_r_neu',0)} / {meta.get('cnt_r_neg',0)}"],
        ]
        t2 = Table(k2, colWidths=[6.0*cm, 8.3*cm])
        t2.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
            ("ALIGN",(0,0),(0,-1),"RIGHT"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        s.append(t2); s.append(Spacer(1, 6))

        # Analysis
        if meta.get("analysis_md"):
            s.append(Paragraph("<b>Expert analysis</b>", H2))
            s.append(Paragraph(meta["analysis_md"].replace("\n","<br/>"), P))
            s.append(Spacer(1, 8))

        # Charts
        for title, key in [
            ("Mean sentiment by backend", "mean_bar"),
            ("Pairwise: FinBERT vs RoBERTa", "pair_finbert_roberta"),
            ("Pairwise: VADER vs RoBERTa", "pair_vader_roberta"),
            ("Daily mean by backend", "daily_mean"),
        ]:
            p = png_paths.get(key)
            if p and os.path.exists(p):
                s.append(Paragraph(f"<b>{title}</b>", H2))
                s.append(RLImage(p, width=16.5*cm, height=9.9*cm))
                s.append(Spacer(1, 8))

        doc.build(s)
        return buf.getvalue()

    agree_meta = agreement_rate(dff, pos_thr, neg_thr)
    analysis_compare_pdf = (
        f"- **Agreement**: {agree_meta['rate']:.1f}% — higher means the backends agree more.\n"
        f"- **Means**: VADER={dff['compound_vader'].mean():.3f}, FinBERT={dff['compound_finbert'].mean():.3f}, RoBERTa={dff['compound_roberta'].mean():.3f}.\n"
        "- Higher correlation & lower MAD between FinBERT–RoBERTa indicates good consistency in financial context.\n"
        "- If daily lines converge across all three backends, cross-model sentiment is strong; if they diverge, read the source article."
    )
    meta = {
        "n": len(dff),
        "mean_vader": float(dff["compound_vader"].mean()),
        "mean_finbert": float(dff["compound_finbert"].mean()),
        "mean_roberta": float(dff["compound_roberta"].mean()),
        "agree_rate": float(agree_meta["rate"]),
        "data_file": DATA_MULTI,
        "date_range": (
            f"{to_utc_naive(dff['publishedat']).min()} – {to_utc_naive(dff['publishedat']).max()}"
            if len(dff) else "-"
        ),
        "reading_md": reading_compare,
        "analysis_md": expert_compare + "\n\n" + analysis_compare_pdf,
        "pos_thr": float(pos_thr), "neg_thr": float(neg_thr),
        "cnt_v_pos": c_v["positive"], "cnt_v_neu": c_v["neutral"], "cnt_v_neg": c_v["negative"],
        "cnt_f_pos": c_f["positive"], "cnt_f_neu": c_f["neutral"], "cnt_f_neg": c_f["negative"],
        "cnt_r_pos": c_r["positive"], "cnt_r_neu": c_r["neutral"], "cnt_r_neg": c_r["negative"],
    }
    if st.button("Build PDF report (compare)", type="primary", use_container_width=True):
        try:
            pdf_bytes = build_pdf_multi(pngs_multi, meta)
            st.download_button("Download PDF (compare)", pdf_bytes,
                               "crypto_sentiment_multi_report.pdf", "application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"PDF build failed: {e}")
