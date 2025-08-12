# ============================================
# Crypto Sentiment Dashboard Machine Learning
# Author: wiqilee
# License: MIT
# ============================================
# sentiment.py — core utils: load_data, takeaways, PDF builder (clean & localized)

from __future__ import annotations
from io import BytesIO
from typing import Dict, Any
import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

# ----------------- data loader -----------------
def load_data(path: str) -> pd.DataFrame:
    """
    Minimal CSV loader used by the UI:
      - Reads a CSV
      - Normalizes 'publishedat' to pandas datetime if present
    """
    df = pd.read_csv(path)
    if "publishedat" in df.columns:
        df["publishedat"] = pd.to_datetime(df["publishedat"], errors="coerce")
    return df

# ----------------- simple takeaways (customize if needed) -----------------
def derive_crypto_takeaways(dff: pd.DataFrame) -> Dict[str, str]:
    """
    Returns short, human-readable insights for the UI.
    Keys expected by the app:
      - insight_text, analysis_md, recommendations_md, conclusion_md
    """
    n = len(dff)
    if n == 0:
        return dict(
            insight_text="No data.",
            analysis_md="- Data is empty.",
            recommendations_md="- Re-run the fetcher to pull fresh news.",
            conclusion_md="No conclusion available."
        )
    pos = (dff["compound"] > 0.05).mean() if "compound" in dff.columns else 0.0
    bullets = []
    if pos > 0.5:
        bullets.append("Overall tone skews positive across recent headlines.")
    elif pos < 0.3:
        bullets.append("Tone leans neutral-to-negative; risk remains elevated.")
    else:
        bullets.append("Tone is mixed; catalysts likely vary by asset.")
    return dict(
        insight_text=" ".join(bullets),
        analysis_md="\n".join(f"- {s}" for s in bullets),
        recommendations_md="- Use multiple sources and compare model outputs.\n- Treat sentiment as context, not a standalone trading signal.",
        conclusion_md="Models tend to agree on strong polarity; disagreements often appear on nuanced or ironic wording."
    )

# ----------------- PDF builder (clean layout + guidance table) -----------------
def _img(path: str, maxw: float, maxh: float):
    """
    Safely load an image and scale it to fit within (maxw, maxh)
    while preserving aspect ratio. Returns a Platypus Image or None.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        im = RLImage(path)
        w, h = im.wrap(0, 0)
        k = min(maxw / w, maxh / h, 1.0)
        im.drawWidth = w * k
        im.drawHeight = h * k
        return im
    except Exception:
        return None

def _mk_para(txt: str, style):
    """
    Convert plain text to a Paragraph, replacing newlines with <br/>
    so that long text renders correctly inside table cells.
    """
    if not txt:
        txt = "-"
    txt = "<br/>".join(str(txt).splitlines())
    return Paragraph(txt, style)

def build_pdf_report(dff: pd.DataFrame, agg_tbl: pd.DataFrame, pngs: Dict[str, str], meta: Dict[str, Any]) -> bytes:
    """
    Build a print-ready PDF for the Single or Compare view (the caller aligns columns):
      - dff: filtered frame shown in the UI (Single: has 'compound')
      - agg_tbl: aggregated stats (e.g., mean by label)
      - pngs: dict of figure paths (by_source, by_label, ts_label, heatmap, backend_compare?)
      - meta: misc info (counters, date range, text sections, localized labels)

    Returns:
      Raw PDF bytes.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.5*cm,
        rightMargin=1.5*cm,
        topMargin=1.2*cm,
        bottomMargin=1.2*cm
    )

    styles = getSampleStyleSheet()
    H1 = styles["Heading1"]
    H2 = styles["Heading2"]
    H3 = styles["Heading3"]
    P  = styles["BodyText"]
    Psmall = ParagraphStyle("Psmall", parent=P, fontSize=9, leading=12)

    story = []

    # --- Title ---
    title = meta.get("title", "Crypto Sentiment Report")
    story.append(Paragraph(title, H1))
    story.append(Paragraph(f"{meta.get('generated_label','Generated')}: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}", P))
    story.append(Spacer(1, 0.3*cm))

    # --- Top metrics table ---
    n  = meta.get("n", len(dff))
    avg = meta.get("avg", float("nan"))
    med = meta.get("median", float("nan"))
    std = meta.get("std", float("nan"))
    pos = meta.get("pos_n", 0)
    neu = meta.get("neu_n", 0)
    neg = meta.get("neg_n", 0)
    dfile = meta.get("data_file", "-")
    drng  = meta.get("date_range", "-")

    tdata = [
        [meta.get("rows_label","Rows"), str(n),
         meta.get("datafile_label","Data file"), dfile],
        [meta.get("range_label","Date range"), drng,
         "Avg / Median / Std", f"{avg:.3f} / {med:.3f} / {std:.3f}"],
        ["+ / 0 / −", f"{pos} / {neu} / {neg}", "", ""]
    ]
    t = Table(tdata, colWidths=[3.5*cm, 8*cm, 3.5*cm, 4.0*cm])
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story += [t, Spacer(1, 0.35*cm)]

    # --- Charts (if present) ---
    def add_fig(title_key: str, p: str):
        im = _img(p, maxw=17.5*cm, maxh=10.5*cm)
        if im:
            story.append(Paragraph(meta.get(title_key, title_key), H2))
            story.append(im)
            story.append(Spacer(1, 0.25*cm))

    add_fig("src_chart_title", pngs.get("by_source"))
    add_fig("lbl_chart_title", pngs.get("by_label"))
    add_fig("ts_chart_title",  pngs.get("ts_label"))
    add_fig("hm_chart_title",  pngs.get("heatmap"))
    # Optional: compare-specific figure
    if pngs.get("backend_compare"):
        add_fig("cmp_chart_title", pngs.get("backend_compare"))

    # --- Guidance table (How to read / Model notes / Notes / Recommendations / Conclusion) ---
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Guidance", H2))

    rows = []
    def add_row(title, body):
        if body and str(body).strip():
            rows.append([Paragraph(f"<b>{title}</b>", P), _mk_para(str(body), Psmall)])

    add_row(meta.get("readme_title", "How to read the data"),
            meta.get("readme_md", ""))

    add_row(meta.get("model_notes_title", "Model notes"),
            meta.get("model_notes_md", ""))

    # Keep “analysis” distinct from “how to read” and “model notes”
    add_row(meta.get("analysis_title", "Notes"),
            meta.get("analysis_md", ""))

    add_row(meta.get("reco_title", "Recommendations"),
            meta.get("recommendations_md", ""))

    if meta.get("conclusion_md"):
        add_row(meta.get("conclusion_title", "Conclusion"),
                meta.get("conclusion_md", ""))

    if rows:
        gt = Table(rows, colWidths=[5.0*cm, 12.0*cm])
        gt.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ]))
        story += [gt]

    doc.build(story)
    return buf.getvalue()
