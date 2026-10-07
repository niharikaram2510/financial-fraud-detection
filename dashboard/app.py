import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import streamlit as st

# ============================================================
# PROJECT SETUP
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detector import (
    analyze_raw_transaction,
    explain_transaction,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fraud Detection Console",
    page_icon="FD",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN TOKENS + CSS
# ============================================================
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:#0a0f1d;
    --panel:#0f1729;
    --panel-2:#131d33;
    --line:#22304b;
    --line-soft:#1a263d;
    --text:#eaf0f8;
    --text-2:#aab6c9;
    --text-3:#74839c;
    --teal:#27d3b2;
    --blue:#55b7ff;
    --violet:#8b7cf6;
    --amber:#f2b84b;
    --red:#f05d6c;
}

html, body, .stApp, [class*="css"] { font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp { background:var(--bg); color:var(--text); }
.block-container { max-width:1500px; padding:3.2rem 1.6rem 2rem; }
header { background:transparent !important; }
footer { visibility:hidden; }
.stDeployButton, [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"], [data-testid="stToolbarActions"] { display:none !important; }
hr { border-color:var(--line-soft) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background:#0c1324; border-right:1px solid var(--line-soft); }
[data-testid="stSidebar"][aria-expanded="true"] {
    width:250px !important; min-width:250px !important; max-width:250px !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top:1.4rem; }
.brand { font-size:22px; font-weight:700; letter-spacing:-.02em; color:var(--text); }
.brand small { display:block; margin-top:2px; font-size:12px; font-weight:500; color:var(--text-3); letter-spacing:0; }
.status-pill { display:flex; align-items:center; gap:8px; margin:14px 0 4px; padding:9px 12px;
    border:1px solid rgba(39,211,178,.28); background:rgba(39,211,178,.08); border-radius:8px;
    color:var(--teal); font-size:13px; font-weight:600; }
.status-pill i { width:8px; height:8px; border-radius:50%; background:var(--teal); box-shadow:0 0 0 4px rgba(39,211,178,.15); }
.side-label { font-size:12px; font-weight:600; color:var(--text-3); margin:22px 0 8px; }
.side-stat { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.side-stat .n { font-size:24px; font-weight:700; color:var(--text); font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
.side-stat .l { font-size:12px; color:var(--text-3); margin-top:1px; }
.progress-track { height:5px; border-radius:99px; background:var(--line-soft); margin-top:9px; overflow:hidden; }
.progress-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,var(--blue),var(--teal)); }

[data-testid="stSidebar"] .stButton button { min-height:42px; border-radius:8px; font-size:14px; font-weight:600; transition:all .15s; }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background:var(--teal); border:1px solid var(--teal); color:#04201a; }
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover { background:#3fe0c1; border-color:#3fe0c1; color:#04201a; }
[data-testid="stSidebar"] .stButton button[kind="secondary"] { background:transparent; border:1px solid var(--line); color:var(--text-2); }
[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover { border-color:var(--text-3); color:var(--text); }
[data-testid="stSlider"] [role="slider"] { background:var(--teal) !important; box-shadow:none !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] { color:var(--text-3); }

/* ---------- Sidebar open / close controls ---------- */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display:flex !important; visibility:visible !important; opacity:1 !important;
    z-index:999999; background:var(--panel); border:1px solid var(--line);
    border-radius:8px; color:var(--teal);
}
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button { background:transparent; border:0; color:var(--teal); }
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="collapsedControl"]:hover { border-color:var(--teal); }
[data-testid="stSidebarCollapseButton"] button { opacity:1 !important; color:var(--text-2); }
[data-testid="stSidebarCollapseButton"] button:hover { color:var(--teal); }

/* ---------- Header ---------- */
.topbar { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px; }
.page-title { font-size:30px; line-height:1.1; font-weight:700; letter-spacing:-.03em; color:var(--text); }
.page-subtitle { color:var(--text-3); font-size:14px; margin-top:6px; }
.demo-pill { display:flex; align-items:center; gap:7px; border:1px solid var(--line); background:var(--panel); color:var(--text-2);
    padding:6px 12px; border-radius:99px; font-size:12px; font-weight:600; }
.demo-pill i { width:7px; height:7px; border-radius:50%; background:var(--teal); }

.model-strip { display:flex; align-items:center; gap:10px; padding:10px 14px; border:1px solid var(--line);
    background:var(--panel); border-radius:8px; color:var(--text-3); font-size:13px; margin-bottom:14px; }
.model-strip b { color:var(--text); font-weight:600; }
.live-dot { width:8px; height:8px; border-radius:50%; background:var(--teal); box-shadow:0 0 0 4px rgba(39,211,178,.15); }
.model-divider { color:#3a4a67; }
.model-note { margin-left:auto; color:var(--text-3); }

/* ---------- KPI cards ---------- */
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px 12px; min-height:98px; position:relative; overflow:hidden; }
.kpi::before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--blue); }
.kpi.high::before { background:var(--red); }
.kpi.medium::before { background:var(--amber); }
.kpi.accent::before { background:var(--teal); }
.kpi-label { font-size:12px; font-weight:600; color:var(--text-3); }
.kpi-value { font-size:30px; font-weight:700; color:var(--text); line-height:1.15; margin-top:6px; letter-spacing:-.03em; font-variant-numeric:tabular-nums; }
.kpi-foot { color:var(--text-3); font-size:12px; margin-top:2px; }

/* ---------- Chart cards (real bordered containers) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] { background:var(--panel); border:1px solid var(--line) !important; border-radius:10px; }
.section-title { color:var(--text); font-size:14px; font-weight:600; margin:2px 0 6px; }
.section-title span { color:var(--text-3); font-weight:400; margin-left:8px; font-size:12px; }
.empty-chart { height:170px; display:flex; align-items:center; justify-content:center; text-align:center;
    border:1px dashed var(--line); background:rgba(255,255,255,.01); color:var(--text-3); border-radius:8px; font-size:13px; padding:0 16px; }

/* ---------- Attention queue ---------- */
.queue-header { display:flex; justify-content:space-between; align-items:baseline; margin:26px 0 8px; }
.queue-header .t { color:var(--text); font-size:16px; font-weight:600; }
.queue-header .s { color:var(--text-3); font-size:12px; }
.queue-empty { border:1px solid var(--line); background:var(--panel); color:var(--text-3); border-radius:10px; padding:18px; font-size:13px; text-align:center; }
.queue-wrap { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.queue-row { display:grid; grid-template-columns:72px 120px 1fr 1.3fr 100px 90px; gap:10px; align-items:center; padding:11px 16px; border-bottom:1px solid var(--line-soft); font-size:14px; }
.queue-row:last-child { border-bottom:0; }
.queue-row:not(.queue-head):hover { background:var(--panel-2); }
.queue-head { background:var(--panel-2); color:var(--text-3); font-size:12px; font-weight:600; padding:9px 16px; }
.queue-cell { color:var(--text-2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-variant-numeric:tabular-nums; }
.queue-id { color:var(--text); font-weight:600; }
.prob-cell { display:flex; align-items:center; gap:10px; }
.prob-bar { flex:1; height:5px; background:var(--line-soft); border-radius:99px; overflow:hidden; }
.prob-bar i { display:block; height:100%; border-radius:99px; }
.prob-bar.high i { background:var(--red); } .prob-bar.medium i { background:var(--amber); }
.risk-badge { display:inline-block; min-width:62px; text-align:center; padding:3px 10px; border-radius:99px; font-size:11px; font-weight:700; letter-spacing:.03em; }
.risk-high { color:#ff8d98; background:rgba(240,93,108,.12); border:1px solid rgba(240,93,108,.35); }
.risk-medium { color:#ffd17a; background:rgba(242,184,75,.11); border:1px solid rgba(242,184,75,.32); }
.risk-low { color:#6ee7c8; background:rgba(39,211,178,.10); border:1px solid rgba(39,211,178,.30); }

/* ---------- Investigation ---------- */
.investigation-card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 16px; }
.investigation-label { color:var(--text-3); font-size:12px; font-weight:600; margin-bottom:4px; }
.investigation-value { color:var(--text); font-size:22px; font-weight:700; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }
.detail-card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; min-height:96px; height:100%; }
.detail-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
.detail-grid span { display:block; color:var(--text-3); font-size:12px; font-weight:600; margin-bottom:4px; }
.detail-grid b { color:var(--text); font-size:15px; font-weight:600; }
.reason-title { color:var(--teal); font-size:12px; font-weight:700; margin-bottom:6px; }
.detail-card ul { margin:0 0 0 18px; padding:0; color:var(--text-2); font-size:13.5px; line-height:1.5; }
.detail-card li { margin:2px 0; }
.muted { color:var(--text-3); font-size:13px; }

/* ---------- Widgets ---------- */
div[data-baseweb="select"] > div { background:var(--panel) !important; border-color:var(--line) !important; border-radius:8px; }
[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:10px; margin-top:16px; }
[data-testid="stExpander"] summary { font-weight:600; color:var(--text-2); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:8px; }
[data-testid="stCaptionContainer"] { color:var(--text-3); margin-top:12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
DEMO_PATH = PROJECT_ROOT / "data" / "processed" / "demo_transactions.csv"


@st.cache_data
def load_demo_data():
    if not DEMO_PATH.exists():
        return None
    return pd.read_csv(DEMO_PATH)


demo_df = load_demo_data()

if demo_df is None:
    st.error("Demo dataset not found. Expected: data/processed/demo_transactions.csv")
    st.stop()

# ============================================================
# SESSION STATE
# ============================================================
if "processed_transactions" not in st.session_state:
    st.session_state.processed_transactions = []
if "stream_index" not in st.session_state:
    st.session_state.stream_index = 0
if "selected_transaction" not in st.session_state:
    st.session_state.selected_transaction = None

# ============================================================
# SIDEBAR: controls only
# ============================================================
total_demo = len(demo_df)
done = len(st.session_state.processed_transactions)
progress_pct = min(100, (st.session_state.stream_index / total_demo) * 100) if total_demo else 0

st.sidebar.markdown(
    '<div class="brand">FDx // Risk<small>Transaction monitoring</small></div>'
    '<div class="status-pill"><i></i>System ready</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="side-label">Dataset</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div class="side-stat"><div class="n">{total_demo:,}</div>'
    f'<div class="l">demo transactions &middot; {st.session_state.stream_index:,} streamed</div>'
    f'<div class="progress-track"><div class="progress-fill" style="width:{progress_pct:.1f}%"></div></div></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="side-label">Monitoring controls</div>', unsafe_allow_html=True)
process_count = st.sidebar.slider("Transactions per batch", 1, 25, 5)

if st.sidebar.button("▶  Process next batch", use_container_width=True, type="primary"):
    for _ in range(process_count):
        if st.session_state.stream_index >= len(demo_df):
            st.session_state.stream_index = 0
        row = demo_df.iloc[st.session_state.stream_index]
        result = analyze_raw_transaction(row)
        st.session_state.processed_transactions.append({
            "Transaction": st.session_state.stream_index + 1,
            "Step": int(row["step"]),
            "Type": row["type"],
            "Amount": float(row["amount"]),
            "Fraud Probability": result["fraud_probability"],
            "Anomaly Score": result["anomaly_score"],
            "Risk": result["risk_level"],
        })
        st.session_state.stream_index += 1
    st.rerun()

if st.sidebar.button("↻  Reset monitoring", use_container_width=True):
    st.session_state.processed_transactions = []
    st.session_state.stream_index = 0
    st.session_state.selected_transaction = None
    st.rerun()

processed = pd.DataFrame(st.session_state.processed_transactions)

# ============================================================
# CHART HELPER
# ============================================================
plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]


def render_chart(series, kind="line", height=2.1, colors=None, percent=False):
    """Dark analytics chart. Sized so fonts stay readable inside the column."""
    fig, ax = plt.subplots(figsize=(6.4, height), dpi=130)
    fig.patch.set_facecolor("#0f1729")
    ax.set_facecolor("#0f1729")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", colors="#9aa8bd", labelsize=10, length=0, pad=5)
    ax.grid(axis="y", color="#1d2940", linewidth=0.8)
    ax.set_axisbelow(True)

    vals = [float(v) for v in series.values]
    vmax = max(vals) if vals else 0

    if kind == "line":
        x = list(range(1, len(vals) + 1))
        color = (colors or ["#55b7ff"])[0]
        ax.plot(x, vals, color=color, linewidth=2.2, solid_capstyle="round")
        ax.fill_between(x, vals, 0, color=color, alpha=0.10)
        # highlight latest point
        ax.scatter([x[-1]], [vals[-1]], s=38, color=color, edgecolor="#0f1729", linewidth=1.5, zorder=5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.set_ylim(0, max(vmax * 1.3, 1e-6))
        ax.margins(x=0.03)
        if percent:
            if vmax < 0.01:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}%"))
            elif vmax < 1:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}%"))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    else:
        labels = list(series.index)
        cols = colors or ["#27d3b2"] * len(vals)
        bars = ax.bar(range(len(vals)), vals, color=cols, width=0.55)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=0, color="#aab6c9", fontsize=10)
        top = max(vmax * 1.3, 1)
        ax.set_ylim(0, top)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.03, f"{int(v):,}",
                    ha="center", va="bottom", color="#e4eaf3", fontsize=10, fontweight="bold")

    fig.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True, clear_figure=True)


def empty_state(message):
    st.markdown(f'<div class="empty-chart">{message}</div>', unsafe_allow_html=True)


def card_title(title, sub):
    st.markdown(f'<div class="section-title">{title}<span>{sub}</span></div>', unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="topbar">
        <div>
            <div class="page-title">Financial Fraud Detection</div>
            <div class="page-subtitle">Transaction monitoring &amp; suspicious activity analysis</div>
        </div>
        <div class="demo-pill"><i></i>Demo mode</div>
    </div>
    <div class="model-strip">
        <span class="live-dot"></span>
        <b>Monitoring active</b>
        <span class="model-divider">&bull;</span>
        XGBoost + Isolation Forest + SHAP
        <span class="model-note">PaySim held-out test data</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DERIVED METRICS
# ============================================================
if processed.empty:
    high = medium = low = 0
    fraud_rate = 0.0
else:
    high = int((processed["Risk"] == "HIGH").sum())
    medium = int((processed["Risk"] == "MEDIUM").sum())
    low = int((processed["Risk"] == "LOW").sum())
    fraud_rate = float((processed["Fraud Probability"] >= 0.50).mean() * 100)

n_proc = len(processed)


def share(n):
    return f"{(n / n_proc * 100):.1f}% of processed" if n_proc else "no data yet"


# ============================================================
# KPI ROW
# ============================================================
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi"><div class="kpi-label">Transactions</div><div class="kpi-value">{n_proc:,}</div><div class="kpi-foot">processed this session</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi high"><div class="kpi-label">High risk</div><div class="kpi-value">{high:,}</div><div class="kpi-foot">{share(high)}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi medium"><div class="kpi-label">Medium risk</div><div class="kpi-value">{medium:,}</div><div class="kpi-foot">{share(medium)}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi accent"><div class="kpi-label">Fraud signal</div><div class="kpi-value">{fraud_rate:.2f}%</div><div class="kpi-foot">probability ≥ 50%</div></div>', unsafe_allow_html=True)

st.write("")

# ============================================================
# CHARTS ROW 1
# ============================================================
left, right = st.columns([1.35, 1])

with left:
    with st.container(border=True):
        card_title("Transaction activity", "fraud probability by processing order")
        if processed.empty:
            empty_state("Process a batch to populate the monitoring view.")
        else:
            activity = processed.copy().reset_index(drop=True)
            activity["Index"] = range(1, len(activity) + 1)
            activity["Fraud Probability %"] = activity["Fraud Probability"] * 100
            render_chart(activity.set_index("Index")["Fraud Probability %"], "line", 1.75, ["#55b7ff"], percent=True)

with right:
    with st.container(border=True):
        card_title("Risk distribution", "current session")
        risk_df = pd.DataFrame({"Risk": ["LOW", "MEDIUM", "HIGH"], "Count": [low, medium, high]})
        render_chart(risk_df.set_index("Risk")["Count"], "bar", 1.75, ["#27d3b2", "#f2b84b", "#f05d6c"])

# ============================================================
# CHARTS ROW 2
# ============================================================
left, right = st.columns([1, 1.35])

with left:
    with st.container(border=True):
        card_title("Transactions by type", "volume")
        type_counts = processed["Type"].value_counts().sort_values() if not processed.empty else pd.Series(dtype=int)
        if type_counts.empty:
            empty_state("No transactions yet.")
        else:
            render_chart(type_counts, "bar", 1.75, ["#8b7cf6", "#55b7ff", "#27d3b2", "#f2b84b", "#f05d6c"])

with right:
    with st.container(border=True):
        card_title("Fraud probability", "latest 20 transactions")
        if processed.empty:
            empty_state("No model scores yet.")
        else:
            probability = processed.tail(20).copy().reset_index(drop=True)
            probability["Index"] = range(1, len(probability) + 1)
            probability["Fraud Probability %"] = probability["Fraud Probability"] * 100
            render_chart(probability.set_index("Index")["Fraud Probability %"], "line", 1.75, ["#8b7cf6"], percent=True)

# ============================================================
# ATTENTION QUEUE + INVESTIGATION
# ============================================================
st.markdown('<div class="queue-header"><span class="t">Attention queue</span><span class="s">highest-risk transactions</span></div>', unsafe_allow_html=True)

if processed.empty:
    st.markdown('<div class="queue-empty">No transactions processed yet. Use <b>Process next batch</b> in the sidebar.</div>', unsafe_allow_html=True)
else:
    queue = processed[processed["Risk"].isin(["HIGH", "MEDIUM"])].copy()
    queue = queue.sort_values(["Fraud Probability", "Anomaly Score"], ascending=False).head(5)
    if queue.empty:
        st.markdown('<div class="queue-empty">✓ No high- or medium-risk transactions in the current session.</div>', unsafe_allow_html=True)
    else:
        rows = ['<div class="queue-wrap">']
        rows.append('<div class="queue-row queue-head"><div>ID</div><div>Type</div><div>Amount</div><div>Fraud probability</div><div>Anomaly</div><div>Risk</div></div>')
        for _, r in queue.iterrows():
            risk_cls = str(r["Risk"]).lower()
            prob_pct = float(r["Fraud Probability"]) * 100
            bar_w = max(2, min(prob_pct, 100))
            rows.append(
                f'<div class="queue-row">'
                f'<div class="queue-cell queue-id">#{int(r["Transaction"])}</div>'
                f'<div class="queue-cell">{r["Type"]}</div>'
                f'<div class="queue-cell">₹{r["Amount"]:,.0f}</div>'
                f'<div class="queue-cell prob-cell"><span>{prob_pct:.2f}%</span>'
                f'<div class="prob-bar {risk_cls}"><i style="width:{bar_w:.1f}%"></i></div></div>'
                f'<div class="queue-cell">{r["Anomaly Score"]:.3f}</div>'
                f'<div><span class="risk-badge risk-{risk_cls}">{r["Risk"]}</span></div>'
                f'</div>'
            )
        rows.append('</div>')
        st.markdown("".join(rows), unsafe_allow_html=True)

        st.write("")
        options = queue["Transaction"].astype(int).tolist()
        default_idx = 0
        if st.session_state.selected_transaction in options:
            default_idx = options.index(st.session_state.selected_transaction)
        selected_id = st.selectbox(
            "Inspect flagged transaction", options, index=default_idx,
            format_func=lambda x: f"Transaction #{x}",
        )
        st.session_state.selected_transaction = selected_id
        selected = processed[processed["Transaction"] == selected_id].iloc[0]
        risk = selected["Risk"]
        probability = float(selected["Fraud Probability"])
        anomaly = float(selected["Anomaly Score"])

        card_title("Investigation", "model evidence &amp; explanation")
        a, b, c, d = st.columns(4)
        with a:
            st.markdown(f'<div class="investigation-card"><div class="investigation-label">Fraud probability</div><div class="investigation-value">{probability*100:.2f}%</div></div>', unsafe_allow_html=True)
        with b:
            st.markdown(f'<div class="investigation-card"><div class="investigation-label">Anomaly score</div><div class="investigation-value">{anomaly:.3f}</div></div>', unsafe_allow_html=True)
        with c:
            badge = str(risk).lower()
            st.markdown(f'<div class="investigation-card"><div class="investigation-label">Risk level</div><div class="investigation-value"><span class="risk-badge risk-{badge}">{risk}</span></div></div>', unsafe_allow_html=True)
        with d:
            st.markdown(f'<div class="investigation-card"><div class="investigation-label">Transaction</div><div class="investigation-value">#{int(selected["Transaction"])}</div></div>', unsafe_allow_html=True)

        try:
            raw_row = demo_df.iloc[int(selected["Transaction"]) - 1]
            raw_transaction = analyze_raw_transaction(raw_row)["transaction"]
            explanation = explain_transaction(raw_transaction, risk, top_n=3)
        except Exception:
            explanation = []

        st.write("")
        detail_left, detail_right = st.columns([1.1, 1])
        with detail_left:
            st.markdown(
                f'<div class="detail-card"><div class="detail-grid">'
                f'<div><span>Type</span><b>{selected["Type"]}</b></div>'
                f'<div><span>Amount</span><b>₹{selected["Amount"]:,.2f}</b></div>'
                f'<div><span>Step</span><b>{int(selected["Step"])}</b></div>'
                f'<div><span>Risk</span><b>{risk}</b></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        with detail_right:
            if explanation:
                reason_text = "".join([f"<li>{str(x)}</li>" for x in explanation[:3]])
                st.markdown(f'<div class="detail-card"><div class="reason-title">Why it was flagged</div><ul>{reason_text}</ul></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="detail-card"><div class="reason-title">Why it was flagged</div><div class="muted">Model explanation unavailable for this transaction.</div></div>', unsafe_allow_html=True)

with st.expander("View processed transactions", expanded=False):
    if processed.empty:
        st.info("No transactions processed yet.")
    else:
        display_all = processed.copy()
        display_all["Amount"] = display_all["Amount"].map(lambda x: f"₹{x:,.2f}")
        display_all["Fraud Probability"] = display_all["Fraud Probability"].map(lambda x: f"{x*100:.3f}%")
        display_all["Anomaly Score"] = display_all["Anomaly Score"].map(lambda x: f"{x:.4f}")
        st.dataframe(display_all.iloc[::-1], use_container_width=True, hide_index=True, height=260)

st.caption("Simulated monitoring environment using the held-out PaySim test dataset. Not connected to live banking data.")