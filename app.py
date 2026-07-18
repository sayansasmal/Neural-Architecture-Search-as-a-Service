# app.py  ── NAS As A Service  |  Full Dark Dashboard Redesign
import streamlit as st
import pandas as pd
import os, zipfile, io, shutil, json, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils import default_transforms, ImageFolderCSV
from train import fine_tune_model
from models import get_model

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="NAS As A Service",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Master CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
#MainMenu, footer { visibility: hidden; }

/* NOTE: we no longer hide the whole <header>. The header element also
   contains the sidebar collapse/expand arrow (data-testid="collapsedControl").
   Hiding it entirely made the sidebar impossible to reopen once collapsed
   (e.g. on narrower screens, or after a user clicks the collapse arrow).
   Instead we make the header visually transparent but keep the toggle
   control visible and clickable. */
header {
    background: transparent !important;
    box-shadow: none !important;
}
header [data-testid="stDecoration"] { visibility: hidden; }

[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}
[data-testid="collapsedControl"] svg {
    fill: #a5b4fc !important;
}
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Background ── */
.stApp {
    background: #060912;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,102,241,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(56,189,248,0.08) 0%, transparent 50%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Sidebar nav radio ── */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
}
[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #64748b !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    border: 1px solid transparent !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,102,241,0.08) !important;
    color: #e2e8f0 !important;
    border-color: rgba(99,102,241,0.2) !important;
}
[data-testid="stSidebar"] [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio [data-checked="true"] label {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
    border-color: rgba(99,102,241,0.3) !important;
}

/* ── Cards ── */
.card {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.card:hover {
    border-color: rgba(99,102,241,0.28);
    box-shadow: 0 0 40px rgba(99,102,241,0.06);
}
.card-glow {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 0 30px rgba(56,189,248,0.05), inset 0 1px 0 rgba(56,189,248,0.08);
}
.card-purple {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 0 30px rgba(139,92,246,0.05);
}

/* ── Stat grid ── */
.stat-grid { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
.stat-card {
    flex: 1; min-width: 110px;
    background: rgba(17,24,39,0.9);
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    border: 1px solid rgba(99,102,241,0.1);
    position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, linear-gradient(90deg, #6366f1, #38bdf8));
}
.stat-card:hover { transform: translateY(-2px); border-color: rgba(99,102,241,0.25); }
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem; font-weight: 800;
    color: var(--val-color, #38bdf8);
    line-height: 1; margin-bottom: 6px;
}
.stat-lbl {
    font-size: 0.68rem; font-weight: 600;
    color: #475569; text-transform: uppercase; letter-spacing: 0.1em;
}

/* ── Hero ── */
.hero {
    position: relative; overflow: hidden;
    background: rgba(10,14,26,0.9);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 48px 48px 40px;
    margin-bottom: 28px;
}
.hero::before {
    content: ''; position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 65%);
    border-radius: 50%; pointer-events: none;
}
.hero::after {
    content: ''; position: absolute;
    bottom: -60px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(56,189,248,0.1) 0%, transparent 65%);
    border-radius: 50%; pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 100px; padding: 5px 14px;
    font-size: 0.72rem; font-weight: 600;
    color: #a5b4fc; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 20px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800; line-height: 1.1;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #e2e8f0 0%, #a5b4fc 50%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.95rem; color: #64748b; line-height: 1.7;
    max-width: 580px; margin-bottom: 24px;
}
.hero-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    background: rgba(30,41,59,0.8);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 8px; padding: 5px 12px;
    font-size: 0.73rem; color: #94a3b8; font-weight: 500;
}

/* ── Section label ── */
.sec-label {
    display: flex; align-items: center; gap: 10px;
    margin: 32px 0 14px;
}
.sec-num {
    width: 26px; height: 26px;
    background: linear-gradient(135deg, #6366f1, #38bdf8);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; color: white;
    flex-shrink: 0;
}
.sec-title {
    font-size: 0.82rem; font-weight: 700;
    color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;
}
.sec-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.2), transparent);
}

/* ── Terminal log ── */
.terminal {
    background: #020408;
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 12px;
    overflow: hidden;
    font-family: 'JetBrains Mono', monospace;
}
.terminal-bar {
    background: rgba(15,23,42,0.9);
    padding: 10px 16px;
    display: flex; align-items: center; gap: 8px;
    border-bottom: 1px solid rgba(56,189,248,0.1);
}
.t-dot { width: 10px; height: 10px; border-radius: 50%; }
.terminal-body {
    padding: 14px 16px;
    font-size: 0.76rem;
    color: #4ade80;
    line-height: 1.85;
    min-height: 120px;
    max-height: 280px;
    overflow-y: auto;
}

/* ── Badge ── */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 11px; border-radius: 100px;
    font-size: 0.71rem; font-weight: 600;
}
.badge-green  { background: rgba(5,46,22,0.8);  color:#4ade80; border:1px solid rgba(74,222,128,0.25); }
.badge-blue   { background: rgba(12,26,46,0.8);  color:#38bdf8; border:1px solid rgba(56,189,248,0.25); }
.badge-purple { background: rgba(30,15,56,0.8);  color:#a78bfa; border:1px solid rgba(139,92,246,0.25); }
.badge-yellow { background: rgba(28,16,3,0.8);   color:#fbbf24; border:1px solid rgba(251,191,36,0.25); }
.badge-red    { background: rgba(46,5,5,0.8);    color:#f87171; border:1px solid rgba(248,113,113,0.25); }

/* ── Best result banner ── */
.result-banner {
    background: linear-gradient(135deg, rgba(5,46,22,0.6), rgba(12,26,46,0.6));
    border: 1px solid rgba(74,222,128,0.3);
    border-left: 4px solid #4ade80;
    border-radius: 14px; padding: 20px 24px; margin: 20px 0;
}
.rb-label { font-size:0.7rem; font-weight:700; color:#4ade80; text-transform:uppercase; letter-spacing:0.1em; }
.rb-name  { font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800; color:#e2e8f0; margin:4px 0 8px; }
.rb-meta  { display:flex; flex-wrap:wrap; gap:8px; }

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.88rem !important; padding: 11px 22px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.25) !important;
}
.stButton > button:hover {
    opacity: 0.9 !important; transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.35) !important;
}
.stDownloadButton > button {
    background: rgba(17,24,39,0.9) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stDownloadButton > button:hover {
    background: rgba(56,189,248,0.08) !important;
    border-color: rgba(56,189,248,0.5) !important;
}
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(17,24,39,0.9) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
}
.stTextInput > div > div > input {
    background: rgba(17,24,39,0.9) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #38bdf8) !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #38bdf8) !important;
    border-radius: 6px !important;
}
div[data-testid="stExpander"] {
    background: rgba(17,24,39,0.7) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(99,102,241,0.25) !important;
}
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
[data-testid="stMetricValue"] {
    color: #38bdf8 !important; font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 0.72rem !important; }
.stAlert { border-radius: 12px !important; }
.stFileUploader > div {
    background: rgba(17,24,39,0.6) !important;
    border: 2px dashed rgba(99,102,241,0.25) !important;
    border-radius: 14px !important;
    transition: border-color 0.3s !important;
}
.stFileUploader > div:hover {
    border-color: rgba(99,102,241,0.5) !important;
    background: rgba(99,102,241,0.04) !important;
}
h1,h2,h3,h4,h5 { color: #e2e8f0 !important; font-family:'Space Grotesk',sans-serif !important; }
p, li, .stMarkdown p { color: #94a3b8 !important; }
code { background: rgba(17,24,39,0.9) !important; color: #7dd3fc !important;
       border-radius: 6px !important; font-family:'JetBrains Mono',monospace !important; }
hr  { border-color: rgba(99,102,241,0.12) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(17,24,39,0.8) !important;
    border-radius: 12px !important; gap: 4px !important;
    padding: 5px !important; border: 1px solid rgba(99,102,241,0.12) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important; color: #64748b !important;
    font-weight: 500 !important; font-size: 0.83rem !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #38bdf8) !important;
    color: white !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060912; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }
</style>
""", unsafe_allow_html=True)

# ── Paths ───────────────────────────────────────────────────────
DATA_TMP_DIR = "tmp_dataset"
OUTPUT_DIR   = "outputs"
MODEL_PATH   = Path(OUTPUT_DIR) / "best_model.pt"
META_PATH    = Path(OUTPUT_DIR) / "meta.json"


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding: 28px 16px 20px; border-bottom: 1px solid rgba(99,102,241,0.12); margin-bottom: 16px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <div style="width:34px;height:34px;border-radius:10px;
                        background:linear-gradient(135deg,#6366f1,#38bdf8);
                        display:flex;align-items:center;justify-content:center;
                        font-size:1rem;flex-shrink:0;">⚡</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                            background:linear-gradient(90deg,#a5b4fc,#38bdf8);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;">NAS Service</div>
                <div style="font-size:0.67rem;color:#334155;margin-top:-1px;">Neural Architecture Search</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "", ["🏠  Home", "🔬  Train & Search", "🔍  Inference"],
        label_visibility="collapsed",
    )

    # Sidebar info
    st.markdown("""
    <div style="margin-top:24px;padding:16px;background:rgba(99,102,241,0.05);
                border-radius:12px;border:1px solid rgba(99,102,241,0.1);">
        <div style="font-size:0.68rem;font-weight:700;color:#475569;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">
            Search Space
        </div>
        <div style="font-size:0.78rem;color:#64748b;line-height:2;">
            <span style="color:#a5b4fc">Depth</span> · 2 / 3 / 4 blocks<br>
            <span style="color:#a5b4fc">Width</span> · 16 / 32 / 64 ch<br>
            <span style="color:#a5b4fc">Block</span> · plain / residual<br>
            <span style="color:#a5b4fc">Dropout</span> · 0 / 0.2 / 0.5<br>
            <span style="color:#38bdf8;font-weight:600;">81 total configs</span>
        </div>
    </div>

    <div style="margin-top:12px;padding:12px 16px;border-radius:10px;
                background:rgba(17,24,39,0.6);border:1px solid rgba(99,102,241,0.08);">
        <div style="font-size:0.68rem;color:#334155;line-height:1.9;">
            ✦ CPU-only · no GPU needed<br>
            ✦ PyTorch + Streamlit<br>
            ✦ Skin lesion NAS project
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def stat_grid(items):
    """items = [(value, label, color_hex), ...]"""
    cards = ""
    for val, lbl, col in items:
        cards += f"""
        <div class="stat-card" style="--val-color:{col};">
            <div class="stat-val" style="color:{col};">{val}</div>
            <div class="stat-lbl">{lbl}</div>
        </div>"""
    st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)

def sec(num, title):
    st.markdown(f"""
    <div class="sec-label">
        <div class="sec-num">{num}</div>
        <div class="sec-title">{title}</div>
        <div class="sec-line"></div>
    </div>""", unsafe_allow_html=True)

def badge(text, kind="blue"):
    st.markdown(f'<span class="badge badge-{kind}">{text}</span>', unsafe_allow_html=True)

def terminal_log(lines, title="Training log"):
    body = "<br>".join(lines[-14:]) if lines else "<span style='color:#334155'>Waiting…</span>"
    st.markdown(f"""
    <div class="terminal">
        <div class="terminal-bar">
            <div class="t-dot" style="background:#f87171;"></div>
            <div class="t-dot" style="background:#fbbf24;"></div>
            <div class="t-dot" style="background:#4ade80;"></div>
            <span style="margin-left:8px;font-size:0.7rem;color:#475569;">{title}</span>
        </div>
        <div class="terminal-body">{body}</div>
    </div>""", unsafe_allow_html=True)

def format_params(n):
    if n >= 1_000_000: return f"{n/1e6:.2f}M"
    if n >= 1_000:     return f"{n/1e3:.1f}K"
    return str(n)

def dark_chart(nrows=1, ncols=2, figsize=(10, 3.5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="#0a0e1a")
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axes_list:
        ax.set_facecolor("#060912")
        ax.tick_params(colors="#475569", labelsize=8)
        for spine in ax.spines.values(): spine.set_color("#1e2d4a")
        ax.xaxis.label.set_color("#475569")
        ax.yaxis.label.set_color("#475569")
        ax.title.set_color("#64748b")
        ax.grid(True, color="#0f172a", linewidth=0.8, alpha=0.7)
    return fig, axes

@torch.no_grad()
def eval_full(model, dataset, batch_size=16, device="cpu"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs  = F.softmax(logits, dim=1)
        all_true.extend(y.cpu().numpy())
        all_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_prob.extend(probs.cpu().numpy())
    return np.array(all_true), np.array(all_pred), np.array(all_prob)

def plot_curves(log, title=""):
    tl = log.get("train_loss_history") or log.get("train_loss", [])
    vl = log.get("val_loss_history")   or log.get("val_loss",   [])
    ta = log.get("train_acc_history")  or log.get("train_acc",  [])
    va = log.get("val_acc_history")    or log.get("val_acc",    [])
    ep = range(1, len(tl)+1)
    fig, (ax1, ax2) = dark_chart()
    if title: fig.suptitle(title, color="#64748b", fontsize=9.5, fontweight="600", y=1.01)
    ax1.plot(ep, tl, "o-", color="#6366f1", lw=2, markersize=4, label="Train")
    ax1.plot(ep, vl, "s--",color="#38bdf8", lw=2, markersize=4, label="Val")
    ax1.set_title("Loss", fontsize=9); ax1.legend(fontsize=8, labelcolor="#94a3b8", framealpha=0.1)
    ax2.plot(ep, ta, "o-", color="#6366f1", lw=2, markersize=4, label="Train")
    ax2.plot(ep, va, "s--",color="#38bdf8", lw=2, markersize=4, label="Val")
    ax2.set_title("Accuracy", fontsize=9); ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, labelcolor="#94a3b8", framealpha=0.1)
    plt.tight_layout()
    return fig

def plot_nas_bars(logs):
    names  = [l["name"] for l in logs]
    accs   = [l["val_acc"] for l in logs]
    params = [l["params"]/1e6 for l in logs]
    bi     = accs.index(max(accs))
    ac     = ["#6366f1" if i==bi else "#1e2d4a" for i in range(len(names))]
    ec     = ["#818cf8" if i==bi else "#243656" for i in range(len(names))]
    pc     = ["#38bdf8" if i==bi else "#0e3a52" for i in range(len(names))]
    w      = max(10, len(names)*1.7)
    fig, (ax1, ax2) = dark_chart(figsize=(w, 4))
    fig.suptitle("Architecture Comparison", color="#64748b", fontsize=10, fontweight="600")
    bars = ax1.bar(names, accs, color=ac, edgecolor=ec, linewidth=1.2)
    ax1.set_ylim(0, 1.12); ax1.set_title("Validation Accuracy", fontsize=9)
    for b, v in zip(bars, accs):
        ax1.text(b.get_x()+b.get_width()/2, v+0.025, f"{v:.3f}",
                 ha="center", fontsize=8, color="#94a3b8", fontweight="600")
    ax1.tick_params(axis="x", rotation=22, labelsize=7.5)
    ax2.bar(names, params, color=pc, edgecolor=ec, linewidth=1.2)
    ax2.set_title("Parameter Count (M)", fontsize=9)
    for i, p in enumerate(params):
        ax2.text(i, p+0.02, f"{p:.2f}M", ha="center", fontsize=8, color="#94a3b8")
    ax2.tick_params(axis="x", rotation=22, labelsize=7.5)
    plt.tight_layout()
    return fig

def auto_labels(root_dir, allowed={".jpg",".jpeg",".png",".bmp",".gif"}):
    def subdirs(d):
        return [n for n in sorted(os.listdir(d))
                if os.path.isdir(os.path.join(d,n)) and not n.startswith(".")]
    subs = subdirs(root_dir); cr = root_dir
    if len(subs)==1:
        cand = os.path.join(root_dir, subs[0]); inner = subdirs(cand)
        if inner: cr, subs = cand, inner
    rows, cmap, cid = [], {}, 0
    for cls in subs:
        cmap[cid] = cls
        for r,_,files in os.walk(os.path.join(cr,cls)):
            for f in files:
                if Path(f).suffix.lower() in allowed:
                    rel = os.path.relpath(os.path.join(r,f), root_dir).replace("\\","/")
                    rows.append((rel, cid))
        cid += 1
    return pd.DataFrame(rows, columns=["fname","label"]), cmap, len(rows)

def validate_sample(root_dir, df, n=150):
    from PIL import Image as PILImage
    import random
    paths = random.Random(42).sample(df["fname"].tolist(), min(n, len(df)))
    miss, corrupt = [], []
    for rel in paths:
        fp = os.path.join(root_dir, rel)
        if not os.path.exists(fp): miss.append(rel); continue
        try:
            with PILImage.open(fp) as img: img.verify()
        except: corrupt.append(rel)
    return miss, corrupt


# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if "Home" in page:

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">⚡ Final Year Project · By Sayan Sasmal, Subhamoy Maity, Deep Dolai & Sayan Patra </div>
        <div class="hero-title">Neural Architecture<br>Search as a Service</div>
        <div class="hero-sub">
            Automatically discover the best deep learning architecture for your image
            classification task — no GPU required, no coding needed. Upload your dataset,
            configure the search, and let the engine do the work.
        </div>
        <div class="hero-chips">
            <span class="chip">⚡ CPU-compatible</span>
            <span class="chip">🔬 Configurable search space</span>
            <span class="chip">📊 Live training logs</span>
            <span class="chip">🧠 3 block types</span>
            <span class="chip">🎯 Auto model selection</span>
            <span class="chip">🩺 Skin lesion NAS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    stat_grid([
        ("81",  "Search configs",  "#6366f1"),
        ("3",   "Block types",     "#38bdf8"),
        ("5",   "Preset models",   "#a78bfa"),
        ("2+",  "Classes",         "#4ade80"),
        ("CPU", "GPU-free",        "#f472b6"),
    ])

    # Two columns: how it works + block types
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown("""
        <div class="card">
            <div style="font-size:0.7rem;font-weight:700;color:#6366f1;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px;">
                How it works
            </div>
            <div style="font-size:0.82rem;color:#64748b;line-height:2.2;">
                <div><span style="color:#a5b4fc;font-weight:600;">①</span>
                    &nbsp;Upload a folder-per-class ZIP dataset</div>
                <div><span style="color:#a5b4fc;font-weight:600;">②</span>
                    &nbsp;Select search space dimensions</div>
                <div><span style="color:#a5b4fc;font-weight:600;">③</span>
                    &nbsp;Engine trains each candidate with early stopping</div>
                <div><span style="color:#a5b4fc;font-weight:600;">④</span>
                    &nbsp;Best architecture selected automatically</div>
                <div><span style="color:#a5b4fc;font-weight:600;">⑤</span>
                    &nbsp;Final model fine-tuned and ready to download</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card-purple">
            <div style="font-size:0.7rem;font-weight:700;color:#8b5cf6;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:14px;">
                Search space dimensions
            </div>
            <div style="font-size:0.82rem;line-height:2.2;">
                <div><span style="color:#e2e8f0;font-weight:600;">Depth&nbsp;</span>
                    <span style="color:#64748b;">2 / 3 / 4 conv blocks stacked</span></div>
                <div><span style="color:#e2e8f0;font-weight:600;">Width&nbsp;</span>
                    <span style="color:#64748b;">16 / 32 / 64 base channels</span></div>
                <div><span style="color:#e2e8f0;font-weight:600;">Block&nbsp;</span>
                    <span style="color:#64748b;">plain / residual / bottleneck</span></div>
                <div><span style="color:#e2e8f0;font-weight:600;">Dropout&nbsp;</span>
                    <span style="color:#64748b;">0.0 / 0.2 / 0.5 regularisation</span></div>
                <div style="margin-top:4px;">
                    <span class="badge badge-purple">3×3×3×3 = 81 combinations</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Block type cards
    st.markdown("""
    <div style="font-size:0.7rem;font-weight:700;color:#475569;
                text-transform:uppercase;letter-spacing:0.1em;margin:28px 0 12px;">
        Block Types
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    for col, name, icon, desc, accent in [
        (c1,"Plain Block","▣","Conv → BN → ReLU → MaxPool. Simple, fast, low parameter count. Best baseline on CPU.","#6366f1"),
        (c2,"Residual Block","⊕","Adds skip connection: output = F(x) + x. Solves vanishing gradients in deeper networks.","#38bdf8"),
        (c3,"Bottleneck Block","⊗","1×1 squeeze → 3×3 → 1×1 expand. Parameter-efficient — same idea as MobileNet/EfficientNet.","#a78bfa"),
    ]:
        col.markdown(f"""
        <div class="card" style="border-color:rgba(99,102,241,0.08);min-height:150px;">
            <div style="font-size:1.4rem;margin-bottom:8px;">{icon}</div>
            <div style="font-size:0.85rem;font-weight:700;color:{accent};margin-bottom:6px;">{name}</div>
            <div style="font-size:0.77rem;color:#475569;line-height:1.65;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card-glow" style="text-align:center;padding:20px;margin-top:8px;">
        <span style="color:#64748b;font-size:0.85rem;">
            Ready to start?&nbsp;&nbsp;
            <span style="color:#38bdf8;font-weight:600;">→ Open 🔬 Train & Search in the sidebar</span>
        </span>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: TRAIN & SEARCH
# ══════════════════════════════════════════════════════════════════
elif "Train" in page:

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                    color:#e2e8f0;margin-bottom:4px;">🔬 Train & Search</div>
        <div style="color:#475569;font-size:0.83rem;">
            Upload your dataset · configure the search space · launch the NAS engine.
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Step 1: Dataset ──────────────────────────────────────────
    sec(1, "Upload Dataset")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    ingestion = st.radio(
        "", ["📦  ZIP file upload"],
        horizontal=True, label_visibility="collapsed",
    )

    root_dir = None

    if "ZIP" in ingestion:
        st.markdown("""<div style="font-size:0.78rem;color:#475569;margin-bottom:10px;">
            Folder-per-class ZIP · Up to 2 GB · Set in <code>.streamlit/config.toml</code>
        </div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["zip"], label_visibility="collapsed")

        if uploaded:
            file_mb = uploaded.size / 1e6
            st.markdown(f'<span class="badge badge-blue">📦 {file_mb:.1f} MB received</span>',
                        unsafe_allow_html=True)
            key = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get("_upload_key") != key:
                if os.path.exists(DATA_TMP_DIR):
                    try: shutil.rmtree(DATA_TMP_DIR)
                    except: pass
                st.session_state["_upload_key"]    = key
                st.session_state["_dataset_ready"] = False

            if not st.session_state.get("_dataset_ready"):
                prog = st.progress(0)
                stat = st.empty()
                stat.info("Extracting…")
                os.makedirs(DATA_TMP_DIR, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(uploaded.read())) as z:
                    members = [m for m in z.infolist() if not m.filename.endswith("/")]
                    for i, m in enumerate(members):
                        z.extract(m, DATA_TMP_DIR)
                        prog.progress((i+1)/len(members))
                prog.empty(); stat.empty()
                st.session_state["_dataset_ready"] = True
                st.session_state["_dataset_root"]  = DATA_TMP_DIR
            root_dir = st.session_state.get("_dataset_root", DATA_TMP_DIR)

    st.markdown('</div>', unsafe_allow_html=True)

    if root_dir and st.session_state.get("_dataset_ready"):

        with st.spinner("Scanning dataset…"):
            df, class_id_to_name, total = auto_labels(root_dir)

        if total == 0:
            st.error("No images found. Check folder structure."); st.stop()

        # Dataset stats
        total_bytes = sum(
            os.path.getsize(os.path.join(root_dir, r))
            for r in df["fname"] if os.path.exists(os.path.join(root_dir, r))
        )
        mb = total_bytes/1e6
        size_str = f"{mb:.1f}MB" if mb < 1000 else f"{mb/1000:.2f}GB"

        stat_grid([
            (str(total),                  "Total images",  "#38bdf8"),
            (str(len(class_id_to_name)), "Classes",       "#6366f1"),
            (size_str,                   "Dataset size",  "#a78bfa"),
        ])

        with st.expander("📋 View class map"):
            st.json({str(k): v for k, v in class_id_to_name.items()})

        # Distribution chart
        count_df = pd.DataFrame({
            "class": [class_id_to_name[i] for i in sorted(class_id_to_name)],
            "images": [len(df[df.label==i]) for i in sorted(class_id_to_name)],
        })
        st.bar_chart(count_df.set_index("class"), color="#6366f1")

        with st.spinner("Validating sample images…"):
            miss, corrupt = validate_sample(root_dir, df)
        if not miss and not corrupt:
            st.markdown('<span class="badge badge-green">✓ Sample validation passed (150 images)</span>',
                        unsafe_allow_html=True)
        else:
            if miss:    st.warning(f"{len(miss)} missing")
            if corrupt: st.warning(f"{len(corrupt)} corrupted")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            val_frac = st.slider("Validation split", 0.1, 0.4, 0.2)
        with c2:
            image_size = st.selectbox("Image size (px)", [128, 160, 224], index=0,
                                      help="Smaller = faster on CPU")

        train_df, val_df = train_test_split(df, test_size=val_frac,
                                            stratify=df["label"], random_state=42)
        st.markdown(
            f'<span class="badge badge-blue">Train {len(train_df)}</span>'
            f'&nbsp;<span class="badge badge-purple">Val {len(val_df)}</span>',
            unsafe_allow_html=True,
        )
        train_t, val_t = default_transforms(image_size=image_size)
        train_ds = ImageFolderCSV(root_dir, train_df.values.tolist(), transform=train_t)
        val_ds   = ImageFolderCSV(root_dir, val_df.values.tolist(),   transform=val_t)
        num_classes = int(df.label.nunique())

        # ── Step 2: NAS config ───────────────────────────────────
        sec(2, "Configure NAS Search")

        st.markdown("""<div style="font-size:0.78rem;color:#64748b;margin:0 0 16px;">
            Every run first trains all selected preset architectures, then every
            selected configurable-space candidate — the single best result across
            <b style="color:#a5b4fc;">both pools combined</b> is chosen automatically
            and taken forward to fine-tuning.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.7rem;font-weight:700;color:#6366f1;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">
            Pool 1 · Preset architectures
        </div>""", unsafe_allow_html=True)
        preset_candidates = st.multiselect(
            "Architectures",
            ["tiny_cnn","small_cnn","resnet18","mobilenet_v2","efficientnet_b0"],
            default=["tiny_cnn","mobilenet_v2"],
        )
        if not preset_candidates:
            st.caption("No preset architectures selected — this pool will be skipped.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.7rem;font-weight:700;color:#8b5cf6;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;">
            Pool 2 · Configurable search space
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.78rem;color:#64748b;margin:0 0 14px;">
            Build architectures dynamically from dimensions below.
            Every combination of selected values becomes one additional candidate.
            Leave any dimension empty to skip this pool entirely.
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            sel_d  = st.multiselect("Depth",        [2,3,4],                    default=[2,3])
            sel_b  = st.multiselect("Block type",   ["plain","residual","bottleneck"], default=["plain","residual"])
        with c2:
            sel_ch = st.multiselect("Base channels",[16,32,64],                 default=[16,32])
            sel_dr = st.multiselect("Dropout",      [0.0,0.2,0.5],              default=[0.0,0.2])

        config_candidates = []
        max_p = None
        if all([sel_d, sel_b, sel_ch, sel_dr]):
            import itertools
            all_config_candidates = [
                {"depth":d,"base_channels":c,"block_type":b,"dropout":dr}
                for d,c,b,dr in itertools.product(sel_d,sel_ch,sel_b,sel_dr)
            ]
            max_m = st.slider("Max parameter budget (M)", 0.1, 5.0, 2.0, 0.1)
            max_p = int(max_m * 1_000_000)

            from search_space import config_to_name, estimate_params
            prev_rows = []
            skip_n = 0
            for cfg in all_config_candidates:
                est = estimate_params(cfg, num_classes)
                sk  = est > max_p
                if sk: skip_n += 1
                prev_rows.append({
                    "name": config_to_name(cfg),
                    "depth":cfg["depth"],"base_ch":cfg["base_channels"],
                    "block":cfg["block_type"],"dropout":cfg["dropout"],
                    "est. params": format_params(est),
                    "status": "⛔" if sk else "✅",
                })
            st.dataframe(pd.DataFrame(prev_rows), use_container_width=True, height=200)
            wt = len(all_config_candidates)-skip_n
            st.markdown(
                f'<span class="badge badge-green">✅ {wt} will train</span>'
                f'&nbsp;<span class="badge badge-yellow">⛔ {skip_n} skipped</span>',
                unsafe_allow_html=True,
            )
            strat = st.radio("Strategy",["All (exhaustive)","Random sample"],horizontal=True)
            valid_configs = [c for c in all_config_candidates
                             if estimate_params(c,num_classes)<=max_p]
            if "Random" in strat:
                ns = st.slider("Sample size", 1, max(len(valid_configs),1),
                               min(6, max(len(valid_configs),1)))
                config_candidates = __import__('random').Random(42).sample(
                    valid_configs, min(ns, len(valid_configs))
                )
            else:
                config_candidates = valid_configs
        else:
            st.info("Select at least one value in every dimension (Depth, Block type, "
                    "Base channels, Dropout) to include this pool in the run.")
        st.markdown('</div>', unsafe_allow_html=True)

        if not preset_candidates and not config_candidates:
            st.warning("Select at least one preset architecture or configure the search space above.")
            st.stop()

        # ── Step 3: Hyperparams ──────────────────────────────────
        sec(3, "Hyperparameters & Fine-tuning")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: epochs   = st.slider("Epochs / trial",  1, 10, 3)
        with c2: patience = st.slider("Early stop patience", 1, epochs, min(2, epochs))
        with c3: batch    = st.selectbox("Batch size", [4,8,16,32], index=1)
        with c4: lr       = float(st.text_input("Learning rate", "0.001"))

        c5, c6 = st.columns(2)
        with c5: final_ep  = st.slider("Fine-tune epochs",   0, 20, 5)
        with c6: final_pat = st.slider("Fine-tune patience", 1, max(final_ep,1), min(3,max(final_ep,1)))
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Launch ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("⚡  Launch NAS Engine", type="primary", use_container_width=True)

        if go and (preset_candidates or config_candidates):
            st.markdown("---")
            st.markdown("""
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                        font-weight:800;color:#e2e8f0;margin-bottom:16px;">
                ⚡ Search in progress
            </div>""", unsafe_allow_html=True)

            trial_ph = st.empty()
            log_ph   = st.empty()
            prog_ph  = st.progress(0)
            lines    = []
            total_t  = len(preset_candidates) + len(config_candidates)

            def on_trial(name, params):
                trial_ph.markdown(
                    f'<span class="badge badge-blue">▶ {name}</span>'
                    f'<span style="color:#334155;font-size:0.75rem;margin-left:8px;">'
                    f'{format_params(params)} params</span>',
                    unsafe_allow_html=True,
                )

            def on_epoch(name, ep, total_ep, trl, tra, vll, vla, stopped_early):
                icon = "⏹" if stopped_early else "›"
                lines.append(
                    f'<span style="color:#334155">[{name}]</span> '
                    f'ep<span style="color:#a78bfa">{ep}/{total_ep}</span>  '
                    f'train <span style="color:#6366f1">{tra:.3f}</span>  '
                    f'val <span style="color:#38bdf8">{vla:.3f}</span>  '
                    f'loss <span style="color:#f472b6">{vll:.4f}</span> {icon}'
                )
                log_ph.markdown(
                    '<div class="terminal"><div class="terminal-bar">'
                    '<div class="t-dot" style="background:#f87171"></div>'
                    '<div class="t-dot" style="background:#fbbf24"></div>'
                    '<div class="t-dot" style="background:#4ade80"></div>'
                    '<span style="margin-left:8px;font-size:0.7rem;color:#475569;">nas_engine.py</span>'
                    '</div><div class="terminal-body">' +
                    "<br>".join(lines[-14:]) +
                    "</div></div>",
                    unsafe_allow_html=True,
                )
                done = sum(1 for l in lines if "ep<" in l and "/1" in l) + \
                       sum(1 for l in lines if "⏹" in l)
                prog_ph.progress(min(done / max(total_t,1), 1.0))

            t0 = time.time()
            from train import run_unified_search
            from search_space import config_to_name
            best_kind, best_id, best_score, best_state, logs = run_unified_search(
                preset_candidates, config_candidates, train_ds, val_ds, num_classes,
                device="cpu", epochs_per_trial=epochs, batch_size=batch,
                lr=lr, early_stopping_patience=patience,
                max_params=max_p,
                epoch_callback=on_epoch, trial_start_callback=on_trial,
            )
            if best_kind == "preset":
                best_name = best_id
            elif best_kind == "config":
                best_name = config_to_name(best_id)
            else:
                best_name = None

            elapsed = time.time() - t0
            trial_ph.empty(); log_ph.empty(); prog_ph.empty()

            if not logs:
                st.error("No architectures trained. Relax the parameter budget or select more candidates."); st.stop()

            # Result banner
            kind_label = "Preset" if best_kind == "preset" else "Configurable"
            st.markdown(f"""
            <div class="result-banner">
                <div class="rb-label">✓ Best architecture found</div>
                <div class="rb-name">{best_name}</div>
                <div class="rb-meta">
                    <span class="badge badge-green">Val acc {best_score:.3f}</span>
                    <span class="badge badge-blue">{len(logs)} candidates trained</span>
                    <span class="badge badge-purple">{kind_label}</span>
                    <span class="badge badge-purple">{elapsed:.0f}s total</span>
                </div>
            </div>""", unsafe_allow_html=True)

            # Results table
            st.markdown("""<div style="font-size:0.7rem;font-weight:700;color:#475569;
                          text-transform:uppercase;letter-spacing:0.1em;margin:20px 0 8px;">
                Search Results
            </div>""", unsafe_allow_html=True)
            rows = []
            for l in logs:
                row = {
                    "Architecture": l["name"],
                    "Type": "Preset" if l["kind"]=="preset" else "Configurable",
                    "Val Acc": f"{l['val_acc']:.3f}",
                    "Params": format_params(l["params"]),
                    "Epochs": l["epochs_run"],
                    "Time(s)": l["time_sec"],
                    "": "⭐" if l["name"]==best_name else "",
                }
                if l.get("config"):
                    row.update(l["config"])
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Charts
            st.markdown("""<div style="font-size:0.7rem;font-weight:700;color:#475569;
                text-transform:uppercase;letter-spacing:0.1em;margin:24px 0 10px;">
                Architecture Comparison
            </div>""", unsafe_allow_html=True)
            fig_bar = plot_nas_bars(logs)
            st.pyplot(fig_bar, use_container_width=True); plt.close(fig_bar)

            st.markdown("""<div style="font-size:0.7rem;font-weight:700;color:#475569;
                text-transform:uppercase;letter-spacing:0.1em;margin:24px 0 10px;">
                Training Curve · Best Model
            </div>""", unsafe_allow_html=True)
            best_log = next(l for l in logs if l["name"] == best_name)
            fig_c = plot_curves(best_log, title=best_name)
            st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)

            # CSV
            export_df = pd.DataFrame([{
                "name":l["name"],"val_acc":l["val_acc"],"params":l["params"],
                "epochs":l["epochs_run"],"time_s":l["time_sec"],
                **(l.get("config") or {}),
            } for l in logs])
            st.download_button("⬇  Download NAS Summary (CSV)",
                               export_df.to_csv(index=False).encode(),
                               file_name="nas_summary.csv", mime="text/csv")

            # Build model
            if best_kind == "config":
                from models import get_model_from_config
                model = get_model_from_config(best_id, num_classes)
            else:
                model = get_model(best_id, num_classes, pretrained=False)
            model.load_state_dict(best_state); model.to("cpu")

            # Fine-tune
            final_val_acc = best_score
            if final_ep > 0:
                st.markdown("---")
                st.markdown(f"""
                <div style="font-family:'Syne',sans-serif;font-size:1rem;
                            font-weight:800;color:#e2e8f0;margin-bottom:12px;">
                    🔧 Fine-tuning &nbsp;<span style="color:#38bdf8">{best_name}</span>
                </div>""", unsafe_allow_html=True)

                ft_ph = st.empty(); ft_lines = []

                def on_ft(ep, total, trl, tra, vll, vla, is_best, stopped_early):
                    icon = "⭐" if is_best else ("⏹" if stopped_early else "›")
                    ft_lines.append(
                        f'ep<span style="color:#a78bfa">{ep}/{total}</span>  '
                        f'train <span style="color:#6366f1">{tra:.3f}</span>  '
                        f'val <span style="color:#38bdf8">{vla:.3f}</span> {icon}'
                    )
                    ft_ph.markdown(
                        '<div class="terminal"><div class="terminal-bar">'
                        '<div class="t-dot" style="background:#f87171"></div>'
                        '<div class="t-dot" style="background:#fbbf24"></div>'
                        '<div class="t-dot" style="background:#4ade80"></div>'
                        '<span style="margin-left:8px;font-size:0.7rem;color:#475569;">fine_tune.py</span>'
                        '</div><div class="terminal-body">' +
                        "<br>".join(ft_lines[-12:]) +
                        "</div></div>",
                        unsafe_allow_html=True,
                    )

                with st.spinner("Fine-tuning…"):
                    model, final_val_acc, ft_hist = fine_tune_model(
                        model, train_ds, val_ds, device="cpu",
                        epochs=final_ep, batch_size=batch, lr=lr,
                        early_stopping_patience=final_pat,
                        epoch_callback=on_ft,
                    )
                ft_ph.empty()
                st.markdown(
                    f'<span class="badge badge-green">✓ Fine-tune complete · Val acc {final_val_acc:.3f}</span>',
                    unsafe_allow_html=True,
                )
                if ft_hist:
                    fig_ft = plot_curves(ft_hist, title=f"Fine-tune · {best_name}")
                    st.pyplot(fig_ft, use_container_width=True); plt.close(fig_ft)

            # Save
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save(model.state_dict(), str(MODEL_PATH))
            meta = {
                "model_name": best_name, "num_classes": int(num_classes),
                "image_size": int(image_size),
                "class_names": {str(int(k)):str(v) for k,v in class_id_to_name.items()},
                "best_config": best_id if best_kind == "config" else None,
            }
            with open(META_PATH, "w") as f: json.dump(meta, f)

            # Final eval
            st.markdown("---")
            st.markdown("""<div style="font-family:'Syne',sans-serif;font-size:1rem;
                font-weight:800;color:#e2e8f0;margin-bottom:16px;">
                📊 Final Evaluation
            </div>""", unsafe_allow_html=True)

            y_true, y_pred, _ = eval_full(model, val_ds, batch_size=batch)
            acc = (y_true==y_pred).mean()

            stat_grid([
                (f"{acc:.3f}",           "Val Accuracy",   "#4ade80"),
                (f"{final_val_acc:.3f}", "Best Val Acc",   "#38bdf8"),
                (str(len(val_ds)),       "Val Samples",    "#a78bfa"),
                (str(num_classes),       "Classes",        "#6366f1"),
            ])

            label_ids   = sorted(class_id_to_name.keys())
            label_names = [class_id_to_name[i] for i in label_ids]

            # Dark confusion matrix heatmap
            cm = confusion_matrix(y_true, y_pred, labels=label_ids)
            fig_cm, ax = plt.subplots(figsize=(max(6, len(label_ids)*1.4),
                                               max(5, len(label_ids)*1.2)),
                                      facecolor="#0a0e1a")
            ax.set_facecolor("#060912")
            sns.heatmap(cm, annot=True, fmt="d",
                        cmap=sns.color_palette("mako_r", as_cmap=True),
                        xticklabels=label_names, yticklabels=label_names,
                        ax=ax, linewidths=0.5, linecolor="#1e2d4a",
                        annot_kws={"size":10,"color":"white","weight":"600"})
            ax.set_title("Confusion Matrix", color="#64748b", fontsize=11, pad=14)
            ax.set_xlabel("Predicted", color="#475569", fontsize=9)
            ax.set_ylabel("Actual",    color="#475569", fontsize=9)
            ax.tick_params(colors="#475569")
            plt.tight_layout()

            c1, c2 = st.columns([1, 1], gap="medium")
            with c1:
                st.markdown("""<div style="font-size:0.7rem;font-weight:700;color:#475569;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
                    Confusion Matrix</div>""", unsafe_allow_html=True)
                st.pyplot(fig_cm, use_container_width=True); plt.close(fig_cm)
            with c2:
                st.markdown("""<div style="font-size:0.7rem;font-weight:700;color:#475569;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
                    Classification Report</div>""", unsafe_allow_html=True)
                report = classification_report(y_true, y_pred, labels=label_ids,
                                               target_names=label_names, zero_division=0)
                st.markdown(f'<div class="terminal"><div class="terminal-bar">'
                            f'<div class="t-dot" style="background:#f87171"></div>'
                            f'<div class="t-dot" style="background:#fbbf24"></div>'
                            f'<div class="t-dot" style="background:#4ade80"></div>'
                            f'<span style="margin-left:8px;font-size:0.7rem;color:#475569;">report.txt</span>'
                            f'</div><div class="terminal-body" style="color:#94a3b8;white-space:pre;">'
                            f'{report}</div></div>', unsafe_allow_html=True)

            with open(MODEL_PATH, "rb") as f:
                st.download_button("⬇  Download Trained Model (.pt)", f,
                                   file_name="best_model.pt", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: INFERENCE
# ══════════════════════════════════════════════════════════════════
elif "Inference" in page:

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                    color:#e2e8f0;margin-bottom:4px;">🔍 Inference</div>
        <div style="color:#475569;font-size:0.83rem;">
            Classify a single image using your trained NAS model.
        </div>
    </div>""", unsafe_allow_html=True)

    if not MODEL_PATH.exists() or not META_PATH.exists():
        st.markdown("""
        <div class="card" style="text-align:center;padding:48px 24px;">
            <div style="font-size:3rem;margin-bottom:14px;">🤖</div>
            <div style="color:#64748b;font-size:0.9rem;">No trained model found.</div>
            <div style="color:#334155;font-size:0.78rem;margin-top:6px;">
                Train one on the
                <span style="color:#38bdf8;font-weight:600;">🔬 Train & Search</span>
                page first.
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    with open(META_PATH) as f: meta = json.load(f)
    model_name       = meta["model_name"]
    num_classes      = meta["num_classes"]
    image_size       = meta["image_size"]
    class_id_to_name = {int(k):v for k,v in meta.get("class_names",{}).items()}
    best_config      = meta.get("best_config")

    # Model card
    st.markdown(f"""
    <div class="card-glow" style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div style="font-size:0.68rem;font-weight:700;color:#475569;
                        text-transform:uppercase;letter-spacing:0.1em;">Active Model</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.15rem;
                        font-weight:800;color:#e2e8f0;margin:4px 0;">
                {model_name}
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;">
                <span class="badge badge-blue">{num_classes} classes</span>
                <span class="badge badge-purple">{image_size}px</span>
                {"<span class='badge badge-green'>Configurable</span>" if best_config else "<span class='badge badge-yellow'>Preset</span>"}
            </div>
        </div>
        <div style="font-size:2.5rem;opacity:0.4;">🧠</div>
    </div>""", unsafe_allow_html=True)

    if best_config:
        from models import get_model_from_config
        model = get_model_from_config(best_config, num_classes)
    else:
        model = get_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    _, val_t = default_transforms(image_size=image_size)

    if "pred_history" not in st.session_state:
        st.session_state["pred_history"] = []

    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("""<div style="font-size:0.7rem;font-weight:700;color:#475569;
            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">
            Upload Image</div>""", unsafe_allow_html=True)
        img_file = st.file_uploader("", type=["jpg","jpeg","png"],
                                    label_visibility="collapsed")
        if img_file:
            from PIL import Image as PILImage
            img = PILImage.open(img_file).convert("RGB")
            st.image(img, use_column_width=True,
                     caption=img_file.name)

    with col_res:
        if img_file:
            x = val_t(img).unsqueeze(0)
            with torch.no_grad():
                logits     = model(x)
                probs      = F.softmax(logits, dim=1).cpu().numpy().flatten()
                pred_class = int(np.argmax(probs))
                confidence = float(probs[pred_class])

            pred_name   = class_id_to_name.get(pred_class, str(pred_class))
            conf_color  = "#4ade80" if confidence>.72 else "#fbbf24" if confidence>.45 else "#f87171"
            conf_badge  = "badge-green" if confidence>.72 else "badge-yellow" if confidence>.45 else "badge-red"

            st.markdown(f"""
            <div class="card" style="margin-bottom:12px;border-color:rgba(74,222,128,0.2);">
                <div style="font-size:0.68rem;font-weight:700;color:#475569;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
                    Prediction
                </div>
                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;
                            font-weight:800;color:#e2e8f0;margin-bottom:10px;
                            line-height:1.1;">
                    {pred_name}
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <span class="badge {conf_badge}">
                        {confidence:.1%} confidence
                    </span>
                    <span style="font-size:0.75rem;color:#334155;">
                        class {pred_class}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

            # Probability bars — all classes
            st.markdown("""<div style="font-size:0.7rem;font-weight:700;color:#475569;
                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
                Class Probabilities</div>""", unsafe_allow_html=True)
            prob_df = pd.DataFrame({
                "class": [class_id_to_name.get(i,str(i)) for i in range(num_classes)],
                "probability": probs,
            }).set_index("class")
            st.bar_chart(prob_df, color="#6366f1")

            st.session_state["pred_history"].append({
                "Image": img_file.name,
                "Prediction": pred_name,
                "Confidence": f"{confidence:.1%}",
            })
        else:
            st.markdown("""
            <div class="card" style="min-height:260px;display:flex;
                 align-items:center;justify-content:center;text-align:center;">
                <div>
                    <div style="font-size:3rem;margin-bottom:12px;opacity:0.3;">🖼️</div>
                    <div style="color:#334155;font-size:0.82rem;">
                        Upload an image to classify
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    if st.session_state["pred_history"]:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.7rem;font-weight:700;color:#475569;
            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
            Session History</div>""", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state["pred_history"]),
                     use_container_width=True, hide_index=True)
        if st.button("🗑  Clear history"):
            st.session_state["pred_history"] = []; st.rerun().set_index("class")
            st.bar_chart(prob_df, color="#6366f1")

            st.session_state["pred_history"].append({
                "Image": img_file.name,
                "Prediction": pred_name,
                "Confidence": f"{confidence:.1%}",
            })
        else:
            st.markdown("""
            <div class="card" style="min-height:260px;display:flex;
                 align-items:center;justify-content:center;text-align:center;">
                <div>
                    <div style="font-size:3rem;margin-bottom:12px;opacity:0.3;">🖼️</div>
                    <div style="color:#334155;font-size:0.82rem;">
                        Upload an image to classify
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    if st.session_state["pred_history"]:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.7rem;font-weight:700;color:#475569;
            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
            Session History</div>""", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state["pred_history"]),
                     use_container_width=True, hide_index=True)
        if st.button(
    "🗑 Clear history",
    key="clear_prediction_history_btn"
):
            st.session_state["pred_history"] = []
            st.rerun()
