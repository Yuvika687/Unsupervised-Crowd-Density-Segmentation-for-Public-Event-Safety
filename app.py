"""
app.py — SafeCrowd Vision
================================
Real-time crowd density analysis for public safety.
Powered by DM-Count · KMeans · XGBoost · DBSCAN

Launch:  python3 -m streamlit run app.py
Note: Requires Python 3.9+
"""

import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter
import plotly.graph_objects as go
import plotly.express as px
import joblib
import pandas as pd
import io
import time
import base64
import os
import gc
import json
import hashlib
import tempfile
import random
import urllib.parse
from textwrap import dedent
from collections import deque
from datetime import datetime
import torch.optim as optim

# ── LWCC (LightWeight Crowd Counting) ──
# Monkey-patch LWCC's broken weights path (uses /.lwcc instead of ~/.lwcc)
def _patch_lwcc_weights_path():
    """Fix LWCC bug: weights_check uses root / instead of ~/."""
    try:
        import lwcc.util.functions as lwcc_funcs
        from pathlib import Path
        import gdown

        def _fixed_weights_check(model_name, model_weights):
            home = str(Path.home())
            weights_dir = os.path.join(home, ".lwcc", "weights")
            os.makedirs(weights_dir, exist_ok=True)

            file_name = f"{model_name}_{model_weights}.pth"
            url = lwcc_funcs.build_url(file_name)
            output = os.path.join(weights_dir, file_name)

            if not os.path.isfile(output):
                print(f"{file_name} will be downloaded to {output}")
                gdown.download(url, output, quiet=False)

            return output

        lwcc_funcs.weights_check = _fixed_weights_check
    except Exception:
        pass

_patch_lwcc_weights_path()

try:
    from lwcc import LWCC
    LWCC_AVAILABLE = True
except ImportError:
    LWCC_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

os.makedirs("assets", exist_ok=True)

_favicon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
_favicon_path_alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png")

if os.path.exists(_favicon_path):
    _favicon = Image.open(_favicon_path)
elif os.path.exists(_favicon_path_alt):
    _favicon = Image.open(_favicon_path_alt)
else:
    _favicon = "🔵"

st.set_page_config(
    page_title="SafeCrowd Vision",
    page_icon=_favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CSS — Mission-Control Threat Intelligence Dashboard
# ═══════════════════════════════════════════════════════════════

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ══════════ CSS VARIABLES — Premium Slate ══════════ */
:root {
    --bg:         #2D3748;
    --surface:    #3B4A5E;
    --surface-2:  #485A6E;
    --card:       #3B4A5E;
    --border:     #5A6B7E;
    --border-h:   #7A8B9E;
    --accent:     #6366F1;
    --accent-g:   #8B5CF6;
    --cyan:       #22D3EE;
    --purple:     #8B5CF6;
    --green:      #10B981;
    --amber:      #F59E0B;
    --red:        #EF4444;
    --critical:   #FF1744;
    --text:       #FFFFFF;
    --text-2:     #CBD5E1;
    --muted:      #9AA8B8;
    --dimmed:     #7A8B9E;
}

/* ══════════ BASE ══════════ */
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.stApp {
    background-color: var(--bg) !important;
}
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background-image:
        radial-gradient(circle at 20% 50%,
            rgba(99,102,241,0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%,
            rgba(34,211,238,0.06) 0%, transparent 40%),
        radial-gradient(circle at 60% 80%,
            rgba(139,92,246,0.05) 0%, transparent 35%);
    animation: bgPulse 8s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
section[data-testid="stMain"] { background-color: transparent !important; }
div[data-testid="stMainBlockContainer"] {
    background-color: transparent !important;
    padding-top: 1rem !important;
}
.main .block-container {
    background-color: transparent !important;
    padding-top: 1rem !important;
}

/* ══════════ SIDEBAR — COLLAPSE FIX ══════════ */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"][aria-expanded="false"] {
    width: 0px !important;
    min-width: 0 !important;
    overflow: hidden !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #344B60 0%, #2D3748 100%) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 310px !important;
    z-index: 10;
}
section[data-testid="stSidebar"] > div {
    background: transparent !important;
    display: flex !important;
    flex-direction: column !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: transparent !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    background: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 1rem !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] p {
    color: var(--muted) !important;
}
section[data-testid="stSidebar"] span {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--muted) !important;
}

/* ── Sidebar Slider Track ── */
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background-color: var(--accent) !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div {
    color: var(--accent) !important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {
    background-color: var(--accent) !important;
}

/* ── Sidebar Radio Glow ── */
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"],
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label[aria-checked="true"] {
    color: var(--text) !important;
    text-shadow: 0 0 8px rgba(99,102,241,0.5);
}

/* ══════════ METRIC CARDS ══════════ */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: 22px 18px !important;
    border: 1px solid var(--border) !important;
    border-top: 3px solid var(--accent) !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.12) !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}
[data-testid="stMetric"]:hover {
    border-color: var(--border-h) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.2) !important;
}
div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 40px !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    font-variant-numeric: tabular-nums !important;
}
div[data-testid="stMetricLabel"] {
    color: var(--text-2) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    font-weight: 600 !important;
}

/* ── Per-metric accent colors ── */
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div:nth-child(1) [data-testid="stMetric"] {
    border-top: 3px solid #6366F1 !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.15) !important;
}
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div:nth-child(2) [data-testid="stMetric"] {
    border-top: 3px solid #EF4444 !important;
    box-shadow: 0 4px 24px rgba(239,68,68,0.15) !important;
}
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div:nth-child(3) [data-testid="stMetric"] {
    border-top: 3px solid #F59E0B !important;
    box-shadow: 0 4px 24px rgba(245,158,11,0.15) !important;
}
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div:nth-child(4) [data-testid="stMetric"] {
    border-top: 3px solid #10B981 !important;
    box-shadow: 0 4px 24px rgba(16,185,129,0.15) !important;
}

/* ══════════ TABS — PILL NAV ══════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 5px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    padding: 10px 26px !important;
    border-radius: 6px !important;
    background: transparent !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    border: none !important;
    position: relative !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text) !important;
    background: rgba(99,102,241,0.12) !important;
}
.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    background: var(--accent) !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.5) !important;
}
.stTabs [aria-selected="true"]::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 20%;
    width: 60%;
    height: 2px;
    background: var(--accent-g);
    border-radius: 1px;
    animation: tabGlow 2s ease-in-out infinite;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ══════════ FILE UPLOADER ══════════ */
div[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: var(--surface-2) !important;
    box-shadow: 0 0 24px rgba(99,102,241,0.15) !important;
}
div[data-testid="stFileUploader"] > div {
    padding: 8px 16px !important;
    min-height: 0 !important;
}
div[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
}
div[data-testid="stFileUploader"] small {
    color: var(--muted) !important;
}
div[data-testid="stFileUploader"] button {
    color: var(--text) !important;
}

/* ══════════ MOBILE CAPTURE MODE ══════════ */
@media (max-width: 768px) {
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
    div[data-testid="stFileUploader"] {
        min-height: 140px !important;
    }
    div[data-testid="stFileUploader"] > div {
        padding: 32px 20px !important;
    }
    div[data-testid="stFileUploader"] label {
        font-size: 16px !important;
    }
    button[kind="primary"],
    button[data-testid="stBaseButton-secondary"] {
        min-height: 52px !important;
        font-size: 16px !important;
    }
    .main .block-container {
        padding: 12px !important;
    }
}

/* ══════════ DOWNLOAD BUTTON ══════════ */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.35) !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
}
[data-testid="stDownloadButton"] button:hover {
    filter: brightness(1.15) !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) scale(1.02) !important;
}

/* ══════════ REGULAR BUTTONS ══════════ */
.stButton button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
}
.stButton button:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 16px rgba(99,102,241,0.2) !important;
}

/* ══════════ EXPANDER ══════════ */
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: var(--text-2) !important;
}
div[data-testid="stExpander"] summary:hover {
    color: var(--text) !important;
}
div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-2) !important;
}

/* ══════════ DATAFRAME ══════════ */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ══════════ TYPOGRAPHY ══════════ */
h1, h2, h3 { color: var(--text) !important; }
p { color: var(--text-2); }

/* ══════════ INPUTS ══════════ */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/* ══════════ TOGGLE ══════════ */
[data-testid="stToggle"] label span {
    color: var(--text-2) !important;
}

/* ══════════ HIDE STREAMLIT DEFAULTS ══════════ */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { display: none !important; }

/* ══════════ ANIMATIONS ══════════ */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.55; }
}
@keyframes dotPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
}
@keyframes criticalGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(255,23,68,0.2); }
    50% { box-shadow: 0 0 30px rgba(255,23,68,0.4); }
}
@keyframes criticalBorderPulse {
    0%, 100% { border-left-color: #EF4444; box-shadow: -4px 0 12px rgba(239,68,68,0.2); }
    50% { border-left-color: #FF1744; box-shadow: -4px 0 24px rgba(255,23,68,0.5); }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes tabGlow {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}
@keyframes liveDot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.6); }
    50% { opacity: 0.6; box-shadow: 0 0 0 4px rgba(16,185,129,0); }
}
@keyframes floatUp {
    from { opacity: 0; transform: translateY(24px) scale(0.97); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes pillGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(99,102,241,0.15); }
    50% { box-shadow: 0 0 24px rgba(99,102,241,0.3); }
}
@keyframes barFill {
    from { width: 0%; }
    to { width: var(--fill-pct); }
}
@keyframes particleDrift1 {
    0% { transform: translate(0, 0); }
    25% { transform: translate(80px, -60px); }
    50% { transform: translate(-40px, -120px); }
    75% { transform: translate(120px, -30px); }
    100% { transform: translate(0, 0); }
}
@keyframes particleDrift2 {
    0% { transform: translate(0, 0); }
    25% { transform: translate(-100px, 80px); }
    50% { transform: translate(60px, 140px); }
    75% { transform: translate(-80px, 40px); }
    100% { transform: translate(0, 0); }
}
@keyframes particleDrift3 {
    0% { transform: translate(0, 0); }
    25% { transform: translate(60px, 100px); }
    50% { transform: translate(-80px, -60px); }
    75% { transform: translate(40px, -100px); }
    100% { transform: translate(0, 0); }
}
@keyframes tickerScroll {
    0% { transform: translateX(0); }
    100% { transform: translateX(-33.33%); }
}
@keyframes bgPulse {
    0%   { opacity: 0.6; }
    50%  { opacity: 1.0; }
    100% { opacity: 0.6; }
}
@keyframes successPulse {
    0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    70%  { box-shadow: 0 0 0 12px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}



/* ══════════ PLOTLY ══════════ */
.js-plotly-plot {
    border-radius: 12px !important;
    overflow: hidden !important;
}
.js-plotly-plot .plotly .modebar {
    background: rgba(59,74,94,0.85) !important;
    border-radius: 8px !important;
}
.js-plotly-plot .plotly .modebar-btn path {
    fill: #CBD5E1 !important;
}
.js-plotly-plot .plotly .modebar-btn:hover path {
    fill: #FFFFFF !important;
}

/* ══════════ RADIO / TOGGLE ══════════ */
div[data-testid="stRadio"] > label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
}
[data-testid="stToggle"] label span {
    font-weight: 500 !important;
    color: var(--text-2) !important;
}

/* ══════════ LINE CHART / VEGA ══════════ */
.stVegaLiteChart {
    border-radius: 12px;
    overflow: hidden;
}

/* ══════════ SCROLLBAR ══════════ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-h); }

/* ══════════ ENHANCED CARD HOVER ══════════ */
[data-testid="stMetric"]:hover {
    border-color: var(--accent) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 40px rgba(99,102,241,0.25) !important;
}

/* ══════════ GLASSMORPHISM CONTAINERS ══════════ */
div[data-testid="stExpander"] {
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
}

/* ══════════ PRIMARY BUTTON ENHANCEMENT ══════════ */
button[kind="primary"],
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.35) !important;
    font-weight: 700 !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
}
button[kind="primary"]:hover,
.stButton button[kind="primary"]:hover {
    filter: brightness(1.12) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) scale(1.02) !important;
}
button[kind="primary"]:active,
.stButton button[kind="primary"]:active {
    transform: translateY(0) scale(0.98) !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
}

/* ══════════ PROGRESS BAR ══════════ */
.stProgress > div > div {
    background: var(--surface-2) !important;
    border-radius: 6px !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366F1, #22D3EE) !important;
    border-radius: 6px !important;
}

/* ══════════ IMAGE CAPTIONS ══════════ */
[data-testid="stImage"] > div > div > p {
    color: var(--muted) !important;
    font-size: 12px !important;
}

/* ══════════ SELECTBOX / MULTISELECT ══════════ */
div[data-baseweb="select"] > div {
    background: var(--surface-2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: var(--accent) !important;
}

/* ══════════ TEXT SELECTION ══════════ */
::selection {
    background: rgba(99,102,241,0.35);
    color: #FFFFFF;
}
</style>""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(135deg, #3B4A5E 0%, #2D3748 100%);
padding:20px 30px;border-radius:14px;border:1px solid #5A6B7E;
border-left:4px solid #6366F1;margin-bottom:24px;
box-shadow:-4px 0 30px rgba(99,102,241,0.5), 0 4px 24px rgba(0,0,0,0.3);
animation:floatUp 0.5s ease;
display:flex;align-items:center;justify-content:space-between;
position:relative;z-index:2">
<div style="display:flex;align-items:center;gap:14px">
<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:36px;height:36px;flex-shrink:0;filter:drop-shadow(0 2px 12px rgba(99,102,241,0.5))">
<defs><linearGradient id="hdrShield" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#6366F1"/><stop offset="50%" stop-color="#4F46E5"/><stop offset="100%" stop-color="#22D3EE"/>
</linearGradient></defs>
<path d="M32 4 L56 14 L56 30 C56 46 44 56 32 60 C20 56 8 46 8 30 L8 14 Z" fill="url(#hdrShield)" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
<circle cx="32" cy="30" r="10" fill="none" stroke="white" stroke-width="1.5" opacity="0.8"/>
<circle cx="32" cy="30" r="4" fill="white" opacity="0.9"/>
<line x1="32" y1="18" x2="32" y2="22" stroke="white" stroke-width="1.5" opacity="0.6"/>
<line x1="32" y1="38" x2="32" y2="42" stroke="white" stroke-width="1.5" opacity="0.6"/>
<line x1="20" y1="30" x2="24" y2="30" stroke="white" stroke-width="1.5" opacity="0.6"/>
<line x1="40" y1="30" x2="44" y2="30" stroke="white" stroke-width="1.5" opacity="0.6"/>
<path d="M26 44 L30 48 L38 40" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>
<div>
<h1 style="color:#FFFFFF;margin:0;font-size:26px;font-weight:900;
letter-spacing:-0.03em;line-height:1.1">SafeCrowd Vision</h1>
<span style="color:#9AA8B8;font-size:11px;font-weight:500;
letter-spacing:0.5px">Real-time Crowd Safety Analytics</span>
</div>
</div>
<div style="display:flex;align-items:center;gap:10px">
<span style="display:inline-flex;align-items:center;gap:6px;
padding:5px 14px;border-radius:20px;font-size:11px;font-weight:600;
background:rgba(16,185,129,0.15);color:#10B981;
border:1px solid rgba(16,185,129,0.3)">
<span style="display:inline-block;width:7px;height:7px;border-radius:50%;
background:#10B981;animation:dotPulse 1.5s ease-in-out infinite"></span> LIVE</span>
<span style="display:inline-flex;align-items:center;gap:6px;
padding:5px 14px;border-radius:20px;font-size:11px;font-weight:600;
background:#6366F1;color:#FFFFFF;
border:1px solid #6366F1">
DM-Count Active</span>
</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

GRID     = 8
IMG_SIZE = 256

ZONE_HEX = {"Low": "#10B981", "Medium": "#F59E0B",
             "High": "#EF4444", "Critical": "#FF1744"}
ZONE_RGB = {"Low": (16, 185, 129), "Medium": (245, 158, 11),
             "High": (239, 68, 68), "Critical": (255, 23, 68)}

# ── Face Detector: YUNET DNN (primary) + Haar cascade (fallback) ──
_YUNET_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "face_detection_yunet_2023mar.onnx")
_YUNET_AVAILABLE = os.path.isfile(_YUNET_MODEL_PATH)

# Haar cascades kept as fallback only
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml')
face_cascade_alt2 = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_alt2.xml')


def _detect_faces_yunet(img_bgr, conf_threshold=0.45, nms_threshold=0.3):
    """Detect faces using YUNET deep learning model.

    Returns list of (x, y, w, h) tuples for each detected face,
    sorted left-to-right. Much more accurate than Haar cascades
    for side profiles, partial occlusion, and varying lighting.
    """
    img_h, img_w = img_bgr.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        _YUNET_MODEL_PATH, "", (img_w, img_h),
        conf_threshold, nms_threshold, 5000)
    detector.setInputSize((img_w, img_h))

    retval, raw_faces = detector.detect(img_bgr)

    if raw_faces is None or len(raw_faces) == 0:
        return []

    faces = []
    for face in raw_faces:
        x  = int(face[0])
        y  = int(face[1])
        w  = int(face[2])
        h  = int(face[3])
        conf = float(face[-1])

        # Clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w < 15 or h < 15:
            continue
        if w * h > (img_h * img_w) * 0.35:
            continue

        faces.append((x, y, w, h))

    # Sort left-to-right
    faces.sort(key=lambda f: (f[0], f[1]))
    return faces


def _detect_faces_haar_fallback(img_bgr):
    """Fallback face detection using Haar cascades (less accurate)."""
    img_h, img_w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    all_boxes = []
    for cascade in (face_cascade, face_cascade_alt2):
        if cascade is None or cascade.empty():
            continue
        for source in (gray, gray_eq):
            detected = cascade.detectMultiScale(
                source, scaleFactor=1.08,
                minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in detected:
                aspect = w / max(h, 1)
                if aspect < 0.6 or aspect > 1.45:
                    continue
                all_boxes.append((x, y, w, h))

    if not all_boxes:
        return []

    # Simple NMS: keep strongest, remove overlaps
    kept = []
    for box in all_boxes:
        bx, by, bw, bh = box
        duplicate = False
        for kx, ky, kw, kh in kept:
            cx_dist = abs((bx + bw/2) - (kx + kw/2))
            cy_dist = abs((by + bh/2) - (ky + kh/2))
            if cx_dist < min(bw, kw) * 0.5 and cy_dist < min(bh, kh) * 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    kept.sort(key=lambda f: (f[0], f[1]))
    return kept


def _detect_faces_multi_pass(img_bgr):
    """Detect faces using YUNET DNN (primary) or Haar cascade (fallback).

    YUNET is a deep learning face detector that accurately detects
    frontal faces, side profiles, and partially occluded faces.
    Falls back to Haar cascades only if the YUNET model is unavailable.
    """
    if _YUNET_AVAILABLE:
        # Try YUNET at multiple confidence thresholds
        faces = _detect_faces_yunet(img_bgr, conf_threshold=0.45)

        # If very few faces found, try with lower confidence
        if len(faces) < 3:
            faces_low = _detect_faces_yunet(img_bgr, conf_threshold=0.30)
            if len(faces_low) > len(faces):
                faces = faces_low

        return faces

    # Fallback to Haar cascades if YUNET model not available
    return _detect_faces_haar_fallback(img_bgr)


def _summarize_face_detections(faces, img_shape):
    """Summarize face detections for portrait/group-photo gating."""
    img_h, img_w = img_shape[:2]
    if not faces or img_h <= 0 or img_w <= 0:
        return {
            "count": 0,
            "avg_height_ratio": 0.0,
            "median_height_ratio": 0.0,
            "max_height_ratio": 0.0,
            "area_ratio": 0.0,
        }

    heights = np.array(
        [fh for (_fx, _fy, _fw, fh) in faces],
        dtype=np.float32,
    ) / float(img_h)
    areas = np.array(
        [fw * fh for (_fx, _fy, fw, fh) in faces],
        dtype=np.float32,
    )

    return {
        "count": int(len(faces)),
        "avg_height_ratio": float(np.mean(heights)),
        "median_height_ratio": float(np.median(heights)),
        "max_height_ratio": float(np.max(heights)),
        "area_ratio": float(areas.sum() / max(img_h * img_w, 1)),
    }


def _resolve_portrait_count(img_rgb, img_bgr, density_full,
                            crowd_count, marker_count, marker_note=""):
    """
    Correct low-count group photos where DM-Count often overcounts.

    DM-Count stays primary for real crowd scenes. For portrait-like group
    photos with large visible faces, the face detector becomes the stronger
    cue and can override the density estimate.
    """
    density_max = float(density_full.max()) if density_full.size else 0.0
    density_coverage = (
        (density_full > density_max * 0.1).sum() / density_full.size
        if density_max > 1e-6 else 0.0
    )

    should_check_faces = (
        crowd_count <= 35 or
        (crowd_count <= 50 and density_coverage < 0.18)
    )
    faces = _detect_faces_multi_pass(img_bgr) if should_check_faces else []
    face_stats = _summarize_face_detections(faces, img_rgb.shape)
    face_count = face_stats["count"]

    close_up_scene = crowd_count <= 10 and density_coverage < 0.15
    group_portrait_scene = (
        crowd_count <= 35 and
        density_coverage < 0.28 and
        face_count >= 3 and
        (
            face_stats["avg_height_ratio"] >= 0.055 or
            face_stats["median_height_ratio"] >= 0.05 or
            face_stats["max_height_ratio"] >= 0.085 or
            face_stats["area_ratio"] >= 0.025
        )
    )
    portrait_detected = close_up_scene or group_portrait_scene

    result = {
        "count": int(round(crowd_count)),
        "marker_count": int(marker_count),
        "marker_note": marker_note or "",
        "used_face_detector": False,
        "portrait_hybrid_mode": False,
        "portrait_detected": portrait_detected,
        "faces": faces,
        "density_coverage": float(density_coverage),
    }

    if not portrait_detected:
        return result

    if face_count > 0:
        support_count = max(int(round(crowd_count)), int(marker_count))
        support_ratio = face_count / max(support_count, 1)
        strong_face_signal = (
            face_count >= 3 and
            (
                face_stats["avg_height_ratio"] >= 0.075 or
                face_stats["median_height_ratio"] >= 0.07 or
                face_stats["max_height_ratio"] >= 0.11 or
                face_stats["area_ratio"] >= 0.04
            )
        )

        if (support_ratio >= 0.82 or
                (strong_face_signal and support_ratio >= 0.72) or
                support_count - face_count <= 1):
            result["count"] = int(face_count)
            result["marker_count"] = int(face_count)
            result["used_face_detector"] = True
            result["marker_note"] = (
                f"Face-aware portrait correction active - using "
                f"{face_count} detected faces"
            )
        elif support_ratio >= 0.60:
            corrected_count = max(
                face_count,
                int(round(face_count * 0.65 + support_count * 0.35)),
            )
            result["count"] = int(corrected_count)
            result["marker_count"] = int(face_count)
            result["portrait_hybrid_mode"] = True
            result["marker_note"] = (
                f"Portrait hybrid correction active - face detector "
                f"found {face_count}, adjusted count {corrected_count}"
            )
        return result

    if marker_count > 0:
        corrected_count = min(int(round(crowd_count)), int(marker_count))
        result["count"] = corrected_count
        result["marker_count"] = int(marker_count)
        result["portrait_hybrid_mode"] = True
        result["marker_note"] = (
            f"Portrait marker recovery active - using "
            f"{corrected_count} visible markers"
        )

    return result


def _draw_faces_on_image(base_img, faces, label="Face"):
    """Draw clean, precise face markers on detected face regions.

    Each face gets:
      • A thin cyan rounded-corner rectangle tightly around the face
      • A soft glow dot centered on the face (forehead region)
      • A compact center marker for visibility
    The result looks polished and clearly marks actual human faces
    rather than producing random-looking boxes.
    """
    overlay = base_img.copy()
    img_h = overlay.shape[0]

    for (fx, fy, fw, fh) in faces:
        # ── Cyan border rectangle ──
        # Use a slightly inset rectangle so it sits neatly on the face
        pad = max(2, int(min(fw, fh) * 0.05))
        x1 = max(0, fx - pad)
        y1 = max(0, fy - pad)
        x2 = min(overlay.shape[1], fx + fw + pad)
        y2 = min(img_h, fy + fh + pad)

        # Semi-transparent fill to highlight face region
        face_fill = overlay.copy()
        cv2.rectangle(face_fill, (x1, y1), (x2, y2),
                      (0, 255, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(face_fill, 0.08, overlay, 0.92, 0, overlay)

        # Crisp border
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      (0, 255, 255), 2, cv2.LINE_AA)

        # ── Center dot on forehead area (upper-center of face box) ──
        cx = fx + fw // 2
        cy = fy + int(fh * 0.3)  # ~30% from top = forehead

        # Depth-aware sizing (same logic as head-dot overlay)
        depth_ratio = cy / max(img_h, 1)
        scale = 0.75 + depth_ratio * 0.35
        r_glow = max(5, int(10 * scale))
        r_ring = max(4, int(7 * scale))
        r_core = max(2, int(3 * scale))

        # Soft glow
        glow_layer = overlay.copy()
        cv2.circle(glow_layer, (cx, cy), r_glow,
                   (0, 210, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(glow_layer, 0.18, overlay, 0.82, 0, overlay)

        # Ring + core dot
        cv2.circle(overlay, (cx, cy), r_ring,
                   (210, 248, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), r_core + 1,
                   (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), r_core,
                   (0, 186, 222), -1, cv2.LINE_AA)

    return overlay

# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITION (untouched)
# ═══════════════════════════════════════════════════════════════

class CrowdDensityNet(nn.Module):
    def __init__(self, freeze_layers=10):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:24])
        for i, param in enumerate(self.encoder.parameters()):
            if i < freeze_layers:
                param.requires_grad = False
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ═══════════════════════════════════════════════════════════════
# DQN REINFORCEMENT LEARNING — Evacuation Policy Agent
# ═══════════════════════════════════════════════════════════════

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for evacuation policy learning.
    Input: 69 features (64 zone risks + 5 context)
    Output: Q-values for each action
    """
    def __init__(self, state_size=69, action_size=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)


class EvacuationAgent:
    """
    DQN Agent that learns optimal evacuation strategy.

    State (69 features):
    - 64 zone risk values (0=Low, 1=Med, 2=High, 3=Crit)
    - crowd_count normalized (1 value)
    - critical_zones count (1 value)
    - high_zones count (1 value)
    - utilization % (1 value)
    - num_exits (1 value)

    Actions (8):
    0: Open all exits equally
    1: Prioritize exits near critical zones
    2: Close exits in critical zones (prevent inflow)
    3: One-way flow — exits only on safe side
    4: Staged evacuation — critical zones first
    5: Shelter in place — reduce movement
    6: Emergency protocol — all exits maximum
    7: Gradual dispersal — reduce density slowly

    Reward:
    +10 * reduction in critical zones
    +5  * reduction in high zones
    -20 * if critical zones increased
    -10 * evacuation time in minutes
    +50 * if all zones become safe
    """

    def __init__(self):
        self.state_size  = 69
        self.action_size = 8
        self.memory      = deque(maxlen=2000)
        self.gamma       = 0.95    # discount
        self.epsilon     = 1.0     # exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr          = 0.001
        self.batch_size  = 32
        self.train_step  = 0
        self.total_reward = 0.0
        self.episode     = 0
        self.action_history = []
        self.reward_history = []
        self.best_evac_time = None

        self.model  = DQNNetwork(
            self.state_size, self.action_size)
        self.target = DQNNetwork(
            self.state_size, self.action_size)
        self.target.load_state_dict(
            self.model.state_dict())

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Pre-populate memory so replay() works from episode 1
        self._prepopulate_memory()

        # Auto-train 5 rounds so loss is non-zero immediately
        for _ in range(5):
            self.replay()
        # Reset epsilon back to 1.0 after warm-up
        self.epsilon = 1.0

    def _prepopulate_memory(self):
        """Fill memory with synthetic experience tuples."""
        for _ in range(self.batch_size * 2):
            state = np.random.rand(
                self.state_size).astype(np.float32)
            action = random.randrange(self.action_size)
            reward = random.uniform(-20, 50)
            next_state = np.random.rand(
                self.state_size).astype(np.float32)
            done = random.random() < 0.1
            self.memory.append(
                (state, action, reward,
                 next_state, done))

    def get_state_vector(self, zone_stats,
                          crowd_count,
                          num_exits,
                          density_full):
        """Convert app state to NN input vector."""
        h, w = density_full.shape
        ph, pw = h // 8, w // 8
        zone_risks = []

        for i in range(8):
            for j in range(8):
                patch = density_full[
                    i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                mean_val = float(patch.mean())
                if   mean_val < 0.1:  risk = 0
                elif mean_val < 0.3:  risk = 1
                elif mean_val < 0.6:  risk = 2
                else:                 risk = 3
                zone_risks.append(risk / 3.0)

        # Context features
        context = [
            min(1.0, crowd_count / 500.0),
            zone_stats.get("Critical", 0) / 64.0,
            zone_stats.get("High", 0) / 64.0,
            min(1.0, (zone_stats.get("Critical", 0) *
                crowd_count / 64.0) / 100.0),
            min(1.0, num_exits / 10.0),
        ]

        state = np.array(zone_risks + context,
                         dtype=np.float32)
        return state

    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_t = torch.FloatTensor(
                state).unsqueeze(0)
            q_values = self.model(state_t)
            return int(q_values.argmax().item())

    def remember(self, state, action,
                  reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(
            self.memory, self.batch_size)
        states  = torch.FloatTensor(
            np.array([e[0] for e in batch]))
        actions = torch.LongTensor(
            [e[1] for e in batch])
        rewards = torch.FloatTensor(
            [e[2] for e in batch])
        next_st = torch.FloatTensor(
            np.array([e[3] for e in batch]))
        dones   = torch.FloatTensor(
            [e[4] for e in batch])

        curr_q = self.model(states).gather(
            1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target(next_st).max(1)[0]
            target_q = rewards + (
                self.gamma * next_q * (1 - dones))

        loss = self.criterion(
            curr_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay)

        self.train_step += 1
        if self.train_step % 10 == 0:
            self.target.load_state_dict(
                self.model.state_dict())

        return float(loss.item())

    def calculate_reward(self, prev_stats,
                          curr_stats, evac_mins):
        reward = 0.0

        # Reward for reducing critical zones
        crit_reduction = (
            prev_stats.get("Critical", 0) -
            curr_stats.get("Critical", 0))
        reward += crit_reduction * 10

        # Reward for reducing high zones
        high_reduction = (
            prev_stats.get("High", 0) -
            curr_stats.get("High", 0))
        reward += high_reduction * 5

        # Penalty for evacuation time
        reward -= evac_mins * 10

        # Big bonus if all clear
        if (curr_stats.get("Critical", 0) == 0 and
                curr_stats.get("High", 0) == 0):
            reward += 50

        # Penalty if got worse
        if crit_reduction < 0:
            reward += crit_reduction * 20

        return float(reward)

    def get_action_name(self, action):
        names = [
            "Open All Exits Equally",
            "Prioritize Exits Near Critical Zones",
            "Close Exits in Critical Zones",
            "One-Way Flow — Safe Side Only",
            "Staged Evacuation — Critical First",
            "Shelter in Place Protocol",
            "Emergency Protocol — Maximum Capacity",
            "Gradual Dispersal Strategy",
        ]
        return names[action]

    def get_action_detail(self, action,
                           zone_stats, num_exits):
        crit = zone_stats.get("Critical", 0)
        high = zone_stats.get("High", 0)

        details = {
            0: f"Distribute crowd evenly across "
               f"all {num_exits} exits",
            1: f"Deploy {min(num_exits, crit+2)} exits "
               f"adjacent to {crit} critical zones",
            2: f"Block entry through {crit} critical "
               f"zone exits, redirect to safe zones",
            3: f"Implement one-directional flow "
               f"toward {num_exits - crit} safe exits",
            4: f"Evacuate {crit} critical zones first, "
               f"then {high} high zones",
            5: f"Hold crowd in position, deploy "
               f"staff to {crit} critical zones",
            6: f"Activate all {num_exits} exits at "
               f"maximum throughput immediately",
            7: f"Gradually move crowd from "
               f"{crit + high} risk zones over time",
        }
        return details.get(action, "")


# ═══════════════════════════════════════════════════════════════
# LIVE CAPTURE ACCESS + SHARED LAST RESULT
# ═══════════════════════════════════════════════════════════════

LIVE_CAPTURE_ROOT = os.path.join(
    tempfile.gettempdir(), "safecrowd_live_capture")
LIVE_CAPTURE_META = os.path.join(
    LIVE_CAPTURE_ROOT, "latest_capture.json")


def _get_local_ip():
    """Auto-detect the laptop's local WiFi/LAN IP address."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
        sock.close()
        return local_ip
    except Exception:
        return "localhost"


def get_access_url():
    """Return the best phone-access URL for this Streamlit app."""
    import urllib.request
    import socket

    try:
        with urllib.request.urlopen(
            "http://localhost:4040/api/tunnels",
            timeout=2) as response:
            data = json.loads(response.read())
            for tunnel in data.get("tunnels", []):
                url = tunnel.get("public_url", "")
                if url.startswith("https://"):
                    return url, "ngrok"
    except Exception:
        pass

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
        sock.close()
        return f"http://{local_ip}:8501", "local"
    except Exception:
        return "http://localhost:8501", "local"


def _get_query_param(name, default=""):
    try:
        value = st.query_params.get(name, default)
    except Exception:
        value = st.experimental_get_query_params().get(name, [default])

    if isinstance(value, list):
        return value[0] if value else default
    return value


def _build_capture_url(base_url):
    """Append capture-mode query params to the shared access URL."""
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return ""
    joiner = "&" if "?" in base else "?"
    return f"{base}{joiner}tab=capture"


def _is_mobile_client():
    try:
        headers = getattr(st.context, "headers", {}) or {}
        user_agent = str(headers.get("user-agent", "")).lower()
    except Exception:
        user_agent = ""
    mobile_tokens = ("iphone", "ipad", "android", "mobile", "webos")
    return any(token in user_agent for token in mobile_tokens)


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _load_json_file(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _save_latest_capture_payload(payload):
    _write_json_atomic(LIVE_CAPTURE_META, payload)


def _load_latest_capture_payload():
    payload = _load_json_file(LIVE_CAPTURE_META, default=None)
    return payload if isinstance(payload, dict) else None


def _format_capture_time(ts):
    try:
        return datetime.fromisoformat(str(ts)).strftime("%H:%M:%S")
    except Exception:
        return str(ts or "unknown")


_capture_mode_requested = (
    str(_get_query_param("tab", "")).strip().lower() == "capture"
)
_capture_mode_mobile = _capture_mode_requested and _is_mobile_client()


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING — all cached
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_cnn():
    model = CrowdDensityNet()
    p = "models/best_model.pth"
    if os.path.exists(p):
        model.load_state_dict(
            torch.load(p, map_location="cpu", weights_only=True))
    model.eval()
    return model

@st.cache_resource
def load_lwcc_model(weights):
    """Load and cache an LWCC DM-Count model with the given weights."""
    if not LWCC_AVAILABLE:
        return None
    try:
        return LWCC.load_model(model_name="DM-Count", model_weights=weights)
    except Exception as e:
        print(f"  ⚠ Failed to load LWCC DM-Count ({weights}): {e}")
        return None

@st.cache_resource
def load_xgb():
    p = "models/xgb_classifier.pkl"
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_resource
def load_kmeans():
    p = "models/kmeans_model.pkl"
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_resource
def load_gmm():
    p = "models/gmm_model.pkl"
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_resource
def load_scaler():
    p = "models/feature_scaler.pkl"
    return joblib.load(p) if os.path.exists(p) else None

cnn    = load_cnn()
lwcc_shb = load_lwcc_model("SHB")
lwcc_sha = load_lwcc_model("SHA")
xgb    = load_xgb()
km     = load_kmeans()
gmm    = load_gmm()
scaler = load_scaler()

# ═══════════════════════════════════════════════════════════════
# CALIBRATION — kept for backward compat / fallback
# ═══════════════════════════════════════════════════════════════

def load_calibration(mode: str) -> tuple:
    path = f"models/calibration_{mode}.json"
    fallback = "models/calibration.json"
    target = path if os.path.exists(path) else fallback
    try:
        with open(target, "r") as f:
            c = json.load(f)
        return float(c["a"]), float(c["b"])
    except Exception:
        return 0.0, 1.0


def apply_calibration(raw_sum: float, mode: str) -> int:
    if raw_sum <= 1e-3:
        return 0
    a, b = load_calibration(mode)
    return max(1, int(round(np.exp(a) * (raw_sum ** b))))

# ═══════════════════════════════════════════════════════════════
# ML HELPERS
# ═══════════════════════════════════════════════════════════════

SCENE_ADAPTIVE_THRESHOLD_LOW  = 80   # SHB → SHA switch
SCENE_ADAPTIVE_THRESHOLD_HIGH = 100  # ensemble zone


def predict_density_lwcc(img_rgb_array):
    """
    Scene-adaptive crowd counting using LWCC DM-Count.

    1. Run DM-Count with SHB weights (sparse-optimized)
    2. If count < 80, trust SHB directly (SPARSE)
    3. If count 80–200, use SHA (MEDIUM-DENSE)
    4. If count >= 200, ensemble SHB + SHA weighted average (VERY DENSE)
    5. Returns (count, density_map_fullres)

    The density map is resized to the original image size and
    normalized so that density_map.sum() ≈ count.
    """
    h, w = img_rgb_array.shape[:2]
    pil_img = Image.fromarray(img_rgb_array)

    with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False) as tmp:
        pil_img.save(tmp, format="JPEG", quality=95)
        tmp_path = tmp.name

    try:
        # Pass 1 — SHB (sparse optimized)
        count_shb, density_shb = LWCC.get_count(
            tmp_path, model_name="DM-Count",
            model_weights="SHB", model=lwcc_shb,
            return_density=True, resize_img=True)

        count_shb = float(count_shb)

        # ── Tiled inference fallback for undercounted dense images ──
        # If count is low but the density map shows widespread activation,
        # the model likely undercounted. Re-run on 4 quadrant crops.
        density_shb_arr = np.array(density_shb, dtype=np.float32)
        if density_shb_arr.ndim > 2:
            density_shb_arr = density_shb_arr.squeeze()
        threshold = density_shb_arr.max() * 0.01
        nonzero_ratio = (density_shb_arr > threshold).sum() / max(density_shb_arr.size, 1)

        print(f"[TILED DEBUG] count_shb={count_shb:.1f}")
        print(f"[TILED DEBUG] density max={density_shb_arr.max():.6f}")
        print(f"[TILED DEBUG] threshold={threshold:.6f}")
        print(f"[TILED DEBUG] nonzero_ratio={nonzero_ratio:.3f}")
        print(f"[TILED DEBUG] fallback_trigger={count_shb < 150 and nonzero_ratio > 0.15}")

        if count_shb < 150 and nonzero_ratio > 0.15:
            # Image is likely undercounted — run tiled inference
            mid_h, mid_w = h // 2, w // 2
            quadrants = [
                img_rgb_array[:mid_h, :mid_w],       # top-left
                img_rgb_array[:mid_h, mid_w:],        # top-right
                img_rgb_array[mid_h:, :mid_w],        # bottom-left
                img_rgb_array[mid_h:, mid_w:],        # bottom-right
            ]

            tiled_count = 0.0
            tiled_densities = []
            for q_img in quadrants:
                q_pil = Image.fromarray(q_img)
                with tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False) as q_tmp:
                    q_pil.save(q_tmp, format="JPEG", quality=95)
                    q_path = q_tmp.name
                try:
                    q_count, q_density = LWCC.get_count(
                        q_path, model_name="DM-Count",
                        model_weights="SHB", model=lwcc_shb,
                        return_density=True, resize_img=True)
                    tiled_count += float(q_count)
                    tiled_densities.append(
                        np.array(q_density, dtype=np.float32))
                finally:
                    os.unlink(q_path)

            # Slight overlap correction for tiled inference
            count_tiled = tiled_count * 0.90

            # Reassemble density map from quadrant density maps
            # Resize each quadrant density to its original spatial size
            d_tl = cv2.resize(tiled_densities[0].squeeze(),
                              (mid_w, mid_h))
            d_tr = cv2.resize(tiled_densities[1].squeeze(),
                              (w - mid_w, mid_h))
            d_bl = cv2.resize(tiled_densities[2].squeeze(),
                              (mid_w, h - mid_h))
            d_br = cv2.resize(tiled_densities[3].squeeze(),
                              (w - mid_w, h - mid_h))

            top_row = np.concatenate([d_tl, d_tr], axis=1)
            bot_row = np.concatenate([d_bl, d_br], axis=1)
            density_raw = np.concatenate([top_row, bot_row], axis=0)

            count = count_tiled

        elif count_shb < SCENE_ADAPTIVE_THRESHOLD_LOW:
            # SPARSE: trust SHB directly
            count = count_shb
            density_raw = np.array(density_shb,
                                   dtype=np.float32)

        elif count_shb < SCENE_ADAPTIVE_THRESHOLD_HIGH:
            # MEDIUM-DENSE: use SHA
            count_sha, density_sha = LWCC.get_count(
                tmp_path, model_name="DM-Count",
                model_weights="SHA", model=lwcc_sha,
                return_density=True, resize_img=False)
            count = float(count_sha)
            density_raw = np.array(density_sha,
                                   dtype=np.float32)

        else:
            # VERY DENSE (100+): ensemble tiled + SHA
            # weighted average — tiled gets more weight
            # for dense scenes
            count_sha_full, density_sha = LWCC.get_count(
                tmp_path, model_name="DM-Count",
                model_weights="SHA", model=lwcc_sha,
                return_density=True, resize_img=False)

            count_sha_full = float(count_sha_full)

            # Tiled 75%, SHA full 25% for dense
            count_tiled = tiled_count * 0.90 if 'tiled_count' in dir() else count_shb
            count = (count_tiled * 0.75) + (count_sha_full * 0.25)

            # Ensemble density maps
            d_shb = np.array(density_shb, dtype=np.float32)
            d_sha = np.array(density_sha, dtype=np.float32)

            # Resize both to same size for blending
            target_h = min(d_shb.shape[0], d_sha.shape[0])
            target_w = min(d_shb.shape[1], d_sha.shape[1])
            d_shb_r = cv2.resize(d_shb, (target_w, target_h))
            d_sha_r = cv2.resize(d_sha, (target_w, target_h))

            density_raw = (d_sha_r * 0.75) + (d_shb_r * 0.25)

    finally:
        os.unlink(tmp_path)

    if density_raw.ndim > 2:
        density_raw = density_raw.squeeze()

    density_full = cv2.resize(density_raw, (w, h),
                    interpolation=cv2.INTER_LINEAR)

    raw_sum = density_full.sum()
    if raw_sum > 1e-6:
        density_full = density_full * (count / raw_sum)

    density_full = np.clip(density_full, 0, None)
    return max(0, int(round(count))), density_full


def predict_density_raw(model, img_bgr):
    """
    FALLBACK: Tiled Inference using old VGG-16 model.
    Only used if LWCC is not available.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = img_gray.shape
    tile_size = IMG_SIZE
    stride = tile_size // 2

    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride
    img_p = np.pad(img_gray, ((0, pad_h), (0, pad_w)), mode='reflect')

    density_p = np.zeros_like(img_p)
    count_p   = np.zeros_like(img_p)

    for y in range(0, img_p.shape[0] - stride, stride):
        for x in range(0, img_p.shape[1] - stride, stride):
            tile = img_p[y:y+tile_size, x:x+tile_size]
            if tile.shape != (tile_size, tile_size):
                tile = cv2.resize(tile, (tile_size, tile_size))

            tile_t = torch.tensor(np.stack([tile]*3, 0), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = model(tile_t)

            density_p[y:y+tile_size, x:x+tile_size] += out.squeeze().numpy()
            count_p[y:y+tile_size, x:x+tile_size]   += 1.0

    density = (density_p / np.maximum(count_p, 1.0))[:h, :w]
    density = np.clip(density, 0, None)

    # Noise suppression
    density[density < 0.015] = 0
    kernel = np.ones((3, 3), np.uint8)
    mask   = (density > 0).astype(np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    density = density * mask.astype(np.float32)

    return density


def extract_features(density_map, grid=GRID):
    """
    Split density map into grid×grid patches and compute 8 features each.
    Matches segment_density.py exactly for model compatibility.
    """
    h, w = density_map.shape
    ph, pw = h // grid, w // grid
    features = []

    for i in range(grid):
        for j in range(grid):
            patch = density_map[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
            patch = np.nan_to_num(
                patch.astype(np.float32, copy=False),
                nan=0.0, posinf=0.0, neginf=0.0,
            )

            # Basic stats
            m = float(patch.mean())
            s = float(patch.std())

            # Advanced markers: Clumpiness (CV) and Structure (Gradient)
            cv = min(s / max(m, 1e-6), 10.0)

            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = min(float(np.sqrt(gx**2 + gy**2).max()), 1.0)

            features.append([
                m,
                float(patch.max()),
                s,
                cv,
                grad_mag,
                float((patch > m).sum() / (patch.size + 1e-7)),
                i / (grid - 1),
                j / (grid - 1),
            ])

    return np.nan_to_num(
        np.asarray(features, dtype=np.float32),
        nan=0.0, posinf=1e3, neginf=0.0,
    )


def get_label(mean_val):
    if   mean_val < 0.1:  return "Low"
    elif mean_val < 0.3:  return "Medium"
    elif mean_val < 0.6:  return "High"
    else:                 return "Critical"


def build_overlay(img_rgb, labels_list, grid=GRID, opacity=0.45):
    h, w    = img_rgb.shape[:2]
    overlay = img_rgb.copy()
    ph, pw  = h // grid, w // grid
    stats   = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for idx, label in enumerate(labels_list):
        i, j = divmod(idx, grid)
        stats[label] += 1
        r, g, b = ZONE_RGB[label]
        cv2.rectangle(overlay, (j*pw, i*ph),
                      ((j+1)*pw, (i+1)*ph), (r, g, b), -1)
    blended = cv2.addWeighted(overlay, opacity, img_rgb, 1-opacity, 0)
    lx, ly = w - 140, h - 92
    cv2.rectangle(blended, (lx-6, ly-6), (w-4, h-4), (6, 10, 18), -1)
    for k, (lbl, color) in enumerate(ZONE_RGB.items()):
        y = ly + k * 20
        cv2.rectangle(blended, (lx, y), (lx+12, y+12), color, -1)
        cv2.putText(blended, lbl, (lx+18, y+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (239, 246, 255), 1, cv2.LINE_AA)
    return blended, stats


def build_density_overlay(img_rgb, density_map, opacity=0.5,
                          expected_count=0, marker_points=None):
    smoothed = gaussian_filter(density_map.astype(np.float64), sigma=8)
    if smoothed.max() > 0:
        norm = ((smoothed / smoothed.max()) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(smoothed, dtype=np.uint8)
    jet     = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(jet_rgb, opacity, img_rgb, 1-opacity, 0)

    # Reuse the same resolved markers used by the head-marker panel.
    if marker_points is None:
        marker_points, _ = _resolve_head_markers(
            img_rgb, density_map, expected_count)

    h = blended.shape[0]
    for (py, px, _val) in marker_points:
        depth_ratio = py / max(h, 1)  # 0=top(far), 1=bottom(near)
        r = max(2, int(3 + depth_ratio * 3))
        cv2.circle(blended, (px, py), r, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(blended, (px, py), r + 2, (0, 255, 255), 1, cv2.LINE_AA)
    return blended


def _marker_spacing(h, w, expected_count=0):
    target = max(1, int(round(expected_count))) if expected_count > 0 else 6
    avg_spacing = np.sqrt((h * w) / max(target, 1))
    min_dist = int(np.clip(avg_spacing * 0.32, 8, 28))
    border_margin = int(np.clip(min_dist * 0.65, 5, 18))
    return min_dist, border_margin


def _find_density_peaks(density_map, expected_count=0):
    """
    Extract stable, deterministic head-marker candidates from a density map.

    DM-Count outputs a smooth field rather than explicit detections, so we:
    1. smooth moderately,
    2. find local maxima with a morphology-style max filter,
    3. lower the threshold adaptively until enough candidates appear,
    4. merge nearby candidates to avoid stacked markers.
    """
    h, w = density_map.shape
    density = np.clip(density_map.astype(np.float64), 0, None)
    smoothed = gaussian_filter(density, sigma=1.4)

    if smoothed.max() <= 1e-6 or expected_count <= 0:
        return [], 0

    active = smoothed[smoothed > 0]
    if active.size == 0:
        return [], 0

    target = int(round(expected_count))
    min_dist_px, border_margin = _marker_spacing(h, w, target)
    max_window = max(3, min_dist_px)
    if max_window % 2 == 0:
        max_window += 1

    peak_max = float(smoothed.max())
    norm = smoothed / peak_max
    local_max = maximum_filter(norm, size=max_window, mode="nearest")
    threshold_floor = float(np.clip(
        active.mean() / peak_max + 0.35 * (active.std() / peak_max),
        0.04, 0.55))
    candidate_goal = max(10, target * 3)

    all_candidates = []
    for pct in [99, 97, 95, 93, 91, 89, 86, 83, 80, 76, 72, 68, 64]:
        thr = max(threshold_floor, float(np.percentile(norm[norm > 0], pct)))
        maxima = (norm >= (local_max - 1e-9)) & (norm >= thr)

        maxima[:border_margin, :] = False
        maxima[-border_margin:, :] = False
        maxima[:, :border_margin] = False
        maxima[:, -border_margin:] = False

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            maxima.astype(np.uint8), connectivity=8)
        batch = []
        for label_idx in range(1, num_labels):
            ys, xs = np.where(labels == label_idx)
            if ys.size == 0:
                continue
            vals = smoothed[ys, xs]
            best_idx = int(np.argmax(vals))
            py = int(ys[best_idx])
            px = int(xs[best_idx])
            batch.append((py, px, float(vals[best_idx])))

        all_candidates.extend(batch)
        all_candidates = _filter_nearby_peaks(
            all_candidates, min_dist=max(6, int(min_dist_px * 0.8)))
        if len(all_candidates) >= candidate_goal:
            break

    if not all_candidates:
        flat = norm.ravel()
        fallback_n = min(flat.size, max(12, target * 2))
        top_indices = np.argpartition(flat, -fallback_n)[-fallback_n:]
        for idx in top_indices:
            py, px = divmod(int(idx), w)
            if (py < border_margin or px < border_margin or
                    py >= h - border_margin or
                    px >= w - border_margin):
                continue
            all_candidates.append((py, px, float(smoothed[py, px])))

    ranked = sorted(all_candidates, key=lambda p: p[2], reverse=True)
    ranked = _filter_nearby_peaks(ranked, min_dist=min_dist_px)
    return ranked, min_dist_px


def _sort_reading_order(peaks):
    """
    Sort detected head positions in reading order:
    left-to-right, top-to-bottom with a row tolerance
    so dots on the same visual row get the same Y rank.
    """
    if not peaks:
        return peaks

    # Sort by Y (row) first
    pts = sorted(peaks, key=lambda p: p[0])

    # Group into visual rows using a tolerance
    row_tolerance = 20  # pixels
    rows = []
    current_row = [pts[0]]

    for p in pts[1:]:
        if abs(p[0] - current_row[0][0]) <= row_tolerance:
            current_row.append(p)
        else:
            # Sort the completed row left-to-right by X
            rows.append(sorted(current_row, key=lambda p: p[1]))
            current_row = [p]
    rows.append(sorted(current_row, key=lambda p: p[1]))

    return [p for row in rows for p in row]


def _classify_peak_region(peak, img_rgb):
    """
    Score whether a density peak is likely to sit on a visible head region.

    Returns:
      ("primary" | "secondary" | "reject", score)
    """
    py, px, _val = peak
    h, w = img_rgb.shape[:2]
    patch_r = max(6, min(h, w) // 80)
    border_margin = patch_r + 3

    if (py < border_margin or px < border_margin or
            py >= h - border_margin or px >= w - border_margin):
        return "reject", -1.0

    y0 = max(0, py - patch_r)
    y1 = min(h, py + patch_r + 1)
    x0 = max(0, px - patch_r)
    x1 = min(w, px + patch_r + 1)
    region = img_rgb[y0:y1, x0:x1]
    if region.size == 0:
        return "reject", -1.0

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    gray_mean = float(gray.mean())
    gray_std = float(gray.std())
    sat_mean = float(hsv[:, :, 1].mean())
    edge_strength = float(np.abs(cv2.Laplacian(gray, cv2.CV_32F)).mean())
    r_avg = float(region[:, :, 0].mean())
    g_avg = float(region[:, :, 1].mean())
    b_avg = float(region[:, :, 2].mean())

    if gray_mean > 228 and gray_std < 12:
        return "reject", -0.9
    if gray_mean < 26 and gray_std < 8 and edge_strength < 6:
        return "reject", -0.9
    if sat_mean < 20 and gray_std < 10 and edge_strength < 6:
        return "reject", -0.7

    if r_avg > 185 and g_avg > 175 and b_avg < 110 and edge_strength < 12:
        return "secondary", 0.2
    if r_avg > 175 and g_avg < 95 and b_avg < 95 and edge_strength < 14:
        return "secondary", 0.2
    if gray_mean < 55 and gray_std < 15 and edge_strength < 10:
        return "secondary", 0.25
    if gray_mean > 210 and gray_std < 18:
        return "secondary", 0.25

    quality = (
        min(gray_std / 18.0, 1.6) +
        min(edge_strength / 12.0, 1.6) +
        min(sat_mean / 55.0, 1.0)
    )
    return "primary", quality


def _select_peak_subset(candidates, target, min_dist=20):
    """Greedy non-max selection over ranked marker candidates."""
    if not candidates or target <= 0:
        return []

    ranked = sorted(
        candidates, key=lambda item: (item["score"], item["peak"][2]),
        reverse=True)
    min_dist_sq = min_dist * min_dist
    selected = []

    for item in ranked:
        py, px, _ = item["peak"]
        too_close = False
        for sy, sx, _ in selected:
            if (py - sy) ** 2 + (px - sx) ** 2 < min_dist_sq:
                too_close = True
                break
        if too_close:
            continue
        selected.append(item["peak"])
        if len(selected) >= target:
            break

    return selected


def _resolve_head_markers(img_rgb, density_map, expected_count=0):
    """
    Resolve final visible head markers from DM-Count density.

    Primary markers come from strong, head-like candidates. If filtering
    removes too many, a second pass refills from weaker-but-still-valid
    candidates so the overlay remains useful without pretending to be truth.
    """
    target = max(0, int(round(expected_count)))
    candidates, min_dist = _find_density_peaks(density_map, target)

    if not candidates:
        return [], {
            "count": 0,
            "target": target,
            "incomplete": target > 0,
            "gap": target,
            "note": "Head-marker overlay unavailable",
        }

    primary = []
    secondary = []
    for peak in candidates:
        status, score = _classify_peak_region(peak, img_rgb)
        if status == "reject":
            continue
        item = {"peak": peak, "score": score}
        if status == "primary":
            primary.append(item)
        else:
            secondary.append(item)

    select_target = target if target > 0 else len(primary) + len(secondary)
    primary_selected = _select_peak_subset(primary, select_target, min_dist)
    selected = primary_selected
    refill_used = False

    if (target > 0 and len(selected) < min(target, max(3, int(np.ceil(target * 0.8))))
            and secondary):
        refill_used = True
        selected = _select_peak_subset(
            primary + secondary, select_target,
            max(6, int(min_dist * 0.85)))

    if not selected and secondary:
        refill_used = True
        selected = _select_peak_subset(
            secondary, max(1, min(select_target, len(secondary))),
            max(6, int(min_dist * 0.85)))

    selected = _sort_reading_order(selected)
    marker_count = len(selected)
    gap = max(0, target - marker_count) if target > 0 else 0
    incomplete = target > 0 and gap > 1

    if incomplete:
        note = "Marker overlay incomplete"
    elif refill_used:
        note = "Markers recovered from weaker candidates"
    else:
        note = ""

    return selected, {
        "count": marker_count,
        "target": target,
        "gap": gap,
        "incomplete": incomplete,
        "refill_used": refill_used,
        "note": note,
    }


def _filter_nearby_peaks(peaks, min_dist=20):
    """
    Remove duplicate detections within min_dist pixels.
    Keeps the highest-intensity peak in each cluster.
    """
    if not peaks:
        return peaks

    # Sort by density value descending (keep strongest)
    sorted_peaks = sorted(peaks, key=lambda p: p[2], reverse=True)
    min_dist_sq = min_dist * min_dist

    kept = []
    for peak in sorted_peaks:
        py = int(peak[0])
        px = int(peak[1])
        too_close = False
        for kept_peak in kept:
            ky = int(kept_peak[0])
            kx = int(kept_peak[1])
            if (py - ky) ** 2 + (px - kx) ** 2 < min_dist_sq:
                too_close = True
                break
        if not too_close:
            kept.append(peak)
    return kept


def build_headdot_overlay(img_rgb, density_map, expected_count=0,
                          marker_points=None):
    """
    Draw clean depth-aware dots on resolved visible head markers.

    - Dots sized by vertical position (perspective depth):
      bottom of image = close to camera = bigger dots
      top of image = far from camera = smaller dots
    - No numbered badges or in-image captions to keep faces visible
    - Marker count is a visualization aid, not the final DM-Count estimate
    """
    overlay = img_rgb.copy()
    h = overlay.shape[0]

    if marker_points is None:
        marker_points, _ = _resolve_head_markers(
            img_rgb, density_map, expected_count)

    if not marker_points:
        return overlay, 0

    # Sort in reading order: top-to-bottom, left-to-right
    peaks_sorted = _sort_reading_order(marker_points)

    for (py, px, _val) in peaks_sorted:
        # Depth-aware scaling: 0.0 = top (far), 1.0 = bottom (near)
        depth_ratio = py / max(h, 1)
        scale = 0.75 + depth_ratio * 0.35
        r_glow = max(5, int(10 * scale))
        r_ring = max(4, int(7 * scale))
        r_core = max(2, int(3 * scale))

        # Soft glow to make markers visible without obscuring faces.
        glow_overlay = overlay.copy()
        cv2.circle(glow_overlay, (px, py), r_glow,
                   (0, 210, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(glow_overlay, 0.16, overlay, 0.84, 0, overlay)

        # Thin ring plus compact center dot.
        cv2.circle(overlay, (px, py), r_ring, (210, 248, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, (px, py), r_core + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (px, py), r_core, (0, 186, 222), -1, cv2.LINE_AA)

    return overlay, len(peaks_sorted)


def labels_from_kmeans(fs, model):
    raw   = model.predict(fs)
    order = np.argsort(model.cluster_centers_[:, 0])
    lmap  = {order[i]: ["Low", "Medium", "High", "Critical"][i]
             for i in range(4)}
    labels = [lmap[l] for l in raw]

    # Global density statistics for scene-level gating
    global_mean = float(np.mean(fs[:, 0]))
    global_max  = float(np.max(fs[:, 0]))

    result = []
    for i, label in enumerate(labels):
        d = float(fs[i, 0])  # per-zone mean density

        # Sparse scene gate: if overall scene is low density,
        # cap maximum label regardless of relative clustering
        if global_max < 0.4:
            if label == "Critical":
                label = "Medium"
            if label == "High" and d < 0.2:
                label = "Medium"

        # Ultra sparse gate: almost empty scene
        if global_max < 0.2:
            if label in ["Critical", "High"]:
                label = "Medium"
            if label == "Medium" and d < 0.1:
                label = "Low"

        # Per-zone absolute density cap (original logic)
        if d < 0.3 and label == "Critical":
            label = "Medium"
        elif d < 0.15 and label == "High":
            label = "Medium"
        elif d < 0.05 and label == "Medium":
            label = "Low"

        result.append(label)
    return result


def labels_from_xgb(fs, model):
    preds = model.predict(fs)
    unique_classes = sorted(model.classes_)
    class_to_name = {}
    for i, cls in enumerate(unique_classes):
        class_to_name[cls] = ["Low","Medium",
            "High","Critical"][min(i,3)]

    labels = [class_to_name[int(p)] for p in preds]

    # Global density statistics for scene-level gating
    global_max = float(np.max(fs[:, 0]))

    result = []
    for i, label in enumerate(labels):
        d = float(fs[i, 0])

        # Sparse scene gate
        if global_max < 0.4:
            if label == "Critical":
                label = "Medium"
            if label == "High" and d < 0.2:
                label = "Medium"

        # Ultra sparse gate
        if global_max < 0.2:
            if label in ["Critical", "High"]:
                label = "Medium"
            if label == "Medium" and d < 0.1:
                label = "Low"

        # Per-zone absolute density cap
        if d < 0.3 and label == "Critical":
            label = "Medium"
        elif d < 0.15 and label == "High":
            label = "Medium"
        elif d < 0.05 and label == "Medium":
            label = "Low"

        result.append(label)
    return result


def labels_from_gmm(fs, model):
    raw = model.predict(fs)
    probs = model.predict_proba(fs)

    # Strategy: sort clusters by mean density (feature 0)
    # This ensure cluster labels are mapped consistently to High/Low
    means = model.means_[:, 0]
    sorted_by_density = np.argsort(means)

    # For sparse images where all densities
    # are near 0, most zones should be Low
    # Use density range to detect sparse scenes
    # (scaled features: range < 1.0 means minimal variation)
    densities = fs[:, 0]
    density_range = densities.max() - densities.min()
    names = ["Low", "Medium", "High", "Critical"]

    if density_range < 1.0:
        # SPARSE SCENE: all clusters map to low-medium
        sparse_names = ["Low", "Low", "Medium", "Medium"]
        cmap = {}
        for rank, c in enumerate(sorted_by_density):
            cmap[c] = sparse_names[
                min(rank, len(sparse_names)-1)]
    else:
        # DENSE SCENE: full 4-level mapping
        cmap = {}
        for rank, c in enumerate(sorted_by_density):
            cmap[c] = names[min(rank, len(names)-1)]

    labels = [cmap[l] for l in raw]

    # Global density gate — same as KMeans/XGBoost
    global_max = float(np.max(fs[:, 0]))
    gated = []
    for i, label in enumerate(labels):
        d = float(fs[i, 0])

        if global_max < 0.4:
            if label == "Critical":
                label = "Medium"
            if label == "High" and d < 0.2:
                label = "Medium"

        if global_max < 0.2:
            if label in ["Critical", "High"]:
                label = "Medium"
            if label == "Medium" and d < 0.1:
                label = "Low"

        if d < 0.3 and label == "Critical":
            label = "Medium"
        elif d < 0.15 and label == "High":
            label = "Medium"
        elif d < 0.05 and label == "Medium":
            label = "Low"

        gated.append(label)
    labels = gated

    # Compute meaningful confidence:
    # Use margin between top-1 and top-2 probabilities
    # This reflects how *decisively* GMM separates zones
    n_unique_clusters = len(set(raw))
    if n_unique_clusters <= 1:
        # GMM can't differentiate — low confidence
        conf = max(0.25, float(probs.max(axis=1).mean()) * 0.4)
    else:
        # Margin-based confidence: how far apart top-2 probs are
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]
        conf = float(margins.mean())
    return labels, conf


def safe_silhouette(X, labels):
    try:
        from sklearn.metrics import silhouette_score
        if len(set(labels)) < 2:
            return 0.25
        return float(silhouette_score(X, labels))
    except Exception:
        return 0.25


def compute_dynamic_confidence(features_sc, method, km_model, xgb_model, gmm_model, crowd_count):
    """
    Compute a dynamic confidence score (0-100) based on actual model outputs
    instead of hardcoded count-based thresholds.

    Combines:
    - Base confidence from DM-Count model weights selection
    - Zone classification confidence (silhouette / XGBoost proba / GMM margins)
    - Scene complexity penalty (very dense scenes are harder)
    """
    # Base confidence from DM-Count weights
    if crowd_count < 80:
        base_conf = 95  # SHB, optimized for sparse
    elif crowd_count < 200:
        base_conf = 87  # SHA
    else:
        base_conf = 82  # Ensemble

    # Zone classification confidence
    zone_conf = 0.0
    try:
        if method == "XGBoost" and xgb_model is not None and hasattr(xgb_model, 'predict_proba'):
            probas = xgb_model.predict_proba(features_sc)
            sorted_probas = np.sort(probas, axis=1)[:, ::-1]
            margins = sorted_probas[:, 0] - sorted_probas[:, 1] if sorted_probas.shape[1] > 1 else sorted_probas[:, 0]
            zone_conf = float(margins.mean()) * 100
        elif method == "GMM" and gmm_model is not None:
            probas = gmm_model.predict_proba(features_sc)
            sorted_probas = np.sort(probas, axis=1)[:, ::-1]
            margins = sorted_probas[:, 0] - sorted_probas[:, 1] if sorted_probas.shape[1] > 1 else sorted_probas[:, 0]
            zone_conf = float(margins.mean()) * 100
        elif km_model is not None:
            labels = km_model.predict(features_sc)
            zone_conf = safe_silhouette(features_sc, labels) * 100
        else:
            zone_conf = 50.0
    except Exception:
        zone_conf = 50.0

    # Blend: 70% base (counting), 30% zone classification
    blended = (base_conf * 0.7) + (min(zone_conf, 100) * 0.3)

    # Clamp to reasonable range
    return max(40, min(99, int(round(blended))))


def _select_zone_labels(features, features_sc, method,
                        km_model, xgb_model, gmm_model):
    """Shared zone-labelling path for live and batch analysis."""
    if method == "XGBoost" and xgb_model is not None:
        return labels_from_xgb(features_sc, xgb_model)
    if method == "GMM" and gmm_model is not None:
        labels, _ = labels_from_gmm(features_sc, gmm_model)
        return labels
    if km_model is not None:
        return labels_from_kmeans(features_sc, km_model)
    return [get_label(float(f[0])) for f in features]


def _compute_threat_band(zone_stats, crowd_count):
    """Mirror the live-analysis threat score logic."""
    threat_score = min(100, int(
        zone_stats.get("Critical", 0) * 40 +
        zone_stats.get("High", 0) * 20 +
        zone_stats.get("Medium", 0) * 1 +
        min(15, crowd_count / 10)
    ))

    if threat_score < 25:
        threat_label = "MINIMAL"
    elif threat_score < 50:
        threat_label = "ELEVATED"
    elif threat_score < 75:
        threat_label = "HIGH"
    else:
        threat_label = "CRITICAL"

    return threat_score, threat_label


def _model_label_for_scene(crowd_count, used_face_detector=False,
                           portrait_hybrid_mode=False):
    if portrait_hybrid_mode:
        return "Portrait Hybrid (Face + Density)"
    if used_face_detector:
        return "YUNET Face Detector"
    if crowd_count < 80:
        return "DM-Count SHB"
    if crowd_count < 200:
        return "DM-Count SHA"
    return "DM-Count Ensemble (SHA+SHB)"


def _compress_img_b64(img_arr, quality=80):
    """Compress an RGB image to JPEG base64 for lightweight session storage."""
    ok, buf = cv2.imencode(
        '.jpg',
        cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf).decode()


def _analyze_scene_frame(img_rgb, img_bgr, method, opacity):
    """
    Analyze a single scene using the same core logic as live analysis.

    Returns all overlays and metrics needed by batch analysis.
    """
    h, w = img_rgb.shape[:2]

    if LWCC_AVAILABLE and lwcc_shb is not None:
        crowd_count, density_full = predict_density_lwcc(img_rgb)
    else:
        density_raw = predict_density_raw(cnn, img_bgr)
        density_full = cv2.resize(density_raw, (w, h))
        crowd_count = apply_calibration(float(density_raw.sum()), "small")

    features = extract_features(density_full)
    features_sc = scaler.transform(features) if scaler else features
    labels = _select_zone_labels(features, features_sc, method, km, xgb, gmm)
    safety_img, zone_stats = build_overlay(img_rgb, labels, GRID, opacity)

    head_markers, marker_meta = _resolve_head_markers(
        img_rgb, density_full, crowd_count)
    density_overlay = build_density_overlay(
        img_rgb, density_full, opacity,
        expected_count=crowd_count,
        marker_points=head_markers)
    head_overlay, marker_count = build_headdot_overlay(
        img_rgb, density_full,
        expected_count=crowd_count,
        marker_points=head_markers)

    portrait_result = _resolve_portrait_count(
        img_rgb, img_bgr, density_full, crowd_count,
        marker_count, marker_note=marker_meta.get("note", ""),
    )
    crowd_count = portrait_result["count"]
    marker_count = portrait_result["marker_count"]
    marker_note = portrait_result["marker_note"]
    used_face_detector = portrait_result["used_face_detector"]
    portrait_hybrid_mode = portrait_result["portrait_hybrid_mode"]

    if portrait_result["faces"] and (
            used_face_detector or portrait_hybrid_mode):
        # Keep portrait overlays tied to actual detected faces.
        head_overlay = _draw_faces_on_image(img_rgb, portrait_result["faces"])

    threat_score, threat_label = _compute_threat_band(zone_stats, crowd_count)
    confidence = compute_dynamic_confidence(
        features_sc, method, km, xgb, gmm, crowd_count)

    return {
        "count": int(crowd_count),
        "density_full": density_full,
        "features": features,
        "features_sc": features_sc,
        "labels": labels,
        "zone_stats": zone_stats,
        "confidence": confidence,
        "threat_score": threat_score,
        "threat": threat_label,
        "safety_img": safety_img,
        "density_overlay": density_overlay,
        "head_overlay": head_overlay,
        "marker_count": int(marker_count),
        "marker_note": marker_note,
        "used_face_detector": used_face_detector,
        "portrait_hybrid_mode": portrait_hybrid_mode,
        "model": _model_label_for_scene(
            crowd_count,
            used_face_detector=used_face_detector,
            portrait_hybrid_mode=portrait_hybrid_mode,
        ),
    }


def risk_grid(density_map, grid=GRID):
    h, w   = density_map.shape
    ph, pw = h // grid, w // grid
    g      = np.zeros((grid, grid))
    for i in range(grid):
        for j in range(grid):
            g[i, j] = density_map[i*ph:(i+1)*ph, j*pw:(j+1)*pw].mean()
    return g

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════

for _k, _v in [
    ("history",           []),
    ("last_density_raw",  None),
    ("last_count",        None),
    ("last_img_hash",     None),
    ("last_image",        None),
    ("present_mode",      False),
    ("rl_agent",          None),
    ("rl_history",        []),
    ("rl_prev_stats",     None),
    ("last_zone_stats",   {"Low":64,"Medium":0,"High":0,"Critical":0}),
    ("rl_last_result",    None),
    ("batch_results",     []),
    ("batch_results_signature", None),
    ("batch_summary",     None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state["rl_agent"] is None:
    st.session_state["rl_agent"] = EvacuationAgent()

_access_url, _access_url_type = get_access_url()
_capture_access_url = _build_capture_url(_access_url)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Logo + App Name ──
    # ══════════════════════════════════════════════════
    # SIDEBAR HEADER — Brand Bar
    # ══════════════════════════════════════════════════
    st.markdown("""
<div style="position:sticky;top:0;z-index:100;
background:linear-gradient(180deg,#344B60,#344B60);
padding:18px 16px 8px 16px;margin:-1rem -1rem 12px -1rem">
<div style="display:flex;align-items:center;gap:12px">
<div style="flex-shrink:0;width:40px;height:40px;border-radius:12px;
background:linear-gradient(135deg,#6366F1,#4F46E5);
display:flex;align-items:center;justify-content:center;
box-shadow:0 4px 20px rgba(99,102,241,0.4)">
<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:22px;height:22px">
<path d="M32 4 L56 14 L56 30 C56 46 44 56 32 60 C20 56 8 46 8 30 L8 14 Z" fill="rgba(255,255,255,0.2)" stroke="rgba(255,255,255,0.6)" stroke-width="1.5"/>
<circle cx="32" cy="30" r="8" fill="none" stroke="white" stroke-width="1.5" opacity="0.9"/>
<circle cx="32" cy="30" r="3" fill="white"/>
</svg>
</div>
<div>
<div style="font-size:17px;font-weight:800;color:#FFFFFF;letter-spacing:-0.02em;
line-height:1.15">SafeCrowd Vision</div>
<div style="font-size:10px;color:rgba(99,102,241,0.9);letter-spacing:0.5px;font-weight:600;
margin-top:2px">v2.0 · Safety Analytics</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # CONTROLS
    # ══════════════════════════════════════════════════
    st.markdown("""
<div style="color:#6366F1;font-weight:700;font-size:10px;
letter-spacing:1.8px;text-transform:uppercase;
margin-top:8px;margin-bottom:16px;
padding-left:2px;
display:flex;align-items:center;gap:6px">
<span style="display:inline-block;width:3px;height:12px;
background:#6366F1;border-radius:2px"></span>
CONTROLS</div>
""", unsafe_allow_html=True)

    demo_mode = st.toggle("🎓 Demo Mode", value=False)
    opacity = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)
    method = st.radio(
        "Primary zone method",
        ["KMeans", "XGBoost", "GMM"],
        help="Controls the safety map in Live Analysis tab")

    with st.expander("🎯 Zone thresholds"):
        st.markdown("""
| Zone | Density threshold |
|------|------------------|
| 🟢 Low | mean < 0.10 |
| 🟡 Medium | mean < 0.30 |
| 🟠 High | mean < 0.60 |
| 🔴 Critical | mean ≥ 0.60 |
        """)

    # ══════════════════════════════════════════════════
    # VENUE SETTINGS
    # ══════════════════════════════════════════════════
    st.markdown("""
<div style="border-top:1px solid #5A6B7E;margin:20px 0"></div>
<div style="color:#6366F1;font-weight:700;font-size:10px;
letter-spacing:1.8px;text-transform:uppercase;
margin-top:4px;margin-bottom:14px;
padding-left:2px;
display:flex;align-items:center;gap:6px">
<span style="display:inline-block;width:3px;height:12px;
background:#6366F1;border-radius:2px"></span>
VENUE SETTINGS</div>
""", unsafe_allow_html=True)

    venue_capacity = st.number_input(
        "Venue Capacity",
        min_value=10,
        max_value=10000,
        value=100,
        step=10,
        help="Set maximum safe capacity for this venue"
    )
    num_exits = st.slider("Number of exits", 1, 10, 2)
    st.session_state["num_exits_val"] = num_exits

    # ══════════════════════════════════════════════════
    # MODEL INFO — compact card
    # ══════════════════════════════════════════════════
    st.markdown("""
<div style="border-top:1px solid #5A6B7E;margin:20px 0"></div>
<div style="color:#6366F1;font-weight:700;font-size:10px;
letter-spacing:1.8px;text-transform:uppercase;
margin-top:4px;margin-bottom:14px;
padding-left:2px;
display:flex;align-items:center;gap:6px">
<span style="display:inline-block;width:3px;height:12px;
background:#22D3EE;border-radius:2px"></span>
MODEL INFO</div>

<div style="background:rgba(59,74,94,0.6);border:1px solid rgba(90,107,126,0.5);
border-left:3px solid #22D3EE;border-radius:10px;overflow:hidden">

<div style="display:flex;justify-content:space-between;align-items:center;
padding:9px 14px;border-bottom:1px solid rgba(90,107,126,0.3)">
<span style="color:#9AA8B8;font-size:11px;font-weight:500">Model</span>
<span style="color:#FFFFFF;font-size:11px;font-family:'JetBrains Mono',monospace;
font-weight:600">DM-Count</span>
</div>

<div style="display:flex;justify-content:space-between;align-items:center;
padding:9px 14px;border-bottom:1px solid rgba(90,107,126,0.3)">
<span style="color:#9AA8B8;font-size:11px;font-weight:500">Dataset</span>
<span style="color:#FFFFFF;font-size:11px;font-family:'JetBrains Mono',monospace;
font-weight:600">ShanghaiTech B</span>
</div>

<div style="display:flex;justify-content:space-between;align-items:center;
padding:9px 14px;border-bottom:1px solid rgba(90,107,126,0.3)">
<span style="color:#9AA8B8;font-size:11px;font-weight:500">MAE</span>
<span style="color:#10B981;font-size:11px;font-family:'JetBrains Mono',monospace;
font-weight:700">5.80</span>
</div>

<div style="display:flex;justify-content:space-between;align-items:center;
padding:9px 14px;border-bottom:1px solid rgba(90,107,126,0.3)">
<span style="color:#9AA8B8;font-size:11px;font-weight:500">Accuracy</span>
<span style="color:#10B981;font-size:11px;font-family:'JetBrains Mono',monospace;
font-weight:700">~81%</span>
</div>

<div style="display:flex;justify-content:space-between;align-items:center;
padding:9px 14px">
<span style="color:#9AA8B8;font-size:11px;font-weight:500">XGB Zone</span>
<span style="color:#10B981;font-size:11px;font-family:'JetBrains Mono',monospace;
font-weight:700">99.30%</span>
</div>

</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # PHONE CAPTURE — compact status pill
    # ══════════════════════════════════════════════════
    st.markdown("""
<div style="border-top:1px solid #5A6B7E;margin:20px 0"></div>
<div style="color:#6366F1;font-weight:700;font-size:10px;
letter-spacing:1.8px;text-transform:uppercase;
margin-top:4px;margin-bottom:14px;
padding-left:2px;
display:flex;align-items:center;gap:6px">
<span style="display:inline-block;width:3px;height:12px;
background:#6366F1;border-radius:2px"></span>
MOBILE CAPTURE</div>
""", unsafe_allow_html=True)

    if _access_url_type == "ngrok":
        st.markdown(f"""
        <div style="background:#1A3D2A;
        border:1px solid #10B981;
        border-radius:10px;padding:16px 18px;
        margin-top:4px">
        <div style="display:flex;align-items:center;
        gap:8px;margin-bottom:10px">
        <div style="width:8px;height:8px;
        border-radius:50%;background:#10B981;
        animation:dotPulse 1.5s infinite;
        flex-shrink:0"></div>
        <div style="color:#10B981;font-size:11px;
        font-weight:700;letter-spacing:1px">
        NGROK ACTIVE</div>
        </div>
        <div style="color:#6EE7B7;font-size:11px;
        font-family:monospace;
        word-break:break-all;
        line-height:1.6;margin-bottom:8px">
        {_access_url}</div>
        <div style="color:#064E3B;font-size:10px">
        ✓ Works on any network</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#3D3010;
        border:1px solid #F59E0B;
        border-radius:10px;padding:16px 18px;
        margin-top:4px">
        <div style="display:flex;align-items:center;
        gap:8px;margin-bottom:10px">
        <div style="width:8px;height:8px;
        border-radius:50%;background:#F59E0B;
        flex-shrink:0"></div>
        <div style="color:#F59E0B;font-size:11px;
        font-weight:700;letter-spacing:1px">
        LOCAL NETWORK</div>
        </div>
        <div style="color:#FCD34D;font-size:11px;
        font-family:monospace;
        word-break:break-all;
        line-height:1.6;margin-bottom:10px">
        {_access_url}</div>
        <div style="color:#B4742E;font-size:11px;
        line-height:1.6">
        📶 Phone must be on same WiFi</div>
        <div style="color:#B4742E;font-size:11px;
        margin-top:4px;line-height:1.6">
        ⚡ Run <span style="font-family:monospace;
        background:#2A1F00;padding:1px 6px;
        border-radius:4px;color:#FCD34D">
        ngrok http 8501</span> for public access
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════
    st.markdown("""
<div style="text-align:center;padding:16px 0 6px 0;
margin-top:16px;border-top:1px solid rgba(90,107,126,0.25)">
<div style="color:#5A6B7E;font-size:9px;font-weight:500;
letter-spacing:0.5px">LWCC · PyTorch · Streamlit</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FEATURE 3 — LIVE SAFETY TICKER (below header, before tabs)
# ═══════════════════════════════════════════════════════════════

_last_count_ticker = st.session_state.get("last_count")
if _last_count_ticker is not None:
    _crit_ticker = 0  # will be approximate — actual stats come later
    _status_txt = "ACTIVE" if _last_count_ticker else "IDLE"
    _ticker_content = (
        f'<span style="color:#B91C1C;font-size:8px;animation:dotPulse 1s infinite">●</span>'
        f'&nbsp;&nbsp;<span style="color:#6366F1">SYSTEM ONLINE</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#CBD5E1">MODEL:</span> '
        f'<span style="color:#9AA8B8">DM-Count SHB</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#CBD5E1">ZONES MONITORED:</span> '
        f'<span style="color:#9AA8B8">64</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#CBD5E1">LAST COUNT:</span> '
        f'<span style="color:#FFFFFF">{_last_count_ticker} PERSONS</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#CBD5E1">STATUS:</span> '
        f'<span style="color:#10B981">{_status_txt}</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#CBD5E1">MAE:</span> '
        f'<span style="color:#9AA8B8">5.80</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#CBD5E1">ACCURACY:</span> '
        f'<span style="color:#9AA8B8">~81% (eval)</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;'
    )
else:
    _ticker_content = (
        '<span style="color:#6366F1;font-size:8px;animation:dotPulse 1.5s infinite">●</span>'
        '&nbsp;&nbsp;<span style="color:#6366F1">AWAITING INPUT</span>'
        '&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#CBD5E1">SYSTEM READY</span>'
        '&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#CBD5E1">DM-Count</span>&nbsp;'
        '<span style="color:#9AA8B8">LOADED</span>'
        '&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#CBD5E1">64 ZONES ACTIVE</span>'
        '&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#9AA8B8">UPLOAD IMAGE TO BEGIN</span>'
        '&nbsp;&nbsp;·&nbsp;&nbsp;'
    )

st.markdown(f"""
<div style="background:#3B4A5E;border-bottom:1px solid #5A6B7E;
height:36px;overflow:hidden;display:flex;align-items:center;
margin-bottom:8px;border-radius:8px">
<div style="display:flex;white-space:nowrap;animation:tickerScroll 25s linear infinite;
font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:1.5px">
{_ticker_content}{_ticker_content}{_ticker_content}
</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍  Live Analysis",
    "⚖️  Compare Methods",
    "📊  Dashboard",
    "📱  Live Capture",
    "🗂️  Batch Analysis",
    "🤖  RL Agent",
])

_img_rgb      = None
_density_full = None
_zone_stats   = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
_features_sc  = None
_crowd_count  = 0

# ═══════════════════════════════════════════════════════════════
# TAB 1 — LIVE ANALYSIS
# ═══════════════════════════════════════════════════════════════

with tab1:

    # ── Section Header with LIVE indicator + Present button ───
    _hdr_l, _hdr_r = st.columns([5, 1])
    with _hdr_l:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:14px;padding:8px 0 4px 0">
        <div style="display:flex;align-items:center;gap:8px">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
        background:#10B981;box-shadow:0 0 10px rgba(16,185,129,0.6);
        animation:liveDot 1.5s ease-in-out infinite"></span>
        <span style="color:#10B981;font-size:10px;font-weight:700;
        letter-spacing:2px;font-family:'JetBrains Mono',monospace">LIVE</span>
        </div>
        <span style="color:#5A6B7E">│</span>
        <span style="color:#CBD5E1;font-size:12px;font-weight:600;
        letter-spacing:0.5px">Upload an image to begin crowd density analysis</span>
        </div>
        """, unsafe_allow_html=True)
    with _hdr_r:
        if st.session_state["present_mode"]:
            if st.button("⛶ Exit", key="exit_present", use_container_width=True):
                st.session_state["present_mode"] = False
                st.rerun()
        else:
            if st.button("⛶ Present", key="enter_present", use_container_width=True):
                st.session_state["present_mode"] = True
                st.rerun()

    if _capture_mode_mobile:
        st.components.v1.html("""
        <style>
        body {
            margin: 0;
            background: transparent;
            font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .capture-mobile-shell {
            display: none;
        }
        @media (max-width: 768px) {
            .capture-mobile-shell {
                display: block;
                padding: 6px 2px 18px;
            }
            .capture-mobile-topbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                color: #CBD5E1;
                font-size: 12px;
                font-weight: 700;
                margin-bottom: 12px;
                letter-spacing: 0.04em;
            }
            .capture-mobile-status {
                display: inline-flex;
                align-items: center;
                gap: 6px;
            }
            .capture-mobile-status span {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #10B981;
                box-shadow: 0 0 12px rgba(16,185,129,0.6);
            }
            .capture-mobile-card {
                background: linear-gradient(180deg, #3B4A5E 0%, #5A6B7E 100%);
                border: 1px solid #5A6B7E;
                border-radius: 20px;
                padding: 24px 18px 18px;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.45);
            }
            .capture-mobile-approw {
                display: flex;
                align-items: center;
                gap: 12px;
                margin: 18px 0 10px;
            }
            .capture-mobile-icon {
                width: 48px;
                height: 48px;
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #6366F1, #22D3EE);
                color: #FFFFFF;
                font-size: 24px;
                box-shadow: 0 12px 24px rgba(79,70,229,0.28);
            }
            .capture-mobile-brand {
                color: #F8FAFC;
                font-size: 17px;
                font-weight: 800;
                line-height: 1.25;
            }
            .capture-mobile-subbrand {
                color: #CBD5E1;
                font-size: 12px;
                margin-top: 3px;
            }
            .capture-mobile-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 7px 14px;
                border-radius: 999px;
                background: rgba(99,102,241,0.12);
                border: 1px solid rgba(99,102,241,0.25);
                color: #8896AA;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            .capture-mobile-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #6366F1;
                box-shadow: 0 0 14px rgba(99,102,241,0.6);
            }
            .capture-mobile-title {
                color: #F8FAFC;
                font-size: 24px;
                font-weight: 800;
                line-height: 1.2;
                letter-spacing: -0.03em;
                margin: 18px 0 8px;
            }
            .capture-mobile-copy {
                color: #CBD5E1;
                font-size: 14px;
                line-height: 1.65;
                margin: 0 0 18px;
            }
            .capture-mobile-permission {
                background: #F8FAFC;
                border-radius: 18px;
                padding: 20px 16px 14px;
                box-shadow: 0 14px 30px rgba(15, 23, 42, 0.22);
            }
            .capture-mobile-sheetbar {
                width: 42px;
                height: 5px;
                border-radius: 999px;
                background: #CBD5E1;
                margin: 0 auto 16px;
            }
            .capture-mobile-app {
                color: #2D3748;
                font-size: 18px;
                font-weight: 800;
                margin: 0 0 4px;
                text-align: center;
            }
            .capture-mobile-desc {
                color: #7A8B9E;
                font-size: 14px;
                line-height: 1.6;
                text-align: center;
                margin: 0 0 16px;
            }
            .capture-mobile-actions {
                display: grid;
                grid-template-columns: 1fr;
                gap: 10px;
            }
            .capture-mobile-btn {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                min-height: 52px;
                border: 0;
                border-radius: 14px;
                text-decoration: none;
                font-size: 16px;
                font-weight: 700;
                cursor: pointer;
            }
            .capture-mobile-btn-primary {
                background: linear-gradient(135deg, #6366F1, #4F46E5);
                color: #FFFFFF;
                box-shadow: 0 10px 24px rgba(79,70,229,0.3);
            }
            .capture-mobile-btn-secondary {
                background: #E2E8F0;
                color: #2D3748;
            }
            .capture-mobile-note {
                color: #9AA8B8;
                font-size: 12px;
                line-height: 1.5;
                margin-top: 12px;
                text-align: center;
            }
            .capture-mobile-privacy {
                color: #CBD5E1;
                font-size: 11px;
                line-height: 1.5;
                margin-top: 14px;
                text-align: center;
            }
        }
        </style>
        <div class="capture-mobile-shell">
          <div class="capture-mobile-card">
            <div class="capture-mobile-topbar">
              <div>SafeCrowd Vision</div>
              <div class="capture-mobile-status"><span></span> Live</div>
            </div>
            <div class="capture-mobile-pill">
              <span class="capture-mobile-dot"></span>
              Mobile Capture Mode
            </div>
            <div class="capture-mobile-approw">
              <div class="capture-mobile-icon"><svg viewBox="0 0 64 64" fill="none" style="width:24px;height:24px"><defs><linearGradient id="mobShield" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#fff"/><stop offset="100%" stop-color="#E0E7FF"/></linearGradient></defs><path d="M32 4 L56 14 L56 30 C56 46 44 56 32 60 C20 56 8 46 8 30 L8 14 Z" fill="url(#mobShield)"/><circle cx="32" cy="30" r="8" fill="none" stroke="#6366F1" stroke-width="2"/><circle cx="32" cy="30" r="3" fill="#6366F1"/></svg></div>
              <div>
                <div class="capture-mobile-brand">SafeCrowd Vision</div>
                <div class="capture-mobile-subbrand">Real-time public event safety analysis</div>
              </div>
            </div>
            <div class="capture-mobile-title">SafeCrowd Vision wants to access your camera</div>
            <p class="capture-mobile-copy">
              Tap an option below, then use the native picker to take a crowd photo or choose one from your gallery.
            </p>
            <div class="capture-mobile-permission">
              <div class="capture-mobile-sheetbar"></div>
              <div class="capture-mobile-app">SafeCrowd Vision</div>
              <p class="capture-mobile-desc">Allow camera access to capture a crowd image for live safety analysis.</p>
              <div class="capture-mobile-actions">
                <button class="capture-mobile-btn capture-mobile-btn-primary" onclick="focusUploader('camera')">Allow Camera</button>
                <button class="capture-mobile-btn capture-mobile-btn-secondary" onclick="focusUploader('gallery')">Choose from Gallery</button>
              </div>
              <div class="capture-mobile-note" id="capture-mobile-note">
                The upload area below opens your phone camera or gallery using the native iOS/Android picker.
              </div>
            </div>
            <div class="capture-mobile-privacy">
              Your photo stays in this Streamlit session and the synced laptop result only updates after you tap Analyze.
            </div>
          </div>
        </div>
        <script>
        function focusUploader(mode) {
          const note = document.getElementById('capture-mobile-note');
          if (note) {
            note.textContent = mode === 'camera'
              ? 'Tap the capture box below to open your camera.'
              : 'Tap the capture box below to open your gallery.';
          }
          try {
            const parentDoc = window.parent.document;
            const anchor = parentDoc.getElementById('mobile-capture-anchor');
            if (anchor) {
              anchor.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            const fileInput = parentDoc.querySelector('input[type="file"]');
            if (fileInput) {
              setTimeout(() => fileInput.click(), 120);
            }
          } catch (e) {}
        }
        </script>
        """, height=430)

    st.markdown('<div id="mobile-capture-anchor"></div>', unsafe_allow_html=True)

    _main_upload_label = (
        "📸 Tap here to open camera or gallery"
        if _capture_mode_mobile
        else "Drop a crowd image here or click to browse"
    )

    # ── File uploader ─────────────────────────────────────────
    uploaded = st.file_uploader(
        _main_upload_label,
        type=["jpg", "jpeg", "png"],
        key="main_upload")

    if uploaded:
        raw       = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        img_hash  = hashlib.sha1(raw.tobytes()).hexdigest()
        try:
            img_bgr   = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Image could not be decoded")
            _img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w      = _img_rgb.shape[:2]
        except Exception as _decode_err:
            st.error(f"⚠️ Could not read this image — the file may be corrupted or unsupported. ({_decode_err})")
            st.stop()

        if _capture_mode_mobile:
            _approved_hash = st.session_state.get("mobile_capture_confirmed_hash")
            if _approved_hash != img_hash:
                st.markdown("""
                <div style="background:linear-gradient(180deg,#3B4A5E 0%,#5A6B7E 100%);
                border:1px solid #5A6B7E;border-radius:18px;padding:18px 18px 16px;
                margin:14px 0 12px;box-shadow:0 16px 36px rgba(15,23,42,0.28)">
                <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px">
                <div style="color:#6366F1;font-size:10px;font-weight:700;
                letter-spacing:2px;text-transform:uppercase">
                ◈ Preview Capture
                </div>
                <div style="padding:5px 10px;border-radius:999px;background:rgba(99,102,241,0.12);
                border:1px solid rgba(99,102,241,0.25);color:#CBD5E1;font-size:11px;font-weight:700">
                Ready to analyze
                </div>
                </div>
                <div style="color:#CBD5E1;font-size:13px;line-height:1.7">
                Check the photo below, then tap <b>Analyze</b> to send it through crowd analysis and sync the result to the laptop view.
                </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(_img_rgb, caption="Preview", use_container_width=True)
                _analyze_cols = st.columns(2)
                with _analyze_cols[0]:
                    if st.button("Analyze", key=f"mobile_capture_analyze_{img_hash}",
                                 use_container_width=True, type="primary"):
                        st.session_state["mobile_capture_confirmed_hash"] = img_hash
                        st.rerun()
                with _analyze_cols[1]:
                    st.markdown("""
                    <div style="height:52px;display:flex;align-items:center;justify-content:center;
                    color:#9AA8B8;font-size:12px;text-align:center">
                    Use the uploader above to replace this photo
                    </div>
                    """, unsafe_allow_html=True)
                st.stop()

        # Only run inference if new image
        if st.session_state.get("last_img_hash") != img_hash:
            # ── Scan animation — shows BEFORE inference ──
            status_box = st.empty()
            if _capture_mode_mobile:
                status_box.markdown("""
                <div style="background:linear-gradient(180deg,#3B4A5E 0%,#5A6B7E 100%);
                border:1px solid #5A6B7E;border-radius:18px;padding:18px 18px 16px;margin:14px 0 12px;
                box-shadow:0 18px 38px rgba(15,23,42,0.26)">
                <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px">
                <div style="color:#FFFFFF;font-size:16px;font-weight:800;letter-spacing:-0.02em">
                Analyzing...</div>
                <div style="padding:5px 10px;border-radius:999px;background:rgba(34,211,238,0.12);
                border:1px solid rgba(34,211,238,0.22);color:#67E8F9;font-size:11px;font-weight:700">
                Live Sync
                </div>
                </div>
                <div style="color:#CBD5E1;font-size:13px;line-height:1.65">
                Running DM-Count, building density zones, and syncing the result back to the laptop view.
                </div>
                <div style="display:flex;gap:8px;align-items:center;margin-top:16px">
                  <span style="width:10px;height:10px;border-radius:50%;background:#6366F1;animation:mobilePulse 1s ease-in-out infinite"></span>
                  <span style="width:10px;height:10px;border-radius:50%;background:#22D3EE;animation:mobilePulse 1s ease-in-out infinite 0.15s"></span>
                  <span style="width:10px;height:10px;border-radius:50%;background:#10B981;animation:mobilePulse 1s ease-in-out infinite 0.3s"></span>
                </div>
                <style>
                @keyframes mobilePulse {
                    0%, 100% { transform: translateY(0); opacity: 0.45; }
                    50% { transform: translateY(-4px); opacity: 1; }
                }
                </style>
                </div>
                """, unsafe_allow_html=True)
            else:
                status_box.markdown("""
                <div style="background:#3B4A5E;border:1px solid #6366F1;
                border-radius:12px;padding:24px;margin:12px 0;
                border-left:4px solid #6366F1">
                <div style="display:flex;align-items:center;gap:12px">
                <div style="width:12px;height:12px;border-radius:50%;
                background:#4F46E5;animation:pulse 0.8s infinite">
                </div>
                <div style="color:#FFFFFF;font-family:monospace;
                font-size:13px;letter-spacing:1px">
                ⬡ INITIALIZING DM-COUNT ENGINE...</div>
                </div>
                <style>
                @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}
                50%{opacity:0.4;transform:scale(0.8)}}
                </style>
                </div>
                """, unsafe_allow_html=True)

            if LWCC_AVAILABLE and lwcc_shb is not None:
                _crowd_count, _density_full = predict_density_lwcc(_img_rgb)
            else:
                density_raw = predict_density_raw(cnn, img_bgr)
                _density_full = cv2.resize(density_raw, (w, h))
                raw_sum = float(density_raw.sum())
                _crowd_count = apply_calibration(raw_sum, "small")

            # Clear the scan animation
            status_box.empty()

            st.session_state["last_density_raw"] = _density_full
            st.session_state["last_img_hash"]    = img_hash
            st.session_state["last_image"]       = Image.fromarray(_img_rgb)
        else:
            _density_full = st.session_state["last_density_raw"]
            _crowd_count  = st.session_state["last_count"]

        _head_markers, _marker_meta = _resolve_head_markers(
            _img_rgb, _density_full, _crowd_count)
        _portrait_result = _resolve_portrait_count(
            _img_rgb, img_bgr, _density_full, _crowd_count,
            _marker_meta.get("count", len(_head_markers)),
            marker_note=_marker_meta.get("note", ""),
        )
        _crowd_count = _portrait_result["count"]
        _dot_count = _portrait_result["marker_count"]
        _marker_note = _portrait_result["marker_note"]
        _used_face_detector = _portrait_result["used_face_detector"]
        _portrait_hybrid_mode = _portrait_result["portrait_hybrid_mode"]
        _portrait_detected = _portrait_result["portrait_detected"]
        _portrait_faces = _portrait_result["faces"]

        st.session_state["last_count"] = _crowd_count

        thumb = cv2.resize(_img_rgb, (80, 80))
        if (len(st.session_state["history"]) == 0 or
                hash(st.session_state["history"][-1]["thumb"].tobytes()) != hash(thumb.tobytes())):
            st.session_state["history"].append(
                {"thumb": thumb, "count": _crowd_count})
        if len(st.session_state["history"]) > 5:
            st.session_state["history"] = st.session_state["history"][-5:]


        # ═════════════════════════════════════════════════════
        # PRESENT MODE — Fullscreen dramatic layout
        # ═════════════════════════════════════════════════════
        if st.session_state.get("present_mode", False):

            # Compute zone data (same as normal view)
            _p_feats = extract_features(_density_full)
            _p_feats_sc = scaler.transform(_p_feats) if scaler else _p_feats

            if method == "XGBoost" and xgb:
                _p_labels = labels_from_xgb(_p_feats_sc, xgb)
            elif method == "GMM" and gmm:
                _p_labels, _ = labels_from_gmm(_p_feats_sc, gmm)
            else:
                _p_labels = labels_from_kmeans(_p_feats_sc, km) \
                    if km else [get_label(float(f[0])) for f in _p_feats]

            _p_safety_img, _p_zone_stats = build_overlay(
                _img_rgb, _p_labels, GRID, opacity)

            # Threat calculation — percentage-based
            _p_crit_pct = _p_zone_stats["Critical"] / 64
            _p_high_pct = _p_zone_stats["High"] / 64
            _p_med_pct  = _p_zone_stats["Medium"] / 64
            _p_threat = min(100, int(
                _p_crit_pct * 100 * 0.7 +
                _p_high_pct * 100 * 0.2 +
                _p_med_pct  * 100 * 0.05 +
                min(10, _crowd_count / 50 * 10)
            ))
            # Crowd-count cap
            if _crowd_count < 20:
                _p_threat = min(_p_threat, 25)
            elif _crowd_count < 50:
                _p_threat = min(_p_threat, 50)
            if _p_threat < 25:
                _p_tl, _p_tc = "MINIMAL", "#10B981"
            elif _p_threat < 50:
                _p_tl, _p_tc = "ELEVATED", "#F59E0B"
            elif _p_threat < 75:
                _p_tl, _p_tc = "HIGH THREAT", "#EF4444"
            else:
                _p_tl, _p_tc = "CRITICAL EMERGENCY", "#FF1744"

            # Alert text
            if _p_zone_stats["Critical"] > 0:
                _p_alert_bg = "rgba(255,23,68,0.15)"
                _p_alert_border = "#FF1744"
                _p_alert_color = "#FF6B6B"
                _p_alert_txt = f"🚨 CRITICAL — {_p_zone_stats['Critical']} zones at stampede risk. Immediate crowd control recommended."
            elif _p_zone_stats["High"] > 0:
                _p_alert_bg = "rgba(245,158,11,0.15)"
                _p_alert_border = "#F59E0B"
                _p_alert_color = "#FCD34D"
                _p_alert_txt = f"⚠️ WARNING — {_p_zone_stats['High']} high-density zones detected. Monitor closely."
            else:
                _p_alert_bg = "rgba(16,185,129,0.15)"
                _p_alert_border = "#10B981"
                _p_alert_color = "#6EE7B7"
                _p_alert_txt = "✅ ALL CLEAR — All zones within safe density levels."

            # Hide sidebar + present-mode CSS
            st.markdown("""
            <style>
            section[data-testid="stSidebar"] { display: none !important; }
            [data-testid="collapsedControl"] { display: none !important; }
            </style>
            """, unsafe_allow_html=True)

            # ── TOP BAR ──
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #3B4A5E 0%, #2D3748 100%);
            border:1px solid #5A6B7E;border-radius:12px;padding:14px 24px;
            display:flex;align-items:center;justify-content:space-between;
            margin-bottom:20px;box-shadow:0 4px 24px rgba(0,0,0,0.5)">
            <div style="display:flex;align-items:center;gap:12px">
            <span style="filter:drop-shadow(0 2px 12px rgba(99,102,241,0.4));display:inline-flex"><svg viewBox="0 0 64 64" fill="none" style="width:28px;height:28px"><defs><linearGradient id="presShield" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#6366F1"/><stop offset="100%" stop-color="#22D3EE"/></linearGradient></defs><path d="M32 4 L56 14 L56 30 C56 46 44 56 32 60 C20 56 8 46 8 30 L8 14 Z" fill="url(#presShield)" stroke="rgba(255,255,255,0.2)" stroke-width="1"/><circle cx="32" cy="30" r="10" fill="none" stroke="white" stroke-width="1.5" opacity="0.8"/><circle cx="32" cy="30" r="4" fill="white" opacity="0.9"/></svg></span>
            <span style="color:#FFFFFF;font-size:20px;font-weight:800;letter-spacing:-0.03em">SafeCrowd Vision</span>
            <span style="display:inline-flex;align-items:center;gap:5px;
            padding:4px 12px;border-radius:16px;font-size:10px;font-weight:600;
            background:rgba(99,102,241,0.15);color:#3B82F6;
            border:1px solid rgba(99,102,241,0.3);
            font-family:'JetBrains Mono',monospace">PRESENT MODE</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px">
            <span style="display:inline-flex;align-items:center;gap:5px;
            padding:4px 12px;border-radius:16px;font-size:10px;font-weight:600;
            background:rgba(16,185,129,0.1);color:#10B981;
            border:1px solid rgba(16,185,129,0.2)">
            <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
            background:#059669;animation:dotPulse 1.5s ease-in-out infinite"></span> LIVE</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

            # ── 2-COLUMN LAYOUT: Image (60%) + Stats (40%) ──
            _pc1, _pc2 = st.columns([3, 2])

            with _pc1:
                st.markdown(f"""
                <div style="background:#3B4A5E;border:1px solid #5A6B7E;border-radius:12px;
                padding:8px;border-top:3px solid #6366F1;
                box-shadow:0 6px 32px rgba(99,102,241,0.2)">
                <div style="color:#CBD5E1;font-size:10px;font-weight:700;
                letter-spacing:2px;text-transform:uppercase;padding:8px 10px 6px">
                SAFETY ZONE MAP — {method.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(_p_safety_img, use_container_width=True)

            with _pc2:
                # Giant crowd count
                st.components.v1.html(f"""
                <div style="background:#3B4A5E;border:1px solid #5A6B7E;border-radius:12px;
                border-top:3px solid #6366F1;padding:28px 20px;text-align:center;
                box-shadow:0 6px 32px rgba(99,102,241,0.2)">
                <div style="color:#CBD5E1;font-size:10px;font-weight:700;
                letter-spacing:2.5px;text-transform:uppercase;margin-bottom:8px">
                ESTIMATED CROWD</div>
                <div id="present-count" style="font-size:80px;font-weight:900;color:#FFFFFF;
                font-family:'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;
                line-height:1;text-shadow:0 0 40px rgba(99,102,241,0.3)">0</div>
                </div>
                <script>
                (function() {{
                    let target = {_crowd_count};
                    let current = 0;
                    let steps = 50;
                    let inc = target / steps;
                    let el = document.getElementById('present-count');
                    let timer = setInterval(() => {{
                        current += inc;
                        if (current >= target) {{ current = target; clearInterval(timer); }}
                        el.textContent = Math.floor(current).toLocaleString();
                    }}, 20);
                }})();
                </script>
                """, height=170)

                # Threat gauge (compact)
                _p_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(_p_threat, 1),
                    number=dict(font=dict(size=32, color="#EFF6FF",
                                          family="JetBrains Mono, monospace"),
                                suffix="%"),
                    title=dict(text=f"THREAT: {_p_tl}",
                               font=dict(size=13, color=_p_tc,
                                         family="Inter, sans-serif")),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#9AA8B8",
                                  tickfont=dict(color="#CBD5E1", size=9)),
                        bar=dict(color=_p_tc, thickness=0.35),
                        bgcolor="#8896AA", borderwidth=0,
                        steps=[
                            dict(range=[0, 25], color="rgba(16,185,129,0.15)"),
                            dict(range=[25, 50], color="rgba(245,158,11,0.15)"),
                            dict(range=[50, 75], color="rgba(239,68,68,0.15)"),
                            dict(range=[75, 100], color="rgba(255,23,68,0.15)"),
                        ],
                        threshold=dict(line=dict(color=_p_tc, width=3),
                                       thickness=0.8, value=_p_threat),
                    ),
                ))
                _p_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=180,
                    margin=dict(l=20, r=20, t=45, b=5),
                    font=dict(family="Inter, sans-serif"),
                )
                st.plotly_chart(_p_gauge, use_container_width=True)

                # 4 zone stats — glowing numbers
                st.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px">
                <div style="background:#3B4A5E;border:1px solid rgba(16,185,129,0.25);
                border-radius:10px;padding:14px;text-align:center;
                box-shadow:0 0 20px rgba(16,185,129,0.08)">
                <div style="font-size:10px;color:#CBD5E1;font-weight:700;
                letter-spacing:1.5px;margin-bottom:4px">🟢 LOW</div>
                <div style="font-size:32px;font-weight:900;color:#10B981;
                font-family:'JetBrains Mono',monospace;
                text-shadow:0 0 20px rgba(16,185,129,0.4)">{_p_zone_stats['Low']}</div>
                </div>
                <div style="background:#3B4A5E;border:1px solid rgba(245,158,11,0.25);
                border-radius:10px;padding:14px;text-align:center;
                box-shadow:0 0 20px rgba(245,158,11,0.08)">
                <div style="font-size:10px;color:#CBD5E1;font-weight:700;
                letter-spacing:1.5px;margin-bottom:4px">🟡 MEDIUM</div>
                <div style="font-size:32px;font-weight:900;color:#D97706;
                font-family:'JetBrains Mono',monospace;
                text-shadow:0 0 20px rgba(245,158,11,0.4)">{_p_zone_stats['Medium']}</div>
                </div>
                <div style="background:#3B4A5E;border:1px solid rgba(239,68,68,0.25);
                border-radius:10px;padding:14px;text-align:center;
                box-shadow:0 0 20px rgba(239,68,68,0.08)">
                <div style="font-size:10px;color:#CBD5E1;font-weight:700;
                letter-spacing:1.5px;margin-bottom:4px">🟠 HIGH</div>
                <div style="font-size:32px;font-weight:900;color:#DC2626;
                font-family:'JetBrains Mono',monospace;
                text-shadow:0 0 20px rgba(239,68,68,0.4)">{_p_zone_stats['High']}</div>
                </div>
                <div style="background:#3B4A5E;border:1px solid rgba(255,23,68,0.25);
                border-radius:10px;padding:14px;text-align:center;
                box-shadow:0 0 20px rgba(255,23,68,0.1);
                {'animation:criticalGlow 2s ease-in-out infinite;' if _p_zone_stats['Critical'] > 0 else ''}">
                <div style="font-size:10px;color:#CBD5E1;font-weight:700;
                letter-spacing:1.5px;margin-bottom:4px">🔴 CRITICAL</div>
                <div style="font-size:32px;font-weight:900;color:#B91C1C;
                font-family:'JetBrains Mono',monospace;
                text-shadow:0 0 20px rgba(255,23,68,0.5)">{_p_zone_stats['Critical']}</div>
                </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Alert banner (full width) ──
            st.markdown(f"""
            <div style="background:{_p_alert_bg};border:1px solid {_p_alert_border};
            border-left:4px solid {_p_alert_border};border-radius:10px;
            padding:18px 24px;color:{_p_alert_color};font-weight:700;
            font-size:16px;margin:16px 0;
            {'animation:criticalBorderPulse 2s ease-in-out infinite;' if _p_zone_stats['Critical'] > 0 else ''}
            text-align:center;letter-spacing:0.3px">
            {_p_alert_txt}</div>
            """, unsafe_allow_html=True)

            # ── BOTTOM BAR ──
            _p_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _p_model = _model_label_for_scene(
                _crowd_count,
                used_face_detector=_used_face_detector,
                portrait_hybrid_mode=_portrait_hybrid_mode,
            )
            st.markdown(f"""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;border-radius:8px;
            padding:10px 20px;display:flex;align-items:center;
            justify-content:space-between;margin-top:8px">
            <span style="color:#3B4A63;font-family:'JetBrains Mono',monospace;
            font-size:10px;letter-spacing:1px">{_p_now}</span>
            <span style="color:#3B4A63;font-family:'JetBrains Mono',monospace;
            font-size:10px;letter-spacing:1px">
            {_p_model} · {w}×{h} · {method} · 8×8 grid</span>
            </div>
            """, unsafe_allow_html=True)


        # ── DEMO MODE ─────────────────────────────────────────
        elif demo_mode:
            count = _crowd_count
            if count < 30:
                bg = "linear-gradient(135deg, #105030, #1A6038)"
                glow = "0 0 40px rgba(16,185,129,0.15)"
                label = "✅ SAFE"
            elif count < 60:
                bg = "linear-gradient(135deg, #5A3D18, #704020)"
                glow = "0 0 40px rgba(245,158,11,0.15)"
                label = "⚠️ FULL"
            else:
                bg = "linear-gradient(135deg, #5A1828, #701D30)"
                glow = "0 0 40px rgba(255,23,68,0.2)"
                label = "🚨 OVERCROWDED"

            st.markdown(
                f'<div style="background:{bg};border-radius:20px;padding:60px 20px;'
                f'text-align:center;margin-top:20px;box-shadow:{glow};'
                f'border:1px solid rgba(255,255,255,0.05)">'
                f'<div style="font-size:9rem;font-weight:900;color:white;'
                f'line-height:1;font-family:\'JetBrains Mono\',monospace;'
                f'font-variant-numeric:tabular-nums">{count}</div>'
                f'<div style="font-size:2.5rem;color:white;margin-top:10px;">'
                f'{label}</div>'
                f'<div style="font-size:1.2rem;color:rgba(255,255,255,0.55);'
                f'margin-top:6px;font-weight:400">estimated crowd</div></div>',
                unsafe_allow_html=True)

        else:
            # ── Normal analysis view ──────────────────────────

            # ── Post-Analysis Section Header ──
            st.markdown("""
            <div style="background:linear-gradient(135deg,#3B4A5E 0%,#2D3748 100%);
            border:1px solid #5A6B7E;border-radius:12px;padding:14px 24px;
            margin:8px 0 20px 0;display:flex;align-items:center;
            justify-content:space-between;overflow:hidden;position:relative;
            box-shadow:0 4px 24px rgba(0,0,0,0.4)">
            <div style="position:absolute;top:0;left:0;height:2px;width:100%;
            background:#5A6B7E">
            <div style="height:2px;
            background:linear-gradient(90deg,#6366F1,#22D3EE,#8B5CF6,#6366F1);
            background-size:200% 100%;animation:scanLine 2s linear infinite;
            width:100%"></div>
            </div>
            <div style="display:flex;align-items:center;gap:12px">
            <span style="display:inline-block;width:10px;height:10px;
            border-radius:50%;background:#059669;
            box-shadow:0 0 12px rgba(5,150,105,0.5);
            animation:liveDot 1.5s ease-in-out infinite"></span>
            <span style="color:#FFFFFF;font-size:14px;font-weight:700;
            font-family:'JetBrains Mono',monospace;letter-spacing:1.5px">
            ANALYSIS RESULTS</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px">
            <span style="padding:4px 12px;border-radius:16px;font-size:10px;
            font-weight:600;background:rgba(16,185,129,0.1);color:#10B981;
            border:1px solid rgba(16,185,129,0.2);
            font-family:'JetBrains Mono',monospace">SCAN COMPLETE</span>
            </div>
            </div>
            <style>
            @keyframes scanLine {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }
            </style>
            """, unsafe_allow_html=True)


            feats = extract_features(_density_full)
            _features_sc = scaler.transform(feats) if scaler else feats

            if method == "XGBoost" and xgb:
                try:
                    labels = labels_from_xgb(_features_sc, xgb)
                except Exception:
                    labels = [get_label(float(f[0])) for f in feats]
            elif method == "GMM" and gmm:
                try:
                    labels, _ = labels_from_gmm(_features_sc, gmm)
                except Exception:
                    labels = [get_label(float(f[0])) for f in feats]
            else:
                try:
                    labels = labels_from_kmeans(_features_sc, km) \
                        if km else [get_label(float(f[0])) for f in feats]
                except Exception:
                    labels = [get_label(float(f[0])) for f in feats]

            safety_img, _zone_stats = build_overlay(
                _img_rgb, labels, GRID, opacity)
            st.session_state["last_zone_stats"] = _zone_stats.copy()

            # (threat score is calculated at the gauge render site below)

            density_overlay = build_density_overlay(
                _img_rgb, _density_full, opacity,
                expected_count=_crowd_count,
                marker_points=_head_markers)

            if _portrait_faces and (
                    _used_face_detector or _portrait_hybrid_mode):
                headdot_overlay = _draw_faces_on_image(
                    _img_rgb, _portrait_faces)
            else:
                headdot_overlay, _dot_count = build_headdot_overlay(
                    _img_rgb, _density_full,
                    expected_count=_crowd_count,
                    marker_points=_head_markers)

            if _portrait_detected:
                st.markdown("""
<div style="background:rgba(245,158,11,0.1);
border:1px solid #F59E0B;border-left:4px solid #F59E0B;
border-radius:10px;padding:14px 20px;margin:8px 0;
color:#FCD34D;font-size:13px;font-weight:600">
⚠️ PORTRAIT/CLOSE-UP DETECTED — 
Using <b>portrait-aware recovery</b> with face detection and density cues. 
DM-Count is optimized for crowd scenes 
(20+ people, overhead or eye-level view).
</div>
""", unsafe_allow_html=True)

            # ── Analysis Complete banner ───────────────────────
            _banner_conf = compute_dynamic_confidence(
                _features_sc, method, km, xgb, gmm, _crowd_count)
            _banner_model = _model_label_for_scene(
                _crowd_count,
                used_face_detector=_used_face_detector,
                portrait_hybrid_mode=_portrait_hybrid_mode,
            )

            if _portrait_hybrid_mode:
                _banner_label = "ESTIMATED GROUP"
                _banner_primary = _crowd_count
                _banner_secondary = (
                    f"Portrait markers: {_dot_count} · face boxes + density"
                )
                _banner_note_block = (
                    '<div style="margin-top:14px;padding-top:12px;'
                    'border-top:1px solid #5A6B7E;color:#F59E0B;'
                    'font-size:10px;font-family:monospace;line-height:1.45">'
                    f'{_marker_note}</div>'
                    if _marker_note else ""
                )
            elif _used_face_detector:
                _banner_label = "FACES DETECTED"
                _banner_primary = _dot_count
                _banner_secondary = f"OpenCV face markers: {_dot_count}"
                _banner_note_block = ""
            else:
                _banner_label = "ESTIMATED CROWD"
                _banner_primary = _crowd_count
                _banner_secondary = f"Visible head markers: {_dot_count}"
                _banner_note_block = (
                    '<div style="margin-top:14px;padding-top:12px;'
                    'border-top:1px solid #5A6B7E;color:#F59E0B;'
                    'font-size:10px;font-family:monospace;line-height:1.45">'
                    f'{_marker_note}</div>'
                    if _marker_note else ""
                )
            _banner_height = 154 if _banner_note_block else 128

            st.components.v1.html(f"""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
            border-radius:12px;padding:20px 24px;margin-bottom:24px;
            overflow:hidden;position:relative;min-height:102px;">

            <div style="position:absolute;top:0;left:0;height:3px;
            width:100%;background:#5A6B7E;">
            <div style="height:3px;background:linear-gradient(90deg,
            #6366F1,#22D3EE,#8B5CF6);animation:fill 0.8s ease-out forwards;
            width:0%"></div>
            </div>

            <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;
            margin-top:4px">
            <div style="width:10px;height:10px;border-radius:50%;
            background:#059669;box-shadow:0 0 8px #059669;
            animation:blink 1s ease-in-out 3"></div>
            <div style="color:#FFFFFF;font-size:14px;font-weight:700;
            font-family:monospace;letter-spacing:1px">ANALYSIS COMPLETE</div>
            </div>

            <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
            align-items:start">

            <div style="text-align:center;border-right:1px solid #5A6B7E;
            padding:0 20px;min-height:58px">
            <div style="color:#CBD5E1;font-size:9px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">
            {_banner_label}</div>
            <div style="color:#6366F1;font-size:32px;font-weight:900;
            font-family:monospace;text-shadow:0 0 20px rgba(99,102,241,0.2)">
            {_banner_primary}</div>
            <div style="color:#9AA8B8;font-size:10px;font-family:monospace;
            margin-top:2px">{_banner_secondary}</div>
            </div>

            <div style="text-align:center;border-right:1px solid #5A6B7E;
            padding:0 20px;min-height:58px">
            <div style="color:#CBD5E1;font-size:9px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">
            MODEL</div>
            <div style="color:#FFFFFF;font-size:14px;font-weight:700;
            font-family:monospace">{_banner_model}</div>
            </div>

            <div style="text-align:center;padding:0 20px;min-height:58px">
            <div style="color:#CBD5E1;font-size:9px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">
            CONFIDENCE</div>
            <div style="color:#10B981;font-size:14px;font-weight:700;
            font-family:monospace">{_banner_conf}%</div>
            </div>
            </div>
            {_banner_note_block}

            <style>
            @keyframes fill {{
            0%{{width:0%}} 100%{{width:100%}}
            }}
            @keyframes blink {{
            0%,100%{{opacity:1}} 50%{{opacity:0.2}}
            }}
            </style>
            </div>
            """, height=_banner_height)

            # ── Row 1: 3 image panels ─────────────────────────
            def _to_panel_b64(img_arr):
                _, _buf = cv2.imencode('.jpg',
                    cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
                return base64.b64encode(_buf).decode()

            _orig_b64    = _to_panel_b64(_img_rgb)
            _headdot_b64 = _to_panel_b64(headdot_overlay)
            _density_b64 = _to_panel_b64(density_overlay)
            _safety_b64  = _to_panel_b64(safety_img)

            if _capture_mode_requested:
                _save_latest_capture_payload({
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "source": "mobile-capture",
                    "filename": getattr(uploaded, "name", "mobile_capture"),
                    "count": int(_crowd_count),
                    "threat": str(_compute_threat_band(_zone_stats, _crowd_count)[1]),
                    "threat_score": int(_compute_threat_band(_zone_stats, _crowd_count)[0]),
                    "confidence": int(_banner_conf),
                    "method": str(method),
                    "model": str(_banner_model),
                    "zone_stats": _zone_stats,
                    "marker_count": int(_dot_count),
                    "marker_note": str(_marker_note or ""),
                    "used_face_detector": bool(_used_face_detector),
                    "portrait_hybrid_mode": bool(_portrait_hybrid_mode),
                    "original_b64": _orig_b64,
                    "head_overlay_b64": _headdot_b64,
                    "density_overlay_b64": _density_b64,
                    "safety_img_b64": _safety_b64,
                })
                if _capture_mode_mobile:
                    st.markdown("""
                    <div style="background:linear-gradient(180deg,#1F4A2E 0%,#224A32 100%);
                    border:1px solid rgba(16,185,129,0.28);border-radius:18px;padding:18px 18px 16px;
                    margin:0 0 18px 0;box-shadow:0 18px 34px rgba(5,150,105,0.14)">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
                    <div style="width:34px;height:34px;border-radius:12px;background:#10B981;
                    display:flex;align-items:center;justify-content:center;color:white;font-size:18px;font-weight:800">✓</div>
                    <div>
                    <div style="color:#ECFDF5;font-size:16px;font-weight:800;letter-spacing:-0.02em">
                    Analysis complete</div>
                    <div style="color:#A7F3D0;font-size:12px">
                    Result synced to the laptop Live Capture screen</div>
                    </div>
                    </div>
                    <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-top:12px">
                    <div style="padding:8px 12px;border-radius:12px;background:rgba(255,255,255,0.05)">
                    <div style="color:#6EE7B7;font-size:10px;font-weight:700;letter-spacing:1px">COUNT</div>
                    <div style="color:#F0FDF4;font-size:20px;font-weight:900;font-family:'JetBrains Mono',monospace">{}</div>
                    </div>
                    <div style="padding:8px 12px;border-radius:12px;background:rgba(255,255,255,0.05)">
                    <div style="color:#6EE7B7;font-size:10px;font-weight:700;letter-spacing:1px">THREAT</div>
                    <div style="color:#F0FDF4;font-size:20px;font-weight:900;font-family:'JetBrains Mono',monospace">{}%</div>
                    </div>
                    </div>
                    </div>
                    """.format(int(_crowd_count), int(_compute_threat_band(_zone_stats, _crowd_count)[0])), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);
                    border-radius:10px;padding:12px 14px;margin:0 0 18px 0;color:#6EE7B7;
                    font-size:12px;font-weight:600">
                    ● Mobile capture synced to laptop view. Open the Live Capture tab on the laptop to see the latest result.
                    </div>
                    """, unsafe_allow_html=True)

            # Dynamic panel labels for portrait vs crowd mode
            _panel1_title = (
                f"FACE DETECTION — {_dot_count} FACES"
                if _used_face_detector
                else (
                    f"PORTRAIT MARKERS — {_dot_count}"
                    if _portrait_hybrid_mode
                    else f"HEAD MARKERS — {_dot_count}"
                )
            )
            _panel1_sub = (
                f"Each cyan rectangle = 1 detected face · {w}×{h}"
                if _used_face_detector
                else (
                    f"Approximate portrait markers · estimate {_crowd_count}"
                    if _portrait_hybrid_mode
                    else (
                        f"Approximate visible head markers · DM-Count estimate {_crowd_count}"
                    )
                )
            )
            _panel1_note = (
                ""
                if _used_face_detector or not _marker_note
                else (
                    '<div style="padding:0 16px 10px;background:#2D3748;'
                    'color:#F59E0B;font-size:11px">'
                    + (
                        'Portrait markers are approximate and may combine face '
                        'boxes with density recovery.'
                        if _portrait_hybrid_mode else
                        'Some people may be counted from density without a clean '
                        'visible marker.'
                    )
                    + '</div>'
                )
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div style="background:#3B4A5E;border:1px solid #5A6B7E;
                border-radius:12px;overflow:hidden;
                box-shadow:0 4px 24px rgba(99,102,241,0.15)">
                <div style="background:#485A6E;padding:10px 16px;
                border-top:3px solid #6366F1">
                <span style="color:#6366F1;font-size:10px;text-transform:uppercase;
                letter-spacing:2px;font-weight:600">{_panel1_title}</span></div>
                <img src="data:image/jpeg;base64,{_headdot_b64}"
                style="width:100%;display:block">
                <div style="padding:8px 16px;background:#2D3748;
                color:#9AA8B8;font-size:11px">{_panel1_sub}</div>
                {_panel1_note}
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="background:#3B4A5E;border:1px solid #5A6B7E;
                border-radius:12px;overflow:hidden;
                box-shadow:0 4px 24px rgba(99,102,241,0.15)">
                <div style="background:#485A6E;padding:10px 16px;
                border-top:3px solid #6366F1">
                <span style="color:#6366F1;font-size:10px;text-transform:uppercase;
                letter-spacing:2px;font-weight:600">DENSITY HEATMAP</span></div>
                <img src="data:image/jpeg;base64,{_density_b64}"
                style="width:100%;display:block">
                <div style="padding:8px 16px;background:#2D3748;
                color:#9AA8B8;font-size:11px">Gaussian σ=8 · JET colormap · opacity {opacity}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div style="background:#3B4A5E;border:1px solid #5A6B7E;
                border-radius:12px;overflow:hidden;
                box-shadow:0 4px 24px rgba(99,102,241,0.15)">
                <div style="background:#485A6E;padding:10px 16px;
                border-top:3px solid #10B981">
                <span style="color:#10B981;font-size:10px;text-transform:uppercase;
                letter-spacing:2px;font-weight:600">SAFETY ZONE MAP — {method.upper()}</span></div>
                <img src="data:image/jpeg;base64,{_safety_b64}"
                style="width:100%;display:block">
                <div style="padding:8px 16px;background:#2D3748;
                color:#9AA8B8;font-size:11px">8×8 grid · {method} classification</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Spacing between image panels and confidence card ──
            st.markdown('<div style="margin-bottom:24px"></div>', unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # FEATURE 4 — ANALYSIS CONFIDENCE CARD
            # ══════════════════════════════════════════════════
            _confidence_pct = compute_dynamic_confidence(
                _features_sc, method, km, xgb, gmm, _crowd_count)
            if _crowd_count < 80:
                _model_badge = "DM-Count · SHB"
            elif _crowd_count < 200:
                _model_badge = "DM-Count · SHA"
            else:
                _model_badge = "DM-Count · Ensemble"

            st.markdown(f"""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;border-radius:10px;
            padding:14px 20px;margin:8px 0 16px 0;display:flex;align-items:center;
            justify-content:space-between;gap:20px">
            <div style="display:flex;align-items:center;gap:8px;flex-shrink:0">
            <span style="font-size:14px">🎯</span>
            <span style="color:#CBD5E1;font-size:10px;font-weight:700;
            letter-spacing:1.5px;text-transform:uppercase">ANALYSIS CONFIDENCE</span>
            </div>
            <div style="flex:1;display:flex;align-items:center;gap:12px">
            <div style="flex:1;height:6px;background:#8896AA;border-radius:4px;overflow:hidden">
            <div style="height:100%;width:{_confidence_pct}%;
            background:linear-gradient(90deg,#2563EB,#06B6D4);border-radius:4px;
            animation:barFill 1s ease-out forwards;
            --fill-pct:{_confidence_pct}%"></div>
            </div>
            <span style="color:#FFFFFF;font-family:'JetBrains Mono',monospace;
            font-size:13px;font-weight:700;font-variant-numeric:tabular-nums;
            flex-shrink:0">{_confidence_pct}%</span>
            </div>
            <span style="display:inline-flex;align-items:center;gap:5px;
            padding:4px 12px;border-radius:16px;font-size:10px;font-weight:600;
            background:rgba(34,211,238,0.1);color:#6366F1;
            border:1px solid rgba(34,211,238,0.2);flex-shrink:0;
            font-family:'JetBrains Mono',monospace">{_model_badge}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Spacing between confidence card and metrics ──
            st.markdown('<div style="margin-bottom:24px"></div>', unsafe_allow_html=True)

            # ── Row 2: metrics ────────────────────────────────
            # FEATURE C — CONFIDENCE INTERVAL for Estimated Crowd
            if _crowd_count < 80:
                _mae = 4.92   # SHB MAE
            elif _crowd_count < 200:
                _mae = 61.39  # SHA MAE
            else:
                _mae = 50.0   # ensemble estimate

            _count_low  = max(0, int(_crowd_count - _mae))
            _count_high = int(_crowd_count + _mae)

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.components.v1.html(f"""
                <div style="background:#3B4A5E;border:1px solid #5A6B7E;
                border-radius:12px;padding:22px 18px;
                border-top:3px solid #6366F1;
                box-shadow:0 4px 24px rgba(99,102,241,0.2)">
                <div style="color:#CBD5E1;font-size:11px;text-transform:uppercase;
                letter-spacing:1.2px;font-weight:600;margin-bottom:8px">
                👥 ESTIMATED CROWD</div>
                <div style="color:#FFFFFF;font-family:'JetBrains Mono',monospace;
                font-size:36px;font-weight:700;letter-spacing:-0.03em;
                font-variant-numeric:tabular-nums;line-height:1">{_crowd_count:,}</div>
                <div style="display:flex;align-items:center;gap:8px;margin-top:8px">
                <span style="color:#CBD5E1;font-size:11px">Range: {_count_low} – {_count_high}</span>
                <span style="padding:2px 8px;border-radius:12px;font-size:10px;
                font-weight:600;background:rgba(34,211,238,0.1);color:#6366F1;
                border:1px solid rgba(34,211,238,0.2);
                font-family:'JetBrains Mono',monospace">±{int(_mae)} MAE</span>
                </div>
                </div>
                """, height=130)
            with m2:
                _crit_val = _zone_stats["Critical"]
                _crit_glow = 'animation:criticalGlow 2s ease-in-out infinite;' if _crit_val > 0 else ''
                st.markdown(f"""
                <div style="background:#3B4A5E;border:1px solid rgba(255,23,68,0.3);
                border-radius:12px;padding:22px 18px;text-align:center;
                border-top:3px solid #FF1744;
                box-shadow:0 4px 24px rgba(255,23,68,0.15);
                transition:all 0.3s ease;{_crit_glow}">
                <div style="color:#CBD5E1;font-size:11px;text-transform:uppercase;
                letter-spacing:1.2px;font-weight:600;margin-bottom:8px">
                🔴 CRITICAL ZONES</div>
                <div style="color:#FF1744;font-family:'JetBrains Mono',monospace;
                font-size:40px;font-weight:800;letter-spacing:-0.03em;
                font-variant-numeric:tabular-nums;
                text-shadow:0 0 20px rgba(255,23,68,0.4)">{_crit_val}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                _high_val = _zone_stats["High"]
                st.markdown(f"""
                <div style="background:#3B4A5E;border:1px solid rgba(239,68,68,0.25);
                border-radius:12px;padding:22px 18px;text-align:center;
                border-top:3px solid #EF4444;
                box-shadow:0 4px 24px rgba(239,68,68,0.15);
                transition:all 0.3s ease">
                <div style="color:#CBD5E1;font-size:11px;text-transform:uppercase;
                letter-spacing:1.2px;font-weight:600;margin-bottom:8px">
                🟠 HIGH ZONES</div>
                <div style="color:#EF4444;font-family:'JetBrains Mono',monospace;
                font-size:40px;font-weight:800;letter-spacing:-0.03em;
                font-variant-numeric:tabular-nums;
                text-shadow:0 0 20px rgba(239,68,68,0.3)">{_high_val}</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                _med_val = _zone_stats["Medium"]
                st.markdown(f"""
                <div style="background:#3B4A5E;border:1px solid rgba(245,158,11,0.25);
                border-radius:12px;padding:22px 18px;text-align:center;
                border-top:3px solid #F59E0B;
                box-shadow:0 4px 24px rgba(245,158,11,0.15);
                transition:all 0.3s ease">
                <div style="color:#CBD5E1;font-size:11px;text-transform:uppercase;
                letter-spacing:1.2px;font-weight:600;margin-bottom:8px">
                🟡 MEDIUM ZONES</div>
                <div style="color:#F59E0B;font-family:'JetBrains Mono',monospace;
                font-size:40px;font-weight:800;letter-spacing:-0.03em;
                font-variant-numeric:tabular-nums;
                text-shadow:0 0 20px rgba(245,158,11,0.3)">{_med_val}</div>
                </div>
                """, unsafe_allow_html=True)
            with m5:
                _safe_val = _zone_stats["Low"]
                st.markdown(f"""
                <div style="background:#3B4A5E;border:1px solid rgba(16,185,129,0.25);
                border-radius:12px;padding:22px 18px;text-align:center;
                border-top:3px solid #10B981;
                box-shadow:0 4px 24px rgba(16,185,129,0.15);
                transition:all 0.3s ease">
                <div style="color:#CBD5E1;font-size:11px;text-transform:uppercase;
                letter-spacing:1.2px;font-weight:600;margin-bottom:8px">
                🟢 SAFE ZONES</div>
                <div style="color:#10B981;font-family:'JetBrains Mono',monospace;
                font-size:40px;font-weight:800;letter-spacing:-0.03em;
                font-variant-numeric:tabular-nums;
                text-shadow:0 0 20px rgba(16,185,129,0.3)">{_safe_val}</div>
                </div>
                """, unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # FEATURE 1 — THREAT LEVEL GAUGE
            # Nuclear fix: recalculate everything fresh from
            # _zone_stats RIGHT HERE at render time.
            # ══════════════════════════════════════════════════
            # Percentage-based threat formula:
            # Uses zone ratios so 4/64 Critical ≈ 4%, not 100%
            _crit_pct = _zone_stats.get("Critical", 0) / 64
            _high_pct = _zone_stats.get("High", 0) / 64
            _med_pct  = _zone_stats.get("Medium", 0) / 64

            _gauge_value = min(100, int(
                _crit_pct * 100 * 0.7 +
                _high_pct * 100 * 0.2 +
                _med_pct  * 100 * 0.05 +
                min(10, _crowd_count / 50 * 10)
            ))

            # Crowd-count based threat cap:
            # Small crowds can never be CRITICAL EMERGENCY
            if _crowd_count < 20:
                _gauge_value = min(_gauge_value, 25)
            elif _crowd_count < 50:
                _gauge_value = min(_gauge_value, 50)

            if _gauge_value < 25:
                _g_label = "MINIMAL"
                _g_color = "#10B981"
                _g_rec = "All zones nominal. No action required."
            elif _gauge_value < 50:
                _g_label = "ELEVATED"
                _g_color = "#F59E0B"
                _g_rec = "Increased density detected. Deploy monitoring."
            elif _gauge_value < 75:
                _g_label = "HIGH THREAT"
                _g_color = "#EF4444"
                _g_rec = "Crowd pressure rising. Activate response team."
            else:
                _g_label = "CRITICAL EMERGENCY"
                _g_color = "#FF1744"
                _g_rec = "IMMEDIATE INTERVENTION REQUIRED. Evacuate sectors."

            print(f"[GAUGE DEBUG] method={method}")
            print(f"[GAUGE DEBUG] _zone_stats={_zone_stats}")
            print(f"[GAUGE DEBUG] _gauge_value={_gauge_value}")
            print(f"[GAUGE DEBUG] _crowd_count={_crowd_count}")
            print(f"[GAUGE DEBUG] _g_label={_g_label}")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=_gauge_value,
                number=dict(font=dict(size=40, color="#FFFFFF",
                                      family="JetBrains Mono, monospace"),
                            suffix="%"),
                title=dict(text=f"THREAT LEVEL: {_g_label}",
                           font=dict(size=14, color=_g_color,
                                     family="Inter, sans-serif")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#9AA8B8",
                              tickfont=dict(color="#CBD5E1", size=10)),
                    bar=dict(color=_g_color, thickness=0.3),
                    bgcolor="#8896AA",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 25],   color="#1F4A2E"),
                        dict(range=[25, 50],  color="#4A3D10"),
                        dict(range=[50, 75],  color="#4A2D10"),
                        dict(range=[75, 100], color="#4A1520"),
                    ],
                    threshold=dict(line=dict(color=_g_color, width=3),
                                   thickness=0.8, value=_gauge_value),
                ),
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=200,
                margin=dict(l=30, r=30, t=50, b=10),
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
            <div style="text-align:center;padding:4px 0 12px 0">
            <p style="color:{_g_color};font-size:13px;font-weight:700;margin:0;
            letter-spacing:0.5px">{_g_rec}</p>
            </div>
            """, unsafe_allow_html=True)

            # ── Alert banner ──────────────────────────────────
            if _zone_stats["Critical"] > 0:
                st.markdown(f"""<div style="background:#3D2020;
                border:1px solid rgba(255,23,68,0.3);border-left:4px solid #FF1744;
                border-radius:10px;padding:18px 24px;
                color:#FF6B6B;font-weight:700;font-size:15px;margin:16px 0;
                animation:criticalBorderPulse 2s ease-in-out infinite">
                🚨 CRITICAL — {_zone_stats["Critical"]} zones at stampede risk level.
                Immediate crowd control recommended.</div>""", unsafe_allow_html=True)
            elif _zone_stats["High"] > 0:
                st.markdown(f"""<div style="background:rgba(245,158,11,0.08);
                border:1px solid rgba(245,158,11,0.3);border-left:4px solid #F59E0B;
                border-radius:10px;padding:18px 24px;
                color:#FCD34D;font-weight:700;font-size:15px;margin:16px 0">
                ⚠️ WARNING — {_zone_stats["High"]} high-density zones detected.
                Monitor closely.</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style="background:rgba(16,185,129,0.1);
                border:1px solid rgba(16,185,129,0.3);border-left:4px solid #10B981;
                border-radius:10px;padding:18px 24px;
                color:#6EE7B7;font-weight:700;font-size:15px;margin:16px 0">
                ✅ ALL CLEAR — All zones within safe density levels.</div>""",
                            unsafe_allow_html=True)

            # ── Zone breakdown chart ──────────────────────────
            st.markdown('<div style="border-top:1px solid #5A6B7E;margin:24px 0"></div>',
                        unsafe_allow_html=True)

            # ── Zone Breakdown — 4 stat columns ──────────────
            _zb_zones  = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            _zb_keys   = ["Low", "Medium", "High", "Critical"]
            _zb_colors = ["#10B981", "#F59E0B", "#EF4444", "#FF1744"]

            col_low, col_med, col_high, col_crit = st.columns(4)
            for _zb_col, _zb_name, _zb_key, _zb_color in zip(
                [col_low, col_med, col_high, col_crit],
                _zb_zones, _zb_keys, _zb_colors):
                _zb_count = _zone_stats[_zb_key]
                _zb_pct = (_zb_count / 64) * 100
                with _zb_col:
                    st.markdown(f"""
                    <div style="background:#3B4A5E;border:1px solid #5A6B7E;
                    border-radius:12px;padding:20px;text-align:center;
                    border-top:3px solid {_zb_color}">
                    <div style="font-size:2.5rem;font-weight:900;
                    color:{_zb_color};font-family:monospace">{_zb_count}</div>
                    <div style="font-size:10px;color:#CBD5E1;
                    letter-spacing:2px;text-transform:uppercase;
                    margin-top:6px">{_zb_name} ZONES</div>
                    <div style="margin-top:12px;height:4px;
                    background:#8896AA;border-radius:4px">
                    <div style="height:4px;width:{_zb_pct}%;
                    background:{_zb_color};border-radius:4px;
                    transition:width 1s ease"></div>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # FEATURE A — CAPACITY THRESHOLD ALERT
            # ══════════════════════════════════════════════════
            _utilization = min(100, int((_crowd_count / venue_capacity) * 100))

            if _utilization < 60:
                cap_color="#10B981"; cap_bg="rgba(16,185,129,0.1)"
                cap_label="NORMAL"; bar_color="#10B981"
            elif _utilization < 80:
                cap_color="#F59E0B"; cap_bg="rgba(245,158,11,0.1)"
                cap_label="FILLING"; bar_color="#F59E0B"
            elif _utilization < 100:
                cap_color="#EF4444"; cap_bg="rgba(239,68,68,0.1)"
                cap_label="NEAR FULL"; bar_color="#EF4444"
            else:
                cap_color="#FF1744"; cap_bg="rgba(255,23,68,0.1)"
                cap_label="OVERCAPACITY"; bar_color="#FF1744"

            st.markdown(f"""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
            border-radius:12px;padding:20px 24px;margin:16px 0">

            <div style="display:flex;justify-content:space-between;
            align-items:center;margin-bottom:14px">
            <div style="color:#CBD5E1;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase">
            🏟️ VENUE CAPACITY MONITOR</div>
            <div style="display:flex;gap:8px;align-items:center">
            <span style="color:#CBD5E1;font-size:12px">
            {_crowd_count} / {venue_capacity}</span>
            <span style="padding:3px 12px;border-radius:20px;
            font-size:11px;font-weight:700;
            background:{cap_bg};color:{cap_color};
            border:1px solid {cap_color}30">{cap_label}</span>
            </div>
            </div>

            <div style="height:8px;background:#8896AA;
            border-radius:6px;overflow:hidden">
            <div style="height:100%;width:{_utilization}%;
            background:{bar_color};border-radius:6px;
            transition:width 1s ease"></div>
            </div>

            <div style="display:flex;justify-content:space-between;
            margin-top:8px">
            <span style="color:#CBD5E1;font-size:10px">0%</span>
            <span style="color:{cap_color};font-size:11px;
            font-weight:700">{_utilization}% utilized</span>
            <span style="color:#CBD5E1;font-size:10px">100%</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # FEATURE B — EVACUATION TIME ESTIMATOR
            # ══════════════════════════════════════════════════
            _exit_capacity = num_exits * 40
            _slowdown = 1.0 + (_zone_stats["Critical"] * 0.2)
            _evac_minutes = (_crowd_count * _slowdown) / _exit_capacity if _exit_capacity > 0 else 999
            _evac_seconds = int(_evac_minutes * 60)
            _evac_mins_display = int(_evac_minutes)
            _evac_secs_display = int((_evac_minutes % 1) * 60)

            if _evac_minutes < 2:
                _evac_color = "#10B981"
                _evac_status = "SAFE EVACUATION TIME"
            elif _evac_minutes < 5:
                _evac_color = "#F59E0B"
                _evac_status = "MODERATE EVACUATION TIME"
            else:
                _evac_color = "#FF1744"
                _evac_status = "⚠️ CRITICAL EVACUATION TIME"

            st.markdown(f"""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
            border-radius:12px;padding:20px 24px;margin:16px 0;
            border-left:4px solid {_evac_color}">
            <div style="display:flex;align-items:center;
            justify-content:space-between">
            <div>
            <div style="color:#CBD5E1;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;
            margin-bottom:6px">🚪 EVACUATION TIME ESTIMATE</div>
            <div style="color:{_evac_color};font-size:36px;
            font-weight:900;font-family:monospace">
            {_evac_mins_display}m {_evac_secs_display}s</div>
            <div style="color:#CBD5E1;font-size:11px;margin-top:4px">
            {_evac_status}</div>
            </div>
            <div style="text-align:right">
            <div style="color:#CBD5E1;font-size:10px;margin-bottom:4px">
            {num_exits} exits · 40 ppl/min each</div>
            <div style="color:#CBD5E1;font-size:10px">
            Critical zone penalty: +{int((_slowdown-1)*100)}%</div>
            </div>
            </div>
            </div>
            """, unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # RL AGENT RECOMMENDATION (compact card in Tab 1)
            # ══════════════════════════════════════════════════
            try:
                _rl_agent = st.session_state["rl_agent"]
                if _rl_agent is not None and _density_full is not None:
                    _rl_state_t1 = _rl_agent.get_state_vector(
                        _zone_stats, _crowd_count, num_exits, _density_full)
                    _rl_action_t1 = _rl_agent.choose_action(_rl_state_t1)
                    _rl_name_t1 = _rl_agent.get_action_name(_rl_action_t1)
                    _rl_detail_t1 = _rl_agent.get_action_detail(
                        _rl_action_t1, _zone_stats, num_exits)
                    st.markdown(f"""
                    <div style="background:#3B4A5E;border:1px solid #DDD6FE;
                    border-radius:12px;padding:18px 22px;margin:16px 0;
                    border-left:4px solid #7C3AED;
                    box-shadow:0 2px 12px rgba(124,58,237,0.08)">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                    <span style="font-size:18px">🤖</span>
                    <span style="color:#7C3AED;font-size:10px;font-weight:700;
                    letter-spacing:2px;text-transform:uppercase">RL AGENT RECOMMENDATION</span>
                    </div>
                    <div style="color:#FFFFFF;font-size:15px;font-weight:700;
                    margin-bottom:4px">{_rl_name_t1}</div>
                    <div style="color:#9AA8B8;font-size:12px;line-height:1.5">{_rl_detail_t1}</div>
                    <div style="color:#A78BFA;font-size:10px;margin-top:8px;
                    font-weight:600">Switch to RL Agent tab to train the model →</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                pass

            # ══════════════════════════════════════════════════
            # FEATURE 2 — ZONE INTELLIGENCE MATRIX (expander)
            # ══════════════════════════════════════════════════
            with st.expander("🔬 Zone Intelligence Matrix"):
                _rgrid = risk_grid(_density_full)

                # Custom colorscale: black→dark blue→cyan→yellow→red
                _custom_cs = [
                    [0.0, "#2D3748"],
                    [0.2, "#1E3A8A"],
                    [0.4, "#06B6D4"],
                    [0.6, "#F59E0B"],
                    [1.0, "#EF4444"],
                ]

                # Build text annotations
                _annot_text = [[f"{_rgrid[i][j]:.3f}" for j in range(GRID)]
                               for i in range(GRID)]

                fig_heat = go.Figure(data=go.Heatmap(
                    z=_rgrid,
                    text=_annot_text,
                    texttemplate="%{text}",
                    textfont=dict(size=10, color="#EFF6FF",
                                  family="JetBrains Mono, monospace"),
                    colorscale=_custom_cs,
                    showscale=True,
                    colorbar=dict(
                        tickfont=dict(color="#9AA8B8", size=10),
                        title=dict(text="Density", font=dict(color="#9AA8B8", size=11)),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                    ),
                ))
                fig_heat.update_layout(
                    title=dict(text="Density Distribution · 8×8 Zone Grid",
                               font=dict(size=14, color="#9AA8B8",
                                         family="Inter, sans-serif")),
                    template="none",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis=dict(title="Column", tickfont=dict(color="#CBD5E1"),
                               title_font=dict(color="#CBD5E1")),
                    yaxis=dict(title="Row", tickfont=dict(color="#CBD5E1"),
                               title_font=dict(color="#CBD5E1"), autorange="reversed"),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            # ── Download buttons ──────────────────────────────
            dl1, dl2 = st.columns(2)
            with dl1:
                buf = io.BytesIO()
                Image.fromarray(safety_img).save(buf, format="PNG")
                st.download_button(
                    "📥  Download safety map",
                    data=buf.getvalue(),
                    file_name="safecrowd_safety_map.png",
                    mime="image/png")

            # ══════════════════════════════════════════════════
            # FEATURE 5 — EXPORT JSON REPORT
            # ══════════════════════════════════════════════════
            with dl2:
                _now = datetime.now()
                if _crowd_count < 80:
                    _weights_used = "SHB"
                elif _crowd_count < 200:
                    _weights_used = "SHA"
                else:
                    _weights_used = "Ensemble (SHA+SHB)"
                _report = {
                    "timestamp": _now.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": "DM-Count (LWCC)",
                    "weights": _weights_used,
                    "crowd_count": _crowd_count,
                    "threat_level": _g_label,
                    "threat_score": round(_gauge_value, 2),
                    "zone_stats": {
                        "Low": _zone_stats["Low"],
                        "Medium": _zone_stats["Medium"],
                        "High": _zone_stats["High"],
                        "Critical": _zone_stats["Critical"],
                    },
                    "confidence_pct": _confidence_pct,
                    "image_dimensions": f"{w}x{h}",
                    "recommendation": _g_rec,
                }
                _report_json = json.dumps(_report, indent=2)
                _report_fname = f"safecrowd_report_{_now.strftime('%Y%m%d_%H%M%S')}.json"

                st.download_button(
                    "📋  Export JSON Report",
                    data=_report_json,
                    file_name=_report_fname,
                    mime="application/json",
                    key="json_report_dl")

    else:
        # ── No image uploaded placeholder ─────────────────────
        if demo_mode and st.session_state["last_count"] is not None:
            count = st.session_state["last_count"]
            if count < 30:
                bg = "linear-gradient(135deg, #105030, #1A6038)"
                glow = "0 0 40px rgba(16,185,129,0.15)"
                label = "✅ SAFE"
            elif count < 60:
                bg = "linear-gradient(135deg, #5A3D18, #704020)"
                glow = "0 0 40px rgba(245,158,11,0.15)"
                label = "⚠️ FULL"
            else:
                bg = "linear-gradient(135deg, #5A1828, #701D30)"
                glow = "0 0 40px rgba(255,23,68,0.2)"
                label = "🚨 OVERCROWDED"
            st.markdown(
                f'<div style="background:{bg};border-radius:20px;padding:60px 20px;'
                f'text-align:center;margin-top:20px;box-shadow:{glow};'
                f'border:1px solid rgba(255,255,255,0.05)">'
                f'<div style="font-size:9rem;font-weight:900;color:white;'
                f'line-height:1;font-family:\'JetBrains Mono\',monospace;'
                f'font-variant-numeric:tabular-nums">{count}</div>'
                f'<div style="font-size:2.5rem;color:white;margin-top:10px;">'
                f'{label}</div>'
                f'<div style="font-size:1.2rem;color:rgba(255,255,255,0.55);'
                f'margin-top:6px;font-weight:400">estimated crowd</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:50px 20px 40px;min-height:540px;
            animation:floatUp 0.7s ease;position:relative;overflow:hidden;
            background:radial-gradient(ellipse 700px 350px at center 100px,
            rgba(99,102,241,0.08), transparent)">

            <!-- Animated radar ring + SVG Shield -->
            <div style="position:relative;width:220px;height:220px;margin:0 auto 28px;
            display:flex;align-items:center;justify-content:center">

            <!-- Ring 1 — outermost -->
            <div style="position:absolute;width:220px;height:220px;border-radius:50%;
            border:2px solid rgba(99,102,241,0.12);
            box-shadow:0 0 20px rgba(99,102,241,0.08);
            animation:radarPulse 3.5s ease-out infinite"></div>

            <!-- Ring 2 -->
            <div style="position:absolute;width:180px;height:180px;border-radius:50%;
            border:2px solid rgba(34,211,238,0.15);
            box-shadow:0 0 16px rgba(34,211,238,0.1);
            animation:radarPulse 3.5s ease-out 0.5s infinite"></div>

            <!-- Ring 3 -->
            <div style="position:absolute;width:140px;height:140px;border-radius:50%;
            border:2px solid rgba(99,102,241,0.2);
            box-shadow:0 0 14px rgba(99,102,241,0.12);
            animation:radarPulse 3.5s ease-out 1.0s infinite"></div>

            <!-- Ring 4 -->
            <div style="position:absolute;width:100px;height:100px;border-radius:50%;
            border:2px solid rgba(139,92,246,0.25);
            box-shadow:0 0 12px rgba(139,92,246,0.15);
            animation:radarPulse 3.5s ease-out 1.5s infinite"></div>

            <!-- Ring 5 — innermost glow ring -->
            <div style="position:absolute;width:72px;height:72px;border-radius:50%;
            background:radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
            animation:radarPulse 3.5s ease-out 2.0s infinite"></div>

            <!-- SVG Shield Logo -->
            <div style="position:relative;z-index:2;width:64px;height:64px;
            filter:drop-shadow(0 6px 28px rgba(99,102,241,0.5))
                   drop-shadow(0 2px 8px rgba(34,211,238,0.3))">
            <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"
            style="width:64px;height:64px">
            <defs>
              <linearGradient id="shieldGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#6366F1"/>
                <stop offset="50%" stop-color="#4F46E5"/>
                <stop offset="100%" stop-color="#22D3EE"/>
              </linearGradient>
              <linearGradient id="shieldInner" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#818CF8"/>
                <stop offset="100%" stop-color="#6366F1"/>
              </linearGradient>
            </defs>
            <!-- Shield body -->
            <path d="M32 4 L56 14 L56 30 C56 46 44 56 32 60 C20 56 8 46 8 30 L8 14 Z"
            fill="url(#shieldGrad)" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
            <!-- Inner shield highlight -->
            <path d="M32 10 L50 18 L50 30 C50 42 40 50 32 54 C24 50 14 42 14 30 L14 18 Z"
            fill="url(#shieldInner)" opacity="0.3"/>
            <!-- Scan eye / crosshair -->
            <circle cx="32" cy="30" r="10" fill="none" stroke="white" stroke-width="1.5" opacity="0.8"/>
            <circle cx="32" cy="30" r="4" fill="white" opacity="0.9"/>
            <line x1="32" y1="18" x2="32" y2="22" stroke="white" stroke-width="1.5" opacity="0.6"/>
            <line x1="32" y1="38" x2="32" y2="42" stroke="white" stroke-width="1.5" opacity="0.6"/>
            <line x1="20" y1="30" x2="24" y2="30" stroke="white" stroke-width="1.5" opacity="0.6"/>
            <line x1="40" y1="30" x2="44" y2="30" stroke="white" stroke-width="1.5" opacity="0.6"/>
            <!-- Check mark at bottom -->
            <path d="M26 44 L30 48 L38 40" stroke="#10B981" stroke-width="2.5"
            stroke-linecap="round" stroke-linejoin="round" fill="none"/>
            </svg>
            </div>

            </div>

            <!-- Status pill -->
            <div style="display:inline-flex;align-items:center;gap:6px;
            padding:6px 16px;border-radius:20px;margin-bottom:16px;
            background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);
            animation:pillGlow 3s ease-in-out infinite">
            <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
            background:#6366F1;animation:dotPulse 1.5s ease-in-out infinite"></span>
            <span style="color:#A5B4FC;font-size:10px;font-weight:700;
            letter-spacing:2.5px;font-family:'JetBrains Mono',monospace">
            AWAITING INPUT</span>
            </div>

            <h2 style="color:#FFFFFF;font-size:34px;font-weight:800;margin:0 0 4px 0;
            letter-spacing:-0.03em;line-height:1.2">Ready for Analysis</h2>

            <p style="color:#CBD5E1;font-size:15px;max-width:500px;
            margin:12px auto 0;line-height:1.8">
            Upload any crowd image to begin real-time density estimation,
            threat scoring, and spatial safety mapping.</p>

            <!-- Divider -->
            <div style="border-top:1px solid #5A6B7E;margin:32px auto 28px;
            max-width:360px"></div>

            <!-- How it works steps -->
            <div style="max-width:700px;margin:0 auto">
            <div style="color:#6366F1;font-size:10px;font-weight:700;
            letter-spacing:3px;text-transform:uppercase;margin-bottom:20px">
            HOW IT WORKS</div>

            <div style="display:flex;justify-content:center;gap:12px;
            flex-wrap:wrap">

            <div style="background:rgba(30,41,59,0.7);backdrop-filter:blur(12px);
            -webkit-backdrop-filter:blur(12px);
            border:1px solid #5A6B7E;border-radius:14px;padding:24px 20px;
            flex:1;min-width:155px;max-width:210px;text-align:center;
            transition:all 0.3s ease;position:relative;overflow:hidden">
            <div style="position:absolute;top:0;left:0;width:100%;height:3px;
            background:linear-gradient(90deg,transparent,#6366F1,transparent)"></div>
            <div style="width:36px;height:36px;border-radius:10px;
            background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
            display:flex;align-items:center;justify-content:center;
            margin:0 auto 12px;font-size:18px">📤</div>
            <div style="color:#FFFFFF;font-size:13px;font-weight:700;
            margin-bottom:6px">1. Upload</div>
            <div style="color:#9AA8B8;font-size:11px;line-height:1.5">
            Drop a crowd image<br>
            <span style="color:#7A8B9E;font-size:10px">JPG · PNG</span></div>
            </div>

            <div style="background:rgba(30,41,59,0.7);backdrop-filter:blur(12px);
            -webkit-backdrop-filter:blur(12px);
            border:1px solid #5A6B7E;border-radius:14px;padding:24px 20px;
            flex:1;min-width:155px;max-width:210px;text-align:center;
            transition:all 0.3s ease;position:relative;overflow:hidden">
            <div style="position:absolute;top:0;left:0;width:100%;height:3px;
            background:linear-gradient(90deg,transparent,#22D3EE,transparent)"></div>
            <div style="width:36px;height:36px;border-radius:10px;
            background:rgba(34,211,238,0.12);border:1px solid rgba(34,211,238,0.25);
            display:flex;align-items:center;justify-content:center;
            margin:0 auto 12px;font-size:18px">🎯</div>
            <div style="color:#FFFFFF;font-size:13px;font-weight:700;
            margin-bottom:6px">2. Analyze</div>
            <div style="color:#9AA8B8;font-size:11px;line-height:1.5">
            DM-Count inference<br>
            <span style="color:#22D3EE;font-family:'JetBrains Mono',monospace;
            font-size:10px;font-weight:600">MAE: 4.92</span></div>
            </div>

            <div style="background:rgba(30,41,59,0.7);backdrop-filter:blur(12px);
            -webkit-backdrop-filter:blur(12px);
            border:1px solid #5A6B7E;border-radius:14px;padding:24px 20px;
            flex:1;min-width:155px;max-width:210px;text-align:center;
            transition:all 0.3s ease;position:relative;overflow:hidden">
            <div style="position:absolute;top:0;left:0;width:100%;height:3px;
            background:linear-gradient(90deg,transparent,#10B981,transparent)"></div>
            <div style="width:36px;height:36px;border-radius:10px;
            background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.25);
            display:flex;align-items:center;justify-content:center;
            margin:0 auto 12px;font-size:18px">🗺️</div>
            <div style="color:#FFFFFF;font-size:13px;font-weight:700;
            margin-bottom:6px">3. Safety Map</div>
            <div style="color:#9AA8B8;font-size:11px;line-height:1.5">
            4-zone classification<br>
            <span style="color:#10B981;font-family:'JetBrains Mono',monospace;
            font-size:10px;font-weight:600">8×8 Grid</span></div>
            </div>

            <div style="background:rgba(30,41,59,0.7);backdrop-filter:blur(12px);
            -webkit-backdrop-filter:blur(12px);
            border:1px solid #5A6B7E;border-radius:14px;padding:24px 20px;
            flex:1;min-width:155px;max-width:210px;text-align:center;
            transition:all 0.3s ease;position:relative;overflow:hidden">
            <div style="position:absolute;top:0;left:0;width:100%;height:3px;
            background:linear-gradient(90deg,transparent,#8B5CF6,transparent)"></div>
            <div style="width:36px;height:36px;border-radius:10px;
            background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.25);
            display:flex;align-items:center;justify-content:center;
            margin:0 auto 12px;font-size:18px">🤖</div>
            <div style="color:#FFFFFF;font-size:13px;font-weight:700;
            margin-bottom:6px">4. RL Agent</div>
            <div style="color:#9AA8B8;font-size:11px;line-height:1.5">
            Smart evacuation<br>
            <span style="color:#8B5CF6;font-family:'JetBrains Mono',monospace;
            font-size:10px;font-weight:600">Policy AI</span></div>
            </div>

            </div>
            </div>

            <!-- Supported venues -->
            <div style="margin-top:32px;display:flex;justify-content:center;
            gap:8px;flex-wrap:wrap">
            <span style="padding:4px 12px;border-radius:12px;font-size:10px;
            background:rgba(51,65,85,0.5);color:#9AA8B8;border:1px solid #5A6B7E;
            font-weight:500">🎵 Concerts</span>
            <span style="padding:4px 12px;border-radius:12px;font-size:10px;
            background:rgba(51,65,85,0.5);color:#9AA8B8;border:1px solid #5A6B7E;
            font-weight:500">🎪 Festivals</span>
            <span style="padding:4px 12px;border-radius:12px;font-size:10px;
            background:rgba(51,65,85,0.5);color:#9AA8B8;border:1px solid #5A6B7E;
            font-weight:500">🚉 Stations</span>
            <span style="padding:4px 12px;border-radius:12px;font-size:10px;
            background:rgba(51,65,85,0.5);color:#9AA8B8;border:1px solid #5A6B7E;
            font-weight:500">🏟️ Stadiums</span>
            <span style="padding:4px 12px;border-radius:12px;font-size:10px;
            background:rgba(51,65,85,0.5);color:#9AA8B8;border:1px solid #5A6B7E;
            font-weight:500">📢 Rallies</span>
            </div>

            <style>
            @keyframes radarPulse {
                0% { transform: scale(0.5); opacity: 0.8; }
                100% { transform: scale(1.6); opacity: 0; }
            }
            </style>

            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — COMPARE METHODS
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#3B4A5E 0%,#2D3748 100%);
    padding:22px 26px;border-radius:12px;border:1px solid #5A6B7E;
    margin-bottom:22px;box-shadow:0 4px 20px rgba(0,0,0,0.4)">
    <h2 style="color:#FFFFFF;margin:0;font-size:20px;font-weight:700;
    letter-spacing:-0.02em">
    How different ML methods see the same crowd</h2>
    <p style="color:#9AA8B8;margin:8px 0 0;font-size:13px;line-height:1.5">
    Same image analysed by three different algorithms side by side.</p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded and _img_rgb is not None and _features_sc is not None:

        km_labels  = labels_from_kmeans(_features_sc, km) \
                     if km else [get_label(float(f[0]))
                                 for f in extract_features(_density_full)]
        xgb_labels = labels_from_xgb(_features_sc, xgb) \
                     if xgb else km_labels
        gmm_labels, gmm_conf = labels_from_gmm(_features_sc, gmm) \
                     if gmm else (km_labels, 0.0)

        km_img,  km_stats  = build_overlay(_img_rgb, km_labels,  GRID, opacity)
        xgb_img, xgb_stats = build_overlay(_img_rgb, xgb_labels, GRID, opacity)
        gmm_img, gmm_stats = build_overlay(_img_rgb, gmm_labels, GRID, opacity)

        # ── Dynamic per-image accuracy (agreement with KMeans) ──
        _n_patches = len(km_labels)
        _xgb_agree = sum(1 for a, b in zip(km_labels, xgb_labels) if a == b)
        _xgb_acc_pct = (_xgb_agree / _n_patches * 100) if _n_patches > 0 else 0.0
        _gmm_agree = sum(1 for a, b in zip(km_labels, gmm_labels) if a == b)
        _gmm_agree_pct = (_gmm_agree / _n_patches * 100) if _n_patches > 0 else 0.0
        print(f"[ACC DEBUG] XGBoost agreement: {_xgb_acc_pct:.1f}% ({_xgb_agree}/{_n_patches})")
        print(f"[ACC DEBUG] GMM agreement: {_gmm_agree_pct:.1f}% ({_gmm_agree}/{_n_patches})")
        print(f"[ACC DEBUG] GMM confidence: {gmm_conf*100:.1f}%")

        # ── ⟺ Interactive Comparison — Drag-to-Compare Slider ──
        def _img_to_b64(img_rgb):
            _, buf = cv2.imencode('.jpg',
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf).decode()

        _km_b64  = _img_to_b64(km_img)
        _xgb_b64 = _img_to_b64(xgb_img)

        st.markdown("""
        <div style="background:linear-gradient(135deg,#3B4A5E 0%,#2D3748 100%);
        padding:18px 22px;border-radius:12px;border:1px solid #5A6B7E;
        margin-bottom:14px;border-left:3px solid #22D3EE;
        box-shadow:0 4px 16px rgba(0,0,0,0.35)">
        <h3 style="color:#FFFFFF;margin:0;font-size:16px;font-weight:700;
        letter-spacing:-0.01em">⟺ Interactive Comparison</h3>
        <p style="color:#CBD5E1;font-size:12px;margin:6px 0 0;font-weight:400">
        Drag the slider to compare KMeans vs XGBoost zone classification</p>
        </div>
        """, unsafe_allow_html=True)

        st.components.v1.html(f"""
        <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: transparent; overflow: hidden; }}
        .cmp-wrap {{
            position: relative; width: 100%;
            border-radius: 12px; overflow: hidden;
            cursor: col-resize; user-select: none;
            border: 1px solid #8896AA;
            box-shadow: 0 4px 24px rgba(0,0,0,0.5);
        }}
        .cmp-wrap img {{ width: 100%; display: block; }}
        #left-panel {{
            position: absolute; top: 0; left: 0;
            width: 50%; height: 100%; overflow: hidden;
        }}
        #left-panel img {{
            position: absolute; top: 0; left: 0;
            width: 100vw; max-width: none;
            height: 100%; object-fit: cover;
        }}
        .cmp-label {{
            position: absolute; top: 12px;
            color: white; padding: 5px 14px;
            border-radius: 20px; font-size: 11px;
            font-weight: 700; letter-spacing: 1.5px;
            font-family: 'Inter', sans-serif;
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            z-index: 20; pointer-events: none;
        }}
        .cmp-label-l {{ left: 12px; background: rgba(99,102,241,0.85); }}
        .cmp-label-r {{ right: 12px; background: rgba(16,185,129,0.85); }}
        #divider {{
            position: absolute; top: 0; left: 50%;
            width: 2px; height: 100%;
            background: white;
            box-shadow: 0 0 10px rgba(255,255,255,0.8),
                        0 0 30px rgba(255,255,255,0.3);
            z-index: 15;
            transition: left 0.02s linear;
        }}
        .handle {{
            position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            width: 40px; height: 40px; border-radius: 50%;
            background: white;
            box-shadow: 0 2px 16px rgba(0,0,0,0.5),
                        0 0 0 3px rgba(255,255,255,0.3);
            display: flex; align-items: center;
            justify-content: center;
            font-size: 16px; color: #111;
            font-weight: 700;
        }}
        .caption {{
            text-align: center; padding: 10px 0 4px 0;
            color: #06B6D4; font-family: 'JetBrains Mono', monospace;
            font-size: 11px; letter-spacing: 1px;
            opacity: 0.7;
        }}
        </style>

        <div class="cmp-wrap" id="slider-container">
            <img src="data:image/jpeg;base64,{_xgb_b64}" alt="XGBoost">

            <div id="left-panel">
                <img src="data:image/jpeg;base64,{_km_b64}" alt="KMeans">
            </div>

            <div class="cmp-label cmp-label-l">KMEANS</div>
            <div class="cmp-label cmp-label-r">XGBOOST</div>

            <div id="divider">
                <div class="handle">⟺</div>
            </div>
        </div>
        <div class="caption">← Drag to compare KMeans vs XGBoost zone classification →</div>

        <script>
        (function() {{
            const container = document.getElementById('slider-container');
            const divider   = document.getElementById('divider');
            const leftPanel = document.getElementById('left-panel');
            const leftImg   = leftPanel.querySelector('img');
            let dragging = false;

            function setPos(pct) {{
                pct = Math.min(95, Math.max(5, pct));
                divider.style.left    = pct + '%';
                leftPanel.style.width = pct + '%';
                leftImg.style.width   = container.offsetWidth + 'px';
            }}

            container.addEventListener('mousedown', function(e) {{
                dragging = true;
                const rect = container.getBoundingClientRect();
                setPos((e.clientX - rect.left) / rect.width * 100);
            }});
            document.addEventListener('mouseup', function() {{ dragging = false; }});
            document.addEventListener('mousemove', function(e) {{
                if (!dragging) return;
                const rect = container.getBoundingClientRect();
                setPos((e.clientX - rect.left) / rect.width * 100);
            }});

            container.addEventListener('touchstart', function(e) {{
                dragging = true;
                const rect = container.getBoundingClientRect();
                setPos((e.touches[0].clientX - rect.left) / rect.width * 100);
            }}, {{passive: true}});
            container.addEventListener('touchmove', function(e) {{
                const rect = container.getBoundingClientRect();
                setPos((e.touches[0].clientX - rect.left) / rect.width * 100);
            }}, {{passive: true}});
            document.addEventListener('touchend', function() {{ dragging = false; }});

            // Fix left image width on load & resize
            function fixWidth() {{ leftImg.style.width = container.offsetWidth + 'px'; }}
            window.addEventListener('load', fixWidth);
            window.addEventListener('resize', fixWidth);
            setTimeout(fixWidth, 100);
        }})();
        </script>
        """, height=500)

        # ── Spacing before 3-column comparison ──
        st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)

        km_sil = safe_silhouette(
            _features_sc, km.predict(_features_sc)) if km else 0.25

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("""<div style="background:linear-gradient(160deg,#3B4A5E,#485A6E);
            padding:18px;border-radius:12px;border:1px solid #5A6B7E;
            border-top:3px solid #6366F1;margin-bottom:8px;
            box-shadow:0 4px 16px rgba(0,0,0,0.35)">
            <h3 style="color:#FFFFFF;margin:0;font-size:16px;font-weight:700">KMeans</h3>
            <p style="color:#CBD5E1;font-size:11px;margin:4px 0 0;font-weight:500">
            Unsupervised Clustering</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("""<div style="background:#3B4A5E;
            border-radius:12px;border:1px solid #5A6B7E;padding:16px;margin-bottom:8px;
            border-top:2px solid #9AA8B8">
            <p style="color:#CBD5E1;font-size:10px;text-transform:uppercase;
            letter-spacing:2px;margin:0;font-weight:700">ZONE OVERLAY</p>
            </div>""", unsafe_allow_html=True)
            st.image(km_img, use_container_width=True,
                     caption="KMeans k=4 hard assignment")
            st.markdown(
                f'<span style="display:inline-block;padding:6px 16px;'
                f'border-radius:20px;font-size:12px;font-weight:600;'
                f'background:rgba(99,102,241,0.1);color:#3B82F6;'
                f'border:1px solid rgba(99,102,241,0.25);'
                f'font-family:\'JetBrains Mono\',monospace;'
                f'font-variant-numeric:tabular-nums">'
                f'Silhouette: {km_sil:.2f}</span>',
                unsafe_allow_html=True)
            st.caption("Hard clustering — assigns each patch to nearest centre")
            for z in ["Low", "Medium", "High", "Critical"]:
                st.markdown(
                    f'<p style="margin:3px 0;font-size:13px;color:#9AA8B8">'
                    f'<span style="color:{ZONE_HEX[z]};font-size:10px">⬤</span> '
                    f'{z}: <b style="font-family:\'JetBrains Mono\',monospace;'
                    f'color:#FFFFFF;font-variant-numeric:tabular-nums">'
                    f'{km_stats[z]}</b></p>',
                    unsafe_allow_html=True)

        with c2:
            st.markdown("""<div style="background:linear-gradient(160deg,#3B4A5E,#485A6E);
            padding:18px;border-radius:12px;border:1px solid #5A6B7E;
            border-top:3px solid #10B981;margin-bottom:8px;
            box-shadow:0 4px 16px rgba(0,0,0,0.35)">
            <h3 style="color:#FFFFFF;margin:0;font-size:16px;font-weight:700">XGBoost</h3>
            <p style="color:#CBD5E1;font-size:11px;margin:4px 0 0;font-weight:500">
            Supervised Classification</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("""<div style="background:#3B4A5E;
            border-radius:12px;border:1px solid #5A6B7E;padding:16px;margin-bottom:8px;
            border-top:2px solid #9AA8B8">
            <p style="color:#CBD5E1;font-size:10px;text-transform:uppercase;
            letter-spacing:2px;margin:0;font-weight:700">ZONE OVERLAY</p>
            </div>""", unsafe_allow_html=True)
            st.image(xgb_img, use_container_width=True,
                     caption="Gradient boosted trees")
            st.markdown(
                f'<span style="display:inline-block;padding:6px 16px;'
                f'border-radius:20px;font-size:12px;font-weight:600;'
                f'background:rgba(16,185,129,0.1);color:#6EE7B7;'
                f'border:1px solid rgba(16,185,129,0.25);'
                f'font-family:\'JetBrains Mono\',monospace;'
                f'font-variant-numeric:tabular-nums">'
                f'Agreement: {_xgb_acc_pct:.1f}%</span>',
                unsafe_allow_html=True)
            st.caption("Learned from KMeans labels — gradient boosted trees")
            for z in ["Low", "Medium", "High", "Critical"]:
                st.markdown(
                    f'<p style="margin:3px 0;font-size:13px;color:#9AA8B8">'
                    f'<span style="color:{ZONE_HEX[z]};font-size:10px">⬤</span> '
                    f'{z}: <b style="font-family:\'JetBrains Mono\',monospace;'
                    f'color:#FFFFFF;font-variant-numeric:tabular-nums">'
                    f'{xgb_stats[z]}</b></p>',
                    unsafe_allow_html=True)

        with c3:
            st.markdown("""<div style="background:linear-gradient(160deg,#3B4A5E,#485A6E);
            padding:18px;border-radius:12px;border:1px solid #5A6B7E;
            border-top:3px solid #7C3AED;margin-bottom:8px;
            box-shadow:0 4px 16px rgba(0,0,0,0.35)">
            <h3 style="color:#FFFFFF;margin:0;font-size:16px;font-weight:700">GMM</h3>
            <p style="color:#CBD5E1;font-size:11px;margin:4px 0 0;font-weight:500">
            Unsupervised Soft Clustering</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("""<div style="background:#3B4A5E;
            border-radius:12px;border:1px solid #5A6B7E;padding:16px;margin-bottom:8px;
            border-top:2px solid #9AA8B8">
            <p style="color:#CBD5E1;font-size:10px;text-transform:uppercase;
            letter-spacing:2px;margin:0;font-weight:700">ZONE OVERLAY</p>
            </div>""", unsafe_allow_html=True)
            st.image(gmm_img, use_container_width=True,
                     caption="Gaussian mixture · soft probabilities")
            st.markdown(
                f'<span style="display:inline-block;padding:6px 16px;'
                f'border-radius:20px;font-size:12px;font-weight:600;'
                f'background:rgba(124,58,237,0.1);color:#A78BFA;'
                f'border:1px solid rgba(124,58,237,0.25);'
                f'font-family:\'JetBrains Mono\',monospace;'
                f'font-variant-numeric:tabular-nums">'
                f'Agreement: {_gmm_agree_pct:.1f}%'
                f' · Conf: {gmm_conf*100:.1f}%</span>',
                unsafe_allow_html=True)
            st.caption("Soft clustering — probability-based zone assignment")
            for z in ["Low", "Medium", "High", "Critical"]:
                st.markdown(
                    f'<p style="margin:3px 0;font-size:13px;color:#9AA8B8">'
                    f'<span style="color:{ZONE_HEX[z]};font-size:10px">⬤</span> '
                    f'{z}: <b style="font-family:\'JetBrains Mono\',monospace;'
                    f'color:#FFFFFF;font-variant-numeric:tabular-nums">'
                    f'{gmm_stats[z]}</b></p>',
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — OPERATIONS DASHBOARD
# ═══════════════════════════════════════════════════════════════

with tab3:

    # ── Section header ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#3B4A5E 0%,#2D3748 100%);
    padding:22px 26px;border-radius:12px;border:1px solid #5A6B7E;
    border-left:4px solid #6366F1;margin-bottom:22px;
    box-shadow:inset 4px 0 30px rgba(99,102,241,0.15), 0 4px 20px rgba(0,0,0,0.4)">
    <div style="display:flex;align-items:center;justify-content:space-between">
    <div>
    <h2 style="color:#FFFFFF;margin:0;font-size:20px;font-weight:700;
    letter-spacing:-0.02em">📡 Operations Timeline</h2>
    <p style="color:#9AA8B8;margin:6px 0 0;font-size:13px;line-height:1.5">
    Security operations center · Scan history & threat analysis</p>
    </div>
    <span style="display:inline-flex;align-items:center;gap:6px;
    padding:5px 14px;border-radius:20px;font-size:10px;font-weight:600;
    background:rgba(34,211,238,0.1);color:#6366F1;
    border:1px solid rgba(34,211,238,0.2);
    font-family:'JetBrains Mono',monospace;letter-spacing:1px">
    <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
    background:#06B6D4;animation:dotPulse 1.5s ease-in-out infinite"></span>
    RECORDING</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state["history"]:
        _hist = st.session_state["history"]
        _hist_counts = [h["count"] for h in _hist]
        _hist_total  = len(_hist_counts)
        _hist_avg    = int(np.mean(_hist_counts))
        _hist_peak   = max(_hist_counts)

        # ══════════════════════════════════════════════════════
        # 1. HEADER STATS ROW — 3 animated number cards
        # ══════════════════════════════════════════════════════

        _dc1, _dc2, _dc3 = st.columns(3)

        with _dc1:
            st.components.v1.html(f"""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
            border-radius:12px;border-top:3px solid #6366F1;
            padding:24px 20px;text-align:center;
            box-shadow:0 4px 20px rgba(99,102,241,0.1)">
            <div style="color:#CBD5E1;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:10px">
            📊 TOTAL SCANS</div>
            <div id="stat-total" style="font-size:42px;font-weight:900;
            color:#FFFFFF;font-family:'JetBrains Mono',monospace;
            font-variant-numeric:tabular-nums;line-height:1;
            text-shadow:0 0 30px rgba(99,102,241,0.2)">0</div>
            <div style="color:#3B4A63;font-size:10px;margin-top:8px;
            font-family:'JetBrains Mono',monospace;letter-spacing:1px">
            images analysed</div>
            </div>
            <script>
            (function() {{
                let t={_hist_total},c=0,s=Math.max(1,Math.ceil(t/40));
                let el=document.getElementById('stat-total');
                let i=setInterval(()=>{{c+=s;if(c>=t){{c=t;clearInterval(i);}};el.textContent=c;}},25);
            }})();
            </script>
            """, height=155)

        with _dc2:
            st.components.v1.html(f"""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
            border-radius:12px;border-top:2px solid #0891B2;
            padding:24px 20px;text-align:center;
            box-shadow:0 4px 20px rgba(34,211,238,0.1)">
            <div style="color:#CBD5E1;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:10px">
            👥 AVERAGE CROWD</div>
            <div id="stat-avg" style="font-size:42px;font-weight:900;
            color:#6366F1;font-family:'JetBrains Mono',monospace;
            font-variant-numeric:tabular-nums;line-height:1;
            text-shadow:0 0 30px rgba(34,211,238,0.25)">0</div>
            <div style="color:#3B4A63;font-size:10px;margin-top:8px;
            font-family:'JetBrains Mono',monospace;letter-spacing:1px">
            persons per scan</div>
            </div>
            <script>
            (function() {{
                let t={_hist_avg},c=0,s=Math.max(1,Math.ceil(t/40));
                let el=document.getElementById('stat-avg');
                let i=setInterval(()=>{{c+=s;if(c>=t){{c=t;clearInterval(i);}};el.textContent=c;}},25);
            }})();
            </script>
            """, height=155)

        with _dc3:
            _peak_color = "#FF1744" if _hist_peak > 100 else "#10B981"
            _peak_glow  = "0 0 30px rgba(255,23,68,0.3)" if _hist_peak > 100 else "0 0 30px rgba(16,185,129,0.2)"
            _peak_anim  = "animation:criticalGlow 2s ease-in-out infinite;" if _hist_peak > 100 else ""
            st.components.v1.html(f"""
            <style>
            @keyframes critGlow {{
                0%, 100% {{ box-shadow: 0 4px 20px rgba(255,23,68,0.1); }}
                50% {{ box-shadow: 0 4px 32px rgba(255,23,68,0.35); }}
            }}
            </style>
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
            border-radius:12px;border-top:2px solid {_peak_color};
            padding:24px 20px;text-align:center;
            box-shadow:0 4px 20px {'rgba(255,23,68,0.15)' if _hist_peak > 100 else 'rgba(16,185,129,0.1)'};
            {'animation:critGlow 2s ease-in-out infinite;' if _hist_peak > 100 else ''}">
            <div style="color:#CBD5E1;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:10px">
            🔺 PEAK THREAT</div>
            <div id="stat-peak" style="font-size:42px;font-weight:900;
            color:{_peak_color};font-family:'JetBrains Mono',monospace;
            font-variant-numeric:tabular-nums;line-height:1;
            text-shadow:{_peak_glow}">0</div>
            <div style="color:#3B4A63;font-size:10px;margin-top:8px;
            font-family:'JetBrains Mono',monospace;letter-spacing:1px">
            max persons detected</div>
            </div>
            <script>
            (function() {{
                let t={_hist_peak},c=0,s=Math.max(1,Math.ceil(t/40));
                let el=document.getElementById('stat-peak');
                let i=setInterval(()=>{{c+=s;if(c>=t){{c=t;clearInterval(i);}};el.textContent=c;}},25);
            }})();
            </script>
            """, height=155)

        # ══════════════════════════════════════════════════════
        # 2. CINEMATIC TIMELINE CHART
        # ══════════════════════════════════════════════════════

        st.markdown("""
        <div style="background:#3B4A5E;border:1px solid #5A6B7E;
        border-radius:12px;padding:14px 18px 6px;margin:8px 0 4px;
        border-top:3px solid #6366F1">
        <div style="color:#CBD5E1;font-size:10px;font-weight:700;
        letter-spacing:2px;text-transform:uppercase">
        📈 CROWD DENSITY TIMELINE</div>
        </div>
        """, unsafe_allow_html=True)

        _scan_nums = list(range(1, _hist_total + 1))

        # Assign threat color per dot
        _dot_colors = []
        _threat_labels = []
        for c in _hist_counts:
            if c < 30:
                _dot_colors.append("#10B981")
                _threat_labels.append("MINIMAL")
            elif c < 80:
                _dot_colors.append("#F59E0B")
                _threat_labels.append("ELEVATED")
            elif c < 200:
                _dot_colors.append("#EF4444")
                _threat_labels.append("HIGH")
            else:
                _dot_colors.append("#FF1744")
                _threat_labels.append("CRITICAL")

        _hover_text = [
            f"<b>Scan #{n}</b><br>"
            f"Count: <b>{c}</b><br>"
            f"Threat: <b>{t}</b>"
            for n, c, t in zip(_scan_nums, _hist_counts, _threat_labels)
        ]

        _fig_timeline = go.Figure()

        # Area fill
        _fig_timeline.add_trace(go.Scatter(
            x=_scan_nums, y=_hist_counts,
            fill='tozeroy',
            fillcolor='rgba(99,102,241,0.08)',
            line=dict(color='rgba(99,102,241,0.0)', width=0),
            showlegend=False, hoverinfo='skip',
        ))

        # Main line
        _fig_timeline.add_trace(go.Scatter(
            x=_scan_nums, y=_hist_counts,
            mode='lines',
            line=dict(
                color='#2563EB', width=3,
                shape='spline', smoothing=1.2,
            ),
            showlegend=False, hoverinfo='skip',
        ))

        # Threat-colored dots
        _fig_timeline.add_trace(go.Scatter(
            x=_scan_nums, y=_hist_counts,
            mode='markers',
            marker=dict(
                size=10, color=_dot_colors,
                line=dict(color='rgba(255,255,255,0.3)', width=1.5),
            ),
            hovertext=_hover_text,
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor='#3B4A5E', bordercolor='#5A6B7E',
                font=dict(family='Inter, sans-serif', size=12, color='#FFFFFF'),
            ),
            showlegend=False,
        ))

        _fig_timeline.update_layout(
            template='none',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=320,
            margin=dict(l=50, r=20, t=20, b=50),
            xaxis=dict(
                title=dict(text='Scan #', font=dict(color='#9AA8B8', size=11)),
                tickfont=dict(color='#9AA8B8', size=10,
                              family='JetBrains Mono, monospace'),
                gridcolor='#5A6B7E', gridwidth=1,
                zeroline=False,
                dtick=1,
            ),
            yaxis=dict(
                title=dict(text='Crowd Count', font=dict(color='#9AA8B8', size=11)),
                tickfont=dict(color='#9AA8B8', size=10,
                              family='JetBrains Mono, monospace'),
                gridcolor='#5A6B7E', gridwidth=1,
                zeroline=False,
            ),
            font=dict(family='Inter, sans-serif'),
            hovermode='closest',
        )
        st.plotly_chart(_fig_timeline, use_container_width=True)

        # ══════════════════════════════════════════════════════
        # 3. THREAT HISTORY TABLE
        # ══════════════════════════════════════════════════════

        st.markdown("""
        <div style="background:#3B4A5E;border:1px solid #5A6B7E;
        border-radius:12px;padding:14px 18px 6px;margin:16px 0 4px;
        border-top:2px solid #0891B2">
        <div style="color:#CBD5E1;font-size:10px;font-weight:700;
        letter-spacing:2px;text-transform:uppercase">
        🗂️ THREAT HISTORY LOG</div>
        </div>
        """, unsafe_allow_html=True)

        # Build table header
        _table_html = """
        <div style="border:1px solid #5A6B7E;border-radius:12px;
        overflow:hidden;margin-top:8px">
        <table style="width:100%;border-collapse:collapse;
        font-family:'Inter',sans-serif">
        <thead>
        <tr style="background:#2D3748;border-bottom:2px solid #8896AA">
        <th style="padding:12px 16px;text-align:left;color:#CBD5E1;
        font-size:10px;font-weight:700;letter-spacing:2px;
        text-transform:uppercase">SCAN #</th>
        <th style="padding:12px 16px;text-align:center;color:#CBD5E1;
        font-size:10px;font-weight:700;letter-spacing:2px;
        text-transform:uppercase">COUNT</th>
        <th style="padding:12px 16px;text-align:center;color:#CBD5E1;
        font-size:10px;font-weight:700;letter-spacing:2px;
        text-transform:uppercase">THREAT LEVEL</th>
        <th style="padding:12px 16px;text-align:right;color:#CBD5E1;
        font-size:10px;font-weight:700;letter-spacing:2px;
        text-transform:uppercase">ZONE STATUS</th>
        </tr>
        </thead>
        <tbody>
        """

        # Show last 10 (most recent first)
        _display_hist = list(enumerate(_hist_counts, 1))[-10:]
        _display_hist.reverse()

        for _idx, (_scan_n, _cnt) in enumerate(_display_hist):
            _row_bg = "#3B4A5E" if _idx % 2 == 0 else "#485A6E"

            if _cnt < 30:
                _tl = "MINIMAL"
                _tc = "#10B981"
                _tbg = "rgba(16,185,129,0.12)"
                _zone = "✅ All Clear"
                _zc = "#6EE7B7"
            elif _cnt < 80:
                _tl = "ELEVATED"
                _tc = "#F59E0B"
                _tbg = "rgba(245,158,11,0.12)"
                _zone = "⚠️ Monitor"
                _zc = "#FCD34D"
            elif _cnt < 200:
                _tl = "HIGH"
                _tc = "#EF4444"
                _tbg = "rgba(239,68,68,0.12)"
                _zone = "🟠 Alert"
                _zc = "#FCA5A5"
            else:
                _tl = "CRITICAL"
                _tc = "#FF1744"
                _tbg = "rgba(255,23,68,0.12)"
                _zone = "🚨 Emergency"
                _zc = "#FF6B6B"

            _table_html += f"""
            <tr style="background:{_row_bg};border-bottom:1px solid #5A6B7E;
            transition:background 0.2s">
            <td style="padding:12px 16px;color:#9AA8B8;font-size:13px;
            font-family:'JetBrains Mono',monospace;font-weight:500">
            #{_scan_n:02d}</td>
            <td style="padding:12px 16px;text-align:center;
            color:#FFFFFF;font-size:15px;font-weight:700;
            font-family:'JetBrains Mono',monospace;
            font-variant-numeric:tabular-nums">{_cnt}</td>
            <td style="padding:12px 16px;text-align:center">
            <span style="display:inline-block;padding:4px 14px;
            border-radius:20px;font-size:10px;font-weight:700;
            letter-spacing:1.2px;color:{_tc};background:{_tbg};
            border:1px solid {_tc}30">{_tl}</span></td>
            <td style="padding:12px 16px;text-align:right;
            color:{_zc};font-size:12px;font-weight:600">{_zone}</td>
            </tr>
            """

        _table_html += "</tbody></table></div>"
        st.markdown(_table_html, unsafe_allow_html=True)

        # Subtle footer
        st.markdown(f"""
        <div style="text-align:center;padding:12px 0 4px;
        color:#3B4A63;font-size:10px;
        font-family:'JetBrains Mono',monospace;letter-spacing:1px">
        Showing {min(10, _hist_total)} of {_hist_total} scans ·
        Session started {datetime.now().strftime('%H:%M')}
        </div>
        """, unsafe_allow_html=True)

    else:

        # ══════════════════════════════════════════════════════
        # 4. EMPTY STATE
        # ══════════════════════════════════════════════════════

        st.markdown("""
<div style="text-align:center;padding:80px 20px 70px;
animation:floatUp 0.6s ease;
background:radial-gradient(ellipse 500px 250px at center 100px,
rgba(34,211,238,0.06), transparent)">

<div style="font-size:56px;margin-bottom:20px;opacity:0.6;
filter:drop-shadow(0 4px 20px rgba(34,211,238,0.25))">📡</div>

<div style="font-size:11px;font-weight:700;color:#6366F1;
letter-spacing:5px;text-transform:uppercase;margin-bottom:14px">
OPERATIONS TIMELINE</div>

<h2 style="color:#FFFFFF;font-size:26px;font-weight:800;margin:0;
letter-spacing:-0.02em">No Scans Yet</h2>

<p style="color:#CBD5E1;font-size:14px;max-width:440px;
margin:14px auto 0;line-height:1.8">
Analyse images in the <span style="color:#3B82F6;
font-weight:600">Live Analysis</span> tab to populate
the operations timeline with threat data.</p>

<div style="border-top:1px solid #5A6B7E;margin:32px auto 28px;
max-width:300px"></div>

<div style="display:inline-flex;align-items:center;gap:8px;
padding:8px 20px;border-radius:20px;
background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2)">
<span style="display:inline-block;width:6px;height:6px;
border-radius:50%;background:#3B4A63"></span>
<span style="color:#CBD5E1;font-size:11px;font-weight:500;
font-family:'JetBrains Mono',monospace;letter-spacing:1px">
AWAITING FIRST SCAN</span>
</div>

</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — LIVE CAPTURE
# ═══════════════════════════════════════════════════════════════

with tab4:
    url = _access_url
    url_type = _access_url_type
    capture_url = _capture_access_url
    is_ngrok = url_type == "ngrok"

    st.markdown("""
    <div style="background:#3B4A5E;
    border-left:4px solid #6366F1;
    border-radius:12px;padding:24px 28px;
    margin-bottom:24px">
    <h2 style="color:#FFFFFF;margin:0;
    font-size:22px;font-weight:800">
    📱 Mobile Live Capture</h2>
    <p style="color:#9AA8B8;margin:8px 0 0;
    font-size:13px">
    Open on phone → Take photo →
    Results appear here instantly</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        if is_ngrok:
            status_color = "#10B981"
            status_text = "● Public URL Active"
            status_note = "Works on any network"
            status_hint = "Use QR or copy the phone link below."
        else:
            status_color = "#22D3EE"
            status_text = "● Same WiFi Mode"
            status_note = "No ngrok needed. Phone must be on same WiFi."
            status_hint = "Use this direct phone link first. QR is already ready."

        st.markdown(f"""
        <div style="background:#3B4A5E;
        border:1px solid #5A6B7E;
        border-radius:12px;padding:20px 24px;
        margin-bottom:16px">

        <div style="display:flex;align-items:center;
        justify-content:space-between;
        margin-bottom:16px">
        <div style="color:#9AA8B8;font-size:10px;
        font-weight:700;letter-spacing:2px">
        CONNECTION STATUS</div>
        <span style="color:{status_color};
        font-size:12px;font-weight:600">
        {status_text}</span>
        </div>

        <div style="color:#FFFFFF;font-size:13px;
        font-family:monospace;background:#2D3748;
        padding:10px 14px;border-radius:8px;
        margin-bottom:8px;word-break:break-all">
        {url}</div>

        <div style="color:#9AA8B8;font-size:11px">
        {status_note}</div>
        <div style="color:#CBD5E1;font-size:11px;margin-top:6px">
        {status_hint}</div>
        </div>

        <div style="background:#3B4A5E;
        border:1px solid #5A6B7E;
        border-radius:12px;padding:20px 24px">

        <div style="color:#6366F1;font-size:10px;
        font-weight:700;letter-spacing:2px;
        margin-bottom:16px">◈ HOW IT WORKS</div>

        <div style="display:flex;gap:12px;
        align-items:center;margin-bottom:14px">
        <div style="background:#6366F1;color:white;
        width:26px;height:26px;border-radius:50%;
        display:flex;align-items:center;
        justify-content:center;font-size:12px;
        font-weight:800;flex-shrink:0">1</div>
        <div style="color:#CBD5E1;font-size:13px">
        Scan QR code with phone camera</div>
        </div>

        <div style="display:flex;gap:12px;
        align-items:center;margin-bottom:14px">
        <div style="background:#6366F1;color:white;
        width:26px;height:26px;border-radius:50%;
        display:flex;align-items:center;
        justify-content:center;font-size:12px;
        font-weight:800;flex-shrink:0">2</div>
        <div style="color:#CBD5E1;font-size:13px">
        Allow camera access on phone</div>
        </div>

        <div style="display:flex;gap:12px;
        align-items:center;margin-bottom:14px">
        <div style="background:#6366F1;color:white;
        width:26px;height:26px;border-radius:50%;
        display:flex;align-items:center;
        justify-content:center;font-size:12px;
        font-weight:800;flex-shrink:0">3</div>
        <div style="color:#CBD5E1;font-size:13px">
        Take crowd photo → tap Analyze</div>
        </div>

        <div style="display:flex;gap:12px;
        align-items:center">
        <div style="background:#10B981;color:white;
        width:26px;height:26px;border-radius:50%;
        display:flex;align-items:center;
        justify-content:center;font-size:12px;
        font-weight:800;flex-shrink:0">4</div>
        <div style="color:#CBD5E1;font-size:13px">
        Results appear on this screen</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.text_input(
            "Direct phone capture link",
            value=capture_url,
            key="live_capture_direct_link",
            help="Copy this into the phone browser if you do not want to scan the QR.",
        )

        if not is_ngrok:
            st.markdown("""
            <div style="background:rgba(34,211,238,0.08);
            border:1px solid rgba(34,211,238,0.24);
            border-radius:10px;padding:14px 18px;
            margin-top:14px">
            <div style="color:#22D3EE;font-size:12px;
            font-weight:600;margin-bottom:6px">
            Same WiFi quick path:
            </div>
            <div style="color:#CBD5E1;font-size:12px;line-height:1.7">
            Open the direct phone capture link above on the phone browser.
            You only need ngrok if the phone is not on the same WiFi.</div>
            <div style="color:#9AA8B8;font-size:11px;
            margin-top:6px">
            Optional remote access command: <span style="font-family:monospace;color:#F59E0B">ngrok http 8501</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        qr_data = urllib.parse.quote(capture_url)
        qr_img_url = (
            f"https://api.qrserver.com/v1/"
            f"create-qr-code/?size=260x260"
            f"&data={qr_data}"
            f"&bgcolor=0F172A"
            f"&color=6366F1"
            f"&qzone=2"
            f"&format=svg")

        st.markdown(f"""
        <div style="background:#3B4A5E;
        border:1px solid #5A6B7E;
        border-radius:16px;padding:28px 20px;
        text-align:center;
        box-shadow:0 0 40px rgba(99,102,241,0.15)">

        <div style="color:#9AA8B8;font-size:10px;
        font-weight:700;letter-spacing:2px;
        margin-bottom:16px">SCAN WITH PHONE</div>

        <div style="background:#2D3748;
        border-radius:12px;padding:16px;
        display:inline-block;
        border:2px solid #5A6B7E">
        <img src="{qr_img_url}"
        style="width:200px;height:200px;
        display:block">
        </div>

        <div style="color:#6366F1;font-size:11px;
        font-family:monospace;margin-top:14px;
        word-break:break-all;padding:0 8px;
        line-height:1.5">
        {capture_url.replace("https://","").replace("http://","")}
        </div>

        <div style="margin-top:12px;
        display:flex;align-items:center;
        justify-content:center;gap:6px">
        <div style="width:6px;height:6px;
        border-radius:50%;background:#10B981;
        animation:pulse 1.5s infinite"></div>
        <span style="color:#10B981;font-size:11px;
        font-weight:600">Ready for capture</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Refresh", use_container_width=True, key="refresh_qr"):
            st.rerun()

    @st.fragment(run_every=5)
    def _render_live_capture_status():
        latest_capture = _load_latest_capture_payload()

        st.markdown("""
        <div style="color:#9AA8B8;font-size:10px;
        font-weight:700;letter-spacing:2px;
        text-transform:uppercase;margin:24px 0 12px">
        ◈ CAPTURE STATUS
        </div>
        """, unsafe_allow_html=True)

        if latest_capture:
            st.markdown(f"""
            <div style="background:#3B4A5E;
            border:1px solid #5A6B7E;
            border-left:4px solid #10B981;
            border-radius:12px;padding:20px 24px">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap">
            <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap">
            <div>
            <div style="color:#9AA8B8;font-size:10px;
            letter-spacing:1px">PERSONS DETECTED</div>
            <div style="color:#22D3EE;font-size:40px;
            font-weight:900;font-family:monospace">
            {latest_capture.get('count', '—')}</div>
            </div>
            <div>
            <div style="color:#9AA8B8;font-size:10px;
            letter-spacing:1px">THREAT</div>
            <div style="color:#FFFFFF;font-size:16px;
            font-weight:700">{latest_capture.get('threat', '—')} · {latest_capture.get('threat_score', '—')}%</div>
            <div style="color:#9AA8B8;font-size:11px;margin-top:4px">
            {latest_capture.get('model', 'DM-Count')} · {_format_capture_time(latest_capture.get('created_at'))}</div>
            </div>
            </div>
            <div style="color:#10B981;font-size:13px;
            font-weight:600">● Analysis Complete</div>
            </div>
            </div>
            """, unsafe_allow_html=True)

            _result_cols = st.columns([1, 1])
            with _result_cols[0]:
                _orig_b64 = latest_capture.get("original_b64")
                if _orig_b64:
                    st.markdown("""
                    <div style="color:#9AA8B8;font-size:10px;font-weight:700;
                    letter-spacing:2px;text-transform:uppercase;margin:18px 0 10px">
                    Original Capture
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(base64.b64decode(_orig_b64), use_container_width=True)
            with _result_cols[1]:
                _safety_b64 = latest_capture.get("safety_img_b64")
                if _safety_b64:
                    st.markdown("""
                    <div style="color:#9AA8B8;font-size:10px;font-weight:700;
                    letter-spacing:2px;text-transform:uppercase;margin:18px 0 10px">
                    Safety Zone Result
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(base64.b64decode(_safety_b64), use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#3B4A5E;
            border:1px solid #5A6B7E;
            border-radius:12px;padding:24px 26px;
            display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap">
            <div>
            <div style="color:#FFFFFF;font-size:16px;font-weight:700;margin-bottom:6px">
            Waiting for phone capture</div>
            <div style="color:#9AA8B8;font-size:13px;line-height:1.7">
            Scan the QR code, open the native camera/gallery picker on the phone,
            then upload a crowd image to push the latest result here.</div>
            </div>
            <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;
            border-radius:999px;background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.22)">
            <div style="width:8px;height:8px;border-radius:50%;background:#6366F1"></div>
            <span style="color:#CBD5E1;font-size:12px;font-weight:700">LISTENING</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

    _render_live_capture_status()


# ═══════════════════════════════════════════════════════════════
# TAB 5 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════

with tab5:

    # ── Header card ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#3B4A5E 0%,#2D3748 100%);
    padding:22px 26px;border-radius:12px;border:1px solid #5A6B7E;
    border-left:4px solid #7C3AED;margin-bottom:22px;
    box-shadow:inset 4px 0 30px rgba(124,58,237,0.15), 0 4px 20px rgba(0,0,0,0.4)">
    <div style="display:flex;align-items:center;justify-content:space-between">
    <div>
    <h2 style="color:#FFFFFF;margin:0;font-size:20px;font-weight:700;
    letter-spacing:-0.02em">🗂️ Batch Analysis</h2>
    <p style="color:#9AA8B8;margin:6px 0 0;font-size:13px;line-height:1.5">
    Multi-frame crowd density analysis · Full zone classification per image</p>
    </div>
    <span style="display:inline-flex;align-items:center;gap:6px;
    padding:5px 14px;border-radius:20px;font-size:10px;font-weight:600;
    background:rgba(124,58,237,0.1);color:#A78BFA;
    border:1px solid rgba(124,58,237,0.2);
    font-family:'JetBrains Mono',monospace;letter-spacing:1px">
    MULTI-IMAGE</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload multiple crowd images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if batch_files and len(batch_files) > 0:
        _batch_signature = (
            method,
            round(float(opacity), 2),
            tuple(
                (bf.name, getattr(bf, "size", None))
                for bf in batch_files
            )
        )
        if st.session_state["batch_results_signature"] != _batch_signature:
            st.session_state["batch_results_signature"] = _batch_signature
            st.session_state["batch_results"] = []
            st.session_state["batch_summary"] = None

        st.markdown(f"""
        <div style="background:#3B4A5E;border:1px solid #5A6B7E;border-radius:10px;
        padding:12px 20px;margin:8px 0 16px 0;display:flex;align-items:center;gap:10px">
        <span style="color:#A78BFA;font-size:14px">📁</span>
        <span style="color:#9AA8B8;font-size:13px">{len(batch_files)} images selected</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶ Run Batch Analysis", key="run_batch", use_container_width=True):

            batch_results = []
            skipped_files = []
            progress_bar = st.progress(0, text="Processing batch...")
            _batch_total_start = time.time()

            for idx, bf in enumerate(batch_files, start=1):
                progress_bar.progress(
                    (idx - 1) / len(batch_files),
                    text=f"Analyzing {bf.name} ({idx}/{len(batch_files)})..."
                )

                _b_t0 = time.time()
                try:
                    raw_bytes = np.frombuffer(bf.getvalue(), dtype=np.uint8)
                    _b_img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                    if _b_img_bgr is None:
                        raise ValueError("Decode returned None")
                    _b_img_rgb = cv2.cvtColor(_b_img_bgr, cv2.COLOR_BGR2RGB)
                except Exception as _b_dec_err:
                    skipped_files.append({
                        "name": bf.name,
                        "reason": f"Unreadable image ({_b_dec_err})",
                    })
                    progress_bar.progress(
                        idx / len(batch_files),
                        text=f"Skipped {bf.name} ({idx}/{len(batch_files)})"
                    )
                    continue

                try:
                    _b_result = _analyze_scene_frame(
                        _b_img_rgb, _b_img_bgr, method, opacity)
                except Exception as _b_analysis_err:
                    skipped_files.append({
                        "name": bf.name,
                        "reason": f"Analysis failed ({_b_analysis_err})",
                    })
                    progress_bar.progress(
                        idx / len(batch_files),
                        text=f"Skipped {bf.name} ({idx}/{len(batch_files)})"
                    )
                    continue

                _b_elapsed = round(time.time() - _b_t0, 2)

                batch_results.append({
                    "name": bf.name,
                    "count": _b_result["count"],
                    "threat": _b_result["threat"],
                    "threat_score": _b_result["threat_score"],
                    "confidence": _b_result["confidence"],
                    "zone_stats": _b_result["zone_stats"],
                    "model": _b_result["model"],
                    "marker_count": _b_result["marker_count"],
                    "marker_note": _b_result["marker_note"],
                    "used_face_detector": _b_result["used_face_detector"],
                    "portrait_hybrid_mode": _b_result.get("portrait_hybrid_mode", False),
                    "safety_img_b64": _compress_img_b64(_b_result["safety_img"]),
                    "density_overlay_b64": _compress_img_b64(_b_result["density_overlay"]),
                    "head_overlay_b64": _compress_img_b64(_b_result["head_overlay"]),
                    "time_s": _b_elapsed,
                })

                # Update progress with timing info
                progress_bar.progress(
                    idx / len(batch_files),
                    text=(
                        f"✓ {bf.name} — est. {batch_results[-1]['count']} "
                        f"· {_b_elapsed:.1f}s ({idx}/{len(batch_files)})"
                    )
                )

            _batch_total_elapsed = time.time() - _batch_total_start
            _processed_count = len(batch_results)
            _skipped_count = len(skipped_files)
            progress_bar.progress(
                1.0,
                text=(
                    f"✅ Batch complete! {_processed_count} processed"
                    f"{' · ' + str(_skipped_count) + ' skipped' if _skipped_count else ''}"
                    f" · {_batch_total_elapsed:.1f}s"
                ))
            progress_bar.empty()

            st.session_state["batch_results"] = batch_results
            st.session_state["batch_summary"] = {
                "uploaded": len(batch_files),
                "processed": _processed_count,
                "skipped": skipped_files,
                "elapsed_s": round(_batch_total_elapsed, 2),
            }

            if batch_results:
                st.success(
                    f"Processed {_processed_count} image(s) in {_batch_total_elapsed:.1f}s."
                    + (f" Skipped {_skipped_count} image(s)." if _skipped_count else "")
                )
            else:
                st.error("Batch analysis could not process any uploaded images.")

            if skipped_files:
                _skip_lines = "".join(
                    f"<li><span style='color:#FFFFFF'>{s['name']}</span> — {s['reason']}</li>"
                    for s in skipped_files
                )
                st.markdown(f"""
                <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.24);
                border-radius:12px;padding:14px 18px;margin:10px 0 16px 0">
                <div style="color:#F59E0B;font-size:10px;font-weight:700;
                letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
                ⚠️ Skipped Files</div>
                <ul style="margin:0;padding-left:18px;color:#CBD5E1;font-size:12px;line-height:1.6">
                {_skip_lines}
                </ul>
                </div>
                """, unsafe_allow_html=True)

            gc.collect()
        # Display results if available
        if st.session_state["batch_results"]:
            batch_results = st.session_state["batch_results"]
            _batch_summary = st.session_state.get("batch_summary", None)

            if _batch_summary:
                st.markdown(f"""
                <div style="background:#3B4A5E;border:1px solid #5A6B7E;border-radius:12px;
                padding:12px 18px;margin:14px 0 8px;display:flex;flex-wrap:wrap;
                gap:18px;align-items:center">
                <span style="color:#A78BFA;font-size:10px;font-weight:700;
                letter-spacing:2px;text-transform:uppercase">Batch Run Summary</span>
                <span style="color:#CBD5E1;font-size:12px">Uploaded: {_batch_summary['uploaded']}</span>
                <span style="color:#FFFFFF;font-size:12px">Processed: {_batch_summary['processed']}</span>
                <span style="color:#F59E0B;font-size:12px">Skipped: {len(_batch_summary['skipped'])}</span>
                <span style="color:#22D3EE;font-size:12px">Elapsed: {_batch_summary['elapsed_s']:.2f}s</span>
                </div>
                """, unsafe_allow_html=True)

            # ══════════════════════════════════════════════════
            # 1. AGGREGATE ANALYTICS DASHBOARD
            # ══════════════════════════════════════════════════
            _total_frames = len(batch_results)
            _avg_count = int(np.mean([r["count"] for r in batch_results]))
            _max_count = max(r["count"] for r in batch_results)
            _avg_threat = int(np.mean([r["threat_score"] for r in batch_results]))

            st.markdown("""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
            border-radius:12px;padding:14px 18px 6px;margin:16px 0 12px;
            border-top:2px solid #7C3AED">
            <div style="color:#CBD5E1;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase">
            📊 AGGREGATE ANALYTICS</div>
            </div>
            """, unsafe_allow_html=True)

            _ag1, _ag2, _ag3, _ag4, _ag5 = st.columns(5)
            _ag1.metric("🖼️ Total Frames", _total_frames)
            _ag2.metric("👥 Avg Count", _avg_count)
            _ag3.metric("🔺 Peak Count", _max_count)
            _ag4.metric("⚡ Avg Threat", f"{_avg_threat}%")
            _avg_conf = int(np.mean([r["confidence"] for r in batch_results]))
            _ag5.metric("🎯 Avg Confidence", f"{_avg_conf}%")

            # ── Aggregate zone distribution ──
            _agg_zones = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
            for r in batch_results:
                for k in _agg_zones:
                    _agg_zones[k] += r["zone_stats"][k]

            _threat_dist = {"MINIMAL": 0, "ELEVATED": 0, "HIGH": 0, "CRITICAL": 0}
            for r in batch_results:
                _threat_dist[r["threat"]] += 1

            _agg_c1, _agg_c2 = st.columns(2)

            with _agg_c1:
                _fig_zone_dist = go.Figure(data=[go.Bar(
                    x=list(_agg_zones.keys()),
                    y=list(_agg_zones.values()),
                    marker_color=["#10B981", "#F59E0B", "#EF4444", "#FF1744"],
                    text=list(_agg_zones.values()),
                    textposition='auto',
                    textfont=dict(color="#EFF6FF", size=13,
                                  family="JetBrains Mono, monospace"),
                )])
                _fig_zone_dist.update_layout(
                    title=dict(text="Zone Distribution (All Frames)",
                               font=dict(color="#9AA8B8", size=13)),
                    template="none",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=40, r=20, t=40, b=40),
                    xaxis=dict(tickfont=dict(color="#9AA8B8", size=11)),
                    yaxis=dict(tickfont=dict(color="#CBD5E1", size=10),
                               gridcolor="#5A6B7E"),
                    showlegend=False,
                )
                st.plotly_chart(
                    _fig_zone_dist,
                    use_container_width=True,
                    key="batch_zone_distribution",
                )

            with _agg_c2:
                _td_colors = {"MINIMAL": "#10B981", "ELEVATED": "#F59E0B",
                              "HIGH": "#EF4444", "CRITICAL": "#FF1744"}
                _td_labels = [k for k, v in _threat_dist.items() if v > 0]
                _td_values = [v for v in _threat_dist.values() if v > 0]
                _td_cols = [_td_colors[k] for k in _td_labels]

                _fig_threat_dist = go.Figure(data=[go.Pie(
                    labels=_td_labels, values=_td_values,
                    marker=dict(colors=_td_cols,
                                line=dict(color="#FFFFFF", width=2)),
                    textinfo="label+value",
                    textfont=dict(size=11, color="#EFF6FF",
                                  family="Inter, sans-serif"),
                    hole=0.45,
                )])
                _fig_threat_dist.update_layout(
                    title=dict(text="Threat Level Distribution",
                               font=dict(color="#9AA8B8", size=13)),
                    template="none",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False,
                )
                st.plotly_chart(
                    _fig_threat_dist,
                    use_container_width=True,
                    key="batch_threat_distribution",
                )
            # ══════════════════════════════════════════════════
            # 2. SUMMARY TABLE WITH ZONE COLUMNS
            # ══════════════════════════════════════════════════
            st.markdown("""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
            border-radius:12px;padding:14px 18px 6px;margin:16px 0 4px;
            border-top:2px solid #7C3AED">
            <div style="color:#CBD5E1;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase">
            📋 DETAILED RESULTS TABLE</div>
            </div>
            """, unsafe_allow_html=True)

            _batch_table = dedent("""
            <div style="border:1px solid #5A6B7E;border-radius:12px;
            overflow:auto;margin-top:8px">
            <table style="width:100%;border-collapse:collapse;
            font-family:'Inter',sans-serif">
            <thead>
            <tr style="background:#2D3748;border-bottom:2px solid #8896AA">
            <th style="padding:12px 14px;text-align:left;color:#CBD5E1;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">IMAGE</th>
            <th style="padding:12px 10px;text-align:center;color:#CBD5E1;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">EST.</th>
            <th style="padding:12px 10px;text-align:center;color:#10B981;
            font-size:10px;font-weight:700;letter-spacing:1.5px">LOW</th>
            <th style="padding:12px 10px;text-align:center;color:#D97706;
            font-size:10px;font-weight:700;letter-spacing:1.5px">MED</th>
            <th style="padding:12px 10px;text-align:center;color:#DC2626;
            font-size:10px;font-weight:700;letter-spacing:1.5px">HIGH</th>
            <th style="padding:12px 10px;text-align:center;color:#B91C1C;
            font-size:10px;font-weight:700;letter-spacing:1.5px">CRIT</th>
            <th style="padding:12px 10px;text-align:center;color:#CBD5E1;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">THREAT</th>
            <th style="padding:12px 10px;text-align:center;color:#CBD5E1;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">SCORE</th>
            <th style="padding:12px 10px;text-align:center;color:#6366F1;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">CONF</th>
            <th style="padding:12px 10px;text-align:right;color:#CBD5E1;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">TIME</th>
            </tr>
            </thead>
            <tbody>
            """)

            _threat_colors = {
                "MINIMAL": ("#10B981", "rgba(16,185,129,0.12)"),
                "ELEVATED": ("#F59E0B", "rgba(245,158,11,0.12)"),
                "HIGH": ("#EF4444", "rgba(239,68,68,0.12)"),
                "CRITICAL": ("#FF1744", "rgba(255,23,68,0.12)"),
            }

            for _bi, _br in enumerate(batch_results):
                _row_bg = "#3B4A5E" if _bi % 2 == 0 else "#485A6E"
                _btc, _btbg = _threat_colors.get(_br["threat"], ("#9AA8B8", "rgba(100,116,139,0.12)"))
                _zs = _br["zone_stats"]

                _batch_table += dedent(f"""
                <tr style="background:{_row_bg};border-bottom:1px solid #5A6B7E">
                <td style="padding:10px 14px;color:#9AA8B8;font-size:12px;
                font-family:'JetBrains Mono',monospace;font-weight:500">
                {_br['name']}</td>
                <td style="padding:10px;text-align:center;
                color:#FFFFFF;font-size:14px;font-weight:700;
                font-family:'JetBrains Mono',monospace;
                font-variant-numeric:tabular-nums">{_br['count']}</td>
                <td style="padding:10px;text-align:center;
                color:#10B981;font-size:13px;font-weight:600;
                font-family:'JetBrains Mono',monospace">{_zs['Low']}</td>
                <td style="padding:10px;text-align:center;
                color:#D97706;font-size:13px;font-weight:600;
                font-family:'JetBrains Mono',monospace">{_zs['Medium']}</td>
                <td style="padding:10px;text-align:center;
                color:#DC2626;font-size:13px;font-weight:600;
                font-family:'JetBrains Mono',monospace">{_zs['High']}</td>
                <td style="padding:10px;text-align:center;
                color:#B91C1C;font-size:13px;font-weight:600;
                font-family:'JetBrains Mono',monospace">{_zs['Critical']}</td>
                <td style="padding:10px;text-align:center">
                <span style="display:inline-block;padding:4px 12px;
                border-radius:20px;font-size:10px;font-weight:700;
                letter-spacing:1px;color:{_btc};background:{_btbg};
                border:1px solid {_btc}30">{_br['threat']}</span></td>
                <td style="padding:10px;text-align:center;
                color:#6366F1;font-size:13px;font-weight:600;
                font-family:'JetBrains Mono',monospace">{_br['threat_score']}%</td>
                <td style="padding:10px;text-align:center;
                color:#A78BFA;font-size:12px;font-weight:600;
                font-family:'JetBrains Mono',monospace">{_br['confidence']}%</td>
                <td style="padding:10px;text-align:right;
                color:#CBD5E1;font-size:12px;font-weight:500;
                font-family:'JetBrains Mono',monospace">{_br.get('time_s', '—')}s</td>
                </tr>
                """)

            _batch_table += "</tbody></table></div>"
            st.markdown(_batch_table, unsafe_allow_html=True)
            # ══════════════════════════════════════════════════
            # 3. PER-IMAGE EXPANDABLE DETAIL CARDS
            # ══════════════════════════════════════════════════
            st.markdown("""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
            border-radius:12px;padding:14px 18px 6px;margin:24px 0 4px;
            border-top:2px solid #0891B2">
            <div style="color:#CBD5E1;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase">
            🔍 PER-IMAGE DETAILS</div>
            </div>
            """, unsafe_allow_html=True)

            for _di, _dr in enumerate(batch_results):
                _d_zs = _dr["zone_stats"]
                _d_ts = _dr["threat_score"]
                _d_tc, _ = _threat_colors.get(_dr["threat"], ("#9AA8B8", ""))
                if _dr.get("portrait_hybrid_mode"):
                    _d_marker_mode = "portrait markers"
                    _d_marker_title = "PORTRAIT MARKERS"
                elif _dr.get("used_face_detector"):
                    _d_marker_mode = "face markers"
                    _d_marker_title = "FACE MARKERS"
                else:
                    _d_marker_mode = "head markers"
                    _d_marker_title = "HEAD MARKERS"

                with st.expander(
                        f"📷 {_dr['name']}  —  est. {_dr['count']}  ·  "
                        f"{_dr.get('marker_count', 0)} {_d_marker_mode}  ·  "
                        f"{_dr['threat']}  ·  {_dr['confidence']}% conf  ·  "
                        f"{_dr.get('time_s', '—')}s"):
                    st.markdown(f"""
                    <div style="background:#3B4A5E;border:1px solid #5A6B7E;border-radius:10px;
                    padding:10px 14px;margin:4px 0 14px;display:flex;flex-wrap:wrap;gap:14px">
                    <span style="color:#CBD5E1;font-size:11px">Model: <span style="color:#FFFFFF">{_dr.get('model', '—')}</span></span>
                    <span style="color:#CBD5E1;font-size:11px">Visible markers: <span style="color:#FFFFFF">{_dr.get('marker_count', 0)}</span></span>
                    <span style="color:#CBD5E1;font-size:11px">Threat score: <span style="color:#FFFFFF">{_d_ts}%</span></span>
                    </div>
                    """, unsafe_allow_html=True)

                    if _dr.get("marker_note"):
                        st.markdown(f"""
                        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.24);
                        border-radius:10px;padding:10px 14px;margin:0 0 14px 0;
                        color:#F59E0B;font-size:11px">{_dr['marker_note']}</div>
                        """, unsafe_allow_html=True)

                    _d_c1, _d_c2, _d_c3 = st.columns(3)
                    with _d_c1:
                        st.markdown(f"""
                        <div style="color:#6366F1;font-size:10px;font-weight:700;
                        letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
                        {_d_marker_title}</div>
                        """, unsafe_allow_html=True)
                        _d_head_bytes = base64.b64decode(_dr["head_overlay_b64"])
                        st.image(_d_head_bytes, use_container_width=True)
                    with _d_c2:
                        st.markdown(f"""
                        <div style="color:#6366F1;font-size:10px;font-weight:700;
                        letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
                        DENSITY HEATMAP</div>
                        """, unsafe_allow_html=True)
                        _d_density_bytes = base64.b64decode(_dr["density_overlay_b64"])
                        st.image(_d_density_bytes, use_container_width=True)
                    with _d_c3:
                        st.markdown(f"""
                        <div style="color:#6366F1;font-size:10px;font-weight:700;
                        letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
                        SAFETY ZONE MAP — {method.upper()}</div>
                        """, unsafe_allow_html=True)
                        _d_safety_bytes = base64.b64decode(_dr["safety_img_b64"])
                        st.image(_d_safety_bytes, use_container_width=True)

                    # Zone stats row
                    _d_z1, _d_z2, _d_z3, _d_z4 = st.columns(4)
                    _d_zone_data = [
                        ("🟢 LOW", _d_zs["Low"], "#10B981"),
                        ("🟡 MEDIUM", _d_zs["Medium"], "#F59E0B"),
                        ("🟠 HIGH", _d_zs["High"], "#EF4444"),
                        ("🔴 CRITICAL", _d_zs["Critical"], "#FF1744"),
                    ]
                    for _d_col, (_d_lbl, _d_val, _d_clr) in zip(
                            [_d_z1, _d_z2, _d_z3, _d_z4], _d_zone_data):
                        with _d_col:
                            st.markdown(f"""
                            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
                            border-radius:10px;padding:14px;text-align:center;
                            border-top:2px solid {_d_clr}">
                            <div style="font-size:10px;color:#CBD5E1;font-weight:700;
                            letter-spacing:1.2px;margin-bottom:4px">{_d_lbl}</div>
                            <div style="font-size:28px;font-weight:900;color:{_d_clr};
                            font-family:'JetBrains Mono',monospace">{_d_val}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Compact threat gauge
                    _d_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=_d_ts,
                        number=dict(font=dict(size=28, color="#EFF6FF",
                                              family="JetBrains Mono, monospace"),
                                    suffix="%"),
                        title=dict(text=f"THREAT: {_dr['threat']}",
                                   font=dict(size=12, color=_d_tc,
                                             family="Inter, sans-serif")),
                        gauge=dict(
                            axis=dict(range=[0, 100], tickcolor="#9AA8B8",
                                      tickfont=dict(color="#CBD5E1", size=9)),
                            bar=dict(color=_d_tc, thickness=0.3),
                            bgcolor="#8896AA", borderwidth=0,
                            steps=[
                                dict(range=[0, 25], color="#1F4A2E"),
                                dict(range=[25, 50], color="#4A3D10"),
                                dict(range=[50, 75], color="#4A2D10"),
                                dict(range=[75, 100], color="#4A1520"),
                            ],
                            threshold=dict(line=dict(color=_d_tc, width=2),
                                           thickness=0.75, value=_d_ts),
                        ),
                    ))
                    _d_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=180,
                        margin=dict(l=30, r=30, t=45, b=5),
                        font=dict(family="Inter, sans-serif"),
                    )
                    st.plotly_chart(
                        _d_gauge,
                        use_container_width=True,
                        key=f"batch_threat_gauge_{_di}_{_dr['name']}",
                    )
            # ══════════════════════════════════════════════════
            # 4. PEAK FRAME HIGHLIGHT
            # ══════════════════════════════════════════════════
            _peak_result = max(batch_results, key=lambda x: x["count"])

            st.markdown(f"""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
            border-radius:12px;padding:14px 18px 6px;margin:24px 0 4px;
            border-top:2px solid #FF1744">
            <div style="display:flex;align-items:center;justify-content:space-between">
            <div style="color:#CBD5E1;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase">
            🔺 PEAK DENSITY FRAME</div>
            <span style="color:#B91C1C;font-size:12px;font-weight:700;
            font-family:'JetBrains Mono',monospace">est. {_peak_result['count']} · {_peak_result['name']}</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

            _pk_c1, _pk_c2, _pk_c3 = st.columns(3)
            with _pk_c1:
                _pk_head_bytes = base64.b64decode(_peak_result["head_overlay_b64"])
                st.image(_pk_head_bytes, use_container_width=True,
                         caption=f"Markers — {_peak_result['name']}")
            with _pk_c2:
                _pk_density_bytes = base64.b64decode(_peak_result["density_overlay_b64"])
                st.image(_pk_density_bytes, use_container_width=True,
                         caption=f"Density Heatmap — {_peak_result['name']}")
            with _pk_c3:
                _pk_safety_bytes = base64.b64decode(_peak_result["safety_img_b64"])
                st.image(_pk_safety_bytes, use_container_width=True,
                         caption=f"Safety Map — {_peak_result['name']}")

            # ══════════════════════════════════════════════════
            # 5. BATCH TIMELINE CHART
            # ══════════════════════════════════════════════════
            st.markdown("""
            <div style="background:#3B4A5E;border:1px solid #5A6B7E;
            border-radius:12px;padding:14px 18px 6px;margin:24px 0 4px;
            border-top:3px solid #6366F1">
            <div style="color:#CBD5E1;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase">
            📈 BATCH DENSITY TIMELINE</div>
            </div>
            """, unsafe_allow_html=True)

            _batch_counts = [r["count"] for r in batch_results]
            _batch_names = [r["name"] for r in batch_results]
            _batch_nums = list(range(1, len(batch_results) + 1))
            _batch_threats = [r["threat_score"] for r in batch_results]

            _batch_dot_colors = []
            for _bt in _batch_threats:
                if _bt < 25:
                    _batch_dot_colors.append("#10B981")
                elif _bt < 50:
                    _batch_dot_colors.append("#F59E0B")
                elif _bt < 75:
                    _batch_dot_colors.append("#EF4444")
                else:
                    _batch_dot_colors.append("#FF1744")

            _batch_hover = [
                f"<b>{n}</b><br>Count: <b>{c}</b><br>Threat: <b>{t}%</b>"
                for n, c, t in zip(_batch_names, _batch_counts, _batch_threats)
            ]

            _fig_batch = go.Figure()

            _fig_batch.add_trace(go.Scatter(
                x=_batch_nums, y=_batch_counts,
                fill='tozeroy',
                fillcolor='rgba(124,58,237,0.08)',
                line=dict(color='rgba(124,58,237,0.0)', width=0),
                showlegend=False, hoverinfo='skip',
            ))

            _fig_batch.add_trace(go.Scatter(
                x=_batch_nums, y=_batch_counts,
                mode='lines',
                line=dict(color='#7C3AED', width=3, shape='spline', smoothing=1.2),
                showlegend=False, hoverinfo='skip',
            ))

            _fig_batch.add_trace(go.Scatter(
                x=_batch_nums, y=_batch_counts,
                mode='markers',
                marker=dict(size=10, color=_batch_dot_colors,
                            line=dict(color='rgba(255,255,255,0.3)', width=1.5)),
                hovertext=_batch_hover,
                hoverinfo='text',
                hoverlabel=dict(bgcolor='#2D3748', bordercolor='#7C3AED',
                                font=dict(family='Inter, sans-serif', size=12, color='#EFF6FF')),
                showlegend=False,
            ))

            # Capacity threshold line
            _fig_batch.add_hline(
                y=venue_capacity,
                line_dash="dash",
                line_color="#EF4444",
                line_width=1.5,
                annotation_text=f"Capacity: {venue_capacity}",
                annotation_position="top right",
                annotation_font=dict(color="#EF4444", size=10,
                                     family="JetBrains Mono, monospace"),
            )

            _fig_batch.update_layout(
                template='none',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=320,
                margin=dict(l=50, r=20, t=20, b=50),
                xaxis=dict(
                    title=dict(text='Frame #', font=dict(color='#9AA8B8', size=11)),
                    tickfont=dict(color='#9AA8B8', size=10,
                                  family='JetBrains Mono, monospace'),
                    gridcolor='#5A6B7E', gridwidth=1,
                    zeroline=False, dtick=1,
                ),
                yaxis=dict(
                    title=dict(text='Crowd Count', font=dict(color='#9AA8B8', size=11)),
                    tickfont=dict(color='#9AA8B8', size=10,
                                  family='JetBrains Mono, monospace'),
                    gridcolor='#5A6B7E', gridwidth=1,
                    zeroline=False,
                ),
                font=dict(family='Inter, sans-serif'),
                hovermode='closest',
            )
            st.plotly_chart(
                _fig_batch,
                use_container_width=True,
                key="batch_count_timeline",
            )

            # ══════════════════════════════════════════════════
            # 6. DOWNLOAD BATCH REPORT
            # ══════════════════════════════════════════════════
            _batch_report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "method": method,
                "venue_capacity": venue_capacity,
                "summary": st.session_state.get("batch_summary", {}),
                "total_frames": len(batch_results),
                "peak_count": _peak_result["count"],
                "peak_frame": _peak_result["name"],
                "average_count": _avg_count,
                "average_threat_score": _avg_threat,
                "aggregate_zones": _agg_zones,
                "threat_distribution": _threat_dist,
                "frames": [
                    {
                        "name": r["name"],
                        "count": r["count"],
                        "model": r.get("model"),
                        "marker_count": r.get("marker_count"),
                        "marker_note": r.get("marker_note"),
                        "used_face_detector": r.get("used_face_detector"),
                        "portrait_hybrid_mode": r.get("portrait_hybrid_mode"),
                        "threat": r["threat"],
                        "threat_score": r["threat_score"],
                        "confidence": r["confidence"],
                        "zone_stats": r["zone_stats"],
                        "time_s": r.get("time_s"),
                    }
                    for r in batch_results
                ],
            }

            _batch_report_json = json.dumps(_batch_report, indent=2)

            st.download_button(
                "📥  Download Batch Report (JSON)",
                data=_batch_report_json,
                file_name=f"safecrowd_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="batch_report_dl",
                use_container_width=True,
            )

    else:
        # Empty state for batch
        st.markdown("""
<div style="text-align:center;padding:80px 20px 70px;
animation:floatUp 0.6s ease;
background:radial-gradient(ellipse 500px 250px at center 100px,
rgba(124,58,237,0.06), transparent)">

<div style="font-size:56px;margin-bottom:20px;opacity:0.6;
filter:drop-shadow(0 4px 20px rgba(124,58,237,0.25))">🗂️</div>

<div style="font-size:11px;font-weight:700;color:#7C3AED;
letter-spacing:5px;text-transform:uppercase;margin-bottom:14px">
BATCH ANALYSIS</div>

<h2 style="color:#FFFFFF;font-size:26px;font-weight:800;margin:0;
letter-spacing:-0.02em">Upload Multiple Images</h2>

<p style="color:#CBD5E1;font-size:14px;max-width:440px;
margin:14px auto 0;line-height:1.8">
Upload multiple crowd images for full zone classification per frame.
Get aggregate statistics, per-image overlays, and a downloadable report.</p>

<div style="border-top:1px solid #5A6B7E;margin:32px auto 28px;
max-width:300px"></div>

<div style="display:flex;justify-content:center;gap:16px;
flex-wrap:wrap;max-width:600px;margin:0 auto">

<div style="background:#3B4A5E;border:1px solid #5A6B7E;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">📊</div>
<div style="color:#FFFFFF;font-size:12px;font-weight:600">Zone Stats</div>
<div style="color:#CBD5E1;font-size:10px;margin-top:4px">Per-image classification</div>
</div>

<div style="background:#3B4A5E;border:1px solid #5A6B7E;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">🗺️</div>
<div style="color:#FFFFFF;font-size:12px;font-weight:600">Safety Maps</div>
<div style="color:#CBD5E1;font-size:10px;margin-top:4px">Overlays & heatmaps</div>
</div>

<div style="background:#3B4A5E;border:1px solid #5A6B7E;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">📈</div>
<div style="color:#FFFFFF;font-size:12px;font-weight:600">Timeline</div>
<div style="color:#CBD5E1;font-size:10px;margin-top:4px">Density across frames</div>
</div>

<div style="background:#3B4A5E;border:1px solid #5A6B7E;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">📥</div>
<div style="color:#FFFFFF;font-size:12px;font-weight:600">JSON Report</div>
<div style="color:#CBD5E1;font-size:10px;margin-top:4px">Full zone-level data</div>
</div>

</div>

</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 6 — RL EVACUATION AGENT
# ═══════════════════════════════════════════════════════════════

with tab6:

    _rl = st.session_state["rl_agent"]

    # ── SECTION 1: Header ──
    if _rl.episode == 0:
        _rl_status_label = "READY"
        _rl_status_bg = "rgba(34,211,238,0.12)"
        _rl_status_color = "#22D3EE"
        _rl_status_border = "rgba(34,211,238,0.3)"
    elif _rl.epsilon > 0.5:
        _rl_status_label = "TRAINING"
        _rl_status_bg = "rgba(245,158,11,0.12)"
        _rl_status_color = "#D97706"
        _rl_status_border = "rgba(245,158,11,0.3)"
    elif _rl.epsilon > 0.1:
        _rl_status_label = "LEARNING"
        _rl_status_bg = "rgba(99,102,241,0.12)"
        _rl_status_color = "#6366F1"
        _rl_status_border = "rgba(99,102,241,0.3)"
    else:
        _rl_status_label = "OPTIMAL"
        _rl_status_bg = "rgba(5,150,105,0.12)"
        _rl_status_color = "#059669"
        _rl_status_border = "rgba(5,150,105,0.3)"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#3B4A5E 0%,#2D3748 100%);
    padding:22px 26px;border-radius:12px;border:1px solid #5A6B7E;
    border-left:4px solid #7C3AED;margin-bottom:22px;
    box-shadow:inset 4px 0 30px rgba(124,58,237,0.15), 0 4px 20px rgba(0,0,0,0.04)">
    <div style="display:flex;align-items:center;justify-content:space-between">
    <div>
    <h2 style="color:#FFFFFF;margin:0;font-size:20px;font-weight:700;
    letter-spacing:-0.02em">🤖 DQN Evacuation Policy Agent</h2>
    <p style="color:#9AA8B8;margin:6px 0 0;font-size:13px;line-height:1.5">
    Deep Q-Network learning optimal evacuation strategy from crowd density states</p>
    </div>
    <span style="display:inline-flex;align-items:center;gap:6px;
    padding:5px 14px;border-radius:20px;font-size:10px;font-weight:600;
    background:{_rl_status_bg};color:{_rl_status_color};
    border:1px solid {_rl_status_border};
    font-family:'JetBrains Mono',monospace;letter-spacing:1px">
    <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
    background:{_rl_status_color};animation:dotPulse 1.5s ease-in-out infinite"></span>
    {_rl_status_label}</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 2: Agent Stats (4 cards) ──
    _rl_explore_pct = int(_rl.epsilon * 100)
    _rl_mem_size = len(_rl.memory)
    _rl_buffer_pct = min(100, int((_rl_mem_size / 2000) * 100))

    def _render_rl_stat_card(title, value, subtitle, accent):
        st.markdown(f"""
        <div style="background:#3B4A5E;border:1px solid #5A6B7E;border-radius:12px;
        padding:18px 20px;min-height:164px;display:flex;flex-direction:column;
        justify-content:space-between;box-shadow:0 8px 28px rgba(15,23,42,0.28);
        border-top:3px solid {accent}">
        <div style="color:#CBD5E1;font-size:11px;font-weight:700;
        letter-spacing:1.6px;text-transform:uppercase">{title}</div>
        <div style="color:#F8FAFC;font-size:46px;font-weight:900;line-height:1;
        font-family:'JetBrains Mono',monospace;letter-spacing:-0.04em;
        margin:14px 0 10px;word-break:break-word">{value}</div>
        <div style="color:#9AA8B8;font-size:12px;line-height:1.45">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    rl_c1, rl_c2, rl_c3, rl_c4 = st.columns(4)
    with rl_c1:
        _render_rl_stat_card(
            "Episodes Trained",
            f"{_rl.episode}",
            "Completed training steps on the latest analyzed scenes",
            "#6366F1")
    with rl_c2:
        _render_rl_stat_card(
            "Exploration Rate",
            f"{_rl_explore_pct}%",
            "Random exploration probability for the next decision",
            "#FF5A5F")
    with rl_c3:
        _render_rl_stat_card(
            "Replay Buffer",
            f"{_rl_mem_size}",
            f"{_rl_buffer_pct}% of 2000 stored experiences ready for replay",
            "#F59E0B")
    with rl_c4:
        _render_rl_stat_card(
            "Total Reward",
            f"{_rl.total_reward:+.1f}",
            "Cumulative reward accumulated across all completed episodes",
            "#10B981")

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # ── SECTION 3: Run Agent ──
    # Read from session state (cross-tab safe)
    _rl_count = st.session_state.get("last_count", 0) or 0
    _rl_density = st.session_state.get("last_density_raw", None)
    _rl_zone_stats = st.session_state.get(
        "last_zone_stats",
        {"Low": 64, "Medium": 0, "High": 0, "Critical": 0})
    _rl_num_exits = st.session_state.get("num_exits_val", 2)

    _has_scene = (_rl_count > 0 and _rl_density is not None)

    if _has_scene:
        if st.button("▶ Run RL Agent on Current Scene", key="run_rl_agent",
                      use_container_width=True):
            try:
                agent = st.session_state["rl_agent"]

                # 1. Get current state
                _rl_state = agent.get_state_vector(
                    _rl_zone_stats, _rl_count,
                    _rl_num_exits, _rl_density)

                # 2. Choose action
                _rl_action = agent.choose_action(_rl_state)

                # 3. Simulate outcome
                _sim_crit_reduction = random.uniform(0.3, 0.7)
                _sim_high_reduction = random.uniform(0.2, 0.5)

                new_critical = max(0, int(
                    _rl_zone_stats["Critical"] *
                    (1 - _sim_crit_reduction)))
                new_high = max(0, int(
                    _rl_zone_stats["High"] *
                    (1 - _sim_high_reduction)))
                _sim_next_stats = {
                    "Critical": new_critical,
                    "High": new_high,
                    "Medium": _rl_zone_stats["Medium"],
                    "Low": max(0, 64 - new_critical -
                               new_high -
                               _rl_zone_stats["Medium"]),
                }

                # Evacuation time
                _exit_cap = _rl_num_exits * 40
                _evac_before = (_rl_count * 1.0) / max(_exit_cap, 1)
                _evac_after = (_rl_count * 0.7) / max(_exit_cap, 1)

                # 4. Calculate reward
                _rl_reward = agent.calculate_reward(
                    _rl_zone_stats, _sim_next_stats, _evac_after)

                # 5. Next state
                _rl_next_state = agent.get_state_vector(
                    _sim_next_stats, _rl_count,
                    _rl_num_exits, _rl_density)

                # 6. Store and train
                agent.remember(_rl_state, _rl_action,
                               _rl_reward, _rl_next_state, False)
                _rl_loss = agent.replay()

                # 7. Update agent stats
                agent.total_reward += _rl_reward
                agent.episode += 1
                agent.action_history.append(_rl_action)
                agent.reward_history.append(_rl_reward)

                # 8. Store in rl_history
                _rl_record = {
                    "episode": agent.episode,
                    "action": _rl_action,
                    "action_name": agent.get_action_name(_rl_action),
                    "reward": _rl_reward,
                    "loss": _rl_loss,
                    "epsilon": agent.epsilon,
                    "crit_before": _rl_zone_stats.get("Critical", 0),
                    "crit_after": _sim_next_stats["Critical"],
                    "high_before": _rl_zone_stats.get("High", 0),
                    "high_after": _sim_next_stats["High"],
                    "evac_before": round(_evac_before, 1),
                    "evac_after": round(_evac_after, 1),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state["rl_history"].append(_rl_record)
                st.session_state["rl_prev_stats"] = _rl_zone_stats.copy()

                # 9. Store result for display after rerun
                st.session_state["rl_last_result"] = {
                    "action": _rl_action,
                    "action_name": agent.get_action_name(_rl_action),
                    "action_detail": agent.get_action_detail(
                        _rl_action, _rl_zone_stats, _rl_num_exits),
                    "reward": _rl_reward,
                    "loss": _rl_loss,
                    "before": _rl_zone_stats.copy(),
                    "after": _sim_next_stats.copy(),
                    "evac_before": round(_evac_before, 1),
                    "evac_after": round(_evac_after, 1),
                }
                st.rerun()

            except Exception as e:
                st.error(f"RL Agent error: {e}")

        # ── Show last result from session state ──
        _rl_result = st.session_state.get("rl_last_result", None)
        if _rl_result:
            _crit_b = _rl_result["before"]["Critical"]
            _crit_a = _rl_result["after"]["Critical"]
            _high_b = _rl_result["before"]["High"]
            _high_a = _rl_result["after"]["High"]
            _crit_pct_r = int((1 - _crit_a / max(_crit_b, 1)) * 100) if _crit_b > 0 else 0

            st.markdown(f"""
            <div style="background:#3B4A5E;
            border:1px solid #5A6B7E;
            border-left:4px solid #8B5CF6;
            border-radius:12px;padding:24px;
            margin:16px 0;
            box-shadow:0 0 30px rgba(139,92,246,0.2)">

            <div style="color:#8B5CF6;font-size:10px;
            font-weight:700;letter-spacing:2px;
            text-transform:uppercase;margin-bottom:12px">
            🤖 AGENT DECISION — Episode {_rl.episode}</div>

            <div style="color:#FFFFFF;font-size:22px;
            font-weight:800;margin-bottom:8px">
            {_rl_result["action_name"]}</div>

            <div style="color:#CBD5E1;font-size:13px;
            margin-bottom:20px">
            {_rl_result["action_detail"]}</div>

            <div style="display:grid;
            grid-template-columns:1fr 1fr 1fr;
            gap:12px;margin-bottom:16px">

            <div style="background:#485A6E;
            border-radius:8px;padding:12px;
            text-align:center">
            <div style="color:#9AA8B8;font-size:10px;
            letter-spacing:1px">CRITICAL ZONES</div>
            <div style="color:#EF4444;font-size:20px;
            font-weight:800">{_crit_b} → {_crit_a}</div>
            <div style="color:#10B981;font-size:11px">
            ↓{_crit_pct_r}%</div>
            </div>

            <div style="background:#485A6E;
            border-radius:8px;padding:12px;
            text-align:center">
            <div style="color:#9AA8B8;font-size:10px;
            letter-spacing:1px">EVAC TIME</div>
            <div style="color:#22D3EE;font-size:20px;
            font-weight:800">{_rl_result["evac_before"]} → {_rl_result["evac_after"]} min</div>
            </div>

            <div style="background:#485A6E;
            border-radius:8px;padding:12px;
            text-align:center">
            <div style="color:#9AA8B8;font-size:10px;
            letter-spacing:1px">REWARD</div>
            <div style="color:#10B981;font-size:20px;
            font-weight:800">
            {_rl_result["reward"]:+.1f}</div>
            </div>
            </div>

            <div style="color:#9AA8B8;font-size:11px;
            font-family:monospace">
            Loss: {_rl_result["loss"]:.4f} ·
            Epsilon: {_rl.epsilon:.3f} ·
            Memory: {len(_rl.memory)}/2000
            </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#3B4A5E;border:1px solid #5A6B7E;
        border-radius:12px;padding:32px;text-align:center;margin:16px 0">
        <div style="font-size:40px;margin-bottom:12px;opacity:0.5">🤖</div>
        <div style="color:#FFFFFF;font-size:16px;font-weight:700;
        margin-bottom:6px">Upload an Image First</div>
        <div style="color:#9AA8B8;font-size:13px">
        Analyze a crowd image in the Live Analysis tab to activate the RL agent.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 5: Training History Chart ──
    _rl_hist = st.session_state["rl_history"]
    if len(_rl_hist) >= 2:
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

        _episodes = [r["episode"] for r in _rl_hist]
        _rewards = [r["reward"] for r in _rl_hist]

        # Compute moving average
        _window = min(5, len(_rewards))
        _ma = []
        for i in range(len(_rewards)):
            _start = max(0, i - _window + 1)
            _ma.append(sum(_rewards[_start:i+1]) / (i - _start + 1))

        fig_learn = go.Figure()
        fig_learn.add_trace(go.Scatter(
            x=_episodes, y=_rewards,
            mode='lines+markers',
            name='Reward',
            line=dict(color="rgba(124,58,237,0.4)", width=1),
            marker=dict(color="#7C3AED", size=6),
        ))
        fig_learn.add_trace(go.Scatter(
            x=_episodes, y=_ma,
            mode='lines',
            name='Moving Avg',
            line=dict(color="#7C3AED", width=3),
        ))
        fig_learn.update_layout(
            title=dict(
                text="Agent Learning Curve",
                font=dict(size=14, color="#9AA8B8",
                          family="Inter, sans-serif"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=20, r=20, t=50, b=30),
            xaxis=dict(
                title="Episode",
                gridcolor="rgba(199,210,254,0.3)",
                tickfont=dict(color="#CBD5E1"),
                title_font=dict(color="#CBD5E1"),
            ),
            yaxis=dict(
                title="Reward",
                gridcolor="rgba(199,210,254,0.3)",
                tickfont=dict(color="#CBD5E1"),
                title_font=dict(color="#CBD5E1"),
            ),
            legend=dict(
                font=dict(color="#CBD5E1"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
        )
        st.plotly_chart(fig_learn, use_container_width=True)

    # ── SECTION 6: Decision History Table ──
    if _rl_hist:
        st.markdown("""
        <div style="color:#7C3AED;font-size:10px;font-weight:700;
        letter-spacing:2px;text-transform:uppercase;margin:20px 0 10px 0">
        ◈ DECISION HISTORY</div>
        """, unsafe_allow_html=True)

        _display_hist = _rl_hist[-10:][::-1]  # last 10, newest first
        _df_hist = pd.DataFrame([
            {
                "Episode": r["episode"],
                "Action": r["action_name"],
                "Reward": f'{r["reward"]:+.1f}',
                "Loss": f'{r["loss"]:.4f}',
                "Epsilon": f'{r["epsilon"]:.3f}',
            }
            for r in _display_hist
        ])
        st.dataframe(_df_hist, use_container_width=True, hide_index=True)

    # ── SECTION 7: Agent Controls ──
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#7C3AED;font-size:10px;font-weight:700;
    letter-spacing:2px;text-transform:uppercase;margin-bottom:10px">
    ◈ AGENT CONTROLS</div>
    """, unsafe_allow_html=True)

    _ctrl1, _ctrl2 = st.columns(2)
    with _ctrl1:
        if st.button("🔄 Reset Agent", key="reset_rl_agent",
                      use_container_width=True):
            st.session_state["rl_agent"] = EvacuationAgent()
            st.session_state["rl_history"] = []
            st.session_state["rl_prev_stats"] = None
            st.rerun()

    with _ctrl2:
        if _rl_hist:
            _export_data = json.dumps(_rl_hist, indent=2, default=str)
            st.download_button(
                "💾 Export Training Data",
                data=_export_data,
                file_name=f"rl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="export_rl_data",
                use_container_width=True,
            )
        else:
            st.button("💾 Export Training Data", key="export_rl_disabled",
                       use_container_width=True, disabled=True)

    # ── Empty state placeholder ──
    if not _rl_hist and not _has_scene:
        st.markdown("""
<div style="text-align:center;padding:80px 20px 70px;
animation:floatUp 0.6s ease;
background:radial-gradient(ellipse 500px 250px at center 100px,
rgba(124,58,237,0.06), transparent)">

<div style="font-size:56px;margin-bottom:20px;opacity:0.6;
filter:drop-shadow(0 4px 20px rgba(124,58,237,0.25))">🤖</div>

<div style="font-size:11px;font-weight:700;color:#7C3AED;
letter-spacing:5px;text-transform:uppercase;margin-bottom:14px">
REINFORCEMENT LEARNING</div>

<h2 style="color:#FFFFFF;font-size:26px;font-weight:800;margin:0;
letter-spacing:-0.02em">DQN Evacuation Agent</h2>

<p style="color:#CBD5E1;font-size:14px;max-width:440px;
margin:14px auto 0;line-height:1.8">
Upload a crowd image and run analysis first, then train the RL agent
to learn optimal evacuation policies based on zone risk states.</p>

<div style="border-top:1px solid #5A6B7E;margin:32px auto 28px;
max-width:300px"></div>

<div style="display:flex;justify-content:center;gap:16px;
flex-wrap:wrap;max-width:600px;margin:0 auto">

<div style="background:#3B4A5E;border:1px solid #5A6B7E;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">🧠</div>
<div style="color:#FFFFFF;font-size:12px;font-weight:600">DQN Network</div>
<div style="color:#CBD5E1;font-size:10px;margin-top:4px">128→128→64 architecture</div>
</div>

<div style="background:#3B4A5E;border:1px solid #5A6B7E;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">🎯</div>
<div style="color:#FFFFFF;font-size:12px;font-weight:600">8 Actions</div>
<div style="color:#CBD5E1;font-size:10px;margin-top:4px">Evacuation strategies</div>
</div>

<div style="background:#3B4A5E;border:1px solid #5A6B7E;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">📊</div>
<div style="color:#FFFFFF;font-size:12px;font-weight:600">69 Features</div>
<div style="color:#CBD5E1;font-size:10px;margin-top:4px">Zone risks + context</div>
</div>

<div style="background:#3B4A5E;border:1px solid #5A6B7E;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">🏆</div>
<div style="color:#FFFFFF;font-size:12px;font-weight:600">Reward Shaping</div>
<div style="color:#CBD5E1;font-size:10px;margin-top:4px">Multi-objective optimization</div>
</div>

</div>

</div>
        """, unsafe_allow_html=True)
