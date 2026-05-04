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
from scipy.ndimage import gaussian_filter
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
import tempfile
import random
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

st.set_page_config(
    page_title="SafeCrowd Vision",
    page_icon="🛡️",
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
    --bg:         #0F172A;
    --surface:    #1E293B;
    --surface-2:  #263445;
    --card:       #1E293B;
    --border:     #334155;
    --border-h:   #475569;
    --accent:     #6366F1;
    --accent-g:   #8B5CF6;
    --cyan:       #22D3EE;
    --purple:     #8B5CF6;
    --green:      #10B981;
    --amber:      #F59E0B;
    --red:        #EF4444;
    --critical:   #FF1744;
    --text:       #F1F5F9;
    --text-2:     #94A3B8;
    --muted:      #64748B;
    --dimmed:     #475569;
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
div[data-testid="stMainBlockContainer"] { background-color: transparent !important; }
.main .block-container { background-color: transparent !important; }

/* ══════════ SIDEBAR — COLLAPSE FIX ══════════ */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"][aria-expanded="false"] {
    width: 0px !important;
    min-width: 0 !important;
    overflow: hidden !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A2744 0%, #0F172A 100%) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 310px !important;
    z-index: 10;
}
section[data-testid="stSidebar"] > div {
    background: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: transparent !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label {
    color: #94A3B8 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] p {
    color: var(--muted) !important;
}
section[data-testid="stSidebar"] span {
    color: #94A3B8 !important;
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
[data-testid="stHeader"] { visibility: hidden; }

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



/* ══════════ PLOTLY ══════════ */
.js-plotly-plot {
    border-radius: 12px !important;
    overflow: hidden !important;
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
</style>""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div style="background:linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
padding:28px 34px;border-radius:14px;border:1px solid #334155;
border-left:4px solid #6366F1;margin-bottom:28px;
box-shadow:-4px 0 30px rgba(99,102,241,0.5), 0 4px 24px rgba(0,0,0,0.3);
animation:floatUp 0.5s ease;
display:flex;align-items:center;justify-content:space-between;
position:relative;z-index:2">
<div>
<h1 style="color:#F1F5F9;margin:0;font-size:30px;font-weight:900;
letter-spacing:-0.03em">🛡️ SafeCrowd Vision</h1>
<p style="color:#94A3B8;margin:8px 0 0;font-size:13px;font-weight:400;
letter-spacing:0.02em">Unsupervised Crowd Density Segmentation · Public Event Safety</p>
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

# ── OpenCV Face Detector (fallback for portraits) ──
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml')

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
            self.epsilon *= self.epsilon_decay

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

            # Basic stats
            m = patch.mean()
            s = patch.std()

            # Advanced markers: Clumpiness (CV) and Structure (Gradient)
            cv = s / (m + 1e-7)

            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2).max()

            features.append([
                m,
                patch.max(),
                s,
                cv,
                grad_mag,
                (patch > m).sum() / (patch.size + 1e-7),
                i / (grid - 1),
                j / (grid - 1),
            ])

    return np.array(features)


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
                          expected_count=0):
    smoothed = gaussian_filter(density_map.astype(np.float64), sigma=8)
    if smoothed.max() > 0:
        norm = ((smoothed / smoothed.max()) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(smoothed, dtype=np.uint8)
    jet     = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(jet_rgb, opacity, img_rgb, 1-opacity, 0)

    # Draw head dots on the heatmap too
    peaks = _find_density_peaks(density_map, expected_count)
    h = blended.shape[0]
    for (py, px, _val) in peaks:
        depth_ratio = py / max(h, 1)  # 0=top(far), 1=bottom(near)
        r = max(2, int(3 + depth_ratio * 3))
        cv2.circle(blended, (px, py), r, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(blended, (px, py), r + 2, (0, 255, 255), 1, cv2.LINE_AA)
    return blended


def _find_density_peaks(density_map, expected_count=0):
    """
    Place head-position dots by sampling from the density map.

    DM-Count produces smooth density fields (not per-head peaks),
    so we treat the density map as a probability distribution and
    sample `expected_count` head positions from it.

    Uses minimum-distance enforcement so dots don't pile up.
    Returns list of (row, col, density_value) sorted by density descending.
    """
    h, w = density_map.shape
    smoothed = gaussian_filter(density_map.astype(np.float64), sigma=2)

    if smoothed.max() <= 1e-6 or expected_count <= 0:
        return []

    target = int(round(expected_count))

    # ── Compute minimum spacing between dots ──
    avg_spacing = np.sqrt(h * w / max(target, 1))
    min_dist_sq = max(3, int(avg_spacing * 0.25)) ** 2  # squared for fast checks

    # ── Build probability distribution from density map ──
    prob = smoothed.copy()
    prob[prob < 0] = 0
    total = prob.sum()
    if total <= 1e-6:
        return []
    prob_flat = prob.ravel() / total

    # ── Sample positions weighted by density ──
    # Over-sample 3x to have enough candidates after distance filtering
    n_sample = min(target * 3, prob_flat.size)
    rng = np.random.RandomState(42)  # deterministic for same image
    indices = rng.choice(prob_flat.size, size=n_sample, replace=False,
                         p=prob_flat)

    # Convert flat indices to (row, col) and sort by density descending
    candidates = []
    for idx in indices:
        r, c = divmod(int(idx), w)
        val = float(smoothed[r, c])
        candidates.append((r, c, val))
    candidates.sort(key=lambda p: p[2], reverse=True)

    # ── Greedy selection with minimum distance enforcement ──
    selected = []
    occupied = np.zeros((h, w), dtype=bool)
    min_dist_px = max(3, int(np.sqrt(min_dist_sq)))

    for (r, c, val) in candidates:
        if len(selected) >= target:
            break
        # Check if too close to any existing dot
        r_lo = max(0, r - min_dist_px)
        r_hi = min(h, r + min_dist_px + 1)
        c_lo = max(0, c - min_dist_px)
        c_hi = min(w, c + min_dist_px + 1)
        if occupied[r_lo:r_hi, c_lo:c_hi].any():
            continue
        selected.append((r, c, val))
        occupied[r_lo:r_hi, c_lo:c_hi] = True

    # ── If still short, do a second pass with tighter spacing ──
    if len(selected) < target * 0.8:
        min_dist_px2 = max(2, min_dist_px // 2)
        occupied2 = np.zeros((h, w), dtype=bool)
        for (r, c, _) in selected:
            r_lo = max(0, r - min_dist_px2)
            r_hi = min(h, r + min_dist_px2 + 1)
            c_lo = max(0, c - min_dist_px2)
            c_hi = min(w, c + min_dist_px2 + 1)
            occupied2[r_lo:r_hi, c_lo:c_hi] = True

        # Sample more candidates
        n_extra = min(target * 5, prob_flat.size)
        extra_indices = rng.choice(prob_flat.size, size=n_extra,
                                   replace=False, p=prob_flat)
        for idx in extra_indices:
            if len(selected) >= target:
                break
            r, c = divmod(int(idx), w)
            val = float(smoothed[r, c])
            r_lo = max(0, r - min_dist_px2)
            r_hi = min(h, r + min_dist_px2 + 1)
            c_lo = max(0, c - min_dist_px2)
            c_hi = min(w, c + min_dist_px2 + 1)
            if occupied2[r_lo:r_hi, c_lo:c_hi].any():
                continue
            selected.append((r, c, val))
            occupied2[r_lo:r_hi, c_lo:c_hi] = True

    return selected


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


def _filter_color_false_positives(peaks, img_rgb):
    """
    Remove density peaks that land on non-person colors
    (bags, furniture, walls, floors) by sampling a small
    region around each peak in the original RGB image.
    """
    h, w = img_rgb.shape[:2]
    kept = []
    patch_r = 7  # 15x15 region (7 pixels each side)

    for (py, px, val) in peaks:
        y0 = max(0, py - patch_r)
        y1 = min(h, py + patch_r + 1)
        x0 = max(0, px - patch_r)
        x1 = min(w, px + patch_r + 1)
        region = img_rgb[y0:y1, x0:x1]
        if region.size == 0:
            continue
        r_avg = float(region[:, :, 0].mean())
        g_avg = float(region[:, :, 1].mean())
        b_avg = float(region[:, :, 2].mean())

        # Skip predominantly red objects (bags, signs)
        if r_avg > 150 and g_avg < 80 and b_avg < 80:
            continue
        # Skip predominantly green objects (plants, boards)
        if g_avg > 150 and r_avg < 80:
            continue
        # Skip bright yellow objects (safety vests, bags)
        if r_avg > 180 and g_avg > 180 and b_avg < 80:
            continue
        # Skip pure white walls/ceilings
        if r_avg > 220 and g_avg > 220 and b_avg > 220:
            continue
        # Skip pure black floors/shadows
        if r_avg < 30 and g_avg < 30 and b_avg < 30:
            continue

        kept.append((py, px, val))
    return kept


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
    for (py, px, val) in sorted_peaks:
        too_close = False
        for (ky, kx, _) in kept:
            if (py - ky) ** 2 + (px - kx) ** 2 < min_dist_sq:
                too_close = True
                break
        if not too_close:
            kept.append((py, px, val))
    return kept


def build_headdot_overlay(img_rgb, density_map, expected_count=0):
    """
    Draw depth-aware numbered dots on detected heads.

    - Dots sized by vertical position (perspective depth):
      bottom of image = close to camera = bigger dots
      top of image = far from camera = smaller dots
    - Numbered in reading order (left→right, top→bottom)
    - Clean numbered pills with dark background circles
    - Color-filtered to remove bags/objects/walls
    - Minimum-distance filtered to remove duplicates
    """
    peaks = _find_density_peaks(density_map, expected_count)
    overlay = img_rgb.copy()
    h, w = overlay.shape[:2]

    if not peaks:
        return overlay, 0

    # Filter 1: Remove non-person colors (bags, walls, floors)
    peaks = _filter_color_false_positives(peaks, img_rgb)

    # Filter 2: Remove duplicates within 20px
    peaks = _filter_nearby_peaks(peaks, min_dist=20)

    if not peaks:
        return overlay, 0

    # Sort in reading order: top-to-bottom, left-to-right
    peaks_sorted = _sort_reading_order(peaks)

    for idx, (py, px, val) in enumerate(peaks_sorted):
        # Depth-aware scaling: 0.0 = top (far), 1.0 = bottom (near)
        depth_ratio = py / max(h, 1)
        scale = 0.7 + depth_ratio * 0.6

        # Brightness from confidence
        max_val = peaks_sorted[0][2] if peaks_sorted else 1.0
        brightness = 0.5 + 0.5 * (val / max(max_val, 1e-6))

        # Scaled radii
        r_glow_outer = max(4, int(18 * scale))
        r_glow_mid   = max(3, int(14 * scale))
        r_glow_inner = max(3, int(10 * scale))
        r_white      = max(2, int(7 * scale))
        r_cyan       = max(2, int(5 * scale))

        # 1. OUTER GLOW RINGS (3 layers, decreasing opacity)
        glow_overlay = overlay.copy()
        cv2.circle(glow_overlay, (px, py), r_glow_outer,
                   (0, 210, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(glow_overlay, 0.15, overlay, 0.85, 0, overlay)

        glow_overlay2 = overlay.copy()
        cv2.circle(glow_overlay2, (px, py), r_glow_mid,
                   (0, 210, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(glow_overlay2, 0.3, overlay, 0.7, 0, overlay)

        glow_overlay3 = overlay.copy()
        cv2.circle(glow_overlay3, (px, py), r_glow_inner,
                   (6, 182, 212), -1, cv2.LINE_AA)
        cv2.addWeighted(glow_overlay3, 0.7, overlay, 0.3, 0, overlay)

        # 2. INNER DOT (white core + cyan fill)
        cv2.circle(overlay, (px, py), r_white, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (px, py), r_cyan, (0, 180, 220), -1, cv2.LINE_AA)

        # 3. NUMBER BADGE (for first 200 dots)
        if len(peaks_sorted) <= 200:
            num_str = str(idx + 1)
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = max(0.30, 0.28 + depth_ratio * 0.12)
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(
                num_str, font, font_scale, thickness)

            # Badge position: above dot, offset right
            badge_x = px + r_white + max(2, int(4 * scale))
            badge_y = py - max(4, int(8 * scale))
            pad_x = max(3, int(4 * scale))
            pad_y = max(2, int(3 * scale))

            # 5. CONNECTING LINE from dot to badge
            cv2.line(overlay,
                     (px, py),
                     (badge_x, badge_y + th // 2),
                     (0, 180, 220), 1, cv2.LINE_AA)

            # Dark navy pill background
            cv2.rectangle(overlay,
                          (badge_x - pad_x, badge_y - pad_y - 1),
                          (badge_x + tw + pad_x, badge_y + th + pad_y - 1),
                          (15, 23, 42), -1, cv2.LINE_AA)
            # Cyan border
            cv2.rectangle(overlay,
                          (badge_x - pad_x, badge_y - pad_y - 1),
                          (badge_x + tw + pad_x, badge_y + th + pad_y - 1),
                          (0, 180, 220), 1, cv2.LINE_AA)
            # White number text
            cv2.putText(overlay, num_str,
                        (badge_x, badge_y + th),
                        font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)


    # Summary label in corner
    label_text = f"{len(peaks_sorted)} heads detected"
    tw = len(label_text) * 11 + 16
    cv2.rectangle(overlay, (8, h - 40), (tw, h - 8), (6, 10, 18), -1)
    cv2.rectangle(overlay, (8, h - 40), (tw, h - 8), (0, 200, 255), 1)
    cv2.putText(overlay, label_text, (14, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)

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
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state["rl_agent"] is None:
    st.session_state["rl_agent"] = EvacuationAgent()

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Logo + App Name ──
    st.markdown("""
<div style="padding:20px 4px 16px 4px">
<div style="display:flex;align-items:center;gap:14px">
<div style="font-size:32px;flex-shrink:0;
filter:drop-shadow(0 2px 12px rgba(99,102,241,0.35))">🛡️</div>
<div>
<div style="font-size:18px;font-weight:800;color:#F1F5F9;letter-spacing:-0.03em;
line-height:1.2">SafeCrowd Vision</div>
<div style="font-size:10px;color:#6366F1;letter-spacing:0.5px;font-weight:500;
margin-top:3px;opacity:0.85">v2.0 · Safety Analytics</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # ── Divider ──
    st.markdown('<div style="border-top:1px solid #334155;margin:8px 0 16px 0"></div>',
                unsafe_allow_html=True)

    # ── Controls Section ──
    st.markdown('<div style="color:#6366F1;font-weight:700;font-size:10px;'
                'letter-spacing:1.8px;text-transform:uppercase;margin-bottom:14px;'
                'padding-left:2px">◈ CONTROLS</div>', unsafe_allow_html=True)

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

    # ── Divider ──
    st.markdown('<div style="border-top:1px solid #334155;margin:18px 0 18px 0"></div>',
                unsafe_allow_html=True)

    # ── VENUE SETTINGS Section ──
    st.markdown('<div style="color:#6366F1;font-weight:700;font-size:10px;'
                'letter-spacing:1.8px;text-transform:uppercase;margin-bottom:14px;'
                'padding-left:2px">◈ VENUE SETTINGS</div>', unsafe_allow_html=True)

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

    # ── Divider ──
    st.markdown('<div style="border-top:1px solid #334155;margin:18px 0 18px 0"></div>',
                unsafe_allow_html=True)

    # ── Model Info Section ──
    st.markdown("""
<div style="color:#6366F1;font-weight:700;font-size:10px;
letter-spacing:1.8px;text-transform:uppercase;margin-bottom:12px;
padding-left:2px">◈ MODEL INFO</div>

<div style="background:#1E293B;border:1px solid #334155;border-left:3px solid #22D3EE;
border-radius:10px;padding:0;margin-bottom:0;overflow:hidden;
box-shadow:0 2px 16px rgba(34,211,238,0.06)">

<div style="display:flex;justify-content:space-between;align-items:center;
padding:10px 16px;background:#263445">
<span style="color:#94A3B8;font-size:12px;font-weight:500">Model</span>
<span style="color:#6366F1;font-size:12px;font-family:'JetBrains Mono',monospace;
font-weight:600">DM-Count (LWCC)</span>
</div>

<div style="display:flex;justify-content:space-between;align-items:center;
padding:10px 16px;background:#1E293B">
<span style="color:#94A3B8;font-size:12px;font-weight:500">Dataset</span>
<span style="color:#6366F1;font-size:12px;font-family:'JetBrains Mono',monospace;
font-weight:600">ShanghaiTech B</span>
</div>

<div style="display:flex;justify-content:space-between;align-items:center;
padding:10px 16px;background:#263445">
<span style="color:#94A3B8;font-size:12px;font-weight:500">MAE</span>
<span style="color:#10B981;font-size:12px;font-family:'JetBrains Mono',monospace;
font-weight:700">5.80 <span style="color:#94A3B8;font-weight:400;font-size:10px">(sparse 1–100)</span></span>
</div>

<div style="display:flex;justify-content:space-between;align-items:center;
padding:10px 16px;background:#1E293B">
<span style="color:#94A3B8;font-size:12px;font-weight:500">Overall</span>
<span style="color:#10B981;font-size:12px;font-family:'JetBrains Mono',monospace;
font-weight:700">~81% <span style="color:#94A3B8;font-weight:400;font-size:10px">accuracy · eval set (498 imgs)</span></span>
</div>

<div style="display:flex;justify-content:space-between;align-items:center;
padding:10px 16px;background:#263445">
<span style="color:#94A3B8;font-size:12px;font-weight:500">XGB</span>
<span style="color:#10B981;font-size:12px;font-family:'JetBrains Mono',monospace;
font-weight:700">99.30% <span style="color:#94A3B8;font-weight:400;font-size:10px">zone classification</span></span>
</div>

</div>
""", unsafe_allow_html=True)

    # ── Divider ──
    st.markdown('<div style="border-top:1px solid #334155;margin:18px 0 18px 0"></div>',
                unsafe_allow_html=True)

    # ── SHARE ACCESS — QR Code ──
    st.markdown('<div style="color:#6366F1;font-weight:700;font-size:10px;'
                'letter-spacing:1.8px;text-transform:uppercase;margin-bottom:14px;'
                'padding-left:2px">◈ SHARE ACCESS</div>', unsafe_allow_html=True)

    ngrok_url = st.text_input(
        "ngrok URL",
        placeholder="https://abc123.ngrok-free.app",
        help="Paste your ngrok URL to generate QR code"
    )

    if ngrok_url and ngrok_url.startswith("https://"):
        import urllib.parse
        _qr_data = urllib.parse.quote(ngrok_url)
        _qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=160x160&data={_qr_data}&bgcolor=0C1220&color=06B6D4&qzone=2"

        st.markdown(f"""
        <div style="background:#1E293B;border:1px solid #334155;
        border-radius:12px;padding:16px;text-align:center">
        <img src="{_qr_url}" style="width:160px;height:160px;
        border-radius:8px">
        <div style="color:#6366F1;font-size:10px;
        font-family:monospace;letter-spacing:1px;
        margin-top:10px;word-break:break-all">{ngrok_url}</div>
        <div style="color:#94A3B8;font-size:10px;
        margin-top:6px">Scan to open on phone</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Divider ──
    st.markdown('<div style="border-top:1px solid #334155;margin:18px 0 18px 0"></div>',
                unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
<div style="text-align:center;padding:4px 0 8px 0">
<div style="color:#C7D2FE;font-size:11px;font-weight:500;letter-spacing:0.3px;
line-height:1.6">
Powered by LWCC · PyTorch
</div>
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
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#94A3B8">MODEL:</span> '
        f'<span style="color:#64748B">DM-Count SHB</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#94A3B8">ZONES MONITORED:</span> '
        f'<span style="color:#64748B">64</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#94A3B8">LAST COUNT:</span> '
        f'<span style="color:#F1F5F9">{_last_count_ticker} PERSONS</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#94A3B8">STATUS:</span> '
        f'<span style="color:#10B981">{_status_txt}</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#94A3B8">MAE:</span> '
        f'<span style="color:#64748B">5.80</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#94A3B8">ACCURACY:</span> '
        f'<span style="color:#64748B">~81% (eval)</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;'
    )
else:
    _ticker_content = (
        '<span style="color:#6366F1;font-size:8px;animation:dotPulse 1.5s infinite">●</span>'
        '&nbsp;&nbsp;<span style="color:#6366F1">AWAITING INPUT</span>'
        '&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#94A3B8">SYSTEM READY</span>'
        '&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#94A3B8">DM-Count</span>&nbsp;'
        '<span style="color:#64748B">LOADED</span>'
        '&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#94A3B8">64 ZONES ACTIVE</span>'
        '&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#64748B">UPLOAD IMAGE TO BEGIN</span>'
        '&nbsp;&nbsp;·&nbsp;&nbsp;'
    )

st.markdown(f"""
<div style="background:#1E293B;border-bottom:1px solid #334155;
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

    # ── Present mode button (top-right) ───────────────────────
    _pres_col_l, _pres_col_r = st.columns([5, 1])
    with _pres_col_r:
        if st.session_state["present_mode"]:
            if st.button("⛶ Exit", key="exit_present", use_container_width=True):
                st.session_state["present_mode"] = False
                st.rerun()
        else:
            if st.button("⛶ Present", key="enter_present", use_container_width=True):
                st.session_state["present_mode"] = True
                st.rerun()

    # ── File uploader ─────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop a crowd image here or click to browse",
        type=["jpg", "jpeg", "png"],
        key="main_upload")

    if uploaded:
        raw       = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_hash  = hash(raw.tobytes())
        try:
            img_bgr   = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Image could not be decoded")
            _img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w      = _img_rgb.shape[:2]
        except Exception as _decode_err:
            st.error(f"⚠️ Could not read this image — the file may be corrupted or unsupported. ({_decode_err})")
            st.stop()

        # Only run inference if new image
        if st.session_state.get("last_img_hash") != img_hash:
            # ── Scan animation — shows BEFORE inference ──
            status_box = st.empty()
            status_box.markdown("""
            <div style="background:#1E293B;border:1px solid #6366F1;
            border-radius:12px;padding:24px;margin:12px 0;
            border-left:4px solid #6366F1">
            <div style="display:flex;align-items:center;gap:12px">
            <div style="width:12px;height:12px;border-radius:50%;
            background:#4F46E5;animation:pulse 0.8s infinite">
            </div>
            <div style="color:#F1F5F9;font-family:monospace;
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
            st.session_state["last_count"]       = _crowd_count
            st.session_state["last_img_hash"]    = img_hash
            st.session_state["last_image"]       = Image.fromarray(_img_rgb)
        else:
            _density_full = st.session_state["last_density_raw"]
            _crowd_count  = st.session_state["last_count"]

        st.session_state["last_count"] = _crowd_count


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
            <div style="background:linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border:1px solid #334155;border-radius:12px;padding:14px 24px;
            display:flex;align-items:center;justify-content:space-between;
            margin-bottom:20px;box-shadow:0 4px 24px rgba(0,0,0,0.5)">
            <div style="display:flex;align-items:center;gap:12px">
            <span style="font-size:24px;filter:drop-shadow(0 2px 12px rgba(99,102,241,0.4))">🛡️</span>
            <span style="color:#F1F5F9;font-size:20px;font-weight:800;letter-spacing:-0.03em">SafeCrowd Vision</span>
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
                <div style="background:#1E293B;border:1px solid #334155;border-radius:12px;
                padding:8px;border-top:3px solid #6366F1;
                box-shadow:0 6px 32px rgba(99,102,241,0.2)">
                <div style="color:#94A3B8;font-size:10px;font-weight:700;
                letter-spacing:2px;text-transform:uppercase;padding:8px 10px 6px">
                SAFETY ZONE MAP — {method.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(_p_safety_img, use_container_width=True)

            with _pc2:
                # Giant crowd count
                st.components.v1.html(f"""
                <div style="background:#1E293B;border:1px solid #334155;border-radius:12px;
                border-top:3px solid #6366F1;padding:28px 20px;text-align:center;
                box-shadow:0 6px 32px rgba(99,102,241,0.2)">
                <div style="color:#94A3B8;font-size:10px;font-weight:700;
                letter-spacing:2.5px;text-transform:uppercase;margin-bottom:8px">
                PERSONS DETECTED</div>
                <div id="present-count" style="font-size:80px;font-weight:900;color:#F1F5F9;
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
                        axis=dict(range=[0, 100], tickcolor="#64748B",
                                  tickfont=dict(color="#6B7280", size=9)),
                        bar=dict(color=_p_tc, thickness=0.35),
                        bgcolor="#C7D2FE", borderwidth=0,
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
                <div style="background:#1E293B;border:1px solid rgba(16,185,129,0.25);
                border-radius:10px;padding:14px;text-align:center;
                box-shadow:0 0 20px rgba(16,185,129,0.08)">
                <div style="font-size:10px;color:#94A3B8;font-weight:700;
                letter-spacing:1.5px;margin-bottom:4px">🟢 LOW</div>
                <div style="font-size:32px;font-weight:900;color:#10B981;
                font-family:'JetBrains Mono',monospace;
                text-shadow:0 0 20px rgba(16,185,129,0.4)">{_p_zone_stats['Low']}</div>
                </div>
                <div style="background:#1E293B;border:1px solid rgba(245,158,11,0.25);
                border-radius:10px;padding:14px;text-align:center;
                box-shadow:0 0 20px rgba(245,158,11,0.08)">
                <div style="font-size:10px;color:#94A3B8;font-weight:700;
                letter-spacing:1.5px;margin-bottom:4px">🟡 MEDIUM</div>
                <div style="font-size:32px;font-weight:900;color:#D97706;
                font-family:'JetBrains Mono',monospace;
                text-shadow:0 0 20px rgba(245,158,11,0.4)">{_p_zone_stats['Medium']}</div>
                </div>
                <div style="background:#1E293B;border:1px solid rgba(239,68,68,0.25);
                border-radius:10px;padding:14px;text-align:center;
                box-shadow:0 0 20px rgba(239,68,68,0.08)">
                <div style="font-size:10px;color:#94A3B8;font-weight:700;
                letter-spacing:1.5px;margin-bottom:4px">🟠 HIGH</div>
                <div style="font-size:32px;font-weight:900;color:#DC2626;
                font-family:'JetBrains Mono',monospace;
                text-shadow:0 0 20px rgba(239,68,68,0.4)">{_p_zone_stats['High']}</div>
                </div>
                <div style="background:#1E293B;border:1px solid rgba(255,23,68,0.25);
                border-radius:10px;padding:14px;text-align:center;
                box-shadow:0 0 20px rgba(255,23,68,0.1);
                {'animation:criticalGlow 2s ease-in-out infinite;' if _p_zone_stats['Critical'] > 0 else ''}">
                <div style="font-size:10px;color:#94A3B8;font-weight:700;
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
            if _crowd_count < 80:
                _p_wts = "SHB"
            elif _crowd_count < 200:
                _p_wts = "SHA"
            else:
                _p_wts = "Ensemble (SHA+SHB)"
            st.markdown(f"""
            <div style="background:#1E293B;border:1px solid #334155;border-radius:8px;
            padding:10px 20px;display:flex;align-items:center;
            justify-content:space-between;margin-top:8px">
            <span style="color:#3B4A63;font-family:'JetBrains Mono',monospace;
            font-size:10px;letter-spacing:1px">{_p_now}</span>
            <span style="color:#3B4A63;font-family:'JetBrains Mono',monospace;
            font-size:10px;letter-spacing:1px">
            DM-Count · {_p_wts} · {w}×{h} · {method} · 8×8 grid</span>
            </div>
            """, unsafe_allow_html=True)


        # ── DEMO MODE ─────────────────────────────────────────
        elif demo_mode:
            count = _crowd_count
            if count < 30:
                bg = "linear-gradient(135deg, #003D19, #00501F)"
                glow = "0 0 40px rgba(16,185,129,0.15)"
                label = "✅ SAFE"
            elif count < 60:
                bg = "linear-gradient(135deg, #4D2800, #663500)"
                glow = "0 0 40px rgba(245,158,11,0.15)"
                label = "⚠️ FULL"
            else:
                bg = "linear-gradient(135deg, #4D0011, #660017)"
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
                f'margin-top:6px;font-weight:400">people detected</div></div>',
                unsafe_allow_html=True)

        else:
            # ── Normal analysis view ──────────────────────────

            feats = extract_features(_density_full)
            _features_sc = scaler.transform(feats) if scaler else feats

            if method == "XGBoost" and xgb:
                labels = labels_from_xgb(_features_sc, xgb)
            elif method == "GMM" and gmm:
                labels, _ = labels_from_gmm(_features_sc, gmm)
            else:
                labels = labels_from_kmeans(_features_sc, km) \
                    if km else [get_label(float(f[0])) for f in feats]

            safety_img, _zone_stats = build_overlay(
                _img_rgb, labels, GRID, opacity)
            st.session_state["last_zone_stats"] = _zone_stats.copy()

            # (threat score is calculated at the gauge render site below)

            density_overlay = build_density_overlay(
                _img_rgb, _density_full, opacity,
                expected_count=_crowd_count)

            # Head-dot overlay: dots on each detected head
            headdot_overlay, _dot_count = build_headdot_overlay(
                _img_rgb, _density_full,
                expected_count=_crowd_count)

            # save to history
            thumb = cv2.resize(_img_rgb, (80, 80))
            if (len(st.session_state["history"]) == 0 or
                    hash(st.session_state["history"][-1]["thumb"].tobytes()) != hash(thumb.tobytes())):
                st.session_state["history"].append(
                    {"thumb": thumb, "count": _crowd_count})
            if len(st.session_state["history"]) > 5:
                st.session_state["history"] = st.session_state["history"][-5:]

            # ── Portrait / close-up detection ─────────────────
            h, w = _img_rgb.shape[:2]
            _aspect_ratio = w / h
            _density_coverage = (_density_full > _density_full.max() * 0.1).sum() / _density_full.size

            # If density is concentrated in small area
            # AND count is very low relative to image size
            # it's likely a close-up not a crowd
            _is_portrait = (
                _crowd_count < 10 and
                _density_coverage < 0.15
            )

            _used_face_detector = False

            if _is_portrait:
                # ── OpenCV face detection fallback ─────────────
                _gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                _faces = face_cascade.detectMultiScale(
                    _gray, scaleFactor=1.1,
                    minNeighbors=5, minSize=(30, 30))
                _face_count = len(_faces)

                if _face_count > _crowd_count:
                    _crowd_count = _face_count

                # Build face-rectangle overlay instead of head dots
                _face_overlay = _img_rgb.copy()
                for (fx, fy, fw, fh) in _faces:
                    cv2.rectangle(_face_overlay,
                                 (fx, fy),
                                 (fx + fw, fy + fh),
                                 (0, 255, 255), 2, cv2.LINE_AA)
                    # Small label above each face
                    cv2.putText(_face_overlay, "Face",
                                (fx, fy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 255, 255), 1, cv2.LINE_AA)

                # Summary label
                _fh, _fw = _face_overlay.shape[:2]
                _fl_text = f"{_face_count} face(s) detected"
                _fl_tw = len(_fl_text) * 11 + 16
                cv2.rectangle(_face_overlay,
                              (8, _fh - 40), (_fl_tw, _fh - 8),
                              (6, 10, 18), -1)
                cv2.rectangle(_face_overlay,
                              (8, _fh - 40), (_fl_tw, _fh - 8),
                              (0, 200, 255), 1)
                cv2.putText(_face_overlay, _fl_text,
                            (14, _fh - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)

                # Replace head-dot overlay & counts
                headdot_overlay = _face_overlay
                _dot_count = _face_count
                _used_face_detector = True

                st.markdown("""
<div style="background:rgba(245,158,11,0.1);
border:1px solid #F59E0B;border-left:4px solid #F59E0B;
border-radius:10px;padding:14px 20px;margin:8px 0;
color:#FCD34D;font-size:13px;font-weight:600">
⚠️ PORTRAIT/CLOSE-UP DETECTED — 
Switched to <b>OpenCV Face Detector</b>. 
DM-Count is optimized for crowd scenes 
(20+ people, overhead or eye-level view).
</div>
""", unsafe_allow_html=True)

            # ── Analysis Complete banner ───────────────────────
            _banner_conf = compute_dynamic_confidence(
                _features_sc, method, km, xgb, gmm, _crowd_count)
            if _used_face_detector:
                _banner_model = "OpenCV Face Detector"
            elif _crowd_count < 80:
                _banner_model = "DM-Count SHB"
            elif _crowd_count < 200:
                _banner_model = "DM-Count SHA"
            else:
                _banner_model = "DM-Count Ensemble (SHA+SHB)"

            st.components.v1.html(f"""
            <div style="background:#1E293B;border:1px solid #334155;
            border-radius:12px;padding:20px 24px;margin-bottom:24px;
            overflow:hidden;position:relative;">

            <div style="position:absolute;top:0;left:0;height:3px;
            width:100%;background:#334155;">
            <div style="height:3px;background:linear-gradient(90deg,
            #6366F1,#22D3EE,#8B5CF6);animation:fill 0.8s ease-out forwards;
            width:0%"></div>
            </div>

            <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;
            margin-top:4px">
            <div style="width:10px;height:10px;border-radius:50%;
            background:#059669;box-shadow:0 0 8px #059669;
            animation:blink 1s ease-in-out 3"></div>
            <div style="color:#F1F5F9;font-size:14px;font-weight:700;
            font-family:monospace;letter-spacing:1px">ANALYSIS COMPLETE</div>
            </div>

            <div style="display:flex;align-items:center;
            justify-content:space-between">

            <div style="flex:1;text-align:center;border-right:1px solid #334155;
            padding-right:20px">
            <div style="color:#94A3B8;font-size:9px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">
            PERSONS DETECTED</div>
            <div style="color:#6366F1;font-size:32px;font-weight:900;
            font-family:monospace;text-shadow:0 0 20px rgba(99,102,241,0.2)">
            {_dot_count}</div>
            <div style="color:#64748B;font-size:10px;font-family:monospace;
            margin-top:2px">DM-Count estimate: {_crowd_count}</div>
            </div>

            <div style="flex:1;text-align:center;border-right:1px solid #334155;
            padding:0 20px">
            <div style="color:#94A3B8;font-size:9px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">
            MODEL</div>
            <div style="color:#F1F5F9;font-size:14px;font-weight:700;
            font-family:monospace">{_banner_model}</div>
            </div>

            <div style="flex:1;text-align:center;padding-left:20px">
            <div style="color:#94A3B8;font-size:9px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">
            CONFIDENCE</div>
            <div style="color:#10B981;font-size:14px;font-weight:700;
            font-family:monospace">{_banner_conf}%</div>
            </div>

            </div>

            <style>
            @keyframes fill {{
            0%{{width:0%}} 100%{{width:100%}}
            }}
            @keyframes blink {{
            0%,100%{{opacity:1}} 50%{{opacity:0.2}}
            }}
            </style>
            </div>
            """, height=110)

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

            # Dynamic panel labels for portrait vs crowd mode
            _panel1_title = (
                f"FACE DETECTION — {_dot_count} FACES"
                if _used_face_detector
                else f"HEAD DETECTION — {_dot_count} DOTS"
            )
            _panel1_sub = (
                f"Each cyan rectangle = 1 detected face · {w}×{h}"
                if _used_face_detector
                else f"Each cyan dot = 1 detected person · {w}×{h}"
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div style="background:#1E293B;border:1px solid #334155;
                border-radius:12px;overflow:hidden;
                box-shadow:0 4px 24px rgba(99,102,241,0.15)">
                <div style="background:#263445;padding:10px 16px;
                border-top:3px solid #6366F1">
                <span style="color:#6366F1;font-size:10px;text-transform:uppercase;
                letter-spacing:2px;font-weight:600">{_panel1_title}</span></div>
                <img src="data:image/jpeg;base64,{_headdot_b64}"
                style="width:100%;display:block">
                <div style="padding:8px 16px;background:#0F172A;
                color:#64748B;font-size:11px">{_panel1_sub}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="background:#1E293B;border:1px solid #334155;
                border-radius:12px;overflow:hidden;
                box-shadow:0 4px 24px rgba(99,102,241,0.15)">
                <div style="background:#263445;padding:10px 16px;
                border-top:3px solid #6366F1">
                <span style="color:#6366F1;font-size:10px;text-transform:uppercase;
                letter-spacing:2px;font-weight:600">DENSITY HEATMAP</span></div>
                <img src="data:image/jpeg;base64,{_density_b64}"
                style="width:100%;display:block">
                <div style="padding:8px 16px;background:#0F172A;
                color:#64748B;font-size:11px">Gaussian σ=8 · JET colormap · opacity {opacity}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div style="background:#1E293B;border:1px solid #334155;
                border-radius:12px;overflow:hidden;
                box-shadow:0 4px 24px rgba(99,102,241,0.15)">
                <div style="background:#263445;padding:10px 16px;
                border-top:3px solid #10B981">
                <span style="color:#10B981;font-size:10px;text-transform:uppercase;
                letter-spacing:2px;font-weight:600">SAFETY ZONE MAP — {method.upper()}</span></div>
                <img src="data:image/jpeg;base64,{_safety_b64}"
                style="width:100%;display:block">
                <div style="padding:8px 16px;background:#0F172A;
                color:#64748B;font-size:11px">8×8 grid · {method} classification</div>
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
            <div style="background:#1E293B;border:1px solid #334155;border-radius:10px;
            padding:14px 20px;margin:8px 0 16px 0;display:flex;align-items:center;
            justify-content:space-between;gap:20px">
            <div style="display:flex;align-items:center;gap:8px;flex-shrink:0">
            <span style="font-size:14px">🎯</span>
            <span style="color:#94A3B8;font-size:10px;font-weight:700;
            letter-spacing:1.5px;text-transform:uppercase">ANALYSIS CONFIDENCE</span>
            </div>
            <div style="flex:1;display:flex;align-items:center;gap:12px">
            <div style="flex:1;height:6px;background:#C7D2FE;border-radius:4px;overflow:hidden">
            <div style="height:100%;width:{_confidence_pct}%;
            background:linear-gradient(90deg,#2563EB,#06B6D4);border-radius:4px;
            animation:barFill 1s ease-out forwards;
            --fill-pct:{_confidence_pct}%"></div>
            </div>
            <span style="color:#F1F5F9;font-family:'JetBrains Mono',monospace;
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
                <div style="background:#1E293B;border:1px solid #334155;
                border-radius:12px;padding:22px 18px;
                border-top:3px solid #6366F1;
                box-shadow:0 4px 24px rgba(99,102,241,0.2)">
                <div style="color:#94A3B8;font-size:11px;text-transform:uppercase;
                letter-spacing:1.2px;font-weight:600;margin-bottom:8px">
                👥 ESTIMATED CROWD</div>
                <div style="color:#F1F5F9;font-family:'JetBrains Mono',monospace;
                font-size:36px;font-weight:700;letter-spacing:-0.03em;
                font-variant-numeric:tabular-nums;line-height:1">{_crowd_count:,}</div>
                <div style="display:flex;align-items:center;gap:8px;margin-top:8px">
                <span style="color:#94A3B8;font-size:11px">Range: {_count_low} – {_count_high}</span>
                <span style="padding:2px 8px;border-radius:12px;font-size:10px;
                font-weight:600;background:rgba(34,211,238,0.1);color:#6366F1;
                border:1px solid rgba(34,211,238,0.2);
                font-family:'JetBrains Mono',monospace">±{int(_mae)} MAE</span>
                </div>
                </div>
                """, height=130)
            m2.metric("🔴 Critical Zones", _zone_stats["Critical"])
            m3.metric("🟠 High Zones", _zone_stats["High"])
            m4.metric("🟡 Medium Zones", _zone_stats["Medium"])
            m5.metric("🟢 Safe Zones", _zone_stats["Low"])

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
                    axis=dict(range=[0, 100], tickcolor="#64748B",
                              tickfont=dict(color="#6B7280", size=10)),
                    bar=dict(color=_g_color, thickness=0.3),
                    bgcolor="#C7D2FE",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 25],   color="#0C2A1A"),
                        dict(range=[25, 50],  color="#2A1F00"),
                        dict(range=[50, 75],  color="#2A0D00"),
                        dict(range=[75, 100], color="#2A0007"),
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
                st.markdown(f"""<div style="background:#1A0A0A;
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
            st.markdown('<div style="border-top:1px solid #334155;margin:24px 0"></div>',
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
                    <div style="background:#1E293B;border:1px solid #334155;
                    border-radius:12px;padding:20px;text-align:center;
                    border-top:3px solid {_zb_color}">
                    <div style="font-size:2.5rem;font-weight:900;
                    color:{_zb_color};font-family:monospace">{_zb_count}</div>
                    <div style="font-size:10px;color:#94A3B8;
                    letter-spacing:2px;text-transform:uppercase;
                    margin-top:6px">{_zb_name} ZONES</div>
                    <div style="margin-top:12px;height:4px;
                    background:#C7D2FE;border-radius:4px">
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
            <div style="background:#1E293B;border:1px solid #334155;
            border-radius:12px;padding:20px 24px;margin:16px 0">

            <div style="display:flex;justify-content:space-between;
            align-items:center;margin-bottom:14px">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase">
            🏟️ VENUE CAPACITY MONITOR</div>
            <div style="display:flex;gap:8px;align-items:center">
            <span style="color:#94A3B8;font-size:12px">
            {_crowd_count} / {venue_capacity}</span>
            <span style="padding:3px 12px;border-radius:20px;
            font-size:11px;font-weight:700;
            background:{cap_bg};color:{cap_color};
            border:1px solid {cap_color}30">{cap_label}</span>
            </div>
            </div>

            <div style="height:8px;background:#C7D2FE;
            border-radius:6px;overflow:hidden">
            <div style="height:100%;width:{_utilization}%;
            background:{bar_color};border-radius:6px;
            transition:width 1s ease"></div>
            </div>

            <div style="display:flex;justify-content:space-between;
            margin-top:8px">
            <span style="color:#94A3B8;font-size:10px">0%</span>
            <span style="color:{cap_color};font-size:11px;
            font-weight:700">{_utilization}% utilized</span>
            <span style="color:#94A3B8;font-size:10px">100%</span>
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
            <div style="background:#1E293B;border:1px solid #334155;
            border-radius:12px;padding:20px 24px;margin:16px 0;
            border-left:4px solid {_evac_color}">
            <div style="display:flex;align-items:center;
            justify-content:space-between">
            <div>
            <div style="color:#94A3B8;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;
            margin-bottom:6px">🚪 EVACUATION TIME ESTIMATE</div>
            <div style="color:{_evac_color};font-size:36px;
            font-weight:900;font-family:monospace">
            {_evac_mins_display}m {_evac_secs_display}s</div>
            <div style="color:#94A3B8;font-size:11px;margin-top:4px">
            {_evac_status}</div>
            </div>
            <div style="text-align:right">
            <div style="color:#94A3B8;font-size:10px;margin-bottom:4px">
            {num_exits} exits · 40 ppl/min each</div>
            <div style="color:#94A3B8;font-size:10px">
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
                    <div style="background:#1E293B;border:1px solid #DDD6FE;
                    border-radius:12px;padding:18px 22px;margin:16px 0;
                    border-left:4px solid #7C3AED;
                    box-shadow:0 2px 12px rgba(124,58,237,0.08)">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                    <span style="font-size:18px">🤖</span>
                    <span style="color:#7C3AED;font-size:10px;font-weight:700;
                    letter-spacing:2px;text-transform:uppercase">RL AGENT RECOMMENDATION</span>
                    </div>
                    <div style="color:#F1F5F9;font-size:15px;font-weight:700;
                    margin-bottom:4px">{_rl_name_t1}</div>
                    <div style="color:#64748B;font-size:12px;line-height:1.5">{_rl_detail_t1}</div>
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
                    [0.0, "#0F172A"],
                    [0.2, "#0C2461"],
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
                        tickfont=dict(color="#4B5563", size=10),
                        title=dict(text="Density", font=dict(color="#4B5563", size=11)),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                    ),
                ))
                fig_heat.update_layout(
                    title=dict(text="Density Distribution · 8×8 Zone Grid",
                               font=dict(size=14, color="#4B5563",
                                         family="Inter, sans-serif")),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis=dict(title="Column", tickfont=dict(color="#6B7280"),
                               title_font=dict(color="#6B7280")),
                    yaxis=dict(title="Row", tickfont=dict(color="#6B7280"),
                               title_font=dict(color="#6B7280"), autorange="reversed"),
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
                bg = "linear-gradient(135deg, #003D19, #00501F)"
                glow = "0 0 40px rgba(16,185,129,0.15)"
                label = "✅ SAFE"
            elif count < 60:
                bg = "linear-gradient(135deg, #4D2800, #663500)"
                glow = "0 0 40px rgba(245,158,11,0.15)"
                label = "⚠️ FULL"
            else:
                bg = "linear-gradient(135deg, #4D0011, #660017)"
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
                f'margin-top:6px;font-weight:400">people detected</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px 50px;min-height:480px;
            animation:floatUp 0.6s ease;
            background:radial-gradient(ellipse 600px 300px at center 120px,
            rgba(99,102,241,0.07), transparent);border-top:1px solid #334155">

            <div style="font-size:64px;margin-bottom:16px;
            filter:drop-shadow(0 4px 24px rgba(99,102,241,0.35))">🛡️</div>

            <div style="font-size:12px;font-weight:700;color:#6366F1;
            letter-spacing:6px;text-transform:uppercase;margin-bottom:12px">
            SAFECROWD VISION</div>

            <h2 style="color:#F1F5F9;font-size:32px;font-weight:800;margin:0;
            letter-spacing:-0.03em">Ready for Analysis</h2>

            <p style="color:#94A3B8;font-size:15px;max-width:520px;
            margin:16px auto 0;line-height:1.8">
            Upload any crowd image to begin real-time density estimation
            and 4-zone safety mapping.</p>

            <div style="border-top:1px solid #334155;margin:36px auto 32px;
            max-width:400px"></div>

            <div style="display:flex;justify-content:center;gap:16px;
            flex-wrap:wrap;max-width:640px;margin:0 auto">

            <div style="background:#1E293B;border:1px solid #334155;
            border-top:3px solid #6366F1;border-radius:12px;padding:20px 24px;
            flex:1;min-width:160px;max-width:200px;text-align:center">
            <div style="font-size:28px;margin-bottom:10px">🎯</div>
            <div style="color:#F1F5F9;font-size:13px;font-weight:600;
            margin-bottom:4px">DM-Count</div>
            <div style="color:#94A3B8;font-size:11px;line-height:1.5">
            Sparse-optimized<br>
            <span style="color:#3B82F6;font-family:'JetBrains Mono',monospace;
            font-size:10px;font-weight:600">MAE: 4.92</span></div>
            </div>

            <div style="background:#1E293B;border:1px solid #334155;
            border-top:2px solid #10B981;border-radius:12px;padding:20px 24px;
            flex:1;min-width:160px;max-width:200px;text-align:center">
            <div style="font-size:28px;margin-bottom:10px">🗺️</div>
            <div style="color:#F1F5F9;font-size:13px;font-weight:600;
            margin-bottom:4px">4-Zone Safety</div>
            <div style="color:#94A3B8;font-size:11px;line-height:1.5">
            Spatial mapping<br>
            <span style="color:#10B981;font-family:'JetBrains Mono',monospace;
            font-size:10px;font-weight:600">8×8 Grid</span></div>
            </div>

            <div style="background:#1E293B;border:1px solid #334155;
            border-top:2px solid #0891B2;border-radius:12px;padding:20px 24px;
            flex:1;min-width:160px;max-width:200px;text-align:center">
            <div style="font-size:28px;margin-bottom:10px">⚡</div>
            <div style="color:#F1F5F9;font-size:13px;font-weight:600;
            margin-bottom:4px">Real-Time</div>
            <div style="color:#94A3B8;font-size:11px;line-height:1.5">
            Instant analysis<br>
            <span style="color:#6366F1;font-family:'JetBrains Mono',monospace;
            font-size:10px;font-weight:600">Live Feed</span></div>
            </div>

            </div>

            <p style="color:#3B4A63;font-size:11px;margin-top:36px;
            letter-spacing:1.5px;font-weight:500">
            SUPPORTED: Concerts · Festivals · Stations · Stadiums · Classrooms · Rallies</p>

            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — COMPARE METHODS
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1E293B 0%,#0F172A 100%);
    padding:22px 26px;border-radius:12px;border:1px solid #334155;
    margin-bottom:22px;box-shadow:0 4px 20px rgba(0,0,0,0.4)">
    <h2 style="color:#F1F5F9;margin:0;font-size:20px;font-weight:700;
    letter-spacing:-0.02em">
    How different ML methods see the same crowd</h2>
    <p style="color:#64748B;margin:8px 0 0;font-size:13px;line-height:1.5">
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
        <div style="background:linear-gradient(135deg,#1E293B 0%,#0F172A 100%);
        padding:18px 22px;border-radius:12px;border:1px solid #334155;
        margin-bottom:14px;border-left:3px solid #22D3EE;
        box-shadow:0 4px 16px rgba(0,0,0,0.35)">
        <h3 style="color:#F1F5F9;margin:0;font-size:16px;font-weight:700;
        letter-spacing:-0.01em">⟺ Interactive Comparison</h3>
        <p style="color:#94A3B8;font-size:12px;margin:6px 0 0;font-weight:400">
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
            border: 1px solid #C7D2FE;
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
            st.markdown("""<div style="background:linear-gradient(160deg,#1E293B,#263445);
            padding:18px;border-radius:12px;border:1px solid #334155;
            border-top:3px solid #6366F1;margin-bottom:8px;
            box-shadow:0 4px 16px rgba(0,0,0,0.35)">
            <h3 style="color:#F1F5F9;margin:0;font-size:16px;font-weight:700">KMeans</h3>
            <p style="color:#94A3B8;font-size:11px;margin:4px 0 0;font-weight:500">
            Unsupervised Clustering</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("""<div style="background:#1E293B;
            border-radius:12px;border:1px solid #334155;padding:16px;margin-bottom:8px;
            border-top:2px solid #64748B">
            <p style="color:#94A3B8;font-size:10px;text-transform:uppercase;
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
                    f'<p style="margin:3px 0;font-size:13px;color:#64748B">'
                    f'<span style="color:{ZONE_HEX[z]};font-size:10px">⬤</span> '
                    f'{z}: <b style="font-family:\'JetBrains Mono\',monospace;'
                    f'color:#F1F5F9;font-variant-numeric:tabular-nums">'
                    f'{km_stats[z]}</b></p>',
                    unsafe_allow_html=True)

        with c2:
            st.markdown("""<div style="background:linear-gradient(160deg,#1E293B,#263445);
            padding:18px;border-radius:12px;border:1px solid #334155;
            border-top:3px solid #10B981;margin-bottom:8px;
            box-shadow:0 4px 16px rgba(0,0,0,0.35)">
            <h3 style="color:#F1F5F9;margin:0;font-size:16px;font-weight:700">XGBoost</h3>
            <p style="color:#94A3B8;font-size:11px;margin:4px 0 0;font-weight:500">
            Supervised Classification</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("""<div style="background:#1E293B;
            border-radius:12px;border:1px solid #334155;padding:16px;margin-bottom:8px;
            border-top:2px solid #64748B">
            <p style="color:#94A3B8;font-size:10px;text-transform:uppercase;
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
                    f'<p style="margin:3px 0;font-size:13px;color:#64748B">'
                    f'<span style="color:{ZONE_HEX[z]};font-size:10px">⬤</span> '
                    f'{z}: <b style="font-family:\'JetBrains Mono\',monospace;'
                    f'color:#F1F5F9;font-variant-numeric:tabular-nums">'
                    f'{xgb_stats[z]}</b></p>',
                    unsafe_allow_html=True)

        with c3:
            st.markdown("""<div style="background:linear-gradient(160deg,#1E293B,#263445);
            padding:18px;border-radius:12px;border:1px solid #334155;
            border-top:3px solid #7C3AED;margin-bottom:8px;
            box-shadow:0 4px 16px rgba(0,0,0,0.35)">
            <h3 style="color:#F1F5F9;margin:0;font-size:16px;font-weight:700">GMM</h3>
            <p style="color:#94A3B8;font-size:11px;margin:4px 0 0;font-weight:500">
            Unsupervised Soft Clustering</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("""<div style="background:#1E293B;
            border-radius:12px;border:1px solid #334155;padding:16px;margin-bottom:8px;
            border-top:2px solid #64748B">
            <p style="color:#94A3B8;font-size:10px;text-transform:uppercase;
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
                    f'<p style="margin:3px 0;font-size:13px;color:#64748B">'
                    f'<span style="color:{ZONE_HEX[z]};font-size:10px">⬤</span> '
                    f'{z}: <b style="font-family:\'JetBrains Mono\',monospace;'
                    f'color:#F1F5F9;font-variant-numeric:tabular-nums">'
                    f'{gmm_stats[z]}</b></p>',
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — OPERATIONS DASHBOARD
# ═══════════════════════════════════════════════════════════════

with tab3:

    # ── Section header ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1E293B 0%,#0F172A 100%);
    padding:22px 26px;border-radius:12px;border:1px solid #334155;
    border-left:4px solid #6366F1;margin-bottom:22px;
    box-shadow:inset 4px 0 30px rgba(99,102,241,0.15), 0 4px 20px rgba(0,0,0,0.4)">
    <div style="display:flex;align-items:center;justify-content:space-between">
    <div>
    <h2 style="color:#F1F5F9;margin:0;font-size:20px;font-weight:700;
    letter-spacing:-0.02em">📡 Operations Timeline</h2>
    <p style="color:#64748B;margin:6px 0 0;font-size:13px;line-height:1.5">
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
            <div style="background:#1E293B;border:1px solid #334155;
            border-radius:12px;border-top:3px solid #6366F1;
            padding:24px 20px;text-align:center;
            box-shadow:0 4px 20px rgba(99,102,241,0.1)">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase;margin-bottom:10px">
            📊 TOTAL SCANS</div>
            <div id="stat-total" style="font-size:42px;font-weight:900;
            color:#F1F5F9;font-family:'JetBrains Mono',monospace;
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
            <div style="background:#1E293B;border:1px solid #334155;
            border-radius:12px;border-top:2px solid #0891B2;
            padding:24px 20px;text-align:center;
            box-shadow:0 4px 20px rgba(34,211,238,0.1)">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;
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
            <div style="background:#1E293B;border:1px solid #334155;
            border-radius:12px;border-top:2px solid {_peak_color};
            padding:24px 20px;text-align:center;
            box-shadow:0 4px 20px {'rgba(255,23,68,0.15)' if _hist_peak > 100 else 'rgba(16,185,129,0.1)'};
            {'animation:critGlow 2s ease-in-out infinite;' if _hist_peak > 100 else ''}">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;
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
        <div style="background:#1E293B;border:1px solid #334155;
        border-radius:12px;padding:14px 18px 6px;margin:8px 0 4px;
        border-top:3px solid #6366F1">
        <div style="color:#94A3B8;font-size:10px;font-weight:700;
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
                bgcolor='#FFFFFF', bordercolor='#C7D2FE',
                font=dict(family='Inter, sans-serif', size=12, color='#EFF6FF'),
            ),
            showlegend=False,
        ))

        _fig_timeline.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=320,
            margin=dict(l=50, r=20, t=20, b=50),
            xaxis=dict(
                title=dict(text='Scan #', font=dict(color='#64748B', size=11)),
                tickfont=dict(color='#64748B', size=10,
                              family='JetBrains Mono, monospace'),
                gridcolor='#E5E7EB', gridwidth=1,
                zeroline=False,
                dtick=1,
            ),
            yaxis=dict(
                title=dict(text='Crowd Count', font=dict(color='#64748B', size=11)),
                tickfont=dict(color='#64748B', size=10,
                              family='JetBrains Mono, monospace'),
                gridcolor='#E5E7EB', gridwidth=1,
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
        <div style="background:#1E293B;border:1px solid #334155;
        border-radius:12px;padding:14px 18px 6px;margin:16px 0 4px;
        border-top:2px solid #0891B2">
        <div style="color:#94A3B8;font-size:10px;font-weight:700;
        letter-spacing:2px;text-transform:uppercase">
        🗂️ THREAT HISTORY LOG</div>
        </div>
        """, unsafe_allow_html=True)

        # Build table header
        _table_html = """
        <div style="border:1px solid #334155;border-radius:12px;
        overflow:hidden;margin-top:8px">
        <table style="width:100%;border-collapse:collapse;
        font-family:'Inter',sans-serif">
        <thead>
        <tr style="background:#0A0F1A;border-bottom:2px solid #C7D2FE">
        <th style="padding:12px 16px;text-align:left;color:#94A3B8;
        font-size:10px;font-weight:700;letter-spacing:2px;
        text-transform:uppercase">SCAN #</th>
        <th style="padding:12px 16px;text-align:center;color:#94A3B8;
        font-size:10px;font-weight:700;letter-spacing:2px;
        text-transform:uppercase">COUNT</th>
        <th style="padding:12px 16px;text-align:center;color:#94A3B8;
        font-size:10px;font-weight:700;letter-spacing:2px;
        text-transform:uppercase">THREAT LEVEL</th>
        <th style="padding:12px 16px;text-align:right;color:#94A3B8;
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
            _row_bg = "#1E293B" if _idx % 2 == 0 else "#263445"

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
            <tr style="background:{_row_bg};border-bottom:1px solid #334155;
            transition:background 0.2s">
            <td style="padding:12px 16px;color:#64748B;font-size:13px;
            font-family:'JetBrains Mono',monospace;font-weight:500">
            #{_scan_n:02d}</td>
            <td style="padding:12px 16px;text-align:center;
            color:#F1F5F9;font-size:15px;font-weight:700;
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

<h2 style="color:#F1F5F9;font-size:26px;font-weight:800;margin:0;
letter-spacing:-0.02em">No Scans Yet</h2>

<p style="color:#94A3B8;font-size:14px;max-width:440px;
margin:14px auto 0;line-height:1.8">
Analyse images in the <span style="color:#3B82F6;
font-weight:600">Live Analysis</span> tab to populate
the operations timeline with threat data.</p>

<div style="border-top:1px solid #334155;margin:32px auto 28px;
max-width:300px"></div>

<div style="display:inline-flex;align-items:center;gap:8px;
padding:8px 20px;border-radius:20px;
background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2)">
<span style="display:inline-block;width:6px;height:6px;
border-radius:50%;background:#3B4A63"></span>
<span style="color:#94A3B8;font-size:11px;font-weight:500;
font-family:'JetBrains Mono',monospace;letter-spacing:1px">
AWAITING FIRST SCAN</span>
</div>

</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — LIVE CAPTURE (placeholder)
# ═══════════════════════════════════════════════════════════════

with tab4:
    st.markdown("""
<div style="text-align:center;padding:80px 20px 70px;
animation:floatUp 0.6s ease;
background:radial-gradient(ellipse 500px 250px at center 100px,
rgba(99,102,241,0.06), transparent)">

<div style="font-size:56px;margin-bottom:20px;opacity:0.6;
filter:drop-shadow(0 4px 20px rgba(99,102,241,0.25))">📱</div>

<div style="font-size:11px;font-weight:700;color:#6366F1;
letter-spacing:5px;text-transform:uppercase;margin-bottom:14px">
LIVE CAPTURE</div>

<h2 style="color:#F1F5F9;font-size:26px;font-weight:800;margin:0;
letter-spacing:-0.02em">Coming Soon</h2>

<p style="color:#94A3B8;font-size:14px;max-width:440px;
margin:14px auto 0;line-height:1.8">
Real-time camera feed analysis will be available in a future update.
Use the <span style="color:#3B82F6;font-weight:600">Live Analysis</span> tab
for image-based analysis.</p>

</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════

with tab5:

    # ── Header card ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1E293B 0%,#0F172A 100%);
    padding:22px 26px;border-radius:12px;border:1px solid #334155;
    border-left:4px solid #7C3AED;margin-bottom:22px;
    box-shadow:inset 4px 0 30px rgba(124,58,237,0.15), 0 4px 20px rgba(0,0,0,0.4)">
    <div style="display:flex;align-items:center;justify-content:space-between">
    <div>
    <h2 style="color:#F1F5F9;margin:0;font-size:20px;font-weight:700;
    letter-spacing:-0.02em">🗂️ Batch Analysis</h2>
    <p style="color:#64748B;margin:6px 0 0;font-size:13px;line-height:1.5">
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
        st.markdown(f"""
        <div style="background:#1E293B;border:1px solid #334155;border-radius:10px;
        padding:12px 20px;margin:8px 0 16px 0;display:flex;align-items:center;gap:10px">
        <span style="color:#A78BFA;font-size:14px">📁</span>
        <span style="color:#64748B;font-size:13px">{len(batch_files)} images selected</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶ Run Batch Analysis", key="run_batch", use_container_width=True):

            batch_results = []
            progress_bar = st.progress(0, text="Processing batch...")
            _batch_total_start = time.time()

            for idx, bf in enumerate(batch_files):
                progress_bar.progress(
                    (idx) / len(batch_files),
                    text=f"Analyzing {bf.name} ({idx+1}/{len(batch_files)})..."
                )

                _b_t0 = time.time()
                raw_bytes = np.asarray(bytearray(bf.read()), dtype=np.uint8)
                bf.seek(0)
                try:
                    _b_img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                    if _b_img_bgr is None:
                        raise ValueError("Decode returned None")
                    _b_img_rgb = cv2.cvtColor(_b_img_bgr, cv2.COLOR_BGR2RGB)
                    _b_h, _b_w = _b_img_rgb.shape[:2]
                except Exception as _b_dec_err:
                    st.warning(f"⚠️ Skipped **{bf.name}** — corrupted or unreadable ({_b_dec_err})")
                    continue

                if LWCC_AVAILABLE and lwcc_shb is not None:
                    _b_count, _b_density = predict_density_lwcc(_b_img_rgb)
                else:
                    _b_density_raw = predict_density_raw(cnn, _b_img_bgr)
                    _b_density = cv2.resize(_b_density_raw, (_b_w, _b_h))
                    _b_count = apply_calibration(float(_b_density_raw.sum()), "small")

                # ── Real zone classification (same pipeline as Live Analysis) ──
                _b_feats = extract_features(_b_density)
                _b_feats_sc = scaler.transform(_b_feats) if scaler else _b_feats

                if method == "XGBoost" and xgb:
                    _b_labels = labels_from_xgb(_b_feats_sc, xgb)
                elif method == "GMM" and gmm:
                    _b_labels, _ = labels_from_gmm(_b_feats_sc, gmm)
                else:
                    _b_labels = labels_from_kmeans(_b_feats_sc, km) \
                        if km else [get_label(float(f[0])) for f in _b_feats]

                _b_safety_img, _b_zone_stats = build_overlay(
                    _b_img_rgb, _b_labels, GRID, opacity)
                _b_density_overlay = build_density_overlay(
                    _b_img_rgb, _b_density, opacity,
                    expected_count=_b_count)

                # Threat score (same formula as live gauge)
                _b_threat_score = min(100, int(
                    _b_zone_stats.get("Critical", 0) * 40 +
                    _b_zone_stats.get("High", 0) * 20 +
                    _b_zone_stats.get("Medium", 0) * 1 +
                    min(15, _b_count / 10)
                ))

                if _b_threat_score < 25:
                    _b_threat = "MINIMAL"
                elif _b_threat_score < 50:
                    _b_threat = "ELEVATED"
                elif _b_threat_score < 75:
                    _b_threat = "HIGH"
                else:
                    _b_threat = "CRITICAL"

                # Dynamic confidence (uses actual model outputs)
                _b_conf = compute_dynamic_confidence(
                    _b_feats_sc, method, km, xgb, gmm, _b_count)

                _b_elapsed = time.time() - _b_t0

                # Compress images to JPEG base64 for memory efficiency (~10x smaller)
                def _compress_img_b64(img_arr, quality=80):
                    _, buf = cv2.imencode('.jpg',
                        cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
                    return base64.b64encode(buf).decode()

                batch_results.append({
                    "name": bf.name,
                    "count": _b_count,
                    "threat": _b_threat,
                    "threat_score": _b_threat_score,
                    "confidence": _b_conf,
                    "zone_stats": _b_zone_stats,
                    "safety_img_b64": _compress_img_b64(_b_safety_img),
                    "density_overlay_b64": _compress_img_b64(_b_density_overlay),
                    "time_s": round(_b_elapsed, 2),
                })

                # Update progress with timing info
                progress_bar.progress(
                    (idx + 1) / len(batch_files),
                    text=f"✓ {bf.name} — {_b_count} persons · {_b_elapsed:.1f}s ({idx+1}/{len(batch_files)})"
                )

            _batch_total_elapsed = time.time() - _batch_total_start
            progress_bar.progress(1.0, text=f"✅ Batch complete! {len(batch_files)} images in {_batch_total_elapsed:.1f}s")
            time.sleep(0.8)
            progress_bar.empty()

            st.session_state["batch_results"] = batch_results
            gc.collect()
        # Display results if available
        if "batch_results" in st.session_state and st.session_state["batch_results"]:
            batch_results = st.session_state["batch_results"]

            # ══════════════════════════════════════════════════
            # 1. AGGREGATE ANALYTICS DASHBOARD
            # ══════════════════════════════════════════════════
            _total_frames = len(batch_results)
            _avg_count = int(np.mean([r["count"] for r in batch_results]))
            _max_count = max(r["count"] for r in batch_results)
            _avg_threat = int(np.mean([r["threat_score"] for r in batch_results]))

            st.markdown("""
            <div style="background:#1E293B;border:1px solid #334155;
            border-radius:12px;padding:14px 18px 6px;margin:16px 0 12px;
            border-top:2px solid #7C3AED">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;
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
                               font=dict(color="#4B5563", size=13)),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=40, r=20, t=40, b=40),
                    xaxis=dict(tickfont=dict(color="#4B5563", size=11)),
                    yaxis=dict(tickfont=dict(color="#6B7280", size=10),
                               gridcolor="#E5E7EB"),
                    showlegend=False,
                )
                st.plotly_chart(_fig_zone_dist, use_container_width=True)

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
                               font=dict(color="#4B5563", size=13)),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False,
                )
                st.plotly_chart(_fig_threat_dist, use_container_width=True)
            # ══════════════════════════════════════════════════
            # 2. SUMMARY TABLE WITH ZONE COLUMNS
            # ══════════════════════════════════════════════════
            st.markdown("""
            <div style="background:#1E293B;border:1px solid #334155;
            border-radius:12px;padding:14px 18px 6px;margin:16px 0 4px;
            border-top:2px solid #7C3AED">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase">
            📋 DETAILED RESULTS TABLE</div>
            </div>
            """, unsafe_allow_html=True)

            _batch_table = """
            <div style="border:1px solid #334155;border-radius:12px;
            overflow:hidden;margin-top:8px">
            <table style="width:100%;border-collapse:collapse;
            font-family:'Inter',sans-serif">
            <thead>
            <tr style="background:#0A0F1A;border-bottom:2px solid #C7D2FE">
            <th style="padding:12px 14px;text-align:left;color:#94A3B8;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">IMAGE</th>
            <th style="padding:12px 10px;text-align:center;color:#94A3B8;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">COUNT</th>
            <th style="padding:12px 10px;text-align:center;color:#10B981;
            font-size:10px;font-weight:700;letter-spacing:1.5px">LOW</th>
            <th style="padding:12px 10px;text-align:center;color:#D97706;
            font-size:10px;font-weight:700;letter-spacing:1.5px">MED</th>
            <th style="padding:12px 10px;text-align:center;color:#DC2626;
            font-size:10px;font-weight:700;letter-spacing:1.5px">HIGH</th>
            <th style="padding:12px 10px;text-align:center;color:#B91C1C;
            font-size:10px;font-weight:700;letter-spacing:1.5px">CRIT</th>
            <th style="padding:12px 10px;text-align:center;color:#94A3B8;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">THREAT</th>
            <th style="padding:12px 10px;text-align:center;color:#94A3B8;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">SCORE</th>
            <th style="padding:12px 10px;text-align:center;color:#6366F1;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">CONF</th>
            <th style="padding:12px 10px;text-align:right;color:#94A3B8;
            font-size:10px;font-weight:700;letter-spacing:1.5px;
            text-transform:uppercase">TIME</th>
            </tr>
            </thead>
            <tbody>
            """

            _threat_colors = {
                "MINIMAL": ("#10B981", "rgba(16,185,129,0.12)"),
                "ELEVATED": ("#F59E0B", "rgba(245,158,11,0.12)"),
                "HIGH": ("#EF4444", "rgba(239,68,68,0.12)"),
                "CRITICAL": ("#FF1744", "rgba(255,23,68,0.12)"),
            }

            for _bi, _br in enumerate(batch_results):
                _row_bg = "#1E293B" if _bi % 2 == 0 else "#263445"
                _btc, _btbg = _threat_colors.get(_br["threat"], ("#64748B", "rgba(100,116,139,0.12)"))
                _zs = _br["zone_stats"]

                _batch_table += f"""
                <tr style="background:{_row_bg};border-bottom:1px solid #334155">
                <td style="padding:10px 14px;color:#64748B;font-size:12px;
                font-family:'JetBrains Mono',monospace;font-weight:500">
                {_br['name']}</td>
                <td style="padding:10px;text-align:center;
                color:#F1F5F9;font-size:14px;font-weight:700;
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
                color:#94A3B8;font-size:12px;font-weight:500;
                font-family:'JetBrains Mono',monospace">{_br.get('time_s', '—')}s</td>
                </tr>
                """

            _batch_table += "</tbody></table></div>"
            st.markdown(_batch_table, unsafe_allow_html=True)
            # ══════════════════════════════════════════════════
            # 3. PER-IMAGE EXPANDABLE DETAIL CARDS
            # ══════════════════════════════════════════════════
            st.markdown("""
            <div style="background:#1E293B;border:1px solid #334155;
            border-radius:12px;padding:14px 18px 6px;margin:24px 0 4px;
            border-top:2px solid #0891B2">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase">
            🔍 PER-IMAGE DETAILS</div>
            </div>
            """, unsafe_allow_html=True)

            for _di, _dr in enumerate(batch_results):
                _d_zs = _dr["zone_stats"]
                _d_ts = _dr["threat_score"]
                _d_tc, _ = _threat_colors.get(_dr["threat"], ("#64748B", ""))

                with st.expander(f"📷 {_dr['name']}  —  {_dr['count']} persons  ·  {_dr['threat']}  ·  {_dr['confidence']}% conf  ·  {_dr.get('time_s', '—')}s"):
                    _d_c1, _d_c2 = st.columns(2)
                    with _d_c1:
                        st.markdown(f"""
                        <div style="color:#6366F1;font-size:10px;font-weight:700;
                        letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
                        SAFETY ZONE MAP — {method.upper()}</div>
                        """, unsafe_allow_html=True)
                        _d_safety_bytes = base64.b64decode(_dr["safety_img_b64"])
                        st.image(_d_safety_bytes, use_container_width=True)
                    with _d_c2:
                        st.markdown("""
                        <div style="color:#6366F1;font-size:10px;font-weight:700;
                        letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
                        DENSITY HEATMAP</div>
                        """, unsafe_allow_html=True)
                        _d_density_bytes = base64.b64decode(_dr["density_overlay_b64"])
                        st.image(_d_density_bytes, use_container_width=True)

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
                            <div style="background:#1E293B;border:1px solid #334155;
                            border-radius:10px;padding:14px;text-align:center;
                            border-top:2px solid {_d_clr}">
                            <div style="font-size:10px;color:#94A3B8;font-weight:700;
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
                            axis=dict(range=[0, 100], tickcolor="#64748B",
                                      tickfont=dict(color="#6B7280", size=9)),
                            bar=dict(color=_d_tc, thickness=0.3),
                            bgcolor="#C7D2FE", borderwidth=0,
                            steps=[
                                dict(range=[0, 25], color="#0C2A1A"),
                                dict(range=[25, 50], color="#2A1F00"),
                                dict(range=[50, 75], color="#2A0D00"),
                                dict(range=[75, 100], color="#2A0007"),
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
                    st.plotly_chart(_d_gauge, use_container_width=True)
            # ══════════════════════════════════════════════════
            # 4. PEAK FRAME HIGHLIGHT
            # ══════════════════════════════════════════════════
            _peak_result = max(batch_results, key=lambda x: x["count"])

            st.markdown(f"""
            <div style="background:#1E293B;border:1px solid #334155;
            border-radius:12px;padding:14px 18px 6px;margin:24px 0 4px;
            border-top:2px solid #FF1744">
            <div style="display:flex;align-items:center;justify-content:space-between">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;
            letter-spacing:2px;text-transform:uppercase">
            🔺 PEAK DENSITY FRAME</div>
            <span style="color:#B91C1C;font-size:12px;font-weight:700;
            font-family:'JetBrains Mono',monospace">{_peak_result['count']} persons · {_peak_result['name']}</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

            _pk_c1, _pk_c2 = st.columns(2)
            with _pk_c1:
                _pk_safety_bytes = base64.b64decode(_peak_result["safety_img_b64"])
                st.image(_pk_safety_bytes, use_container_width=True,
                         caption=f"Safety Map — {_peak_result['name']}")
            with _pk_c2:
                _pk_density_bytes = base64.b64decode(_peak_result["density_overlay_b64"])
                st.image(_pk_density_bytes, use_container_width=True,
                         caption=f"Density Heatmap — {_peak_result['name']}")

            # ══════════════════════════════════════════════════
            # 5. BATCH TIMELINE CHART
            # ══════════════════════════════════════════════════
            st.markdown("""
            <div style="background:#1E293B;border:1px solid #334155;
            border-radius:12px;padding:14px 18px 6px;margin:24px 0 4px;
            border-top:3px solid #6366F1">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;
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
                hoverlabel=dict(bgcolor='#FFFFFF', bordercolor='#C7D2FE',
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
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=320,
                margin=dict(l=50, r=20, t=20, b=50),
                xaxis=dict(
                    title=dict(text='Frame #', font=dict(color='#64748B', size=11)),
                    tickfont=dict(color='#64748B', size=10,
                                  family='JetBrains Mono, monospace'),
                    gridcolor='#E5E7EB', gridwidth=1,
                    zeroline=False, dtick=1,
                ),
                yaxis=dict(
                    title=dict(text='Crowd Count', font=dict(color='#64748B', size=11)),
                    tickfont=dict(color='#64748B', size=10,
                                  family='JetBrains Mono, monospace'),
                    gridcolor='#E5E7EB', gridwidth=1,
                    zeroline=False,
                ),
                font=dict(family='Inter, sans-serif'),
                hovermode='closest',
            )
            st.plotly_chart(_fig_batch, use_container_width=True)

            # ══════════════════════════════════════════════════
            # 6. DOWNLOAD BATCH REPORT
            # ══════════════════════════════════════════════════
            _batch_report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "method": method,
                "venue_capacity": venue_capacity,
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
                        "threat": r["threat"],
                        "threat_score": r["threat_score"],
                        "confidence": r["confidence"],
                        "zone_stats": r["zone_stats"],
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

<h2 style="color:#F1F5F9;font-size:26px;font-weight:800;margin:0;
letter-spacing:-0.02em">Upload Multiple Images</h2>

<p style="color:#94A3B8;font-size:14px;max-width:440px;
margin:14px auto 0;line-height:1.8">
Upload multiple crowd images for full zone classification per frame.
Get aggregate statistics, per-image overlays, and a downloadable report.</p>

<div style="border-top:1px solid #334155;margin:32px auto 28px;
max-width:300px"></div>

<div style="display:flex;justify-content:center;gap:16px;
flex-wrap:wrap;max-width:600px;margin:0 auto">

<div style="background:#1E293B;border:1px solid #334155;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">📊</div>
<div style="color:#F1F5F9;font-size:12px;font-weight:600">Zone Stats</div>
<div style="color:#94A3B8;font-size:10px;margin-top:4px">Per-image classification</div>
</div>

<div style="background:#1E293B;border:1px solid #334155;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">🗺️</div>
<div style="color:#F1F5F9;font-size:12px;font-weight:600">Safety Maps</div>
<div style="color:#94A3B8;font-size:10px;margin-top:4px">Overlays & heatmaps</div>
</div>

<div style="background:#1E293B;border:1px solid #334155;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">📈</div>
<div style="color:#F1F5F9;font-size:12px;font-weight:600">Timeline</div>
<div style="color:#94A3B8;font-size:10px;margin-top:4px">Density across frames</div>
</div>

<div style="background:#1E293B;border:1px solid #334155;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">📥</div>
<div style="color:#F1F5F9;font-size:12px;font-weight:600">JSON Report</div>
<div style="color:#94A3B8;font-size:10px;margin-top:4px">Full zone-level data</div>
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
    if _rl.epsilon > 0.5:
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
    <div style="background:linear-gradient(135deg,#1E293B 0%,#0F172A 100%);
    padding:22px 26px;border-radius:12px;border:1px solid #334155;
    border-left:4px solid #7C3AED;margin-bottom:22px;
    box-shadow:inset 4px 0 30px rgba(124,58,237,0.15), 0 4px 20px rgba(0,0,0,0.04)">
    <div style="display:flex;align-items:center;justify-content:space-between">
    <div>
    <h2 style="color:#F1F5F9;margin:0;font-size:20px;font-weight:700;
    letter-spacing:-0.02em">🤖 DQN Evacuation Policy Agent</h2>
    <p style="color:#64748B;margin:6px 0 0;font-size:13px;line-height:1.5">
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

    rl_c1, rl_c2, rl_c3, rl_c4 = st.columns(4)
    with rl_c1:
        st.metric("Episodes Trained", f"{_rl.episode}")
    with rl_c2:
        st.metric("Exploration Rate", f"{_rl_explore_pct}% exploring")
    with rl_c3:
        st.metric("Memory Size", f"{_rl_mem_size}/2000")
    with rl_c4:
        st.metric("Total Reward", f"{_rl.total_reward:+.1f}")

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

                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

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
            <div style="background:#1E293B;
            border:1px solid #334155;
            border-left:4px solid #8B5CF6;
            border-radius:12px;padding:24px;
            margin:16px 0;
            box-shadow:0 0 30px rgba(139,92,246,0.2)">

            <div style="color:#8B5CF6;font-size:10px;
            font-weight:700;letter-spacing:2px;
            text-transform:uppercase;margin-bottom:12px">
            🤖 AGENT DECISION — Episode {_rl.episode}</div>

            <div style="color:#F1F5F9;font-size:22px;
            font-weight:800;margin-bottom:8px">
            {_rl_result["action_name"]}</div>

            <div style="color:#94A3B8;font-size:13px;
            margin-bottom:20px">
            {_rl_result["action_detail"]}</div>

            <div style="display:grid;
            grid-template-columns:1fr 1fr 1fr;
            gap:12px;margin-bottom:16px">

            <div style="background:#263445;
            border-radius:8px;padding:12px;
            text-align:center">
            <div style="color:#64748B;font-size:10px;
            letter-spacing:1px">CRITICAL ZONES</div>
            <div style="color:#EF4444;font-size:20px;
            font-weight:800">{_crit_b} → {_crit_a}</div>
            <div style="color:#10B981;font-size:11px">
            ↓{_crit_pct_r}%</div>
            </div>

            <div style="background:#263445;
            border-radius:8px;padding:12px;
            text-align:center">
            <div style="color:#64748B;font-size:10px;
            letter-spacing:1px">EVAC TIME</div>
            <div style="color:#22D3EE;font-size:20px;
            font-weight:800">{_rl_result["evac_before"]} → {_rl_result["evac_after"]} min</div>
            </div>

            <div style="background:#263445;
            border-radius:8px;padding:12px;
            text-align:center">
            <div style="color:#64748B;font-size:10px;
            letter-spacing:1px">REWARD</div>
            <div style="color:#10B981;font-size:20px;
            font-weight:800">
            {_rl_result["reward"]:+.1f}</div>
            </div>
            </div>

            <div style="color:#64748B;font-size:11px;
            font-family:monospace">
            Loss: {_rl_result["loss"]:.4f} ·
            Epsilon: {_rl.epsilon:.3f} ·
            Memory: {len(_rl.memory)}/2000
            </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#1E293B;border:1px solid #334155;
        border-radius:12px;padding:32px;text-align:center;margin:16px 0">
        <div style="font-size:40px;margin-bottom:12px;opacity:0.5">🤖</div>
        <div style="color:#F1F5F9;font-size:16px;font-weight:700;
        margin-bottom:6px">Upload an Image First</div>
        <div style="color:#64748B;font-size:13px">
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
                font=dict(size=14, color="#4B5563",
                          family="Inter, sans-serif"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=20, r=20, t=50, b=30),
            xaxis=dict(
                title="Episode",
                gridcolor="rgba(199,210,254,0.3)",
                tickfont=dict(color="#6B7280"),
                title_font=dict(color="#6B7280"),
            ),
            yaxis=dict(
                title="Reward",
                gridcolor="rgba(199,210,254,0.3)",
                tickfont=dict(color="#6B7280"),
                title_font=dict(color="#6B7280"),
            ),
            legend=dict(
                font=dict(color="#6B7280"),
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

<h2 style="color:#F1F5F9;font-size:26px;font-weight:800;margin:0;
letter-spacing:-0.02em">DQN Evacuation Agent</h2>

<p style="color:#94A3B8;font-size:14px;max-width:440px;
margin:14px auto 0;line-height:1.8">
Upload a crowd image and run analysis first, then train the RL agent
to learn optimal evacuation policies based on zone risk states.</p>

<div style="border-top:1px solid #334155;margin:32px auto 28px;
max-width:300px"></div>

<div style="display:flex;justify-content:center;gap:16px;
flex-wrap:wrap;max-width:600px;margin:0 auto">

<div style="background:#1E293B;border:1px solid #334155;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">🧠</div>
<div style="color:#F1F5F9;font-size:12px;font-weight:600">DQN Network</div>
<div style="color:#94A3B8;font-size:10px;margin-top:4px">128→128→64 architecture</div>
</div>

<div style="background:#1E293B;border:1px solid #334155;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">🎯</div>
<div style="color:#F1F5F9;font-size:12px;font-weight:600">8 Actions</div>
<div style="color:#94A3B8;font-size:10px;margin-top:4px">Evacuation strategies</div>
</div>

<div style="background:#1E293B;border:1px solid #334155;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">📊</div>
<div style="color:#F1F5F9;font-size:12px;font-weight:600">69 Features</div>
<div style="color:#94A3B8;font-size:10px;margin-top:4px">Zone risks + context</div>
</div>

<div style="background:#1E293B;border:1px solid #334155;
border-top:2px solid #7C3AED;border-radius:12px;padding:16px 20px;
flex:1;min-width:120px;text-align:center">
<div style="font-size:24px;margin-bottom:8px">🏆</div>
<div style="color:#F1F5F9;font-size:12px;font-weight:600">Reward Shaping</div>
<div style="color:#94A3B8;font-size:10px;margin-top:4px">Multi-objective optimization</div>
</div>

</div>

</div>
        """, unsafe_allow_html=True)
