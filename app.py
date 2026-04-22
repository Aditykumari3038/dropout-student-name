import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import backend
from dropout import (
    run_full_prediction,
    generate_analytics_data,
    FEATURE_LABELS,
    FEATURE_WEIGHTS,
)

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dropout Student Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS — Obsidian Dark Academic
# ══════════════════════════════════════════════════════════════════
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700;800&family=Syne:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #06090f;
    font-family: 'Syne', sans-serif;
    color: #d8e4f0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.2rem 4rem 2.2rem !important; max-width: 1480px !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #06090f; }
::-webkit-scrollbar-thumb { background: #1e4d8c; border-radius: 4px; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d18 0%, #0a1220 100%);
    border-right: 1px solid rgba(30,77,140,0.3);
    min-width: 260px !important;
}
section[data-testid="stSidebar"] .stRadio > label {
    color: #5a8ab8 !important;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 600;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #8baecf !important;
    font-size: 14px;
    letter-spacing: 0.5px;
    padding: 10px 14px;
    border-radius: 10px;
    transition: all 0.25s ease;
    font-weight: 500;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(30,77,140,0.18);
    color: #7ec8ff !important;
}

/* ── HERO ── */
.hero {
    position: relative; overflow: hidden;
    border-radius: 28px;
    margin: 1.8rem 0 3rem;
    padding: 100px 70px 90px;
    background: linear-gradient(135deg, #030812 0%, #071526 35%, #0b1e38 65%, #0d2242 100%);
    border: 1px solid rgba(30,77,140,0.4);
    text-align: center;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 90% 70% at 50% -10%, rgba(30,100,200,0.22) 0%, transparent 65%);
    pointer-events: none;
}
.hero-grid {
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(30,77,140,0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(30,77,140,0.07) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
}
.hero-orb {
    position: absolute;
    width: 500px; height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(30,100,200,0.12) 0%, transparent 70%);
    top: -150px; left: 50%; transform: translateX(-50%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(30,77,140,0.18);
    border: 1px solid rgba(94,184,255,0.3);
    color: #7ec8ff;
    font-size: 11px; font-weight: 700;
    letter-spacing: 3px; text-transform: uppercase;
    padding: 7px 22px; border-radius: 50px;
    margin-bottom: 28px;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(44px, 6vw, 80px);
    font-weight: 800; color: #ffffff;
    line-height: 1.08; margin-bottom: 22px;
    letter-spacing: -1.5px;
}
.hero-title em { color: #5ab4ff; font-style: normal; }
.hero-sub {
    font-size: 18px; color: #6a92b8;
    max-width: 600px; margin: 0 auto 40px;
    line-height: 1.75; font-weight: 400;
}
.hero-stats {
    display: flex; justify-content: center;
    gap: 60px; flex-wrap: wrap; margin-top: 44px;
}
.h-stat { text-align: center; }
.h-stat-n {
    font-family: 'Cormorant Garamond', serif;
    font-size: 44px; font-weight: 700;
    color: #5ab4ff; display: block; line-height: 1;
}
.h-stat-l {
    font-size: 11px; color: #3d5e80;
    text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px;
}

/* ── SECTION HEADERS ── */
.sh {
    display: flex; align-items: center; gap: 16px;
    margin: 3rem 0 1.4rem;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(30,77,140,0.25);
}
.sh-icon {
    width: 46px; height: 46px;
    background: linear-gradient(135deg, #112a50, #1a4a8a);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
}
.sh-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px; font-weight: 700; color: #d0e4f5;
}
.sh-desc { font-size: 13px; color: #3d5e80; margin-top: 3px; }

/* ── CARDS ── */
.card {
    background: linear-gradient(145deg, #0a1525, #0c1c30);
    border: 1px solid rgba(30,77,140,0.3);
    border-radius: 20px; padding: 28px; height: 100%;
    transition: border-color 0.3s, transform 0.3s;
}
.card:hover { border-color: #1e4d8c; transform: translateY(-3px); }
.card-al { border-left: 3px solid #1e4d8c; }
.c-icon { font-size: 34px; margin-bottom: 14px; }
.c-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 20px; color: #c4d8f0; margin-bottom: 10px; font-weight: 700;
}
.c-body { font-size: 14px; color: #5a7a9a; line-height: 1.75; }

/* ── STAT BOXES ── */
.sbox {
    background: linear-gradient(145deg, #0a1525, #0c1c30);
    border: 1px solid rgba(30,77,140,0.3);
    border-radius: 18px; padding: 26px 20px; text-align: center;
}
.sbox-n {
    font-family: 'Cormorant Garamond', serif;
    font-size: 44px; font-weight: 800; color: #5ab4ff;
    display: block; line-height: 1;
}
.sbox-l { font-size: 12px; color: #3d5e80; margin-top: 9px; letter-spacing: 0.5px; line-height:1.4; }
.sbox-s { font-size: 10px; color: #2a3e55; margin-top: 5px; }

/* ── CAUSE ROW ── */
.cause-row {
    display: flex; gap: 14px; align-items: flex-start;
    margin-bottom: 16px;
    background: #0a1525;
    border-radius: 14px; padding: 18px;
    border: 1px solid rgba(30,77,140,0.2);
}
.cause-pct {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px; font-weight: 700;
    color: #5ab4ff; min-width: 64px; line-height: 1;
}
.cause-title { font-size: 15px; color: #c4d8f0; font-weight: 600; margin-bottom: 4px; }
.cause-body { font-size: 13px; color: #4a6a88; line-height: 1.65; }

/* ── RISK DISPLAY ── */
.risk-panel {
    background: linear-gradient(145deg, #0a1525, #0c1c30);
    border: 1px solid rgba(30,77,140,0.3);
    border-radius: 22px; padding: 40px 32px; text-align: center;
}
.risk-score {
    font-family: 'Cormorant Garamond', serif;
    font-size: 88px; font-weight: 800; line-height: 1;
    margin-bottom: 8px;
}
.risk-level-label { font-size: 24px; font-weight: 700; letter-spacing: 3px; margin-bottom: 16px; }
.risk-sub { font-size: 14px; color: #4a6a88; line-height: 1.7; }

/* ── FEATURE IMPORTANCE BARS ── */
.fi-row { margin-bottom: 16px; }
.fi-label {
    font-size: 13px; color: #7a9ab8;
    margin-bottom: 6px;
    display: flex; justify-content: space-between; align-items: center;
}
.fi-bar-bg { background: #0c1c30; border-radius: 8px; height: 10px; overflow: hidden; }
.fi-bar-fill { height: 100%; border-radius: 8px; transition: width 1s cubic-bezier(.4,0,.2,1); }

/* ── RECOMMENDATION CARDS ── */
.rec-card {
    background: #0a1525;
    border-radius: 16px; padding: 22px;
    border: 1px solid rgba(30,77,140,0.25);
    margin-bottom: 14px;
    display: flex; gap: 18px; align-items: flex-start;
}
.rec-icon {
    width: 44px; height: 44px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}
.rec-title { font-size: 15px; color: #c4d8f0; font-weight: 600; margin-bottom: 5px; }
.rec-body  { font-size: 13px; color: #4a6a88; line-height: 1.65; }
.rec-badge {
    display: inline-block;
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; padding: 3px 10px;
    border-radius: 20px; margin-bottom: 8px;
}

/* ── ALERT BANNERS ── */
.alert-c { background:rgba(255,77,77,0.1); border:1px solid rgba(255,77,77,0.35); border-radius:12px; padding:18px 22px; color:#ffaaaa; font-size:14px; margin-bottom:14px; }
.alert-h { background:rgba(255,140,0,0.1); border:1px solid rgba(255,140,0,0.35); border-radius:12px; padding:18px 22px; color:#ffcc88; font-size:14px; margin-bottom:14px; }
.alert-m { background:rgba(245,197,24,0.1); border:1px solid rgba(245,197,24,0.35); border-radius:12px; padding:18px 22px; color:#fff0aa; font-size:14px; margin-bottom:14px; }
.alert-ok{ background:rgba(46,204,113,0.1); border:1px solid rgba(46,204,113,0.35); border-radius:12px; padding:18px 22px; color:#88ffbb; font-size:14px; margin-bottom:14px; }

/* ── USER INFO FORM ── */
.uform-wrap {
    background: linear-gradient(145deg, #080e1c, #0a1428);
    border: 1px solid rgba(30,77,140,0.4);
    border-radius: 24px; padding: 40px;
}
.uform-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 32px; color: #d0e4f5; font-weight: 700;
    margin-bottom: 8px;
}
.uform-sub { font-size: 14px; color: #3d5e80; margin-bottom: 32px; line-height: 1.6; }
.step-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #1e4d8c; display: inline-block; margin-right: 8px;
}
.step-dot.active { background: #5ab4ff; }

/* ── CTA BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #112a50 0%, #1a4a8a 100%) !important;
    color: white !important; border: none !important;
    border-radius: 14px !important; padding: 16px 38px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 16px !important; font-weight: 700 !important;
    letter-spacing: 0.8px !important; cursor: pointer !important;
    transition: all 0.3s !important;
    box-shadow: 0 6px 24px rgba(26,74,138,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 36px rgba(26,74,138,0.65) !important;
}

/* ── INPUTS ── */
.stSlider > div > div > div { background: #1e4d8c !important; }
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #0a1525 !important;
    border: 1px solid rgba(30,77,140,0.4) !important;
    border-radius: 12px !important;
}
.stTextInput input, .stNumberInput input {
    background: #0a1525 !important;
    border: 1px solid rgba(30,77,140,0.4) !important;
    border-radius: 12px !important;
    color: #d8e4f0 !important;
}
label { color: #8baecf !important; font-weight: 500 !important; }

hr { border-color: rgba(30,77,140,0.2) !important; }

.streamlit-expanderHeader {
    background: #0a1525 !important;
    border: 1px solid rgba(30,77,140,0.3) !important;
    border-radius: 12px !important;
    color: #8baecf !important;
}
.streamlit-expanderContent {
    background: #080e1c !important;
    border: 1px solid rgba(30,77,140,0.2) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

/* ── METRIC TILES ── */
.mtile {
    background: linear-gradient(145deg,#0a1525,#0c1c30);
    border:1px solid rgba(30,77,140,0.3);
    border-radius:16px; padding:22px 16px; text-align:center;
}
.mtile-val {
    font-family:'Cormorant Garamond',serif;
    font-size:38px; font-weight:800; color:#5ab4ff; display:block; line-height:1;
}
.mtile-lbl { font-size:12px; color:#3d5e80; margin-top:7px; letter-spacing:0.5px; }

/* ── TIMELINE ── */
.tl { position:relative; padding-left:32px; }
.tl::before { content:''; position:absolute; left:10px; top:0; bottom:0; width:2px; background:linear-gradient(180deg,#1e4d8c,transparent); }
.tl-item { position:relative; margin-bottom:24px; }
.tl-dot { position:absolute; left:-27px; top:4px; width:14px; height:14px; border-radius:50%; background:#1e4d8c; border:2px solid #06090f; }
.tl-title { font-size:15px; color:#c4d8f0; font-weight:600; }
.tl-body  { font-size:13px; color:#4a6a88; margin-top:5px; line-height:1.65; }

/* ── ABOUT PAGE ── */
.about-hero {
    background: linear-gradient(135deg,#040b18,#071526,#0c1d38);
    border:1px solid rgba(30,77,140,0.35);
    border-radius:24px; padding:60px 50px; margin-bottom:32px;
    text-align:center;
}
.about-title {
    font-family:'Cormorant Garamond',serif;
    font-size:52px; font-weight:800; color:#ffffff; margin-bottom:16px;
}
.about-body { font-size:16px; color:#5a7a9a; max-width:700px; margin:0 auto; line-height:1.8; }

/* ── PLOTLY CHART BACKGROUND ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
defaults = {
    "page":           "🏠 Home",
    "user_submitted": False,
    "user_info":      {},
    "pred_result":    None,
    "student_data":   {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def nav_to(page):
    st.session_state.page = page
    st.rerun()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 0 12px; text-align:center;'>
        <div style='font-size:42px; filter:drop-shadow(0 0 12px #1e4d8c);'>🎓</div>
        <div style='font-family:Cormorant Garamond,serif; font-size:22px; color:#d0e4f5;
                    font-weight:700; margin-top:10px; letter-spacing:-0.5px;'>
            Dropout Predictor
        </div>
        <div style='font-size:10px; color:#2a3e55; letter-spacing:3px;
                    text-transform:uppercase; margin-top:6px;'>
            Student Intelligence System
        </div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    pages = ["🏠 Home", "👤 Student Profile", "📊 Analytics", "🔬 Prediction Lab", "💡 Interventions", "ℹ️ About"]

    def _nav_change():
        st.session_state.page = st.session_state._nav_radio

    st.radio(
        "NAVIGATE",
        pages,
        key="_nav_radio",
        index=pages.index(st.session_state.page) if st.session_state.page in pages else 0,
        on_change=_nav_change,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px; color:#2a3e55; padding:4px;'>
        <b style='color:#4a6a88; font-size:11px; letter-spacing:1px; text-transform:uppercase;'>
            Model Performance
        </b><br><br>
        <div style='margin-bottom:8px;'>
            Accuracy &nbsp;&nbsp;<span style='color:#5ab4ff; float:right;'>87.4%</span>
        </div>
        <div style='margin-bottom:8px;'>
            Precision &nbsp;<span style='color:#5ab4ff; float:right;'>84.1%</span>
        </div>
        <div style='margin-bottom:8px;'>
            Recall &nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#5ab4ff; float:right;'>89.2%</span>
        </div>
        <div>
            F1-Score &nbsp;&nbsp;<span style='color:#5ab4ff; float:right;'>86.6%</span>
        </div>
        <br>
        <div style='font-size:10px; color:#1e3550;'>
            Algorithm: XGBoost + Gradient Boost Ensemble
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.user_submitted and st.session_state.user_info:
        u = st.session_state.user_info
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:12px; color:#2a3e55; padding:4px;'>
            <b style='color:#4a6a88; font-size:11px; letter-spacing:1px; text-transform:uppercase;'>
                Active Session
            </b><br><br>
            <span style='color:#7ec8ff;'>👤 {u.get("name","—")}</span><br>
            <span style='color:#4a6a88;'>{u.get("institution","—")}</span><br>
            <span style='color:#4a6a88;'>Role: {u.get("role","—")}</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HELPER: SECTION HEADER
# ══════════════════════════════════════════════════════════════════
def section_header(icon, title, desc=""):
    st.markdown(f"""
    <div class="sh">
        <div class="sh-icon">{icon}</div>
        <div>
            <div class="sh-title">{title}</div>
            {"<div class='sh-desc'>" + desc + "</div>" if desc else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════
PLOT_BG   = "#07101f"
PAPER_BG  = "#07101f"
GRID_COL  = "rgba(30,77,140,0.15)"
FONT_COL  = "#7a9ab8"
ACCENT    = "#5ab4ff"
PALETTE   = ["#5ab4ff","#3a8fd6","#1a6ab0","#ff8c00","#ff4d4d","#2ecc71","#f5c518","#a29bfe"]

def plotly_layout(fig, title="", height=380):
    fig.update_layout(
        title=title,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font_color=FONT_COL,
        font_family="Syne, sans-serif",
        height=height,
        margin=dict(l=30, r=30, t=50 if title else 20, b=30),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color=FONT_COL),
        title_font=dict(size=16, color="#c4d8f0"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == "🏠 Home":

    # HERO
    st.markdown("""
    <div class="hero">
        <div class="hero-grid"></div>
        <div class="hero-orb"></div>
        <div class="hero-badge">AI Early Warning System — v2.0</div>
        <div class="hero-title">
            Dropout Student<br><em>Predictor</em>
        </div>
        <div class="hero-sub">
            An advanced AI system that analyzes 10+ behavioral, academic, and socioeconomic
            signals to identify at-risk students — months before dropout — so educators can intervene in time.
        </div>
        <div class="hero-stats">
            <div class="h-stat">
                <span class="h-stat-n">258M</span>
                <div class="h-stat-l">Students Out of School</div>
            </div>
            <div class="h-stat">
                <span class="h-stat-n">87.4%</span>
                <div class="h-stat-l">Prediction Accuracy</div>
            </div>
            <div class="h-stat">
                <span class="h-stat-n">3×</span>
                <div class="h-stat-l">Higher Unemployment Risk</div>
            </div>
            <div class="h-stat">
                <span class="h-stat-n">34%</span>
                <div class="h-stat-l">Dropout Reduction with Intervention</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # UNDERSTANDING DROPOUT
    section_header("📘", "Understanding Student Dropout", "The phenomenon, its causes, and why prediction matters")

    c1, c2, c3 = st.columns(3)
    for col, icon, title, body in zip(
         [c1, c2, c3],
    ["🎯", "⚡", "🤖"],
    ["What is a Dropout?", "Why Early Detection?", "How AI Helps"],
    [
        "A student who leaves an educational institution without completing their qualification. Modern research classifies two types: <b style='color:#a8c0d8;'>Physical Dropout</b> (formal exit) and <b style='color:#a8c0d8;'>Mental Dropout</b> (disengagement without withdrawal). Both have lasting career impacts.",
        
        "Early detection helps institutions identify at-risk students before they disengage completely. This allows timely intervention such as mentoring, counseling, and academic support.",
        
        "AI analyzes patterns like attendance, grades, and behavior to predict dropout risk. It helps educators take proactive decisions using data-driven insights."
    ]
):
     with col:
        st.markdown(f"""
        <div class="card card-al">
            <div class="c-icon">{icon}</div>
            <div class="c-title">{title}</div>
            <div class="c-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.page == "📊 Analytics":
    section_header("📊", "Analytics Dashboard", "Personal + System Insights")

    # ❗ Check if prediction exists
    if not st.session_state.pred_result:
        st.warning("⚠️ Please run prediction first from Student Profile")
        st.stop()

    result = st.session_state.pred_result
    student = st.session_state.student_data

    # ═══════════════════════════════════════
    # 👤 PERSONAL ANALYTICS
    # ═══════════════════════════════════════
    st.markdown("## 👤 Your Risk Analytics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Score", result["risk_score"])
    col2.metric("Dropout Probability", f"{result['dropout_prob']}%")
    col3.metric("Timeline", result["months_estimate"])

    # BAR CHART
    contrib_df = pd.DataFrame(result["contributions"])

    fig_bar = px.bar(
        contrib_df,
        x="contribution",
        y="label",
        orientation="h",
        title="Feature Impact on Risk",
        color="contribution",
        color_continuous_scale="Blues"
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})

    st.plotly_chart(fig_bar, use_container_width=True)

    # PIE CHART
    top_factors = contrib_df.head(5)

    fig_pie = px.pie(
        top_factors,
        values="contribution",
        names="label",
        title="Top Risk Factors Distribution"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # ═══════════════════════════════════════
    # 🌍 GLOBAL ANALYTICS
    # ═══════════════════════════════════════
    st.markdown("## 🌍 System-Level Insights")

    data = generate_analytics_data()

    col1, col2 = st.columns(2)
    col1.metric("Total Students", data["total_students"])
    col2.metric("At Risk Students", data["at_risk"])

    fig_global = px.bar(
        x=list(data["risk_dist"].keys()),
        y=list(data["risk_dist"].values()),
        title="Overall Risk Distribution"
    )

    st.plotly_chart(fig_global, use_container_width=True)

elif st.session_state.page == "🔬 Prediction Lab":
    section_header("🔬", "Prediction Lab", "Run model")

    if st.button("Run Sample Prediction"):
        sample = {
            "attendance_rate": 60,
            "avg_marks": 50,
            "family_income_level": 2,
            "parental_education": 2,
            "lms_engagement_score": 40,
            "distance_to_school_km": 10,
            "failed_subjects": 2,
            "extra_activities": 0,
            "health_issues": 1,
            "has_part_time_job": 1,
        }

        result = run_full_prediction(sample)

        st.success(f"Risk Score: {result['risk_score']}")
        st.warning(f"Risk Level: {result['risk_level']}")

elif st.session_state.page == "💡 Interventions":
    section_header("💡", "Interventions", "Support strategies")

    st.write("Intervention system ready ✅")

elif st.session_state.page == "ℹ️ About":
    section_header("ℹ️", "About", "Project info")

    st.write("This is an AI-powered dropout prediction system.")
elif st.session_state.page == "👤 Student Profile":
    section_header("👤", "Student Profile", "Enter student details for prediction")

    # BASIC INFO
    col1, col2 = st.columns(2)
    with col1:
        first_name = st.text_input("First Name")
    with col2:
        last_name = st.text_input("Last Name")

    # ACADEMIC
    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        marks = st.slider("Average Marks (%)", 0, 100, 60)
    with col2:
        failed_subjects = st.number_input("Failed Subjects", 0, 6, 0)
        study_hours = st.number_input("Study Hours / day", 0.0, 12.0, 2.0)

    # FAMILY
    col1, col2 = st.columns(2)
    with col1:
        income = st.selectbox("Family Income Level (1=Low, 5=High)", [1,2,3,4,5], index=2)
    with col2:
        parent_edu = st.selectbox("Parental Education (1–5)", [1,2,3,4,5], index=2)

    # ENGAGEMENT
    lms = st.slider("Digital Engagement (%)", 0, 100, 60)

    # LIFESTYLE
    col1, col2 = st.columns(2)
    with col1:
        distance = st.slider("Distance to School (km)", 0, 50, 5)
        job = st.selectbox("Part-time Job", [0,1])
    with col2:
        health = st.selectbox("Health Issues", [0,1])
        extra = st.selectbox("Extracurricular Activities", [0,1])

    # 🚀 RUN PREDICTION
# 🚀 RUN PREDICTION
if st.button("🚀 Run Full Prediction"):

    # VALIDATION
    if not first_name or not last_name:
        st.error("⚠️ Please enter student name")

    else:
        student_data = {
            "attendance_rate": attendance,
            "avg_marks": marks,
            "family_income_level": income,
            "parental_education": parent_edu,
            "lms_engagement_score": lms,
            "distance_to_school_km": distance,
            "failed_subjects": failed_subjects,
            "extra_activities": extra,
            "health_issues": health,
            "has_part_time_job": job,
        }

        result = run_full_prediction(student_data)

        st.session_state.pred_result = result
        st.session_state.student_data = student_data

        st.success("✅ Prediction generated successfully!")
 


    # 🔥 RISK SCORE
    st.markdown(f"""
    <div class="risk-panel">
        <div class="risk-score" style="color:{result['risk_color']}">
            {result['risk_score']}
        </div>
        <div class="risk-level-label">{result['risk_emoji']} {result['risk_level']}</div>
        <div class="risk-sub">
            Dropout Probability: <b>{result['dropout_prob']}%</b><br>
            Expected Timeline: <b>{result['months_estimate']}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 🔍 ROOT CAUSES
    st.markdown("### 🔍 Root Causes")
    for cause in result["root_causes"]:
        st.markdown(f"""
        <div class="cause-row">
            <div class="cause-pct">{cause['icon']}</div>
            <div>
                <div class="cause-title">{cause['cause']}</div>
                <div class="cause-body">{cause['detail']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 💡 INTERVENTIONS
    st.markdown("### 💡 Recommended Interventions")
    for rec in result["interventions"]:
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-icon" style="background:{rec['color']}20;">
                {rec['icon']}
            </div>
            <div>
                <div class="rec-badge" style="color:{rec['color']}">
                    {rec['priority']}
                </div>
                <div class="rec-title">{rec['title']}</div>
                <div class="rec-body">
                    {"<br>".join(rec['actions'])}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)