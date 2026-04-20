"""
Melodia — Music Recommendation Explorer
Light, vibrant design — coral + amber + teal on cream white.
Run with:  streamlit run app.py
"""
 
import warnings
warnings.filterwarnings("ignore")
 
import streamlit as st
import pandas as pd
import numpy as np
 
st.set_page_config(
    page_title="Melodia · Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Design tokens ─────────────────────────────────────────────────────────────
# Palette:
#   Coral   #FF6B4A  (primary accent — warm, energetic)
#   Amber   #FFB347  (secondary accent — sunny)
#   Teal    #1DADA8  (contrast accent — cool)
#   Cream   #FFF8F3  (page background)
#   Sand    #FFF0E6  (card background)
#   Linen   #FDEBD7  (subtle surface)
#   Ink     #2D2420  (primary text — warm dark brown, not cold black)
#   Stone   #7A6660  (secondary text)
#   Blush   #FFD5C8  (light coral tint for borders/hover)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
 
/* ════════════════════════════════════════════════════════════════
   FORCE LIGHT THEME — override Streamlit JS dark-mode injection
   The JS writes body{background-color:#0e1117} into <head>.
   We win with a more-specific selector + !important.
   ════════════════════════════════════════════════════════════════ */
html                                          { background: #FFF8F3 !important; }
body                                          { background-color: #FFF8F3 !important; color: #2D2420 !important; }
.stApp                                        { background-color: #FFF8F3 !important; }
[data-testid="stAppViewContainer"]            { background-color: #FFF8F3 !important; }
[data-testid="stMain"]                        { background-color: #FFF8F3 !important; }
[data-testid="stHeader"]                      { background-color: #FFF8F3 !important; }
section[data-testid="stSidebar"] ~ div        { background-color: #FFF8F3 !important; }
 
/* Every generic div/span/p that might inherit dark bg */
.main .block-container                        { background-color: transparent !important; color: #2D2420 !important; }
.stElementContainer, [data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"]             { background-color: transparent !important; color: #2D2420 !important; }
 
/* ════════════════════════════════════════════════════════════════
   GLOBAL TEXT — every element on the page
   ════════════════════════════════════════════════════════════════ */
*, *::before, *::after                        { color: inherit; }
p, span, div, li, td, th, caption, label,
small, strong, em, b, a                       { color: #2D2420; }
 
/* ════════════════════════════════════════════════════════════════
   TYPOGRAPHY
   ════════════════════════════════════════════════════════════════ */
html, body, .stApp                            { font-family: 'DM Sans', sans-serif !important; }
h1 { font-family: 'Playfair Display', serif !important; font-size: 2.4rem !important; font-weight: 800 !important; color: #FF6B4A !important; }
h2 { font-family: 'Playfair Display', serif !important; font-size: 1.5rem !important; font-weight: 700 !important; color: #2D2420 !important; }
h3 { font-family: 'DM Sans', sans-serif !important; font-size: 1.1rem !important; font-weight: 600 !important; color: #2D2420 !important; }
 
/* ════════════════════════════════════════════════════════════════
   HIDE STREAMLIT CHROME
   ════════════════════════════════════════════════════════════════ */
#MainMenu, footer, header [data-testid="stDecoration"] { visibility: hidden; }
.block-container { padding-top: 2rem !important; padding-bottom: 4rem !important; }
 
/* ════════════════════════════════════════════════════════════════
   SIDEBAR — gradient, all white text
   ════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] { background: linear-gradient(160deg,#FF6B4A 0%,#FF8C5A 40%,#FFB347 100%) !important; border-right: none !important; }
[data-testid="stSidebar"] > div { background: transparent !important; }
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] div, [data-testid="stSidebar"] small { color: #fff !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.22) !important; color: #fff !important;
    border: 1.5px solid rgba(255,255,255,0.45) !important; border-radius: 10px !important;
    box-shadow: none !important; transition: all 0.18s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.38) !important; transform: translateX(3px) !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(255,255,255,0.42) !important; border-color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] code, [data-testid="stSidebar"] pre {
    background: rgba(255,255,255,0.25) !important; color: #fff !important;
    border: none !important; border-radius: 6px !important; word-break: break-all; }
 
/* ════════════════════════════════════════════════════════════════
   WIDGET LABELS (text above every selectbox / input)
   Renders as <div data-testid="stWidgetLabel">
   ════════════════════════════════════════════════════════════════ */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label { color: #2D2420 !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }
 
/* ════════════════════════════════════════════════════════════════
   TEXT INPUT
   ════════════════════════════════════════════════════════════════ */
[data-testid="stTextInput"] input,
[data-testid="stTextInputRootElement"] input {
    background-color: #FFFFFF !important; border: 1.5px solid #FFD5C8 !important;
    border-radius: 10px !important; color: #2D2420 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stTextInput"] input::placeholder { color: #B09890 !important; }
[data-testid="stTextInput"] input:focus {
    border-color: #FF6B4A !important; box-shadow: 0 0 0 3px rgba(255,107,74,0.15) !important; }
 
/* ════════════════════════════════════════════════════════════════
   SELECTBOX — the closed control box
   data-baseweb="select" is placed by BaseWeb on the outer wrapper
   ════════════════════════════════════════════════════════════════ */
[data-testid="stSelectbox"] { color: #2D2420 !important; }
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
    background-color: #FFFFFF !important;
    border-color: #FFD5C8 !important;
    color: #2D2420 !important;
}
/* The input, value text, placeholder inside the control */
[data-testid="stSelectbox"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div { color: #2D2420 !important; }
/* Dropdown arrow SVG */
[data-testid="stSelectbox"] svg { fill: #FF6B4A !important; color: #FF6B4A !important; }
 
/* ════════════════════════════════════════════════════════════════
   SELECTBOX DROPDOWN MENU
   Renders via portal into <body> — target by data-baseweb="menu"
   which BaseWeb always puts on the menu container element.
   ════════════════════════════════════════════════════════════════ */
[data-baseweb="popover"] { background-color: #FFFFFF !important; }
[data-baseweb="menu"] {
    background-color: #FFFFFF !important; color: #2D2420 !important;
    border: 1.5px solid #FFD5C8 !important; border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(255,107,74,0.18) !important; overflow: hidden !important;
}
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"],
[data-baseweb="menu"] ul > li {
    background-color: #FFFFFF !important; color: #2D2420 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] li[aria-selected="true"],
[data-baseweb="menu"] [role="option"][aria-selected="true"],
[data-baseweb="menu"] li[data-highlighted="true"] {
    background-color: #FFF0E6 !important; color: #FF6B4A !important;
}
/* stSelectboxVirtualDropdown — the list container testid */
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] * {
    background-color: #FFFFFF !important; color: #2D2420 !important;
}
 
/* ════════════════════════════════════════════════════════════════
   DATAFRAME — glide-data-grid uses CSS custom properties
   Must be set on the wrapper + all child divs (grid reads from
   nearest parent that defines the var).
   ════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] > div > div,
[data-testid="stDataFrameResizable"],
[data-testid="stDataFrameResizable"] > div {
    --gdg-bg-cell:                 #FFFFFF;
    --gdg-bg-cell-medium:          #FFF8F3;
    --gdg-bg-header:               #FFF0E6;
    --gdg-bg-header-has-focus:     #FFD5C8;
    --gdg-bg-header-hovered:       #FFE4D4;
    --gdg-text-dark:               #2D2420;
    --gdg-text-medium:             #7A6660;
    --gdg-text-light:              #B09890;
    --gdg-text-header:             #2D2420;
    --gdg-text-header-selected:    #FFFFFF;
    --gdg-text-bubble:             #2D2420;
    --gdg-text-group-header:       #2D2420;
    --gdg-accent-color:            #FF6B4A;
    --gdg-accent-fg:               #FFFFFF;
    --gdg-accent-light:            rgba(255,107,74,0.15);
    --gdg-border-color:            rgba(255,213,200,0.8);
    --gdg-horizontal-border-color: rgba(255,213,200,0.6);
    --gdg-bg-bubble:               #FFF0E6;
    --gdg-bg-bubble-selected:      #FFD5C8;
    --gdg-link-color:              #FF6B4A;
    --gdg-bg-icon-header:          #FF8C5A;
    --gdg-fg-icon-header:          #FFFFFF;
    background-color: #FFFFFF !important;
    border-radius: 14px !important; overflow: hidden !important;
    border: 1.5px solid #FFD5C8 !important;
    box-shadow: 0 2px 12px rgba(255,107,74,0.07) !important;
}
[data-testid="stDataFrameToolbar"] {
    background-color: #FFF0E6 !important; border-bottom: 1px solid #FFD5C8 !important; }
[data-testid="stDataFrameToolbar"] button {
    color: #FF6B4A !important; background: transparent !important;
    border: none !important; box-shadow: none !important; }
[data-testid="stDataFrameToolbar"] button:hover { background: #FFD5C8 !important; transform: none !important; }
 
/* ════════════════════════════════════════════════════════════════
   VEGA / BAR CHARTS — SVG background + text colours
   ════════════════════════════════════════════════════════════════ */
[data-testid="stVegaLiteChart"],
.stVegaLiteChart, .vega-embed { background: #FFF8F3 !important; border-radius: 12px !important; }
[data-testid="stVegaLiteChart"] svg, .stVegaLiteChart svg, .vega-embed svg { background: #FFF8F3 !important; }
[data-testid="stVegaLiteChart"] svg text, .stVegaLiteChart svg text { fill: #2D2420 !important; }
.stVegaLiteChart svg .role-axis path,
.stVegaLiteChart svg .role-axis line { stroke: #B09890 !important; }
.stVegaLiteChart svg .role-axis .grid line { stroke: #FFD5C8 !important; }
 
/* ════════════════════════════════════════════════════════════════
   TABS
   ════════════════════════════════════════════════════════════════ */
[data-baseweb="tab-list"] {
    background: #FFF0E6 !important; border-radius: 12px !important;
    padding: 4px !important; gap: 4px !important; border-bottom: none !important; }
[data-baseweb="tab"] {
    background: transparent !important; color: #7A6660 !important;
    font-weight: 500 !important; border-radius: 8px !important;
    padding: 6px 16px !important; border: none !important; }
[data-baseweb="tab"][aria-selected="true"] {
    background: #FF6B4A !important; color: #fff !important; font-weight: 600 !important; }
[data-baseweb="tab-panel"] { background: transparent !important; color: #2D2420 !important; }
 
/* ════════════════════════════════════════════════════════════════
   EXPANDER
   ════════════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: #FFFFFF !important; border: 1.5px solid #FFD5C8 !important; border-radius: 12px !important; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *  { color: #2D2420 !important; font-weight: 500 !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"],
[data-testid="stExpander"] [data-testid="stExpanderDetails"] * { color: #2D2420 !important; background: transparent !important; }
 
/* ════════════════════════════════════════════════════════════════
   MARKDOWN TABLES
   ════════════════════════════════════════════════════════════════ */
[data-testid="stMarkdownContainer"] table { background: #FFFFFF !important; border-collapse: collapse !important; width: 100% !important; }
[data-testid="stMarkdownContainer"] thead th {
    background: #FFF0E6 !important; color: #2D2420 !important; font-weight: 600 !important;
    font-size: 0.82rem !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important; border: 1px solid #FFD5C8 !important; padding: 10px 14px !important; }
[data-testid="stMarkdownContainer"] tbody tr { background: #FFFFFF !important; }
[data-testid="stMarkdownContainer"] tbody tr:nth-child(even) { background: #FFF8F3 !important; }
[data-testid="stMarkdownContainer"] tbody td {
    color: #2D2420 !important; border: 1px solid #FFD5C8 !important; padding: 9px 14px !important; background: inherit !important; }
 
/* ════════════════════════════════════════════════════════════════
   METRICS
   ════════════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: #FFFFFF !important; border: 1.5px solid #FFD5C8 !important;
    border-radius: 14px !important; padding: 14px 18px !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
    color: #FF6B4A !important; font-family: 'Playfair Display', serif !important;
    font-size: 1.7rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
    color: #7A6660 !important; text-transform: uppercase !important;
    letter-spacing: 0.07em !important; font-size: 0.78rem !important; }
 
/* ════════════════════════════════════════════════════════════════
   ALERTS / INFO BOXES
   ════════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] { border-radius: 12px !important; }
[data-testid="stAlert"] * { font-family: 'DM Sans', sans-serif !important; }
 
/* ════════════════════════════════════════════════════════════════
   CODE / CAPTIONS
   ════════════════════════════════════════════════════════════════ */
[data-testid="stCaptionContainer"] p, small { color: #7A6660 !important; }
code { background: #FFF0E6 !important; border: 1px solid #FFD5C8 !important; border-radius: 8px !important; color: #C04020 !important; }
[data-testid="stSidebar"] code { color: #fff !important; background: rgba(255,255,255,0.25) !important; border: none !important; }
 
/* ════════════════════════════════════════════════════════════════
   BUTTONS
   ════════════════════════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg,#FF6B4A,#FF8C5A) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(255,107,74,0.30) !important; transition: all 0.18s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#FF5235,#FF7842) !important;
    box-shadow: 0 4px 18px rgba(255,107,74,0.40) !important; transform: translateY(-1px) !important; }
[data-testid="stDownloadButton"] button { background: #1DADA8 !important; box-shadow: 0 2px 10px rgba(29,173,168,0.25) !important; }
[data-testid="stDownloadButton"] button:hover { background: #159A95 !important; }
 
/* ════════════════════════════════════════════════════════════════
   MISC
   ════════════════════════════════════════════════════════════════ */
hr { border-color: #FFD5C8 !important; margin: 1.2rem 0 !important; }
 
/* ════════════════════════════════════════════════════════════════
   CUSTOM HTML CLASSES
   ════════════════════════════════════════════════════════════════ */
.mcard {
    background:#FFFFFF; border:1.5px solid #FFD5C8; border-radius:16px;
    padding:20px 18px; text-align:center; box-shadow:0 2px 16px rgba(255,107,74,0.08);
    transition:transform 0.18s,box-shadow 0.18s;
}
.mcard:hover { transform:translateY(-2px); box-shadow:0 6px 24px rgba(255,107,74,0.14); }
.mcard .val { font-family:'Playfair Display',serif; font-size:2rem; font-weight:800; color:#FF6B4A; line-height:1.1; }
.mcard .lbl { font-size:0.72rem; color:#7A6660; margin-top:6px; text-transform:uppercase; letter-spacing:0.09em; font-weight:500; }
 
.rec-card { background:#FFFFFF; border:1.5px solid #FFD5C8; border-radius:14px; padding:16px 20px; margin-bottom:12px; box-shadow:0 1px 8px rgba(255,107,74,0.06); transition:all 0.18s; }
.rec-card:hover { border-color:#FF6B4A; box-shadow:0 4px 20px rgba(255,107,74,0.15); transform:translateY(-1px); }
.rec-rank { font-family:'Playfair Display',serif; color:#FF6B4A; font-weight:800; font-size:1.15rem; }
.rec-title { font-weight:600; font-size:0.98rem; color:#2D2420; }
.rec-artist { color:#7A6660; font-size:0.85rem; }
.rec-explain { font-size:0.78rem; color:#1DADA8; margin-top:7px; font-style:italic; }
.bar-bg  { background:#FDEBD7; border-radius:6px; height:6px; margin-top:10px; }
.bar-fill { background:linear-gradient(90deg,#FF6B4A,#FFB347); border-radius:6px; height:6px; }
.pill { display:inline-block; background:#FFF0E6; border:1px solid #FFD5C8; border-radius:20px; padding:2px 10px; font-size:0.72rem; color:#FF6B4A; margin-right:5px; font-weight:500; }
.uid-banner { background:linear-gradient(135deg,#FFF0E6,#FFF8F3); border:2px solid #FF6B4A; border-radius:14px; padding:16px 22px; font-size:0.83rem; color:#2D2420; margin-bottom:14px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px; box-shadow:0 2px 12px rgba(255,107,74,0.10); }
.uid-banner b { color:#FF6B4A; }
.data-note { background:linear-gradient(135deg,#E8FAF8,#F0FDF9); border:1.5px solid #1DADA8; border-radius:12px; padding:12px 18px; font-size:0.83rem; color:#136E6B; margin-bottom:16px; font-weight:500; }
</style>
""", unsafe_allow_html=True)
 
 
# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🏠 Overview"
if "selected_uid" not in st.session_state:
    st.session_state.selected_uid = ""
 
 
# ── Load & cache ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from recommender import (
        load_data, fill_genre_from_tags, preprocess,
        build_content_features, build_collaborative_model,
    )
    music_raw, listen_raw = load_data()
    music_raw = fill_genre_from_tags(music_raw)
    music, listen = preprocess(music_raw, listen_raw)
    X_content    = build_content_features(music)
    collab_model = build_collaborative_model(listen, music)
    return music, listen, X_content, collab_model
 
 
@st.cache_data(show_spinner=False)
def build_user_table(_listen, _model_user_set):
    stats = (
        _listen.groupby("user_id")
        .agg(songs_played=("song_id", "nunique"), total_plays=("play_count", "sum"))
        .reset_index()
        .sort_values("total_plays", ascending=False)
        .reset_index(drop=True)
    )
    stats.index += 1
    stats["in_model"] = stats["user_id"].isin(_model_user_set)
    stats["short_id"] = stats["user_id"].str[:20] + "…"
    return stats
 
 
@st.cache_data(show_spinner=False)
def get_top_tracks(_listen, _music, user_id, n=20):
    hist = _listen[_listen["user_id"] == user_id].copy()
    hist = hist.merge(_music[["song_id","title","artist","genre","year"]], on="song_id", how="left")
    return hist.sort_values("play_count", ascending=False).head(n).reset_index(drop=True)
 
 
def run_recs(user_id, music, listen, X_content, collab_model, n):
    from recommender import recommend_for_user
    return recommend_for_user(user_id=user_id, music=music, listen=listen,
                               X_content=X_content, collab_model=collab_model, top_n=n)
 
 
# ── Boot ──────────────────────────────────────────────────────────────────────
with st.spinner("🎵 Warming up Melodia — loading all 32,947 users…"):
    music, listen, X_content, collab_model = load_model()
 
model_user_set  = set(collab_model["user_to_idx"].keys())
user_table_full = build_user_table(listen, model_user_set)
 
N_TOTAL   = listen["user_id"].nunique()
N_MODEL   = len(model_user_set)
N_SONGS   = listen["song_id"].nunique()
N_EVENTS  = len(listen)
N_CATALOG = len(music)
 
 
# ── Metric row helper ─────────────────────────────────────────────────────────
def metric_row(items):
    cols = st.columns(len(items))
    for col, (val, lbl) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="mcard"><div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)
 
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Playfair Display,serif;font-size:1.6rem;"
        "font-weight:800;color:#fff;margin-bottom:2px'>🎵 Melodia</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.8rem;color:rgba(255,255,255,0.80);"
        "margin-bottom:18px;font-family:DM Sans,sans-serif'>"
        "Music Recommendation Explorer</div>",
        unsafe_allow_html=True,
    )
 
    for label in ["🏠 Overview", "👥 Browse Users", "🔍 Recommendations"]:
        active = st.session_state.page == label
        if st.button(label, key=f"nav_{label}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.page = label
            st.rerun()
 
    st.markdown("<hr style='border-color:rgba(255,255,255,0.25)!important;margin:18px 0'>",
                unsafe_allow_html=True)
 
    if st.session_state.selected_uid:
        st.markdown(
            "<div style='font-size:0.75rem;font-weight:600;color:rgba(255,255,255,0.75);"
            "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px'>"
            "Active user</div>",
            unsafe_allow_html=True,
        )
        st.code(st.session_state.selected_uid, language=None)
        if st.button("✕ Clear selection", use_container_width=True):
            st.session_state.selected_uid = ""
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
 
    st.markdown(
        "<div style='font-size:0.72rem;color:rgba(255,255,255,0.60);"
        "font-family:DM Sans,sans-serif;line-height:1.6'>"
        "CSC 6740 · Data Mining<br>Team: Data Miners<br>"
        "All 32,947 users · Full data</div>",
        unsafe_allow_html=True,
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "🏠 Overview":
 
    st.markdown("# 🎵 Welcome to Melodia")
    st.markdown(
        "<p style='font-size:1.05rem;color:#7A6660;margin-top:-10px;margin-bottom:24px'>"
        "Hybrid music recommendations powered by <b>Collaborative Filtering</b> "
        "+ <b>Audio Content Analysis</b> + <b>Novelty Discovery</b></p>",
        unsafe_allow_html=True,
    )
 
    st.markdown("""<div class="data-note">
        ✅ <b>Full data mode</b> — using all 32,947 qualifying users and all 50,683 tracks.
        No artificial caps applied.
    </div>""", unsafe_allow_html=True)
 
    metric_row([
        (f"{N_TOTAL:,}",   "Users in listening history"),
        (f"{N_MODEL:,}",   "Users in CF model"),
        (f"{N_CATALOG:,}", "Tracks in catalog"),
        (f"{N_EVENTS:,}",  "Listen interactions"),
    ])
 
    col_l, col_r = st.columns([1, 1], gap="large")
 
    with col_l:
        st.markdown("### How recommendations work")
        st.markdown("""
<div style='background:#fff;border:1.5px solid #FFD5C8;border-radius:14px;
            padding:18px 20px;font-family:"DM Sans",sans-serif;font-size:0.88rem'>
  <div style='margin-bottom:12px'>
    <span style='background:#FF6B4A;color:#fff;border-radius:6px;
                 padding:2px 8px;font-weight:600;font-size:0.75rem'>CF · up to 70%</span>
    <span style='margin-left:8px;color:#2D2420'>Users with similar taste patterns</span>
  </div>
  <div style='margin-bottom:12px'>
    <span style='background:#FFB347;color:#fff;border-radius:6px;
                 padding:2px 8px;font-weight:600;font-size:0.75rem'>Content · up to 80%</span>
    <span style='margin-left:8px;color:#2D2420'>Audio features: energy, valence, tempo</span>
  </div>
  <div style='margin-bottom:16px'>
    <span style='background:#1DADA8;color:#fff;border-radius:6px;
                 padding:2px 8px;font-weight:600;font-size:0.75rem'>Novelty · 8–20%</span>
    <span style='margin-left:8px;color:#2D2420'>Fresh tracks outside your comfort zone</span>
  </div>
  <div style='border-top:1px solid #FFD5C8;padding-top:12px;color:#7A6660;font-size:0.82rem'>
    Weights adapt dynamically — new users get content-heavy recs,
    heavy listeners get CF-heavy recs. MMR re-ranking ensures artist diversity.
  </div>
</div>
        """, unsafe_allow_html=True)
 
    with col_r:
        st.markdown("### Data usage")
        st.markdown("""
| Dimension | CSV total | Used |
|---|---|---|
| Users | 43,636 | **32,947** |
| Tracks (CF) | 22,470 | **15,829** |
| Tracks (content) | 50,683 | **50,683** |
| Events | 448,523 | **416,986** |
        """)
 
    st.markdown("---")
    col_a, col_b = st.columns(2, gap="large")
 
    with col_a:
        st.markdown("### User activity distribution")
        user_plays = listen.groupby("user_id")["play_count"].sum()
        bins   = [0,10,20,50,100,250,500,99999]
        labels = ["1–10","11–20","21–50","51–100","101–250","251–500","500+"]
        dist   = pd.cut(user_plays, bins=bins, labels=labels).value_counts().sort_index()
        st.bar_chart(
            pd.DataFrame({"Users": dist.values}, index=dist.index.astype(str)),
            color="#FF6B4A",
        )
 
    with col_b:
        st.markdown("### Genre breakdown")
        gc = music["genre"].value_counts().head(12)
        st.bar_chart(
            pd.DataFrame({"Tracks": gc.values}, index=gc.index.astype(str)),
            color="#1DADA8",
        )
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;padding:20px;background:#FFF0E6;border-radius:14px;"
        "border:1.5px solid #FFD5C8'>"
        "<span style='font-size:1rem;color:#7A6660;font-family:DM Sans,sans-serif'>"
        "👈 Go to <b>Browse Users</b> — click any row to jump straight to their recommendations"
        "</span></div>",
        unsafe_allow_html=True,
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — BROWSE USERS
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "👥 Browse Users":
 
    st.markdown("# 👥 Browse Users")
    st.markdown(
        f"<p style='color:#7A6660;margin-top:-10px'>"
        f"<b style='color:#FF6B4A'>{N_TOTAL:,} users</b> in listening history · "
        f"<b style='color:#FF6B4A'>{N_MODEL:,}</b> in the trained CF model</p>",
        unsafe_allow_html=True,
    )
 
    st.markdown("""<div class="data-note">
        ✅ All 32,947 qualifying users shown — no cap applied.
    </div>""", unsafe_allow_html=True)
 
    st.info(
        "🖱️ **Click any row** to select a user · "
        "Full ID appears below with a copy-ready box · "
        "Press **▶ Go to recommendations** to see their personalised results"
    )
 
    st.markdown("---")
 
    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns([3, 1, 1, 1])
    with fc1:
        search_q = st.text_input(
            "Search", placeholder="🔍  Type part of a user ID…",
            label_visibility="collapsed",
        )
    with fc2:
        min_plays = st.selectbox("Min plays", [0,10,20,50,100,250],
                                 label_visibility="collapsed")
    with fc3:
        model_filter = st.selectbox("Status", ["All","In CF model","Cold start only"],
                                    label_visibility="collapsed")
    with fc4:
        page_size = st.selectbox("Show", [25,50,100,200,500], index=1,
                                  label_visibility="collapsed")
 
    tbl = user_table_full.copy()
    if search_q.strip():
        tbl = tbl[tbl["user_id"].str.contains(search_q.strip(), case=False, na=False)]
    if min_plays > 0:
        tbl = tbl[tbl["total_plays"] >= min_plays]
    if model_filter == "In CF model":
        tbl = tbl[tbl["in_model"]]
    elif model_filter == "Cold start only":
        tbl = tbl[~tbl["in_model"]]
 
    tbl_page = tbl.head(page_size).reset_index(drop=True)
    st.caption(f"Showing {len(tbl_page):,} of {len(tbl):,} matching users")
 
    # ── Clickable dataframe ───────────────────────────────────────────────────
    display_df = pd.DataFrame({
        "User ID (preview)": tbl_page["short_id"],
        "Unique songs":      tbl_page["songs_played"].astype(int),
        "Total plays":       tbl_page["total_plays"].astype(int),
        "In CF model":       tbl_page["in_model"].map({True: "✅ Yes", False: "⚠️ Cold start"}),
    })
 
    selection = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "User ID (preview)": st.column_config.TextColumn(
                "User ID (preview)",
                help="Click a row — full ID appears below",
                width="large",
            ),
            "Unique songs": st.column_config.NumberColumn(width="small"),
            "Total plays":  st.column_config.NumberColumn(width="small"),
            "In CF model":  st.column_config.TextColumn(width="medium"),
        },
    )
 
    # ── UID banner & action buttons ───────────────────────────────────────────
    def show_uid_actions(full_uid, row_data):
        st.markdown(
            f"""<div class="uid-banner">
              <div>
                <b>Selected user</b><br>
                <span style='font-size:0.92rem;font-family:monospace'>{full_uid}</span>
              </div>
              <div style='font-size:0.8rem;color:#7A6660;font-family:"DM Sans",sans-serif'>
                <b style='color:#FF6B4A'>{int(row_data['songs_played'])}</b> songs &nbsp;·&nbsp;
                <b style='color:#FF6B4A'>{int(row_data['total_plays'])}</b> plays &nbsp;·&nbsp;
                {'✅ In CF model' if row_data['in_model'] else '⚠️ Cold start'}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        b1, b2, _ = st.columns([1.3, 2.0, 2.7])
        with b1:
            if st.button("▶ Go to recommendations", type="primary", use_container_width=True):
                st.session_state.page = "🔍 Recommendations"
                st.rerun()
        with b2:
            st.code(full_uid, language=None)
 
    selected_rows = selection.selection.rows
 
    if selected_rows:
        row_idx  = selected_rows[0]
        full_uid = tbl_page.iloc[row_idx]["user_id"]
        row_data = tbl_page.iloc[row_idx]
        st.session_state.selected_uid = full_uid
        show_uid_actions(full_uid, row_data)
 
    elif st.session_state.selected_uid:
        uid = st.session_state.selected_uid
        prev = user_table_full[user_table_full["user_id"] == uid]
        if not prev.empty:
            show_uid_actions(uid, prev.iloc[0])
        else:
            st.markdown(
                f'<div class="uid-banner"><b>Previously selected:</b>&nbsp;{uid}</div>',
                unsafe_allow_html=True,
            )
            if st.button("▶ Go to recommendations", type="primary"):
                st.session_state.page = "🔍 Recommendations"
                st.rerun()
    else:
        st.markdown(
            "<div style='color:#7A6660;font-size:0.87rem;padding:12px 0;"
            "font-family:\"DM Sans\",sans-serif'>"
            "← Click any row above to select a user and see their full ID</div>",
            unsafe_allow_html=True,
        )
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "🔍 Recommendations":
 
    st.markdown("# 🔍 Recommendations")
    st.markdown(
        "<p style='color:#7A6660;margin-top:-10px;font-family:\"DM Sans\",sans-serif'>"
        "Enter any user ID to see their personalised recommendations</p>",
        unsafe_allow_html=True,
    )
 
    ic1, ic2 = st.columns([4, 1])
    with ic1:
        uid_input = st.text_input(
            "User ID",
            value=st.session_state.selected_uid,
            placeholder="Paste a full user ID, or go to Browse Users and click a row",
        )
        st.session_state.selected_uid = uid_input
    with ic2:
        top_n = st.selectbox("# Recs", [5,10,15,20], index=1)
 
    with st.expander("📋 Pick from most active users"):
        top50    = user_table_full.head(50)
        pick_opts = top50.apply(
            lambda r: (
                f"{r['user_id'][:28]}…  |  "
                f"{int(r['songs_played'])} songs  |  "
                f"{int(r['total_plays'])} plays"
            ),
            axis=1,
        ).tolist()
        pick_ids = top50["user_id"].tolist()
        picked   = st.selectbox("Select", ["— choose —"] + pick_opts,
                                label_visibility="collapsed")
        if picked != "— choose —" and st.button("Use this user"):
            st.session_state.selected_uid = pick_ids[pick_opts.index(picked)]
            st.rerun()
 
    st.markdown("---")
 
    uid = uid_input.strip()
    if not uid:
        st.markdown(
            "<div style='text-align:center;padding:40px;background:#FFF0E6;"
            "border-radius:16px;border:1.5px solid #FFD5C8'>"
            "<div style='font-size:2rem'>🎵</div>"
            "<div style='color:#7A6660;font-family:\"DM Sans\",sans-serif;margin-top:8px'>"
            "Enter a user ID above, or click a row in Browse Users</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.stop()
 
    if uid not in set(listen["user_id"].unique()):
        st.error(f"❌ User `{uid}` not found. Go to Browse Users to find valid IDs.")
        st.stop()
 
    user_hist = listen[listen["user_id"] == uid]
    n_u_songs = int(user_hist["song_id"].nunique())
    n_u_plays = int(user_hist["play_count"].sum())
    in_model  = uid in model_user_set
 
    metric_row([
        (n_u_songs,  "Unique songs played"),
        (n_u_plays,  "Total play count"),
        ("CF + Content" if in_model else "Content only", "Recommendation mode"),
        ("✅ In CF model" if in_model else "⚠️ Cold start", "CF model status"),
    ])
 
    if not in_model:
        st.info("ℹ️ Limited history — using content features + popularity instead of CF.")
 
    tab_reco, tab_hist, tab_sig = st.tabs(
        ["🎵 Recommendations", "📚 Listening History", "📊 Signal Breakdown"]
    )
 
    with st.spinner("🎵 Finding your perfect tracks…"):
        recs = run_recs(uid, music, listen, X_content, collab_model, top_n)
 
    # ── Recommendations tab ───────────────────────────────────────────────────
    with tab_reco:
        if recs.empty:
            st.warning("No recommendations could be generated for this user.")
        else:
            st.caption(f"Weight regime: {recs['weight_regime'].iloc[0]}")
            st.markdown("---")
 
            for i, row in recs.iterrows():
                score_pct  = int(row["final_score"] * 100)
                genre_pill = (
                    f'<span class="pill">{row["genre"]}</span>'
                    if str(row.get("genre","")) not in ("","Unknown","nan") else ""
                )
                year_pill = (
                    f'<span class="pill">{int(row["year"])}</span>'
                    if row.get("year",0) and int(row.get("year",0)) > 1000 else ""
                )
                st.markdown(f"""
                <div class="rec-card">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div>
                      <span class="rec-rank">#{i+1}</span>&nbsp;&nbsp;
                      <span class="rec-title">{row['title']}</span><br>
                      <span class="rec-artist">by {row['artist']}</span>
                      &nbsp;{genre_pill}{year_pill}
                    </div>
                    <div style="text-align:right;min-width:68px">
                      <span style="color:#FF6B4A;font-family:'Playfair Display',serif;
                                   font-weight:800;font-size:1.15rem">{score_pct}%</span>
                      <div style="color:#7A6660;font-size:0.68rem;
                                  font-family:'DM Sans',sans-serif">match</div>
                    </div>
                  </div>
                  <div class="bar-bg">
                    <div class="bar-fill" style="width:{score_pct}%"></div>
                  </div>
                  <div class="rec-explain">✦ {row['explanation']}</div>
                </div>
                """, unsafe_allow_html=True)
 
            csv_out = recs[["title","artist","genre","year",
                             "final_score","explanation"]].to_csv(index=False)
            st.download_button(
                "⬇️ Download recommendations as CSV",
                data=csv_out,
                file_name=f"recs_{uid[:12]}.csv",
                mime="text/csv",
            )
 
    # ── History tab ───────────────────────────────────────────────────────────
    with tab_hist:
        top_tracks = get_top_tracks(listen, music, uid, n=20)
 
        if top_tracks.empty:
            st.info("No listening history found.")
        else:
            user_genres = top_tracks["genre"].value_counts().head(6)
            if not user_genres.empty:
                st.markdown("**Genre taste:**")
                gcols = st.columns(min(len(user_genres), 6))
                for i, (g, cnt) in enumerate(user_genres.items()):
                    with gcols[i]:
                        st.markdown(
                            f'<div class="mcard" style="padding:12px">'
                            f'<div class="val" style="font-size:1.3rem">{cnt}</div>'
                            f'<div class="lbl">{g}</div></div>',
                            unsafe_allow_html=True,
                        )
                st.markdown("<br>", unsafe_allow_html=True)
 
            st.markdown("**Most played tracks:**")
            st.dataframe(
                top_tracks[["title","artist","genre","year","play_count"]].rename(columns={
                    "title":"Track","artist":"Artist","genre":"Genre",
                    "year":"Year","play_count":"Times played",
                }),
                use_container_width=True,
                hide_index=True,
            )
 
    # ── Signal breakdown tab ──────────────────────────────────────────────────
    with tab_sig:
        if recs.empty:
            st.info("No recommendations to analyse.")
        else:
            st.caption(
                "CF = collaborative filter · "
                "Content = audio feature similarity · "
                "Novelty = how fresh/unfamiliar the track is"
            )
            sig_df = recs[["title","artist","cf_score",
                            "content_score","novelty_score","final_score"]].copy()
            sig_df.columns = ["Track","Artist","CF","Content","Novelty","Final"]
            st.dataframe(sig_df.round(3), use_container_width=True, hide_index=True)
 
            st.markdown("---")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Avg CF score",      f"{recs['cf_score'].mean():.3f}")
            with sc2:
                st.metric("Avg Content score", f"{recs['content_score'].mean():.3f}")
            with sc3:
                st.metric("Avg Novelty score", f"{recs['novelty_score'].mean():.3f}")
 
            st.markdown("---")
            st.markdown("**Signal contribution per recommendation:**")
            chart_df = recs[["title","cf_score","content_score","novelty_score"]].set_index("title")
            chart_df.columns = ["CF","Content","Novelty"]
            st.bar_chart(chart_df, color=["#FF6B4A","#FFB347","#1DADA8"])