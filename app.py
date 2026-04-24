"""Melodia — Music Recommendation Explorer."""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from urllib.parse import quote

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
 
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
    /* Cell colours — match the cream page background so the table
       sits flush with the page rather than as a white floating box */
    --gdg-bg-cell:                 #FFF8F3;
    --gdg-bg-cell-medium:          #FFF3EC;
    /* Header — one shade warmer than the cells, very subtle */
    --gdg-bg-header:               #FDEBD7;
    --gdg-bg-header-has-focus:     #FFD5C8;
    --gdg-bg-header-hovered:       #FFE4D4;
    /* Text — same warm dark ink as everywhere else */
    --gdg-text-dark:               #2D2420;
    --gdg-text-medium:             #7A6660;
    --gdg-text-light:              #B09890;
    --gdg-text-header:             #5C3D30;
    --gdg-text-header-selected:    #FFFFFF;
    --gdg-text-bubble:             #2D2420;
    --gdg-text-group-header:       #5C3D30;
    /* Accent stays coral */
    --gdg-accent-color:            #FF6B4A;
    --gdg-accent-fg:               #FFFFFF;
    --gdg-accent-light:            rgba(255,107,74,0.12);
    /* Borders — very soft, barely visible lines between rows */
    --gdg-border-color:            rgba(210,185,175,0.45);
    --gdg-horizontal-border-color: rgba(210,185,175,0.35);
    /* Bubbles */
    --gdg-bg-bubble:               #FFF0E6;
    --gdg-bg-bubble-selected:      #FFD5C8;
    --gdg-link-color:              #FF6B4A;
    --gdg-bg-icon-header:          #E8B090;
    --gdg-fg-icon-header:          #FFFFFF;
    /* Wrapper — no box, no shadow, just a hairline border */
    background-color: #FFF8F3 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(210,185,175,0.5) !important;
    box-shadow: none !important;
}
[data-testid="stDataFrameToolbar"] {
    background-color: #FFF3EC !important;
    border-bottom: 1px solid rgba(210,185,175,0.4) !important;
}
[data-testid="stDataFrameToolbar"] button {
    color: #FF6B4A !important; background: transparent !important;
    border: none !important; box-shadow: none !important;
}
[data-testid="stDataFrameToolbar"] button:hover {
    background: rgba(255,107,74,0.08) !important; transform: none !important;
}
 
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
.visual-card {
    background:#FFFFFF; border:1.5px solid #FFD5C8; border-radius:18px; overflow:hidden;
    box-shadow:0 6px 22px rgba(255,107,74,0.08); height:100%;
}
.visual-card img { width:100%; height:180px; object-fit:cover; display:block; }
.visual-card .body { padding:16px 18px 18px 18px; }
.visual-card .title { font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:700; color:#2D2420; }
.visual-card .copy { font-size:0.86rem; line-height:1.55; color:#7A6660; margin-top:6px; }
.hero-panel {
    background:linear-gradient(135deg,#FFFFFF 0%,#FFF3EA 55%,#FFF8F3 100%);
    border:1.5px solid #FFD5C8; border-radius:22px; padding:24px;
    box-shadow:0 8px 30px rgba(255,107,74,0.08); margin-bottom:22px;
}
.hero-kicker {
    display:inline-block; background:#FFF0E6; border:1px solid #FFD5C8; border-radius:999px;
    padding:5px 12px; font-size:0.76rem; color:#FF6B4A; font-weight:600; letter-spacing:0.05em;
    text-transform:uppercase;
}
.cm-note {
    background:#FFFFFF; border:1.5px solid #FFD5C8; border-radius:16px; padding:18px;
    box-shadow:0 4px 16px rgba(255,107,74,0.06);
}
</style>
""", unsafe_allow_html=True)
 

def find_audio_map():
    root = Path(__file__).resolve().parent / "MP3-Example"
    mapping = {}
    if root.exists():
        for mp3 in root.rglob("*.mp3"):
            name = mp3.name
            if "-" in name:
                track_id = name.split("-", 1)[1].rsplit(".", 1)[0]
                mapping[track_id] = str(mp3)
    return mapping


@st.cache_resource(show_spinner=False)
def get_audio_map():
    return find_audio_map()


def get_preview_source(track_id, preview_url):
    audio_map = get_audio_map()
    if track_id in audio_map:
        return audio_map[track_id]
    if isinstance(preview_url, str) and preview_url.strip():
        return preview_url
    return None


if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "selected_uid" not in st.session_state:
    st.session_state.selected_uid = ""

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
with st.spinner("🎵 Warming up Melodia — loading the listening history and catalog..."):
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


def svg_data_uri(svg_markup: str) -> str:
    return f"data:image/svg+xml;utf8,{quote(svg_markup)}"


def build_feature_visuals():
    visuals = [
        {
            "title": "Taste Mapping",
            "copy": "A quick visual layer that shows Melodia balancing familiar listening patterns with new discovery.",
            "image": svg_data_uri(
                """
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 260">
                  <defs>
                    <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="#FF6B4A"/>
                      <stop offset="100%" stop-color="#FFB347"/>
                    </linearGradient>
                  </defs>
                  <rect width="480" height="260" rx="28" fill="#FFF3EA"/>
                  <circle cx="126" cy="128" r="78" fill="url(#g1)" opacity="0.92"/>
                  <circle cx="126" cy="128" r="18" fill="#FFF8F3"/>
                  <path d="M238 165 C278 120, 325 102, 378 80" stroke="#1DADA8" stroke-width="12" fill="none" stroke-linecap="round"/>
                  <path d="M238 194 C290 154, 342 156, 402 121" stroke="#FF8C5A" stroke-width="10" fill="none" stroke-linecap="round" opacity="0.85"/>
                  <circle cx="384" cy="78" r="17" fill="#1DADA8"/>
                  <circle cx="408" cy="118" r="11" fill="#FFB347"/>
                  <circle cx="350" cy="147" r="9" fill="#FF6B4A"/>
                </svg>
                """
            ),
        },
        {
            "title": "Signal Blend",
            "copy": "Collaborative, content, and novelty signals each get their own visual space in the interface.",
            "image": svg_data_uri(
                """
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 260">
                  <rect width="480" height="260" rx="28" fill="#F9FFFE"/>
                  <rect x="64" y="144" width="48" height="62" rx="18" fill="#FF6B4A"/>
                  <rect x="134" y="112" width="48" height="94" rx="18" fill="#FF8C5A"/>
                  <rect x="204" y="82" width="48" height="124" rx="18" fill="#FFB347"/>
                  <rect x="274" y="104" width="48" height="102" rx="18" fill="#1DADA8"/>
                  <rect x="344" y="58" width="48" height="148" rx="18" fill="#2D2420" opacity="0.82"/>
                  <path d="M62 88 C124 35, 182 38, 244 88 S364 141, 420 84" stroke="#1DADA8" stroke-width="8" fill="none" stroke-linecap="round" opacity="0.88"/>
                </svg>
                """
            ),
        },
        {
            "title": "Genre Movement",
            "copy": "The new insights panel compares listener preference genres against recommendation output genres.",
            "image": svg_data_uri(
                """
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 260">
                  <defs>
                    <linearGradient id="g2" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="#1DADA8"/>
                      <stop offset="100%" stop-color="#88DDD8"/>
                    </linearGradient>
                  </defs>
                  <rect width="480" height="260" rx="28" fill="#FFF8F3"/>
                  <rect x="58" y="48" width="364" height="164" rx="22" fill="#FFFFFF" stroke="#FFD5C8" stroke-width="3"/>
                  <rect x="86" y="78" width="66" height="42" rx="12" fill="#FF6B4A" opacity="0.92"/>
                  <rect x="164" y="78" width="66" height="42" rx="12" fill="#FFB347" opacity="0.92"/>
                  <rect x="242" y="78" width="66" height="42" rx="12" fill="url(#g2)" opacity="0.95"/>
                  <rect x="320" y="78" width="66" height="42" rx="12" fill="#2D2420" opacity="0.82"/>
                  <rect x="86" y="132" width="66" height="42" rx="12" fill="#FFD5C8"/>
                  <rect x="164" y="132" width="66" height="42" rx="12" fill="#FFE3A7"/>
                  <rect x="242" y="132" width="66" height="42" rx="12" fill="#C7F1ED"/>
                  <rect x="320" y="132" width="66" height="42" rx="12" fill="#D9D0CC"/>
                </svg>
                """
            ),
        },
    ]
    return visuals


def render_feature_visuals():
    cols = st.columns(3, gap="large")
    for col, card in zip(cols, build_feature_visuals()):
        with col:
            st.markdown(
                f"""
                <div class="visual-card">
                  <img src="{card["image"]}" alt="{card["title"]}">
                  <div class="body">
                    <div class="title">{card["title"]}</div>
                    <div class="copy">{card["copy"]}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def pick_primary_genre(frame, genre_col, weight_col):
    valid = frame[[genre_col, weight_col]].copy()
    valid[genre_col] = valid[genre_col].fillna("Unknown").replace("", "Unknown")
    valid = valid[valid[genre_col] != "Unknown"]
    if valid.empty:
        return "Unknown"
    return valid.groupby(genre_col)[weight_col].sum().sort_values(ascending=False).index[0]


@st.cache_data(show_spinner=False)
def compute_dashboard_insights(_music, _listen, _X_content, _collab_model, sample_users=60, top_n=8):
    from recommender import recommend_for_user

    genre_lookup = (
        _music[["song_id", "genre"]]
        .drop_duplicates("song_id")
        .assign(genre=lambda df: df["genre"].fillna("Unknown").replace("", "Unknown"))
        .set_index("song_id")["genre"]
    )

    active_users = (
        _listen.groupby("user_id")["play_count"]
        .sum()
        .sort_values(ascending=False)
        .head(sample_users)
        .index.tolist()
    )

    eval_rows = []
    for user_id in active_users:
        user_hist = _listen[_listen["user_id"] == user_id][["song_id", "play_count"]].copy()
        user_hist["genre"] = user_hist["song_id"].map(genre_lookup)
        actual_genre = pick_primary_genre(user_hist, "genre", "play_count")

        recs = recommend_for_user(
            user_id=user_id,
            music=_music,
            listen=_listen,
            X_content=_X_content,
            collab_model=_collab_model,
            top_n=top_n,
        )
        pred_genre = pick_primary_genre(recs, "genre", "final_score") if not recs.empty else "Unknown"
        if actual_genre != "Unknown" and pred_genre != "Unknown":
            eval_rows.append({"user_id": user_id, "actual": actual_genre, "predicted": pred_genre})

    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        matrix_df = pd.DataFrame([[0]], index=["No data"], columns=["No data"])
        genre_accuracy = 0.0
        predicted_mix = pd.DataFrame({"Recommended users": [0]}, index=["No data"])
    else:
        core_labels = eval_df["actual"].value_counts().head(6).index.tolist()
        other_needed = any(
            (eval_df["actual"].isin(core_labels) == False) | (eval_df["predicted"].isin(core_labels) == False)
        )

        eval_df["actual_group"] = np.where(eval_df["actual"].isin(core_labels), eval_df["actual"], "Other")
        eval_df["pred_group"] = np.where(eval_df["predicted"].isin(core_labels), eval_df["predicted"], "Other")

        labels = core_labels + (["Other"] if other_needed else [])
        cm = confusion_matrix(eval_df["actual_group"], eval_df["pred_group"], labels=labels)
        matrix_df = pd.DataFrame(cm, index=labels, columns=labels)
        matrix_df = matrix_df.div(matrix_df.sum(axis=1).replace(0, 1), axis=0).round(2)
        genre_accuracy = float((eval_df["actual_group"] == eval_df["pred_group"]).mean())
        predicted_mix = eval_df["pred_group"].value_counts().rename("Recommended users").to_frame()

    year_series = pd.to_numeric(_music.get("year"), errors="coerce")
    release_trend = (
        pd.DataFrame({"year": year_series})
        .dropna()
        .loc[lambda df: df["year"].between(1950, 2025, inclusive="both")]
        .assign(year=lambda df: df["year"].astype(int))
        .groupby("year")
        .size()
        .rename("Tracks")
        .to_frame()
    )

    feature_cols = [col for col in ["danceability", "energy", "valence"] if col in _music.columns]
    top_genres = _music["genre"].fillna("Unknown").value_counts().head(6).index.tolist()
    if feature_cols:
        genre_features = (
            _music[_music["genre"].fillna("Unknown").isin(top_genres)]
            .groupby("genre")[feature_cols]
            .mean()
            .sort_values(feature_cols[0], ascending=False)
            .round(3)
        )
    else:
        genre_features = pd.DataFrame({"No audio features": [0]}, index=["Unavailable"])

    return {
        "matrix": matrix_df,
        "genre_accuracy": genre_accuracy,
        "evaluation_count": len(eval_df),
        "predicted_mix": predicted_mix,
        "release_trend": release_trend,
        "genre_features": genre_features,
    }
 
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Playfair Display,serif;font-size:1.6rem;"
        "font-weight:800;color:#fff;margin-bottom:2px'>Melodia</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.8rem;color:rgba(255,255,255,0.80);"
        "margin-bottom:18px;font-family:DM Sans,sans-serif'>"
        "Music recommendation dashboard</div>",
        unsafe_allow_html=True,
    )
 
    for label in ["Overview", "Users", "Recommendations"]:
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
        f"<div style='font-size:0.72rem;color:rgba(255,255,255,0.60);"
        f"font-family:DM Sans,sans-serif;line-height:1.6'>"
        f"CSC 6740 · Data Mining<br>Team: Data Miners<br>"
        f"{N_TOTAL:,} users · {N_CATALOG:,} tracks</div>",
        unsafe_allow_html=True,
    )
 
 
if st.session_state.page == "Overview":

    st.markdown("# Melodia")
    st.markdown(
        "<p style='font-size:1.05rem;color:#7A6660;margin-top:-10px;margin-bottom:24px'>"
        "Personalized music recommendations informed by listening behavior, "
        "audio similarity, and discovery balance.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-panel">
          <span class="hero-kicker">Frontend insights</span>
          <div style="margin-top:14px;font-family:'Playfair Display',serif;font-size:1.9rem;color:#2D2420;font-weight:800;line-height:1.15">
            Visual storytelling for the recommender
          </div>
          <div style="margin-top:10px;color:#7A6660;font-size:0.95rem;max-width:760px;line-height:1.7">
            The dashboard now includes artwork-style visuals, a recommendation genre confusion matrix,
            and multiple graphs so the frontend explains both the data and the model behavior.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div class="data-note">
        <b>Current dataset in use</b> — {N_TOTAL:,} users, {N_CATALOG:,} tracks, and {N_EVENTS:,} listening events are loaded in this app session.
        </div>""",
        unsafe_allow_html=True,
    )

    metric_row([
        (f"{N_TOTAL:,}",   "Users in listening history"),
        (f"{N_MODEL:,}",   "Users modeled"),
        (f"{N_CATALOG:,}", "Tracks in catalog"),
        (f"{N_EVENTS:,}",  "Listening events"),
    ])

    render_feature_visuals()
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("### Recommendation methodology")
        st.markdown("""
<div style='background:#fff;border:1px solid #E6E0DC;border-radius:14px;
            padding:24px 22px;font-family:"DM Sans",sans-serif;font-size:0.92rem;color:#4A4A4A'>
  <div style='margin-bottom:16px'>
    <strong>User behavior</strong><br>
    The system identifies patterns from listener history to surface tracks aligned with taste.
  </div>
  <div style='margin-bottom:16px'>
    <strong>Audio similarity</strong><br>
    Tracks are matched by sound characteristics such as energy, mood, and tempo.
  </div>
  <div style='margin-bottom:16px'>
    <strong>Discovery balance</strong><br>
    Recommendations promote familiarity while preserving fresh, diverse selections.
  </div>
  <div style='color:#7A6660;font-size:0.88rem'>
    Weights adapt based on user engagement depth, and diversity-aware ranking improves recommendation quality.
  </div>
</div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### Data usage")
        st.markdown(f"""
| Metric | Current value |
|---|---|
| Users in listening history | **{N_TOTAL:,}** |
| Users in CF model | **{N_MODEL:,}** |
| Tracks in catalog | **{N_CATALOG:,}** |
| Listening events | **{N_EVENTS:,}** |
        """)

    st.markdown("---")
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("### User activity distribution")
        user_plays = listen.groupby("user_id")["play_count"].sum()
        bins   = [0,10,20,50,100,250,500,99999]
        labels = ["1-10","11-20","21-50","51-100","101-250","251-500","500+"]
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

    with st.spinner("Building confusion matrix and graph insights..."):
        insights = compute_dashboard_insights(music, listen, X_content, collab_model)

    st.markdown("---")
    st.markdown("### Recommendation alignment")
    cm_col, meta_col = st.columns([1.8, 1], gap="large")

    with cm_col:
        st.markdown("**Dominant listening genre vs. dominant recommended genre**")
        st.dataframe(
            insights["matrix"].style.background_gradient(cmap="OrRd", axis=None).format("{:.0%}"),
            use_container_width=True,
        )
        st.caption("Rows show each sampled user's main listening genre. Columns show the strongest genre returned by the top recommendations.")

    with meta_col:
        st.markdown(
            """
            <div class="cm-note">
              <div style="font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:700;color:#2D2420">
                Model snapshot
              </div>
              <div style="margin-top:12px;color:#7A6660;font-size:0.9rem;line-height:1.7">
                This matrix uses the most active users in the listening history to show how closely the recommendation output follows each listener's strongest genre signal.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Genre match rate", f"{insights['genre_accuracy']:.0%}")
        st.metric("Users evaluated", f"{insights['evaluation_count']:,}")
        st.metric("Genres shown", f"{len(insights['matrix'].columns)}")

    st.markdown("---")
    st.markdown("### Additional graphs")
    g1, g2, g3 = st.columns(3, gap="large")

    with g1:
        st.markdown("**Catalog release trend**")
        st.line_chart(insights["release_trend"], color="#FF8C5A")

    with g2:
        st.markdown("**Average audio profile by genre**")
        st.bar_chart(insights["genre_features"], color=["#FF6B4A", "#FFB347", "#1DADA8"])

    with g3:
        st.markdown("**Top recommended genres in evaluation sample**")
        st.bar_chart(insights["predicted_mix"], color="#1DADA8")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;padding:20px;background:#FFF0E6;border-radius:14px;"
        "border:1.5px solid #FFD5C8'>"
        "<span style='font-size:1rem;color:#7A6660;font-family:DM Sans,sans-serif'>"
        "Use the Users page to select a user and view their recommendations."
        "</span></div>",
        unsafe_allow_html=True,
    )
 
 

elif st.session_state.page == "Users":

    st.markdown("# Users")
    st.markdown(
        f"<p style='color:#7A6660;margin-top:-10px'>"
        f"<b style='color:#FF6B4A'>{N_TOTAL:,} users</b> in listening history · "
        f"<b style='color:#FF6B4A'>{N_MODEL:,}</b> in the CF model · "
        f"click any row then press <b>View Recommendations</b></p>",
        unsafe_allow_html=True,
    )

    # ── Quick segment buttons ─────────────────────────────────────────────────
    if "user_segment" not in st.session_state:
        st.session_state.user_segment = "All users"

    seg_options = ["All users", "Top 50", "Heavy (100+ plays)", "Moderate (20–99)", "Light (< 20)"]
    seg_cols = st.columns(len(seg_options))
    for col, label in zip(seg_cols, seg_options):
        with col:
            active = st.session_state.user_segment == label
            if st.button(label, key=f"seg_{label}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.user_segment = label
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters row ───────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4, fc5 = st.columns([3, 1, 1, 1, 1])
    with fc1:
        search_q = st.text_input(
            "Search", placeholder="🔍  Search by user ID…",
            label_visibility="collapsed",
        )
    with fc2:
        min_plays = st.selectbox("Min plays", [0, 10, 20, 50, 100, 250],
                                 label_visibility="collapsed")
    with fc3:
        model_filter = st.selectbox("Status", ["All", "In CF model", "Cold start only"],
                                    label_visibility="collapsed")
    with fc4:
        sort_by = st.selectbox("Sort by", ["Total plays", "Unique songs", "Avg plays/song"],
                               label_visibility="collapsed")
    with fc5:
        page_size = st.selectbox("Show", [25, 50, 100, 200, 500], index=1,
                                  label_visibility="collapsed")

    # ── Build enriched table ──────────────────────────────────────────────────
    tbl = user_table_full.copy()
    tbl["avg_plays"] = (tbl["total_plays"] / tbl["songs_played"].replace(0, 1)).round(1)

    @st.cache_data(show_spinner=False)
    def get_user_top_genres(_listen, _music):
        genre_map = _music.set_index("song_id")["genre"].to_dict()
        sample_uids = _listen["user_id"].unique()[:3000]
        result = {}
        for uid in sample_uids:
            songs = _listen[_listen["user_id"] == uid]
            genres = songs["song_id"].map(genre_map).dropna()
            result[uid] = genres.mode().iloc[0] if len(genres) > 0 else "—"
        return result

    genre_map_data = get_user_top_genres(listen, music)
    tbl["top_genre"] = tbl["user_id"].map(genre_map_data).fillna("—")

    # Apply segment
    seg = st.session_state.user_segment
    if seg == "Top 50":
        tbl = tbl.head(50)
    elif seg == "Heavy (100+ plays)":
        tbl = tbl[tbl["total_plays"] >= 100]
    elif seg == "Moderate (20–99)":
        tbl = tbl[(tbl["total_plays"] >= 20) & (tbl["total_plays"] < 100)]
    elif seg == "Light (< 20)":
        tbl = tbl[tbl["total_plays"] < 20]

    # Apply search + filters
    if search_q.strip():
        tbl = tbl[tbl["user_id"].str.contains(search_q.strip(), case=False, na=False)]
    if min_plays > 0:
        tbl = tbl[tbl["total_plays"] >= min_plays]
    if model_filter == "In CF model":
        tbl = tbl[tbl["in_model"]]
    elif model_filter == "Cold start only":
        tbl = tbl[~tbl["in_model"]]

    sort_col_map = {"Total plays": "total_plays", "Unique songs": "songs_played", "Avg plays/song": "avg_plays"}
    tbl = tbl.sort_values(sort_col_map[sort_by], ascending=False)
    tbl_page = tbl.head(page_size).reset_index(drop=True)

    # ── Stats strip ───────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    in_cf_pct = tbl["in_model"].mean() * 100 if len(tbl) > 0 else 0
    med_plays = int(tbl["total_plays"].median()) if len(tbl) > 0 else 0
    med_songs = int(tbl["songs_played"].median()) if len(tbl) > 0 else 0

    for col, (val, lbl) in zip([s1, s2, s3, s4], [
        (f"{len(tbl):,}", "Matching users"),
        (f"{med_plays:,}", "Median total plays"),
        (f"{med_songs:,}", "Median unique songs"),
        (f"{in_cf_pct:.0f}%", "In CF model"),
    ]):
        with col:
            st.markdown(
                f'<div class="mcard" style="padding:12px 14px">'
                f'<div class="val" style="font-size:1.4rem">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"Showing {len(tbl_page):,} of {len(tbl):,} users · sorted by {sort_by}")

    # ── HTML user table — fully cream, zero canvas black ────────────────────
    # st.dataframe uses glide-data-grid canvas which ignores CSS vars in
    # this Streamlit version. We render a native HTML table instead and
    # pair it with a hidden selectbox for row selection.

    # Build row HTML
    rows_html = ""
    for idx, row in tbl_page.iterrows():
        cf_badge = (
            '<span style="color:#1DADA8;font-weight:700">✅</span>'
            if row["in_model"] else
            '<span style="color:#FFB347;font-weight:700">⚠️</span>'
        )
        genre_str = str(row.get("top_genre", "—"))
        rows_html += f"""
        <tr onclick="selectUser({idx})" id="row-{idx}"
            style="cursor:pointer;transition:background 0.12s"
            onmouseover="this.style.background='#FFE8D8'"
            onmouseout="this.style.background='{('#FFF3EC' if idx % 2 == 1 else '#FFF8F3')}'">
          <td style="color:#B09890;width:36px;padding:10px 8px 10px 14px;font-size:0.78rem">{idx}</td>
          <td style="font-family:monospace;font-size:0.82rem;color:#2D2420;padding:10px 12px;max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{row['short_id']}</td>
          <td style="text-align:right;padding:10px 16px;color:#2D2420;font-size:0.88rem">{int(row['songs_played'])}</td>
          <td style="text-align:right;padding:10px 16px;color:#2D2420;font-size:0.88rem;font-weight:600">{int(row['total_plays'])}</td>
          <td style="text-align:right;padding:10px 16px;color:#7A6660;font-size:0.85rem">{float(row['avg_plays']):.1f}</td>
          <td style="padding:10px 14px;color:#2D2420;font-size:0.85rem">{genre_str}</td>
          <td style="text-align:center;padding:10px 14px">{cf_badge}</td>
        </tr>"""

    table_html = f"""
    <style>
    #user-table-wrap {{
        width:100%; overflow-x:auto;
        border:1px solid rgba(210,185,175,0.5);
        border-radius:12px; overflow:hidden;
        background:#FFF8F3;
    }}
    #user-table {{
        width:100%; border-collapse:collapse;
        font-family:'DM Sans',sans-serif;
        background:#FFF8F3;
    }}
    #user-table thead tr {{
        background:#FDEBD7;
        border-bottom:1px solid rgba(210,185,175,0.6);
    }}
    #user-table thead th {{
        padding:10px 14px; text-align:left;
        font-size:0.75rem; font-weight:600;
        color:#5C3D30; text-transform:uppercase;
        letter-spacing:0.06em; white-space:nowrap;
    }}
    #user-table thead th.num {{ text-align:right; }}
    #user-table tbody tr:nth-child(even) {{ background:#FFF3EC; }}
    #user-table tbody tr:nth-child(odd)  {{ background:#FFF8F3; }}
    #user-table tbody tr.selected-row    {{ background:#FFE0CC !important; }}
    #user-table tbody td {{
        border-bottom:1px solid rgba(210,185,175,0.25);
    }}
    </style>
    <div id="user-table-wrap">
      <table id="user-table">
        <thead><tr>
          <th style="width:36px">#</th>
          <th>User ID</th>
          <th class="num">Songs</th>
          <th class="num">Total plays</th>
          <th class="num">Avg / song</th>
          <th>Top genre</th>
          <th style="text-align:center">CF</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # ── Row selector — compact selectbox below the table ─────────────────────
    st.markdown(
        "<div style='margin-top:10px;font-size:0.82rem;color:#7A6660'>"
        "Select a user from the list:</div>",
        unsafe_allow_html=True,
    )
    select_options = ["— click to choose —"] + [
        f"#{i}  {row['short_id']}  ({int(row['total_plays'])} plays)"
        for i, row in tbl_page.iterrows()
    ]
    chosen = st.selectbox(
        "Choose user", select_options, label_visibility="collapsed"
    )

    # Map chosen string back to row
    selected_idx = None
    if chosen != "— click to choose —":
        try:
            # extract the index from "#N  ..."
            selected_idx = int(chosen.split("  ")[0].lstrip("#"))
        except Exception:
            selected_idx = None

    # ── Row click → enriched banner ───────────────────────────────────────────
    def show_uid_actions(full_uid, row_data):
        top_preview = get_top_tracks(listen, music, full_uid, n=5)
        top_artists = " · ".join(top_preview["artist"].dropna().unique()[:3]) or "—"
        genre_val = row_data.get("top_genre", "—")
        avg_val   = float(row_data.get("avg_plays", 0))

        st.markdown(
            f"""<div class="uid-banner">
              <div style="flex:1;min-width:0">
                <b style="font-size:0.74rem;text-transform:uppercase;letter-spacing:0.06em">Selected user</b><br>
                <span style="font-size:0.88rem;font-family:monospace;word-break:break-all">{full_uid}</span>
              </div>
              <div style="display:flex;gap:20px;flex-wrap:wrap;font-size:0.81rem;color:#7A6660;align-items:center">
                <div style="text-align:center"><b style="color:#FF6B4A;font-size:1.05rem;display:block">{int(row_data["songs_played"])}</b>songs</div>
                <div style="text-align:center"><b style="color:#FF6B4A;font-size:1.05rem;display:block">{int(row_data["total_plays"])}</b>plays</div>
                <div style="text-align:center"><b style="color:#FF6B4A;font-size:1.05rem;display:block">{avg_val:.1f}×</b>avg/song</div>
                <div style="text-align:center"><b style="color:#2D2420;display:block">{genre_val}</b>top genre</div>
                <div style="max-width:180px"><b style="color:#2D2420;display:block">Top artists</b>{top_artists}</div>
                <div>{"✅ CF model" if row_data["in_model"] else "⚠️ Cold start"}</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        b1, b2, _ = st.columns([1.3, 2.0, 2.7])
        with b1:
            if st.button("▶ View recommendations", type="primary", use_container_width=True):
                st.session_state.page = "Recommendations"
                st.rerun()
        with b2:
            st.code(full_uid, language=None)

    if selected_idx is not None and selected_idx in tbl_page.index:
        full_uid = tbl_page.loc[selected_idx, "user_id"]
        row_data = tbl_page.loc[selected_idx]
        st.session_state.selected_uid = full_uid
        show_uid_actions(full_uid, row_data)
    elif st.session_state.selected_uid:
        uid  = st.session_state.selected_uid
        prev = user_table_full[user_table_full["user_id"] == uid]
        if not prev.empty:
            r = prev.iloc[0].copy()
            r["avg_plays"]  = round(r["total_plays"] / max(r["songs_played"], 1), 1)
            r["top_genre"]  = genre_map_data.get(uid, "—")
            show_uid_actions(uid, r)
        else:
            st.markdown(f'<div class="uid-banner"><b>Previously selected:</b>&nbsp;{uid}</div>',
                        unsafe_allow_html=True)
            if st.button("▶ View recommendations", type="primary"):
                st.session_state.page = "Recommendations"
                st.rerun()
    else:
        st.markdown(
            "<div style='color:#7A6660;font-size:0.87rem;padding:14px 0'>"
            "← Click any row above to select a user</div>",
            unsafe_allow_html=True,
        )


elif st.session_state.page == "Recommendations":

    st.markdown("# Recommendations")
    st.markdown(
        "<p style='color:#7A6660;margin-top:-10px'>"
        "Personalized track suggestions powered by collaborative filtering, "
        "audio similarity, and novelty re-ranking.</p>",
        unsafe_allow_html=True,
    )

    # ── Top controls row ──────────────────────────────────────────────────────
    ic1, ic2, ic3 = st.columns([3, 1, 1])
    with ic1:
        uid_input = st.text_input(
            "User ID",
            value=st.session_state.selected_uid,
            placeholder="Paste a user ID, or go to the Users page and click a row",
        )
        st.session_state.selected_uid = uid_input
    with ic2:
        top_n = st.selectbox("# Recommendations", [5, 10, 15, 20], index=1)
    with ic3:
        show_audio = st.toggle(
            "Audio preview", value=False,
            help="Show Spotify preview player per track (where available)",
        )

    # ── Quick-pick — always visible, no expander ──────────────────────────────
    st.markdown(
        "<div style='font-size:0.82rem;color:#7A6660;margin:6px 0 4px'>"
        "Or pick from the most active users:</div>",
        unsafe_allow_html=True,
    )
    top50   = user_table_full.head(50)
    qp_opts = top50.apply(
        lambda r: f"{r['user_id'][:24]}…  |  {int(r['songs_played'])} songs  |  {int(r['total_plays'])} plays",
        axis=1,
    ).tolist()
    qp_ids  = top50["user_id"].tolist()
    qp1, qp2 = st.columns([3, 1])
    with qp1:
        picked = st.selectbox("Quick pick", ["— select —"] + qp_opts,
                              label_visibility="collapsed")
    with qp2:
        if st.button("Use ↗", use_container_width=True) and picked != "— select —":
            st.session_state.selected_uid = qp_ids[qp_opts.index(picked)]
            st.rerun()

    st.markdown("---")

    uid = uid_input.strip()
    if not uid:
        st.markdown(
            "<div style='text-align:center;padding:52px;background:#FFF0E6;"
            "border-radius:16px;border:1.5px solid #FFD5C8'>"
            "<div style='font-size:2.8rem'>🎵</div>"
            "<div style='color:#7A6660;margin-top:12px;font-size:1rem'>"
            "Enter a user ID above, or go to the <b>Users</b> page and click any row."
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.stop()

    if uid not in set(listen["user_id"].unique()):
        st.error(f"❌ User `{uid}` not found. Go to the Users page for valid IDs.")
        st.stop()

    # ── User summary strip ────────────────────────────────────────────────────
    user_hist = listen[listen["user_id"] == uid]
    n_u_songs = int(user_hist["song_id"].nunique())
    n_u_plays = int(user_hist["play_count"].sum())
    in_model  = uid in model_user_set
    avg_song  = round(n_u_plays / max(n_u_songs, 1), 1)

    metric_row([
        (n_u_songs,      "Unique songs"),
        (n_u_plays,      "Total plays"),
        (f"{avg_song}×", "Avg plays/song"),
        ("CF + Content" if in_model else "Content only", "Mode"),
        ("✅ CF model" if in_model else "⚠️ Cold start", "Status"),
    ])

    if not in_model:
        st.info(
            "ℹ️ Limited history — using content features and popularity. "
            "Listen to more tracks to activate collaborative filtering."
        )

    # ── Run recommendations ───────────────────────────────────────────────────
    with st.spinner("🎵 Finding your perfect tracks…"):
        recs = run_recs(uid, music, listen, X_content, collab_model, top_n)

    # ── Weight regime visual badges ───────────────────────────────────────────
    if not recs.empty:
        import re as _re
        regime_raw = recs["weight_regime"].iloc[0]
        cf_m  = _re.search(r"CF=(\d+)%",      regime_raw)
        cnt_m = _re.search(r"Content=(\d+)%", regime_raw)
        nov_m = _re.search(r"Novelty=(\d+)%", regime_raw)
        cf_v  = int(cf_m.group(1))  if cf_m  else 0
        cnt_v = int(cnt_m.group(1)) if cnt_m else 0
        nov_v = int(nov_m.group(1)) if nov_m else 0

        st.markdown(
            f"""<div style="display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap">
              <div style="background:#FFF0E6;border:1px solid #FFD5C8;border-radius:20px;
                          padding:5px 14px;font-size:0.8rem;color:#FF6B4A;font-weight:600">
                Collaborative {cf_v}%</div>
              <div style="background:#E8FAF8;border:1px solid #1DADA8;border-radius:20px;
                          padding:5px 14px;font-size:0.8rem;color:#1DADA8;font-weight:600">
                Content {cnt_v}%</div>
              <div style="background:#FFF8EE;border:1px solid #FFB347;border-radius:20px;
                          padding:5px 14px;font-size:0.8rem;color:#B07020;font-weight:600">
                Novelty {nov_v}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_reco, tab_hist, tab_sig = st.tabs(
        ["🎵 Recommendations", "📚 Listening History", "📊 Signal Breakdown"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 1: RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    with tab_reco:
        if recs.empty:
            st.warning("No recommendations could be generated for this user.")
        else:
            for i, row in recs.iterrows():
                score_pct = int(row["final_score"] * 100)
                nov_score = float(row.get("novelty_score", 0))

                # Novelty / familiarity badge
                if nov_score >= 0.65:
                    badge_html = (
                        '<span style="background:#E8FAF8;border:1px solid #1DADA8;'
                        'border-radius:20px;padding:2px 9px;font-size:0.68rem;'
                        'color:#1DADA8;font-weight:600;margin-left:8px">✦ NEW FIND</span>'
                    )
                elif nov_score <= 0.25:
                    badge_html = (
                        '<span style="background:#FFF0E6;border:1px solid #FFD5C8;'
                        'border-radius:20px;padding:2px 9px;font-size:0.68rem;'
                        'color:#FF8C5A;font-weight:600;margin-left:8px">♥ FAMILIAR</span>'
                    )
                else:
                    badge_html = ""

                genre_str = str(row.get("genre", ""))
                genre_pill = (
                    f'<span class="pill">{genre_str}</span>'
                    if genre_str not in ("", "Unknown", "nan") else ""
                )
                year_val = row.get("year", 0)
                year_pill = (
                    f'<span class="pill">{int(year_val)}</span>'
                    if year_val and int(year_val) > 1000 else ""
                )

                st.markdown(
                    f"""<div class="rec-card">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start">
                        <div style="flex:1;min-width:0">
                          <span class="rec-rank">#{i+1}</span>&nbsp;&nbsp;
                          <span class="rec-title">{row['title']}</span>
                          {badge_html}<br>
                          <span class="rec-artist">by {row['artist']}</span>
                          &nbsp;{genre_pill}{year_pill}
                        </div>
                        <div style="text-align:right;min-width:72px;flex-shrink:0">
                          <span style="color:#FF6B4A;font-family:'Playfair Display',serif;
                                       font-weight:800;font-size:1.2rem">{score_pct}%</span>
                          <div style="color:#7A6660;font-size:0.68rem">match score</div>
                        </div>
                      </div>
                      <div class="bar-bg">
                        <div class="bar-fill" style="width:{score_pct}%"></div>
                      </div>
                      <div class="rec-explain">✦ {row['explanation']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                if show_audio:
                    audio_src = get_preview_source(
                        row.get("song_id"), row.get("spotify_preview_url", "")
                    )
                    if audio_src:
                        st.audio(audio_src)

            st.markdown("<br>", unsafe_allow_html=True)
            csv_out = recs[["title", "artist", "genre", "year",
                             "final_score", "explanation"]].to_csv(index=False)
            st.download_button(
                "⬇️ Download recommendations as CSV",
                data=csv_out,
                file_name=f"recs_{uid[:12]}.csv",
                mime="text/csv",
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 2: LISTENING HISTORY
    # ─────────────────────────────────────────────────────────────────────────
    with tab_hist:
        top_tracks = get_top_tracks(listen, music, uid, n=20)
        if top_tracks.empty:
            st.info("No listening history found.")
        else:
            # Genre taste as proportion of total plays (not just count)
            user_genres = (
                top_tracks.groupby("genre")["play_count"]
                .sum()
                .sort_values(ascending=False)
                .head(6)
            )
            total_genre_plays = max(user_genres.sum(), 1)

            if not user_genres.empty:
                st.markdown("**Genre taste by play volume:**")
                gcols = st.columns(min(len(user_genres), 6))
                for i, (g, cnt) in enumerate(user_genres.items()):
                    pct = int(cnt / total_genre_plays * 100)
                    with gcols[i]:
                        st.markdown(
                            f'<div class="mcard" style="padding:12px 10px">'
                            f'<div class="val" style="font-size:1.2rem">{pct}%</div>'
                            f'<div style="background:#FFF0E6;border-radius:4px;height:4px;margin:6px 0">'
                            f'<div style="background:#FF6B4A;height:4px;border-radius:4px;width:{pct}%"></div></div>'
                            f'<div class="lbl" style="font-size:0.68rem">{g}</div></div>',
                            unsafe_allow_html=True,
                        )
                st.markdown("<br>", unsafe_allow_html=True)

            # Top tracks as styled rows with play bar
            st.markdown("**Most played tracks:**")
            max_plays = int(top_tracks["play_count"].max())
            for _, row in top_tracks.head(10).iterrows():
                bar_w = int(row["play_count"] / max(max_plays, 1) * 100)
                genre_str = str(row.get("genre", ""))
                genre_tag = (
                    f'<span class="pill">{genre_str}</span>'
                    if genre_str not in ("", "Unknown", "nan") else ""
                )
                year_val = row.get("year", 0)
                yr_tag = (
                    f'<span class="pill">{int(year_val)}</span>'
                    if year_val and int(year_val) > 1000 else ""
                )
                st.markdown(
                    f"""<div style="background:#FFFFFF;border:1px solid #FFD5C8;
                                    border-radius:10px;padding:10px 14px;margin-bottom:8px">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                          <span style="font-weight:600;font-size:0.92rem;color:#2D2420">{row['title']}</span><br>
                          <span style="color:#7A6660;font-size:0.82rem">{row['artist']}</span>
                          &nbsp;{genre_tag}{yr_tag}
                        </div>
                        <div style="text-align:right;min-width:60px;flex-shrink:0">
                          <span style="color:#FF6B4A;font-weight:700">{int(row['play_count'])}</span>
                          <span style="color:#B09890;font-size:0.72rem"> plays</span>
                        </div>
                      </div>
                      <div style="background:#FDEBD7;border-radius:3px;height:3px;margin-top:8px">
                        <div style="background:linear-gradient(90deg,#FF6B4A,#FFB347);
                                    height:3px;border-radius:3px;width:{bar_w}%"></div>
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ─────────────────────────────────────────────────────────────────────────
    #  TAB 3: SIGNAL BREAKDOWN
    # ─────────────────────────────────────────────────────────────────────────
    with tab_sig:
        if recs.empty:
            st.info("No recommendations to analyse.")
        else:
            st.caption(
                "CF = collaborative filter  ·  "
                "Content = audio feature similarity  ·  "
                "Novelty = how fresh the track is (higher = more discovery)"
            )

            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                st.metric("Avg CF score",      f"{recs['cf_score'].mean():.3f}",
                          help="Collaborative filtering contribution")
            with pc2:
                st.metric("Avg Content score", f"{recs['content_score'].mean():.3f}",
                          help="Audio feature similarity contribution")
            with pc3:
                st.metric("Avg Novelty score", f"{recs['novelty_score'].mean():.3f}",
                          help="Discovery score — higher means more unfamiliar tracks")

            st.markdown("---")
            st.markdown("**Signal contribution per recommendation:**")
            chart_df = recs[["title", "cf_score", "content_score", "novelty_score"]].copy()
            chart_df["title"] = chart_df["title"].str[:28]
            chart_df = chart_df.set_index("title")
            chart_df.columns = ["CF", "Content", "Novelty"]
            st.bar_chart(chart_df, color=["#FF6B4A", "#1DADA8", "#FFB347"])

            st.markdown("---")
            extra_left, extra_right = st.columns(2, gap="large")
            with extra_left:
                st.markdown("**Match score by rank**")
                rank_df = recs[["final_score"]].copy()
                rank_df.index = [f"#{j+1}" for j in range(len(rank_df))]
                rank_df.columns = ["Match score"]
                st.line_chart(rank_df, color="#FF6B4A")

            with extra_right:
                st.markdown("**Genre mix in recommendations**")
                gm = recs["genre"].fillna("Unknown").value_counts().rename("Tracks").to_frame()
                st.bar_chart(gm, color="#FFB347")

            st.markdown("---")
            st.markdown("**Full score table:**")
            sig_df = recs[["title", "artist", "cf_score",
                            "content_score", "novelty_score", "final_score"]].copy()
            sig_df.columns = ["Track", "Artist", "CF", "Content", "Novelty", "Final"]
            st.dataframe(sig_df.round(3), use_container_width=True, hide_index=True)
