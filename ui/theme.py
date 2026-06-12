"""
扁平深色主题 + 大字号，供 app 与多页面共用。
"""

import streamlit as st

_FLAT_THEME_CSS = """
<style>
/* ── 全局字号 ── */
html, body, [class*="css"] {
    font-size: 20px;
}
.stApp {
    background-color: #0e1117;
}

/* ── 标题 ── */
h1 {
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}
h2 {
    font-size: 2rem !important;
    font-weight: 600 !important;
}
h3, h4 {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}

/* ── 正文 / 说明 ── */
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
    font-size: 1.15rem !important;
    line-height: 1.65 !important;
}
[data-testid="stCaptionContainer"] {
    font-size: 1.05rem !important;
}

/* ── 侧栏 ── */
[data-testid="stSidebar"] {
    background-color: #161b22;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div {
    font-size: 1.2rem !important;
}

/* ── 标签页（扁平 + 红色激活） ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #333;
}
.stTabs [data-baseweb="tab"] {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
.stTabs [aria-selected="true"] {
    color: #e53935 !important;
    border-bottom: 3px solid #e53935 !important;
}

/* ── 表单输入 ── */
.stNumberInput label,
.stTextInput label,
.stSelectbox label,
.stFileUploader label {
    font-size: 1.25rem !important;
    font-weight: 500 !important;
}
.stNumberInput input,
.stTextInput input,
.stSelectbox [data-baseweb="select"] > div {
    font-size: 1.3rem !important;
    border-radius: 0 !important;
    min-height: 2.8rem;
}

/* ── 按钮（扁平红） ── */
.stButton > button {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    padding: 0.85rem 1.6rem !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    border: none !important;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button {
    background-color: #e53935 !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button:hover {
    background-color: #c62828 !important;
    color: #fff !important;
}

/* ── 指标卡片（结果区大字） ── */
[data-testid="stMetricLabel"] {
    font-size: 1.35rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
}

/* ── 提示框（扁平无圆角） ── */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    font-size: 1.2rem !important;
    line-height: 1.65 !important;
    box-shadow: none !important;
}
[data-testid="stNotificationContent"] {
    font-size: 1.2rem !important;
}

/* ── 表格 ── */
[data-testid="stDataFrame"] {
    font-size: 1.1rem !important;
}

/* ── 下载按钮 ── */
.stDownloadButton > button {
    font-size: 1.2rem !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

/* ── 去掉多余阴影，保持扁平 ── */
div[data-testid="stForm"] {
    border: none;
    padding: 0;
}
</style>
"""


def apply_flat_theme() -> None:
    """注入全局扁平大字号样式（每个页面入口调用一次）。"""
    st.markdown(_FLAT_THEME_CSS, unsafe_allow_html=True)
