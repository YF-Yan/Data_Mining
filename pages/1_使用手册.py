"""
使用手册（独立页面）— 由 Streamlit 多页面自动挂载到侧栏。
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.layout import init_app_context
from ui.manual_page import render_manual_page
from ui.theme import apply_flat_theme
from utils.io import artifacts_ready

st.set_page_config(page_title="使用手册", page_icon="📖", layout="wide")
apply_flat_theme()

st.title("使用手册")

if not artifacts_ready():
    st.warning("模型尚未就绪，部分周期映射无法展示。请先完成训练。")
    st.stop()

period_key, bundle, meta, manifest = init_app_context()
if meta is None:
    st.error("无法加载模型，请重新训练后刷新页面。")
    st.stop()

render_manual_page(meta, manifest)
