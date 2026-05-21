"""
Streamlit 多页面共用的侧边栏与模型加载。
"""

from typing import Optional, Tuple

import streamlit as st

from ui.components import get_period_label, period_options_from_manifest
from utils.io import (
    artifacts_ready,
    load_manifest,
    load_model_bundle,
    load_model_meta,
    load_segment_catalog,
    list_period_options,
)


@st.cache_resource(show_spinner=False)
def _cached_bundle_meta(period_key: str):
    return load_model_bundle(period_key=period_key), load_model_meta(period_key=period_key)


def render_period_selector_sidebar(manifest: dict) -> str:
    options = period_options_from_manifest(manifest)
    labels = [p["label"] for p in options]
    keys = [p["key"] for p in options]
    default_key = (manifest or {}).get("default_period", keys[0] if keys else "1y")
    default_idx = keys.index(default_key) if default_key in keys else 0

    selected_label = st.sidebar.selectbox(
        "统计周期",
        labels,
        index=default_idx,
        label_visibility="collapsed",
    )
    return keys[labels.index(selected_label)]


def render_sidebar_footer(meta: dict) -> None:
    st.sidebar.caption(
        f"模型 {meta.get('trained_at', '—')[:10]} · "
        f"{get_period_label(meta)} · "
        f"{meta.get('n_clusters', '—')} 类"
    )


def init_app_context() -> Tuple[Optional[str], Optional[dict], Optional[dict], Optional[dict]]:
    """
    侧边栏选择周期并加载模型。
    返回 (period_key, bundle, meta, manifest)；未就绪时 meta 为 None。
    """
    if not artifacts_ready():
        return None, None, None, None

    manifest = load_manifest() or {}
    if not list_period_options():
        return None, None, None, None

    period_key = render_period_selector_sidebar(manifest)

    try:
        bundle, meta = _cached_bundle_meta(period_key)
    except FileNotFoundError as e:
        st.sidebar.error(str(e))
        return period_key, None, None, manifest

    render_sidebar_footer(meta)
    return period_key, bundle, meta, manifest
