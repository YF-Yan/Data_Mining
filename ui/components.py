"""
RFM 指标在界面上的展示文案（不展示具体历史日期）。
"""

from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from core.periods import PERIOD_BY_KEY, TRAINING_PERIODS
from core.segments import get_segment_catalog, profiles_for_period_table


def get_period_label(meta: dict) -> str:
    """当前模型对应的统计周期名称，如「近 1 年」。"""
    return meta.get("period_label") or PERIOD_BY_KEY.get(
        meta.get("period_key", ""), {}
    ).get("label", "近 1 年")


def period_options_from_manifest(manifest: Optional[dict]) -> List[Dict]:
    """从 manifest 或默认配置得到下拉选项。"""
    if manifest and manifest.get("periods"):
        return manifest["periods"]
    return [{"key": p["key"], "label": p["label"], "days": p["days"]} for p in TRAINING_PERIODS]


def label_for_period_key(period_key: str) -> str:
    return PERIOD_BY_KEY.get(period_key, {}).get("label", period_key)


def render_statistics_period_notice(meta: dict) -> str:
    """页面顶部统计周期说明。"""
    label = get_period_label(meta)
    return (
        f"**当前统计周期：{label}**  \n"
        "购买次数、消费金额为该时段内累计；最近消费天数为距分析截止日的天数。"
        "切换周期将使用对应模型判别，请按相同窗口汇总 RFM 后填写。"
    )


def label_frequency(meta: dict) -> str:
    return f"购买次数（{get_period_label(meta)}内）"


def label_monetary(meta: dict) -> str:
    return f"消费金额（{get_period_label(meta)}内，元）"


def help_frequency(meta: dict) -> str:
    label = get_period_label(meta)
    return (
        f"在 **{label}** 内，该用户的独立订单笔数（按订单号去重）。"
        "须与当前所选统计周期口径一致。"
    )


def help_monetary(meta: dict) -> str:
    label = get_period_label(meta)
    return (
        f"在 **{label}** 内，该用户所有订单金额合计（数量×单价）。"
        "须与当前所选统计周期口径一致。"
    )


def help_recency() -> str:
    return (
        "距分析截止日最近一次购买的天数；越小表示越近期活跃。"
        "与购买次数/金额的统计窗口相互独立。"
    )


def training_period_note(meta: dict) -> str:
    label = get_period_label(meta)
    return (
        f"当前模型在 **{label}** 窗口下训练。"
        "请按该窗口汇总 RFM 后填入；切换周期将自动换用对应模型。"
    )


def render_segment_catalog_table(catalog: Optional[List[Dict]] = None) -> None:
    """全项目固定的「群体编号 ↔ 所属群体」对照表。"""
    catalog = catalog or get_segment_catalog()
    rows = [
        {
            "群体编号": f"第 {s['id']} 类",
            "所属群体": s["name"],
            "典型 RFM 特征": s.get("rfm_hint", ""),
            "运营建议": s.get("advice", ""),
        }
        for s in sorted(catalog, key=lambda x: x["id"])
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_period_cluster_mapping(meta: dict) -> None:
    """当前统计周期下：统一编号与 GMM 簇下标的对应关系。"""
    profiles = meta.get("segment_profiles") or {}
    if not profiles:
        st.caption("暂无本周期映射数据。")
        return
    rows = profiles_for_period_table(profiles)
    df = pd.DataFrame(rows)
    df["群体编号"] = df["群体编号"].map(lambda x: f"第 {x} 类")
    st.dataframe(df, use_container_width=True, hide_index=True)

