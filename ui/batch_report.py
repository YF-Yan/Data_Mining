"""
批量判别结果汇总与可视化（整体报告页使用）。
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from ui.charts import plot_rfm_by_segment, plot_segment_counts

SESSION_KEY = "batch_report"


def save_batch_report(period_key: str, period_label: str, result_df: pd.DataFrame, file_name: str = ""):
    st.session_state[SESSION_KEY] = {
        "period_key": period_key,
        "period_label": period_label,
        "result_df": result_df,
        "file_name": file_name or "",
    }


def load_batch_report() -> Optional[dict]:
    return st.session_state.get(SESSION_KEY)


def build_batch_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    """按统一群体编号汇总批量判别结果。"""
    if "SegmentId" in result_df.columns:
        group_cols = ["SegmentId", "SegmentName"]
    else:
        group_cols = ["SegmentName"]

    summary = (
        result_df.groupby(group_cols, as_index=False)
        .agg(
            UserCount=("Recency", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            AvgConfidence=("Confidence", "mean"),
        )
        .sort_values(group_cols[0] if group_cols[0] == "SegmentId" else "SegmentName")
    )
    for col in ("AvgRecency", "AvgFrequency", "AvgMonetary", "AvgConfidence"):
        if col in summary.columns:
            summary[col] = summary[col].round(2)
    return summary


def plot_batch_pca(
    result_df: pd.DataFrame,
    feature_cols: list,
    scaler,
    clip_bounds: dict = None,
    title: str = "批量用户分群分布 (PCA 2D)",
):
    """对批量样本做与线上一致的特征变换后 PCA 可视化。"""
    from core.dataset import engineer_rfm_features

    work = result_df[feature_cols].astype(float)
    if clip_bounds:
        engineered, _ = engineer_rfm_features(
            work, clip_bounds=clip_bounds, fit_bounds=False
        )
    else:
        engineered = work
    X = scaler.transform(engineered[feature_cols].values.astype(float))

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    labels = result_df["SegmentId"].values if "SegmentId" in result_df.columns else result_df["Cluster"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.65,
        edgecolors="w",
        linewidths=0.3,
        s=36,
    )
    plt.colorbar(scatter, ax=ax, label="群体编号")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_batch_report_tab(
    period_key: str,
    period_label: str,
    bundle: dict,
) -> None:
    report = load_batch_report()

    if report is None:
        st.info("请先在 **「批量判别」** 页上传 CSV 并点击「批量判别」，完成后将在此展示汇总报告。")
        return

    if report["period_key"] != period_key:
        st.warning(
            f"当前报告来自 **{report['period_label']}** 的批量结果；"
            f"侧边栏已切换为 **{period_label}**，请重新执行批量判别以更新报告。"
        )

    df = report["result_df"]
    summary = build_batch_summary(df)

    fname = report.get("file_name") or "—"
    st.caption(f"数据来源：{fname} · 周期：**{report['period_label']}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("判别用户数", f"{len(df):,}")
    c2.metric("涉及群体数", df["SegmentName"].nunique() if "SegmentName" in df.columns else "—")
    c3.metric("平均置信度", f"{df['Confidence'].mean() * 100:.1f}%" if "Confidence" in df.columns else "—")

    st.markdown("#### 各群体汇总")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("#### 明细数据")
    st.dataframe(df, use_container_width=True, height=280)
    st.download_button(
        "下载本批结果 CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="batch_predict_result.csv",
        mime="text/csv",
        use_container_width=True,
    )

    t1, t2, t3 = st.tabs(["分群分布", "群体规模", "RFM 对比"])
    with t1:
        fig = plot_batch_pca(
            df,
            bundle["feature_cols"],
            bundle["scaler"],
            clip_bounds=bundle.get("clip_bounds"),
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with t2:
        fig = plot_segment_counts(summary, title="本批各群体人数")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with t3:
        fig = plot_rfm_by_segment(df, title="本批各群体 RFM 分布")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
