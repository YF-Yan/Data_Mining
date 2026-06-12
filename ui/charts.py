"""
可视化模块（客户端批量报告用）
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    }
)


def plot_segment_counts(summary_df: pd.DataFrame, title: str = "各客户群体人数"):
    """按业务群体名称展示用户数量。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = summary_df["SegmentName"].tolist()
    counts = summary_df["UserCount"].tolist()
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))

    bars = ax.bar(names, counts, color=colors, edgecolor="white")
    ax.set_ylabel("用户数量")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{cnt:,}",
            ha="center",
            va="bottom",
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_rfm_by_segment(
    user_df: pd.DataFrame,
    feature_cols: list = None,
    title: str = "各群体消费行为分布",
):
    """按 SegmentName 绘制 RFM 箱线图。"""
    if feature_cols is None:
        feature_cols = ["Recency", "Frequency", "Monetary"]

    name_col = "SegmentName" if "SegmentName" in user_df.columns else "Cluster"
    segments = user_df[name_col].unique().tolist()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    rfm_labels = {
        "Recency": "最近消费(天)",
        "Frequency": "购买次数",
        "Monetary": "消费金额",
    }

    for ax, col in zip(axes, feature_cols):
        data = [user_df.loc[user_df[name_col] == s, col].values for s in segments]
        bp = ax.boxplot(data, labels=segments, patch_artist=True)
        for patch, idx in zip(bp["boxes"], range(len(segments))):
            patch.set_facecolor(plt.cm.viridis(idx / max(len(segments) - 1, 1)))
            patch.set_alpha(0.75)
        ax.set_title(rfm_labels.get(col, col))
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig
