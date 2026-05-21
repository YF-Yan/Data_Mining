"""
可视化模块
功能：PCA 2D 散点图、簇规模柱状图、RFM 分布图
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import List


# 支持中文显示（Windows 常用 SimHei / Microsoft YaHei）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_pca_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    title: str = "GMM 聚类结果 (PCA 2D)",
):
    """
    PCA 降维到 2 维，绘制聚类散点图。
    颜色表示所属簇。
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.65,
        edgecolors="w",
        linewidths=0.3,
        s=40,
    )
    plt.colorbar(scatter, ax=ax, label="簇标签")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cluster_counts(labels: np.ndarray, title: str = "各簇用户数量"):
    """
    柱状图展示每个簇的用户数量。
    """
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(unique)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [f"簇 {u}" for u in unique[order]],
        counts[order],
        color=plt.cm.viridis(np.linspace(0.2, 0.9, len(unique))),
        edgecolor="white",
    )
    ax.set_xlabel("聚类簇")
    ax.set_ylabel("用户数量")
    ax.set_title(title)
    for bar, cnt in zip(bars, counts[order]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(cnt),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_rfm_by_cluster(
    rfm_df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str] = None,
    title: str = "各簇 RFM 特征分布（箱线图）",
):
    """
    按簇展示 Recency / Frequency / Monetary 的箱线图分布。
    """
    if feature_cols is None:
        feature_cols = ["Recency", "Frequency", "Monetary"]

    plot_df = rfm_df.copy()
    plot_df["Cluster"] = labels

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    clusters = sorted(plot_df["Cluster"].unique())

    for ax, col in zip(axes, feature_cols):
        data_by_cluster = [
            plot_df.loc[plot_df["Cluster"] == c, col].values for c in clusters
        ]
        bp = ax.boxplot(
            data_by_cluster,
            labels=[f"簇 {c}" for c in clusters],
            patch_artist=True,
        )
        for patch, c in zip(bp["boxes"], clusters):
            patch.set_facecolor(plt.cm.viridis(c / max(clusters, default=1)))
            patch.set_alpha(0.7)
        ax.set_title(col)
        ax.set_ylabel("原始 RFM 值")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_rfm_means_heatmap(
    rfm_df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str] = None,
    title: str = "各簇 RFM 均值",
):
    """
    热力图展示各簇 RFM 均值，便于解读用户类型。
    """
    if feature_cols is None:
        feature_cols = ["Recency", "Frequency", "Monetary"]

    plot_df = rfm_df.copy()
    plot_df["Cluster"] = labels
    means = plot_df.groupby("Cluster")[feature_cols].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(means.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels([f"簇 {i}" for i in means.index])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="均值")
    for i in range(len(means)):
        for j in range(len(feature_cols)):
            ax.text(j, i, f"{means.values[i, j]:.1f}", ha="center", va="center")
    fig.tight_layout()
    return fig


# ---------- 客户端展示（使用训练阶段已保存的数据） ----------


def plot_pca_from_saved(
    pca_df: pd.DataFrame,
    variance_ratio: list = None,
    title: str = "用户分群分布图",
):
    """使用已保存的 PCA 坐标绘制散点图（无需 sklearn）。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    clusters = sorted(pca_df["Cluster"].unique())
    cmap = plt.cm.get_cmap("viridis", len(clusters))

    for i, cid in enumerate(clusters):
        sub = pca_df[pca_df["Cluster"] == cid]
        label = (
            sub["SegmentName"].iloc[0]
            if "SegmentName" in sub.columns
            else f"群体 {cid}"
        )
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            label=label,
            alpha=0.6,
            s=35,
            color=cmap(i),
            edgecolors="w",
            linewidths=0.2,
        )

    ax.legend(title="客户群体", loc="best", fontsize=9)
    if variance_ratio and len(variance_ratio) >= 2:
        ax.set_xlabel(f"维度1（解释方差 {variance_ratio[0]:.1%}）")
        ax.set_ylabel(f"维度2（解释方差 {variance_ratio[1]:.1%}）")
    else:
        ax.set_xlabel("维度1")
        ax.set_ylabel("维度2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


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
