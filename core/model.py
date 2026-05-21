"""
高斯混合模型（GMM）聚类模块
使用 sklearn.mixture.GaussianMixture 实现 EM 算法
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple


def fit_gmm(
    X_scaled: np.ndarray,
    n_components: int,
    random_state: int = 42,
    max_iter: int = 200,
) -> Tuple[GaussianMixture, np.ndarray]:
    """
    训练 GMM 模型并输出每个样本的簇标签。

    参数：
        X_scaled: 标准化后的特征矩阵 (n_samples, n_features)
        n_components: 聚类数 K
        random_state: 随机种子，保证结果可复现
        max_iter: EM 最大迭代次数

    返回：
        gmm: 已训练的 GaussianMixture 模型
        labels: 每个用户的聚类标签 (0 ~ K-1)
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        max_iter=max_iter,
        n_init=10,  # 多次随机初始化，降低局部最优风险
        reg_covar=1e-6,  # 协方差对角正则，防止奇异矩阵
    )
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    return gmm, labels


def evaluate_clustering(
    gmm: GaussianMixture,
    X_scaled: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    计算聚类评价指标：BIC、AIC、轮廓系数。

    BIC/AIC 越小通常表示模型与数据的拟合-复杂度权衡越好（需结合 K 比较）；
    轮廓系数越大表示簇内紧、簇间疏，取值范围 [-1, 1]。
    """
    bic = float(gmm.bic(X_scaled))
    aic = float(gmm.aic(X_scaled))

    # 轮廓系数要求 K >= 2 且每簇至少有一个样本
    n_unique = len(np.unique(labels))
    if n_unique >= 2 and len(labels) > n_unique:
        sil = float(silhouette_score(X_scaled, labels))
    else:
        sil = float("nan")

    return {
        "bic": bic,
        "aic": aic,
        "silhouette": sil,
    }


def run_clustering(
    X_scaled: np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> Tuple[GaussianMixture, np.ndarray, Dict[str, float]]:
    """
    一站式：训练 GMM + 预测标签 + 计算指标。
    """
    gmm, labels = fit_gmm(X_scaled, n_components, random_state=random_state)
    metrics = evaluate_clustering(gmm, X_scaled, labels)

    return gmm, labels, metrics
