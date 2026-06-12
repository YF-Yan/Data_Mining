"""GMM 聚类训练与评价指标计算。"""

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
    """训练 GMM 并返回模型与各样本簇标签。"""
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        max_iter=max_iter,
        n_init=10,
        reg_covar=1e-6,
    )
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    return gmm, labels


def evaluate_clustering(
    gmm: GaussianMixture,
    X_scaled: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """计算 BIC、AIC、轮廓系数。"""
    bic = float(gmm.bic(X_scaled))
    aic = float(gmm.aic(X_scaled))

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
    """训练 GMM、预测标签并计算评价指标。"""
    gmm, labels = fit_gmm(X_scaled, n_components, random_state=random_state)
    metrics = evaluate_clustering(gmm, X_scaled, labels)

    return gmm, labels, metrics
