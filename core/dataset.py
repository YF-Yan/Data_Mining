"""
数据预处理模块
功能：读取 CSV、清洗交易数据、构建 RFM 特征、标准化
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple

# Frequency、Monetary 做 log1p；训练集上拟合 99 分位上界用于裁剪
LOG1P_COLS = ("Frequency", "Monetary")
CLIP_PERCENTILE = 99.0


def engineer_rfm_features(
    rfm_df: pd.DataFrame,
    clip_bounds: Optional[Dict[str, float]] = None,
    fit_bounds: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    RFM 特征工程：非负裁剪 → log1p → 上分位裁剪（与实验报告一致）。

    clip_bounds 在训练阶段拟合，预测阶段复用，保证与 scaler 同分布。
    """
    out = rfm_df.copy()
    bounds = dict(clip_bounds or {})

    for col in LOG1P_COLS:
        if col not in out.columns:
            continue
        out[col] = out[col].astype(float).clip(lower=0)
        out[col] = np.log1p(out[col])

    if fit_bounds:
        for col in LOG1P_COLS:
            if col in out.columns:
                bounds[col] = float(np.percentile(out[col], CLIP_PERCENTILE))

    for col in LOG1P_COLS:
        upper = bounds.get(col)
        if col in out.columns and upper is not None:
            out[col] = out[col].clip(upper=upper)

    return out, bounds


def load_data(file_path: str) -> pd.DataFrame:
    """
    读取 Online Retail CSV 数据文件。

    UCI 数据集常用 latin-1 编码，列名可能含空格，此处统一去除首尾空格。
    """
    df = pd.read_csv(file_path, encoding="latin-1")
    df.columns = df.columns.str.strip()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    按课程要求清洗原始交易数据：
    1. 删除 CustomerID 为空
    2. 删除 Quantity <= 0
    3. 删除 UnitPrice <= 0
    4. 删除退款订单（InvoiceNo 以 'C' 开头）
    """
    data = df.copy()

    # 删除 CustomerID 为空（含 NaN 与空字符串）
    data["CustomerID"] = pd.to_numeric(data["CustomerID"], errors="coerce")
    data = data.dropna(subset=["CustomerID"])
    data["CustomerID"] = data["CustomerID"].astype(int)

    # 删除无效数量与单价
    data = data[data["Quantity"] > 0]
    data = data[data["UnitPrice"] > 0]

    # 删除退款/取消订单
    data["InvoiceNo"] = data["InvoiceNo"].astype(str)
    data = data[~data["InvoiceNo"].str.startswith("C")]

    return data.reset_index(drop=True)


def build_rfm(
    df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    按用户聚合，构建 RFM 特征：
    - Recency：距快照日最近一次消费的天数（越小越近）
    - Frequency：独立订单数（InvoiceNo 去重计数）
    - Monetary：总消费金额（Quantity * UnitPrice 求和）

    返回列：CustomerID, Recency, Frequency, Monetary
    """
    data = df.copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    if snapshot_date is None:
        # 以数据最大日期次日为分析基准日，避免 Recency 为 0 的边界情况
        snapshot_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        data.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )

    return rfm


def standardize_features(
    rfm_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    clip_bounds: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, StandardScaler, list, Dict[str, float]]:
    """
    log1p + 百分位裁剪后，对 RFM 做 Z-score 标准化（StandardScaler）。

    返回：
        X_scaled, scaler, feature_cols, clip_bounds（供在线预测复用）
    """
    if feature_cols is None:
        feature_cols = ["Recency", "Frequency", "Monetary"]

    engineered, bounds = engineer_rfm_features(
        rfm_df,
        clip_bounds=clip_bounds,
        fit_bounds=(clip_bounds is None),
    )

    X = engineered[feature_cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, feature_cols, bounds


def filter_by_window(
    df: pd.DataFrame,
    window_days: int,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    保留分析截止日前 window_days 天内的交易（不含截止日当天）。
    """
    data = df.copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    if snapshot_date is None:
        snapshot_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)
    window_start = snapshot_date - pd.Timedelta(days=window_days)
    mask = (data["InvoiceDate"] >= window_start) & (data["InvoiceDate"] < snapshot_date)
    return data.loc[mask].reset_index(drop=True)


def preprocess_for_window(
    cleaned_df: pd.DataFrame,
    window_days: int,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, StandardScaler, list, Dict[str, float]]:
    """
    在指定统计窗口内：截取交易 -> RFM -> 特征工程 -> 标准化。
    """
    if snapshot_date is None:
        snapshot_date = pd.to_datetime(cleaned_df["InvoiceDate"]).max() + pd.Timedelta(days=1)

    windowed = filter_by_window(cleaned_df, window_days, snapshot_date)
    rfm = build_rfm(windowed, snapshot_date=snapshot_date)
    X_scaled, scaler, feature_cols, clip_bounds = standardize_features(rfm)
    return windowed, rfm, X_scaled, scaler, feature_cols, clip_bounds


'''
测试使用
'''
# def preprocess_pipeline(
#     file_path: str,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, StandardScaler, list, Dict[str, float]]:
#     """
#     完整预处理流水线：加载 -> 清洗 -> RFM -> 特征工程 -> 标准化。
#     """
#     raw = load_data(file_path)
#     cleaned = clean_data(raw)
#     rfm = build_rfm(cleaned)
#     X_scaled, scaler, feature_cols, clip_bounds = standardize_features(rfm)
#
#     return cleaned, rfm, X_scaled, scaler, feature_cols, clip_bounds
#
#
# def preprocess_from_dataframe(
#     df: pd.DataFrame,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, StandardScaler, list, Dict[str, float]]:
#     """从已加载的 DataFrame 执行预处理。"""
#     cleaned = clean_data(df)
#     rfm = build_rfm(cleaned)
#     X_scaled, scaler, feature_cols, clip_bounds = standardize_features(rfm)
#
#     return cleaned, rfm, X_scaled, scaler, feature_cols, clip_bounds
