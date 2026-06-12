"""加载已训练模型，对 RFM 输入做单用户或批量分群预测。"""

from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd

from core.dataset import engineer_rfm_features
from core.segments import resolve_segment_display

def build_personalized_insight(
    recency: float,
    frequency: float,
    monetary: float,
    segment_profiles: dict,
    cluster: int,
    rfm_bounds: dict = None,
) -> str:
    """根据用户 RFM 与所属簇生成个性化解读文本。"""
    bounds = rfm_bounds or {}
    r_med = bounds.get("recency", {}).get("median", 50)
    f_med = bounds.get("frequency", {}).get("median", 2)
    m_med = bounds.get("monetary", {}).get("median", 500)

    parts = []

    if recency <= 7:
        parts.append(f"最近消费仅 **{recency:.0f} 天** 前，属于**近期活跃**（训练数据中位数约 {r_med:.0f} 天）。")
    elif recency <= r_med:
        parts.append(f"最近消费 **{recency:.0f} 天** 前，活跃程度**中等偏上**。")
    else:
        parts.append(f"最近消费 **{recency:.0f} 天** 前，距今较久，存在**流失风险**。")

    if frequency <= 1:
        parts.append(f"购买次数仅 **{frequency:.0f} 次**，属于**极低频**（中位数约 {f_med:.0f} 次），更像新客或只买过一次。")
    elif frequency < f_med:
        parts.append(f"购买 **{frequency:.0f} 次**，低于数据中位数，频次偏低。")
    else:
        parts.append(f"购买 **{frequency:.0f} 次**，频次高于一般用户。")

    if monetary < m_med * 0.5:
        parts.append(f"总消费 **{monetary:.0f}**，金额**偏低**。")
    elif monetary < m_med * 1.5:
        parts.append(f"总消费 **{monetary:.0f}**，金额接近中等水平。")
    else:
        parts.append(f"总消费 **{monetary:.0f}**，金额**较高**。")

    prof = _profile_lookup(segment_profiles, cluster)
    c_r = prof.get("mean_recency", 0)
    c_f = prof.get("mean_frequency", 0)
    c_m = prof.get("mean_monetary", 0)

    display = resolve_segment_display(cluster, segment_profiles)
    parts.append(
        f"模型将该用户划入 **「{display['segment_name']}」**（第 {display['segment_id']} 类），"
        f"因购买次数与金额更接近该类典型值（该类平均：近 {c_r:.0f} 天、{c_f:.1f} 次、{c_m:.0f} 元）。"
    )

    if recency <= 14 and display["segment_id"] == 2:
        parts.append(
            "**说明：** 您输入的「最近消费」很新，与「流失风险」字面含义不完全一致；"
            "该标签描述的是这一**群体整体的平均画像**（该类多数人很久没买），"
            "而您的情况更接近 **「刚买过、但只买 1 次、金额不高」的低频新客**。"
        )

    return " ".join(parts)


ADVICE_MAP = {
    "核心高价值客户": "维护 VIP 权益、专属客服、新品优先体验，防止流失。",
    "流失风险客户": "发放召回券、邮件/短信唤醒、调研流失原因。",
    "活跃潜力客户": "捆绑销售、满减升单、推荐高毛利商品。",
    "高消费客户": "大额客户礼品、分期免息、定制化服务。",
    "一般发展客户": "常规促销、积分计划，培养复购习惯。",
}


def _profile_lookup(segment_profiles: dict, cluster_id: int) -> dict:
    """按簇编号读取 profile（兼容 JSON 字符串键）。"""
    return segment_profiles.get(cluster_id) or segment_profiles.get(str(cluster_id), {})


def _prepare_features_for_model(
    rfm_df: pd.DataFrame,
    feature_cols: List[str],
    scaler,
    clip_bounds: dict = None,
) -> np.ndarray:
    """与训练一致的特征变换后送入 scaler。"""
    if clip_bounds:
        engineered, _ = engineer_rfm_features(
            rfm_df, clip_bounds=clip_bounds, fit_bounds=False
        )
    else:
        engineered = rfm_df
    return scaler.transform(engineered[feature_cols].values.astype(float))


def load_model_bundle(output_dir: Path) -> dict:
    """加载 gmm_model.pkl。"""
    path = output_dir / "gmm_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到模型文件：{path}\n请先执行 python scripts/run_train.py 重新训练并导出模型。"
        )
    return joblib.load(path)


def predict_rfm(
    recency: float,
    frequency: float,
    monetary: float,
    gmm,
    scaler,
    feature_cols: List[str],
    segment_profiles: dict,
    rfm_bounds: dict = None,
    clip_bounds: dict = None,
) -> Dict:
    """对单个用户 RFM 做分群预测。"""
    if recency < 0 or frequency < 0 or monetary < 0:
        raise ValueError("RFM 数值不能为负数。")

    row = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=feature_cols,
    )
    X_scaled = _prepare_features_for_model(row, feature_cols, scaler, clip_bounds)
    cluster = int(gmm.predict(X_scaled)[0])
    proba = gmm.predict_proba(X_scaled)[0]

    display = resolve_segment_display(cluster, segment_profiles)

    prob_dict = {int(i): float(p) for i, p in enumerate(proba)}
    confidence = float(proba.max())

    return {
        "cluster": cluster,
        "segment_id": display["segment_id"],
        "segment_name": display["segment_name"],
        "description": display["description"],
        "advice": display["advice"],
        "probabilities": prob_dict,
        "confidence": confidence,
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "personalized_insight": build_personalized_insight(
            recency, frequency, monetary, segment_profiles, cluster, rfm_bounds
        ),
    }


def predict_batch(
    df: pd.DataFrame,
    gmm,
    scaler,
    feature_cols: List[str],
    segment_profiles: dict,
    id_col: str = None,
    clip_bounds: dict = None,
) -> pd.DataFrame:
    """批量预测，df 需含 Recency、Frequency、Monetary 列。"""
    col_map = {c.lower(): c for c in df.columns}
    required = ["recency", "frequency", "monetary"]
    for r in required:
        if r not in col_map:
            raise ValueError(f"批量文件缺少列：{r.capitalize()}")

    work = pd.DataFrame(
        {
            "Recency": df[col_map["recency"]].astype(float),
            "Frequency": df[col_map["frequency"]].astype(float),
            "Monetary": df[col_map["monetary"]].astype(float),
        }
    )

    if (work < 0).any().any():
        raise ValueError("RFM 数值不能包含负数。")

    X_scaled = _prepare_features_for_model(work, feature_cols, scaler, clip_bounds)
    clusters = gmm.predict(X_scaled)
    probas = gmm.predict_proba(X_scaled)

    out = work.copy()
    if id_col and id_col in df.columns:
        out.insert(0, id_col, df[id_col].values)
    elif "CustomerID" in df.columns:
        out.insert(0, "CustomerID", df["CustomerID"].values)
    elif "customerid" in col_map:
        out.insert(0, "CustomerID", df[col_map["customerid"]].values)

    out["Cluster"] = clusters
    displays = [
        resolve_segment_display(int(c), segment_profiles) for c in clusters
    ]
    out["SegmentId"] = [d["segment_id"] for d in displays]
    out["SegmentName"] = [d["segment_name"] for d in displays]
    out["Confidence"] = probas.max(axis=1).round(4)
    out["SegmentDescription"] = [d["description"] for d in displays]
    out["Advice"] = [d["advice"] for d in displays]

    return out
