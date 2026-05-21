"""
本地训练脚本（开发者/分析人员使用）

按四个统计周期（近1年 / 近半年 / 近1季度 / 近1个月）分别训练 GMM，
产物保存至 output/{period_key}/，并生成 output/manifest.json。

用法：
    python scripts/run_train.py
    python scripts/run_train.py --data data/online_retail.csv --k 3
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import clean_data, load_data, preprocess_for_window
from core.model import run_clustering
from core.periods import DEFAULT_PERIOD_KEY, TRAINING_PERIODS
from core.segments import attach_canonical_to_profiles, get_segment_catalog

OUTPUT_DIR = PROJECT_ROOT / "output"


def assign_segment_labels(
    rfm_df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list,
) -> dict:
    """根据各簇 RFM 均值相对全局均值的偏离，生成面向业务的群体名称。"""
    df = rfm_df.copy()
    df["Cluster"] = labels
    global_mean = df[feature_cols].mean()
    profiles = {}

    for cid in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cid]
        mean = sub[feature_cols].mean()
        r_score = "低" if mean["Recency"] < global_mean["Recency"] else "高"
        f_score = "高" if mean["Frequency"] >= global_mean["Frequency"] else "低"
        m_score = "高" if mean["Monetary"] >= global_mean["Monetary"] else "低"

        if f_score == "高" and m_score == "高" and r_score == "低":
            name, desc = "核心高价值客户", "近期活跃、消费频次与金额均高于平均水平"
        elif r_score == "高" and f_score == "低":
            name, desc = "流失风险客户", "许久未消费，购买频次偏低，需唤醒运营"
        elif f_score == "高" and m_score == "低":
            name, desc = "活跃潜力客户", "购买较频繁但客单价偏低，适合交叉销售与升单"
        elif m_score == "高":
            name, desc = "高消费客户", "单笔贡献较高，适合会员权益与专属服务"
        else:
            name, desc = "一般发展客户", "消费活跃度中等，可通过促销提升复购"

        profiles[int(cid)] = {
            "name": name,
            "description": desc,
            "size": int(len(sub)),
            "mean_recency": round(float(mean["Recency"]), 1),
            "mean_frequency": round(float(mean["Frequency"]), 1),
            "mean_monetary": round(float(mean["Monetary"]), 2),
        }

    return profiles


def train_single_period(
    cleaned_df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    period_cfg: dict,
    n_components: int,
    random_state: int,
    output_dir: Path,
    data_source_name: str,
) -> dict:
    """训练单个统计周期并写入 output_dir。"""
    period_key = period_cfg["key"]
    period_label = period_cfg["label"]
    window_days = period_cfg["days"]

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"统计周期：{period_label}（窗口 {window_days} 天）")
    print(f"{'=' * 50}")

    print("[1/4] 截取窗口并构建 RFM ...")
    windowed_df, rfm_df, X_scaled, scaler, feature_cols, clip_bounds = preprocess_for_window(
        cleaned_df,
        window_days=window_days,
        snapshot_date=snapshot_date,
    )

    if len(rfm_df) < n_components + 1:
        raise ValueError(
            f"{period_label} 有效用户数 {len(rfm_df)} 过少，无法训练 K={n_components} 的 GMM。"
        )

    print(f"  窗口内交易 {len(windowed_df):,} 条，用户 {len(rfm_df):,} 人")

    print(f"[2/4] 训练 GMM（K={n_components}）...")
    gmm, labels, metrics = run_clustering(
        X_scaled,
        n_components=n_components,
        random_state=random_state,
    )

    print("[3/4] 生成业务标签与 PCA 坐标...")
    segment_profiles = assign_segment_labels(rfm_df, labels, feature_cols)
    attach_canonical_to_profiles(segment_profiles)

    result_df = rfm_df.copy()
    result_df["Cluster"] = labels
    result_df["SegmentId"] = result_df["Cluster"].map(
        lambda c: segment_profiles[int(c)]["canonical_id"]
    )
    result_df["SegmentName"] = result_df["Cluster"].map(
        lambda c: segment_profiles[int(c)]["name"]
    )
    result_df["SegmentDescription"] = result_df["Cluster"].map(
        lambda c: segment_profiles[int(c)]["description"]
    )

    summary_rows = []
    for cid, prof in segment_profiles.items():
        summary_rows.append(
            {
                "Cluster": cid,
                "SegmentId": prof["canonical_id"],
                "SegmentName": prof["name"],
                "UserCount": prof["size"],
                "AvgRecency": prof["mean_recency"],
                "AvgFrequency": prof["mean_frequency"],
                "AvgMonetary": prof["mean_monetary"],
                "Description": prof["description"],
            }
        )
    cluster_summary = pd.DataFrame(summary_rows)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": labels,
            "SegmentName": result_df["SegmentName"].values,
            "CustomerID": result_df["CustomerID"].values,
        }
    )

    print(f"[4/4] 保存产物到 {output_dir} ...")
    joblib.dump(
        {
            "gmm": gmm,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "clip_bounds": clip_bounds,
        },
        output_dir / "gmm_model.pkl",
    )
    result_df.to_csv(output_dir / "user_segments.csv", index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(
        output_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig"
    )
    pca_df.to_csv(output_dir / "pca_points.csv", index=False, encoding="utf-8-sig")

    meta = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": data_source_name,
        "period_key": period_key,
        "period_label": period_label,
        "window_days": window_days,
        "n_users": int(len(rfm_df)),
        "n_transactions": int(len(windowed_df)),
        "n_clusters": int(n_components),
        "feature_cols": feature_cols,
        "metrics": metrics,
        "pca_explained_variance_ratio": [
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        ],
        "segment_profiles": segment_profiles,
        "segment_catalog": get_segment_catalog(),
        "rfm_bounds": {
            "recency": {
                "min": float(rfm_df["Recency"].min()),
                "max": float(rfm_df["Recency"].max()),
                "median": float(rfm_df["Recency"].median()),
            },
            "frequency": {
                "min": float(rfm_df["Frequency"].min()),
                "max": float(rfm_df["Frequency"].max()),
                "median": float(rfm_df["Frequency"].median()),
            },
            "monetary": {
                "min": float(rfm_df["Monetary"].min()),
                "max": float(rfm_df["Monetary"].max()),
                "median": float(rfm_df["Monetary"].median()),
            },
        },
    }

    with open(output_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  用户数：{meta['n_users']:,}  BIC：{metrics['bic']:,.2f}")
    sil = metrics["silhouette"]
    print(f"  轮廓系数：{sil:.4f}" if sil == sil else "  轮廓系数：N/A")

    return meta


def train_all_periods(
    data_path: str,
    n_components: int = 3,
    random_state: int = 42,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """为全部统计周期训练并写入 manifest。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载数据：{data_path}")
    raw = load_data(data_path)
    cleaned_df = clean_data(raw)
    cleaned_df["InvoiceDate"] = pd.to_datetime(cleaned_df["InvoiceDate"])
    snapshot_date = cleaned_df["InvoiceDate"].max() + pd.Timedelta(days=1)
    data_source_name = Path(data_path).name

    trained_periods = []
    for period_cfg in TRAINING_PERIODS:
        period_dir = output_dir / period_cfg["key"]
        meta = train_single_period(
            cleaned_df=cleaned_df,
            snapshot_date=snapshot_date,
            period_cfg=period_cfg,
            n_components=n_components,
            random_state=random_state,
            output_dir=period_dir,
            data_source_name=data_source_name,
        )
        trained_periods.append(
            {
                "key": period_cfg["key"],
                "label": period_cfg["label"],
                "days": period_cfg["days"],
                "n_users": meta["n_users"],
            }
        )

    manifest = {
        "version": 1,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": data_source_name,
        "default_period": DEFAULT_PERIOD_KEY,
        "segment_catalog": get_segment_catalog(),
        "periods": trained_periods,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n全部周期训练完成！")
    for p in trained_periods:
        print(f"  · {p['label']}：{p['n_users']:,} 用户 → output/{p['key']}/")
    print("\n请运行客户界面：streamlit run app.py")


def parse_args():
    parser = argparse.ArgumentParser(description="本地训练多周期 GMM 用户分群模型")
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "online_retail.csv"),
        help="原始交易 CSV 路径",
    )
    parser.add_argument("--k", type=int, default=3, help="聚类数 K")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="产物根目录",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_file = Path(args.data)
    if not data_file.exists():
        print(f"错误：数据文件不存在 → {data_file}")
        print("请将 online_retail.csv 放入 data/ 目录，或通过 --data 指定路径。")
        sys.exit(1)

    train_all_periods(
        data_path=str(data_file),
        n_components=args.k,
        random_state=args.seed,
        output_dir=Path(args.output),
    )
