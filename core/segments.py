"""统一群体编号与名称映射。"""

from typing import Dict, List, Optional

CANONICAL_SEGMENTS: List[Dict] = [
    {
        "id": 1,
        "name": "核心高价值客户",
        "description": "近期活跃、购买频次与消费金额均高于平均水平",
        "advice": "维护 VIP 权益、专属客服、新品优先体验，防止流失。",
        "rfm_hint": "最近消费天数较少，购买次数与金额较高",
    },
    {
        "id": 2,
        "name": "流失风险客户",
        "description": "许久未消费或购买频次偏低，需唤醒与召回运营",
        "advice": "发放召回券、邮件/短信唤醒、调研流失原因。",
        "rfm_hint": "最近消费距今较久，购买次数偏少",
    },
    {
        "id": 3,
        "name": "活跃潜力客户",
        "description": "有一定活跃度，频次或客单价仍有提升空间",
        "advice": "捆绑销售、满减升单、推荐高毛利商品，培养复购。",
        "rfm_hint": "购买较勤但客单价中等或偏低，适合升单",
    },
]

SEGMENT_NAME_TO_CANONICAL_ID: Dict[str, int] = {
    "核心高价值客户": 1,
    "流失风险客户": 2,
    "活跃潜力客户": 3,
    "高消费客户": 1,
    "一般发展客户": 3,
}

CANONICAL_BY_ID = {s["id"]: s for s in CANONICAL_SEGMENTS}


def get_segment_catalog() -> List[Dict]:
    """返回可写入 manifest / model_meta 的群体手册。"""
    return [dict(s) for s in CANONICAL_SEGMENTS]


def canonical_id_for_name(name: str, fallback_cluster: int = 0) -> int:
    """按训练阶段业务名称解析统一编号。"""
    if name in SEGMENT_NAME_TO_CANONICAL_ID:
        return SEGMENT_NAME_TO_CANONICAL_ID[name]
    return min(max(fallback_cluster + 1, 1), len(CANONICAL_SEGMENTS))


def attach_canonical_to_profiles(segment_profiles: dict) -> dict:
    """为各簇 profile 写入 canonical_id。"""
    for key, prof in segment_profiles.items():
        try:
            cluster = int(key)
        except (TypeError, ValueError):
            cluster = 0
        prof["canonical_id"] = canonical_id_for_name(
            prof.get("name", ""), fallback_cluster=cluster
        )
    return segment_profiles


def ensure_meta_segment_catalog(meta: dict) -> dict:
    """补全旧模型的 segment_catalog 与 canonical_id。"""
    if "segment_catalog" not in meta:
        meta["segment_catalog"] = get_segment_catalog()
    if meta.get("segment_profiles"):
        meta["segment_profiles"] = attach_canonical_to_profiles(
            dict(meta["segment_profiles"])
        )
    return meta


def resolve_segment_display(cluster: int, segment_profiles: dict) -> Dict:
    """由 GMM 簇编号得到界面展示用的群体字段。"""
    prof = segment_profiles.get(cluster) or segment_profiles.get(str(cluster), {})
    canonical_id = prof.get("canonical_id")
    if canonical_id is None:
        canonical_id = canonical_id_for_name(prof.get("name", ""), cluster)

    canonical = CANONICAL_BY_ID.get(
        int(canonical_id), CANONICAL_SEGMENTS[0]
    )
    return {
        "cluster": int(cluster),
        "segment_id": int(canonical_id),
        "segment_name": canonical["name"],
        "description": prof.get("description") or canonical["description"],
        "advice": canonical["advice"],
        "rfm_hint": canonical.get("rfm_hint", ""),
        "profile_size": prof.get("size"),
    }


def profiles_for_period_table(segment_profiles: dict) -> List[Dict]:
    """生成当前周期群体映射表行（编号、名称、簇下标、人数）。"""
    rows = []
    for key, prof in segment_profiles.items():
        try:
            cluster = int(key)
        except (TypeError, ValueError):
            continue
        display = resolve_segment_display(cluster, segment_profiles)
        rows.append(
            {
                "群体编号": display["segment_id"],
                "所属群体": display["segment_name"],
                "模型簇下标": cluster,
                "训练用户数": prof.get("size", "—"),
            }
        )
    rows.sort(key=lambda r: r["群体编号"])
    return rows
