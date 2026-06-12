"""四档统计周期配置（训练与前端共用）。"""

from typing import List, Dict

TRAINING_PERIODS: List[Dict] = [
    {"key": "1y", "label": "近 1 年", "days": 365},
    {"key": "6m", "label": "近半年", "days": 182},
    {"key": "3m", "label": "近 1 季度", "days": 91},
    {"key": "1m", "label": "近 1 个月", "days": 30},
]

DEFAULT_PERIOD_KEY = "1y"

PERIOD_BY_KEY = {p["key"]: p for p in TRAINING_PERIODS}
