"""
RFM 统计周期配置（训练与前端共用）。

各周期以分析截止日前推 N 天截取交易明细，再构建 RFM 并单独训练 GMM。
"""

from typing import List, Dict

# key 用于 output 子目录名；days 为窗口天数（含首尾习惯按自然日近似）
TRAINING_PERIODS: List[Dict] = [
    {"key": "1y", "label": "近 1 年", "days": 365},
    {"key": "6m", "label": "近半年", "days": 182},
    {"key": "3m", "label": "近 1 季度", "days": 91},
    {"key": "1m", "label": "近 1 个月", "days": 30},
]

DEFAULT_PERIOD_KEY = "1y"

PERIOD_BY_KEY = {p["key"]: p for p in TRAINING_PERIODS}
