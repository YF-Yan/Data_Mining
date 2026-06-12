"""
训练产物加载模块（供客户端 app 使用）
支持多统计周期：output/{period_key}/ 及 output/manifest.json
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from core.periods import DEFAULT_PERIOD_KEY, PERIOD_BY_KEY, TRAINING_PERIODS
from core.segments import ensure_meta_segment_catalog, get_segment_catalog

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
MANIFEST_FILE = "manifest.json"
REQUIRED_ARTIFACTS = [
    "model_meta.json",
    "gmm_model.pkl",
]


def get_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"未找到训练产物：{path}\n"
            f"请先在项目根目录执行：python scripts/run_train.py\n{hint}"
        )


def period_dir_complete(path: Path) -> bool:
    """某周期目录下 gmm_model.pkl 与 model_meta.json 是否齐全。"""
    return all((path / name).exists() for name in REQUIRED_ARTIFACTS)


def _period_dir_complete(path: Path) -> bool:
    return period_dir_complete(path)


def load_manifest(output_dir: Optional[Path] = None) -> Optional[Dict]:
    """加载多周期清单；不存在则返回 None（兼容旧版单模型目录）。"""
    out = Path(output_dir) if output_dir else get_output_dir()
    manifest_path = out / MANIFEST_FILE
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_period_options(output_dir: Optional[Path] = None) -> List[Dict]:
    """
    返回可选周期列表 [{"key","label","days"}, ...]。
    优先 manifest；否则尝试旧版根目录单模型并映射为近 1 年。
    """
    out = Path(output_dir) if output_dir else get_output_dir()
    manifest = load_manifest(out)
    if manifest and manifest.get("periods"):
        return manifest["periods"]

    if _period_dir_complete(out):
        meta = load_model_meta(out)
        return [
            {
                "key": meta.get("period_key", DEFAULT_PERIOD_KEY),
                "label": meta.get("period_label", "近 1 年"),
                "days": meta.get("window_days", 365),
            }
        ]
    return []


def get_period_output_dir(
    period_key: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """解析某统计周期对应的产物目录。"""
    out = Path(output_dir) if output_dir else get_output_dir()
    manifest = load_manifest(out)
    if manifest:
        keys = {p["key"] for p in manifest.get("periods", [])}
        if period_key not in keys:
            raise ValueError(
                f"未知统计周期：{period_key}，可选：{', '.join(sorted(keys))}"
            )
        return out / period_key

    # 旧版：产物直接在 output/ 根目录
    if period_key != DEFAULT_PERIOD_KEY and not (out / period_key).exists():
        raise ValueError("当前为旧版单模型产物，请重新执行 python scripts/run_train.py")
    if (out / period_key).is_dir() and _period_dir_complete(out / period_key):
        return out / period_key
    return out


def resolve_period_key(period_key: Optional[str], output_dir: Optional[Path] = None) -> str:
    if period_key:
        return period_key
    manifest = load_manifest(output_dir)
    if manifest:
        return manifest.get("default_period", DEFAULT_PERIOD_KEY)
    return DEFAULT_PERIOD_KEY


def artifacts_ready(output_dir: Optional[Path] = None) -> bool:
    """检查训练产物是否齐全（manifest 所列周期或旧版单目录）。"""
    out = Path(output_dir) if output_dir else get_output_dir()
    manifest = load_manifest(out)
    if manifest:
        periods = manifest.get("periods", [])
        return bool(periods) and all(
            _period_dir_complete(out / p["key"]) for p in periods
        )
    return _period_dir_complete(out)


def all_periods_trained(output_dir: Optional[Path] = None) -> bool:
    """四统计周期训练产物是否齐全（manifest + 各周期模型文件）。"""
    out = Path(output_dir) if output_dir else get_output_dir()
    manifest = load_manifest(out)
    if not manifest or not manifest.get("periods"):
        return False
    keys = {p["key"] for p in manifest["periods"]}
    expected = {p["key"] for p in TRAINING_PERIODS}
    if keys != expected:
        return False
    return all(_period_dir_complete(out / k) for k in expected)


def load_model_meta(
    output_dir: Optional[Path] = None,
    period_key: Optional[str] = None,
) -> Dict:
    out = get_period_output_dir(resolve_period_key(period_key, output_dir), output_dir)
    meta_path = out / "model_meta.json"
    _require_file(meta_path, "训练脚本会生成 model_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return ensure_meta_segment_catalog(meta)


def load_segment_catalog(
    output_dir: Optional[Path] = None,
    period_key: Optional[str] = None,
) -> List[Dict]:
    """统一群体手册；优先 manifest，否则 model_meta 或内置默认。"""
    out = Path(output_dir) if output_dir else get_output_dir()
    manifest = load_manifest(out)
    if manifest and manifest.get("segment_catalog"):
        return manifest["segment_catalog"]
    meta = load_model_meta(output_dir=out, period_key=period_key)
    return meta.get("segment_catalog") or get_segment_catalog()


def load_model_bundle(
    output_dir: Optional[Path] = None,
    period_key: Optional[str] = None,
):
    from core.predictor import load_model_bundle as _load

    out = get_period_output_dir(resolve_period_key(period_key, output_dir), output_dir)
    return _load(out)
