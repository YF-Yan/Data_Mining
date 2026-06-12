"""
项目路径解析：所有相对路径均基于仓库根目录，便于跨机器移植。
"""

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

_PROJECT_ROOT: Path | None = None


def get_project_root() -> Path:
    """仓库根目录（含 app.py、core/、data/、output/）。"""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    return _PROJECT_ROOT


def resolve_project_path(path: PathLike) -> Path:
    """
    解析项目内路径。
    - 已是绝对路径：原样返回
    - 相对路径：相对于项目根目录
    """
    p = Path(path)
    return p if p.is_absolute() else get_project_root() / p
