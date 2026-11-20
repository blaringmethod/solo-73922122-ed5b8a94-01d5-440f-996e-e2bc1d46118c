"""容器启动时的示例数据装载（仅在数据库为空时执行一次）。

这样从 /docs 演示时无需先手工粘贴样本与目标：直接 POST /calculate、
GET /result 即可。已有数据的数据库绝不会被覆盖。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_EXAMPLES = Path(os.environ.get("EXAMPLES_DIR", "/srv/examples"))


def _fallback_examples_dir() -> Path:
    # 本地无 Docker 运行时，examples/ 位于仓库根目录（app/ 的上一级）。
    return Path(__file__).resolve().parent.parent / "examples"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_examples(db, examples_dir: Path | str | None = None) -> bool:
    """数据库为空时写入 examples/feasible_*；返回是否写入。"""
    root = Path(examples_dir) if examples_dir else DEFAULT_EXAMPLES
    if not (root / "feasible_samples.json").exists():
        root = _fallback_examples_dir()
    # 只空库装载；启动阶段无并发写入。
    if db.list_samples() or db.list_targets():
        return False
    samples = _read(root / "feasible_samples.json")["samples"]
    targets = _read(root / "feasible_targets.json")["targets"]
    # 只在空库执行；两次写入各自 bump epoch，不会与在途计算混淆。
    db.replace_samples(samples)
    db.replace_targets(targets)
    return True
