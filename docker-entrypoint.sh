#!/bin/sh
# 启动包装：先装载示例数据（空库时），再启动 API（默认）；
# 传入 "demo" 作为第一个参数则运行端到端演示。
set -eu

DB="${WEIGHTS_DB:-/data/weights.db}"
DB_DIR="$(dirname "$DB")"
mkdir -p "$DB_DIR"

PYTHON="$(command -v python || command -v python3)"

"$PYTHON" - <<'PY'
import os

from app.db import Database
from app.demo_seed import load_examples

path = os.environ.get("WEIGHTS_DB", "/data/weights.db")
db = Database(path)
try:
    seed = os.environ.get("WEIGHTS_SEED", "1").strip().lower()
    if seed not in ("0", "false", "no", "off"):
        load_examples(db)
finally:
    db.close()
PY

if [ "${1:-api}" = "demo" ]; then
    shift || true
    exec "$PYTHON" -u /srv/examples/demo.py "$@"
fi
if [ "${1:-api}" = "api" ]; then
    shift || true
fi

exec "$PYTHON" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 "$@"
