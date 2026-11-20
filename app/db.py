"""SQLite 存储层。

版本化与发布规则（核心）：
- 每次成功提交样本或目标修改（PUT）都会令全局 epoch +1；
- epoch 变更会删除全部旧作业记录，并置位取消事件，使在途计算永久失效
  （即使把数据改回原值也不会恢复旧结果，必须重新计算）；
- 结果只能在“作业仍存在、且作业 epoch == 数据库当前 epoch”的事务内提交；
  提交失败立即回滚，因此未提交的权重绝不会被同一连接上的查询看到，
  也不会被之后的写入（例如再提交一笔计算）顺带写入；
- 所有读写共用同一把可重入锁。发布事务从 UPDATE 到 COMMIT 全程持锁，
  发布期间发起的查询会等待，只能看到提交前或提交后的已提交状态；
- 保存失败后，只有作业仍然有效时才尝试单独写入 `failed` 状态；
  若作业已因样本/目标更正而失效，则不做任何写入，绝不复活旧作业。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    epoch INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    person_id TEXT PRIMARY KEY,
    region TEXT NOT NULL,
    age_group TEXT NOT NULL,
    d INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS targets (
    dimension TEXT NOT NULL,
    code TEXT NOT NULL,
    population INTEGER NOT NULL,
    PRIMARY KEY (dimension, code)
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    epoch INTEGER NOT NULL,
    status TEXT NOT NULL,            -- queued|running|succeeded|infeasible|failed|interrupted
    submitted_at TEXT NOT NULL,
    finished_at TEXT,
    input_snapshot TEXT NOT NULL,    -- 提交时刻完整输入快照（JSON）
    result TEXT,                     -- 仅终态写入（完整权重+依据，或冲突集，或错误）
    error TEXT
);
"""

LIVE_STATUSES = ("queued", "running")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpochChangedError(RuntimeError):
    """在构造作业快照时输入版本已变化，调用方应重新取快照。"""


class EmptyInputError(RuntimeError):
    """当前输入为空（无样本或无目标），不应创建作业。"""

    def __init__(self, what: str):
        self.what = what
        super().__init__(what)


class DatabaseUnavailableError(RuntimeError):
    """连接在回滚失败后无法恢复；后续读写一律拒绝，避免泄漏未提交数据。"""


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._write_lock = threading.RLock()
        self._cancel = threading.Event()
        self._conn: sqlite3.Connection | None = self._open_conn()
        self._init_schema()

    # ------------------------------------------------------------------ 连接
    def _open_conn(self) -> sqlite3.Connection:
        # isolation_level=None：完全显式事务。任何未被 COMMIT 的写入都不会
        # 因为后续语句被隐式提交，回滚边界清晰。
        conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_schema(self) -> None:
        with self._write_lock:
            conn = self._require_conn()
            conn.executescript(SCHEMA)
            row = conn.execute("SELECT epoch FROM meta WHERE id=1").fetchone()
            if row is None:
                self._write_locked(lambda c: c.execute(
                    "INSERT INTO meta(id, epoch) VALUES (1, 1)"))
                self.epoch = 1
            else:
                self.epoch = int(row["epoch"])
            # 重启：未完成作业一律中断，不允许读到半成品
            self._write_locked(lambda c: c.execute(
                "UPDATE jobs SET status='interrupted', finished_at=?, "
                "error='process restarted' WHERE status IN ('queued','running')",
                (_now(),),
            ))

    def close(self) -> None:
        with self._write_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise DatabaseUnavailableError(
                "数据库连接不可用（回滚失败后已拒绝后续读写）"
            )
        return self._conn

    # ------------------------------------------------------------------ 事务
    def _commit_locked(self, conn: sqlite3.Connection) -> None:
        # 单独成方法，便于测试注入“提交失败”。提交失败时事务上下文会回滚。
        conn.commit()

    def _rollback_locked(self) -> None:
        """回滚当前事务；若回滚本身失败，丢弃连接，防止脏事务被后续提交。"""
        conn = self._conn
        if conn is None:
            return
        try:
            conn.rollback()
            # 防御性：回滚后仍有活动事务说明连接状态不可信
            if getattr(conn, "in_transaction", False):
                raise sqlite3.OperationalError("rollback left transaction open")
            return
        except Exception:
            pass
        # 回滚失败：关闭连接（close 会隐式回滚），再尝试重连。
        try:
            conn.close()
        except Exception:
            pass
        try:
            self._conn = self._open_conn()
        except Exception:
            self._conn = None

    @contextmanager
    def _transaction(self, begin: str = "BEGIN IMMEDIATE"):
        """持锁的显式事务；任何异常（含 COMMIT 失败）都回滚。"""
        with self._write_lock:
            conn = self._require_conn()
            conn.execute(begin)
            try:
                yield conn
            except BaseException:
                self._rollback_locked()
                raise
            else:
                try:
                    self._commit_locked(conn)
                except BaseException:
                    self._rollback_locked()
                    raise

    def _write_transaction(self):
        return self._transaction("BEGIN IMMEDIATE")

    def _write_locked(self, fn):
        """在事务中执行 fn(conn)；失败回滚并抛出。"""
        with self._write_transaction() as conn:
            return fn(conn)

    # ------------------------------------------------------------------ 读取
    def get_epoch(self) -> int:
        with self._write_lock:
            return self.epoch

    def list_samples(self) -> list[dict]:
        with self._write_lock:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT person_id, region, age_group, d FROM samples ORDER BY rowid"
            ).fetchall()
            return [dict(r) for r in rows]

    def list_targets(self) -> list[dict]:
        with self._write_lock:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT dimension, code, population FROM targets ORDER BY rowid"
            ).fetchall()
            return [dict(r) for r in rows]

    def snapshot(self) -> dict:
        # 在同一个事务内读取 epoch 与两份输入，避免出现“新 epoch + 旧数据”
        # 或“旧 epoch + 新数据”的撕裂快照。
        with self._transaction("BEGIN") as conn:
            epoch = int(conn.execute(
                "SELECT epoch FROM meta WHERE id=1"
            ).fetchone()["epoch"])
            samples = [dict(r) for r in conn.execute(
                "SELECT person_id, region, age_group, d FROM samples "
                "ORDER BY rowid"
            )]
            targets = [dict(r) for r in conn.execute(
                "SELECT dimension, code, population FROM targets ORDER BY rowid"
            )]
        return {"epoch": epoch, "samples": samples, "targets": targets}

    def get_job(self, job_id: str) -> dict | None:
        # 读也持锁：发布事务提交完成前不可见，提交失败回滚后也不可见。
        with self._write_lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return dict(row) if row else None

    def current_result(self) -> dict | None:
        """在同一把锁内取“当前版本最近一次成功提交”的结果并核对输入。

        返回 None 表示没有可用结果。核对失败（理论不可达，防御性）也返回
        None。因为读取与输入修改、结果发布共用同一把锁，本方法返回的行
        必然是某个完整 COMMIT 之后的状态；发布事务进行中调用会等待。
        """
        with self._write_lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT * FROM jobs WHERE status='succeeded' AND epoch=? "
                "AND result IS NOT NULL "
                "ORDER BY finished_at DESC, rowid DESC LIMIT 1",
                (self.epoch,),
            ).fetchone()
            if row is None:
                return None
            samples = [dict(r) for r in conn.execute(
                "SELECT person_id, region, age_group, d FROM samples ORDER BY rowid"
            )]
            targets = [dict(r) for r in conn.execute(
                "SELECT dimension, code, population FROM targets ORDER BY rowid"
            )]
            job = dict(row)
            snap = json.loads(job["input_snapshot"])
            if (snap.get("epoch") != self.epoch
                    or snap.get("samples") != samples
                    or snap.get("targets") != targets):
                return None
            return job

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._write_lock:
            conn = self._require_conn()
            rows = conn.execute(
                "SELECT job_id, epoch, status, submitted_at, finished_at, error "
                "FROM jobs ORDER BY submitted_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ 输入
    def _invalidate_locked(self, conn: sqlite3.Connection, new_epoch: int) -> None:
        conn.execute("UPDATE meta SET epoch=?", (new_epoch,))
        # 旧作业（含在途）永久失效：同一事务内删除，提交后不可再查询
        conn.execute("DELETE FROM jobs")

    def _replace_and_bump(self, delete_sql: str, insert_sql: str,
                          rows: list[dict]) -> int:
        with self._write_lock:
            new_epoch = self.epoch + 1
            with self._transaction() as conn:
                conn.execute(delete_sql)
                if rows:
                    conn.executemany(insert_sql, rows)
                self._invalidate_locked(conn, new_epoch)
            # 仅在 COMMIT 成功后更新内存 epoch 并取消在途计算。
            # 持锁执行，读取方不会看到 DB 新 epoch / 内存旧 epoch 的错位。
            self.epoch = new_epoch
            self._cancel.set()
            self._cancel = threading.Event()
            return new_epoch

    def replace_samples(self, samples: list[dict]) -> int:
        return self._replace_and_bump(
            "DELETE FROM samples",
            "INSERT INTO samples(person_id, region, age_group, d) "
            "VALUES (:person_id, :region, :age_group, :d)",
            samples,
        )

    def replace_targets(self, targets: list[dict]) -> int:
        return self._replace_and_bump(
            "DELETE FROM targets",
            "INSERT INTO targets(dimension, code, population) "
            "VALUES (:dimension, :code, :population)",
            targets,
        )

    # ------------------------------------------------------------------ 作业
    def current_cancel_event(self) -> threading.Event:
        with self._write_lock:
            return self._cancel

    def create_job(self, snapshot: dict | None = None) -> dict:
        """原子地创建作业。

        snapshot 为 None 时在事务内读取当前样本/目标，保证快照与 epoch 一致；
        传入历史 snapshot 时校验其 epoch，若已变化抛出 EpochChangedError。
        """
        with self._write_lock:
            job_id = uuid.uuid4().hex
            now = _now()
            with self._transaction() as conn:
                row = conn.execute("SELECT epoch FROM meta WHERE id=1").fetchone()
                epoch = int(row["epoch"])
                if snapshot is None:
                    samples = [dict(r) for r in conn.execute(
                        "SELECT person_id, region, age_group, d FROM samples "
                        "ORDER BY rowid"
                    )]
                    targets = [dict(r) for r in conn.execute(
                        "SELECT dimension, code, population FROM targets "
                        "ORDER BY rowid"
                    )]
                    # 空输入直接拒绝，且在事务内判定，避免与更正请求竞争
                    if not samples:
                        raise EmptyInputError("samples")
                    if not targets:
                        raise EmptyInputError("targets")
                    effective = {"epoch": epoch, "samples": samples,
                                 "targets": targets}
                else:
                    if snapshot.get("epoch") != epoch:
                        raise EpochChangedError(
                            "输入版本在提交计算前已变化，请重试"
                        )
                    effective = snapshot
                conn.execute(
                    "INSERT INTO jobs(job_id, epoch, status, submitted_at, "
                    "input_snapshot) VALUES (?,?,?,?,?)",
                    (job_id, epoch, "queued", now,
                     json.dumps(effective, ensure_ascii=False)),
                )
            self.epoch = epoch
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return dict(row)

    def mark_running(self, job_id: str) -> None:
        with self._write_transaction() as conn:
            conn.execute(
                "UPDATE jobs SET status='running' WHERE job_id=? AND status='queued'",
                (job_id,),
            )

    def discard_job(self, job_id: str) -> None:
        """仅用于从未入队的作业（例如空输入被拒绝）；不触碰已有终态。"""
        with self._write_transaction() as conn:
            conn.execute(
                "DELETE FROM jobs WHERE job_id=? AND status='queued'", (job_id,)
            )

    def finish_job(self, job_id: str, status: str, result: dict | None,
                   error: str | None, *, job_epoch: int | None = None) -> bool:
        """原子发布终态。

        在同一个事务里重新核对：作业仍存在、作业 epoch 等于数据库当前
        epoch（未被样本/目标更正失效）、作业尚未进入终态。任一条不满足
        就回滚并返回 False，绝不写入旧结果。COMMIT 失败会回滚并抛出，
        未提交的权重对任何查询都不可见。
        """
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT epoch, status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            meta = conn.execute(
                "SELECT epoch FROM meta WHERE id=1"
            ).fetchone()
            if row is None or meta is None:
                return False
            row_epoch = int(row["epoch"])
            current_epoch = int(meta["epoch"])
            if row_epoch != current_epoch:
                return False
            if job_epoch is not None and row_epoch != int(job_epoch):
                return False
            if row["status"] not in LIVE_STATUSES:
                return False
            conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, result=?, error=? "
                "WHERE job_id=? AND status IN ('queued','running')",
                (status, _now(),
                 json.dumps(result, ensure_ascii=False)
                 if result is not None else None,
                 error, job_id),
            )
        return True

    def mark_save_failed(self, job_id: str, error: str, *,
                         job_epoch: int | None = None) -> bool:
        """发布失败后的状态兜底：仅当作业仍有效时标记 failed。

        - 若作业已被样本/目标更正删除，或作业 epoch 已过期，返回 False，
          不写任何记录（绝不复活失效作业）；
        - 若作业已成功提交（例如提交结果不确定），不覆盖已有成功结果；
        - 写入的是单独的 failed 状态且 result=NULL，不携带任何权重。
        """
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT epoch, status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            meta = conn.execute(
                "SELECT epoch FROM meta WHERE id=1"
            ).fetchone()
            if row is None or meta is None:
                return False
            row_epoch = int(row["epoch"])
            if row_epoch != int(meta["epoch"]):
                return False
            if job_epoch is not None and row_epoch != int(job_epoch):
                return False
            if row["status"] not in LIVE_STATUSES:
                return False
            conn.execute(
                "UPDATE jobs SET status='failed', finished_at=?, "
                "result=NULL, error=? "
                "WHERE job_id=? AND status IN ('queued','running')",
                (_now(), error, job_id),
            )
        return True
