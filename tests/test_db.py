"""存储层测试：发布事务、提交失败回滚、失效后不复活。"""

from __future__ import annotations

import json

import pytest

from app.db import Database, EmptyInputError, EpochChangedError

SAMPLES = [
    {"person_id": "A", "region": "E", "age_group": "y", "d": 2},
    {"person_id": "B", "region": "E", "age_group": "m", "d": 1},
]
TARGETS = [
    {"dimension": "region", "code": "E", "population": 3},
    {"dimension": "age_group", "code": "y", "population": 2},
]


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "db.sqlite")
    d.replace_samples(SAMPLES)
    d.replace_targets(TARGETS)
    yield d
    d.close()


def _fail_commits(db, monkeypatch, times=1):
    real = type(db)._commit_locked
    state = {"remaining": times}

    def wrapper(self, conn):
        if state["remaining"]:
            state["remaining"] -= 1
            raise RuntimeError("simulated COMMIT failure")
        return real(self, conn)

    monkeypatch.setattr(type(db), "_commit_locked", wrapper)


def test_failed_commit_does_not_leak_into_later_write(db, monkeypatch):
    job = db.create_job(None)
    _fail_commits(db, monkeypatch, times=1)
    with pytest.raises(RuntimeError):
        db.finish_job(job["job_id"], "succeeded",
                      {"weights": [{"person_id": "A", "w": 1.0}]}, None)
    # 回滚后不可见；内存中的行仍无结果
    row = db.get_job(job["job_id"])
    assert row["status"] == "queued"
    assert row["result"] is None
    assert db.current_result() is None

    # 事务已回滚：后续写入/提交不会把上一笔结果顺带写入
    job2 = db.create_job(None)
    assert db.finish_job(job2["job_id"], "succeeded",
                         {"weights": [{"person_id": "B", "w": 2.0}]}, None)
    rows = db._conn.execute(
        "SELECT job_id, status, result FROM jobs ORDER BY rowid"
    ).fetchall()
    assert [r["job_id"] for r in rows] == [job["job_id"], job2["job_id"]]
    assert rows[0]["status"] == "queued" and rows[0]["result"] is None
    assert rows[1]["status"] == "succeeded"
    result = db.current_result()
    assert result is not None and result["job_id"] == job2["job_id"]


def test_save_failure_marks_only_valid_jobs_failed(db, monkeypatch):
    job = db.create_job(None)
    _fail_commits(db, monkeypatch, times=1)
    with pytest.raises(RuntimeError):
        db.finish_job(job["job_id"], "succeeded", {"weights": []}, None)
    # 数据库可写且作业仍有效 -> 显示保存失败
    assert db.mark_save_failed(job["job_id"], "保存结果失败", job_epoch=job["epoch"])
    row = db.get_job(job["job_id"])
    assert row["status"] == "failed" and row["result"] is None
    assert row["error"].startswith("保存")


def test_save_failure_after_invalidation_does_not_revive(db, monkeypatch):
    job = db.create_job(None)
    db.replace_samples(SAMPLES)  # epoch 变化，作业被删除
    assert db.get_job(job["job_id"]) is None
    assert not db.mark_save_failed(job["job_id"], "保存失败", job_epoch=job["epoch"])
    assert db.get_job(job["job_id"]) is None
    assert db._conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE job_id=?", (job["job_id"],)
    ).fetchone()[0] == 0


def test_save_failure_does_not_override_committed_result(db):
    job = db.create_job(None)
    assert db.finish_job(job["job_id"], "succeeded", {"weights": []}, None)
    # 若提交实际成功，只是外层误判失败，不得把成功结果改成 failed
    assert not db.mark_save_failed(job["job_id"], "错误")
    assert db.get_job(job["job_id"])["status"] == "succeeded"


def test_finish_job_refuses_stale_epoch(db):
    job = db.create_job(None)
    # 用错误的 job_epoch 发布（模拟快照过期）
    assert not db.finish_job(job["job_id"], "succeeded", {"weights": []}, None,
                             job_epoch=job["epoch"] - 1)
    assert db.get_job(job["job_id"])["result"] is None
    # 作业被更正删除后，旧 epoch 的发布被拒绝
    db.replace_targets(TARGETS)
    assert not db.finish_job(job["job_id"], "succeeded", {"weights": []}, None,
                             job_epoch=job["epoch"])
    assert db.get_job(job["job_id"]) is None


def test_create_job_empty_input_is_rejected(tmp_path):
    d = Database(tmp_path / "empty.sqlite")
    try:
        with pytest.raises(EmptyInputError):
            d.create_job(None)
        assert d.list_jobs() == []
    finally:
        d.close()


def test_create_job_with_stale_snapshot_rejected(db):
    snap = db.snapshot()
    db.replace_targets(TARGETS)
    with pytest.raises(EpochChangedError):
        db.create_job(snap)
    assert db.list_jobs() == []


def test_result_only_after_commit_and_epoch_match(db):
    job = db.create_job(None)
    payload = {"weights": [{"person_id": "A", "d": 2, "w": 2.0,
                            "lower": 1.0, "upper": 4.0}]}
    assert db.finish_job(job["job_id"], "succeeded", payload, None)
    got = db.current_result()
    assert got is not None and json.loads(got["result"]) == payload
    # 输入更正后，旧结果（即使作业行理论上残留）不得作为当前结果
    db.replace_samples(SAMPLES)
    assert db.current_result() is None
