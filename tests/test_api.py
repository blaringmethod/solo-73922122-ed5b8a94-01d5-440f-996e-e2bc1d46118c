"""API 端到端测试：校验、作业生命周期、失效语义、重启中断。"""

import importlib
import os
import time
from pathlib import Path

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHTS_DB", str(tmp_path / "test.db"))
    # 确保干净导入（DB 路径在模块导入时读取）
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c, main
    # 测试进程长期运行、可能受容器线程配额限制：等待工作线程真正退出。
    main.app.state.jobs.shutdown(wait=True)


FEAS_SAMPLES = [
    {"person_id": "P1", "region": "East",  "age_group": "youth",  "d": 1},
    {"person_id": "P2", "region": "East",  "age_group": "mid",    "d": 1},
    {"person_id": "P3", "region": "East",  "age_group": "senior", "d": 2},
    {"person_id": "P4", "region": "South", "age_group": "youth",  "d": 1},
    {"person_id": "P5", "region": "South", "age_group": "mid",    "d": 3},
    {"person_id": "P6", "region": "West",  "age_group": "youth",  "d": 1},
    {"person_id": "P7", "region": "West",  "age_group": "senior", "d": 2},
    {"person_id": "P8", "region": "North", "age_group": "mid",    "d": 1},
]
FEAS_TARGETS = [
    {"dimension": "region", "code": "East", "population": 4},
    {"dimension": "region", "code": "South", "population": 2},
    {"dimension": "region", "code": "West", "population": 3},
    {"dimension": "region", "code": "North", "population": 1},
    {"dimension": "age_group", "code": "youth", "population": 4},
    {"dimension": "age_group", "code": "mid", "population": 4},
    {"dimension": "age_group", "code": "senior", "population": 2},
]
CONFLICT_SAMPLES = [
    {"person_id": "X", "region": "R1", "age_group": "a1", "d": 1},
    {"person_id": "Y", "region": "R1", "age_group": "a2", "d": 1},
    {"person_id": "Z", "region": "R2", "age_group": "a1", "d": 1},
]
CONFLICT_TARGETS = [
    {"dimension": "region", "code": "R1", "population": 4},
    {"dimension": "age_group", "code": "a1", "population": 1},
]


def poll(c, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = c.get(f"/jobs/{job_id}")
        if r.status_code == 404:
            return r.json()
        body = r.json()
        if body["status"] not in ("queued", "running"):
            return body
        time.sleep(0.05)
    raise TimeoutError


def setup_feasible(c):
    assert c.put("/samples", json={"samples": FEAS_SAMPLES}).status_code == 200
    assert c.put("/targets", json={"targets": FEAS_TARGETS}).status_code == 200


def test_feasible_full_flow(client):
    c, _ = client
    setup_feasible(c)
    r = c.post("/calculate")
    assert r.status_code == 202
    job = poll(c, r.json()["job_id"])
    assert job["status"] == "succeeded"
    assert len(job["weights"]) == len(FEAS_SAMPLES)
    # 界限与容差
    for item in job["weights"]:
        assert item["lower"] <= item["w"] <= item["upper"] + 1e-12
    assert job["residual_max_abs"] <= 1e-6
    assert job["residual_abs_sum"] <= 1e-6
    for t in job["target_totals"]:
        assert t["abs_error"] <= 1e-6
    # 输入依据随结果一起返回
    assert job["input_snapshot"]["samples"] == FEAS_SAMPLES
    assert job["input_snapshot"]["targets"] == FEAS_TARGETS
    # /result 仅提供完整一致结果
    r = c.get("/result")
    assert r.status_code == 200
    assert r.json()["weights"] == job["weights"]
    # 非成功状态没有半份权重
    assert "weights" in job


def test_conflict_returns_iis(client):
    c, _ = client
    assert c.put("/samples", json={"samples": CONFLICT_SAMPLES}).status_code == 200
    assert c.put("/targets", json={"targets": CONFLICT_TARGETS}).status_code == 200
    r = c.post("/calculate")
    job = poll(c, r.json()["job_id"])
    assert job["status"] == "infeasible"
    assert job["weights"] is None
    iis = {(t["dimension"], t["code"]) for t in job["infeasible_targets"]}
    assert iis == {("region", "R1"), ("age_group", "a1")}
    # 冲突不开放结果
    assert c.get("/result").status_code == 409


def test_any_modification_invalidates_old_results(client):
    c, _ = client
    setup_feasible(c)
    r = c.post("/calculate")
    job_id = r.json()["job_id"]
    poll(c, job_id)
    assert c.get("/result").status_code == 200

    epoch_before = c.get("/health").json()["epoch"]
    # 即使提交完全相同的样本，任何已提交修改都使旧结果失效
    r = c.put("/samples", json={"samples": FEAS_SAMPLES})
    assert r.json()["epoch"] == epoch_before + 1
    assert c.get(f"/jobs/{job_id}").status_code == 404
    assert c.get("/result").status_code == 409

    # 重新计算后获得新完整结果
    r = c.post("/calculate")
    job = poll(c, r.json()["job_id"])
    assert job["status"] == "succeeded"
    assert c.get("/result").status_code == 200


def test_modification_during_flight_invalidates(client, monkeypatch):
    c, main = client
    setup_feasible(c)
    # 在求解器入口加延迟，确保修改发生在作业运行期间
    import app.solver as solver
    orig = solver.solve_qp

    def slow(samples, targets):
        time.sleep(0.6)
        return orig(samples, targets)

    monkeypatch.setattr(solver, "solve_qp", slow)
    r = c.post("/calculate")
    job_id = r.json()["job_id"]
    time.sleep(0.15)
    # 修改输入：在途计算应永久失效
    c.put("/targets", json={"targets": FEAS_TARGETS})
    deadline = time.time() + 10
    while time.time() < deadline:
        got = c.get(f"/jobs/{job_id}")
        if got.status_code == 404:
            break
        time.sleep(0.05)
    assert c.get(f"/jobs/{job_id}").status_code == 404
    assert c.get("/result").status_code == 409


def test_restart_marks_running_jobs_interrupted(tmp_path, monkeypatch):
    db_file = tmp_path / "restart.db"
    monkeypatch.setenv("WEIGHTS_DB", str(db_file))
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        setup_feasible(c)
        r = c.post("/calculate")
        job_id = r.json()["job_id"]
        poll(c, job_id)
        # 手工制造一个 queued 作业模拟重启残留
        c.post("/calculate")
    # 重新实例化 Database（模拟进程重启）
    main.app.state.jobs.shutdown(wait=True)
    db2 = main.Database(str(db_file))
    rows = db2._conn.execute(
        "SELECT status FROM jobs WHERE status IN ('queued','running')"
    ).fetchall()
    assert rows == []
    interrupted = db2._conn.execute(
        "SELECT status FROM jobs WHERE status='interrupted'"
    ).fetchall()
    assert len(interrupted) >= 1
    db2.close()


def test_validation_rules(client):
    c, _ = client
    # 布尔值不得冒充整数
    r = c.put("/samples", json={"samples": [
        {"person_id": "A", "region": "r", "age_group": "g", "d": True}]})
    assert r.status_code == 422
    # d 越界
    r = c.put("/samples", json={"samples": [
        {"person_id": "A", "region": "r", "age_group": "g", "d": 101}]})
    assert r.status_code == 422
    r = c.put("/samples", json={"samples": [
        {"person_id": "A", "region": "r", "age_group": "g", "d": 0}]})
    assert r.status_code == 422
    # 重复 person_id
    r = c.put("/samples", json={"samples": [
        {"person_id": "A", "region": "r", "age_group": "g", "d": 1},
        {"person_id": "A", "region": "r", "age_group": "g", "d": 1}]})
    assert r.status_code == 422
    # 超员
    r = c.put("/samples", json={"samples": [
        {"person_id": f"P{i}", "region": "r", "age_group": "g", "d": 1}
        for i in range(201)]})
    assert r.status_code == 422
    # 负人口
    r = c.put("/targets", json={"targets": [
        {"dimension": "region", "code": "r", "population": -1}]})
    assert r.status_code == 422
    # 非法维度
    r = c.put("/targets", json={"targets": [
        {"dimension": "gender", "code": "r", "population": 1}]})
    assert r.status_code == 422
    # 目标超项
    r = c.put("/targets", json={"targets": [
        {"dimension": "region", "code": f"r{i}", "population": 1}
        for i in range(13)]})
    assert r.status_code == 422
    # 额外字段禁止
    r = c.put("/samples", json={"samples": [
        {"person_id": "A", "region": "r", "age_group": "g", "d": 1,
         "extra": 1}]})
    assert r.status_code == 422


def test_calculate_requires_data(client):
    c, _ = client
    assert c.post("/calculate").status_code == 422
    c.put("/samples", json={"samples": FEAS_SAMPLES})
    assert c.post("/calculate").status_code == 422
    c.put("/targets", json={"targets": FEAS_TARGETS})
    assert c.post("/calculate").status_code == 202


def test_solver_failure_is_failed_not_infeasible(client, monkeypatch):
    c, main = client
    setup_feasible(c)

    def boom(samples, targets):
        return solver_mod.QPResult(status="failed", reason="simulated numeric failure")

    import app.solver as solver_mod
    monkeypatch.setattr(solver_mod, "solve_qp", boom)
    r = c.post("/calculate")
    job = poll(c, r.json()["job_id"])
    assert job["status"] == "failed"
    assert job["infeasible_targets"] is None
    assert "simulated numeric failure" in job["error"]
    assert job["weights"] is None
    assert c.get("/result").status_code == 409


def test_docs_and_openapi_available(client):
    c, _ = client
    assert c.get("/docs").status_code == 200
    spec = c.get("/openapi.json").json()
    assert "/calculate" in spec["paths"]


# ---------------------------------------------------------------- 结果发布与保存失败
# POST /calculate 之后，工作线程依次提交：create_job、mark_running、终态发布。
# 终态发布是第 3 次提交；下面的注入点都针对它。
PUBLISH_COMMIT_INDEX = 3


def _commit_plan(db_obj, monkeypatch, action):
    """用计数器包装 Database._commit_locked；action(n, conn) 决定行为。"""
    real = type(db_obj)._commit_locked
    state = {"n": 0}

    def wrapper(self, conn):
        state["n"] += 1
        return action(state["n"], real, self, conn)

    monkeypatch.setattr(type(db_obj), "_commit_locked", wrapper)
    return state


def test_failed_result_commit_is_invisible_and_not_saved_later(
        client, monkeypatch):
    c, main = client
    setup_feasible(c)
    db_obj = main.app.state.db
    state = {"failed": 0}

    def action(n, real, self, conn):
        if n == PUBLISH_COMMIT_INDEX:
            state["failed"] += 1
            raise RuntimeError("simulated disk I/O error on COMMIT")
        return real(self, conn)

    _commit_plan(db_obj, monkeypatch, action)

    r = c.post("/calculate")
    job_id = r.json()["job_id"]
    while True:
        got = c.get(f"/jobs/{job_id}")
        assert got.status_code == 200, "保存失败后作业仍应可查询"
        body = got.json()
        if body["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert state["failed"] == 1
    # 发布事务回滚后：作业不携带任何权重，且被标记为保存失败
    assert body["status"] == "failed"
    assert body["weights"] is None
    assert body["target_totals"] is None
    assert "save" in body["error"].lower() or "保存" in body["error"]
    # /result 不提供这笔未提交的结果
    assert c.get("/result").status_code == 409

    # 再提交一笔计算：提交与发布都正常，且不得把上一笔权重一起存下。
    r2 = c.post("/calculate")
    job2 = poll(c, r2.json()["job_id"])
    assert job2["status"] == "succeeded"
    assert job2["weights"] is not None
    result = c.get("/result")
    assert result.status_code == 200
    # 数据库中只能有这一笔成功结果，绝不包含失败作业的半份结果
    rows = db_obj._conn.execute(
        "SELECT job_id, status, result FROM jobs WHERE status='succeeded'"
    ).fetchall()
    assert {row["job_id"] for row in rows} == {job2["job_id"]}
    assert all(row["result"] is not None for row in rows)
    # 作业表里也没有任何其他行携带结果
    stray = db_obj._conn.execute(
        "SELECT job_id FROM jobs WHERE result IS NOT NULL"
    ).fetchall()
    assert {row["job_id"] for row in stray} == {job2["job_id"]}
    # 失败作业依旧是 failed 且无结果
    failed = c.get(f"/jobs/{job_id}").json()
    assert failed["status"] == "failed" and failed["weights"] is None
    # 恢复后确认无残留事务/可见性异常
    assert c.get("/result").json()["weights"] == job2["weights"]


def test_queries_during_publication_see_only_committed_weights(
        client, monkeypatch):
    import threading

    c, main = client
    setup_feasible(c)
    db_obj = main.app.state.db
    in_commit = threading.Event()
    release = threading.Event()
    observed = []

    def action(n, real, self, conn):
        if n == PUBLISH_COMMIT_INDEX and not in_commit.is_set():
            in_commit.set()
            # 发布事务的 UPDATE 已执行但尚未 COMMIT，期间发起查询
            release.wait(5)
        return real(self, conn)

    _commit_plan(db_obj, monkeypatch, action)

    r = c.post("/calculate")
    job_id = r.json()["job_id"]
    assert in_commit.wait(10), "发布未进入提交阶段"

    def query():
        observed.append(c.get("/result"))
        observed.append(c.get(f"/jobs/{job_id}"))

    t = threading.Thread(target=query)
    t.start()
    time.sleep(0.3)  # 查询应阻塞在发布事务持有的锁上
    assert t.is_alive(), "发布期间查询不应看到未提交状态"
    release.set()
    t.join(10)
    assert not t.is_alive()

    res, job_resp = observed
    # 提交完成后查询可见完整结果；提交前既不能看到权重也不能看到 succeeded
    assert res.status_code == 200
    assert res.json()["weights"] is not None
    assert job_resp.status_code == 200
    assert job_resp.json()["status"] == "succeeded"
    assert job_resp.json()["weights"] is not None


def test_invalidation_during_save_failure_does_not_revive_job(
        client, monkeypatch):
    import threading

    c, main = client
    setup_feasible(c)
    db_obj = main.app.state.db
    epoch_before = c.get("/health").json()["epoch"]
    correction_done = threading.Event()

    def action(n, real, self, conn):
        if n == PUBLISH_COMMIT_INDEX:
            # 终态发布事务进行中：另一线程提交样本更正（会阻塞在写锁上，
            # 直到发布回滚释放锁）；更正随后使作业永久失效。
            def correct():
                c.put("/samples", json={"samples": FEAS_SAMPLES[:4]})
                correction_done.set()

            threading.Thread(target=correct, daemon=True).start()
            time.sleep(0.3)
            raise RuntimeError("simulated disk I/O error on COMMIT")
        return real(self, conn)

    _commit_plan(db_obj, monkeypatch, action)
    r = c.post("/calculate")
    job_id = r.json()["job_id"]
    assert correction_done.wait(10), "样本更正未生效"
    assert c.get("/health").json()["epoch"] == epoch_before + 1
    time.sleep(0.3)  # 给“保存失败兜底写入”留出时间
    # 已失效作业不得复活
    assert c.get(f"/jobs/{job_id}").status_code == 404
    assert c.get("/result").status_code == 409
    rows = db_obj._conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE job_id=?", (job_id,)
    ).fetchone()
    assert rows["n"] == 0


def test_result_follows_current_epoch_after_new_inputs(client):
    c, _ = client
    setup_feasible(c)
    r = c.post("/calculate")
    first = poll(c, r.json()["job_id"])
    assert first["status"] == "succeeded"
    assert c.get("/result").json()["weights"] == first["weights"]

    # 更正样本（扩大样本集并相应调整目标），旧结果作废
    new_samples = FEAS_SAMPLES + [
        {"person_id": "P9", "region": "North", "age_group": "senior", "d": 1}]
    new_targets = [dict(t) for t in FEAS_TARGETS]
    new_targets = [
        {**t, "population": t["population"] + (
            1 if (t["dimension"], t["code"]) in {("region", "North"),
                                                 ("age_group", "senior")} else 0)}
        for t in new_targets
    ]
    c.put("/samples", json={"samples": new_samples})
    c.put("/targets", json={"targets": new_targets})
    assert c.get("/result").status_code == 409
    r = c.post("/calculate")
    second = poll(c, r.json()["job_id"])
    assert second["status"] == "succeeded", second["error"]
    result = c.get("/result")
    assert result.status_code == 200
    assert result.json()["job_id"] == second["job_id"]
    assert result.json()["input_snapshot"]["epoch"] == second["epoch"]

