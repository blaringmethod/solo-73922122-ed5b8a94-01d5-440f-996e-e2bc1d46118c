"""后台计算调度。

- 计算在固定线程池中执行，HTTP 请求只负责入队，不阻塞样本/目标更正；
- 作业启动前与求解过程中检查 epoch：一旦输入被修改（cancel 事件置位
  或库内 epoch 变化），在途作业立即放弃且不写任何结果；
- 结果发布是一个事务：事务内重新核对作业仍存在、仍属于当前 epoch，
  才写入终态。提交失败即回滚，未提交的权重对任何查询都不可见，
  也不会被“再提交一笔计算”等后续写入顺带保存；
- 保存失败后，仅当作业仍然有效时另行写入 `failed`（无权重）；
  作业若已被样本/目标更正删除，则不做任何写入，绝不复活旧作业。
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from . import solver
from .db import Database

log = logging.getLogger(__name__)


class JobManager:
    def __init__(self, db: Database, max_workers: int = 4):
        self.db = db
        self.pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="weight-worker"
        )

    def submit(self) -> dict:
        # 快照在数据库事务内读取，保证快照与作业 epoch 属于同一版本。
        # 空输入会抛 EmptyInputError，且不会留下作业记录。
        job = self.db.create_job(None)
        self.pool.submit(self._run, job["job_id"])
        return job

    def shutdown(self, wait: bool = False) -> None:
        # wait=True（测试/优雅退出）会等待工作线程结束，释放线程资源。
        self.pool.shutdown(wait=wait, cancel_futures=True)

    # ------------------------------------------------------------------
    def _cancelled(self, job_id: str, job_epoch: int) -> bool:
        if self.db.current_cancel_event().is_set():
            return True
        if self.db.get_epoch() != job_epoch:
            return True
        if self.db.get_job(job_id) is None:  # 已被新 epoch 清除
            return True
        return False

    def _run(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if job is None:
            return
        job_epoch = job["epoch"]
        cancel = self.db.current_cancel_event()
        if self._cancelled(job_id, job_epoch):
            return

        try:
            self.db.mark_running(job_id)
        except Exception:
            log.exception("作业 %s 置为 running 失败", job_id)
            self._mark_save_failed(job_id, job_epoch, "作业状态写入失败")
            return
        if cancel.is_set() or self._cancelled(job_id, job_epoch):
            return

        snapshot = json.loads(job["input_snapshot"])
        samples, targets = snapshot["samples"], snapshot["targets"]

        try:
            # LP 可行性检查点
            if cancel.is_set() or self._cancelled(job_id, job_epoch):
                return
            res = solver.solve_qp(samples, targets)
            if cancel.is_set() or self._cancelled(job_id, job_epoch):
                return
        except solver.SolverError as exc:
            # 数值失败 ≠ 无解：单独标记 failed，保留输入，绝不返回冲突集
            self._publish(job_id, job_epoch, "failed", None, str(exc))
            return
        except Exception as exc:  # pragma: no cover - 防御性
            self._publish(
                job_id, job_epoch, "failed", None,
                f"{type(exc).__name__}: {exc}",
            )
            return

        if res.status == "infeasible":
            iis_rows = [targets[i] for i in res.infeasible_targets]
            self._publish(
                job_id, job_epoch, "infeasible",
                {"infeasible_targets": iis_rows}, None,
            )
            return

        if res.status == "failed":
            self._publish(
                job_id, job_epoch, "failed", None,
                f"数值求解失败（未判定为无解）：{res.reason}",
            )
            return

        # 成功：组装完整结果与输入依据。组装/序列化出错同样视为保存失败，
        # 不发布任何半份权重。
        try:
            weights, totals = self._assemble(samples, targets, res.weights)
            payload = {
                "weights": weights,
                "objective": res.objective,
                "residual_abs_sum": res.residual_abs_sum,
                "residual_max_abs": res.residual_max_abs,
                "target_totals": totals,
            }
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except Exception as exc:
            self._publish(
                job_id, job_epoch, "failed", None,
                f"结果组装失败: {type(exc).__name__}: {exc}",
            )
            return
        self._publish(job_id, job_epoch, "succeeded", payload, None)

    # ------------------------------------------------------------------
    def _publish(self, job_id: str, job_epoch: int, status: str,
                 result: dict | None, error: str | None) -> bool:
        """事务内发布终态；提交失败后尝试把仍有效的作业标记为保存失败。"""
        try:
            return self.db.finish_job(
                job_id, status, result, error, job_epoch=job_epoch
            )
        except Exception as exc:
            log.exception("作业 %s 保存 %s 结果失败", job_id, status)
            self._mark_save_failed(
                job_id, job_epoch,
                f"保存{('结果' if result is not None else '状态')}失败："
                f"{type(exc).__name__}: {exc}",
            )
            return False

    def _mark_save_failed(self, job_id: str, job_epoch: int, error: str) -> bool:
        try:
            # mark_save_failed 内部核对 epoch/作业存在性：
            # 已失效（被更正删除）的作业不会被写入，更不会被复活。
            return self.db.mark_save_failed(job_id, error, job_epoch=job_epoch)
        except Exception:
            log.exception("作业 %s 的失败状态也无法写入数据库", job_id)
            return False

    @staticmethod
    def _assemble(samples, targets, w):
        weights = []
        for s, wi in zip(samples, w):
            weights.append({
                "person_id": s["person_id"],
                "d": s["d"],
                "w": wi,
                "lower": 0.5 * s["d"],
                "upper": 2.0 * s["d"],
            })
        a, p = solver.build_matrix(samples, targets)
        import numpy as np
        wa = np.array(w, dtype=float)
        totals = []
        for j, t in enumerate(targets):
            total = float(a[j] @ wa)
            totals.append({
                "dimension": t["dimension"],
                "code": t["code"],
                "population": t["population"],
                "weighted_total": total,
                "abs_error": abs(total - t["population"]),
            })
        return weights, totals
