#!/usr/bin/env python3
"""电子枪贝叶斯闭环优化器（单文件，仅通过 app.py 子命令驱动）。

真机上只要 ``app.py`` 能跑，本文件就能跑；不依赖 camera_cap/gun_modbus/analysis 内部 API。

用法::

    python bayes_optlr_three.py --max-iter 50 --init-points 15
    python bayes_optlr_three.py --max-iter 30 --plateau-patience 15

每轮流程::

    每组开始前 app.py modbus-off，休整 60s 后 app.py modbus-on
    设置 Ua/Ia/Uc/Ug/If；第 1 张前循环 app.py modbus-read，直到 Ia、Ug 同时连续稳定
    连续三次 app.py capture + app.py analyze；第 2、3 张仅读取一次 FBK，不等待 Ug 稳定
    三张图像的全部 score 均记录；选 score 最接近且相差不超过 10% 的两张，取其平均
    重命名 captured_tif/single.tif / out/single_debug.png 为 iter_XXX_shot_XXX
    每组完成后 app.py modbus-off

失败时返回 ``nan``，优化器会丢弃该组参数并重新采样。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Integer

# 屏蔽 skopt "objective has been evaluated at point ... before" 重复点警告
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="the objective has been evaluated",
)

ROOT = Path(__file__).resolve().parent
# 代码始终从 ROOT 调用 app.py；最终实验结果写入独立目录 RESULT_ROOT。
DEFAULT_RESULT_ROOT = Path(r"D:\RESULT")
RESULT_ROOT = DEFAULT_RESULT_ROOT
DEFAULT_BIAS_V: float = 200.0
_POWER_RECOVER_WAIT_S: float = 5.0
_CAPTURE_PRE_WAIT_S: float = 5.0
_STABILITY_WINDOW_SIZE: int = 5  # 滑动平均窗口大小（个 Ia 采样点）
_STABILITY_ABS_TOL: float = 0.2  # 相邻滑动平均绝对差阈值（uA）
_EXPOSURE_MIN_MS: int = 10       # 曝光数值下限（ms），防异常值
_EXPOSURE_MAX_MS: int = 1000     # 曝光数值上限（ms），防异常值
_SHOTS_PER_EVALUATION: int = 3
_SCORE_RELATIVE_TOLERANCE: float = 0.10
_MIN_VALID_EVALUATIONS: int = 30
_MAX_EVALUATIONS: int = 50
_GROUP_REST_S: float = 60.0
_EI_EXPLORATION_BANDS: int = 5
_LOCAL_SEARCHES_PER_TEN: int = 3

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """调用 app.py 子命令，打印输出，失败时抛异常（check=False 时不抛）。"""
    full_cmd = [sys.executable, "app.py", *cmd]
    print(f"[{_timestamp()}] RUN {' '.join(full_cmd)}")
    proc = subprocess.run(
        full_cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, full_cmd, output=proc.stdout, stderr=proc.stderr
        )
    return proc


def _parse_fbk(output: str) -> dict[str, float] | None:
    """从 app.py modbus-read 输出解析 FBK 行。"""
    for line in output.splitlines():
        if line.strip().startswith("FBK"):
            m = re.search(
                r"Ua\s*=\s*([\d.]+)\s+Ia\s*=\s*([\d.]+)\s+"
                r"Uc\s*=\s*([\d.]+)\s+Ug\s*=\s*([\d.]+)\s+If\s*=\s*([\d.]+)",
                line,
            )
            if m:
                return {
                    "Ua": float(m.group(1)),
                    "Ia": float(m.group(2)),
                    "Uc": float(m.group(3)),
                    "Ug": float(m.group(4)),
                    "If": float(m.group(5)),
                }
    return None


def _parse_analyze(output: str) -> tuple[bool, float, float, float, str]:
    """从 app.py analyze 输出解析 ok/score/x1/x2/reason。"""
    ok = False
    score = x1 = x2 = float("nan")
    reason = ""
    for line in output.splitlines():
        s = line.strip()
        if s.startswith("ok"):
            ok = "True" in s
        elif s.startswith("x1 / x2"):
            m = re.search(r"([\d.]+)\s*/\s*([\d.]+)", s)
            if m:
                x1, x2 = float(m.group(1)), float(m.group(2))
        elif s.startswith("score"):
            m = re.search(r"([\d.]+)", s)
            if m:
                score = float(m.group(1))
        elif s.startswith("reason"):
            reason = s.split(":", 1)[1].strip()
    return ok, score, x1, x2, reason


def _integer_bounds(bounds: tuple[float, float], name: str) -> tuple[int, int]:
    """把连续配置范围收紧为可供贝叶斯搜索的整数闭区间。"""
    lower, upper = math.ceil(float(bounds[0])), math.floor(float(bounds[1]))
    if lower > upper:
        raise ValueError(f"{name} 范围内不存在整数值: {bounds}")
    return lower, upper


def _relative_score_difference(first: float, second: float) -> float:
    """Return |a-b| / min(|a|, |b|), treating two zeros as identical."""
    denominator = min(abs(first), abs(second))
    if denominator == 0:
        return 0.0 if first == second else float("inf")
    return abs(first - second) / denominator


def _normalized_distance(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> float:
    """三维参数按各自量程归一化后的欧氏距离。"""
    total = 0.0
    for value_a, value_b, (lower, upper) in zip(first, second, bounds):
        width = max(upper - lower, 1)
        total += ((value_a - value_b) / width) ** 2
    return math.sqrt(total)


def _random_untried_points(
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    attempted: set[tuple[int, int, int]],
    rng: np.random.Generator,
    *,
    count: int = 12000,
) -> list[tuple[int, int, int]]:
    """生成覆盖整个三维空间的未尝试整数候选池。"""
    candidates: set[tuple[int, int, int]] = set()
    batch_size = max(count * 2, 1000)
    for _ in range(4):
        columns = [
            rng.integers(lower, upper + 1, size=batch_size)
            for lower, upper in bounds
        ]
        candidates.update(zip(*(column.tolist() for column in columns)))
        candidates.difference_update(attempted)
        if len(candidates) >= count:
            break

    # 边界和中心点显式加入候选池，避免随机池遗漏重要区域。
    centers = [(lower + upper) // 2 for lower, upper in bounds]
    for ua in (bounds[0][0], centers[0], bounds[0][1]):
        for ia in (bounds[1][0], centers[1], bounds[1][1]):
            for uc in (bounds[2][0], centers[2], bounds[2][1]):
                point = (ua, ia, uc)
                if point not in attempted:
                    candidates.add(point)
    return list(candidates)[:count]


def _least_sampled_cells(
    candidates: list[tuple[int, int, int]],
    attempted: set[tuple[int, int, int]],
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    *,
    band_count: int = _EI_EXPLORATION_BANDS,
) -> list[tuple[int, int, int]]:
    """保留三维等宽网格中实际尝试次数最少的候选点。"""
    def cell(point: tuple[int, int, int]) -> tuple[int, int, int]:
        indices: list[int] = []
        for value, (lower, upper) in zip(point, bounds):
            width = max(upper - lower + 1, 1)
            indices.append(min(band_count - 1, (value - lower) * band_count // width))
        return tuple(indices)  # type: ignore[return-value]

    counts: dict[tuple[int, int, int], int] = {}
    for point in attempted:
        key = cell(point)
        counts[key] = counts.get(key, 0) + 1
    candidate_cells = {cell(point) for point in candidates}
    least_count = min((counts.get(key, 0) for key in candidate_cells), default=0)
    return [point for point in candidates if counts.get(cell(point), 0) == least_count]


def _suggest_three_parameters(
    opt: Optimizer,
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    observed_points: list[tuple[tuple[int, int, int], float]],
    attempted: set[tuple[int, int, int]],
    rng: np.random.Generator,
) -> tuple[list[int], str]:
    """推荐未重复的 Ua/Ia/Uc：30%最优邻域，70%分层最大不确定性。"""
    candidate_pool = _random_untried_points(bounds, attempted, rng)
    if not candidate_pool:
        raise RuntimeError("三维参数空间内没有可用的未尝试组合")

    # 前 n_initial_points 个点由 skopt 的 LHS 产生；如果遇到重复点，则从
    # 随机候选池选取距已有点最远的点，保持初始覆盖均匀。
    if not opt.models or not observed_points:
        raw = opt.ask()
        proposed = tuple(
            min(max(int(round(float(value))), lower), upper)
            for value, (lower, upper) in zip(raw, bounds)
        )
        if proposed not in attempted:
            return list(proposed), "LHS 三维初始覆盖"
        if not attempted:
            return list(candidate_pool[0]), "LHS 三维初始覆盖"
        supplement = max(
            candidate_pool,
            key=lambda point: min(
                _normalized_distance(point, old, bounds) for old in attempted
            ),
        )
        return list(supplement), "LHS 三维补充覆盖"

    best_point, _ = min(observed_points, key=lambda item: item[1])
    model = opt.models[-1]

    # 每10个有效观测中3次在实测最优点附近继续搜索。距离相同时优先选择
    # GP不确定性更高的候选，避免总沿同一坐标轴移动。
    if len(observed_points) % 10 < _LOCAL_SEARCHES_PER_TEN:
        local_pool = [
            point for point in candidate_pool
            if abs(point[0] - best_point[0]) <= 3
            and abs(point[1] - best_point[1]) <= 6
            and abs(point[2] - best_point[2]) <= 25
        ]
        if not local_pool:
            local_pool = candidate_pool
        distances = np.asarray([
            _normalized_distance(point, best_point, bounds) for point in local_pool
        ])
        nearest = np.flatnonzero(np.isclose(distances, float(np.min(distances))))
        if len(nearest) == 1:
            chosen = local_pool[int(nearest[0])]
        else:
            tied = [local_pool[int(index)] for index in nearest]
            transformed = opt.space.transform([list(point) for point in tied])
            _, std = model.predict(transformed, return_std=True)
            chosen = tied[int(np.argmax(std))]
        return list(chosen), f"三维最优邻域（当前最佳={best_point}）"

    # 其余7/10先限制到采样最少的三维网格，再选GP预测标准差最大的点。
    exploration_pool = _least_sampled_cells(candidate_pool, attempted, bounds)
    if not exploration_pool:
        exploration_pool = candidate_pool
    transformed = opt.space.transform([list(point) for point in exploration_pool])
    _, std = model.predict(transformed, return_std=True)
    max_std = float(np.max(std))
    tied_indices = np.flatnonzero(np.isclose(std, max_std, rtol=1e-12, atol=1e-12))

    # 不确定性并列时选离全部已尝试点最远者，避免固定落在网格起点。
    def _tie_key(index: int) -> float:
        point = exploration_pool[int(index)]
        return min(
            (_normalized_distance(point, old, bounds) for old in attempted),
            default=float("inf"),
        )

    chosen_index = max(tied_indices, key=_tie_key)
    chosen = exploration_pool[int(chosen_index)]
    return list(chosen), (
        f"三维分层最大不确定性（每维{_EI_EXPLORATION_BANDS}段，σ={max_std:.4g}）"
    )


def _wait_beam_stable(
    target_ua: float,
    target_ia: float,
    *,
    eps_v: float,
    samples: int = 5,
    poll_s: float = 2.0,
) -> dict[str, float]:
    """循环读取反馈，确认 Ua、Ia、Ug 稳定并核对 Ua/Ia 设定偏差。

    自身稳定：用最近 _STABILITY_WINDOW_SIZE 个 Ia 算滑动平均；连续 ``samples`` 个
    相邻滑动平均的绝对差 <= _STABILITY_ABS_TOL（uA）。
    稳定条件：最近 ``samples`` 次 Ua、Ia 的极差不超过各自目标值的2%，
    Ug极差不超过 ``eps_v`` 且 1 < Ug < 199。稳定后若 Ua 或 Ia 均值
    偏离设定值超过2%，直接抛出异常并舍弃本组，不把错误点交给GP。
    """
    if target_ua <= 0 or target_ia <= 0:
        raise RuntimeError(f"目标 Ua/Ia 必须为正，got Ua={target_ua}, Ia={target_ia}")

    if eps_v < 0 or samples <= 0 or poll_s < 0:
        raise RuntimeError("Ia/Ug 稳定性参数无效")
    ua_values: list[float] = []
    ia_values: list[float] = []
    ug_values: list[float] = []

    while True:
        proc = _run(["modbus-read"])
        fbk = _parse_fbk(proc.stdout)
        if fbk is None:
            raise RuntimeError("无法从 modbus-read 解析 FBK")
        ua_values.append(float(fbk["Ua"]))
        ia_values.append(float(fbk["Ia"]))
        ug_values.append(float(fbk["Ug"]))
        if len(ia_values) > samples:
            ua_values.pop(0)
            ia_values.pop(0)
            ug_values.pop(0)
        if len(ia_values) == samples:
            ua_mean = sum(ua_values) / len(ua_values)
            ia_mean = sum(ia_values) / len(ia_values)
            ua_span = max(ua_values) - min(ua_values)
            ia_span = max(ia_values) - min(ia_values)
            ug_span = max(ug_values) - min(ug_values)
            if (
                ua_span <= target_ua * 0.02
                and ia_span <= target_ia * 0.02
                and ug_span <= eps_v
                and 1.0 < fbk["Ug"] < 199.0
            ):
                ua_error = abs(ua_mean - target_ua) / target_ua
                ia_error = abs(ia_mean - target_ia) / target_ia
                if ua_error > 0.02 or ia_error > 0.02:
                    raise RuntimeError(
                        "束流已经稳定但设定偏差超限："
                        f"Ua均值={ua_mean:.3f}kV/目标={target_ua:.3f}kV({ua_error:.2%})，"
                        f"Ia均值={ia_mean:.3f}uA/目标={target_ia:.3f}uA({ia_error:.2%})"
                    )
                print(
                    f"[{_timestamp()}] Ua/Ia/Ug 稳定: Ua均值={ua_mean:.3f}kV, "
                    f"Ia均值={ia_mean:.3f}uA, Ua范围={ua_span:.3f}kV, "
                    f"Ia范围={ia_span:.3f}uA, Ug范围={ug_span:.3f}V"
                )
                return fbk
        # ia_window.append(ia)
        # if len(ia_window) > _STABILITY_WINDOW_SIZE:
        #     ia_window.pop(0)
        #
        # if len(ia_window) >= _STABILITY_WINDOW_SIZE:
        #     avg_ia = sum(ia_window) / len(ia_window)
        #     avg_history.append(avg_ia)
        #     if len(avg_history) > samples + 1:
        #         avg_history.pop(0)

            # if len(avg_history) >= samples + 1:
            #     # 检查最近 samples 次相邻滑动平均的绝对差
            #     stable = True
            #     for i in range(samples):
            #         if abs(avg_history[-1 - i] - avg_history[-2 - i]) > _STABILITY_ABS_TOL:
            #             stable = False
            #             break
            #     if stable:
            #         if abs(avg_ia - target_ia) / target_ia <= 0.02 and 1.0 < ug < 199.0:
            #             return fbk
            #         raise RuntimeError(
            #             f"Ia 已自身稳定但参数不满足"
            #             f"（avg={avg_ia:.2f}, target={target_ia:.2f}, Ug={ug:.2f}）"
            #         )

        time.sleep(poll_s)


def _wait_ug_stable(
    *,
    eps_v: float,
    samples: int,
    poll_s: float,
) -> dict[str, float]:
    """Read FBK until the latest ``samples`` Ug readings span no more than eps_v.

    The last FBK is returned and is the value recorded immediately before the
    corresponding image capture.  The 1<Ug<199 guard matches the existing
    beam-ready guard used by this optimizer.
    """
    if eps_v < 0 or samples <= 0:
        raise ValueError("Ug 稳定参数无效")
    readings: list[float] = []
    last_fbk: dict[str, float] | None = None
    while True:
        proc = _run(["modbus-read"])
        fbk = _parse_fbk(proc.stdout)
        if fbk is None:
            raise RuntimeError("无法从 modbus-read 解析 FBK")
        last_fbk = fbk
        readings.append(fbk["Ug"])
        if len(readings) > samples:
            readings.pop(0)
        if len(readings) == samples:
            ug_span = max(readings) - min(readings)
            if ug_span <= eps_v and 1.0 < fbk["Ug"] < 199.0:
                print(
                    f"[{_timestamp()}] Ug 稳定: {fbk['Ug']:.2f}V "
                    f"(最近 {samples} 次范围 {ug_span:.3f}V <= {eps_v:.3f}V)"
                )
                return last_fbk
        time.sleep(poll_s)


class BayesianOptimizer:
    """通过 app.py 子命令驱动的贝叶斯优化器。"""

    def __init__(
        self,
        *,
        eps_v: float = 2.0,
        samples: int = 5,
        poll_s: float = 2.0,
        pre_wait_s: float = 120.0,
        run_id: str | None = None,
    ) -> None:
        self.eps_v = eps_v
        self.samples = samples
        self.poll_s = poll_s
        self.pre_wait_s = pre_wait_s
        self.run_id = run_id or _run_id()
        self.tif_dir = RESULT_ROOT / "captured_tif" / self.run_id
        self.debug_dir = RESULT_ROOT / "out" / self.run_id
        self.tif_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._results: list[dict] = []
        self._failed_records: list[dict] = []
        self._attempt_count: int = 0

    def _record_failure(self, record: dict, reason: str) -> None:
        """记录失败尝试到独立列表，不混入 success history，不影响 GP。"""

        def _clean(v: Any) -> Any:
            if isinstance(v, float) and not np.isfinite(v):
                return None
            return v

        self._failed_records.append({
            "iteration": record.get("iteration"),
            "params": {k: _clean(v) for k, v in record["params"].items()},
            "feedback": {k: _clean(v) for k, v in record["feedback"].items()},
            "reason": reason,
            "image_path": record.get("image_path"),
            "debug_path": record.get("debug_path"),
            "shots": record.get("shots", []),
            "timestamp": _timestamp(),
        })
        print(f"[bayes-optlr] 记录失败 iter {record['iteration']}: {reason}")

    def _set_params(self, params: list[float]) -> None:
        """调用 app.py modbus-set。"""
        _run([
            "modbus-set",
            "--Ua", str(params[0]),
            "--Ia", str(params[1]),
            "--Uc", str(params[2]),
            "--Ug", str(DEFAULT_BIAS_V),
            "--If", str(params[3]),
            # "--wait-bias",
        ])

    def _read_fbk(self) -> dict[str, float] | None:
        """读取一次 FBK，解析失败返回 None。"""
        proc = _run(["modbus-read"], check=False)
        if proc.returncode != 0:
            return None
        return _parse_fbk(proc.stdout)

    def _save_iter_files(self, iteration: int, record: dict, *, shot: int | None = None) -> None:
        """把 single.tif / single_debug.png 归档；重复拍摄使用 shot 后缀。"""
        src_tif = ROOT / "captured_tif" / "single.tif"
        src_debug = ROOT / "out" / "single_debug.png"
        stem = f"iter_{iteration:03d}" if shot is None else f"iter_{iteration:03d}_shot_{shot:03d}"
        dst_tif = self.tif_dir / f"{stem}.tif"
        dst_debug = self.debug_dir / f"{stem}_debug.png"
        if src_tif.is_file():
            shutil.move(str(src_tif), str(dst_tif))
            record["image_path"] = str(dst_tif)
        if src_debug.is_file():
            shutil.move(str(src_debug), str(dst_debug))
            record["debug_path"] = str(dst_debug)

    def _set_and_check(self, params: list[float]) -> dict[str, float] | None:
        """执行 modbus-set 并读取反馈；返回 FBK（解析失败返回 None）。"""
        self._set_params(params)
        time.sleep(_POWER_RECOVER_WAIT_S)
        fbk = self._read_fbk()
        if fbk is not None:
            print(f"[{_timestamp()}] set 后 FBK Ua={fbk['Ua']:.2f}kV")
        return fbk

    def _evaluate(self, x: list[float]) -> float:
        """skopt 目标函数。"""
        self._attempt_count += 1
        iteration = self._attempt_count
        print(f"\n[bayes-optlr] iter {iteration} 开始")
        print(
            f"[bayes-optlr] 提议参数: Ua={x[0]:.2f}kV Ia={x[1]:.2f}uA "
            f"Uc={x[2]:.2f}V If={x[3]:.3f}A"
        )

        record: dict[str, Any] = {
            "iteration": iteration,
            "params": {
                "anode_kv": float(x[0]),
                "anode_ua": float(x[1]),
                "cathode_v": float(x[2]),
                "bias_v": DEFAULT_BIAS_V,
                "fil_a": float(x[3]),
            },
            "feedback": {
                "anode_kv": float("nan"),
                "anode_ua": float("nan"),
                "cathode_v": float("nan"),
                "bias_v": float("nan"),
                "fil_a": float("nan"),
            },
            "score": float("inf"),
            "x1": float("nan"),
            "x2": float("nan"),
            "success": False,
            "reason": "",
            "image_path": None,
            "debug_path": None,
            "shots": [],
            "accepted_shots": [],
            "score_relative_difference": None,
        }

        def _update_feedback_from_fbk(fbk: dict[str, float]) -> None:
            record["feedback"]["anode_kv"] = float(fbk.get("Ua", float("nan")))
            record["feedback"]["anode_ua"] = float(fbk.get("Ia", float("nan")))
            record["feedback"]["cathode_v"] = float(fbk.get("Uc", float("nan")))
            record["feedback"]["bias_v"] = float(fbk.get("Ug", float("nan")))
            record["feedback"]["fil_a"] = float(fbk.get("If", float("nan")))

        fbk: dict[str, float] | None = None
        ok = False
        score = x1 = x2 = float("nan")
        reason = "not_attempted"

        try:
            # 1) 写参并读取反馈
            fbk = self._set_and_check(x)
            if fbk is None:
                reason = "set 后无法读取 FBK"
            else:
                _update_feedback_from_fbk(fbk)
                # 2) 强制等待
                print(f"[{_timestamp()}] 等待 {self.pre_wait_s}s ...")
                time.sleep(self.pre_wait_s)

                # 3) 等束流稳定（Ia 自身滑动平均稳定后，检查 avg Ia 与 Ug）
                print(
                    f"[{_timestamp()}] 等待束流稳定 "
                    f"(Ua≈{x[0]:.2f}kV, Ia≈{x[1]:.2f}uA, 1<Ug<199) ..."
                )
                fbk = _wait_beam_stable(
                    target_ua=x[0],
                    target_ia=x[1],
                    eps_v=self.eps_v,
                    samples=self.samples,
                    poll_s=self.poll_s,
                )
                _update_feedback_from_fbk(fbk)
                print(
                    f"[{_timestamp()}] FBK Ua={fbk['Ua']:.2f} Ia={fbk['Ia']:.2f} "
                    f"Uc={fbk['Uc']:.2f} Ug={fbk['Ug']:.2f} If={fbk['If']:.3f}"
                )

                # 4) 同一候选参数连续拍摄 3 次。每张都独立分析、归档。
                src_tif = ROOT / "captured_tif" / "single.tif"
                src_debug = ROOT / "out" / "single_debug.png"
                for shot_number in range(1, _SHOTS_PER_EVALUATION + 1):
                    shot_record: dict[str, Any] = {
                        "shot": shot_number,
                        "feedback_before_capture": None,
                        "success": False,
                        "score": None,
                        "x1": None,
                        "x2": None,
                        "reason": "",
                        "image_path": None,
                        "debug_path": None,
                    }
                    for temporary in (src_tif, src_debug):
                        if temporary.is_file():
                            temporary.unlink()
                    try:
                        if shot_number == 1:
                            print(
                                f"[{_timestamp()}] 第 1/{_SHOTS_PER_EVALUATION} 张拍照前读取 FBK 并等待 Ug 稳定 ..."
                            )
                            shot_fbk = _wait_ug_stable(
                                eps_v=self.eps_v,
                                samples=self.samples,
                                poll_s=self.poll_s,
                            )
                        else:
                            print(
                                f"[{_timestamp()}] 第 {shot_number}/{_SHOTS_PER_EVALUATION} 张拍照前读取一次当前 FBK（不等待 Ug 稳定）..."
                            )
                            shot_fbk = self._read_fbk()
                            if shot_fbk is None:
                                raise RuntimeError("拍照前无法读取当前 FBK")
                        shot_record["feedback_before_capture"] = {
                            "anode_kv": float(shot_fbk["Ua"]),
                            "anode_ua": float(shot_fbk["Ia"]),
                            "cathode_v": float(shot_fbk["Uc"]),
                            "bias_v": float(shot_fbk["Ug"]),
                            "fil_a": float(shot_fbk["If"]),
                        }
                        _update_feedback_from_fbk(shot_fbk)
                        print(f"[{_timestamp()}] 取图前等待 {_CAPTURE_PRE_WAIT_S:.0f}s ...")
                        time.sleep(_CAPTURE_PRE_WAIT_S)
                        _run(["capture"])
                    except subprocess.CalledProcessError as exc:
                        shot_record["reason"] = f"capture 失败: {exc}"
                    except Exception as exc:
                        shot_record["reason"] = f"拍照前 Ug 稳定检查失败: {exc}"
                    if not shot_record["reason"] and not src_tif.is_file():
                        shot_record["reason"] = "capture 失败，无 tif 生成"
                    if not shot_record["reason"]:
                        ana_proc = _run(["analyze"], check=False)
                        if ana_proc.returncode != 0:
                            shot_record["reason"] = "analyze 命令失败"
                        else:
                            shot_ok, shot_score, shot_x1, shot_x2, shot_reason = _parse_analyze(ana_proc.stdout)
                            if shot_ok and all(np.isfinite(v) for v in (shot_score, shot_x1, shot_x2)):
                                shot_record.update({
                                    "success": True,
                                    "score": float(shot_score),
                                    "x1": float(shot_x1),
                                    "x2": float(shot_x2),
                                })
                            else:
                                shot_record["reason"] = shot_reason or "图像分析不合格"
                    self._save_iter_files(iteration, shot_record, shot=shot_number)
                    record["shots"].append(shot_record)

                # 5) 三拍均保留；在所有有效拍中选 score 最接近的两张。
                valid_shots = [item for item in record["shots"] if item["success"]]
                if len(valid_shots) < 2:
                    ok = False
                    reason = (
                        "三次拍摄中不足两张得到有效 score"
                        f"（有效图 {len(valid_shots)}/{_SHOTS_PER_EVALUATION}）"
                    )
                else:
                    closest_pair, difference = min(
                        (
                            (pair, _relative_score_difference(pair[0]["score"], pair[1]["score"]))
                            for pair in combinations(valid_shots, 2)
                        ),
                        key=lambda item: item[1],
                    )
                    record["score_relative_difference"] = difference
                    if difference > _SCORE_RELATIVE_TOLERANCE:
                        ok = False
                        reason = (
                            "最接近两张有效图的 score 相对差异超过 "
                            f"{_SCORE_RELATIVE_TOLERANCE:.0%}（实际 {difference:.2%}）"
                        )
                    else:
                        first, second = closest_pair
                        score = (float(first["score"]) + float(second["score"])) / 2.0
                        x1 = (float(first["x1"]) + float(second["x1"])) / 2.0
                        x2 = (float(first["x2"]) + float(second["x2"])) / 2.0
                        record["accepted_shots"] = [first["shot"], second["shot"]]
                        record["image_path"] = first.get("image_path")
                        record["debug_path"] = first.get("debug_path")
                        ok = True
                        reason = ""

        except Exception as exc:
            self._save_iter_files(iteration, record)
            self._record_failure(record, f"异常: {exc}")
            print(f"[bayes-optlr] 异常: {exc}", file=sys.stderr)
            return float("nan")

        if not ok:
            self._save_iter_files(iteration, record)
            self._record_failure(record, reason)
            print(f"[bayes-optlr] 尝试失败: {reason}")
            return float("nan")

        # 成功：保存文件 + 记录
        self._save_iter_files(iteration, record)

        record["score"] = score
        record["x1"] = x1
        record["x2"] = x2
        record["success"] = True
        record["reason"] = ""
        self._results.append(record)

        print(
            f"[bayes-optlr] iter {iteration} 完成: score={score:.4f} "
            f"x1={x1:.4f} x2={x2:.4f} "
            f"(shots={record['accepted_shots']}, 差异={record['score_relative_difference']:.2%})"
        )
        return float(score)

    def optimize(
        self,
        limits: dict[str, tuple[float, float] | list[float] | float],
        *,
        max_iterations: int = _MAX_EVALUATIONS,
        n_initial_points: int = 5,
        min_valid_evaluations: int = _MIN_VALID_EVALUATIONS,
        plateau_patience: int = 15,
        min_relative_improvement: float = 0.01,
    ) -> dict | None:
        """执行 Ua、Ia、Uc 三变量联合贝叶斯优化，Ug、If保持固定。

        ``limits`` 须包含：
        - ``anode_kv``: Ua 搜索范围
        - ``anode_ua``: Ia 搜索范围
        - ``cathode_v``: Uc 搜索范围
        - ``fil_a``: 固定 If 单值

        总共最多评估 ``max_iterations`` 个三维候选点（含舍弃组）。
        三拍中至少两张有效且最接近两张的 score 差异不超过 10% 的候选点才会 ``tell`` 给 GP。
        至少得到 ``min_valid_evaluations`` 个有效点后，若连续
        ``plateau_patience`` 个有效点未将当前最优值降低至少
        ``min_relative_improvement``，则提前结束。三个变量均取整数，完整三维组合
        不重复。初始阶段采用三维拉丁超立方抽样；之后每10个有效点中3次搜索
        当前实测最优点邻域，另7次在采样最少的三维分区中选择GP预测标准差最大点。
        """
        ua_limit = limits["anode_kv"]
        ia_limit = limits["anode_ua"]
        uc_limit = limits["cathode_v"]
        if not isinstance(ua_limit, tuple) or not isinstance(ia_limit, tuple) or not isinstance(uc_limit, tuple):
            raise ValueError("三变量程序要求 Ua、Ia、Uc 在配置中均写成 min/max 范围")

        bounds = (
            _integer_bounds(ua_limit, "Ua"),
            _integer_bounds(ia_limit, "Ia"),
            _integer_bounds(uc_limit, "Uc"),
        )
        fixed_if = float(limits["fil_a"])
        exposure_c = float(limits.get("exposure_constant_c", 0.0))

        print("\n[bayes-optlr-three] 开始 Ua/Ia/Uc 三变量联合优化")
        print(
            f"[bayes-optlr-three] Ua={bounds[0]} kV，Ia={bounds[1]} uA，"
            f"Uc={bounds[2]} V；Ug={DEFAULT_BIAS_V:.0f} V、If={fixed_if:.3f} A 固定"
        )
        print(
            f"[bayes-optlr-three] 最多尝试 {max_iterations} 个三维组合；"
            f"至少 {min_valid_evaluations} 个有效三拍组；LHS 初始覆盖 {n_initial_points} 次"
        )
        print(
            f"[bayes-optlr-three] 提前停止：至少 {min_valid_evaluations} 个有效组后，"
            f"连续 {plateau_patience} 个有效组未改善最优 score ≥{min_relative_improvement:.1%}"
        )
        print(
            "[bayes-optlr-three] 三变量均取整数、完整组合不重复；LHS 后 "
            f"3/10 最优邻域 + 7/10 三维分层最大不确定性（每维{_EI_EXPLORATION_BANDS}段）"
        )
        if exposure_c > 0:
            print(f"[bayes-optlr-three] 曝光常数 C={exposure_c:.0f}，每组按推荐Ua/Ia自动更新曝光")
        else:
            print("[bayes-optlr-three] exposure_constant_c 未配置，使用相机配置中的固定曝光")

        dimensions = [
            Integer(*bounds[0], name="anode_kv"),
            Integer(*bounds[1], name="anode_ua"),
            Integer(*bounds[2], name="cathode_v"),
        ]
        opt = Optimizer(
            dimensions=dimensions,
            base_estimator=GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-2,
                normalize_y=True,
                random_state=42,
            ),
            acq_func="EI",
            acq_optimizer="sampling",
            acq_optimizer_kwargs={"n_points": 12000},
            n_initial_points=min(n_initial_points, max_iterations),
            random_state=42,
            initial_point_generator="lhs",
        )

        successful_evals = 0
        attempts = 0
        best_score = float("inf")
        plateau_count = 0
        attempted: set[tuple[int, int, int]] = set()
        observed_points: list[tuple[tuple[int, int, int], float]] = []
        rng = np.random.default_rng(42)

        while attempts < max_iterations:
            x, mode = _suggest_three_parameters(opt, bounds, observed_points, attempted, rng)
            point = (int(x[0]), int(x[1]), int(x[2]))
            if point in attempted:
                raise RuntimeError(f"推荐器返回了已尝试的重复三维组合: {point}")
            attempted.add(point)
            full_x = [float(point[0]), float(point[1]), float(point[2]), fixed_if]
            group_number = attempts + 1
            print(
                f"[bayes-optlr-three] {mode} 推荐 Ua={point[0]}kV "
                f"Ia={point[1]}uA Uc={point[2]}V（第 {group_number}/{max_iterations} 组）"
            )

            if exposure_c > 0:
                _update_camera_exposure(point[0], point[1], exposure_c)

            print(f"[{_timestamp()}] 第 {group_number} 组开始前：modbus-off")
            _run(["modbus-off"], check=False)
            print(
                f"[{_timestamp()}] 第 {group_number} 组：设备休整 "
                f"{_GROUP_REST_S:.0f}s 后重新开启 ..."
            )
            time.sleep(_GROUP_REST_S)
            print(f"[{_timestamp()}] 第 {group_number} 组：modbus-on")
            _run(["modbus-on"])
            try:
                y = self._evaluate(full_x)
            finally:
                print(f"[{_timestamp()}] 第 {group_number} 组完成：modbus-off")
                _run(["modbus-off"], check=False)

            attempts += 1
            if np.isfinite(y):
                opt.tell(list(point), float(y))
                successful_evals += 1
                observed_points.append((point, float(y)))
                if y < best_score * (1.0 - min_relative_improvement):
                    best_score = float(y)
                    plateau_count = 0
                else:
                    plateau_count += 1
                if (
                    successful_evals >= min_valid_evaluations
                    and plateau_count >= plateau_patience
                ):
                    print(
                        f"[bayes-optlr-three] 提前终止：已有效 {successful_evals} 组，"
                        f"连续 {plateau_count} 个有效组无显著改善"
                    )
                    break
            else:
                # 舍弃组不进入GP，但占用一次总尝试且该完整组合不再推荐。
                if opt._n_initial_points > 0:
                    opt._n_initial_points -= 1
                    opt.cache_ = {}

        print(
            f"[bayes-optlr-three] 优化结束：{successful_evals}/{attempts} 次有效评估，"
            f"尝试了 {len(attempted)} 个不同三维组合"
        )

        successful = [r for r in self._results if r["success"] and np.isfinite(r["score"])]
        if not successful:
            print("[bayes-optlr] 没有成功的迭代")
            return None
        return min(successful, key=lambda r: r["score"])

    def save_failed_records(self, path: Path) -> None:
        if not self._failed_records:
            return
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._failed_records, f, indent=2, ensure_ascii=False)
        print(f"[bayes-optlr] 失败记录已保存: {path}")

    def save_history(self, path: Path) -> None:
        summary = {
            "successful": sum(1 for r in self._results if r["success"]),
            "total_successful": len(self._results),
            "failed_attempts": len(self._failed_records),
            "total_attempts": len(self._results) + len(self._failed_records),
            "best": None,
            "top5": [],
        }
        ok_results = [r for r in self._results if r["success"] and np.isfinite(r["score"])]
        if ok_results:
            ok_results_sorted = sorted(ok_results, key=lambda r: r["score"])
            best = ok_results_sorted[0]
            summary["best"] = {
                "iteration": best["iteration"],
                "params": best["params"],
                "score": best["score"],
                "x1": best["x1"],
                "x2": best["x2"],
                "image_path": best.get("image_path"),
                "debug_path": best.get("debug_path"),
            }
            summary["top5"] = [
                {
                    "iteration": r["iteration"],
                    "params": r["params"],
                    "score": r["score"],
                    "x1": r["x1"],
                    "x2": r["x2"],
                }
                for r in ok_results_sorted[:5]
            ]

        payload = {
            "summary": summary,
            "results": self._results,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[bayes-optlr] 历史已保存: {path}")

    def save_text_report(self, path: Path, *, settings: dict[str, Any]) -> None:
        """Export execution order and score ranking as one readable Markdown report."""
        successful = sorted(
            (record for record in self._results if record["success"] and np.isfinite(record["score"])),
            key=lambda record: record["iteration"],
        )
        failed = sorted(self._failed_records, key=lambda record: record.get("iteration", 0))
        sequence = sorted(
            [*successful, *failed], key=lambda record: record.get("iteration", 0)
        )

        def _number(value: Any, digits: int = 4) -> str:
            return "-" if value is None or not isinstance(value, (int, float)) or not np.isfinite(value) else f"{float(value):.{digits}f}"

        def _params(record: dict[str, Any]) -> str:
            params = record.get("params") or {}
            return (
                f"Ua={_number(params.get('anode_kv'), 2)} kV, "
                f"Ia={_number(params.get('anode_ua'), 2)} uA, "
                f"Uc={_number(params.get('cathode_v'), 2)} V, "
                f"Ug={_number(params.get('bias_v'), 2)} V, "
                f"If={_number(params.get('fil_a'), 4)} A"
            )

        def _shot_scores(record: dict[str, Any]) -> str:
            shots = record.get("shots") or []
            if not shots:
                return "-"
            return ", ".join(
                f"{item.get('shot', '?')}:{_number(item.get('score'))}"
                for item in shots
            )

        def _shot_ug(record: dict[str, Any]) -> str:
            shots = record.get("shots") or []
            values = []
            for item in shots:
                feedback = item.get("feedback_before_capture") or {}
                values.append(f"{item.get('shot', '?')}:{_number(feedback.get('bias_v'), 2)}")
            return ", ".join(values) if values else "-"

        lines = [
            "# Bayes OptLR Ua/Ia/Uc 三变量优化报告",
            "",
            f"- 运行编号: `{self.run_id}`",
            f"- 有效三拍组: {len(successful)}",
            f"- 舍弃组: {len(failed)}",
            f"- 候选组总数: {len(sequence)}",
            f"- 每组拍摄: {settings['shots_per_evaluation']} 次；接受条件：全部有效图中 score 最接近的两张相对差异不超过 {settings['score_relative_tolerance']:.0%}，最终 score/x1/x2 取该两张平均。",
            f"- 停止规则：至少 {settings['min_valid_evaluations']} 个有效组后，连续 {settings['plateau_patience']} 个有效组未让最优 score 下降 ≥ {settings['min_relative_improvement']:.1%}",
            f"- 上限: 三变量联合优化最多 {settings['max_iterations']} 个候选组",
            f"- 参数推荐: Ua/Ia/Uc均取整数且完整组合不重复；初始三维LHS覆盖后，每10个有效观测中3次搜索实测最优点邻域、7次在采样最少的三维分区中取GP预测不确定性最大的组合。",
            f"- 设备周期: 每组开始前关闭设备并休整 {settings['group_rest_s']:.0f} 秒，之后重新开启；组完成后再次关闭。",
            "",
            "## 贝叶斯优化实际顺序",
            "",
            "| 顺序 | 状态 | 输入参数 | 拍前 Ug(V) | 三拍 score | 选取拍次 | 选中两拍误差 | 最终 score | x1 | x2 | 原因 |",
            "| ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for record in sequence:
            is_success = bool(record.get("success"))
            selected = ", ".join(str(value) for value in (record.get("accepted_shots") or [])) or "-"
            reason = record.get("reason") or ("接受" if is_success else "舍弃")
            difference = record.get("score_relative_difference")
            difference_text = "-" if difference is None or not isinstance(difference, (int, float)) or not np.isfinite(difference) else f"{float(difference):.2%}"
            lines.append(
                f"| {record.get('iteration', '-')} | {'有效' if is_success else '舍弃'} | "
                f"{_params(record)} | {_shot_ug(record)} | {_shot_scores(record)} | {selected} | {difference_text} | "
                f"{_number(record.get('score'))} | {_number(record.get('x1'))} | "
                f"{_number(record.get('x2'))} | {reason} |"
            )

        lines.extend([
            "",
            "## score 排名（由小到大）",
            "",
            "| 排名 | 迭代 | 最终 score | 三拍 score | 选中两拍误差 | x1 | x2 | Ua (kV) | Ia (uA) | Uc (V) | 选取拍次 | 原图 | 调试图 |",
            "| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ])
        for rank, record in enumerate(sorted(successful, key=lambda item: item["score"]), 1):
            params = record["params"]
            selected = ", ".join(str(value) for value in (record.get("accepted_shots") or [])) or "-"
            difference = record.get("score_relative_difference")
            difference_text = "-" if difference is None or not isinstance(difference, (int, float)) or not np.isfinite(difference) else f"{float(difference):.2%}"
            lines.append(
                f"| {rank} | {record['iteration']} | {_number(record['score'])} | "
                f"{_shot_scores(record)} | {difference_text} | {_number(record['x1'])} | {_number(record['x2'])} | "
                f"{_number(params.get('anode_kv'), 2)} | {_number(params.get('anode_ua'), 2)} | "
                f"{_number(params.get('cathode_v'), 2)} | {selected} | "
                f"`{record.get('image_path') or '-'}` | `{record.get('debug_path') or '-'}` |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[bayes-optlr] 文字报告已保存: {path}")


def _compute_exposure_ms(ua_kv: float, ia_ua: float, constant_c: float) -> int:
    """根据曝光常数 C = Ua^2 * Ia * exposure_time 计算曝光数值（ms）。"""
    raw = int(round(constant_c / (ua_kv * ua_kv * ia_ua)))
    if raw < _EXPOSURE_MIN_MS:
        print(
            f"[bayes-optlr] WARN 曝光 {raw}ms < {_EXPOSURE_MIN_MS}ms，裁剪到下限 "
            f"(Ua={ua_kv} Ia={ia_ua} C={constant_c})",
            file=sys.stderr,
        )
        return _EXPOSURE_MIN_MS
    if raw > _EXPOSURE_MAX_MS:
        print(
            f"[bayes-optlr] WARN 曝光 {raw}ms > {_EXPOSURE_MAX_MS}ms，裁剪到上限 "
            f"(Ua={ua_kv} Ia={ia_ua} C={constant_c})",
            file=sys.stderr,
        )
        return _EXPOSURE_MAX_MS
    return raw


def _update_camera_exposure(ua_kv: float, ia_ua: float, constant_c: float) -> int:
    """根据 (Ua, Ia) 计算曝光时间并写回 camera_cap_config.json。返回写入的值。"""
    if constant_c <= 0:
        print("[bayes-optlr] exposure_constant_c 未配置，跳过曝光更新")
        return 0
    exposure_ms = _compute_exposure_ms(ua_kv, ia_ua, constant_c)
    config_path = ROOT / "camera_cap_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cam_cfg = json.load(f)
    # Legacy JSON field name; the active SDK integer API interprets it as ms.
    cam_cfg["exposure_time_us"] = exposure_ms
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cam_cfg, f, indent=2, ensure_ascii=False)
    print(
        f"[bayes-optlr] Ua={ua_kv:.0f}kV Ia={ia_ua:.0f}uA "
        f"曝光时间={exposure_ms}ms (C={constant_c:.0f})"
    )
    return exposure_ms


def _load_limits(gun_config_path: Path) -> dict[str, tuple[float, float] | list[float] | float]:
    """从 gun_modbus_config.json 解析参数。

    支持三种写法：
    - 连续范围：{"min": a, "max": b} -> 返回 (a, b)
    - 枚举列表：[70, 90, 110] -> 返回 [70.0, 90.0, 110.0]
    - 固定单值：0.422 -> 返回 0.422
    """
    import json as _json

    with gun_config_path.open("r", encoding="utf-8") as f:
        raw = _json.load(f)
    pl = raw.get("param_limits", {})

    def _get(key: str, default: tuple[float, float]):
        block = pl.get(key, default)
        if isinstance(block, (int, float)):
            return float(block)
        if isinstance(block, list):
            return [float(v) for v in block]
        if isinstance(block, dict):
            return (float(block.get("min", default[0])), float(block.get("max", default[1])))
        return default

    exposure_c = pl.get("exposure_constant_c", 0.0)
    exposure_c = float(exposure_c) if isinstance(exposure_c, (int, float)) else 0.0

    return {
        "anode_kv": _get("anode_kv", (70.0, 110.0)),
        "anode_ua": _get("anode_ua", (30.0, 300.0)),
        "cathode_v": _get("cathode_v", (500.0, 750.0)),
        "fil_a": _get("fil_a", (0.42, 0.46)),
        "exposure_constant_c": exposure_c,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="电子枪 Ua/Ia/Uc 三变量贝叶斯闭环优化（通过 app.py 驱动）")
    parser.add_argument(
        "--max-iter", type=int, default=_MAX_EVALUATIONS,
        help="Ua/Ia/Uc三维优化最多候选组数，范围30--50（含舍弃组）",
    )
    parser.add_argument("--init-points", type=int, default=15, help="LHS 均匀随机初始采样点数（默认前 15 次）")
    parser.add_argument(
        "--min-valid-evals", type=int, default=_MIN_VALID_EVALUATIONS,
        help="提前停止前至少需要的有效三拍组数，不能低于 30",
    )
    parser.add_argument(
        "--plateau-patience", type=int, default=15,
        help="有效 score 连续多少组无显著下降后可提前停止",
    )
    parser.add_argument(
        "--min-relative-improvement", type=float, default=0.01,
        help="score 相对当前最优值至少下降该比例才算改善，默认 0.01（1%）",
    )
    parser.add_argument(
        "--target-score",
        type=float,
        default=None,
        help="目标 score（仅记录，暂未用于提前停止）",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="优化历史保存路径，默认 bayes_opt_history_<run_id>.json",
    )
    parser.add_argument(
        "--gun-config",
        type=Path,
        default=ROOT / "gun_modbus_config_bayes_three.json",
        help="Ua/Ia/Uc三变量优化专用电子枪配置",
    )
    parser.add_argument("--eps-v", type=float, default=1.0, help="Ug 稳定阈值 (V)")
    parser.add_argument("--samples", type=int, default=6, help="Ug 连续稳定次数")
    parser.add_argument("--poll-s", type=float, default=5.0, help="Ug 查询间隔 (s)")
    parser.add_argument("--pre-wait-s", type=float, default=5.0, help="set_params 后强制等待 (s)")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="运行标识，用于创建独立输出目录（默认当前时间 YYYYMMDD_HHMMSS）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=r"结果根目录，默认 D:\\RESULT（其下创建 captured_tif、out、reports）",
    )
    args = parser.parse_args()
    if not _MIN_VALID_EVALUATIONS <= args.max_iter <= _MAX_EVALUATIONS:
        parser.error("max-iter 必须在 30 到 50 之间")
    if args.min_valid_evals < _MIN_VALID_EVALUATIONS or args.min_valid_evals > args.max_iter:
        parser.error("min-valid-evals 必须不小于 30，且不能大于 max-iter")
    if args.plateau_patience <= 0:
        parser.error("plateau-patience 必须为正整数")
    if not 0 < args.min_relative_improvement < 1:
        parser.error("min-relative-improvement 必须在 0 和 1 之间")
    global RESULT_ROOT
    RESULT_ROOT = args.output_root.resolve()
    args.gun_config = args.gun_config.resolve()
    os.environ["GUN_MODBUS_CONFIG"] = str(args.gun_config)
    limits = _load_limits(args.gun_config)

    try:
        opt = BayesianOptimizer(
            eps_v=args.eps_v,
            samples=args.samples,
            poll_s=args.poll_s,
            pre_wait_s=args.pre_wait_s,
            run_id=args.run_id,
        )
        reports_dir = RESULT_ROOT / "reports" / opt.run_id
        reports_dir.mkdir(parents=True, exist_ok=True)
        history_path = args.history or reports_dir / f"bayes_optlr_three_history_{opt.run_id}.json"
        best = opt.optimize(
            limits,
            max_iterations=args.max_iter,
            n_initial_points=args.init_points,
            min_valid_evaluations=args.min_valid_evals,
            plateau_patience=args.plateau_patience,
            min_relative_improvement=args.min_relative_improvement,
        )
        opt.save_history(history_path)

        failed_path = reports_dir / f"bayes_optlr_three_failed_{opt.run_id}.json"
        opt.save_failed_records(failed_path)

        text_report_path = reports_dir / f"bayes_optlr_three_report_{opt.run_id}.md"
        opt.save_text_report(
            text_report_path,
            settings={
                "shots_per_evaluation": _SHOTS_PER_EVALUATION,
                "score_relative_tolerance": _SCORE_RELATIVE_TOLERANCE,
                "min_valid_evaluations": args.min_valid_evals,
                "plateau_patience": args.plateau_patience,
                "min_relative_improvement": args.min_relative_improvement,
                "max_iterations": args.max_iter,
                "group_rest_s": _GROUP_REST_S,
            },
        )

        try:
            from scripts.export_bayes_ranking import export_ranking

            ranking_path = reports_dir / f"bayes_optlr_three_ranking_{opt.run_id}.md"
            export_ranking(history_path, ranking_path)
        except Exception as exc:
            print(f"[bayes-optlr] 导出排名报告失败: {exc}", file=sys.stderr)

        if best is None:
            print("[bayes-optlr] 优化结束，无成功结果")
            return 1

        print("\n" + "=" * 60)
        print("[bayes-optlr] 优化完成")
        print(f"  最优 score : {best['score']:.4f}")
        print(f"  最优 x1/x2 : {best['x1']:.4f} / {best['x2']:.4f}")
        p = best["params"]
        print(
            f"  最优参数   : Ua={p['anode_kv']}kV Ia={p['anode_ua']}uA "
            f"Uc={p['cathode_v']}V Ug={p['bias_v']}V If={p['fil_a']}A"
        )
        print(f"  图像       : {best.get('image_path')}")
        print(f"  debug      : {best.get('debug_path')}")
        print("=" * 60)
        return 0
    except Exception as exc:
        print(f"[bayes-optlr] FATAL: {exc}", file=sys.stderr)
        return 1
    finally:
        print(f"[{_timestamp()}] modbus-off")
        _run(["modbus-off"], check=False)


if __name__ == "__main__":
    np.random.seed(42)
    raise SystemExit(main())
