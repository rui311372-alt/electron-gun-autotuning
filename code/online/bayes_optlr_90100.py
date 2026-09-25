#!/usr/bin/env python3
"""电子枪贝叶斯闭环优化器（单文件，仅通过 app.py 子命令驱动）。

真机上只要 ``app.py`` 能跑，本文件就能跑；不依赖 camera_cap/gun_modbus/analysis 内部 API。

用法::

    python bayes_optlr_90100.py
    python bayes_optlr_90100.py --run-id test_90100

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
_SHORT_MIN_VALID_EVALUATIONS: int = 30
_SHORT_MAX_EVALUATIONS: int = 50
_LONG_MIN_VALID_EVALUATIONS: int = 60
_LONG_MAX_EVALUATIONS: int = 80
_DEFAULT_ROUNDS_PER_BUDGET: int = 2
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


def _integer_uc_grid(bounds: tuple[float, float]) -> list[int]:
    """Return every legal integer Uc value inside the configured range."""
    lower, upper = math.ceil(float(bounds[0])), math.floor(float(bounds[1]))
    if lower > upper:
        raise ValueError(f"Uc 范围内不存在整数值: {bounds}")
    return list(range(lower, upper + 1))


def _relative_score_difference(first: float, second: float) -> float:
    """Return |a-b| / min(|a|, |b|), treating two zeros as identical."""
    denominator = min(abs(first), abs(second))
    if denominator == 0:
        return 0.0 if first == second else float("inf")
    return abs(first - second) / denominator


def _least_sampled_uc_candidates(
    available_uc: list[int],
    attempted_uc: list[int],
    *,
    band_count: int = _EI_EXPLORATION_BANDS,
) -> list[int]:
    """Return untried Uc values in the least-sampled equal-width search band."""
    if not available_uc:
        return []
    all_values = [*available_uc, *attempted_uc]
    lower, upper = min(all_values), max(all_values)
    width = max(upper - lower + 1, 1)

    def band(value: int) -> int:
        return min(band_count - 1, (value - lower) * band_count // width)

    counts = [0] * band_count
    for value in attempted_uc:
        counts[band(value)] += 1
    candidate_bands = {band(value) for value in available_uc}
    least_count = min(counts[index] for index in candidate_bands)
    return [
        value for value in available_uc
        if counts[band(value)] == least_count
    ]


def _suggest_integer_uc(
    opt: Optimizer,
    available_uc: list[int],
    observed_points: list[tuple[int, float]],
    attempted_uc: list[int],
) -> tuple[list[int], str]:
    """Choose an untried Uc using 30% local search and 70% uncertainty exploration."""
    if not available_uc:
        raise ValueError("没有未尝试的整数 Uc 候选点")

    # LHS 初始阶段优先沿用 skopt 的建议；若建议已因失败尝试过，则采用
    # 与历史点距离最大的剩余整数，保持初始覆盖而不随机替换。
    if not opt.models or not observed_points:
        proposed = int(round(float(opt.ask()[0])))
        if proposed in available_uc:
            return [proposed], "LHS 初始覆盖"
        if not attempted_uc:
            return [available_uc[0]], "LHS 初始覆盖"
        return [max(available_uc, key=lambda value: min(abs(value - old) for old in attempted_uc))], "LHS 补充覆盖"

    # 每 10 个有效观测中，3 次围绕当前实测最优点扩展尚未测过的整数邻域；
    # 其余 7 次用分层最大不确定性探索：只在实际尝试次数最少的 1/5 Uc
    # 区段中挑选预测标准差最大的点，避免“全局探索”仍被局部低分区吸走。
    best_uc, _ = min(observed_points, key=lambda item: item[1])
    if len(observed_points) % 10 < _LOCAL_SEARCHES_PER_TEN:
        local_uc = min(available_uc, key=lambda value: (abs(value - best_uc), value))
        return [local_uc], f"最优邻域（当前最佳 Uc={best_uc}V）"

    exploration_uc = _least_sampled_uc_candidates(available_uc, attempted_uc)
    if not exploration_uc:
        exploration_uc = available_uc
    model = opt.models[-1]
    transformed = opt.space.transform([[value] for value in exploration_uc])
    _, std = model.predict(transformed, return_std=True)
    max_std = float(np.max(std))
    tied_indices = np.flatnonzero(np.isclose(std, max_std, rtol=1e-12, atol=1e-12))

    # 不确定性完全相同（常见于模型尚无足够信息）时，优先补足最大的未测间隔，
    # 再以候选范围中部为次级规则，避免 np.argmax 固定取升序列表中的区段起点。
    center = (min(exploration_uc) + max(exploration_uc)) / 2.0
    def _tie_key(index: int) -> tuple[float, float]:
        value = exploration_uc[int(index)]
        distance = min(abs(value - old) for old in attempted_uc) if attempted_uc else float("inf")
        return distance, -abs(value - center)

    chosen_index = max(tied_indices, key=_tie_key)
    return [exploration_uc[int(chosen_index)]], (
        f"分层最大不确定性探索（{_EI_EXPLORATION_BANDS} 段中尝试最少区段，σ={max_std:.4g}）"
    )


def _wait_beam_stable(
    target_ia: float,
    *,
    eps_v: float,
    samples: int = 5,
    poll_s: float = 2.0,
) -> dict[str, float]:
    """循环 modbus-read，先判断 Ia 自身稳定，再判断参数稳定。返回最后一次 FBK。

    自身稳定：用最近 _STABILITY_WINDOW_SIZE 个 Ia 算滑动平均；连续 ``samples`` 个
    相邻滑动平均的绝对差 <= _STABILITY_ABS_TOL（uA）。
    参数稳定：自身稳定后，当前滑动平均 Ia 在目标值 ±2% 且 1 < Ug < 199。
    Ia 自身一定会稳定，因此不设采样上限。
    """
    if target_ia <= 0:
        raise RuntimeError(f"目标 Ia 必须为正，got {target_ia}")

    if eps_v < 0 or samples <= 0 or poll_s < 0:
        raise RuntimeError("Ia/Ug 稳定性参数无效")
    ia_values: list[float] = []
    ug_values: list[float] = []

    while True:
        proc = _run(["modbus-read"])
        fbk = _parse_fbk(proc.stdout)
        if fbk is None:
            raise RuntimeError("无法从 modbus-read 解析 FBK")
        ia_values.append(float(fbk["Ia"]))
        ug_values.append(float(fbk["Ug"]))
        if len(ia_values) > samples:
            ia_values.pop(0)
            ug_values.pop(0)
        if len(ia_values) == samples:
            ia_mean = sum(ia_values) / len(ia_values)
            ia_span = max(ia_values) - min(ia_values)
            ug_span = max(ug_values) - min(ug_values)
            if (
                abs(ia_mean - target_ia) / target_ia <= 0.02
                and ia_span <= target_ia * 0.02
                and ug_span <= eps_v
                and 1.0 < fbk["Ug"] < 199.0
            ):
                print(
                    f"[{_timestamp()}] Ia/Ug 稳定: Ia均值={ia_mean:.3f}uA, "
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
                print(f"[{_timestamp()}] 等待束流稳定 (Ia≈{x[1]:.2f}uA, 1<Ug<199) ...")
                fbk = _wait_beam_stable(
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

    def _evaluate_fixed_ua_ia_if(
        self,
        x: list[float],
        *,
        ua: float,
        ia: float,
        if_value: float,
    ) -> float:
        """在 Ua、Ia、If 固定时，对 [Uc] 做目标函数评估。"""
        full_x = [ua, ia, x[0], if_value]
        return self._evaluate(full_x)

    def optimize(
        self,
        limits: dict[str, tuple[float, float] | list[float] | float],
        *,
        max_iterations: int = _LONG_MAX_EVALUATIONS,
        n_initial_points: int = 5,
        min_valid_evaluations: int = _LONG_MIN_VALID_EVALUATIONS,
        plateau_patience: int = 15,
        min_relative_improvement: float = 0.01,
    ) -> dict | None:
        """执行网格贝叶斯优化：固定 Ua/Ia/If/Ug，只优化 Uc。

        ``limits`` 须包含：
        - ``anode_kv``: Ua 枚举列表
        - ``anode_ua``: Ia 枚举列表
        - ``cathode_v``: Uc 搜索范围
        - ``fil_a``: 固定 If 单值

        每个 (Ua, Ia) 组合最多评估 ``max_iterations`` 个候选点（含舍弃组）。
        三拍中至少两张有效且最接近两张的 score 差异不超过 10% 的候选点才会 ``tell`` 给 GP。
        至少得到 ``min_valid_evaluations`` 个有效点后，若连续
        ``plateau_patience`` 个有效点未将当前最优值降低至少
        ``min_relative_improvement``，则提前结束该组合。
        Uc 仅取搜索范围内的整数，且每个整数最多实际尝试一次。初始阶段采用
        拉丁超立方抽样覆盖空间；之后每 10 个有效点中 3 次取当前最佳点邻域，
        另 7 次仅在尝试次数最少的 Uc 分区中选择 GP 预测标准差最大的点，以兼顾
        局部收敛与全局覆盖。
        """
        ua_values = [float(v) for v in limits["anode_kv"]]
        ia_values = [float(v) for v in limits["anode_ua"]]
        fixed_if = float(limits["fil_a"])
        exposure_c = float(limits.get("exposure_constant_c", 0.0))

        print(f"\n[bayes-optlr] 开始网格优化: Ua={ua_values} Ia={ia_values} If={fixed_if}")
        print(f"[bayes-optlr] Ua/Ia 组合数: {len(ua_values) * len(ia_values)}")
        print(
            f"[bayes-optlr] 每组合最多尝试 Uc {max_iterations} 次；"
            f"至少 {min_valid_evaluations} 个有效三拍组；LHS 初始覆盖 {n_initial_points} 次"
        )
        print(
            f"[bayes-optlr] 提前停止：至少 {min_valid_evaluations} 个有效组后，"
            f"连续 {plateau_patience} 个有效组未改善最优 score ≥{min_relative_improvement:.1%}"
        )
        print(
            "[bayes-optlr] Uc 推荐: 仅整数、全程不重复；LHS 后 3/10 最优邻域 + "
            f"7/10 分层最大不确定性（{_EI_EXPLORATION_BANDS} 个等宽区段）"
        )
        print(f"[bayes-optlr] Uc 搜索空间: {limits['cathode_v']}")
        if exposure_c > 0:
            print(f"[bayes-optlr] 曝光常数 C={exposure_c:.0f}，每组合自动更新曝光时间")
        else:
            print("[bayes-optlr] exposure_constant_c 未配置，使用 camera_cap_config.json 中的固定曝光")

        uc_grid = _integer_uc_grid(limits["cathode_v"])
        uc_search_space = [Integer(min(uc_grid), max(uc_grid), name="cathode_v")]

        for ua in ua_values:
            for ia in ia_values:
                print(f"\n[bayes-optlr] ===== Ua={ua:.0f}kV Ia={ia:.0f}uA =====")

                if exposure_c > 0:
                    _update_camera_exposure(ua, ia, exposure_c)

                opt = Optimizer(
                    dimensions=uc_search_space,
                    base_estimator=GaussianProcessRegressor(
                        kernel=Matern(nu=2.5),
                        alpha=1e-2,  # 噪声方差，让 GP 对 750 附近保留不确定性
                        normalize_y=True,
                        random_state=42,
                    ),
                    acq_func="EI",
                    acq_optimizer="sampling",
                    acq_optimizer_kwargs={"n_points": 10000},
                    n_initial_points=min(n_initial_points, max_iterations),
                    random_state=42,
                    initial_point_generator="lhs",
                )

                successful_evals = 0
                attempts = 0
                best_score = float("inf")
                plateau_count = 0
                attempted_uc: list[int] = []
                observed_points: list[tuple[int, float]] = []
                while attempts < max_iterations:
                    available_uc = [value for value in uc_grid if value not in attempted_uc]
                    if not available_uc:
                        print("[bayes-optlr] 所有整数 Uc 均已尝试，结束当前组合")
                        break
                    x, mode = _suggest_integer_uc(opt, available_uc, observed_points, attempted_uc)
                    print(
                        f"[bayes-optlr] {mode} 推荐 Uc={x[0]:.0f}V "
                        f"（剩余未尝试整数 {len(available_uc)} 个）"
                    )
                    attempted_uc.append(int(x[0]))
                    group_number = attempts + 1
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
                        y = self._evaluate_fixed_ua_ia_if(
                            x, ua=ua, ia=ia, if_value=fixed_if
                        )
                    finally:
                        print(f"[{_timestamp()}] 第 {group_number} 组完成：modbus-off")
                        _run(["modbus-off"], check=False)
                    attempts += 1
                    if np.isfinite(y):
                        opt.tell(x, y)
                        successful_evals += 1
                        observed_points.append((int(x[0]), float(y)))
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
                                f"[bayes-optlr] Ua={ua} Ia={ia} 终止："
                                f"已有效 {successful_evals} 组，连续 {plateau_count} 组无显著改善"
                            )
                            break
                    else:
                        # 失败：不记录到 GP（零影响）
                        if opt._n_initial_points > 0:
                            # 舍弃组不 tell 给 GP；推进初始计数，避免初始阶段无限延长。
                            opt._n_initial_points -= 1
                            opt.cache_ = {}

                print(
                    f"[bayes-optlr] Ua={ua} Ia={ia} 完成: "
                    f"{successful_evals}/{attempts} 次有效评估"
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
            "# Bayes OptLR 优化报告",
            "",
            f"- 运行编号: `{self.run_id}`",
            f"- 有效三拍组: {len(successful)}",
            f"- 舍弃组: {len(failed)}",
            f"- 候选组总数: {len(sequence)}",
            f"- 每组拍摄: {settings['shots_per_evaluation']} 次；接受条件：全部有效图中 score 最接近的两张相对差异不超过 {settings['score_relative_tolerance']:.0%}，最终 score/x1/x2 取该两张平均。",
            f"- 停止规则：至少 {settings['min_valid_evaluations']} 个有效组后，连续 {settings['plateau_patience']} 个有效组未让最优 score 下降 ≥ {settings['min_relative_improvement']:.1%}",
            f"- 上限: 每个 Ua/Ia 组合最多 {settings['max_iterations']} 个候选组",
            f"- Uc 推荐: 仅取整数且不重复；初始 LHS 覆盖后，每 10 个有效观测中 3 次测最优邻域、7 次在尝试最少的 {_EI_EXPLORATION_BANDS} 个等宽分区中取 GP 预测不确定性最大的 Uc。",
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
    parser = argparse.ArgumentParser(description="电子枪贝叶斯闭环优化（通过 app.py 驱动）")
    parser.add_argument("--init-points", type=int, default=15, help="LHS 均匀随机初始采样点数（默认前 15 次）")
    parser.add_argument(
        "--budget",
        choices=("short", "long", "both"),
        default="both",
        help="运行预算：short=短预算，long=长预算，both=先短后长（默认 both）",
    )
    parser.add_argument(
        "--rounds-per-budget",
        type=int,
        default=_DEFAULT_ROUNDS_PER_BUDGET,
        help="每种预算独立重复轮数（默认 2）",
    )
    parser.add_argument(
        "--short-max-iter", type=int, default=_SHORT_MAX_EVALUATIONS,
        help="短预算每轮最多候选组数，范围 30--50（默认 50，含舍弃组）",
    )
    parser.add_argument(
        "--short-min-valid-evals", type=int, default=_SHORT_MIN_VALID_EVALUATIONS,
        help="短预算提前停止前至少需要的有效三拍组数（默认 30）",
    )
    parser.add_argument(
        "--long-max-iter", type=int, default=_LONG_MAX_EVALUATIONS,
        help="长预算每轮最多候选组数，范围 60--80（默认 80，含舍弃组）",
    )
    parser.add_argument(
        "--long-min-valid-evals", type=int, default=_LONG_MIN_VALID_EVALUATIONS,
        help="长预算提前停止前至少需要的有效三拍组数（默认 60）",
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
    parser.add_argument("--gun-config", type=Path, default=ROOT / "gun_modbus_config_bayes_90_100.json")
    parser.add_argument("--eps-v", type=float, default=1.0, help="Ug 稳定阈值 (V)")
    parser.add_argument("--samples", type=int, default=6, help="Ug 连续稳定次数")
    parser.add_argument("--poll-s", type=float, default=5.0, help="Ug 查询间隔 (s)")
    parser.add_argument("--pre-wait-s", type=float, default=5.0, help="set_params 后强制等待 (s)")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="四轮任务的共同标识；实际目录自动添加 short/long 和轮次",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=r"结果根目录，默认 D:\\RESULT（其下创建 captured_tif、out、reports）",
    )
    args = parser.parse_args()
    if not _SHORT_MIN_VALID_EVALUATIONS <= args.short_max_iter <= _SHORT_MAX_EVALUATIONS:
        parser.error("short-max-iter 必须在 30 到 50 之间")
    if not _SHORT_MIN_VALID_EVALUATIONS <= args.short_min_valid_evals <= args.short_max_iter:
        parser.error("short-min-valid-evals 必须在 30 与 short-max-iter 之间")
    if not _LONG_MIN_VALID_EVALUATIONS <= args.long_max_iter <= _LONG_MAX_EVALUATIONS:
        parser.error("long-max-iter 必须在 60 到 80 之间")
    if not _LONG_MIN_VALID_EVALUATIONS <= args.long_min_valid_evals <= args.long_max_iter:
        parser.error("long-min-valid-evals 必须在 60 与 long-max-iter 之间")
    if args.rounds_per_budget <= 0:
        parser.error("rounds-per-budget 必须为正整数")
    if args.plateau_patience <= 0:
        parser.error("plateau-patience 必须为正整数")
    if not 0 < args.min_relative_improvement < 1:
        parser.error("min-relative-improvement 必须在 0 和 1 之间")
    global RESULT_ROOT
    RESULT_ROOT = args.output_root.resolve()
    args.gun_config = args.gun_config.resolve()
    os.environ["GUN_MODBUS_CONFIG"] = str(args.gun_config)
    limits = _load_limits(args.gun_config)

    base_run_id = args.run_id or _run_id()
    round_results: list[tuple[str, int, str, dict]] = []
    failed_rounds: list[tuple[str, int]] = []

    budget_settings = {
        "short": (args.short_max_iter, args.short_min_valid_evals),
        "long": (args.long_max_iter, args.long_min_valid_evals),
    }
    selected_budgets = ("short", "long") if args.budget == "both" else (args.budget,)
    run_plan = [
        (budget_name, round_index, *budget_settings[budget_name])
        for budget_name in selected_budgets
        for round_index in range(1, args.rounds_per_budget + 1)
    ]

    try:
        for plan_index, (budget_name, round_index, max_iter, min_valid_evals) in enumerate(run_plan, 1):
            round_run_id = f"{base_run_id}_{budget_name}_round_{round_index:02d}"
            print("\n" + "#" * 72)
            print(
                f"[bayes-optlr-90100] 开始任务 {plan_index}/{len(run_plan)}："
                f"{budget_name} 第 {round_index}/{args.rounds_per_budget} 轮，"
                f"有效组至少 {min_valid_evals}，最多尝试 {max_iter} 组"
            )
            print(f"[bayes-optlr-90100] 本轮目录标识: {round_run_id}")
            print("#" * 72)

            opt = BayesianOptimizer(
                eps_v=args.eps_v,
                samples=args.samples,
                poll_s=args.poll_s,
                pre_wait_s=args.pre_wait_s,
                run_id=round_run_id,
            )
            reports_dir = RESULT_ROOT / "reports" / opt.run_id
            reports_dir.mkdir(parents=True, exist_ok=True)
            if args.history is None:
                history_path = reports_dir / f"bayes_optlr_history_{opt.run_id}.json"
            else:
                suffix = args.history.suffix or ".json"
                history_path = args.history.with_name(
                    f"{args.history.stem}_{budget_name}_round_{round_index:02d}{suffix}"
                )

            best = opt.optimize(
                limits,
                max_iterations=max_iter,
                n_initial_points=args.init_points,
                min_valid_evaluations=min_valid_evals,
                plateau_patience=args.plateau_patience,
                min_relative_improvement=args.min_relative_improvement,
            )
            opt.save_history(history_path)

            failed_path = reports_dir / f"bayes_optlr_failed_{opt.run_id}.json"
            opt.save_failed_records(failed_path)

            text_report_path = reports_dir / f"bayes_optlr_report_{opt.run_id}.md"
            opt.save_text_report(
                text_report_path,
                settings={
                    "shots_per_evaluation": _SHOTS_PER_EVALUATION,
                    "score_relative_tolerance": _SCORE_RELATIVE_TOLERANCE,
                    "budget": budget_name,
                    "budget_round": round_index,
                    "min_valid_evaluations": min_valid_evals,
                    "plateau_patience": args.plateau_patience,
                    "min_relative_improvement": args.min_relative_improvement,
                    "max_iterations": max_iter,
                    "group_rest_s": _GROUP_REST_S,
                },
            )

            try:
                from scripts.export_bayes_ranking import export_ranking

                ranking_path = reports_dir / f"bayes_optlr_ranking_{opt.run_id}.md"
                export_ranking(history_path, ranking_path)
            except Exception as exc:
                print(f"[bayes-optlr] 导出排名报告失败: {exc}", file=sys.stderr)

            if best is None:
                failed_rounds.append((budget_name, round_index))
                print(f"[bayes-optlr-90100] {budget_name} 第 {round_index} 轮结束，无成功结果")
                continue

            round_results.append((budget_name, round_index, round_run_id, best))
            print("\n" + "=" * 60)
            print(f"[bayes-optlr-90100] {budget_name} 第 {round_index}/{args.rounds_per_budget} 轮完成")
            print(f"  最优 score : {best['score']:.4f}")
            print(f"  最优 x1/x2 : {best['x1']:.4f} / {best['x2']:.4f}")
            p = best["params"]
            print(
                f"  最优参数   : Ua={p['anode_kv']}kV Ia={p['anode_ua']}uA "
                f"Uc={p['cathode_v']}V Ug={p['bias_v']}V If={p['fil_a']}A"
            )
            print(f"  图像       : {best.get('image_path')}")
            print(f"  debug      : {best.get('debug_path')}")
            print(f"  报告目录   : {reports_dir}")
            print("=" * 60)

        print("\n" + "#" * 72)
        print(f"[bayes-optlr-90100] 全部任务结束，成功 {len(round_results)}/{len(run_plan)} 轮")
        for budget_name, round_index, round_run_id, best in round_results:
            print(
                f"  {budget_name} 第 {round_index} 轮: score={best['score']:.4f}, "
                f"目录={round_run_id}"
            )
        if failed_rounds:
            print(f"  无成功结果的轮次: {failed_rounds}")
        print("#" * 72)
        return 1 if failed_rounds else 0
    except Exception as exc:
        print(f"[bayes-optlr] FATAL: {exc}", file=sys.stderr)
        return 1
    finally:
        print(f"[{_timestamp()}] modbus-off")
        _run(["modbus-off"], check=False)


if __name__ == "__main__":
    np.random.seed(42)
    raise SystemExit(main())
