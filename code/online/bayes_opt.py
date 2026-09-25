#!/usr/bin/env python3
"""电子枪贝叶斯闭环优化器（单文件，仅通过 app.py 子命令驱动）。

真机上只要 ``app.py`` 能跑，本文件就能跑；不依赖 camera_cap/gun_modbus/analysis 内部 API。

用法::

    python bayes_opt.py --max-iter 30 --init-points 5
    python bayes_opt.py --max-iter 50 --target-score 3.0

每轮流程::

    app.py modbus-set --Ua ... --Ia ... --Uc ... --Ug 200 --If ...
    等待 120s
    循环 app.py modbus-read，直到 Ug 自身连续稳定
    app.py capture
    app.py analyze
    重命名 captured_tif/single.tif / out/single_debug.png 为 iter_XXX

失败时返回 ``nan``，优化器会丢弃该组参数并重新采样。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt.space import Categorical, Real

# 屏蔽 skopt "objective has been evaluated at point ... before" 重复点警告
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="the objective has been evaluated",
)

ROOT = Path(__file__).resolve().parent
DEFAULT_BIAS_V: float = 200.0
_POWER_RECOVER_WAIT_S: float = 5.0
_STABILITY_WINDOW_SIZE: int = 5  # 滑动平均窗口大小（个 Ia 采样点）
_STABILITY_ABS_TOL: float = 0.2  # 相邻滑动平均绝对差阈值（uA）
_EXPOSURE_MIN_US: int = 10       # 曝光时间下限（us），防异常值
_EXPOSURE_MAX_US: int = 1000    # 曝光时间上限（us），防异常值

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


def _wait_beam_stable(
    target_ia: float,
    *,
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

    ia_window: list[float] = []
    avg_history: list[float] = []
    fbk: dict[str, float] | None = None
    ug = 0.0

    while True:
        proc = _run(["modbus-read"])
        fbk = _parse_fbk(proc.stdout)
        if fbk is None:
            raise RuntimeError("无法从 modbus-read 解析 FBK")
        ua = fbk["Ua"]
        ia = fbk["Ia"]
        ug = fbk["Ug"]
        if abs(ia-target_ia)/target_ia <= 0.02 and 1.0 < ug < 199:
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
        self.tif_dir = ROOT / "captured_tif" / self.run_id
        self.debug_dir = ROOT / "out" / self.run_id
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
            "timestamp": _timestamp(),
        })
        print(f"[bayes-opt] 记录失败 iter {record['iteration']}: {reason}")

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

    def _save_iter_files(self, iteration: int, record: dict) -> None:
        """把 single.tif / single_debug.png 移到 iter_XXX 命名，成功失败都保存。"""
        src_tif = ROOT / "captured_tif" / "single.tif"
        src_debug = ROOT / "out" / "single_debug.png"
        stem = f"iter_{iteration:03d}"
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
        print(f"\n[bayes-opt] iter {iteration} 开始")
        print(
            f"[bayes-opt] 提议参数: Ua={x[0]:.2f}kV Ia={x[1]:.2f}uA "
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
                    samples=self.samples,
                    poll_s=self.poll_s,
                )
                _update_feedback_from_fbk(fbk)
                print(
                    f"[{_timestamp()}] FBK Ua={fbk['Ua']:.2f} Ia={fbk['Ia']:.2f} "
                    f"Uc={fbk['Uc']:.2f} Ug={fbk['Ug']:.2f} If={fbk['If']:.3f}"
                )

                # 4) 采图（单次，不重试）
                src_tif = ROOT / "captured_tif" / "single.tif"
                if src_tif.is_file():
                    src_tif.unlink()
                try:
                    _run(["capture"])
                except subprocess.CalledProcessError:
                    pass
                if not src_tif.is_file():
                    reason = "capture 失败，无 tif 生成"
                else:
                    # 5) 分析（单次，不重试）
                    try:
                        ana_proc = _run(["analyze"])
                    except subprocess.CalledProcessError:
                        ana_proc = None

                    if ana_proc is None:
                        reason = "analyze 失败"
                    else:
                        ok, score, x1, x2, reason = _parse_analyze(ana_proc.stdout)
                        if not (ok and np.isfinite(x1) and np.isfinite(x2)):
                            ok = False

        except Exception as exc:
            self._save_iter_files(iteration, record)
            self._record_failure(record, f"异常: {exc}")
            print(f"[bayes-opt] 异常: {exc}", file=sys.stderr)
            return float("nan")

        if not ok:
            self._save_iter_files(iteration, record)
            self._record_failure(record, reason)
            print(f"[bayes-opt] 尝试失败: {reason}")
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
            f"[bayes-opt] iter {iteration} 完成: score={score:.4f} "
            f"x1={x1:.4f} x2={x2:.4f}"
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
        max_iterations: int = 30,
        n_initial_points: int = 5,
    ) -> dict | None:
        """执行网格贝叶斯优化：固定 Ua/Ia/If/Ug，只优化 Uc。

        ``limits`` 须包含：
        - ``anode_kv``: Ua 枚举列表
        - ``anode_ua``: Ia 枚举列表
        - ``cathode_v``: Uc 搜索范围
        - ``fil_a``: 固定 If 单值

        ``max_iterations`` 指每个 (Ua, Ia) 组合的**总尝试次数**（含失败）。
        失败的评估会被记录，但不会 ``tell`` 给优化器，因此不影响 GP。
        """
        ua_values = [float(v) for v in limits["anode_kv"]]
        ia_values = [float(v) for v in limits["anode_ua"]]
        fixed_if = float(limits["fil_a"])
        exposure_c = float(limits.get("exposure_constant_c", 0.0))

        print(f"\n[bayes-opt] 开始网格优化: Ua={ua_values} Ia={ia_values} If={fixed_if}")
        print(f"[bayes-opt] Ua/Ia 组合数: {len(ua_values) * len(ia_values)}")
        print(f"[bayes-opt] 每组合最多尝试 Uc {max_iterations} 次，其中初始随机 {n_initial_points} 次")
        print(f"[bayes-opt] Uc 搜索空间: {limits['cathode_v']}")
        if exposure_c > 0:
            print(f"[bayes-opt] 曝光常数 C={exposure_c:.0f}，每组合自动更新曝光时间")
        else:
            print("[bayes-opt] 曝光常数 C 未配置，使用 camera_cap_config.json 中的固定曝光")

        uc_search_space = [Real(*limits["cathode_v"], name="cathode_v")]

        for ua in ua_values:
            for ia in ia_values:
                print(f"\n[bayes-opt] ===== Ua={ua:.0f}kV Ia={ia:.0f}uA =====")

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
                    acq_func_kwargs={"xi": 0.01},
                    acq_optimizer="sampling",
                    acq_optimizer_kwargs={"n_points": 10000},
                    n_initial_points=min(n_initial_points, max_iterations),
                    random_state=42,
                    initial_point_generator="lhs",
                )

                successful_evals = 0
                attempts = 0
                need_random = False  # 上次失败后是否需要随机点
                while attempts < max_iterations:
                    if need_random:
                        x = opt.space.rvs(random_state=opt.rng)[0]
                    elif opt._n_initial_points <= 0 and not opt.models:
                        x = opt.space.rvs(random_state=opt.rng)[0]
                    else:
                        x = opt.ask()
                    y = self._evaluate_fixed_ua_ia_if(
                        x, ua=ua, ia=ia, if_value=fixed_if
                    )
                    attempts += 1
                    if np.isfinite(y):
                        opt.tell(x, y)
                        successful_evals += 1
                        need_random = False
                    else:
                        # 失败：不记录到 GP（零影响）
                        if opt._n_initial_points > 0:
                            # 初始阶段：手动推进计数器，下次 ask 返回新初始点
                            opt._n_initial_points -= 1
                            opt.cache_ = {}
                            need_random = False
                        else:
                            # 模型阶段：_next_x 无法更新，下次用随机点
                            need_random = True

                print(
                    f"[bayes-opt] Ua={ua} Ia={ia} 完成: "
                    f"{successful_evals}/{attempts} 次有效评估"
                )

        successful = [r for r in self._results if r["success"] and np.isfinite(r["score"])]
        if not successful:
            print("[bayes-opt] 没有成功的迭代")
            return None
        return min(successful, key=lambda r: r["score"])

    def save_failed_records(self, path: Path) -> None:
        if not self._failed_records:
            return
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._failed_records, f, indent=2, ensure_ascii=False)
        print(f"[bayes-opt] 失败记录已保存: {path}")

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
        print(f"[bayes-opt] 历史已保存: {path}")


def _compute_exposure_us(ua_kv: float, ia_ua: float, constant_c: float) -> int:
    """根据曝光常数 C = Ua^2 * Ia * exposure_time 计算 exposure_time (us)。"""
    raw = int(round(constant_c / (ua_kv * ua_kv * ia_ua)))
    if raw < _EXPOSURE_MIN_US:
        print(
            f"[bayes-opt] WARN 曝光 {raw}us < {_EXPOSURE_MIN_US}us，裁剪到下限 "
            f"(Ua={ua_kv} Ia={ia_ua} C={constant_c})",
            file=sys.stderr,
        )
        return _EXPOSURE_MIN_US
    if raw > _EXPOSURE_MAX_US:
        print(
            f"[bayes-opt] WARN 曝光 {raw}us > {_EXPOSURE_MAX_US}us，裁剪到上限 "
            f"(Ua={ua_kv} Ia={ia_ua} C={constant_c})",
            file=sys.stderr,
        )
        return _EXPOSURE_MAX_US
    return raw


def _update_camera_exposure(ua_kv: float, ia_ua: float, constant_c: float) -> int:
    """根据 (Ua, Ia) 计算曝光时间并写回 camera_cap_config.json。返回写入的值。"""
    if constant_c <= 0:
        print("[bayes-opt] exposure_constant_c 未配置，跳过曝光更新")
        return 0
    exposure_us = _compute_exposure_us(ua_kv, ia_ua, constant_c)
    config_path = ROOT / "camera_cap_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cam_cfg = json.load(f)
    cam_cfg["exposure_time_us"] = exposure_us
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cam_cfg, f, indent=2, ensure_ascii=False)
    print(
        f"[bayes-opt] Ua={ua_kv:.0f}kV Ia={ia_ua:.0f}uA "
        f"曝光时间={exposure_us}us (C={constant_c:.0f})"
    )
    return exposure_us


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
    parser.add_argument("--max-iter", type=int, default=30, help="最大迭代次数")
    parser.add_argument("--init-points", type=int, default=5, help="初始随机采样点数")
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
    parser.add_argument("--gun-config", type=Path, default=ROOT / "gun_modbus_config.json")
    parser.add_argument("--eps-v", type=float, default=2.0, help="Ug 稳定阈值 (V)")
    parser.add_argument("--samples", type=int, default=6, help="Ug 连续稳定次数")
    parser.add_argument("--poll-s", type=float, default=5.0, help="Ug 查询间隔 (s)")
    parser.add_argument("--pre-wait-s", type=float, default=5.0, help="set_params 后强制等待 (s)")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="运行标识，用于创建独立输出目录（默认当前时间 YYYYMMDD_HHMMSS）",
    )
    args = parser.parse_args()

    limits = _load_limits(args.gun_config)

    try:
        print(f"[{_timestamp()}] modbus-on")
        _run(["modbus-on"])

        opt = BayesianOptimizer(
            eps_v=args.eps_v,
            samples=args.samples,
            poll_s=args.poll_s,
            pre_wait_s=args.pre_wait_s,
            run_id=args.run_id,
        )
        if args.history is None:
            args.history = ROOT / f"bayes_opt_history_{opt.run_id}.json"
        best = opt.optimize(
            limits,
            max_iterations=args.max_iter,
            n_initial_points=args.init_points,
        )
        opt.save_history(args.history)

        failed_path = ROOT / f"bayes_opt_failed_{opt.run_id}.json"
        opt.save_failed_records(failed_path)

        # 导出按 score 排序的 Markdown 报告
        try:
            from scripts.export_bayes_ranking import export_ranking

            ranking_path = ROOT / f"bayes_opt_ranking_{opt.run_id}.md"
            export_ranking(args.history, ranking_path)
        except Exception as exc:
            print(f"[bayes-opt] 导出排名报告失败: {exc}", file=sys.stderr)

        if best is None:
            print("[bayes-opt] 优化结束，无成功结果")
            return 1

        print("\n" + "=" * 60)
        print("[bayes-opt] 优化完成")
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
        print(f"[bayes-opt] FATAL: {exc}", file=sys.stderr)
        return 1
    finally:
        print(f"[{_timestamp()}] modbus-off")
        _run(["modbus-off"], check=False)


if __name__ == "__main__":
    np.random.seed(42)
    raise SystemExit(main())
