#!/usr/bin/env python3
"""电子枪随机参数重复性测试。

默认从 ``gun_modbus_config.json`` 的五参范围随机抽取 40 组参数。每组完成
一次设参和稳定等待后，连续采集、分析 5 次，记录每一张图像的 x1、x2 和
score。原始 TIF 和调试 PNG 分别保存在既有的 ``captured_tif/<run_id>/`` 和
``out/<run_id>/`` 目录中。默认归档根目录为 ``D:\\RESULT``，使实验图片和
报告与代码项目分离；也可用 ``--output-root`` 修改。

示例::

    python steadytext.py
    python steadytext.py --groups 10 --shots 5 --pre-wait-s 120 --seed 42

终端逐次打印结果，并生成按参数组顺序整理的 JSON 和 Markdown 文字报告。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
os_encoding = "utf-8"
DEFAULT_GROUPS = 40
DEFAULT_SHOTS = 5
DEFAULT_OUTPUT_ROOT = Path(r"D:\RESULT")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """运行 app.py 子命令，并实时转发其输出。"""
    full_cmd = [sys.executable, "app.py", *cmd]
    print(f"[{_timestamp()}] RUN {' '.join(full_cmd)}")
    proc = subprocess.run(
        full_cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding=os_encoding,
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
    for line in output.splitlines():
        if line.strip().startswith("FBK"):
            match = re.search(
                r"Ua\s*=\s*([\d.]+)\s+Ia\s*=\s*([\d.]+)\s+"
                r"Uc\s*=\s*([\d.]+)\s+Ug\s*=\s*([\d.]+)\s+If\s*=\s*([\d.]+)",
                line,
            )
            if match:
                return {
                    "anode_kv": float(match.group(1)),
                    "anode_ua": float(match.group(2)),
                    "cathode_v": float(match.group(3)),
                    "bias_v": float(match.group(4)),
                    "fil_a": float(match.group(5)),
                }
    return None


def _parse_analyze(output: str) -> tuple[bool, float, float, float, str]:
    """解析 ``app.py analyze`` 的标准文字输出。"""
    ok = False
    score = x1 = x2 = float("nan")
    reason = ""
    for line in output.splitlines():
        text = line.strip()
        if text.startswith("ok"):
            ok = "True" in text
        elif text.startswith("x1 / x2"):
            match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", text)
            if match:
                x1, x2 = float(match.group(1)), float(match.group(2))
        elif text.startswith("score"):
            match = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
            if match:
                score = float(match.group(1))
        elif text.startswith("reason"):
            reason = text.split(":", 1)[1].strip() if ":" in text else ""
    if not (ok and math.isfinite(score) and math.isfinite(x1) and math.isfinite(x2)):
        ok = False
    return ok, score, x1, x2, reason


def _load_limits(path: Path) -> tuple[dict[str, list[float] | tuple[float, float] | float], float]:
    """读取与 bayes_opt.py 同一份参数范围，补充 bias_v。"""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    source = raw.get("param_limits", {})

    defaults: dict[str, tuple[float, float]] = {
        "anode_kv": (70.0, 110.0),
        "anode_ua": (30.0, 300.0),
        "cathode_v": (500.0, 750.0),
        "bias_v": (200.0, 200.0),
        "fil_a": (0.42, 0.46),
    }

    def normalize(key: str) -> list[float] | tuple[float, float] | float:
        value = source.get(key, defaults[key])
        if isinstance(value, bool):
            raise ValueError(f"param_limits.{key} 不能是布尔值")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list):
            if not value:
                raise ValueError(f"param_limits.{key} 不能为空列表")
            return [float(item) for item in value]
        if isinstance(value, dict):
            lo = float(value.get("min", defaults[key][0]))
            hi = float(value.get("max", defaults[key][1]))
            if lo > hi:
                raise ValueError(f"param_limits.{key}.min 不能大于 max")
            return (lo, hi)
        raise ValueError(f"不支持的 param_limits.{key}: {value!r}")

    exposure = source.get("exposure_constant_c", 0.0)
    return ({key: normalize(key) for key in defaults}, float(exposure))


def _sample_value(
    limit: list[float] | tuple[float, float] | float,
    rng: random.Random,
    *,
    digits: int,
) -> float:
    if isinstance(limit, list):
        return float(rng.choice(limit))
    if isinstance(limit, tuple):
        lo, hi = limit
        return round(rng.uniform(lo, hi), digits)
    return float(limit)


def _random_params(
    limits: dict[str, list[float] | tuple[float, float] | float],
    count: int,
    rng: random.Random,
) -> list[dict[str, float]]:
    """对连续范围做分层随机抽样，使每个变量在 40 组中尽量均匀覆盖。"""
    digits = {"anode_kv": 2, "anode_ua": 2, "cathode_v": 2, "bias_v": 2, "fil_a": 4}
    keys = tuple(digits)

    def stratified(limit: list[float] | tuple[float, float] | float, precision: int) -> list[float]:
        if isinstance(limit, tuple):
            lo, hi = limit
            values = [round(lo + (index + rng.random()) * (hi - lo) / count, precision) for index in range(count)]
            rng.shuffle(values)
            return values
        if isinstance(limit, list):
            # 枚举值按轮转方式尽量均衡使用，再随机打乱对应关系。
            values = [float(limit[index % len(limit)]) for index in range(count)]
            rng.shuffle(values)
            return values
        return [float(limit)] * count

    columns = {key: stratified(limits[key], digits[key]) for key in keys}
    return [{key: columns[key][index] for key in keys} for index in range(count)]


def _wait_ug_stable(
    *,
    eps_v: float,
    samples: int,
    poll_s: float,
    timeout_s: float,
) -> dict[str, float]:
    """沿用现有 repeatability_test.py 的 Ug 稳定判据。"""
    deadline = time.monotonic() + timeout_s
    stable_count = 0
    values: list[float] = []
    last: dict[str, float] | None = None
    while time.monotonic() < deadline:
        proc = _run(["modbus-read"])
        feedback = _parse_fbk(proc.stdout)
        if feedback is None:
            raise RuntimeError("无法从 modbus-read 输出解析 FBK")
        last = feedback
        values.append(feedback["bias_v"])
        if len(values) >= 4:
            moving_avg = sum(values[-3:]) / 3
            if abs(feedback["bias_v"] - moving_avg) <= eps_v:
                stable_count += 1
                if stable_count >= samples:
                    return feedback
            else:
                stable_count = 0
        time.sleep(poll_s)
    if last is None:
        raise RuntimeError("未读取到任何 FBK")
    raise RuntimeError(f"Ug 未在 {timeout_s:.0f}s 内稳定（最后值 {last['bias_v']:.2f}V）")


def _write_camera_exposure(exposure_us: int, *, source: str) -> int:
    """Write the legacy microsecond exposure estimate to the JSON config."""
    exposure = max(10, min(1000, int(round(exposure_us))))
    path = ROOT / "camera_cap_config.json"
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["exposure_time_us"] = exposure
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[steadytext] 曝光时间={exposure}us ({source})")
    return exposure


def _update_camera_exposure(params: dict[str, float], constant_c: float) -> int | None:
    """Use the legacy Ua^2 * Ia exposure estimate."""
    if constant_c <= 0:
        return None
    exposure = int(round(constant_c / (params["anode_kv"] ** 2 * params["anode_ua"])))
    return _write_camera_exposure(exposure, source=f"initial C={constant_c:.0f}")


def _read_camera_exposure() -> int:
    path = ROOT / "camera_cap_config.json"
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return max(10, int(config.get("exposure_time_us", 230)))


def _camera_exposure_override_enabled() -> bool:
    path = ROOT / "camera_cap_config.json"
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return bool(config.get("apply_exposure_time", False))


def _exposure_stats(tif_path: Path, saturation_level: int) -> dict[str, float]:
    """Return robust brightness diagnostics for a raw 14-bit frame."""
    # Keep steadytext importable on a control-only PC. The same analysis
    # dependencies are already required by `app.py analyze` on the experiment PC.
    import numpy as np
    from analysis.io_tif import load_tif14

    image = load_tif14(tif_path)
    return {
        "mean": float(np.mean(image)),
        "p99_9": float(np.percentile(image, 99.9)),
        "max": float(np.max(image)),
        "saturated_fraction": float(np.mean(image >= saturation_level)),
    }


def _exposure_is_acceptable(stats: dict[str, float], *, low_mean: float, high_mean: float) -> bool:
    """A clear image can contain intentional white regions, so use mean ADU, not p99.9."""
    return low_mean <= stats["mean"] <= high_mean


def _next_exposure_ms(current_ms: int, stats: dict[str, float], target_mean: float) -> int:
    """Scale the XView Exposure Time by measured mean ADU."""
    proposed = max(1, min(1000, current_ms * target_mean / max(stats["mean"], 1.0)))
    next_ms = max(1, min(1000, int(round(proposed))))
    if stats["mean"] > target_mean and next_ms >= current_ms:
        next_ms = max(1, current_ms - 1)
    elif stats["mean"] < target_mean and next_ms <= current_ms:
        next_ms = min(1000, current_ms + 1)
    return next_ms


def _discard_temporary_capture() -> None:
    """Delete a rejected frame so only scored, valid images reach D:/RESULT."""
    for path in (ROOT / "captured_tif" / "single.tif", ROOT / "out" / "single_debug.png"):
        if path.is_file():
            path.unlink()


def _move_output(group: int, shot: int, record: dict[str, Any], tif_dir: Path, debug_dir: Path) -> None:
    src_tif = ROOT / "captured_tif" / "single.tif"
    src_debug = ROOT / "out" / "single_debug.png"
    stem = f"group_{group:03d}_shot_{shot:03d}"
    if src_tif.is_file():
        dst_tif = tif_dir / f"{stem}.tif"
        shutil.move(str(src_tif), str(dst_tif))
        record["image_path"] = str(dst_tif)
    if src_debug.is_file():
        dst_debug = debug_dir / f"{stem}_debug.png"
        shutil.move(str(src_debug), str(dst_debug))
        record["debug_path"] = str(dst_debug)


def _take_shot_plain(
    group: int,
    shot: int,
    shots_per_group: int,
    params: dict[str, float],
    tif_dir: Path,
    debug_dir: Path,
) -> dict[str, Any]:
    """Original one-capture/one-analysis steadytext behaviour, without grey control."""
    record: dict[str, Any] = {
        "iteration": (group - 1) * shots_per_group + shot,
        "group": group,
        "shot": shot,
        "params": params.copy(),
        "score": None,
        "x1": None,
        "x2": None,
        "success": False,
        "reason": "",
        "image_path": None,
        "debug_path": None,
    }
    src_tif = ROOT / "captured_tif" / "single.tif"
    src_debug = ROOT / "out" / "single_debug.png"
    for path in (src_tif, src_debug):
        if path.is_file():
            path.unlink()
    try:
        _run(["capture"])
        if not src_tif.is_file():
            record["reason"] = "capture 失败，无 TIF 生成"
        else:
            analyze = _run(["analyze"], check=False)
            if analyze.returncode != 0:
                record["reason"] = "analyze 命令失败"
            else:
                ok, score, x1, x2, reason = _parse_analyze(analyze.stdout)
                record.update({"success": ok, "score": score if ok else None, "x1": x1 if ok else None, "x2": x2 if ok else None, "reason": reason})
                if not ok and not record["reason"]:
                    record["reason"] = "分析未得到有效 x1/x2/score"
    except Exception as exc:
        record["reason"] = f"异常: {exc}"
    finally:
        _move_output(group, shot, record, tif_dir, debug_dir)
    return record


def _take_shot(
    group: int,
    shot: int,
    shots_per_group: int,
    params: dict[str, float],
    tif_dir: Path,
    debug_dir: Path,
    *,
    auto_exposure: bool,
    exposure_target_mean: float,
    exposure_low_mean: float,
    exposure_high_mean: float,
    exposure_max_retries: int,
    exposure_reference_mean: float | None,
    exposure_reference_tolerance: float,
) -> dict[str, Any]:
    """Capture one scored frame, correcting excessive/insufficient exposure first."""
    if not auto_exposure:
        return _take_shot_plain(group, shot, shots_per_group, params, tif_dir, debug_dir)
    record: dict[str, Any] = {
        "iteration": (group - 1) * shots_per_group + shot,
        "group": group,
        "shot": shot,
        "params": params.copy(),
        "score": None,
        "x1": None,
        "x2": None,
        "success": False,
        "reason": "",
        "image_path": None,
        "debug_path": None,
        "exposure_ms": _read_camera_exposure(),
        "exposure_stats": None,
        "exposure_reference_mean": exposure_reference_mean,
    }
    src_tif = ROOT / "captured_tif" / "single.tif"
    src_debug = ROOT / "out" / "single_debug.png"
    for path in (src_tif, src_debug):
        if path.is_file():
            path.unlink()
    try:
        for exposure_try in range(exposure_max_retries + 1):
            _run(["capture"])
            if not src_tif.is_file():
                record["reason"] = "capture 失败，无 TIF 生成"
                break

            record["exposure_ms"] = _read_camera_exposure()
            if not auto_exposure:
                break

            stats = _exposure_stats(src_tif, 16383)
            record["exposure_stats"] = stats
            target_mean = exposure_reference_mean or exposure_target_mean
            if exposure_reference_mean is None:
                low_mean, high_mean = exposure_low_mean, exposure_high_mean
                target_label = "initial target"
            else:
                low_mean = target_mean * (1.0 - exposure_reference_tolerance)
                high_mean = target_mean * (1.0 + exposure_reference_tolerance)
                target_label = "group reference"
            acceptable = _exposure_is_acceptable(stats, low_mean=low_mean, high_mean=high_mean)
            print(
                "[steadytext] exposure check: "
                f"{record['exposure_ms']}ms, mean={stats['mean']:.0f}, p99.9={stats['p99_9']:.0f}, "
                f"max={stats['max']:.0f}, saturated={stats['saturated_fraction']:.3%}, "
                f"{target_label} mean={target_mean:.0f}"
            )
            if acceptable:
                break

            _discard_temporary_capture()
            if exposure_try >= exposure_max_retries:
                record["reason"] = (
                    f"自动曝光 {exposure_max_retries + 1} 次仍未达到目标平均灰度: "
                    f"mean={stats['mean']:.0f}, target={target_mean:.0f}"
                )
                break

            next_ms = _next_exposure_ms(record["exposure_ms"], stats, target_mean)
            if next_ms == record["exposure_ms"]:
                record["reason"] = f"自动曝光无法继续调整（{next_ms}ms）"
                break
            _write_camera_exposure(next_ms, source="image-feedback")
            print(f"[steadytext] 图像亮度不合格，重拍 {exposure_try + 1}/{exposure_max_retries}: {next_ms}ms")

        if not record["reason"] and src_tif.is_file():
            analyze = _run(["analyze"], check=False)
            if analyze.returncode != 0:
                record["reason"] = "analyze 命令失败"
            else:
                ok, score, x1, x2, reason = _parse_analyze(analyze.stdout)
                record.update({"success": ok, "score": score if ok else None, "x1": x1 if ok else None, "x2": x2 if ok else None, "reason": reason})
                if not ok and not record["reason"]:
                    record["reason"] = "分析未得到有效 x1/x2/score"
    except Exception as exc:
        record["reason"] = f"异常: {exc}"
    finally:
        if record["success"]:
            _move_output(group, shot, record, tif_dir, debug_dir)
        else:
            _discard_temporary_capture()
    return record


def _finite_values(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in records if r.get("success") and r.get(key) is not None and math.isfinite(float(r[key]))]


def _stats(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = {key: _finite_values(records, key) for key in ("score", "x1", "x2")}
    stats: dict[str, float | int | None] = {"successful_shots": len(values["score"]), "total_shots": len(records)}
    for key, series in values.items():
        stats[f"{key}_mean"] = float(statistics.mean(series)) if series else None
        stats[f"{key}_std"] = float(statistics.stdev(series)) if len(series) >= 2 else (0.0 if series else None)
    return stats


def _fmt(value: Any, digits: int = 4) -> str:
    return "-" if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _params_text(params: dict[str, float]) -> str:
    return (f"Ua={params['anode_kv']:.2f} kV, Ia={params['anode_ua']:.2f} uA, "
            f"Uc={params['cathode_v']:.2f} V, Ug={params['bias_v']:.2f} V, If={params['fil_a']:.4f} A")


def _write_report(path: Path, run_id: str, groups: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    successful = [record for record in results if record["success"]]
    lines = [
        "# 随机参数连续测量报告",
        "",
        f"- 运行编号: `{run_id}`",
        f"- 参数组数: {len(groups)}",
        f"- 每组连续测量: {groups[0]['stats']['total_shots'] if groups else 0} 次",
        f"- 成功测量: {len(successful)}/{len(results)}",
        "",
    ]
    for group in groups:
        stat = group["stats"]
        lines.extend([
            f"## 第 {group['group']:02d} 组",
            "",
            f"- 设定参数: {_params_text(group['params'])}",
            f"- 稳定后反馈: {_params_text(group['feedback']) if group.get('feedback') else '-'}",
            f"- 曝光时间: {group.get('exposure_us') if group.get('exposure_us') is not None else '-'} us",
            f"- 汇总: score={_fmt(stat['score_mean'])} +/- {_fmt(stat['score_std'])}，"
            f"x1={_fmt(stat['x1_mean'])} +/- {_fmt(stat['x1_std'])}，x2={_fmt(stat['x2_mean'])} +/- {_fmt(stat['x2_std'])}",
            "",
            "| 次数 | 成功 | score | x1 | x2 | 原始图像 | 输出图像 | 原因 |",
            "| ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
        ])
        for record in group["shots"]:
            lines.append(
                f"| {record['shot']} | {'是' if record['success'] else '否'} | {_fmt(record['score'])} | "
                f"{_fmt(record['x1'])} | {_fmt(record['x2'])} | `{record.get('image_path') or '-'}` | "
                f"`{record.get('debug_path') or '-'}` | {record.get('reason') or '-'} |"
            )
        lines.append("")
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="随机 40 组五参、每组连续测量 5 次的稳定性测试")
    parser.add_argument("--groups", type=int, default=DEFAULT_GROUPS, help="随机参数组数，默认 40")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="每组连续测量次数，默认 5")
    parser.add_argument("--seed", type=int, default=None, help="随机种子；不提供则每次随机")
    parser.add_argument("--gun-config", type=Path, default=ROOT / "gun_modbus_config_three.json")
    parser.add_argument("--pre-wait-s", type=float, default=5.0, help="每组设参后的强制等待秒数，默认 5")
    parser.add_argument("--eps-v", type=float, default=2.0, help="Ug 稳定阈值 (V)，默认 2")
    parser.add_argument("--samples", type=int, default=6, help="Ug 连续稳定次数，默认 6")
    parser.add_argument("--poll-s", type=float, default=5.0, help="Ug 查询间隔秒数，默认 5")
    parser.add_argument("--timeout-s", type=float, default=180.0, help="单组 Ug 稳定超时秒数，默认 180")
    parser.add_argument("--interval-s", type=float, default=0.0, help="同组两次测量之间的等待秒数，默认 0")
    parser.add_argument("--run-id", type=str, default=None, help="输出目录编号，默认当前时间")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=r"实验结果根目录，默认 D:\RESULT",
    )
    parser.add_argument("--history", type=Path, default=None, help="历史 JSON 路径；默认写入输出根目录")
    parser.add_argument("--no-auto-exposure", action="store_true", help="不按 Ua^2 * Ia 自动更新曝光时间")
    parser.add_argument("--exposure-target-mean", type=float, default=2000.0, help="自动曝光目标平均灰度，默认 2000（对应已验证的 XView 图像）")
    parser.add_argument("--exposure-low-mean", type=float, default=1200.0, help="首张可接受平均灰度下限，默认 1200")
    parser.add_argument("--exposure-high-mean", type=float, default=3200.0, help="首张可接受平均灰度上限，默认 3200")
    parser.add_argument("--exposure-max-retries", type=int, default=8, help="每张正式图最多自动重拍次数，默认 8")
    parser.add_argument("--exposure-reference-tolerance", type=float, default=0.03, help="后续图片相对首张有效图灰度的允许偏差，默认 0.03（3%）")
    args = parser.parse_args(argv)
    if args.groups <= 0 or args.shots <= 0 or args.samples <= 0:
        parser.error("--groups、--shots、--samples 必须为正整数")
    if min(args.pre_wait_s, args.poll_s, args.timeout_s, args.interval_s) < 0:
        parser.error("等待时间不能为负数")
    if not (0 < args.exposure_low_mean <= args.exposure_target_mean <= args.exposure_high_mean <= 16383):
        parser.error("曝光平均灰度阈值须满足 0 < low <= target <= high <= 16383")
    if args.exposure_max_retries < 0 or args.exposure_reference_tolerance < 0:
        parser.error("重拍次数和灰度容差不能为负数")

    args.gun_config = args.gun_config.resolve()
    # 让 app.py 的 Modbus 参数校验使用同一份三变量范围配置。
    os.environ["GUN_MODBUS_CONFIG"] = str(args.gun_config)
    limits, exposure_constant = _load_limits(args.gun_config)
    # Restore the pre-camera-change behaviour: steadytext does not inspect or
    # correct grey level and the capture layer leaves camera exposure untouched.
    automatic_exposure_enabled = False
    rng = random.Random(args.seed)
    params_list = _random_params(limits, args.groups, rng)
    run_id = args.run_id or _run_id()
    output_root = args.output_root.resolve()
    tif_dir = output_root / "captured_tif" / run_id
    debug_dir = output_root / "out" / run_id
    reports_dir = output_root / "reports" / run_id
    tif_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.history or reports_dir / f"steadytext_history_{run_id}.json"
    report_path = reports_dir / f"steadytext_report_{run_id}.md"

    print(f"[steadytext] 开始：随机 {args.groups} 组参数，每组连续测量 {args.shots} 次")
    print(f"[steadytext] 实验结果根目录: {output_root}")
    print(f"[steadytext] 原始图像目录: {tif_dir}")
    print(f"[steadytext] 输出图像目录: {debug_dir}")
    print(f"[steadytext] 报告目录: {reports_dir}")
    groups: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    code = 0
    try:
        _run(["modbus-on"])
        for group_number, params in enumerate(params_list, 1):
            print(f"\n[steadytext] ===== 第 {group_number:02d}/{args.groups:02d} 组 =====")
            print(f"[steadytext] 设定参数: {_params_text(params)}")
            group: dict[str, Any] = {
                "group": group_number, "params": params, "feedback": None,
                "exposure_us": None, "shots": [],
            }
            try:
                if not args.no_auto_exposure:
                    group["exposure_us"] = _update_camera_exposure(params, exposure_constant)
                _run(["modbus-set", "--Ua", str(params["anode_kv"]), "--Ia", str(params["anode_ua"]), "--Uc", str(params["cathode_v"]), "--Ug", str(params["bias_v"]), "--If", str(params["fil_a"])])
                if args.pre_wait_s:
                    print(f"[{_timestamp()}] 等待 {args.pre_wait_s:g}s ...")
                    time.sleep(args.pre_wait_s)
                print(f"[{_timestamp()}] 等待 Ug 稳定 ...")
                group["feedback"] = _wait_ug_stable(eps_v=args.eps_v, samples=args.samples, poll_s=args.poll_s, timeout_s=args.timeout_s)
                print(f"[steadytext] 稳定后反馈: {_params_text(group['feedback'])}")
                for shot in range(1, args.shots + 1):
                    print(f"[steadytext] group={group_number:03d} shot={shot:03d}: capture + analyze")
                    record = _take_shot(
                        group_number, shot, args.shots, params, tif_dir, debug_dir,
                        auto_exposure=automatic_exposure_enabled,
                        exposure_target_mean=args.exposure_target_mean,
                        exposure_low_mean=args.exposure_low_mean,
                        exposure_high_mean=args.exposure_high_mean,
                        exposure_max_retries=args.exposure_max_retries,
                        exposure_reference_mean=None,
                        exposure_reference_tolerance=args.exposure_reference_tolerance,
                    )
                    record["feedback"] = group["feedback"].copy()
                    group["shots"].append(record)
                    results.append(record)
                    print(f"[steadytext] RESULT group={group_number:03d} shot={shot:03d}: ok={record['success']} score={_fmt(record['score'])} x1={_fmt(record['x1'])} x2={_fmt(record['x2'])}")
                    if shot < args.shots and args.interval_s:
                        time.sleep(args.interval_s)
            except Exception as exc:
                code = 1
                print(f"[steadytext] 第 {group_number:03d} 组失败: {exc}", file=sys.stderr)
                while len(group["shots"]) < args.shots:
                    shot = len(group["shots"]) + 1
                    record = {"iteration": (group_number - 1) * args.shots + shot, "group": group_number, "shot": shot, "params": params.copy(), "feedback": group["feedback"], "score": None, "x1": None, "x2": None, "success": False, "reason": f"参数组失败: {exc}", "image_path": None, "debug_path": None}
                    group["shots"].append(record)
                    results.append(record)
            group["stats"] = _stats(group["shots"])
            groups.append(group)
            print(f"[steadytext] 第 {group_number:03d} 组完成: score_mean={_fmt(group['stats']['score_mean'])}, 有效 {group['stats']['successful_shots']}/{args.shots}")
    except Exception as exc:
        code = 1
        print(f"[steadytext] FATAL: {exc}", file=sys.stderr)
    finally:
        print(f"[{_timestamp()}] modbus-off")
        _run(["modbus-off"], check=False)

    summary = {
        "run_id": run_id,
        "parameter_groups": args.groups,
        "shots_per_group": args.shots,
        "successful_shots": sum(1 for r in results if r["success"]),
        "total_shots": len(results),
    }
    payload = {"summary": summary, "groups": groups, "results": results}
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_report(report_path, run_id, groups, results)
    print(f"[steadytext] 历史已保存: {history_path}")
    print(f"[steadytext] 文字报告已保存: {report_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
