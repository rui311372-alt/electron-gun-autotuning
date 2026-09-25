#!/usr/bin/env python3
"""固定 Ua=100、Ia=80，对10个均匀随机Uc分别连续拍摄5张。

每组连续拍5张后关闭设备；下一组开始前保持断电5分钟，再重新开启、设参并等待Ia/Ug稳定，
逐张记录反馈、灰度、score、x1和x2，结果归档到 D:\RESULT。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = Path(r"D:\RESULT")
PARAMS = {"Ua": 100.0, "Ia": 80.0, "Ug": 200.0, "If": 0.45}
DEFAULT_GUN_CONFIG = ROOT / "gun_modbus_config_bayes_100_80.json"
DEFAULT_UC_MIN = 640
DEFAULT_UC_MAX = 750
DEFAULT_GROUPS = 10
DEFAULT_SHOTS = 5
DEFAULT_GROUP_REST_S = 300.0
DEFAULT_CAPTURE_PRE_WAIT_S = 5.0


def _run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    full = [sys.executable, "app.py", *cmd]
    print(f"[{_timestamp()}] RUN {' '.join(full)}")
    proc = subprocess.run(full, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, full, output=proc.stdout, stderr=proc.stderr)
    return proc


def _parse_analyze(output: str) -> tuple[bool, float | None, float | None, float | None, str]:
    ok, score, x1, x2, reason = False, None, None, None, ""
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("ok"):
            ok = "True" in line
        elif line.startswith("x1 / x2"):
            import re
            match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", line)
            if match:
                x1, x2 = float(match.group(1)), float(match.group(2))
        elif line.startswith("score"):
            import re
            match = re.search(r"([-+]?\d+(?:\.\d+)?)", line)
            if match:
                score = float(match.group(1))
        elif line.startswith("reason"):
            reason = line.split(":", 1)[1].strip() if ":" in line else ""
    valid = ok and all(value is not None and math.isfinite(value) for value in (score, x1, x2))
    return valid, score if valid else None, x1 if valid else None, x2 if valid else None, reason


def _gray_stats(tif_path: Path) -> dict[str, float]:
    import numpy as np
    from analysis.io_tif import load_tif14

    image = load_tif14(tif_path)
    return {
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "p50": float(np.percentile(image, 50)),
        "p90": float(np.percentile(image, 90)),
        "p99": float(np.percentile(image, 99)),
        # 14-bit TIFF 的满量程为 16383。这个值只记录，不参与曝光调节。
        "saturated_pct": float(np.mean(image >= 16383) * 100.0),
    }


def _parse_fbk(output: str) -> dict[str, float] | None:
    import re
    for line in output.splitlines():
        if line.strip().startswith("FBK"):
            match = re.search(r"Ua\s*=\s*([\d.]+)\s+Ia\s*=\s*([\d.]+)\s+Uc\s*=\s*([\d.]+)\s+Ug\s*=\s*([\d.]+)\s+If\s*=\s*([\d.]+)", line)
            if match:
                return dict(zip(("Ua", "Ia", "Uc", "Ug", "If"), (float(match.group(i)) for i in range(1, 6))))
    return None


def _wait_ia_ug_stable(
    target_ia: float,
    *,
    samples: int,
    poll_s: float,
    ia_target_tolerance: float,
    ia_span_tolerance: float,
    ug_span_tolerance: float,
    timeout_s: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Wait for a stable beam state and return the final feedback plus evidence.

    A capture is permitted only when the latest ``samples`` readings satisfy all
    of these conditions: mean Ia is within the target tolerance, Ia itself has
    a bounded range, and Ug has a bounded range.  This prevents the test from
    treating a single coincidentally-good feedback reading as stable.
    """
    if target_ia <= 0 or samples <= 0 or poll_s < 0 or timeout_s <= 0:
        raise ValueError("Ia/Ug 稳定性参数无效")
    ia_values: list[float] = []
    ug_values: list[float] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proc = _run(["modbus-read"], check=False)
        fbk = _parse_fbk(proc.stdout) if proc.returncode == 0 else None
        if fbk is not None:
            ia_values.append(float(fbk["Ia"]))
            ug_values.append(float(fbk["Ug"]))
            if len(ia_values) > samples:
                ia_values.pop(0)
                ug_values.pop(0)
            if len(ia_values) == samples:
                ia_mean = statistics.mean(ia_values)
                ia_span = max(ia_values) - min(ia_values)
                ug_span = max(ug_values) - min(ug_values)
                ia_on_target = abs(ia_mean - target_ia) / target_ia <= ia_target_tolerance
                ug_ready = 1.0 < fbk["Ug"] < 199.0
                if ia_on_target and ia_span <= ia_span_tolerance and ug_span <= ug_span_tolerance and ug_ready:
                    evidence = {
                        "samples": float(samples),
                        "ia_mean": float(ia_mean),
                        "ia_span": float(ia_span),
                        "ug_span": float(ug_span),
                    }
                    print(
                        f"[text-10] Ia/Ug 稳定: Ia均值={ia_mean:.3f}uA, Ia范围={ia_span:.3f}uA, "
                        f"Ug范围={ug_span:.3f}V（连续 {samples} 次）"
                    )
                    return fbk, evidence
        time.sleep(poll_s)
    raise TimeoutError(
        f"等待 Ia/Ug 稳定超时 {timeout_s:.0f}s（Ia目标={target_ia:.3f}uA；"
        f"要求 Ia均值±{ia_target_tolerance:.1%}、Ia范围≤{ia_span_tolerance:g}uA、"
        f"Ug范围≤{ug_span_tolerance:g}V）"
    )


def _write_camera_exposure(config_path: Path, original: dict[str, Any], exposure_ms: int) -> None:
    config = original.copy()
    # 字段名为历史遗留；当前相机SDK的整数接口按ms解释，与100/80贝叶斯一致。
    config["exposure_time_us"] = exposure_ms
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _exposure_from_gun_config(config_path: Path, ua: float, ia: float) -> tuple[float, int]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    constant_c = float((raw.get("param_limits") or {}).get("exposure_constant_c", 0.0))
    if constant_c <= 0 or ua <= 0 or ia <= 0:
        raise ValueError("100/80配置中的曝光常数或Ua/Ia无效")
    exposure_ms = max(10, min(1000, int(round(constant_c / (ua * ua * ia)))))
    return constant_c, exposure_ms


def _uniform_random_uc_values(lower: int, upper: int, count: int, seed: int) -> list[int]:
    """把Uc整数范围等分为count层，每层随机取一点，再随机实验顺序。"""
    if lower > upper or count <= 0 or count > upper - lower + 1:
        raise ValueError("Uc均匀随机参数无效")
    rng = random.Random(seed)
    total = upper - lower + 1
    values: list[int] = []
    for index in range(count):
        band_low = lower + index * total // count
        band_high = lower + (index + 1) * total // count - 1
        values.append(rng.randint(band_low, band_high))
    rng.shuffle(values)
    return values


def _read_fbk_once() -> dict[str, float]:
    proc = _run(["modbus-read"])
    fbk = _parse_fbk(proc.stdout)
    if fbk is None:
        raise RuntimeError("无法从modbus-read解析拍照前FBK")
    return fbk


def _clear_temporary_files() -> None:
    for path in (ROOT / "captured_tif" / "single.tif", ROOT / "out" / "single_debug.png"):
        if path.is_file():
            path.unlink()


def _archive(group_number: int, uc: int, shot_number: int, record: dict[str, Any], tif_dir: Path, debug_dir: Path) -> None:
    stem = f"group_{group_number:02d}_Uc_{uc:03d}_shot_{shot_number:02d}"
    for source, target, key in (
        (ROOT / "captured_tif" / "single.tif", tif_dir / f"{stem}.tif", "image_path"),
        (ROOT / "out" / "single_debug.png", debug_dir / f"{stem}_debug.png", "debug_path"),
    ):
        if source.is_file():
            shutil.move(str(source), str(target))
            record[key] = str(target)


def _fmt(value: Any, digits: int = 4) -> str:
    return "-" if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _write_reports(
    report_path: Path,
    csv_path: Path,
    run_id: str,
    records: list[dict[str, Any]],
    uc_values: list[int],
    shots_per_group: int,
    params: dict[str, float],
    exposure_ms: int,
    group_rest_s: float,
    seed: int,
) -> None:
    """输出逐张数据和每个Uc组的五连拍稳定性诊断。"""

    def _values(items: list[dict[str, Any]], key: str) -> list[float]:
        return [
            float(item[key])
            for item in items
            if isinstance(item.get(key), (int, float)) and math.isfinite(float(item[key]))
        ]

    def _relative_range(values: list[float]) -> float | None:
        if not values:
            return None
        low = min(values)
        return None if low == 0 else (max(values) - low) / abs(low)

    def _diagnose(valid: list[dict[str, Any]]) -> str:
        if len(valid) < 3:
            return "有效图少于 3 张，无法判断"
        score_gap = _relative_range(_values(valid, "score"))
        if score_gap is None or score_gap <= 0.10:
            return "score 组内稳定（≤10%）"

        x1_gap = _relative_range(_values(valid, "x1")) or 0.0
        x2_gap = _relative_range(_values(valid, "x2")) or 0.0
        gray_means = [
            float((item.get("gray_stats") or {}).get("mean"))
            for item in valid
            if isinstance((item.get("gray_stats") or {}).get("mean"), (int, float))
        ]
        gray_gap = _relative_range(gray_means) or 0.0
        saturation = [
            float((item.get("gray_stats") or {}).get("saturated_pct"))
            for item in valid
            if isinstance((item.get("gray_stats") or {}).get("saturated_pct"), (int, float))
        ]
        ia_values = [
            float((item.get("feedback") or {}).get("Ia"))
            for item in valid
            if isinstance((item.get("feedback") or {}).get("Ia"), (int, float))
        ]
        ug_values = [
            float((item.get("feedback") or {}).get("Ug"))
            for item in valid
            if isinstance((item.get("feedback") or {}).get("Ug"), (int, float))
        ]

        evidence: list[str] = []
        if x2_gap > max(0.10, x1_gap * 1.5):
            evidence.append("x2 波动主导，优先检查 L2/FWHM 拟合")
        if gray_gap > 0.03 or (saturation and max(saturation) - min(saturation) > 0.5):
            evidence.append("灰度或饱和比例不稳，可能是曝光/图像亮度变化")
        if ia_values and max(ia_values) - min(ia_values) > 1.0:
            evidence.append("五连拍期间Ia反馈变化超过1uA")
        if ug_values and max(ug_values) - min(ug_values) > 1.0:
            evidence.append("五连拍期间Ug反馈变化超过1V")
        return "；".join(evidence) if evidence else "score 波动存在，但灰度、Ug 与 x1/x2 未显示单一主因"

    lines = [
        "# text-10：100/80条件下Uc均匀随机五连拍报告", "",
        f"- 运行编号: `{run_id}`",
        f"- 固定输入: Ua={params['Ua']:g} kV, Ia={params['Ia']:g} uA, Ug={params['Ug']:g} V, If={params['If']:g} A",
        f"- Uc范围: 640–750 V；按10个等宽分层分别随机取1个整数，随机种子={seed}",
        f"- 实际Uc顺序: {uc_values}",
        f"- 曝光时间: {exposure_ms} ms（按100/80配置中的曝光公式计算）",
        f"- 电源周期: 每组连续拍摄 {shots_per_group} 张后关机；下一组前断电休整 {group_rest_s / 60:.1f} 分钟再重启。", "",
        "| 组号 | 设定Uc(V) | 拍次 | 状态 | 曝光(ms) | 平均灰度 | P99 | 饱和像素(%) | score | x1 | x2 | x2/x1 | 拍前 Ua/Ia/Uc/Ug | 原始图像 | 输出图像 | 原因 |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for record in records:
        gray = record.get("gray_stats") or {}
        fbk = record.get("feedback") or {}
        feedback_text = "/".join(
            _fmt(fbk.get(name), 2) for name in ("Ua", "Ia", "Uc", "Ug")
        ) if fbk else "-"
        lines.append(
            f"| {record['group']} | {record['set_uc']} | {record.get('shot', '-')} | {'有效' if record['success'] else '失败'} | {record['exposure_ms']} | "
            f"{_fmt(gray.get('mean'))} | {_fmt(gray.get('p99'))} | {_fmt(gray.get('saturated_pct'), 3)} | "
            f"{_fmt(record.get('score'))} | {_fmt(record.get('x1'))} | {_fmt(record.get('x2'))} | "
            f"{_fmt(record.get('x2_x1_ratio'))} | {feedback_text} | `{record.get('image_path') or '-'}` | "
            f"`{record.get('debug_path') or '-'}` | {record.get('reason') or '-'} |"
        )

    lines.extend([
        "", "## 各Uc组五连拍稳定性", "",
        "| 组号 | Uc(V) | 有效/总张数 | score均值 | score中位数 | score组内范围/最小值 | x1范围 | x2范围 | 平均灰度范围 | Ia范围 | Ug范围 | 饱和像素最大值(%) | 诊断 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- |",
    ])
    by_group: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_group.setdefault(int(record["group"]), []).append(record)
    for group_number, group in sorted(by_group.items()):
        valid = [item for item in group if item.get("success")]
        scores = _values(valid, "score")
        x1s = _values(valid, "x1")
        x2s = _values(valid, "x2")
        means = [
            float((item.get("gray_stats") or {}).get("mean"))
            for item in valid
            if isinstance((item.get("gray_stats") or {}).get("mean"), (int, float))
        ]
        saturation = [
            float((item.get("gray_stats") or {}).get("saturated_pct"))
            for item in valid
            if isinstance((item.get("gray_stats") or {}).get("saturated_pct"), (int, float))
        ]
        ia_values = _values([
            {"value": (item.get("feedback") or {}).get("Ia")} for item in group
        ], "value")
        ug_values = _values([
            {"value": (item.get("feedback") or {}).get("Ug")} for item in group
        ], "value")
        span = lambda values: "-" if not values else f"{min(values):.4f}–{max(values):.4f}"
        lines.append(
            f"| {group_number} | {group[0]['set_uc']} | {len(valid)}/{len(group)} | "
            f"{_fmt(statistics.mean(scores) if scores else None)} | "
            f"{_fmt(statistics.median(scores) if scores else None)} | "
            f"{_fmt(_relative_range(scores), 2)} | {span(x1s)} | {span(x2s)} | "
            f"{span(means)} | {span(ia_values)} | {span(ug_values)} | "
            f"{_fmt(max(saturation) if saturation else None, 3)} | {_diagnose(valid)} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fields = ["group", "set_uc", "exposure_ms", "shot", "success", "gray_mean", "gray_std", "gray_min", "gray_max", "gray_p50", "gray_p90", "gray_p99", "saturated_pct", "score", "x1", "x2", "x2_x1_ratio", "stability_ia_mean", "stability_ia_span", "stability_ug_span", "fbk_ua", "fbk_ia", "fbk_uc", "fbk_ug", "fbk_if", "image_path", "debug_path", "reason"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            gray, fbk, stability = record.get("gray_stats") or {}, record.get("feedback") or {}, record.get("stability") or {}
            writer.writerow({
                "group": record["group"], "set_uc": record["set_uc"], "exposure_ms": record["exposure_ms"], "shot": record.get("shot"), "success": record["success"],
                "gray_mean": gray.get("mean"), "gray_std": gray.get("std"), "gray_min": gray.get("min"), "gray_max": gray.get("max"), "gray_p50": gray.get("p50"), "gray_p90": gray.get("p90"), "gray_p99": gray.get("p99"), "saturated_pct": gray.get("saturated_pct"),
                "score": record.get("score"), "x1": record.get("x1"), "x2": record.get("x2"),
                "x2_x1_ratio": record.get("x2_x1_ratio"),
                "stability_ia_mean": stability.get("ia_mean"), "stability_ia_span": stability.get("ia_span"), "stability_ug_span": stability.get("ug_span"),
                "fbk_ua": fbk.get("Ua"), "fbk_ia": fbk.get("Ia"), "fbk_uc": fbk.get("Uc"), "fbk_ug": fbk.get("Ug"), "fbk_if": fbk.get("If"),
                "image_path": record.get("image_path"), "debug_path": record.get("debug_path"), "reason": record.get("reason", ""),
            })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ua=100/Ia=80下，10个均匀随机Uc的五连拍稳定性测试")
    parser.add_argument("--groups", type=int, default=DEFAULT_GROUPS, help="Uc组数（默认10）")
    parser.add_argument("--uc-min", type=int, default=DEFAULT_UC_MIN, help="Uc下限（默认640V）")
    parser.add_argument("--uc-max", type=int, default=DEFAULT_UC_MAX, help="Uc上限（默认750V）")
    parser.add_argument("--seed", type=int, default=42, help="均匀随机Uc种子（默认42，可复现）")
    parser.add_argument("--shots-per-group", type=int, default=DEFAULT_SHOTS, help="每个Uc连续拍摄张数（默认5）")
    parser.add_argument("--ua", type=float, default=PARAMS["Ua"])
    parser.add_argument("--ia", type=float, default=PARAMS["Ia"])
    parser.add_argument("--ug", type=float, default=PARAMS["Ug"])
    parser.add_argument("--filament-a", type=float, default=PARAMS["If"])
    parser.add_argument("--gun-config", type=Path, default=DEFAULT_GUN_CONFIG, help="默认使用100/80专用配置")
    parser.add_argument("--group-rest-s", type=float, default=DEFAULT_GROUP_REST_S, help="每5拍关机后、下一组重启前的断电秒数（默认300秒）")
    parser.add_argument("--pre-wait-s", type=float, default=5.0)
    parser.add_argument("--capture-pre-wait-s", type=float, default=DEFAULT_CAPTURE_PRE_WAIT_S, help="每张拍照前等待秒数（默认5秒）")
    parser.add_argument("--stable-samples", type=int, default=6, help="Ia/Ug 连续稳定读数次数（默认 6）")
    parser.add_argument("--stable-poll-s", type=float, default=5.0, help="Ia/Ug 稳定查询间隔秒数（默认 5）")
    parser.add_argument("--ia-target-tolerance", type=float, default=0.02, help="Ia 平均值相对目标误差（默认 0.02，即 ±2%%）")
    parser.add_argument("--ia-span-ua", type=float, default=1.0, help="连续读数 Ia 最大范围，单位 uA（默认 1）")
    parser.add_argument("--ug-span-v", type=float, default=1.0, help="连续读数 Ug 最大范围，单位 V（默认 1）")
    parser.add_argument("--stability-timeout-s", type=float, default=180.0, help="每次等待 Ia/Ug 稳定的最长秒数（默认 180）")
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args(argv)
    if args.groups <= 0 or args.groups > args.uc_max - args.uc_min + 1:
        parser.error("Uc组数必须为正，且不能超过范围内整数个数")
    if args.uc_min > args.uc_max:
        parser.error("Uc范围无效")
    if args.shots_per_group <= 0:
        parser.error("--shots-per-group 必须大于0")
    if args.group_rest_s < 0 or args.pre_wait_s < 0 or args.capture_pre_wait_s < 0:
        parser.error("等待时间不能为负数")
    if args.stable_samples <= 0 or args.stable_poll_s < 0 or args.ia_target_tolerance < 0 or args.ia_span_ua < 0 or args.ug_span_v < 0 or args.stability_timeout_s <= 0:
        parser.error("Ia/Ug 稳定性参数无效")

    params = {"Ua": args.ua, "Ia": args.ia, "Ug": args.ug, "If": args.filament_a}
    gun_config_path = args.gun_config.resolve()
    os.environ["GUN_MODBUS_CONFIG"] = str(gun_config_path)
    uc_values = _uniform_random_uc_values(args.uc_min, args.uc_max, args.groups, args.seed)
    exposure_constant, exposure_ms = _exposure_from_gun_config(
        gun_config_path, params["Ua"], params["Ia"]
    )

    run_id = _run_id()
    tif_dir = args.output_root.resolve() / "captured_tif" / run_id
    debug_dir = args.output_root.resolve() / "out" / run_id
    reports_dir = args.output_root.resolve() / "reports" / run_id
    for directory in (tif_dir, debug_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    camera_config_path = ROOT / "camera_cap_config.json"
    original_camera_config = json.loads(camera_config_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    exit_code = 0

    print(f"[text-10] 固定参数: {params}")
    print(f"[text-10] Uc均匀随机顺序: {uc_values}")
    print(
        f"[text-10] 每组连续拍摄 {args.shots_per_group} 张后关机；"
        f"下一组在断电 {args.group_rest_s:.0f}s 后重启"
    )
    print(
        f"[text-10] 曝光时间={exposure_ms}ms "
        f"(C={exposure_constant:.0f}, Ua={params['Ua']:g}, Ia={params['Ia']:g})"
    )
    try:
        _write_camera_exposure(camera_config_path, original_camera_config, exposure_ms)
        _run(["modbus-off"], check=False)
        for group_number, uc in enumerate(uc_values, 1):
            print(f"\n[text-10] ===== 第 {group_number:02d}/{len(uc_values):02d} 组，Uc={uc}V =====")
            if group_number > 1:
                print(
                    f"[text-10] 上一组五拍后设备已关闭，继续断电休整 "
                    f"{args.group_rest_s:.0f}s 后重启 ..."
                )
                time.sleep(args.group_rest_s)
            else:
                print("[text-10] 第一组直接启动，不进行组间休整")

            setup_fbk: dict[str, float] | None = None
            setup_stability: dict[str, float] | None = None
            setup_error = ""
            try:
                _run(["modbus-on"])
                _run([
                    "modbus-set",
                    "--Ua", str(params["Ua"]),
                    "--Ia", str(params["Ia"]),
                    "--Uc", str(uc),
                    "--Ug", str(params["Ug"]),
                    "--If", str(params["If"]),
                ])
                time.sleep(args.pre_wait_s)
                print(f"[text-10] 等待 Ia={params['Ia']:.2f}uA、Ug 同时稳定 ...")
                setup_fbk, setup_stability = _wait_ia_ug_stable(
                    params["Ia"],
                    samples=args.stable_samples,
                    poll_s=args.stable_poll_s,
                    ia_target_tolerance=args.ia_target_tolerance,
                    ia_span_tolerance=args.ia_span_ua,
                    ug_span_tolerance=args.ug_span_v,
                    timeout_s=args.stability_timeout_s,
                )
            except Exception as exc:
                exit_code = 1
                setup_error = f"本组设参或稳定检查失败: {exc}"
                print(f"[text-10] {setup_error}", file=sys.stderr)

            for shot_number in range(1, args.shots_per_group + 1):
                record: dict[str, Any] = {
                    "group": group_number,
                    "set_uc": uc,
                    "exposure_ms": exposure_ms,
                    "shot": shot_number,
                    "success": False,
                    "score": None,
                    "x1": None,
                    "x2": None,
                    "x2_x1_ratio": None,
                    "gray_stats": None,
                    "feedback": None,
                    "stability": setup_stability if shot_number == 1 else None,
                    "image_path": None,
                    "debug_path": None,
                    "reason": setup_error,
                }
                if setup_error:
                    records.append(record)
                    continue
                try:
                    _clear_temporary_files()
                    if shot_number == 1:
                        record["feedback"] = setup_fbk
                    else:
                        record["feedback"] = _read_fbk_once()
                    print(
                        f"[text-10] group={group_number:02d} Uc={uc}V "
                        f"shot={shot_number:02d}: 拍照前等待 {args.capture_pre_wait_s:.0f}s ..."
                    )
                    time.sleep(args.capture_pre_wait_s)
                    _run(["capture"])
                    tif_path = ROOT / "captured_tif" / "single.tif"
                    if not tif_path.is_file():
                        record["reason"] = "capture 失败，无 TIF 生成"
                    else:
                        try:
                            record["gray_stats"] = _gray_stats(tif_path)
                        except Exception as exc:
                            record["reason"] = f"灰度计算失败: {exc}"
                        analyze = _run(["analyze"], check=False)
                        ok, score, x1, x2, reason = _parse_analyze(analyze.stdout)
                        record.update({"success": ok, "score": score, "x1": x1, "x2": x2})
                        if x1 is not None and x2 is not None and x1 != 0:
                            record["x2_x1_ratio"] = x2 / x1
                        if not ok and not record["reason"]:
                            record["reason"] = reason or "图像分析失败"
                except Exception as exc:
                    exit_code = 1
                    record["reason"] = f"异常: {exc}"
                finally:
                    _archive(group_number, uc, shot_number, record, tif_dir, debug_dir)
                gray = record.get("gray_stats") or {}
                print(
                    f"[text-10] group={group_number:02d} Uc={uc:03d}V shot={shot_number:02d}, "
                    f"gray_mean={_fmt(gray.get('mean'))}, saturated={_fmt(gray.get('saturated_pct'), 3)}%, "
                    f"score={_fmt(record.get('score'))}, x2/x1={_fmt(record.get('x2_x1_ratio'))}, ok={record['success']}"
                )
                records.append(record)
            print(f"[text-10] 第 {group_number:02d} 组五拍结束：关闭设备")
            _run(["modbus-off"], check=False)
    finally:
        camera_config_path.write_text(json.dumps(original_camera_config, ensure_ascii=False, indent=2), encoding="utf-8")
        _run(["modbus-off"], check=False)

    report_path = reports_dir / f"text_10_report_{run_id}.md"
    csv_path = reports_dir / f"text_10_summary_{run_id}.csv"
    json_path = reports_dir / f"text_10_history_{run_id}.json"
    _write_reports(
        report_path, csv_path, run_id, records, uc_values, args.shots_per_group,
        params, exposure_ms, args.group_rest_s, args.seed,
    )
    json_path.write_text(json.dumps({
        "run_id": run_id,
        "params": params,
        "uc_values": uc_values,
        "shots_per_group": args.shots_per_group,
        "group_rest_s": args.group_rest_s,
        "exposure_ms": exposure_ms,
        "exposure_constant_c": exposure_constant,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[text-10] 文字报告: {report_path}")
    print(f"[text-10] 表格: {csv_path}")
    print(f"[text-10] 完整记录: {json_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
