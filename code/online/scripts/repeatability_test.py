"""重复性测试：固定一组参数，连续采集多张图，观察峰图差异。

示例::

    python scripts/repeatability_test.py --n-shots 10 --interval-s 2
    python scripts/repeatability_test.py --Ua 90 --Ia 100 --Uc 650 --If 0.43 --n-shots 5

流程::

    python app.py modbus-on
    python app.py modbus-set --Ua ... --Ia ... --Uc ... --Ug ... --If ...
    等待 120s
    循环 python app.py modbus-read，直到 Ug 连续稳定
    重复 n 次：
        python app.py capture
        python app.py analyze
        重命名 captured_tif/single.tif 和 out/single_debug.png
    python app.py modbus-off

采集结束后，脚本加载所有保存的 tif，重新运行 analysis 得到 L1/L2 法向剖面，
绘制所有剖面叠加图以及 score/x1/x2 随 shot 变化的统计图，并输出统计摘要。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _run(cmd: list[str], *, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    """运行 app.py 子命令，并打印输出。"""
    full_cmd = [sys.executable, "app.py", *cmd]
    print(f"[{_timestamp()}] RUN {' '.join(full_cmd)}")
    proc = subprocess.run(
        full_cmd,
        cwd=str(cwd),
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
    """从 modbus-read 输出解析 FBK 行。"""
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


def _wait_ug_stable(
    *,
    eps_v: float = 2.0,
    samples: int = 5,
    poll_s: float = 2.0,
    timeout_s: float = 180.0,
    ma_window: int = 3,
) -> dict[str, float]:
    """循环调用 app.py modbus-read，直到 Ug 的滑动平均连续稳定。

    判据：连续 ``samples`` 次，当前 Ug 与最近 ``ma_window`` 个 Ug 的滑动平均
    之差的绝对值 <= ``eps_v``。
    """
    deadline = time.monotonic() + timeout_s
    stable_count = 0
    ug_history: list[float] = []
    last_fbk: dict[str, float] | None = None

    while time.monotonic() < deadline:
        proc = _run(["modbus-read"])
        fbk = _parse_fbk(proc.stdout)
        if fbk is None:
            raise RuntimeError("无法从 modbus-read 输出解析 FBK")
        last_fbk = fbk
        ug = fbk["Ug"]
        ug_history.append(ug)

        if len(ug_history) >= ma_window + 1:
            ma = sum(ug_history[-ma_window:]) / ma_window
            if abs(ug - ma) <= eps_v:
                stable_count += 1
                if stable_count >= samples:
                    return fbk
            else:
                stable_count = 0

        time.sleep(poll_s)

    raise RuntimeError(
        f"Ug 未在 {timeout_s}s 内稳定（连续 {samples} 次 |Ug - MA{ma_window}| <= {eps_v}V）"
    )


def _single_shot(
    shot: int,
    ua: float,
    ia: float,
    uc: float,
    ug: float,
    if_val: float,
    out_dir: Path,
    *,
    pre_wait_s: float,
    eps_v: float,
    samples: int,
    poll_s: float,
    timeout_s: float,
    ma_window: int,
) -> dict:
    """完成一次完整的独立实验：set -> 等待 -> 稳定 -> capture -> analyze。"""
    print(f"[{_timestamp()}] shot={shot}: modbus-set")
    _run([
        "modbus-set",
        "--Ua", str(ua),
        "--Ia", str(ia),
        "--Uc", str(uc),
        "--Ug", str(ug),
        "--If", str(if_val),
    ])

    print(f"[{_timestamp()}] shot={shot}: 等待 {pre_wait_s}s ...")
    time.sleep(pre_wait_s)

    print(f"[{_timestamp()}] shot={shot}: 等待 Ug 稳定 ...")
    fbk = _wait_ug_stable(
        eps_v=eps_v,
        samples=samples,
        poll_s=poll_s,
        timeout_s=timeout_s,
        ma_window=ma_window,
    )
    print(
        f"[{_timestamp()}] shot={shot}: 稳定后 FBK "
        f"Ua={fbk['Ua']:.2f} Ia={fbk['Ia']:.2f} "
        f"Uc={fbk['Uc']:.2f} Ug={fbk['Ug']:.2f} If={fbk['If']:.3f}"
    )

    print(f"[{_timestamp()}] shot={shot}: capture")
    _run(["capture"])

    print(f"[{_timestamp()}] shot={shot}: analyze")
    ana_proc = _run(["analyze"])

    stem = f"repeat_{shot:03d}"
    tif_dir = ROOT / "captured_tif"
    debug_dir = ROOT / "out"
    tif_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    src_tif = tif_dir / "single.tif"
    dst_tif = out_dir / f"{stem}.tif"
    src_debug = debug_dir / "single_debug.png"
    dst_debug = out_dir / f"{stem}_debug.png"

    if not src_tif.is_file():
        raise RuntimeError(f"shot={shot}: 未找到 {src_tif}")
    shutil.move(str(src_tif), str(dst_tif))

    if src_debug.is_file():
        shutil.move(str(src_debug), str(dst_debug))
    else:
        dst_debug = None

    ok = True
    score = x1 = x2 = float("nan")
    reason = ""
    for line in ana_proc.stdout.splitlines():
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

    return {
        "shot": shot,
        "ok": ok,
        "score": score,
        "x1": x1,
        "x2": x2,
        "reason": reason,
        "tif_path": str(dst_tif.relative_to(ROOT)),
        "debug_path": str(dst_debug.relative_to(ROOT)) if dst_debug else None,
        "fbk": fbk,
    }


def _analyze_saved(
    tif_path: Path,
    config: "AnalysisConfig",
) -> tuple["ScoreResult", np.ndarray] | tuple[None, None]:
    """加载已保存的 tif 并重新运行 analysis，返回 ScoreResult 和 14-bit 图像。"""
    from analysis.io_tif import load_tif14
    from analysis.score import compute_score

    img = load_tif14(str(tif_path))
    if img is None:
        return None, None
    res = compute_score(img, config=config)
    return res, img


def _plot_profiles(
    results: list[dict],
    out_dir: Path,
) -> None:
    """绘制所有 shot 的 L1/L2 法向剖面叠加图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from analysis import load_analysis_config
    from analysis.config_schema import resolved_config_path

    config_path = resolved_config_path(None)
    config = load_analysis_config(config_path)[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.get_cmap("tab10")

    for idx, r in enumerate(results):
        if not r["ok"]:
            continue
        tif_path = ROOT / r["tif_path"]
        res, _ = _analyze_saved(tif_path, config)
        if res is None or not res.ok:
            continue
        color = cmap(idx % 10)
        p1 = res.profile1
        p2 = res.profile2
        if p1 is not None:
            axes[0].plot(p1.t, p1.y, color=color, alpha=0.7, label=f"shot {r['shot']} x1={p1.fwhm:.2f}")
        if p2 is not None:
            axes[1].plot(p2.t, p2.y, color=color, alpha=0.7, label=f"shot {r['shot']} x2={p2.fwhm:.2f}")

    for ax, title in zip(axes, ["L1 (horizontal)", "L2 (vertical)"]):
        ax.set_title(title)
        ax.set_xlabel("t (pixel)")
        ax.set_ylabel("gray level")
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    profile_png = out_dir / "repeat_profiles.png"
    fig.savefig(profile_png, dpi=150)
    plt.close(fig)
    print(f"[{_timestamp()}] 剖面叠加图保存至: {profile_png.relative_to(ROOT)}")


def _plot_stats(results: list[dict], out_dir: Path) -> None:
    """绘制 score/x1/x2 随 shot 的变化及均值±std 参考线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shots = [r["shot"] for r in results if r["ok"]]
    scores = [r["score"] for r in results if r["ok"]]
    x1s = [r["x1"] for r in results if r["ok"]]
    x2s = [r["x2"] for r in results if r["ok"]]

    if not shots:
        print(f"[{_timestamp()}] 没有成功的 shot，跳过统计图")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, label in zip(axes, [scores, x1s, x2s], ["score", "x1", "x2"]):
        ax.plot(shots, vals, "o-", label=label)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ax.axhline(mean, color="r", linestyle="--", label=f"mean={mean:.3f}")
        ax.axhspan(mean - std, mean + std, color="r", alpha=0.1, label=f"±std={std:.3f}")
        ax.set_xlabel("shot")
        ax.set_ylabel(label)
        ax.set_title(f"{label}: mean={mean:.3f}, std={std:.3f}, RSD={std / mean * 100:.1f}%" if mean else label)
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    stats_png = out_dir / "repeat_stats.png"
    fig.savefig(stats_png, dpi=150)
    plt.close(fig)
    print(f"[{_timestamp()}] 统计图保存至: {stats_png.relative_to(ROOT)}")


def _print_summary(results: list[dict]) -> None:
    """打印统计摘要。"""
    ok_results = [r for r in results if r["ok"]]
    print("\n" + "=" * 80)
    print(f"重复性测试完成：共 {len(results)} 张，成功 {len(ok_results)} 张")
    for r in results:
        print(
            f"shot {r['shot']:03d}: ok={r['ok']} score={r['score']:.4f} "
            f"x1={r['x1']:.4f} x2={r['x2']:.4f} "
            f"reason={r['reason'] or '-'}"
        )

    if not ok_results:
        return

    for key in ["score", "x1", "x2"]:
        vals = np.array([r[key] for r in ok_results], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rsd = std / mean * 100 if mean else 0.0
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        print(
            f"{key:6s}: mean={mean:.4f} std={std:.4f} RSD={rsd:.2f}% "
            f"min={min_v:.4f} max={max_v:.4f}"
        )
    print("=" * 80)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="固定参数重复性测试：观察峰图差异")
    parser.add_argument("--Ua", type=float, default=70, help="阳极高压 (kV)，默认 70")
    parser.add_argument("--Ia", type=float, default=100, help="阳极电流 (uA)，默认 100")
    parser.add_argument("--Uc", type=float, default=650, help="阴极电压 (V)，默认 650")
    parser.add_argument("--Ug", type=float, default=200, help="偏压 (V)，默认 200")
    parser.add_argument("--If", type=float, default=0.42, help="灯丝电流 (A)，默认 0.42")
    parser.add_argument("--n-shots", type=int, default=10, help="采集次数，默认 10")
    parser.add_argument("--interval-s", type=float, default=2.0, help="每次采图间隔 (s)，默认 2")
    parser.add_argument("--pre-wait-s", type=float, default=120.0, help="set 后强制等待时间 (s)，默认 120")
    parser.add_argument("--eps-v", type=float, default=2.0, help="Ug 稳定阈值 (V)，默认 2")
    parser.add_argument("--samples", type=int, default=6, help="Ug 连续稳定次数，默认 5")
    parser.add_argument("--poll-s", type=float, default=5.0, help="Ug 查询间隔 (s)，默认 2")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录，默认 out/repeat_YYYYMMDD_HHMMSS",
    )
    parser.add_argument("--skip-hardware", action="store_true", help="跳过硬件采集，只分析 out-dir 中已存在的 repeat_*.tif")
    args = parser.parse_args(argv)

    if args.out_dir:
        out_dir = ROOT / args.out_dir
    else:
        out_dir = ROOT / "out" / f"repeat_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    code = 0

    if not args.skip_hardware:
        try:
            print(f"[{_timestamp()}] modbus-on")
            _run(["modbus-on"])

            for shot in range(1, args.n_shots + 1):
                try:
                    r = _single_shot(
                        shot=shot,
                        ua=args.Ua,
                        ia=args.Ia,
                        uc=args.Uc,
                        ug=args.Ug,
                        if_val=args.If,
                        out_dir=out_dir,
                        pre_wait_s=args.pre_wait_s,
                        eps_v=args.eps_v,
                        samples=args.samples,
                        poll_s=args.poll_s,
                        timeout_s=180.0,
                        ma_window=3,
                    )
                    results.append(r)
                    print(
                        f"[{_timestamp()}] RESULT shot={r['shot']:03d}: "
                        f"ok={r['ok']} score={r['score']:.4f} x1={r['x1']:.4f} x2={r['x2']:.4f}"
                    )
                except Exception as exc:
                    print(f"[{_timestamp()}] ERROR shot={shot}: {exc}", file=sys.stderr)
                    code = 1
                    break

                if shot < args.n_shots and args.interval_s > 0:
                    time.sleep(args.interval_s)

        finally:
            print(f"[{_timestamp()}] modbus-off")
            _run(["modbus-off"], check=False)

    else:
        # 离线模式：从已有 out_dir 加载 repeat_*.tif
        print(f"[{_timestamp()}] 离线模式：加载 {out_dir} 中的 repeat_*.tif")
        for tif_path in sorted(out_dir.glob("repeat_*.tif")):
            m = re.search(r"repeat_(\d{3})\.tif", tif_path.name)
            if not m:
                continue
            shot = int(m.group(1))
            # 尝试读取同目录 debug png 对应的 analyze 结果？离线时重新分析
            from analysis import load_analysis_config
            from analysis.config_schema import resolved_config_path

            config_path = resolved_config_path(None)
            config = load_analysis_config(config_path)[0]
            res, _ = _analyze_saved(tif_path, config)
            if res is None:
                continue
            debug_png = tif_path.with_suffix("").with_name(tif_path.stem + "_debug.png")
            results.append({
                "shot": shot,
                "ok": res.ok,
                "score": res.score,
                "x1": res.x1,
                "x2": res.x2,
                "reason": res.reason,
                "tif_path": str(tif_path.relative_to(ROOT)),
                "debug_path": str(debug_png.relative_to(ROOT)) if debug_png.is_file() else None,
            })

    # 保存结果 json
    result_json = out_dir / "repeat_results.json"
    with result_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[{_timestamp()}] 结果保存至: {result_json.relative_to(ROOT)}")

    _print_summary(results)

    # 绘图
    try:
        _plot_profiles(results, out_dir)
        _plot_stats(results, out_dir)
    except Exception as exc:
        print(f"[{_timestamp()}] 绘图失败: {exc}", file=sys.stderr)
        code = 1

    return code


if __name__ == "__main__":
    raise SystemExit(main())
