"""自动采集脚本：通过调用 app.py 子命令完成。

示例::

    python scripts/auto_collect.py --ua-list 70 80 90 100 110 --n-shots 3

流程::

    python app.py modbus-on（仅一次）
    对每个 Ua、每张子图：
        python app.py modbus-set --Ua ...
        等待 120s
        循环 python app.py modbus-read，直到 Ug 连续稳定
        python app.py capture
        python app.py analyze
        重命名 captured_tif/single.tif 和 out/single_debug.png
    python app.py modbus-off（结束）

文件名根据最后一次 ``modbus-read`` 的实际反馈值命名，如::

    captured_tif/ua90-ia101-uc756-ug162-if424_001.tif
    out/ua90-ia101-uc756-ug162-if424_001_debug.png
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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


def _build_stem(fbk: dict[str, float], shot: int) -> str:
    """根据实际反馈值构建文件名 stem，如 ua90-ia101-uc756-ug162-if424_001。"""
    return (
        f"ua{int(round(fbk['Ua'])):d}-"
        f"ia{int(round(fbk['Ia'])):d}-"
        f"uc{int(round(fbk['Uc'])):d}-"
        f"ug{int(round(fbk['Ug'])):d}-"
        f"if{int(round(fbk['If'] * 1000)):d}_"
        f"{shot:03d}"
    )


def _wait_ug_stable(
    *,
    eps_v: float = 2.0,
    samples: int = 5,
    poll_s: float = 2.0,
    timeout_s: float = 180.0,
) -> dict[str, float]:
    """循环调用 app.py modbus-read，直到 Ug 自身连续稳定。

    返回最后一次成功解析的 FBK 字典。
    """
    deadline = time.monotonic() + timeout_s
    stable_count = 0
    last_ug: float | None = None
    last_fbk: dict[str, float] | None = None

    while time.monotonic() < deadline:
        proc = _run(["modbus-read"])
        fbk = _parse_fbk(proc.stdout)
        if fbk is None:
            raise RuntimeError("无法从 modbus-read 输出解析 FBK")
        last_fbk = fbk
        ug = fbk["Ug"]

        if last_ug is not None and abs(ug - last_ug) <= eps_v:
            stable_count += 1
            if stable_count >= samples:
                return fbk
        else:
            stable_count = 0
        last_ug = ug

        time.sleep(poll_s)

    raise RuntimeError(
        f"Ug 未在 {timeout_s}s 内稳定（连续 {samples} 次变化 <= {eps_v}V）"
    )


def _single_shot(
    ua: float,
    shot: int,
    ia: float,
    uc: float,
    ug: float,
    if_val: float,
) -> dict:
    """完成一次完整采集。"""
    print(f"[{_timestamp()}] Ua={ua} shot={shot}: modbus-set")
    _run([
        "modbus-set",
        "--Ua", str(ua),
        "--Ia", str(ia),
        "--Uc", str(uc),
        "--Ug", str(ug),
        "--If", str(if_val),
    ])

    print(f"[{_timestamp()}] Ua={ua} shot={shot}: waiting 120s before stability check ...")
    time.sleep(120)

    print(f"[{_timestamp()}] Ua={ua} shot={shot}: waiting Ug stable ...")
    fbk = _wait_ug_stable()
    print(
        f"[{_timestamp()}] Ua={ua} shot={shot}: FBK "
        f"Ua={fbk['Ua']:.2f} Ia={fbk['Ia']:.2f} "
        f"Uc={fbk['Uc']:.2f} Ug={fbk['Ug']:.2f} If={fbk['If']:.3f}"
    )

    print(f"[{_timestamp()}] Ua={ua} shot={shot}: capture")
    _run(["capture"])

    print(f"[{_timestamp()}] Ua={ua} shot={shot}: analyze")
    ana_proc = _run(["analyze"])

    stem = _build_stem(fbk, shot)
    tif_dir = ROOT / "captured_tif"
    debug_dir = ROOT / "out"
    tif_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    src_tif = tif_dir / "single.tif"
    dst_tif = tif_dir / f"{stem}.tif"
    src_debug = debug_dir / "single_debug.png"
    dst_debug = debug_dir / f"{stem}_debug.png"

    if not src_tif.is_file():
        raise RuntimeError(f"Ua={ua} shot={shot}: 未找到 {src_tif}")
    shutil.move(str(src_tif), str(dst_tif))

    if src_debug.is_file():
        shutil.move(str(src_debug), str(dst_debug))
    else:
        dst_debug = None

    # 从 analyze 输出解析 score/x1/x2
    ok = True
    score = x1 = x2 = float("nan")
    reason = ""
    for line in ana_proc.stdout.splitlines():
        if line.strip().startswith("ok"):
            ok = "True" in line
        elif line.strip().startswith("x1 / x2"):
            m = re.search(r"([\d.]+)\s*/\s*([\d.]+)", line)
            if m:
                x1, x2 = float(m.group(1)), float(m.group(2))
        elif line.strip().startswith("score"):
            m = re.search(r"([\d.]+)", line)
            if m:
                score = float(m.group(1))
        elif line.strip().startswith("reason"):
            reason = line.split(":", 1)[1].strip()

    return {
        "ua": ua,
        "shot": shot,
        "ok": ok,
        "score": score,
        "x1": x1,
        "x2": x2,
        "reason": reason,
        "tif_path": dst_tif,
        "debug_path": dst_debug,
        "fbk": fbk,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="电子枪自动采集：调用 app.py 子命令")
    parser.add_argument(
        "--ua-list",
        type=float,
        nargs="+",
        default=[70, 80, 90, 100, 110],
        help="阳极高压列表 (kV)，默认 70 80 90 100 110",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=3,
        help="每个 Ua 采集次数，默认 3",
    )
    parser.add_argument("--ia", type=float, default=100, help="阳极电流 (uA)，默认 100")
    parser.add_argument("--uc", type=float, default=650, help="阴极电压 (V)，默认 650")
    parser.add_argument("--ug", type=float, default=200, help="偏压 (V)，默认 200")
    parser.add_argument("--if", type=float, default=0.42, help="灯丝电流 (A)，默认 0.42")
    parser.add_argument(
        "--eps-v",
        type=float,
        default=2.0,
        help="Ug 稳定阈值 (V)，默认 2",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Ug 连续稳定次数，默认 5",
    )
    parser.add_argument(
        "--poll-s",
        type=float,
        default=2.0,
        help="Ug 查询间隔 (s)，默认 2",
    )
    parser.add_argument(
        "--pre-wait-s",
        type=float,
        default=120.0,
        help="set_params 后强制等待时间 (s)，默认 120",
    )
    args = parser.parse_args(argv)

    results: list[dict] = []
    code = 0

    try:
        print(f"[{_timestamp()}] modbus-on")
        _run(["modbus-on"])

        for ua in args.ua_list:
            for shot in range(1, args.n_shots + 1):
                try:
                    r = _single_shot(
                        ua=ua,
                        shot=shot,
                        ia=args.ia,
                        uc=args.uc,
                        ug=args.ug,
                        if_val=getattr(args, "if"),
                    )
                    results.append(r)
                    print(
                        f"[{_timestamp()}] RESULT Ua={r['ua']} shot={r['shot']}: "
                        f"ok={r['ok']} score={r['score']:.4f} x1={r['x1']:.4f} x2={r['x2']:.4f} "
                        f"debug={r['debug_path']}"
                    )
                except Exception as exc:
                    print(f"[{_timestamp()}] ERROR Ua={ua} shot={shot}: {exc}", file=sys.stderr)
                    code = 1
                    raise

                print("-" * 80)

    finally:
        print(f"[{_timestamp()}] modbus-off")
        _run(["modbus-off"], check=False)

    print(f"\n完成：共 {len(results)} 张图")
    for r in results:
        stem = _build_stem(r["fbk"], r["shot"])
        print(
            f"{stem}: ok={r['ok']} score={r['score']:.4f} "
            f"x1={r['x1']:.4f} x2={r['x2']:.4f}"
        )

    return code


if __name__ == "__main__":
    raise SystemExit(main())
