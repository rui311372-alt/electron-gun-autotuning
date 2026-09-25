"""批量评估目录中的 TIF：输出 results.csv 与 debug_imgs/。

无命令行参数时，从工程根目录 ``analysis_config.json`` 读取全部配置；
也可通过环境变量 ``ANALYSIS_CONFIG`` 或首个命令行参数指定其它配置文件路径。
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.config_schema import (  # noqa: E402
    load_analysis_config,
    resolve_under_config_dir,
    resolved_config_path,
)
from analysis.io_tif import load_tif14  # noqa: E402
from analysis.score import compute_score  # noqa: E402
from analysis.visualize import save_debug_figure  # noqa: E402


CSV_FIELDS = [
    "file",
    "ok",
    "score",
    "x1",
    "x2",
    "spot_area",
    "spot_cx",
    "spot_cy",
    "L1_length",
    "L2_length",
    "reason",
]


def main() -> int:
    cfg_path = resolved_config_path(
        Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    )
    try:
        cfg, cfg_path = load_analysis_config(cfg_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "请放置 analysis_config.json 于工程根目录，或设置环境变量 ANALYSIS_CONFIG。",
            file=sys.stderr,
        )
        return 2

    rel_in = (cfg.io.batch_input_dir or "").strip()
    if not rel_in:
        print(
            "请在配置文件的 io.batch_input_dir 中填写包含 TIF 的输入目录。",
            file=sys.stderr,
        )
        return 2

    in_dir = resolve_under_config_dir(cfg_path, rel_in)
    if not in_dir.is_dir():
        print(f"input_dir not found: {in_dir}", file=sys.stderr)
        return 2

    batch_out = (cfg.io.batch_output_dir or "").strip()
    if not batch_out:
        print(
            "请在配置文件的 io.batch_output_dir 中填写批量结果输出根目录（相对路径相对配置文件所在目录）。",
            file=sys.stderr,
        )
        return 2
    out_dir = resolve_under_config_dir(cfg_path, batch_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir = out_dir / "debug_imgs"
    if cfg.io.save_debug_images:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(in_dir.glob("*.tif")) + sorted(in_dir.glob("*.tiff"))
    if not tif_files:
        print(f"no tif files in {in_dir}", file=sys.stderr)
        return 2

    print(f"config      : {cfg_path}")

    csv_path = out_dir / "results.csv"
    rows: list[dict[str, object]] = []
    t_start = time.time()
    n_ok = 0

    for i, path in enumerate(tif_files, 1):
        try:
            img = load_tif14(path)
        except Exception as exc:
            row = {f: "" for f in CSV_FIELDS}
            row["file"] = path.name
            row["ok"] = False
            row["reason"] = f"load_error:{exc}"
            rows.append(row)
            print(f"[{i:3d}/{len(tif_files)}] {path.name}  LOAD ERROR: {exc}")
            continue

        try:
            result = compute_score(img, config=cfg)
        except Exception as exc:
            row = {f: "" for f in CSV_FIELDS}
            row["file"] = path.name
            row["ok"] = False
            row["reason"] = f"compute_error:{exc}"
            rows.append(row)
            print(f"[{i:3d}/{len(tif_files)}] {path.name}  COMPUTE ERROR: {exc}")
            continue

        if result.ok:
            n_ok += 1
        row = {
            "file": path.name,
            "ok": result.ok,
            "score": result.score,
            "x1": result.x1,
            "x2": result.x2,
            "spot_area": result.spot.area if result.spot else "",
            "spot_cx": result.spot.centroid[0] if result.spot else "",
            "spot_cy": result.spot.centroid[1] if result.spot else "",
            "L1_length": (
                float(result.line1.length) if result.line1 is not None else ""
            ),
            "L2_length": (
                float(result.line2.length) if result.line2 is not None else ""
            ),
            "reason": result.reason,
        }
        rows.append(row)

        flag = "OK " if result.ok else "FAIL"
        print(
            f"[{i:3d}/{len(tif_files)}] {flag}  score={result.score:8.3f}  "
            f"x1={result.x1:7.3f}  x2={result.x2:7.3f}  {path.name}"
        )

        if cfg.io.save_debug_images:
            out_png = dbg_dir / (path.stem + ".png")
            try:
                save_debug_figure(img, result, out_png, title=path.name)
            except Exception as exc:
                print(f"    debug save error: {exc}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t_start
    print()
    print(
        f"done. total={len(tif_files)}, ok={n_ok}, fail={len(tif_files) - n_ok}, "
        f"success_rate={n_ok / len(tif_files):.1%}, elapsed={elapsed:.1f}s"
    )
    print(f"csv : {csv_path}")
    if cfg.io.save_debug_images:
        print(f"imgs: {dbg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
