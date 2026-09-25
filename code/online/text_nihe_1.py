#!/usr/bin/env python3
"""批量拟合已有 TIF 图像，不连接相机或电子枪。

默认读取 ``D:\\RESULT\\11.14`` 中的 TIF，逐张调用项目当前的
``analysis_config.json`` 和 ``compute_score`` 算法。原始图不移动、不覆盖；
调试图写入 ``D:\\RESULT\\out\\<run_id>``，文字与结构化结果写入
``D:\\RESULT\\reports\\<run_id>``。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from analysis.config_schema import load_analysis_config
from analysis.io_tif import load_tif14
from analysis.score import compute_score
from analysis.visualize import save_debug_figure


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = Path(r"D:\RESULT")
DEFAULT_INPUT_DIR = RESULT_ROOT / "11.14"


def _run_id() -> str:
    return f"text_nihe_1_{time.strftime('%Y%m%d_%H%M%S')}"


def _number(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _write_reports(report_path: Path, csv_path: Path, run_id: str, records: list[dict[str, Any]]) -> None:
    valid = [record for record in records if record.get("success")]
    scores = [float(record["score"]) for record in valid]
    lines = [
        "# text_nihe_1 批量拟合报告",
        "",
        f"- 运行编号: `{run_id}`",
        f"- 输入图像: {len(records)} 张",
        f"- 成功拟合: {len(valid)} 张",
        f"- 失败拟合: {len(records) - len(valid)} 张",
        f"- score 最小/平均/最大: {_number(min(scores) if scores else None)} / "
        f"{_number(sum(scores) / len(scores) if scores else None)} / {_number(max(scores) if scores else None)}",
        "",
        "## 原始顺序",
        "",
        "| 序号 | 原始图像 | 状态 | score | x1 | x2 | 原因 | 调试图 |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for index, record in enumerate(records, 1):
        lines.append(
            f"| {index} | `{record['image_path']}` | {'有效' if record['success'] else '失败'} | "
            f"{_number(record.get('score'))} | {_number(record.get('x1'))} | "
            f"{_number(record.get('x2'))} | {record.get('reason') or '-'} | "
            f"`{record.get('debug_path') or '-'}` |"
        )

    lines.extend([
        "",
        "## score 排名（由小到大）",
        "",
        "| 排名 | 原始图像 | score | x1 | x2 | 调试图 |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ])
    for rank, record in enumerate(sorted(valid, key=lambda item: float(item["score"])), 1):
        lines.append(
            f"| {rank} | `{record['image_path']}` | {_number(record['score'])} | "
            f"{_number(record['x1'])} | {_number(record['x2'])} | "
            f"`{record.get('debug_path') or '-'}` |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fields = ["image_path", "success", "score", "x1", "x2", "reason", "debug_path"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record.get(field) for field in fields} for record in records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批量计算已有 TIF 的 x1、x2 和 score，不连接设备")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help=r"输入 TIF 文件夹，默认 D:\RESULT\11.14")
    parser.add_argument("--analysis-config", type=Path, default=ROOT / "analysis_config.json")
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT, help=r"结果根目录，默认 D:\RESULT")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args(argv)

    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    if not input_dir.is_dir():
        parser.error(f"输入文件夹不存在: {input_dir}")
    tif_files = sorted(
        {*input_dir.glob("*.tif"), *input_dir.glob("*.TIF"), *input_dir.glob("*.tiff"), *input_dir.glob("*.TIFF")},
        key=lambda path: path.name.lower(),
    )
    if not tif_files:
        parser.error(f"输入文件夹中没有 TIF: {input_dir}")

    analysis_cfg, analysis_cfg_path = load_analysis_config(args.analysis_config)
    run_id = args.run_id or _run_id()
    debug_dir = output_root / "out" / run_id
    reports_dir = output_root / "reports" / run_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[text_nihe_1] 输入目录: {input_dir}")
    print(f"[text_nihe_1] 分析配置: {analysis_cfg_path}")
    print(f"[text_nihe_1] 图像数: {len(tif_files)}")
    print(f"[text_nihe_1] 调试图目录: {debug_dir}")
    print(f"[text_nihe_1] 报告目录: {reports_dir}")

    records: list[dict[str, Any]] = []
    for index, tif_path in enumerate(tif_files, 1):
        record: dict[str, Any] = {
            "image_path": str(tif_path), "success": False,
            "score": None, "x1": None, "x2": None,
            "reason": "", "debug_path": None,
        }
        try:
            image = load_tif14(tif_path)
            result = compute_score(image, config=analysis_cfg)
            record.update({
                "success": bool(result.ok),
                "score": float(result.score),
                "x1": float(result.x1),
                "x2": float(result.x2),
                "reason": result.reason or "",
            })
            debug_path = debug_dir / f"{index:02d}_{tif_path.stem}_debug.png"
            save_debug_figure(image, result, debug_path, title=tif_path.name)
            record["debug_path"] = str(debug_path)
            if not result.ok:
                record["reason"] = record["reason"] or "拟合结果无效"
        except Exception as exc:
            record["reason"] = f"分析异常: {exc}"
        records.append(record)
        print(
            f"[text_nihe_1] {index:02d}/{len(tif_files):02d} {tif_path.name}: "
            f"ok={record['success']} score={_number(record.get('score'))} "
            f"x1={_number(record.get('x1'))} x2={_number(record.get('x2'))}"
        )

    report_path = reports_dir / f"text_nihe_1_report_{run_id}.md"
    csv_path = reports_dir / f"text_nihe_1_summary_{run_id}.csv"
    json_path = reports_dir / f"text_nihe_1_history_{run_id}.json"
    _write_reports(report_path, csv_path, run_id, records)
    json_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "input_dir": str(input_dir),
                "analysis_config": str(analysis_cfg_path),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[text_nihe_1] 文字报告: {report_path}")
    print(f"[text_nihe_1] 表格: {csv_path}")
    print(f"[text_nihe_1] 完整记录: {json_path}")
    return 0 if all(record["success"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
