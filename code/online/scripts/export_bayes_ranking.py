"""把 ``bayes_opt_history.json`` 导出为按 score 排序的 Markdown 报告。

用法::

    python scripts/export_bayes_ranking.py [history.json] [ranking.md]

默认输入 ``bayes_opt_history.json``，默认输出 ``bayes_opt_ranking_YYYYMMDD.md``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _format_param(p: dict) -> str:
    return (
        f"Ua={p['anode_kv']:.2f} "
        f"Ia={p['anode_ua']:.2f} "
        f"Uc={p['cathode_v']:.2f} "
        f"Ug={p['bias_v']:.1f} "
        f"If={p['fil_a']:.4f}"
    )


def _format_feedback(fb: dict) -> str:
    return (
        f"Ua={fb.get('anode_kv', float('nan')):.2f} "
        f"Ia={fb.get('anode_ua', float('nan')):.2f} "
        f"Uc={fb.get('cathode_v', float('nan')):.2f} "
        f"Ug={fb.get('bias_v', float('nan')):.1f} "
        f"If={fb.get('fil_a', float('nan')):.4f}"
    )


def export_ranking(history_path: Path, output_path: Path) -> int:
    if not history_path.is_file():
        print(f"未找到历史文件: {history_path}", file=sys.stderr)
        return 1

    with history_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    records: list[dict] = []
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("results", [])

    if not records:
        print("历史文件为空或格式错误", file=sys.stderr)
        return 1

    ok_records = [r for r in records if r.get("success")]
    fail_records = [r for r in records if not r.get("success")]

    # 按 (Ua, Ia) 分组
    def _group_key(r: dict) -> tuple[float, float]:
        p = r["params"]
        return (round(float(p["anode_kv"]), 2), round(float(p["anode_ua"]), 2))

    from collections import defaultdict

    ok_groups: dict[tuple[float, float], list[dict]] = defaultdict(list)
    fail_groups: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for r in ok_records:
        ok_groups[_group_key(r)].append(r)
    for r in fail_records:
        fail_groups[_group_key(r)].append(r)

    # 组内排序
    for g in ok_groups.values():
        g.sort(key=lambda r: float(r.get("score", float("inf"))))
    for g in fail_groups.values():
        g.sort(key=lambda r: r.get("iteration", 0))

    # 全局最优
    global_best = min(ok_records, key=lambda r: float(r.get("score", float("inf")))) if ok_records else None

    lines: list[str] = []
    lines.append("# 贝叶斯优化参数排序报告")
    lines.append("")
    lines.append(f"- 历史文件: `{history_path}`")
    lines.append(f"- 总迭代数: {len(records)}")
    lines.append(f"- 成功迭代: {len(ok_records)}")
    lines.append(f"- 失败迭代: {len(fail_records)}")
    lines.append(f"- (Ua, Ia) 组合数: {len(ok_groups)}")
    if global_best:
        lines.append(
            f"- 全局最优: score={global_best['score']:.4f}, "
            f"x1={global_best.get('x1', float('nan')):.4f}, "
            f"x2={global_best.get('x2', float('nan')):.4f}"
        )
    lines.append("")

    # 按组合输出成功排名
    for (ua, ia) in sorted(ok_groups.keys()):
        group = ok_groups[(ua, ia)]
        lines.append(f"## Ua={ua:.0f}kV Ia={ia:.0f}uA（{len(group)} 次成功）")
        lines.append("")
        if global_best and _group_key(global_best) == (ua, ia):
            lines.append("> **此组合包含全局最优**")
            lines.append("")
        lines.append(
            "| 排名 | 迭代 | 设定参数 | 反馈参数 | score | x1 | x2 | TIF | debug |"
        )
        lines.append("| ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- |")
        for rank, r in enumerate(group, start=1):
            params_str = _format_param(r["params"])
            fbk_str = _format_feedback(r.get("feedback", {}))
            tif = r.get("image_path") or "-"
            debug = r.get("debug_path") or "-"
            lines.append(
                f"| {rank} | {r['iteration']} | {params_str} | {fbk_str} | "
                f"{r['score']:.4f} | {r.get('x1', float('nan')):.4f} | "
                f"{r.get('x2', float('nan')):.4f} | `{tif}` | `{debug}` |"
            )
        lines.append("")

        # 同组合的失败记录
        fails = fail_groups.get((ua, ia), [])
        if fails:
            lines.append(f"### 失败迭代（{len(fails)} 次）")
            lines.append("")
            lines.append("| 迭代 | 设定参数 | 反馈参数 | 原因 |")
            lines.append("| ---: | --- | --- | --- |")
            for r in fails:
                params_str = _format_param(r["params"])
                fbk_str = _format_feedback(r.get("feedback", {}))
                reason = r.get("reason", "unknown")
                lines.append(f"| {r['iteration']} | {params_str} | {fbk_str} | {reason} |")
            lines.append("")

    # 不在成功组里的失败记录（Ua/Ia 组合全部失败）
    remaining_fails = [
        (ua, ia, g) for (ua, ia), g in fail_groups.items() if (ua, ia) not in ok_groups
    ]
    for (ua, ia, fails) in sorted(remaining_fails):
        lines.append(f"## Ua={ua:.0f}kV Ia={ia:.0f}uA（全部失败 {len(fails)} 次）")
        lines.append("")
        lines.append("| 迭代 | 设定参数 | 反馈参数 | 原因 |")
        lines.append("| ---: | --- | --- | --- |")
        for r in fails:
            params_str = _format_param(r["params"])
            fbk_str = _format_feedback(r.get("feedback", {}))
            reason = r.get("reason", "unknown")
            lines.append(f"| {r['iteration']} | {params_str} | {fbk_str} | {reason} |")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"排名报告已保存: {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出贝叶斯优化排名报告")
    parser.add_argument(
        "history",
        type=Path,
        nargs="?",
        default=ROOT / "bayes_opt_history.json",
        help="历史 JSON 路径，默认 bayes_opt_history.json",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="输出 md 路径，默认 bayes_opt_ranking_YYYYMMDD.md",
    )
    args = parser.parse_args(argv)

    if args.output is None:
        import time

        date_str = time.strftime("%Y%m%d")
        args.output = ROOT / f"bayes_opt_ranking_{date_str}.md"

    return export_ranking(args.history, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
