"""离线统计 auto_collect 已有数据的重复性（标准库版本，无需 numpy）。

用法::

    python scripts/analyze_existing_repeatability.py

自动扫描 out/ 下 ``ua<ua>-ia<ia>-uc<uc>-ug<ug>-if<if>_<shot>_debug.png`` 命名的文件，
按 Ua 分组，重新 analyze 对应 tif，输出每组 score/x1/x2 的均值、std、RSD。
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import load_analysis_config
from analysis.config_schema import resolved_config_path
from analysis.io_tif import load_tif14
from analysis.score import compute_score


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def main() -> int:
    config, _config_path = load_analysis_config(None)

    out_dir = ROOT / "out"
    captured_dir = ROOT / "captured_tif"

    pattern = re.compile(
        r"ua(\d+)-ia(\d+)-uc(\d+)-ug(\d+)-if(\d+)_(\d{3})_debug\.png"
    )

    groups: dict[float, list[dict]] = {}
    for png in sorted(out_dir.glob("ua*-ia*-uc*-ug*-if*_*_debug.png")):
        m = pattern.match(png.name)
        if not m:
            continue
        ua = int(m.group(1))
        shot = int(m.group(6))
        tif = captured_dir / png.name.replace("_debug.png", ".tif")
        if not tif.is_file():
            print(f"WARN: 未找到 {tif}", file=sys.stderr)
            continue

        img = load_tif14(str(tif))
        if img is None:
            print(f"WARN: 无法读取 {tif}", file=sys.stderr)
            continue
        res = compute_score(img, config=config)

        groups.setdefault(float(ua), []).append({
            "shot": shot,
            "ok": res.ok,
            "score": res.score,
            "x1": res.x1,
            "x2": res.x2,
            "reason": res.reason,
            "tif": str(tif.relative_to(ROOT)),
        })

    print(f"{'Ua':>4s} {'n':>3s} {'score_mean':>10s} {'score_std':>9s} {'score_RSD%':>9s} "
          f"{'x1_mean':>8s} {'x1_std':>7s} {'x1_RSD%':>7s} "
          f"{'x2_mean':>8s} {'x2_std':>7s} {'x2_RSD%':>7s}")
    print("-" * 90)

    for ua in sorted(groups):
        rows = [r for r in groups[ua] if r["ok"]]
        n = len(rows)
        if n < 2:
            print(f"{ua:>4.0f} {n:>3d} 成功样本不足")
            continue

        def _stat(key: str) -> tuple[float, float, float]:
            vals = [r[key] for r in rows if math.isfinite(r[key])]
            if len(vals) < 2:
                return (vals[0] if vals else float("nan"), 0.0, 0.0)
            mean = _mean(vals)
            std = _std(vals)
            rsd = std / mean * 100 if mean else 0.0
            return mean, std, rsd

        sm, ss, sr = _stat("score")
        x1m, x1s, x1r = _stat("x1")
        x2m, x2s, x2r = _stat("x2")
        print(
            f"{ua:>4.0f} {n:>3d} "
            f"{sm:>10.4f} {ss:>9.4f} {sr:>9.2f} "
            f"{x1m:>8.4f} {x1s:>7.4f} {x1r:>7.2f} "
            f"{x2m:>8.4f} {x2s:>7.4f} {x2r:>7.2f}"
        )

        for r in rows:
            print(f"       shot {r['shot']:03d}: score={r['score']:.4f} x1={r['x1']:.4f} x2={r['x2']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
