"""单张 TIF 调试入口: 算 score 并保存调试图。

无命令行参数时，从工程根目录 ``analysis_config.json`` 读取全部配置（I/O 与算法参数）；
也可通过环境变量 ``ANALYSIS_CONFIG`` 或首个命令行参数指定其它配置文件路径。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.config_schema import (  # noqa: E402
    load_analysis_config,
    resolve_single_save_debug_path,
    resolve_under_config_dir,
    resolved_config_path,
)
from analysis.io_tif import load_tif14  # noqa: E402
from analysis.score import compute_score  # noqa: E402
from analysis.visualize import save_debug_figure  # noqa: E402


def main() -> int:
    cfg_path = resolved_config_path(
        Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    )
    try:
        cfg, cfg_path = load_analysis_config(cfg_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "请放置 analysis_config.json 于工程根目录，或设置环境变量 ANALYSIS_CONFIG 指向配置文件。",
            file=sys.stderr,
        )
        return 2

    rel = (cfg.io.single_image or "").strip()
    if not rel:
        print(
            "请在配置文件的 io.single_image 中填写待分析的 TIF 路径（相对路径相对配置文件所在目录）。",
            file=sys.stderr,
        )
        return 2

    img_path = resolve_under_config_dir(cfg_path, rel)
    if not img_path.is_file():
        print(f"图像文件不存在: {img_path}", file=sys.stderr)
        return 2

    img = load_tif14(img_path)

    print(f"config      : {cfg_path}")
    print(f"image       : {img_path.name}")
    print(f"shape/dtype : {img.shape}, {img.dtype}")
    print(f"min/max     : {int(img.min())}, {int(img.max())}")

    result = compute_score(img, config=cfg)

    print(f"ok          : {result.ok}")
    print(f"x1 / x2     : {result.x1:.4f}  /  {result.x2:.4f}")
    print(f"score       : {result.score:.4f}")
    if result.reason:
        print(f"reason      : {result.reason}")

    title = img_path.name
    save_raw = (cfg.io.save_debug_path or "").strip()
    if not save_raw:
        print(
            "请在配置文件的 io.save_debug_path 中填写调试图输出目录或文件路径（相对路径相对配置文件所在目录）。",
            file=sys.stderr,
        )
        return 2
    out_path = resolve_single_save_debug_path(cfg_path, save_raw, img_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_debug_figure(img, result, out_path, title=title)
    print(f"saved       : {out_path}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
