"""分析管线配置：从 JSON 载入为强类型对象，路径与默认值集中在此模块。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def default_config_path() -> Path:
    """默认配置文件：工程根目录下的 ``analysis_config.json``（与 ``analysis/`` 同级）。"""
    return Path(__file__).resolve().parent.parent / "analysis_config.json"


def resolved_config_path(explicit: Path | None = None) -> Path:
    """解析本次运行使用的配置文件路径。

    优先级：``explicit``（一般为脚本首个命令行参数）→ 环境变量 ``ANALYSIS_CONFIG`` →
    :func:`default_config_path`（无参数且未设环境变量时，即工程根目录 ``analysis_config.json``）。
    """
    if explicit is not None:
        return explicit.expanduser()
    env = os.environ.get("ANALYSIS_CONFIG", "").strip()
    if env:
        return Path(env).expanduser()
    return default_config_path()


def _strip_meta_keys(obj: Any) -> Any:
    """去掉 ``_note``、``*_comment`` 等不参与反序列化的键。"""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if k == "_note" or k.endswith("__comment") or k.endswith("_comment"):
                continue
            out[k] = _strip_meta_keys(v)
        return out
    if isinstance(obj, list):
        return [_strip_meta_keys(x) for x in obj]
    return obj


def _section(cls: type[T], data: dict[str, Any] | None) -> T:
    data = data or {}
    names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in names}
    return cls(**kwargs)


@dataclass
class IOConfig:
    """脚本 I/O，与算法参数分离。"""

    single_image: str = ""
    save_debug_path: str | None = None
    batch_input_dir: str = ""
    batch_output_dir: str = "out"
    save_debug_images: bool = True


@dataclass
class SpotConfig:
    percentile: float = 99.5
    min_area: int = 200
    morph_kernel: int = 5
    saturation_floor: int | None = 16380
    relaxed_saturation_floors: list[int] = field(
        default_factory=lambda: [16300, 16200, 16000]
    )


@dataclass
class LinesConfig:
    min_segment_length: float = 10.0
    black_delta: float = 0.0
    center_size: int = 20
    black_band_width: int = 20
    end_length: int = 50
    stop_before_mask: int = 50
    grad_min: float = 80.0
    angle_half_range_deg: float = 25.0
    angle_step_deg: float = 2.0
    edge_shrink_ratio: float = 0.10
    line_keep_ratio: float = 0.75


@dataclass
class ProfileConfig:
    half_length: float = 50.0
    samples: int = 301
    smooth_window: int = 21
    smooth_polyorder: int = 3
    prefilter_sigma: float = 1.5
    prefilter_gaussian_ksize: int = 9
    base_offset: float = 2.0
    peak_search_half: float = 5.0
    parallel_avg_half_width: float = 4.0
    parallel_avg_step_scale: float = 1.0
    multi_profile_half_width: float = 10.0
    multi_profile_step_scale: float = 2.0
    profile_method: str = "parallel"  # "parallel" 或 "multi"
    monotone_fit: bool = True
    monotone_method: str = "erf"  # "pchip" | "erf" | "sigmoid"
@dataclass
class FwhmConfig:
    """主峰旁瓣判据（与 ``dy/dt`` 剖面相关）。"""

    side_peak_ratio: float = 0.95


@dataclass
class AnalysisConfig:
    io: IOConfig = field(default_factory=IOConfig)
    spot: SpotConfig = field(default_factory=SpotConfig)
    lines: LinesConfig = field(default_factory=LinesConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    fwhm: FwhmConfig = field(default_factory=FwhmConfig)


def parse_analysis_config_dict(raw: dict[str, Any]) -> AnalysisConfig:
    """将已加载的 dict（可含注释键）转为 :class:`AnalysisConfig`。"""
    d = _strip_meta_keys(raw)
    if not isinstance(d, dict):
        d = {}
    return AnalysisConfig(
        io=_section(IOConfig, d.get("io") if isinstance(d.get("io"), dict) else None),
        spot=_section(SpotConfig, d.get("spot") if isinstance(d.get("spot"), dict) else None),
        lines=_section(LinesConfig, d.get("lines") if isinstance(d.get("lines"), dict) else None),
        profile=_section(
            ProfileConfig, d.get("profile") if isinstance(d.get("profile"), dict) else None
        ),
        fwhm=_section(FwhmConfig, d.get("fwhm") if isinstance(d.get("fwhm"), dict) else None),
    )


def load_analysis_config(path: Path | None = None) -> tuple[AnalysisConfig, Path]:
    """从 JSON 加载配置，返回 ``(config, 实际使用的配置文件绝对路径)``。"""
    p = resolved_config_path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"analysis config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    return parse_analysis_config_dict(raw), p


def resolve_under_config_dir(config_path: Path, value: str) -> Path:
    """相对路径相对于配置文件所在目录解析。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


_SINGLE_DEBUG_FILE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf", ".svg", ".webp"}
)


def resolve_single_save_debug_path(
    config_path: Path,
    save_debug_raw: str,
    source_image_path: Path,
) -> Path:
    """将 ``io.save_debug_path`` 解析为单张调试图的完整文件路径。

    以常见图像或 PDF 等扩展名结尾时视为**完整文件路径**（与旧行为一致）；否则视为**目录**
    （可尚不存在），写入 ``<源图主文件名>_debug.png``。单张闭环默认与采集一致：
    ``captured_tif/single.tif`` → ``out/single_debug.png``（见 ``camera_capture.frame_names``）。
    """
    raw = (save_debug_raw or "").strip()
    if not raw:
        raise ValueError("save_debug_raw must be non-empty")
    base = resolve_under_config_dir(config_path, raw)
    if base.suffix.lower() in _SINGLE_DEBUG_FILE_SUFFIXES:
        return base
    return base / f"{source_image_path.stem}_debug.png"
