"""串起 detect_spot/fit_lines/profile, 输出 (x1, x2, score) 和调试中间产物。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

import numpy as np

from .config_schema import AnalysisConfig
from .detect_spot import SpotDetection, detect_lower_left_spot
from .fit_lines import LineFit, get_line_candidates
from .profile import (
    ProfileResult,
    compute_multi_normal_profile,
    compute_normal_profile,
    compute_parallel_smoothed_profile,
)


@dataclass
class ScoreResult:
    """完整的评分结果与调试信息。"""

    ok: bool
    score: float  # score: float  # sqrt(x1^2 + x2^2), 失败时为 nan
    x1: float  # 上线法向 dy/dt<0 主峰的 FWHM (亮斑→阴影过渡宽度)
    x2: float  # 右线法向 dy/dt<0 主峰的 FWHM
    reason: str = ""
    spot: SpotDetection | None = None
    line1: LineFit | None = None
    line2: LineFit | None = None
    profile1: ProfileResult | None = None
    profile2: ProfileResult | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def compute_score(img_14bit: np.ndarray, *, config: AnalysisConfig) -> ScoreResult:
    """主评分函数；全部算法参数来自 ``config``。

    返回 :class:`ScoreResult`,其中:
        - `ok=True`  表示两条线的 FWHM 都成功测得
        - `ok=False` 时 `reason` 给出失败原因, `score=nan`,但中间产物尽量保留以便调试
    """
    s = config.spot
    ln = config.lines
    pr = config.profile
    fw = config.fwhm

    spot = detect_lower_left_spot(
        img_14bit,
        percentile=s.percentile,
        min_area=s.min_area,
        morph_kernel=s.morph_kernel,
        saturation_floor=s.saturation_floor,
        relaxed_saturation_floors=s.relaxed_saturation_floors,
    )
    if spot is None:
        return ScoreResult(
            ok=False,
            score=float("nan"),
            x1=float("nan"),
            x2=float("nan"),
            reason="spot_not_found",
        )

    horiz_cands, vert_cands = get_line_candidates(
        img_14bit,
        spot.mask,
        min_seg_len=ln.min_segment_length,
        black_delta=ln.black_delta,
        center_size=ln.center_size,
        black_band_width=ln.black_band_width,
        end_length=ln.end_length,
        stop_before_mask=ln.stop_before_mask,
        grad_min=ln.grad_min,
        angle_half_range_deg=ln.angle_half_range_deg,
        angle_step_deg=ln.angle_step_deg,
        edge_shrink_ratio=ln.edge_shrink_ratio,
        line_keep_ratio=ln.line_keep_ratio,
    )
    if not horiz_cands and not vert_cands:
        return ScoreResult(
            ok=False,
            score=float("nan"),
            x1=float("nan"),
            x2=float("nan"),
            reason="both_lines_failed",
            spot=spot,
        )

    def _try_candidates(
            cands: list[LineFit],
            profile_fn: Callable[..., ProfileResult],
            profile_kwargs: dict[str, Any],
    ) -> tuple[LineFit | None, ProfileResult | None]:
        for cand in cands[:3]:
            # p_res = compute_normal_profile(
            #     img_14bit,
            #     cand,
            #     half_length=pr.half_length,
            #     num_samples=pr.samples,
            #     smooth_window=pr.smooth_window,
            #     smooth_polyorder=pr.smooth_polyorder,
            #     prefilter_sigma=pr.prefilter_sigma,
            #     prefilter_gaussian_ksize=pr.prefilter_gaussian_ksize,
            #     peak_search_half=pr.peak_search_half,
            #     base_offset=pr.base_offset,
            #     side_peak_ratio=fw.side_peak_ratio,
            # )
            p_res = profile_fn(
                img_14bit,
                cand,
                half_length=pr.half_length,
                num_samples=pr.samples,
                smooth_window=pr.smooth_window,
                smooth_polyorder=pr.smooth_polyorder,
                prefilter_sigma=pr.prefilter_sigma,
                prefilter_gaussian_ksize=pr.prefilter_gaussian_ksize,
                peak_search_half=pr.peak_search_half,
                base_offset=pr.base_offset,
                side_peak_ratio=fw.side_peak_ratio,
                **profile_kwargs,
            )
            if p_res.ok:
                return cand, p_res
        if cands:
            # fallback = compute_normal_profile(
            #     img_14bit,
            #     cands[0],
            #     half_length=pr.half_length,
            #     num_samples=pr.samples,
            #     smooth_window=pr.smooth_window,
            #     smooth_polyorder=pr.smooth_polyorder,
            #     prefilter_sigma=pr.prefilter_sigma,
            #     prefilter_gaussian_ksize=pr.prefilter_gaussian_ksize,
            #     peak_search_half=pr.peak_search_half,
            #     base_offset=pr.base_offset,
            #     side_peak_ratio=fw.side_peak_ratio,
            # )
            fallback = profile_fn(
                img_14bit,
                cands[0],
                half_length=pr.half_length,
                num_samples=pr.samples,
                smooth_window=pr.smooth_window,
                smooth_polyorder=pr.smooth_polyorder,
                prefilter_sigma=pr.prefilter_sigma,
                prefilter_gaussian_ksize=pr.prefilter_gaussian_ksize,
                peak_search_half=pr.peak_search_half,
                base_offset=pr.base_offset,
                side_peak_ratio=fw.side_peak_ratio,
                **profile_kwargs,
            )
            return cands[0], fallback
        return None, None

    method = (pr.profile_method or "parallel").strip().lower()
    if method == "multi":
        profile_fn = compute_multi_normal_profile
        profile_kwargs: dict[str, Any] = {
            "multi_profile_half_width": pr.multi_profile_half_width,
            "multi_profile_step_scale": pr.multi_profile_step_scale,
        }
    else:
        profile_fn = compute_parallel_smoothed_profile
        profile_kwargs = {
            "parallel_avg_half_width": pr.parallel_avg_half_width,
            "parallel_avg_step_scale": pr.parallel_avg_step_scale,
            "monotone_fit": pr.monotone_fit,
            "monotone_method": pr.monotone_method,
        }

    line1, profile1 = _try_candidates(horiz_cands, profile_fn, profile_kwargs)
    line2, profile2 = _try_candidates(vert_cands, profile_fn, profile_kwargs)

    x1 = profile1.fwhm if profile1 is not None and profile1.ok else float("nan")
    x2 = profile2.fwhm if profile2 is not None and profile2.ok else float("nan")

    reasons: list[str] = []
    if line1 is None:
        reasons.append("L1_fit_failed")
    elif profile1 is not None and not profile1.ok:
        reasons.append(f"L1_fwhm_failed:{profile1.reason}")
    if line2 is None:
        reasons.append("L2_fit_failed")
    elif profile2 is not None and not profile2.ok:
        reasons.append(f"L2_fwhm_failed:{profile2.reason}")

    ok = np.isfinite(x1) and np.isfinite(x2)
    score = float(np.sqrt(x1 * x1 + x2 * x2)) if ok else float("nan")

    return ScoreResult(
        ok=ok,
        score=score,
        x1=float(x1),
        x2=float(x2),
        reason=";".join(reasons),
        spot=spot,
        line1=line1,
        line2=line2,
        profile1=profile1,
        profile2=profile2,
    )
