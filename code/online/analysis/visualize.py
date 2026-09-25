"""叠加可视化: 底图 + mask 描边 + L1/L2 + 法向采样路径 + 剖面/微分/FWHM 子图。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .fit_lines import LineFit
from .io_tif import stretch_to_uint8
from .profile import ProfileResult
from .score import ScoreResult


def _draw_line_fullspan(
    ax, line: LineFit, color: str, label: str, img_shape: tuple[int, int]
) -> None:
    """画拟合线: 粗线段为实际多边形边, 细虚线为它的整段延长。"""
    h, w = img_shape
    p0 = line.p0
    d = line.direction
    if abs(d[0]) > abs(d[1]):
        xs = np.array([0, w - 1], dtype=float)
        ts = (xs - p0[0]) / d[0]
        ys = p0[1] + ts * d[1]
    else:
        ys = np.array([0, h - 1], dtype=float)
        ts = (ys - p0[1]) / d[1]
        xs = p0[0] + ts * d[0]
    ax.plot(xs, ys, color=color, linewidth=0.8, linestyle="--", alpha=0.6)
    p1, p2 = line.endpoints
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.0, label=label)


def _draw_sample_path(ax, profile: ProfileResult, color: str) -> None:
    xy = profile.sample_xy
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.0, alpha=0.8)
    bp = profile.base_point
    ax.plot([bp[0]], [bp[1]], marker="o", color=color, markersize=4)


def _plot_profile_subplot(ax, profile: ProfileResult, title: str) -> None:
    if profile is None:
        ax.set_title(title + " (n/a)")
        ax.axis("off")
        return

    label = "y=f(x) (monotone fit)" if profile.y_monotone else "y=f(x)"
    ax.plot(profile.t, profile.y, color="tab:blue", label=label)
    ax.set_ylabel("pixel value", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_xlabel("position along normal [px]")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(profile.t, profile.dy, color="tab:red", label="dy/dt")
    ax2.set_ylabel("dy/dt", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.invert_yaxis()

    if profile.ok:
        half_dy = -0.5 * profile.peak_value
        ax2.axhline(half_dy, color="tab:red", linestyle="--", alpha=0.5)
        ax2.axvline(profile.t_left, color="tab:red", linestyle=":", alpha=0.7)
        ax2.axvline(profile.t_right, color="tab:red", linestyle=":", alpha=0.7)
        ax.set_title(f"{title}  FWHM = {profile.fwhm:.3f} px")
    else:
        ax.set_title(f"{title}  FAIL ({profile.reason})")


def make_debug_figure(
    img_14bit: np.ndarray,
    result: ScoreResult,
    title: str = "",
) -> Figure:
    """生成调试总图: 左侧叠图,右侧两个剖面/微分子图。"""
    bg = stretch_to_uint8(img_14bit)
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(bg_rgb)
    ax_img.set_title(title or "overlay")
    ax_img.set_xlabel("x [px]")
    ax_img.set_ylabel("y [px]")

    if result.spot is not None:
        mask_u8 = (result.spot.mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            cnt = cnt.squeeze()
            if cnt.ndim != 2 or len(cnt) < 2:
                continue
            ax_img.plot(cnt[:, 0], cnt[:, 1], color="lime", linewidth=1.0)
        x, y, w, h = result.spot.bbox
        ax_img.add_patch(
            Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=0.8, alpha=0.6)
        )
        cx, cy = result.spot.centroid
        ax_img.plot([cx], [cy], marker="+", color="lime", markersize=10)

    if result.line1 is not None:
        _draw_line_fullspan(
            ax_img, result.line1, "deepskyblue", "L1 (leftward)", img_14bit.shape
        )
    if result.line2 is not None:
        _draw_line_fullspan(
            ax_img, result.line2, "orange", "L2 (downward)", img_14bit.shape
        )

    if result.profile1 is not None:
        _draw_sample_path(ax_img, result.profile1, "deepskyblue")
    if result.profile2 is not None:
        _draw_sample_path(ax_img, result.profile2, "orange")

    handles, labels = ax_img.get_legend_handles_labels()
    if handles:
        ax_img.legend(loc="upper right", fontsize=8)
    info = (
        f"score = {result.score:.3f}    "
        f"x1 = {result.x1:.3f}    x2 = {result.x2:.3f}    "
        f"ok = {result.ok}"
    )
    if result.reason:
        info += f"\nreason: {result.reason}"
    ax_img.text(
        0.01,
        0.99,
        info,
        transform=ax_img.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.5, pad=4),
    )

    ax_p1 = fig.add_subplot(gs[0, 1])
    _plot_profile_subplot(ax_p1, result.profile1, "L1 normal profile")
    ax_p2 = fig.add_subplot(gs[1, 1])
    _plot_profile_subplot(ax_p2, result.profile2, "L2 normal profile")

    fig.tight_layout()
    return fig


def save_debug_figure(
    img_14bit: np.ndarray,
    result: ScoreResult,
    out_path: str | Path,
    title: str = "",
    dpi: int = 110,
) -> None:
    """渲染并保存调试图到文件, 不弹窗。

    先写入同目录临时文件再 ``os.replace`` 覆盖目标路径,避免 Windows 下直接覆盖已存在
    PNG 时偶发未更新或句柄占用导致的问题。
    """
    path = Path(out_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = make_debug_figure(img_14bit, result, title=title)
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            suffix=path.suffix if path.suffix else ".png",
            dir=str(path.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight")
        finally:
            plt.close(fig)
        os.replace(tmp_path, path)
        tmp_path = None
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
