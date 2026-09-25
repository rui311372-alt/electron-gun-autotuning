"""直接用 mask 的上/右边拟合 L1/L2。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LineFit:
    p0: np.ndarray
    direction: np.ndarray
    endpoints: tuple[np.ndarray, np.ndarray]  # (p1, p2)
    length: float
    outward_normal: np.ndarray
    angle_deg: float

    @property
    def midpoint(self) -> np.ndarray:
        p1, p2 = self.endpoints
        return (p1 + p2) / 2.0

    def normal(self) -> np.ndarray:
        return self.outward_normal


def _fit_line_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 8:
        return None
    center = points.mean(axis=0)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    n = float(np.linalg.norm(direction))
    if n < 1e-9:
        return None
    return center, direction / n


def _project_endpoints(points: np.ndarray, center: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    t = (points - center) @ direction
    t0, t1 = float(t.min()), float(t.max())
    p0 = center + t0 * direction
    p1 = center + t1 * direction
    return p0, p1, abs(t1 - t0)


def _middle_segment(p0: np.ndarray, p1: np.ndarray, keep_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    mid = 0.5 * (p0 + p1)
    d = p1 - p0
    half = 0.5 * float(np.clip(keep_ratio, 0.2, 1.0))
    return mid - half * d, mid + half * d


def _angle_deg(direction: np.ndarray) -> float:
    dx, dy = float(direction[0]), float(direction[1])
    if abs(dx) < 1e-9:
        return 90.0
    return float(np.degrees(np.arctan2(abs(dy), abs(dx))))


def _unit_normal_to_segment(seg_dir: np.ndarray) -> np.ndarray:
    """与线段方向 seg_dir 垂直的单位法向量 (-dy, dx)。"""
    n = np.array([-float(seg_dir[1]), float(seg_dir[0])], dtype=float)
    ln = float(np.linalg.norm(n))
    if ln < 1e-12:
        return np.array([0.0, 1.0], dtype=float)
    return n / ln


def _sign_normal_by_axis_rule(
    n0: np.ndarray,
    line_id: str,
    normal_hint: np.ndarray,
) -> np.ndarray:
    """按几何规定选 n 的符号 (与 seg 垂直, 不用灰度).

    剖面 sample = base + t*n, t 从负到正.
    L1: t 最小端 y 最大, t 最大端 y 最小 -> n_y < 0.
    L2: t 最小端 x 最小, t 最大端 x 最大 -> n_x > 0.
    退化时退回与 normal_hint 同半空间.
    """
    h = normal_hint / (np.linalg.norm(normal_hint) + 1e-12)
    if line_id == "l1":
        if abs(float(n0[1])) > 1e-9:
            return (n0 if float(n0[1]) < 0.0 else -n0).astype(float)
    elif line_id == "l2":
        if abs(float(n0[0])) > 1e-9:
            return (n0 if float(n0[0]) > 0.0 else -n0).astype(float)
    return (n0 if float(np.dot(n0, h)) >= 0.0 else -n0).astype(float)


def _line_x_at_y(line: LineFit, y: float) -> float | None:
    """L2 所在无限直线与水平线 y=常数的交点横坐标。L2 近似竖直时 dy 非零。"""
    p0 = line.p0
    d = line.direction
    dy = float(d[1])
    if abs(dy) < 1e-9:
        return None
    t = (float(y) - float(p0[1])) / dy
    return float(p0[0] + t * float(d[0]))


def _extract_top_points(
    mask: np.ndarray,
    x_max_cap: int | None,
    edge_shrink_ratio: float,
) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=float)
    x0, x1 = int(xs.min()), int(xs.max())
    if x_max_cap is not None:
        cap = int(round(float(x_max_cap)))
        if cap >= x0 + 8:
            x1 = min(x1, cap)
    xs = np.arange(x0, x1 + 1)
    top_y = np.full(xs.shape, np.nan, dtype=float)
    for i, x in enumerate(xs):
        y_hits = np.where(mask[:, x])[0]
        if len(y_hits) == 0:
            continue
        top_y[i] = float(y_hits.min())

    finite = np.isfinite(top_y)
    idx = np.where(finite)[0]
    if len(idx) < 8:
        return np.empty((0, 2), dtype=float)

    # 整段上边：仅在两端缩小一点，避开连接处
    shrink = max(int(edge_shrink_ratio * len(idx)), 3)
    if len(idx) > 2 * shrink + 6:
        idx = idx[shrink:-shrink]
    if len(idx) < 8:
        return np.empty((0, 2), dtype=float)
    pts = np.stack([xs[idx].astype(float), top_y[idx]], axis=1)
    return pts


def _extract_right_points(mask: np.ndarray, edge_shrink_ratio: float) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=float)
    y0, y1 = int(ys.min()), int(ys.max())
    ys = np.arange(y0, y1 + 1)
    right_x = np.full(ys.shape, np.nan, dtype=float)
    for i, y in enumerate(ys):
        x_hits = np.where(mask[y, :])[0]
        if len(x_hits) == 0:
            continue
        right_x[i] = float(x_hits.max())

    finite = np.isfinite(right_x)
    idx = np.where(finite)[0]
    if len(idx) < 8:
        return np.empty((0, 2), dtype=float)

    # 整段右边：仅在两端缩小一点，避开连接处
    shrink = max(int(edge_shrink_ratio * len(idx)), 3)
    if len(idx) > 2 * shrink + 6:
        idx = idx[shrink:-shrink]
    if len(idx) < 8:
        return np.empty((0, 2), dtype=float)
    pts = np.stack([right_x[idx], ys[idx].astype(float)], axis=1)
    return pts


def _build_line(
    points: np.ndarray,
    line_id: str,
    normal_hint: np.ndarray,
    line_keep_ratio: float,
) -> LineFit | None:
    fit = _fit_line_from_points(points)
    if fit is None:
        return None
    center, direction = fit
    p0, p1, full_len = _project_endpoints(points, center, direction)
    if full_len < 8.0:
        return None
    q0, q1 = _middle_segment(p0, p1, line_keep_ratio)
    seg = q1 - q0
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-9:
        return None
    seg_dir = seg / seg_len
    if np.dot(seg_dir, direction) < 0:
        seg_dir = -seg_dir
    n0 = _unit_normal_to_segment(seg_dir)
    mid = 0.5 * (q0 + q1)
    n = _sign_normal_by_axis_rule(n0, line_id, normal_hint)
    return LineFit(
        p0=mid,
        direction=seg_dir,
        endpoints=(q0, q1),
        length=seg_len,
        outward_normal=n,
        angle_deg=_angle_deg(seg_dir),
    )


def get_line_candidates(
    img_14bit: np.ndarray,  # keep signature for compatibility
    mask: np.ndarray,
    *,
    min_seg_len: float = 30.0,
    black_delta: float = 0.0,
    center_size: int = 20,
    black_band_width: int = 20,
    end_length: int = 50,
    stop_before_mask: int = 50,
    grad_min: float = 80.0,
    angle_half_range_deg: float = 25.0,
    angle_step_deg: float = 2.0,
    edge_shrink_ratio: float = 0.10,
    line_keep_ratio: float = 0.75,
) -> tuple[list[LineFit], list[LineFit]]:
    # black_delta / center_* / end_* / stop_* / grad_min / angle_*：与 JSON 对齐保留，供后续扩展；当前未使用。
    del img_14bit, black_delta, center_size, black_band_width, end_length, stop_before_mask
    del grad_min, angle_half_range_deg, angle_step_deg

    esr = float(np.clip(edge_shrink_ratio, 0.0, 0.45))
    lkr = float(np.clip(line_keep_ratio, 0.2, 1.0))

    right_pts = _extract_right_points(mask, esr)
    l2 = _build_line(
        right_pts,
        "l2",
        normal_hint=np.array([1.0, 0.0], dtype=float),
        line_keep_ratio=lkr,
    )

    x_max_cap: int | None = None
    if l2 is not None:
        ys_m, xs_m = np.where(mask)
        y_top_bbox = int(ys_m.min())
        x_int = _line_x_at_y(l2, float(y_top_bbox))
        if x_int is not None and np.isfinite(x_int):
            x0_m = int(xs_m.min())
            x1_m = int(xs_m.max())
            cap = int(round(x_int))
            if cap >= x0_m + 8:
                x_max_cap = min(x1_m, cap)

    top_pts = _extract_top_points(mask, x_max_cap, esr)
    l1 = _build_line(
        top_pts,
        "l1",
        normal_hint=np.array([0.0, -1.0], dtype=float),
        line_keep_ratio=lkr,
    )

    l1_list = [l1] if (l1 is not None and l1.length >= min_seg_len) else []
    l2_list = [l2] if (l2 is not None and l2.length >= min_seg_len) else []
    return l1_list, l2_list


def fit_two_lines(
    img_14bit: np.ndarray,
    mask: np.ndarray,
    *,
    min_seg_len: float = 30.0,
) -> tuple[LineFit | None, LineFit | None]:
    horiz, vert = get_line_candidates(img_14bit, mask, min_seg_len=min_seg_len)
    return (horiz[0] if horiz else None, vert[0] if vert else None)
