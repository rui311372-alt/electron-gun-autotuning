"""沿直线法向亚像素采样 + Savitzky-Golay 平滑求导 + FWHM。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.special import erf

from .fit_lines import LineFit


@dataclass
class ProfileResult:
    """一条法向剖面分析结果。"""

    base_point: np.ndarray  # (x, y) 法线基点 (像素)
    normal: np.ndarray  # 单位法向量
    t: np.ndarray  # 沿法向的位置坐标 (像素, 0 在 base_point)
    y: np.ndarray  # 该位置处的像素值 (双线性插值, 14-bit float)
    y_monotone: bool  # y 是否经过单调拟合
    dy: np.ndarray  # dy/dt, SG 求导结果
    abs_dy: np.ndarray  # |dy/dt|
    sample_xy: np.ndarray  # (K, 2), 采样点在图像中的 (x, y) 坐标
    peak_idx: int
    peak_value: float  # 负微分主峰深度 P = -dy[peak] (>0), 与旧 |dy| 峰值同量纲
    t_left: float  # 半高左交点 (亚像素)
    t_right: float  # 半高右交点 (亚像素)
    fwhm: float  # = t_right - t_left, 单位: 像素
    ok: bool
    reason: str = ""


def interpolate_along_normal(
    img_14bit: np.ndarray,
    base_point: np.ndarray,
    normal: np.ndarray,
    t: np.ndarray,
    *,
    prefilter_sigma: float = 1.5,
    prefilter_gaussian_ksize: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """沿单位法线在距离 ``t`` (像素) 处双线性采样灰度；预处理与 :func:`compute_normal_profile` 一致。
    高斯核边长 ``prefilter_gaussian_ksize``（奇数）, ``σ = prefilter_sigma``;
    ``prefilter_sigma <= 0`` 时不做高斯。

    返回 ``(t, sample_x, sample_y, values)``，均为 float 数组。
    """
    base_point = np.asarray(base_point, dtype=float)
    normal = np.asarray(normal, dtype=float)
    t = np.asarray(t, dtype=float)
    sample_x = base_point[0] + t * normal[0]
    sample_y = base_point[1] + t * normal[1]

    img_float = img_14bit.astype(np.float64)
    if prefilter_sigma > 0:
        k = int(prefilter_gaussian_ksize)
        if k % 2 == 0:
            k += 1
        k = max(k, 1)
        img_float = cv2.GaussianBlur(img_float, (k, k), prefilter_sigma)

    coords = np.vstack([sample_y, sample_x])
    values = map_coordinates(img_float, coords, order=1, mode="nearest")
    return t, sample_x, sample_y, values


def _project_point_to_line(line: LineFit, point: np.ndarray) -> np.ndarray:
    """把 point 投影到 line 上,返回投影点 (x, y)。"""
    diff = point - line.p0
    t = float(diff @ line.direction)
    return line.p0 + t * line.direction


def _interp_crossing(
    t: np.ndarray, y: np.ndarray, idx_a: int, idx_b: int, level: float
) -> float:
    """在 (t[idx_a], y[idx_a]) 和 (t[idx_b], y[idx_b]) 之间线性插值, 求 y == level 的 t。"""
    ta, tb = t[idx_a], t[idx_b]
    ya, yb = y[idx_a], y[idx_b]
    if abs(yb - ya) < 1e-12:
        return float(ta)
    return float(ta + (level - ya) * (tb - ta) / (yb - ya))

def _erf_model(t: np.ndarray, a: float, b: float, t0: float, w: float) -> np.ndarray:
    """误差函数边缘模型：a + b * erf((t - t0) / w)。

    ``w > 0`` 时函数严格单调；``b < 0`` 对应灰度沿 +t 方向减小（亮→阴影）。
    """
    return a + b * erf((t - t0) / w)


def _fit_erf(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
    """用 Erf 模型拟合灰度剖面，返回 (拟合后的 y, 拟合参数)。

    参数顺序 (a, b, t0, w)，失败时返回 (y, None)。
    """
    y_min, y_max = float(np.min(y)), float(np.max(y))
    a0 = (y_min + y_max) / 2.0
    b0 = (y_max - y_min) / 2.0
    if y[-1] < y[0]:
        b0 = -abs(b0)
    else:
        b0 = abs(b0)
    t0_0 = float(t[len(t) // 2])
    w0 = max(float(t[-1] - t[0]) / 10.0, 1e-3)

    bounds = (
        [y_min - abs(b0), -2.0 * abs(b0), t[0], 1e-3],
        [y_max + abs(b0), 2.0 * abs(b0), t[-1], t[-1] - t[0]],
    )

    try:
        popt, _ = curve_fit(
            _erf_model,
            t,
            y,
            p0=[a0, b0, t0_0, w0],
            bounds=bounds,
            maxfev=5000,
        )
        return _erf_model(t, *popt), tuple(float(v) for v in popt)
    except Exception:
        return y, None


def _erf_derivative(t: np.ndarray, a: float, b: float, t0: float, w: float) -> np.ndarray:
    """Erf 模型 y(t) = a + b * erf((t - t0) / w) 的解析一阶导数。"""
    return (2.0 * b / (w * np.sqrt(np.pi))) * np.exp(-((t - t0) / w) ** 2)

# def _find_fwhm(
#     t: np.ndarray,
#     dy: np.ndarray,
#     peak_search_half: float | None = None,
#     *,
#     side_peak_ratio: float = 0.95,
# ) -> tuple[int, float, float, float, bool, str]:
#     """在带符号 ``dy/dt`` 上找 **负微分** 主峰并量 FWHM (亮斑→阴影, 灰度沿法线减小).
#
#     仅在 ``dy < 0`` 的点中选主波峰 (``dy`` 最负). ``dy >= 0`` 的峰不参与.
#     主峰深度 ``P = -dy[peak] > 0``; 半高取 ``dy = -P/2``, 向两侧插值得 ``t_left``, ``t_right``.
#     返回 ``peak_val`` 为 ``P`` (正数, 与旧版 ``|dy|`` 峰值同量纲便于作图).
#     """
#     n = len(dy)
#     if peak_search_half is not None:
#         search_mask = np.abs(t) <= peak_search_half
#     else:
#         search_mask = np.ones(n, dtype=bool)
#
#     neg_mask = search_mask & (dy < 0)
#     if not neg_mask.any():
#         return 0, 0.0, np.nan, np.nan, False, "no_negative_dy_in_search"
#
#     neg_indices = np.where(neg_mask)[0]
#     local_j = int(np.argmin(dy[neg_mask]))
#     peak_idx = int(neg_indices[local_j])
#
#     peak_val = float(-dy[peak_idx])
#     if peak_val <= 0:
#         return peak_idx, peak_val, np.nan, np.nan, False, "peak<=0"
#
#     half_dy = -0.5 * peak_val
#
#     left_idx = peak_idx
#     while left_idx > 0 and dy[left_idx] <= half_dy:
#         left_idx -= 1
#     if dy[left_idx] <= half_dy:
#         return peak_idx, peak_val, np.nan, np.nan, False, "no_left_crossing"
#     t_left = _interp_crossing(t, dy, left_idx, left_idx + 1, half_dy)
#
#     right_idx = peak_idx
#     while right_idx < n - 1 and dy[right_idx] <= half_dy:
#         right_idx += 1
#     if dy[right_idx] <= half_dy:
#         return peak_idx, peak_val, np.nan, np.nan, False, "no_right_crossing"
#     t_right = _interp_crossing(t, dy, right_idx - 1, right_idx, half_dy)
#
#     if t_right <= t_left:
#         return peak_idx, peak_val, t_left, t_right, False, "right<=left"
#
#     if peak_search_half is not None:
#         in_search = np.abs(t) <= peak_search_half
#         fwhm_width_idx = max(right_idx - left_idx, 1)
#         side_exclude = max(int(2.0 * fwhm_width_idx), 5)
#         side_mask = in_search.copy()
#         lo = max(peak_idx - side_exclude, 0)
#         hi = min(peak_idx + side_exclude + 1, n)
#         side_mask[lo:hi] = False
#         if side_mask.any():
#             side_min_dy = float(np.min(dy[side_mask]))
#             thr = -float(side_peak_ratio) * peak_val
#             if side_min_dy < thr:
#                 return (
#                     peak_idx,
#                     peak_val,
#                     t_left,
#                     t_right,
#                     False,
#                     f"side_peak_too_strong({(-side_min_dy) / peak_val:.2f})",
#                 )
#
#     return peak_idx, peak_val, t_left, t_right, True, ""

def _find_fwhm(
    t: np.ndarray,
    dy: np.ndarray,
    peak_search_half: float | None = None,
    *,
    side_peak_ratio: float = 0.95,
) -> tuple[int, float, float, float, bool, str]:
    """v2: 在 ``dy/dt`` 上选负微分主峰并量 FWHM，兼顾峰深度与到法线基点的距离。

    背景
    ----
    旧版 ``_find_fwhm`` 直接取 ``dy < 0`` 的全局最负点。当剖面在阴影区有多个深度
    相近的谷时，它会在不同谷之间跳动，导致重复性差。

    选峰策略
    --------
    1. 在 ``|t| <= peak_search_half`` 内找出所有 ``dy < 0`` 的局部极小值；
    2. 对每个局部极小值按 ``score = depth * exp(-|t| / tau)`` 打分：
       - ``depth = -dy[idx]``：负微分峰深度，越深分越高；
       - ``|t|``：到法线基点（拟合线段中点沿外法向偏移后的点）的距离，
         越靠近基点惩罚越小；
       - ``tau = peak_search_half / 3``：距离衰减尺度。
    3. 选最高分者为主峰。这样可在多个候选谷中稳定地选择同一条边缘。

    FWHM 计算
    ---------
    主峰深度 ``P = -dy[peak] > 0``，取半高 ``dy = -P/2``，向两侧线性插值得到
    ``t_left``、``t_right``，FWHM = ``t_right - t_left``。

    旁峰检查
    --------
    只检查**主峰外侧（阴影侧）**、且距离主峰不太远的范围内是否存在更强的负峰：
    - 排除主峰附近 ``2.5 * fwhm_width_idx`` 像素内的点；
    - 旁峰检查区限制在 ``|t| <= |t_peak| + 3 * fwhm_width``；
    - 避免阴影区远处的振荡峰被误判为强旁峰。
    若旁峰深度 ``> side_peak_ratio * 主峰深度``，则返回失败
    ``side_peak_too_strong(ratio)``。

    参数
    ----
    t : np.ndarray
        沿法线的位置坐标（像素），0 在法线基点。
    dy : np.ndarray
        沿法线的灰度一阶导数（SG 平滑后）。
    peak_search_half : float | None
        主峰搜索半范围。None 时表示搜索整条剖面。
    side_peak_ratio : float
        旁峰判定阈值，0~1。1.0 表示禁用旁峰检查。

    返回
    ----
    (peak_idx, peak_val, t_left, t_right, ok, reason)
        与旧版 ``_find_fwhm`` 保持兼容。
    """
    n = len(dy)
    if peak_search_half is not None:
        search_mask = np.abs(t) <= peak_search_half
    else:
        search_mask = np.ones(n, dtype=bool)

    neg_mask = search_mask & (dy < 0)
    if not neg_mask.any():
        return 0, 0.0, np.nan, np.nan, False, "no_negative_dy_in_search"

    # 搜索区内所有 dy < 0 的局部极小值
    local_minima: list[int] = []
    neg_indices = np.where(neg_mask)[0]
    for idx in neg_indices:
        if idx == 0 or idx == n - 1:
            continue
        if dy[idx] < dy[idx - 1] and dy[idx] <= dy[idx + 1]:
            local_minima.append(idx)

    if not local_minima:
        # 没有严格局部极小：退化为最深的负点
        local_minima = [int(neg_indices[int(np.argmin(dy[neg_mask]))])]

    # 用深度 * exp(-|t|/tau) 打分选主峰
    # tau 取 peak_search_half/3，让 |t| 接近 peak_search_half 时分数衰减到约 5%
    tau = peak_search_half / 3.0 if peak_search_half is not None else (np.max(np.abs(t)) / 3.0)
    best_score = -np.inf
    peak_idx = local_minima[0]
    for idx in local_minima:
        depth = -dy[idx]
        score = depth * np.exp(-abs(t[idx]) / tau)
        if score > best_score:
            best_score = score
            peak_idx = idx

    peak_val = float(-dy[peak_idx])
    if peak_val <= 0:
        return peak_idx, peak_val, np.nan, np.nan, False, "peak<=0"

    half_dy = -0.5 * peak_val

    left_idx = peak_idx
    while left_idx > 0 and dy[left_idx] <= half_dy:
        left_idx -= 1
    if dy[left_idx] <= half_dy:
        return peak_idx, peak_val, np.nan, np.nan, False, "no_left_crossing"
    t_left = _interp_crossing(t, dy, left_idx, left_idx + 1, half_dy)

    right_idx = peak_idx
    while right_idx < n - 1 and dy[right_idx] <= half_dy:
        right_idx += 1
    if dy[right_idx] <= half_dy:
        return peak_idx, peak_val, np.nan, np.nan, False, "no_right_crossing"
    t_right = _interp_crossing(t, dy, right_idx - 1, right_idx, half_dy)

    if t_right <= t_left:
        return peak_idx, peak_val, t_left, t_right, False, "right<=left"

    if peak_search_half is not None:
        fwhm_width_idx = max(right_idx - left_idx, 1)
        # 旁峰排除区：主峰附近 2.5 倍 FWHM 像素宽度
        side_exclude = max(int(1.5 * fwhm_width_idx), 5)
        # 旁峰搜索最远限制：|t| 不超过 |t_peak| + 3 倍 FWHM 像素对应的范围
        margin_idx = max(int(2.0 * fwhm_width_idx), 10)
        far_t_limit = abs(t[peak_idx]) + abs(t[min(peak_idx + margin_idx, n - 1)] - t[peak_idx])
        far_t_limit = max(far_t_limit, abs(t[peak_idx]) + 15.0)

        lo = max(peak_idx - side_exclude, 0)
        hi = min(peak_idx + side_exclude + 1, n)
        side_mask = np.zeros(n, dtype=bool)
        side_mask[lo:hi] = True
        side_mask ^= True  # 取反，得到旁峰检查区
        side_mask &= np.abs(t) <= far_t_limit
        # 只检查主峰外侧（阴影侧）：|t| 比主峰更大的那一侧
        if t[peak_idx] >= 0:
            side_mask &= t >= t[peak_idx]
        else:
            side_mask &= t <= t[peak_idx]

        if side_mask.any():
            side_min_dy = float(np.min(dy[side_mask]))
            thr = -float(side_peak_ratio) * peak_val
            if side_min_dy < thr:
                return (
                    peak_idx,
                    peak_val,
                    t_left,
                    t_right,
                    False,
                    f"side_peak_too_strong({(-side_min_dy) / peak_val:.2f})",
                )

    return peak_idx, peak_val, t_left, t_right, True, ""

def _fit_monotone(t: np.ndarray, y: np.ndarray, *, method: str = "pchip") -> np.ndarray:
    """对灰度剖面 ``y(t)`` 做单调拟合，返回拟合后的 ``y_fit``。

    - ``pchip``：保形分段三次 Hermite 插值，保持数据单调性，通过采样点。
    - ``erf``：用误差函数 ``a + b * erf((t - t0) / w)`` 做严格单调的非线性最小二乘拟合，
      一阶导数为单峰高斯形，适合亮斑→阴影的单一过渡边缘。
    """
    method = (method or "pchip").strip().lower()
    if method == "pchip":
        return PchipInterpolator(t, y)(t)

    if method == "erf":
        return _fit_erf(t, y)[0]

    raise ValueError(f"不支持的单调拟合方法: {method!r}")


def _fit_monotone_with_params(
    t: np.ndarray, y: np.ndarray, *, method: str = "pchip"
) -> tuple[np.ndarray, tuple[float, ...] | None]:
    """同 :func:`_fit_monotone`，但额外返回拟合参数（如可用）。"""
    method = (method or "pchip").strip().lower()
    if method == "pchip":
        return PchipInterpolator(t, y)(t), None
    if method == "erf":
        return _fit_erf(t, y)
    raise ValueError(f"不支持的单调拟合方法: {method!r}")

def interpolate_along_segments(
    img_14bit: np.ndarray,
    centers: np.ndarray,
    direction: np.ndarray,
    s: np.ndarray,
    *,
    prefilter_sigma: float = 1.5,
    prefilter_gaussian_ksize: int = 9,
) -> np.ndarray:
    """在 ``centers`` 上沿同一方向 ``direction`` 的线段采样，并取平均。

    Parameters
    ----------
    img_14bit
        输入 14-bit 灰度图。
    centers
        (N, 2) 线段中心点坐标。
    direction
        (2,) 单位方向向量。
    s
        (M,) 沿 direction 的偏移坐标（像素），0 为中心。
    prefilter_sigma, prefilter_gaussian_ksize
        与 :func:`interpolate_along_normal` 一致的高斯预处理参数。

    Returns
    -------
    values : np.ndarray
        形状 (N,)，第 i 个元素是 centers[i] 处线段上 M 个采样点的平均灰度。
    """
    direction = np.asarray(direction, dtype=float)
    s = np.asarray(s, dtype=float)
    centers = np.asarray(centers, dtype=float)
    sample_x = centers[:, 0][:, None] + s[None, :] * direction[0]
    sample_y = centers[:, 1][:, None] + s[None, :] * direction[1]

    img_float = img_14bit.astype(np.float64)
    if prefilter_sigma > 0:
        k = int(prefilter_gaussian_ksize)
        if k % 2 == 0:
            k += 1
        k = max(k, 1)
        img_float = cv2.GaussianBlur(img_float, (k, k), prefilter_sigma)

    coords = np.stack([sample_y.ravel(), sample_x.ravel()], axis=0)
    vals = map_coordinates(img_float, coords, order=1, mode="nearest")
    vals = vals.reshape(sample_x.shape)
    return vals.mean(axis=1)

def compute_parallel_smoothed_profile(
    img_14bit: np.ndarray,
    line: LineFit,
    base_point: np.ndarray | None = None,
    base_offset: float = 2.0,
    half_length: float = 50.0,
    num_samples: int = 301,
    smooth_window: int = 21,
    smooth_polyorder: int = 3,
    prefilter_sigma: float = 1.5,
    prefilter_gaussian_ksize: int = 9,
    peak_search_half: float | None = 5.0,
    side_peak_ratio: float = 0.95,
    parallel_avg_half_width: float = 4.0,
    parallel_avg_step_scale: float = 1.0,
    monotone_fit: bool = False,
    monotone_method: str = "pchip",
) -> ProfileResult:
    """沿 `line` 的法线方向做亚像素剖面，法线上每一点用平行于 line 的短线段灰度平均作为代表值，再求 FWHM。

    与 :func:`compute_normal_profile` 的区别：
    - 后者直接取法线上单点的灰度；
    - 本函数在法线上每个位置先沿 ``line.direction`` 方向做一条短线段，取线段上灰度平均，
      用该平均值构成法向剖面，再做 SG 求导和 FWHM。
    这样可以抑制沿边缘方向的噪声和局部毛刺，让主峰更稳定。

    Parameters
    ----------
    img_14bit
        14-bit 灰度图 (H, W), uint16 / float 均可。
    line
        已拟合的直线。
    base_point
        法线基点 (x, y)。默认: 取拟合线段的中点。
    half_length
        法线半长度 (像素), 采样区间为 ``[-half_length, +half_length]``。
    num_samples
        法线方向采样点数 (建议奇数, base_point 落在正中)。
    smooth_window, smooth_polyorder
        Savitzky-Golay 滤波参数。`smooth_window` 必须是奇数。
    prefilter_sigma
        剖面前整图高斯平滑 σ; ``<=0`` 时不做高斯。核边长见 ``prefilter_gaussian_ksize``。
    peak_search_half
        主峰搜索半范围，None 表示搜索整条剖面。
    side_peak_ratio
        主峰两侧搜索区外若存在过强的负微分旁峰，则判 FWHM 不可靠。
    parallel_avg_half_width
        平行方向平均半宽度（像素）。最终点数取 ``2 * n + 1``，n 由法线步长和本参数共同决定。
    parallel_avg_step_scale
        平行方向采样步长 = 法线方向步长 × 本系数。
    monotone_fit
        是否在 SG 求导前对平均后的灰度剖面做单调拟合。
    monotone_method
        单调拟合方法，当前仅支持 "pchip"。
    """
    if line is None:
        raise ValueError("line is None")

    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window <= smooth_polyorder:
        smooth_window = smooth_polyorder + 1 + ((smooth_polyorder + 1) % 2)

    normal = line.normal()
    if base_point is None:
        base_point = line.midpoint + base_offset * normal
    base_point = np.asarray(base_point, dtype=float)

    t = np.linspace(-half_length, half_length, num_samples)
    dt = float(t[1] - t[0])

    # 法线上每个位置的中心点
    centers = base_point[None, :] + t[:, None] * normal[None, :]

    # 平行方向采样坐标：以法线步长为基准
    p_step = dt * float(parallel_avg_step_scale)
    if p_step <= 0:
        p_step = dt if dt > 0 else 1.0
    n_parallel = max(int(round(parallel_avg_half_width / p_step)), 0)
    if n_parallel == 0:
        # 退化到原始单点采样
        s = np.array([0.0])
    else:
        s = np.arange(-n_parallel, n_parallel + 1, dtype=float) * p_step

    parallel_dir = line.direction
    y = interpolate_along_segments(
        img_14bit,
        centers,
        parallel_dir,
        s,
        prefilter_sigma=prefilter_sigma,
        prefilter_gaussian_ksize=prefilter_gaussian_ksize,
    )

    # 重建 sample_xy（仅保留法线中心点，用于调试可视化）
    sample_xy = centers.copy()

    if monotone_fit:
        y, erf_params = _fit_monotone_with_params(t, y, method=monotone_method)
        if monotone_method.strip().lower() == "erf" and erf_params is not None:
            # Erf 拟合：用解析导数，天然严格单峰，无需 SG 数值求导
            dy = _erf_derivative(t, *erf_params)
        else:
            dy = savgol_filter(
                y,
                window_length=smooth_window,
                polyorder=smooth_polyorder,
                deriv=1,
                delta=dt,
            )
    else:
        dy = savgol_filter(
            y, window_length=smooth_window, polyorder=smooth_polyorder, deriv=1, delta=dt
        )
    abs_dy = np.abs(dy)

    peak_idx, peak_val, t_left, t_right, ok, reason = _find_fwhm(
        t, dy, peak_search_half=peak_search_half, side_peak_ratio=side_peak_ratio
    )
    fwhm = (t_right - t_left) if ok else float("nan")

    return ProfileResult(
        base_point=base_point,
        normal=normal,
        t=t,
        y=y,
        y_monotone=monotone_fit,
        dy=dy,
        abs_dy=abs_dy,
        sample_xy=sample_xy,
        peak_idx=peak_idx,
        peak_value=peak_val,
        t_left=t_left,
        t_right=t_right,
        fwhm=fwhm,
        ok=ok,
        reason=reason,
    )

def compute_normal_profile(
    img_14bit: np.ndarray,
    line: LineFit,
    base_point: np.ndarray | None = None,
    base_offset: float = 2.0,
    half_length: float = 50.0,
    num_samples: int = 301,
    smooth_window: int = 21,
    smooth_polyorder: int = 3,
    prefilter_sigma: float = 1.5,
    prefilter_gaussian_ksize: int = 9,
    peak_search_half: float | None = 5.0,
    side_peak_ratio: float = 0.95,
) -> ProfileResult:
    """沿 `line` 的法线方向, 在 `base_point` 附近做亚像素剖面 + 微分 + FWHM。

    Parameters
    ----------
    img_14bit
        14-bit 灰度图 (H, W), uint16 / float 均可。
    line
        已拟合的直线。
    base_point
        法线基点 (x, y)。默认: 取拟合线段的中点。
    half_length
        法线半长度 (像素), 采样区间为 ``[-half_length, +half_length]``。
    num_samples
        采样点数 (建议奇数, base_point 落在正中)。
    smooth_window, smooth_polyorder
        Savitzky-Golay 滤波参数。`smooth_window` 必须是奇数。
    prefilter_sigma
        剖面前整图高斯平滑 σ; ``<=0`` 时不做高斯。核边长见 ``prefilter_gaussian_ksize``。
    side_peak_ratio
        主峰两侧搜索区外若存在过强的负微分旁峰（相对主峰深度），则判 FWHM 不可靠。
    FWHM
        仅在 ``dy/dt < 0`` 的主峰 (最陡下降) 上量半高宽; 正微分峰不参与。
    """
    if line is None:
        raise ValueError("line is None")

    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window <= smooth_polyorder:
        smooth_window = smooth_polyorder + 1 + ((smooth_polyorder + 1) % 2)

    normal = line.normal()
    if base_point is None:
        base_point = line.midpoint + base_offset * normal
    base_point = np.asarray(base_point, dtype=float)

    t = np.linspace(-half_length, half_length, num_samples)
    dt = float(t[1] - t[0])

    _, sample_x, sample_y, y = interpolate_along_normal(
        img_14bit,
        base_point,
        normal,
        t,
        prefilter_sigma=prefilter_sigma,
        prefilter_gaussian_ksize=prefilter_gaussian_ksize,
    )
    sample_xy = np.stack([sample_x, sample_y], axis=1)

    dy = savgol_filter(
        y, window_length=smooth_window, polyorder=smooth_polyorder, deriv=1, delta=dt
    )
    abs_dy = np.abs(dy)

    peak_idx, peak_val, t_left, t_right, ok, reason = _find_fwhm(
        t, dy, peak_search_half=peak_search_half, side_peak_ratio=side_peak_ratio
    )
    fwhm = (t_right - t_left) if ok else float("nan")

    return ProfileResult(
        base_point=base_point,
        normal=normal,
        t=t,
        y=y,
        dy=dy,
        abs_dy=abs_dy,
        sample_xy=sample_xy,
        peak_idx=peak_idx,
        peak_value=peak_val,
        t_left=t_left,
        t_right=t_right,
        fwhm=fwhm,
        ok=ok,
        reason=reason,
    )

def compute_multi_normal_profile(
    img_14bit: np.ndarray,
    line: LineFit,
    base_point: np.ndarray | None = None,
    base_offset: float = 2.0,
    half_length: float = 50.0,
    num_samples: int = 301,
    smooth_window: int = 21,
    smooth_polyorder: int = 3,
    prefilter_sigma: float = 1.5,
    prefilter_gaussian_ksize: int = 9,
    peak_search_half: float | None = 5.0,
    side_peak_ratio: float = 0.95,
    multi_profile_half_width: float = 10.0,
    multi_profile_step_scale: float = 2.0,
) -> ProfileResult:
    """多条平行法线剖面叠加：在 L 线附近生成多条平行法线，分别求导后对齐并平均 dy/dt，再量 FWHM。

    与 :func:`compute_parallel_smoothed_profile` 的区别：
    - 后者先沿平行方向平均灰度，再对一条法向剖面求导；
    - 本函数取 L 线附近的若干条平行法线，每条独立求导得到 dy/dt，
      再把所有 dy/dt 对齐叠加平均，得到最终 dy/dt，在上面量 FWHM。
    这样可以在边缘略有弯曲时仍得到稳定的导数主峰。

    Parameters
    ----------
    img_14bit
        14-bit 灰度图 (H, W), uint16 / float 均可。
    line
        已拟合的直线。
    base_point
        法线基点 (x, y)。默认: 取拟合线段的中点。
    half_length
        每条法线半长度 (像素), 采样区间为 ``[-half_length, +half_length]``。
    num_samples
        每条法线方向采样点数。
    smooth_window, smooth_polyorder
        Savitzky-Golay 滤波参数。
    prefilter_sigma
        剖面前整图高斯平滑 σ; ``<=0`` 时不做高斯。
    peak_search_half
        主峰搜索半范围。
    side_peak_ratio
        主峰旁峰判据；``>=1.0`` 时禁用。
    multi_profile_half_width
        平行于 L 线的方向、以 base_point 为中心的半宽度（像素）。
    multi_profile_step_scale
        相邻法线间距 = 法线方向步长 ``dt`` × 本系数。
    """
    if line is None:
        raise ValueError("line is None")

    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window <= smooth_polyorder:
        smooth_window = smooth_polyorder + 1 + ((smooth_polyorder + 1) % 2)

    normal = line.normal()
    if base_point is None:
        base_point = line.midpoint + base_offset * normal
    base_point = np.asarray(base_point, dtype=float)

    t = np.linspace(-half_length, half_length, num_samples)
    dt = float(t[1] - t[0])

    parallel_dir = line.direction
    spacing = dt * float(multi_profile_step_scale)
    if spacing <= 0:
        spacing = dt if dt > 0 else 1.0
    n_offsets = max(int(round(multi_profile_half_width / spacing)), 0)
    if n_offsets == 0:
        offsets = np.array([0.0])
    else:
        offsets = np.arange(-n_offsets, n_offsets + 1, dtype=float) * spacing

    img_float = img_14bit.astype(np.float64)
    if prefilter_sigma > 0:
        k = int(prefilter_gaussian_ksize)
        if k % 2 == 0:
            k += 1
        k = max(k, 1)
        img_float = cv2.GaussianBlur(img_float, (k, k), prefilter_sigma)

    # 收集每条平行法线的灰度和 dy/dt
    dy_stack: list[np.ndarray] = []
    y_stack: list[np.ndarray] = []
    sample_xy_list: list[np.ndarray] = []
    for off in offsets:
        base_off = base_point + off * parallel_dir
        sample_x = base_off[0] + t * normal[0]
        sample_y = base_off[1] + t * normal[1]
        sample_xy_list.append(np.stack([sample_x, sample_y], axis=1))
        coords = np.vstack([sample_y, sample_x])
        y_off = map_coordinates(img_float, coords, order=1, mode="nearest")
        y_stack.append(y_off)
        dy_off = savgol_filter(
            y_off,
            window_length=smooth_window,
            polyorder=smooth_polyorder,
            deriv=1,
            delta=dt,
        )
        dy_stack.append(dy_off)

    # 直接平均各条法线的灰度和 dy/dt
    y = np.mean(np.stack(y_stack, axis=0), axis=0)
    dy = np.mean(np.stack(dy_stack, axis=0), axis=0)
    abs_dy = np.abs(dy)
    sample_xy = np.mean(np.stack(sample_xy_list, axis=0), axis=0)

    peak_idx, peak_val, t_left, t_right, ok, reason = _find_fwhm(
        t, dy, peak_search_half=peak_search_half, side_peak_ratio=side_peak_ratio
    )
    fwhm = (t_right - t_left) if ok else float("nan")

    return ProfileResult(
        base_point=base_point,
        normal=normal,
        t=t,
        y=y,  # 多条法线对应位置的灰度平均，用于 debug 显示
        dy=dy,
        abs_dy=abs_dy,
        sample_xy=sample_xy,
        peak_idx=peak_idx,
        peak_value=peak_val,
        t_left=t_left,
        t_right=t_right,
        fwhm=fwhm,
        ok=ok,
        reason=reason,
    )