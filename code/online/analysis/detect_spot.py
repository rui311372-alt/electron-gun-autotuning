"""高分位阈值 + 连通域 -> 左下角白斑 mask。

实际数据特点(14-bit, max=16383):
    - 圆斑内部相当一部分像素饱和到 16383
    - T 形灯丝阴影把饱和区切成多块,左下/右下角是 L 形的"白斑"
    - 阈值取饱和门附近,只用开运算保留 T 形阴影的细缝
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


def _build_binary(img_14bit: np.ndarray, thresh: float, morph_kernel: int) -> np.ndarray:
    """`>=` 阈值 + 形态学开运算 (只去毛刺,不闭合细缝)。"""
    binary = (img_14bit >= thresh).astype(np.uint8) * 255
    if morph_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


@dataclass
class SpotDetection:
    """白斑检测结果。"""

    mask: np.ndarray
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int
    threshold: float


def detect_lower_left_spot(
    img_14bit: np.ndarray,
    percentile: float = 99.5,
    min_area: int = 3000,
    morph_kernel: int = 5,
    saturation_floor: int | None = 16380,
    relaxed_saturation_floors: Sequence[int] | None = None,
) -> SpotDetection | None:
    """检测左下角白斑(L 形饱和区)。

    Parameters
    ----------
    saturation_floor
        若不为 None, 阈值不会低于这个值。14-bit 满量程 16383, 取 16380 可以
        让阈值始终贴近饱和门, 即使 percentile 取到 16383 也能正常工作。
    relaxed_saturation_floors
        当分位数二值图面积仍不足 ``min_area`` 时, 依次尝试这些固定阈值重算二值图。
        传 ``()`` 可关闭该回退逻辑。
    """
    if img_14bit.ndim != 2:
        raise ValueError("expected 2D image")
    h, w = img_14bit.shape

    thresh = float(np.percentile(img_14bit, percentile))
    if saturation_floor is not None:
        thresh = max(thresh, float(saturation_floor))

    thresh_source = "percentile/floor"
    binary = _build_binary(img_14bit, thresh, morph_kernel)

    floors = (
        tuple(relaxed_saturation_floors)
        if relaxed_saturation_floors is not None
        else (16300, 16200, 16000)
    )

    def _max_cc_area(bin_img: np.ndarray) -> int:
        """返回最大连通域面积（不含背景）。"""
        n, _, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        if n <= 1:
            return 0
        return int(max(stats[i, cv2.CC_STAT_AREA] for i in range(1, n)))

    # 用最大连通域面积判断，而非全图白像素总数
    if _max_cc_area(binary) < min_area:
        for relaxed_floor in floors:
            binary = _build_binary(img_14bit, float(relaxed_floor), morph_kernel)
            max_area = _max_cc_area(binary)
            if max_area >= min_area:
                thresh = float(relaxed_floor)
                thresh_source = f"relaxed_floor={relaxed_floor}"
                break

    max_cc_area = _max_cc_area(binary)
    print(f"[detect_spot] 阈值={thresh:.0f} (来源={thresh_source}), "
          f"最大连通域面积={max_cc_area}, min_area={min_area}")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    candidates: list[int] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        candidates.append(label)

    if not candidates:
        return None

    lower_left = [
        lb
        for lb in candidates
        if centroids[lb, 0] < w / 2 and centroids[lb, 1] > h / 2
    ]
    if lower_left:
        chosen = min(lower_left, key=lambda lb: centroids[lb, 0])
    else:
        chosen = min(candidates, key=lambda lb: centroids[lb, 0] - centroids[lb, 1])

    mask = labels == chosen
    cx, cy = float(centroids[chosen, 0]), float(centroids[chosen, 1])
    x = int(stats[chosen, cv2.CC_STAT_LEFT])
    y = int(stats[chosen, cv2.CC_STAT_TOP])
    bw = int(stats[chosen, cv2.CC_STAT_WIDTH])
    bh = int(stats[chosen, cv2.CC_STAT_HEIGHT])
    area = int(stats[chosen, cv2.CC_STAT_AREA])

    return SpotDetection(
        mask=mask,
        centroid=(cx, cy),
        bbox=(x, y, bw, bh),
        area=area,
        threshold=thresh,
    )
