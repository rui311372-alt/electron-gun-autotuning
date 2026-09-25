"""读 14-bit TIF 图像。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_tif14(path: str | Path) -> np.ndarray:
    """读 14-bit TIF,返回 uint16 二维数组。

    14-bit 数据通常以 uint16 存储,有效值范围 [0, 16383]。
    在 Windows 上若路径含非 ASCII 字符, ``cv2.imread`` 常失败,此时改用内存解码。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"图像文件不存在: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint16:
        img = img.astype(np.uint16)
    return img


def stretch_to_uint8(
    img_14bit: np.ndarray,
    pct_low: float = 1.0,
    pct_high: float = 99.9,
) -> np.ndarray:
    """把 14-bit 图按百分位窗口拉伸到 uint8,仅用于可视化。"""
    lo = np.percentile(img_14bit, pct_low)
    hi = np.percentile(img_14bit, pct_high)
    if hi <= lo:
        hi = lo + 1
    clipped = np.clip(img_14bit, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
