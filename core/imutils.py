"""图像 I/O 工具 - 兼容中文/Unicode 路径 (Windows cv2.imread 静默失败修复)

Windows 上 cv2.imread/imwrite 对含非 ASCII 字符的路径静默返回 None / 不写入。
本模块提供 drop-in 替代: imread_unicode / imwrite_unicode。
"""
from __future__ import annotations

import numpy as np


def imread_unicode(path: str, flags: int = -1) -> np.ndarray | None:
    """读取图像, 兼容中文路径。

    等价于 cv2.imread(path, flags), 但支持 Unicode 路径。
    flags: cv2.IMREAD_COLOR(1) / IMREAD_GRAYSCALE(0) / IMREAD_UNCHANGED(-1)
    """
    import cv2
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray,
                    params: list[int] | None = None) -> bool:
    """写入图像, 兼容中文路径。

    等价于 cv2.imwrite(path, img, params), 但支持 Unicode 路径。
    """
    import cv2
    import os
    try:
        ext = os.path.splitext(path)[1]
        if params:
            ok, buf = cv2.imencode(ext, img, params)
        else:
            ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
        return False
    except Exception:
        return False
