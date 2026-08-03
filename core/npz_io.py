"""NPZ 特征库安全加载 (Pickle 反序列化硬化, v1.5.0)

np.load(allow_pickle=True) 等价于 pickle.load, 恶意 .npz 可执行任意代码。
本项目保存的特征库仅含数值数组与字符串元数据 (无需 pickle),
故默认 allow_pickle=False; 仅旧版遗留库加载失败时回退并告警。
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger("visionocr.npz")


def _npz_has_object_arrays(path: Path | str) -> bool:
    """从 npy 文件头读 dtype (不触发数组反序列化)。"""
    from numpy.lib import format as _fmt
    try:
        with zipfile.ZipFile(str(path)) as zf:
            for name in zf.namelist():
                if not name.endswith(".npy"):
                    continue
                with zf.open(name) as fp:
                    version = _fmt.read_magic(fp)
                    if version is None:
                        continue
                    shape, fortran, dtype = _fmt.read_array_header_1_0(fp) \
                        if version == (1, 0) else _fmt.read_array_header_2_0(fp)
                    if dtype == object:
                        return True
    except Exception:
        # 头部解析失败 → 保守视为含对象, 交由后续策略处理
        return True
    return False


def load_npz_safe(path: Path | str,
                  allow_legacy_pickle: bool = False) -> np.lib.npyio.NpzFile:
    """安全加载 .npz: 默认禁用 pickle (拒绝加载含对象数组的文件)。

    Args:
        allow_legacy_pickle: 显式置 True 时才允许回退 pickle 反序列化,
            仅限加载自产可信目录 (data/banks*) 的遗留库, 并记录安全告警。

    Raises:
        FileNotFoundError / ValueError 等由 np.load 原样抛出。
    """
    if _npz_has_object_arrays(path):
        if not allow_legacy_pickle:
            raise ValueError(
                f"特征库 {path} 含 pickle 对象, 拒绝加载 (安全策略); "
                f"请重新训练/保存该特征库")
        # 含 object 数组 (旧版库): 仅自产可信库可回退, 记录安全告警
        logger.warning(
            "特征库 %s 含 pickle 对象 (旧版格式), 已回退加载; "
            "建议重新保存以消除反序列化风险", path)
        return np.load(str(path), allow_pickle=True)
    return np.load(str(path), allow_pickle=False)
