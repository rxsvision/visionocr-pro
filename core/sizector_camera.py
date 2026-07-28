"""Sizector 3D 结构光深度相机采集模块 (Phase 4C)
==================================================

基于 MPSizectorS .NET wrapper (MPSizectorS_DotNet.dll) 通过 pythonnet 调用,
获取 **深度图 (Z-Map, mm) + 辅助 RGB/灰度图**, 用于 3D 融合缺陷检测。

为何选 .NET wrapper 而非 ctypes 直调 C++ DLL:
    C++ API (IMPSizectorS) 是纯虚接口 (vtable), 且大量结构体按值返回 +
    非托管内存生命周期管理, Python 侧用 ctypes 复刻 vtable 调用极易崩溃。
    厂商已提供完整托管封装 (枚举/Open/Snap/结构体 marshaling/内存释放),
    pythonnet 直接复用是最稳、最省维护成本的路径 (少造轮子原则)。

数据链路 (已通过反射核实):
    SnapUnmanaged(softTrigger, out fmt, out umData, timeoutMS)
      -> umData.ToManagedFixZMapSimple()
      -> ManagedDataFrameFixZMapSimpleStruct {
             FrameInfo,            # 含 DataInfo.XPixResolution/YPixResolution
             UInt16[] Z,           # 原始深度 (无符号短整型)
             Byte[] AuxiliaryWhite,# 辅助灰度图 (H*W)
             Byte[] AuxiliaryRGB,  # 辅助彩色图 (平面 R|G|B, 各 H*W)
         }
      -> Z_real_mm = Z * PointScaleSetting.ZIncrement + Z0Pos
      -> Utils.FreeUnmanagedData(umData)   # 必须释放非托管内存

对外暴露:
    - DepthFrame               深度帧数据类 (depth_mm / rgb / gray / 元数据)
    - SizectorCamera           真实相机后端 (pythonnet + .NET wrapper)
    - MockSizectorCamera       无硬件时的合成深度相机 (开发/演示/CI)
    - create_depth_camera(cfg) 工厂函数 (按 config 选择真实 / Mock)

依赖:
    pip install pythonnet numpy opencv-python
    运行时需要 .NET Framework 4.7.2+ (Win10/11 自带 4.8) 与
    MPSizectorS_DotNet.dll (+ 同目录 native MPSizectorS_API.dll)。

配置示例 (config.yaml):
    sizector:
      enabled: true
      dll_dir: "E:/.../MPSizectorS_SDK_V2_70/02_Binary/03_SizectorS_NET"
      index: 0
      working_mode: "precise"      # fastest | fast | precise | super_precise
      use_rgb: true                # 采集辅助 RGB (否则仅灰度)
      timeout_ms: 5000
      mock: false                  # true 时不连硬件, 输出合成深度帧
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 深度帧数据结构
# =============================================================================
@dataclass
class DepthFrame:
    """一帧深度 + RGB 采集结果。

    Attributes:
        depth_mm: (H, W) float32 深度图, 单位 mm; 无效点为 np.nan。
        rgb:      (H, W, 3) uint8 BGR 彩色图 (OpenCV 顺序); 无辅助彩色时为 None。
        gray:     (H, W) uint8 灰度图; 无辅助灰度时为 None。
        width:    图像宽度 (像素)。
        height:   图像高度 (像素)。
        z_min:    有效深度最小值 (mm)。
        z_max:    有效深度最大值 (mm)。
        valid_ratio: 有效深度点占比 (0~1)。
        serial:   设备序列号 (Mock 时为 "MOCK")。
        meta:     附加元数据 (工作模式 / 帧号等)。
    """
    depth_mm: np.ndarray
    rgb: np.ndarray | None = None
    gray: np.ndarray | None = None
    width: int = 0
    height: int = 0
    z_min: float = 0.0
    z_max: float = 0.0
    valid_ratio: float = 0.0
    serial: str = ""
    meta: dict = field(default_factory=dict)

    def depth_colormap(self) -> np.ndarray:
        """将深度图渲染为 JET 伪彩色 BGR (无效点黑色), 便于人工查看。"""
        import cv2

        d = self.depth_mm
        valid = np.isfinite(d)
        if not valid.any():
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        lo, hi = np.nanmin(d), np.nanmax(d)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        norm = np.clip((d - lo) / (hi - lo), 0, 1)
        norm_u8 = (norm * 255).astype(np.uint8)
        color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
        color[~valid] = 0  # 无效点黑色
        return color


# =============================================================================
# 工作模式映射 (config 字符串 -> .NET WorkingModeType 枚举名)
# =============================================================================
# 反射核实的工作模式枚举:
#   Fast3D=0, Standard3D=1, Precise3D=2, SuperPrecise3D=3, White2D=4,
#   Black2D=5, Grid2D=6, ExposurePrediction2D=7, SuperFast3D=8,
#   FastPrecision3D=9, SuperDynamic3D=10
_WORKING_MODE_MAP = {
    "fastest": "SuperFast3D",
    "fast": "Fast3D",
    "standard": "Standard3D",
    "precise": "Precise3D",
    "super_precise": "SuperPrecise3D",
    "fast_precision": "FastPrecision3D",
    "dynamic": "SuperDynamic3D",
}

# 默认 SDK 安装 / 常见路径 (用于自动定位 .NET wrapper 目录)
_DEFAULT_DLL_DIRS = [
    r"C:\Program Files (x86)\MPSizectorS SDK\02_Binary\03_SizectorS_NET",
    r"C:\Program Files\MPSizectorS SDK\02_Binary\03_SizectorS_NET",
]


# =============================================================================
# 真实相机后端 (pythonnet + .NET wrapper)
# =============================================================================
class SizectorCamera:
    """Sizector 3D 结构光相机 (通过 pythonnet 调用 MPSizectorS_DotNet.dll)。

    用法::

        cam = SizectorCamera(dll_dir="...", index=0, working_mode="precise")
        if cam.open():
            frame = cam.capture()      # -> DepthFrame
            cam.close()
    """

    def __init__(
        self,
        dll_dir: str | None = None,
        index: int = 0,
        serial: str | None = None,
        working_mode: str = "precise",
        use_rgb: bool = True,
        timeout_ms: int = 5000,
    ):
        self.dll_dir = dll_dir
        self.index = index
        self.serial_match = serial
        self.working_mode = working_mode
        self.use_rgb = use_rgb
        self.timeout_ms = int(timeout_ms)

        self._mp = None          # MPSizectorS_DotNet 命名空间模块
        self._utils = None       # Utils 类 (释放非托管内存)
        self._sensor = None      # MPSizectorS 实例
        self._enums = None       # 枚举类型缓存
        self._opened = False
        self._device_info = None

    # ------------------------------------------------------------------ 加载 SDK
    def _locate_dll_dir(self) -> str | None:
        """定位包含 MPSizectorS_DotNet.dll 的目录。"""
        candidates = []
        if self.dll_dir:
            candidates.append(self.dll_dir)
        candidates.extend(_DEFAULT_DLL_DIRS)
        for c in candidates:
            if c and (Path(c) / "MPSizectorS_DotNet.dll").exists():
                return c
        return None

    def _load_sdk(self) -> bool:
        """加载 pythonnet 运行时并导入 MPSizectorS_DotNet 程序集。"""
        dll_dir = self._locate_dll_dir()
        if dll_dir is None:
            logger.error(
                "未找到 MPSizectorS_DotNet.dll。\n"
                "  请在 config.yaml 的 sizector.dll_dir 指定 .NET wrapper 目录,\n"
                "  或将 sizector.mock 设为 true 使用合成深度相机。\n"
                "  已尝试路径: %s", [self.dll_dir] + _DEFAULT_DLL_DIRS,
            )
            return False

        # 将 native DLL 目录加入搜索路径 (MPSizectorS_API.dll 在同目录或 x64 子目录)
        for sub in (dll_dir, str(Path(dll_dir) / "x64")):
            if Path(sub).is_dir():
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(sub)
                    except OSError:
                        pass
                os.environ["PATH"] = sub + os.pathsep + os.environ.get("PATH", "")

        try:
            import clr  # pythonnet
        except ImportError:
            logger.error(
                "未安装 pythonnet, Sizector 深度相机不可用。\n"
                "  安装方式: pip install pythonnet\n"
                "  (需要 .NET Framework 4.7.2+, Win10/11 自带)\n"
                "  或将 sizector.mock 设为 true 使用合成深度相机。"
            )
            return False

        dll_path = str(Path(dll_dir) / "MPSizectorS_DotNet.dll")
        try:
            clr.AddReference(dll_path)
        except Exception as e:  # noqa: BLE001 - .NET 加载异常类型多样
            logger.error("加载 MPSizectorS_DotNet.dll 失败: %s "
                         "(可能缺少 .NET Framework 或 native 依赖)", e)
            return False

        try:
            import MPSizectorS_DotNet as mp  # noqa: N813
            self._mp = mp
            self._utils = mp.Utils
            self._enums = {
                "WorkingModeType": mp.WorkingModeType,
                "DataOutModeType": mp.DataOutModeType,
                "TriggerSourceType": mp.TriggerSourceType,
                "DataFormatType": mp.DataFormatType,
                "DeviceStateType": mp.DeviceStateType,
            }
        except Exception as e:  # noqa: BLE001
            logger.error("导入 MPSizectorS_DotNet 命名空间失败: %s", e)
            return False

        logger.info("Sizector .NET wrapper 已加载: %s", dll_path)
        return True

    # ------------------------------------------------------------------ 设备枚举
    def _select_device(self) -> bool:
        """枚举设备并按 serial / index 选择目标相机。"""
        mp = self._mp
        self._sensor = mp.MPSizectorS()

        if not self._sensor.UpdateDeviceList():
            logger.error("Sizector 枚举设备失败 (UpdateDeviceList 返回 false)")
            return False

        count = self._sensor.GetDeviceCount()
        if count <= 0:
            logger.error("未发现任何 Sizector 深度相机, 请检查 USB3.0 接线与驱动")
            return False

        logger.info("枚举到 %d 台 Sizector 相机", count)

        # 优先按序列号精确匹配
        if self.serial_match:
            for i in range(count):
                info = self._sensor.GetDeviceInfo(i)
                if str(info.DeviceSerialNumber) == self.serial_match:
                    self._device_info = info
                    logger.info("按序列号匹配到设备 #%d: %s", i, self.serial_match)
                    return True
            logger.error("未找到序列号为 %s 的设备", self.serial_match)
            return False

        # 否则按 index 选择
        if 0 <= self.index < count:
            self._device_info = self._sensor.GetDeviceInfo(self.index)
            return True

        logger.error("相机 index=%s 越界, 共 %d 台", self.index, count)
        return False

    def _apply_params(self) -> None:
        """配置工作模式 / 数据输出模式 / 软触发 (失败仅告警)。"""
        mp = self._mp
        s = self._sensor

        # 工作模式
        wm_name = _WORKING_MODE_MAP.get((self.working_mode or "precise").lower(),
                                        "Precise3D")
        try:
            s.WorkingMode = getattr(self._enums["WorkingModeType"], wm_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("设置工作模式 %s 失败: %s", wm_name, e)

        # 数据输出: FixZMapSimple (深度图 + 辅助图, 数据量最小且含 RGB)
        try:
            s.DataOutMode = self._enums["DataOutModeType"].FixZMapSimple
        except Exception as e:  # noqa: BLE001
            logger.warning("设置 DataOutMode=FixZMapSimple 失败: %s", e)

        # 软触发模式 (每次 capture 主动下发触发)
        try:
            s.TriggerSource = self._enums["TriggerSourceType"].SoftTriggerOnly
        except Exception as e:  # noqa: BLE001
            logger.warning("设置软触发失败: %s", e)

        # 辅助 RGB 开关 (通过外部曝光使能位控制, 0x2 = 启用彩色辅助)
        try:
            s.ExternalExposureEnable = 0x2 if self.use_rgb else 0x0
        except Exception as e:  # noqa: BLE001
            logger.debug("设置辅助彩色开关失败 (可忽略): %s", e)

    # ------------------------------------------------------------------ 公共接口
    def open(self) -> bool:
        if not self._load_sdk():
            return False
        if not self._select_device():
            return False

        try:
            if not self._sensor.Open(self._device_info):
                logger.error("打开 Sizector 设备失败 (可能被其他进程占用)")
                return False
        except Exception as e:  # noqa: BLE001
            logger.error("打开设备异常: %s", e)
            return False

        self._sensor.SetAutoReconnect(True)
        self._apply_params()

        # 等待设备初始化完成 (Disconnected / UnderInit -> StandBy)
        import time
        ds = self._enums["DeviceStateType"]
        for _ in range(200):  # 最多等 10s
            try:
                state = self._sensor.GetDeviceState()
                if state != ds.Disconnected and state != ds.UnderInit:
                    break
            except Exception:  # noqa: BLE001
                break
            time.sleep(0.05)

        self._opened = True
        name = getattr(self._device_info, "DeviceName", "?")
        sn = getattr(self._device_info, "DeviceSerialNumber", "?")
        logger.info("Sizector 相机已就绪: %s (SN=%s, mode=%s)",
                    name, sn, self.working_mode)
        return True

    def capture(self) -> DepthFrame | None:
        """软触发采集一帧, 返回 DepthFrame (深度 mm + RGB), 失败返回 None。"""
        if not self._opened or self._sensor is None:
            return None

        mp = self._mp
        s = self._sensor

        # 通过引用参数返回 DataFormat 与未托管帧数据
        fmt_box = mp.DataFormatType.FixZMapSimple
        um_box = mp.UnmanagedDataFrameUndefinedStruct()

        try:
            ok = s.SnapUnmanaged(True, fmt_box, um_box, self.timeout_ms)
        except Exception as e:  # noqa: BLE001
            logger.error("SnapUnmanaged 调用异常: %s", e)
            return None

        if not ok:
            logger.error("Sizector 采集失败 (SnapUnmanaged 返回 false), "
                         "请检查曝光/工作距离/触发设置")
            return None

        try:
            return self._convert_frame(um_box)
        finally:
            # 无论成功失败都必须释放非托管内存, 否则泄漏
            try:
                self._utils.FreeUnmanagedData(um_box)
            except Exception as e:  # noqa: BLE001
                logger.debug("释放非托管帧失败 (可忽略): %s", e)

    def _convert_frame(self, um_box) -> DepthFrame | None:
        """将未托管帧转为 DepthFrame (深度 mm + numpy 数组)。"""
        # 转为托管 FixZMapSimple 帧 (含 UInt16[] Z, Byte[] AuxiliaryWhite/RGB)
        managed = um_box.ToManagedFixZMapSimple()

        info = managed.FrameInfo.DataInfo
        w = int(info.XPixResolution)
        h = int(info.YPixResolution)
        if w <= 0 or h <= 0:
            logger.error("帧尺寸非法: %dx%d", w, h)
            return None

        # 深度标定参数: Z_real = Z_raw * ZIncrement + Z0Pos
        pss = managed.FrameInfo.PostProcessSettings.PointScaleSetting
        z_inc = float(pss.ZIncrement)
        z0 = float(pss.Z0Pos)

        # UInt16[] Z -> numpy (H, W), 转 mm; 原始 0 视为无效点
        z_raw = np.array(managed.Z, dtype=np.uint16).reshape(h, w)
        depth_mm = z_raw.astype(np.float32) * z_inc + z0
        depth_mm[z_raw == 0] = np.nan

        valid = np.isfinite(depth_mm)
        valid_ratio = float(valid.mean()) if valid.size else 0.0
        z_min = float(np.nanmin(depth_mm)) if valid.any() else 0.0
        z_max = float(np.nanmax(depth_mm)) if valid.any() else 0.0

        # 辅助灰度图
        gray = None
        if managed.AuxiliaryWhite is not None and len(managed.AuxiliaryWhite) >= w * h:
            gray = np.array(managed.AuxiliaryWhite, dtype=np.uint8)[: w * h].reshape(h, w)

        # 辅助 RGB (平面排布 R|G|B, 各 H*W) -> BGR (OpenCV 顺序)
        rgb = None
        if self.use_rgb and managed.AuxiliaryRGB is not None \
                and len(managed.AuxiliaryRGB) >= 3 * w * h:
            buf = np.array(managed.AuxiliaryRGB, dtype=np.uint8)
            r = buf[0 * w * h:1 * w * h].reshape(h, w)
            g = buf[1 * w * h:2 * w * h].reshape(h, w)
            b = buf[2 * w * h:3 * w * h].reshape(h, w)
            rgb = np.stack([b, g, r], axis=-1)  # BGR

        sn = ""
        try:
            sn = str(managed.FrameInfo.DeviceInfo.DeviceSerialNumber)
        except Exception:  # noqa: BLE001
            pass

        return DepthFrame(
            depth_mm=depth_mm,
            rgb=rgb,
            gray=gray,
            width=w,
            height=h,
            z_min=z_min,
            z_max=z_max,
            valid_ratio=valid_ratio,
            serial=sn,
            meta={
                "working_mode": self.working_mode,
                "z_increment": z_inc,
                "z0_pos": z0,
                "frame_sn": int(getattr(info.TSN, "SN", 0)),
            },
        )

    def close(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.SetAutoReconnect(False)
                self._sensor.Close()
            except Exception:  # noqa: BLE001
                pass
            self._sensor = None
        self._opened = False
        self._device_info = None
        logger.info("Sizector 相机已关闭")

    # 上下文管理
    def __enter__(self) -> "SizectorCamera":
        if not self.open():
            raise RuntimeError("无法打开 Sizector 深度相机")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# =============================================================================
# Mock 深度相机 (无硬件时合成深度帧, 用于开发/演示/CI)
# =============================================================================
class MockSizectorCamera:
    """合成深度相机: 生成带模拟缺陷 (凸起/凹坑) 的深度图 + RGB。

    与 SizectorCamera 接口完全一致 (open/capture/close), 便于无硬件开发调试。
    每次 capture 随机生成一个平面基准 + 若干局部高度异常, 模拟真实工业场景。
    """

    def __init__(self, width: int = 640, height: int = 480,
                 working_mode: str = "precise", use_rgb: bool = True, **_):
        self.width = width
        self.height = height
        self.working_mode = working_mode
        self.use_rgb = use_rgb
        self._opened = False
        self._rng = np.random.default_rng()

    def open(self) -> bool:
        self._opened = True
        logger.info("MockSizectorCamera 已打开 (%dx%d, 合成深度帧)",
                    self.width, self.height)
        return True

    def capture(self) -> DepthFrame | None:
        if not self._opened:
            return None

        w, h = self.width, self.height
        rng = self._rng

        # 基准平面: 缓慢倾斜 + 噪声, 中心距离 ~100mm
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = 100.0 + 0.01 * (xx - w / 2) + 0.008 * (yy - h / 2)
        base += rng.normal(0, 0.05, size=(h, w)).astype(np.float32)
        depth = base.copy()

        # 随机注入 1~3 个局部高度异常 (模拟凸起/凹坑缺陷)
        n_defects = int(rng.integers(1, 4))
        defect_centers = []
        for _ in range(n_defects):
            cx = int(rng.integers(w // 5, 4 * w // 5))
            cy = int(rng.integers(h // 5, 4 * h // 5))
            radius = int(rng.integers(15, 45))
            amp = float(rng.uniform(0.8, 3.0)) * rng.choice([-1, 1])
            rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            bump = amp * np.exp(-(rr ** 2) / (2 * (radius / 2) ** 2))
            depth += bump.astype(np.float32)
            defect_centers.append((cx, cy, radius, amp))

        # 随机挖一个无效区 (模拟反光/遮挡导致的深度缺失)
        if rng.random() < 0.5:
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            depth[rr < int(rng.integers(10, 30))] = np.nan

        valid = np.isfinite(depth)

        # 合成 RGB: 灰底 + 缺陷处颜色微扰 (模拟表面色差)
        rgb = None
        if self.use_rgb:
            gray_bg = rng.integers(120, 160, size=(h, w), dtype=np.uint8)
            bgr = np.stack([gray_bg, gray_bg, gray_bg], axis=-1).astype(np.uint8)
            for cx, cy, radius, amp in defect_centers:
                rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                mask = rr < radius
                if amp > 0:  # 凸起偏红
                    bgr[mask, 2] = np.clip(bgr[mask, 2].astype(int) + 40, 0, 255).astype(np.uint8)
                else:        # 凹坑偏暗
                    bgr[mask] = (bgr[mask].astype(int) * 0.6).astype(np.uint8)
            rgb = bgr

        gray = None
        if rgb is not None:
            gray = rgb.mean(axis=-1).astype(np.uint8)

        return DepthFrame(
            depth_mm=depth,
            rgb=rgb,
            gray=gray,
            width=w,
            height=h,
            z_min=float(np.nanmin(depth)) if valid.any() else 0.0,
            z_max=float(np.nanmax(depth)) if valid.any() else 0.0,
            valid_ratio=float(valid.mean()),
            serial="MOCK",
            meta={"working_mode": self.working_mode, "mock": True,
                  "n_defects": n_defects},
        )

    def close(self) -> None:
        self._opened = False
        logger.info("MockSizectorCamera 已关闭")

    def __enter__(self) -> "MockSizectorCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# =============================================================================
# 工厂函数
# =============================================================================
def create_depth_camera(config: dict):
    """根据 config["sizector"] 创建深度相机实例。

    - sizector.mock=true 或 enabled=false 时返回 MockSizectorCamera;
    - 否则返回 SizectorCamera (真实硬件)。
    返回对象统一支持 open() / capture() / close()。
    """
    cfg = (config or {}).get("sizector", {}) or {}

    if cfg.get("mock", False) or not cfg.get("enabled", True):
        logger.info("使用 Mock 深度相机 (sizector.mock=%s, enabled=%s)",
                    cfg.get("mock"), cfg.get("enabled"))
        return MockSizectorCamera(
            width=cfg.get("mock_width", 640),
            height=cfg.get("mock_height", 480),
            working_mode=cfg.get("working_mode", "precise"),
            use_rgb=cfg.get("use_rgb", True),
        )

    return SizectorCamera(
        dll_dir=cfg.get("dll_dir"),
        index=cfg.get("index", 0),
        serial=cfg.get("serial"),
        working_mode=cfg.get("working_mode", "precise"),
        use_rgb=cfg.get("use_rgb", True),
        timeout_ms=cfg.get("timeout_ms", 5000),
    )
