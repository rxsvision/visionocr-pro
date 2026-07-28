"""相机采集抽象层 (Camera Acquisition Abstraction Layer)
=========================================================

本模块为 VisionOCR Pro 提供统一的相机采集接口, 支持以下三种后端:

1. OpenCVCamera      —— 基于 OpenCV VideoCapture, 适用于 USB 摄像头 / 笔记本内置相机;
2. HikvisionCamera   —— 基于海康机器人 MVS SDK (MvCameraControl.dll, ctypes 直调),
                        支持 GigE 与 USB3 工业相机, 支持软触发与曝光配置;
3. GigEVisionCamera  —— 基于 harvesters + GenTL Producer 的通用 GigE Vision 实现,
                        可适配海康 / 大恒 / Basler / FLIR 等任意 GenTL 厂商。

对外仅暴露:
    - BaseCamera            抽象基类
    - OpenCVCamera          OpenCV 实现
    - HikvisionCamera       海康 MVS 实现
    - GigEVisionCamera      harvesters 实现
    - create_camera(config) 工厂函数 (读取 config["camera"]["type"])

配置示例 (config.yaml):
    camera:
      type: "hikvision"          # opencv | gigevision | hikvision
      index: 0                   # 多相机时按枚举顺序选择
      gigE_ip: "192.168.1.100"   # 可选, 按 IP 精确匹配目标 GigE 相机
      trigger: "software"        # software | freerun
      exposure_us: 5000          # 曝光时间 (微秒)
      timeout_ms: 3000           # 取流超时 (毫秒)
      hik_sdk_path: "C:\\Program Files (x86)\\MVS\\Development\\Bin\\Win64_x64"
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 抽象基类
# =============================================================================
class BaseCamera(ABC):
    """相机抽象基类: 所有具体相机后端必须实现 open / grab / close 三个方法。"""

    @abstractmethod
    def open(self) -> bool:
        """打开相机, 成功返回 True, 失败返回 False。"""
        ...

    @abstractmethod
    def grab(self) -> np.ndarray | None:
        """采集一帧图像, 统一返回 BGR 格式的 numpy ndarray (H, W, 3), 失败返回 None。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭相机并释放底层资源, 必须保证可重复调用且不抛异常。"""
        ...

    # 上下文管理支持: with create_camera(cfg) as cam: ...
    def __enter__(self) -> "BaseCamera":
        if not self.open():
            raise RuntimeError(f"无法打开相机: {type(self).__name__}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# =============================================================================
# OpenCV 实现 (USB 摄像头 / 通用 V4L2 / DirectShow)
# =============================================================================
class OpenCVCamera(BaseCamera):
    """基于 OpenCV 的通用相机实现, 适合 USB 摄像头与调试场景。"""

    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080):
        self.index = index
        self.width = width
        self.height = height
        self._cap = None

    def open(self) -> bool:
        import cv2  # 延迟导入, 避免无相机环境下拖累启动速度

        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            logger.error("OpenCV 无法打开相机 index=%s", self.index)
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info("OpenCV 相机已打开: index=%s", self.index)
        return True

    def grab(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
            logger.info("OpenCV 相机已关闭")


# =============================================================================
# 海康 MVS SDK ctypes 绑定 (常量 / 结构体 / 错误码)
# =============================================================================
# ---- 传输层类型 (nTLayerType) ----
MV_GIGE_DEVICE = 0x00000001   # GigE 相机
MV_USB_DEVICE = 0x00000004    # USB3 相机

# ---- 像素格式 (PixelType, 与 GenICam PFNC 编码兼容) ----
MV_GVSP_PIX_MONO = 0x01000000
MV_GVSP_PIX_RGB = 0x02000000
MV_GVSP_PIX_COLOR = 0x02000000
MV_GVSP_PIX_CUSTOM = 0x80000000

PIXEL_MONO8 = MV_GVSP_PIX_MONO | 0x0001
PIXEL_MONO10 = MV_GVSP_PIX_MONO | 0x0003
PIXEL_MONO12 = MV_GVSP_PIX_MONO | 0x0005
PIXEL_YUV422_8 = MV_GVSP_PIX_COLOR | 0x0032
PIXEL_YUV422_YUYV_Packed = MV_GVSP_PIX_COLOR | 0x0032
PIXEL_RGB8 = MV_GVSP_PIX_COLOR | 0x0014
PIXEL_BGR8 = MV_GVSP_PIX_COLOR | 0x0015
PIXEL_BAYER_GR8 = MV_GVSP_PIX_MONO | 0x0008
PIXEL_BAYER_RG8 = MV_GVSP_PIX_MONO | 0x0009
PIXEL_BAYER_GB8 = MV_GVSP_PIX_MONO | 0x000A
PIXEL_BAYER_BG8 = MV_GVSP_PIX_MONO | 0x000B

# ---- 常见 SDK 错误码 (摘自 MVS 开发手册) ----
MV_OK = 0x00000000
MV_E_HANDLE = 0x80000001
MV_E_PARAMETER = 0x80000003
MV_E_NOENUM = 0x80000006
MV_E_NODATA = 0x80000007
MV_E_OPENED = 0x80000009
MV_E_CALLORDER = 0x8000000D
MV_E_BUFOVER = 0x8000001C
MV_E_TIMEOUT = 0x800000FE

_MV_ERR_NAMES = {
    MV_OK: "MV_OK (成功)",
    MV_E_HANDLE: "MV_E_HANDLE (无效句柄)",
    MV_E_PARAMETER: "MV_E_PARAMETER (参数错误)",
    MV_E_NOENUM: "MV_E_NOENUM (枚举失败)",
    MV_E_NODATA: "MV_E_NODATA (无数据)",
    MV_E_OPENED: "MV_E_OPENED (设备已被打开)",
    MV_E_CALLORDER: "MV_E_CALLORDER (调用顺序错误)",
    MV_E_BUFOVER: "MV_E_BUFOVER (缓冲区溢出)",
    MV_E_TIMEOUT: "MV_E_TIMEOUT (超时)",
}


def mv_err_str(code: int) -> str:
    """将 SDK 返回码格式化为可读字符串 (含十六进制原码)。"""
    code &= 0xFFFFFFFF
    name = _MV_ERR_NAMES.get(code)
    if name:
        return f"{name} [0x{code:08X}]"
    return f"MVS SDK 错误码 [0x{code:08X}]"


class _MV_GIGE_DEVICE_INFO(ctypes.Structure):
    """GigE 设备信息 (MV_GIGE_DEVICE_INFO 简化版, 仅保留常用字段)。"""
    _fields_ = [
        ("nIpCfgOption", ctypes.c_uint),
        ("nIpCfgCurrent", ctypes.c_uint),
        ("nCurrentIp", ctypes.c_uint),
        ("nCurrentSubNetMask", ctypes.c_uint),
        ("nDefultGateWay", ctypes.c_uint),
        ("chManufacturerName", ctypes.c_ubyte * 32),
        ("chModelName", ctypes.c_ubyte * 32),
        ("chDeviceVersion", ctypes.c_ubyte * 32),
        ("chManufacturerSpecificInfo", ctypes.c_ubyte * 48),
        ("chSerialNumber", ctypes.c_ubyte * 16),
        ("chUserDefinedName", ctypes.c_ubyte * 16),
        ("nNetExport", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 4),
    ]


class _MV_USB3_DEVICE_INFO(ctypes.Structure):
    """USB3 设备信息 (MV_USB3_DEVICE_INFO 简化版)。"""
    _fields_ = [
        ("nCrtlInEndPoint", ctypes.c_ubyte),
        ("nCrtlOutEndPoint", ctypes.c_ubyte),
        ("nStreamEndPoint", ctypes.c_ubyte),
        ("nEventEndPoint", ctypes.c_ubyte),
        ("idVendor", ctypes.c_ushort),
        ("idProduct", ctypes.c_ushort),
        ("nDeviceNumber", ctypes.c_uint),
        ("chDeviceGUID", ctypes.c_ubyte * 64),
        ("chVendorName", ctypes.c_ubyte * 64),
        ("chModelName", ctypes.c_ubyte * 64),
        ("chFamilyName", ctypes.c_ubyte * 64),
        ("chDeviceVersion", ctypes.c_ubyte * 64),
        ("chManufacturerName", ctypes.c_ubyte * 64),
        ("chSerialNumber", ctypes.c_ubyte * 64),
        ("chUserDefinedName", ctypes.c_ubyte * 64),
        ("nbcdUSB", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 3),
    ]


class _MV_SPEC_DEV_INFO(ctypes.Union):
    """设备特定信息联合体 (GigE / USB3 二选一)。"""
    _fields_ = [
        ("stGigEInfo", _MV_GIGE_DEVICE_INFO),
        ("stUsb3VInfo", _MV_USB3_DEVICE_INFO),
        ("chReserved", ctypes.c_ubyte * 1024),  # 兜底, 保证联合体尺寸不小于 SDK 实际值
    ]


class MV_CC_DEVICE_INFO(ctypes.Structure):
    """设备信息 (MV_CC_DEVICE_INFO), 枚举设备时由 SDK 填充。"""
    _fields_ = [
        ("nMajorVer", ctypes.c_ushort),
        ("nMinorVer", ctypes.c_ushort),
        ("nMacAddrHigh", ctypes.c_uint),
        ("nMacAddrLow", ctypes.c_uint),
        ("nTLayerType", ctypes.c_uint),
        ("nDevClass", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 2),
        ("SpecialInfo", _MV_SPEC_DEV_INFO),
    ]


class _MV_CC_DEVICE_INFO_PTR(ctypes.Structure):
    """设备列表中的指针项 (MV_CC_DEVICE_INFO_LIST.pDeviceInfo[i])。"""
    _fields_ = [("pDeviceInfo", ctypes.POINTER(MV_CC_DEVICE_INFO))]


class MV_CC_DEVICE_INFO_LIST(ctypes.Structure):
    """枚举到的设备列表 (MV_CC_DEVICE_INFO_LIST)。"""
    _fields_ = [
        ("nDeviceNum", ctypes.c_uint),
        ("pDeviceInfo", _MV_CC_DEVICE_INFO_PTR * 256),
    ]


class MV_FRAME_OUT_INFO_EX(ctypes.Structure):
    """帧输出信息 (MV_FRAME_OUT_INFO_EX), 含宽高 / 帧号 / 像素格式等。"""
    _fields_ = [
        ("nWidth", ctypes.c_ushort),
        ("nHeight", ctypes.c_ushort),
        ("nFrameLen", ctypes.c_uint),
        ("nDevTimeStampHigh", ctypes.c_uint),
        ("nDevTimeStampLow", ctypes.c_uint),
        ("nReserved0", ctypes.c_uint),
        ("nHostTimeStamp", ctypes.c_int64),
        ("nFrameNum", ctypes.c_uint),
        ("enPixelType", ctypes.c_uint),  # 像素格式 (GenICam PFNC 编码)
        ("nSecondCount", ctypes.c_uint),
        ("nCycleCount", ctypes.c_uint),
        ("nCycleOffset", ctypes.c_uint),
        ("fGain", ctypes.c_float),
        ("fExposureTime", ctypes.c_float),
        ("nAverageBrightness", ctypes.c_uint),
        ("nRed", ctypes.c_uint),
        ("nGreen", ctypes.c_uint),
        ("nBlue", ctypes.c_uint),
        ("nFrameCounter", ctypes.c_uint),
        ("nTriggerIndex", ctypes.c_uint),
        ("nInput", ctypes.c_uint),
        ("nOutput", ctypes.c_uint),
        ("nOffsetX", ctypes.c_ushort),
        ("nOffsetY", ctypes.c_ushort),
        ("nChunkWidth", ctypes.c_ushort),
        ("nChunkHeight", ctypes.c_ushort),
        ("nLostPacket", ctypes.c_uint),
        ("nUnparsedChunkNum", ctypes.c_uint),
        ("nReserved", ctypes.c_uint * 36),
    ]


def _bytes_to_str(buf) -> str:
    """将 ctypes 定长 ubyte 数组转换为 Python 字符串 (遇 \\0 截断, ASCII 解码)。"""
    raw = bytes(buf)
    end = raw.find(b"\x00")
    if end >= 0:
        raw = raw[:end]
    return raw.decode("ascii", errors="replace")


# =============================================================================
# 海康 MVS SDK 实现
# =============================================================================
_DEFAULT_HIK_SDK_PATH = r"C:\Program Files (x86)\MVS\Development\Bin\Win64_x64"


class HikvisionCamera(BaseCamera):
    """海康机器人工业相机实现 (基于 MVS SDK, 通过 ctypes 直接调用 C 接口)。

    特性:
        - 自动枚举 GigE / USB3 设备, 支持按 index 或 gigE_ip 选择目标相机;
        - 支持软触发 (software) 与自由采集 (freerun) 两种触发模式;
        - 支持曝光时间配置 (exposure_us);
        - grab() 内部完成 Mono8 / BayerRG8 / YUV422 → BGR 的像素格式转换;
        - SDK 缺失时打印安装指引并返回 False, 由工厂函数兜底回退到 OpenCV。
    """

    def __init__(
        self,
        index: int = 0,
        gigE_ip: str | None = None,
        trigger: str = "software",
        exposure_us: int | None = None,
        sdk_path: str | None = None,
        timeout_ms: int = 3000,
    ):
        self.index = index
        self.gigE_ip = gigE_ip
        self.trigger = (trigger or "freerun").lower()
        self.exposure_us = exposure_us
        self.sdk_path = sdk_path or _DEFAULT_HIK_SDK_PATH
        self.timeout_ms = int(timeout_ms)

        self._dll: ctypes.CDLL | None = None
        self._handle = ctypes.c_void_p(None)
        self._opened = False
        self._grabbing = False
        self._initialized = False
        self._frame_buf: np.ndarray | None = None
        self._frame_info = MV_FRAME_OUT_INFO_EX()

    # ------------------------------------------------------------------ 加载 SDK
    def _load_sdk(self) -> bool:
        """加载 MvCameraControl.dll 并配置常用函数原型。"""
        dll_file = Path(self.sdk_path) / "MvCameraControl.dll"
        if not dll_file.exists():
            logger.error(
                "未找到海康 MVS SDK 动态库: %s\n"
                "  请安装 MVS 客户端 (海康机器人官网 https://www.hikrobotics.com),\n"
                "  或在 config.yaml 的 camera.hik_sdk_path 中指定正确的 Win64_x64 目录。\n"
                "  默认路径: %s",
                dll_file, _DEFAULT_HIK_SDK_PATH,
            )
            return False

        try:
            # 将 SDK 目录加入 DLL 搜索路径 (Python 3.8+ 默认不再搜索 PATH)
            sdk_dir = str(dll_file.parent)
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(sdk_dir)
            os.environ["PATH"] = sdk_dir + os.pathsep + os.environ.get("PATH", "")
            self._dll = ctypes.CDLL(str(dll_file))
        except OSError as e:
            logger.error("加载 MvCameraControl.dll 失败: %s (可能缺少 VC++ 运行库或位数不匹配)", e)
            return False

        self._setup_prototypes()
        return True

    def _setup_prototypes(self) -> None:
        """声明常用 SDK 函数的参数 / 返回类型, 避免 64 位下指针被截断。"""
        d = self._dll
        d.MV_CC_Initialize.restype = ctypes.c_int
        d.MV_CC_Initialize.argtypes = []

        d.MV_CC_Finalize.restype = ctypes.c_int
        d.MV_CC_Finalize.argtypes = []

        d.MV_CC_EnumDevices.restype = ctypes.c_int
        d.MV_CC_EnumDevices.argtypes = [ctypes.c_uint, ctypes.POINTER(MV_CC_DEVICE_INFO_LIST)]

        d.MV_CC_CreateHandle.restype = ctypes.c_int
        d.MV_CC_CreateHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.POINTER(MV_CC_DEVICE_INFO)]

        d.MV_CC_DestroyHandle.restype = ctypes.c_int
        d.MV_CC_DestroyHandle.argtypes = [ctypes.c_void_p]

        d.MV_CC_OpenDevice.restype = ctypes.c_int
        d.MV_CC_OpenDevice.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_ushort]

        d.MV_CC_CloseDevice.restype = ctypes.c_int
        d.MV_CC_CloseDevice.argtypes = [ctypes.c_void_p]

        d.MV_CC_StartGrabbing.restype = ctypes.c_int
        d.MV_CC_StartGrabbing.argtypes = [ctypes.c_void_p]

        d.MV_CC_StopGrabbing.restype = ctypes.c_int
        d.MV_CC_StopGrabbing.argtypes = [ctypes.c_void_p]

        d.MV_CC_GetOneFrameTimeout.restype = ctypes.c_int
        d.MV_CC_GetOneFrameTimeout.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
            ctypes.POINTER(MV_FRAME_OUT_INFO_EX), ctypes.c_uint,
        ]

        d.MV_CC_SetEnumValue.restype = ctypes.c_int
        d.MV_CC_SetEnumValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]

        d.MV_CC_SetFloatValue.restype = ctypes.c_int
        d.MV_CC_SetFloatValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]

        d.MV_CC_SetCommandValue.restype = ctypes.c_int
        d.MV_CC_SetCommandValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        d.MV_CC_GetIntValue.restype = ctypes.c_int
        d.MV_CC_GetIntValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    # ------------------------------------------------------------------ 设备枚举
    def _enum_devices(self) -> list[dict]:
        """枚举所有 GigE / USB3 设备, 返回精简后的设备字典列表。"""
        dev_list = MV_CC_DEVICE_INFO_LIST()
        ret = self._dll.MV_CC_EnumDevices(
            MV_GIGE_DEVICE | MV_USB_DEVICE, ctypes.byref(dev_list)
        )
        if ret != MV_OK:
            logger.error("枚举设备失败: %s", mv_err_str(ret))
            return []

        devices: list[dict] = []
        for i in range(dev_list.nDeviceNum):
            ptr = dev_list.pDeviceInfo[i].pDeviceInfo
            if not ptr:
                continue
            info = ptr.contents
            item = {
                "index": i,
                "ptr": ptr,
                "tlayer": info.nTLayerType,
                "model": "",
                "serial": "",
                "ip": None,
            }
            if info.nTLayerType == MV_GIGE_DEVICE:
                gige = info.SpecialInfo.stGigEInfo
                item["model"] = _bytes_to_str(gige.chModelName)
                item["serial"] = _bytes_to_str(gige.chSerialNumber)
                ip = gige.nCurrentIp
                item["ip"] = f"{(ip >> 24) & 0xFF}.{(ip >> 16) & 0xFF}.{(ip >> 8) & 0xFF}.{ip & 0xFF}"
            elif info.nTLayerType == MV_USB_DEVICE:
                usb = info.SpecialInfo.stUsb3VInfo
                item["model"] = _bytes_to_str(usb.chModelName)
                item["serial"] = _bytes_to_str(usb.chSerialNumber)
            devices.append(item)
        return devices

    def _select_device(self, devices: list[dict]):
        """根据 gigE_ip / index 选择目标设备, 返回设备字典或 None。"""
        if not devices:
            return None
        # 优先按 IP 精确匹配 (仅对 GigE 相机有效)
        if self.gigE_ip:
            for d in devices:
                if d["ip"] == self.gigE_ip:
                    return d
            logger.error("未找到 IP 为 %s 的 GigE 相机, 已枚举: %s",
                         self.gigE_ip, [d["ip"] for d in devices if d["ip"]])
            return None
        # 否则按枚举顺序选择
        if 0 <= self.index < len(devices):
            return devices[self.index]
        logger.error("相机 index=%s 越界, 共枚举到 %d 台设备", self.index, len(devices))
        return None

    # ------------------------------------------------------------------ 参数配置
    def _apply_params(self) -> None:
        """配置触发模式与曝光时间 (失败仅告警, 不阻断采集流程)。"""
        h = self._handle
        if self.trigger == "software":
            # TriggerMode=1 (开启), TriggerSource=7 (Software)
            r = self._dll.MV_CC_SetEnumValue(h, b"TriggerMode", 1)
            if r != MV_OK:
                logger.warning("设置 TriggerMode 失败: %s", mv_err_str(r))
            r = self._dll.MV_CC_SetEnumValue(h, b"TriggerSource", 7)
            if r != MV_OK:
                logger.warning("设置 TriggerSource=Software 失败: %s", mv_err_str(r))
        else:
            # 自由采集: 关闭触发模式
            r = self._dll.MV_CC_SetEnumValue(h, b"TriggerMode", 0)
            if r != MV_OK:
                logger.warning("关闭 TriggerMode 失败: %s", mv_err_str(r))

        if self.exposure_us is not None:
            # 先关闭自动曝光 (ExposureAuto=0), 再设置曝光时间 (微秒)
            self._dll.MV_CC_SetEnumValue(h, b"ExposureAuto", 0)
            r = self._dll.MV_CC_SetFloatValue(h, b"ExposureTime", float(self.exposure_us))
            if r != MV_OK:
                logger.warning("设置曝光时间 %s us 失败: %s",
                               self.exposure_us, mv_err_str(r))

    def _software_trigger(self) -> bool:
        """执行一次软触发, 成功返回 True。"""
        ret = self._dll.MV_CC_SetCommandValue(self._handle, b"TriggerSoftware")
        if ret != MV_OK:
            logger.error("软触发失败: %s", mv_err_str(ret))
            return False
        return True

    # ------------------------------------------------------------------ 公共接口
    def open(self) -> bool:
        if not self._load_sdk():
            return False

        ret = self._dll.MV_CC_Initialize()
        if ret != MV_OK:
            logger.error("MVS SDK 初始化失败: %s", mv_err_str(ret))
            return False
        self._initialized = True

        devices = self._enum_devices()
        if not devices:
            logger.error("未枚举到任何海康 GigE/USB3 相机, 请检查接线 / IP / 网卡巨帧设置")
            self._finalize()
            return False
        logger.info("枚举到 %d 台海康相机: %s", len(devices),
                    [(d["model"], d["ip"] or d["serial"]) for d in devices])

        target = self._select_device(devices)
        if target is None:
            self._finalize()
            return False

        ret = self._dll.MV_CC_CreateHandle(ctypes.byref(self._handle), target["ptr"])
        if ret != MV_OK:
            logger.error("创建句柄失败: %s", mv_err_str(ret))
            self._finalize()
            return False

        ret = self._dll.MV_CC_OpenDevice(self._handle)
        if ret != MV_OK:
            logger.error("打开设备失败: %s (相机可能被其他进程占用)", mv_err_str(ret))
            self._dll.MV_CC_DestroyHandle(self._handle)
            self._handle = ctypes.c_void_p(None)
            self._finalize()
            return False
        self._opened = True

        # GigE 相机优化: 自动协商最佳包大小, 降低丢包率
        if target["tlayer"] == MV_GIGE_DEVICE and hasattr(self._dll, "MV_GIGE_GetOptimalPacketSize"):
            try:
                self._dll.MV_GIGE_GetOptimalPacketSize.restype = ctypes.c_int
                self._dll.MV_GIGE_GetOptimalPacketSize.argtypes = [ctypes.c_void_p]
                pkt = self._dll.MV_GIGE_GetOptimalPacketSize(self._handle)
                if pkt > 0:
                    self._dll.MV_CC_SetIntValue(self._handle, b"GevSCPSPacketSize", pkt)
            except Exception as e:  # noqa: BLE001 - 优化项失败不影响主流程
                logger.debug("协商 GigE 包大小失败 (可忽略): %s", e)

        self._apply_params()

        ret = self._dll.MV_CC_StartGrabbing(self._handle)
        if ret != MV_OK:
            logger.error("启动取流失败: %s", mv_err_str(ret))
            self.close()
            return False
        self._grabbing = True

        logger.info("海康相机已就绪: model=%s ip=%s trigger=%s",
                    target["model"], target["ip"], self.trigger)
        return True

    def grab(self) -> np.ndarray | None:
        if not self._opened or not self._grabbing:
            return None

        # 软触发模式: 每次 grab 主动下发一次触发命令
        if self.trigger == "software" and not self._software_trigger():
            return None

        if self._frame_buf is None:
            # 初始分配: 按 5000 万像素 RGB 估算 (C4: 首次兜底, 后续按需扩容)
            self._frame_buf = np.zeros(5000 * 1024 * 3, dtype=np.uint8)

        ret = self._dll.MV_CC_GetOneFrameTimeout(
            self._handle,
            self._frame_buf.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint(self._frame_buf.nbytes),
            ctypes.byref(self._frame_info),
            ctypes.c_uint(self.timeout_ms),
        )
        # C4 修复: 缓冲区不足时按实际帧大小扩容后重试
        if ret == MV_E_BUF_OVERFLOW or (ret != MV_OK and ret != MV_E_NODATA
                                         and ret != MV_E_TIMEOUT):
            needed = int(self._frame_info.nFrameLen)
            if needed > self._frame_buf.nbytes:
                logger.info("帧缓冲扩容: %d → %d bytes", self._frame_buf.nbytes, needed)
                self._frame_buf = np.zeros(needed + 1024, dtype=np.uint8)
                ret = self._dll.MV_CC_GetOneFrameTimeout(
                    self._handle,
                    self._frame_buf.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_uint(self._frame_buf.nbytes),
                    ctypes.byref(self._frame_info),
                    ctypes.c_uint(self.timeout_ms),
                )

        if ret == MV_E_NODATA or ret == MV_E_TIMEOUT:
            logger.debug("等待帧超时 (%s)", mv_err_str(ret))
            return None
        if ret != MV_OK:
            logger.error("取帧失败: %s", mv_err_str(ret))
            return None

        return self._convert_frame()

    # ------------------------------------------------------------------ 像素格式转换
    def _convert_frame(self) -> np.ndarray | None:
        """将 SDK 原始帧数据转换为 OpenCV 友好的 BGR ndarray。"""
        import cv2

        info = self._frame_info
        w, h = int(info.nWidth), int(info.nHeight)
        if w <= 0 or h <= 0:
            logger.error("帧尺寸非法: %dx%d", w, h)
            return None

        pixel_format = self._read_pixel_format()
        raw = self._frame_buf

        try:
            if pixel_format == PIXEL_MONO8:
                gray = np.frombuffer(raw[: w * h], dtype=np.uint8).reshape(h, w)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if pixel_format in (PIXEL_MONO10, PIXEL_MONO12):
                # 10/12bit 数据按 uint16 读取后右移对齐到 8bit
                u16 = np.frombuffer(raw[: w * h * 2], dtype=np.uint16).reshape(h, w)
                shift = 2 if pixel_format == PIXEL_MONO10 else 4
                gray = (u16 >> shift).astype(np.uint8)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if pixel_format == PIXEL_BAYER_RG8:
                bayer = np.frombuffer(raw[: w * h], dtype=np.uint8).reshape(h, w)
                return cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)

            if pixel_format == PIXEL_BAYER_BG8:
                bayer = np.frombuffer(raw[: w * h], dtype=np.uint8).reshape(h, w)
                return cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)

            if pixel_format == PIXEL_BAYER_GR8:
                bayer = np.frombuffer(raw[: w * h], dtype=np.uint8).reshape(h, w)
                return cv2.cvtColor(bayer, cv2.COLOR_BayerGR2BGR)

            if pixel_format == PIXEL_BAYER_GB8:
                bayer = np.frombuffer(raw[: w * h], dtype=np.uint8).reshape(h, w)
                return cv2.cvtColor(bayer, cv2.COLOR_BayerGB2BGR)

            if pixel_format in (PIXEL_YUV422_8, PIXEL_YUV422_YUYV_Packed):
                yuv = np.frombuffer(raw[: w * h * 2], dtype=np.uint8).reshape(h, w, 2)
                rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)
                return rgb

            if pixel_format == PIXEL_RGB8:
                rgb = np.frombuffer(raw[: w * h * 3], dtype=np.uint8).reshape(h, w, 3)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if pixel_format == PIXEL_BGR8:
                return np.frombuffer(raw[: w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy()

            logger.warning("暂不支持的像素格式 0x%08X, 按 Mono8 兜底处理", pixel_format)
            gray = np.frombuffer(raw[: w * h], dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        except cv2.error as e:
            logger.error("像素格式转换失败 (fmt=0x%08X, %dx%d): %s", pixel_format, w, h, e)
            return None

    def _read_pixel_format(self) -> int:
        """读取当前帧的像素格式 (enPixelType), 异常时回退到 Mono8。"""
        try:
            return int(self._frame_info.enPixelType)
        except Exception:  # noqa: BLE001
            return PIXEL_MONO8

    # ------------------------------------------------------------------ 资源释放
    def _finalize(self) -> None:
        """反初始化 SDK (与 MV_CC_Initialize 配对)。"""
        if self._initialized and self._dll is not None:
            try:
                self._dll.MV_CC_Finalize()
            except Exception:  # noqa: BLE001
                pass
            self._initialized = False

    def close(self) -> None:
        if self._dll is None:
            return
        if self._grabbing:
            self._dll.MV_CC_StopGrabbing(self._handle)
            self._grabbing = False
        if self._opened:
            self._dll.MV_CC_CloseDevice(self._handle)
            self._opened = False
        if self._handle:
            self._dll.MV_CC_DestroyHandle(self._handle)
            self._handle = ctypes.c_void_p(None)
        self._finalize()
        self._dll = None
        self._frame_buf = None
        logger.info("海康相机已关闭")


# =============================================================================
# harvesters 通用 GigE Vision 实现
# =============================================================================
class GigEVisionCamera(BaseCamera):
    """基于 harvesters + GenTL Producer 的通用 GigE Vision 相机实现。

    依赖:
        pip install harvesters numpy

    需要安装任一 GenTL Producer (.cti), 常见来源:
        - 海康 MVS:  C:\\Program Files\\MVS\\Development\\GenTL\\MVS_GenTL.cti
        - 大恒图像:  Galaxy_GenTL.cti
        - Basler:    ProducerGEV.cti
        - FLIR/Teledyne: FLIR_GenTL.cti

    可通过 config["camera"]["gentl_cti"] 显式指定 .cti 路径,
    或设置环境变量 GENICAM_GENTL64_PATH。
    """

    def __init__(
        self,
        index: int = 0,
        gigE_ip: str | None = None,
        trigger: str = "software",
        exposure_us: int | None = None,
        cti_path: str | None = None,
        timeout_ms: int = 3000,
    ):
        self.index = index
        self.gigE_ip = gigE_ip
        self.trigger = (trigger or "freerun").lower()
        self.exposure_us = exposure_us
        self.cti_path = cti_path
        self.timeout_ms = float(timeout_ms)

        self._harvester = None
        self._ia = None  # ImageAcquirer

    def _find_cti(self) -> str | None:
        """定位 GenTL Producer (.cti) 文件。"""
        # 1) 配置显式指定
        if self.cti_path and Path(self.cti_path).exists():
            return self.cti_path
        # 2) 环境变量 GENICAM_GENTL64_PATH (可包含多个目录, 以分号分隔)
        env = os.environ.get("GENICAM_GENTL64_PATH", "")
        for d in env.split(os.pathsep):
            if not d:
                continue
            p = Path(d)
            if p.is_file() and p.suffix.lower() == ".cti":
                return str(p)
            if p.is_dir():
                for cti in p.glob("*.cti"):
                    return str(cti)
        # 3) 扫描常见安装目录
        candidates = [
            r"C:\Program Files\MVS\Development\GenTL",
            r"C:\Program Files (x86)\MVS\Development\GenTL",
            r"C:\Program Files\Basler\GenTL",
            r"C:\Program Files\FLIR\GenTL",
        ]
        for c in candidates:
            p = Path(c)
            if p.is_dir():
                for cti in p.glob("*.cti"):
                    return str(cti)
        return None

    def open(self) -> bool:
        try:
            from harvesters.core import Harvester
        except ImportError:
            logger.error(
                "未安装 harvesters 库, GigE Vision 后端不可用。\n"
                "  安装方式: pip install harvesters\n"
                "  并安装 GenTL Producer (.cti), 例如海康 MVS 自带的 MVS_GenTL.cti。\n"
                "  或将 config.yaml 中 camera.type 改为 'opencv' / 'hikvision'。"
            )
            return False

        cti = self._find_cti()
        if not cti:
            logger.error(
                "未找到 GenTL Producer (.cti) 文件。\n"
                "  请安装相机厂商的 GenTL 驱动, 或在 config.yaml 中配置 camera.gentl_cti,\n"
                "  或设置环境变量 GENICAM_GENTL64_PATH 指向 .cti 所在目录。"
            )
            return False

        try:
            self._harvester = Harvester()
            self._harvester.add_file(cti)
            self._harvester.update()

            device_list = self._harvester.device_info_list
            if not device_list:
                logger.error("harvesters 未枚举到任何 GigE Vision 设备 (cti=%s)", cti)
                return False
            logger.info("harvesters 枚举到 %d 台设备: %s", len(device_list),
                        [d.display_name for d in device_list])

            # 选择目标设备: 优先按 IP 匹配, 否则按 index
            target_idx = self.index
            if self.gigE_ip:
                matched = None
                for i, d in enumerate(device_list):
                    if self.gigE_ip in getattr(d, "display_name", ""):
                        matched = i
                        break
                if matched is None:
                    logger.error("未找到 IP 为 %s 的设备", self.gigE_ip)
                    return False
                target_idx = matched

            self._ia = self._harvester.create_image_acquirer(target_idx)
            self._ia.remote_device.node_map  # 触发节点映射加载

            # 配置曝光
            if self.exposure_us is not None:
                try:
                    self._ia.remote_device.node_map.ExposureAuto.value = "Off"
                    self._ia.remote_device.node_map.ExposureTime.value = float(self.exposure_us)
                except Exception as e:  # noqa: BLE001
                    logger.warning("配置曝光失败 (可能不支持): %s", e)

            # 配置触发
            if self.trigger == "software":
                try:
                    self._ia.remote_device.node_map.TriggerMode.value = "On"
                    self._ia.remote_device.node_map.TriggerSource.value = "Software"
                except Exception as e:  # noqa: BLE001
                    logger.warning("配置软触发失败: %s", e)
            else:
                try:
                    self._ia.remote_device.node_map.TriggerMode.value = "Off"
                except Exception:  # noqa: BLE001
                    pass

            self._ia.start_acquisition()
            logger.info("GigE Vision 相机已就绪 (harvesters, cti=%s)", cti)
            return True

        except Exception as e:  # noqa: BLE001 - harvesters 异常类型多样, 统一兜底
            logger.error("harvesters 打开相机失败: %s", e)
            self.close()
            return False

    def grab(self) -> np.ndarray | None:
        import cv2

        if self._ia is None:
            return None
        try:
            if self.trigger == "software":
                self._ia.remote_device.node_map.TriggerSoftware.execute()

            with self._ia.fetch_buffer(timeout=self.timeout_ms / 1000.0) as buffer:
                arr = buffer.payload.components[0].data
                # 统一转为 3 通道 BGR
                if arr.ndim == 2:
                    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    # harvesters 通常输出 RGB, 转 BGR 以兼容 OpenCV
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return arr
        except Exception as e:  # noqa: BLE001
            logger.debug("harvesters 取帧失败/超时: %s", e)
            return None

    def close(self) -> None:
        try:
            if self._ia is not None:
                try:
                    self._ia.stop_acquisition()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self._ia.destroy()
                except Exception:  # noqa: BLE001
                    pass
                self._ia = None
            if self._harvester is not None:
                try:
                    self._harvester.reset()
                except Exception:  # noqa: BLE001
                    pass
                self._harvester = None
        finally:
            logger.info("GigE Vision 相机已关闭")


# =============================================================================
# 工厂函数
# =============================================================================
def create_camera(config: dict) -> BaseCamera:
    """根据 config["camera"]["type"] 创建对应相机实例。

    支持的 type:
        - "opencv"      : OpenCVCamera (默认, USB 摄像头)
        - "hikvision"   : HikvisionCamera (海康 MVS SDK)
        - "gigevision"  : GigEVisionCamera (harvesters + GenTL)

    未知 type 会记录告警并回退到 OpenCV。
    """
    cam_cfg = (config or {}).get("camera", {}) or {}
    cam_type = str(cam_cfg.get("type", "opencv")).lower()

    if cam_type == "hikvision":
        return HikvisionCamera(
            index=cam_cfg.get("index", 0),
            gigE_ip=cam_cfg.get("gigE_ip"),
            trigger=cam_cfg.get("trigger", "software"),
            exposure_us=cam_cfg.get("exposure_us"),
            sdk_path=cam_cfg.get("hik_sdk_path"),
            timeout_ms=cam_cfg.get("timeout_ms", 3000),
        )

    if cam_type == "gigevision":
        return GigEVisionCamera(
            index=cam_cfg.get("index", 0),
            gigE_ip=cam_cfg.get("gigE_ip"),
            trigger=cam_cfg.get("trigger", "software"),
            exposure_us=cam_cfg.get("exposure_us"),
            cti_path=cam_cfg.get("gentl_cti"),
            timeout_ms=cam_cfg.get("timeout_ms", 3000),
        )

    if cam_type != "opencv":
        logger.warning("未知的相机类型 '%s', 回退到 opencv", cam_type)

    return OpenCVCamera(
        index=cam_cfg.get("index", 0),
        width=cam_cfg.get("width", 1920),
        height=cam_cfg.get("height", 1080),
    )
