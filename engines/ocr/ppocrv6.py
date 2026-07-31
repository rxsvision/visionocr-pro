"""PP-OCRv6 引擎 - Docker/子进程隔离 (Windows 主力 OCR)

架构:
- Windows: paddle 3.x 存在不可修复的 PIR+OneDNN bug + torch DLL 冲突,
  必须通过 Docker 容器 (Linux) 运行 PaddleOCR PP-OCRv6。
- Linux/Jetson: paddle 原生可用, 走子进程隔离 (避免 torch 同进程冲突)。

通信协议:
- 引擎 → Docker/subprocess: 传入图像路径 + 参数
- Docker/subprocess → 引擎: JSON stdout (ensure_ascii=True)
  {"text": "...", "lines": [...], "confidence": 0.95, "engine": "ppocrv6"}

精度基准 (60张工业合成图, CPU):
  PP-OCRv6: 93.3% exact match, CER 0.021, 117ms/img
  RapidOCR: 56.7% exact match, CER 0.332, 927ms/img

依赖:
- Docker 模式: Docker Desktop + visionocr-paddleocr 镜像
  构建: docker build -f docker/Dockerfile.paddleocr -t visionocr-paddleocr .
- 子进程模式: paddleocr>=3.7 (Linux only)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.ppocrv6")

_DOCKER_IMAGE = "visionocr-paddleocr"
_DEFAULT_TIMEOUT = 120  # 秒


class PPOCRv6Engine(BaseEngine):
    """PP-OCRv6 工业 OCR (Docker 隔离, 零漏检主力)"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._backend: str = ""  # "docker" | "subprocess"
        self._docker_available: bool = False
        self._use_gpu: bool = True

    @property
    def meta(self) -> EngineMeta:
        return EngineMeta(
            name="ppocrv6",
            display_name="PP-OCRv6 (Docker·工业主力)",
            category="ocr",
            vram_gb=0.0,  # 显存在容器内, 主机不占用
            license="Apache-2.0",
            description="工业 OCR 主力, 93.3% 精确匹配, Docker 隔离运行",
            tags=["OCR", "工业", "PP-OCRv6", "Docker", "主力"],
        )

    # ─── 生命周期 ────────────────────────────────────────────
    def load(self) -> None:
        """检测可用后端: Docker 优先, 子进程兜底 (Linux)"""
        self.state = EngineState.LOADING

        ocr_cfg = (self.config or {}).get("ocr", {})
        pp_cfg = ocr_cfg.get("ppocrv6", {})
        self._use_gpu = pp_cfg.get("gpu", True)
        self._timeout = pp_cfg.get("timeout", _DEFAULT_TIMEOUT)
        self._docker_image = pp_cfg.get("docker_image", _DOCKER_IMAGE)

        # 1) 检测 Docker
        if self._check_docker():
            self._backend = "docker"
            self._docker_available = True
            self.state = EngineState.READY
            logger.info("PP-OCRv6 就绪 (Docker: %s, GPU=%s)",
                        self._docker_image, self._use_gpu)
            return

        # 2) Linux 子进程模式 (paddle 原生可用)
        if sys.platform != "win32" and self._check_subprocess():
            self._backend = "subprocess"
            self.state = EngineState.READY
            logger.info("PP-OCRv6 就绪 (子进程, Linux 原生)")
            return

        # 3) 都不可用
        self.state = EngineState.ERROR
        logger.error(
            "PP-OCRv6 不可用: Docker 未运行或镜像未构建。"
            "请执行: docker build -f docker/Dockerfile.paddleocr "
            "-t %s .", self._docker_image
        )

    def unload(self) -> None:
        """Docker 模式无需卸载 (容器即用即销)"""
        self._backend = ""
        self._docker_available = False
        self.state = EngineState.UNLOADED

    # ─── 推理 ────────────────────────────────────────────────
    def infer(self, image_path: str, **kwargs: Any) -> dict:
        """对单张图片执行 PP-OCRv6 OCR。

        Returns:
            {"text": str, "lines": [...], "confidence": float,
             "engine": "ppocrv6", "backend": "docker"|"subprocess"}
        """
        if not self.is_ready():
            return self._empty("引擎未就绪, 请先调用 load()")
        if not image_path or not os.path.isfile(image_path):
            return self._empty(f"图片不存在: {image_path}")

        if self._backend == "docker":
            return self._infer_docker(image_path)
        if self._backend == "subprocess":
            return self._infer_subprocess(image_path)
        return self._empty("无可用后端")

    # ─── Docker 推理 ─────────────────────────────────────────
    def _infer_docker(self, image_path: str) -> dict:
        """通过 Docker 容器执行 OCR (Windows 主力路径)"""
        abs_path = os.path.abspath(image_path)
        img_dir = os.path.dirname(abs_path)
        img_name = os.path.basename(abs_path)

        # Docker volume mount: 将图像所在目录挂载到 /data
        # Windows: -v "D:\images:/data" → 容器内 /data/test.png
        mount_src = img_dir
        container_path = f"/data/{img_name}"

        cmd = ["docker", "run", "--rm"]

        # GPU 基础镜像需要 libcuda.so.1 即使跑 CPU 推理,
        # 因此始终挂载 GPU; worker --device 控制实际计算设备
        cmd += ["--gpus", "all"]

        # 模型缓存持久化: 首次下载后复用, 避免每次重新下载 (~35MB)
        model_cache = Path.home() / ".paddlex"
        model_cache.mkdir(parents=True, exist_ok=True)
        cmd += ["-v", f"{model_cache}:/root/.paddlex"]

        cmd += [
            "-v", f"{mount_src}:/data",
            self._docker_image,
            container_path,
            "--device", "gpu" if self._use_gpu else "cpu",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self._timeout,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                # 尝试从 stdout 解析错误 JSON
                err_msg = self._parse_error(stdout, proc.stderr, proc.returncode)
                return self._empty(err_msg)

            return self._parse_output(stdout)

        except subprocess.TimeoutExpired:
            return self._empty(
                f"Docker 推理超时 ({self._timeout}s), 图像: {img_name}")
        except FileNotFoundError:
            return self._empty("docker 命令不可用, 请确认 Docker Desktop 已启动")
        except Exception as e:
            return self._empty(f"Docker 调用异常: {e}")

    # ─── 子进程推理 (Linux) ──────────────────────────────────
    def _infer_subprocess(self, image_path: str) -> dict:
        """通过子进程调用 _paddle_worker.py (Linux 原生 paddle)"""
        worker = Path(__file__).parent / "_paddle_worker.py"
        if not worker.exists():
            return self._empty(f"Worker 脚本缺失: {worker}")

        device = "gpu" if self._use_gpu else "cpu"
        cmd = [
            sys.executable, str(worker), image_path,
            "--device", device,
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=self._timeout,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                err_msg = self._parse_error(stdout, proc.stderr, proc.returncode)
                return self._empty(err_msg)

            return self._parse_output(stdout)

        except subprocess.TimeoutExpired:
            return self._empty(f"子进程推理超时 ({self._timeout}s)")
        except Exception as e:
            return self._empty(f"子进程调用异常: {e}")

    # ─── 检测后端可用性 ──────────────────────────────────────
    def _check_docker(self) -> bool:
        """检测 Docker 是否可用 + 镜像是否存在"""
        if shutil.which("docker") is None:
            logger.debug("docker 命令不在 PATH")
            return False
        try:
            # 检查 Docker daemon
            r = subprocess.run(
                ["docker", "info"],
                capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                logger.debug("Docker daemon 未运行")
                return False

            # 检查镜像是否存在
            r = subprocess.run(
                ["docker", "image", "inspect", self._docker_image],
                capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                logger.warning(
                    "Docker 镜像 '%s' 不存在, 请构建: "
                    "docker build -f docker/Dockerfile.paddleocr -t %s .",
                    self._docker_image, self._docker_image
                )
                return False

            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug("Docker 检测失败: %s", e)
            return False

    def _check_subprocess(self) -> bool:
        """检测子进程模式 (Linux: paddleocr 已装)"""
        worker = Path(__file__).parent / "_paddle_worker.py"
        if not worker.exists():
            return False
        try:
            r = subprocess.run(
                [sys.executable, "-c", "import paddleocr"],
                capture_output=True, timeout=15,
            )
            return r.returncode == 0
        except Exception:
            return False

    # ─── 输出解析 ────────────────────────────────────────────
    def _parse_output(self, stdout: str) -> dict:
        """解析 worker JSON 输出, 标记引擎名和后端"""
        if not stdout:
            return self._empty("Docker/子进程无输出")

        # worker 可能在 JSON 前输出日志行, 取最后一个有效 JSON
        lines = stdout.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    if "error" in data:
                        return self._empty(data["error"])
                    # 统一引擎标识
                    data["engine"] = "ppocrv6"
                    data["backend"] = self._backend
                    return data
                except json.JSONDecodeError:
                    continue

        return self._empty(f"无法解析输出: {stdout[:200]}")

    def _parse_error(self, stdout: str, stderr: bytes,
                     returncode: int) -> str:
        """从 stdout/stderr 提取错误信息"""
        # 先尝试 stdout JSON
        if stdout:
            for line in reversed(stdout.strip().split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        if "error" in data:
                            return data["error"]
                    except json.JSONDecodeError:
                        continue

        # 退化到 stderr
        err_text = stderr.decode("utf-8", errors="replace").strip()
        if err_text:
            # 截取最后 300 字符 (通常包含关键错误)
            return f"rc={returncode}: {err_text[-300:]}"
        return f"未知错误 (rc={returncode})"

    def _empty(self, error: str) -> dict:
        return {
            "text": "",
            "lines": [],
            "confidence": 0.0,
            "engine": "ppocrv6",
            "error": error,
        }
