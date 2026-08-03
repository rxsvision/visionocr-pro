"""PP-OCRv6 引擎 - Docker/子进程隔离 (Windows 主力 OCR)

架构:
- Windows: paddle 3.x 存在不可修复的 PIR+OneDNN bug + torch DLL 冲突,
  必须通过 Docker 容器 (Linux) 运行 PaddleOCR PP-OCRv6。
- Linux/Jetson: paddle 原生可用, 走子进程隔离 (避免 torch 同进程冲突)。

Docker 两种模式 (v1.3.0+):
- server (默认): 常驻容器内跑 FastAPI 服务 (docker/paddle_server.py),
  模型加载一次常驻, 推理 HTTP 往返, 亚秒级响应 (旧模式每次 5~13s)。
  宿主机引擎负责容器启停与健康检查; unload() 时停止容器释放资源。
- run (降级): 每次推理 docker run --rm 新建容器 (旧行为),
  当镜像过旧 (未含 paddle_server.py) 或服务启动失败时自动降级,
  保证升级期间零中断。

通信协议 (两模式一致):
- 引擎 → Docker/subprocess: 图像字节 (server) 或图像路径 (run/子进程)
- Docker/subprocess → 引擎: JSON
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
import time
from pathlib import Path
from typing import Any

import requests

from engines.base import BaseEngine, EngineMeta, EngineState

logger = logging.getLogger("visionocr.ppocrv6")

_DOCKER_IMAGE = "visionocr-paddleocr"
_DEFAULT_TIMEOUT = 120  # 秒 (单次推理)
_SERVER_PORT = 8686  # 常驻服务宿主映射端口 (127.0.0.1 only)
_CONTAINER_NAME = "visionocr-paddle-serve"
_STARTUP_TIMEOUT = 120  # 秒 (容器启动 + 模型加载 + warmup)


class PPOCRv6Engine(BaseEngine):
    """PP-OCRv6 工业 OCR (Docker 隔离, 零漏检主力)"""

    def __init__(self, config: dict):
        super().__init__(config)
        self._backend: str = ""  # "docker" | "subprocess"
        self._docker_mode: str = ""  # "server" | "run" (仅 docker backend)
        self._docker_available: bool = False
        self._use_gpu: bool = True
        self._timeout = _DEFAULT_TIMEOUT
        self._docker_image = _DOCKER_IMAGE
        self._port = _SERVER_PORT
        self._container_name = _CONTAINER_NAME
        self._startup_timeout = _STARTUP_TIMEOUT

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
        """检测可用后端: Docker 优先 (常驻服务→单次降级), 子进程兜底 (Linux)"""
        self.state = EngineState.LOADING

        ocr_cfg = (self.config or {}).get("ocr", {})
        pp_cfg = ocr_cfg.get("ppocrv6", {})
        self._use_gpu = pp_cfg.get("gpu", True)
        self._timeout = pp_cfg.get("timeout", _DEFAULT_TIMEOUT)
        self._docker_image = pp_cfg.get("docker_image", _DOCKER_IMAGE)
        self._port = int(pp_cfg.get("port", _SERVER_PORT))
        self._container_name = pp_cfg.get("container_name", _CONTAINER_NAME)
        self._startup_timeout = int(
            pp_cfg.get("startup_timeout", _STARTUP_TIMEOUT))

        # 1) 检测 Docker
        if self._check_docker():
            self._backend = "docker"
            self._docker_available = True
            # v1.3.0 P0-2: 优先常驻服务容器; 失败自动降级单次调用
            if self._start_server():
                self._docker_mode = "server"
                self.state = EngineState.READY
                logger.info(
                    "PP-OCRv6 就绪 (Docker 常驻服务: 127.0.0.1:%d, GPU=%s)",
                    self._port, self._use_gpu)
            else:
                self._docker_mode = "run"
                self.state = EngineState.READY
                logger.warning(
                    "PP-OCRv6 常驻服务不可用, 降级单次 docker run 模式 "
                    "(每次 5~13s 开销)。请重新构建镜像: docker build "
                    "-f docker/Dockerfile.paddleocr -t %s .",
                    self._docker_image)
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
        """server 模式: 停止常驻容器释放资源; run 模式: 容器即用即销无需操作"""
        if self._backend == "docker" and self._docker_mode == "server":
            self._stop_server()
        self._backend = ""
        self._docker_mode = ""
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

    # ─── Docker 推理 (分发) ──────────────────────────────────
    def _infer_docker(self, image_path: str) -> dict:
        """Docker backend 分发: server 模式 (常驻) 优先, run 模式 (单次) 兜底"""
        if self._docker_mode == "server":
            return self._infer_server(image_path)
        return self._infer_docker_run(image_path)

    # ─── 常驻服务推理 (v1.3.0 P0-2) ─────────────────────────
    @property
    def _server_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def _infer_server(self, image_path: str) -> dict:
        """通过常驻容器 HTTP 服务执行 OCR (亚秒级响应)。

        连接失败时自动尝试重启容器一次 (服务崩溃/被外部停止的自愈)。
        """
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
        except OSError as e:
            return self._empty(f"读取图像失败: {e}")

        name = os.path.basename(image_path)
        for attempt in (1, 2):
            try:
                resp = requests.post(
                    f"{self._server_url}/ocr",
                    files={"file": (name, img_bytes)},
                    timeout=self._timeout,
                )
                data = resp.json()
                if "error" in data:
                    return self._empty(data["error"])
                data["engine"] = "ppocrv6"
                data["backend"] = "docker"
                return data
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == 1:
                    logger.warning(
                        "PP-OCRv6 常驻服务连接失败 (%s), 尝试重启容器...", e)
                    self._stop_server()
                    if not self._start_server():
                        return self._empty(
                            "常驻服务连接失败且重启失败, "
                            "请检查 Docker Desktop 与镜像版本")
                else:
                    return self._empty(f"常驻服务重启后仍连接失败: {e}")
            except requests.RequestException as e:
                return self._empty(f"常驻服务请求异常: {e}")
            except ValueError:
                return self._empty("常驻服务返回非 JSON 响应")
        return self._empty("常驻服务推理失败 (未知)")

    # ─── 常驻容器管理 ────────────────────────────────────────
    def _start_server(self) -> bool:
        """确保常驻服务容器运行且健康。

        流程: 已运行→轮询健康; 否则 docker run -d 启动→轮询健康。
        失败时收集容器日志并清理, 返回 False。
        """
        if self._container_running():
            return self._wait_healthy()

        # 清理残留 (已退出的同名容器)
        subprocess.run(["docker", "rm", "-f", self._container_name],
                       capture_output=True, timeout=30)

        model_cache = Path.home() / ".paddlex"
        model_cache.mkdir(parents=True, exist_ok=True)

        cmd = [
            "docker", "run", "-d",
            "--name", self._container_name,
            # GPU 基础镜像需要 libcuda.so.1 即使跑 CPU 推理, 始终挂载 GPU
            "--gpus", "all",
            "--restart", "unless-stopped",
            "-p", f"127.0.0.1:{self._port}:8000",
            "-v", f"{model_cache}:/root/.paddlex",
            "--entrypoint", "python",
            self._docker_image,
            "paddle_server.py",
            "--device", "gpu" if self._use_gpu else "cpu",
            "--port", "8000",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60)
            if r.returncode != 0:
                logger.warning(
                    "常驻容器启动失败: %s",
                    r.stderr.decode("utf-8", errors="replace").strip()[-300:])
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning("常驻容器启动异常: %s", e)
            return False

        if self._wait_healthy():
            return True

        # 启动失败: 收集日志便于诊断, 然后清理
        try:
            logs = subprocess.run(
                ["docker", "logs", "--tail", "30", self._container_name],
                capture_output=True, timeout=15)
            tail = logs.stderr.decode("utf-8", errors="replace").strip()
            if tail:
                logger.warning("常驻容器日志:\n%s", tail[-1000:])
        except Exception:
            pass
        self._stop_server()
        return False

    def _stop_server(self) -> None:
        """停止并移除常驻容器 (失败不抛异常)。"""
        try:
            subprocess.run(["docker", "rm", "-f", self._container_name],
                           capture_output=True, timeout=30)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug("停止常驻容器失败: %s", e)

    def _container_running(self) -> bool:
        """检查常驻容器是否在运行。"""
        try:
            r = subprocess.run(
                ["docker", "ps", "--filter",
                 f"name=^{self._container_name}$",
                 "--format", "{{.Names}}"],
                capture_output=True, timeout=10,
            )
            names = r.stdout.decode("utf-8", errors="replace").split()
            return self._container_name in names
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _wait_healthy(self) -> bool:
        """轮询 /health 直到 ready 或超时 (模型加载 + warmup 需要时间)。"""
        deadline = time.time() + self._startup_timeout
        while time.time() < deadline:
            if self._health_ok():
                return True
            time.sleep(2)
        logger.warning("常驻服务健康检查超时 (%ds)", self._startup_timeout)
        return False

    def _health_ok(self) -> bool:
        """单次健康探测: GET /health → ready=True"""
        try:
            r = requests.get(f"{self._server_url}/health", timeout=3)
            return r.status_code == 200 and r.json().get("ready") is True
        except (requests.RequestException, ValueError):
            return False

    # ─── 单次容器推理 (旧模式, 降级兜底) ─────────────────────
    def _infer_docker_run(self, image_path: str) -> dict:
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
