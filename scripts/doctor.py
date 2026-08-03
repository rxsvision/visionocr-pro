# -*- coding: utf-8 -*-
"""VisionOCR Pro 环境自检 (doctor)

用法:
    .venv\\Scripts\\python.exe scripts/doctor.py        # 完整自检
    .venv\\Scripts\\python.exe scripts/doctor.py --quick  # 快速模式 (跳过重依赖导入)

设计原则 (随身插拔式):
    - 本脚本只读不写, 任何机器 clone 仓库后可直接运行诊断;
    - 重依赖 (torch/transformers/paddle/gradio) 缺失只报 WARN —— 代码内均为
      函数级懒加载 + ImportError 容错, 缺失仅降级对应功能;
    - 核心依赖 (yaml/numpy/cv2) 或 config.yaml 加载失败报 FAIL (exit 1)。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (模块, 级别) 级别: core=缺失即 FAIL, heavy=缺失仅 WARN
CHECKS: list[tuple[str, str]] = [
    ("yaml", "core"),
    ("numpy", "core"),
    ("cv2", "core"),
    ("PIL", "core"),
    ("requests", "core"),
    ("openpyxl", "core"),
    ("torch", "heavy"),
    ("torchvision", "heavy"),
    ("transformers", "heavy"),
    ("accelerate", "heavy"),
    ("sklearn", "heavy"),
    ("paddle", "heavy"),
    ("paddleocr", "heavy"),
    ("onnxruntime", "heavy"),
    ("gradio", "heavy"),
    ("pyzbar", "heavy"),
    ("datasette", "heavy"),
]

OK, WARN, FAIL = "[OK]  ", "[WARN]", "[FAIL]"


def _importable(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def main() -> int:
    quick = "--quick" in sys.argv
    failures: list[str] = []
    warnings: list[str] = []

    print("=" * 60)
    print("  VisionOCR Pro doctor - 环境自检")
    print("=" * 60)

    # 1. Python 版本
    v = sys.version_info
    ver_ok = (3, 11) <= (v.major, v.minor) <= (3, 13)
    print(f"{OK if ver_ok else FAIL} Python {v.major}.{v.minor}.{v.micro} @ {sys.executable}")
    if not ver_ok:
        failures.append(f"Python 版本 {v.major}.{v.minor} 不在支持范围 3.11-3.13")

    # 2. 解释器位置提示 (裸 python 陷阱警示)
    exe = Path(sys.executable).resolve()
    in_venv = exe.parent.parent == (REPO_ROOT / ".venv").resolve() or (
        sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    )
    if in_venv:
        print(f"{OK} 运行于虚拟环境 (venv 前缀: {sys.prefix})")
    else:
        print(f"{WARN} 当前解释器不在仓库 .venv 内 —— 请优先使用 "
              f".venv\\Scripts\\python.exe, 避免依赖漂移")
        warnings.append("解释器不在仓库 .venv")

    # 3. 依赖导入检查
    print("-" * 60)
    for mod, level in CHECKS:
        if quick and level == "heavy":
            continue
        if _importable(mod):
            print(f"{OK} import {mod}")
        elif level == "core":
            print(f"{FAIL} import {mod} —— 核心依赖缺失")
            failures.append(f"核心依赖缺失: {mod}")
        else:
            print(f"{WARN} import {mod} —— 可选/重依赖缺失, 对应功能降级")
            warnings.append(f"重依赖缺失: {mod}")

    # 4. config.yaml 加载
    print("-" * 60)
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from core.config import load_config  # noqa: E402

        cfg = load_config()
        models_dir = cfg.get("models_dir")
        data_dir = cfg.get("data_dir")
        print(f"{OK} config.yaml 加载成功 (profile 覆盖/env 替换正常)")
        if models_dir:
            exists = Path(models_dir).is_dir()
            print(f"{OK if exists else WARN} models_dir: {models_dir}"
                  f"{'' if exists else ' (目录不存在, 首次运行会自动创建)'}")
            if not exists:
                warnings.append("models_dir 不存在")
        if data_dir:
            print(f"{OK} data_dir: {data_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} config.yaml 加载失败: {exc}")
        failures.append(f"config 加载失败: {exc}")

    # 5. Ollama (可选)
    print("-" * 60)
    if _on_path("ollama"):
        print(f"{OK} ollama 在 PATH (VLM 解释/复核可用)")
    else:
        print(f"{WARN} ollama 不在 PATH —— VLM 辅助功能不可用 (可选)")
        warnings.append("ollama 未安装")

    # 汇总
    print("=" * 60)
    if failures:
        print(f"  结果: FAIL ({len(failures)} 项致命, {len(warnings)} 项警告)")
        for f in failures:
            print(f"    - {f}")
        print("  修复: 运行 setup.bat (Windows) 或 setup.sh (Linux)")
        return 1
    print(f"  结果: PASS ({len(warnings)} 项警告)")
    for w in warnings:
        print(f"    - {w}")
    return 0


def _on_path(cmd: str) -> bool:
    exts = os.environ.get("PATHEXT", "").split(os.pathsep) or [""]
    for d in os.environ.get("PATH", "").split(os.pathsep):
        for ext in exts:
            if Path(d, cmd + ext).is_file():
                return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
