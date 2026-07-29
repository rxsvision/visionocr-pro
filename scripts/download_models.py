"""模型下载管理工具

用法:
    python scripts/download_models.py              # 下载所有推荐模型
    python scripts/download_models.py ovisocr2     # 仅下载 OvisOCR2
    python scripts/download_models.py --status     # 查看本地模型状态

支持模型:
    ovisocr2    - 印刷文档/表格 OCR (ATH-MaaS/OvisOCR2, ~1.5GB, 需 ~5GB VRAM)
    hunyuan     - 手写体 OCR (tencent/HunyuanOCR, ~14GB, 需 ~12GB VRAM)

下载源优先级: HF镜像 → HuggingFace 直连
"""
import os
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# 模型注册表
MODEL_REGISTRY = {
    "ovisocr2": {
        "repo_id": "ATH-MaaS/OvisOCR2",
        "local_dir": "ovis-ocr2",
        "description": "印刷文档/表格/公式 OCR (qwen3_5 架构, ~1.5GB)",
        "vram_gb": 5.0,
        "ignore_patterns": ["*.png", "*.gitattributes"],
    },
    "hunyuan": {
        "repo_id": "tencent/HunyuanOCR",
        "local_dir": "hunyuan-ocr",
        "description": "手写体/复杂版式 OCR (需 ~12GB VRAM, 4070Ti 勉强)",
        "vram_gb": 12.0,
        "ignore_patterns": ["*.png", "*.gitattributes", "v1.0/*", "assets/*"],
    },
}


def setup_hf_mirror():
    """设置 HF 镜像 (国内加速)"""
    mirror = "https://hf-mirror.com"
    os.environ.setdefault("HF_ENDPOINT", mirror)
    print(f"[Mirror] HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")


def check_status():
    """检查本地模型状态"""
    print(f"\n模型目录: {MODELS_DIR}")
    print("-" * 60)
    for key, info in MODEL_REGISTRY.items():
        local_path = MODELS_DIR / info["local_dir"]
        if local_path.is_dir():
            files = list(local_path.iterdir())
            has_weights = any(
                f.suffix == ".safetensors" or f.name == "pytorch_model.bin"
                for f in files
            )
            has_config = (local_path / "config.json").exists()
            if has_weights and has_config:
                size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
                print(f"  [OK] {key:12s} → {info['local_dir']} ({size_mb:.0f} MB)")
            else:
                print(f"  [!!] {key:12s} → 目录存在但不完整 (缺权重或config)")
        else:
            print(f"  [--] {key:12s} → 未下载")
    print()


def download_model(key: str):
    """下载指定模型"""
    info = MODEL_REGISTRY.get(key)
    if info is None:
        print(f"未知模型: {key}, 可选: {list(MODEL_REGISTRY.keys())}")
        return False

    local_path = MODELS_DIR / info["local_dir"]
    if local_path.is_dir() and (local_path / "config.json").exists():
        print(f"[Skip] {key} 已存在: {local_path}")
        return True

    print(f"[Download] {key}: {info['repo_id']} → {local_path}")
    print(f"           {info['description']}")

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=info["repo_id"],
            local_dir=str(local_path),
            ignore_patterns=info.get("ignore_patterns", []),
        )
        print(f"[Done] {key} → {path}")
        return True
    except KeyboardInterrupt:
        print(f"\n[Interrupted] {key} 下载中断, 重新运行可续传")
        return False
    except Exception as e:
        print(f"[Error] {key} 下载失败: {e}")
        print("  建议: 检查网络, 或手动下载后放入对应目录")
        return False


def main():
    setup_hf_mirror()

    if "--status" in sys.argv:
        check_status()
        return

    targets = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not targets:
        # 默认只下载 ovisocr2 (适合 4070 Ti)
        targets = ["ovisocr2"]
        print("未指定模型, 默认下载 ovisocr2 (适合 RTX 4070 Ti)")
        print("如需全部: python scripts/download_models.py ovisocr2 hunyuan\n")

    results = {}
    for key in targets:
        results[key] = download_model(key)

    print("\n" + "=" * 40)
    for key, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {key}: {status}")
    check_status()


if __name__ == "__main__":
    main()
