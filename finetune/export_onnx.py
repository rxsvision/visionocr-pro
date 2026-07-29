"""导出训练模型为 ONNX → 替换 RapidOCR 引擎权重

流程:
    Paddle inference model (.pdmodel + .pdiparams)
        → paddle2onnx → ONNX (.onnx)
        → 复制到 rapidocr 模型目录 → 热替换

用法:
    python finetune/export_onnx.py                           # 导出最新训练结果
    python finetune/export_onnx.py --model output/inference  # 指定模型目录
    python finetune/export_onnx.py --deploy                  # 导出并部署到 rapidocr

前置:
    pip install paddle2onnx
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
OUTPUT_DIR = ROOT / "output"
DEPLOY_DIR = ROOT / "deployed_onnx"  # 导出的 ONNX 存放处


def find_inference_model(model_dir: Path | None = None) -> Path | None:
    """查找 Paddle 推理模型目录。"""
    candidates = []
    if model_dir:
        candidates.append(model_dir)
    candidates.extend([
        OUTPUT_DIR / "inference",
        OUTPUT_DIR / "rec_industrial" / "best_accuracy" / "inference",
    ])

    for cand in candidates:
        if cand.is_dir():
            # Paddle 推理模型标志: .pdmodel 或 inference.pdmodel
            pdmodels = list(cand.glob("*.pdmodel"))
            if pdmodels:
                return cand
            # 也可能是训练 checkpoint 需要先 export
            pdparams = list(cand.glob("*.pdparams"))
            if pdparams:
                print(f"[Info] 找到训练 checkpoint: {cand}")
                print(f"  需先导出推理模型: python tools/export_model.py")
                return cand
    return None


def export_to_onnx(model_dir: Path) -> Path | None:
    """使用 paddle2onnx 导出 ONNX。"""
    pdmodels = list(model_dir.glob("*.pdmodel"))
    if not pdmodels:
        print(f"[Error] 未找到 .pdmodel 文件: {model_dir}")
        return None

    pdmodel = pdmodels[0]
    pdiparams = pdmodel.with_suffix(".pdiparams")
    if not pdiparams.exists():
        print(f"[Error] 未找到权重文件: {pdiparams}")
        return None

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = DEPLOY_DIR / "rec_industrial.onnx"

    cmd = [
        sys.executable, "-m", "paddle2onnx",
        "--model_dir", str(model_dir),
        "--model_filename", pdmodel.name,
        "--params_filename", pdiparams.name,
        "--save_file", str(onnx_path),
        "--opset_version", "16",
        "--enable_onnx_checker", "True",
    ]

    print(f"[Export] paddle2onnx ...")
    print(f"  输入: {pdmodel}")
    print(f"  输出: {onnx_path}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and onnx_path.exists():
            size_mb = onnx_path.stat().st_size / 1024 / 1024
            print(f"[OK] ONNX 导出成功: {onnx_path} ({size_mb:.1f} MB)")
            return onnx_path
        else:
            print(f"[Error] paddle2onnx 失败:")
            print(f"  {proc.stderr[:500]}")
            return None
    except FileNotFoundError:
        print("[Error] paddle2onnx 未安装: pip install paddle2onnx")
        return None
    except subprocess.TimeoutExpired:
        print("[Error] 导出超时")
        return None


def deploy_to_rapidocr(onnx_path: Path):
    """将 ONNX 模型部署到 RapidOCR 引擎。

    RapidOCR 支持自定义模型路径, 通过 config 或环境变量指定。
    """
    # 方案: 在 config.yaml 中配置自定义 rec 模型路径
    # rapidocr_onnxruntime 支持 rec_model_path 参数
    print(f"\n[Deploy] ONNX 模型已就绪: {onnx_path}")
    print(f"\n  部署方式 (二选一):")
    print(f"  1. 代码级: 修改 engines/ocr/rapidocr.py 的 RapidOCR() 初始化:")
    print(f"     RapidOCR(rec_model_path=r'{onnx_path}')")
    print(f"  2. 配置级: 在 config.yaml 添加:")
    print(f"     ocr:")
    print(f"       custom_rec_model: \"{onnx_path}\"")
    print(f"\n  注意: 部署前请先用 evaluate.py 确认精度达标")


def main():
    parser = argparse.ArgumentParser(description="导出 ONNX 模型")
    parser.add_argument("--model", type=str, default=None,
                        help="Paddle 推理模型目录")
    parser.add_argument("--deploy", action="store_true",
                        help="导出后显示部署指引")
    args = parser.parse_args()

    print("=== ONNX 导出 ===\n")

    model_dir = find_inference_model(
        Path(args.model) if args.model else None
    )
    if model_dir is None:
        print("[Error] 未找到可导出的模型")
        print(f"  请先完成训练: python finetune/train.py")
        print(f"  然后导出推理模型:")
        print(f"    cd finetune/PaddleOCR")
        print(f"    python tools/export_model.py -c {ROOT / 'configs' / 'rec_industrial_finetune.yaml'} -o Global.pretrained_model={OUTPUT_DIR / 'rec_industrial' / 'best_accuracy'} Global.save_inference_dir={OUTPUT_DIR / 'inference'}")
        sys.exit(1)

    print(f"模型目录: {model_dir}\n")

    onnx_path = export_to_onnx(model_dir)
    if onnx_path and args.deploy:
        deploy_to_rapidocr(onnx_path)

    sys.exit(0 if onnx_path else 1)


if __name__ == "__main__":
    main()
