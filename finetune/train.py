"""PP-OCRv6 识别模型 fine-tune 训练编排器

设计:
    - 训练通过 PaddleOCR Git 仓库的 tools/train.py 执行 (只需 paddle, 不需 paddleocr 包)
    - 子进程隔离: 避免 torch/paddle CUDA DLL 冲突
    - 自动克隆 PaddleOCR 仓库 (shallow, ~50MB)
    - 生成适配的训练配置 (基于 PP-OCRv5 rec 配置)

用法:
    python finetune/train.py                    # 使用默认配置训练
    python finetune/train.py --epochs 100       # 自定义 epoch
    python finetune/train.py --pretrained path  # 指定预训练权重
    python finetune/train.py --dry-run          # 仅生成配置, 不执行训练

前置条件:
    1. python finetune/prepare_data.py 完成数据准备
    2. PaddlePaddle GPU 已安装 (paddlepaddle-gpu)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
PADDLEOCR_DIR = ROOT / "PaddleOCR"
CONFIG_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "output"
DATA_DIR = ROOT / "data"

# PaddleOCR 仓库 (用于训练工具)
PADDLEOCR_REPO = "https://github.com/PaddlePaddle/PaddleOCR.git"
PADDLEOCR_BRANCH = "main"


def ensure_paddleocr_repo():
    """确保 PaddleOCR 仓库已克隆 (shallow clone)。"""
    if PADDLEOCR_DIR.is_dir() and (PADDLEOCR_DIR / "tools" / "train.py").exists():
        print(f"[OK] PaddleOCR 仓库已存在: {PADDLEOCR_DIR}")
        return True

    print(f"[Clone] PaddleOCR 仓库 (shallow) ...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", PADDLEOCR_BRANCH,
             PADDLEOCR_REPO, str(PADDLEOCR_DIR)],
            check=True, capture_output=True, text=True, timeout=300,
        )
        print(f"[OK] 克隆完成: {PADDLEOCR_DIR}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] git clone 失败: {e.stderr[:500]}")
        return False
    except FileNotFoundError:
        print("[Error] git 未安装或不在 PATH 中")
        return False
    except subprocess.TimeoutExpired:
        print("[Error] git clone 超时 (300s), 请检查网络")
        return False


def check_data():
    """检查训练数据是否就绪。"""
    train_label = DATA_DIR / "train" / "label.txt"
    val_label = DATA_DIR / "val" / "label.txt"

    if not train_label.exists():
        print(f"[Error] 训练数据未准备: {train_label} 不存在")
        print(f"  请先运行: python finetune/prepare_data.py --input <图片目录>")
        return False

    n_train = len(train_label.read_text(encoding="utf-8").strip().splitlines())
    n_val = 0
    if val_label.exists():
        n_val = len(val_label.read_text(encoding="utf-8").strip().splitlines())

    print(f"[Data] 训练集: {n_train} 条, 验证集: {n_val} 条")

    if n_train < 10:
        print(f"[Warning] 训练样本过少 ({n_train}), 建议至少 50 张")
    return True


def find_pretrained_model():
    """查找预训练模型权重。

    优先级:
    1. finetune/pretrained/ 目录下的 .pdparams
    2. PaddleOCR 官方预训练模型 (自动下载)
    """
    pretrained_dir = ROOT / "pretrained"
    if pretrained_dir.is_dir():
        candidates = list(pretrained_dir.glob("*.pdparams"))
        if candidates:
            print(f"[Pretrained] 使用本地权重: {candidates[0]}")
            return str(candidates[0])

    # 使用 PaddleOCR 官方 PP-OCRv5 rec 预训练模型
    # 训练脚本会自动从 PaddleOCR 的预训练模型 URL 下载
    print("[Pretrained] 未找到本地权重, 将使用 PaddleOCR 官方预训练模型 (自动下载)")
    return None


def generate_config(epochs: int, batch_size: int, lr: float,
                    pretrained: str | None) -> Path:
    """生成训练配置文件。"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / "rec_industrial_finetune.yaml"

    train_label = str(DATA_DIR / "train" / "label.txt").replace("\\", "/")
    val_label = str(DATA_DIR / "val" / "label.txt").replace("\\", "/")

    pretrained_yaml = ""
    if pretrained:
        pretrained_yaml = f"  pretrained_model: {pretrained}"
    else:
        # PaddleOCR 官方 PP-OCRv5 rec 预训练模型
        pretrained_yaml = (
            "  pretrained_model: "
            "https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/"
            "PP-OCRv5_mobile_rec_pretrained.pdparams"
        )

    config_content = f"""# PP-OCRv6 工业场景识别模型 fine-tune 配置
# 自动生成 by finetune/train.py

Global:
  debug: false
  use_gpu: true
  epoch_num: {epochs}
  log_smooth_window: 10
  print_batch_step: 5
  save_model_dir: {str(OUTPUT_DIR / 'rec_industrial').replace(chr(92), '/')}
  save_epoch_step: 10
  eval_batch_step: [0, 50]
  cal_metric_during_train: true
  checkpoints:
  save_inference_dir: {str(OUTPUT_DIR / 'inference').replace(chr(92), '/')}
  use_visualdl: false
  infer_img: doc/imgs_words/ch/word_1.jpg
  character_dict_path: ppocr/utils/ppocr_keys_v1.txt
  max_text_length: &max_text_length 40
  infer_mode: false
  use_space_char: true
  distributed: false

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: {lr}
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetV3
    scale: 0.95
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 120
            depth: 2
            hidden_dims: 120
            kernel_size: [1, 3]
            use_guide: True
          Head:
            fc_decay: 0.00001
      - NRTRHead:
          nrtr_dim: 384
          max_text_length: *max_text_length

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - NRTRLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc
  ignore_space: false

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {str(DATA_DIR / 'train').replace(chr(92), '/')}
    label_file_list:
      - {train_label}
    ratio_list: [1.0]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - RecAug:
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_gtc
            - length
            - valid_ratio
  loader:
    shuffle: true
    batch_size_per_card: {batch_size}
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {str(DATA_DIR / 'val').replace(chr(92), '/')}
    label_file_list:
      - {val_label}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - MultiLabelEncode:
          gtc_encode: NRTRLabelEncode
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys:
            - image
            - label_ctc
            - label_gtc
            - length
            - valid_ratio
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: {batch_size}
    num_workers: 2
"""

    config_path.write_text(config_content, encoding="utf-8")
    print(f"[Config] 训练配置已生成: {config_path}")
    return config_path


def run_training(config_path: Path, dry_run: bool = False):
    """执行训练 (子进程)。"""
    train_script = PADDLEOCR_DIR / "tools" / "train.py"

    if dry_run:
        print("[Dry-run] 仅生成配置, 不执行训练")
        print(f"  训练脚本: {train_script}")
        print(f"  手动执行: cd {PADDLEOCR_DIR} && {sys.executable} tools/train.py -c {config_path}")
        return True

    if not train_script.exists():
        print(f"[Error] 训练脚本不存在: {train_script}")
        return False

    cmd = [
        sys.executable,
        str(train_script),
        "-c", str(config_path),
    ]

    print(f"\n{'=' * 50}")
    print(f"[Train] 启动训练")
    print(f"  命令: {' '.join(cmd)}")
    print(f"  工作目录: {PADDLEOCR_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 50}\n")

    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PADDLEOCR_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    # 确保 CUDA 可见
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PADDLEOCR_DIR),
            env=env,
            timeout=7200,  # 2小时上限
        )
        if proc.returncode == 0:
            print(f"\n[Done] 训练完成! 模型保存在: {OUTPUT_DIR / 'rec_industrial'}")
            return True
        else:
            print(f"\n[Error] 训练退出码: {proc.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("[Error] 训练超时 (2h)")
        return False
    except KeyboardInterrupt:
        print("\n[Interrupted] 训练中断")
        return False


def main():
    parser = argparse.ArgumentParser(description="PP-OCRv6 fine-tune 训练")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数 (默认 50, 小数据集建议 100-200)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="批大小 (默认 16, 显存不足可降至 8)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率 (默认 0.001, fine-tune 建议 0.0005-0.001)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练权重路径 (.pdparams)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅生成配置, 不执行训练")
    args = parser.parse_args()

    print("=== PP-OCRv6 Fine-tune ===")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print()

    # Step 1: 检查数据
    if not check_data():
        sys.exit(1)

    # Step 2: 确保 PaddleOCR 仓库
    if not args.dry_run and not ensure_paddleocr_repo():
        sys.exit(1)

    # Step 3: 查找预训练模型
    pretrained = args.pretrained or find_pretrained_model()

    # Step 4: 生成配置
    config_path = generate_config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=pretrained,
    )

    # Step 5: 执行训练
    if args.dry_run:
        run_training(config_path, dry_run=True)
    else:
        ok = run_training(config_path)
        if ok:
            print(f"\n下一步:")
            print(f"  评估: python finetune/evaluate.py")
            print(f"  导出: python finetune/export_onnx.py")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
