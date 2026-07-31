"""PCB 缺陷 YOLO 少样本训练编排器

设计:
    - 基于 ultralytics YOLOv8, COCO 预训练权重微调
    - 默认 yolov8n (快, 少样本足够); 可 --model yolov8s/m 提精度
    - 训练在独立进程运行 (python finetune/train_yolo.py), 与主应用 torch 上下文隔离
    - 最优权重输出到 finetune/output_yolo/pcb_defect/weights/best.pt

用法:
    python finetune/train_yolo.py --epochs 100 --batch 16 --imgsz 1280
    python finetune/train_yolo.py --epochs 2 --imgsz 640   # 快速冒烟
    python finetune/train_yolo.py --model yolov8s --epochs 150

前置:
    python finetune/prepare_pcb_data.py --src "<PCB_DATASET路径>"
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data_pcb" / "data.yaml"
OUTPUT_DIR = ROOT / "output_yolo"


def train(model: str, epochs: int, batch: int, imgsz: int,
          device: str, project_name: str) -> Path:
    from ultralytics import YOLO

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"数据配置缺失: {DATA_YAML}\n"
            f"  请先运行: python finetune/prepare_pcb_data.py --src <PCB_DATASET路径>")

    print(f"=== PCB 缺陷 YOLO 训练 ===")
    print(f"  模型: {model}  epochs: {epochs}  batch: {batch}  imgsz: {imgsz}")
    print(f"  数据: {DATA_YAML}")
    print(f"  设备: {device}")

    m = YOLO(f"{model}.yaml") if not model.endswith(".pt") else YOLO(model)
    # 加载 COCO 预训练权重 (yolov8n.pt 等自动下载)
    if not model.endswith(".pt"):
        m = YOLO(f"{model}.pt")

    results = m.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(OUTPUT_DIR),
        name=project_name,
        exist_ok=True,
        patience=max(20, epochs // 5),  # 早停
        save=True,
        plots=True,
        verbose=True,
        # 少样本工业缺陷: 适度增强, 关闭 mosaic 末轮以稳定收敛
        mosaic=1.0,
        close_mosaic=10,
        flipud=0.5,   # 上下翻转 (PCB 无方向性)
        fliplr=0.5,
    )

    best = OUTPUT_DIR / project_name / "weights" / "best.pt"
    if best.exists():
        print(f"\n[Done] 训练完成, 最优权重: {best}")
        # 打印验证指标
        try:
            metrics = results.results_dict
            mp = metrics.get("metrics/mAP50(B)", 0)
            print(f"  mAP50: {mp:.4f}")
        except Exception:
            pass
    else:
        print(f"\n[Warn] 未找到 best.pt: {best}")
    return best


def main():
    parser = argparse.ArgumentParser(description="PCB 缺陷 YOLO 训练")
    parser.add_argument("--model", default="yolov8n",
                        help="yolov8n/s/m/x 或自有 .pt 权重 (默认 yolov8n)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="训练分辨率 (PCB 微缺陷建议 1280, 冒烟用 640)")
    parser.add_argument("--device", default="0",
                        help="cuda 设备号 / 'cpu' (默认 '0')")
    parser.add_argument("--name", default="pcb_defect",
                        help="输出子目录名 (默认 pcb_defect)")
    args = parser.parse_args()

    try:
        train(args.model, args.epochs, args.batch, args.imgsz,
              args.device, args.name)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] 训练失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
