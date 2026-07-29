"""数据准备工具 - 将用户标注图片转换为 PaddleOCR 训练格式

PaddleOCR 训练数据格式:
    train/images/img_001.png
    train/label.txt  →  每行: 图片相对路径\t标注文本

支持的输入方式:
    1. 目录模式: 图片文件名即标注 (如 HDPE_P20110328.png → "HDPE P20110328")
    2. CSV 模式: annotations.csv (filename, text) 两列
    3. 手动标注: 运行本脚本进入交互模式, 逐张输入文本

用法:
    python finetune/prepare_data.py --input data/raw_images --mode csv
    python finetune/prepare_data.py --input data/raw_images --mode filename
    python finetune/prepare_data.py --interactive
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def prepare_from_csv(input_dir: Path, csv_name: str = "annotations.csv",
                     val_ratio: float = 0.15):
    """从 CSV 标注文件准备数据。

    CSV 格式: filename,text (首行可为 header)
    例: img_001.png,HDPE P20110328
    """
    csv_path = input_dir / csv_name
    if not csv_path.exists():
        print(f"[Error] 未找到标注文件: {csv_path}")
        print(f"  请创建 CSV, 格式: filename,text")
        print(f"  例: img_001.png,HDPE P20110328")
        return False

    records = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            fname, text = row[0].strip(), row[1].strip()
            if fname.lower() == "filename":  # skip header
                continue
            img_path = input_dir / fname
            if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTS:
                records.append((img_path, text))
            else:
                print(f"  [Skip] 图片不存在: {fname}")

    if not records:
        print("[Error] 无有效记录")
        return False

    _split_and_copy(records, val_ratio)
    return True


def prepare_from_filename(input_dir: Path, val_ratio: float = 0.15):
    """从文件名推断标注 (下划线→空格)。

    例: HDPE_P20110328.png → "HDPE P20110328"
    适合批量命名规范的工业场景。
    """
    records = []
    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        text = img_path.stem.replace("_", " ").replace("-", " ")
        records.append((img_path, text))

    if not records:
        print(f"[Error] {input_dir} 中无图片文件")
        return False

    print(f"  从文件名推断标注 (下划线→空格), 共 {len(records)} 张")
    print(f"  示例: {records[0][0].name} → \"{records[0][1]}\"")
    _split_and_copy(records, val_ratio)
    return True


def prepare_interactive(input_dir: Path, val_ratio: float = 0.15):
    """交互模式: 逐张图片输入标注文本。"""
    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"[Error] {input_dir} 中无图片文件")
        return False

    print(f"共 {len(images)} 张图片, 逐张输入标注文本 (输入 q 跳过, Ctrl+C 结束)")
    records = []
    for i, img_path in enumerate(images, 1):
        try:
            text = input(f"  [{i}/{len(images)}] {img_path.name} → ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  中断, 已标注部分仍会保存")
            break
        if text.lower() == "q":
            continue
        if text:
            records.append((img_path, text))

    if not records:
        print("[Error] 无有效标注")
        return False

    _split_and_copy(records, val_ratio)
    return True


def _split_and_copy(records: list, val_ratio: float):
    """划分 train/val 并复制图片 + 生成 label.txt"""
    import random
    random.seed(42)
    random.shuffle(records)

    n_val = max(1, int(len(records) * val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    for split_name, split_records, split_dir in [
        ("train", train_records, TRAIN_DIR),
        ("val", val_records, VAL_DIR),
    ]:
        img_dir = split_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_path = split_dir / "label.txt"

        lines = []
        for img_path, text in split_records:
            dst = img_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)
            # PaddleOCR 格式: 相对路径\t文本
            rel_path = f"images/{img_path.name}"
            lines.append(f"{rel_path}\t{text}")

        label_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [{split_name}] {len(split_records)} 张 → {split_dir}")

    print(f"\n[Done] 数据准备完成:")
    print(f"  训练集: {TRAIN_DIR / 'label.txt'} ({len(train_records)} 条)")
    print(f"  验证集: {VAL_DIR / 'label.txt'} ({len(val_records)} 条)")
    print(f"\n  下一步: python finetune/train.py")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR fine-tune 数据准备")
    parser.add_argument("--input", type=str, default=None,
                        help="原始图片目录")
    parser.add_argument("--mode", choices=["csv", "filename", "interactive"],
                        default="csv", help="标注方式")
    parser.add_argument("--csv", type=str, default="annotations.csv",
                        help="CSV 标注文件名 (mode=csv 时)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="验证集比例 (默认 0.15)")
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式逐张标注")
    args = parser.parse_args()

    if args.interactive:
        args.mode = "interactive"

    if args.input is None:
        # 默认: 检查 data/raw_images 是否存在
        default_input = ROOT / "data" / "raw_images"
        if default_input.is_dir():
            args.input = str(default_input)
        else:
            print("请指定 --input 目录, 或将图片放入 finetune/data/raw_images/")
            print("支持模式: --mode csv | filename | interactive")
            sys.exit(1)

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[Error] 目录不存在: {input_dir}")
        sys.exit(1)

    print(f"=== 数据准备 ===")
    print(f"  输入: {input_dir}")
    print(f"  模式: {args.mode}")
    print(f"  验证集比例: {args.val_ratio:.0%}")
    print()

    if args.mode == "csv":
        ok = prepare_from_csv(input_dir, args.csv, args.val_ratio)
    elif args.mode == "filename":
        ok = prepare_from_filename(input_dir, args.val_ratio)
    else:
        ok = prepare_interactive(input_dir, args.val_ratio)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
