"""PCB 缺陷检测 YOLO 数据准备

将 PCB_DATASET (VOC XML) 转换为 YOLO 训练格式, 并按类别分层划分 train/val。

源数据结构:
    <src>/Annotations/<class>/*.xml   (VOC 标注)
    <src>/images/<class>/*.jpg        (图像)

输出结构:
    finetune/data_pcb/
        images/train/*.jpg  images/val/*.jpg
        labels/train/*.txt  labels/val/*.txt
        data.yaml           (ultralytics 训练配置)

YOLO 标签格式: <class_id> <x_center> <y_center> <width> <height>  (归一化 0-1)

用法:
    python finetune/prepare_pcb_data.py --src "<PCB_DATASET路径>"
    python finetune/prepare_pcb_data.py --src "<路径>" --val-ratio 0.15
"""
import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data_pcb"

# 类别顺序固定 (训练/推理一致), 小写
CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _find_image(img_dir: Path, stem: str) -> Path | None:
    """按 stem 在图像目录中查找对应图片 (扩展名不定)。"""
    for ext in _IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _parse_voc(xml_path: Path) -> tuple[int, int, list[tuple[str, list[float]]]]:
    """解析 VOC XML, 返回 (width, height, [(class_name, [xmin,ymin,xmax,ymax])])。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.findtext("width", "0"))
    h = int(size.findtext("height", "0"))
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        bb = obj.find("bndbox")
        if bb is None or name not in CLASSES:
            continue
        xmin = float(bb.findtext("xmin", "0"))
        ymin = float(bb.findtext("ymin", "0"))
        xmax = float(bb.findtext("xmax", "0"))
        ymax = float(bb.findtext("ymax", "0"))
        boxes.append((name, [xmin, ymin, xmax, ymax]))
    return w, h, boxes


def _to_yolo_line(class_id: int, box: list[float],
                  img_w: int, img_h: int) -> str | None:
    """VOC 绝对框 → YOLO 归一化行。越界框裁剪到图像范围。"""
    xmin, ymin, xmax, ymax = box
    xmin = max(0.0, min(xmin, img_w))
    xmax = max(0.0, min(xmax, img_w))
    ymin = max(0.0, min(ymin, img_h))
    ymax = max(0.0, min(ymax, img_h))
    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 1 or bh <= 1:  # 退化框丢弃
        return None
    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    return f"{class_id} {xc:.6f} {yc:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"


def prepare(src: Path, val_ratio: float, seed: int = 42) -> dict:
    """执行转换与划分, 返回统计。"""
    ann_root = src / "Annotations"
    img_root = src / "images"
    if not ann_root.is_dir() or not img_root.is_dir():
        raise FileNotFoundError(
            f"源目录缺少 Annotations/ 或 images/: {src}")

    random.seed(seed)

    # 清空输出 (重建)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = OUT_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    stats = {c: {"train": 0, "val": 0, "boxes": 0} for c in CLASSES}
    skipped = 0

    for cls in CLASSES:
        cls_id = CLASSES.index(cls)
        ann_dir = ann_root / cls
        img_dir = img_root / cls
        if not ann_dir.is_dir():
            print(f"[Warn] 类别目录缺失: {ann_dir}")
            continue

        xmls = sorted(ann_dir.glob("*.xml"))
        random.shuffle(xmls)
        n_val = int(round(len(xmls) * val_ratio))

        for i, xml_path in enumerate(xmls):
            split = "val" if i < n_val else "train"
            img_path = _find_image(img_dir, xml_path.stem)
            if img_path is None:
                skipped += 1
                continue

            img_w, img_h, boxes = _parse_voc(xml_path)
            if img_w <= 0 or img_h <= 0 or not boxes:
                skipped += 1
                continue

            # 写图像 (复制)
            dst_img = OUT_DIR / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            # 写标签
            lines = []
            for name, box in boxes:
                line = _to_yolo_line(CLASSES.index(name), box, img_w, img_h)
                if line:
                    lines.append(line)
            dst_lbl = OUT_DIR / "labels" / split / f"{xml_path.stem}.txt"
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

            stats[cls][split] += 1
            stats[cls]["boxes"] += len(lines)

    # 写 data.yaml
    yaml_path = OUT_DIR / "data.yaml"
    names_yaml = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    yaml_path.write_text(
        f"# PCB 缺陷检测 YOLO 配置 (自动生成)\n"
        f"path: {str(OUT_DIR).replace(chr(92), '/')}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )

    total_train = sum(s["train"] for s in stats.values())
    total_val = sum(s["val"] for s in stats.values())
    print(f"\n[Done] 转换完成 → {OUT_DIR}")
    print(f"  训练集: {total_train}  验证集: {total_val}  跳过: {skipped}")
    for cls in CLASSES:
        s = stats[cls]
        print(f"    {cls:18s} train={s['train']:3d} val={s['val']:3d} "
              f"boxes={s['boxes']}")
    print(f"  data.yaml: {yaml_path}")
    return {"train": total_train, "val": total_val, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description="PCB VOC→YOLO 数据准备")
    parser.add_argument("--src", required=True,
                        help="PCB_DATASET 根目录 (含 Annotations/ 和 images/)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="验证集比例 (默认 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        print(f"[Error] 源目录不存在: {src}")
        sys.exit(1)

    try:
        prepare(src, args.val_ratio, args.seed)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
