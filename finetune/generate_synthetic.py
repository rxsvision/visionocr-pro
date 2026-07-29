"""合成训练数据生成器 - 用于验证 fine-tune pipeline 端到端可用

生成工业场景常见的文字图片:
- 材料标记 (HDPE, PP, ABS, SUS304)
- 日期码 (P20110328, 20240315)
- 序列号 (SN-2024-001234)
- 混合字符 (含易混淆字符 0/O, 1/I, 5/S)

生成后自动调用 prepare_data 逻辑写入 train/val 目录。

用法:
    python finetune/generate_synthetic.py --count 100
    python finetune/generate_synthetic.py --count 50 --difficulty hard
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SYNTH_DIR = ROOT / "data" / "synthetic"

# 工业常见文本样本
INDUSTRIAL_TEXTS = [
    # 材料标记
    "HDPE", "PP", "ABS", "SUS304", "SUS316L", "PA66", "POM", "PC",
    "PET", "PVC", "PTFE", "AL6061", "CuZn39Pb3",
    # 日期码
    "P20110328", "P20240315", "20230801", "20251231", "D20260101",
    # 序列号
    "SN-2024-001234", "SN-2025-009876", "LOT20240315A",
    "RXS-2026-0001", "FX-8842-001",
    # 规格
    "DC24V 500mA", "AC220V 50Hz", "IP67", "M8x1.25",
    "100-240V~50/60Hz", "Class II", "CE", "UL",
    # 易混淆字符组合
    "0O1I5S8B", "6008", "80O8", "S1234", "B0082",
    "R3X-2019", "QW-1180", "Z2206",
    # 混合
    "HDPE P20110328", "PP 20240315", "SUS304 SN-001",
    "MADE IN CHINA", "LOT:20260730",
]

# 字体配置
FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]


def generate_image(text: str, difficulty: str = "normal") -> np.ndarray:
    """生成单张训练图片。

    difficulty:
        normal - 白底黑字, 清晰
        medium - 灰底, 轻微噪声
        hard   - 低对比度, 模糊, 噪声, 模拟工业刻字
    """
    # 图片尺寸 (根据文本长度自适应)
    font_scale = random.uniform(0.8, 1.5)
    font = random.choice(FONTS)
    thickness = random.randint(1, 3)

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = random.randint(15, 40), random.randint(15, 35)
    w = tw + pad_x * 2
    h = th + pad_y * 2 + baseline

    # 背景
    if difficulty == "normal":
        bg_val = random.randint(230, 255)
        fg_val = random.randint(0, 40)
    elif difficulty == "medium":
        bg_val = random.randint(180, 220)
        fg_val = random.randint(20, 60)
    else:  # hard - 模拟金属刻字
        bg_val = random.randint(140, 190)
        fg_val = bg_val - random.randint(30, 70)

    img = np.full((h, w, 3), bg_val, dtype=np.uint8)

    # 添加纹理/噪声
    if difficulty in ("medium", "hard"):
        noise_std = 5 if difficulty == "medium" else 12
        noise = np.random.normal(0, noise_std, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 绘制文本
    org = (pad_x, pad_y + th)
    color = (fg_val, fg_val, fg_val)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    # hard 模式: 轻微模糊 (模拟拍照失焦)
    if difficulty == "hard":
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # 随机旋转 (±3°, 模拟拍照倾斜)
    if random.random() < 0.3:
        angle = random.uniform(-3, 3)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_REPLICATE)

    return img


def generate_dataset(count: int = 100, difficulty: str = "mixed"):
    """生成合成数据集。"""
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    difficulties = (
        ["normal", "medium", "hard"] if difficulty == "mixed"
        else [difficulty]
    )

    records = []
    for i in range(count):
        text = random.choice(INDUSTRIAL_TEXTS)
        diff = random.choice(difficulties)
        img = generate_image(text, diff)

        fname = f"synth_{i:04d}_{diff}.png"
        img_path = SYNTH_DIR / fname
        cv2.imwrite(str(img_path), img)
        records.append((img_path, text))

    # 写入 CSV 供 prepare_data 使用
    csv_path = SYNTH_DIR / "annotations.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("filename,text\n")
        for img_path, text in records:
            f.write(f"{img_path.name},{text}\n")

    print(f"[Done] 生成 {count} 张合成图片 → {SYNTH_DIR}")
    print(f"  标注: {csv_path}")
    print(f"  难度分布: {difficulty}")
    print(f"\n  下一步: python finetune/prepare_data.py --input {SYNTH_DIR} --mode csv")
    return records


def main():
    parser = argparse.ArgumentParser(description="合成训练数据生成")
    parser.add_argument("--count", type=int, default=100,
                        help="生成图片数量 (默认 100)")
    parser.add_argument("--difficulty",
                        choices=["normal", "medium", "hard", "mixed"],
                        default="mixed", help="难度 (默认 mixed)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== 合成数据生成 ===")
    print(f"  数量: {args.count}")
    print(f"  难度: {args.difficulty}")
    print()

    generate_dataset(args.count, args.difficulty)


if __name__ == "__main__":
    main()
