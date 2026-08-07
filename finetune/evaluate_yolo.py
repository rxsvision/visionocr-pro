"""YOLO 缺陷检测验收评估 (标准化指标)

设计出处: 竞品调研 (roboflow/supervision) 结论——评估指标应标准化
(mAP + 混淆矩阵), 替代训练日志中顺带打印的单一 mAP50。
本脚本为离线工具链 (独立进程), 不进运行时检测链路。

用法:
    python finetune/evaluate_yolo.py                       # 评估 pcb_defect/best.pt
    python finetune/evaluate_yolo.py --name pcb_smoke      # 评估其他项目
    python finetune/evaluate_yolo.py --weights path/to.pt  # 自定义权重
    python finetune/evaluate_yolo.py --gate                # 附加团标 §6.2 门控

输出:
    - mAP@50 / mAP@50-95 (总体 + 逐类)
    - Precision / Recall 逐类汇总表
    - 混淆矩阵 PNG (ultralytics 内置生成, 落盘路径打印)
    - --gate: 团标 §6.2 定位精度门控判定 (FAIL 时 exit 1),
      规程见 docs/spec_alignment.md

前置:
    已完成训练: python finetune/train_yolo.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data_pcb" / "data.yaml"
OUTPUT_DIR = ROOT / "output_yolo"

# 团标 §6.2 定位精度门槛 (数值参数化: 草案定稿后只改数值)
SPEC_TIERS = {
    "standard": {"map50": 0.80, "p": 0.85, "r": 0.85},
    "high": {"map50": 0.85, "p": 0.90, "r": 0.90},
}


def spec_gate(metrics: dict, tier: str = "standard") -> tuple[bool, list[str]]:
    """团标 §6.2 定位精度门控 (纯函数, 不依赖模型加载)。

    主门控仅 mAP@50: 曲线下面积与 conf 阈值无关, 不可被参数选择操纵;
    P/R 为 PR 曲线单点值, 仅作次级参考打印, 不参与 FAIL。
    所有指标同用 ultralytics val 默认口径测量 (单一口径,
    避免门控基准漂移)。

    Args:
        metrics: {"map50": float} (ultralytics val 的 box.map50)
        tier: "standard" (mAP50≥0.80) / "high" (≥0.85)

    Returns:
        (pass, 未达标原因列表)
    """
    if tier not in SPEC_TIERS:
        raise ValueError(
            f"未知 spec tier: {tier} (可选: {sorted(SPEC_TIERS)})")
    th = SPEC_TIERS[tier]
    reasons: list[str] = []
    map50 = metrics.get("map50")
    if map50 is None:
        reasons.append("map50 指标缺失, 无法判定")
    elif float(map50) < th["map50"]:
        reasons.append(
            f"mAP@50 {float(map50):.4f} < {th['map50']:.2f} ({tier} 档)")
    return (not reasons), reasons


def evaluate(weights: Path, imgsz: int, device: str,
             name: str, gate: bool = False, tier: str = "standard") -> int:
    """运行 ultralytics val, 打印标准化验收指标。返回退出码。

    gate=True 时附加团标 §6.2 门控: FAIL 返回 1 (PASS 返回 0);
    gate=False 时行为与 v1.5.5 完全一致 (仅评估, 恒返回 0)。
    两种模式测量口径完全一致 (ultralytics val 默认参数)。
    """
    from ultralytics import YOLO

    if not weights.exists():
        print(f"[Error] 权重不存在: {weights}\n"
              f"  请先训练: python finetune/train_yolo.py --name {name}")
        return 1
    if not DATA_YAML.exists():
        print(f"[Error] 数据配置缺失: {DATA_YAML}\n"
              f"  请先准备数据: python finetune/prepare_pcb_data.py --src <路径>")
        return 1

    print("=== YOLO 缺陷检测验收评估 ===")
    print(f"  权重: {weights}")
    print(f"  数据: {DATA_YAML}")
    print(f"  imgsz: {imgsz}  设备: {device}")

    model = YOLO(str(weights))
    # val 输出目录隔离到 output_yolo/{name}_eval, 混淆矩阵 PNG 落盘于此
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=imgsz,
        device=device,
        project=str(OUTPUT_DIR),
        name=f"{name}_eval",
        exist_ok=True,
        verbose=False,
        plots=True,
    )

    box = metrics.box
    names = metrics.names  # {class_id: class_name}

    print(f"\n{'=' * 58}")
    print(f"  总体指标")
    print(f"  mAP@50-95: {box.map:.4f}   mAP@50: {box.map50:.4f}   "
          f"mAP@75: {box.map75:.4f}")
    print(f"  Precision: {box.mp:.4f}   Recall: {box.mr:.4f}")
    print(f"{'=' * 58}")

    # 逐类指标表
    cls_ids = sorted(names.keys())
    col = max(len(str(names[i])) for i in cls_ids) + 2
    print(f"\n  {'类别'.ljust(col)}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    print(f"  {'-' * (col + 34)}")
    for i in cls_ids:
        print(f"  {str(names[i]).ljust(col)}"
              f"{box.p[i]:>8.4f}{box.r[i]:>8.4f}"
              f"{box.map50s[i]:>8.4f}{box.maps[i]:>10.4f}")

    # 混淆矩阵路径 (ultralytics 内置生成)
    eval_dir = OUTPUT_DIR / f"{name}_eval"
    cm_files = sorted(eval_dir.glob("confusion_matrix*.png"))
    if cm_files:
        print(f"\n  混淆矩阵: {cm_files[0]}")
    else:
        print(f"\n  [Warn] 未找到混淆矩阵 PNG (预期目录: {eval_dir})")

    speed = metrics.speed
    print(f"\n  推理速度: preprocess={speed['preprocess']:.1f}ms "
          f"inference={speed['inference']:.1f}ms "
          f"postprocess={speed['postprocess']:.1f}ms")

    # 团标 §6.2 门控判定 (规程: docs/spec_alignment.md)
    th = SPEC_TIERS[tier]
    passed, reasons = spec_gate({"map50": float(box.map50)}, tier=tier)
    verdict = "PASS" if passed else "FAIL"
    print(f"\n  团标 §6.2 门控 [{tier} 档]: mAP@50 {box.map50:.4f} "
          f"vs ≥{th['map50']:.2f} → {verdict}")
    for r in reasons:
        print(f"    [FAIL] {r}")
    pr_ok = box.mp >= th["p"] and box.mr >= th["r"]
    print(f"  次级参考 (val 默认口径): "
          f"P {box.mp:.4f} / R {box.mr:.4f} vs "
          f"≥{th['p']:.2f}/{th['r']:.2f} "
          f"({'达标' if pr_ok else '未达标, 不影响门控'})")
    if gate and not passed:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="YOLO 缺陷检测验收评估")
    parser.add_argument("--name", default="pcb_defect",
                        help="训练项目名 (默认 pcb_defect, "
                             "权重取 output_yolo/{name}/weights/best.pt)")
    parser.add_argument("--weights", default=None,
                        help="自定义权重路径 (默认取训练输出的 best.pt)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="评估分辨率 (与训练默认一致, 冒烟用 640)")
    parser.add_argument("--device", default="0",
                        help="cuda 设备号 / 'cpu' (默认 '0')")
    parser.add_argument("--gate", action="store_true",
                        help="启用团标 §6.2 门控 (FAIL 时 exit 1)")
    parser.add_argument("--spec-tier", default="standard",
                        choices=sorted(SPEC_TIERS),
                        help="门控档位: standard=P/R≥85,mAP50≥0.80; "
                             "high=90/90/0.85 (默认 standard)")
    args = parser.parse_args()

    weights = (Path(args.weights) if args.weights
               else OUTPUT_DIR / args.name / "weights" / "best.pt")
    try:
        sys.exit(evaluate(weights, args.imgsz, args.device, args.name,
                          gate=args.gate, tier=args.spec_tier))
    except Exception as e:
        print(f"[Error] 评估失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
