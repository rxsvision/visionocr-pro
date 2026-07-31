"""跨域验证: 在三个真实工业项目图像上测试 YOLO + PatchCore

数据集性质 (诚实前提):
- 均为小样本打光可行性集, 无正常/缺陷划分, 无 ground-truth 框
- 故无法统计有效 recall/FP; 本脚本只做"方向性"验证 + 暴露域差

测试内容:
1. YOLO(PCB训练) 跨域推理 → 预期近零检出 (PCB结构缺陷→金属/玻璃表面不迁移)
2. PatchCore 留一法 (project_b inner wall, 缺陷图互建库) → 仅演示, 非有效指标
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from core.imutils import imread_unicode, imwrite_unicode

PROJECTS = {
    "project_a": Path(r"X:\data\project_a"),
    "project_b_scratch": Path(r"X:\data\project_b\scratch"),
    "project_b_oil": Path(r"X:\data\project_b\oil"),
    "project_c": Path(r"X:\data\project_c"),
    "化妆瓶_薄膜": Path(r"X:\data\project_dilm"),
    "化妆瓶_瓶口螺纹": Path(r"X:\data\project_d	hread"),
}
IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png"}
OUT_DIR = Path(r"results/cross_domain")


def _imgs(folder: Path, limit: int = 6) -> list[Path]:
    if not folder.is_dir():
        return []
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in IMG_EXT)
    # 跳过线扫超大图 (height>6000 需分块, 此处略)
    out = []
    for p in files:
        img = imread_unicode(str(p))
        if img is None:
            continue
        if max(img.shape[:2]) > 6000:
            continue
        out.append(p)
        if len(out) >= limit:
            break
    return out


def test_yolo_cross_domain():
    from engines.vision.yolo_defect import YOLODefectEngine
    eng = YOLODefectEngine({"yolo_defect": {"confidence_threshold": 0.25,
                                            "imgsz": 1280}})
    eng.load()
    print(f"\n{'='*60}\n[YOLO 跨域] state={eng.state.value}\n{'='*60}")
    total_imgs = total_det = 0
    for name, folder in PROJECTS.items():
        imgs = _imgs(folder)
        det = 0
        labels = []
        t0 = time.time()
        for p in imgs:
            r = eng.infer(str(p))
            det += r["count"]
            labels.extend(r["labels"])
        dt = (time.time() - t0) / max(len(imgs), 1) * 1000
        total_imgs += len(imgs)
        total_det += det
        print(f"  {name:16s} {len(imgs)}图 检出{det} "
              f"({dt:.0f}ms/图) labels={labels[:4]}")
    print(f"  --- 合计: {total_imgs}图, 检出{total_det} "
          f"(PCB模型跨域迁移率 ≈ {total_det}/{total_imgs})")


def test_patchcore_loo():
    """留一法演示: project_b inner wall 划伤+油污, 缺陷图互建库。"""
    from engines.vision.anomalib_engine import AnomalibEngine
    print(f"\n{'='*60}\n[PatchCore 留一法演示] project_b inner wall (非有效指标, 仅方向)\n{'='*60}")
    scratch = _imgs(PROJECTS["project_b_scratch"], limit=10)
    oil = _imgs(PROJECTS["project_b_oil"], limit=5)
    all_imgs = scratch + oil
    if len(all_imgs) < 5:
        print("  样本不足, 跳过")
        return

    cfg = {"qc": {"patchcore": {"input_size": 512, "coreset_ratio": 0.1,
                                "conservative_mode": False}}}
    scores = []
    for i, target in enumerate(all_imgs):
        bank_imgs = [p for j, p in enumerate(all_imgs) if j != i]
        eng = AnomalibEngine(cfg)
        eng.load()
        eng.train([str(p) for p in bank_imgs])
        r = eng.infer(str(target))
        scores.append((target.name, r.get("score", 0), r.get("pred_label", "?")))
        eng.unload()

    print(f"  {'图像':24s} {'异常分':>10s}  判定")
    for name, sc, lbl in sorted(scores, key=lambda x: -x[1]):
        print(f"  {name:24s} {sc:10.4f}  {lbl}")
    vals = sorted(s for _, s, _ in scores)
    print(f"  分数范围: {vals[0]:.4f} ~ {vals[-1]:.4f} "
          f"(全为缺陷图, 无正常基准, 分离度无意义)")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_yolo_cross_domain()
    test_patchcore_loo()
    print(f"\n[Done] 结果目录: {OUT_DIR}")
