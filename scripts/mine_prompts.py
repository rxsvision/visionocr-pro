"""GDDM 式缺陷提示词挖掘 (轻量自研实现, §5.3 探索性)

原版 GDDM (GS-CLIP, CVPR2026) 用扩散模型挖掘伪缺陷提示词; GS-CLIP 仓库
无 LICENSE 文件 (全版权保留) → 仅借鉴"离群区域 → 提示词挖掘"思想,
不继承任何代码/权重。本脚本是其工程化轻量替代:

    缺陷 mask 标注 → 连通域 → 离群区域裁切 (带上下文)
    → 光度学描述 (对比极性/长宽比/紧凑度) → 规则映射候选缺陷词
    → (+可选) 本地 VLM (Ollama qwen3-vl) 描述精炼
    → 按频次聚合 top-K 词 → 输出 GDINO 提示词候选 (点分隔)

用法:
    python scripts/mine_prompts.py --root <KolektorSDD路径> --out mined.json
    python scripts/mine_prompts.py --root <路径> --vlm        # 可选 VLM 精炼

定位: 探索性。挖掘结果须经 `eval_acceptance.py prompts` A/B 门控
(存在 Recall≥20% ∧ FPR-hold≤10% ∧ 平均框数≤3 的置信度工作点才采纳),
不达标即砍, 不直接进默认提示词。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_MIN_AREA_PX = 15        # 连通域最小面积 (噪点过滤)
_POLARITY_DELTA = 8.0    # 区域-邻域灰度差阈值 (暗/亮判定)
_ELONGATED_ASPECT = 3.0  # 细长判定长宽比
_TOP_K = 12              # 输出提示词数上限
_VLM_MAX_IMAGES = 20     # VLM 精炼图数上限 (控时)
_VLM_TIMEOUT_SEC = 30

# 光度学描述 → 候选词 (英文, GDINO 直接使用)
_WORDS_ELONGATED = ["crack", "scratch", "line mark"]
_WORDS_DARK_COMPACT = ["dark spot", "pit", "hole"]
_WORDS_BRIGHT_COMPACT = ["bright spot", "burr", "foreign material"]
_WORDS_OTHER = ["stain", "discoloration", "surface mark"]


def imdecode_gray(path: str) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                        cv2.IMREAD_GRAYSCALE)


def scan_defect_pairs(root: Path) -> list[tuple[str, str]]:
    """KolektorSDD 式扫描: Part*.jpg + Part*_label.bmp 成对。"""
    pairs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for img_p in sorted(d.glob("Part*.jpg")):
            lbl_p = img_p.with_name(img_p.stem + "_label.bmp")
            if lbl_p.exists():
                pairs.append((str(img_p), str(lbl_p)))
    return pairs


def extract_regions(img: np.ndarray, mask: np.ndarray,
                    img_path: str) -> list[dict]:
    """从 mask 连通域提取离群区域描述。

    每个区域输出: bbox、极性 (dark/bright/flat)、形态 (elongated/compact/
    blob) 及规则映射候选词。
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 127).astype(np.uint8), connectivity=8)
    regions = []
    h, w = img.shape[:2]
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < _MIN_AREA_PX:
            continue
        # 邻域环: bbox 外扩 1.5 倍 (裁到图内), 减去区域本身
        pad_x, pad_y = int(bw * 0.75), int(bh * 0.75)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
        ring = img[y0:y1, x0:x1].astype(np.float64)
        m = mask[y0:y1, x0:x1] > 127
        if m.sum() == 0 or ring.size == m.sum():
            continue
        region_mean = float(img[y:y + bh, x:x + bw][mask[y:y + bh, x:x + bw] > 127].mean())
        surround_mean = float(ring[~m].mean())
        delta = region_mean - surround_mean
        polarity = ("dark" if delta < -_POLARITY_DELTA
                    else "bright" if delta > _POLARITY_DELTA else "flat")
        aspect = max(bw, bh) / max(1, min(bw, bh))
        compactness = area / max(1.0, bw * bh)
        if aspect >= _ELONGATED_ASPECT:
            shape, words = "elongated", _WORDS_ELONGATED
        elif polarity == "dark":
            shape, words = "compact", _WORDS_DARK_COMPACT
        elif polarity == "bright":
            shape, words = "compact", _WORDS_BRIGHT_COMPACT
        else:
            shape, words = "blob", _WORDS_OTHER
        regions.append({
            "image": Path(img_path).name,
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "area": int(area),
            "polarity": polarity,
            "delta": round(delta, 2),
            "aspect": round(aspect, 2),
            "compactness": round(compactness, 3),
            "shape": shape,
            "words": list(words),
        })
    return regions


def vlm_refine(img_path: str, region: dict,
               url: str = "http://127.0.0.1:11434/api/chat") -> list[str]:
    """可选: 本地 Ollama qwen3-vl 对离群区域裁切图补充英文缺陷词。

    best-effort: 任何失败返回空列表, 不阻断主流程。
    """
    try:
        import requests
        img = imdecode_gray(img_path)
        if img is None:
            return []
        x, y, bw, bh = region["bbox"]
        h, w = img.shape[:2]
        pad = max(bw, bh) // 2
        crop = img[max(0, y - pad):min(h, y + bh + pad),
                   max(0, x - pad):min(w, x + bw + pad)]
        if crop.size == 0:
            return []
        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            return []
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        payload = {
            "model": "qwen3-vl",
            "stream": False,
            "messages": [{"role": "user", "content": [
                {"type": "text",
                 "text": ("This crop is from an industrial surface inspection "
                          "image. The marked region contains a defect. List "
                          "up to 5 English defect keywords describing the "
                          "visible anomaly, comma-separated, no explanation. "
                          "/no_think")},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
        }
        resp = requests.post(url, json=payload, timeout=_VLM_TIMEOUT_SEC)
        resp.raise_for_status()
        text = resp.json()["message"]["content"]
        words = []
        for tok in text.replace("、", ",").replace("。", ",").split(","):
            t = tok.strip().strip(".").lower()
            # 仅保留纯英文短语 (≤3 词), 过滤幻觉/解释性文本
            if t and t.isascii() and len(t.split()) <= 3 \
                    and all(c.isalpha() or c == " " for c in t):
                words.append(t)
        return words[:5]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="GDDM 式提示词挖掘 (轻量)")
    ap.add_argument("--root", required=True,
                    help="KolektorSDD 式数据集根目录 (Part*.jpg + _label.bmp)")
    ap.add_argument("--out", default="mined_prompts.json")
    ap.add_argument("--vlm", action="store_true",
                    help="启用本地 Ollama qwen3-vl 精炼 (best-effort)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[Error] 目录不存在: {root}")
        return 1
    pairs = scan_defect_pairs(root)
    if not pairs:
        print(f"[Error] 未发现 Part*.jpg + _label.bmp 成对文件: {root}")
        return 1
    print(f"[mine_prompts] 缺陷图 {len(pairs)} 张, VLM={'开' if args.vlm else '关'}")

    all_regions: list[dict] = []
    counts: Counter = Counter()
    n_vlm_ok = 0
    for k, (img_p, lbl_p) in enumerate(pairs):
        img, mask = imdecode_gray(img_p), imdecode_gray(lbl_p)
        if img is None or mask is None:
            continue
        regions = extract_regions(img, mask, img_p)
        if args.vlm and regions and n_vlm_ok < _VLM_MAX_IMAGES:
            r0 = regions[0]
            extra = vlm_refine(img_p, r0)
            if extra:
                r0["vlm_words"] = extra
                counts.update(extra)
                n_vlm_ok += 1
        for r in regions:
            counts.update(r["words"])
        all_regions.extend(regions)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(pairs)}", flush=True)

    top = [w for w, _ in counts.most_common(_TOP_K)]
    out = {
        "source": "scripts/mine_prompts.py (GDDM 思想轻量实现, §5.3)",
        "dataset": str(root),
        "n_images": len(pairs),
        "n_regions": len(all_regions),
        "vlm_used": bool(args.vlm),
        "vlm_refined_images": n_vlm_ok,
        "word_counts": dict(counts.most_common()),
        "prompt_candidate": ".".join(top),
        "regions": all_regions[:50],
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    print(f"\n离群区域 {len(all_regions)} 个, 候选词 {len(top)} 个:")
    print(f"  {out['prompt_candidate']}")
    print(f"结果已写: {out_p}")
    print("下一步: python scripts/eval_acceptance.py prompts "
          f"{root} --mined {out_p}  (A/B 门控)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
