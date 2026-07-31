"""OCR 精度对比评估脚本

用法:
  python scripts/eval_ocr_accuracy.py --engine rapidocr --data-dir finetune/data/synthetic --annotations finetune/data/synthetic/annotations.csv --output results_rapid.json
  python scripts/eval_ocr_accuracy.py --engine paddleocr --data-dir /data --annotations /data/annotations.csv --output /data/results_paddle.json

输出 JSON: {filename: {text, confidence, time_ms, match, cer}}
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


def load_annotations(csv_path: str) -> dict:
    """加载 ground truth: {filename: expected_text}"""
    gt = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["filename"]] = row["text"].strip()
    return gt


def char_error_rate(pred: str, gt: str) -> float:
    """字符错误率 (CER): 编辑距离 / GT长度。0=完美, 1=全错。"""
    pred = pred.strip().replace(" ", "").upper()
    gt = gt.strip().replace(" ", "").upper()
    if not gt:
        return 0.0 if not pred else 1.0
    # Levenshtein
    m, n = len(pred), len(gt)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if pred[i - 1] == gt[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / len(gt)


def run_rapidocr(image_path: str) -> dict:
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    t0 = time.perf_counter()
    result = ocr(image_path)
    elapsed = (time.perf_counter() - t0) * 1000

    # rapidocr_onnxruntime 返回 2-tuple: (records, aux)
    records = result[0] if isinstance(result, tuple) else result
    lines = []
    if records:
        for item in records:
            # item: [box, text, score]
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                lines.append({"text": str(item[1]), "confidence": float(item[2])})
    text = "\n".join(l["text"] for l in lines)
    avg_conf = sum(l["confidence"] for l in lines) / len(lines) if lines else 0.0
    return {"text": text, "confidence": round(avg_conf, 4), "time_ms": round(elapsed, 1)}


def run_paddleocr(image_path: str, ocr_instance) -> dict:
    t0 = time.perf_counter()
    result = ocr_instance.predict(image_path)
    elapsed = (time.perf_counter() - t0) * 1000

    lines = []
    for page in result:
        # OCRResult is dict-like (paddlex OCRResult)
        if hasattr(page, "get"):
            texts = page.get("rec_texts", None)
            scores = page.get("rec_scores", None)
        else:
            texts = getattr(page, "rec_texts", None)
            scores = getattr(page, "rec_scores", None)
        if texts is None:
            continue
        scores = scores if scores is not None else [0.0] * len(texts)
        for txt, score in zip(texts, scores):
            lines.append({"text": str(txt), "confidence": float(score)})

    text = "\n".join(l["text"] for l in lines)
    avg_conf = sum(l["confidence"] for l in lines) / len(lines) if lines else 0.0
    return {"text": text, "confidence": round(avg_conf, 4), "time_ms": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser(description="OCR accuracy evaluation")
    parser.add_argument("--engine", required=True, choices=["rapidocr", "paddleocr"])
    parser.add_argument("--data-dir", required=True, help="Directory containing images")
    parser.add_argument("--annotations", required=True, help="CSV with filename,text columns")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="gpu", help="Device for paddleocr (gpu/cpu)")
    args = parser.parse_args()

    gt = load_annotations(args.annotations)
    data_dir = Path(args.data_dir)
    results = {}

    # Initialize engine
    ocr_instance = None
    if args.engine == "paddleocr":
        os.environ["GLOG_minloglevel"] = "2"
        os.environ["FLAGS_minloglevel"] = "2"
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["FLAGS_enable_pir_in_executor"] = "0"
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_new_executor"] = "0"
        import logging
        logging.disable(logging.INFO)
        import paddle
        try:
            paddle.set_flags({
                "FLAGS_enable_pir_api": 0,
                "FLAGS_enable_pir_in_executor": 0,
                "FLAGS_use_mkldnn": 0,
            })
        except Exception:
            pass
        from paddleocr import PaddleOCR
        ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=args.device,
            lang="ch",
            enable_mkldnn=False,
        )
        print(f"[INFO] PaddleOCR initialized (device={args.device})", file=sys.stderr)

    total = len(gt)
    exact_matches = 0
    cer_sum = 0.0

    for i, (filename, expected) in enumerate(gt.items()):
        img_path = data_dir / filename
        if not img_path.exists():
            print(f"[SKIP] {filename} not found", file=sys.stderr)
            continue

        try:
            if args.engine == "rapidocr":
                r = run_rapidocr(str(img_path))
            else:
                r = run_paddleocr(str(img_path), ocr_instance)
        except Exception as e:
            r = {"text": "", "confidence": 0.0, "time_ms": 0.0, "error": str(e)}

        # Compare with ground truth
        pred_clean = r["text"].strip().replace(" ", "").upper()
        gt_clean = expected.strip().replace(" ", "").upper()
        match = pred_clean == gt_clean
        cer = char_error_rate(r["text"], expected)

        r["expected"] = expected
        r["match"] = match
        r["cer"] = round(cer, 4)
        results[filename] = r

        if match:
            exact_matches += 1
        cer_sum += cer

        if (i + 1) % 10 == 0:
            print(f"[PROGRESS] {i+1}/{total}", file=sys.stderr)

    # Summary
    evaluated = len(results)
    summary = {
        "engine": args.engine,
        "total_images": total,
        "evaluated": evaluated,
        "exact_match_count": exact_matches,
        "exact_match_rate": round(exact_matches / evaluated, 4) if evaluated else 0.0,
        "mean_cer": round(cer_sum / evaluated, 4) if evaluated else 0.0,
        "mean_confidence": round(
            sum(r["confidence"] for r in results.values()) / evaluated, 4
        ) if evaluated else 0.0,
        "mean_time_ms": round(
            sum(r["time_ms"] for r in results.values()) / evaluated, 1
        ) if evaluated else 0.0,
    }

    output = {"summary": summary, "results": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Engine: {args.engine}", file=sys.stderr)
    print(f"Exact match: {exact_matches}/{evaluated} ({summary['exact_match_rate']*100:.1f}%)", file=sys.stderr)
    print(f"Mean CER: {summary['mean_cer']:.4f}", file=sys.stderr)
    print(f"Mean confidence: {summary['mean_confidence']:.4f}", file=sys.stderr)
    print(f"Mean time: {summary['mean_time_ms']:.1f} ms", file=sys.stderr)
    print(f"Results saved to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
