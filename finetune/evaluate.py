"""训练模型评估 - CER/WER 指标 + 逐样本对比

用法:
    python finetune/evaluate.py                          # 评估最新模型
    python finetune/evaluate.py --model output/rec_industrial/best_accuracy
    python finetune/evaluate.py --compare                # 对比 fine-tuned vs 原始 rapidocr

指标:
    CER (Character Error Rate): 字符级错误率, 越低越好
    WER (Word Error Rate): 词级错误率
    Accuracy: 完全匹配率 (工业场景核心指标)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def compute_cer(pred: str, gt: str) -> float:
    """计算字符错误率 (Levenshtein distance / gt length)。"""
    if not gt:
        return 0.0 if not pred else 1.0
    # 编辑距离
    n, m = len(pred), len(gt)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if pred[i - 1] == gt[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / m


def compute_wer(pred: str, gt: str) -> float:
    """计算词错误率。"""
    pred_words = pred.split()
    gt_words = gt.split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    n, m = len(pred_words), len(gt_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if pred_words[i - 1] == gt_words[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / m


def load_val_data() -> list[tuple[str, str]]:
    """加载验证集 (image_path, ground_truth)。"""
    val_label = DATA_DIR / "val" / "label.txt"
    if not val_label.exists():
        print(f"[Error] 验证集不存在: {val_label}")
        return []

    records = []
    val_dir = DATA_DIR / "val"
    for line in val_label.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            img_rel, text = parts
            img_path = val_dir / img_rel
            if img_path.exists():
                records.append((str(img_path), text))
    return records


def evaluate_with_rapidocr(records: list[tuple[str, str]]) -> dict:
    """用 RapidOCR (当前生产引擎) 评估。"""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
    except ImportError:
        print("[Error] rapidocr_onnxruntime 未安装")
        return {}

    results = []
    for img_path, gt in records:
        try:
            result, _ = engine(img_path)
            if result:
                pred = " ".join(line[1] for line in result)
            else:
                pred = ""
        except Exception:
            pred = ""

        cer = compute_cer(pred, gt)
        wer = compute_wer(pred, gt)
        exact = pred.strip() == gt.strip()
        results.append({
            "image": Path(img_path).name,
            "gt": gt,
            "pred": pred,
            "cer": cer,
            "wer": wer,
            "exact_match": exact,
        })

    return _summarize(results, "RapidOCR (baseline)")


def evaluate_with_paddle_model(records: list[tuple[str, str]],
                               model_dir: str) -> dict:
    """用训练好的 Paddle 推理模型评估 (子进程隔离)。"""
    import subprocess
    import json
    import tempfile

    # 生成评估脚本 (在子进程中运行, 避免 torch DLL 冲突)
    eval_script = f"""
import sys, json
sys.path.insert(0, r'{ROOT / "PaddleOCR"}')
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    rec_model_dir=r'{model_dir}',
    use_angle_cls=False,
    show_log=False,
)

records = {json.dumps(records, ensure_ascii=False)}
results = []
for img_path, gt in records:
    try:
        res = ocr.ocr(img_path, cls=False)
        if res and res[0]:
            pred = ' '.join(line[1][0] for line in res[0])
        else:
            pred = ''
    except Exception:
        pred = ''
    results.append({{'gt': gt, 'pred': pred}})

print(json.dumps(results, ensure_ascii=False))
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, encoding="utf-8") as f:
        f.write(eval_script)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            print(f"[Error] 评估子进程失败: {proc.stderr[:300]}")
            return {}

        raw_results = json.loads(proc.stdout.strip().splitlines()[-1])
        results = []
        for r in raw_results:
            pred, gt = r["pred"], r["gt"]
            results.append({
                "gt": gt,
                "pred": pred,
                "cer": compute_cer(pred, gt),
                "wer": compute_wer(pred, gt),
                "exact_match": pred.strip() == gt.strip(),
            })
        return _summarize(results, f"Fine-tuned ({Path(model_dir).name})")
    except Exception as e:
        print(f"[Error] 评估失败: {e}")
        return {}
    finally:
        Path(script_path).unlink(missing_ok=True)


def _summarize(results: list[dict], name: str) -> dict:
    """汇总指标。"""
    if not results:
        return {"name": name, "count": 0}

    n = len(results)
    avg_cer = sum(r["cer"] for r in results) / n
    avg_wer = sum(r["wer"] for r in results) / n
    accuracy = sum(r["exact_match"] for r in results) / n

    summary = {
        "name": name,
        "count": n,
        "cer": round(avg_cer, 4),
        "wer": round(avg_wer, 4),
        "accuracy": round(accuracy, 4),
    }

    print(f"\n{'=' * 50}")
    print(f"  模型: {name}")
    print(f"  样本数: {n}")
    print(f"  CER: {avg_cer:.2%} (字符错误率)")
    print(f"  WER: {avg_wer:.2%} (词错误率)")
    print(f"  Accuracy: {accuracy:.1%} (完全匹配率)")
    print(f"{'=' * 50}")

    # 显示错误样本
    errors = [r for r in results if not r.get("exact_match", False)]
    if errors:
        print(f"\n  错误样本 ({len(errors)}/{n}):")
        for r in errors[:10]:
            gt = r.get("gt", "")
            pred = r.get("pred", "")
            print(f"    GT:   \"{gt}\"")
            print(f"    Pred: \"{pred}\" (CER={r['cer']:.2%})")
            print()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Fine-tune 模型评估")
    parser.add_argument("--model", type=str, default=None,
                        help="Paddle 推理模型目录 (默认: output/inference)")
    parser.add_argument("--compare", action="store_true",
                        help="对比 fine-tuned vs RapidOCR baseline")
    args = parser.parse_args()

    print("=== 模型评估 ===\n")

    records = load_val_data()
    if not records:
        sys.exit(1)

    print(f"验证集: {len(records)} 条\n")

    # Baseline: RapidOCR
    if args.compare or args.model is None:
        evaluate_with_rapidocr(records)

    # Fine-tuned model
    model_dir = args.model or str(OUTPUT_DIR / "inference")
    if Path(model_dir).is_dir():
        evaluate_with_paddle_model(records, model_dir)
    elif args.model:
        print(f"[Warning] 模型目录不存在: {model_dir}")
        print(f"  请先完成训练: python finetune/train.py")


if __name__ == "__main__":
    main()
