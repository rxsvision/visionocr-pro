"""PP-OCRv6 Docker OCR - Industrial Image Recognition Accuracy Test

Tests character recognition on real industrial images.
Image directories are read from scripts/local_data_paths.yaml (gitignored).

Usage:
  1. Copy scripts/local_data_paths.example.yaml → scripts/local_data_paths.yaml
  2. Fill in your local data directories under the `ocr_eval` section
  3. Run: python scripts/eval_ppocrv6_accuracy.py
"""
import os
import sys
import time
import glob
import io
from pathlib import Path

import yaml

# Fix Windows console encoding for Unicode OCR output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path (脚本位于 scripts/, 取父目录)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.ocr.ppocrv6 import PPOCRv6Engine

_CONFIG_PATH = Path(__file__).resolve().parent / "local_data_paths.yaml"


def _load_ocr_eval_paths() -> dict[str, str]:
    """从本地配置加载 OCR 评估数据路径。"""
    if not _CONFIG_PATH.is_file():
        print(f"[ERROR] 未找到本地数据配置: {_CONFIG_PATH}")
        print(f"        请复制 local_data_paths.example.yaml → local_data_paths.yaml")
        print(f"        并在 ocr_eval 段填入本机实际数据目录。")
        sys.exit(1)
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    paths = cfg.get("ocr_eval", {})
    if not paths:
        print("[ERROR] local_data_paths.yaml 中 ocr_eval 段为空。")
        sys.exit(1)
    return paths


def format_result(result: dict) -> str:
    """Format a single OCR result for display."""
    lines_info = []
    if result.get("lines"):
        for i, line in enumerate(result["lines"][:10]):  # Show max 10 lines
            if isinstance(line, dict):
                text = line.get("text", "")
                conf = line.get("confidence", 0)
                lines_info.append(f"    [{i+1}] \"{text}\" (conf: {conf:.3f})")
            else:
                lines_info.append(f"    [{i+1}] {line}")
    return "\n".join(lines_info)


def run_test():
    print("=" * 70)
    print("PP-OCRv6 Docker OCR - Industrial Image Recognition Test")
    print("=" * 70)

    # Load data paths from local config
    eval_paths = _load_ocr_eval_paths()

    # Configuration
    config = {"ocr": {"ppocrv6": {"gpu": True, "timeout": 180}}}

    # Initialize and load engine
    print("\n[1] Initializing PPOCRv6Engine...")
    engine = PPOCRv6Engine(config)
    engine.load()
    print(f"    Engine state: {engine.state.value}")
    print(f"    Backend: {engine._backend}")
    print(f"    GPU: {engine._use_gpu}")
    print(f"    Docker image: {engine._docker_image}")

    if not engine.is_ready():
        print("ERROR: Engine failed to load. Aborting.")
        return

    # Collect test images from configured paths
    test_groups = []
    group_configs = [
        ("chip_characters", "Chip Characters", "*.bmp", 5),
        ("bottle_characters", "Bottle Characters", "*.bmp", None),
        ("mixed_samples", "Mixed Samples", "*.png", 3),
    ]
    for key, label, pattern, limit in group_configs:
        dir_path = eval_paths.get(key, "")
        if not dir_path:
            continue
        files = sorted(glob.glob(os.path.join(dir_path, pattern)))
        if limit:
            files = files[:limit]
        test_groups.append((f"{label} ({len(files)} images)", files))

    if not test_groups:
        print("\n[WARN] No valid data directories configured. Nothing to test.")
        return

    # Run tests
    all_results = []
    total_start = time.time()

    for group_name, files in test_groups:
        print(f"\n{'=' * 70}")
        print(f"  GROUP: {group_name}")
        print(f"{'=' * 70}")

        if not files:
            print("  No files found!")
            continue

        for fpath in files:
            fname = os.path.basename(fpath)
            fsize_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"\n  --- {fname} ({fsize_mb:.1f} MB) ---")

            t0 = time.time()
            result = engine.infer(fpath)
            elapsed = time.time() - t0

            # Extract metrics
            text = result.get("text", "")
            confidence = result.get("confidence", 0.0)
            lines = result.get("lines", [])
            error = result.get("error", "")
            num_lines = len(lines) if lines else 0

            success = bool(text) and not error

            print(f"    Time: {elapsed:.2f}s")
            print(f"    Success: {success}")
            print(f"    Confidence: {confidence:.4f}")
            print(f"    Lines detected: {num_lines}")
            if error:
                print(f"    ERROR: {error}")
            if text:
                # Show first 200 chars of recognized text
                display_text = text[:200] + ("..." if len(text) > 200 else "")
                print(f"    Text: \"{display_text}\"")
            if lines:
                print(f"    Line details:")
                print(format_result(result))

            all_results.append({
                "file": fname,
                "group": group_name,
                "size_mb": fsize_mb,
                "time": elapsed,
                "confidence": confidence,
                "num_lines": num_lines,
                "success": success,
                "error": error,
                "text_len": len(text),
            })

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")

    total = len(all_results)
    successes = sum(1 for r in all_results if r["success"])
    failures = total - successes

    if total > 0:
        avg_conf = sum(r["confidence"] for r in all_results if r["success"]) / max(successes, 1)
        avg_time = sum(r["time"] for r in all_results) / total
        total_lines = sum(r["num_lines"] for r in all_results)

        print(f"  Total images tested: {total}")
        print(f"  Successful: {successes}/{total} ({100*successes/total:.1f}%)")
        print(f"  Failed: {failures}")
        print(f"  Average confidence (successful): {avg_conf:.4f}")
        print(f"  Average time per image: {avg_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Total text lines detected: {total_lines}")
        print(f"\n  Per-group breakdown:")

        for group_name, _ in test_groups:
            group_results = [r for r in all_results if r["group"] == group_name]
            if not group_results:
                continue
            g_total = len(group_results)
            g_success = sum(1 for r in group_results if r["success"])
            g_avg_conf = sum(r["confidence"] for r in group_results if r["success"]) / max(g_success, 1)
            g_avg_time = sum(r["time"] for r in group_results) / g_total
            print(f"    {group_name}:")
            print(f"      {g_success}/{g_total} success, avg_conf={g_avg_conf:.4f}, avg_time={g_avg_time:.2f}s")

        if failures > 0:
            print(f"\n  Failed images:")
            for r in all_results:
                if not r["success"]:
                    print(f"    - {r['file']}: {r['error']}")
    else:
        print("  No images were tested!")

    print(f"\n{'=' * 70}")
    print("  TEST COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_test()
