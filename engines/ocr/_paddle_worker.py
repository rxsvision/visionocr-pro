"""PaddleOCR 子进程 worker (隔离 torch cudnn 冲突)

用法: python _paddle_worker.py <image_path> [--device gpu|cpu] [--lang ch]
输出: JSON 到 stdout (纯 ASCII, ensure_ascii=True)
错误: JSON {"error": "..."} + exit code 1

注意: paddle 的日志全部重定向到 stderr, stdout 只输出最终 JSON。
"""
import os
import sys

# ─── 必须在 import paddle 之前设置 ───────────────────────────
os.environ["GLOG_minloglevel"] = "2"
os.environ["FLAGS_minloglevel"] = "2"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_new_executor"] = "0"

import json


def main():
    # 强制 stdout 为 UTF-8 (Windows cmd 默认 GBK)
    if sys.stdout.encoding != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--no-orient", action="store_true")
    parser.add_argument("--no-unwarp", action="store_true")
    args = parser.parse_args()

    # 抑制 paddle/paddlex 的 stdout 日志噪声
    import logging
    logging.disable(logging.INFO)

    try:
        # 先导入 paddle 并禁用 PIR/OneDNN (3.x Windows CPU bug)
        import paddle
        try:
            paddle.set_flags({
                "FLAGS_enable_pir_api": 0,
                "FLAGS_enable_pir_in_executor": 0,
                "FLAGS_use_mkldnn": 0,
            })
        except Exception:
            pass  # 旧版不支持 set_flags, 忽略

        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            use_doc_orientation_classify=not args.no_orient,
            use_doc_unwarping=not args.no_unwarp,
            use_textline_orientation=True,
            device=args.device,
            lang=args.lang,
            enable_mkldnn=False,
        )
        result = ocr.predict(args.image_path)

        lines = []
        for page in result:
            # OCRResult 可能是 dict-like (paddlex) 或普通对象
            if hasattr(page, "get"):
                texts = page.get("rec_texts", None)
                polys = page.get("rec_polys", page.get("dt_polys", None))
                scores = page.get("rec_scores", None)
            else:
                texts = getattr(page, "rec_texts", None)
                polys = getattr(page, "rec_polys", None)
                scores = getattr(page, "rec_scores", None)

            if texts is None:
                if isinstance(page, (list, tuple)):
                    for item in page:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            box, ts = item
                            txt, score = ts
                            lines.append({"text": str(txt), "box": _norm_box(box),
                                          "confidence": float(score)})
                continue

            polys = polys if polys is not None else [[] for _ in texts]
            scores = scores if scores is not None else [0.0 for _ in texts]
            for txt, box, score in zip(texts, polys, scores):
                lines.append({"text": str(txt), "box": _norm_box(box),
                              "confidence": float(score)})

        avg = sum(l["confidence"] for l in lines) / len(lines) if lines else 0.0
        output = {
            "text": "\n".join(l["text"] for l in lines),
            "lines": lines,
            "confidence": round(avg, 4),
            "engine": "paddleocr_vl",
        }
        # ensure_ascii=True: 纯 ASCII 输出, 避免任何编码问题
        print(json.dumps(output, ensure_ascii=True))

    except Exception as e:
        print(json.dumps({"error": str(e), "engine": "paddleocr_vl"},
                         ensure_ascii=True))
        sys.exit(1)


def _norm_box(box):
    try:
        if box is not None and len(box) > 0:
            first = box[0]
            if isinstance(first, (list, tuple)):
                return [[float(p[0]), float(p[1])] for p in box]
            pts = list(box)
            return [[float(pts[i]), float(pts[i + 1])]
                    for i in range(0, len(pts) - 1, 2)]
    except Exception:
        pass
    return []


if __name__ == "__main__":
    main()
