"""PaddleOCR PP-OCRv6 常驻 HTTP 服务 (容器内运行)

配合 engines/ocr/ppocrv6.py 的 server 模式, 消除每次推理
`docker run --rm` 的容器创建 + 模型加载开销 (5~13s → 亚秒级)。

端点:
  GET  /health → {"status": "ok", "engine": "ppocrv6", "ready": true}
  POST /ocr    → multipart 上传图像字节 (field: file)
                 → {"text", "lines", "confidence", "engine": "ppocrv6"}
                 失败时返回 {"error": "...", "engine": "ppocrv6"} (HTTP 200,
                 与 worker.py stdout 协议一致, 宿主端按 error 键判断)

容器内启动:
  python paddle_server.py --device gpu --port 8000

协议约定:
- 模型在服务启动时加载一次并常驻 (含 warmup 预编译)
- 推理串行化 (threading.Lock): paddle predict 非线程安全,
  宿主端本身也是单请求串行调用, 锁仅作保险
"""
import os
import sys
import tempfile
import threading

# ─── 必须在 import paddle 之前设置 (与 worker.py 一致) ────────
os.environ["GLOG_minloglevel"] = "2"
os.environ["FLAGS_minloglevel"] = "2"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_new_executor"] = "0"

import argparse
import logging
import time

logging.disable(logging.INFO)

from fastapi import FastAPI, File, UploadFile
import uvicorn

# worker.py 与本文件同目录 (镜像内 /app), 复用格式化逻辑
from worker import format_ocr_result

_OCR = None
_OCR_LOCK = threading.Lock()
_DEVICE = "gpu"
_READY = False


def _init_ocr(device: str, lang: str) -> None:
    """加载模型一次, 常驻内存; warmup 触发图编译/模型下载。"""
    global _OCR, _READY

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

    t0 = time.time()
    _OCR = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        device=device,
        lang=lang,
        enable_mkldnn=False,
    )
    print(f"[paddle_server] model loaded in {time.time() - t0:.1f}s",
          file=sys.stderr, flush=True)

    # warmup: 小图推理一次, 触发算子编译与惰性初始化,
    # 避免首张真实图片承担全部编译延迟
    try:
        import numpy as np
        warm = np.full((96, 320, 3), 255, dtype=np.uint8)
        t0 = time.time()
        _OCR.predict(warm)
        print(f"[paddle_server] warmup done in {time.time() - t0:.1f}s",
              file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[paddle_server] warmup failed (non-fatal): {e}",
              file=sys.stderr, flush=True)

    _READY = True


app = FastAPI(title="VisionOCR PP-OCRv6 Server")


@app.get("/health")
def health():
    return {"status": "ok", "engine": "ppocrv6",
            "ready": _READY, "device": _DEVICE}


@app.post("/ocr")
def ocr(file: UploadFile = File(...)):
    """接收图像字节 → PP-OCRv6 推理 → JSON (协议与 worker.py 一致)。"""
    if _OCR is None or not _READY:
        return {"error": "模型尚未就绪", "engine": "ppocrv6"}

    try:
        data = file.file.read()
        if not data:
            return {"error": "上传图像为空", "engine": "ppocrv6"}

        # PaddleOCR 接受路径; 写临时文件最稳 (兼容所有输入格式)
        suffix = os.path.splitext(file.filename or "img.png")[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            with _OCR_LOCK:
                result = _OCR.predict(tmp_path)
            output = format_ocr_result(result)
            output["engine"] = "ppocrv6"
            return output
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except Exception as e:
        return {"error": str(e), "engine": "ppocrv6"}


def main():
    global _DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--lang", default="ch")
    args = parser.parse_args()
    _DEVICE = args.device

    _init_ocr(args.device, args.lang)
    print(f"[paddle_server] listening on {args.host}:{args.port}",
          file=sys.stderr, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
