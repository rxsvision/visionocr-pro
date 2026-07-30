"""Gradio generator 防御层

问题: Gradio generator 函数内未捕获异常会导致前端静默卡死 (无错误提示, 无日志)。
方案: safe_generator 装饰器包裹 generator, 捕获任何异常后:
  1. 记录完整 traceback 到日志
  2. yield 一个用户可见的错误输出 (而非静默中断)
"""
import functools
import logging
import traceback

logger = logging.getLogger("visionocr.ui")


def safe_generator(error_fn):
    """装饰器: 为 Gradio generator 添加顶层异常兜底。

    Args:
        error_fn: callable(Exception) -> tuple, 返回与 generator 正常输出
                  同形状的 tuple, 用于在异常时 yield 给前端显示。

    Usage:
        @safe_generator(lambda e: (None, f"内部错误: {e}", ...))
        def _run_stream(...):
            yield ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                yield from fn(*args, **kwargs)
            except Exception as e:
                logger.exception("Generator [%s] 未捕获异常", fn.__name__)
                try:
                    yield error_fn(e)
                except Exception:
                    # error_fn 本身出错时最后兜底
                    logger.critical("error_fn 也失败, 原始异常: %s", e)
        return wrapper
    return decorator
