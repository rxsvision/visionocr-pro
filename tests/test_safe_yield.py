"""ui/safe_yield.py 单元测试"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.safe_yield import safe_generator


class TestSafeGenerator:
    def test_normal_yield_passthrough(self):
        """正常 yield 不受影响"""
        @safe_generator(lambda e: ("error",))
        def gen():
            yield (1,)
            yield (2,)

        assert list(gen()) == [(1,), (2,)]

    def test_exception_yields_error(self):
        """异常时 yield error_fn 输出"""
        @safe_generator(lambda e: (f"ERR: {e}",))
        def gen():
            yield ("ok",)
            raise ValueError("boom")

        results = list(gen())
        assert results[0] == ("ok",)
        assert "boom" in results[1][0]

    def test_exception_before_any_yield(self):
        """第一个 yield 前就崩溃也能兜底"""
        @safe_generator(lambda e: (None, str(e)))
        def gen():
            raise RuntimeError("early crash")
            yield  # noqa: unreachable

        results = list(gen())
        assert len(results) == 1
        assert "early crash" in results[0][1]

    def test_error_fn_failure_does_not_raise(self):
        """error_fn 本身出错时不向外抛异常"""
        def bad_error_fn(e):
            raise TypeError("error_fn broken")

        @safe_generator(bad_error_fn)
        def gen():
            raise ValueError("original")
            yield  # noqa

        # 不应抛异常, 返回空列表
        results = list(gen())
        assert results == []
