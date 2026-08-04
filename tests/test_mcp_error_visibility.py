"""Unit tests for _leaf_exceptions ExceptionGroup unwrapping.

These guard the error-visibility helper that flattens anyio's nested
ExceptionGroup wrappers down to the underlying leaf causes, so MCP
connection failures log the real error instead of "unhandled errors in a
TaskGroup".
"""

from glados.mcp.manager import _leaf_exceptions


def test_leaf_exceptions_flat_exception():
    assert _leaf_exceptions(ValueError("boom")) == ["ValueError: boom"]


def test_leaf_exceptions_flattens_group_preserving_order():
    eg = ExceptionGroup("grp", [ValueError("a"), KeyError("b")])
    assert _leaf_exceptions(eg) == ["ValueError: a", "KeyError: 'b'"]


def test_leaf_exceptions_flattens_nested_groups():
    inner = ExceptionGroup("inner", [RuntimeError("deep")])
    outer = ExceptionGroup("outer", [inner, OSError("shallow")])
    assert _leaf_exceptions(outer) == ["RuntimeError: deep", "OSError: shallow"]


def test_leaf_exceptions_respects_depth_cap():
    eg = ExceptionGroup("grp", [ValueError("x")])
    out = _leaf_exceptions(eg, max_depth=0)
    assert len(out) == 1
    assert out[0].startswith("(unwrap depth exceeded)")
    assert "ValueError" in out[0]
