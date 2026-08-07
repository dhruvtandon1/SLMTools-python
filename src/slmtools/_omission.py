"""Shared marker for distinguishing an omitted argument from explicit ``None``."""

from __future__ import annotations


class _Omitted:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<omitted>"


_OMITTED = _Omitted()


__all__ = ["_OMITTED"]
