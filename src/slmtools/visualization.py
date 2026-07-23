"""Array-valued visualizations corresponding to SLMTools.jl ``look``."""

from __future__ import annotations

from typing import Any

import numpy as np

from .lattice_field import _is_real_number

__all__ = ["look"]


def _is_field(value: Any) -> bool:
    return hasattr(value, "data") and hasattr(value, "L") and hasattr(value, "flambda")


def _tag_name(field: Any) -> str:
    for attribute in ("field_type", "field_val", "fieldval", "kind", "T"):
        if hasattr(field, attribute):
            tag = getattr(field, attribute)
            if isinstance(tag, str):
                return tag.rsplit(".", 1)[-1]
            return getattr(tag, "__name__", type(tag).__name__)
    # Some implementations expose a ``tag`` property.
    if hasattr(field, "tag"):
        tag = getattr(field, "tag")
        return tag if isinstance(tag, str) else getattr(tag, "__name__", type(tag).__name__)
    raise TypeError(f"cannot determine the field-value type of {type(field)!r}")


def _cycle1(values: np.ndarray) -> np.ndarray:
    return (np.angle(values) + np.pi) / (2.0 * np.pi)


def _look_one(value: Any) -> np.ndarray:
    if not _is_field(value):
        arr = np.asarray(value)
        machine_real = (
            arr.dtype == np.bool_
            or np.issubdtype(arr.dtype, np.floating)
            or np.issubdtype(arr.dtype, np.integer)
        )
        exact_real = (
            arr.dtype == np.dtype(object)
            and arr.size > 0
            and all(_is_real_number(item) for item in arr.flat)
        )
        if not machine_real and not exact_real:
            raise TypeError("look is implemented for real arrays only")
        with np.errstate(divide="ignore", invalid="ignore"):
            return arr / np.max(arr)

    data = np.asarray(value.data)
    tag = _tag_name(value)
    if tag in {"RealPhase", "UPhase", "UnwrappedPhase"}:
        return np.mod(data - np.min(data), 1)
    if tag in {"ComplexPhase", "S1Phase"}:
        return _cycle1(data)
    if tag in {"Modulus", "RealAmplitude", "RealAmp", "Intensity"}:
        with np.errstate(divide="ignore", invalid="ignore"):
            return data / np.max(data)
    if tag in {"ComplexAmplitude", "ComplexAmp"}:
        magnitude = np.abs(data)
        with np.errstate(divide="ignore", invalid="ignore"):
            magnitude = magnitude / np.max(magnitude)
        return np.hstack((magnitude, _cycle1(data)))
    raise TypeError(f"Behavior of look not implemented for this input type: {type(value)!r}")


def look(*values: Any) -> np.ndarray:
    """Return the original field-type-aware greyscale visualization array.

    Multiple arguments are concatenated horizontally.  Unlike plotting
    helpers, this function intentionally returns data and has no GUI side
    effects.
    """

    if not values:
        raise TypeError("look expects at least one argument")
    if len(values) == 1:
        return _look_one(values[0])
    if not all(_is_field(value) for value in values):
        raise TypeError("multiple look arguments must all be lattice fields")
    return np.hstack(tuple(_look_one(value) for value in values))
