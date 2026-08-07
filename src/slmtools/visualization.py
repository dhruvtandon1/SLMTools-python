"""Array-valued visualizations corresponding to SLMTools.jl ``look``."""

from __future__ import annotations

from fractions import Fraction
from numbers import Number
from typing import Any

import gmpy2
import numpy as np

from ._bigfloat import (
    _MPFRComplex,
    _bigfloat_context,
    _is_mpc,
    _to_mpfr,
)
from .lattice_field import (
    _exact_real_to_machine_float,
    _is_real_number,
    _julia_array_scalar_operation,
    _julia_literal_array,
)

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


def _has_bigfloat_complex(values: np.ndarray) -> bool:
    return values.dtype.kind == "O" and any(
        isinstance(value, _MPFRComplex) or _is_mpc(value)
        for value in values.flat
    )


def _is_complex_value(value: Any) -> bool:
    return (
        isinstance(value, (complex, np.complexfloating, _MPFRComplex))
        or _is_mpc(value)
    )


def _has_complex_values(values: np.ndarray) -> bool:
    return values.size > 0 and (
        values.dtype.kind == "c"
        or (
            values.dtype.kind == "O"
            and any(_is_complex_value(value) for value in values.flat)
        )
    )


def _bigfloat_complex_components(value: Any) -> tuple[Any, Any]:
    if isinstance(value, _MPFRComplex) or _is_mpc(value):
        return _to_mpfr(value.real), _to_mpfr(value.imag)
    if isinstance(value, Number):
        return _to_mpfr(value.real), _to_mpfr(value.imag)
    raise TypeError("BigFloat-like complex visualization requires complex values")


def _bigfloat_magnitude(values: np.ndarray) -> np.ndarray:
    output = np.empty(values.shape, dtype=object)
    with _bigfloat_context():
        for index in np.ndindex(values.shape):
            real, imaginary = _bigfloat_complex_components(values[index])
            output[index] = gmpy2.sqrt(real * real + imaginary * imaginary)
    return output


def _cycle1(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size and not (
        values.dtype.kind == "c"
        or (
            values.dtype.kind == "O"
            and all(_is_complex_value(value) for value in values.flat)
        )
    ):
        raise TypeError("Julia cycle1 requires complex visualization values")
    if _has_bigfloat_complex(values):
        output = np.empty(values.shape, dtype=object)
        with _bigfloat_context():
            pi = gmpy2.const_pi()
            denominator = _to_mpfr(np.float64(2.0 * np.pi))
            for index in np.ndindex(values.shape):
                real, imaginary = _bigfloat_complex_components(values[index])
                output[index] = (gmpy2.atan2(imaginary, real) + pi) / denominator
        return output
    if values.dtype.kind == "O":
        scalar_angles = [np.angle(value) for value in values.flat]
        angles = _julia_literal_array(scalar_angles).reshape(values.shape)
    else:
        angles = np.angle(values)
    numerator_pi = np.asarray(np.pi, dtype=angles.dtype)[()]
    denominator = np.float64(2.0 * np.pi)
    return (angles + numerator_pi) / denominator


def _julia_hcat(values: tuple[np.ndarray, ...]) -> np.ndarray:
    """Concatenate like Julia ``hcat``, including vector-to-column lifting."""

    arrays = tuple(np.asarray(value) for value in values)
    has_bigfloat_channels = any(
        array.dtype.kind == "O"
        and any(isinstance(value, gmpy2.mpfr) for value in array.flat)
        for array in arrays
    )
    if has_bigfloat_channels:
        promoted = []
        for array in arrays:
            output = np.empty(array.shape, dtype=object)
            for index in np.ndindex(array.shape):
                output[index] = _to_mpfr(array[index])
            promoted.append(output)
        arrays = tuple(promoted)
    if arrays and all(array.ndim <= 1 for array in arrays):
        return np.column_stack(arrays)
    return np.concatenate(arrays, axis=1)


def _normalize_for_look(values: np.ndarray) -> np.ndarray:
    """Divide by the maximum using Julia numeric promotion for object data."""

    values = np.asarray(values)
    if _has_complex_values(values):
        raise TypeError(
            "Julia maximum cannot order complex visualization values"
        )
    maximum = np.max(values)
    if values.dtype.kind == "O":
        return _julia_array_scalar_operation(values, maximum, np.divide)
    with np.errstate(divide="ignore", invalid="ignore"):
        return values / maximum


def _gray_channel_values(values: np.ndarray) -> np.ndarray:
    """Expose Julia ``Gray`` channel values using NumPy representations.

    ``Gray(x::Rational)`` chooses ``Gray{N0f8}``, so exact rational channels
    are first converted to Float32 and then rounded to an 8-bit normalized
    value.  NumPy object arrays can also represent Julia ``Array{Real}``
    results whose runtime values are ordinary machine floats; materialize
    those as their concrete floating dtype rather than leaking object storage.
    """

    array = np.asarray(values)
    if array.dtype.kind != "O" or array.size == 0:
        return array

    homogeneous_rational = all(
        isinstance(value, Fraction) for value in array.flat
    ) or all(isinstance(value, gmpy2.mpq) for value in array.flat)
    if homogeneous_rational:
        output = np.empty(array.shape, dtype=np.float64)
        for index in np.ndindex(array.shape):
            value = array[index]
            if value < 0 or value > 1:
                raise ValueError(
                    "N0f8 grayscale channels must lie between 0 and 1"
                )
            float32_value = _exact_real_to_machine_float(value, np.float32)
            scaled = np.float32(float32_value * np.float32(255))
            output[index] = int(np.rint(scaled)) / 255.0
        return output

    if all(isinstance(value, (float, np.floating)) for value in array.flat):
        dtypes = {np.asarray(value).dtype for value in array.flat}
        dtype = np.result_type(*dtypes)
        return array.astype(dtype)

    return array


def _look_one(value: Any) -> np.ndarray:
    if not _is_field(value):
        if not isinstance(value, (list, range, np.ndarray)):
            raise TypeError(
                "look raw input must be a Julia-like real AbstractArray"
            )
        arr = (
            _julia_literal_array(value)
            if isinstance(value, list)
            else np.asarray(value)
        )
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
        return _gray_channel_values(_normalize_for_look(arr))

    data = np.asarray(value.data)
    tag = _tag_name(value)
    if tag in {"RealPhase", "UPhase", "UnwrappedPhase"}:
        shifted = _julia_array_scalar_operation(
            data, np.min(data), np.subtract
        )
        return _gray_channel_values(
            _julia_array_scalar_operation(shifted, 1, np.remainder)
        )
    if tag in {"ComplexPhase", "S1Phase"}:
        return _gray_channel_values(_cycle1(data))
    if tag in {"Modulus", "RealAmplitude", "RealAmp", "Intensity"}:
        return _gray_channel_values(_normalize_for_look(data))
    if tag in {"ComplexAmplitude", "ComplexAmp"}:
        magnitude = (
            _bigfloat_magnitude(data)
            if _has_bigfloat_complex(data)
            else np.abs(data)
        )
        magnitude = _gray_channel_values(_normalize_for_look(magnitude))
        phase = _gray_channel_values(_cycle1(data))
        return _julia_hcat((magnitude, phase))
    raise TypeError(f"Behavior of look not implemented for this input type: {type(value)!r}")


def look(*values: Any) -> np.ndarray:
    """Return the original field-type-aware greyscale visualization array.

    Multiple arguments are concatenated horizontally, while zero arguments
    return Julia's empty ``Any[]`` counterpart. Unlike plotting helpers, this
    function intentionally returns data and has no GUI side effects.
    """

    if not values:
        # Julia's zero-length ``LF...`` method reaches ``hcat()`` and returns
        # ``Any[]`` rather than raising.
        return np.empty(0, dtype=object)
    if len(values) == 1:
        return _look_one(values[0])
    if not all(_is_field(value) for value in values):
        raise TypeError("multiple look arguments must all be lattice fields")
    return _julia_hcat(tuple(_look_one(value) for value in values))
