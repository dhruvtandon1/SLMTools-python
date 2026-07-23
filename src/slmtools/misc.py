"""Small numerical helpers ported from ``LFIO/Misc.jl``."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
import math
from numbers import Number, Real
from typing import Any, Sequence

import numpy as np

from .lattice_field import (
    Intensity,
    LatticeField,
    _as_decimal_approx,
    _is_real_number,
    _julia_assignment_values,
    _julia_array_array_operation,
    _julia_array_scalar_operation,
    _julia_promote_numeric_dtypes,
    _julia_scalar_dtype,
)

__all__ = [
    "ramp",
    "nabs",
    "window",
    "safeInverse",
    "centroid",
    "collapse",
    "clip",
    "SchroffError",
]


def _is_lattice_field(value: Any) -> bool:
    return isinstance(value, LatticeField)


def _require_real_ordered_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in "buif":
        return array
    if array.dtype.kind == "O" and all(
        _is_real_number(value) for value in array.flat
    ):
        return array
    raise TypeError(f"{name} must contain real numeric values")


def _lattice(field: Any) -> tuple[np.ndarray, ...]:
    lattice = getattr(field, "L")
    return tuple(np.asarray(axis) for axis in lattice)


def ramp(x: Any) -> Any:
    """Return zero for a negative number and the number otherwise.

    Array inputs are accepted as a NumPy convenience and follow the same
    element-wise rule as Julia broadcasting ``ramp.(x)``.
    """

    if np.ndim(x) != 0:
        arr = np.asarray(x)
        if arr.dtype.kind == "c" or (
            arr.dtype.kind == "O"
            and not all(_is_real_number(value) for value in arr.flat)
        ):
            raise TypeError("ramp values must support real ordering")
        if arr.dtype.kind == "O":
            output = np.array(arr, dtype=object, copy=True)
            for index in np.ndindex(output.shape):
                value = output[index]
                if value < 0:
                    output[index] = type(value)(0)
            return output
        return np.where(arr < 0, np.zeros((), dtype=arr.dtype), arr)
    if not _is_real_number(x):
        raise TypeError("ramp value must support real ordering")
    return type(x)(0) if x < 0 else x


def nabs(v: Any) -> np.ndarray:
    """Absolute value of *v*, normalized by its Euclidean norm."""

    arr = np.asarray(v)
    magnitude = np.abs(arr)
    if magnitude.dtype.kind == "O" and all(
        isinstance(value, Fraction) for value in magnitude.flat
    ):
        # Julia's ``sqrt(::Rational)`` returns Float64, so its broadcasted
        # division also returns a Float64 array.  NumPy's object sqrt cannot
        # call ``Fraction.sqrt`` (which does not exist), so make that same
        # promotion explicit without weakening other object-number paths.
        squared_norm = sum(
            (value * value for value in magnitude.flat), Fraction(0)
        )
        norm = math.sqrt(squared_norm)
        return np.asarray(
            [float(value) / norm for value in magnitude.flat],
            dtype=np.float64,
        ).reshape(magnitude.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        return magnitude / np.sqrt(np.sum(magnitude**2))


def safeInverse(x: Number) -> Number:
    """Return ``1 / x``, or a zero of the input type when ``x == 0``."""

    if x == 0:
        try:
            return type(x)(0)
        except TypeError:
            return 0
    return 1 / x


def collapse(x: Any, i: int) -> np.ndarray:
    """Sum every dimension except Julia-style, one-based dimension *i*."""

    arr = np.asarray(x)
    if isinstance(i, (bool, np.bool_)) or not (
        type(i) is int or isinstance(i, np.int64)
    ):
        raise TypeError("i must have Julia Int (Int64) type")
    i = int(i)
    limits = np.iinfo(np.int64)
    if not limits.min <= i <= limits.max:
        raise OverflowError("i does not fit Julia Int64")
    if i < 0:
        raise ValueError("dimension must not be negative")
    if not 1 <= i <= arr.ndim:
        axes = tuple(range(arr.ndim))
    else:
        axes = tuple(axis for axis in range(arr.ndim) if axis != i - 1)
    # The surviving dimension is already one-dimensional. ``reshape(order='F')``
    # records Julia's column-major ``[:]`` convention for the N=1 case too.
    return np.asarray(np.sum(arr, axis=axes)).reshape(-1, order="F")


def clip(x: Any, threshold: Number) -> Any:
    """Return *x* only when it is strictly greater than *threshold*."""

    if not _is_real_number(threshold):
        raise TypeError("threshold must be a real scalar")
    if np.ndim(x) != 0:
        arr = np.asarray(x)
        if arr.dtype.kind == "c" or (
            arr.dtype.kind == "O"
            and not all(_is_real_number(value) for value in arr.flat)
        ):
            raise TypeError("clip values must support real ordering")
        keep = _julia_array_scalar_operation(
            arr, threshold, np.greater
        )
        if arr.dtype.kind == "O":
            output = np.empty(arr.shape, dtype=object)
            for index in np.ndindex(arr.shape):
                value = arr[index]
                output[index] = (
                    value if keep[index] else type(value)(0)
                )
            return output
        return np.where(keep, arr, np.zeros((), dtype=arr.dtype))
    if not _is_real_number(x):
        raise TypeError("clip value must support real ordering")
    keep = _julia_array_scalar_operation(
        np.asarray(x), threshold, np.greater
    ).reshape(())[()]
    return x if keep else type(x)(0)


def _julia_vector_literal(values: Sequence[Any]) -> np.ndarray:
    """Construct Julia's promoted ``[values...]`` numeric vector.

    NumPy uses value-preserving promotion for mixed integer/float and
    signed/unsigned inputs. Julia's vector literal instead calls its numeric
    promotion and converts every element to that concrete type before the
    later operation. Centroid uses exactly this splatted-vector syntax.
    """

    items = tuple(values)
    if not items:
        return np.empty(0, dtype=object)
    if any(isinstance(value, Decimal) for value in items):
        return np.asarray(
            [_as_decimal_approx(value) for value in items], dtype=object
        )
    if any(isinstance(value, Fraction) for value in items):
        machine = [
            value
            for value in items
            if not isinstance(value, (Fraction, int, np.integer, bool, np.bool_))
        ]
        if not machine:
            return np.asarray(
                [
                    value
                    if isinstance(value, Fraction)
                    else Fraction(int(value), 1)
                    for value in items
                ],
                dtype=object,
            )
        dtype = _julia_scalar_dtype(machine[0])
        for value in machine[1:]:
            dtype = _julia_promote_numeric_dtypes(
                dtype, _julia_scalar_dtype(value)
            )
        output = np.empty(len(items), dtype=dtype)
        for index, value in enumerate(items):
            output[index] = _julia_assignment_values(
                value, dtype
            ).reshape(())[()]
        return output

    dtype = _julia_scalar_dtype(items[0])
    for value in items[1:]:
        dtype = _julia_promote_numeric_dtypes(
            dtype, _julia_scalar_dtype(value)
        )
    output = np.empty(len(items), dtype=dtype)
    for index, value in enumerate(items):
        output[index] = _julia_assignment_values(value, dtype).reshape(())[()]
    return output


def centroid(img: Any, threshold: float = 0.1) -> np.ndarray:
    """Compute an intensity centroid using the original threshold conventions.

    For a lattice field, ``threshold`` is absolute and coordinates come from
    the field's lattice.  For a plain array it is relative to the array maximum
    and returned pixel coordinates are one-based, matching Julia.
    """

    if not _is_real_number(threshold):
        raise TypeError("threshold must be a real scalar")
    is_field = _is_lattice_field(img)
    if is_field:
        if img.field_type is not Intensity:
            raise TypeError("centroid(field) requires an Intensity lattice field")
        data = _require_real_ordered_array(img.data, "centroid input")
    elif isinstance(img, np.ndarray):
        data = _require_real_ordered_array(img, "centroid input")
    else:
        raise TypeError("centroid expects a NumPy array or Intensity lattice field")
    cutoff = (
        threshold
        if is_field
        else _julia_array_scalar_operation(
            np.asarray(np.max(data)),
            threshold,
            np.multiply,
        ).reshape(())[()]
    )
    keep = _julia_array_scalar_operation(data, cutoff, np.greater)
    clipped = np.where(keep, data, np.zeros((), dtype=data.dtype))
    total = np.sum(clipped)
    if not total > 0:
        raise ValueError("Black image.  Can't normalize")

    if is_field:
        coords = _lattice(img)
        if tuple(len(axis) for axis in coords) != data.shape:
            raise ValueError("Field data size does not match lattice size.")
    else:
        coords = tuple(np.arange(1, n + 1) for n in data.shape)

    numerators_list = []
    for i in range(data.ndim):
        masses = collapse(clipped, i + 1)
        coordinate_axis = np.asarray(coords[i])
        products = _julia_array_array_operation(
            masses,
            coordinate_axis,
            np.multiply,
        )
        numerators_list.append(np.sum(products))
    numerators = _julia_vector_literal(numerators_list)
    return _julia_array_scalar_operation(numerators, total, np.divide)


def window(img: Any, w: int | Sequence[int]) -> tuple[np.ndarray, ...]:
    """Return a NumPy Cartesian index tuple around an image centroid.

    The original indexing formula is intentionally retained, including its
    right/down bias for odd widths.  Integer index arrays, rather than slices,
    preserve Julia's bounds failures instead of clipping or wrapping them.
    """

    is_field = _is_lattice_field(img)
    data = np.asarray(img.data if is_field else img)
    if isinstance(w, (int, np.integer)):
        # The field overload is concretely typed with Julia's platform Int;
        # the raw-array overload is generic and also admits other integral
        # scalar widths such as Int32 and Bool.
        if is_field and (isinstance(w, (bool, np.bool_)) or not (
            type(w) is int or isinstance(w, np.int64)
        )):
            raise TypeError("field window widths must be Julia Int values")
        widths = (int(w),) * data.ndim
    else:
        try:
            raw_widths = tuple(w)
        except TypeError as exc:
            raise TypeError("window widths must be integral") from exc
        if any(
            isinstance(value, (bool, np.bool_))
            or not (type(value) is int or isinstance(value, np.int64))
            for value in raw_widths
        ):
            raise TypeError("window width tuples must contain Julia Int values")
        widths = tuple(int(value) for value in raw_widths)
        if len(widths) != data.ndim:
            raise ValueError("window width tuple must match image dimensions")

    # centroid(data) is deliberately used even for lattice fields: the Julia
    # LF overload delegates to ``window(f.data, w)``.
    center = centroid(data)
    if center.dtype.kind == "O":
        center_1based = np.asarray(
            [
                int(value.to_integral_value(rounding=ROUND_HALF_EVEN))
                if isinstance(value, Decimal)
                else int(round(value))
                for value in center
            ],
            dtype=int,
        )
    else:
        center_1based = np.rint(center).astype(int)
    starts_0based = center_1based - np.asarray(widths) // 2
    stops_0based = starts_0based + np.asarray(widths)
    axes_list: list[np.ndarray] = []
    for start, stop, size in zip(
        starts_0based, stops_0based, data.shape
    ):
        axis = np.arange(int(start), int(stop), dtype=np.intp)
        if np.any(axis < 0):
            # Negative NumPy integer indices wrap from the far edge. Julia's
            # zero/negative Cartesian coordinates are invalid instead. Encode
            # them below NumPy's valid negative range so applying the index
            # deterministically raises rather than selecting tail pixels.
            axis = np.where(axis < 0, axis - int(size), axis)
        axes_list.append(axis)
    axes = tuple(axes_list)
    # ``np.ix_`` is the direct NumPy counterpart of a CartesianIndices box:
    # it retains the Cartesian-product shape, permits an empty dimension, and
    # lets NumPy raise IndexError when an actual coordinate lies out of bounds.
    return np.ix_(*axes)


def _normalize_schroff_values(values: np.ndarray) -> np.ndarray:
    """Apply Julia's in-place ``./=`` conversion rules to a masked vector."""

    output = np.array(values, copy=True)
    total = np.sum(output)
    if np.issubdtype(output.dtype, np.integer) or np.issubdtype(output.dtype, np.bool_):
        with np.errstate(divide="ignore", invalid="ignore"):
            quotient = output.astype(np.float64) / total
        if not np.all(np.isfinite(quotient)):
            raise ValueError("Inexact normalization of integer intensity data")
        converted = quotient.astype(output.dtype)
        if not np.array_equal(converted.astype(np.float64), quotient):
            raise ValueError("Inexact normalization of integer intensity data")
        output[...] = converted
        return output
    with np.errstate(divide="ignore", invalid="ignore"):
        output /= total
    return output


def SchroffError(target: Any, reality: Any, threshold: float = 0.5) -> Any:
    """RMS relative error used by Schroff et al. (2023)."""

    if not _is_real_number(threshold):
        raise TypeError("threshold must be a real scalar")
    if not (
        isinstance(target, LatticeField)
        and target.field_type is Intensity
        and isinstance(reality, LatticeField)
        and reality.field_type is Intensity
    ):
        raise TypeError("SchroffError requires two Intensity lattice fields")
    xdata = _require_real_ordered_array(target.data, "Schroff target")
    ydata = _require_real_ordered_array(reality.data, "Schroff reality")
    maximum = np.max(xdata)
    cutoff = _julia_array_scalar_operation(
        np.asarray(maximum), threshold, np.multiply
    ).reshape(())[()]
    mask = _julia_array_scalar_operation(xdata, cutoff, np.greater)
    # Julia's logical indexing follows column-major linear order.
    linear_mask = np.asarray(mask).ravel(order="F")
    x = _normalize_schroff_values(
        np.asarray(xdata).ravel(order="F")[linear_mask]
    )
    y = _normalize_schroff_values(
        np.asarray(ydata).ravel(order="F")[linear_mask]
    )
    n_pixels = int(np.sum(mask))
    with np.errstate(divide="ignore", invalid="ignore"):
        difference = _julia_array_array_operation(x, y, np.subtract)
        difference_squared = _julia_array_array_operation(
            difference, difference, np.multiply
        )
        target_squared = _julia_array_array_operation(
            x, x, np.multiply
        )
        # The Julia source deliberately uses non-broadcast ``/`` between two
        # vectors here. Julia's right division returns the minimum-norm matrix
        # ``a * b' / dot(b,b)``, not elementwise quotients.
        target_norm_squared = np.sum(
            _julia_array_array_operation(
                target_squared, target_squared, np.multiply
            )
        )
        # Julia's vector right division is `a * pinv(b)`: divide `b`
        # before forming the outer product so low-precision rounding agrees.
        target_pseudoinverse = _julia_array_scalar_operation(
            target_squared, target_norm_squared, np.divide
        )
        relative_squared = _julia_array_array_operation(
            difference_squared[:, None],
            target_pseudoinverse[None, :],
            np.multiply,
        )
        if relative_squared.dtype.kind == "O":
            relative_sum = sum(relative_squared.ravel(order="F"))
        else:
            relative_sum = np.zeros((), dtype=relative_squared.dtype)[()]
            for item in relative_squared.ravel(order="F"):
                relative_sum = np.asarray(
                    relative_sum + item, dtype=relative_squared.dtype
                ).reshape(())[()]
        radicand = _julia_array_scalar_operation(
            np.asarray(relative_sum),
            n_pixels,
            np.divide,
        ).reshape(())[()]
        value = (
            math.sqrt(radicand)
            if isinstance(radicand, Fraction)
            else np.sqrt(radicand)
        )
    return value
