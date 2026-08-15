"""Lattice-field template generators from ``LFIO/LFTemplates.jl``.

Each generator accepts either ``(FieldTag, lattice, ...)`` or an existing
``LatticeField`` as its template argument.  The public camelCase names are kept
so Julia examples translate by changing syntax rather than vocabulary.
"""

from __future__ import annotations

import math
import os
import re
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import ImageFont

import gmpy2

from ._bigfloat import (
    _MPC,
    _MPFR,
    _MPFRComplex,
    _MPQ,
    _MPZ,
    _bigfloat_context,
    _to_mpfr,
)
from ._omission import _OMITTED
from .lattice_field import (
    ComplexPhase,
    Intensity,
    LatticeField,
    Modulus,
    RealPhase,
    _axis,
    _exact_real_to_machine_float,
    _is_julia_number,
    _is_real_number,
    _julia_asarray,
    _julia_literal_array,
    _julia_array_array_operation,
    _julia_assignment_values,
    _julia_promote_numeric_dtypes,
    _julia_sum,
    _julia_typed_zero,
    _object_contains_gmp,
    _logical_axis_scalar_operation,
    _require_dense_ndarray,
    as_lattice,
    wrap,
)
from .dual_lattices import isft, sft
from .lattice_utils import _step

__all__ = [
    "lfParabola",
    "lfGaussian",
    "lfRing",
    "lfCap",
    "ftaText",
    "lfText",
    "lfRect",
    "lfRand",
    "lfHeart",
    "lfSmile",
    "lfPointer",
    "lfBlur",
]


class _DefaultFlambda:
    def __repr__(self) -> str:
        return "1.0"


_UNSET = _DefaultFlambda()


def _template_data_type(value: Any) -> np.dtype[Any]:
    """Validate the concrete ``DataType`` keyword used by Julia templates."""

    if value is type(None) or not isinstance(value, (type, np.dtype)):
        raise TypeError("R must be a Julia DataType")
    try:
        return np.dtype(value)
    except TypeError as error:
        raise TypeError("R must be a Julia DataType") from error


def _julia_bool_condition(value: Any, name: str) -> bool:
    """Return a Julia ``Bool`` condition without Python truth coercion."""

    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a Julia Bool")
    return bool(value)


def _is_arbitrary_precision_real(value: Any) -> bool:
    return isinstance(value, (Decimal, _MPFR, _MPQ, _MPZ)) or (
        type(value) is int
        and not np.iinfo(np.int64).min <= value <= np.iinfo(np.int64).max
    ) or (
        isinstance(value, Fraction)
        and (
            not np.iinfo(np.int64).min
            <= value.numerator
            <= np.iinfo(np.int64).max
            or not np.iinfo(np.int64).min
            <= value.denominator
            <= np.iinfo(np.int64).max
        )
    )


def _is_field(value: Any) -> bool:
    return isinstance(value, LatticeField)


def _split_template(
    template: Any, args: tuple[Any, ...], flambda: Any
) -> tuple[type, tuple[np.ndarray, ...], float, tuple[Any, ...]]:
    if _is_field(template):
        if flambda is not _UNSET:
            raise TypeError(
                "template field overloads inherit flambda and do not accept an explicit flambda"
            )
        return (
            template.field_type,
            template.L,
            template.flambda,
            args,
        )
    if not args:
        raise TypeError("a lattice must follow the field-value type")
    lattice = as_lattice(args[0])
    output_flambda = 1.0 if flambda is _UNSET else flambda
    return template, lattice, output_flambda, args[1:]


def _center_lattice(
    lattice: tuple[np.ndarray, ...], center: Any
) -> tuple[np.ndarray, ...]:
    if center is _OMITTED:
        center_values = (0.0,) * len(lattice)
    else:
        if center is None:
            raise TypeError("center must be a tuple of real coordinates")
        if not isinstance(center, tuple):
            raise TypeError("center must be a tuple of real coordinates")
        center_values = center
    if len(center_values) != len(lattice):
        raise ValueError("center length must match lattice dimension")
    centered = []
    for index, axis in enumerate(lattice):
        # Broadcast subtraction preserves Julia's StepRangeLen representation.
        # In particular, Float16/Float32 ranges carry a Float64 reference and
        # logical step through this operation before their coordinates are
        # materialized.
        centered.append(
            _logical_axis_scalar_operation(
                axis, center_values[index], np.subtract
            )
        )
    return tuple(centered)


def _coords(lattice: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return tuple(np.meshgrid(*lattice, indexing="ij", sparse=True))


def _object_items(value: Any) -> list[Any]:
    array = _julia_asarray(value)
    if array.dtype != np.dtype(object):
        return []
    return list(array.flat)


def _contains_complex_values(value: Any) -> bool:
    array = np.asarray(value)
    return array.dtype.kind == "c" or (
        array.dtype.kind == "O"
        and any(
            isinstance(
                item,
                (_MPC, _MPFRComplex, complex, np.complexfloating),
            )
            for item in array.flat
        )
    )


def _to_decimal(value: Any) -> Decimal:
    """Promote one real scalar like Julia's ``BigFloat(value)`` conversion."""

    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, (float, np.floating)):
        # Julia BigFloat(Float16/32/64) preserves the represented binary value.
        return Decimal.from_float(float(value))
    if isinstance(value, Integral) or isinstance(value, np.bool_):
        return Decimal(int(value))
    raise TypeError(f"cannot promote {type(value).__name__} to Decimal")


def _decimal_array(value: Any) -> np.ndarray:
    source = np.asarray(value)
    result = np.empty(source.shape, dtype=object)
    for index in np.ndindex(source.shape):
        result[index] = _to_decimal(source[index])
    return result


def _object_inexact_dtype(value: Any) -> np.dtype[Any] | None:
    dtypes: list[np.dtype[Any]] = []
    for item in _object_items(value):
        if isinstance(item, (float, np.floating, complex, np.complexfloating)):
            dtypes.append(np.asarray(item).dtype)
    if not dtypes:
        return None
    result = dtypes[0]
    for dtype in dtypes[1:]:
        result = _julia_promote_numeric_dtypes(result, dtype)
    return np.dtype(result)


def _convert_for_values(value: Any, values: Any) -> Any:
    """Convert retained range metadata to the centered coordinate domain."""

    array = np.asarray(values)
    items = _object_items(array)
    if _object_contains_gmp(array) or isinstance(
        value, (_MPFR, _MPC, _MPFRComplex, _MPQ, _MPZ)
    ):
        if isinstance(value, (_MPC, _MPFRComplex, complex, np.complexfloating)):
            with _bigfloat_context():
                return _MPC(_to_mpfr(value.real), _to_mpfr(value.imag))
        return _to_mpfr(value)
    if any(isinstance(item, Decimal) for item in items) or isinstance(value, Decimal):
        return _to_decimal(value)
    if any(isinstance(item, Fraction) for item in items):
        return value if isinstance(value, Fraction) else Fraction(value)
    if array.dtype == np.dtype(object):
        return value
    return np.asarray(value, dtype=array.dtype)[()]


def _julia_binary(left: Any, right: Any, operation: np.ufunc) -> np.ndarray:
    """Apply a numeric operation with Julia's strong scalar promotion.

    NumPy 2 treats Python scalars as weak, so (for example) a Python ``float``
    can leave a Float32 array at Float32.  Julia treats that value as Float64.
    Conversely, Julia keeps Float32 for arithmetic with ordinary integers,
    where NumPy's array promotion often widens to Float64.
    """

    left_array = _julia_asarray(left)
    right_array = _julia_asarray(right)

    if _object_contains_gmp(left_array) or _object_contains_gmp(right_array):
        return _julia_array_array_operation(
            left_array, right_array, operation
        )

    object_items = _object_items(left_array) + _object_items(right_array)
    if any(isinstance(item, Decimal) for item in object_items):
        if left_array.dtype.kind == "c" or right_array.dtype.kind == "c" or any(
            isinstance(item, (complex, np.complexfloating)) for item in object_items
        ):
            raise TypeError(
                "Decimal template arithmetic has no standard-library "
                "arbitrary-precision complex dtype"
            )
        return np.asarray(
            operation(_decimal_array(left_array), _decimal_array(right_array))
        )

    if left_array.dtype == np.dtype(object) or right_array.dtype == np.dtype(object):
        # Rational with an inexact operand promotes to that operand's machine
        # float/complex type in Julia.  With integral/Rational operands it
        # remains exact.  Mixed object arrays are normalized here because a
        # Julia array literal would already have promoted its elements.
        inexact_dtypes = [
            dtype
            for dtype in (
                left_array.dtype
                if left_array.dtype.kind in "fc"
                else _object_inexact_dtype(left_array),
                right_array.dtype
                if right_array.dtype.kind in "fc"
                else _object_inexact_dtype(right_array),
            )
            if dtype is not None
        ]
        if inexact_dtypes:
            dtype = inexact_dtypes[0]
            for candidate in inexact_dtypes[1:]:
                dtype = _julia_promote_numeric_dtypes(
                    dtype,
                    candidate,
                    division=operation is np.divide,
                    operation=operation,
                )
            if operation is np.divide and np.dtype(dtype).kind in "bui":
                dtype = np.dtype(np.float64)
            with np.errstate(
                divide="ignore",
                invalid="ignore",
                over="ignore",
                under="ignore",
            ):
                return np.asarray(
                    operation(
                        left_array.astype(dtype, copy=False),
                        right_array.astype(dtype, copy=False),
                    )
                )
        return np.asarray(operation(left_array, right_array))

    dtype = _julia_promote_numeric_dtypes(
        left_array.dtype,
        right_array.dtype,
        division=operation is np.divide,
        operation=operation,
    )
    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
        under="ignore",
    ):
        return np.asarray(
            operation(
                left_array.astype(dtype, copy=False),
                right_array.astype(dtype, copy=False),
            )
        )


def _julia_exp(value: Any) -> np.ndarray:
    array = _julia_asarray(value)
    items = _object_items(array)
    if _object_contains_gmp(array):
        output = np.empty(array.shape, dtype=object)
        with _bigfloat_context():
            for index in np.ndindex(array.shape):
                item = array[index]
                if isinstance(item, (_MPC, _MPFRComplex)):
                    argument = _MPC(
                        _to_mpfr(item.real), _to_mpfr(item.imag)
                    )
                else:
                    argument = _to_mpfr(item)
                output[index] = gmpy2.exp(argument)
        return output
    if any(isinstance(item, Decimal) for item in items):
        return np.asarray(np.exp(array))
    if array.dtype == np.dtype(object):
        # Julia defines exp(::Rational) through the Float64 transcendental.
        return np.asarray(np.exp(array.astype(np.float64)))
    return np.asarray(np.exp(array))


def _julia_sqrt(value: Any) -> np.ndarray:
    array = _julia_asarray(value)
    items = _object_items(array)
    if _object_contains_gmp(array):
        output = np.empty(array.shape, dtype=object)
        with _bigfloat_context():
            for index in np.ndindex(array.shape):
                item = array[index]
                if isinstance(item, (_MPC, _MPFRComplex)):
                    argument = _MPC(
                        _to_mpfr(item.real), _to_mpfr(item.imag)
                    )
                else:
                    argument = _to_mpfr(item)
                output[index] = gmpy2.sqrt(argument)
        return output
    if any(isinstance(item, Decimal) for item in items):
        return np.asarray(np.sqrt(array))
    if array.dtype == np.dtype(object):
        # As with exp, sqrt(::Rational) returns Float64 in Base Julia.
        return np.asarray(np.sqrt(array.astype(np.float64)))
    return np.asarray(np.sqrt(array))


def _zero_for_values(values: Any) -> Any:
    array = _julia_asarray(values)
    items = _object_items(array)
    if items and _object_contains_gmp(array):
        return _julia_typed_zero(items[0])
    if any(isinstance(item, Decimal) for item in items):
        return Decimal(0)
    if any(isinstance(item, Fraction) for item in items):
        return Fraction(0)
    return np.zeros((), dtype=array.dtype)[()]


def _sum_julia_terms(terms: Sequence[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if not terms:
        return np.zeros(shape, dtype=np.float64)
    result = np.asarray(terms[0])
    for term in terms[1:]:
        result = _julia_binary(result, term, np.add)
    return result


def _r2(lattice: tuple[np.ndarray, ...]) -> np.ndarray:
    grids = _coords(lattice)
    return _sum_julia_terms(
        [_julia_binary(grid, grid, np.multiply) for grid in grids],
        tuple(len(axis) for axis in lattice),
    )


def _ldot(vector: Sequence[float], lattice: tuple[np.ndarray, ...]) -> np.ndarray:
    values = tuple(vector)
    if len(values) != len(lattice):
        raise ValueError("linear coefficient length must match lattice dimension")
    terms = []
    ndim = len(lattice)
    for dimension, (value, axis) in enumerate(
        zip(values, lattice, strict=True)
    ):
        # Julia evaluates ``axis * coefficient`` while ``axis`` is still an
        # AbstractRange, then reshapes it with toDim.  Multiplying a meshgrid
        # would prematurely materialize low-precision StepRangeLen values.
        product = _logical_axis_scalar_operation(axis, value, np.multiply)
        shape = [1] * ndim
        shape[dimension] = len(product)
        terms.append(np.asarray(product).reshape(shape, order="F"))
    return _sum_julia_terms(terms, tuple(len(axis) for axis in lattice))


def _l2form(lattice: tuple[np.ndarray, ...], matrix: Any) -> np.ndarray:
    mat = (
        _julia_literal_array(matrix)
        if isinstance(matrix, (list, tuple))
        else np.asarray(matrix)
    )
    ndim = len(lattice)
    if mat.ndim != 2 or mat.shape[0] < ndim or mat.shape[1] < ndim:
        raise ValueError(
            "quadratic matrix must contain a leading block matching "
            "the lattice dimension"
        )
    grids = _coords(lattice)
    terms: list[np.ndarray] = []
    for i in range(ndim):
        for j in range(ndim):
            first = _julia_binary(mat[i, j], grids[i], np.multiply)
            terms.append(_julia_binary(first, grids[j], np.multiply))
    return _sum_julia_terms(terms, tuple(len(axis) for axis in lattice))


def _tag_name(tag: Any) -> str:
    return tag if isinstance(tag, str) else getattr(tag, "__name__", type(tag).__name__)


def _standard_output(
    tag: type,
    data: Any,
    lattice: tuple[np.ndarray, ...],
    flambda: Any,
) -> LatticeField:
    values = _require_dense_ndarray(data, "lfStandardOutputFormat data")
    name = _tag_name(tag)
    if tag is ComplexPhase or name in {"ComplexPhase", "S1Phase"}:
        object_items = _object_items(values)
        real_values = (
            all(_is_real_number(value) for value in object_items)
            if values.dtype.kind == "O"
            else np.isrealobj(values)
        )
        complex_values = (
            bool(object_items)
            and all(
                isinstance(value, (_MPC, _MPFRComplex, complex, np.complexfloating))
                for value in object_items
            )
            if values.dtype.kind == "O"
            else np.iscomplexobj(values)
        )
        if real_values:
            # Route through the field helper so Rational phases use Julia's
            # ComplexF64 path and Decimal phases retain Complex{BigFloat}-like
            # components rather than being silently narrowed.
            return wrap(
                LatticeField(
                    values, lattice, flambda, field_type=RealPhase
                )
            )
        elif not complex_values:
            raise TypeError("data type not understood")
    elif tag in {Intensity, Modulus} or name in {
        "Intensity",
        "Modulus",
        "RealAmplitude",
        "RealAmp",
    }:
        if _contains_complex_values(values):
            raise TypeError(
                "Julia cannot order complex values in a non-negative field"
            )
        if np.any(values < 0):
            print("Warning: negative values in nominally non-negative LF data field. Clipping to zero.")
        values = np.where(values < 0, _zero_for_values(values), values)
    return LatticeField(values, lattice, flambda, field_type=tag)


def _prepare(
    template: Any,
    args: tuple[Any, ...],
    center: Any,
    flambda: Any,
) -> tuple[type, tuple[np.ndarray, ...], tuple[np.ndarray, ...], float, tuple[Any, ...]]:
    tag, lattice, output_flambda, remaining = _split_template(template, args, flambda)
    centered = _center_lattice(lattice, center)
    return tag, lattice, centered, output_flambda, remaining


def lfRand(
    template: Any,
    *args: Any,
    center: Any = _OMITTED,
    flambda: Any = _UNSET,
    R: Any = np.float64,
) -> LatticeField:
    """Generate a uniformly random lattice field."""

    dtype = _template_data_type(R)
    tag, lattice, _, output_flambda, remaining = _prepare(template, args, center, flambda)
    if remaining:
        raise TypeError("lfRand accepts no positional arguments after the template")
    shape = tuple(len(axis) for axis in lattice)
    if R in (_MPFR, _MPC, _MPFRComplex):
        precision_bits = 256
        byte_count = precision_bits // 8
        denominator = gmpy2.mpz(1) << precision_bits

        def random_mpfr() -> Any:
            numerator = int.from_bytes(
                np.random.bytes(byte_count), byteorder="little"
            )
            with _bigfloat_context():
                return _MPFR(gmpy2.mpz(numerator)) / _MPFR(denominator)

        data = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            real = random_mpfr()
            if R is _MPFR:
                data[index] = real
            else:
                imaginary = random_mpfr()
                with _bigfloat_context():
                    data[index] = (
                        _MPFRComplex(real, imaginary)
                        if R is _MPFRComplex
                        else _MPC(real, imaginary)
                    )
        return _standard_output(tag, data, lattice, output_flambda)
    if R is Decimal:
        # ``rand(BigFloat, ...)`` draws at the active BigFloat precision.  Use
        # the active Decimal context as its Python analogue and NumPy's global
        # RNG so seeding follows the other template paths.
        precision_bits = max(
            1, math.ceil(getcontext().prec / math.log10(2.0))
        )
        byte_count = (precision_bits + 7) // 8
        mask = (1 << precision_bits) - 1
        denominator = Decimal(1 << precision_bits)
        data = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            numerator = int.from_bytes(
                np.random.bytes(byte_count), byteorder="little"
            ) & mask
            data[index] = Decimal(numerator) / denominator
        return _standard_output(tag, data, lattice, output_flambda)
    if R in (Fraction, _MPQ, _MPZ):
        # Julia has no Random.Sampler for Rational, despite accepting a
        # DataType syntactically in the template keyword.
        raise TypeError("Julia rand has no Rational sampler")
    if np.issubdtype(dtype, np.complexfloating):
        component_dtype = np.empty((), dtype=dtype).real.dtype
        real = np.random.random(shape).astype(component_dtype)
        imaginary = np.random.random(shape).astype(component_dtype)
        ceiling = np.nextafter(
            component_dtype.type(1), component_dtype.type(0)
        )
        real[real == component_dtype.type(1)] = ceiling
        imaginary[imaginary == component_dtype.type(1)] = ceiling
        data = (real + 1j * imaginary).astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        data = np.random.random(shape).astype(dtype)
        ceiling = np.nextafter(dtype.type(1), dtype.type(0))
        data[data == dtype.type(1)] = ceiling
    elif np.issubdtype(dtype, np.bool_):
        data = np.random.randint(0, 2, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        # Julia samples integer machine words across their complete bit range.
        count = int(np.prod(shape, dtype=np.int64))
        data = np.frombuffer(np.random.bytes(count * dtype.itemsize), dtype=dtype).copy()
        data = data.reshape(shape)
    else:
        raise TypeError(f"unsupported random data type {dtype}")
    return _standard_output(tag, data, lattice, output_flambda)


def lfParabola(
    template: Any,
    *args: Any,
    center: Any = _OMITTED,
    flambda: Any = _UNSET,
) -> LatticeField:
    """Generate a scalar- or matrix-quadratic parabola."""

    tag, lattice, centered, output_flambda, remaining = _prepare(template, args, center, flambda)
    if center is not _OMITTED and any(
        _is_arbitrary_precision_real(value) for value in center
    ):
        raise TypeError(
            "Julia 1.11 cannot form lfParabola's linear term after an "
            "arbitrary-precision center shift"
        )
    if not remaining:
        raise TypeError("lfParabola requires a quadratic coefficient")
    quad, *rest = remaining
    if len(rest) > 1:
        raise TypeError("too many positional arguments for lfParabola")
    if not rest:
        lin = (0.0,) * len(lattice)
    else:
        lin = rest[0]
    if lin is None:
        raise TypeError(
            "lfParabola linear coefficients must be an N-tuple of real values"
        )
    if not isinstance(lin, tuple):
        raise TypeError(
            "lfParabola linear coefficients must be an N-tuple of real values"
        )
    if len(lin) != len(lattice) or not all(
        _is_real_number(value) for value in lin
    ):
        raise TypeError("lfParabola linear coefficients must be real")
    if isinstance(quad, np.ndarray) and quad.ndim == 0:
        raise TypeError(
            "lfParabola expects a Real scalar or an AbstractMatrix, not a "
            "zero-dimensional array"
        )
    if isinstance(quad, tuple):
        raise TypeError("lfParabola matrix input must be a dense Matrix")
    if isinstance(quad, np.ndarray) and quad.ndim == 2:
        quad = _require_dense_ndarray(quad, "lfParabola matrix")
    q = (
        _julia_literal_array(quad)
        if isinstance(quad, (list, tuple))
        else _julia_asarray(quad)
    )
    if q.ndim == 0:
        if not _is_real_number(q.reshape(())[()]):
            raise TypeError("lfParabola quadratic coefficient must be real")
        coefficient = _julia_binary(q, 2, np.divide)[()]
        quadratic = _julia_binary(coefficient, _r2(centered), np.multiply)
        data = _julia_binary(quadratic, _ldot(lin, centered), np.add)
    elif q.ndim == 2:
        if not all(_is_real_number(value) for value in q.flat):
            raise TypeError("lfParabola quadratic matrix must be real")
        quadratic = _julia_binary(_l2form(centered, q), 2, np.divide)
        data = _julia_binary(quadratic, _ldot(lin, centered), np.add)
    else:
        raise TypeError("quad must be a scalar or square matrix")
    return _standard_output(tag, data, lattice, output_flambda)


def lfGaussian(
    template: Any,
    *args: Any,
    center: Any = _OMITTED,
    flambda: Any = _UNSET,
) -> LatticeField:
    """Generate an L2/energy-normalized scalar or anisotropic Gaussian."""

    tag, lattice, centered, output_flambda, remaining = _prepare(template, args, center, flambda)
    if not remaining:
        raise TypeError("lfGaussian requires a radius or matrix")
    radius_or_matrix, *rest = remaining
    if len(rest) > 1:
        raise TypeError("too many positional arguments for lfGaussian")
    norm = rest[0] if rest else 1.0
    if not _is_real_number(norm):
        raise TypeError("lfGaussian norm must be real")
    if isinstance(radius_or_matrix, np.ndarray) and radius_or_matrix.ndim == 0:
        raise TypeError(
            "lfGaussian expects a Real scalar or an AbstractMatrix, not a "
            "zero-dimensional array"
        )
    if isinstance(radius_or_matrix, tuple):
        raise TypeError("lfGaussian matrix input must be a dense Matrix")
    if isinstance(radius_or_matrix, np.ndarray) and radius_or_matrix.ndim == 2:
        radius_or_matrix = _require_dense_ndarray(
            radius_or_matrix, "lfGaussian matrix"
        )
    parameter = (
        _julia_literal_array(radius_or_matrix)
        if isinstance(radius_or_matrix, (list, tuple))
        else _julia_asarray(radius_or_matrix)
    )
    if parameter.ndim == 0:
        if not _is_real_number(parameter.reshape(())[()]):
            raise TypeError("lfGaussian scalar radius must be real")
        squared = _julia_binary(parameter, parameter, np.multiply)
        denominator = _julia_binary(2, squared, np.multiply)
        data = _julia_exp(-_julia_binary(_r2(centered), denominator, np.divide))
    elif parameter.ndim == 2:
        data = _julia_exp(-_julia_binary(_l2form(centered, parameter), 2, np.divide))
    else:
        raise TypeError("Gaussian parameter must be a radius or square matrix")
    cell_volume = np.asarray(1, dtype=np.int64)
    for axis in centered:
        cell_volume = _julia_binary(cell_volume, _step(axis), np.multiply)
    energy = _julia_sum(_julia_binary(data, data, np.multiply))
    denominator = _julia_binary(energy, cell_volume, np.multiply)
    scale = _julia_sqrt(_julia_binary(norm, denominator, np.divide))
    # ``p .*= scale`` assigns back into the original element type in Julia.
    data = _julia_binary(data, scale, np.multiply).astype(data.dtype)
    return _standard_output(tag, data, lattice, output_flambda)


def lfRing(
    template: Any,
    *args: Any,
    center: Any = _OMITTED,
    flambda: Any = _UNSET,
) -> LatticeField:
    """Generate a Gaussian-profile ring."""

    tag, lattice, centered, output_flambda, remaining = _prepare(template, args, center, flambda)
    if len(remaining) != 2:
        raise TypeError("lfRing requires radius and width")
    radius, width = remaining
    if not _is_julia_number(radius) or not _is_julia_number(width):
        raise TypeError("lfRing radius and width must be scalar numbers")
    radial_offset = _julia_binary(_julia_sqrt(_r2(centered)), radius, np.subtract)
    numerator = _julia_binary(radial_offset, radial_offset, np.multiply)
    width_squared = _julia_binary(width, width, np.multiply)
    denominator = _julia_binary(2, width_squared, np.multiply)
    data = _julia_exp(-_julia_binary(numerator, denominator, np.divide))
    return _standard_output(tag, data, lattice, output_flambda)


def lfCap(
    template: Any,
    *args: Any,
    center: Any = _OMITTED,
    flambda: Any = _UNSET,
) -> LatticeField:
    """Generate the non-negative cap ``max(height-curvature*r^2/2, 0)``."""

    tag, lattice, centered, output_flambda, remaining = _prepare(template, args, center, flambda)
    if len(remaining) != 2:
        raise TypeError("lfCap requires curvature and height")
    curvature, height = remaining
    if not _is_real_number(curvature) or not _is_real_number(height):
        raise TypeError("lfCap curvature and height must be real")
    curved = _julia_binary(curvature, _r2(centered), np.multiply)
    curved = _julia_binary(curved, 2, np.divide)
    data = _julia_binary(height, curved, np.subtract)
    if _contains_complex_values(data):
        raise TypeError("Julia lfCap cannot order complex values")
    data = np.where(data < 0, _zero_for_values(data), data)
    return _standard_output(tag, data, lattice, output_flambda)


def lfRect(
    template: Any,
    *args: Any,
    center: Any = _OMITTED,
    flambda: Any = _UNSET,
) -> LatticeField:
    """Generate an axis-aligned rectangular boxcar field."""

    tag, lattice, centered, output_flambda, remaining = _prepare(template, args, center, flambda)
    if not remaining:
        raise TypeError("lfRect requires side lengths")
    sides, *rest = remaining
    if len(rest) > 1:
        raise TypeError("too many positional arguments for lfRect")
    height = rest[0] if rest else 1.0
    if not isinstance(sides, tuple):
        raise TypeError("lfRect side lengths must be an N-tuple")
    if len(sides) != len(lattice):
        raise ValueError("side count must match lattice dimension")
    if not all(_is_real_number(side) for side in sides) or not _is_real_number(
        height
    ):
        raise TypeError("lfRect side lengths and height must be real")
    data = np.zeros(tuple(len(axis) for axis in lattice), dtype=np.float64)
    mask = np.ones(data.shape, dtype=bool)
    for grid, side in zip(_coords(centered), sides):
        half_side = _julia_binary(side, 2, np.divide)
        boundary = _julia_binary(half_side, np.finfo(np.float64).eps, np.add)
        mask &= np.abs(grid) <= boundary
    data[mask] = _exact_real_to_machine_float(height, np.float64)
    return _standard_output(tag, data, lattice, output_flambda)


def _font_directories() -> tuple[Path, ...]:
    """Return the platform font roots used by BaseDirs/FreeTypeAbstraction."""

    directories: list[Path] = []
    if os.name == "nt":
        windows = Path(os.environ.get("WINDIR", r"C:\Windows"))
        directories.append(windows / "Fonts")
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            directories.append(
                Path(local_app_data) / "Microsoft" / "Windows" / "Fonts"
            )
    elif sys.platform == "darwin":
        directories.extend(
            (
                Path.home() / "Library" / "Fonts",
                Path("/Library/Fonts"),
                Path("/Network/Library/Fonts"),
                Path("/System/Library/Fonts"),
                Path("/System/Library/Fonts/Supplemental"),
            )
        )
    else:
        data_home = Path(
            os.environ.get(
                "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
            )
        )
        directories.extend((data_home / "fonts", Path.home() / ".fonts"))
        data_dirs = os.environ.get(
            "XDG_DATA_DIRS", "/usr/local/share:/usr/share"
        )
        directories.extend(
            Path(directory) / "fonts"
            for directory in data_dirs.split(os.pathsep)
            if directory
        )
    configured = os.environ.get("FREETYPE_ABSTRACTION_FONT_PATH")
    if configured:
        configured_path = Path(configured)
        if not configured_path.is_dir():
            raise RuntimeError(
                "Path in environment variable "
                "FREETYPE_ABSTRACTION_FONT_PATH is not a valid directory!"
            )
        directories.append(configured_path)

    result: list[Path] = []
    seen: set[str] = set()
    for directory in directories:
        key = os.path.normcase(os.path.abspath(directory))
        if key not in seen and directory.is_dir():
            seen.add(key)
            result.append(directory)
    return tuple(result)


def _font_files() -> tuple[Path, ...]:
    """Enumerate fonts in FreeTypeAbstraction's deterministic folder order."""

    directories = list(_font_directories())
    files: list[Path] = []
    index = 0
    seen_directories = {
        os.path.normcase(os.path.abspath(directory)) for directory in directories
    }
    while index < len(directories):
        directory = directories[index]
        index += 1
        try:
            entries = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_file():
                files.append(entry)
            elif sys.platform.startswith("linux") and entry.is_dir():
                key = os.path.normcase(os.path.abspath(entry))
                if key not in seen_directories:
                    seen_directories.add(key)
                    directories.append(entry)
    return tuple(files)


@lru_cache(maxsize=1)
def _font_catalog() -> tuple[tuple[Path, str, str], ...]:
    catalog: list[tuple[Path, str, str]] = []
    for path in _font_files():
        try:
            face = ImageFont.truetype(
                str(path), 12, layout_engine=ImageFont.Layout.BASIC
            )
            family, style = face.getname()
        except (OSError, TypeError):
            continue
        catalog.append((path, family.lower(), style.lower()))
    return tuple(catalog)


def _load_font(font: str, pixelsize: int) -> ImageFont.ImageFont:
    """Resolve a font by FreeTypeAbstraction's family/style scoring rules."""

    # FreeTypeAbstraction's public string is a family/style query, not a font
    # filename.  Letting Pillow interpret path-like strings directly (or
    # tokenizing their directory names) can silently select an unrelated face.
    if "/" in font or "\\" in font:
        raise OSError(f"could not find font {font!r}")

    searchparts = tuple(
        dict.fromkeys(
            part
            for part in re.split(r"\W+", font.lower())
            if part
        )
    )
    regular_styles = (
        "regular",
        "normal",
        "medium",
        "standard",
        "roman",
        "book",
    )
    best_path: Path | None = None
    best_score = (0, 0, False, -(1 << 63))
    for path, family, style in _font_catalog():
        family_matches = tuple(
            part for part in searchparts if part in family
        )
        if not family_matches:
            continue
        family_score = sum(map(len, family_matches))
        remaining = tuple(
            part for part in searchparts if part not in family
        )
        style_score = (
            sum(len(part) for part in remaining if part in style)
            if any(part in style for part in remaining)
            else 0
        )
        score = (
            family_score,
            style_score,
            any(candidate in style for candidate in regular_styles),
            -(len(family) + len(style)),
        )
        if score > best_score:
            best_path = path
            best_score = score
    if best_path is None:
        raise OSError(f"could not find font {font!r}")
    return ImageFont.truetype(
        str(best_path), pixelsize, layout_engine=ImageFont.Layout.BASIC
    )


def _text_int(value: Any, name: str) -> int:
    """Accept the concrete platform-``Int`` domain used by the Julia API."""

    if isinstance(value, (bool, np.bool_)) or not (
        type(value) is int or isinstance(value, np.int64)
    ):
        raise TypeError(f"{name} must be a Julia Int value")
    return int(value)


def _text_alignment(value: Any, name: str) -> str:
    """Translate a Julia ``Symbol`` spelling without truth/string coercion."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a Julia Symbol")
    # Symbol equality is case-sensitive in Julia. A different-case symbol is
    # valid but falls through to the left/baseline alignment branches.
    return value.removeprefix(":")


def _glyph_raster(
    font: ImageFont.ImageFont, character: str
) -> tuple[np.ndarray, int, int, int]:
    """Return FreeTypeAbstraction-compatible bitmap, bearings, and advance."""

    if hasattr(font, "getmask2"):
        mask, offset = font.getmask2(character, mode="L", anchor="ls")
    else:
        # Pillow 10's built-in bitmap ImageFont lacks getmask2.  Its getbbox
        # origin is the equivalent mask offset; FreeTypeFont continues through
        # the exact getmask2 path above.
        mask = font.getmask(character, mode="L")
        bounds = font.getbbox(character, anchor="ls")
        offset = (bounds[0], bounds[1])
    width, height = mask.size
    if width and height:
        bitmap = np.frombuffer(bytes(mask), dtype=np.uint8).reshape(height, width)
    else:
        bitmap = np.empty((height, width), dtype=np.uint8)

    # Keep the complete FreeType bitmap extent, including antialiased columns
    # that happen to be all zero. Trimming and shifting preserves ordinary
    # glyph pixels but changes bbox_glyph, which uses the original extent.
    bearing_x = int(round(offset[0]))
    bearing_y = int(round(-offset[1]))
    advance_x = int(round(font.getlength(character)))
    return bitmap, bearing_x, bearing_y, advance_x


def _text_u8_scalar(
    value: Any, name: str, *, allow_none: bool = False
) -> int | None:
    """Validate one ``UInt8`` renderstring! keyword value."""

    if value is None and allow_none:
        return None
    if not isinstance(value, np.uint8):
        suffix = " or nothing" if allow_none else ""
        raise TypeError(f"{name} must be a Julia UInt8{suffix}")
    return int(value)


def _text_u8_scalar_or_vector(
    value: Any, name: str, *, allow_none: bool = False
) -> tuple[int | np.ndarray[Any, np.dtype[np.uint8]] | None, bool]:
    """Validate ``Union{UInt8,AbstractVector{UInt8}}`` options."""

    if isinstance(value, np.uint8):
        return int(value), False
    if value is None and allow_none:
        return None, False
    if isinstance(value, list):
        array = np.asarray(value)
    elif isinstance(value, np.ndarray):
        array = np.asarray(value)
    else:
        suffix = ", or nothing" if allow_none else ""
        raise TypeError(
            f"{name} must be a Julia UInt8, UInt8 vector{suffix}"
        )
    if array.ndim != 1 or array.dtype != np.dtype(np.uint8):
        suffix = ", or nothing" if allow_none else ""
        raise TypeError(
            f"{name} must be a Julia UInt8, UInt8 vector{suffix}"
        )
    return array, True


def _round_text_u8(value: Any) -> int:
    """Apply renderstring!'s internal ``round(UInt8, value)`` conversion."""

    converted = int(round(float(value)))
    if converted < 0 or converted > 255:
        raise ValueError("text color is outside the UInt8 range")
    return converted


def _text_background_characters(value: Any) -> list[str] | None:
    """Validate renderstring!'s ``Union{String,AbstractVector{Char},Nothing}``."""

    if value is None:
        return None
    if isinstance(value, str):
        return list(value)
    if isinstance(value, list):
        characters = value
    elif isinstance(value, np.ndarray) and value.ndim == 1:
        characters = value.tolist()
    else:
        raise TypeError("gstr must be a string, Julia Char vector, or nothing")
    if not all(
        isinstance(character, str) and len(character) == 1
        for character in characters
    ):
        raise TypeError("gstr must be a string, Julia Char vector, or nothing")
    return list(characters)


def _glyph_kerning(
    font: ImageFont.ImageFont, previous: str, character: str
) -> int:
    """Return locked FreeType's grid-fitted kerning in integer pixels.

    Pillow's BASIC layout is deliberately used above because it exposes the
    same unshaped glyph bitmaps as FreeTypeAbstraction. Its pair-length delta,
    however, is expressed with an additional 26.6 fixed-point scale: a
    FreeType kerning of ``-1`` pixel appears as ``-1/64``. Recover that pixel
    value before applying Julia's integer rounding. Other Pillow layout
    engines already report their pair-length delta in pixels.
    """

    difference = (
        font.getlength(previous + character)
        - font.getlength(previous)
        - font.getlength(character)
    )
    if getattr(font, "layout_engine", None) == ImageFont.Layout.BASIC:
        difference *= 64
    return int(round(difference))


def ftaText(
    string: str,
    sz: tuple[int, int],
    *,
    fnt: Any = "arial bold",
    pixelsize: int | None = None,
    halign: str = "hcenter",
    valign: str = "vcenter",
    **options: Any,
) -> np.ndarray:
    """Render text with locked FreeTypeAbstraction glyph-placement rules."""

    if not isinstance(string, str):
        raise TypeError("ftaText string must be a Julia String")
    if not isinstance(sz, tuple) or len(sz) != 2:
        raise ValueError("text output size must have two dimensions")
    rows, columns = (
        _text_int(sz[0], "text row count"),
        _text_int(sz[1], "text column count"),
    )
    if pixelsize is None:
        pixelsize = columns // len(string)
    pixelsize = _text_int(pixelsize, "pixelsize")
    if not isinstance(fnt, str):
        raise TypeError("fnt must be a Julia String")
    font = _load_font(fnt, pixelsize)

    # FreeTypeAbstraction's renderstring! keywords are concretely typed as
    # Symbol.  Validate after resolving the font, at the same call boundary,
    # instead of letting Python's str(None)/str(0) invent an alignment mode.
    horizontal = _text_alignment(halign, "halign")
    vertical = _text_alignment(valign, "valign")

    foreground, foreground_is_vector = _text_u8_scalar_or_vector(
        options.pop("fcolor", np.uint8(255)), "fcolor"
    )
    glyph_background, background_is_vector = _text_u8_scalar_or_vector(
        options.pop("gcolor", None), "gcolor", allow_none=True
    )
    canvas_background = _text_u8_scalar(
        options.pop("bcolor", np.uint8(0)),
        "bcolor",
        allow_none=True,
    )
    glyph_box = _text_u8_scalar(
        options.pop("bbox_glyph", None),
        "bbox_glyph",
        allow_none=True,
    )
    background_box = _text_u8_scalar(
        options.pop("bbox", None), "bbox", allow_none=True
    )
    background_characters = _text_background_characters(
        options.pop("gstr", None)
    )
    background_inset = _text_int(options.pop("off_bg", 0), "off_bg")
    extra_advance = _text_int(options.pop("incx", 0), "incx")
    if options:
        unexpected = next(iter(options))
        raise TypeError(f"unexpected text-rendering option {unexpected!r}")

    # Typed renderstring! keyword dispatch above precedes its body. Once that
    # succeeds, an explicitly sized empty string reaches first(bitmaps) and
    # fails rather than returning a blank image.
    if not string:
        raise IndexError("cannot render an empty string")

    characters = list(string)
    glyphs = [_glyph_raster(font, character) for character in characters]
    advances = [glyph[3] for glyph in glyphs]
    y_min = min([0] + [bearing_y - bitmap.shape[0] for bitmap, _, bearing_y, _ in glyphs])
    y_max = max([0] + [bearing_y for _, _, bearing_y, _ in glyphs])
    total_advance = sum(advances)

    row_origin = rows // 2
    column_origin = columns // 2

    px = column_origin - (
        total_advance
        if horizontal == "hright"
        else total_advance >> 1
        if horizontal == "hcenter"
        else 0
    )
    py = row_origin + (
        y_max
        if vertical == "vtop"
        else y_min
        if vertical == "vbottom"
        else ((y_max - y_min) >> 1) + y_min
        if vertical == "vcenter"
        else 0
    )

    output = np.zeros((rows, columns), dtype=np.uint8)

    def fill_box(
        r1: int,
        r2: int,
        c1: int,
        c2: int,
        color: Any,
        *,
        clamp: bool = True,
    ) -> None:
        if columns == 0 and rows > 0:
            # Julia permits assignment through its empty second dimension in
            # this orientation, while an empty first dimension still reaches
            # the upstream BoundsError path.
            return
        if clamp:
            r1, r2 = min(max(r1, 1), rows), min(max(r2, 1), rows)
            c1, c2 = min(max(c1, 1), columns), min(max(c2, 1), columns)
        if r1 <= r2 and c1 <= c2:
            if r1 < 1 or r2 > rows or c1 < 1 or c2 > columns:
                raise IndexError("text background index is out of bounds")
            output[r1 - 1 : r2, c1 - 1 : c2] = color

    if canvas_background is not None:
        fill_box(
            py - y_max,
            py - y_min,
            px,
            px + total_advance,
            canvas_background,
        )

    previous: str | None = None
    for index, (character, glyph) in enumerate(zip(characters, glyphs, strict=True)):
        bitmap, bearing_x, bearing_y, advance_x = glyph
        height, width = bitmap.shape
        if index == 0:
            previous = character
        elif previous is not None:
            # FreeTypeAbstraction 0.10.8 never advances ``prev_char`` after
            # the first glyph.  Preserve that successful upstream placement
            # convention instead of silently repairing it to adjacent-pair
            # kerning.
            px += _glyph_kerning(font, previous, character)

        oy = py - bearing_y
        ox = px + bearing_x
        foreground_color = (
            foreground[index] if foreground_is_vector else foreground
        )
        background_color = (
            glyph_background[index]
            if background_is_vector
            else glyph_background
        )

        if background_color is not None:
            if background_characters is not None:
                background_glyph = _glyph_raster(font, background_characters[index])
                background_bitmap, _, background_bearing_y, _ = background_glyph
                glyph_y_min = background_bearing_y - background_bitmap.shape[0]
                glyph_y_max = background_bearing_y
            else:
                glyph_y_min, glyph_y_max = y_min, y_max
            r1 = min(max(py - glyph_y_max, 1), rows)
            r2 = min(max(py - glyph_y_min, 1), rows)
            c1 = min(max(px, 1), columns)
            c2 = min(max(px + advance_x, 1), columns)
            fill_box(
                r1 + background_inset,
                r2 - background_inset,
                c1 + background_inset,
                c2 - background_inset,
                background_color,
                clamp=False,
            )

        for bitmap_row, bitmap_column in zip(*np.nonzero(bitmap), strict=True):
            target_row = oy + int(bitmap_row)
            target_column = ox + int(bitmap_column)
            if not (0 <= target_row < rows and 0 <= target_column < columns):
                continue
            weight = int(bitmap[bitmap_row, bitmap_column]) / 255.0
            if background_color is None:
                color = weight * float(foreground_color)
            else:
                color = weight * float(foreground_color) + (
                    1.0 - weight
                ) * float(background_color)
            output[target_row, target_column] = _round_text_u8(color)

        # FreeTypeAbstraction draws the background bounding box after the
        # blended glyph. This order is observable at overlapping pixels.
        if (
            background_color is not None
            and background_box is not None
            and r2 > r1
            and c2 > c1
        ):
            output[r1 - 1, c1 - 1 : c2] = background_box
            output[r2 - 1, c1 - 1 : c2] = background_box
            output[r1 - 1 : r2, c1 - 1] = background_box
            output[r1 - 1 : r2, c2 - 1] = background_box

        if glyph_box is not None and height and width:
            row_lo = max(0, -oy)
            row_hi = height - 1 - max(0, oy + height - rows)
            column_lo = max(0, -ox)
            column_hi = width - 1 - max(0, ox + width - columns)
            r1, r2 = oy + row_lo, oy + row_hi
            c1, c2 = ox + column_lo, ox + column_hi
            if r2 > r1 and c2 > c1:
                output[r1, c1 : c2 + 1] = glyph_box
                output[r2, c1 : c2 + 1] = glyph_box
                output[r1 : r2 + 1, c1] = glyph_box
                output[r1 : r2 + 1, c2] = glyph_box

        px += advance_x + extra_advance

    return output.astype(np.float64) / 255.0


def lfText(
    template: Any,
    *args: Any,
    center: Any = _OMITTED,
    flambda: Any = _UNSET,
    R: Any = np.float64,
    pixelsize: int | None = None,
    fnt: Any = "arial bold",
    halign: str = "hcenter",
    valign: str = "vcenter",
    **options: Any,
) -> LatticeField:
    """Generate a two-dimensional text field."""

    dtype = _template_data_type(R)
    tag, lattice, _, output_flambda, remaining = _prepare(template, args, center, flambda)
    if len(remaining) != 1 or not isinstance(remaining[0], str):
        raise TypeError("lfText requires one string")
    data = ftaText(
        remaining[0],
        tuple(len(axis) for axis in lattice),
        pixelsize=pixelsize,
        fnt=fnt,
        halign=halign,
        valign=valign,
        **options,
    )
    if R in (Decimal, Fraction, _MPQ, _MPZ):
        prototype = np.empty(data.shape, dtype=object)
        zero = {
            Decimal: Decimal(0),
            Fraction: Fraction(0),
            _MPQ: _MPQ(0),
            _MPZ: _MPZ(0),
        }[R]
        prototype[...] = zero
        data = _julia_assignment_values(data, prototype)
    elif R in (_MPFR, _MPC, _MPFRComplex):
        converted = np.empty(data.shape, dtype=object)
        with _bigfloat_context():
            for index in np.ndindex(data.shape):
                real = _to_mpfr(data[index])
                if R is _MPFR:
                    converted[index] = real
                elif R is _MPC:
                    converted[index] = _MPC(real, _to_mpfr(0))
                else:
                    converted[index] = _MPFRComplex(real, 0)
        data = converted
    else:
        data = _julia_assignment_values(data, dtype).astype(
            dtype, copy=False
        )
    return _standard_output(tag, data, lattice, output_flambda)


def heartQ(x: float, y: float, *args: float) -> float:
    if len(args) == 1:
        scale = args[0]
        return heartQ(x, y, scale, scale * 1.2 / np.sqrt(2), scale / np.sqrt(2))
    if len(args) != 3:
        raise TypeError("heartQ expects scale or w, t, b")
    w, t, b = args
    r = 1 - np.cbrt((x / w) ** 2)
    if r < 0:
        return 0.0
    c = np.sqrt(r)
    yt = t * (2 * c - c**2 - c**3)
    yb = b * (-2 * c - c**2 + c**3)
    return 1.0 if yb <= y <= yt else 0.0


def smileQ(x: float, y: float, *args: float) -> float:
    if len(args) == 1:
        scale = args[0]
        return smileQ(
            x,
            y,
            scale,
            scale * 0.6,
            3 * np.pi / 8,
            scale * 0.05,
            scale * 0.12,
            scale * 0.25,
            scale * 0.3,
            scale * 0.3,
        )
    if len(args) != 8:
        raise TypeError("smileQ expects scale or eight shape parameters")
    hr, mr, ma, mt, erx, ery, ex, ey = args
    if x**2 + y**2 > hr**2:
        return 0.0
    if y < 0 and (mr - mt) ** 2 < x**2 + y**2 < (mr + mt) ** 2 and abs(x / y) < np.tan(ma):
        return 0.1
    if (abs(x) - ex) ** 2 / erx**2 + (y - ey) ** 2 / ery**2 < 1:
        return 0.1
    return 1.0


def pointerOutlineQ(x: float, y: float, *args: float) -> float:
    if len(args) == 1:
        scale = args[0]
        return pointerOutlineQ(
            x,
            y,
            scale * 0.025,
            0.5,
            scale * 0.1,
            scale * 0.7,
            scale * 0.25,
            scale * 0.2,
            scale * 0.15,
            scale * 0.15,
            scale * 0.1,
            scale * 0.2,
            scale * 0.5,
        )
    if len(args) != 11:
        raise TypeError("pointerOutlineQ expects scale or eleven shape parameters")
    bt, bh, fr, l1, l2, l3, l4, tr, l0, hl0, hl1 = args
    radius = (fr * 8 + tr) / 2
    if x <= 0 and (radius - bt) ** 2 <= x**2 + y**2 <= (radius + bt) ** 2:
        return bh
    if -radius - bt <= y <= -radius + bt and 0 <= x <= hl1:
        return bh
    tcx = hl1 + np.asarray([l1, l2, l3, l4])
    tcy = (-radius + fr) + 2 * fr * np.asarray([3, 2, 1, 0])
    if any(
        (fr - bt) ** 2 <= (x - tcx[j]) ** 2 + (y - tcy[j]) ** 2 <= (fr + bt) ** 2
        and x > tcx[j]
        for j in range(4)
    ):
        return bh
    if any(tcy[j] - fr - bt <= y <= tcy[j] - fr + bt and hl1 <= x <= tcx[j] for j in range(4)):
        return bh
    if any((y - (tcy[j] - fr)) ** 2 + (x - hl1) ** 2 <= bt**2 for j in range(3)):
        return bh
    if tcy[0] + fr - bt <= y <= tcy[0] + fr + bt and hl0 <= x <= tcx[0]:
        return bh
    if (x - hl0) ** 2 + (y - (tcy[0] + fr)) ** 2 <= bt**2:
        return bh
    if (
        x >= hl0 + l0
        and y >= tcy[0] + fr
        and (tr - bt) ** 2
        <= (x - (hl0 + l0)) ** 2 + (y - (tcy[0] + fr)) ** 2
        <= (tr + bt) ** 2
    ):
        return bh
    if 0 <= x <= hl0 + l0 and radius - bt <= y <= radius + bt:
        return bh
    return 0.0


def pointerFillQ(x: float, y: float, *args: float) -> float:
    if len(args) == 1:
        scale = args[0]
        return pointerFillQ(
            x,
            y,
            scale * 0.1,
            scale * 0.7,
            scale * 0.25,
            scale * 0.2,
            scale * 0.15,
            scale * 0.15,
            scale * 0.1,
            scale * 0.2,
            scale * 0.5,
        )
    if len(args) != 9:
        raise TypeError("pointerFillQ expects scale or nine shape parameters")
    fr, l1, l2, l3, l4, tr, l0, hl0, hl1 = args
    radius = (fr * 8 + tr) / 2
    tcx = hl1 + np.asarray([l1, l2, l3, l4])
    tcy = (-radius + fr) + 2 * fr * np.asarray([3, 2, 1, 0])
    if x <= 0 and x**2 + y**2 <= radius**2:
        return 1.0
    if -radius <= y <= tcy[0] + fr and 0 <= x <= hl1:
        return 1.0
    if any(tcy[j] - fr <= y <= tcy[j] + fr and hl1 <= x <= tcx[j] for j in range(4)):
        return 1.0
    if any(x > tcx[j] and (x - tcx[j]) ** 2 + (y - tcy[j]) ** 2 <= fr**2 for j in range(4)):
        return 1.0
    if 0 <= x <= hl0 + l0 and tcy[0] + fr <= y <= radius:
        return 1.0
    if x >= hl0 + l0 and y >= tcy[0] + fr and (x - (hl0 + l0)) ** 2 + (y - (tcy[0] + fr)) ** 2 < tr**2:
        return 1.0
    return 0.0


def _emoji(
    template: Any,
    args: tuple[Any, ...],
    generator: Callable[[float, float, float], float],
    *,
    flip: bool,
    center: Any,
    flambda: Any,
) -> LatticeField:
    tag, lattice, centered, output_flambda, remaining = _prepare(template, args, center, flambda)
    if len(remaining) != 1:
        raise TypeError("emoji templates require a scale")
    if len(lattice) != 2:
        raise ValueError("emoji templates require a two-dimensional lattice")
    scale = remaining[0]
    data = np.asarray([[generator(x, y, scale) for y in centered[1]] for x in centered[0]])
    if _julia_bool_condition(flip, "flip"):
        # Julia's ``reverse(transpose(p), dims=1)`` materializes an ``Array``.
        # NumPy's equivalent is a negative-stride view, so copy it before the
        # dense-array constructor gate. Square flipped templates are valid;
        # rectangular ones retain Julia's later shape/lattice failure.
        data = np.flip(data.T, axis=0).copy(order="C")
    return _standard_output(tag, data, lattice, output_flambda)


def lfHeart(
    template: Any,
    *args: Any,
    center: Any = _OMITTED,
    flambda: Any = _UNSET,
    flip: bool = False,
) -> LatticeField:
    """Generate the Julia heart mask on a two-dimensional lattice."""

    return _emoji(template, args, heartQ, flip=flip, center=center, flambda=flambda)


def lfSmile(
    template: Any,
    *args: Any,
    center: Any = _OMITTED,
    flambda: Any = _UNSET,
    flip: bool = False,
) -> LatticeField:
    """Generate the Julia smiley mask on a two-dimensional lattice."""

    return _emoji(template, args, smileQ, flip=flip, center=center, flambda=flambda)


def lfPointer(
    template: Any,
    *args: Any,
    center: Any = _OMITTED,
    flambda: Any = _UNSET,
    flip: bool = False,
) -> LatticeField:
    """Generate the Julia pointing-hand mask on a two-dimensional lattice."""

    def pointer(x: float, y: float, scale: float) -> float:
        border = pointerOutlineQ(x, y, scale)
        return pointerFillQ(x, y, scale) if border == 0 else border

    return _emoji(template, args, pointer, flip=flip, center=center, flambda=flambda)


def lfBlur(field: LatticeField, radius: float) -> LatticeField:
    """Apply the Julia package's circular, shifted-FFT Gaussian blur."""

    if not isinstance(field, LatticeField):
        raise TypeError("lfBlur requires a LatticeField")
    if not _is_real_number(radius):
        raise TypeError("lfBlur radius must be real")
    kernel = lfGaussian(Intensity, field.L, radius).data
    data = isft(sft(kernel) * sft(field.data))
    if np.isrealobj(field.data):
        data = np.abs(data)
    return LatticeField(data, field.L, field.flambda, field_type=field.field_type)
