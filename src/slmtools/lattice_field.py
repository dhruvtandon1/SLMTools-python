"""Lattice-field value types and arithmetic.

This module ports ``LatticeField.jl``.  Field kinds remain *type tags* (rather
than values), and ``LF[Intensity](data, lattice)`` is supported as the closest
Python spelling of Julia's ``LF{Intensity}(data, lattice)``.  The public full
typed spelling is ``LF[Intensity, np.float32, 2](data, lattice)``; like Julia's
``LF{Intensity,Float32,2}``, it checks rather than converts its data and bypasses
the partial tagged constructor's semantic coercions.

Two Julia quirks are deliberate here: phase values are measured in cycles, and
integer-only indexing of a field is linear in Fortran/Julia order.  Python
multi-axis indices are zero-based and slices use normal stop-exclusive Python
semantics.  The Julia constructor-from-another-field bypasses tag coercions;
the bracket constructor preserves that behavior.
"""

from __future__ import annotations

import inspect
import math
import os
from dataclasses import dataclass
from decimal import (
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow as DecimalOverflow,
    getcontext,
    localcontext,
)
from fractions import Fraction
from numbers import Complex, Integral, Number, Real
from typing import Any, Iterator, TypeAlias
import weakref

import gmpy2
import numpy as np

from ._bigfloat import (
    _MPFR,
    _MPC,
    _MPFRComplex as _DecimalComplex,
    _MPQ,
    _MPZ,
    _bigfloat_context,
    _is_bigfloat_input,
    _is_mpfr,
    _mpfr_object_operation,
    _mpfr_pi,
    _mpfr_rtol,
    _mpfr_sincos,
    _mpfr_sqrt,
    _to_mpfr,
    _to_mpfr_array,
)

_NUMPY_NO_VALUE = object()
_NDARRAY_OUT_POSITION_CACHE: dict[str, int | None] = {}
_NDARRAY_OUT_POSITION_FALLBACK: dict[str, int] = {
    "all": 1,
    "any": 1,
    "argmax": 1,
    "argmin": 1,
    "choose": 1,
    "clip": 2,
    "compress": 2,
    "cumprod": 2,
    "cumsum": 2,
    "dot": 1,
    "max": 1,
    "mean": 2,
    "min": 1,
    "prod": 2,
    "ptp": 1,
    "round": 1,
    "std": 2,
    "sum": 2,
    "take": 2,
    "trace": 4,
    "var": 2,
}
_ARRAY_FUNCTION_OUT_POSITION_CACHE: dict[Any, int | None] = {}
_ARRAY_FUNCTION_OUT_POSITION_FALLBACK: dict[Any, int] = {
    np.all: 2,
    np.amax: 2,
    np.amin: 2,
    np.any: 2,
    np.argmax: 2,
    np.argmin: 2,
    np.around: 2,
    np.choose: 2,
    np.concatenate: 2,
    np.concat: 2,
    np.clip: 3,
    np.compress: 3,
    np.cumprod: 3,
    np.cumsum: 3,
    np.dot: 2,
    np.max: 2,
    np.median: 2,
    np.mean: 3,
    np.min: 2,
    np.nanmax: 2,
    np.nanmean: 3,
    np.nanmedian: 2,
    np.nanmin: 2,
    np.nanpercentile: 3,
    np.nanprod: 3,
    np.nanquantile: 3,
    np.nanstd: 3,
    np.nansum: 3,
    np.nanvar: 3,
    np.percentile: 3,
    np.prod: 3,
    np.ptp: 2,
    np.quantile: 3,
    np.round: 2,
    np.stack: 2,
    np.std: 3,
    np.sum: 3,
    np.take: 3,
    np.trace: 5,
    np.var: 3,
}


def _ndarray_method_out_position(name: str) -> int | None:
    """Return ``out``'s positional index after ``self`` for an ndarray method."""

    if name in _NDARRAY_OUT_POSITION_CACHE:
        return _NDARRAY_OUT_POSITION_CACHE[name]
    position: int | None = None
    try:
        parameters = tuple(
            inspect.signature(getattr(np.ndarray, name)).parameters.values()
        )
    except (TypeError, ValueError):
        parameters = ()
    user_position = 0
    for parameter in parameters:
        if parameter.name == "self":
            continue
        if parameter.name == "out":
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                position = user_position
            break
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            user_position += 1
    if position is None:
        position = _NDARRAY_OUT_POSITION_FALLBACK.get(name)
    _NDARRAY_OUT_POSITION_CACHE[name] = position
    return position


def _array_function_out_position(function: Any) -> int | None:
    """Return a NumPy function's positional ``out`` index."""

    if function in _ARRAY_FUNCTION_OUT_POSITION_CACHE:
        return _ARRAY_FUNCTION_OUT_POSITION_CACHE[function]
    position: int | None = None
    try:
        parameters = tuple(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        parameters = ()
    positional_index = 0
    for parameter in parameters:
        if parameter.name == "out":
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                position = positional_index
            break
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_index += 1
    if position is None:
        position = _ARRAY_FUNCTION_OUT_POSITION_FALLBACK.get(function)
    _ARRAY_FUNCTION_OUT_POSITION_CACHE[function] = position
    return position


class DimensionMismatch(ValueError):
    """Python counterpart of Julia's ``DimensionMismatch``."""


class DomainError(ValueError):
    """Python counterpart of Julia's ``DomainError``."""


def _is_dense_ndarray(value: Any) -> bool:
    """Return whether *value* maps to a concrete Julia ``Array``.

    C- and Fortran-contiguous NumPy arrays both own or address one dense
    strided memory block, so either is the Python counterpart of Julia's
    concrete ``Array``.  A non-contiguous view instead corresponds to an
    ``AbstractArray`` wrapper such as ``SubArray`` and must not satisfy
    methods declared specifically for ``Array``/``Vector``.
    """

    if not isinstance(value, np.ndarray):
        return False
    array = (
        value._storage()
        if isinstance(value, _CheckedFieldArray)
        else value
    )
    return bool(array.flags.c_contiguous or array.flags.f_contiguous)


def _require_dense_ndarray(value: Any, name: str) -> np.ndarray:
    """Validate and return a concrete-``Array`` Python argument."""

    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a dense NumPy array")
    if not _is_dense_ndarray(value):
        raise TypeError(
            f"{name} must be C- or Fortran-contiguous; non-contiguous "
            "views correspond to Julia SubArray, not Array"
        )
    return (
        value._storage()
        if isinstance(value, _CheckedFieldArray)
        else np.asarray(value)
    )


class _AbstractFieldTagMeta(type):
    """Keep only Julia's built-in abstract field tags non-instantiable."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls in globals().get("_JULIA_ABSTRACT_FIELD_TAGS", ()):
            raise TypeError(
                f"{cls.__name__} is an abstract field tag and cannot be "
                "instantiated"
            )
        # Declaring a Python subclass is the counterpart of declaring a
        # concrete Julia ``struct <: FieldVal``.  Do not inherit the built-in
        # tags' abstract-instantiation guard into that user-defined class.
        return super().__call__(*args, **kwargs)


class FieldVal(metaclass=_AbstractFieldTagMeta):
    """Base class for Julia-compatible abstract field-value tags."""


class Generic(FieldVal):
    pass


class Phase(FieldVal):
    pass


class RealPhase(Phase):
    pass


UPhase = RealPhase
UnwrappedPhase = RealPhase


class ComplexPhase(Phase):
    pass


S1Phase = ComplexPhase


class Intensity(FieldVal):
    pass


class Amplitude(FieldVal):
    pass


class Modulus(Amplitude):
    pass


RealAmplitude = Modulus
RealAmp = Modulus


class ComplexAmplitude(Amplitude):
    pass


ComplexAmp = ComplexAmplitude


_JULIA_ABSTRACT_FIELD_TAGS = frozenset(
    (
        FieldVal,
        Generic,
        Phase,
        RealPhase,
        ComplexPhase,
        Intensity,
        Amplitude,
        Modulus,
        ComplexAmplitude,
    )
)


def _julia_float_rat(value: Any, dtype: np.dtype[Any]) -> tuple[int, int]:
    """Reproduce Base's continued-fraction probe used by ``StepRangeLen``."""

    scalar_type = np.dtype(dtype).type
    x = scalar_type(value)
    y = scalar_type(x)
    # Base uses ``maxintfloat(narrow(T), Int)``.  Float16 is its own narrow
    # type, Float32 narrows to Float16, and Float64 narrows to Float32.
    limit = (
        16_777_216
        if dtype == np.dtype(np.float64)
        else 2048
    )
    a = d = 1
    b = c = 0
    while np.isfinite(y) and abs(float(y)) <= limit:
        integer = math.trunc(float(y))
        y = scalar_type(y - scalar_type(integer))
        a, c = integer * a + c, a
        b, d = integer * b + d, b
        if max(abs(a), abs(b)) > limit:
            return c, d
        if b != 0 and scalar_type(scalar_type(a) / scalar_type(b)) == x:
            break
        if y == 0:
            break
        # Base performs this continued-fraction reciprocal in the narrow
        # floating type.  Subnormal inputs legitimately overflow to Inf while
        # ending the search; that is not a user-facing numerical warning.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            y = scalar_type(scalar_type(1) / y)
    return a, b


@dataclass(frozen=True, slots=True)
class _TwicePrecision:
    """Python representation of Julia Base's ``TwicePrecision{Float64}``."""

    hi: float
    lo: float


def _canonicalize2(big: float, little: float) -> _TwicePrecision:
    high = float(big + little)
    return _TwicePrecision(high, float((big - high) + little))


def _add12(left: float, right: float) -> _TwicePrecision:
    if abs(right) > abs(left):
        left, right = right, left
    return _canonicalize2(left, right)


def _truncbits64(value: float, bits: int) -> float:
    if bits <= 0:
        return float(value)
    raw = np.asarray(float(value), dtype=np.float64).view(np.uint64)[()]
    mask = np.uint64((2**64 - 1) ^ (2**bits - 1))
    return float(np.asarray(raw & mask, dtype=np.uint64).view(np.float64)[()])


def _tp_from_fraction(value: Fraction) -> _TwicePrecision:
    high = float(value)
    if not math.isfinite(high):
        return _TwicePrecision(high, high)
    return _TwicePrecision(
        high,
        float(value - Fraction.from_float(high)),
    )


def _tp_truncate(value: _TwicePrecision, bits: int) -> _TwicePrecision:
    high = _truncbits64(value.hi, bits)
    return _TwicePrecision(high, float((value.hi - high) + value.lo))


def _tp_add_number(value: _TwicePrecision, scalar: float) -> _TwicePrecision:
    total = _add12(value.hi, float(scalar))
    return _canonicalize2(total.hi, float(total.lo + value.lo))


def _tp_add(left: _TwicePrecision, right: _TwicePrecision) -> _TwicePrecision:
    high = float(left.hi + right.hi)
    if abs(left.hi) > abs(right.hi):
        low = (
            (((left.hi - high) + right.hi) + right.lo)
            + left.lo
        )
    else:
        low = (
            (((right.hi - high) + left.hi) + left.lo)
            + right.lo
        )
    return _canonicalize2(high, float(low))


def _tp_negate(value: _TwicePrecision) -> _TwicePrecision:
    return _TwicePrecision(-value.hi, -value.lo)


def _mul12(left: float, right: float) -> _TwicePrecision:
    high = float(left * right)
    if not math.isfinite(high):
        return _TwicePrecision(high, high)
    # This is the exact residual returned by Julia's fused ``two_mul``.  The
    # Fraction work occurs only while constructing/transforming range
    # metadata, never per array element.
    exact = Fraction.from_float(left) * Fraction.from_float(right)
    low = float(exact - Fraction.from_float(high))
    return _TwicePrecision(high, low)


def _tp_multiply(
    left: _TwicePrecision, scalar: Any
) -> _TwicePrecision:
    if isinstance(scalar, (int, np.integer)) and not isinstance(
        scalar, (bool, np.bool_)
    ):
        integer = int(scalar)
        if integer == 0:
            return _TwicePrecision(left.hi * 0.0, left.lo * 0.0)
        bits = max(0, (abs(integer) - 1).bit_length())
        truncated = _truncbits64(left.hi, bits)
        return _canonicalize2(
            float(truncated * integer),
            float(((left.hi - truncated) + left.lo) * integer),
        )
    right = _TwicePrecision(float(scalar), 0.0)
    product = _mul12(left.hi, right.hi)
    result = _canonicalize2(
        product.hi,
        float(
            (left.hi * right.lo + left.lo * right.hi)
            + product.lo
        ),
    )
    if product.hi == 0.0 or not math.isfinite(product.hi):
        return _TwicePrecision(product.hi, product.hi)
    return result


def _tp_divide(left: _TwicePrecision, scalar: Any) -> _TwicePrecision:
    right = _TwicePrecision(float(scalar), 0.0)
    high = float(left.hi / right.hi)
    product = _mul12(high, right.hi)
    low = float(
        (
            (((left.hi - product.hi) - product.lo) + left.lo)
            - high * right.lo
        )
        / right.hi
    )
    result = _canonicalize2(high, low)
    if high == 0.0 or not math.isfinite(high):
        return _TwicePrecision(high, high)
    return result


def _tp_value(value: _TwicePrecision) -> float:
    return float(value.hi + value.lo)


def _nbitslen(length: int, offset: int) -> int:
    if length < 2:
        return 0
    distance = max(offset, length - offset - 1) - 1
    return min(27, max(0, distance).bit_length() + 1)


def _infer_float_range_metadata(
    first: Any, step: Any, length: int, dtype: np.dtype[Any]
) -> tuple[Any, Any, int, str] | None:
    """Infer Julia's machine-float ``StepRangeLen`` representation."""

    dtype = np.dtype(dtype)
    if dtype not in (
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    ):
        return None
    scalar_type = dtype.type
    start = scalar_type(first)
    increment = scalar_type(step)
    if not np.isfinite(start) or not np.isfinite(increment):
        if dtype == np.dtype(np.float64):
            return (
                _TwicePrecision(float(start), 0.0),
                _TwicePrecision(float(increment), 0.0),
                0,
                "tp",
            )
        return float(start), float(increment), 0, "srl"

    start_n, start_d = _julia_float_rat(start, dtype)
    step_n, step_d = _julia_float_rat(increment, dtype)
    valid = start_d != 0 and step_d != 0
    if valid:
        valid = (
            scalar_type(start_n / start_d) == start
            and scalar_type(step_n / step_d) == increment
        )
    if valid:
        denominator = math.lcm(start_d, step_d)
        if dtype == np.dtype(np.float16):
            max_exact = 2048
        elif dtype == np.dtype(np.float32):
            max_exact = 16_777_216
        else:
            max_exact = 9_007_199_254_740_992
        # Julia evaluates ``den * start`` and ``den * step`` in T here.  In
        # particular, a large common denominator can overflow Float16 even
        # when the corresponding mathematical products are small, forcing
        # StepRangeLen to retain the literal input values instead of their
        # rational approximations.
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_start = scalar_type(denominator) * start
            scaled_step = scalar_type(denominator) * increment
        valid = (
            denominator != 0
            and abs(scaled_start) <= scalar_type(max_exact)
            and abs(scaled_step) <= scalar_type(max_exact)
            and denominator % start_d == 0
            and denominator % step_d == 0
        )
    if not valid:
        if dtype == np.dtype(np.float64):
            return (
                _TwicePrecision(float(start), 0.0),
                _TwicePrecision(float(increment), 0.0),
                0,
                "tp",
            )
        return float(start), float(increment), 0, "srl"

    # These are the same narrow products tested above.  Recomputing them in
    # Float64 can cross a rounding boundary when ``denominator`` itself is not
    # exactly representable in Float16/Float32.
    start_n = round(float(scaled_start))
    step_n = round(float(scaled_step))
    if length == 0:
        reference_fraction = Fraction(start_n, denominator)
        step_fraction = Fraction(step_n, denominator)
        if dtype == np.dtype(np.float64):
            return (
                _tp_from_fraction(reference_fraction),
                _tp_from_fraction(step_fraction),
                0,
                "tp",
            )
        return (
            float(reference_fraction),
            float(step_fraction),
            0,
            "srl",
        )
    if step_n == 0:
        reference_fraction = Fraction(start_n, denominator)
        if dtype == np.dtype(np.float64):
            return (
                _tp_from_fraction(reference_fraction),
                _TwicePrecision(0.0, 0.0),
                0,
                "tp",
            )
        return float(reference_fraction), 0.0, 0, "srl"
    # Julia stores the smallest-magnitude element as its reference. Its offset
    # is one-based; the Python metadata is deliberately zero-based.
    offset = round(-start_n / step_n + 1) - 1
    offset = min(max(offset, 0), length - 1)
    reference_n = start_n + offset * step_n
    if dtype == np.dtype(np.float64):
        reference = _tp_from_fraction(Fraction(reference_n, denominator))
        logical_step = _tp_truncate(
            _tp_from_fraction(Fraction(step_n, denominator)),
            _nbitslen(length, int(offset)),
        )
        return reference, logical_step, int(offset), "tp"
    return (
        reference_n / denominator,
        step_n / denominator,
        int(offset),
        "srl",
    )


def _materialize_range(
    dtype: np.dtype[Any],
    reference: Any,
    logical_step: Any,
    offset: int,
    length: int,
) -> np.ndarray:
    """Materialize a retained Julia range without changing its arithmetic."""

    dtype = np.dtype(dtype)
    _require_materializable_axis_length(length, dtype)
    values = np.empty(length, dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        if isinstance(reference, _TwicePrecision):
            assert isinstance(logical_step, _TwicePrecision)
            for index in range(length):
                shift = index - offset
                shift_high = float(shift * logical_step.hi)
                shift_low = float(shift * logical_step.lo)
                total = _add12(reference.hi, shift_high)
                values[index] = dtype.type(
                    total.hi
                    + (
                        total.lo
                        + (shift_low + reference.lo)
                    )
                )
        else:
            for index in range(length):
                shift = _julia_array_scalar_operation(
                    np.asarray(logical_step),
                    np.int64(index - offset),
                    np.multiply,
                ).reshape(())[()]
                values[index] = dtype.type(
                    _julia_array_scalar_operation(
                        np.asarray(reference),
                        shift,
                        np.add,
                    ).reshape(())[()]
                )
        return values


def _require_materializable_axis_length(
    length: int,
    dtype: np.dtype[Any],
) -> None:
    """Reject genuine platform or host-capacity allocation failures.

    Julia may keep enormous ranges lazy, while ``LatticeAxis`` deliberately
    exposes a NumPy array and therefore must materialize them.  Checking only
    ``intp`` is unsafe on overcommitting operating systems: ``np.empty`` may
    appear to succeed and the subsequent fill can make the kernel kill the
    process.  Derive the ceiling from this host's available/physical memory
    instead of imposing a project-specific element-count cap.
    """

    if length < 0 or length > np.iinfo(np.intp).max:
        raise MemoryError("LatticeAxis length is not representable.")
    itemsize = max(1, int(np.dtype(dtype).itemsize))
    required_bytes = length * itemsize
    page_size: int | None = None
    for name in ("SC_PAGE_SIZE", "SC_PAGESIZE"):
        try:
            page_size = int(os.sysconf(name))
            break
        except (AttributeError, OSError, ValueError):
            continue
    memory_budget: int | None = None
    if page_size is not None:
        try:
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        except (AttributeError, OSError, ValueError):
            available_pages = 0
        if available_pages > 0:
            memory_budget = available_pages * page_size
        else:
            try:
                physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
            except (AttributeError, OSError, ValueError):
                physical_pages = 0
            if physical_pages > 0:
                # Without an available-memory counter, retain half of physical
                # RAM for the interpreter, source operands, and the OS.
                memory_budget = physical_pages * page_size // 2
    if memory_budget is None and os.name == "nt":
        # ``os.sysconf`` is unavailable on Windows. Query the standard Win32
        # available-physical-memory counter without adding a psutil runtime
        # dependency.
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = (
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                )

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)
            ):
                memory_budget = int(status.ullAvailPhys)
        except (AttributeError, OSError):
            memory_budget = None
    if memory_budget is not None and required_bytes > memory_budget:
        raise MemoryError(
            "LatticeAxis cannot eagerly materialize this Julia lazy range "
            f"({required_bytes} bytes exceed the host allocation budget)."
        )


def _julia_integer_binary(left: Any, right: Any, operation: np.ufunc) -> Any:
    """Evaluate one Julia machine-integer scalar operation.

    ``range(start; step, length)`` performs its endpoint arithmetic before
    choosing ``StepRange`` versus ``StepRangeLen``.  NumPy's weak-scalar
    rules and signed/unsigned promotion differ from Julia at exactly the
    overflow boundaries that decide between those two range families, so the
    constructor uses the package's explicit Julia promotion table here.
    """

    with np.errstate(over="ignore", invalid="ignore"):
        return _julia_array_array_operation(
            np.asarray(left), np.asarray(right), operation
        ).reshape(())[()]


def _julia_integer_negate(value: Any) -> Any:
    """Return Julia's machine-integer unary negation, including wrapping."""

    dtype = np.asarray(value).dtype
    if dtype.kind == "b":
        return np.int64(-int(value))
    with np.errstate(over="ignore", invalid="ignore"):
        return np.negative(np.asarray(value), dtype=dtype).reshape(())[()]


def _julia_integer_remainder(left: Any, right: Any) -> Any:
    """Return Julia ``rem`` for the nonnegative StepRange operands.

    The only negative divisor possible here is ``-typemin(T)`` wrapping back
    to ``typemin(T)``.  Julia's remainder keeps the dividend's sign, unlike
    Python's ``%`` for a negative divisor.
    """

    left_value = int(left)
    right_value = int(right)
    if right_value == 0:
        raise ZeroDivisionError("integer division or modulo by zero")
    quotient = abs(left_value) // abs(right_value)
    if (left_value < 0) != (right_value < 0):
        quotient = -quotient
    return left_value - quotient * right_value


def _julia_111_integer_range_unavailable(
    start: Any,
    step: Any,
    length_dtype: np.dtype,
    length_value: int,
) -> bool:
    """Identify the exact unavailable Base 1.11.6 range boundaries.

    These predicates are the 20 distinct typed scalar inputs corresponding
    to 21 entries in the audited matrix (``one(Bool)`` and
    ``typemax(Bool)`` are the same value). Keep the check intentionally
    enumerated: adjacent starts, steps, length values, and length types retain
    their independently verified behavior.
    """

    start_dtype = np.asarray(start).dtype
    step_dtype = np.asarray(step).dtype
    if (
        start_dtype != np.dtype(np.int64)
        or length_dtype != np.dtype(np.int64)
        or length_value != 0
    ):
        return False

    start_value = int(start)
    step_value = int(step)
    int64_info = np.iinfo(np.int64)

    if start_value == int64_info.min:
        if step_dtype == np.dtype(np.bool_):
            return bool(step)
        if step_dtype in (
            np.dtype(np.int8),
            np.dtype(np.int16),
            np.dtype(np.int32),
            np.dtype(np.int64),
        ):
            return step_value == 1
        if step_dtype in (
            np.dtype(np.uint8),
            np.dtype(np.uint16),
            np.dtype(np.uint32),
        ):
            return step_value in (1, int(np.iinfo(step_dtype).max))
        return False

    if start_value == int64_info.min + 1 and step_dtype in (
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
    ):
        return step_value == int(np.iinfo(step_dtype).max)

    if start_value in (int64_info.max - 1, int64_info.max) and step_dtype in (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
    ):
        return step_value == int(np.iinfo(step_dtype).min)

    return False


def _materialize_julia_step_range(
    start: Any,
    step: Any,
    stop: Any,
) -> np.ndarray:
    """Materialize Base ``StepRange(start, step, stop)`` exactly.

    Base normalizes a signed range's terminal element after endpoint
    arithmetic has already wrapped.  Requested ``length`` is consequently
    *not* authoritative for signed machine integers: a nominal two-element
    range can become empty, and the ``typemin`` step can expose Base's
    wraparound edge cases.  This is a direct translation of
    ``steprange_last`` followed by range iteration.
    """

    dtype = np.asarray(stop).dtype
    converted_start = _julia_assignment_values(np.asarray(start), dtype)
    start_value = np.asarray(
        converted_start, dtype=dtype
    ).reshape(())[()]
    stop_value = np.asarray(stop, dtype=dtype).reshape(())[()]
    step_dtype = np.asarray(step).dtype
    zero_step = np.zeros((), dtype=step_dtype)[()]
    if step == zero_step:
        raise ValueError("step cannot be zero")

    if stop_value == start_value:
        last = stop_value
        empty = False
    else:
        step_positive = bool(step > zero_step)
        stop_after_start = bool(stop_value > start_value)
        if step_positive != stop_after_start:
            one_step = np.ones((), dtype=step_dtype)[()]
            last = _julia_integer_binary(
                start_value,
                one_step,
                np.subtract if step_positive else np.add,
            )
            last = np.asarray(last, dtype=dtype).reshape(())[()]
            empty = True
        else:
            empty = False
            if stop_after_start:
                absdiff = _julia_integer_binary(
                    stop_value, start_value, np.subtract
                )
                absstep = step
            else:
                absdiff = _julia_integer_binary(
                    start_value, stop_value, np.subtract
                )
                absstep = _julia_integer_negate(step)

            # Base detects signed subtraction overflow and performs the
            # remainder in the corresponding unsigned type.
            absdiff_dtype = np.asarray(absdiff).dtype
            if absdiff_dtype.kind == "i" and int(absdiff) < 0:
                unsigned_dtype = np.dtype(f"u{absdiff_dtype.itemsize}")
                unsigned_diff = np.asarray(absdiff, dtype=absdiff_dtype).view(
                    unsigned_dtype
                ).reshape(())[()]
                promoted_dtype = _julia_promote_numeric_dtypes(
                    unsigned_dtype, np.asarray(absstep).dtype
                )
                dividend = np.asarray(
                    unsigned_diff, dtype=promoted_dtype
                ).reshape(())[()]
                divisor = np.asarray(
                    absstep, dtype=promoted_dtype
                ).reshape(())[()]
                remain_value = _julia_integer_remainder(dividend, divisor)
            else:
                remain_value = _julia_integer_remainder(absdiff, absstep)
            remain = np.asarray(remain_value, dtype=absdiff_dtype)[()]
            last = _julia_integer_binary(
                stop_value,
                remain,
                np.subtract if stop_after_start else np.add,
            )
            last = np.asarray(last, dtype=dtype).reshape(())[()]

    if empty:
        return np.empty(0, dtype=dtype)

    def trunc_div(left: int, right: int) -> int:
        quotient = abs(left) // abs(right)
        return -quotient if (left < 0) != (right < 0) else quotient

    def wrap_signed(value: int, bits: int) -> int:
        modulus = 1 << bits
        wrapped = value % modulus
        sign = 1 << (bits - 1)
        return wrapped - modulus if wrapped >= sign else wrapped

    # Julia specializes ``length`` for Int64-backed StepRange. Reproduce its
    # modular quotient rather than using the mathematical distance: at a
    # handful of overflow boundaries Base deliberately reports length zero
    # even though ``start``/``stop`` appear directionally compatible.
    diff = _julia_integer_binary(last, start_value, np.subtract)
    step_value = int(step)
    negated_step = int(_julia_integer_negate(step))
    if (
        np.asarray(step).dtype.kind == "u"
        or -1 <= step_value <= 1
        or step_value == negated_step
    ):
        quotient = trunc_div(int(diff), step_value)
    elif step_value < 0:
        negative_diff = _julia_integer_negate(diff)
        unsigned_diff = np.asarray(
            negative_diff, dtype=np.int64
        ).view(np.uint64).reshape(())[()]
        quotient = int(unsigned_diff) // negated_step
    else:
        unsigned_diff = np.asarray(
            diff, dtype=np.int64
        ).view(np.uint64).reshape(())[()]
        quotient = int(unsigned_diff) // step_value
    count = wrap_signed(quotient, dtype.itemsize * 8)
    count = wrap_signed(count + 1, dtype.itemsize * 8)
    if count <= 0:
        return np.empty(0, dtype=dtype)

    # NumPy arrays are materialized while Julia ranges are lazy. Reject only
    # lengths that the platform index type cannot represent; otherwise let
    # NumPy report a genuine allocation failure rather than imposing an
    # arbitrary project-level sample cap.
    if count > np.iinfo(np.intp).max:
        raise MemoryError("LatticeAxis is too large to materialize.")

    # Allocate before iteration so impossible lazy Julia ranges fail promptly
    # at the real eager-storage boundary instead of growing a Python list.
    _require_materializable_axis_length(count, dtype)
    output = np.empty(count, dtype=dtype)
    mathematical_last = int(start_value) + (count - 1) * int(step)
    if mathematical_last == int(last):
        # Ordinary non-wrapping StepRanges can be filled in bounded vector
        # chunks. Keep the scalar translation below for Base's signed-wrap
        # boundary cases. The bounded temporary also avoids doubling a large
        # axis allocation.
        absolute_step = abs(int(step))
        safe_width = (
            1_000_000
            if absolute_step == 0
            else min(
                1_000_000,
                int(np.iinfo(dtype).max) // absolute_step + 1,
            )
        )
        if safe_width >= 2:
            with np.errstate(over="ignore", invalid="ignore"):
                for block_start in range(0, count, safe_width):
                    width = min(safe_width, count - block_start)
                    base = int(start_value) + block_start * int(step)
                    offsets = np.arange(width, dtype=dtype)
                    output[block_start : block_start + width] = np.add(
                        np.asarray(base, dtype=dtype),
                        np.multiply(
                            offsets,
                            np.asarray(step, dtype=dtype),
                            dtype=dtype,
                        ),
                        dtype=dtype,
                    )
            if int(output[-1]) != int(last):
                raise OverflowError(
                    "StepRange iteration did not reach its endpoint."
                )
            return output
    current = start_value
    for index in range(count):
        output[index] = current
        if index + 1 < count:
            current = _julia_integer_binary(current, step, np.add)
            current = np.asarray(current, dtype=dtype).reshape(())[()]
    if output[-1] != last:
        raise OverflowError("StepRange iteration did not reach its endpoint.")
    return output


class LatticeAxis(np.ndarray):
    """Immutable, one-dimensional, regular lattice coordinate axis.

    Julia distinguishes ``AbstractRange`` from an ordinary data array.  This
    small ndarray subclass retains that distinction while remaining directly
    usable in NumPy expressions.
    """

    def __new__(
        cls,
        values: Any,
        step_hint: Any | None = None,
        *,
        _logical_ref: Any | None = None,
        _logical_step: Any | None = None,
        _logical_offset: int | None = None,
        _range_kind: str | None = None,
        _length_kind: str | None = None,
    ) -> "LatticeAxis":
        inherited_length_kind = getattr(values, "_length_kind", None)
        array = np.asarray(values)
        if array.ndim != 1:
            raise DimensionMismatch("A lattice axis must be one-dimensional.")
        result = np.array(array, copy=True).view(cls)
        step_is_logical = step_hint is not None
        if step_hint is None and len(result) >= 2:
            step_hint = (
                np.bool_(True)
                if result.dtype == np.dtype(np.bool_)
                else result[1] - result[0]
            )
        result._step_hint = step_hint
        result._step_hint_is_logical = step_is_logical
        if _logical_ref is not None and _logical_step is not None:
            result._logical_ref = _logical_ref
            result._logical_step = _logical_step
            result._logical_offset = int(0 if _logical_offset is None else _logical_offset)
            result._range_kind = (
                "tp"
                if isinstance(_logical_ref, _TwicePrecision)
                else (_range_kind or "srl")
            )
        elif step_is_logical and len(result):
            if result.dtype.kind in "iu":
                result._logical_ref = result[0]
                result._logical_step = np.asarray(step_hint).reshape(())[()]
                result._logical_offset = 0
                result._range_kind = _range_kind or "ordinal"
            else:
                metadata = _infer_float_range_metadata(
                    result[0], step_hint, len(result), result.dtype
                )
                if metadata is None:
                    result._logical_ref = None
                    result._logical_step = None
                    result._logical_offset = None
                    result._range_kind = None
                else:
                    (
                        result._logical_ref,
                        result._logical_step,
                        result._logical_offset,
                        inferred_kind,
                    ) = metadata
                    result._range_kind = _range_kind or inferred_kind
        else:
            result._logical_ref = None
            result._logical_step = None
            result._logical_offset = None
            result._range_kind = None
        if _length_kind is None:
            _length_kind = inherited_length_kind
        if _length_kind is None:
            if result.dtype == np.dtype(np.uint64):
                _length_kind = "uint64"
            elif result.dtype.kind == "O" and result.size:
                int64 = np.iinfo(np.int64)
                items = tuple(result.flat)
                if all(
                    isinstance(item, _MPZ)
                    or (
                        type(item) is int
                        and not int64.min <= item <= int64.max
                    )
                    for item in items
                ) or all(
                    isinstance(item, _MPQ)
                    or (
                        isinstance(item, Fraction)
                        and (
                            not int64.min <= item.numerator <= int64.max
                            or not 1 <= item.denominator <= int64.max
                        )
                    )
                    for item in items
                ):
                    _length_kind = "bigint"
            if _length_kind is None:
                _length_kind = "int64"
        result._length_kind = _length_kind
        result.setflags(write=False)
        return result

    @classmethod
    def from_start_step(
        cls,
        start: Any,
        step: Any,
        length: Any,
    ) -> "LatticeAxis":
        """Construct Julia's logical integer or machine-float range."""

        if not isinstance(length, (int, np.integer, bool, np.bool_)):
            raise TypeError("length must be a Julia machine integer.")
        length_dtype = _julia_scalar_dtype(length)
        if length_dtype.kind not in "biu":
            raise TypeError("length must be a Julia machine integer.")
        length_scalar = np.asarray(length, dtype=length_dtype)[()]
        length_value = int(length)
        one_length = np.ones((), dtype=length_dtype)[()]
        length_factor = _julia_integer_binary(
            length_scalar, one_length, np.subtract
        )

        start_dtype = _julia_scalar_dtype(start)
        step_dtype = _julia_scalar_dtype(step)
        if start_dtype.kind in "biu" and step_dtype.kind in "biu":
            start_value = np.asarray(start, dtype=start_dtype)[()]
            step_value = np.asarray(step, dtype=step_dtype)[()]
            if _julia_111_integer_range_unavailable(
                start_value,
                step_value,
                length_dtype,
                length_value,
            ):
                raise NotImplementedError(
                    "this exact Julia 1.11.6 machine-integer range "
                    "boundary is unavailable"
                )
            stop_delta = _julia_array_scalar_operation(
                np.asarray(step_value),
                length_factor,
                np.multiply,
            ).reshape(())[()]
            stop = _julia_array_scalar_operation(
                np.asarray(start_value),
                stop_delta,
                np.add,
            ).reshape(())[()]
            dtype = np.asarray(stop).dtype
            if dtype.kind == "i":
                values = _materialize_julia_step_range(
                    start_value, step_value, stop
                )
                reference = np.asarray(
                    _julia_assignment_values(
                        np.asarray(start_value), dtype
                    ),
                    dtype=dtype,
                ).reshape(())[()]
                range_kind = "ordinal"
            else:
                # Unsigned (in practice UInt64) arithmetic cannot use
                # StepRange's signed overflow logic. Julia retains the
                # original reference and step types in StepRangeLen.
                reference = start_value
                range_kind = "srl"
                if length_value < 0:
                    raise ValueError(
                        f"length cannot be negative, got {length_value}"
                    )
                if length_value > np.iinfo(np.intp).max:
                    raise MemoryError("LatticeAxis is too large to materialize.")
                values = _materialize_range(
                    dtype,
                    reference,
                    step_value,
                    0,
                    length_value,
                )
            return cls(
                values,
                step_hint=step_value,
                _logical_ref=reference,
                _logical_step=step_value,
                _logical_offset=0,
                _range_kind=range_kind,
                _length_kind=(
                    "uint64"
                    if length_dtype == np.dtype(np.uint64)
                    else "int64"
                ),
            )
        dtype = _julia_promote_numeric_dtypes(start_dtype, step_dtype)
        if dtype not in (
            np.dtype(np.float16),
            np.dtype(np.float32),
            np.dtype(np.float64),
        ):
            raise TypeError(
                "from_start_step supports Julia machine integer and "
                "machine-float ranges."
            )
        start_value = np.asarray(start, dtype=dtype)[()]
        step_value = np.asarray(step, dtype=dtype)[()]
        if length_value < 0:
            raise ValueError(f"length cannot be negative, got {length_value}")
        if length_value > np.iinfo(np.intp).max:
            raise MemoryError("LatticeAxis is too large to materialize.")

        metadata = _infer_float_range_metadata(
            start_value, step_value, length_value, dtype
        )
        assert metadata is not None
        reference, logical_step, offset, range_kind = metadata
        values = _materialize_range(
            dtype, reference, logical_step, offset, length_value
        )
        return cls(
            values,
            step_hint=(
                dtype.type(0.0)
                if step_value == 0
                else (
                    dtype.type(np.nan)
                    if dtype == np.dtype(np.float64)
                    and not np.isfinite(step_value)
                    else step_value
                )
            ),
            _logical_ref=reference,
            _logical_step=logical_step,
            _logical_offset=offset,
            _range_kind=range_kind,
            _length_kind=(
                "uint64"
                if length_dtype == np.dtype(np.uint64)
                else "int64"
            ),
        )

    def __array_finalize__(self, obj: Any) -> None:
        self._step_hint = getattr(obj, "_step_hint", None)
        self._step_hint_is_logical = getattr(obj, "_step_hint_is_logical", False)
        self._logical_ref = getattr(obj, "_logical_ref", None)
        self._logical_step = getattr(obj, "_logical_step", None)
        self._logical_offset = getattr(obj, "_logical_offset", None)
        self._range_kind = getattr(obj, "_range_kind", None)
        self._length_kind = getattr(obj, "_length_kind", None)
        # ``np.array(axis, copy=True, subok=True)`` bypasses both
        # ``__array_function__`` and ``copy``.  NumPy must keep that fresh
        # destination writable while filling it, so it cannot remain an
        # immutable logical range.  Detect independent storage and discard the
        # inherited metadata; later canonicalization will validate its actual
        # values instead of trusting a stale reference/step.
        if (
            obj is not None
            and self.ndim == 1
            and not np.shares_memory(self, obj)
        ):
            self._step_hint = None
            self._step_hint_is_logical = False
            self._logical_ref = None
            self._logical_step = None
            self._logical_offset = None
            self._range_kind = None
            self._length_kind = None
        # Views inherit the read-only flag from their base.  Fresh ufunc output
        # must remain writable while NumPy fills it; canonicalization through
        # ``_axis`` makes public lattice axes read-only again.

    def copy(self, order: str = "C") -> np.ndarray:
        """Copy true axes immutably; copy dense derived views normally."""

        if self.ndim != 1:
            # NumPy may temporarily propagate the subclass through operations
            # such as ``meshgrid``.  Those are dense coordinate arrays, not
            # lattice axes, and callers need normal writable ndarray copies.
            return np.asarray(self).copy(order=order)
        return LatticeAxis(
            np.asarray(self).copy(order=order),
            step_hint=self._step_hint,
            _logical_ref=self._logical_ref,
            _logical_step=self._logical_step,
            _logical_offset=self._logical_offset,
            _range_kind=self._range_kind,
            _length_kind=self._length_kind,
        )

    def __copy__(self) -> np.ndarray:
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> np.ndarray:
        result = self.copy()
        memo[id(self)] = result
        return result

    def __array_function__(
        self,
        func: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if func is np.copy and args and args[0] is self:
            order = kwargs.get("order", "K")
            if kwargs.get("subok", False):
                return self.copy(order=order)
            return np.asarray(self).copy(order=order)
        return super().__array_function__(func, types, args, kwargs)

    def astype(
        self,
        dtype: Any,
        order: str = "K",
        casting: str = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> np.ndarray:
        """Cast materialized coordinates without propagating stale metadata.

        A dtype-changing elementwise cast is not a Julia range operation: its
        rounded coordinates may no longer be representable by the source
        reference and step.  Return a read-only ordinary array in that case.
        Same-dtype casts retain the exact immutable range representation.
        """

        target = np.dtype(dtype)
        if self.ndim != 1:
            return np.asarray(self).astype(
                target,
                order=order,
                casting=casting,
                subok=False,
                copy=copy,
            )
        if subok and target == self.dtype:
            if not copy:
                return self
            return self.copy(order=order)
        converted = np.asarray(self).astype(
            target,
            order=order,
            casting=casting,
            subok=False,
            copy=copy,
        )
        # ``copy=False`` may return a view of the immutable source; either way,
        # dtype-cast coordinate values are public and must not become mutable.
        converted.setflags(write=False)
        return converted

    def view(self, dtype: Any = None, type: Any = None) -> np.ndarray:
        """Avoid attaching range metadata to dtype-reinterpreted storage."""

        if dtype is None and type is None:
            result = super().view()
        elif type is None:
            result = super().view(dtype)
        elif dtype is None:
            result = super().view(type=type)
        else:
            result = super().view(dtype=dtype, type=type)
        if (
            isinstance(result, LatticeAxis)
            and (result.ndim != 1 or result.dtype != self.dtype)
        ):
            dense = np.asarray(result)
            if self.ndim == 1:
                dense.setflags(write=False)
            return dense
        return result

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Preserve ranges only for Julia's range-preserving scalar ufuncs.

        NumPy otherwise propagates ndarray subclasses while copying their
        metadata verbatim.  That made ``axis * scalar`` writable with the old
        step and made nonlinear results such as ``square(axis)`` masquerade as
        lattices.  Julia instead keeps a range for scalar affine operations
        and materializes ordinary arrays for general broadcasts.
        """

        output = kwargs.get("out")
        if method == "__call__" and output is None:
            axes = [
                (index, value)
                for index, value in enumerate(inputs)
                if isinstance(value, LatticeAxis)
            ]
            if len(axes) == 1 and axes[0][1].ndim == 1:
                axis_index, axis = axes[0]
                if len(inputs) == 1 and ufunc is np.positive:
                    return axis
                if len(inputs) == 1 and ufunc is np.negative:
                    return _logical_axis_scalar_operation(
                        axis, -1, np.multiply
                    )
                if len(inputs) == 2:
                    scalar = inputs[1 - axis_index]
                    if np.asarray(scalar).ndim == 0 and (
                        ufunc in (np.add, np.subtract, np.multiply)
                        or (ufunc is np.divide and axis_index == 0)
                    ):
                        return _logical_axis_scalar_operation(
                            axis,
                            scalar,
                            ufunc,
                            reflected=axis_index == 1,
                        )

        raw_inputs = tuple(
            np.asarray(value) if isinstance(value, LatticeAxis) else value
            for value in inputs
        )
        if output is not None:
            kwargs["out"] = tuple(
                np.asarray(value) if isinstance(value, LatticeAxis) else value
                for value in output
            )
        return getattr(ufunc, method)(*raw_inputs, **kwargs)

    def __getitem__(self, key: Any) -> Any:
        result = super().__getitem__(key)
        if not isinstance(result, LatticeAxis):
            return result
        if isinstance(key, slice):
            selected = range(*key.indices(len(self)))
            stride = selected.step
            first_index = selected.start
            length = len(selected)
            reference = self._logical_ref
            logical_step = self._logical_step
            old_offset = self._logical_offset
            kind = self._range_kind
            if (
                reference is not None
                and logical_step is not None
                and old_offset is not None
                and kind is not None
            ):
                output_kind = kind
                if kind in ("ordinal", "unit"):
                    output_offset = 0
                    reference_shift = _julia_array_scalar_operation(
                        np.asarray(logical_step),
                        np.int64(first_index - old_offset),
                        np.multiply,
                    ).reshape(())[()]
                    new_reference = _julia_array_scalar_operation(
                        np.asarray(reference),
                        reference_shift,
                        np.add,
                    ).reshape(())[()]
                    new_step = _julia_array_scalar_operation(
                        np.asarray(logical_step),
                        np.int64(stride),
                        np.multiply,
                    ).reshape(())[()]
                    if kind == "unit" and stride != 1:
                        output_kind = "ordinal"
                elif kind == "tp":
                    assert isinstance(reference, _TwicePrecision)
                    assert isinstance(logical_step, _TwicePrecision)
                    output_offset = round(
                        (old_offset - first_index) / stride + 1
                    ) - 1
                    output_offset = min(max(output_offset, 0), length - 1)
                    input_reference_index = (
                        first_index + output_offset * stride
                    )
                    if stride == 1 or length < 2:
                        new_step = logical_step
                    else:
                        new_step = _tp_truncate(
                            _tp_multiply(logical_step, stride),
                            _nbitslen(length, output_offset),
                        )
                    output_offset = max(0, output_offset)
                    if input_reference_index == old_offset:
                        new_reference = reference
                    else:
                        new_reference = _tp_add(
                            reference,
                            _tp_multiply(
                                logical_step,
                                input_reference_index - old_offset,
                            ),
                        )
                else:
                    # Base's generic StepRangeLen slicing selects the retained
                    # reference nearest this ordinal selector and always
                    # applies the selector stride, including empty slices.
                    output_offset = round(
                        (old_offset - first_index) / stride
                    )
                    output_offset = max(
                        min(output_offset, length - 1),
                        0,
                    )
                    input_reference_index = (
                        first_index + output_offset * stride
                    )
                    reference_shift = _julia_array_scalar_operation(
                        np.asarray(logical_step),
                        np.int64(input_reference_index - old_offset),
                        np.multiply,
                    ).reshape(())[()]
                    new_reference = _julia_array_scalar_operation(
                        np.asarray(reference),
                        reference_shift,
                        np.add,
                    ).reshape(())[()]
                    new_step = _julia_array_scalar_operation(
                        np.asarray(logical_step),
                        np.int64(stride),
                        np.multiply,
                    ).reshape(())[()]
                values = _materialize_range(
                    self.dtype,
                    new_reference,
                    new_step,
                    output_offset,
                    length,
                )
                visible_step = (
                    _tp_value(new_step)
                    if isinstance(new_step, _TwicePrecision)
                    else new_step
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    step_hint = np.asarray(
                        visible_step, dtype=self.dtype
                    )[()]
                return LatticeAxis(
                    values,
                    step_hint=step_hint,
                    _logical_ref=new_reference,
                    _logical_step=new_step,
                    _logical_offset=output_offset,
                    _range_kind=output_kind,
                    _length_kind=self._length_kind,
                )
            hint = self._step_hint
            result._step_hint = None if hint is None else hint * stride
        else:
            # Julia preserves a range only when it is indexed by another
            # ordinal range.  Fancy/boolean indexing materializes an array;
            # retaining this subclass would attach a fabricated step to
            # potentially irregular coordinates.
            dense = np.asarray(result)
            dense.setflags(write=False)
            return dense
        result.setflags(write=False)
        return result


Lattice: TypeAlias = tuple[LatticeAxis, ...]


class _DefaultFlambda:
    """Omission sentinel whose signature representation stays user-facing."""

    def __repr__(self) -> str:
        return "1.0"


_FLAMBDA_UNSET = _DefaultFlambda()


def _is_real_number(value: Any) -> bool:
    """Recognize Python counterparts of Julia ``Real`` values.

    ``decimal.Decimal`` is registered as a ``Number`` but not as a
    ``numbers.Real``.  It is nevertheless the standard-library scalar closest
    to Julia's arbitrary-precision ``BigFloat``, so constructor metadata must
    accept it while continuing to reject complex wavelengths.
    """

    return isinstance(value, Real) or (
        isinstance(value, Decimal) and not isinstance(value, Complex)
    )


def _is_julia_number(value: Any) -> bool:
    """Recognize scalar values that can represent a Julia ``Number``."""

    return isinstance(value, (Number, _DecimalComplex))


def _logical_object_numeric_key(value: Any) -> type[Any] | None:
    """Classify one object scalar by its concrete Julia numeric type."""

    if isinstance(value, (Decimal, _MPFR)):
        return _MPFR
    if isinstance(value, (_MPC, _DecimalComplex)):
        return _MPC
    if isinstance(value, _MPQ):
        return _MPQ
    if isinstance(value, _MPZ):
        return _MPZ
    if isinstance(value, Fraction):
        limits = np.iinfo(np.int64)
        return (
            Fraction
            if limits.min <= value.numerator <= limits.max
            and 1 <= value.denominator <= limits.max
            else _MPQ
        )
    if type(value) is int and not _is_julia_platform_int(value):
        return _MPZ
    # Ordinary machine numbers stored behind dtype=object correspond to
    # ``Array{Any}``, not a concrete ``Array{T<:Number}``.
    return None


def _object_numeric_element_key(value: Any) -> type[Any] | None:
    """Return the one concrete exact numeric type represented by an array."""

    array = np.asarray(value)
    if array.dtype.kind != "O" or array.size == 0:
        return None
    keys = {_logical_object_numeric_key(item) for item in array.flat}
    if None in keys or len(keys) != 1:
        return None
    return next(iter(keys))


def _logical_object_type_matches(
    value: Any, expected: type[Any]
) -> bool:
    """Check a boxed value against a concrete or abstract Julia type."""

    if expected is Number:
        return _is_julia_number(value)
    if expected is Real:
        return _is_real_number(value)
    if expected is Complex:
        return isinstance(
            value,
            (_MPC, _DecimalComplex, complex, np.complexfloating),
        )
    if expected is object:
        return True
    return type(value) is expected


def _require_julia_numeric_array(value: Any, name: str) -> np.ndarray:
    """Enforce a public Julia ``AbstractArray{T} where T<:Number`` gate.

    Machine numeric dtypes carry their element type directly.  Object arrays
    are the port's storage for Rational/BigFloat-like values, so a nonempty
    homogeneous numeric object array remains valid.  An empty object array has
    no retained numeric element type and corresponds to ``Array{Any}``.
    """

    array = np.asarray(value)
    if array.dtype.kind in "buifc":
        return array
    if array.dtype.kind == "O" and array.size and all(
        _is_julia_number(item) for item in array.flat
    ):
        return array
    raise TypeError(f"{name} requires an array with Julia numeric element type")


def _axis(
    values: Any,
    step_hint: Any | None = None,
    *,
    _logical_ref: Any | None = None,
    _logical_step: Any | None = None,
    _logical_offset: int | None = None,
    _range_kind: str | None = None,
    _length_kind: str | None = None,
) -> LatticeAxis:
    if isinstance(values, LatticeAxis):
        if (
            not values.flags.writeable
            and (step_hint is None or values._step_hint == step_hint)
            and _logical_ref is None
            and _logical_step is None
            and _logical_offset is None
            and _range_kind is None
            and _length_kind is None
        ):
            return values
        return LatticeAxis(
            np.asarray(values),
            step_hint=step_hint,
            _logical_ref=_logical_ref,
            _logical_step=_logical_step,
            _logical_offset=_logical_offset,
            _range_kind=_range_kind,
            _length_kind=_length_kind,
        )
    if isinstance(values, range):
        axis = LatticeAxis.from_start_step(
            np.int64(values.start),
            np.int64(values.step),
            len(values),
        )
        if values.step != 1 or axis._range_kind != "ordinal":
            return axis
        # Python has one ``range`` type. A unit-step instance is the public
        # spelling corresponding to Julia's UnitRange, whose scalar-broadcast
        # overloads differ from an explicit ``start:1:stop`` StepRange.
        return LatticeAxis(
            np.asarray(axis),
            step_hint=axis._step_hint,
            _logical_ref=axis._logical_ref,
            _logical_step=axis._logical_step,
            _logical_offset=axis._logical_offset,
            _range_kind="unit",
            _length_kind=axis._length_kind,
        )
    return LatticeAxis(
        values,
        step_hint=step_hint,
        _logical_ref=_logical_ref,
        _logical_step=_logical_step,
        _logical_offset=_logical_offset,
        _range_kind=_range_kind,
        _length_kind=_length_kind,
    )


def _with_axis_length_kind(axis: Any, length_kind: str) -> LatticeAxis:
    """Copy one logical range while retaining Julia's ``length`` type."""

    canonical = _axis(axis)
    if getattr(canonical, "_length_kind", "int64") == length_kind:
        return canonical
    return LatticeAxis(
        np.asarray(canonical),
        step_hint=canonical._step_hint,
        _logical_ref=canonical._logical_ref,
        _logical_step=canonical._logical_step,
        _logical_offset=canonical._logical_offset,
        _range_kind=canonical._range_kind,
        _length_kind=length_kind,
    )


def as_lattice(lattice: Any) -> Lattice:
    """Canonicalize a tuple/list of one-dimensional coordinate axes."""

    if not isinstance(lattice, (tuple, list)):
        raise TypeError("A lattice must be a tuple or list of coordinate axes.")
    axes = tuple(_axis(item) for item in lattice)
    for axis in axes:
        _validate_regular_axis(axis)
    return axes


def _validate_regular_axis(axis: LatticeAxis) -> None:
    """Reject coordinate arrays that cannot represent Julia ``AbstractRange``.

    ``LatticeAxis`` with an explicit logical step is the Python representation
    of a range such as Julia's ``StepRangeLen``.  Its materialized Float32
    differences need not be identical, so retained logical metadata is
    authoritative.  Plain coordinate arrays, however, must actually be
    regular; accepting arbitrary arrays would be broader than Julia's
    ``Lattice`` type and only postpones a confusing failure.
    """

    values = np.asarray(axis)
    if len(values) < 2:
        return
    if bool(getattr(axis, "_step_hint_is_logical", False)):
        step = axis._step_hint
        try:
            reference = getattr(axis, "_logical_ref", None)
            logical_step = getattr(axis, "_logical_step", None)
            logical_offset = getattr(axis, "_logical_offset", None)
            if (
                reference is not None
                and logical_step is not None
                and logical_offset is not None
            ):
                expected_values = _materialize_range(
                    values.dtype,
                    reference,
                    logical_step,
                    int(logical_offset),
                    len(values),
                )
            else:
                first = values[0]
                expected_values = [
                    first + index * step for index in range(len(values))
                ]
            if values.dtype.kind == "O":
                pairs = tuple(
                    zip(values.tolist(), expected_values, strict=True)
                )
                if any(
                    isinstance(actual, Decimal)
                    or isinstance(expected, Decimal)
                    for actual, expected in pairs
                ):
                    # BigFloat StepRangeLen materialization rounds each
                    # coordinate at the active precision. Reconstructing an
                    # endpoint as ``first + i*step`` can therefore differ in
                    # the final few BigFloat bits even though the retained
                    # logical range is valid.
                    regular = all(
                        _isapprox_scalar(actual, expected)
                        for actual, expected in pairs
                    )
                else:
                    regular = all(
                        actual == expected for actual, expected in pairs
                    )
            elif values.dtype.kind in "buifc":
                expected = np.asarray(expected_values, dtype=values.dtype)
                if values.dtype.kind in "fc":
                    epsilon = np.finfo(values.dtype).eps
                    scale = max(
                        1.0,
                        float(np.nanmax(np.abs(values)))
                        if np.any(np.isfinite(values))
                        else 1.0,
                        float(np.nanmax(np.abs(expected)))
                        if np.any(np.isfinite(expected))
                        else 1.0,
                    )
                    if values.dtype.kind == "c":
                        regular = np.allclose(
                            values,
                            expected,
                            rtol=8 * epsilon,
                            atol=8 * epsilon * scale,
                            equal_nan=True,
                        )
                    else:
                        matching_nonfinite = (
                            (np.isnan(values) & np.isnan(expected))
                            | (np.isposinf(values) & np.isposinf(expected))
                            | (np.isneginf(values) & np.isneginf(expected))
                        )
                        with np.errstate(invalid="ignore", over="ignore"):
                            matching_finite = (
                                np.isfinite(values)
                                & np.isfinite(expected)
                                & (
                                    np.abs(values - expected)
                                    <= 8 * epsilon * scale
                                )
                            )
                        regular = np.all(
                            matching_nonfinite | matching_finite
                        )
                else:
                    regular = np.array_equal(values, expected)
            else:
                regular = False
        except (TypeError, ValueError, OverflowError):
            regular = False
        if not bool(regular):
            raise ValueError(
                "Lattice axis values are inconsistent with their logical step."
            )
        return
    if values.dtype.kind == "O":
        coordinates = values.tolist()
        if not all(_is_julia_number(value) for value in coordinates):
            raise TypeError("Lattice axes must contain numeric coordinates.")
        if _object_contains_mpfr(values):
            with _bigfloat_context():
                differences = [
                    right - left
                    for left, right in zip(
                        coordinates[:-1], coordinates[1:], strict=True
                    )
                ]
        else:
            differences = [
                right - left
                for left, right in zip(
                    coordinates[:-1], coordinates[1:], strict=True
                )
            ]
        candidate = differences[0]
        if not all(difference == candidate for difference in differences):
            raise ValueError("Lattice axes must be regularly spaced.")
        return
    if values.dtype.kind not in "buifc":
        raise TypeError("Lattice axes must contain numeric coordinates.")
    if values.dtype == np.dtype(np.bool_):
        differences = np.diff(values.astype(np.int8))
        if not np.all(differences == 1):
            raise ValueError("Lattice axes must be regularly spaced.")
        return
    differences = np.diff(values)
    candidate = differences[0]
    real_dtype = values.real.dtype
    eps = (
        np.finfo(real_dtype).eps
        if np.issubdtype(real_dtype, np.floating)
        else np.finfo(float).eps
    )
    scale = max(1.0, float(np.abs(candidate)), float(np.max(np.abs(values))))
    try:
        regular = np.allclose(
            differences,
            candidate,
            rtol=16 * eps,
            atol=16 * eps * scale,
        )
    except (TypeError, ValueError, OverflowError):
        regular = False
    if not bool(regular):
        raise ValueError("Lattice axes must be regularly spaced.")


def _is_field_tag(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, FieldVal)


def _is_julia_platform_int(value: Any) -> bool:
    """Whether ``value`` corresponds to Julia's concrete platform ``Int``."""

    if type(value) is int:
        limits = np.iinfo(np.int64)
        return limits.min <= value <= limits.max
    return type(value) is np.int64


def _julia_scalar_dtype(value: Any) -> np.dtype[Any]:
    """Return the Julia scalar type corresponding to a Python/NumPy number."""

    if type(value) is bool:
        return np.dtype(np.bool_)
    if type(value) is int:
        limits = np.iinfo(np.int64)
        if not limits.min <= value <= limits.max:
            return np.dtype(object)
        return np.dtype(np.int64)
    if type(value) is float:
        return np.dtype(np.float64)
    if type(value) is complex:
        return np.dtype(np.complex128)
    if isinstance(
        value,
        (Fraction, Decimal, _MPFR, _MPC, _MPQ, _MPZ, _DecimalComplex),
    ):
        return np.dtype(object)
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "buifc":
        raise TypeError("Scalar arithmetic requires a numeric scalar.")
    return array.dtype


def _julia_asarray(value: Any) -> np.ndarray:
    """Preserve exact scalar identity that NumPy otherwise narrows.

    In particular, ``np.asarray(gmpy2.mpz(1))`` becomes an ``int64`` scalar.
    That erases Julia's distinction between ``BigInt`` and ``Int64`` before
    promotion. Arrays already carry their declared element type, so only
    scalar adapters need explicit object storage here.
    """

    if isinstance(
        value,
        (Fraction, Decimal, _MPFR, _MPC, _MPQ, _MPZ, _DecimalComplex),
    ) or (type(value) is int and not _is_julia_platform_int(value)):
        scalar = np.empty((), dtype=object)
        scalar[()] = value
        return scalar
    return np.asarray(value)


def _julia_promote_numeric_dtypes(
    left: Any,
    right: Any,
    *,
    division: bool = False,
    operation: np.ufunc | None = None,
) -> np.dtype[Any]:
    """Return Julia's promotion for the corresponding machine-number types.

    NumPy deliberately chooses a value-preserving common type in several
    mixed signed/unsigned and integer/float cases.  Julia instead promotes
    machine integers to the widest operand type (using unsigned at equal
    widths), and any machine float fixes the floating precision regardless of
    the integer width.  Those differences are observable both in result
    dtypes and in overflow/rounding, so callers must convert *before* applying
    the operation.
    """

    first = np.dtype(left)
    second = np.dtype(right)
    kinds = (first.kind, second.kind)

    if "c" in kinds:
        real_types = [
            np.empty((), dtype=dtype).real.dtype
            for dtype in (first, second)
            if dtype.kind in "fc"
        ]
        real_dtype = max(real_types, key=lambda dtype: dtype.itemsize)
        promoted = np.promote_types(real_dtype, np.dtype(np.complex64))
    elif "f" in kinds:
        float_types = [dtype for dtype in (first, second) if dtype.kind == "f"]
        promoted = max(float_types, key=lambda dtype: dtype.itemsize)
    else:
        integer_types = [dtype for dtype in (first, second) if dtype.kind != "b"]
        if not integer_types:
            promoted = np.dtype(np.bool_)
        elif len(integer_types) == 1:
            promoted = integer_types[0]
        elif integer_types[0].kind == integer_types[1].kind:
            promoted = max(integer_types, key=lambda dtype: dtype.itemsize)
        else:
            signed = next(dtype for dtype in integer_types if dtype.kind == "i")
            unsigned = next(dtype for dtype in integer_types if dtype.kind == "u")
            # Julia chooses the signed type only when it is strictly wider;
            # equal-width signed/unsigned pairs promote to the unsigned type.
            promoted = signed if signed.itemsize > unsigned.itemsize else unsigned

    # Julia's Bool multiplication is Bool, while +/- are integer arithmetic.
    if first.kind == second.kind == "b" and operation in (np.add, np.subtract):
        promoted = np.dtype(np.int64)
    if division and promoted.kind in "bui":
        promoted = np.dtype(np.float64)
    return np.dtype(promoted)


def _julia_literal_array(value: Any) -> np.ndarray:
    """Construct a Python sequence with Julia array-literal promotion.

    NumPy treats Python/NumPy scalar mixtures as a value-preserving sequence
    conversion.  Julia array literals instead use ``promote_type`` before
    converting any element: ``[Int64(1), Float32(1)]`` is therefore
    ``Vector{Float32}``, not Float64.  Callers use this helper only for plain
    Python list/tuple literals; explicitly typed ndarrays already declare
    their Julia-like element type and must remain untouched.
    """

    source = np.asarray(value, dtype=object)
    if source.size == 0:
        return source

    items = tuple(source.flat)
    if not all(_is_julia_number(item) for item in items):
        if all(isinstance(item, str) for item in items):
            return np.asarray(value)
        if all(isinstance(item, bytes) for item in items):
            return np.asarray(value)
        output = np.empty(source.shape, dtype=object)
        for index in np.ndindex(source.shape):
            output[index] = source[index]
        return output
    has_mpfr_complex = any(
        isinstance(item, (_MPC, _DecimalComplex)) for item in items
    )
    has_machine_complex = any(
        isinstance(item, (complex, np.complexfloating)) for item in items
    )
    has_mpfr = any(isinstance(item, _MPFR) for item in items)
    has_mpq = any(isinstance(item, _MPQ) for item in items)
    has_decimal = any(isinstance(item, Decimal) for item in items)
    has_fraction = any(isinstance(item, Fraction) for item in items)
    has_mpz = any(isinstance(item, _MPZ) for item in items) or any(
        type(item) is int and not _is_julia_platform_int(item)
        for item in items
    )
    if has_mpfr_complex or (
        has_machine_complex
        and (has_mpfr or has_mpq or has_mpz or has_decimal)
    ):
        output = np.empty(source.shape, dtype=object)
        with _bigfloat_context():
            for index in np.ndindex(source.shape):
                item = source[index]
                if isinstance(item, _MPC):
                    output[index] = _MPC(
                        _to_mpfr(item.real), _to_mpfr(item.imag)
                    )
                elif isinstance(item, _DecimalComplex):
                    output[index] = _MPC(item.real, item.imag)
                elif isinstance(item, (complex, np.complexfloating)):
                    output[index] = _MPC(
                        _to_mpfr(item.real), _to_mpfr(item.imag)
                    )
                else:
                    output[index] = _MPC(_to_mpfr(item), _to_mpfr(0))
        return output
    if has_mpfr or (has_decimal and (has_mpq or has_mpz)) or (
        (has_mpq or has_mpz)
        and any(isinstance(item, (float, np.floating)) for item in items)
    ):
        output = np.empty(source.shape, dtype=object)
        with _bigfloat_context():
            for index in np.ndindex(source.shape):
                output[index] = _to_mpfr(source[index])
        return output
    if has_mpq or (has_mpz and has_fraction):
        output = np.empty(source.shape, dtype=object)
        for index in np.ndindex(source.shape):
            item = source[index]
            if isinstance(item, _MPQ):
                output[index] = item
            elif isinstance(item, Fraction):
                output[index] = _MPQ(item.numerator, item.denominator)
            else:
                output[index] = _MPQ(item)
        return output
    if has_mpz:
        output = np.empty(source.shape, dtype=object)
        for index in np.ndindex(source.shape):
            output[index] = _MPZ(source[index])
        return output
    if has_decimal:
        if any(isinstance(item, (complex, np.complexfloating)) for item in items):
            raise TypeError(
                "Complex{BigFloat}-like literal arrays are not supported."
            )
        return _as_decimal_array(source)

    machine_dtypes: list[np.dtype[Any]] = []
    for item in items:
        if isinstance(item, Fraction):
            continue
        machine_dtypes.append(_julia_scalar_dtype(item))

    if has_fraction and not any(
        dtype.kind in "fc" for dtype in machine_dtypes
    ):
        result = np.empty(source.shape, dtype=object)
        for index in np.ndindex(source.shape):
            item = source[index]
            if isinstance(item, Fraction):
                result[index] = item
            elif isinstance(item, (Integral, np.integer, bool, np.bool_)):
                result[index] = Fraction(int(item), 1)
            else:
                raise TypeError("Rational literals require real numeric values.")
        return result

    if not machine_dtypes:
        # The only nonempty case left is a Fraction-only literal, handled by
        # the exact branch above.  Keep this guard defensive.
        return source

    promoted = machine_dtypes[0]
    for dtype in machine_dtypes[1:]:
        promoted = _julia_promote_numeric_dtypes(promoted, dtype)
    converted = _julia_assignment_values(source, promoted)
    return np.asarray(converted, dtype=promoted)


def _julia_field_literal_array(value: list[Any], lattice: Any) -> np.ndarray:
    """Resolve matrix literals versus composite scalar vector elements."""

    target_shape = tuple(len(axis) for axis in as_lattice(lattice))
    expanded = np.asarray(value, dtype=object)
    if expanded.shape == target_shape:
        return _julia_literal_array(value)
    if len(target_shape) == 1 and len(value) == target_shape[0]:
        output = np.empty(len(value), dtype=object)
        for index, item in enumerate(value):
            output[index] = item
        return output
    return _julia_literal_array(value)


def _julia_collect_results(values: Any) -> np.ndarray:
    """Collect comprehension results without expanding composite scalars.

    Julia stores a tuple, vector, array, or ``nothing`` returned by a callback
    as one array element.  NumPy instead interprets homogeneous Python
    sequences as an additional dimension, so explicitly box those values.
    Numeric scalar results retain Julia literal-promotion behavior.
    """

    items = tuple(values)
    must_box = any(
        item is None or isinstance(item, (list, tuple, np.ndarray))
        for item in items
    )
    if not must_box:
        try:
            return _julia_literal_array(items)
        except (TypeError, ValueError):
            # Arbitrary callable results form a Julia comprehension with a
            # widened element type; they are not constrained to Number.
            must_box = True
    if must_box:
        output = np.empty(len(items), dtype=object)
        for index, item in enumerate(items):
            output[index] = item
        return output
    raise AssertionError("unreachable comprehension collector state")


def _julia_collect_comprehension_results(values: Any) -> np.ndarray:
    """Collect callback results using Julia comprehension type joining.

    Unlike an array literal, a comprehension does not call ``promote`` on
    heterogeneous runtime result types.  It widens its element type instead,
    preserving values such as ``Int64(1)`` and ``Float64(2.5)`` side by side.
    Homogeneous numeric results still materialize with their concrete dtype.
    """

    items = tuple(values)
    if not items:
        return np.empty(0, dtype=object)
    if any(
        item is None or isinstance(item, (list, tuple, np.ndarray))
        for item in items
    ):
        output = np.empty(len(items), dtype=object)
        for index, item in enumerate(items):
            output[index] = item
        return output
    def result_type_key(item: Any) -> Any:
        if type(item) is bool or isinstance(item, np.bool_):
            return np.dtype(np.bool_)
        if type(item) is int:
            return (
                np.dtype(np.int64)
                if _is_julia_platform_int(item)
                else _MPZ
            )
        if type(item) is float:
            return np.dtype(np.float64)
        if type(item) is complex:
            return np.dtype(np.complex128)
        if isinstance(item, np.generic):
            return item.dtype
        return type(item)

    if len({result_type_key(item) for item in items}) != 1:
        output = np.empty(len(items), dtype=object)
        for index, item in enumerate(items):
            output[index] = item
        return output
    try:
        return _julia_literal_array(items)
    except (TypeError, ValueError):
        output = np.empty(len(items), dtype=object)
        for index, item in enumerate(items):
            output[index] = item
        return output


def _julia_fill(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Create Julia ``fill(value, shape)`` storage without broadcasting it.

    A Julia vector, tuple, or other object is a scalar cell value here.  NumPy
    normally broadcasts list/tuple/array inputs across the destination, so
    object-like and non-scalar inputs must be assigned cell by cell.  Mutable
    values intentionally retain Julia ``fill`` aliasing semantics.
    """

    if type(value) is int and not _is_julia_platform_int(value):
        value = _MPZ(value)
    scalar = np.asarray(value)
    object_scalar = isinstance(
        value,
        (Fraction, Decimal, _MPFR, _MPC, _MPQ, _MPZ, _DecimalComplex),
    ) or (type(value) is int and not _is_julia_platform_int(value))
    if (
        isinstance(value, np.ndarray)
        or scalar.ndim != 0
        or scalar.dtype.kind == "O"
        or object_scalar
    ):
        output = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            output[index] = value
        return output
    return np.full(shape, value, dtype=scalar.dtype)


def _julia_sum_widened(value: Any) -> np.ndarray:
    """Apply Base.add_sum's scalar accumulator widening."""

    array = _julia_asarray(value)
    if array.dtype.kind == "b":
        return array.astype(np.int64)
    if array.dtype.kind == "i" and array.dtype.itemsize < 8:
        return array.astype(np.int64)
    if array.dtype.kind == "u" and array.dtype.itemsize < 8:
        return array.astype(np.uint64)
    return array


def _julia_typed_zero(value: Any) -> Any:
    """Construct ``zero(typeof(value))`` in the represented numeric context."""

    if isinstance(value, _MPFR):
        return _to_mpfr(0)
    if isinstance(value, _MPC):
        with _bigfloat_context():
            return _MPC(_to_mpfr(0), _to_mpfr(0))
    if isinstance(value, _DecimalComplex):
        return _DecimalComplex(0, 0)
    if isinstance(value, Decimal):
        return Decimal(0)
    if isinstance(value, Fraction):
        return Fraction(0, 1)
    if isinstance(value, _MPQ):
        return _MPQ(0)
    if isinstance(value, _MPZ) or (
        type(value) is int and not _is_julia_platform_int(value)
    ):
        return _MPZ(0)
    return type(value)(0)


def _julia_float_vector_reduce(
    vector: np.ndarray,
    dtype: np.dtype[Any],
) -> Any:
    """Reproduce LLVM's arm64 vector.reduce.fadd lane tree."""

    scalar_type = dtype.type
    # LLVM floating additions quietly propagate IEEE infinities/NaNs. NumPy
    # reports the same arithmetic as RuntimeWarning, which is observably
    # different under ``-W error`` even when the result bits agree.
    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        if len(vector) == 2:
            return scalar_type(vector[0] + vector[1])
        level = np.asarray(vector, dtype=dtype)
        while len(level) > 1:
            level = np.asarray(
                [
                    scalar_type(level[index] + level[index + 1])
                    for index in range(0, len(level), 2)
                ],
                dtype=dtype,
            )
        return level[0]


def _julia_sum_zero(array: np.ndarray) -> Any:
    """Return the typed additive identity used by Julia's ``sum``."""

    if array.dtype.kind != "O":
        zero = np.zeros((), dtype=array.dtype)
        return _julia_sum_widened(zero).reshape(())[()]
    if array.size:
        sample = array.ravel(order="F")[0]
        try:
            return _julia_typed_zero(sample)
        except (TypeError, ValueError):
            pass
    # Python object arrays do not retain their concrete Julia element type
    # when empty.  ``Int`` is the only honest neutral fallback for that
    # representation.
    return np.int64(0)


def _julia_accumulate_value(left: Any, right: Any) -> Any:
    """Apply Julia's checked/nontrapping ``add_sum`` scalar operation."""

    result = _julia_array_array_operation(
        left, right, np.add
    )
    return (
        result.reshape(())[()]
        if np.asarray(result).ndim == 0
        else result
    )


def _julia_hypot_float64(x: float, y: float) -> float:
    """Julia 1.11's FMA-corrected ``hypot(Float64, Float64)``."""

    ax = abs(float(x))
    ay = abs(float(y))
    if math.isinf(ax) or math.isinf(ay):
        return math.inf
    if ay > ax:
        ax, ay = ay, ax
    info = np.finfo(np.float64)
    if ay <= ax * math.sqrt(info.eps / 2):
        return ax
    scale = info.eps * math.sqrt(info.tiny)
    if ax > math.sqrt(info.max / 2):
        ax *= scale
        ay *= scale
        scale = 1.0 / scale
    elif ay < math.sqrt(info.tiny):
        ax /= scale
        ay /= scale
    else:
        scale = 1.0
    h = math.sqrt(math.fma(ax, ax, ay * ay))
    h_squared = h * h
    ax_squared = ax * ax
    correction = (
        math.fma(-ay, ay, h_squared - ax_squared)
        + math.fma(h, h, -h_squared)
        - math.fma(ax, ax, -ax_squared)
    ) / (2 * h)
    return (h - correction) * scale


def _julia_abs(values: Any) -> np.ndarray:
    """Elementwise ``abs`` using Julia's complex ``hypot`` kernels."""

    array = np.asarray(values)
    if array.dtype.kind == "O" and _object_contains_mpfr(array):
        with _bigfloat_context():
            return np.asarray(np.abs(array), dtype=object)
    if array.dtype.kind != "c":
        return np.asarray(np.abs(array))
    component_dtype = array.real.dtype
    output = np.empty(array.shape, dtype=component_dtype)
    if component_dtype == np.dtype(np.float32):
        for index in np.ndindex(array.shape):
            value = array[index]
            real = float(value.real)
            imag = float(value.imag)
            output[index] = (
                np.float32(np.inf)
                if math.isinf(real) or math.isinf(imag)
                else np.float32(
                    math.sqrt(math.fma(real, real, imag * imag))
                )
            )
        return output
    if component_dtype == np.dtype(np.float64):
        for index in np.ndindex(array.shape):
            value = array[index]
            output[index] = _julia_hypot_float64(
                float(value.real), float(value.imag)
            )
        return output
    return np.asarray(np.abs(array))


def _julia_complex_divide_array(
    numerator: Any,
    denominator: Any,
    result_dtype: Any,
    *,
    real_denominator: bool,
) -> np.ndarray:
    """Elementwise complex division using Julia Base's native kernels.

    NumPy evaluates ``ComplexF32 / Real`` through its complex-division
    kernel. Julia instead divides the two Float32 components independently.
    That one operation-order difference is enough to move normalized values
    by an ULP. Complex/complex division likewise has explicit Julia kernels:
    Float32 widens the calculation to Float64 and Float64 uses the robust
    scaled Borges algorithm from ``Base.complex.jl``.
    """

    dtype = np.dtype(result_dtype)
    if dtype.kind != "c":
        raise TypeError("complex division requires a complex result dtype")
    first, second = np.broadcast_arrays(
        np.asarray(numerator), np.asarray(denominator)
    )
    converted_first = first.astype(dtype, copy=False)
    component_dtype = np.empty((), dtype=dtype).real.dtype

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        if real_denominator:
            converted_second = second.astype(component_dtype, copy=False)
            output = np.empty(first.shape, dtype=dtype)
            output.real = np.divide(
                converted_first.real,
                converted_second,
                dtype=component_dtype,
            )
            output.imag = np.divide(
                converted_first.imag,
                converted_second,
                dtype=component_dtype,
            )
            return output

        converted_second = second.astype(dtype, copy=False)
        output = np.empty(first.shape, dtype=dtype)
        float_info = np.finfo(np.float64)

        def float32_divide(z: complex, w: complex) -> complex:
            # ``widen`` maps ComplexF32 to ComplexF64 before the fused
            # multiply-add calculation.
            a = float(z.real)
            b = float(z.imag)
            c = float(w.real)
            d = float(w.imag)
            if math.isinf(c) or math.isinf(d):
                if math.isfinite(a) and math.isfinite(b):
                    real = (
                        np.float64(0.0)
                        * np.sign(np.float64(a))
                        * np.sign(np.float64(c))
                    )
                    imag = (
                        np.float64(-0.0)
                        * np.sign(np.float64(b))
                        * np.sign(np.float64(d))
                    )
                    return complex(float(real), float(imag))
                return complex(math.nan, math.nan)
            magnitude = np.divide(
                np.float64(1.0),
                np.float64(math.fma(c, c, d * d)),
            )
            real = np.float64(math.fma(a, c, b * d)) * magnitude
            imag = np.float64(math.fma(b, c, -(a * d))) * magnitude
            return complex(float(real), float(imag))

        def robust_part(
            a: np.float64,
            b: np.float64,
            c: np.float64,
            d: np.float64,
            ratio: np.float64,
            reciprocal: np.float64,
        ) -> np.float64:
            if ratio != 0:
                product = b * ratio
                if product != 0:
                    return (a + product) * reciprocal
                return a * reciprocal + (b * reciprocal) * ratio
            return (a + d * np.divide(b, c)) * reciprocal

        def robust_divide(
            a: np.float64,
            b: np.float64,
            c: np.float64,
            d: np.float64,
        ) -> tuple[np.float64, np.float64]:
            if abs(d) <= abs(c):
                ratio = np.divide(d, c)
                reciprocal = np.divide(
                    np.float64(1.0), c + d * ratio
                )
                return (
                    robust_part(a, b, c, d, ratio, reciprocal),
                    robust_part(b, -a, c, d, ratio, reciprocal),
                )
            ratio = np.divide(c, d)
            reciprocal = np.divide(np.float64(1.0), d + c * ratio)
            real = robust_part(b, a, d, c, ratio, reciprocal)
            imag = -robust_part(a, -b, d, c, ratio, reciprocal)
            return real, imag

        def float64_divide(z: complex, w: complex) -> complex:
            a = np.float64(z.real)
            b = np.float64(z.imag)
            c = np.float64(w.real)
            d = np.float64(w.imag)
            abs_a = abs(a)
            abs_b = abs(b)
            largest_numerator = abs_a if abs_a >= abs_b else abs_b
            abs_c = abs(c)
            abs_d = abs(d)
            largest_denominator = abs_c if abs_c >= abs_d else abs_d
            if np.isinf(c) or np.isinf(d):
                if np.isfinite(a) and np.isfinite(b):
                    real = (
                        np.float64(0.0) * np.sign(a) * np.sign(c)
                    )
                    imag = (
                        np.float64(-0.0) * np.sign(b) * np.sign(d)
                    )
                    return complex(float(real), float(imag))
                return complex(math.nan, math.nan)

            half_overflow = np.float64(0.5 * float_info.max)
            twice_under_epsilon = np.float64(
                float_info.tiny * 2.0 / float_info.eps
            )
            scale = np.float64(1.0)
            if (
                largest_numerator >= half_overflow
                or largest_numerator <= twice_under_epsilon
                or largest_denominator >= half_overflow
                or largest_denominator <= twice_under_epsilon
            ):
                big_scale = np.float64(
                    2.0 / (float_info.eps * float_info.eps)
                )
                if largest_numerator >= half_overflow:
                    a *= np.float64(0.5)
                    b *= np.float64(0.5)
                    scale *= np.float64(2.0)
                elif largest_numerator <= twice_under_epsilon:
                    a *= big_scale
                    b *= big_scale
                    scale /= big_scale
                if largest_denominator >= half_overflow:
                    c *= np.float64(0.5)
                    d *= np.float64(0.5)
                    scale *= np.float64(0.5)
                elif largest_denominator <= twice_under_epsilon:
                    c *= big_scale
                    d *= big_scale
                    scale *= big_scale
            real, imag = robust_divide(a, b, c, d)
            return complex(float(real * scale), float(imag * scale))

        for index in np.ndindex(first.shape):
            left_value = converted_first[index]
            right_value = converted_second[index]
            if dtype == np.dtype(np.complex64):
                output[index] = float32_divide(
                    complex(left_value), complex(right_value)
                )
            else:
                output[index] = float64_divide(
                    complex(left_value), complex(right_value)
                )
        return output


def _julia_simd_accumulate(
    initial: Any,
    values: np.ndarray,
    dtype: np.dtype[Any],
) -> Any:
    """Reproduce a reducedim ``@simd`` loop with an existing accumulator."""

    length = len(values)
    lanes = {
        np.dtype(np.float16): 8,
        np.dtype(np.float32): 4,
        np.dtype(np.float64): 2,
    }[dtype]
    chunk = 4 * lanes
    vector_length = length - (length % chunk)
    if not vector_length:
        result = initial
        for value in values:
            result = _julia_accumulate_value(result, value)
        return result

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        negative_zero = dtype.type(-0.0)
        accumulators = [
            np.full(lanes, negative_zero, dtype=dtype)
            for _ in range(4)
        ]
        accumulators[0][0] = dtype.type(initial)
        cursor = 0
        while cursor < vector_length:
            for accumulator_index in range(4):
                segment = np.asarray(
                    values[
                        cursor
                        + accumulator_index * lanes :
                        cursor
                        + (accumulator_index + 1) * lanes
                    ],
                    dtype=dtype,
                )
                accumulators[accumulator_index] = np.add(
                    accumulators[accumulator_index],
                    segment,
                    dtype=dtype,
                )
            cursor += chunk
        combined = np.add(accumulators[1], accumulators[0], dtype=dtype)
        combined = np.add(accumulators[2], combined, dtype=dtype)
        combined = np.add(accumulators[3], combined, dtype=dtype)
        result = _julia_float_vector_reduce(combined, dtype)
        for value in values[vector_length:]:
            result = dtype.type(result + value)
        return result


def _julia_sum_sequence(
    values: Any,
    start: int,
    stop: int,
    *,
    native_dtype: np.dtype[Any] | None,
    top_level: bool,
) -> Any:
    """Exact Julia 1.11.6 Base/LLVM add_sum tree on audited arm64."""

    length = stop - start
    if length == 0:
        if isinstance(values, np.ndarray):
            return _julia_sum_zero(values)
        if native_dtype is None:
            return np.int64(0)
        return _julia_sum_zero(np.empty(0, dtype=native_dtype))
    if length == 1:
        only = _julia_sum_widened(values[start])
        return only.reshape(())[()] if only.ndim == 0 else only

    # ``sum`` uses a scalar left fold below 16 elements. Longer inputs enter
    # mapreduce_impl, which recursively halves blocks larger than 1024.
    if top_level and length >= 16:
        return _julia_sum_sequence(
            values,
            start,
            stop,
            native_dtype=native_dtype,
            top_level=False,
        )
    if not top_level and length > 1024:
        left_length = (length - 1) // 2 + 1
        left = _julia_sum_sequence(
            values,
            start,
            start + left_length,
            native_dtype=native_dtype,
            top_level=False,
        )
        right = _julia_sum_sequence(
            values,
            start + left_length,
            stop,
            native_dtype=native_dtype,
            top_level=False,
        )
        combined = _julia_array_array_operation(
            left, right, np.add
        )
        return (
            combined.reshape(())[()]
            if combined.ndim == 0
            else combined
        )

    first = _julia_sum_widened(values[start])
    second = _julia_sum_widened(values[start + 1])
    first_result = _julia_array_array_operation(
        first, second, np.add
    )
    result = (
        first_result.reshape(())[()]
        if first_result.ndim == 0
        else first_result
    )

    use_llvm_block = not top_level or length >= 16
    if (
        use_llvm_block
        and native_dtype
        in (
            np.dtype(np.float16),
            np.dtype(np.float32),
            np.dtype(np.float64),
        )
        and np.asarray(result).ndim == 0
        and np.asarray(values[start]).ndim == 0
    ):
        dtype = np.dtype(native_dtype)
        lanes = {
            np.dtype(np.float16): 8,
            np.dtype(np.float32): 4,
            np.dtype(np.float64): 2,
        }[dtype]
        chunk = 4 * lanes
        remaining = length - 2
        vector_length = remaining - (remaining % chunk)
        if vector_length:
            with np.errstate(
                over="ignore",
                under="ignore",
                invalid="ignore",
                divide="ignore",
            ):
                negative_zero = dtype.type(-0.0)
                accumulators = [
                    np.full(lanes, negative_zero, dtype=dtype)
                    for _ in range(4)
                ]
                accumulators[0][0] = dtype.type(result)
                cursor = start + 2
                vector_stop = cursor + vector_length
                while cursor < vector_stop:
                    for accumulator_index in range(4):
                        segment = np.asarray(
                            values[
                                cursor
                                + accumulator_index * lanes :
                                cursor
                                + (accumulator_index + 1) * lanes
                            ],
                            dtype=dtype,
                        )
                        accumulators[accumulator_index] = np.add(
                            accumulators[accumulator_index],
                            segment,
                            dtype=dtype,
                        )
                    cursor += chunk
                combined = np.add(
                    accumulators[1], accumulators[0], dtype=dtype
                )
                combined = np.add(
                    accumulators[2], combined, dtype=dtype
                )
                combined = np.add(
                    accumulators[3], combined, dtype=dtype
                )
            result = _julia_float_vector_reduce(combined, dtype)
            tail_start = vector_stop
        else:
            tail_start = start + 2
    else:
        tail_start = start + 2

    # Complex{Float32/64} is LLVM-unrolled four elements at a time but remains
    # a component-wise left fold, so this scalar loop is bit-equivalent.
    for index in range(tail_start, stop):
        next_result = _julia_array_array_operation(
            np.asarray(result),
            _julia_sum_widened(values[index]),
            np.add,
        )
        result = (
            next_result.reshape(())[()]
            if next_result.ndim == 0
            else next_result
        )
    return result


def _julia_add_sum(values: Any) -> Any:
    """Reduce an explicit sequence with Julia's Base.add_sum semantics."""

    terms = tuple(values)
    native_dtype: np.dtype[Any] | None = None
    if terms:
        dtypes = [np.asarray(term).dtype for term in terms]
        if all(dtype == dtypes[0] for dtype in dtypes):
            native_dtype = dtypes[0]
    return _julia_sum_sequence(
        terms,
        0,
        len(terms),
        native_dtype=native_dtype,
        top_level=True,
    )


def _julia_sum(
    values: Any,
    axis: Any = None,
    *,
    keepdims: bool = False,
) -> Any:
    """Sum an Array in Julia/F-order with the audited Base/LLVM tree."""

    array = (
        np.asarray(values._storage())
        if isinstance(values, _CheckedFieldArray)
        else np.asarray(values)
    )
    if axis is None:
        flattened = np.asarray(array).ravel(order="F")
        if (
            array.dtype.kind == "O"
            and array.size
            and all(isinstance(value, Decimal) for value in flattened)
        ):
            # Base/mpfr.jl specializes plain ``sum(Array{BigFloat})`` as a
            # strict MPFR fold from typed zero. Axis reductions continue to
            # use the generic reducedim tree below.
            result: Any = Decimal(0)
            for value in flattened:
                result = _julia_accumulate_value(result, value)
            return result
        return _julia_sum_sequence(
            flattened,
            0,
            len(flattened),
            native_dtype=array.dtype,
            top_level=True,
        )

    raw_axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    normalized_axes: set[int] = set()
    for item in raw_axes:
        normalized = int(item)
        if normalized < 0:
            normalized += array.ndim
        if normalized < 0:
            raise np.exceptions.AxisError(axis, ndim=array.ndim)
        # Julia permits positive ``dims`` beyond ``ndims(A)`` and simply
        # leaves the input dimensions unreduced.
        if normalized < array.ndim:
            normalized_axes.add(normalized)
    axes = tuple(sorted(normalized_axes))
    if array.ndim == 0:
        return _julia_accumulate_value(
            _julia_sum_zero(array), array.reshape(())[()]
        )

    # ``sum(A; dims=...)`` in Julia initializes a keep-dimensions result with
    # the widened typed zero, then calls Base._mapreducedim!.  That routine is
    # observably different from independently applying the global sum tree to
    # each NumPy slice, especially for cancellation-sensitive Float32/64
    # inputs.  Mirror its two execution paths below.
    output_shape = tuple(
        1 if dimension in axes else size
        for dimension, size in enumerate(array.shape)
    )
    zero = _julia_sum_zero(array)
    output_dtype = (
        np.dtype(object)
        if array.dtype.kind == "O"
        else np.asarray(zero).dtype
    )
    output = np.empty(output_shape, dtype=output_dtype, order="F")
    output[...] = zero
    if not array.size:
        result = output
    else:
        # Base.check_reducedims returns the physical length of a contiguous
        # leading reduced slice, or zero when a later reduced dimension
        # follows a retained non-singleton one.
        leading_size = 1
        had_nonreduced = False
        for size, result_size in zip(
            array.shape, output_shape, strict=True
        ):
            if result_size == 1:
                if size > 1:
                    if had_nonreduced:
                        leading_size = 0
                    else:
                        leading_size *= size
            else:
                had_nonreduced = True

        if leading_size > 16:
            flattened = array.ravel(order="F")
            flat_output = output.ravel(order="F")
            for output_index in range(flat_output.size):
                start = output_index * leading_size
                reduced = _julia_sum_sequence(
                    flattened,
                    start,
                    start + leading_size,
                    native_dtype=array.dtype,
                    top_level=False,
                )
                flat_output[output_index] = _julia_accumulate_value(
                    flat_output[output_index], reduced
                )
        else:
            tail_shape = array.shape[1:]
            tail_count = math.prod(tail_shape)
            reduced_first_dimension = output_shape[0] == 1
            native_simd_dtype = (
                array.dtype
                if array.dtype
                in (
                    np.dtype(np.float16),
                    np.dtype(np.float32),
                    np.dtype(np.float64),
                )
                else None
            )
            for tail_linear in range(tail_count):
                tail_position = np.unravel_index(
                    tail_linear, tail_shape, order="F"
                )
                output_tail = tuple(
                    0 if dimension in axes else coordinate
                    for dimension, coordinate in enumerate(
                        tail_position, start=1
                    )
                )
                if reduced_first_dimension:
                    destination = (0,) + output_tail
                    current = output[destination]
                    line = np.asarray(
                        array[(slice(None),) + tail_position]
                    )
                    if native_simd_dtype is not None:
                        current = _julia_simd_accumulate(
                            current, line, native_simd_dtype
                        )
                    else:
                        for value in line:
                            current = _julia_accumulate_value(
                                current, value
                            )
                    output[destination] = current
                else:
                    for first_index in range(array.shape[0]):
                        destination = (first_index,) + output_tail
                        output[destination] = _julia_accumulate_value(
                            output[destination],
                            array[(first_index,) + tail_position],
                        )
        result = output

    if not keepdims and axes:
        result = np.squeeze(result, axis=axes)
    if np.asarray(result).ndim == 0:
        return np.asarray(result).reshape(())[()]
    return result


def _julia_cumsum(values: Any, axis: int | None = None) -> np.ndarray:
    """Cumulative ``add_sum`` with Julia ordering and object arithmetic."""

    array = (
        np.asarray(values._storage())
        if isinstance(values, _CheckedFieldArray)
        else np.asarray(values)
    )
    normalized_axis: int | None = None
    if axis is not None:
        normalized_axis = int(axis)
        if normalized_axis < 0:
            normalized_axis += array.ndim
        if normalized_axis < 0 or normalized_axis >= array.ndim:
            raise np.exceptions.AxisError(axis, ndim=array.ndim)

    if axis is None or array.ndim == 1:
        source = array.ravel(order="F")
        output_shape = source.shape
    else:
        assert normalized_axis is not None
        output_shape = array.shape
        moved = np.moveaxis(array, normalized_axis, 0)
        output_moved = np.empty(
            moved.shape,
            dtype=(
                object
                if array.dtype.kind == "O"
                else _julia_sum_widened(
                    np.zeros((), dtype=array.dtype)
                ).dtype
            ),
            order="F",
        )
        tail_shape = moved.shape[1:]
        for tail_linear in range(math.prod(tail_shape)):
            tail = np.unravel_index(tail_linear, tail_shape, order="F")
            source_line = moved[(slice(None),) + tail]
            if len(source_line):
                current = _julia_sum_widened(source_line[0]).reshape(())[()]
                output_moved[(0,) + tail] = current
                for index in range(1, len(source_line)):
                    current = _julia_accumulate_value(
                        current, source_line[index]
                    )
                    output_moved[(index,) + tail] = current
        return np.moveaxis(output_moved, 0, normalized_axis)

    output = np.empty(
        output_shape,
        dtype=(
            object
            if array.dtype.kind == "O"
            else _julia_sum_widened(
                np.zeros((), dtype=array.dtype)
            ).dtype
        ),
    )
    if source.size:
        first = _julia_sum_widened(source[0])
        output[0] = (
            first.reshape(())[()] if first.ndim == 0 else first
        )

        # Julia dispatches vector cumsum for floating, complex, and
        # arithmetic-unknown element types through accumulate_pairwise!.
        # Machine integers use the ordinary strict accumulate! path. The
        # recursive local-sum tree starts only once 127 values remain after
        # the first element. Multidimensional cumsum also uses the strict
        # per-axis accumulate! path handled above.
        def accumulate_pairwise(
            seed: Any,
            start: int,
            length: int,
        ) -> Any:
            if length < 128:
                local = source[start]
                output[start] = _julia_accumulate_value(seed, local)
                for index in range(start + 1, start + length):
                    local = _julia_accumulate_value(local, source[index])
                    output[index] = _julia_accumulate_value(seed, local)
                return local
            left_length = length >> 1
            left = accumulate_pairwise(seed, start, left_length)
            right_seed = _julia_accumulate_value(seed, left)
            right = accumulate_pairwise(
                right_seed,
                start + left_length,
                length - left_length,
            )
            return _julia_accumulate_value(left, right)

        if len(source) > 1:
            if array.dtype.kind in "fcO":
                accumulate_pairwise(output[0], 1, len(source) - 1)
            else:
                current = output[0]
                for index in range(1, len(source)):
                    current = _julia_accumulate_value(
                        current, source[index]
                    )
                    output[index] = current
    return output


def _julia_array_scalar_operation(
    array: Any,
    scalar: Any,
    operation: np.ufunc,
    *,
    reflected: bool = False,
) -> np.ndarray:
    """Apply an array/scalar operation with Julia rather than NumPy promotion."""

    values = _julia_asarray(array)
    scalar_array = _julia_asarray(scalar)
    if _object_contains_gmp(values) or _object_contains_gmp(scalar_array):
        converted_values = values.astype(object, copy=False)
        converted_scalar = scalar_array.reshape(())[()]
        if reflected:
            return _mpfr_object_operation(
                operation, converted_scalar, converted_values
            )
        return _mpfr_object_operation(
            operation, converted_values, converted_scalar
        )
    if (
        values.dtype.kind == "O"
        and values.size > 0
        and all(isinstance(value, Fraction) for value in values.flat)
        and isinstance(
            scalar, (Fraction, bool, int, np.integer, np.bool_)
        )
        and operation in (np.add, np.subtract, np.multiply)
    ):
        scalar_fraction = (
            scalar
            if isinstance(scalar, Fraction)
            else Fraction(int(scalar), 1)
        )
        if reflected:
            return _fraction_int64_array_operation(
                np.asarray(scalar_fraction, dtype=object), values, operation
            )
        return _fraction_int64_array_operation(
            values, np.asarray(scalar_fraction, dtype=object), operation
        )
    if isinstance(scalar, Fraction) and values.dtype.kind in "fc":
        # Julia promotes Rational with a machine Float/Complex to that machine
        # type.  Keeping NumPy object arithmetic here would return an object
        # array and spuriously fail the full typed field constructor.
        component_dtype = np.empty((), dtype=values.dtype).real.dtype
        converted_real = _exact_real_to_machine_float(
            scalar, component_dtype
        )
        converted_scalar = np.asarray(converted_real, dtype=values.dtype)[()]
        if reflected:
            return np.asarray(operation(converted_scalar, values))
        return np.asarray(operation(values, converted_scalar))
    if (
        values.dtype.kind == "O"
        and scalar_array.dtype.kind in "fc"
        and all(isinstance(value, Fraction) for value in values.flat)
    ):
        converted_values = values.astype(scalar_array.dtype)
        converted_scalar = scalar_array.item()
        if reflected:
            return np.asarray(operation(converted_scalar, converted_values))
        return np.asarray(operation(converted_values, converted_scalar))
    if values.dtype.kind == "O" or scalar_array.dtype.kind == "O":
        # NumPy has no native Rational or arbitrary-precision real dtype.
        # Object ufuncs dispatch to the underlying Python numeric operators,
        # preserving Fraction/Decimal arithmetic exactly.  Keep this gate
        # wholly separate from the machine-dtype promotion path below.
        decimal_complex = _object_contains_decimal_complex(
            values
        ) or _object_contains_decimal_complex(scalar_array)
        decimal_real = _object_contains_decimal(
            values
        ) or _object_contains_decimal(scalar_array)
        if decimal_complex:
            converted_values = values.astype(object, copy=False)
            converted_scalar = scalar_array.item()
        elif decimal_real:
            converted_values = _as_decimal_array(values)
            converted_scalar = _as_decimal_approx(scalar_array.item())
        else:
            converted_values = values.astype(object, copy=False)
            converted_scalar = scalar_array.item()
        if reflected:
            if decimal_complex or decimal_real:
                return _decimal_object_operation(
                    operation, converted_scalar, converted_values
                )
            return np.asarray(operation(converted_scalar, converted_values))
        if decimal_complex or decimal_real:
            return _decimal_object_operation(
                operation, converted_values, converted_scalar
            )
        return np.asarray(operation(converted_values, converted_scalar))
    scalar_dtype = _julia_scalar_dtype(scalar)
    result_dtype = _julia_promote_numeric_dtypes(
        values.dtype,
        scalar_dtype,
        division=operation is np.divide,
        operation=operation,
    )
    if operation is np.divide and result_dtype.kind == "c":
        numerator = scalar_array if reflected else values
        denominator = values if reflected else scalar_array
        return _julia_complex_divide_array(
            numerator,
            denominator,
            result_dtype,
            real_denominator=(
                np.asarray(numerator).dtype.kind == "c"
                and np.asarray(denominator).dtype.kind != "c"
            ),
        )
    converted_values = values.astype(result_dtype, copy=False)
    converted_scalar = np.asarray(scalar, dtype=result_dtype)[()]
    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        if reflected:
            return np.asarray(operation(converted_scalar, converted_values))
        return np.asarray(operation(converted_values, converted_scalar))


def _logical_axis_scalar_operation(
    axis: Any,
    scalar: Any,
    operation: np.ufunc,
    *,
    reflected: bool = False,
) -> LatticeAxis:
    """Apply Julia Base's range-preserving dotted scalar operation."""

    canonical = _axis(axis)
    reference = getattr(canonical, "_logical_ref", None)
    logical_step = getattr(canonical, "_logical_step", None)
    offset = getattr(canonical, "_logical_offset", None)
    range_kind = getattr(canonical, "_range_kind", None)
    length_kind = getattr(canonical, "_length_kind", "int64")
    scalar_array = _julia_asarray(scalar)
    exact_range_scalar = isinstance(
        scalar,
        (Decimal, _MPFR, _MPQ, _MPZ),
    ) or (
        type(scalar) is int and not _is_julia_platform_int(scalar)
    ) or (
        isinstance(scalar, Fraction)
        and (
            not _is_julia_platform_int(scalar.numerator)
            or not _is_julia_platform_int(scalar.denominator)
        )
    )
    if (
        operation is np.multiply
        and range_kind == "tp"
        and exact_range_scalar
    ):
        raise TypeError(
            "Julia 1.11 cannot multiply this Float64 StepRangeLen by an "
            "arbitrary-precision scalar"
        )
    if (
        reference is None
        or logical_step is None
        or offset is None
        or range_kind is None
        or canonical.dtype.kind == "O"
        or scalar_array.ndim != 0
        or scalar_array.dtype.kind not in "buif"
        or operation not in (np.add, np.subtract, np.multiply, np.divide)
    ):
        values = _julia_array_scalar_operation(
            np.asarray(canonical), scalar, operation, reflected=reflected
        )
        old_step = canonical._step_hint
        if old_step is None:
            new_step = None
        elif operation in (np.add, np.subtract):
            # A range translation leaves the numerical step unchanged, but
            # Julia still promotes its *type* with the translation scalar.
            # This is observable for BigInt ranges shifted by BigFloat.
            promoted_step = _julia_array_array_operation(
                _julia_asarray(old_step),
                _julia_asarray(_julia_typed_zero(scalar)),
                np.add,
            ).reshape(())[()]
            new_step = (
                _julia_array_scalar_operation(
                    _julia_asarray(promoted_step),
                    np.int64(-1),
                    np.multiply,
                ).reshape(())[()]
                if reflected and operation is np.subtract
                else promoted_step
            )
        elif operation is np.multiply:
            new_step = _julia_array_scalar_operation(
                _julia_asarray(old_step), scalar, np.multiply
            ).reshape(())[()]
        elif reflected:
            new_step = None
        else:
            new_step = _julia_array_scalar_operation(
                _julia_asarray(old_step), scalar, np.divide
            ).reshape(())[()]
        return _axis(
            values,
            step_hint=new_step,
            _length_kind=length_kind,
        )

    scalar_dtype = _julia_scalar_dtype(scalar)
    result_dtype = _julia_promote_numeric_dtypes(
        canonical.dtype,
        scalar_dtype,
        division=operation is np.divide,
        operation=operation,
    )
    converted_scalar = np.asarray(scalar, dtype=result_dtype)[()]
    if reflected and operation is np.divide:
        # Scalar/range division is not range-preserving in Julia.
        values = _julia_array_scalar_operation(
            np.asarray(canonical), scalar, operation, reflected=True
        )
        return _axis(values, _length_kind=length_kind)

    if range_kind in ("ordinal", "unit"):
        is_integer_scalar = scalar_array.dtype.kind in "bui"
        if operation in (np.add, np.subtract) and is_integer_scalar:
            if operation is np.add:
                new_reference = (
                    _julia_array_scalar_operation(
                        np.asarray(reference),
                        scalar,
                        np.add,
                        reflected=True,
                    ).reshape(())[()]
                    if reflected
                    else _julia_array_scalar_operation(
                        np.asarray(reference),
                        scalar,
                        np.add,
                    ).reshape(())[()]
                )
                new_step = logical_step
            elif reflected:
                new_reference = _julia_array_scalar_operation(
                    np.asarray(reference),
                    scalar,
                    np.subtract,
                    reflected=True,
                ).reshape(())[()]
                new_step = _julia_array_scalar_operation(
                    np.asarray(logical_step),
                    np.int64(-1),
                    np.multiply,
                ).reshape(())[()]
            else:
                new_reference = _julia_array_scalar_operation(
                    np.asarray(reference),
                    scalar,
                    np.subtract,
                ).reshape(())[()]
                new_step = logical_step
            generated = LatticeAxis.from_start_step(
                new_reference,
                new_step,
                len(canonical),
            )
            if range_kind == "unit" and (
                operation is np.add
                or (operation is np.subtract and not reflected)
            ):
                return LatticeAxis(
                    np.asarray(generated),
                    step_hint=generated._step_hint,
                    _logical_ref=generated._logical_ref,
                    _logical_step=generated._logical_step,
                    _logical_offset=generated._logical_offset,
                    _range_kind="unit",
                    _length_kind=length_kind,
                )
            return LatticeAxis(
                np.asarray(generated),
                step_hint=generated._step_hint,
                _logical_ref=generated._logical_ref,
                _logical_step=generated._logical_step,
                _logical_offset=generated._logical_offset,
                _range_kind=generated._range_kind,
                _length_kind=length_kind,
            )
        elif (
            operation in (np.add, np.subtract)
            and range_kind == "unit"
            and not (operation is np.subtract and reflected)
        ):
            # AbstractUnitRange +/- Real calls
            # ``range(first +/- x, length=...)``. An explicit unit-step
            # StepRange instead keeps its integer step, so it must continue
            # through the generic start/step/length branch below.
            if operation is np.add:
                new_reference = _julia_array_scalar_operation(
                    np.asarray(reference),
                    scalar,
                    np.add,
                    reflected=reflected,
                ).reshape(())[()]
                new_step = np.asarray(1, dtype=result_dtype)[()]
            elif reflected:
                new_reference = _julia_array_scalar_operation(
                    np.asarray(reference),
                    scalar,
                    np.subtract,
                    reflected=True,
                ).reshape(())[()]
                new_step = np.asarray(-1, dtype=result_dtype)[()]
            else:
                new_reference = _julia_array_scalar_operation(
                    np.asarray(reference),
                    scalar,
                    np.subtract,
                ).reshape(())[()]
                new_step = np.asarray(1, dtype=result_dtype)[()]
            new_kind = "srl"
        else:
            # OrdinalRange .* AbstractFloat and AbstractRange ./ Number route
            # through ``range_start_step_length``; this is exactly where Julia
            # introduces low-float high references or Float64
            # TwicePrecision.
            first_value = np.asarray(reference, dtype=canonical.dtype)[()]
            step_value = np.asarray(logical_step).reshape(())[()]
            if operation is np.multiply:
                start = _julia_array_scalar_operation(
                    np.asarray(first_value),
                    scalar,
                    np.multiply,
                    reflected=reflected,
                ).reshape(())[()]
                step = _julia_array_scalar_operation(
                    np.asarray(step_value),
                    scalar,
                    np.multiply,
                ).reshape(())[()]
            elif operation is np.divide:
                start = _julia_array_scalar_operation(
                    np.asarray(first_value),
                    scalar,
                    np.divide,
                ).reshape(())[()]
                step = _julia_array_scalar_operation(
                    np.asarray(step_value),
                    scalar,
                    np.divide,
                ).reshape(())[()]
            elif operation is np.add:
                start = _julia_array_scalar_operation(
                    np.asarray(first_value),
                    scalar,
                    np.add,
                    reflected=reflected,
                ).reshape(())[()]
                step = step_value
            elif reflected:
                start = _julia_array_scalar_operation(
                    np.asarray(first_value),
                    scalar,
                    np.subtract,
                    reflected=True,
                ).reshape(())[()]
                step = _julia_array_scalar_operation(
                    np.asarray(step_value),
                    np.int64(-1),
                    np.multiply,
                ).reshape(())[()]
            else:
                start = _julia_array_scalar_operation(
                    np.asarray(first_value),
                    scalar,
                    np.subtract,
                ).reshape(())[()]
                step = step_value
            generated = LatticeAxis.from_start_step(
                start,
                step,
                len(canonical),
            )
            if (
                operation is np.multiply
                and is_integer_scalar
            ):
                # The generic integer multiply overload directly constructs a
                # StepRangeLen, even though its values remain integral.
                return LatticeAxis(
                    np.asarray(generated),
                    step_hint=generated._step_hint,
                    _logical_ref=generated._logical_ref,
                    _logical_step=generated._logical_step,
                    _logical_offset=generated._logical_offset,
                    _range_kind="srl",
                    _length_kind=length_kind,
                )
            return LatticeAxis(
                np.asarray(generated),
                step_hint=generated._step_hint,
                _logical_ref=generated._logical_ref,
                _logical_step=generated._logical_step,
                _logical_offset=generated._logical_offset,
                _range_kind=generated._range_kind,
                _length_kind=length_kind,
            )
    elif range_kind == "tp":
        assert isinstance(reference, _TwicePrecision)
        assert isinstance(logical_step, _TwicePrecision)
        high_scalar = float(converted_scalar)
        if operation is np.add:
            new_reference = _tp_add_number(reference, high_scalar)
            new_step = logical_step
        elif operation is np.subtract:
            if reflected:
                new_reference = _tp_add_number(
                    _tp_negate(reference), high_scalar
                )
                new_step = _tp_negate(logical_step)
            else:
                new_reference = _tp_add_number(reference, -high_scalar)
                new_step = logical_step
        elif operation is np.multiply:
            new_reference = _tp_multiply(reference, converted_scalar)
            new_step = _tp_truncate(
                _tp_multiply(logical_step, converted_scalar),
                _nbitslen(len(canonical), int(offset)),
            )
        else:
            new_reference = _tp_divide(reference, converted_scalar)
            # Base's StepRangeLen scalar-division specialization preserves
            # the complete divided TwicePrecision step.  Scalar
            # multiplication, above, deliberately re-truncates it for the
            # range length; the two operations are not implemented as exact
            # inverses in Base.
            new_step = _tp_divide(logical_step, converted_scalar)
        new_kind = "tp"
    else:
        reference_array = np.asarray(reference)
        step_array = np.asarray(logical_step)
        if operation is np.add:
            new_reference = _julia_array_scalar_operation(
                reference_array,
                scalar,
                np.add,
                reflected=reflected,
            ).reshape(())[()]
            new_step = logical_step
        elif operation is np.subtract:
            if reflected:
                new_reference = _julia_array_scalar_operation(
                    reference_array,
                    scalar,
                    np.subtract,
                    reflected=True,
                ).reshape(())[()]
                new_step = _julia_array_scalar_operation(
                    step_array,
                    -1,
                    np.multiply,
                ).reshape(())[()]
            else:
                new_reference = _julia_array_scalar_operation(
                    reference_array,
                    scalar,
                    np.subtract,
                ).reshape(())[()]
                new_step = logical_step
        elif operation is np.multiply:
            new_reference = _julia_array_scalar_operation(
                reference_array,
                scalar,
                np.multiply,
                reflected=reflected,
            ).reshape(())[()]
            new_step = _julia_array_scalar_operation(
                step_array,
                scalar,
                np.multiply,
            ).reshape(())[()]
        else:
            new_reference = _julia_array_scalar_operation(
                reference_array,
                scalar,
                np.divide,
            ).reshape(())[()]
            new_step = _julia_array_scalar_operation(
                step_array,
                scalar,
                np.divide,
            ).reshape(())[()]
        new_kind = "srl"

    values = _materialize_range(
        result_dtype,
        new_reference,
        new_step,
        int(offset),
        len(canonical),
    )
    visible_step = (
        _tp_value(new_step)
        if isinstance(new_step, _TwicePrecision)
        else new_step
    )
    with np.errstate(over="ignore", invalid="ignore"):
        step_hint = np.asarray(visible_step, dtype=result_dtype)[()]
    return _axis(
        values,
        step_hint=step_hint,
        _logical_ref=new_reference,
        _logical_step=new_step,
        _logical_offset=int(offset),
        _range_kind=new_kind,
        _length_kind=length_kind,
    )


def _decimal_rtol() -> Decimal:
    """Context-sized counterpart of Julia's ``sqrt(eps(BigFloat))``."""

    # A p-digit Decimal context represents approximately ceil(p/log10(2))
    # bits. Julia's default isapprox tolerance is sqrt(2^(1-precision)).
    bits = math.ceil(getcontext().prec / math.log10(2))
    return (Decimal(2) ** (1 - bits)).sqrt()


def _object_contains_decimal(value: Any) -> bool:
    array = np.asarray(value)
    if array.dtype.kind != "O":
        return False
    return any(isinstance(item, Decimal) for item in array.flat)


def _object_contains_decimal_complex(value: Any) -> bool:
    array = np.asarray(value)
    if array.dtype.kind != "O":
        return False
    return any(isinstance(item, _DecimalComplex) for item in array.flat)


def _object_contains_mpfr(value: Any) -> bool:
    """Whether an object container participates in Julia BigFloat arithmetic."""

    array = np.asarray(value)
    if array.dtype.kind != "O":
        return False
    return any(
        isinstance(item, (_MPFR, _MPC, _DecimalComplex))
        for item in array.flat
    )


def _object_contains_gmp(value: Any) -> bool:
    """Whether object arithmetic must run in Julia's 256-bit MPFR context."""

    array = np.asarray(value)
    if array.dtype.kind != "O":
        return False
    return any(
        isinstance(item, (_MPFR, _MPC, _MPQ, _MPZ, _DecimalComplex))
        or (type(item) is int and not _is_julia_platform_int(item))
        for item in array.flat
    )


_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


def _checked_int64(value: int) -> int:
    integer = int(value)
    if not _INT64_MIN <= integer <= _INT64_MAX:
        raise OverflowError("Rational{Int64} arithmetic overflowed.")
    return integer


def _checked_int64_add(left: int, right: int) -> int:
    return _checked_int64(int(left) + int(right))


def _checked_int64_subtract(left: int, right: int) -> int:
    return _checked_int64(int(left) - int(right))


def _checked_int64_multiply(left: int, right: int) -> int:
    return _checked_int64(int(left) * int(right))


def _fraction_int64(value: Any) -> Fraction:
    """Validate the concrete Julia counterpart ``Rational{Int64}``."""

    if not isinstance(value, Fraction):
        if isinstance(value, (bool, int, np.integer)):
            value = Fraction(int(value), 1)
        else:
            raise TypeError("expected Rational{Int64}-compatible value")
    _checked_int64(value.numerator)
    if not 1 <= value.denominator <= _INT64_MAX:
        raise OverflowError("Rational{Int64} denominator overflowed.")
    return value


def _fraction_int64_add(left: Any, right: Any) -> Fraction:
    """Translate Base ``+(::Rational{Int64}, ::Rational{Int64})``."""

    x = _fraction_int64(left)
    y = _fraction_int64(right)
    divisor = math.gcd(x.denominator, y.denominator)
    xd = x.denominator // divisor
    yd = y.denominator // divisor
    numerator = _checked_int64_add(
        _checked_int64_multiply(x.numerator, yd),
        _checked_int64_multiply(y.numerator, xd),
    )
    denominator = _checked_int64_multiply(x.denominator, yd)
    return _fraction_int64(Fraction(numerator, denominator))


def _fraction_int64_subtract(left: Any, right: Any) -> Fraction:
    x = _fraction_int64(left)
    y = _fraction_int64(right)
    divisor = math.gcd(x.denominator, y.denominator)
    xd = x.denominator // divisor
    yd = y.denominator // divisor
    numerator = _checked_int64_subtract(
        _checked_int64_multiply(x.numerator, yd),
        _checked_int64_multiply(y.numerator, xd),
    )
    denominator = _checked_int64_multiply(x.denominator, yd)
    return _fraction_int64(Fraction(numerator, denominator))


def _fraction_int64_multiply(left: Any, right: Any) -> Fraction:
    """Translate Base's cross-cancelling checked Rational product."""

    x = _fraction_int64(left)
    y = _fraction_int64(right)
    first_divisor = math.gcd(x.numerator, y.denominator)
    second_divisor = math.gcd(x.denominator, y.numerator)
    xn = x.numerator // first_divisor
    yd = y.denominator // first_divisor
    xd = x.denominator // second_divisor
    yn = y.numerator // second_divisor
    numerator = _checked_int64_multiply(xn, yn)
    denominator = _checked_int64_multiply(xd, yd)
    return _fraction_int64(Fraction(numerator, denominator))


def _fraction_int64_negate(value: Any) -> Fraction:
    rational = _fraction_int64(value)
    if rational.numerator == _INT64_MIN:
        raise OverflowError("Rational{Int64} numerator is typemin(Int64).")
    return Fraction(-rational.numerator, rational.denominator)


def _fraction_int64_array_operation(
    left: Any, right: Any, operation: np.ufunc
) -> np.ndarray:
    first, second = np.broadcast_arrays(
        np.asarray(left, dtype=object),
        np.asarray(right, dtype=object),
    )
    if operation is np.add:
        scalar_operation = _fraction_int64_add
    elif operation is np.subtract:
        scalar_operation = _fraction_int64_subtract
    elif operation is np.multiply:
        scalar_operation = _fraction_int64_multiply
    else:
        return np.asarray(operation(first, second))
    output = np.empty(first.shape, dtype=object)
    for index in np.ndindex(first.shape):
        output[index] = scalar_operation(first[index], second[index])
    return output


def _as_decimal_approx(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, (int, np.integer)):
        return Decimal(int(value))
    if isinstance(value, (float, np.floating)):
        return Decimal.from_float(float(value))
    raise TypeError("value cannot be promoted to Decimal")


def _as_decimal_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    output = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        output[index] = _as_decimal_approx(array[index])
    return output


def _decimal_object_operation(
    operation: np.ufunc,
    left: Any,
    right: Any,
) -> np.ndarray:
    """Apply Decimal arithmetic with Julia BigFloat's nontrapping specials."""

    with localcontext() as context:
        context.traps[DivisionByZero] = False
        context.traps[InvalidOperation] = False
        context.traps[DecimalOverflow] = False
        return np.asarray(operation(left, right))


def _julia_array_array_operation(
    left: Any, right: Any, operation: np.ufunc
) -> np.ndarray:
    """Apply an array operation with Julia numeric promotion.

    The machine-dtype branch explicitly casts both operands to Julia's
    promoted type before evaluating the ufunc.  Falling through to NumPy here
    is incorrect: for example, NumPy promotes ``UInt64 + Int64`` to Float64
    while Julia performs wrapping UInt64 arithmetic.
    """

    first = _julia_asarray(left)
    second = _julia_asarray(right)
    if _object_contains_gmp(first) or _object_contains_gmp(second):
        if operation is np.matmul:
            return _mpfr_object_operation(operation, first, second)
        first, second = np.broadcast_arrays(
            first.astype(object, copy=False),
            second.astype(object, copy=False),
        )
        return _mpfr_object_operation(operation, first, second)
    if _object_contains_decimal_complex(first) or _object_contains_decimal_complex(
        second
    ):
        first, second = np.broadcast_arrays(
            first.astype(object, copy=False),
            second.astype(object, copy=False),
        )
        return _decimal_object_operation(operation, first, second)
    if _object_contains_decimal(first) or _object_contains_decimal(second):
        first, second = np.broadcast_arrays(first, second)
        return _decimal_object_operation(
            operation,
            _as_decimal_array(first),
            _as_decimal_array(second),
        )
    first_is_fraction = (
        first.dtype.kind == "O"
        and first.size > 0
        and all(isinstance(value, Fraction) for value in first.flat)
    )
    second_is_fraction = (
        second.dtype.kind == "O"
        and second.size > 0
        and all(isinstance(value, Fraction) for value in second.flat)
    )
    if first_is_fraction and second_is_fraction:
        return _fraction_int64_array_operation(first, second, operation)
    if first.dtype.kind == "O" and second.dtype.kind in "fc" and all(
        isinstance(value, Fraction) for value in first.flat
    ):
        converted_first = _julia_assignment_values(first, second.dtype)
        return np.asarray(operation(converted_first, second))
    if second.dtype.kind == "O" and first.dtype.kind in "fc" and all(
        isinstance(value, Fraction) for value in second.flat
    ):
        converted_second = _julia_assignment_values(second, first.dtype)
        return np.asarray(operation(first, converted_second))
    if first.dtype.kind in "buifc" and second.dtype.kind in "buifc":
        result_dtype = _julia_promote_numeric_dtypes(
            first.dtype,
            second.dtype,
            division=operation in (np.divide, np.true_divide),
            operation=operation,
        )
        if (
            operation in (np.divide, np.true_divide)
            and result_dtype.kind == "c"
        ):
            return _julia_complex_divide_array(
                first,
                second,
                result_dtype,
                real_denominator=(
                    first.dtype.kind == "c" and second.dtype.kind != "c"
                ),
            )
        converted_first = first.astype(result_dtype, copy=False)
        converted_second = second.astype(result_dtype, copy=False)
        with np.errstate(
            over="ignore",
            under="ignore",
            invalid="ignore",
            divide="ignore",
        ):
            return np.asarray(
                operation(
                    converted_first,
                    converted_second,
                    dtype=result_dtype,
                )
            )
    return np.asarray(operation(first, second))


def _round_fraction_ties_even(value: Fraction) -> int:
    """Round a nonnegative rational to the nearest integer, ties to even."""

    quotient, remainder = divmod(value.numerator, value.denominator)
    comparison = 2 * remainder - value.denominator
    if comparison > 0 or (comparison == 0 and quotient % 2):
        return quotient + 1
    return quotient


def _exact_real_to_machine_float(value: Any, dtype: Any) -> Any:
    """Convert an exact real directly to an IEEE machine float.

    Going through Python ``float`` first double-rounds exact Rational and
    Decimal values on their way to Float16/Float32.  Julia converts directly
    to the destination format.  This routine performs that same
    round-to-nearest, ties-to-even operation with integer arithmetic.
    """

    destination = np.dtype(dtype)
    if destination.kind != "f":
        raise TypeError("destination must be a machine floating dtype")

    if isinstance(value, _MPFR):
        if gmpy2.is_nan(value):
            return np.asarray(math.nan, dtype=destination)[()]
        if gmpy2.is_infinite(value):
            infinity = -math.inf if value < 0 else math.inf
            return np.asarray(infinity, dtype=destination)[()]
        numerator, denominator = value.as_integer_ratio()
        exact = Fraction(int(numerator), int(denominator))
    elif isinstance(value, _MPQ):
        exact = Fraction(int(value.numerator), int(value.denominator))
    elif isinstance(value, _MPZ):
        exact = Fraction(int(value), 1)
    elif isinstance(value, Decimal):
        if value.is_nan():
            return np.asarray(math.nan, dtype=destination)[()]
        if value.is_infinite():
            infinity = -math.inf if value.is_signed() else math.inf
            return np.asarray(infinity, dtype=destination)[()]
        exact = Fraction(value)
    elif isinstance(value, Fraction):
        exact = value
    elif isinstance(value, (int, np.integer, bool, np.bool_)):
        exact = Fraction(int(value), 1)
    else:
        # A machine float is already rounded in its source format. NumPy's
        # IEEE cast correctly rounds that represented value to the target.
        return np.asarray(value, dtype=destination)[()]

    if exact == 0:
        # Fraction and finite Decimal retain a negative zero only in Decimal.
        negative = isinstance(value, Decimal) and value.is_signed()
        return np.asarray(-0.0 if negative else 0.0, dtype=destination)[()]

    negative = exact < 0
    magnitude = abs(exact)
    info = np.finfo(destination)
    precision = info.nmant + 1

    # floor(log2(magnitude)), calculated without an inexact logarithm.
    exponent = (
        magnitude.numerator.bit_length()
        - magnitude.denominator.bit_length()
    )
    if exponent >= 0:
        if magnitude < Fraction(1 << exponent, 1):
            exponent -= 1
    elif magnitude < Fraction(1, 1 << (-exponent)):
        exponent -= 1

    if exponent >= info.minexp:
        quantum_exponent = exponent - (precision - 1)
    else:
        # Every subnormal has the fixed quantum 2^(minexp-nmant).
        quantum_exponent = info.minexp - (precision - 1)

    if quantum_exponent >= 0:
        scaled = magnitude / (1 << quantum_exponent)
    else:
        scaled = magnitude * (1 << (-quantum_exponent))
    significand = _round_fraction_ties_even(scaled)

    if significand == 0:
        result = np.asarray(0.0, dtype=destination)[()]
    else:
        # All finite Float16/Float32 values, and the integer significand used
        # for Float64, are exactly representable in Python's binary64 here.
        result_exponent = quantum_exponent
        if result_exponent + significand.bit_length() - 1 >= info.maxexp:
            result = np.asarray(math.inf, dtype=destination)[()]
        else:
            result = np.asarray(
                math.ldexp(float(significand), result_exponent),
                dtype=destination,
            )[()]
    return -result if negative else result


def _object_destination_element_type(array: np.ndarray | None) -> type[Any] | None:
    """Recover a concrete Julia-like element type from object storage.

    NumPy's object dtype erases the distinction between Rational and
    BigFloat-like arrays.  A homogeneous, nonempty destination still carries
    enough value information to preserve the concrete element type on
    assignment.  Heterogeneous object arrays correspond to Julia abstract
    element storage and intentionally remain unconstrained.
    """

    if array is None or array.dtype.kind != "O" or array.size == 0:
        return None
    types = {type(value) for value in array.flat}
    if len(types) != 1:
        return None
    element_type = next(iter(types))
    if element_type in (
        Fraction,
        Decimal,
        _MPFR,
        _MPC,
        _MPQ,
        _MPZ,
        _DecimalComplex,
    ):
        return element_type
    return None


def _julia_assignment_values(array: Any, dtype: Any) -> np.ndarray:
    """Validate an assignment conversion using Julia's ``setindex!`` rules.

    NumPy deliberately permits several lossy assignments that Julia rejects,
    most importantly fractional/overflowing values into integer arrays and
    genuinely complex values into real arrays.  Perform the conversion check
    before mutating the destination so a failed assignment cannot partially or
    silently corrupt a field.  Floating-point narrowing remains allowed, as it
    is by Julia's floating ``convert`` methods.
    """

    destination_array = dtype if isinstance(dtype, np.ndarray) else None
    destination = np.dtype(
        destination_array.dtype if destination_array is not None else dtype
    )
    source = np.asarray(array)

    source_kind = source.dtype.kind
    destination_kind = destination.kind
    if source_kind in "US" and destination_kind not in "US":
        raise ValueError(
            f"Inexact assignment to {destination}: strings are not numeric."
        )
    if destination_kind in "US" and source_kind not in "US":
        raise ValueError(
            f"Inexact assignment to {destination}: numeric-to-string "
            "conversion is not defined by Julia."
        )

    concrete_object_type = _object_destination_element_type(destination_array)
    if concrete_object_type is not None:
        converted_object = np.empty(source.shape, dtype=object)

        def numeric_parts(value: Any) -> tuple[Any, Any]:
            if isinstance(value, _DecimalComplex):
                return value.real, value.imag
            if isinstance(value, _MPC):
                return value.real, value.imag
            if isinstance(value, Decimal):
                return value, Decimal(0)
            if isinstance(value, Fraction):
                return value, Fraction(0)
            if isinstance(value, (Complex, np.number)):
                return value.real, value.imag
            raise ValueError(
                f"Inexact assignment to object-backed "
                f"{concrete_object_type.__name__}: source value {value!r} "
                "is not numeric."
            )

        for index in np.ndindex(source.shape):
            real_value, imaginary_value = numeric_parts(source[index])
            if concrete_object_type is Fraction:
                if imaginary_value != 0:
                    raise ValueError(
                        "Inexact assignment to Rational: complex values have "
                        "nonzero imaginary parts."
                    )
                try:
                    if isinstance(real_value, Fraction):
                        result = real_value
                    elif isinstance(real_value, Decimal):
                        if not real_value.is_finite():
                            raise ValueError
                        result = Fraction(real_value)
                    elif isinstance(real_value, (float, np.floating)):
                        if not math.isfinite(float(real_value)):
                            raise ValueError
                        result = Fraction.from_float(float(real_value))
                    else:
                        result = Fraction(int(real_value), 1)
                except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
                    raise ValueError("Inexact assignment to Rational.") from error
                # A homogeneous Fraction-backed destination represents the
                # concrete Julia type Rational{Int64}, not the abstract
                # Rational supertype. Fraction itself is unbounded, so enforce
                # the parameter type after canonical reduction; otherwise an
                # assignment Julia rejects would silently change the
                # represented element type to Rational{BigInt}.
                int64 = np.iinfo(np.int64)
                if not (
                    int64.min <= result.numerator <= int64.max
                    and 1 <= result.denominator <= int64.max
                ):
                    raise ValueError(
                        "Inexact assignment to Rational{Int64}: reduced "
                        "numerator or denominator is out of range."
                    )
                converted_object[index] = result
                continue

            if concrete_object_type is _MPQ:
                if imaginary_value != 0:
                    raise ValueError(
                        "Inexact assignment to Rational{BigInt}: complex "
                        "values have nonzero imaginary parts."
                    )
                try:
                    if isinstance(real_value, Fraction):
                        converted_object[index] = _MPQ(
                            real_value.numerator, real_value.denominator
                        )
                    else:
                        converted_object[index] = _MPQ(real_value)
                except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
                    raise ValueError(
                        "Inexact assignment to Rational{BigInt}."
                    ) from error
                continue

            if concrete_object_type is _MPZ:
                if imaginary_value != 0:
                    raise ValueError(
                        "Inexact assignment to BigInt: complex values have "
                        "nonzero imaginary parts."
                    )
                try:
                    converted_integer = _MPZ(real_value)
                    if converted_integer != real_value:
                        raise ValueError
                    converted_object[index] = converted_integer
                except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
                    raise ValueError("Inexact assignment to BigInt.") from error
                continue

            if concrete_object_type is Decimal:
                if imaginary_value != 0:
                    raise ValueError(
                        "Inexact assignment to BigFloat-like storage: complex "
                        "values have nonzero imaginary parts."
                    )
                try:
                    converted_object[index] = _as_decimal_approx(real_value)
                except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
                    raise ValueError(
                        "Inexact assignment to BigFloat-like storage."
                    ) from error
                continue

            if concrete_object_type is _MPFR:
                if imaginary_value != 0:
                    raise ValueError(
                        "Inexact assignment to BigFloat: complex values have "
                        "nonzero imaginary parts."
                    )
                try:
                    with _bigfloat_context():
                        converted_object[index] = _to_mpfr(real_value)
                except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
                    raise ValueError("Inexact assignment to BigFloat.") from error
                continue

            if concrete_object_type is _MPC:
                try:
                    with _bigfloat_context():
                        converted_object[index] = _MPC(
                            _to_mpfr(real_value), _to_mpfr(imaginary_value)
                        )
                except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
                    raise ValueError(
                        "Inexact assignment to Complex{BigFloat}."
                    ) from error
                continue

            try:
                converted_object[index] = _DecimalComplex(
                    _as_decimal_approx(real_value),
                    _as_decimal_approx(imaginary_value),
                )
            except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "Inexact assignment to Complex{BigFloat}-like storage."
                ) from error
        return converted_object

    if np.can_cast(source.dtype, destination, casting="safe"):
        return source

    numeric_source = source_kind in "buifc"

    if numeric_source and destination_kind in "biufc":
        if source_kind == "c":
            if not np.all(np.imag(source) == 0):
                raise ValueError(
                    f"Inexact assignment to {destination}: complex values have "
                    "nonzero imaginary parts."
                )
            real_values = np.real(source)
        else:
            real_values = source

        if destination_kind == "c":
            return source
        if destination_kind == "f":
            return real_values

        if destination_kind == "b":
            if np.all((real_values == 0) | (real_values == 1)):
                return real_values
            raise ValueError(
                f"Inexact assignment to {destination}: values must be 0 or 1."
            )

        info = np.iinfo(destination)
        try:
            exact = np.isfinite(real_values) & (real_values == np.trunc(real_values))
            in_range = (real_values >= info.min) & (real_values <= info.max)
            with np.errstate(invalid="ignore", over="ignore"):
                converted = real_values.astype(destination)
                round_trip = converted.astype(real_values.dtype)
            representable = round_trip == real_values
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Inexact assignment to {destination}.") from error
        if np.all(exact & in_range & representable):
            return real_values
        raise ValueError(
            f"Inexact assignment to {destination}: source values are not "
            "exactly representable."
        )

    # Julia permits ordinary numeric conversion from exact/arbitrary-precision
    # numbers to machine floating-point types.  In particular, assigning a
    # Rational such as 1//3 or a BigFloat such as 0.1 to Float64 is *not* an
    # ``InexactError`` merely because the machine result is rounded.  NumPy's
    # object dtype hides the scalar type, so validate and convert each element
    # explicitly instead of requiring a round trip through the source object.
    if destination_kind == "O":
        return source
    if source_kind == "O" and destination_kind in "biufc":
        converted = np.empty(source.shape, dtype=destination)
        integer_destination = destination_kind in "biu"
        bounds = np.iinfo(destination) if destination_kind in "iu" else None

        for index in np.ndindex(source.shape):
            value = source[index]
            if isinstance(value, _DecimalComplex):
                real_value, imaginary_value = value.real, value.imag
            elif isinstance(value, Decimal):
                real_value, imaginary_value = value, Decimal(0)
            elif isinstance(value, Fraction):
                real_value, imaginary_value = value, Fraction(0)
            elif isinstance(value, (Complex, np.number)):
                real_value, imaginary_value = value.real, value.imag
            else:
                raise ValueError(
                    f"Inexact assignment to {destination}: source value "
                    f"{value!r} is not numeric."
                )

            if destination_kind != "c" and imaginary_value != 0:
                raise ValueError(
                    f"Inexact assignment to {destination}: complex values have "
                    "nonzero imaginary parts."
                )

            if integer_destination:
                try:
                    if isinstance(real_value, Fraction):
                        integral = real_value.denominator == 1
                    elif isinstance(real_value, Decimal):
                        integral = (
                            real_value.is_finite()
                            and real_value == real_value.to_integral_value()
                        )
                    elif isinstance(real_value, (float, np.floating)):
                        integral = math.isfinite(float(real_value)) and float(
                            real_value
                        ).is_integer()
                    else:
                        integral = int(real_value) == real_value
                    integer_value = int(real_value) if integral else 0
                except (ArithmeticError, TypeError, ValueError, OverflowError):
                    integral = False
                    integer_value = 0

                if destination_kind == "b":
                    representable = integral and integer_value in (0, 1)
                else:
                    assert bounds is not None
                    representable = (
                        integral and bounds.min <= integer_value <= bounds.max
                    )
                if not representable:
                    raise ValueError(
                        f"Inexact assignment to {destination}: source values are "
                        "not exactly representable."
                    )
                converted[index] = integer_value
                continue

            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    if destination_kind == "c":
                        component_dtype = np.empty(
                            (), dtype=destination
                        ).real.dtype
                        converted[index] = complex(
                            _exact_real_to_machine_float(
                                real_value, component_dtype
                            ),
                            _exact_real_to_machine_float(
                                imaginary_value, component_dtype
                            ),
                        )
                    else:
                        converted[index] = _exact_real_to_machine_float(
                            real_value, destination
                        )
            except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"Inexact assignment to {destination}.") from error
        return converted

    # Nonnumeric fixed-width values need an exact round trip.
    try:
        converted = source.astype(destination)
        restored = converted.astype(source.dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Inexact assignment to {destination}.") from error
    if not np.array_equal(restored, source):
        raise ValueError(f"Inexact assignment to {destination}.")
    return converted


def _julia_rtol(*arrays: Any) -> Any:
    tolerances: list[Any] = []
    for item in arrays:
        if _object_contains_mpfr(item):
            tolerances.append(_mpfr_rtol())
            continue
        if _object_contains_decimal(item):
            tolerances.append(_decimal_rtol())
            continue
        dtype = np.asarray(item).real.dtype
        if np.issubdtype(dtype, np.floating):
            tolerances.append(float(np.sqrt(np.finfo(dtype).eps)))
        else:
            # Julia's default relative tolerance for two exact numbers is
            # zero. Mixed exact/inexact comparisons use the inexact operand's
            # tolerance; mixed floats use the less precise operand.
            tolerances.append(0.0)
    return max(tolerances, default=0.0)


def _isapprox_array(left: Any, right: Any) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return False
    if a is b or np.array_equal(a, b):
        return True
    if a.dtype.kind == "O" or b.dtype.kind == "O":
        tolerance = _julia_rtol(a, b)
        if isinstance(tolerance, Decimal):
            try:
                left_values = [_as_decimal_approx(value) for value in a.flat]
                right_values = [_as_decimal_approx(value) for value in b.flat]
                difference = sum(
                    (
                        (x - y) ** 2
                        for x, y in zip(
                            left_values, right_values, strict=True
                        )
                    ),
                    Decimal(0),
                ).sqrt()
                scale = max(
                    sum((value**2 for value in left_values), Decimal(0)).sqrt(),
                    sum((value**2 for value in right_values), Decimal(0)).sqrt(),
                )
                return difference <= tolerance * scale
            except (TypeError, ValueError, ArithmeticError):
                return False
        if tolerance == 0:
            return all(
                x == y for x, y in zip(a.flat, b.flat, strict=True)
            )
        try:
            difference_values = _julia_array_array_operation(
                a, b, np.subtract
            )
            difference = float(np.linalg.norm(difference_values))

            def object_norm(values: np.ndarray) -> float:
                if values.dtype.kind != "O":
                    return float(np.linalg.norm(values))
                return math.sqrt(
                    sum(float(abs(value)) ** 2 for value in values.flat)
                )

            scale = max(
                object_norm(a),
                object_norm(b),
            )
            if np.isfinite(difference):
                return difference <= float(tolerance) * scale
        except (TypeError, ValueError, ArithmeticError, OverflowError):
            pass
        return all(
            _isapprox_scalar(x, y, rtol=tolerance)
            for x, y in zip(a.flat, b.flat, strict=True)
        )
    try:
        tolerance = _julia_rtol(a, b)
        difference_values = _julia_array_array_operation(
            a, b, np.subtract
        )
        norm_values = difference_values
        if norm_values.dtype.kind in "fc" and norm_values.dtype.itemsize < 8:
            norm_values = norm_values.astype(
                np.complex128 if norm_values.dtype.kind == "c" else np.float64
            )
        difference = float(np.linalg.norm(norm_values))
        if np.isfinite(difference):
            if tolerance == 0:
                return difference <= 0
            norm_a = a
            norm_b = b
            if a.dtype.kind in "fc" and a.dtype.itemsize < 8:
                widened = np.complex128 if a.dtype.kind == "c" else np.float64
                norm_a = a.astype(widened)
                norm_b = b.astype(widened)
            scale = max(
                float(np.linalg.norm(norm_a)),
                float(np.linalg.norm(norm_b)),
            )
            return difference <= tolerance * scale
        return all(
            _isapprox_scalar(x, y, rtol=tolerance)
            for x, y in zip(a.flat, b.flat, strict=True)
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _isapprox_scalar(
    left: Real, right: Real, *, rtol: Any | None = None
) -> bool:
    if left == right:
        return True
    tolerance = _julia_rtol(left, right) if rtol is None else rtol
    if tolerance == 0:
        return False
    try:
        finite = math.isfinite(left) and math.isfinite(right)
    except (TypeError, ValueError, OverflowError):
        finite = False
    if not finite:
        return False
    if isinstance(tolerance, Decimal):
        try:
            decimal_left = _as_decimal_approx(left)
            decimal_right = _as_decimal_approx(right)
        except TypeError:
            return False
        return abs(decimal_left - decimal_right) <= tolerance * max(
            abs(decimal_left), abs(decimal_right)
        )
    difference = _julia_array_array_operation(
        np.asarray(left), np.asarray(right), np.subtract
    ).reshape(())[()]
    return abs(difference) <= tolerance * max(abs(left), abs(right))


def elq(left: Any, right: Any) -> None:
    """Require two lattices, or two fields, to have approximately equal grids."""

    if isinstance(left, LatticeField) and isinstance(right, LatticeField):
        try:
            _lattice_equal(left.L, right.L)
        except DomainError as error:
            raise DomainError("Unequal lattices or flambdas.") from error
        if not _isapprox_scalar(left.flambda, right.flambda):
            raise DomainError("Unequal lattices or flambdas.")
        return None
    _lattice_equal(as_lattice(left), as_lattice(right))
    return None


def _lattice_equal(left: Lattice, right: Lattice) -> None:
    if len(left) != len(right):
        raise DomainError("Unequal lattices.")
    if tuple(len(item) for item in left) != tuple(len(item) for item in right):
        raise DomainError("Unequal lattices.")
    if not all(_isapprox_array(a, b) for a, b in zip(left, right, strict=True)):
        raise DomainError("Unequal lattices.")


class _TaggedConstructor:
    def __init__(self, tag: type[FieldVal]):
        self.tag = tag

    def __call__(
        self,
        data: Any,
        lattice_or_field: Any,
        flambda: Any = _FLAMBDA_UNSET,
    ) -> "LatticeField":
        if isinstance(lattice_or_field, LatticeField):
            # Julia's constructor-from-field overload has exactly two
            # arguments.  A supplied third argument does not become valid just
            # because it happens to equal the unrelated lattice constructor's
            # default wavelength.
            if flambda is not _FLAMBDA_UNSET:
                raise TypeError(
                    "flambda is inherited when constructing from a LatticeField."
                )
            return LatticeField._from_full(
                data,
                lattice_or_field.L,
                lattice_or_field.flambda,
                self.tag,
            )
        if flambda is _FLAMBDA_UNSET:
            flambda = 1.0
        return LatticeField(
            data,
            lattice_or_field,
            flambda=flambda,
            field_type=self.tag,
        )


class _FullTypedConstructor:
    """Public counterpart of Julia's ``LatticeField{S,T,N}`` constructor."""

    def __init__(self, tag: type[FieldVal], dtype: Any, ndim: Any):
        if not _is_field_tag(tag):
            raise TypeError("The first LatticeField parameter must be a FieldVal tag.")
        if dtype is None:
            # ``nothing`` may appear syntactically as Julia's unconstrained T
            # parameter, but no ``AbstractArray{nothing}`` can satisfy the
            # full constructor. NumPy instead treats ``dtype=None`` as
            # Float64, which would invent a successful typed constructor.
            raise TypeError("The second LatticeField parameter must be a dtype.")
        exact_object_types = (
            Fraction,
            Decimal,
            _MPFR,
            _MPC,
            _MPQ,
            _MPZ,
            _DecimalComplex,
        )
        python_machine_types = (bool, int, float, complex)
        arbitrary_python_type = (
            isinstance(dtype, type)
            and dtype not in python_machine_types
            and not issubclass(dtype, np.generic)
            and dtype is not object
        )
        self.logical_object_type = (
            dtype
            if dtype in exact_object_types or arbitrary_python_type
            else None
        )
        if self.logical_object_type is not None:
            self.dtype = np.dtype(object)
        else:
            try:
                self.dtype = np.dtype(dtype)
            except TypeError as error:
                raise TypeError(
                    "The second LatticeField parameter must be a dtype."
                ) from error
        if not _is_julia_platform_int(ndim):
            raise TypeError(
                "The third LatticeField parameter must be a Julia platform Int."
            )
        self.ndim = int(ndim)
        if self.ndim < 0:
            raise ValueError("LatticeField dimensionality must be nonnegative.")
        self.tag = tag

    def __call__(
        self,
        data: Any,
        lattice: Any,
        flambda: Real = 1.0,
    ) -> "LatticeField":
        if isinstance(data, _CheckedFieldArray):
            array = data
        elif isinstance(data, list):
            array = _julia_field_literal_array(data, lattice)
        elif isinstance(data, np.ndarray):
            array = data
        else:
            raise TypeError(
                "Full typed LatticeField data must be an array or list literal."
            )
        if self.logical_object_type is not None and array.ndim == self.ndim:
            boxed = np.empty(array.shape, dtype=object)
            for index in np.ndindex(array.shape):
                value = array[index]
                boxed[index] = value.item() if isinstance(value, np.generic) else value
            array = boxed
        if array.dtype != self.dtype or array.ndim != self.ndim:
            raise TypeError(
                "Full typed LatticeField constructor requires data with exact "
                f"dtype {self.dtype} and ndim {self.ndim}; got {array.dtype} "
                f"and ndim {array.ndim}."
            )
        if self.logical_object_type is not None and any(
            not _logical_object_type_matches(
                value, self.logical_object_type
            )
            for value in array.flat
        ):
            raise TypeError(
                "Full typed LatticeField constructor requires exact object "
                f"elements of type {self.logical_object_type.__name__}."
            )
        return LatticeField._from_full(
            array,
            lattice,
            flambda,
            self.tag,
            expected_dtype=self.dtype,
            expected_object_type=self.logical_object_type,
        )


class _CheckedFlatIterator:
    """Flat view whose assignments keep Julia array conversion semantics."""

    __slots__ = ("_owner",)

    def __init__(self, owner: "_CheckedFieldArray") -> None:
        self._owner = owner

    def __iter__(self) -> Iterator[Any]:
        return iter(self._owner._storage().flat)

    def __len__(self) -> int:
        return self._owner.size

    def __getitem__(self, key: Any) -> Any:
        return self._owner._storage().flat[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        flat_indices = np.arange(self._owner.size)[key]
        target_indices = np.asarray(flat_indices)
        if target_indices.ndim == 0:
            coordinates = np.unravel_index(
                int(target_indices), self._owner.shape, order="C"
            )
            self._owner[coordinates] = value
            return
        raw_value = _CheckedFieldArray._unwrap_checked(value)
        values = np.broadcast_to(
            np.asarray(raw_value), target_indices.shape
        )
        state = self._owner._state()
        state.begin_write()
        try:
            for index in np.ndindex(target_indices.shape):
                coordinates = np.unravel_index(
                    int(target_indices[index]), self._owner.shape, order="C"
                )
                self._owner[coordinates] = values[index]
        finally:
            state.end_write()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._owner._storage().flat, name)


class _FieldReadOnlyGuard(np.ndarray):
    """Keep public ndarray aliases behind a permanently read-only base.

    NumPy normally collapses a plain-ndarray view chain to its owning base.
    Retaining one subclass in the chain makes ``np.asarray(field.data)`` and
    ``field.data.view(np.ndarray)`` base themselves on the checked public
    view.  Neither those aliases nor the public view can consequently enable
    their WRITEABLE flag.
    """

    @property
    def base(self) -> None:
        """Hide private storage from public ndarray base-chain traversal."""

        return None


def _julia_setindex_shapes_match(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> bool:
    """Translate Base.setindex_shape_check for array right-hand sides."""

    source_size = math.prod(source_shape)
    target_size = math.prod(target_shape)
    source_ndim = len(source_shape)
    target_ndim = len(target_shape)

    if target_ndim == 0:
        return source_size == 1
    if source_ndim == 0:
        return target_size == 1
    if source_ndim == 1 and target_ndim in (1, 2):
        return source_size == target_size
    if source_ndim == 2 and target_ndim == 2:
        return source_size == target_size and (
            target_shape[0] == 1
            or target_shape[0] == source_shape[0]
            or source_shape[0] == 1
        )

    source_index = target_index = 0
    while True:
        source_length = source_shape[source_index]
        target_length = target_shape[target_index]
        if (
            source_index == source_ndim - 1
            or target_index == target_ndim - 1
        ):
            source_length *= math.prod(source_shape[source_index + 1 :])
            target_length *= math.prod(target_shape[target_index + 1 :])
            return source_length == target_length
        if source_length == target_length:
            source_index += 1
            target_index += 1
        elif source_length == 1:
            source_index += 1
        elif target_length == 1:
            target_index += 1
        else:
            return False


def _normalized_integer_indices(
    values: Any,
    size: int,
) -> np.ndarray:
    """Normalize zero-based signed/unsigned integer indices without wrapping."""

    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise TypeError("indices must be integer or Boolean")
    normalized = np.empty(raw.shape, dtype=np.intp)
    for index in np.ndindex(raw.shape):
        value = int(raw[index])
        if value < 0:
            value += size
        if value < 0 or value >= size:
            raise IndexError("index is outside checked field data")
        normalized[index] = value
    return normalized


def _julia_checked_storage_key(
    key: Any,
    shape: tuple[int, ...],
) -> Any:
    """Normalize multi-selector dense-array arity like Julia.

    A lone selector deliberately keeps ordinary NumPy/Python basic-index
    behavior at the public ``.data`` binding boundary.  With two or more
    selectors, Julia permits omitted trailing singleton dimensions and
    redundant trailing scalar-zero selectors (the zero-based spelling of its
    one-valued extra indices).
    """

    if not isinstance(key, tuple) or len(key) <= 1:
        return key
    selectors = list(key)
    if sum(selector is Ellipsis for selector in selectors) > 1:
        return key
    if any(selector is Ellipsis for selector in selectors):
        position = next(
            index
            for index, selector in enumerate(selectors)
            if selector is Ellipsis
        )
        explicit = len(selectors) - 1
        selectors[position : position + 1] = [
            slice(None)
        ] * max(0, len(shape) - explicit)
    if len(selectors) < len(shape):
        trailing = shape[len(selectors) :]
        if not all(size == 1 for size in trailing):
            raise IndexError(
                "omitted trailing checked-array axes must be singleton"
            )
        selectors.extend([0] * len(trailing))
    elif len(selectors) > len(shape):
        extras = selectors[len(shape) :]
        for selector in extras:
            raw = np.asarray(selector)
            if (
                raw.ndim != 0
                or raw.dtype.kind not in "iu"
                or raw.dtype.kind == "b"
                or int(raw) != 0
            ):
                raise IndexError(
                    "extra checked-array indices must select trailing "
                    "singleton coordinates"
                )
        selectors = selectors[: len(shape)]
    return tuple(selectors)


def _julia_linear_index_plan(
    key: Any,
    shape: tuple[int, ...],
) -> np.ndarray | np.intp | None:
    """Return Julia/F-order linear destinations for non-basic indexing.

    One index addresses Julia's linear column-major storage. An integer index
    array retains its own shape; one Boolean index selects true entries in
    column-major mask order. With multiple dimensional selectors, every
    non-scalar selector contributes all of its dimensions and selectors form
    a Cartesian product rather than NumPy's paired advanced index.

    ``None`` means the key is an ordinary full-arity scalar/slice tuple whose
    storage-sharing NumPy view already has Julia's Cartesian topology.
    """

    selectors = list(key if isinstance(key, tuple) else (key,))
    if any(selector is None for selector in selectors):
        return None
    if sum(selector is Ellipsis for selector in selectors) > 1:
        return None

    if len(selectors) == 1 and selectors[0] is not Ellipsis:
        selector = selectors[0]
        total = math.prod(shape)
        if isinstance(selector, slice):
            return None
        raw = np.asarray(selector)
        if raw.dtype.kind == "b":
            if raw.ndim == 0:
                return None
            if not (
                (raw.ndim == 1 and raw.size == total)
                or raw.shape == shape
            ):
                raise IndexError(
                    "Boolean linear index must be a length-total vector or "
                    "have exactly the field shape"
                )
            return np.flatnonzero(raw.ravel(order="F")).astype(
                np.intp, copy=False
            )
        if raw.ndim == 0:
            return None
        return _normalized_integer_indices(raw, total)

    if any(selector is Ellipsis for selector in selectors):
        ellipsis_index = next(
            index
            for index, selector in enumerate(selectors)
            if selector is Ellipsis
        )
        explicit_axes = len(selectors) - 1
        selectors[ellipsis_index : ellipsis_index + 1] = [
            slice(None)
        ] * max(0, len(shape) - explicit_axes)
    if len(selectors) < len(shape):
        trailing_shape = shape[len(selectors) :]
        if not all(size == 1 for size in trailing_shape):
            raise IndexError(
                "omitted trailing checked-array axes must be singleton"
            )
        selectors.extend([0] * (len(shape) - len(selectors)))
    if len(selectors) != len(shape):
        return None

    has_array_selector = any(
        not isinstance(selector, slice)
        and np.asarray(selector).ndim > 0
        for selector in selectors
    )
    if not has_array_selector:
        return None

    dimensional: list[tuple[int, np.ndarray]] = []
    scalar_indices: dict[int, int] = {}
    for axis, selector in enumerate(selectors):
        if isinstance(selector, slice):
            dimensional.append(
                (
                    axis,
                    np.arange(shape[axis], dtype=np.intp)[selector],
                )
            )
            continue
        raw = np.asarray(selector)
        if raw.ndim == 0:
            if raw.dtype.kind == "b":
                raise TypeError("Boolean scalar indices are not supported")
            scalar_indices[axis] = int(
                _normalized_integer_indices(raw, shape[axis]).reshape(())[()]
            )
            continue
        if raw.dtype.kind == "b":
            if raw.ndim != 1 or raw.size != shape[axis]:
                raise IndexError(
                    "Boolean dimensional index must match its axis"
                )
            indices = np.flatnonzero(raw).astype(np.intp, copy=False)
        else:
            indices = _normalized_integer_indices(raw, shape[axis])
        dimensional.append((axis, indices))

    output_shape = tuple(
        extent
        for _, indices in dimensional
        for extent in indices.shape
    )
    strides: list[int] = []
    stride = 1
    for size in shape:
        strides.append(stride)
        stride *= size
    linear = np.zeros(output_shape, dtype=np.intp)
    dimension_offset = 0
    total_dimensions = len(output_shape)
    for axis, indices in dimensional:
        index_dimensions = indices.ndim
        reshape = (
            (1,) * dimension_offset
            + indices.shape
            + (1,) * (
                total_dimensions - dimension_offset - index_dimensions
            )
        )
        linear += indices.reshape(reshape) * strides[axis]
        dimension_offset += index_dimensions
    for axis, index in scalar_indices.items():
        linear += index * strides[axis]
    return linear


class _FieldStorageState:
    """Shared authoritative storage and weakly held public façades."""

    __slots__ = (
        "root",
        "_facades",
        "_write_depth",
        "_dirty",
        "__weakref__",
    )

    def __init__(self, root: np.ndarray) -> None:
        self.root = root
        self._facades: dict[
            int, weakref.ReferenceType[_CheckedFieldArray]
        ] = {}
        self._write_depth = 0
        self._dirty = False

    def prune(self) -> None:
        """Discard dead façade references during ordinary use, not just writes."""

        dead = [
            key
            for key, reference in self._facades.items()
            if reference() is None
        ]
        for key in dead:
            self._facades.pop(key, None)

    def register(self, facade: "_CheckedFieldArray") -> None:
        key = id(facade)
        state_reference = weakref.ref(self)

        def discard(
            reference: weakref.ReferenceType[_CheckedFieldArray],
            *,
            facade_key: int = key,
            owner: weakref.ReferenceType[_FieldStorageState] = state_reference,
        ) -> None:
            state = owner()
            if state is not None and state._facades.get(facade_key) is reference:
                state._facades.pop(facade_key, None)

        self._facades[key] = weakref.ref(facade, discard)

    def read(self) -> None:
        # Reads are deliberately O(1). Weakref callbacks remove dead façades;
        # a write synchronization necessarily visits each live façade once.
        return None

    def synchronize(self) -> None:
        for reference in tuple(self._facades.values()):
            facade = reference()
            if facade is not None:
                facade._refresh_snapshot()
        self._dirty = False

    def begin_write(self) -> None:
        self._write_depth += 1

    def changed(self) -> None:
        self._dirty = True
        if self._write_depth == 0:
            self.synchronize()

    def end_write(self) -> None:
        if self._write_depth <= 0:
            raise RuntimeError("unbalanced checked-storage write batch")
        self._write_depth -= 1
        if self._write_depth == 0 and self._dirty:
            self.synchronize()


class _CheckedFieldArray(np.ndarray):
    """Checked façade over private field storage.

    NumPy 2.5's ``ufunc.at`` writes through an ndarray even when its
    ``WRITEABLE`` flag is false.  A read-only ndarray view therefore cannot be
    a security boundary for Julia-style checked assignment.  Each public
    façade owns a read-only *snapshot* while ``_storage_view`` points at the
    private authoritative field array.  Checked methods update the private
    array; raw aliases made by ``np.asarray`` or ``view(np.ndarray)`` can at
    worst modify their detached snapshot.

    A fresh façade is returned for each ``field.data`` access.  Basic slicing
    and transposition return another checked façade over the corresponding
    private view.  Operations that allocate independent results return normal
    mutable ndarrays instead of half-connected subclass instances.
    """

    __array_priority__ = 1000

    _SNAPSHOT_ATTRIBUTES = frozenset(
        {
            "__array_interface__",
            "__array_struct__",
            "base",
            "ctypes",
            "data",
            "flags",
            "setflags",
        }
    )
    _UNSAFE_MUTATING_METHODS = frozenset(
        {
            "byteswap",
            "itemset",
            "partition",
            "resize",
            "setfield",
            "sort",
        }
    )
    _UNSAFE_ARRAY_FUNCTIONS = frozenset(
        {
            np.copyto,
            np.fill_diagonal,
            np.place,
            np.put,
            np.put_along_axis,
            np.putmask,
        }
    )

    def __new__(
        cls,
        storage: Any,
        state: _FieldStorageState | None = None,
    ) -> "_CheckedFieldArray":
        storage_view = np.asarray(storage)
        if state is None:
            state = _FieldStorageState(storage_view)
        snapshot = np.array(
            storage_view,
            copy=True,
            order="F" if storage_view.flags.f_contiguous else "C",
        )
        # Keep a private writable handle to the detached snapshot so checked
        # mutation can keep an already-returned façade coherent.  No public
        # base chain reaches the authoritative field storage.
        writer = snapshot.view(np.ndarray)
        snapshot.setflags(write=False)
        guard = snapshot.view(_FieldReadOnlyGuard)
        guard.setflags(write=False)
        result = guard.view(cls)
        result.setflags(write=False)
        result._writer_view = writer
        result._storage_view = storage_view
        result._storage_state = state
        state.register(result)
        return result

    def __array_finalize__(self, obj: Any) -> None:
        # ndarray may create temporary subclass objects before a public method
        # can decide whether the result is a live view or an independent
        # allocation.  Such objects must never inherit an unrelated storage
        # writer.  Explicit view-producing methods below construct a new
        # façade with the correct private view.
        raw = np.ndarray.view(self, np.ndarray)
        state = _FieldStorageState(raw)
        self._writer_view = raw
        self._storage_view = raw
        self._storage_state = state
        state.register(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"shape", "strides", "dtype", "real", "imag"}:
            raise AttributeError(
                f"checked field-data metadata {name!r} is immutable; "
                "use checked item assignment"
            )
        np.ndarray.__setattr__(self, name, value)

    def __getattribute__(self, name: str) -> Any:
        """Forward inherited ndarray reads through current private storage.

        Explicit checked methods remain normal class attributes. Every other
        ndarray property/method is evaluated on authoritative storage and its
        result is classified generically as either a checked storage-sharing
        view or an ordinary detached result.
        """

        if (
            name.startswith("_")
            or name in type(self).__dict__
            or name in _CheckedFieldArray._SNAPSHOT_ATTRIBUTES
            or not hasattr(np.ndarray, name)
        ):
            return np.ndarray.__getattribute__(self, name)
        storage = np.ndarray.__getattribute__(self, "_storage")()
        attribute = getattr(storage, name)
        if not callable(attribute):
            return np.ndarray.__getattribute__(self, "_classify_result")(
                attribute
            )
        if name in _CheckedFieldArray._UNSAFE_MUTATING_METHODS:
            def rejected_mutation(*args: Any, **kwargs: Any) -> Any:
                del args, kwargs
                raise TypeError(
                    f"{name} cannot mutate checked field storage; use checked "
                    "item assignment or operate on field.data.copy()"
                )

            return rejected_mutation

        def current_method(*args: Any, **kwargs: Any) -> Any:
            output_position = _ndarray_method_out_position(name)
            positional_output = (
                args[output_position]
                if output_position is not None
                and output_position < len(args)
                else _NUMPY_NO_VALUE
            )
            keyword_output = kwargs.get("out", _NUMPY_NO_VALUE)
            output = (
                keyword_output
                if keyword_output is not _NUMPY_NO_VALUE
                else positional_output
            )
            checked_outputs = np.ndarray.__getattribute__(
                self, "_collect_checked"
            )(output)
            if checked_outputs:
                natural_kwargs = dict(kwargs)
                natural_kwargs.pop("out", None)
                natural_args = list(args)
                if (
                    positional_output is not _NUMPY_NO_VALUE
                    and output_position is not None
                ):
                    natural_args[output_position] = None
                raw_args = np.ndarray.__getattribute__(
                    self, "_unwrap_checked"
                )(tuple(natural_args))
                raw_kwargs = np.ndarray.__getattribute__(
                    self, "_unwrap_checked"
                )(natural_kwargs)
                natural_result = getattr(
                    np.ndarray.__getattribute__(self, "_storage")(), name
                )(*raw_args, **raw_kwargs)
                output_items = output if isinstance(output, tuple) else (output,)
                result_items = (
                    natural_result
                    if isinstance(natural_result, tuple)
                    else (natural_result,)
                )
                if len(output_items) != len(result_items):
                    raise TypeError(
                        f"{name} returned {len(result_items)} outputs for "
                        f"{len(output_items)} destinations"
                    )
                returned: list[Any] = []
                for destination, produced in zip(
                    output_items, result_items, strict=True
                ):
                    if isinstance(destination, _CheckedFieldArray):
                        destination[...] = np.asarray(produced)
                        returned.append(destination)
                    elif destination is None:
                        returned.append(produced)
                    else:
                        np.copyto(destination, produced, casting="unsafe")
                        returned.append(destination)
                return (
                    tuple(returned)
                    if isinstance(output, tuple)
                    else returned[0]
                )
            raw_args = np.ndarray.__getattribute__(
                self, "_unwrap_checked"
            )(args)
            raw_kwargs = np.ndarray.__getattribute__(
                self, "_unwrap_checked"
            )(kwargs)
            result = getattr(
                np.ndarray.__getattribute__(self, "_storage")(), name
            )(*raw_args, **raw_kwargs)
            return np.ndarray.__getattribute__(
                self, "_classify_result"
            )(result)

        return current_method

    def copy(self, order: str = "C") -> np.ndarray:
        """Return an ordinary independent mutable array."""

        return np.array(self._storage(), copy=True, order=order)

    def flatten(self, order: str = "C") -> np.ndarray:
        """Return an ordinary independent mutable flattened array."""

        return np.asarray(self._storage()).flatten(order=order)

    def astype(
        self,
        dtype: Any,
        order: str = "K",
        casting: str = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> np.ndarray:
        """Cast into an ordinary ndarray, never a disconnected façade."""

        detached = np.array(self._storage(), copy=True, order="K")
        return detached.astype(
            dtype,
            order=order,
            casting=casting,
            subok=False,
            copy=copy,
        )

    def take(
        self,
        indices: Any,
        axis: int | None = None,
        out: np.ndarray | None = None,
        mode: str = "raise",
    ) -> np.ndarray:
        """Take values into ordinary mutable storage."""

        if isinstance(out, _CheckedFieldArray):
            result = np.asarray(self._storage()).take(
                indices, axis=axis, mode=mode
            )
            out[...] = result
            return out
        return np.asarray(self._storage()).take(
            indices, axis=axis, out=out, mode=mode
        )

    @staticmethod
    def _return_reduction_out(out: Any, result: Any) -> Any:
        """Commit a reduction result through checked ``out`` conversion."""

        if out is None:
            return result
        if isinstance(out, _CheckedFieldArray):
            produced = np.asarray(result)
            if produced.ndim == 0:
                out[()] = produced.reshape(())[()]
            else:
                out[...] = produced
            return out
        np.copyto(np.asarray(out), np.asarray(result), casting="unsafe")
        return out

    def sum(
        self,
        axis: Any = None,
        dtype: Any = None,
        out: Any = None,
        keepdims: bool = False,
        initial: Any = _NUMPY_NO_VALUE,
        where: Any = _NUMPY_NO_VALUE,
    ) -> Any:
        """Use Julia's reduction order on current authoritative storage."""

        if (
            dtype is None
            and initial is _NUMPY_NO_VALUE
            and where is _NUMPY_NO_VALUE
        ):
            result = _julia_sum(
                self._storage(), axis=axis, keepdims=keepdims
            )
        else:
            kwargs: dict[str, Any] = {
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
            }
            if initial is not _NUMPY_NO_VALUE:
                kwargs["initial"] = initial
            if where is not _NUMPY_NO_VALUE:
                kwargs["where"] = self._unwrap_checked(where)
            with localcontext() as context:
                context.traps[DivisionByZero] = False
                context.traps[InvalidOperation] = False
                context.traps[DecimalOverflow] = False
                result = np.sum(self._storage(), **kwargs)
        return self._return_reduction_out(out, result)

    def cumsum(
        self,
        axis: int | None = None,
        dtype: Any = None,
        out: Any = None,
    ) -> Any:
        """Cumulative Julia addition without Decimal trapping."""

        if dtype is None:
            result = _julia_cumsum(self._storage(), axis=axis)
        else:
            with localcontext() as context:
                context.traps[DivisionByZero] = False
                context.traps[InvalidOperation] = False
                context.traps[DecimalOverflow] = False
                result = np.cumsum(
                    self._storage(), axis=axis, dtype=dtype
                )
        return self._return_reduction_out(out, result)

    def view(self, dtype: Any = None, type: Any = None) -> np.ndarray:
        """Return a detached ordinary NumPy view/copy.

        NumPy's raw-ndarray escape hatch deliberately loses checked mutation.
        Detaching here ensures that loss can never expose field storage.
        """

        detached = np.array(self._storage(), copy=True, order="K")
        if dtype is None and type is None:
            return detached.view()
        if type is None:
            return detached.view(dtype)
        if dtype is None:
            return detached.view(type=type)
        return detached.view(dtype=dtype, type=type)

    @staticmethod
    def _unwrap_checked(value: Any) -> Any:
        if isinstance(value, _CheckedFieldArray):
            return value._storage()
        if isinstance(value, tuple):
            return tuple(
                _CheckedFieldArray._unwrap_checked(item) for item in value
            )
        if isinstance(value, list):
            return [
                _CheckedFieldArray._unwrap_checked(item) for item in value
            ]
        if isinstance(value, dict):
            return {
                key: _CheckedFieldArray._unwrap_checked(item)
                for key, item in value.items()
            }
        return value

    def _classify_result(self, result: Any) -> Any:
        """Turn a storage result into a checked view or ordinary allocation."""

        if isinstance(result, tuple):
            return tuple(self._classify_result(item) for item in result)
        if isinstance(result, list):
            return [self._classify_result(item) for item in result]
        if not isinstance(result, np.ndarray):
            return result
        storage = self._storage()
        if np.shares_memory(result, storage):
            if not result.flags.writeable and self._state().root.flags.writeable:
                try:
                    result.setflags(write=True)
                except ValueError:
                    pass
            return _CheckedFieldArray(result, self._state())
        return np.array(result, copy=True, order="K", subok=False)

    def _refresh_snapshot(self) -> None:
        writer = getattr(self, "_writer_view", None)
        storage = getattr(self, "_storage_view", None)
        if writer is None or storage is None:
            return
        np.copyto(writer, storage, casting="no")

    def _state(self) -> _FieldStorageState:
        state = getattr(self, "_storage_state", None)
        if state is None:
            raise ValueError(
                "This derived array is not connected to field storage."
            )
        return state

    @staticmethod
    def _collect_checked(value: Any) -> tuple["_CheckedFieldArray", ...]:
        if isinstance(value, _CheckedFieldArray):
            return (value,)
        if isinstance(value, (tuple, list)):
            return tuple(
                item
                for child in value
                for item in _CheckedFieldArray._collect_checked(child)
            )
        if isinstance(value, dict):
            return tuple(
                item
                for child in value.values()
                for item in _CheckedFieldArray._collect_checked(child)
            )
        return ()

    @property
    def base(self) -> None:
        """Terminate the public base chain at the checked storage façade."""

        return None

    def _writer(self) -> np.ndarray:
        writer = getattr(self, "_writer_view", None)
        if writer is None:
            raise ValueError(
                "This derived array no longer aliases checked field storage."
            )
        return writer

    def _storage(self) -> np.ndarray:
        storage = getattr(self, "_storage_view", None)
        if storage is None:
            raise ValueError(
                "This derived array no longer aliases checked field storage."
            )
        state = getattr(self, "_storage_state", None)
        if state is not None:
            state.read()
        return storage

    @staticmethod
    def _plain_result(value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(_CheckedFieldArray._plain_result(item) for item in value)
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True, subok=False)
        return value

    def __getitem__(self, key: Any) -> Any:
        storage = self._storage()
        effective_key = _julia_checked_storage_key(key, storage.shape)
        linear_plan = _julia_linear_index_plan(
            effective_key, storage.shape
        )
        if linear_plan is not None:
            if np.asarray(linear_plan).ndim == 0:
                coordinates = np.unravel_index(
                    int(linear_plan), storage.shape, order="F"
                )
                return np.ndarray.__getitem__(storage, coordinates)
            flat_storage = np.asarray(storage).ravel(order="F")
            return np.asarray(
                np.ndarray.__getitem__(flat_storage, linear_plan)
            ).copy(order="K")
        result = np.ndarray.__getitem__(storage, effective_key)
        if isinstance(result, np.ndarray) and result.ndim == 0:
            return np.asarray(result).reshape(())[()]
        if not isinstance(result, np.ndarray):
            return result
        if np.shares_memory(result, storage):
            return _CheckedFieldArray(result, self._state())
        return np.array(result, copy=True, subok=False)

    def __setitem__(self, key: Any, value: Any) -> None:
        storage = self._storage()
        effective_input_key = _julia_checked_storage_key(
            key, storage.shape
        )
        linear_plan = _julia_linear_index_plan(
            effective_input_key, storage.shape
        )
        if linear_plan is not None and np.asarray(linear_plan).ndim == 0:
            coordinates = np.unravel_index(
                int(linear_plan), storage.shape, order="F"
            )
            target = np.ndarray.__getitem__(storage, coordinates)
            effective_key: Any = coordinates
        elif linear_plan is not None:
            target = np.asarray(storage).ravel(order="F")[linear_plan]
            effective_key = None
        else:
            target = np.ndarray.__getitem__(storage, effective_input_key)
            effective_key = effective_input_key
        if not isinstance(target, np.ndarray):
            if isinstance(value, np.ndarray) and value.ndim != 0:
                raise ValueError(
                    f"Inexact assignment to scalar element type {self.dtype}."
                )
            if isinstance(value, np.ndarray) and value.ndim == 0:
                # Julia does not convert a zero-dimensional Array to its
                # scalar element for setindex!.
                raise ValueError(
                    f"Inexact assignment to scalar element type {self.dtype}."
                )
            converted = _julia_assignment_values(
                np.asarray(value), storage
            )
            if converted.ndim != 0:
                raise ValueError(
                    f"Inexact assignment to scalar element type {self.dtype}."
                )
            scalar = converted.reshape(())[()]
            np.ndarray.__setitem__(storage, effective_key, scalar)
            self._state().changed()
            return

        raw_value = self._unwrap_checked(value)
        source = np.asarray(raw_value)
        if source.ndim == 0 and not isinstance(value, np.ndarray):
            raise ValueError(
                "Indexed assignment with one scalar value to multiple "
                "locations is not supported; use fill for scalar assignment."
            )
        if not _julia_setindex_shapes_match(source.shape, target.shape):
            raise DimensionMismatch(
                f"Cannot assign array shape {source.shape} to selected "
                f"shape {target.shape}."
            )

        # Julia consumes the right-hand side and the selected destination in
        # column-major linear order. Its setindex shape check permits only
        # singleton-dimension rearrangements that preserve that order; it
        # never applies NumPy-style singleton broadcasting.
        source_values = np.array(
            source, copy=True, order="F", subok=False
        ).ravel(order="F")
        shares_storage = (
            linear_plan is None
            and bool(target.size)
            and np.shares_memory(target, storage)
        )
        if shares_storage:
            destinations = (
                ("view", np.unravel_index(index, target.shape, order="F"))
                for index in range(target.size)
            )
        else:
            if linear_plan is not None:
                selected = np.asarray(linear_plan).ravel(order="F")
            else:
                linear_map = np.arange(
                    storage.size, dtype=np.intp
                ).reshape(storage.shape, order="C")
                selected = np.asarray(
                    np.ndarray.__getitem__(linear_map, effective_key)
                ).ravel(order="F")
            destinations = (("flat", int(index)) for index in selected)

        state = self._state()
        state.begin_write()
        try:
            for source_index, (kind, destination) in enumerate(destinations):
                converted = _julia_assignment_values(
                    np.asarray(source_values[source_index]), storage
                )
                scalar = converted.reshape(())[()]
                if kind == "view":
                    np.ndarray.__setitem__(target, destination, scalar)
                else:
                    coordinates = np.unravel_index(
                        destination,
                        storage.shape,
                        order=(
                            "F" if linear_plan is not None else "C"
                        ),
                    )
                    np.ndarray.__setitem__(storage, coordinates, scalar)
                state.changed()
        finally:
            state.end_write()

    @property
    def T(self) -> "_CheckedFieldArray":
        return _CheckedFieldArray(self._storage().T, self._state())

    def transpose(self, *axes: Any) -> "_CheckedFieldArray":
        return _CheckedFieldArray(
            self._storage().transpose(*axes), self._state()
        )

    def swapaxes(self, axis1: int, axis2: int) -> "_CheckedFieldArray":
        return _CheckedFieldArray(
            self._storage().swapaxes(axis1, axis2), self._state()
        )

    def squeeze(self, axis: Any = None) -> Any:
        result = self._storage().squeeze(axis=axis)
        if isinstance(result, np.ndarray) and np.shares_memory(
            result, self._storage()
        ):
            return _CheckedFieldArray(result, self._state())
        return self._plain_result(result)

    def reshape(
        self,
        *shape: Any,
        order: str = "C",
        copy: bool | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"order": order}
        if copy is not None:
            kwargs["copy"] = copy
        result = self._storage().reshape(*shape, **kwargs)
        if np.shares_memory(result, self._storage()):
            return _CheckedFieldArray(result, self._state())
        return np.array(result, copy=True, subok=False)

    def ravel(self, order: str = "C") -> Any:
        result = self._storage().ravel(order=order)
        if np.shares_memory(result, self._storage()):
            return _CheckedFieldArray(result, self._state())
        return np.array(result, copy=True, subok=False)

    @property
    def flat(self) -> _CheckedFlatIterator:
        return _CheckedFlatIterator(self)

    @flat.setter
    def flat(self, value: Any) -> None:
        self.flat[:] = value

    def fill(self, value: Any) -> None:
        converted = _julia_assignment_values(value, self._storage())
        if converted.ndim != 0:
            raise ValueError("fill requires a scalar value")
        scalar = converted.reshape(())[()]
        np.ndarray.fill(self._storage(), scalar)
        self._state().changed()

    def put(
        self,
        indices: Any,
        values: Any,
        mode: str = "raise",
    ) -> None:
        raw_indices = np.asarray(indices)
        raw_values = np.broadcast_to(
            np.asarray(self._unwrap_checked(values)), raw_indices.shape
        )
        state = self._state()
        state.begin_write()
        try:
            for index in np.ndindex(raw_indices.shape):
                flat_index = int(raw_indices[index])
                if mode == "wrap":
                    flat_index %= self.size
                elif mode == "clip":
                    flat_index = min(max(flat_index, 0), self.size - 1)
                elif flat_index < 0:
                    flat_index += self.size
                coordinates = np.unravel_index(
                    flat_index, self.shape, order="C"
                )
                self[coordinates] = raw_values[index]
        finally:
            state.end_write()

    def _inplace(self, other: Any, operation: np.ufunc) -> "_CheckedFieldArray":
        values = np.asarray(self._storage())
        raw_other = self._unwrap_checked(other)
        if np.asarray(raw_other).ndim == 0:
            result = _julia_array_scalar_operation(
                values, np.asarray(raw_other).reshape(())[()], operation
            )
        else:
            result = _julia_array_array_operation(
                values, np.asarray(raw_other), operation
            )
        self[...] = result
        return self

    def _ufunc_at(
        self,
        ufunc: np.ufunc,
        indices: Any,
        operands: tuple[Any, ...],
    ) -> None:
        """Perform unbuffered updates through the checked scalar gate."""

        if ufunc.nin not in (1, 2) or len(operands) != ufunc.nin - 1:
            raise TypeError(
                f"checked {ufunc.__name__}.at with {ufunc.nin} inputs is "
                "not supported"
            )
        selected = np.arange(self.size, dtype=np.intp).reshape(self.shape)[
            indices
        ]
        selected_array = np.asarray(selected)
        operand_arrays = tuple(
            np.broadcast_to(
                np.asarray(self._unwrap_checked(operand)),
                selected_array.shape,
            )
            for operand in operands
        )
        positions: Any = (
            ((),)
            if selected_array.ndim == 0
            else np.ndindex(selected_array.shape)
        )
        state = self._state()
        state.begin_write()
        try:
            for position in positions:
                flat_index = int(selected_array[position])
                coordinates = np.unravel_index(
                    flat_index, self.shape, order="C"
                )
                current = self._storage()[coordinates]
                if ufunc.nin == 1:
                    result = np.asarray(ufunc(np.asarray(current)))
                else:
                    result = _julia_array_scalar_operation(
                        np.asarray(current),
                        operand_arrays[0][position],
                        ufunc,
                    )
                self[coordinates] = np.asarray(result).reshape(())[()]
        finally:
            state.end_write()

    def __array_function__(
        self,
        func: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Classify every NumPy function result by storage provenance."""

        del types
        if args and isinstance(args[0], _CheckedFieldArray):
            if func is np.sum:
                return args[0].sum(*args[1:], **kwargs)
            if func is np.cumsum:
                return args[0].cumsum(*args[1:], **kwargs)
        sources = self._collect_checked((args, kwargs))
        if func is np.copyto:
            destination = args[0] if args else kwargs.get("dst")
            if isinstance(destination, _CheckedFieldArray):
                raise TypeError(
                    "copyto cannot mutate checked field storage; use checked "
                    "item assignment or operate on field.data.copy()"
                )
        elif func in self._UNSAFE_ARRAY_FUNCTIONS:
            raise TypeError(
                f"{getattr(func, '__name__', func)!s} cannot mutate checked "
                "field storage; use checked item assignment or operate on "
                "field.data.copy()"
            )
        if (
            func is np.nan_to_num
            and kwargs.get("copy", True) is False
        ):
            raise TypeError(
                "nan_to_num(copy=False) cannot mutate checked field storage"
            )

        output_position = _array_function_out_position(func)
        positional_output = (
            args[output_position]
            if output_position is not None
            and output_position < len(args)
            else _NUMPY_NO_VALUE
        )
        keyword_output = kwargs.get("out", _NUMPY_NO_VALUE)
        output = (
            keyword_output
            if keyword_output is not _NUMPY_NO_VALUE
            else positional_output
        )
        if self._collect_checked(output):
            natural_args = list(args)
            if (
                positional_output is not _NUMPY_NO_VALUE
                and output_position is not None
            ):
                natural_args[output_position] = None
            natural_kwargs = dict(kwargs)
            natural_kwargs.pop("out", None)
            raw_args = self._unwrap_checked(tuple(natural_args))
            raw_kwargs = self._unwrap_checked(natural_kwargs)
            with localcontext() as context:
                context.traps[DivisionByZero] = False
                context.traps[InvalidOperation] = False
                context.traps[DecimalOverflow] = False
                natural_result = func(*raw_args, **raw_kwargs)
            if isinstance(output, tuple):
                produced_items = (
                    natural_result
                    if isinstance(natural_result, tuple)
                    else (natural_result,)
                )
                if len(output) != len(produced_items):
                    raise TypeError(
                        f"{func.__name__} produced {len(produced_items)} "
                        f"outputs for {len(output)} destinations"
                    )
                return tuple(
                    self._return_reduction_out(destination, produced)
                    if destination is not None
                    else produced
                    for destination, produced in zip(
                        output, produced_items, strict=True
                    )
                )
            return self._return_reduction_out(output, natural_result)

        raw_args = self._unwrap_checked(args)
        raw_kwargs = self._unwrap_checked(kwargs)
        result = func(*raw_args, **raw_kwargs)

        def classify(value: Any) -> Any:
            if isinstance(value, tuple):
                return tuple(classify(item) for item in value)
            if isinstance(value, list):
                return [classify(item) for item in value]
            if not isinstance(value, np.ndarray):
                return value
            for source in sources:
                if np.shares_memory(value, source._storage()):
                    return source._classify_result(value)
            return np.array(value, copy=True, order="K", subok=False)

        return classify(result)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Keep every mutating ufunc behind checked field conversion."""

        if method == "at":
            target = inputs[0]
            if isinstance(target, _CheckedFieldArray):
                target._ufunc_at(ufunc, inputs[1], tuple(inputs[2:]))
                return None
            raw_inputs = tuple(
                np.asarray(value._storage())
                if isinstance(value, _CheckedFieldArray)
                else value
                for value in inputs
            )
            return getattr(ufunc, method)(*raw_inputs, **kwargs)

        raw_inputs = tuple(
            np.asarray(value._storage())
            if isinstance(value, _CheckedFieldArray)
            else value
            for value in inputs
        )
        outputs = kwargs.get("out")
        output_items = (
            ()
            if outputs is None
            else outputs if isinstance(outputs, tuple) else (outputs,)
        )
        has_checked_output = any(
            isinstance(value, _CheckedFieldArray)
            for value in output_items
        )
        natural_kwargs = dict(kwargs)
        if has_checked_output:
            # A same-dtype staging output would perform NumPy's unchecked
            # cast before the Julia conversion gate (e.g. 4.5 -> Int64(4)).
            # Compute the natural result first, then commit element by element.
            natural_kwargs.pop("out", None)
        else:
            natural_kwargs = self._unwrap_checked(natural_kwargs)
        with localcontext() as context:
            context.traps[DivisionByZero] = False
            context.traps[InvalidOperation] = False
            context.traps[DecimalOverflow] = False
            result = getattr(ufunc, method)(*raw_inputs, **natural_kwargs)
        if outputs is None:
            return self._plain_result(result)
        if not has_checked_output:
            return result
        staged_result = result if isinstance(result, tuple) else (result,)
        if len(output_items) != len(staged_result):
            raise TypeError(
                f"{ufunc.__name__}.{method} produced "
                f"{len(staged_result)} outputs for {len(output_items)} "
                "destinations"
            )
        returned_outputs: list[Any] = []
        for destination, staged in zip(
            output_items, staged_result, strict=True
        ):
            if isinstance(destination, _CheckedFieldArray):
                produced = np.asarray(staged)
                if destination.ndim == 0:
                    destination[()] = produced.reshape(())[()]
                else:
                    destination[...] = produced
                returned_outputs.append(destination)
                continue
            if destination is None:
                returned_outputs.append(staged)
            else:
                np.copyto(
                    np.asarray(destination),
                    np.asarray(staged),
                    casting="unsafe",
                )
                returned_outputs.append(destination)
        return (
            tuple(returned_outputs)
            if len(returned_outputs) != 1
            else returned_outputs[0]
        )

    def __iadd__(self, other: Any) -> "_CheckedFieldArray":
        return self._inplace(other, np.add)

    def __isub__(self, other: Any) -> "_CheckedFieldArray":
        return self._inplace(other, np.subtract)

    def __imul__(self, other: Any) -> "_CheckedFieldArray":
        return self._inplace(other, np.multiply)

    def __itruediv__(self, other: Any) -> "_CheckedFieldArray":
        return self._inplace(other, np.divide)


class LatticeField:
    """An N-dimensional numerical field and its N coordinate axes."""

    __array_priority__ = 1000
    __slots__ = (
        "_data",
        "_storage_state",
        "L",
        "flambda",
        "field_type",
        "_logical_object_type",
        "_frozen",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False) and name in {
            "data",
            "_data",
            "_storage_state",
            "L",
            "flambda",
            "field_type",
            "_logical_object_type",
        }:
            if (
                name == "data"
                and isinstance(value, _CheckedFieldArray)
                and value._state() is self._storage_state
                and value._storage() is self._data
            ):
                # Python writes the result of ``field.data += value`` back to
                # the attribute after the checked in-place mutation. This is
                # not a metadata change. Require the exact root façade rather
                # than merely shared memory so assigning a slice/view cannot
                # masquerade as augmented-assignment writeback.
                return
            raise AttributeError(f"LatticeField metadata {name!r} is immutable.")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        data: Any,
        lattice: Any,
        flambda: Real = 1.0,
        field_type: type[FieldVal] = Generic,
    ) -> None:
        if not _is_field_tag(field_type):
            raise TypeError("field_type must be a FieldVal subclass.")
        if not _is_real_number(flambda):
            raise TypeError("flambda must be real.")
        if isinstance(data, _CheckedFieldArray):
            array = data
        elif isinstance(data, list):
            array = _julia_field_literal_array(data, lattice)
        elif isinstance(data, np.ndarray):
            array = data
        else:
            raise TypeError(
                "LatticeField data must be an array or list literal."
            )
        axes = as_lattice(lattice)
        if field_type is Intensity:
            # Deliberately allocate, matching Julia's broadcast constructor.
            if isinstance(array, _CheckedFieldArray):
                array = array._storage()
            if array.dtype.kind == "c":
                raise TypeError(
                    "Julia's partial Intensity constructor cannot order complex "
                    "values; use LF[Intensity, dtype, ndim] to invoke the full "
                    "typed constructor explicitly."
                )
            if array.dtype.kind == "O":
                array = np.array(array, dtype=object, copy=True)
                for index in np.ndindex(array.shape):
                    value = array[index]
                    if isinstance(value, Decimal) and value.is_nan():
                        continue
                    if value < 0:
                        array[index] = _julia_typed_zero(value)
            else:
                array = np.where(array < 0, np.zeros((), dtype=array.dtype), array)
        elif field_type is ComplexAmplitude:
            if isinstance(array, _CheckedFieldArray):
                array = array._storage()
            array = np.asarray(array, dtype=np.complex128).copy()
        self._initialize(array, axes, flambda, field_type)

    @classmethod
    def _from_full(
        cls,
        data: Any,
        lattice: Any,
        flambda: Real,
        field_type: type[FieldVal],
        *,
        expected_dtype: np.dtype[Any] | None = None,
        expected_object_type: type[Any] | None = None,
    ) -> "LatticeField":
        if not _is_real_number(flambda):
            raise TypeError("flambda must be real.")
        array = (
            data
            if isinstance(data, _CheckedFieldArray)
            else np.asarray(data)
        )
        if expected_dtype is not None and array.dtype != np.dtype(expected_dtype):
            raise TypeError(
                "Operation changed the element dtype; Julia's typed constructor "
                f"requires {np.dtype(expected_dtype)}, got {array.dtype}."
            )
        if expected_object_type is not None and any(
            not _logical_object_type_matches(value, expected_object_type)
            for value in array.flat
        ):
            raise TypeError(
                "Operation changed the logical object element type; Julia's "
                f"typed constructor requires {expected_object_type.__name__}."
            )
        result = cls.__new__(cls)
        result._initialize(
            array,
            as_lattice(lattice),
            flambda,
            field_type,
            logical_object_type=expected_object_type,
        )
        return result

    def _initialize(
        self,
        data: np.ndarray,
        lattice: Lattice,
        flambda: Real,
        field_type: type[FieldVal],
        *,
        logical_object_type: type[Any] | None = None,
    ) -> None:
        object.__setattr__(self, "_frozen", False)
        # Julia's inner constructor compares ``size(data) !== length.(L)``.
        # The identity comparison makes otherwise matching BigInt, UInt64,
        # Int128, and Rational{BigInt} endpoint ranges fail because their
        # ``length`` result is not the platform ``Int64`` used by ``size``.
        nonplatform_length = any(
            getattr(axis, "_length_kind", "int64") != "int64"
            for axis in lattice
        )
        if (
            nonplatform_length
            or data.ndim != len(lattice)
            or data.shape != tuple(map(len, lattice))
        ):
            raise DimensionMismatch("Field data size does not match lattice size.")
        if isinstance(data, _CheckedFieldArray):
            # Julia's field constructor stores an existing Array by
            # reference. Preserve that aliasing when one field's checked data
            # is passed directly to another field constructor.
            storage = data._storage()
            state = data._state()
        else:
            # Partial constructors for Intensity and ComplexAmplitude have
            # already allocated their broadcast/conversion results above.
            # Every other Julia constructor retains its input array, so keep
            # the NumPy array itself rather than introducing a Python-only
            # defensive copy.
            storage = np.asarray(data)
            state = _FieldStorageState(storage)
        if (
            logical_object_type is None
            and storage.dtype.kind == "O"
            and storage.size
        ):
            element_types = {type(value) for value in storage.flat}
            if len(element_types) == 1:
                logical_object_type = element_types.pop()
        object.__setattr__(self, "_data", storage)
        object.__setattr__(self, "_storage_state", state)
        self.L = lattice
        self.flambda = flambda
        self.field_type = field_type
        object.__setattr__(self, "_logical_object_type", logical_object_type)
        object.__setattr__(self, "_frozen", True)

    @property
    def data(self) -> _CheckedFieldArray:
        """Return a checked façade whose raw NumPy aliases are snapshots."""

        return _CheckedFieldArray(self._data, self._storage_state)

    @classmethod
    def __class_getitem__(cls, parameters: Any) -> Any:
        if isinstance(parameters, tuple):
            if len(parameters) != 3:
                raise TypeError(
                    "Full typed LatticeField spelling is LF[tag, dtype, ndim]."
                )
            return _FullTypedConstructor(*parameters)
        if not _is_field_tag(parameters):
            raise TypeError("LatticeField[...] expects a FieldVal subclass.")
        return _TaggedConstructor(parameters)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    @property
    def S(self) -> type[FieldVal]:
        return self.field_type

    def __len__(self) -> int:
        return self.data.size

    def __repr__(self) -> str:
        return (
            f"LatticeField[{self.field_type.__name__}]("
            f"data={self.data!r}, lattice={self.L!r}, flambda={self.flambda!r})"
        )

    def copy(self) -> "LatticeField":
        # Julia's ``copy`` intentionally invokes ``LF{S}``, not the full typed
        # constructor.  Consequently an unusual full-typed Intensity is clipped
        # and a ComplexAmplitude with non-ComplexF64 storage is promoted.
        return LatticeField[self.field_type](
            self.data.copy(), self.L, self.flambda
        )

    def __copy__(self) -> "LatticeField":
        return self.copy()

    def _linear_index(self, index: int) -> tuple[int, ...]:
        if index < 0 or index >= self.data.size:
            raise IndexError("LatticeField linear index out of bounds")
        return tuple(np.unravel_index(index, self.data.shape, order="F"))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (bool, np.bool_)):
            raise TypeError("Boolean values are not valid Julia array indices.")
        if isinstance(key, (int, np.integer)):
            return self.data[self._linear_index(int(key))]
        if not isinstance(key, tuple):
            key = (key,)
        if any(isinstance(selector, (bool, np.bool_)) for selector in key):
            raise TypeError("Boolean values are not valid Julia array indices.")
        if any(
            isinstance(selector, (list, np.ndarray)) for selector in key
        ):
            raise TypeError(
                "Julia's LatticeField overload has no vector-index method."
            )
        ellipsis_positions = [i for i, item in enumerate(key) if item is Ellipsis]
        if ellipsis_positions:
            if len(ellipsis_positions) != 1:
                raise IndexError("Only one ellipsis is allowed.")
            position = ellipsis_positions[0]
            missing = self.ndim - (len(key) - 1)
            if missing < 0:
                raise DimensionMismatch("Wrong number of indices for LatticeField.")
            key = key[:position] + (slice(None),) * missing + key[position + 1 :]
        integer_key = all(
            isinstance(selector, (int, np.integer)) for selector in key
        )
        range_key = all(isinstance(selector, (slice, range)) for selector in key)

        # A one-integer call is Julia's linear-index overload.  Python passes
        # ``field[0,]`` as a one-item tuple, so it must follow the same path as
        # ``field[0]``.
        if integer_key and len(key) == 1:
            return self.data[self._linear_index(int(key[0]))]

        # Julia's all-range overload always calls ``sublattice`` with the full
        # field dimensionality, so omitted or extra axes are a dimension
        # mismatch even when the underlying dense array has singleton axes.
        if range_key and len(key) != self.ndim:
            raise DimensionMismatch("Wrong number of indices for LatticeField.")

        if len(key) < self.ndim:
            if any(size != 1 for size in self.shape[len(key) :]):
                raise IndexError("LatticeField index out of bounds")
            # Dense Julia arrays permit omitted trailing singleton dimensions.
            # They are scalar dimensions and therefore do not appear in a
            # mixed-selector result lattice.
            key = key + (0,) * (self.ndim - len(key))
        elif len(key) > self.ndim:
            extras = key[self.ndim :]
            if not all(
                isinstance(selector, (int, np.integer)) and int(selector) == 0
                for selector in extras
            ):
                raise IndexError("LatticeField index out of bounds")
            # Julia permits integer index 1 into implicit trailing singleton
            # dimensions; zero is its Python-index counterpart.
            key = key[: self.ndim]

        normalized: list[Any] = []
        retained_axes: list[LatticeAxis] = []
        all_scalar = True
        for axis, selector in zip(self.L, key, strict=True):
            if isinstance(selector, (int, np.integer)):
                index = int(selector)
                if index < 0 or index >= len(axis):
                    raise IndexError("LatticeField index out of bounds")
                normalized.append(index)
                continue
            all_scalar = False
            if isinstance(selector, range):
                selector = slice(
                    selector.start, selector.stop, selector.step
                )
            if not isinstance(selector, slice):
                raise TypeError(
                    "Field selectors must be integers, slices, or ranges; "
                    "Julia's LatticeField overload has no vector-index method."
                )
            normalized_selector = selector
            normalized.append(normalized_selector)
            # Index the range object itself.  Going through ``np.asarray``
            # discards StepRangeLen's high-precision reference/step metadata
            # and cannot be repaired from pathological Float16 materialized
            # coordinates.
            retained_axes.append(axis[normalized_selector])

        result = self.data[tuple(normalized)]
        if all_scalar:
            return result
        result_array = np.array(result, copy=True)
        return LatticeField._from_full(
            result_array,
            tuple(retained_axes),
            self.flambda,
            self.field_type,
            expected_dtype=self.data.dtype,
            expected_object_type=self._logical_object_type,
        )

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, (bool, np.bool_)):
            raise TypeError("Boolean values are not valid Julia array indices.")
        if isinstance(key, (int, np.integer)) and not _is_julia_platform_int(
            key
        ):
            raise TypeError(
                "LatticeField assignment indices must be Julia platform Int."
            )
        if isinstance(value, np.ndarray):
            raise TypeError(
                "LatticeField scalar assignment does not accept array "
                "right-hand sides."
            )
        if _is_julia_platform_int(key):
            target = self._linear_index(int(key))
        else:
            if not isinstance(key, tuple):
                key = (key,)
            if any(isinstance(index, (bool, np.bool_)) for index in key):
                raise TypeError("Boolean values are not valid Julia array indices.")
            if any(
                isinstance(index, (int, np.integer))
                and not _is_julia_platform_int(index)
                for index in key
            ):
                raise TypeError(
                    "LatticeField assignment indices must be Julia platform "
                    "Int."
                )
            if len(key) == 1 and _is_julia_platform_int(key[0]):
                target = self._linear_index(int(key[0]))
            else:
                if len(key) != self.ndim:
                    raise DimensionMismatch(
                        "Wrong number of indices for LatticeField."
                    )
                if not all(_is_julia_platform_int(index) for index in key):
                    raise TypeError(
                        "LatticeField assignment accepts one Julia platform "
                        "Int or one platform Int per dimension."
                    )
                target = tuple(int(index) for index in key)
                if any(index < 0 or index >= size for index, size in zip(
                    target, self.shape, strict=True
                )):
                    raise IndexError("LatticeField index out of bounds")

        converted = _julia_assignment_values(value, self.data)
        if converted.ndim != 0:
            raise ValueError(
                f"Inexact assignment to scalar element type {self.data.dtype}."
            )
        try:
            self.data[target] = converted[()]
        except (TypeError, ValueError, OverflowError) as error:
            # NumPy's shape errors are the counterpart of Julia's failed
            # scalar conversion.  Crucially, conversion was validated before
            # touching the destination.
            raise ValueError(
                f"Inexact assignment to {self.data.dtype}."
            ) from error

    def _scalar_result(self, value: Any) -> "LatticeField":
        return LatticeField._from_full(
            value,
            self.L,
            self.flambda,
            self.field_type,
            expected_dtype=self.data.dtype,
            expected_object_type=self._logical_object_type,
        )

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return _is_real_number(value) or isinstance(
            value, (Complex, np.number, np.bool_)
        )

    def _apply_scalar(
        self,
        other: Any,
        operation: np.ufunc,
        *,
        reflected: bool = False,
    ) -> "LatticeField" | Any:
        if not self._is_scalar(other):
            return NotImplemented
        scalar_dtype = _julia_scalar_dtype(other)
        logical_complex = (
            self.data.dtype.kind == "c"
            or self._logical_object_type in (_MPC, _DecimalComplex)
        )
        scalar_complex = scalar_dtype.kind == "c" or isinstance(
            other, (_MPC, _DecimalComplex)
        )
        if scalar_complex and not logical_complex:
            return NotImplemented
        result = _julia_array_scalar_operation(
            self.data, other, operation, reflected=reflected
        )
        return self._scalar_result(result)

    def __pos__(self) -> "LatticeField":
        # Julia's vararg addition overloads also supply the unary method, but
        # only for these two semantic field kinds. Route through the public
        # tagged constructor just as Julia does, including Intensity clipping
        # and ComplexAmplitude's ComplexF64 conversion.
        if self.field_type not in (Intensity, ComplexAmplitude):
            raise TypeError("Behavior undefined for this combination of inputs.")
        values = np.asarray(self.data)
        if values.dtype == np.dtype(np.bool_):
            positive = values.astype(np.int64)
        elif self.field_type is Intensity and _object_contains_mpfr(values):
            with _bigfloat_context():
                positive = np.asarray(np.positive(values), dtype=object)
        else:
            positive = +values
        return LatticeField[self.field_type](positive, self.L, self.flambda)

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, LatticeField):
            return _multiply_fields(self, other)
        return self._apply_scalar(other, np.multiply)

    def __rmul__(self, other: Any) -> Any:
        return self._apply_scalar(other, np.multiply, reflected=True)

    def __add__(self, other: Any) -> Any:
        if isinstance(other, LatticeField):
            return _add_fields(self, other)
        return self._apply_scalar(other, np.add)

    def __radd__(self, other: Any) -> Any:
        if self._is_scalar(other):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, LatticeField):
            raise TypeError("Behavior undefined for this combination of inputs.")
        return self._apply_scalar(other, np.subtract)

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, LatticeField):
            raise TypeError("Behavior undefined for this combination of inputs.")
        return self._apply_scalar(other, np.divide)

    def __abs__(self) -> "LatticeField":
        if self.field_type is not ComplexAmplitude:
            raise TypeError("Behavior undefined for this combination of inputs.")
        return LatticeField[Modulus](
            _julia_abs(self.data), self.L, self.flambda
        )

    def sqrt(self) -> "LatticeField":
        if self.field_type is not Intensity:
            raise TypeError("Behavior undefined for this combination of inputs.")
        values = np.asarray(self.data)
        if values.dtype.kind == "O" and all(
            isinstance(value, Fraction) for value in values.flat
        ):
            # Julia's sqrt(::Rational) is Float64 rather than Rational.
            rooted = np.asarray(
                [math.sqrt(value) for value in values.flat], dtype=np.float64
            ).reshape(values.shape)
        elif values.dtype.kind in "bui":
            # ``sqrt(::Integer)`` is Float64 throughout Julia Base. NumPy
            # instead selects Float16/Float32 for narrow integer arrays.
            rooted = np.sqrt(values.astype(np.float64))
        elif values.dtype.kind == "O" and (
            _object_contains_mpfr(values)
            or all(
                isinstance(value, (_MPQ, _MPZ))
                or (
                    type(value) is int
                    and not _is_julia_platform_int(value)
                )
                for value in values.flat
            )
        ):
            rooted = np.empty(values.shape, dtype=object)
            with _bigfloat_context():
                for index in np.ndindex(values.shape):
                    item = values[index]
                    if isinstance(item, _MPC):
                        rooted[index] = gmpy2.sqrt(item)
                    elif isinstance(item, _DecimalComplex):
                        rooted[index] = gmpy2.sqrt(
                            _MPC(item.real, item.imag)
                        )
                    else:
                        rooted[index] = gmpy2.sqrt(_to_mpfr(item))
        else:
            rooted = np.sqrt(values)
        return LatticeField[Modulus](rooted, self.L, self.flambda)

    def conj(self) -> "LatticeField":
        if self.field_type is RealPhase:
            values = np.asarray(self.data)
            if values.dtype == np.dtype(np.bool_):
                conjugated = -values.astype(np.int64)
            elif values.dtype.kind == "O" and values.size and all(
                isinstance(value, Fraction) for value in values.flat
            ):
                conjugated = np.empty(values.shape, dtype=object)
                for index in np.ndindex(values.shape):
                    conjugated[index] = _fraction_int64_negate(values[index])
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    if _object_contains_mpfr(values):
                        with _bigfloat_context():
                            conjugated = np.asarray(
                                np.negative(values), dtype=object
                            )
                    else:
                        conjugated = -values
            return LatticeField[RealPhase](
                conjugated, self.L, self.flambda
            )
        if self.field_type is ComplexPhase:
            if _object_contains_mpfr(self.data):
                with _bigfloat_context():
                    conjugated = np.asarray(
                        np.conjugate(self.data), dtype=object
                    )
            else:
                conjugated = np.conjugate(self.data)
            return LatticeField[ComplexPhase](
                conjugated, self.L, self.flambda
            )
        raise TypeError("Conjugation is only defined for phase fields.")

    conjugate = conj

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        if len(inputs) == 2 and ufunc in (
            np.add,
            np.multiply,
            np.subtract,
            np.divide,
            np.true_divide,
        ):
            if inputs[0] is self:
                other = inputs[1]
                if ufunc is np.add:
                    return self.__add__(other)
                if ufunc is np.multiply:
                    return self.__mul__(other)
                if ufunc is np.subtract:
                    return self.__sub__(other)
                return self.__truediv__(other)
            if inputs[1] is self:
                other = inputs[0]
                if ufunc is np.add:
                    return self.__radd__(other)
                if ufunc is np.multiply:
                    return self.__rmul__(other)
                # Julia defines neither scalar - field nor scalar / field.
                return NotImplemented
        if ufunc is np.sqrt and len(inputs) == 1:
            return self.sqrt()
        if ufunc is np.absolute and len(inputs) == 1:
            return abs(self)
        if ufunc is np.conjugate and len(inputs) == 1:
            return self.conj()
        return NotImplemented


LF = LatticeField


def _multiply_fields(left: LatticeField, right: LatticeField) -> LatticeField:
    ls = left.field_type
    rs = right.field_type
    supported = (
        (issubclass(ls, Amplitude) and rs in (ComplexPhase, RealPhase))
        or (ls in (ComplexPhase, RealPhase) and issubclass(rs, Amplitude))
        or (ls is ComplexPhase and rs in (ComplexPhase, RealPhase))
        or (ls is RealPhase and rs is ComplexPhase)
    )
    if not supported:
        raise TypeError("Behavior undefined for this combination of inputs.")
    elq(left, right)
    if issubclass(ls, Amplitude) and rs is ComplexPhase:
        return LatticeField[ComplexAmplitude](
            _julia_array_array_operation(
                left.data, right.data, np.multiply
            ),
            left.L,
            left.flambda,
        )
    if issubclass(ls, Amplitude) and rs is RealPhase:
        return LatticeField[ComplexAmplitude](
            _julia_array_array_operation(
                left.data,
                _real_phase_phasors(right.data),
                np.multiply,
            ),
            left.L,
            left.flambda,
        )
    if ls is ComplexPhase and issubclass(rs, Amplitude):
        return LatticeField[ComplexAmplitude](
            _julia_array_array_operation(
                right.data, left.data, np.multiply
            ),
            left.L,
            left.flambda,
        )
    if ls is RealPhase and issubclass(rs, Amplitude):
        return LatticeField[ComplexAmplitude](
            _julia_array_array_operation(
                right.data,
                _real_phase_phasors(left.data),
                np.multiply,
            ),
            left.L,
            left.flambda,
        )
    if ls is ComplexPhase and rs is ComplexPhase:
        return LatticeField[ComplexPhase](
            _julia_array_array_operation(
                left.data, right.data, np.multiply
            ),
            left.L,
            left.flambda,
        )
    if ls is RealPhase and rs is ComplexPhase:
        return LatticeField[ComplexPhase](
            _julia_array_array_operation(
                _real_phase_phasors(left.data),
                right.data,
                np.multiply,
            ),
            left.L,
            left.flambda,
        )
    if ls is ComplexPhase and rs is RealPhase:
        return LatticeField[ComplexPhase](
            _julia_array_array_operation(
                left.data,
                _real_phase_phasors(right.data),
                np.multiply,
            ),
            left.L,
            left.flambda,
        )
    raise TypeError("Behavior undefined for this combination of inputs.")


def _add_fields(left: LatticeField, right: LatticeField) -> LatticeField:
    if left.field_type is not right.field_type or left.field_type not in (
        RealPhase,
        Intensity,
        ComplexAmplitude,
    ):
        raise TypeError("Behavior undefined for this combination of inputs.")
    elq(left, right)
    if left.field_type is Intensity:
        # Python infix operators are binary and left-associated. Preserve
        # eager value semantics: each visible intermediate is the complete
        # input to the next operation.
        return LatticeField[Intensity](
            _julia_array_array_operation(left.data, right.data, np.add),
            left.L,
            left.flambda,
        )
    return LatticeField[left.field_type](
        _julia_array_array_operation(left.data, right.data, np.add),
        left.L,
        left.flambda,
    )


def sublattice(lattice: Any, *selectors: Any) -> Lattice:
    """Select regular coordinate subsets from every lattice axis.

    Integer selectors retain a one-point axis; slices and ranges retain their
    regular step.  Indices are zero-based Python indices.
    """

    axes = as_lattice(lattice)
    # Python spelling of Julia's one-argument ``CartesianIndices`` overload:
    # ``sublattice(L, (slice(...), range(...)))``.  Keep it distinct from the
    # ordinary vararg spelling and require one range-like selector per axis.
    if (
        len(selectors) == 1
        and isinstance(selectors[0], (tuple, list))
        and len(selectors[0]) == len(axes)
        and all(isinstance(item, (slice, range)) for item in selectors[0])
    ):
        selectors = tuple(selectors[0])
    if len(selectors) != len(axes):
        raise DimensionMismatch(
            "Wrong number of indices while attempting to make sublattice."
        )
    if any(isinstance(selector, (bool, np.bool_)) for selector in selectors):
        raise TypeError("Boolean values do not match Julia Integer selectors.")
    output: list[LatticeAxis] = []
    for axis, selector in zip(axes, selectors, strict=True):
        if isinstance(selector, (int, np.integer)):
            index = int(selector)
            if index < 0 or index >= len(axis):
                raise IndexError("Lattice index out of bounds")
            output.append(axis[index : index + 1])
        else:
            if isinstance(selector, range):
                selector = slice(selector.start, selector.stop, selector.step)
            if not isinstance(selector, slice):
                raise TypeError("Lattice selectors must be integers, slices, or ranges.")
            output.append(axis[selector])
    return tuple(output)


def subfield(field: LatticeField, *selectors: Any) -> Any:
    """Apply Julia's vararg ``subfield`` dispatch with zero-based indices.

    One integer is a Fortran-order linear index, just as one Julia integer is a
    linear array index.  A complete Cartesian integer tuple returns a scalar;
    ranges/slices retain the corresponding axes.  Unlike ordinary Python
    indexing, omitted dimensions are not padded implicitly.
    """

    if not selectors:
        # Julia has two ambiguous empty-vararg methods here.
        raise TypeError("subfield requires at least one index or range.")
    if any(isinstance(selector, (bool, np.bool_)) for selector in selectors):
        raise TypeError("Boolean values are not valid Julia array indices.")
    if not all(
        isinstance(selector, (int, np.integer, slice, range))
        for selector in selectors
    ):
        raise TypeError("subfield selectors must be integers, slices, or ranges.")
    integer_selectors = all(
        isinstance(selector, (int, np.integer)) for selector in selectors
    )
    if integer_selectors:
        indices = tuple(int(selector) for selector in selectors)
        if any(index < 0 for index in indices):
            raise IndexError("LatticeField index out of bounds")
        if len(indices) == 1:
            return field[indices[0]]

        if len(indices) < field.ndim:
            if any(size != 1 for size in field.shape[len(indices) :]):
                raise IndexError("LatticeField index out of bounds")
            indices = indices + (0,) * (field.ndim - len(indices))
        elif len(indices) > field.ndim:
            if any(index != 0 for index in indices[field.ndim :]):
                raise IndexError("LatticeField index out of bounds")
            indices = indices[: field.ndim]
        if any(index >= size for index, size in zip(indices, field.shape, strict=True)):
            raise IndexError("LatticeField index out of bounds")
        return field.data[indices]

    range_selectors = all(
        isinstance(selector, (slice, range)) for selector in selectors
    )
    if range_selectors:
        if len(selectors) != field.ndim:
            raise DimensionMismatch(
                "Wrong number of indices while attempting to make sublattice."
            )
        return field[selectors]

    # Julia permits omitted trailing singleton dimensions, and extra integer
    # indices into implicit singleton dimensions.  Its mixed overload drops
    # both classes of scalar dimension from the returned lattice.
    normalized = selectors
    if len(normalized) < field.ndim:
        if any(size != 1 for size in field.shape[len(normalized) :]):
            raise IndexError("LatticeField index out of bounds")
        normalized = normalized + (0,) * (field.ndim - len(normalized))
    elif len(normalized) > field.ndim:
        extras = normalized[field.ndim :]
        if not all(
            isinstance(selector, (int, np.integer)) and int(selector) == 0
            for selector in extras
        ):
            raise IndexError("LatticeField index out of bounds")
        normalized = normalized[: field.ndim]
    return field[normalized]


def square(field: LatticeField) -> LatticeField:
    """Return ``abs(field.data)**2`` tagged as :class:`Intensity`."""

    if not issubclass(field.field_type, Amplitude):
        raise TypeError("Behavior undefined for this combination of inputs.")
    values = np.asarray(field.data)
    if values.dtype.kind == "O" and values.size and all(
        isinstance(value, Fraction) for value in values.flat
    ):
        magnitude = np.empty(values.shape, dtype=object)
        for index in np.ndindex(values.shape):
            rational = _fraction_int64(values[index])
            if rational.numerator == _INT64_MIN:
                raise OverflowError(
                    "Rational{Int64} numerator is typemin(Int64)."
                )
            magnitude[index] = abs(rational)
    else:
        magnitude = _julia_abs(values)
    return LatticeField[Intensity](
        _julia_array_array_operation(
            magnitude, magnitude, np.multiply
        ),
        field.L,
        field.flambda,
    )


def _real_phase_phasors(data: Any) -> np.ndarray:
    """Evaluate phase exponentials in Julia's Rational/BigFloat domains."""

    values = np.asarray(data)
    if values.dtype.kind == "O" and all(
        isinstance(value, Fraction) for value in values.flat
    ):
        # ``2pi`` is Float64 in Julia, so multiplying it by Rational phase
        # samples promotes those samples to Float64 before ``exp``.
        values = np.asarray(values, dtype=np.float64)
    elif values.dtype.kind == "O" and all(
        isinstance(value, (Decimal, _MPFR, _MPQ, _MPZ))
        or (type(value) is int and not _is_julia_platform_int(value))
        for value in values.flat
    ):
        output = np.empty(values.shape, dtype=object)
        # Julia's ``2pi`` token is Float64. Multiplication promotes that exact
        # binary64 value to BigFloat before applying the high-precision exp.
        with _bigfloat_context():
            factor = _to_mpfr(float(2 * np.pi))
            for index in np.ndindex(values.shape):
                phase = _to_mpfr(values[index])
                if not gmpy2.is_finite(phase):
                    nan = _MPFR("nan")
                    output[index] = _DecimalComplex(nan, nan)
                else:
                    sine, cosine = gmpy2.sin_cos(factor * phase)
                    output[index] = _DecimalComplex(cosine, sine)
        return output
    # ``2pi * im`` is ComplexF64 in Julia and therefore widens every machine
    # real/complex phase array before exponentiation.
    # NumPy 1.x treats a NumPy scalar as weak when combined with an array and
    # can keep Float32 input at Complex64.  Julia's ComplexF64 factor widens
    # every machine input before multiplication, independent of NumPy's
    # version-specific scalar-promotion rules.
    complex_values = np.asarray(values, dtype=np.complex128)
    return np.exp(np.complex128(2j * np.pi) * complex_values)


def _decimal_pi() -> Decimal:
    """Compute pi to the active Decimal precision with guard digits."""

    with localcontext() as context:
        requested = context.prec
        context.prec = requested + 12
        terms = context.prec // 14 + 2
        multiplier = 1
        linear = 13_591_409
        exponential = 1
        step = 6
        series = Decimal(linear)
        for iteration in range(1, terms):
            multiplier = (
                multiplier * (step**3 - 16 * step) // (iteration**3)
            )
            linear += 545_140_134
            exponential *= -262_537_412_640_768_000
            series += Decimal(multiplier * linear) / Decimal(exponential)
            step += 12
        value = Decimal(426_880) * Decimal(10_005).sqrt() / series
        context.prec = requested
        return +value


def _decimal_sincos(angle: Decimal) -> tuple[Decimal, Decimal]:
    """Return high-precision sine/cosine for a finite Decimal angle."""

    if not angle.is_finite():
        raise ValueError("phase must be finite")
    with localcontext() as context:
        requested = context.prec
        # Near quarter turns, cosine is the cancellation of two values near
        # one. Keep enough guard digits to retain the active precision in that
        # small residual, as MPFR/BigFloat does.
        context.prec = requested + 50
        pi = _decimal_pi()
        two_pi = pi * 2
        reduced = angle % two_pi
        if reduced > pi:
            reduced -= two_pi
        elif reduced < -pi:
            reduced += two_pi

        squared = reduced * reduced
        sine = reduced
        sine_term = reduced
        cosine = Decimal(1)
        cosine_term = Decimal(1)
        threshold = Decimal(1).scaleb(-(context.prec + 2))
        index = 1
        while True:
            sine_term *= -squared / Decimal((2 * index) * (2 * index + 1))
            cosine_term *= -squared / Decimal((2 * index - 1) * (2 * index))
            sine += sine_term
            cosine += cosine_term
            if abs(sine_term) <= threshold and abs(cosine_term) <= threshold:
                break
            index += 1
        context.prec = requested
        return +sine, +cosine


def _decimal_atan(value: Decimal) -> Decimal:
    """Evaluate atan at the active Decimal precision."""

    requested = getcontext().prec
    with localcontext() as context:
        context.prec = requested + 12
        x = +value
        negative = x < 0
        if negative:
            x = -x
        pi = _decimal_pi()
        if x > 1:
            result = pi / 2 - _decimal_atan(1 / x)
        else:
            threshold = Decimal(2).sqrt() - 1
            offset = Decimal(0)
            if x > threshold:
                offset = pi / 4
                x = (x - 1) / (x + 1)
            power = x
            result = x
            denominator = 3
            tolerance = Decimal(10) ** -(requested + 7)
            while True:
                power *= -(x * x)
                addend = power / denominator
                updated = result + addend
                if updated == result or abs(addend) < tolerance:
                    result = updated
                    break
                result = updated
                denominator += 2
            result += offset
        if negative:
            result = -result
        context.prec = requested
        return +result


def _decimal_atan2(y: Decimal, x: Decimal) -> Decimal:
    pi = _decimal_pi()
    if x > 0:
        return _decimal_atan(y / x)
    if x < 0:
        angle = _decimal_atan(y / x)
        positive_side = y > 0 or (y == 0 and not y.is_signed())
        return angle + pi if positive_side else angle - pi
    if y > 0:
        return pi / 2
    if y < 0:
        return -pi / 2
    return Decimal("-0") if y.is_signed() else Decimal(0)


def _decimal_complex_power(
    value: _DecimalComplex, exponent: Decimal
) -> _DecimalComplex:
    modulus_squared = value.real * value.real + value.imag * value.imag
    if modulus_squared == 0:
        if exponent > 0:
            return _DecimalComplex(Decimal(0), Decimal(0))
        raise ZeroDivisionError("zero complex value cannot have this exponent")
    log_radius = modulus_squared.ln() / 2
    angle = _decimal_atan2(value.imag, value.real)
    magnitude = (exponent * log_radius).exp()
    sine, cosine = _decimal_sincos(exponent * angle)
    return _DecimalComplex(magnitude * cosine, magnitude * sine)


def wrap(field: LatticeField) -> LatticeField:
    """Convert real phase cycles to phasors, or rewrap a complex phase."""

    if field.field_type is RealPhase:
        return LatticeField[ComplexPhase](
            _real_phase_phasors(field.data), field.L, field.flambda
        )
    if field.field_type is ComplexPhase:
        # Preserve Julia's aliasing behavior rather than the misleading docstring.
        return LatticeField[ComplexPhase](field.data, field.L, field.flambda)
    raise TypeError("wrap is only defined for phase fields.")


def normalizeLF(field: LatticeField) -> LatticeField:
    """Normalize intensity by its sum or amplitude by its discrete L2 norm."""

    values = np.asarray(field.data)
    if field.field_type is Intensity:
        total = _julia_sum(values)
        quotient = _julia_array_scalar_operation(
            values, total, np.divide
        )
        if (
            quotient.dtype.kind == "O"
            and quotient.size > 0
            and all(isinstance(value, Fraction) for value in quotient.flat)
        ):
            result = np.empty(quotient.shape, dtype=object)
            for index in np.ndindex(quotient.shape):
                rational = _fraction_int64(quotient[index])
                result[index] = (
                    _fraction_int64_negate(rational)
                    if rational.numerator < 0
                    else rational
                )
        else:
            result = _julia_abs(quotient)
    elif issubclass(field.field_type, Amplitude):
        if _object_contains_mpfr(values):
            with _bigfloat_context():
                magnitude = _julia_abs(values)
                squared = _julia_array_array_operation(
                    magnitude, magnitude, np.multiply
                )
                squared_sum = _julia_sum(squared)
                norm = _mpfr_sqrt(squared_sum)
                result = _julia_array_scalar_operation(
                    values, norm, np.divide
                )
        else:
            magnitude = _julia_abs(values)
            squared = _julia_array_array_operation(
                magnitude, magnitude, np.multiply
            )
            squared_sum = _julia_sum(squared)
        if _object_contains_mpfr(values):
            pass
        elif isinstance(squared_sum, Decimal):
            with localcontext() as context:
                context.traps[DivisionByZero] = False
                context.traps[InvalidOperation] = False
                context.traps[DecimalOverflow] = False
                norm = squared_sum.sqrt()
            result = _julia_array_scalar_operation(
                values, norm, np.divide
            )
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                norm = np.sqrt(squared_sum)
            result = _julia_array_scalar_operation(
                values, norm, np.divide
            )
    else:
        raise TypeError(
            "normalizeLF is only defined for intensity/amplitude fields."
        )
    return LatticeField._from_full(
        result,
        field.L,
        field.flambda,
        field.field_type,
        expected_dtype=field.data.dtype,
        expected_object_type=field._logical_object_type,
    )


def phasor(value: Any) -> Any:
    """Implement the two Julia methods: ComplexF64 scalar or ComplexAmp field."""

    if isinstance(value, LatticeField):
        if value.field_type is not ComplexAmplitude:
            raise TypeError("phasor(field) requires ComplexAmplitude.")
        if value.data.dtype != np.dtype(np.complex128):
            raise TypeError(
                "phasor(ComplexAmplitude) requires ComplexF64 field data."
            )
        # ``field.data`` is a checked facade whose Boolean indexing follows
        # Julia's indexing contract.  NumPy's mask assignment expects its own
        # row-major rules, so operate on one authoritative plain-array copy.
        data = value.data.copy()
        vectorized = np.ones(value.shape, dtype=np.complex128)
        np.divide(
            data,
            np.abs(data),
            out=vectorized,
            where=data != 0,
        )
        return LatticeField[ComplexPhase](vectorized, value.L, value.flambda)
    if type(value) is complex:
        scalar = value
    elif isinstance(value, np.complex128) and np.ndim(value) == 0:
        scalar = complex(value)
    else:
        raise TypeError("phasor scalar input must be ComplexF64.")
    return np.complex128(1.0 + 0.0j if scalar == 0 else scalar / abs(scalar))


def iter_linear(field: LatticeField) -> Iterator[Any]:
    """Iterate in Julia's column-major linear order."""

    return iter(field.data.ravel(order="F"))


__all__ = [
    "Amplitude",
    "ComplexAmp",
    "ComplexAmplitude",
    "ComplexPhase",
    "DimensionMismatch",
    "DomainError",
    "FieldVal",
    "Generic",
    "Intensity",
    "LF",
    "Lattice",
    "LatticeAxis",
    "LatticeField",
    "Modulus",
    "Phase",
    "RealAmp",
    "RealAmplitude",
    "RealPhase",
    "S1Phase",
    "UPhase",
    "UnwrappedPhase",
    "as_lattice",
    "elq",
    "iter_linear",
    "normalizeLF",
    "phasor",
    "square",
    "subfield",
    "sublattice",
    "wrap",
]
