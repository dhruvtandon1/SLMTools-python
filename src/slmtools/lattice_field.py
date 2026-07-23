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

import math
from dataclasses import dataclass
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
from numbers import Complex, Integral, Real
from typing import Any, Iterator, TypeAlias

import numpy as np


class DimensionMismatch(ValueError):
    """Python counterpart of Julia's ``DimensionMismatch``."""


class DomainError(ValueError):
    """Python counterpart of Julia's ``DomainError``."""


class FieldVal:
    """Base class for semantic field-value tags."""


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


@dataclass(frozen=True, slots=True)
class _DecimalComplex:
    """Small object-number counterpart of Julia ``Complex{BigFloat}``."""

    real: Decimal
    imag: Decimal = Decimal(0)

    @staticmethod
    def _coerce(value: Any) -> "_DecimalComplex" | Any:
        if isinstance(value, _DecimalComplex):
            return value
        if isinstance(value, Decimal):
            return _DecimalComplex(value)
        if isinstance(value, Fraction):
            return _DecimalComplex(
                Decimal(value.numerator) / Decimal(value.denominator)
            )
        if isinstance(value, (bool, int, np.integer)):
            return _DecimalComplex(Decimal(int(value)))
        if isinstance(value, (float, np.floating)):
            return _DecimalComplex(Decimal.from_float(float(value)))
        if isinstance(value, (complex, np.complexfloating)):
            scalar = complex(value)
            return _DecimalComplex(
                Decimal.from_float(scalar.real),
                Decimal.from_float(scalar.imag),
            )
        return NotImplemented

    def __complex__(self) -> complex:
        return complex(float(self.real), float(self.imag))

    def __abs__(self) -> Decimal:
        return (self.real * self.real + self.imag * self.imag).sqrt()

    def conjugate(self) -> "_DecimalComplex":
        return _DecimalComplex(self.real, -self.imag)

    def __neg__(self) -> "_DecimalComplex":
        return _DecimalComplex(-self.real, -self.imag)

    def __add__(self, other: Any) -> "_DecimalComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        return _DecimalComplex(self.real + converted.real, self.imag + converted.imag)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "_DecimalComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        return _DecimalComplex(self.real - converted.real, self.imag - converted.imag)

    def __rsub__(self, other: Any) -> "_DecimalComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        return converted - self

    def __mul__(self, other: Any) -> "_DecimalComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        return _DecimalComplex(
            self.real * converted.real - self.imag * converted.imag,
            self.real * converted.imag + self.imag * converted.real,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "_DecimalComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        denominator = converted.real**2 + converted.imag**2
        return _DecimalComplex(
            (self.real * converted.real + self.imag * converted.imag) / denominator,
            (self.imag * converted.real - self.real * converted.imag) / denominator,
        )

    def __rtruediv__(self, other: Any) -> "_DecimalComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        return converted / self

    def __pow__(self, exponent: Any) -> "_DecimalComplex" | Any:
        if isinstance(exponent, Decimal):
            return _decimal_complex_power(self, exponent)
        return NotImplemented


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

    # NumPy arrays are materialized while Julia ranges are lazy.  Reject an
    # allocation that cannot be represented (or would be unreasonably large
    # for this eager public type) before entering the iteration loop.
    if count > min(np.iinfo(np.intp).max, 10_000_000):
        raise MemoryError("LatticeAxis is too large to materialize.")

    # A LatticeAxis is necessarily materialized.  Iterate the same wrapped
    # machine arithmetic as StepRange and stop on Base's normalized last
    # element.  The cap turns an otherwise impossible allocation into the
    # same practical failure boundary as any other materialized Python axis.
    output: list[Any] = []
    current = start_value
    seen: set[int] = set()
    while True:
        output.append(current)
        if current == last:
            break
        marker = int(current)
        if marker in seen:
            raise OverflowError("StepRange iteration did not reach its endpoint.")
        seen.add(marker)
        current = _julia_integer_binary(current, step, np.add)
        current = np.asarray(current, dtype=dtype).reshape(())[()]
    return np.asarray(output, dtype=dtype)


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
    ) -> "LatticeAxis":
        array = np.asarray(values)
        if array.ndim != 1:
            raise DimensionMismatch("A lattice axis must be one-dimensional.")
        result = np.array(array, copy=True).view(cls)
        step_is_logical = step_hint is not None
        if step_hint is None and len(result) >= 2:
            step_hint = result[1] - result[0]
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

        if isinstance(length, (bool, np.bool_)) or not isinstance(
            length, (int, np.integer)
        ):
            raise TypeError("length must be a signed machine integer.")
        length_value = int(length)
        if length_value < 0:
            raise ValueError("length must be nonnegative.")
        if not np.iinfo(np.int64).min <= length_value <= np.iinfo(np.int64).max:
            raise OverflowError("length does not fit Julia Int64.")

        start_dtype = _julia_scalar_dtype(start)
        step_dtype = _julia_scalar_dtype(step)
        if start_dtype.kind in "biu" and step_dtype.kind in "biu":
            start_value = np.asarray(start, dtype=start_dtype)[()]
            step_value = np.asarray(step, dtype=step_dtype)[()]
            stop_delta = _julia_array_scalar_operation(
                np.asarray(step_value),
                np.int64(length_value - 1),
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
        )

    def __array_finalize__(self, obj: Any) -> None:
        self._step_hint = getattr(obj, "_step_hint", None)
        self._step_hint_is_logical = getattr(obj, "_step_hint_is_logical", False)
        self._logical_ref = getattr(obj, "_logical_ref", None)
        self._logical_step = getattr(obj, "_logical_step", None)
        self._logical_offset = getattr(obj, "_logical_offset", None)
        self._range_kind = getattr(obj, "_range_kind", None)
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
                if kind == "ordinal":
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
                    _range_kind=kind,
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


def _axis(
    values: Any,
    step_hint: Any | None = None,
    *,
    _logical_ref: Any | None = None,
    _logical_step: Any | None = None,
    _logical_offset: int | None = None,
    _range_kind: str | None = None,
) -> LatticeAxis:
    if isinstance(values, LatticeAxis):
        if (
            not values.flags.writeable
            and (step_hint is None or values._step_hint == step_hint)
            and _logical_ref is None
            and _logical_step is None
            and _logical_offset is None
            and _range_kind is None
        ):
            return values
        return LatticeAxis(
            np.asarray(values),
            step_hint=step_hint,
            _logical_ref=_logical_ref,
            _logical_step=_logical_step,
            _logical_offset=_logical_offset,
            _range_kind=_range_kind,
        )
    if isinstance(values, range):
        return LatticeAxis.from_start_step(
            np.int64(values.start),
            np.int64(values.step),
            len(values),
        )
    return LatticeAxis(
        values,
        step_hint=step_hint,
        _logical_ref=_logical_ref,
        _logical_step=_logical_step,
        _logical_offset=_logical_offset,
        _range_kind=_range_kind,
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
                regular = all(
                    actual == expected
                    for actual, expected in zip(
                        values.tolist(), expected_values, strict=True
                    )
                )
            elif values.dtype.kind in "buif":
                expected = np.asarray(expected_values, dtype=values.dtype)
                if values.dtype.kind == "f":
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
                    matching_nonfinite = (
                        (np.isnan(values) & np.isnan(expected))
                        | (np.isposinf(values) & np.isposinf(expected))
                        | (np.isneginf(values) & np.isneginf(expected))
                    )
                    matching_finite = (
                        np.isfinite(values)
                        & np.isfinite(expected)
                        & (np.abs(values - expected) <= 8 * epsilon * scale)
                    )
                    regular = np.all(matching_nonfinite | matching_finite)
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
        if not all(_is_real_number(value) for value in coordinates):
            raise TypeError("Lattice axes must contain real numeric coordinates.")
        differences = [
            right - left
            for left, right in zip(coordinates[:-1], coordinates[1:], strict=True)
        ]
        candidate = differences[0]
        if not all(difference == candidate for difference in differences):
            raise ValueError("Lattice axes must be regularly spaced.")
        return
    if values.dtype.kind not in "buifc":
        raise TypeError("Lattice axes must contain numeric coordinates.")
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
            raise OverflowError("Python integer does not fit Julia Int64.")
        return np.dtype(np.int64)
    if type(value) is float:
        return np.dtype(np.float64)
    if type(value) is complex:
        return np.dtype(np.complex128)
    if isinstance(value, (Fraction, Decimal)):
        return np.dtype(object)
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "buifc":
        raise TypeError("Scalar arithmetic requires a numeric scalar.")
    return array.dtype


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
    if any(isinstance(item, Decimal) for item in items):
        if any(isinstance(item, (complex, np.complexfloating)) for item in items):
            raise TypeError(
                "Complex{BigFloat}-like literal arrays are not supported."
            )
        return _as_decimal_array(source)

    has_fraction = any(isinstance(item, Fraction) for item in items)
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


def _julia_add_sum(values: Any) -> Any:
    """Reduce values with Julia's ``Base.add_sum`` accumulator widening."""

    terms = tuple(values)
    if not terms:
        return np.int64(0)

    def widened(value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype.kind == "b":
            return array.astype(np.int64)
        if array.dtype.kind == "i" and array.dtype.itemsize < 8:
            return array.astype(np.int64)
        if array.dtype.kind == "u" and array.dtype.itemsize < 8:
            return array.astype(np.uint64)
        return array

    result = widened(terms[0])
    for term in terms[1:]:
        result = _julia_array_array_operation(
            result, widened(term), np.add
        )
    if result.ndim == 0:
        return result.reshape(())[()]
    return result


def _julia_array_scalar_operation(
    array: Any,
    scalar: Any,
    operation: np.ufunc,
    *,
    reflected: bool = False,
) -> np.ndarray:
    """Apply an array/scalar operation with Julia rather than NumPy promotion."""

    values = np.asarray(array)
    scalar_array = np.asarray(scalar)
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
        if _object_contains_decimal(values) or _object_contains_decimal(
            scalar_array
        ):
            converted_values = _as_decimal_array(values)
            converted_scalar = _as_decimal_approx(scalar_array.item())
        else:
            converted_values = values.astype(object, copy=False)
            converted_scalar = scalar_array.item()
        if reflected:
            return np.asarray(operation(converted_scalar, converted_values))
        return np.asarray(operation(converted_values, converted_scalar))
    scalar_dtype = _julia_scalar_dtype(scalar)
    result_dtype = _julia_promote_numeric_dtypes(
        values.dtype,
        scalar_dtype,
        division=operation is np.divide,
        operation=operation,
    )
    converted_values = values.astype(result_dtype, copy=False)
    converted_scalar = np.asarray(scalar, dtype=result_dtype)[()]
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
    scalar_array = np.asarray(scalar)
    if (
        reference is None
        or logical_step is None
        or offset is None
        or range_kind is None
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
            new_step = -old_step if reflected and operation is np.subtract else old_step
        elif operation is np.multiply:
            new_step = _julia_array_scalar_operation(
                np.asarray(old_step), scalar, np.multiply
            ).reshape(())[()]
        elif reflected:
            new_step = None
        else:
            new_step = _julia_array_scalar_operation(
                np.asarray(old_step), scalar, np.divide
            ).reshape(())[()]
        return _axis(values, step_hint=new_step)

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
        return _axis(values)

    if range_kind == "ordinal":
        is_integer_scalar = scalar_array.dtype.kind in "bui"
        unit_step = logical_step == 1
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
            return LatticeAxis.from_start_step(
                new_reference,
                new_step,
                len(canonical),
            )
        elif (
            operation in (np.add, np.subtract)
            and unit_step
            and not (operation is np.subtract and reflected)
        ):
            # AbstractUnitRange +/- Real calls ``range(first +/- x,
            # length=...)``.  That constructor stores an ordinary
            # StepRangeLen reference/step rather than a TwicePrecision pair.
            # For Float16/Float32, both stored values remain in that low type;
            # widening them here changes every later range index operation.
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
                )
            return generated
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
            new_step = _tp_multiply(logical_step, converted_scalar)
        else:
            new_reference = _tp_divide(reference, converted_scalar)
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
    )


def _decimal_rtol() -> Decimal:
    """Context-sized counterpart of Julia's ``sqrt(eps(BigFloat))``."""

    # A p-digit Decimal context represents approximately ceil(p/log10(2))
    # bits. Julia's default isapprox tolerance is sqrt(2^(1-precision)).
    bits = math.ceil(getcontext().prec / math.log10(2))
    return (Decimal(2) ** (1 - bits)).sqrt()


def _object_contains_decimal(value: Any) -> bool:
    array = np.asarray(value, dtype=object)
    return any(isinstance(item, Decimal) for item in array.flat)


def _object_contains_decimal_complex(value: Any) -> bool:
    array = np.asarray(value, dtype=object)
    return any(isinstance(item, _DecimalComplex) for item in array.flat)


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


def _julia_array_array_operation(
    left: Any, right: Any, operation: np.ufunc
) -> np.ndarray:
    """Apply an array operation with Julia numeric promotion.

    The machine-dtype branch explicitly casts both operands to Julia's
    promoted type before evaluating the ufunc.  Falling through to NumPy here
    is incorrect: for example, NumPy promotes ``UInt64 + Int64`` to Float64
    while Julia performs wrapping UInt64 arithmetic.
    """

    first = np.asarray(left)
    second = np.asarray(right)
    if _object_contains_decimal_complex(first) or _object_contains_decimal_complex(
        second
    ):
        first, second = np.broadcast_arrays(
            first.astype(object, copy=False),
            second.astype(object, copy=False),
        )
        return np.asarray(operation(first, second))
    if _object_contains_decimal(first) or _object_contains_decimal(second):
        first, second = np.broadcast_arrays(first, second)
        return np.asarray(
            operation(_as_decimal_array(first), _as_decimal_array(second))
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

    if isinstance(value, Decimal):
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
    if element_type in (Fraction, Decimal, _DecimalComplex):
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
        difference = float(np.linalg.norm(difference_values))
        if np.isfinite(difference):
            if tolerance == 0:
                return difference <= 0
            scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
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
        try:
            self.dtype = np.dtype(dtype)
        except TypeError as error:
            raise TypeError("The second LatticeField parameter must be a dtype.") from error
        if not isinstance(ndim, (int, np.integer)):
            raise TypeError("The third LatticeField parameter must be an integer.")
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
        array = (
            data
            if isinstance(data, _CheckedFieldArray)
            else np.asarray(data)
        )
        if array.dtype != self.dtype or array.ndim != self.ndim:
            raise TypeError(
                "Full typed LatticeField constructor requires data with exact "
                f"dtype {self.dtype} and ndim {self.ndim}; got {array.dtype} "
                f"and ndim {array.ndim}."
            )
        return LatticeField._from_full(
            array,
            lattice,
            flambda,
            self.tag,
            expected_dtype=self.dtype,
        )


class _CheckedFlatIterator:
    """Flat view whose assignments keep Julia array conversion semantics."""

    __slots__ = ("_owner",)

    def __init__(self, owner: "_CheckedFieldArray") -> None:
        self._owner = owner

    def __iter__(self) -> Iterator[Any]:
        return iter(np.asarray(self._owner).flat)

    def __len__(self) -> int:
        return self._owner.size

    def __getitem__(self, key: Any) -> Any:
        return np.asarray(self._owner).flat[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        flat_indices = np.arange(self._owner.size)[key]
        target_indices = np.asarray(flat_indices)
        if target_indices.ndim == 0:
            coordinates = np.unravel_index(
                int(target_indices), self._owner.shape, order="C"
            )
            self._owner[coordinates] = value
            return
        values = np.broadcast_to(np.asarray(value), target_indices.shape)
        for index in np.ndindex(target_indices.shape):
            coordinates = np.unravel_index(
                int(target_indices[index]), self._owner.shape, order="C"
            )
            self._owner[coordinates] = values[index]

    def __getattr__(self, name: str) -> Any:
        return getattr(np.asarray(self._owner).flat, name)


class _CheckedFieldArray(np.ndarray):
    """Field storage that rejects NumPy's lossy implicit assignments.

    Julia exposes ``field.data`` as a mutable typed array, whose own
    ``setindex!`` performs checked conversion. A plain NumPy ndarray instead
    truncates float-to-int and complex-to-real assignments. This view retains
    ndarray interoperability and storage aliasing while validating every
    ordinary assignment through the same conversion routine as
    ``LatticeField.__setitem__``.
    """

    def __new__(cls, values: Any) -> "_CheckedFieldArray":
        if isinstance(values, cls):
            return values
        source = np.asarray(values)
        result = np.array(
            source,
            copy=True,
            order="F" if source.flags.f_contiguous else "C",
        ).view(cls)
        result.setflags(write=False)
        return result

    def __array_finalize__(self, obj: Any) -> None:
        del obj

    def __getitem__(self, key: Any) -> Any:
        result = super().__getitem__(key)
        if isinstance(result, np.ndarray) and result.ndim == 0:
            return np.asarray(result).reshape(())[()]
        return result

    def __setitem__(self, key: Any, value: Any) -> None:
        target = np.ndarray.__getitem__(self, key)
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
                np.asarray(value), np.asarray(self)
            )
            if converted.ndim != 0:
                raise ValueError(
                    f"Inexact assignment to scalar element type {self.dtype}."
                )
            self.setflags(write=True)
            try:
                np.ndarray.__setitem__(self, key, converted.reshape(())[()])
            finally:
                self.setflags(write=False)
            return

        try:
            broadcast = np.broadcast_to(np.asarray(value), target.shape)
        except ValueError as error:
            raise ValueError(
                "Assignment source cannot be broadcast to the selected shape."
            ) from error

        # Julia's broadcast assignment converts and stores one element at a
        # time in column-major Cartesian order. A late InexactError therefore
        # leaves the successfully converted prefix mutated.
        self.setflags(write=True)
        try:
            writable_target = np.ndarray.__getitem__(self, key)
            for linear_index in range(writable_target.size):
                index = np.unravel_index(
                    linear_index, writable_target.shape, order="F"
                )
                converted = _julia_assignment_values(
                    np.asarray(broadcast[index]), np.asarray(self)
                )
                np.ndarray.__setitem__(
                    writable_target, index, converted.reshape(())[()]
                )
        finally:
            self.setflags(write=False)

    @property
    def flat(self) -> _CheckedFlatIterator:
        return _CheckedFlatIterator(self)

    @flat.setter
    def flat(self, value: Any) -> None:
        self.flat[:] = value

    def fill(self, value: Any) -> None:
        converted = _julia_assignment_values(value, np.asarray(self))
        if converted.ndim != 0:
            raise ValueError("fill requires a scalar value")
        self.setflags(write=True)
        try:
            np.ndarray.fill(self, converted.reshape(())[()])
        finally:
            self.setflags(write=False)

    def put(
        self,
        indices: Any,
        values: Any,
        mode: str = "raise",
    ) -> None:
        raw_indices = np.asarray(indices)
        raw_values = np.broadcast_to(np.asarray(values), raw_indices.shape)
        for index in np.ndindex(raw_indices.shape):
            flat_index = int(raw_indices[index])
            if mode == "wrap":
                flat_index %= self.size
            elif mode == "clip":
                flat_index = min(max(flat_index, 0), self.size - 1)
            elif flat_index < 0:
                flat_index += self.size
            coordinates = np.unravel_index(flat_index, self.shape, order="C")
            self[coordinates] = raw_values[index]

    def _inplace(self, other: Any, operation: np.ufunc) -> "_CheckedFieldArray":
        if np.asarray(other).ndim == 0:
            result = _julia_array_scalar_operation(
                np.asarray(self), np.asarray(other).reshape(())[()], operation
            )
        else:
            result = _julia_array_array_operation(
                np.asarray(self), np.asarray(other), operation
            )
        self[...] = result
        return self

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
        "data",
        "L",
        "flambda",
        "field_type",
        "_frozen",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False) and name in {
            "data",
            "L",
            "flambda",
            "field_type",
        }:
            if name == "data" and value is getattr(self, "data", None):
                # Python writes the result of ``field.data += value`` back to
                # the attribute after the checked in-place mutation. This is
                # not a metadata change.
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
        array = (
            data
            if isinstance(data, _CheckedFieldArray)
            else np.asarray(data)
        )
        axes = as_lattice(lattice)
        if field_type is Intensity:
            # Deliberately allocate, matching Julia's broadcast constructor.
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
                    if value < 0:
                        array[index] = type(value)(0)
            else:
                array = np.where(array < 0, np.zeros((), dtype=array.dtype), array)
        elif field_type is ComplexAmplitude:
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
        result = cls.__new__(cls)
        result._initialize(array, as_lattice(lattice), flambda, field_type)
        return result

    def _initialize(
        self,
        data: np.ndarray,
        lattice: Lattice,
        flambda: Real,
        field_type: type[FieldVal],
    ) -> None:
        object.__setattr__(self, "_frozen", False)
        if data.ndim != len(lattice) or data.shape != tuple(map(len, lattice)):
            raise DimensionMismatch("Field data size does not match lattice size.")
        self.data = _CheckedFieldArray(data)
        self.L = lattice
        self.flambda = flambda
        self.field_type = field_type
        object.__setattr__(self, "_frozen", True)

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
        if scalar_dtype.kind == "c" and self.data.dtype.kind != "c":
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
        positive = (
            values.astype(np.int64)
            if values.dtype == np.dtype(np.bool_)
            else +values
        )
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
        return LatticeField[Modulus](np.abs(self.data), self.L, self.flambda)

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
                    conjugated = -values
            return LatticeField[RealPhase](
                conjugated, self.L, self.flambda
            )
        if self.field_type is ComplexPhase:
            return LatticeField[ComplexPhase](
                np.conjugate(self.data), self.L, self.flambda
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
    elq(left, right)
    ls = left.field_type
    rs = right.field_type
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
        # Python infix operators are necessarily binary.  Preserve ordinary
        # value semantics: a subsequent operation observes exactly the data
        # visible in each operand.  Julia's distinct unparenthesized n-ary
        # overload has no equivalent Python syntax and must not be emulated by
        # storing hidden expression history on the returned field.
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
        magnitude = np.abs(values)
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
        isinstance(value, Decimal) for value in values.flat
    ):
        output = np.empty(values.shape, dtype=object)
        # Julia's ``2pi`` token is Float64. Multiplication promotes that exact
        # binary64 value to BigFloat before applying the high-precision exp.
        factor = Decimal.from_float(float(2 * np.pi))
        for index in np.ndindex(values.shape):
            sine, cosine = _decimal_sincos(factor * values[index])
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

    with np.errstate(divide="ignore", invalid="ignore"):
        if field.field_type is Intensity:
            result = np.abs(field.data / np.sum(field.data))
        elif issubclass(field.field_type, Amplitude):
            result = field.data / np.sqrt(np.sum(np.abs(field.data) ** 2))
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
        vectorized = np.ones(value.shape, dtype=np.complex128)
        nonzero = value.data != 0
        vectorized[nonzero] = value.data[nonzero] / np.abs(value.data[nonzero])
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
