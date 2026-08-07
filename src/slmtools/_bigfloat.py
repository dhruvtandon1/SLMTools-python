"""Julia ``BigFloat`` compatibility on top of MPFR.

``decimal.Decimal`` is accepted as an input adapter because it is convenient
for spelling exact decimal values, but it is not a numerical implementation
of Julia's binary ``BigFloat``.  Julia 1.11.6 uses MPFR at a default precision
of 256 bits.  This module performs every elementary operation in an isolated
gmpy2/MPFR context with that precision and returns :class:`gmpy2.mpfr`
components so decimal radix and exponent limits cannot silently alter a
result.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import operator
from typing import Any

import gmpy2
import numpy as np


JULIA_BIGFLOAT_PRECISION = 256
_MPFR = gmpy2.mpfr
_MPC = gmpy2.mpc
_MPQ = gmpy2.mpq
_MPZ = gmpy2.mpz

# gmpy2's exponent limits are already much wider than Decimal's and cover the
# locked Julia probes.  Copy them into the dedicated context rather than
# changing the process-global gmpy2 context.
_DEFAULT_MPFR_CONTEXT = gmpy2.get_context()
_JULIA_BIGFLOAT_CONTEXT = gmpy2.context(
    precision=JULIA_BIGFLOAT_PRECISION,
    real_prec=JULIA_BIGFLOAT_PRECISION,
    imag_prec=JULIA_BIGFLOAT_PRECISION,
    round=gmpy2.RoundToNearest,
    real_round=gmpy2.RoundToNearest,
    imag_round=gmpy2.RoundToNearest,
    emax=_DEFAULT_MPFR_CONTEXT.emax,
    emin=_DEFAULT_MPFR_CONTEXT.emin,
    subnormalize=False,
    trap_underflow=False,
    trap_overflow=False,
    trap_inexact=False,
    trap_invalid=False,
    trap_erange=False,
    trap_divzero=False,
    allow_complex=False,
    rational_division=False,
)


def _bigfloat_context() -> gmpy2.context:
    """Return a context manager for Julia's default BigFloat arithmetic."""

    return gmpy2.context(_JULIA_BIGFLOAT_CONTEXT)


def _is_mpfr(value: Any) -> bool:
    return isinstance(value, _MPFR)


def _is_mpc(value: Any) -> bool:
    return isinstance(value, _MPC)


def _is_bigfloat_input(value: Any) -> bool:
    """Whether *value* denotes a Julia-BigFloat-domain scalar."""

    return isinstance(value, (Decimal, _MPFR))


def _to_mpfr(value: Any) -> gmpy2.mpfr:
    """Convert a real scalar to a 256-bit MPFR value with one rounding."""

    with _bigfloat_context():
        if isinstance(value, _MPFR):
            return _MPFR(value)
        if isinstance(value, Decimal):
            # Decimal -> string -> MPFR parses the represented decimal value
            # directly.  Converting through binary64 would double-round and
            # would reject Decimal exponents outside the float range.
            return _MPFR(str(value))
        if isinstance(value, Fraction):
            return _MPFR(_MPQ(value.numerator, value.denominator))
        if isinstance(value, (_MPQ, _MPZ)):
            return _MPFR(value)
        if isinstance(value, (bool, int, np.integer)):
            return _MPFR(int(value))
        if isinstance(value, (float, np.floating)):
            return _MPFR(float(value))
    raise TypeError("value cannot be promoted to Julia BigFloat")


def _to_mpfr_array(value: Any) -> np.ndarray:
    source = np.asarray(value)
    output = np.empty(source.shape, dtype=object)
    for index in np.ndindex(source.shape):
        output[index] = _to_mpfr(source[index])
    return output


def _mpfr_pi() -> gmpy2.mpfr:
    with _bigfloat_context():
        return gmpy2.const_pi(JULIA_BIGFLOAT_PRECISION)


def _mpfr_sincos(angle: Any) -> tuple[gmpy2.mpfr, gmpy2.mpfr]:
    """Return ``(sin(angle), cos(angle))`` in the Julia MPFR context."""

    with _bigfloat_context():
        value = _to_mpfr(angle)
        sine, cosine = gmpy2.sin_cos(value)
        return _MPFR(sine), _MPFR(cosine)


def _mpfr_sqrt(value: Any) -> gmpy2.mpfr:
    with _bigfloat_context():
        return _MPFR(gmpy2.sqrt(_to_mpfr(value)))


def _mpfr_rtol() -> gmpy2.mpfr:
    """Return Julia's default ``isapprox`` tolerance for ``BigFloat``."""

    with _bigfloat_context():
        epsilon = gmpy2.exp2(1 - JULIA_BIGFLOAT_PRECISION)
        return _MPFR(gmpy2.sqrt(epsilon))


def _mpfr_abs(value: Any) -> gmpy2.mpfr:
    with _bigfloat_context():
        return _MPFR(abs(_to_mpfr(value)))


def _mpfr_object_operation(
    operation: np.ufunc, left: Any, right: Any
) -> np.ndarray:
    """Apply an object-array ufunc inside the 256-bit MPFR context."""

    raw_left = np.asarray(left, dtype=object)
    raw_right = np.asarray(right, dtype=object)
    items = (*raw_left.flat, *raw_right.flat)
    has_mpc = any(isinstance(value, _MPC) for value in items)
    has_wrapped_complex = any(
        isinstance(value, _MPFRComplex) for value in items
    )
    has_machine_complex = any(
        isinstance(value, (complex, np.complexfloating)) for value in items
    )
    has_mpfr = any(
        isinstance(value, (_MPFR, Decimal)) for value in items
    )
    has_machine_float = any(
        isinstance(value, (float, np.floating)) for value in items
    )
    has_rational = any(
        isinstance(value, (_MPQ, Fraction)) for value in items
    )

    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if has_mpc or has_machine_complex:
            if isinstance(value, _MPC):
                return _MPC(value)
            if isinstance(value, _MPFRComplex):
                return _MPC(value.real, value.imag)
            if isinstance(value, (complex, np.complexfloating)):
                return _MPC(_to_mpfr(value.real), _to_mpfr(value.imag))
            # Keep real operands real. Julia distinguishes Complex/Real
            # division from Complex/Complex division at zero: the former
            # yields component-wise infinities while the latter is NaN.
            return _to_mpfr(value)
        if has_wrapped_complex:
            return (
                value
                if isinstance(value, _MPFRComplex)
                else _to_mpfr(value)
            )
        if has_mpfr or has_machine_float:
            return _to_mpfr(value)
        if has_rational:
            if isinstance(value, _MPQ):
                return value
            if isinstance(value, Fraction):
                return _MPQ(value.numerator, value.denominator)
            return _MPQ(value)
        return value if isinstance(value, _MPZ) else _MPZ(value)

    if operation is np.matmul:
        converted_left = np.empty(raw_left.shape, dtype=object)
        converted_right = np.empty(raw_right.shape, dtype=object)
        with _bigfloat_context():
            for index in np.ndindex(raw_left.shape):
                converted_left[index] = convert(raw_left[index])
            for index in np.ndindex(raw_right.shape):
                converted_right[index] = convert(raw_right[index])
            result = np.matmul(converted_left, converted_right)
        if isinstance(result, np.ndarray):
            return np.asarray(result, dtype=object)
        output = np.empty((), dtype=object)
        output[()] = result
        return output

    left_array, right_array = np.broadcast_arrays(raw_left, raw_right)

    scalar_operations = {
        np.add: operator.add,
        np.subtract: operator.sub,
        np.multiply: operator.mul,
        np.divide: operator.truediv,
        np.true_divide: operator.truediv,
        np.remainder: operator.mod,
        np.power: operator.pow,
        np.greater: operator.gt,
        np.greater_equal: operator.ge,
        np.less: operator.lt,
        np.less_equal: operator.le,
        np.equal: operator.eq,
        np.not_equal: operator.ne,
    }
    scalar_operation = scalar_operations.get(operation)
    if scalar_operation is None:
        raise TypeError(
            f"unsupported Julia BigFloat object operation {operation.__name__}"
        )
    comparison = operation in {
        np.greater,
        np.greater_equal,
        np.less,
        np.less_equal,
        np.equal,
        np.not_equal,
    }
    output = np.empty(
        left_array.shape,
        dtype=np.bool_ if comparison else object,
    )
    with _bigfloat_context():
        for index in np.ndindex(left_array.shape):
            output[index] = scalar_operation(
                convert(left_array[index]),
                convert(right_array[index]),
            )
    return output


@dataclass(frozen=True, slots=True)
class _MPFRComplex:
    """Python object-number counterpart of Julia ``Complex{BigFloat}``."""

    real: Any
    imag: Any = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "real", _to_mpfr(self.real))
        object.__setattr__(self, "imag", _to_mpfr(self.imag))

    @staticmethod
    def _coerce(value: Any) -> "_MPFRComplex" | Any:
        if isinstance(value, _MPFRComplex):
            return value
        if isinstance(
            value,
            (
                Decimal,
                _MPFR,
                _MPQ,
                _MPZ,
                Fraction,
                bool,
                int,
                np.integer,
                float,
                np.floating,
            ),
        ):
            return _MPFRComplex(value)
        if isinstance(value, _MPC):
            return _MPFRComplex(value.real, value.imag)
        if isinstance(value, (complex, np.complexfloating)):
            scalar = complex(value)
            return _MPFRComplex(scalar.real, scalar.imag)
        return NotImplemented

    def __complex__(self) -> complex:
        return complex(float(self.real), float(self.imag))

    def __abs__(self) -> gmpy2.mpfr:
        with _bigfloat_context():
            return _MPFR(gmpy2.hypot(self.real, self.imag))

    def __eq__(self, other: Any) -> bool | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        return self.real == converted.real and self.imag == converted.imag

    def __hash__(self) -> int:
        return (
            hash(self.real)
            if self.imag == 0
            else hash((self.real, self.imag))
        )

    def conjugate(self) -> "_MPFRComplex":
        return _MPFRComplex(self.real, -self.imag)

    def __neg__(self) -> "_MPFRComplex":
        return _MPFRComplex(-self.real, -self.imag)

    def __add__(self, other: Any) -> "_MPFRComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        with _bigfloat_context():
            return _MPFRComplex(
                self.real + converted.real,
                self.imag + converted.imag,
            )

    __radd__ = __add__

    def __sub__(self, other: Any) -> "_MPFRComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        with _bigfloat_context():
            return _MPFRComplex(
                self.real - converted.real,
                self.imag - converted.imag,
            )

    def __rsub__(self, other: Any) -> "_MPFRComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        return converted - self

    def __mul__(self, other: Any) -> "_MPFRComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        with _bigfloat_context():
            return _MPFRComplex(
                self.real * converted.real - self.imag * converted.imag,
                self.real * converted.imag + self.imag * converted.real,
            )

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "_MPFRComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        with _bigfloat_context():
            if not isinstance(
                other,
                (_MPFRComplex, _MPC, complex, np.complexfloating),
            ):
                return _MPFRComplex(
                    self.real / converted.real,
                    self.imag / converted.real,
                )
            are, aim = self.real, self.imag
            bre, bim = converted.real, converted.imag
            if gmpy2.is_infinite(bre) or gmpy2.is_infinite(bim):
                if gmpy2.is_finite(are) and gmpy2.is_finite(aim):
                    real_negative = bool(
                        gmpy2.is_signed(are) ^ gmpy2.is_signed(bre)
                    )
                    imag_negative = bool(
                        True ^ gmpy2.is_signed(aim) ^ gmpy2.is_signed(bim)
                    )
                    return _MPFRComplex(
                        _MPFR("-0") if real_negative else _MPFR(0),
                        _MPFR("-0") if imag_negative else _MPFR(0),
                    )
                nan = _MPFR("nan")
                return _MPFRComplex(nan, nan)
            # Match Base.Complex's Smith-style scaled division rather than
            # delegating to MPC, whose elementary rounding sequence differs.
            if abs(bre) <= abs(bim):
                ratio = bre / bim
                denominator = bim + ratio * bre
                return _MPFRComplex(
                    (are * ratio + aim) / denominator,
                    (aim * ratio - are) / denominator,
                )
            ratio = bim / bre
            denominator = bre + ratio * bim
            return _MPFRComplex(
                (are + aim * ratio) / denominator,
                (aim - are * ratio) / denominator,
            )

    def __rtruediv__(self, other: Any) -> "_MPFRComplex" | Any:
        converted = self._coerce(other)
        if converted is NotImplemented:
            return NotImplemented
        return converted / self

    def __pow__(self, exponent: Any) -> "_MPFRComplex" | Any:
        if not _is_bigfloat_input(exponent):
            return NotImplemented
        with _bigfloat_context():
            power = _to_mpfr(exponent)
            modulus_squared = self.real * self.real + self.imag * self.imag
            if modulus_squared == 0:
                if power > 0:
                    return _MPFRComplex(0, 0)
                raise ZeroDivisionError(
                    "zero complex value cannot have this exponent"
                )
            log_radius = gmpy2.log(modulus_squared) / 2
            angle = gmpy2.atan2(self.imag, self.real)
            magnitude = gmpy2.exp(power * log_radius)
            sine, cosine = gmpy2.sin_cos(power * angle)
            return _MPFRComplex(magnitude * cosine, magnitude * sine)

