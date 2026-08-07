"""Dual coordinate lattices and centered discrete Fourier transforms."""

from __future__ import annotations

from fractions import Fraction
from numbers import Number, Real
from typing import Any

import numpy as np
from pyfftw.interfaces import numpy_fft as _fftw_fft

from ._bigfloat import _MPFRComplex, _MPQ, _MPZ, _is_bigfloat_input, _is_mpc
from .lattice_field import (
    ComplexAmplitude,
    DomainError,
    Lattice,
    LatticeField,
    RealPhase,
    _axis,
    _isapprox_array,
    _julia_array_scalar_operation,
    _julia_literal_array,
    _logical_axis_scalar_operation,
    _with_axis_length_kind,
    as_lattice,
)
from .lattice_utils import _step, latticeDisplacement, toDim


class _DefaultFlambda:
    def __repr__(self) -> str:
        return "1"


_UNSET = _DefaultFlambda()


def dualLattice(lattice: Any, flambda: Number = 1) -> Lattice:
    """Return the unshifted all-nonnegative DFT frequency lattice."""

    if not isinstance(flambda, (Number, np.bool_)):
        raise TypeError("flambda must be a Julia Number")
    axes = as_lattice(lattice)
    output = []
    for axis in axes:
        step = _step(axis)
        # Julia propagates IEEE Inf/NaN for empty axes without an emitted
        # warning. NumPy reports the same arithmetic through RuntimeWarning
        # unless it is locally suppressed.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denominator = _julia_array_scalar_operation(
                np.asarray(len(axis), dtype=np.int64), step, np.multiply
            )[()]
            indices = _axis(
                np.arange(len(axis), dtype=np.int64),
                step_hint=np.int64(1),
            )
            scaled = _logical_axis_scalar_operation(
                indices, flambda, np.multiply
            )
            result = _logical_axis_scalar_operation(
                scaled, denominator, np.divide
            )
            output.append(
                _with_axis_length_kind(
                    result, getattr(axis, "_length_kind", "int64")
                )
            )
    return tuple(output)


def dualShiftLattice(lattice: Any, flambda: Number = 1) -> Lattice:
    """Return the centered/fftshifted dual lattice."""

    if not isinstance(flambda, (Number, np.bool_)):
        raise TypeError("flambda must be a Julia Number")
    axes = as_lattice(lattice)
    output = []
    for axis in axes:
        n = len(axis)
        step = _step(axis)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denominator = _julia_array_scalar_operation(
                np.asarray(n, dtype=np.int64), step, np.multiply
            )[()]
            frequencies = _axis(
                np.arange(-(n // 2), (n - 1) // 2 + 1, dtype=np.int64),
                step_hint=np.int64(1),
            )
            scaled = _logical_axis_scalar_operation(
                frequencies, flambda, np.multiply
            )
            result = _logical_axis_scalar_operation(
                scaled, denominator, np.divide
            )
            output.append(
                _with_axis_length_kind(
                    result, getattr(axis, "_length_kind", "int64")
                )
            )
    return tuple(output)


def ldq(left: Any, right: Any, flambda: Any = _UNSET) -> None:
    """Require lattices or lattice fields to be shifted Fourier duals."""

    if isinstance(left, LatticeField) and isinstance(right, LatticeField):
        if flambda is not _UNSET:
            raise TypeError("ldq(field, field) does not accept a flambda argument")
        expected = dualShiftLattice(left.L, left.flambda)
        if len(expected) != len(right.L) or not all(
            _isapprox_array(a, b) for a, b in zip(expected, right.L, strict=True)
        ):
            raise DomainError("Non-dual lattices.")
        # Intentionally exact, matching the Julia LF overload.
        if left.flambda != right.flambda:
            raise DomainError("Non-dual lattices.")
        return None
    if isinstance(left, LatticeField) or isinstance(right, LatticeField):
        raise TypeError("ldq arguments must both be lattices or both be lattice fields")
    if flambda is _UNSET:
        flambda = 1
    first = as_lattice(left)
    second = as_lattice(right)
    expected = dualShiftLattice(first, flambda)
    if len(expected) != len(second) or not all(
        _isapprox_array(a, b) for a, b in zip(expected, second, strict=True)
    ):
        raise DomainError("Non-dual lattices.")
    return None


def dualPhase(
    lattice: Any,
    flambda: Real = 1.0,
    *,
    dL: Any | None = None,
) -> LatticeField:
    """Return the real phase ramp caused by displacement of a lattice origin."""

    axes = as_lattice(lattice)
    if any(
        np.asarray(axis).dtype.kind == "O"
        and any(
            isinstance(value, _MPZ)
            or (
                type(value) is int
                and not np.iinfo(np.int64).min
                <= value
                <= np.iinfo(np.int64).max
            )
            for value in np.asarray(axis).flat
        )
        for axis in axes
    ):
        raise TypeError(
            "Julia dualPhase fails for BigInt lattice ranges in range construction"
        )
    if len(axes) == 0:
        # Julia reaches the ambiguous zero-argument broadcasted `+()` in this
        # case.  Do not invent a scalar zero-dimensional phase field.
        raise TypeError("dualPhase is undefined for a zero-dimensional lattice.")
    dual = dualShiftLattice(axes, flambda) if dL is None else as_lattice(dL)
    if len(dual) != len(axes):
        raise DomainError("Dual lattice dimension does not match lattice dimension.")
    displacement = latticeDisplacement(axes)
    if len(dual) == 1:
        # ``toDim(range, 1, 1)`` remains the range itself in Julia.  The
        # following scalar multiply and divide therefore retain a
        # Float16/Float32 StepRangeLen's high-precision reference/step.  In
        # two or more dimensions, reshape produces a dense-array operation,
        # which is handled by the materialized branch below.
        phase_axis = _logical_axis_scalar_operation(
            dual[0], displacement[0], np.multiply
        )
        phase_axis = _logical_axis_scalar_operation(
            phase_axis, flambda, np.divide
        )
        return LatticeField[RealPhase](np.asarray(phase_axis), dual, flambda)
    phase: np.ndarray | None = None
    for i, (offset, axis) in enumerate(zip(displacement, dual, strict=True), start=1):
        term = _julia_array_scalar_operation(
            toDim(axis, i, len(dual)), offset, np.multiply
        )
        phase = term if phase is None else phase + term
    if phase is None:
        phase = np.asarray(0.0)
    phase = _julia_array_scalar_operation(phase, flambda, np.divide)
    return LatticeField[RealPhase](phase, dual, flambda)


def _sft_array(value: Any) -> np.ndarray:
    array = _fft_input(value)
    return np.fft.fftshift(
        _fftw_fft.fftn(
            np.fft.ifftshift(array),
            planner_effort="FFTW_ESTIMATE",
        )
    )


def _isft_array(value: Any) -> np.ndarray:
    array = _fft_input(value)
    return np.fft.fftshift(
        _fftw_fft.ifftn(
            np.fft.ifftshift(array),
            planner_effort="FFTW_ESTIMATE",
        )
    )


def _fft_input(value: Any) -> np.ndarray:
    """Validate Julia FFTW domains and convert Rational input to Float64."""

    array = (
        _julia_literal_array(value)
        if isinstance(value, list)
        else np.asarray(value)
    )
    if array.dtype.kind == "O":
        items = tuple(array.flat)
        if items and all(isinstance(item, Fraction) for item in items):
            int64 = np.iinfo(np.int64)
            if not all(
                int64.min <= item.numerator <= int64.max
                and 1 <= item.denominator <= int64.max
                for item in items
            ):
                raise TypeError("type BigFloat not supported")
            return np.asarray(array, dtype=np.float64)
        if any(
            _is_bigfloat_input(item)
            or _is_mpc(item)
            or isinstance(item, _MPFRComplex)
            or isinstance(item, (_MPQ, _MPZ))
            or (
                type(item) is int
                and not np.iinfo(np.int64).min
                <= item
                <= np.iinfo(np.int64).max
            )
            for item in items
        ):
            raise TypeError("type BigFloat not supported")
        # NumPy/pyFFTW would coerce arbitrary object arrays to complex128.
        # Julia's FFTW dispatch instead rejects arrays without a concrete
        # supported numeric element type.
        raise TypeError("object-array element type not supported by FFTW")

    if array.dtype.kind in "bui":
        return array
    if array.dtype in (
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.complex64),
        np.dtype(np.complex128),
    ):
        return array
    raise TypeError(f"type {array.dtype.name} not supported by FFTW")


def sft(value: Any) -> Any:
    """Apply ``fftshift(fftn(ifftshift(value)))`` over every dimension."""

    if isinstance(value, LatticeField):
        if value.field_type is not ComplexAmplitude:
            raise TypeError("sft(field) requires ComplexAmplitude.")
        return LatticeField[ComplexAmplitude](
            _sft_array(value.data),
            dualShiftLattice(value.L, value.flambda),
            value.flambda,
        )
    if isinstance(value, tuple):
        raise TypeError("sft requires an array or range, not a tuple")
    return _sft_array(value)


def isft(value: Any) -> Any:
    """Apply ``fftshift(ifftn(ifftshift(value)))`` over every dimension."""

    if isinstance(value, LatticeField):
        if value.field_type is not ComplexAmplitude:
            raise TypeError("isft(field) requires ComplexAmplitude.")
        return LatticeField[ComplexAmplitude](
            _isft_array(value.data),
            dualShiftLattice(value.L, value.flambda),
            value.flambda,
        )
    if isinstance(value, tuple):
        raise TypeError("isft requires an array or range, not a tuple")
    return _isft_array(value)


__all__ = [
    "dualLattice",
    "dualPhase",
    "dualShiftLattice",
    "isft",
    "ldq",
    "sft",
]
