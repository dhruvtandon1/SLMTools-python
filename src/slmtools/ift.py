"""Iterative Fourier-transform phase retrieval.

This module is a direct NumPy port of ``src/PhaseRetrieval/IFT.jl``.  Phase
values represented by :class:`RealPhase` are measured in cycles, not radians;
``exp(2j*pi*phase)`` is therefore used whenever a real phase is applied.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal
from fractions import Fraction
import os
from typing import Any

import numpy as np
import pyfftw
from pyfftw.interfaces import numpy_fft as _fftw_fft

from ._bigfloat import (
    _MPC,
    _MPFR,
    _MPFRComplex,
    _MPQ,
    _MPZ,
    _mpfr_sqrt,
)
from ._omission import _OMITTED
from .lattice_field import (
    ComplexAmp,
    ComplexPhase,
    Intensity,
    LF,
    LatticeField,
    Modulus,
    Phase,
    RealPhase,
    _is_real_number,
    _julia_add_sum,
    _julia_abs,
    _julia_array_array_operation,
    _julia_array_scalar_operation,
    _julia_sum,
    _real_phase_phasors,
    _require_dense_ndarray,
    as_lattice,
    elq,
)
from .dual_lattices import dualPhase, dualShiftLattice, isft as _field_isft
from .lattice_utils import (
    _step as _lattice_step,
    ldot as _lattice_ldot,
    r2 as _lattice_r2,
)


__all__ = [
    "gs",
    "gsIter",
    "gsLog",
    "gsError",
    "pdgs",
    "pdgsIter",
    "pdgsLog",
    "pdgsError",
    "oneShot",
    "mraf",
]


def _make_field(tag: type, data: Any, lattice: Sequence[Any], flambda: float) -> LatticeField:
    """Construct an LF without depending on operator overloads in the core."""

    array = np.asarray(data)
    try:
        return LF[tag](array, tuple(lattice), flambda)
    except (TypeError, AttributeError):
        return LatticeField(array, tuple(lattice), flambda=flambda, field_type=tag)


def _tag_is(field: LatticeField, tag: type) -> bool:
    actual = getattr(field, "field_type", None)
    if actual is tag:
        return True
    try:
        return isinstance(actual, type) and issubclass(actual, tag)
    except TypeError:
        return False


def _is_numeric_data(value: Any) -> bool:
    """Return whether an array has a Julia ``<:Number`` element analogue."""

    array = np.asarray(value)
    dtype = array.dtype
    if dtype.kind == "O":
        return bool(array.size) and all(
            _is_real_number(item)
            or isinstance(item, (complex, np.complexfloating))
            for item in array.flat
        )
    return bool(
        np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)
    )


def _is_real_numeric_data(value: Any) -> bool:
    """Return whether an array has a Julia ``<:Real`` element analogue."""

    array = np.asarray(value)
    dtype = array.dtype
    if dtype.kind == "O":
        return bool(array.size) and all(
            _is_real_number(item) for item in array.flat
        )
    return _is_numeric_data(value) and not np.issubdtype(dtype, np.complexfloating)


def _has_arbitrary_precision_data(value: Any) -> bool:
    array = np.asarray(value)
    if array.dtype.kind != "O":
        return False
    limits = np.iinfo(np.int64)
    return any(
        isinstance(
            item,
            (Decimal, _MPFR, _MPC, _MPFRComplex, _MPQ, _MPZ),
        )
        or (
            type(item) is int and not limits.min <= item <= limits.max
        )
        or (
            isinstance(item, Fraction)
            and (
                not limits.min <= item.numerator <= limits.max
                or not 1 <= item.denominator <= limits.max
            )
        )
        for item in array.flat
    )


def _has_arbitrary_precision_modulus(value: Any) -> bool:
    return _has_arbitrary_precision_data(value)


def _is_complex_numeric_data(value: Any) -> bool:
    """Return whether an array has a concrete Julia ``<:Complex`` element."""

    array = np.asarray(value)
    if array.dtype.kind == "c":
        return True
    return bool(array.size) and array.dtype.kind == "O" and all(
        isinstance(item, (complex, np.complexfloating, _MPC, _MPFRComplex))
        for item in array.flat
    )


def _require_real_field_data(field: LatticeField, name: str) -> None:
    if not _is_real_numeric_data(field.data):
        raise TypeError(f"{name} must have real numeric data")


def _is_phase(field: LatticeField) -> bool:
    return _tag_is(field, Phase) or _tag_is(field, RealPhase) or _tag_is(field, ComplexPhase)


def _as_lattice(lattice: Sequence[Any]) -> tuple[np.ndarray, ...]:
    """Canonicalize without discarding ``LatticeAxis`` logical-step metadata."""

    return as_lattice(lattice)


def _step(axis: Any) -> float:
    """Return the retained range step, including for singleton axes."""

    return float(_lattice_step(axis))


def _same_lattice(left: Sequence[Any], right: Sequence[Any]) -> bool:
    try:
        elq(tuple(left), tuple(right))
    except (TypeError, ValueError):
        return False
    return True


def _dual_shift_lattice(lattice: Sequence[Any], flambda: float = 1.0) -> tuple[np.ndarray, ...]:
    return dualShiftLattice(lattice, flambda)


def _to_dim(values: Any, dimension: int, ndim: int) -> np.ndarray:
    shape = [1] * ndim
    shape[dimension] = len(values)
    return np.asarray(values).reshape(shape)


def _r2(lattice: Sequence[Any]) -> np.ndarray:
    ndim = len(lattice)
    result: np.ndarray | float = 0.0
    for dimension, axis in enumerate(lattice):
        result = result + _to_dim(np.asarray(axis, dtype=float) ** 2, dimension, ndim)
    return np.asarray(result)


def _ldot(vector: Sequence[float], lattice: Sequence[Any]) -> np.ndarray:
    if len(vector) != len(lattice):
        raise ValueError("vector length must equal lattice dimensionality")
    ndim = len(lattice)
    result: np.ndarray | float = 0.0
    for dimension, (coefficient, axis) in enumerate(zip(vector, lattice, strict=True)):
        result = result + float(coefficient) * _to_dim(axis, dimension, ndim)
    return np.asarray(result)


def _lattice_displacement(lattice: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [
            np.asarray(axis)[0] + np.floor(len(axis) / 2) * _step(axis)
            for axis in lattice
        ],
        dtype=float,
    )


def _dual_phase_array(lattice: Sequence[Any], flambda: float) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    dual = _dual_shift_lattice(lattice, flambda)
    # Share the public implementation so Float16/Float32 StepRangeLen
    # reference arithmetic is retained exactly as it is in Julia.  The old
    # private copy converted both the displacement and the dual coordinates
    # to materialized Float64 arrays too early.
    return np.asarray(dualPhase(lattice, flambda, dL=dual).data), dual


def _phasor(values: Any) -> np.ndarray:
    """Return unit phasors, defining the phase of zero as ``1 + 0j``."""

    z = np.asarray(values, dtype=np.complex128)
    magnitude = np.abs(z)
    result = np.ones(z.shape, dtype=np.complex128)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(z, magnitude, out=result, where=magnitude != 0)
    return result


def _phase_array(phase: LatticeField) -> np.ndarray:
    if _tag_is(phase, RealPhase):
        # ``RealPhase`` describes how the values are interpreted, not a
        # restriction on their storage type: Julia's public signatures admit
        # ``LF{RealPhase,<:Number}``.  In particular, ``wrap`` evaluates the
        # complete complex exponential for complex-valued data.  Multiplying
        # by Float64 ``2pi`` in Julia makes the work array ComplexF64 for all
        # ordinary integer/Float16/Float32 inputs, hence the explicit
        # Complex128 conversion here.
        if _has_arbitrary_precision_data(phase.data):
            return _real_phase_phasors(phase.data)
        return np.exp(2j * np.pi * np.asarray(phase.data, dtype=np.complex128))
    if _tag_is(phase, ComplexPhase):
        if _has_arbitrary_precision_data(phase.data):
            return np.asarray(phase.data)
        return np.asarray(phase.data, dtype=np.complex128)
    if _is_phase(phase):
        data = np.asarray(phase.data)
        return np.exp(2j * np.pi * data) if np.isrealobj(data) else data.astype(np.complex128)
    raise TypeError("expected a RealPhase or ComplexPhase LatticeField")


def _modulus_field(field: LatticeField) -> LatticeField:
    if getattr(field, "field_type", None) is Modulus:
        return field
    if getattr(field, "field_type", None) is Intensity:
        data = np.asarray(field.data)
        if not _is_numeric_data(data):
            raise TypeError("intensity values must be numeric")
        if _is_real_numeric_data(data) and np.any(data < 0):
            raise ValueError("intensity values must be nonnegative")
        return field.sqrt()
    raise TypeError("expected a Modulus or Intensity LatticeField")


def _homogeneous_modulus_pair(
    left: LatticeField,
    right: LatticeField,
    *,
    intensity_supported: bool,
    function_name: str,
) -> tuple[LatticeField, LatticeField]:
    """Apply Julia's invariant, homogeneous amplitude-field dispatch."""

    left_tag = getattr(left, "field_type", None)
    right_tag = getattr(right, "field_type", None)
    if left_tag is Modulus and right_tag is Modulus:
        fields = (left, right)
    elif intensity_supported and left_tag is Intensity and right_tag is Intensity:
        fields = (left, right)
    else:
        accepted = "both Modulus or both Intensity" if intensity_supported else "both Modulus"
        raise TypeError(f"{function_name} inputs must be {accepted} LatticeFields")
    for field in fields:
        _require_real_field_data(field, f"{function_name} input")
    if left_tag is Intensity:
        return _modulus_field(left), _modulus_field(right)
    return fields


def _sft_array(values: Any) -> np.ndarray:
    return np.fft.fftshift(
        _fftw_fft.fftn(
            np.fft.ifftshift(values),
            planner_effort="FFTW_ESTIMATE",
            threads=_fftw_thread_count(),
        )
    )


def _isft_array(values: Any) -> np.ndarray:
    return np.fft.fftshift(
        _fftw_fft.ifftn(
            np.fft.ifftshift(values),
            planner_effort="FFTW_ESTIMATE",
            threads=_fftw_thread_count(),
        )
    )


def _validate_equal(left: LatticeField, right: LatticeField, message: str = "Non-equal lattices.") -> None:
    try:
        elq(left, right)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


def _validate_dual(left: LatticeField, right: LatticeField) -> None:
    if (
        len(left.L) != len(right.L)
        or not _same_lattice(_dual_shift_lattice(left.L, left.flambda), right.L)
        or left.flambda != right.flambda
    ):
        raise ValueError("Non-dual lattices.")


def _iterations(nit: int) -> int:
    if not isinstance(nit, (int, np.integer)):
        raise TypeError("nit must be an integer")
    # Julia's ``1:nit`` is empty when ``nit`` is negative.  Return the signed
    # count and let Python's ``range`` preserve that empty-loop behavior.
    return int(nit)


def _int64_keyword(value: Any, name: str) -> int:
    """Validate a Julia ``::Int`` keyword on the audited 64-bit platform."""

    if type(value) is int:
        limits = np.iinfo(np.int64)
        if not limits.min <= value <= limits.max:
            raise OverflowError(f"{name} does not fit Julia Int64")
        return value
    if isinstance(value, np.int64):
        return int(value)
    raise TypeError(f"{name} must have Julia Int (Int64) type")


def _scalar_operation(left: Any, right: Any, operation: np.ufunc) -> Any:
    """Apply Julia scalar promotion through the shared array/scalar helper."""

    result = _julia_array_scalar_operation(left, right, operation)
    return np.asarray(result).reshape(())[()]


def _literal_square(value: Any) -> Any:
    """Evaluate Julia's literal ``value^2`` without promoting the exponent."""

    # Julia lowers ``x^2`` through literal_pow, retaining x's type for Bool,
    # small integers, and Float16. Multiplying the scalar by itself reproduces
    # both that result type and its overflow behavior.
    with np.errstate(
        over="ignore", under="ignore", invalid="ignore", divide="ignore"
    ):
        return _scalar_operation(value, value, np.multiply)


def _quadratic_linear_phase(
    lattice: Sequence[Any], alpha: Any, beta: Sequence[Any]
) -> np.ndarray:
    """Evaluate ``alpha / 2 * r2(L) + ldot(beta, L)`` with Julia promotion."""

    half_alpha = _scalar_operation(alpha, 2, np.divide)
    quadratic = _julia_array_scalar_operation(
        _lattice_r2(lattice), half_alpha, np.multiply
    )
    linear = _lattice_ldot(beta, lattice)
    return _julia_array_array_operation(quadratic, linear, np.add)


def _apply_bare_pyfftw_plan(
    plan: pyfftw.FFTW, values: np.ndarray
) -> np.ndarray:
    """Apply a pyFFTW plan without normalizing either transform direction."""

    # pyFFTW couples forward/backward normalization: disabling inverse
    # normalization scales a forward transform by 1/N. Select the flag by
    # direction so both Julia cFFTWPlan directions remain unnormalized.
    normalize_inverse = plan.direction == "FFTW_FORWARD"
    return np.asarray(
        plan(values, normalise_idft=normalize_inverse)
    )


def _fft_with_plan(
    values: np.ndarray, plan: Any, *, scaled: bool
) -> np.ndarray:
    """Apply Julia's bare- versus scaled-FFTW plan dispatch families."""

    if not scaled:
        if isinstance(plan, _FFTWComplexPlan):
            if plan.scaled:
                raise TypeError("ft must be a bare pyFFTW FFTW plan")
            return plan(values)
        if isinstance(plan, pyfftw.FFTW):
            # A Julia cFFTWPlan may be forward or backward. Applying the bare
            # plan never adds inverse normalization.
            return _apply_bare_pyfftw_plan(plan, values)
        raise TypeError("ft must be a bare pyFFTW FFTW plan")

    if isinstance(plan, _FFTWComplexPlan):
        if not plan.scaled:
            raise TypeError("ift must be a scaled pyFFTW FFTW plan")
        return plan(values)
    if isinstance(plan, ScaledFFTWPlan):
        return plan(values)
    if (
        isinstance(plan, pyfftw.FFTW)
        and plan.direction == "FFTW_BACKWARD"
        and plan.normalise_idft
    ):
        # pyFFTW represents its normalized inverse as the same concrete plan
        # class. Treat that common form as AbstractFFTs.plan_ifft.
        return np.asarray(plan(values, normalise_idft=True))
    raise TypeError("ift must be a scaled pyFFTW FFTW plan")


def _fftw_thread_count() -> int:
    """Return the requested FFTW worker count, defaulting to Julia parity."""

    value = os.environ.get("SLMTOOLS_FFT_THREADS", "1")
    try:
        threads = int(value)
    except ValueError as error:
        raise ValueError("SLMTOOLS_FFT_THREADS must be a positive integer") from error
    if threads < 1:
        raise ValueError("SLMTOOLS_FFT_THREADS must be a positive integer")
    return threads


class _FFTWComplexPlan:
    """Column-major FFTW plan with an opt-in worker-count override.

    One thread remains the library default for strict Julia-port parity.
    Notebook applications can set ``SLMTOOLS_FFT_THREADS`` before constructing
    a solver to use the threaded FFTW backend shipped by pyFFTW.
    """

    def __init__(self, shape: tuple[int, ...], *, inverse: bool) -> None:
        self.direction = "FFTW_BACKWARD" if inverse else "FFTW_FORWARD"
        self.scaled = inverse
        self._input = pyfftw.empty_aligned(
            shape, dtype=np.complex128, order="F"
        )
        self._output = pyfftw.empty_aligned(
            shape, dtype=np.complex128, order="F"
        )
        self._plan = pyfftw.FFTW(
            self._input,
            self._output,
            axes=tuple(range(len(shape))),
            direction=self.direction,
            flags=("FFTW_ESTIMATE",),
            threads=_fftw_thread_count(),
            normalise_idft=True,
        )

    def __call__(self, values: Any) -> np.ndarray:
        self._input[...] = np.asarray(values, dtype=np.complex128)
        self._plan()
        return self._output.copy(order="F")


class ScaledFFTWPlan:
    """Python counterpart of an ``AbstractFFTs.ScaledPlan``.

    Julia permits any bare complex FFTW plan, including a forward plan, to be
    multiplied by a scalar and passed as ``gsIter``/``pdgsIter``'s scaled plan.
    pyFFTW has no separate scaled-plan type, so this small explicit wrapper
    preserves that successful plan family without accepting arbitrary
    callables or operator protocols.
    """

    def __init__(self, plan: Any, scale: Any) -> None:
        if not isinstance(plan, (pyfftw.FFTW, _FFTWComplexPlan)):
            raise TypeError("ScaledFFTWPlan requires a pyFFTW FFTW plan")
        if not np.isscalar(scale) or isinstance(scale, (str, bytes)):
            raise TypeError("ScaledFFTWPlan scale must be numeric")
        self.plan = plan
        self.scale = scale

    def __call__(self, values: Any) -> np.ndarray:
        if isinstance(self.plan, pyfftw.FFTW):
            transformed = _apply_bare_pyfftw_plan(self.plan, values)
        else:
            transformed = self.plan(values)
        return np.asarray(transformed) * self.scale


def _fftw_plan_pair(shape: tuple[int, ...]) -> tuple[_FFTWComplexPlan, _FFTWComplexPlan]:
    return (
        _FFTWComplexPlan(shape, inverse=False),
        _FFTWComplexPlan(shape, inverse=True),
    )


def gsIter(
    guess: Any,
    u: Any,
    v: Any,
    ft: Any,
    ift: Any,
) -> np.ndarray:
    """Perform one Gerchberg--Saxton iteration on unshifted arrays.

    ``ft`` must be a bare pyFFTW plan and ``ift`` a scaled plan, matching
    Julia's concrete ``cFFTWPlan``/``ScaledPlan`` dispatch. Transform
    direction is deliberately unrestricted. Use :class:`ScaledFFTWPlan` for
    scaled plan families that pyFFTW does not represent directly.
    """

    guess_array = _require_dense_ndarray(guess, "gsIter guess")
    u_array = _require_dense_ndarray(u, "gsIter u")
    v_array = _require_dense_ndarray(v, "gsIter v")
    if guess_array.dtype != np.dtype(np.complex128):
        raise TypeError("guess must have Julia ComplexF64 element type")
    if u_array.dtype != np.dtype(np.float64) or v_array.dtype != np.dtype(
        np.float64
    ):
        raise TypeError("u and v must have Julia Float64 element type")
    if guess_array.shape != u_array.shape or guess_array.shape != v_array.shape:
        raise ValueError("guess, u, and v must have the same shape")
    transformed = _fft_with_plan(guess_array, ft, scaled=False)
    update = _fft_with_plan(
        _phasor(transformed) * v_array, ift, scaled=True
    )
    return np.asarray(u_array * _phasor(update), dtype=np.complex128)


def gs(
    U: LatticeField,
    V: LatticeField,
    nit: int,
    phi0: Any = _OMITTED,
) -> LatticeField:
    """Run Gerchberg--Saxton and return a ``ComplexPhase`` field.

    Inputs must be a homogeneous pair of modulus fields or intensity fields.
    The audited Julia Intensity overload forwards an omitted/``nothing``
    initial phase to a nonexistent four-argument Modulus overload. That
    upstream failure is deliberately retained: Intensity inputs require an
    explicit phase. Modulus inputs retain Julia's working random initializer.
    """

    iterations = _iterations(nit)
    intensity_inputs = _tag_is(U, Intensity) and _tag_is(V, Intensity)
    u_field, v_field = _homogeneous_modulus_pair(
        U, V, intensity_supported=True, function_name="gs"
    )
    _validate_dual(u_field, v_field)
    if np.shape(u_field.data) != np.shape(v_field.data):
        raise ValueError("input and target must have the same shape")
    if phi0 is _OMITTED:
        if intensity_inputs:
            raise TypeError(
                "the Julia Intensity gs overload forwards nothing to a "
                "nonexistent four-argument Modulus method; supply phi0"
            )
        random = np.random.random(np.shape(u_field.data))
        phi0 = _make_field(RealPhase, random, u_field.L, u_field.flambda)
    if not _is_phase(phi0):
        raise TypeError("phi0 must be a phase LatticeField")
    _validate_equal(u_field, phi0)
    if _tag_is(phi0, ComplexPhase) and np.asarray(phi0.data).dtype != np.dtype(
        np.complex128
    ):
        # Julia calls phasor(::ComplexF64) on the wrapped initial samples
        # before entering the loop, so a full-typed ComplexF32 phase fails even
        # when nit is zero.
        raise TypeError("ComplexPhase phi0 must have Julia ComplexF64 storage")

    u_data = np.asarray(u_field.data)
    v_data = np.asarray(v_field.data)
    if _has_arbitrary_precision_modulus(
        u_data
    ) or _has_arbitrary_precision_modulus(v_data):
        raise TypeError(
            "Julia's gs path reaches unsupported phasor(Complex{BigFloat})"
        )
    guess = np.fft.ifftshift(u_data * _phasor(_phase_array(phi0)))
    if iterations > 0:
        if u_data.dtype == np.dtype(np.float64) and v_data.dtype == np.dtype(
            np.float64
        ):
            u_work = u_data
            v_work = v_data
        else:
            raise TypeError(
                "positive-iteration gs has no matching Julia gsIter method "
                "for this Modulus element type"
            )
    else:
        u_work = u_data
        v_work = v_data
    ushift = np.fft.ifftshift(u_work)
    vshift = np.fft.ifftshift(v_work)
    ft, ift = _fftw_plan_pair(tuple(guess.shape))
    for _ in range(iterations):
        guess = gsIter(guess, ushift, vshift, ft, ift)
    return _make_field(ComplexPhase, np.fft.fftshift(_phasor(guess)), u_field.L, u_field.flambda)


def gsLog(
    U: LatticeField,
    V: LatticeField,
    nit: int,
    phi0: Any = _OMITTED,
    *,
    every: int = 1,
) -> tuple[LatticeField, list[float]]:
    """Run GS and record the squared normalized modulus error.

    The first sample is recorded after iteration one and subsequent samples
    occur at Julia's ``(iteration - 1) % every == 0`` cadence. As in the
    audited Julia source, the Intensity overload is unusable without an
    explicit phase because it forwards ``nothing`` to a nonexistent Modulus
    method.
    """

    iterations = _iterations(nit)
    cadence = _int64_keyword(every, "every")
    intensity_inputs = _tag_is(U, Intensity) and _tag_is(V, Intensity)
    u_field, v_field = _homogeneous_modulus_pair(
        U, V, intensity_supported=True, function_name="gsLog"
    )
    _validate_dual(u_field, v_field)
    if np.shape(u_field.data) != np.shape(v_field.data):
        raise ValueError("input and target must have the same shape")
    if phi0 is _OMITTED:
        if intensity_inputs:
            raise TypeError(
                "the Julia Intensity gsLog overload forwards nothing to a "
                "nonexistent four-argument Modulus method; supply phi0"
            )
        random = np.random.random(np.shape(u_field.data))
        phi0 = _make_field(RealPhase, random, u_field.L, u_field.flambda)
    if not _is_phase(phi0):
        raise TypeError("phi0 must be a phase LatticeField")
    _validate_equal(u_field, phi0)
    if _tag_is(phi0, ComplexPhase) and np.asarray(phi0.data).dtype != np.dtype(
        np.complex128
    ):
        raise TypeError("ComplexPhase phi0 must have Julia ComplexF64 storage")

    # Unlike ``gs``, this implementation does not dispatch through the
    # Float64-only ``gsIter`` helper in Julia.  Its normalized modulus arrays
    # retain their input precision (including Float16), while the FFT work is
    # ComplexF64.  Keeping the reduction scalar in its NumPy dtype reproduces
    # those component-wise low-precision divisions and their logged error.
    u_data = np.asarray(u_field.data)
    v_data = np.asarray(v_field.data)
    if _has_arbitrary_precision_modulus(
        u_data
    ) or _has_arbitrary_precision_modulus(v_data):
        raise TypeError(
            "Julia's gsLog path reaches unsupported "
            "phasor(Complex{BigFloat})"
        )
    rational_u = u_data.dtype.kind == "O" and all(
        isinstance(value, Fraction) for value in u_data.flat
    )
    rational_v = v_data.dtype.kind == "O" and all(
        isinstance(value, Fraction) for value in v_data.flat
    )
    u_squared = _julia_array_array_operation(u_data, u_data, np.multiply)
    v_squared = _julia_array_array_operation(v_data, v_data, np.multiply)
    u_total = _julia_sum(u_squared)
    v_total = _julia_sum(v_squared)
    u_norm = (
        np.sqrt(float(u_total)) if rational_u else np.sqrt(u_total)
    )
    v_norm = (
        np.sqrt(float(v_total)) if rational_v else np.sqrt(v_total)
    )
    # The ComplexF64 FFT plan converts Rational work arrays only after their
    # exact, checked normalization reductions have completed.
    if rational_u:
        u_data = np.asarray(u_data, dtype=np.float64)
    if rational_v:
        v_data = np.asarray(v_data, dtype=np.float64)
    guess = np.fft.ifftshift(u_data * _phasor(_phase_array(phi0)))
    # Julia applies the divisions directly. Zero-norm inputs therefore remain
    # valid method calls and propagate IEEE NaN/Inf through later iterations
    # rather than raising a defensive domain error.
    with np.errstate(divide="ignore", invalid="ignore"):
        ushift = np.fft.ifftshift(u_data) / u_norm
        vshift = np.fft.ifftshift(v_data) / v_norm
    errors: list[float] = []
    ft, ift = _fftw_plan_pair(tuple(guess.shape))
    for iteration in range(iterations):
        update = ift(_phasor(ft(guess)) * vshift)
        if iteration % cadence == 0:
            update_norm = float(
                np.sqrt(_julia_sum(np.abs(update) ** 2))
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized = update / update_norm
            errors.append(
                float(
                    _julia_sum(
                        (np.abs(ushift) - np.abs(normalized)) ** 2
                    )
                )
            )
        guess = ushift * _phasor(update)
    phase = _make_field(ComplexPhase, np.fft.fftshift(_phasor(guess)), u_field.L, u_field.flambda)
    return phase, errors


def _normalize_modulus(values: Any) -> np.ndarray:
    array = np.asarray(values)
    magnitudes = _julia_abs(array)
    squared = _julia_array_array_operation(
        magnitudes, magnitudes, np.multiply
    )
    total = _julia_sum(squared)
    if isinstance(total, Fraction):
        norm = np.sqrt(float(total))
    elif isinstance(total, (_MPFR, _MPQ, _MPZ)):
        norm = _mpfr_sqrt(total)
    else:
        norm = np.sqrt(total)
    return _julia_array_scalar_operation(array, norm, np.divide)


def gsError(U: LatticeField, V: LatticeField, phase: LatticeField) -> float:
    """Return the normalized squared L2 beam-reshaping error."""

    u_field, v_field = _homogeneous_modulus_pair(
        U, V, intensity_supported=True, function_name="gsError"
    )
    _validate_dual(u_field, v_field)
    _validate_equal(u_field, phase)
    reconstruction = np.abs(_sft_array(np.asarray(u_field.data) * _phase_array(phase)))
    difference = _normalize_modulus(reconstruction) - _normalize_modulus(v_field.data)
    return float(_julia_sum(difference**2))


def _prepare_pdgs(
    imgs: Iterable[LatticeField],
    div_phases: Iterable[LatticeField],
    beam_guess: LatticeField,
    *,
    intensity_supported: bool,
    real_moduli_required: bool,
    function_name: str,
) -> tuple[tuple[LatticeField, ...], tuple[LatticeField, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if not isinstance(imgs, tuple) or not isinstance(div_phases, tuple):
        raise TypeError("imgs and divPhases must be NTuples")
    images = imgs
    phases = div_phases
    if not images or len(images) != len(phases):
        raise ValueError("imgs and divPhases must be nonempty and have equal length")
    if not _tag_is(beam_guess, ComplexAmp):
        raise TypeError("beamGuess must be a ComplexAmp LatticeField")
    if not _is_complex_numeric_data(beam_guess.data):
        raise TypeError("beamGuess must have a concrete complex element type")
    image_tags = tuple(getattr(image, "field_type", None) for image in images)
    if all(tag is Modulus for tag in image_tags):
        modulus_images = images
    elif intensity_supported and all(tag is Intensity for tag in image_tags):
        modulus_images = tuple(_modulus_field(image) for image in images)
    else:
        accepted = "all Modulus or all Intensity" if intensity_supported else "all Modulus"
        raise TypeError(
            f"{function_name} images must be homogeneous: {accepted} LatticeFields"
        )
    # Julia's iterative PDGS methods admit ``<:Number`` at the outer boundary,
    # but their internal tuple assertion requires Float64 modulus arrays.
    # ``pdgsError`` has no such assertion and legitimately accepts complex
    # Modulus data, so only callers that request it enforce real arrays here.
    if real_moduli_required and any(
        not _is_real_numeric_data(image.data) for image in modulus_images
    ):
        raise TypeError(f"{function_name} images must have real numeric data")
    image_shape = np.shape(modulus_images[0].data)
    for image, phase in zip(modulus_images, phases, strict=True):
        if np.shape(image.data) != image_shape or np.shape(phase.data) != np.shape(beam_guess.data):
            raise ValueError("Input size mismatch.")
        if not _is_phase(phase):
            raise TypeError("all diversity fields must be phases")
        _validate_equal(phase, beam_guess)
        _validate_dual(image, beam_guess)
    if image_shape != np.shape(beam_guess.data):
        raise ValueError("Input size mismatch.")

    phis: list[np.ndarray] = []
    mods: list[np.ndarray] = []
    for image, phase in zip(modulus_images, phases, strict=True):
        dual_phase, dual_lattice = _dual_phase_array(image.L, image.flambda)
        if not _same_lattice(dual_lattice, beam_guess.L):
            raise ValueError("Non-dual lattices.")
        phis.append(np.fft.ifftshift(_phase_array(phase) * np.exp(2j * np.pi * dual_phase)))
        modulus_data = np.asarray(image.data)
        if real_moduli_required:
            # Julia spells this ``i.data * 1.0`` and subsequently asserts a
            # tuple of ``Array{Float64}``.  That promotes Float16/Float32 and
            # Rational inputs, but deliberately does not narrow BigFloat.
            modulus_data = _julia_array_scalar_operation(
                modulus_data, np.float64(1.0), np.multiply
            )
        mods.append(np.fft.ifftshift(modulus_data))
    if real_moduli_required:
        if any(np.asarray(phi).dtype != np.dtype(np.complex128) for phi in phis):
            raise TypeError(
                f"{function_name} diversity work arrays must have Julia "
                "ComplexF64 element type"
            )
        if any(np.asarray(modulus).dtype != np.dtype(np.float64) for modulus in mods):
            raise TypeError(
                f"{function_name} modulus work arrays must have Julia "
                "Float64 element type"
            )
    return modulus_images, phases, tuple(phis), tuple(mods)


def pdgsIter(
    guess: Any,
    phis: Sequence[Any],
    mods: Sequence[Any],
    ft: Any,
    ift: Any,
) -> np.ndarray:
    """Perform one phase-diversity GS iteration on unshifted arrays."""

    if not isinstance(phis, tuple) or not isinstance(mods, tuple):
        raise TypeError("pdgsIter phases and moduli must be tuples")
    if not phis or len(phis) != len(mods):
        raise ValueError("phis and mods must be nonempty and have equal length")
    guess_array = _require_dense_ndarray(guess, "pdgsIter guess")
    if guess_array.dtype != np.dtype(np.complex128):
        raise TypeError("pdgsIter guess must have Julia ComplexF64 element type")
    updates: list[np.ndarray] = []
    for phi, modulus in zip(phis, mods, strict=True):
        phi_array = _require_dense_ndarray(phi, "pdgsIter phase")
        modulus_array = _require_dense_ndarray(modulus, "pdgsIter modulus")
        if phi_array.dtype != np.dtype(np.complex128):
            raise TypeError("pdgsIter phases must have Julia ComplexF64 element type")
        if modulus_array.dtype != np.dtype(np.float64):
            raise TypeError("pdgsIter moduli must have Julia Float64 element type")
        if phi_array.shape != guess_array.shape or modulus_array.shape != guess_array.shape:
            raise ValueError("guess, phases, and moduli must have the same shape")
        propagated = _fft_with_plan(
            guess_array * phi_array, ft, scaled=False
        )
        update = _fft_with_plan(
            modulus_array * _phasor(propagated), ift, scaled=True
        )
        updates.append(update * np.conjugate(phi_array))
    return np.asarray(_julia_add_sum(tuple(updates))) / len(phis)


def pdgs(
    imgs: Iterable[LatticeField],
    divPhases: Iterable[LatticeField],
    nit: int,
    beamGuess: LatticeField,
) -> LatticeField:
    """Estimate a complex beam from phase-diverse modulus/intensity images."""

    iterations = _iterations(nit)
    images, phases, phis, mods = _prepare_pdgs(
        imgs,
        divPhases,
        beamGuess,
        intensity_supported=True,
        real_moduli_required=True,
        function_name="pdgs",
    )
    if iterations > 0 and np.asarray(beamGuess.data).dtype != np.dtype(
        np.complex128
    ):
        raise TypeError(
            "positive-iteration pdgs has no matching ComplexF64 pdgsIter method"
        )
    guess = np.fft.ifftshift(np.asarray(beamGuess.data))
    ft, ift = _fftw_plan_pair(tuple(guess.shape))
    for _ in range(iterations):
        guess = pdgsIter(guess, phis, mods, ft, ift)
    return _make_field(ComplexAmp, np.fft.fftshift(guess), beamGuess.L, beamGuess.flambda)


def pdgsLog(
    imgs: Iterable[LatticeField],
    divPhases: Iterable[LatticeField],
    nit: int,
    beamGuess: LatticeField,
    *,
    every: int = 1,
) -> tuple[LatticeField, list[float]]:
    """Run PDGS and log the branch-spread convergence metric."""

    iterations = _iterations(nit)
    cadence = _int64_keyword(every, "every")
    images, phases, phis, mods = _prepare_pdgs(
        imgs,
        divPhases,
        beamGuess,
        intensity_supported=False,
        real_moduli_required=True,
        function_name="pdgsLog",
    )
    guess = np.fft.ifftshift(np.asarray(beamGuess.data))
    errors: list[float] = []
    ft, ift = _fftw_plan_pair(tuple(guess.shape))
    for iteration in range(iterations):
        updates = tuple(
            ift(modulus * _phasor(ft(guess * phi))) * np.conjugate(phi)
            for phi, modulus in zip(phis, mods, strict=True)
        )
        guess = _julia_add_sum(updates) / len(updates)
        if iteration % cadence == 0:
            spread = sum(
                float(_julia_sum(np.abs(update - guess) ** 2))
                for update in updates
            )
            errors.append(float(np.sqrt(spread) / len(updates)))
    result = _make_field(ComplexAmp, np.fft.fftshift(guess), beamGuess.L, beamGuess.flambda)
    return result, errors


def pdgsError(
    divMods: Iterable[LatticeField],
    divPhases: Iterable[LatticeField],
    beamGuess: LatticeField,
) -> Any:
    """Return the average normalized squared L2 error over diversity images."""

    images, phases, _, _ = _prepare_pdgs(
        divMods,
        divPhases,
        beamGuess,
        intensity_supported=False,
        real_moduli_required=False,
        function_name="pdgsError",
    )
    errors = []
    beam = np.asarray(beamGuess.data)
    for image, phase in zip(images, phases, strict=True):
        target = _normalize_modulus(image.data)
        # Multiplying the two fields in Julia routes through the partial
        # ComplexAmplitude constructor, which converts its work array to
        # ComplexF64 before FFTW sees it.
        propagated = np.asarray(
            _julia_array_array_operation(
                beam, _phase_array(phase), np.multiply
            ),
            dtype=np.complex128,
        )
        reconstruction = _normalize_modulus(_sft_array(propagated))
        difference = _julia_array_array_operation(
            _julia_abs(target),
            _julia_abs(reconstruction),
            np.subtract,
        )
        errors.append(
            _julia_sum(
                _julia_array_array_operation(
                    difference, difference, np.multiply
                )
            )
        )
    total = _julia_add_sum(tuple(errors))
    return _julia_array_scalar_operation(
        total, len(errors), np.divide
    ).reshape(())[()]


def oneShot(img: LatticeField, alpha: float, beta: Sequence[float]) -> LatticeField:
    """Recover a beam from one known quadratic/linear diversity image."""

    if not _tag_is(img, Intensity):
        raise TypeError("img must be an Intensity LatticeField")
    _require_real_field_data(img, "img")
    if not _is_real_number(alpha):
        raise TypeError("alpha must be real")
    if not isinstance(beta, tuple):
        raise TypeError("beta must be an NTuple of real coefficients")
    if not all(_is_real_number(value) for value in beta):
        raise TypeError("beta values must be real")
    if len(beta) != np.ndim(img.data):
        raise ValueError("beta length must equal image dimensionality")
    if np.any(np.asarray(img.data) < 0):
        raise ValueError("intensity values must be nonnegative")
    lattice = _as_lattice(img.L)
    flambda = img.flambda
    dual_lattice = _dual_shift_lattice(lattice, flambda)
    # Julia broadcasts over the heterogeneous NTuple element by element.
    # Coercing ``beta`` to one NumPy array here changes Int64/Float32 products
    # to Float64 before either ldot call sees them.
    center = tuple(
        _scalar_operation(value, flambda, np.multiply) for value in beta
    )
    diversity = _quadratic_linear_phase(dual_lattice, alpha, beta)

    # Julia's ordinary `/` permits a zero real alpha and propagates Inf/NaN
    # through the phase and FFT calculation.  Keep that numerical result
    # rather than adding a defensive domain restriction.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        radial = _julia_array_array_operation(
            _lattice_r2(lattice),
            _julia_array_scalar_operation(
                _lattice_ldot(lattice, center), 2, np.multiply
            ),
            np.subtract,
        )
        center_squared = tuple(
            _literal_square(value) for value in center
        )
        center_squared_sum = _julia_add_sum(center_squared)
        radial = _julia_array_scalar_operation(
            radial, center_squared_sum, np.add
        )
        flambda_squared = _literal_square(flambda)
        denominator = _scalar_operation(
            _scalar_operation(2, alpha, np.multiply),
            flambda_squared,
            np.multiply,
        )
        coefficient = _scalar_operation(-1, denominator, np.divide)
        dual_diversity = _julia_array_scalar_operation(
            radial, coefficient, np.multiply
        )

        div_phase = _make_field(RealPhase, diversity, dual_lattice, flambda)
        dual_div_phase = _make_field(
            RealPhase, dual_diversity, lattice, flambda
        )
        camera_field = img.sqrt() * dual_div_phase
        return _field_isft(camera_field) * div_phase.conj()


def mraf(
    U: LatticeField,
    V: LatticeField,
    nit: int,
    phi0: LatticeField,
    roi: Any,
    m: float,
) -> LatticeField:
    """Run mixed-region amplitude freedom (MRAF).

    ``roi`` uses normal NumPy indexing (typically a tuple of slices).  The
    weighting ``m`` is intentionally not clamped, matching the Julia API.
    """

    if not _is_real_number(m):
        raise TypeError("m must be real")
    iterations = _iterations(nit)
    u_field, v_field = _homogeneous_modulus_pair(
        U, V, intensity_supported=False, function_name="mraf"
    )
    _validate_dual(u_field, v_field)
    _validate_equal(u_field, phi0)
    u = _normalize_modulus(u_field.data)
    v = _normalize_modulus(v_field.data)
    guess = u * _phase_array(phi0)
    compensation = float(
        np.sqrt(_julia_sum(np.abs(_sft_array(guess)) ** 2))
    )
    for _ in range(iterations):
        output = _sft_array(guess)
        target = _phasor(output[roi]) * v[roi] * m
        with np.errstate(divide="ignore", invalid="ignore"):
            output *= (1 - m) / compensation
        output[roi] = target
        guess = _phasor(_isft_array(output)) * u
    return _make_field(ComplexPhase, _phasor(guess), u_field.L, u_field.flambda)
