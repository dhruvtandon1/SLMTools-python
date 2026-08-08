"""Interpolation, block coarsening, and lattice resampling.

The default interpolator is a dependency-free tensor product of natural cubic
splines, matching Interpolations.jl's ``Cubic(Line(OnGrid()))`` construction.
The ``bc`` argument controls the cubic endpoint equations, while
``extrapolation_bc`` selects numeric fill, ``Flat``, ``Periodic``, ``Linear``,
or throwing behavior outside the source grid.

The range/lattice upsampler retains Julia's unusual but successful empty
``StepRangeLen`` results for zero and negative factors. Downsampling and block
coarsening keep focused errors for their corresponding failing source paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
from numbers import Integral, Number
from operator import index as integer_index
from typing import Any, Callable

import gmpy2
import numpy as np

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
    DimensionMismatch,
    DomainError,
    FieldVal,
    Lattice,
    LatticeAxis,
    LatticeField,
    _axis,
    _julia_collect_comprehension_results,
    _julia_literal_array,
    _julia_array_scalar_operation,
    _julia_sum,
    _julia_typed_zero,
    _logical_axis_scalar_operation,
    _with_axis_length_kind,
    _require_julia_numeric_array,
    as_lattice,
)
from .lattice_utils import _looks_like_lattice, _step
from .misc import (
    _enable_decimal_nonfinite,
    _fraction_int64_divide,
)


@dataclass(frozen=True)
class OnGrid:
    """Place a spline boundary condition on the endpoint samples."""


@dataclass(frozen=True)
class OnCell:
    """Place a spline boundary condition half a cell beyond the samples."""


_GRID_TYPES = (OnGrid, OnCell)


@dataclass(frozen=True)
class Flat:
    """Clamp extrapolation, or impose a placed flat spline boundary."""

    grid: Any | None = None

    def __post_init__(self) -> None:
        if self.grid is not None and not isinstance(self.grid, _GRID_TYPES):
            raise TypeError("Flat placement must be OnGrid() or OnCell().")


@dataclass(frozen=True)
class Periodic:
    """Wrap extrapolation, or impose a placed periodic spline boundary."""

    grid: Any | None = None

    def __post_init__(self) -> None:
        if self.grid is not None and not isinstance(self.grid, _GRID_TYPES):
            raise TypeError("Periodic placement must be OnGrid() or OnCell().")


@dataclass(frozen=True)
class Throw:
    """Raise ``IndexError`` for out-of-bounds coordinates."""


@dataclass(frozen=True)
class Linear:
    """Marker for piecewise-linear interpolation.

    This represents ``Interpolations.Linear()``, which is an interpolation
    degree, not the distinct ``Interpolations.Line()`` boundary condition.
    Interpolations.jl 0.16.2 nevertheless accepts ``Linear()`` as an
    extrapolation flag for backwards compatibility and translates it to
    endpoint-tangent (``Line()``) extrapolation.  The evaluator below mirrors
    that translation because SLMTools' OT code relies on it.
    """

    bc: Any | None = None

    def __post_init__(self) -> None:
        # Interpolations.Linear(Periodic()) is a public constructor used to
        # request periodic linear B-splines.  SLMTools only needs the marker
        # as an extrapolation flag, but retaining the constructor surface is
        # important for qualified API compatibility.
        if self.bc is not None and not isinstance(self.bc, Periodic):
            raise TypeError("An explicit Linear boundary must be Periodic().")


_DEFAULT_CUBIC_BC = object()


def _julia_numeric_dtype(*items: Any) -> np.dtype[Any]:
    """Approximate Julia's numeric promotion for interpolation arithmetic.

    Julia promotes an integer together with ``Float32`` to ``Float32`` while
    NumPy promotes that pair to ``Float64``.  Interpolations.jl performs its
    spline arithmetic with Julia promotion, and uses ``Float64`` coefficients
    only when every input is integer-valued.
    """

    dtypes = [np.dtype(np.asarray(item).dtype) for item in items]
    complex_dtypes = [dtype for dtype in dtypes if dtype.kind == "c"]
    real_dtypes = [dtype for dtype in dtypes if dtype.kind == "f"]
    if complex_dtypes:
        real_sizes = [dtype.itemsize // 2 for dtype in complex_dtypes]
        real_sizes.extend(dtype.itemsize for dtype in real_dtypes)
        return np.dtype(np.complex64 if max(real_sizes) <= 4 else np.complex128)
    if real_dtypes:
        size = max(item.itemsize for item in real_dtypes)
        if size <= 2:
            return np.dtype(np.float16)
        return np.dtype(np.float32 if size <= 4 else np.float64)
    return np.dtype(np.float64)


def _object_values(item: Any) -> list[Any]:
    """Return nonempty scalar values from an object-typed numeric container."""

    array = np.asarray(item)
    if array.dtype != np.dtype(object):
        return []
    return [value for value in array.flat if value is not None]


def _julia_differences(values: Any) -> np.ndarray:
    """Form adjacent differences in the represented exact context."""

    array = np.asarray(values)
    if array.dtype.kind == "O" and any(
        isinstance(item, (_MPFR, _MPC, _MPFRComplex, _MPQ, _MPZ))
        for item in array.flat
    ):
        output = np.empty(max(0, len(array) - 1), dtype=object)
        with _bigfloat_context():
            for index in range(len(output)):
                output[index] = array[index + 1] - array[index]
        return output
    return np.diff(array)


def _exact_domain(values: Any, target: Any) -> type[Any] | None:
    """Select Python's analogue of Julia Rational/BigFloat promotion."""

    value_items = _object_values(values)
    target_items = _object_values(target)
    all_items = value_items + target_items
    if any(isinstance(item, _MPFRComplex) for item in value_items):
        return _MPFRComplex
    if any(isinstance(item, _MPC) for item in value_items):
        return _MPC
    if any(isinstance(item, _MPFR) for item in all_items):
        return _MPFR
    if any(isinstance(item, (gmpy2.mpz, _MPQ)) for item in value_items):
        # Interpolations.jl promotes BigInt/Rational{BigInt} coefficients to
        # the process-default BigFloat domain rather than retaining rationals.
        return _MPFR
    if any(isinstance(item, Decimal) for item in all_items):
        return Decimal
    if any(isinstance(item, Fraction) for item in value_items):
        target_array = np.asarray(target)
        target_is_exact = (
            all(
                isinstance(item, (Integral, Fraction))
                for item in target_items
            )
            if target_array.dtype == np.dtype(object)
            else target_array.dtype.kind in "bui"
        )
        if target_is_exact:
            return Fraction
    return None


def _convert_exact(value: Any, domain: type[Any]) -> Any:
    """Convert a scalar without silently passing through binary Float64."""

    if domain is _MPFRComplex:
        if isinstance(value, _MPFRComplex):
            return value
        if isinstance(value, _MPC):
            return _MPFRComplex(value.real, value.imag)
        if isinstance(value, (complex, np.complexfloating)):
            return _MPFRComplex(value.real, value.imag)
        return _MPFRComplex(value)
    if domain is _MPC:
        if isinstance(value, (_MPC, _MPFRComplex, complex, np.complexfloating)):
            return _MPC(_to_mpfr(value.real), _to_mpfr(value.imag))
        return _MPC(_to_mpfr(value), _to_mpfr(0))
    if domain is _MPFR:
        return _to_mpfr(value)
    if domain is _MPQ:
        if isinstance(value, _MPQ):
            return value
        if isinstance(value, Fraction):
            return _MPQ(value.numerator, value.denominator)
        return _MPQ(value)
    if domain is Fraction:
        if isinstance(value, Fraction):
            return value
        if isinstance(value, Decimal):
            return Fraction(value)
        return Fraction(value)
    if isinstance(value, Decimal):
        return value
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    if isinstance(value, (float, np.floating)):
        # Julia BigFloat(Float64) preserves the represented binary value.
        return Decimal.from_float(float(value))
    return Decimal(int(value)) if isinstance(value, Integral) else Decimal(value)


def _exact_coordinate_domain(domain: type[Any]) -> type[Any]:
    """Return the real coordinate type paired with an exact value domain."""

    return _MPFR if domain in (_MPC, _MPFRComplex) else domain


def _as_exact_array(array: Any, domain: type[Any]) -> np.ndarray:
    values = np.asarray(array, dtype=object)
    converted = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        converted[index] = _convert_exact(values[index], domain)
    return converted


def _periodic_coordinates(
    target: np.ndarray, lower: Any, upper: Any, dtype: np.dtype[Any]
) -> np.ndarray:
    """Apply Julia's floor-based ``mod`` to machine and exact coordinates."""

    if dtype != np.dtype(object):
        return (
            np.mod(target - lower, upper - lower) + lower
        ).astype(dtype, copy=False)
    period = upper - lower
    mapped = np.empty(target.shape, dtype=object)
    for index in np.ndindex(target.shape):
        displacement = target[index] - lower
        quotient = displacement / period
        if isinstance(quotient, Decimal):
            turns = quotient.to_integral_value(rounding=ROUND_FLOOR)
        else:
            turns = quotient.__floor__()
        mapped[index] = displacement - turns * period + lower
    return mapped


def _interpolation_coefficient_dtype(values: Any) -> np.dtype[Any]:
    """Return Interpolations.jl's coefficient type for a numeric array."""

    dtype = np.asarray(values).dtype
    # Interpolations.tcoef converts all built-in integer arrays to Float64.
    # Query coordinates must not narrow those coefficients back to Float16 or
    # Float32 during evaluation.
    if dtype.kind in "bui":
        return np.dtype(np.float64)
    return dtype


def _interpolation_output_dtype(values: Any, target: Any) -> np.dtype[Any]:
    """Return the coefficient/weight product type used for evaluation."""

    coefficient_dtype = _interpolation_coefficient_dtype(values)
    if coefficient_dtype == np.dtype(object):
        if _exact_domain(values, target) is not None:
            return np.dtype(object)
        target_dtype = np.asarray(target).dtype
        if target_dtype.kind in "fc":
            return target_dtype
        return np.dtype(np.float64)
    return _julia_numeric_dtype(
        np.empty((), dtype=coefficient_dtype), np.asarray(target)
    )


def _coordinate_dtype(source: Any, target: Any) -> np.dtype[Any]:
    """Return the real working dtype for source/target coordinate arithmetic."""

    if any(np.asarray(item).dtype == np.dtype(object) for item in (source, target)):
        return np.dtype(object)
    dtype = _julia_numeric_dtype(source, target)
    if dtype.kind == "c":
        return np.dtype(np.float32 if dtype.itemsize == 8 else np.float64)
    return dtype


def _logical_step(axis: Any) -> Any | None:
    """Return trusted logical range spacing, or ``None`` for irregular knots.

    A Float32 range at a large origin can materialize with alternating or even
    repeated differences.  Its retained step is nevertheless the coordinate
    system used by Interpolations.jl's scaled B-spline.  The absolute tolerance
    below accounts for subtraction cancellation at the magnitude of the
    origin while still rejecting ordinary nonuniform explicit vectors.
    """

    values = np.asarray(axis)
    hint = getattr(axis, "_step_hint", None)
    if hint is None:
        return None
    if hint <= 0:
        raise ValueError("Interpolation lattice axes must have positive steps.")
    if bool(getattr(axis, "_step_hint_is_logical", False)):
        # An immutable LatticeAxis made from a range retains the range's
        # logical step.  It remains authoritative even if a materialized
        # Float32 difference happens to equal that step at one index but not
        # at later indices.
        return hint
    if len(values) < 2:
        return hint
    differences = _julia_differences(values)
    if values.dtype == np.dtype(object):
        return hint if all(item == hint for item in differences) else None
    # LatticeAxis infers its hint from the first materialized difference when
    # handed an explicit vector.  Equality identifies that case; evaluating in
    # physical coordinates preserves Gridded(Linear()) rounding.  A true
    # logical range whose small step is distorted by a large origin retains a
    # distinct hint and must instead be evaluated in index coordinates.
    if hint == differences[0]:
        return None
    real_dtype = values.real.dtype
    if not np.issubdtype(real_dtype, np.inexact):
        real_dtype = np.dtype(np.float64)
    epsilon = np.finfo(real_dtype).eps
    magnitude = max(1.0, float(np.max(np.abs(values))), float(abs(hint)))
    if np.allclose(
        differences,
        hint,
        rtol=16 * epsilon,
        atol=2 * epsilon * magnitude,
    ):
        return hint
    return None


def _regular_step(axis: Any, *, require_positive: bool = True) -> Any:
    """Return a logical range step and reject irregular coordinate vectors."""

    values = np.asarray(axis)
    hint = getattr(axis, "_step_hint", None)
    if hint is None:
        hint = _julia_differences(values)[0]
    if hint == 0 or (require_positive and hint < 0):
        raise ValueError("Interpolation lattice axes must have positive steps.")
    if bool(getattr(axis, "_step_hint_is_logical", False)):
        return hint
    differences = _julia_differences(values)
    if values.dtype == np.dtype(object):
        if not all(item == hint for item in differences):
            raise ValueError("Interpolation lattice axes must be regularly spaced.")
        return hint
    real_dtype = values.real.dtype
    if not np.issubdtype(real_dtype, np.inexact):
        real_dtype = np.dtype(np.float64)
    epsilon = np.finfo(real_dtype).eps
    magnitude = max(1.0, float(np.max(np.abs(values))), float(abs(hint)))
    if not np.allclose(
        differences,
        hint,
        rtol=16 * epsilon,
        atol=2 * epsilon * magnitude,
    ):
        raise ValueError("Interpolation lattice axes must be regularly spaced.")
    return hint


def _regular_positive_step(axis: Any) -> Any:
    """Return a positive logical range step for interpolation constructors."""

    return _regular_step(axis, require_positive=True)


def _evaluation_axis(
    axis: LatticeAxis,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map physical coordinates to a range's logical interpolation indices."""

    coordinate_dtype = _coordinate_dtype(axis, target)
    target_work = np.asarray(target, dtype=coordinate_dtype)
    logical_step = _logical_step(axis)
    if logical_step is None:
        return np.asarray(axis, dtype=coordinate_dtype), target_work
    origin = np.asarray(axis[0], dtype=coordinate_dtype)[()]
    step = np.asarray(logical_step, dtype=coordinate_dtype)[()]
    source_work = np.arange(len(axis), dtype=coordinate_dtype)
    return source_work, (target_work - origin) / step


def _solve_symmetric_tridiagonal(
    off_diagonal: np.ndarray,
    diagonal: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Solve a symmetric tridiagonal system with the O(n) Thomas algorithm."""

    diagonal = np.array(diagonal, copy=True)
    rhs = np.array(right_hand_side, copy=True)
    for row in range(1, len(diagonal)):
        multiplier = off_diagonal[row - 1] / diagonal[row - 1]
        diagonal[row] -= multiplier * off_diagonal[row - 1]
        rhs[row] -= multiplier * rhs[row - 1]

    solution = np.empty_like(rhs)
    solution[-1] = rhs[-1] / diagonal[-1]
    for row in range(len(diagonal) - 2, -1, -1):
        solution[row] = (
            rhs[row] - off_diagonal[row] * solution[row + 1]
        ) / diagonal[row]
    return solution


def _solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Solve a nonsymmetric tridiagonal system with scalar object support."""

    lower = np.array(lower, copy=True)
    diagonal = np.array(diagonal, copy=True)
    upper = np.array(upper, copy=True)
    rhs = np.array(right_hand_side, copy=True)
    for row in range(1, len(diagonal)):
        multiplier = lower[row - 1] / diagonal[row - 1]
        diagonal[row] -= multiplier * upper[row - 1]
        rhs[row] -= multiplier * rhs[row - 1]
    solution = np.empty_like(rhs)
    solution[-1] = rhs[-1] / diagonal[-1]
    for row in range(len(diagonal) - 2, -1, -1):
        solution[row] = (
            rhs[row] - upper[row] * solution[row + 1]
        ) / diagonal[row]
    return solution


def _solve_cyclic_symmetric_tridiagonal(
    off_diagonal: np.ndarray,
    diagonal: np.ndarray,
    corner: Any,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Solve a symmetric cyclic tridiagonal system in O(n) storage/time."""

    n = len(diagonal)
    if n == 2:
        coupling = off_diagonal[0] + corner
        determinant = diagonal[0] * diagonal[1] - coupling * coupling
        solution = np.empty_like(right_hand_side)
        solution[0] = (
            diagonal[1] * right_hand_side[0]
            - coupling * right_hand_side[1]
        ) / determinant
        solution[1] = (
            diagonal[0] * right_hand_side[1]
            - coupling * right_hand_side[0]
        ) / determinant
        return solution

    gamma = -diagonal[0]
    reduced_diagonal = np.array(diagonal, copy=True)
    reduced_diagonal[0] -= gamma
    reduced_diagonal[-1] -= corner * corner / gamma
    solution = _solve_symmetric_tridiagonal(
        off_diagonal, reduced_diagonal, right_hand_side
    )
    correction_rhs = np.zeros((n, 1), dtype=right_hand_side.dtype)
    correction_rhs[0, 0] = gamma
    correction_rhs[-1, 0] = corner
    correction = _solve_symmetric_tridiagonal(
        off_diagonal, reduced_diagonal, correction_rhs
    )[:, 0]
    numerator = solution[0] + (corner / gamma) * solution[-1]
    denominator = 1 + correction[0] + (corner / gamma) * correction[-1]
    return solution - correction[:, None] * (numerator / denominator)[None, :]


def _cubic_second_derivatives(
    flat: np.ndarray,
    spacing: np.ndarray,
    boundary: str,
) -> np.ndarray:
    """Compute spline second derivatives for supported OnGrid boundaries."""

    n = flat.shape[0]
    second = np.zeros_like(flat)
    slopes = np.diff(flat, axis=0) / spacing[:, None]
    if boundary == "natural":
        if n > 2:
            diagonal = 2 * (spacing[:-1] + spacing[1:])
            rhs = 6 * (slopes[1:] - slopes[:-1])
            second[1:-1] = _solve_symmetric_tridiagonal(
                spacing[1:-1], diagonal, rhs
            )
        return second
    if boundary == "flat":
        diagonal = np.empty(n, dtype=spacing.dtype)
        diagonal[0] = 2 * spacing[0]
        diagonal[-1] = 2 * spacing[-1]
        if n > 2:
            diagonal[1:-1] = 2 * (spacing[:-1] + spacing[1:])
        rhs = np.empty_like(flat)
        rhs[0] = 6 * slopes[0]
        rhs[-1] = -6 * slopes[-1]
        if n > 2:
            rhs[1:-1] = 6 * (slopes[1:] - slopes[:-1])
        return _solve_symmetric_tridiagonal(spacing, diagonal, rhs)
    if boundary == "flat_oncell":
        # Flat(OnCell()) imposes zero derivative half a cell beyond each
        # endpoint. Expressing those two conditions in second-derivative
        # form gives the endpoint rows below; interior rows are the ordinary
        # cubic-spline continuity equations.
        if not np.all(spacing == spacing[0]):
            raise ValueError(
                "Flat(OnCell()) cubic interpolation requires a regular axis."
            )
        step = spacing[0]
        if flat.dtype == np.dtype(object):
            diagonal = np.empty(n, dtype=object)
            lower = np.empty(n - 1, dtype=object)
            upper = np.empty(n - 1, dtype=object)
            rhs = np.empty_like(flat)
            diagonal[0] = step * 23 / 24
            upper[0] = step / 24
            rhs[0] = slopes[0]
            lower[-1] = step / 24
            diagonal[-1] = step * 23 / 24
            rhs[-1] = -slopes[-1]
            for index in range(1, n - 1):
                lower[index - 1] = spacing[index - 1]
                diagonal[index] = 2 * (
                    spacing[index - 1] + spacing[index]
                )
                upper[index] = spacing[index]
                rhs[index] = 6 * (
                    slopes[index] - slopes[index - 1]
                )
            return _solve_tridiagonal(
                lower, diagonal, upper, rhs
            )
        matrix = np.zeros((n, n), dtype=flat.dtype)
        rhs = np.empty_like(flat)
        matrix[0, 0] = step * np.asarray(23 / 24, dtype=flat.dtype)
        matrix[0, 1] = step * np.asarray(1 / 24, dtype=flat.dtype)
        rhs[0] = slopes[0]
        matrix[-1, -2] = step * np.asarray(1 / 24, dtype=flat.dtype)
        matrix[-1, -1] = step * np.asarray(23 / 24, dtype=flat.dtype)
        rhs[-1] = -slopes[-1]
        for index in range(1, n - 1):
            matrix[index, index - 1] = spacing[index - 1]
            matrix[index, index] = 2 * (
                spacing[index - 1] + spacing[index]
            )
            matrix[index, index + 1] = spacing[index]
            rhs[index] = 6 * (slopes[index] - slopes[index - 1])
        return np.linalg.solve(matrix, rhs)
    if boundary == "periodic":
        step = spacing[0]
        cyclic_slopes = np.empty_like(flat)
        cyclic_slopes[:-1] = slopes
        cyclic_slopes[-1] = (flat[0] - flat[-1]) / step
        rhs = 6 * (cyclic_slopes - np.roll(cyclic_slopes, 1, axis=0))
        diagonal = np.full(n, 4 * step, dtype=spacing.dtype)
        off_diagonal = np.full(n - 1, step, dtype=spacing.dtype)
        return _solve_cyclic_symmetric_tridiagonal(
            off_diagonal, diagonal, step, rhs
        )
    raise AssertionError(f"unknown cubic boundary mode {boundary!r}")


def _float16_tridiagonal_factor(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor a Float16 tridiagonal matrix like Julia's ``lu!`` path."""

    lower = np.array(lower, dtype=np.float16, copy=True)
    diagonal = np.array(diagonal, dtype=np.float16, copy=True)
    upper = np.array(upper, dtype=np.float16, copy=True)
    for index in range(len(lower)):
        lower[index] = np.float16(lower[index] / diagonal[index])
        diagonal[index + 1] = np.float16(
            diagonal[index + 1]
            - np.float16(lower[index] * upper[index])
        )
    return lower, diagonal, upper


def _float16_tridiagonal_solve(
    factor: tuple[np.ndarray, np.ndarray, np.ndarray],
    right_hand_side: np.ndarray,
    *,
    divide_last: bool = False,
) -> np.ndarray:
    """Apply Interpolations.jl's Float16 forward/back substitution order."""

    lower, diagonal, upper = factor
    solution = np.array(right_hand_side, dtype=np.float16, copy=True)
    vector = solution.ndim == 1
    if vector:
        solution = solution[:, None]
    inverse_diagonal = np.empty_like(diagonal)
    for index, item in enumerate(diagonal):
        inverse_diagonal[index] = np.float16(np.float16(1) / item)
    for index in range(1, len(diagonal)):
        solution[index] = np.subtract(
            solution[index],
            np.multiply(lower[index - 1], solution[index - 1], dtype=np.float16),
            dtype=np.float16,
        )
    if divide_last:
        # AxisAlgorithms' no-offset solve divides the final row directly.
        # Interpolations' offset-aware first solve instead multiplies by its
        # precomputed reciprocal.  These differ for some Float16 inputs.
        solution[-1] = np.divide(
            solution[-1], diagonal[-1], dtype=np.float16
        )
    else:
        solution[-1] = np.multiply(
            solution[-1], inverse_diagonal[-1], dtype=np.float16
        )
    for index in range(len(diagonal) - 2, -1, -1):
        solution[index] = np.multiply(
            np.subtract(
                solution[index],
                np.multiply(upper[index], solution[index + 1], dtype=np.float16),
                dtype=np.float16,
            ),
            inverse_diagonal[index],
            dtype=np.float16,
        )
    return solution[:, 0] if vector else solution


def _float16_tridiagonal_matrix_solve(
    factor: tuple[np.ndarray, np.ndarray, np.ndarray],
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Apply Julia LinearAlgebra's Float16 tridiagonal matrix solve."""

    lower, diagonal, upper = factor
    solution = np.array(right_hand_side, dtype=np.float16, copy=True)
    vector = solution.ndim == 1
    if vector:
        solution = solution[:, None]
    for index in range(1, len(diagonal)):
        solution[index] = np.subtract(
            solution[index],
            np.multiply(lower[index - 1], solution[index - 1], dtype=np.float16),
            dtype=np.float16,
        )
    solution[-1] = np.divide(
        solution[-1], diagonal[-1], dtype=np.float16
    )
    if len(diagonal) > 1:
        solution[-2] = np.divide(
            np.subtract(
                solution[-2],
                np.multiply(upper[-1], solution[-1], dtype=np.float16),
                dtype=np.float16,
            ),
            diagonal[-2],
            dtype=np.float16,
        )
    for index in range(len(diagonal) - 3, -1, -1):
        # Tridiagonal LU retains a second superdiagonal filled with zero when
        # pivoting was unnecessary; LinearAlgebra still evaluates that term.
        numerator = np.subtract(
            solution[index],
            np.multiply(upper[index], solution[index + 1], dtype=np.float16),
            dtype=np.float16,
        )
        numerator = np.subtract(
            numerator,
            np.multiply(
                np.float16(0), solution[index + 2], dtype=np.float16
            ),
            dtype=np.float16,
        )
        solution[index] = np.divide(
            numerator, diagonal[index], dtype=np.float16
        )
    return solution[:, 0] if vector else solution


def _float16_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Small Float16 matrix product with Julia's per-operation rounding."""

    left = np.asarray(left, dtype=np.float16)
    right = np.asarray(right, dtype=np.float16)
    vector = right.ndim == 1
    if vector:
        right = right[:, None]
    output = np.zeros((left.shape[0], right.shape[1]), dtype=np.float16)
    for inner in range(left.shape[1]):
        product = np.multiply(
            left[:, inner, None], right[inner, None, :], dtype=np.float16
        )
        output = np.add(output, product, dtype=np.float16)
    return output[:, 0] if vector else output


def _float16_dense_solve(
    matrix: np.ndarray, right_hand_side: np.ndarray
) -> np.ndarray:
    """Solve the two-by-two Woodbury system in Float16 with Julia LU order."""

    factors = np.array(matrix, dtype=np.float16, copy=True)
    solution = np.array(right_hand_side, dtype=np.float16, copy=True)
    vector = solution.ndim == 1
    if vector:
        solution = solution[:, None]
    size = len(factors)
    pivots = list(range(size))
    for column in range(size - 1):
        pivot = column + int(np.argmax(np.abs(factors[column:, column])))
        pivots[column] = pivot
        if pivot != column:
            factors[[column, pivot]] = factors[[pivot, column]]
        pivot_inverse = np.float16(np.float16(1) / factors[column, column])
        for row in range(column + 1, size):
            factors[row, column] = np.float16(
                factors[row, column] * pivot_inverse
            )
        for inner in range(column + 1, size):
            for row in range(column + 1, size):
                factors[row, inner] = np.subtract(
                    factors[row, inner],
                    np.multiply(
                        factors[row, column],
                        factors[column, inner],
                        dtype=np.float16,
                    ),
                    dtype=np.float16,
                )
    pivots[-1] = size - 1

    for column, pivot in enumerate(pivots):
        if pivot != column:
            solution[[column, pivot]] = solution[[pivot, column]]

    # Unit-lower triangular solve.  Julia's generic implementation updates
    # the remaining column after fixing each entry, rather than forming a dot
    # product per row.
    for right_column in range(solution.shape[1]):
        first = solution[0, right_column]
        for row in range(1, size):
            solution[row, right_column] = np.subtract(
                solution[row, right_column],
                np.multiply(
                    factors[row, 0], first, dtype=np.float16
                ),
                dtype=np.float16,
            )
        for column in range(1, size):
            fixed = solution[column, right_column]
            for row in range(column + 1, size):
                solution[row, right_column] = np.subtract(
                    solution[row, right_column],
                    np.multiply(
                        factors[row, column], fixed, dtype=np.float16
                    ),
                    dtype=np.float16,
                )

        # Upper triangular solve in the same column-update order.
        last = np.divide(
            solution[-1, right_column],
            factors[-1, -1],
            dtype=np.float16,
        )
        solution[-1, right_column] = last
        for row in range(size - 2, -1, -1):
            solution[row, right_column] = np.subtract(
                solution[row, right_column],
                np.multiply(
                    factors[row, -1], last, dtype=np.float16
                ),
                dtype=np.float16,
            )
        for column in range(size - 2, -1, -1):
            fixed = np.divide(
                solution[column, right_column],
                factors[column, column],
                dtype=np.float16,
            )
            solution[column, right_column] = fixed
            for row in range(column - 1, -1, -1):
                solution[row, right_column] = np.subtract(
                    solution[row, right_column],
                    np.multiply(
                        factors[row, column], fixed, dtype=np.float16
                    ),
                    dtype=np.float16,
                )
    return solution[:, 0] if vector else solution


def _float16_cubic_coefficients(
    values: np.ndarray,
    boundary: str,
    *,
    input_is_padded: bool = False,
    correction_divide_last: bool = True,
) -> np.ndarray:
    """Prefilter Float16 values exactly as Interpolations.jl 0.16.2 does."""

    padded = boundary != "periodic"
    if input_is_padded and not padded:
        raise ValueError("Periodic cubic coefficients do not use padding.")
    source_length = values.shape[0] - (2 if input_is_padded else 0)
    size = values.shape[0] if input_is_padded else (
        source_length + 2 if padded else source_length
    )
    one_sixth = np.float16(1 / 6)
    lower = np.full(size - 1, one_sixth, dtype=np.float16)
    diagonal = np.full(size, np.float16(2 / 3), dtype=np.float16)
    upper = np.full(size - 1, one_sixth, dtype=np.float16)
    if boundary == "natural":
        diagonal[0] = diagonal[-1] = np.float16(1)
        upper[0] = lower[-1] = np.float16(-2)
        corrections = (
            (0, 2, np.float16(1)),
            (size - 1, size - 3, np.float16(1)),
        )
    elif boundary == "flat":
        diagonal[0] = diagonal[-1] = np.float16(-1)
        upper[0] = lower[-1] = np.float16(0)
        corrections = (
            (0, 2, np.float16(1)),
            (size - 1, size - 3, np.float16(1)),
        )
    elif boundary == "flat_oncell":
        # Cubic(Flat(OnCell())) applies its zero-slope condition half a cell
        # beyond each endpoint. Interpolations.jl expresses the two boundary
        # rows as [-9, 11, -3, 1] and its reversal. Keep those coefficients
        # and the Woodbury solve in Float16: widening this prefilter changes
        # both the stored coefficient type and the knot values.
        diagonal[0] = diagonal[-1] = np.float16(-9)
        upper[0] = lower[-1] = np.float16(11)
        corrections = (
            (0, 2, np.float16(-3)),
            (size - 1, size - 3, np.float16(-3)),
            (0, 3, np.float16(1)),
            (size - 1, size - 4, np.float16(1)),
        )
    elif boundary == "periodic":
        corrections = (
            (0, size - 1, upper[0]),
            (size - 1, 0, lower[-1]),
        )
    else:  # pragma: no cover - guarded by the public constructor
        raise AssertionError(f"unknown cubic boundary mode {boundary!r}")

    factor = _float16_tridiagonal_factor(lower, diagonal, upper)
    correction_count = len(corrections)
    rows = np.zeros((size, correction_count), dtype=np.float16)
    columns = np.zeros((correction_count, size), dtype=np.float16)
    inverse_values = np.zeros(
        (correction_count, correction_count), dtype=np.float16
    )
    for correction, (row, column, value) in enumerate(corrections):
        rows[row, correction] = np.float16(1)
        columns[correction, column] = np.float16(1)
        inverse_values[correction, correction] = np.float16(
            np.float16(1) / value
        )
    base_inverse_rows = _float16_tridiagonal_matrix_solve(factor, rows)
    reduced = np.add(
        inverse_values,
        _float16_matmul(columns, base_inverse_rows),
        dtype=np.float16,
    )
    woodbury_inverse = _float16_dense_solve(
        reduced, np.eye(correction_count, dtype=np.float16)
    )

    if input_is_padded:
        right_hand_side = np.asarray(values, dtype=np.float16)
    elif padded:
        right_hand_side = np.zeros(
            (size, values.shape[1]), dtype=np.float16
        )
        right_hand_side[1:-1] = values
    else:
        right_hand_side = np.asarray(values, dtype=np.float16)
    solution = _float16_tridiagonal_solve(factor, right_hand_side)
    correction = _float16_matmul(columns, solution)
    correction = _float16_matmul(woodbury_inverse, correction)
    correction = _float16_matmul(rows, correction)
    correction = _float16_tridiagonal_solve(
        factor, correction, divide_last=correction_divide_last
    )
    return np.subtract(solution, correction, dtype=np.float16)


def _float16_cubic_tensor_coefficients(
    values: np.ndarray,
    boundary: str,
) -> np.ndarray:
    """Prefilter every tensor axis before evaluation, in Julia's axis order.

    Interpolations.jl pads the complete coefficient array and then solves its
    spline system along dimensions ``1:N``.  Evaluating and rounding one axis
    before prefiltering the next is not equivalent for Float16 data.
    """

    padded = boundary != "periodic"
    source = np.asarray(values, dtype=np.float16)
    coefficients = (
        np.pad(
            source,
            ((1, 1),) * source.ndim,
            mode="constant",
        )
        if padded
        else np.array(source, copy=True)
    )
    for axis in range(coefficients.ndim):
        moved = np.moveaxis(coefficients, axis, 0)
        moved_shape = moved.shape
        filtered = _float16_cubic_coefficients(
            moved.reshape(moved_shape[0], -1),
            boundary,
            input_is_padded=padded,
            # AxisAlgorithms uses direct division for the final row only in
            # its first-dimension solver.  Later dimensions multiply by the
            # precomputed reciprocal inside their SIMD loop.
            correction_divide_last=axis == 0,
        )
        coefficients = np.moveaxis(filtered.reshape(moved_shape), 0, axis)
    return coefficients


def _float16_cubic_weights(
    fractional: np.ndarray, *, derivative: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate cubic weights with Julia's expression/rounding order."""

    dtype = fractional.dtype
    one = np.asarray(1, dtype=dtype)[()]
    complement = np.subtract(one, fractional, dtype=dtype)
    square = np.multiply(fractional, fractional, dtype=dtype)
    complement_square = np.multiply(complement, complement, dtype=dtype)
    if derivative:
        return (
            np.multiply(
                np.asarray(-0.5, dtype=dtype), complement_square, dtype=dtype
            ),
            np.add(
                np.multiply(
                    np.asarray(-2, dtype=dtype), fractional, dtype=dtype
                ),
                np.multiply(
                    np.asarray(1.5, dtype=dtype), square, dtype=dtype
                ),
                dtype=dtype,
            ),
            np.subtract(
                np.multiply(
                    np.asarray(2, dtype=dtype), complement, dtype=dtype
                ),
                np.multiply(
                    np.asarray(1.5, dtype=dtype), complement_square, dtype=dtype
                ),
                dtype=dtype,
            ),
            np.multiply(np.asarray(0.5, dtype=dtype), square, dtype=dtype),
        )
    cube = np.multiply(square, fractional, dtype=dtype)
    complement_cube = np.multiply(
        complement_square, complement, dtype=dtype
    )
    return (
        np.multiply(
            np.asarray(1 / 6, dtype=dtype), complement_cube, dtype=dtype
        ),
        np.add(
            np.subtract(
                np.asarray(2 / 3, dtype=dtype), square, dtype=dtype
            ),
            np.multiply(np.asarray(0.5, dtype=dtype), cube, dtype=dtype),
            dtype=dtype,
        ),
        np.add(
            np.subtract(
                np.asarray(2 / 3, dtype=dtype), complement_square, dtype=dtype
            ),
            np.multiply(
                np.asarray(0.5, dtype=dtype), complement_cube, dtype=dtype
            ),
            dtype=dtype,
        ),
        np.multiply(np.asarray(1 / 6, dtype=dtype), cube, dtype=dtype),
    )


def _float16_cubic_contract_axis(
    coefficients: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    axis: int,
    *,
    calculation_dtype: np.dtype[Any],
    spline_boundary: str,
    derivative: bool = False,
) -> np.ndarray:
    """Contract one cubic coefficient axis without rounding the tensor."""

    moved = np.moveaxis(np.asarray(coefficients), axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    weight_dtype = np.dtype(target.dtype)
    source_work = source.astype(weight_dtype, copy=False)
    target_work = target.astype(weight_dtype, copy=False)
    spacing = np.subtract(source_work[1], source_work[0], dtype=weight_dtype)
    lower = source_work[0]
    upper = source_work[-1]
    if spline_boundary in ("periodic", "flat_oncell"):
        half_spacing = np.divide(
            spacing, np.asarray(2, dtype=weight_dtype), dtype=weight_dtype
        )
        evaluation_lower = np.subtract(
            lower, half_spacing, dtype=weight_dtype
        )
        evaluation_upper = np.add(
            upper, half_spacing, dtype=weight_dtype
        )
    else:
        evaluation_lower = lower
        evaluation_upper = upper
    evaluation_target = np.clip(
        target_work, evaluation_lower, evaluation_upper
    )
    internal = np.add(
        np.divide(
            np.subtract(evaluation_target, lower, dtype=weight_dtype),
            spacing,
            dtype=weight_dtype,
        ),
        np.asarray(1, dtype=weight_dtype),
        dtype=weight_dtype,
    )
    if spline_boundary == "flat_oncell":
        base = np.floor(
            np.where(
                internal < np.asarray(1, dtype=weight_dtype),
                np.add(
                    internal,
                    np.asarray(0.5, dtype=weight_dtype),
                    dtype=weight_dtype,
                ),
                internal,
            )
        ).astype(np.int64)
    else:
        base = np.floor(internal).astype(np.int64)
    if spline_boundary != "periodic":
        base = np.where(base > len(source) - 1, base - 1, base)
    fractional = np.subtract(
        internal, base.astype(weight_dtype), dtype=weight_dtype
    )
    indexes = (
        tuple(
            np.mod(base - 2 + offset, len(source))
            for offset in range(4)
        )
        if spline_boundary == "periodic"
        else tuple(base - 1 + offset for offset in range(4))
    )
    weights = _float16_cubic_weights(fractional, derivative=derivative)
    result = np.multiply(
        weights[0][:, None],
        flat[indexes[0]],
        dtype=calculation_dtype,
    )
    for position in range(1, 4):
        result = np.add(
            np.multiply(
                weights[position][:, None],
                flat[indexes[position]],
                dtype=calculation_dtype,
            ),
            result,
            dtype=calculation_dtype,
        )
    if derivative:
        result = np.divide(result, spacing, dtype=calculation_dtype)
    reshaped = result.reshape((len(target),) + moved.shape[1:])
    return np.moveaxis(reshaped, 0, axis)


def _real_cubic_grid_storage_dtype(
    values: np.ndarray,
    output_targets: tuple[Any, ...],
) -> np.dtype[Any] | None:
    """Return Julia's real low-precision range-indexing storage dtype."""

    value_dtype = np.dtype(values.dtype)
    if value_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        return None
    if all(np.asarray(target).ndim == 0 for target in output_targets):
        # Scalar interpolation returns the promoted calculation type; only
        # array/range getindex stores into the interpolation's element type.
        return None
    output_dtype = value_dtype
    for target in output_targets:
        output_dtype = _julia_numeric_dtype(
            np.empty((), dtype=output_dtype), np.asarray(target)
        )
    return output_dtype


def _float16_cubic_tensor(
    values: np.ndarray,
    sources: tuple[np.ndarray, ...],
    targets: tuple[np.ndarray, ...],
    extrapolation_targets: tuple[np.ndarray, ...],
    linear_axes: tuple[tuple[bool, bool], ...],
    output_targets: tuple[Any, ...],
    spline_boundary: str,
) -> np.ndarray:
    """Evaluate a Float16 cubic spline as one tensor expression."""

    calculation_dtype = np.dtype(np.float16)
    for target in targets:
        calculation_dtype = _julia_numeric_dtype(
            np.empty((), dtype=calculation_dtype), target
        )
    storage_dtype = _real_cubic_grid_storage_dtype(values, output_targets)
    output_dtype = calculation_dtype if storage_dtype is None else storage_dtype
    # Range indexing converts only the completed tensor result back to the
    # interpolation's declared element type.  Intermediate contractions keep
    # the promotion selected by scaled coordinates and cubic weights.
    coefficients = _float16_cubic_tensor_coefficients(values, spline_boundary)

    def contract(derivative_axis: int | None = None) -> np.ndarray:
        output = coefficients
        # ``interp_getindex`` recursively expands dimension 1 around the
        # complete expansion of dimensions 2:N, so contractions run from the
        # last axis to the first and retain their promoted intermediates.
        for axis in range(output.ndim - 1, -1, -1):
            output = _float16_cubic_contract_axis(
                output,
                sources[axis],
                targets[axis],
                axis,
                calculation_dtype=calculation_dtype,
                spline_boundary=spline_boundary,
                derivative=axis == derivative_axis,
            )
        return output

    output = contract()
    for axis, (target, reference, line_sides) in enumerate(
        zip(targets, extrapolation_targets, linear_axes, strict=True)
    ):
        lower_linear, upper_linear = line_sides
        outside = ((reference < target) & lower_linear) | (
            (reference > target) & upper_linear
        )
        if not np.any(outside):
            continue
        displacement = np.where(
            outside,
            np.subtract(reference, target, dtype=target.dtype),
            np.asarray(0, dtype=target.dtype),
        )
        shape = [1] * output.ndim
        shape[axis] = len(displacement)
        output = np.add(
            output,
            np.multiply(
                displacement.reshape(shape),
                contract(axis),
                dtype=calculation_dtype,
            ),
            dtype=calculation_dtype,
        )
    return output.astype(output_dtype, copy=False)


def _float16_cubic_axis(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    axis: int,
    *,
    linear_extrapolation: bool | tuple[bool, bool],
    extrapolation_target: np.ndarray | None = None,
    output_target: np.ndarray | None = None,
    spline_boundary: str,
) -> np.ndarray:
    """Evaluate a cubic whose stored coefficients use Float16 arithmetic."""

    moved = np.moveaxis(np.asarray(values), axis, 0)
    flat = moved.reshape(len(source), -1)
    coefficients = _float16_cubic_coefficients(flat, spline_boundary)
    output_dtype = _interpolation_output_dtype(
        moved,
        (
            output_target
            if spline_boundary == "flat_oncell" and output_target is not None
            else target
        ),
    )
    weight_dtype = np.dtype(target.dtype)
    source_work = source.astype(weight_dtype, copy=False)
    target_work = target.astype(weight_dtype, copy=False)
    spacing = np.subtract(source_work[1], source_work[0], dtype=weight_dtype)
    lower = source_work[0]
    upper = source_work[-1]
    if spline_boundary in ("periodic", "flat_oncell"):
        half_spacing = np.divide(
            spacing, np.asarray(2, dtype=weight_dtype), dtype=weight_dtype
        )
        evaluation_lower = np.subtract(
            lower, half_spacing, dtype=weight_dtype
        )
        evaluation_upper = np.add(
            upper, half_spacing, dtype=weight_dtype
        )
    else:
        evaluation_lower = lower
        evaluation_upper = upper
    evaluation_target = np.clip(
        target_work, evaluation_lower, evaluation_upper
    )
    internal = np.add(
        np.divide(
            np.subtract(evaluation_target, lower, dtype=weight_dtype),
            spacing,
            dtype=weight_dtype,
        ),
        np.asarray(1, dtype=weight_dtype),
        dtype=weight_dtype,
    )
    if spline_boundary == "flat_oncell":
        # Interpolations.floorbounds rounds the lower half-cell toward the
        # interior before applying the ordinary upper-end adjustment.
        base = np.floor(
            np.where(
                internal < np.asarray(1, dtype=weight_dtype),
                np.add(
                    internal,
                    np.asarray(0.5, dtype=weight_dtype),
                    dtype=weight_dtype,
                ),
                internal,
            )
        ).astype(np.int64)
    else:
        base = np.floor(internal).astype(np.int64)
    if spline_boundary != "periodic":
        base = np.where(base > len(source) - 1, base - 1, base)
    fractional = np.subtract(
        internal, base.astype(weight_dtype), dtype=weight_dtype
    )

    if spline_boundary == "periodic":
        indexes = tuple(
            np.mod(base - 2 + offset, len(source)) for offset in range(4)
        )
    else:
        indexes = tuple(base - 1 + offset for offset in range(4))

    def evaluate(*, derivative: bool = False) -> np.ndarray:
        if (
            not derivative
            and spline_boundary == "flat_oncell"
            and output_dtype == np.dtype(np.float16)
        ):
            # OnCell's scaled coordinates and weights are Float64 even when
            # the samples are Float16. Interpolations.jl collects that weighted
            # sum in the coordinate type, then converts each result back to
            # the interpolation's Float16 element type.
            ratio_weights = _float16_cubic_weights(
                fractional.astype(np.float64, copy=False)
            )
            ratio_result = (
                ratio_weights[0][:, None]
                * coefficients[indexes[0]].astype(np.float64)
            )
            for position in range(1, 4):
                ratio_result = (
                    ratio_weights[position][:, None]
                    * coefficients[indexes[position]].astype(np.float64)
                    + ratio_result
                )
            return ratio_result.astype(np.float16)

        weights = _float16_cubic_weights(
            fractional, derivative=derivative
        )
        result = np.multiply(
            weights[0][:, None],
            coefficients[indexes[0]],
            dtype=output_dtype,
        )
        for position in range(1, 4):
            result = np.add(
                np.multiply(
                    weights[position][:, None],
                    coefficients[indexes[position]],
                    dtype=output_dtype,
                ),
                result,
                dtype=output_dtype,
            )
        return result

    output = evaluate()
    lower_linear, upper_linear = (
        linear_extrapolation
        if isinstance(linear_extrapolation, tuple)
        else (linear_extrapolation, linear_extrapolation)
    )
    if lower_linear or upper_linear:
        reference = (
            target_work
            if extrapolation_target is None
            else np.asarray(extrapolation_target, dtype=weight_dtype)
        )
        outside = ((reference < evaluation_target) & lower_linear) | (
            (reference > evaluation_target) & upper_linear
        )
        if np.any(outside):
            derivative = np.divide(
                evaluate(derivative=True), spacing, dtype=output_dtype
            )
            displacement = np.subtract(
                reference, evaluation_target, dtype=weight_dtype
            )
            output[outside] = np.add(
                output[outside],
                np.multiply(
                    displacement[outside, None],
                    derivative[outside],
                    dtype=output_dtype,
                ),
                dtype=output_dtype,
            )
    reshaped = output.reshape((len(target),) + moved.shape[1:])
    return np.moveaxis(reshaped, 0, axis)


def _natural_cubic_axis(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    axis: int,
    *,
    linear_extrapolation: bool | tuple[bool, bool] = False,
    extrapolation_target: np.ndarray | None = None,
    output_target: np.ndarray | None = None,
    spline_boundary: str = "natural",
    derivative: bool = False,
    _mpfr_active: bool = False,
) -> np.ndarray:
    """Evaluate a natural cubic spline along one array axis."""

    source = np.asarray(source)
    target = np.asarray(target)
    moved = np.moveaxis(np.asarray(values), axis, 0)
    n = len(source)
    if moved.shape[0] != n:
        raise DimensionMismatch("Array size does not match interpolation lattice.")
    if n == 0:
        raise ValueError("Cannot interpolate an empty axis.")
    exact_domain = _exact_domain(moved, target)
    if exact_domain in (_MPFR, _MPC, _MPFRComplex) and not _mpfr_active:
        with _bigfloat_context():
            return _natural_cubic_axis(
                values,
                source,
                target,
                axis,
                linear_extrapolation=linear_extrapolation,
                extrapolation_target=extrapolation_target,
                output_target=output_target,
                spline_boundary=spline_boundary,
                derivative=derivative,
                _mpfr_active=True,
            )
    if exact_domain is not None:
        moved = _as_exact_array(moved, exact_domain)
        coordinate_domain = _exact_coordinate_domain(exact_domain)
        source = _as_exact_array(source, coordinate_domain)
        target = _as_exact_array(target, coordinate_domain)
    coordinate_dtype = _coordinate_dtype(source, target)
    coefficient_dtype = _interpolation_coefficient_dtype(moved)
    output_dtype = _interpolation_output_dtype(moved, target)
    dtype = output_dtype
    source = source.astype(coordinate_dtype, copy=False)
    target = target.astype(coordinate_dtype, copy=False)
    if (
        coefficient_dtype == np.dtype(np.float16)
        and n > 1
    ):
        if derivative:
            raise TypeError(
                "Float16 tensor derivatives require the tensor spline path."
            )
        return _float16_cubic_axis(
            np.asarray(values),
            source,
            target,
            axis,
            linear_extrapolation=linear_extrapolation,
            extrapolation_target=extrapolation_target,
            output_target=output_target,
            spline_boundary=spline_boundary,
        )
    calculation_coordinate_dtype = (
        np.dtype(np.float32)
        if coordinate_dtype == np.dtype(np.float16)
        else coordinate_dtype
    )
    source = source.astype(calculation_coordinate_dtype, copy=False)
    target = target.astype(calculation_coordinate_dtype, copy=False)
    flat = moved.astype(dtype, copy=False).reshape(n, -1)
    if n == 1:
        output = np.repeat(flat, len(target), axis=0)
        result = np.moveaxis(
            output.reshape((len(target),) + moved.shape[1:]), 0, axis
        )
        return result.astype(output_dtype, copy=False)


    spacing = np.diff(source)
    if np.any(spacing <= 0):
        raise ValueError("Interpolation lattice axes must have positive steps.")

    second = _cubic_second_derivatives(flat, spacing, spline_boundary)

    if spline_boundary == "periodic":
        step = spacing[0]
        half_step = step / 2
        evaluation_target = np.clip(
            target, source[0] - half_step, source[-1] + half_step
        )
        # A Periodic(OnCell()) cubic has one wrapped interval on either side
        # of the stored nodes.  Extending the knots and coefficient data by
        # one sample lets the ordinary cubic polynomial evaluate those
        # half-cells exactly like Interpolations.jl.
        extended_source = np.concatenate(
            (
                np.asarray([source[0] - step], dtype=source.dtype),
                source,
                np.asarray([source[-1] + step], dtype=source.dtype),
            )
        )
        extended_flat = np.concatenate((flat[-1:], flat, flat[:1]), axis=0)
        extended_second = np.concatenate(
            (second[-1:], second, second[:1]), axis=0
        )
        interval = (
            np.searchsorted(
                extended_source, evaluation_target, side="right"
            )
            - 1
        )
        interval = np.clip(interval, 0, n)
        x0 = extended_source[interval]
        x1 = extended_source[interval + 1]
        value0 = extended_flat[interval]
        value1 = extended_flat[interval + 1]
        second0 = extended_second[interval]
        second1 = extended_second[interval + 1]
    else:
        if spline_boundary == "flat_oncell":
            half_step = spacing[0] / 2
            evaluation_target = np.clip(
                target, source[0] - half_step, source[-1] + half_step
            )
        else:
            evaluation_target = np.clip(target, source[0], source[-1])
        interval = np.searchsorted(
            source, evaluation_target, side="right"
        ) - 1
        interval = np.clip(interval, 0, n - 2)
        x0 = source[interval]
        x1 = source[interval + 1]
        value0 = flat[interval]
        value1 = flat[interval + 1]
        second0 = second[interval]
        second1 = second[interval + 1]
    width = x1 - x0
    a = (x1 - evaluation_target) / width
    b = (evaluation_target - x0) / width
    if derivative:
        output = (
            (value1 - value0) / width[:, None]
            + ((1 - 3 * a**2) * width)[:, None] * second0 / 6
            + ((3 * b**2 - 1) * width)[:, None] * second1 / 6
        )
    else:
        output = (
            a[:, None] * value0
            + b[:, None] * value1
            + ((a**3 - a) * width**2)[:, None] * second0 / 6
            + ((b**3 - b) * width**2)[:, None] * second1 / 6
        )
    lower_linear, upper_linear = (
        linear_extrapolation
        if isinstance(linear_extrapolation, tuple)
        else (linear_extrapolation, linear_extrapolation)
    )
    if not derivative and (lower_linear or upper_linear):
        reference = (
            target
            if extrapolation_target is None
            else np.asarray(extrapolation_target, dtype=target.dtype)
        )
        below = (reference < evaluation_target) & lower_linear
        above = (reference > evaluation_target) & upper_linear
        if spline_boundary in {"periodic", "flat_oncell"}:
            derivative = (
                (value1 - value0) / width[:, None]
                + ((1 - 3 * a**2) * width)[:, None] * second0 / 6
                + ((3 * b**2 - 1) * width)[:, None] * second1 / 6
            )
            outside = below | above
            if np.any(outside):
                output[outside] += (
                    reference[outside] - evaluation_target[outside]
                )[:, None] * derivative[outside]
        else:
            if np.any(below):
                lower_slope = (
                    (flat[1] - flat[0]) / spacing[0]
                    - spacing[0] * (2 * second[0] + second[1]) / 6
                )
                output[below] += (
                    reference[below] - evaluation_target[below]
                )[:, None] * lower_slope
            if np.any(above):
                upper_slope = (
                    (flat[-1] - flat[-2]) / spacing[-1]
                    + spacing[-1] * (second[-2] + 2 * second[-1]) / 6
                )
                output[above] += (
                    reference[above] - evaluation_target[above]
                )[:, None] * upper_slope
    reshaped = output.reshape((len(target),) + moved.shape[1:])
    return np.moveaxis(reshaped, 0, axis).astype(output_dtype, copy=False)


def _linear_axis(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    axis: int,
    *,
    linear_extrapolation: bool | tuple[bool, bool] = False,
    extrapolation_target: np.ndarray | None = None,
    _mpfr_active: bool = False,
) -> np.ndarray:
    """Evaluate piecewise-linear interpolation along one array axis."""

    source = np.asarray(source)
    target = np.asarray(target)
    moved = np.moveaxis(np.asarray(values), axis, 0)
    n = len(source)
    if moved.shape[0] != n:
        raise DimensionMismatch("Array size does not match interpolation lattice.")
    if n == 0:
        raise ValueError("Cannot interpolate an empty axis.")
    exact_domain = _exact_domain(moved, target)
    if exact_domain in (_MPFR, _MPC, _MPFRComplex) and not _mpfr_active:
        with _bigfloat_context():
            return _linear_axis(
                values,
                source,
                target,
                axis,
                linear_extrapolation=linear_extrapolation,
                extrapolation_target=extrapolation_target,
                _mpfr_active=True,
            )
    if exact_domain is not None:
        moved = _as_exact_array(moved, exact_domain)
        coordinate_domain = _exact_coordinate_domain(exact_domain)
        source = _as_exact_array(source, coordinate_domain)
        target = _as_exact_array(target, coordinate_domain)
    coordinate_dtype = _coordinate_dtype(source, target)
    dtype = _interpolation_output_dtype(moved, target)
    source = source.astype(coordinate_dtype, copy=False)
    target = target.astype(coordinate_dtype, copy=False)
    flat = moved.astype(dtype, copy=False).reshape(n, -1)
    if n == 1:
        output = np.repeat(flat, len(target), axis=0)
        return np.moveaxis(
            output.reshape((len(target),) + moved.shape[1:]), 0, axis
        )

    spacing = np.diff(source)
    if np.any(spacing <= 0):
        raise ValueError("Interpolation lattice axes must have positive steps.")
    evaluation_target = np.clip(target, source[0], source[-1])
    interval = np.searchsorted(source, evaluation_target, side="right") - 1
    interval = np.clip(interval, 0, n - 2)
    weight = (
        (evaluation_target - source[interval])
        / (source[interval + 1] - source[interval])
    )
    output = flat[interval] + weight[:, None] * (
        flat[interval + 1] - flat[interval]
    )
    lower_linear, upper_linear = (
        linear_extrapolation
        if isinstance(linear_extrapolation, tuple)
        else (linear_extrapolation, linear_extrapolation)
    )
    if lower_linear or upper_linear:
        reference = (
            target
            if extrapolation_target is None
            else np.asarray(extrapolation_target, dtype=target.dtype)
        )
        below = (reference < evaluation_target) & lower_linear
        above = (reference > evaluation_target) & upper_linear
        if np.any(below):
            lower_slope = (flat[1] - flat[0]) / spacing[0]
            output[below] += (
                reference[below] - evaluation_target[below]
            )[:, None] * lower_slope
        if np.any(above):
            upper_slope = (flat[-1] - flat[-2]) / spacing[-1]
            output[above] += (
                reference[above] - evaluation_target[above]
            )[:, None] * upper_slope
    reshaped = output.reshape((len(target),) + moved.shape[1:])
    return np.moveaxis(reshaped, 0, axis)


_BOUNDARY_TYPES = (Flat, Periodic, Throw, Linear)


def _is_axis_boundary(boundary: Any) -> bool:
    """Return whether *boundary* is one Julia extrapolation dimension spec."""

    return isinstance(boundary, _BOUNDARY_TYPES) or (
        isinstance(boundary, tuple)
        and len(boundary) == 2
        and all(isinstance(item, _BOUNDARY_TYPES) for item in boundary)
    )


def _axis_policies(boundary: Any, ndim: int) -> tuple[Any, ...]:
    """Expand only genuine Julia boundary tuples into per-axis policies.

    Julia dispatches a tuple such as ``(10.0, 20.0)`` to
    ``FilledExtrapolation``: the complete tuple is the value returned outside
    the interpolation domain.  A tuple is an axis specification only when
    every member is a boundary condition or a lower/upper boundary pair.
    Treating every Python tuple as an axis list lost that important
    distinction.
    """

    # Interpolations.jl first treats an unnested tuple of boundary objects as
    # the per-dimension tuple.  Consequently, in one dimension
    # ``(Flat(), Periodic())`` selects the first (Flat) policy; it is *not* a
    # lower/upper pair.  A directional one-dimensional policy needs the same
    # nesting as Julia: ``((Flat(), Periodic()),)``.
    if (
        ndim == 1
        and isinstance(boundary, tuple)
        and boundary
        and all(_is_axis_boundary(item) for item in boundary)
    ):
        return (boundary[0],)
    if isinstance(boundary, tuple) and all(
        _is_axis_boundary(item) for item in boundary
    ):
        if len(boundary) != ndim:
            raise ValueError("Boundary-condition tuple must match lattice dimension.")
        return boundary
    return (boundary,) * ndim


def _normalize_ranges(
    ranges: Any,
    values: np.ndarray,
    *,
    require_regular: bool = True,
) -> Lattice:
    """Accept both Julia's one-axis and tuple-of-axes constructor spellings."""

    if values.ndim == 1 and not _looks_like_lattice(ranges):
        axis = np.asarray(ranges)
        if axis.ndim == 1:
            return (_axis(ranges),)
    if require_regular:
        return as_lattice(ranges)
    if not isinstance(ranges, (tuple, list)):
        raise TypeError("Interpolation ranges must be a tuple or list of axes.")
    # Gridded(Linear()) accepts arbitrary strictly-increasing AbstractVectors.
    # Do not route those vectors through ``as_lattice``, whose regularity
    # check intentionally models SLMTools' AbstractRange-based Lattice type.
    return tuple(_axis(item) for item in ranges)


def _map_targets(
    ranges: Lattice,
    coordinates: tuple[Any, ...],
    boundary: Any,
    *,
    oncell_periodic_axes: tuple[bool, ...] | None = None,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[tuple[bool, bool], ...],
    Any,
]:
    """Apply Interpolations.jl-compatible extrapolation policies."""

    if len(coordinates) != len(ranges):
        raise DimensionMismatch("Wrong number of interpolation coordinates.")
    targets = tuple(np.atleast_1d(np.asarray(item)) for item in coordinates)
    policies = _axis_policies(boundary, len(ranges))
    evaluation_sources: list[np.ndarray] = []
    mapped: list[np.ndarray] = []
    extrapolation_targets: list[np.ndarray] = []
    outside_masks: list[np.ndarray] = []
    linear_axes: list[tuple[bool, bool]] = []
    fill_value: Any = _OMITTED
    if oncell_periodic_axes is None:
        oncell_periodic_axes = (False,) * len(ranges)
    if len(oncell_periodic_axes) != len(ranges):
        raise DimensionMismatch(
            "Periodic-axis metadata must match lattice dimension."
        )
    for source_axis, target, policy, oncell_periodic in zip(
        ranges,
        targets,
        policies,
        oncell_periodic_axes,
        strict=True,
    ):
        if target.ndim != 1:
            raise DimensionMismatch("Interpolation coordinates must be vectors.")
        source_work, target_work = _evaluation_axis(source_axis, target)
        coordinate_dtype = source_work.dtype
        if oncell_periodic:
            object_values = (
                _object_values(source_work) + _object_values(target_work)
            )
            if not any(
                isinstance(value, Decimal) for value in object_values
            ):
                # Interpolations' OnCell bounds use the Float64 literal 0.5.
                # It widens machine-float and Rational coordinate arithmetic
                # to Float64 (BigFloat remains BigFloat).
                source_work = np.asarray(source_work, dtype=np.float64)
                target_work = np.asarray(target_work, dtype=np.float64)
                coordinate_dtype = np.dtype(np.float64)
        lower = source_work[0]
        upper = source_work[-1]
        if oncell_periodic:
            # ``Cubic(Periodic(OnCell()))`` is defined on a half-cell beyond
            # each end node.  Its scaled bounds are therefore
            # ``first-step/2:last+step/2`` and its extrapolation period spans
            # all ``n`` cells, not the ``n-1`` gaps between stored samples.
            # Interpolations.jl exposes those bounds through ``bounds(itp)``;
            # using the node endpoints here changes both Periodic and Line
            # extrapolation while leaving in-domain samples deceptively right.
            step = source_work[1] - source_work[0]
            lower = lower - step / 2
            upper = upper + step / 2
        outside = (target_work < lower) | (target_work > upper)
        linear_sides = (False, False)
        if isinstance(policy, tuple) and _is_axis_boundary(policy):
            lower_policy, upper_policy = policy
            mapped_target = np.array(target_work, copy=True)

            if isinstance(lower_policy, Throw):
                if np.any(mapped_target < lower):
                    raise IndexError("Interpolation coordinate out of bounds.")
            elif isinstance(lower_policy, (Flat, Linear)):
                mapped_target = np.maximum(mapped_target, lower)
            elif isinstance(lower_policy, Periodic):
                mapped_target = (
                    np.full(mapped_target.shape, lower, dtype=coordinate_dtype)
                    if upper == lower
                    else _periodic_coordinates(
                        mapped_target, lower, upper, coordinate_dtype
                    )
                )

            if isinstance(upper_policy, Throw):
                if np.any(mapped_target > upper):
                    raise IndexError("Interpolation coordinate out of bounds.")
            elif isinstance(upper_policy, (Flat, Linear)):
                mapped_target = np.minimum(mapped_target, upper)
            elif isinstance(upper_policy, Periodic):
                mapped_target = (
                    np.full(mapped_target.shape, lower, dtype=coordinate_dtype)
                    if upper == lower
                    else _periodic_coordinates(
                        mapped_target, lower, upper, coordinate_dtype
                    )
                )
            outside = np.zeros_like(outside)
            linear_sides = (
                isinstance(lower_policy, Linear),
                isinstance(upper_policy, Linear),
            )
        elif isinstance(policy, Flat):
            mapped_target = np.clip(target_work, lower, upper)
            outside = np.zeros_like(outside)
        elif isinstance(policy, Periodic):
            if upper == lower:
                mapped_target = np.full(target_work.shape, lower, dtype=coordinate_dtype)
            else:
                # Always allocate in the coordinate working dtype.  Copying an
                # integer query vector and assigning wrapped fractions into it
                # silently truncated the coordinates in the original port.
                # Interpolations.periodic maps the closed upper bound back to
                # the lower bound too; it does not wrap only strictly
                # out-of-bounds coordinates.
                mapped_target = _periodic_coordinates(
                    target_work, lower, upper, coordinate_dtype
                )
            outside = np.zeros_like(outside)
        elif isinstance(policy, Throw):
            if np.any(outside):
                raise IndexError("Interpolation coordinate out of bounds.")
            mapped_target = target_work
        elif isinstance(policy, Linear):
            # Interpolations.jl's replace_linear_line compatibility hook.
            mapped_target = np.clip(target_work, lower, upper)
            outside = np.zeros_like(outside)
            linear_sides = (True, True)
        elif (
            policy is None
            or isinstance(policy, Number)
            or np.isscalar(policy)
            or isinstance(policy, tuple)
        ):
            if fill_value is _OMITTED:
                fill_value = policy
            elif not (
                policy == fill_value
                or (
                    np.isscalar(policy)
                    and np.isscalar(fill_value)
                    and bool(np.isnan(policy))
                    and bool(np.isnan(fill_value))
                )
            ):
                raise ValueError("Filled extrapolation uses one scalar fill value.")
            mapped_target = np.clip(target_work, lower, upper)
        else:
            raise TypeError(f"Unsupported extrapolation boundary {policy!r}.")
        evaluation_sources.append(source_work)
        mapped.append(np.asarray(mapped_target))
        extrapolation_targets.append(np.asarray(target_work))
        outside_masks.append(np.asarray(outside))
        linear_axes.append(linear_sides)
    return (
        tuple(evaluation_sources),
        tuple(mapped),
        tuple(extrapolation_targets),
        tuple(outside_masks),
        tuple(linear_axes),
        fill_value,
    )


def _apply_fill(
    output: np.ndarray,
    outside_masks: tuple[np.ndarray, ...],
    fill_value: Any,
) -> np.ndarray:
    """Apply a scalar fill boundary to the tensor-product result."""

    if fill_value is _OMITTED:
        return output
    has_outside = any(np.any(mask) for mask in outside_masks)
    output_dtype = (
        np.dtype(object)
        if (
            output.dtype == np.dtype(object)
            or fill_value is None
            or isinstance(fill_value, tuple)
        )
        else _julia_numeric_dtype(output, fill_value)
    )
    output = output.astype(output_dtype, copy=False)
    if not has_outside:
        return output
    combined = np.zeros(tuple(map(len, outside_masks)), dtype=bool)
    for axis_number, mask in enumerate(outside_masks):
        shape = [1] * len(outside_masks)
        shape[axis_number] = len(mask)
        combined |= mask.reshape(shape)
    if isinstance(fill_value, tuple):
        # NumPy interprets a tuple assigned through a boolean mask as a vector
        # to broadcast.  A Julia FilledExtrapolation instead stores and
        # returns that tuple as one scalar value.
        for index in zip(*np.nonzero(combined), strict=True):
            output[index] = fill_value
    else:
        output[combined] = fill_value
    return output


class _CubicSpline:
    def __init__(
        self,
        ranges: Any,
        values: Any,
        *,
        extrapolation_bc: Any = 0,
        spline_boundary: str = "natural",
        spline_oncell: bool = False,
    ) -> None:
        self.values = np.asarray(values)
        self.ranges = _normalize_ranges(ranges, self.values)
        if self.values.ndim != len(self.ranges) or self.values.shape != tuple(
            map(len, self.ranges)
        ):
            raise DimensionMismatch("Array size does not match interpolation lattice.")
        if any(len(axis) < 2 for axis in self.ranges):
            raise ValueError(
                "Cubic interpolation does not support singleton source axes."
            )
        for axis in self.ranges:
            _regular_positive_step(axis)
        self.extrapolation_bc = extrapolation_bc
        self.spline_boundary = spline_boundary
        self.spline_oncell = spline_oncell

    def _evaluate_paired_2d(
        self,
        coordinates: tuple[Any, Any],
        *,
        chunk_size: int = 131_072,
    ) -> np.ndarray | Any:
        """Evaluate matching 2-D coordinate arrays without a tensor grid.

        ``dualate`` asks for one spline value at each pair of coordinates.  Its
        compatibility fallback calls ``__call__`` once per point, which is
        important for user-provided interpolation factories but needlessly
        rebuilds both natural-spline systems for the built-in interpolator.

        Natural tensor-product splines are separable and linear.  Precomputing
        the x, y, and mixed second derivatives therefore gives the same cubic
        polynomial at every paired point while keeping both the source-grid
        preparation and target evaluation vectorized.  Unsupported numeric or
        boundary cases return ``NotImplemented`` so callers can retain the
        fully general scalar behavior.
        """

        if (
            self.values.ndim != 2
            or self.spline_boundary != "natural"
            or self.spline_oncell
            or self.values.dtype.kind not in "fc"
            or self.values.dtype == np.dtype(np.float16)
        ):
            return NotImplemented

        x_coordinate = np.asarray(coordinates[0])
        y_coordinate = np.asarray(coordinates[1])
        if (
            x_coordinate.shape != y_coordinate.shape
            or x_coordinate.dtype.kind != "f"
            or y_coordinate.dtype.kind != "f"
        ):
            return NotImplemented

        fill_array = np.asarray(self.extrapolation_bc)
        if fill_array.ndim != 0 or fill_array.dtype.kind not in "buifc":
            return NotImplemented

        # Julia's first array dimension advances fastest.  Flattening and
        # reshaping in Fortran order keeps paired masks and values in that
        # order, including the empty-grid case.
        x_flat = x_coordinate.ravel(order="F")
        y_flat = y_coordinate.ravel(order="F")
        (
            sources,
            mapped,
            _extrapolation_targets,
            outside_masks,
            linear_axes,
            fill_value,
        ) = _map_targets(
            self.ranges,
            (x_flat, y_flat),
            self.extrapolation_bc,
        )
        if any(lower or upper for lower, upper in linear_axes):
            return NotImplemented
        if any(
            np.asarray(item).dtype == np.dtype(np.float16)
            for item in (*sources, *mapped)
        ):
            return NotImplemented

        x_mapped, y_mapped = mapped
        x_dtype = _interpolation_output_dtype(self.values, x_mapped)
        y_dtype = _interpolation_output_dtype(
            np.empty((), dtype=x_dtype), y_mapped
        )
        if x_dtype != y_dtype or y_dtype.kind not in "fc":
            return NotImplemented

        values = self.values.astype(y_dtype, copy=False)
        x_source = np.asarray(sources[0])
        y_source = np.asarray(sources[1])
        x_spacing = np.diff(x_source)
        y_spacing = np.diff(y_source)
        if np.any(x_spacing <= 0) or np.any(y_spacing <= 0):
            raise ValueError("Interpolation lattice axes must have positive steps.")

        def second_derivative(array: np.ndarray, axis: int) -> np.ndarray:
            moved = np.moveaxis(array, axis, 0)
            spacing = x_spacing if axis == 0 else y_spacing
            second = _cubic_second_derivatives(
                moved.reshape(moved.shape[0], -1), spacing, "natural"
            )
            return np.moveaxis(second.reshape(moved.shape), 0, axis)

        second_x = second_derivative(values, 0)
        second_y = second_derivative(values, 1)
        second_xy = second_derivative(second_x, 1)

        def weights(
            source: np.ndarray, target: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            interval = np.searchsorted(source, target, side="right") - 1
            interval = np.clip(interval, 0, len(source) - 2)
            lower = source[interval]
            upper = source[interval + 1]
            width = upper - lower
            a = (upper - target) / width
            b = (target - lower) / width
            second_a = (a**3 - a) * width**2 / 6
            second_b = (b**3 - b) * width**2 / 6
            return interval, a, b, second_a, second_b

        output = np.empty(x_flat.size, dtype=y_dtype)
        for start in range(0, x_flat.size, chunk_size):
            stop = min(start + chunk_size, x_flat.size)
            ix, ax, bx, cx, dx = weights(x_source, x_mapped[start:stop])
            iy, ay, by, cy, dy = weights(y_source, y_mapped[start:stop])
            ix1 = ix + 1
            iy1 = iy + 1

            value_y0 = (
                ax * values[ix, iy]
                + bx * values[ix1, iy]
                + cx * second_x[ix, iy]
                + dx * second_x[ix1, iy]
            )
            value_y1 = (
                ax * values[ix, iy1]
                + bx * values[ix1, iy1]
                + cx * second_x[ix, iy1]
                + dx * second_x[ix1, iy1]
            )
            second_y0 = (
                ax * second_y[ix, iy]
                + bx * second_y[ix1, iy]
                + cx * second_xy[ix, iy]
                + dx * second_xy[ix1, iy]
            )
            second_y1 = (
                ax * second_y[ix, iy1]
                + bx * second_y[ix1, iy1]
                + cx * second_xy[ix, iy1]
                + dx * second_xy[ix1, iy1]
            )
            output[start:stop] = (
                ay * value_y0
                + by * value_y1
                + cy * second_y0
                + dy * second_y1
            )

        output_dtype = _julia_numeric_dtype(output, fill_value)
        output = output.astype(output_dtype, copy=False)
        outside = outside_masks[0] | outside_masks[1]
        if np.any(outside):
            output[outside] = fill_value
        return output.reshape(x_coordinate.shape, order="F")

    def _evaluate_grid(self, coordinates: tuple[Any, ...]) -> np.ndarray:
        (
            sources,
            mapped,
            extrapolation_targets,
            outside_masks,
            linear_axes,
            fill_value,
        ) = _map_targets(
            self.ranges,
            coordinates,
            self.extrapolation_bc,
            oncell_periodic_axes=(
                (True,) * len(self.ranges)
                if self.spline_oncell
                else None
            ),
        )
        active_linear_extrapolation = any(
            (lower_linear or upper_linear)
            and np.any(reference != target)
            for target, reference, (lower_linear, upper_linear) in zip(
                mapped,
                extrapolation_targets,
                linear_axes,
                strict=True,
            )
        )

        if (
            self.values.dtype == np.dtype(np.float16)
            and self.values.ndim > 1
        ):
            output = _float16_cubic_tensor(
                self.values,
                sources,
                mapped,
                extrapolation_targets,
                linear_axes,
                coordinates,
                self.spline_boundary,
            )
            return _apply_fill(output, outside_masks, fill_value)

        def evaluate_component(derivative_axis: int | None = None) -> np.ndarray:
            component = self.values
            for axis_number, (
                source_axis,
                target_axis,
                extrapolation_target,
                linear_axis,
                output_target,
            ) in enumerate(
                zip(
                    sources,
                    mapped,
                    extrapolation_targets,
                    linear_axes,
                    coordinates,
                    strict=True,
                )
            ):
                component = _natural_cubic_axis(
                    component,
                    source_axis,
                    target_axis,
                    axis_number,
                    linear_extrapolation=(
                        (False, False)
                        if active_linear_extrapolation and self.values.ndim > 1
                        else linear_axis
                    ),
                    extrapolation_target=extrapolation_target,
                    output_target=np.atleast_1d(np.asarray(output_target)),
                    spline_boundary=self.spline_boundary,
                    derivative=axis_number == derivative_axis,
                )
            return component

        output = evaluate_component()
        if active_linear_extrapolation and self.values.ndim > 1:
            # Extrapolations.jl evaluates the spline and its full gradient at
            # the clamped point, then adds one displacement-gradient term per
            # Line axis.  Sequentially extrapolating tensor axes would invent
            # mixed derivative terms at corners.
            for axis_number, (target_axis, reference, line_axis) in enumerate(
                zip(mapped, extrapolation_targets, linear_axes, strict=True)
            ):
                lower_linear, upper_linear = line_axis
                outside = ((reference < target_axis) & lower_linear) | (
                    (reference > target_axis) & upper_linear
                )
                if not np.any(outside):
                    continue
                displacement = np.where(
                    outside,
                    reference - target_axis,
                    np.asarray(0, dtype=target_axis.dtype),
                )
                shape = [1] * output.ndim
                shape[axis_number] = len(displacement)
                output = output + displacement.reshape(shape) * evaluate_component(
                    axis_number
                )

        storage_dtype = _real_cubic_grid_storage_dtype(
            self.values, coordinates
        )
        if storage_dtype is not None:
            output = output.astype(storage_dtype, copy=False)

        return _apply_fill(output, outside_masks, fill_value)

    def __getitem__(self, coordinates: Any) -> Any:
        if not isinstance(coordinates, tuple):
            coordinates = (coordinates,)
        output = self._evaluate_grid(coordinates)
        index = tuple(
            0 if np.asarray(item).ndim == 0 else slice(None)
            for item in coordinates
        )
        return output[index]

    def __call__(self, *coordinates: Any) -> Any:
        return self[coordinates]


def cubic_spline_interpolation(
    ranges: Any,
    values: Any,
    *,
    bc: Any = _DEFAULT_CUBIC_BC,
    extrapolation_bc: Any = Throw(),
) -> _CubicSpline:
    """Construct a cubic spline with independent spline/extrapolation BCs.

    ``bc`` mirrors Interpolations.jl's internal cubic boundary keyword.  The
    default is its ``Line(OnGrid())`` (natural second-derivative) condition.
    Flat and periodic spline conditions require explicit ``OnGrid()`` or
    ``OnCell()`` placement, just as in the locked dependency. Bare ``Flat()``
    and ``Periodic()`` remain valid extrapolation policies.
    ``extrapolation_bc`` controls queries outside the source domain
    independently.
    """

    spline_oncell = False
    if bc is _DEFAULT_CUBIC_BC:
        spline_boundary = "natural"
    elif isinstance(bc, Flat):
        if bc.grid is None:
            raise TypeError(
                "Flat spline boundaries require Flat(OnGrid()) or "
                "Flat(OnCell())."
            )
        spline_oncell = isinstance(bc.grid, OnCell)
        spline_boundary = "flat_oncell" if spline_oncell else "flat"
    elif isinstance(bc, Periodic):
        if bc.grid is None:
            raise TypeError(
                "Periodic spline boundaries require Periodic(OnGrid()) or "
                "Periodic(OnCell())."
            )
        spline_oncell = isinstance(bc.grid, OnCell)
        spline_boundary = "periodic"
    else:
        raise TypeError(
            "bc must be Flat(OnGrid/OnCell) or "
            "Periodic(OnGrid/OnCell); omit it for Julia's default "
            "Line(OnGrid()) cubic boundary."
        )
    return _CubicSpline(
        ranges,
        values,
        extrapolation_bc=extrapolation_bc,
        spline_boundary=spline_boundary,
        spline_oncell=spline_oncell,
    )


CubicSplineInterpolation = cubic_spline_interpolation


class _LinearInterpolator:
    """Tensor-product counterpart of Interpolations.linear_interpolation."""

    def __init__(
        self,
        ranges: Any,
        values: Any,
        *,
        extrapolation_bc: Any = Throw(),
    ) -> None:
        self.values = np.asarray(values)
        self.ranges = _normalize_ranges(
            ranges, self.values, require_regular=False
        )
        if self.values.ndim != len(self.ranges) or self.values.shape != tuple(
            map(len, self.ranges)
        ):
            raise DimensionMismatch("Array size does not match interpolation lattice.")
        if any(len(axis) < 2 for axis in self.ranges):
            raise ValueError(
                "Linear interpolation does not support singleton source axes."
            )
        # The AbstractVector constructor in Interpolations.jl uses
        # Gridded(Linear()) and permits nonuniform, strictly increasing knots.
        for axis in self.ranges:
            if _logical_step(axis) is not None:
                continue
            if not np.all(_julia_differences(axis) > 0):
                raise ValueError(
                    "Interpolation lattice axes must be strictly increasing."
                )
        self.extrapolation_bc = extrapolation_bc

    def _evaluate_grid(self, coordinates: tuple[Any, ...]) -> np.ndarray:
        (
            sources,
            mapped,
            extrapolation_targets,
            outside_masks,
            linear_axes,
            fill_value,
        ) = _map_targets(self.ranges, coordinates, self.extrapolation_bc)
        output = self.values
        for axis_number, (
            source_axis,
            target_axis,
            extrapolation_target,
            linear_axis,
        ) in enumerate(
            zip(
                sources,
                mapped,
                extrapolation_targets,
                linear_axes,
                strict=True,
            )
        ):
            output = _linear_axis(
                output,
                source_axis,
                target_axis,
                axis_number,
                linear_extrapolation=linear_axis,
                extrapolation_target=extrapolation_target,
            )
        return _apply_fill(output, outside_masks, fill_value)

    def __getitem__(self, coordinates: Any) -> Any:
        if not isinstance(coordinates, tuple):
            coordinates = (coordinates,)
        output = self._evaluate_grid(coordinates)
        index = tuple(
            0 if np.asarray(item).ndim == 0 else slice(None)
            for item in coordinates
        )
        return output[index]

    def __call__(self, *coordinates: Any) -> Any:
        return self[coordinates]


def linear_interpolation(
    ranges: Any,
    values: Any,
    *,
    extrapolation_bc: Any = Throw(),
) -> _LinearInterpolator:
    """Construct a tensor-product piecewise-linear interpolator."""

    return _LinearInterpolator(
        ranges, values, extrapolation_bc=extrapolation_bc
    )


# Interpolations.jl 0.16.2 retains this deprecated capitalized constructor,
# and SLMTools brings that exact qualified name into its module namespace.
LinearInterpolation = linear_interpolation


def _integer_factor(factor: Any) -> int:
    """Require Julia's ``Int``-dispatch contract without numeric truncation."""

    if not (type(factor) is int or isinstance(factor, np.int64)):
        raise TypeError("Scale factors must be integers.")
    result = int(factor)
    if result < np.iinfo(np.int64).min or result > np.iinfo(np.int64).max:
        # Python's unbounded ``int`` is BigInt-like outside this interval;
        # Julia's resampling overloads are declared specifically for Int64.
        raise TypeError("Scale factors must fit Julia's signed Int64 type.")
    return result


def _factor_tuple(
    factor: Any,
    ndim: int,
    *,
    require_positive: bool = True,
) -> tuple[int, ...]:
    if type(factor) is int or isinstance(factor, np.int64):
        factors = (_integer_factor(factor),) * ndim
    else:
        try:
            raw_factors = tuple(factor)
        except TypeError as error:
            raise TypeError("Scale factors must be integers.") from error
        factors = tuple(_integer_factor(item) for item in raw_factors)
        if len(factors) != ndim:
            raise DimensionMismatch("Scale factors must match lattice dimension.")
    if require_positive and any(item <= 0 for item in factors):
        raise DomainError("Scale factors must be positive.")
    return factors


def _downsample_axis(axis: Any, factor: Any) -> LatticeAxis:
    factor = _integer_factor(factor)
    values = np.asarray(axis)
    if factor <= 0:
        raise DomainError("Downsample factor must be positive.")
    if len(values) % factor:
        raise DomainError(
            "downsample: Downsample factor n does not divide the length of the range r."
        )
    if len(values) == 0:
        # Julia evaluates ``r[n]`` and ``r[1]`` while forming the block
        # center, so an empty source raises BoundsError.
        raise IndexError("Lattice axis index out of bounds")
    step = (
        _regular_step(axis, require_positive=False)
        if isinstance(axis, LatticeAxis)
        else _step(axis)
    )
    exact_axis = values.dtype.kind == "O" and values.size and all(
        isinstance(item, (Fraction, Decimal, _MPFR, _MPQ, _MPZ))
        or (type(item) is int and not np.iinfo(np.int64).min <= item <= np.iinfo(np.int64).max)
        for item in values.flat
    )
    if exact_axis:
        values_work = values.astype(object, copy=False)
    else:
        dtype = _julia_numeric_dtype(values, step)
        values_work = values.astype(dtype, copy=False)
    endpoint_sum = _julia_array_scalar_operation(
        values_work[factor - 1], values_work[0], np.add
    )
    start = _julia_array_scalar_operation(
        endpoint_sum, 2, np.divide
    ).reshape(())[()]
    output_step = _julia_array_scalar_operation(
        step, factor, np.multiply
    ).reshape(())[()]
    output_length = len(values) // factor
    offsets = _axis(range(output_length))
    scaled = _logical_axis_scalar_operation(
        offsets, output_step, np.multiply
    )
    result = _logical_axis_scalar_operation(scaled, start, np.add)
    return _with_axis_length_kind(
        result, getattr(_axis(axis), "_length_kind", "int64")
    )


def _downsample_lattice(lattice: Any, factor: Any) -> Lattice:
    axes = as_lattice(lattice)
    factors = _factor_tuple(factor, len(axes))
    return tuple(
        _downsample_axis(axis, item)
        for axis, item in zip(axes, factors, strict=True)
    )


def _upsample_axis(axis: Any, factor: Any) -> LatticeAxis:
    factor = _integer_factor(factor)
    values = np.asarray(axis)
    step = (
        _regular_step(axis, require_positive=False)
        if isinstance(axis, LatticeAxis)
        else _step(axis)
    )
    # Julia computes ``step(r) / n`` *before* combining it with the Float64
    # index-centering term.  Low-precision division therefore rounds in the
    # source dtype, while the final coordinates are widened to Float64.
    with np.errstate(divide="ignore", invalid="ignore"):
        output_step = _julia_array_scalar_operation(
            step, factor, np.divide
        ).reshape(())[()]
    # The Julia source spells this index axis ``1:length(r)*n``, a UnitRange.
    # Its scalar subtraction overload is observably different from the
    # explicit unit-step StepRange produced by ``range(start, step, length)``.
    k = _axis(range(1, len(values) * factor + 1))
    center = np.float64((1 + factor) / 2)
    centered = _logical_axis_scalar_operation(k, center, np.subtract)
    scaled = _logical_axis_scalar_operation(
        centered, output_step, np.multiply
    )
    if len(values) == 0:
        # The final ``.+ r[1]`` in Julia raises BoundsError for an empty
        # source even though the generated ordinal range is also empty.
        raise IndexError("Lattice axis index out of bounds")
    result = _logical_axis_scalar_operation(scaled, values[0], np.add)
    return _with_axis_length_kind(
        result, getattr(_axis(axis), "_length_kind", "int64")
    )


def _upsample_lattice(lattice: Any, factor: Any) -> Lattice:
    axes = as_lattice(lattice)
    factors = _factor_tuple(
        factor, len(axes), require_positive=False
    )
    return tuple(
        _upsample_axis(axis, item)
        for axis, item in zip(axes, factors, strict=True)
    )


def _index_lattice(shape: tuple[int, ...]) -> Lattice:
    # Julia's implicit array lattice is 1:size, not zero based.
    return tuple(_axis(np.arange(1, size + 1), 1) for size in shape)


def _builtin_range_indexing_succeeds(axis: LatticeAxis) -> bool:
    """Whether Julia's interpolation ``getindex`` accepts this range family."""

    dtype_kind = np.asarray(axis).dtype.kind
    if dtype_kind == "i":
        return True
    return dtype_kind == "u" and getattr(axis, "_range_kind", None) == "srl"


def _index_numpy_array_like_julia(
    array: np.ndarray, target_lattice: Lattice
) -> np.ndarray:
    """Apply Julia's one-based Cartesian-product indexing to a NumPy array."""

    index_count = len(target_lattice)
    if index_count == 0:
        if array.ndim == 0:
            return array[()]
        raise DimensionMismatch(
            "A non-scalar custom interpolation array requires an index."
        )

    # With fewer indices Julia linearly collapses all dimensions consumed by
    # the final index. With extra indices it appends singleton dimensions.
    # Reshape in column-major order to retain Julia's linear-index ordering.
    if index_count < array.ndim:
        effective_shape = (
            array.shape[: index_count - 1]
            + (int(np.prod(array.shape[index_count - 1 :])),)
        )
        effective = np.reshape(array, effective_shape, order="F")
    elif index_count > array.ndim:
        effective_shape = array.shape + (1,) * (index_count - array.ndim)
        effective = np.reshape(array, effective_shape)
    else:
        effective = array

    indexes: list[np.ndarray] = []
    for dimension, (axis, size) in enumerate(
        zip(target_lattice, effective.shape, strict=True), start=1
    ):
        converted = np.empty(len(axis), dtype=np.intp)
        for position, item in enumerate(np.asarray(axis)):
            try:
                one_based = integer_index(item)
            except TypeError as error:
                raise TypeError(
                    f"invalid Julia array index {item!r} in dimension {dimension}"
                ) from error
            if one_based < 1 or one_based > size:
                raise IndexError(
                    f"Julia array index {one_based} is out of bounds for "
                    f"dimension {dimension} with size {size}"
                )
            converted[position] = one_based - 1
        indexes.append(converted)
    return np.asarray(effective[np.ix_(*indexes)])


def _interpolate(
    values: Any,
    source: Any,
    target: Any,
    interpolation: Callable[..., Any],
    boundary: Any,
) -> Any:
    source_lattice = as_lattice(source)
    target_lattice = as_lattice(target)
    array = np.asarray(values)
    if array.shape != tuple(map(len, source_lattice)):
        raise DomainError(
            "Size of array does not match size of interpolation source lattice."
        )
    storage_state = None
    callback_values = values
    if hasattr(values, "_storage") and hasattr(values, "_state"):
        # A custom factory receives Julia's ordinary mutable ``f.data`` Array,
        # including NumPy scalar-broadcast semantics.  Use the authoritative
        # storage and synchronize every retained checked facade afterward.
        callback_values = values._storage()
        storage_state = values._state()
    try:
        interpolator = interpolation(
            source_lattice,
            callback_values,
            extrapolation_bc=boundary,
        )
    finally:
        if storage_state is not None:
            storage_state.changed()
    if (
        isinstance(interpolator, (_CubicSpline, _LinearInterpolator))
        and not any(len(axis) == 0 for axis in target_lattice)
        and any(
            not _builtin_range_indexing_succeeds(axis)
            for axis in target_lattice
        )
    ):
        raise NotImplementedError(
            "the audited Julia built-in resampling path does not produce "
            "defined values for this target range family"
        )
    # Julia's source uses range ``getindex`` here.  A custom interpolation
    # factory is therefore supported only when its result implements the
    # corresponding subscription operation; calling it would invent a
    # successful fallback that the source never attempts.
    output = (
        _index_numpy_array_like_julia(interpolator, target_lattice)
        if isinstance(interpolator, np.ndarray)
        else interpolator[target_lattice]
    )
    return output


def _julia_zero_for_array(values: Any) -> Any:
    """Return ``zero(eltype(values))`` without erasing object element types."""

    array = np.asarray(values)
    if array.dtype.kind == "O" and array.size:
        return _julia_typed_zero(array.ravel(order="F")[0])
    return np.zeros((), dtype=array.dtype)[()]


def downsample(
    value: Any,
    *arguments: Any,
    interpolation: Any = _OMITTED,
    bc: Any = _OMITTED,
) -> Any:
    """Downsample an axis/lattice/array/field using Julia-compatible overloads."""

    if isinstance(value, LatticeField):
        _require_julia_numeric_array(value.data, "downsample field")
        if len(arguments) != 1:
            raise TypeError("downsample(field, target_or_factor) expected.")
        specification = arguments[0]
        if (
            value.ndim == 0
            and isinstance(specification, tuple)
            and len(specification) == 0
        ):
            raise TypeError(
                "zero-dimensional field resampling with an empty tuple is "
                "ambiguous in Julia"
            )
        target = (
            as_lattice(specification)
            if _looks_like_lattice(specification)
            else _downsample_lattice(value.L, specification)
        )
        boundary = _julia_zero_for_array(value.data) if bc is _OMITTED else bc
        interpolation_factory = (
            cubic_spline_interpolation
            if interpolation is _OMITTED
            else interpolation
        )
        data = _interpolate(
            value.data,
            value.L,
            target,
            interpolation_factory,
            boundary,
        )
        return LatticeField[value.field_type](data, target, value.flambda)
    if isinstance(value, (LatticeAxis, range)):
        if interpolation is not _OMITTED:
            raise TypeError(
                "downsample(axis, factor) does not accept interpolation"
            )
        if bc is not _OMITTED:
            raise TypeError("downsample(axis, factor) does not accept bc")
        if len(arguments) != 1:
            raise TypeError("downsample(axis, factor) expected.")
        return _downsample_axis(value, arguments[0])
    if _looks_like_lattice(value):
        if interpolation is not _OMITTED:
            raise TypeError(
                "downsample(lattice, factor) does not accept interpolation"
            )
        if bc is not _OMITTED:
            raise TypeError("downsample(lattice, factor) does not accept bc")
        if len(arguments) != 1:
            raise TypeError("downsample(lattice, factor) expected.")
        return _downsample_lattice(value, arguments[0])

    if not isinstance(value, (list, np.ndarray)):
        raise TypeError(
            "downsample array input must be a dense array or list literal"
        )

    array = (
        _julia_literal_array(value)
        if isinstance(value, list)
        else np.asarray(value)
    )
    _require_julia_numeric_array(array, "downsample array")
    boundary = _julia_zero_for_array(array) if bc is _OMITTED else bc
    if len(arguments) == 1:
        source = _index_lattice(array.shape)
        target = _downsample_lattice(source, arguments[0])
    elif len(arguments) == 2:
        source, target = map(as_lattice, arguments)
    else:
        raise TypeError("downsample(array, factor) or (array, source, target).")
    interpolation_factory = (
        cubic_spline_interpolation
        if interpolation is _OMITTED
        else interpolation
    )
    return _interpolate(
        array,
        source,
        target,
        interpolation_factory,
        boundary,
    )


def _empty_coarsen_default_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Return Julia's inferred default-reducer dtype for an empty result."""

    if dtype.kind in "biu":
        return np.dtype(np.float64)
    return dtype


def _empty_coarsen_reducer_dtype(
    dtype: np.dtype[Any],
    reducer: Any,
) -> np.dtype[Any]:
    """Infer an empty coarsening result without evaluating ``reducer``.

    Julia obtains this type from compiler inference for the comprehension.
    Python exposes less static callable information, so common NumPy reductions
    are handled explicitly. An arbitrary Python callable has no result value
    when it is not called and therefore maps to object storage.
    """

    if reducer is _OMITTED or reducer is np.mean:
        return _empty_coarsen_default_dtype(dtype)

    if any(
        reducer is candidate
        for candidate in (np.max, np.amax, np.min, np.amin)
    ):
        return dtype

    if reducer is np.sum:
        if dtype.kind in "bi":
            return np.dtype(np.int64)
        if dtype.kind == "u":
            return np.dtype(np.uint64)
        return dtype

    return np.dtype(object)


def coarsen(
    value: Any,
    factor: Any,
    *,
    reducer: Any = _OMITTED,
) -> Any:
    """Reduce disjoint superpixels; the default reducer is arithmetic mean."""

    if isinstance(value, LatticeField):
        _require_julia_numeric_array(value.data, "coarsen field")
        target = _downsample_lattice(value.L, factor)
        data = coarsen(value.data, factor, reducer=reducer)
        return LatticeField[value.field_type](data, target, value.flambda)
    if not isinstance(value, (list, np.ndarray)):
        raise TypeError(
            "coarsen array input must be a dense array or list literal"
        )
    array = (
        _julia_literal_array(value)
        if isinstance(value, list)
        else np.asarray(value)
    )
    _require_julia_numeric_array(array, "coarsen array")
    factors = _factor_tuple(factor, array.ndim)
    if any(size % item for size, item in zip(array.shape, factors, strict=True)):
        raise DomainError(
            "coarsen: Downsample factors ns do not divide the size of array x."
        )
    output_shape = tuple(
        size // item for size, item in zip(array.shape, factors, strict=True)
    )
    if any(size == 0 for size in output_shape):
        return np.empty(
            output_shape,
            dtype=_empty_coarsen_reducer_dtype(array.dtype, reducer),
        )
    extrema_reducer = any(
        reducer is candidate
        for candidate in (np.max, np.amax, np.min, np.amin)
    )
    complex_values = array.dtype.kind == "c" or (
        array.dtype.kind == "O"
        and any(
            isinstance(
                item,
                (_MPC, _MPFRComplex, complex, np.complexfloating),
            )
            for item in array.flat
        )
    )
    if extrema_reducer and complex_values:
        raise TypeError("Julia extrema reducers cannot order complex values")
    default_reducer = reducer is _OMITTED
    sum_reducer = reducer is np.sum
    object_output = np.empty(output_shape, dtype=object)
    indices = (
        tuple(reversed(index))
        for index in np.ndindex(tuple(reversed(output_shape)))
    )
    for index in indices:
        block = tuple(
            slice(i * width, (i + 1) * width)
            for i, width in zip(index, factors, strict=True)
        )
        # Julia's ``x[I .+ box]`` is advanced indexing and allocates a dense
        # block for each callback.  A Python basic slice is a mutable view;
        # detach it so a reducer cannot mutate the caller's source array.
        superpixel = np.array(array[block], copy=True, order="F")
        if default_reducer or sum_reducer:
            total = _julia_sum(superpixel)
            if default_reducer:
                if isinstance(total, Fraction):
                    total = _fraction_int64_divide(
                        total, Fraction(superpixel.size, 1)
                    )
                elif isinstance(total, Decimal):
                    with localcontext() as context:
                        _enable_decimal_nonfinite(context)
                        total = total / Decimal(superpixel.size)
                elif isinstance(total, _MPZ):
                    with _bigfloat_context():
                        total = _to_mpfr(total) / _to_mpfr(superpixel.size)
                else:
                    divided = _julia_array_scalar_operation(
                        total,
                        np.int64(superpixel.size),
                        np.divide,
                    )
                    total = divided.reshape(())[()]
            object_output[index] = total
        else:
            object_output[index] = reducer(superpixel)
    collected = _julia_collect_comprehension_results(
        object_output.ravel(order="F")
    )
    return collected.reshape(output_shape, order="F")


def upsample(
    value: Any,
    *arguments: Any,
    interpolation: Any = _OMITTED,
    bc: Any = _OMITTED,
) -> Any:
    """Upsample an axis/lattice/array/field using Julia-compatible overloads."""

    if isinstance(value, LatticeField):
        _require_julia_numeric_array(value.data, "upsample field")
        if len(arguments) != 1:
            raise TypeError("upsample(field, target_or_factor) expected.")
        specification = arguments[0]
        if (
            value.ndim == 0
            and isinstance(specification, tuple)
            and len(specification) == 0
        ):
            raise TypeError(
                "zero-dimensional field resampling with an empty tuple is "
                "ambiguous in Julia"
            )
        target = (
            as_lattice(specification)
            if _looks_like_lattice(specification)
            else _upsample_lattice(value.L, specification)
        )
        boundary = _julia_zero_for_array(value.data) if bc is _OMITTED else bc
        interpolation_factory = (
            cubic_spline_interpolation
            if interpolation is _OMITTED
            else interpolation
        )
        data = _interpolate(
            value.data,
            value.L,
            target,
            interpolation_factory,
            boundary,
        )
        return LatticeField[value.field_type](data, target, value.flambda)
    if isinstance(value, (LatticeAxis, range)):
        if interpolation is not _OMITTED:
            raise TypeError(
                "upsample(axis, factor) does not accept interpolation"
            )
        if bc is not _OMITTED:
            raise TypeError("upsample(axis, factor) does not accept bc")
        if len(arguments) != 1:
            raise TypeError("upsample(axis, factor) expected.")
        return _upsample_axis(value, arguments[0])
    if _looks_like_lattice(value):
        if interpolation is not _OMITTED:
            raise TypeError(
                "upsample(lattice, factor) does not accept interpolation"
            )
        if bc is not _OMITTED:
            raise TypeError("upsample(lattice, factor) does not accept bc")
        if len(arguments) != 1:
            raise TypeError("upsample(lattice, factor) expected.")
        return _upsample_lattice(value, arguments[0])

    if not isinstance(value, (list, np.ndarray)):
        raise TypeError(
            "upsample array input must be a dense array or list literal"
        )

    array = (
        _julia_literal_array(value)
        if isinstance(value, list)
        else np.asarray(value)
    )
    _require_julia_numeric_array(array, "upsample array")
    boundary = _julia_zero_for_array(array) if bc is _OMITTED else bc
    if len(arguments) == 1:
        source = _index_lattice(array.shape)
        target = _upsample_lattice(source, arguments[0])
    elif len(arguments) == 2:
        source, target = map(as_lattice, arguments)
    else:
        raise TypeError("upsample(array, factor) or (array, source, target).")
    interpolation_factory = (
        cubic_spline_interpolation
        if interpolation is _OMITTED
        else interpolation
    )
    return _interpolate(
        array,
        source,
        target,
        interpolation_factory,
        boundary,
    )


__all__ = [
    "CubicSplineInterpolation",
    "Flat",
    "Linear",
    "LinearInterpolation",
    "Periodic",
    "Throw",
    "coarsen",
    "cubic_spline_interpolation",
    "downsample",
    "upsample",
]
