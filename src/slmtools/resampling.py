"""Interpolation, block coarsening, and lattice resampling.

The default interpolator is a dependency-free tensor product of natural cubic
splines, matching Interpolations.jl's ``Cubic(Line(OnGrid()))`` construction.
The ``bc`` argument controls the cubic endpoint equations, while
``extrapolation_bc`` selects numeric fill, ``Flat``, ``Periodic``, ``Linear``,
or throwing behavior outside the source grid.

Unlike the Julia methods, nonpositive scale factors are rejected explicitly;
their original behavior was an assortment of divide/bounds errors rather than
a useful supported convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from fractions import Fraction
from numbers import Integral, Number
from typing import Any, Callable

import numpy as np

from .lattice_field import (
    DimensionMismatch,
    DomainError,
    FieldVal,
    Lattice,
    LatticeAxis,
    LatticeField,
    _axis,
    _julia_array_scalar_operation,
    _logical_axis_scalar_operation,
    as_lattice,
)
from .lattice_utils import _looks_like_lattice, _step


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


def _exact_domain(values: Any, target: Any) -> type[Any] | None:
    """Select Python's analogue of Julia Rational/BigFloat promotion."""

    value_items = _object_values(values)
    target_items = _object_values(target)
    all_items = value_items + target_items
    if any(isinstance(item, Decimal) for item in all_items):
        return Decimal
    if any(isinstance(item, Fraction) for item in value_items):
        target_array = np.asarray(target)
        target_is_exact = (
            all(isinstance(item, (Integral, Fraction)) for item in target_items)
            if target_array.dtype == np.dtype(object)
            else target_array.dtype.kind in "bui"
        )
        if target_is_exact:
            return Fraction
    return None


def _convert_exact(value: Any, domain: type[Any]) -> Any:
    """Convert a scalar without silently passing through binary Float64."""

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
    differences = np.diff(values)
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
        hint = np.diff(values)[0]
    if hint == 0 or (require_positive and hint < 0):
        raise ValueError("Interpolation lattice axes must have positive steps.")
    if bool(getattr(axis, "_step_hint_is_logical", False)):
        return hint
    differences = np.diff(values)
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
    values: np.ndarray, boundary: str
) -> np.ndarray:
    """Prefilter Float16 values exactly as Interpolations.jl 0.16.2 does."""

    source_length = values.shape[0]
    padded = boundary != "periodic"
    size = source_length + 2 if padded else source_length
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
    base_inverse_rows = _float16_tridiagonal_solve(factor, rows)
    reduced = np.add(
        inverse_values,
        _float16_matmul(columns, base_inverse_rows),
        dtype=np.float16,
    )
    woodbury_inverse = _float16_dense_solve(
        reduced, np.eye(correction_count, dtype=np.float16)
    )

    if padded:
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
    correction = _float16_tridiagonal_solve(factor, correction)
    return np.subtract(solution, correction, dtype=np.float16)


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


def _float16_cubic_axis(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    axis: int,
    *,
    linear_extrapolation: bool | tuple[bool, bool],
    extrapolation_target: np.ndarray | None = None,
    spline_boundary: str,
) -> np.ndarray:
    """Evaluate a cubic whose stored coefficients use Float16 arithmetic."""

    moved = np.moveaxis(np.asarray(values), axis, 0)
    flat = moved.reshape(len(source), -1)
    coefficients = _float16_cubic_coefficients(flat, spline_boundary)
    output_dtype = _interpolation_output_dtype(moved, target)
    weight_dtype = np.dtype(target.dtype)
    source_work = source.astype(weight_dtype, copy=False)
    target_work = target.astype(weight_dtype, copy=False)
    spacing = np.subtract(source_work[1], source_work[0], dtype=weight_dtype)
    lower = source_work[0]
    upper = source_work[-1]
    if spline_boundary == "periodic":
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
    spline_boundary: str = "natural",
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
    if exact_domain is not None:
        moved = _as_exact_array(moved, exact_domain)
        source = _as_exact_array(source, exact_domain)
        target = _as_exact_array(target, exact_domain)
    coordinate_dtype = _coordinate_dtype(source, target)
    coefficient_dtype = _interpolation_coefficient_dtype(moved)
    output_dtype = _interpolation_output_dtype(moved, target)
    dtype = output_dtype
    source = source.astype(coordinate_dtype, copy=False)
    target = target.astype(coordinate_dtype, copy=False)
    if (
        coefficient_dtype == np.dtype(np.float16)
        and n > 1
        and spline_boundary != "flat_oncell"
    ):
        return _float16_cubic_axis(
            np.asarray(values),
            source,
            target,
            axis,
            linear_extrapolation=linear_extrapolation,
            extrapolation_target=extrapolation_target,
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
    if lower_linear or upper_linear:
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
    if exact_domain is not None:
        moved = _as_exact_array(moved, exact_domain)
        source = _as_exact_array(source, exact_domain)
        target = _as_exact_array(target, exact_domain)
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
    Any | None,
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
    fill_value: Any | None = None
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
        elif isinstance(policy, Number) or np.isscalar(policy) or isinstance(
            policy, tuple
        ):
            if fill_value is None:
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
    fill_value: Any | None,
) -> np.ndarray:
    """Apply a scalar fill boundary to the tensor-product result."""

    if fill_value is None or not any(np.any(mask) for mask in outside_masks):
        return output
    output_dtype = (
        np.dtype(object)
        if output.dtype == np.dtype(object) or isinstance(fill_value, tuple)
        else _julia_numeric_dtype(output, fill_value)
    )
    output = output.astype(output_dtype, copy=False)
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
            output = _natural_cubic_axis(
                output,
                source_axis,
                target_axis,
                axis_number,
                linear_extrapolation=linear_axis,
                extrapolation_target=extrapolation_target,
                spline_boundary=self.spline_boundary,
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
            if not np.all(np.diff(np.asarray(axis)) > 0):
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


def _factor_tuple(factor: Any, ndim: int) -> tuple[int, ...]:
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
    if any(item <= 0 for item in factors):
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
    dtype = _julia_numeric_dtype(values, step)
    values_work = values.astype(dtype, copy=False)
    endpoint_sum = _julia_array_scalar_operation(
        np.asarray(values_work[factor - 1]), values_work[0], np.add
    )
    start = _julia_array_scalar_operation(
        endpoint_sum, 2, np.divide
    ).reshape(())[()]
    output_step = _julia_array_scalar_operation(
        np.asarray(step), factor, np.multiply
    ).reshape(())[()]
    output_length = len(values) // factor
    offsets = _axis(range(output_length))
    scaled = _logical_axis_scalar_operation(
        offsets, output_step, np.multiply
    )
    return _logical_axis_scalar_operation(scaled, start, np.add)


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
    if factor <= 0:
        raise DomainError("Upsample factor must be positive.")
    step = (
        _regular_step(axis, require_positive=False)
        if isinstance(axis, LatticeAxis)
        else _step(axis)
    )
    # Julia computes ``step(r) / n`` *before* combining it with the Float64
    # index-centering term.  Low-precision division therefore rounds in the
    # source dtype, while the final coordinates are widened to Float64.
    output_step = _julia_array_scalar_operation(
        np.asarray(step), factor, np.divide
    ).reshape(())[()]
    k = LatticeAxis.from_start_step(
        np.int64(1),
        np.int64(1),
        len(values) * factor,
    )
    center = np.float64((1 + factor) / 2)
    centered = _logical_axis_scalar_operation(k, center, np.subtract)
    scaled = _logical_axis_scalar_operation(
        centered, output_step, np.multiply
    )
    if len(values) == 0:
        # The final ``.+ r[1]`` in Julia raises BoundsError for an empty
        # source even though the generated ordinal range is also empty.
        raise IndexError("Lattice axis index out of bounds")
    return _logical_axis_scalar_operation(scaled, values[0], np.add)


def _upsample_lattice(lattice: Any, factor: Any) -> Lattice:
    axes = as_lattice(lattice)
    factors = _factor_tuple(factor, len(axes))
    return tuple(
        _upsample_axis(axis, item)
        for axis, item in zip(axes, factors, strict=True)
    )


def _index_lattice(shape: tuple[int, ...]) -> Lattice:
    # Julia's implicit array lattice is 1:size, not zero based.
    return tuple(_axis(np.arange(1, size + 1), 1) for size in shape)


def _interpolate(
    values: Any,
    source: Any,
    target: Any,
    interpolation: Callable[..., Any],
    boundary: Any,
) -> np.ndarray:
    source_lattice = as_lattice(source)
    target_lattice = as_lattice(target)
    array = np.asarray(values)
    if array.shape != tuple(map(len, source_lattice)):
        raise DomainError(
            "Size of array does not match size of interpolation source lattice."
        )
    interpolator = interpolation(
        source_lattice, array, extrapolation_bc=boundary
    )
    try:
        output = interpolator[target_lattice]
    except TypeError:
        output = interpolator(*target_lattice)
    return np.asarray(output)


def downsample(
    value: Any,
    *arguments: Any,
    interpolation: Callable[..., Any] = cubic_spline_interpolation,
    bc: Any = None,
) -> Any:
    """Downsample an axis/lattice/array/field using Julia-compatible overloads."""

    if isinstance(value, LatticeField):
        if len(arguments) != 1:
            raise TypeError("downsample(field, target_or_factor) expected.")
        specification = arguments[0]
        target = (
            as_lattice(specification)
            if _looks_like_lattice(specification)
            else _downsample_lattice(value.L, specification)
        )
        boundary = np.zeros((), dtype=value.dtype)[()] if bc is None else bc
        data = _interpolate(value.data, value.L, target, interpolation, boundary)
        return LatticeField[value.field_type](data, target, value.flambda)
    if isinstance(value, (LatticeAxis, range)):
        if len(arguments) != 1:
            raise TypeError("downsample(axis, factor) expected.")
        return _downsample_axis(value, arguments[0])
    if _looks_like_lattice(value):
        if len(arguments) != 1:
            raise TypeError("downsample(lattice, factor) expected.")
        return _downsample_lattice(value, arguments[0])

    array = np.asarray(value)
    boundary = np.zeros((), dtype=array.dtype)[()] if bc is None else bc
    if len(arguments) == 1:
        source = _index_lattice(array.shape)
        target = _downsample_lattice(source, arguments[0])
    elif len(arguments) == 2:
        source, target = map(as_lattice, arguments)
    else:
        raise TypeError("downsample(array, factor) or (array, source, target).")
    return _interpolate(array, source, target, interpolation, boundary)


def coarsen(
    value: Any,
    factor: Any,
    *,
    reducer: Callable[[np.ndarray], Any] | None = None,
) -> Any:
    """Reduce disjoint superpixels; the default reducer is arithmetic mean."""

    if isinstance(value, LatticeField):
        target = _downsample_lattice(value.L, factor)
        data = coarsen(value.data, factor, reducer=reducer)
        return LatticeField[value.field_type](data, target, value.flambda)
    array = np.asarray(value)
    factors = _factor_tuple(factor, array.ndim)
    if any(size % item for size, item in zip(array.shape, factors, strict=True)):
        raise DomainError(
            "coarsen: Downsample factors ns do not divide the size of array x."
        )
    if reducer is None:
        reducer = lambda block: np.sum(block) / block.size
    output_shape = tuple(
        size // item for size, item in zip(array.shape, factors, strict=True)
    )
    object_output = np.empty(output_shape, dtype=object)
    for index in np.ndindex(output_shape):
        block = tuple(
            slice(i * width, (i + 1) * width)
            for i, width in zip(index, factors, strict=True)
        )
        object_output[index] = reducer(array[block])
    if object_output.size == 0:
        # Julia infers the comprehension element type even when its
        # Cartesian index set is empty.  NumPy has no callable return-type
        # inference, so evaluate the reducer once on a one-sample-per-axis
        # type probe.  The probe is deliberately independent of the possibly
        # enormous coarsening factors: only the reducer's scalar result dtype
        # is needed, not a materialized superpixel.
        #
        # This preserves, for example, Float32/ComplexF32 default means and
        # custom reducers returning Int16 or ComplexF32 instead of silently
        # manufacturing a Float64 empty result.
        probe = np.zeros((1,) * array.ndim, dtype=array.dtype)
        try:
            reduced_probe = reducer(probe)
        except Exception:
            # A Python callable can be value- or shape-dependent in ways that
            # Julia's inferred comprehension element type is not.  Object is
            # the only non-narrowing representation when a safe probe cannot
            # determine that type.
            return np.empty(output_shape, dtype=object)
        reduced_array = np.asarray(reduced_probe)
        if reduced_array.ndim != 0:
            return np.empty(output_shape, dtype=object)
        return np.empty(output_shape, dtype=reduced_array.dtype)
    sample_values = list(object_output.flat)
    dtype = np.result_type(*[np.asarray(item).dtype for item in sample_values])
    output = np.empty(output_shape, dtype=dtype)
    for index in np.ndindex(output_shape):
        output[index] = object_output[index]
    return output


def upsample(
    value: Any,
    *arguments: Any,
    interpolation: Callable[..., Any] = cubic_spline_interpolation,
    bc: Any = None,
) -> Any:
    """Upsample an axis/lattice/array/field using Julia-compatible overloads."""

    if isinstance(value, LatticeField):
        if len(arguments) != 1:
            raise TypeError("upsample(field, target_or_factor) expected.")
        specification = arguments[0]
        target = (
            as_lattice(specification)
            if _looks_like_lattice(specification)
            else _upsample_lattice(value.L, specification)
        )
        boundary = np.zeros((), dtype=value.dtype)[()] if bc is None else bc
        data = _interpolate(value.data, value.L, target, interpolation, boundary)
        return LatticeField[value.field_type](data, target, value.flambda)
    if isinstance(value, (LatticeAxis, range)):
        if len(arguments) != 1:
            raise TypeError("upsample(axis, factor) expected.")
        return _upsample_axis(value, arguments[0])
    if _looks_like_lattice(value):
        if len(arguments) != 1:
            raise TypeError("upsample(lattice, factor) expected.")
        return _upsample_lattice(value, arguments[0])

    array = np.asarray(value)
    boundary = np.zeros((), dtype=array.dtype)[()] if bc is None else bc
    if len(arguments) == 1:
        source = _index_lattice(array.shape)
        target = _upsample_lattice(source, arguments[0])
    elif len(arguments) == 2:
        source, target = map(as_lattice, arguments)
    else:
        raise TypeError("upsample(array, factor) or (array, source, target).")
    return _interpolate(array, source, target, interpolation, boundary)


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
