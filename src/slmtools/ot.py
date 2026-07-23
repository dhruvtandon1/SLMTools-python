"""Optimal-transport phase generation and phase-diversity estimation.

The dense solver deliberately reproduces OptimalTransport.jl 0.3.20's
``SinkhornGibbs`` initialization, update order, defaults, and strict stopping
criterion.  Arrays are flattened and reshaped in Fortran order wherever the
Julia implementation traverses ``CartesianIndices``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any
import warnings

import numpy as np

from .lattice_field import (
    ComplexAmp,
    Intensity,
    LatticeField,
    Modulus,
    RealPhase,
    _as_decimal_approx,
    _as_decimal_array,
    _decimal_rtol,
    _axis,
    _julia_add_sum,
    _julia_assignment_values,
    _julia_array_array_operation,
    _julia_array_scalar_operation,
    _julia_literal_array,
    _julia_rtol,
    _is_real_number,
    _object_contains_decimal,
    as_lattice,
    square,
)
from .dual_lattices import isft as _field_isft
from .lattice_utils import _step as _exact_step
from .ift import (
    _quadratic_linear_phase,
    _literal_square,
    _scalar_operation,
    _dual_shift_lattice,
    _isft_array,
    _ldot,
    _make_field,
    _is_numeric_data,
    _is_real_numeric_data,
    _r2,
    _same_lattice,
    _sft_array,
    _step,
    _tag_is,
    _to_dim,
)


__all__ = [
    "getCostMatrix",
    "pdCostMatrix",
    "mapify",
    "hyperSum",
    "hyperSum2",
    "scalarPotentialN",
    "normalizeDistribution",
    "otPhase",
    "pdotPhase",
    "pdotBeamEstimate",
    "SinkhornIterBase",
    "SinkhornConvN",
    "dualToGradients",
    "otQuickPhase",
    "otPhase2",
]


def _as_lattice(
    lattice: Sequence[Any],
    *,
    allow_empty_axes: bool = False,
) -> tuple[np.ndarray, ...]:
    # Keep LatticeAxis logical-step metadata; singleton axes have no second
    # coordinate from which their valid range step could be reconstructed.
    axes = as_lattice(lattice)
    if not axes or any(
        axis.ndim != 1 or (axis.size == 0 and not allow_empty_axes)
        for axis in axes
    ):
        requirement = (
            "a nonempty tuple of 1-D axes"
            if allow_empty_axes
            else "a nonempty tuple of nonempty 1-D axes"
        )
        raise ValueError(f"a lattice must be {requirement}")
    return axes


def _point_components(lattice: Sequence[Any]) -> tuple[np.ndarray, ...]:
    """List each Cartesian coordinate component in Julia traversal order.

    Components deliberately remain separate. ``column_stack`` would first
    force heterogeneous lattice dimensions through NumPy's common-dtype
    rules, while Julia evaluates each coordinate difference/product in its
    own element type and promotes only when those terms are combined.
    """

    # Cost-matrix comprehensions accept empty AbstractRanges.  Their default
    # ``maximum`` normalization subsequently fails on the empty matrix, while
    # a user-supplied normalization may validly return an empty result.
    axes = _as_lattice(lattice, allow_empty_axes=True)
    shape = tuple(len(axis) for axis in axes)
    components: list[np.ndarray] = []
    for dimension, axis in enumerate(axes):
        view_shape = [1] * len(axes)
        view_shape[dimension] = len(axis)
        grid = np.broadcast_to(np.asarray(axis).reshape(view_shape), shape)
        components.append(np.asarray(grid).ravel(order="F"))
    return tuple(components)


def _sum_cost_terms(terms: Sequence[np.ndarray]) -> np.ndarray:
    """Sum squared-coordinate terms with Julia's reduction widening."""

    if not terms:
        return np.asarray(0.0)
    total: np.ndarray | None = None
    for term in terms:
        value = np.asarray(term)
        # Base.sum/add_sum widens small machine integers even for a one-element
        # generator. Apply that widening per term before heterogeneous
        # dimensions are combined.
        if value.dtype.kind in "bi" and value.dtype.itemsize < 8:
            value = value.astype(np.int64)
        elif value.dtype.kind == "u" and value.dtype.itemsize < 8:
            value = value.astype(np.uint64)
        total = (
            value
            if total is None
            else _julia_array_array_operation(total, value, np.add)
        )
    assert total is not None
    return total


def _cost_matrix_terms(
    source: Sequence[Any],
    target: Sequence[Any],
    *,
    target_scale: Any | None = None,
) -> np.ndarray:
    """Evaluate dense squared distance without cross-axis pre-coercion."""

    source_components = _point_components(source)
    target_components = _point_components(target)
    terms: list[np.ndarray] = []
    for source_component, target_component in zip(
        source_components, target_components, strict=True
    ):
        target_values = target_component
        if target_scale is not None:
            target_values = _julia_array_scalar_operation(
                target_values, target_scale, np.divide
            )
        difference = _julia_array_array_operation(
            source_component[:, None],
            target_values[None, :],
            np.subtract,
        )
        terms.append(
            _julia_array_array_operation(
                difference, difference, np.multiply
            )
        )
    return _sum_cost_terms(terms)


def _normalization_value(
    matrix: np.ndarray,
    normalization: Callable[[np.ndarray], Any] | None,
) -> Any:
    if normalization is None:
        return 1.0
    if not callable(normalization):
        raise TypeError("normalization must be callable or None")
    return normalization(matrix)


def _normalize_cost(
    matrix: np.ndarray,
    normalization: Callable[[np.ndarray], Any] | None,
) -> np.ndarray:
    """Apply Julia's broadcasted cost normalization."""

    divisor = _normalization_value(matrix, normalization)
    divisor_array = np.asarray(divisor)
    if divisor_array.ndim == 0:
        return _julia_array_scalar_operation(
            matrix, divisor_array.reshape(())[()], np.divide
        )
    # Julia broadcasting aligns dimension 1 with dimension 1 and appends
    # singleton trailing dimensions. NumPy aligns from the right, which would
    # turn a length-m normalization vector into column rather than row
    # normalization for an m×n cost matrix.
    ndim = max(matrix.ndim, divisor_array.ndim)
    matrix_view = matrix.reshape(
        matrix.shape + (1,) * (ndim - matrix.ndim)
    )
    divisor_view = divisor_array.reshape(
        divisor_array.shape + (1,) * (ndim - divisor_array.ndim)
    )
    return _julia_array_array_operation(
        matrix_view, divisor_view, np.divide
    )


def getCostMatrix(
    Lmu: Sequence[Any],
    Lv: Sequence[Any] | None = None,
    *,
    normalization: Callable[[np.ndarray], Any] | None = np.max,
) -> np.ndarray:
    """Return the normalized dense squared-Euclidean transport cost matrix."""

    source = _as_lattice(Lmu, allow_empty_axes=True)
    target = (
        source
        if Lv is None
        else _as_lattice(Lv, allow_empty_axes=True)
    )
    if len(source) != len(target):
        raise ValueError("source and target lattices must have the same dimensionality")
    matrix = _cost_matrix_terms(source, target)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _normalize_cost(matrix, normalization)


def pdCostMatrix(
    LRoot: Sequence[Any],
    LTarget: Sequence[Any],
    alphaRoot: float,
    alphaTarget: float,
    *,
    normalization: Callable[[np.ndarray], Any] | None = np.max,
    flambda: float = 1.0,
) -> np.ndarray:
    """Return the legacy phase-diversity cost matrix.

    This API is retained for Julia compatibility; ``getCostMatrix`` is the
    preferred and numerically better-conditioned formulation.
    """

    for name, value in (
        ("alphaRoot", alphaRoot),
        ("alphaTarget", alphaTarget),
        ("flambda", flambda),
    ):
        if not _is_real_number(value):
            raise TypeError(f"{name} must be real")
    root = _as_lattice(LRoot, allow_empty_axes=True)
    target = _as_lattice(LTarget, allow_empty_axes=True)
    if len(root) != len(target):
        raise ValueError("root and target lattices must have the same dimensionality")
    delta_array = _julia_array_scalar_operation(
        np.asarray(alphaRoot), alphaTarget, np.subtract
    )
    delta = _julia_array_scalar_operation(
        delta_array, flambda, np.multiply
    ).reshape(())[()]
    matrix = _cost_matrix_terms(root, target, target_scale=delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _normalize_cost(matrix, normalization)


def _safe_inverse(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "O":
        result = np.empty(array.shape, dtype=object)
        for index in np.ndindex(array.shape):
            value = array[index]
            result[index] = type(value)(0) if value == 0 else 1 / value
        return result
    if array.dtype.kind in "fc":
        # Julia's literal ``1 / x`` keeps Float16/Float32 and
        # ComplexF16/ComplexF32 storage.  A Float64 work array here changes
        # the rounded barycenter before it is assigned to ``vf::Float64``.
        result = np.zeros(array.shape, dtype=array.dtype)
        one = np.asarray(1, dtype=array.dtype)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(one, array, out=result, where=array != 0)
        return result
    # For machine integers, Julia's nonzero branch is Float64 while its zero
    # branch keeps the integer type.  Float64 can represent both numerical
    # outcomes without changing their later assignment to ``vf::Float64``.
    result = np.zeros(array.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(1.0, array, out=result, where=array != 0)
    return result


def _julia_matmul(left: Any, right: Any) -> np.ndarray:
    """Matrix multiplication after Julia-compatible scalar promotion."""

    first, second = np.asarray(left), np.asarray(right)
    if _object_contains_decimal(first) or _object_contains_decimal(second):
        return np.asarray(
            np.matmul(_as_decimal_array(first), _as_decimal_array(second))
        )
    if first.dtype.kind == second.dtype.kind == "b":
        # Bool products are Bool, but Julia's dot-product reduction widens
        # their addition to Int64; NumPy otherwise uses Boolean matmul.
        return np.asarray(
            np.matmul(first.astype(np.int64), second.astype(np.int64))
        )
    return _julia_array_array_operation(first, second, np.matmul)


def mapify(plan: Any, Lmu: Sequence[Any], Lv: Sequence[Any]) -> np.ndarray:
    """Convert a transport plan to its target-coordinate barycentric map."""

    source = _as_lattice(Lmu, allow_empty_axes=True)
    target = _as_lattice(Lv, allow_empty_axes=True)
    if len(source) != len(target):
        raise ValueError("source and target lattices must have the same dimensionality")
    matrix = np.asarray(plan)
    expected = (int(np.prod([len(axis) for axis in source])), int(np.prod([len(axis) for axis in target])))
    if matrix.ndim != 2 or matrix.shape != expected:
        raise ValueError(f"plan must have shape {expected}")
    if not source:
        # Julia reshapes ``zeros(1, 0)`` to ``(0,)`` for Lattice{0}.  The
        # plan is shape-checked but otherwise unobservable.
        return np.empty((0,), dtype=np.float64)
    target_shape = tuple(len(axis) for axis in target)
    reshaped_plan = np.reshape(
        matrix, (matrix.shape[0], *target_shape), order="F"
    )
    source_shape = tuple(len(axis) for axis in source)
    # Julia allocates ``vf = zeros(lmu, N)`` without an element type.  It first
    # multiplies the weighted coordinate by ``safeInverse(row_mass)`` and only
    # then assigns that barycenter to Float64.  In particular, an empty row
    # maps to zero (not 0/0 NaN), while assignment still raises InexactError
    # for a genuinely complex barycenter.  Keeping coordinate components
    # separate also avoids NumPy pre-promoting heterogeneous lattice axes.
    components: list[np.ndarray] = []
    for dimension, coordinate in enumerate(target):
        collapsed_axes = tuple(
            axis + 1
            for axis in range(len(target))
            if axis != dimension
        )
        marginal = (
            np.sum(reshaped_plan, axis=collapsed_axes, keepdims=True)
            if collapsed_axes
            else reshaped_plan
        )
        marginal = np.reshape(
            marginal, (matrix.shape[0], len(coordinate)), order="F"
        )
        weighted = _julia_matmul(marginal, np.asarray(coordinate))
        row_mass = np.sum(marginal, axis=1)
        barycenter = _julia_array_array_operation(
            weighted, _safe_inverse(row_mass), np.multiply
        )
        assigned = np.asarray(
            _julia_assignment_values(barycenter, np.dtype(np.float64)),
            dtype=np.float64,
        )
        components.append(
            assigned.reshape(source_shape, order="F")
        )
    return np.stack(components, axis=-1)


def _int64_value(value: Any, name: str) -> int:
    """Validate Julia's concrete platform ``Int`` without truncation."""

    if type(value) is int:
        limits = np.iinfo(np.int64)
        if not limits.min <= value <= limits.max:
            raise OverflowError(f"{name} does not fit Julia Int64")
        return value
    if isinstance(value, np.int64):
        return int(value)
    raise TypeError(f"{name} must have Julia Int (Int64) type")


def _axis_index(dimension: Any, ndim: int) -> int:
    """Accept Julia's 1-based dimensions and unambiguous Python axis zero."""

    exact_dimension = _int64_value(dimension, "dimension")
    if exact_dimension == 0:
        return 0
    if 1 <= exact_dimension <= ndim:
        return exact_dimension - 1
    raise ValueError(f"dimension must be between 1 and {ndim} (or Python axis 0)")


def _origin_tuple(origin: Any, shape: Sequence[int]) -> tuple[int, ...]:
    if hasattr(origin, "I"):
        values = tuple(
            _int64_value(value, "origin index") - 1 for value in origin.I
        )
    else:
        try:
            values = tuple(
                _int64_value(value, "origin index") for value in origin
            )
        except TypeError:
            if not hasattr(origin, "__iter__"):
                raise TypeError("originIdx must be an integer index sequence") from None
            raise
    if len(values) < len(shape):
        raise IndexError("originIdx is outside the array")
    values = values[: len(shape)]
    if any(
        value < 0 or value >= size
        for value, size in zip(values, shape, strict=True)
    ):
        raise IndexError("originIdx is outside the array")
    return values


def _hyper_sum(
    A: Any,
    originIdx: Sequence[int],
    sumDim: int,
    fixDims: Sequence[int],
    *,
    trapezoid: bool,
) -> np.ndarray:
    array = np.asarray(A)
    ndim = array.ndim
    axis = _axis_index(sumDim, ndim)
    fixed = tuple(_axis_index(item, ndim) for item in fixDims)
    if axis in fixed:
        raise ValueError("sumDim can't be in fixDims")
    origin = _origin_tuple(originIdx, array.shape)
    selection = tuple(
        slice(None) if dimension == axis or dimension in fixed else slice(origin[dimension], origin[dimension] + 1)
        for dimension in range(ndim)
    )
    selected = array[selection]
    cumulative = np.cumsum(selected, axis=axis)
    if trapezoid:
        cumulative = cumulative - selected / 2
    origin_selection = [slice(None)] * ndim
    origin_selection[axis] = slice(origin[axis], origin[axis] + 1)
    return cumulative - cumulative[tuple(origin_selection)]


def hyperSum(A: Any, originIdx: Sequence[int], sumDim: int, fixDims: Sequence[int]) -> np.ndarray:
    """Path-integrate one component without trapezoidal correction.

    ``originIdx`` follows Python's zero-based indexing.  Dimension numbers
    retain Julia's one-based convention; zero is also accepted for axis zero.
    """

    return _hyper_sum(A, originIdx, sumDim, fixDims, trapezoid=False)


def hyperSum2(A: Any, originIdx: Sequence[int], sumDim: int, fixDims: Sequence[int]) -> np.ndarray:
    """Path-integrate one component using the Julia trapezoidal correction."""

    return _hyper_sum(A, originIdx, sumDim, fixDims, trapezoid=True)


def scalarPotentialN(
    A: Any,
    L: Sequence[Any],
    *,
    idx: Sequence[int] | None = None,
    dimOrder: Sequence[int] | None = None,
) -> np.ndarray:
    """Integrate an N-D vector field along the ordered coordinate path.

    The default anchor exactly preserves Julia's ``length(axis) ÷ 2``
    one-based ``CartesianIndex``: in Python that is ``len(axis)//2 - 1``.
    This deliberately remains invalid for a singleton axis.
    """

    vector_field = np.asarray(A)
    lattice = _as_lattice(L)
    ndim = len(lattice)
    spatial_shape = tuple(len(axis) for axis in lattice)
    if vector_field.ndim != ndim + 1 or vector_field.shape[:-1] != spatial_shape or vector_field.shape[-1] != ndim:
        raise ValueError("A must have dimension one greater than L and its last size must equal dim(L)")
    origin = (
        _origin_tuple(
            tuple(len(axis) // 2 - 1 for axis in lattice), spatial_shape
        )
        if idx is None
        else _origin_tuple(idx, spatial_shape)
    )
    if dimOrder is None:
        order_spec = tuple(range(1, ndim + 1))
    else:
        supplied_order = tuple(dimOrder)
        if len(supplied_order) < ndim:
            raise IndexError(
                "dimOrder has fewer entries than the lattice dimension"
            )
        # Julia loops over i=1:N and indexes dimOrder[i]. Any trailing entries
        # are unobservable rather than part of the dispatch contract.
        order_spec = tuple(
            _int64_value(item, "dimOrder entry")
            for item in supplied_order[:ndim]
        )
    order = tuple(_axis_index(item, ndim) for item in order_spec)
    potential: np.ndarray | None = None
    fixed: list[int] = []
    for axis in order:
        # hyperSum2 takes Julia one-based dimensions.
        integrated = hyperSum2(
            vector_field[..., axis],
            origin,
            axis + 1,
            tuple(item + 1 for item in fixed),
        )
        term = _julia_array_scalar_operation(
            integrated, _exact_step(lattice[axis]), np.multiply
        )
        potential = (
            term
            if potential is None
            else _julia_array_array_operation(potential, term, np.add)
        )
        fixed.append(axis)
    assert potential is not None
    return np.asarray(potential)


def normalizeDistribution(U: Any) -> np.ndarray:
    """Normalize absolute values to a probability distribution."""

    values = np.abs(np.asarray(U))
    with np.errstate(divide="ignore", invalid="ignore"):
        return values / np.sum(values)


def _contains_nan(values: Any) -> bool:
    array = np.asarray(values)
    if array.dtype.kind != "O":
        return bool(np.any(np.isnan(array)))
    for value in array.flat:
        if isinstance(value, Decimal):
            if value.is_nan():
                return True
            continue
        try:
            if bool(np.isnan(value)):
                return True
        except TypeError:
            continue
    return False


def _decimal_exp(values: Any) -> np.ndarray:
    """Elementwise Decimal exponential for Julia BigFloat work arrays."""

    array = _as_decimal_array(values)
    output = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        output[index] = array[index].exp()
    return output


def _decimal_sinkhorn_gibbs(
    source: np.ndarray,
    target: np.ndarray,
    cost: np.ndarray,
    epsilon: Any,
    *,
    absolute_tolerance: Any,
    relative_tolerance: Any,
    interval: int,
    maxiter: int,
) -> tuple[np.ndarray, bool]:
    """Run SinkhornGibbs in the active Decimal context.

    OptimalTransport.jl promotes its work cache to BigFloat whenever a
    marginal, the cost matrix, or ``one(eltype(C))/epsilon`` is BigFloat.
    NumPy has no arbitrary-precision dtype, so this small dense path uses
    object arrays of Decimal scalars rather than silently narrowing to
    Float64.
    """

    source_decimal = _as_decimal_array(source)
    target_decimal = _as_decimal_array(target)
    cost_decimal = _as_decimal_array(cost)
    epsilon_decimal = _as_decimal_approx(epsilon)
    atol_decimal = _as_decimal_approx(absolute_tolerance)
    rtol_decimal = _as_decimal_approx(relative_tolerance)

    kernel = _decimal_exp(
        np.asarray(
            [
                -value / epsilon_decimal
                for value in cost_decimal.flat
            ],
            dtype=object,
        ).reshape(cost_decimal.shape)
    )
    one = Decimal(1)
    u = np.full(source_decimal.shape, one, dtype=object)
    v = np.full(target_decimal.shape, one, dtype=object)
    Kv = np.matmul(kernel, v)
    converged = False
    countdown = interval
    for iteration in range(1, maxiter + 1):
        u = source_decimal / Kv
        v = target_decimal / np.matmul(kernel.T, u)
        Kv = np.matmul(kernel, v)
        countdown -= 1
        if countdown == 0 or iteration == maxiter:
            countdown = interval
            current = u * Kv
            norm_current = sum(abs(value) for value in current.flat)
            error = sum(
                abs(left - right)
                for left, right in zip(
                    source_decimal.flat, current.flat, strict=True
                )
            )
            source_norm = sum(
                abs(value) for value in source_decimal.flat
            )
            converged = error < max(
                atol_decimal,
                rtol_decimal * max(source_norm, norm_current),
            )
            if converged:
                break
    return kernel * u[:, None] * v[None, :], converged


def _sinkhorn_gibbs(
    mu: Any,
    nu: Any,
    cost: Any,
    epsilon: float,
    *,
    tol: float | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    check_marginal_step: int | None = None,
    check_convergence: int | None = None,
    maxiter: int = 1000,
) -> np.ndarray:
    """OptimalTransport.jl 0.3.20 ``SinkhornGibbs`` for vector marginals."""

    if not _is_real_numeric_data(mu) or not _is_real_numeric_data(nu):
        raise TypeError("source and target marginals must have real numeric data")
    source_input = np.asarray(mu)
    target_input = np.asarray(nu)
    cost_input = np.asarray(cost)
    if (
        source_input.ndim != 1
        or target_input.ndim != 1
        or cost_input.shape
        != (source_input.size, target_input.size)
    ):
        raise ValueError("source, target, and cost dimensions are inconsistent")
    decimal_work = (
        _object_contains_decimal(source_input)
        or _object_contains_decimal(target_input)
        or _object_contains_decimal(cost_input)
        or isinstance(epsilon, Decimal)
    )
    if decimal_work:
        source_check = _as_decimal_array(source_input)
        target_check = _as_decimal_array(target_input)
        invalid_marginal = any(
            not value.is_finite() or value < 0
            for value in (*source_check.flat, *target_check.flat)
        )
    else:
        source_check = np.asarray(source_input, dtype=float)
        target_check = np.asarray(target_input, dtype=float)
        invalid_marginal = bool(
            np.any(source_check < 0)
            or np.any(target_check < 0)
            or not np.all(np.isfinite(source_check))
            or not np.all(np.isfinite(target_check))
        )
    if invalid_marginal:
        raise ValueError("marginals must be finite and nonnegative")
    # OptimalTransport.checkbalanced applies scalar ``isapprox`` before its
    # solver cache is promoted.  In particular, Float16 normalized marginals
    # use Float16's default tolerance here, not Float64's cache tolerance.
    source_mass = np.sum(source_input)
    target_mass = np.sum(target_input)
    balance_rtol = _julia_rtol(source_input, target_input)
    if abs(source_mass - target_mass) > balance_rtol * max(
        abs(source_mass), abs(target_mass)
    ):
        raise ValueError("source and target marginals must have equal mass")
    maxiter_value = _int64_value(maxiter, "maxiter")
    if tol is not None:
        warnings.warn("tol is deprecated; use atol and rtol", DeprecationWarning, stacklevel=2)
    absolute_tolerance = (
        tol
        if atol is None and tol is not None
        else (0 if atol is None else atol)
    )
    if rtol is None:
        relative_tolerance = (
            0
            if absolute_tolerance > 0
            else (_decimal_rtol() if decimal_work else np.sqrt(np.finfo(float).eps))
        )
    else:
        relative_tolerance = rtol
    if check_marginal_step is not None:
        warnings.warn(
            "check_marginal_step is deprecated; use check_convergence",
            DeprecationWarning,
            stacklevel=2,
        )
    interval_value = (
        check_convergence
        if check_convergence is not None
        else check_marginal_step
    )
    interval = (
        10
        if interval_value is None
        else _int64_value(interval_value, "check_convergence")
    )

    if decimal_work:
        plan, converged = _decimal_sinkhorn_gibbs(
            source_input,
            target_input,
            cost_input,
            epsilon,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            interval=interval,
            maxiter=maxiter_value,
        )
        if not converged:
            warnings.warn(
                f"Sinkhorn algorithm ({maxiter_value}/{maxiter_value}): not converged",
                RuntimeWarning,
                stacklevel=2,
            )
        return plan

    source = np.asarray(source_input, dtype=float)
    target = np.asarray(target_input, dtype=float)
    C = np.asarray(cost_input, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kernel = np.exp(-C / float(epsilon))
    u = np.ones(source.shape, dtype=float)
    v = np.ones(target.shape, dtype=float)
    Kv = kernel @ v  # OptimalTransport.jl's init_step!
    converged = False
    countdown = interval
    for iteration in range(1, maxiter_value + 1):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            u = source / Kv
            v = target / (kernel.T @ u)
            Kv = kernel @ v
        countdown -= 1
        if countdown == 0 or iteration == maxiter_value:
            countdown = interval
            current = u * Kv
            norm_current = float(np.sum(np.abs(current)))
            error = float(np.sum(np.abs(source - current)))
            converged = error < max(
                absolute_tolerance,
                relative_tolerance * max(float(np.sum(np.abs(source))), norm_current),
            )
            if converged:
                break
    if not converged:
        warnings.warn(
            f"Sinkhorn algorithm ({maxiter_value}/{maxiter_value}): not converged",
            RuntimeWarning,
            stacklevel=2,
        )
    return kernel * u[:, None] * v[None, :]


def _natural_axis(size: int) -> np.ndarray:
    if size < 1:
        raise ValueError("lattice sizes must be positive")
    return np.arange(-(size // 2), (size - 1) // 2 + 1, dtype=float) / np.sqrt(size)


def _ot_natural_lattices(source_shape: Sequence[int], target_shape: Sequence[int]) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if len(source_shape) != len(target_shape):
        raise ValueError("source and target dimensionality must agree")
    source = tuple(_natural_axis(int(size)) for size in source_shape)
    target_unscaled = tuple(_natural_axis(int(size)) for size in target_shape)
    target: list[np.ndarray] = []
    for source_axis, target_axis in zip(source, target_unscaled, strict=True):
        source_max = float(np.max(source_axis))
        target_max = float(np.max(target_axis))
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = float(np.divide(source_max, target_max))
            if np.isfinite(scale):
                target_values = target_axis * scale
                target_step = (
                    1 / np.sqrt(len(target_axis))
                ) * scale
            else:
                # Broadcasting ``StepRangeLen .* Inf/NaN`` constructs a new
                # range from its zero reference.  Since ``0 * Inf`` is NaN,
                # every materialized coordinate (not only the zero endpoint)
                # is NaN in Julia.
                target_values = np.full(
                    target_axis.shape, np.nan, dtype=np.float64
                )
                target_step = np.nan
        target.append(_axis(target_values, target_step))
    return source, tuple(target)


def _validate_ot_fields(
    U: LatticeField,
    V: LatticeField,
    *,
    target_real: bool = False,
) -> None:
    if getattr(U, "field_type", None) is not Intensity or getattr(V, "field_type", None) is not Intensity:
        raise TypeError("OT inputs must be Intensity LatticeFields")
    if not _is_real_numeric_data(U.data):
        raise TypeError("OT source must have real numeric data")
    target_ok = (
        _is_real_numeric_data(V.data)
        if target_real
        else _is_numeric_data(V.data)
    )
    if not target_ok:
        kind = "real numeric" if target_real else "numeric"
        raise TypeError(f"OT target must have {kind} data")
    if np.ndim(U.data) != np.ndim(V.data) or len(U.L) != len(V.L):
        raise ValueError("OT inputs must have equal dimensionality")
    if U.flambda != V.flambda:
        raise ValueError("Unequal flambdas.")


def _beta_values(values: Sequence[float]) -> tuple[Any, ...]:
    """Return beta scalars with Julia tuple/vector construction semantics."""

    if isinstance(values, tuple):
        return values
    array = (
        _julia_literal_array(values)
        if isinstance(values, list)
        else np.asarray(values)
    )
    if array.ndim != 1:
        raise TypeError("beta vectors must be one-dimensional")
    return tuple(array)


def _pd_squared_radius(
    lattice: Sequence[Any], delta_beta: Sequence[Any]
) -> np.ndarray:
    """Evaluate pdot's squared offset with Julia ``sum`` semantics."""

    ndim = len(lattice)
    terms: list[np.ndarray] = []
    for dimension, (axis, offset) in enumerate(
        zip(lattice, delta_beta, strict=True)
    ):
        shifted = _julia_array_scalar_operation(
            np.asarray(axis), offset, np.subtract
        )
        squared = _julia_array_array_operation(
            shifted, shifted, np.multiply
        )
        terms.append(_to_dim(squared, dimension, ndim))
    return np.asarray(_julia_add_sum(terms))


def otPhase(U: LatticeField, V: LatticeField, epsilon: float, **options: Any) -> LatticeField:
    """Generate a phase from the dense entropic optimal-transport map."""

    if not _is_real_number(epsilon):
        raise TypeError("epsilon must be real")
    _validate_ot_fields(U, V)
    source, target = normalizeDistribution(U.data), normalizeDistribution(V.data)
    natural_source, natural_target = _ot_natural_lattices(source.shape, target.shape)
    cost = getCostMatrix(natural_source, natural_target)
    plan = _sinkhorn_gibbs(
        source.ravel(order="F"),
        target.ravel(order="F"),
        cost,
        epsilon,
        **options,
    )
    if _contains_nan(plan):
        raise FloatingPointError("sinkhorn returned nan; try changing epsilon")
    transport_map = mapify(plan, U.L, V.L)
    order = tuple(range(np.ndim(U.data), 0, -1))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        phase = _julia_array_scalar_operation(
            scalarPotentialN(
                transport_map, U.L, dimOrder=order
            ),
            U.flambda,
            np.divide,
        )
    return _make_field(RealPhase, phase, U.L, U.flambda)


def pdotPhase(
    G2Root: LatticeField,
    G2Target: LatticeField,
    alphaRoot: float,
    alphaTarget: float,
    betaRoot: Sequence[float],
    betaTarget: Sequence[float],
    epsilon: float,
    **options: Any,
) -> LatticeField:
    """Infer the root-plane phase from two phase-diverse intensities."""

    for name, value in (
        ("alphaRoot", alphaRoot),
        ("alphaTarget", alphaTarget),
        ("epsilon", epsilon),
    ):
        if not _is_real_number(value):
            raise TypeError(f"{name} must be real")
    _validate_ot_fields(G2Root, G2Target, target_real=True)
    ndim = np.ndim(G2Root.data)
    for name, values in (
        ("betaRoot", betaRoot),
        ("betaTarget", betaTarget),
    ):
        if isinstance(values, tuple):
            if len(values) != ndim:
                raise ValueError(
                    f"{name} NTuple must match the image dimensionality"
                )
        elif len(values) < ndim:
            raise ValueError(
                f"{name} Vector must have at least the image dimensionality"
            )
    if not all(_is_real_number(value) for value in betaRoot):
        raise TypeError("betaRoot values must be real")
    if not all(_is_real_number(value) for value in betaTarget):
        raise TypeError("betaTarget values must be real")
    delta_alpha = _scalar_operation(alphaRoot, alphaTarget, np.subtract)
    source, target = normalizeDistribution(G2Root.data), normalizeDistribution(G2Target.data)
    natural_source, natural_target = _ot_natural_lattices(source.shape, target.shape)
    cost = getCostMatrix(natural_source, natural_target)
    plan = _sinkhorn_gibbs(
        source.ravel(order="F"),
        target.ravel(order="F"),
        cost,
        epsilon,
        **options,
    )
    if _contains_nan(plan):
        raise FloatingPointError("sinkhorn returned nan; try changing epsilon")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        transport_map = _julia_array_scalar_operation(
            mapify(plan, G2Root.L, G2Target.L), delta_alpha, np.divide
        )
    phase = scalarPotentialN(transport_map, G2Root.L)
    # Preserve heterogeneous NTuple elements through the elementwise
    # subtraction.  NumPy's eager common-dtype inference is not Julia's
    # scalar promotion rule (notably for Int64/Float32 pairs).
    root_beta_values = _beta_values(betaRoot)
    target_beta_values = _beta_values(betaTarget)
    delta_beta = tuple(
        _scalar_operation(root, target, np.subtract)
        for root, target in zip(
            root_beta_values[:ndim],
            target_beta_values[:ndim],
            strict=True,
        )
    )
    denominator = _scalar_operation(2, delta_alpha, np.multiply)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        correction = _julia_array_scalar_operation(
            _pd_squared_radius(G2Root.L, delta_beta),
            denominator,
            np.divide,
        )
    phase = _julia_array_array_operation(phase, correction, np.subtract)
    flambda_squared = _literal_square(G2Root.flambda)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        phase = _julia_array_scalar_operation(
            phase, flambda_squared, np.divide
        )
    return _make_field(RealPhase, phase, G2Root.L, G2Root.flambda)


def pdotBeamEstimate(
    G2Root: LatticeField,
    G2Target: LatticeField,
    alphaRoot: float,
    alphaTarget: float,
    betaRoot: Sequence[float],
    betaTarget: Sequence[float],
    epsilon: float,
    *,
    LFine: Sequence[Any] | None = None,
    **options: Any,
) -> LatticeField:
    """Infer a complex beam from two phase-diverse intensity images."""

    phase = pdotPhase(
        G2Root,
        G2Target,
        alphaRoot,
        alphaTarget,
        betaRoot,
        betaTarget,
        epsilon,
        **options,
    )
    if np.any(np.asarray(G2Root.data) < 0):
        raise ValueError("intensity values must be nonnegative")
    amplitude_field = G2Root.sqrt()
    if LFine is None:
        camera_field = amplitude_field * phase
    else:
        fine_lattice = as_lattice(LFine)
        if len(fine_lattice) != np.ndim(G2Root.data):
            raise ValueError(
                "LFine must match the image dimensionality"
            )
        # Interpolations.jl 0.16.2's obsolete ``Linear()`` extrapolation
        # route succeeds for integer AbstractRanges, but recurses until stack
        # overflow for the probed Float, Rational, and BigFloat ranges.  Keep
        # the working source overload without silently repairing the broken
        # upstream range families.
        if any(
            np.asarray(axis).dtype.kind not in "iu"
            for axis in fine_lattice
        ):
            raise NotImplementedError(
                "the audited Julia pdotBeamEstimate LFine path fails for "
                "non-integer target ranges"
            )
        from .resampling import Linear, upsample

        amplitude_field = upsample(
            amplitude_field, fine_lattice, bc=0
        )
        phase = upsample(
            phase, fine_lattice, bc=Linear()
        )
        camera_field = amplitude_field * phase
    camera_lattice = tuple(camera_field.L)
    beam_lattice = _dual_shift_lattice(camera_lattice, G2Root.flambda)
    beta_root_values = _beta_values(betaRoot)
    diversity = _quadratic_linear_phase(
        beam_lattice,
        alphaRoot,
        beta_root_values[: len(beam_lattice)],
    )
    div_phase = _make_field(
        RealPhase, diversity, beam_lattice, G2Root.flambda
    )
    return _field_isft(camera_field) * div_phase.conj()


def _assign_julia_linear(
    destination: np.ndarray, values: Any
) -> None:
    """Assign in Julia's column-major scalar conversion order.

    Julia's ``A[:] = rhs[:]`` and in-place broadcasts do not prevalidate a
    complete converted temporary. Each successful element is visible if a
    later element raises ``InexactError``. NumPy's whole-array assignment
    instead preconverts (and normally behaves atomically), so perform the
    concrete destination conversion one element at a time.
    """

    source = np.asarray(values)
    if source.shape != destination.shape:
        raise ValueError("assignment shapes must match")
    for linear_index in range(destination.size):
        index = np.unravel_index(
            linear_index, destination.shape, order="F"
        )
        converted = _julia_assignment_values(
            np.asarray(source[index]), destination.dtype
        )
        destination[index] = np.asarray(converted).reshape(())[()]


def _sinkhorn_iter_base_inplace(
    u: np.ndarray,
    v: np.ndarray,
    U: Any,
    V: Any,
    FAu: Any,
    FAv: Any,
) -> np.ndarray:
    """Apply one Julia ``SinkhornIterBase!`` update and return mutated ``u``."""

    if not all(
        isinstance(value, np.ndarray) for value in (u, v, U, V, FAu, FAv)
    ):
        raise TypeError("SinkhornIterBase! requires dense NumPy arrays")
    if any(not _is_real_numeric_data(value) for value in (u, v, U, V)):
        raise TypeError("u, v, U, and V must have real numeric data")
    if not _is_numeric_data(FAu) or not _is_numeric_data(FAv):
        raise TypeError("FAu and FAv must have numeric data")
    if u.dtype != v.dtype:
        raise TypeError("u and v must have the same Julia element type")
    if np.asarray(FAu).dtype != np.asarray(FAv).dtype:
        raise TypeError("FAu and FAv must have the same Julia element type")
    U_array, V_array = np.asarray(U), np.asarray(V)
    if not (u.shape == v.shape == U_array.shape == V_array.shape == np.shape(FAu) == np.shape(FAv)):
        raise ValueError("all Sinkhorn arrays must have the same shape")
    # Preserve both statement order and each broadcast's element-by-element
    # conversion/mutation order.
    row_sum = np.real(_isft_array(_sft_array(u) * np.asarray(FAu)))
    _assign_julia_linear(
        v,
        _julia_array_array_operation(
            V_array, _safe_inverse(row_sum), np.multiply
        ),
    )
    _assign_julia_linear(
        v,
        _julia_array_scalar_operation(
            v, _safe_inverse(np.sum(v)).reshape(())[()], np.multiply
        ),
    )
    column_sum = np.real(
        _isft_array(_sft_array(v) * np.asarray(FAv))
    )
    _assign_julia_linear(
        u,
        _julia_array_array_operation(
            U_array, _safe_inverse(column_sum), np.multiply
        ),
    )
    _assign_julia_linear(
        u,
        _julia_array_scalar_operation(
            u, _safe_inverse(np.sum(u)).reshape(())[()], np.multiply
        ),
    )
    # Julia's final ``u .*= ...`` expression returns ``u``; ``v`` is mutated
    # as a side effect but is not part of the return value.
    return u


def SinkhornIterBase(
    u: np.ndarray,
    v: np.ndarray,
    U: Any,
    V: Any,
    FAu: Any,
    FAv: Any,
) -> np.ndarray:
    """Python-spellable wrapper for Julia's mutating helper."""

    return _sinkhorn_iter_base_inplace(u, v, U, V, FAu, FAv)


# Julia's spelling is available through ``getattr(module, "SinkhornIterBase!")``.
globals()["SinkhornIterBase!"] = _sinkhorn_iter_base_inplace


def SinkhornConvN(
    U: Any,
    V: Any,
    epsilon: float,
    max_iter: int,
    *,
    every: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run the experimental circular-convolution Sinkhorn iteration."""

    if not _is_real_numeric_data(U) or not _is_real_numeric_data(V):
        raise TypeError("U and V must have real numeric data")
    if (
        isinstance(epsilon, Decimal)
        or _object_contains_decimal(U)
        or _object_contains_decimal(V)
    ):
        raise TypeError("type BigFloat not supported by FFTW")
    # Julia creates ``u`` as Float64 but creates ``v`` with the target's
    # element type. Preserve that dispatch boundary: a positive iteration with
    # a non-Float64 target reaches the same-type Sinkhorn helper and fails.
    source, target = np.asarray(U), np.asarray(V)
    if source.shape != target.shape or source.ndim == 0:
        raise ValueError("U and V must be same-shaped arrays")
    if not isinstance(max_iter, (int, np.integer)):
        raise TypeError("max_iter must be an integer")
    if every is not None and not isinstance(
        every, (int, np.integer, bool, np.bool_)
    ):
        raise TypeError("every must be a Julia Integer or None")
    u = np.full(source.shape, 1 / source.size, dtype=float)
    v = np.empty(target.shape, dtype=target.dtype)
    previous = u.copy()
    loss: list[float] = []
    kernel_u = _r2(tuple(_natural_axis(size) for size in u.shape))
    kernel_v = _r2(tuple(_natural_axis(size) for size in v.shape))
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel_u = kernel_u / kernel_u.flat[0]
        kernel_v = kernel_v / kernel_v.flat[0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        FAu = _sft_array(np.exp(-kernel_u / float(epsilon)))
        FAv = _sft_array(np.exp(-kernel_v / float(epsilon)))
    for iteration in range(1, int(max_iter) + 1):
        _sinkhorn_iter_base_inplace(u, v, source, target, FAu, FAv)
        if every is not None and iteration % int(every) == 0:
            loss.append(float(np.linalg.norm(u - previous)))
            previous = u.copy()
    row_sum = np.real(_isft_array(_sft_array(u) * FAu))
    v = target * _safe_inverse(row_sum)
    column_sum = np.real(_isft_array(_sft_array(v) * FAv))
    u = source * _safe_inverse(column_sum)
    return u, v, loss


def dualToGradients(u: Any, v: Any, U: Any, LV: Sequence[Any], epsilon: float) -> np.ndarray:
    """Convert convolutional Sinkhorn dual variables to map gradients."""

    if not all(isinstance(value, np.ndarray) for value in (u, v, U)):
        raise TypeError("u, v, and U must be dense NumPy arrays")
    if any(not _is_real_numeric_data(value) for value in (u, v, U)):
        raise TypeError("u, v, and U must have real numeric data")
    u_array, v_array, source = np.asarray(u), np.asarray(v), np.asarray(U)
    if u_array.dtype != v_array.dtype:
        raise TypeError("u and v must have the same Julia element type")
    lattice = _as_lattice(LV)
    if u_array.shape != v_array.shape or u_array.shape != source.shape or u_array.shape != tuple(len(axis) for axis in lattice):
        raise ValueError("dual variables, source, and target lattice must have matching shapes")
    if not _is_real_number(epsilon):
        raise TypeError("epsilon must be real")
    if (
        isinstance(epsilon, Decimal)
        or _object_contains_decimal(u_array)
        or _object_contains_decimal(v_array)
        or _object_contains_decimal(source)
        or any(
            _object_contains_decimal(axis)
            for axis in lattice
        )
    ):
        raise TypeError("type BigFloat not supported by FFTW")
    kernel = _r2(tuple(_natural_axis(size) for size in v_array.shape))
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel /= kernel.flat[0]
    FAv = _sft_array(np.exp(-kernel / float(epsilon)))
    scale = _julia_array_array_operation(
        u_array, _safe_inverse(source), np.multiply
    )
    gradient = np.zeros(u_array.shape + (u_array.ndim,), dtype=float)
    for dimension, axis in enumerate(lattice):
        moment = _julia_array_array_operation(
            v_array,
            _to_dim(axis, dimension, u_array.ndim),
            np.multiply,
        )
        convolution = np.real(
            _isft_array(_sft_array(moment) * FAv)
        )
        gradient[..., dimension] = _julia_array_array_operation(
            scale, convolution, np.multiply
        )
    return gradient


def otQuickPhase(
    g2: LatticeField,
    G2: LatticeField,
    epsilon: float,
    max_iter: int,
    *,
    return_loss: bool = False,
) -> LatticeField | tuple[LatticeField, list[float]]:
    """Experimental convolutional OT phase (retained, but deprecated)."""

    _validate_ot_fields(g2, G2, target_real=True)
    if np.asarray(g2.data).dtype != np.asarray(G2.data).dtype:
        raise TypeError(
            "otQuickPhase inputs must have the same Julia element type"
        )
    if not isinstance(return_loss, (bool, np.bool_)):
        raise TypeError("return_loss must be Bool")
    if np.shape(g2.data) != np.shape(G2.data):
        raise ValueError("convolutional OT inputs must have equal shape")
    u, v, loss = SinkhornConvN(g2.data, G2.data, epsilon, max_iter, every=1)
    vector_field = dualToGradients(u, v, g2.data, G2.L, epsilon)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        phase_data = _julia_array_scalar_operation(
            scalarPotentialN(vector_field, g2.L),
            g2.flambda,
            np.divide,
        )
    phase = _make_field(RealPhase, phase_data, g2.L, g2.flambda)
    return (phase, loss) if return_loss else phase


def _otphase2_kernel(
    axis: np.ndarray, size: int, epsilon: Any
) -> np.ndarray:
    difference = _julia_array_array_operation(
        axis[:, None], axis[None, :], np.subtract
    )
    squared = _julia_array_array_operation(
        difference, difference, np.multiply
    )
    denominator = _scalar_operation(
        _scalar_operation(2, size, np.multiply),
        epsilon,
        np.multiply,
    )
    exponent = _julia_array_scalar_operation(
        squared, denominator, np.divide
    )
    exponent = _julia_array_scalar_operation(
        exponent, -1, np.multiply
    )
    if _object_contains_decimal(exponent):
        return _decimal_exp(exponent)
    return np.exp(exponent)


def otPhase2(
    inputField: LatticeField,
    targetField: LatticeField,
    epsilon: float,
    iterations: int,
    **options: Any,
) -> LatticeField:
    """Fast separable 2-D Sinkhorn phase for square arrays.

    The audited Julia implementation's square work allocation makes
    rectangular inputs unusable, so the port rejects that path explicitly
    rather than inventing corrected geometry. Its ineffective target
    wavelength check and IEEE final-division behavior are retained. Arbitrary
    keywords are accepted and ignored and exactly ``iterations`` updates run.
    """

    # The Julia signature captures ``options...`` but never reads or forwards
    # them.  Preserve that behavior, including for names such as return_loss,
    # rather than inventing convergence or alternate-return contracts.
    del options
    input_tag = getattr(inputField, "field_type", None)
    target_tag = getattr(targetField, "field_type", None)
    if input_tag is Modulus and target_tag is Modulus:
        if not _is_real_numeric_data(inputField.data):
            raise TypeError("otPhase2 Modulus source must have real numeric data")
        if not _is_numeric_data(targetField.data):
            raise TypeError("otPhase2 Modulus target must have numeric data")
        # Route through the same abs2 implementation as Julia's Modulus
        # overload.  Besides avoiding duplicated semantics, this preserves
        # Bool storage (NumPy's ``abs(bool_array) ** 2`` becomes Int8).
        source_data = np.asarray(square(inputField).data)
        target_data = np.asarray(square(targetField).data)
    elif input_tag is Intensity and target_tag is Intensity:
        if not _is_real_numeric_data(inputField.data):
            raise TypeError("otPhase2 Intensity source must have real numeric data")
        if not _is_numeric_data(targetField.data):
            raise TypeError("otPhase2 Intensity target must have numeric data")
        source_data = np.asarray(inputField.data)
        target_data = np.asarray(targetField.data)
    else:
        raise TypeError("otPhase2 inputs must both be Intensity or both be Modulus fields")
    if source_data.ndim != 2 or target_data.ndim != 2:
        raise ValueError("otPhase2 is only implemented for two-dimensional problems")
    if source_data.shape != target_data.shape:
        raise ValueError("input and target must have the same size")
    # The source asserts ``input.flambda == input.flambda`` and never inspects
    # the target wavelength.
    if inputField.flambda != inputField.flambda:
        raise ValueError("Unequal flambdas.")
    if type(iterations) is int:
        limits = np.iinfo(np.int64)
        if not limits.min <= iterations <= limits.max:
            raise OverflowError("iterations does not fit Julia Int64")
    elif not isinstance(iterations, np.int64):
        # Julia declares this argument as exactly ``Int`` rather than the
        # abstract ``Integer`` used by the other iterative routines. On the
        # audited 64-bit platform Bool and Int32 therefore do not dispatch.
        raise TypeError("iterations must have Julia Int (Int64) type")

    # Unlike ``normalizeDistribution``, Julia's factorized implementation uses
    # the raw (possibly complex) Intensity arrays and divides by their sums.
    with np.errstate(divide="ignore", invalid="ignore"):
        a = source_data / np.sum(source_data)
        b = target_data / np.sum(target_data)
    # Julia keeps Rational normalization exact, then promotes it to Float64
    # as soon as it interacts with the Float64 Sinkhorn scalings.  NumPy keeps
    # Python Fraction arithmetic in an object array instead, even after those
    # mixed operations.  Convert the already-normalized Rational values at
    # that same public work boundary so every Fraction/Float orientation has
    # Julia's Matrix{Float64} result type.
    for name, values in (("a", a), ("b", b)):
        if values.dtype.kind != "O":
            continue
        items = tuple(values.flat)
        rational_only = all(isinstance(value, Fraction) for value in items)
        heterogeneous_real = (
            all(
                isinstance(
                    value,
                    (
                        bool,
                        int,
                        float,
                        np.bool_,
                        np.integer,
                        np.floating,
                        Fraction,
                    ),
                )
                for value in items
            )
            and any(isinstance(value, (float, np.floating)) for value in items)
        )
        if rational_only or heterogeneous_real:
            converted = np.asarray(values, dtype=np.float64)
            if name == "a":
                a = converted
            else:
                b = converted
    n, m = a.shape
    if n != m:
        raise NotImplementedError(
            "rectangular otPhase2 is unusable in the audited Julia source "
            "because its Sinkhorn scalings are allocated as (n, n)"
        )
    X, Y = _natural_axis(n), _natural_axis(m)
    Kx = _otphase2_kernel(X, n, epsilon)
    Ky = _otphase2_kernel(Y, n, epsilon)
    u = np.full((n, n), 1 / (n * n), dtype=float)
    v = np.full((n, n), 1 / (n * n), dtype=float)
    for _ in range(int(iterations)):
        denominator_u = _julia_matmul(
            _julia_matmul(Kx, v), Ky
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            u = _julia_array_array_operation(
                a, denominator_u, np.divide
            )
        denominator_v = _julia_matmul(
            _julia_matmul(Kx, u), Ky
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            v = _julia_array_array_operation(
                b, denominator_v, np.divide
            )
    # This is Julia's unguarded ``u ./ a`` after the loop.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scale = _julia_array_array_operation(
            u, a, np.divide
        )
    with np.errstate(invalid="ignore", over="ignore"):
        dphi_dx = _julia_array_array_operation(
            scale,
            _julia_matmul(
                _julia_matmul(
                    Kx,
                    _julia_array_array_operation(
                        v, X[:, None], np.multiply
                    ),
                ),
                Ky,
            ),
            np.multiply,
        )
        dphi_dy = _julia_array_array_operation(
            scale,
            _julia_matmul(
                _julia_matmul(
                    Kx,
                    _julia_array_array_operation(
                        v, Y[None, :], np.multiply
                    ),
                ),
                Ky,
            ),
            np.multiply,
        )
    vector_field = np.stack((dphi_dx, dphi_dy), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        phase_data = _julia_array_scalar_operation(
            scalarPotentialN(
                vector_field, inputField.L, dimOrder=(2, 1)
            ),
            inputField.flambda,
            np.divide,
        )
    return _make_field(RealPhase, phase_data, inputField.L, inputField.flambda)
