"""Coordinate-lattice utilities and numerical basis helpers.

The audited Julia ``hermiteBasis`` implementation passes a coordinate range
to an integer-only DFT-basis method, making both it and ``FrFTBasis``
unreachable. The Python port deliberately preserves that public limitation
instead of choosing an unverified correction for the upstream algorithm.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import math
import struct
from numbers import Integral, Number, Real
from typing import Any, Callable

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
    Amplitude,
    ComplexAmp,
    ComplexAmplitude,
    ComplexPhase,
    DimensionMismatch,
    DomainError,
    FieldVal,
    Generic,
    Intensity,
    LF,
    Lattice,
    LatticeAxis,
    LatticeField,
    Modulus,
    Phase,
    RealAmp,
    RealAmplitude,
    RealPhase,
    S1Phase,
    UPhase,
    UnwrappedPhase,
    _axis,
    _is_real_number,
    _is_julia_number,
    _is_julia_platform_int,
    _julia_fill,
    _julia_literal_array,
    _julia_typed_zero,
    _julia_assignment_values,
    _julia_array_array_operation,
    _julia_array_scalar_operation,
    _logical_axis_scalar_operation,
    _with_axis_length_kind,
    _object_contains_mpfr,
    _require_dense_ndarray,
    as_lattice,
    elq,
    normalizeLF,
    phasor,
    square,
    subfield,
    sublattice,
    wrap,
)


def _step(axis: Any) -> Any:
    values = np.asarray(axis)
    if values.ndim != 1:
        raise DimensionMismatch("A lattice axis must be one-dimensional.")
    hint = getattr(axis, "_step_hint", None)
    flags = getattr(axis, "flags", None)
    logical_hint = (
        bool(getattr(axis, "_step_hint_is_logical", False))
        and flags is not None
        and not bool(flags.writeable)
    )
    if len(values) < 2:
        if hint is None:
            raise ValueError(
                "A singleton lattice axis needs retained range step metadata."
            )
        return hint

    if hint is not None and logical_hint:
        # StepRangeLen retains a logical step even when materialized Float32
        # differences alternate by one ulp. That metadata is authoritative.
        return hint

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
        if hint is not None and all(difference == hint for difference in differences):
            return hint
        step = differences[0]
        if not all(difference == step for difference in differences):
            raise ValueError("Lattice axes must be regularly spaced.")
        return step

    if values.dtype == np.dtype(np.bool_):
        differences = np.diff(values.astype(np.int8))
        if not np.all(differences == 1):
            raise ValueError("Lattice axes must be regularly spaced.")
        return np.bool_(True)
    differences = np.diff(values)
    real_dtype = values.real.dtype
    eps = (
        np.finfo(real_dtype).eps
        if np.issubdtype(real_dtype, np.floating)
        else np.finfo(float).eps
    )

    def is_regular(candidate: Any) -> bool:
        scale = max(
            1.0,
            float(np.abs(candidate)),
            float(np.max(np.abs(values))),
        )
        return bool(
            np.allclose(
                differences,
                candidate,
                rtol=16 * eps,
                atol=16 * eps * scale,
            )
        )

    # Inferred hints still require validation: arbitrary coordinate arrays can
    # have a first difference without being regular lattices.
    if hint is not None and is_regular(hint):
        return hint
    step = differences[0]
    if not is_regular(step):
        raise ValueError("Lattice axes must be regularly spaced.")
    return step


def natrange(n: int) -> LatticeAxis:
    """Return the length-*n* centered range that is self-dual under a DFT."""

    if not _is_julia_platform_int(n):
        raise TypeError("natrange expects Julia's concrete platform Int.")
    n = int(n)
    if n < 0:
        raise DomainError("natrange expects a nonnegative length.")
    if n == 0:
        # Julia evaluates an empty centered range divided by ``sqrt(0)``.
        # Its values stay empty, but the retained range step is positive Inf.
        return _axis(np.empty(0, dtype=float), np.inf)
    # Julia's ``floor(n/2)`` endpoints are Float64, so the source itself is a
    # TwicePrecision StepRangeLen before the dotted division.
    indices = LatticeAxis.from_start_step(
        np.float64(-math.floor(n / 2)),
        np.float64(1),
        n,
    )
    return _logical_axis_scalar_operation(indices, np.sqrt(n), np.divide)


def natlat(*sizes: Any) -> Lattice:
    """Return natural axes for a tuple or positional sequence of sizes."""

    if len(sizes) == 1 and isinstance(sizes[0], tuple):
        sizes = tuple(sizes[0])
    if not all(_is_julia_platform_int(item) for item in sizes):
        raise TypeError("natlat ultimately requires Julia platform Int sizes.")
    return tuple(natrange(int(item)) for item in sizes)


def naturalize(field: LatticeField) -> LatticeField:
    """Put a field on its natural lattice and reset ``flambda`` to 1.0."""

    return LatticeField[field.field_type](field.data, natlat(field.shape), 1.0)


def _normalize_padding(
    padding: int | tuple[int, ...], ndim: int
) -> tuple[tuple[int, int], ...]:
    if type(padding) is int or isinstance(padding, np.int64):
        value = int(padding)
        pairs = ((value, value),) * ndim
    else:
        try:
            raw_values = tuple(padding)
        except TypeError as exc:
            raise TypeError("padding must contain Julia Int values") from exc
        if any(
            isinstance(item, (bool, np.bool_))
            or not (type(item) is int or isinstance(item, np.int64))
            for item in raw_values
        ):
            raise TypeError("padding must contain Julia Int values")
        values = tuple(int(item) for item in raw_values)
        if len(values) == ndim:
            pairs = tuple((item, item) for item in values)
        elif len(values) == 2 * ndim:
            pairs = tuple((values[2 * i], values[2 * i + 1]) for i in range(ndim))
        else:
            raise ValueError("Bad tuple length.")
    return pairs


def _pad_axis(axis: Any, pair: tuple[int, int]) -> LatticeAxis:
    values = np.asarray(axis)
    if len(values) == 0:
        raise IndexError("Cannot pad an empty lattice axis.")
    step = _step(axis)
    before, after = pair
    # Julia ranges support negative padding as coordinate cropping.  When the
    # requested crop consumes the axis, ``0:negative`` is an empty range.
    length = max(0, len(values) + before + after)
    before_offset = _julia_array_scalar_operation(
        np.asarray(before, dtype=np.int64), step, np.multiply
    )[()]
    start = _julia_array_scalar_operation(
        values[0], before_offset, np.subtract
    )[()]
    # Preserve the source expression's exact operation family:
    # ``(0:m-1) .* step(L) .+ start``.  In particular, multiplying an ordinal
    # range by Float64 introduces a TwicePrecision StepRangeLen, while
    # translating an existing low-precision StepRangeLen retains its Float64
    # reference/step.
    offsets = _axis(range(length))
    scaled = _logical_axis_scalar_operation(offsets, step, np.multiply)
    result = _logical_axis_scalar_operation(scaled, start, np.add)
    return _with_axis_length_kind(
        result, getattr(_axis(axis), "_length_kind", "int64")
    )


def _looks_like_lattice(value: Any) -> bool:
    # A tuple mirrors Julia's NTuple lattice and avoids mistaking nested Python
    # lists (ordinary array input) for a lattice.
    return isinstance(value, tuple) and all(
        isinstance(item, (LatticeAxis, range, list, tuple, np.ndarray))
        and np.asarray(item).ndim == 1
        for item in value
    )


def padout(
    value: Any,
    padding: int | tuple[int, ...],
    filler: Any = _OMITTED,
) -> Any:
    """Pad an axis, lattice, dense array, or lattice field.

    A tuple with N entries means symmetric per-axis padding; a tuple with 2N
    entries means before/after pairs.  As in Julia, field padding accepts only
    the N-entry symmetric form.
    """

    if isinstance(value, LatticeField):
        if not isinstance(padding, tuple) or len(padding) != value.ndim:
            raise TypeError("Field padding requires one symmetric value per axis.")
        padded_data = padout(value.data, padding, filler)
        padded_lattice = padout(value.L, padding)
        return LatticeField._from_full(
            padded_data,
            padded_lattice,
            value.flambda,
            value.field_type,
            expected_dtype=value.dtype,
        )
    if isinstance(value, (LatticeAxis, range)):
        if filler is not _OMITTED:
            raise TypeError("padout(axis, padding) does not accept a filler")
        if type(padding) is int or isinstance(padding, np.int64):
            pair = (int(padding), int(padding))
        else:
            try:
                raw_values = tuple(padding)
            except TypeError as exc:
                raise TypeError("padding must contain Julia Int values") from exc
            if any(
                isinstance(item, (bool, np.bool_))
                or not (type(item) is int or isinstance(item, np.int64))
                for item in raw_values
            ):
                raise TypeError("padding must contain Julia Int values")
            values = tuple(int(item) for item in raw_values)
            if len(values) != 2:
                raise ValueError("Bad tuple length.")
            pair = values
        return _pad_axis(value, pair)
    if _looks_like_lattice(value):
        if filler is not _OMITTED:
            raise TypeError("padout(lattice, padding) does not accept a filler")
        lattice = as_lattice(value)
        pairs = _normalize_padding(padding, len(lattice))
        return tuple(_pad_axis(axis, pair) for axis, pair in zip(lattice, pairs))

    array = (
        _julia_literal_array(value)
        if isinstance(value, list)
        else _require_dense_ndarray(value, "padout array")
    )
    pairs = _normalize_padding(padding, array.ndim)
    if any(before < 0 or after < 0 for before, after in pairs):
        # Julia's AbstractRange overload supports negative coordinate cropping,
        # but its dense-array assignment indexes out of bounds.  Keep that
        # distinction instead of inventing dense-array cropping semantics.
        raise IndexError("Negative dense-array padding is out of bounds.")
    if filler is _OMITTED:
        if array.dtype.kind == "O" and array.size:
            filler = _julia_typed_zero(array.flat[0])
        else:
            filler = np.zeros((), dtype=array.dtype)[()]
    shape = tuple(
        size + before + after
        for size, (before, after) in zip(array.shape, pairs, strict=True)
    )
    output = _julia_fill(filler, shape)
    dtype = output.dtype
    if filler is None and any(item is not None for item in array.flat):
        raise TypeError("cannot convert a value to nothing for assignment")
    interior = tuple(
        slice(before, before + size)
        for size, (before, _after) in zip(array.shape, pairs, strict=True)
    )
    if isinstance(filler, (tuple, list, np.ndarray)):
        converted = _convert_pad_composite_array(array, filler)
    else:
        converted = _julia_assignment_values(array, output)
    if dtype == np.dtype(object) and isinstance(filler, Fraction):
        rational_values = np.empty(converted.shape, dtype=object)
        for index in np.ndindex(converted.shape):
            item = converted[index]
            if isinstance(item, np.generic):
                item = item.item()
            rational_values[index] = Fraction(item)
        converted = rational_values
    output[interior] = converted
    return output


def _convert_pad_composite_scalar(value: Any, filler: Any) -> Any:
    """Convert one composite cell to ``typeof(filler)`` as Julia does."""

    if isinstance(filler, tuple):
        if not isinstance(value, tuple) or len(value) != len(filler):
            raise TypeError("padout interior cannot convert to tuple filler type")
        return tuple(
            _convert_pad_composite_scalar(item, prototype)
            for item, prototype in zip(value, filler, strict=True)
        )

    if isinstance(filler, (list, np.ndarray)):
        if not isinstance(value, (list, np.ndarray)):
            raise TypeError("padout interior cannot convert to array filler type")
        filler_values = (
            _julia_literal_array(filler)
            if isinstance(filler, list)
            else np.asarray(filler)
        )
        source_values = (
            _julia_literal_array(value)
            if isinstance(value, list)
            else np.asarray(value)
        )
        if source_values.ndim != filler_values.ndim:
            raise TypeError("padout interior array rank differs from filler type")
        converted = _julia_assignment_values(source_values, filler_values)
        if (
            type(value) is type(filler)
            and source_values.dtype == filler_values.dtype
        ):
            # ``convert(T, x)::T`` returns the original mutable array when it
            # already has exactly the destination array type.
            return value
        if isinstance(filler, list):
            return np.asarray(converted, dtype=filler_values.dtype).tolist()
        return np.asarray(converted, dtype=filler_values.dtype)

    if value is None or filler is None:
        if value is filler:
            return value
        raise TypeError("padout interior cannot convert to Nothing")
    prototype = _julia_fill(filler, ())
    converted = _julia_assignment_values(np.asarray(value), prototype)
    return np.asarray(converted, dtype=prototype.dtype).reshape(())[()]


def _convert_pad_composite_array(array: np.ndarray, filler: Any) -> np.ndarray:
    converted = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        converted[index] = _convert_pad_composite_scalar(array[index], filler)
    return converted


def latticeDisplacement(lattice: Any) -> np.ndarray:
    """Return each axis coordinate at its FFT-shifted center index."""

    axes = as_lattice(lattice)
    # Julia's ``floor(length(l) / 2)`` is Float64 even when coordinates and
    # steps are Float32.  An integer multiplier would trigger NumPy's weak
    # scalar rules and incorrectly retain Float32 arithmetic.
    centers = []
    for axis in axes:
        step = _step(axis)
        multiplier: Any = np.floor(len(axis) / 2)
        if isinstance(step, Decimal):
            # Julia promotes the Float64 result of ``floor(length / 2)`` to
            # BigFloat before multiplying by a BigFloat range step.  Python
            # deliberately forbids float * Decimal, so spell out that exact
            # promotion for the BigFloat counterpart only.
            multiplier = Decimal.from_float(float(multiplier))
        if isinstance(step, (_MPC, _MPFRComplex)):
            with _bigfloat_context():
                promoted_step = _MPC(
                    _to_mpfr(step.real), _to_mpfr(step.imag)
                )
                promoted_multiplier = _to_mpfr(multiplier)
                first = axis[0]
                promoted_first = _MPC(
                    _to_mpfr(first.real), _to_mpfr(first.imag)
                )
                centers.append(
                    promoted_first + promoted_multiplier * promoted_step
                )
        elif isinstance(step, (_MPFR, _MPQ, _MPZ)):
            with _bigfloat_context():
                promoted_step = _to_mpfr(step)
                promoted_multiplier = _to_mpfr(multiplier)
                centers.append(
                    _to_mpfr(axis[0])
                    + promoted_multiplier * promoted_step
                )
        else:
            centers.append(axis[0] + multiplier * step)
    return np.asarray(centers)


def toDim(vector: Any, dimension: int, total_dimensions: int) -> np.ndarray:
    """Reshape a vector along a Julia-style one-based dimension number."""

    if not _is_julia_platform_int(dimension) or not _is_julia_platform_int(
        total_dimensions
    ):
        raise TypeError("toDim dimensions must be Julia platform Int values.")
    if isinstance(vector, tuple) or isinstance(vector, (str, bytes)) or np.isscalar(
        vector
    ):
        raise TypeError("toDim expects a Julia array or AbstractRange value.")
    if not isinstance(vector, (list, range, np.ndarray)):
        raise TypeError("toDim expects a Julia array or AbstractRange value.")
    dimension = int(dimension)
    total_dimensions = int(total_dimensions)
    # Base.reshape returns an AbstractRange unchanged when its requested
    # one-dimensional shape already matches.  Besides preserving object
    # identity, this keeps StepRangeLen's high-precision reference/step and
    # arbitrary-precision range metadata available to one-dimensional ldot.
    if (
        isinstance(vector, LatticeAxis)
        and dimension == 1
        and total_dimensions == 1
    ):
        return vector
    values = (
        _julia_literal_array(vector)
        if isinstance(vector, list)
        else np.asarray(vector)
    )
    if values.ndim != 1:
        values = values.reshape(-1, order="F")
    # The Julia source does not bounds-check d against n. If d lies outside
    # 1:n every generated dimension is singleton; reshape itself decides
    # whether that shape is possible.
    shape = [
        len(values) if index == dimension else 1
        for index in range(1, total_dimensions + 1)
    ]
    return values.reshape(shape, order="F")


def r2(lattice: Any) -> np.ndarray:
    """Evaluate the squared Euclidean radius on a tensor-product lattice."""

    axes = as_lattice(lattice)
    if not axes:
        raise TypeError("r2 on a zero-dimensional lattice is ambiguous in Julia.")
    output: np.ndarray | None = None
    for i, axis in enumerate(axes, start=1):
        values = np.asarray(axis)
        squared = _julia_array_array_operation(
            values, values, np.multiply
        )
        term = toDim(squared, i, len(axes))
        output = (
            term
            if output is None
            else _julia_array_array_operation(output, term, np.add)
        )
    assert output is not None
    # Julia spells this reduction as broadcasted splatted ``.+``. With one
    # axis that is unary ``+``; uniquely, ``+(::Bool)`` widens to platform
    # Int even though ``Bool * Bool`` remains Bool.
    if len(axes) == 1 and output.dtype == np.dtype(np.bool_):
        return output.astype(np.int64)
    return output


def ldot(left: Any, right: Any) -> np.ndarray:
    """Dot a vector with lattice coordinates, accepting either argument order."""

    if (
        isinstance(left, tuple)
        and isinstance(right, tuple)
        and not left
        and not right
    ):
        raise TypeError("ldot((), ()) is ambiguous in Julia.")
    if _looks_like_lattice(left):
        lattice = as_lattice(left)
        vector_input = right
    else:
        vector_input = left
        lattice = as_lattice(right)

    # A Julia ``NTuple{N,Real}`` may be heterogeneous.  Converting its Python
    # counterpart to one NumPy array first would choose a common NumPy dtype
    # before the range/scalar products run (for example Int64 + Float32 would
    # become Float64). Preserve tuple elements exactly. A Python list models a
    # Julia vector literal and therefore uses Julia's literal promotion;
    # explicitly typed ndarrays retain their declared element type.
    if isinstance(vector_input, tuple):
        coefficients = vector_input
        if not all(_is_real_number(value) for value in coefficients):
            raise TypeError("Lattice dot-product coefficients must be real.")
    else:
        if isinstance(vector_input, list):
            vector = _julia_literal_array(vector_input)
        elif isinstance(vector_input, np.ndarray):
            vector = _require_dense_ndarray(vector_input, "ldot vector")
        else:
            raise TypeError(
                "ldot vector input must be a list, dense NumPy vector, "
                "or real tuple"
            )
        if vector.ndim != 1:
            raise ValueError("Vector length != Lattice dimension.")
        coefficients = tuple(vector)

    if len(coefficients) != len(lattice):
        raise ValueError("Vector length != Lattice dimension.")
    if not lattice:
        raise TypeError(
            "ldot on a zero-dimensional lattice reaches ambiguous +() in Julia."
        )
    output: np.ndarray | None = None
    for i, (coefficient, axis) in enumerate(
        zip(coefficients, lattice, strict=True), start=1
    ):
        # Julia multiplies the AbstractRange before reshaping it.  For a
        # Float16/Float32 StepRangeLen that operation uses the range's retained
        # Float64 reference/step, not its already-rounded materialization.
        product = _logical_axis_scalar_operation(axis, coefficient, np.multiply)
        term = toDim(product, i, len(lattice))
        output = (
            term
            if output is None
            else _julia_array_array_operation(output, term, np.add)
        )
    if output is None:
        raise TypeError(
            "ldot on a zero-dimensional lattice reaches ambiguous +() in Julia."
        )
    # Julia expresses ldot as broadcasted splatted ``.+``.  The one-axis case
    # is unary plus, for which ``+(::Bool)`` returns platform Int.
    if len(lattice) == 1 and output.dtype == np.dtype(np.bool_):
        return output.astype(np.int64)
    return output


def Nyquist(lattice: Any) -> tuple[Any, ...]:
    """Return the per-axis Nyquist coordinate ``1 / (2*step)``."""

    output: list[Any] = []
    for axis in as_lattice(lattice):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            doubled = _julia_array_scalar_operation(
                _step(axis), np.int64(2), np.multiply
            ).reshape(())[()]
            output.append(
                _julia_array_scalar_operation(
                    np.asarray(np.int64(1)), doubled, np.divide
                ).reshape(())[()]
            )
    return tuple(output)


# OpenLibm/fdlibm constants used by Julia for Float64 trigonometry.  FrFTBasis
# raises nominal Fourier eigenvalues to a fractional power; an ulp difference
# in a DFT entry can put a nominal -1 on the other side of the complex-power
# branch cut and change the answer by order one.  Reproducing the locked
# Julia runtime's elementary operations therefore matters here rather than
# being cosmetic numerical noise.
_SIN_S1 = -1.66666666666666324348e-01
_SIN_S2 = 8.33333333332248946124e-03
_SIN_S3 = -1.98412698298579493134e-04
_SIN_S4 = 2.75573137070700676789e-06
_SIN_S5 = -2.50507602534068634195e-08
_SIN_S6 = 1.58969099521155010221e-10
_COS_C1 = 4.16666666666666019037e-02
_COS_C2 = -1.38888888888741095749e-03
_COS_C3 = 2.48015872894767294178e-05
_COS_C4 = -2.75573143513906633035e-07
_COS_C5 = 2.08757232129817482790e-09
_COS_C6 = -1.13596475577881948265e-11
_INV_PIO2 = 6.36619772367581382433e-01
_PIO2_1 = 1.57079632673412561417e00
_PIO2_1T = 6.07710050650619224932e-11
_PIO2_2 = 6.07710050630396597660e-11
_PIO2_2T = 2.02226624879595063154e-21
_PIO2_3 = 2.02226624871116645580e-21
_PIO2_3T = 8.47842766036889956997e-32


def _float_high_word(value: float) -> int:
    return struct.unpack(">I", struct.pack(">d", value)[:4])[0]


def _kernel_sin(x: float, tail: float, has_tail: bool) -> float:
    z = x * x
    w = z * z
    remainder = _SIN_S2 + z * (_SIN_S3 + z * _SIN_S4) + z * w * (
        _SIN_S5 + z * _SIN_S6
    )
    v = z * x
    if not has_tail:
        return x + v * (_SIN_S1 + z * remainder)
    return x - ((z * (0.5 * tail - v * remainder) - tail) - v * _SIN_S1)


def _kernel_cos(x: float, tail: float) -> float:
    z = x * x
    w = z * z
    remainder = z * (_COS_C1 + z * (_COS_C2 + z * _COS_C3)) + w * w * (
        _COS_C4 + z * (_COS_C5 + z * _COS_C6)
    )
    half_z = 0.5 * z
    one_minus = 1.0 - half_z
    return one_minus + (((1.0 - one_minus) - half_z) + (z * remainder - x * tail))


def _reduce_pio2(x: float) -> tuple[int, float, float]:
    """fdlibm's medium-size argument reduction used by Julia OpenLibm."""

    high_x = _float_high_word(x)
    exponent_x = (high_x & 0x7FFFFFFF) >> 20
    magnitude = abs(x)
    quadrant = int(magnitude * _INV_PIO2 + 0.5)
    quadrant_float = float(quadrant)
    remainder = magnitude - quadrant_float * _PIO2_1
    correction = quadrant_float * _PIO2_1T
    head = remainder - correction
    cancellation = exponent_x - (
        (_float_high_word(head) & 0x7FFFFFFF) >> 20
    )
    if cancellation > 16:
        intermediate = remainder
        correction = quadrant_float * _PIO2_2
        remainder = intermediate - correction
        correction = quadrant_float * _PIO2_2T - (
            (intermediate - remainder) - correction
        )
        head = remainder - correction
        cancellation = exponent_x - (
            (_float_high_word(head) & 0x7FFFFFFF) >> 20
        )
        if cancellation > 49:
            intermediate = remainder
            correction = quadrant_float * _PIO2_3
            remainder = intermediate - correction
            correction = quadrant_float * _PIO2_3T - (
                (intermediate - remainder) - correction
            )
            head = remainder - correction
    tail = (remainder - head) - correction
    if x < 0:
        return -quadrant, -head, -tail
    return quadrant, head, tail


def _openlibm_sincos(x: float) -> tuple[float, float]:
    """Return Julia/OpenLibm-compatible Float64 sine and cosine."""

    if abs(x) <= math.pi / 4:
        return _kernel_sin(x, 0.0, False), _kernel_cos(x, 0.0)
    quadrant, head, tail = _reduce_pio2(x)
    sin_head = _kernel_sin(head, tail, True)
    cos_head = _kernel_cos(head, tail)
    branch = quadrant & 3
    if branch == 0:
        return sin_head, cos_head
    if branch == 1:
        return cos_head, -sin_head
    if branch == 2:
        return -sin_head, -cos_head
    return -cos_head, sin_head


def shiftedDFTBasis(n: int) -> np.ndarray:
    if not isinstance(n, Integral):
        raise TypeError("shiftedDFTBasis expects an integer.")
    n = int(n)
    if n <= 0:
        # ``0:n-1`` is empty for every nonpositive n, so Julia's matrix
        # comprehension never evaluates the otherwise-invalid sqrt/division.
        return np.empty((0, 0), dtype=np.complex128)
    shift = n // 2
    scale = 1.0 / math.sqrt(n)
    output = np.empty((n, n), dtype=np.complex128)
    for j in range(n):
        for k in range(n):
            # Preserve Julia's left-associated Float64 operation order.
            angle = -2.0 * math.pi
            angle *= j - shift
            angle *= k - shift
            angle /= n
            sine, cosine = _openlibm_sincos(angle)
            output[j, k] = scale * complex(cosine, sine)
    return output


def hermiteBasis(n: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(n, Integral):
        raise TypeError("hermiteBasis expects an integer.")
    raise NotImplementedError(
        "hermiteBasis is unusable in the audited Julia source because it "
        "passes a coordinate range to shiftedDFTBasis(::Integer)"
    )


def FrFTBasis(n: int, alpha: Real) -> np.ndarray:
    if not _is_real_number(alpha):
        raise TypeError("FrFTBasis alpha must be real.")
    if not isinstance(n, Integral):
        raise TypeError("FrFTBasis expects an integer size.")
    raise NotImplementedError(
        "FrFTBasis depends on the unusable hermiteBasis implementation in "
        "the audited Julia source"
    )


def wigner_fft(signal: Any) -> np.ndarray:
    """Return the Julia-compatible ``n × 2n`` real Wigner distribution."""

    # Julia declares a concrete ``Vector{T} where T<:Number`` method, not an
    # AbstractVector method. Python lists are the natural Vector literal;
    # tuples and ranges intentionally do not match that dispatch.
    if isinstance(signal, tuple) or isinstance(signal, range) or not isinstance(
        signal, (list, np.ndarray)
    ):
        raise TypeError("wigner_fft expects a dense numeric vector.")
    values = (
        _julia_literal_array(signal)
        if isinstance(signal, list)
        else _require_dense_ndarray(signal, "wigner_fft signal")
    )
    if values.ndim != 1:
        raise TypeError("wigner_fft expects a one-dimensional vector.")
    if (
        isinstance(signal, np.ndarray)
        and values.dtype.kind == "O"
        and not any(
            isinstance(value, (Fraction, Decimal))
            for value in values.flat
        )
    ):
        # Explicit object storage for ordinary machine numbers corresponds
        # to Julia ``Vector{Any}``, which does not satisfy
        # ``Vector{T} where T<:Number``. Fraction and Decimal are retained
        # because NumPy has no concrete Rational or BigFloat dtype for them.
        raise TypeError(
            "wigner_fft object arrays must represent a Fraction or Decimal "
            "numeric vector."
        )
    if values.dtype.kind not in "buifcO" or (
        values.dtype.kind == "O"
        and not all(
            isinstance(value, (Number, Decimal, Fraction))
            or hasattr(value, "__complex__")
            for value in values.flat
        )
    ):
        raise TypeError("wigner_fft expects a numeric element type.")
    values = values.astype(np.complex128, copy=False)
    n = len(values)
    if n == 0:
        raise ValueError("wigner_fft requires a nonempty signal.")
    prefft = np.zeros((n, 2 * n), dtype=np.complex128)
    i, j = np.indices(prefft.shape)
    lag = j - i
    second = i + j - n
    valid = (lag >= 1) & (lag <= n) & (second >= 0) & (second < n)
    prefft[valid] = np.conjugate(values[n - lag[valid]]) * values[second[valid]]
    transformed = np.fft.ifftshift(
        np.fft.fft(np.fft.fftshift(prefft, axes=1), axis=1), axes=1
    ) / (2 * n)
    return np.real(transformed)


__all__ = [
    "Amplitude",
    "ComplexAmp",
    "ComplexAmplitude",
    "ComplexPhase",
    "FieldVal",
    "FrFTBasis",
    "Generic",
    "Intensity",
    "LF",
    "Lattice",
    "LatticeField",
    "Modulus",
    "Nyquist",
    "Phase",
    "RealAmp",
    "RealAmplitude",
    "RealPhase",
    "S1Phase",
    "UPhase",
    "UnwrappedPhase",
    "hermiteBasis",
    "latticeDisplacement",
    "ldot",
    "naturalize",
    "natlat",
    "natrange",
    "normalizeLF",
    "padout",
    "phasor",
    "r2",
    "shiftedDFTBasis",
    "square",
    "subfield",
    "sublattice",
    "toDim",
    "wigner_fft",
    "wrap",
]
