"""Focused regressions for Julia-compatible lattice and padding behavior."""

from decimal import Decimal, localcontext
from fractions import Fraction
import operator

import numpy as np
import pytest

from slmtools._bigfloat import _MPFR, _to_mpfr
from slmtools.lattice_field import (
    ComplexAmplitude,
    ComplexPhase,
    DomainError,
    Generic,
    Intensity,
    LF,
    LatticeAxis,
    Modulus,
    RealPhase,
    _axis,
    _julia_literal_array,
    elq,
    phasor,
    square,
    wrap,
)
from slmtools.dual_lattices import dualLattice, dualShiftLattice, ldq
from slmtools.lattice_utils import _step, latticeDisplacement, ldot, padout, r2
from slmtools.misc import (
    SchroffError,
    _fraction_int64_divide,
    centroid,
    clip,
    nabs,
    ramp,
    window,
)


def test_lattice_equality_uses_array_norm_isapprox_per_axis() -> None:
    exact = (np.array([0.0, 1.0]),)
    differs_from_zero = (np.array([1e-12, 1.0]),)

    # Julia broadcasts isapprox over the tuple of axes, so each complete axis
    # is compared with LinearAlgebra's norm-based array method.
    elq(exact, differs_from_zero)
    with pytest.raises(DomainError):
        elq(exact, (np.array([1e-6, 1.0]),))

    # Julia's default scalar tolerance remains relative away from zero.
    elq((np.array([1.0, 2.0]),), (np.array([1.0 + 1e-12, 2.0]),))

    # Exact numeric types have zero default relative tolerance in Julia.
    with pytest.raises(DomainError):
        elq(
            (np.array([100_000_000], dtype=np.int64),),
            (np.array([100_000_001], dtype=np.int64),),
        )

    # Mixed floating-point comparisons use the less precise type's epsilon.
    elq(
        (np.array([1.0], dtype=np.float32),),
        (np.array([1.0 + 1e-5], dtype=np.float64),),
    )

    # The object-backed Rational path must still use array-level norm
    # isapprox, not independent scalar comparisons (the zero coordinate is
    # the distinguishing case).
    elq(
        (np.array([Fraction(0), Fraction(1)], dtype=object),),
        (np.array([1e-9, 1.000000001]),),
    )


def test_lattice_equality_ignores_hidden_singleton_step_metadata() -> None:
    first = (LatticeAxis([0.0], step_hint=np.float32(0.25)),)
    second = (LatticeAxis([0.0], step_hint=np.float32(9.0)),)
    elq(first, second)

    # ldq likewise compares the materialized dual axis, not its hidden step.
    ldq(first, second)


def test_float32_logical_step_and_factor_generated_lattices() -> None:
    logical_step = np.float32(1 / 3)
    materialized = np.linspace(-1, 1, 7, dtype=np.float32)
    axis = LatticeAxis(materialized, step_hint=logical_step)
    assert _step(axis) is logical_step

    # An inferred first difference is not authoritative for an irregular axis.
    with pytest.raises(ValueError, match="regularly spaced"):
        _step(LatticeAxis(np.array([0.0, 1.0, 3.0], dtype=np.float32)))

    padded = padout(axis, 1)
    assert padded.dtype == np.dtype(np.float32)
    assert _step(padded).dtype == np.dtype(np.float32)

    dotted = ldot(np.array([2], dtype=np.int64), (axis,))
    assert dotted.dtype == np.dtype(np.float32)


def test_ldot_preserves_tuple_vector_literal_and_typed_array_semantics() -> None:
    axis = LatticeAxis(
        np.asarray([0.0, 1.0], dtype=np.float32),
        step_hint=np.float32(1),
    )
    lattice = (axis, axis)
    heterogeneous = (np.int64(100_000_001), np.float32(1))

    # Julia's NTuple keeps its element types, while an ordinary vector literal
    # promotes Int64/Float32 to Float32 before ldot. Both produce this locked
    # Matrix{Float32} oracle for Float32 ranges.
    expected = np.asarray(
        [[0.0, 1.0], [100_000_000.0, 100_000_000.0]],
        dtype=np.float32,
    )
    tuple_result = ldot(heterogeneous, lattice)
    list_result = ldot(list(heterogeneous), lattice)
    object_vector_result = ldot(
        np.asarray(heterogeneous, dtype=object), lattice
    )
    for result in (tuple_result, list_result, object_vector_result):
        assert result.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(result, expected)

    # An explicitly typed Vector{Float64} is already a declared element type;
    # it must not be reinterpreted as a Python literal and narrowed.
    typed_result = ldot(
        np.asarray(heterogeneous, dtype=np.float64), lattice
    )
    assert typed_result.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        typed_result,
        np.asarray(
            [[0.0, 1.0], [100_000_001.0, 100_000_002.0]],
            dtype=np.float64,
        ),
    )


def test_ldot_single_bool_axis_applies_julia_unary_plus() -> None:
    axis = LatticeAxis(
        np.asarray([False, True], dtype=np.bool_),
        step_hint=np.bool_(True),
    )
    result = ldot((True,), (axis,))
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.asarray([0, 1], dtype=np.int64))


def test_python_rational_list_literal_promotes_integers_to_fraction() -> None:
    result = _julia_literal_array([Fraction(1, 3), np.int64(2)])
    assert result.dtype == np.dtype(object)
    np.testing.assert_array_equal(
        result,
        np.asarray([Fraction(1, 3), Fraction(2, 1)], dtype=object),
    )
    assert all(isinstance(value, Fraction) for value in result)


def test_float32_dual_lattices_match_julia_dtype_promotion() -> None:
    axis = LatticeAxis(
        np.array([-1.0, -0.7, -0.4, -0.1, 0.2], dtype=np.float32),
        step_hint=np.float32(0.3),
    )

    default_positive = dualLattice((axis,))[0]
    explicit32_shifted = dualShiftLattice((axis,), np.float32(1))[0]
    assert default_positive.dtype == np.dtype(np.float32)
    assert explicit32_shifted.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        default_positive,
        np.array([0, 2 / 3, 4 / 3, 2, 8 / 3], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        explicit32_shifted,
        np.array([-4 / 3, -2 / 3, 0, 2 / 3, 4 / 3], dtype=np.float32),
    )

    # A Python float corresponds to Julia Float64 and therefore promotes.
    assert dualLattice((axis,), 1.0)[0].dtype == np.dtype(np.float64)
    assert dualShiftLattice((axis,), np.float64(1))[0].dtype == np.dtype(
        np.float64
    )


@pytest.mark.parametrize("operation", [operator.add, operator.mul, operator.sub, operator.truediv])
def test_float32_field_scalar_promotion_matches_julia(operation: object) -> None:
    field = LF[RealPhase](np.array([2.0, 4.0], dtype=np.float32), (range(2),))

    for scalar in (2, np.int64(2), np.float32(2)):
        result = operation(field, scalar)
        assert result.dtype == np.dtype(np.float32)

    for scalar in (2.0, np.float64(2)):
        with pytest.raises(TypeError):
            operation(field, scalar)


def test_complex64_field_scalar_promotion_matches_julia() -> None:
    field = LF[ComplexPhase](
        np.array([1.0 + 2.0j, 3.0 - 1.0j], dtype=np.complex64),
        (range(2),),
    )

    for scalar in (2, np.int64(2), np.float32(2), np.complex64(2j)):
        assert (field + scalar).dtype == np.dtype(np.complex64)
        assert (scalar + field).dtype == np.dtype(np.complex64)
        assert (field * scalar).dtype == np.dtype(np.complex64)
        assert (scalar * field).dtype == np.dtype(np.complex64)

    for scalar in (2.0, np.float64(2), 2j, np.complex128(2j)):
        with pytest.raises(TypeError):
            _ = field + scalar

    real_field = LF[RealPhase](np.ones(2, dtype=np.float64), (range(2),))
    with pytest.raises(TypeError):
        _ = real_field + np.complex128(1j)


def test_integer_field_division_and_widening_fail_like_typed_julia_constructor() -> None:
    int32_field = LF[Generic](np.array([2, 4], dtype=np.int32), (range(2),))
    with pytest.raises(TypeError):
        _ = int32_field + np.int64(1)
    with pytest.raises(TypeError):
        _ = int32_field / np.int32(2)

    uint64_field = LF[Generic](np.array([2, 4], dtype=np.uint64), (range(2),))
    assert (uint64_field + np.int64(1)).dtype == np.dtype(np.uint64)

    int64_field = LF[Generic](np.array([2, 4], dtype=np.int64), (range(2),))
    with pytest.raises(TypeError):
        _ = int64_field + np.uint64(1)


def test_padout_rejects_inexact_assignment_instead_of_truncating() -> None:
    with pytest.raises(ValueError, match="Inexact assignment"):
        padout(np.array([1.5]), 1, filler=0)
    with pytest.raises(ValueError, match="Inexact assignment"):
        padout(np.array([1.0 + 2.0j]), 1, filler=0.0)


def test_padout_preserves_julia_filler_type_when_assignment_is_exact() -> None:
    default_zero = padout(np.array([1.0, 2.0], dtype=np.float32), 1)
    assert default_zero.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(default_zero, [0.0, 1.0, 2.0, 0.0])

    integral_float_data = padout(np.array([1.0, 2.0]), 1, filler=0)
    assert integral_float_data.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(integral_float_data, [0, 1, 2, 0])

    promoted_by_filler = padout(np.array([1, 2]), 1, filler=0.5)
    assert promoted_by_filler.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(promoted_by_filler, [0.5, 1.0, 2.0, 0.5])

    # Julia converts Rational/BigFloat values approximately when assigning
    # them to a floating destination; exact object round-tripping is not
    # required for that conversion.
    rational_data = padout(
        np.array([Fraction(1, 3)], dtype=object), 1, filler=0.0
    )
    decimal_data = padout(
        np.array([Decimal("0.1")], dtype=object), 1, filler=np.float32(0)
    )
    assert rational_data.dtype == np.dtype(np.float64)
    assert decimal_data.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(rational_data, [0.0, 1 / 3, 0.0])
    np.testing.assert_array_equal(
        decimal_data, np.array([0.0, 0.1, 0.0], dtype=np.float32)
    )


def test_object_exact_number_assignment_uses_julia_numeric_conversion() -> None:
    floating = LF[Generic](np.zeros(2, dtype=np.float64), (range(2),))
    floating[0] = Fraction(1, 3)
    floating[1] = Decimal("0.1")
    np.testing.assert_array_equal(floating.data, [1 / 3, 0.1])

    complex_field = LF[Generic](
        np.zeros(2, dtype=np.complex128), (range(2),)
    )
    complex_field[0] = Fraction(1, 3)
    complex_field[1] = Decimal("0.1")
    np.testing.assert_array_equal(complex_field.data, [1 / 3 + 0j, 0.1 + 0j])

    integer = LF[Generic](np.zeros(2, dtype=np.int64), (range(2),))
    integer[0] = Fraction(3, 1)
    integer[1] = Decimal("4")
    np.testing.assert_array_equal(integer.data, [3, 4])
    for inexact in (Fraction(1, 3), Decimal("0.1"), 1 + 2j):
        with pytest.raises(ValueError, match="Inexact assignment"):
            integer[0] = inexact

    # Rational{BigInt} conversion saturates/underflows at machine-float
    # boundaries; Python's Fraction.__float__ raises for the overflow case, so
    # the shared assignment helper must translate it explicitly.
    extremes = LF[Generic](np.zeros(3, dtype=np.float64), (range(3),))
    extremes[0] = Fraction(10**400)
    extremes[1] = Fraction(-(10**400))
    extremes[2] = Fraction(1, 10**4000)
    np.testing.assert_array_equal(extremes.data, [np.inf, -np.inf, 0.0])


def test_range_padout_supports_julia_negative_coordinate_cropping() -> None:
    axis = padout(range(1, 5), -1)
    np.testing.assert_array_equal(axis, [2, 3])
    assert _step(axis) == 1

    empty = padout(range(1, 5), -2)
    assert len(empty) == 0
    assert _step(empty) == 1


def test_lattice_displacement_uses_julia_float64_center_multiplier() -> None:
    axis = LatticeAxis(
        np.array([100000.0, 100000.1, 100000.2, 100000.3], dtype=np.float32),
        step_hint=np.float32(0.1),
    )
    displacement = latticeDisplacement((axis,))
    assert displacement.dtype == np.dtype(np.float64)
    assert displacement[0] == np.float64(100000.20000000298)


def test_irregular_coordinate_arrays_are_not_lattices() -> None:
    with pytest.raises(ValueError, match="regularly spaced"):
        LF[Generic](np.ones(3), (np.array([0.0, 1.0, 3.0]),))
    with pytest.raises(ValueError, match="inconsistent with their logical step"):
        LF[Generic](
            np.ones(3),
            (LatticeAxis([0.0, 1.0, 3.0], step_hint=1.0),),
        )


def test_fraction_lattices_remain_regular_and_exact_through_dualization() -> None:
    axis = np.array([Fraction(0), Fraction(1, 3)], dtype=object)
    field = LF[Generic](np.ones(2), (axis,))
    assert _step(field.L[0]) == Fraction(1, 3)

    dual = dualShiftLattice(field.L, Fraction(1, 2))[0]
    assert dual.dtype == np.dtype(object)
    assert list(dual) == [Fraction(-3, 4), Fraction(0)]
    assert _step(dual) == Fraction(3, 4)

    with pytest.raises(ValueError, match="regularly spaced"):
        LF[Generic](
            np.ones(3),
            (
                np.array(
                    [Fraction(0), Fraction(1, 3), Fraction(3, 4)],
                    dtype=object,
                ),
            ),
        )


def test_fraction_centroid_and_decimal_flambda_preserve_exact_scalars() -> None:
    center = centroid(
        np.array([Fraction(1), Fraction(2)], dtype=object)
    )
    assert center.dtype == np.dtype(object)
    assert center[0] == Fraction(5, 3)

    decimal_center = centroid(
        np.array([Decimal("1"), Decimal("2")], dtype=object)
    )
    assert decimal_center.dtype == np.dtype(object)
    assert decimal_center[0] == Decimal(5) / Decimal(3)

    wavelength = Decimal("1.25")
    partial = LF[Intensity](np.ones(2), (range(2),), wavelength)
    full = LF[Intensity, np.float64, 1](
        np.ones(2, dtype=np.float64), (range(2),), wavelength
    )
    assert partial.flambda is wavelength
    assert full.flambda is wavelength


def test_decimal_lattice_and_flambda_isapprox_use_context_tolerance() -> None:
    data = np.asarray([Decimal(0), Decimal(1)], dtype=object)
    close_left = LF[RealPhase, object, 1](
        data.copy(),
        (np.asarray([Decimal(1), Decimal(2)], dtype=object),),
        Decimal(1),
    )
    close_right = LF[RealPhase, object, 1](
        data.copy(),
        (
            np.asarray(
                [Decimal("1.000000000000001"), Decimal(2)], dtype=object
            ),
        ),
        Decimal("1.000000000000001"),
    )
    assert elq(close_left, close_right) is None

    far = LF[RealPhase, object, 1](
        data.copy(),
        (np.asarray([Decimal("1.0000000001"), Decimal(2)], dtype=object),),
        Decimal(1),
    )
    with pytest.raises(DomainError):
        elq(close_left, far)

    with localcontext() as context:
        context.prec = 77  # Decimal counterpart of Julia's 256-bit BigFloat.
        base = np.asarray([Decimal(1), Decimal(0)], dtype=object)
        elq(
            (base,),
            (
                np.asarray(
                    [Decimal(1) + Decimal("1e-40"), Decimal(0)],
                    dtype=object,
                ),
            ),
        )
        with pytest.raises(DomainError):
            elq(
                (base,),
                (
                    np.asarray(
                        [Decimal(1) + Decimal("8e-39"), Decimal(0)],
                        dtype=object,
                    ),
                ),
            )


def test_decimal_schroff_error_promotes_default_float_threshold() -> None:
    values = np.asarray([Decimal(1), Decimal(2)], dtype=object)
    field = LF[Intensity, object, 1](
        values,
        (np.asarray([Decimal(0), Decimal(1)], dtype=object),),
    )
    assert SchroffError(field, field) == Decimal(0)


def test_mixed_decimal_lattice_helpers_follow_bigfloat_promotion() -> None:
    decimal_axis = np.asarray([Decimal(0), Decimal(1)], dtype=object)
    dual = dualShiftLattice((decimal_axis,))[0]
    assert dual.tolist() == [Decimal("-0.5"), Decimal(0)]

    intensity = LF[Intensity](np.asarray([1.0, 3.0]), (decimal_axis,))
    center = centroid(intensity)
    assert center.dtype == np.dtype(object)
    assert center[0] == Decimal("0.75")

    dotted = ldot(np.asarray([2.0]), (decimal_axis,))
    assert dotted.tolist() == [Decimal(0), Decimal(2)]

    radii = r2((np.asarray([0.0, 1.0]), decimal_axis))
    assert radii.dtype == np.dtype(object)
    assert radii.tolist() == [
        [Decimal(0), Decimal(1)],
        [Decimal(1), Decimal(2)],
    ]

    fraction_axis = np.asarray([Fraction(0), Fraction(1)], dtype=object)
    fraction_intensity = LF[Intensity](
        np.asarray([1.0, 3.0]), (fraction_axis,)
    )
    fraction_center = centroid(fraction_intensity)
    assert fraction_center.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(fraction_center, [0.75])
    fraction_dot = ldot(np.asarray([2.0]), (fraction_axis,))
    assert fraction_dot.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(fraction_dot, [0.0, 2.0])


def test_working_rational_and_bigfloat_misc_paths() -> None:
    rational = np.array([Fraction(3), Fraction(4)], dtype=object)
    normalized = nabs(rational)
    assert normalized.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(normalized, np.array([0.6, 0.8]))

    decimal_axis = _axis(
        np.array([Decimal("0.1"), Decimal("0.3"), Decimal("0.5")], dtype=object),
        Decimal("0.2"),
    )
    displacement = latticeDisplacement((decimal_axis,))
    assert displacement.dtype == np.dtype(object)
    assert displacement[0] == Decimal("0.3")

    np.testing.assert_array_equal(window(rational, 1)[0], [2])
    np.testing.assert_array_equal(
        window(np.array([Decimal(3), Decimal(4)], dtype=object), 1)[0],
        [2],
    )

    exact_intensity = LF[Intensity, object, 1](rational.copy(), (range(2),))
    assert SchroffError(exact_intensity, exact_intensity, threshold=0) == 0.0

    rational_signed = np.array([Fraction(-1), Fraction(2)], dtype=object)
    assert all(isinstance(ramp(value), Fraction) for value in rational_signed)
    assert all(
        isinstance(clip(value, Fraction(0)), Fraction)
        for value in rational_signed
    )
    clipped_field = LF[Intensity](rational_signed, (range(2),))
    assert isinstance(clipped_field.data[0], Fraction)
    rational_padded = padout(rational_signed, 1)
    assert isinstance(rational_padded[0], Fraction)


def test_bigfloat_nabs_and_centroid_propagate_nonfinite_values() -> None:
    with localcontext() as context:
        context.prec = 77
        zero_normalized = nabs(
            np.asarray([Decimal(0), Decimal(0)], dtype=object)
        )
        assert zero_normalized.dtype == np.dtype(object)
        assert all(
            isinstance(value, Decimal) and value.is_nan()
            for value in zero_normalized
        )

        infinite_normalized = nabs(
            np.asarray(
                [Decimal("Infinity"), Decimal(1)], dtype=object
            )
        )
        assert isinstance(infinite_normalized[0], Decimal)
        assert infinite_normalized[0].is_nan()
        assert isinstance(infinite_normalized[1], Decimal)
        assert infinite_normalized[1].is_zero()

        axis = LatticeAxis(
            np.asarray([Decimal(0), Decimal(1)], dtype=object),
            step_hint=Decimal(1),
        )
        field = LF[Intensity, object, 1](
            np.asarray(
                [Decimal("Infinity"), Decimal(1)], dtype=object
            ),
            (axis,),
        )
        center = centroid(field, threshold=Decimal(0))
        assert center.dtype == np.dtype(object)
        assert isinstance(center[0], Decimal)
        assert center[0].is_nan()


def test_rational_int64_nabs_and_centroid_overflow_match_julia() -> None:
    limits = np.iinfo(np.int64)
    maximum = Fraction(int(limits.max))
    minimum = Fraction(int(limits.min))

    with pytest.raises(OverflowError):
        nabs(np.asarray([maximum, maximum], dtype=object))
    with pytest.raises(OverflowError):
        nabs(np.asarray([minimum], dtype=object))
    with pytest.raises(OverflowError):
        centroid(
            np.asarray([maximum, Fraction(1)], dtype=object),
            threshold=Fraction(0),
        )


def test_rational_int64_division_cross_cancels_before_checked_products() -> None:
    limits = np.iinfo(np.int64)
    minimum = Fraction(int(limits.min))
    maximum = Fraction(int(limits.max))

    assert _fraction_int64_divide(minimum, minimum) == Fraction(1)
    assert _fraction_int64_divide(minimum, Fraction(1)) == minimum
    assert _fraction_int64_divide(minimum, maximum) == Fraction(
        int(limits.min), int(limits.max)
    )
    with pytest.raises(OverflowError):
        _fraction_int64_divide(minimum, Fraction(-1))


def test_window_empty_widths_and_integer_dispatch_match_julia() -> None:
    image = np.asarray([0.0, 1.0, 0.0])
    zero = window(image, 0)
    negative = window(image, -2)
    assert image[zero].size == 0
    assert image[negative].size == 0

    # The raw Array method is generic in its scalar width, unlike its tuple
    # and LatticeField overloads.
    np.testing.assert_array_equal(window(image, np.int32(1))[0], [2])
    with pytest.raises(TypeError):
        window(image, 1.9)
    with pytest.raises(TypeError):
        window(np.ones((2, 2)), (1.9, 1.9))

    # Locked Julia constructs CartesianIndices(3:5) here, and indexing the
    # three-element source raises BoundsError.  NumPy slices would otherwise
    # clip to the final element and silently return the wrong window.
    right_biased = np.asarray([0.0, 0.0, 1.0])
    oversized = np.asarray([0.0, 1.0, 0.0])
    with pytest.raises(IndexError):
        right_biased[window(right_biased, 3)]
    with pytest.raises(IndexError):
        oversized[window(oversized, 5)]
    negative_start = np.zeros(10)
    negative_start[0] = 1
    with pytest.raises(IndexError):
        negative_start[window(negative_start, 4)]


@pytest.mark.parametrize("padding", [1.9, np.int32(1), True, (1, 2.9)])
def test_padout_rejects_non_platform_integer_padding(padding: object) -> None:
    with pytest.raises(TypeError, match="Julia Int"):
        padout(np.ones((2, 2)), padding)


def test_rational_real_phase_wrap_uses_julia_float64_exponential() -> None:
    lattice = (range(2),)
    phase = LF[RealPhase, object, 1](
        np.array([Fraction(1, 4), Fraction(1, 2)], dtype=object), lattice
    )
    wrapped = wrap(phase)
    assert wrapped.dtype == np.dtype(np.complex128)
    np.testing.assert_allclose(wrapped.data, [1j, -1.0], atol=2e-16)

    modulus = LF[Modulus, object, 1](
        np.array([Fraction(2), Fraction(3)], dtype=object), lattice
    )
    product = modulus * phase
    assert product.dtype == np.dtype(np.complex128)
    np.testing.assert_allclose(product.data, [2j, -3.0], atol=4e-16)


def test_bigfloat_like_wrap_and_rational_intensity_sqrt() -> None:
    with localcontext() as context:
        # Julia's default BigFloat precision is 256 bits (about 77 decimal
        # digits), so use the matching Decimal context for the golden probe.
        context.prec = 77
        phase = LF[RealPhase, object, 1](
            np.array([Decimal("0.25")], dtype=object), (range(1),)
        )
        value = wrap(phase).data[0]
        assert isinstance(value.real, _MPFR)
        assert isinstance(value.imag, _MPFR)
        assert value.real == _to_mpfr(
            Decimal(
                "6.1232339957367658861303296613750014646403777988362830520960549827724863083977e-17"
            )
        )
        assert value.imag == _to_mpfr(
            Decimal(
                "0.99999999999999999999999999999999812530027167267800669190544314290300696003654"
            )
        )

    intensity = LF[Intensity, object, 1](
        np.array([Fraction(1), Fraction(4)], dtype=object), (range(2),)
    )
    modulus = intensity.sqrt()
    assert modulus.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(modulus.data, [1.0, 2.0])


def test_fraction_scalar_field_arithmetic_uses_julia_machine_promotion() -> None:
    field64 = LF[RealPhase](np.array([1.0, 2.0]), (range(2),))
    half = Fraction(1, 2)
    np.testing.assert_array_equal((field64 * half).data, [0.5, 1.0])
    np.testing.assert_array_equal((field64 + half).data, [1.5, 2.5])
    np.testing.assert_array_equal((field64 - half).data, [0.5, 1.5])
    np.testing.assert_array_equal((field64 / half).data, [2.0, 4.0])

    field32 = LF[RealPhase](np.array([1.0, 2.0], dtype=np.float32), (range(2),))
    assert (field32 * half).dtype == np.dtype(np.float32)

    # Int .* Rational produces Rational data in Julia, which cannot satisfy
    # the method's original full typed LF{S,Int,N} constructor.
    integer = LF[RealPhase](np.array([1, 2]), (range(2),))
    with pytest.raises(TypeError, match="changed the element dtype"):
        integer * half


def test_phasor_has_only_julia_complexf64_scalar_domain() -> None:
    assert phasor(0.0 + 0.0j) == np.complex128(1.0 + 0.0j)
    assert phasor(np.complex128(0.0 + 1.0j)) == np.complex128(0.0 + 1.0j)
    for unsupported in (0, 0.0, np.complex64(1.0j)):
        with pytest.raises(TypeError, match="ComplexF64"):
            phasor(unsupported)

    narrow = LF[ComplexAmplitude, np.complex64, 1](
        np.array([1.0j], dtype=np.complex64), (range(1),)
    )
    with pytest.raises(TypeError, match="ComplexF64"):
        phasor(narrow)


def test_phasor_preserves_multidimensional_field_positions() -> None:
    source = np.reshape(
        np.arange(1, 25, dtype=np.float64)
        + 1j * np.arange(24, 0, -1, dtype=np.float64),
        (6, 4),
        order="F",
    )
    source[2, 1] = 0.0
    amplitude = LF[ComplexAmplitude](source, (range(6), range(4)))

    expected = np.ones_like(source)
    np.divide(source, np.abs(source), out=expected, where=source != 0)
    result = phasor(amplitude)

    assert result.shape == source.shape
    assert result.field_type is ComplexPhase
    np.testing.assert_allclose(result.data.copy(), expected, rtol=1e-15)


def test_unary_plus_matches_julia_supported_tags() -> None:
    lattice = (range(2),)

    # The unary Intensity method routes back through its tagged constructor.
    template = LF[Intensity](np.zeros(2), lattice)
    unusual_intensity = LF[Intensity](np.array([-2.0, 1.0]), template)
    positive_intensity = +unusual_intensity
    assert positive_intensity is not unusual_intensity
    assert positive_intensity.data is not unusual_intensity.data
    np.testing.assert_array_equal(positive_intensity.data, [0.0, 1.0])

    template_phase = LF[RealPhase](np.zeros(2), lattice)
    unusual_amplitude = LF[ComplexAmplitude](
        np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64),
        template_phase,
    )
    positive_amplitude = +unusual_amplitude
    assert positive_amplitude.field_type is ComplexAmplitude
    assert positive_amplitude.dtype == np.dtype(np.complex128)
    np.testing.assert_array_equal(positive_amplitude.data, unusual_amplitude.data)

    boolean_intensity = LF[Intensity, np.bool_, 1](
        np.asarray([True, False]), lattice
    )
    positive_boolean = +boolean_intensity
    assert positive_boolean.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(positive_boolean.data, [1, 0])

    boolean_phase = LF[RealPhase, np.bool_, 1](
        np.asarray([True, False]), lattice
    )
    conjugated_boolean = boolean_phase.conj()
    assert conjugated_boolean.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(conjugated_boolean.data, [-1, 0])


def test_rational_int64_arithmetic_uses_checked_base_algorithms() -> None:
    maximum = np.iinfo(np.int64).max
    lattice = (range(1),)

    left = LF[RealPhase, object, 1](
        np.asarray([Fraction(maximum)], dtype=object), lattice
    )
    right = LF[RealPhase, object, 1](
        np.asarray([Fraction(1)], dtype=object), lattice
    )
    with pytest.raises(OverflowError):
        left + right

    first_phase = LF[ComplexPhase, object, 1](
        np.asarray([Fraction(maximum)], dtype=object), lattice
    )
    second_phase = LF[ComplexPhase, object, 1](
        np.asarray([Fraction(2)], dtype=object), lattice
    )
    with pytest.raises(OverflowError):
        first_phase * second_phase

    # Base cross-cancels before checked multiplication, so this product is
    # valid despite both source numerators touching the boundary.
    cancellable_left = LF[ComplexPhase, object, 1](
        np.asarray([Fraction(maximum, 2)], dtype=object), lattice
    )
    cancellable_right = LF[ComplexPhase, object, 1](
        np.asarray([Fraction(2, maximum)], dtype=object), lattice
    )
    np.testing.assert_array_equal(
        (cancellable_left * cancellable_right).data,
        np.asarray([Fraction(1)], dtype=object),
    )

    modulus = LF[Modulus, object, 1](
        np.asarray([Fraction(maximum)], dtype=object), lattice
    )
    with pytest.raises(OverflowError):
        square(modulus)


def test_binary_intensity_addition_has_value_semantics() -> None:
    lattice = (range(1),)
    first = LF[Intensity, np.int64, 1](np.array([-10]), lattice)
    second = LF[Intensity, np.int64, 1](np.array([5]), lattice)
    third = LF[Intensity, np.int64, 1](np.array([6]), lattice)

    # Julia's binary overload clips the binary result.  Every later Python
    # binary operation must observe that visible value, independent of whether
    # the field has been copied or assigned to an intermediate variable.
    binary = first + second
    np.testing.assert_array_equal(binary.data, [0])
    np.testing.assert_array_equal((binary + third).data, [6])
    np.testing.assert_array_equal((first + second + third).data, [6])
    np.testing.assert_array_equal((binary.copy() + third).data, [6])
    binary[0] = binary[0]
    np.testing.assert_array_equal((binary + third).data, [6])
    assert not hasattr(binary, "_raw_intensity_sum")

    # Python has no n-ary infix dispatch hook: a three-term RealPhase sum is a
    # sequence of the same valid Julia binary overload and stores no expression
    # history. Exact translation of Julia's single variadic call uses
    # explicitly parenthesized Julia binary expression instead.
    phase = LF[RealPhase](np.array([1]), lattice)
    np.testing.assert_array_equal((phase + phase + phase).data, [3])


@pytest.mark.parametrize("tag", [Generic, RealPhase, ComplexPhase, Modulus])
def test_unary_plus_rejects_unsupported_tags(tag: type) -> None:
    field = LF[tag](np.ones(2), (range(2),))
    with pytest.raises(TypeError):
        operator.pos(field)
