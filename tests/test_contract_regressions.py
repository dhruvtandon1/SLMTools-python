from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
from PIL import Image
import pytest

import slmtools as slm
from slmtools.lattice_utils import _step
from slmtools.subimages import (
    padadd as subimage_padadd,
    padmultiple as subimage_padmultiple,
    padout as subimage_padout,
    plotToImage,
    trimWhitespace,
)


class _PngPlot:
    def __init__(self, pixels: np.ndarray) -> None:
        self._pixels = pixels

    def savefig(self, stream: object, *, format: str) -> None:
        assert format == "png"
        Image.fromarray(self._pixels, mode="RGBA").save(stream, format="PNG")


def test_plot_image_and_trim_whitespace_share_unit_color_range() -> None:
    pixels = np.full((3, 3, 4), 255, dtype=np.uint8)
    pixels[1, 1, :3] = 0

    rendered = plotToImage(_PngPlot(pixels))
    assert rendered.dtype == np.float64
    assert rendered.min() == 0.0
    assert rendered.max() == 1.0
    assert trimWhitespace(rendered).shape == (1, 1, 4)

    # HxWx4 bytes represent a Julia matrix of colorants, so channels are
    # normalized; a 2-D integer matrix retains literal numeric semantics.
    assert trimWhitespace(pixels).shape == (1, 1, 4)
    numeric = np.ones((3, 3), dtype=np.uint8)
    numeric[1, 1] = 0
    cropped = trimWhitespace(numeric)
    assert cropped.shape == (1, 1)
    assert not np.shares_memory(cropped, numeric)

    invalid_gray = np.full((3, 3), 255, dtype=np.uint8)
    invalid_gray[1, 1] = 0
    with pytest.raises(ValueError):
        trimWhitespace(invalid_gray)

    near_white = np.full((3, 3), np.nextafter(1.0, 0.0))
    near_white[1, 1] = 0.0
    assert trimWhitespace(near_white).shape == (3, 3)

    colored_pixel = np.asarray([[[128, 64, 32, 255]]], dtype=np.uint8)
    colored = plotToImage(_PngPlot(colored_pixel))
    np.testing.assert_allclose(
        colored[0, 0],
        np.asarray([128, 64, 32, 255], dtype=float) / 255,
        rtol=0,
        atol=0,
    )


def test_subimage_padding_rejects_inexact_fill_conversion() -> None:
    image = np.ones((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="Inexact assignment"):
        subimage_padout(image, (4, 4), fillval=0.5)
    with pytest.raises(ValueError, match="Inexact assignment"):
        subimage_padadd(image, 1, "left", fillval=0.5)
    with pytest.raises(ValueError, match="Inexact assignment"):
        subimage_padmultiple(image, padleft=1, padall=1, fillval=0.5)


def test_subimage_padding_rejects_fractional_or_narrow_integer_widths() -> None:
    image = np.ones((2, 2))
    with pytest.raises(TypeError, match="Julia Int"):
        subimage_padout(image, (4, 4.5))
    with pytest.raises(TypeError, match="Julia Int"):
        subimage_padadd(image, 1.5, "left")
    with pytest.raises(TypeError, match="Julia Int"):
        subimage_padmultiple(image, padall=np.int32(1))


# Generated from locked Julia scalar ``typeof(one(A) + one(B))`` and
# ``typeof(one(A) * one(B))`` probes.  Addition differs from this promotion
# table only for Bool+Bool, whose arithmetic result is Int64; Bool*Bool remains
# Bool.  Keeping the oracle literal prevents the production promotion helper
# from merely testing itself.
_MACHINE_DTYPES = (
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
)
_JULIA_MACHINE_PROMOTION_NAMES = (
    ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128"),
    ("int8", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128"),
    ("int16", "int16", "int16", "int32", "int64", "int16", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128"),
    ("int32", "int32", "int32", "int32", "int64", "int32", "int32", "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128"),
    ("int64", "int64", "int64", "int64", "int64", "int64", "int64", "int64", "uint64", "float16", "float32", "float64", "complex64", "complex128"),
    ("uint8", "uint8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128"),
    ("uint16", "uint16", "uint16", "int32", "int64", "uint16", "uint16", "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128"),
    ("uint32", "uint32", "uint32", "uint32", "int64", "uint32", "uint32", "uint32", "uint64", "float16", "float32", "float64", "complex64", "complex128"),
    ("uint64", "uint64", "uint64", "uint64", "uint64", "uint64", "uint64", "uint64", "uint64", "float16", "float32", "float64", "complex64", "complex128"),
    ("float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float16", "float32", "float64", "complex64", "complex128"),
    ("float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float32", "float64", "complex64", "complex128"),
    ("float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "float64", "complex128", "complex128"),
    ("complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex64", "complex128", "complex64", "complex128"),
    ("complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128", "complex128"),
)


def test_all_machine_dtype_field_pairs_follow_julia_promotion() -> None:
    lattice = (range(1),)
    for row, left_dtype in enumerate(_MACHINE_DTYPES):
        for column, right_dtype in enumerate(_MACHINE_DTYPES):
            expected = np.dtype(
                _JULIA_MACHINE_PROMOTION_NAMES[row][column]
            )
            left_add = slm.LF[slm.RealPhase, left_dtype, 1](
                np.ones(1, dtype=left_dtype), lattice
            )
            right_add = slm.LF[slm.RealPhase, right_dtype, 1](
                np.ones(1, dtype=right_dtype), lattice
            )
            added = left_add + right_add
            expected_add = (
                np.dtype(np.int64)
                if row == 0 and column == 0
                else expected
            )
            assert added.dtype == expected_add, (
                "Julia addition promotion mismatch for "
                f"{np.dtype(left_dtype)} + {np.dtype(right_dtype)}"
            )

            left_multiply = slm.LF[slm.ComplexPhase, left_dtype, 1](
                np.ones(1, dtype=left_dtype), lattice
            )
            right_multiply = slm.LF[slm.ComplexPhase, right_dtype, 1](
                np.ones(1, dtype=right_dtype), lattice
            )
            multiplied = left_multiply * right_multiply
            assert multiplied.dtype == expected, (
                "Julia multiplication promotion mismatch for "
                f"{np.dtype(left_dtype)} * {np.dtype(right_dtype)}"
            )


def test_mixed_machine_arithmetic_matches_julia_values_not_numpy_promotion() -> None:
    lattice = (range(1),)

    bool_sum = (
        slm.LF[slm.RealPhase, np.bool_, 1](
            np.asarray([True]), lattice
        )
        + slm.LF[slm.RealPhase, np.bool_, 1](
            np.asarray([True]), lattice
        )
    )
    assert bool_sum.dtype == np.dtype(np.int64)
    assert bool_sum.data.tolist() == [2]

    wrapped = (
        slm.LF[slm.RealPhase, np.uint64, 1](
            np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64),
            lattice,
        )
        + slm.LF[slm.RealPhase, np.int64, 1](
            np.asarray([1], dtype=np.int64), lattice
        )
    )
    assert wrapped.dtype == np.dtype(np.uint64)
    assert wrapped.data.tolist() == [0]

    narrow_float = (
        slm.LF[slm.RealPhase, np.float32, 1](
            np.asarray([1e8], dtype=np.float32), lattice
        )
        + slm.LF[slm.RealPhase, np.int64, 1](
            np.asarray([1], dtype=np.int64), lattice
        )
    )
    assert narrow_float.dtype == np.dtype(np.float32)
    assert narrow_float.data.tolist() == [np.float32(1e8)]

    overflow = (
        slm.LF[slm.ComplexPhase, np.float16, 1](
            np.asarray([30000], dtype=np.float16), lattice
        )
        * slm.LF[slm.ComplexPhase, np.int16, 1](
            np.asarray([3], dtype=np.int16), lattice
        )
    )
    assert overflow.dtype == np.dtype(np.float16)
    assert np.isposinf(overflow.data[0])

    complex_product = (
        slm.LF[slm.ComplexPhase, np.complex64, 1](
            np.asarray([1 + 2j], dtype=np.complex64), lattice
        )
        * slm.LF[slm.ComplexPhase, np.int64, 1](
            np.asarray([2], dtype=np.int64), lattice
        )
    )
    assert complex_product.dtype == np.dtype(np.complex64)
    assert complex_product.data.tolist() == [np.complex64(2 + 4j)]


def test_mixed_unsigned_cost_geometry_uses_wrapping_julia_arithmetic() -> None:
    source = (
        slm.LatticeAxis(
            np.asarray([0, 1], dtype=np.uint64), step_hint=np.uint64(1)
        ),
    )
    target = (
        slm.LatticeAxis(
            np.asarray([2, 3], dtype=np.int64), step_hint=np.int64(1)
        ),
    )
    result = slm.getCostMatrix(
        source, target, normalization=lambda _matrix: 1
    )
    np.testing.assert_array_equal(result, [[4.0, 9.0], [1.0, 4.0]])


def test_assignment_preserves_concrete_object_element_type() -> None:
    rational = slm.LF[slm.Generic](
        np.asarray([Fraction(1, 3)], dtype=object), (range(1),)
    )
    rational[0] = Decimal("0.1")
    assert rational.data.tolist() == [Fraction(1, 10)]
    assert type(rational.data[0]) is Fraction

    bigfloat = slm.LF[slm.Generic](
        np.asarray([Decimal("0.2")], dtype=object), (range(1),)
    )
    bigfloat[0] = Fraction(1, 10)
    assert bigfloat.data.tolist() == [Decimal("0.1")]
    assert type(bigfloat.data[0]) is Decimal


@pytest.mark.parametrize(
    "value",
    (
        Fraction(2**63, 1),
        Fraction(1, 2**63),
        Decimal("1e100"),
    ),
)
def test_fraction_destination_enforces_rational_int64_parameter(
    value: object,
) -> None:
    rational = slm.LF[slm.Generic, object, 1](
        np.asarray([Fraction(1, 3)], dtype=object), (range(1),)
    )
    with pytest.raises(ValueError, match=r"Rational\{Int64\}"):
        rational[0] = value
    assert rational.data.tolist() == [Fraction(1, 3)]


def test_exact_values_round_directly_to_low_precision_machine_floats() -> None:
    # Both values are just above a target-format midpoint but too close for a
    # binary64 intermediate to retain that fact. Julia rounds upward directly;
    # the old float(value)->Float16/32 path double-rounded downward.
    probes = (
        (np.float16, np.uint16),
        (np.float32, np.uint32),
    )
    for dtype, bits_dtype in probes:
        lower = Fraction(1)
        upper = Fraction(float(np.nextafter(dtype(1), dtype(np.inf))))
        just_above_midpoint = (
            (lower + upper) / 2 + Fraction(1, 2**100)
        )
        field = slm.LF[slm.Generic](
            np.zeros(1, dtype=dtype), (range(1),)
        )
        field[0] = just_above_midpoint
        expected = np.nextafter(dtype(1), dtype(np.inf))
        assert field.data[0].view(bits_dtype) == expected.view(bits_dtype)


def test_assignment_rejects_undefined_string_and_array_conversions() -> None:
    floating = slm.LF[slm.Generic](
        np.zeros(1, dtype=np.float64), (range(1),)
    )
    integer = slm.LF[slm.Generic](
        np.zeros(1, dtype=np.int64), (range(1),)
    )
    string = slm.LF[slm.Generic](
        np.asarray(["x"], dtype="U8"), (range(1),)
    )
    with pytest.raises(ValueError, match="strings are not numeric"):
        floating[0] = "1.0"
    with pytest.raises(ValueError, match="strings are not numeric"):
        integer[0] = "1"
    with pytest.raises(ValueError, match="numeric-to-string"):
        string[0] = 1.0
    with pytest.raises(TypeError, match="array right-hand"):
        floating[0] = np.asarray(1.0)


def test_assignment_requires_platform_int_and_getindex_rejects_vectors() -> None:
    field = slm.LF[slm.Generic](
        np.arange(4, dtype=np.int64).reshape((2, 2), order="F"),
        (range(2), range(2)),
    )
    original = field.data.copy()
    for key in (
        np.int16(0),
        np.int32(0),
        np.uint64(0),
        (np.int16(0), np.int16(0)),
        (np.int32(0), np.int32(0)),
    ):
        with pytest.raises(TypeError):
            field[key] = 9
    np.testing.assert_array_equal(field.data, original)

    for selector in ([0, 1], np.asarray([0, 1], dtype=np.int64)):
        with pytest.raises(TypeError, match="no vector-index method"):
            _ = field[selector, :]


def test_clip_and_centroid_use_julia_strong_scalar_comparisons() -> None:
    boundary = np.float32(0.1)
    # Float32(0.1) is greater than the distinct Float64 literal 0.1 in Julia.
    # NumPy weak-scalar comparison instead narrows the literal and sees them
    # as equal.
    assert slm.clip(boundary, 0.1) == boundary
    np.testing.assert_array_equal(
        slm.clip(np.asarray([boundary], dtype=np.float32), 0.1),
        [boundary],
    )

    field = slm.LF[slm.Intensity](
        np.asarray([boundary, 1], dtype=np.float32), (range(2),)
    )
    field_center = slm.centroid(field)
    assert field_center.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        field_center, np.asarray([np.float32(0.9090909)])
    )

    raw_center = slm.centroid(
        np.asarray([boundary, 1], dtype=np.float32)
    )
    assert raw_center.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        raw_center, np.asarray([np.float32(1.9090908)])
    )


def test_centroid_constructs_cross_dimension_vector_like_julia() -> None:
    lattice = (
        slm.LatticeAxis.from_start_step(
            np.float32(0), np.float32(1), 2
        ),
        slm.LatticeAxis(
            np.asarray([1], dtype=np.int64), step_hint=np.int64(1)
        ),
    )
    field = slm.LF[slm.Intensity](
        np.asarray([[1], [2]], dtype=np.int64), lattice
    )
    result = slm.centroid(field)
    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        result, np.asarray([np.float32(2 / 3), np.float32(1)])
    )


def test_schroff_error_uses_julia_cutoff_and_mixed_arithmetic() -> None:
    lattice = (range(2),)
    target = slm.LF[slm.Intensity](
        np.asarray([0.1, 1], dtype=np.float32), lattice
    )
    reality = slm.LF[slm.Intensity](
        np.asarray([0.2, 0.8], dtype=np.float32), lattice
    )
    threshold_result = slm.SchroffError(target, reality, 0.1)
    assert type(threshold_result) is np.float32
    assert threshold_result == np.float32(0.12059246)

    mixed_reality = slm.LF[slm.Intensity](
        np.asarray([1, 0], dtype=np.int64), lattice
    )
    mixed_target = slm.LF[slm.Intensity](
        np.asarray([0.5, 0.5], dtype=np.float32), lattice
    )
    mixed_result = slm.SchroffError(
        mixed_target, mixed_reality, np.float32(0)
    )
    assert type(mixed_result) is np.float32
    assert mixed_result == np.float32(1)


def test_r2_boolean_unary_addition_widens_to_julia_int() -> None:
    axis = slm.LatticeAxis(
        np.asarray([False, True], dtype=np.bool_), step_hint=np.bool_(True)
    )
    result = slm.r2((axis,))
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, [0, 1])

    modulus = slm.LF[slm.Modulus](
        np.asarray([False, True], dtype=np.bool_), (axis,)
    )
    intensity = slm.square(modulus)
    assert intensity.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(intensity.data, [False, True])


def test_ordered_scalar_helpers_reject_complex_values_like_julia() -> None:
    with pytest.raises(TypeError):
        slm.ramp(1 + 2j)
    with pytest.raises(TypeError):
        slm.ramp(np.asarray([1 + 0j]))
    with pytest.raises(TypeError):
        slm.clip(1 + 0j, 0)
    with pytest.raises(TypeError):
        slm.clip(np.asarray([1 + 0j]), 0)
    with pytest.raises(TypeError):
        slm.clip(1, 0 + 0j)
    with pytest.raises(TypeError):
        slm.centroid(np.asarray([1 + 0j]))
    complex_intensity = slm.LF[
        slm.Intensity, np.complex128, 1
    ](
        np.asarray([1 + 0j]),
        (slm.LatticeAxis([0], step_hint=1),),
    )
    with pytest.raises(TypeError):
        slm.centroid(complex_intensity)
    with pytest.raises(TypeError):
        slm.SchroffError(complex_intensity, complex_intensity)


def test_lattice_isapprox_subtracts_in_julia_promoted_dtype() -> None:
    # Julia first converts the Int64 coordinate to Float32, making the
    # promoted difference 34536 and therefore inside the Float32 isapprox
    # tolerance. NumPy's Float64 mixed subtraction sees 34539 and rejects it.
    left = slm.LatticeAxis.from_start_step(
        np.float32(1e8), np.float32(1), 1
    )
    right = slm.LatticeAxis(
        np.asarray([100034539], dtype=np.int64), step_hint=np.int64(1)
    )
    assert slm.elq((left,), (right,)) is None

    left_field = slm.LF[slm.Generic](
        np.zeros(1), (left,), np.float32(1e8)
    )
    right_field = slm.LF[slm.Generic](
        np.zeros(1), (right,), np.int64(100034539)
    )
    assert slm.elq(left_field, right_field) is None


def test_subimage_padmultiple_treats_negative_widths_as_noops() -> None:
    image = np.arange(4).reshape(2, 2)
    np.testing.assert_array_equal(
        subimage_padmultiple(
            image,
            padleft=-3,
            padright=-2,
            padtop=-1,
            padbottom=-4,
            padall=-5,
        ),
        image,
    )
    mixed = subimage_padmultiple(
        image,
        padleft=-3,
        padright=1,
        padtop=2,
        padbottom=-4,
        padall=-5,
        fillval=-1,
    )
    np.testing.assert_array_equal(
        mixed,
        np.pad(image, ((2, 0), (0, 1)), constant_values=-1),
    )


def test_dualate_uses_retained_step_for_singleton_target_axis() -> None:
    source_lattice = (
        slm.LatticeAxis([-1.0, 0.0, 1.0], step_hint=1.0),
        slm.LatticeAxis([-1.0, -0.5, 0.0, 0.5, 1.0], step_hint=0.5),
    )
    xx, yy = np.meshgrid(*source_lattice, indexing="ij")
    source = slm.LF[slm.Generic](xx + 2 * yy, source_lattice)
    target_source_lattice = (
        slm.LatticeAxis([0.0], step_hint=0.5),
        slm.LatticeAxis([-0.5, 0.5], step_hint=1.0),
    )

    result = slm.dualate(
        source,
        target_source_lattice,
        [0.0, 0.0],
        0.0,
    )

    assert result.shape == (1, 2)
    np.testing.assert_allclose(result.data, [[-1.0, 0.0]], atol=1e-14)
    assert _step(result.L[0]) == pytest.approx(2.0)


def test_dualate_uses_julia_strong_coordinate_promotion() -> None:
    source_lattice = (
        slm.LatticeAxis(
            np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
            step_hint=np.float32(1),
        ),
        slm.LatticeAxis(
            np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
            step_hint=np.float32(1),
        ),
    )
    xx, yy = np.meshgrid(*source_lattice, indexing="ij")
    source = slm.LF[slm.Generic](
        (xx + 2 * yy).astype(np.float32), source_lattice
    )
    target_lattice = (
        slm.LatticeAxis(
            np.asarray([-0.5, 0.5], dtype=np.float32),
            step_hint=np.float32(1),
        ),
        slm.LatticeAxis(
            np.asarray([-0.5, 0.5], dtype=np.float32),
            step_hint=np.float32(1),
        ),
    )

    narrow = slm.dualate(
        source,
        target_lattice,
        [np.float32(0), np.float32(0)],
        np.float32(0),
        np.float32(1),
    )
    assert narrow.dtype == np.dtype(np.float32)

    # A Python float is Julia Float64, so subtracting this center promotes the
    # source interpolation ranges and the evaluated values.
    wide = slm.dualate(
        source,
        target_lattice,
        [0.0, 0.0],
        np.float32(0),
        np.float32(1),
    )
    assert wide.dtype == np.dtype(np.float64)


def test_dualate_preserves_pathological_range_reference_for_interpolation() -> None:
    start = np.asarray([0x348C], dtype=np.uint16).view(np.float16)[0]
    step = np.asarray([0x20C6], dtype=np.uint16).view(np.float16)[0]
    axis = slm.LatticeAxis.from_start_step(start, step, 25)
    source = slm.LF[slm.Generic](
        np.zeros((25, 25), dtype=np.float16),
        (axis, axis),
    )
    target = (
        slm.LatticeAxis.from_start_step(np.float16(-1), np.float16(1), 2),
        slm.LatticeAxis.from_start_step(np.float16(-1), np.float16(1), 2),
    )
    captured: dict[str, object] = {}

    def capture_interpolator(
        ranges: object,
        _values: object,
        *,
        extrapolation_bc: object = 0,
    ) -> object:
        del extrapolation_bc
        captured["ranges"] = ranges
        return lambda x, _y: np.zeros_like(x)

    slm.dualate(
        source,
        target,
        [np.float16(0), np.float16(0)],
        np.float16(0),
        np.float16(1),
        interpolation=capture_interpolator,
    )
    shifted = captured["ranges"]
    assert isinstance(shifted, tuple)
    assert shifted[0]._logical_ref == 0.28438228438228436
    assert shifted[0]._logical_step == 0.009324009324009324
    assert shifted[0]._logical_offset == 0


def test_dualate_promotes_heterogeneous_axis_queries_without_downcast() -> None:
    source_axis = slm.LatticeAxis(
        np.asarray([-1, 0, 1], dtype=np.float32),
        step_hint=np.float32(1),
    )
    source = slm.LF[slm.Generic](
        np.zeros((3, 3), dtype=np.float32),
        (source_axis, source_axis),
    )
    wanted_dual_step = np.float64(100000003.25)
    step1 = np.float64(1) / (np.float64(3) * wanted_dual_step)
    target = (
        slm.LatticeAxis(
            np.asarray([-0.5, 0.5], dtype=np.float32),
            step_hint=np.float32(1),
        ),
        slm.LatticeAxis(
            np.asarray([-step1, 0, step1], dtype=np.float64),
            step_hint=step1,
        ),
    )

    def x_coordinate_interpolator(
        _ranges: object, _values: object, *, extrapolation_bc: object = 0
    ) -> object:
        del extrapolation_bc
        return lambda x, _y: x

    result = slm.dualate(
        source,
        target,
        [np.float32(0), np.float32(0)],
        np.float32(0.7),
        np.float32(1),
        interpolation=x_coordinate_interpolator,
    )
    expected = np.asarray(
        [
            [
                6.442176870766999e7,
                -0.3824211061000824,
                -6.442176947251220e7,
            ],
            [6.4421769090091094e7, 0.0, -6.4421769090091094e7],
        ]
    )
    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(result.data, expected, rtol=0, atol=1e-8)
    assert result.L[0].dtype == np.dtype(np.float32)
    assert result.L[1].dtype == np.dtype(np.float64)


def test_dualate_scalar_interpolator_collects_before_dtype_selection() -> None:
    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic](np.zeros((2, 2)), lattice)

    def scalar_factory(
        _ranges: object,
        _values: object,
        *,
        extrapolation_bc: object = 0,
    ) -> object:
        del extrapolation_bc
        return lambda x, _y: 0 if x < 0 else 0.5

    result = slm.dualate(
        source,
        lattice,
        [0.0, 0.0],
        0.0,
        interpolation=scalar_factory,
    )
    np.testing.assert_array_equal(result.data, [[0.0, 0.0], [0.5, 0.5]])


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (
            Fraction(1, 3),
            [
                [-0.30888112475929275, -0.47247847315736885],
                [0.1635973483980761, 0.0],
            ],
        ),
        (
            np.int8(1),
            [
                [0.15058433946987837, -0.2701511529340699],
                [0.42073549240394825, 0.0],
            ],
        ),
    ],
)
def test_dualate_uses_julia_integer_rational_trig_promotion(
    theta: object, expected: list[list[float]]
) -> None:
    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic, object, 2](
        np.full((2, 2), Fraction(1, 3), dtype=object),
        lattice,
        Fraction(1),
    )

    def x_coordinate_factory(
        _ranges: object,
        _values: object,
        *,
        extrapolation_bc: object = 0,
    ) -> object:
        del extrapolation_bc
        return lambda x, _y: x

    result = slm.dualate(
        source,
        lattice,
        [Fraction(0), Fraction(0)],
        theta,
        Fraction(1),
        interpolation=x_coordinate_factory,
    )

    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(result.data, expected)


def test_dualate_default_boundary_uses_concrete_object_zero() -> None:
    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic, object, 2](
        np.full((2, 2), Fraction(1, 3), dtype=object),
        lattice,
        Fraction(1),
    )
    captured: dict[str, object] = {}

    def boundary_factory(
        _ranges: object,
        _values: object,
        *,
        extrapolation_bc: object = 99,
    ) -> object:
        captured["boundary"] = extrapolation_bc

        def fill_boundary(x: object, _y: object) -> np.ndarray:
            output = np.empty(np.shape(x), dtype=object)
            output.fill(extrapolation_bc)
            return output

        return fill_boundary

    result = slm.dualate(
        source,
        lattice,
        [Fraction(0), Fraction(0)],
        Fraction(0),
        Fraction(1),
        interpolation=boundary_factory,
    )

    assert type(captured["boundary"]) is Fraction
    assert captured["boundary"] == Fraction(0)
    assert result.dtype == np.dtype(object)
    assert all(type(value) is Fraction for value in result.data.flat)
    np.testing.assert_array_equal(
        result.data,
        np.full((2, 2), Fraction(0), dtype=object),
    )


def test_dualate_wrong_shape_vectorized_result_falls_back_pointwise() -> None:
    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic](np.zeros((2, 2)), lattice)

    def constant_scalar_factory(
        _ranges: object,
        _values: object,
        *,
        extrapolation_bc: object = 0,
    ) -> object:
        del extrapolation_bc
        return lambda _x, _y: 0.25

    result = slm.dualate(
        source,
        lattice,
        [0.0, 0.0],
        0.0,
        interpolation=constant_scalar_factory,
    )

    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.data, np.full((2, 2), 0.25))


def test_float32_template_coordinates_remain_float32() -> None:
    lattice = (
        slm.LatticeAxis(
            np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
            step_hint=np.float32(0.5),
        ),
    )
    center = (np.float32(0),)
    parabola = slm.lfParabola(
        slm.Generic,
        lattice,
        np.float32(2),
        lin=(np.float32(0),),
        center=center,
    )
    gaussian = slm.lfGaussian(
        slm.Generic,
        lattice,
        np.float32(1),
        np.float32(1),
        center=center,
    )
    gaussian_matrix = slm.lfGaussian(
        slm.Generic,
        lattice,
        np.asarray([[1]], dtype=np.int64),
        np.float32(1),
        center=center,
    )
    assert parabola.dtype == np.dtype(np.float32)
    assert gaussian.dtype == np.dtype(np.float32)
    assert gaussian_matrix.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(gaussian_matrix.data, gaussian.data)


def test_template_scalar_promotion_and_complex_wrapping_match_julia() -> None:
    lattice = (
        slm.LatticeAxis(
            np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
            step_hint=np.float32(0.5),
        ),
    )
    center32 = (np.float32(0),)
    lin32 = (np.float32(0),)

    # Julia's default center and Python floats are Float64 and therefore
    # widen Float32 coordinates. Ordinary integers do not.
    default_center = slm.lfParabola(
        slm.Generic, lattice, np.float32(2), lin=lin32
    )
    integer_linear = slm.lfParabola(
        slm.Generic, lattice, np.float32(2), lin=(0,), center=center32
    )
    integer_matrix = slm.lfParabola(
        slm.Generic,
        lattice,
        np.asarray([[2]], dtype=np.int64),
        lin=lin32,
        center=center32,
    )
    assert default_center.dtype == np.dtype(np.float64)
    assert integer_linear.dtype == np.dtype(np.float32)
    assert integer_matrix.dtype == np.dtype(np.float32)

    ring32 = slm.lfRing(
        slm.Generic,
        lattice,
        np.float32(0.5),
        np.float32(0.2),
        center=center32,
    )
    ring64 = slm.lfRing(
        slm.Generic, lattice, 0.5, 0.2, center=center32
    )
    cap32 = slm.lfCap(
        slm.Generic,
        lattice,
        np.float32(1),
        np.float32(2),
        center=center32,
    )
    cap64 = slm.lfCap(slm.Generic, lattice, 1.0, 2.0, center=center32)
    assert ring32.dtype == np.dtype(np.float32)
    assert ring64.dtype == np.dtype(np.float64)
    assert cap32.dtype == np.dtype(np.float32)
    assert cap64.dtype == np.dtype(np.float64)

    wrapped = slm.lfParabola(
        slm.ComplexPhase,
        lattice,
        np.float32(2),
        lin=lin32,
        center=center32,
    )
    assert wrapped.dtype == np.dtype(np.complex128)


def test_template_rational_and_bigfloat_counterparts_match_julia() -> None:
    rational_axis = slm.LatticeAxis(
        np.asarray(
            [Fraction(-1, 5), Fraction(0), Fraction(1, 5)], dtype=object
        ),
        step_hint=Fraction(1, 5),
    )
    rational_lattice = (rational_axis,)
    rational_outputs = (
        slm.lfParabola(slm.Generic, rational_lattice, Fraction(2, 5)),
        slm.lfGaussian(slm.Generic, rational_lattice, Fraction(2, 5)),
        slm.lfRing(
            slm.Generic, rational_lattice, Fraction(2, 5), Fraction(1, 10)
        ),
        slm.lfCap(
            slm.Generic, rational_lattice, Fraction(2, 5), Fraction(4, 5)
        ),
        slm.lfRect(slm.Generic, rational_lattice, (Fraction(3, 10),)),
    )
    expected_rational = (
        [0.008000000000000002, 0.0, 0.008000000000000002],
        [1.2339050660814306, 1.3981976168614934, 1.2339050660814306],
        [0.13533528323661262, 0.00033546262790251126, 0.13533528323661262],
        [0.792, 0.8, 0.792],
        [0.0, 1.0, 0.0],
    )
    for result, expected in zip(
        rational_outputs, expected_rational, strict=True
    ):
        assert result.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(result.data, expected)

    exact_parabola = slm.lfParabola(
        slm.Generic,
        rational_lattice,
        Fraction(2, 5),
        lin=(Fraction(0),),
        center=(Fraction(0),),
    )
    exact_cap = slm.lfCap(
        slm.Generic,
        rational_lattice,
        Fraction(2, 5),
        Fraction(4, 5),
        center=(Fraction(0),),
    )
    assert exact_parabola.dtype == np.dtype(object)
    assert exact_cap.dtype == np.dtype(object)
    np.testing.assert_array_equal(
        exact_parabola.data,
        [Fraction(1, 125), Fraction(0), Fraction(1, 125)],
    )
    np.testing.assert_array_equal(
        exact_cap.data,
        [Fraction(99, 125), Fraction(4, 5), Fraction(99, 125)],
    )

    # These values mirror `BigFloat(-.2):BigFloat(.2):BigFloat(.2)` and
    # BigFloat parameters constructed from Julia Float64 literals.  Decimal's
    # context is the Python arbitrary-precision analogue; at 80 digits it
    # agrees with Julia's default 256-bit results through their precision.
    with localcontext() as context:
        context.prec = 80
        decimal_axis = slm.LatticeAxis(
            np.asarray(
                [
                    Decimal.from_float(-0.2),
                    Decimal(0),
                    Decimal.from_float(0.2),
                ],
                dtype=object,
            ),
            step_hint=Decimal.from_float(0.2),
        )
        decimal_lattice = (decimal_axis,)
        decimal_outputs = (
            slm.lfParabola(
                slm.Generic, decimal_lattice, Decimal.from_float(0.4)
            ),
            slm.lfGaussian(
                slm.Generic, decimal_lattice, Decimal.from_float(0.4)
            ),
            slm.lfRing(
                slm.Generic,
                decimal_lattice,
                Decimal.from_float(0.4),
                Decimal.from_float(0.1),
            ),
            slm.lfCap(
                slm.Generic,
                decimal_lattice,
                Decimal.from_float(0.4),
                Decimal.from_float(0.8),
            ),
        )
        expected_centers = (
            Decimal(0),
            Decimal(
                "1.398197616861493410012156662380817913135768483096105167578079194807872917943948"
            ),
            Decimal(
                "0.0003354626279025118388213891257808610193109001337203193605445757479116405207086566"
            ),
            Decimal(
                "0.8000000000000000444089209850062616169452667236328125"
            ),
        )
        tolerance = Decimal("2e-76")
        for result, expected in zip(
            decimal_outputs, expected_centers, strict=True
        ):
            assert result.dtype == np.dtype(object)
            assert all(isinstance(value, Decimal) for value in result.data.flat)
            assert abs(result.data[1] - expected) < tolerance

        complex_template = slm.lfParabola(
            slm.ComplexPhase,
            decimal_lattice,
            Decimal.from_float(0.4),
        )
        assert complex_template.dtype == np.dtype(object)
        assert complex_template.data[1].real == Decimal(1)
        assert complex_template.data[1].imag == Decimal(0)
        assert all(
            isinstance(value.real, Decimal)
            and isinstance(value.imag, Decimal)
            for value in complex_template.data.flat
        )

        decimal_rect = slm.lfRect(
            slm.Generic,
            decimal_lattice,
            (Decimal.from_float(0.3),),
        )
        assert decimal_rect.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(decimal_rect.data, [0.0, 1.0, 0.0])

        decimal_matrix = np.asarray(
            [[Decimal.from_float(0.4)]], dtype=object
        )
        matrix_parabola = slm.lfParabola(
            slm.Generic, decimal_lattice, decimal_matrix
        )
        matrix_gaussian = slm.lfGaussian(
            slm.Generic,
            decimal_lattice,
            np.asarray([[Decimal("6.25")]], dtype=object),
        )
        assert matrix_parabola.dtype == np.dtype(object)
        assert matrix_gaussian.dtype == np.dtype(object)
        assert all(
            isinstance(value, Decimal) for value in matrix_parabola.data.flat
        )
        assert all(
            isinstance(value, Decimal) for value in matrix_gaussian.data.flat
        )


def test_decimal_random_template_and_rational_sampler_boundary() -> None:
    lattice = (range(4),)
    with localcontext() as context:
        context.prec = 50
        np.random.seed(1234)
        result = slm.lfRand(slm.Generic, lattice, R=Decimal)
        assert result.dtype == np.dtype(object)
        assert all(isinstance(value, Decimal) for value in result.data)
        assert all(Decimal(0) <= value < Decimal(1) for value in result.data)

    with pytest.raises(TypeError, match="no Rational sampler"):
        slm.lfRand(slm.Generic, lattice, R=Fraction)


def test_intensity_addition_is_an_eager_value_semantic_left_fold() -> None:
    lattice = (range(1),)
    a = slm.LF[slm.Intensity, np.float64, 1](
        np.asarray([-10.0]), lattice
    )
    b = slm.LF[slm.Intensity, np.float64, 1](np.asarray([5.0]), lattice)
    c = slm.LF[slm.Intensity, np.float64, 1](np.asarray([6.0]), lattice)
    inputs_before = tuple(field.data.copy() for field in (a, b, c))

    # Julia's explicitly parenthesized binary expression `(a + b) + c`
    # clips the first binary result to zero, then adds six. Python's chained
    # syntax is the same left fold. It must not recover the hidden raw sum
    # (-5) from the first operation.
    intermediate = a + b
    assert isinstance(intermediate, slm.LatticeField)
    np.testing.assert_array_equal(intermediate.data, [0.0])
    np.testing.assert_array_equal((a + b + c).data, [6.0])
    np.testing.assert_array_equal(((a + b) + c).data, [6.0])

    # A field reached through arithmetic is interchangeable with a fresh
    # field or copy having the same visible value. This specifically guards
    # against expression-history accumulators surviving behind `.data`.
    fresh = slm.LF[slm.Intensity, np.float64, 1](
        intermediate.data.copy(), intermediate.L, intermediate.flambda
    )
    copied = intermediate.copy()
    np.testing.assert_array_equal((intermediate + c).data, [6.0])
    np.testing.assert_array_equal((fresh + c).data, [6.0])
    np.testing.assert_array_equal((copied + c).data, [6.0])

    assert not np.shares_memory(intermediate.data, a.data)
    assert not np.shares_memory(intermediate.data, b.data)
    for field, before in zip((a, b, c), inputs_before, strict=True):
        np.testing.assert_array_equal(field.data, before)

    # The result owns its storage: mutating it cannot retroactively mutate an
    # operand or alter the arithmetic history of another equal-valued field.
    intermediate.data[0] = 3.0
    np.testing.assert_array_equal(a.data, [-10.0])
    np.testing.assert_array_equal(b.data, [5.0])
    np.testing.assert_array_equal(fresh.data, [0.0])


def test_field_add_and_multiply_chains_match_julia_binary_parenthesization() -> None:
    lattice = (range(2),)
    phases = (
        slm.LF[slm.RealPhase](np.asarray([0.25, -0.5]), lattice),
        slm.LF[slm.RealPhase](np.asarray([0.5, 0.25]), lattice),
        slm.LF[slm.RealPhase](np.asarray([-0.125, 0.5]), lattice),
    )
    phase_inputs = tuple(field.data.copy() for field in phases)

    # Julia `(p1 + p2) + p3` invokes the binary RealPhase method twice and
    # yields these values. An unparenthesized Python chain has exactly that
    # left-associated meaning.
    phase_chain = phases[0] + phases[1] + phases[2]
    phase_explicit = (phases[0] + phases[1]) + phases[2]
    np.testing.assert_array_equal(phase_chain.data, [0.625, 0.25])
    np.testing.assert_array_equal(phase_chain.data, phase_explicit.data)

    modulus = slm.LF[slm.Modulus](np.asarray([2.0, 3.0]), lattice)
    real_phase = slm.LF[slm.RealPhase](
        np.asarray([0.25, -0.25]), lattice
    )
    complex_phase = slm.LF[slm.ComplexPhase](
        np.asarray([1.0j, -1.0j]), lattice
    )
    product_inputs = tuple(
        field.data.copy() for field in (modulus, real_phase, complex_phase)
    )

    # Julia `(modulus * real_phase) * complex_phase` first constructs a
    # ComplexAmplitude, then applies the second binary field product.
    product_chain = modulus * real_phase * complex_phase
    product_explicit = (modulus * real_phase) * complex_phase
    np.testing.assert_allclose(
        product_chain.data,
        np.asarray([-2.0 + 0.0j, -3.0 + 0.0j]),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(product_chain.data, product_explicit.data)
    assert product_chain.field_type is slm.ComplexAmplitude
    assert not np.shares_memory(product_chain.data, modulus.data)
    assert not np.shares_memory(product_chain.data, real_phase.data)
    assert not np.shares_memory(product_chain.data, complex_phase.data)

    for field, before in zip(phases, phase_inputs, strict=True):
        np.testing.assert_array_equal(field.data, before)
    for field, before in zip(
        (modulus, real_phase, complex_phase), product_inputs, strict=True
    ):
        np.testing.assert_array_equal(field.data, before)


def test_no_julia_vararg_arithmetic_helpers_are_exposed() -> None:
    assert not hasattr(slm, "julia_add")
    assert not hasattr(slm, "julia_mul")
    assert "julia_add" not in slm.__all__
    assert "julia_mul" not in slm.__all__


def test_centroid_and_schroff_error_match_julia_dispatch_boundaries() -> None:
    lattice = slm.natlat(3, 3)
    intensity = slm.LF[slm.Intensity](np.eye(3), lattice)
    modulus = slm.LF[slm.Modulus](np.eye(3), lattice)

    np.testing.assert_allclose(slm.centroid(intensity), [0.0, 0.0])
    with pytest.raises(TypeError):
        slm.centroid(modulus)
    with pytest.raises(TypeError):
        slm.centroid([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(TypeError):
        slm.centroid(intensity, threshold=0.1 + 0.2j)
    with pytest.raises(TypeError):
        slm.centroid(intensity, threshold=np.asarray(0.1))
    with pytest.raises(TypeError):
        slm.SchroffError(intensity.data, intensity.data)
    with pytest.raises(TypeError):
        slm.SchroffError(intensity, modulus)

    scalar_array = np.asarray(2.0)
    scalar_field = slm.LF[slm.Intensity](scalar_array, ())
    assert slm.centroid(scalar_array).shape == (0,)
    assert slm.centroid(scalar_field).shape == (0,)


def test_schroff_error_preserves_float32_and_integer_inexactness() -> None:
    lattice = (range(2),)
    left = slm.LF[slm.Intensity](np.array([1.0, 2.0], dtype=np.float32), lattice)
    right = slm.LF[slm.Intensity](np.array([1.0, 2.0], dtype=np.float32), lattice)
    result = slm.SchroffError(left, right, threshold=0.0)
    assert isinstance(result, np.float32)
    assert result == np.float32(0)

    integer = slm.LF[slm.Intensity](np.array([1, 1], dtype=np.int64), lattice)
    with pytest.raises(ValueError, match="Inexact normalization"):
        slm.SchroffError(integer, integer, threshold=0.0)
    with pytest.raises(TypeError):
        slm.SchroffError(left, right, threshold=np.asarray(0.5))

    lattice_2d = slm.natlat((2, 2))
    target = slm.LF[slm.Intensity](
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        lattice_2d,
    )
    reality = slm.LF[slm.Intensity](
        np.asarray([[1.1, 1.9], [2.8, 4.2]], dtype=np.float32),
        lattice_2d,
    )
    zero_threshold = slm.SchroffError(target, reality, threshold=0.0)
    half_threshold = slm.SchroffError(target, reality, threshold=0.5)
    assert np.asarray(zero_threshold).view(np.uint32) == np.uint32(0x3D3C889E)
    assert np.asarray(half_threshold).view(np.uint32) == np.uint32(0x3D5F1F82)


def test_field_ldq_rejects_the_lattice_only_flambda_argument() -> None:
    lattice = slm.natlat(2, 3)
    dual = slm.dualShiftLattice(lattice, 2.0)
    left = slm.LF[slm.Intensity](np.ones((2, 3)), lattice, 2.0)
    right = slm.LF[slm.Intensity](np.ones((2, 3)), dual, 2.0)

    assert slm.ldq(left, right) is None
    with pytest.raises(TypeError):
        slm.ldq(left, right, 2.0)
    with pytest.raises(TypeError):
        slm.ldq(left, right, flambda=2.0)


def test_field_template_inherits_but_cannot_override_flambda() -> None:
    pattern = slm.LF[slm.Intensity](np.ones((3, 3)), slm.natlat(3, 3), 2.5)
    result = slm.lfGaussian(pattern, 1.0)
    assert result.flambda == 2.5
    with pytest.raises(TypeError):
        slm.lfGaussian(pattern, 1.0, flambda=2.5)


def test_save_beam_retains_unusable_upstream_output_paths(tmp_path) -> None:
    beam = np.asarray([[1 + 2j, -3 - 4j], [1j, 2 + 0j]], dtype=np.complex128)
    with pytest.raises(NotImplementedError, match="audited Julia"):
        slm.saveBeam(beam, "probe", data=("beamCsv",), dir=tmp_path)
    # Unknown selectors take none of Julia's broken branches and remain a
    # no-op, as does an empty selector collection.
    assert slm.saveBeam(beam, "probe", data=("unknown",), dir=tmp_path) is None
    assert slm.saveBeam(beam, "probe", data=(), dir=tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_save_beam_keeps_julia_complex128_matrix_dispatch(tmp_path) -> None:
    with pytest.raises(TypeError):
        slm.saveBeam(np.ones((2, 2), dtype=np.complex64), "bad", dir=tmp_path)
    with pytest.raises(TypeError):
        slm.saveBeam([[1 + 0j]], "bad", dir=tmp_path)
    with pytest.raises(TypeError):
        slm.saveBeam(np.ones((1, 1), dtype=np.complex128), tmp_path / "bad", dir=tmp_path)
