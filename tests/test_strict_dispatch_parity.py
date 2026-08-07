from decimal import Decimal
from fractions import Fraction
from numbers import Complex, Number, Real

import gmpy2
import numpy as np
import pytest

import slmtools as slm
from slmtools.lattice_field import LatticeAxis
from slmtools.lattice_utils import _step


def test_bigint_scalar_nabs_uses_julia_bigfloat_precision() -> None:
    for value in (gmpy2.mpz(2), gmpy2.mpz(10) ** 30):
        result = slm.nabs(value)
        assert isinstance(result, gmpy2.mpfr)
        assert result.precision == 256
        assert result == 1


def test_full_field_constructor_accepts_abstract_numeric_element_types() -> None:
    lattice = (range(2),)
    number_values = np.asarray([1 + 1j, 2.0], dtype=object)
    number_field = slm.LF[slm.Generic, Number, 1](
        number_values, lattice
    )
    assert number_field._logical_object_type is Number

    real_values = np.asarray([Decimal("1.5"), gmpy2.mpfr(2)], dtype=object)
    real_field = slm.LF[slm.Generic, Real, 1](real_values, lattice)
    assert real_field._logical_object_type is Real

    complex_values = np.asarray([1 + 1j, gmpy2.mpc(2, -1)], dtype=object)
    complex_field = slm.LF[slm.ComplexPhase, Complex, 1](
        complex_values, lattice
    )
    assert complex_field._logical_object_type is Complex

    with pytest.raises(TypeError, match="exact object elements"):
        slm.LF[slm.Generic, Real, 1](number_values, lattice)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (slm.lfParabola, (np.asarray(1.0),)),
        (slm.lfGaussian, (np.asarray(1.0),)),
        (slm.lfRing, (np.asarray(1.0), 0.5)),
        (slm.lfRing, (1.0, np.asarray(0.5))),
    ],
)
def test_template_scalar_overloads_reject_zero_dimensional_arrays(
    function, arguments: tuple[object, ...]
) -> None:
    with pytest.raises(TypeError):
        function(slm.Generic, (range(3),), *arguments)


@pytest.mark.parametrize("function", [slm.lfParabola, slm.lfGaussian])
def test_template_matrix_overloads_require_dense_matrices(function) -> None:
    lattice = (range(2), range(2))
    with pytest.raises(TypeError, match="dense Matrix"):
        function(slm.Generic, lattice, ((1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(TypeError, match="contiguous"):
        function(slm.Generic, lattice, np.eye(4)[::2, ::2])
    result = function(
        slm.Generic, lattice, [[1.0, 0.0], [0.0, 1.0]]
    )
    assert result.shape == (2, 2)


@pytest.mark.parametrize(
    "height", [10**400, gmpy2.mpz(10) ** 400]
)
def test_lfrect_huge_exact_height_converts_to_float64_infinity(height) -> None:
    result = slm.lfRect(slm.Generic, slm.natlat((1,)), (2.0,), height)
    assert result.dtype == np.dtype(np.float64)
    assert np.isposinf(result.data[0])


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_lfrand_low_precision_never_rounds_up_to_one(dtype) -> None:
    np.random.seed(0)
    result = slm.lfRand(
        slm.Generic, (range(20_000),), R=dtype
    )
    assert result.dtype == np.dtype(dtype)
    assert np.all(result.data >= dtype(0))
    assert np.all(result.data < dtype(1))


def test_nonplatform_range_lengths_preserve_julia_constructor_failure() -> None:
    bigint_axis = LatticeAxis(
        np.asarray([gmpy2.mpz(i) for i in range(4)], dtype=object),
        step_hint=gmpy2.mpz(1),
    )
    rational_bigint_axis = LatticeAxis(
        np.asarray(
            [gmpy2.mpq(i, 3) for i in range(4)], dtype=object
        ),
        step_hint=gmpy2.mpq(1, 3),
    )
    uint64_axis = LatticeAxis(np.arange(4, dtype=np.uint64))
    for axis in (bigint_axis, rational_bigint_axis, uint64_axis):
        with pytest.raises(slm.DimensionMismatch):
            slm.LF[slm.Generic](np.zeros(4), (axis,))

    downsampled = slm.downsample((bigint_axis,), 2)[0]
    assert isinstance(_step(downsampled), gmpy2.mpfr)
    assert _step(downsampled).precision == 256
    with pytest.raises(slm.DimensionMismatch):
        slm.LF[slm.Generic](np.zeros(2), (downsampled,))

    # ``range(start=UInt64, step=UInt64, length=Int64)`` has platform-Int
    # length and therefore remains a successful Julia constructor path.
    explicit_length = LatticeAxis.from_start_step(
        np.uint64(0), np.uint64(1), 4
    )
    field = slm.LF[slm.Generic](np.zeros(4), (explicit_length,))
    assert field.shape == (4,)


def test_bool_and_complex_ranges_are_valid_lattice_domains() -> None:
    bool_axis = LatticeAxis(np.asarray([False, True]))
    bool_field = slm.LF[slm.Generic](np.zeros(2), (bool_axis,))
    assert bool_field.shape == (2,)
    assert _step(bool_field.L[0]) == np.bool_(True)
    np.testing.assert_array_equal(slm.r2((bool_axis,)), [0, 1])
    assert slm.Nyquist((bool_axis,)) == (np.float64(0.5),)

    complex_axis = LatticeAxis(
        np.arange(4, dtype=np.complex64),
        step_hint=np.complex64(1),
    )
    complex_field = slm.LF[slm.Generic](np.zeros(4), (complex_axis,))
    assert complex_field.shape == (4,)
    assert np.asarray(slm.dualLattice((complex_axis,))[0]).dtype == np.dtype(
        np.complex64
    )

    with gmpy2.context(gmpy2.get_context(), precision=256):
        mpc_axis = np.asarray(
            [gmpy2.mpc(gmpy2.mpfr(i), 0) for i in range(4)],
            dtype=object,
        )
    mpc_field = slm.LF[slm.Generic](np.zeros(4), (mpc_axis,))
    assert mpc_field.shape == (4,)
    displacement = slm.latticeDisplacement(mpc_field.L)[0]
    assert isinstance(displacement, gmpy2.mpc)
    assert displacement.precision == (256, 256)


def test_complex_cost_default_extrema_fail_but_sum_normalization_succeeds() -> None:
    axis = LatticeAxis(
        np.asarray([0 + 0j, 1 + 1j], dtype=np.complex128),
        step_hint=np.complex128(1 + 1j),
    )
    lattice = (axis,)
    with pytest.raises(TypeError, match="cannot order complex"):
        slm.getCostMatrix(lattice)
    with pytest.raises(TypeError, match="cannot order complex"):
        slm.pdCostMatrix(lattice, lattice, 1.0, 0.0)
    for matrix in (
        slm.getCostMatrix(lattice, normalization=np.sum),
        slm.pdCostMatrix(
            lattice,
            lattice,
            1.0,
            0.0,
            normalization=np.sum,
        ),
    ):
        assert matrix.shape == (2, 2)
        assert matrix.dtype == np.dtype(np.complex128)


def test_complex_template_ordering_failures_are_preserved() -> None:
    axis = LatticeAxis(
        np.asarray([0 + 0j, 1 + 1j], dtype=np.complex128),
        step_hint=np.complex128(1 + 1j),
    )
    lattice = (axis,)
    with pytest.raises(TypeError, match="lfCap cannot order complex"):
        slm.lfCap(slm.Generic, lattice, 1.0, 2.0)
    for function, arguments in (
        (slm.lfParabola, (1.0,)),
        (slm.lfGaussian, (1.0,)),
        (slm.lfRing, (1.0, 0.5)),
    ):
        generic = function(slm.Generic, lattice, *arguments)
        assert generic.dtype == np.dtype(np.complex128)
        with pytest.raises(TypeError, match="cannot order complex"):
            function(slm.Modulus, lattice, *arguments)


@pytest.mark.parametrize("reducer", [np.min, np.amin, np.max, np.amax])
def test_coarsen_extrema_reducers_cannot_order_nonempty_complex_blocks(
    reducer,
) -> None:
    values = np.asarray([1 + 2j, 3 + 4j])
    with pytest.raises(TypeError, match="cannot order complex"):
        slm.coarsen(values, 1, reducer=reducer)
    empty = slm.coarsen(np.empty(0, dtype=np.complex128), 1, reducer=reducer)
    assert empty.shape == (0,)
    assert empty.dtype == np.dtype(np.complex128)


@pytest.mark.parametrize("transform", [slm.sft, slm.isft])
def test_shifted_fft_rejects_bigfloat_like_arrays_and_fields(transform) -> None:
    values = np.asarray([Decimal(1), Decimal(2)], dtype=object)

    with pytest.raises(TypeError, match="type BigFloat not supported"):
        transform(values)

    field = slm.LF[slm.ComplexAmplitude, object, 1](
        values, (range(2),)
    )
    with pytest.raises(TypeError, match="type BigFloat not supported"):
        transform(field)

    with pytest.raises(TypeError, match="not a tuple"):
        transform((1.0, 2.0))


@pytest.mark.parametrize("helper", [slm.pdotPhase, slm.pdotBeamEstimate])
def test_pdot_rejects_unmatched_beta_vector_components(helper) -> None:
    lattice = slm.natlat((3, 3))
    field = slm.LF[slm.Intensity](
        np.arange(1.0, 10.0).reshape((3, 3), order="F"),
        lattice,
    )

    with pytest.raises(
        slm.DimensionMismatch,
        match="betaRoot and betaTarget must have matching lengths",
    ):
        helper(
            field,
            field,
            1.0,
            0.0,
            [0.0, 0.0, 9.0],
            [0.0, 0.0],
            1.0,
        )


def test_unsupported_field_multiplication_precedes_lattice_validation() -> None:
    first_lattice = (range(2),)
    second_lattice = (range(2, 4),)
    left = slm.LF[slm.Intensity](np.ones(2), first_lattice)
    right = slm.LF[slm.Intensity](np.ones(2), second_lattice)

    with pytest.raises(
        TypeError, match="Behavior undefined for this combination of inputs"
    ):
        _ = left * right

    supported_left = slm.LF[slm.Modulus](np.ones(2), first_lattice)
    supported_right = slm.LF[slm.ComplexPhase](
        np.ones(2, dtype=np.complex128), second_lattice
    )
    with pytest.raises(slm.DomainError, match="Unequal lattices or flambdas"):
        _ = supported_left * supported_right


def test_padout_keeps_composite_fillers_as_scalar_cells() -> None:
    tuple_source = np.empty(1, dtype=object)
    tuple_source[0] = (1.0, 2.0)
    tuple_output = slm.padout(tuple_source, 1, (0.0, 0.0))

    assert tuple_output.shape == (3,)
    assert tuple_output.dtype == np.dtype(object)
    assert tuple_output.tolist() == [
        (0.0, 0.0),
        (1.0, 2.0),
        (0.0, 0.0),
    ]

    filler = [0, 0]
    value = [1, 2]
    vector_source = np.empty(1, dtype=object)
    vector_source[0] = value
    vector_output = slm.padout(vector_source, 1, filler)
    assert vector_output[0] is filler
    assert vector_output[1] is value
    assert vector_output[2] is filler


def test_padout_converts_interior_to_mpfr_filler_element_type() -> None:
    real = slm.padout(np.asarray([1, 2], dtype=np.int64), 1, gmpy2.mpfr("0.5"))
    assert all(isinstance(value, gmpy2.mpfr) for value in real)
    assert real.tolist() == [
        gmpy2.mpfr("0.5"),
        gmpy2.mpfr(1),
        gmpy2.mpfr(2),
        gmpy2.mpfr("0.5"),
    ]

    filler = gmpy2.mpc(gmpy2.mpfr("0.5"), gmpy2.mpfr("0.25"))
    complex_values = slm.padout(np.asarray([1, 2]), 1, filler)
    assert all(isinstance(value, gmpy2.mpc) for value in complex_values)
    assert complex_values[1] == gmpy2.mpc(1)
    assert complex_values[2] == gmpy2.mpc(2)


def test_coarsen_boxes_composite_results_in_julia_callback_order() -> None:
    source = np.asarray([[1, 3], [2, 4]], order="F")
    seen: list[int] = []

    def tuple_reducer(block: np.ndarray) -> tuple[int, int]:
        value = int(block[0, 0])
        seen.append(value)
        return value, value + 1

    tuples = slm.coarsen(source, (1, 1), reducer=tuple_reducer)
    assert seen == [1, 2, 3, 4]
    assert tuples.dtype == np.dtype(object)
    assert tuples[0, 0] == (1, 2)
    assert tuples[1, 0] == (2, 3)
    assert tuples[0, 1] == (3, 4)
    assert tuples[1, 1] == (4, 5)

    vectors = slm.coarsen(
        np.asarray([1, 2]),
        1,
        reducer=lambda block: [int(block[0]), int(block[0]) + 1],
    )
    assert vectors.dtype == np.dtype(object)
    assert vectors.tolist() == [[1, 2], [2, 3]]


def test_coarsen_passes_detached_superpixels_to_custom_reducer() -> None:
    source = np.asarray([[1, 3], [2, 4]], order="F")
    original = source.copy(order="F")

    def mutate(block: np.ndarray) -> np.ndarray:
        block[...] = -9
        return block

    result = slm.coarsen(source, (1, 1), reducer=mutate)
    np.testing.assert_array_equal(source, original)
    assert result.dtype == np.dtype(object)
    assert all(
        isinstance(item, np.ndarray) and item.item() == -9
        for item in result.flat
    )


def test_numeric_array_dispatch_rejects_string_storage_before_callbacks() -> None:
    strings = np.asarray(["a", "b"])
    with pytest.raises(TypeError, match="Julia numeric element type"):
        slm.coarsen(strings, 1, reducer=lambda block: block[0])
    with pytest.raises(TypeError, match="Julia numeric element type"):
        slm.collapse(strings.reshape(1, 2), 1)

    called = False

    def factory(*_args, **_kwargs):
        nonlocal called
        called = True
        return np.asarray([9, 8])

    lattice = (range(1, 3),)
    for function in (slm.downsample, slm.upsample):
        with pytest.raises(TypeError, match="Julia numeric element type"):
            function(
                strings,
                lattice,
                lattice,
                interpolation=factory,
            )
    assert not called

    field_values = np.full((2, 2), "x")
    field_lattice = slm.natlat((2, 2))
    field = slm.LF[slm.Generic, field_values.dtype, 2](
        field_values, field_lattice
    )
    with pytest.raises(TypeError, match="Julia numeric element type"):
        slm.dualate(
            field,
            field_lattice,
            [0.0, 0.0],
            0.0,
            interpolation=factory,
            bc=0,
        )
    assert not called


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (slm.ramp, (np.asarray([-1.0, 2.0]),)),
        (slm.clip, (np.asarray([-1.0, 2.0]), 0.0)),
        (slm.safeInverse, (np.asarray([2.0]),)),
    ],
)
def test_scalar_numeric_helpers_reject_array_broadcast_extensions(
    function, arguments: tuple[object, ...]
) -> None:
    with pytest.raises(TypeError):
        function(*arguments)


@pytest.mark.parametrize("function", [slm.downsample, slm.upsample])
def test_field_resampling_factory_receives_mutable_source_storage(function) -> None:
    lattice = (range(1, 3),)
    field = slm.LF[slm.Generic](np.asarray([1.0, 2.0]), lattice)

    def factory(_ranges, values, *, extrapolation_bc):
        assert extrapolation_bc == 0.0
        values[...] = 9.0
        return values

    result = function(field, lattice, interpolation=factory)
    np.testing.assert_array_equal(field.data.copy(), [9.0, 9.0])
    np.testing.assert_array_equal(result.data.copy(), [9.0, 9.0])

@pytest.mark.parametrize(
    "out_type", [str, list, tuple, object, lambda value: ("ok", value)]
)
def test_filename_numeric_parsers_reject_nonnumeric_data_types(out_type) -> None:
    with pytest.raises(TypeError, match="numeric parsing"):
        slm.parseStringToNum("12", outType=out_type)
    with pytest.raises(TypeError, match="numeric parsing"):
        slm.parseFileName("12.bmp", outType=out_type)


def test_filename_parser_retains_bigfloat_numeric_domain() -> None:
    value = slm.parseStringToNum("1,25", outType=gmpy2.mpfr)
    assert isinstance(value, gmpy2.mpfr)
    assert value == gmpy2.mpfr("1.25")
    integer = slm.parseStringToNum(
        "123456789012345678901234567890", outType=gmpy2.mpz
    )
    assert isinstance(integer, gmpy2.mpz)
    assert integer == gmpy2.mpz("123456789012345678901234567890")
    rational = slm.parseStringToNum("1//2", outType=gmpy2.mpq)
    assert isinstance(rational, gmpy2.mpq)
    assert rational == gmpy2.mpq(1, 2)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    ("text", "expected"),
    [("1+2im", 1 + 2j), ("1 - 2im", 1 - 2j), ("2im", 2j)],
)
def test_filename_parsers_accept_julia_complex_literal_spelling(
    dtype, text: str, expected: complex
) -> None:
    assert slm.parseStringToNum(text, outType=dtype) == dtype(expected)
    assert slm.parseFileName(f"{text}.bmp", outType=dtype) == dtype(expected)


def test_filename_parsers_follow_julia_base_prefix_and_separator_grammar() -> None:
    assert slm.parseStringToNum("0x10") == 16
    assert slm.parseStringToNum("0b101") == 5
    assert slm.parseStringToNum("0x10", outType=np.int64) == np.int64(16)
    assert slm.parseStringToNum("0x10", outType=np.float64) == np.float64(16)
    assert slm.parseStringToNum("0x10", outType=np.complex128) == 16 + 0j

    for out_type in (None, np.int64, np.float64, np.complex128):
        options = {} if out_type is None else {"outType": out_type}
        with pytest.raises(ValueError, match="underscores"):
            slm.parseStringToNum("1_000", **options)


def test_filename_parser_default_is_checked_julia_int64() -> None:
    limits = np.iinfo(np.int64)
    assert slm.parseStringToNum(str(limits.min)) == int(limits.min)
    assert slm.parseStringToNum(str(limits.max)) == int(limits.max)
    assert slm.parseStringToNum("12", outType=None) == 12

    for text in (str(int(limits.min) - 1), str(int(limits.max) + 1)):
        with pytest.raises(OverflowError, match="Julia Int64"):
            slm.parseStringToNum(text)
        with pytest.raises(OverflowError, match="Julia Int64"):
            slm.parseStringToNum(text, outType=None)


@pytest.mark.parametrize("text", ["0x10", "-0x10", "0xdead"])
def test_filename_parser_hex_integer_converts_to_exact_domains(text: str) -> None:
    integer = int(text, 0)
    assert slm.parseStringToNum(text, outType=Fraction) == Fraction(integer, 1)
    assert slm.parseStringToNum(text, outType=gmpy2.mpq) == gmpy2.mpq(
        integer, 1
    )
    complex_bigfloat = slm.parseStringToNum(text, outType=gmpy2.mpc)
    assert isinstance(complex_bigfloat, gmpy2.mpc)
    assert complex_bigfloat.real == gmpy2.mpfr(integer)
    assert complex_bigfloat.imag == 0


@pytest.mark.parametrize(
    ("text", "real", "imaginary"),
    [
        ("0x1.8p1", 3, 0),
        ("0x3im", 0, 3),
        ("0x10+0x3im", 16, 3),
    ],
)
def test_filename_parser_bigfloat_complex_parses_each_hex_component(
    text: str, real: int, imaginary: int
) -> None:
    value = slm.parseStringToNum(text, outType=gmpy2.mpc)
    assert value.real == gmpy2.mpfr(real)
    assert value.imag == gmpy2.mpfr(imaginary)


@pytest.mark.parametrize(
    "text", ["0x1p-1", "0x1p-1+0x2im", "0x1p+1+0x2im", "1 2"]
)
def test_filename_parser_bigfloat_complex_keeps_base_split_failures(
    text: str,
) -> None:
    with pytest.raises(ValueError):
        slm.parseStringToNum(text, outType=gmpy2.mpc)


@pytest.mark.parametrize("text", ["1.25", "1,25", "1.0//2"])
def test_filename_parser_big_rational_rejects_decimal_points(text: str) -> None:
    with pytest.raises(ValueError, match="Rational literal"):
        slm.parseStringToNum(text, outType=gmpy2.mpq)


@pytest.mark.parametrize("roi_type", [list, tuple])
def test_image_roi_selectors_are_splatted_across_dimensions(roi_type) -> None:
    lattice = slm.natlat((3, 3))
    first = slm.LF[slm.Intensity](
        np.asarray(
            [[1.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 1.0]]
        ),
        lattice,
    )
    second = slm.LF[slm.Intensity](first.data.copy(), lattice)
    roi = roi_type((slice(0, 2), slice(0, 2)))

    center, theta = slm.getOrientation(
        [first, second], [1.0, 2.0], roi=roi
    )
    assert center.shape == (2,)
    assert np.isscalar(theta)

    result = slm.dualate(
        first,
        lattice,
        [0.0, 0.0],
        0.0,
        roi=roi,
    )
    assert result.shape == (3, 3)
