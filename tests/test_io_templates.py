from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import os
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageFont
import pytest

from slmtools.bmp8 import HEADER_SZ, save_gray8bmp
from slmtools.image_processing import (
    castImage,
    dualate,
    getImagesAndFilenames,
    getOrientation,
    imageToFloatArray,
    linearFit,
    loadDir,
    parseFileName,
    parseStringToNum,
    saveBeam,
    savePhase,
    savePhase8BMP,
)
from slmtools.lattice_field import (
    ComplexAmplitude,
    ComplexPhase,
    Intensity,
    LatticeAxis,
    LatticeField,
    Modulus,
    RealPhase,
)
from slmtools.lattice_utils import natlat
from slmtools.misc import SchroffError, centroid, collapse, nabs, safeInverse, window
from slmtools.resampling import LinearInterpolation
from slmtools.templates import (
    ftaText,
    lfBlur,
    lfCap,
    lfGaussian,
    lfHeart,
    lfParabola,
    lfPointer,
    lfRect,
    lfRing,
    lfSmile,
)
from slmtools.visualization import look


def test_image_conversion_uses_rec601_and_fixed_point_quantization() -> None:
    rgb = np.asarray([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    expected = np.asarray([[76, 150, 29]], dtype=float) / 255
    np.testing.assert_array_equal(imageToFloatArray(rgb), expected)

    rgba = np.concatenate((rgb, np.zeros((1, 3, 1), dtype=np.uint8)), axis=-1)
    np.testing.assert_array_equal(imageToFloatArray(rgba), expected)
    np.testing.assert_array_equal(
        imageToFloatArray(np.asarray([[0, 128, 255]], dtype=np.uint8)),
        np.asarray([[0, 128, 255]], dtype=float) / 255,
    )
    floats = np.asarray([[[0.25, 0.5, 0.75]]])
    assert imageToFloatArray(floats)[0, 0] == pytest.approx(
        0.299 * 0.25 + 0.587 * 0.5 + 0.114 * 0.75
    )

    # Colors.jl combines raw N0f8 channels using Float32 arithmetic before
    # rounding.  Both probes sit just above a binary half-way value there.
    rounding_probes = np.asarray([[[0, 0, 250], [0, 12, 4]]], dtype=np.uint8)
    np.testing.assert_array_equal(
        imageToFloatArray(rounding_probes),
        np.asarray([[29, 8]], dtype=np.float64) / 255,
    )


def test_filename_parsing_preserves_cues_and_decimal_comma() -> None:
    assert parseStringToNum("-12") == -12
    assert parseStringToNum("-12,5") == -12.5
    assert parseStringToNum("12", outType=np.float32) == np.float32(12)
    assert parseStringToNum("1,25", outType=Decimal) == Decimal("1.25")
    assert parseStringToNum("1//8", outType=Fraction) == Fraction(1, 8)
    with pytest.raises(ValueError, match="Rational literal"):
        parseStringToNum("0,125", outType=Fraction)
    assert parseStringToNum("false", outType=np.bool_) is np.False_
    assert parseStringToNum("true", outType=bool) is np.True_
    assert parseStringToNum(" true ", outType=bool) is np.True_
    with pytest.raises(ValueError, match="Bool literal"):
        parseStringToNum(" 1 ", outType=bool)
    with pytest.raises(ValueError, match="Bool literal"):
        parseStringToNum("not-a-bool", outType=np.bool_)
    assert parseFileName("17.bmp") == 17
    assert parseFileName("prefix x-3,25 tail.bmp", "x", "after") == -3.25
    assert parseFileName("prefix -19 x tail.bmp", "x", "before") == -19
    assert parseFileName("x1 x2.bmp", "x", "a") == 2
    with pytest.raises(ValueError, match="Unrecognized look"):
        parseFileName("x1.bmp", "x", "sideways")


def test_directory_loading_retains_upstream_concatenation_contract(tmp_path: Path) -> None:
    Image.fromarray(np.full((2, 3), 64, dtype=np.uint8)).save(tmp_path / "2.bmp")
    Image.fromarray(np.full((2, 3), 128, dtype=np.uint8)).save(tmp_path / "1.bmp")
    Image.fromarray(np.full((2, 3), 255, dtype=np.uint8)).save(tmp_path / "3.BMP")

    directory = str(tmp_path) + os.sep
    with pytest.raises(TypeError, match="string"):
        getImagesAndFilenames(tmp_path, ".bmp")
    # Julia filters first and only then broadcasts `load` over concatenated
    # paths.  No matches therefore succeeds without a trailing separator.
    empty_images, empty_names = getImagesAndFilenames(str(tmp_path), ".png")
    assert empty_images == [] and empty_names == []
    # A match exposes the source's literal `directory * filename` path bug.
    with pytest.raises(FileNotFoundError):
        getImagesAndFilenames(str(tmp_path), ".bmp")
    images, names = getImagesAndFilenames(directory, ".bmp")
    assert names == ["1.bmp", "2.bmp"]
    assert len(images) == 2
    fields, params = loadDir(directory, ".bmp", L=(0.5, 2.0), flambda=7)
    assert params == [1.0, 2.0]
    assert [field.field_type for field in fields] == [Intensity, Intensity]
    assert fields[0].shape == (2, 3)
    np.testing.assert_array_equal(fields[0].L[0], [0.5, 1.0])
    np.testing.assert_array_equal(fields[0].L[1], [2.0, 4.0, 6.0])
    assert fields[0].flambda == 7
    assert fields[0].data[0, 0] == pytest.approx(128 / 255)

    direct = castImage(RealPhase, images[0], fields[0].L, 3)
    assert direct.field_type is RealPhase and direct.flambda == 3

    float32_fields, _ = loadDir(directory, ".bmp", L=np.float32(0.1))
    assert all(axis.dtype == np.float32 for axis in float32_fields[0].L)


def test_load_dir_preserves_logical_range_arithmetic_and_metadata(
    tmp_path: Path,
) -> None:
    Image.fromarray(np.zeros((1, 11), dtype=np.uint8)).save(tmp_path / "1.bmp")

    directory = str(tmp_path) + os.sep
    generated, _ = loadDir(directory, ".bmp", L=np.float16(0.001))
    assert generated[0].L[1].dtype == np.dtype(np.float16)
    assert generated[0].L[1].view(np.uint16)[-1] == np.uint16(0x21A3)

    row = LatticeAxis.from_start_step(
        np.float16(0.25), np.float16(0.125), 1
    )
    columns = LatticeAxis.from_start_step(
        np.float16(0.001), np.float16(0.001), 11
    )
    explicit, _ = loadDir(directory, ".bmp", L=(row, columns))
    assert explicit[0].L[0] is row
    assert explicit[0].L[1] is columns
    assert explicit[0].L[1]._logical_ref == columns._logical_ref
    assert explicit[0].L[1]._logical_step == columns._logical_step


@pytest.mark.parametrize("width", range(1, 6))
def test_gray8bmp_header_palette_orientation_and_padding(tmp_path: Path, width: int) -> None:
    source = np.vstack(
        (
            np.linspace(0, 1, width, dtype=float),
            np.linspace(1, 0, width, dtype=float),
        )
    )
    path = tmp_path / f"w{width}.bmp"
    save_gray8bmp(path, source)
    raw = path.read_bytes()
    row_size = width + (-width) % 4
    assert raw[:2] == b"BM"
    assert struct.unpack_from("<I", raw, 2)[0] == HEADER_SZ + 2 * row_size
    assert struct.unpack_from("<I", raw, 10)[0] == HEADER_SZ
    assert struct.unpack_from("<ii", raw, 18) == (width, 2)
    assert struct.unpack_from("<H", raw, 28)[0] == 8
    assert raw[54:58] == b"\x00\x00\x00\x00"
    assert raw[54 + 4 * 255 : 54 + 4 * 256] == b"\xff\xff\xff\x00"
    assert raw[HEADER_SZ : HEADER_SZ + width] == np.rint(source[-1] * 255).astype(np.uint8).tobytes()
    assert raw[HEADER_SZ + width : HEADER_SZ + row_size] == b"\x00" * ((-width) % 4)
    with Image.open(path) as loaded:
        np.testing.assert_array_equal(np.asarray(loaded), np.rint(source * 255).astype(np.uint8))


def test_phase_saving_and_unavailable_upstream_save_beam(tmp_path: Path) -> None:
    phase = np.exp(1j * np.asarray([[-np.pi, -np.pi / 2, 0, np.pi / 2]]))
    png = tmp_path / "phase.png"
    bmp = tmp_path / "phase.bmp"
    savePhase(phase.astype(np.complex128), png)
    savePhase8BMP(phase.astype(np.complex128), bmp)
    expected = np.asarray([[0, 64, 128, 191]], dtype=np.uint8)
    with Image.open(png) as image:
        np.testing.assert_array_equal(np.asarray(image), expected)
    with Image.open(bmp) as image:
        np.testing.assert_array_equal(np.asarray(image), expected)

    real_phase = LatticeField(
        np.asarray([[-0.25, 1.25]]),
        (np.asarray([0.0]), np.asarray([0.0, 1.0])),
        field_type=RealPhase,
    )
    real_png = tmp_path / "real.png"
    savePhase(real_phase, real_png)
    with Image.open(real_png) as image:
        np.testing.assert_array_equal(np.asarray(image), [[191, 64]])

    beam = np.exp(1j * np.asarray([[np.pi / 2, -np.pi / 2]]))
    with pytest.raises(NotImplementedError, match="audited Julia"):
        saveBeam(
            beam,
            "sample",
            ("beamCsv", "angleCsv", "anglePng", "negativeAnglePng"),
            dir=tmp_path,
        )
    assert list(tmp_path.glob("sample-*")) == []

    with pytest.raises(TypeError, match="complex128"):
        savePhase(phase.astype(np.complex64), tmp_path / "complex64.png")
    with pytest.raises(TypeError, match="complex128"):
        savePhase8BMP(phase.astype(np.complex64), tmp_path / "complex64.bmp")


def test_misc_helpers_keep_julia_index_and_threshold_conventions() -> None:
    array = np.zeros((4, 5))
    array[1, 2] = 10
    np.testing.assert_array_equal(centroid(array), [2, 3])
    index = window(array, 3)
    np.testing.assert_array_equal(index[0].ravel(), [1, 2, 3])
    np.testing.assert_array_equal(index[1].ravel(), [2, 3, 4])
    assert array[index].shape == (3, 3)
    np.testing.assert_array_equal(collapse(np.arange(24).reshape(2, 3, 4), 2), np.sum(np.arange(24).reshape(2, 3, 4), axis=(0, 2)))
    np.testing.assert_array_equal(collapse(np.asarray([1.0, 2.0]), 0), [3.0])
    np.testing.assert_array_equal(collapse(np.asarray([1.0, 2.0]), 1), [1.0, 2.0])
    np.testing.assert_array_equal(collapse(np.asarray([1.0, 2.0]), 2), [3.0])
    np.testing.assert_array_equal(collapse(np.asarray(3.0), 0), [3.0])
    with pytest.raises(ValueError):
        collapse(np.asarray([1.0, 2.0]), -1)
    with pytest.raises(TypeError):
        collapse(np.asarray([1.0, 2.0]), np.int32(1))
    np.testing.assert_allclose(nabs([3, -4]), [0.6, 0.8])
    assert safeInverse(0.0) == 0.0 and safeInverse(4) == 0.25

    lattice = (np.asarray([10.0, 20.0, 30.0, 40.0]), np.arange(5.0))
    field = LatticeField(array, lattice, field_type=Intensity)
    np.testing.assert_array_equal(centroid(field), [20, 2])
    assert SchroffError(field, field) == pytest.approx(0)

    large_axis_field = LatticeField(
        np.ones(4, dtype=np.float32),
        (np.arange(100_000_000, 100_000_004),),
        field_type=Intensity,
    )
    result = centroid(large_axis_field)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.asarray([100_000_000], dtype=np.float32))


def test_linear_fit_orientation_and_exact_node_dualation() -> None:
    slope, intercept = linearFit([-1, 0, 1], [8, 10, 12])
    assert slope == pytest.approx(2) and intercept == pytest.approx(10)

    rational_slope, rational_intercept = linearFit(
        [Fraction(0), Fraction(1), Fraction(2)],
        [Fraction(1), Fraction(3), Fraction(5)],
    )
    assert isinstance(rational_slope, np.float64)
    assert isinstance(rational_intercept, np.float64)
    assert rational_slope == pytest.approx(2)
    assert rational_intercept == pytest.approx(1)
    mixed_slope, mixed_intercept = linearFit(
        [Fraction(0), 1.0, 2.0], [1.0, 3.0, 5.0]
    )
    assert isinstance(mixed_slope, np.float64)
    assert isinstance(mixed_intercept, np.float64)
    assert mixed_slope == pytest.approx(2.0)
    assert mixed_intercept == pytest.approx(0.9999999999999994)

    decimal_slope, decimal_intercept = linearFit(
        [Decimal(0), Decimal(1), Decimal(2)],
        [Decimal(1), Decimal(3), Decimal(5)],
    )
    assert decimal_slope == Decimal(2)
    assert abs(decimal_intercept - Decimal(1)) <= Decimal("1e-27")

    # Julia's backslash polyalgorithm uses LU for the square two-sample
    # design, so coincident x coordinates are singular rather than silently
    # returning a least-squares minimum-norm pair.
    with pytest.raises(np.linalg.LinAlgError):
        linearFit([1.0, 1.0], [2.0, 3.0])
    with pytest.raises(np.linalg.LinAlgError):
        linearFit(
            [Decimal(1), Decimal(1)],
            [Decimal(2), Decimal(3)],
        )

    # Rectangular rank-deficient designs instead use pivoted QR.
    one_slope, one_intercept = linearFit([1.0], [2.0])
    assert one_slope == pytest.approx(0.9999999999999996)
    assert one_intercept == pytest.approx(0.9999999999999999)
    tall_slope, tall_intercept = linearFit(
        [1.0, 1.0, 1.0], [2.0, 3.0, 4.0]
    )
    assert tall_slope == pytest.approx(1.4999999999999993)
    assert tall_intercept == pytest.approx(1.4999999999999993)

    bool_slope, bool_intercept = linearFit(
        np.asarray([False, True]), np.asarray([True, False])
    )
    assert bool_slope == pytest.approx(-1.0)
    assert bool_intercept == pytest.approx(1.0)
    int_slope, int_intercept = linearFit(
        np.asarray([1, 2], dtype=np.int32),
        np.asarray([3, 5], dtype=np.uint64),
    )
    assert int_slope == pytest.approx(2.0)
    assert int_intercept == pytest.approx(1.0)
    for invalid_xs in ((1, 2), range(1, 3), ["1", "2"]):
        with pytest.raises(TypeError):
            linearFit(invalid_xs, [3, 5])

    lattice = (np.asarray([8.0, 10.0, 12.0]), np.asarray([19.0, 20.0, 21.0]))
    fields = []
    for position in [(0, 2), (1, 1), (2, 0)]:
        data = np.zeros((3, 3))
        data[position] = 1
        fields.append(LatticeField(data, lattice, field_type=Intensity))
    center, theta = getOrientation(fields, [-1, 0, 1])
    np.testing.assert_allclose(center, [10, 20])
    assert theta == pytest.approx(np.arctan2(-1, 2))
    with pytest.raises(TypeError, match="Vector"):
        getOrientation(tuple(fields), [-1, 0, 1])
    with pytest.raises(TypeError, match="Vector"):
        getOrientation(fields, range(-1, 2))

    rational_axis = np.asarray(
        [Fraction(0), Fraction(1)], dtype=object
    )
    floating_axis = np.asarray([0.0, 1.0])
    mixed_fields = []
    for index, position in enumerate(((0, 0), (1, 0), (1, 1))):
        values = np.zeros((2, 2), dtype=np.int64)
        values[position] = 1
        axes = (
            (rational_axis, rational_axis)
            if index == 0
            else (floating_axis, floating_axis)
        )
        mixed_fields.append(
            LatticeField[Intensity, np.int64, 2](values, axes)
        )
    mixed_center, mixed_theta = getOrientation(
        mixed_fields, [0.0, 1.0, 2.0], threshold=0
    )
    np.testing.assert_allclose(
        mixed_center,
        [0.16666666666666669, -0.16666666666666674],
        rtol=0,
        atol=1e-15,
    )
    assert mixed_theta == pytest.approx(np.pi / 4)

    decimal_lattice = (
        np.asarray([Decimal(0), Decimal(1)], dtype=object),
        np.asarray([Decimal(0), Decimal(1)], dtype=object),
    )
    decimal_fields = []
    for position in ((0, 0), (1, 0), (1, 1)):
        values = np.full((2, 2), Decimal(0), dtype=object)
        values[position] = Decimal(1)
        decimal_fields.append(
            LatticeField[Intensity, object, 2](values, decimal_lattice)
        )
    decimal_center, decimal_theta = getOrientation(
        decimal_fields, [Decimal(0), Decimal(1), Decimal(2)]
    )
    expected_decimal_center = (
        Decimal(1) / Decimal(6),
        -Decimal(1) / Decimal(6),
    )
    assert all(
        abs(actual - expected) <= Decimal("1e-27")
        for actual, expected in zip(
            decimal_center, expected_decimal_center, strict=True
        )
    )
    assert isinstance(decimal_theta, Decimal)
    assert decimal_theta == Decimal("0.7853981633974483096156608458")

    decimal_axis = np.asarray(
        [Decimal(0), Decimal(1), Decimal(2)], dtype=object
    )
    decimal_input = LatticeField[Intensity, object, 2](
        np.asarray(
            [
                [Decimal(1), Decimal(2), Decimal(3)],
                [Decimal(4), Decimal(5), Decimal(6)],
                [Decimal(7), Decimal(8), Decimal(9)],
            ],
            dtype=object,
        ),
        (decimal_axis, decimal_axis),
        Decimal(1),
    )
    decimal_dual = dualate(
        decimal_input,
        (decimal_axis, decimal_axis),
        [Decimal(1), Decimal(1)],
        Decimal(0),
        Decimal(1),
        interpolation=LinearInterpolation,
    )
    assert decimal_dual.data.dtype == np.dtype(object)
    assert decimal_dual.flambda == Decimal(1)
    assert all(axis.dtype == np.dtype(object) for axis in decimal_dual.L)
    assert decimal_dual.data[1, 1] == Decimal(5)
    assert abs(
        decimal_dual.data[0, 0] - Decimal(11) / Decimal(3)
    ) < Decimal("1e-25")

    source_lattice = (np.asarray([-1.0, 0.0, 1.0]),) * 2
    target_lattice = (np.asarray([-1 / 3, 0.0, 1 / 3]),) * 2
    data = np.arange(9, dtype=float).reshape(3, 3)
    source = LatticeField(data, source_lattice, 4, field_type=RealPhase)
    result = dualate(source, target_lattice, [0, 0], 0)
    np.testing.assert_allclose(result.data, data, atol=1e-13)
    assert result.field_type is RealPhase and result.flambda == 1
    natural = dualate(source, target_lattice, [0, 0], 0, naturalize=True)
    assert natural.flambda == 1
    for actual, expected_axis in zip(natural.L, natlat((3, 3))):
        np.testing.assert_allclose(actual, expected_axis)


def test_template_overloads_standard_output_and_normalization() -> None:
    lattice = (np.asarray([-1.0, 0.0, 1.0]), np.asarray([-2.0, 0.0, 2.0]))
    parabola = lfParabola(RealPhase, lattice, 2.0, (1.0, -1.0), flambda=3)
    x, y = np.meshgrid(*lattice, indexing="ij")
    np.testing.assert_allclose(parabola.data, x**2 + y**2 + x - y)
    inherited = lfParabola(parabola, np.eye(2), center=(1, 0))
    assert inherited.field_type is RealPhase and inherited.flambda == 3
    np.testing.assert_allclose(inherited.data, ((x - 1) ** 2 + y**2) / 2)

    wrapped = lfParabola(ComplexPhase, lattice, 1.0)
    np.testing.assert_allclose(np.abs(wrapped.data), 1)
    clipped = lfParabola(Modulus, lattice, -1.0)
    assert np.all(clipped.data >= 0) and np.count_nonzero(clipped.data) == 0

    gaussian = lfGaussian(Intensity, lattice, 0.8, 2.5)
    cell = (lattice[0][1] - lattice[0][0]) * (lattice[1][1] - lattice[1][0])
    assert np.sum(gaussian.data**2) * cell == pytest.approx(2.5)
    matrix_gaussian = lfGaussian(Intensity, lattice, np.diag([1.0, 2.0]))
    assert matrix_gaussian.data.shape == (3, 3)
    assert lfRing(Intensity, lattice, 1.0, 0.2).shape == (3, 3)
    np.testing.assert_array_equal(lfCap(Intensity, lattice, 2.0, 1.0).data, np.maximum(1 - x**2 - y**2, 0))
    rectangle = lfRect(Intensity, lattice, (2.0, 2.0), 4.0)
    np.testing.assert_array_equal(
        rectangle.data, 4.0 * ((np.abs(x) <= 1) & (np.abs(y) <= 1))
    )
    assert np.max(rectangle.data) == 4


def test_template_matrix_overloads_use_the_leading_dimension_block() -> None:
    lattice = (np.asarray([-1.0, 0.0, 1.0]),)
    oversized = np.asarray([[2.0, 91.0], [73.0, 101.0]])

    parabola = lfParabola(RealPhase, lattice, oversized)
    reference_parabola = lfParabola(
        RealPhase, lattice, np.asarray([[2.0]])
    )
    np.testing.assert_array_equal(parabola.data, reference_parabola.data)

    gaussian = lfGaussian(Intensity, lattice, oversized)
    reference_gaussian = lfGaussian(
        Intensity, lattice, np.asarray([[2.0]])
    )
    np.testing.assert_array_equal(gaussian.data, reference_gaussian.data)

    with pytest.raises(ValueError, match="leading block"):
        lfParabola(
            RealPhase,
            (lattice[0], lattice[0]),
            np.asarray([[2.0]]),
        )


@pytest.mark.parametrize(
    "matrix_factory",
    [
        lambda values: values,
        lambda values: tuple(tuple(row) for row in values),
    ],
)
def test_template_matrix_literals_use_julia_element_promotion(
    matrix_factory: object,
) -> None:
    lattice = (
        LatticeAxis(
            np.asarray([1.0, 2.0], dtype=np.float32),
            step_hint=np.float32(1),
        ),
    )
    values = [
        [np.int64(100_000_001), np.float32(0)],
        [np.float32(0), np.float32(1)],
    ]
    matrix = matrix_factory(values)

    # Julia constructs this literal as Matrix{Float32}; the leading 1x1 block
    # is therefore 100_000_000 before the parabola is evaluated.
    result = lfParabola(RealPhase, lattice, matrix)
    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        result.data, np.asarray([50_000_000.0, 200_000_000.0])
    )


def test_text_emojis_blur_and_look_return_arrays() -> None:
    lattice = natlat(24, 24)
    font = ImageFont.load_default()
    text = ftaText("A", (24, 24), fnt=font, pixelsize=12)
    assert text.shape == (24, 24) and text.dtype == np.float64 and np.max(text) > 0
    for field in (
        lfHeart(Intensity, lattice, 0.4),
        lfSmile(Intensity, lattice, 0.4),
        lfPointer(Intensity, lattice, 0.4),
    ):
        assert field.shape == (24, 24)
        assert np.max(field.data) > 0

    impulse = np.zeros((5, 5))
    impulse[2, 2] = 1
    blurred = lfBlur(LatticeField(impulse, natlat(5, 5), 9, field_type=Intensity), 0.5)
    assert blurred.field_type is Intensity and blurred.flambda == 9
    assert blurred.shape == (5, 5) and np.all(blurred.data >= 0)
    complex_blur = lfBlur(
        LatticeField(impulse * (1 + 1j), natlat(5, 5), field_type=ComplexAmplitude),
        0.5,
    )
    assert np.iscomplexobj(complex_blur.data)

    real_phase = LatticeField(np.asarray([[2.2, 2.8]]), (np.asarray([0]), np.arange(2)), field_type=RealPhase)
    np.testing.assert_allclose(look(real_phase), [[0, 0.6]])
    amplitude = LatticeField(np.asarray([[1 + 0j, 1j]]), (np.asarray([0]), np.arange(2)), field_type=ComplexAmplitude)
    assert look(amplitude).shape == (1, 4)
    assert look(real_phase, amplitude).shape == (1, 6)


def test_default_text_font_matches_freetypeabstraction_resolution() -> None:
    rounded_arial = Path(
        "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"
    )
    if not rounded_arial.exists():
        pytest.skip("locked macOS Arial Rounded MT Bold font is unavailable")
    rendered = ftaText("A", (32, 32), pixelsize=16)
    assert np.count_nonzero(rendered) == 74
    assert np.sum(rendered) == pytest.approx(50.88235294117648)


def test_default_text_matches_locked_freetype_pixel_masks() -> None:
    rounded_arial = Path(
        "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"
    )
    if not rounded_arial.exists():
        pytest.skip("locked macOS Arial Rounded MT Bold font is unavailable")

    def rows(*hex_rows: str) -> np.ndarray:
        return np.vstack(
            [np.frombuffer(bytes.fromhex(row), dtype=np.uint8) for row in hex_rows]
        )

    expected_a = np.zeros((32, 32), dtype=np.uint8)
    expected_a[10:21, 10:22] = rows(
        "000000006fefc10e00000000",
        "00000018f9ffff8000000000",
        "0000007eff96fce706000000",
        "000005e5f714acff5e000000",
        "000058ff9e003effcd000000",
        "0000c4ff320000ceff3e0000",
        "0030ffffffffffffffae0000",
        "009dfffffffffffffffc2000",
        "12f6ff3400000000c5ff8d00",
        "6effd6000000000064ffed02",
        "55f35c00000000000ad6ca02",
    )

    expected_hi = np.zeros((32, 48), dtype=np.uint8)
    expected_hi[10:21, 17:32] = rows(
        "74f4570000000032ee9f0000a6e518",
        "baff970000000067ffea0000a9e718",
        "bcff980000000068ffec0000000000",
        "bcff980000000068ffec0000a6e415",
        "bcffffffffffffffffec0000e3ff3b",
        "bcffffffffffffffffec0000e4ff3c",
        "bcff980000000068ffec0000e4ff3c",
        "bcff980000000068ffec0000e4ff3c",
        "bcff980000000068ffec0000e4ff3c",
        "baff950000000066ffea0000e3ff3b",
        "74f451000000002ded9f0000a3e413",
    )

    expected_gyp = np.zeros((32, 48), dtype=np.uint8)
    expected_gyp[11:22, 10:39] = rows(
        "00098decf2a00cc6a90060f13f00000079e81b00bbc90a9df3e78b0800",
        "00a7ffffffffbcffec0067ffb8000007edf91100efffbbffffffffa900",
        "2affff750d26c5fff4000aedfc180055ffa70000f0ffcb2a127effff25",
        "58ffda0000003bfff4000084ff7200b5ff3e0000f0ff44000000d5ff56",
        "52ffd900000033fff4000018f8cf18fcd5000000f0ff43000000d6ff58",
        "20ffff730d2ac6fff40000009dffa1ff6c000000f0ffc928107bfffe23",
        "00a8ffffffffcdfff40000002bfffff40e000000f0ffc9ffffffff9a00",
        "000890e9f39e49fff000000000d9ff9900000000f0ff23aaf5e8810500",
        "07daad250317acffd500010a57ffff3000000000f0ff14000000000000",
        "02daffffffffffff7700b3ffffffb80000000000efff11000000000000",
        "001795e0fbf5d172020085f4f6b3130000000000bcce00000000000000",
    )

    for text, size, expected in (
        ("A", (32, 32), expected_a),
        ("Hi", (32, 48), expected_hi),
        ("gyp", (32, 48), expected_gyp),
    ):
        actual = np.rint(ftaText(text, size, pixelsize=16) * 255).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected)

    with pytest.raises(IndexError):
        ftaText("", (32, 32), pixelsize=16)


@pytest.mark.parametrize(
    ("text", "expected_digest"),
    (
        ("AV", "4bb47eeb3be6c84756958536ba2e2fd94bb851ce89ac317a783829cba9a5676f"),
        ("VA", "fb397173324fd8f2c13fb4bf97cb315a2cf5f23b5f4814ee5aa6e558716c94b9"),
        ("To", "7cc4fd4e7480b31deab7d3b2b87e05e7a3ae14f79d7c0813651f13d6b80294de"),
    ),
)
def test_text_kerning_matches_locked_freetype(
    text: str, expected_digest: str
) -> None:
    import hashlib

    rounded_arial = Path(
        "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"
    )
    if not rounded_arial.exists():
        pytest.skip("locked macOS Arial Rounded MT Bold font is unavailable")
    rendered = np.rint(
        ftaText(text, (32, 40), pixelsize=16) * 255
    ).astype(np.uint8)
    assert hashlib.sha256(rendered.tobytes()).hexdigest() == expected_digest


def test_look_supports_exact_real_arrays_but_not_array_varargs() -> None:
    rational = np.asarray([Fraction(1, 2), Fraction(1)], dtype=object)
    assert look(rational).tolist() == [Fraction(1, 2), Fraction(1)]

    with pytest.raises(TypeError, match="lattice fields"):
        look(np.ones((1, 1)), np.ones((1, 1)))


def test_lfparabola_rejects_complex_real_signature_arguments() -> None:
    lattice = (np.arange(3.0),)
    with pytest.raises(TypeError, match="quadratic coefficient must be real"):
        lfParabola(RealPhase, lattice, 1 + 2j)
    with pytest.raises(TypeError, match="linear coefficients must be real"):
        lfParabola(RealPhase, lattice, 1.0, (1 + 0j,))
    with pytest.raises(TypeError, match="scalar radius must be real"):
        lfGaussian(Intensity, lattice, 1 + 0j)
    with pytest.raises(TypeError, match="norm must be real"):
        lfGaussian(Intensity, lattice, 1.0, 1 + 0j)
    with pytest.raises(TypeError, match="curvature and height must be real"):
        lfCap(Intensity, lattice, 1 + 0j, 1.0)
    with pytest.raises(TypeError, match="side lengths and height must be real"):
        lfRect(Intensity, lattice, (1 + 0j,), 1.0)
    field = LatticeField(np.ones(3), lattice, field_type=Intensity)
    with pytest.raises(TypeError, match="radius must be real"):
        lfBlur(field, 1 + 0j)
