from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from slmtools.lattice_field import Intensity, LF
from slmtools.resampling import (
    CubicSplineInterpolation,
    Flat,
    Linear,
    OnCell,
    OnGrid,
    Periodic,
    cubic_spline_interpolation,
    downsample,
    upsample,
)


def _flat_oncell_factory(
    ranges: object,
    values: object,
    *,
    extrapolation_bc: object = 0,
) -> object:
    return CubicSplineInterpolation(
        ranges,
        values,
        bc=Flat(OnCell()),
        extrapolation_bc=extrapolation_bc,
    )


def _flat_ongrid_factory(
    ranges: object,
    values: object,
    *,
    extrapolation_bc: object = 0,
) -> object:
    return CubicSplineInterpolation(
        ranges,
        values,
        bc=Flat(OnGrid()),
        extrapolation_bc=extrapolation_bc,
    )


def test_float16_flat_oncell_upsample_matches_julia_prefilter_bits() -> None:
    for values, expected_bits in (
        ([1, 4], [0x3C00, 0x43FF]),
        ([1, 4, 2, 8], [0x3C00, 0x43FF, 0x4000, 0x4801]),
    ):
        size = len(values)
        result = upsample(
            np.array(values, dtype=np.float16),
            (range(1, size + 1),),
            (range(1, size + 1),),
            interpolation=_flat_oncell_factory,
        )

        assert result.dtype == np.dtype(np.float16)
        np.testing.assert_array_equal(
            result.view(np.uint16),
            np.array(expected_bits, dtype=np.uint16),
        )


def test_float16_flat_oncell_mixed_queries_match_julia_bits() -> None:
    interpolator = CubicSplineInterpolation(
        np.arange(1, 5, dtype=np.float16),
        np.array([1, 4, 2, 8], dtype=np.float16),
        bc=Flat(OnCell()),
        extrapolation_bc=Flat(),
    )
    queries = np.array(
        [0.5, 1, 1.25, 1.5, 2, 2.5, 3.5, 4, 4.5],
        dtype=np.float16,
    )

    result = interpolator(queries)

    assert result.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(
        result.view(np.uint16),
        np.array(
            [
                0xB072,
                0x3C00,
                0x4016,
                0x421E,
                0x43FF,
                0x4177,
                0x4462,
                0x4801,
                0x48F0,
            ],
            dtype=np.uint16,
        ),
    )


@pytest.mark.parametrize("resampler", [upsample, downsample])
@pytest.mark.parametrize(
    ("source_bits", "expected_bits"),
    [
        (
            [
                0x3C00,
                0x4400,
                0x4000,
                0x4800,
                0xC200,
                0x3400,
                0x4600,
                0xBC00,
                0x3E00,
                0x4500,
                0xC000,
                0x4200,
            ],
            [
                0x3BFF,
                0x43FE,
                0x4000,
                0x4801,
                0xC200,
                0x3408,
                0x4601,
                0xBBF6,
                0x3E01,
                0x44FF,
                0xC001,
                0x4202,
            ],
        ),
        (
            [
                0x3C00,
                0xC080,
                0x3555,
                0x43A0,
                0xB800,
                0x4100,
                0x2E66,
                0xC240,
                0x3E00,
                0xBC00,
                0x4480,
                0x3400,
            ],
            [
                0x3C01,
                0xC080,
                0x3555,
                0x43A0,
                0xB7FC,
                0x40FF,
                0x2E72,
                0xC240,
                0x3E00,
                0xBC02,
                0x4482,
                0x3406,
            ],
        ),
    ],
)
def test_float16_flat_oncell_2d_resampling_matches_julia_bits(
    resampler: Callable[..., np.ndarray],
    source_bits: list[int],
    expected_bits: list[int],
) -> None:
    values = np.asarray(source_bits, dtype=np.uint16).view(np.float16).reshape(
        (4, 3), order="F"
    )
    lattice = (range(1, 5), range(1, 4))

    result = resampler(
        values,
        lattice,
        lattice,
        interpolation=_flat_oncell_factory,
    )

    assert result.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(
        result.ravel(order="F").view(np.uint16),
        np.asarray(expected_bits, dtype=np.uint16),
    )


def test_float16_flat_oncell_2d_fractional_tensor_matches_julia_bits() -> None:
    values = np.asarray(
        [1, 4, 2, 8, -3, 0.25, 6, -1, 1.5, 5, -2, 3],
        dtype=np.float16,
    ).reshape((4, 3), order="F")
    interpolator = CubicSplineInterpolation(
        (range(1, 5), range(1, 4)),
        values,
        bc=Flat(OnCell()),
    )
    result = interpolator(
        np.asarray([0.5, 1, 1.25, 2.5, 4, 4.5], dtype=np.float16),
        np.asarray([0.5, 1, 1.5, 2.25, 3, 3.5], dtype=np.float16),
    )

    assert result.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(
        result.ravel(order="F").view(np.uint16),
        np.asarray(
            [
                0x3B2E,
                0x40E2,
                0x4392,
                0x404E,
                0x495B,
                0x4B0B,
                0xB078,
                0x3BFF,
                0x4015,
                0x4176,
                0x4801,
                0x48F0,
                0xC025,
                0xBE9E,
                0xBC94,
                0x4375,
                0x4164,
                0x3F26,
                0xC1B5,
                0xC101,
                0xC010,
                0x4348,
                0xBC4A,
                0xC1EE,
                0xB3CA,
                0x3E01,
                0x420E,
                0x3D73,
                0x4202,
                0x451F,
                0x3A38,
                0x4211,
                0x44F2,
                0x385F,
                0x44D9,
                0x4841,
            ],
            dtype=np.uint16,
        ),
    )


@pytest.mark.parametrize(
    ("boundary", "same_bits", "fractional_bits"),
    [
        (
            None,
            [
                0x3BFE,
                0xC080,
                0x3559,
                0x43A1,
                0xB7FE,
                0x4101,
                0x2E64,
                0xC240,
                0x3E01,
                0xBC02,
                0x4481,
                0x3401,
            ],
            [
                0x3BFF,
                0xBCBE,
                0x3D03,
                0x43A0,
                0xAD08,
                0x3919,
                0xB6D3,
                0xB8FA,
                0x3E02,
                0xBA5E,
                0x4477,
                0x3400,
            ],
        ),
        (
            Flat(OnGrid()),
            [
                0x3BFD,
                0xC080,
                0x355B,
                0x43A2,
                0xB801,
                0x40FD,
                0x2E7F,
                0xC23E,
                0x3DFF,
                0xBC01,
                0x4480,
                0x3409,
            ],
            [
                0x3BFD,
                0xB685,
                0x3E50,
                0x43A0,
                0x3268,
                0x3102,
                0x2A23,
                0x3977,
                0x3DFD,
                0xA5F0,
                0x43C8,
                0x3404,
            ],
        ),
        (
            Periodic(OnGrid()),
            [
                0x3BFD,
                0xC07E,
                0x3555,
                0x439F,
                0xB803,
                0x40FD,
                0x2E40,
                0xC242,
                0x3E03,
                0xBBFF,
                0x4480,
                0x33FC,
            ],
            [
                0x3BFD,
                0xBC85,
                0x3DF7,
                0x439E,
                0xAC2C,
                0x33E8,
                0xB992,
                0x35D1,
                0x3E01,
                0xB0CA,
                0x4412,
                0x33E4,
            ],
        ),
        (
            Periodic(OnCell()),
            [
                0x3BFD,
                0xC07E,
                0x3555,
                0x439F,
                0xB803,
                0x40FD,
                0x2E40,
                0xC242,
                0x3E03,
                0xBBFF,
                0x4480,
                0x33FC,
            ],
            [
                0x3BFD,
                0xBC84,
                0x3DF8,
                0x439F,
                0xAC25,
                0x33F1,
                0xB994,
                0x35D1,
                0x3E03,
                0xB0B0,
                0x4412,
                0x33FC,
            ],
        ),
    ],
)
def test_float16_2d_cubic_boundary_families_match_julia_bits(
    boundary: object,
    same_bits: list[int],
    fractional_bits: list[int],
) -> None:
    values = np.asarray(
        [
            0x3C00,
            0xC080,
            0x3555,
            0x43A0,
            0xB800,
            0x4100,
            0x2E66,
            0xC240,
            0x3E00,
            0xBC00,
            0x4480,
            0x3400,
        ],
        dtype=np.uint16,
    ).view(np.float16).reshape((4, 3), order="F")
    keywords = {} if boundary is None else {"bc": boundary}
    interpolator = CubicSplineInterpolation(
        (range(1, 5), range(1, 4)), values, **keywords
    )

    same = interpolator(range(1, 5), range(1, 4))
    fractional = interpolator(
        np.asarray([1, 1.5, 3.25, 4], dtype=np.float16),
        np.asarray([1, 1.5, 3], dtype=np.float16),
    )

    assert same.dtype == np.dtype(np.float16)
    assert fractional.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(
        same.ravel(order="F").view(np.uint16),
        np.asarray(same_bits, dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        fractional.ravel(order="F").view(np.uint16),
        np.asarray(fractional_bits, dtype=np.uint16),
    )


def test_default_cubic_float16_range_storage_matches_julia_bits() -> None:
    vector = np.asarray([1, 4, 2, 8], dtype=np.float16)
    one_dimensional = CubicSplineInterpolation(
        (range(1, 5),), vector
    )
    matrix = np.asarray(
        [1, 4, 2, 8, -3, 0.25, 6, -1, 1.5, 5, -2, 3],
        dtype=np.float16,
    ).reshape((4, 3), order="F")
    two_dimensional = CubicSplineInterpolation(
        (range(1, 5), range(1, 4)), matrix
    )

    result_1d = one_dimensional(range(1, 5))
    result_2d = two_dimensional(range(1, 5), range(1, 4))

    assert result_1d.dtype == np.dtype(np.float16)
    assert result_2d.dtype == np.dtype(np.float16)
    assert np.asarray(one_dimensional(1)).dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        result_1d.view(np.uint16),
        np.asarray([15359, 17408, 16383, 18433], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        result_2d.ravel(order="F").view(np.uint16),
        np.asarray(
            [
                15360,
                17407,
                16382,
                18433,
                49665,
                13301,
                17921,
                48115,
                15873,
                17665,
                49155,
                16895,
            ],
            dtype=np.uint16,
        ),
    )


def test_default_cubic_float32_range_storage_matches_julia_values() -> None:
    vector = np.asarray([1, 4, 2, 8], dtype=np.float32)
    one_dimensional = CubicSplineInterpolation(
        (range(1, 5),), vector
    )
    matrix = np.asarray(
        [1, 4, 2, 8, -3, 0.25, 6, -1, 1.5, 5, -2, 3],
        dtype=np.float32,
    ).reshape((4, 3), order="F")
    two_dimensional = CubicSplineInterpolation(
        (range(1, 5), range(1, 4)), matrix
    )

    result_1d = one_dimensional(range(1, 5))
    result_2d = two_dimensional(range(1, 5), range(1, 4))

    assert result_1d.dtype == np.dtype(np.float32)
    assert result_2d.dtype == np.dtype(np.float32)
    assert np.asarray(one_dimensional(1)).dtype == np.dtype(np.float64)
    np.testing.assert_allclose(
        result_1d,
        np.asarray([1, 3.9999998, 1.9999999, 8], dtype=np.float32),
        rtol=4 * np.finfo(np.float32).eps,
        atol=4 * np.finfo(np.float32).eps,
    )
    np.testing.assert_allclose(
        result_2d.ravel(order="F"),
        np.asarray(
            [
                1,
                4,
                2,
                8,
                -3,
                0.25000015,
                6,
                -1.0000002,
                1.4999999,
                5.0000005,
                -1.9999992,
                3,
            ],
            dtype=np.float32,
        ),
        rtol=8 * np.finfo(np.float32).eps,
        atol=8 * np.finfo(np.float32).eps,
    )


def test_float16_cubic_line_range_storage_matches_julia_bits() -> None:
    interpolator = CubicSplineInterpolation(
        (range(1, 5),),
        np.asarray([1, 4, 2, 8], dtype=np.float16),
        extrapolation_bc=Linear(),
    )

    result = interpolator(range(0, 6))

    assert result.dtype == np.dtype(np.float16)
    assert np.asarray(interpolator(0)).dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        result.view(np.uint16),
        np.asarray(
            [0xC3BC, 0x3BFF, 0x4400, 0x3FFF, 0x4801, 0x4C1E],
            dtype=np.uint16,
        ),
    )


def test_float32_cubic_line_range_storage_matches_julia_values() -> None:
    interpolator = CubicSplineInterpolation(
        (range(1, 5),),
        np.asarray([1, 4, 2, 8], dtype=np.float32),
        extrapolation_bc=Linear(),
    )

    result = interpolator(range(0, 6))

    assert result.dtype == np.dtype(np.float32)
    assert np.asarray(interpolator(0)).dtype == np.dtype(np.float64)
    np.testing.assert_allclose(
        result,
        np.asarray(
            [-3.8666663, 1, 3.9999998, 1.9999999, 8, 16.466667],
            dtype=np.float32,
        ),
        rtol=4 * np.finfo(np.float32).eps,
        atol=4 * np.finfo(np.float32).eps,
    )


def _line_extrapolation_matrix() -> np.ndarray:
    return np.asarray(
        [
            0x3C00,
            0xC080,
            0x3555,
            0x43A0,
            0xB800,
            0x4100,
            0x2E66,
            0xC240,
            0x3E00,
            0xBC00,
            0x4480,
            0x3400,
        ],
        dtype=np.uint16,
    ).view(np.float16).reshape((4, 3), order="F")


def test_float16_cubic_line_2d_uses_julia_additive_gradient_bits() -> None:
    interpolator = CubicSplineInterpolation(
        (range(1, 5), range(1, 4)),
        _line_extrapolation_matrix(),
        extrapolation_bc=Linear(),
    )

    result = interpolator(range(0, 6), range(0, 5))

    assert result.dtype == np.dtype(np.float16)
    assert np.asarray(interpolator(0, 0)).dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        result.ravel(order="F").view(np.uint16),
        np.asarray(
            [
                0x480F,
                0x42BF,
                0xC888,
                0x3EE9,
                0x4AAB,
                0x4C2A,
                0x45BE,
                0x3BFE,
                0xC080,
                0x3559,
                0x43A1,
                0x4724,
                0xC4E3,
                0xB7FE,
                0x4101,
                0x2E64,
                0xC240,
                0xC636,
                0x46CA,
                0x3E01,
                0xBC02,
                0x4481,
                0x3401,
                0xC724,
                0x48D5,
                0x4460,
                0xC691,
                0x4909,
                0x4634,
                0xBCBF,
            ],
            dtype=np.uint16,
        ),
    )


def test_float32_cubic_line_2d_uses_julia_additive_gradient() -> None:
    interpolator = CubicSplineInterpolation(
        (range(1, 5), range(1, 4)),
        _line_extrapolation_matrix().astype(np.float32),
        extrapolation_bc=Linear(),
    )

    result = interpolator(range(0, 6), range(0, 5))

    assert result.dtype == np.dtype(np.float32)
    assert np.asarray(interpolator(0, 0)).dtype == np.dtype(np.float64)
    np.testing.assert_allclose(
        result.ravel(order="F"),
        np.asarray(
            [
                8.120801,
                3.3750002,
                -9.062499,
                1.7248536,
                13.328125,
                16.657421,
                5.745801,
                1.0000001,
                -2.2499998,
                0.333252,
                3.8124998,
                7.1417966,
                -4.88501,
                -0.5000001,
                2.4999998,
                0.09997562,
                -3.125,
                -6.2099614,
                6.7833333,
                1.5,
                -1.0000001,
                4.4999995,
                0.24999985,
                -7.1333337,
                9.658334,
                4.375,
                -6.5625,
                10.058349,
                6.2031245,
                -1.1802089,
            ],
            dtype=np.float32,
        ),
        rtol=8 * np.finfo(np.float32).eps,
        atol=8 * np.finfo(np.float32).eps,
    )


def _raw_float16_cube() -> np.ndarray:
    return np.asarray(
        [
            0x3C00,
            0xC000,
            0x3555,
            0x4200,
            0xB800,
            0x4000,
            0x2E66,
            0xC100,
            0x3E00,
            0xBC00,
            0x4300,
            0x3400,
            0x4400,
            0xB400,
            0x3A00,
            0xC300,
            0x3800,
            0x4080,
            0xBE00,
            0x4500,
            0xC080,
            0x3D00,
            0x3000,
            0x4180,
        ],
        dtype=np.uint16,
    ).view(np.float16).reshape((4, 3, 2), order="F")


@pytest.mark.parametrize("resampler", [upsample, downsample])
@pytest.mark.parametrize(
    ("interpolation", "expected_bits"),
    [
        (
            _flat_oncell_factory,
            [
                0x3C02,
                0xBFFF,
                0x355B,
                0x4203,
                0xB7FF,
                0x4000,
                0x2E7A,
                0xC100,
                0x3E02,
                0xBBFE,
                0x4305,
                0x3408,
                0x4401,
                0xB3F9,
                0x3A09,
                0xC2FF,
                0x3800,
                0x4081,
                0xBDFD,
                0x4501,
                0xC082,
                0x3D01,
                0x300A,
                0x4180,
            ],
        ),
        (
            _flat_ongrid_factory,
            [
                0x3C05,
                0xBFF8,
                0x355F,
                0x4201,
                0xB7F0,
                0x4000,
                0x2E9D,
                0xC0FC,
                0x3DFB,
                0xBC02,
                0x4300,
                0x3408,
                0x4401,
                0xB3EB,
                0x39FB,
                0xC2FB,
                0x37F7,
                0x407E,
                0xBE02,
                0x4500,
                0xC083,
                0x3CFB,
                0x3002,
                0x4181,
            ],
        ),
    ],
    ids=["flat-oncell", "flat-ongrid"],
)
def test_float16_flat_3d_resampling_matches_julia_bits(
    resampler: Callable[..., np.ndarray],
    interpolation: Callable[..., object],
    expected_bits: list[int],
) -> None:
    lattice = (range(1, 5), range(1, 4), range(1, 3))

    result = resampler(
        _raw_float16_cube(),
        lattice,
        lattice,
        interpolation=interpolation,
    )

    assert result.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(
        result.ravel(order="F").view(np.uint16),
        np.asarray(expected_bits, dtype=np.uint16),
    )


def test_float16_flat_oncell_3d_fractional_tensor_matches_julia_bits() -> None:
    interpolator = CubicSplineInterpolation(
        (range(1, 5), range(1, 4), range(1, 3)),
        _raw_float16_cube(),
        bc=Flat(OnCell()),
    )

    result = interpolator(
        np.asarray([0.5, 1.25, 2.5, 4.5], dtype=np.float16),
        np.asarray([0.5, 1.5, 3.5], dtype=np.float16),
        np.asarray([0.5, 1.25, 2.5], dtype=np.float16),
    )

    assert result.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(
        result.ravel(order="F").view(np.uint16),
        np.asarray(
            [
                0x40CC,
                0xB8A3,
                0xC154,
                0x4828,
                0xBAC6,
                0xB52C,
                0x345F,
                0xB9FC,
                0x45C2,
                0x3BFC,
                0x3C1D,
                0xB7C9,
                0x43F5,
                0x39B9,
                0xBE0E,
                0x402F,
                0x366F,
                0x392B,
                0x3186,
                0x34D2,
                0x4080,
                0xA818,
                0x3C15,
                0x352D,
                0x4763,
                0x4307,
                0x3BC4,
                0xC9AE,
                0x4233,
                0x417B,
                0xA9E4,
                0x4123,
                0xC555,
                0xC085,
                0x3C02,
                0x4026,
            ],
            dtype=np.uint16,
        ),
    )


@pytest.mark.parametrize("resampler", [upsample, downsample])
def test_numpy_factory_uses_one_based_cartesian_range_indexing(
    resampler: Callable[..., np.ndarray],
) -> None:
    matrix = np.arange(1, 17).reshape((4, 4), order="F")

    def array_factory(
        _ranges: object,
        _values: object,
        *,
        extrapolation_bc: object = 0,
    ) -> np.ndarray:
        del extrapolation_bc
        return matrix

    result = resampler(
        np.zeros((4, 4), dtype=np.int64),
        (range(1, 5), range(1, 5)),
        (range(1, 3), range(1, 3)),
        interpolation=array_factory,
    )

    np.testing.assert_array_equal(result, np.array([[1, 5], [2, 6]]))
    assert result.shape == (2, 2)


def test_numpy_factory_uses_julia_linear_and_singleton_index_rules() -> None:
    matrix = np.arange(1, 17).reshape((4, 4), order="F")

    def matrix_factory(
        _ranges: object,
        _values: object,
        *,
        extrapolation_bc: object = 0,
    ) -> np.ndarray:
        del extrapolation_bc
        return matrix

    linear = upsample(
        np.zeros(4, dtype=np.int64),
        (range(1, 5),),
        (range(1, 3),),
        interpolation=matrix_factory,
    )
    np.testing.assert_array_equal(linear, np.array([1, 2]))

    def vector_factory(
        _ranges: object,
        _values: object,
        *,
        extrapolation_bc: object = 0,
    ) -> np.ndarray:
        del extrapolation_bc
        return np.arange(1, 5)

    trailing_singleton = upsample(
        np.zeros((4, 4), dtype=np.int64),
        (range(1, 5), range(1, 5)),
        (range(1, 3), range(1, 2)),
        interpolation=vector_factory,
    )
    np.testing.assert_array_equal(
        trailing_singleton, np.array([[1], [2]])
    )


@pytest.mark.parametrize("resampler", [upsample, downsample])
@pytest.mark.parametrize(
    "geometry",
    [range(1, 5), (range(1, 5), range(1, 5))],
)
@pytest.mark.parametrize(
    "explicit_interpolation",
    [None, cubic_spline_interpolation],
)
def test_geometry_resampling_rejects_explicit_interpolation_keyword(
    resampler: Callable[..., object],
    geometry: object,
    explicit_interpolation: object,
) -> None:
    # The keyword belongs only to Julia's array and LatticeField methods.
    resampler(geometry, 2)
    with pytest.raises(TypeError, match="does not accept interpolation"):
        resampler(geometry, 2, interpolation=explicit_interpolation)


@pytest.mark.parametrize("resampler", [upsample, downsample])
@pytest.mark.parametrize(
    "geometry",
    [range(1, 5), (range(1, 5), range(1, 5))],
)
@pytest.mark.parametrize("explicit_bc", [None, 0])
def test_geometry_resampling_rejects_explicit_bc_keyword(
    resampler: Callable[..., object],
    geometry: object,
    explicit_bc: object,
) -> None:
    with pytest.raises(TypeError, match="does not accept bc"):
        resampler(geometry, 2, bc=explicit_bc)


@pytest.mark.parametrize("resampler", [upsample, downsample])
def test_field_resampling_retains_custom_interpolation_keyword(
    resampler: Callable[..., object],
) -> None:
    source = (range(1, 5), range(1, 5))
    target = (range(1, 3), range(1, 3))
    matrix = np.arange(1, 17).reshape((4, 4), order="F")
    field = LF[Intensity](np.zeros((4, 4)), source, 2.0)

    def matrix_factory(
        _ranges: object,
        _values: object,
        *,
        extrapolation_bc: object = 0,
    ) -> np.ndarray:
        del extrapolation_bc
        return matrix

    result = resampler(
        field,
        target,
        interpolation=matrix_factory,
    )

    assert result.field_type is Intensity
    assert result.flambda == 2.0
    np.testing.assert_array_equal(
        np.asarray(result.data), np.array([[1, 5], [2, 6]])
    )
