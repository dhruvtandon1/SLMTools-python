"""Regression coverage for Interpolations.jl compatibility."""

from decimal import Decimal, localcontext
from fractions import Fraction
import unittest

import numpy as np

import slmtools as slm
import slmtools.resampling as resampling
from slmtools.lattice_field import DomainError, Generic, LF, LatticeAxis
from slmtools.resampling import (
    CubicSplineInterpolation,
    Flat,
    Linear,
    LinearInterpolation,
    OnCell,
    OnGrid,
    Periodic,
    Throw,
    coarsen,
    cubic_spline_interpolation,
    downsample,
    upsample,
)


class ResamplingRegressionTests(unittest.TestCase):
    def test_periodic_integer_query_retains_fractional_wrapping(self) -> None:
        axis = np.array([0.1, 0.4, 0.7])
        values = np.array([10.0, 20.0, 30.0])
        interpolator = cubic_spline_interpolation(
            (axis,), values, extrapolation_bc=Periodic()
        )

        self.assertAlmostEqual(interpolator[2], 13.33333333333333, places=13)
        self.assertAlmostEqual(interpolator[2.0], 13.33333333333333, places=13)

        axis32 = axis.astype(np.float32)
        values32 = values.astype(np.float32)
        interpolator32 = cubic_spline_interpolation(
            (axis32,), values32, extrapolation_bc=Periodic()
        )
        wrapped_integer = interpolator32[2]
        self.assertEqual(np.asarray(wrapped_integer).dtype, np.dtype(np.float32))
        self.assertEqual(wrapped_integer, interpolator32[np.float32(2)])

    def test_cubic_float32_and_complex64_explicit_grids_preserve_dtype(self) -> None:
        source = (np.linspace(1, 4, 4, dtype=np.float32),)
        target = (np.linspace(1, 4, 13, dtype=np.float32),)

        real = np.array([1, 4, 2, 8], dtype=np.float32)
        complex_values = real.astype(np.complex64) * np.complex64(1 + 2j)
        real_result = upsample(real, source, target)
        complex_result = upsample(complex_values, source, target)

        self.assertEqual(real_result.dtype, np.dtype(np.float32))
        self.assertEqual(complex_result.dtype, np.dtype(np.complex64))
        np.testing.assert_allclose(
            real_result,
            np.array(
                [
                    1.0,
                    2.1875,
                    3.2,
                    3.8625,
                    4.0,
                    3.534375,
                    2.775,
                    2.128125,
                    2.0,
                    2.690625,
                    4.075,
                    5.921875,
                    8.0,
                ],
                dtype=np.float32,
            ),
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            complex_result,
            real_result * np.complex64(1 + 2j),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_nonaffine_float64_matches_locked_julia_golden(self) -> None:
        values = np.array([1.0, 4.0, 2.0, 8.0])
        targets = np.arange(1.0, 4.01, 0.25)
        expected = np.array(
            [
                1.0,
                2.1875,
                3.2,
                3.8625,
                4.0,
                3.534375,
                2.775,
                2.128125,
                2.0,
                2.690625,
                4.075,
                5.921875,
                8.0,
            ]
        )
        result = cubic_spline_interpolation(range(1, 5), values)[targets]
        np.testing.assert_allclose(result, expected, atol=2e-15, rtol=0)

    def test_linear_interpolation_api_and_linear_compatibility_boundary(self) -> None:
        axis = np.array([0.1, 0.4, 0.7], dtype=np.float32)
        values = np.array([10, 20, 30], dtype=np.float32)
        interpolator = LinearInterpolation(
            axis, values, extrapolation_bc=Linear()
        )

        self.assertIsInstance(Linear(), Linear)
        self.assertEqual(np.asarray(interpolator(np.float32(0.25))).dtype, np.float32)
        self.assertAlmostEqual(float(interpolator(np.float32(0.25))), 15.0, places=5)
        self.assertAlmostEqual(
            float(interpolator(np.float32(2.0))), 73.333336, places=5
        )
        self.assertNotIn("Line", resampling.__dict__)
        self.assertIs(slm.LinearInterpolation, LinearInterpolation)
        self.assertIs(slm.Linear, Linear)
        self.assertFalse(hasattr(slm, "Line"))
        self.assertFalse(hasattr(slm, "Throw"))

    def test_multidimensional_nonuniform_linear_interpolation(self) -> None:
        x = np.array([0.0, 1.0, 3.0])
        y = np.array([0.0, 2.0, 5.0])
        values = x[:, None] + y[None, :]

        # Locked Interpolations.jl 0.16.2 Gridded(Linear()) golden.
        interpolator = LinearInterpolation((x, y), values)
        self.assertEqual(interpolator(1.5, 4.0), 5.5)
        np.testing.assert_allclose(
            interpolator(np.array([0.5, 2.0]), np.array([1.0, 3.5])),
            [[1.5, 4.0], [3.0, 5.5]],
            rtol=0,
            atol=0,
        )

        with self.assertRaises(ValueError):
            LinearInterpolation(
                (x, np.array([0.0, 2.0, 1.0])), values
            )

    def test_tuple_fill_is_one_value_not_an_axis_policy(self) -> None:
        axis = np.arange(3.0)
        values = axis[:, None] + axis[None, :]
        fill = (10.0, 20.0)

        for constructor in (CubicSplineInterpolation, LinearInterpolation):
            interpolator = constructor(
                (axis, axis), values, extrapolation_bc=fill
            )
            self.assertEqual(interpolator(-1.0, 1.0), fill)
            self.assertEqual(interpolator(1.0, 1.0), 2.0)

        # A tuple made entirely from boundary objects remains Julia's
        # per-dimension extrapolation form.
        per_axis = LinearInterpolation(
            (axis, axis),
            values,
            extrapolation_bc=(Flat(), Periodic()),
        )
        self.assertEqual(per_axis(-1.0, 3.0), 1.0)

        directional = LinearInterpolation(
            (axis, axis),
            values,
            extrapolation_bc=((Linear(), Flat()), Flat()),
        )
        self.assertEqual(directional(-1.0, 1.0), 0.0)
        self.assertEqual(directional(3.0, 1.0), 3.0)
        self.assertEqual(directional(1.0, 3.0), 3.0)

    def test_one_dimensional_directional_tuple_requires_julia_nesting(self) -> None:
        axis = np.arange(3.0)
        values = np.array([10.0, 20.0, 30.0])

        for constructor in (LinearInterpolation, CubicSplineInterpolation):
            direct = constructor(
                axis,
                values,
                extrapolation_bc=(Flat(), Periodic()),
            )
            np.testing.assert_array_equal(
                direct[np.array([-1.0, 3.0, 5.0])],
                np.array([10.0, 30.0, 30.0]),
            )

            directional = constructor(
                axis,
                values,
                extrapolation_bc=((Flat(), Periodic()),),
            )
            np.testing.assert_array_equal(
                directional[np.array([-1.0, 3.0, 5.0])],
                np.array([10.0, 20.0, 20.0]),
            )

        fill = (10.0, 20.0)
        filled = LinearInterpolation(axis, values, extrapolation_bc=fill)
        self.assertEqual(filled(-1.0), fill)

    def test_large_axis_uses_linear_memory_solver(self) -> None:
        # A dense natural-spline system for this axis would require about
        # 800 MB before factorization.  The tridiagonal implementation keeps
        # only O(n) vectors and completes as an ordinary smoke test.
        source = np.arange(10_000, dtype=np.float64)
        values = np.sin(source / 317.0) + source / 100_000.0
        targets = np.array([0.0, 123.25, 4000.5, 9998.75, 9999.0])

        result = cubic_spline_interpolation(source, values)[targets]

        self.assertEqual(result.shape, targets.shape)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(result[0], values[0])
        self.assertEqual(result[-1], values[-1])

    def test_large_origin_float32_range_uses_logical_step(self) -> None:
        start = np.float32(1e5)
        step = np.float32(0.1)
        materialized = start + np.arange(10, dtype=np.float32) * step
        source = LatticeAxis(materialized, step_hint=step)
        values = np.arange(10, dtype=np.float32)
        query = np.float32(100000.45)

        # Direct locked-Interpolations.jl 0.16.2 golden.  Interpolating between
        # adjacent materialized floats instead gives the incorrect 4.5384617.
        for constructor in (CubicSplineInterpolation, LinearInterpolation):
            result = constructor(source, values)(query)
            self.assertEqual(np.asarray(result).dtype, np.dtype(np.float32))
            self.assertEqual(result, np.float32(4.53125))

        # The logical marker, rather than equality with only the first
        # materialized difference, decides the range semantics.
        tricky_step = np.float32(319.62222)
        tricky_start = np.float32(426.61673)
        tricky = LatticeAxis(
            tricky_start + np.arange(20, dtype=np.float32) * tricky_step,
            step_hint=tricky_step,
        )
        self.assertEqual(np.diff(tricky)[0], tricky_step)
        self.assertTrue(np.any(np.diff(tricky) != tricky_step))
        self.assertEqual(resampling._logical_step(tricky), tricky_step)

    def test_float16_cubic_and_factor_generated_axis_dtypes(self) -> None:
        source16 = LatticeAxis(
            np.arange(1, 5, dtype=np.float16), step_hint=np.float16(1)
        )
        values16 = np.array([1, 4, 2, 8], dtype=np.float16)
        result16 = CubicSplineInterpolation(source16, values16)(np.float16(1.5))
        self.assertEqual(np.asarray(result16).dtype, np.dtype(np.float16))
        self.assertEqual(result16, np.float16(3.2))

        source32 = LatticeAxis(
            np.arange(1, 2, 0.25, dtype=np.float32),
            step_hint=np.float32(0.25),
        )
        down = downsample(source32, 2)
        up = upsample(source32, 2)
        self.assertEqual(down.dtype, np.dtype(np.float32))
        # Julia's upsample formula contains ``(1 + n) / 2`` and therefore
        # promotes a Float32 range to Float64; downsample has no such term.
        self.assertEqual(up.dtype, np.dtype(np.float64))
        self.assertEqual(np.asarray(down._step_hint).dtype, np.dtype(np.float32))
        self.assertEqual(np.asarray(up._step_hint).dtype, np.dtype(np.float64))

    def test_mixed_scalar_vector_queries_drop_only_scalar_axes(self) -> None:
        x = np.arange(3.0)
        y = np.arange(4.0)
        values = x[:, None] + 10 * y[None, :]

        for constructor in (CubicSplineInterpolation, LinearInterpolation):
            interpolator = constructor((x, y), values)
            row = interpolator[(1.0, np.array([0.5, 1.5, 2.5]))]
            column = interpolator[(np.array([0.5, 1.5]), 2.0)]
            self.assertEqual(row.shape, (3,))
            self.assertEqual(column.shape, (2,))
            np.testing.assert_allclose(row, [6.0, 16.0, 26.0])
            np.testing.assert_allclose(column, [20.5, 21.5])

    def test_cubic_bc_keyword_is_independent_from_extrapolation(self) -> None:
        source = range(1, 5)
        values = np.array([1.0, 4.0, 2.0, 8.0])
        targets = np.array([1.0, 1.25, 1.5, 2.5, 3.75, 4.0])

        flat = CubicSplineInterpolation(
            source, values, bc=Flat(OnGrid()), extrapolation_bc=-9.0
        )
        periodic = cubic_spline_interpolation(
            source,
            values,
            bc=Periodic(OnCell()),
            extrapolation_bc=Flat(),
        )
        np.testing.assert_allclose(
            flat[targets], [1.0, 1.46875, 2.5, 2.625, 7.203125, 8.0]
        )
        np.testing.assert_allclose(
            periodic[targets],
            [1.0, 1.01171875, 2.03125, 2.71875, 7.30859375, 8.0],
        )
        self.assertEqual(flat[0.0], -9.0)
        # Periodic(OnCell()) extends half a cell past each endpoint before
        # Flat extrapolation clamps. Locked Interpolations.jl returns the
        # spline value at x=0.5 here, not the value at the first node x=1.
        self.assertAlmostEqual(periodic[0.0], 4.78125)
        with self.assertRaises(TypeError):
            CubicSplineInterpolation(source, values, bc=Linear())
        with self.assertRaises(TypeError):
            CubicSplineInterpolation(source, values, bc=Throw())
        with self.assertRaises(TypeError):
            CubicSplineInterpolation(source, values, bc=Flat())
        with self.assertRaises(TypeError):
            CubicSplineInterpolation(source, values, bc=Periodic())

    def test_non_natural_endpoint_tangents_match_locked_julia(self) -> None:
        source = np.arange(0.0, 2.0, 0.5)
        values = np.array([0.2, 0.7, -0.3, 2.0])
        targets = np.array([-0.4, 0.0, 0.2, 0.75, 1.5, 1.9])

        flat = CubicSplineInterpolation(
            source,
            values,
            bc=Flat(OnGrid()),
            extrapolation_bc=Linear(),
        )
        periodic = CubicSplineInterpolation(
            source,
            values,
            bc=Periodic(OnCell()),
            extrapolation_bc=Linear(),
        )
        np.testing.assert_allclose(
            flat(targets),
            [0.2, 0.2, 0.43936, -0.025, 2.0, 2.0],
            rtol=0,
            atol=8e-16,
        )
        np.testing.assert_allclose(
            periodic(targets),
            [2.03375, 0.2, 0.2716, 0.03125, 2.0, 0.50375],
            rtol=0,
            atol=8e-16,
        )

    def test_periodic_extrapolation_wraps_closed_upper_endpoint(self) -> None:
        source = np.arange(0.0, 2.0, 0.5)
        values = np.array([0.2, 0.7, -0.3, 2.0])
        for constructor in (CubicSplineInterpolation, LinearInterpolation):
            interpolator = constructor(
                source, values, extrapolation_bc=Periodic()
            )
            np.testing.assert_allclose(
                interpolator(np.array([0.0, 1.5, 3.0])),
                [0.2, 0.2, 0.2],
                rtol=0,
                atol=8e-16,
            )

    def test_periodic_oncell_cubic_uses_half_cell_bounds(self) -> None:
        source = np.arange(4.0)
        values = np.array([0.2, 1.1, -0.4, 2.0])
        interpolator = CubicSplineInterpolation(
            source,
            values,
            bc=Periodic(OnCell()),
            extrapolation_bc=Periodic(),
        )
        np.testing.assert_allclose(
            interpolator(np.array([-1.0, 0.0, 3.0, 4.0, 7.0])),
            [2.0, 0.2, 2.0, 0.2, 2.0],
            rtol=0,
            atol=8e-16,
        )

    def test_placed_cubic_boundaries_match_locked_julia(self) -> None:
        source = np.arange(4.0)
        values = np.array([0.2, 1.1, -0.4, 2.0])
        queries = np.array([-1.0, 0.0, 0.2, 3.0, 4.0])
        line_expected = (
            (Flat(OnGrid()), [0.2, 0.2, 0.31472, 2.0, 2.0]),
            (
                Flat(OnCell()),
                [-0.245559210526316, 0.2, 0.5116842105263156, 2.0,
                 2.8106907894736852],
            ),
            (Periodic(OnGrid()), [0.875, 0.2, 0.2216, 2.0, 2.45]),
            (
                Periodic(OnCell()),
                [2.5625, 0.2, 0.2216, 2.0, -0.08125],
            ),
        )
        periodic_expected = (
            (Flat(OnGrid()), [-0.4, 0.2, 0.31472, 0.2, 1.1]),
            (Flat(OnCell()), [2.0, 0.2, 0.5116842105263156, 2.0, 0.2]),
            (Periodic(OnGrid()), [-0.4, 0.2, 0.2216, 0.2, 1.1]),
            (Periodic(OnCell()), [2.0, 0.2, 0.2216, 2.0, 0.2]),
        )
        for boundary, expected in line_expected:
            result = CubicSplineInterpolation(
                source,
                values,
                bc=boundary,
                extrapolation_bc=Linear(),
            )(queries)
            np.testing.assert_allclose(result, expected, rtol=0, atol=8e-15)
        for boundary, expected in periodic_expected:
            result = CubicSplineInterpolation(
                source,
                values,
                bc=boundary,
                extrapolation_bc=Periodic(),
            )(queries)
            np.testing.assert_allclose(result, expected, rtol=0, atol=8e-15)

    def test_float16_cubic_rounding_matches_locked_julia_bits(self) -> None:
        source = LatticeAxis(
            np.arange(1, 5, dtype=np.float16), step_hint=np.float16(1)
        )
        values = np.array([1, 4, 2, 8], dtype=np.float16)
        targets = np.array(
            [0, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5],
            dtype=np.float16,
        )
        expected = {
            None: [
                0xC3BC, 0xBDBD, 0x3BFE, 0x405F, 0x4266, 0x43B9, 0x4400,
                0x418C, 0x3FFE, 0x4413, 0x4800, 0x4A1E, 0x4C1E,
            ],
            "flat": [
                0x3BFA, 0x3BFC, 0x3BFE, 0x3DDF, 0x40FF, 0x430E, 0x43FE,
                0x413F, 0x3FFD, 0x455F, 0x47FF, 0x47FF, 0x47FF,
            ],
        }
        for name, boundary in (
            (None, None),
            ("flat", Flat(OnGrid())),
        ):
            keywords = {} if boundary is None else {"bc": boundary}
            result = CubicSplineInterpolation(
                source,
                values,
                extrapolation_bc=Linear(),
                **keywords,
            )(targets)
            np.testing.assert_array_equal(
                result.view(np.uint16), np.asarray(expected[name], dtype=np.uint16)
            )

        periodic = CubicSplineInterpolation(
            source,
            values,
            bc=Periodic(OnCell()),
            extrapolation_bc=Linear(),
        )(targets)
        self.assertEqual(periodic.dtype, np.dtype(np.float64))
        # Periodic(OnCell()) introduces Float64 half-cell bounds in the
        # locked dependency, so even Float16 coefficients evaluate as
        # Float64 on this path.
        np.testing.assert_array_equal(
            periodic.view(np.uint64),
            np.asarray(
                [
                    0x4023204000000000,
                    0x40131FDFFFFFFFFF,
                    0x3FEFF55555555552,
                    0x3FF025EFFFFFFFFF,
                    0x4000386AAAAAAAAB,
                    0x400A3E7D55555554,
                    0x400FF5FFFFFFFFFF,
                    0x4005BA4000000000,
                    0x3FFFFFFFFFFFFFFE,
                    0x4015E0CAAAAAAAAA,
                    0x4020002AAAAAAAAB,
                    0x40131FDFFFFFFFFF,
                    0xBF48000000002000,
                ],
                dtype=np.uint64,
            ),
        )

    def test_integer_coefficients_and_nan_fill_types(self) -> None:
        source = np.arange(4, dtype=np.float16)
        values = np.array([1, 4, 2, 8], dtype=np.int16)
        for constructor in (CubicSplineInterpolation, LinearInterpolation):
            result = constructor(source, values)(
                np.array([0.5], dtype=np.float16)
            )
            self.assertEqual(result.dtype, np.dtype(np.float64))

        coordinates = np.arange(3.0)
        plane = coordinates[:, None] + coordinates[None, :]
        result = CubicSplineInterpolation(
            (coordinates, coordinates), plane, extrapolation_bc=np.nan
        )[(np.array([-1.0, 1.0]), np.array([1.0, 4.0]))]
        np.testing.assert_array_equal(
            np.isnan(result), [[True, True], [False, True]]
        )
        self.assertEqual(result[1, 0], 2.0)

    def test_boundary_constructor_and_signed_lattice_geometry(self) -> None:
        nested = Linear(Periodic())
        self.assertIsInstance(nested.bc, Periodic)
        with self.assertRaises(TypeError):
            Linear(Throw())

        descending = range(4, 0, -1)
        expected_down = np.array([3.5, 1.5])
        expected_up = np.array(
            [4.25, 3.75, 3.25, 2.75, 2.25, 1.75, 1.25, 0.75]
        )
        np.testing.assert_array_equal(downsample(descending, 2), expected_down)
        np.testing.assert_array_equal(
            downsample((descending,), 2)[0], expected_down
        )
        np.testing.assert_array_equal(upsample(descending, 2), expected_up)
        np.testing.assert_array_equal(
            upsample((descending,), 2)[0], expected_up
        )

        field = LF[Generic](np.arange(1.0, 5.0), (range(1, 5),), 1.0)
        reversed_field = downsample(field, (descending,))
        np.testing.assert_allclose(reversed_field.data, [4, 3, 2, 1])
        np.testing.assert_array_equal(reversed_field.L[0], np.asarray(descending))

    def test_positive_noninteger_factors_do_not_truncate(self) -> None:
        source = (range(4),)
        array = np.arange(4.0)
        for factor in (2.0, 2.9, (2.0,)):
            for operation, value in (
                (downsample, source),
                (upsample, source),
                (coarsen, array),
            ):
                with self.assertRaises(TypeError):
                    operation(value, factor)

        for factor in (True, np.bool_(True), np.int32(2), np.uint64(2)):
            for operation, value in (
                (downsample, source),
                (upsample, source),
                (coarsen, array),
            ):
                with self.assertRaises(TypeError):
                    operation(value, factor)

        np.testing.assert_array_equal(
            downsample(range(4), np.int64(2)),
            downsample(range(4), 2),
        )

    def test_python_integer_factors_must_fit_julia_int64(self) -> None:
        for factor in (2**63, -(2**63) - 1):
            with self.assertRaises(TypeError):
                downsample(range(4), factor)
            with self.assertRaises(TypeError):
                upsample(range(4), factor)
            with self.assertRaises(TypeError):
                coarsen(np.arange(4.0), factor)

        with self.assertRaises(TypeError):
            coarsen(np.zeros((2, 2)), (1, 2**63))

    def test_empty_coarsen_retains_julia_inferred_reducer_dtype(self) -> None:
        # Locked Julia goldens:
        #   coarsen(Bool[], 2)      :: Vector{Float64}
        #   coarsen(Float32[], 2)   :: Vector{Float32}
        #   coarsen(ComplexF32[],2) :: Vector{ComplexF32}
        #   coarsen(Int8[], 2)      :: Vector{Float64}
        for dtype, expected in (
            (np.bool_, np.float64),
            (np.int8, np.float64),
            (np.uint8, np.float64),
            (np.float16, np.float16),
            (np.float32, np.float32),
            (np.float64, np.float64),
            (np.complex64, np.complex64),
            (np.complex128, np.complex128),
        ):
            result = coarsen(np.empty(0, dtype=dtype), 2)
            self.assertEqual(result.shape, (0,))
            self.assertEqual(result.dtype, np.dtype(expected))

        matrix = coarsen(np.empty((0, 4), dtype=np.float32), (2, 2))
        self.assertEqual(matrix.shape, (0, 2))
        self.assertEqual(matrix.dtype, np.dtype(np.float32))

        for reducer, dtype, expected in (
            (np.max, np.float32, np.float32),
            (np.amax, np.complex64, np.complex64),
            (np.min, np.float16, np.float16),
            (np.amin, np.uint8, np.uint8),
            (np.mean, np.int8, np.float64),
            (np.sum, np.int8, np.int64),
            (np.sum, np.uint8, np.uint64),
        ):
            result = coarsen(
                np.empty(0, dtype=dtype), 2, reducer=reducer
            )
            self.assertEqual(result.dtype, np.dtype(expected))

    def test_empty_coarsen_does_not_execute_custom_reducer(self) -> None:
        calls: list[tuple[int, ...]] = []

        class Reducer:
            def __call__(self, block: np.ndarray) -> np.complex64:
                calls.append(block.shape)
                return np.complex64(1 + 2j)

        result = coarsen(
            np.empty(0, dtype=np.float32),
            2,
            reducer=Reducer(),
        )
        self.assertEqual(calls, [])
        self.assertEqual(result.dtype, np.dtype(object))

        unknown_result = coarsen(
            np.empty((4, 0), dtype=np.float32),
            (2, 2),
            reducer=Reducer(),
        )
        self.assertEqual(calls, [])
        self.assertEqual(unknown_result.shape, (2, 0))
        self.assertEqual(unknown_result.dtype, np.dtype(object))

    def test_rational_interpolation_is_exact(self) -> None:
        source = np.array([Fraction(index) for index in range(4)], dtype=object)
        values = np.array(
            [Fraction(value) for value in (1, 4, 2, 8)], dtype=object
        )
        query = Fraction(3, 2)

        self.assertEqual(LinearInterpolation(source, values)(query), Fraction(3))
        self.assertEqual(
            CubicSplineInterpolation(source, values)(query), Fraction(111, 40)
        )

        targets = np.array(
            [
                Fraction(-1),
                Fraction(0),
                Fraction(1, 4),
                Fraction(1, 2),
                Fraction(3, 2),
                Fraction(11, 4),
                Fraction(3),
                Fraction(4),
            ],
            dtype=object,
        )
        expected = {
            None: [
                Fraction(-58, 15), Fraction(1), Fraction(35, 16),
                Fraction(16, 5), Fraction(111, 40), Fraction(379, 64),
                Fraction(8), Fraction(247, 15),
            ],
            "flat": [
                Fraction(1), Fraction(1), Fraction(47, 32), Fraction(5, 2),
                Fraction(21, 8), Fraction(461, 64), Fraction(8), Fraction(8),
            ],
        }
        for name, boundary in (
            (None, None),
            ("flat", Flat(OnGrid())),
        ):
            keywords = {} if boundary is None else {"bc": boundary}
            result = CubicSplineInterpolation(
                source,
                values,
                extrapolation_bc=Linear(),
                **keywords,
            )(targets)
            self.assertEqual(result.dtype, np.dtype(object))
            self.assertEqual(list(result), expected[name])

        periodic = CubicSplineInterpolation(
            source,
            values,
            bc=Periodic(OnCell()),
            extrapolation_bc=Linear(),
        )(targets)
        self.assertEqual(periodic.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(
            periodic,
            [
                9.5625,
                1.0,
                1.01171875,
                2.03125,
                2.71875,
                7.30859375,
                8.0,
                0.0,
            ],
            rtol=0,
            atol=2e-15,
        )

        filled = CubicSplineInterpolation(
            source, values, extrapolation_bc=Fraction(-1, 3)
        )(np.array([Fraction(-1), Fraction(1, 2)], dtype=object))
        self.assertEqual(list(filled), [Fraction(-1, 3), Fraction(16, 5)])

    def test_decimal_bigfloat_like_interpolation_retains_precision(self) -> None:
        with localcontext() as context:
            context.prec = 80
            source = np.array(
                [Decimal("0.1") + index * Decimal("0.2") for index in range(4)],
                dtype=object,
            )
            values = np.array(
                [Decimal(value) for value in (1, 4, 2, 8)], dtype=object
            )
            query = Decimal("0.4")

            linear = LinearInterpolation(source, values)(query)
            cubic = CubicSplineInterpolation(source, values)(query)
            self.assertIsInstance(linear, Decimal)
            self.assertIsInstance(cubic, Decimal)
            self.assertEqual(linear, Decimal("3.0"))
            self.assertLess(abs(cubic - Decimal("2.775")), Decimal("1e-75"))

            integer_axis_cubic = CubicSplineInterpolation(
                range(4), values
            )(Decimal("1.5"))
            self.assertIsInstance(integer_axis_cubic, Decimal)
            self.assertLess(
                abs(integer_axis_cubic - Decimal("2.775")), Decimal("1e-75")
            )

    def test_nonpositive_factors_and_singleton_sources_are_explicit(self) -> None:
        source = LatticeAxis(np.arange(4.0), step_hint=1.0)
        for factor in (0, -1):
            with self.assertRaises(DomainError):
                downsample(source, factor)
            with self.assertRaises(DomainError):
                coarsen(np.arange(4.0), factor)
        zero_factor = upsample(source, 0)
        self.assertEqual(len(zero_factor), 0)
        self.assertEqual(zero_factor._range_kind, "srl")
        self.assertTrue(np.isinf(zero_factor._logical_ref))
        self.assertTrue(np.isinf(zero_factor._logical_step))
        negative_factor = upsample(source, -1)
        self.assertEqual(len(negative_factor), 0)
        self.assertEqual(negative_factor._range_kind, "srl")
        self.assertEqual(negative_factor._logical_ref, -1.0)
        self.assertEqual(negative_factor._logical_step, -1.0)

        singleton = LatticeAxis(np.array([2.0]), step_hint=0.5)
        for constructor in (CubicSplineInterpolation, LinearInterpolation):
            with self.assertRaises(ValueError):
                constructor(singleton, np.array([7.0]))


if __name__ == "__main__":
    unittest.main()
