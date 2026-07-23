"""Dense, convolutional, and separable optimal-transport tests."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import unittest
import warnings

import numpy as np

import slmtools as slm
from slmtools.ot import _sinkhorn_gibbs


class CostMapAndIntegrationTests(unittest.TestCase):
    def test_cost_matrices_use_julia_cartesian_order(self) -> None:
        cost = slm.getCostMatrix((np.asarray([0.0, 2.0]),))
        np.testing.assert_array_equal(cost, [[0.0, 1.0], [1.0, 0.0]])

        source = (np.asarray([0.0, 1.0]), np.asarray([10.0, 20.0]))
        target = (np.asarray([-1.0, 2.0]), np.asarray([3.0, 8.0]))
        raw = slm.getCostMatrix(source, target, normalization=None)
        source_points = np.asarray([[0, 10], [1, 10], [0, 20], [1, 20]])
        target_points = np.asarray([[-1, 3], [2, 3], [-1, 8], [2, 8]])
        expected = np.sum(
            (source_points[:, None, :] - target_points[None, :, :]) ** 2,
            axis=-1,
        )
        np.testing.assert_array_equal(raw, expected)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = slm.pdCostMatrix(source, target, 2.0, 1.0, normalization=None)
        self.assertFalse(
            any(issubclass(item.category, DeprecationWarning) for item in caught)
        )
        np.testing.assert_array_equal(legacy, expected)

    def test_cost_normalization_accepts_julia_broadcast_arrays(self) -> None:
        lattice = (np.arange(3.0),)
        raw_expected = np.asarray(
            [[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]]
        )
        raw = slm.getCostMatrix(
            lattice,
            normalization=lambda matrix: np.ones_like(matrix),
        )
        np.testing.assert_array_equal(raw, raw_expected)

        self_normalized = slm.getCostMatrix(
            lattice, normalization=lambda matrix: matrix
        )
        self.assertTrue(np.all(np.isnan(np.diag(self_normalized))))
        np.testing.assert_array_equal(
            self_normalized[~np.eye(3, dtype=bool)], np.ones(6)
        )

        row_scaled = slm.getCostMatrix(
            lattice,
            normalization=lambda _matrix: np.asarray([1.0, 2.0, 4.0]),
        )
        np.testing.assert_array_equal(
            row_scaled,
            np.asarray(
                [[0.0, 1.0, 4.0], [0.5, 0.0, 0.5], [1.0, 0.25, 0.0]]
            ),
        )

        legacy = slm.pdCostMatrix(
            lattice,
            lattice,
            2.0,
            1.0,
            normalization=lambda matrix: np.ones_like(matrix),
        )
        np.testing.assert_array_equal(legacy, raw_expected)

    def test_mapify_identity_and_julia_zero_mass_safe_inverse(self) -> None:
        lattice = (np.asarray([0.0, 1.0]), np.asarray([10.0, 20.0]))
        plan = np.eye(4)
        vector_map = slm.mapify(plan, lattice, lattice)
        np.testing.assert_array_equal(vector_map[..., 0], [[0, 0], [1, 1]])
        np.testing.assert_array_equal(vector_map[..., 1], [[10, 20], [10, 20]])
        plan[2] = 0
        vector_map = slm.mapify(plan, lattice, lattice)
        # Julia multiplies by ``safeInverse(0) == 0``; it does not divide
        # the already-assigned Float64 barycenter by zero.
        np.testing.assert_array_equal(vector_map[0, 1], [0.0, 0.0])

    def test_heterogeneous_coordinate_dimensions_promote_only_per_term(self) -> None:
        source = (
            slm.LatticeAxis(
                np.asarray([10000], dtype=np.float32),
                step_hint=np.float32(1),
            ),
            slm.LatticeAxis(
                np.asarray([1], dtype=np.int64), step_hint=np.int64(1)
            ),
        )
        target = (
            slm.LatticeAxis(
                np.asarray([0], dtype=np.float32), step_hint=np.float32(1)
            ),
            slm.LatticeAxis(
                np.asarray([0], dtype=np.int64), step_hint=np.int64(1)
            ),
        )
        cost = slm.getCostMatrix(
            source, target, normalization=lambda _matrix: 1
        )
        self.assertEqual(cost.dtype, np.dtype(np.float32))
        self.assertEqual(cost[0, 0], np.float32(1e8))
        pd_cost = slm.pdCostMatrix(
            source,
            target,
            np.float32(2),
            np.float32(1),
            flambda=np.float32(1),
            normalization=lambda _matrix: 1,
        )
        self.assertEqual(pd_cost.dtype, np.dtype(np.float32))
        self.assertEqual(pd_cost[0, 0], np.float32(1e8))

        map_source = (
            slm.LatticeAxis(
                np.asarray([0], dtype=np.float32), step_hint=np.float32(1)
            ),
            slm.LatticeAxis(
                np.asarray([0], dtype=np.int64), step_hint=np.int64(1)
            ),
        )
        map_target = (
            slm.LatticeAxis(
                np.asarray([0, 1], dtype=np.float32),
                step_hint=np.float32(1),
            ),
            slm.LatticeAxis(
                np.asarray([0], dtype=np.int64), step_hint=np.int64(1)
            ),
        )
        plan = np.asarray(
            [[Fraction(1, 3), Fraction(2, 3)]], dtype=object
        )
        mapped = slm.mapify(plan, map_source, map_target)
        self.assertEqual(mapped.dtype, np.dtype(np.float64))
        self.assertEqual(mapped[0, 0, 0], float(np.float32(2 / 3)))
        self.assertEqual(mapped[0, 0, 1], 0.0)

    def test_mapify_checks_complex_to_float_assignment(self) -> None:
        lattice = (np.asarray([-1.0, 0.0]),)
        exactly_real = np.eye(2, dtype=np.complex128)
        result = slm.mapify(exactly_real, lattice, lattice)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(result[..., 0], lattice[0])

        complex_barycenter = np.asarray(
            [[1.0 + 1.0j, 1.0], [0.0, 1.0]], dtype=np.complex128
        )
        with self.assertRaisesRegex(ValueError, "complex values"):
            slm.mapify(complex_barycenter, lattice, lattice)

    def test_mapify_safe_inverse_keeps_float16_rounding_before_assignment(self) -> None:
        source = (
            slm.LatticeAxis(
                np.asarray([0], dtype=np.float16), step_hint=np.float16(1)
            ),
        )
        target = (
            slm.LatticeAxis(
                np.asarray([0.1, 0.3], dtype=np.float16),
                step_hint=np.float16(0.2),
            ),
        )
        result = slm.mapify(
            np.asarray([[1, 2]], dtype=np.float16), source, target
        )
        self.assertEqual(result.dtype, np.dtype(np.float64))
        # Locked Julia: weighted coordinate 0x399a, inverse mass 0x3555,
        # product 0x3378, then assignment to Float64.
        self.assertEqual(
            result.view(np.uint64)[0, 0], np.uint64(0x3FCDE00000000000)
        )

    def test_mapify_uses_julia_matrix_product_reduction_order(self) -> None:
        plan = np.reshape(
            np.arange(1, 22, dtype=np.float32) / np.float32(10),
            (3, 7),
            order="F",
        )
        source = (
            slm.LatticeAxis(
                np.arange(3, dtype=np.float32), step_hint=np.float32(1)
            ),
        )
        target = (
            slm.LatticeAxis(
                np.arange(-3, 4, dtype=np.float32) / np.float32(7),
                step_hint=np.float32(1 / 7),
            ),
        )
        result = slm.mapify(plan, source, target)[..., 0]
        # A flattened elementwise multiply followed by np.sum differs in the
        # final component. Julia evaluates ``x * Lv[i]`` as matrix-vector
        # multiplication before applying safeInverse.
        np.testing.assert_array_equal(
            result.view(np.uint64),
            np.asarray(
                [
                    0x3FC5F15F40000000,
                    0x3FC3F2B3A0000000,
                    0x3FC24924A0000000,
                ],
                dtype=np.uint64,
            ),
        )

    def test_hyper_sums_and_julia_default_anchor(self) -> None:
        lattice = (np.arange(-2.0, 2.0), np.arange(-3.0, 2.0))
        vector_field = np.empty((4, 5, 2))
        vector_field[..., 0] = 1.0
        vector_field[..., 1] = 2.0
        potential = slm.scalarPotentialN(vector_field, lattice)
        # Julia chooses one-based length÷2, hence zero-based anchors (1, 1).
        expected = (
            lattice[0][:, None] - lattice[0][1]
            + 2 * (lattice[1][None, :] - lattice[1][1])
        )
        np.testing.assert_allclose(potential, expected, atol=1e-15)
        np.testing.assert_array_equal(
            slm.scalarPotentialN(
                vector_field, lattice, dimOrder=(1, 2, 99)
            ),
            potential,
        )
        with self.assertRaises(IndexError):
            slm.scalarPotentialN(vector_field, lattice, dimOrder=(1,))

        singleton_lattice = (
            slm.LatticeAxis([0.0], step_hint=1.0),
        )
        singleton_vector = np.ones((1, 1))
        with self.assertRaisesRegex(IndexError, "outside"):
            slm.scalarPotentialN(singleton_vector, singleton_lattice)
        np.testing.assert_array_equal(
            slm.scalarPotentialN(
                singleton_vector, singleton_lattice, idx=(0,)
            ),
            [0.0],
        )

        line = np.arange(5.0)
        np.testing.assert_allclose(
            slm.hyperSum(line, (2,), 1, ()), [-3, -2, 0, 3, 7]
        )
        np.testing.assert_allclose(
            slm.hyperSum2(np.ones(5), (2,), 1, ()), [-2, -1, 0, 1, 2]
        )
        grid = np.arange(1.0, 7.0).reshape((2, 3), order="F")
        np.testing.assert_array_equal(
            slm.hyperSum2(grid, (0, 1), 1, (2, 2)),
            np.asarray([[0.0, 0.0, 0.0], [1.5, 3.5, 5.5]]),
        )
        np.testing.assert_array_equal(
            slm.hyperSum2(grid, (0, 1, 99), 1, (2, 2)),
            slm.hyperSum2(grid, (0, 1), 1, (2, 2)),
        )

    def test_distribution_normalization_preserves_julia_nan_degeneracy(self) -> None:
        np.testing.assert_allclose(
            slm.normalizeDistribution([-1.0, 3.0]), [0.25, 0.75]
        )
        self.assertTrue(np.all(np.isnan(slm.normalizeDistribution(np.zeros(3)))))

        singleton_cost = slm.getCostMatrix((slm.LatticeAxis([0.0], step_hint=1.0),))
        self.assertEqual(singleton_cost.shape, (1, 1))
        self.assertTrue(np.isnan(singleton_cost[0, 0]))

    def test_integration_indices_reject_lossy_integer_coercion(self) -> None:
        vector = np.arange(4.0)
        lattice = (np.arange(4.0),)
        field = np.ones((4, 1))

        with self.assertRaises(TypeError):
            slm.hyperSum(vector, (1,), 1.5, ())
        with self.assertRaises(TypeError):
            slm.hyperSum2(vector, (1.5,), 1, ())
        with self.assertRaises(TypeError):
            slm.hyperSum2(vector, (1,), 1, (1.5,))
        with self.assertRaises(TypeError):
            slm.scalarPotentialN(field, lattice, idx=(1.5,))
        with self.assertRaises(TypeError):
            slm.scalarPotentialN(field, lattice, dimOrder=(1.5,))

    def test_exact_number_cost_normalization_and_potential(self) -> None:
        rational_axis = slm.LatticeAxis(
            np.array([Fraction(0), Fraction(1), Fraction(2)], dtype=object),
            step_hint=Fraction(1),
        )
        rational_field = np.array(
            [Fraction(1), Fraction(2), Fraction(3)], dtype=object
        ).reshape(3, 1)
        rational_cost = slm.getCostMatrix((rational_axis,))
        self.assertEqual(rational_cost.dtype, np.dtype(object))
        self.assertEqual(rational_cost[0, 1], Fraction(1, 4))
        self.assertEqual(
            slm.normalizeDistribution(rational_field[:, 0]).tolist(),
            [Fraction(1, 6), Fraction(1, 3), Fraction(1, 2)],
        )
        rational_potential = slm.scalarPotentialN(
            rational_field, (rational_axis,), idx=(1,)
        )
        self.assertEqual(
            rational_potential.tolist(),
            [Fraction(-3, 2), Fraction(0), Fraction(5, 2)],
        )
        rational_map = slm.mapify(
            np.eye(3, dtype=object), (rational_axis,), (rational_axis,)
        )
        self.assertEqual(rational_map.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(rational_map[..., 0], [0.0, 1.0, 2.0])
        third_map = slm.mapify(
            np.asarray([[Fraction(2, 3), Fraction(1, 3)]], dtype=object),
            (slm.LatticeAxis([0], step_hint=1),),
            (slm.LatticeAxis([0, 1], step_hint=1),),
        )
        self.assertEqual(third_map.dtype, np.dtype(np.float64))
        self.assertEqual(third_map[0, 0], 1 / 3)
        rational_pd_cost = slm.pdCostMatrix(
            (rational_axis,),
            (rational_axis,),
            Fraction(2),
            Fraction(1),
            flambda=Fraction(1),
        )
        self.assertEqual(rational_pd_cost.dtype, np.dtype(object))
        self.assertEqual(rational_pd_cost[0, 1], Fraction(1, 4))

        decimal_axis = slm.LatticeAxis(
            np.array([Decimal(0), Decimal(1), Decimal(2)], dtype=object),
            step_hint=Decimal(1),
        )
        decimal_field = np.array(
            [Decimal(1), Decimal(2), Decimal(3)], dtype=object
        ).reshape(3, 1)
        decimal_potential = slm.scalarPotentialN(
            decimal_field, (decimal_axis,), idx=(1,)
        )
        self.assertEqual(
            decimal_potential.tolist(),
            [Decimal("-1.5"), Decimal(0), Decimal("2.5")],
        )

        float_axis = np.asarray([0.0, 1.0, 2.0])
        mixed_cost = slm.getCostMatrix(
            (decimal_axis,), (float_axis,), normalization=None
        )
        self.assertEqual(mixed_cost.dtype, np.dtype(object))
        self.assertEqual(mixed_cost[0, 2], Decimal(4))

        mixed_map = slm.mapify(
            np.eye(3), (float_axis,), (decimal_axis,)
        )
        self.assertEqual(mixed_map.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(mixed_map[..., 0], float_axis)

        mixed_potential = slm.scalarPotentialN(
            np.asarray([[1.0], [2.0], [3.0]]),
            (decimal_axis,),
            idx=(1,),
        )
        self.assertEqual(
            mixed_potential.tolist(),
            [Decimal("-1.5"), Decimal(0), Decimal("2.5")],
        )

        mixed_pd_cost = slm.pdCostMatrix(
            (decimal_axis,),
            (float_axis,),
            2.0,
            1.0,
            normalization=None,
        )
        self.assertEqual(mixed_pd_cost.dtype, np.dtype(object))
        self.assertEqual(mixed_pd_cost[0, 2], Decimal(4))

    def test_empty_cost_and_map_domains_follow_julia(self) -> None:
        empty = (range(0),)
        with self.assertRaises(ValueError):
            slm.getCostMatrix(empty)
        custom = slm.getCostMatrix(
            empty, normalization=lambda matrix: matrix
        )
        self.assertEqual(custom.shape, (0, 0))

        source_empty = slm.mapify(
            np.zeros((0, 2)), empty, (range(2),)
        )
        self.assertEqual(source_empty.shape, (0, 1))
        target_empty = slm.mapify(
            np.zeros((2, 0)), (range(2),), empty
        )
        np.testing.assert_array_equal(
            target_empty, np.zeros((2, 1))
        )

    def test_decimal_dense_ot_preserves_bigfloat_like_domain(self) -> None:
        axis = slm.LatticeAxis(
            np.asarray(
                [Decimal(-1), Decimal(0), Decimal(1)],
                dtype=object,
            ),
            step_hint=Decimal(1),
        )
        source = slm.LF[slm.Intensity](
            np.asarray(
                [Decimal(1), Decimal(2), Decimal(1)],
                dtype=object,
            ),
            (axis,),
            Decimal("1.25"),
        )
        target = slm.LF[slm.Intensity](
            np.asarray(
                [Decimal(1), Decimal(1), Decimal(2)],
                dtype=object,
            ),
            (axis,),
            Decimal("1.25"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = slm.otPhase(
                source, target, 0.5, maxiter=20
            )
        self.assertEqual(result.dtype, np.dtype(object))
        self.assertTrue(
            all(isinstance(value, Decimal) for value in result.data)
        )
        expected = (
            Decimal(0),
            Decimal(
                "-0.02262384686291962765380958444438874721527099609375"
            ),
            Decimal(
                "0.39999999999993303134715461055748164653778076171875"
            ),
        )
        for actual, golden in zip(
            result.data, expected, strict=True
        ):
            self.assertLessEqual(
                abs(actual - golden), Decimal("3e-16")
            )


class DenseSinkhornTests(unittest.TestCase):
    def test_one_iteration_matches_optimaltransport_0320_update_order(self) -> None:
        source = np.asarray([0.4, 0.6])
        target = np.asarray([0.5, 0.5])
        cost = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        epsilon = 0.5
        kernel = np.exp(-cost / epsilon)
        u = source / (kernel @ np.ones(2))
        v = target / (kernel.T @ u)
        expected = kernel * u[:, None] * v[None, :]
        with self.assertWarns(RuntimeWarning):
            result = _sinkhorn_gibbs(
                source,
                target,
                cost,
                epsilon,
                maxiter=1,
                check_convergence=1,
            )
        np.testing.assert_allclose(result, expected, rtol=0, atol=0)

    def test_sinkhorn_plan_marginals_and_dense_phase(self) -> None:
        source = np.asarray([0.1, 0.2, 0.3, 0.4])
        target = np.asarray([0.4, 0.3, 0.2, 0.1])
        coordinates = np.arange(4.0)
        cost = (coordinates[:, None] - coordinates[None, :]) ** 2
        plan = _sinkhorn_gibbs(source, target, cost, 0.3)
        np.testing.assert_allclose(np.sum(plan, axis=1), source, atol=2e-10)
        np.testing.assert_allclose(np.sum(plan, axis=0), target, atol=2e-15)

        lattice = slm.natlat((4, 4))
        x, y = np.meshgrid(*lattice, indexing="ij")
        U = slm.LF[slm.Intensity](np.exp(-(x**2 + y**2)), lattice)
        V = slm.LF[slm.Intensity](
            np.exp(-((x - 0.2) ** 2 + (y + 0.1) ** 2)), lattice
        )
        phase = slm.otPhase(U, V, 0.1, maxiter=100)
        self.assertIs(phase.field_type, slm.RealPhase)
        self.assertTrue(np.all(np.isfinite(phase.data)))

    def test_pdot_phase_and_resampled_beam_metadata(self) -> None:
        lattice = slm.natlat((5,))
        x = np.asarray(lattice[0])
        root = slm.LF[slm.Intensity](np.exp(-(x - 0.1) ** 2), lattice, 1.3)
        target = slm.LF[slm.Intensity](np.exp(-1.4 * (x + 0.1) ** 2), lattice, 1.3)
        phase = slm.pdotPhase(
            root, target, 0.5, 0.1, (0.03,), (-0.02,), 0.15, maxiter=100
        )
        self.assertIs(phase.field_type, slm.RealPhase)
        self.assertTrue(np.all(np.isfinite(phase.data)))

        fine = (np.linspace(x[0] - 0.1, x[-1] + 0.1, 7),)
        with self.assertRaisesRegex(
            NotImplementedError, "non-integer target ranges"
        ):
            slm.pdotBeamEstimate(
                root,
                target,
                0.5,
                0.1,
                (0.03,),
                (-0.02,),
                0.15,
                LFine=fine,
                maxiter=100,
            )
        beam = slm.pdotBeamEstimate(
            root,
            target,
            0.5,
            0.1,
            (0.03,),
            (-0.02,),
            0.15,
            maxiter=100,
        )
        self.assertEqual(beam.shape, (5,))
        self.assertIs(beam.field_type, slm.ComplexAmp)
        self.assertTrue(np.all(np.isfinite(beam.data)))
        np.testing.assert_allclose(
            beam.L[0], slm.dualShiftLattice(root.L, root.flambda)[0]
        )

    def test_pdot_beam_integer_lfine_matches_working_julia_overload(self) -> None:
        lattice = (range(1, 4),)
        root = slm.LF[slm.Intensity](
            np.asarray([1.0, 2.0, 3.0]), lattice
        )
        target = slm.LF[slm.Intensity](
            np.asarray([3.0, 2.0, 1.0]), lattice
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = slm.pdotBeamEstimate(
                root,
                target,
                0.5,
                0.1,
                [0.0],
                [0.0],
                0.5,
                LFine=(range(1, 4),),
                maxiter=10,
            )
        expected = np.asarray(
            [
                0.8296264510647069 + 0.45616814629549507j,
                -0.3923927830054344 + 0.6335154170900323j,
                -0.6937041984358028 + 0.2590196502702916j,
            ]
        )
        np.testing.assert_allclose(
            result.data, expected, rtol=2e-15, atol=2e-15
        )

    def test_pdot_beam_keeps_float16_and_float32_root_arithmetic(self) -> None:
        expected = {
            np.float16: np.asarray(
                [
                    0.11615153187482093 - 0.16818070629211693j,
                    0.4262664687722065 - 0.3182359104498057j,
                    0.004855671347014423 + 0.20455107073629336j,
                ]
            ),
            np.float32: np.asarray(
                [
                    0.11686554642613532 - 0.1677893799797453j,
                    0.42639440489083713 - 0.3180854801793792j,
                    0.004030524215494241 + 0.20457031660248434j,
                ]
            ),
        }
        for dtype, oracle in expected.items():
            lattice = (
                slm.LatticeAxis(
                    np.asarray([-0.3, -0.1, 0.1], dtype=dtype),
                    step_hint=dtype(0.2),
                ),
            )
            root = slm.LF[slm.Intensity, dtype, 1](
                np.asarray([0.04, 0.25, 0.81], dtype=dtype),
                lattice,
                dtype(1.25),
            )
            target = slm.LF[slm.Intensity, dtype, 1](
                np.asarray([0.09, 0.36, 0.64], dtype=dtype),
                lattice,
                dtype(1.25),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = slm.pdotBeamEstimate(
                    root,
                    target,
                    dtype(0.4),
                    dtype(0.1),
                    (dtype(0.03),),
                    (dtype(-0.02),),
                    dtype(0.2),
                    maxiter=20,
                )
            self.assertEqual(result.dtype, np.dtype(np.complex128))
            self.assertIsInstance(result.flambda, dtype)
            self.assertEqual(result.L[0].dtype, np.dtype(dtype))
            np.testing.assert_allclose(result.data, oracle, rtol=2e-15, atol=2e-15)


class ConvolutionalAndSeparableTests(unittest.TestCase):
    def test_convolutional_helpers_shapes_loss_and_deprecation(self) -> None:
        lattice = slm.natlat((5, 4))
        x, y = np.meshgrid(*lattice, indexing="ij")
        source = np.exp(-(x**2 + y**2))
        target = np.exp(-((x - 0.1) ** 2 + (y + 0.2) ** 2))
        source /= np.sum(source)
        target /= np.sum(target)
        u, v, loss = slm.SinkhornConvN(source, target, 0.2, 4, every=2)
        self.assertEqual(len(loss), 2)
        self.assertEqual(u.shape, source.shape)
        self.assertEqual(v.shape, source.shape)
        self.assertTrue(np.all(np.isfinite(u)))
        gradient = slm.dualToGradients(u, v, source, lattice, 0.2)
        self.assertEqual(gradient.shape, source.shape + (2,))

        U = slm.LF[slm.Intensity](source, lattice)
        V = slm.LF[slm.Intensity](target, lattice)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            phase, quick_loss = slm.otQuickPhase(
                U, V, 0.2, 2, return_loss=True
            )
        self.assertFalse(
            any(issubclass(item.category, DeprecationWarning) for item in caught)
        )
        self.assertIs(phase.field_type, slm.RealPhase)
        self.assertEqual(len(quick_loss), 2)

    def test_dual_to_gradients_retains_float32_scale_arithmetic(self) -> None:
        u = np.asarray([0.1, 0.2, 0.7], dtype=np.float32)
        v = np.asarray([0.3, 0.4, 0.3], dtype=np.float32)
        source = np.asarray([0.2, 0.3, 0.5], dtype=np.float32)
        lattice = (
            slm.LatticeAxis.from_start_step(
                np.float32(-1), np.float32(1), 3
            ),
        )
        gradient = slm.dualToGradients(
            u, v, source, lattice, np.float32(0.2)
        )
        self.assertEqual(gradient.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(
            gradient[..., 0],
            np.asarray(
                [
                    -0.14898930520074016,
                    0.0,
                    0.41717004745770875,
                ]
            ),
            rtol=2e-15,
            atol=2e-15,
        )
        with self.assertRaisesRegex(TypeError, "same Julia element type"):
            slm.dualToGradients(
                u, v.astype(np.float64), source, lattice, np.float32(0.2)
            )

    def test_otphase2_retains_rectangular_and_wavelength_defects(self) -> None:
        lattice = slm.natlat((5, 7))
        x, y = np.meshgrid(*lattice, indexing="ij")
        source = slm.LF[slm.Intensity](np.exp(-(x**2 + y**2)), lattice, 1.2)
        shifted_lattice = tuple(np.asarray(axis) + 0.03 for axis in lattice)
        target = slm.LF[slm.Intensity](
            np.exp(-((x - 0.15) ** 2 + (y + 0.1) ** 2)),
            shifted_lattice,
            1.2,
        )
        with self.assertRaisesRegex(NotImplementedError, "rectangular"):
            slm.otPhase2(
                source, target, 0.08, 5, return_loss=True, rtol=0.0
            )

        square_lattice = slm.natlat((5, 5))
        sx, sy = np.meshgrid(*square_lattice, indexing="ij")
        square_source = slm.LF[slm.Intensity](
            np.exp(-(sx**2 + sy**2)), square_lattice, 1.2
        )
        square_target = slm.LF[slm.Intensity](
            np.exp(-((sx - 0.15) ** 2 + (sy + 0.1) ** 2)),
            square_lattice,
            1.2,
        )
        phase = slm.otPhase2(
            square_source,
            square_target,
            0.08,
            5,
            return_loss=True,
            rtol=0.0,
        )
        baseline = slm.otPhase2(square_source, square_target, 0.08, 5)
        np.testing.assert_allclose(phase.data, baseline.data)

        wrong_wavelength = slm.LF[slm.Intensity](
            square_target.data, square_target.L, 2.0
        )
        mismatched = slm.otPhase2(
            square_source, wrong_wavelength, 0.08, 2
        )
        ignored = slm.otPhase2(
            square_source, square_target, 0.08, 2, ignored_option=True
        )
        expected = slm.otPhase2(square_source, square_target, 0.08, 2)
        np.testing.assert_allclose(mismatched.data, expected.data)
        np.testing.assert_allclose(ignored.data, expected.data)

    def test_otphase2_zero_iterations_matches_julia_u_over_a(self) -> None:
        lattice = slm.natlat((3, 3))
        x, y = np.meshgrid(*lattice, indexing="ij")
        source_data = np.exp(-(x**2 + 0.7 * y**2))
        target_data = np.exp(-((x - 0.2) ** 2 + y**2))
        source = slm.LF[slm.Intensity](source_data, lattice)
        target = slm.LF[slm.Intensity](target_data, lattice)
        epsilon = 0.12
        result = slm.otPhase2(source, target, epsilon, 0)

        a = source_data / np.sum(source_data)
        n, m = a.shape
        X, Y = slm.natlat((n, m))
        Kx = np.exp(-((X[:, None] - X[None, :]) ** 2) / (2 * n * epsilon))
        Ky = np.exp(-((Y[:, None] - Y[None, :]) ** 2) / (2 * n * epsilon))
        u = np.full((n, n), 1 / (n * n))
        v = np.full((n, n), 1 / (n * n))
        scale = u / a
        dx = scale * (Kx @ (v * X[:, None]) @ Ky)
        dy = scale * (Kx @ (v * Y[None, :]) @ Ky)
        expected = slm.scalarPotentialN(
            np.stack((dx, dy), axis=-1), lattice, dimOrder=(2, 1)
        )
        np.testing.assert_allclose(result.data, expected, rtol=2e-15, atol=2e-15)

    def test_fraction_otphase2_finishes_in_float64(self) -> None:
        lattice = slm.natlat((2, 2))
        source_data = np.asarray(
            [[Fraction(1), Fraction(2)], [Fraction(3), Fraction(4)]],
            dtype=object,
        )
        target_data = np.flip(source_data, axis=0).copy()
        source = slm.LF[slm.Intensity, object, 2](source_data, lattice)
        target = slm.LF[slm.Intensity, object, 2](target_data, lattice)
        result = slm.otPhase2(source, target, 0.2, 0)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(
            result.data,
            np.asarray(
                [
                    [0.0, -0.3040850845638407],
                    [-0.28268492244459115, -0.456127626845761],
                ]
            ),
            rtol=2e-15,
            atol=2e-15,
        )

    def test_mixed_fraction_float_otphase2_always_finishes_in_float64(self) -> None:
        lattice = slm.natlat((2, 2))
        rational_source = np.asarray(
            [[Fraction(1), Fraction(2)], [Fraction(3), Fraction(4)]],
            dtype=object,
        )
        rational_target = np.flip(rational_source, axis=0).copy()
        float_source = np.asarray(rational_source, dtype=np.float64)
        float_target = np.asarray(rational_target, dtype=np.float64)

        for source_data, target_data in (
            (rational_source, rational_target),
            (rational_source, float_target),
            (float_source, rational_target),
        ):
            source = (
                slm.LF[slm.Intensity, object, 2](source_data, lattice)
                if source_data.dtype.kind == "O"
                else slm.LF[slm.Intensity](source_data, lattice)
            )
            target = (
                slm.LF[slm.Intensity, object, 2](target_data, lattice)
                if target_data.dtype.kind == "O"
                else slm.LF[slm.Intensity](target_data, lattice)
            )
            float_source_field = slm.LF[slm.Intensity](
                np.asarray(source_data, dtype=np.float64), lattice
            )
            float_target_field = slm.LF[slm.Intensity](
                np.asarray(target_data, dtype=np.float64), lattice
            )
            for iterations in (0, 1):
                result = slm.otPhase2(
                    source, target, 0.2, iterations
                )
                expected = slm.otPhase2(
                    float_source_field,
                    float_target_field,
                    0.2,
                    iterations,
                )
                self.assertEqual(result.dtype, np.dtype(np.float64))
                np.testing.assert_allclose(
                    result.data, expected.data, rtol=2e-15, atol=2e-15
                )

    def test_heterogeneous_real_otphase2_matches_julia_matrix_real_promotion(self) -> None:
        lattice = slm.natlat((2, 2))
        source_data = np.asarray(
            [[Fraction(1), 2.0], [3, Fraction(4)]], dtype=object
        )
        target_data = np.asarray(
            [[4.0, Fraction(3)], [Fraction(2), 1]], dtype=object
        )
        source = slm.LF[slm.Intensity, object, 2](source_data, lattice)
        target = slm.LF[slm.Intensity, object, 2](target_data, lattice)
        float_source = slm.LF[slm.Intensity](
            np.asarray(source_data, dtype=np.float64), lattice
        )
        float_target = slm.LF[slm.Intensity](
            np.asarray(target_data, dtype=np.float64), lattice
        )

        for iterations in (0, 1):
            result = slm.otPhase2(source, target, 0.2, iterations)
            expected = slm.otPhase2(
                float_source, float_target, 0.2, iterations
            )
            self.assertEqual(result.dtype, np.dtype(np.float64))
            np.testing.assert_allclose(
                result.data, expected.data, rtol=2e-15, atol=2e-15
            )


if __name__ == "__main__":
    unittest.main()
