from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import unittest
import warnings

import numpy as np

import slmtools as slm
from slmtools import ot


class SingletonPhaseRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = slm.natlat((1, 4))
        self.dual = slm.dualShiftLattice(self.lattice)
        self.modulus = slm.LF[slm.Modulus](
            np.asarray([[1.0, 2.0, 1.5, 0.5]]), self.lattice
        )
        self.target_modulus = slm.LF[slm.Modulus](
            np.asarray([[0.5, 1.0, 2.0, 1.5]]), self.dual
        )
        self.phase = slm.LF[slm.RealPhase](np.zeros((1, 4)), self.lattice)

    def test_gs_accepts_singleton_axis_at_zero_and_nonzero_iterations(self) -> None:
        for iterations in (0, 2):
            result = slm.gs(
                self.modulus, self.target_modulus, iterations, self.phase
            )
            self.assertEqual(result.shape, (1, 4))
            self.assertIs(result.field_type, slm.ComplexPhase)
            self.assertTrue(np.all(np.isfinite(result.data)))
            np.testing.assert_allclose(np.abs(result.data), 1.0)

        # A singleton coordinate cannot reveal its range step from its lone
        # value.  Exercise a non-unit retained step so this checks metadata,
        # rather than passing only because natlat's singleton step is one.
        custom_lattice = (
            slm.LatticeAxis([0.0], step_hint=2.0),
            slm.LatticeAxis([-0.75, -0.25, 0.25, 0.75], step_hint=0.5),
        )
        custom_dual = slm.dualShiftLattice(custom_lattice)
        custom_source = slm.LF[slm.Modulus](np.ones((1, 4)), custom_lattice)
        custom_target = slm.LF[slm.Modulus](np.ones((1, 4)), custom_dual)
        custom_phase = slm.LF[slm.RealPhase](np.zeros((1, 4)), custom_lattice)
        result = slm.gs(custom_source, custom_target, 1, custom_phase)
        self.assertTrue(np.all(np.isfinite(result.data)))

        # Julia's lattice equality compares the materialized coordinates.  A
        # singleton range's retained step is not observable once collected,
        # so a different hidden step must not turn equal coordinates into a
        # false dual-lattice mismatch.
        mismatched_step_dual = (
            slm.LatticeAxis(custom_dual[0], step_hint=7.0),
            custom_dual[1],
        )
        target_with_hidden_step = slm.LF[slm.Modulus](
            np.ones((1, 4)), mismatched_step_dual
        )
        hidden_step_result = slm.gs(
            custom_source, target_with_hidden_step, 0, custom_phase
        )
        self.assertIs(hidden_step_result.field_type, slm.ComplexPhase)

    def test_pdgs_and_mraf_use_singleton_step_metadata(self) -> None:
        image = slm.LF[slm.Modulus](
            np.asarray([[1.0, 0.5, 1.5, 2.0]]), self.dual
        )
        beam = slm.LF[slm.ComplexAmp](
            np.asarray([[1.0, 0.75j, -0.5, -0.25j]]), self.lattice
        )
        for iterations in (0, 1):
            estimate = slm.pdgs((image,), (self.phase,), iterations, beam)
            self.assertEqual(estimate.shape, (1, 4))
            self.assertTrue(np.all(np.isfinite(estimate.data)))
            mraf_phase = slm.mraf(
                self.modulus,
                self.target_modulus,
                iterations,
                self.phase,
                (slice(None), slice(None)),
                0.4,
            )
            self.assertEqual(mraf_phase.shape, (1, 4))
            self.assertTrue(np.all(np.isfinite(mraf_phase.data)))


class HomogeneousDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = slm.natlat((2, 3))
        self.dual = slm.dualShiftLattice(self.lattice)
        self.modulus = slm.LF[slm.Modulus](np.ones((2, 3)), self.lattice)
        self.modulus_dual = slm.LF[slm.Modulus](
            np.ones((2, 3)), self.dual
        )
        self.intensity = slm.LF[slm.Intensity](
            np.ones((2, 3)), self.lattice
        )
        self.intensity_dual = slm.LF[slm.Intensity](
            np.ones((2, 3)), self.dual
        )
        self.phase = slm.LF[slm.RealPhase](np.zeros((2, 3)), self.lattice)
        self.beam = slm.LF[slm.ComplexAmp](np.ones((2, 3)), self.lattice)

    def test_gs_family_rejects_mixed_modulus_intensity_pairs(self) -> None:
        with self.assertRaises(TypeError):
            slm.gs(self.modulus, self.intensity_dual, 0, self.phase)
        with self.assertRaises(TypeError):
            slm.gsLog(self.modulus, self.intensity_dual, 0, self.phase)
        with self.assertRaises(TypeError):
            slm.gsError(self.modulus, self.intensity_dual, self.phase)

        # The two homogeneous Julia overloads remain supported.
        self.assertIs(
            slm.gs(self.modulus, self.modulus_dual, 0, self.phase).field_type,
            slm.ComplexPhase,
        )
        self.assertIs(
            slm.gs(self.intensity, self.intensity_dual, 0, self.phase).field_type,
            slm.ComplexPhase,
        )

    def test_pdgs_family_rejects_heterogeneous_image_tuples(self) -> None:
        phases = (self.phase, self.phase)
        mixed = (self.modulus_dual, self.intensity_dual)
        with self.assertRaises(TypeError):
            slm.pdgs(mixed, phases, 0, self.beam)
        with self.assertRaises(TypeError):
            slm.pdgsLog(mixed, phases, 0, self.beam)
        with self.assertRaises(TypeError):
            slm.pdgsError(mixed, phases, self.beam)

        # Julia defines the intensity conversion overload for pdgs itself, but
        # pdgsLog and pdgsError are Modulus-only.
        result = slm.pdgs(
            (self.intensity_dual, self.intensity_dual), phases, 0, self.beam
        )
        self.assertIs(result.field_type, slm.ComplexAmp)
        with self.assertRaises(TypeError):
            slm.pdgsLog(
                (self.intensity_dual, self.intensity_dual), phases, 0, self.beam
            )
        with self.assertRaises(TypeError):
            slm.pdgsError(
                (self.intensity_dual, self.intensity_dual), phases, self.beam
            )

    def test_mraf_is_modulus_only(self) -> None:
        roi = (slice(None), slice(None))
        with self.assertRaises(TypeError):
            slm.mraf(
                self.modulus,
                self.intensity_dual,
                0,
                self.phase,
                roi,
                0.5,
            )
        with self.assertRaises(TypeError):
            slm.mraf(
                self.intensity,
                self.intensity_dual,
                0,
                self.phase,
                roi,
                0.5,
            )
        with self.assertRaisesRegex(TypeError, "m must be real"):
            slm.mraf(
                self.modulus,
                self.modulus_dual,
                0,
                self.phase,
                roi,
                0.5 + 0.1j,
            )

    def test_complex_modulus_inputs_fail_without_lossy_casts(self) -> None:
        complex_source = slm.LF[slm.Modulus](
            np.ones((2, 3), dtype=np.complex128), self.lattice
        )
        complex_target = slm.LF[slm.Modulus](
            np.ones((2, 3), dtype=np.complex128), self.dual
        )
        roi = (slice(None), slice(None))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for operation in (
                lambda: slm.gs(complex_source, complex_target, 0, self.phase),
                lambda: slm.gsLog(complex_source, complex_target, 0, self.phase),
                lambda: slm.gsError(complex_source, complex_target, self.phase),
                lambda: slm.mraf(
                    complex_source, complex_target, 0, self.phase, roi, 0.5
                ),
            ):
                with self.assertRaises(TypeError):
                    operation()
        self.assertFalse(
            any(issubclass(item.category, np.exceptions.ComplexWarning) for item in caught)
        )

    def test_pdgs_error_keeps_julia_complex_modulus_method(self) -> None:
        lattice = slm.natlat((2, 2))
        dual = slm.dualShiftLattice(lattice)
        image = slm.LF[slm.Modulus](
            np.full((2, 2), 1.0 + 0.5j, dtype=np.complex128), dual
        )
        phase = slm.LF[slm.RealPhase](np.zeros((2, 2)), lattice)
        beam = slm.LF[slm.ComplexAmp](np.ones((2, 2), dtype=np.complex128), lattice)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            error = slm.pdgsError((image,), (phase,), beam)
        self.assertEqual(error, 1.0)
        self.assertFalse(
            any(issubclass(item.category, np.exceptions.ComplexWarning) for item in caught)
        )

    def test_complex_real_phase_data_matches_julia_number_overloads(self) -> None:
        lattice = slm.natlat((4,))
        dual = slm.dualShiftLattice(lattice)
        source = slm.LF[slm.Modulus](
            np.asarray([1.0, 2.0, 3.0, 4.0]), lattice
        )
        target = slm.LF[slm.Modulus](
            np.asarray([4.0, 1.0, 2.0, 3.0]), dual
        )
        phase = slm.LF[slm.RealPhase](
            np.asarray([0.0 + 0.1j, 0.1 - 0.2j, 0.2 + 0.05j, -0.15 + 0.3j]),
            lattice,
        )
        beam = slm.LF[slm.ComplexAmp](
            np.asarray([1.0 + 0.2j, 0.5 - 0.1j, -0.25 + 0.5j, 0.75 - 0.3j]),
            lattice,
        )
        expected_mraf = np.asarray(
            [
                0.1700272343627151 - 0.9854393637230889j,
                0.8550478373681822 + 0.5185491257460517j,
                -0.5762647971380606 + 0.8172630443005666j,
                -0.33105735414382403 - 0.9436106338248265j,
            ]
        )
        expected_pdgs = np.asarray(
            [
                0.07000029930080803 - 0.21131414592940528j,
                8.314519686678258 - 1.2143074369049929j,
                -0.21517360591760037 + 0.7473370726299412j,
                0.06670319224075012 + 0.08033661746759664j,
            ]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gs_error = slm.gsError(source, target, phase)
            mraf_phase = slm.mraf(source, target, 1, phase, slice(1, 3), 0.4)
            pdgs_error = slm.pdgsError((target,), (phase,), beam)
            pdgs_beam = slm.pdgs((target,), (phase,), 1, beam)
            logged_beam, logged_errors = slm.pdgsLog(
                (target,), (phase,), 1, beam
            )

        self.assertAlmostEqual(gs_error, 0.26121439696114274, places=15)
        np.testing.assert_allclose(
            mraf_phase.data, expected_mraf, rtol=2e-15, atol=2e-15
        )
        self.assertAlmostEqual(pdgs_error, 0.1898188245508838, places=15)
        np.testing.assert_allclose(
            pdgs_beam.data, expected_pdgs, rtol=2e-15, atol=2e-15
        )
        np.testing.assert_allclose(
            logged_beam.data, expected_pdgs, rtol=2e-15, atol=2e-15
        )
        self.assertEqual(logged_errors, [0.0])
        self.assertFalse(
            any(issubclass(item.category, np.exceptions.ComplexWarning) for item in caught)
        )

    def test_zero_norm_ift_paths_retain_julia_ieee_propagation(self) -> None:
        lattice = slm.natlat((2,))
        dual = slm.dualShiftLattice(lattice)
        zero_source = slm.LF[slm.Modulus](np.zeros(2), lattice)
        unit_target = slm.LF[slm.Modulus](np.ones(2), dual)
        phase0 = slm.LF[slm.RealPhase](np.zeros(2), lattice)

        zero_iteration, zero_errors = slm.gsLog(
            zero_source, unit_target, 0, phase0
        )
        np.testing.assert_array_equal(
            zero_iteration.data, np.ones(2, dtype=np.complex128)
        )
        self.assertEqual(zero_errors, [])

        one_iteration, one_errors = slm.gsLog(
            zero_source, unit_target, 1, phase0
        )
        self.assertTrue(np.all(np.isnan(one_iteration.data)))
        self.assertEqual(len(one_errors), 1)
        self.assertTrue(np.isnan(one_errors[0]))

        mraf_phase = slm.mraf(
            zero_source,
            unit_target,
            0,
            phase0,
            (slice(None),),
            0.5,
        )
        self.assertTrue(np.all(np.isnan(mraf_phase.data)))

        zero_dual = slm.LF[slm.Modulus](np.zeros(2), dual)
        beam = slm.LF[slm.ComplexAmplitude](
            np.ones(2, dtype=np.complex128), lattice
        )
        self.assertTrue(np.isnan(slm.pdgsError((zero_dual,), (phase0,), beam)))


class OptimalTransportContractTests(unittest.TestCase):
    def test_concrete_array_ot_helpers_reject_strided_subarray_views(
        self,
    ) -> None:
        dense = np.full((2, 2), 0.25, dtype=np.float64)
        strided = np.full((2, 4), 0.25, dtype=np.float64)[:, ::2]
        transform = np.ones((2, 2), dtype=np.complex128)
        strided_transform = np.ones(
            (2, 4), dtype=np.complex128
        )[:, ::2]

        for position in range(6):
            arguments = [
                dense.copy(),
                dense.copy(),
                dense.copy(),
                dense.copy(),
                transform.copy(),
                transform.copy(),
            ]
            arguments[position] = (
                strided_transform
                if position >= 4
                else strided
            )
            with self.assertRaisesRegex(
                TypeError, "dense contiguous"
            ):
                getattr(slm, "SinkhornIterBase!")(*arguments)

        for source, target in (
            (strided, dense),
            (dense, strided),
        ):
            with self.assertRaisesRegex(
                TypeError, "dense contiguous"
            ):
                slm.SinkhornConvN(source, target, 0.2, 0)

        lattice = slm.natlat((2, 2))
        for u, v, source in (
            (strided, dense, dense),
            (dense, strided, dense),
            (dense, dense, strided),
        ):
            with self.assertRaisesRegex(
                TypeError, "dense contiguous"
            ):
                slm.dualToGradients(u, v, source, lattice, 0.2)

        field = slm.LF[slm.Intensity](dense, lattice)
        beta_view = np.arange(4.0)[::2]
        for helper in (slm.pdotPhase, slm.pdotBeamEstimate):
            with self.assertRaisesRegex(
                TypeError, "dense contiguous"
            ):
                helper(
                    field,
                    field,
                    1.0,
                    0.0,
                    beta_view,
                    np.zeros(2),
                    0.2,
                )

        # A contiguous reshape is the Python counterpart of
        # reshape(::Array), even though NumPy records a non-owning base.
        reshaped = np.full(4, 0.25).reshape(2, 2)
        u, v, loss = slm.SinkhornConvN(
            reshaped, reshaped.copy(), 0.2, 0
        )
        self.assertEqual(u.shape, (2, 2))
        self.assertEqual(v.shape, (2, 2))
        self.assertEqual(loss, [])

    def test_sinkhorn_mutating_helper_returns_u_identity(self) -> None:
        source = np.asarray([[0.2, 0.3], [0.1, 0.4]])
        target = np.asarray([[0.1, 0.4], [0.2, 0.3]])
        transform = np.ones((2, 2), dtype=np.complex128)

        for helper in (
            ot._sinkhorn_iter_base_inplace,
            ot.SinkhornIterBase,
            getattr(ot, "SinkhornIterBase!"),
            getattr(slm, "SinkhornIterBase!"),
        ):
            u = np.full((2, 2), 0.25)
            v = np.full((2, 2), 0.25)
            returned = helper(u, v, source, target, transform, transform)
            self.assertIs(returned, u)
            self.assertEqual(returned.shape, source.shape)
            self.assertEqual(v.shape, target.shape)
            self.assertAlmostEqual(float(np.sum(u)), 1.0)
            self.assertAlmostEqual(float(np.sum(v)), 1.0)
        self.assertFalse(hasattr(slm, "SinkhornIterBase"))

    def test_sinkhorn_mutating_helper_preserves_rational_domain_across_calls(
        self,
    ) -> None:
        source = np.asarray(
            [Fraction(1, 2), Fraction(1, 2)], dtype=object
        )
        target = source.copy()
        u = source.copy()
        v = source.copy()
        transform = np.ones(2, dtype=np.complex128)

        for _ in range(2):
            returned = getattr(slm, "SinkhornIterBase!")(
                u, v, source, target, transform, transform
            )
            self.assertIs(returned, u)
            self.assertTrue(
                all(isinstance(value, Fraction) for value in u)
            )
            self.assertTrue(
                all(isinstance(value, Fraction) for value in v)
            )
            np.testing.assert_equal(u, source)
            np.testing.assert_equal(v, target)

    def test_sinkhorn_mutating_helper_requires_matching_scaling_types(self) -> None:
        source = np.full((2, 2), 0.25, dtype=np.float32)
        target = np.full((2, 2), 0.25, dtype=np.float64)
        transform = np.ones((2, 2), dtype=np.complex128)
        with self.assertRaisesRegex(TypeError, "same Julia element type"):
            getattr(slm, "SinkhornIterBase!")(
                source.copy(), target.copy(), source, target, transform, transform
            )
        with self.assertRaisesRegex(TypeError, "same Julia element type"):
            getattr(slm, "SinkhornIterBase!")(
                source.copy(),
                source.copy(),
                source,
                target,
                transform,
                transform.astype(np.complex64),
            )

    def test_sinkhorn_mutating_helper_preserves_julia_statement_order(self) -> None:
        transform = np.ones((1,), dtype=np.complex128)
        u = np.ones((1,), dtype=np.int64)
        v = np.ones((1,), dtype=np.int64)
        returned = getattr(slm, "SinkhornIterBase!")(
            u,
            v,
            np.ones((1,), dtype=np.int64),
            np.ones((1,), dtype=np.int64),
            transform,
            transform,
        )
        self.assertIs(returned, u)
        np.testing.assert_array_equal(u, [1])
        np.testing.assert_array_equal(v, [1])

        u = np.asarray([1, 1], dtype=np.int64)
        v = np.asarray([7, 9], dtype=np.int64)
        original_u, original_v = u.copy(), v.copy()
        transform = np.ones((2,), dtype=np.complex128)
        with self.assertRaisesRegex(ValueError, "Inexact assignment"):
            getattr(slm, "SinkhornIterBase!")(
                u,
                v,
                np.asarray([1, 1], dtype=np.int64),
                np.asarray([1, 2], dtype=np.int64),
                transform,
                transform,
            )
        np.testing.assert_array_equal(u, original_u)
        # Julia completes both `v` statements before the later `u`
        # conversion fails. The successful mutation remains visible.
        np.testing.assert_array_equal(v, [1, 2])
        self.assertFalse(np.array_equal(v, original_v))

    def test_sinkhorn_mutations_are_elementwise_not_atomic(self) -> None:
        cases = (
            (
                np.asarray([1, 2]),
                np.asarray([9, 9]),
                np.asarray([1, 1]),
                np.asarray([1, 3]),
                np.asarray([1, 2]),
                np.asarray([1, 9]),
            ),
            (
                np.ones(3, dtype=np.int64),
                np.asarray([9, 9, 9]),
                np.ones(3, dtype=np.int64),
                np.asarray([4, 1, -1]),
                np.asarray([1, 1, 1]),
                np.asarray([1, 1, -1]),
            ),
            (
                np.ones(4, dtype=np.int64),
                np.full(4, 9, dtype=np.int64),
                np.asarray([4, 1, 1, 0]),
                np.asarray([2, 2, -3, 0]),
                np.asarray([2, 1, 1, 1]),
                np.asarray([2, 2, -3, 0]),
            ),
            (
                np.ones(4, dtype=np.int64),
                np.full(4, 9, dtype=np.int64),
                np.asarray([8, 2, 3, 0]),
                np.asarray([2, 2, -3, 0]),
                np.asarray([1, 1, -1, 0]),
                np.asarray([2, 2, -3, 0]),
            ),
        )
        for u, v, source, target, expected_u, expected_v in cases:
            transform = np.ones(len(u), dtype=np.complex128)
            with self.assertRaisesRegex(
                ValueError, "Inexact assignment"
            ):
                getattr(slm, "SinkhornIterBase!")(
                    u, v, source, target, transform, transform
                )
            np.testing.assert_array_equal(u, expected_u)
            np.testing.assert_array_equal(v, expected_v)

    def test_sinkhorn_mutating_helper_uses_float32_fftw_path(self) -> None:
        u = np.asarray(
            [1e-5, 0.13, 0.27, 0.19, 0.40999], dtype=np.float32
        )
        v = np.asarray(
            [0.31, 0.07, 0.22, 0.18, 0.22], dtype=np.float32
        )
        source = np.asarray(
            [0.05, 0.15, 0.4, 0.25, 0.15], dtype=np.float32
        )
        target = np.asarray(
            [0.21, 0.33, 0.17, 0.09, 0.2], dtype=np.float32
        )
        transform = slm.sft(
            np.asarray([1.0, 0.07, 0.4, 0.11, 0.22], dtype=np.float32)
        )
        getattr(slm, "SinkhornIterBase!")(
            u, v, source, target, transform, transform
        )
        # The oracle values come from Julia/FFTW.  Maintained FFTW builds may
        # select different codelets and differ by a few ULP at Float32.
        expected_u = np.asarray(
            [
                0x3D8592FF,
                0x3E0B7F79,
                0x3EE2C82B,
                0x3E74F3BE,
                0x3DEE65DB,
            ],
            dtype=np.uint32,
        ).view(np.float32)
        expected_v = np.asarray(
            [
                0x3E49C8AC,
                0x3EA5464D,
                0x3DDB291C,
                0x3E422B7B,
                0x3E3BEAB4,
            ],
            dtype=np.uint32,
        ).view(np.float32)
        np.testing.assert_array_max_ulp(u, expected_u, maxulp=3)
        np.testing.assert_array_max_ulp(v, expected_v, maxulp=2)

    def test_otphase2_keyword_contract_on_working_square_input(self) -> None:
        lattice = slm.natlat((2, 2))
        source = slm.LF[slm.Intensity](
            np.asarray([[1.0, 2.0], [3.0, 4.0]]), lattice
        )
        target = slm.LF[slm.Intensity](
            np.asarray([[4.0, 3.0], [2.0, 1.0]]), lattice
        )

        zero = slm.otPhase2(source, target, 0.1, 0)
        one = slm.otPhase2(source, target, 0.1, 1)
        baseline = slm.otPhase2(source, target, 0.1, 3)
        ignored = slm.otPhase2(
            source,
            target,
            0.1,
            3,
            atol=np.inf,
            check_convergence=0,
            return_loss=True,
            arbitrary_future_keyword="ignored",
        )
        for result in (zero, one, baseline, ignored):
            self.assertIsInstance(result, slm.LatticeField)
            self.assertEqual(result.shape, (2, 2))
            self.assertTrue(np.all(np.isfinite(result.data)))
        np.testing.assert_allclose(ignored.data, baseline.data)
        self.assertFalse(np.allclose(ignored.data, one.data))

    def test_otphase2_zero_iteration_positive_over_zero_stays_invalid(self) -> None:
        lattice = slm.natlat((2, 2))
        source = slm.LF[slm.Intensity](
            np.asarray([[0.0, 1.0], [1.0, 1.0]]), lattice
        )
        target = slm.LF[slm.Intensity](np.ones((2, 2)), lattice)
        with np.errstate(invalid="ignore"):
            result = slm.otPhase2(source, target, 0.1, 0)
        self.assertTrue(np.all(np.isnan(result.data)))

    def test_otphase2_nonpositive_parameters_retain_julia_ieee_results(self) -> None:
        lattice = slm.natlat((2, 2))
        source = slm.LF[slm.Intensity](np.ones((2, 2)), lattice)
        target = slm.LF[slm.Intensity](np.ones((2, 2)), lattice)

        negative = slm.otPhase2(source, target, -0.2, 0)
        # Exact Julia 1.11.6/locked-project result. These values depend on
        # natrange(2)'s TwicePrecision coordinates, not merely ±sqrt(1/2).
        np.testing.assert_allclose(
            negative.data,
            np.asarray(
                [
                    [0.0, -0.5141771795203925],
                    [-0.5141771795203927, -1.0283543590407853],
                ]
            ),
            rtol=2e-15,
            atol=2e-15,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            zero_epsilon = slm.otPhase2(source, target, 0.0, 0)
        self.assertTrue(np.all(np.isnan(zero_epsilon.data)))

        zero_wavelength_source = slm.LF[slm.Intensity](
            np.ones((2, 2)), lattice, 0.0
        )
        zero_wavelength_target = slm.LF[slm.Intensity](
            np.ones((2, 2)), lattice, 0.0
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            zero_wavelength = slm.otPhase2(
                zero_wavelength_source,
                zero_wavelength_target,
                0.2,
                0,
            )
        self.assertTrue(np.isnan(zero_wavelength.data[0, 0]))
        self.assertTrue(np.all(np.isneginf(zero_wavelength.data.flat[1:])))

    def test_dense_ot_retains_nonpositive_epsilon_and_zero_divisors(self) -> None:
        marginal = np.asarray([0.5, 0.5])
        cost = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            zero_epsilon = ot._sinkhorn_gibbs(
                marginal, marginal, cost, 0.0, maxiter=2
            )
        self.assertTrue(np.all(np.isnan(zero_epsilon)))
        self.assertTrue(
            any("not converged" in str(item.message) for item in caught)
        )
        negative_epsilon = ot._sinkhorn_gibbs(
            marginal, marginal, cost, -1.0, maxiter=10
        )
        self.assertTrue(np.all(np.isfinite(negative_epsilon)))

        lattice = slm.natlat((3,))
        zero_wavelength = slm.LF[slm.Intensity](
            np.asarray([1.0, 2.0, 1.0]), lattice, 0.0
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            phase = slm.otPhase(
                zero_wavelength, zero_wavelength, 1.0, maxiter=100
            )
        self.assertTrue(np.any(~np.isfinite(phase.data)))

        ordinary = slm.LF[slm.Intensity](
            np.asarray([1.0, 2.0, 1.0]), lattice, 1.0
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            equal_alpha = slm.pdotPhase(
                ordinary,
                ordinary,
                0.5,
                0.5,
                (0.0,),
                (0.0,),
                1.0,
                maxiter=100,
            )
        self.assertTrue(np.any(~np.isfinite(equal_alpha.data)))

    def test_otphase2_complex_intensity_target_matches_julia_golden(self) -> None:
        lattice = slm.natlat((2, 2))
        source = slm.LF[slm.Intensity](
            np.asarray([[1.0, 2.0], [3.0, 4.0]]), lattice
        )
        # Julia's constructor-from-field bypasses Intensity clipping and makes
        # the advertised target ``<:Number`` overload directly reachable.
        target = slm.LF[slm.Intensity](
            np.asarray(
                [[1.0 + 0.1j, 2.0 + 0.2j], [3.0 - 0.1j, 4.0 + 0.3j]],
                dtype=np.complex128,
            ),
            source,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = slm.otPhase2(source, target, 0.2, 2)
        self.assertFalse(
            any(issubclass(item.category, np.exceptions.ComplexWarning) for item in caught)
        )
        expected = np.asarray(
            [
                [
                    0.0 + 0.0j,
                    -0.20443714217173523 + 0.007761863944109464j,
                ],
                [
                    -0.1699163832770825 - 0.009780187576964678j,
                    -0.38673908621179143 + 0.0006811648167050167j,
                ],
            ]
        )
        self.assertIs(result.field_type, slm.RealPhase)
        self.assertTrue(np.iscomplexobj(result.data))
        np.testing.assert_allclose(
            result.data.copy(), expected, rtol=3e-15, atol=3e-15
        )

    def test_otphase2_iteration_division_exposes_tiny_kernel_failure(self) -> None:
        lattice = slm.natlat((2, 2))
        source = slm.LF[slm.Intensity](
            np.asarray([[1.0, 0.0], [0.0, 0.0]]), lattice
        )
        target = slm.LF[slm.Intensity](
            np.asarray([[0.0, 0.0], [0.0, 1.0]]), lattice
        )
        result = slm.otPhase2(source, target, np.finfo(float).tiny, 1)
        self.assertTrue(np.any(np.isnan(result.data)))
        self.assertFalse(np.all(np.isfinite(result.data)))

    def test_otphase2_rejects_complex_sources_but_allows_complex_modulus_target(self) -> None:
        lattice = slm.natlat((2, 2))
        real_intensity = slm.LF[slm.Intensity](np.ones((2, 2)), lattice)
        complex_intensity = slm.LF[slm.Intensity](
            np.ones((2, 2), dtype=np.complex128), real_intensity
        )
        real_modulus = slm.LF[slm.Modulus](np.ones((2, 2)), lattice)
        complex_modulus = slm.LF[slm.Modulus](
            np.full((2, 2), 1.0 + 0.5j), lattice
        )
        with self.assertRaises(TypeError):
            slm.otPhase2(complex_intensity, real_intensity, 0.2, 0)
        with self.assertRaises(TypeError):
            slm.otPhase2(complex_modulus, real_modulus, 0.2, 0)
        with self.assertRaises(TypeError):
            slm.otPhase(complex_intensity, real_intensity, 0.2)
        with self.assertRaises(TypeError):
            slm.pdotPhase(
                real_intensity,
                complex_intensity,
                0.4,
                0.1,
                (0.0, 0.0),
                (0.0, 0.0),
                0.2,
            )
        result = slm.otPhase2(real_modulus, complex_modulus, 0.2, 0)
        self.assertTrue(np.all(np.isfinite(result.data)))

    def test_negative_ot_iteration_counts_follow_julia_empty_loops(self) -> None:
        lattice = slm.natlat((3, 3))
        source_data = np.arange(1, 10, dtype=float).reshape(3, 3)
        target_data = np.flip(source_data)
        source = slm.LF[slm.Intensity](source_data, lattice)
        target = slm.LF[slm.Intensity](target_data, lattice)
        negative = slm.otPhase2(source, target, 0.2, -4)
        zero = slm.otPhase2(source, target, 0.2, 0)
        np.testing.assert_array_equal(negative.data, zero.data)

        normalized_source = source_data / np.sum(source_data)
        normalized_target = target_data / np.sum(target_data)
        u_negative, v_negative, loss_negative = slm.SinkhornConvN(
            normalized_source, normalized_target, 0.2, -3, every=1
        )
        u_zero, v_zero, loss_zero = slm.SinkhornConvN(
            normalized_source, normalized_target, 0.2, 0, every=1
        )
        np.testing.assert_array_equal(u_negative, u_zero)
        np.testing.assert_array_equal(v_negative, v_zero)
        self.assertEqual(loss_negative, loss_zero)

        quick_negative, quick_negative_loss = slm.otQuickPhase(
            source, target, 0.2, -3, return_loss=True
        )
        quick_zero, quick_zero_loss = slm.otQuickPhase(
            source, target, 0.2, 0, return_loss=True
        )
        np.testing.assert_array_equal(quick_negative.data, quick_zero.data)
        self.assertEqual(quick_negative_loss, quick_zero_loss)

    def test_sinkhorn_every_uses_julia_integer_and_modulo_semantics(self) -> None:
        source = np.arange(1, 10, dtype=float).reshape(3, 3)
        source /= np.sum(source)
        target = np.flip(source).copy()

        _, _, empty = slm.SinkhornConvN(
            source, target, 0.2, 0, every=0
        )
        self.assertEqual(empty, [])
        with self.assertRaises(ZeroDivisionError):
            slm.SinkhornConvN(source, target, 0.2, 1, every=0)
        _, _, negative = slm.SinkhornConvN(
            source, target, 0.2, 4, every=-2
        )
        self.assertEqual(len(negative), 2)
        _, _, boolean = slm.SinkhornConvN(
            source, target, 0.2, 2, every=True
        )
        self.assertEqual(len(boolean), 2)
        with self.assertRaises(TypeError):
            slm.SinkhornConvN(source, target, 0.2, 0, every=1.0)

    def test_dense_sinkhorn_keywords_require_julia_int64(self) -> None:
        lattice = (range(1, 4),)
        source = slm.LF[slm.Intensity](
            np.asarray([1.0, 2.0, 3.0]), lattice
        )
        target = slm.LF[slm.Intensity](
            np.asarray([3.0, 2.0, 1.0]), lattice
        )
        for keyword in ("maxiter", "check_convergence"):
            for invalid in (True, np.int32(1)):
                with self.assertRaisesRegex(TypeError, "Int"):
                    slm.otPhase(
                        source,
                        target,
                        0.5,
                        **{keyword: invalid},
                    )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = slm.otPhase(
                source,
                target,
                0.5,
                maxiter=np.int64(1),
                check_convergence=np.int64(0),
            )
        self.assertEqual(result.shape, (3,))

    def test_float32_sinkhorn_conv_retains_upstream_dispatch_failure(self) -> None:
        # Julia allocates ``u`` as Float64 but ``v`` as Float32 and therefore
        # misses SinkhornIterBase!'s same-type dispatch on the first update.
        source = np.arange(1, 10, dtype=np.float32).reshape(3, 3, order="F")
        source /= np.sum(source)
        target = np.flip(source).copy()
        with self.assertRaisesRegex(TypeError, "same Julia element type"):
            slm.SinkhornConvN(source, target, 0.2, 2, every=1)

    def test_sinkhorn_conv_rejects_every_non_float64_target_on_update(self) -> None:
        source = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
        for dtype in (np.float16, np.float32, np.int64):
            target = np.flip(np.arange(1, 10).reshape(3, 3)).astype(dtype)
            with self.assertRaisesRegex(TypeError, "same Julia element type"):
                slm.SinkhornConvN(source, target, 0.2, 1)

    def test_sinkhorn_conv_singleton_retains_julia_nan_result(self) -> None:
        for iterations in (0, 1):
            u, v, loss = slm.SinkhornConvN(
                np.ones(1), np.ones(1), 0.2, iterations
            )
            self.assertTrue(np.all(np.isnan(u)))
            self.assertTrue(np.all(np.isnan(v)))
            self.assertEqual(loss, [])

    def test_sinkhorn_conv_retains_nonpositive_and_nonfinite_epsilon(self) -> None:
        source = np.asarray([0.2, 0.3, 0.5])
        target = np.asarray([0.4, 0.1, 0.5])
        for epsilon in (0.0, np.nan):
            u, v, _ = slm.SinkhornConvN(
                source, target, epsilon, 1
            )
            self.assertTrue(np.all(np.isnan(u)))
            self.assertTrue(np.all(np.isnan(v)))

        u, v, _ = slm.SinkhornConvN(
            source, target, -1.0, 1
        )
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertTrue(np.all(np.isfinite(v)))

        u, v, _ = slm.SinkhornConvN(
            source, target, np.inf, 1
        )
        np.testing.assert_array_equal(u, source)
        np.testing.assert_array_equal(v, target)

    def test_convolutional_ot_rejects_bigfloat_work_like_fftw(self) -> None:
        source = np.asarray([0.2, 0.3, 0.5])
        target = np.asarray([0.4, 0.1, 0.5])
        lattice = (np.arange(3.0),)
        with self.assertRaisesRegex(TypeError, "BigFloat"):
            slm.SinkhornConvN(
                source, target, Decimal("0.2"), 0
            )
        with self.assertRaisesRegex(TypeError, "BigFloat"):
            slm.dualToGradients(
                np.ones(3),
                np.ones(3),
                source,
                lattice,
                Decimal("0.2"),
            )
        source_field = slm.LF[slm.Intensity](source, lattice)
        target_field = slm.LF[slm.Intensity](target, lattice)
        with self.assertRaisesRegex(TypeError, "BigFloat"):
            slm.otQuickPhase(
                source_field,
                target_field,
                Decimal("0.2"),
                0,
            )

    def test_otphase2_iteration_argument_matches_exact_julia_int_dispatch(self) -> None:
        lattice = slm.natlat((2, 2))
        source = slm.LF[slm.Intensity](np.ones((2, 2)), lattice)
        target = slm.LF[slm.Intensity](np.ones((2, 2)), lattice)

        for invalid in (True, np.bool_(True), np.int32(0), np.uint64(0)):
            with self.assertRaises(TypeError):
                slm.otPhase2(source, target, 0.2, invalid)
        result = slm.otPhase2(source, target, 0.2, np.int64(0))
        self.assertEqual(result.shape, (2, 2))
        with self.assertRaises(OverflowError):
            slm.otPhase2(source, target, 0.2, 2**63)

    def test_dense_ot_preserves_finite_julia_signed_maximum_geometry(self) -> None:
        for source_size in range(1, 9):
            for target_size in (3, 4, 5, 6, 7, 8):
                source, target = ot._ot_natural_lattices(
                    (source_size,), (target_size,)
                )
                raw_target = ot._natural_axis(target_size)
                julia_scale = (
                    float(np.max(source[0])) / float(np.max(raw_target))
                )
                np.testing.assert_array_equal(
                    target[0], raw_target * julia_scale
                )

    def test_dense_ot_target_two_retains_upstream_invalid_geometry(self) -> None:
        for source_size in range(2, 9):
            _, target = ot._ot_natural_lattices((source_size,), (2,))
            self.assertTrue(np.any(~np.isfinite(target[0])))

        # A two-sample source and working three-sample target has Julia's exact
        # zero scale and remains authoritative.
        _, target = ot._ot_natural_lattices((2,), (3,))
        np.testing.assert_array_equal(target[0], np.zeros(3))

        source_field = slm.LF[slm.Intensity](
            np.asarray([1.0, 2.0]), (np.arange(2.0),)
        )
        target_field = slm.LF[slm.Intensity](
            np.asarray([1.0, 3.0, 2.0]), (np.arange(3.0),)
        )
        phase = slm.otPhase(source_field, target_field, 0.5, maxiter=200)
        np.testing.assert_allclose(
            phase.data,
            np.asarray([0.0, 1.1666666666666667]),
            rtol=2e-15,
            atol=2e-15,
        )

    def test_dense_ot_two_sample_source_plan_remains_working(self) -> None:
        source_lattice, target_lattice = ot._ot_natural_lattices((2,), (5,))
        source = np.asarray([1.0, 2.0])
        source /= np.sum(source)
        target = np.arange(5, 0, -1, dtype=float)
        target /= np.sum(target)
        cost = slm.getCostMatrix(source_lattice, target_lattice)
        plan = ot._sinkhorn_gibbs(
            source,
            target,
            cost,
            1.0,
            atol=1e-13,
            check_convergence=1,
            maxiter=10_000,
        )
        self.assertTrue(np.all(np.isfinite(plan)))
        np.testing.assert_allclose(np.sum(plan, axis=1), source, atol=1e-13)
        np.testing.assert_allclose(np.sum(plan, axis=0), target, atol=2e-16)

    def test_dense_ot_two_sample_public_phase_failure_is_directional(self) -> None:
        for source_size, target_size in ((2, 5),):
            source = slm.LF[slm.Intensity](
                np.arange(1, source_size + 1, dtype=float),
                (np.arange(source_size, dtype=float),),
            )
            target = slm.LF[slm.Intensity](
                np.arange(target_size, 0, -1, dtype=float),
                (np.arange(target_size, dtype=float),),
            )
            phase = slm.otPhase(
                source, target, 1.0, atol=1e-13, maxiter=10_000
            )
            self.assertIs(phase.field_type, slm.RealPhase)
            self.assertEqual(phase.shape, (source_size,))
            self.assertTrue(np.all(np.isfinite(phase.data)))

            diverse_phase = slm.pdotPhase(
                source,
                target,
                0.5,
                0.1,
                (0.0,),
                (0.0,),
                1.0,
                atol=1e-13,
                maxiter=10_000,
            )
            self.assertIs(diverse_phase.field_type, slm.RealPhase)
            self.assertEqual(diverse_phase.shape, (source_size,))
            self.assertTrue(np.all(np.isfinite(diverse_phase.data)))

        source = slm.LF[slm.Intensity](
            np.arange(1, 6, dtype=float), (np.arange(5.0),)
        )
        target = slm.LF[slm.Intensity](
            np.asarray([2.0, 1.0]), (np.arange(2.0),)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(
                FloatingPointError, "sinkhorn returned nan"
            ):
                slm.otPhase(source, target, 1.0, maxiter=10)

    def test_dense_ot_singleton_axis_retains_julia_invalid_geometry(self) -> None:
        for source_shape, target_shape in (
            ((1,), (2,)),
            ((2,), (1,)),
            ((3, 1), (3, 2)),
            ((3, 2), (3, 1)),
        ):
            with np.errstate(divide="ignore", invalid="ignore"):
                _, target_lattice = ot._ot_natural_lattices(
                    source_shape, target_shape
                )
            self.assertTrue(
                any(np.any(np.isnan(axis)) for axis in target_lattice)
            )

        source = slm.LF[slm.Intensity](
            np.ones(1), (slm.LatticeAxis([0.0], step_hint=1.0),)
        )
        target = slm.LF[slm.Intensity](np.ones(2), (np.arange(2.0),))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(
                FloatingPointError, "sinkhorn returned nan"
            ):
                slm.otPhase(source, target, 0.2, maxiter=2)

    def test_phase_diversity_ot_rejects_nonreal_parameters(self) -> None:
        lattice = (np.arange(3.0),)
        field = slm.LF[slm.Intensity](np.ones(3), lattice)

        for alpha_root, alpha_target, flambda in (
            (0.5 + 0.1j, 0.1, 1.0),
            (0.5, 0.1 + 0.1j, 1.0),
            (0.5, 0.1, 1.0 + 0.1j),
        ):
            with self.assertRaises(TypeError):
                slm.pdCostMatrix(
                    lattice,
                    lattice,
                    alpha_root,
                    alpha_target,
                    flambda=flambda,
                )

        invalid_calls = (
            (0.5 + 0.1j, 0.1, (0.0,), (0.0,), 0.2),
            (0.5, 0.1 + 0.1j, (0.0,), (0.0,), 0.2),
            (0.5, 0.1, (0.0 + 0.1j,), (0.0,), 0.2),
            (0.5, 0.1, (0.0,), (0.0 + 0.1j,), 0.2),
            (0.5, 0.1, (0.0,), (0.0,), 0.2 + 0.1j),
        )
        for arguments in invalid_calls:
            with self.assertRaises(TypeError):
                slm.pdotPhase(field, field, *arguments)

    def test_pdot_beta_tuple_list_and_typed_vector_promotion(self) -> None:
        lattice = (
            slm.LatticeAxis(
                np.arange(99_999_999, 100_000_002, dtype=np.int64),
                step_hint=np.int64(1),
            ),
            slm.LatticeAxis(
                np.arange(3, dtype=np.int64), step_hint=np.int64(1)
            ),
        )
        field = slm.LF[slm.Intensity, np.float64, 2](
            np.arange(1, 10, dtype=np.float64).reshape((3, 3), order="F"),
            lattice,
            1.0,
        )
        heterogeneous = (np.int64(100_000_001), np.float32(1))
        heterogeneous_zero = (np.int64(0), np.float32(0))

        tuple_result = slm.pdotPhase(
            field,
            field,
            1.0,
            0.0,
            heterogeneous,
            heterogeneous_zero,
            1.0,
            maxiter=1000,
        )
        list_result = slm.pdotPhase(
            field,
            field,
            1.0,
            0.0,
            list(heterogeneous),
            list(heterogeneous_zero),
            1.0,
            maxiter=1000,
        )
        float32_result = slm.pdotPhase(
            field,
            field,
            1.0,
            0.0,
            np.asarray(heterogeneous, dtype=np.float32),
            np.asarray(heterogeneous_zero, dtype=np.float32),
            1.0,
            maxiter=1000,
        )
        object_result = slm.pdotPhase(
            field,
            field,
            1.0,
            0.0,
            np.asarray(heterogeneous, dtype=object),
            np.asarray(heterogeneous_zero, dtype=object),
            1.0,
            maxiter=1000,
        )
        extra_list_result = slm.pdotPhase(
            field,
            field,
            1.0,
            0.0,
            [*heterogeneous, np.float32(123)],
            [*heterogeneous_zero, np.float32(-456)],
            1.0,
            maxiter=1000,
        )
        extra_array_result = slm.pdotPhase(
            field,
            field,
            1.0,
            0.0,
            np.asarray([*heterogeneous, 123.0], dtype=object),
            np.asarray([*heterogeneous_zero, -456.0], dtype=object),
            1.0,
            maxiter=1000,
        )
        np.testing.assert_array_equal(extra_list_result.data, list_result.data)
        np.testing.assert_array_equal(
            extra_array_result.data, tuple_result.data
        )
        with self.assertRaisesRegex(ValueError, "NTuple"):
            slm.pdotPhase(
                field,
                field,
                1.0,
                0.0,
                (*heterogeneous, 123.0),
                (*heterogeneous_zero, -456.0),
                1.0,
            )
        with self.assertRaisesRegex(ValueError, "at least"):
            slm.pdotPhase(
                field, field, 1.0, 0.0, [0.0], [0.0], 1.0
            )

        # A Julia vector literal is Vector{Float32}; an explicit object vector
        # models Vector{Real} and retains the same heterogeneous scalar values
        # as the NTuple. The transport solve is identical, so these differences
        # isolate the beta correction.
        np.testing.assert_array_equal(float32_result.data, list_result.data)
        np.testing.assert_array_equal(object_result.data, tuple_result.data)
        np.testing.assert_allclose(
            tuple_result.data - list_result.data,
            np.asarray(
                [
                    [-2.0, -2.0, -2.0],
                    [-0.5, -0.5, -0.5],
                    [0.0, 0.0, 0.0],
                ]
            ),
            rtol=0,
            atol=1e-12,
        )

    def test_pdot_squared_offset_uses_julia_add_sum_widening(self) -> None:
        cases = (
            (
                np.asarray([100, 101], dtype=np.int8),
                np.int8(0),
                np.dtype(np.int64),
                np.asarray([16, -39], dtype=np.int64),
            ),
            (
                np.asarray([20, 21], dtype=np.uint8),
                np.uint8(0),
                np.dtype(np.uint64),
                np.asarray([144, 185], dtype=np.uint64),
            ),
            (
                np.asarray([0.25, 0.5], dtype=np.float16),
                np.float16(0),
                np.dtype(np.float16),
                np.asarray([0.0625, 0.25], dtype=np.float16),
            ),
        )
        for axis, offset, dtype, expected in cases:
            result = ot._pd_squared_radius((axis,), (offset,))
            self.assertEqual(result.dtype, dtype)
            np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
