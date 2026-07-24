"""Deterministic coverage for the IFT/phase-diversity port."""

from __future__ import annotations

from fractions import Fraction
import unittest

import numpy as np

import slmtools as slm
from slmtools.ift import _literal_square
from slmtools.lattice_field import _julia_add_sum


class GerchbergSaxtonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = slm.natlat((8, 8))
        x, y = np.meshgrid(*self.lattice, indexing="ij")
        self.source = slm.LF[slm.Modulus](np.exp(-(x**2 + y**2) / 2), self.lattice)
        self.zero_phase = slm.LF[slm.ComplexPhase](
            np.zeros(self.source.shape, dtype=np.complex128), self.lattice
        )
        applied = np.exp(2j * np.pi * (0.07 * x**2 - 0.04 * y))
        target_data = np.abs(slm.sft(self.source.data * applied))
        self.target = slm.LF[slm.Modulus](
            target_data, slm.dualShiftLattice(self.lattice)
        )

    def test_gs_iter_defines_zero_phasor_as_one(self) -> None:
        update = slm.gsIter(
            np.zeros(4, dtype=np.complex128), np.ones(4), np.ones(4)
        )
        np.testing.assert_array_equal(update, np.ones(4, dtype=np.complex128))
        self.assertEqual(update.dtype, np.complex128)

    def test_raw_iteration_helpers_keep_their_concrete_julia_dispatch(self) -> None:
        guess = np.ones(4, dtype=np.complex128)
        real = np.ones(4, dtype=np.float64)
        phases = (np.ones(4, dtype=np.complex128),)
        moduli = (real.copy(),)
        self.assertEqual(slm.gsIter(guess, real, real).dtype, np.complex128)
        self.assertEqual(
            slm.pdgsIter(guess, phases, moduli).dtype, np.complex128
        )

        with self.assertRaisesRegex(TypeError, "ComplexF64"):
            slm.gsIter(guess.astype(np.complex64), real, real)
        with self.assertRaisesRegex(TypeError, "Float64"):
            slm.gsIter(guess, real.astype(np.float32), real)
        with self.assertRaisesRegex(TypeError, "dense NumPy"):
            slm.gsIter(guess.tolist(), real, real)
        with self.assertRaisesRegex(TypeError, "ComplexF64"):
            slm.pdgsIter(guess, (phases[0].astype(np.complex64),), moduli)
        with self.assertRaisesRegex(TypeError, "Float64"):
            slm.pdgsIter(guess, phases, (real.astype(np.float32),))

    def test_intensity_gs_retains_upstream_default_phase_failure(self) -> None:
        source = slm.square(self.source)
        target = slm.square(self.target)
        with self.assertRaisesRegex(TypeError, "nonexistent four-argument"):
            slm.gs(source, target, 3, rng=np.random.default_rng(1234))
        with self.assertRaisesRegex(TypeError, "nonexistent four-argument"):
            slm.gsLog(source, target, 0)

        result = slm.gs(source, target, 0, self.zero_phase)
        self.assertIs(result.field_type, slm.ComplexPhase)
        self.assertEqual(result.flambda, source.flambda)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(result.L, source.L)))
        np.testing.assert_allclose(np.abs(result.data), 1.0, atol=2e-16)

    def test_integer_intensity_sqrt_widens_to_julia_float64_work(self) -> None:
        lattice = slm.natlat((4,))
        source = slm.LF[slm.Intensity, np.int8, 1](
            np.asarray([1, 4, 9, 16], dtype=np.int8), lattice
        )
        target = slm.LF[slm.Intensity, np.int8, 1](
            np.asarray([16, 9, 4, 1], dtype=np.int8),
            slm.dualShiftLattice(lattice),
        )
        phase = slm.LF[slm.RealPhase](np.zeros(4), lattice)

        # Locked Julia's sqrt.(::Vector{Int8}) is Float64, so the converted
        # modulus fields reach the concrete Float64 gsIter method.
        expected = np.asarray(
            [
                -0.9596829822606673 - 0.28108463771482023j,
                0.9133165073168443 + 0.4072504849138436j,
                0.5054494651244235 + 0.8628562094610168j,
                0.9874115596689939 - 0.15817209561754267j,
            ]
        )
        retrieved = slm.gs(source, target, 1, phase)
        logged, errors = slm.gsLog(source, target, 1, phase)
        np.testing.assert_allclose(
            retrieved.data, expected, rtol=3e-15, atol=3e-15
        )
        np.testing.assert_allclose(
            logged.data, expected, rtol=3e-15, atol=3e-15
        )
        self.assertAlmostEqual(errors[0], 0.24475518098344098, places=14)

    def test_complex_initial_phase_is_normalized_like_julia_phasor(self) -> None:
        initial_data = (2.0 + 3.0j) * np.ones(self.source.shape)
        initial = slm.LF[slm.ComplexPhase](initial_data, self.lattice)
        result = slm.gs(self.source, self.target, 0, initial)
        expected = initial_data / np.abs(initial_data)
        np.testing.assert_allclose(result.data, expected, atol=1e-15)

    def test_complex64_phase_reaches_julia_phasor_dispatch_failure(self) -> None:
        initial = slm.LF[slm.ComplexPhase, np.complex64, 2](
            np.ones(self.source.shape, dtype=np.complex64), self.lattice
        )
        with self.assertRaisesRegex(TypeError, "ComplexF64"):
            slm.gs(self.source, self.target, 0, initial)
        with self.assertRaisesRegex(TypeError, "ComplexF64"):
            slm.gsLog(self.source, self.target, 0, initial)

    def test_gs_logger_cadence_and_exact_error(self) -> None:
        result, errors = slm.gsLog(
            self.source, self.target, 6, self.zero_phase, every=2
        )
        self.assertEqual(len(errors), 3)  # iterations 1, 3, 5 in Julia notation
        self.assertTrue(np.all(np.isfinite(errors)))
        self.assertIs(result.field_type, slm.ComplexPhase)

        x, y = np.meshgrid(*self.lattice, indexing="ij")
        phase = slm.LF[slm.RealPhase](0.07 * x**2 - 0.04 * y, self.lattice)
        self.assertLess(slm.gsError(self.source, self.target, phase), 2e-30)

    def test_equal_lattice_check_uses_array_norm_julia_tolerance(self) -> None:
        shifted_lattice = tuple(np.asarray(axis) * (1.0 + 1e-10) for axis in self.lattice)
        approximate_phase = slm.LF[slm.RealPhase](
            np.zeros(self.source.shape), shifted_lattice, 1.0 + 1e-10
        )
        # Julia compares each complete axis using the array norm-based
        # isapprox method, and compares flambda separately as a scalar.
        result = slm.gs(self.source, self.target, 0, approximate_phase)
        self.assertIs(result.field_type, slm.ComplexPhase)

    def test_negative_gs_counts_are_empty_loops(self) -> None:
        negative = slm.gs(self.source, self.target, -3, self.zero_phase)
        zero = slm.gs(self.source, self.target, 0, self.zero_phase)
        np.testing.assert_array_equal(negative.data, zero.data)

        negative_log, negative_errors = slm.gsLog(
            self.source, self.target, -3, self.zero_phase
        )
        zero_log, zero_errors = slm.gsLog(
            self.source, self.target, 0, self.zero_phase
        )
        np.testing.assert_array_equal(negative_log.data, zero_log.data)
        self.assertEqual(negative_errors, zero_errors)

    def test_gs_log_nonpositive_every_matches_julia_modulo_timing(self) -> None:
        result, errors = slm.gsLog(
            self.source, self.target, 0, self.zero_phase, every=0
        )
        self.assertIs(result.field_type, slm.ComplexPhase)
        self.assertEqual(errors, [])
        with self.assertRaises(ZeroDivisionError):
            slm.gsLog(self.source, self.target, 1, self.zero_phase, every=0)

        _, negative_errors = slm.gsLog(
            self.source, self.target, 5, self.zero_phase, every=-2
        )
        self.assertEqual(len(negative_errors), 3)
        with self.assertRaises(TypeError):
            slm.gsLog(
                self.source,
                self.target,
                0,
                self.zero_phase,
                every=np.int32(1),
            )

    def test_float32_gs_retains_upstream_positive_iteration_failure(self) -> None:
        lattice = slm.natlat((4,))
        source = slm.LF[slm.Modulus](
            np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32), lattice
        )
        target = slm.LF[slm.Modulus](
            np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
            slm.dualShiftLattice(lattice),
        )
        phase = slm.LF[slm.RealPhase](np.zeros(4, dtype=np.float32), lattice)
        empty = slm.gs(source, target, 0, phase)
        self.assertEqual(empty.dtype, np.dtype(np.complex128))
        with self.assertRaisesRegex(TypeError, "no matching Julia gsIter"):
            slm.gs(source, target, 1, phase)

    def test_rational_modulus_gs_only_has_the_valid_empty_loop(self) -> None:
        lattice = slm.natlat((2,))
        dual = slm.dualShiftLattice(lattice)
        values = np.array([Fraction(1), Fraction(1)], dtype=object)
        source = slm.LF[slm.Modulus, object, 1](values.copy(), lattice)
        target = slm.LF[slm.Modulus, object, 1](values.copy(), dual)
        phase = slm.LF[slm.RealPhase, object, 1](
            np.array([Fraction(0), Fraction(0)], dtype=object), lattice
        )
        result = slm.gs(source, target, 0, phase)
        np.testing.assert_array_equal(result.data, np.ones(2, dtype=np.complex128))
        with self.assertRaisesRegex(TypeError, "no matching Julia gsIter"):
            slm.gs(source, target, 1, phase)
        logged, errors = slm.gsLog(source, target, 1, phase)
        np.testing.assert_array_equal(logged.data, np.ones(2, dtype=np.complex128))
        self.assertEqual(errors, [0.5857864376269049])

    def test_gs_log_checks_rational_int64_square_before_fft_conversion(self) -> None:
        lattice = slm.natlat((1,))
        value = Fraction(3_037_000_500)
        source = slm.LF[slm.Modulus, object, 1](
            np.asarray([value], dtype=object), lattice
        )
        target = slm.LF[slm.Modulus, object, 1](
            np.asarray([value], dtype=object),
            slm.dualShiftLattice(lattice),
        )
        phase = slm.LF[slm.RealPhase, object, 1](
            np.asarray([Fraction(0)], dtype=object), lattice
        )
        with self.assertRaisesRegex(OverflowError, "Rational"):
            slm.gsLog(source, target, 0, phase)

    def test_gs_log_retains_julia_low_precision_normalization(self) -> None:
        expected = {
            np.float16: (
                np.asarray(
                    [
                        0.9271975318332463 - 0.37457273921407624j,
                        -0.44725591232056094 + 0.8944060313382858j,
                        0.6275271171219212 - 0.7785947066841968j,
                        -0.3010837336881195 + 0.9535977062201971j,
                    ]
                ),
                0.42271084611871196,
            ),
            np.float32: (
                np.asarray(
                    [
                        0.9269521627009157 - 0.37517954110544877j,
                        -0.4468232072310975 + 0.8946222786627414j,
                        0.6252847264145274 - 0.7803967009878435j,
                        -0.30064440885900207 + 0.9537363049720929j,
                    ]
                ),
                0.4226727170217658,
            ),
        }
        lattice = slm.natlat((4,))
        dual = slm.dualShiftLattice(lattice)
        for dtype, (expected_phase, expected_error) in expected.items():
            source = slm.LF[slm.Modulus](
                np.asarray([1, 2, 3, 4], dtype=dtype), lattice
            )
            target = slm.LF[slm.Modulus](
                np.asarray([4, 3, 2, 1], dtype=dtype), dual
            )
            phase = slm.LF[slm.RealPhase](
                np.asarray([0, 0.1, 0.2, 0.3], dtype=dtype), lattice
            )
            result, errors = slm.gsLog(source, target, 1, phase)
            np.testing.assert_allclose(
                result.data, expected_phase, rtol=2e-15, atol=2e-15
            )
            self.assertEqual(len(errors), 1)
            self.assertAlmostEqual(errors[0], expected_error, places=15)


class PhaseDiversityTests(unittest.TestCase):
    def test_pdgs_matches_existing_julia_regression(self) -> None:
        n = 128
        lattice = slm.natlat((n,))
        x = np.asarray(lattice[0])
        source = slm.LF[slm.Intensity](np.exp(-(x**2) / 2), lattice)
        initial_modulus = slm.LF[slm.Intensity](np.exp(-(x**2)), lattice)
        phases = tuple(
            slm.LF[slm.RealPhase](alpha * x**2 / 2, lattice)
            for alpha in np.arange(0.1, 1.01, 0.1)
        )
        images = tuple(slm.square(slm.sft(np.sqrt(source) * phase)) for phase in phases)
        estimate = slm.pdgs(images, phases, 100, np.sqrt(initial_modulus) * phases[0])
        expected = np.asarray(
            [
                0.5510345631134064 + 0.14050740448745414j,
                0.5877216316040312 + 0.14985954977667548j,
                0.6244073206579513 + 0.1592112010176915j,
            ]
        )
        # FFTW and pocketfft differ by a few ulps per iteration; after 100
        # updates the deterministic cross-backend envelope is about 1e-13.
        np.testing.assert_allclose(estimate.data[47:50], expected, rtol=2e-13, atol=1e-14)
        self.assertIs(estimate.field_type, slm.ComplexAmp)

    def test_pdgs_iter_log_and_error_invariants(self) -> None:
        lattice = slm.natlat((9,))
        x = np.asarray(lattice[0])
        beam = slm.LF[slm.ComplexAmp](np.exp(-x**2) * np.exp(0.3j * x), lattice)
        phase = slm.LF[slm.RealPhase](0.08 * x**2, lattice)
        image = slm.square(slm.sft(beam * phase))
        image_modulus = np.sqrt(image)
        estimate, errors = slm.pdgsLog((image_modulus,), (phase,), 2, beam, every=1)
        self.assertEqual(errors, [0.0, 0.0])
        self.assertIs(estimate.field_type, slm.ComplexAmp)
        self.assertLess(slm.pdgsError((image_modulus,), (phase,), beam), 2e-30)

    def test_one_shot_and_mraf_return_finite_semantic_fields(self) -> None:
        lattice = slm.natlat((8, 8))
        x, y = np.meshgrid(*lattice, indexing="ij")
        intensity = slm.LF[slm.Intensity](np.exp(-(x**2 + y**2)), lattice)
        recovered = slm.oneShot(intensity, 0.4, (0.03, -0.02))
        self.assertIs(recovered.field_type, slm.ComplexAmp)
        self.assertTrue(np.all(np.isfinite(recovered.data)))
        self.assertTrue(
            all(
                np.allclose(a, b)
                for a, b in zip(recovered.L, slm.dualShiftLattice(lattice))
            )
        )
        zero_alpha = slm.oneShot(intensity, 0.0, (0.0, 0.0))
        self.assertIs(zero_alpha.field_type, slm.ComplexAmp)
        self.assertTrue(np.all(np.isnan(zero_alpha.data)))
        complex_intensity = slm.LF[slm.Intensity](
            np.ones(intensity.shape, dtype=np.complex128), intensity
        )
        with self.assertRaises(TypeError):
            slm.oneShot(complex_intensity, 0.4, (0.03, -0.02))
        with self.assertRaisesRegex(TypeError, "alpha must be real"):
            slm.oneShot(intensity, 0.4 + 0.1j, (0.03, -0.02))
        with self.assertRaisesRegex(TypeError, "beta values must be real"):
            slm.oneShot(intensity, 0.4, (0.03 + 0.1j, -0.02))

        source = np.sqrt(intensity)
        target = slm.LF[slm.Modulus](
            np.abs(slm.sft(source.data)), slm.dualShiftLattice(lattice)
        )
        phase0 = slm.LF[slm.RealPhase](np.zeros(source.shape), lattice)
        output = slm.mraf(source, target, 2, phase0, (slice(2, 6), slice(1, 7)), 0.48)
        self.assertIs(output.field_type, slm.ComplexPhase)
        np.testing.assert_allclose(np.abs(output.data), 1.0, atol=2e-16)

    def test_one_shot_preserves_low_precision_range_and_wavelength_semantics(self) -> None:
        expected = {
            np.float16: np.asarray(
                [
                    -0.12007764446878572 - 0.05539984923575711j,
                    -0.08673173532520619 + 0.03462106897699646j,
                    0.5531140301484158 - 0.11873787893017895j,
                    -0.2169765617128122 - 0.06488692198402504j,
                ]
            ),
            np.float32: np.asarray(
                [
                    -0.12016187230636514 - 0.05543533364432913j,
                    -0.08670570820920742 + 0.03466707876567068j,
                    0.5530664380002607 - 0.11876853296722198j,
                    -0.2169519821324755 - 0.06489788610991264j,
                ]
            ),
        }
        for dtype, oracle in expected.items():
            lattice = (
                slm.LatticeAxis(
                    np.asarray([-0.3, -0.1, 0.1, 0.3], dtype=dtype),
                    step_hint=dtype(0.2),
                ),
            )
            image = slm.LF[slm.Intensity, dtype, 1](
                np.asarray([0.04, 0.25, 0.49, 0.81], dtype=dtype),
                lattice,
                dtype(1.25),
            )
            result = slm.oneShot(image, dtype(0.4), (dtype(0.03),))
            self.assertEqual(result.dtype, np.dtype(np.complex128))
            self.assertIsInstance(result.flambda, dtype)
            self.assertEqual(result.L[0].dtype, np.dtype(dtype))
            np.testing.assert_allclose(result.data, oracle, rtol=2e-15, atol=2e-15)

    def test_one_shot_preserves_heterogeneous_ntuple_promotion(self) -> None:
        axis = slm.LatticeAxis(
            np.asarray([-0.5, 0.0], dtype=np.float32),
            step_hint=np.float32(0.5),
        )
        lattice = (axis, axis)
        image = slm.LF[slm.Intensity, np.float32, 2](
            np.asarray([[0.25, 1.0], [0.5, 0.75]], dtype=np.float32),
            lattice,
            np.float32(1),
        )
        result = slm.oneShot(
            image,
            np.float32(0.5),
            (np.int64(100_000_001), np.float32(1)),
        )
        expected = np.asarray(
            [
                [
                    -0.08220698821156162 - 0.02265044947881239j,
                    0.017626216887843556 + 0.004856542537396357j,
                ],
                [
                    0.04375732000078307 - 0.15881170017741958j,
                    0.7406822763636951 + 0.20407987163128982j,
                ],
            ]
        )
        # FFTW and pocketfft differ by a handful of binary64 ulps.
        np.testing.assert_allclose(
            result.data.copy(), expected, rtol=2e-15, atol=2e-15
        )
        self.assertIsInstance(result.flambda, np.float32)
        self.assertTrue(
            all(axis.dtype == np.dtype(np.float32) for axis in result.L)
        )

        # The Julia method is NTuple-only; accepting lists/arrays would apply
        # vector-literal promotion and silently select different arithmetic.
        with self.assertRaisesRegex(TypeError, "NTuple"):
            slm.oneShot(
                image,
                np.float32(0.5),
                [np.int64(100_000_001), np.float32(1)],
            )
        with self.assertRaisesRegex(TypeError, "NTuple"):
            slm.oneShot(
                image,
                np.float32(0.5),
                np.asarray([100_000_001, 1], dtype=np.float32),
            )

    def test_literal_square_retains_julia_base_type_and_overflow(self) -> None:
        probes = (
            (True, np.bool_(True)),
            (np.int8(120), np.int8(64)),
            (np.uint8(20), np.uint8(144)),
            (np.float16(0.3), np.float16(0.09)),
            (np.float16(300), np.float16(np.inf)),
        )
        for value, expected in probes:
            result = _literal_square(value)
            self.assertEqual(np.asarray(result).dtype, np.asarray(expected).dtype)
            self.assertEqual(result, expected)

        int8_sum = _julia_add_sum(
            (_literal_square(np.int8(100)), _literal_square(np.int8(100)))
        )
        self.assertIsInstance(int8_sum, np.int64)
        self.assertEqual(int8_sum, 32)

        mixed_sum = _julia_add_sum(
            (
                _literal_square(np.float32(100_000_000)),
                _literal_square(np.int64(1)),
            )
        )
        self.assertIsInstance(mixed_sum, np.float32)
        self.assertEqual(mixed_sum, np.float32(1.0e16))

        unsigned_wrap = _julia_add_sum((np.uint8(1), np.int64(-2)))
        self.assertIsInstance(unsigned_wrap, np.uint64)
        self.assertEqual(unsigned_wrap, np.iinfo(np.uint64).max)

    def test_float16_pdgs_retains_step_range_dual_phase_arithmetic(self) -> None:
        lattice = (
            slm.LatticeAxis(
                np.asarray([-0.3, -0.1, 0.1, 0.3], dtype=np.float16),
                step_hint=np.float16(0.2),
            ),
        )
        flambda = np.float16(0.5)
        dual = slm.dualShiftLattice(lattice, flambda)
        image = slm.LF[slm.Modulus](
            np.asarray([0.2, 0.7, 1.0, 0.4], dtype=np.float16),
            lattice,
            flambda,
        )
        phase = slm.LF[slm.RealPhase](
            np.asarray([0.0, 0.1, -0.2, 0.3], dtype=np.float16),
            dual,
            flambda,
        )
        beam = slm.LF[slm.ComplexAmp](
            np.asarray(
                [1 + 0.1j, 0.2 - 0.3j, -0.4 + 0.8j, 0.7 - 0.2j],
                dtype=np.complex128,
            ),
            dual,
            flambda,
        )
        expected = np.asarray(
            [
                0.39936772279430255 + 0.26209387306250265j,
                0.12168704184668054 - 0.1870287869352003j,
                -0.021994167778589133 + 0.2825008307581525j,
                -0.22581105323850792 - 0.11527178174563604j,
            ]
        )
        np.testing.assert_allclose(
            slm.pdgs((image,), (phase,), 1, beam).data,
            expected,
            rtol=2e-15,
            atol=2e-15,
        )

    def test_complex64_pdgs_beam_fails_only_when_julia_enters_loop(self) -> None:
        lattice = slm.natlat((4,))
        dual = slm.dualShiftLattice(lattice)
        image = slm.LF[slm.Modulus](np.ones(4), lattice)
        phase = slm.LF[slm.RealPhase](np.zeros(4), dual)
        beam = slm.LF[slm.ComplexAmp, np.complex64, 1](
            np.ones(4, dtype=np.complex64), dual
        )
        empty = slm.pdgs((image,), (phase,), 0, beam)
        self.assertEqual(empty.dtype, np.dtype(np.complex128))
        with self.assertRaisesRegex(TypeError, "ComplexF64"):
            slm.pdgs((image,), (phase,), 1, beam)
        with self.assertRaisesRegex(TypeError, "ComplexF64"):
            slm.pdgsLog((image,), (phase,), 1, beam)

    def test_pdgs_log_nonpositive_every_matches_julia_modulo_timing(self) -> None:
        lattice = slm.natlat((5,))
        x = np.asarray(lattice[0])
        beam = slm.LF[slm.ComplexAmp](np.exp(-x**2), lattice)
        phase = slm.LF[slm.RealPhase](np.zeros(5), lattice)
        image = np.sqrt(slm.square(slm.sft(beam * phase)))

        _, empty = slm.pdgsLog((image,), (phase,), 0, beam, every=0)
        self.assertEqual(empty, [])
        with self.assertRaises(ZeroDivisionError):
            slm.pdgsLog((image,), (phase,), 1, beam, every=0)
        _, negative = slm.pdgsLog((image,), (phase,), 5, beam, every=-2)
        self.assertEqual(len(negative), 3)

    def test_negative_pdgs_and_mraf_counts_are_empty_loops(self) -> None:
        lattice = slm.natlat((5,))
        x = np.asarray(lattice[0])
        beam = slm.LF[slm.ComplexAmp](np.exp(-x**2) * np.exp(0.2j * x), lattice)
        phase = slm.LF[slm.RealPhase](0.1 * x**2, lattice)
        image = np.sqrt(slm.square(slm.sft(beam * phase)))

        negative = slm.pdgs((image,), (phase,), -2, beam)
        zero = slm.pdgs((image,), (phase,), 0, beam)
        np.testing.assert_array_equal(negative.data, zero.data)
        negative_log, negative_errors = slm.pdgsLog(
            (image,), (phase,), -2, beam
        )
        zero_log, zero_errors = slm.pdgsLog((image,), (phase,), 0, beam)
        np.testing.assert_array_equal(negative_log.data, zero_log.data)
        self.assertEqual(negative_errors, zero_errors)

        source = slm.LF[slm.Modulus](np.exp(-x**2), lattice)
        target = slm.LF[slm.Modulus](
            np.abs(slm.sft(source.data)), slm.dualShiftLattice(lattice)
        )
        roi = (slice(None),)
        negative_mraf = slm.mraf(source, target, -2, phase, roi, 0.5)
        zero_mraf = slm.mraf(source, target, 0, phase, roi, 0.5)
        np.testing.assert_array_equal(negative_mraf.data, zero_mraf.data)


if __name__ == "__main__":
    unittest.main()
