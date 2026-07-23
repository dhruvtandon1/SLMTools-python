import copy
import unittest
import warnings
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

from slmtools.dual_lattices import (
    dualLattice,
    dualPhase,
    dualShiftLattice,
    isft,
    ldq,
    sft,
)
from slmtools.lattice_field import (
    ComplexAmplitude,
    DomainError,
    Generic,
    Intensity,
    LF,
    LatticeAxis,
    RealPhase,
    sublattice,
)
from slmtools.lattice_utils import (
    FrFTBasis,
    hermiteBasis,
    ldot,
    natlat,
    natrange,
    padout,
    shiftedDFTBasis,
    wigner_fft,
)
from slmtools.resampling import (
    Flat,
    Linear,
    Periodic,
    Throw,
    coarsen,
    cubic_spline_interpolation,
    downsample,
    upsample,
)
from slmtools.templates import lfBlur, lfCap, lfGaussian, lfParabola, lfRing


class ResamplingTests(unittest.TestCase):
    def test_julia_affine_downsample_example(self):
        array = np.arange(1, 17).reshape((4, 4), order="F")
        np.testing.assert_allclose(
            downsample(array, 2),
            np.array([[3.5, 11.5], [5.5, 13.5]]),
            atol=1e-12,
        )

    def test_downsample_and_upsample_lattice_geometry(self):
        lattice = (range(1, 13), range(1, 13))
        down = downsample(lattice, 2)
        np.testing.assert_allclose(down[0], np.arange(1.5, 12, 2))
        up = upsample((range(1, 4),), 2)
        np.testing.assert_allclose(up[0], [0.75, 1.25, 1.75, 2.25, 2.75, 3.25])
        singleton = natlat((1,))[0]
        np.testing.assert_allclose(downsample(singleton, 1), [0])
        np.testing.assert_allclose(upsample(singleton, 3), [-1 / 3, 0, 1 / 3])

    def test_explicit_grid_upsampling_and_boundaries(self):
        source = (range(1, 5),)
        target = (np.arange(1.0, 4.01, 0.5),)
        signal = np.array([2.0, 4.0, 6.0, 8.0])
        np.testing.assert_allclose(upsample(signal, source, target), 2 * target[0])

        spline = cubic_spline_interpolation(
            source, signal, extrapolation_bc=-9.0
        )
        np.testing.assert_allclose(spline[np.array([0.0, 1.0, 4.0, 5.0])], [-9, 2, 8, -9])
        flat = cubic_spline_interpolation(source, signal, extrapolation_bc=Flat())
        np.testing.assert_allclose(flat[np.array([0.0, 5.0])], [2, 8])
        periodic = cubic_spline_interpolation(
            source, signal, extrapolation_bc=Periodic()
        )
        np.testing.assert_allclose(periodic[np.array([0.0, 5.0])], [6, 4])
        linear = cubic_spline_interpolation(
            source, signal, extrapolation_bc=Linear()
        )
        np.testing.assert_allclose(linear[np.array([0.0, 5.0])], [0, 10])
        throwing = cubic_spline_interpolation(
            source, signal, extrapolation_bc=Throw()
        )
        with self.assertRaises(IndexError):
            _ = throwing[0.0]

    def test_natural_cubic_reproduces_knots_and_is_tensor_product(self):
        x = np.arange(5.0)
        y = np.arange(4.0)
        data = x[:, None] ** 3 - 2 * y[None, :] ** 2
        spline = cubic_spline_interpolation((x, y), data, extrapolation_bc=0)
        np.testing.assert_allclose(spline[(x, y)], data, atol=1e-12)
        result = spline[(np.array([0.5, 2.5]), np.array([0.25, 1.5, 2.75]))]
        self.assertEqual(result.shape, (2, 3))

    def test_natural_cubic_matches_julia_nonaffine_golden(self):
        signal = np.array([1.0, 4.0, 2.0, 8.0])
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
        spline = cubic_spline_interpolation((range(1, 5),), signal)
        np.testing.assert_allclose(spline[targets], expected, atol=2e-15)

    def test_coarsen_array_and_field(self):
        array = np.arange(1, 17).reshape((4, 4), order="F")
        np.testing.assert_allclose(
            coarsen(array, 2), np.array([[3.5, 11.5], [5.5, 13.5]])
        )
        maximum = coarsen(array, (2, 2), reducer=np.max)
        np.testing.assert_array_equal(maximum, [[6, 14], [8, 16]])
        field = LF[Intensity](array.astype(float), (range(4), range(4)), 2.0)
        reduced = coarsen(field, 2)
        self.assertIs(reduced.field_type, Intensity)
        self.assertEqual(reduced.flambda, 2.0)
        self.assertEqual(reduced.shape, (2, 2))

    def test_invalid_factors_and_shape(self):
        with self.assertRaises(DomainError):
            downsample((range(5),), 2)
        with self.assertRaises(DomainError):
            coarsen(np.zeros((3, 4)), (2, 2))
        with self.assertRaises(DomainError):
            upsample((range(4),), 0)


class DualAndTransformTests(unittest.TestCase):
    def test_dual_lattice_values_even_and_odd(self):
        lattice = (range(1, 11), range(1, 6))
        positive = dualLattice(lattice, 2.0)
        shifted = dualShiftLattice(lattice, 2.0)
        np.testing.assert_allclose(positive[0], np.arange(0.0, 2.0, 0.2))
        np.testing.assert_allclose(
            shifted[0], np.arange(-1.0, 1.0, 0.2), atol=1e-15
        )
        np.testing.assert_allclose(shifted[1], np.arange(-2, 3) * 0.4)
        self.assertIsNone(ldq(lattice, shifted, 2.0))
        with self.assertRaises(DomainError):
            ldq(lattice, lattice, 2.0)

    def test_dual_phase_formula(self):
        lattice = (range(1, 5), range(-2, 2))
        dual = dualShiftLattice(lattice, 2.0)
        phase = dualPhase(lattice, 2.0, dL=dual)
        expected = (
            3 * np.asarray(dual[0])[:, None]
            + 0 * np.asarray(dual[1])[None, :]
        ) / 2
        np.testing.assert_allclose(phase.data, expected)
        self.assertIs(phase.field_type, RealPhase)

    def test_float16_steprangelen_arithmetic_matches_julia_goldens(self):
        axis = LatticeAxis(
            np.asarray([-0.3, -0.1, 0.1, 0.3], dtype=np.float16),
            step_hint=np.float16(0.2),
        )
        lattice = (axis,)
        goldens = {
            "gaussian": np.asarray(
                [
                    0.9714371953438454,
                    1.2475214529227476,
                    1.2475214529227476,
                    0.9714371953438454,
                ]
            ),
            "parabola": np.asarray(
                [
                    0.017995605468750005,
                    0.0019995117187500003,
                    0.0019995117187500003,
                    0.017995605468750005,
                ]
            ),
            "ring": np.asarray(
                [
                    0.882651753113333,
                    0.3246896047060614,
                    0.3246896047060614,
                    0.882651753113333,
                ]
            ),
            "cap": np.asarray(
                [
                    0.991002197265625,
                    0.999000244140625,
                    0.999000244140625,
                    0.991002197265625,
                ]
            ),
            "ldot": np.asarray(
                [-0.009000000000000001, -0.003, 0.003, 0.009000000000000001]
            ),
            "dual_phase": np.asarray(
                [
                    -0.2496947496947497,
                    -0.12484737484737485,
                    0.0,
                    0.12484737484737485,
                ]
            ),
            "blur": np.asarray(
                [
                    2.275241407039518,
                    2.4409526246317377,
                    2.606630140532358,
                    2.4409189229401385,
                ]
            ),
        }
        actual = {
            "gaussian": lfGaussian(Intensity, lattice, np.float16(0.4)).data,
            "parabola": lfParabola(Intensity, lattice, np.float16(0.4)).data,
            "ring": lfRing(
                Intensity, lattice, np.float16(0.4), np.float16(0.2)
            ).data,
            "cap": lfCap(
                Intensity, lattice, np.float16(0.2), np.float16(1)
            ).data,
            "ldot": ldot((np.float64(0.03),), lattice),
            "dual_phase": dualPhase(lattice, np.float16(0.5)).data,
        }
        source = LF[Intensity](
            np.asarray([0.1, 0.4, 0.7, 1.0], dtype=np.float16),
            lattice,
            np.float16(1),
        )
        actual["blur"] = lfBlur(source, np.float16(0.4)).data
        for name, expected in goldens.items():
            np.testing.assert_array_equal(actual[name], expected, err_msg=name)

    def test_float16_steprangelen_common_denominator_overflow_falls_back(self):
        # Julia's rational-range probe finds denominators 489 and 1027 here.
        # Their common denominator overflows Float16 during Base's validity
        # check, so the range retains its literal binary start and step.
        values = np.asarray(
            [38960, 34400, 6036, 7111, 7650, 8160, 8432, 8687],
            dtype=np.uint16,
        ).view(np.float16)
        step = np.asarray([0x17FA], dtype=np.uint16).view(np.float16)[0]
        axis = LatticeAxis(values, step_hint=step)

        self.assertEqual(axis._logical_ref, -0.002044677734375)
        self.assertEqual(axis._logical_step, 0.0019474029541015625)
        self.assertEqual(axis._logical_offset, 0)
        np.testing.assert_array_equal(
            ldot((0.03,), (axis,)).view(np.uint64),
            np.asarray(
                [
                    13767526578869743124,
                    13747372970537260144,
                    4543315746584369562,
                    4548034674568923710,
                    4550490543740724183,
                    4552645938374886032,
                    4553970278140309668,
                    4555047975457390592,
                ],
                dtype=np.uint64,
            ),
        )

    def test_float16_steprangelen_rounds_scaled_rational_numerators(self):
        values = np.asarray(
            [
                44304,
                42578,
                10140,
                11618,
                12399,
                12845,
                13290,
                13524,
                13747,
                13970,
                14193,
                14376,
                14487,
                14599,
                14710,
            ],
            dtype=np.uint16,
        ).view(np.float16)
        step = np.asarray([0x2AF5], dtype=np.uint16).view(np.float16)[0]
        axis = LatticeAxis(values, step_hint=step)

        self.assertEqual(
            np.asarray(axis._logical_ref, dtype=np.float64).view(np.uint64),
            np.uint64(13806144746546974500),
        )
        self.assertEqual(
            np.asarray(axis._logical_step, dtype=np.float64).view(np.uint64),
            np.uint64(4588002018323635767),
        )
        self.assertEqual(axis._logical_offset, 1)

    def test_float16_range_factory_retains_literal_start_before_materialization(self):
        # Julia range(start=0x348c, step=0x20c6, length=25) materializes a
        # *different* first Float16 (0x348d).  Values plus a step hint cannot
        # recover that discarded input, so the explicit logical factory must
        # reproduce Base's reference/step/offset before materialization.
        start = np.asarray([0x348C], dtype=np.uint16).view(np.float16)[0]
        step = np.asarray([0x20C6], dtype=np.uint16).view(np.float16)[0]
        axis = LatticeAxis.from_start_step(start, step, 25)

        np.testing.assert_array_equal(
            axis.view(np.uint16),
            np.asarray(
                [
                    0x348D,
                    0x34B3,
                    0x34D9,
                    0x34FF,
                    0x3526,
                    0x354C,
                    0x3572,
                    0x3598,
                    0x35BE,
                    0x35E5,
                    0x360B,
                    0x3631,
                    0x3657,
                    0x367D,
                    0x36A4,
                    0x36CA,
                    0x36F0,
                    0x3716,
                    0x373C,
                    0x3762,
                    0x3789,
                    0x37AF,
                    0x37D5,
                    0x37FB,
                    0x3811,
                ],
                dtype=np.uint16,
            ),
        )
        self.assertEqual(
            np.asarray(axis._logical_ref, dtype=np.float64).view(np.uint64),
            np.uint64(0x3FD23351C0BEF4A9),
        )
        self.assertEqual(
            np.asarray(axis._logical_step, dtype=np.float64).view(np.uint64),
            np.uint64(0x3F83187758E9EBB6),
        )
        self.assertEqual(axis._logical_offset, 0)
        self.assertNotEqual(axis[0], start)

    def test_pathological_float16_metadata_survives_field_and_sublattice_slices(self):
        start = np.asarray([0x348C], dtype=np.uint16).view(np.float16)[0]
        step = np.asarray([0x20C6], dtype=np.uint16).view(np.float16)[0]
        axis = LatticeAxis.from_start_step(start, step, 25)
        field = LF[Generic](np.arange(25), (axis,))

        direct = axis[:10]
        through_field = field[:10].L[0]
        through_sublattice = sublattice((axis,), slice(None, 10))[0]
        for candidate in (through_field, through_sublattice):
            np.testing.assert_array_equal(candidate.view(np.uint16), direct.view(np.uint16))
            self.assertEqual(candidate._logical_ref, direct._logical_ref)
            self.assertEqual(candidate._logical_step, direct._logical_step)
            self.assertEqual(candidate._logical_offset, direct._logical_offset)
            np.testing.assert_array_equal(
                ldot((0.03,), (candidate,)).view(np.uint64),
                ldot((0.03,), (direct,)).view(np.uint64),
            )
            np.testing.assert_array_equal(
                lfGaussian(Intensity, (candidate,), 0.3).data.view(np.uint64),
                lfGaussian(Intensity, (direct,), 0.3).data.view(np.uint64),
            )

    def test_float16_padding_and_factor_resampling_match_julia_bits(self):
        axis = LatticeAxis.from_start_step(
            np.float16(-0.3), np.float16(0.2), 4
        )
        padded = padout(axis, 1)
        down = downsample(axis, 2)
        up = upsample(axis, 5)

        np.testing.assert_array_equal(
            padded.view(np.uint16),
            np.asarray([0xB800, 0xB4CD, 0xAE66, 0x2E66, 0x34CD, 0x3800], dtype=np.uint16),
        )
        np.testing.assert_array_equal(
            down.view(np.uint16),
            np.asarray([0xB266, 0x3267], dtype=np.uint16),
        )
        np.testing.assert_array_equal(
            up.view(np.uint64),
            np.asarray(
                [
                    0xBFD8520000000000,
                    0xBFD5C30000000000,
                    0xBFD3340000000000,
                    0xBFD0A50000000000,
                    0xBFCC2C0000000000,
                    0xBFC70E0000000000,
                    0xBFC1F00000000000,
                    0xBFB9A40000000000,
                    0xBFAED00000000000,
                    0xBF94B00000000000,
                    0x3F94400000000000,
                    0x3FAE980000000000,
                    0x3FB9880000000000,
                    0x3FC1E20000000000,
                    0x3FC7000000000000,
                    0x3FCC1E0000000000,
                    0x3FD09E0000000000,
                    0x3FD32D0000000000,
                    0x3FD5BC0000000000,
                    0x3FD84B0000000000,
                ],
                dtype=np.uint64,
            ),
        )
        self.assertEqual(up._step_hint, 0.03997802734375)

    def test_pathological_float16_padding_retains_zero_anchored_range_step(self):
        start = np.asarray([0x348C], dtype=np.uint16).view(np.float16)[0]
        step = np.asarray([0x20C6], dtype=np.uint16).view(np.float16)[0]
        axis = LatticeAxis.from_start_step(start, step, 25)
        padded = padout(axis, (2, 2))
        down = downsample(axis, 5)

        self.assertEqual(padded._logical_ref, 0.265869140625)
        self.assertEqual(padded._logical_step, 0.009324009324009324)
        self.assertEqual(down._logical_ref, 0.30322265625)
        self.assertEqual(down._logical_step, 0.046632124352331605)
        np.testing.assert_array_equal(
            down.view(np.uint16),
            np.asarray([0x34DA, 0x3599, 0x3658, 0x3717, 0x37D6], dtype=np.uint16),
        )

    def test_large_origin_float32_padding_and_factor_resampling_match_julia(self):
        axis = LatticeAxis.from_start_step(
            np.float32(100000), np.float32(0.1), 10
        )
        padded = padout(axis, (2, 2))
        down = downsample(axis, 2)
        up = upsample(axis, 5)

        np.testing.assert_array_equal(
            padded.view(np.uint32),
            np.asarray(
                [
                    0x47C34FE6,
                    0x47C34FF3,
                    0x47C35000,
                    0x47C3500C,
                    0x47C35019,
                    0x47C35026,
                    0x47C35033,
                    0x47C35040,
                    0x47C3504C,
                    0x47C35059,
                    0x47C35066,
                    0x47C35073,
                    0x47C35080,
                    0x47C3508C,
                ],
                dtype=np.uint32,
            ),
        )
        np.testing.assert_array_equal(
            down.view(np.uint32),
            np.asarray(
                [0x47C35006, 0x47C35020, 0x47C35039, 0x47C35053, 0x47C3506C],
                dtype=np.uint32,
            ),
        )
        self.assertEqual(up._step_hint, 0.019999999552965164)
        np.testing.assert_array_equal(
            up[[0, 2, 49]].view(np.uint64),
            np.asarray(
                [0x40F869FF5C28F600, 0x40F86A0000000000, 0x40F86A0F0A3D6B00],
                dtype=np.uint64,
            ),
        )

    def test_axis_copy_cast_empty_natural_range_and_rational_padding(self):
        axis = LatticeAxis.from_start_step(
            np.float16(-0.3), np.float16(0.2), 4
        )
        duplicate = axis.copy()
        self.assertIsInstance(duplicate, LatticeAxis)
        self.assertFalse(duplicate.flags.writeable)
        self.assertEqual(duplicate._logical_ref, axis._logical_ref)
        self.assertEqual(duplicate._logical_step, axis._logical_step)
        for copied in (copy.copy(axis), copy.deepcopy(axis), np.copy(axis, subok=True)):
            self.assertIsInstance(copied, LatticeAxis)
            self.assertFalse(copied.flags.writeable)
            self.assertEqual(copied._logical_ref, axis._logical_ref)
            self.assertEqual(copied._logical_step, axis._logical_step)

        cast = axis.astype(np.float32)
        self.assertIs(type(cast), np.ndarray)
        self.assertFalse(cast.flags.writeable)
        self.assertFalse(hasattr(cast, "_logical_ref"))
        reinterpreted = axis.view(np.uint16)
        self.assertIs(type(reinterpreted), np.ndarray)
        self.assertFalse(reinterpreted.flags.writeable)
        self.assertFalse(hasattr(reinterpreted, "_logical_ref"))
        fancy = axis[[0, 2]]
        self.assertIs(type(fancy), np.ndarray)
        self.assertFalse(fancy.flags.writeable)
        self.assertFalse(hasattr(fancy, "_logical_ref"))

        empty = natrange(0)
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty._step_hint, np.inf)

        rational = padout(np.asarray([1, 2]), 1, Fraction(1, 3))
        self.assertEqual(
            rational.tolist(),
            [Fraction(1, 3), Fraction(1), Fraction(2), Fraction(1, 3)],
        )
        self.assertTrue(all(isinstance(value, Fraction) for value in rational))
        decimal = padout(np.asarray([1, 2]), 1, Decimal("0.1"))
        self.assertEqual(
            decimal.tolist(),
            [Decimal("0.1"), Decimal(1), Decimal(2), Decimal("0.1")],
        )
        self.assertTrue(all(isinstance(value, Decimal) for value in decimal))

    def test_lattice_axis_scalar_ufuncs_preserve_only_range_operations(self):
        axis = LatticeAxis(
            np.asarray([-0.3, -0.1, 0.1, 0.3], dtype=np.float16),
            step_hint=np.float16(0.2),
        )
        scaled = axis * 0.03
        self.assertIsInstance(scaled, LatticeAxis)
        self.assertFalse(scaled.flags.writeable)
        self.assertEqual(scaled._logical_ref, -0.003)
        self.assertEqual(scaled._logical_step, 0.006)
        self.assertEqual(scaled._logical_offset, 1)
        np.testing.assert_array_equal(
            scaled,
            np.asarray(
                [-0.009000000000000001, -0.003, 0.003, 0.009000000000000001]
            ),
        )

        reflected = 0.5 - axis
        self.assertIsInstance(reflected, LatticeAxis)
        self.assertEqual(reflected._logical_step, -0.2)
        self.assertIsInstance(-axis, LatticeAxis)

        self.assertNotIsInstance(np.square(axis), LatticeAxis)
        self.assertNotIsInstance(1 / axis, LatticeAxis)
        self.assertNotIsInstance(np.add(axis, axis), LatticeAxis)

    def test_general_range_affine_paths_match_live_julia_bits(self):
        # Live Julia 1.12.6 probes.  These exercise three distinct Base range
        # representations: UnitRange{Int64}, low-float StepRangeLen with
        # Float64 ref/step, and Float64 TwicePrecision StepRangeLen.
        dot = ldot((np.float16(0.001),), (range(1, 12),))
        np.testing.assert_array_equal(
            dot.view(np.uint16),
            np.asarray(
                [
                    0x1419,
                    0x1819,
                    0x1A26,
                    0x1C19,
                    0x1D1F,
                    0x1E26,
                    0x1F2C,
                    0x2019,
                    0x209C,
                    0x211F,
                    0x21A3,
                ],
                dtype=np.uint16,
            ),
        )

        source = LatticeAxis.from_start_step(
            np.float16(0), np.float16(0.001), 6
        )
        dual = dualShiftLattice((source,))[0]
        np.testing.assert_array_equal(
            dual.view(np.uint16),
            np.asarray(
                [0xDFCF, 0xDD35, 0xD936, 0xB400, 0x5932, 0x5D33],
                dtype=np.uint16,
            ),
        )

        natural = natrange(10)
        np.testing.assert_array_equal(
            natural.view(np.uint64),
            np.asarray(
                [
                    0xBFF94C583ADA5B52,
                    0xBFF43D136248490F,
                    0xBFEE5B9D136C6D95,
                    0xBFE43D136248490F,
                    0xBFD43D136248490F,
                    0x0000000000000000,
                    0x3FD43D136248490F,
                    0x3FE43D136248490F,
                    0x3FEE5B9D136C6D95,
                    0x3FF43D136248490F,
                ],
                dtype=np.uint64,
            ),
        )

        ordinal = LatticeAxis.from_start_step(
            np.int64(-3), np.int64(1), 7
        )
        np.testing.assert_array_equal(
            (ordinal + np.float16(0.3)).view(np.uint16),
            np.asarray(
                [
                    0xC166,
                    0xBECC,
                    0xB998,
                    0x34D0,
                    0x3D34,
                    0x409A,
                    0x429A,
                ],
                dtype=np.uint16,
            ),
        )
        np.testing.assert_array_equal(
            (np.float16(0.3) - ordinal).view(np.uint16),
            np.asarray(
                [
                    0x429A,
                    0x409A,
                    0x3D33,
                    0x34CD,
                    0xB99A,
                    0xBECD,
                    0xC166,
                ],
                dtype=np.uint16,
            ),
        )

    def test_float64_twiceprecision_pad_and_factor_ranges_match_julia(self):
        axis = LatticeAxis.from_start_step(-0.3, 0.2, 4)
        np.testing.assert_array_equal(
            padout(axis, 1).view(np.uint64),
            np.asarray(
                [
                    0xBFE0000000000000,
                    0xBFD3333333333333,
                    0xBFB999999999999A,
                    0x3FB999999999999A,
                    0x3FD3333333333333,
                    0x3FE0000000000000,
                ],
                dtype=np.uint64,
            ),
        )
        np.testing.assert_array_equal(
            downsample(axis, 2).view(np.uint64),
            np.asarray(
                [0xBFC999999999999A, 0x3FC9999999999999],
                dtype=np.uint64,
            ),
        )
        np.testing.assert_array_equal(
            upsample(axis, 3).view(np.uint64),
            np.asarray(
                [
                    0xBFD7777777777777,
                    0xBFD3333333333333,
                    0xBFCDDDDDDDDDDDDD,
                    0xBFC5555555555554,
                    0xBFB9999999999998,
                    0xBFA1111111111110,
                    0x3FA1111111111118,
                    0x3FB999999999999C,
                    0x3FC5555555555556,
                    0x3FCDDDDDDDDDDDDE,
                    0x3FD3333333333333,
                    0x3FD7777777777777,
                ],
                dtype=np.uint64,
            ),
        )

    def test_range_slice_copy_and_overflow_metadata_are_safe(self):
        low = LatticeAxis.from_start_step(
            np.float16(-0.3), np.float16(0.2), 4
        )
        empty = low[0:0:2]
        self.assertEqual(empty._logical_ref, -0.30000000000000004)
        self.assertEqual(empty._logical_step, 0.4)
        self.assertEqual(empty._logical_offset, 0)

        twice = LatticeAxis.from_start_step(-0.3, 0.2, 4)
        sliced = twice[1::2]
        np.testing.assert_array_equal(
            sliced.view(np.uint64),
            np.asarray(
                [0xBFB999999999999A, 0x3FD3333333333333],
                dtype=np.uint64,
            ),
        )
        self.assertEqual(sliced._logical_step.hi, 0.4)
        self.assertEqual(
            sliced._logical_step.lo, -2.2204460492503126e-17
        )

        raw_copy = np.array(low, copy=True, subok=True)
        self.assertTrue(raw_copy.flags.writeable)
        self.assertIsNone(raw_copy._logical_ref)
        self.assertIsNone(raw_copy._logical_step)
        raw_copy[0] = np.float16(9)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            overflowing = LatticeAxis.from_start_step(
                np.float16(60000),
                np.float16(60000),
                4,
            )
        self.assertTrue(np.isinf(overflowing[-1]))
        self.assertGreater(overflowing[-1], 0)

        large_start = np.asarray(
            [0x15D3], dtype=np.uint16
        ).view(np.float16)[0]
        large_step = np.asarray(
            [0x79F6], dtype=np.uint16
        ).view(np.float16)[0]
        large_axis = LatticeAxis.from_start_step(
            large_start, large_step, 16
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            large_slice = large_axis[1::2]
        self.assertTrue(np.isinf(large_slice._step_hint))

    def test_integer_range_storage_and_unsigned_wrap_match_julia(self):
        signed = LatticeAxis.from_start_step(
            np.int8(2), np.int8(3), 6
        )
        self.assertEqual(signed.dtype, np.dtype(np.int64))
        self.assertEqual(np.asarray(signed._logical_step).dtype, np.dtype(np.int8))
        np.testing.assert_array_equal(signed, [2, 5, 8, 11, 14, 17])

        narrow_unsigned = LatticeAxis.from_start_step(
            np.uint8(2), np.uint8(3), 6
        )
        self.assertEqual(narrow_unsigned.dtype, np.dtype(np.int64))
        self.assertEqual(
            np.asarray(narrow_unsigned._logical_step).dtype,
            np.dtype(np.uint8),
        )

        wrapped = LatticeAxis.from_start_step(
            np.uint64(7), np.int8(-2), 5
        )
        self.assertEqual(wrapped.dtype, np.dtype(np.uint64))
        self.assertEqual(
            np.asarray(wrapped._logical_step).dtype,
            np.dtype(np.int8),
        )
        np.testing.assert_array_equal(
            wrapped,
            np.asarray(
                [7, 5, 3, 1, np.iinfo(np.uint64).max],
                dtype=np.uint64,
            ),
        )

        boolean = LatticeAxis.from_start_step(False, True, 4)
        self.assertEqual(boolean.dtype, np.dtype(np.int64))
        self.assertEqual(
            np.asarray(boolean._logical_step).dtype,
            np.dtype(np.bool_),
        )
        np.testing.assert_array_equal(boolean, [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "step cannot be zero"):
            LatticeAxis.from_start_step(np.uint64(2), np.uint64(0), 3)

    def test_empty_range_consumers_and_uint64_dual_match_julia(self):
        empty = LatticeAxis.from_start_step(
            np.float32(1), np.float32(0.1), 0
        )
        with self.assertRaises(IndexError):
            downsample(empty, 1)
        with self.assertRaises(IndexError):
            upsample(empty, 1)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            positive = dualLattice((empty,))[0]
            shifted = dualShiftLattice((empty,))[0]
        for axis in (positive, shifted):
            self.assertEqual(axis.dtype, np.dtype(np.float32))
            self.assertEqual(len(axis), 0)
            self.assertTrue(np.isinf(axis._step_hint))

        source = LatticeAxis.from_start_step(
            np.float32(-0.3), np.float32(0.2), 6
        )
        # Julia's UInt64 multiplication wraps the negative shifted-frequency
        # indices before division. Preserve this surprising Base behavior.
        unsigned = dualShiftLattice((source,), np.uint64(2))[0]
        np.testing.assert_array_equal(
            unsigned.view(np.uint32),
            np.full(6, 0x5F555555, dtype=np.uint32),
        )

    def test_shifted_fft_matches_definition_and_round_trip(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
        expected = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data)))
        np.testing.assert_allclose(sft(data), expected)
        np.testing.assert_allclose(isft(sft(data)), data, atol=1e-12)

        lattice = natlat(data.shape)
        field = LF[ComplexAmplitude](data, lattice, 1.5)
        transformed = sft(field)
        self.assertIs(transformed.field_type, ComplexAmplitude)
        np.testing.assert_allclose(isft(transformed).data, data, atol=1e-12)

        rational = np.array([Fraction(1), Fraction(2)], dtype=object)
        rational_transform = sft(rational)
        self.assertEqual(rational_transform.dtype, np.dtype(np.complex128))
        np.testing.assert_array_equal(rational_transform, [1.0, 3.0])

        rational_field = LF[ComplexAmplitude, object, 1](
            rational, (range(2),)
        )
        transformed_field = sft(rational_field)
        self.assertEqual(transformed_field.dtype, np.dtype(np.complex128))
        np.testing.assert_array_equal(transformed_field.data, [1.0, 3.0])


class BasisAndWignerTests(unittest.TestCase):
    def test_shifted_dft_is_unitary(self):
        for n in (1, 4, 5):
            basis = shiftedDFTBasis(n)
            np.testing.assert_allclose(
                basis.conj().T @ basis, np.eye(n), atol=2e-14
            )

    def test_unusable_upstream_hermite_and_fractional_fourier_bases(self):
        with self.assertRaisesRegex(NotImplementedError, "audited Julia"):
            hermiteBasis(5)
        with self.assertRaisesRegex(NotImplementedError, "hermiteBasis"):
            FrFTBasis(5, 0.73)

    def test_fractional_fourier_basis_accepts_bigfloat_like_alpha(self):
        with localcontext() as context:
            context.prec = 77
            with self.assertRaisesRegex(NotImplementedError, "hermiteBasis"):
                FrFTBasis(3, Decimal("0.73"))

    def test_wigner_fft_shape_realness_and_scaling(self):
        signal = np.array([1 + 2j, 2 - 1j, -0.5 + 0.25j, 3 + 0j])
        result = wigner_fft(signal)
        self.assertEqual(result.shape, (4, 8))
        self.assertTrue(np.isrealobj(result))
        np.testing.assert_allclose(wigner_fft(2 * signal), 4 * result, atol=1e-12)

        # Direct golden from Julia's ToeplitzMatrices/FFTW implementation.
        short = np.array([1 + 2j, -0.5 + 0.3j, 2 - 1j])
        expected_fortran = np.array(
            [
                0.8333333333333333,
                0.05666666666666666,
                0.8333333333333333,
                0.8333333333333333,
                1.5000423396407307,
                0.8333333333333333,
                0.8333333333333333,
                1.5000423396407307,
                0.8333333333333333,
                0.8333333333333333,
                0.05666666666666666,
                0.8333333333333333,
                0.8333333333333333,
                -1.3867090063073975,
                0.8333333333333333,
                0.8333333333333333,
                -1.3867090063073975,
                0.8333333333333333,
            ]
        )
        np.testing.assert_allclose(
            wigner_fft(short).ravel(order="F"), expected_fortran, atol=5e-16
        )


if __name__ == "__main__":
    unittest.main()
