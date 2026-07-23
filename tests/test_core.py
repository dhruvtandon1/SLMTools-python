import copy
import unittest

import numpy as np

from slmtools.lattice_field import (
    Amplitude,
    ComplexAmp,
    ComplexAmplitude,
    ComplexPhase,
    DimensionMismatch,
    DomainError,
    Generic,
    Intensity,
    LF,
    Modulus,
    RealAmp,
    RealAmplitude,
    RealPhase,
    S1Phase,
    UPhase,
    UnwrappedPhase,
    elq,
    normalizeLF,
    phasor,
    square,
    subfield,
    sublattice,
    wrap,
)
from slmtools.lattice_utils import (
    Nyquist,
    _step,
    latticeDisplacement,
    ldot,
    naturalize,
    natlat,
    natrange,
    padout,
    r2,
    toDim,
)


class LatticeFieldCoreTests(unittest.TestCase):
    def setUp(self):
        self.lattice = (range(1, 4), range(10, 14))
        self.data = np.arange(1, 13, dtype=float).reshape((3, 4), order="F")

    def test_tag_hierarchy_and_aliases(self):
        self.assertTrue(issubclass(Modulus, Amplitude))
        self.assertIs(UPhase, RealPhase)
        self.assertIs(UnwrappedPhase, RealPhase)
        self.assertIs(S1Phase, ComplexPhase)
        self.assertIs(RealAmplitude, Modulus)
        self.assertIs(RealAmp, Modulus)
        self.assertIs(ComplexAmp, ComplexAmplitude)

    def test_constructor_semantics_and_shape_validation(self):
        intensity = LF[Intensity](np.array([-1.0, 2.0]), (range(2),))
        np.testing.assert_array_equal(intensity.data, [0.0, 2.0])
        complex_field = LF[ComplexAmplitude](
            np.array([1.0, 2.0], dtype=np.float32), (range(2),)
        )
        self.assertEqual(complex_field.dtype, np.dtype(np.complex128))
        with self.assertRaises(DimensionMismatch):
            LF[RealPhase](np.zeros((2, 2)), (range(3), range(2)))

        # Julia's constructor-from-field calls the full constructor and bypasses
        # Intensity clipping.
        inherited = LF[Intensity](np.array([-2.0, 1.0]), intensity)
        np.testing.assert_array_equal(inherited.data, [-2.0, 1.0])
        # That overload has exactly two arguments; explicitly spelling the
        # unrelated lattice constructor's default does not match it.
        with self.assertRaises(TypeError):
            LF[Intensity](np.array([1.0, 2.0]), intensity, 1.0)
        with self.assertRaises(TypeError):
            LF[Intensity](
                np.array([1.0, 2.0]), intensity, flambda=1.0
            )

    def test_metadata_is_immutable_but_data_is_mutable(self):
        field = LF[RealPhase](self.data.copy(), self.lattice, 1.5)
        field.data[0, 0] = 99
        self.assertEqual(field.data[0, 0], 99)
        with self.assertRaises(AttributeError):
            field.flambda = 2.0
        with self.assertRaises(AttributeError):
            field.L = natlat(field.shape)
        with self.assertRaises(ValueError):
            field.L[0][0] = 7

    def test_public_data_array_uses_julia_checked_assignment(self):
        integer = LF[Generic, np.int64, 1](
            np.asarray([2], dtype=np.int64), (range(1),)
        )
        for assign in (
            lambda: integer.data.__setitem__(0, 1.5),
            lambda: integer.data.flat.__setitem__(0, 1.5),
            lambda: integer.data.fill(1.5),
            lambda: integer.data.put([0], [1.5]),
        ):
            with self.assertRaises(ValueError):
                assign()
            np.testing.assert_array_equal(integer.data, [2])

        real = LF[Generic, np.float64, 1](
            np.asarray([2.0]), (range(1),)
        )
        with self.assertRaises(ValueError):
            real.data[0] = 1.0 + 2.0j
        np.testing.assert_array_equal(real.data, [2.0])

    def test_full_typed_constructor_and_copy_follow_julia_partial_copy(self):
        unusual = LF[ComplexAmplitude, np.complex64, 1](
            np.array([1, 2], dtype=np.complex64), (range(2),)
        )
        duplicate = copy.copy(unusual)
        self.assertEqual(duplicate.dtype, np.dtype(np.complex128))
        self.assertIs(duplicate.field_type, ComplexAmplitude)
        self.assertIsNot(duplicate.data, unusual.data)

        negative = LF[Intensity, np.float32, 1](
            np.array([-2.0, 1.0], dtype=np.float32), (range(2),)
        )
        np.testing.assert_array_equal(negative.data, [-2.0, 1.0])
        np.testing.assert_array_equal(copy.copy(negative).data, [0.0, 1.0])

        with self.assertRaises(TypeError):
            LF[Intensity, np.float64, 1](
                np.array([-2.0, 1.0], dtype=np.float32), (range(2),)
            )
        with self.assertRaises(TypeError):
            LF[Intensity, np.float32, 1](
                np.array([1.0], dtype=np.float32), (range(1),), 1.0j
            )
        with self.assertRaises(TypeError):
            LF[Intensity](np.array([-1.0 + 2.0j]), (range(1),))

    def test_elq_and_fortran_linear_indexing(self):
        field = LF[RealPhase](self.data.copy(), self.lattice, 1.5)
        other = LF[RealPhase](self.data.copy(), self.lattice, 1.5)
        self.assertIsNone(elq(field, other))
        self.assertEqual(field[1], self.data[1, 0])
        field[1] = -4
        self.assertEqual(field.data[1, 0], -4)
        with self.assertRaises(DomainError):
            elq(self.lattice, (range(1, 3), range(10, 14)))

    def test_direct_indexing_requires_full_cartesian_arity(self):
        field = LF[RealPhase](self.data.copy(), self.lattice)

        # A bare integer is the documented Julia/Fortran linear overload.
        self.assertEqual(field[1], self.data[1, 0])
        # A trailing comma is still a one-integer call in Python and therefore
        # follows that same linear overload.
        self.assertEqual(field[(1,)], self.data[1, 0])
        # Every Cartesian/range spelling supplies one selector per dimension;
        # omitted axes are not silently padded like a raw NumPy array.
        with self.assertRaises(DimensionMismatch):
            _ = field[:2]
        with self.assertRaises(DimensionMismatch):
            _ = field[(slice(0, 2),)]

        region = field[:2, 1:3]
        self.assertEqual(region.shape, (2, 2))
        np.testing.assert_array_equal(region.data, self.data[:2, 1:3])

    def test_direct_indexing_uses_julia_trailing_singleton_rules(self):
        data = np.arange(6).reshape((2, 3, 1), order="F")
        field = LF[Generic](data, (range(2), range(3), range(1)))

        # Dense Julia indexing permits omitted trailing singleton dimensions
        # for integer and mixed integer/range calls.
        self.assertEqual(field[1, 1], data[1, 1, 0])
        np.testing.assert_array_equal(field[1, :].data, data[1, :, 0])
        np.testing.assert_array_equal(field[:, 1].data, data[:, 1, 0])
        np.testing.assert_array_equal(field[:, :, 0, 0].data, data[:, :, 0])
        self.assertEqual(field[1, 1, 0, 0], data[1, 1, 0])

        # The all-range overload calls Julia's exact-arity sublattice helper,
        # and omitted non-singleton dimensions are never implicitly filled.
        with self.assertRaises(DimensionMismatch):
            _ = field[:, :]
        nonsingleton = LF[Generic](
            np.arange(24).reshape((2, 3, 4), order="F"),
            (range(2), range(3), range(4)),
        )
        with self.assertRaises(IndexError):
            _ = nonsingleton[1, :]
        with self.assertRaises(IndexError):
            _ = field[1, 1, 0, 1]

    def test_direct_boolean_indices_are_rejected(self):
        field = LF[Generic](self.data.astype(np.int64), self.lattice)
        original = field.data.copy()
        for key in (True, np.bool_(False), (True,), (0, np.bool_(True))):
            with self.assertRaises(TypeError):
                _ = field[key]
            with self.assertRaises(TypeError):
                field[key] = 99
        np.testing.assert_array_equal(field.data, original)

    def test_direct_assignment_is_scalar_arity_checked_and_inexact_safe(self):
        field = LF[Generic](
            np.arange(6, dtype=np.int64).reshape((2, 3), order="F"),
            (range(2), range(3)),
        )
        original = field.data.copy()

        with self.assertRaisesRegex(ValueError, "Inexact assignment"):
            field[0] = 1.5
        np.testing.assert_array_equal(field.data, original)
        with self.assertRaisesRegex(ValueError, "Inexact assignment"):
            field[0, 0] = 1.0 + 2.0j
        np.testing.assert_array_equal(field.data, original)

        # A one-item integer tuple is the same one-argument linear call as a
        # bare integer; incomplete Cartesian/range assignments still fail.
        with self.assertRaises(DimensionMismatch):
            field[:1] = 7
        with self.assertRaises(TypeError):
            field[:, 0] = 7
        np.testing.assert_array_equal(field.data, original)

        # Exactly representable scalar conversions and both Julia integer
        # indexing overloads remain available.
        field[(0,)] = 7
        field[0] = 7.0
        field[1, 2] = np.int16(9)
        self.assertEqual(field.data[0, 0], 7)
        self.assertEqual(field.data[1, 2], 9)

    def test_slicing_drops_scalar_axes_and_handles_descending_ranges(self):
        field = LF[RealPhase](self.data.copy(), self.lattice)
        region = field[1:, ::-1]
        self.assertEqual(region.shape, (2, 4))
        np.testing.assert_array_equal(region.data, self.data[1:, ::-1])
        np.testing.assert_array_equal(region.L[0], [2, 3])
        np.testing.assert_array_equal(region.L[1], [13, 12, 11, 10])
        row = subfield(field, 1, slice(1, 4))
        self.assertEqual(row.shape, (3,))
        np.testing.assert_array_equal(row.L[0], [11, 12, 13])
        one_point_axes = sublattice(field.L, 1, slice(1, 3))
        self.assertEqual(tuple(map(len, one_point_axes)), (1, 2))
        self.assertEqual(_step(one_point_axes[0]), 1)

        # A single tuple of range-like selectors is the Python literal-box
        # counterpart of Julia's CartesianIndices overload.
        boxed_axes = sublattice(
            field.L, (slice(0, 2), range(1, 4))
        )
        np.testing.assert_array_equal(boxed_axes[0], [1, 2])
        np.testing.assert_array_equal(boxed_axes[1], [11, 12, 13])

    def test_subfield_matches_julia_vararg_index_arity(self):
        data = np.arange(1, 25).reshape((2, 3, 4), order="F")
        field = LF[Generic](data, (range(2), range(3), range(4)))

        self.assertEqual(subfield(field, 1), 2)
        self.assertEqual(subfield(field, 1, 1, 1), 10)
        self.assertEqual(subfield(field, 0, 0, 0, 0), 1)
        with self.assertRaises(TypeError):
            subfield(field)
        with self.assertRaises(IndexError):
            subfield(field, 1, 1)
        with self.assertRaises(DimensionMismatch):
            subfield(field, slice(0, 2))

        sliced = subfield(field, 1, slice(0, 2), slice(0, 3))
        self.assertEqual(sliced.shape, (2, 3))
        np.testing.assert_array_equal(sliced.data, [[2, 8, 14], [4, 10, 16]])

        singleton_tail = LF[Generic](
            data[:, :, :1], (range(2), range(3), range(1))
        )
        shortened = subfield(singleton_tail, 1, slice(0, 2))
        np.testing.assert_array_equal(shortened.data, [2, 4])

    def test_phase_amplitude_arithmetic(self):
        lattice = (range(3),)
        modulus = LF[Modulus](np.array([1.0, 2.0, 3.0]), lattice)
        phase = LF[RealPhase](np.array([0.0, 0.25, 0.5]), lattice)
        amplitude = modulus * phase
        self.assertIs(amplitude.field_type, ComplexAmplitude)
        np.testing.assert_allclose(
            amplitude.data,
            np.array([1.0, 2.0j, -3.0], dtype=complex),
            atol=1e-14,
        )
        np.testing.assert_allclose(abs(amplitude).data, modulus.data)
        np.testing.assert_allclose(square(amplitude).data, modulus.data**2)
        wrapped = wrap(phase)
        self.assertIs(wrapped.field_type, ComplexPhase)
        self.assertIs(wrap(wrapped).data, wrapped.data)
        np.testing.assert_allclose(phasor(amplitude).data, wrapped.data)

    def test_normalization(self):
        lattice = (range(3),)
        intensity = LF[Intensity](np.array([1.0, 2.0, 3.0]), lattice)
        self.assertAlmostEqual(float(np.sum(normalizeLF(intensity).data)), 1.0)
        amplitude = LF[Modulus](np.array([1.0, 2.0, 3.0]), lattice)
        self.assertAlmostEqual(
            float(np.sum(np.abs(normalizeLF(amplitude).data) ** 2)), 1.0
        )


class LatticeUtilityTests(unittest.TestCase):
    def test_natural_lattices_and_singleton_step(self):
        np.testing.assert_allclose(natrange(5), np.arange(-2, 3) / np.sqrt(5))
        lattice = natlat((1, 4))
        self.assertEqual(Nyquist(lattice), (0.5, 1.0))
        field = LF[RealPhase](np.zeros((1, 4)), lattice, 8.0)
        made_natural = naturalize(field)
        self.assertEqual(made_natural.flambda, 1.0)
        np.testing.assert_allclose(made_natural.L[1], lattice[1])

        # Range coordinates accumulate several ulps of rounding at realistic
        # sizes; retained range metadata must keep the logical step exact.
        axis32 = natlat((32,))[0]
        self.assertAlmostEqual(_step(axis32), 1 / np.sqrt(32))
        self.assertAlmostEqual(_step(axis32[::2]), 2 / np.sqrt(32))
        np.testing.assert_allclose(padout(natlat((1,))[0], 1), [-1, 0, 1])

    def test_coordinate_grid_helpers(self):
        lattice = (range(-1, 2), range(2, 6))
        expected_r2 = np.asarray(lattice[0])[:, None] ** 2 + np.asarray(
            lattice[1]
        )[None, :] ** 2
        np.testing.assert_array_equal(r2(lattice), expected_r2)
        np.testing.assert_array_equal(
            ldot((2.0, -1.0), lattice),
            2 * np.asarray(lattice[0])[:, None]
            - np.asarray(lattice[1])[None, :],
        )
        np.testing.assert_array_equal(toDim([1, 2, 3], 2, 3).shape, (1, 3, 1))
        np.testing.assert_array_equal(latticeDisplacement(lattice), [0, 4])

        matrix = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(
            toDim(matrix, 1, 1), np.array([1, 3, 2, 4])
        )

    def test_padout_axis_array_lattice_and_field(self):
        axis = padout(range(2, 5), (2, 1))
        np.testing.assert_array_equal(axis, [0, 1, 2, 3, 4, 5])
        array = np.array([[1, 2], [3, 4]])
        padded = padout(array, (1, 2), filler=-1)
        self.assertEqual(padded.shape, (4, 6))
        np.testing.assert_array_equal(padded[1:3, 2:4], array)
        lattice = (range(2), range(10, 12))
        padded_lattice = padout(lattice, (1, 2))
        self.assertEqual(tuple(map(len, padded_lattice)), (4, 6))
        field = LF[RealPhase](array.astype(float), lattice)
        padded_field = padout(field, (1, 2))
        self.assertEqual(padded_field.shape, (4, 6))


if __name__ == "__main__":
    unittest.main()
