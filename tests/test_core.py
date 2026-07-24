import copy
from decimal import Decimal
import gc
import unittest
from unittest import mock

import numpy as np
import slmtools.lattice_field as lattice_field_module

from slmtools._bigfloat import (
    _MPFR,
    _to_mpfr,
)
from slmtools.lattice_field import (
    _DecimalComplex,
    _FieldStorageState,
    Amplitude,
    ComplexAmp,
    ComplexAmplitude,
    ComplexPhase,
    DimensionMismatch,
    DomainError,
    Generic,
    Intensity,
    LatticeAxis,
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
from slmtools.misc import safeInverse


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
        field.data += 1
        self.assertEqual(field.data[0, 0], 100)
        with self.assertRaises(AttributeError):
            field.data = field.data[:, :]
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

    def test_general_constructor_retains_the_input_array_alias(self):
        source = np.asarray([1.0, 2.0])
        field = LF[Generic](source, (range(2),))
        retained = field.data
        detached = np.asarray(retained)

        source[0] = 7.0
        self.assertEqual(field.data[0], 7.0)
        self.assertEqual(retained[0], 7.0)
        self.assertEqual(detached[0], 1.0)

        field.data[1] = 9.0
        np.testing.assert_array_equal(source, [7.0, 9.0])

        intensity_source = np.asarray([-1.0, 2.0])
        intensity = LF[Intensity](intensity_source, (range(2),))
        intensity_source[:] = 5.0
        np.testing.assert_array_equal(intensity.data, [0.0, 2.0])

        phase_source = np.asarray([0.0, 0.25])
        phase = LF[RealPhase](phase_source, (range(2),))
        phase_source[1] = 0.5
        self.assertEqual(phase.data[1], 0.5)

        full_source = np.asarray([-2.0, 1.0], dtype=np.float32)
        full_intensity = LF[Intensity, np.float32, 1](
            full_source, (range(2),)
        )
        full_source[0] = -3.0
        self.assertEqual(full_intensity.data[0], np.float32(-3.0))

        inherited_source = np.asarray([3.0, 4.0])
        inherited = LF[RealPhase](inherited_source, field)
        inherited_source[0] = 8.0
        self.assertEqual(inherited.data[0], 8.0)

        amplitude_source = np.asarray([1.0, 2.0])
        amplitude = LF[ComplexAmplitude](
            amplitude_source, (range(2),)
        )
        amplitude_source[:] = 9.0
        np.testing.assert_array_equal(amplitude.data, [1.0 + 0j, 2.0 + 0j])

    def test_checked_data_advanced_and_julia_shaped_assignment(self):
        field = LF[Generic, np.int64, 2](
            np.zeros((2, 3), dtype=np.int64),
            (range(2), range(3)),
        )
        # Julia: zeros(Int, 2, 3)[[1,2], [2,3]] = [7 9; 8 10].
        # Multiple vector selectors form a Cartesian product rather than
        # NumPy's elementwise paired advanced index.
        field.data[[0, 1], [1, 2]] = [[7, 9], [8, 10]]
        np.testing.assert_array_equal(
            field.data.copy(),
            [[0, 7, 9], [0, 8, 10]],
        )
        self.assertTrue(
            np.array_equal(
                field.data,
                np.asarray([[0, 7, 9], [0, 8, 10]]),
            )
        )
        selected = field.data[[0, 1], [1, 2]]
        self.assertEqual(selected.shape, (2, 2))
        np.testing.assert_array_equal(selected, [[7, 9], [8, 10]])

        three_dimensional = LF[Generic](
            np.arange(1, 25).reshape((2, 3, 4), order="F"),
            (range(2), range(3), range(4)),
        )
        separated = three_dimensional.data[
            [0, 1], :, [0, 1]
        ]
        self.assertEqual(separated.shape, (2, 3, 2))
        np.testing.assert_array_equal(
            separated.ravel(order="F"), np.arange(1, 13)
        )

        matrix = LF[Generic](
            np.arange(1, 17).reshape((4, 4), order="F"),
            (range(4), range(4)),
        )
        matrix_indices = np.asarray([[0, 1], [2, 3]])
        vector_indices = np.asarray([0, 1])
        selected_blocks = matrix.data[matrix_indices, vector_indices]
        self.assertEqual(selected_blocks.shape, (2, 2, 2))
        np.testing.assert_array_equal(
            selected_blocks,
            np.asarray(
                [
                    [[1, 5], [2, 6]],
                    [[3, 7], [4, 8]],
                ]
            ),
        )

        singleton_tail = LF[Generic](
            np.arange(6).reshape((2, 3, 1), order="F"),
            (range(2), range(3), range(1)),
        )
        np.testing.assert_array_equal(
            singleton_tail.data[1, :], [1, 3, 5]
        )
        np.testing.assert_array_equal(
            singleton_tail.data[:, 1], [2, 3]
        )
        non_square_mask = np.asarray(
            [[True, False, False], [False, True, False]]
        )
        np.testing.assert_array_equal(
            field.data[non_square_mask],
            field.data.copy().ravel(order="F")[
                non_square_mask.ravel(order="F")
            ],
        )
        with self.assertRaises(IndexError):
            _ = field.data[non_square_mask.T]

        logical_axes = LF[Generic](
            np.arange(1, 7).reshape((2, 3), order="F"),
            (range(2), range(3)),
        )
        np.testing.assert_array_equal(
            logical_axes.data[
                [True, False], [True, False, True]
            ],
            [[1, 5]],
        )

        mask = np.asarray(
            [[True, False, False], [False, True, False]]
        )
        field.data[mask] = [11, 12]
        np.testing.assert_array_equal(
            field.data.copy(),
            [[11, 7, 9], [0, 12, 10]],
        )

        before = field.data.copy()
        with self.assertRaises(DimensionMismatch):
            field.data[:, :] = np.asarray([[1], [2]])
        np.testing.assert_array_equal(field.data.copy(), before)

        # Base.setindex_shape_check accepts singleton rearrangements that
        # retain linear order; it does not apply NumPy broadcasting.
        field.data[:, 0] = np.asarray([[3], [4]])
        np.testing.assert_array_equal(field.data[:, 0], [3, 4])
        field.data[:, :2] = np.asarray([[1, 2, 3, 4]])
        np.testing.assert_array_equal(
            field.data.copy(),
            [[1, 3, 9], [2, 4, 10]],
        )
        with self.assertRaises(DimensionMismatch):
            field.data[:, :2] = np.asarray([[1], [2], [3], [4]])

        # Failed conversion retains only the column-major prefix that Julia
        # stored before raising InexactError.
        with self.assertRaises(ValueError):
            field.data[:, :2] = np.asarray([[5.0, 6.5, 7.0, 8.0]])
        np.testing.assert_array_equal(
            field.data.copy(),
            [[5, 3, 9], [2, 4, 10]],
        )

    def test_inherited_ndarray_out_stages_checked_partial_mutation(self):
        source = LF[Generic](
            np.asarray([1.0, 1.5, 2.0]), (range(3),)
        )
        integer_output = LF[Generic, np.int64, 1](
            np.asarray([9, 9, 9], dtype=np.int64), (range(3),)
        )
        with self.assertRaises(ValueError):
            source.data.cumsum(out=integer_output.data)
        np.testing.assert_array_equal(integer_output.data, [1, 9, 9])

        floating_output = LF[Generic](
            np.zeros(3, dtype=np.float64), (range(3),)
        )
        checked_output = floating_output.data
        returned = source.data.cumsum(out=checked_output)
        self.assertIs(returned, checked_output)
        np.testing.assert_array_equal(
            floating_output.data, [1.0, 2.5, 4.5]
        )

        scalar_output = LF[Generic, np.float64, 0](
            np.zeros((), dtype=np.float64), ()
        )
        scalar_facade = scalar_output.data
        self.assertIs(source.data.sum(out=scalar_facade), scalar_facade)
        self.assertEqual(scalar_output.data[()], 4.5)

        def integer_vector():
            return LF[Generic, np.int64, 1](
                np.asarray([9, 9, 9], dtype=np.int64), (range(3),)
            )

        for operation in (
            lambda output: source.data.compress(
                [True, True, True], None, output.data
            ),
            lambda output: np.compress(
                [True, True, True], source.data, None, output.data
            ),
            lambda output: source.data.clip(0, 3, output.data),
            lambda output: np.clip(source.data, 0, 3, output.data),
        ):
            output = integer_vector()
            with self.assertRaises(ValueError):
                operation(output)
            np.testing.assert_array_equal(output.data, [1, 9, 9])

        integer_scalar = LF[Generic, np.int64, 0](
            np.asarray(9, dtype=np.int64), ()
        )
        for operation in (
            lambda: source.data.sum(None, None, integer_scalar.data),
            lambda: np.add.reduce(
                source.data, None, None, integer_scalar.data
            ),
        ):
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(integer_scalar.data[()], 9)

        trace_source = LF[Generic](
            np.asarray([[1.0, 0.0], [0.0, 1.5]]),
            (range(2), range(2)),
        )
        for operation in (
            lambda: trace_source.data.trace(
                0, 0, 1, None, integer_scalar.data
            ),
            lambda: np.trace(
                trace_source.data, 0, 0, 1, None, integer_scalar.data
            ),
        ):
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(integer_scalar.data[()], 9)

    def test_decimal_complex_zero_and_nontrapping_field_division(self):
        zero = _DecimalComplex(Decimal(0), Decimal(0))
        self.assertEqual(zero, 0)
        inverse_magnitude = abs(safeInverse(_DecimalComplex(Decimal(0))))
        self.assertIsInstance(inverse_magnitude, _MPFR)
        self.assertEqual(inverse_magnitude, 0)
        self.assertEqual(_DecimalComplex(Decimal(1)), 1)
        self.assertNotEqual(_DecimalComplex(Decimal(0), Decimal(1)), 0)

        real = LF[Generic](
            np.asarray([Decimal(1), Decimal(0)], dtype=object),
            (range(2),),
        )
        real_result = real / Decimal(0)
        self.assertEqual(real_result.data[0], Decimal("Infinity"))
        self.assertTrue(real_result.data[1].is_nan())

        complex_values = np.asarray(
            [
                _DecimalComplex(Decimal(1), Decimal(2)),
                _DecimalComplex(Decimal(0), Decimal(0)),
            ],
            dtype=object,
        )
        complex_result = (
            LF[Generic](complex_values, (range(2),)) / Decimal(0)
        )
        self.assertTrue(complex_result.data[0].real.is_infinite())
        self.assertTrue(complex_result.data[0].imag.is_infinite())
        self.assertTrue(complex_result.data[1].real.is_nan())
        self.assertTrue(complex_result.data[1].imag.is_nan())

        huge = Decimal("1e999999")
        magnitude = abs(_DecimalComplex(huge, huge))
        self.assertTrue(magnitude.is_finite())
        self.assertAlmostEqual(
            float(magnitude / _to_mpfr(huge)),
            np.sqrt(2),
            places=15,
        )

        nonfinite = LF[Generic, object, 1](
            np.asarray(
                [
                    _DecimalComplex(Decimal("Infinity"), Decimal(0)),
                    _DecimalComplex(Decimal("-Infinity"), Decimal(0)),
                    _DecimalComplex(Decimal(1), Decimal(2)),
                ],
                dtype=object,
            ),
            (range(3),),
        )
        for reduced in (
            np.sum(nonfinite.data),
            nonfinite.data.sum(),
            np.add.reduce(nonfinite.data),
        ):
            self.assertTrue(reduced.real.is_nan())
            self.assertIsInstance(reduced.imag, _MPFR)
            self.assertEqual(reduced.imag, 2)
        for cumulative in (
            np.cumsum(nonfinite.data),
            nonfinite.data.cumsum(),
            np.add.accumulate(nonfinite.data),
        ):
            self.assertTrue(cumulative[0].real.is_infinite())
            self.assertTrue(cumulative[1].real.is_nan())
            self.assertIsInstance(cumulative[1].imag, _MPFR)
            self.assertEqual(cumulative[1].imag, 0)
            self.assertTrue(cumulative[2].real.is_nan())
            self.assertIsInstance(cumulative[2].imag, _MPFR)
            self.assertEqual(cumulative[2].imag, 2)

    def test_axis_has_no_arbitrary_cap_and_facade_sync_is_batched(self):
        class MaterializerReached(Exception):
            pass

        with mock.patch.object(
            lattice_field_module,
            "_materialize_range",
            side_effect=MaterializerReached,
        ):
            with self.assertRaises(MaterializerReached):
                LatticeAxis.from_start_step(
                    np.float64(0),
                    np.float64(1),
                    10_000_001,
                )

        field = LF[Generic](
            np.arange(8, dtype=np.int64), (range(8),)
        )
        retained = [field.data for _ in range(2_000)]
        state = field._storage_state
        self.assertEqual(len(state._facades), 2_000)
        del retained[1:]
        gc.collect()
        self.assertEqual(len(state._facades), 1)

        synchronization_calls: list[int] = []
        original = _FieldStorageState.synchronize

        def counted(storage_state):
            synchronization_calls.append(1)
            return original(storage_state)

        with mock.patch.object(
            _FieldStorageState, "synchronize", counted
        ):
            field.data[:] = np.arange(8, dtype=np.int64) + 10
        self.assertEqual(len(synchronization_calls), 1)
        np.testing.assert_array_equal(field.data, np.arange(8) + 10)

    def test_public_data_aliases_cannot_bypass_checked_assignment(self):
        field = LF[Generic, np.int64, 1](
            np.asarray([1, 2], dtype=np.int64), (range(2),)
        )

        # Checked façades and their basic views retain live, Julia-style
        # conversion. No failed update may touch the private field storage.
        for alias in (field.data, field.data[:], field.data[::-1], field.data.T):
            with self.assertRaises(ValueError):
                alias[0] = 1.5
        np.testing.assert_array_equal(field.data, [1, 2])

        matrix = LF[Generic, np.int64, 2](
            np.arange(6, dtype=np.int64).reshape(2, 3),
            (range(2), range(3)),
        )
        for alias in (matrix.data.T, matrix.data[:, ::-1]):
            with self.assertRaises(ValueError):
                alias[0, 0] = 8.5
        np.testing.assert_array_equal(
            matrix.data.copy(),
            np.arange(6, dtype=np.int64).reshape(2, 3),
        )

        # NumPy 2.5 ufunc.at ignores ndarray WRITEABLE flags. Direct calls on
        # the checked façade are intercepted and update one element at a time.
        with self.assertRaises(ValueError):
            np.add.at(field.data, [0], 1.5)
        np.testing.assert_array_equal(field.data, [1, 2])
        np.add.at(field.data, [0, 0], [1, 2])
        np.testing.assert_array_equal(field.data, [4, 2])
        field.data[0] = 1
        with self.assertRaises(ValueError):
            np.add.at(field.data, [0, 1], [2, 0.5])
        np.testing.assert_array_equal(field.data, [3, 2])
        field.data[0] = 1

        # Raw ndarray escape hatches receive detached snapshots. Even NumPy's
        # flag-bypassing ufunc.at can only corrupt the snapshot, not the field.
        for raw_alias in (
            field.data.copy(),
            field.data.view(np.ndarray),
            np.asarray(field.data[:]),
            np.asarray(field.data.T),
        ):
            np.add.at(raw_alias, [0], 7)
            np.testing.assert_array_equal(field.data, [1, 2])

        # Allocating ndarray operations return normal, independently mutable
        # arrays instead of disconnected checked-array subclasses.
        detached_results = (
            field.data.flatten(),
            field.data[[0, 1]],
            field.data.astype(np.float64),
            field.data.astype(np.int64, copy=False),
            field.data.take([0, 1]),
        )
        for detached in detached_results:
            self.assertIs(type(detached), np.ndarray)
            self.assertTrue(detached.flags.writeable)
            detached[0] = 99
            np.testing.assert_array_equal(field.data, [1, 2])

        # Checked basic views remain live and mutable through the same
        # conversion gate even though their WRITEABLE flag stays false.
        checked_slice = field.data[1:]
        checked_slice[0] = 4
        np.testing.assert_array_equal(field.data, [1, 4])
        field.data[1] = 2

        # Attribute augmented assignment performs the checked in-place array
        # mutation and then writes the identical object back to ``data``.
        field.data += 1
        np.testing.assert_array_equal(field.data, [2, 3])
        self.assertFalse(field.data.flags.writeable)

        # Julia broadcast assignment stores each successfully converted
        # element before a later InexactError.
        with self.assertRaises(ValueError):
            field.data[:] = [5, 1.5]
        np.testing.assert_array_equal(field.data, [5, 3])

        # A zero-dimensional array is still an Array in Julia, not a scalar
        # accepted by setindex!.
        with self.assertRaises(ValueError):
            field.data[0] = np.asarray(7)
        self.assertEqual(field.data[0], 5)

    def test_checked_data_generic_views_allocations_and_synchronization(self):
        field = LF[Generic, np.int64, 2](
            np.arange(9, dtype=np.int64).reshape(3, 3),
            (range(3), range(3)),
        )
        stored = field.data
        diagonal = stored.diagonal()
        reshaped = np.reshape(stored, 9)

        # Every already-returned façade is refreshed when another façade
        # mutates the same authoritative field storage.
        field.data[0, 0] = 9
        self.assertEqual(stored[0, 0], 9)
        self.assertEqual(stored.item(0), 9)
        self.assertEqual(stored.tolist()[0][0], 9)
        self.assertEqual(np.asarray(stored)[0, 0], 9)
        self.assertEqual(diagonal[0], 9)
        self.assertEqual(reshaped[0], 9)

        # Generic storage-sharing ndarray results stay live and checked.
        with self.assertRaises(ValueError):
            diagonal[1] = 1.5
        diagonal[1] = 12
        self.assertEqual(field.data[1, 1], 12)
        stored.real[0, 1] = 14
        self.assertEqual(field.data[0, 1], 14)
        np.diagonal(stored)[2] = 16
        self.assertEqual(field.data[2, 2], 16)

        complex_field = LF[Generic, np.complex128, 2](
            np.asarray([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]),
            (range(2), range(2)),
        )
        complex_field.data.real[0, 0] = 10
        complex_field.data.imag[0, 0] = 11
        self.assertEqual(complex_field.data[0, 0], 10 + 11j)
        with self.assertRaises(ValueError):
            complex_field.data.real[0, 0] = 1 + 2j

        # Generic allocating ndarray methods/functions detach as ordinary
        # independently mutable arrays, never disconnected checked subclasses.
        allocating_results = (
            stored.repeat(2, axis=0),
            stored.compress([True, False, True], axis=0),
            np.repeat(stored, 2, axis=1),
            np.concatenate((stored, stored), axis=0),
            np.stack((stored, stored), axis=0),
        )
        field_before = field.data.copy()
        for result in allocating_results:
            self.assertIs(type(result), np.ndarray)
            self.assertTrue(result.flags.writeable)
            result.flat[0] = -100
            np.testing.assert_array_equal(
                field.data.copy(), field_before
            )

        # Explicit subok=True is the only request that may retain the subclass;
        # its independent copy is nevertheless fully usable and detached.
        subclass_copy = np.array(stored, copy=True, subok=True)
        subclass_copy[0, 0] = -200
        self.assertEqual(subclass_copy[0, 0], -200)
        np.testing.assert_array_equal(field.data.copy(), field_before)

    def test_checked_copy_and_boolean_index_read_authoritative_storage(self):
        source = np.arange(6, dtype=np.int64).reshape(
            (2, 3), order="F"
        )
        field = LF[Generic, np.int64, 2](
            source, (range(2), range(3))
        )
        retained = field.data
        source[0, 1] = 99

        copied = retained.copy()
        self.assertEqual(copied[0, 1], 99)
        mask = np.zeros(source.shape, dtype=bool)
        mask[0, 1] = True
        np.testing.assert_array_equal(retained[mask], [99])

    def test_checked_data_copyto_source_and_multi_output_ufunc_contract(self):
        source = LF[Generic, np.float64, 1](
            np.asarray([1.25, 2.75]), (range(2),)
        )

        # A checked façade is safe as a read-only copy source. Only a checked
        # destination would bypass Julia-style assignment and must be refused.
        raw_destination = np.zeros(2, dtype=np.float64)
        self.assertIsNone(np.copyto(raw_destination, source.data))
        np.testing.assert_array_equal(raw_destination, [1.25, 2.75])
        with self.assertRaises(TypeError):
            np.copyto(source.data, np.asarray([3.0, 4.0]))
        np.testing.assert_array_equal(source.data, [1.25, 2.75])

        fractional = LF[Generic, np.float64, 1](
            np.zeros(2, dtype=np.float64), (range(2),)
        )
        checked_output = fractional.data
        fraction_result, integral_result = np.modf(
            source.data, out=(checked_output, None)
        )
        self.assertIs(fraction_result, checked_output)
        np.testing.assert_array_equal(fractional.data, [0.25, 0.75])
        self.assertIs(type(integral_result), np.ndarray)
        np.testing.assert_array_equal(integral_result, [1.0, 2.0])

        allocated_single_output = np.add(source.data, 1.0, out=(None,))
        self.assertIs(type(allocated_single_output), np.ndarray)
        np.testing.assert_array_equal(allocated_single_output, [2.25, 3.75])

    def test_integer_intensity_sqrt_always_widens_to_float64(self):
        for dtype in (
            np.bool_,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
        ):
            field = LF[Intensity, dtype, 1](
                np.asarray([0, 1], dtype=dtype), (range(2),)
            )
            rooted = field.sqrt()
            self.assertEqual(rooted.dtype, np.dtype(np.float64))
            np.testing.assert_array_equal(rooted.data, [0.0, 1.0])

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
        np.testing.assert_array_equal(
            region.data.copy(), self.data[:2, 1:3]
        )

    def test_direct_indexing_uses_julia_trailing_singleton_rules(self):
        data = np.arange(6).reshape((2, 3, 1), order="F")
        field = LF[Generic](data, (range(2), range(3), range(1)))

        # Dense Julia indexing permits omitted trailing singleton dimensions
        # for integer and mixed integer/range calls.
        self.assertEqual(field[1, 1], data[1, 1, 0])
        np.testing.assert_array_equal(field[1, :].data, data[1, :, 0])
        np.testing.assert_array_equal(field[:, 1].data, data[:, 1, 0])
        np.testing.assert_array_equal(
            field[:, :, 0, 0].data.copy(), data[:, :, 0]
        )
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
        np.testing.assert_array_equal(field.data.copy(), original)

    def test_direct_assignment_is_scalar_arity_checked_and_inexact_safe(self):
        field = LF[Generic](
            np.arange(6, dtype=np.int64).reshape((2, 3), order="F"),
            (range(2), range(3)),
        )
        original = field.data.copy()

        with self.assertRaisesRegex(ValueError, "Inexact assignment"):
            field[0] = 1.5
        np.testing.assert_array_equal(field.data.copy(), original)
        with self.assertRaisesRegex(ValueError, "Inexact assignment"):
            field[0, 0] = 1.0 + 2.0j
        np.testing.assert_array_equal(field.data.copy(), original)

        # A one-item integer tuple is the same one-argument linear call as a
        # bare integer; incomplete Cartesian/range assignments still fail.
        with self.assertRaises(DimensionMismatch):
            field[:1] = 7
        with self.assertRaises(TypeError):
            field[:, 0] = 7
        np.testing.assert_array_equal(field.data.copy(), original)

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
        np.testing.assert_array_equal(
            region.data.copy(), self.data[1:, ::-1]
        )
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
        with self.assertRaises(TypeError):
            sublattice(field.L, True, slice(1, 3))
        with self.assertRaises(TypeError):
            sublattice(field.L, np.bool_(False), slice(1, 3))

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
        np.testing.assert_array_equal(
            sliced.data.copy(), [[2, 8, 14], [4, 10, 16]]
        )

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
        rewrapped = wrap(wrapped)
        rewrapped.data[0] = 2.0 + 0.0j
        self.assertEqual(wrapped.data[0], 2.0 + 0.0j)
        rewrapped.data[0] = 1.0 + 0.0j
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
