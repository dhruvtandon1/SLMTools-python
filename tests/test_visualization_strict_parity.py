from numbers import Complex, Number

import gmpy2
import numpy as np
import pytest

import slmtools as slm


@pytest.mark.parametrize("field_type", [slm.Modulus, slm.Intensity])
@pytest.mark.parametrize(
    ("element_type", "data"),
    [
        (
            np.complex64,
            np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64),
        ),
        (
            np.complex128,
            np.asarray([1 + 2j, 3 + 4j], dtype=np.complex128),
        ),
        (
            gmpy2.mpc,
            np.asarray([gmpy2.mpc(1, 2), gmpy2.mpc(3, 4)], dtype=object),
        ),
    ],
    ids=("complex64", "complex128", "gmpy2-mpc"),
)
def test_look_rejects_complex_modulus_and_intensity_storage(
    field_type: type,
    element_type: type,
    data: np.ndarray,
) -> None:
    field = slm.LF[field_type, element_type, 1](data.copy(), (range(2),))

    with pytest.raises(
        TypeError, match="maximum cannot order complex visualization values"
    ):
        slm.look(field)


@pytest.mark.parametrize(
    "field_type",
    [slm.ComplexPhase, slm.ComplexAmplitude],
    ids=("ComplexPhase", "ComplexAmplitude-ComplexAmp"),
)
def test_look_rejects_real_storage_for_complex_visualizations(
    field_type: type,
) -> None:
    data = np.asarray([1.0, 2.0], dtype=np.float64)
    field = slm.LF[field_type, np.float64, 1](data, (range(2),))

    with pytest.raises(
        TypeError, match="cycle1 requires complex visualization values"
    ):
        slm.look(field)


@pytest.mark.parametrize(
    "field_type",
    [slm.ComplexPhase, slm.ComplexAmplitude],
    ids=("ComplexPhase", "ComplexAmplitude-phase-panel"),
)
def test_cycle1_keeps_float32_numerator_and_float64_denominator(
    field_type: type,
) -> None:
    data = np.asarray([1 + 1j], dtype=np.complex64)
    field = slm.LF[field_type, np.complex64, 1](data, (range(1),))

    image = slm.look(field)
    phase = image if field_type is slm.ComplexPhase else image[:, 1]

    assert phase.dtype == np.dtype(np.float64)
    assert phase[0] == np.float64(0.6250000268785834)


@pytest.mark.parametrize(
    "field_type",
    [slm.ComplexPhase, slm.ComplexAmplitude],
    ids=("ComplexPhase", "ComplexAmplitude-phase-panel"),
)
def test_cycle1_promotes_binary64_denominator_to_bigfloat(
    field_type: type,
) -> None:
    data = np.asarray([gmpy2.mpc(1, 0)], dtype=object)
    field = slm.LF[field_type, gmpy2.mpc, 1](data, (range(1),))

    image = slm.look(field)
    phase = image if field_type is slm.ComplexPhase else image[:, 1]
    with gmpy2.context(gmpy2.get_context(), precision=256):
        expected = gmpy2.mpfr(
            "0.500000000000000019490859162596877992609001380422712116291"
            "2775933997823872255114"
        )

    assert phase.dtype == np.dtype(object)
    assert phase[0] == expected
    assert phase[0] != gmpy2.mpfr("0.5")


def test_look_quantizes_gmpy2_rationals_for_raw_and_field_inputs() -> None:
    data = np.asarray([gmpy2.mpq(1, 2), gmpy2.mpq(1)], dtype=object)
    field = slm.LF[slm.Intensity, gmpy2.mpq, 1](
        data.copy(), (range(2),)
    )
    expected = np.asarray([128.0 / 255.0, 1.0], dtype=np.float64)

    for image in (slm.look(data), slm.look(field)):
        assert image.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(image, expected)


def test_multi_field_look_promotes_all_channels_to_bigfloat() -> None:
    lattice = (range(2),)
    machine = slm.LF[slm.Modulus, np.float64, 1](
        np.asarray([1.0, 2.0], dtype=np.float64), lattice
    )
    bigfloat_data = np.asarray(
        [gmpy2.mpfr(1), gmpy2.mpfr(2)], dtype=object
    )
    bigfloat = slm.LF[slm.Modulus, gmpy2.mpfr, 1](
        bigfloat_data, lattice
    )

    image = slm.look(machine, bigfloat)

    assert image.shape == (2, 2)
    assert image.dtype == np.dtype(object)
    assert all(isinstance(value, gmpy2.mpfr) for value in image.flat)
    assert image.tolist() == [
        [gmpy2.mpfr("0.5"), gmpy2.mpfr("0.5")],
        [gmpy2.mpfr(1), gmpy2.mpfr(1)],
    ]


@pytest.mark.parametrize("element_type", [object, Number, Complex])
def test_look_cycles_object_backed_abstract_complex_fields(
    element_type: type,
) -> None:
    values = np.asarray([1 + 1j, 1 - 1j], dtype=object)
    field = slm.LF[slm.ComplexPhase, element_type, 1](
        values, (range(2),)
    )
    result = slm.look(field)
    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(result, [0.625, 0.375])
