from __future__ import annotations

import inspect
from decimal import Decimal
from fractions import Fraction
from typing import Any

import gmpy2
import numpy as np
import pytest

import slmtools as slm
from slmtools._bigfloat import _MPFRComplex, _bigfloat_context
from slmtools._omission import _OMITTED


_COMPLEX_BIGFLOAT_CASES = (
    (
        ((1, 2), (3, -1)),
        ((5, 1), (-2, 4)),
        (
            (
                "-1.769230769230769230769230769230769230769230769230769230769230769230769230769228",
                "-1.153846153846153846153846153846153846153846153846153846153846153846153846153859",
            ),
            (
                "4.461538461538461538461538461538461538461538461538461538461538461538461538461475",
                "5.69230769230769230769230769230769230769230769230769230769230769230769230769235",
            ),
        ),
    ),
    (
        ((1, 2), (3, -1), (-2, "0.5")),
        ((5, 1), (-2, 4), (3, -2)),
        (
            (
                "-0.9029126213592233009708737864077669902912621359223300970873786407766990291262066",
                "0.2621359223300970873786407766990291262135922330097087378640776699029126213592274",
            ),
            (
                "2.733009708737864077669902912621359223300970873786407766990291262135922330097083",
                "1.276699029126213592233009708737864077669902912621359223300970873786407766990295",
            ),
        ),
    ),
    (
        ((1, 2),),
        ((5, 1),),
        (
            (
                "1.166666666666666666666666666666666666666666666666666666666666666666666666666644",
                "-1.499999999999999999999999999999999999999999999999999999999999999999999999999965",
            ),
            (
                "0.8333333333333333333333333333333333333333333333333333333333333333333333333333045",
                "0.1666666666666666666666666666666666666666666666666666666666666666666666666666609",
            ),
        ),
    ),
)


@pytest.mark.parametrize("adapter", ["internal", "gmpy2"])
@pytest.mark.parametrize(
    ("xs", "ys", "expected"),
    _COMPLEX_BIGFLOAT_CASES,
    ids=("square-lu", "tall-qr", "wide-qr"),
)
def test_linear_fit_supports_julia_complex_bigfloat_domain(
    adapter: str,
    xs: tuple[tuple[Any, Any], ...],
    ys: tuple[tuple[Any, Any], ...],
    expected: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    with _bigfloat_context():
        def make(value: tuple[Any, Any]) -> Any:
            real = gmpy2.mpfr(value[0])
            imag = gmpy2.mpfr(value[1])
            if adapter == "internal":
                return _MPFRComplex(real, imag)
            return gmpy2.mpc(real, imag)

        actual = slm.linearFit(
            [make(value) for value in xs],
            [make(value) for value in ys],
        )
        for value, (expected_real, expected_imag) in zip(
            actual, expected, strict=True
        ):
            assert isinstance(value, _MPFRComplex)
            assert value.real == gmpy2.mpfr(expected_real)
            assert value.imag == gmpy2.mpfr(expected_imag)


@pytest.mark.parametrize(
    ("xs", "ys", "expected"),
    (
        (
            ("1.25", "-2.5"),
            ("3.75", "0.125"),
            (
                "0.9666666666666666666666666666666666666666666666666666666666666666666666666666713",
                "2.541666666666666666666666666666666666666666666666666666666666666666666666666678",
            ),
        ),
        (
            ("1.25", "-2.5", "0.375"),
            ("3.75", "0.125", "-1.625"),
            (
                "0.5744248985115020297699594046008119079837618403247631935047361299052774018944501",
                "0.9175405953991880920162381596752368064952638700947225981055480378890392422192449",
            ),
        ),
        (
            ("1.25",),
            ("3.75",),
            (
                "1.829268292682926829268292682926829268292682926829268292682926829268292682926873",
                "1.46341463414634146341463414634146341463414634146341463414634146341463414634146",
            ),
        ),
    ),
    ids=("square-lu", "tall-qr", "wide-qr"),
)
def test_linear_fit_accepts_plain_mpfr_bigfloat_vectors(
    xs: tuple[str, ...],
    ys: tuple[str, ...],
    expected: tuple[str, str],
) -> None:
    with _bigfloat_context():
        actual = slm.linearFit(
            [gmpy2.mpfr(value) for value in xs],
            [gmpy2.mpfr(value) for value in ys],
        )
        assert all(isinstance(value, gmpy2.mpfr) for value in actual)
        assert actual == tuple(gmpy2.mpfr(value) for value in expected)


@pytest.mark.parametrize(
    "adapter",
    ("real", "internal-complex", "gmpy2-complex"),
)
def test_linear_fit_square_bigfloat_nonfinite_parity(adapter: str) -> None:
    with _bigfloat_context():
        def make(value: str) -> Any:
            scalar = gmpy2.mpfr(value)
            if adapter == "real":
                return scalar
            if adapter == "internal-complex":
                return _MPFRComplex(scalar, 0)
            return gmpy2.mpc(scalar, gmpy2.mpfr(0))

        with pytest.raises(
            ValueError,
            match="design matrix contains Infs or NaNs",
        ):
            slm.linearFit(
                [make("0"), make("inf")],
                [make("2"), make("3")],
            )

        nonfinite_rhs = slm.linearFit(
            [make("0"), make("1")],
            [make("2"), make("inf")],
        )
        upper_inf = slm.linearFit(
            [make("inf"), make("0")],
            [make("2"), make("3")],
        )
        upper_nan = slm.linearFit(
            [make("nan"), make("0")],
            [make("2"), make("3")],
        )

        if adapter == "real":
            assert gmpy2.is_infinite(nonfinite_rhs[0])
            assert nonfinite_rhs[1] == 2
            assert upper_inf[0] == 0
            assert gmpy2.is_signed(upper_inf[0])
            assert upper_inf[1] == 3
            assert gmpy2.is_nan(upper_nan[0])
            assert upper_nan[1] == 3
            return

        assert all(
            isinstance(value, _MPFRComplex)
            for value in (*nonfinite_rhs, *upper_inf, *upper_nan)
        )
        assert gmpy2.is_nan(nonfinite_rhs[0].real)
        assert gmpy2.is_nan(nonfinite_rhs[0].imag)
        assert nonfinite_rhs[1].real == 2
        assert nonfinite_rhs[1].imag == 0
        assert gmpy2.is_signed(nonfinite_rhs[1].imag)
        assert upper_inf[0].real == upper_inf[0].imag == 0
        assert gmpy2.is_signed(upper_inf[0].real)
        assert gmpy2.is_signed(upper_inf[0].imag)
        assert upper_inf[1].real == 3
        assert upper_inf[1].imag == 0
        assert gmpy2.is_signed(upper_inf[1].imag)
        assert gmpy2.is_nan(upper_nan[0].real)
        assert gmpy2.is_nan(upper_nan[0].imag)
        assert upper_nan[1].real == 3


@pytest.mark.parametrize("complex_rhs", (False, True))
def test_linear_fit_machine_factor_bigfloat_rhs_singular_is_linalg_error(
    complex_rhs: bool,
) -> None:
    with _bigfloat_context():
        if complex_rhs:
            values = [
                gmpy2.mpc(gmpy2.mpfr(1), gmpy2.mpfr(0)),
                gmpy2.mpc(gmpy2.mpfr(2), gmpy2.mpfr(0)),
            ]
        else:
            values = [gmpy2.mpfr(1), gmpy2.mpfr(2)]

        with pytest.raises(np.linalg.LinAlgError):
            slm.linearFit([1.0, 1.0], values)


@pytest.mark.parametrize("complex_domain", (False, True))
def test_linear_fit_machine_square_nonfinite_design_parity(
    complex_domain: bool,
) -> None:
    if complex_domain:
        make = complex
    else:
        make = float

    for nonfinite in (float("inf"), float("nan")):
        with pytest.raises(
            ValueError,
            match="design matrix contains Infs or NaNs",
        ):
            slm.linearFit(
                [make(nonfinite), make(1)],
                [make(2), make(3)],
            )

    upper_inf = slm.linearFit(
        [make(float("inf")), make(0)],
        [make(2), make(3)],
    )
    upper_nan = slm.linearFit(
        [make(float("nan")), make(0)],
        [make(2), make(3)],
    )

    if not complex_domain:
        assert upper_inf[0] == 0
        assert np.signbit(upper_inf[0])
        assert upper_inf[1] == 3
        assert np.isnan(upper_nan[0])
        assert upper_nan[1] == 3
        return

    assert upper_inf[0].real == upper_inf[0].imag == 0
    assert not np.signbit(upper_inf[0].real)
    assert not np.signbit(upper_inf[0].imag)
    assert upper_inf[1].real == 3
    assert upper_inf[1].imag == 0
    assert not np.signbit(upper_inf[1].imag)
    assert np.isnan(upper_nan[0].real)
    assert np.isnan(upper_nan[0].imag)
    assert upper_nan[1].real == 3


def test_linear_fit_machine_complex_square_nonfinite_rhs_lapack_parity() -> None:
    pivoted = slm.linearFit(
        [0j, 1 + 0j],
        [complex(float("inf"), 0), 2 + 0j],
    )
    general = slm.linearFit(
        [1 + 2j, 3 - 1j],
        [2 + 0j, complex(float("inf"), 0)],
    )
    triangular = slm.linearFit(
        [1 + 2j, 0j],
        [2 + 0j, complex(float("inf"), 0)],
    )

    for actual in (pivoted, triangular):
        assert np.isnan(actual[0].real)
        assert np.isnan(actual[0].imag)
        assert np.isposinf(actual[1].real)
        assert np.isnan(actual[1].imag)

    assert np.isnan(general[0].real)
    assert np.isnan(general[0].imag)
    assert np.isnan(general[1].real)
    assert np.isneginf(general[1].imag)


def test_linear_fit_machine_rectangular_nonfinite_real_parity() -> None:
    cases = (
        ([float("inf")], [2.0], (float("nan"), float("nan"))),
        (
            [0.0, float("inf"), 2.0],
            [2.0, 3.0, 6.0],
            (float("nan"), 4.499999999999998),
        ),
        ([1.0], [float("inf")], (float("nan"), float("inf"))),
        ([float("nan")], [2.0], (float("nan"), float("nan"))),
        (
            [0.0, float("nan"), 2.0],
            [2.0, 3.0, 6.0],
            (float("nan"), float("nan")),
        ),
        (
            [0.0, 1.0, 2.0],
            [2.0, float("inf"), 6.0],
            (float("nan"), float("nan")),
        ),
        (
            [1.0, 1.0, 1.0],
            [float("inf"), 2.0, 3.0],
            (float("nan"), float("nan")),
        ),
    )

    for xs, ys, expected in cases:
        actual = slm.linearFit(xs, ys)
        for value, expected_value in zip(actual, expected, strict=True):
            if np.isnan(expected_value):
                assert np.isnan(value)
            else:
                assert value == expected_value


def test_linear_fit_machine_rectangular_nonfinite_complex_parity() -> None:
    wide = slm.linearFit(
        [complex(float("inf"), 0)],
        [2 + 0j],
    )
    tall = slm.linearFit(
        [0j, complex(float("inf"), 0), 2 + 0j],
        [2 + 0j, 3 + 0j, 6 + 0j],
    )
    finite_design = slm.linearFit(
        [1 + 2j, 3 - 1j, -2 + 0.5j],
        [2 + 0j, complex(float("inf"), 0), 6 + 0j],
    )

    assert all(
        np.isnan(value.real) and np.isnan(value.imag)
        for value in wide
    )
    assert np.isnan(tall[0].real) and np.isnan(tall[0].imag)
    assert tall[1] == 4.499999999999998 + 0j
    assert all(
        np.isnan(value.real) and np.isnan(value.imag)
        for value in finite_design
    )


@pytest.mark.parametrize("complex_rhs", (False, True))
def test_linear_fit_machine_design_bigfloat_rhs_nonfinite_parity(
    complex_rhs: bool,
) -> None:
    with _bigfloat_context():
        def make(value: str) -> Any:
            scalar = gmpy2.mpfr(value)
            if complex_rhs:
                return gmpy2.mpc(scalar, gmpy2.mpfr(0))
            return scalar

        with pytest.raises(
            ValueError,
            match="design matrix contains Infs or NaNs",
        ):
            slm.linearFit(
                [float("inf"), 1.0],
                [make("2"), make("3")],
            )

        upper = slm.linearFit(
            [float("inf"), 0.0],
            [make("2"), make("3")],
        )
        if not complex_rhs:
            assert upper[0] == 0 and gmpy2.is_signed(upper[0])
            assert upper[1] == 3
        else:
            assert upper[0].real == upper[0].imag == 0
            assert gmpy2.is_signed(upper[0].real)
            assert gmpy2.is_signed(upper[0].imag)
            assert upper[1].real == 3
            assert gmpy2.is_signed(upper[1].imag)


def test_linear_fit_decimal_nonfinite_upper_triangular_parity() -> None:
    upper_inf = slm.linearFit(
        [Decimal("Infinity"), Decimal(0)],
        [Decimal(2), Decimal(3)],
    )
    upper_nan = slm.linearFit(
        [Decimal("NaN"), Decimal(0)],
        [Decimal(2), Decimal(3)],
    )

    assert upper_inf[0].is_zero() and upper_inf[0].is_signed()
    assert upper_inf[1] == 3
    assert upper_nan[0].is_nan()
    assert upper_nan[1] == 3


@pytest.mark.parametrize(
    ("xs", "ys", "expected"),
    (
        (
            [Decimal("1.25"), Decimal("-2.5"), Decimal("0.375")],
            [3 + 1j, -2 + 0.5j, 1 - 4j],
            (
                (
                    "1.261163734776725304465493910690121786197564276048714479025710419485791610284138",
                    "-0.2895805142083897158322056833558863328822733423545331529093369418132611637347808",
                ),
                (
                    "1.034506089309878213802435723951285520974289580514208389715832205683355886332871",
                    "-0.9177943166441136671177266576454668470906630581867388362652232746955345060893287",
                ),
            ),
        ),
        (
            [1 + 2j, 3 - 1j, -2 + 0.5j],
            [Decimal(5), Decimal(-2), Decimal(3)],
            (
                (
                    "-0.6407766990291261679715229054769544502641883883940106194981761951436549024620147",
                    "-0.6116504854368931618756314576580800091520021458629238806159060356544714995365769",
                ),
                (
                    "2.121359223300971475379439094576237649162336294039896809917968288634796249837296",
                    "0.7281553398058251765979571426141985796140219039773661228653735658635025469297859",
                ),
            ),
        ),
        (
            [1 + 2j, 3 - 1j],
            [Decimal(5), Decimal(-2)],
            (
                (
                    "-1.076923076923077009792567603858380585181130523973875496743987535953150566109798",
                    "-1.615384615384615365235751937016494390537126507383980076253864898542352685645993",
                ),
                (
                    "2.846153846153846394613454748591636146080518079305606566485827506401804383975371",
                    "3.769230769230769085914688207191102586430248998178064732017607159673907490828129",
                ),
            ),
        ),
        (
            [1 + 2j],
            [Decimal(5)],
            (
                (
                    "0.8333333333333331954147692640403894751126135404691741647228376439751452113569614",
                    "-1.666666666666666597707384632020192169649714253920280872419306618994547863780673",
                ),
                (
                    "0.8333333333333331954147692640403997467389836057263990044912864559472441789480722",
                    "0.0",
                ),
            ),
        ),
    ),
    ids=(
        "bigfloat-design-tall",
        "bigfloat-rhs-tall",
        "bigfloat-rhs-square",
        "bigfloat-rhs-wide",
    ),
)
def test_linear_fit_promotes_decimal_and_machine_complex_vectors(
    xs: list[Any],
    ys: list[Any],
    expected: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    actual = slm.linearFit(xs, ys)

    with _bigfloat_context():
        machine_factors = not any(
            isinstance(value, Decimal) for value in xs
        )
        for value, (expected_real, expected_imag) in zip(
            actual, expected, strict=True
        ):
            assert isinstance(value, _MPFRComplex)
            expected_components = (
                gmpy2.mpfr(expected_real),
                gmpy2.mpfr(expected_imag),
            )
            if machine_factors:
                for component, expected_component in zip(
                    (value.real, value.imag),
                    expected_components,
                    strict=True,
                ):
                    tolerance = gmpy2.mpfr("2e-15") * max(
                        gmpy2.mpfr(1),
                        abs(expected_component),
                    )
                    assert abs(component - expected_component) <= tolerance
            else:
                assert (value.real, value.imag) == expected_components


@pytest.mark.parametrize(
    ("xs", "ys", "expected"),
    (
        (
            [Fraction(1, 2), Fraction(3, 2), Fraction(5, 2)],
            [1 + 2j, -3 + 0.5j, 4 - 1j],
            (
                1.5 - 1.4999999999999998j,
                -1.5833333333333344 + 2.7500000000000004j,
            ),
        ),
        (
            [1 + 2j, 3 - 1j, -2 + 0.5j],
            [Fraction(5, 2), Fraction(-2, 3), Fraction(3, 4)],
            (
                -0.15857605177993528 - 0.2766990291262135j,
                0.8284789644012948 + 0.2637540453074433j,
            ),
        ),
    ),
    ids=("rational-design", "rational-rhs"),
)
def test_linear_fit_promotes_fraction_and_machine_complex_vectors(
    xs: list[Any],
    ys: list[Any],
    expected: tuple[complex, complex],
) -> None:
    actual = slm.linearFit(xs, ys)

    assert all(isinstance(value, np.complex128) for value in actual)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)


def test_linear_fit_uses_julia_rectangular_pivoted_qr() -> None:
    indices = [1.0, 1.0 + 1e-15, 1.0 + 2e-15]

    actual = slm.linearFit(indices, [1.0, 2.0, 3.0])

    # Julia 1.11.6's rectangular backslash path retains the second QR
    # direction here. An SVD rank threshold instead returns approximately
    # (1, 1), changing both the algorithm and the result.
    expected = (8.277502778019098e14, -8.277502778019084e14)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)
    assert abs(actual[0]) > 1e14


def test_linear_fit_uses_julia_pivoted_qr_rank_cutoff() -> None:
    epsilon = np.finfo(np.float64).eps
    values = [1.0, 2.0, 3.0]

    for multiplier in (1, 2, 3):
        indices = [
            1.0,
            1.0 + multiplier * epsilon,
            1.0 + 2 * multiplier * epsilon,
        ]
        actual = slm.linearFit(indices, values)
        np.testing.assert_allclose(actual, (1.0, 1.0), rtol=2e-15, atol=0)

    rank_two = slm.linearFit(
        [1.0, 1.0 + 4 * epsilon, 1.0 + 8 * epsilon],
        values,
    )
    assert abs(rank_two[0]) > 1e14


def test_get_orientation_inherits_julia_rectangular_qr_fit() -> None:
    lattice = (np.asarray([1.0, 2.0, 3.0]), np.asarray([0.0]))
    fields: list[slm.LatticeField] = []
    for selected in range(3):
        data = np.zeros((3, 1))
        data[selected, 0] = 1
        fields.append(slm.LF[slm.Intensity](data, lattice))

    center, theta = slm.getOrientation(
        fields,
        [1.0, 1.0 + 1e-15, 1.0 + 2e-15],
    )

    np.testing.assert_allclose(
        center,
        [-8.277502778019084e14, 0.0],
        rtol=2e-15,
        atol=0,
    )
    assert theta == 0
    assert np.signbit(theta)


def test_get_orientation_preserves_mpfr_angle_precision() -> None:
    lattice = (np.arange(5.0), np.arange(3.0))
    fields: list[slm.LatticeField] = []
    for first, second in ((0, 0), (2, 1), (4, 2)):
        data = np.zeros((5, 3))
        data[first, second] = 1
        fields.append(slm.LF[slm.Intensity](data, lattice))

    with _bigfloat_context():
        center, theta = slm.getOrientation(
            fields,
            [gmpy2.mpfr(0), gmpy2.mpfr(1), gmpy2.mpfr(2)],
        )
        expected = gmpy2.mpfr(
            "0.4636476090008061162142562314612144020285370542861202638109330887201978641657418"
        )

    assert center.dtype == np.dtype(object)
    assert all(isinstance(value, gmpy2.mpfr) for value in center)
    assert all(abs(value) < gmpy2.mpfr("1e-70") for value in center)
    assert isinstance(theta, gmpy2.mpfr)
    assert theta == expected


def test_dualate_calls_custom_interpolator_at_scalar_cartesian_indices() -> None:
    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic](np.zeros((2, 2)), lattice)
    calls: list[tuple[float, float]] = []

    def factory(
        _ranges: Any,
        _values: Any,
        *,
        extrapolation_bc: Any = 0,
    ) -> Any:
        del extrapolation_bc

        def interpolate(x: Any, y: Any) -> int:
            assert np.ndim(x) == 0
            assert np.ndim(y) == 0
            calls.append((float(x), float(y)))
            return len(calls)

        return interpolate

    result = slm.dualate(
        source,
        lattice,
        [0.0, 0.0],
        0.0,
        interpolation=factory,
    )

    np.testing.assert_array_equal(result.data.copy(), [[1, 3], [2, 4]])
    assert calls == [
        (-0.5, -0.5),
        (0.0, -0.5),
        (-0.5, 0.0),
        (0.0, 0.0),
    ]


def test_dualate_preserves_custom_interpolator_error_order() -> None:
    class InterpolationFailure(Exception):
        pass

    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic](np.zeros((2, 2)), lattice)
    calls: list[tuple[float, float]] = []

    def factory(
        _ranges: Any,
        _values: Any,
        *,
        extrapolation_bc: Any = 0,
    ) -> Any:
        del extrapolation_bc

        def interpolate(x: Any, y: Any) -> float | np.ndarray:
            # The former Python extension called this branch once and
            # accepted its whole-array result, hiding scalar-call failures.
            if np.ndim(x) != 0 or np.ndim(y) != 0:
                return np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))
            calls.append((float(x), float(y)))
            if len(calls) == 2:
                raise InterpolationFailure
            return 1.0

        return interpolate

    with pytest.raises(InterpolationFailure):
        slm.dualate(
            source,
            lattice,
            [0.0, 0.0],
            0.0,
            interpolation=factory,
        )

    assert calls == [(-0.5, -0.5), (0.0, -0.5)]


def test_dualate_preserves_zero_dimensional_array_elements() -> None:
    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic](np.zeros((2, 2)), lattice)

    def factory(
        _ranges: Any,
        _values: Any,
        *,
        extrapolation_bc: Any = 0,
    ) -> Any:
        del extrapolation_bc
        return lambda _x, _y: np.array(1.0)

    result = slm.dualate(
        source,
        lattice,
        [0.0, 0.0],
        0.0,
        interpolation=factory,
    )

    assert result.data.dtype == np.dtype(object)
    for value in result.data.copy().flat:
        assert isinstance(value, np.ndarray)
        assert value.shape == ()
        assert value.item() == 1.0


def test_dualate_preserves_homogeneous_tuple_elements() -> None:
    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic](np.zeros((2, 2)), lattice)
    calls: list[tuple[float, float]] = []

    def factory(
        _ranges: Any,
        _values: Any,
        *,
        extrapolation_bc: Any = 0,
    ) -> Any:
        del extrapolation_bc

        def interpolate(x: Any, y: Any) -> tuple[int, float]:
            calls.append((float(x), float(y)))
            return len(calls), float(x)

        return interpolate

    result = slm.dualate(
        source,
        lattice,
        [0.0, 0.0],
        0.0,
        interpolation=factory,
    )

    assert result.data.shape == (2, 2)
    assert result.data.dtype == np.dtype(object)
    assert result.data[0, 0] == (1, -0.5)
    assert result.data[1, 0] == (2, 0.0)
    assert result.data[0, 1] == (3, -0.5)
    assert result.data[1, 1] == (4, 0.0)
    assert calls == [
        (-0.5, -0.5),
        (0.0, -0.5),
        (-0.5, 0.0),
        (0.0, 0.0),
    ]


def test_dualate_preserves_vector_cells_and_column_major_identity() -> None:
    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic](np.zeros((2, 2)), lattice)
    calls: list[tuple[float, float]] = []
    returned: list[list[float]] = []

    def factory(
        _ranges: Any,
        _values: Any,
        *,
        extrapolation_bc: Any = 0,
    ) -> Any:
        del extrapolation_bc

        def interpolate(x: Any, y: Any) -> list[float]:
            calls.append((float(x), float(y)))
            cell = [float(len(calls)), float(x)]
            returned.append(cell)
            return cell

        return interpolate

    result = slm.dualate(
        source,
        lattice,
        [0.0, 0.0],
        0.0,
        interpolation=factory,
    )

    assert result.data.shape == (2, 2)
    assert result.data.dtype == np.dtype(object)
    assert result.data[0, 0] is returned[0]
    assert result.data[1, 0] is returned[1]
    assert result.data[0, 1] is returned[2]
    assert result.data[1, 1] is returned[3]
    assert calls == [
        (-0.5, -0.5),
        (0.0, -0.5),
        (-0.5, 0.0),
        (0.0, 0.0),
    ]


def test_dualate_keeps_scalar_callback_promotion() -> None:
    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic](np.zeros((2, 2)), lattice)

    def factory(
        _ranges: Any,
        _values: Any,
        *,
        extrapolation_bc: Any = 0,
    ) -> Any:
        del extrapolation_bc
        return lambda _x, _y: np.float32(1.25)

    result = slm.dualate(
        source,
        lattice,
        [0.0, 0.0],
        0.0,
        interpolation=factory,
    )

    assert result.data.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        result.data.copy(),
        np.full((2, 2), 1.25, np.float32),
    )


@pytest.mark.parametrize("naturalize", [None, 0, 1, "true"])
def test_dualate_rejects_non_boolean_naturalize_before_scalar_calls(
    naturalize: Any,
) -> None:
    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic](np.zeros((2, 2)), lattice)
    factory_calls = 0
    scalar_calls = 0

    def factory(
        _ranges: Any,
        _values: Any,
        *,
        extrapolation_bc: Any = 0,
    ) -> Any:
        nonlocal factory_calls, scalar_calls
        del extrapolation_bc
        factory_calls += 1

        def interpolate(_x: Any, _y: Any) -> float:
            nonlocal scalar_calls
            scalar_calls += 1
            return 1.0

        return interpolate

    with pytest.raises(TypeError, match="naturalize"):
        slm.dualate(
            source,
            lattice,
            [0.0, 0.0],
            0.0,
            interpolation=factory,
            naturalize=naturalize,
        )

    assert factory_calls == 1
    assert scalar_calls == 0


def test_dualate_distinguishes_omitted_and_explicit_none_boundary() -> None:
    lattice = (range(2), range(2))
    source = slm.LF[slm.Generic](
        np.arange(4.0).reshape(2, 2),
        lattice,
    )
    boundaries: list[Any] = []

    def factory(
        _ranges: Any,
        _values: Any,
        *,
        extrapolation_bc: Any = _OMITTED,
    ) -> Any:
        boundaries.append(extrapolation_bc)
        return lambda _x, _y: 0.0

    slm.dualate(
        source,
        lattice,
        [0.0, 0.0],
        0.0,
        interpolation=factory,
    )
    slm.dualate(
        source,
        lattice,
        [0.0, 0.0],
        0.0,
        interpolation=factory,
        bc=None,
    )

    assert boundaries[0] == 0.0
    assert boundaries[1] is None

    filled = slm.dualate(
        source,
        lattice,
        [0.0, 0.0],
        0.0,
        bc=None,
    )
    assert filled.data.dtype == np.dtype(object)
    assert filled.data.tolist() == [[None, None], [None, 0.0]]


def test_parse_file_name_distinguishes_omitted_and_explicit_none_cue() -> None:
    assert slm.parseFileName("12.bmp") == 12
    with pytest.raises(TypeError):
        slm.parseFileName("12.bmp", None)


def test_parse_file_name_cueless_overload_rejects_explicit_look() -> None:
    assert (
        inspect.signature(slm.parseFileName).parameters["look"].default
        is _OMITTED
    )
    with pytest.raises(TypeError, match="positional-only"):
        slm.parseFileName("12.bmp", look="after")
    with pytest.raises(TypeError, match="positional-only"):
        slm.parseFileName("12.bmp", look=None)
    with pytest.raises(TypeError, match="positional-only"):
        slm.parseFileName("x12.bmp", "x", look="after")
    with pytest.raises(TypeError, match="positional-only"):
        slm.parseFileName("x12.bmp", "x", look=None)
    with pytest.raises(TypeError, match="look"):
        slm.parseFileName("x12.bmp", "x", None)
    assert slm.parseFileName("x12.bmp", "x") == 12
    assert slm.parseFileName("x12.bmp", "x", "after") == 12


def test_load_dir_validates_typed_keywords_before_entering_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BodyEntered(RuntimeError):
        pass

    def enter_body(_directory: str, _extension: str) -> Any:
        raise BodyEntered

    monkeypatch.setattr(
        "slmtools.image_processing.getImagesAndFilenames",
        enter_body,
    )
    invalid_keywords = (
        ({"T": None}, "T"),
        ({"outType": None}, "outType"),
        ({"L": 1j}, "L"),
        ({"L": (1.0,)}, "L"),
        ({"L": [range(2), range(2)]}, "L"),
        ({"L": (np.array(1.0), np.array(2.0))}, "L"),
        ({"flambda": None}, "flambda"),
        ({"flambda": 1j}, "flambda"),
        ({"cue": 1}, "cue"),
        ({"look": None}, "look"),
        ({"look": 1}, "look"),
    )
    for keywords, name in invalid_keywords:
        with pytest.raises(TypeError, match=name):
            slm.loadDir("unused", ".bmp", **keywords)

    # These are the two typed-union keywords whose Julia contract includes
    # ``nothing``; they must pass validation and reach the body.
    with pytest.raises(BodyEntered):
        slm.loadDir("unused", ".bmp", L=None, cue=None)
    # NumPy dtype descriptors are the established Python spelling accepted by
    # parseStringToNum, while class objects map directly to Julia DataType.
    with pytest.raises(BodyEntered):
        slm.loadDir("unused", ".bmp", outType=np.dtype(np.float32))
    with pytest.raises(BodyEntered):
        slm.loadDir("unused", ".bmp", outType=Decimal)


def test_save_beam_directory_uses_an_omission_sentinel() -> None:
    default = inspect.signature(slm.saveBeam).parameters["dir"].default
    assert default is _OMITTED

    beam = np.zeros((1, 1), dtype=np.complex128)
    assert slm.saveBeam(beam, "empty", data=()) is None
    assert slm.saveBeam(beam, "empty", data=(), dir=None) is None
