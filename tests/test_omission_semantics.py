from __future__ import annotations

import inspect

import numpy as np
import pytest

import slmtools as slm
from slmtools._omission import _OMITTED


def test_reported_none_ambiguities_share_one_omission_sentinel() -> None:
    affected = {
        slm.gs: "phi0",
        slm.gsLog: "phi0",
        slm.getCostMatrix: "Lv",
        slm.coarsen: "reducer",
        slm.parseFileName: "cue",
        slm.scalarPotentialN: "dimOrder",
        slm.padout: "filler",
        slm.downsample: "bc",
        slm.upsample: "bc",
        slm.lfRand: "center",
        slm.lfParabola: "center",
        slm.lfGaussian: "center",
        slm.lfRing: "center",
        slm.lfCap: "center",
        slm.lfRect: "center",
        slm.lfText: "center",
        slm.lfHeart: "center",
        slm.lfSmile: "center",
        slm.lfPointer: "center",
    }

    assert len(affected) == 19
    assert {
        inspect.signature(function).parameters[parameter].default
        for function, parameter in affected.items()
    } == {_OMITTED}
    for function, positional_parameter in (
        (slm.lfParabola, "lin"),
        (slm.lfGaussian, "norm"),
        (slm.lfRect, "height"),
    ):
        assert positional_parameter not in inspect.signature(
            function
        ).parameters
    independently_audited = {
        slm.dualate: "bc",
        slm.saveBeam: "dir",
    }
    assert {
        inspect.signature(function).parameters[parameter].default
        for function, parameter in independently_audited.items()
    } == {_OMITTED}


def test_none_defaults_that_are_julia_nothing_remain_explicit_none() -> None:
    legitimate_nothing = (
        (slm.dualPhase, "dL"),
        (slm.loadDir, "L"),
        (slm.loadDir, "cue"),
        (slm.parseFileName, "outType"),
        (slm.parseStringToNum, "outType"),
        (slm.getOrientation, "roi"),
        (slm.dualate, "roi"),
        (slm.ftaText, "pixelsize"),
        (slm.lfText, "pixelsize"),
        (slm.pdotBeamEstimate, "LFine"),
        (slm.scalarPotentialN, "idx"),
        (slm.SinkhornConvN, "every"),
    )

    for function, parameter in legitimate_nothing:
        assert (
            inspect.signature(function).parameters[parameter].default is None
        )


def test_padout_distinguishes_default_fill_from_explicit_nothing() -> None:
    values = np.asarray([1.0, 2.0])
    np.testing.assert_array_equal(
        slm.padout(values, 1), [0.0, 1.0, 2.0, 0.0]
    )

    with pytest.raises(TypeError, match="nothing"):
        slm.padout(values, 1, None)
    with pytest.raises(TypeError, match="does not accept a filler"):
        slm.padout(range(2), 1, None)
    with pytest.raises(TypeError, match="does not accept a filler"):
        slm.padout((range(2),), 1, None)

    empty = slm.padout(np.empty(0), 1, None)
    assert empty.dtype == np.dtype(object)
    assert empty.tolist() == [None, None]

    all_nothing = np.asarray([None, None], dtype=object)
    padded_nothing = slm.padout(all_nothing, 1, None)
    assert padded_nothing.dtype == np.dtype(object)
    assert padded_nothing.tolist() == [None, None, None, None]

    for incompatible in (
        np.asarray([None, 1], dtype=object),
        np.asarray([1, 2], dtype=object),
    ):
        with pytest.raises(TypeError, match="nothing"):
            slm.padout(incompatible, 1, None)

    nothing_field = slm.LF[slm.Generic](
        all_nothing, (range(1, 3),)
    )
    padded_field = slm.padout(nothing_field, (1,), None)
    assert padded_field.dtype == np.dtype(object)
    assert padded_field.data.copy().tolist() == [None, None, None, None]

    mixed_field = slm.LF[slm.Generic](
        np.asarray([None, 1], dtype=object), (range(1, 3),)
    )
    with pytest.raises(TypeError, match="nothing"):
        slm.padout(mixed_field, (1,), None)

    empty_field = slm.LF[slm.Intensity](
        np.empty(0), (range(1, 1),)
    )
    with pytest.raises(IndexError, match="empty lattice axis"):
        slm.padout(empty_field, (1,), None)


@pytest.mark.parametrize("resampler", [slm.downsample, slm.upsample])
def test_resampling_passes_explicit_nothing_but_synthesizes_omitted_zero(
    resampler,
) -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    source = (range(1, 5),)
    target = (range(0, 3),)

    omitted = resampler(values, source, target)
    explicit = resampler(values, source, target, bc=None)
    np.testing.assert_array_equal(omitted, [0.0, 1.0, 2.0])
    assert explicit.dtype == np.dtype(object)
    assert explicit.tolist() == [None, 1.0, 2.0]

    with pytest.raises(TypeError, match="does not accept bc"):
        resampler(range(1, 5), 2, bc=None)
    with pytest.raises(TypeError, match="does not accept bc"):
        resampler((range(1, 5),), 2, bc=None)


def test_coarsen_distinguishes_default_reducer_from_explicit_nothing() -> None:
    values = np.arange(1.0, 5.0).reshape((2, 2), order="F")
    np.testing.assert_array_equal(slm.coarsen(values, 1), values)

    with pytest.raises(TypeError, match="not callable"):
        slm.coarsen(values, 1, reducer=None)

    empty = slm.coarsen(
        np.empty((0, 2), dtype=np.float64),
        (1, 1),
        reducer=None,
    )
    assert empty.shape == (0, 2)
    assert empty.dtype == np.dtype(object)


def test_gs_omission_selects_random_overload_but_none_does_not() -> None:
    source_lattice = slm.natlat((4,))
    target_lattice = slm.dualShiftLattice(source_lattice)
    source = slm.LF[slm.Modulus](np.ones(4), source_lattice)
    target = slm.LF[slm.Modulus](np.ones(4), target_lattice)

    assert slm.gs(source, target, 0).shape == (4,)
    assert slm.gsLog(source, target, 0)[0].shape == (4,)
    for function in (slm.gs, slm.gsLog):
        with pytest.raises(TypeError, match="phase LatticeField"):
            function(source, target, 0, None)


def test_cost_and_scalar_potential_distinguish_omitted_arguments() -> None:
    lattice = (range(1, 5),)
    assert slm.getCostMatrix(lattice).shape == (4, 4)
    with pytest.raises(TypeError):
        slm.getCostMatrix(lattice, None)

    vector_field = np.ones((4, 1), dtype=np.float64)
    expected = slm.scalarPotentialN(vector_field, lattice)
    np.testing.assert_array_equal(
        slm.scalarPotentialN(vector_field, lattice, idx=None),
        expected,
    )
    with pytest.raises(TypeError):
        slm.scalarPotentialN(vector_field, lattice, dimOrder=None)


def test_parse_filename_omission_is_not_an_explicit_nothing_cue() -> None:
    assert slm.parseFileName("12.txt") == 12
    with pytest.raises(TypeError, match="cue"):
        slm.parseFileName("12.txt", None)


def test_dualate_passes_explicit_nothing_instead_of_default_zero() -> None:
    lattice = (range(2), range(2))
    field = slm.LF[slm.Generic](np.zeros((2, 2)), lattice)
    boundaries: list[object] = []

    class ConstantInterpolator:
        def __call__(self, *_coordinates):
            return 0.0

    def factory(_ranges, _values, *, extrapolation_bc):
        boundaries.append(extrapolation_bc)
        return ConstantInterpolator()

    slm.dualate(field, lattice, [0.0, 0.0], 0.0, interpolation=factory)
    slm.dualate(
        field,
        lattice,
        [0.0, 0.0],
        0.0,
        interpolation=factory,
        bc=None,
    )

    assert boundaries[0] == 0.0
    assert boundaries[1] is None


def test_template_centers_and_parabola_linear_term_reject_none() -> None:
    lattice = slm.natlat((3, 3))
    calls = (
        lambda: slm.lfRand(slm.Intensity, lattice, center=None),
        lambda: slm.lfParabola(
            slm.Intensity, lattice, 1.0, center=None
        ),
        lambda: slm.lfGaussian(
            slm.Intensity, lattice, 1.0, center=None
        ),
        lambda: slm.lfRing(
            slm.Intensity, lattice, 1.0, 0.5, center=None
        ),
        lambda: slm.lfCap(
            slm.Intensity, lattice, 1.0, 1.0, center=None
        ),
        lambda: slm.lfRect(
            slm.Intensity, lattice, (1.0, 1.0), center=None
        ),
        lambda: slm.lfText(
            slm.Intensity, lattice, "A", center=None
        ),
        lambda: slm.lfHeart(
            slm.Intensity, lattice, 1.0, center=None
        ),
        lambda: slm.lfSmile(
            slm.Intensity, lattice, 1.0, center=None
        ),
        lambda: slm.lfPointer(
            slm.Intensity, lattice, 1.0, center=None
        ),
    )

    for call in calls:
        with pytest.raises(TypeError, match="center"):
            call()
    with pytest.raises(TypeError, match="linear coefficients"):
        slm.lfParabola(slm.Intensity, lattice, 1.0, None)


@pytest.mark.parametrize(
    ("function", "positional", "keyword", "explicit_values"),
    [
        (
            slm.lfParabola,
            (1.0,),
            "lin",
            ((0.0, 0.0), None),
        ),
        (slm.lfGaussian, (1.0,), "norm", (1.0, None)),
        (
            slm.lfRect,
            ((1.0, 1.0),),
            "height",
            (1.0, None),
        ),
    ],
)
@pytest.mark.parametrize("field_template", [False, True])
def test_template_optional_positionals_do_not_become_keywords(
    function,
    positional: tuple[object, ...],
    keyword: str,
    explicit_values: tuple[object, ...],
    field_template: bool,
) -> None:
    lattice = slm.natlat((2, 2))
    template = (
        slm.LF[slm.Generic](np.ones((2, 2)), lattice)
        if field_template
        else slm.Generic
    )
    arguments = (
        (template, *positional)
        if field_template
        else (template, lattice, *positional)
    )

    for value in explicit_values:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            function(*arguments, **{keyword: value})


@pytest.mark.parametrize("function", [slm.lfRand, slm.lfText])
@pytest.mark.parametrize("field_template", [False, True])
@pytest.mark.parametrize("invalid_type", [None, "float64", np.float64(1.0)])
def test_template_R_requires_a_julia_data_type_before_body_work(
    function, field_template: bool, invalid_type: object
) -> None:
    lattice = slm.natlat((2, 2))
    template = (
        slm.LF[slm.Generic](np.ones((2, 2)), lattice)
        if field_template
        else slm.Generic
    )
    positional = ("A",) if function is slm.lfText else ()
    arguments = (
        (template, *positional)
        if field_template
        else (template, lattice, *positional)
    )

    with pytest.raises(TypeError, match="R must be a Julia DataType"):
        function(*arguments, R=invalid_type)


def test_template_R_validation_precedes_rng_and_text_render(monkeypatch) -> None:
    import slmtools.templates as templates

    lattice = slm.natlat((2, 2))

    def unexpected_work(*_args, **_kwargs):
        raise AssertionError("template body ran before validating R")

    monkeypatch.setattr(templates.np.random, "random", unexpected_work)
    with pytest.raises(TypeError, match="R must be a Julia DataType"):
        slm.lfRand(slm.Generic, lattice, R=None)

    monkeypatch.setattr(templates, "ftaText", unexpected_work)
    with pytest.raises(TypeError, match="R must be a Julia DataType"):
        slm.lfText(slm.Generic, lattice, "A", R=None)


@pytest.mark.parametrize("function", [slm.lfHeart, slm.lfSmile, slm.lfPointer])
@pytest.mark.parametrize("field_template", [False, True])
@pytest.mark.parametrize("invalid_flip", [None, 0, 1, "false", np.int64(0)])
def test_emoji_flip_uses_julia_bool_condition_semantics(
    function, field_template: bool, invalid_flip: object
) -> None:
    lattice = slm.natlat((2, 2))
    template = (
        slm.LF[slm.Generic](np.ones((2, 2)), lattice)
        if field_template
        else slm.Generic
    )
    arguments = (
        (template, 1.0)
        if field_template
        else (template, lattice, 1.0)
    )

    with pytest.raises(TypeError, match="flip must be a Julia Bool"):
        function(*arguments, flip=invalid_flip)


def test_full_typed_field_rejects_none_dtype_parameter() -> None:
    with pytest.raises(TypeError, match="second LatticeField parameter"):
        slm.LF[slm.Generic, None, 1]

    for invalid_dimension in (True, np.bool_(True), np.int32(1), np.uint64(1)):
        with pytest.raises(TypeError, match="Julia platform Int"):
            slm.LF[slm.Generic, np.float64, invalid_dimension]

    lattice = slm.natlat((2,))
    result = slm.LF[slm.Generic, np.float64, np.int64(1)](
        np.ones(2), lattice
    )
    assert result.shape == (2,)

    # NumPy's concrete Bool scalar remains the direct Python counterpart of
    # Julia Bool in template conditions; it must not be rejected with integers.
    lattice = slm.natlat((2, 2))
    result = slm.lfHeart(
        slm.Generic, lattice, 1.0, flip=np.bool_(False)
    )
    assert result.shape == (2, 2)


@pytest.mark.parametrize("function", [slm.ftaText, slm.lfText])
@pytest.mark.parametrize("keyword", ["halign", "valign"])
@pytest.mark.parametrize("invalid_alignment", [None, 0])
def test_text_alignments_require_julia_symbol_counterparts(
    monkeypatch,
    function,
    keyword: str,
    invalid_alignment: object,
) -> None:
    import slmtools.templates as templates

    # Julia resolves the font and then rejects the concretely typed Symbol
    # keyword at the renderstring! call boundary.  Avoid depending on a host
    # font while proving that no glyph work can follow the type rejection.
    monkeypatch.setattr(templates, "_load_font", lambda *_args: object())
    arguments = (
        ("A", (2, 2))
        if function is slm.ftaText
        else (slm.Intensity, slm.natlat((2, 2)), "A")
    )

    with pytest.raises(TypeError, match="must be a Julia Symbol"):
        function(*arguments, pixelsize=1, **{keyword: invalid_alignment})


@pytest.mark.parametrize(
    "call",
    [
        lambda: slm.dualLattice((), None),
        lambda: slm.dualShiftLattice((), None),
        lambda: slm.ldq((), (), None),
    ],
)
def test_empty_lattice_does_not_bypass_number_dispatch(call) -> None:
    with pytest.raises(TypeError, match="flambda must be a Julia Number"):
        call()
