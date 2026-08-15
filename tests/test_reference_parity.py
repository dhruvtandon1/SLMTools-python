"""Small cross-language goldens and package-wide acceptance checks.

The dense-OT values below were generated with Julia SLMTools commit
ea1c1c9c06b4b2dc46372ac7ee031301b604a007, the manifest's exact Julia
1.11.6 runtime, and OptimalTransport 0.3.20.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import re
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFont

import slmtools as slm
from slmtools import subimages
from slmtools._omission import _OMITTED
from slmtools.templates import _UNSET


_AUDITED_JULIA_COMMIT = "ea1c1c9c06b4b2dc46372ac7ee031301b604a007"
_EXPORT_FIXTURE = Path(__file__).with_name("julia_exports.txt")

# Union of the declared keyword names, in source order, across each exported
# Julia generic function's local methods.  Python's merged dispatchers expose
# the same union and reject keywords that do not belong to the selected
# overload at call time.
_JULIA_PUBLIC_KEYWORDS = {
    "downsample": ("interpolation", "bc"),
    "coarsen": ("reducer",),
    "upsample": ("interpolation", "bc"),
    "dualPhase": ("dL",),
    "loadDir": ("T", "outType", "L", "flambda", "cue", "look"),
    "parseFileName": ("outType",),
    "parseStringToNum": ("outType",),
    "getOrientation": ("roi", "threshold"),
    "dualate": ("roi", "interpolation", "naturalize", "bc"),
    "saveBeam": ("dir",),
    "lfParabola": ("center", "flambda"),
    "lfGaussian": ("center", "flambda"),
    "lfRing": ("center", "flambda"),
    "lfCap": ("center", "flambda"),
    "ftaText": ("fnt", "pixelsize", "halign", "valign", "options"),
    "lfText": (
        "center",
        "flambda",
        "R",
        "pixelsize",
        "fnt",
        "halign",
        "valign",
        "options",
    ),
    "lfRect": ("center", "flambda"),
    "lfRand": ("center", "flambda", "R"),
    "lfHeart": ("center", "flambda", "flip"),
    "lfSmile": ("center", "flambda", "flip"),
    "lfPointer": ("center", "flambda", "flip"),
    "gsLog": ("every",),
    "pdgsLog": ("every",),
    "getCostMatrix": ("normalization",),
    "pdCostMatrix": ("normalization", "flambda"),
    "pdotPhase": ("options",),
    "pdotBeamEstimate": ("LFine", "options"),
    "scalarPotentialN": ("idx", "dimOrder"),
    "otPhase": ("options",),
    "otQuickPhase": ("return_loss",),
    "SinkhornConvN": ("every",),
    "otPhase2": ("options",),
}

# Exact positional arities declared by the audited Julia methods.  ``N+``
# denotes a vararg method accepting N or more public arguments.  The live
# reflection gate below independently regenerates this fixture.
_JULIA_PUBLIC_POSITIONAL_ARITIES = {
    "Nyquist": ("1",),
    "SchroffError": ("2", "3"),
    "SinkhornConvN": ("4",),
    "castImage": ("4",),
    "centroid": ("1", "2"),
    "clip": ("2",),
    "coarsen": ("2",),
    "collapse": ("2",),
    "downsample": ("2", "3"),
    "dualLattice": ("1", "2"),
    "dualPhase": ("1", "2"),
    "dualShiftLattice": ("1", "2"),
    "dualToGradients": ("5",),
    "dualate": ("4", "5"),
    "elq": ("2",),
    "ftaText": ("2",),
    "getCostMatrix": ("1", "2"),
    "getImagesAndFilenames": ("2",),
    "getOrientation": ("2",),
    "gs": ("3", "4"),
    "gsIter": ("5",),
    "gsLog": ("3", "4"),
    "hyperSum2": ("4",),
    "imageToFloatArray": ("1",),
    "isft": ("1",),
    "itfa": ("1",),
    "latticeDisplacement": ("1",),
    "ldot": ("2",),
    "ldq": ("2", "3"),
    "lfBlur": ("2",),
    "lfCap": ("3", "4"),
    "lfGaussian": ("2", "3", "4"),
    "lfHeart": ("2", "3"),
    "lfParabola": ("2", "3", "4"),
    "lfPointer": ("2", "3"),
    "lfRand": ("1", "2"),
    "lfRect": ("2", "3", "4"),
    "lfRing": ("3", "4"),
    "lfSmile": ("2", "3"),
    "lfText": ("2", "3"),
    "linearFit": ("2",),
    "loadDir": ("2",),
    "look": ("0+", "1"),
    "mapify": ("3",),
    "mraf": ("6",),
    "nabs": ("1",),
    "natlat": ("0+", "1"),
    "natrange": ("1",),
    "naturalize": ("1",),
    "normalizeDistribution": ("1",),
    "normalizeLF": ("1",),
    "oneShot": ("3",),
    "otPhase": ("3",),
    "otPhase2": ("4",),
    "otQuickPhase": ("4",),
    "padout": ("2", "3"),
    "parseFileName": ("1", "2", "3"),
    "parseStringToNum": ("1",),
    "pdCostMatrix": ("4",),
    "pdgs": ("4",),
    "pdgsError": ("3",),
    "pdgsIter": ("5",),
    "pdgsLog": ("4",),
    "pdotBeamEstimate": ("7",),
    "pdotPhase": ("7",),
    "phasor": ("1",),
    "r2": ("1",),
    "ramp": ("1",),
    "safeInverse": ("1",),
    "saveBeam": ("2", "3"),
    "savePhase": ("2",),
    "savePhase8BMP": ("2",),
    "save_gray8bmp": ("2",),
    "scalarPotentialN": ("2",),
    "sft": ("1",),
    "square": ("0+", "1"),
    "subfield": ("1+",),
    "sublattice": ("1+", "2"),
    "toDim": ("3",),
    "upsample": ("2", "3"),
    "window": ("2",),
    "wrap": ("1",),
}

# These merged Python dispatchers deliberately expose ``*args`` and validate
# the finite Julia overload set after inspecting the template/value type.
# ``square`` is different: Julia's extra vararg method is an error-only
# fallback, so Python exposes only the one successful arity.
_POSITIONAL_SIGNATURE_ADAPTATIONS = {
    "downsample": (("1+",), ("2", "3")),
    "upsample": (("1+",), ("2", "3")),
    "lfParabola": (("1+",), ("2", "3", "4")),
    "lfGaussian": (("1+",), ("2", "3", "4")),
    "lfRing": (("1+",), ("3", "4")),
    "lfCap": (("1+",), ("3", "4")),
    "lfText": (("1+",), ("2", "3")),
    "lfRect": (("1+",), ("2", "3", "4")),
    "lfRand": (("1+",), ("1", "2")),
    "lfHeart": (("1+",), ("2", "3")),
    "lfSmile": (("1+",), ("2", "3")),
    "lfPointer": (("1+",), ("2", "3")),
    "square": (("1",), ("0+",)),
}

# Julia does not expose keyword default expressions through a stable public
# reflection API.  Defaults that depend on an overload's type/dimension or on
# call time are therefore kept as sentinels and recorded with their exact
# audited source expressions here; all remaining defaults are compared by a
# language-neutral token below.
_DEFERRED_JULIA_KEYWORD_DEFAULTS = {
    ("downsample", "interpolation"): "cubic_spline_interpolation",
    ("downsample", "bc"): "zero(T)",
    ("coarsen", "reducer"): "sum(x) / length(x[:])",
    ("upsample", "interpolation"): "cubic_spline_interpolation",
    ("upsample", "bc"): "zero(T)",
    ("dualate", "bc"): "zero(T)",
    ("saveBeam", "dir"): "pwd()",
    ("lfParabola", "center"): "Tuple(0.0 for i=1:N)",
    ("lfParabola", "flambda"): "1.0 for the type/lattice overload",
    ("lfGaussian", "center"): "Tuple(0.0 for i=1:N)",
    ("lfGaussian", "flambda"): "1.0 for the type/lattice overload",
    ("lfRing", "center"): "Tuple(0.0 for i=1:N)",
    ("lfRing", "flambda"): "1.0 for the type/lattice overload",
    ("lfCap", "center"): "Tuple(0.0 for i=1:N)",
    ("lfCap", "flambda"): "1.0 for the type/lattice overload",
    ("lfText", "center"): "Tuple(0.0 for i=1:N)",
    ("lfText", "flambda"): "1.0 for the type/lattice overload",
    ("lfRect", "center"): "Tuple(0.0 for i=1:N)",
    ("lfRect", "flambda"): "1.0 for the type/lattice overload",
    ("lfRand", "center"): "Tuple(0.0 for i=1:N)",
    ("lfRand", "flambda"): "1.0 for the type/lattice overload",
    ("lfHeart", "center"): "Tuple(0.0 for i=1:N)",
    ("lfHeart", "flambda"): "1.0 for the type/lattice overload",
    ("lfSmile", "center"): "Tuple(0.0 for i=1:N)",
    ("lfSmile", "flambda"): "1.0 for the type/lattice overload",
    ("lfPointer", "center"): "Tuple(0.0 for i=1:N)",
    ("lfPointer", "flambda"): "1.0 for the type/lattice overload",
    ("scalarPotentialN", "dimOrder"): "1:N",
}

_STATIC_JULIA_KEYWORD_DEFAULTS = {
    ("dualPhase", "dL"): "nothing",
    ("loadDir", "T"): "Intensity",
    ("loadDir", "outType"): "Float64",
    ("loadDir", "L"): "nothing",
    ("loadDir", "flambda"): "1.0",
    ("loadDir", "cue"): "nothing",
    ("loadDir", "look"): ":after",
    ("parseFileName", "outType"): "nothing",
    ("parseStringToNum", "outType"): "nothing",
    ("getOrientation", "roi"): "nothing",
    ("getOrientation", "threshold"): "0.1",
    ("dualate", "roi"): "nothing",
    ("dualate", "interpolation"): "cubic_spline_interpolation",
    ("dualate", "naturalize"): "false",
    ("ftaText", "fnt"): '"arial bold"',
    ("ftaText", "pixelsize"): "nothing",
    ("ftaText", "halign"): ":hcenter",
    ("ftaText", "valign"): ":vcenter",
    ("lfText", "R"): "Float64",
    ("lfText", "pixelsize"): "nothing",
    ("lfText", "fnt"): '"arial bold"',
    ("lfText", "halign"): ":hcenter",
    ("lfText", "valign"): ":vcenter",
    ("lfRand", "R"): "Float64",
    ("lfHeart", "flip"): "false",
    ("lfSmile", "flip"): "false",
    ("lfPointer", "flip"): "false",
    ("gsLog", "every"): "1",
    ("pdgsLog", "every"): "1",
    ("getCostMatrix", "normalization"): "maximum",
    ("pdCostMatrix", "normalization"): "maximum",
    ("pdCostMatrix", "flambda"): "1.0",
    ("pdotBeamEstimate", "LFine"): "nothing",
    ("scalarPotentialN", "idx"): "nothing",
    ("otQuickPhase", "return_loss"): "false",
    ("SinkhornConvN", "every"): "nothing",
}


def _fixture_exports() -> tuple[str, ...]:
    return tuple(
        line
        for raw_line in _EXPORT_FIXTURE.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    )


def _source_exports(julia_repo: Path) -> tuple[str, ...]:
    module_source = (
        julia_repo / "src" / "SLMTools.jl"
    ).read_text(encoding="utf-8")
    include_paths = re.findall(r'^\s*include\("([^"]+)"\)', module_source, re.M)
    exports: list[str] = []
    for include_path in include_paths:
        source = (
            julia_repo / "src" / include_path
        ).read_text(encoding="utf-8")
        for declaration in re.findall(
            r"^\s*export\s+([^\r\n]+)", source, re.M
        ):
            for name in declaration.split(","):
                clean_name = name.strip()
                if clean_name and clean_name not in exports:
                    exports.append(clean_name)
    return tuple(exports)


def _runtime_public_contract(
    julia_repo: Path,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    script = r'''
        using SLMTools
        for name_string in split(ARGS[1], ",")
            name = Symbol(name_string)
            value = getfield(SLMTools, name)
            value isa Function || continue
            arities = String[]
            keywords = String[]
            for method in methods(value)
                method.module === SLMTools || continue
                arity = method.isva ? string(method.nargs - 2, "+") : string(method.nargs - 1)
                arity in arities || push!(arities, arity)
                for raw_keyword in Base.kwarg_decl(method)
                    keyword = replace(string(raw_keyword), r"\.\.\.$" => "")
                    keyword in ("", "kwargs") && continue
                    keyword in keywords || push!(keywords, keyword)
                end
            end
            sort!(arities; by=arity -> (
                parse(Int, replace(arity, "+" => "")),
                endswith(arity, "+") ? 0 : 1,
            ))
            println(name_string, '\t', join(arities, ','), '\t', join(keywords, ','))
        end
    '''
    completed = subprocess.run(
        [
            "julia",
            f"--project={julia_repo}",
            "--startup-file=no",
            "-e",
            script,
            ",".join(_fixture_exports()),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    arity_result: dict[str, tuple[str, ...]] = {}
    keyword_result: dict[str, tuple[str, ...]] = {}
    for line in completed.stdout.splitlines():
        name, arity_text, keyword_text = line.split("\t")
        arity_result[name] = tuple(arity_text.split(","))
        if keyword_text:
            keyword_result[name] = tuple(keyword_text.split(","))
    return arity_result, keyword_result


def _python_positional_arity(function: object) -> tuple[str, ...]:
    parameters = tuple(inspect.signature(function).parameters.values())
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )
    minimum = sum(
        parameter.default is inspect.Parameter.empty
        for parameter in positional
    )
    if any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters
    ):
        return (f"{minimum}+",)
    return tuple(str(arity) for arity in range(minimum, len(positional) + 1))


def _semantic_arity_union(arities: tuple[str, ...]) -> tuple[str, ...]:
    vararg_minima = tuple(
        int(arity[:-1]) for arity in arities if arity.endswith("+")
    )
    if not vararg_minima:
        return arities
    minimum = min(vararg_minima)
    return tuple(
        arity
        for arity in arities
        if (
            arity.endswith("+") and int(arity[:-1]) == minimum
        ) or (
            not arity.endswith("+") and int(arity) < minimum
        )
    )


def _static_keyword_default_token(default: object) -> str:
    if default is None:
        return "nothing"
    if default is slm.Intensity:
        return "Intensity"
    if default is np.float64:
        return "Float64"
    if default is slm.cubic_spline_interpolation:
        return "cubic_spline_interpolation"
    if default is np.max:
        return "maximum"
    if default is False:
        return "false"
    if default in ("after", "hcenter", "vcenter"):
        return f":{default}"
    if default == "arial bold":
        return '"arial bold"'
    return repr(default)


class ReferenceParityTests(unittest.TestCase):
    def test_exact_julia_export_surface(self) -> None:
        expected = _fixture_exports()
        self.assertEqual(len(expected), 100)
        self.assertEqual(len(set(expected)), 100)
        self.assertEqual(tuple(slm.JULIA_EXPORTS), expected)
        self.assertEqual(tuple(slm.__all__), expected)
        self.assertTrue(all(hasattr(slm, name) for name in expected))
        self.assertIs(slm.LF, slm.LatticeField)
        self.assertIs(slm.UPhase, slm.RealPhase)
        self.assertIs(slm.UnwrappedPhase, slm.RealPhase)
        self.assertIs(slm.S1Phase, slm.ComplexPhase)
        self.assertIs(slm.RealAmplitude, slm.Modulus)
        self.assertIs(slm.RealAmp, slm.Modulus)
        self.assertIs(slm.ComplexAmp, slm.ComplexAmplitude)
        self.assertIs(slm.itfa, slm.imageToFloatArray)
        self.assertTrue(issubclass(slm.Generic, slm.FieldVal))
        for qualified in (
            "shiftedDFTBasis",
            "hermiteBasis",
            "FrFTBasis",
            "wigner_fft",
            "cubic_spline_interpolation",
            "heartQ",
            "gsError",
            "hyperSum",
            "rampPrivate",
            "SinkhornIterBase!",
        ):
            self.assertTrue(callable(getattr(slm, qualified)))
        self.assertIsNot(slm.rampPrivate, slm.ramp)
        self.assertEqual(slm.rampPrivate(-3.0), slm.ramp(-3.0))
        self.assertFalse(hasattr(slm, "julia_add"))
        self.assertFalse(hasattr(slm, "julia_mul"))

    def test_public_keyword_names_and_order_match_julia(self) -> None:
        actual = {}
        for name in slm.JULIA_EXPORTS:
            function = getattr(slm, name)
            if not inspect.isfunction(function):
                continue
            keywords = tuple(
                parameter.name
                for parameter in inspect.signature(function).parameters.values()
                if parameter.kind
                in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
            )
            if keywords:
                actual[name] = keywords

        self.assertEqual(actual, _JULIA_PUBLIC_KEYWORDS)

    def test_every_declared_keyword_default_matches_julia(self) -> None:
        observed = set()
        for function_name, keyword_names in _JULIA_PUBLIC_KEYWORDS.items():
            parameters = inspect.signature(
                getattr(slm, function_name)
            ).parameters
            for keyword_name in keyword_names:
                parameter = parameters[keyword_name]
                if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    continue
                key = (function_name, keyword_name)
                observed.add(key)
                if key in _DEFERRED_JULIA_KEYWORD_DEFAULTS:
                    self.assertIn(parameter.default, (_OMITTED, _UNSET))
                else:
                    self.assertEqual(
                        _static_keyword_default_token(parameter.default),
                        _STATIC_JULIA_KEYWORD_DEFAULTS[key],
                    )

        self.assertEqual(
            observed,
            set(_DEFERRED_JULIA_KEYWORD_DEFAULTS)
            | set(_STATIC_JULIA_KEYWORD_DEFAULTS),
        )

    def test_public_positional_signatures_match_or_are_classified(self) -> None:
        exported_functions = {
            name
            for name in slm.JULIA_EXPORTS
            if inspect.isfunction(getattr(slm, name))
        }
        self.assertEqual(
            set(_JULIA_PUBLIC_POSITIONAL_ARITIES), exported_functions
        )
        actual_adaptations = {}
        for name, julia_arities in _JULIA_PUBLIC_POSITIONAL_ARITIES.items():
            python_arities = _python_positional_arity(getattr(slm, name))
            semantic_julia_arities = _semantic_arity_union(julia_arities)
            if python_arities != semantic_julia_arities:
                actual_adaptations[name] = (
                    python_arities,
                    semantic_julia_arities,
                )

        self.assertEqual(actual_adaptations, _POSITIONAL_SIGNATURE_ADAPTATIONS)

    def test_merged_dispatchers_enforce_every_julia_positional_arity(self) -> None:
        source_lattice = (range(1, 5),)
        target_lattice = (range(2, 4),)
        signal = np.arange(4.0)
        template_lattice = slm.natlat((3, 3))
        template = slm.LF[slm.Generic](
            np.ones((3, 3)), template_lattice
        )
        modulus = slm.LF[slm.Modulus](
            np.ones((3, 3)), template_lattice
        )
        valid_calls = {
            "downsample": (
                (range(1, 5), 2),
                (signal, source_lattice, target_lattice),
            ),
            "upsample": (
                (range(1, 5), 2),
                (signal, source_lattice, target_lattice),
            ),
            "lfParabola": (
                (template, 1.0),
                (slm.Generic, template_lattice, 1.0),
                (slm.Generic, template_lattice, 1.0, (0.0, 0.0)),
            ),
            "lfGaussian": (
                (template, 1.0),
                (slm.Generic, template_lattice, 1.0),
                (slm.Generic, template_lattice, 1.0, 1.0),
            ),
            "lfRing": (
                (template, 1.0, 1.0),
                (slm.Generic, template_lattice, 1.0, 1.0),
            ),
            "lfCap": (
                (template, 1.0, 1.0),
                (slm.Generic, template_lattice, 1.0, 1.0),
            ),
            "lfText": (
                (template, "A"),
                (slm.Generic, template_lattice, "A"),
            ),
            "lfRect": (
                (template, (1.0, 1.0)),
                (slm.Generic, template_lattice, (1.0, 1.0)),
                (slm.Generic, template_lattice, (1.0, 1.0), 1.0),
            ),
            "lfRand": (
                (template,),
                (slm.Generic, template_lattice),
            ),
            "lfHeart": (
                (template, 1.0),
                (slm.Generic, template_lattice, 1.0),
            ),
            "lfSmile": (
                (template, 1.0),
                (slm.Generic, template_lattice, 1.0),
            ),
            "lfPointer": (
                (template, 1.0),
                (slm.Generic, template_lattice, 1.0),
            ),
        }

        with patch(
            "slmtools.templates.ftaText",
            return_value=np.ones((3, 3)),
        ):
            for name, calls in valid_calls.items():
                function = getattr(slm, name)
                for arguments in calls:
                    with self.subTest(function=name, arity=len(arguments)):
                        function(*arguments)

                with self.subTest(function=name, arity=0):
                    with self.assertRaises(TypeError):
                        function()
                with self.subTest(function=name, arity="too-many"):
                    with self.assertRaises(TypeError):
                        function(*calls[-1], object())

        squared = slm.square(modulus)
        self.assertIs(squared.field_type, slm.Intensity)
        with self.assertRaises(TypeError):
            slm.square()
        with self.assertRaises(TypeError):
            slm.square(modulus, modulus)

    @unittest.skipUnless(
        os.environ.get("SLMTOOLS_JULIA_REPO"),
        "set SLMTOOLS_JULIA_REPO for live Julia export extraction",
    )
    def test_export_fixture_is_derived_from_audited_julia_source(self) -> None:
        julia_repo = Path(os.environ["SLMTOOLS_JULIA_REPO"])
        self.assertEqual(_source_exports(julia_repo), _fixture_exports())

    @unittest.skipUnless(
        os.environ.get("SLMTOOLS_JULIA_REPO"),
        "set SLMTOOLS_JULIA_REPO for live Julia signature extraction",
    )
    def test_signature_fixtures_are_derived_from_audited_julia_runtime(self) -> None:
        julia_repo = Path(os.environ["SLMTOOLS_JULIA_REPO"])
        arities, keywords = _runtime_public_contract(julia_repo)
        self.assertEqual(arities, _JULIA_PUBLIC_POSITIONAL_ARITIES)
        self.assertEqual(keywords, _JULIA_PUBLIC_KEYWORDS)

    def test_exported_julia_abstract_tags_are_not_instantiable(self) -> None:
        tags = (
            slm.FieldVal,
            slm.Generic,
            slm.Phase,
            slm.RealPhase,
            slm.ComplexPhase,
            slm.Intensity,
            slm.Amplitude,
            slm.Modulus,
            slm.ComplexAmplitude,
        )
        for tag in tags:
            with self.subTest(tag=tag.__name__):
                with self.assertRaisesRegex(TypeError, "abstract field tag"):
                    tag()

    def test_user_field_tag_subclass_is_concrete_and_instantiable(self) -> None:
        class UserFieldTag(slm.FieldVal):
            def __init__(self, label: str) -> None:
                self.label = label

        class UserPhaseTag(slm.Phase):
            pass

        tag = UserFieldTag("custom")
        self.assertIsInstance(tag, slm.FieldVal)
        self.assertEqual(tag.label, "custom")
        self.assertIsInstance(UserPhaseTag(), slm.Phase)

        field = slm.LF[UserFieldTag](
            np.ones(2), (range(2),)
        )
        self.assertIs(field.field_type, UserFieldTag)

    def test_remaining_template_and_scalar_entry_points(self) -> None:
        lattice = slm.natlat((12, 12))
        np.random.seed(7)
        random_field = slm.lfRand(slm.Intensity, lattice)
        from unittest.mock import patch

        with patch(
            "slmtools.templates._load_font",
            return_value=ImageFont.load_default(),
        ):
            text_field = slm.lfText(
                slm.Intensity,
                lattice,
                "A",
                pixelsize=8,
            )
        generic = slm.LF(np.zeros((12, 12)), lattice)
        self.assertEqual(random_field.shape, (12, 12))
        self.assertGreater(np.max(text_field.data), 0)
        self.assertIs(generic.field_type, slm.Generic)
        self.assertEqual(slm.ramp(-3.0), 0.0)

    def test_dense_ot_matches_julia_golden(self) -> None:
        lattice = slm.natlat((4, 4))
        dual = slm.dualShiftLattice(lattice)
        source = slm.lfGaussian(slm.Intensity, lattice, 0.8)
        target = slm.lfRing(slm.Intensity, dual, 0.5, 0.4)
        phase = slm.otPhase(source, target, 0.1, maxiter=50)

        expected = np.asarray(
            [
                0.49427105834925306,
                0.2566597567738322,
                0.1792217920522724,
                0.25400311824030514,
                0.2566597567738321,
                0.0,
                -0.08790203141485209,
                -0.009288840596768988,
                0.1795792330808621,
                -0.08790203141485209,
                -0.18237886770350065,
                -0.10156412809828824,
                0.2530199367901494,
                -0.009288840596768988,
                -0.10094500122561556,
                -0.022220416553216382,
            ]
        )
        np.testing.assert_allclose(
            phase.data.ravel(order="F"), expected, rtol=2e-14, atol=2e-15
        )
        self.assertIs(phase.field_type, slm.RealPhase)

    def test_exact_readme_quick_start(self) -> None:
        size = 16
        lattice = slm.natlat((size, size))
        dual = slm.dualShiftLattice(lattice)
        source = slm.lfGaussian(slm.Intensity, lattice, 1.0)
        target = slm.lfRing(slm.Intensity, dual, 2.5, 0.5)
        phase_ot = slm.otPhase(source, target, 0.002)
        phase_ot2 = slm.otPhase2(source, target, 0.0002, 200)
        phase_gs = slm.gs(source, target, 100, phase_ot)
        output_ot = slm.square(slm.sft(np.sqrt(source) * phase_ot))
        output_ot2 = slm.square(slm.sft(np.sqrt(source) * phase_ot2))
        output_gs = slm.square(slm.sft(np.sqrt(source) * phase_gs))
        display = slm.look(target, output_ot, output_ot2, output_gs)
        self.assertEqual(display.shape, (size, 4 * size))
        self.assertTrue(np.all(np.isfinite(output_ot.data)))
        self.assertTrue(np.all(np.isfinite(output_ot2.data)))
        self.assertTrue(np.all(np.isfinite(output_gs.data)))
        self.assertIs(phase_gs.field_type, slm.ComplexPhase)

    def test_subimages_keeps_julia_grid_order_and_upstream_padall_behavior(self) -> None:
        cells = tuple(np.full((1, 1), value, dtype=float) for value in range(1, 5))
        grid = subimages.arrange((2, 2), *cells)
        merged = subimages.mergeStrict(grid)
        np.testing.assert_array_equal(merged, np.asarray([[1.0, 3.0], [2.0, 4.0]]))
        padded = subimages.padmultiple(cells[0], padall=2, fillval=-1)
        np.testing.assert_array_equal(padded, cells[0])
        repeated = subimages.padmultiple(
            cells[0], padleft=1, padtop=2, padall=7, fillval=-1
        )
        self.assertEqual(repeated.shape, (5, 3))
        self.assertEqual(repeated[4, 2], 1)

    def test_subimages_composition_and_plot_adapter(self) -> None:
        grayscale = np.asarray([[1.0, 0.0], [1.0, 1.0]])
        color = np.dstack((grayscale, grayscale, grayscale))
        promoted = subimages.colorPromote([[grayscale, color]])
        self.assertTrue(all(np.asarray(cell).shape[-1] == 3 for cell in promoted.flat))
        self.assertFalse(subimages.checkCommonSize([[np.ones((1, 2)), np.ones((2, 2))]]))
        merged = subimages.mergeFill([[np.ones((1, 2)), np.ones((2, 1))]], fillval=0)
        self.assertEqual(merged.shape, (2, 3))
        np.testing.assert_array_equal(
            subimages.trimWhitespace(np.pad(np.zeros((1, 1)), 1, constant_values=1)),
            np.zeros((1, 1)),
        )
        self.assertEqual(subimages.padadd(np.ones((1, 1)), 2, "left").shape, (1, 3))

        class FakePlot:
            @staticmethod
            def savefig(stream: object, *, format: str) -> None:
                assert format == "png"
                Image.new("RGB", (2, 3), (1, 2, 3)).save(stream, format="PNG")

        self.assertEqual(subimages.plotToImage(FakePlot()).shape, (3, 2, 4))

    @unittest.skipUnless(
        os.environ.get("SLMTOOLS_JULIA_REPO"),
        "set SLMTOOLS_JULIA_REPO for the large read-only image integration",
    )
    def test_original_orientation_fixture_read_only(self) -> None:
        original = Path(os.environ["SLMTOOLS_JULIA_REPO"])
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(original),
                    "merge-base",
                    "--is-ancestor",
                    _AUDITED_JULIA_COMMIT,
                    "HEAD",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(original),
                    "diff",
                    "--quiet",
                    _AUDITED_JULIA_COMMIT,
                    "HEAD",
                    "--",
                    ":(glob)src/**/*.jl",
                    "Project.toml",
                    "Manifest.toml",
                    ":(glob)test/test_data/test_images_B/LinearPhases/**",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(original),
                    "diff",
                    "--quiet",
                    _AUDITED_JULIA_COMMIT,
                    "--",
                    ":(glob)src/**/*.jl",
                    "Project.toml",
                    "Manifest.toml",
                    ":(glob)test/test_data/test_images_B/LinearPhases/**",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            self.fail(
                "SLMTOOLS_JULIA_REPO must descend from the audited commit "
                "without changing Julia source, its environment, or the "
                "required fixture files"
            )
        directory = original / "test/test_data/test_images_B/LinearPhases"
        fields, indices = slm.loadDir(str(directory) + os.sep, ".bmp")
        roi = (slice(539, 840), slice(899, 1200))
        center, angle = slm.getOrientation(fields[18:], indices[18:], roi=roi)
        np.testing.assert_allclose(
            center,
            [483.319089895145, 1009.4572223180837],
            rtol=2e-14,
            atol=2e-12,
        )
        self.assertAlmostEqual(angle, 0.022304388965840503, places=13)


if __name__ == "__main__":
    unittest.main()
