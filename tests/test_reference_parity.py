"""Small cross-language goldens and package-wide acceptance checks.

The dense-OT values below were generated with Julia SLMTools commit
ea1c1c9c06b4b2dc46372ac7ee031301b604a007, Julia 1.12.6 loading the pinned
project manifest, OptimalTransport 0.3.20, and ``--compiled-modules=no``.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np
from PIL import Image, ImageFont

import slmtools as slm
from slmtools import subimages


class ReferenceParityTests(unittest.TestCase):
    def test_exact_julia_export_surface(self) -> None:
        self.assertEqual(len(slm.JULIA_EXPORTS), 100)
        self.assertEqual(len(set(slm.JULIA_EXPORTS)), 100)
        self.assertEqual(slm.__all__, list(slm.JULIA_EXPORTS))
        self.assertTrue(all(hasattr(slm, name) for name in slm.JULIA_EXPORTS))
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
            "SinkhornIterBase!",
        ):
            self.assertTrue(callable(getattr(slm, qualified)))

    def test_remaining_template_and_scalar_entry_points(self) -> None:
        lattice = slm.natlat((12, 12))
        np.random.seed(7)
        random_field = slm.lfRand(slm.Intensity, lattice)
        text_field = slm.lfText(
            slm.Intensity,
            lattice,
            "A",
            fnt=ImageFont.load_default(),
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
                0.49427105834925295,
                0.25665975677383207,
                0.1792217920522723,
                0.2540031182403051,
                0.25665975677383207,
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
                -0.022220416553216327,
            ]
        )
        np.testing.assert_allclose(
            phase.data.ravel(order="F"), expected, rtol=2e-14, atol=2e-15
        )
        self.assertIs(phase.field_type, slm.RealPhase)

    def test_reduced_readme_workflow(self) -> None:
        lattice = slm.natlat((8, 8))
        dual = slm.dualShiftLattice(lattice)
        source = slm.lfGaussian(slm.Intensity, lattice, 1.0)
        target = slm.lfRing(slm.Intensity, dual, 1.0, 0.4)
        phase_ot = slm.otPhase2(source, target, 0.05, 8)
        phase_gs = slm.gs(source, target, 4, phase_ot)
        output = slm.square(slm.sft(np.sqrt(source) * phase_gs))
        display = slm.look(target, output)
        self.assertEqual(display.shape, (8, 16))
        self.assertTrue(np.all(np.isfinite(output.data)))
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

    def test_port_is_not_a_git_repository(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertFalse(any(path.name == ".git" for path in root.rglob(".git")))

    @unittest.skipUnless(
        os.environ.get("SLMTOOLS_JULIA_REPO"),
        "set SLMTOOLS_JULIA_REPO for the large read-only image integration",
    )
    def test_original_orientation_fixture_read_only(self) -> None:
        original = Path(os.environ["SLMTOOLS_JULIA_REPO"])
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
