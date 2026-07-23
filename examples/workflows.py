"""Deterministic, scaled translations of the four Julia example notebooks.

The originals contain publication-scale 1024²/1536² sweeps that run for hours.
These functions preserve each workflow's package usage while using small arrays
and explicit initial phases, making them suitable for documentation and smoke
testing.  They write no files.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from slmtools import (
    Intensity,
    RealPhase,
    dualShiftLattice,
    gs,
    gsError,
    lfGaussian,
    lfParabola,
    lfRect,
    lfRing,
    mraf,
    natlat,
    oneShot,
    otPhase,
    otPhase2,
    pdgs,
    pdgsError,
    pdotBeamEstimate,
    sft,
    square,
)


def phase_generation_methods_comparison(n: int = 32, iterations: int = 20) -> dict[str, float]:
    lattice = natlat((n, n))
    source = lfGaussian(Intensity, lattice, 0.8)
    target = lfRing(Intensity, dualShiftLattice(lattice), 1.0, 0.25)
    phase_ot = otPhase2(source, target, 0.001, max(4, iterations // 2))
    phase_gs = gs(source, target, iterations, phase_ot)
    radius = max(2, n // 3)
    start = (n - radius) // 2
    roi = (slice(start, start + radius), slice(start, start + radius))
    phase_mraf = mraf(np.sqrt(source), np.sqrt(target), iterations, phase_ot, roi, 0.48)
    return {
        "ot_error": gsError(source, target, phase_ot),
        "gs_error": gsError(source, target, phase_gs),
        "mraf_error": gsError(source, target, phase_mraf),
    }


def beam_estimation_performance(n: int = 32, iterations: int = 20) -> dict[str, float]:
    lattice = natlat((n, n))
    beam_intensity = lfGaussian(Intensity, lattice, 0.8)
    alphas = (0.2, 0.5, 0.8)
    phases = tuple(lfParabola(RealPhase, lattice, alpha) for alpha in alphas)
    images = tuple(square(sft(np.sqrt(beam_intensity) * phase)) for phase in phases)
    guess = np.sqrt(beam_intensity) * phases[0]
    estimate = pdgs(images, phases, iterations, guess)
    oneshot = oneShot(images[0], alphas[0], (0.0, 0.0))
    return {
        "pdgs_error": pdgsError(tuple(np.sqrt(image) for image in images), phases, estimate),
        "oneshot_error": pdgsError(tuple(np.sqrt(image) for image in images), phases, oneshot),
    }


def pdot_vs_alpha(n: int = 24) -> dict[str, float]:
    lattice = natlat((n, n))
    beam = lfGaussian(Intensity, lattice, 0.75)
    alpha_root, alpha_target = 0.25, 0.75
    phase_root = lfParabola(RealPhase, lattice, alpha_root)
    phase_target = lfParabola(RealPhase, lattice, alpha_target)
    image_root = square(sft(np.sqrt(beam) * phase_root))
    image_target = square(sft(np.sqrt(beam) * phase_target))
    estimate = pdotBeamEstimate(
        image_root,
        image_target,
        alpha_root,
        alpha_target,
        (0.0, 0.0),
        (0.0, 0.0),
        0.01,
        maxiter=200,
    )
    return {
        "two_image_error": pdgsError(
            (np.sqrt(image_root), np.sqrt(image_target)),
            (phase_root, phase_target),
            estimate,
        )
    }


def paper_ot_initialization_comparison(n: int = 24, iterations: int = 12) -> dict[str, float]:
    lattice = natlat((n, n))
    target_lattice = dualShiftLattice(lattice)
    source = lfGaussian(Intensity, lattice, 0.9)
    target = lfRect(Intensity, target_lattice, (1.2, 0.6))
    phase_dense = otPhase(source, target, 0.01, maxiter=300)
    phase_fast = otPhase2(source, target, 0.002, 40)
    refined = gs(source, target, iterations, phase_fast)
    return {
        "dense_ot_error": gsError(source, target, phase_dense),
        "fast_ot_error": gsError(source, target, phase_fast),
        "refined_error": gsError(source, target, refined),
    }


WORKFLOWS = {
    "phase-generation": phase_generation_methods_comparison,
    "beam-estimation": beam_estimation_performance,
    "pdot-alpha": pdot_vs_alpha,
    "paper-ot": paper_ot_initialization_comparison,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", choices=tuple(WORKFLOWS), default="phase-generation", nargs="?")
    arguments = parser.parse_args()
    print(json.dumps(WORKFLOWS[arguments.workflow](), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
