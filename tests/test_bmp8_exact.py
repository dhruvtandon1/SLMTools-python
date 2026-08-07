from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import struct

import gmpy2
import numpy as np
import pytest

from slmtools._bigfloat import _bigfloat_context
from slmtools.bmp8 import HEADER_SZ, save_gray8bmp
from slmtools.image_processing import savePhase8BMP
from slmtools.lattice_field import LF, RealPhase


def _pixel_payload(path: Path) -> bytes:
    return path.read_bytes()[HEADER_SZ:]


def test_gray8bmp_machine_float_quantization_uses_source_dtype(
    tmp_path: Path,
) -> None:
    probe = np.float32("0.0019607844296842813")
    cases = (
        ("f16", np.asarray([[probe]], dtype=np.float16), b"\x00"),
        ("f32", np.asarray([[probe]], dtype=np.float32), b"\x00"),
        ("f64", np.asarray([[probe]], dtype=np.float64), b"\x01"),
    )
    for name, values, expected in cases:
        path = tmp_path / f"{name}.bmp"
        save_gray8bmp(path, values)
        assert _pixel_payload(path) == expected + b"\x00\x00\x00"


def test_gray8bmp_object_channels_follow_julia_runtime_conversion(
    tmp_path: Path,
) -> None:
    rational_delta = Fraction(1, 2**200)
    rational_values = np.asarray(
        [
            [
                Fraction(1, 2) - rational_delta,
                Fraction(1, 2),
                Fraction(1, 2) + rational_delta,
            ]
        ],
        dtype=object,
    )
    mpq_values = np.asarray(
        [
            [
                gmpy2.mpq(value.numerator, value.denominator)
                for value in rational_values[0]
            ]
        ],
        dtype=object,
    )

    with _bigfloat_context():
        mpfr_half = gmpy2.mpfr(1) / 2
        mpfr_delta = gmpy2.exp2(-200)
        mpfr_values = np.asarray(
            [[mpfr_half - mpfr_delta, mpfr_half, mpfr_half + mpfr_delta]],
            dtype=object,
        )

    with localcontext() as context:
        context.prec = 260
        decimal_half = Decimal(1) / 2
        decimal_delta = Decimal(2) ** -200
        decimal_values = np.asarray(
            [
                [
                    decimal_half - decimal_delta,
                    decimal_half,
                    decimal_half + decimal_delta,
                ]
            ],
            dtype=object,
        )

    for name, values, expected in (
        ("fraction", rational_values, bytes((128, 128, 128))),
        ("mpq", mpq_values, bytes((128, 128, 128))),
        ("mpfr", mpfr_values, bytes((127, 128, 128))),
        ("decimal", decimal_values, bytes((127, 128, 128))),
    ):
        path = tmp_path / f"{name}.bmp"
        save_gray8bmp(path, values)
        assert _pixel_payload(path) == expected + b"\x00"

    integer_path = tmp_path / "bigint.bmp"
    save_gray8bmp(
        integer_path,
        np.asarray([[gmpy2.mpz(0), gmpy2.mpz(1)]], dtype=object),
    )
    assert _pixel_payload(integer_path) == bytes((0, 255, 0, 0))


def test_save_phase8bmp_preserves_decimal_bigfloat_precision(
    tmp_path: Path,
) -> None:
    with localcontext() as context:
        context.prec = 260
        half = Decimal(1) / 2
        delta = Decimal(2) ** -200
        values = np.asarray(
            [[half - delta, half, half + delta]], dtype=object
        )

    field = LF[RealPhase, object, 2](
        values, (range(1), range(3))
    )
    path = tmp_path / "phase-exact.bmp"
    savePhase8BMP(field, path)
    assert _pixel_payload(path) == bytes((127, 128, 128, 0))


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (2, 0)])
def test_gray8bmp_writes_header_only_for_zero_sized_images(
    tmp_path: Path, shape: tuple[int, int]
) -> None:
    path = tmp_path / f"empty-{shape[0]}-{shape[1]}.bmp"
    assert save_gray8bmp(path, np.empty(shape, dtype=np.float64)) is None
    raw = path.read_bytes()
    assert len(raw) == HEADER_SZ
    assert struct.unpack_from("<I", raw, 2)[0] == HEADER_SZ
    assert struct.unpack_from("<ii", raw, 18) == (shape[1], shape[0])
    assert struct.unpack_from("<I", raw, 34)[0] == 0


def test_save_phase8bmp_inherits_zero_sized_bmp_output(tmp_path: Path) -> None:
    path = tmp_path / "empty-phase.bmp"
    savePhase8BMP(np.empty((0, 2), dtype=np.complex128), path)
    raw = path.read_bytes()
    assert len(raw) == HEADER_SZ
    assert struct.unpack_from("<ii", raw, 18) == (2, 0)
