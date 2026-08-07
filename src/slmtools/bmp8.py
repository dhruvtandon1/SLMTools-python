"""Writer for the uncompressed 8-bit greyscale BMP used by SLMTools.

The implementation deliberately writes the file format itself.  This preserves
the 1,078-byte header, greyscale palette, bottom-up scanline order, and zero
padding of ``save_gray8bmp`` in the Julia package.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from os import PathLike
from pathlib import Path
import struct
from typing import Any

import gmpy2
import numpy as np
from PIL import Image

from ._bigfloat import _MPFR, _MPQ, _MPZ, _bigfloat_context, _to_mpfr
from .lattice_field import _exact_real_to_machine_float

FILE_HDR = 14
DIB_HDR = 40
PALETTE = 256 * 4
HEADER_SZ = FILE_HDR + DIB_HDR + PALETTE

__all__ = ["save_gray8bmp"]


def _gray_array(img: Any) -> np.ndarray:
    """Return two-dimensional greyscale channels without narrowing them."""

    if isinstance(img, Image.Image):
        if img.mode not in {"1", "L", "I", "F", "I;16", "I;16L", "I;16B"}:
            raise TypeError("img must be a two-dimensional greyscale image")
        arr = np.asarray(img)
    else:
        arr = np.asarray(img)

    if arr.ndim != 2:
        raise TypeError("img must be a two-dimensional greyscale image")

    if arr.dtype == np.bool_:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if info.min < 0:
            # Julia's Gray{N0f8} conversion is defined on [0, 1], not on a
            # signed integer's full machine range.
            return arr
        return arr.astype(np.float64) / float(info.max)
    return arr


def _float_n0f8(values: np.ndarray) -> np.ndarray:
    """Quantize machine floats using FixedPointNumbers' dtype arithmetic."""

    working = (
        values.astype(np.float32)
        if values.dtype == np.dtype(np.float16)
        else values
    )
    if working.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("unsupported greyscale floating-point dtype")
    if (
        not np.all(np.isfinite(working))
        or np.any(working < 0)
        or np.any(working > 1)
    ):
        raise ValueError("greyscale values must be finite and lie in [0, 1]")
    scale = working.dtype.type(255)
    scaled = np.multiply(working, scale, dtype=working.dtype)
    return np.rint(scaled).astype(np.uint8)


def _object_n0f8(value: Any) -> int:
    """Quantize one type-erased channel through its Julia runtime method."""

    if isinstance(value, (_MPQ, Fraction)):
        rational = (
            Fraction(int(value.numerator), int(value.denominator))
            if isinstance(value, _MPQ)
            else value
        )
        if rational < 0 or rational > 1:
            raise ValueError(
                "greyscale values must be finite and lie in [0, 1]"
            )
        converted = _exact_real_to_machine_float(rational, np.float32)
        scaled = np.multiply(
            converted, np.float32(255), dtype=np.float32
        )
        return int(np.rint(scaled))

    if isinstance(value, (Decimal, _MPFR)):
        with _bigfloat_context():
            converted = _to_mpfr(value)
            scaled = gmpy2.rint(converted * gmpy2.mpfr(255))
            if (
                not gmpy2.is_finite(converted)
                or scaled < 0
                or scaled > 255
            ):
                raise ValueError(
                    "greyscale values must be finite and lie in [0, 1]"
                )
            return int(scaled)

    if isinstance(value, (_MPZ, int, np.integer, bool, np.bool_)):
        scaled = 255 * int(value)
        if scaled < 0 or scaled > 255:
            raise ValueError(
                "greyscale values must be finite and lie in [0, 1]"
            )
        return scaled

    if isinstance(value, (float, np.floating)):
        dtype = (
            np.float32
            if isinstance(value, np.float16)
            else np.asarray(value).dtype
        )
        array = np.asarray([[value]], dtype=dtype)
        return int(_float_n0f8(array)[0, 0])

    raise TypeError("greyscale channels must be real numeric values")


def _n0f8(img: Any) -> np.ndarray:
    """Convert like ``N0f8.(img)``: validate [0, 1], then round to 8 bits."""

    values = _gray_array(img)
    if values.dtype == np.dtype(object):
        output = np.empty(values.shape, dtype=np.uint8)
        for index in np.ndindex(values.shape):
            output[index] = _object_n0f8(values[index])
        return output
    if values.dtype == np.dtype(np.bool_):
        return values.astype(np.uint8) * np.uint8(255)
    if np.issubdtype(values.dtype, np.integer):
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError(
                "greyscale values must be finite and lie in [0, 1]"
            )
        return values.astype(np.uint8) * np.uint8(255)
    if np.issubdtype(values.dtype, np.floating):
        return _float_n0f8(values)
    raise TypeError("greyscale channels must be real numeric values")


def save_gray8bmp(path: str | PathLike[str], img: Any) -> None:
    """Write *img* as an uncompressed, paletted 8-bit greyscale BMP.

    Rows are stored bottom-up and padded with zero bytes to a multiple of four.
    The parent directory must already exist, matching ordinary file-save
    behavior in the original package.
    """

    pixels = _n0f8(img)
    height, width = pixels.shape

    rowpad = (-width) % 4
    rowsz = width + rowpad
    datasz = rowsz * height
    filesize = HEADER_SZ + datasz

    target = Path(path)
    with target.open("wb") as stream:
        stream.write(b"BM")
        stream.write(struct.pack("<III", filesize, 0, HEADER_SZ))
        stream.write(
            struct.pack(
                "<IiiHHIIIIII",
                DIB_HDR,
                width,
                height,
                1,
                8,
                0,
                datasz,
                0,
                0,
                0,
                0,
            )
        )
        palette = bytearray(PALETTE)
        for level in range(256):
            start = 4 * level
            palette[start : start + 4] = bytes((level, level, level, 0))
        stream.write(palette)

        padding = b"\x00" * rowpad
        for row in pixels[::-1]:
            stream.write(row.tobytes(order="C"))
            stream.write(padding)

