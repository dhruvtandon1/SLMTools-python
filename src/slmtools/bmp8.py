"""Writer for the uncompressed 8-bit greyscale BMP used by SLMTools.

The implementation deliberately writes the file format itself.  This preserves
the 1,078-byte header, greyscale palette, bottom-up scanline order, and zero
padding of ``save_gray8bmp`` in the Julia package.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
import struct
from typing import Any

import numpy as np
from PIL import Image

FILE_HDR = 14
DIB_HDR = 40
PALETTE = 256 * 4
HEADER_SZ = FILE_HDR + DIB_HDR + PALETTE

__all__ = ["save_gray8bmp"]


def _gray_float_array(img: Any) -> np.ndarray:
    """Return a two-dimensional greyscale array without silently clipping."""

    if isinstance(img, Image.Image):
        if img.mode not in {"1", "L", "I", "F", "I;16", "I;16L", "I;16B"}:
            raise TypeError("img must be a two-dimensional greyscale image")
        arr = np.asarray(img)
    else:
        arr = np.asarray(img)

    if arr.ndim != 2:
        raise TypeError("img must be a two-dimensional greyscale image")

    if arr.dtype == np.bool_:
        return arr.astype(np.float64)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if info.min < 0:
            # Julia's Gray{N0f8} conversion is defined on [0, 1], not on a
            # signed integer's full machine range.
            return arr.astype(np.float64)
        return arr.astype(np.float64) / float(info.max)
    return arr.astype(np.float64, copy=False)


def _n0f8(img: Any) -> np.ndarray:
    """Convert like ``N0f8.(img)``: validate [0, 1], then round to 8 bits."""

    values = _gray_float_array(img)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("greyscale values must be finite and lie in [0, 1]")
    # NumPy and Julia's relevant N0f8 conversion path both use nearest-even
    # rounding for exact half-way cases.
    return np.rint(values * 255.0).astype(np.uint8)


def save_gray8bmp(path: str | PathLike[str], img: Any) -> None:
    """Write *img* as an uncompressed, paletted 8-bit greyscale BMP.

    Rows are stored bottom-up and padded with zero bytes to a multiple of four.
    The parent directory must already exist, matching ordinary file-save
    behavior in the original package.
    """

    pixels = _n0f8(img)
    height, width = pixels.shape
    if width <= 0 or height <= 0:
        raise ValueError("BMP images must have non-zero width and height")

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

