"""Optional image-composition helpers ported from ``src/SubImages.jl``.

This module intentionally remains separate from the root :mod:`slmtools` API,
matching the Julia repository where ``SubImages.jl`` is included manually by
notebooks.  Arrays use image order ``(rows, columns[, channels])``.  Grids of
images are traversed in Fortran order so their layout and automatic labels
match Julia's column-major indexing.

The Julia ``padmultiple(..., padall=n)`` implementation accidentally reapplies
the four directional padding arguments and ignores the magnitude of ``n``.
That upstream behavior is preserved: a positive ``padall`` repeats each
positive directional pad once, while it does nothing if those four widths are
zero.
"""

from __future__ import annotations

from io import BytesIO
import string
import warnings
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .lattice_utils import _julia_assignment_values


def _subimage_int(value: object, name: str) -> int:
    """Match concrete ``Int`` annotations in the optional Julia helpers."""

    if isinstance(value, (bool, np.bool_)) or not (
        type(value) is int or isinstance(value, np.int64)
    ):
        raise TypeError(f"{name} must be a Julia Int value")
    return int(value)


def _subimage_pair(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-element Julia Int tuple")
    return (
        _subimage_int(value[0], name),
        _subimage_int(value[1], name),
    )


def ftaText(
    text: str,
    size: tuple[int, int],
    *,
    fnt: str = "arial bold",
    pixelsize: int | None = None,
    halign: str = "hcenter",
    valign: str = "vcenter",
    **options: object,
) -> np.ndarray:
    """Render text using the same helper as :func:`slmtools.ftaText`."""

    from .templates import ftaText as _fta_text

    return _fta_text(
        text,
        size,
        fnt=fnt,
        pixelsize=pixelsize,
        halign=halign,
        valign=valign,
        **options,
    )


def plotToImage(plot: object) -> np.ndarray:
    """Render a Matplotlib-compatible plot or figure to an RGBA image.

    Pillow exposes PNG channels as bytes, while Julia's ``FileIO.load``
    returns fixed-point colorants whose numeric values lie in ``[0, 1]``.
    Convert here so downstream helpers observe the same value convention.
    """

    figure = getattr(plot, "figure", plot)
    savefig = getattr(figure, "savefig", None)
    if savefig is None:
        raise TypeError("plotToImage expects an object with a savefig method")
    stream = BytesIO()
    savefig(stream, format="png")
    stream.seek(0)
    with Image.open(stream) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0


def imageToHeatmap(image: np.ndarray, **options: object) -> object:
    """Create and return a Matplotlib image artist for a grayscale image.

    Matplotlib is an optional dependency because ``SubImages.jl`` is likewise
    an optional notebook utility rather than part of the loaded Julia module.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "imageToHeatmap requires the optional 'matplotlib' dependency"
        ) from exc
    _, axis = plt.subplots()
    return axis.imshow(np.asarray(image, dtype=float), **options)


def grayAnnotation(
    text: str,
    pixelsize: int,
    padto: tuple[int, int] | None = None,
    **options: object,
) -> np.ndarray:
    """Render text, crop its empty border, and optionally center-pad it."""

    pixelsize = _subimage_int(pixelsize, "pixelsize")
    if padto is not None:
        padto = _subimage_pair(padto, "padto")

    rendered = ftaText(
        text,
        (round(pixelsize * 2), round(pixelsize * 2)),
        pixelsize=pixelsize,
        **options,
    )
    occupied_rows = np.flatnonzero(np.sum(rendered, axis=1) != 0)
    occupied_cols = np.flatnonzero(np.sum(rendered, axis=0) != 0)
    if not occupied_rows.size or not occupied_cols.size:
        cropped = rendered[:0, :0]
    else:
        cropped = rendered[
            occupied_rows[0] : occupied_rows[-1] + 1,
            occupied_cols[0] : occupied_cols[-1] + 1,
        ]
    if padto is None:
        return np.asarray(cropped, dtype=float)
    if any(target < actual for target, actual in zip(padto, cropped.shape)):
        raise ValueError("grayAnnotation: text exceeds target output size")
    return padout(cropped, padto)


def _is_color(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[-1] in (3, 4)


def _rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if _is_color(array):
        return array[..., :3]
    if array.ndim != 2:
        raise ValueError("images must be 2-D grayscale or HxWx3/4 color arrays")
    return np.repeat(array[..., None], 3, axis=-1)


def _object_grid(images: object) -> np.ndarray:
    """Return a 1-D/2-D object array without stacking differently-sized data."""

    if isinstance(images, np.ndarray) and images.dtype == object:
        return images.copy()
    if isinstance(images, np.ndarray) and images.ndim in (2, 3):
        result = np.empty(1, dtype=object)
        result[0] = images
        return result
    rows = list(images)  # type: ignore[arg-type]
    if not rows:
        return np.empty((0,), dtype=object)
    nested = isinstance(rows[0], (list, tuple))
    if nested:
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("image grid must be rectangular")
        result = np.empty((len(rows), width), dtype=object)
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                result[row_index, column_index] = np.asarray(value)
        return result
    result = np.empty(len(rows), dtype=object)
    for index, value in enumerate(rows):
        result[index] = np.asarray(value)
    return result


def colorPromote(images: object) -> np.ndarray:
    """Promote every image in a grid to RGB if any image is RGB."""

    grid = _object_grid(images)
    use_rgb = any(_is_color(np.asarray(item)) for item in grid.ravel(order="F"))
    if use_rgb:
        for index in np.ndindex(grid.shape):
            grid[index] = _rgb(np.asarray(grid[index]))
    return grid


def padout(
    image: np.ndarray,
    size: tuple[int, int],
    fillval: object = 0,
    *,
    shift: tuple[int, int] | None = None,
) -> np.ndarray:
    """Embed an image in the requested canvas, centered by default."""

    matrix = np.asarray(image)
    size = _subimage_pair(size, "padding size")
    if matrix.ndim not in (2, 3):
        raise ValueError("padout expects a grayscale or color image")
    if any(target < actual for target, actual in zip(size, matrix.shape[:2])):
        raise ValueError("padout: image is bigger than intended padded size")
    if shift is None:
        shift = tuple((target - actual) // 2 for target, actual in zip(size, matrix.shape[:2]))
    else:
        shift = _subimage_pair(shift, "padding shift")
    row, column = shift
    if row < 0 or column < 0 or row + matrix.shape[0] > size[0] or column + matrix.shape[1] > size[1]:
        raise ValueError("padout: shifted image falls outside output")
    shape = size + matrix.shape[2:]
    converted_fill = _julia_assignment_values(np.asarray(fillval), matrix.dtype)[()]
    output = np.full(shape, converted_fill, dtype=matrix.dtype)
    output[row : row + matrix.shape[0], column : column + matrix.shape[1], ...] = matrix
    return output


def padadd(image: np.ndarray, amount: int, side: str, fillval: object = 0) -> np.ndarray:
    """Add rows or columns on one side of an image."""

    matrix = np.asarray(image)
    amount = _subimage_int(amount, "padding amount")
    if amount < 0:
        raise ValueError("padding must be nonnegative")
    normalized = side.lower()
    if normalized in {"l", "left"}:
        pads = ((0, 0), (amount, 0))
    elif normalized in {"r", "right"}:
        pads = ((0, 0), (0, amount))
    elif normalized in {"t", "top"}:
        pads = ((amount, 0), (0, 0))
    elif normalized in {"b", "bottom"}:
        pads = ((0, amount), (0, 0))
    else:
        raise ValueError(f"unknown side: {side!r}")
    if matrix.ndim == 3:
        pads += ((0, 0),)
    converted_fill = _julia_assignment_values(np.asarray(fillval), matrix.dtype)[()]
    return np.pad(matrix, pads, constant_values=converted_fill)


def _pad_one(
    image: np.ndarray,
    *,
    padleft: int,
    padright: int,
    padtop: int,
    padbottom: int,
    padall: int,
    fillval: object,
) -> np.ndarray:
    values = tuple(
        _subimage_int(value, name)
        for value, name in zip(
            (padleft, padright, padtop, padbottom, padall),
            ("padleft", "padright", "padtop", "padbottom", "padall"),
            strict=True,
        )
    )
    padleft, padright, padtop, padbottom, padall = values
    # Julia guards every directional width with ``if width > 0``. Its
    # ``padall`` branch recursively reapplies those same four widths without
    # forwarding padall, so any positive padall doubles the directional pads
    # but its numeric magnitude is otherwise ignored.
    padleft, padright, padtop, padbottom = (
        max(value, 0) for value in values[:4]
    )
    repeat = 2 if padall > 0 else 1
    left, right = repeat * padleft, repeat * padright
    top, bottom = repeat * padtop, repeat * padbottom
    matrix = np.asarray(image)
    if left == right == top == bottom == 0:
        return matrix.copy()
    pads = ((top, bottom), (left, right))
    if matrix.ndim == 3:
        pads += ((0, 0),)
    converted_fill = _julia_assignment_values(np.asarray(fillval), matrix.dtype)[()]
    return np.pad(matrix, pads, constant_values=converted_fill)


def padmultiple(
    images: object,
    *,
    padleft: int = 0,
    padright: int = 0,
    padtop: int = 0,
    padbottom: int = 0,
    padall: int = 0,
    fillval: object = 0,
) -> object:
    """Pad one image or every image in a grid."""

    if isinstance(images, np.ndarray) and images.dtype != object and images.ndim in (2, 3):
        return _pad_one(
            images,
            padleft=padleft,
            padright=padright,
            padtop=padtop,
            padbottom=padbottom,
            padall=padall,
            fillval=fillval,
        )
    grid = _object_grid(images)
    for index in np.ndindex(grid.shape):
        grid[index] = _pad_one(
            np.asarray(grid[index]),
            padleft=padleft,
            padright=padright,
            padtop=padtop,
            padbottom=padbottom,
            padall=padall,
            fillval=fillval,
        )
    return grid


def padCommon(images: object, fillval: object | None = None) -> np.ndarray:
    """Pad a 2-D image grid to common row heights and column widths."""

    grid = colorPromote(images)
    if grid.ndim != 2 or not grid.size:
        raise ValueError("padCommon expects a nonempty 2-D image grid")
    if fillval is None:
        fillval = 0
    row_heights = [max(np.asarray(grid[row, col]).shape[0] for col in range(grid.shape[1])) for row in range(grid.shape[0])]
    col_widths = [max(np.asarray(grid[row, col]).shape[1] for row in range(grid.shape[0])) for col in range(grid.shape[1])]
    output = np.empty_like(grid)
    for row, col in np.ndindex(grid.shape):
        output[row, col] = padout(np.asarray(grid[row, col]), (row_heights[row], col_widths[col]), fillval)
    return output


def _gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if _is_color(array):
        rgb = array[..., :3].astype(float)
        # HxWx3/4 integer arrays are Python's channel representation of a
        # Julia Matrix{RGB/RGBA}; convert byte channels to colorant values.
        if np.issubdtype(array.dtype, np.integer):
            rgb /= np.iinfo(array.dtype).max
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        # Neutral colors convert exactly to their channel value in ColorTypes;
        # avoid floating coefficient roundoff turning RGB white into 1-eps.
        neutral = np.all(rgb == rgb[..., :1], axis=-1)
        gray = np.where(neutral, rgb[..., 0], gray)
    else:
        if np.issubdtype(array.dtype, np.integer) and np.any(
            (array < 0) | (array > 1)
        ):
            raise ValueError(
                "integer grayscale values must be 0 or 1 for Julia Gray conversion"
            )
        gray = array.astype(float)
    return gray


def trimWhitespace(image: np.ndarray) -> np.ndarray:
    """Crop rows and columns whose pixels are all white (value one)."""

    matrix = np.asarray(image)
    gray = _gray(matrix)
    white = gray == 1
    rows = np.flatnonzero(~np.all(white, axis=1))
    cols = np.flatnonzero(~np.all(white, axis=0))
    if not rows.size or not cols.size:
        raise ValueError("trimWhitespace cannot crop an entirely white image")
    return matrix[
        rows[0] : rows[-1] + 1,
        cols[0] : cols[-1] + 1,
        ...,
    ].copy()


def arrange(layout: tuple[int, int], *images: np.ndarray) -> np.ndarray:
    """Arrange images using Julia/Fortran linear ordering."""

    if len(images) != int(np.prod(layout)):
        raise ValueError("number of images does not match layout")
    output = np.empty(layout, dtype=object)
    for index, image in enumerate(images):
        output[np.unravel_index(index, layout, order="F")] = np.asarray(image)
    return output


def checkCommonSize(images: object) -> bool:
    """Return whether images can be concatenated without common padding."""

    grid = _object_grid(images)
    if not grid.size:
        return True
    if grid.ndim == 1:
        width = np.asarray(grid[0]).shape[1]
        return all(np.asarray(item).shape[1] == width for item in grid)
    if grid.ndim != 2:
        return False
    row_ok = all(
        all(np.asarray(grid[row, col]).shape[0] == np.asarray(grid[row, 0]).shape[0] for col in range(grid.shape[1]))
        for row in range(grid.shape[0])
    )
    col_ok = all(
        all(np.asarray(grid[row, col]).shape[1] == np.asarray(grid[0, col]).shape[1] for row in range(grid.shape[0]))
        for col in range(grid.shape[1])
    )
    return row_ok and col_ok


def _merge(grid: np.ndarray) -> np.ndarray:
    if grid.ndim == 1:
        return np.concatenate([np.asarray(item) for item in grid], axis=0)
    rows = [np.concatenate([np.asarray(grid[row, col]) for col in range(grid.shape[1])], axis=1) for row in range(grid.shape[0])]
    return np.concatenate(rows, axis=0)


def _coerce_merge_input(first: object, images: tuple[np.ndarray, ...]) -> np.ndarray:
    if isinstance(first, tuple) and len(first) == 2 and all(isinstance(v, int) for v in first):
        return arrange(first, *images)
    if images:
        raise TypeError("extra positional images require a (rows, columns) layout")
    return _object_grid(first)


def mergeStrict(
    images_or_layout: object,
    *images: np.ndarray,
    padleft: int = 0,
    padright: int = 0,
    padtop: int = 0,
    padbottom: int = 0,
    padall: int = 0,
    fillval: object = 0,
) -> np.ndarray:
    """Merge a compatible image grid, rejecting inconsistent cell sizes."""

    grid = _coerce_merge_input(images_or_layout, images)
    if not checkCommonSize(grid):
        raise ValueError("mergeStrict: images have incompatible sizes")
    grid = colorPromote(grid)
    grid = padmultiple(
        grid,
        padleft=padleft,
        padright=padright,
        padtop=padtop,
        padbottom=padbottom,
        padall=padall,
        fillval=fillval,
    )
    return _merge(grid)


def mergeFill(
    images_or_layout: object,
    *images: np.ndarray,
    padleft: int = 0,
    padright: int = 0,
    padtop: int = 0,
    padbottom: int = 0,
    padall: int = 0,
    fillval: object = 0,
) -> np.ndarray:
    """Pad an image grid to compatible cell sizes, then merge it."""

    grid = colorPromote(_coerce_merge_input(images_or_layout, images))
    if not checkCommonSize(grid):
        grid = padCommon(grid, fillval)
    grid = padmultiple(
        grid,
        padleft=padleft,
        padright=padright,
        padtop=padtop,
        padbottom=padbottom,
        padall=padall,
        fillval=fillval,
    )
    return _merge(grid)


def _annotation_for_dtype(annotation: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        maximum = np.iinfo(dtype).max
        return np.rint(annotation * maximum).astype(dtype)
    return annotation.astype(dtype)


def handAnnotate(
    images: object,
    labels: str | Sequence[str],
    pixelsize: int,
    padto: tuple[int, int] | None = None,
    *,
    location: str = "topleft",
    shift: tuple[int, int] = (0, 0),
    inverta: bool = True,
    **options: object,
) -> object:
    """Place one text label, or a Fortran-ordered sequence of labels, on images."""

    pixelsize = _subimage_int(pixelsize, "pixelsize")
    if padto is not None:
        padto = _subimage_pair(padto, "padto")
    shift = _subimage_pair(shift, "annotation shift")

    if isinstance(images, np.ndarray) and images.dtype != object and images.ndim in (2, 3):
        if not isinstance(labels, str):
            raise TypeError("a single image requires a string label")
        output = np.asarray(images).copy()
        annotation = grayAnnotation(labels, pixelsize, padto, **options)
        if inverta:
            annotation = 1 - annotation
        if _is_color(output):
            annotation = _rgb(annotation)
        annotation = _annotation_for_dtype(annotation, output.dtype)
        if location.lower().lstrip(":") != "topleft":
            warnings.warn("This location is not implemented; no change made", RuntimeWarning, stacklevel=2)
            return output
        row, column = shift
        if row < 0 or column < 0 or row + annotation.shape[0] > output.shape[0] or column + annotation.shape[1] > output.shape[1]:
            raise ValueError("annotation does not fit inside image")
        output[row : row + annotation.shape[0], column : column + annotation.shape[1], ...] = annotation
        return output

    if isinstance(labels, str):
        raise TypeError("an image grid requires one label per image")
    grid = _object_grid(images)
    labels_tuple = tuple(labels)
    if len(labels_tuple) != grid.size:
        raise ValueError("number of labels does not match number of images")
    output = grid.copy()
    for linear_index, index in enumerate(np.ndindex(tuple(reversed(grid.shape)))):
        actual_index = tuple(reversed(index))
        output[actual_index] = handAnnotate(
            np.asarray(grid[actual_index]),
            labels_tuple[linear_index],
            pixelsize,
            padto,
            location=location,
            shift=shift,
            inverta=inverta,
            **options,
        )
    return output


def autoAnnotate(
    images: object,
    pixelsize: int,
    padto: tuple[int, int] | None = None,
    *,
    labelOffset: int = 0,
    location: str = "topleft",
    shift: tuple[int, int] = (0, 0),
    **options: object,
) -> object:
    """Annotate images ``(a)`` … ``(z)``, then ``(A)`` … ``(Z)``."""

    pixelsize = _subimage_int(pixelsize, "pixelsize")
    if padto is not None:
        padto = _subimage_pair(padto, "padto")
    labelOffset = _subimage_int(labelOffset, "labelOffset")
    shift = _subimage_pair(shift, "annotation shift")
    grid = _object_grid(images)
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    stop = labelOffset + grid.size
    if labelOffset < 0 or stop > len(alphabet):
        raise ValueError("automatic labels support at most 52 images")
    labels = tuple(f"({letter})" for letter in alphabet[labelOffset:stop])
    return handAnnotate(
        grid,
        labels,
        pixelsize,
        padto,
        location=location,
        shift=shift,
        **options,
    )


__all__ = [
    "ftaText",
    "plotToImage",
    "imageToHeatmap",
    "grayAnnotation",
    "colorPromote",
    "padout",
    "padadd",
    "padmultiple",
    "padCommon",
    "trimWhitespace",
    "arrange",
    "checkCommonSize",
    "mergeStrict",
    "mergeFill",
    "autoAnnotate",
    "handAnnotate",
]
