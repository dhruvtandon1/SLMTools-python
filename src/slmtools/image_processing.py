"""Image loading, filename parsing, orientation, resampling, and phase I/O."""

from __future__ import annotations

from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
from numbers import Number
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image

from .bmp8 import save_gray8bmp
from .dual_lattices import dualShiftLattice
from .lattice_field import (
    ComplexPhase,
    Intensity,
    LatticeField,
    RealPhase,
    _DecimalComplex,
    _as_decimal_array,
    _as_decimal_approx,
    _axis,
    _decimal_pi,
    _decimal_sincos,
    _julia_array_array_operation,
    _julia_array_scalar_operation,
    _julia_literal_array,
    _logical_axis_scalar_operation,
    _object_contains_decimal,
    _object_destination_element_type,
    _is_real_number,
    as_lattice,
)
from .lattice_utils import _step, natlat
from .misc import centroid
from .resampling import cubic_spline_interpolation

__all__ = [
    "getImagesAndFilenames",
    "imageToFloatArray",
    "itfa",
    "castImage",
    "loadDir",
    "parseFileName",
    "parseStringToNum",
    "getOrientation",
    "dualate",
    "linearFit",
    "savePhase",
    "saveBeam",
    "savePhase8BMP",
]


def _is_field(value: Any) -> bool:
    return isinstance(value, LatticeField) or (
        hasattr(value, "data") and hasattr(value, "L") and hasattr(value, "field_type")
    )


def _pil_array(image: Image.Image) -> np.ndarray:
    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    elif image.mode in {"CMYK", "YCbCr", "LAB", "HSV"}:
        image = image.convert("RGB")
    return np.asarray(image)


def _integer_unit_array(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.bool_:
        return array.astype(np.float64)
    info = np.iinfo(array.dtype)
    if info.min < 0:
        # Colorant channels in the Julia package are unsigned normalized fixed
        # point values.  A signed integer array has no unambiguous color scale.
        return array.astype(np.float64)
    return array.astype(np.float64) / float(info.max)


def imageToFloatArray(img: Any) -> np.ndarray:
    """Convert an image to ``float64`` grayscale using Rec. 601 luminance.

    For integer RGB/RGBA input, the grayscale value is requantized at the
    source bit depth before conversion to float, matching ``Gray.(img)`` for
    Julia normalized fixed-point colorants.  Alpha is intentionally ignored.
    """

    source = _pil_array(img) if isinstance(img, Image.Image) else np.asarray(img)
    if source.ndim == 2:
        if np.issubdtype(source.dtype, np.integer) or source.dtype == np.bool_:
            return _integer_unit_array(source)
        return source.astype(np.float64)

    if source.ndim != 3 or source.shape[-1] not in {2, 3, 4}:
        raise TypeError("image must be a 2-D grayscale or channel-last color array")
    if source.shape[-1] == 2:  # grayscale + alpha
        gray = source[..., 0]
        if np.issubdtype(gray.dtype, np.integer) or gray.dtype == np.bool_:
            return _integer_unit_array(gray)
        return gray.astype(np.float64)

    rgb = source[..., :3]
    integer_input = np.issubdtype(rgb.dtype, np.integer) or rgb.dtype == np.bool_
    normalized_integer = integer_input and (
        rgb.dtype == np.bool_ or np.iinfo(rgb.dtype).min >= 0
    )
    if normalized_integer and rgb.dtype != np.bool_:
        # Colors.jl performs its N0f8/N0f16 conversion on the raw fixed-point
        # channel integers.  In particular, its literal ``Tf(0.001)`` is a
        # Float32 for 8- and 16-bit channels; evaluating normalized Float64
        # channels first changes half-way rounding for thousands of RGB8
        # triples.
        raw = rgb.astype(np.uint64, copy=False)
        if rgb.dtype.itemsize < 4:
            raw_sum = 299 * raw[..., 0] + 587 * raw[..., 1] + 114 * raw[..., 2]
            weighted = np.float32(0.001) * raw_sum.astype(np.float32)
        elif rgb.dtype.itemsize < 8:
            weighted = np.float64(0.001) * (
                299 * raw[..., 0] + 587 * raw[..., 1] + 114 * raw[..., 2]
            )
        else:
            weighted = (
                np.float64(0.299) * raw[..., 0]
                + np.float64(0.587) * raw[..., 1]
                + np.float64(0.114) * raw[..., 2]
            )
        quantized = np.rint(weighted)
        return quantized.astype(np.float64) / float(np.iinfo(rgb.dtype).max)

    # The generic Colors.jl path spells these coefficients as Float32
    # literals.  NumPy then applies the same Float32/Float64 promotion dictated
    # by the source channel dtype before the final conversion to Float64.
    terms = tuple(
        _julia_array_scalar_operation(channel, coefficient, np.multiply)
        for channel, coefficient in zip(
            (rgb[..., 0], rgb[..., 1], rgb[..., 2]),
            (np.float32(0.299), np.float32(0.587), np.float32(0.114)),
        )
    )
    gray = terms[0] + terms[1] + terms[2]
    return np.asarray(gray, dtype=np.float64)


# Original short alias.
itfa = imageToFloatArray


def castImage(
    field_type: type, img: Any, L: Sequence[Sequence[float]], flambda: float
) -> LatticeField:
    """Cast a color image to grayscale and attach lattice metadata."""

    return LatticeField(itfa(img), L, flambda, field_type=field_type)


def getImagesAndFilenames(
    directory: str, extension: str
) -> tuple[list[Image.Image], list[str]]:
    """Load sorted exact-suffix files using Julia's concatenation contract.

    The audited source concatenates ``directory * filename`` instead of
    joining paths.  That only becomes observable when at least one filename
    matches: an empty broadcast performs no path loads and succeeds even
    without a trailing separator.
    """

    if not isinstance(directory, str):
        raise TypeError("directory must be a string, matching Julia dispatch")
    if not isinstance(extension, str):
        raise TypeError("extension must be a string, matching Julia dispatch")
    root = Path(directory)
    filenames = sorted(
        name.name
        for name in root.iterdir()
        if len(name.name) > len(extension) and name.name.endswith(extension)
    )
    images: list[Image.Image] = []
    for filename in filenames:
        # Do not replace this with Path joining.  ``dir .* fileNames`` in the
        # Julia authority is literal string concatenation, including its
        # failure when a matching file exists and ``dir`` lacks a separator.
        with Image.open(directory + filename) as image:
            image.load()
            images.append(image.copy())
    return images, filenames


def loadDir(
    directory: str,
    extension: str,
    *,
    T: type = Intensity,
    outType: Any = np.float64,
    L: Sequence[Sequence[float]] | float | tuple[float, float] | None = None,
    flambda: float = 1.0,
    cue: str | None = None,
    look: str = "after",
) -> tuple[list[LatticeField], list[Any]]:
    """Load same-sized images as lattice fields and parse their filenames."""

    images, filenames = getImagesAndFilenames(directory, extension)
    if not images:
        raise ValueError(f"no files ending in {extension!r} found in {str(directory)!r}")
    arrays = [itfa(image) for image in images]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("Inconsistent image size.")

    if L is None:
        lattice = tuple(_axis(range(1, size + 1)) for size in shape)
    elif np.isscalar(L):
        lattice = tuple(
            _logical_axis_scalar_operation(
                _axis(range(1, size + 1)),
                L,
                np.multiply,
            )
            for size in shape
        )
    elif (
        isinstance(L, tuple)
        and len(L) == 2
        and all(np.isscalar(spacing) for spacing in L)
    ):
        lattice = tuple(
            _logical_axis_scalar_operation(
                _axis(range(1, size + 1)),
                spacing,
                np.multiply,
            )
            for size, spacing in zip(shape, L)
        )
    else:
        lattice = as_lattice(L)

    fields = [LatticeField(array, lattice, flambda, field_type=T) for array in arrays]
    if cue is None:
        params = [parseFileName(filename, outType=outType) for filename in filenames]
    else:
        params = [
            parseFileName(filename, cue, look, outType=outType) for filename in filenames
        ]
    return fields, params


def parseStringToNum(string: str, *, outType: Any | None = None) -> Any:
    """Parse Julia's filename number syntax (comma denotes the decimal point)."""

    if outType is None:
        if "," in string:
            return float(string.replace(",", "."))
        return int(string)
    clean = string.replace(",", ".")
    if outType is Decimal:
        return Decimal(clean)
    if outType is Fraction:
        if any(marker in clean for marker in (".", "e", "E")):
            raise ValueError(f"invalid Rational literal: {string!r}")
        return Fraction(clean.replace("//", "/"))
    try:
        dtype = np.dtype(outType)
        converter = dtype.type
    except TypeError:
        converter = outType
        dtype = None
    if dtype is not None and dtype.kind == "b":
        bool_text = clean.strip()
        if bool_text == "true" or clean == "1":
            return converter(True)
        if bool_text == "false" or clean == "0":
            return converter(False)
        raise ValueError(f"invalid Bool literal: {string!r}")
    if outType is bool:
        bool_text = clean.strip()
        if bool_text == "true" or clean == "1":
            return True
        if bool_text == "false" or clean == "0":
            return False
        raise ValueError(f"invalid Bool literal: {string!r}")
    return converter(clean)


def parseFileName(
    name: str,
    cue: str | None = None,
    look: str = "after",
    *,
    outType: Any | None = None,
) -> Any:
    """Extract a number from a filename using the original cue convention."""

    if cue is None:
        dot = name.find(".")
        if dot < 0:
            raise ValueError("filename has no extension separator")
        return parseStringToNum(name[:dot], outType=outType)

    direction = str(look).lstrip(":")
    if direction not in {"before", "b", "after", "a"}:
        raise ValueError("Unrecognized look value.")
    direction = "b" if direction in {"before", "b"} else "a"
    extension_dot = name.rfind(".")
    if extension_dot < 0:
        raise ValueError("filename has no extension separator")
    searchable = "." + name[:extension_dot] + "."
    cue_at = searchable.rfind(str(cue))
    if cue_at < 0:
        raise ValueError(f"cue {cue!r} not found in filename {name!r}")
    allowed = frozenset(" 0123456789,-")

    if direction == "a":
        start = cue_at + len(str(cue))
        stop = start
        while stop < len(searchable) and searchable[stop] in allowed:
            stop += 1
        number = searchable[start:stop]
    else:
        stop = cue_at
        start = stop - 1
        while start >= 0 and searchable[start] in allowed:
            start -= 1
        number = searchable[start + 1 : stop]
    return parseStringToNum(number, outType=outType)


def linearFit(xs: Sequence[Any], ys: Sequence[Any]) -> tuple[Any, Any]:
    """Return ``(slope, intercept)`` from an ordinary least-squares line fit."""

    def vector(value: Any, name: str) -> np.ndarray:
        # Julia declares this method for concrete Vector inputs, not arbitrary
        # iterables/AbstractVectors.  A Python list is the native spelling of
        # a freshly allocated Vector literal; tuples and ranges retain their
        # distinct container semantics and therefore do not dispatch.
        if isinstance(value, list):
            result = _julia_literal_array(value)
        elif isinstance(value, np.ndarray):
            result = np.asarray(value)
        else:
            raise TypeError(f"linearFit {name} must be a vector")
        if result.ndim != 1:
            raise TypeError("linearFit expects two one-dimensional vectors")
        if result.dtype.kind not in "buifc":
            if result.dtype.kind != "O" or not all(
                isinstance(item, (Number, Decimal, np.number))
                for item in result.flat
            ):
                raise TypeError("linearFit vectors must contain numbers")
        return result

    x = vector(xs, "xs")
    y = vector(ys, "ys")
    if len(x) != len(y):
        raise ValueError("linearFit coordinate vectors must have equal length")

    if x.dtype.kind == "O" or y.dtype.kind == "O":
        values = (*x.flat, *y.flat)
        if any(isinstance(value, Decimal) for value in values):
            # Julia promotes a BigFloat operand, the Float64 ones column, and
            # the right-hand side to BigFloat before solving. Decimal is the
            # port's context-sized BigFloat counterpart. Julia's backslash
            # polyalgorithm uses LU for a square design and a column-pivoted
            # generic QR otherwise, so retain that solver split rather than
            # reducing every case to the normal equations.
            def as_decimal(value: Any) -> Decimal:
                if isinstance(value, Decimal):
                    return value
                if isinstance(value, Fraction):
                    return Decimal(value.numerator) / Decimal(value.denominator)
                if isinstance(value, (int, np.integer)):
                    return Decimal(int(value))
                if isinstance(value, (float, np.floating)):
                    return Decimal.from_float(float(value))
                raise TypeError("Decimal linearFit data must be real numbers")

            xd = [as_decimal(value) for value in x]
            yd = [as_decimal(value) for value in y]
            return _decimal_linear_fit(xd, yd)

        if all(isinstance(value, (Fraction, int, np.integer)) for value in values):
            # ``hcat(xs, ones(...))`` promotes Julia Rational input to
            # Float64 because the intercept column is Float64.
            x = x.astype(np.float64)
            y = y.astype(np.float64)

    design = np.column_stack((x, np.ones(len(x))))
    # Julia's `A \ b` chooses LU when the two-column design is square. In
    # particular, two identical x coordinates raise SingularException rather
    # than returning the minimum-norm solution produced by `lstsq`. Rectangular
    # designs use pivoted QR and retain their minimum-norm behavior.
    if design.shape[0] == design.shape[1]:
        slope, intercept = np.linalg.solve(design, y)
    else:
        slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return slope, intercept


def _decimal_reflector_inplace(
    matrix: list[list[Decimal]], column: int, row: int
) -> Decimal:
    """Apply Julia's generic `reflector!` storage convention to one column."""

    norm = sum(
        (matrix[index][column] * matrix[index][column]
         for index in range(row, len(matrix))),
        Decimal(0),
    ).sqrt()
    first = matrix[row][column]
    if norm == 0:
        return Decimal(0)
    signed_norm = norm.copy_sign(first)
    leading = first + signed_norm
    matrix[row][column] = -signed_norm
    for index in range(row + 1, len(matrix)):
        matrix[index][column] /= leading
    return leading / signed_norm


def _decimal_linear_fit(
    xs: list[Decimal], ys: list[Decimal]
) -> tuple[Decimal, Decimal]:
    """Two-column counterpart of Julia's BigFloat left-division algorithm."""

    rows = len(xs)
    if rows == 2:
        determinant = xs[0] - xs[1]
        if determinant == 0:
            raise np.linalg.LinAlgError(
                "linearFit design matrix is singular at pivot 2"
            )
        # Solve [x₁ 1; x₂ 1] * [slope; intercept] by the same square-system
        # contract as Julia's LU branch.
        slope = (ys[0] - ys[1]) / determinant
        intercept = ys[0] - xs[0] * slope
        return slope, intercept

    if rows == 0:
        return Decimal(0), Decimal(0)
    if rows == 1:
        # Julia's wide pivoted-QR solve returns the minimum-norm solution.
        denominator = xs[0] * xs[0] + Decimal(1)
        return xs[0] * ys[0] / denominator, ys[0] / denominator

    # Julia 1.12's generic (non-BLAS) column-pivoted QR deliberately uses the
    # unblocked reflector path and then triangular substitution. Reproducing
    # that order matters for rank-deficient BigFloat-like inputs: replacing it
    # with normal equations either raises too eagerly or invents a different
    # minimum-norm result.
    matrix = [[value, Decimal(1)] for value in xs]
    permutation = [0, 1]
    taus: list[Decimal] = []
    for column in range(2):
        norms = []
        for candidate in range(column, 2):
            norms.append(
                sum(
                    (
                        matrix[row][candidate] * matrix[row][candidate]
                        for row in range(column, rows)
                    ),
                    Decimal(0),
                ).sqrt()
            )
        pivot = column + (1 if len(norms) == 2 and norms[1] > norms[0] else 0)
        if pivot != column:
            permutation[pivot], permutation[column] = (
                permutation[column],
                permutation[pivot],
            )
            for row in range(rows):
                matrix[row][pivot], matrix[row][column] = (
                    matrix[row][column],
                    matrix[row][pivot],
                )
        tau = _decimal_reflector_inplace(matrix, column, column)
        taus.append(tau)
        for target_column in range(column + 1, 2):
            projection = matrix[column][target_column]
            for row in range(column + 1, rows):
                projection += (
                    matrix[row][column] * matrix[row][target_column]
                )
            projection *= tau
            matrix[column][target_column] -= projection
            for row in range(column + 1, rows):
                matrix[row][target_column] -= (
                    projection * matrix[row][column]
                )

    transformed = list(ys)
    for column, tau in enumerate(taus):
        projection = transformed[column]
        for row in range(column + 1, rows):
            projection += matrix[row][column] * transformed[row]
        projection *= tau
        transformed[column] -= projection
        for row in range(column + 1, rows):
            transformed[row] -= projection * matrix[row][column]

    if matrix[1][1] == 0:
        raise np.linalg.LinAlgError(
            "linearFit design matrix is singular at pivot 2"
        )
    second = transformed[1] / matrix[1][1]
    first = (
        transformed[0] - matrix[0][1] * second
    ) / matrix[0][0]
    solution = [first, second]
    inverse = [0, 0]
    for index, original in enumerate(permutation):
        inverse[original] = index
    return solution[inverse[0]], solution[inverse[1]]


def _decimal_atan(value: Decimal) -> Decimal:
    """Evaluate atan at the active Decimal precision."""

    precision = getcontext().prec
    with localcontext() as work:
        work.prec = precision + 10
        x = +value
        negative = x < 0
        if negative:
            x = -x
        pi = _decimal_pi()
        if x > 1:
            result = pi / 2 - _decimal_atan(1 / x)
        else:
            threshold = Decimal(2).sqrt() - 1
            offset = Decimal(0)
            if x > threshold:
                offset = pi / 4
                x = (x - 1) / (x + 1)
            power = x
            result = x
            denominator = 3
            tolerance = Decimal(10) ** -(precision + 5)
            while True:
                power *= -(x * x)
                addend = power / denominator
                updated = result + addend
                if updated == result or abs(addend) < tolerance:
                    result = updated
                    break
                result = updated
                denominator += 2
            result += offset
        if negative:
            result = -result
    return +result


def _decimal_atan2(y: Any, x: Any) -> Decimal:
    decimal_y = _as_decimal_approx(y)
    decimal_x = _as_decimal_approx(x)
    pi = _decimal_pi()
    if decimal_x > 0:
        return _decimal_atan(decimal_y / decimal_x)
    if decimal_x < 0:
        angle = _decimal_atan(decimal_y / decimal_x)
        return angle + pi if decimal_y >= 0 else angle - pi
    if decimal_y > 0:
        return pi / 2
    if decimal_y < 0:
        return -pi / 2
    return Decimal(0)


def getOrientation(
    linImgs: Sequence[LatticeField],
    idxs: Sequence[Any],
    *,
    roi: Any | None = None,
    threshold: float = 0.1,
) -> tuple[np.ndarray, Any]:
    """Fit centroid motion and return its index-zero intercept and angle."""

    if not isinstance(linImgs, list):
        raise TypeError("linImgs must be a Julia-style Vector of fields")
    if not isinstance(idxs, (list, np.ndarray)):
        raise TypeError("idxs must be a Julia-style numeric Vector")
    images = list(linImgs)
    if any(
        not isinstance(image, LatticeField)
        or image.field_type is not Intensity
        or np.ndim(image.data) != 2
        for image in images
    ):
        raise TypeError("linImgs must contain two-dimensional Intensity fields")
    if images and any(
        np.asarray(image.data).dtype != np.asarray(images[0].data).dtype
        for image in images[1:]
    ):
        raise TypeError("linImgs must have one concrete element type")
    if roi is not None:
        images = [image[roi] for image in images]
    center_rows = [np.asarray(centroid(image, threshold)) for image in images]
    if not center_rows:
        # Preserve vcat's empty-input failure instead of fabricating a shape.
        raise ValueError("getOrientation requires at least one image")
    center_width = len(center_rows[0])
    if any(len(row) != center_width for row in center_rows):
        raise ValueError("centroid dimensionalities must match")
    # Julia's vcat promotes all row element types before allocation. NumPy
    # instead chooses object for Rational/Float mixtures, which later bypasses
    # LAPACK dispatch. Treat this as an array literal over the scalar values,
    # then restore the row matrix.
    centers = _julia_literal_array(
        [value for row in center_rows for value in row]
    ).reshape((len(center_rows), center_width))
    xs, xi = linearFit(idxs, centers[:, 0])
    ys, yi = linearFit(idxs, centers[:, 1])
    if isinstance(xs, Decimal) or isinstance(ys, Decimal):
        theta = _decimal_atan2(ys, xs)
        return np.asarray([xi, yi], dtype=object), theta
    theta = np.angle(xs + 1j * ys)
    return np.asarray([xi, yi]), float(theta)


def _julia_zero_for_field_data(data: Any) -> Any:
    """Return ``zero(T)`` for the concrete element type represented by *data*."""

    array = np.asarray(data)
    object_type = _object_destination_element_type(array)
    if object_type is Fraction:
        return Fraction(0)
    if object_type is Decimal:
        return Decimal(0)
    if object_type is _DecimalComplex:
        return _DecimalComplex(Decimal(0), Decimal(0))
    return np.zeros((), dtype=array.dtype)[()]


def _julia_sincos(theta: Any) -> tuple[Any, Any]:
    """Evaluate Julia's scalar trigonometric promotion used by ``dualate``."""

    if isinstance(theta, Decimal):
        return _decimal_sincos(theta)
    if isinstance(theta, (Fraction, int, np.integer, bool, np.bool_)):
        # Julia promotes Integer/Rational trigonometry to Float64. NumPy
        # instead selects Float16 for Int8, Float32 for Int16/Int32, and
        # object ufuncs that look for nonexistent Fraction methods.
        theta = float(theta)
    return np.sin(theta), np.cos(theta)


def dualate(
    f: LatticeField | Sequence[LatticeField],
    L: Sequence[Sequence[float]],
    center: Sequence[float],
    theta: float,
    flambda: float = 1.0,
    *,
    roi: Any | None = None,
    interpolation: Callable[..., Any] = cubic_spline_interpolation,
    naturalize: bool = False,
    bc: Any | None = None,
) -> LatticeField | list[LatticeField]:
    """Offset, rotate, and interpolate a field onto ``dualShiftLattice(L)``."""

    if isinstance(f, tuple):
        raise TypeError("dualate's collection overload requires a Julia-style Vector")
    if isinstance(f, list):
        if not f or not all(isinstance(field, LatticeField) for field in f):
            raise TypeError("dualate requires a nonempty Vector of lattice fields")
        field_type = f[0].field_type
        if any(field.field_type is not field_type for field in f):
            raise TypeError("dualate requires homogeneous field tags")
        return [
            dualate(
                field,
                L,
                center,
                theta,
                flambda,
                roi=roi,
                interpolation=interpolation,
                naturalize=naturalize,
                bc=bc,
            )
            for field in f
        ]
    if not isinstance(f, LatticeField):
        raise TypeError("dualate requires a lattice field")
    if isinstance(center, tuple) or not isinstance(center, (list, np.ndarray)):
        raise TypeError("center must be a Julia-style real Vector")
    center_array = np.asarray(center)
    if center_array.ndim != 1 or not all(
        _is_real_number(value) for value in center_array.flat
    ):
        raise TypeError("center must be a Julia-style real Vector")
    if not _is_real_number(theta):
        raise TypeError("theta must be real")
    if not _is_real_number(flambda):
        raise TypeError("flambda must be real")
    if len(center) != len(f.L):
        raise ValueError("Incompatible lengths for center and f.L.")
    if len(f.L) != 2:
        raise ValueError("dualate is defined for two-dimensional fields")
    field = f if roi is None else f[roi]
    boundary = (
        _julia_zero_for_field_data(field.data) if bc is None else bc
    )
    shifted_axes = []
    for axis in range(2):
        shifted_axes.append(
            _logical_axis_scalar_operation(
                field.L[axis],
                center[axis],
                np.subtract,
            )
        )
    shifted_lattice = tuple(shifted_axes)
    interp = interpolation(shifted_lattice, field.data, extrapolation_bc=boundary)
    dual_lattice = dualShiftLattice(L, flambda)

    sine, cosine = _julia_sincos(theta)
    if isinstance(theta, Decimal):
        rotation = np.asarray(
            [[cosine, -sine], [sine, cosine]], dtype=object
        )
    else:
        rotation = np.asarray(
            [[cosine, -sine], [sine, cosine]]
        )
    dual_origin = np.asarray([dual_lattice[0][0], dual_lattice[1][0]])
    if _object_contains_decimal(rotation) or _object_contains_decimal(dual_origin):
        origin = _as_decimal_array(rotation) @ _as_decimal_array(dual_origin)
    else:
        origin = rotation @ dual_origin
    dx = _julia_array_scalar_operation(
        rotation[:, 0], _step(dual_lattice[0]), np.multiply
    )
    dy = _julia_array_scalar_operation(
        rotation[:, 1], _step(dual_lattice[1]), np.multiply
    )
    ii, jj = np.meshgrid(
        np.arange(len(dual_lattice[0])),
        np.arange(len(dual_lattice[1])),
        indexing="ij",
    )
    x_from_i = _julia_array_scalar_operation(ii, dx[0], np.multiply)
    x_from_j = _julia_array_scalar_operation(jj, dy[0], np.multiply)
    xcoords = _julia_array_array_operation(x_from_i, x_from_j, np.add)
    xcoords = _julia_array_scalar_operation(xcoords, origin[0], np.add)
    y_from_i = _julia_array_scalar_operation(ii, dx[1], np.multiply)
    y_from_j = _julia_array_scalar_operation(jj, dy[1], np.multiply)
    ycoords = _julia_array_array_operation(y_from_i, y_from_j, np.add)
    ycoords = _julia_array_scalar_operation(ycoords, origin[1], np.add)
    vectorized = False
    try:
        candidate = np.asarray(interp(xcoords, ycoords))
        if candidate.shape == xcoords.shape:
            data = candidate
            vectorized = True
    except (TypeError, ValueError):
        pass
    if not vectorized:
        # A user-supplied Julia-style scalar interpolator need not be
        # vectorized.  Preserve that extension point without penalizing the
        # NumPy default. A scalar or otherwise wrong-shaped result from an
        # array call is not sufficient evidence of vectorization: Julia calls
        # the interpolator independently at every Cartesian index.
        indices = iter(np.ndindex(xcoords.shape))
        try:
            first_index = next(indices)
        except StopIteration:
            data = np.empty(
                xcoords.shape,
                dtype=np.result_type(field.data.dtype, boundary),
            )
        else:
            evaluated = [
                (
                    first_index,
                    interp(xcoords[first_index], ycoords[first_index]),
                )
            ]
            for index in indices:
                evaluated.append(
                    (index, interp(xcoords[index], ycoords[index]))
                )
            promoted = _julia_literal_array(
                [value for _index, value in evaluated]
            )
            data = np.empty(xcoords.shape, dtype=promoted.dtype)
            for (index, _value), converted in zip(
                evaluated, promoted, strict=True
            ):
                data[index] = converted

    if naturalize:
        return LatticeField(data, natlat(data.shape), 1.0, field_type=field.field_type)
    return LatticeField(data, dual_lattice, flambda, field_type=field.field_type)


def _unit_to_u8(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
        raise ValueError("phase image values must be finite and lie in [0, 1]")
    return np.rint(array * 255).astype(np.uint8)


def _save_unit_image(values: Any, name: str | PathLike[str]) -> None:
    Image.fromarray(_unit_to_u8(values), mode="L").save(name)


def _field_tag_name(field: Any) -> str:
    tag = field.field_type
    return tag if isinstance(tag, str) else getattr(tag, "__name__", type(tag).__name__)


def _phase_image(value: Any) -> np.ndarray:
    if _is_field(value):
        tag = _field_tag_name(value)
        if tag in {"RealPhase", "UPhase", "UnwrappedPhase"}:
            return np.mod(np.asarray(value.data), 1)
        if tag not in {"ComplexPhase", "S1Phase"}:
            raise TypeError("savePhase is implemented only for phase lattice fields")
        data = np.asarray(value.data)
        if data.dtype != np.complex128:
            raise TypeError(
                "ComplexPhase saving requires ComplexF64/complex128 field data"
            )
    else:
        data = np.asarray(value)
        if data.ndim != 2 or data.dtype != np.complex128:
            raise TypeError(
                "savePhase expects a 2-D complex128 array or phase lattice field"
            )
    return (np.angle(data) + np.pi) / (2 * np.pi)


def savePhase(value: Any, name: str | PathLike[str]) -> None:
    """Save complex or real phase using the original grayscale mapping."""

    _save_unit_image(_phase_image(value), name)


def savePhase8BMP(value: Any, name: str | PathLike[str]) -> None:
    """Save complex or real phase as the exact uncompressed 8-bit BMP variant."""

    save_gray8bmp(name, _phase_image(value))


def saveBeam(
    beam: Any,
    name: str,
    data: Iterable[str] = ("beamCsv", "angleCsv", "anglePng", "negativeAnglePng"),
    *,
    dir: str | PathLike[str] | None = None,
) -> None:
    """Expose the unusable upstream beam-saving entry point.

    The audited Julia body references unimported ``now`` and ``writedlm``,
    uses a Windows-only separator, and writes the positive phase for the
    negative-angle output. Recognized output requests therefore remain
    explicitly unsupported instead of defining new file semantics.
    """

    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not isinstance(beam, np.ndarray) or beam.ndim != 2 or beam.dtype != np.complex128:
        raise TypeError("beam must be a two-dimensional NumPy complex128 array")
    if isinstance(data, str):
        data = (data,)
    requested = {str(item).lstrip(":") for item in data}
    recognized = {
        "beamCsv",
        "angleCsv",
        "anglePng",
        "negativeAnglePng",
    }
    if requested & recognized:
        raise NotImplementedError(
            "saveBeam is unusable in the audited Julia source because its "
            "required I/O names are not imported"
        )
