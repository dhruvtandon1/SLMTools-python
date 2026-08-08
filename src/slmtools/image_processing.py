"""Image loading, filename parsing, orientation, resampling, and phase I/O."""

from __future__ import annotations

from decimal import (
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    getcontext,
    localcontext,
)
from fractions import Fraction
from math import ceil, isqrt, log2
from numbers import Number
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import gmpy2
import numpy as np
from PIL import Image
from scipy import linalg as scipy_linalg

from ._bigfloat import (
    _MPC,
    _MPFR,
    _MPFRComplex,
    _MPQ,
    _MPZ,
    _bigfloat_context,
    _is_mpfr,
    _is_mpc,
    _mpfr_object_operation,
    _to_mpfr,
)
from ._omission import _OMITTED
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
    _julia_collect_comprehension_results,
    _julia_literal_array,
    _julia_typed_zero,
    _logical_axis_scalar_operation,
    _object_contains_decimal,
    _object_contains_mpfr,
    _object_numeric_element_key,
    _object_destination_element_type,
    _is_real_number,
    _require_julia_numeric_array,
    _require_dense_ndarray,
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


def _is_load_dir_lattice(value: Any) -> bool:
    """Recognize Python spellings of Julia's two-dimensional ``L`` union."""

    if value is None or _is_real_number(value):
        return True
    if not isinstance(value, tuple):
        return False
    if len(value) != 2:
        return False
    components = value
    if all(
        _is_real_number(component) for component in components
    ):
        return True
    # Julia's Lattice{2} is a tuple of ranges. Python coordinate axes may be
    # concrete one-dimensional sequences (including regular NumPy arrays and
    # LatticeAxis instances), which ``as_lattice`` validates below.
    for component in components:
        if isinstance(component, (str, bytes)):
            return False
        try:
            if np.asarray(component).ndim != 1:
                return False
        except (TypeError, ValueError):
            return False
    return True


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

    # Julia's typed keyword wrapper rejects these values before entering the
    # function body (and therefore before touching the filesystem).  Keep the
    # two intentional ``nothing`` cases, ``L`` and ``cue``, separate below.
    if not isinstance(T, type):
        raise TypeError("loadDir T must be a data type")
    if not isinstance(outType, (type, np.dtype)):
        raise TypeError("loadDir outType must be a data type")
    if not _is_load_dir_lattice(L):
        raise TypeError(
            "loadDir L must be a two-dimensional lattice or real spacing"
        )
    if not _is_real_number(flambda):
        raise TypeError("loadDir flambda must be real")
    if cue is not None and not isinstance(cue, str):
        raise TypeError("loadDir cue must be a string, character, or None")
    if not isinstance(look, str):
        raise TypeError("loadDir look must be a string or character")

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


def _julia_integer_literal(string: str) -> int:
    """Parse Base-style decimal/binary/octal/hex integer spellings."""

    stripped = string.strip()
    unsigned = stripped[1:] if stripped[:1] in "+-" else stripped
    base = 0 if unsigned.lower().startswith(("0x", "0o", "0b")) else 10
    return int(stripped, base)


def _julia_int64_literal(string: str) -> int:
    """Parse the concrete platform ``Int`` used by the audited Julia build."""

    value = _julia_integer_literal(string)
    limits = np.iinfo(np.int64)
    if value < limits.min or value > limits.max:
        raise OverflowError(f"overflow parsing {string!r} as Julia Int64")
    return value


def _julia_rational_literal(string: str, *, bigint: bool) -> Any:
    """Parse Base's integer-only ``Rational{T}`` string grammar."""

    components = string.split("/", 1)
    parser = _julia_integer_literal if bigint else _julia_int64_literal
    try:
        numerator = parser(components[0])
        if len(components) == 1:
            denominator = 1
        else:
            denominator_text = components[1]
            if denominator_text.startswith("/"):
                denominator_text = denominator_text[1:]
            denominator = parser(denominator_text)
    except ValueError as error:
        raise ValueError(f"invalid Rational literal: {string!r}") from error

    if bigint:
        return gmpy2.mpq(numerator, denominator)
    return Fraction(numerator, denominator)


def _julia_bigfloat_complex_literal(string: str) -> gmpy2.mpc:
    """Parse Julia complex spelling with each component parsed as BigFloat."""

    def component(text: str) -> gmpy2.mpfr:
        stripped = text.strip()
        if any(character.isspace() for character in stripped):
            raise ValueError(f"invalid BigFloat literal: {text!r}")
        return gmpy2.mpfr(stripped)

    compact = string.strip()
    unit_length = (
        2
        if compact.endswith("im")
        else (1 if compact.endswith(("i", "j")) else 0)
    )
    body = compact[:-unit_length] if unit_length else compact
    separator = None
    for index, character in enumerate(body[1:], start=1):
        if character in "+-" and body[index - 1] not in "eE":
            separator = index
            break
    if unit_length == 0:
        if separator is not None:
            raise ValueError("missing imaginary unit")
        return gmpy2.mpc(component(body), gmpy2.mpfr(0))
    if separator is None:
        return gmpy2.mpc(gmpy2.mpfr(0), component(body))
    imaginary = component(body[separator + 1 :])
    if body[separator] == "-":
        imaginary = -imaginary
    return gmpy2.mpc(
        component(body[:separator]),
        imaginary,
    )


def _julia_hex_float_literal(string: str) -> float | None:
    stripped = string.strip()
    unsigned = stripped[1:] if stripped[:1] in "+-" else stripped
    return float.fromhex(stripped) if unsigned.lower().startswith("0x") else None


def parseStringToNum(string: str, *, outType: Any | None = None) -> Any:
    """Parse Julia's filename number syntax (comma denotes the decimal point)."""

    clean = string.replace(",", ".")
    if "_" in clean:
        raise ValueError("Julia numeric parsing does not accept underscores")
    if outType is None:
        if "," in string:
            hexadecimal = _julia_hex_float_literal(clean)
            return float(clean) if hexadecimal is None else hexadecimal
        return _julia_int64_literal(clean)
    if outType is Decimal:
        return Decimal(clean)
    if outType is Fraction:
        return _julia_rational_literal(clean, bigint=False)
    if outType is gmpy2.mpz:
        return gmpy2.mpz(clean)
    if outType is gmpy2.mpq:
        return _julia_rational_literal(clean, bigint=True)
    if outType is gmpy2.mpfr:
        with _bigfloat_context():
            return gmpy2.mpfr(clean)
    if outType is gmpy2.mpc:
        with _bigfloat_context():
            return _julia_bigfloat_complex_literal(clean)
    try:
        dtype = np.dtype(outType)
        converter = dtype.type
    except TypeError as error:
        raise TypeError("outType must support Julia numeric parsing") from error
    if dtype.kind not in "buifc":
        raise TypeError("outType must support Julia numeric parsing")
    if dtype.kind == "b":
        bool_text = clean.strip()
        if bool_text == "true" or clean == "1":
            return converter(True)
        if bool_text == "false" or clean == "0":
            return converter(False)
        raise ValueError(f"invalid Bool literal: {string!r}")
    if dtype.kind in "iu":
        return converter(_julia_integer_literal(clean))
    if dtype.kind == "f":
        hexadecimal = _julia_hex_float_literal(clean)
        return converter(clean if hexadecimal is None else hexadecimal)
    if dtype.kind == "c":
        hexadecimal = _julia_hex_float_literal(clean)
        if hexadecimal is not None:
            return converter(hexadecimal)
        return converter(clean.replace(" ", "").replace("im", "j"))
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
    cue: Any = _OMITTED,
    look: Any = _OMITTED,
    /,
    *,
    outType: Any | None = None,
) -> Any:
    """Extract a number from a filename using the original cue convention."""

    if not isinstance(name, str):
        raise TypeError("parseFileName name must be a string")
    if cue is _OMITTED:
        if look is not _OMITTED:
            raise TypeError(
                "parseFileName's cue-less overload does not accept look"
            )
        dot = name.find(".")
        if dot < 0:
            raise ValueError("filename has no extension separator")
        return parseStringToNum(name[:dot], outType=outType)
    if not isinstance(cue, str):
        raise TypeError("parseFileName cue must be a string or character")
    if look is _OMITTED:
        look = "after"
    elif not isinstance(look, str):
        raise TypeError("parseFileName look must be a string or character")

    direction = look.lstrip(":")
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


def _is_complex_bigfloat_scalar(value: Any) -> bool:
    return isinstance(value, _MPFRComplex) or _is_mpc(value)


def _is_bigfloat_linear_scalar(value: Any) -> bool:
    return (
        isinstance(value, Decimal)
        or _is_mpfr(value)
        or isinstance(value, (_MPQ, _MPZ))
        or (
            type(value) is int
            and not np.iinfo(np.int64).min <= value <= np.iinfo(np.int64).max
        )
        or (
            isinstance(value, Fraction)
            and (
                not np.iinfo(np.int64).min
                <= value.numerator
                <= np.iinfo(np.int64).max
                or not np.iinfo(np.int64).min
                <= value.denominator
                <= np.iinfo(np.int64).max
            )
        )
        or _is_complex_bigfloat_scalar(value)
    )


def _as_mpfr_complex(value: Any) -> _MPFRComplex:
    if isinstance(value, _MPFRComplex):
        return _MPFRComplex(value.real, value.imag)
    if _is_mpc(value):
        return _MPFRComplex(value.real, value.imag)
    if isinstance(value, (complex, np.complexfloating)):
        scalar = complex(value)
        return _MPFRComplex(scalar.real, scalar.imag)
    return _MPFRComplex(value)


def _mpfr_complex_abs2(value: _MPFRComplex) -> Any:
    with _bigfloat_context():
        return value.real * value.real + value.imag * value.imag


def _mpfr_complex_is_finite(value: _MPFRComplex) -> bool:
    return bool(
        gmpy2.is_finite(value.real) and gmpy2.is_finite(value.imag)
    )


def _mpfr_complex_norm(values: Sequence[_MPFRComplex]) -> Any:
    """Mirror Julia's finite generic two-norm for Complex{BigFloat}."""

    maximum = max(abs(value) for value in values)
    if maximum == 0 or gmpy2.is_infinite(maximum):
        return maximum
    with _bigfloat_context():
        scale_check = len(values) * maximum * maximum
        if gmpy2.is_finite(scale_check) and scale_check != 0:
            total = _mpfr_complex_abs2(values[0])
            for value in values[1:]:
                total += _mpfr_complex_abs2(value)
            return gmpy2.sqrt(total)

        total = _mpfr_complex_abs2(values[0] / maximum)
        for value in values[1:]:
            total += _mpfr_complex_abs2(value / maximum)
        return maximum * gmpy2.sqrt(total)


def _mpfr_complex_dot(
    left: Sequence[_MPFRComplex],
    right: Sequence[_MPFRComplex],
) -> _MPFRComplex:
    total = _MPFRComplex(0)
    for left_value, right_value in zip(left, right, strict=True):
        total = total + left_value.conjugate() * right_value
    return total


def _mpfr_complex_reflector_values_inplace(
    values: list[_MPFRComplex],
) -> _MPFRComplex:
    """Apply Julia's generic complex Householder storage convention."""

    norm = _mpfr_complex_norm(values)
    first = values[0]
    if norm == 0:
        return _MPFRComplex(0)
    with _bigfloat_context():
        signed_norm = gmpy2.copy_sign(norm, first.real)
    nu = _MPFRComplex(signed_norm)
    leading = first + nu
    values[0] = -nu
    for index in range(1, len(values)):
        values[index] = values[index] / leading
    return leading / nu


def _mpfr_complex_reflector_inplace(
    matrix: list[list[_MPFRComplex]],
    column: int,
    row: int,
) -> _MPFRComplex:
    values = [
        matrix[index][column] for index in range(row, len(matrix))
    ]
    tau = _mpfr_complex_reflector_values_inplace(values)
    for index, value in enumerate(values, start=row):
        matrix[index][column] = value
    return tau


def _mpfr_complex_qr_solve(
    matrix: list[list[_MPFRComplex]],
    taus: Sequence[_MPFRComplex],
    permutation: Sequence[int],
    ys: list[_MPFRComplex],
) -> tuple[_MPFRComplex, _MPFRComplex]:
    """Solve with Julia-layout QR factors already rounded to their source type."""

    rows = len(matrix)
    zero = _MPFRComplex(0)
    transformed = list(ys)
    if rows < 2:
        transformed.append(zero)
    for column, tau in enumerate(taus):
        projection = transformed[column] + _mpfr_complex_dot(
            [
                matrix[row][column]
                for row in range(column + 1, rows)
            ],
            [
                transformed[row]
                for row in range(column + 1, rows)
            ],
        )
        projection = tau.conjugate() * projection
        transformed[column] = transformed[column] - projection
        for row in range(column + 1, rows):
            transformed[row] = (
                transformed[row] - matrix[row][column] * projection
            )

    if rows == 1:
        wide_row = [matrix[0][0], matrix[0][1]]
        wide_tau = _mpfr_complex_reflector_values_inplace(wide_row)
        transformed[0] = transformed[0] / wide_row[0]
        transformed[1] = zero
        projection = wide_tau * transformed[0]
        transformed[0] = transformed[0] - projection
        transformed[1] = (
            transformed[1]
            - wide_row[1].conjugate() * projection
        )
        solution = transformed
    else:
        if matrix[1][1] == zero:
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


def _mpfr_complex_linear_fit(
    xs: list[_MPFRComplex],
    ys: list[_MPFRComplex],
    *,
    complex_domain: bool = True,
) -> tuple[_MPFRComplex, _MPFRComplex]:
    """Two-column Julia QR/LU solve for Complex{BigFloat} regression."""

    rows = len(xs)
    zero = _MPFRComplex(0)
    one = _MPFRComplex(1)
    if rows == 0:
        return zero, zero

    if rows == 2:
        # Julia recognizes this special upper-triangular design before LU.
        if xs[1] == zero:
            if xs[0] == zero:
                raise np.linalg.LinAlgError(
                    "linearFit design matrix is singular at pivot 1"
                )
            second = ys[1]
            if complex_domain and second.imag == 0:
                second = _MPFRComplex(second.real, gmpy2.mpfr("-0"))
            first = (ys[0] - second) / xs[0]
            return first, second

        # Julia checks a non-triangular matrix before generic LU. Preserve the
        # successful upper-triangular Inf/NaN path above, but reject non-finite
        # designs that actually enter LU.
        if not all(_mpfr_complex_is_finite(value) for value in xs):
            raise ValueError(
                "linearFit design matrix contains Infs or NaNs"
            )

        matrix = [[xs[0], one], [xs[1], one]]
        rhs = list(ys)
        if abs(matrix[1][0]) > abs(matrix[0][0]):
            matrix[0], matrix[1] = matrix[1], matrix[0]
            rhs[0], rhs[1] = rhs[1], rhs[0]
        if matrix[0][0] == zero:
            raise np.linalg.LinAlgError(
                "linearFit design matrix is singular at pivot 1"
            )
        lower = matrix[1][0] * (one / matrix[0][0])
        matrix[1][0] = lower
        matrix[1][1] = matrix[1][1] - lower * matrix[0][1]
        if matrix[1][1] == zero:
            raise np.linalg.LinAlgError(
                "linearFit design matrix is singular at pivot 2"
            )
        if lower == zero:
            # UnitLowerTriangular dynamically follows its structurally-upper
            # solve when this multiplier is exactly zero. This avoids
            # evaluating ``0 * Inf`` into the second component. For complex
            # arithmetic the same path still turns a non-finite leading RHS
            # into NaN+NaN*im and gives the finite component a negative-zero
            # imaginary part.
            if complex_domain and not _mpfr_complex_is_finite(rhs[0]):
                second = rhs[1] / matrix[1][1]
                if second.imag == 0:
                    second = _MPFRComplex(
                        second.real,
                        gmpy2.mpfr("-0"),
                    )
                return (
                    _MPFRComplex(
                        gmpy2.mpfr("nan"),
                        -gmpy2.mpfr("nan"),
                    ),
                    second,
                )
        else:
            rhs[1] = rhs[1] - lower * rhs[0]
        second = rhs[1] / matrix[1][1]
        first = (rhs[0] - matrix[0][1] * second) / matrix[0][0]
        return first, second

    matrix = [[value, one] for value in xs]
    permutation = [0, 1]
    taus: list[_MPFRComplex] = []
    for column in range(min(rows, 2)):
        norms = [
            _mpfr_complex_norm(
                [
                    matrix[row][candidate]
                    for row in range(column, rows)
                ]
            )
            for candidate in range(column, 2)
        ]
        pivot = (
            column + 1
            if len(norms) == 2 and norms[1] > norms[0]
            else column
        )
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

        tau = _mpfr_complex_reflector_inplace(
            matrix, column, column
        )
        taus.append(tau)
        for target_column in range(column + 1, 2):
            projection = matrix[column][target_column] + _mpfr_complex_dot(
                [
                    matrix[row][column]
                    for row in range(column + 1, rows)
                ],
                [
                    matrix[row][target_column]
                    for row in range(column + 1, rows)
                ],
            )
            projection = tau.conjugate() * projection
            matrix[column][target_column] = (
                matrix[column][target_column] - projection
            )
            for row in range(column + 1, rows):
                matrix[row][target_column] = (
                    matrix[row][target_column]
                    - projection * matrix[row][column]
                )

    return _mpfr_complex_qr_solve(matrix, taus, permutation, ys)


def _machine_factor_then_mpfr_linear_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    complex_result: bool,
) -> tuple[Any, Any]:
    """Factor a machine design before promoting its factors to BigFloat.

    Julia chooses the QR/LU factorization from the left-hand matrix before a
    higher-precision right-hand side determines the factorization solve type.
    Thus a ComplexF64 matrix with a BigFloat right-hand side first rounds
    LAPACK's factors in ComplexF64, then converts those stored factors to
    Complex{BigFloat}. Refactoring the original matrix in MPFR changes the
    result.
    """

    if xs.dtype.kind == "O":
        machine_xs = xs.astype(np.float64)
    else:
        machine_xs = xs
    design = np.column_stack((machine_xs, np.ones(len(machine_xs))))
    rhs = [_as_mpfr_complex(value) for value in ys]
    rows = len(machine_xs)
    zero = _MPFRComplex(0)
    if rows == 0:
        result = (zero, zero)
    elif rows == 2:
        # The matrix polyalgorithm recognizes this upper-triangular shape
        # before factorization and promotes its entries only for the solve.
        if design[1, 0] == 0:
            result = _mpfr_complex_linear_fit(
                [_as_mpfr_complex(value) for value in design[:, 0]],
                rhs,
                complex_domain=complex_result,
            )
        else:
            if not np.all(np.isfinite(design)):
                raise ValueError(
                    "linearFit design matrix contains Infs or NaNs"
                )
            getrf = scipy_linalg.get_lapack_funcs("getrf", (design,))
            factors, pivots, info = getrf(
                np.array(design, copy=True),
                overwrite_a=True,
            )
            if info < 0:
                raise ValueError(
                    f"LAPACK getrf received an invalid argument {-info}"
                )
            if info > 0:
                raise np.linalg.LinAlgError(
                    f"linearFit design matrix is singular at pivot {info}"
                )
            matrix = [
                [_as_mpfr_complex(value) for value in row]
                for row in factors
            ]
            transformed = list(rhs)
            for row, pivot in enumerate(pivots):
                if row != pivot:
                    transformed[row], transformed[pivot] = (
                        transformed[pivot],
                        transformed[row],
                    )
            if matrix[0][0] == zero or matrix[1][1] == zero:
                raise np.linalg.LinAlgError(
                    "linearFit design matrix is singular"
                )
            transformed[1] = (
                transformed[1] - matrix[1][0] * transformed[0]
            )
            intercept = transformed[1] / matrix[1][1]
            slope = (
                transformed[0] - matrix[0][1] * intercept
            ) / matrix[0][0]
            result = (slope, intercept)
    else:
        raw_factors, _r, permutation = scipy_linalg.qr(
            design,
            mode="raw",
            pivoting=True,
            check_finite=False,
        )
        factors, taus = raw_factors
        matrix = [
            [_as_mpfr_complex(value) for value in row]
            for row in factors
        ]
        result = _mpfr_complex_qr_solve(
            matrix,
            [_as_mpfr_complex(value) for value in taus],
            [int(value) for value in permutation],
            rhs,
        )

    if complex_result:
        return result
    return result[0].real, result[1].real


def _machine_nonfinite_qr_factors(
    design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build Julia-compatible raw pivoted-QR storage for non-finite inputs."""

    rows = len(design)
    if np.all(np.isfinite(design)):
        raw_factors, _r, permutation = scipy_linalg.qr(
            design,
            mode="raw",
            pivoting=True,
            check_finite=False,
        )
        factors, taus = raw_factors
        return factors, taus, [int(value) for value in permutation]

    # The only non-finite design column is the user-supplied x vector; the
    # intercept column is Float64 one. LAPACK keeps that column order.
    factors = np.array(design, copy=True)
    taus = np.empty(min(rows, 2), dtype=design.dtype)
    permutation = [0, 1]
    if rows == 1:
        taus[0] = 0
        return factors, taus, permutation

    complex_dtype = design.dtype.kind == "c"
    x_column = design[:, 0]
    if np.any(np.isnan(x_column.real)) or (
        complex_dtype and np.any(np.isnan(x_column.imag))
    ):
        nan = (
            complex(float("nan"), float("nan"))
            if complex_dtype
            else float("nan")
        )
        factors[:, :] = nan
        taus[:] = nan
        return factors, taus, permutation

    # Julia's LAPACK reflector for an infinite norm stores a signed infinite
    # diagonal and zero tail. Applying its NaN tau then affects only the
    # leading row; BLAS skips the exact-zero reflector components.
    signed_norm = np.copysign(float("inf"), float(x_column[0].real))
    factors[0, 0] = (
        complex(-signed_norm, 0.0)
        if complex_dtype
        else -signed_norm
    )
    factors[1:, 0] = 0
    factors[0, 1] = (
        complex(float("nan"), float("nan"))
        if complex_dtype
        else float("nan")
    )
    taus[0] = (
        complex(float("nan"), 0.0)
        if complex_dtype
        else float("nan")
    )

    raw_second, _r = scipy_linalg.qr(
        factors[1:, 1:2],
        mode="raw",
        pivoting=False,
        check_finite=False,
    )
    second_factors, second_tau = raw_second
    factors[1:, 1] = second_factors[:, 0]
    taus[1] = second_tau[0]
    return factors, taus, permutation


def _machine_nonfinite_qr_solve(
    factors: np.ndarray,
    taus: np.ndarray,
    permutation: Sequence[int],
    ys: np.ndarray,
) -> tuple[Any, Any]:
    """Apply Julia's two-column QRPivoted ldiv operation order."""

    rows = len(factors)
    result_dtype = np.result_type(factors.dtype, ys.dtype)
    transformed = np.zeros(max(rows, 2), dtype=result_dtype)
    transformed[:rows] = ys

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        for column, tau in enumerate(taus):
            if tau == 0:
                continue
            projection = transformed[column]
            for row in range(column + 1, rows):
                projection = (
                    projection
                    + np.conjugate(factors[row, column])
                    * transformed[row]
                )
            projection = np.conjugate(tau) * projection
            transformed[column] = transformed[column] - projection
            for row in range(column + 1, rows):
                reflector = factors[row, column]
                if reflector != 0:
                    transformed[row] = (
                        transformed[row] - reflector * projection
                    )

        if rows == 1:
            wide = np.asarray(
                [factors[0, 0], factors[0, 1]],
                dtype=factors.dtype,
            )
            norm = scipy_linalg.norm(wide, check_finite=False)
            signed_norm = np.copysign(
                norm,
                float(wide[0].real),
            )
            nu = np.asarray(
                complex(signed_norm, 0.0)
                if factors.dtype.kind == "c"
                else signed_norm,
                dtype=factors.dtype,
            )[()]
            leading = wide[0] + nu
            wide[0] = -nu
            wide[1] = wide[1] / leading
            wide_tau = leading / nu

            transformed[0] = transformed[0] / wide[0]
            transformed[1] = 0
            projection = wide_tau * transformed[0]
            transformed[0] = transformed[0] - projection
            transformed[1] = (
                transformed[1] - np.conjugate(wide[1]) * projection
            )
            solution = transformed[:2]
        else:
            if factors[1, 1] == 0:
                nan = (
                    complex(float("nan"), float("nan"))
                    if result_dtype.kind == "c"
                    else float("nan")
                )
                solution = np.asarray([nan, nan], dtype=result_dtype)
            else:
                second = transformed[1] / factors[1, 1]
                first = (
                    transformed[0] - factors[0, 1] * second
                ) / factors[0, 0]
                solution = np.asarray([first, second], dtype=result_dtype)

    inverse = [0, 0]
    for index, original in enumerate(permutation):
        inverse[original] = index
    ordered = [solution[inverse[0]], solution[inverse[1]]]
    if result_dtype.kind == "c":
        for index, value in enumerate(ordered):
            if value.imag == 0:
                ordered[index] = np.asarray(
                    complex(float(value.real), 0.0),
                    dtype=result_dtype,
                )[()]
    return ordered[0], ordered[1]


def _machine_nonfinite_linear_fit(
    design: np.ndarray,
    ys: np.ndarray,
) -> tuple[Any, Any]:
    factors, taus, permutation = _machine_nonfinite_qr_factors(design)
    return _machine_nonfinite_qr_solve(
        factors,
        taus,
        permutation,
        ys,
    )


def linearFit(xs: Sequence[Any], ys: Sequence[Any]) -> tuple[Any, Any]:
    """Return ``(slope, intercept)`` from an ordinary least-squares line fit."""

    def vector(value: Any, name: str) -> np.ndarray:
        # Julia declares this method for concrete Vector inputs, not arbitrary
        # iterables/AbstractVectors.  A Python list is the native spelling of
        # a freshly allocated Vector literal; tuples and ranges retain their
        # distinct container semantics and therefore do not dispatch.
        if isinstance(value, list):
            # Julia spells ``[]`` as ``Vector{Any}``, which does not satisfy
            # ``Vector{<:Number}``.  An explicitly typed empty NumPy vector
            # remains valid (for example, ``np.empty(0, dtype=np.float64)``).
            if not value:
                raise TypeError(
                    f"linearFit {name} empty list has no concrete numeric "
                    "element type"
                )
            if any(_is_bigfloat_linear_scalar(item) for item in value):
                # NumPy has no native BigFloat or Complex{BigFloat} dtype.
                # Preserve the concrete object vector until Julia-compatible
                # promotion below.
                result = np.empty(len(value), dtype=object)
                for index, item in enumerate(value):
                    result[index] = item
            else:
                result = _julia_literal_array(value)
        elif isinstance(value, np.ndarray):
            result = _require_dense_ndarray(value, f"linearFit {name}")
            # A deliberately object-typed ndarray of ordinary machine
            # numbers is the Python counterpart of ``Vector{Any}``, which
            # does not satisfy Julia's ``Vector{<:Number}`` method. Object
            # storage remains necessary for the concrete numeric domains
            # NumPy lacks: Rational and BigFloat are represented by Fraction
            # and Decimal, respectively.
            if result.dtype.kind == "O" and not all(
                isinstance(item, (Number, Decimal, np.number, _MPQ, _MPZ))
                or isinstance(item, _MPFRComplex)
                for item in result.flat
            ):
                raise TypeError(
                    f"linearFit {name} object arrays must represent a "
                    "Julia numeric vector"
                )
        else:
            raise TypeError(f"linearFit {name} must be a vector")
        if result.ndim != 1:
            raise TypeError("linearFit expects two one-dimensional vectors")
        if result.dtype.kind not in "buifc":
            if result.dtype.kind != "O" or not all(
                isinstance(
                    item,
                    (Number, Decimal, np.number, _MPQ, _MPZ),
                )
                or isinstance(item, _MPFRComplex)
                for item in result.flat
            ):
                raise TypeError("linearFit vectors must contain numbers")
        return result

    x = vector(xs, "xs")
    y = vector(ys, "ys")
    if len(x) != len(y):
        raise ValueError("linearFit coordinate vectors must have equal length")

    # Julia also permits vectors whose declared element type is an abstract
    # numeric supertype such as Real or Number. Generic QR promotes their
    # runtime scalar values as it works. NumPy object arrays carry no such
    # declaration, so normalize their numeric values through Julia literal
    # promotion before selecting the equivalent concrete solver.
    if x.dtype.kind == "O" and _object_numeric_element_key(x) is None:
        x = _julia_literal_array(list(x))
    if y.dtype.kind == "O" and _object_numeric_element_key(y) is None:
        y = _julia_literal_array(list(y))

    if x.dtype.kind == "O" or y.dtype.kind == "O":
        values = (*x.flat, *y.flat)
        x_has_bigfloat = any(
            _is_bigfloat_linear_scalar(value) for value in x.flat
        )
        y_has_bigfloat = any(
            _is_bigfloat_linear_scalar(value) for value in y.flat
        )
        has_decimal = any(isinstance(value, Decimal) for value in values)
        has_mpfr = any(_is_mpfr(value) for value in values)
        has_big_exact = any(
            isinstance(value, (_MPQ, _MPZ))
            or (
                type(value) is int
                and not np.iinfo(np.int64).min
                <= value
                <= np.iinfo(np.int64).max
            )
            or (
                isinstance(value, Fraction)
                and (
                    not np.iinfo(np.int64).min
                    <= value.numerator
                    <= np.iinfo(np.int64).max
                    or not np.iinfo(np.int64).min
                    <= value.denominator
                    <= np.iinfo(np.int64).max
                )
            )
            for value in values
        )
        has_mpfr_complex = any(
            _is_complex_bigfloat_scalar(value) for value in values
        )
        has_machine_complex = any(
            isinstance(value, (complex, np.complexfloating))
            for value in values
        )
        if y_has_bigfloat and not x_has_bigfloat:
            with _bigfloat_context():
                return _machine_factor_then_mpfr_linear_fit(
                    x,
                    y,
                    complex_result=(
                        x.dtype.kind == "c"
                        or any(
                            _is_complex_bigfloat_scalar(value)
                            or isinstance(
                                value, (complex, np.complexfloating)
                            )
                            for value in y.flat
                        )
                    ),
                )

        if has_mpfr_complex or (
            (has_decimal or has_mpfr or has_big_exact)
            and has_machine_complex
        ):
            xc = [_as_mpfr_complex(value) for value in x]
            yc = [_as_mpfr_complex(value) for value in y]
            with _bigfloat_context():
                return _mpfr_complex_linear_fit(xc, yc)

        if has_mpfr or has_big_exact:
            # The real BigFloat and Complex{BigFloat} generic solvers have
            # identical operation order when every imaginary component is
            # zero. Reusing that MPFR implementation preserves Julia's exact
            # 256-bit result while returning the real scalar element type.
            xc = [_as_mpfr_complex(value) for value in x]
            yc = [_as_mpfr_complex(value) for value in y]
            with _bigfloat_context():
                slope, intercept = _mpfr_complex_linear_fit(
                    xc,
                    yc,
                    complex_domain=False,
                )
            return slope.real, intercept.real

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

        # ``hcat(xs, ones(...))`` promotes a Rational design column to
        # Float64. Backslash then promotes a Rational right-hand side to the
        # real or complex machine element type of that design independently.
        # Treat the vectors separately so Rational/ComplexF64 mixes do not
        # leave an object array for LAPACK.
        if all(
            isinstance(value, (Fraction, int, np.integer))
            for value in x.flat
        ):
            x = x.astype(np.float64)
        if all(
            isinstance(value, (Fraction, int, np.integer))
            for value in y.flat
        ):
            y = y.astype(np.float64)

    design = np.column_stack((x, np.ones(len(x))))
    # Julia's `A \ b` chooses LU when the two-column design is square. In
    # particular, two identical x coordinates raise SingularException rather
    # than returning a minimum-norm solution. Rectangular StridedMatrix
    # designs go through LAPACK's pivoted-QR least-squares driver (gelsy).
    # NumPy's SVD-based lstsq uses a different rank threshold and collapses
    # Julia's large, finite result for nearly dependent designs.
    if design.shape[0] == design.shape[1]:
        if design[1, 0] == 0:
            if design[0, 0] == 0:
                raise np.linalg.LinAlgError(
                    "linearFit design matrix is singular at pivot 1"
                )
            if (
                design.dtype.kind == "c"
                and not np.isfinite(design[0, 0])
            ):
                # Accelerate's complex ``trtrs`` maps a finite numerator
                # divided by ``Inf + 0im`` to ``NaN + NaN*im``. Julia's
                # structurally upper-triangular solve performs the scalar
                # division instead, producing a complex zero. Keep this
                # narrow non-finite branch outside LAPACK so the result does
                # not depend on the SciPy/BLAS backend.
                result_dtype = np.result_type(design.dtype, y.dtype)
                intercept = np.asarray(y[1], dtype=result_dtype)[()]
                with np.errstate(invalid="ignore", divide="ignore"):
                    slope = np.asarray(
                        (y[0] - intercept) / design[0, 0],
                        dtype=result_dtype,
                    )[()]
                if slope.real == 0 and slope.imag == 0:
                    slope = np.asarray(0j, dtype=result_dtype)[()]
                return slope, intercept
            slope, intercept = scipy_linalg.solve_triangular(
                design,
                y,
                lower=False,
                check_finite=False,
            )
            return slope, intercept
        if not np.all(np.isfinite(design)):
            raise ValueError(
                "linearFit design matrix contains Infs or NaNs"
            )
        getrf = scipy_linalg.get_lapack_funcs("getrf", (design,))
        factors, pivots, info = getrf(
            np.array(design, copy=True),
            overwrite_a=True,
        )
        if info < 0:
            raise ValueError(
                f"LAPACK getrf received an invalid argument {-info}"
            )
        if info > 0:
            raise np.linalg.LinAlgError(
                f"linearFit design matrix is singular at pivot {info}"
            )
        getrs = scipy_linalg.get_lapack_funcs("getrs", (factors, y))
        solution, info = getrs(
            factors,
            pivots,
            np.array(y, copy=True),
            trans=0,
            overwrite_b=True,
        )
        if info != 0:
            raise ValueError(
                f"LAPACK getrs received an invalid argument {-info}"
            )
        slope, intercept = solution
    else:
        if not np.all(np.isfinite(design)) or not np.all(np.isfinite(y)):
            return _machine_nonfinite_linear_fit(design, y)
        # QRPivoted's Julia ldiv path estimates rank with
        # ``min(m, n) * eps(real(T))``. LAPACK's bare gelsy default is only
        # ``eps`` and retains a spurious second direction at the boundary.
        rank_cutoff = (
            min(design.shape)
            * np.finfo(np.asarray(design.real).dtype).eps
        )
        solution, _residuals, rank, _singular_values = scipy_linalg.lstsq(
            design,
            y,
            cond=rank_cutoff,
            lapack_driver="gelsy",
            # Julia passes non-finite entries through to LAPACK, where they
            # propagate into the result rather than being pre-rejected.
            check_finite=False,
        )
        # Accelerate's ``gelsy`` retains a second direction for one 3x2
        # Float64 system that the Julia 1.11.6 OpenBLAS reference classifies
        # as rank one. Use a scalar Householder step to measure the residual
        # second direction without another LAPACK-dependent rank decision.
        # The sqrt(m) factor is the usual normwise conversion of a component
        # roundoff bound; the adjacent four-epsilon rank-two case remains
        # outside it.
        if (
            rank == 2
            and design.shape == (3, 2)
            and design.dtype == np.dtype(np.float64)
        ):
            coordinates = design[:, 0]
            column_norm = np.sqrt(
                coordinates[0] * coordinates[0]
                + coordinates[1] * coordinates[1]
                + coordinates[2] * coordinates[2]
            )
            beta = -np.copysign(column_norm, coordinates[0])
            tau = (beta - coordinates[0]) / beta
            tail = coordinates[1:] / (coordinates[0] - beta)
            projection = tau * (1.0 + tail[0] + tail[1])
            second_diagonal = np.hypot(
                1.0 - tail[0] * projection,
                1.0 - tail[1] * projection,
            )
            roundoff_limit = (
                rank_cutoff
                * np.sqrt(design.shape[0])
                * abs(beta)
            )
            if second_diagonal <= roundoff_limit:
                # Solve the resulting rank-one model A = x*[1, c]
                # directly. This is the same minimum-norm problem as gelsy's
                # rank-one branch, without sending the boundary case back to
                # a backend-specific rank estimator.
                gram = (
                    coordinates[0] * coordinates[0]
                    + coordinates[1] * coordinates[1]
                    + coordinates[2] * coordinates[2]
                )
                column_ratio = (
                    coordinates[0]
                    + coordinates[1]
                    + coordinates[2]
                ) / gram
                projected_rhs = (
                    coordinates[0] * y[0]
                    + coordinates[1] * y[1]
                    + coordinates[2] * y[2]
                ) / gram
                scale = projected_rhs / (
                    1.0 + column_ratio * column_ratio
                )
                solution = np.asarray(
                    [scale, scale * column_ratio]
                )
        slope, intercept = solution
    return slope, intercept


class _BinaryBigFloat:
    """Finite MPFR-style arithmetic used by Julia's generic BigFloat QR.

    ``Decimal`` is the public arbitrary-precision scalar in this port, but
    Julia's ``BigFloat`` rounds every elementary operation in base two.  A
    rank-deficient QR exposes those otherwise invisible rounding decisions.
    Keeping exact dyadic fractions and rounding after each operation gives us
    Julia 1.11's operation order without adding an external MPFR dependency.
    """

    def __init__(self, precision: int):
        self.precision = precision

    @staticmethod
    def _floor_log2(value: Fraction) -> int:
        value = abs(value)
        numerator = value.numerator
        denominator = value.denominator
        exponent = numerator.bit_length() - denominator.bit_length()
        if exponent >= 0:
            if numerator < denominator << exponent:
                exponent -= 1
        elif numerator << -exponent < denominator:
            exponent -= 1
        return exponent

    @staticmethod
    def _round_integer_ratio(numerator: int, denominator: int) -> int:
        quotient, remainder = divmod(numerator, denominator)
        comparison = 2 * remainder - denominator
        if comparison > 0 or (comparison == 0 and quotient % 2):
            quotient += 1
        return quotient

    def round(self, value: Fraction | int) -> Fraction:
        value = Fraction(value)
        if value == 0:
            return value
        sign = -1 if value < 0 else 1
        value = abs(value)
        shift = self._floor_log2(value) - (self.precision - 1)
        if shift >= 0:
            significand = self._round_integer_ratio(
                value.numerator, value.denominator << shift
            )
        else:
            significand = self._round_integer_ratio(
                value.numerator << -shift, value.denominator
            )
        if significand == 1 << self.precision:
            significand >>= 1
            shift += 1
        rounded = (
            Fraction(significand << shift)
            if shift >= 0
            else Fraction(significand, 1 << -shift)
        )
        return sign * rounded

    def from_decimal(self, value: Decimal) -> Fraction:
        if not value.is_finite():
            raise ValueError("binary BigFloat emulation requires finite values")
        return self.round(Fraction(value))

    def to_decimal(self, value: Fraction) -> Decimal:
        # A p-bit significand can require ceil(p*log10(2)) significant
        # decimal digits to expose its final binary ulp.  Decimal's nominal
        # context may be one digit shorter (77 digits maps to 256 bits), so
        # convert with that representation width instead of erasing the
        # low-order bit on the way back to the public scalar.
        representation_digits = ceil(1 + self.precision / log2(10))
        with localcontext() as conversion:
            conversion.prec = representation_digits
            return Decimal(value.numerator) / Decimal(value.denominator)

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        return self.round(left + right)

    def subtract(self, left: Fraction, right: Fraction) -> Fraction:
        return self.round(left - right)

    def multiply(self, left: Fraction, right: Fraction) -> Fraction:
        return self.round(left * right)

    def divide(self, numerator: Fraction, denominator: Fraction) -> Fraction:
        return self.round(numerator / denominator)

    def inverse(self, value: Fraction) -> Fraction:
        return self.divide(Fraction(1), value)

    def sqrt(self, value: Fraction) -> Fraction:
        if value == 0:
            return value
        if value < 0:
            raise ValueError("cannot take the square root of a negative value")

        # Round sqrt(value) directly to a p-bit dyadic.  Comparing the exact
        # rational square against (k + 1/2)^2 implements MPFR's nearest-even
        # tie rule without a guard-precision approximation.
        shift = self._floor_log2(value) // 2 - (self.precision - 1)
        squared_scale = 2 * shift
        scaled = (
            value / Fraction(1 << squared_scale)
            if squared_scale >= 0
            else value * Fraction(1 << -squared_scale)
        )
        numerator = scaled.numerator
        denominator = scaled.denominator
        floor_root = isqrt(numerator // denominator)
        midpoint_comparison = (
            4 * numerator
            - denominator * (2 * floor_root + 1) ** 2
        )
        significand = floor_root
        if midpoint_comparison > 0 or (
            midpoint_comparison == 0 and floor_root % 2
        ):
            significand += 1
        if significand == 1 << self.precision:
            significand >>= 1
            shift += 1
        return (
            Fraction(significand << shift)
            if shift >= 0
            else Fraction(significand, 1 << -shift)
        )

    def norm(self, values: Sequence[Fraction]) -> Fraction:
        # These finite, two-column regression systems use Julia's unscaled
        # generic_norm2 branch.  Its accumulator starts with the first square.
        total = self.multiply(values[0], values[0])
        for value in values[1:]:
            total = self.add(total, self.multiply(value, value))
        return self.sqrt(total)

    def dot(self, left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
        # Julia's generic dot starts from a typed zero and accumulates in
        # iteration order.
        total = Fraction(0)
        for left_value, right_value in zip(left, right):
            total = self.add(
                total, self.multiply(left_value, right_value)
            )
        return total


def _decimal_bigfloat_precision() -> int:
    """Map Decimal significant digits to an equally capable binary precision."""

    return max(2, ceil(getcontext().prec * log2(10)))


def _bigfloat_reflector_inplace(
    arithmetic: _BinaryBigFloat,
    matrix: list[list[Fraction]],
    column: int,
    row: int,
) -> Fraction:
    """Apply Julia 1.11's generic ``reflector!`` storage convention."""

    norm = arithmetic.norm(
        [matrix[index][column] for index in range(row, len(matrix))]
    )
    first = matrix[row][column]
    if norm == 0:
        return Fraction(0)
    signed_norm = norm if first >= 0 else -norm
    leading = arithmetic.add(first, signed_norm)
    matrix[row][column] = -signed_norm
    for index in range(row + 1, len(matrix)):
        matrix[index][column] = arithmetic.divide(
            matrix[index][column], leading
        )
    return arithmetic.divide(leading, signed_norm)


def _decimal_linear_fit(
    xs: list[Decimal], ys: list[Decimal]
) -> tuple[Decimal, Decimal]:
    """Two-column counterpart of Julia 1.11's BigFloat left division."""

    rows = len(xs)
    if not all(value.is_finite() for value in (*xs, *ys)):
        # Julia's square BigFloat path checks the *matrix* for non-finite
        # entries before generic LU, but does not reject a non-finite RHS.
        # Rectangular generic QR instead propagates non-finite arithmetic to a
        # pair of NaNs.  Decimal traps these operations by default, so model
        # the two Julia branches explicitly.
        if rows != 2:
            nan = Decimal("NaN")
            return nan, nan

        # This is Julia 1.11's two-by-two generic row-pivoted LU operation
        # order, evaluated with Decimal's IEEE non-finite propagation enabled.
        with localcontext() as context:
            context.traps[InvalidOperation] = False
            context.traps[DivisionByZero] = False
            context.traps[Overflow] = False
            # The generic backslash polyalgorithm recognizes an already
            # upper-triangular two-column design before attempting LU.
            if xs[1] == 0:
                if xs[0] == 0:
                    raise np.linalg.LinAlgError(
                        "linearFit design matrix is singular at pivot 1"
                    )
                second = ys[1]
                first = (ys[0] - second) / xs[0]
                return first, second
            if not all(value.is_finite() for value in xs):
                raise ValueError(
                    "linearFit design matrix contains Infs or NaNs"
                )

            matrix = [
                [xs[0], Decimal(1)],
                [xs[1], Decimal(1)],
            ]
            rhs = list(ys)
            if abs(matrix[1][0]) > abs(matrix[0][0]):
                matrix[0], matrix[1] = matrix[1], matrix[0]
                rhs[0], rhs[1] = rhs[1], rhs[0]
            if matrix[0][0] == 0:
                raise np.linalg.LinAlgError(
                    "linearFit design matrix is singular at pivot 1"
                )
            lower = matrix[1][0] * (Decimal(1) / matrix[0][0])
            second_pivot = matrix[1][1] - lower * matrix[0][1]
            if second_pivot == 0:
                raise np.linalg.LinAlgError(
                    "linearFit design matrix is singular at pivot 2"
                )

            # LinearAlgebra.ldiv! dynamically routes a numerically diagonal
            # UnitLowerTriangular factor through its upper-triangular kernel.
            # That kernel still evaluates the structural-zero product, which
            # is observable for an infinite RHS (0*Inf -> NaN).
            if lower == 0:
                rhs[0] = rhs[0] - Decimal(0) * rhs[1]
            else:
                rhs[1] = rhs[1] - lower * rhs[0]
            second = rhs[1] / second_pivot
            first = (rhs[0] - matrix[0][1] * second) / matrix[0][0]
            return first, second

    arithmetic = _BinaryBigFloat(_decimal_bigfloat_precision())
    x_values = [arithmetic.from_decimal(value) for value in xs]
    y_values = [arithmetic.from_decimal(value) for value in ys]

    if rows == 0:
        return Decimal(0), Decimal(0)

    if rows == 2:
        # Square BigFloat designs use Julia's generic row-pivoted LU.  Preserve
        # the separate inverse/multiply operations used by generic_lufact!,
        # followed by its unit-lower and upper triangular substitutions.
        matrix = [
            [x_values[0], arithmetic.round(1)],
            [x_values[1], arithmetic.round(1)],
        ]
        rhs = list(y_values)
        if abs(matrix[1][0]) > abs(matrix[0][0]):
            matrix[0], matrix[1] = matrix[1], matrix[0]
            rhs[0], rhs[1] = rhs[1], rhs[0]
        if matrix[0][0] == 0:
            raise np.linalg.LinAlgError(
                "linearFit design matrix is singular at pivot 1"
            )
        lower = arithmetic.multiply(
            matrix[1][0], arithmetic.inverse(matrix[0][0])
        )
        matrix[1][0] = lower
        matrix[1][1] = arithmetic.subtract(
            matrix[1][1],
            arithmetic.multiply(lower, matrix[0][1]),
        )
        if matrix[1][1] == 0:
            raise np.linalg.LinAlgError(
                "linearFit design matrix is singular at pivot 2"
            )
        rhs[1] = arithmetic.subtract(
            rhs[1], arithmetic.multiply(lower, rhs[0])
        )
        second = arithmetic.divide(rhs[1], matrix[1][1])
        first = arithmetic.divide(
            arithmetic.subtract(
                rhs[0], arithmetic.multiply(matrix[0][1], second)
            ),
            matrix[0][0],
        )
        return (
            arithmetic.to_decimal(first),
            arithmetic.to_decimal(second),
        )

    # Rectangular designs use Julia 1.11's generic unblocked, column-pivoted
    # QR.  This intentionally does not estimate rank: an exactly dependent
    # column can leave a one-ulp pivot whose division is observable.
    matrix = [
        [value, arithmetic.round(1)] for value in x_values
    ]
    permutation = [0, 1]
    taus: list[Fraction] = []
    for column in range(min(rows, 2)):
        norms = []
        for candidate in range(column, 2):
            norms.append(
                arithmetic.norm(
                    [
                        matrix[row][candidate]
                        for row in range(column, rows)
                    ]
                )
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
        tau = _bigfloat_reflector_inplace(
            arithmetic, matrix, column, column
        )
        taus.append(tau)
        for target_column in range(column + 1, 2):
            projection = arithmetic.add(
                matrix[column][target_column],
                arithmetic.dot(
                    [
                        matrix[row][column]
                        for row in range(column + 1, rows)
                    ],
                    [
                        matrix[row][target_column]
                        for row in range(column + 1, rows)
                    ],
                ),
            )
            projection = arithmetic.multiply(tau, projection)
            matrix[column][target_column] = arithmetic.subtract(
                matrix[column][target_column], projection
            )
            for row in range(column + 1, rows):
                matrix[row][target_column] = arithmetic.subtract(
                    matrix[row][target_column],
                    arithmetic.multiply(
                        projection, matrix[row][column]
                    ),
                )

    transformed = list(y_values)
    if rows < 2:
        transformed.append(Fraction(0))
    for column, tau in enumerate(taus):
        projection = arithmetic.add(
            transformed[column],
            arithmetic.dot(
                [
                    matrix[row][column]
                    for row in range(column + 1, rows)
                ],
                [
                    transformed[row]
                    for row in range(column + 1, rows)
                ],
            ),
        )
        projection = arithmetic.multiply(tau, projection)
        transformed[column] = arithmetic.subtract(
            transformed[column], projection
        )
        for row in range(column + 1, rows):
            transformed[row] = arithmetic.subtract(
                transformed[row],
                arithmetic.multiply(matrix[row][column], projection),
            )

    if rows == 1:
        # Julia's _wide_qr_ldiv! converts the 1×2 trapezoid to triangular
        # form with a second reflector, solves the leading system, pads with
        # zero, and then applies that reflector to the minimum-norm solution.
        wide_matrix = [[matrix[0][0], matrix[0][1]]]
        wide_norm = arithmetic.norm(wide_matrix[0])
        wide_first = wide_matrix[0][0]
        wide_signed_norm = (
            wide_norm if wide_first >= 0 else -wide_norm
        )
        wide_leading = arithmetic.add(
            wide_first, wide_signed_norm
        )
        wide_matrix[0][0] = -wide_signed_norm
        wide_matrix[0][1] = arithmetic.divide(
            wide_matrix[0][1], wide_leading
        )
        wide_tau = arithmetic.divide(
            wide_leading, wide_signed_norm
        )
        transformed[0] = arithmetic.divide(
            transformed[0], wide_matrix[0][0]
        )
        transformed[1] = Fraction(0)
        projection = arithmetic.multiply(wide_tau, transformed[0])
        transformed[0] = arithmetic.subtract(
            transformed[0], projection
        )
        transformed[1] = arithmetic.subtract(
            transformed[1],
            arithmetic.multiply(wide_matrix[0][1], projection),
        )
        solution = transformed
    else:
        if matrix[1][1] == 0:
            raise np.linalg.LinAlgError(
                "linearFit design matrix is singular at pivot 2"
            )
        second = arithmetic.divide(transformed[1], matrix[1][1])
        first = arithmetic.divide(
            arithmetic.subtract(
                transformed[0],
                arithmetic.multiply(matrix[0][1], second),
            ),
            matrix[0][0],
        )
        solution = [first, second]

    inverse = [0, 0]
    for index, original in enumerate(permutation):
        inverse[original] = index
    return (
        arithmetic.to_decimal(solution[inverse[0]]),
        arithmetic.to_decimal(solution[inverse[1]]),
    )


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


def _splat_roi_index(roi: Any) -> Any:
    """Translate Julia's ``field[roi...]`` selector convention."""

    if isinstance(roi, list):
        return tuple(roi)
    if isinstance(roi, np.ndarray) and roi.ndim == 1:
        return tuple(roi.tolist())
    return roi


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
    for image in images:
        _require_julia_numeric_array(image.data, "getOrientation image")
    if images:
        def element_key(image: LatticeField) -> Any:
            dtype = np.asarray(image.data).dtype
            if dtype.kind != "O":
                return dtype
            key = _object_numeric_element_key(image.data)
            if key is not None:
                return key
            logical = image._logical_object_type
            if logical in (Decimal, _MPFR):
                return _MPFR
            if logical in (_MPFRComplex, _MPC):
                return _MPC
            return logical

        first_key = element_key(images[0])
        if any(element_key(image) != first_key for image in images[1:]):
            raise TypeError("linImgs must have one concrete element type")
    if roi is not None:
        roi_index = _splat_roi_index(roi)
        images = [image[roi_index] for image in images]
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
    # Julia's ordinary ``cs[:, i]`` indexing allocates a concrete Vector;
    # NumPy returns a strided view for the analogous column selection.
    # Materialize here before entering linearFit's concrete-Vector boundary.
    xs, xi = linearFit(idxs, centers[:, 0].copy())
    ys, yi = linearFit(idxs, centers[:, 1].copy())
    if isinstance(xs, Decimal) or isinstance(ys, Decimal):
        theta = _decimal_atan2(ys, xs)
        return np.asarray([xi, yi], dtype=object), theta
    if (
        _is_mpfr(xs)
        or _is_mpfr(ys)
        or _is_complex_bigfloat_scalar(xs)
        or _is_complex_bigfloat_scalar(ys)
    ):
        with _bigfloat_context():
            direction = (
                _as_mpfr_complex(xs)
                + _MPFRComplex(0, 1) * _as_mpfr_complex(ys)
            )
            theta = gmpy2.atan2(direction.imag, direction.real)
        return np.asarray([xi, yi], dtype=object), theta
    # Construct the complex value directly so a QR-produced ``-0.0`` slope
    # keeps its sign, as it does in Julia's ``xs + im * ys`` expression.
    theta = np.angle(complex(xs, ys))
    return np.asarray([xi, yi]), float(theta)


def _julia_zero_for_field_data(data: Any) -> Any:
    """Return ``zero(T)`` for the concrete element type represented by *data*."""

    array = np.asarray(data)
    if array.dtype.kind == "O" and array.size:
        return _julia_typed_zero(array.ravel(order="F")[0])
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
    if isinstance(theta, (_MPFR, _MPQ, _MPZ)):
        with _bigfloat_context():
            sine, cosine = gmpy2.sin_cos(_to_mpfr(theta))
            return _MPFR(sine), _MPFR(cosine)
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
    bc: Any = _OMITTED,
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
    _require_julia_numeric_array(f.data, "dualate field")
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
    field = f if roi is None else f[_splat_roi_index(roi)]
    boundary = (
        _julia_zero_for_field_data(field.data)
        if bc is _OMITTED
        else bc
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
    if _object_contains_mpfr(rotation) or _object_contains_mpfr(dual_origin):
        origin = _julia_array_array_operation(
            rotation, dual_origin, np.matmul
        )
    elif _object_contains_decimal(rotation) or _object_contains_decimal(dual_origin):
        origin = _as_decimal_array(rotation) @ _as_decimal_array(dual_origin)
    else:
        origin = rotation @ dual_origin
    dx = _julia_array_scalar_operation(
        rotation[:, 0], _step(dual_lattice[0]), np.multiply
    )
    dy = _julia_array_scalar_operation(
        rotation[:, 1], _step(dual_lattice[1]), np.multiply
    )
    # Julia evaluates this untyped keyword in a Boolean context here, before
    # constructing or evaluating the interpolation comprehension.  Reject
    # Python's truthy values (including ``None`` and integers) at the same
    # point so factory side effects remain observable but no scalar
    # interpolation call occurs.
    if not isinstance(naturalize, (bool, np.bool_)):
        raise TypeError("dualate naturalize must be boolean")
    naturalize = bool(naturalize)
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
    # The built-in natural spline has an optimized paired-coordinate path.
    # Keep the scalar CartesianIndex loop for custom factories: their call
    # order, side effects, heterogeneous values, and failure point are part of
    # the compatibility contract.
    data = NotImplemented
    if interpolation is cubic_spline_interpolation:
        data = interp._evaluate_paired_2d((xcoords, ycoords))
    if data is NotImplemented:
        indices = (
            tuple(reversed(index))
            for index in np.ndindex(tuple(reversed(xcoords.shape)))
        )
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
                    (
                        index,
                        interp(xcoords[index], ycoords[index]),
                    )
                )
            values = [value for _index, value in evaluated]
            promoted = _julia_collect_comprehension_results(values)
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
            data = np.asarray(value.data)
            if data.dtype.kind == "O" and _object_contains_decimal(data):
                return _mpfr_object_operation(np.remainder, data, 1)
            return _julia_array_scalar_operation(data, 1, np.remainder)
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
    dir: Any = _OMITTED,
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
