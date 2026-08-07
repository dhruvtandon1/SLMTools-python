# SLMTools for Python

`slmtools` is a Python translation of Hogan Lab's Julia
[`SLMTools`](https://github.com/hoganphysics/SLMTools) package. The current
release tracks Julia `SLMTools` 0.3.0 at
[`ea1c1c9`](https://github.com/hoganphysics/SLMTools/commit/ea1c1c9c06b4b2dc46372ac7ee031301b604a007).
It preserves the original public names, field tags, lattice metadata, phase
and FFT conventions, numerical algorithms, and column-major point ordering.

The Python translation is maintained as its own Git repository. The audited
Julia checkout is the reference used for parity tests.

## Install

Python 3.13 or newer is supported. From a source checkout:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[test,plot]"
python -m pytest -p no:cacheprovider -W error
```

The distribution name is `hoganlab-slmtools`; the Python import is
`slmtools`. The older PyPI distribution named `slmtools` is unrelated to this
project.

## Example

Public functions retain their Julia names, so an original workflow translates
directly:

```python
import numpy as np
from slmtools import (
    Intensity,
    dualShiftLattice,
    gs,
    lfGaussian,
    lfRing,
    look,
    natlat,
    otPhase,
    otPhase2,
    sft,
    square,
)

# Dense OT constructs a matrix over all lattice points.  Keep this quick-start
# deliberately small; 16×16 needs a 256×256 cost matrix.
size = 16
input_lattice = natlat((size, size))
target_lattice = dualShiftLattice(input_lattice)

input_beam = lfGaussian(Intensity, input_lattice, 1.0)
target_beam = lfRing(Intensity, target_lattice, 2.5, 0.5)

phase_ot = otPhase(input_beam, target_beam, 0.002)
phase_ot2 = otPhase2(input_beam, target_beam, 0.0002, 200)
phase_gs = gs(input_beam, target_beam, 100, phase_ot)

output_ot = square(sft(np.sqrt(input_beam) * phase_ot))
output_ot2 = square(sft(np.sqrt(input_beam) * phase_ot2))
output_gs = square(sft(np.sqrt(input_beam) * phase_gs))
image = look(target_beam, output_ot, output_ot2, output_gs)
```

`image` is a horizontally concatenated, display-ready grayscale NumPy array.
As in Julia, `look` returns image data and does not open a plotting window.

## Core model

`LatticeField` (`LF`) combines:

- an N-dimensional checked NumPy-compatible `data` façade;
- an N-tuple of regular coordinate axes in `L`;
- wavelength times focal length in `flambda`;
- a semantic `field_type`, such as `Intensity`, `RealPhase`, or
  `ComplexAmplitude`.

```python
import numpy as np
from slmtools import Intensity, LF, natlat

lattice = natlat((32, 32))
field = LF[Intensity](np.ones((32, 32)), lattice)
```

Real phase is measured in cycles, not radians. `wrap(phi)` uses
`exp(2πi*phi)`. `sft` and `isft` use the original shifted FFT definitions
without adding a physical sampling factor to the values.

Ordinary Python array and field indices are zero-based. A few public
dimension-number arguments remain one-based where that number is part of the
Julia calculation; `PORTING.md` lists those exceptions. Operations that
flatten or reshape lattice data use Fortran order so that Julia's
column-major point enumeration remains unchanged.

Direct Boolean and advanced indexing through `field.data` follows Julia's
column-major indexing rules. When passing field values to a NumPy consumer
that assumes NumPy's row-major advanced-index semantics (including
`numpy.testing`), use `field.data.copy()` to obtain a current ordinary ndarray
snapshot.

Python evaluates chained operators from left to right, whereas Julia can
dispatch an unparenthesized field expression as one variadic call. Python
field operators are deliberately eager, binary, non-mutating, and
value-semantic, so `a + b + c` means `(a + b) + c`. When translating or
comparing such a chain, use the equivalently parenthesized binary Julia
expression. Algorithms that genuinely require aggregate-before-construction
semantics perform that reduction internally and construct one final field;
they do not retain hidden operands or change Python's operator behavior.

## Port contract

[`PORTING.md`](https://github.com/dhruvtandon1/SLMTools-python/blob/main/PORTING.md)
is the single maintenance reference for the source baseline, module map,
translation rules, and parity checks. Function-level API details live with
the implementation docstrings, and the exact Julia export surface is
`slmtools.JULIA_EXPORTS`.

The repository also contains deterministic translations of the original
notebook workflows in
[`examples/workflows.py`](https://github.com/dhruvtandon1/SLMTools-python/blob/main/examples/workflows.py).

## License

MIT, matching the Julia project. See
[`LICENSE`](https://github.com/dhruvtandon1/SLMTools-python/blob/main/LICENSE).
