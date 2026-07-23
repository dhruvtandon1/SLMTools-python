# SLMTools for Python

This is a local Python port of Hogan Lab's Julia `SLMTools` package.  It keeps
the Julia package's public names, phase convention, lattice metadata, shifted
FFT convention, field-tag rules, and numerical algorithms.  The port targets
the audited Julia source at commit `ea1c1c9c06b4b2dc46372ac7ee031301b604a007`
(`SLMTools` 0.3.0).

The port is deliberately a plain local folder.  It has no Git repository and
does not modify, import files from, or write outputs into the Julia checkout.

## Install

Python 3.11 or newer is required.  From this directory:

```bash
python -m pip install -e .
```

NumPy, Pillow, and pyFFTW are the runtime dependencies. pyFFTW supplies the
same FFTW backend family used by the Julia package; Matplotlib is optional and
is used only by `slmtools.subimages.imageToHeatmap`.
[`requirements-lock.txt`](requirements-lock.txt) records the exact NumPy,
Pillow, pyFFTW, and pytest versions used by the locked regression environment.

The public distribution name is `hoganlab-slmtools` because the unrelated
PyPI project named `slmtools` predates this port. The import remains
`import slmtools`, preserving the Julia package name in Python code.

## Example

The original workflow translates directly; public functions retain their
Julia camel-case names:

```python
import numpy as np
from slmtools import (
    Intensity, natlat, dualShiftLattice, lfGaussian, lfRing,
    otPhase, otPhase2, gs, sft, square, look,
)

N = 128
L0 = natlat((N, N))
dL0 = dualShiftLattice(L0)

inputBeam = lfGaussian(Intensity, L0, 1.0)
targetBeam = lfRing(Intensity, dL0, 2.5, 0.5)

phiOT = otPhase(inputBeam, targetBeam, 0.001)
phiOT2 = otPhase2(inputBeam, targetBeam, 0.0002, 200)
phiGS = gs(inputBeam, targetBeam, 100, phiOT)

outputOT = square(sft(np.sqrt(inputBeam) * phiOT))
outputOT2 = square(sft(np.sqrt(inputBeam) * phiOT2))
outputGS = square(sft(np.sqrt(inputBeam) * phiGS))
image = look(targetBeam, outputOT, outputOT2, outputGS)
```

`image` is one horizontally concatenated, display-ready grayscale NumPy
array.  As in Julia, `look` does not create or show a plotting window.

## Core model

`LatticeField` (`LF`) stores four pieces of information:

- `data`: an N-dimensional NumPy array;
- `L`: an N-tuple of regular, one-dimensional coordinate arrays;
- `flambda`: wavelength times focal length, defaulting to `1.0`;
- `field_type`: one of the semantic tags such as `Intensity`, `RealPhase`, or
  `ComplexAmplitude`.

Both ordinary Python and Julia-shaped construction are supported:

```python
import numpy as np
from slmtools import LF, Intensity, natlat

L = natlat((32, 32))
a = LF([[1.0] * 32] * 32, L, field_type=Intensity)
b = LF[Intensity]([[1.0] * 32] * 32, L)
# Julia's fully parameterized LF{S,T,N} constructor is also available. It
# requires exact dtype/dimension and deliberately bypasses partial clipping:
c = LF[Intensity, np.float32, 2](np.ones((32, 32), dtype=np.float32), L)
```

Real phase values are measured in **cycles**, not radians.  `wrap(phi)` uses
`exp(2πi*phi)`.  `sft` and `isft` mean exactly
`fftshift(fft(ifftshift(x)))` and `fftshift(ifft(ifftshift(x)))` over every
axis; no physical sampling factor is applied to the values.

NumPy indexing is necessarily zero-based and half-open.  A single scalar index
on an `LF` follows Julia's column-major linear order, Boolean selectors are
rejected, and omitted trailing selectors are accepted only for singleton
dimensions where Julia's dense indexing accepts them. Assignment accepts only
one linear integer or exactly one integer per field dimension. Every
reshape/flatten operation inside the OT algorithms uses Fortran order to
preserve Julia point enumeration.

`LatticeField` arithmetic is Python-native: `+` and `*` are eager, binary,
non-mutating, and value-semantic. Thus `a+b+c` is the left fold `(a+b)+c` and
matches Julia's explicitly parenthesized binary expression. Intermediates
contain no hidden operands, and the port does not inspect source/bytecode or
add alternate `julia_add`/`julia_mul` helpers. An internal algorithm that
requires aggregate-before-construction semantics validates its fields and
performs that reduction locally before constructing one result.

Dense `otPhase` and `pdotPhase` retain Julia's signed-maximum automatic-axis
scaling exactly. This includes the upstream zero denominator for a two-sample
target axis and the resulting invalid Inf/NaN geometry; the port does not
invent a replacement radius convention.

Broken or unfinished Julia paths are not silently completed in Python. They
remain failing or explicitly unsupported, and are listed as future work in
the compatibility document. This keeps the published port auditable as a
language translation rather than an unreviewed algorithm fork.

## Documentation and tests

- [`docs/API.md`](docs/API.md) maps every Julia export and the supported
  package-qualified helpers.
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) records conventions,
  dependency substitutions, and known upstream defects/future work.
- [`docs/TESTING.md`](docs/TESTING.md) describes the correctness and
  differential-parity strategy.
- [`examples/workflows.py`](examples/workflows.py) contains deterministic,
  scaled translations of all four publication-scale Julia notebooks.

Run the test suite with:

```bash
python -m pytest
```

The tests create all outputs in temporary directories.  Optional integration
checks can read the original large image fixtures by setting
`SLMTOOLS_JULIA_REPO`, but they never write to that repository.

## License

MIT, matching the Julia project.  See [`LICENSE`](LICENSE).
