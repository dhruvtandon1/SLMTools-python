# Port contract

This file is the shared maintenance reference for humans and coding agents.
It describes what the Python port tracks and how parity is checked. Behavioral
mismatches with the audited Julia source are bugs, not approved port
exceptions.

## Source authority

- Upstream repository: <https://github.com/hoganphysics/SLMTools>
- Upstream version: `0.3.0`
- Audited commit: `ea1c1c9c06b4b2dc46372ac7ee031301b604a007`
- Audited Julia environment: Julia `1.11.6` with the upstream
  `Project.toml` and `Manifest.toml`
- Python package version: `0.3.0`
- Supported Python runtime: CPython `3.13` or newer

The commit above is the behavioral source of truth. A future upstream sync
updates the commit, implementation, tests, and package version together.
`requirements-validation.txt` records the exact macOS arm64 Python environment
used for the release audit without acting as a cross-platform dependency lock.

## Scope

The package root exposes the same 100 unique names exported by the audited
Julia module. The ordered list is `slmtools.JULIA_EXPORTS` in
`src/slmtools/__init__.py`. Tests compare it with a separately stored fixture
and, when the audited checkout is available, independently parse the Julia
`include`/`export` declarations and verify that fixture.

`src/SubImages.jl` exists in the upstream repository but is not included by
`SLMTools.jl`. Its translation therefore remains the optional
`slmtools.subimages` module and stays outside the package-root export list.

## Module map

| Julia source | Python source |
|---|---|
| `src/SLMTools.jl` | `src/slmtools/__init__.py` |
| `src/LatticeFields/LatticeField.jl` | `src/slmtools/lattice_field.py` |
| `src/LatticeFields/LatticeUtils.jl` | `src/slmtools/lattice_utils.py` |
| `src/LatticeFields/Resampling.jl` | `src/slmtools/resampling.py` |
| `src/LatticeFields/DualLattices.jl` | `src/slmtools/dual_lattices.py` |
| `src/LFIO/Misc.jl` | `src/slmtools/misc.py` |
| `src/LFIO/ImageProcessing.jl` | `src/slmtools/image_processing.py` |
| `src/LFIO/LFTemplates.jl` | `src/slmtools/templates.py` |
| `src/PhaseRetrieval/IFT.jl` | `src/slmtools/ift.py` |
| `src/PhaseRetrieval/OT.jl` | `src/slmtools/ot.py` |
| `src/LFIO/Visualization.jl` | `src/slmtools/visualization.py` |
| `src/LFIO/BMP8Writer.jl` | `src/slmtools/bmp8.py` |
| `src/SubImages.jl` | `src/slmtools/subimages.py` |

## Translation rules

1. Preserve public function names and aliases, including Julia camel case.
2. Preserve field tags, lattice values, `flambda`, shapes, dtypes, mutation
   behavior, numerical operation order, and FFT/phase conventions.
3. Translate positional array, field, and plain-sequence indices from Julia's
   one-based convention to Python's zero-based convention. Dimension-number
   arguments retain their source convention when that number is part of the
   public calculation: `collapse(..., i)` and `toDim(..., d, n)` remain
   Julia-style one-based, including their upstream out-of-range behavior.
   OT plain-sequence `originIdx`/`idx` positions are zero-based, while
   `sumDim`, `fixDims`, and `dimOrder` are one-based; OT dimension selectors
   additionally accept zero as an explicit Python alias for the first axis.
   Preserve Julia's column-major enumeration with Fortran-order reshape and
   flatten operations.
4. Keep field `+` and `*` eager, binary, non-mutating, and value-semantic.
   Python chains are left folds, so verify them against explicitly
   parenthesized Julia binary operations. Do not retain hidden operands,
   inspect source or bytecode, or add alternate public arithmetic helpers.
   If a translated internal algorithm genuinely requires
   aggregate-before-construction semantics, validate every field, reduce its
   arrays locally with Julia-compatible dtype and order, and construct one
   field afterward.
5. Map Julia arrays and ranges to NumPy arrays plus retained range metadata.
   Map Julia's FFTW, linear algebra, image, font, interpolation, and
   optimal-transport operations to the declared Python dependencies.
6. Represent a Julia failure with the corresponding Python error type. A
   source path that does not execute successfully at the audited commit is
   outside the source's defined functionality and receives no new semantics
   in the translation.
7. Keep Python binding mechanics small and local while preserving each
   translated algorithm's observable result.
8. Treat Python `None` as Julia `nothing`, not as a universal spelling of
   "argument omitted." Where Julia overload selection or a non-`nothing`
   default distinguishes those cases, public functions use a private sentinel
   so omission selects the source default while explicit `None` follows the
   source's explicit-`nothing` result or failure. A merged Python dispatcher
   also rejects a supplied keyword when the selected Julia overload does not
   declare it; positional Julia parameters are not silently added as
   keyword-only API extensions.
9. An empty `coarsen` result never evaluates its reducer. The default and
   supported NumPy reductions (`sum`, `mean`, `min`/`amin`, and `max`/`amax`)
   derive their result dtype from the input dtype.
   Python cannot recover Julia's compiler-inferred return type for an arbitrary
   callable without executing or inspecting user code, so an empty result with
   a custom reducer uses object storage without calling the reducer. Explicit
   `reducer=None` on an empty result likewise uses object storage as NumPy's
   representable counterpart of Julia's empty `Array{Union{}}`.
10. Successful source behavior is preserved except for the explicit
   uninitialized-resampling safety boundary below. The Python binding
   mechanics documented here—checked detached raw array snapshots, eager
   range storage, and the unknowable dtype of an empty arbitrary-reducer
   result—do not create new numerical functionality. Any other algorithmic
   divergence is a deliberate versioned change with a focused test and an
   update to this contract.

Function docstrings and tests carry detailed input contracts and regression
examples; they are preferable to a duplicate hand-maintained API catalog.

### Known failures inherited from the audited source

These paths do not complete successfully at the audited Julia commit, so this
port records and preserves their lack of functionality rather than inventing
an algorithmic repair:

- default-phase `gs`/`gsLog` for `Intensity`, positive-iteration Float32
  `gs`, arbitrary-precision `gs`/`gsLog` phasors, and exact-number PDGS work
  tuples that cannot satisfy the source's `Float64`/`ComplexF64` assertions;
- `hermiteBasis` and, transitively, `FrFTBasis`;
- rectangular `otPhase2`, its ignored wavelength behavior, singleton
  `scalarPotentialN` without an explicit anchor, and dense-OT target axes of
  length two;
- `dualToGradients` with arbitrary-precision FFT inputs, `dualPhase` on a
  BigInt range, and Float64 `StepRangeLen` multiplication by an
  arbitrary-precision coefficient (including the affected `ldot` and
  centered `lfParabola` paths);
- every `LatticeField` constructor whose lattice range reports a non-platform
  length type, including endpoint-built BigInt, Rational{BigInt}, UInt64, and
  Int128 ranges: the source compares `size(data)` and `length.(L)` with
  type-sensitive `!==`, so matching numeric lengths still fail;
- default maximum normalization of complex OT cost matrices, complex
  `minimum`/`maximum` coarsening reducers, and visualization/template paths
  that try to order complex values;
- recognized `saveBeam` outputs, matched directory loading without the
  source's required trailing separator, and unsupported
  `pdotBeamEstimate(..., LFine=...)` range families;
- positive-iteration convolutional Sinkhorn with a non-Float64 target;
- `lfHeart`, `lfSmile`, and `lfPointer` with `flip=true` on a rectangular
  lattice, because the generated data is transposed and paired with the
  original lattice;
- text rendering when the selected FreeType strike is monochrome, including
  Windows Calibri at 12 and 16 pixels, because FreeTypeAbstraction asserts
  that every glyph bitmap is grayscale;
- the upstream `Pkg.test()` entry point, whose test sources still import the
  removed `SLMTools.LatticeTools` submodule.

Pillow normalizes monochrome strikes to grayscale before exposing their pixel
data, so the Python backend cannot identify that upstream failure without
also rejecting valid grayscale glyphs that happen to contain only binary
values. Such host-font calls may therefore render in Python and remain a
documented backend limitation; the port does not special-case font names or
pixel sizes.

Where literal translation would hang, depend on uninitialized memory, or
attempt an unbounded allocation, Python raises a focused corresponding error.
It still does not return a fabricated successful result. Each boundary has a
regression test and an explanatory implementation docstring; future fixes
belong in the Julia authority first and are ported only after the audited
commit is advanced.

### Intentional safety deviation

Some successful array and field `downsample`/`upsample` calls route
non-integral target lattices through `Interpolations.jl` and return zeros,
subnormals, or other uninitialized-looking values in the audited Julia
environment. Python intentionally raises a focused error for those nonempty
built-in paths instead of exposing nondeterministic memory-derived output.
This is a successful-call behavior change, not an inherited Julia exception.
Supported integer target ranges, empty targets, axis/lattice overloads, block
`coarsen`, and custom indexable interpolation factories retain Julia behavior.

### Successful upstream bugs retained

The optional `SubImages.padmultiple(..., padall=...)` call completes in Julia
but accidentally reapplies the four directional padding widths and ignores
the magnitude of a positive `padall`. Python preserves that result: positive
`padall` doubles positive directional pads and has no effect when they are
zero.

### Python representation boundary

Julia's parametric runtime alias `Lattice{N}` is represented by the Python
typing alias `tuple[LatticeAxis, ...]`; ordinary values remain tuples of
regular axes. Python has no direct runtime equivalent of Julia's
dimension-parameterized tuple alias. Exported field-tag classes, like their
Julia abstract-type counterparts, cannot be instantiated. A user-defined
`FieldVal` subclass is concrete and instantiable by default, paralleling a
user-declared Julia `struct <: FieldVal`.
Julia's non-exported `AbstractFFTs.ScaledPlan` wrapper is represented by
`slmtools.ift.ScaledFFTWPlan`; it remains outside the package-root export list.

### Lattice-field storage

Constructors that pass an `AbstractArray` directly to Julia's full field
constructor retain that array by reference; their Python counterparts retain
the supplied NumPy array as well. The partial `Intensity` and
`ComplexAmplitude` constructors still allocate their ramped or converted
results exactly where Julia broadcasts a new array.

`field.data` exposes checked ndarray façades over that authoritative storage.
Raw `np.asarray`/`view` escape hatches see detached snapshots because NumPy
`ufunc.at` can ignore a read-only flag. Storage-sharing views remain checked,
while allocating ndarray operations return ordinary mutable arrays. Checked
slice and advanced-index assignment uses Julia's `setindex_shape_check` and
column-major source/destination order: singleton dimensions may be rearranged
only when linear order is preserved, never broadcast. Multiple vector
selectors form Julia's Cartesian product rather than NumPy's paired advanced
index. Conversion failures retain the already-written column-major prefix.
Retained façades are weakly tracked, pruned during reads/registration, and
synchronized once per multi-element write batch. Reads made through a retained
checked façade resolve the authoritative storage and therefore see mutations
through a constructor array alias; an earlier raw `np.asarray`/`view` snapshot
intentionally remains detached.

That indexing contract also means NumPy utilities which internally apply
row-major Boolean masks must not consume the façade directly. In particular,
value assertions should pass `field.data.copy()` to `numpy.testing`. The
copy reads authoritative storage at the time it is made and returns a normal
ndarray; `np.asarray(field.data)` is only a detached snapshot and may be stale
after a later mutation through a constructor-array alias.

`LatticeAxis` is necessarily eager although Julia ranges are lazy. It has no
project-defined sample-count ceiling: every nonnegative length representable
by the platform index type reaches the host materialization-feasibility or
NumPy allocator boundary. A request larger than currently available host
memory raises a focused `MemoryError` before a fill can trigger an operating-
system OOM kill.

## Verification

Install and run the complete Python suite:

```bash
python -m pip install -e ".[test,plot]"
python -m pytest -p no:cacheprovider -W error
```

The suite checks:

- the exact 100-name Julia export surface;
- constructors, aliases, field arithmetic, indexing, mutation, and dtypes;
- lattice, interpolation, resampling, FFT, phase-retrieval, and OT numerics;
- image conversion, templates, visualization, and byte-level BMP output;
- fixed cross-language goldens and the translated workflow components;
- package metadata and import behavior.

The optional large-fixture check reads images from a separate upstream
checkout:

```bash
SLMTOOLS_JULIA_REPO=/absolute/path/to/SLMTools \
  python -m pytest -p no:cacheprovider -W error
```

The test only reads that checkout. Generated test output uses temporary
directories. Before using a fixture, it verifies that the checkout descends
from the audited commit and that the tracked Julia sources, Julia environment,
and required fixture files are unchanged; housekeeping-only descendants are
therefore accepted. Pixel-exact Arial Rounded MT Bold goldens run in the macOS
validation gate; Linux CI skips those cases when that system font is not
available. Linux CI likewise cannot certify the audited macOS-arm64 FFTW
planner/codelet behavior; that numerical gate must run separately on the
audited architecture with the matching SIMD-enabled FFTW backend.
Cancellation-heavy IFT cases can assign different phases to bins whose exact
value is zero when FFT backends leave different last-bit residuals there;
nondegenerate bins are compared with dtype-appropriate numerical tolerances.
Julia and NumPy use different random-number engines and array-filling
conventions. Reusing the same integer seed across languages therefore does not
promise identical samples from `lfRand` or from the randomized `gs`/`gsLog`
overloads. Cross-language numerical comparisons supply an explicit initial
phase; seeded stochastic tests assert the Python behavior without treating a
Julia seed as a portable random stream.

Artifact construction is a separate CI/release gate, not a claim made by the
unit-test suite. Before release, build and validate both package artifacts:

```bash
python -m build
python -m twine check dist/*
```
