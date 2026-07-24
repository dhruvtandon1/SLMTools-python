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

The commit above is the behavioral source of truth. A future upstream sync
updates the commit, implementation, tests, and package version together.
`requirements-validation.txt` records the exact macOS arm64 Python environment
used for the release audit without acting as a cross-platform dependency lock.

## Scope

The package root exposes the same 100 unique names exported by the audited
Julia module. The ordered list is `slmtools.JULIA_EXPORTS` in
`src/slmtools/__init__.py` and is checked by the test suite.

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
3. Translate Julia's one-based user indices to zero-based Python indices.
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
   Map Julia's FFTW, image, font, interpolation, and optimal-transport
   operations to the declared Python dependencies.
6. Represent a Julia failure with the corresponding Python error type. A
   source path that does not execute successfully at the audited commit is
   outside the source's defined functionality and receives no new semantics
   in the translation.
7. Keep Python binding mechanics small and local while preserving each
   translated algorithm's observable result.
8. An empty `coarsen` result never evaluates its reducer. The default and
   supported NumPy reductions (`sum`, `mean`, `min`/`amin`, and `max`/`amax`)
   derive their result dtype from the input dtype.
   Python cannot recover Julia's compiler-inferred return type for an arbitrary
   callable without executing or inspecting user code, so an empty result with
   a custom reducer uses object storage without calling the reducer.
9. No successful source behavior is intentionally changed in this release.
   The Python binding mechanics documented here—checked detached raw array
   snapshots, eager range storage, and the unknowable dtype of an empty
   arbitrary-reducer result—do not create new numerical functionality. Any
   future algorithmic divergence is a deliberate versioned change with a
   focused test and an update to this contract.

Function docstrings and tests carry detailed input contracts and regression
examples; they are preferable to a duplicate hand-maintained API catalog.

### Known failures inherited from the audited source

These paths do not complete successfully at the audited Julia commit, so this
port records and preserves their lack of functionality rather than inventing
an algorithmic repair:

- default-phase `gs`/`gsLog` for `Intensity`, positive-iteration Float32
  `gs`, and several concrete phase/helper dispatch combinations;
- `hermiteBasis` and, transitively, `FrFTBasis`;
- rectangular `otPhase2`, its ignored wavelength behavior, singleton
  `scalarPotentialN` without an explicit anchor, and dense-OT target axes of
  length two;
- recognized `saveBeam` outputs, matched directory loading without the
  source's required trailing separator, and unsupported
  `pdotBeamEstimate(..., LFine=...)` range families;
- positive-iteration convolutional Sinkhorn with a non-Float64 target;
- the optional `SubImages.padmultiple(..., padall=...)` source bug.

Where literal translation would hang, depend on uninitialized memory, or
attempt an unbounded allocation, Python raises a focused corresponding error.
It still does not return a fabricated successful result. Each boundary has a
regression test and an explanatory implementation docstring; future fixes
belong in the Julia authority first and are ported only after the audited
commit is advanced.

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

Artifact construction is a separate CI/release gate, not a claim made by the
unit-test suite. Before release, build and validate both package artifacts:

```bash
python -m build
python -m twine check dist/*
```
