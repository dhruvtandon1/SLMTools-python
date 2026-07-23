# Correctness strategy

The committed suite is intentionally compact.  Test files are grouped by
subsystem instead of creating one file for each module.

1. **API and contracts.** Every public function is exercised for values,
   shapes, dtype, semantic tag, lattice, `flambda`, copy/mutation behavior, and
   representative errors.  Aliases and the package-root export surface are
   checked explicitly.
2. **Numerical properties.** Tests cover normalization, dual reciprocal
   spacing, resampling centers and extrapolation, shifted-FFT inversion,
   phase unit modulus, GS/PDGS constraints, OT marginals, path integration,
   and explicit rejection of the upstream-broken rectangular `otPhase2` path.
3. **Deterministic Julia parity.** Small fixed Float64/ComplexF64 examples are
   evaluated with the checked-in Julia project/manifest, then compared to
   Python with operation-specific tolerances.  Random initial phases are
   supplied explicitly; the suites do not assume identical Julia and NumPy
   random streams.
4. **I/O bytes.** Synthetic RGB/RGBA inputs test the locked Colors.jl
   fixed-point Rec.601 intermediate arithmetic (including half-level rounding)
   and alpha removal. BMP tests use widths one through five so every
   row-padding case, palette entry, header field, and bottom-up row order is
   checked. `saveBeam` is checked as explicitly unavailable rather than
   assigning portable output semantics to its unusable Julia body.
5. **Workflow tests.** Reduced forms of the README and notebook workflows run
   end to end.  Multi-hour 1024²/1536² notebook experiments are excluded from
   ordinary tests.
6. **Independent-review regressions.** Focused cases cover integer-coordinate
   periodic wrapping; logical-range Float32, Float16, and Complex64
   interpolation; a 10,000-point spline axis; mixed scalar/vector query shape;
   cubic natural/Flat/Periodic endpoint and extrapolation combinations;
   periodic upper-endpoint wrapping; whole-axis norm-based lattice comparison; hidden singleton
   metadata; singleton-axis IFTs and `dualate`; the complete Julia 14×14
   machine-promotion matrix for binary addition/multiplication plus wrapping,
   overflow, and precision-sensitive values;
   filler-typed inexact padding; plot-image whitespace trimming; homogeneous
   and complex IFT dispatch; exact raw `gsIter`/`pdgsIter` element types;
   zero/negative logging cadence; low-precision `oneShot` arithmetic and
   metadata; `SinkhornIterBase!` same-type dispatch and mutation/return semantics;
   complex-target, Rational-result dtype, square-only dispatch, and
   underflowing `otPhase2`; checked complex `mapify`; direct-helper NaNs;
   Julia signed-maximum dense-OT geometry including its invalid
   target-length-two denominator; full typed construction and
   partial-copy coercion;
   checked scalar assignment (including concrete Rational/BigFloat-like
   object destinations and direct exact-to-Float16/32 rounding);
   Boolean and trailing-singleton direct indexing; stable binary Intensity
   value semantics and the documented n-ary syntax boundary; linear
   `subfield`; single-box `sublattice`; Fortran
   `toDim`; Float32 displacement/centroid arithmetic;
   exact Fraction lattices/centroids, helper zero types, `nabs`, intensity
   square root, Rational FFT conversion, and interpolation; Decimal centroid,
   windowing, lattice displacement, interpolation, complex phase wrapping,
   norm tolerance, mixed promotion, line orientation, and dualation;
   nonuniform multidimensional linear interpolation, direct versus nested
   one-dimensional boundary tuples, and tuple-valued fills;
   exact-number OT costs/integration and linear fitting; low-precision
   the working `pdotBeamEstimate(LFine=None)` path and explicit failure of its
   broken fine-lattice branch; signed-Int64 factor dispatch;
   OpenLibm DFT and the explicitly unavailable upstream Hermite/FrFT helpers;
   Rational and Decimal template promotion, values, matrix overloads, and
   random-sampler boundaries; integer, Float16/Float32 StepRangeLen, and
   Float64 TwicePrecision template/`ldot`/dual-lattice/pad/down/up goldens,
   plus low-precision dual-phase/blur and PDGS propagation; machine template dtypes;
   Julia-compatible exact-number parsing; pathological Float16 ranges whose
   materialized first coordinate differs from their literal start, including
   metadata-preserving field/sublattice slicing, empty-slice metadata,
   literal-concatenation `loadDir` routing including separator-free no-match
   scans, logical `dualate` routing, writable-subclass
   copy safety, and
   bit-exact pad/down/up consumers; exact locked-font raster placement and pair kerning;
   Rational display, Cartesian window bounds and empty nonpositive windows,
   empty-`coarsen` reducer dtypes, integer-only padding, integer-Intensity
   Float64 square-root propagation;
   statement-ordered integer Sinkhorn conversion, non-Float64 wrapper
   dispatch failure, nonpositive/nonfinite epsilon propagation, mixed
   Fraction/Float `otPhase2`, exact
   zero-iteration invalid propagation, no Python-only deprecation warnings;
   ComplexF32 IFT boundary failures; Real template dispatch; and unavailable
   upstream beam saving.

Dense-OT geometry tests compare the signed-maximum formula for source and
target lengths one through eight, including unequal even sizes and the
authority's zero-scale `2→N` cases. The undefined target-length-two
denominator is verified to remain invalid; no radius fallback or finite
`N→2` plan is invented. Singleton automatic geometry likewise remains a
failing NaN/Inf path.

Arithmetic tests treat each Python chain as an eager binary left fold and
compare it with the corresponding explicitly parenthesized Julia binary
expression. They also verify non-mutation and value semantics: equal visible
intermediates remain interchangeable after copying or reconstruction, with no
hidden operands or expression-history state.

All generated data uses the test runner's temporary directory.  Large image
fixtures in the Julia checkout are optional, read-only integration inputs and
are never copied into this port.

The audited manifest identifies Julia 1.11.6; the local differential run used
Julia 1.12.6 with that same project/manifest and compiled modules disabled.
The exact Python package versions are in `requirements-lock.txt`.

## Commands

```bash
python -m pytest
python -m pytest -q tests/test_reference_parity.py
```

To enable optional read-only fixture checks:

```bash
SLMTOOLS_JULIA_REPO=/absolute/path/to/SLMTools python -m pytest
```

Before and after differential testing, record both `git status --short` and a
SHA-256 aggregate of every non-`.git` file in the Julia checkout.  Differential
Julia processes run from a temporary working directory with compiled modules
disabled so neither precompile caches nor generated outputs land in the
original repository.  Acceptance requires identical before/after status and
hash, plus absence of `.git` anywhere in this Python folder.
