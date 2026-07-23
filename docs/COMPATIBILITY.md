# Julia compatibility contract

## Audited baseline

The reference implementation is the local Julia repository at commit
`ea1c1c9c06b4b2dc46372ac7ee031301b604a007`, package version 0.3.0.  Its
manifest was generated with Julia 1.11.6 and pins FFTW 1.9.0,
Interpolations 0.16.2, OptimalTransport 0.3.20, ToeplitzMatrices 0.8.5,
Images 0.26.2, and FreeTypeAbstraction 0.10.8.

The Python implementation mirrors the actual flat `SLMTools` module.  The
namespaces used by the Julia test suite (`SLMTools.LatticeTools`,
`SLMTools.LatticeCore`, and similar) no longer exist in the audited Julia
source and are not recreated.

## Preserved conventions

- A lattice is an N-tuple of regular one-dimensional coordinates; tuple axis
  `i` describes data dimension `i`.
- `LatticeAxis` retains the Julia range representation behind its materialized
  coordinates: exact integer ordinal ranges, Float16/Float32 `StepRangeLen`
  axes with Float64 reference/step, and Float64 `StepRangeLen` axes with
  Base-compatible twice-precision reference/step pairs.
  Range-preserving centering/scaling is used by templates, `ldot`, dual
  lattices, `dualPhase`, PDGS, field/sublattice slicing, padding, and factor
  resampling before coordinates are materialized. Use
  `LatticeAxis.from_start_step(start, step, length)` whenever the logical
  inputs are available; it implements Julia's machine-integer and
  machine-float range construction, including rare cases where the first
  materialized coordinate differs from the literal start.
- Singleton axes retain their logical range step in `LatticeAxis`; transforms
  use that metadata even though one coordinate cannot encode it. Equality,
  however, compares only materialized coordinates, so an unobservable hidden
  singleton step cannot create a mismatch.
- Each complete coordinate axis is compared with Julia's array-level,
  norm-based `isapprox`; axes are not compared coordinate by coordinate.
- `RealPhase` is in cycles. Wrapping is `exp(2πi*phase)`. The tag does not
  force real storage: overloads declared with `<:Number` preserve a valid
  complex-valued `RealPhase` instead of discarding its imaginary component.
- `ComplexPhase` is a semantic tag; construction does not force unit modulus.
- `ComplexAmplitude` partial construction coerces data to complex128.
- Intensity partial construction clips negative real samples to zero.
- The full typed constructor is spelled `LF[Tag, dtype, ndim](...)` and bypasses
  those partial-constructor conversions, like Julia's `LF{S,T,N}`. `copy`
  deliberately re-enters the partial constructor, so copying can clip an
  unusual full-typed Intensity or promote a narrow ComplexAmplitude.
- Scalar `phasor` has Julia's exact `ComplexF64` domain. Thus
  `phasor(0.0 + 0.0j)` is `1+0j`, while integer or real `phasor(0)` has no
  matching method.
- `lfGaussian` enforces `sum(p**2) * product(axis_step) == norm`, not peak or sum normalization.
- `lfBlur` is circular convolution using the shifted FFT convention.
- `sft(x)` is `fftshift(fftn(ifftshift(x)))`; `isft` uses NumPy's normalized
  inverse.  Both operate across every dimension.
- Dual lattices depend on sample count, step, and `flambda`, not on source
  origin.  A double field transform therefore does not recover a translated
  input lattice even though it recovers the samples.
- OT Cartesian point enumeration, vectorization, and reshaping are Fortran
  order, matching Julia's column-major `[:]` behavior.
- Dense `otPhase`/`pdotPhase` automatic geometry retains Julia's
  `maximum(source_axis)/maximum(target_axis)` scaling, including its invalid
  zero denominator for a two-sample target axis.
- IFT routines keep Julia's homogeneous dispatch: GS pairs are both Modulus
  or both Intensity, PDGS image tuples are homogeneous, and MRAF is Modulus-only.
- Negative IFT and factorized/convolutional OT iteration counts execute empty
  loops, just as Julia's corresponding `1:n` ranges do.
- Python `float`/`complex` operands model Julia `Float64`/`ComplexF64` as
  strong scalar types. They therefore widen—or fail a typed field constructor
  after widening—in the same cases as explicit NumPy 64-bit scalars.
- The default `scalarPotentialN` anchor reproduces Julia's `length÷2`
  one-based index on axes of length at least two. In zero-based Python it is
  `n//2 - 1`, including that counterintuitive off-center choice. A singleton
  default anchor remains invalid, as it is upstream.
- Raw-array centroids use one-based pixel coordinates and a relative
  threshold.  Lattice-field centroids use physical lattice coordinates and
  an absolute threshold. Coordinate multiplication and reduction retain
  Julia's element type, including Float32 rounding and exact Fraction paths.
- `itfa is imageToFloatArray` is true.
- `savePhase8BMP` does not append an extension.
- `look` converts values to display arrays; it does not plot them.

Python slicing remains zero-based and half-open.  This is the sole pervasive
syntax difference.  Single-integer linear access on `LatticeField` is
explicitly Fortran ordered. Boolean selectors are rejected. Integer and mixed
integer/range access supplies omitted trailing selectors only where Julia's
dense indexing accepts trailing singleton dimensions, and zero-valued extra
integers address implicit singleton dimensions; omitted nonsingleton dimensions
still fail. The all-range overload retains exact field arity. Assignment
accepts only one linear integer or exactly one integer per field dimension,
matching Julia's narrower `setindex!` methods. Python `slice` and index tuples
replace Julia ranges and `CartesianIndices`.

## Dependency mappings

| Julia behavior | Python implementation |
|---|---|
| FFTW shifted transforms | pyFFTW's NumPy-compatible FFTW interface with the same shift and normalization order; maintained FFTW builds can select different codelets and differ by a few ULP in low-precision transforms |
| Interpolations cubic B-spline | Internal separable evaluator with O(n) tridiagonal/cyclic solves, retained logical range steps, bit-checked Float16 coefficients, Float64 coefficients for integral data, Float32/Complex64 promotion, explicit `OnGrid`/`OnCell` placement for Flat/Periodic spline boundaries, independent extrapolation boundaries, and no dense matrices; bare Flat/Periodic spline boundaries reproduce the dependency's missing-placement failure |
| Interpolations `LinearInterpolation` / `Linear` | Qualified tensor-product constructor supports strictly increasing nonuniform knot vectors; direct boundary tuples are per-dimension and nested tuples express directional pairs, while ordinary tuples remain tuple-valued fills; the locked `Linear()`-as-extrapolation compatibility translation is retained |
| OptimalTransport Gibbs Sinkhorn | Internal loop with the 0.3.20 initialization, update order, marginal check cadence, and tolerances; Decimal object arrays retain the active arbitrary-precision context when Julia promotes its cache to BigFloat |
| ToeplitzMatrices in `wigner_fft` | Direct Toeplitz construction with NumPy |
| Images/ColorTypes grayscale | Locked Colors.jl Rec.601 arithmetic, including raw fixed-point channel accumulation and Float32 `0.001` before RGB8/RGB16 rounding; alpha is discarded |
| FreeTypeAbstraction text | Pillow's FreeType binding with the locked baseline/ascender placement and integer-pixel pair kerning; default `"arial bold"` resolves to Arial Rounded MT Bold where the locked catalog font exists, then deterministic documented fallbacks |

The optional notebook-only `SubImages.jl` helpers live in
`slmtools.subimages`.  Plot conversion accepts Matplotlib figures, and
`imageToHeatmap` requires the optional `plot` dependency.

`fractions.Fraction` is the exact Python counterpart used for Julia Rational
coordinates, interpolation, OT integration, and algebraic helper paths.
Julia's inexact operations retain their actual promotion: Rational `nabs` and
intensity square root return Float64, and Rational shifted FFT input becomes
ComplexF64. Assigning Rational/BigFloat-like values to machine floating or
complex fields performs Julia's ordinary approximate conversion; exactness is
required only when the destination is integral or would discard a genuine
imaginary component. `decimal.Decimal` covers working BigFloat-like metadata,
interpolation, centroid/window/error helpers, lattice displacement, OT
integration, dense `otPhase`, square `otPhase2`, `pdotPhase`, the working
`pdotBeamEstimate` paths, linear fitting/orientation, dualation, analytic template kernels,
precision-sized random templates, and phase
wrapping. Wrapped phases use object scalars with Decimal `real` and
`imag` components, corresponding to Julia `Complex{BigFloat}`. The template path mirrors
Julia promotion rather than treating every object array alike: a default
Float64 center promotes Rational coordinates/results to Float64, Decimal
coordinates retain BigFloat-like arithmetic, Rational `exp`/`sqrt` return
Float64, and `lfRect` always allocates Float64. The pyFFTW/FFTW, Pillow/NumPy
image, and NumPy LAPACK backends do not provide arbitrary-precision array kernels, so Decimal/BigFloat
data arrays are not represented as a general computational dtype outside the
explicit paths above. This boundary is explicit rather than silently coercing
exact values to Float64. In particular, convolutional
`SinkhornConvN`/`dualToGradients`/`otQuickPhase` reject Decimal work just as
the Julia calls fail when BigFloat reaches FFTW.

## Known upstream defects and future work

This port does not turn a broken or unfinished Julia path into a new working
algorithm. Valid Julia behavior is preserved, including surprising numerical
behavior. Where an incidental Julia exception has no useful Python analogue,
the port raises a clear `NotImplementedError`, `DomainError`, `ValueError`, or
`TypeError` instead; it never fabricates a successful result.

| Upstream Julia issue | Python exposure (not a repair) |
|---|---|
| Intensity `gs(..., phase=nothing)` and `gsLog` dispatch to a nonexistent four-argument Modulus method | Intensity inputs with `phase=None` raise `TypeError`; supplying an explicit phase retains the working overload |
| Positive-iteration Float32 `gs` reaches the Float64-only `gsIter` method | Positive iterations raise `TypeError`; the valid zero-iteration path and the independent low-precision `gsLog` implementation remain available |
| `hermiteBasis(n)` passes a range to `shiftedDFTBasis(::Integer)`, making `FrFTBasis` unreachable too | Both helpers remain bound but raise `NotImplementedError`; no corrected eigensystem is invented |
| `otPhase2` allocates square `(n,n)` scalings, uses `n` in the y kernel, and compares the source wavelength with itself | Rectangular input is explicitly unsupported; square input retains the source formula and the ineffective target-wavelength check |
| `otPhase2` computes final `u/a` at zero-support pixels | Ordinary IEEE `0/0` propagation is retained |
| Dense OT divides by the zero maximum of an automatic two-sample target axis | Julia's range broadcast produces an all-NaN target axis; that exact invalid geometry reaches the public `sinkhorn returned nan` failure, with no radius fallback |
| Default `scalarPotentialN` requests Julia index zero on a singleton axis | The default singleton call raises; an explicitly supplied valid origin remains usable |
| `saveBeam` is unusable because required names are not imported; it also uses a Windows-only separator and writes the positive phase twice | Any recognized output request raises `NotImplementedError`; the export remains present for API completeness |
| Invalid `parseFileName(..., look=...)` tries to construct an undefined Julia `ValueError` | Invalid directions still fail explicitly with Python `ValueError`; no successful behavior is added |
| Directory loading concatenates the directory and filename strings | Literal concatenation is retained when matching files exist; with no matches Julia performs no load and succeeds with empty results even when the directory string has no trailing separator |
| `pdotBeamEstimate(..., LFine=...)` reaches obsolete noninteger bracket evaluation for noninteger target range types | `UnitRange` and signed-integer `StepRange` counterparts execute the working interpolation overload; floating, Rational, and BigFloat-like target ranges raise `NotImplementedError`, matching the probed upstream stack-overflow boundary |
| `SubImages.padmultiple(..., padall=n)` ignores `n` and recursively reapplies the four directional widths | A positive `padall` repeats each positive directional pad once; with all directional widths zero it is a no-op |
| Singleton interpolation constructors fail internally | Construction is rejected directly with `ValueError`; no constant-extension algorithm is invented |
| Nonpositive resampling factors lead to incidental failures or nonsensical grids | They are explicitly unsupported with `DomainError` |
| Positive-iteration `SinkhornConvN` creates Float64 `u` and target-typed `v`, which cannot dispatch for non-Float64 targets | Positive non-Float64 targets raise `TypeError`; no work-array promotion is applied |

Other invalid edge behavior is likewise not repaired. In particular,
`normalizeDistribution(zeros(...))` and a singleton default-normalized
`getCostMatrix` return NaNs. `otPhase2` permits a numeric, including complex,
target Intensity while requiring a real source, and preserves a complex-valued
result even though Julia tags it `RealPhase`. Rational inputs cross the
algorithm's Float64 work-scaling boundary before iteration, so every mixed
Fraction/Float orientation returns a Float64 field rather than an object array.
`dualShiftLattice(..., flambda=UInt64(...))` also retains Julia Base's
unsigned wrap of negative frequency indices (including collapsed huge values
or Float16 infinities); the port does not invent a successful signed algorithm
for that broken source path.

Real-valued OT parameters retain the source domain rather than acquiring
defensive positivity/nonzero checks. Dense Sinkhorn accepts negative, zero,
infinite, and NaN epsilon values and exposes the corresponding finite,
nonconvergent, or NaN result. `otPhase` with zero `flambda` and `pdotPhase`
with equal alpha values likewise propagate division by zero. For
`pdotPhase`, tuple beta arguments model `NTuple{N,Real}` and require exactly
`N` entries; list and one-dimensional ndarray arguments model `Vector` and
may contain extra trailing entries, which the source computation does not
index.

`pdCostMatrix` and `otQuickPhase` remain callable. Their deprecation remains a
documentation marker, as in this Julia version; calling them does not invent a
runtime warning. `gsError`, which is defined and documented but omitted from
Julia's export list, is available as `slmtools.gsError`.

`SinkhornIterBase!` mutates both scaling arrays and returns `u`, the value of
Julia's final mutating expression. It is reachable with
`getattr(slmtools, "SinkhornIterBase!")`; Python-spellable aliases remain in
the `slmtools.ot` module. Its direct-call dispatch requires dense arrays,
matching `u`/`v` element types, and matching Fourier-kernel element types;
integer work arrays are supported when the corresponding assignment converts
exactly. Mutations follow Julia's column-major element order as well as its
statement order: each converted element is immediately visible, so an
`InexactError` can leave a partially updated `v` or `u`, and a late failure in
the `u` statements leaves the earlier `v` mutations visible.
Likewise, direct `gsIter` and `pdgsIter` calls require the Julia methods'
concrete ComplexF64/Float64 array types. `otPhase2` has no Python-only early
stopping or `return_loss` alternate return.

Julia parses an unparenthesized expression such as `a+b+c` as one n-ary call;
Python evaluates it as the binary left fold `(a+b)+c`. `LatticeField.__add__`
and `LatticeField.__mul__` are therefore eager, binary, non-mutating, and
value-semantic, and Python chains are verified against Julia's explicitly
parenthesized binary operations. Fields retain no hidden operands or
expression-history state; the implementation does not inspect source or
bytecode and exposes no `julia_add`/`julia_mul` workaround. A package-internal
algorithm that genuinely requires aggregate-before-construction semantics
must validate each field, reduce the arrays locally with Julia-compatible
dtype and order, and construct one result. Julia-only n-ary dispatch
(clip-once Intensity addition, promote-after-reduction ComplexAmplitude
addition, and fallback errors for other tag combinations) has no transparent
infix spelling in Python.

Logging cadences retain Julia's evaluation timing: `every=0` is harmless for
an empty loop but raises division-by-zero when modulo is first evaluated, and
negative integer cadences are accepted. `oneShot` and `pdotBeamEstimate`
preserve source Float16/Float32 square-root, scalar, lattice, wavelength, and
phase arithmetic instead of widening those valid paths prematurely. `oneShot`
also enforces Julia's real `alpha` and `beta` dispatch. ComplexF32 phase/beam
storage is not silently widened when Julia reaches a ComplexF64-only phasor,
FFT-plan, or raw-iteration boundary.

Positive signed-Int64-range Python `int`/NumPy `int64` resampling factors model
Julia's concrete 64-bit `Int` dispatch. Out-of-range Python integers, other
integral dtypes, Bool, and all floating values are
rejected at dispatch rather than being silently converted or truncated.
Interpolations.jl 0.16.2 itself rejects
descending interpolation ranges; signed steps remain supported for the
separate downsample/upsample lattice-geometry overloads.

## Legacy artifacts

`examples/beam-ground-truth-2024-1-15.jld2` embeds pre-refactor Julia type
names under `SLMTools.LatticeTools...`; current Julia also cannot deserialize
it as a current `LatticeField`.  General JLD2 object deserialization is not a
Python-port feature.  A consumer should convert that one artifact in Julia to
plain arrays plus lattice and `flambda` metadata, preserving Fortran order.

Notebook checkpoints are treated as historical records, not authoritative
goldens: they use Julia 1.10.x, some embed OptimalTransport 0.3.19, several use
unseeded randomness, and the full experiments take hours or require several
gigabytes.
