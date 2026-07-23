# API map

All 100 names exported by the audited Julia module are available directly from
`slmtools`.  Camel-case spelling is canonical.  Python overloads inspect
argument types and tags at runtime; optional keyword names retain their Julia
spelling where Python syntax permits it.

## Field types and operations

| Julia export | Python form | Contract |
|---|---|---|
| `Lattice` | `Lattice` | Type alias for a tuple of immutable regular `LatticeAxis` values |
| `FieldVal` | `FieldVal` | Root semantic tag |
| `Generic` | `Generic` | Untyped field values |
| `Phase` | `Phase` | Phase-tag base |
| `RealPhase` | `RealPhase` | Unwrapped phase in cycles |
| `UPhase` | `UPhase` | Exact alias of `RealPhase` |
| `UnwrappedPhase` | `UnwrappedPhase` | Exact alias of `RealPhase` |
| `ComplexPhase` | `ComplexPhase` | Complex phasor tag |
| `S1Phase` | `S1Phase` | Exact alias of `ComplexPhase` |
| `Intensity` | `Intensity` | Intensity tag; partial construction clips negatives |
| `Amplitude` | `Amplitude` | Amplitude-tag base |
| `Modulus` | `Modulus` | Real-amplitude tag |
| `RealAmplitude` | `RealAmplitude` | Exact alias of `Modulus` |
| `RealAmp` | `RealAmp` | Exact alias of `Modulus` |
| `ComplexAmplitude` | `ComplexAmplitude` | Complex-amplitude tag |
| `ComplexAmp` | `ComplexAmp` | Exact alias of `ComplexAmplitude` |
| `LatticeField` | `LatticeField(data, L, flambda=1, field_type=...)` | Tagged array and sampling metadata; partial Intensity/ComplexAmplitude conversions apply |
| `LF` | `LF`, `LF[Tag](...)`, or `LF[Tag, dtype, ndim](...)` | Exact class alias, partial tagged constructor, and public full typed constructor; the full form requires exact dtype/dimension and bypasses clipping/promotion |
| `elq` | `elq(x, y)` | Validate each complete axis with Julia's norm-based array `isapprox` and, for fields, approximately equal `flambda`; hidden singleton steps are not equality data; returns `None` |
| `subfield` | `subfield(field, *indices)` | Julia vararg behavior with zero-based indices: no indices errors, one integer is a Fortran-linear scalar index, and range/mixed Cartesian forms preserve remaining metadata |
| `sublattice` | `sublattice(L, *indices)` or `sublattice(L, (ranges...))` | Coordinate subset without field data, including Julia's single Cartesian-box spelling |
| `wrap` | `wrap(field)` | Real cycles to complex phase; complex phase remains complex phase |
| `square` | `square(amplitude)` | Elementwise magnitude squared as `Intensity` |
| `normalizeLF` | `normalizeLF(field)` | Unit-sum intensity or unit-L2 amplitude |
| `phasor` | `phasor(z_or_field)` | Exact Julia dispatch: scalar input and ComplexAmplitude storage must be ComplexF64/complex128; complex zero maps to one |

`LatticeField` implements `shape`, `ndim`, `dtype`, `len`, item access and
assignment, NumPy `sqrt`/`abs`/`conjugate`, and the valid Julia field/scalar
arithmetic combinations.  Unsupported semantic combinations raise
`TypeError` instead of silently dropping tags. A bare integer is Fortran-linear
indexing. Boolean indices are rejected. Integer and mixed integer/range access
accepts omitted trailing selectors only when Julia's dense indexing can fill
singleton dimensions, and accepts zero-valued integer indices into implicit
trailing singleton dimensions. The all-range overload still requires exact
field arity, as Julia's `sublattice` call does. Assignment is narrower: it
accepts one Fortran-linear integer or exactly one integer per field dimension,
not slices or omitted singleton selectors. It validates Julia-style conversion
before mutation, so fractional or genuinely complex values cannot silently
truncate into integer/real fields. Fraction/Decimal values may be approximately
converted to a floating or complex destination, just as Rational/BigFloat
values are in Julia. Unary `+` is defined only for `Intensity` and
`ComplexAmplitude`, matching the Julia vararg additions.

`LatticeField.__add__` and `LatticeField.__mul__` are eager, binary,
non-mutating, and value-semantic. Python evaluates `a+b+c` as the left fold
`(a+b)+c`, so chains are verified against Julia's explicitly parenthesized
binary form, including intermediate Intensity clipping and intermediate
ComplexAmplitude promotion. Copying, reconstructing, or assigning a visible
intermediate cannot change later arithmetic: no hidden operands are retained,
no source or bytecode is inspected, and no `julia_add`/`julia_mul` helper API
is introduced. When a package-internal algorithm genuinely requires
aggregate-before-construction semantics, it validates every field, reduces
the underlying arrays locally with Julia-compatible dtype and order, and
constructs one result afterward. Julia instead parses an unparenthesized
n-ary expression as one call, clips a complete n-ary Intensity sum once,
promotes ComplexAmplitude after its n-ary reduction, and routes unsupported
n-ary tag combinations to its fallback error; those parser-level behaviors
have no transparent Python infix spelling.
Before a machine-number operation, both arrays are converted with Julia's
promotion table rather than NumPy's: this includes Bool arithmetic,
equal-width signed/unsigned wrapping, integer-plus-Float16/32 rounding, and
ComplexF32 retention. Object-backed Rational and BigFloat-like fields preserve
their concrete Fraction/Decimal element domain on assignment. Exact values are
rounded directly to a Float16/32/64 destination (without a binary64
intermediate), while string-to-number, number-to-string, and ndarray-scalar
setter conversions that have no Julia method are rejected.
`copy.copy(field)` and `field.copy()` invoke the partial tagged constructor,
matching Julia even when that changes unusual full-typed data.
Built-in Python `float` and `complex` have the same strong Float64/ComplexF64
dispatch effect as `np.float64` and `np.complex128`; they are not NumPy weak
scalars for field arithmetic.

`LatticeAxis.from_start_step(start, step, length)` is the qualified exact
constructor for Julia machine-integer ranges and Float16/Float32/Float64
`StepRangeLen` axes. It preserves the literal start long enough to reproduce
Julia's logical reference, step, and offset—including Base's Float64
twice-precision representation and rare low-precision ranges whose first
materialized coordinate differs from the supplied start. Ordinary
`LatticeAxis(values, step_hint=...)` remains appropriate when the original
logical start is not separately available.
Mixed machine-integer construction also retains Julia's concrete range form:
all non-UInt64 combinations materialize Int64 `StepRange` values while
preserving the supplied step dtype; any UInt64 operand selects the wrapped
UInt64 `StepRangeLen` path.
Field/sublattice slices and range padding/downsampling keep this metadata;
upsampling follows Julia's source-precision step division before its Float64
centering. A true axis copy remains immutable with the same metadata, while a
dtype-changing cast or reinterpretation materializes a read-only plain array
so stale range metadata cannot survive. NumPy's special
`np.array(axis, copy=True, subok=True)` path must fill a writable subclass
instance; that copy therefore drops all logical metadata and is validated from
its actual values if later reused as a lattice.
`fractions.Fraction` supports exact Rational lattice/basic-algebra,
interpolation, OT-integration, helper, and template paths; operations such as
`nabs`, intensity `sqrt`, and FFT follow Julia by returning Float64 or
ComplexF64. `decimal.Decimal` supports context-sized BigFloat-like metadata,
interpolation, helpers, OT integration, linear fitting/orientation, dualation,
analytic templates, and phase wrapping. Wrapped
Decimal phases use object scalars with Decimal `real` and
`imag` components, the Python counterpart of `Complex{BigFloat}`. Decimal
arrays are not a general NumPy computational dtype; see the explicit numeric
boundary in `COMPATIBILITY.md`.

## Lattices, resampling, and transforms

| Julia export | Python form | Contract |
|---|---|---|
| `natrange` | `natrange(n)` | Centered coordinates divided by `sqrt(n)`; `n=0` is empty with Julia's retained `Inf` logical step |
| `natlat` | `natlat(shape)` or `natlat(*shape)` | Natural lattice |
| `naturalize` | `naturalize(field)` | Replace lattice with natural coordinates and reset `flambda=1` |
| `padout` | `padout(axis_or_lattice_or_array_or_field, padding, filler=None)` | Extend consistently; padding counts require Julia-compatible signed machine integers, negative axis/lattice padding performs Julia range cropping, dense negative padding retains Julia's bounds failure, and filler-typed lossy assignment raises `ValueError` (Julia `InexactError`) |
| `latticeDisplacement` | `latticeDisplacement(L)` | Coordinate at each FFT-shifted center index |
| `toDim` | `toDim(values, dimension, ndim)` | Flatten input in Fortran order, then reshape for broadcasting; dimension accepts Julia-style `1..N` |
| `r2` | `r2(L)` | Grid of squared radius |
| `ldot` | `ldot(vector, L)` or `ldot(L, vector)` | Grid of coordinate dot products; integer, Float16/Float32, and Float64 range products use their retained Julia range arithmetic before reshaping |
| `Nyquist` | `Nyquist(L)` | Per-axis `1/(2*step)` |
| `downsample` | `downsample(...)` | Block-center axes or natural-cubic resampling; signed lattice geometry and Float32 factor-generated axes are preserved, and factors must be positive signed-Int64 Python `int`/NumPy `int64` values |
| `coarsen` | `coarsen(data_or_field, factors, reducer=...)` | Disjoint block reduction; empty outputs retain the reducer result dtype rather than defaulting to Float64 |
| `upsample` | `upsample(...)` | Center-preserving fine axes or natural-cubic resampling with positive signed-Int64 Python `int`/NumPy `int64` factors; signed lattice geometry is supported and Julia's `(1+n)/2` term widens a factor-generated Float32 axis to Float64 |
| `dualLattice` | `dualLattice(L, flambda=1)` | Unshifted nonnegative dual coordinates with retained low-precision logical range arithmetic |
| `dualShiftLattice` | `dualShiftLattice(L, flambda=1)` | FFT-shifted dual coordinates with retained low-precision logical range arithmetic |
| `dualPhase` | `dualPhase(L, flambda=1, dL=None)` | Translation phase on a dual lattice, including one-dimensional range-preserving scalar operations |
| `ldq` | `ldq(L1, L2, flambda=1)` or `ldq(field1, field2)` | Validate shifted-dual relationship; the field overload inherits wavelength and rejects an explicit third argument |
| `sft` | `sft(array_or_complex_amplitude)` | Shifted forward FFT over all axes; Rational/Fraction input follows FFTW's ComplexF64 conversion |
| `isft` | `isft(array_or_complex_amplitude)` | Shifted normalized inverse FFT; Rational/Fraction input follows FFTW's ComplexF64 conversion |

Qualified helpers retained in their source module are
`lattice_utils.shiftedDFTBasis`, `lattice_utils.hermiteBasis`,
`lattice_utils.FrFTBasis`, `lattice_utils.wigner_fft`, and
`resampling.cubic_spline_interpolation`.  The latter accepts extrapolation
modes `constant`, `flat`, `periodic`, `linear`, and `throw`.

As in Julia, useful non-exported definitions are also package-qualified:
`shiftedDFTBasis`, `hermiteBasis`, `FrFTBasis`, `wigner_fft`,
`cubic_spline_interpolation`, `CubicSplineInterpolation`,
`LinearInterpolation`, `Flat`, `Periodic`, `Linear`, `rampPrivate`, `heartQ`, `smileQ`,
`pointerOutlineQ`, `pointerFillQ`, `l2form`, `lfStandardOutputFormat`,
`gsError`, `hyperSum`, `cycle1`, and the dynamically
addressable `getattr(slmtools, "SinkhornIterBase!")`.  They do not enter
`from slmtools import *`, matching Julia's export declarations.

`shiftedDFTBasis` uses Julia/OpenLibm-compatible Float64 trigonometry.
`hermiteBasis` and `FrFTBasis` remain bound for name-level API completeness
but raise `NotImplementedError`: the former is unreachable in the audited
Julia source because it passes a coordinate range to an integer-only DFT
basis method, and the latter depends on it.

`Linear()` is Interpolations.jl's interpolation-degree marker, not an
endpoint-boundary type. `LinearInterpolation(...)` constructs the qualified
tensor-product piecewise-linear interpolator and accepts strictly increasing
nonuniform coordinate vectors in every dimension, matching `Gridded(Linear())`
without broadening SLMTools field lattices beyond regular ranges. The locked Interpolations.jl compatibility
conversion that accepts `Linear()` as an extrapolation argument is retained.
The qualified constructor surface also accepts `Linear(Periodic())`, like the
locked dependency. Explicit `Linear(Throw())` has no Julia method; no-argument
`Linear()` remains the dependency's default Throw(OnGrid) degree marker.
`CubicSplineInterpolation(..., bc=..., extrapolation_bc=...)` exposes the
locked constructor's independent spline and extrapolation boundaries. Spline
boundaries require the same explicit placement as Interpolations.jl:
`Flat(OnGrid())`, `Flat(OnCell())`, `Periodic(OnGrid())`, or
`Periodic(OnCell())`, using the non-root-qualified
`slmtools.resampling.OnGrid` and `slmtools.resampling.OnCell` markers. Bare
`Flat()` and `Periodic()` remain valid extrapolation policies but, like the
locked dependency, fail when passed as the spline `bc`.
Extrapolation tuples follow the locked dependency's dimensional nesting. A
direct one-dimensional `(Flat(), Periodic())` tuple supplies one boundary
object per dimension and therefore uses the first entry; a directional
left/right pair is nested as `((Flat(), Periodic()),)`. Numeric or other
tuples remain one tuple-valued filled-extrapolation result.
Interpolation also has an isolated arbitrary-precision path: object arrays of
`fractions.Fraction` coordinates and data retain exact Rational coefficients
and results for linear, natural, and `Flat(OnGrid())` splines, while
`decimal.Decimal` coordinates/data provide a context-sized BigFloat-like
calculation. OnCell placement introduces the dependency's Float64 half-cell
bounds, so machine-float and Rational periodic-OnCell evaluation returns the
same widened Float64 result as Julia.
Python-only `Throw` remains available as `slmtools.resampling.Throw`; the
package root exposes neither `Throw` nor a fabricated `Line` binding.

## General helpers

| Julia export | Python form | Contract |
|---|---|---|
| `ramp` | `ramp(x)` | Clamp negative real values to zero |
| `nabs` | `nabs(array)` | L2-normalized magnitude; Rational input follows Julia's Float64 square-root promotion and Decimal remains context-sized |
| `window` | `window(array_or_field, width)` | Centered Cartesian NumPy index tuple around the centroid, including exact-number centroids, ties-to-even rounding, and empty regions for zero/negative widths; integer index grids preserve Julia's later bounds failure instead of allowing slice clipping/wrapping |
| `safeInverse` | `safeInverse(x)` | Zero-preserving reciprocal |
| `centroid` | `centroid(ndarray_or_intensity, threshold=0.1)` | Weighted raw-pixel or `Intensity`-field centroid with a real scalar threshold; 0-D inputs return an empty coordinate vector |
| `collapse` | `collapse(array, dimension)` | Sum over all axes except one |
| `clip` | `clip(x, threshold)` | Zero values below threshold |
| `SchroffError` | `SchroffError(target, reality, threshold=0.5)` | Thresholded normalized error for two `Intensity` fields; preserves floating/exact promotion and rejects inexact integer normalization |

## Images, templates, display, and BMP

| Julia export | Python form | Contract |
|---|---|---|
| `getImagesAndFilenames` | `getImagesAndFilenames(directory, extension)` | Sorted exact-suffix image load with literal upstream path concatenation; a separator is needed only when matches are loaded, while a no-match scan returns two empty lists |
| `imageToFloatArray` | `imageToFloatArray(image)` | Colors.jl-compatible Rec.601 grayscale Float64 array, including normalized fixed-point intermediate precision and rounding |
| `itfa` | `itfa` | Exact alias of `imageToFloatArray` |
| `castImage` | `castImage(Tag, image, L, flambda)` | Convert image and attach field metadata |
| `loadDir` | `loadDir(directory, extension, ...)` | Load and convert matching images using the source's literal directory/filename concatenation; the later first-image access still fails for an empty result |
| `parseFileName` | `parseFileName(name, cue=None, look="after", outType=None)` | Extract and optionally parse filename metadata |
| `parseStringToNum` | `parseStringToNum(text, outType=None)` | Integer/float/Bool/Decimal parsing with Julia spellings and explicit override; Fraction accepts Julia Rational integer or slash syntax and rejects decimal literals |
| `getOrientation` | `getOrientation(images, indices, roi=None, threshold=0.1)` | Fit center and line-image orientation for Python-list/NumPy-vector counterparts of Julia's concrete vectors, retaining Decimal/BigFloat-like results |
| `dualate` | `dualate(field_or_fields, L, center, theta, flambda=1, ...)` | Rotate/resample camera fields onto a dual lattice, including singleton target axes and Decimal/BigFloat-like geometry; retained steps and Julia scalar/axis dtype promotion are preserved |
| `linearFit` | `linearFit(xs, ys)` | Slope/intercept least-squares fit on one-dimensional NumPy arrays or Python list vector literals; tuples/ranges and nonnumeric vectors do not dispatch, Rational/Fraction input follows Julia's Float64 promotion, and Decimal input retains context-sized arithmetic |
| `savePhase` | `savePhase(phase_or_array, name)` | Save wrapped phase as an 8-bit image; raw complex arrays and ComplexPhase field storage require complex128, matching `ComplexF64` dispatch |
| `saveBeam` | `saveBeam(beam, name, data=..., dir=...)` | Exported for API completeness but recognized output requests raise `NotImplementedError`, matching the unusable upstream implementation without inventing portable output semantics |
| `savePhase8BMP` | `savePhase8BMP(phase_or_array, name)` | Same ComplexF64 phase dispatch through the exact indexed-BMP writer |
| `lfParabola` | `lfParabola(Tag_or_pattern, ..., center=..., flambda=...)` | Quadratic form template with Julia `Real` scalar/matrix/linear-coefficient dispatch; oversized matrices use the leading dimensional block |
| `lfGaussian` | `lfGaussian(Tag_or_pattern, ..., center=..., flambda=...)` | Energy-normalized Gaussian template; oversized matrices use the leading dimensional block |
| `lfRing` | `lfRing(Tag_or_pattern, ..., center=..., flambda=...)` | Annular Gaussian template |
| `lfCap` | `lfCap(Tag_or_pattern, ..., center=..., flambda=...)` | Spherical-cap phase template |
| `ftaText` | `ftaText(text, size, ...)` | FreeType/Pillow grayscale raster text with locked ascender/baseline placement and integer-pixel pair kerning; the default `"arial bold"` resolves first to Arial Rounded MT Bold, and empty text with an explicit pixel size follows Julia's failure |
| `lfText` | `lfText(Tag_or_pattern, ..., center=..., flambda=...)` | Text template |
| `lfRect` | `lfRect(Tag_or_pattern, ..., center=..., flambda=...)` | Rectangular mask template |
| `lfRand` | `lfRand(Tag_or_pattern, ..., R=np.float64, center=..., flambda=...)` | Random template using NumPy's global RNG |
| `lfHeart` | `lfHeart(Tag_or_pattern, ..., center=..., flambda=...)` | Heart mask template |
| `lfSmile` | `lfSmile(Tag_or_pattern, ..., center=..., flambda=...)` | Smiley mask template |
| `lfPointer` | `lfPointer(Tag_or_pattern, ..., center=..., flambda=...)` | Pointer mask template |
| `lfBlur` | `lfBlur(field, radius)` | Circular shifted-FFT Gaussian blur with the source's real-radius dispatch |
| `look` | `look(*fields_or_arrays)` | Display-ready grayscale output with Julia's overloads: Rational arrays are valid, while multiple arguments must be lattice fields rather than arbitrary arrays |
| `save_gray8bmp` | `save_gray8bmp(path, image)` | Exact uncompressed indexed 8-bit grayscale BMP |

The field-pattern overloads of every `lf*` template inherit `L`, field tag,
and `flambda`; as in Julia, they do not accept an explicit `flambda` keyword.
Template coordinate and parameter arithmetic uses Julia's strong promotion
and retained Float16/Float32 `StepRangeLen` reference/step arithmetic before
materialization:
an all-explicit Float32 parabola/Gaussian/ring/cap remains Float32, the default
Float64 center or a Python-float parameter can widen it, and wrapping a real
Float32 template as `ComplexPhase` produces ComplexF64.
Fraction-backed Rational ranges follow the same promotion: the templates above
return Float64 with their default Float64 center, while exact intermediate
Rational arithmetic is retained until a Julia operation (such as `exp` or
`sqrt`) promotes it. Decimal-backed BigFloat-like ranges and parameters retain
context-sized Decimal results for `lfParabola`, `lfGaussian`, `lfRing`, and
`lfCap`; ComplexPhase templates retain Decimal real/imaginary components, and
`lfRect` deliberately allocates Float64 just like Julia.
`lfRand(..., R=Decimal)` follows the active Decimal precision, while
`R=Fraction` rejects
because Julia has no Rational random sampler.

## Iterative Fourier-transform retrieval

| Julia export | Python form | Contract |
|---|---|---|
| `gs` | `gs(input, target, iterations, phase=None, rng=None)` | Gerchberg–Saxton retrieval for homogeneous real Modulus/Modulus or Intensity/Intensity pairs; Intensity inputs require an explicit phase because the upstream `nothing` dispatch is broken; positive Float32 iterations retain the upstream Float64-only `gsIter` failure; negative counts perform no updates |
| `gsIter` | `gsIter(guess, input_modulus, target_modulus, ft=None, ift=None)` | One raw-array GS update; dense `complex128` guess and `float64` modulus arrays are required exactly, matching Julia's concrete element-type dispatch |
| `gsLog` | `gsLog(..., every=1)` | GS result plus sampled error history; Intensity inputs require an explicit phase, while valid Float16/Float32 Modulus paths retain Julia's low-precision reductions; zero cadence fails only when an iteration evaluates modulo and negative cadence is valid |
| `pdgs` | `pdgs(images, diversity_phases, iterations, beam_guess)` | Phase-diversity GS estimate; image tuple tags must be homogeneous, low-precision dual-phase ranges are retained, and negative counts perform no updates |
| `pdgsIter` | `pdgsIter(guess, phasors, moduli, ft=None, ift=None)` | One raw-array PDGS update; requires a dense `complex128` guess, tuples of dense `complex128` phasors and `float64` moduli |
| `pdgsLog` | `pdgsLog(..., every=1)` | Modulus-image PDGS result plus branch-disagreement history with Julia's zero/negative cadence timing |
| `pdgsError` | `pdgsError(moduli, diversity_phases, beam_guess)` | Modulus-image mean normalized branch error |
| `oneShot` | `oneShot(image, alpha, beta)` | Analytic one-image beam estimate; `alpha` and every `beta` value must be real, and Float16/Float32 lattice, wavelength, square-root, and phase arithmetic are retained while FFT output is ComplexF64 |
| `mraf` | `mraf(input_modulus, target_modulus, iterations, phase, roi, m)` | Modulus-only mixed-region amplitude freedom retrieval |

The Julia-qualified but unexported `gsError` is available as
`slmtools.gsError`. Complex Modulus data follows the Julia method boundaries:
`pdgs` and `pdgsLog` fail their Float64 work-array requirement, while the
numeric `pdgsError` overload remains valid. Negative counts in GS, PDGS, and
MRAF families execute no updates. A phase overload declared as `<:Number`
preserves complex-valued `RealPhase` data; the semantic tag does not authorize
a lossy cast to real. Full-typed ComplexF32 ComplexPhase/beam storage is not
silently widened across Julia's ComplexF64-only phasor and iterative FFT
boundaries. Rational Modulus `gs` preserves Julia's split behavior:
an empty loop is valid, positive iterations have no matching Float64-only
`gsIter`, while the separately implemented `gsLog` promotes through Julia's
Rational square-root/ComplexF64 plan arithmetic and remains valid.
The raw iteration helpers retain Python's convenient omitted-plan form because
the pyFFTW NumPy-compatible interface needs no explicit plan object; supplying callables applies them in
the same forward/inverse positions. This binding convention does not relax
their concrete Julia array element-type contracts.

## Optimal transport

| Julia export | Python form | Contract |
|---|---|---|
| `getCostMatrix` | `getCostMatrix(L_mu, L_v=None, normalization=np.max)` | Dense normalized squared-distance cost, retaining Rational/BigFloat-like arithmetic; default normalization of an all-zero singleton cost retains Julia's NaN, and empty range domains reach the supplied normalization (the default `np.max` consequently raises) |
| `pdCostMatrix` | `pdCostMatrix(...)` | Documentation-deprecated real-parameter phase-diversity cost with Julia numeric promotion; no Python-only runtime warning |
| `mapify` | `mapify(plan, L_mu, L_v)` | Barycentric target-coordinate map; output is Float64 because Julia allocates its destination with `zeros(...)`, a genuinely complex barycenter raises instead of discarding its imaginary part, and empty source/target ranges retain Julia's empty/zero map results |
| `hyperSum2` | `hyperSum2(A, origin, sumDim, fixDims)` | Anchored trapezoidal cumulative sum; dimension and origin indices require Julia-compatible signed `Int64` values |
| `scalarPotentialN` | `scalarPotentialN(vector_field, L, idx=None, dimOrder=None)` | Path-integrated scalar potential, retaining Rational/BigFloat-like arithmetic and exact signed-`Int64` index dispatch; the upstream default anchor is invalid for a singleton axis, though an explicit valid origin works |
| `normalizeDistribution` | `normalizeDistribution(array)` | Unit-sum magnitude distribution with exact-number preservation; zero total mass retains Julia's NaN result |
| `otPhase` | `otPhase(input, target, epsilon, **sinkhorn_options)` | Dense Gibbs-Sinkhorn phase, including Decimal/BigFloat-like cache, lattice, and wavelength promotion; automatic geometry retains Julia's signed-maximum scaling and its all-NaN two-sample-target range; `maxiter` and convergence cadence require signed Int64 counterparts, while nonpositive/nonfinite epsilon and zero wavelength are not defensively rejected |
| `pdotPhase` | `pdotPhase(...)` | Dense phase-diversity OT phase using the same upstream automatic geometry and real-only parameter dispatch; equal alphas propagate invalid division, beta tuples require exactly N entries, and beta lists/1-D arrays may have ignored trailing entries |
| `pdotBeamEstimate` | `pdotBeamEstimate(..., LFine=None, **options)` | OT beam estimate preserving Decimal/BigFloat-like metadata and valid computation; `LFine=None`, integer `UnitRange`, and signed-integer `StepRange` counterparts work, while the upstream-broken floating/Rational/BigFloat-like `LFine` interpolation path is explicitly unsupported |
| `SinkhornConvN` | `SinkhornConvN(U, V, epsilon, max_iter, every=None)` | Experimental convolutional scaling solver; positive non-Float64 targets retain the upstream work-array dispatch failure, Decimal/BigFloat-like FFT work is unsupported as in Julia FFTW, while zero/NaN epsilon yields NaNs, negative epsilon executes, and infinite epsilon returns the input marginals |
| `dualToGradients` | `dualToGradients(u, v, U, L_v, epsilon)` | Convolutional dual-to-gradient map; Decimal/BigFloat-like FFT work retains Julia's unsupported-type failure |
| `otQuickPhase` | `otQuickPhase(..., return_loss=False)` | Documentation-deprecated homogeneous-element-type convolutional phase wrapper; Decimal/BigFloat-like FFT work retains Julia's failure and there is no Python-only runtime warning |
| `otPhase2` | `otPhase2(input, target, epsilon, iterations, **options)` | Factorized 2-D Sinkhorn phase for square inputs and complex numeric targets, with Decimal/BigFloat-like real data/axis/wavelength/epsilon promotion; rectangular input is explicitly unsupported, the upstream ineffective target-wavelength check and final `0/0` propagation are retained, `iterations` follows exact 64-bit `Int` dispatch, and arbitrary options are ignored |

The internal helpers are `ot.hyperSum`, `ot._sinkhorn_gibbs`, and
`ot._sinkhorn_iter_base_inplace`. The latter, `ot.SinkhornIterBase`,
`getattr(ot, "SinkhornIterBase!")`, and
`getattr(slmtools, "SinkhornIterBase!")` mutate both `u` and `v` and return
the same `u` object, matching Julia's final expression. They require dense
arrays, the same real element dtype for `u` and `v`, and the same numeric
element dtype for the two Fourier kernels; mixed work arrays do not dispatch.
Integer scalings are valid when their individual assignments convert exactly.
Updates follow Julia's column-major element order as well as statement order:
if conversion fails, earlier elements of that same statement remain mutated;
if a later `u` statement fails, all earlier `v` mutations remain visible.

For dense `otPhase` and `pdotPhase`, each automatically generated target axis
uses Julia's exact
`scale = maximum(source_axis) / maximum(target_axis)`. This preserves all
working odd, even, equal, and unequal geometries. A two-sample target natural
axis has maximum zero, so Julia's range broadcast produces an all-NaN axis and
the public calculation reaches `sinkhorn returned nan` rather than
substituting a radius convention. A two-sample
source with a longer target remains on Julia's signed rule, including its
zero scale. Singleton axes are likewise not redefined as valid. These source
semantics are shared by both public functions through
`ot._ot_natural_lattices`; `natlat(1)` and generic `getCostMatrix` are
unchanged.

## Optional `subimages` API

The non-included Julia notebook helper is ported as `slmtools.subimages` with
`ftaText`, `plotToImage`, `imageToHeatmap`, `grayAnnotation`, `colorPromote`,
`padout`, `padadd`, `padmultiple`, `padCommon`, `trimWhitespace`, `arrange`,
`checkCommonSize`, `mergeStrict`, `mergeFill`, `autoAnnotate`, and
`handAnnotate`. `plotToImage` returns RGBA floating values in `[0, 1]`, the
numeric semantics of Julia fixed-point colorants, so its output can be passed
directly to `trimWhitespace`. Whitespace comparison is exact, and the cropped
array is an independent copy like Julia range indexing. Padding validates
assignment conversion first, so fractional fillers cannot silently truncate
into integer image arrays. Optional negative `padmultiple` widths are no-ops,
matching Julia's positive-width guards. A positive `padall` reproduces the
upstream mistake: it repeats each positive directional width once, and is a
no-op when those widths are all zero.
