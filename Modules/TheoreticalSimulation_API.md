# TheoreticalSimulation — API

_Generated: 2025-10-01 14:33_

## `AtomParams` (binding)

```
AtomParams{T}
```

Container for atomic constants used in the SG/beamline calculations.

Fields

  * `name::Symbol` — species tag (e.g. `:K39`, `:Rb87`).
  * `R::T`         — van der Waals radius (m).
  * `μn::T`        — nuclear magneton (J/T).
  * `γn::T`        — nuclear gyromagnetic ratio (s⁻¹·T⁻¹).
  * `Ispin::T`     — nuclear spin quantum number `I`.
  * `Ahfs::T`      — hyperfine constant (Hz).
  * `M::T`         — atomic mass (kg).

Notes

  * Defined with `Base.@kwdef`, so you can construct with keywords: `AtomParams(; name=:K39, R=…, μn=…, …)`.

```
AtomParams(atom; T=Float64) -> AtomParams{T}
```

Build an `AtomParams` from the lookup `AtomicSpecies.atoms(atom)`. The lookup is expected to return a tuple/array where positions `(1,2,3,4,6,7)` correspond to `(R, μn, γn, Ispin, Ahfs, M)`. Values are converted to the element type `T` (default `Float64`) and `name` is set to `Symbol(atom)`.

Requirements

  * `AtomicSpecies` must be loaded and provide `atoms(::Any)`.


## `BeamEffusionParams` (function)

- `BeamEffusionParams(xx_furnace, zz_furnace, xx_slit, zz_slit, yy_FurnaceToSlit, T, p::AtomParams) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_Params.jl:93`

```
BeamEffusionParams(xx_furnace, zz_furnace, xx_slit, zz_slit, yy_FurnaceToSlit, T, p::AtomParams)
    -> EffusionParams
```

Compute effusive-beam parameters from furnace/slit geometry and temperature.

Definitions

  * `Δxz = (-xx_furnace/2, -zz_furnace/2) − (xx_slit/2, zz_slit/2)` (m).
  * `θvmax = 1.25 * atan(norm(Δxz), yy_FurnaceToSlit)` (rad) — geometric half-angle with a 1.25 fudge factor.
  * Returns `EffusionParams(sin(θvmax), α2)` with `α2 = kb*T/p.M`.

Arguments

  * `xx_furnace, zz_furnace` — furnace aperture size in x/z (m).
  * `xx_slit, zz_slit`       — slit aperture size in x/z (m).
  * `yy_FurnaceToSlit`       — furnace→slit separation (m).
  * `T`                      — furnace temperature (K).
  * `p::AtomParams`          — provides the mass via `p.M` (kg).

Assumptions

  * `kb` (Boltzmann constant) is defined in scope.
  * Units are SI (m, K, kg).

Returns

  * `EffusionParams` ready to use in velocity samplers.


## `CQD_EqOfMotion` (function)

- `CQD_EqOfMotion(t, Ix, μ, r0::Vector{Float64}, v0::Vector{Float64}, θe::Float64, θn::Float64, kx::Float64, p::AtomParams) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:6`

No documentation found for public symbol.

`Main.TheoreticalSimulation.CQD_EqOfMotion` is a `Function`.

```
# 1 method for generic function "CQD_EqOfMotion" from Main.TheoreticalSimulation:
 [1] CQD_EqOfMotion(t, Ix, μ, r0::Vector{Float64}, v0::Vector{Float64}, θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
     @ d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:6
```


## `CQD_EqOfMotion_z` (function)

- `CQD_EqOfMotion_z(t, Ix::Float64, μ::Float64, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}, θe::Float64, θn::Float64, kx::Float64, p::AtomParams) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:60`

No documentation found for public symbol.

`Main.TheoreticalSimulation.CQD_EqOfMotion_z` is a `Function`.

```
# 1 method for generic function "CQD_EqOfMotion_z" from Main.TheoreticalSimulation:
 [1] CQD_EqOfMotion_z(t, Ix::Float64, μ::Float64, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}, θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
     @ d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:60
```


## `CQD_Screen_position` (function)

- `CQD_Screen_position(Ix, μ::Float64, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}, θe::Float64, θn::Float64, kx::Float64, p::AtomParams) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:104`

No documentation found for public symbol.

`Main.TheoreticalSimulation.CQD_Screen_position` is a `Function`.

```
# 1 method for generic function "CQD_Screen_position" from Main.TheoreticalSimulation:
 [1] CQD_Screen_position(Ix, μ::Float64, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}, θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
     @ d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:104
```


## `CQD_Screen_velocity` (function)

- `CQD_Screen_velocity(Ix, μ::Float64, v0::AbstractVector{Float64}, θe::Float64, θn::Float64, kx::Float64, p::AtomParams) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:134`

No documentation found for public symbol.

`Main.TheoreticalSimulation.CQD_Screen_velocity` is a `Function`.

```
# 1 method for generic function "CQD_Screen_velocity" from Main.TheoreticalSimulation:
 [1] CQD_Screen_velocity(Ix, μ::Float64, v0::AbstractVector{Float64}, θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
     @ d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:134
```


## `EffusionParams` (type)

```
EffusionParams{T}
```

Container of precomputed beam-sampling parameters.

Fields

  * `sinθmax::Float64` — max sine of the polar angle (0 ≤ sinθmax ≤ 1)
  * `α2::Float64`      — speed scale `kB*T/M` (m²/s²)


## `FreedmanDiaconisBins` (function)

- `FreedmanDiaconisBins(data::AbstractVector{<:Real}) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation.jl:162`

```
FreedmanDiaconisBins(data::AbstractVector{<:Real}) -> Int
```

Return the optimal number of histogram bins for `data` using the **Freedman–Diaconis rule**:

```
bin_width = 2 * IQR / n^(1/3)
bins      = ceil( range / bin_width )
```

where:

  * `IQR` is the interquartile range (Q3 − Q1).
  * `n` is the number of samples.
  * `range` is `maximum(data) - minimum(data)`.

This rule balances resolution with statistical noise and is robust to outliers.

# Arguments

  * `data::AbstractVector{<:Real}`: 1D array of real numeric values.

# Returns

  * `Int`: Number of bins.

# Notes

  * If `IQR` is zero (e.g., all values identical), returns `1`.
  * Assumes `data` has at least one element.
  * Automatically promotes input to `Float64` for calculations.


## `QM_EqOfMotion` (function)

- `QM_EqOfMotion(t, Ix, f, mf, r0::Vector{Float64}, v0::Vector{Float64}, p::AtomParams) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:162`

No documentation found for public symbol.

`Main.TheoreticalSimulation.QM_EqOfMotion` is a `Function`.

```
# 1 method for generic function "QM_EqOfMotion" from Main.TheoreticalSimulation:
 [1] QM_EqOfMotion(t, Ix, f, mf, r0::Vector{Float64}, v0::Vector{Float64}, p::AtomParams)
     @ d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:162
```


## `QM_EqOfMotion_z` (function)

- `QM_EqOfMotion_z(t, Ix::Float64, f, mf, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}, p::AtomParams) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:207`

No documentation found for public symbol.

`Main.TheoreticalSimulation.QM_EqOfMotion_z` is a `Function`.

```
# 1 method for generic function "QM_EqOfMotion_z" from Main.TheoreticalSimulation:
 [1] QM_EqOfMotion_z(t, Ix::Float64, f, mf, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}, p::AtomParams)
     @ d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:207
```


## `QM_Screen_position` (function)

- `QM_Screen_position(Ix, f, mf, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}, p::AtomParams) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:230`

No documentation found for public symbol.

`Main.TheoreticalSimulation.QM_Screen_position` is a `Function`.

```
# 1 method for generic function "QM_Screen_position" from Main.TheoreticalSimulation:
 [1] QM_Screen_position(Ix, f, mf, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}, p::AtomParams)
     @ d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:230
```


## `QM_Screen_velocity` (function)

- `QM_Screen_velocity(Ix, f, mf, v0::AbstractVector{Float64}, p::AtomParams) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:251`

No documentation found for public symbol.

`Main.TheoreticalSimulation.QM_Screen_velocity` is a `Function`.

```
# 1 method for generic function "QM_Screen_velocity" from Main.TheoreticalSimulation:
 [1] QM_Screen_velocity(Ix, f, mf, v0::AbstractVector{Float64}, p::AtomParams)
     @ d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_EquationsOfMotion.jl:251
```


## `QM_analyze_profiles_to_dict` (function)

- `QM_analyze_profiles_to_dict(data::OrderedCollections.OrderedDict{Symbol, Any}, p::AtomParams; manifold, n_bins, width_mm, add_plot, plot_xrange, λ_raw, λ_smooth, mode) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_QMSpline.jl:404`

```
QM_analyze_profiles_to_dict(
    data::OrderedDict{Symbol,Any},
    p::AtomParams;
    manifold::Symbol = :F_bottom,
    n_bins::Tuple = (1, 4),
    width_mm::Float64 = 0.150,
    add_plot::Bool = false,
    plot_xrange::Symbol = :all,
    λ_raw::Float64 = 0.01,
    λ_smooth::Float64 = 1e-3,
    mode::Symbol = :probability
) -> OrderedDict{Int, OrderedDict{Symbol,Any}}
```

Batch-process screen-hit datasets across multiple coil currents by repeatedly calling [`analyze_screen_profile`](@ref). For each current `Ix = data[:Icoils][i]`, this function extracts the (x, z) hit positions from `data[:data][i]`, converts them from **meters** to **millimeters**, analyzes the vertical z-profile, and stores the peak summaries in a nested dictionary keyed by the dataset index `i`.

# Inputs

  * `data::OrderedDict{Symbol,Any}`: Must contain:

      * `:Icoils :: AbstractVector{<:Real}` — coil currents (A), length `N`.
      * `:data` — a length-`N` container; each `data[:data][i]` is an indexable collection of matrices whose **columns 7:8** are `[x, z]` in **meters**.
  * `p::AtomParams`: Used to determine level grouping via `p.Ispin` when selecting manifolds.

# Keyword Arguments

  * `manifold::Symbol = :F_bottom`: Which manifold(s) to aggregate:

      * `:F_top`    → vertically concatenate levels `1 : (2I + 2)`
      * `:F_bottom` → vertically concatenate levels `(2I + 3) : (4I + 2)`
      * `:1`, `:2`, … (numeric-like symbol) → use that single level

    Here `I = p.Ispin`. In all cases, columns `7:8` are taken and multiplied by `1e3` to convert to **mm**.
  * `n_bins::Tuple = (1, 4)`: `(nx_bins, nz_bins)` binning multipliers forwarded to `analyze_screen_profile`.
  * `width_mm::Float64 = 0.150`: Gaussian σ (mm) used for profile smoothing.
  * `add_plot::Bool = false`, `plot_xrange::Symbol = :all`: Plot options forwarded to `analyze_screen_profile`.
  * `λ_raw::Float64 = 0.01`, `λ_smooth::Float64 = 1e-3`: Spline regularization parameters forwarded downstream.
  * `mode::Symbol = :probability`: Histogram normalization mode (`:probability`, `:pdf`, etc.).

# Returns

`OrderedDict{Int, OrderedDict{Symbol,Any}}` where, for each index `i`:

  * `:Icoil`                   → current `data[:Icoils][i]` (A)
  * `:z_max_raw_mm`            → z at raw-profile maximum (mm)
  * `:z_max_raw_spline_mm`     → z at spline-fitted raw maximum (mm)
  * `:z_max_smooth_mm`         → z at smoothed-profile maximum (mm)
  * `:z_max_smooth_spline_mm`  → z at spline-fitted smoothed maximum (mm)
  * `:z_profile`               → `Nz × 3` matrix `[z_center  raw  smooth]`

# Notes

  * Requires that `analyze_screen_profile` is available and that the global `default_camera_pixel_size` (meters) is defined for bin sizing in that routine.
  * Level selection uses `p.Ispin`; e.g., for `I = 3/2`, `:F_top` picks levels `1:5` and `:F_bottom` picks `6:8` (1-based).
  * Throws an `AssertionError` if `data` is missing `:Icoils` or `:data`, or if `manifold` is a non-numeric symbol not equal to `:F_top`/`:F_bottom`.

# Example

```julia out = QM*analyze*profiles*to*dict(run*data, atom*params;     manifold=:F*top, n*bins=(1,4), width*mm=0.15,     add*plot=true, plot*xrange=:right, λ*raw=0.01, λ_smooth=1e-3)

z*first = out[1][:z*max*smooth*spline_mm]

# See also

[`analyze_screen_profile`](@ref), [`max_of_bspline_positions`](@ref), [`smooth_profile`](@ref)


## `QM_build_alive_screen` (function)

- `QM_build_alive_screen(Ix::AbstractVector{<:Real}, pairs::AbstractMatrix{T}, bad::OrderedCollections.OrderedDict{Int8, Vector{Vector{Int64}}}, p::AtomParams) where T @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_DiscardedParticles.jl:267`

```
QM_build_alive_screen(Ix, pairs, bad, p) 
    -> OrderedDict{Int8, Vector{Matrix{T}}}
```

For each current index `idx` and each hyperfine level `(F,mF) = fmf[k]`, build the matrix of **alive** particles (rows from `pairs` not listed in `bad[idx][k]`), and append the **screen** quantities as three new columns: `x_scr`, `z_scr` from `QM_Screen_position`, and `vz_scr` from `QM_Screen_velocity`.

Output shape

  * `OrderedDict{Int8, Vector{Matrix{T}}}`:

      * key: current index `Int8` (1-based index into `Ix`)
      * value: a vector of length `length(fmf)`; element `k` is a `(#alive, 9)` matrix with columns `[x, y, z, vx, vy, vz, x_scr, z_scr, vz_scr]`.

Arguments

  * `Ix::AbstractVector{<:Real}`            : currents (A), length = ncurrents.
  * `pairs::AbstractMatrix{T}`              : `N×6` base matrix (`x y z vx vy vz`) in SI units.
  * `bad::OrderedDict{Int8, Vector{Vector{Int}}}` : for each current index, a vector of *drop* indices per level (your finder’s result).
  * `p::AtomParams`                          : atomic parameters.

Notes

  * Uses `QM_Screen_position(I, F, mF, r0, v0, p)` and `QM_Screen_velocity(I, F, mF, v0, p)`. Adapt the signatures here if yours differ.
  * Returns **fresh matrices** (not views), because of the hcat of computed columns.
  * If a level keeps zero rows, returns a `0×9` matrix for that slot.

Columns `[x, y, z, vx, vy, vz, x_scr, z_scr, vz_scr]`

Example ```julia alive*with*screen = QM*build*alive*screen(Icoils, pairs, bad, p) M = alive*with_screen[Int8(3)][1]  # current idx=3, level k=1; size == (#alive, 9)


## `QM_build_filtered_pairs` (function)

- `QM_build_filtered_pairs(pairs::AbstractMatrix{T}, bad::OrderedCollections.OrderedDict{Int8, Vector{Vector{Int64}}}; copy) where T @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_DiscardedParticles.jl:203`

```
build_filtered_pairs(pairs, bad; copy=false) -> Dict{Int8, Vector{<:AbstractMatrix}}
```

From the original `pairs::N×6` matrix and the `bad` index lists `bad[Int8(idx)][k]` (particles to discard for current `idx` and level `k`), build a ragged container:

  * key:   current index as `Int8` (1..length(Ix))
  * value: `Vector` of length `nlevels`, where element `k` is a matrix of the        **kept** rows (particles) for level `k` at that current.

By default it returns **views** into `pairs` (no copying). Set `copy=true` to materialize independent matrices.

Notes

  * Works even if a level keeps zero rows (returns an empty `0×size(pairs,2)` matrix).
  * The relative order of rows is preserved.


## `QM_find_discarded_particles` (function)

- `QM_find_discarded_particles(Ix, pairs, p::AtomParams; t_length, verbose) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_DiscardedParticles.jl:34`

```
QM_find_bad_particles_ix(Ix, pairs, p::AtomParams;
                         t_length=1000, verbose=false)
    -> Dict{Int8, Vector{Vector{Int}}}
```

For each current `Ix[idx]`, determine which particles are **discarded** for each hyperfine state `(F,mF)` and return a nested container:

  * The result is a `Dict{Int8, Vector{Vector{Int}}}`.
  * For each current index `idx`, `result[Int8(idx)]` is a vector of length `nlevels` (number of `(F,mF)` pairs). Entry `k` is a vector of particle indices `j` that violate either the **SG cavity** constraint or the **post-SG pipe** clearance for level `fmf_levels[k]`.

Arguments

  * `Ix::AbstractVector`: currents to analyze.
  * `pairs::AbstractMatrix`: `N×6` matrix with columns `[x, y, z, vx, vy, vz]` (SI).
  * `p::AtomParams`: atomic parameters (mass, hyperfine constant, etc.).

Keywords

  * `t_length::Int=1000`: number of time samples across the SG region per particle.
  * `verbose::Bool=false`: print progress per current.

Returns

  * `Dict{Int8, Vector{Vector{Int}}}` as described above.

Notes

  * Assumes in scope: `y_FurnaceToSlit`, `y_SlitToSG`, `y_SG`, `default_R_tube`, and functions: `QM_EqOfMotion_z`, `z_magnet_edge_time`, `z_magnet_trench_time`, `QM_Screen_position`
  * Geometry envelopes (`z_top`, `z_bottom`) are computed once per particle and reused across all `(F,mF)` for that current to avoid redundant work.


## `QM_find_discarded_particles_multithreading` (function)

- `QM_find_discarded_particles_multithreading(Ix, pairs, p::AtomParams; t_length, verbose) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_DiscardedParticles.jl:101`

No documentation found for public symbol.

`Main.TheoreticalSimulation.QM_find_discarded_particles_multithreading` is a `Function`.

```
# 1 method for generic function "QM_find_discarded_particles_multithreading" from Main.TheoreticalSimulation:
 [1] QM_find_discarded_particles_multithreading(Ix, pairs, p::AtomParams; t_length, verbose)
     @ d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_DiscardedParticles.jl:101
```


## `TheoreticalSimulation` (binding)

No docstring or readme file found for internal module `Main.TheoreticalSimulation`.

# Public names

`AtomParams`, `BeamEffusionParams`, `CQD_EqOfMotion`, `CQD_EqOfMotion_z`, `CQD_Screen_position`, `CQD_Screen_velocity`, `EffusionParams`, `FreedmanDiaconisBins`, `QM_EqOfMotion`, `QM_EqOfMotion_z`, `QM_Screen_position`, `QM_Screen_velocity`, `QM_analyze_profiles_to_dict`, `QM_build_alive_screen`, `QM_build_filtered_pairs`, `QM_find_discarded_particles`, `QM_find_discarded_particles_multithreading`, `analyze_screen_profile`, `build_initial_conditions`, `clear_all`, `compute_weights`, `fmf_levels`, `generate_samples`, `pixel_coordinates`, `plot_SG_geometry`, `plot_velocity_stats`, `plot_μeff`, `z_magnet_edge_time`, `z_magnet_trench_time`, `μF_effective`


## `analyze_screen_profile` (function)

- `analyze_screen_profile(Ix, data_mm::AbstractMatrix; manifold, nx_bins, nz_bins, add_plot, plot_xrange, width_mm, λ_raw, λ_smooth, mode) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_QMSpline.jl:195`

```
analyze_screen_profile(
    Ix::Real,
    data_mm::AbstractMatrix;
    manifold::Symbol = :F_top,
    nx_bins::Integer = 2,
    nz_bins::Integer = 2,
    add_plot::Bool = false,
    plot_xrange::Symbol = :all,
    width_mm::Float64 = 0.150,
    λ_raw::Float64 = 0.01,
    λ_smooth::Float64 = 1e-3,
    mode::Symbol = :probability
) -> NamedTuple
```

Analyze the vertical **z** profile of particle hits on a screen for a given coil current. The routine builds a 2D histogram in **mm** over fixed x/z windows, averages over x to obtain a z-profile, smooths it, and finds peak locations using both discrete maxima and spline-fitted maxima.

This function supports single-current analysis and batch workflows (see the companion batch function). The `Ix` argument is used for plot labels/titles.

# Arguments

  * `Ix::Real`: Coil current (A). Used for plot annotation; does not affect the analysis.
  * `data_mm::AbstractMatrix`: `N×2` array of hit positions **in millimeters** with columns: `(:, 1) = x`, `(:, 2) = z`.

# Keyword Arguments

  * `manifold::Symbol = :F_top`: Label forwarded to plot titles/filenames (no effect on computation).
  * `nx_bins::Integer = 2`, `nz_bins::Integer = 2`: Binning multipliers. The physical bin sizes (mm) are `x_bin_size = 1e3 * nx_bins * default_camera_pixel_size` and `z_bin_size = 1e3 * nz_bins * default_camera_pixel_size`, where `default_camera_pixel_size` is a **global** in meters.
  * `add_plot::Bool = false`: If `true`, plot raw/smoothed/spline profiles and mark their maxima.
  * `plot_xrange::Symbol = :all`: Limits for the x-axis of the profile plot. `:all` → full z window; `:left` → left quarter; `:right` → right quarter.
  * `width_mm::Float64 = 0.150`: Gaussian kernel σ (mm) used by `smooth_profile`.
  * `λ_raw::Float64 = 0.01`, `λ_smooth::Float64 = 1e-3`: Spline regularization parameters for the raw and smoothed profiles.
  * `mode::Symbol = :probability`: Normalization mode passed to `StatsBase.normalize` for the 2D histogram (e.g., `:probability`, `:pdf`, `:density`).

# Method (overview)

1. Fixed analysis windows: `x ∈ [-8.0, 8.0]` mm, `z ∈ [-12.5, 12.5]` mm. Bin **centers** are symmetric about 0 (with a center exactly at 0).
2. Build a 2D histogram in `(x, z)`; average over x to get a 1D `z` profile.
3. Find raw discrete maximum (`argmax`).
4. Smooth the profile with a Gaussian (σ = `width_mm`) and find its discrete maximum.
5. Fit smoothing splines to both raw and smoothed profiles and extract peak positions via `max_of_bspline_positions`.

# Returns

A `NamedTuple` with fields:

  * `z_profile::Matrix{Float64}` — `Nz × 3` matrix: `[z_center  raw  smooth]`
  * `z_max_raw_mm::Float64` — z at raw-profile maximum (mm)
  * `z_max_raw_spline_mm::Float64` — z at spline-fitted raw maximum (mm)
  * `z_max_smooth_mm::Float64` — z at smoothed-profile maximum (mm)
  * `z_max_smooth_spline_mm::Float64` — z at spline-fitted smoothed maximum (mm)

# Notes

  * Requires a global `default_camera_pixel_size::Real` (meters).
  * Uses `max_of_bspline_positions` to obtain sub-bin-accurate maxima.
  * The `manifold` only affects labels/filenames, not the numerical results.

# Throws

  * `AssertionError` if `data_mm` is not `N×2` (x,z in mm), if `nx_bins/nz_bins ≤ 0`, if `width_mm ≤ 0`, or if `default_camera_pixel_size` is not defined.

# Example

```julia res = analyze*screen*profile(0.125, hits*mm;     manifold=:F*top, nx*bins=1, nz*bins=4,     width*mm=0.15, λ*raw=0.01, λ*smooth=1e-3, add*plot=true)

@show res.z*max*smooth*spline*mm

# See also

[`max_of_bspline_positions`](@ref), [`QM_analyze_profiles_to_dict`](@ref), [`smooth_profile`](@ref)


## `build_initial_conditions` (function)

- `build_initial_conditions(No::Integer, alive::AbstractMatrix{T}, rng::Random.AbstractRNG; mode) where T<:Real @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_Sampling.jl:219`

No documentation found for public symbol.

`Main.TheoreticalSimulation.build_initial_conditions` is a `Function`.

```
# 1 method for generic function "build_initial_conditions" from Main.TheoreticalSimulation:
 [1] build_initial_conditions(No::Integer, alive::AbstractMatrix{T}, rng::Random.AbstractRNG; mode) where T<:Real
     @ d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_Sampling.jl:219
```


## `clear_all` (function)

- `clear_all() @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation.jl:106`

```
clear_all() -> Nothing
```

Set to `nothing` every **non-const** binding in `Main`, except a small skip-list (`:Base`, `:Core`, `:Main`, and `Symbol("@__dot__")`). This effectively clears user-defined variables and functions from the current session without restarting Julia.

What it does

  * Iterates `names(Main; all=true)` and, for each name:
  * Skips if it is one of `:Base`, `:Core`, `:Main`, or `Symbol("@__dot__")`.
  * Skips if the binding is **not defined** or is **const**.
  * Otherwise sets the binding to `nothing` in `Main`.
  * Triggers a `GC.gc()` afterward.
  * Prints a summary message.

Notes & caveats

  * This will clear **user functions** too (they’re non-const bindings).
  * Type names and imported modules are usually `const` in `Main`, so they are **not** cleared.
  * This does not unload packages or reset the environment; it only nukes non-const globals.
  * There is no undo; you’ll need to re-run definitions after clearing.

Example ```julia julia> x = 1; y = "hi"; f(x) = x+1;

julia> clear_all() All user-defined variables (except constants) cleared.

julia> x, y, f (nothing, nothing, nothing)


## `compute_weights` (function)

- `compute_weights(x_array, λ0) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation.jl:131`

```
For BSplineKit fitting, compute weights for the B-spline fit.
Compute uniform weights scaled by (1 - λ0). Returns an array of the same size as `x_array`.
```


## `fmf_levels` (function)

- `fmf_levels(p::AtomParams; J, stretched_only, Fsel) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_Params.jl:118`

```
fm_pairs_biordered(p::AtomParams; J=1//2, stretched_only=false, Fsel=nothing)
    -> Vector{Tuple{Float64,Float64}}
```

Construct hyperfine `(F, mF)` pairs using `I = p.Ispin` and ordering driven by `sign(p.γn)`:

  * If `p.γn > 0`: for each manifold, those with `F ≥ I` are listed **mF descending**, and those with `F < I` are **mF ascending** (upper desc / lower asc for J=1/2).
  * If `p.γn < 0`: the directions are flipped.

Keywords

  * `J`                — electronic angular momentum (default `1//2`).
  * `stretched_only`   — if `true`, return only `(F, ±F)` for each selected `F`.
  * `Fsel`             — if provided, keep only that F (≈ comparison).

Returns a vector of `(F, mF)` as `Float64`.


## `generate_samples` (function)

- `generate_samples(No::Int64, p::EffusionParams; v_pdf, rng, multithreaded, base_seed) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_Sampling.jl:106`

No documentation found for public symbol.

`Main.TheoreticalSimulation.generate_samples` is a `Function`.

```
# 1 method for generic function "generate_samples" from Main.TheoreticalSimulation:
 [1] generate_samples(No::Int64, p::EffusionParams; v_pdf, rng, multithreaded, base_seed)
     @ d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_Sampling.jl:106
```


## `pixel_coordinates` (function)

- `pixel_coordinates(img_size::Integer, bin_size::Integer, pixel_size::Real) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation.jl:221`

Centers of binned pixels (1D) in physical units. First center at (bin*size*pixel*size)/2. Requires img*size % bin*size == 0.


## `plot_SG_geometry` (function)

- `plot_SG_geometry(filename::AbstractString) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_Plots.jl:112`

```
plot_SG_geometry(filename::AbstractString) -> Nothing
```

Render and save a 2D cross-section of the Stern–Gerlach slit geometry.

What is drawn

  * **Top magnet edge**: shaded region above `z_magnet_edge(x)`.
  * **Bottom trench**: shaded region below `z_magnet_trench(x)`.
  * **Slit aperture**: centered rectangle of width `default_x_slit` and height `default_z_slit`.

Axes & units

  * Plots use millimetres on both axes (internally samples `x` in metres).
  * Limits: `x ∈ [-8, 8] mm`, `y ∈ [-3, 7] mm`. Aspect ratio is 1:1.
  * Labels use LaTeX (`L"...")`.

Sampling

  * `x` is sampled uniformly over `[-10, 10] mm` with 10_001 points for smooth shapes.

Side effects

  * Displays the figure and saves it to `joinpath(OUTDIR, "filename.FIG_EXT")`.

Assumptions / dependencies

  * Functions `z_magnet_edge(x)` and `z_magnet_trench(x)` are in scope and return `z(x)` in metres.
  * Globals `default_x_slit`, `default_z_slit`, `OUTDIR`, and `FIG_EXT` are defined.
  * Uses `Plots.jl` (with LaTeXStrings) and an initialized backend.

Example ```julia plot*SG*geometry("sg_geometry")

# writes OUTDIR/sg*geometry.<FIG*EXT>


## `plot_velocity_stats` (function)

- `plot_velocity_stats(alive::Matrix{Float64}, title::String, filename::String) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_Plots.jl:191`

```
plot_velocity_stats(alive::Matrix{Float64}, title::String, filename::String) -> Plots.Plot
```

Build, display, and save a dashboard of velocity/position statistics for a set of particles.

Input

  * `alive`: `N × ≥6` matrix with columns 1: `x` (m), 2: `y` (m), 3: `z` (m), 4: `vₓ` (m/s), 5: `vᵧ` (m/s), 6: `v_z` (m/s).
  * `title`: Plot title (shown atop the dashboard).
  * `filename`: Basename for saving the figure (written to `joinpath(OUTDIR, "filename.FIG_EXT")`).

What it plots (7 panels)

1. **Speed histogram** of `‖v‖`, with vertical lines at the mean `⟨v₀⟩` and RMS `√⟨v₀²⟩`.
2. **Polar angle** `θ_v = acos(v_z/‖v‖)` histogram (radians).
3. **Azimuth** `φ_v = atan(v_y, vₓ)` histogram (radians).
4. **2D position histogram** of `(x, z)` with axes in mm (x) and μm (z).

5–7. **Component histograms** for `vₓ`, `vᵧ`, `v_z` (PDF-normalized).

Details

  * Histogram bin counts are chosen via `FreedmanDiaconisBins`.
  * Axes use LaTeX labels; units: mm for x, μm for z, m/s for velocities.
  * The figure is displayed and saved; the function returns the assembled `Plots.Plot`.

Assumptions / dependencies

  * Globals `OUTDIR` and `FIG_EXT` are defined.
  * Functions/packages in scope: `FreedmanDiaconisBins`, `Plots`, `LaTeXStrings`.
  * Assumes nonzero speeds for angle calculations (`‖v‖ > 0`).

Example ```julia fig = plot*velocity*stats(alive, "Beam velocity statistics", "vel*stats*run42")


## `plot_μeff` (function)

- `plot_μeff(p::AtomParams, filename::AbstractString) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_Plots.jl:30`

```
plot_ueff(II, path_filename::AbstractString) -> Plot

Plot the effective magnetic moment μ_F/μ_B versus coil current for all hyperfine
levels (F, m_F) of a spin-I system, and annotate the magnetic crossing point.

# Arguments
- `II`: Nuclear spin quantum number (e.g., 3/2, 4, etc.).
- `path_filename::AbstractString`: Output file path for saving the figure.

# Behavior
- Computes and plots μ_F/μ_B curves for all (F, m_F) states using `μF_effective`.
- Uses solid lines for most F = I + 1/2 states, a dashed line for the lowest m_F
in F = I + 1/2, and dashed lines for all F = I – 1/2 states.
- Colors each curve using the `:phase` palette.
- Finds the magnetic crossing current `I₀` by solving `BvsI(I) = …` and annotates
the plot with:
    - I₀ in A
    - ∂ₓBₓ at I₀ in T/m
    - B_z at I₀ in mT
- Plots current on a logarithmic x-axis.

# Returns
- The `Plots.Plot` object for the generated figure.

# Notes
- Requires `μF_effective`, `μB`, `BvsI`, `GvsI`, `ħ`, `Ahfs`, `Ispin`,
`γₙ`, and `γₑ` to be defined in scope.
```


## `z_magnet_edge_time` (function)

- `z_magnet_edge_time(t, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_MagneticField.jl:188`

```
z_magnet_edge_time(t, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}) -> Float64
```

Edge profile evaluated **along a trajectory** at time `t`.

Computes the instantaneous horizontal position `x(t) = r0[1] + v0[1]*t` (m) and returns `z_magnet_edge(x(t))` (m), using the same geometry as `z_magnet_edge`.

Arguments

  * `t`  : time (s)
  * `r0` : initial position vector; only `r0[1]` (x, m) is used
  * `v0` : initial velocity vector; only `v0[1]` (vx, m/s) is used


## `z_magnet_trench_time` (function)

- `z_magnet_trench_time(t, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_MagneticField.jl:219`

```
z_magnet_trench_time(t, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}) -> Float64
```

Trench profile evaluated **along a trajectory** at time `t`.

Computes `x(t) = r0[1] + v0[1]*t` (m) and returns `z_magnet_trench(x(t))` (m), using the same geometry as `z_magnet_trench`.

Arguments

  * `t`  : time (s)
  * `r0` : initial position vector; only `r0[1]` (x, m) is used
  * `v0` : initial velocity vector; only `v0[1]` (vx, m/s) is used


## `μF_effective` (function)

- `μF_effective(Ix, F, mF, p::AtomParams) @ Main.TheoreticalSimulation d:\titimbo\SingleSG_Xukun2025\Modules\TheoreticalSimulation_muF.jl:32`

```
μF_effective(Ix, F, mF, p::AtomParams) -> Float64

Effective magnetic moment μ_F for a given hyperfine manifold and Zeeman sublevel,
based on the (Breit–Rabi–style) expression you coded.

Inputs
- `Ix`  : Coil current (units consistent with `BvsI(Ix)` → magnetic field).
- `II`  : Nuclear spin quantum number (I). Can be integer or half-integer.
- `F`   : Total angular momentum (must be `I ± 1/2`).
- `mF`  : Magnetic quantum number (must satisfy `-F ≤ mF ≤ F`).

Assumptions / Globals
- Uses global constants: `ħ, Ahfs, γₑ, γₙ, μB, gₑ`.
- Uses global field map/function: `BvsI(Ix)` returning B (same units used in Δ).
- Defines the adimensional field parameter
`normalized_B = (γₑ - γₙ) * ħ / ΔE * BvsI(Ix)`,
where `ΔE = 2π * ħ * Ahfs * (I + 1/2)`.

Details
- For the upper manifold `F = I + 1/2`, the `mF = ±F` edges use the simplified
analytic form `μF = ± gₑ/2 * (1 + 2*γₙ/γₑ * I) * μB`.
- For other `mF` and for the lower manifold `F = I - 1/2`, uses the full expressions
with the square‑root denominator
`sqrt(1 - 4*mF/(2I+1)*normalized_B + normalized_B^2)`; the argument is clamped
to ≥ 0 to avoid numerical noise causing `NaN`.

Returns
- `Float64` effective magnetic moment (units of μB if you keep the constants consistent).
```


