# ==============================================================================
# Velocity sampling and analytic/semi-analytic Stern–Gerlach profile models
#
# This file implements three layers:
#
#   1. Single-particle velocity sampling
#      AtomicBeamVelocity_v3 / AtomicBeamVelocity_v2
#      Draw one (vx, vy, vz) velocity vector for an effusive-beam particle,
#      using two different (but statistically equivalent) inverse-CDF methods
#      for the Maxwell–Boltzmann speed distribution.
#
#   2. Closed-form / semi-analytic detector profiles
#      getProbDist_v3 / QM_PDF_profile / QM_PDF_profile_smooth
#      Evaluate the expected Stern–Gerlach screen profile directly from a
#      closed-form kinematic model (no particle-by-particle Monte Carlo),
#      for a single hyperfine level's effective magnetic moment.
#
#   3. Profile combination and post-processing
#      ProbDist_convolved / ProbDist_convolved! / weighted_QM_PDF_profile /
#      weighted_QM_PDF_profile_smooth / normalize_profile
#      Combine per-level profiles into a weighted total (summed over hyperfine
#      sublevels), apply Gaussian smoothing, and normalize the result.
# ==============================================================================
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Module-level cached objects
# ──────────────────────────────────────────────────────────────────────────────
# Cached Gamma(3/2, 1) distribution, shared by AtomicBeamVelocity_v2. Building
# a `Gamma` object allocates; since this sampler can be called millions of
# times per simulation run (once per candidate particle in the rejection
# samplers), constructing it fresh on every call was a real, easily avoidable
# cost. Built once here, at include time, instead.
const _GAMMA_3_2 = Gamma(3/2, 1.0)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Velocity sampling
# ──────────────────────────────────────────────────────────────────────────────
"""
    AtomicBeamVelocity_v3(rng::AbstractRNG, p::EffusionParams) -> SVector{3,Float64}
 
Draw a single velocity vector `(vx, vy, vz)` for a particle in an effusive beam,
with directions restricted to a cone of half-angle `asin(p.sinθmax)` around the
beam axis (taken as `y`). The **direction** is sampled uniformly over solid angle
within the cone, and the **speed** is sampled from the Maxwell–Boltzmann speed
distribution using an analytic inverse-CDF based on the Lambert-W function:
 
- Direction: `ϕ ~ U(0, 2π)`, `θ = asin(p.sinθmax * √u)`, which yields uniform
  solid-angle density inside the cone.
- Speed: `v = √[-2 p.α2 * (1 + W_{-1}((u - 1)/e))]`, i.e. PDF ∝ `v^3 exp(-v²/(2 p.α2))`
  with `p.α2 = k_B T / M`.
 
Returns an `SVector{3,Float64}` with components in meters per second, where `vy`
is along the beam axis.
 
Notes
- Uses `LambertW.lambertw` (branch `-1`) and the module constants `TWOπ` and `INV_E`.
- `rng` is any `AbstractRNG` (e.g., `MersenneTwister`).
"""
@inline function AtomicBeamVelocity_v3(rng::AbstractRNG, p::EffusionParams)::SVector{3,Float64}
    ϕ = TWOπ * rand(rng)
    θ = asin(p.sinθmax * sqrt(rand(rng)))
    v = sqrt(-2*p.α2 * (1 + lambertw((rand(rng)-1)*INV_E, -1)))
    sθ = sin(θ); cθ = cos(θ); sϕ = sin(ϕ); cϕ = cos(ϕ)
    return SVector(v*sθ*sϕ, v*cθ, v*sθ*cϕ)
end

"""
    AtomicBeamVelocity_v2(rng::AbstractRNG, p::EffusionParams) -> SVector{3,Float64}
 
Draw a single velocity vector `(vx, vy, vz)` for a particle in an effusive beam,
with directions restricted to a cone of half-angle `asin(p.sinθmax)` around the
beam axis (`y`). The **direction** is sampled uniformly over solid angle within
the cone, and the **speed** is sampled from the Maxwell–Boltzmann speed
distribution via a Gamma draw:
 
- Direction: `ϕ ~ U(0, 2π)`, `θ = asin(p.sinθmax * √u)` (uniform in solid angle).
- Speed: `v = √(2 p.α2 * X)`, where `X ~ Gamma(3/2, 1)`. This is equivalent to
  sampling MB speed with PDF ∝ `v² exp(-v²/(2 p.α2))`, since `v²/(2 p.α2) ~ Gamma(3/2,1)`.
 
Returns an `SVector{3,Float64}` in m/s.
 
Notes
- Uses the module-level cached `_GAMMA_3_2 = Gamma(3/2, 1.0)` rather than
  constructing a new `Gamma` object on every call.
- `p.α2 = k_B T / M` controls the speed scale; `p.sinθmax` controls the angular cone.
"""
@inline function AtomicBeamVelocity_v2(rng::AbstractRNG, p::EffusionParams)::SVector{3,Float64}
    ϕ = TWOπ * rand(rng)
    θ = asin(p.sinθmax * sqrt(rand(rng)))
    v = sqrt(2 * p.α2 * rand(rng, _GAMMA_3_2))
    sθ = sin(θ); cθ = cos(θ); sϕ = sin(ϕ); cϕ = cos(ϕ)
    return SVector(v*sθ*sϕ, v*cθ, v*sθ*cϕ)
end

# ──────────────────────────────────────────────────────────────────────────────
# Closed-form / semi-analytic detector profiles
# ──────────────────────────────────────────────────────────────────────────────
"""
    getProbDist_v3(μ, dBzdz, zd, p::AtomParams, q::EffusionParams;
                   wfurnace::Float64=DEFAULT_z_furnace,
                   npts::Int=5001,
                   pdf::Symbol=:point) -> Vector{Float64}
 
Closed-form Stern–Gerlach screen profile for a constant field gradient.
 
Computes the transverse probability density along the screen positions `zd`
(optionally averaged over a finite furnace width). Uses simple kinematics in a
piecewise-drift geometry and an effusive-beam speed scale.
 
# Arguments
- `μ::Real`: effective magnetic moment (J/T). **Sign matters**; if `μ < 0`,
  the function evaluates the profile at the mirrored coordinate `-zd` (folded
  into the per-point evaluation, not by allocating a negated copy of `zd`) so
  the returned profile is oriented consistently.
- `dBzdz::Real`: magnetic-field gradient (T/m), assumed constant over the SG region.
- `zd::AbstractVector`: screen coordinates (m) where the profile is evaluated.
- `p::AtomParams`: must provide `p.M` (mass, kg).
- `q::EffusionParams`: must provide `q.α2` so that `β² = 2α2` (with `β` the
  effusive-beam speed scale).
- `wfurnace`: source width (m). Used only when `pdf = :finite`.
- `npts`: number of trapezoid samples across `[-wfurnace/2, +wfurnace/2]`
  when `pdf = :finite`. Must be odd, so the grid includes `z0 = 0` exactly.
- `pdf`: `:point` (infinitesimal source) or `:finite` (averaged over furnace width).
 
# Geometry (read from globals)
Uses (all in meters) `DEFAULT_y_FurnaceToSlit`, `DEFAULT_y_SlitToSG`,
`DEFAULT_y_SG`, `DEFAULT_y_SGToScreen`, and entrance slit width `DEFAULT_z_slit`.
 
Let `LOS = y_FurnaceToSlit`, `LSG = y_SG`, `LSGD = y_SGToScreen`,
`Ltot = LOS + y_SlitToSG + LSG + LSGD`, and `ℓ = Ltot/LOS`.
 
# Model (per source offset `z0`)
- Acceleration parameter: `a = (μ * dBzdz / p.M) * LSG * (LSG + 2LSGD) / 2`
- Speed scale: `β² = 2 q.α2`, define `c = a / β²`
- Denominators:
  - `d1 = z − z0 − (w/2 − z0) * ℓ`
  - `d2 = z − z0 + (w/2 + z0) * ℓ`, with `w = DEFAULT_z_slit`
- Per-point contribution (only where `d > 0`):
    pdf(z|z0) = [ -exp(-p1)(p1+1) + exp(-p2)(p2+1) ] / ℓ
    p1 = c / d1, p2 = c / d2
- If `pdf = :finite`, the returned profile is the trapezoidal average of `pdf(z|z0)`
over `z0 ∈ [-wfurnace/2, +wfurnace/2]`, divided by `wfurnace`.
 
# Returns
A vector with `length(zd)` giving the (unnormalized) profile at each `zd`.
 
# Notes
- The result is **not normalized**; normalize afterwards if required.
- Uses an in-place kernel to avoid allocations and is suitable for large `zd`.
- Assumes SI units throughout.
- Asserts `pdf ∈ (:point, :finite)`, `LOS > 0` (required since `ℓ = Ltot/LOS`
  divides by it), and — when `pdf = :finite` — that `npts ≥ 3` **and odd**.
 
# Example
julia
zd = range(-12.5e-3, 12.5e-3; length=20_001)
μ    = μB_eff           # your effective moment (J/T)
grad = GvsI(I0)        # dB/dz for current I0 (T/m)
 
prof_point  = getProbDist_v3(μ, grad, zd, K39_params, effusion_params; pdf=:point)
prof_finite = getProbDist_v3(μ, grad, zd, K39_params, effusion_params;
                           wfurnace=100e-6, npts=2001, pdf=:finite)
"""
function getProbDist_v3(μ::Real, dBzdz::Real, zd::AbstractVector, p::AtomParams, q::EffusionParams;
                     wfurnace::Float64=DEFAULT_z_furnace, npts::Int=5001, pdf::Symbol=:point)
 
    @assert pdf === :point || pdf === :finite "pdf must be :point or :finite"
 
    # --- Geometry (m) ---
    LOS   = DEFAULT_y_FurnaceToSlit
    LSSG  = DEFAULT_y_SlitToSG
    LSG   = DEFAULT_y_SG
    LSGD  = DEFAULT_y_SGToScreen
    @assert LOS > 0           "Furnace-to-slit distance (LOS) must be > 0 (ℓ = Ltot/LOS divides by it below)."
    @assert LSSG ≥ 0 && LSG ≥ 0 && LSGD ≥ 0  "Segment lengths must be ≥ 0."
 
    Ltot  = LOS + LSSG + LSG + LSGD
    lfrac = Ltot / LOS
    inv_lfrac  = inv(lfrac)
 
    # --- Slit width at SG entrance (m) ---
    w     = DEFAULT_z_slit
    halfw = w/2
 
    # Mirror if μ < 0 (preserves shape). The sign flip is folded into the
    # per-point evaluation inside `_shifted_core!` below as a scalar
    # multiplier, instead of allocating a negated copy of the whole `zd`
    # array (an O(length(zd)) allocation that's unnecessary for a sign flip).
    sign_flip = μ < 0 ? -1.0 : 1.0
    dBzdz     = sign_flip * dBzdz   # scalar reassignment — no allocation either way
 
    # --- Derived quantities (scalar path) ---
    aSG = μ * dBzdz / p.M
    a   = aSG * LSG * (LSG + 2*LSGD) / 2
    c   = a / (2*q.α2)  # = a/β²
 
    # preallocate once
    out = similar(zd, Float64)
    tmp = similar(zd, Float64)
 
    # in-place version of shifted_core
    @inline function _shifted_core!(dest, z0::Real)
        @inbounds @simd for i in eachindex(dest, zd)
            zi = sign_flip * zd[i]
            # denominators
            d1 = zi - z0 - (halfw - z0) * lfrac
            d2 = zi - z0 + (halfw + z0) * lfrac
 
            t = 0.0
            if d1 > 0.0
                p1 = c / d1
                t += -exp(-p1) * (p1 + 1.0) * inv_lfrac
            end
            if d2 > 0.0
                p2 = c / d2
                # Equivalent to the docstring's "+ exp(-p2)(p2+1)" term,
                # written directly instead of as `t -= -exp(-p2)*(...)`.
                t += exp(-p2) * (p2 + 1.0) * inv_lfrac
            end
            dest[i] = t
        end
        return dest
    end
 
    # --- Mode selection ---
    if pdf === :point
        # Infinitely thin source (z0 = 0)
        return _shifted_core!(out, 0.0)
    else
        @assert wfurnace > 0 "wfurnace must be > 0."
        @assert npts ≥ 3 && isodd(npts) "npts must be ≥ 3 and odd."
 
        z0s = range(-wfurnace/2, wfurnace/2; length=npts)
        h   = step(z0s)
 
        # Trapezoidal integration over z0 (streaming to reduce allocations)
        _shifted_core!(out, first(z0s));  @. out *= (h/2)
        _shifted_core!(tmp, last(z0s));   @. out += (h/2) * tmp
        @inbounds for z0 in z0s[2:end-1]
            _shifted_core!(tmp, z0)
            @. out += h * tmp
        end
        @. out /= wfurnace
        return out
    end
end

"""
    QM_PDF_profile(Ic, μ_effective, z, p, q) -> Vector{Float64}
 
Return the detector-space probability-density profile evaluated on the grid `z`
for coil current `Ic` and effective magnetic moment `μ_effective`.
 
The model assumes:
- a rectangular furnace opening of width `DEFAULT_z_furnace`,
- a slit of width `DEFAULT_z_slit`,
- ballistic propagation from furnace to slit, slit to SG, through the SG region,
  and then from SG to the detector,
- a Stern–Gerlach acceleration determined by `μ_effective * GvsI(Ic) / p.M`.
 
All lengths are in meters.
 
# Arguments
- `Ic::Real`: coil current (A).
- `μ_effective::Real`: effective magnetic moment (J/T). This function does
  **not** itself handle the sign of `μ_effective` — it is expected to already
  be non-negative (e.g. `abs(μF_effective(...))`), with any mirroring for a
  physically negative moment applied by the caller (see
  [`weighted_QM_PDF_profile`](@ref), which passes `abs(μ)` here and mirrors
  the resulting array itself for `μ < 0`).
- `z::AbstractVector{T}`: detector coordinate grid (m).
- `p::AtomParams`: must provide `p.M` (mass, kg).
- `q::EffusionParams`: must provide `q.α2` (thermal speed scale).
 
# Returns
`Vector` of `length(z)` with the (unnormalized) profile, eltype promoted from
`T` and `Float64`.
"""
function QM_PDF_profile(Ic::Real, μ_effective::Real, z::AbstractVector{T}, p::AtomParams, q::EffusionParams) where {T<:Real}
 
    # --- Geometry (m) ---
    LOS   = DEFAULT_y_FurnaceToSlit
    LSSG  = DEFAULT_y_SlitToSG
    LSG   = DEFAULT_y_SG
    LSGD  = DEFAULT_y_SGToScreen
    @assert LOS  > 0 "Furnace-to-slit distance (LOS) must be > 0."
    @assert LSSG ≥ 0 "Slit-to-SG distance (LSSG) must be ≥ 0."
    @assert LSG  ≥ 0 "SG length (LSG) must be ≥ 0."
    @assert LSGD ≥ 0 "SG-to-screen distance (LSGD) must be ≥ 0."
 
    Ltot  = LOS + LSSG + LSG + LSGD
 
    # Aperture sizes (m)
    wSG1 = DEFAULT_z_slit
    wfurnace = DEFAULT_z_furnace
    @assert wSG1    > 0 "Slit width must be > 0."
    @assert wfurnace > 0 "Furnace width must be > 0."
 
    # ---------------- Kinematics ----------------
    # SG acceleration in the z direction.
    aSG = μ_effective * GvsI(Ic) / p.M
 
    # Thermal velocity scale. If q.α2 = kT/M, then β = sqrt(2α2).
    β = sqrt(2 * q.α2)
 
    # ---------------- Derived constants ----------------
    c1 = Ltot
    c2 = aSG * LSG * (LSG + 2 * LSGD) / 2
 
    # Geometric factors from source/slit projection
    g1 = c1 / LOS - 1
    g2 = c1 * wSG1 / (2 * LOS)
 
    z3 = g2 - g1 * wfurnace / 2
    z4 = g2 + g1 * wfurnace / 2
 
    # Output array
    pdf = zeros(promote_type(T, Float64), length(z))
 
    # Overall normalization
    prefac = 1 / (2 * g1 * g2 * wfurnace)
    β² = β^2
 
    @inbounds for i in eachindex(z)
        zi = z[i]
 
        # Helper terms; each contributes only in its allowed support region.
        f4p = 0.0
        if zi > -z4
            u = zi + z4
            f4p = u * exp(-c2 / u / β²)
        end
 
        f3p = 0.0
        if zi > -z3
            u = zi + z3
            f3p = u * exp(-c2 / u / β²)
        end
 
        f3m = 0.0
        if zi > z3
            u = zi - z3
            f3m = u * exp(-c2 / u / β²)
        end
 
        f4m = 0.0
        if zi > z4
            u = zi - z4
            f4m = u * exp(-c2 / u / β²)
        end
 
        pdf[i] = prefac * (f4p - f3p - f3m + f4m)
    end
 
    return pdf
 
end

"""
    QM_PDF_profile_smooth(Ic, μeff, z, p, q, wd) -> Vector{Float64}
 
Convenience wrapper: compute the closed-form profile via
[`QM_PDF_profile`](@ref) and apply Gaussian smoothing of width `wd` via
`smooth_profile`.
 
# Arguments
- `Ic::Real`: coil current (A).
- `μeff::Real`: effective magnetic moment (J/T); see [`QM_PDF_profile`](@ref)
  for the sign convention (expected non-negative).
- `z::AbstractVector{<:Real}`: detector coordinate grid (m).
- `p::AtomParams`, `q::EffusionParams`: atomic and effusion parameters.
- `wd::Real`: Gaussian smoothing width (same units as `z`).
 
# Returns
`Vector` of `length(z)`, the smoothed (unnormalized) profile.
"""
function QM_PDF_profile_smooth(Ic::Real, μeff::Real, z::AbstractVector{<:Real}, p::AtomParams, q::EffusionParams, wd::Real)
    pdf = QM_PDF_profile(Ic, μeff, z, p, q)
    return smooth_profile(z, pdf, wd)
end


# ──────────────────────────────────────────────────────────────────────────────
# Profile combination and post-processing
# ──────────────────────────────────────────────────────────────────────────────
"""
    ProbDist_convolved(z::AbstractVector{<:Real}, z_pdf::AbstractVector{<:Real}, w_width::Real)
        -> Vector
 
Convolve/smooth `z_pdf` sampled at locations `z` with a Gaussian kernel of
width `w_width`. Delegates to `smooth_profile`.
 
# Throws
- `ArgumentError` if `z` and `z_pdf` have different lengths.
"""
function ProbDist_convolved(z::AbstractVector{<:Real}, z_pdf::AbstractVector{<:Real}, w_width::Real)
    length(z) == length(z_pdf) ||
        throw(ArgumentError("z and z_pdf must have the same length"))
    return smooth_profile(z, z_pdf, w_width)
end
 
"""
    ProbDist_convolved!(out::AbstractVector, z::AbstractVector, z_pdf::AbstractVector, w_width::Number)
        -> out
 
Convolve/smooth `z_pdf` (sampled at `z`) with a Gaussian kernel of width
`w_width`, writing the result into the caller-supplied buffer `out`.
 
# Allocation behavior
This avoids one allocation relative to calling [`ProbDist_convolved`](@ref)
and copying the result yourself: the kernel and the full convolution output
are still allocated internally (both are genuinely required — convolving
necessarily materializes at least the kernel and the raw convolution
result), but the final 'same'-mode trim is written directly into `out`
through a view, instead of first allocating a separate trimmed array and
then copying that into `out`. This is **not** a zero-allocation operation;
achieving that would require an in-place Gaussian-kernel/convolution
implementation.
 
# Throws
- `ArgumentError` if `out`, `z`, and `z_pdf` don't all have the same length.
"""
@inline function ProbDist_convolved!(out::AbstractVector, z::AbstractVector, z_pdf::AbstractVector, w_width::Number)
    length(out) == length(z) == length(z_pdf) ||
        throw(ArgumentError("out, z, and z_pdf must all have the same length"))
 
    # Build the kernel and run the convolution directly here (rather than via
    # `smooth_profile`, which would itself allocate an already-trimmed copy
    # of the result) so the final 'same'-mode trim can be written straight
    # into `out` through a view, with no extra allocation for that step.
    kernel = gaussian_kernel(z, w_width)
    full   = DSP.conv(z_pdf, kernel)        # length(z_pdf) + length(kernel) - 1
 
    n         = length(z_pdf)
    start_idx = div(length(kernel), 2) + 1
    copyto!(out, @view full[start_idx:start_idx + n - 1])
    return out
end

"""
    weighted_QM_PDF_profile(Ic, z, p, q; Fsel, weights=nothing, normalize=true)
 
Compute the weighted Stern–Gerlach detector probability density profile for a
set of hyperfine sublevels (F, mF).
 
For each level selected by `Fsel`:
- The effective magnetic moment μ = μF_effective(Ic, F, mF, p) is computed.
- The profile is evaluated using abs(μ).
- If μ < 0, the resulting profile is mirrored (reversed) to account for
  deflection in the opposite direction.
- The contribution is multiplied by the corresponding weight.
 
All contributions are summed to produce the total profile.
 
Arguments
---------
Ic::Real
    Coil current.
 
z::AbstractVector
    Detector coordinate grid (must be uniformly spaced).
 
p::AtomParams
    Atomic parameters.
 
q::EffusionParams
    Effusion / thermal parameters.
 
Keyword Arguments
-----------------
Fsel::Integer
    Hyperfine manifold to include (e.g. F = 1 or F = 2).
 
weights
    Optional weights for each (F, mF) level. Must have the same length
    as the number of selected levels. If `nothing`, equal weights are used.
 
normalize::Bool = true
    If true, normalize the output so that sum(pdf) * Δz = 1.
 
Returns
-------
Vector{Float64}
    Weighted probability density evaluated on `z`.
 
Notes
-----
- The grid `z` is assumed to be uniformly spaced; normalization uses Δz = z[2] - z[1].
- If `weights` are provided, they are internally normalized to sum to 1.
- The mirroring for μ < 0 enforces the physical symmetry of opposite
  Stern–Gerlach deflections, and assumes `z` is symmetric about zero (since
  it reverses the *output array* rather than re-evaluating on a mirrored
  coordinate grid).
 
Example
-------
pdf = weighted_QM_PDF_profile(Ic, z, p, q; Fsel=2)
"""
function weighted_QM_PDF_profile(
    Ic::Real,
    z::AbstractVector,
    p::AtomParams,
    q::EffusionParams;
    Fsel::Integer,
    weights=nothing,
    normalize::Bool=true,
)
 
    @assert length(z) ≥ 2 "z must contain at least two points."
 
    levels = fmf_levels(p; Fsel=Fsel)
    nlev = length(levels)
    @assert nlev > 0 "No levels found for Fsel=$Fsel."
 
    # Default: equal weights
    w = weights === nothing ? fill(1.0 / nlev, nlev) : collect(weights)
 
    @assert length(w) == nlev "weights must have the same length as the number of selected levels."
 
    sw = sum(w)
    @assert sw > 0 "Sum of weights must be positive."
    w ./= sw
 
    pdf_total = zeros(promote_type(eltype(z), Float64), length(z))
 
    for (j, v) in enumerate(levels)
        μ = μF_effective(Ic, v[1], v[2], p)
        pdf_j = QM_PDF_profile(Ic, abs(μ), z, p, q)
 
        if μ < 0
            pdf_total .+= w[j] .* reverse(pdf_j)
        else
            pdf_total .+= w[j] .* pdf_j
        end
    end
 
    if normalize
        Δz = z[2] - z[1]
        norm = sum(pdf_total) * Δz
        @assert norm > 0 "Profile normalization is non-positive."
        pdf_total ./= norm
    end
 
    return pdf_total
end

"""
    weighted_QM_PDF_profile_smooth(Ic, z, p, q, wd; Fsel, weights=nothing, normalize=true)
 
Compute a smoothed version of the weighted Stern–Gerlach detector profile.
 
This function:
1. Computes the weighted profile using `weighted_QM_PDF_profile` (without normalization),
2. Applies Gaussian smoothing with width `wd`,
3. Optionally normalizes the final result.
 
Arguments
---------
Ic::Real
    Coil current.
 
z::AbstractVector
    Detector coordinate grid (must be uniformly spaced).
 
p::AtomParams
    Atomic parameters.
 
q::EffusionParams
    Effusion / thermal parameters.
 
wd::Real
    Gaussian smoothing width (standard deviation, in the same units as `z`).
 
Keyword Arguments
-----------------
Fsel::Integer
    Hyperfine manifold to include.
 
weights
    Optional weights for each (F, mF) level.
 
normalize::Bool = true
    If true, normalize the final smoothed profile so that sum(pdf) * Δz = 1.
 
Returns
-------
Vector{Float64}
    Smoothed and optionally normalized profile.
 
Notes
-----
- Smoothing can slightly change the total integral; normalization is therefore
  applied after smoothing.
- The grid `z` must be uniformly spaced.
 
Example
-------
pdf = weighted_QM_PDF_profile_smooth(Ic, z, p, q, 0.2e-3; Fsel=2)
"""
function weighted_QM_PDF_profile_smooth(
    Ic::Real,
    z::AbstractVector,
    p::AtomParams,
    q::EffusionParams,
    wd::Number;
    Fsel::Integer,
    weights=nothing,
    normalize::Bool=true,
)
 
    @assert length(z) ≥ 2 "z must contain at least two points."
 
    pdf = weighted_QM_PDF_profile(
        Ic, z, p, q;
        Fsel=Fsel,
        weights=weights,
        normalize=false
    )
 
    pdf = smooth_profile(z, pdf, wd)
 
    if normalize
        Δz = z[2] - z[1]
        norm = sum(pdf) * Δz
        @assert norm > 0 "Smoothed profile normalization is non-positive."
        pdf ./= norm
    end
 
    return pdf
end

"""
    normalize_profile(x, y; method = :area)
 
Normalize a 1D profile `y` defined on a uniformly spaced grid `x`.
 
# Arguments
- `x::AbstractVector`: Grid points (must be uniformly spaced and strictly increasing).
- `y::AbstractVector`: Values of the profile at points `x`.
- `method::Symbol`: Normalization method. Options are:
    - `:area` (default): Normalize such that ∫ y(x) dx = 1 using a Riemann sum.
    - `:max`: Normalize such that `maximum(y) = 1`.
 
# Returns
- A vector with the same size as `y`, containing the normalized profile.
 
# Details
- For `method = :area`, the normalization is computed as:
  `norm = sum(y) * Δx`, where `Δx = x[2] - x[1]`.
- Assumes `x` is uniformly spaced.
- The function does not modify the input arrays.
 
# Errors
- Throws an `AssertionError` if:
    - `x` and `y` have different lengths.
    - `length(x) < 2`.
    - `x` is not strictly increasing (for `:area`).
    - The normalization factor is non-positive.
- Throws an error if an unknown normalization method is provided.
 
# Examples
julia
x = range(0, 1, length=100)
y = exp.(-((x .- 0.5).^2) ./ 0.01)
 
y_area = normalize_profile(x, y)                # area normalization
y_max  = normalize_profile(x, y; method=:max)  # max normalization
"""
function normalize_profile(x::AbstractVector, y::AbstractVector; method::Symbol = :area)
    @assert length(x) == length(y) "x and y must have the same length."
    @assert length(x) ≥ 2 "Need at least two points."
 
    if method == :area
        Δx = x[2] - x[1]
        @assert Δx > 0 "x must be strictly increasing."
 
        norm = sum(y) * Δx
        @assert norm > 0 "Normalization is non-positive."
 
        return y ./ norm
 
    elseif method == :max
        m = maximum(y)
        @assert m > 0 "Maximum is non-positive."
 
        return y ./ m
 
    else
        error("Unknown normalization method: $method. Use :area or :max.")
    end
end