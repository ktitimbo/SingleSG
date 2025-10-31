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
    AtomicBeamVelocity_v2(rng, p::EffusionParams) -> SVector{3,Float64}

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
- Uses `Distributions.Gamma(3/2, 1)`. Constructing the `Gamma` object inside the
  call can allocate; consider caching it at module scope if this is performance-critical.
- `p.α2 = k_B T / M` controls the speed scale; `p.sinθmax` controls the angular cone.
"""
@inline function AtomicBeamVelocity_v2(rng,p::EffusionParams)::SVector{3,Float64} 
    ϕ = TWOπ * rand(rng)
    θ = asin(p.sinθmax * sqrt(rand(rng)))
    v = sqrt(2 .* p.α2 .* rand(rng, Gamma(3/2,1.0)))
    sθ = sin(θ); cθ = cos(θ); sϕ = sin(ϕ); cϕ = cos(ϕ)
    return SVector(v*sθ*sϕ, v*cθ, v*sθ*cϕ)
end

"""
    getProbDist_v3(μ, dBzdz, zd, p::AtomParams, q::EffusionParams;
                   wfurnace::Float64=default_z_furnace,
                   npts::Int=2001,
                   pdf::Symbol=:point) -> Vector{Float64}

Closed-form Stern–Gerlach screen profile for a constant field gradient.

Computes the transverse probability density along the screen positions `zd`
(optionally averaged over a finite furnace width). Uses simple kinematics in a
piecewise-drift geometry and an effusive-beam speed scale.

# Arguments
- `μ::Float64`: effective magnetic moment (J/T). **Sign matters**; if `μ < 0`,
  the function flips the `zd` axis internally so the returned profile is oriented
  consistently.
- `dBzdz::Float64`: magnetic-field gradient (T/m), assumed constant over the SG region.
- `zd::AbstractVector`: screen coordinates (m) where the profile is evaluated.
- `p::AtomParams`: must provide `p.M` (mass, kg).
- `q::EffusionParams`: must provide `q.α2` so that `β² = 2α2` (with `β` the
  effusive-beam speed scale).
- `wfurnace`: source width (m). Used only when `pdf = :finite`.
- `npts`: number of trapezoid samples across `[-wfurnace/2, +wfurnace/2]`
  when `pdf = :finite` (recommend an odd value to include `z0 = 0`).
- `pdf`: `:point` (infinitesimal source) or `:finite` (averaged over furnace width).

# Geometry (read from globals)
Uses (all in meters) `default_y_FurnaceToSlit`, `default_y_SlitToSG`,
`default_y_SG`, `default_y_SGToScreen`, and entrance slit width `default_z_slit`.

Let `LOS = y_FurnaceToSlit`, `LSG = y_SG`, `LSGD = y_SGToScreen`,
`Ltot = LOS + y_SlitToSG + LSG + LSGD`, and `ℓ = Ltot/LOS`.

# Model (per source offset `z0`)
- Acceleration parameter: `a = (μ * dBzdz / p.M) * LSG * (LSG + 2LSGD) / 2`
- Speed scale: `β² = 2 q.α2`, define `c = a / β²`
- Denominators:
  - `d1 = z − z0 − (w/2 − z0) * ℓ`
  - `d2 = z − z0 + (w/2 + z0) * ℓ`, with `w = default_z_slit`
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
- Asserts `pdf ∈ (:point, :finite)` and basic input sanity.

# Example
```julia
zd = range(-12.5e-3, 12.5e-3; length=20_001)
μ    = μB_eff           # your effective moment (J/T)
grad = GvsI(I0)        # dB/dz for current I0 (T/m)

prof_point  = getProbDist_v3(μ, grad, zd, K39_params, effusion_params; pdf=:point)
prof_finite = getProbDist_v3(μ, grad, zd, K39_params, effusion_params;
                           wfurnace=100e-6, npts=2001, pdf=:finite)
"""
function getProbDist_v3(μ::Float64, dBzdz::Float64, zd::AbstractVector, p::AtomParams, q::EffusionParams;
                     wfurnace::Float64=default_z_furnace, npts::Int=5001, pdf::Symbol=:point)
    
    @assert pdf === :point || pdf === :finite "pdf must be :point or :finite"

    # --- Geometry (m) ---
    LOS   = default_y_FurnaceToSlit
    LSSG  = default_y_SlitToSG
    LSG   = default_y_SG
    LSGD  = default_y_SGToScreen
    @assert LOS >= 0          "Furnace-to-slit distance (LOS) must be ≥ 0."
    @assert LSSG ≥ 0 && LSG ≥ 0 && LSGD ≥ 0  "Segment lengths must be ≥ 0."

    Ltot  = LOS + LSSG + LSG + LSGD
    lfrac = Ltot / LOS
    inv_lfrac  = inv(lfrac)

    # --- Slit width at SG entrance (m) ---
    w     = default_z_slit
    halfw = w/2

    # Mirror if μ < 0 (preserves shape)
    zd    = μ < 0 ? -zd : zd
    dBzdz = μ < 0 ? -dBzdz : dBzdz

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
            zi = zd[i]
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
                t -= -exp(-p2) * (p2 + 1.0) * inv_lfrac
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
        @assert npts ≥ 3 "npts must be ≥ 3 and odd"

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

ProbDist_convolved(z::AbstractVector{<:Real},z_pdf::AbstractVector{<:Real}, w_width::Real) = smooth_profile(z, z_pdf, w_width)



"""
    ProbDist_convolved(z::AbstractVector{<:Real},
                       z_pdf::AbstractVector{<:Real},
                       w_width::Real)

Convolve/smooth `z_pdf` sampled at locations `z` with kernel width `w_width`.
Delegates to `smooth_profile`. Throws if lengths mismatch.
"""
@inline function ProbDist_convolved(z::AbstractVector{T},
                                    z_pdf::AbstractVector{T},
                                    w_width::T) where {T<:Real}
    @boundscheck length(z) == length(z_pdf) || throw(ArgumentError("z and z_pdf must have same length"))
    return smooth_profile(z, z_pdf, w_width)
end

@inline function ProbDist_convolved!(out::AbstractVector,
                                     z::AbstractVector,
                                     z_pdf::AbstractVector,
                                     w_width::Number)
    @boundscheck length(out) == length(z) == length(z_pdf) ||
        throw(ArgumentError("lengths must match"))
    tmp = smooth_profile(z, z_pdf, w_width)   # may be Dual eltype
    copyto!(out, tmp)                         # out must accept that eltype
    return out
end
