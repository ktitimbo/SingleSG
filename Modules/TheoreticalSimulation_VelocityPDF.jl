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