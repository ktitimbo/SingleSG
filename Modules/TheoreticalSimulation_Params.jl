"""
    AtomParams{T}

Container for atomic constants used in the SG/beamline calculations.

Fields
- `name::Symbol` — species tag (e.g. `:K39`, `:Rb87`).
- `R::T`         — van der Waals radius (m).
- `μn::T`        — nuclear magneton (J/T).
- `γn::T`        — nuclear gyromagnetic ratio (s⁻¹·T⁻¹).
- `Ispin::T`     — nuclear spin quantum number `I`.
- `Ahfs::T`      — hyperfine constant (Hz).
- `M::T`         — atomic mass (kg).

Notes
- Defined with `Base.@kwdef`, so you can construct with keywords:
  `AtomParams(; name=:K39, R=…, μn=…, …)`.
"""
Base.@kwdef struct AtomParams{T<:Real}
    name::Symbol = :unknown
    R::T         # van der Waals radius
    μn::T        # nuclear magneton
    γn::T        # nuclear gyromagnetic ratio
    Ispin::T     # nuclear spin I
    Ahfs::T      # hyperfine constant
    M::T         # mass
end

"""
    AtomParams(atom; T=Float64) -> AtomParams{T}

Build an `AtomParams` from the lookup `AtomicSpecies.atoms(atom)`. The lookup is
expected to return a tuple/array where positions `(1,2,3,4,6,7)` correspond to
`(R, μn, γn, Ispin, Ahfs, M)`. Values are converted to the element type `T`
(default `Float64`) and `name` is set to `Symbol(atom)`.

Requirements
- `AtomicSpecies` must be loaded and provide `atoms(::Any)`.
"""
AtomParams(atom; T=Float64) = begin
    ai = AtomicSpecies.atoms(atom)
    AtomParams{T}(
        name = Symbol(atom),
        R    = T(ai[1]),
        μn   = T(ai[2]),
        γn   = T(ai[3]),
        Ispin= T(ai[4]),
        Ahfs = T(ai[6]),
        M    = T(ai[7]),
    )
end

"""
    EffusionParams{T}

Container of precomputed beam-sampling parameters.

Fields
- `sinθmax::Float64` — max sine of the polar angle (0 ≤ sinθmax ≤ 1)
- `α2::Float64`      — speed scale `kB*T/M` (m²/s²)
"""
Base.@kwdef struct EffusionParams
    sinθmax::Float64
    α2::Float64
end

"""
    BeamEffusionParams(xx_furnace, zz_furnace, xx_slit, zz_slit, yy_FurnaceToSlit, T, p::AtomParams)
        -> EffusionParams

Compute effusive-beam parameters from furnace/slit geometry and temperature.

Definitions
- `Δxz = (-xx_furnace/2, -zz_furnace/2) − (xx_slit/2, zz_slit/2)` (m).
- `θvmax = 1.25 * atan(norm(Δxz), yy_FurnaceToSlit)` (rad) — geometric half-angle
  with a 1.25 fudge factor.
- Returns `EffusionParams(sin(θvmax), α2)` with `α2 = kb*T/p.M`.

Arguments
- `xx_furnace, zz_furnace` — furnace aperture size in x/z (m).
- `xx_slit, zz_slit`       — slit aperture size in x/z (m).
- `yy_FurnaceToSlit`       — furnace→slit separation (m).
- `T`                      — furnace temperature (K).
- `p::AtomParams`          — provides the mass via `p.M` (kg).

Assumptions
- `kb` (Boltzmann constant) is defined in scope.
- Units are SI (m, K, kg).

Returns
- `EffusionParams` ready to use in velocity samplers.
"""
@inline function BeamEffusionParams(xx_furnace, zz_furnace, xx_slit, zz_slit, yy_FurnaceToSlit, T, p::AtomParams )
    Δxz   = SVector(-xx_furnace/2, -zz_furnace/2) - SVector(xx_slit/2, zz_slit/2)
    θvmax = 1.25 * atan(norm(Δxz), yy_FurnaceToSlit)
    return EffusionParams(sin(θvmax), kb*T/p.M)
end