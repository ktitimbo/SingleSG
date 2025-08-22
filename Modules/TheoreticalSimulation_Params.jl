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


"""
    fm_pairs_biordered(p::AtomParams; J=1//2, stretched_only=false, Fsel=nothing)
        -> Vector{Tuple{Float64,Float64}}

Construct hyperfine `(F, mF)` pairs using `I = p.Ispin` and ordering
driven by `sign(p.γn)`:

- If `p.γn > 0`: for each manifold, those with `F ≥ I` are listed **mF descending**,
  and those with `F < I` are **mF ascending** (upper desc / lower asc for J=1/2).
- If `p.γn < 0`: the directions are flipped.

Keywords
- `J`                — electronic angular momentum (default `1//2`).
- `stretched_only`   — if `true`, return only `(F, ±F)` for each selected `F`.
- `Fsel`             — if provided, keep only that F (≈ comparison).

Returns a vector of `(F, mF)` as `Float64`.
"""
function fmf_pairs(p::AtomParams; J=1//2, stretched_only::Bool=false, Fsel=nothing)
    I = float(p.Ispin)
    Jf = float(J)
    Fs = reverse(collect(range(abs(I - Jf), I + Jf; step=1.0)))
    if Fsel !== nothing
        Fs = [F for F in Fs if isapprox(F, float(Fsel); atol=1e-12)]
    end

    upper_desc = (p.γn > 0)     # if false, flip directions

    pairs = Tuple{Float64,Float64}[]
    for F in Fs
        # Decide direction for this manifold
        dir_desc = upper_desc ? (F >= I) : (F < I)
        if stretched_only
            if dir_desc
                push!(pairs, (F,  F)); push!(pairs, (F, -F))
            else
                push!(pairs, (F, -F)); push!(pairs, (F,  F))
            end
        else
            if dir_desc
                for mF in F:-1.0:-F
                    push!(pairs, (F, mF))
                end
            else
                for mF in -F:1.0:F
                    push!(pairs, (F, mF))
                end
            end
        end
    end
    return pairs
end

