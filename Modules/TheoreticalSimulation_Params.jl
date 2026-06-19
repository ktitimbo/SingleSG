# ==============================================================================
# Parameter structs and constructors for the SG beamline simulation
#
# Three types are defined here:
#   AtomParams         — atomic species constants (mass, spin, hyperfine, ...)
#   EffusionParams     — precomputed beam-sampling geometry/thermal parameters
#   BeamEffusionParams — constructor: derives EffusionParams from geometry + T
#
# Plus one helper:
#   fmf_levels         — ordered (F, mF) pairs for a given species and J
# ==============================================================================


# ==============================================================================
# AtomParams
# ==============================================================================

"""
    AtomParams{T<:Real}

Immutable container for atomic constants used throughout the SG/beamline
calculations. Parametric in `T` so the entire pipeline can run in a consistent
floating-point precision (typically `Float64`, but `Float32` for GPU work).

# Fields
| Field    | Unit     | Description                          |
|----------|----------|--------------------------------------|
| `name`   | —        | Species tag, e.g. `:K39`, `:Rb87`   |
| `R`      | m        | van der Waals radius                 |
| `μn`     | J/T      | Nuclear magneton                     |
| `γn`     | 1/(s·T)  | Nuclear gyromagnetic ratio           |
| `Ispin`  | —        | Nuclear spin quantum number I        |
| `Ahfs`   | Hz       | Ground-state hyperfine constant      |
| `M`      | kg       | Atomic mass                          |

# Construction
Keyword constructor (via `Base.@kwdef`):
```julia
p = AtomParams(; name=:K39, R=…, μn=…, γn=…, Ispin=…, Ahfs=…, M=…)
```
Or from the `AtomicSpecies` lookup table:
```julia
p = AtomParams(:K39)          # Float64 (default)
p = AtomParams(:K39; T=Float32)
```
"""
Base.@kwdef struct AtomParams{T<:Real}
    name::Symbol = :unknown
    R::T         # van der Waals radius (m)
    μn::T        # nuclear magneton (J/T)
    γn::T        # nuclear gyromagnetic ratio (s⁻¹·T⁻¹); sign determines Zeeman ordering
    Ispin::T     # nuclear spin quantum number I
    Ahfs::T      # ground-state hyperfine constant (Hz)
    M::T         # atomic mass (kg)
end

"""
    AtomParams(atom; T=Float64) -> AtomParams{T}

Build an `AtomParams` by looking up `atom` in `AtomicSpecies.atoms`.

The lookup is expected to return an indexable collection where the positions map as:
    index 1 → R      (van der Waals radius, m)
    index 2 → μn     (nuclear magneton, J/T)
    index 3 → γn     (nuclear gyromagnetic ratio, s⁻¹·T⁻¹)
    index 4 → Ispin  (nuclear spin I)
    index 5 → (reserved / unused field in AtomicSpecies — e.g. abundance or Z)
    index 6 → Ahfs   (hyperfine constant, Hz)
    index 7 → M      (atomic mass, kg)

# Arguments
- `atom` — any key accepted by `AtomicSpecies.atoms` (e.g. `:K39`, `"Rb87"`).
- `T`    — element type for all numeric fields (default `Float64`).

# Requirements
`AtomicSpecies` must be loaded and provide `atoms(::Any)`.
"""
function AtomParams(atom; T=Float64)
    ai = AtomicSpecies.atoms(atom)
    return AtomParams{T}(
        name  = Symbol(atom),
        R     = T(ai[1]),
        μn    = T(ai[2]),
        γn    = T(ai[3]),
        Ispin = T(ai[4]),
        # ai[5] intentionally skipped — reserved field in AtomicSpecies
        Ahfs  = T(ai[6]),
        M     = T(ai[7]),
    )
end


# ==============================================================================
# EffusionParams
# ==============================================================================

"""
    EffusionParams

Precomputed parameters for effusive-beam velocity and angle sampling.
Derived from furnace/slit geometry and temperature by [`BeamEffusionParams`](@ref).

# Fields
| Field      | Unit  | Description                                      |
|------------|-------|--------------------------------------------------|
| `sinθmax`  | —     | sin of the beam's geometric half-angle (0–1)     |
| `α2`       | m²/s² | Thermal speed scale: `kB·T / M`                  |

`sinθmax` bounds the transverse velocity sampling: `v⊥_max = v · sinθmax`.
`α2` sets the scale of the Maxwell–Boltzmann speed distribution for this species.
"""
Base.@kwdef struct EffusionParams
    sinθmax::Float64   # sin(θ_max), dimensionless, ∈ [0, 1]
    α2::Float64        # kB·T/M (m²/s²) — thermal speed scale
end


# ==============================================================================
# BeamEffusionParams  (EffusionParams constructor from geometry)
# ==============================================================================

# Fudge factor on the geometric half-angle that widens the sampled cone slightly
# beyond the strict furnace→slit line-of-sight, accounting for diffraction and
# finite aperture effects. Calibrated empirically; treat as a tuning parameter.
const _BEAM_ANGLE_FUDGE = 1.25

"""
    BeamEffusionParams(xx_furnace, zz_furnace, xx_slit, zz_slit,
                       yy_FurnaceToSlit, T_furnace, p::AtomParams)
        -> EffusionParams

Compute effusive-beam sampling parameters from furnace/slit geometry and
furnace temperature.

# Geometry
The worst-case transverse offset between furnace and slit edges is:
```
Δxz = √[ (xx_furnace/2 + xx_slit/2)² + (zz_furnace/2 + zz_slit/2)² ]
```
The geometric half-angle (expanded by `$(_BEAM_ANGLE_FUDGE)×` empirical fudge) is:
```
θ_max = $_BEAM_ANGLE_FUDGE · atan(Δxz, yy_FurnaceToSlit)
```

# Arguments
- `xx_furnace, zz_furnace` — furnace aperture full-width in x and z (m).
- `xx_slit, zz_slit`       — collimating slit full-width in x and z (m).
- `yy_FurnaceToSlit`       — furnace-to-slit propagation distance (m).
- `T_furnace`              — furnace temperature (K).
- `p::AtomParams`          — atomic parameters; only `p.M` (mass, kg) is used here.

# Returns
`EffusionParams(sinθmax, α2)` where `α2 = kB·T_furnace / p.M`.
"""
@inline function BeamEffusionParams(
    xx_furnace::Real, zz_furnace::Real,
    xx_slit::Real,    zz_slit::Real,
    yy_FurnaceToSlit::Real,
    T_furnace::Real,
    p::AtomParams,
)
    # Corner-to-corner transverse offset (worst-case ray from furnace edge to slit edge)
    Δxz   = SVector(-xx_furnace/2, -zz_furnace/2) - SVector(xx_slit/2, zz_slit/2)
    θvmax = _BEAM_ANGLE_FUDGE * atan(norm(Δxz), yy_FurnaceToSlit)
    return EffusionParams(
        sinθmax = sin(θvmax),
        α2      = kb * T_furnace / p.M,
    )
end


# ==============================================================================
# fmf_levels — ordered (F, mF) pairs for a hyperfine manifold
# ==============================================================================

"""
    fmf_levels(p::AtomParams; J=1//2, stretched_only=false, Fsel=nothing)
        -> Vector{Tuple{Float64,Float64}}

Return an ordered list of hyperfine `(F, mF)` pairs for a species with nuclear
spin `I = p.Ispin` and electronic angular momentum `J`.

# Ordering convention ("bio-ordered")
The ordering follows the energy-level ordering driven by `sign(p.γn)`:

- `p.γn > 0` (e.g. ³⁹K): upper manifold (F = I + J) listed **mF descending**,
  lower manifold (F = I − J) listed **mF ascending**.
- `p.γn < 0`: directions are flipped (upper ascending, lower descending).

This ordering places the most energetically split (magnetically sensitive)
states first within each manifold, which is convenient for Zeeman-diagram plots.

# Keywords
- `J             :: Rational` — electronic angular momentum (default `1//2`).
- `stretched_only :: Bool`    — if `true`, return only the `(F, +F)` and `(F, -F)`
                                 stretched states for each manifold.
- `Fsel`                      — if provided, only return pairs for the manifold
                                 with `F ≈ Fsel` (floating-point safe comparison).

# Returns
`Vector{Tuple{Float64,Float64}}` of `(F, mF)` pairs in the specified order.

# Example
```julia
p  = AtomParams(:K39)
fm = fmf_levels(p)                     # all (F, mF) for K39, J=1/2
fm = fmf_levels(p; stretched_only=true) # only (F, ±F) states
fm = fmf_levels(p; Fsel=2)             # only F=2 manifold
```
"""
function fmf_levels(p::AtomParams; J=1//2, stretched_only::Bool=false, Fsel=nothing)
    I  = float(p.Ispin)
    Jf = float(J)

    # Manifolds from highest F down. Using a descending StepRange avoids the
    # intermediate allocation of `reverse(collect(range(...)))`.
    F_high = I + Jf
    F_low  = abs(I - Jf)
    Fs     = F_high:-1.0:F_low       # e.g. [2.0, 1.0] for K39 with J=1/2

    upper_desc = (p.γn > 0)          # ordering direction for upper manifold

    pairs = Tuple{Float64,Float64}[]

    for F in Fs
        # Skip manifolds not matching Fsel, if specified.
        # Done here (not before the loop) to avoid allocating a filtered array.
        if Fsel !== nothing && !isapprox(F, float(Fsel); atol=1e-12)
            continue
        end

        # Upper manifold (F ≥ I): descending if γn > 0, ascending if γn < 0.
        # Lower manifold (F < I): ascending  if γn > 0, descending if γn < 0.
        dir_desc = upper_desc ? (F >= I) : (F < I)

        if stretched_only
            # Only the two extremal mF values per manifold.
            # Maintain the same direction convention so the ordering is consistent
            # with the full list (useful when overlaying stretched vs. full plots).
            if dir_desc
                push!(pairs, (F,  F))
                push!(pairs, (F, -F))
            else
                push!(pairs, (F, -F))
                push!(pairs, (F,  F))
            end
        else
            mF_range = dir_desc ? (F:-1.0:-F) : (-F:1.0:F)
            for mF in mF_range
                push!(pairs, (F, mF))
            end
        end
    end

    return pairs
end