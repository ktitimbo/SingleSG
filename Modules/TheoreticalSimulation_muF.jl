# ==============================================================================
# Quantum Magnetic Moment μF : electron (S=1/2) – nucleus (I=3/2) system
#
# This file implements the effective magnetic moment μF for each hyperfine
# sublevel (F, mF) of an alkali atom in the Breit–Rabi regime.
#
# Physical background
# -------------------
# In an external magnetic field B the hyperfine levels split according to the
# Breit–Rabi formula. The effective magnetic moment is the field derivative of
# each energy eigenvalue:
#
#       μF_eff(B) = -dE(F,mF,B)/dB
#
# For a spin-1/2 electron coupled to a nucleus of spin I, the two hyperfine
# manifolds are F = I ± 1/2. All expressions below follow directly from
# differentiating the Breit–Rabi eigenvalues.
#
# The adimensional field parameter used throughout is:
#
#       x ≡ (γₑ - γₙ)·ħ / ΔE · B,   ΔE = 2π·ħ·Ahfs·(I + 1/2)
#
# so that x = 0 is the pure-hyperfine (zero-field) limit and |x| → ∞ is the
# Paschen-Back limit.
#
# Two entry points are provided:
#   μF_effective    — takes coil current Ix; intended for call sites that only
#                     know the current and call this O(millions) of times at
#                     varying Ix. BvsI(Ix) is evaluated internally.
#   μF_effective_B  — takes field B0 directly; use this inside hot loops where
#                     B0 is already known or has been precomputed once for a
#                     fixed current, avoiding redundant BvsI calls.
# ==============================================================================


"""
    μF_effective(Ix, F, mF, p::AtomParams) -> Float64

Effective magnetic moment μF for hyperfine state (F, mF) at coil current `Ix`.

The magnetic field is obtained internally via `BvsI(Ix)`. This entry point is
intended for call sites where only the coil current is known. If you already
have B0 (e.g. after precomputing `BvsI` once for a fixed current before a
particle loop), prefer [`μF_effective_B`](@ref) to avoid redundant field lookups.

# Arguments
- `Ix  :: Real` — coil current (A); converted to field by the calibration map `BvsI`.
- `F   :: Real` — total angular-momentum quantum number; must equal `p.Ispin ± 1/2`.
- `mF  :: Real` — magnetic quantum number; must satisfy `-F ≤ mF ≤ F`.
- `p   :: AtomParams` — atom parameter struct carrying `Ispin`, `Ahfs`, `γn`.

# Physics
The adimensional field parameter is

    x = (γₑ - γₙ)·ħ / [2π·ħ·Ahfs·(I + 1/2)] · BvsI(Ix)

The effective moment is then:

- Upper manifold (F = I + 1/2):
    - Stretched states mF = ±F (field-independent analytic form):
        μF = sign(mF)·(gₑ/2)·(1 + 2·γₙ/γₑ·I)·μB
    - All other mF:
        μF = gₑ·μB·[ mF·(γₙ/γₑ) + (1 - γₙ/γₑ)/D · (mF/(2I+1) - x/2) ]

- Lower manifold (F = I - 1/2):
        μF = gₑ·μB·[ mF·(γₙ/γₑ) - (1 - γₙ/γₑ)/D · (mF/(2I+1) - x/2) ]

where the common denominator is

    D = sqrt( max(1 - 4·mF/(2I+1)·x + x², 0) )

The `max(..., 0)` clamp prevents `NaN` from floating-point rounding when the
argument should be exactly zero (stretched states at finite field).

# Returns
`Float64` effective magnetic moment in Joules/Tesla (same units as `μB`).

# Throws
- `ArgumentError` if F ≠ I ± 1/2.
- `ArgumentError` if mF ∉ [-F, F].
"""
@inline function μF_effective(Ix::Real, F::Real, mF::Real, p::AtomParams)

    # ── Promote quantum numbers to Float64 together (they share arithmetic).
    #    Ix is converted separately — it is physically unrelated to F and mF.
    II      = float(p.Ispin)
    γₙ      = p.γn
    F, mF   = promote(float(F), float(mF))

    # ── Adimensional field parameter x ────────────────────────────────────
    ΔE           = 2π * ħ * p.Ahfs * (II + 0.5)        # hyperfine splitting (J)
    normalized_B = (γₑ - γₙ) * ħ / ΔE * BvsI(float(Ix))   # dimensionless x

    # ── Validate quantum numbers ───────────────────────────────────────────
    is_F_upper = isapprox(F, II + 0.5; atol=1e-12)
    is_F_lower = isapprox(F, II - 0.5; atol=1e-12)

    if !(is_F_upper || is_F_lower)
        throw(ArgumentError("F must be I±1/2; got F=$F for I=$II"))
    end
    if mF < -F - 1e-12 || mF > F + 1e-12
        throw(ArgumentError("mF must be in [-F, F]; got mF=$mF for F=$F"))
    end

    # ── Common prefactor ──────────────────────────────────────────────────
    ratio = γₙ / γₑ    # γₙ/γₑ ≪ 1 for most nuclei

    # ── Compute and return μF ─────────────────────────────────────────────
    if is_F_upper

        if isapprox(mF, F; atol=1e-12) || isapprox(mF, -F; atol=1e-12)
            # Stretched states: field-independent analytic form.
            # The square-root denominator equals 1 exactly at these states,
            # and the expression reduces to a constant.
            return sign(mF) * (gₑ/2) * (1 + 2*ratio*II) * μB
        end

        # Non-stretched upper states: full Breit–Rabi derivative.
        # `denom` is only computed here, after the early return above.
        denom_arg = 1 - 4*mF/(2*II + 1) * normalized_B + normalized_B^2
        denom     = sqrt(max(denom_arg, 0.0))
        return Float64(gₑ * μB * (mF*ratio + (1 - ratio)/denom * (mF/(2*II + 1) - 0.5*normalized_B)))

    else  # is_F_lower — no stretched-state simplification in the lower manifold

        denom_arg = 1 - 4*mF/(2*II + 1) * normalized_B + normalized_B^2
        denom     = sqrt(max(denom_arg, 0.0))
        return Float64(gₑ * μB * (mF*ratio - (1 - ratio)/denom * (mF/(2*II + 1) - 0.5*normalized_B)))

    end
    # NOTE: No NaN fallback — every valid (F, mF) pair hits an explicit return.
    # A future missing branch will surface as a compiler warning, not a silent NaN.
end


"""
    μF_effective_B(B0, F, mF, p::AtomParams) -> Float64

Effective magnetic moment μF for hyperfine state (F, mF) at field strength `B0`.

This entry point takes the magnetic field directly and is the preferred choice
inside hot particle loops where B0 has already been precomputed (e.g. once per
coil current before iterating over millions of particles), avoiding redundant
`BvsI` evaluations. For call sites that only know the coil current, use
[`μF_effective`](@ref) instead.

# Arguments
- `B0  :: Real` — external magnetic field magnitude (Tesla).
- `F   :: Real` — total angular-momentum quantum number; must equal `p.Ispin ± 1/2`.
- `mF  :: Real` — magnetic quantum number; must satisfy `-F ≤ mF ≤ F`.
- `p   :: AtomParams` — atom parameter struct carrying `Ispin`, `Ahfs`, `γn`.

# Physics
The adimensional field parameter is

    x = (γₑ - γₙ)·ħ / [2π·ħ·Ahfs·(I + 1/2)] · B0

The effective moment is then:

- Upper manifold (F = I + 1/2):
    - Stretched states mF = ±F (field-independent analytic form):
        μF = sign(mF)·(gₑ/2)·(1 + 2·γₙ/γₑ·I)·μB
    - All other mF:
        μF = gₑ·μB·[ mF·(γₙ/γₑ) + (1 - γₙ/γₑ)/D · (mF/(2I+1) - x/2) ]

- Lower manifold (F = I - 1/2):
        μF = gₑ·μB·[ mF·(γₙ/γₑ) - (1 - γₙ/γₑ)/D · (mF/(2I+1) - x/2) ]

where the common denominator is

    D = sqrt( max(1 - 4·mF/(2I+1)·x + x², 0) )

The `max(..., 0)` clamp prevents `NaN` from floating-point rounding when the
argument should be exactly zero (stretched states at finite field).

# Returns
`Float64` effective magnetic moment in Joules/Tesla (same units as `μB`).

# Throws
- `ArgumentError` if F ≠ I ± 1/2.
- `ArgumentError` if mF ∉ [-F, F].
"""
@inline function μF_effective_B(B0::Real, F::Real, mF::Real, p::AtomParams)

    # ── Promote quantum numbers to Float64 together (they share arithmetic).
    #    B0 is converted separately — it is physically unrelated to F and mF.
    II      = float(p.Ispin)
    γₙ      = p.γn
    F, mF   = promote(float(F), float(mF))

    # ── Adimensional field parameter x ────────────────────────────────────
    ΔE           = 2π * ħ * p.Ahfs * (II + 0.5)        # hyperfine splitting (J)
    normalized_B = (γₑ - γₙ) * ħ / ΔE * float(B0)     # dimensionless x

    # ── Validate quantum numbers ───────────────────────────────────────────
    is_F_upper = isapprox(F, II + 0.5; atol=1e-12)
    is_F_lower = isapprox(F, II - 0.5; atol=1e-12)

    if !(is_F_upper || is_F_lower)
        throw(ArgumentError("F must be I±1/2; got F=$F for I=$II"))
    end
    if mF < -F - 1e-12 || mF > F + 1e-12
        throw(ArgumentError("mF must be in [-F, F]; got mF=$mF for F=$F"))
    end

    # ── Common prefactor ──────────────────────────────────────────────────
    ratio = γₙ / γₑ    # γₙ/γₑ ≪ 1 for most nuclei

    # ── Compute and return μF ─────────────────────────────────────────────
    if is_F_upper

        if isapprox(mF, F; atol=1e-12) || isapprox(mF, -F; atol=1e-12)
            # Stretched states: field-independent analytic form.
            # The square-root denominator equals 1 exactly at these states,
            # and the expression reduces to a constant.
            return sign(mF) * (gₑ/2) * (1 + 2*ratio*II) * μB
        end

        # Non-stretched upper states: full Breit–Rabi derivative.
        # `denom` is only computed here, after the early return above.
        denom_arg = 1 - 4*mF/(2*II + 1) * normalized_B + normalized_B^2
        denom     = sqrt(max(denom_arg, 0.0))
        return Float64(gₑ * μB * (mF*ratio + (1 - ratio)/denom * (mF/(2*II + 1) - 0.5*normalized_B)))

    else  # is_F_lower — no stretched-state simplification in the lower manifold

        denom_arg = 1 - 4*mF/(2*II + 1) * normalized_B + normalized_B^2
        denom     = sqrt(max(denom_arg, 0.0))
        return Float64(gₑ * μB * (mF*ratio - (1 - ratio)/denom * (mF/(2*II + 1) - 0.5*normalized_B)))

    end
    # NOTE: No NaN fallback — every valid (F, mF) pair hits an explicit return.
    # A future missing branch will surface as a compiler warning, not a silent NaN.
end