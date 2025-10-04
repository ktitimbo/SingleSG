# Quantum Magnetic Moment μF : electron(1/2)-nucleus(3/2)
"""
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
"""
function μF_effective(Ix,F,mF,p::AtomParams)
    # Promote to Float64 to avoid mixed-type arithmetic issues
    II = float(p.Ispin)
    γₙ = p.γn

    Ix, F, mF = promote(float(Ix), float(F), float(mF))

    # Energy scale and adimensional field
    ΔE = 2π * ħ * p.Ahfs * ( II + 1/2 )
    normalized_B = (γₑ-γₙ)*ħ / ΔE * BvsI(Ix) 
    
    # Validate quantum numbers
    is_F_upper = isapprox(F, II + 0.5; atol = 1e-12)
    is_F_lower = isapprox(F, II - 0.5; atol = 1e-12)
    if !(is_F_upper || is_F_lower)
        throw(ArgumentError("F must be I±1/2; got F=$F for I=$II"))
    end
    if mF < -F - 1e-12 || mF > F + 1e-12
        throw(ArgumentError("mF must be in [-F, F]; got mF=$mF for F=$F"))
    end

    # Common pieces
    ratio = γₙ / γₑ
    denom_arg = 1 - 4*mF/(2*II + 1) * normalized_B + normalized_B^2
    # Clamp tiny negative due to rounding
    denom = sqrt(max(denom_arg, 0.0))

    μF::Float64 = NaN  # <-- initialize
    if is_F_upper 
        if isapprox(mF,  F; atol = 1e-12) || isapprox(mF, -F; atol = 1e-12)
            s = sign(mF)
            μF = s * (gₑ/2) * (1 + 2*ratio*II) * μB
        else
            μF = gₑ * μB * ( mF*ratio + (1 - ratio)/denom * ( mF/(2*II + 1) - 0.5*normalized_B ) )
        end
    else # is_F_lower
        μF = gₑ * μB * ( mF*ratio - (1 - ratio)/denom * ( mF/(2*II + 1) - 0.5*normalized_B ) )
    end

    return Float64(μF)
end
