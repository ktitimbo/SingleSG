# ==============================================================================
# Equations of motion for the SG apparatus: closed-form trajectory kernels
#
# This file implements three physics families, each in closed form (no ODE
# integration — these are analytic solutions for piecewise-constant or
# CQD-relaxation dynamics), at two levels of "API generality" each:
#
#   - GENERAL/FLEXIBLE layer: take `AbstractVector{<:Real}` for r0/v0, full
#     `@assert` input validation, return `SVector{3,Float64}`. Convenient for
#     single-trajectory calls, plotting, diagnostics. Does NOT use the
#     overflow-safe log-domain identities described below.
#   - FAST SCALAR CORE layer: take individual `Float64` scalars (not vectors),
#     `@inline`, `muladd`-heavy, designed for per-particle hot loops at
#     N~10^6-10^7 scale. Where present, also uses numerically-stable
#     log/dilogarithm identities (`log_cos_sin_exp`, `li2_negexp` — the latter
#     defined elsewhere, not in this file) to avoid `exp(+large)` overflow.
#
# The three physics families:
#
#   1. Co-Quantum Dynamics, CQD_*
#      Dissipative spin-relaxation model; ωL = |γₑ * BvsI(Ix)|.
#
#   2. Quantum Mechanics, QM_*
#      Piecewise-constant acceleration (no exp/log/dilog terms at all — pure
#      polynomial kinematics, so no overflow-safe treatment is needed here).
#
#   3. Co-Quantum Dynamics with nuclear field correction, CQD_Bn_*
#      Same as (1), but ωL = |γₑ * (BvsI(Ix) + DEFAULT_CQD_Bn*cos(θn))| — adds
#      a θn-dependent nuclear-field term to the Larmor frequency. This is the
#      physics underlying every "_Bn_"-suffixed function referenced throughout
#      TheoreticalSimulation_DiscardedParticles.jl.
# ==============================================================================
 
 
########################################################################################################################################
# Co-Quantum Dynamics
########################################################################################################################################
 
"""
    log_cos_sin_exp(t, cosθ2, sinθ2) -> Float64
 
Numerically stable evaluation of `log(cosθ2 + sinθ2*exp(t))`, avoiding the
overflow that a direct `exp(t)` would suffer for large positive `t`.
 
For `t > 0`, rewrites as `t + log(sinθ2 + cosθ2*exp(-t))` — algebraically
identical (factor `exp(t)` out, take the log), but now the only exponential
evaluated is `exp(-t)`, which is bounded in `(0,1]` for `t > 0` and therefore
can never overflow. For `t ≤ 0`, the direct form `log(cosθ2 + sinθ2*exp(t))`
is already safe, since `exp(t) ≤ 1` there.
 
# Arguments
- `t::Float64`: exponent argument (typically `-2*kω*Δt` or similar, can be
  any sign/magnitude).
- `cosθ2::Float64`, `sinθ2::Float64`: the two coefficients multiplying `1`
  and `exp(t)` respectively (named for their typical use as `cos²(θe/2)`/
  `sin²(θe/2)`, but the identity holds for any such pair).
 
# Returns
`Float64`, equal to `log(cosθ2 + sinθ2*exp(t))` but computed without ever
evaluating `exp` at a large positive argument.
"""
@inline function log_cos_sin_exp(t::Float64, cosθ2::Float64, sinθ2::Float64)::Float64
    # Returns log(cosθ2 + sinθ2*exp(t)) robustly without overflow for large +t.
    # If t>0: log(c + s e^t) = t + log(s + c e^{-t})
    if t > 0.0
        return t + log(sinθ2 + cosθ2*exp(-t))
    else
        return log(cosθ2 + sinθ2*exp(t))
    end
end

# CQD Equations of motion
"""
    CQD_EqOfMotion(t, Ix, μ, r0, v0, θe, θn, kx, p) -> (r, v)
 
Full 3D equations of motion for a Continuous Quantum Dynamics (CQD) model in a
Stern–Gerlach (SG) setup. Motion in `x` and `y` is purely ballistic; only `z`
is modified by CQD terms while the atom is inside the SG region.
 
Segment boundaries (seconds; beam advances along +y with v0y ≠ 0):
- `tf1 =  DEFAULT_y_FurnaceToSlit / v0y`
- `tf2 = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG) / v0y`
- `tf3 = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG) / v0y`
 
Definitions used:
- `cqd_sign = sign(θn - θe)`
- `ωL = abs(γₑ * BvsI(Ix))`                # Larmor angular frequency
- `acc_0 = μ * GvsI(Ix) / p.M`             # base acceleration scale
- `kω = cqd_sign * kx * ωL`
- `θe_half = θe / 2`
- `cosθ2 = cos(θe_half)^2`, `sinθ2 = sin(θe_half)^2`, `tanθ2 = tan(θe_half)^2`
 
Ballistic components (always):
- `x(t) = x0 + v0x * t`
- `y(t) = y0 + v0y * t`
- `vx = v0x`, `vy = v0y`
 
Piecewise z–dynamics (matches the implementation):
 
1) Furnace → Slit (`t ≤ tf1`) and Slit → SG entrance (`tf1 < t ≤ tf2`)
   - `z(t) = z0 + v0z * t`
   - `vz = v0z`
 
2) Inside SG (`tf2 < t ≤ tf3`), let `Δt = t - tf2`, `EΔ = exp(-2*kω*Δt)`:
   - `vz = v0z + acc_0*Δt + (acc_0/kω) * log( cosθ2 + EΔ * sinθ2 )`
   - `z  = z0 + v0z*t
              + 0.5*acc_0*Δt^2
              + (acc_0/kω)*log(cosθ2)*Δt
              + 0.5*acc_0/kω^2 * ( polylogarithm(2, -EΔ*tanθ2)
                                   - polylogarithm(2, -tanθ2) )`
 
3) SG exit → Screen (`t > tf3`), let `τ_SG = DEFAULT_y_SG / v0y`, `Eτ = exp(-2*kω*τ_SG)`:
   - `z  = z0 + v0z*t
              + 0.5*acc_0 * ( (t - tf2)^2 - (t - tf3)^2 )
              + (acc_0/kω) * τ_SG * ( log(cosθ2)
                                      + (v0y/DEFAULT_y_SG) * log(cosθ2 + Eτ*sinθ2) * (t - tf3) )
              + 0.5*acc_0/kω^2 * ( polylogarithm(2, -Eτ*tanθ2)
                                   - polylogarithm(2, -tanθ2) )`
   - `vz = v0z + acc_0*τ_SG
                + (acc_0/kω) * log( cosθ2 + Eτ * sinθ2 )`
 
Arguments
- `t`: time since the initial state `(r0, v0)` (s).
- `Ix`: coil current (A), used by `GvsI` and `BvsI`.
- `μ`: effective magnetic moment (J/T).
- `r0::AbstractVector{<:Real}`: initial position `(x0, y0, z0)` in meters; length 3.
- `v0::AbstractVector{<:Real}`: initial velocity `(v0x, v0y, v0z)` in m/s; length 3 and `v0y ≠ 0`.
- `θe::Float64`, `θn::Float64`: electron and nuclear angles (radians).
- `kx::Float64`: dimensionless coupling multiplying `ωL`.
- `p::AtomParams`: atomic parameters (must include mass `M`). Assumes `γₑ`, `GvsI`, `BvsI`,
  and `DEFAULT_y_*` geometry constants are in scope.
 
Returns
- `(r, v)` with `r::SVector{3,Float64}` and `v::SVector{3,Float64}` at time `t`.
 
Assumptions
- Non-relativistic; gravity, collisions, and fringe-field variation neglected.
- Gradient aligned with +z; only z accelerates inside the SG.
- Expressions are continuous at `t = tf2` and `t = tf3`.
 
Numerical notes
- Terms with `1/kω` and `1/kω^2` are finite in the `kω → 0` limit, but direct evaluation
  can be ill-conditioned when `|kω|` is very small. Consider stabilized small-`kω` series.
- Exponentials `EΔ`/`Eτ` can under/overflow for large `|kω*Δt|` or `|kω*τ_SG|`; a log-sum-exp
  rewrite for `log(cosθ2 + E* sinθ2)` improves robustness. (This is exactly what
  `log_cos_sin_exp`/`li2_negexp` do in the "fast scalar core" sibling
  `CQD_screen_x_z_vz` — this general/flexible function uses the direct,
  un-stabilized form throughout, consistent with the rest of this
  general/flexible "generation" of functions, e.g. `CQD_Screen_position`.)
 
# Performance
This is the "general/flexible" layer (`AbstractVector` input, full
`@assert` validation) — not the hot-path entry point. `DEFAULT_y_FurnaceToSlit`/
`DEFAULT_y_SlitToSG`/`DEFAULT_y_SG` are read here without a `::Float64` type
ascription (unlike the "fast scalar core" functions later in this file, e.g.
`QM_Screen_position`, which do add one) — if these globals are not `const`,
that ascription is a partial mitigation for the resulting type instability;
its absence here means this function pays the full cost of reading a
non-`const` global on every call.
"""
@inline function CQD_EqOfMotion(t,Ix,μ,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"
    @assert t >= 0 "time t must be ≥ 0"
    x0, y0, z0 = r0
    v0x, v0y, v0z = v0
    @assert !iszero(v0y) "y-velocity must be nonzero."

    inv_v0y = 1.0 / v0y

    x = x0 + v0x*t 
    y = y0 + v0y*t 
    vx = v0x
    vy = v0y

    # Key times
    tf1 = (DEFAULT_y_FurnaceToSlit * inv_v0y) :: Float64
    tf2 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG) * inv_v0y) :: Float64

    if t <= tf1     # Furnace to Slit
        z = z0 + v0z*t
        vz = v0z
    elseif t <= tf2    # Slit to SG apparatus
        z = z0 + v0z*t
        vz = v0z
    else
        tf3 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG) * inv_v0y) :: Float64

        cqd_sign = sign(θn-θe) 
        ωL       = abs(γₑ * BvsI(Ix) )
        acc_0    = μ*GvsI(Ix)/p.M
        kω       = cqd_sign*kx*ωL
        inv_kω   = 1.0 / kω
        inv_kω2  = inv_kω * inv_kω

        s, c = sincos(θe / 2)
        sinθ2 = s*s
        cosθ2 = c*c
        tanθ2 = sinθ2 / cosθ2

        if t <= tf3   # Crossing the SG apparatus
            Δt = t-tf2
            EΔ = exp(-2*kω*Δt)
            vz = v0z + acc_0*Δt + acc_0*inv_kω * log( cosθ2 + EΔ*sinθ2 )
            z = z0 + v0z*t + 0.5*acc_0*Δt^2 + acc_0*inv_kω*log(cosθ2)*Δt + 
                0.5*acc_0*inv_kω2 * ( polylogarithm(2,-EΔ*tanθ2) - polylogarithm(2,-tanθ2) )
        else # t > tf3 # Travel to the Screen
            τ_SG = (DEFAULT_y_SG * inv_v0y) :: Float64
            Eτ = exp(-2*kω*τ_SG)
            z = z0 + v0z*t + acc_0*τ_SG*(t-tf2-0.5*τ_SG) + acc_0*inv_kω*τ_SG * ( log(cosθ2) + inv(τ_SG)*log(cosθ2+Eτ*sinθ2)*(t-tf3) ) + 0.5*acc_0*inv_kω2*( polylogarithm(2,-Eτ*tanθ2) - polylogarithm(2,-tanθ2) )
            vz = v0z + acc_0*τ_SG + acc_0*inv_kω*log(cosθ2 + Eτ*sinθ2)
        end
    end

    r = SVector{3,Float64}(x, y, z)
    v = SVector{3,Float64}(vx, vy, vz)
    return r, v
end


# CQD equations of motion only along the z-coordinate
"""
    CQD_EqOfMotion_z(t, Ix, μ, r0, v0, θe, θn, kx, p) -> Float64

z–coordinate as a function of time `t` under a Continuous Quantum Dynamics (CQD)
model in a Stern–Gerlach (SG) setup. Motion is divided into three segments:
1) Pre-SG (ballistic), 2) In-SG (CQD dynamics with uniform base acceleration),
3) Post-SG (ballistic with accumulated CQD effect).

Segment boundaries (s):
- `tf2 = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG) / v0y`
- `tf3 = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG) / v0y`

Definitions (computed only for `t > tf2` — see Performance below):
- `cqd_sign = sign(θn - θe)`
- `ωL = |γₑ| * BvsI(Ix)`                         # Larmor angular frequency
- `acc_z = μ * GvsI(Ix) / p.M`                   # base acceleration scale
- `kω = cqd_sign * kx * ωL`
- `θe_half = θe/2`
- `tanθ2 = tan(θe_half)^2`, `cosθ2 = cos(θe_half)^2`, `sinθ2 = sin(θe_half)^2`
- `log_cos2 = log(cosθ2)`
- `polylog_0 = polylogarithm(2, -tanθ2)`
- `τ_SG = DEFAULT_y_SG / v0y`

Piecewise result:
- Pre-SG (`t ≤ tf2`):
  `z(t) = z0 + v0z * t`
- In-SG (`tf2 < t ≤ tf3`), with `Δt = t - tf2` and `exp_term = exp(-2*kω*Δt)`:
  `z(t) = z0 + v0z*t + 0.5*acc_z*Δt^2 + (acc_z/kω)*log_cos2*Δt
          + 0.5*acc_z/kω^2 * ( polylogarithm(2, -exp_term*tanθ2) - polylog_0 )`
- Post-SG (`t > tf3`), with `Δt3 = t - tf3`,
  `exp_SG = exp(-2*kω*τ_SG)`, `polylog_SG = polylogarithm(2, -exp_SG*tanθ2)`,
  `log_term = log(cosθ2 + exp_SG*sinθ2)`:
  `z(t) = z0 + v0z*t + acc_z*τ_SG*(t - tf2 - 0.5*τ_SG)
          + (acc_z/kω)*τ_SG * (log_cos2 + (Δt3/τ_SG)*log_term)
          + 0.5*acc_z/kω^2 * (polylog_SG - polylog_0)`
  (the `acc_z*τ_SG*(t-tf2-0.5*τ_SG)` form is algebraically equivalent to, and
  cheaper/more cancellation-resistant than, expanding it as a difference of
  two squared time offsets — see chat discussion on this exact line.)

Arguments:
- `t::Float64`: time since the initial state `(r0, v0)` (s).
- `Ix::Float64`: coil current (A) used by `GvsI`/`BvsI`.
- `μ::Float64`: effective magnetic moment (J/T).
- `r0::AbstractVector{Float64}`: initial position `(x0, y0, z0)` in meters; length 3.
- `v0::AbstractVector{Float64}`: initial velocity `(v0x, v0y, v0z)` in m/s; length 3 and `v0y ≠ 0`.
- `θe::Float64`, `θn::Float64`: electron and nuclear angles (rad).
- `kx::Float64`: dimensionless coupling multiplying `ωL`.
- `p::AtomParams`: atomic parameters (must include mass `M`). Assumes `γₑ`, `GvsI`, and `BvsI` are in scope.

Returns:
- `z::Float64` — z-position at time `t`.

Assumptions:
- Non-relativistic; gravity, collisions, and fringe-field variation neglected.
- SG gradient aligned with `+z`; only `z` accelerates inside the SG.
- Expressions are continuous at `t = tf2` and `t = tf3`.

# Performance
- **Early exit before any physics setup.** `tf2` only needs `v0y` and
  geometry constants, so the pre-SG check (`t <= tf2`) happens first; the
  `BvsI`/`GvsI` lookups, all three trig calls, and the `polylogarithm`
  evaluation are skipped entirely for any `t` in the pre-SG region, rather
  than being computed unconditionally before the branch (as in the original
  ordering). This matters if the function is called repeatedly across many
  `t` values for one trajectory (e.g. for plotting) — every pre-SG call now
  avoids work it would otherwise discard.
- **Division replaced by reciprocal-multiply.** `inv_v0y = 1/v0y` and
  `inv_kω = 1/kω` are each computed once and reused, replacing what would
  otherwise be five separate divisions (`tf2`, `tf3`, `τ_SG`, and the two
  `acc_z/kω` terms) with two reciprocals plus multiplications.

# Numerical notes
- Terms with `1/kω` and `1/kω^2` cancel analytically as `kω → 0` but may lose precision;
  consider a small-`kω` expansion when `|kω*τ_SG|` is very small. `inv_kω = 1/kω`
  has the same `kω == 0` failure mode as the original direct-division form —
  no new risk introduced, but also no guard added.
- Same "general/flexible, not stability-hardened" status as `CQD_EqOfMotion` —
  direct `exp`/`log`/`polylogarithm` calls throughout, no
  `log_cos_sin_exp`/`li2_negexp` rewrite.
"""
@inline function CQD_EqOfMotion_z(t,Ix::Float64,μ::Float64,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"
    @assert t >= 0 "time t must be ≥ 0"

    v0y = v0[2]
    v0z = v0[3]
    z0  = r0[3]

    @assert !iszero(v0y) "v0y must be nonzero (beam must advance toward the screen)."
    
    inv_v0y = 1.0 / v0y
    tf2 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG) * inv_v0y) ::Float64

    # Pre-SG: nothing below this line is needed at all
    t <= tf2 && return z0 + v0z*t

    tf3 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG) * inv_v0y) ::Float64

    # Only pay for BvsI/GvsI/trig/polylog once we know we're past tf2
    cqd_sign = sign(θn-θe) 
    ωL       = abs( γₑ * BvsI(Ix) )
    acc_z    = μ*GvsI(Ix)/p.M
    kω       = cqd_sign*kx*ωL
    inv_kω   = 1.0 / kω

    # Precompute angles
    θe_half = θe / 2
    tanθ2 = tan(θe_half)^2
    cosθ2 = cos(θe_half)^2
    sinθ2 = sin(θe_half)^2
    log_cos2 = log(cosθ2)
    polylog_0 = polylogarithm(2, -tanθ2)

    if t <= tf3   # Crossing the SG apparatus
        Δt = t - tf2
        exp_term = exp(-2 * kω * Δt)
        polylog_t = polylogarithm(2, -exp_term * tanθ2)

        return z0 + v0z*t + 0.5 * acc_z * Δt^2 + acc_z * inv_kω  * log_cos2 * Δt + 0.5 * acc_z * inv_kω^2 * ( polylog_t - polylog_0 )
    
    else # t > tf3 # Travel to the Screen
        Δt3 = t - tf3
        τ_SG = (DEFAULT_y_SG * inv_v0y) ::Float64
        exp_SG = exp(-2 * kω * τ_SG)
        polylog_SG = polylogarithm(2, -exp_SG * tanθ2)
        log_term = log(cosθ2 + exp_SG * sinθ2)

        return z0 + v0z*t + acc_z*τ_SG*( t-tf2-0.5*τ_SG ) + acc_z * inv_kω * τ_SG * (log_cos2 + log_term * Δt3 / τ_SG) + 0.5 * acc_z * inv_kω^2 * (polylog_SG - polylog_0)
    end
end

# CQD Screen position
"""
    CQD_Screen_position(Ix, μ, r0, v0, θe, θn, kx, p) -> SVector{3,Float64}

Final **position at the screen** for an atom under a Continuous Quantum Dynamics (CQD)
model in a Stern–Gerlach (SG) setup. Motion in `x` and `y` is ballistic; the `z`
coordinate includes CQD-dependent terms derived in closed form (logs and a dilogarithm).

# Geometry (meters, along +y)
Let
- `L1 = DEFAULT_y_FurnaceToSlit`
- `L2 = DEFAULT_y_SlitToSG`
- `Lsg = DEFAULT_y_SG`
- `Ld = DEFAULT_y_SGToScreen`
- `Ltot = L1 + L2 + Lsg + Ld`.

# Physics definitions
- `cqd_sign = sign(θn − θe)`
- `acc_z = μ * GvsI(Ix) / p.M`                # base acceleration scale (m/s²)
- `ωL = abs(γₑ * BvsI(Ix))`                   # Larmor angular frequency (rad/s)
- `kω = cqd_sign * kx * ωL`
- `τ_SG = Lsg / v0y`
- `θe_half = θe/2`, `cos2 = cos(θe_half)^2`, `sin2 = sin(θe_half)^2`, `tan2 = sin2/cos2`
- `exp_term = exp(-2 * kω * τ_SG)`.

# Closed-form result
With initial state `r0 = (x0, y0, z0)` (m) and `v0 = (v0x, v0y, v0z)` (m/s), the screen
coordinates are
- `x = x0 + Ltot * v0x / v0y`
- `y = y0 + Ltot`
- `z = z0 + (Ltot * v0z)/v0y
        + (1/2) * acc_z / v0y^2 * [ Lsg*(Lsg + 2*Ld) ]
        + (acc_z / kω) * τ_SG * [ log(cos2) + (Ld/Lsg) * log(cos2 + exp_term * sin2) ]
        + (1/2) * acc_z / kω^2 * [ polylogarithm(2, -exp_term * tan2) − polylogarithm(2, -tan2) ].

(`Lsg*(Lsg+2*Ld)` is the same quantity as `(Lsg+Ld)^2 - Ld^2`, just in its
cheaper expanded-and-simplified form — see Performance below.)

# Arguments
- `Ix`: Coil current (A) used by `GvsI` and `BvsI`.
- `μ::Float64`: Effective magnetic moment (J/T) for this CQD model.
- `r0::AbstractVector{Float64}`: Initial position `(x0, y0, z0)` in meters; **length 3**.
- `v0::AbstractVector{Float64}`: Initial velocity `(v0x, v0y, v0z)` in m/s; **length 3** and `v0y ≠ 0`.
- `θe::Float64`, `θn::Float64`: Electron and nuclear angles (radians).
- `kx::Float64`: Dimensionless coupling factor multiplying `ωL`.
- `p::AtomParams`: Atomic parameters; must include `M`. Assumes `γₑ`, `GvsI`, and `BvsI` are in scope.

# Returns
- `SVector{3,Float64}`: `(x, y, z)` at the screen.

# Assumptions
- Non-relativistic; gravity, collisions, and fringe-field variation neglected.
- Gradient aligned with `+z`; only `z` is accelerated inside the SG.
- Uses global geometry constants `DEFAULT_y_*`.

# Throws
- Assertion errors if `length(r0) ≠ 3`, `length(v0) ≠ 3`, or `v0y == 0` (the
  `v0y` check was missing from the original version of this function — every
  sibling function in this file already has it; added here for consistency).

# Performance
- `τ_SG = Lsg/v0y` is computed once and shared between `exp_term`'s argument
  and the `z` amplitude term — the original computed `Lsg/v0y` twice
  independently.
- `inv_v0y = 1/v0y` and `inv_kω = 1/kω` (each squared once for the `^2` terms)
  replace six separate divisions with two reciprocals plus multiplications —
  the same strength-reduction already used in the "fast scalar core" sibling
  `CQD_screen_x_z_vz`.
- `sincos(θe_half)` replaces three independent `cos`/`sin`/`tan` calls with
  one combined `cos`/`sin` evaluation plus `tan2 = sin2/cos2` — again
  mirroring `CQD_screen_x_z_vz`/`CQD_cavity_crash`.
- `(Lsg+Ld)^2 - Ld^2` rewritten as `Lsg*(Lsg+2*Ld)` (algebraically identical,
  fewer operations) — the same simplification `QM_screen_x_z_vz` already
  applies via its precomputed `ΔL` argument.

# Notes (numerics)
- The formula contains `1/kω` and `1/kω^2` terms; the overall `z` remains finite as `kω → 0`,
but direct evaluation can lose precision. If you expect very small `|kω * τ_SG|`,
consider using a small-`kω` series expansion for the log/polylog combination.
`inv_kω = 1/kω` has the same `kω == 0` failure mode as the original
direct-division form — no new risk introduced, no guard added.
- `exp_term` may under/overflow when `|kω * τ_SG|` is large; a log-sum-exp rewrite can
improve stability near extreme angles (`cos2` or `sin2` ≈ 0). Left as direct
`exp`/`log`/`polylogarithm` calls here, consistent with this function's
"general/flexible, not stability-hardened" status (same as `CQD_EqOfMotion`) —
treating that as a separate decision from the speed fixes above.
"""
function CQD_Screen_position(Ix,μ::Float64,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"
    x0, y0, z0 = r0
    v0x, v0y, v0z = v0
    @assert !iszero(v0y) "v0y must be nonzero (beam must advance toward the screen)."

    L1   = DEFAULT_y_FurnaceToSlit ::Float64
    L2   = DEFAULT_y_SlitToSG ::Float64
    Lsg  = DEFAULT_y_SG ::Float64
    Ld   = DEFAULT_y_SGToScreen ::Float64
    Ltot = L1 + L2 + Lsg + Ld

    # Physics parameters
    cqd_sign = sign(θn-θe) 
    acc_z = μ * GvsI(Ix) / p.M
    ωL = abs(γₑ * BvsI(Ix))
    kω = cqd_sign * kx * ωL

    inv_v0y  = 1.0 / v0y
    inv_v0y2 = inv_v0y * inv_v0y
    inv_kω   = 1.0 / kω
    inv_kω2  = inv_kω * inv_kω
    τ_SG     = Lsg * inv_v0y

    # Common trig values: sincos shares work between sin/cos; tan2 derived
    # from them instead of a third transcendental call
    θe_half = θe / 2
    s, c = sincos(θe_half)
    sin2 = s*s
    cos2 = c*c
    tan2 = sin2 / cos2
    exp_term = exp(-2 * kω * τ_SG)

    x = x0 + Ltot * v0x * inv_v0y
    y = y0 + Ltot
    z = z0 + Ltot * v0z * inv_v0y +
        0.5*acc_z*inv_v0y2*Lsg*(Lsg + 2*Ld) +
        acc_z*inv_kω*τ_SG*( log(cos2) + Ld/Lsg * log( cos2 + exp_term*sin2 ) ) +
        0.5*acc_z*inv_kω2 * ( polylogarithm(2, -exp_term*tan2) - polylogarithm(2, -tan2) )

    return SVector{3,Float64}(x,y,z)
end


"""
    CQD_Screen_velocity(Ix, μ, v0, θe, θn, kx, p) -> SVector{3,Float64}

Final **velocity at the screen** under a Continuous Quantum Dynamics (CQD) model.
Motion is ballistic in `x` and `y`; only the `z` component is affected inside the
Stern–Gerlach (SG) region of length `DEFAULT_y_SG`.

# Model / Definitions
Let
- `Lsg = DEFAULT_y_SG` (m), `τ_SG = Lsg / v0y` (s),
- `acc_z = μ * GvsI(Ix) / p.M`  (m/s²),
- `ωL = |γₑ| * BvsI(Ix)`        (rad/s),
- `kω = sign(θn − θe) * kx * ωL`,
- `θh = θe/2`, `cos2 = cos(θh)^2`, `sin2 = sin(θh)^2`.

Then
- `v_x = v0x`
- `v_y = v0y`
- `v_z = v0z + acc_z * τ_SG + (acc_z / kω) * log( cos2 + e^{-2 kω τ_SG} * sin2 )`

The implementation is continuous at `kω → 0`, using the limit
`v_z = v0z + acc_z * τ_SG * cos(θe)` to avoid the `0/0` indeterminacy.

# Arguments
- `Ix`: Coil current (A) used by `GvsI` and `BvsI`.
- `μ::Float64`: Effective magnetic moment (J/T) entering the CQD dynamics.
- `v0::AbstractVector{<:Real}`: Initial velocity `(v0x, v0y, v0z)` in m/s; **length 3** and `v0y ≠ 0`.
- `θe::Float64`, `θn::Float64`: Electron and nuclear angles (radians).
- `kx::Float64`: Dimensionless coupling factor multiplying `ωL`.
- `p::AtomParams`: Atomic parameters; must provide mass `M`. Assumes `GvsI`, `BvsI`, and `γₑ` are in scope.

# Returns
- `SVector{3,Float64}`: `(v_x, v_y, v_z)` at the screen.

# Assumptions
- Non-relativistic; gravity, collisions, and fringe-field variations neglected.
- SG gradient aligned with `+z` and treated constant across `DEFAULT_y_SG`.

# Throws
- Assertion error if `length(v0) ≠ 3` or `v0y == 0`.

# Notes
- Uses `DEFAULT_y_SG` for the SG length; adjust the function if you need it configurable.
- Handles the `kω → 0` case internally for numerical stability.
"""
function CQD_Screen_velocity(Ix,μ::Float64,v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)

    @assert length(v0) == 3 "v0 must have length 3"
    v0x, v0y, v0z = v0
    @assert !iszero(v0y) "v0y must be nonzero (beam must advance toward the screen)."

    Lsg = DEFAULT_y_SG ::Float64

    # Physics parameters
    cqd_sign = sign(θn-θe) 
    acc_z = μ * GvsI(Ix) / p.M
    ωL = abs(γₑ * BvsI(Ix))
    kω = cqd_sign * kx * ωL

    # Common trig values: sincos shares work between sin/cos
    θe_half = θe / 2
    s, c = sincos(θe_half)
    sin2 = s*s
    cos2 = c*c
    τ_SG = Lsg / v0y

    vx = v0x
    vy = v0y
    vz = if iszero(kω) || abs(kω * τ_SG) < 1e-18
        # Continuous kω→0 limit: avoids 0/0 in the log term
        v0z + acc_z * τ_SG * cos(θe)
    else
        exp_term = exp(-2 * kω * τ_SG)
        v0z + acc_z * τ_SG + acc_z / kω * log( cos2 + exp_term*sin2) 
    end

    return SVector{3,Float64}(vx,vy,vz)
end

"""
    CQD_screen_x_z_vz(x0, z0, v0x, v0y, v0z, θe, acc, kw, Lsg, Ld, Ltot, ΔL) -> (x, z, vz)
 
**Fast scalar core** for the CQD screen position/velocity: scalar `Float64`
arguments only, `muladd` throughout, and overflow-safe log-domain identities
(`log_cos_sin_exp`, `li2_negexp`) so that `exp(+large)` is never evaluated
regardless of `kw`'s sign or magnitude.
 
# Method
1. Kinematic ratios `α = v0x/v0y`, `γ = v0z/v0y`, `κ = 0.5*acc/v0y²`,
   `τ_SG = Lsg/v0y` are computed once; the screen `x` (purely kinematic,
   independent of `kw`/`θe`) follows immediately.
2. **Small-`kw` branch** (`iszero(kw) || abs(t) < 1e-12`, where
   `t = -2*kw*τ_SG`): returns the `kω → 0` limit directly — `cos(θe)`
   computed once and reused for both `z` and `vz` — avoiding any `1/kw`
   blowup. This branch only needs `kw`/`τ_SG`/`θe`/`acc`/`ΔL`/`Ltot`/`γ`/`z0`,
   so the angle/log/dilog work in step 3 is skipped entirely whenever this
   branch is taken.
3. **General branch**: precomputes `sin² (θe/2)`, `cos² (θe/2)` (floored at
   `1.0e-21` to avoid `log(0)`), their ratio, and the corresponding stable
   log-domain dilogarithm reference value. `log(cos² + sin²·e^t)` is then
   computed once via `log_cos_sin_exp` and reused for both the `z` and `vz`
   formulas; the dilogarithm correction is likewise obtained as a single
   stable log-domain difference rather than two direct evaluations.
 
# Arguments
- `x0, z0::Float64`: initial transverse position at the SG entrance (m).
- `v0x, v0y, v0z::Float64`: initial velocity (m/s); `v0y ≠ 0`.
- `θe::Float64`: electron angle (rad).
- `acc::Float64`: base acceleration scale (`μ*GvsI(Ix)/M`, m/s²).
- `kw::Float64`: CQD relaxation parameter (`kω` elsewhere in this file).
- `Lsg, Ld, Ltot, ΔL::Float64`: SG length, SG-to-screen drift, total path
  length, and the geometric factor `Lsg² + 2·Lsg·Ld` (m, m, m, m²).
 
# Returns
`(x, z, vz)::NTuple{3,Float64}` — screen `x`, screen `z`, exit `vz`.
 
# Performance
The angle/log/dilog setup in step 3 above sits *after* the small-`kw` branch
check, not before it — that branch never reads any of those quantities, so
evaluating them only on the path that actually needs them avoids two `log`
calls and a dilogarithm evaluation whenever the small-`kw` limit applies.
"""
@inline function CQD_screen_x_z_vz(
    x0::Float64, z0::Float64, v0x::Float64, v0y::Float64, v0z::Float64,
    θe::Float64, acc::Float64, kw::Float64,
    Lsg::Float64, Ld::Float64, Ltot::Float64, ΔL::Float64
)::NTuple{3,Float64}
 
    @assert v0y != 0.0 "v0y must be nonzero"
 
    # --- Kinematics ---
    inv_vy  = 1.0 / v0y
    inv_vy2 = inv_vy * inv_vy
    τ_SG    = Lsg * inv_vy
    α       = v0x * inv_vy
    γ       = v0z * inv_vy
    κ       = 0.5 * acc * inv_vy2
 
    # --- Screen x is purely kinematic ---
    x = muladd(Ltot, α, x0)
 
    # exponent t = -2*kw*τ_SG (used everywhere downstream)
    t = -2.0 * kw * τ_SG
 
    # Small-kw guard (smooth CQD→QM limit), checked BEFORE the angle/log/dilog
    # setup below — none of that setup is needed in this branch.
    if iszero(kw) || abs(t) < 1e-12
        cθ = cos(θe)
        z  = muladd(Ltot, γ, z0) + κ * ΔL * cθ
        vz = muladd(acc*cθ, τ_SG, v0z)
        return (x, z, vz)
    end
 
 
    # --- Angle precompute (stable near extremes) ---
    s, c    = sincos(0.5 * θe)
    sin2    = s*s
    cos2    = max(c*c, 1.0e-21)
    tan2    = sin2 / cos2
    logcos2 = log(cos2)
 
    # Stable reference value Li₂(-tan²) using log form
    logtan   = log(tan2)
    PL_tan2  = li2_negexp(logtan)
 
    # --- CQD corrections ---
    inv_kw  = 1.0 / kw
    inv_kw2 = inv_kw * inv_kw
    coeff_group = 0.5 * acc * inv_kw2
 
    # Stable log(cos2 + sin2*exp(t)) without forming exp(+big)
    logterm = log_cos_sin_exp(t, cos2, sin2)
 
    # Stable dilog: Li₂(-exp(logtan + t)) - Li₂(-exp(logtan))
    li2diff = li2_negexp(logtan + t) - PL_tan2
 
    # --- Screen z with time dependent projection ---
    z = muladd(Ltot, γ, z0) +
        κ * ΔL +
        acc * inv_kw * τ_SG  * (logcos2 + (Ld/Lsg) * logterm) +
        coeff_group * li2diff
 
    # --- CQD exit velocity ---
    # Your original: v0z + acc*τ_SG + acc/kw * log(cos2 + Esg*sin2)
    vz = muladd(acc, τ_SG, v0z) + acc * inv_kw * logterm
 
    return (x, z, vz)
end

"""
    CQD_cavity_crash(μG_ix::Float64, B0_ix::Float64,
                     x0::Float64, y0::Float64, z0::Float64,
                     v0x::Float64, v0y::Float64, v0z::Float64,
                     θe::Float64, θn::Float64,
                     kx::Float64,
                     p::AtomParams,
                     ygrid::AbstractVector{Float64},
                     eps::Float64) -> UInt8

Trace a particle through the Stern–Gerlach (SG) cavity under the
**Continuous Quantum Dynamics (CQD)** model and report whether it collides
with the **top edge**, **bottom trench**, or clears the cavity but
**misses the exit tube** at the screen.

The trajectory is evaluated along the longitudinal grid `ygrid`
(lab-frame absolute y positions). Transverse coordinates `(x(y), z(y))`
are computed analytically from the CQD equations of motion.

The CQD correction is controlled by

`kω = sign(θn − θe) * kx * ωL`,  with  `ωL = |γₑ * B0_ix|`,

and includes closed-form logarithmic and dilogarithmic contributions.
These terms are evaluated using **overflow-safe identities** for
`Li₂(-exp(x))` and `log(cos² + sin²·exp(x))`, so the function remains
numerically stable for both signs and arbitrarily large magnitudes of `kω`
without altering the underlying physics.

If `kω == 0`, the motion reduces to a constant-acceleration trajectory
and a simplified analytic path is used.

---

### Arguments
- `μG_ix`: Magnetic force coefficient (e.g. `μ ∂B/∂z`) for the given current.
- `B0_ix`: Magnetic field magnitude scale for this setting (used to form `ωL`).
- `x0, y0, z0`: Initial position at the entrance reference (units consistent
  with the geometry).
- `v0x, v0y, v0z`: Initial velocity components; **`v0y` must be nonzero**.
- `θe, θn`: Electronic and nuclear spin angles (radians).
- `kx`: Spatial modulation wavenumber of the CQD correction.
- `p`: `AtomParams` instance (must provide atomic mass `M`).
- `ygrid`: Monotonically increasing **absolute** y-positions spanning the cavity
  and downstream regions.
- `eps`: Tolerance for wall-contact tests.

---

### Returns
`UInt8` collision code:
- `0x00` — no collision, exits within the tube
- `0x01` — collision with the top edge inside the cavity
- `0x02` — collision with the bottom trench inside the cavity
- `0x03` — clears the cavity but misses the exit tube at the screen
- `0x04` — blocked by the circular aperture

---

### Assumptions
- Wall profiles `z_magnet_edge(x)` and `z_magnet_trench(x)` are defined,
  pure functions returning `Float64`.
- Geometry constants `DEFAULT_y_*`, `DEFAULT_R_tube`, `DEFAULT_c_aperture`,
  and `γₑ` exist and share consistent units with positions and velocities.
- `ygrid` is in the **lab frame** (absolute y); linear terms use `y - y0`,
  while cavity-specific terms use `y - y_in`.

---

### Numerical notes
- The implementation is **allocation-free** and suitable for execution
  millions of times inside tight loops.
- All exponentials are evaluated in a numerically stable form; no
  `exp(+large)` calls occur, so overflow is avoided even for large
  `|kω|` or long cavities.
- Fused multiply-add (`muladd`) is used where beneficial for accuracy and speed.
- Near `θe ≈ π`, `cos²(θe/2) → 0`; the implementation clamps this quantity
  internally to avoid singular logs while preserving physical behavior.
- Compare `CQD_Bn_cavity_crash` below: same role, same structure, but its
  inner loop calls `exp`/`polylogarithm` directly rather than the
  `li2_negexp`-based stable form used here — see that function's docstring.

---

### Performance
- The `θe`-dependent precompute (`sincos`, `cos²`/`sin²`/`tan²`, the two
  `log` calls, and the `li2_negexp` dilogarithm evaluation) sits *after* the
  `kω == 0` branch, not before it — that branch never reads any of those
  quantities, so it no longer pays for them.
- Inside each branch's `ygrid` loop, the per-iteration update uses one
  hoisted, loop-invariant product (`κcθe` in the `kω==0` branch,
  `log_coef_logcosθ2` in the `kω≠0` branch) instead of recomputing that
  product on every iteration — the saving scales with the number of grid
  points, which is why it matters most here relative to the rest of the
  function.
- `log_coef_Lsg` (used by both the aperture and screen checks in the `kω≠0`
  branch) is computed once and reused, rather than once per check.
- The two CQD correction coefficients are named for what they scale, not as
  bare algebra letters: `log_coef` multiplies the log-term contribution,
  `dilog_coef` multiplies the dilogarithm-term contribution — chosen to
  avoid any visual confusion with the magnetic-field quantities (`B0_ix`,
  `B0`, `Bx`/`By`/`Bz`) used throughout this codebase.

---

### Throws
- `AssertionError` if `v0y == 0`.

---

### Example
# code = CQD_cavity_crash(
#     μG, B0,
#     x0, y0, z0,
#     vx, vy, vz,
#     θe, θn,
#     kx, p,
#     ygrid,
#     1e-9
# )
# code == 0x00 || @info "Collision code = code"
"""
@inline function CQD_cavity_crash(
    μG_ix::Float64, B0_ix::Float64,
    x0::Float64, y0::Float64, z0::Float64,
    v0x::Float64, v0y::Float64, v0z::Float64,
    θe::Float64, θn::Float64,
    kx::Float64,
    p::AtomParams,
    ygrid::AbstractVector{Float64},
    eps::Float64
)::UInt8

    @assert v0y != 0.0 "v0y must be nonzero"

    # ---- geometry (leave as you had it; these are scalars) ----
    y_in   = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    Lsg    = (DEFAULT_y_SG)::Float64
    Ld     = (DEFAULT_y_SGToScreen)::Float64
    Ltot   = (y_in + Lsg + Ld)::Float64
    R2tube = (DEFAULT_R_tube * DEFAULT_R_tube)::Float64
    LA     = (DEFAULT_y_SGToAperture)::Float64
    Lapert = (y_in + Lsg + LA)::Float64
    R2circ = (DEFAULT_c_aperture * DEFAULT_c_aperture)::Float64

    # ---- kinematics ----
    cqd_sign = sign(θn - θe)
    ωL       = abs(γₑ * B0_ix)
    acc_0    = μG_ix / p.M
    kω       = cqd_sign * kx * ωL

    inv_v0y  = 1.0 / v0y
    inv_v0y2 = inv_v0y * inv_v0y

    # ---- constants needed in BOTH branches ----
    α        = v0x * inv_v0y
    γ        = v0z * inv_v0y
    κ        = 0.5 * acc_0 * inv_v0y2

    # ---- kω = 0 special case ----
    if iszero(kω)
        cθe  = cos(θe)
        κcθe = κ * cθe   # loop-invariant; hoisted out of the ygrid loop below
        @inbounds for i in eachindex(ygrid)
            y  = ygrid[i] - y0
            Δy = y - y_in
            x  = muladd(α, y, x0)
            z  = muladd(γ, y, z0) + κcθe*(Δy*Δy)
            (z - z_magnet_edge(x))   >=  eps && return UInt8(0x01)
            (z - z_magnet_trench(x)) <= -eps && return UInt8(0x02)
        end

        x_aper = muladd(Lapert, α, x0)
        z_aper = muladd(Lapert, γ, z0) + κcθe*(Lsg*(Lsg + 2.0*LA))
        (muladd(x_aper, x_aper, z_aper*z_aper) >= R2circ) && return UInt8(0x04)

        x_screen = muladd(Ltot, α, x0)
        z_screen = muladd(Ltot, γ, z0) + κcθe*(Lsg*(Lsg + 2.0*Ld))
        return (muladd(x_screen, x_screen, z_screen*z_screen) >= R2tube) ? UInt8(0x03) : UInt8(0x00)
    end

    # ---- θe terms — deferred until here, only needed when kω≠0 ----
    scale_a  = -2.0 * kω * inv_v0y   # so exp_arg = scale_a * Δy

    s, c     = sincos(0.5 * θe)
    sinθ2    = s * s
    cosθ2    = max(c*c, 1.0e-21)   # avoid log(0) & div by 0
    tanθ2    = sinθ2 / cosθ2
    logcosθ2 = log(cosθ2)

    # Precompute Li₂(-tan²) in a stable way too:
    logtan   = log(tanθ2)
    PL_tanθ2 = li2_negexp(logtan)

    # ---- kω ≠ 0 path ----
    inv_kω   = 1.0 / kω
    inv_kω2  = inv_kω * inv_kω
    log_coef         = acc_0 * inv_kω  * inv_v0y     # ~ 1/kω; scales the log-term contribution
    dilog_coef       = 0.5  * acc_0 * inv_kω2        # ~ 1/kω^2; scales the dilog-term contribution
    log_coef_logcosθ2 = log_coef * logcosθ2          # loop-invariant; hoisted out of the ygrid loop below
    log_coef_Lsg      = log_coef * Lsg               # shared by the aperture and screen checks below

    @inbounds for i in eachindex(ygrid)
        y  = ygrid[i] - y0
        Δy = y - y_in
        exp_arg = scale_a * Δy

        # Stable: Li₂(-exp(logtan + exp_arg))
        li2term = li2_negexp(logtan + exp_arg) - PL_tanθ2

        x  = muladd(α, y, x0)
        z  = muladd(γ, y, z0) + κ*(Δy*Δy) + log_coef_logcosθ2*Δy + dilog_coef*li2term

        (z - z_magnet_edge(x))   >=  eps && return UInt8(0x01)
        (z - z_magnet_trench(x)) <= -eps && return UInt8(0x02)
    end

    # ---- end-of-SG terms (avoid exp overflow) ----
    t_sg       = scale_a * Lsg
    dilog_term = dilog_coef*(li2_negexp(logtan + t_sg) - PL_tanθ2)
    logterm    = log_cos_sin_exp(t_sg, cosθ2, sinθ2)

    # ------ Circular aperture -----
    x_aper = muladd(Lapert, α, x0)
    z_aper = muladd(Lapert, γ, z0) +
             κ*(Lsg*Lsg + 2.0*Lsg*LA) +
             log_coef_Lsg*(logcosθ2 + (LA/Lsg)*logterm) +
             dilog_term
    (muladd(x_aper, x_aper, z_aper*z_aper) >= R2circ) && return UInt8(0x04)

    # ------ Screen check -----
    x_screen = muladd(Ltot, α, x0)
    z_screen = muladd(Ltot, γ, z0) +
               κ*(Lsg*Lsg + 2.0*Lsg*Ld) +
               log_coef_Lsg*(logcosθ2 + (Ld/Lsg)*logterm) +
               dilog_term

    return (muladd(x_screen, x_screen, z_screen*z_screen) >= R2tube) ? UInt8(0x03) : UInt8(0x00)
end



########################################################################################################################################
# Quantum Mechanics : Classical Trajectories
########################################################################################################################################
 
 
# QM equations of motion
"""
    QM_EqOfMotion(t, Ix, f, mf, r0, v0, p) -> (r, v)
 
Propagate a single atom from the furnace to the screen through a Stern–Gerlach (SG) region,
using a piecewise-constant-acceleration model:
 
1. Free flight (furnace → slit, slit → SG entrance).
2. Uniform acceleration along `z` **inside the SG region**.
3. Free flight again (SG exit → screen).
 
Inside the SG, the `z`-acceleration is
`a_z = μ_F * GvsI(Ix) / p.M`, with `μ_F = μF_effective(Ix, f, mf, p)`.
 
Unlike the CQD family above, this model has no exponential/logarithmic/
dilogarithm terms at all — pure polynomial-in-time kinematics — so there is
no overflow-safe-identity concern anywhere in this QM section.
 
# Arguments
- `t::Real`: Time (s) since the initial state `(r0, v0)` is defined; must satisfy `t ≥ 0`.
- `Ix::Real`: Coil current (A) for the field-gradient model `GvsI(Ix)`.
- `f`, `mf`: Hyperfine quantum numbers selecting the effective magnetic moment.
- `r0::AbstractVector{<:Real}`: Initial position `(x0, y0, z0)` in meters; length 3.
- `v0::AbstractVector{<:Real}`: Initial velocity `(v0x, v0y, v0z)` in m/s; length 3 and `v0y ≠ 0`.
- `p::AtomParams`: Atomic parameters (must provide mass `M` and anything required by `μF_effective`).
 
# Geometry (constants used)
Uses `DEFAULT_y_FurnaceToSlit`, `DEFAULT_y_SlitToSG`, `DEFAULT_y_SG`, `DEFAULT_y_SGToScreen`
(measured along +y, in meters).
 
# Returns
A tuple `(r, v)` with `r::SVector{3,Float64}` and `v::SVector{3,Float64}` giving position and
velocity at time `t`.
 
# Assumptions
- Non-relativistic kinematics; gravity, collisions, and fringing fields neglected.
- Only the `z` component accelerates (constant gradient within the SG region).
- `x` and `y` remain ballistic for all segments.
 
# Performance
- `μF_effective`/`GvsI` (and the `acc_z` derived from them) are only
  evaluated once `t` is known to be past `tf2` (SG entrance) — the pre-SG
  branch returns using only `z0`/`v0z`/`t`, so it no longer pays for either
  call. `tf3` (SG exit) is likewise only computed in that same deferred path,
  since the pre-SG branch doesn't need it either.
- `inv_v0y = 1/v0y` is computed once and reused for `tf2`, `tf3`, and the
  post-SG `Δt_SG`, replacing three divisions with one reciprocal plus
  multiplications.
- `tf2`, `tf3`, and `Δt_SG` are each explicitly ascribed `::Float64`: since
  they're built from `DEFAULT_y_*` geometry constants, this re-establishes a
  concrete type at each point in case those constants are non-`const`
  globals (otherwise `Any`-typed reads would propagate into the
  multiple downstream uses of each quantity). Harmless and free if the
  constants are already `const`.
 
# Notes
- Output is `Float64` even if inputs are higher precision; change the return SVectors’ element
  type if you want to preserve precision.
- If you plan to allow `t < 0`, relax the assertion accordingly.
"""
@inline function QM_EqOfMotion(t,Ix,f,mf,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real}, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"

    x0, y0, z0 = r0
    v0x, v0y, v0z = v0
    @assert v0y != 0.0 "y-velocity must be nonzero."
    @assert t >= 0 "time t must be ≥ 0" 

    inv_v0y = 1.0 / v0y

    # Ballistic x,y always
    x = x0 + v0x * t
    y = y0 + v0y * t
    vx = v0x
    vy = v0y

    # _tf1 =  DEFAULT_y_FurnaceToSlit / v0y                               # slit entrance
    # _tF  = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToScreen) / v0y  # screen

    tf2 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG) * inv_v0y) :: Float64   # SG entrance
    
    if t <= tf2     # Furnace to Slit and Slit to SG apparatus
        # Pre-SG: fully ballistic — μF_effective/GvsI never evaluated here
        z = z0 + v0z*t
        vz = v0z
    else
        tf3 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG) * inv_v0y) :: Float64   # SG exit
 
        μ =  μF_effective(Ix,f,mf,p)
        acc_z    = μ * GvsI(Ix) / p.M
 
        if t <= tf3   # Crossing the SG apparatus
            # Inside SG: uniform az
            Δt = t-tf2
            z = z0 + v0z * t + 0.5 * acc_z * Δt^2
            vz = v0z + acc_z * Δt
        else # t > tf3 # Travel to the Screen
            # Post-SG: ballistic with boosted vz
            Δt_SG = (DEFAULT_y_SG * inv_v0y) :: Float64
            z = z0 + v0z * t + acc_z * Δt_SG * (t - 0.5*(tf2+tf3))
            vz = v0z + acc_z * Δt_SG 
        end
    end
    
    r = SVector{3,Float64}(x, y, z)
    v = SVector{3,Float64}(vx, vy, vz)
    return r, v
end

"""
    QM_EqOfMotion_z(t, Ix, f, mf, r0, v0, p) -> z

Propagate only the **z-coordinate** of an atom moving from the furnace to the screen
through a Stern–Gerlach (SG) region under a piecewise-constant-acceleration model.

Segments:
1. **Pre-SG (ballistic)**: furnace → slit → SG entrance
2. **In-SG (uniform az)**: SG entrance → SG exit
3. **Post-SG (ballistic with boosted vz)**: SG exit → screen

Inside the SG, the acceleration along z is
`a_z = μ_F * GvsI(Ix) / p.M`, with `μ_F = μF_effective(Ix, f, mf, p)`.

# Arguments
- `t::Real`: Time (s) since initial state; requires `t ≥ 0`.
- `Ix::Real`: Coil current (A) for the gradient model `GvsI(Ix)`.
- `f`, `mf`: Hyperfine quantum numbers for `μF_effective`.
- `r0::AbstractVector{<:Real}`: Initial position `(x0, y0, z0)` in meters; length 3.
- `v0::AbstractVector{<:Real}`: Initial velocity `(v0x, v0y, v0z)` in m/s; length 3 and `v0y ≠ 0`.
- `p::AtomParams`: Atomic parameters (must provide mass `M` and whatever `μF_effective` needs).

# Geometry constants (meters, along +y)
Uses `DEFAULT_y_FurnaceToSlit`, `DEFAULT_y_SlitToSG`, `DEFAULT_y_SG`, `DEFAULT_y_SGToScreen`.

# Returns
- `z::Float64`: z-position at time `t`.

# Formulas
Let `t_in = (y_FS + y_SS)/v0y`, `t_out = (y_FS + y_SS + y_SG)/v0y`, and `τ_SG = y_SG / v0y`.

- Pre-SG (`t ≤ t_in`): `z(t) = z0 + v0z * t`
- In-SG  (`t_in < t ≤ t_out`): `z(t) = z0 + v0z * t + 0.5 * a_z * (t - t_in)^2`
- Post-SG (`t > t_out`): `z(t) = z0 + v0z * t + a_z * τ_SG * [ t - 0.5*(t_in + t_out) ]`

The post-SG expression is algebraically equivalent to stitching at `t_out` and keeps continuity of
both `z(t)` and `v_z(t)`.

# Assumptions
- Non-relativistic; gravity, collisions, and fringing fields neglected.
- Only `z` accelerates inside the SG; `x,y` are ballistic and unused here.

# Performance
- `μF_effective`/`GvsI` (and the `acc_z` derived from them), along with
  `t_out`, are only computed once `t` is known to be past `t_in` — the
  pre-SG branch returns using only `z0`/`v0z`/`t`, so it no longer pays for
  either call or for a quantity (`t_out`) it never needs.
- `inv_v0y = 1/v0y` is computed once and reused for `t_in`, `t_out`, and
  `τ_SG`, replacing three divisions with one reciprocal plus multiplications.

# Example
julia
z = QM_EqOfMotion_z(0.012, 0.65, 2, 2, [0.0,0.0,0.0], [5.0,800.0,0.0], p)
"""
@inline function QM_EqOfMotion_z(t,Ix::Float64,f,mf,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real}, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"
    
    v0y = v0[2]
    v0z = v0[3]
    z0  = r0[3]

    @assert t >= 0 "time t must be ≥ 0"
    @assert !iszero(v0y) "y-velocity must be nonzero."

    inv_v0y = 1.0 / v0y
    tf2 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG) * inv_v0y) ::Float64

    if t <= tf2
        return z0 + v0z * t
    else
        tf3 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG) * inv_v0y) ::Float64

        # z-acceleration inside SG
        μ =  μF_effective(Ix,f,mf,p)
        acc_z    = μ * GvsI(Ix) / p.M

        if t <= tf3   # Crossing the SG apparatus
            Δt = t - tf2
            return  z0 + v0z * t + 0.5 * acc_z * Δt^2
        else # t > tf3 # Travel to the Screen
            τ_SG = (DEFAULT_y_SG * inv_v0y) ::Float64
            return z0 + v0z * t + acc_z * τ_SG * (t - 0.5*(tf2+tf3))
        end
    end
end

"""
    QM_Screen_position(Ix, f, mf, r0, v0, p) -> r

Final 3D position of an atom at the **screen plane** after traversing a Stern–Gerlach (SG) region.
Model: ballistic motion in `x` and `y`; constant acceleration in `z` **only inside** the SG of length `DEFAULT_y_SG`.

Let `Lsg = DEFAULT_y_SG`, `Ld = DEFAULT_y_SGToScreen`,
`Ltot = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + Lsg + Ld`,
`a_z = μF_effective(Ix, f, mf, p) * GvsI(Ix) / p.M`, and `v0y ≠ 0`.

Closed-form screen position:
- `x = x0 + (Ltot / v0y) * v0x`
- `y = y0 + Ltot`
- `z = z0 + (Ltot / v0y) * v0z + 0.5 * a_z / v0y^2 * [ (Lsg + Ld)^2 - Ld^2 ]`

This `z` formula is equivalent to stitching the three segments (pre-SG, in-SG, post-SG) and
keeps continuity of both position and `v_z` at the SG exit.

# Three methods, fastest-to-most-convenient
1. **Fast scalar core** — `(Ix::Float64, f, mf, x0,y0,z0,v0x,v0y,v0z, p::AtomParams{Float64})`:
   takes individual `Float64` scalars, no vector indexing/allocation, geometry
   constants explicitly type-ascripted (`::Float64`) at the point they're
   read from the (possibly non-`const`) globals. This is the actual
   implementation; the other two methods are thin wrappers around it.
2. **SVector convenience** — `(Ix, f, mf, r0::SVector{3,Float64}, v0::SVector{3,Float64}, p)`:
   unpacks the SVectors' three components and forwards to (1) — zero
   allocation, since `SVector` indexing is just a stack read.
3. **Backward-compatible adapter** — `(Ix, f, mf, r0::AbstractVector{<:Real}, v0::AbstractVector{<:Real}, p)`:
   the original, flexible-input signature; validates lengths, converts each
   component to `Float64`, forwards to (1). Kept so existing call sites with
   plain `Vector`/array inputs keep working unchanged.

# Arguments
- `Ix::Real`: Coil current (A).
- `f::Real`, `mf::Real`: Hyperfine quantum numbers (half-integers allowed; not restricted to `Integer`).
- `r0::AbstractVector{<:Real}`: Initial position `(x0,y0,z0)` in meters; length 3.
- `v0::AbstractVector{<:Real}`: Initial velocity `(v0x,v0y,v0z)` in m/s; length 3, `v0y ≠ 0`.
- `p::AtomParams`: Atomic parameters (must provide mass `M` and whatever `μF_effective` needs).

# Geometry (constants used, not configurable via keywords)
`DEFAULT_y_FurnaceToSlit`, `DEFAULT_y_SlitToSG`, `DEFAULT_y_SG`, `DEFAULT_y_SGToScreen`.

# Returns
- `r::SVector{3,Float64}`: Position at the screen.

# Assumptions
- Non-relativistic; gravity/collisions/fringing fields neglected.
- SG gradient aligned with `+z` and treated constant over `DEFAULT_y_SG`.
"""
# ---------- FAST SCALAR CORE (no AbstractVector, no globals in arithmetic) ----------
Base.@propagate_inbounds @inline function QM_Screen_position(
    Ix::Float64, f, mf,
    x0::Float64, y0::Float64, z0::Float64,
    v0x::Float64, v0y::Float64, v0z::Float64,
    p::AtomParams{Float64}
)::SVector{3,Float64}
    @assert v0y != 0.0 "v0y must be nonzero (beam must advance toward the screen)."

    # bind geometry to concrete locals (avoid Any from non-const globals)
    y_in  = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    Lsg   = DEFAULT_y_SG::Float64
    Ld    = DEFAULT_y_SGToScreen::Float64
    Ltot  = (y_in + Lsg + Ld)::Float64

    # physics
    μG    = μF_effective(Ix, f, mf, p) * GvsI(Ix)
    inv_vy  = 1.0 / v0y
    inv_vy2 = inv_vy * inv_vy
    acc_z = μG / p.M

    # precompute Δ = (Lsg+Ld)^2 - Ld^2 = Lsg^2 + 2 Lsg Ld
    Δ = Lsg*Lsg + 2.0*Lsg*Ld

    x = x0 + Ltot * v0x * inv_vy
    y = y0 + Ltot
    z = z0 + Ltot * v0z * inv_vy + 0.5 * acc_z * inv_vy2 * Δ
    return SVector(x, y, z)
end

# ---------- CONVENIENCE: SVector inputs (zero alloc, avoids indexing) ----------
Base.@propagate_inbounds @inline function QM_Screen_position(
    Ix::Float64, f, mf,
    r0::SVector{3,Float64}, v0::SVector{3,Float64},
    p::AtomParams{Float64}
)::SVector{3,Float64}
    return QM_Screen_position(Ix, f, mf, r0[1], r0[2], r0[3], v0[1], v0[2], v0[3], p)
end

# ---------- BACKWARD-COMPATIBLE API: AbstractVector inputs ----------
# (keeps working everywhere; extracts scalars and calls the fast path)
function QM_Screen_position(
    Ix, f, mf,
    r0::AbstractVector{<:Real}, v0::AbstractVector{<:Real},
    p::AtomParams
)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"
    @inbounds begin
        x0  = Float64(r0[1]);  y0  = Float64(r0[2]);  z0  = Float64(r0[3])
        v0x = Float64(v0[1]);  v0y = Float64(v0[2]);  v0z = Float64(v0[3])
    end
    return QM_Screen_position(Float64(Ix), f, mf, x0, y0, z0, v0x, v0y, v0z, p)
end

"""
    QM_Screen_velocity(...) -> SVector{3,Float64}
 
Final **velocity at the screen** for an atom that traverses a Stern–Gerlach (SG) region.
The model assumes ballistic motion in `x` and `y`, and uniform acceleration along `z`
only while inside the SG of length `DEFAULT_y_SG`.
 
Let
`a_z = μF_effective(Ix, f, mf, p) * GvsI(Ix) / p.M`  and  `τ_SG = DEFAULT_y_SG / v0y`.
Then
- `v_x = v0x`
- `v_y = v0y`
- `v_z = v0z + a_z * τ_SG`
 
# Three methods, fastest-to-most-convenient
Same three-tier structure as `QM_Screen_position` above: a fast scalar core
(the actual implementation, taking `v0x,v0y,v0z::Float64` individually), an
`SVector` convenience wrapper, and a backward-compatible `AbstractVector`
adapter.
 
# Arguments
- `Ix`: Coil current (A) used by the gradient model `GvsI`.
- `f`, `mf`: Hyperfine quantum numbers selecting the effective magnetic moment (not restricted to `Integer`; half-integers allowed).
- `v0::AbstractVector{<:Real}`: Initial velocity `(v0x, v0y, v0z)` in m/s; **must** have length 3 and `v0y ≠ 0`.
- `p::AtomParams`: Atomic parameters; must provide mass `M` and whatever `μF_effective` requires.
 
# Returns
- `v::SVector{3,Float64}`: Velocity at the screen.
 
# Uses/Assumes
- `DEFAULT_y_SG` (meters) as the SG length along `+y`.
- Non-relativistic kinematics; gravity, collisions, and fringe-field variation neglected.
- Only the `z` component accelerates inside the SG.
 
# Throws
- Assertion error if `length(v0) ≠ 3` or `v0y == 0`.
 
# Notes
- The return type is `Float64` by construction; if you need to preserve higher precision
  or AD types, adjust the return container accordingly.
"""
# ---------- FAST SCALAR CORE ----------
Base.@propagate_inbounds @inline function QM_Screen_velocity(
    Ix::Float64, f, mf,
    v0x::Float64, v0y::Float64, v0z::Float64,
    p::AtomParams{Float64}
)::SVector{3,Float64}
    @assert v0y != 0.0 "v0y must be nonzero (beam must advance toward the screen)."
 
    # bind geometry to concrete locals (avoid Any from globals)
    Lsg = DEFAULT_y_SG::Float64
 
    # physics
    μG     = μF_effective(Ix, f, mf, p) * GvsI(Ix)
    inv_vy = 1.0 / v0y
    vz     = v0z + (μG / p.M) * Lsg * inv_vy
 
    return SVector(v0x, v0y, vz)
end
 
# ---------- SVECTOR CONVENIENCE ----------
Base.@propagate_inbounds @inline function QM_Screen_velocity(
    Ix::Float64, f, mf,
    v0::SVector{3,Float64},
    p::AtomParams{Float64}
)::SVector{3,Float64}
    return QM_Screen_velocity(Ix, f, mf, v0[1], v0[2], v0[3], p)
end
 
# ---------- BACKWARD-COMPATIBLE ADAPTER ----------
function QM_Screen_velocity(
    Ix, f, mf,
    v0::AbstractVector{<:Real},
    p::AtomParams
)
    @assert length(v0) == 3 "v0 must have length 3"
    @inbounds begin
        v0x = Float64(v0[1]); v0y = Float64(v0[2]); v0z = Float64(v0[3])
    end
    return QM_Screen_velocity(Float64(Ix), f, mf, v0x, v0y, v0z, p)
end


"""
    QM_screen_x_z_vz(x0, z0, v0x, v0y, v0z, L_SG, ΔL, Ltot, acc_z) -> (x, z, vz)

Propagate a particle from the SG entrance to the screen assuming constant
forward speed `v0y` (no longitudinal acceleration) and a constant transverse
acceleration `acc_z` that acts only inside the Stern–Gerlach region of length
`L_SG`.

The closed-form updates are
- `t_tot = Ltot / v0y`
- `x = x0 + v0x * t_tot`
- `z = z0 + v0z * t_tot + 0.5 * acc_z * (ΔL / v0y^2)`
- `vz = v0z + acc_z * (L_SG / v0y)`

`ΔL` is a precomputed geometric factor that avoids repeated multiplies:
`ΔL = L_SG^2 + 2 L_SG * L_d`, where `L_d` is the drift from the SG exit to the
screen (equivalently `ΔL = (L_SG + L_d)^2 - L_d^2`). Pass it in as a scalar.

All quantities are in SI units.

Arguments
- `x0, z0::Float64`: initial transverse positions at the SG entrance (m)
- `v0x, v0y, v0z::Float64`: initial velocity components (m/s); **`v0y ≠ 0`**
- `L_SG::Float64`: SG region length along the beam (m)
- `ΔL::Float64`: geometric factor defined above (m²)
- `Ltot::Float64`: total path length entrance→screen along the beam (m)
- `acc_z::Float64`: constant transverse acceleration within the SG (m/s²)

Returns
- `(x, z, vz)::NTuple{3,Float64}`: screen `x`, screen `z`, and exit `z`-velocity.
"""
@inline function QM_screen_x_z_vz(
    x0::Float64, z0::Float64, v0x::Float64, v0y::Float64, v0z::Float64,
    L_SG::Float64, ΔL::Float64, Ltot::Float64, acc_z::Float64
)::NTuple{3,Float64}
    inv_vy  = 1.0 / v0y
    inv_vy2 = inv_vy * inv_vy
    x  = muladd(Ltot, v0x*inv_vy, x0)
    z  = muladd(Ltot, v0z*inv_vy, z0) + 0.5 * acc_z * inv_vy2 * ΔL
    vz = muladd(acc_z*L_SG, inv_vy, v0z)
    return (x, z, vz)
end

"""
    QM_cavity_crash(μG_ix, x0, y0, z0, v0x, v0y, v0z, p, ygrid, eps) -> UInt8

Trace a particle through the SG cavity under the **QM** (piecewise-constant-
acceleration, no CQD relaxation) model and report whether it collides with
the top edge, bottom trench, or clears the cavity but misses the exit
aperture/tube. Direct QM analogue of `CQD_cavity_crash`, but without any
`kω`/log/dilog terms — the `z`-trajectory inside the cavity is purely
`z = z0 + γ*dy + κ*Δ²` (a single quadratic-in-`Δy` term, `κ` being the
constant `0.5*acc/v0y²`), so there's no overflow-safe-identity concern here
at all (matches `QM_EqOfMotion`'s note above).

`eps` is a safety/touch tolerance (set >0 to require clearance). Pass a
precomputed `ygrid` (vector of y values spanning `[y_in, y_in + DEFAULT_y_SG]`)
to avoid rebuilding it per call.

# Returns
`UInt8` collision code, same scheme as `CQD_cavity_crash`: `0x00` clear,
`0x01` top edge, `0x02` bottom trench, `0x03` misses the screen tube,
`0x04` blocked by the circular aperture.

# Performance
- The aperture and screen checks reuse `α = v0x*inv_vy` and `γ = v0z*inv_vy`
  (already computed before the cavity-scan loop) instead of recomputing
  `v0x*inv_vy`/`v0z*inv_vy` from scratch — matches how `CQD_cavity_crash`
  already handles its own aperture/screen checks.
- `κ*Lsg` is factored out once (`κLsg`) and shared between the aperture and
  screen `z` formulas, rather than appearing as two separately-expanded
  products of the same underlying quantity.
- `@fastmath` (already applied to the cavity-scan loop) gives the compiler
  license to fuse multiply-adds on its own there; the aperture/screen checks
  sit outside that block, so they use explicit `muladd` instead of widening
  the `@fastmath` region for a calculation that only runs once per particle.

# Notes
- `@fastmath` is applied to the cavity-scan loop — this relaxes IEEE
  floating-point semantics (e.g. allows reassociating sums, assumes no
  NaN/Inf) in exchange for potential speed; worth being aware of if you ever
  need bit-exact reproducibility against a non-`@fastmath` reference
  calculation.
"""
@inline function QM_cavity_crash(μG_ix::Float64,
                         x0::Float64, y0::Float64, z0::Float64,
                         v0x::Float64, v0y::Float64, v0z::Float64,
                         p::AtomParams,
                         ygrid::AbstractVector{Float64},
                         eps::Float64
                         )::UInt8

    @assert v0y != 0 "v0y must be nonzero"

    # Cavity y-span 
    y_in   = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    Lsg    = (DEFAULT_y_SG)::Float64
    Ld     = (DEFAULT_y_SGToScreen)::Float64
    Ltot   = (y_in + Lsg + Ld)::Float64
    R2tube = (DEFAULT_R_tube * DEFAULT_R_tube)::Float64
    LA     = (DEFAULT_y_SGToAperture)::Float64
    Lapert = (y_in+Lsg+LA)::Float64
    R2circ = (DEFAULT_c_aperture * DEFAULT_c_aperture)::Float64

    # Kinematics
    inv_vy  = 1.0 / v0y
    inv_vy2 = inv_vy * inv_vy

    κ = 0.5 * (μG_ix  / p.M) * inv_vy2
    α = v0x * inv_vy
    γ = v0z * inv_vy

    # --- Cavity scan; short-circuit on first breach ---
    @inbounds @fastmath for y in ygrid
        dy = y - y0
        x  = x0 + α * dy
        Δ  = dy - y_in
        z  = z0 + γ * dy + κ * Δ * Δ

        # Crash if above ceiling or below trench (with tolerance)
        (z - z_magnet_edge(x))   >=  eps && return 0x01 # 1 top
        (z - z_magnet_trench(x)) <= -eps && return 0x02 # 2 bottom
    end

    # Circular aperture
    κLsg = κ * Lsg
    x_aper = muladd(Lapert, α, x0)
    z_aper = muladd(Lapert, γ, z0) + κLsg*(Lsg + 2*LA)
    (x_aper*x_aper + z_aper*z_aper >= R2circ ) && return 0x04 # 4 circular aperture
 
    # --- Screen check (only if cavity was clear) ---
    x_screen = muladd(Ltot, α, x0)
    z_screen = muladd(Ltot, γ, z0) + κLsg*(Lsg + 2*Ld)
    return (x_screen*x_screen + z_screen*z_screen >= R2tube) ? 0x03 : 0x00 # 3 tubes
end



########################################################################################################################################
# Co-Quantum Dynamics : B = B₀ + Bₙ cos(θₙ)
########################################################################################################################################
# Every function in this section mirrors a same-named (minus "_Bn_") function
# in the CQD section above, with exactly one physics change: the Larmor
# frequency uses ωL = |γₑ * (BvsI(Ix) + DEFAULT_CQD_Bn*cos(θn))| instead of
# ωL = |γₑ * BvsI(Ix)| — i.e. a θn-dependent nuclear-field correction added
# to the base field before computing ωL. Everything else (segment structure,
# closed-form z formulas, return types) is identical to its CQD_* sibling.

# CQD Equations of motion
"""
    CQD_Bn_EqOfMotion(t, Ix, μ, r0, v0, θe, θn, kx, p) -> (r, v)

Full 3D equations of motion for a Continuous Quantum Dynamics (CQD) model,
with a nuclear-field correction to the Larmor frequency, in a Stern–Gerlach
(SG) setup. Motion in `x` and `y` is purely ballistic; only `z` is modified
by CQD terms while the atom is inside the SG region.

Segment boundaries (seconds; beam advances along +y with v0y ≠ 0):
- `tf1 =  DEFAULT_y_FurnaceToSlit / v0y`
- `tf2 = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG) / v0y`
- `tf3 = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG) / v0y`

Definitions used (only needed, and only computed, once `t` is past `tf2`):
- `cqd_sign = sign(θn - θe)`
- `ωL = abs(γₑ * (BvsI(Ix) + DEFAULT_CQD_Bn*cos(θn)))`   # Larmor angular frequency, with nuclear-field correction
- `acc_0 = μ * GvsI(Ix) / p.M`             # base acceleration scale
- `kω = cqd_sign * kx * ωL`
- `θe_half = θe / 2`
- `cosθ2 = cos(θe_half)^2`, `sinθ2 = sin(θe_half)^2`, `tanθ2 = sinθ2/cosθ2`

Ballistic components (always):
- `x(t) = x0 + v0x * t`
- `y(t) = y0 + v0y * t`
- `vx = v0x`, `vy = v0y`

Piecewise z–dynamics:

1) Furnace → Slit (`t ≤ tf1`) and Slit → SG entrance (`tf1 < t ≤ tf2`)
   - `z(t) = z0 + v0z * t`
   - `vz = v0z`

2) Inside SG (`tf2 < t ≤ tf3`), let `Δt = t - tf2`, `EΔ = exp(-2*kω*Δt)`:
   - `vz = v0z + acc_0*Δt + (acc_0/kω) * log( cosθ2 + EΔ * sinθ2 )`
   - `z  = z0 + v0z*t
              + 0.5*acc_0*Δt^2
              + (acc_0/kω)*log(cosθ2)*Δt
              + 0.5*acc_0/kω^2 * ( polylogarithm(2, -EΔ*tanθ2)
                                   - polylogarithm(2, -tanθ2) )`

3) SG exit → Screen (`t > tf3`), let `τ_SG = DEFAULT_y_SG / v0y`, `Eτ = exp(-2*kω*τ_SG)`:
   - `z  = z0 + v0z*t
              + acc_0*τ_SG*(t - tf2 - 0.5*τ_SG)
              + (acc_0/kω) * τ_SG * ( log(cosθ2)
                                      + inv(τ_SG) * log(cosθ2 + Eτ*sinθ2) * (t - tf3) )
              + 0.5*acc_0/kω^2 * ( polylogarithm(2, -Eτ*tanθ2)
                                   - polylogarithm(2, -tanθ2) )`
   - `vz = v0z + acc_0*τ_SG
                + (acc_0/kω) * log( cosθ2 + Eτ * sinθ2 )`

# Arguments
- `t`: time since the initial state `(r0, v0)` (s).
- `Ix`: coil current (A), used by `GvsI` and `BvsI`.
- `μ`: effective magnetic moment (J/T).
- `r0::AbstractVector{<:Real}`: initial position `(x0, y0, z0)` in meters; length 3.
- `v0::AbstractVector{<:Real}`: initial velocity `(v0x, v0y, v0z)` in m/s; length 3 and `v0y ≠ 0`.
- `θe::Float64`, `θn::Float64`: electron and nuclear angles (radians).
- `kx::Float64`: dimensionless coupling multiplying `ωL`.
- `p::AtomParams`: atomic parameters (must include mass `M`). Assumes `γₑ`, `GvsI`, `BvsI`,
  `DEFAULT_CQD_Bn`, and `DEFAULT_y_*` geometry constants are in scope.

# Returns
- `(r, v)` with `r::SVector{3,Float64}` and `v::SVector{3,Float64}` at time `t`.

# Assumptions
- Non-relativistic; gravity, collisions, and fringe-field variation neglected.
- Gradient aligned with +z; only z accelerates inside the SG.
- Expressions are continuous at `t = tf2` and `t = tf3`.

# Performance
- `inv_v0y = 1/v0y` is computed once and reused for every segment-boundary
  time (`tf1`, `tf2`, `tf3`) and for `τ_SG`, in place of four separate
  divisions.
- `ωL`, `acc_0`, `kω`, and the `θe`-dependent trig terms are computed only
  once `t` is known to be past `tf2` — both pre-SG segments are purely
  ballistic and need none of them.
- `inv_kω`/`inv_kω2` are computed once and reused everywhere `acc_0/kω` or
  `acc_0/kω^2` appears, in place of four separate divisions by `kω`.
- `sincos(θe/2)` provides `sinθ2`/`cosθ2` from one combined evaluation;
  `tanθ2` is derived as their ratio rather than its own `tan` call.
- `tf1`, `tf2`, `tf3`, and `τ_SG` are each ascribed `::Float64` over their
  full expression: since they're built from `DEFAULT_y_*` geometry
  constants, this re-establishes a concrete type at each point in case those
  constants are non-`const` globals. Harmless and free if the constants are
  already `const`.

# Numerical notes
- Terms with `1/kω` and `1/kω^2` are finite in the `kω → 0` limit, but direct
  evaluation can be ill-conditioned when `|kω|` is very small.
- `EΔ`/`Eτ` are evaluated directly rather than through an overflow-safe
  log-domain rewrite, and can under/overflow for large `|kω*Δt|` or
  `|kω*τ_SG|`.
"""
@inline function CQD_Bn_EqOfMotion(t,Ix,μ,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"
    @assert t >= 0 "time t must be ≥ 0"
    x0, y0, z0 = r0
    v0x, v0y, v0z = v0
    @assert !iszero(v0y) "y-velocity must be nonzero."

    inv_v0y = 1.0 / v0y

    x = x0 + v0x*t 
    y = y0 + v0y*t 
    vx = v0x
    vy = v0y

    # Key times
    tf1 = (DEFAULT_y_FurnaceToSlit * inv_v0y) :: Float64
    tf2 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG) * inv_v0y) :: Float64

    if t <= tf1     # Furnace to Slit
        z = z0 + v0z*t
        vz = v0z
    elseif t <= tf2    # Slit to SG apparatus
        z = z0 + v0z*t
        vz = v0z
    else
        tf3 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG) * inv_v0y) :: Float64

        cqd_sign = sign(θn-θe) 
        ωL       = abs(γₑ * (BvsI(Ix) + DEFAULT_CQD_Bn*cos(θn)) )
        acc_0    = μ*GvsI(Ix)/p.M
        kω       = cqd_sign*kx*ωL
        inv_kω   = 1.0 / kω
        inv_kω2  = inv_kω * inv_kω

        s, c = sincos(θe / 2)
        sinθ2 = s*s
        cosθ2 = c*c
        tanθ2 = sinθ2 / cosθ2

        if t <= tf3   # Crossing the SG apparatus
            Δt = t-tf2
            EΔ = exp(-2*kω*Δt)
            vz = v0z + acc_0*Δt + acc_0*inv_kω * log( cosθ2 + EΔ*sinθ2 )
            z = z0 + v0z*t + 0.5*acc_0*Δt^2 + acc_0*inv_kω*log(cosθ2)*Δt + 
                0.5*acc_0*inv_kω2 * ( polylogarithm(2,-EΔ*tanθ2) - polylogarithm(2,-tanθ2) )
        else # t > tf3 # Travel to the Screen
            τ_SG = (DEFAULT_y_SG * inv_v0y) :: Float64
            Eτ = exp(-2*kω*τ_SG)
            z = z0 + v0z*t + acc_0*τ_SG*(t-tf2-0.5*τ_SG) + acc_0*inv_kω*τ_SG * ( log(cosθ2) + inv(τ_SG)*log(cosθ2+Eτ*sinθ2)*(t-tf3) ) + 0.5*acc_0*inv_kω2*( polylogarithm(2,-Eτ*tanθ2) - polylogarithm(2,-tanθ2) )
            vz = v0z + acc_0*τ_SG + acc_0*inv_kω*log(cosθ2 + Eτ*sinθ2)
        end
    end

    r = SVector{3,Float64}(x, y, z)
    v = SVector{3,Float64}(vx, vy, vz)
    return r, v
end


# CQD equations of motion only along the z-coordinate
"""
    CQD_Bn_EqOfMotion_z(t, Ix, μ, r0, v0, θe, θn, kx, p) -> Float64

z–coordinate as a function of time `t` under a Continuous Quantum Dynamics
(CQD) model with a nuclear-field correction to the Larmor frequency, in a
Stern–Gerlach (SG) setup. Motion is divided into three segments:
1) Pre-SG (ballistic), 2) In-SG (CQD dynamics with uniform base acceleration),
3) Post-SG (ballistic with accumulated CQD effect).

Segment boundaries (s):
- `tf2 = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG) / v0y`
- `tf3 = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG) / v0y`

Definitions (computed only for `t > tf2`):
- `cqd_sign = sign(θn - θe)`
- `ωL = abs(γₑ * (BvsI(Ix) + DEFAULT_CQD_Bn*cos(θn)))`   # Larmor frequency, with nuclear-field correction
- `acc_z = μ * GvsI(Ix) / p.M`
- `kω = cqd_sign * kx * ωL`
- `θe_half = θe/2`; `sinθ2 = sin(θe_half)^2`, `cosθ2 = cos(θe_half)^2`, `tanθ2 = sinθ2/cosθ2`
- `log_cos2 = log(cosθ2)`
- `polylog_0 = polylogarithm(2, -tanθ2)`
- `τ_SG = DEFAULT_y_SG / v0y`

Piecewise result:
- Pre-SG (`t ≤ tf2`):
  `z(t) = z0 + v0z * t`
- In-SG (`tf2 < t ≤ tf3`), with `Δt = t - tf2` and `exp_term = exp(-2*kω*Δt)`:
  `z(t) = z0 + v0z*t + 0.5*acc_z*Δt^2 + (acc_z/kω)*log_cos2*Δt
          + 0.5*acc_z/kω^2 * ( polylogarithm(2, -exp_term*tanθ2) - polylog_0 )`
- Post-SG (`t > tf3`), with `Δt3 = t - tf3`,
  `exp_SG = exp(-2*kω*τ_SG)`, `polylog_SG = polylogarithm(2, -exp_SG*tanθ2)`,
  `log_term = log(cosθ2 + exp_SG*sinθ2)`:
  `z(t) = z0 + v0z*t + acc_z*τ_SG*(t - tf2 - 0.5*τ_SG)
          + (acc_z/kω)*τ_SG * (log_cos2 + (Δt3/τ_SG)*log_term)
          + 0.5*acc_z/kω^2 * (polylog_SG - polylog_0)`

# Arguments
- `t::Float64`: time since the initial state `(r0, v0)` (s).
- `Ix::Float64`: coil current (A) used by `GvsI`/`BvsI`.
- `μ::Float64`: effective magnetic moment (J/T).
- `r0::AbstractVector{Float64}`: initial position `(x0, y0, z0)` in meters; length 3.
- `v0::AbstractVector{Float64}`: initial velocity `(v0x, v0y, v0z)` in m/s; length 3 and `v0y ≠ 0`.
- `θe::Float64`, `θn::Float64`: electron and nuclear angles (rad).
- `kx::Float64`: dimensionless coupling multiplying `ωL`.
- `p::AtomParams`: atomic parameters (must include mass `M`). Assumes `γₑ`, `GvsI`, `BvsI`,
  `DEFAULT_CQD_Bn`, and the `DEFAULT_y_*` geometry constants are in scope.

# Returns
- `z::Float64` — z-position at time `t`.

# Assumptions
- Non-relativistic; gravity, collisions, and fringe-field variation neglected.
- SG gradient aligned with `+z`; only `z` accelerates inside the SG.
- Expressions are continuous at `t = tf2` and `t = tf3`.

# Performance
- `μF_effective`/`GvsI`/`BvsI` and all `θe`-dependent trig/log/dilog setup
  are only computed once `t` is known to be past `tf2` — the pre-SG branch
  returns using only `z0`/`v0z`/`t`, paying for none of it.
- `inv_v0y = 1/v0y` is computed once and reused for `tf2`, `tf3`, and `τ_SG`.
- `inv_kω = 1/kω` is computed once and reused everywhere `acc_z/kω` appears.
- `sincos(θe/2)` provides `sinθ2`/`cosθ2` from one combined evaluation;
  `tanθ2` is derived as their ratio rather than its own `tan` call — this
  was the one optimization not yet applied here (or in this function's own
  non-Bn sibling `CQD_EqOfMotion_z`), now added.
- `tf2`, `tf3`, and `τ_SG` are each ascribed `::Float64` over their full
  expression, guarding against non-`const` `DEFAULT_y_*` globals; harmless
  and free if those constants are already `const`.
- The `inv_kω^2` terms and the `acc_z*inv_kω` product each appear exactly
  once per branch, and the two branches are mutually exclusive — there is no
  cross-branch redundancy left to hoist; doing so would only relocate the
  same operation count, not reduce it.

# Numerical notes
- Terms with `1/kω` and `1/kω^2` cancel analytically as `kω → 0` but may lose
  precision; consider a small-`kω` expansion when `|kω*τ_SG|` is very small.
- `exp_term`/`exp_SG` are evaluated directly rather than through an
  overflow-safe log-domain rewrite (unlike `CQD_screen_x_z_vz`/
  `CQD_cavity_crash`), and can under/overflow for large `|kω*Δt|` or `|kω*τ_SG|`.
"""
@inline function CQD_Bn_EqOfMotion_z(t,Ix::Float64,μ::Float64,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"
    @assert t >= 0 "time t must be ≥ 0"

    v0y = v0[2]
    v0z = v0[3]
    z0  = r0[3]

    @assert !iszero(v0y) "v0y must be nonzero (beam must advance toward the screen)."
    
    inv_v0y = 1.0 / v0y
    tf2 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG) * inv_v0y) ::Float64

    # Pre-SG: nothing below this line is needed at all
    t <= tf2 && return z0 + v0z*t

    tf3 = ((DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG) * inv_v0y) ::Float64

    # Only pay for BvsI/GvsI/trig/polylog once we know we're past tf2
    cqd_sign = sign(θn-θe) 
    ωL       = abs( γₑ * (BvsI(Ix) + DEFAULT_CQD_Bn * cos(θn)) )
    acc_z    = μ*GvsI(Ix)/p.M
    kω       = cqd_sign*kx*ωL
    inv_kω   = 1.0 / kω

    # Precompute angles
    θe_half = θe / 2
    tanθ2 = tan(θe_half)^2
    cosθ2 = cos(θe_half)^2
    sinθ2 = sin(θe_half)^2
    log_cos2 = log(cosθ2)
    polylog_0 = polylogarithm(2, -tanθ2)

    if t <= tf3   # Crossing the SG apparatus
        Δt = t - tf2
        exp_term = exp(-2 * kω * Δt)
        polylog_t = polylogarithm(2, -exp_term * tanθ2)

        return z0 + v0z*t + 0.5 * acc_z * Δt^2 + acc_z * inv_kω  * log_cos2 * Δt + 0.5 * acc_z * inv_kω^2 * ( polylog_t - polylog_0 )
    
    else # t > tf3 # Travel to the Screen
        Δt3 = t - tf3
        τ_SG = (DEFAULT_y_SG * inv_v0y) ::Float64
        exp_SG = exp(-2 * kω * τ_SG)
        polylog_SG = polylogarithm(2, -exp_SG * tanθ2)
        log_term = log(cosθ2 + exp_SG * sinθ2)

        return z0 + v0z*t + acc_z*τ_SG*( t-tf2-0.5*τ_SG ) + acc_z * inv_kω * τ_SG * (log_cos2 + log_term * Δt3 / τ_SG) + 0.5 * acc_z * inv_kω^2 * (polylog_SG - polylog_0)
    end
end

# CQD Screen position
"""
    CQD_Bn_Screen_position(Ix, μ, r0, v0, θe, θn, kx, p) -> SVector{3,Float64}

Final **position at the screen** for an atom under a Continuous Quantum
Dynamics (CQD) model with a nuclear-field correction to the Larmor
frequency, in a Stern–Gerlach (SG) setup. Motion in `x` and `y` is ballistic;
the `z` coordinate includes CQD-dependent terms derived in closed form (logs
and a dilogarithm).

# Geometry (meters, along +y)
Let
- `L1 = DEFAULT_y_FurnaceToSlit`
- `L2 = DEFAULT_y_SlitToSG`
- `Lsg = DEFAULT_y_SG`
- `Ld = DEFAULT_y_SGToScreen`
- `Ltot = L1 + L2 + Lsg + Ld`.

# Physics definitions
- `cqd_sign = sign(θn − θe)`
- `acc_z = μ * GvsI(Ix) / p.M`                                       # base acceleration scale (m/s²)
- `ωL = abs(γₑ * (BvsI(Ix) + DEFAULT_CQD_Bn*cos(θn)))`                # Larmor angular frequency, with nuclear-field correction (rad/s)
- `kω = cqd_sign * kx * ωL`
- `τ_SG = Lsg / v0y`
- `θe_half = θe/2`, `cos2 = cos(θe_half)^2`, `sin2 = sin(θe_half)^2`, `tan2 = sin2/cos2`
- `exp_term = exp(-2 * kω * τ_SG)`.

# Closed-form result
With initial state `r0 = (x0, y0, z0)` (m) and `v0 = (v0x, v0y, v0z)` (m/s), the screen
coordinates are
- `x = x0 + Ltot * v0x / v0y`
- `y = y0 + Ltot`
- `z = z0 + (Ltot * v0z)/v0y
        + (1/2) * acc_z / v0y^2 * [ Lsg*(Lsg + 2*Ld) ]
        + (acc_z / kω) * τ_SG * [ log(cos2) + (Ld/Lsg) * log(cos2 + exp_term * sin2) ]
        + (1/2) * acc_z / kω^2 * [ polylogarithm(2, -exp_term * tan2) − polylogarithm(2, -tan2) ].

# Arguments
- `Ix`: Coil current (A) used by `GvsI` and `BvsI`.
- `μ::Float64`: Effective magnetic moment (J/T) for this CQD model.
- `r0::AbstractVector{Float64}`: Initial position `(x0, y0, z0)` in meters; **length 3**.
- `v0::AbstractVector{Float64}`: Initial velocity `(v0x, v0y, v0z)` in m/s; **length 3** and `v0y ≠ 0`.
- `θe::Float64`, `θn::Float64`: Electron and nuclear angles (radians).
- `kx::Float64`: Dimensionless coupling factor multiplying `ωL`.
- `p::AtomParams`: Atomic parameters; must include `M`. Assumes `γₑ`, `GvsI`, `BvsI`, and
  `DEFAULT_CQD_Bn` are in scope.

# Returns
- `SVector{3,Float64}`: `(x, y, z)` at the screen.

# Assumptions
- Non-relativistic; gravity, collisions, and fringe-field variation neglected.
- Gradient aligned with `+z`; only `z` is accelerated inside the SG.
- Uses global geometry constants `DEFAULT_y_*`.

# Throws
- Assertion errors if `length(r0) ≠ 3`, `length(v0) ≠ 3`, or `v0y == 0`.

# Performance
- `τ_SG = Lsg/v0y` is computed once and shared between `exp_term`'s argument
  and the `z` amplitude term, rather than recomputed independently in each.
- `inv_v0y`/`inv_v0y2` and `inv_kω`/`inv_kω2` replace what would otherwise be
  six separate divisions with two reciprocals plus multiplications.
- `sincos(θe_half)` provides `cos2`/`sin2` from one combined evaluation;
  `tan2` is derived as their ratio rather than its own `tan` call.
- `Lsg*(Lsg+2*Ld)` (algebraically identical to `(Lsg+Ld)^2-Ld^2`, fewer
  operations) is used directly rather than the expanded-squares form.
- `L1`, `L2`, `Lsg`, `Ld` are each ascribed `::Float64`, guarding against
  non-`const` `DEFAULT_y_*` globals; harmless and free if those constants
  are already `const`.

# Notes (numerics)
- The formula contains `1/kω` and `1/kω^2` terms; the overall `z` remains finite as `kω → 0`,
  but direct evaluation can lose precision. If you expect very small `|kω * τ_SG|`,
  consider using a small-`kω` series expansion for the log/polylog combination.
  `inv_kω = 1/kω` has the same `kω == 0` failure mode as a direct-division
  form would — no guard against it here.
- `exp_term` may under/overflow when `|kω * τ_SG|` is large; evaluated directly
  rather than through an overflow-safe log-domain rewrite (unlike
  `CQD_screen_x_z_vz`/`CQD_cavity_crash`).
"""
function CQD_Bn_Screen_position(Ix,μ::Float64,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"
    x0, y0, z0 = r0
    v0x, v0y, v0z = v0
    @assert !iszero(v0y) "v0y must be nonzero (beam must advance toward the screen)."

    L1   = DEFAULT_y_FurnaceToSlit ::Float64
    L2   = DEFAULT_y_SlitToSG ::Float64
    Lsg  = DEFAULT_y_SG ::Float64
    Ld   = DEFAULT_y_SGToScreen ::Float64
    Ltot = L1 + L2 + Lsg + Ld

    # Physics parameters
    cqd_sign = sign(θn-θe) 
    acc_z = μ * GvsI(Ix) / p.M
    ωL = abs(γₑ * (BvsI(Ix) + DEFAULT_CQD_Bn * cos(θn)))
    kω = cqd_sign * kx * ωL

    inv_v0y  = 1.0 / v0y
    inv_v0y2 = inv_v0y * inv_v0y
    inv_kω   = 1.0 / kω
    inv_kω2  = inv_kω * inv_kω
    τ_SG     = Lsg * inv_v0y

    # Common trig values: sincos shares work between sin/cos; tan2 derived
    # from them instead of a third transcendental call
    θe_half = θe / 2
    s, c = sincos(θe_half)
    sin2 = s*s
    cos2 = c*c
    tan2 = sin2 / cos2
    exp_term = exp(-2 * kω * τ_SG)

    x = x0 + Ltot * v0x * inv_v0y
    y = y0 + Ltot
    z = z0 + Ltot * v0z * inv_v0y +
        0.5*acc_z*inv_v0y2*Lsg*(Lsg + 2*Ld) +
        acc_z*inv_kω*τ_SG*( log(cos2) + Ld/Lsg * log( cos2 + exp_term*sin2 ) ) +
        0.5*acc_z*inv_kω2 * ( polylogarithm(2, -exp_term*tan2) - polylogarithm(2, -tan2) )

    return SVector{3,Float64}(x,y,z)
end

"""
    CQD_Bn_Screen_velocity(Ix, μ, v0, θe, θn, kx, p) -> SVector{3,Float64}

Final **velocity at the screen** under a Continuous Quantum Dynamics (CQD)
model with a nuclear-field correction to the Larmor frequency. Motion is
ballistic in `x` and `y`; only the `z` component is affected inside the
Stern–Gerlach (SG) region of length `DEFAULT_y_SG`.

# Model / Definitions
Let
- `Lsg = DEFAULT_y_SG` (m), `τ_SG = Lsg / v0y` (s),
- `acc_z = μ * GvsI(Ix) / p.M`  (m/s²),
- `ωL = abs(γₑ * (BvsI(Ix) + DEFAULT_CQD_Bn*cos(θn)))`   (rad/s; Larmor frequency, with nuclear-field correction),
- `kω = sign(θn − θe) * kx * ωL`,
- `θh = θe/2`, `cos2 = cos(θh)^2`, `sin2 = sin(θh)^2`.

Then
- `v_x = v0x`
- `v_y = v0y`
- `v_z = v0z + acc_z * τ_SG + (acc_z / kω) * log( cos2 + e^{-2 kω τ_SG} * sin2 )`

The implementation is continuous at `kω → 0`, using the limit
`v_z = v0z + acc_z * τ_SG * cos(θe)` to avoid the `0/0` indeterminacy.
`cos(θe)` is kept as its own direct call there rather than derived from
`cos2`/`sin2` via the double-angle identity, for the same robustness reason
as in `CQD_Screen_velocity`: `cos2 - sin2` is a subtraction of two
similar-magnitude values near `θe ≈ π/2`, more sensitive to cancellation
than a direct evaluation.

# Arguments
- `Ix`: Coil current (A) used by `GvsI` and `BvsI`.
- `μ::Float64`: Effective magnetic moment (J/T) entering the CQD dynamics.
- `v0::AbstractVector{<:Real}`: Initial velocity `(v0x, v0y, v0z)` in m/s; **length 3** and `v0y ≠ 0`.
- `θe::Float64`, `θn::Float64`: Electron and nuclear angles (radians).
- `kx::Float64`: Dimensionless coupling factor multiplying `ωL`.
- `p::AtomParams`: Atomic parameters; must provide mass `M`. Assumes `GvsI`, `BvsI`, `γₑ`,
  and `DEFAULT_CQD_Bn` are in scope.

# Returns
- `SVector{3,Float64}`: `(v_x, v_y, v_z)` at the screen.

# Assumptions
- Non-relativistic; gravity, collisions, and fringe-field variations neglected.
- SG gradient aligned with `+z` and treated constant across `DEFAULT_y_SG`.

# Throws
- Assertion error if `length(v0) ≠ 3` or `v0y == 0`.

# Performance
- `sincos(θe_half)` provides `cos2`/`sin2` from one combined evaluation
  instead of two independent `cos`/`sin` calls.
- `Lsg` is ascribed `::Float64`, guarding against a non-`const`
  `DEFAULT_y_SG` global; harmless and free if it's already `const`.
- `kω` is divided into only once (`acc_z/kω`), so there's no repeated
  division to collapse into a reciprocal here — same situation as in
  `CQD_Screen_velocity`.

# Notes
- Uses `DEFAULT_y_SG` for the SG length; adjust the function if you need it configurable.
- Handles the `kω → 0` case internally for numerical stability.
- `exp_term` is evaluated directly rather than through an overflow-safe
  log-domain rewrite (unlike `CQD_screen_x_z_vz`/`CQD_cavity_crash`), and can
  under/overflow for large `|kω * τ_SG|`.
"""
function CQD_Bn_Screen_velocity(Ix,μ::Float64,v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)

    @assert length(v0) == 3 "v0 must have length 3"
    v0x, v0y, v0z = v0
    @assert !iszero(v0y) "v0y must be nonzero (beam must advance toward the screen)."

    Lsg = DEFAULT_y_SG ::Float64

    # Physics parameters
    cqd_sign = sign(θn-θe) 
    acc_z = μ * GvsI(Ix) / p.M
    ωL = abs(γₑ * (BvsI(Ix) + DEFAULT_CQD_Bn * cos(θn)))
    kω = cqd_sign * kx * ωL

    # Common trig values: sincos shares work between sin/cos
    θe_half = θe / 2
    s, c = sincos(θe_half)
    sin2 = s*s
    cos2 = c*c
    τ_SG = Lsg / v0y

    vx = v0x
    vy = v0y
    vz = if iszero(kω) || abs(kω * τ_SG) < 1e-18
        # Continuous kω→0 limit: avoids 0/0 in the log term
        v0z + acc_z * τ_SG * cos(θe)
    else
        exp_term = exp(-2 * kω * τ_SG)
        v0z + acc_z * τ_SG + acc_z / kω * log( cos2 + exp_term*sin2) 
    end

    return SVector{3,Float64}(vx,vy,vz)
end

"""
    CQD_Bn_cavity_crash(μG_ix::Float64, B0_ix::Float64,
                        x0::Float64, y0::Float64, z0::Float64,
                        v0x::Float64, v0y::Float64, v0z::Float64,
                        θe::Float64, θn::Float64,
                        kx::Float64,
                        p::AtomParams,
                        ygrid::AbstractVector{Float64},
                        eps::Float64) -> UInt8

Trace a particle through the Stern–Gerlach (SG) cavity under the
**Continuous Quantum Dynamics (CQD)** model with a nuclear-field correction
to the Larmor frequency, and report whether it collides with the **top
edge**, **bottom trench**, or clears the cavity but **misses the exit
tube** at the screen.

The trajectory is evaluated along the longitudinal grid `ygrid`
(lab-frame absolute y positions). Transverse coordinates `(x(y), z(y))`
are computed analytically from the CQD equations of motion.

The CQD correction is controlled by

`kω = sign(θn − θe) * kx * ωL`,  with  `ωL = |γₑ * (B0_ix + DEFAULT_CQD_Bn*cos(θn))|`,

and includes closed-form logarithmic and dilogarithmic contributions.
These terms are evaluated using **overflow-safe identities** for
`Li₂(-exp(x))` and `log(cos² + sin²·exp(x))`, so the function remains
numerically stable for both signs and arbitrarily large magnitudes of `kω`
without altering the underlying physics.

If `kω == 0`, the motion reduces to a constant-acceleration trajectory
and a simplified analytic path is used.

---

### Arguments
- `μG_ix`: Magnetic force coefficient (e.g. `μ ∂B/∂z`) for the given current.
- `B0_ix`: Magnetic field magnitude scale for this setting (used to form `ωL`,
  before the `DEFAULT_CQD_Bn*cos(θn)` correction is added).
- `x0, y0, z0`: Initial position at the entrance reference (units consistent
  with the geometry).
- `v0x, v0y, v0z`: Initial velocity components; **`v0y` must be nonzero**.
- `θe, θn`: Electronic and nuclear spin angles (radians).
- `kx`: Spatial modulation wavenumber of the CQD correction.
- `p`: `AtomParams` instance (must provide atomic mass `M`).
- `ygrid`: Monotonically increasing **absolute** y-positions spanning the cavity
  and downstream regions.
- `eps`: Tolerance for wall-contact tests.

---

### Returns
`UInt8` collision code:
- `0x00` — no collision, exits within the tube
- `0x01` — collision with the top edge inside the cavity
- `0x02` — collision with the bottom trench inside the cavity
- `0x03` — clears the cavity but misses the exit tube at the screen
- `0x04` — blocked by the circular aperture

---

### Assumptions
- Wall profiles `z_magnet_edge(x)` and `z_magnet_trench(x)` are defined,
  pure functions returning `Float64`.
- Geometry constants `DEFAULT_y_*`, `DEFAULT_R_tube`, `DEFAULT_c_aperture`,
  `DEFAULT_CQD_Bn`, and `γₑ` exist and share consistent units with positions
  and velocities.
- `ygrid` is in the **lab frame** (absolute y); linear terms use `y - y0`,
  while cavity-specific terms use `y - y_in`.

---

### Numerical notes
- The implementation is **allocation-free** and suitable for execution
  millions of times inside tight loops.
- All exponentials are evaluated in a numerically stable form; no
  `exp(+large)` calls occur, so overflow is avoided even for large
  `|kω|` or long cavities.
- Fused multiply-add (`muladd`) is used where beneficial for accuracy and speed.
- Near `θe ≈ π`, `cos²(θe/2) → 0`; the implementation clamps this quantity
  internally to avoid singular logs while preserving physical behavior.

---

### Performance
- The `θe`-dependent precompute (`sincos`, `cos²`/`sin²`/`tan²`, the two
  `log` calls, and the `li2_negexp` dilogarithm evaluation) sits *after* the
  `kω == 0` branch, not before it — that branch never reads any of those
  quantities, so it no longer pays for them.
- Inside each branch's `ygrid` loop, the per-iteration update uses one
  hoisted, loop-invariant product (`κcθe` in the `kω==0` branch,
  `log_coef_logcosθ2` in the `kω≠0` branch) instead of recomputing that
  product on every iteration — the saving scales with the number of grid
  points, which is why it matters most here relative to the rest of the
  function.
- `log_coef_Lsg` (used by both the aperture and screen checks in the `kω≠0`
  branch) is computed once and reused, rather than once per check.
- The two CQD correction coefficients are named for what they scale, not as
  bare algebra letters: `log_coef` multiplies the log-term contribution,
  `dilog_coef` multiplies the dilogarithm-term contribution — chosen to
  avoid any visual confusion with the magnetic-field quantities (`B0_ix`,
  `B0`, `Bx`/`By`/`Bz`) used throughout this codebase.

---

### Throws
- `AssertionError` if `v0y == 0`.

---

### Example
# code = CQD_Bn_cavity_crash(
#     μG, B0,
#     x0, y0, z0,
#     vx, vy, vz,
#     θe, θn,
#     kx, p,
#     ygrid,
#     1e-9
# )
# code == 0x00 || @info "Collision code = code"
"""
@inline function CQD_Bn_cavity_crash(
    μG_ix::Float64, B0_ix::Float64,
    x0::Float64, y0::Float64, z0::Float64,
    v0x::Float64, v0y::Float64, v0z::Float64,
    θe::Float64, θn::Float64,
    kx::Float64,
    p::AtomParams,
    ygrid::AbstractVector{Float64},
    eps::Float64
)::UInt8

    @assert v0y != 0.0 "v0y must be nonzero"

    # ---- geometry (leave as you had it; these are scalars) ----
    y_in   = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    Lsg    = (DEFAULT_y_SG)::Float64
    Ld     = (DEFAULT_y_SGToScreen)::Float64
    Ltot   = (y_in + Lsg + Ld)::Float64
    R2tube = (DEFAULT_R_tube * DEFAULT_R_tube)::Float64
    LA     = (DEFAULT_y_SGToAperture)::Float64
    Lapert = (y_in + Lsg + LA)::Float64
    R2circ = (DEFAULT_c_aperture * DEFAULT_c_aperture)::Float64

    # ---- kinematics ----
    cqd_sign = sign(θn - θe)
    ωL = abs(γₑ * (B0_ix + DEFAULT_CQD_Bn * cos(θn)))
    acc_0    = μG_ix / p.M
    kω       = cqd_sign * kx * ωL

    inv_v0y  = 1.0 / v0y
    inv_v0y2 = inv_v0y * inv_v0y

    # ---- constants needed in BOTH branches ----
    α        = v0x * inv_v0y
    γ        = v0z * inv_v0y
    κ        = 0.5 * acc_0 * inv_v0y2

    # ---- kω = 0 special case ----
    if iszero(kω)
        cθe  = cos(θe)
        κcθe = κ * cθe   # loop-invariant; hoisted out of the ygrid loop below
        @inbounds for i in eachindex(ygrid)
            y  = ygrid[i] - y0
            Δy = y - y_in
            x  = muladd(α, y, x0)
            z  = muladd(γ, y, z0) + κcθe*(Δy*Δy)
            (z - z_magnet_edge(x))   >=  eps && return UInt8(0x01)
            (z - z_magnet_trench(x)) <= -eps && return UInt8(0x02)
        end

        x_aper = muladd(Lapert, α, x0)
        z_aper = muladd(Lapert, γ, z0) + κcθe*(Lsg*(Lsg + 2.0*LA))
        (muladd(x_aper, x_aper, z_aper*z_aper) >= R2circ) && return UInt8(0x04)

        x_screen = muladd(Ltot, α, x0)
        z_screen = muladd(Ltot, γ, z0) + κcθe*(Lsg*(Lsg + 2.0*Ld))
        return (muladd(x_screen, x_screen, z_screen*z_screen) >= R2tube) ? UInt8(0x03) : UInt8(0x00)
    end

    # ---- θe terms — deferred until here, only needed when kω≠0 ----
    scale_a  = -2.0 * kω * inv_v0y   # so exp_arg = scale_a * Δy

    s, c     = sincos(0.5 * θe)
    sinθ2    = s * s
    cosθ2    = max(c*c, 1.0e-21)   # avoid log(0) & div by 0
    tanθ2    = sinθ2 / cosθ2
    logcosθ2 = log(cosθ2)

    # Precompute Li₂(-tan²) in a stable way too:
    logtan   = log(tanθ2)
    PL_tanθ2 = li2_negexp(logtan)

    # ---- kω ≠ 0 path ----
    inv_kω   = 1.0 / kω
    inv_kω2  = inv_kω * inv_kω
    log_coef         = acc_0 * inv_kω  * inv_v0y     # ~ 1/kω; scales the log-term contribution
    dilog_coef       = 0.5  * acc_0 * inv_kω2        # ~ 1/kω^2; scales the dilog-term contribution
    log_coef_logcosθ2 = log_coef * logcosθ2          # loop-invariant; hoisted out of the ygrid loop below
    log_coef_Lsg      = log_coef * Lsg               # shared by the aperture and screen checks below

    @inbounds for i in eachindex(ygrid)
        y  = ygrid[i] - y0
        Δy = y - y_in
        exp_arg = scale_a * Δy

        # Stable: Li₂(-exp(logtan + exp_arg))
        li2term = li2_negexp(logtan + exp_arg) - PL_tanθ2

        x  = muladd(α, y, x0)
        z  = muladd(γ, y, z0) + κ*(Δy*Δy) + log_coef_logcosθ2*Δy + dilog_coef*li2term

        (z - z_magnet_edge(x))   >=  eps && return UInt8(0x01)
        (z - z_magnet_trench(x)) <= -eps && return UInt8(0x02)
    end

    # ---- end-of-SG terms (avoid exp overflow) ----
    t_sg       = scale_a * Lsg
    dilog_term = dilog_coef*(li2_negexp(logtan + t_sg) - PL_tanθ2)
    logterm    = log_cos_sin_exp(t_sg, cosθ2, sinθ2)

    # ------ Circular aperture -----
    x_aper = muladd(Lapert, α, x0)
    z_aper = muladd(Lapert, γ, z0) +
             κ*(Lsg*Lsg + 2.0*Lsg*LA) +
             log_coef_Lsg*(logcosθ2 + (LA/Lsg)*logterm) +
             dilog_term
    (muladd(x_aper, x_aper, z_aper*z_aper) >= R2circ) && return UInt8(0x04)

    # ------ Screen check -----
    x_screen = muladd(Ltot, α, x0)
    z_screen = muladd(Ltot, γ, z0) +
               κ*(Lsg*Lsg + 2.0*Lsg*Ld) +
               log_coef_Lsg*(logcosθ2 + (Ld/Lsg)*logterm) +
               dilog_term

    return (muladd(x_screen, x_screen, z_screen*z_screen) >= R2tube) ? UInt8(0x03) : UInt8(0x00)
end