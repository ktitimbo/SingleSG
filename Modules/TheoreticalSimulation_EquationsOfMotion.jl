# EQUATIONS OF MOTION

# Co-Quantum Dynamics

# CQD Equations of motion
"""
    CQD_EqOfMotion(t, Ix, μ, r0, v0, θe, θn, kx, p) -> (r, v)

Full 3D equations of motion for a Continuous Quantum Dynamics (CQD) model in a
Stern–Gerlach (SG) setup. Motion in `x` and `y` is purely ballistic; only `z`
is modified by CQD terms while the atom is inside the SG region.

Segment boundaries (seconds; beam advances along +y with v0y ≠ 0):
- `tf1 =  default_y_FurnaceToSlit / v0y`
- `tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG) / v0y`
- `tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v0y`

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

3) SG exit → Screen (`t > tf3`), let `τ_SG = default_y_SG / v0y`, `Eτ = exp(-2*kω*τ_SG)`:
   - `z  = z0 + v0z*t
              + 0.5*acc_0 * ( (t - tf2)^2 - (t - tf3)^2 )
              + (acc_0/kω) * τ_SG * ( log(cosθ2)
                                      + (v0y/default_y_SG) * log(cosθ2 + Eτ*sinθ2) * (t - tf3) )
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
  and `default_y_*` geometry constants are in scope.

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
  rewrite for `log(cosθ2 + E* sinθ2)` improves robustness.
"""
@inline function CQD_EqOfMotion(t,Ix,μ,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"
    @assert t >= 0 "time t must be ≥ 0"
    x0, y0, z0 = r0
    v0x, v0y, v0z = v0
    @assert !iszero(v0y) "y-velocity must be nonzero."


    # Key times
    tf1 =  default_y_FurnaceToSlit / v0y
    tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG) / v0y
    tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v0y
    # tF  = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG + default_y_SGToScreen) / v0y  # (unused here)

    cqd_sign = sign(θn-θe) 
    ωL       = abs(γₑ * BvsI(Ix) )
    acc_0    = μ*GvsI(Ix)/p.M
    kω       = cqd_sign*kx*ωL

    θe_half = θe / 2
    cosθ2 = cos(θe_half)^2
    sinθ2 = sin(θe_half)^2
    tanθ2 = tan(θe_half)^2

    x = x0 + v0x*t 
    y = y0 + v0y*t 
    vx = v0x
    vy = v0y

    if t <= tf1     # Furnace to Slit
        z = z0 + v0z*t
        vz = v0z
    elseif t <= tf2    # Slit to SG apparatus
        z = z0 + v0z*t
        vz = v0z
    elseif t <= tf3   # Crossing the SG apparatus
        Δt = t-tf2
        EΔ = exp(-2*kω*Δt)
        vz = v0z + acc_0*Δt + acc_0/kω * log( cosθ2 + EΔ*sinθ2 )
        z = z0 + v0z*t + 0.5*acc_0*Δt^2 + acc_0/kω*log(cosθ2)*Δt + 
            0.5/(kω)^2 * acc_0 * ( polylogarithm(2,-EΔ*tanθ2) - polylogarithm(2,-tanθ2) )
    else # t > tf3 # Travel to the Screen
        τ_SG = default_y_SG / v0y
        Eτ = exp(-2*kω*τ_SG)
        z = z0 + v0z*t + 0.5*acc_0*( (t-tf2)^2 - (t-tf3)^2) + 
            acc_0/kω*τ_SG * ( log(cosθ2) + v0y/default_y_SG*log(cosθ2+Eτ*sinθ2)*(t-tf3) ) + 
            0.5*acc_0/kω^2*( polylogarithm(2,-Eτ*tanθ2) - polylogarithm(2,-tanθ2) )
        vz = v0z + acc_0*τ_SG + acc_0/kω*log(cosθ2 + Eτ*sinθ2)
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
- `tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG) / v0y`
- `tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v0y`

Definitions:
- `cqd_sign = sign(θn - θe)`
- `ωL = |γₑ| * BvsI(Ix)`                         # Larmor angular frequency
- `acc_0 = μ * GvsI(Ix) / p.M`                   # base acceleration scale
- `kω = cqd_sign * kx * ωL`
- `θe_half = θe/2`
- `tanθ2 = tan(θe_half)^2`, `cosθ2 = cos(θe_half)^2`, `sinθ2 = sin(θe_half)^2`
- `log_cos2 = log(cosθ2)`
- `polylog_0 = polylogarithm(2, -tanθ2)`
- `τ_SG = default_y_SG / v0y`

Piecewise result:
- Pre-SG (`t ≤ tf2`):
  `z(t) = z0 + v0z * t`
- In-SG (`tf2 < t ≤ tf3`), with `Δt = t - tf2` and `exp_term = exp(-2*kω*Δt)`:
  `z(t) = z0 + v0z*t + 0.5*acc_0*Δt^2 + (acc_0/kω)*log_cos2*Δt
          + 0.5*acc_0/kω^2 * ( polylogarithm(2, -exp_term*tanθ2) - polylog_0 )`
- Post-SG (`t > tf3`), with `Δt2 = t - tf2`, `Δt3 = t - tf3`,
  `exp_SG = exp(-2*kω*τ_SG)`, `polylog_SG = polylogarithm(2, -exp_SG*tanθ2)`,
  `log_term = log(cosθ2 + exp_SG*sinθ2)`:
  `z(t) = z0 + v0z*t + 0.5*acc_0*(Δt2^2 - Δt3^2)
          + (acc_0/kω)*τ_SG * (log_cos2 + (Δt3/τ_SG)*log_term)
          + 0.5*acc_0/kω^2 * (polylog_SG - polylog_0)`

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

Numerical notes:
- Terms with `1/kω` and `1/kω^2` cancel analytically as `kω → 0` but may lose precision;
  consider a small-`kω` expansion when `|kω*τ_SG|` is very small.
"""
@inline function CQD_EqOfMotion_z(t,Ix::Float64,μ::Float64,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"
    @assert t >= 0 "time t must be ≥ 0"

    v0y = v0[2]
    v0z = v0[3]
    z0  = r0[3]

    @assert !iszero(v0y) "v0y must be nonzero (beam must advance toward the screen)."
    
    
    tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG ) / v0y
    tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG ) / v0y

    cqd_sign = sign(θn-θe) 
    ωL       = abs( γₑ * BvsI(Ix) )
    acc_z    = μ*GvsI(Ix)/p.M
    kω       = cqd_sign*kx*ωL

    # Precompute angles
    θe_half = θe / 2
    tanθ2 = tan(θe_half)^2
    cosθ2 = cos(θe_half)^2
    sinθ2 = sin(θe_half)^2
    log_cos2 = log(cosθ2)
    polylog_0 = polylogarithm(2, -tanθ2)

    if t <= tf2
        return z0 + v0z*t
    elseif t <= tf3   # Crossing the SG apparatus
        Δt = t - tf2
        exp_term = exp(-2 * kω * Δt)
        polylog_t = polylogarithm(2, -exp_term * tanθ2)

        return z0 + v0z*t + 0.5 * acc_z * Δt^2 + acc_z / kω * log_cos2 * Δt + 0.5 * acc_z / kω^2 * ( polylog_t - polylog_0 )
    
    else # t > tf3 # Travel to the Screen
        Δt2 = t - tf2
        Δt3 = t - tf3
        τ_SG = default_y_SG / v0y
        exp_SG = exp(-2 * kω * τ_SG)
        polylog_SG = polylogarithm(2, -exp_SG * tanθ2)
        log_term = log(cosθ2 + exp_SG * sinθ2)

        return z0 + v0z*t + 0.5*acc_z*( Δt2^2 - Δt3^2 ) + acc_z / kω * τ_SG * (log_cos2 + log_term * Δt3 / τ_SG) + 0.5 * acc_z / kω^2 * (polylog_SG - polylog_0)
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
- `L1 = default_y_FurnaceToSlit`
- `L2 = default_y_SlitToSG`
- `Lsg = default_y_SG`
- `Ld = default_y_SGToScreen`
- `Ltot = L1 + L2 + Lsg + Ld`.

# Physics definitions
- `cqd_sign = sign(θn − θe)`
- `acc_z = μ * GvsI(Ix) / p.M`                # base acceleration scale (m/s²)
- `ωL = abs(γₑ * BvsI(Ix))`                   # Larmor angular frequency (rad/s)
- `kω = cqd_sign * kx * ωL`
- `θe_half = θe/2`, `cos2 = cos(θe_half)^2`, `sin2 = sin(θe_half)^2`, `tan2 = tan(θe_half)^2`
- `exp_term = exp(-2 * kω * Lsg / v0y)`.

# Closed-form result
With initial state `r0 = (x0, y0, z0)` (m) and `v0 = (v0x, v0y, v0z)` (m/s), the screen
coordinates are
- `x = x0 + Ltot * v0x / v0y`
- `y = y0 + Ltot`
- `z = z0 + (Ltot * v0z)/v0y
        + (1/2) * acc_z / v0y^2 * [ (Lsg + Ld)^2 − Ld^2 ]
        + (acc_z / kω) * (Lsg / v0y) * [ log(cos2) + (Ld/Lsg) * log(cos2 + exp_term * sin2) ]
        + (1/2) * acc_z / kω^2 * [ polylogarithm(2, -exp_term * tan2) − polylogarithm(2, -tan2) ].


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
- Uses global geometry constants `default_y_*`.

# Throws
- Assertion errors if `length(r0) ≠ 3`, `length(v0) ≠ 3`, or `v0y == 0`.

# Notes (numerics)
- The formula contains `1/kω` and `1/kω^2` terms; the overall `z` remains finite as `kω → 0`,
but direct evaluation can lose precision. If you expect very small `|kω * Lsg / v0y|`,
consider using a small-`kω` series expansion for the log/polylog combination.
- `exp_term` may under/overflow when `|kω * Lsg / v0y|` is large; a log-sum-exp rewrite can
improve stability near extreme angles (`cos2` or `sin2` ≈ 0).
"""
function CQD_Screen_position(Ix,μ::Float64,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"
    x0, y0, z0 = r0
    v0x, v0y, v0z = v0

    L1   = default_y_FurnaceToSlit 
    L2   = default_y_SlitToSG
    Lsg  = default_y_SG
    Ld   = default_y_SGToScreen
    Ltot = L1 + L2 + Lsg + Ld

    # Physics parameters
    cqd_sign = sign(θn-θe) 
    acc_z = μ * GvsI(Ix) / p.M
    ωL = abs(γₑ * BvsI(Ix))
    kω = cqd_sign * kx * ωL

    # Common trig values
    θe_half = θe / 2
    cos2 = cos(θe_half)^2
    sin2 = sin(θe_half)^2
    tan2 = tan(θe_half)^2
    exp_term = exp(-2 * kω * Lsg / v0y)

    x = x0 + Ltot * v0x / v0y
    y = y0 + Ltot
    z = z0 + Ltot * v0z / v0y + 0.5*acc_z/v0y^2*((Lsg+Ld)^2-Ld^2) + 
        acc_z/kω*Lsg/v0y*( log(cos2) + Ld/Lsg * log( cos2 + exp_term*sin2 ) ) + 
        0.5*acc_z/kω^2 * ( polylogarithm(2, -exp_term*tan2) - polylogarithm(2, -tan2) )
    return SVector{3,Float64}(x,y,z)
end


"""
    CQD_Screen_velocity(Ix, μ, v0, θe, θn, kx, p) -> SVector{3,Float64}

Final **velocity at the screen** under a Continuous Quantum Dynamics (CQD) model.
Motion is ballistic in `x` and `y`; only the `z` component is affected inside the
Stern–Gerlach (SG) region of length `default_y_SG`.

# Model / Definitions
Let
- `Lsg = default_y_SG` (m), `τ_SG = Lsg / v0y` (s),
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
- SG gradient aligned with `+z` and treated constant across `default_y_SG`.

# Throws
- Assertion error if `length(v0) ≠ 3` or `v0y == 0`.

# Notes
- Uses `default_y_SG` for the SG length; adjust the function if you need it configurable.
- Handles the `kω → 0` case internally for numerical stability.
"""
function CQD_Screen_velocity(Ix,μ::Float64,v0::AbstractVector{<:Real},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)

    @assert length(v0) == 3 "v0 must have length 3"
    v0x, v0y, v0z = v0
    @assert !iszero(v0y) "v0y must be nonzero (beam must advance toward the screen)."

    Lsg = default_y_SG

    # Physics parameters
    cqd_sign = sign(θn-θe) 
    acc_z = μ * GvsI(Ix) / p.M
    ωL = abs(γₑ * BvsI(Ix))
    kω = cqd_sign * kx * ωL

    # Common trig values
    θe_half = θe / 2
    cos2 = cos(θe_half)^2
    sin2 = sin(θe_half)^2
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

# Quantum Mechanics : Classical Trajectories

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

# Arguments
- `t::Real`: Time (s) since the initial state `(r0, v0)` is defined; must satisfy `t ≥ 0`.
- `Ix::Real`: Coil current (A) for the field-gradient model `GvsI(Ix)`.
- `f`, `mf`: Hyperfine quantum numbers selecting the effective magnetic moment.
- `r0::AbstractVector{<:Real}`: Initial position `(x0, y0, z0)` in meters; length 3.
- `v0::AbstractVector{<:Real}`: Initial velocity `(v0x, v0y, v0z)` in m/s; length 3 and `v0y ≠ 0`.
- `p::AtomParams`: Atomic parameters (must provide mass `M` and anything required by `μF_effective`).

# Geometry (constants used)
Uses `default_y_FurnaceToSlit`, `default_y_SlitToSG`, `default_y_SG`, `default_y_SGToScreen`
(measured along +y, in meters).

# Returns
A tuple `(r, v)` with `r::SVector{3,Float64}` and `v::SVector{3,Float64}` giving position and
velocity at time `t`.

# Assumptions
- Non-relativistic kinematics; gravity, collisions, and fringing fields neglected.
- Only the `z` component accelerates (constant gradient within the SG region).
- `x` and `y` remain ballistic for all segments.

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

        # Segment times (in seconds)
    _tf1 =  default_y_FurnaceToSlit / v0y                               # slit entrance
    tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG) / v0y                 # SG entrance
    tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v0y          # SG exit
    _tF  = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG + default_y_SGToScreen) / v0y  # screen

    μ =  μF_effective(Ix,f,mf,p)
    acc_z    = μ * GvsI(Ix) / p.M


    # Ballistic x,y always
    x = x0 + v0x * t
    y = y0 + v0y * t
    vx = v0x
    vy = v0y

    if t <= tf2     # Furnace to Slit and Slit to SG apparatus
        # Pre-SG: fully ballistic
        z = z0 + v0z*t
        vz = v0z
    elseif t <= tf3   # Crossing the SG apparatus
        # Inside SG: uniform az
        Δt = t-tf2
        z = z0 + v0z * t + 0.5 * acc_z * Δt^2
        vz = v0z + acc_z * Δt
    else # t > tf3 # Travel to the Screen
        # Post-SG: ballistic with boosted vz
        Δt_SG = default_y_SG / v0y
        z = z0 + v0z * t + acc_z * Δt_SG * (t - 0.5*(tf2+tf3))
        vz = v0z + acc_z * Δt_SG 
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
Uses `default_y_FurnaceToSlit`, `default_y_SlitToSG`, `default_y_SG`, `default_y_SGToScreen`.

# Returns
- `z::Float64`: z-position at time `t`.  
  (If you want to preserve higher precision, remove the final `float(...)` conversion.)

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

# Example
```julia
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
    
    tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG ) / v0y
    tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG ) / v0y

    # z-acceleration inside SG
    μ =  μF_effective(Ix,f,mf,p)
    acc_z    = μ * GvsI(Ix) / p.M

    if t <= tf2
        return z0 + v0z * t
    elseif t <= tf3   # Crossing the SG apparatus
        Δt = t - tf2
        return  z0 + v0z * t + 0.5 * acc_z * Δt^2
    else # t > tf3 # Travel to the Screen
        τ_SG = default_y_SG / v0y
        return z0 + v0z * t + acc_z * τ_SG * (t - 0.5*(tf2+tf3))
    end
end

"""
    QM_Screen_position(Ix, f, mf, r0, v0, p; 
                       y_FurnaceToSlit=default_y_FurnaceToSlit,
                       y_SlitToSG=default_y_SlitToSG,
                       y_SG=default_y_SG,
                       y_SGToScreen=default_y_SGToScreen) -> r

Final 3D position of an atom at the **screen plane** after traversing a Stern–Gerlach (SG) region.
Model: ballistic motion in `x` and `y`; constant acceleration in `z` **only inside** the SG of length `y_SG`.

Let `Lsg = y_SG`, `Ld = y_SGToScreen`, `Ltot = y_FurnaceToSlit + y_SlitToSG + Lsg + Ld`,
`a_z = μF_effective(Ix, f, mf, p) * GvsI(Ix) / p.M`, and `v0y ≠ 0`.

Closed-form screen position:
- `x = x0 + (Ltot / v0y) * v0x`
- `y = y0 + Ltot`
- `z = z0 + (Ltot / v0y) * v0z + 0.5 * a_z / v0y^2 * [ (Lsg + Ld)^2 - Ld^2 ]`

This `z` formula is equivalent to stitching the three segments (pre-SG, in-SG, post-SG) and
keeps continuity of both position and `v_z` at the SG exit.

# Arguments
- `Ix::Real`: Coil current (A).
- `f::Real`, `mf::Real`: Hyperfine quantum numbers (half-integers allowed; not restricted to `Integer`).
- `r0::AbstractVector{<:Real}`: Initial position `(x0,y0,z0)` in meters; length 3.
- `v0::AbstractVector{<:Real}`: Initial velocity `(v0x,v0y,v0z)` in m/s; length 3, `v0y ≠ 0`.
- `p::AtomParams`: Atomic parameters (must provide mass `M` and whatever `μF_effective` needs).

# Keywords (geometry, meters, along +y)
- `y_FurnaceToSlit`, `y_SlitToSG`, `y_SG`, `y_SGToScreen`.

# Returns
- `r::SVector{3,Float64}`: Position at the screen.  
  (Change the element type if you want to preserve higher precision/AD types.)

# Assumptions
- Non-relativistic; gravity/collisions/fringing fields neglected.
- SG gradient aligned with `+z` and treated constant over `y_SG`.
"""
# function QM_Screen_position(Ix,f,mf,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real}, p::AtomParams)
#     @assert length(r0) == 3 "r0 must have length 3"
#     @assert length(v0) == 3 "v0 must have length 3"

#     @inbounds begin
#         x0  = r0[1];  y0  = r0[2];  z0  = r0[3]
#         v0x = v0[1];  v0y = v0[2];  v0z = v0[3]
#     end
#     @assert !iszero(v0y) "v0y must be nonzero (beam must advance toward the screen)."

#     # Geometry
#     Lsg  = default_y_SG::Float64
#     Ld   = default_y_SGToScreen::Float64
#     Ltot = (default_y_FurnaceToSlit  + default_y_SlitToSG + Lsg + Ld)::Float64

#     # Physics parameters
#     μG =  μF_effective(Ix,f,mf,p) * GvsI(Ix)
#     inv_vy  = 1.0 / v0y
#     inv_vy2 = inv_vy * inv_vy
#     acc_z = μG / p.M

#     x = x0 + Ltot * v0x * inv_vy
#     y = y0 + Ltot
#     z = z0 + Ltot * v0z * inv_vy + 0.5 * acc_z * inv_vy2 * ((Lsg+Ld)*(Lsg+Ld) - Ld*Ld)
#     return SVector{3,Float64}(x, y, z)
# end
# ---------- FAST SCALAR CORE (no AbstractVector, no globals in arithmetic) ----------
Base.@propagate_inbounds @inline function QM_Screen_position(
    Ix::Float64, f, mf,
    x0::Float64, y0::Float64, z0::Float64,
    v0x::Float64, v0y::Float64, v0z::Float64,
    p::AtomParams{Float64}
)::SVector{3,Float64}
    @assert v0y != 0.0 "v0y must be nonzero (beam must advance toward the screen)."

    # bind geometry to concrete locals (avoid Any from non-const globals)
    y_in  = (default_y_FurnaceToSlit + default_y_SlitToSG)::Float64
    Lsg   = default_y_SG::Float64
    Ld    = default_y_SGToScreen::Float64
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
    QM_Screen_velocity(Ix, f, mf, v0, p) -> SVector{3,Float64}

Final **velocity at the screen** for an atom that traverses a Stern–Gerlach (SG) region.
The model assumes ballistic motion in `x` and `y`, and uniform acceleration along `z`
only while inside the SG of length `default_y_SG`.

Let
`a_z = μF_effective(Ix, f, mf, p) * GvsI(Ix) / p.M`  and  `τ_SG = default_y_SG / v0y`.
Then
- `v_x = v0x`
- `v_y = v0y`
- `v_z = v0z + a_z * τ_SG`

# Arguments
- `Ix`: Coil current (A) used by the gradient model `GvsI`.
- `f`, `mf`: Hyperfine quantum numbers selecting the effective magnetic moment (not restricted to `Integer`; half-integers allowed).
- `v0::AbstractVector{<:Real}`: Initial velocity `(v0x, v0y, v0z)` in m/s; **must** have length 3 and `v0y ≠ 0`.
- `p::AtomParams`: Atomic parameters; must provide mass `M` and whatever `μF_effective` requires.

# Returns
- `v::SVector{3,Float64}`: Velocity at the screen.

# Uses/Assumes
- `default_y_SG` (meters) as the SG length along `+y`.
- Non-relativistic kinematics; gravity, collisions, and fringe-field variation neglected.
- Only the `z` component accelerates inside the SG.

# Throws
- Assertion error if `length(v0) ≠ 3` or `v0y == 0`.

# Notes
- The return type is `Float64` by construction; if you need to preserve higher precision
  or AD types, adjust the return container accordingly.
"""
# function QM_Screen_velocity(Ix,f,mf,v0::AbstractVector{<:Real}, p::AtomParams)
#     @assert length(v0) == 3 "v0 must have length 3"
#     v0x, v0y, v0z = v0
#     @assert !iszero(v0y) "v0y must be nonzero (beam must advance toward the screen)."

#     # Physics parameters
#     μ =  μF_effective(Ix,f,mf,p)
#     acc_z = μ * GvsI(Ix) / p.M

#     vx = v0x
#     vy = v0y
#     vz = v0z + acc_z * default_y_SG / v0y
#     return SVector{3,Float64}(vx, vy, vz)
# end
# ---------- FAST SCALAR CORE ----------
Base.@propagate_inbounds @inline function QM_Screen_velocity(
    Ix::Float64, f, mf,
    v0x::Float64, v0y::Float64, v0z::Float64,
    p::AtomParams{Float64}
)::SVector{3,Float64}
    @assert v0y != 0.0 "v0y must be nonzero (beam must advance toward the screen)."

    # bind geometry to concrete locals (avoid Any from globals)
    Lsg = default_y_SG::Float64

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

# Fast, cached formulas for screen outputs
@inline function screen_x_z_vz(
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