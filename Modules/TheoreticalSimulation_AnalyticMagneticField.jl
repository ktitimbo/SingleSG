# ==============================================================================
# Analytic two-wire magnetic field model, calibration, and CQD trajectory
# propagation for the SG apparatus
#
# This file implements four layers:
#
#   1. Finite-parallel-wire field model (B_total / grad_B / grad_normB)
#      Closed-form Biot–Savart field (and its gradient) for two finite,
#      anti-parallel straight wires of half-length DEFAULT_ℓ, separated by
#      2·DEFAULT_𝒶 along x — the actual two-wire SG magnet geometry.
#
#   2. Infinite-wire (ℓ → ∞) approximations (approx_B_total / approx_grad_B /
#      approx_grad_normB / approx_dnormBdz)
#      Simpler closed forms valid in the limit of infinitely long wires —
#      used for comparison/calibration against the finite-wire model above.
#
#   3. Calibration (calibrate_Ieff_for_Bz / calibrate_gradient /
#      SGCalibration / build_calibration)
#      Bridges the idealized two-wire model to the real, measured field
#      calibration (BvsI/GvsI from TheoreticalSimulation_MagneticField.jl):
#      finds an effective two-wire current I_eff(I) that reproduces the
#      measured field, then a multiplicative gradient correction S(I) on top.
#
#   4. CQD trajectory propagation (propagate_to_SG_entrance / free_flight /
#      make_eom / propagate_SG / full_trajectory / run_ensemble)
#      Ballistic free-flight legs plus an ODE-integrated leg through the SG
#      field region (using DifferentialEquations.jl), implementing a
#      dissipative-relaxation CQD-style equation of motion. `full_trajectory`
#      runs the full pipeline for a single particle (useful for inspecting
#      one trajectory in detail, via its returned `sol_magnet`); `run_ensemble`
#      is the bulk driver for many particles at once, built on
#      DifferentialEquations.jl's `EnsembleProblem`/`EnsembleThreads()` rather
#      than calling `full_trajectory` in a loop — this avoids retaining a
#      per-particle ODE solution object at N~10^7 scale, and resolves
#      calibration once for the whole ensemble rather than once per particle.
# ==============================================================================


# ──────────────────────────────────────────────────────────────────────────────
# 1. Finite parallel-wire field model
# ──────────────────────────────────────────────────────────────────────────────
# Finite parallel wires

# ── Helpers ────────────────────────────────────────────────────────────────────
# G, F and their derivatives implement the closed-form Biot–Savart expression
# for a single finite straight wire segment of half-length DEFAULT_ℓ (running
# along y, centered at y=0), evaluated at axial position `y` and perpendicular
# distance `ρ` from the wire. These are the shared building blocks `B_total`/
# `grad_B` combine (with opposite sign/offset) to get the two-wire field.

"""
    G(y, ρ) -> Float64

Direction-cosine difference for a finite wire of half-length `DEFAULT_ℓ`:
`(y+ℓ)/√((y+ℓ)²+ρ²) − (y−ℓ)/√((y−ℓ)²+ρ²)`. This is the standard
Biot–Savart angular factor for a straight finite wire segment (the
difference of the cosines of the angles subtended by the two wire endpoints,
as seen from the field point) — multiplying by `μ₀I/(4πρ)` gives the field
magnitude of a finite wire.
"""
G(y, ρ) = (y+DEFAULT_ℓ)/sqrt((y+DEFAULT_ℓ)^2+ρ^2) - (y-DEFAULT_ℓ)/sqrt((y-DEFAULT_ℓ)^2+ρ^2)

"""
    F(y, ρ) -> Float64

`G(y, ρ) / ρ²` — the `1/ρ²` factor folded in, since `B_total`/`grad_B` use
`F` (not `G`) directly when building `Bx`/`Bz` (the extra `1/ρ` from the
Biot–Savart law's standard `1/ρ` prefactor combines with `G`'s own implicit
geometry to give the `1/ρ²` scaling used throughout this section).
"""
F(y, ρ) = G(y, ρ) / ρ^2

"""
    Ap(y, ρ), Am(y, ρ) -> Float64

Intermediate building blocks for the derivatives of `G`/`F`:
`Ap = 1/((y+ℓ)²+ρ²)^(3/2)`, `Am = 1/((y−ℓ)²+ρ²)^(3/2)` (the "+ℓ-endpoint"
and "−ℓ-endpoint" terms respectively). Not physically meaningful on their
own — only used as shared sub-expressions inside `dFdy`/`dFdρ` below.
"""
Ap(y, ρ) = 1 / ((y + DEFAULT_ℓ)^2 + ρ^2)^(3/2)
Am(y, ρ) = 1 / ((y - DEFAULT_ℓ)^2 + ρ^2)^(3/2)

"""
    dFdy(y, ρ) -> Float64

`∂F/∂y = Ap(y,ρ) − Am(y,ρ)`. Derived by differentiating `G`'s two terms
w.r.t. `y` (each gives `ρ²·Ap` / `ρ²·Am` respectively) and dividing by the
same `ρ²` that defines `F = G/ρ²`, so the `ρ²` factors cancel exactly.
"""
dFdy(y, ρ) = Ap(y, ρ) - Am(y, ρ)

"""
    dFdρ(y, ρ) -> Float64

`∂F/∂ρ`, via the product/quotient rule on `F = G(y,ρ)/ρ²`:
`−[(y+ℓ)·Ap − (y−ℓ)·Am]/ρ − 2·G(y,ρ)/ρ³`. The first term comes from
differentiating `G` w.r.t. `ρ` (then dividing by `ρ²`); the second term is
the `∂(1/ρ²)/∂ρ` contribution from the explicit `1/ρ²` in `F`'s definition.
"""
dFdρ(y, ρ) = -((y+DEFAULT_ℓ)*Ap(y, ρ) - (y-DEFAULT_ℓ)*Am(y, ρ))/ρ - 2*G(y, ρ)/ρ^3


"""
    B_total(x, y, z; z0=1.3*DEFAULT_𝒶, Iw=0.2) -> (Bx, By, Bz)

Magnetic field of two finite, anti-parallel straight wires (the SG two-wire
magnet), each of half-length `DEFAULT_ℓ`, located at `(x=+DEFAULT_𝒶, z=z0)`
and `(x=-DEFAULT_𝒶, z=z0)`, both running along `y`, carrying current `Iw`
in opposite directions.

`ρ1`/`ρ2` are the perpendicular distances from `(x,z)` to wire 1/wire 2
respectively. `By` is always returned as exactly `0.0` (not computed) — this
model only resolves the in-plane `(x,z)` field components.

# Arguments
- `x, y, z::Real`: field point (m). `y` is the position along the wires.
- `z0::Real = 1.3*DEFAULT_𝒶`: wire height (m).
- `Iw::Real = 0.2`: wire current (A).

# Returns
`(Bx, By, Bz)::NTuple{3,Float64}`, with `By ≡ 0.0`.

# Throws
- `DomainError` if the field point lies exactly on either wire (`ρ1 == 0`
  or `ρ2 == 0`), where the field is singular/undefined.
"""
function B_total(x,y,z; z0=1.3*DEFAULT_𝒶 ,Iw=0.2)
    ρ1, ρ2 = hypot(x-DEFAULT_𝒶 , z-z0), hypot(x+DEFAULT_𝒶 , z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end
    F1, F2 = F(y, ρ1), F(y, ρ2)
    C = -μ₀*Iw/(4π)
    Bx = C*(z-z0)*(F2-F1)
    Bz = C*((x-DEFAULT_𝒶 )*F1 - (x+DEFAULT_𝒶 )*F2)
    return (Bx,0.0,Bz)
end

"""
    grad_B(x, y, z; z0=1.3*DEFAULT_𝒶, Iw=0.2) -> Matrix{Float64}  (3×3)

Full spatial Jacobian of the two-wire field, `J[i,j] = ∂Bᵢ/∂xⱼ`, with rows
ordered `(Bx, By, Bz)` and columns ordered `(∂/∂x, ∂/∂y, ∂/∂z)`. The middle
row is always `[0, 0, 0]` since `By ≡ 0` everywhere (see `B_total`).

Recomputes `ρ1`, `ρ2`, `F1`, `F2` internally (independently of any prior
`B_total` call at the same point) via the chain rule through `dρdx`/`dρdz`
and `dFdρ`/`dFdy`.

# Arguments
Same as `B_total`.

# Returns
A `3×3 Matrix{Float64}`, freshly allocated every call. Since this function is
used inside the ODE right-hand side (`make_eom`'s `eom!`, evaluated at every
solver step for every particle — see `make_eom`'s Performance note), this
allocation is a real, repeated cost; a fixed-size `StaticArrays.SMatrix{3,3,Float64}`
would avoid it.

# Throws
- `DomainError`, same condition as `B_total`.
"""
function grad_B(x, y, z; z0=1.3*DEFAULT_𝒶, Iw=0.2)
    ρ1 = hypot(x-DEFAULT_𝒶, z-z0)
    ρ2 = hypot(x+DEFAULT_𝒶, z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end
    F1 = F(y, ρ1)
    F2 = F(y, ρ2)
    C = -μ₀*Iw/(4π)
    Δz = z - z0

    # dρ/dx, dρ/dz
    dρ1dx = (x - DEFAULT_𝒶)/ρ1;   dρ2dx = (x + DEFAULT_𝒶)/ρ2
    dρ1dz = Δz/ρ1;        dρ2dz = Δz/ρ2


    # F partials
    dF1dρ = dFdρ(y, ρ1);  dF2dρ = dFdρ(y, ρ2)
    dF1dy = dFdy(y, ρ1);  dF2dy = dFdy(y, ρ2)

    # ∂ᵢBx
    dBxdx = C * Δz * ( dF2dρ*dρ2dx - dF1dρ*dρ1dx )
    dBxdy = C * Δz * ( dF2dy - dF1dy )
    dBxdz = C * (F2-F1 + Δz*(dF2dρ*dρ2dz - dF1dρ*dρ1dz))
    # ∂ᵢBy = 0 ∀ i
    # ∂ᵢBz
    dBzdx = C * ( F1 - F2 + (x-DEFAULT_𝒶)*dF1dρ*dρ1dx - (x+DEFAULT_𝒶)*dF2dρ*dρ2dx )
    dBzdy = C * ( (x-DEFAULT_𝒶)*dF1dy - (x+DEFAULT_𝒶)*dF2dy )
    dBzdz = C * ( (x-DEFAULT_𝒶)*dF1dρ*dρ1dz - (x+DEFAULT_𝒶)*dF2dρ*dρ2dz )

    return @SMatrix [
    dBxdx dBxdy dBxdz;
    0.0 0.0 0.0;
    dBzdx dBzdy dBzdz
    ]
end

"""
    grad_normB(x, y, z; Iw=0.2, z0=1.3*DEFAULT_𝒶) -> (dBdx, dBdy, dBdz)

Gradient of the field **magnitude** `|B|` (not the full vector field
gradient), via the chain rule `∂|B|/∂xⱼ = (B · ∂B/∂xⱼ)/|B|`. Internally
calls `B_total` (to get `B`) and `grad_B` (to get the Jacobian `J`), so the
underlying `ρ1`/`ρ2`/`F1`/`F2` quantities are computed independently inside
each of those two calls — see the precomputed-`B` overload below for a
partial fix (it still doesn't eliminate the deeper redundancy inside
`grad_B` itself).

# Arguments
- `x, y, z::Real`: field point (m).
- `Iw::Real = 0.2`: wire current (A).
- `z0::Real = 1.3*DEFAULT_𝒶`: wire height (m).

# Returns
`(dBdx, dBdy, dBdz)::NTuple{3,Float64}`. Returns `(0.0, 0.0, 0.0)` if
`|B| == 0` exactly (avoids a `0/0` division at that degenerate point).
"""
function grad_normB(x::Real, y::Real, z::Real;
                    Iw::Real = 0.2,
                    z0::Real = 1.3 * DEFAULT_𝒶
)

    Bx, By, Bz = B_total(x, y, z; z0=z0, Iw=Iw)
    Bmag = sqrt(Bx^2 + By^2 + Bz^2)

    if iszero(Bmag)
        return (0.0, 0.0, 0.0)
    end

    # J[i,j] = ∂Bᵢ/∂xⱼ   →   shape (3,3), columns = [∂/∂x, ∂/∂y, ∂/∂z]
    J = grad_B(x, y, z; z0=z0, Iw=Iw)

    # ∂|B|/∂xⱼ = (B · ∂B/∂xⱼ) / |B|  =  (Bx*J[1,j] + By*J[2,j] + Bz*J[3,j]) / |B|
    dBdx = (Bx * J[1,1] + By * J[2,1] + Bz * J[3,1]) / Bmag
    dBdy = (Bx * J[1,2] + By * J[2,2] + Bz * J[3,2]) / Bmag
    dBdz = (Bx * J[1,3] + By * J[2,3] + Bz * J[3,3]) / Bmag

    return (dBdx, dBdy, dBdz)
end

# ── grad_normB accepting precomputed B ───────────────────────────────────────
# avoids a redundant B_total call inside the ODE — B is already known
"""
    grad_normB(x, y, z, Bx, By, Bz; Iw=0.2, z0=1.3*DEFAULT_𝒶) -> (dBdx, dBdy, dBdz)

Same as `grad_normB(x,y,z; ...)` above, but takes `Bx,By,Bz` directly instead
of recomputing them via an internal `B_total` call — used inside `make_eom`'s
`eom!`, where `B_total` has already been evaluated once for the same point a
few lines earlier (see `eom!`'s body). Still calls `grad_B` internally (which
independently recomputes `ρ1`/`ρ2`/`F1`/`F2` from scratch) — this overload
only avoids the *second* `B_total` call; the redundant `ρ`/`F` computation
hidden inside `grad_B` itself remains either way.

# Arguments
- `x, y, z::Real`: field point (m), matching the point `Bx,By,Bz` were
  evaluated at.
- `Bx, By, Bz::Real`: precomputed field components at `(x,y,z)`.
- `Iw::Real = 0.2`, `z0::Real = 1.3*DEFAULT_𝒶`: same role as above.

# Returns
`(dBdx, dBdy, dBdz)::NTuple{3,Float64}`. Returns `(0.0, 0.0, 0.0)` if
`|B| == 0` exactly.
"""
function grad_normB(x::Real, y::Real, z::Real,
                    Bx::Real, By::Real, Bz::Real;
                    Iw::Real = 0.2,
                    z0::Real = 1.3 * DEFAULT_𝒶
)
    Bmag = sqrt(Bx^2 + By^2 + Bz^2)
    iszero(Bmag) && return (0.0, 0.0, 0.0)

    J    = grad_B(x, y, z; Iw = Iw, z0 = z0)

    dBdx = (Bx * J[1,1] + By * J[2,1] + Bz * J[3,1]) / Bmag
    dBdy = (Bx * J[1,2] + By * J[2,2] + Bz * J[3,2]) / Bmag
    dBdz = (Bx * J[1,3] + By * J[2,3] + Bz * J[3,3]) / Bmag

    return (dBdx, dBdy, dBdz)
end


# ──────────────────────────────────────────────────────────────────────────────
# 2. Infinite-wire (ℓ → ∞) approximations
# ──────────────────────────────────────────────────────────────────────────────
# In the limit ℓ → ∞, a finite wire's field reduces to the classic infinite
# straight-wire result B ∝ 1/ρ (instead of the finite-wire G(y,ρ)/ρ factor
# above), which is what `inv_sq_ρ = 1/ρ²` captures here. These functions are
# simpler/cheaper closed forms, presumably used to sanity-check or bound the
# finite-wire model, or as a fast approximation far from the wire ends.

"""
    approx_B_total(x, y, z; z0=1.3*DEFAULT_𝒶, Iw=0.2) -> (Bx, By, Bz)

Infinite-wire (`ℓ → ∞`) limit of `B_total`. Same two-wire geometry, but using
the simpler `1/ρ²` infinite-straight-wire field law instead of the finite-wire
`F(y,ρ)` factor — no `y`-dependence at all (consistent with an infinitely
long wire, whose field doesn't vary along its own axis).

# Arguments / Returns / Throws
Same shape as `B_total` (the `z0`/`Iw` defaults match too), but `y` is
accepted for signature compatibility only and does not affect the result.
"""
function approx_B_total(x,y,z; z0=1.3*DEFAULT_𝒶,Iw=0.2)
    ρ1, ρ2 = hypot(x-DEFAULT_𝒶, z-z0), hypot(x+DEFAULT_𝒶, z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end
    inv_sq_ρ1 = 1/ρ1^2
    inv_sq_ρ2 = 1/ρ2^2
    C = μ₀*Iw/(2π)
    Bx = C*(z-z0)*(inv_sq_ρ2 - inv_sq_ρ1)
    Bz = C*((x-DEFAULT_𝒶)*inv_sq_ρ1 - (x+DEFAULT_𝒶)*inv_sq_ρ2)
    return (Bx,0.0,Bz)
end

"""
    approx_grad_B(x, y, z; z0=1.3*DEFAULT_𝒶, Iw=0.2) -> Matrix{Float64}  (3×3)

Infinite-wire (`ℓ → ∞`) limit of `grad_B`. Since `approx_B_total` has no
`y`-dependence, every `∂/∂y` entry in the returned Jacobian is exactly `0.0`
(not just the middle `By` row, as in the finite-wire `grad_B`) — the second
*column* is also identically zero here.

# Arguments / Returns / Throws
Same shape as `grad_B`.
"""
function approx_grad_B(x, y, z; z0=1.3*DEFAULT_𝒶, Iw=0.2)
    ρ1 = hypot(x-DEFAULT_𝒶, z-z0)
    ρ2 = hypot(x+DEFAULT_𝒶, z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end

    inv_sq_ρ1 = 1/ρ1^2
    inv_sq_ρ2 = 1/ρ2^2
    C = μ₀*Iw/(2π)
    Δz = z - z0


    # ∂ᵢBx
    dBxdx = -2 * C * Δz * ( (x+DEFAULT_𝒶)*inv_sq_ρ2^2 - (x-DEFAULT_𝒶)*inv_sq_ρ1^2 )
    dBxdz = C * (inv_sq_ρ2-inv_sq_ρ1) + 2 * C * Δz^2 * (inv_sq_ρ1^2-inv_sq_ρ2^2 )
    # ∂ᵢBy = 0 ∀ i
    # ∂ᵢBz
    dBzdx = C * (inv_sq_ρ1-inv_sq_ρ2) - 2 * C * ((x-DEFAULT_𝒶)^2*inv_sq_ρ1^2-(x+DEFAULT_𝒶)^2*inv_sq_ρ2^2 )
    dBzdz = -2* C * Δz * ( (x-DEFAULT_𝒶)*inv_sq_ρ1^2 - (x+DEFAULT_𝒶)*inv_sq_ρ2^2 )

    return @SMatrix [
        dBxdx 0.0 dBxdz;
        0.0 0.0 0.0;
        dBzdx 0.0 dBzdz
    ]
end

"""
    approx_grad_normB(x, y, z; Iw=0.2, z0=1.3*DEFAULT_𝒶) -> (dBdx, dBdy, dBdz)

Infinite-wire (`ℓ → ∞`) limit of `grad_normB` — same chain-rule construction
(`∂|B|/∂xⱼ = (B·∂B/∂xⱼ)/|B|`), built from `approx_B_total`/`approx_grad_B`
instead of the finite-wire versions. `dBdy` will always come out `0.0` here
since `approx_grad_B`'s second column is identically zero.

# Arguments / Returns
Same shape as `grad_normB(x,y,z; ...)`.
"""
function approx_grad_normB(x::Real, y::Real, z::Real;
                    Iw::Real = 0.2,
                    z0::Real = 1.3 * DEFAULT_𝒶
)

    Bx, By, Bz = approx_B_total(x, y, z; z0=z0, Iw=Iw)
    Bmag = sqrt(Bx^2 + By^2 + Bz^2)

    if iszero(Bmag)
        return (0.0, 0.0, 0.0)
    end

    # J[i,j] = ∂Bᵢ/∂xⱼ   →   shape (3,3), columns = [∂/∂x, ∂/∂y, ∂/∂z]
    J = approx_grad_B(x, y, z; z0=z0, Iw=Iw)

    # ∂|B|/∂xⱼ = (B · ∂B/∂xⱼ) / |B|  =  (Bx*J[1,j] + By*J[2,j] + Bz*J[3,j]) / |B|
    dBdx = (Bx * J[1,1] + By * J[2,1] + Bz * J[3,1]) / Bmag
    dBdy = (Bx * J[1,2] + By * J[2,2] + Bz * J[3,2]) / Bmag
    dBdz = (Bx * J[1,3] + By * J[2,3] + Bz * J[3,3]) / Bmag

    return (dBdx, dBdy, dBdz)
end

"""
    approx_dnormBdz(x, z; Iw=0.2, z0=1.3*DEFAULT_𝒶) -> Float64

Closed-form **single-component** shortcut: `∂|B|/∂z` only, in the infinite-wire
limit, without going through `approx_B_total`/`approx_grad_B`/the chain-rule
machinery at all — a direct algebraic simplification for just this one
component. Cheaper than calling `approx_grad_normB(x,y,z;...)[3]` when only
`dBdz` is actually needed — worth checking whether call sites elsewhere in
the codebase that only need `dBdz` are actually using this dedicated function
rather than the full gradient.

# Arguments
- `x, z::Real`: field point (m) — note: no `y` argument, consistent with the
  infinite-wire model having no `y`-dependence.
- `Iw::Real = 0.2`, `z0::Real = 1.3*DEFAULT_𝒶`: same role as elsewhere in this section.

# Returns
`Float64`, the `z`-gradient of `|B|` at `(x, z)`.
"""
function approx_dnormBdz(x,z; Iw=0.2, z0=1.3*DEFAULT_𝒶)
    ρ1 = hypot(x-DEFAULT_𝒶, z-z0)
    ρ2 = hypot(x+DEFAULT_𝒶, z-z0)

    Δz = z - z0
    C = μ₀*Iw/(2π)

    return -4 * DEFAULT_𝒶 * C * Δz / (ρ1^3 * ρ2^3) * ( x^2 + DEFAULT_𝒶^2 + Δz^2)
end


# ──────────────────────────────────────────────────────────────────────────────
# 3. Calibration: bridging the analytic two-wire model to measured data
# ──────────────────────────────────────────────────────────────────────────────
# These routines are cold-path (run once, or a handful of times, per
# experimental session — not inside any per-particle or per-ODE-step loop),
# so the performance notes elsewhere in this file mostly don't apply here.

"""
    calibrate_Ieff_for_Bz(I_list; plot_check=true) -> Function

Build a current-correction map `I_eff_B(I)`: given a *nominal* coil current
`I`, returns the *effective* two-wire current `Iw` that makes the idealized
two-wire model (`B_total`) reproduce the **measured** field `BvsI(I)` at the
origin.

# Method
1. `B_measured = BvsI.(I_list)` — the real, measured field at each calibration current.
2. `B_model = [B_total(0,0,0; Iw=x)[3] for x in I_list]` — the idealized
   model's `Bz` at the origin, evaluated *as if* `Iw` were the nominal
   current `I` directly (i.e. probing the model's own `Iw`-dependence).
3. Builds an Akima-spline interpolant mapping `I_list → B_model` (current → idealized field).
4. `I_eff_B(I) = itp_B(BvsI(I))` — composes that interpolant with the
   *measured* field, so feeding in a nominal current `I` returns whichever
   model-`Iw` would have produced the *measured* `Bz` at that current.

# Arguments
- `I_list`: calibration currents (A) — should span the range of currents
  this map will be queried at.
- `plot_check::Bool = true`: if `true`, displays a comparison plot of the
  measured field, the raw (uncorrected) model field, and the model field
  evaluated at the corrected `I_eff_B(I)` — a visual check that the
  correction map actually closes the gap between model and measurement.

# Returns
A closure `I_eff_B(I) -> Float64`, the current-correction map described above.
"""
function calibrate_Ieff_for_Bz(I_list; plot_check=true)

    B_measured = BvsI.(I_list);
    B_model    = [B_total(0, 0, 0; Iw=x)[3]    for x in I_list];

    itp_B = DataInterpolations.AkimaInterpolation(I_list, B_model; extrapolation = ExtrapolationType.Linear);
    I_eff_B = (I) -> itp_B(BvsI(I));


    if plot_check
        B_corrected = [B_total(0, 0, 0; Iw=I_eff_B(x))[3]    for x in I_list];

        plt_B = plot(I_list, B_measured,   
            label="BvsI (target)",  
            marker=(:circle,2,:white), 
            seriestype=:scatter,
            title="Field calibration");
        plot!(plt_B, I_list, B_corrected,  label=L"B_total($I_{eff}$)", lw=2, ls=:dash);
        plot!(plt_B, I_list, B_model,      label=L"B_total($I_{w}$) raw", lw=2, ls=:dot);
        plot!(plt_B, xlabel="Current (A)", ylabel="Magnetic field (T)")



        display(plt_B)
    end

    return I_eff_B
end

"""
    calibrate_gradient(I_list, I_eff_B; span=0.12, degree=3, plot_check=true,
                       epsG=1e-300, anchor_zero=true) -> Function

Build a multiplicative gradient-correction map `S(I)`, fit via LOESS
(local polynomial regression) in log-space, such that
`S(I) * grad_normB(0,0,0; Iw=I_eff_B(I))[3] ≈ GvsI(I)` — i.e. even after
correcting the *current* via `I_eff_B`, the idealized model's raw gradient
may still not exactly match the measured gradient-vs-current curve `GvsI`;
this fits a further scale-factor correction on top.

# Method
1. `G_target = GvsI.(I)` (measured), `G_raw = Gz_raw.(I)` (idealized model,
   evaluated at the `I_eff_B`-corrected current).
2. Builds a boolean `mask` excluding `I=0`, non-finite values, near-zero
   values (`|G| ≤ epsG`), and any points where `G_target`/`G_raw` have
   opposite sign (a ratio/log-ratio wouldn't be meaningful there).
3. Fits `log(|G_target/G_raw|)` vs. `I` (on the masked points) via LOESS.
4. If `anchor_zero`, prepends the point `(I=0, log(S)=0)` to the fit data —
   i.e. forces `S(0) = 1` exactly, since both field and gradient vanish
   together at zero current regardless of any model imperfection.
5. Returns `scale_factor(I) = exp(predict(model, clamp(I, Ifit range)))` —
   note the input is clamped to the fitted current range before prediction,
   so this **extrapolates as a constant** (the boundary value) outside the
   calibration range, not linearly or via the LOESS model's own trend.

# Arguments
- `I_list`: calibration currents (A); sorted internally regardless of input order.
- `I_eff_B`: the current-correction map from `calibrate_Ieff_for_Bz`.
- `span::Real = 0.12`: LOESS smoothing span (fraction of points used per local fit).
- `degree::Int = 3`: LOESS local polynomial degree.
- `plot_check::Bool = true`: if `true`, displays two comparison plots (raw vs.
  corrected model gradient against target; the fitted scale factor itself).
- `epsG::Float64 = 1e-300`: near-zero exclusion threshold for `mask` (note:
  far smaller than any physically meaningful gradient value — effectively
  only excludes exact zeros / underflow, not small-but-real gradients).
- `anchor_zero::Bool = true`: whether to force `S(0) = 1` (see step 4 above).

# Returns
A closure `scale_factor(I) -> Float64`, the gradient-correction map described above.
"""
function calibrate_gradient(
    I_list, I_eff_B;
    span::Real        = 0.12,
    degree::Int       = 3,
    plot_check::Bool  = true,
    epsG::Float64     = 1e-300,
    anchor_zero::Bool = true,
)
    I        = sort(collect(float.(I_list)))
    G_target = GvsI.(I)

    Gz_raw(Ival) = grad_normB(0, 0, 0; Iw = I_eff_B(Ival))[3]
    G_raw        = Gz_raw.(I)

    # exclude I=0 and any non-finite or same-sign violations from log-space fit
    mask = (I .> 0.0)            .&    # I=0 excluded from log fit
           isfinite.(G_target)   .&
           isfinite.(G_raw)      .&
           (abs.(G_target) .> epsG) .&
           (abs.(G_raw)    .> epsG) .&
           (sign.(G_target) .== sign.(G_raw))

    Ifit = I[mask]
    logy = log.(abs.(G_target[mask] ./ G_raw[mask]))

    # anchor at I=0: S(0)=1 → log(S)=0, since both field and gradient vanish together
    if anchor_zero
        Ifit = vcat(0.0, Ifit)
        logy = vcat(0.0, logy)    # S(0) = 1 exactly
    end

    xmin, xmax = extrema(Ifit)
    model      = loess(Ifit, logy; span=span, degree=degree)

    function scale_factor(Ival)
        isfinite(Ival) || return NaN
        return exp(predict(model, [clamp(float(Ival), xmin, xmax)])[1])
    end

    if plot_check
        Iplot       = range(first(I), last(I), length=500)
        G_raw_plot  = Gz_raw.(Iplot)
        G_corr_plot = scale_factor.(Iplot) .* G_raw_plot

        plt1 = plot(I, G_target,
                    seriestype=:scatter, marker=(:circle,3,:white), label="GvsI target",
                    xlabel="Current (A)", ylabel="Gradient (T/m)", title="Gradient calibration")
        plot!(plt1, Iplot, G_raw_plot,  lw=2, ls=:dot,  label=L"raw: $\nabla|B|(I_{\mathrm{eff},B})$")
        plot!(plt1, Iplot, G_corr_plot, lw=2, ls=:dash, label=L"corrected: $S(I)\nabla|B|(I_{\mathrm{eff},B})$")

        plt2 = plot(Ifit, exp.(logy),
                    seriestype=:scatter, marker=(:circle,3,:white), label="target / raw",
                    xlabel="Current (A)", ylabel=L"S(I)", title="Fitted gradient scale")
        plot!(plt2, Iplot, scale_factor.(Iplot), lw=2, label="LOESS fit, span=$span")

        display(plot(plt1, plt2; layout=(1,2), size=(1000,400), left_margin=5mm, bottom_margin=4mm))
    end

    return scale_factor
end

"""
    SGCalibration

Bundles the two calibration maps built by `calibrate_Ieff_for_Bz`/
`calibrate_gradient` into a single object passed around the propagation
functions below (`propagate_SG`, `full_trajectory`, `run_ensemble`;
`make_eom` doesn't take it directly, but receives the already-resolved
`Iw_eff`/`S` values derived from it).

# Fields
- `I_eff_B::Function`    — nominal current → effective two-wire current.
- `grad_scale::Function` — nominal current → multiplicative gradient correction.
"""
struct SGCalibration
    I_eff_B    :: Function
    grad_scale :: Function
end

"""
    build_calibration(I_list; span=0.12, degree=4, plot_check=true) -> SGCalibration

Convenience constructor: runs `calibrate_Ieff_for_Bz` then `calibrate_gradient`
(passing the first's result into the second) and bundles both into an
`SGCalibration`.

Note: `degree` default here is `4`, different from `calibrate_gradient`'s own
default of `3` — this function's default takes precedence whenever
`build_calibration` is used without an explicit `degree`.

# Arguments
- `I_list`: calibration currents (A), forwarded to both calibration routines.
- `span::Real = 0.12`, `degree::Int = 4`: forwarded to `calibrate_gradient`.
- `plot_check::Bool = true`: forwarded to both calibration routines.

# Returns
`SGCalibration(I_eff_B, grad_scale)`.
"""
function build_calibration(I_list; span=0.12, degree=4, plot_check=true)
    I_eff_B    = calibrate_Ieff_for_Bz(I_list; plot_check=plot_check)
    grad_scale = calibrate_gradient(I_list, I_eff_B; span=span, degree=degree,
                                    plot_check=plot_check)
    return SGCalibration(I_eff_B, grad_scale)
end

#______________________________________________________________________________________
# ATOM PROPAGATION CQD

# ──────────────────────────────────────────────────────────────────────────────
# 4. CQD trajectory propagation
# ──────────────────────────────────────────────────────────────────────────────

"""
    propagate_to_SG_entrance(data; y_SG_entrance=DEFAULT_SG_magnet_entrance) -> Matrix{Float64}

Ballistically (free-flight) propagate a batch of particles from their
current state (presumably right after the furnace/slit) up to the SG magnet
entrance plane `y = y_SG_entrance`, assuming straight-line motion (no forces
before the field region).

# Arguments
- `data::AbstractMatrix`: `N × ≥7` matrix; columns `1:3` = `(x,y,z)`,
  columns `4:6` = `(vx,vy,vz)`, column `7` = `θ0` (CQD initial angle,
  carried through unchanged — not modified by free flight).
- `y_SG_entrance::Real = DEFAULT_SG_magnet_entrance`: target `y` plane (m).

# Returns
`Matrix{Float64}` of size `N × 7`: `[x y z vx vy vz θ0]` at the SG entrance,
row order preserved. Computed via `Threads.@threads` over particles
(independent per-row work, safe to parallelize — each thread writes only to
its own row index).
"""
function propagate_to_SG_entrance(data;
                                y_SG_entrance = DEFAULT_SG_magnet_entrance)

    N    = size(data, 1)
    # output: [x y z vx vy vz θ0] at SG entrance
    data_SG = Matrix{Float64}(undef, N, 7)

    Threads.@threads for ii in 1:N
        r0 = @view data[ii, 1:3]
        v0 = @view data[ii, 4:6]
        vy = v0[2]

        # combined free flight: oven → slit → SG entrance
        Δt = (y_SG_entrance - r0[2]) / vy

        data_SG[ii, 1] = r0[1] + v0[1] * Δt
        data_SG[ii, 2] = y_SG_entrance
        data_SG[ii, 3] = r0[3] + v0[3] * Δt
        data_SG[ii, 4] = v0[1]
        data_SG[ii, 5] = vy
        data_SG[ii, 6] = v0[3]
        data_SG[ii, 7] = data[ii, 7]    # θ0 carried through unchanged
    end

    return data_SG
end


# ── Free flight ──────────────────────────────────────────────────────────────
"""
    free_flight(r, v, y_target) -> (r_new, v)

Straight-line ballistic propagation of a single particle from position `r`
(with velocity `v`, unchanged by free flight) to the plane `y = y_target`,
using the constant-`vy` drift time `Δt = (y_target - r[2]) / v[2]`.

# Arguments
- `r`, `v`: 3-element position/velocity (any indexable container with `[1],[2],[3]`).
- `y_target::Real`: target `y` plane (m).

# Returns
`(r_new, v)` — the propagated position and the (unmodified) velocity.
"""
@inline function free_flight(r, v, y_target)
    Δt = (y_target - r[2]) / v[2]
    return r .+ v .* Δt, v
end


# ── EOM ──────────────────────────────────────────────────────────────────────
# grad_mask: NTuple{3} of 0/1 to selectively enable x/y/z force components
"""
    make_eom(Iw_eff, S, μ_over_m, k, θ0, t_in, y_SG_center, grad_mask) -> Function

Build the ODE right-hand-side closure `eom!(du, u, p, t)` for one particle's
trajectory through the SG field region, implementing a dissipative-relaxation
CQD-style equation of motion: the angle `θ` between the particle's magnetic
moment and the local field relaxes exponentially toward alignment (`θ → 0`)
at a rate set by `k` and the local Larmor frequency `|γₑ|·|B|`, and the force
on the particle is the adiabatic Stern–Gerlach term (`∝ cosθ`) plus a
non-adiabatic correction term (`∝ k|γₑ|B0τ·sin²θ`) that grows with elapsed
time `τ` since field entry.

`eom!` is called at every internal ODE-solver stage/step — i.e. very many
times per trajectory, and once per trajectory for every particle in an
ensemble — so this is the hottest code path in this file by a wide margin.

# Performance
`tan(θ0 / 2)` is recomputed inside `eom!` on every call, even though `θ0` is
fixed for the lifetime of this closure (it's a `make_eom` argument, not a
per-step quantity) — precomputing it once here, before defining `eom!`,
would remove a transcendental-function call from every single ODE step.
`sin(θ)^2` could also be obtained as `1 - cos(θ)^2` (`cosθ` is already
computed immediately above it), avoiding a second transcendental call.

# Arguments
- `Iw_eff::Float64`: calibration-corrected effective two-wire current (A).
- `S::Float64`: calibration gradient-scale factor.
- `μ_over_m::Float64`: magnetic-moment-to-mass ratio.
- `k::Float64`: CQD relaxation-rate constant.
- `θ0::Float64`: initial angle between moment and field, at field entry.
- `t_in::Float64`: time (in the ODE's independent variable) at field entry —
  used to compute elapsed time `τ = t - t_in` inside `eom!`.
- `y_SG_center::Float64`: `y`-coordinate of the SG magnet's center, used to
  shift the field-model's local `y`-coordinate (`y_loc = y - y_SG_center`)
  so the two-wire field model (centered at its own local origin) lines up
  with the lab-frame trajectory coordinate.
- `grad_mask::NTuple{3,Float64}`: per-axis (x,y,z) 0/1 mask selectively
  enabling/disabling each force component — e.g. `(0,0,1)` to keep only the
  z-gradient force (the typical SG configuration).

# Returns
`eom!(du, u, p, t)`, a 6-state (`x,y,z,vx,vy,vz`) in-place ODE function
suitable for `DifferentialEquations.ODEProblem`.
"""
function make_eom(Iw_eff::Float64, S::Float64,
                  μ_over_m::Float64, k::Float64, θ0::Float64,
                  t_in::Float64, y_SG_center::Float64,
                  grad_mask::NTuple{3,Float64})

    half_tan_θ0 = tan(θ0 / 2)
    kγ = k * abs(γₑ)

    function eom!(du, u, p, t)
        x, y, z, vx, vy, vz = u
        y_loc = y - y_SG_center

        # single B_total call — result reused in grad_normB
        Bx, By, Bz = B_total(x, y_loc, z; Iw=Iw_eff)
        B0         = sqrt(Bx^2 + By^2 + Bz^2)

        # gradient with precomputed B, scaled by S
        dBdx, dBdy, dBdz = S .* grad_normB(x, y_loc, z, Bx, By, Bz; Iw=Iw_eff)

        # dissipative relaxation: θ(t,|B|)
        τ     = t - t_in
        ξ     = half_tan_θ0 * exp(-kγ * B0 * τ)
        θ     = 2 * atan(ξ)
        cosθ  = cos(θ)
        sin²θ = sin(θ)^2

        prefactor = μ_over_m * (cosθ + kγ * B0 * τ * sin²θ)

        du[1] = vx
        du[2] = vy
        du[3] = vz
        du[4] = grad_mask[1] * prefactor * dBdx
        du[5] = grad_mask[2] * prefactor * dBdy
        du[6] = grad_mask[3] * prefactor * dBdz
    end

    return eom!
end

# ── ODE leg: slit → aperture ─────────────────────────────────────────────────
"""
    propagate_SG(Iw, r_in, v_in, cal::SGCalibration;
                 μ_over_m, k, θ0,
                 y_field_start=DEFAULT_y_FurnaceToSlit,
                 y_field_end=DEFAULT_y_FurnaceToSlit+DEFAULT_y_SlitToSG+DEFAULT_y_SG+DEFAULT_y_SGToAperture,
                 y_SG_center=DEFAULT_center_of_SG_magnet,
                 grad_mask=(0.0,0.0,1.0)) -> (sol, r_end, v_end)

Integrate one particle's trajectory through the SG field region (from
`y_field_start` to `y_field_end`) using `make_eom`'s CQD equation of motion,
stopping exactly at `y = y_field_end` via a `ContinuousCallback`.

Calibration (`Iw_eff`, `S`) is resolved from `cal` **once per call**, before
building `eom!` — not re-resolved at every ODE step. Since this function is
called once per particle (e.g. from `full_trajectory`), that calibration
resolution still happens once per particle, not once per ensemble — fine for
single-particle use, but worth knowing if you ever call this repeatedly over
a fixed `Iw`: `run_ensemble` avoids the per-particle cost entirely by
resolving calibration once for the whole ensemble and not routing through
`propagate_SG` at all.

# Arguments
- `Iw::Real`: nominal coil current (A) for this particle's trajectory.
- `r_in`, `v_in`: 3-element initial position/velocity at `y_field_start`.
- `cal::SGCalibration`: calibration maps.
- `μ_over_m`, `k`, `θ0`: forwarded to `make_eom` (converted to `Float64`).
- `y_field_start`, `y_field_end`, `y_SG_center`, `grad_mask`: geometry/physics
  options, same roles as in `make_eom`/the module's geometry constants.

# Method
Solves with `Vern7()` at tight tolerances (`abstol=1e-14`, `reltol=1e-12`),
saving the state at 5 landmark times (`saveat`): field entry, SG entrance,
SG center, SG exit, and field-region exit — intended for later inspection of
the trajectory through the magnet (e.g. via the `sol_magnet` field returned
by `full_trajectory`), not just the final state.

# Returns
`(sol, r_end, v_end)`: the full `ODESolution` object, and the final
position/velocity (`u_end[1:3]`, `u_end[4:6]`).
"""
function propagate_SG(Iw, r_in, v_in, cal::SGCalibration;
                      μ_over_m,
                      k,
                      θ0,
                      y_field_start = DEFAULT_y_FurnaceToSlit,
                      y_field_end   = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToAperture,
                      y_SG_center   = DEFAULT_center_of_SG_magnet,
                      grad_mask     = (0.0, 0.0, 1.0))

    # resolve calibration once per particle — never inside the ODE
    Iw_eff = cal.I_eff_B(Float64(Iw))
    S      = cal.grad_scale(Float64(Iw))

    # time stamps for saveat (constant vy assumption — valid for thermal beams)
    vy          = v_in[2]
    t_in        = y_field_start                                              / vy
    t_out       = y_field_end                                                / vy
    t_SG_in     = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)            / vy
    t_SG_center = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + 0.5 * DEFAULT_y_SG) / vy
    t_SG_out    = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG)        / vy

    eom! = make_eom(Iw_eff, S,
                    Float64(μ_over_m), Float64(k), Float64(θ0),
                    t_in, Float64(y_SG_center),
                    NTuple{3,Float64}(grad_mask))

    u0    = Float64[r_in[1], r_in[2], r_in[3], v_in[1], v_in[2], v_in[3]]
    tspan = (t_in, 1.02 * t_out)

    cb   = ContinuousCallback((u, t, i) -> u[2] - y_field_end, terminate!)
    prob = ODEProblem(eom!, u0, tspan)
    sol  = solve(prob, Vern7(),
                 callback = cb,
                 abstol   = 1e-14,
                 reltol   = 1e-12,
                 saveat   = [t_in, t_SG_in, t_SG_center, t_SG_out, t_out])

    u_end = sol.u[end]
    return sol, u_end[1:3], u_end[4:6]
end

# ── Full pipeline: oven → slit → field region → aperture → screen ────────────
"""
    full_trajectory(Iw, r0, v0, cal::SGCalibration;
                    μ_over_m, k, θ0,
                    y_slit=DEFAULT_y_FurnaceToSlit,
                    y_aperture=..., y_screen=..., y_SG_center=...,
                    R_aperture=DEFAULT_c_aperture, R_screen=DEFAULT_R_tube,
                    grad_mask=(0.0,0.0,1.0)) -> NamedTuple

Full single-particle pipeline: free flight (oven → slit) → ODE-integrated
field-region leg (slit → aperture, via `propagate_SG`) → aperture
pass/fail check → free flight (aperture → screen) → screen pass/fail check.

# Arguments
Same roles as `propagate_SG`, plus:
- `y_slit`, `y_aperture`, `y_screen`: landmark `y` planes for the three legs.
- `R_aperture::Real = DEFAULT_c_aperture`, `R_screen::Real = DEFAULT_R_tube`:
  transverse radii for the two pass/fail checks (`x² + z² ≤ R²`).

# Returns
A `NamedTuple` with fields:
- `r_screen`, `v_screen`: final position/velocity at the screen.
- `pass::Bool`: `true` only if the particle clears **both** the aperture and
  screen radius checks.
- `sol_magnet`: the full `ODESolution` from the field-region leg (from
  `propagate_SG`) — useful for inspecting/plotting a single particle's
  trajectory through the magnet. Note that `run_ensemble` does not call
  `full_trajectory` at all (it builds its own `ODEProblem`s directly for
  bulk efficiency), so this field is only relevant when calling
  `full_trajectory` directly for a single particle.

Two fields (`r_slit`, `r_aperture`/`v_aperture`) are computed internally but
commented out of the returned `NamedTuple` (left in the source as `#`-commented
lines) — available to re-enable if needed for debugging.
"""
function full_trajectory(Iw, r0, v0, cal::SGCalibration;
                         μ_over_m,
                         k,
                         θ0,
                         y_slit      = DEFAULT_y_FurnaceToSlit,
                         y_aperture  = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToAperture,
                         y_screen    = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToScreen,
                         y_SG_center = DEFAULT_center_of_SG_magnet,
                         R_aperture  = DEFAULT_c_aperture,
                         R_screen    = DEFAULT_R_tube,
                         grad_mask   = (0.0, 0.0, 1.0))

    # 1. free flight: oven → slit
    r_slit, v_slit = free_flight(r0, v0, y_slit)

    # 2. ODE: slit → aperture (fringe fields included)
    sol, r_ap, v_ap = propagate_SG(Iw, r_slit, v_slit, cal;
                                    μ_over_m      = μ_over_m,
                                    k             = k,
                                    θ0            = θ0,
                                    y_field_start = y_slit,
                                    y_field_end   = y_aperture,
                                    y_SG_center   = y_SG_center,
                                    grad_mask     = grad_mask)

    # 3. aperture check: transverse radius at y_aperture
    pass_aper = r_ap[1]^2 + r_ap[3]^2 ≤ R_aperture^2

    # 4. free flight: aperture → screen
    r_screen, v_screen = free_flight(r_ap, v_ap, y_screen)

    # 5. screen check: transverse radius at detector
    pass_screen = r_screen[1]^2 + r_screen[3]^2 ≤ R_screen^2

    return (;
        # r_slit      = r_slit,
        # r_aperture  = r_ap,      v_aperture  = v_ap,
        r_screen    = r_screen,  v_screen    = v_screen,
        pass        = pass_aper && pass_screen,
        sol_magnet  = sol)
end

# ── Ensemble run ──────────────────────────────────────────────────────────────
"""
    run_ensemble(Iw, data, cal::SGCalibration;
                 μ_over_m, k,
                 y_slit=DEFAULT_y_FurnaceToSlit,
                 y_aperture=..., y_screen=..., y_SG_center=...,
                 R_aperture=DEFAULT_c_aperture, R_screen=DEFAULT_R_tube,
                 grad_mask=(0.0,0.0,1.0)) -> Matrix{Float64}

Bulk ensemble driver: propagate every particle in `data` through the full
oven → slit → SG field region → aperture → screen pipeline, and collect the
final screen state and pass/fail outcome for each.

Built on `DifferentialEquations.jl`'s `EnsembleProblem`/`EnsembleThreads()`
rather than looping over `full_trajectory` directly — this avoids ever
retaining a per-particle `ODESolution` object (critical at `N ~ 10^7` scale;
see `output_func` below) and resolves calibration (`Iw_eff`, `S`) **once for
the whole ensemble**, not once per particle.

# Method
1. Resolves `Iw_eff = cal.I_eff_B(Iw)` and `S = cal.grad_scale(Iw)` once, up
   front, along with several other per-call-constant scalars (`gmask`, `μ_m`,
   `kf`, `yc`, squared radii, `Δy_screen`) — all captured by the closures below.
2. Builds a template `ODEProblem` from particle 1's data (`prob_template`),
   required by `EnsembleProblem` but never actually integrated as-is.
3. `prob_func(prob, context)`: called once per particle (`context.sim_id`),
   rebuilds the particle's equation of motion (via `make_eom`, since `θ0`
   varies per particle) and initial state, then applies both via `remake`.
   The oven→slit free-flight step is computed inline here, not via
   `free_flight` — a duplicate of that function's logic, kept in sync by hand.
4. `affect!`: fires via a `ContinuousCallback` exactly when a trajectory
   crosses `y = y_aperture`; computes the remaining aperture→screen free
   flight analytically, performs both pass/fail radius checks, and writes
   directly into `screen_data[ii, :]` — no per-particle allocation.
5. `output_func(sol, context) = (nothing, false)`: discards each trajectory's
   solution object immediately after `affect!` runs.
6. `solve(...; save_everystep=false, save_end=false)`: no internal ODE steps
   or final state retained beyond what `affect!` already extracted.

# Arguments
- `Iw::Real`: nominal coil current (A), the same for every particle in this call.
- `data::AbstractMatrix`: `N × ≥7` matrix; columns `1:3` = position, `4:6` =
  velocity, column `7` = `θ0` (CQD initial angle) — column `7` is currently
  hardcoded for `θ0` inside `prob_func`, with no equivalent of a `θ0_col` keyword.
- `cal::SGCalibration`: calibration maps.
- `μ_over_m`, `k`: forwarded to `make_eom` for every particle.
- Geometry/physics keywords (`y_slit`, `y_aperture`, `y_screen`, `y_SG_center`,
  `R_aperture`, `R_screen`, `grad_mask`): same roles as in
  `full_trajectory`/`propagate_SG`.

# Returns
`Matrix{Float64}` of size `N × 7`: columns `1:3` = final position, `4:6` =
final velocity, `7` = `pass` (`0.0`/`1.0`), all at the screen. Logs a summary
(`@info`) with particle count, pass percentage, thread count, and elapsed time.
"""
function run_ensemble(Iw, data, cal::SGCalibration;
                      μ_over_m,
                      k,
                      y_slit      = DEFAULT_y_FurnaceToSlit,
                      y_aperture  = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToAperture,
                      y_screen    = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToScreen,
                      y_SG_center = DEFAULT_center_of_SG_magnet,
                      R_aperture  = DEFAULT_c_aperture,
                      R_screen    = DEFAULT_R_tube,
                      grad_mask   = (0.0, 0.0, 1.0))

    N = size(data, 1)

    # ── resolve all scalars once — nothing below should call cal or convert types ──
    Iw_eff    = cal.I_eff_B(Float64(Iw))      # effective current for B_total
    S         = cal.grad_scale(Float64(Iw))    # gradient scale factor
    gmask     = NTuple{3,Float64}(grad_mask)   # force component selector
    μ_m       = Float64(μ_over_m)
    kf        = Float64(k)
    yc        = Float64(y_SG_center)
    R_ap_sq   = R_aperture^2                   # squared radii avoid sqrt in checks
    R_sc_sq   = R_screen^2
    Δy_screen = y_screen - y_aperture          # fixed free-flight after aperture

    # output: [x y z vx vy vz pass] at screen, one row per particle
    screen_data = Matrix{Float64}(undef, N, 7)

    # ── template ODEProblem built from particle 1 ─────────────────────────────
    # EnsembleProblem requires a template; prob_func replaces all fields via
    # remake() so the template values never appear in the actual integration
    prob_template = let r0 = @view(data[1, 1:3]), v0 = @view(data[1, 4:6])
        vy   = v0[2]
        t_in = y_slit / vy
        Δt   = (y_slit - r0[2]) / vy
        u0   = Float64[r0[1] + v0[1]*Δt, y_slit, r0[3] + v0[3]*Δt, v0[1], vy, v0[3]]
        eom! = make_eom(Iw_eff, S, μ_m, kf, Float64(data[1,7]), t_in, yc, gmask)
        ODEProblem(eom!, u0, (t_in, 1.01 * y_aperture/vy))
    end

    # DifferentialEquations.jl API (SciMLBase 2.x):
    # prob_func signature : (prob, context)  — particle index = context.sim_id
    # affect! index access: integrator.p     — set via remake(..., p=ii)
    # output_func        : (sol, context)    — return (nothing, false) to discard sol

    # ── prob_func: called once per particle to specialise the template ────────
    # θ0 varies per particle so eom! must be rebuilt each time;
    # everything else (Iw_eff, S, gmask, ...) is captured from outer scope
    function prob_func(prob, context)
        ii   = context.sim_id
        r0   = @view data[ii, 1:3]
        v0   = @view data[ii, 4:6]
        θ0   = Float64(data[ii, 7])
        vy   = v0[2]
        t_in = y_slit / vy
        Δt   = (y_slit - r0[2]) / vy

        u0   = Float64[r0[1] + v0[1]*Δt, y_slit, r0[3] + v0[3]*Δt, v0[1], vy, v0[3]]
        eom! = make_eom(Iw_eff, S, μ_m, kf, θ0, t_in, yc, gmask)

        remake(prob; f=eom!, u0=u0, tspan=(t_in, 1.01*y_aperture/vy), p=ii)
    end

    # ── affect!: fires at y = y_aperture (ContinuousCallback crossing) ───────
    # computes free-flight to screen, performs both geometry checks,
    # writes directly to screen_data[ii,:] — no allocations, no return value
    function affect!(integrator)
        u  = integrator.u
        ii = integrator.p          # particle index passed via p

        vx_ap, vy_ap, vz_ap = u[4], u[5], u[6]
        Δt   = Δy_screen / vy_ap        # time from aperture to screen

        x_sc = u[1] + vx_ap * Δt
        z_sc = u[3] + vz_ap * Δt

        screen_data[ii, 1] = x_sc
        screen_data[ii, 2] = y_screen
        screen_data[ii, 3] = z_sc
        screen_data[ii, 4] = vx_ap
        screen_data[ii, 5] = vy_ap
        screen_data[ii, 6] = vz_ap
        # pass = 1 only if particle clears both aperture and screen radii
        screen_data[ii, 7] = Float64(
            (u[1]^2 + u[3]^2 ≤ R_ap_sq) && (x_sc^2 + z_sc^2 ≤ R_sc_sq)
        )

        terminate!(integrator)
    end

    # stop integration exactly when particle crosses y_aperture
    cb = ContinuousCallback((u, t, i) -> u[2] - y_aperture, affect!)

    # output_func discards the solution object immediately after each trajectory
    # — critical for N~10^7: without this every sol would accumulate in memory
    output_func(sol, context) = (nothing, false)

    ensemble_prob = EnsembleProblem(prob_template;
                                    prob_func   = prob_func,
                                    output_func = output_func)

    # EnsembleThreads: particles distributed across Julia threads (set via
    # JULIA_NUM_THREADS or --threads auto at startup)
    t_elapsed = @elapsed solve(ensemble_prob, Vern7(), EnsembleThreads();
                               trajectories   = N,
                               callback       = cb,
                               abstol         = 1e-14,
                               reltol         = 1e-12,
                               save_everystep = false,   # no internal steps saved
                               save_end       = false)   # affect! handles the exit state

    n_pass = sum(screen_data[:, 7])
    @info "ENSEMBLE COMPLETED" particles=N passed_pct=round(100*n_pass/N, digits=1) threads=Threads.nthreads() time_s=round(t_elapsed, digits=2)

    return screen_data
end