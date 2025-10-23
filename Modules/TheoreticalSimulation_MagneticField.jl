# PROPERTIES OF THE MAGNETIC POLES
# This code is inside `module TheoreticalSimulation` 

# ---- Gradient ↔ current tables (hardcoded) ----

"""
Calibration currents (A) used to define the gradient–current lookup.
"""
const GRAD_CURRENTS = [0, 0.095, 0.2, 0.302, 0.405, 0.498, 0.6, 0.7, 0.75, 0.8, 0.902, 1.01]

"""
Calibration gradients corresponding to `GRAD_CURRENTS`.
Units are (T/m).
"""
const GRAD_GRADIENT = [0, 25.6, 58.4, 92.9, 132.2, 164.2, 196.3, 226, 240, 253.7, 277.2, 298.6]

"""
Internal: prebuilt linear interpolant mapping current → gradient.
Call via `GvsI(I)` instead of using this object directly.
"""
const _GvsI = Interpolations.LinearInterpolation(GRAD_CURRENTS, GRAD_GRADIENT; extrapolation_bc=Line())

"""
Internal: prebuilt linear interpolant mapping gradient → current.
Call via `IvsG(G)` instead of using this object directly.
"""
const _IvsG = Interpolations.LinearInterpolation(GRAD_GRADIENT, GRAD_CURRENTS; extrapolation_bc=Line())

"""
    GvsI(I::Real) -> Real

Gradient (e.g. dB/dz) as a function of coil current `I` (A),
using the hardcoded calibration table and linear interpolation.
"""
GvsI(x) = _GvsI(x)

"""
    IvsG(G::Real) -> Real

Current (A) that produces the given gradient `G`,
using the hardcoded calibration table and linear interpolation.
"""
IvsG(x) = _IvsG(x)


# ---- Magnetic Field vs current from CSV ----

"""
Absolute path to the B-vs-I CSV used at runtime.
Expected CSV columns (no header row override): `dI, Bz`.
"""
const B_TABLE_PATH = joinpath(@__DIR__, "SG_BvsI.csv")
@info "Importing file from $(B_TABLE_PATH)"

# Strictly-positive floor for B (tesla). Adjust if you want a different minimum.
const B_FLOOR = 5.0e-17

# Small helper to clamp to a strictly-positive lower bound
@inline _posfloor(x::Real) = ifelse(x > B_FLOOR, x, 0.0)

"""
Internal: holds the B(I) interpolant once initialized.
Use `BvsI(I)` to evaluate; do not access directly.
"""
const _BvsI = Ref{Any}(nothing)

"""
    __init__() -> Nothing

Module init hook. If `SG_BvsI.csv` exists next to this file, read it and build
a linear interpolant `B(I)` with extrapolation by line. Expects columns named
`dI` (current, A) and `Bz` (magnetic field, T).
"""
function __init__()
    if isfile(B_TABLE_PATH)
        df = CSV.read(B_TABLE_PATH, DataFrame; header=["dI","Bz"])
        # Enforce positivity in source data (handles any zero/negative entries)
        bz_pos = map(_posfloor, df.Bz)
        _BvsI[] = linear_interpolation(df.dI, bz_pos; extrapolation_bc=Line())
    else
        @warn "B table not found at $B_TABLE_PATH; call set_B_table! first."
    end
end

"""
    BvsI(I::Real) -> Real

Magnetic field `B` (e.g. tesla) as a function of current `I` (ampere),
evaluated using the prebuilt interpolation from the CSV.  
Throws an error if the table was not initialized.
"""
@inline function BvsI(I::Real)
    itp = _BvsI[]
    itp === nothing && error("BvsI not initialized. Load the table first.")
    return _posfloor(itp(float(I)))
end


# MAGNET SHAPE

##############################
# Geometry structs + builders
##############################

struct EdgeGeom{T<:AbstractFloat}
    zc::T        # z_center
    a::T         # arc radius
    a2::T        # a^2
    tanφ::T
end

"""
    EdgeGeom(; a=2.5e-3, z_center_factor=1.3, φ=π/6, T=Float64)

Precompute constants for the top *edge* profile.
"""
function EdgeGeom(; a::Float64=2.5e-3, z_center_factor::Float64=1.3, φ::Float64=π/6, T=Float64)
    aa  = T(a)
    zc  = T(z_center_factor) * aa
    tφ  = T(tan(φ))
    EdgeGeom{T}(zc, aa, aa*aa, tφ)
end

struct TrenchGeom{T<:AbstractFloat}
    rtc::T       # r_trench_center
    rt::T        # r_trench
    rt2::T       # r_trench^2
    tanφ::T
    lw::T
    cosφ::T
    sinφ::T
    lwcos::T     # lw*cosφ
    lwsin::T     # lw*sinφ
    cutL::T      # -rt - lwcos
    cutR::T      #  rt + lwcos
end

"""
    TrenchGeom(; a=2.5e-3, z_center_factor=1.3, r_trench_factor=1.362,
                trench_drop_factor=1.018, lw_factor=1.58, φ=π/6, T=Float64)

Precompute constants for the bottom *trench* profile.
"""
function TrenchGeom(; a::Float64=2.5e-3, z_center_factor::Float64=1.3, r_trench_factor::Float64=1.362,
                     trench_drop_factor::Float64=1.018, lw_factor::Float64=1.58,
                     φ::Float64=π/6, T=Float64)
    aa   = T(a)
    zc   = T(z_center_factor) * aa
    rt   = T(r_trench_factor) * aa
    rt2  = rt*rt
    tφ   = T(tan(φ)); cφ = T(cos(φ)); sφ = T(sin(φ))
    lw   = T(lw_factor) * aa
    lwc  = lw * cφ
    lws  = lw * sφ
    rtc  = zc - T(trench_drop_factor) * aa
    cutL = -rt - lwc
    cutR =  rt + lwc
    TrenchGeom{T}(rtc, rt, rt2, tφ, lw, cφ, sφ, lwc, lws, cutL, cutR)
end


"""
    z_magnet_edge(x::Real) -> Float64

Top **edge** profile `z(x)` of the SG magnet in metres.

Geometry (hard-coded inside the function)
- `a = 2.5e-3` m (arc radius)
- `z_center = 1.3a` (arc center height)
- `φ = π/6` (flank angle)

Piecewise definition
- `x < −a`        : straight flank with slope `−tan(φ)` ending at `x = −a`
- `|x| ≤ a`       : circular arc `z = z_center − √(a² − x²)`
- `x > a`         : straight flank with slope `+tan(φ)` starting at `x = +a`

Returns the vertical coordinate `z` (m) of the edge at horizontal position `x` (m).
The profile is continuous at `x = ±a` (slope changes there).
"""
# function z_magnet_edge(x)
#     a = 2.5e-3;
#     z_center = 1.3*a 
#     r_edge = a
#     φ = π/6
#     if x <= -r_edge
#         z = z_center - tan(φ)*(x+r_edge)
#     elseif x <= r_edge
#         z = z_center - sqrt(r_edge^2 - x^2)
#     else # x > r_edge
#         z = z_center + tan(φ)*(x-r_edge)
#     end

#     return z
# end
"""
    z_magnet_edge(x, g::EdgeGeom) -> Float64

Top edge profile using precomputed geometry `g`.
"""
@inline function z_magnet_edge(x::Float64, g::EdgeGeom{T}) where {T<:AbstractFloat}
    xx = T(x)
    xx <= -g.a ? (g.zc - g.tanφ * (xx + g.a)) :
    xx >=  g.a ? (g.zc + g.tanφ * (xx - g.a)) :
                 (g.zc - sqrt(max(zero(T), g.a2 - xx*xx)))
end


"""
    z_magnet_trench(x::Real) -> Float64

Bottom **trench** profile `z(x)` of the SG magnet in metres.

Geometry (hard-coded inside the function)
- `a = 2.5e-3` m (base length scale)
- Trench circular section:
  - radius `r_trench = 1.362a`
  - center height `r_trench_center = 1.3a − 1.018a`
- Ledge/ramp geometry:
  - ramp angle `φ = π/6`
  - ramp/ledge length `lw = 1.58a` along the flank

Piecewise definition (left → right; symmetric about `x = 0`)
- `x ≤ −(r_trench + lw cosφ)`          : flat ledge at `z = r_trench_center + lw sinφ`
- `−(r_trench + lw cosφ) < x ≤ −r_trench` : linear ramp down with slope `−tanφ`
- `|x| ≤ r_trench`                     : circular trench `z = r_trench_center − √(r_trench² − x²)`
- `r_trench < x ≤ r_trench + lw cosφ`  : linear ramp up with slope `+tanφ`
- `x > r_trench + lw cosφ`             : flat ledge at `z = r_trench_center + lw sinφ`

Returns the vertical coordinate `z` (m) at horizontal position `x` (m).
"""
# function z_magnet_trench(x)
#     a = 2.5e-3;
#     z_center = 1.3*a 
#     r_edge = 1.0*a
#     r_trench = 1.362*a
#     r_trench_center = z_center - 1.018*a
#     lw = 1.58*a
#     φ = π/6

#     if x <= -r_trench - lw*cos(φ)
#         z = r_trench_center + lw*sin(φ)
#     elseif x <= -r_trench
#         z = r_trench_center - tan(φ)*(x+r_trench)
#     elseif x <= r_trench
#         z = r_trench_center - sqrt( r_trench^2 - x^2 )
#     elseif x<= r_trench + lw*cos(φ)
#         z = r_trench_center + tan(φ)*(x-r_trench)
#     else # x > r_trench + lw*cos(φ)
#         z = r_trench_center + lw*sin(φ)
#     end

#     return z
# end
"""
    z_magnet_trench(x, g::TrenchGeom) -> Float64

Bottom trench profile using precomputed geometry `g`.
"""
@inline function z_magnet_trench(x::Float64, g::TrenchGeom{T}) where {T<:AbstractFloat}
    xx = T(x)
    if xx <= g.cutL
        return g.rtc + g.lwsin
    elseif xx <= -g.rt
        return g.rtc - g.tanφ * (xx + g.rt)
    elseif xx <=  g.rt
        return g.rtc - sqrt(max(zero(T), g.rt2 - xx*xx))
    elseif xx <= g.cutR
        return g.rtc + g.tanφ * (xx - g.rt)
    else
        return g.rtc + g.lwsin
    end
end

########################################
# Backward-compatible no-arg wrappers
########################################

const _EDGE_DEFAULT   = EdgeGeom()
const _TRENCH_DEFAULT = TrenchGeom()

# Keep your old signatures working (use the default geometry)
@inline z_magnet_edge(x::Float64)::Float64   = z_magnet_edge(x, _EDGE_DEFAULT)
@inline z_magnet_trench(x::Float64)::Float64 = z_magnet_trench(x, _TRENCH_DEFAULT)


"""
    z_magnet_edge_time(t, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}) -> Float64

Edge profile evaluated **along a trajectory** at time `t`.

Computes the instantaneous horizontal position `x(t) = r0[1] + v0[1]*t` (m) and
returns `z_magnet_edge(x(t))` (m), using the same geometry as `z_magnet_edge`.

Arguments
- `t`  : time (s)
- `r0` : initial position vector; only `r0[1]` (x, m) is used
- `v0` : initial velocity vector; only `v0[1]` (vx, m/s) is used
"""
@inline function z_magnet_edge_time(t, r0::AbstractVector{Float64}, v0::AbstractVector{Float64})
    a =2.5e-3;
    z_center = 1.3*a 
    r_edge = a
    φ = π/6

    x = r0[1] + v0[1]*t
    if x <= -r_edge
        z = z_center - tan(φ)*(x+r_edge)
    elseif x <= r_edge
        z = z_center - sqrt(r_edge^2 - x^2)
    else # x > r_edge
        z = z_center + tan(φ)*(x-r_edge)
    end

    return z
end

"""
    z_magnet_trench_time(t, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}) -> Float64

Trench profile evaluated **along a trajectory** at time `t`.

Computes `x(t) = r0[1] + v0[1]*t` (m) and returns `z_magnet_trench(x(t))` (m),
using the same geometry as `z_magnet_trench`.

Arguments
- `t`  : time (s)
- `r0` : initial position vector; only `r0[1]` (x, m) is used
- `v0` : initial velocity vector; only `v0[1]` (vx, m/s) is used
"""
@inline function z_magnet_trench_time(t, r0::AbstractVector{Float64}, v0::AbstractVector{Float64})
    a = 2.5e-3;
    z_center = 1.3*a 
    r_edge = 1.0*a
    r_trench = 1.362*a
    r_trench_center = z_center - 1.018*a
    lw = 1.58*a
    φ = π/6

    x = r0[1] + v0[1]*t
    if x <= -r_trench - lw*cos(φ)
        z = r_trench_center + lw*sin(φ)
    elseif x <= -r_trench
        z = r_trench_center - tan(φ)*(x+r_trench)
    elseif x <= r_trench
        z = r_trench_center - sqrt( r_trench^2 - x^2 )
    elseif x<= r_trench + lw*cos(φ)
        z = r_trench_center + tan(φ)*(x-r_trench)
    else # x > r_trench + lw*cos(φ)
        z = r_trench_center + lw*sin(φ)
    end

    return z
end


