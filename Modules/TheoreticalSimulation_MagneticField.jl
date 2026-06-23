# ==============================================================================
# Magnetic field & magnet-geometry models for the SG apparatus
#
# This file implements three independent pieces, used throughout the rest of
# the pipeline:
#
#   1. Gradient ↔ current calibration (GvsI / IvsG)
#      Hardcoded calibration-table data, interpolated via Akima splines, for
#      converting between coil current and the resulting field gradient.
#
#   2. Magnetic field vs current (BvsI)
#      A second, independent calibration — this one loaded from a CSV file
#      at module-init time rather than hardcoded — mapping coil current to
#      field magnitude.
#
#   3. Magnet shape geometry (EdgeGeom / TrenchGeom / z_magnet_edge / z_magnet_trench)
#      Closed-form piecewise profiles for the physical top-edge and
#      bottom-trench shape of the SG magnet pole pieces, used by the
#      cavity-crash trajectory kernels (TheoreticalSimulation_EquationsOfMotion.jl,
#      not yet reviewed) to test whether a particle's trajectory clears the gap.
#
# NOTE: `B_total` and `grad_normB` (used extensively by the "twowires"
# functions in TheoreticalSimulation_DiscardedParticles.jl) are NOT defined in
# this file — they must live in TheoreticalSimulation_AnalyticMagneticField.jl,
# which we haven't reviewed yet.
# ==============================================================================
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 1. Gradient ↔ current calibration
# ──────────────────────────────────────────────────────────────────────────────
 
"""
Calibration currents (A) used to define the gradient–current lookup.
Must be strictly increasing — `DataInterpolations.AkimaInterpolation` requires
sorted independent-variable (`t`) data.
"""
const GRAD_CURRENTS = [0, 0.095, 0.2, 0.302, 0.405, 0.498, 0.6, 0.7, 0.75, 0.8, 0.902, 1.01]
 
"""
Calibration gradients corresponding to `GRAD_CURRENTS`.
Units are (T/m).
"""
const GRAD_GRADIENT = [0, 25.6, 58.4, 92.9, 132.2, 164.2, 196.3, 226, 240, 253.7, 277.2, 298.6]

"""
Internal: prebuilt Akima-spline interpolant mapping current → gradient.
Call via `GvsI(I)` instead of using this object directly.
 
Built directly as a top-level `const` (no lazy/deferred initialization needed,
unlike `_BvsI` below) since `GRAD_CURRENTS`/`GRAD_GRADIENT` are hardcoded
literals available immediately at module-load time, with no file I/O. This
means `_GvsI`'s type is concretely inferred — calling `_GvsI(x)` is fully
type-stable, with none of the `Ref{Any}` concern that applies to `_BvsI`.
 
NOTE: switched from `Interpolations.LinearInterpolation` (left commented out
below, presumably the previous implementation) to an Akima spline. Unlike
linear interpolation, an Akima spline can produce values that briefly
overshoot/undershoot between calibration points, even when the underlying
table is monotonic — worth keeping in mind if anything downstream (e.g. a
root-find against `GvsI`/`BvsI`) assumes strict monotonicity between table
points, not just at the table points themselves.
"""
# const _GvsI = Interpolations.LinearInterpolation(GRAD_CURRENTS, GRAD_GRADIENT; extrapolation_bc=Line())
const _GvsI = DataInterpolations.AkimaInterpolation(GRAD_GRADIENT, GRAD_CURRENTS; extrapolation = ExtrapolationType.Linear)

"""
Internal: prebuilt Akima-spline interpolant mapping gradient → current.
Call via `IvsG(G)` instead of using this object directly. Same construction
pattern and same monotonicity caveat as `_GvsI` — see its docstring.
"""
# const _IvsG = Interpolations.LinearInterpolation(GRAD_GRADIENT, GRAD_CURRENTS; extrapolation_bc=Line())
const _IvsG = DataInterpolations.AkimaInterpolation(GRAD_CURRENTS, GRAD_GRADIENT; extrapolation = ExtrapolationType.Linear)

"""
    GvsI(I::Real) -> Real
 
Gradient (e.g. dB/dz) as a function of coil current `I` (A),
using the hardcoded calibration table and Akima-spline interpolation.
Extrapolates linearly outside the calibrated current range
(`extrapolation = ExtrapolationType.Linear`).
"""
GvsI(x) = _GvsI(x)

"""
    IvsG(G::Real) -> Real
 
Current (A) that produces the given gradient `G`,
using the hardcoded calibration table and Akima-spline interpolation.
Extrapolates linearly outside the calibrated gradient range.
"""
IvsG(x) = _IvsG(x)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Magnetic field vs current, from CSV
# ──────────────────────────────────────────────────────────────────────────────
 
"""
Absolute path to the B-vs-I CSV used at runtime.
Expected CSV columns (no header row override): `dI, Bz`.
 
NOTE: `header=["dI","Bz"]` is passed to `CSV.read` below (forcing these
column names and treating row 1 of the file as the first *data* row, not a
header row). If `SG_BvsI.csv` actually has its own header line, that line
would be silently read as a data row instead — worth confirming the file's
actual structure matches this assumption (can't verify without the file).
"""
const B_TABLE_PATH = joinpath(@__DIR__, "SG_BvsI.csv")
@info "Importing file from $(B_TABLE_PATH)"
 
# Strictly-positive floor for B (tesla). Adjust if you want a different minimum.
const B_FLOOR = 1.0e-18
 
"""
    _posfloor(x::Real) -> Real
 
Clamp `x` to a strictly-positive lower bound: returns `x` unchanged if
`x > B_FLOOR`, otherwise returns `B_FLOOR`. Used to keep field values away
from exactly zero (and away from negative measurement noise), since
downstream physics divides by `B` in several places (e.g.
`μF_effective_B`'s `normalized_B` calculation).
 
Equivalent to `max(x, B_FLOOR)`.
"""
@inline _posfloor(x::Real) = max(x, B_FLOOR)
 
"""
Internal: holds the B(I) interpolant once initialized, via `__init__` below.
Use `BvsI(I)` to evaluate; do not access directly.
 
Typed as `Ref{Any}`, deliberately: the interpolant's concrete type isn't
known until `__init__` actually reads `SG_BvsI.csv` and constructs it, and
there's nothing else to build that type from ahead of time without coupling
this section to some other interpolant defined elsewhere in the file. This
makes `_BvsI[]` itself type-unstable to read directly — see `BvsI`/
`_bvsi_eval` below for how that instability is contained to a single, cheap
dynamic dispatch rather than spreading through every operation performed on
the interpolant.
"""
const _BvsI = Ref{Any}(nothing)
 
"""
    __init__() -> Nothing
 
Module init hook. If `SG_BvsI.csv` exists next to this file, read it and build
an Akima-spline interpolant `B(I)` with linear extrapolation. Expects columns
named `dI` (current, A) and `Bz` (magnetic field, T).
 
Reading the CSV here (rather than at top-level `const` evaluation time) is
the standard Julia pattern for deferring file I/O until the module is
actually loaded, rather than during precompilation.
 
Every value in the `Bz` column is passed through `_posfloor` before building
the interpolant, so zero/negative/near-zero measured field values are
clamped away from zero (see `_posfloor`'s docstring).
 
Both columns are explicitly converted to `Vector{Float64}` before
construction. This isn't required by anything else in this section — it's
simply to pin down one predictable, known concrete type for `_BvsI[]`,
rather than leaving it to whatever `CSV.read` happens to infer from the file
(e.g. a `dI` column where every entry happens to be a whole number could
otherwise infer as `Int64`).
"""
function __init__()
    if isfile(B_TABLE_PATH)
        df = CSV.read(B_TABLE_PATH, DataFrame; header=["dI","Bz"])
        # Enforce positivity in source data (handles any zero/negative entries),
        # then force Float64 for a single, predictable concrete type (see docstring).
        bz_pos = Float64.(map(_posfloor, df.Bz))
        dI_vec = Float64.(df.dI)
        # _BvsI[] = linear_interpolation(df.dI, bz_pos; extrapolation_bc=Line())
        _BvsI[] = DataInterpolations.AkimaInterpolation(bz_pos, dI_vec; extrapolation = ExtrapolationType.Linear)
    else
        @warn "B table not found at $B_TABLE_PATH."
    end
end
 
"""
    BvsI(I::Real) -> Real
 
Magnetic field `B` (e.g. tesla) as a function of current `I` (ampere),
evaluated using the prebuilt interpolation loaded from the CSV at module init.
The result is passed through `_posfloor` again here (in addition to being
applied to the source data before interpolation), so interpolated/extrapolated
values are also kept away from zero.
 
Throws an error if the table was not initialized (i.e. `SG_BvsI.csv` wasn't
found when the module loaded).
 
# Performance
`_BvsI[]` reads as `Any` (see its docstring), so calling it directly here
would force every subsequent operation — the interpolation call itself, the
`_posfloor` clamp — through dynamic dispatch too. Instead, `BvsI` hands `itp`
off to `_bvsi_eval` immediately: Julia performs exactly **one** dynamic
dispatch, at that call boundary, to resolve `itp`'s concrete type, then
compiles (and reuses, on every subsequent call) a fully specialized version
of `_bvsi_eval` for that type. Inside that specialized version, the
interpolation call and the `_posfloor` clamp both run at full native speed —
no further type-instability cost beyond the one dispatch. This is the
standard "function barrier" pattern for containing an unavoidably `Any`-typed
value to a single, cheap dispatch rather than letting it propagate through
an entire computation.
"""
@inline function BvsI(I::Real)
    itp = _BvsI[]
    itp === nothing && error("BvsI not initialized. Load the table first.")
    return _bvsi_eval(itp, float(I))
end
 
"""
    _bvsi_eval(itp, x::Float64) -> Float64
 
Internal: evaluate the interpolant `itp` at `x` and apply `_posfloor`. Exists
solely to serve as the function barrier described in `BvsI`'s docstring —
not intended to be called directly. Julia compiles one specialized version
of this function per concrete type `itp` is ever called with; since `_BvsI[]`
is assigned exactly once (in `__init__`) and never reassigned to a different
type afterward, only one specialization is ever needed.
"""
@inline _bvsi_eval(itp, x::Float64) = _posfloor(itp(x))
 


# ──────────────────────────────────────────────────────────────────────────────
# 3. Magnet shape geometry
# ──────────────────────────────────────────────────────────────────────────────
# MAGNET SHAPE
##############################
# Geometry structs + builders
##############################
 
"""
    EdgeGeom{T<:AbstractFloat}
 
Precomputed constants for the top **edge** profile (`z_magnet_edge`), so that
trig/derived quantities (`tan(φ)`, `a²`, etc.) are computed once when the
struct is built, not on every call to `z_magnet_edge`.
 
# Fields
- `zc::T`    — arc center height (m)
- `a::T`     — arc radius (m)
- `a2::T`    — `a^2`, precomputed for the circular-arc branch
- `tanφ::T`  — `tan(φ)`, precomputed for the straight-flank branches
"""
struct EdgeGeom{T<:AbstractFloat}
    zc::T        # z_center
    a::T         # arc radius
    a2::T        # a^2
    tanφ::T
end

"""
    EdgeGeom(; a=2.5e-3, z_center_factor=1.3, φ=π/6, T=Float64)
 
Precompute constants for the top *edge* profile.
 
# Keyword Arguments
- `a::Float64=2.5e-3`: arc radius (m).
- `z_center_factor::Float64=1.3`: arc-center height, as a multiple of `a`.
- `φ::Float64=π/6`: flank angle (rad).
- `T=Float64`: element type for the resulting `EdgeGeom{T}`.
 
# Returns
`EdgeGeom{T}` with `zc = z_center_factor*a`, `a`, `a2 = a^2`, `tanφ = tan(φ)`.
"""
function EdgeGeom(; a::Float64=2.5e-3, z_center_factor::Float64=1.3, φ::Float64=π/6, T=Float64)
    aa  = T(a)
    zc  = T(z_center_factor) * aa
    tφ  = T(tan(φ))
    EdgeGeom{T}(zc, aa, aa*aa, tφ)
end

"""
    TrenchGeom{T<:AbstractFloat}
 
Precomputed constants for the bottom **trench** profile (`z_magnet_trench`),
analogous in purpose to `EdgeGeom` — every trig/derived quantity needed by
the piecewise trench formula is computed once here, not per call.
 
# Fields
- `rtc::T`   — trench-circle center height (m)
- `rt::T`    — trench radius (m)
- `rt2::T`   — `rt^2`
- `tanφ::T`, `cosφ::T`, `sinφ::T` — trig of the ramp angle `φ`
- `lw::T`    — ramp/ledge length along the flank (m)
- `lwcos::T`, `lwsin::T` — `lw*cosφ`, `lw*sinφ`
- `cutL::T`, `cutR::T` — left/right x-boundaries between the ramp and the
  flat ledge regions (`∓(rt + lwcos)`)
"""
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
 
# Keyword Arguments
- `a::Float64=2.5e-3`: base length scale (m) — same nominal value as `EdgeGeom`'s `a`.
- `z_center_factor::Float64=1.3`: same role as in `EdgeGeom`.
- `r_trench_factor::Float64=1.362`: trench radius, as a multiple of `a`.
- `trench_drop_factor::Float64=1.018`: trench-center drop below `z_center`, as a multiple of `a`.
- `lw_factor::Float64=1.58`: ramp/ledge length, as a multiple of `a`.
- `φ::Float64=π/6`: ramp angle (rad) — same nominal value as `EdgeGeom`'s `φ`.
- `T=Float64`: element type for the resulting `TrenchGeom{T}`.
 
# Returns
`TrenchGeom{T}` with all derived fields precomputed (see struct docstring).
"""
function TrenchGeom(; a::Float64=2.5e-3, z_center_factor::Float64=1.3, r_trench_factor::Float64=1.362,
                     trench_drop_factor::Float64=1.018, lw_factor::Float64=1.58,
                     φ::Float64=π/6, T=Float64)
    aa   = T(a)
    zc   = T(z_center_factor) * aa
    rt   = T(r_trench_factor) * aa
    rt2  = rt*rt
    tφ   = T(tan(φ))
    cφ   = T(cos(φ))
    sφ   = T(sin(φ))
    lw   = T(lw_factor) * aa
    lwc  = lw * cφ
    lws  = lw * sφ
    rtc  = zc - T(trench_drop_factor) * aa
    cutL = -rt - lwc
    cutR =  rt + lwc
    TrenchGeom{T}(rtc, rt, rt2, tφ, lw, cφ, sφ, lwc, lws, cutL, cutR)
end


"""
    z_magnet_edge(x, g::EdgeGeom) -> Float64
 
Top **edge** profile `z(x)` of the SG magnet (metres), using precomputed
geometry `g` (see `EdgeGeom`).
 
Piecewise definition
- `x < −g.a`  : straight flank with slope `−tan(φ)` ending at `x = −g.a`
- `|x| ≤ g.a` : circular arc `z = g.zc − √(g.a² − x²)`
- `x > g.a`   : straight flank with slope `+tan(φ)` starting at `x = +g.a`
 
where `g.zc` (arc-center height), `g.a` (arc radius), and `g.tanφ`
(precomputed flank slope) all come from `g`, not from literals hard-coded in
this function — see `EdgeGeom`'s constructor for where the actual default
values (`a = 2.5e-3` m, `z_center = 1.3a`, `φ = π/6`) live.
 
Returns the vertical coordinate `z` (m) of the edge at horizontal position
`x` (m). The profile is continuous at `x = ±g.a` (slope changes there).
 
A single-argument convenience method, `z_magnet_edge(x)`, is defined below
(in the "Backward-compatible no-arg wrappers" section) using a default
`EdgeGeom` — same formula, default geometry, no separate docstring of its own.
"""
@inline function z_magnet_edge(x::Float64, g::EdgeGeom{T}) where {T<:AbstractFloat}
    xx = T(x)
    xx <= -g.a ? (g.zc - g.tanφ * (xx + g.a)) :
    xx >=  g.a ? (g.zc + g.tanφ * (xx - g.a)) :
                 (g.zc - sqrt(max(zero(T), g.a2 - xx*xx)))
end


"""
    z_magnet_trench(x, g::TrenchGeom) -> Float64
 
Bottom **trench** profile `z(x)` of the SG magnet (metres), using precomputed
geometry `g` (see `TrenchGeom`).
 
Piecewise definition (left → right; symmetric about `x = 0`)
- `x ≤ g.cutL`         : flat ledge at `z = g.rtc + g.lwsin`
- `g.cutL < x ≤ −g.rt` : linear ramp down with slope `−tan(φ)`
- `|x| ≤ g.rt`         : circular trench `z = g.rtc − √(g.rt² − x²)`
- `g.rt < x ≤ g.cutR`  : linear ramp up with slope `+tan(φ)`
- `x > g.cutR`         : flat ledge at `z = g.rtc + g.lwsin`
 
where `g.rtc` (trench-circle center height), `g.rt` (trench radius),
`g.cutL`/`g.cutR` (ramp/ledge boundaries, `∓(g.rt + g.lwcos)`), and `g.tanφ`
(precomputed ramp slope) all come from `g`, not from literals hard-coded in
this function — see `TrenchGeom`'s constructor for where the actual default
values (`a = 2.5e-3` m, `r_trench = 1.362a`, `lw = 1.58a`, `φ = π/6`, etc.) live.
 
Returns the vertical coordinate `z` (m) at horizontal position `x` (m).
 
A single-argument convenience method, `z_magnet_trench(x)`, is defined below
(in the "Backward-compatible no-arg wrappers" section) using a default
`TrenchGeom` — same formula, default geometry, no separate docstring of its own.
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


########################################
# Backward-compatible no-arg wrappers
########################################
 
# Built once at module load — every call to the no-arg z_magnet_edge/
# z_magnet_trench below reuses these same precomputed geometry objects,
# rather than rebuilding them per call.
const _EDGE_DEFAULT   = EdgeGeom()
const _TRENCH_DEFAULT = TrenchGeom()
 
# Keep your old signatures working (use the default geometry)
@inline z_magnet_edge(x::Float64)::Float64   = z_magnet_edge(x, _EDGE_DEFAULT)
@inline z_magnet_trench(x::Float64)::Float64 = z_magnet_trench(x, _TRENCH_DEFAULT)