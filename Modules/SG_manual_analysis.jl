# Plotting backend and general appearance settings
using Plots; gr()
# Set default plot aesthetics
Plots.default(
    dpi=800, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
# Data I/O and numerical tools
using LinearAlgebra
using HCubature
using Interpolations
using Statistics
# Aesthetics and output formatting
using Colors, ColorSchemes
using Printf, LaTeXStrings, PrettyTables
# Time-stamping/logging
using Dates
using Alert


# Magnetic field and gradient for two finite anti-parallel x-directed wires
# Wire 1: y = -a, z = -z0, current +I xhat
# Wire 2: y =  +a, z = -z0, current -I xhat
# Segment along x ∈ [-ℓ, ℓ]
const μ0 = 4π*1e-7
const ℓ = 3.5e-2
const a = 2.5e-3
const x_slit = 4e-3
const z_slit = 300e-6


"""
    z_magnet_edge(x::Real) -> Float64

Top **edge** profile `z(x)` of the SG magnet in metres.

Geometry (hard-coded inside the function)
- `a = 2.5e-3` m (arc radius)
- `z_center = 1.3a` (arc center height)
- `φ = π/6` (flank angle)

Piecewise definition
- `x ≤ −a`        : straight flank with slope `−tan(φ)` ending at `x = −a`
- `|x| ≤ a`       : circular arc `z = z_center − √(a² − x²)`
- `x > a`         : straight flank with slope `+tan(φ)` starting at `x = +a`

Returns the vertical coordinate `z` (m) of the edge at horizontal position `x` (m).
The profile is continuous at `x = ±a` (slope changes there).
"""
function z_magnet_edge(x)
    a = 2.5e-3;
    z_center = 1.3*a 
    r_edge = a
    φ = π/6
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
function z_magnet_trench(x)
    a = 2.5e-3;
    z_center = 1.3*a 
    r_edge = 1.0*a
    r_trench = 1.362*a
    r_trench_center = z_center - 1.018*a
    lw = 1.58*a
    φ = π/6

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

# Helpers
G(y, ρ) = (y+ℓ)/sqrt((y+ℓ)^2+ρ^2) - (y-ℓ)/sqrt((y-ℓ)^2+ρ^2)
F(y, ρ) = G(y, ρ) / ρ^2

Ap(y, ρ) = 1 / ((y + ℓ)^2 + ρ^2)^(3/2)
Am(y, ρ) = 1 / ((y - ℓ)^2 + ρ^2)^(3/2)
dFdy(y, ρ) = Ap(y, ρ) - Am(y, ρ)
dFdρ(y, ρ) = -((y+ℓ)*Ap(y, ρ) - (y-ℓ)*Am(y, ρ))/ρ - 2*G(y, ρ)/ρ^3


function B_total(x,y,z; z0=1.3*a,Iw=0.2)
    ρ1, ρ2 = hypot(x-a, z-z0), hypot(x+a, z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end
    F1, F2 = F(y, ρ1), F(y, ρ2)
    C = μ0*Iw/(4π)
    Bx = C*(z-z0)*(F2-F1)
    Bz = C*((x-a)*F1 - (x+a)*F2)
    return (Bx,0.0,Bz)
end

function Bfield_component_grid(
    xs,
    zs;
    y::Real = 0.0,
    Iw::Real = 0.2,
    z0::Real = 1.3a,
    comp::Symbol = :x,
    scale_factor::Real = 1.0,
    axis_order::Symbol = :xz, # rows.columns
)
    value_at(x, z) = begin
        Bx, By, Bz = B_total(x, y, z; z0=z0, Iw=Iw)

        value = if comp === :x
            Bx
        elseif comp === :z
            Bz
        elseif comp === :norm
            sqrt(Bx^2 + By^2 + Bz^2)
        else
            error("comp must be :x, :z, or :norm")
        end

        value / scale_factor
    end

    if axis_order === :xz
        # size = (length(xs), length(zs))
        return [value_at(x, z) for x in xs, z in zs]
    elseif axis_order === :zx
        # size = (length(zs), length(xs))
        return [value_at(x, z) for z in zs, x in xs]
    else
        error("axis_order must be :xz or :zx")
    end
end

function grad_B(x, y, z; z0=1.3*a, Iw=0.2)
    ρ1 = hypot(x-a, z-z0)
    ρ2 = hypot(x+a, z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end
    F1 = F(y, ρ1)
    F2 = F(y, ρ2)
    C = μ0*Iw/(4π)
    Δz = z - z0

    # dρ/dx, dρ/dz
    dρ1dx = (x - a)/ρ1;   dρ2dx = (x + a)/ρ2
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
    dBzdx = C * ( F1 - F2 + (x-a)*dF1dρ*dρ1dx - (x+a)*dF2dρ*dρ2dx )
    dBzdy = C * ( (x-a)*dF1dy - (x+a)*dF2dy )
    dBzdz = C * ( (x-a)*dF1dρ*dρ1dz - (x+a)*dF2dρ*dρ2dz )

    return [
        dBxdx dBxdy dBxdz;
        0 0 0;
        dBzdx dBzdy dBzdz
    ]
end

function grad_dnormBdz(
    x::Real,
    z::Real;
    y::Real = 0.0,
    Iw::Real = 0.2,
    z0::Real = 1.3a,
    scale_factor::Real = 1.0,
)
    Bx, By, Bz = B_total(x, y, z; z0=z0, Iw=Iw)
    J = grad_B(x, y, z; z0=z0, Iw=Iw)

    Bmag = sqrt(Bx^2 + By^2 + Bz^2)

    if iszero(Bmag)
        return NaN
    else
        return (Bx * J[1,3] + By * J[2,3] + Bz * J[3,3]) / Bmag / scale_factor
    end
end

function grad_dnormBdz(
    xs::AbstractVector,
    zs::AbstractVector;
    y::Real = 0.0,
    Iw::Real = 0.2,
    z0::Real = 1.3a,
    scale_factor::Real = 1.0,
    axis_order::Symbol = :xz,
)
    if axis_order === :xz
        return [grad_dnormBdz(x, z; y=y, Iw=Iw, z0=z0, scale_factor=scale_factor) for x in xs, z in zs]
    elseif axis_order === :zx
        return [grad_dnormBdz(x, z; y=y, Iw=Iw, z0=z0, scale_factor=scale_factor) for z in zs, x in xs]
    else
        error("axis_order must be :xz or :zx")
    end
end

function ratio_a_dnormBdz_over_normB(
    x::Real,
    z::Real;
    y::Real = 0.0,
    Iw::Real = 0.2,
    z0::Real = 1.3a,
)
    Bmag = norm(B_total(x, y, z; z0=z0, Iw=Iw))

    if iszero(Bmag)
        return NaN
    else
        return a * grad_dnormBdz(x, z; y=y, Iw=Iw, z0=z0, scale_factor=1.0) / Bmag
    end
end

function ratio_a_dnormBdz_over_normB(
    xs::AbstractVector,
    zs::AbstractVector;
    y::Real = 0.0,
    Iw::Real = 0.2,
    z0::Real = 1.3a,
    axis_order::Symbol = :xz,
)
    if axis_order === :xz
        return [
            ratio_a_dnormBdz_over_normB(x, z; y=y, Iw=Iw, z0=z0)
            for x in xs, z in zs
        ]
    elseif axis_order === :zx
        return [
            ratio_a_dnormBdz_over_normB(x, z; y=y, Iw=Iw, z0=z0)
            for z in zs, x in xs
        ]
    else
        error("axis_order must be :xz or :zx")
    end
end


# In the limit ℓ → ∞
function approx_B_total(x,y,z; z0=1.3*a,Iw=0.2)
    ρ1, ρ2 = hypot(x-a, z-z0), hypot(x+a, z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end
    inv_sq_ρ1 = 1/ρ1^2
    inv_sq_ρ2 = 1/ρ2^2
    C = μ0*Iw/(2π)
    Bx = C*(z-z0)*(inv_sq_ρ2 - inv_sq_ρ1)
    Bz = C*((x-a)*inv_sq_ρ1 - (x+a)*inv_sq_ρ2)
    return (Bx,0.0,Bz)
end

function approx_grad_B(x, y, z; z0=1.3*a, Iw=0.2)
    ρ1 = hypot(x-a, z-z0)
    ρ2 = hypot(x+a, z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end

    inv_sq_ρ1 = 1/ρ1^2
    inv_sq_ρ2 = 1/ρ2^2
    C = μ0*Iw/(2π)
    Δz = z - z0


    # ∂ᵢBx
    dBxdx = -2 * C * Δz * ( (x+a)*inv_sq_ρ2^2 - (x-a)*inv_sq_ρ1^2 )
    dBxdz = C * (inv_sq_ρ2-inv_sq_ρ1) + 2 * C * Δz^2 * (inv_sq_ρ1^2-inv_sq_ρ2^2 )
    # ∂ᵢBy = 0 ∀ i
    # ∂ᵢBz
    dBzdx = C * (inv_sq_ρ1-inv_sq_ρ2) - 2 * C * ((x-a)^2*inv_sq_ρ1^2-(x+a)^2*inv_sq_ρ2^2 )
    dBzdz = -2* C * Δz * ( (x-a)*inv_sq_ρ1^2 - (x+a)*inv_sq_ρ2^2 )

    return [
        dBxdx 0.0 dBxdz;
        0.0 0.0 0.0;
        dBzdx 0.0 dBzdz
    ]
end

function approx_dnormBdz(x,z; Iw=0.2, z0=1.3*a)
    ρ1 = hypot(x-a, z-z0)
    ρ2 = hypot(x+a, z-z0)

    Δz = z - z0
    C = μ0*Iw/(2π)

    return -4 * a * C * Δz / (ρ1^3 * ρ2^3) * ( x^2 + a^2 + Δz^2)
end

function approx_normB(x,z; Iw=0.2, z0=1.3*a)
    ρ1 = hypot(x-a, z-z0)
    ρ2 = hypot(x+a, z-z0)

    C = μ0 * Iw / (2π)

    return 2 * a * C  / (ρ1 * ρ2)  
end

function approx_ratio_dBdz_normB(x,z; z0=1.3*a)
    Δz = z - z0
    ρ1 = hypot(x-a, Δz)
    ρ2 = hypot(x+a, Δz)

    return -2 * Δz * (a^2+x^2+Δz^2)*ρ1*ρ2 / ((x^2-a^2)^2 + 2*(x^2+a^2)*Δz^2 + Δz^4)^(3/2)
end

# -----------------------------
# Common geometry
# -----------------------------
slit_x = 0.5 .* [-x_slit, -x_slit,  x_slit,  x_slit, -x_slit]
slit_z = 0.5 .* [-z_slit,  z_slit,  z_slit, -z_slit, -z_slit]

# Pole piece circle 
zc, xc, r = 1.3, 0.0, 1.0 ;   # center (z=1.3, x=0), radius 1
θ = range(0, 2π; length=721);
z_circle =  zc .+ r*cos.(θ);
x_circle =  xc .+ r*sin.(θ);

x_line = a * collect(range(-2, 2, length=10_001))
# Top magnet edge shape
x_fill = x_line / a; 
y_edge = z_magnet_edge.(x_line) / a;
y_top  = fill(2.0, length(x_fill));
# Bottom trench shape
y_trench = z_magnet_trench.(x_line) / a;
y_bottom = fill(-2, length(x_fill));

nx, nz = 601, 1201 ;

# -----------------------------
# Current
# -----------------------------
Iw_set = 0.100 ;

# -------------- Compute the spatial average ------------------------
xmin, xmax = -0.5*x_slit, 0.5*x_slit ;
ymin, ymax = -ℓ, ℓ;
zmin, zmax = -0.5*z_slit, 0.5*z_slit ;

xmin, xmax = -2.2e-3, 2.2e-3 ;
zmin, zmax = -720e-6, 720e-6 ;

function spatial_stats(
    ;
    mode::Symbol,
    dimensionality::Symbol,
    xmin::Real,
    xmax::Real,
    zmin::Real,
    zmax::Real,
    y0::Real = 0.0,
    Iw::Real,
    ymin::Real = -ℓ,
    ymax::Real =  ℓ,
    rtol::Real = 1e-12,
    atol::Real = 1e-16,
    verbose::Bool = true,
)
    mode ∈ (:gradient, :magnitude) || error("mode must be :gradient or :magnitude")
    dimensionality ∈ (:dim2d, :dim3d) || error("dimensionality must be :dim2d or :dim3d")

    xmax > xmin || error("xmax must be greater than xmin")
    zmax > zmin || error("zmax must be greater than zmin")
    if dimensionality === :dim3d
        ymax > ymin || error("ymax must be greater than ymin")
    end

    area_xz = (xmax - xmin) * (zmax - zmin)

    # For the 3D case, the integration volume is a parallelepiped region:
    # x ∈ [xmin, xmax], y ∈ [ymin, ymax], z ∈ [zmin, zmax].
    volume_xyz = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

    integrand(v) = begin
        if dimensionality === :dim2d
            x, z = v
            if mode === :gradient
                grad_dnormBdz(x, z; y = y0, Iw = Iw, scale_factor = 1.0)
            else
                norm(B_total(x, y0, z; Iw = Iw))
            end
        else
            x, y, z = v
            if mode === :gradient
                grad_dnormBdz(x, z; y = y, Iw = Iw, scale_factor = 1.0)
            else
                norm(B_total(x, y, z; Iw = Iw))
            end
        end
    end

    lower = dimensionality === :dim2d ? [xmin, zmin] : [xmin, ymin, zmin]
    upper = dimensionality === :dim2d ? [xmax, zmax] : [xmax, ymax, zmax]
    measure = dimensionality === :dim2d ? area_xz : volume_xyz

    int_mean, err_mean = hcubature(integrand, lower, upper; rtol = rtol, atol = atol)
    avg = int_mean / measure

    integrand_sq(v) = integrand(v)^2
    int_sq, err_sq = hcubature(integrand_sq, lower, upper; rtol = rtol, atol = atol)

    var = int_sq / measure - avg^2
    std = sqrt(max(var, 0.0))
    rel_inhom = iszero(avg) ? NaN : std / abs(avg)

    if verbose
        quantity_label, avg_unit, var_unit, std_unit =
            mode === :gradient ?
            ("d|B|/dz", "T/m", "(T/m)^2", "T/m") :
            ("|B|", "T", "T^2", "T")

        if dimensionality === :dim2d
            header = @sprintf("---- %s: Spatial 2D at y = %.3e m ----", quantity_label, y0)
            dim_label = "Spatial 2D"
        else
            header = @sprintf("---- %s: Spatial 3D rectang. parallelepiped ----", quantity_label)
            dim_label = "Spatial 3D"
        end

        @info header
        @info @sprintf("%-42s = %.3e %s", "$(dim_label) average of $(quantity_label)", avg, avg_unit)
        @info @sprintf("%-42s = %.3e %s", "$(dim_label) variance of $(quantity_label)", var, var_unit)
        @info @sprintf("%-42s = %.3e %s", "$(dim_label) std of $(quantity_label)", std, std_unit)
        @info @sprintf("%-42s = %.1f %%", "$(dim_label) rel. inhomogeneity of $(quantity_label)", 100 * rel_inhom)
        # @info @sprintf("%-42s = %.3e", "Estimated cubature error on mean", err_mean)
        # @info @sprintf("%-42s = %.3e", "Estimated cubature error on 2nd moment", err_sq)
    end

    return (
        mode = mode,
        dimensionality = dimensionality,
        y0 = dimensionality === :dim2d ? y0 : nothing,
        bounds = dimensionality === :dim2d ?
            (xmin = xmin, xmax = xmax, zmin = zmin, zmax = zmax) :
            (xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, zmin = zmin, zmax = zmax),
        measure = measure,
        integral_mean = int_mean,
        err_mean = err_mean,
        average = avg,
        integral_second_moment = int_sq,
        err_second_moment = err_sq,
        variance = var,
        std = std,
        relative_inhomogeneity = rel_inhom,
    )
end

spatial_stats(
    mode = :magnitude,
    dimensionality = :dim2d,
    xmin = xmin,
    xmax = xmax,
    zmin = zmin,
    zmax = zmax,
    y0 = 0.0,
    Iw = Iw_set,
);

spatial_stats(
    mode = :magnitude,
    dimensionality = :dim3d,
    xmin = xmin,
    xmax = xmax,
    zmin = zmin,
    zmax = zmax,
    y0 = 0.0,
    Iw = Iw_set,
);

spatial_stats(
    mode = :gradient,
    dimensionality = :dim2d,
    xmin = xmin,
    xmax = xmax,
    zmin = zmin,
    zmax = zmax,
    y0 = 0.0,
    Iw = Iw_set,
);

spatial_stats(
    mode = :gradient,
    dimensionality = :dim3d,
    xmin = xmin,
    xmax = xmax,
    zmin = zmin,
    zmax = zmax,
    y0 = 0.0,
    Iw = Iw_set,
);


# Volume in a trapezoidal frustum
xmin_in, xmax_in  = -2.25e-2, 2.25e-2 ;
zmin_in, zmax_in  = -180e-6, 180e-6 ;
xmin_out, xmax_out  = -2.6e-2, 2.6e-2 ;
zmin_out, zmax_out  = -240e-6, 240e-6 ; # Iw=0A
# zmin_out, zmax_out  = -120e-6, 720e-6 # Iw=1A

function frustum_stats(mode::Symbol;
    ℓ::Real,
    xmin_in::Real,
    xmax_in::Real,
    zmin_in::Real,
    zmax_in::Real,
    xmin_out::Real,
    xmax_out::Real,
    zmin_out::Real,
    zmax_out::Real,
    Iw::Real,
    rtol::Real = 1e-12,
    atol::Real = 1e-16,
    verbose::Bool = true,
)
    if mode !== :gradient && mode !== :magnitude
        error("mode must be :gradient or :magnitude")
    end

    t_of_y(y) = (y + ℓ) / (2ℓ)

    x_min(y) = xmin_in  + t_of_y(y) * (xmin_out - xmin_in)
    x_max(y) = xmax_in  + t_of_y(y) * (xmax_out - xmax_in)
    z_min(y) = zmin_in  + t_of_y(y) * (zmin_out - zmin_in)
    z_max(y) = zmax_in  + t_of_y(y) * (zmax_out - zmax_in)

    vol_map(v) = begin
        _, _, y = v
        xmin_y = x_min(y)
        xmax_y = x_max(y)
        zmin_y = z_min(y)
        zmax_y = z_max(y)
        (xmax_y - xmin_y) * (zmax_y - zmin_y)
    end

    volume, err_volume = hcubature(
        vol_map,
        [0.0, 0.0, -ℓ],
        [1.0, 1.0,  ℓ];
        rtol = rtol,
        atol = atol,
    )

    f_xyz(x, y, z) = if mode === :gradient
        grad_dnormBdz(x, z; y=y, Iw=Iw, scale_factor=1.0)
    else
        norm(B_total(x, y, z; Iw=Iw)) / 1.0
    end

    f_map(v) = begin
        u, w, y = v

        xmin_y = x_min(y)
        xmax_y = x_max(y)
        zmin_y = z_min(y)
        zmax_y = z_max(y)

        x = xmin_y + u * (xmax_y - xmin_y)
        z = zmin_y + w * (zmax_y - zmin_y)

        jac = (xmax_y - xmin_y) * (zmax_y - zmin_y)

        f_xyz(x, y, z) * jac
    end

    int_mean, err_mean = hcubature(
        f_map,
        [0.0, 0.0, -ℓ],
        [1.0, 1.0,  ℓ];
        rtol = rtol,
        atol = atol,
    )

    avg = int_mean / volume

    fvar_map(v) = begin
        u, w, y = v

        xmin_y = x_min(y)
        xmax_y = x_max(y)
        zmin_y = z_min(y)
        zmax_y = z_max(y)

        x = xmin_y + u * (xmax_y - xmin_y)
        z = zmin_y + w * (zmax_y - zmin_y)

        jac = (xmax_y - xmin_y) * (zmax_y - zmin_y)

        (f_xyz(x, y, z) - avg)^2 * jac
    end

    int_var, err_var = hcubature(
        fvar_map,
        [0.0, 0.0, -ℓ],
        [1.0, 1.0,  ℓ];
        rtol = rtol,
        atol = atol,
    )

    var = int_var / volume
    std = sqrt(max(var, 0.0))
    rel_inhom = iszero(avg) ? NaN : std / abs(avg)

    if verbose
        if mode === :gradient
            @info "---- Trapezoidal frustum: d|B|/dz ----"
            @info @sprintf("%-42s = %.3e m^3",     "Frustum volume", volume)
            @info @sprintf("%-42s = %.3e T/m",     "Volumetric average of d|B|/dz", avg)
            @info @sprintf("%-42s = %.3e (T/m)^2", "Volumetric variance of d|B|/dz", var)
            @info @sprintf("%-42s = %.3e T/m",     "Volumetric std of d|B|/dz", std)
            @info @sprintf("%-42s = %.2f %%",      "Relative vol inhomogeneity of d|B|/dz", 100 * rel_inhom)
        else
            @info "---- Trapezoidal frustum: |B| ----"
            @info @sprintf("%-42s = %.3e m^3", "Frustum volume", volume)
            @info @sprintf("%-42s = %.3e T",   "Volumetric average of |B|", avg)
            @info @sprintf("%-42s = %.3e T^2", "Volumetric variance of |B|", var)
            @info @sprintf("%-42s = %.3e T",   "Volumetric std of |B|", std)
            @info @sprintf("%-42s = %.2f %%",  "Relative vol inhomogeneity of |B|", 100 * rel_inhom)
        end
    end

    return (
        mode = mode,
        volume = volume,
        err_volume = err_volume,
        integral_mean = int_mean,
        err_mean = err_mean,
        average = avg,
        integral_variance = int_var,
        err_variance = err_var,
        variance = var,
        std = std,
        relative_inhomogeneity = rel_inhom,
    )
end

stats_grad = frustum_stats(:gradient;
    ℓ = ℓ,
    xmin_in = xmin_in,
    xmax_in = xmax_in,
    zmin_in = zmin_in,
    zmax_in = zmax_in,
    xmin_out = xmin_out,
    xmax_out = xmax_out,
    zmin_out = zmin_out,
    zmax_out = zmax_out,
    Iw = Iw_set,
);

stats_mag = frustum_stats(:magnitude;
    ℓ = ℓ,
    xmin_in = xmin_in,
    xmax_in = xmax_in,
    zmin_in = zmin_in,
    zmax_in = zmax_in,
    xmin_out = xmin_out,
    xmax_out = xmax_out,
    zmin_out = zmin_out,
    zmax_out = zmax_out,
    Iw = Iw_set
);


# =========================================================
# Plot 1: Gradient zoomed view near slit / pole piece
# ===============================================;
scale_factor = 1e-3;
scale_10power = round(Int, log10(scale_factor));
xmin, xmax = -1*a, 1*a ;
zmin, zmax = -0.2*a, 0.2*a ;

xs = range(xmin, xmax; length=nx);
zs = range(zmin, zmax; length=nz);
# Evaluate on the grid
Z = [approx_dnormBdz(x, z; Iw=Iw_set) / scale_factor for x in xs, z in zs];  # size (nx, nz)
# Filled contour
plt = contour(zs/a, xs/a, Z; 
            levels=40, fill=true, cbar=true,
            xlabel=L"z/a", ylabel=L"x/a", #aspect_ratio=:equal,
            title=L"$\partial_{z} \vert \mathbf{B}(x,z)\vert$ at $I_{w}=%$(Int(1000*Iw_set))\mathrm{mA}$",
            xflip=true,
            colorbar_title = L"$\times \, 10^{%$(scale_10power)} \ \left(\mathrm{T}/\mathrm{m}\right)$",
            );
plot!(plt,
    Shape(slit_z/a, slit_x/a);
    fill = true,
    fillalpha=0.1,
    linecolor = :black,
    linewidth = 2,
    linestyle = :dashdot,
    label = "slit region",
);
# Spatial average-contour overlay
contour!(plt,
    zs/a, xs/a, Z; 
    label=L"$\langle \partial_{z} \vert \mathbf{B}(x,z)\vert \rangle$",
    levels=[Grad_avg/scale_factor], 
    line=(:dot,:limegreen, 2),
);
plot!(plt,
    z_circle, x_circle; 
    fill=true, 
    fillalpha= 0.5, 
    color=:black, 
    lw=2, 
    ls=:solid, 
    label="pole piece", 
);
plot!(plt,
    legend=:topleft,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    right_margin=3mm,
    xlim=(zmin/a,0.35),
    ylim=(-1.0,1.0),
    # aspect_ratio=:equal,
);
display(plt)

# =========================================================
# Plot 2: Gradient wider view with magnet geometry
# =========================================================
scale_factor = 1e-3;
scale_10power = round(Int, log10(scale_factor));
xmin, xmax = -2a, 2a;
zmin, zmax = -1.2*a, 1.2*a;
xs = range(xmin, xmax; length=nx);
zs = range(zmin, zmax; length=nz);
Z = [approx_dnormBdz(x, z; Iw=Iw_set) / scale_factor  for z in zs, x in xs];  # size (nx, nz)
finite = vec(Z[.!isnan.(Z) .& .!isinf.(Z)]);
vmax   = Statistics.quantile(abs.(finite), 0.90);    # tweak 0.99–0.999 as you like

fig = contour(xs/a, zs/a, Z; 
            levels=111, 
            fill=true, 
            cbar=true,
            clims=(0,vmax),
            xlabel=L"x/a", ylabel=L"z/a", aspect_ratio=:equal,
            title=L"$\partial_{z} \vert \mathbf{B}(x,z)\vert$ at $I_{w}=%$(Int(1000*Iw_set))\mathrm{mA}$",
            colorbar_title = L"$\times \, 10^{%$(scale_10power)} \ \left(\mathrm{T}/\mathrm{m}\right)$",
            # color=:default,
            # xflip=true
);
contour!(fig, xs/a, zs/a, Z; 
    label=L"$\langle \partial_{z} B(x,z) \rangle$",
    levels=[Grad_avg/scale_factor], 
    line=(:dot,:limegreen, 2),
);
plot!(fig, [x_fill; reverse(x_fill)], [y_edge; reverse(y_top)];
    seriestype = :shape, label = "Magnet",
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
plot!(fig, [x_fill; reverse(x_fill)], [y_bottom; reverse(y_trench)];
    seriestype = :shape, label = false,
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
plot!(fig,
    # slit_x / a,
    # slit_z / a;
    Shape(slit_x / a, slit_z / a),
    seriestype = :shape, label = "Slit",
    line = (:solid, :red, 1), color = :red, fillalpha = 0.15
);
vline!(fig,[0], line=(:white,0.2), label=false);
hline!(fig,[0], line=(:white,0.2), label=false);
plot!(fig,
    legend=:outerbottom,
    legend_columns=2,
    background_color_legend = :white,
    foreground_color_legend = nothing,
    xlim=(-1.75,1.75), 
    ylim=(-1.0,1.0),
    bottom_margin=-10mm,
);
display(fig)



# =========================================================
# Plot 3: Magnetic field x-component zoomed view near slit / pole piecey
# =========================================================
scale_factor = 1e-6;
scale_10power = round(Int, log10(scale_factor));
xmin, xmax = -1*a, 1*a ;
zmin, zmax = -0.2*a, 0.2*a ;
xs = range(xmin, xmax; length=nx);
zs = range(zmin, zmax; length=nz);
# Evaluate on the grid
Bx_grid   = Bfield_component_grid(xs, zs;  y=0.0, Iw=Iw_set, comp=:x,     scale_factor=scale_factor, axis_order=:xz);
Bz_grid   = Bfield_component_grid(xs, zs;  y=0.0, Iw=Iw_set, comp=:z,     scale_factor=scale_factor, axis_order=:xz);
Bnorm_grid = Bfield_component_grid(xs, zs; y=0.0, Iw=Iw_set, comp=:norm, scale_factor=scale_factor,  axis_order=:xz);
# Filled contour
plt_Bx = contour(
    zs/a, xs/a, Bx_grid;
    levels = 40,
    # color=:balance,
    fill = true,
    cbar = true,
    xlabel = L"z/a",
    ylabel = L"x/a",
    title = L"$B_x(x,z)$ at $I_w=%$(Iw_set)\,\mathrm{A}$",
    xflip = true,
    colorbar_title = L"\times 10^{%$(scale_10power)} \ \mathrm{T}",
);
plot!(plt_Bx,
    Shape(slit_z/a, slit_x/a);
    fill = true,
    fillalpha=0.1,
    linecolor = :red,
    linewidth = 2,
    linestyle = :dashdot,
    label = "slit region",
);
plot!(plt_Bx,
    z_circle, x_circle; 
    fill=true, 
    fillalpha= 0.5, 
    color=:black, 
    lw=2, 
    ls=:solid, 
    label="pole piece", 
);
# Zero-contour overlay
contour!(plt_Bx,zs/a, xs/a, Bx_grid; levels=[0.0], linecolor=:black, linewidth=2, linestyle=:dot,);
plot!(plt_Bx,
    legend=:topleft,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    right_margin=3mm,
    xlim=(zmin/a,0.35),
    ylim=(-1.0,1.0)
);
display(plt_Bx)

plt_Bz = contour(
    zs/a, xs/a, Bz_grid;
    # color=:balance,
    levels = 40,
    fill = true,
    cbar = true,
    xlabel = L"z/a",
    ylabel = L"x/a",
    title = L"$B_z(x,z)$ at $I_w=%$(Iw_set)\,\mathrm{A}$",
    xflip = true,
    colorbar_title = L"\times 10^{%$(scale_10power)} \ \mathrm{T}",
);
plot!(plt_Bz,
    Shape(slit_z/a, slit_x/a);
    fill = true,
    fillalpha=0.1,
    linecolor = :red,
    linewidth = 2,
    linestyle = :dashdot,
    label = "slit region",
);
# Zero-contour overlay
contour!(plt_Bz,zs/a, xs/a, Bz_grid; levels=[0.0], linecolor=:black, linewidth=2, linestyle=:dot,);
plot!(plt_Bz,
    z_circle, x_circle; 
    fill=true, 
    fillalpha= 0.5, 
    color=:black, 
    lw=2, 
    ls=:solid, 
    label="pole piece", 
);
plot!(plt_Bz,
    legend=:topleft,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    right_margin=3mm,
    xlim=(zmin/a,0.35),
    ylim=(-1.0,1.0)
);
display(plt_Bz)

plt_Bnorm = contour(
    zs/a, xs/a, Bnorm_grid;
    levels = 40,
    fill = true,
    cbar = true,
    xlabel = L"z/a",
    ylabel = L"x/a",
    title = L"$\vert \mathbf{B}(x,z) \vert$ at $I_w=%$(Iw_set)\,\mathrm{A}$",
    xflip = true,
    colorbar_title = L"\times 10^{%$(scale_10power)} \ \mathrm{T}",
);
contour!(plt_Bnorm ,
    zs/a, xs/a, Bnorm_grid; 
    label=L"$\langle \vert \mathbf{B}(x,z) \vert \rangle$",
    levels=[B_avg/scale_factor], 
    line=(:dot,:limegreen, 2),
);
plot!(plt_Bnorm,
    Shape(slit_z/a, slit_x/a);
    fill = true,
    fillalpha=0.1,
    linecolor = :red,
    linewidth = 2,
    linestyle = :dashdot,
    label = "slit region",
);
plot!(plt_Bnorm,
    z_circle, x_circle; 
    fill=true, 
    fillalpha= 0.5, 
    color=:black, 
    lw=2, 
    ls=:solid, 
    label="pole piece", 
);
plot!(plt_Bnorm,
    legend=:topleft,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    right_margin=3mm,
    xlim=(zmin/a,0.35),
    ylim=(-1.0,1.0)
);
display(plt_Bnorm)


plt = plot(plt_Bx, plt_Bz, plt_Bnorm,
layout=@layout([a1 ; a2 ; a3]),
size=(600,800),
left_margin=2mm,
);
display(plt)

# =========================================================
# Plot 4: Magnetic fiel wider view with magnet geometry
# =========================================================
scale_factor = 1e-3;
scale_10power = round(Int, log10(scale_factor));
xmin, xmax = -2a, 2a;
zmin, zmax = -1.2*a, 1.2*a;
xs = range(xmin, xmax; length=nx);
zs = range(zmin, zmax; length=nz);
Bx_grid   = Bfield_component_grid(xs, zs;  y=0.0, Iw=Iw_set, comp=:x,     scale_factor=scale_factor, axis_order=:zx);
Bz_grid   = Bfield_component_grid(xs, zs;  y=0.0, Iw=Iw_set, comp=:z,     scale_factor=scale_factor, axis_order=:zx);
Bnorm_grid = Bfield_component_grid(xs, zs; y=0.0, Iw=Iw_set, comp=:norm, scale_factor=scale_factor,  axis_order=:zx);
vmax_x    = Statistics.quantile(abs.(vec(Bx_grid[.!isnan.(Bx_grid) .& .!isinf.(Bx_grid)])), 0.90);      # tweak 0.99–0.999 as you like
vmax_z    = Statistics.quantile(abs.(vec(Bz_grid[.!isnan.(Bz_grid) .& .!isinf.(Bz_grid)])), 0.90);      # tweak 0.99–0.999 as you like
vmax_norm = Statistics.quantile(abs.(vec(Bnorm_grid[.!isnan.(Bnorm_grid) .& .!isinf.(Bnorm_grid)])), 0.90);      # tweak 0.99–0.999 as you like


fig_Bx = contour(xs/a, zs/a, Bx_grid; 
            levels=111, 
            fill=true, 
            cbar=true,
            clims=(-vmax_x,vmax_x),
            xlabel=L"x/a", ylabel=L"z/a", aspect_ratio=:equal,
            title=L"$B_{x}(x,z)$ at $I_{w}=%$(Int(1000*Iw_set))\mathrm{mA}$",
            colorbar_title = L"$\times \, 10^{%$(scale_10power)} \ \left(\mathrm{T}\right)$",
            color=:balance,
            # xflip=true
);
plot!(fig_Bx, [x_fill; reverse(x_fill)], [y_edge; reverse(y_top)];
    seriestype = :shape, label = "Magnet",
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
plot!(fig_Bx, [x_fill; reverse(x_fill)], [y_bottom; reverse(y_trench)];
    seriestype = :shape, label = false,
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
plot!(fig_Bx,
    Shape(slit_x / a, slit_z / a),
    seriestype = :shape, label = "Slit",
    line = (:solid, :red, 1), color = :red, fillalpha = 0.15
);
vline!(fig_Bx,[0], line=(:white,0.2), label=false);
hline!(fig_Bx,[0], line=(:white,0.2), label=false);
plot!(fig_Bx,
    legend=:outerbottom,
    legend_columns=2,
    background_color_legend = :white,
    foreground_color_legend = nothing,
    xlim=(-1.75,1.75), 
    ylim=(-1.0,1.0),
    bottom_margin=-10mm,
);
display(fig_Bx)


fig_Bz = contour(xs/a, zs/a, Bz_grid; 
            levels=111, 
            fill=true, 
            cbar=true,
            clims=(-vmax_z,vmax_z),
            xlabel=L"x/a", ylabel=L"z/a", aspect_ratio=:equal,
            title=L"$B_{z}(x,z)$ at $I_{w}=%$(Int(1000*Iw_set))\mathrm{mA}$",
            colorbar_title = L"$\times \, 10^{%$(scale_10power)} \ \left(\mathrm{T}\right)$",
            color=:balance,
            # xflip=true
);
plot!(fig_Bz, [x_fill; reverse(x_fill)], [y_edge; reverse(y_top)];
    seriestype = :shape, label = "Magnet",
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
plot!(fig_Bz, [x_fill; reverse(x_fill)], [y_bottom; reverse(y_trench)];
    seriestype = :shape, label = false,
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
plot!(fig_Bz,
    Shape(slit_x / a, slit_z / a),
    seriestype = :shape, label = "Slit",
    line = (:solid, :red, 1), color = :red, fillalpha = 0.15
);
vline!(fig_Bz,[0], line=(:white,0.2), label=false);
hline!(fig_Bz,[0], line=(:white,0.2), label=false);
plot!(fig_Bz,
    legend=:outerbottom,
    legend_columns=2,
    background_color_legend = :white,
    foreground_color_legend = nothing,
    xlim=(-1.75,1.75), 
    ylim=(-1.0,1.0),
    bottom_margin=-10mm,
);
display(fig_Bz)

fig_Bnorm = contour(xs/a, zs/a, Bnorm_grid; 
            levels=111, 
            fill=true, 
            cbar=true,
            clims=(0.0,vmax_norm),
            xlabel=L"x/a", ylabel=L"z/a", aspect_ratio=:equal,
            title=L"$\vert \mathbf{B}(x,z) \vert$ at $I_{w}=%$(Int(1000*Iw_set))\mathrm{mA}$",
            colorbar_title = L"$\times \, 10^{%$(scale_10power)} \ \left(\mathrm{T}\right)$",
            # color=:default,
            # xflip=true
);
contour!(fig_Bnorm, 
    xs/a, zs/a, Bnorm_grid; 
    label=L"$\langle \vert \mathbf{B}(x,z) \vert \rangle$",
    levels=[B_avg/scale_factor], 
    line=(:dot,:limegreen, 2),
);
plot!(fig_Bnorm, [x_fill; reverse(x_fill)], [y_edge; reverse(y_top)];
    seriestype = :shape, label = "Magnet",
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
plot!(fig_Bnorm, [x_fill; reverse(x_fill)], [y_bottom; reverse(y_trench)];
    seriestype = :shape, label = false,
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
plot!(fig_Bnorm,
    Shape(slit_x / a, slit_z / a),
    seriestype = :shape, label = "Slit",
    line = (:solid, :red, 1), color = :red, fillalpha = 0.15
);
vline!(fig_Bnorm,[0], line=(:white,0.2), label=false);
hline!(fig_Bnorm,[0], line=(:white,0.2), label=false);
plot!(fig_Bnorm,
    legend=:outerbottom,
    legend_columns=2,
    background_color_legend = :white,
    foreground_color_legend = nothing,
    xlim=(-1.75,1.75), 
    ylim=(-1.0,1.0),
    bottom_margin=-10mm,
);
display(fig_Bnorm)


plot!(fig_Bx; legend=false, bottom_margin=2mm);
plot!(fig_Bz; legend=false, bottom_margin=2mm);
plot!(fig_Bnorm; legend=false, bottom_margin=2mm);
figA = plot(fig_Bx, fig_Bz, fig_Bnorm,
layout=@layout([a1 ; a2 ; a3]),
size=(500,800),
left_margin=2mm,
link=:x,
);
display(figA)


# =========================================================
# Plot: z-derivative of the magnetic field components
# =========================================================
scale_factor = 1e-3;
scale_10power = round(Int, log10(scale_factor));

xmin, xmax = -2a, 2a;
zmin, zmax = -1.2a, 1.2a;

xs = range(xmin, xmax; length=nx);
zs = range(zmin, zmax; length=nz);

# Build grids with orientation compatible with contour(xs, zs, Z)
dBxdz_grid = [
    grad_B(x, 0.0, z; Iw=Iw_set)[1,3] / scale_factor
    for z in zs, x in xs
];
dBzdz_grid = [
    grad_B(x, 0.0, z; Iw=Iw_set)[3,3] / scale_factor
    for z in zs, x in xs
];
norm_dBdz_grid = [
    hypot(
        grad_B(x, 0.0, z; Iw=Iw_set)[1,3],
        grad_B(x, 0.0, z; Iw=Iw_set)[3,3],
    ) / scale_factor
    for z in zs, x in xs
];
dBmagdz_grid = grad_dnormBdz(xs, zs; y = 0.0, Iw=Iw_set, scale_factor=scale_factor, axis_order = :zx)

# Robust color limits
finite_x    = vec(dBxdz_grid[isfinite.(dBxdz_grid)]);
finite_z    = vec(dBzdz_grid[isfinite.(dBzdz_grid)]);
finite_norm = vec(norm_dBdz_grid[isfinite.(norm_dBdz_grid)]);
finite_maggrad = vec(dBmagdz_grid[isfinite.(dBmagdz_grid)]);

vmax_x    = Statistics.quantile(abs.(finite_x), 0.90);
vmax_z    = Statistics.quantile(abs.(finite_z), 0.90);
vmax_norm = Statistics.quantile(abs.(finite_norm), 0.90);
vmax_maggrad = Statistics.quantile(abs.(finite_maggrad), 0.90);

# ---------------- Panel 1: dBxdz ----------------
fig_dBxdz = contour(
    xs/a, zs/a, dBxdz_grid;
    color=:balance,
    levels = 111,
    fill = true,
    cbar = true,
    clims = (-vmax_x, vmax_x),
    xlabel = L"x/a",
    ylabel = L"z/a",
    aspect_ratio = :equal,
    title = L"$\partial_z B_x(x,z)$ at $I_w=%$(Iw_set)\,\mathrm{A}$",
    colorbar_title = L"$\times 10^{%$(scale_10power)}\ \mathrm{T/m}$",
);
plot!(
    fig_dBxdz,
    [x_fill; reverse(x_fill)],
    [y_edge; reverse(y_top)];
    seriestype = :shape,
    label = "Magnet",
    color = :grey36,
    line = (:solid, :grey36),
    fillalpha = 0.75,
);
plot!(
    fig_dBxdz,
    [x_fill; reverse(x_fill)],
    [y_bottom; reverse(y_trench)];
    seriestype = :shape,
    label = false,
    color = :grey36,
    line = (:solid, :grey36),
    fillalpha = 0.75,
);
plot!(
    fig_dBxdz,
    Shape(slit_x/a, slit_z/a);
    fill = true,
    fillalpha = 0.15,
    linecolor = :red,
    linewidth = 1,
    linestyle = :solid,
    label = "Slit",
);
contour!(
    fig_dBxdz,
    xs/a, zs/a, dBxdz_grid;
    levels = [0.0],
    linecolor = :black,
    linewidth = 2,
    linestyle = :dot,
    label = false,
);
vline!(fig_dBxdz, [0]; line = (:white, 0.2), label = false);
hline!(fig_dBxdz, [0]; line = (:white, 0.2), label = false);
plot!(
    fig_dBxdz;
    legend = :outerbottom,
    legend_columns = 2,
    background_color_legend = :white,
    foreground_color_legend = nothing,
    xlim = (-1.75, 1.75),
    ylim = (-1.0, 1.0),
    bottom_margin = -8mm,
)

# ---------------- Panel 2: dBzdz ----------------
fig_dBzdz = contour(
    xs/a, zs/a, dBzdz_grid;
    color=:balance,
    levels = 111,
    fill = true,
    cbar = true,
    clims = (-vmax_z, vmax_z),
    xlabel = L"x/a",
    ylabel = L"z/a",
    aspect_ratio = :equal,
    title = L"$\partial_z B_z(x,z)$ at $I_w=%$(Iw_set)\,\mathrm{A}$",
    colorbar_title = L"$\times 10^{%$(scale_10power)}\ \mathrm{T/m}$",
);
plot!(
    fig_dBzdz,
    [x_fill; reverse(x_fill)],
    [y_edge; reverse(y_top)];
    seriestype = :shape,
    label = "Magnet",
    color = :grey36,
    line = (:solid, :grey36),
    fillalpha = 0.75,
);
plot!(
    fig_dBzdz,
    [x_fill; reverse(x_fill)],
    [y_bottom; reverse(y_trench)];
    seriestype = :shape,
    label = false,
    color = :grey36,
    line = (:solid, :grey36),
    fillalpha = 0.75,
);
plot!(
    fig_dBzdz,
    Shape(slit_x/a, slit_z/a);
    fill = true,
    fillalpha = 0.15,
    linecolor = :red,
    linewidth = 1,
    linestyle = :solid,
    label = "Slit",
);
contour!(
    fig_dBzdz,
    xs/a, zs/a, dBzdz_grid;
    levels = [0.0],
    linecolor = :black,
    linewidth = 2,
    linestyle = :dot,
    label = false,
);
vline!(fig_dBzdz, [0]; line = (:white, 0.2), label = false);
hline!(fig_dBzdz, [0]; line = (:white, 0.2), label = false);
plot!(
    fig_dBzdz;
    legend = :outerbottom,
    legend_columns = 2,
    background_color_legend = :white,
    foreground_color_legend = nothing,
    xlim = (-1.75, 1.75),
    ylim = (-1.0, 1.0),
)

# ---------------- Panel 3: ||dB/dz|| ----------------
fig_norm_dBdz = contour(
    xs/a, zs/a, norm_dBdz_grid;
    levels = 111,
    fill = true,
    cbar = true,
    clims = (0, vmax_norm),
    xlabel = L"x/a",
    ylabel = L"z/a",
    aspect_ratio = :equal,
    title = L"$\|\partial_z \mathbf{B}(x,z)\|$ at $I_w=%$(Iw_set)\,\mathrm{A}$",
    colorbar_title = L"$\times 10^{%$(scale_10power)}\ \mathrm{T/m}$",
);
contour!(fig_norm_dBdz, xs/a, zs/a, norm_dBdz_grid; 
    label=L"$\langle \vert \mathbf{B}(x,z) \vert \rangle$",
    levels=[Grad_avg/scale_factor], 
    line=(:dot,:limegreen, 2),
);
plot!(
    fig_norm_dBdz,
    [x_fill; reverse(x_fill)],
    [y_edge; reverse(y_top)];
    seriestype = :shape,
    label = "Magnet",
    color = :grey36,
    line = (:solid, :grey36),
    fillalpha = 0.75,
);
plot!(
    fig_norm_dBdz,
    [x_fill; reverse(x_fill)],
    [y_bottom; reverse(y_trench)];
    seriestype = :shape,
    label = false,
    color = :grey36,
    line = (:solid, :grey36),
    fillalpha = 0.75,
);
plot!(
    fig_norm_dBdz,
    Shape(slit_x/a, slit_z/a);
    fill = true,
    fillalpha = 0.15,
    linecolor = :red,
    linewidth = 1,
    linestyle = :solid,
    label = "Slit",
);
vline!(fig_norm_dBdz, [0]; line = (:white, 0.2), label = false);
hline!(fig_norm_dBdz, [0]; line = (:white, 0.2), label = false);
plot!(
    fig_norm_dBdz;
    legend = :outerbottom,
    legend_columns = 2,
    background_color_legend = :white,
    foreground_color_legend = nothing,
    xlim = (-1.75, 1.75),
    ylim = (-1.0, 1.0),
)

# ---------------- Panel 4: d||B||/dz ----------------
fig_dBmagdz = contour(
    xs/a, zs/a, dBmagdz_grid;
    levels = 111,
    fill = true,
    cbar = true,
    clims = (0.0, vmax_maggrad),
    xlabel = L"x/a",
    ylabel = L"z/a",
    aspect_ratio = :equal,
    title = L"$\partial_z |\mathbf{B}(x,z)|$ at $I_w=%$(Iw_set)\,\mathrm{A}$",
    colorbar_title = L"$\times 10^{%$(scale_10power)}\ \mathrm{T/m}$",
);
contour!(fig_dBmagdz, xs/a, zs/a, dBmagdz_grid; 
    label=L"$\langle \partial_{z} \vert \mathbf{B}(x,z) \vert \rangle$",
    levels=[Grad_avg/scale_factor], 
    line=(:dot,:limegreen, 2),
);
plot!(
    fig_dBmagdz,
    [x_fill; reverse(x_fill)],
    [y_edge; reverse(y_top)];
    seriestype = :shape,
    label = "Magnet",
    color = :grey36,
    line = (:solid, :grey36),
    fillalpha = 0.75,
);
plot!(
    fig_dBmagdz,
    [x_fill; reverse(x_fill)],
    [y_bottom; reverse(y_trench)];
    seriestype = :shape,
    label = false,
    color = :grey36,
    line = (:solid, :grey36),
    fillalpha = 0.75,
);
plot!(
    fig_dBmagdz,
    Shape(slit_x/a, slit_z/a);
    fill = true,
    fillalpha = 0.15,
    linecolor = :red,
    linewidth = 1,
    linestyle = :solid,
    label = "Slit",
);
vline!(fig_dBmagdz, [0]; line = (:white, 0.2), label = false);
hline!(fig_dBmagdz, [0]; line = (:white, 0.2), label = false);
plot!(
    fig_dBmagdz;
    legend = :outerbottom,
    legend_columns = 2,
    background_color_legend = :white,
    foreground_color_legend = nothing,
    xlim = (-1.75, 1.75),
    ylim = (-1.0, 1.0),
)

# ---------------- Combined figure ----------------
fig_gradz = plot(
    fig_dBxdz,
    fig_dBzdz,
    fig_norm_dBdz,
    fig_dBmagdz;
    layout = (2, 2),
    size = (900,600),
    link = :xy,
    left_margin = 5mm,
)
display(fig_gradz)


# Plot: Gradient zoomed view near slit / pole piece
scale_10power = round(Int, log10(scale_factor));
xmin, xmax = -1*a, 1*a ;
zmin, zmax = -0.2*a, 0.2*a ;
xs = range(xmin, xmax; length=nx);
zs = range(zmin, zmax; length=nz);
# Evaluate on the grid
Z = grad_dnormBdz(xs, zs; y = 0.0, Iw=Iw_set, scale_factor=scale_factor, axis_order = :xz)
# Filled contour
plt = contour(zs/a, xs/a, Z; 
            levels=40, fill=true, cbar=true,
            xlabel=L"z/a", ylabel=L"x/a", #aspect_ratio=:equal,
            title=L"$\partial_{z} \vert \mathbf{B}(x,z) \vert $ at $I_{w}=%$(Int(1000*Iw_set))\mathrm{mA}$",
            xflip=true,
            colorbar_title = L"$\times \, 10^{%$(scale_10power)} \ \left(\mathrm{T}/\mathrm{m}\right)$",
            );
contour!(plt, zs/a, xs/a, Z; 
    label=L"$\langle \partial_{z} \vert \mathbf{B}(x,z) \vert \rangle$",
    levels=[Grad_avg/scale_factor], 
    line=(:dot,:limegreen, 2),
);
plot!(plt,
    Shape(slit_z/a, slit_x/a);
    fill = true,
    fillalpha=0.1,
    linecolor = :black,
    linewidth = 2,
    linestyle = :dashdot,
    label = "slit region",
);
# Zero-contour overlay
plot!(plt,
    z_circle, x_circle; 
    fill=true, 
    fillalpha= 0.5, 
    color=:black, 
    lw=2, 
    ls=:solid, label="pole piece", 
);
plot!(plt,
    legend=:topleft,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    right_margin=3mm,
    xlim=(zmin/a,0.35),
    ylim=(-1.0,1.0),
    # aspect_ratio=:equal,
);
display(plt)



y_path = range(-7e-2, 7e-2; length=100)
Bz1 = [B_total(0, y, 0; Iw=Iw_set)[3] for y in y_path]
Bz2 = [B_total(0, y, 0.150e-3; Iw=Iw_set)[3] for y in y_path]
Bz3 = [B_total(0, y, 0.300e-3; Iw=Iw_set)[3] for y in y_path]
Bz4 = [B_total(0, y, 0.750e-3; Iw=Iw_set)[3] for y in y_path]
plot(1e2*y_path, abs.(1e3*Bz1),
    xlabel=L"$y \ \mathrm{cm}$",
    ylabel=L"$B_z(\vec{r}) \ (\mathrm{m T})$",
    line=(:solid,:red,2),
    label=L"B_z(0,y,0)")
plot!(1e2*y_path, abs.(1e3*Bz2),
    line=(:solid,:blue,2),
    label=L"B_z(0,y,150\mathrm{\mu m })")
plot!(1e2*y_path, abs.(1e3*Bz3),
    line=(:solid,:green,2),
    label=L"B_z(0,y,300\mathrm{\mu m })")
plot!(1e2*y_path, abs.(1e3*Bz4),
    line=(:solid,:purple,2),
    label=L"B_z(0,y,750\mathrm{\mu m })")
vspan!([-3.5,3.5], color=:gray, fillalpha=0.2, label="magnet")



# -------- Gradient to Magnitud ratio
xmin, xmax = -2a, 2a;
zmin, zmax = -1.2a, 1.2a;
xs = range(xmin, xmax; length=nx);
zs = range(zmin, zmax; length=nz);
Z0 = a * [approx_ratio_dBdz_normB(x, z) for z in zs, x in xs] ;
Z =  ratio_a_dnormBdz_over_normB(xs, zs; y=0,Iw=Iw_set, axis_order=:zx) ;
finite = vec(Z[.!isnan.(Z) .& .!isinf.(Z)]);
vmax   = Statistics.quantile(abs.(finite), 0.90)    # tweak 0.99–0.999 as you like
fig = contour(xs/a, zs/a, Z; 
    levels=101, 
    fill=true, 
    cbar=true,
    clims=(0,vmax),
    xlabel=L"x/a", 
    ylabel=L"z/a", 
    aspect_ratio=:equal,
    size = (600, 600),
    title=L"$a\, {\partial_{z}\vert \mathbf{B}(x,z)\vert}/{\vert \mathbf{B}(x,z) \vert}$",
    colorbar_title = L"$\left(\mathrm{T}/\mathrm{m}\right)$",
    titlefontsize = 12,
    top_margin  = -20mm,
)
plot!(fig, [x_fill; reverse(x_fill)], [y_edge; reverse(y_top)];
    seriestype = :shape, label = "Magnet",
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
plot!(fig, [x_fill; reverse(x_fill)], [y_bottom; reverse(y_trench)];
    seriestype = :shape, label = false,
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
plot!(fig,
    # slit_x / a,
    # slit_z / a;
    Shape(slit_x / a, slit_z / a),
    seriestype = :shape, label = "Slit",
    line = (:solid, :red, 1), color = :red, fillalpha = 0.15
);
vline!(fig,[0], line=(:white,0.2), label=false);
hline!(fig,[0], line=(:white,0.2), label=false);
plot!(fig,
    legend=:outerbottom,
    legend_columns=2,
    background_color_legend = :white,
    foreground_color_legend = nothing,
    xlim=(-1.75,1.75), 
    ylim=(-1.0,1.0),
    bottom_margin=-10mm,
)
plot!(xlim=(-1.25,1.25), ylim=(-1.0,1.0), 
    # left_margin=2mm,
    # bottom_margin=-50mm, 
    # right_margin=2mm, 
    # top_margin=-50mm,
    )
hline!([1.3-sqrt(3)], color=:yellow, label=L"$z_{0}-\sqrt{3}$")
hline!([1.3-sqrt(2)], color=:orange, label=L"$z_{0}-\sqrt{2}$")
display(fig)


xmin, xmax = -4/5*a, 4/5*a
zmin, zmax = -3/50*a, 3/50*a
nx, nz = 401, 601
xs = range(xmin, xmax; length=nx)
zs = range(zmin, zmax; length=nz)
Z = a * [approx_ratio_dBdz_normB(x, z) for z in zs, x in xs] ;
fig = contour(xs/a, zs/a, Z; 
    levels=91, 
    fill=true, 
    cbar=true,
    xlabel=L"x/a", 
    ylabel=L"z/a", 
    # aspect_ratio=:equal,
    size = (1200, 600),
    title=L"$a\, {\partial_{z}B(x,z)}/{\vert B(x,z) \vert}$",
    titlefontsize = 12,
    # top_margin  = -20mm,
    bottom_margin = 5mm,
    left_margin = 5mm,
)
plot!(fig,
    Shape(slit_x / a, slit_z / a),
    seriestype = :shape, label = "Slit",
    line = (:solid, :red, 1), color = :red, fillalpha = 0.15
);
display(fig)





using CSV
using DataFrames
df = CSV.read("./SG_BvsI.csv", DataFrame; header=["dI","Bz"])
BvsI = linear_interpolation(df.dI, df.Bz; extrapolation_bc=Line())

iw = range(2e-3,1,100)


plot(iw,BvsI.(iw))
plot!()

plot(iw,approx_ratio_dBdz_normB(0,0)*BvsI.(iw), label=L"$\mathcal{G}=\frac{\epsilon}{a}B$",
    xlabel="Current (A)",
    ylabel=L"Magnetic field Gradient $\mathcal{G}$ (T/m)",
    )
plot!([0.095, 0.2, 0.302, 0.405, 0.498, 0.6, 0.7, 0.75, 0.8, 0.902, 1.01],
    [25.6, 58.4, 92.9, 132.2, 164.2, 196.3, 226, 240, 253.7, 277.2, 298.6],
    label=L"$\mathcal{G}$ from table p.21",
    seriestype=:scatter,
    # xaxis=:log10,
)












plot(iw, [approx_dnormBdz(0, 0; Iw=ix) for ix in iw] )
plot!(iw, [approx_dnormBdz(4/5*a, 0; Iw=ix) for ix in iw] )



plot(xs/a,[approx_dnormBdz(x, 0) for x in xs] ./ [approx_dnormBdz(x, (sqrt(2)-1.3)*a) for x in xs],
    ylims=(0,1.2))



