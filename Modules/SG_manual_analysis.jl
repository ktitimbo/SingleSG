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
using Interpolations
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

B_total(2e-6,1e-6,0)
grad_B(2e-6,1e-6,0)


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

function approx_dBdz(x,z; Iw=0.2, z0=1.3*a)
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

function ratio_dBdz_normB(x,z; z0=1.3*a)
    Δz = z - z0
    ρ1 = hypot(x-a, Δz)
    ρ2 = hypot(x+a, Δz)

    return -2 * Δz * (a^2+x^2+Δz^2)*ρ1*ρ2 / ((x^2-a^2)^2 + 2*(x^2+a^2)*Δz^2 + Δz^4)^(3/2)
end

# Grid
xmin, xmax = -0.75*a, 0.75*a
zmin, zmax = -0.5*a, 0.5*a
nx, nz = 401, 601
xs = range(xmin, xmax; length=nx)
zs = range(zmin, zmax; length=nz)
# Evaluate on the grid
Z = [approx_dBdz(x, z; Iw=0.002) for x in xs, z in zs]  # size (nx, nz)
# Filled contour
plt = contour(zs/a, xs/a, Z; 
            levels=40, fill=true, cbar=true,
            xlabel="z/a", ylabel="x/a", #aspect_ratio=:equal,
            title="∂zB(x,z) contours",
            xflip=true
            )
# Zero-contour overlay
contour!(zs/a, xs/a, Z; levels=[0.0], linecolor=:black, linewidth=2)
# Circle params (in the same units as your data)
zc, xc, r = 1.3, 0.0, 1.0          # center (z=1.3, x=0), radius 1
θ = range(0, 2π; length=361)
z_circle =  zc .+ r*cos.(θ)
x_circle =  xc .+ r*sin.(θ)
plot!(z_circle, x_circle; 
    fill=true, 
    fillalpha= 0.5, 
    color=:black, 
    lw=2, 
    ls=:dash, label="pole piece", 
    xlim=(-0.5,0.5),
    ylim=(-0.75,0.75))
display(plt)

# Grid
xmin, xmax = -2a, 2a;
zmin, zmax = -1.2*a, 1.0*a;
nx, nz = 401, 801;
xs = range(xmin, xmax; length=nx);
zs = range(zmin, zmax; length=nz);
Z = [approx_dBdz(x, z; Iw=0.1) for z in zs, x in xs]  # size (nx, nz)
finite = vec(Z[.!isnan.(Z) .& .!isinf.(Z)])
vmax   = quantile(abs.(finite), 0.97)      # tweak 0.99–0.999 as you like

fig = contour(xs/a, zs/a, Z; 
            levels=101, 
            fill=true, 
            cbar=true,
            clims=(0,vmax),
            xlabel=L"x/a", ylabel=L"z/a", aspect_ratio=:equal,
            title=L"$\partial_{z}B(x,z)$ – contours",
            # color=:default,
            # xflip=true
);
x_line = a * collect(range(-2, 2, length=10_001))
# Top magnet edge shape
x_fill = x_line / a 
y_edge = z_magnet_edge.(x_line) / a
y_top  = fill(2.0, length(x_fill))
plot!(fig, [x_fill; reverse(x_fill)], [y_edge; reverse(y_top)];
    seriestype = :shape, label = "Magnet",
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
# Bottom trench shape
y_trench = z_magnet_trench.(x_line) / a
y_bottom = fill(-2, length(x_fill))
plot!(fig, [x_fill; reverse(x_fill)], [y_bottom; reverse(y_trench)];
    seriestype = :shape, label = false,
    color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
);
# Slit rectangle
x_slit = 4e-3
z_slit = 300e-6
plot!(fig,
    0.5 .* [-x_slit, -x_slit, x_slit,  x_slit, -x_slit] / a,
    0.5 .* [-z_slit,  z_slit, z_slit, -z_slit, -z_slit] / a;
    seriestype = :shape, label = "Slit",
    line = (:solid, :red, 1), color = :red, fillalpha = 0.2
);
plot!(xlim=(-1.75,1.75), ylim=(-1.0,1.0));
vline!(fig,[0], line=(:white,0.2), label=false)
hline!(fig,[0], line=(:white,0.2), label=false)
display(fig)




Z = a * [ratio_dBdz_normB(x, z) for z in zs, x in xs] ;
fig = contour(xs/a, zs/a, Z; 
    levels=501, 
    fill=true, 
    cbar=true,
    xlabel=L"x/a", 
    ylabel=L"z/a", 
    aspect_ratio=:equal,
    size = (600, 600),
    title=L"$a\, {\partial_{z}B(x,z)}/{\vert B(x,z) \vert}$",
    titlefontsize = 12,
    top_margin  = -20mm,
)
x_line = a * collect(range(-1.75, 1.75, length=10_001))
# Top magnet edge shape
x_fill = x_line / a 
y_edge = z_magnet_edge.(x_line) / a
y_top  = fill(1.6, length(x_fill))
plot!(fig, [x_fill; reverse(x_fill)], [y_edge; reverse(y_top)];
    seriestype = :shape, 
    label = false,
    color = :grey36, 
    line = (:solid, :grey36), 
    fillalpha = 0.75
)
# Bottom trench shape
y_trench = z_magnet_trench.(x_line) / a
y_bottom = fill(-1.6, length(x_fill))
plot!(fig, [x_fill; reverse(x_fill)], [y_bottom; reverse(y_trench)];
    seriestype = :shape, 
    label = false,
    color = :grey36, 
    line = (:solid, :grey36), 
    fillalpha = 0.75
);
# Slit rectangle
x_slit = 4e-3
z_slit = 300e-6
plot!(fig,
    0.5 .* [-x_slit, -x_slit, x_slit,  x_slit, -x_slit] / a,
    0.5 .* [-z_slit,  z_slit, z_slit, -z_slit, -z_slit] / a;
    seriestype = :shape, label = "Slit",
    line = (:solid, :red, 1), color = :red, fillalpha = 0.2
);
plot!(xlim=(-1.25,1.25), ylim=(-1.2,1.0), 
    # left_margin=2mm,
    # bottom_margin=-20mm, 
    # right_margin=2mm, 
    top_margin=-50mm,
    )
hline!([1.3-sqrt(3)], color=:yellow, label=L"$z_{0}-\sqrt{3}$")
hline!([1.3-sqrt(2)], color=:orange, label=L"$z_{0}-\sqrt{2}$")
display(fig)


xmin, xmax = -4/5*a, 4/5*a
zmin, zmax = -3/50*a, 3/50*a
nx, nz = 401, 601
xs = range(xmin, xmax; length=nx)
zs = range(zmin, zmax; length=nz)
Z = a * [ratio_dBdz_normB(x, z) for z in zs, x in xs] ;
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
# Slit rectangle
x_slit = 4e-3
z_slit = 300e-6
plot!(fig,
    0.5 .* [-x_slit, -x_slit, x_slit,  x_slit, -x_slit] / a,
    0.5 .* [-z_slit,  z_slit, z_slit, -z_slit, -z_slit] / a;
    seriestype = :shape, label = false,
    line = (:solid, :red, 1), color = :red, fillalpha = 0.01
);
display(fig)


df = CSV.read("./SG_BvsI.csv", DataFrame; header=["dI","Bz"])
BvsI = linear_interpolation(df.dI, df.Bz; extrapolation_bc=Line())

iw = range(2e-3,1,100)


plot(iw,BvsI.(iw))
plot!()

plot(iw,ratio_dBdz_normB(0,0)*BvsI.(iw), label=L"$\mathcal{G}=\frac{\epsilon}{a}B$",
    xlabel="Current (A)",
    ylabel=L"Magnetic field Gradient $\mathcal{G}$ (T/m)",
    )
plot!([0.095, 0.2, 0.302, 0.405, 0.498, 0.6, 0.7, 0.75, 0.8, 0.902, 1.01],
    [25.6, 58.4, 92.9, 132.2, 164.2, 196.3, 226, 240, 253.7, 277.2, 298.6],
    label=L"$\mathcal{G}$ from table p.21",
    seriestype=:scatter,
    # xaxis=:log10,
)












plot(iw, [approx_dBdz(0, 0; Iw=ix) for ix in iw] )
plot!(iw, [approx_dBdz(4/5*a, 0; Iw=ix) for ix in iw] )



plot(xs/a,[approx_dBdz(x, 0) for x in xs] ./ [approx_dBdz(x, (sqrt(2)-1.3)*a) for x in xs],
    ylims=(0,1.2))



