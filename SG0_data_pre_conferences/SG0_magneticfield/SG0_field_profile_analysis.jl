# =============================================================================
# SG0 field profile along the beam axis - Hall probe data of 2026-10-22
# =============================================================================
#
# Section 1  Raw data: read the Hall-probe readings and plot B(x) for the three
#            coil currents (linear and log scale).
# Section 2  Fitting: center field B(x = 0) vs. coil current, linear and quadratic fits.
# Section 3  Relative intensity: B(x)/B(0) in percent for each current, as in the PDF.
# Section 4  Half-sigmoid fit of the relative intensity.
# Section 5  Global half-sigmoid fit: one set of parameters for all three currents.
# Section 6  Full profile: the fitted functions on both sides of the center (-x and +x).
#
# ---------- Data ----------
#
# Source: SG0_field_profile_2026-10-22.pdf, table "Raw data (G)".
# File:   SG0_field_profile_2026-10-22.csv (same folder as this script)
#
#   x_mm      position along the beam (mm)
#   B_129mA   reading (gauss) at coil current 129 mA
#   B_443mA   reading (gauss) at coil current 443 mA
#   B_963mA   reading (gauss) at coil current 963 mA
#
# Empty cell = no reading at that position (0.443 A at 40.64, 43.18 and 48.26-55.88 mm).
# Note: pdftotext places the last 0.443 A value (43.5 G) at 43.18 mm; in the PDF it
# sits at 45.72 mm, which matches the printed 24.7 % = 43.5/176.1.
# 1 G = 1e-4 T.
#
# ---------- Setup (from the PDF) ----------
#
# Magnet:  SG0, circular pole faces of 40 mm diameter, pole gap just over 40 mm.
# Probe:   Hall probe on a micrometer stage, 0.635 mm per turn; readings every
#          2.54 mm (four turns); no beam tube installed.
# History: magnet not degaussed; all three sets on the rising branch of the same
#          hysteresis loop (after negative current), at positive current.
#
# Coordinates: x is the propagation direction, from SG0 toward SG1; x = 0 is the
# field center and only the exit half (x >= 0) was scanned. The horizontal center
# was found in horizontal scans by (1) the geometric center between the pole faces
# and (2) the field minimum / dB/dz ~ 0 (vertex of a parabola fitted to each scan).
# Beyond x ~ 33 mm the horizontal profile is flat and the geometric center was used.
#
# ---------- Values quoted in the PDF (for later checks) ----------
#
# Remanence, fit B = Br + k I at x = 0 to the 0.129 A and 0.963 A sets:
#   k  = 438 G/A,  Br = -21 G  (measured directly at the center at 0 A: -20 G)
#   |Br| / (k I) = 37 % at 0.129 A and 5 % at 0.963 A; Br falls along x more slowly than k I
# Normalized field B/B(0) at 45.72 mm: 20.9 % (0.129 A), 24.7 % (0.443 A), 25.7 % (0.963 A)
# Field falls to 50 % of B(0) at x ~ 30.5 mm (0.443 A and 0.963 A).
# The remanent field depends on the direction of the last current applied.
#
# ---------- Output ----------
#
#   figures/SG0_fig1_raw.{pdf,png}                  Section 1
#   figures/SG0_fig2_center_vs_current.{pdf,png}    Section 2
#   figures/SG0_fig3_relative.{pdf,png}             Section 3
#   figures/SG0_fig4_relative_fit.{pdf,png}         Section 4
#   figures/SG0_fig5_global_fit.{pdf,png}           Section 5
#   figures/SG0_fig6_full_profile.{pdf,png}         Section 6
#
# Packages: ] add CSV DataFrames CairoMakie LaTeXStrings LsqFit
# Run in a fresh Julia session (not one where Plots was loaded: it clashes with Makie names).
using CSV, DataFrames, CairoMakie
using LaTeXStrings, LinearAlgebra, LsqFit, Printf
# Set the working directory to the current location
cd(@__DIR__) ;

# Computer Modern (New Computer Modern, bundled with Makie), with minor ticks and
# minor gridlines on every axis (lighter than the major grid)
set_theme!(merge(Theme(Axis = (xminorticksvisible = true, yminorticksvisible = true,
                               xminorgridvisible  = true, yminorgridvisible  = true,
                               xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05))),
                 theme_latexfonts()))

# ---------- Constants and shared helpers ----------

const DATAFILE = "SG0_field_profile_2026-10-22.csv"
const FIGDIR   = "figures"

# One-hue ordinal ramp (light -> dark = low -> high current), validated for a light surface
const COLORS   = [colorant"#6da7ec", colorant"#2a78d6", colorant"#104281"]
const MARKERS  = [:circle, :rect, :utriangle]
const INK2     = colorant"#5f5e5a"                     # secondary ink: fit lines, reference lines

# Field at the center measured at I = 0 (G). Quoted in the PDF text, not in the CSV;
# plotted for comparison only, not used in the fits.
const B_CENTER_AT_0A = -20.0

const XLABEL_MM = L"x\ \mathrm{(mm)}"
const POLE_RADIUS_MM = 20.0                            # shaded band x = 0-20 mm in the B(x) plots

savefig(fig, name) = (mkpath(FIGDIR);
                      save(joinpath(FIGDIR, name * ".pdf"), fig);
                      save(joinpath(FIGDIR, name * ".png"), fig; px_per_unit = 3))

# x range rounded up to the next 10 mm, shared by the B(x) plots
x_axis_max(S) = 10 * ceil(maximum(s.x[end] for s in S) / 10)

# Shaded band under the pole face, drawn first so it sits behind the data
pole_band!(ax) = vspan!(ax, 0, POLE_RADIUS_MM; color = (:gray, 0.15))


# #############################################################################
# Section 1 - Raw data: extraction and plotting
# #############################################################################

load_profile(file = DATAFILE) = CSV.read(file, DataFrame)

# Coil currents from the CSV header: every column "B_<n>mA" holds the readings (G)
# at n mA. Returns (currents in A, column names), sorted from low to high current.
function current_columns(df)
    cols = [(parse(Int, m[1]) / 1000, name) for name in names(df)
            for m in (match(r"^B_(\d+)mA$", name),) if m !== nothing]
    isempty(cols) && error("no B_<n>mA columns in the data file")
    length(cols) > length(COLORS) &&
        error("$(length(cols)) currents but only $(length(COLORS)) colors: extend COLORS and MARKERS")
    sort!(cols)
    first.(cols), last.(cols)
end

# One current's readings with the empty cells dropped: (I, label, x (mm), B (G))
function series(df, col, I)
    B = df[!, col]
    m = .!ismissing.(B)
    (I = I, label = @sprintf("%.3f A", I),
     x = Float64.(df.x_mm[m]), B = Float64.(B[m]))
end

# Decade tick label: 10^-1, 1, 10, 10^2, 10^3, ...
decade_label(k) = k == 0 ? L"1" : k == 1 ? L"10" : latexstring("10^{$k}")

# Raw readings, linear (top) and log (bottom) scale, sharing the x axis.
# The log panel keeps the 0.129 A set readable next to the 0.963 A set; its
# y range spans the whole decades that contain the data.
function plot_raw(S)
    fig = Figure(size = (720, 760))
    ylabel = L"B\ \mathrm{(G)}"
    xmax   = x_axis_max(S)
    xticks = 0:10:xmax
    kmin   = floor(Int, log10(minimum(minimum(s.B) for s in S)))
    kmax   = ceil(Int, log10(maximum(maximum(s.B) for s in S)))

    ax1 = Axis(fig[1, 1]; ylabel, xticks, title = "Raw readings")
    ax2 = Axis(fig[2, 1]; ylabel, xticks, xlabel = XLABEL_MM, yscale = log10,
               yticks = (10.0 .^ (kmin:kmax), decade_label.(kmin:kmax)),
               yminorticks = IntervalsBetween(9))
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid = false, ticks = false, minorgrid = false, minorticks = false)
    xlims!(ax2, 0, xmax)
    ylims!(ax2, 10.0^kmin, 10.0^kmax)

    for ax in (ax1, ax2)
        pole_band!(ax)
    end
    for (i, s) in enumerate(S), ax in (ax1, ax2)
        scatterlines!(ax, s.x, s.B; color = COLORS[i], marker = MARKERS[i],
                      markersize = 9, linewidth = 2, label = s.label)
    end
    axislegend(ax1; position = :rt, framevisible = false)
    fig
end

df = load_profile()
currents, columns = current_columns(df)
S  = [series(df, col, I) for (col, I) in zip(columns, currents)]

println("Section 1 - raw data")
for s in S
    @printf("  %s: %2d readings, x = %.2f-%.2f mm, B(0) = %.1f G\n",
            s.label, length(s.x), s.x[1], s.x[end], s.B[1])
end

fig1 = plot_raw(S)
display(fig1)
savefig(fig1, "SG0_fig1_raw")


# #############################################################################
# Section 2 - Fitting: center field B(x = 0) vs. coil current
# #############################################################################

# Center field B(x = 0) of each set: the maximum of the profile
function center_field(S)
    for s in S
        s.x[1] == 0 || error("$(s.label): no reading at x = 0")
    end
    [s.I for s in S], [s.B[1] for s in S]
end

# Polynomial least-squares fit B(I) = c0 + c1 I + ... + c_deg I^deg.
# Returns coefficients, their standard errors (NaN when there are no degrees of freedom),
# residuals, degrees of freedom and the coefficient of determination R².
function polyfit(I, B, deg)
    X   = [Ii^p for Ii in I, p in 0:deg]
    c   = X \ B
    res = B .- X * c
    dof = length(B) - (deg + 1)
    se  = dof > 0 ? sqrt.(sum(abs2, res) / dof .* diag(inv(X' * X))) : fill(NaN, deg + 1)
    r2  = 1 - sum(abs2, res) / sum(abs2, B .- sum(B) / length(B))
    (c = c, se = se, res = res, dof = dof, r2 = r2)
end

polyval(c, I) = sum(c[p+1] .* I .^ p for p in 0:length(c)-1)

# Center field vs. current: data, linear fit (B = Br + k I) and the quadratic through
# the three points (exact, zero degrees of freedom, so its R² = 1 by construction).
function plot_center(Ic, Bc, lin, quad)
    fig = Figure(size = (720, 520))
    ax  = Axis(fig[1, 1]; xticks = 0:0.1:1.0, xlabel = L"I\ \mathrm{(A)}",
               ylabel = L"B(x = 0)\ \mathrm{(G)}", title = "Center field vs. coil current")

    If = range(0, 1.0; length = 200)
    hlines!(ax, 0; color = INK2, linewidth = 1)
    lines!(ax, If, polyval(lin.c, If); color = INK2, linewidth = 2,
           label = @sprintf("linear: %.1f + %.1f I,  R² = %.5f", lin.c..., lin.r2))
    lines!(ax, If, polyval(quad.c, If); color = INK2, linewidth = 1.5, linestyle = :dash,
           label = @sprintf("quadratic: %.1f + %.1f I %+.1f I²,  R² = %.0f", quad.c..., quad.r2))
    scatter!(ax, [0.0], [B_CENTER_AT_0A]; marker = :diamond, markersize = 13,
             color = :transparent, strokecolor = INK2, strokewidth = 1.5,
             label = "0 A, quoted in PDF (not fitted)")
    for i in eachindex(Ic)
        scatter!(ax, [Ic[i]], [Bc[i]]; color = COLORS[i], marker = MARKERS[i], markersize = 13,
                 label = @sprintf("%.3f A", Ic[i]))
    end
    axislegend(ax; position = :lt, framevisible = false)
    xlims!(ax, -0.03, 1.0)   # small margin so the 0 A marker is not clipped
    fig
end

# With three points the linear fit has 1 degree of freedom and the quadratic passes
# through all of them (0 dof), so the data cannot decide between the two; the linear
# model B = Br + k I is the physical default below saturation.
Ic, Bc = center_field(S)
lin    = polyfit(Ic, Bc, 1)
quad   = polyfit(Ic, Bc, 2)

println("\nSection 2 - center field B(x = 0) vs. current")
for (I, B, r) in zip(Ic, Bc, lin.res)
    @printf("  I = %.3f A   B = %6.1f G   linear residual = %+5.2f G\n", I, B, r)
end
@printf("  linear:    B = Br + k I,  Br = %.1f ± %.1f G,  k = %.1f ± %.1f G/A,  R² = %.5f (dof = %d)\n",
        lin.c[1], lin.se[1], lin.c[2], lin.se[2], lin.r2, lin.dof)
@printf("             B = 0 at I = %.1f mA;  B(0 A) = %.1f G vs. %.1f G quoted in the PDF\n",
        -1e3 * lin.c[1] / lin.c[2], lin.c[1], B_CENTER_AT_0A)
@printf("  quadratic: B = %.1f + %.1f I %+.1f I²  (exact through the 3 points)\n", quad.c...)
@printf("             B(0 A) = %.1f G\n", quad.c[1])

fig2 = plot_center(Ic, Bc, lin, quad)
display(fig2)
savefig(fig2, "SG0_fig2_center_vs_current")


# #############################################################################
# Section 3 - Relative intensity B(x)/B(0)
# #############################################################################

# Relative intensity in percent: 100 B(x)/B(x = 0), per current.
# Each set is normalized to its own center reading, so the profile shapes can be
# compared across currents (as in the PDF table "Normalized field B/B(0) vs. position").
relative(s) = 100 .* s.B ./ s.B[1]

# Position where the relative profile first drops to `level` (%), by linear
# interpolation between neighboring readings; NaN if it never gets there.
function crossing(x, y, level)
    i = findfirst(j -> y[j] >= level > y[j+1], 1:length(y)-1)
    i === nothing && return NaN
    x[i] + (level - y[i]) * (x[i+1] - x[i]) / (y[i+1] - y[i])
end

# Relative intensity vs. x, one curve per current, with the 50 % level marked.
function plot_relative(S, R)
    fig  = Figure(size = (720, 480))
    xmax = x_axis_max(S)
    ax   = Axis(fig[1, 1]; xticks = 0:10:xmax, yticks = 0:20:100, xlabel = XLABEL_MM,
                ylabel = L"B/B(0)\ (\%)", title = "Relative intensity along the beam axis")
    pole_band!(ax)
    hlines!(ax, 50; color = INK2, linewidth = 1, linestyle = :dash)
    for (i, (s, r)) in enumerate(zip(S, R))
        scatterlines!(ax, s.x, r; color = COLORS[i], marker = MARKERS[i],
                      markersize = 9, linewidth = 2, label = s.label)
    end
    axislegend(ax; position = :rt, framevisible = false)
    xlims!(ax, 0, xmax)
    ylims!(ax, 0, 105)
    fig
end

R = [relative(s) for s in S]

# Table in the layout of the PDF: one row per position, blank where there is no reading
println("\nSection 3 - relative intensity B/B(0) (%)")
@printf("  %7s", "x (mm)"); foreach(s -> @printf("  %9s", s.label), S); println()
for x in df.x_mm
    @printf("  %7.2f", x)
    for (s, r) in zip(S, R)
        j = findfirst(==(x), s.x)
        j === nothing ? @printf("  %9s", "") : @printf("  %9.1f", r[j])
    end
    println()
end
for (s, r) in zip(S, R)
    @printf("  %s: B/B(0) = 50 %% at x = %.2f mm\n", s.label, crossing(s.x, r, 50.0))
end

fig3 = plot_relative(S, R)
display(fig3)
savefig(fig3, "SG0_fig3_relative")


# #############################################################################
# Section 4 - Half-sigmoid fit of the relative intensity
# #############################################################################
#
# Model (log-logistic / Hill form, a sigmoid in ln x):
#
#     y(x) = A / (1 + (x/x50)^n)        y in %, x in mm, x >= 0
#
# - read as a function of |x| it is the right half of a symmetric bell: even in x,
#   flat at the center (dy/dx = 0 at x = 0 for n > 1), as the field of a symmetric
#   magnet must be
# - y(x50) = A/2: x50 is the half-amplitude point; n sets the steepness of the edge
# - tail ~ x^-n at large x (n -> 3 would be the far field of a magnetic dipole)
# - A is left free (close to 100 %) so the center reading alone does not pin the fit
#
# The plain logistic 1/(1 + exp((x - x0)/w)) was also tried: it needs A ~ 115 % and has
# a nonzero slope at x = 0, and fits worse (rms 1.2-1.7 % points), so it is not used.

halfsigmoid(x, p) = p[1] ./ (1 .+ (x ./ p[2]) .^ p[3])

# Least-squares fit of one relative profile (all points weighted equally).
# Returns parameters, standard errors, R², rms residual (% points) and the
# position where the fitted curve itself crosses 50 %.
function fit_halfsigmoid(x, y)
    f   = curve_fit(halfsigmoid, x, y, [100.0, 30.0, 2.5];
                    lower = [50.0, 1.0, 1.01], upper = [150.0, 200.0, 10.0])
    p   = coef(f)
    r2  = 1 - sum(abs2, f.resid) / sum(abs2, y .- sum(y) / length(y))
    rms = sqrt(sum(abs2, f.resid) / length(f.resid))
    x50pct = p[2] * (p[1] / 50 - 1)^(1 / p[3])
    (p = p, se = stderror(f), r2 = r2, rms = rms, x50pct = x50pct)
end

# Relative intensity (markers) with the fitted half-sigmoids (lines)
function plot_relative_fit(S, R, F)
    fig  = Figure(size = (720, 480))
    xmax = x_axis_max(S)
    ax   = Axis(fig[1, 1]; xticks = 0:10:xmax, yticks = 0:20:100, xlabel = XLABEL_MM,
                ylabel = L"B/B(0)\ (\%)",
                title = "Half-sigmoid fit:  y = A / [1 + (x/x₅₀)ⁿ]")
    pole_band!(ax)
    hlines!(ax, 50; color = INK2, linewidth = 1, linestyle = :dash)
    xf = range(0, xmax; length = 300)
    for (i, (s, r, f)) in enumerate(zip(S, R, F))
        lines!(ax, xf, halfsigmoid(xf, f.p); color = COLORS[i], linewidth = 2,
               label = @sprintf("%s:  x₅₀ = %.1f mm,  n = %.2f,  R² = %.4f", s.label, f.p[2], f.p[3], f.r2))
        scatter!(ax, s.x, r; color = COLORS[i], marker = MARKERS[i], markersize = 9)
    end
    axislegend(ax; position = :rt, framevisible = false)
    xlims!(ax, 0, xmax)
    ylims!(ax, 0, 105)
    fig
end

F = [fit_halfsigmoid(s.x, r) for (s, r) in zip(S, R)]

println("\nSection 4 - half-sigmoid fit  y = A / (1 + (x/x50)^n)")
for (s, f) in zip(S, F)
    @printf("  %s:  A = %.2f ± %.2f %%,  x50 = %.2f ± %.2f mm,  n = %.3f ± %.3f,  R² = %.5f,  rms = %.2f %% pts,  fit = 50 %% at x = %.2f mm\n",
            s.label, f.p[1], f.se[1], f.p[2], f.se[2], f.p[3], f.se[3], f.r2, f.rms, f.x50pct)
end

fig4 = plot_relative_fit(S, R, F)
display(fig4)
savefig(fig4, "SG0_fig4_relative_fit")


# #############################################################################
# Section 5 - Global half-sigmoid fit: one function for all three currents
# #############################################################################
#
# Same model as Section 4, y = A / (1 + (x/x50)^n), but a single (A, x50, n) fitted to
# the relative intensity of all currents at once: least squares on the pooled points,
# every reading weighted equally (so 0.129 A and 0.963 A, with 23 points each, count
# more than 0.443 A with 17). The per-set R² and rms show how well the common curve
# describes each current.

# Goodness of a given curve on one data set: R² and rms residual (% points)
function goodness(x, y, p)
    res = y .- halfsigmoid(x, p)
    (r2 = 1 - sum(abs2, res) / sum(abs2, y .- sum(y) / length(y)),
     rms = sqrt(sum(abs2, res) / length(res)))
end

# Relative intensity of every current (markers) with the single global curve (line)
function plot_global_fit(S, R, G, Gset)
    fig  = Figure(size = (720, 480))
    xmax = x_axis_max(S)
    ax   = Axis(fig[1, 1]; xticks = 0:10:xmax, yticks = 0:20:100, xlabel = XLABEL_MM,
                ylabel = L"B/B(0)\ (\%)",
                title = "Global half-sigmoid fit (all currents):  y = A / [1 + (x/x₅₀)ⁿ]")
    pole_band!(ax)
    hlines!(ax, 50; color = INK2, linewidth = 1, linestyle = :dash)
    xf = range(0, xmax; length = 300)
    lines!(ax, xf, halfsigmoid(xf, G.p); color = :black, linewidth = 2,
           label = @sprintf("A = %.1f %%,  x₅₀ = %.2f mm,  n = %.3f,  R² = %.4f",
                            G.p[1], G.p[2], G.p[3], G.r2))
    for (i, (s, r, g)) in enumerate(zip(S, R, Gset))
        scatter!(ax, s.x, r; color = COLORS[i], marker = MARKERS[i], markersize = 9,
                 label = @sprintf("%s:  R² = %.4f,  rms = %.2f %% pts", s.label, g.r2, g.rms))
    end
    axislegend(ax; position = :rt, framevisible = false)
    xlims!(ax, 0, xmax)
    ylims!(ax, 0, 105)
    fig
end

G    = fit_halfsigmoid(vcat((s.x for s in S)...), vcat(R...))
Gset = [goodness(s.x, r, G.p) for (s, r) in zip(S, R)]

println("\nSection 5 - global half-sigmoid fit (all currents together)")
@printf("  A = %.2f ± %.2f %%,  x50 = %.2f ± %.2f mm,  n = %.3f ± %.3f\n",
        G.p[1], G.se[1], G.p[2], G.se[2], G.p[3], G.se[3])
@printf("  pooled: R² = %.5f,  rms = %.2f %% pts (%d points);  fit = 50 %% at x = %.2f mm\n",
        G.r2, G.rms, sum(length(s.x) for s in S), G.x50pct)
for (s, g, f) in zip(S, Gset, F)
    @printf("  %s:  R² = %.5f,  rms = %.2f %% pts   (own fit, Section 4: rms = %.2f %% pts)\n",
            s.label, g.r2, g.rms, f.rms)
end

fig5 = plot_global_fit(S, R, G, Gset)
display(fig5)
savefig(fig5, "SG0_fig5_global_fit")


# #############################################################################
# Section 6 - Full profile: fitted functions on both sides of the center
# #############################################################################
#
# Only the exit half (x >= 0) was measured. Assuming the field is symmetric about the
# center, the half-sigmoid evaluated at |x| gives the whole profile, entrance (x < 0)
# and exit (x > 0). The experimental points are shown where they were taken (x >= 0);
# the curves on the x < 0 side are the mirror image, not a measurement.

# Full symmetric profile from a half-sigmoid fit
fullprofile(x, p) = halfsigmoid(abs.(x), p)

# Per-current fits (Section 4, colored) and the global fit (Section 5, black dashed)
# over -xmax..xmax, with the measured points on the x >= 0 side
function plot_full_profile(S, R, F, G)
    fig  = Figure(size = (1024, 768))
    # Half-width of the plotted range: 1.5x the measured range, rounded to 10 mm.
    # Ticks, curves, limits and labels all follow it.
    xlim = 10 * ceil(1.5 * x_axis_max(S) / 10)
    ax   = Axis(fig[1, 1]; xticks = -xlim:10:xlim, yticks = 0:20:100, xlabel = XLABEL_MM,
                ylabel = L"B/B(0)\ (\%)",
                title = "Full field profile: half-sigmoid fits mirrored about x = 0")
    vspan!(ax, -POLE_RADIUS_MM, POLE_RADIUS_MM; color = (:gray, 0.15))   # pole face
    hlines!(ax, 50; color = INK2, linewidth = 1, linestyle = :dash)
    vlines!(ax, 0; color = INK2, linewidth = 1)
    # Side labels placed above the curve tails (which are below ~15 % near the edges)
    text!(ax, -xlim + 2, 30; text = "mirrored (not measured)", color = INK2, fontsize = 13)
    text!(ax, xlim - 2, 30; text = "measured", color = INK2, fontsize = 13, align = (:right, :bottom))

    xf = range(-xlim, xlim; length = 600)
    for (i, (s, r, f)) in enumerate(zip(S, R, F))
        lines!(ax, xf, fullprofile(xf, f.p); color = COLORS[i], linewidth = 2,
               label = @sprintf("%s fit:  x₅₀ = %.1f mm,  n = %.2f", s.label, f.p[2], f.p[3]))
        scatter!(ax, s.x, r; color = COLORS[i], marker = MARKERS[i], markersize = 9)
    end
    lines!(ax, xf, fullprofile(xf, G.p); color = :red, linewidth = 1.5, linestyle = :dash,
           label = @sprintf("global fit:  x₅₀ = %.1f mm,  n = %.2f", G.p[2], G.p[3]))
    axislegend(ax; position = :lt, framevisible = false, labelsize = 13)
    xlims!(ax, -xlim, xlim)
    ylims!(ax, 0, 105)
    fig
end

# Full width at half maximum of each fitted profile: twice the 50 % crossing
println("\nSection 6 - full profile (symmetric about x = 0)")
for (s, f) in zip(S, F)
    @printf("  %s fit:   FWHM = %.1f mm\n", s.label, 2 * f.x50pct)
end
@printf("  global fit:    FWHM = %.1f mm\n", 2 * G.x50pct)

fig6 = plot_full_profile(S, R, F, G)
display(fig6)
savefig(fig6, "SG0_fig6_full_profile")
println("\nFigures written to ", abspath(FIGDIR))
