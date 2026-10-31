# =============================================================================
# SG0 field profile along the beam axis - Hall probe data of 2026-10-28 and 2026-10-29
# =============================================================================
#
# CHAPTER 0 - Calibration
#
# Section 0  Zero-current readings: the remanent field B_r(x) of the 0 mA runs of DATAFILE
#            (2026-10-28 and 2026-10-29), linear and log scale.
# Section 0.1  Global half-sigmoid fit B_r = A / (1 + (x/x50)^n) of both 0 A runs (in G).
#
# CHAPTER 1 - SG0 magnetic field profile
#
# Section 1  Raw data: read the Hall-probe readings and plot B(x) for the four
#            nonzero coil currents (linear and log scale).
# Section 2  Fitting: center field B(x = 0) vs. coil current, linear and quadratic fits.
# Section 3  Relative intensity: B(x)/B(0) (1 at the center) for each current.
# Section 4  Half-sigmoid fit of the relative intensity.
# Section 5  Global half-sigmoid fit: one set of parameters for all currents.
# Section 6  Full profile: the fitted functions on both sides of the center (-x and +x).
#
# CHAPTER 2 - Spin-resolved beam separation with the measured SG0 field
#
# Section 7  Separation data: import the spin-resolved separations and plot them vs.
#            SG0 current (linear and log scale).
# Section 8  Current to field: B(x = 0) at every separation current from the CALIBFILE
#            calibration; linear and quadratic calibrations compared (B vs. I);
#            separations plotted vs. the center field.
#
# CHAPTER 3 - CQD fit with the measured SG0 field
#
# Section 9  CQD fit: the double-stage model of the NY script with the SG0 dose taken
#            from the measured field integral instead of the nominal top-hat estimate.
#
# ---------- Data ----------
#
# File: SG0_raw_data_field_profile_2026-10-28_29.csv (DATAFILE, same folder as this script)
#
# One column pair per run, each with its own positions:
#   x_<n>mA_<yyyymmdd>_mm   position along the beam (mm)
#   B_<n>mA_<yyyymmdd>      reading (gauss) at coil current n mA, measured on that date
#
# Runs: 0 mA (2026-10-28, 2026-10-29; Chapter 0), 129 mA (2026-10-29), 443, 735 and
# 1202 mA (2026-10-28; Chapter 1). Steps of 0.635-2.54 mm out to ~77 mm. Empty cell =
# the run has ended. 1 G = 1e-4 T.
#
# Section 8: SG0_calibration_center_BI_2026-10-29.csv (CALIBFILE, same folder)
#
#   I_A   SG0 coil current (A), 0-3.83 A (8 readings, 0 A included)
#   B_G   Hall-probe reading at the field center x = 0 (gauss)
#
# The earlier scan, SG0_field_profile_2026-10-22.{pdf,csv} (0.129, 0.443, 0.963 A, out to
# 55.88 mm), is no longer read; its quoted values are kept below for comparison.
#
# Chapter 2: spin_resolved_sep_20260706.csv (same folder; identical to
# ../fit_20260706/spin_resolved_sep_20260706.csv, the export of spin_resolved_sep_20260706.jld2)
#
#   I0_A                  SG0 coil current (A)
#   dD_parallel_px        separation change, parallel set (pixels)
#   sigma_parallel_px     its 1-sigma uncertainty (pixels)
#   dD_antiparallel_px    separation change, antiparallel set (pixels)
#   sigma_antiparallel_px its 1-sigma uncertainty (pixels)
#
# ---------- Setup (from the 2026-10-22 PDF) ----------
#
# Magnet:  SG0, circular pole faces of 40 mm diameter, pole gap just over 40 mm.
# Probe:   Hall probe on a micrometer stage, 0.635 mm per turn; readings every
#          2.54 mm (four turns); no beam tube installed.
# History: 2026-10-22: magnet not degaussed; all three sets on the rising branch of the
#          same hysteresis loop (after negative current), at positive current.
#          2026-10-28/29: not recorded here. The 0 A center field is +8 G (Chapter 0),
#          opposite in sign to the -20 G of 2026-10-22.
#
# Coordinates: x is the propagation direction, from SG0 toward SG1; x = 0 is the
# field center and only the exit half (x >= 0) was scanned. The horizontal center
# was found in horizontal scans by (1) the geometric center between the pole faces
# and (2) the field minimum / dB/dz ~ 0 (vertex of a parabola fitted to each scan).
# Beyond x ~ 33 mm the horizontal profile is flat and the geometric center was used.
#
# ---------- Values quoted in the 2026-10-22 PDF (for comparison) ----------
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
#   figures/SG0_fig0_zero_current.{pdf,png}         Section 0
#   figures/SG0_fig0b_zero_current_fit.{pdf,png}    Section 0.1
#   figures/SG0_fig1_raw.{pdf,png}                  Section 1
#   figures/SG0_fig1b_raw_minus_zero_current.{pdf,png} Section 1
#   figures/SG0_fig2_center_vs_current.{pdf,png}    Section 2
#   figures/SG0_fig3_relative.{pdf,png}             Section 3
#   figures/SG0_fig4_relative_fit.{pdf,png}         Section 4
#   figures/SG0_fig5_global_fit.{pdf,png}           Section 5
#   figures/SG0_fig6_full_profile.{pdf,png}         Section 6
#   figures/SG0_fig7_separation_vs_current.{pdf,png} Section 7
#   figures/SG0_fig8_calibrations.{pdf,png}          Section 8
#   figures/SG0_fig9_separation_vs_field.{pdf,png}   Section 8
#   figures/SG0_fig10_cqd_fit.{pdf,png}              Section 9
#
# Packages: ] add CSV DataFrames CairoMakie LaTeXStrings LsqFit PrettyTables
# Run in a fresh Julia session (not one where Plots was loaded: it clashes with Makie names).
using CSV, DataFrames, CairoMakie
using LaTeXStrings, LinearAlgebra, LsqFit, Printf, PrettyTables
# Set the working directory to the current location
cd(@__DIR__) ;
include("../../Modules/TheoreticalSimulation.jl");
TheoreticalSimulation.set_magnetic_field!(:calibration)       # back to the default (apparatus manual)
TheoreticalSimulation.magnetic_field_mode()              # which one is active

# Computer Modern (New Computer Modern, bundled with Makie), with minor ticks and
# minor gridlines on every axis (lighter than the major grid)
set_theme!(merge(Theme(Axis = (xminorticksvisible = true, yminorticksvisible = true,
                               xminorgridvisible  = true, yminorgridvisible  = true,
                               xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05))),
                 theme_latexfonts()))

# ---------- Constants and shared helpers ----------

const DATAFILE  = "SG0_raw_data_field_profile_2026-10-28_29.csv"
const CALIBFILE = "SG0_calibration_center_BI_2026-10-29.csv"
const SEPFILE   = "spin_resolved_sep_20260706.csv"      # Chapter 2
const FIGDIR    = "figures"

# One-hue ordinal ramp (light -> dark = low -> high current), validated for a light surface
const COLORS   = [colorant"#86b6ef", colorant"#4a8fe0", colorant"#2a6bc0", colorant"#104281"]
const MARKERS  = [:circle, :rect, :utriangle, :diamond]
const INK2     = colorant"#5f5e5a"                     # secondary ink: fit lines, reference lines

const XLABEL_MM = L"x\ \mathrm{(mm)}"
const POLE_RADIUS_MM = 20.0                            # shaded band x = 0-20 mm in the B(x) plots

savefig(fig, name) = (mkpath(FIGDIR);
                      save(joinpath(FIGDIR, name * ".pdf"), fig);
                      save(joinpath(FIGDIR, name * ".png"), fig; px_per_unit = 3))

# x range rounded up to the next 10 mm, shared by the B(x) plots
x_axis_max(S) = 10 * ceil(maximum(s.x[end] for s in S) / 10)

# Shaded band under the pole face, drawn first so it sits behind the data
pole_band!(ax) = vspan!(ax, 0, POLE_RADIUS_MM; color = (:gray, 0.15))

# Decade tick label: 10^-1, 1, 10, 10^2, 10^3, ...
decade_label(k) = k == 0 ? L"1" : k == 1 ? L"10" : latexstring("10^{$k}")

# Half-sigmoid y = A / (1 + (x/x50)^n), the profile model of Chapters 0 and 1 (Section 4
# explains the choice); p = [A, x50 (mm), n], y in the units of A
halfsigmoid(x, p) = p[1] ./ (1 .+ (x ./ p[2]) .^ p[3])

# Least-squares half-sigmoid fit (all points weighted equally). The defaults suit a relative
# profile (peak 1); pass p0, lower, upper for other units. A = nothing fits A with x50 and n;
# a number fixes A there and fits only x50 and n (its standard error is then 0).
# Returns parameters [A, x50, n], standard errors, R², rms residual (units of y), whether
# A was fitted and, for a relative profile (A > 0.5), the position where the fitted curve
# crosses 0.5 (NaN otherwise).
function fit_halfsigmoid(x, y; p0 = [1.0, 30.0, 2.5],
                         lower = [0.5, 1.0, 1.01], upper = [1.5, 200.0, 10.0], A = nothing)
    if A === nothing
        f  = curve_fit(halfsigmoid, x, y, p0; lower, upper)
        p  = coef(f)
        se = stderror(f)
    else                                                 # A fixed: fit (x50, n) only
        f  = curve_fit((x, q) -> halfsigmoid(x, [A, q[1], q[2]]), x, y, p0[2:3];
                       lower = lower[2:3], upper = upper[2:3])
        p  = [A; coef(f)]
        se = [0.0; stderror(f)]
    end
    r2  = 1 - sum(abs2, f.resid) / sum(abs2, y .- sum(y) / length(y))
    rms = sqrt(sum(abs2, f.resid) / length(f.resid))
    x50pct = p[1] > 0.5 ? p[2] * (2 * p[1] - 1)^(1 / p[3]) : NaN     # where y = 0.5
    (p = p, se = se, r2 = r2, rms = rms, freeA = A === nothing, x50pct = x50pct)
end

# Goodness of a given curve on one data set: R² and rms residual (units of y)
function goodness(x, y, p)
    res = y .- halfsigmoid(x, p)
    (r2 = 1 - sum(abs2, res) / sum(abs2, y .- sum(y) / length(y)),
     rms = sqrt(sum(abs2, res) / length(res)))
end


# =============================================================================
# CHAPTER 0 - Calibration
# =============================================================================

# #############################################################################
# Section 0 - Zero-current readings: the remanent field B_r(x)
# #############################################################################
#
# DATAFILE holds one (x, B) column pair per run, each with its own positions:
#     x_<n>mA_<yyyymmdd>_mm,  B_<n>mA_<yyyymmdd>     (mm, G)
# The runs at 0 mA give the field left in the yoke with the coil off, the offset under
# every profile at nonzero current. Two of them, 2026-10-28 (every 2.54 mm) and
# 2026-10-29 (every 0.635-2.54 mm), both from x = 0.

# Every (x, B) pair at coil current `mA` in `df`, empty cells dropped and sorted by date:
# (date, label, x (mm), B (G))
function runs_at_current(df, mA)
    R = []
    for name in names(df)
        m = match(Regex("^B_$(mA)mA_(\\d{8})\$"), name)
        m === nothing && continue
        d    = m[1]
        xcol = "x_$(mA)mA_$(d)_mm"
        xcol in names(df) || error("$name has no position column $xcol")
        keep = .!ismissing.(df[!, name]) .& .!ismissing.(df[!, xcol])
        push!(R, (date = d, label = "$(d[1:4])-$(d[5:6])-$(d[7:8])",
                  x = Float64.(df[keep, xcol]), B = Float64.(df[keep, name])))
    end
    isempty(R) && error("no B_$(mA)mA_<date> columns in the data file")
    sort!(R; by = r -> r.date)
end

# Zero-current readings, linear (top) and log (bottom) scale, sharing the x axis
function plot_zero_current(Z)
    fig    = Figure(size = (720, 760))
    ylabel = L"B_r\ \mathrm{(G)}"
    xmax   = x_axis_max(Z)
    xticks = 0:10:xmax
    kmin   = floor(Int, log10(minimum(minimum(z.B) for z in Z)))
    kmax   = ceil(Int, log10(maximum(maximum(z.B) for z in Z)))

    ax1 = Axis(fig[1, 1]; ylabel, xticks, title = "Remanent field: Hall-probe readings at 0 A")
    ax2 = Axis(fig[2, 1]; ylabel, xticks, xlabel = XLABEL_MM, yscale = log10,
               yticks = (10.0 .^ (kmin:kmax), decade_label.(kmin:kmax)),
               yminorticks = IntervalsBetween(9))
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid = false, ticks = false, minorgrid = false, minorticks = false)
    xlims!(ax2, 0, xmax)
    ylims!(ax1, 0, nothing)
    ylims!(ax2, 10.0^kmin, 10.0^kmax)

    for ax in (ax1, ax2)
        pole_band!(ax)
    end
    for (i, z) in enumerate(Z), ax in (ax1, ax2)
        scatterlines!(ax, z.x, z.B; color = COLORS[2i], marker = MARKERS[i],
                      markersize = 9, linewidth = 2, label = "0 A, " * z.label)
    end
    axislegend(ax1; position = :rt, framevisible = false)
    fig
end

df_raw = CSV.read(DATAFILE, DataFrame)
Z0     = runs_at_current(df_raw, 0)

println("Section 0 - zero-current readings (", DATAFILE, ")")
for z in Z0
    @printf("  %s: %2d readings, x = %.2f-%.2f mm, B = %.1f G at x = %.2f mm, %.1f G at x = %.2f mm\n",
            z.label, length(z.x), z.x[1], z.x[end], z.B[1], z.x[1], z.B[end], z.x[end])
end

fig0 = plot_zero_current(Z0)
display(fig0)
savefig(fig0, "SG0_fig0_zero_current")


# #############################################################################
# Section 0.1 - Global half-sigmoid fit of the zero-current readings
# #############################################################################
#
# One function for both 0 A runs: y = A / (1 + (x/x50)^n), the model of Section 4,
# fitted to the pooled readings in gauss (every point weighted equally). The fit is done
# on B itself, not on B/B(0), so no single reading sets the normalization. A is then the
# remanent center field (G); x50 and n describe the shape and compare directly with the
# coil-field profile of Section 5. The per-run R² and rms show how well the common curve
# describes each day.

# Readings of every run (markers) with the global curve (line) on top; relative residuals
# (data - fit)/data (%) of each run below
function plot_zero_current_fit(Z, G0, Gz)
    fig  = Figure(size = (720, 700))
    xmax = x_axis_max(Z)
    ax1  = Axis(fig[1, 1]; ylabel = L"B_r\ \mathrm{(G)}", xticks = 0:10:xmax,
                title = "Remanent field at 0 A: global half-sigmoid fit  y = A / [1 + (x/x₅₀)ⁿ]")
    ax2  = Axis(fig[2, 1]; ylabel = L"(\mathrm{data - fit})/\mathrm{data}\ (\%)", xticks = 0:10:xmax,
                xlabel = XLABEL_MM)
    rowsize!(fig.layout, 2, Relative(0.3))
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1; grid = false, ticks = false, minorgrid = false, minorticks = false)

    for ax in (ax1, ax2)
        pole_band!(ax)
    end
    hlines!(ax2, 0; color = INK2, linewidth = 1)
    xf = range(0, xmax; length = 300)
    lines!(ax1, xf, halfsigmoid(xf, G0.p); color = :black, linewidth = 2,
           label = @sprintf("A = %.2f G,  x₅₀ = %.1f mm,  n = %.2f,  R² = %.4f",
                            G0.p[1], G0.p[2], G0.p[3], G0.r2))
    for (i, (z, g)) in enumerate(zip(Z, Gz))
        scatter!(ax1, z.x, z.B; color = COLORS[2i], marker = MARKERS[i], markersize = 9,
                 label = @sprintf("0 A, %s:  rms = %.2f G", z.label, g.rms))
        scatterlines!(ax2, z.x, 100 .* (z.B .- halfsigmoid(z.x, G0.p)) ./ z.B; color = COLORS[2i],
                      marker = MARKERS[i], markersize = 7, linewidth = 1)
    end
    axislegend(ax1; position = :rt, framevisible = false)
    xlims!(ax2, 0, xmax)
    ylims!(ax1, 0, nothing)
    fig
end

# Bounds in gauss: A up to 100 G (the readings are ~8 G at the center)
G0 = fit_halfsigmoid(vcat((z.x for z in Z0)...), vcat((z.B for z in Z0)...);
                     p0 = [8.0, 30.0, 2.5], lower = [0.0, 1.0, 1.01], upper = [100.0, 200.0, 10.0])
Gz = [goodness(z.x, z.B, G0.p) for z in Z0]

# Zero-current (remanent) field B_r(x) in gauss at position x (mm), from the global fit G0.
# Evaluated at |x|: symmetric about the center, like the coil profile (Section 6); the
# entrance half x < 0 is the mirror image, not a measurement. Works on a number or a vector.
zero_current_calib(x) = halfsigmoid(abs.(x), G0.p)

println("\nSection 0.1 - global half-sigmoid fit of the zero-current readings")
println("  B(x) = A / (1 + (|x|/x50)^n)")
@printf("  A = %.3f ± %.3f G,  x50 = %.2f ± %.2f mm,  n = %.3f ± %.3f\n",
        G0.p[1], G0.se[1], G0.p[2], G0.se[2], G0.p[3], G0.se[3])
@printf("  pooled: R² = %.5f,  rms = %.3f G (%d points)\n",
        G0.r2, G0.rms, sum(length(z.x) for z in Z0))
for (z, g) in zip(Z0, Gz)
    @printf("  %s:  R² = %.5f,  rms = %.3f G\n", z.label, g.r2, g.rms)
end
@printf("  zero_current_calib(x):  %.2f G at 0 mm,  %.2f G at ±20 mm,  %.2f G at ±50 mm\n",
        zero_current_calib(0.0), zero_current_calib(20.0), zero_current_calib(50.0))

fig0b =plot_zero_current_fit(Z0, G0, Gz)
display(fig0b)
savefig(fig0b, "SG0_fig0b_zero_current_fit")


# =============================================================================
# CHAPTER 1 - SG0 magnetic field profile (Hall probe, 2026-10-28 and 2026-10-29)
# =============================================================================


# #############################################################################
# Section 1 - Raw data: extraction and plotting
# #############################################################################

# Coil currents (mA) of the nonzero-current runs in the CSV header (columns
# "B_<n>mA_<yyyymmdd>", n > 0; the 0 mA runs are Chapter 0), low to high
function profile_currents(df)
    mA = sort(unique(parse(Int, m[1]) for name in names(df)
                     for m in (match(r"^B_(\d+)mA_\d{8}$", name),) if m !== nothing))
    filter!(>(0), mA)
    isempty(mA) && error("no B_<n>mA_<date> columns with n > 0 in the data file")
    mA
end

# One profile per nonzero current: (I (A), label, date, x (mm), B (G)), empty cells dropped.
# A current measured on more than one date would need its own color and label: refused here.
function load_profiles(df)
    S = map(profile_currents(df)) do mA
        R = runs_at_current(df, mA)
        length(R) == 1 || error("$mA mA measured on $(length(R)) dates: pick one")
        I = mA / 1000
        (I = I, label = @sprintf("%.3f A", I), date = R[1].label, x = R[1].x, B = R[1].B)
    end
    length(S) > length(COLORS) &&
        error("$(length(S)) currents but only $(length(COLORS)) colors: extend COLORS and MARKERS")
    S
end

# Raw readings, linear (top) and log (bottom) scale, sharing the x axis.
# The log panel keeps the lowest current readable next to the highest; its
# y range spans the whole decades that contain the data (all B > 0).
function plot_raw(S; title = "Raw readings", ylabel = L"B\ \mathrm{(G)}")
    fig = Figure(size = (720, 760))
    xmax   = x_axis_max(S)
    xticks = 0:10:xmax
    kmin   = floor(Int, log10(minimum(minimum(s.B) for s in S)))
    kmax   = ceil(Int, log10(maximum(maximum(s.B) for s in S)))

    ax1 = Axis(fig[1, 1]; ylabel, xticks, title)
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
                      markersize = 9, linewidth = 2, label = "$(s.label), $(s.date)")
    end
    axislegend(ax1; position = :rt, framevisible = false)
    fig
end

S        = load_profiles(df_raw)
currents = [s.I for s in S]

println("\nSection 1 - raw data")
for s in S
    @printf("  %s (%s): %2d readings, x = %.2f-%.2f mm, B(0) = %.1f G\n",
            s.label, s.date, length(s.x), s.x[1], s.x[end], s.B[1])
end

fig1 = plot_raw(S)
display(fig1)
savefig(fig1, "SG0_fig1_raw")

# Readings minus the zero-current field: B(x) - B_r(x), with B_r = zero_current_calib
# (Section 0.1) at each reading's position. This removes the field left at 0 A on the
# assumption that it simply adds to the coil field; with hysteresis the remanent part at
# nonzero current need not equal the 0 A value, so this is a first-order correction.
S_sub = [(; s..., B = s.B .- zero_current_calib(s.x)) for s in S];

println("  minus the zero-current field B_r(x) (Section 0.1):")
for (s, t) in zip(S, S_sub)
    @printf("  %s (%s): B - B_r = %.1f G at x = 0 (B_r = %.1f %% of B);  %.1f G at x = %.2f mm (%.1f %%)\n",
            s.label, s.date, t.B[1], 100 * (1 - t.B[1] / s.B[1]),
            t.B[end], s.x[end], 100 * (1 - t.B[end] / s.B[end]))
end

fig1b = plot_raw(S_sub; title = "Readings minus the zero-current field",
                 ylabel = L"B - B_r\ \mathrm{(G)}")
display(fig1b)
savefig(fig1b, "SG0_fig1b_raw_minus_zero_current")

# From here on (Sections 2-6, and through them Chapter 3) the profiles are the coil field,
# B - B_r. The raw readings stay in S_raw: Section 8 compares their centers with CALIBFILE,
# which is raw as well.
S_raw = S;
S     = S_sub;

# Table of the coil field B - B_r (G) at every position read in any run, with the
# subtracted zero-current field B_r(x) beside x; blank = no reading of that run there
function table_minus_zero_current(S)
    xs   = sort(unique(vcat((s.x for s in S)...)))
    rows = map(xs) do x
        cells = map(S) do s
            j = findfirst(==(x), s.x)
            j === nothing ? "" : @sprintf("%.1f", s.B[j])
        end
        Any[@sprintf("%.3f", x), @sprintf("%.2f", zero_current_calib(x)), cells...]
    end
    pretty_table(permutedims(reduce(hcat, rows));
                 title = "Section 1 - field after subtracting the zero-current field, B - B_r (G)",
                 column_labels = [["x", "B_r", (s.label for s in S)...],
                                  ["(mm)", "(G)", (s.date for s in S)...]],
                 alignment = :r,
                 fit_table_in_display_vertically = false,     # print every row, not
                 fit_table_in_display_horizontally = false)   # just what fits the terminal
end

println()
table_minus_zero_current(S)


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

# Center-field calibration readings: (I (A), B (G)), low to high current
function load_calibration(file = CALIBFILE)
    df = sort(CSV.read(file, DataFrame), :I_A)
    (I = Float64.(df.I_A), B = Float64.(df.B_G))
end

# Linear (solid) and quadratic (dashed) fit lines into `ax` over 0..Imax, labeled with
# their coefficients and R²
function fit_lines!(ax, lin, quad, Imax)
    If = range(0, Imax; length = 300)
    hlines!(ax, 0; color = INK2, linewidth = 1)
    lines!(ax, If, polyval(lin.c, If); color = INK2, linewidth = 2,
           label = @sprintf("linear: %.1f + %.1f I,  R² = %.5f", lin.c..., lin.r2))
    lines!(ax, If, polyval(quad.c, If); color = INK2, linewidth = 1.5, linestyle = :dash,
           label = @sprintf("quadratic: %.1f + %.1f I %+.2f I²,  R² = %.5f", quad.c..., quad.r2))
end

# Center field vs. current, two independent calibrations in a column:
# top    - profile-run centers B - B_r (colored) with their linear and quadratic fits, and
#          the 0 A center reading of Chapter 0 minus B_r (`B0c`, measured on `date0`, not fitted);
# bottom - CALIBFILE readings (raw, measured on `caldate`) with their own linear and
#          quadratic fits. Each panel spans its own current range.
function plot_center(Ic, Bc, lin, quad, B0c, date0, cal, cal_lin, cal_quad, caldate)
    fig = Figure(size = (720, 900))
    ylabel = L"B(x = 0)\ \mathrm{(G)}"

    Imax1 = 0.1 * ceil(10 * maximum(Ic))
    ax1   = Axis(fig[1, 1]; xticks = 0:0.1:Imax1, xlabel = L"I\ \mathrm{(A)}",
                 ylabel = L"(B - B_r)(x = 0)\ \mathrm{(G)}",
                 title = "Profile runs, zero-current field subtracted")
    fit_lines!(ax1, lin, quad, Imax1)
    scatter!(ax1, [0.0], [B0c]; marker = :star5, markersize = 13,
             color = :transparent, strokecolor = INK2, strokewidth = 1.5,
             label = "0 A, measured $date0, minus B_r (not fitted)")
    for i in eachindex(Ic)
        scatter!(ax1, [Ic[i]], [Bc[i]]; color = COLORS[i], marker = MARKERS[i], markersize = 13,
                 label = @sprintf("%.3f A", Ic[i]))
    end
    axislegend(ax1; position = :lt, framevisible = false)
    xlims!(ax1, -0.03, Imax1)  # small margin so the 0 A marker is not clipped

    Imax2 = 0.5 * ceil(2 * maximum(cal.I))
    ax2   = Axis(fig[2, 1]; xticks = 0:0.5:Imax2, xlabel = L"I\ \mathrm{(A)}", ylabel,
                 title = "Center calibration file ($caldate), raw")
    fit_lines!(ax2, cal_lin, cal_quad, Imax2)
    f = cal.I .> 0                                      # the 0 A reading is not fitted
    scatter!(ax2, cal.I[f], cal.B[f]; marker = :star5, color = :black, markersize = 13,
             label = "CALIBFILE readings (fitted)")
    scatter!(ax2, cal.I[.!f], cal.B[.!f]; marker = :star5, markersize = 13,
             color = :transparent, strokecolor = :black, strokewidth = 1.5,
             label = "0 A reading (not fitted)")
    axislegend(ax2; position = :lt, framevisible = false)
    xlims!(ax2, -0.1, Imax2)
    fig
end

# Fitted to the center values of the coil field B - B_r (Section 1). The linear model
# B = Br + k I is the physical default below saturation; Br is now what is left of the
# remanence after subtracting the 0 A field (ideally 0). The quadratic term measures the
# curvature the data allow (with four currents: 2 and 1 dof).
# The 0 A center reading (Chapter 0) minus B_r(0) is shown but not fitted: it is the
# residual of the Section 0.1 fit at the center, the level an exact subtraction gives.
Ic, Bc = center_field(S)
lin    = polyfit(Ic, Bc, 1)
quad   = polyfit(Ic, Bc, 2)
z0c    = Z0[findfirst(z -> z.x[1] == 0, Z0)]     # zero-current run with a reading at x = 0
B0c    = z0c.B[1] - zero_current_calib(0.0)

println("\nSection 2 - center field B(x = 0) vs. current")
for (I, B, r) in zip(Ic, Bc, lin.res)
    @printf("  I = %.3f A   B = %6.1f G   linear residual = %+5.2f G\n", I, B, r)
end
@printf("  linear:    B = Br + k I,  Br = %.1f ± %.1f G,  k = %.1f ± %.1f G/A,  R² = %.5f (dof = %d)\n",
        lin.c[1], lin.se[1], lin.c[2], lin.se[2], lin.r2, lin.dof)
@printf("             B = 0 at I = %.1f mA;  B(0 A) = %.1f G vs. %.1f G for the 0 A reading minus B_r (%s)\n",
        -1e3 * lin.c[1] / lin.c[2], lin.c[1], B0c, z0c.label)
@printf("  quadratic: B = %.1f + %.1f I %+.1f I²,  se = (%.1f, %.1f, %.1f),  R² = %.5f (dof = %d)\n",
        quad.c..., quad.se..., quad.r2, quad.dof)
@printf("             B(0 A) = %.1f G\n", quad.c[1])

# Independent calibration from CALIBFILE: center readings at 0-3.83 A (2026-10-29), raw
# (B_r not subtracted), linear and quadratic fits to the readings at I > 0. The 0 A reading
# is left out: it is the field left in the iron, not a point of the coil response, and
# with the coil on that remanent part need not be the same. It is still listed and plotted.
# These are the fits Section 8 and Chapter 3 use: they cover every separation current.
cal      = load_calibration()
caldate  = something(match(r"\d{4}-\d{2}-\d{2}", CALIBFILE), (match = "",)).match
calfit   = cal.I .> 0                                # readings used in the fits
cal_lin  = polyfit(cal.I[calfit], cal.B[calfit], 1)
cal_quad = polyfit(cal.I[calfit], cal.B[calfit], 2)

println("  calibration file ", CALIBFILE, ": ", length(cal.I), " readings, I = ",
        @sprintf("%.3f-%.3f A", extrema(cal.I)...), " (raw); fitted: the ",
        count(calfit), " at I > 0")
@printf("  linear:    B = %.1f ± %.1f + (%.1f ± %.1f) I (G),  R² = %.6f (dof = %d)\n",
        cal_lin.c[1], cal_lin.se[1], cal_lin.c[2], cal_lin.se[2], cal_lin.r2, cal_lin.dof)
@printf("  quadratic: B = %.1f + %.1f I %+.2f I² (G),  se = (%.1f, %.1f, %.2f),  R² = %.6f (dof = %d)\n",
        cal_quad.c..., cal_quad.se..., cal_quad.r2, cal_quad.dof)
@printf("  %8s  %8s  %13s  %13s\n", "I (A)", "B (G)", "lin res (G)", "quad res (G)")
for (I, B, used) in zip(cal.I, cal.B, calfit)
    @printf("  %8.3f  %8.1f  %+13.2f  %+13.2f%s\n", I, B,
            B - polyval(cal_lin.c, I), B - polyval(cal_quad.c, I), used ? "" : "   (not fitted)")
end

fig2 =plot_center(Ic, Bc, lin, quad, B0c, z0c.label, cal, cal_lin, cal_quad, caldate)
display(fig2)
savefig(fig2, "SG0_fig2_center_vs_current")


# #############################################################################
# Section 3 - Relative intensity B(x)/B(0)
# #############################################################################

# Relative intensity B(x)/B(x = 0) (1 at the center), per current, with B the coil field
# B - B_r of Section 1 (S holds the subtracted profiles from there on).
# Each set is normalized to its own center reading, so the profile shapes can be
# compared across currents (as in the PDF table "Normalized field B/B(0) vs. position").
relative(s) = s.B ./ s.B[1]

# Position where the relative profile first drops to `level` (relative), by linear
# interpolation between neighboring readings; NaN if it never gets there.
function crossing(x, y, level)
    i = findfirst(j -> y[j] >= level > y[j+1], 1:length(y)-1)
    i === nothing && return NaN
    x[i] + (level - y[i]) * (x[i+1] - x[i]) / (y[i+1] - y[i])
end

# Relative intensity vs. x, one curve per current, with the 0.5 level marked.
function plot_relative(S, R)
    fig  = Figure(size = (720, 480))
    xmax = x_axis_max(S)
    ax   = Axis(fig[1, 1]; xticks = 0:10:xmax, yticks = 0:0.2:1, xlabel = XLABEL_MM,
                ylabel = L"(B - B_r)/(B - B_r)(0)",
                title = "Relative intensity along the beam axis (zero-current field subtracted)")
    pole_band!(ax)
    hlines!(ax, 0.5; color = INK2, linewidth = 1, linestyle = :dash)
    for (i, (s, r)) in enumerate(zip(S, R))
        scatterlines!(ax, s.x, r; color = COLORS[i], marker = MARKERS[i],
                      markersize = 9, linewidth = 2, label = s.label)
    end
    axislegend(ax; position = :rt, framevisible = false)
    xlims!(ax, 0, xmax)
    ylims!(ax, 0, 1.05)
    fig
end

R = [relative(s) for s in S]

# Table in the layout of the PDF: one row per position, blank where there is no reading
println("\nSection 3 - relative intensity (B - B_r)/(B - B_r)(0), zero-current field subtracted")
@printf("  %7s", "x (mm)"); foreach(s -> @printf("  %9s", s.label), S); println()
for x in sort(unique(vcat((s.x for s in S)...)))     # every position read in any set
    @printf("  %7.2f", x)
    for (s, r) in zip(S, R)
        j = findfirst(==(x), s.x)
        j === nothing ? @printf("  %9s", "") : @printf("  %9.4f", r[j])
    end
    println()
end
for (s, r) in zip(S, R)
    @printf("  %s: (B - B_r)/(B - B_r)(0) = 0.5 at x = %.2f mm\n", s.label, crossing(s.x, r, 0.5))
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
#     y(x) = A / (1 + (x/x50)^n)        y relative (1 at the center), x in mm, x >= 0
#
# - read as a function of |x| it is the right half of a symmetric bell: even in x,
#   flat at the center (dy/dx = 0 at x = 0 for n > 1), as the field of a symmetric
#   magnet must be
# - y(x50) = A/2: x50 is the half-amplitude point; n sets the steepness of the edge
# - tail ~ x^-n at large x (n -> 3 would be the far field of a magnetic dipole)
# - A is left free (close to 1) so the center reading alone does not pin the fit
#
# The plain logistic 1/(1 + exp((x - x0)/w)) was also tried: it needs A ~ 1.15 and has
# a nonzero slope at x = 0, and fits worse (rms 0.012-0.017), so it is not used.

# `halfsigmoid` and `fit_halfsigmoid` are defined with the shared helpers (also used in Chapter 0).

# Relative intensity (markers) with the fitted half-sigmoids (lines)
function plot_relative_fit(S, R, F)
    fig  = Figure(size = (720, 480))
    xmax = x_axis_max(S)
    ax   = Axis(fig[1, 1]; xticks = 0:10:xmax, yticks = 0:0.2:1, xlabel = XLABEL_MM,
                ylabel = L"(B - B_r)/(B - B_r)(0)",
                title = "Half-sigmoid fit:  y = A / [1 + (x/x₅₀)ⁿ]")
    pole_band!(ax)
    hlines!(ax, 0.5; color = INK2, linewidth = 1, linestyle = :dash)
    xf = range(0, xmax; length = 300)
    for (i, (s, r, f)) in enumerate(zip(S, R, F))
        lines!(ax, xf, halfsigmoid(xf, f.p); color = COLORS[i], linewidth = 2,
               label = @sprintf("%s:  x₅₀ = %.1f mm,  n = %.2f,  R² = %.4f", s.label, f.p[2], f.p[3], f.r2))
        scatter!(ax, s.x, r; color = COLORS[i], marker = MARKERS[i], markersize = 9)
    end
    axislegend(ax; position = :rt, framevisible = false)
    xlims!(ax, 0, xmax)
    ylims!(ax, 0, 1.05)
    fig
end

F = [fit_halfsigmoid(s.x, r) for (s, r) in zip(S, R)]

println("\nSection 4 - half-sigmoid fit  y = A / (1 + (x/x50)^n)")
for (s, f) in zip(S, F)
    @printf("  %s:  A = %.4f ± %.4f,  x50 = %.2f ± %.2f mm,  n = %.3f ± %.3f,  R² = %.5f,  rms = %.4f,  fit = 0.5 at x = %.2f mm\n",
            s.label, f.p[1], f.se[1], f.p[2], f.se[2], f.p[3], f.se[3], f.r2, f.rms, f.x50pct)
end

fig4 = plot_relative_fit(S, R, F)
display(fig4)
savefig(fig4, "SG0_fig4_relative_fit")


# #############################################################################
# Section 5 - Global half-sigmoid fit: one function for all currents
# #############################################################################
#
# Same model as Section 4, y = A / (1 + (x/x50)^n), but a single (A, x50, n) fitted to
# the relative intensity of all currents at once: least squares on the pooled points,
# every reading weighted equally (so sets with more readings count more: 0.129 A has 68,
# the others 55-56). The per-set R² and rms show how well the common curve describes
# each current.

# Relative intensity of every current (markers) with the single global curve (line)
function plot_global_fit(S, R, G, Gset)
    fig  = Figure(size = (720, 480))
    xmax = x_axis_max(S)
    ax   = Axis(fig[1, 1]; xticks = 0:10:xmax, yticks = 0:0.2:1, xlabel = XLABEL_MM,
                ylabel = L"(B - B_r)/(B - B_r)(0)",
                title = "Global half-sigmoid fit (all currents):  y = A / [1 + (x/x₅₀)ⁿ]")
    pole_band!(ax)
    hlines!(ax, 0.5; color = INK2, linewidth = 1, linestyle = :dash)
    xf = range(0, xmax; length = 300)
    lines!(ax, xf, halfsigmoid(xf, G.p); color = :black, linewidth = 2,
           label = @sprintf("A = %.3f,  x₅₀ = %.2f mm,  n = %.3f,  R² = %.4f",
                            G.p[1], G.p[2], G.p[3], G.r2))
    for (i, (s, r, g)) in enumerate(zip(S, R, Gset))
        scatter!(ax, s.x, r; color = COLORS[i], marker = MARKERS[i], markersize = 9,
                 label = @sprintf("%s:  R² = %.4f,  rms = %.4f", s.label, g.r2, g.rms))
    end
    axislegend(ax; position = :rt, framevisible = false)
    xlims!(ax, 0, xmax)
    ylims!(ax, 0, 1.05)
    fig
end

# Switch for the amplitude of the global fit:
#   true  - A is a free parameter (fitted with x50 and n)
#   false - A fixed at 1: the curve is exactly 1 at x = 0, as the normalization
#           y = B/B(0) demands; only x50 and n are fitted
# Everything downstream (Section 6, L_eff and the CQD fit of Chapter 3) uses this fit.
const GLOBAL_FIT_FREE_A = false

G    = fit_halfsigmoid(vcat((s.x for s in S)...), vcat(R...);
                       A = GLOBAL_FIT_FREE_A ? nothing : 1.0)
Gset = [goodness(s.x, r, G.p) for (s, r) in zip(S, R)]

println("\nSection 5 - global half-sigmoid fit (all currents together), A ",
        G.freeA ? "fitted" : "fixed at 1")
println("  y = A / (1 + (x/x50)^n)")
@printf("  A = %.4f ± %.4f%s,  x50 = %.2f ± %.2f mm,  n = %.3f ± %.3f\n",
        G.p[1], G.se[1], G.freeA ? "" : " (fixed)", G.p[2], G.se[2], G.p[3], G.se[3])
@printf("  pooled: R² = %.5f,  rms = %.4f (%d points);  fit = 0.5 at x = %.2f mm\n",
        G.r2, G.rms, sum(length(s.x) for s in S), G.x50pct)
@printf("  shape vs. the remanent field at 0 A (Section 0.1):  x50 = %.2f vs. %.2f mm,  n = %.3f vs. %.3f\n",
        G.p[2], G0.p[2], G.p[3], G0.p[3])
for (s, g, f) in zip(S, Gset, F)
    @printf("  %s:  R² = %.5f,  rms = %.4f   (own fit, Section 4: rms = %.4f)\n",
            s.label, g.r2, g.rms, f.rms)
end

fig5 = plot_global_fit(S, R, G, Gset)
display(fig5)
savefig(fig5, "SG0_fig5_global_fit")

# Normalized coil-field profile h(x) from the global fit G, with A = 1 (so h(0) = 1):
#     h(x) = 1 / (1 + (|x|/x50)^n),    x in mm, x50 and n from Section 5.
# Multiplied by the center field Bc(I) it gives the coil field along the beam axis,
# B(x, I) = Bc(I) h(x). Evaluated at |x|: symmetric about the center (Section 6); the
# entrance half x < 0 is the mirror image, not a measurement. Works on a number or a vector.
field_to_current_calibration(x) = halfsigmoid(abs.(x), [1.0, G.p[2], G.p[3]])

println("  field_to_current_calibration(x) = 1 / (1 + (|x|/x50)^n)")
@printf("    x50 = %.2f mm,  n = %.3f  (Section 5; A set to 1)\n", G.p[2], G.p[3])
@printf("    %.3f at 0 mm,  %.3f at ±20 mm,  %.3f at ±x50,  %.3f at ±50 mm\n",
        field_to_current_calibration(0.0), field_to_current_calibration(20.0),
        field_to_current_calibration(G.p[2]), field_to_current_calibration(50.0))


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
    ax   = Axis(fig[1, 1]; xticks = -xlim:10:xlim, yticks = 0:0.2:1, xlabel = XLABEL_MM,
                ylabel = L"(B - B_r)/(B - B_r)(0)",
                title = "Full field profile: half-sigmoid fits mirrored about x = 0")
    vspan!(ax, -POLE_RADIUS_MM, POLE_RADIUS_MM; color = (:gray, 0.15))   # pole face
    hlines!(ax, 0.5; color = INK2, linewidth = 1, linestyle = :dash)
    vlines!(ax, 0; color = INK2, linewidth = 1)
    # Side labels placed above the curve tails (which are below ~15 % near the edges)
    text!(ax, -xlim + 2, 0.3; text = "mirrored (not measured)", color = INK2, fontsize = 13)
    text!(ax, xlim - 2, 0.3; text = "measured", color = INK2, fontsize = 13, align = (:right, :bottom))

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
    ylims!(ax, 0, 1.05)
    fig
end

# Full width at half maximum of each fitted profile: twice the 0.5 crossing
println("\nSection 6 - full profile (symmetric about x = 0)")
for (s, f) in zip(S, F)
    @printf("  %s fit:   FWHM = %.1f mm\n", s.label, 2 * f.x50pct)
end
@printf("  global fit:    FWHM = %.1f mm\n", 2 * G.x50pct)

fig6 = plot_full_profile(S, R, F, G)
display(fig6)
savefig(fig6, "SG0_fig6_full_profile")


# =============================================================================
# CHAPTER 2 - Spin-resolved beam separation with the measured SG0 field
# =============================================================================


# #############################################################################
# Section 7 - Separation data: import
# #############################################################################
#
# Spin-resolved change of the beam separation vs. SG0 coil current, the data fitted by
# the CQD model in ../SG0_plotting_oct15_NYconference.jl. Same curve shape as there:
# a Vector of (label, I, y, yerror), parallel set first. The rows of the file are not in
# current order (1.404 A before 1.403 A, 3.642 A before 3.598 A), so they are sorted here.

function load_separation(file = SEPFILE)
    df = sort(CSV.read(file, DataFrame), :I0_A)
    [(label = "↑↑ (parallel)",     I = df.I0_A, y = df.dD_parallel_px,     yerror = df.sigma_parallel_px),
     (label = "↓↑ (antiparallel)", I = df.I0_A, y = df.dD_antiparallel_px, yerror = df.sigma_antiparallel_px)]
end

curves = load_separation()

println("\nSection 7 - spin-resolved separation data (", SEPFILE, ")")
for c in curves
    @printf("  %-18s %2d points,  I = %.3f-%.3f A,  ΔD = %+.2f to %+.2f px,  σ = %.2f-%.2f px\n",
            c.label, length(c.I), extrema(c.I)..., extrema(c.y)..., extrema(c.yerror)...)
end

# Same colors and markers as the separation plots of ../SG0_plotting_oct15_NYconference.jl
const SEP_COLORS  = [:red, :blue]
const SEP_MARKERS = [:diamond, :rect]
const SEP_COLORS_QUAD = [:darkorange, :teal]            # Fig. 9, quadratic-calibration field
const SEP_YLABEL  = "Relative beam separation (pixels)"
const SEP_YLIMS   = (-3, 6)                        # fits every point with its error bar

# Separations of every set into `ax` at abscissae `x` (one value per point, shared by
# the sets): vertical error bars and open markers, one shape per set unless `marker` sets
# a common one, in `colors` (one per set); `mask` selects the points drawn.
# Returns the marker plots, one per set, for a custom legend.
function separation!(ax, curves, x; marker = nothing, colors = SEP_COLORS,
                     mask = trues(length(x)))
    hlines!(ax, 0; color = INK2, linewidth = 1, linestyle = :dash)
    ylims!(ax, SEP_YLIMS...)
    map(enumerate(curves)) do (i, c)
        errorbars!(ax, x[mask], c.y[mask], c.yerror[mask]; color = (colors[i], 0.7),
                   whiskerwidth = 8, linewidth = 1.5)
        scatter!(ax, x[mask], c.y[mask]; color = :transparent,
                 marker = something(marker, SEP_MARKERS[i]), strokecolor = colors[i],
                 strokewidth = 2, markersize = 12, label = c.label)
    end
end

# Separations vs. SG0 current, the measured data only: linear axis (top) and log axis
# (bottom, I > 0 only), which spreads out the four points below 50 mA that pile up at the
# origin of the linear panel.
function plot_sep_vs_current(curves)
    fig  = Figure(size = (900, 900))
    I    = curves[1].I
    pos  = I .> 0
    kmin = floor(Int, log10(minimum(I[pos])))
    kmax = ceil(Int, log10(maximum(I)))

    ax1 = Axis(fig[1, 1]; ylabel = SEP_YLABEL, xlabel = L"SG0 current $\mathrm{(A)}$", xticks = 0:0.5:4,
               yticks = SEP_YLIMS[1]:SEP_YLIMS[2], 
               title = "Spin-resolved separation vs. SG0 current")
    separation!(ax1, curves, I)
    axislegend(ax1; position = :lt, framevisible = false)
    xlims!(ax1, -0.1, 3.8)

    ax2 = Axis(fig[2, 1]; ylabel = SEP_YLABEL, xlabel = L"SG0 current $\mathrm{(A)}$", xscale = log10,
               xticks = (10.0 .^ (kmin:kmax), decade_label.(kmin:kmax)),
               xminorticks = IntervalsBetween(9),
               yticks = SEP_YLIMS[1]:SEP_YLIMS[2],)
    separation!(ax2, curves, I; mask = pos)
    xlims!(ax2, 10.0^kmin, 10.0^kmax)
    fig
end

fig7 = plot_sep_vs_current(curves)
display(fig7)
savefig(fig7, "SG0_fig7_separation_vs_current")


# #############################################################################
# Section 8 - Current to field: B(x = 0) at the separation currents
# #############################################################################
#
# Center-field calibration from CALIBFILE: Hall-probe readings at x = 0 for currents of
# 0-3.83 A (2026-10-29), covering every separation current (0-3.64 A), so the field is
# interpolated, not extrapolated. Linear B = Br + k I and quadratic fits to all its points,
# the 0 A reading included; their spread is a systematic range, not a statistical error.
# The raw x = 0 readings of the profile runs (Section 1, 0.129-1.202 A; not B - B_r, since
# CALIBFILE is raw too) are compared with it.
# Caveat:
# - remanence: the zero-current center field was +8 G on 2026-10-29 (Chapter 0) but
#   -20 G on 2026-10-22: it follows the magnetic history, which is not recorded for the
#   separation runs. Below ~50 mA (B < 25 G) the remanence is a large fraction of the
#   field, and its size and sign there are uncertain.


# Shaded band of the field range covered by the calibration, labeled at the bottom
function mapped_band!(ax, lo, hi)
    vspan!(ax, lo, hi; color = (:gray, 0.15))
    text!(ax, (lo + hi) / 2, SEP_YLIMS[1] + 0.3; text = "calibrated", color = INK2,
          fontsize = 13, align = (:center, :bottom))
end

# Separations vs. center field, each point plotted twice: circles (red/blue) at the field
# from the linear calibration, squares (orange/teal) at the field from the quadratic one (the same
# measurement, only the field assigned to it differs; the vertical bars are the 1-sigma
# errors of the separation). The shaded band is the field range covered by the calibration.
function plot_sep_vs_field(curves, Blin, Bquad, Bcal)
    fig = Figure(size = (900, 560))
    ax  = Axis(fig[1, 1]; xlabel = L"B(x = 0)\ \mathrm{(G)}", ylabel = SEP_YLABEL,
               title = "Spin-resolved separation vs. SG0 center field")
    mapped_band!(ax, Bcal...)
    hl = separation!(ax, curves, Blin; marker = :circle)
    hq = separation!(ax, curves, Bquad; marker = :rect, colors = SEP_COLORS_QUAD)
    labels = [c.label for c in curves]
    axislegend(ax, [hl, hq], [labels, labels], ["linear calibration", "quadratic calibration"];
               position = :lt, framevisible = false, orientation = :horizontal,
               titleposition = :top, nbanks = 2)
    fig
end

# cal, cal_lin, cal_quad: the CALIBFILE readings and fits of Section 2
Isep  = curves[1].I
Blin  = polyval(cal_lin.c, Isep)
Bquad = polyval(cal_quad.c, Isep)
Bcal  = polyval(cal_lin.c, collect(extrema(cal.I)))     # calibrated field range (G)

println("\nSection 8 - center-field calibration (", CALIBFILE, ")")
@printf("  %d readings, I = %.3f-%.3f A;  %d of %d separation currents lie above %.3f A\n",
        length(cal.I), extrema(cal.I)..., count(>(maximum(cal.I)), Isep), length(Isep),
        maximum(cal.I))
println("  fits: Section 2 (linear and quadratic, raw CALIBFILE)")
Bc_raw = [s.B[1] for s in S_raw]                     # raw center readings of the profile runs
println("  profile runs at x = 0 (raw, Section 1) vs. the calibration:")
@printf("  %8s  %8s  %13s  %13s\n", "I (A)", "B (G)", "lin res (G)", "quad res (G)")
for (I, B) in zip(Ic, Bc_raw)
    @printf("  %8.3f  %8.1f  %+13.2f  %+13.2f\n", I, B,
            B - polyval(cal_lin.c, I), B - polyval(cal_quad.c, I))
end

println("\nSection 8 - center field at the separation currents")
@printf("  %8s  %10s  %10s  %9s  %s\n", "I (A)", "B lin (G)", "B quad (G)", "quad/lin", "note")
for (I, bl, bq) in zip(Isep, Blin, Bquad)
    note = bl < 0 ? "B < 0 (remanence dominates)" :
           I > maximum(cal.I) ? "extrapolated" : ""
    ratio = bl > 0 ? @sprintf("%9.3f", bq / bl) : @sprintf("%9s", "-")   # no ratio across B = 0
    @printf("  %8.3f  %10.1f  %10.1f  %s  %s\n", I, bl, bq, ratio, note)
end

I_top = Isep[end]
@printf("  at %.3f A: quadratic - linear = %+.1f G (%+.1f %% of the linear field)\n",
        I_top, Bquad[end] - Blin[end], 100 * (Bquad[end] / Blin[end] - 1))

# Both calibrations into `ax` along the current grid `If`: linear (solid) and quadratic
# (dashed), each evaluated at the separation currents (open circles), the CALIBFILE readings
# they were fitted to (black stars), the x = 0 readings of the profile runs (colored, not
# fitted) and the shaded calibration region `region` labeled at `xtext`. `keep` selects
# the currents drawn (I > 0 on a log axis).
function calibration!(ax, If, Isep, Blin, Bquad, cal, Ic, Bc, lin, quad;
                      keep = I -> true, region = extrema(cal.I),
                      xtext = sum(region) / 2)
    vspan!(ax, region...; color = (:gray, 0.15))
    text!(ax, xtext, 0.97 * maximum(Blin); text = "calibration\nregion", color = INK2,
          fontsize = 13, align = (:center, :top))
    hlines!(ax, 0; color = INK2, linewidth = 1)
    lines!(ax, If, polyval(lin.c, If); color = :black, linewidth = 2,
           label = @sprintf("linear: %.1f + %.1f I", lin.c...))
    lines!(ax, If, polyval(quad.c, If); color = :black, linewidth = 2, linestyle = :dash,
           label = @sprintf("quadratic: %.1f + %.1f I %+.2f I²", quad.c...))
    m = keep.(Isep)
    for B in (Blin, Bquad)
        scatter!(ax, Isep[m], B[m]; color = :transparent, strokecolor = :black,
                 strokewidth = 1.5, markersize = 9,
                 label = B === Blin ? "separation currents" : nothing)
    end
    k = keep.(cal.I)
    scatter!(ax, cal.I[k], cal.B[k]; marker = :star5, color = INK2, markersize = 15,
             label = "calibration, x = 0 (fitted)")
    for i in eachindex(Ic)
        scatter!(ax, [Ic[i]], [Bc[i]]; color = COLORS[i], marker = MARKERS[i], markersize = 11,
                 label = @sprintf("profile run, %.3f A", Ic[i]))
    end
end

# Center field vs. current from the two calibrations over the whole range of the separation
# data: linear current axis (top) and log axis (bottom, I > 0 only), which spreads out the
# separation currents below 50 mA where the remanence is a large part of B(0). Shaded: the
# current range of the calibration readings.
function plot_calibrations(Isep, Blin, Bquad, cal, Ic, Bc, lin, quad)
    fig  = Figure(size = (900, 900))
    xmax = 4.0
    pos  = Isep .> 0
    kmin = floor(Int, log10(minimum(Isep[pos])))
    kmax = ceil(Int, log10(xmax))
    args = (Isep, Blin, Bquad, cal, Ic, Bc, lin, quad)

    ax1 = Axis(fig[1, 1]; ylabel = L"B(x = 0)\ \mathrm{(G)}", xlabel = L"I\ \mathrm{(A)}",
               xticks = 0:0.5:4,
               title = "Center field vs. SG0 current: linear and quadratic calibration")
    calibration!(ax1, range(0, xmax; length = 300), args...)
    # lower right is free of data; the bottom margin lifts the legend clear of the B = 0 line
    axislegend(ax1; position = :rb, framevisible = false, margin = (10, 10, 40, 10))
    xlims!(ax1, -0.05, xmax)

    # log axis: the calibration region starts at the lowest nonzero calibration current
    lo = minimum(filter(>(0), cal.I))
    ax2 = Axis(fig[2, 1]; ylabel = L"B(x = 0)\ \mathrm{(G)}", xlabel = L"I\ \mathrm{(A)}",
               xscale = log10, xticks = (10.0 .^ (kmin:kmax), decade_label.(kmin:kmax)),
               xminorticks = IntervalsBetween(9))
    calibration!(ax2, 10 .^ range(kmin, log10(xmax); length = 300), args...;
                 keep = >(0), region = (lo, maximum(cal.I)), xtext = sqrt(lo * maximum(cal.I)))
    xlims!(ax2, 10.0^kmin, 10.0^kmax)
    fig
end

fig8 = plot_calibrations(Isep, Blin, Bquad, cal, Ic, Bc_raw, cal_lin, cal_quad)
display(fig8)
savefig(fig8, "SG0_fig8_calibrations")

fig9 = plot_sep_vs_field(curves, Blin, Bquad, Bcal)
display(fig9)
savefig(fig9, "SG0_fig9_separation_vs_field")


# =============================================================================
# CHAPTER 3 - CQD fit with the measured SG0 field
# =============================================================================


# #############################################################################
# Section 9 - CQD fit with the measured SG0 field
# #############################################################################
#
# Double-stage (SG0 -> SG1) CQD model and two-step fit of ../SG0_plotting_oct15_NYconference.jl,
# same equations and algorithm ("parallel first, then borrow k_i").
# Eq. n = ../fit_20260706/cqd_sg0_fit_20260706.pdf;  [n] = ../cqd_sg0_analytical_model_2026_10_13F.pdf.
#
# The only change is the SG0 dose: the measured field profile replaces the top-hat estimate
# (note, Sec. 9). Each stage acts through its longitudinal field integral,
#
#     BL0(I0) = ∫_SG0 |B0(y, I0)| dy,    Dc0(I0) = k_i |γe| BL0(I0) / v_a        [135]-[136]
#     BL1     = ∫_SG1 |B1(y, I1)| dy  ≈  B1 y_SG                                  [137]
#     Dc0(I0) / Dc1 = BL0(I0) / BL1                                             [138]
#
# with the SG0 field from Chapter 1, B0(y, I) = Bc(I) h(y):
# - h(y) = B(y)/B(0), the Section 5 global half-sigmoid mirrored about y = 0 (Section 6);
# - Bc(I), the center field per current: the CALIBFILE calibration of Section 8 (linear or
#   quadratic), or, for comparison, the slope β0 of the direct SG0 scans quoted in the note.
# Since h >= 0, BL0(I) = |Bc(I)| L_eff with L_eff = ∫ h(y) dy over the whole beam axis.
# No top hat and no divisor f: the SG0 scale is fixed by the measurement.
# Assumptions:
# - the profile shape does not depend on the current (Section 5), including above 1.202 A
#   where the calibration is extrapolated, and it is symmetric about y = 0 (Section 6);
# - Bc(I) is the coil part of the center field: the intercept b0 (3.7 G in the linear fit,
#   0.2 % of the field at 3.64 A) is neglected, as in [140]; the separations are referenced to I = 0 and
#   the model to Dc0 = 0.
# SG0 is a uniform field: standard QM ("Existing models") predicts ΔD = 0 (source note, Sec. 23).

# =============================================================================
# Setup: physical constants, atom, apparatus geometry, magnets
# =============================================================================

# ---------- Physical constants (NIST, CODATA 2022) ----------
# RSU : Relative Standard Uncertainty
const kb    = 1.380649e-23 ;       # Boltzmann constant (J/K)
const ħ     = 6.62607015e-34/2π ;  # Reduced Planck constant (J s)
const μ₀    = 1.25663706127e-6;    # Vacuum permeability (Tm/A)
const μB    = 9.2740100657e-24 ;   # Bohr magneton (J/T)
const γₑ    = -1.76085962784e11 ;  # Electron gyromagnetic ratio  (1/sT). Relative Standard Uncertainty = 3.0e-10
const μₑ    = 9.2847646917e-24 ;   # Electron magnetic moment (J/T). RSU = 3.0e-10
const Sspin = 1/2 ;                # Electron spin
const gₑ    = -2.00231930436092 ;  # Electron g-factor

# ---------- Atom ----------
atom = "39K"
K39_params = TheoreticalSimulation.AtomParams(atom); # [R μn γn Ispin Ahfs M ]

# ---------- Camera ----------
CAM_PIXELSIZE = 6.5e-6 ;               # Physical pixel size of camera [m]
Nx_PIXELS , Nz_PIXELS= (2160, 2560);   # (Nx,Nz) pixels

# ---------- Furnace, apertures and propagation distances (m) ----------
# Furnace
const TCelsius = 205
const T_K = 273.15 + TCelsius ; # Furnace temperature (K)
# Furnace aperture
const x_furnace = 2.0e-3 ;
const z_furnace = 100e-6 ;
# Slit : Pre SG
const x_slit  = 4.0e-3 ;
const z_slit  = 300e-6 ;
# Circular Aperture : Post SG
const R_aper            = 5.8e-3/2 ;
const y_SGToAperture    = 1e-3 ;
# Propagation distances
const y_FurnaceToSlit = 541.75e-3 ;
const y_SlitToSG      = 44.0e-3 ;
const y_SG            = 7.0e-2 ;      # SG1 length (also the CQD collapse-equivalent length l1)
const y_SGToScreen    = 395.25e-3 ;
# Connecting pipes
const R_tube = 35e-3/2 ; # Radius of the connecting pipe (m)

# ---------- Effusive beam ----------
effusion_params = TheoreticalSimulation.BeamEffusionParams(x_furnace,z_furnace,x_slit,z_slit,y_FurnaceToSlit,T_K,K39_params);

# ---------- Module defaults ----------
TheoreticalSimulation.DEFAULT_camera_pixel_size = CAM_PIXELSIZE;
TheoreticalSimulation.DEFAULT_x_pixels          = Nx_PIXELS;
TheoreticalSimulation.DEFAULT_z_pixels          = Nz_PIXELS;
TheoreticalSimulation.DEFAULT_x_furnace         = x_furnace;
TheoreticalSimulation.DEFAULT_z_furnace         = z_furnace;
TheoreticalSimulation.DEFAULT_x_slit            = x_slit;
TheoreticalSimulation.DEFAULT_z_slit            = z_slit;
TheoreticalSimulation.DEFAULT_y_FurnaceToSlit   = y_FurnaceToSlit;
TheoreticalSimulation.DEFAULT_y_SlitToSG        = y_SlitToSG;
TheoreticalSimulation.DEFAULT_y_SG              = y_SG;
TheoreticalSimulation.DEFAULT_y_SGToScreen      = y_SGToScreen;
TheoreticalSimulation.DEFAULT_R_tube            = R_tube;
TheoreticalSimulation.DEFAULT_c_aperture        = R_aper;
TheoreticalSimulation.DEFAULT_y_SGToAperture    = y_SGToAperture;

# ---------- Magnets ----------
# SG1 at its fixed operating current, from the module's field and gradient calibration
# (the mode selected at the top of the script: TheoreticalSimulation.set_magnetic_field!)
const I1 = 0.040                                   # SG1 coil current (A)
const G1 = TheoreticalSimulation.GvsI(I1)          # SG1 gradient (T/m)
const B1 = TheoreticalSimulation.BvsI(I1)          # SG1 field (T)
# SG0: field profile and Hall-map slope from Chapter 1; for comparison, the center-field
# slope of the direct SG0 scans quoted in the note ([19])
const β0 = 0.031895                                # SG0 local slope, direct scans (T/A)

println("""
***************************************************
CAMERA FEATURES
    Number of pixels        : $(Nx_PIXELS) × $(Nz_PIXELS)
    Pixel size              : $(1e6*CAM_PIXELSIZE) μm
***************************************************
SETUP FEATURES
    Temperature             : $(T_K)
    Furnace aperture (x,z)  : ($(1e3*x_furnace)mm , $(1e6*z_furnace)μm)
    Slit (x,z)              : ($(1e3*x_slit)mm , $(1e6*z_slit)μm)
    Post-SG aperture radius : $(1e3*R_aper)mm
    Furnace → Slit          : $(1e3*y_FurnaceToSlit)mm
    Slit → SG magnet        : $(1e3*y_SlitToSG)mm
    SG magnet               : $(1e3*y_SG)mm
    SG magnet → Screen      : $(1e3*y_SGToScreen)mm
    SG magnet → Aperture    : $(1e3*y_SGToAperture)mm
    Tube radius             : $(1e3*R_tube)mm
***************************************************
MAGNETS (field mode: $(TheoreticalSimulation.magnetic_field_mode()))
    SG1 at $(1e3*I1) mA        : G1 = $(round(G1; digits = 3)) T/m,  B1 = $(round(1e3*B1; digits = 3)) mT
    SG0 direct-scan slope   : β0 = $(β0) T/A   (Hall map: Chapter 1)
***************************************************
""")


# =============================================================================
# 9.1  CQD model parameters
# =============================================================================
#
# All derived from the setup above. In brackets: the NY script's rounded values, taken from
# the note, whose SG1 inputs were G1 = 10.779 T/m and B1 = 18.931 mT.
#
#   v_a      = sqrt(9π α2 / 8),  α2 = kb T_K / M    mean speed of the effusive beam    [600.5 m/s]
#   γe       = |γₑ|                                 electron gyromagnetic ratio        [1.76086e11 rad/(s T)]
#   T1       = y_SG / v_a                           time inside SG1                    [116.6 µs]
#   BL1      = B1 y_SG                              SG1 field integral, [137]          (T m)
#   Kz       = μₑ G1 T1² / M                        single-beam deflection scale       (m)
#   KΔ_pix   = 2 Kz / d_pix                         peak-to-peak scale, Eq. 25         [6.47 px]
#   Dsep0_px = 13 px                                measured zero-current separation   (input)
#
# One fully aligned atom lands at z = Kz (DIR + 1/2): Kz/2 inside SG1 (½ a T1²) and Kz DIR in
# the drift (a T1 T_d), with DIR = T_d / T1. The two beams move apart symmetrically, so the
# separation is 2z, hence the factor 2 in KΔ_pix. The model separation is
#     D_sep = KΔ_pix (DIR + 1/2) Z,    Z <= 1 the CQD ensemble response,
# and DIR is solved from the baseline D_sep(I0 = 0) = Dsep0_px (`solve_DIR`). That fitted DIR
# absorbs any scale error in KΔ_pix. The geometric value DIR_geom = y_SGToScreen / y_SG is
# printed beside it as a check of the absolute deflection budget.
const CQD = let M = K39_params.M
    v_a      = sqrt(9π * effusion_params.α2 / 8)
    T1       = y_SG / v_a
    Kz       = μₑ * G1 * T1^2 / M
    DIR_geom = y_SGToScreen / y_SG
    (; v_a, 
       γe = abs(γₑ), 
       T1, 
       BL1 = B1 * y_SG,
       Kz, KΔ_pix = 2Kz / CAM_PIXELSIZE, 
       DIR_geom,
       zmax_geom = Kz * (DIR_geom + 0.5),
       Dsep0_px = 13.0)
end

# k_i from the SG1 dose: k_i = Dc1 v_a / (|γe| BL1), Eq. 10
ki_from_dose(Dc1) = Dc1 * CQD.v_a / (CQD.γe * CQD.BL1)

# =============================================================================
# 9.2  Numerical tools
# =============================================================================

"""
    gauss_legendre(n) -> (x, w)

Nodes `x` and weights `w` of the n-point Gauss-Legendre rule on [-1, 1]:

    ∫_{-1}^{1} f(z) dz ≈ Σ_i w_i f(x_i),

exact when f is a polynomial of degree ≤ 2n - 1, and converging very fast for smooth f
(all integrands of the CQD model are smooth on each sub-interval; see `Z_geom` for the kinks).

The nodes are the n roots of the Legendre polynomial P_n, found one by one with Newton's method:
- initial guess (Tricomi asymptotic): x_i ≈ cos(π (i - 1/4) / (n + 1/2)), i = 1..n,
  close enough that Newton converges to the i-th root (nodes come out in descending order);
- P_n(z) by the three-term recurrence, starting from P_0 = 1, P_{-1} = 0:
      j P_j(z) = (2j - 1) z P_{j-1}(z) - (j - 1) P_{j-2}(z);
- its derivative from P_n and P_{n-1}:
      P_n'(z) = n [z P_n(z) - P_{n-1}(z)] / (z² - 1);
- Newton step z ← z - P_n(z) / P_n'(z), until |Δz| < 1e-15 (at most 100 steps);
- weight at the converged root:
      w_i = 2 / [(1 - x_i²) P_n'(x_i)²].

Checks: Σ w_i = 2 (the integral of 1), and the rule is symmetric, x_{n+1-i} = -x_i.
Use `integ` to apply it on any interval [a, b].
"""
function gauss_legendre(n)
    x, w = zeros(n), zeros(n)
    for i in 1:n
        z = cos(π * (i - 0.25) / (n + 0.5)); dp = 0.0        # initial guess for root i
        for _ in 1:100
            p1, p2 = 1.0, 0.0                                 # P_j(z), P_{j-1}(z)
            for j in 1:n
                p1, p2 = ((2j - 1) * z * p1 - (j - 1) * p2) / j, p1
            end                                               # now p1 = P_n(z), p2 = P_{n-1}(z)
            dp = n * (z * p1 - p2) / (z^2 - 1)                # P_n'(z)
            dz = p1 / dp; z -= dz                             # Newton step
            abs(dz) < 1e-15 && break
        end
        x[i], w[i] = z, 2 / ((1 - z^2) * dp^2)
    end
    return x, w
end

# The two quadrature rules of the model, computed once here and reused by every integral
# (`const` keeps them type-stable as globals). Each is a tuple (x, w) of nodes and weights
# on [-1, 1]; `integ` maps them to the actual interval.
# The integrals are nested: every outer node over c calls the inner integral over λ, so one
# ensemble response costs 48 × 24 = 1152 integrand evaluations (three times that in Z_geom,
# split at its kinks). The inner integrand tanh(C + λ) is smooth, so 24 nodes are enough there.
const GL48 = gauss_legendre(48)   # outer: ensemble integrals over the entrance cosine c ∈ [-1, 1]
                                  #        in Z_par, Z_ret, Z_geom (default rule of `integ`)
const GL24 = gauss_legendre(24)   # inner: SG1 dose integral over λ ∈ [0, Dc1] in z_ex (Eq. 17)

"""
    integ(f, a, b, (X, W) = GL48) -> Float64

Approximate ∫_a^b f(y) dy with a Gauss-Legendre rule (nodes `X`, weights `W` on [-1, 1],
from `gauss_legendre`). The rule is moved to [a, b] by the linear change of variable

    y = m + h z,   h = (b - a)/2 (half-width),  m = (a + b)/2 (midpoint),  dy = h dz,

so that

    ∫_a^b f(y) dy = h ∫_{-1}^{1} f(m + h z) dz ≈ h Σ_i W_i f(m + h X_i).

The factor h is the Jacobian of the map. `f` is any one-argument function (usually an
anonymous function closing over the model parameters); the fourth argument is a tuple
destructured in the signature, `GL48` by default, `GL24` for the inner SG1 integral in `z_ex`.

Accuracy: exact for polynomials of degree ≤ 2n - 1 on [a, b], and fast for smooth f. The rule
does not see kinks or singularities inside [a, b]: split the interval there and add the pieces
(as `Z_geom` does at c = ±tanh(Dc0/2)). Works for b < a too (h < 0 flips the sign).

Examples: `integ(c -> (1 + c) * z_ex(c, Dc1, ρ), -1.0, 1.0)` (ensemble average, GL48);
`integ(λ -> tanh(C + λ), 0.0, Dc1, GL24)` (SG1 dose integral).
"""
function integ(f, a, b, (X, W) = GL48)
    h, m = (b - a) / 2, (b + a) / 2                                # half-width, midpoint
    return h * sum(W[i] * f(m + h * X[i]) for i in eachindex(X))   # generator: no allocation
end

"""
    golden(fn, a, b; n = 60) -> (x_min, f_min)

Minimize a one-variable function `fn` on [a, b] by golden-section search (no derivatives).

The bracket [a, b] holds two interior points placed at the golden ratio r = (√5 - 1)/2 ≈ 0.618:

    c = b - r (b - a),   d = a + r (b - a),   a < c < d < b.

Each iteration compares fn(c) and fn(d) and discards the part of the bracket that cannot hold
the minimum:
- fn(c) < fn(d): the minimum is in [a, d]  ->  b ← d, the old c becomes the new d, new c;
- otherwise:     the minimum is in [c, b]  ->  a ← c, the old d becomes the new c, new d.
Because r² = 1 - r, the surviving interior point lands exactly on the golden position of the
new bracket, so each iteration costs one new evaluation of `fn` and shrinks the bracket by r.
After n iterations its width is r^n (b - a): 0.618^60 ≈ 3e-13 of the start (n + 2 evaluations).

Returns the better of the two final interior points and its value.

Assumes `fn` has a single minimum in [a, b] (unimodal); otherwise it converges to one local
minimum. If the minimum is at an edge, the result approaches that edge. Used in `fit_cqd` for
log(Dc1), bracketed first by a coarse scan (n = 50), and for the weight parameter on [0, 5]
(κ or RLC, n = 60). Without derivatives it suits χ², which is costly (nested integrals and a
`solve_DIR` at each point).
"""
function golden(fn, a, b; n = 60)
    r = (sqrt(5) - 1) / 2                                          # golden ratio conjugate, 0.618
    c, d = b - r * (b - a), a + r * (b - a); fc, fd = fn(c), fn(d)  # interior points, a < c < d < b
    for _ in 1:n
        if fc < fd      # minimum in [a, d]: drop (d, b]; old c becomes new d
            b, d, fd = d, c, fc; c = b - r * (b - a); fc = fn(c)
        else            # minimum in [c, b]: drop [a, c); old d becomes new c
            a, c, fc = c, d, fd; d = a + r * (b - a); fd = fn(d)
        end
    end
    return fc < fd ? (c, fc) : (d, fd)
end

"""
    chi2(model, y, σ) -> Float64

Chi-square of a model against data, Eq. 32:

    χ² = Σ_i [(model_i - y_i) / σ_i]²,

with `model`, `y` and `σ` vectors of the same length (σ = per-point 1σ uncertainties from the
data file). Each point is weighted by 1/σ_i², assuming independent Gaussian errors.
For a good fit χ² ≈ N - k (N points, k fitted parameters). The profile Δχ² = 1 rule in
`fit_cqd` gives the 1σ interval of the single fitted parameter (Eq. 33).
"""
chi2(model, y, σ) = sum(((model .- y) ./ σ) .^ 2)

# =============================================================================
# 9.3  Double-stage CQD model
# =============================================================================
#
# Notation: Dc0, Dc1 = SG0, SG1 doses;  c = entrance cosine;  ρ = DIR Dc1 (drift in dose units).

"""
    transfer(c, q, Dc) -> c′

CQD transfer map of one magnet stage, Eq. 8 / [145]: the electron-moment alignment c = cos θ
(θ measured from +z) at the stage entrance becomes c′ at its exit.

Physics. Inside a stage the field B points along q ẑ (q = ±1) and CQD drives the moment
monotonically toward it,

    dc/dDc = q (1 - c²),      dose  Dc = k_i |γe| ∫ B dt = k_i |γe| BL / v_a   [135]-[136].

The dose is the dimensionless collapse "angle": it depends on the field only through its
integral BL along the path (that is why only BL0 and BL1 enter, [138]). Since
d artanh(c) / dDc = q, the rapidity artanh(c) moves at a constant rate:

    artanh(c′) = artanh(c) + q Dc,

and the tanh addition formula gives the closed form

    c′ = tanh(artanh c + q Dc) = (c + q tanh Dc) / (1 + q c tanh Dc).

Equivalently, with artanh(cos θ) = -ln tan(θ/2):  tan(θ′/2) = tan(θ/2) e^{-q Dc}.

Properties (these make the map a one-parameter group):
- c′ stays in [-1, 1]; the fully aligned states c = ±1 are fixed points;
- c′ → q as Dc → ∞ (full collapse onto the field); Dc = 0 is the identity;
- small dose: c′ ≈ c + q Dc (1 - c²), maximal at c = 0 (θ = 90°), zero at the poles;
- composition adds doses: transfer(transfer(c, q, D₁), q, D₂) = transfer(c, q, D₁ + D₂);
- inverse = reversed field: transfer(transfer(c, q, D), -q, D) = c.

Use in the model:
- SG0 (dose Dc0): c′ = transfer(c, ±1, Dc0) in Z_par, Z_ret and Z_geom;
- SG1: z_ex applies the same map continuously, tanh(artanh c + λ) for the running dose
  λ ∈ [0, Dc1] (q1 = +1), weighted by the remaining lever arm (Eq. 17).
"""
transfer(c, q, Dc) = (t = tanh(Dc); (c + q * t) / (1 + q * c * t))   # tanh(artanh c + q Dc)

"""
    z_ex(c, Dc1, ρ) -> z̃ ∈ [-1, 1]

Normalized deflection at the camera of one atom entering SG1 with alignment c = cos θ,
Eq. 17 / [204], [211]; SG1 field along +z (q1 = +1).

Physics. The SG1 force is F = μₑ G1 cos θ(t). A kick at time t is followed by the rest of the
magnet, T1 - t, and the drift T_d, so the camera displacement is

    z = (μₑ G1 / M) ∫_0^T1 (T1 + T_d - t) cos θ(t) dt.

With the running dose λ = Dc1 t / T1 ∈ [0, Dc1], the alignment follows the transfer map,
cos θ(λ) = tanh(artanh c + λ), and the drift in dose units is ρ = DIR Dc1 (DIR = T_d / T1):

    z = Kz / Dc1² ∫_0^Dc1 (Dc1 + ρ - λ) tanh(artanh c + λ) dλ,        Kz = μₑ G1 T1² / M.

(Dc1 + ρ - λ) is the lever arm of the kick received at dose λ. Dividing by the deflection of a
fully aligned atom, z_full = Kz (DIR + 1/2), i.e. by ∫_0^Dc1 (Dc1 + ρ - λ) dλ = ρ Dc1 + Dc1²/2:

    z̃(c) = z / z_full = ∫_0^Dc1 (Dc1 + ρ - λ) tanh(artanh c + λ) dλ / (ρ Dc1 + Dc1²/2).

Limits:
- z̃(±1) = ±1: fully aligned atoms are fixed points of the map;
- small dose: z̃ ≈ c + δ (1 - c²),  δ = Dc1 (ρ/2 + Dc1/6) / (ρ + Dc1/2)   (≈ Dc1/2 for ρ ≫ Dc1):
  a classical moment (z̃ = c) plus the CQD alignment gained inside SG1;
- large dose: z̃ → 1 for every c > -1 (full collapse onto +z).

Numerics: c is clamped just inside (-1, 1) so that artanh stays finite; the λ integral uses
the inner rule GL24.
"""
function z_ex(c, Dc1, ρ)
    C = atanh(clamp(c, -1 + 1e-15, 1 - 1e-15))                        # rapidity artanh(c)
    lever(λ) = Dc1 + ρ - λ                                            # remaining lever arm
    return integ(λ -> lever(λ) * tanh(C + λ), 0.0, Dc1, GL24) /       # ∫ lever × cos θ(λ)
           (ρ * Dc1 + Dc1^2 / 2)                                      # same, fully aligned
end

# -----------------------------------------------------------------------------
# Heart-weighted ensemble responses, Eqs. 19-22 / [217]-[221], [251]
# -----------------------------------------------------------------------------
#
# The beam enters SG0 isotropic: the electron and nuclear cosines c = cos θe and cn = cos θn are
# independent and uniform on [-1, 1]. In CQD the SG0 branch is set by comparing them,
# q0 = sgn(c - cn) ([153]), so for a given c the + branch holds the nuclear interval cn < c, of
# measure 1 + c ([117]). The electron distribution of the + beam is therefore
#
#     w(c) = (1 + c) / 2,     ∫_{-1}^{1} w(c) dc = 1      (= cos²(θ/2), the Born-rule value),
#
# a cardioid ("heart") in polar form, r ∝ 1 + cos θ. The ensemble response is the average of the
# single-atom response over it, and the − beam is its mirror image (-Z), so the beam separation is
#
#     D_sep = KΔ_pix (DIR + 1/2) Z,        ΔD = Dsep0 [Z / Z0 - 1]   (Eq. 27),  Z0 = Z_par(0, ...).
#
# Each atom first crosses SG0, c -> c′ = transfer(c, +1, Dc0), then SG1, response z̃(±c′).
#
# Limits (checked numerically for this fit, Dc1 = 0.0305, DIR = 6.0):
# - Dc0 = 0 (SG0 off): Z_par(0) ≈ 1/3 + 2δ/3 = 0.343. The 1/3 is ⟨c⟩ over the heart: the
#   classical, collapse-free part; 2δ/3 is the CQD gain in SG1. It sets the 13-px baseline:
#   5.823 px × (6.0 + 1/2) × 0.343 = 13.0 px  (`solve_DIR`).
# - Dc0 → ∞ (strong SG0): every atom leaves SG0 with c′ = +1, so Z_par → 1 and
#   ΔD_par → 13 (1/Z0 - 1) ≈ 25 px (saturation); Z_ret → -1.

"""
    Z_par(Dc0, Dc1, ρ) -> Z

Parallel set: SG0 and SG1 fields along the same axis. The SG0 output enters SG1 unchanged:

    Z_par = ∫_{-1}^{1} (1 + c)/2 · z̃(transfer(c, +1, Dc0)) dc.

SG0 pushes the + beam's alignment toward +z, so Z_par rises from Z0 (Dc0 = 0) toward 1.
"""
Z_par(Dc0, Dc1, ρ) = 0.5 * integ(c -> (1 + c) * z_ex( transfer(c, 1, Dc0), Dc1, ρ), -1.0, 1.0)

"""
    Z_ret(Dc0, Dc1, ρ) -> Z

Antiparallel set, retained label [251]: the SG1 field is reversed relative to SG0, and the atom
keeps the branch label (heart weight) assigned in SG0. In SG1's frame its entrance alignment
is reversed:

    Z_ret = ∫_{-1}^{1} (1 + c)/2 · z̃(-transfer(c, +1, Dc0)) dc.

Z_ret(0) ≈ -1/3 + 2δ/3 and Z_ret → -1 as Dc0 → ∞: the beam crosses to the other side.
It is one end of the antiparallel mixture, (1 - w) Z_geom + w Z_ret (Eq. 23).
"""
Z_ret(Dc0, Dc1, ρ) = 0.5 * integ(c -> (1 + c) * z_ex(-transfer(c, 1, Dc0), Dc1, ρ), -1.0, 1.0)

"""
    Z_geom(Dc0, Dc1, ρ) -> Z

Antiparallel set, pure CQD: the strict mirror-and-rebranch rule, Eqs. 20, 22 / [218]-[221].

Physics ([157]-[171]). Between the stages the field reverses. The electron follows it
adiabatically and is mirror-imaged, θe → π - θe, while the nucleus stays frozen (θn fixed):

    SG1 entrance cosine  c1 = -T(c),    T = transfer(c, q0, Dc0) the SG0 output.

The SG1 branch is then re-evaluated from the new pair of angles (rebranch): q1 = +1 when
cn < -T(c) ([165], [168]-[171]). This cuts each SG0 triangle of the (c, cn) square in two:
- q0 = +1 (cn < c):  cn < -T+(c) gives q1 = +1, the retained label (counter-prepared after the
  mirror); -T+ < cn < c gives q1 = -1, the reversed label;
- q0 = -1 (cn > c):  c < cn < -T-(c) gives q1 = +1, the reversed label (constructively prepared);
  cn > -T-(c) gives q1 = -1, the retained label.
By inversion symmetry the separation is set by the q1 = +1 beam, which collects

    L+(c) = 1 + min[c, -T+(c)]       (retained, from q0 = +1),
    L-(c) = max[0, -T-(c) - c]       (reversed, from q0 = -1),

the nuclear measures left in each triangle: the heart weight 1 + c "cut by the new SG1
boundary". Both groups enter SG1 mirrored, so

    Z_geom = (1/2) ∫_{-1}^{1} [ L+(c) z̃(-T+(c)) + L-(c) z̃(-T-(c)) ] dc,    T± = transfer(c, ±1, Dc0).

Limits:
- Dc0 = 0: L+ + L- = 1 - c, and c → -c turns the integral into Z_par(0). Both configurations
  share the baseline Z0 (checked: difference 4e-16).
- Dc0 > 0: Z_geom rises with Dc0 (pure CQD predicts a larger separation, e.g. +0.82 px at
  3.64 A in the note), while the measured antiparallel separation decreases. The fitted
  antiparallel response is therefore the mixture (1 - w) Z_geom + w Z_ret (Eq. 23), with a
  phenomenological retained-label weight w(Dc0) and w(0) = 0, so the baseline is kept.

Numerics: L± are continuous but have kinks where the min/max switch, at c = -T+(c) and
c = -T-(c), i.e. at c = ∓tanh(Dc0/2). Gauss-Legendre assumes a smooth integrand, so the integral
is split there into three smooth pieces. The z_ex(-T-) term is skipped where L- = 0.
"""
function Z_geom(Dc0, Dc1, ρ)
    g = function (c)
        Tp, Tm = transfer(c, 1, Dc0), transfer(c, -1, Dc0)            # SG0 outputs, q0 = ±1
        Lp, Lm = 1 + min(c, -Tp), max(0.0, -Tm - c)                    # q1 = +1 nuclear measures
        return Lp * z_ex(-Tp, Dc1, ρ) + (Lm > 0 ? Lm * z_ex(-Tm, Dc1, ρ) : 0.0)   # mirrored into SG1
    end
    k = tanh(Dc0 / 2)                                                  # kinks of L± at c = ∓k
    return 0.5 * (integ(g, -1.0, -k) + integ(g, -k, k) + integ(g, k, 1.0))
end

"""
    solve_DIR(Dc1) -> DIR

Drift-to-interaction ratio DIR = T_d / T1 inferred from the measured zero-current separation,
Eq. 26 / [249]:

    KΔ_pix (DIR + 1/2) Z_par(0; Dc1, ρ = DIR Dc1) = Dsep0_px  (13 px).

At I0 = 0 SG0 does nothing, so the 13 px come from SG1 alone. The left side is the model
separation: the mechanical scale KΔ_pix times the lever factor DIR + 1/2 times the baseline
ensemble response Z0 ≈ 1/3 + Dc1/3 (weak dose). For each trial Dc1 this one equation fixes DIR,
so the drift length L_d = DIR y_SG is an inferred quantity. It also absorbs any error in the
absolute scale KΔ_pix: the fit only uses ratios Z/Z0. Compare it with the geometric value
CQD.DIR_geom = y_SGToScreen / y_SG.

Method: bisection on [0.01, 200]. g(DIR) = left side - 13 increases with DIR (the lever arm grows,
and Z0 grows slightly through ρ), so the root is unique. g(0.01) < 0 whenever
KΔ_pix < 13 / (0.51 Z0) (about 74 px here), and g(200) > 0. 80 halvings shrink the bracket to
200 / 2^80 ≈ 2e-22, i.e. machine precision. Because g is increasing, the sign of g(m) alone
says which half holds the root: one evaluation of Z_par per step. If there were no sign change
the result would silently stick to an edge.
For this fit (Dc1 = 0.0305): DIR = 6.005 (residual 2e-15 px) vs geometric 5.646.
"""
function solve_DIR(Dc1)
    g(DIR) = CQD.KΔ_pix * (DIR + 0.5) * Z_par(0.0, Dc1, DIR * Dc1) - CQD.Dsep0_px
    a, b = 0.01, 200.0                                    # bracket: g(a) < 0 < g(b)
    for _ in 1:80
        m = (a + b) / 2
        g(m) < 0 ? (a = m) : (b = m)                      # g increasing: root above m if g(m) < 0
    end
    return (a + b) / 2
end

"""
    cqd_responses(Dc1, BL0) -> (; DIR, Z0, Dc0, zp, zg, zr)

All ensemble responses of the model for one trial SG1 dose `Dc1` at the SG0 field integrals
`BL0` (vector, T m; one entry per SG0 current). This is the single place where the two stages
are linked.

Steps:
1. Geometry from the baseline: DIR = solve_DIR(Dc1), ρ = DIR Dc1, and Z0 = Z_par(0; Dc1, ρ),
   the zero-current response that every curve is referenced to.
2. SG0 dose at each current, [138]:

       Dc0 = Dc1 BL0 / BL1.

   Both doses are Dc = k_i |γe| BL / v_a ([136]-[137]) with the same k_i and v_a, so their ratio is
   the ratio of field integrals. Once Dc1 is chosen (equivalently k_i, via `ki_from_dose`), the
   measured SG0 field fixes Dc0 at every current with no further free parameter.
3. The three responses at each Dc0: zp = Z_par (parallel), zg = Z_geom (antiparallel, strict
   rule) and zr = Z_ret (antiparallel, retained label).

Cost: one call = one `solve_DIR` (80 Z_par) plus 3 responses per current. Z_geom, the most
expensive, is computed even when only the parallel set is needed (`fit_cqd`, step 1).
"""
function cqd_responses(Dc1, BL0)
    DIR = solve_DIR(Dc1); ρ = DIR * Dc1                # drift ratio from the 13-px baseline
    Z0  = Z_par(0.0, Dc1, ρ)                           # zero-current reference
    Dc0 = Dc1 .* BL0 ./ CQD.BL1                        # SG0 dose per current, [138]
    return (; DIR, Z0, Dc0,
              zp = [Z_par(d, Dc1, ρ)  for d in Dc0],   # parallel
              zg = [Z_geom(d, Dc1, ρ) for d in Dc0],   # antiparallel, mirror-and-rebranch
              zr = [Z_ret(d, Dc1, ρ)  for d in Dc0])   # antiparallel, retained label
end

# -----------------------------------------------------------------------------
# Model separation changes in pixels, Eq. 27 / [240]-[241]
# -----------------------------------------------------------------------------
#
# The separation is D_sep = KΔ_pix (DIR + 1/2) Z, and `solve_DIR` makes it 13 px at I0 = 0. The
# measured quantity is the change relative to I0 = 0:
#
#     ΔD = D_sep(I0) - D_sep(0) = KΔ_pix (DIR + 1/2) (Z - Z0) = Dsep0 [Z / Z0 - 1],   Dsep0 = 13 px.
#
# The mechanical scale KΔ_pix (G1, T1, v_a, d_pix) cancels: only the ratio Z/Z0 matters.
# Signs: ΔD > 0 is a wider separation. ΔD_par saturates at 13 (1/Z0 - 1) ≈ 25 px for Z → 1.

"""
    dD_par(r) -> Vector (px)

Model separation change of the parallel set at the currents of `r` (a `cqd_responses` output):

    ΔD_par = Dsep0 [Z_par / Z0 - 1],   Dsep0 = 13 px.

Pure CQD with no extra parameter. It increases monotonically with Dc0 (SG0 aligns the + beam
along SG1) and saturates at 13 (1/Z0 - 1) ≈ 25 px.
"""
dD_par(r) = CQD.Dsep0_px .* (r.zp ./ r.Z0 .- 1)

"""
    dD_anti(r, w) -> Vector (px)

Model separation change of the antiparallel set, Eq. 23 / [252]-[253], for a `cqd_responses`
output `r` and retained-label weights `w` (one per current, from `w_ret`):

    ΔD_anti = Dsep0 [((1 - w) Z_geom + w Z_ret) / Z0 - 1].

Each atom keeps its SG0 label with probability w (response Z_ret, which drives the separation
down) and otherwise follows the strict mirror-and-rebranch rule (Z_geom, which pushes it up).
w = 0 is pure CQD (χ² ≈ 129 for 15 points: rejected). The note calls the retained-label term
phenomenological: it stands in for fringe-field dynamics between the stages that the model
does not resolve.
"""
dD_anti(r, w) = CQD.Dsep0_px .* (((1 .- w) .* r.zg .+ w .* r.zr) ./ r.Z0 .- 1)

"""
    w_ret(Dc0, p, form) -> w

Retained-label weight of the antiparallel mixture as a function of the SG0 dose, Eq. 24.
`p` is the single fitted parameter:

    form = :lin   w = min(1, κ Dc0)         (p = κ; capped at 1 so that w stays a probability)
    form = :exp   w = 1 - exp(-RLC Dc0)     (p = RLC; saturates smoothly at 1)

Both give w(0) = 0, so the I0 = 0 baseline is unchanged, since Z_ret(0) ≠ Z0. They share the
initial slope p, w ≈ p Dc0. Here Dc0 ≤ 0.2 and κ ≈ 0.46, so w ≤ 0.09: the cap is never active,
and the two forms differ only at second order in Dc0.
Any `form` other than :lin is treated as :exp.
"""
w_ret(Dc0, p, form) = form === :lin ? min.(1.0, p .* Dc0) : 1 .- exp.(-p .* Dc0)

# =============================================================================
# 9.4  Two-step fit
# =============================================================================
#
# Strategy "parallel first, then borrow k_i" (fit note, Sec. 5; same as the NY script):
# the parallel set is pure CQD with one parameter, so it alone fixes Dc1, i.e. k_i. The
# antiparallel set then only fits the phenomenological retained-label weight, with k_i
# held fixed, so the weight cannot pull k_i.

"""
    fit_cqd(curves, BL0; wform = :lin, Igrid = 0:0.02:3.8) -> NamedTuple

Two-step CQD fit of the spin-resolved separations for one SG0 field model.

# Arguments
- `curves`: the Section 7 data, a vector of `(label, I, y, yerror)`. The set whose label contains
  "antiparallel" is the antiparallel set, the other one the parallel set.
- `BL0`: function I (A) -> SG0 field integral (T m), [135], e.g. `BL0_lin` (Section 9.5).
- `wform`: retained-label weight, `:lin` (w = κ Dc0) or `:exp` (w = 1 - exp(-RLC Dc0)), Eq. 24.
- `Igrid`: currents (A) at which the fitted model curves are returned for plotting.

# Step 1: parallel set → Dc1 and k_i (Eqs. 32-33)
χ²_par(Dc1) = Σ [(ΔD_par(I_j; Dc1) - y_j) / σ_j]² over the parallel points, where every
evaluation re-solves DIR from the 13-px baseline (`cqd_responses`).
1. Coarse scan: 41 points of log(Dc1), evenly spaced on [1e-3, 5]. The search runs in log(Dc1)
   because Dc1 may lie anywhere over orders of magnitude.
2. Refinement: golden section between the neighbours of the best scan point (n = 50).
   `Dc1_at_edge` flags a minimum at an end of the scan (the true minimum may lie outside).
3. 1σ interval from the profile Δχ² = 1 rule: on each side the point where
   χ²_par = χ²_min + 1, found by bisection on the geometric midpoint √(a b) in a bracket up to
   4× (or 1/4×) Dc1. If χ² has not risen by 1 at that distance, the bound is NaN.
k_i = Dc1 v_a / (|γe| BL1) is proportional to Dc1, so its interval maps directly.

# Step 2: antiparallel set → weight parameter, with Dc1 fixed
χ²_anti(p) is minimized by golden section on p ∈ [0, 5] (n = 60). The 1σ error comes from the
curvature: near the minimum χ² ≈ χ²_min + (χ''/2)(p - p̂)², so Δχ² = 1 at
σ_p = √(2 / χ''). χ'' is taken by a central second difference with step h = max(1e-4, 1e-3 p).
This error is conditional on the fixed Dc1: it does not include the uncertainty of k_i.
For reference, `chi2_anti_pure` is χ² of pure CQD (w = 0) with the same Dc1.

The intervals are statistical only. Systematics of the field model (L_eff, the center-field
calibration) are explored by fitting each model in `FIELD_MODELS` separately.

# Returns
A NamedTuple:
- `Dc1`, `Dc1_at_edge`: best SG1 dose, and whether the scan minimum hit an edge;
- `ki`, `ki_lo`, `ki_hi`: k_i and its Δχ² = 1 interval (NaN if not bracketed);
- `chi2_par`, `n_par`: minimum χ² of the parallel set and its number of points;
- `DIR`: drift ratio solved from the baseline at the best Dc1;
- `Dc0_max`: largest SG0 dose reached by the data;
- `wform`, `p`, `p_err`: weight form, fitted κ (or RLC) and its curvature error;
- `chi2_anti`, `chi2_anti_pure`, `n_anti`: antiparallel χ² with the fitted weight, with
  w = 0, and its number of points;
- `I_fit`, `dD = (par, anti)`: model curves (px) on `Igrid`.

Cost: every χ² evaluation calls `cqd_responses` (a `solve_DIR` plus three responses per
point). Including the Δχ² bounds, step 1 takes roughly 200 evaluations.
"""
function fit_cqd(curves, BL0; wform = :lin, Igrid = 0:0.02:3.8)
    par  = curves[findfirst(c -> !occursin("antiparallel", c.label), curves)]
    anti = curves[findfirst(c ->  occursin("antiparallel", c.label), curves)]

    # ---- Step 1: parallel set, one parameter Dc1 ----
    BL0_par = BL0.(par.I)                                       # SG0 field integrals at the data (T m)
    χ²_par(Dc1) = chi2(dD_par(cqd_responses(Dc1, BL0_par)), par.y, par.yerror)

    logs = range(log(1e-3), log(5.0); length = 41)             # coarse scan in log(Dc1)
    ib   = argmin([χ²_par(exp(x)) for x in logs])
    x1, χ²p = golden(x -> χ²_par(exp(x)), logs[max(1, ib - 1)], logs[min(end, ib + 1)]; n = 50)
    Dc1 = exp(x1)                                               # best SG1 dose

    # Profile Δχ² = 1 bound on one side (up = true: above Dc1), by bisection in log(Dc1):
    # a stays inside (Δχ² < 1), b outside, until they meet at χ² = χ²_min + 1
    function bound(up)
        a, b = Dc1, up ? 4Dc1 : Dc1 / 4
        χ²_par(b) - χ²p < 1 && return NaN                       # not bracketed within 4×
        for _ in 1:50
            m = sqrt(a * b); χ²_par(m) - χ²p < 1 ? (a = m) : (b = m)
        end
        return sqrt(a * b)
    end
    Dc1_lo, Dc1_hi = bound(false), bound(true)

    # ---- Step 2: antiparallel set, Dc1 fixed; pure CQD (w = 0) for reference ----
    ra = cqd_responses(Dc1, BL0.(anti.I))                       # responses do not depend on p
    χ²_anti(p) = chi2(dD_anti(ra, w_ret(ra.Dc0, p, wform)), anti.y, anti.yerror)
    p, χ²a = golden(χ²_anti, 0.0, 5.0)                          # κ (or RLC)
    h      = max(1e-4, 1e-3 * p)                                # finite-difference step
    p_err  = sqrt(2 / ((χ²_anti(p + h) - 2χ²_anti(p) + χ²_anti(p - h)) / h^2))   # σ_p = √(2/χ'')
    χ²pure = chi2(dD_anti(ra, zeros(length(anti.I))), anti.y, anti.yerror)        # w = 0

    # ---- Model curves on the plotting grid ----
    I_fit = collect(Igrid)
    rg    = cqd_responses(Dc1, BL0.(I_fit))
    dD    = (par = dD_par(rg), anti = dD_anti(rg, w_ret(rg.Dc0, p, wform)))

    return (; Dc1, Dc1_at_edge = ib in (1, length(logs)),
              ki = ki_from_dose(Dc1), ki_lo = ki_from_dose(Dc1_lo), ki_hi = ki_from_dose(Dc1_hi),
              chi2_par = χ²p, DIR = ra.DIR, Dc0_max = maximum(ra.Dc0),
              wform, p, p_err, chi2_anti = χ²a, chi2_anti_pure = χ²pure,
              n_par = length(par.I), n_anti = length(anti.I), I_fit, dD)
end

# =============================================================================
# 9.5  SG0 field integral BL0(I) from the measured profile, [135]
# =============================================================================
#
# The SG0 dose depends on the field only through its integral along the beam,
#
#     BL0(I) = ∫ |B0(y, I)| dy,     B0(y, I) = Bc(I) h(y),
#
# with h(y) the normalized profile from Chapter 1 and Bc(I) the center field per current
# (Section 8 calibration or the note's direct scans). Because h ≥ 0, the integral factorizes:
#
#     BL0(I) = |Bc(I)| L_eff,       L_eff = ∫ h(y) dy.
#
# So the profile enters only through one number, L_eff (≈ 80 mm), and the current dependence
# only through Bc(I).

"""
    h0(y) -> h

Normalized SG0 field profile along the beam axis, h(y) = B0(y) / B0(0), with y in mm measured
from the magnet center: the Section 5 global half-sigmoid (parameters `G.p`), mirrored about
y = 0 as in Section 6:

    h(y) = A / (1 + (|y| / y50)^n),     A = 0.9929, y50 = 31.93 mm, n = 2.759.

Note that h(0) = A = 0.9929, not exactly 1 (the fit leaves a 0.7 % offset at the center).
Used here for the numerical check of L_eff.
"""
h0(y) = fullprofile(y, G.p)

"""
    leff_mm(p) -> L_eff (mm)

Effective length of the SG0 field, L_eff = ∫_{-∞}^{∞} h(y) dy, in closed form for the mirrored
half-sigmoid with parameters p = [A (relative), y50 (mm), n]:

    L_eff = 2 A y50 (π/n) / sin(π/n)        (n > 1).

Derivation: the substitution t = (y/y50)^n turns ∫_0^∞ dy / (1 + (y/y50)^n) into a Beta
function, y50 Γ(1 + 1/n) Γ(1 - 1/n) = y50 (π/n) / sin(π/n). The factor 2 adds the mirrored
entrance side. The integral converges only for n > 1, since the tail falls as y^-n.

Caveats: about 10 % of L_eff comes from the fitted tail beyond the last Hall reading (77.47 mm),
and the entrance half is the mirror image, not a measurement.
"""
leff_mm(p) = 2 * p[1] * p[2] * (π / p[3]) / sin(π / p[3])

L_eff  = leff_mm(G.p)                         # global fit (Section 5), mm: the value used
L_sets = [leff_mm(f.p) for f in F]            # per-current fits (Section 4): spread, 79.2-79.9 mm
L_num  = let y = -2000:0.05:2000              # Riemann sum of h0 over ±2 m: closed-form check
    step(y) * sum(h0, y)                      # (short by the tail beyond ±2 m, ≈ 0.02 mm)
end

"""
    BL0_lin(I), BL0_quad(I), BL0_β0(I) -> BL0 (T m)

SG0 field integral BL0(I) = |Bc(I)| L_eff at SG0 coil current `I` (A), for three models of the
center field Bc(I):

    BL0_lin    Hall map, linear fit (Section 8):      Bc = k I,          k = cal_lin.c[2] (G/A)
    BL0_quad   Hall map, quadratic fit (Section 8):   Bc = c1 I + c2 I², c1, c2 = cal_quad.c[2:3]
    BL0_β0     direct SG0 scans (note, [19]):          Bc = β0 I,         β0 = 0.031895 T/A

Only the coil part of the center field is kept: the intercept (≈ 3.7 G, the field the fit
extrapolates to I = 0) is dropped, as in [140], so that BL0(0) = 0 and Dc0(0) = 0, matching the
data, which are referenced to I0 = 0.
Units: L_eff is in mm, so 1 G mm = 1e-7 T m (Hall map in G) and 1 T mm = 1e-3 T m (β0 in T/A).
The absolute value makes BL0 ≥ 0, as in the definition ∫ |B0| dy.

The Hall-map slope (431.6 G/A) and the direct-scan slope (318.9 G/A) differ by 35 %. This is
the dominant systematic of k_i, which is why all three models are fitted.
"""
BL0_lin(I)  = 1e-7 * L_eff * abs(cal_lin.c[2] * I)
BL0_quad(I) = 1e-7 * L_eff * abs(cal_quad.c[2] * I + cal_quad.c[3] * I^2)
BL0_β0(I)   = 1e-3 * L_eff * abs(β0 * I)

# The SG0 field models fitted in 9.6, each with its label for the table and Fig. 10
FIELD_MODELS = [(name = "Hall map, linear cal.",    BL0 = BL0_lin),
                (name = "Hall map, quadratic cal.", BL0 = BL0_quad),
                (name = "direct-scan slope β0",     BL0 = BL0_β0)]

# =============================================================================
# 9.6  Fits and summary
# =============================================================================
#
# Printout, in three blocks:
# - CQD inputs: the derived constants of 9.1, and the geometric DIR with the deflection z_max
#   that a fully aligned atom would reach on the camera;
# - SG0 field integral: L_eff (closed form, numerical check, per-current spread), BL0 per ampere
#   for the Hall map and the direct scans, and the dose ratio Dc0/Dc1 = BL0/BL1 at 1 A;
# - fit results, one row per field model in FIELD_MODELS (all with wform = :lin):
#     Dc1            best SG1 dose (step 1)
#     k_i            best value and its Δχ² = 1 interval
#     χ² par         parallel χ² / number of points (one fitted parameter)
#     DIR            drift ratio solved from the baseline, to compare with the geometric value
#     Dc0 max        largest SG0 dose reached by the data (at 3.64 A)
#     κ              retained-label weight slope (step 2) ± its curvature error
#     χ² anti        antiparallel χ² / points, with the fitted weight
#     χ² pure CQD    antiparallel χ² with w = 0 (strict mirror-and-rebranch rule)
#   A note is appended when the Dc1 scan minimum sits at an edge.

println("\nSection 9 - CQD fit with the measured SG0 field")
println("  CQD inputs")
@printf("    v_a = %.2f m/s    T1 = %.2f µs    γe = %.5e rad/(s T)\n",
        CQD.v_a, 1e6 * CQD.T1, CQD.γe)
@printf("    BL1 = B1 y_SG = %.4e T m    (B1 = %.3f mT, y_SG = %.0f mm)\n",
        CQD.BL1, 1e3 * B1, 1e3 * y_SG)
@printf("    Kz = %.3f µm    KΔ_pix = 2 Kz / d_pix = %.4f px\n", 1e6 * CQD.Kz, CQD.KΔ_pix)
@printf("    geometric DIR = y_SGToScreen / y_SG = %.4f  ->  z_max = Kz (DIR + 1/2) = %.1f µm = %.2f px\n",
        CQD.DIR_geom, 1e6 * CQD.zmax_geom, CQD.zmax_geom / CAM_PIXELSIZE)
println("  SG0 field integral")
@printf("    L_eff = ∫ h dy = %.2f mm    (numerical %.2f mm;  per-current fits %.1f-%.1f mm)\n",
        L_eff, L_num, extrema(L_sets)...)
@printf("    BL0 per ampere:  Hall map %.4e T m/A (k = %.1f G/A),  direct scans %.4e T m/A (β0 = %.1f G/A)\n",
        BL0_lin(1.0), cal_lin.c[2], BL0_β0(1.0), 1e4 * β0)
@printf("    Dc0/Dc1 = BL0/BL1 at 1 A:  Hall map %.3f,  direct scans %.3f\n",
        BL0_lin(1.0) / CQD.BL1, BL0_β0(1.0) / CQD.BL1)

# One two-step fit per SG0 field model (the slow part of the script)
cqd_results = [fit_cqd(curves, m.BL0) for m in FIELD_MODELS]

println("  Fit results  (DIR: solved from the $(Int(CQD.Dsep0_px))-px baseline; geometric value ",
        round(CQD.DIR_geom; digits = 3), ")")
@printf("    %-25s %7s %22s %9s %6s %7s %17s %9s %13s\n", "Bc(I)", "Dc1", "k_i (Δχ² = 1)",
        "χ² par", "DIR", "Dc0 max", "κ", "χ² anti", "χ² pure CQD")
for (m, r) in zip(FIELD_MODELS, cqd_results)
    @printf("    %-25s %7.4f %8.3g (%.3g-%.3g) %5.2f/%2d %6.3f %7.3f %8.4f ± %.4f %6.2f/%2d %13.1f%s\n",
            m.name, r.Dc1, r.ki, r.ki_lo, r.ki_hi, r.chi2_par, r.n_par, r.DIR, r.Dc0_max,
            r.p, r.p_err, r.chi2_anti, r.n_anti, r.chi2_anti_pure,
            r.Dc1_at_edge ? "   (Dc1 at the scan edge)" : "")
end

# =============================================================================
# 9.7  Figure: data with the fitted CQD curves
# =============================================================================

"""
    plot_cqd_fits(curves, models, results) -> Figure

Fig. 10: the measured separation changes vs. SG0 current with the fitted CQD curves.

- Data: both sets as in Fig. 7, drawn by `separation!` (parallel red diamonds, antiparallel
  blue squares, with 1σ error bars and the dashed zero line).
- Model: for each field model in `models` (with its fit in `results`, same order), one line
  style for both its curves, the parallel one (rising) and the antiparallel one (falling):
  solid black = Hall map linear, dashed black = Hall map quadratic, dotted gray = direct-scan β0.
  The legend therefore lists line styles (field models), merged so that each appears once,
  next to the two data sets.
- Axes: current 0-3.8 A, separation range SEP_YLIMS with 1-px ticks.
Only three line styles are defined: a fourth model would be dropped silently by `zip`.
"""
function plot_cqd_fits(curves, models, results)
    fig = Figure(size = (900, 600))
    ax  = Axis(fig[1, 1]; xlabel = L"I\ \mathrm{(A)}", ylabel = SEP_YLABEL, xticks = 0:0.5:4,
               yticks = SEP_YLIMS[1]:SEP_YLIMS[2],
               title = "CQD fit with the measured SG0 field integral")
    styles = [(color = :black, linestyle = :solid),
              (color = :black, linestyle = :dash),
              (color = INK2,   linestyle = :dot)]
    for (m, r, st) in zip(models, results, styles)
        lines!(ax, r.I_fit, r.dD.par;  st..., linewidth = 2, label = m.name)   # labelled once
        lines!(ax, r.I_fit, r.dD.anti; st..., linewidth = 2)
    end
    separation!(ax, curves, curves[1].I)                # data on top of the model lines
    axislegend(ax; position = :lt, framevisible = false, merge = true)
    xlims!(ax, -0.1, 3.8)
    return fig
end

fig10 = plot_cqd_fits(curves, FIELD_MODELS, cqd_results)
display(fig10)
savefig(fig10, "SG0_fig10_cqd_fit")                     # figures/SG0_fig10_cqd_fit.{pdf,png}

println("\nFigures written to ", abspath(FIGDIR))
