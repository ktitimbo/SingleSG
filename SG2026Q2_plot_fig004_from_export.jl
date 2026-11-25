# Plot fig004 (F=1 peak position vs SG current), fig005 (vs magnetic-field gradient) and
# fig006 (vs magnetic field), data + calibrated models, for every directory stored in an
# SG_fit_export.jld2 file written by SG2026Q2_fitting_ki_zinit_summary_jld2.jl.
#
# Units in the file: current A, field T, gradient T/m, positions mm, ki ×1e-6.
# Figures are saved next to the export file:
#   <run folder>/<dir>/fig004_from_export.<FIG_EXT>   (vs current)
#   <run folder>/<dir>/fig005_from_export.<FIG_EXT>   (vs gradient)
#   <run folder>/<dir>/fig006_from_export.<FIG_EXT>   (vs magnetic field)
# and, for all directories together with the median-parameter CQD curve (vs current):
#   <run folder>/fig_summary_all_directories_from_export.<FIG_EXT>          every point
#   <run folder>/fig_summary_all_directories_dodge_from_export.<FIG_EXT>    every point, directories shifted sideways
#   <run folder>/fig_summary_all_directories_mean_from_export.<FIG_EXT>     mean ± std per current
#   <run folder>/fig_summary_all_directories_median_from_export.<FIG_EXT>   median ± NMAD per current
#   <run folder>/fig_summary_all_directories_stacked_from_export.<FIG_EXT>  one 3:1 panel per directory, shared x
#   <run folder>/fig_summary_all_directories_stacked_mean_from_export.<FIG_EXT>  the same + mean of the data in every panel

using JLD2
using CairoMakie
using Statistics          # mean, median, std for the summary versions 3a/3b

# ---- settings -------------------------------------------------------------------
EXPORT_PATH = joinpath(@__DIR__, "data_studies",
                       "FIT2026_ki_scale_zinit_dirs_nz2_l0.010_20261124T132534290", "SG_fit_export.jld2")
I_THRESHOLD = 0.020                 # (A) lowest current used in the fits
I_LINE_LO   = 0.98 * I_THRESHOLD    # model curves start here
FIG_EXT     = "png"               # extension of the saved figures: "png" | "pdf" | "svg"

CairoMakie.activate!(type = "png", px_per_unit = 3)
set_theme!(theme_latexfonts())

## ---- loop over the directories ---------------------------------------------------
f = jldopen(EXPORT_PATH, "r")
directories = f["meta/directories"]

for d in directories
    g = "directories/" * d * "/"

    # measured points
    I  = f[g * "current"]
    δI = f[g * "current_error"]
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_error"]

    # calibrated model curves (dense grid); keep only I ≥ I_LINE_LO and z > 0 (log axes)
    I_qm  = f[g * "qm/current"];  z_qm  = f[g * "qm/zf1_position"]
    I_cqd = f[g * "cqd/current"]; z_cqd = f[g * "cqd/zf1_position"]
    keep_qm  = (I_qm  .>= I_LINE_LO) .& (z_qm  .> 0)
    keep_cqd = (I_cqd .>= I_LINE_LO) .& (z_cqd .> 0)

    ki = f[g * "ki"]

    fig = Figure(size = (1200, 600))
    ax  = Axis(fig[1, 1];
        xscale = log10, yscale = log10,
        xlabel = "SG current (mA)", ylabel = L"Peak position $(\mathrm{mm})$",
        # xticks = ([0.020, 0.030, 0.050, 0.080, 0.100, 0.200, 0.300, 0.500, 0.800, 1.000], ["0.02", "0.03", "0.05", "0.08", "0.1", "0.2", "0.3", "0.5", "0.8", "1.0"]),
        # xticks = ([0.020, 0.040, 0.080, 0.100, 0.200, 0.400, 0.800, 1.000], ["0.02", "0.04", "0.08", "0.1", "0.2", "0.4", "0.8", "1.0"]),
        xticks = ([0.020, 0.040, 0.080, 0.100, 0.200, 0.400, 0.800, 1.000], ["20", "40", "80", "100", "200", "400", "800", "1000"]),
        yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xminorticks = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0]),
        # xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        # xticks = ([0.020, 0.030, 0.050, 0.080, 0.100, 0.200, 0.300, 0.500, 0.800, 1.000], ["0.020", "0.030",  "0.050", "0.080", "0.100", "0.200", "0.300", "0.500", "0.800", "1.000"]),
        # yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        # x minor ticks at 2…9 × 10ⁿ (IntervalsBetween would split the custom major ticks unevenly)
        # xminorticks = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0]),
        yminorticks = IntervalsBetween(9),
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,              # minor grid lines
        xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),   # lighter than the major grid
        xlabelsize = 20, ylabelsize = 20, xticklabelsize = 18, yticklabelsize = 18)

    # error bars: the lower half is clipped so it stays > 0 on the log axes
    errorbars!(ax, I, z, min.(δz, 0.999 .* z), δz; color = :black, whiskerwidth = 5, linewidth = 1)
    errorbars!(ax, I, z, min.(δI, 0.999 .* I), δI; direction = :x, color = :black, whiskerwidth = 5, linewidth = 1)
    scatter!(ax, I, z; marker = :circle, markersize = 11, color = :white,
        strokecolor = :black, strokewidth = 1.5, label = "Experiment")   # strokewidth: marker outline thickness

    lines!(ax, I_qm[keep_qm], z_qm[keep_qm]; color = :blue, linewidth = 1.75, linestyle = :dash,
        label = "Existing models")
    lines!(ax, I_cqd[keep_cqd], z_cqd[keep_cqd]; color = :red, linewidth = 2,
        label = "Coquantum dynamics")

    xlims!(ax, 19e-3, 1.05)
    axislegend(ax; position = :lt, labelsize = 16,
        framevisible = false, backgroundcolor = :transparent,
        patchsize = (45, 12))          # (width, height) of the legend symbols: longer line samples
    display(fig)

    outdir = joinpath(dirname(EXPORT_PATH), d)
    mkpath(outdir)
    save(joinpath(outdir, "fig004_from_export.$(FIG_EXT)"), fig)
    println("saved ", joinpath(outdir, "fig004_from_export.$(FIG_EXT)"))
end

for d in directories
    g = "directories/" * d * "/"

    # measured points
    I  = f[g * "current"]
    δI = f[g * "current_error"]
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_error"]

    # calibrated model curves (dense grid); keep only I ≥ I_LINE_LO and z > 0 (log axes)
    I_qm  = f[g * "qm/current"];  z_qm  = f[g * "qm/zf1_position"]
    I_cqd = f[g * "cqd/current"]; z_cqd = f[g * "cqd/zf1_position"]
    keep_qm  = (I_qm  .>= I_LINE_LO) .& (z_qm  .> 0)
    keep_cqd = (I_cqd .>= I_LINE_LO) .& (z_cqd .> 0)

    ki = f[g * "ki"]

    fig = Figure(size = (1200, 600))
    ax  = Axis(fig[1, 1];
        xscale = log10, yscale = log10,
        xlabel = "SG current (mA)", ylabel = L"Peak position $(\mathrm{mm})$",
        # xticks = ([0.020, 0.030, 0.050, 0.080, 0.100, 0.200, 0.300, 0.500, 0.800, 1.000], ["0.02", "0.03", "0.05", "0.08", "0.1", "0.2", "0.3", "0.5", "0.8", "1.0"]),
        # xticks = ([0.020, 0.040, 0.080, 0.100, 0.200, 0.400, 0.800, 1.000], ["0.02", "0.04", "0.08", "0.1", "0.2", "0.4", "0.8", "1.0"]),
        xticks = ([0.020, 0.040, 0.080, 0.100, 0.200, 0.400, 0.800, 1.000], ["20", "40", "80", "100", "200", "400", "800", "1000"]),
        yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xminorticks = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0]),
        # xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        # xticks = ([0.020, 0.030, 0.050, 0.080, 0.100, 0.200, 0.300, 0.500, 0.800, 1.000], ["0.020", "0.030",  "0.050", "0.080", "0.100", "0.200", "0.300", "0.500", "0.800", "1.000"]),
        # yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        # x minor ticks at 2…9 × 10ⁿ (IntervalsBetween would split the custom major ticks unevenly)
        # xminorticks = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0]),
        yminorticks = IntervalsBetween(9),
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,              # minor grid lines
        xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),   # lighter than the major grid
        xlabelsize = 20, ylabelsize = 20, xticklabelsize = 18, yticklabelsize = 18)

    # error bars: the lower half is clipped so it stays > 0 on the log axes
    errorbars!(ax, I, z, min.(δz, 0.999 .* z), δz; color = :black, whiskerwidth = 5, linewidth = 1)
    errorbars!(ax, I, z, min.(δI, 0.999 .* I), δI; direction = :x, color = :black, whiskerwidth = 5, linewidth = 1)
    scatter!(ax, I, z; marker = :circle, markersize = 11, color = :white,
        strokecolor = :black, strokewidth = 1.5, label = "Experiment")   # strokewidth: marker outline thickness

    lines!(ax, I_qm[keep_qm], z_qm[keep_qm]; color = :blue, linewidth = 1.75, linestyle = :dash,
        label = "Existing models")
    # lines!(ax, I_cqd[keep_cqd], z_cqd[keep_cqd]; color = :red, linewidth = 2,
    #     label = "Coquantum dynamics")

    xlims!(ax, 19e-3, 1.05)
    axislegend(ax; position = :lt, labelsize = 16,
        framevisible = false, backgroundcolor = :transparent,
        patchsize = (45, 12))          # (width, height) of the legend symbols: longer line samples
    display(fig)

    outdir = joinpath(dirname(EXPORT_PATH), d)
    mkpath(outdir)
    save(joinpath(outdir, "fig004_from_export.$(FIG_EXT)"), fig)
    println("saved ", joinpath(outdir, "fig004_from_export.$(FIG_EXT)"))
end



## ---- same plot vs the magnetic-field gradient (fig005) ---------------------------
for d in directories
    g = "directories/" * d * "/"

    # measured points
    G  = f[g * "gradient"]
    # gradient error: written by newer runs; older export files have none (no x error bars then)
    δG = haskey(f, g * "gradient_error") ? f[g * "gradient_error"] : zeros(length(G))
    I  = f[g * "current"]
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_error"]

    # calibrated model curves on the same grid as the current plot, converted to gradient
    I_qm  = f[g * "qm/current"];  G_qm  = f[g * "qm/gradient"];  z_qm  = f[g * "qm/zf1_position"]
    I_cqd = f[g * "cqd/current"]; G_cqd = f[g * "cqd/gradient"]; z_cqd = f[g * "cqd/zf1_position"]
    keep_qm  = (I_qm  .>= I_LINE_LO) .& (z_qm  .> 0)
    keep_cqd = (I_cqd .>= I_LINE_LO) .& (z_cqd .> 0)

    fig = Figure(size = (1200, 600))
    ax  = Axis(fig[1, 1];
        xscale = log10, yscale = log10,
        xlabel = "Magnetic field gradient (T/m)", ylabel = L"Peak position $(\mathrm{mm})$",
        xticks = ([4, 5, 10, 20, 50, 100, 200, 300], ["4", "5", "10", "20", "50", "100", "200", "300"]),
        # xminorticks = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0]),
        yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xminorticks = IntervalsBetween(9), yminorticks = IntervalsBetween(9),
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,              # minor grid lines
        xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),   # lighter than the major grid
        xlabelsize = 20, ylabelsize = 20, xticklabelsize = 18, yticklabelsize = 18)

    # error bars: the lower half is clipped so it stays > 0 on the log axes
    errorbars!(ax, G, z, min.(δz, 0.999 .* z), δz; color = :black, whiskerwidth = 5, linewidth = 1)
    errorbars!(ax, G, z, min.(δG, 0.999 .* G), δG; direction = :x, color = :black, whiskerwidth = 5, linewidth = 1)
    scatter!(ax, G, z; marker = :circle, markersize = 11, color = :white,
        strokecolor = :black, strokewidth = 1.5, label = "Experiment")   # strokewidth: marker outline thickness

    lines!(ax, G_qm[keep_qm], z_qm[keep_qm]; color = :blue, linewidth = 1.75, linestyle = :dash,
        label = "Existing models")
    lines!(ax, G_cqd[keep_cqd], z_cqd[keep_cqd]; color = :red, linewidth = 2,
        label = "Coquantum dynamics")

    xlims!(ax, 4, 300)
    axislegend(ax; position = :lt, labelsize = 16,
        framevisible = false, backgroundcolor = :transparent,
        patchsize = (45, 12))          # (width, height) of the legend symbols: longer line samples
    display(fig)

    outdir = joinpath(dirname(EXPORT_PATH), d)
    mkpath(outdir)
    save(joinpath(outdir, "fig005_from_export.$(FIG_EXT)"), fig)
    println("saved ", joinpath(outdir, "fig005_from_export.$(FIG_EXT)"))
end

## ---- same plot vs the magnetic field (fig006) ------------------------------------
for d in directories
    g = "directories/" * d * "/"

    # measured points
    B  = f[g * "magnetic_field"]
    # field error: written by newer runs; older export files have none (no x error bars then)
    δB = haskey(f, g * "magnetic_field_error") ? f[g * "magnetic_field_error"] : zeros(length(B))
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_error"]

    # calibrated model curves on the same grid as the current plot, converted to field
    I_qm  = f[g * "qm/current"];  B_qm  = f[g * "qm/magnetic_field"];  z_qm  = f[g * "qm/zf1_position"]
    I_cqd = f[g * "cqd/current"]; B_cqd = f[g * "cqd/magnetic_field"]; z_cqd = f[g * "cqd/zf1_position"]
    keep_qm  = (I_qm  .>= I_LINE_LO) .& (z_qm  .> 0)
    keep_cqd = (I_cqd .>= I_LINE_LO) .& (z_cqd .> 0)

    fig = Figure(size = (1200, 600))
    ax  = Axis(fig[1, 1];
        xscale = log10, yscale = log10,
        xlabel = "Magnetic field (T)", ylabel = L"Peak position $(\mathrm{mm})$",
        xticks = ([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8], ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "0.8"]),
        yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        # x minor ticks at 2…9 × 10ⁿ (IntervalsBetween would split the custom major ticks unevenly)
        xminorticks = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0]),
        yminorticks = IntervalsBetween(9),
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,              # minor grid lines
        xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),   # lighter than the major grid
        xlabelsize = 20, ylabelsize = 20, xticklabelsize = 18, yticklabelsize = 18)

    # error bars: the lower half is clipped so it stays > 0 on the log axes
    errorbars!(ax, B, z, min.(δz, 0.999 .* z), δz; color = :black, whiskerwidth = 5, linewidth = 1)
    errorbars!(ax, B, z, min.(δB, 0.999 .* B), δB; direction = :x, color = :black, whiskerwidth = 5, linewidth = 1)
    scatter!(ax, B, z; marker = :circle, markersize = 11, color = :white,
        strokecolor = :black, strokewidth = 1.5, label = "Experiment")   # strokewidth: marker outline thickness

    lines!(ax, B_qm[keep_qm], z_qm[keep_qm]; color = :blue, linewidth = 1.75, linestyle = :dash,
        label = "Existing models")
    lines!(ax, B_cqd[keep_cqd], z_cqd[keep_cqd]; color = :red, linewidth = 2,
        label = "Coquantum dynamics")

    xlims!(ax, 0.011, 0.8)          # (lower, upper) in T
    axislegend(ax; position = :lt, labelsize = 16,
        framevisible = false, backgroundcolor = :transparent,
        patchsize = (45, 12))          # (width, height) of the legend symbols: longer line samples
    display(fig)

    outdir = joinpath(dirname(EXPORT_PATH), d)
    mkpath(outdir)
    save(joinpath(outdir, "fig006_from_export.$(FIG_EXT)"), fig)
    println("saved ", joinpath(outdir, "fig006_from_export.$(FIG_EXT)"))
end

## ---- summary: all directories + CQD with the median fit values (vs SG current) ----
# one colour (viridis, without its pale-yellow end) and one marker shape per directory
n_dirs     = length(directories)
colors_dir = [get(cgrad(:viridis), t) for t in range(0.0, 0.85; length = max(n_dirs, 2))]
shapes_dir = [:circle, :rect, :diamond, :utriangle, :dtriangle, :pentagon, :hexagon, :star5, :ltriangle, :rtriangle]

# CQD curve with the medians over the directories
# (global/median/ — also global/wmean/ and global/mean/ in exports written after 2026-11-25)
I_med   = f["global/median/cqd/current"]
z_med   = f["global/median/cqd/zf1_position"]
ki_med  = f["global/median/cqd/ki"]
s_med   = f["global/median/cqd/scale"]
zi_med  = f["global/median/cqd/zinit"]          # mm
keep_med = (I_med .>= I_LINE_LO) .& (z_med .> 0)

fig = Figure(size = (1700, 900))
ax  = Axis(fig[1, 1];
    xscale = log10, yscale = log10,
    xlabel = "SG current(A)", ylabel = L"Peak position $(\mathrm{mm})$",
    xticks = ([0.020, 0.030, 0.050, 0.080, 0.100, 0.200, 0.300, 0.500, 0.800, 1.000], ["0.020", "0.030", "0.050", "0.080", "0.100", "0.200", "0.300", "0.500", "0.800", "1.000"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xminorticks = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0]),
    yminorticks = IntervalsBetween(9),
    xminorticksvisible = true, yminorticksvisible = true,
    xminorgridvisible = true, yminorgridvisible = true,
    xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),
    xlabelsize = 22, ylabelsize = 22, xticklabelsize = 17, yticklabelsize = 17)

for (j, d) in enumerate(directories)
    g  = "directories/" * d * "/"
    I  = f[g * "current"]
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_error"]
    errorbars!(ax, I, z, min.(δz, 0.999 .* z), δz; color = (colors_dir[j], 0.8), whiskerwidth = 5, linewidth = 1)
    # open markers: transparent fill, the directory is told apart by stroke colour and shape
    scatter!(ax, I, z; marker = shapes_dir[mod1(j, length(shapes_dir))], markersize = 14,
        color = :transparent, strokecolor = colors_dir[j], strokewidth = 1.5, label = d)
end

lines!(ax, I_med[keep_med], z_med[keep_med]; color = :red, linewidth = 2.5,
    # label = L"Coquantum dynamics: $\tilde{k}_{i} = %$(round(ki_med, sigdigits = 3))\times 10^{-6}$, $\tilde{z}_{\mathrm{init}} = %$(round(1e3 * zi_med, sigdigits = 3))\ \mathrm{\mu m}$, $\tilde{s} = %$(round(s_med, sigdigits = 4))$",
    label = "Coquantum dynamics",
    )

xlims!(ax, 18e-3, 1.1)
axislegend(ax; position = :lt, labelsize = 16,      # ~ : median over the directories
    framevisible = false, backgroundcolor = :transparent,
    patchsize = (45, 12))
display(fig)

save(joinpath(dirname(EXPORT_PATH), "fig_summary_all_directories_from_export.$(FIG_EXT)"), fig)
println("saved ", joinpath(dirname(EXPORT_PATH), "fig_summary_all_directories_from_export.$(FIG_EXT)"))

## ---- summary, version 1: same plot with each directory shifted slightly in current ----
# On a log axis, multiplying the currents by a constant shifts the points sideways by the
# same amount everywhere, so the directories sit next to each other instead of on top.
DODGE_STEP = 0.018                  # relative shift between neighbouring directories (1.8 %)

fig = Figure(size = (1700, 900))
ax  = Axis(fig[1, 1];
    xscale = log10, yscale = log10,
    xlabel = "SG current(A)", ylabel = L"Peak position $(\mathrm{mm})$",
    xticks = ([0.020, 0.030, 0.050, 0.080, 0.100, 0.200, 0.300, 0.500, 0.800, 1.000], ["0.020", "0.030", "0.050", "0.080", "0.100", "0.200", "0.300", "0.500", "0.800", "1.000"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xminorticks = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0]),
    yminorticks = IntervalsBetween(9),
    xminorticksvisible = true, yminorticksvisible = true,
    xminorgridvisible = true, yminorgridvisible = true,
    xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),
    xlabelsize = 22, ylabelsize = 22, xticklabelsize = 17, yticklabelsize = 17)

for (j, d) in enumerate(directories)
    g  = "directories/" * d * "/"
    I  = f[g * "current"] .* (1 + DODGE_STEP * (j - (n_dirs + 1) / 2))     # shifted current
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_error"]
    errorbars!(ax, I, z, min.(δz, 0.999 .* z), δz; color = (colors_dir[j], 0.8), whiskerwidth = 5, linewidth = 1)
    scatter!(ax, I, z; marker = shapes_dir[mod1(j, length(shapes_dir))], markersize = 14,
        color = :transparent, strokecolor = colors_dir[j], strokewidth = 1.5, label = d)
end

lines!(ax, I_med[keep_med], z_med[keep_med]; color = :red, linewidth = 2.5, label = "Coquantum dynamics")

xlims!(ax, 18e-3, 1.1)
axislegend(ax; position = :lt, labelsize = 16,
    framevisible = false, backgroundcolor = :transparent,
    patchsize = (45, 12))
display(fig)

save(joinpath(dirname(EXPORT_PATH), "fig_summary_all_directories_dodge_from_export.$(FIG_EXT)"), fig)
println("saved ", joinpath(dirname(EXPORT_PATH), "fig_summary_all_directories_dodge_from_export.$(FIG_EXT)"))

## ---- summary, versions 3a (mean ± std) and 3b (median ± NMAD) over the directories ----
# The directories are measured at almost, but not exactly, the same currents, so the points
# of all directories are sorted by current and grouped: a new group starts when the current
# is more than GROUP_TOL above the first current of the current group.
GROUP_TOL = 1.03                    # currents within 3 % are taken as the same set point

all_I = Float64[]; all_z = Float64[]
for d in directories
    append!(all_I, f["directories/" * d * "/current"])
    append!(all_z, f["directories/" * d * "/zf1_position"])
end
order = sortperm(all_I)
all_I = all_I[order]; all_z = all_z[order]

groups = Vector{Vector{Int}}()
for k in eachindex(all_I)
    if isempty(groups) || all_I[k] > GROUP_TOL * all_I[first(last(groups))]
        push!(groups, [k])
    else
        push!(last(groups), k)
    end
end

I_grp    = [mean(all_I[g]) for g in groups]               # mean current of the group
z_mean   = [mean(all_z[g]) for g in groups]
z_std    = [length(g) > 1 ? std(all_z[g]) : 0.0 for g in groups]
z_median = [median(all_z[g]) for g in groups]
z_nmad   = [1.4826 * median(abs.(all_z[g] .- median(all_z[g]))) for g in groups]   # robust spread
z_min    = [minimum(all_z[g]) for g in groups]
z_max    = [maximum(all_z[g]) for g in groups]

# one figure per statistic: (centre, spread, legend label, file name)
for (z_c, z_s, lbl, fname) in ((z_mean,   z_std,  "mean ± std over $(n_dirs) directories",    "fig_summary_all_directories_mean_from_export.$(FIG_EXT)"),
                               (z_median, z_nmad, "median ± NMAD over $(n_dirs) directories", "fig_summary_all_directories_median_from_export.$(FIG_EXT)"))
    local fig = Figure(size = (1700, 900))      # local: a global `fig` already exists
    local ax  = Axis(fig[1, 1];
        xscale = log10, yscale = log10,
        xlabel = "SG current(A)", ylabel = L"Peak position $(\mathrm{mm})$",
        xticks = ([0.020, 0.030, 0.050, 0.080, 0.100, 0.200, 0.300, 0.500, 0.800, 1.000], ["0.020", "0.030", "0.050", "0.080", "0.100", "0.200", "0.300", "0.500", "0.800", "1.000"]),
        yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xminorticks = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0]),
        yminorticks = IntervalsBetween(9),
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),
        xlabelsize = 22, ylabelsize = 22, xticklabelsize = 17, yticklabelsize = 17)

    band!(ax, I_grp, z_min, z_max; color = (:darkgreen, 0.25), label = "min–max over the directories")   # (colour, opacity)
    errorbars!(ax, I_grp, z_c, min.(z_s, 0.999 .* z_c), z_s; color = :black, whiskerwidth = 6, linewidth = 1.2)
    scatter!(ax, I_grp, z_c; marker = :circle, markersize = 12, color = :white,
        strokecolor = :black, strokewidth = 1.5, label = lbl)
    lines!(ax, I_med[keep_med], z_med[keep_med]; color = :red, linewidth = 2.5, label = "Coquantum dynamics")

    xlims!(ax, 18e-3, 1.1)
    axislegend(ax; position = :lt, labelsize = 16,
        framevisible = false, backgroundcolor = :transparent,
        patchsize = (45, 12))
    display(fig)

    save(joinpath(dirname(EXPORT_PATH), fname), fig)
    println("saved ", joinpath(dirname(EXPORT_PATH), fname))
end

## ---- summary, stacked version: one panel per directory in a single column ----------
# Shared current axis (tick labels only on the bottom panel), same y range in every
# panel, each panel 3:1 (width:height); the median-parameter CQD curve in every panel.
PANEL_W = 1050                      # panel width (px); height = PANEL_W/3
fig = Figure(size = (1500, 420 * n_dirs))           # trimmed to the panels by resize_to_layout! below
axs = Axis[]
for (j, d) in enumerate(directories)
    g  = "directories/" * d * "/"
    I  = f[g * "current"]
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_error"]

    local ax = Axis(fig[j, 1];                      # local: a global `ax` already exists
        width = PANEL_W, height = PANEL_W / 3,      # panel width : height = 3 : 1
        xscale = log10, yscale = log10,
        xlabel = j == n_dirs ? "SG current (mA)" : "",
        xticks = ([0.020, 0.040, 0.080, 0.100, 0.200, 0.400, 0.800, 1.000], ["20", "40", "80", "100", "200", "400", "800", "1000"]),
        yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xminorticks = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0]),
        yminorticks = IntervalsBetween(9),
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),
        xlabelsize = 22, xticklabelsize = 17, yticklabelsize = 17)

    errorbars!(ax, I, z, min.(δz, 0.999 .* z), δz; color = (colors_dir[j], 0.8), whiskerwidth = 5, linewidth = 1)
    scatter!(ax, I, z; marker = shapes_dir[mod1(j, length(shapes_dir))], markersize = 14,
        color = :transparent, strokecolor = colors_dir[j], strokewidth = 1.5, label = d)
    lines!(ax, I_med[keep_med], z_med[keep_med]; color = :blue, linewidth = 2.5, label = "Coquantum dynamics", linestyle=:solid)
    # this directory's own CQD fit (its kᵢ, z_init and scale), dashed vermillion
    I_dir = f[g * "cqd/current"]
    z_dir = f[g * "cqd/zf1_position"]
    keep_dir = (I_dir .>= I_LINE_LO) .& (z_dir .> 0)
    lines!(ax, I_dir[keep_dir], z_dir[keep_dir]; color = colorant"#D55E00", linewidth = 3, linestyle = :dash)
    push!(axs, ax)
end

linkaxes!(axs...)                                   # same x and y range in every panel
for ax in axs[1:end-1]                              # x tick labels only on the bottom panel
    hidexdecorations!(ax; grid = false, minorgrid = false, ticks = false, minorticks = false)
end
xlims!(axs[end], 18e-3, 1.1)
# y range just around the data of all directories (less empty space in each panel)
z_all = reduce(vcat, [f["directories/" * d * "/zf1_position"] for d in directories])
ylims!(axs[end], 0.8 * minimum(z_all), 1.25 * maximum(z_all))
Label(fig[1:n_dirs, 0], L"Peak position $(\mathrm{mm})$"; rotation = π / 2, fontsize = 22)   # one y label for all panels
rowgap!(fig.layout, 0)                              # panels touch: no space between them
resize_to_layout!(fig)                              # figure size = panels + labels, no empty space
display(fig)

save(joinpath(dirname(EXPORT_PATH), "fig_summary_all_directories_stacked_from_export.$(FIG_EXT)"), fig)
println("saved ", joinpath(dirname(EXPORT_PATH), "fig_summary_all_directories_stacked_from_export.$(FIG_EXT)"))

## ---- summary, stacked version + mean of the data -----------------------------------
# Same panels as above, and in every panel the mean over all directories at each current
# (I_grp, z_mean from versions 3a/3b) as black crosses, so each directory can be compared
# with the average of the runs.
fig = Figure(size = (1500, 420 * n_dirs))           # trimmed to the panels by resize_to_layout! below
axs = Axis[]
for (j, d) in enumerate(directories)
    g  = "directories/" * d * "/"
    I  = f[g * "current"]
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_error"]

    local ax = Axis(fig[j, 1];                      # local: a global `ax` already exists
        width = PANEL_W, height = PANEL_W / 3,      # panel width : height = 3 : 1
        xscale = log10, yscale = log10,
        xlabel = j == n_dirs ? "SG current (mA)" : "",
        xticks = ([0.020, 0.040, 0.080, 0.100, 0.200, 0.400, 0.800, 1.000], ["20", "40", "80", "100", "200", "400", "800", "1000"]),
        yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xminorticks = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0]),
        yminorticks = IntervalsBetween(9),
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),
        xlabelsize = 22, xticklabelsize = 17, yticklabelsize = 17)

    errorbars!(ax, I, z, min.(δz, 0.999 .* z), δz; color = (colors_dir[j], 0.8), whiskerwidth = 5, linewidth = 1)
    # raw data: open circles in every panel (only the colour changes per directory)
    scatter!(ax, I, z; marker = :circle, markersize = 18,
        color = :transparent, strokecolor = :black, strokewidth = 2.5, label = d)
    # lines!(ax, I_med[keep_med], z_med[keep_med]; color = :red, linewidth = 2.5, label = "Coquantum dynamics", linestyle=:solid)
    # this directory's own CQD fit (its kᵢ, z_init and scale), dashed vermillion
    I_dir = f[g * "cqd/current"]
    z_dir = f[g * "cqd/zf1_position"]
    keep_dir = (I_dir .>= I_LINE_LO) .& (z_dir .> 0)
    lines!(ax, I_dir[keep_dir], z_dir[keep_dir]; color = :blue, linewidth = 3, linestyle = :solid)
    lines!(ax, I_med[keep_med], z_med[keep_med]; color = :red, linewidth = 5.5, label = "Coquantum dynamics", linestyle=:dot)
    # mean of the data over all directories at each current
    # scatter!(ax, I_grp, z_mean; marker = :hline, markersize = 22, color = :red, label = "mean over the directories")   # :hline = horizontal bar ("−"); markersize = bar length
    push!(axs, ax)
end

linkaxes!(axs...)                                   # same x and y range in every panel
for ax in axs[1:end-1]                              # x tick labels only on the bottom panel
    hidexdecorations!(ax; grid = false, minorgrid = false, ticks = false, minorticks = false)
end
xlims!(axs[end], 18e-3, 1.1)
ylims!(axs[end], 0.8 * minimum(z_all), 1.25 * maximum(z_all))
Label(fig[1:n_dirs, 0], L"Peak position $(\mathrm{mm})$"; rotation = π / 2, fontsize = 22)   # one y label for all panels
rowgap!(fig.layout, 0)                              # panels touch: no space between them
resize_to_layout!(fig)                              # figure size = panels + labels, no empty space
display(fig)

save(joinpath(dirname(EXPORT_PATH), "fig_summary_all_directories_stacked_mean_from_export.$(FIG_EXT)"), fig)
println("saved ", joinpath(dirname(EXPORT_PATH), "fig_summary_all_directories_stacked_mean_from_export.$(FIG_EXT)"))


##
close(f)
