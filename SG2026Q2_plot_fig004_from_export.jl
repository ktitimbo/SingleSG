# Plot fig004 (F=1 peak position vs SG current), fig005 (vs magnetic-field gradient) and
# fig006 (vs magnetic field), data + calibrated models, for every directory stored in an
# SG_fit_export.jld2 file written by SG2026Q2_fitting_ki_zinit_summary_jld2.jl.
#
# Units in the file: current A, field T, gradient T/m, positions mm, ki ×1e-6.
# Figures are saved next to the export file:
#   <run folder>/<dir>/fig004_from_export.png   (vs current)
#   <run folder>/<dir>/fig005_from_export.png   (vs gradient)
#   <run folder>/<dir>/fig006_from_export.png   (vs magnetic field)

using JLD2
using CairoMakie

# ---- settings -------------------------------------------------------------------
EXPORT_PATH = joinpath(@__DIR__, "data_studies",
                       "FIT2026_ki_scale_zinit_dirs_nz2_l0.010_20261124T132534290", "SG_fit_export.jld2")
I_THRESHOLD = 0.020                 # (A) lowest current used in the fits
I_LINE_LO   = 0.95 * I_THRESHOLD    # model curves start here

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
        xlabel = "SG current(A)", ylabel = L"Peak position $(\mathrm{mm})$",
        xticks = ([0.020, 0.030, 0.050, 0.080, 0.100, 0.200, 0.300, 0.500, 0.800, 1.000], ["0.020", "0.030", "0.050", "0.080", "0.100", "0.200", "0.300", "0.500", "0.800", "1.000"]),
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

    xlims!(ax, 18e-3, 1.05)
    axislegend(ax; position = :lt, labelsize = 16,
        framevisible = false, backgroundcolor = :transparent,
        patchsize = (45, 12))          # (width, height) of the legend symbols: longer line samples
    display(fig)

    outdir = joinpath(dirname(EXPORT_PATH), d)
    mkpath(outdir)
    save(joinpath(outdir, "fig004_from_export.png"), fig)
    println("saved ", joinpath(outdir, "fig004_from_export.png"))
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
    save(joinpath(outdir, "fig005_from_export.png"), fig)
    println("saved ", joinpath(outdir, "fig005_from_export.png"))
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
    save(joinpath(outdir, "fig006_from_export.png"), fig)
    println("saved ", joinpath(outdir, "fig006_from_export.png"))
end

## ---- summary: all directories + CQD with the median fit values (vs SG current) ----
# one colour (viridis, without its pale-yellow end) and one marker shape per directory
n_dirs     = length(directories)
colors_dir = [get(cgrad(:viridis), t) for t in range(0.0, 0.85; length = max(n_dirs, 2))]
shapes_dir = [:circle, :rect, :diamond, :utriangle, :dtriangle, :pentagon, :hexagon, :star5, :ltriangle, :rtriangle]

# CQD curve with the medians over the directories
I_med   = f["global/cqd/current"]
z_med   = f["global/cqd/zf1_position"]
ki_med  = f["global/cqd/cqd_median_ki"]
s_med   = f["global/cqd/cqd_median_scale"]
zi_med  = f["global/cqd/cqd_median_zinit"]          # mm
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

save(joinpath(dirname(EXPORT_PATH), "fig_summary_all_directories_from_export.png"), fig)
println("saved ", joinpath(dirname(EXPORT_PATH), "fig_summary_all_directories_from_export.png"))

close(f)
