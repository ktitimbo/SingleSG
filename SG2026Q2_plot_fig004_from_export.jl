# Plot the F=1 peak position vs SG current, vs magnetic-field gradient and vs magnetic
# field, data + calibrated models, for every directory stored in an SG_fit_export.jld2
# file written by SG2026Q2_fitting_ki_zinit_summary_jld2.jl, plus summary figures of all
# directories together.
#
# Units in the file: current A, field T, gradient T/m, positions mm, ki ×1e-6.
# Figures are saved in <run folder>/plots/, once per format in FIG_EXTS (<ext> = png, pdf, svg):
#   <run folder>/plots/<dir>/<dir>_vsI.<ext>          vs current: data, existing models, CQD
#   <run folder>/plots/<dir>/<dir>_vsI_noCQD.<ext>    vs current: data and existing models only
#   <run folder>/plots/<dir>/<dir>_vsG.<ext>          vs field gradient
#   <run folder>/plots/<dir>/<dir>_vsB.<ext>          vs magnetic field
#   <run folder>/plots/summary_all.<ext>                 every point of every directory + CQD (median fit parameters)
#   <run folder>/plots/summary_points_mean.<ext>         mean ± std of the points per current
#   <run folder>/plots/summary_points_median.<ext>       median ± NMAD of the points per current
#   <run folder>/plots/summary_stacked_fitmedian.<ext>   one 3:1 panel per directory, CQD with the median fit parameters
#   <run folder>/plots/summary_stacked_fitmean.<ext>     the same, CQD with the mean fit parameters

using JLD2
using CairoMakie
using Statistics          # mean, median, std for the points mean / median summaries

# ---- settings -------------------------------------------------------------------
# Run to plot: a folder data_studies/FIT2026_per_round_fit_<TIMESTAMP>/ (TIMESTAMP = yyyymmddTHHMMSSsss).
#   RUN_FOLDER = nothing                                    → the most recent run
#   RUN_FOLDER = "FIT2026_per_round_fit_20261125T114053197" → that run
RUN_FOLDER  = nothing
RUN_PREFIX  = "FIT2026_per_round_fit_"
FIG_EXTS    = ["png", "pdf", "svg"]   # every figure is saved once in each of these formats
I_LINE_FRAC = 0.98                    # model curves start at I_LINE_FRAC · I_THRESHOLD
Z_LIMS      = (0.02, 2.5)             # (mm) y range of every peak-position axis, the same for every directory and figure
G_LIMS      = (4, 300)                # (T/m) x range of the plots vs field gradient
B_LIMS      = (0.01, 0.8)             # (T)   x range of the plots vs magnetic field
# I_LIMS (x range of every plot vs current) is set below from I_THRESHOLD: 0.95·I_THRESHOLD … 1.05 A
SHOW_DIR_LABEL  = false               # true → directory code name (bold) in the bottom-right corner of every per-directory plot
DIR_LABEL_COLOR = :gray65             # its colour (light gray; lower number = darker)
DIR_LABEL_SIZE  = 18                  # its font size
# model curves: colour, line width and style
QM_LINE      = (color = :blue, linewidth = 1.75, linestyle = :dash)    # existing models (QM), per directory
CQD_LINE     = (color = :red,  linewidth = 2)                          # CQD fit, per directory
CQD_SUM_LINE = (color = :red,  linewidth = 2.5)                        # CQD with the median fit parameters, summary figures
GLOBAL_LINE  = (color = :dodgerblue, linewidth = 2.5)                  # CQD with the median / mean fit parameters, stacked figures
OWN_LINE     = (color = :red, linewidth = 4, linestyle = :dot)         # the directory's own CQD fit, stacked figures

# ---- export file, threshold and output folder ------------------------------------
# (TIMESTAMP has a fixed width, so the alphabetical maximum is the newest run)
run_folders = filter(d -> startswith(d, RUN_PREFIX) && isdir(joinpath(@__DIR__, "data_studies", d)),
                     readdir(joinpath(@__DIR__, "data_studies")))
isempty(run_folders) && error("No $(RUN_PREFIX)<TIMESTAMP> folder in data_studies")
RUN_DIR     = joinpath(@__DIR__, "data_studies", RUN_FOLDER === nothing ? maximum(run_folders) : RUN_FOLDER)
EXPORT_PATH = joinpath(RUN_DIR, "SG_fit_export.jld2")
println("using ", EXPORT_PATH)
PLOT_DIR    = joinpath(RUN_DIR, "plots")          # every figure of this script goes here (summaries) or in plots/<dir>/
mkpath(PLOT_DIR)

f = jldopen(EXPORT_PATH, "r")
directories = f["meta/directories"]
I_THRESHOLD = f["settings/I_threshold_A"]          # (A) lowest current used in the fits, as stored by the fitting script
I_LINE_LO   = I_LINE_FRAC * I_THRESHOLD            # model curves start here
I_LIMS      = (0.95 * I_THRESHOLD, 1.05)           # (A) x range of every plot vs current
# current axis of every plot vs current: ticks labelled in mA, minor ticks at 1…9 × 10ⁿ A
# (IntervalsBetween would split the custom major ticks unevenly)
I_TICKS       = ([0.020, 0.040, 0.080, 0.100, 0.200, 0.400, 0.800, 1.000], ["20", "40", "80", "100", "200", "400", "800", "1000"])
MINOR_DECADE_TICKS = vcat((1:9) .* 1e-3, (1:9) .* 1e-2, (1:9) .* 1e-1, [1.0])   # also used for the field axis (T)

CairoMakie.activate!(type = "png", px_per_unit = 3)
set_theme!(theme_latexfonts())

## ---- per directory: vs SG current (<dir>_vsI) --------------------------------------

for d in directories
    g = "directories/" * d * "/"

    # measured points
    I  = f[g * "current"]
    δI = f[g * "current_Err"]
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_Err"]

    # calibrated model curves (dense grid); keep only I ≥ I_LINE_LO and z > 0 (log axes)
    I_qm  = f[g * "qm/current"];  z_qm  = f[g * "qm/zf1_position"]
    I_cqd = f[g * "cqd/current"]; z_cqd = f[g * "cqd/zf1_position"]
    keep_qm  = (I_qm  .>= I_LINE_LO) .& (z_qm  .> 0)
    keep_cqd = (I_cqd .>= I_LINE_LO) .& (z_cqd .> 0)

    fig = Figure(size = (1200, 600))
    ax  = Axis(fig[1, 1];
        xscale = log10, yscale = log10,
        xlabel = "SG current (mA)", ylabel = L"Peak position $(\mathrm{mm})$",
        xticks = I_TICKS,
        yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xminorticks = MINOR_DECADE_TICKS,
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

    lines!(ax, I_qm[keep_qm], z_qm[keep_qm]; QM_LINE...,
        label = "Existing models")
    lines!(ax, I_cqd[keep_cqd], z_cqd[keep_cqd]; CQD_LINE...,
        label = "Coquantum dynamics")

    xlims!(ax, I_LIMS...)
    ylims!(ax, Z_LIMS...)              # fixed y range (mm): Z_LIMS in settings
    SHOW_DIR_LABEL && text!(ax, 0.985, 0.03; text = d, space = :relative, align = (:right, :bottom),
        font = :bold, color = DIR_LABEL_COLOR, fontsize = DIR_LABEL_SIZE)   # directory code name, bottom right
    axislegend(ax; position = :lt, labelsize = 16,
        framevisible = false, backgroundcolor = :transparent,
        patchsize = (45, 12))          # (width, height) of the legend symbols: longer line samples
    display(fig)

    outdir = joinpath(PLOT_DIR, d)
    mkpath(outdir)
    for ext in FIG_EXTS                 # png, pdf and svg (FIG_EXTS in settings)
        save(joinpath(outdir, "$(d)_vsI.$(ext)"), fig)
        println("saved ", joinpath(outdir, "$(d)_vsI.$(ext)"))
    end
end

## ---- per directory: vs SG current without the CQD curve (<dir>_vsI_noCQD) ---------
for d in directories
    g = "directories/" * d * "/"

    # measured points
    I  = f[g * "current"]
    δI = f[g * "current_Err"]
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_Err"]

    # calibrated QM curve (dense grid); keep only I ≥ I_LINE_LO and z > 0 (log axes)
    I_qm  = f[g * "qm/current"];  z_qm  = f[g * "qm/zf1_position"]
    keep_qm  = (I_qm  .>= I_LINE_LO) .& (z_qm  .> 0)

    fig = Figure(size = (1200, 600))
    ax  = Axis(fig[1, 1];
        xscale = log10, yscale = log10,
        xlabel = "SG current (mA)", ylabel = L"Peak position $(\mathrm{mm})$",
        xticks = I_TICKS,
        yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xminorticks = MINOR_DECADE_TICKS,
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

    lines!(ax, I_qm[keep_qm], z_qm[keep_qm]; QM_LINE...,
        label = "Existing models")

    xlims!(ax, I_LIMS...)
    ylims!(ax, Z_LIMS...)              # fixed y range (mm): Z_LIMS in settings
    SHOW_DIR_LABEL && text!(ax, 0.985, 0.03; text = d, space = :relative, align = (:right, :bottom),
        font = :bold, color = DIR_LABEL_COLOR, fontsize = DIR_LABEL_SIZE)   # directory code name, bottom right
    axislegend(ax; position = :lt, labelsize = 16,
        framevisible = false, backgroundcolor = :transparent,
        patchsize = (45, 12))          # (width, height) of the legend symbols: longer line samples
    display(fig)

    outdir = joinpath(PLOT_DIR, d)
    mkpath(outdir)
    for ext in FIG_EXTS                 # png, pdf and svg (FIG_EXTS in settings)
        save(joinpath(outdir, "$(d)_vsI_noCQD.$(ext)"), fig)
        println("saved ", joinpath(outdir, "$(d)_vsI_noCQD.$(ext)"))
    end
end

## ---- per directory: vs magnetic-field gradient (<dir>_vsG) ---------------------------
for d in directories
    g = "directories/" * d * "/"

    # measured points
    G  = f[g * "gradient"]
    δG = f[g * "gradient_Err"]
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_Err"]

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
        yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xminorticks = vcat((1:9) .* 1e-1, (1:9) .* 1e0, (1:9) .* 1e1, (1:9) .* 1e2,  [1.0]),
        yminorticks = IntervalsBetween(9),
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,              # minor grid lines
        xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),   # lighter than the major grid
        xlabelsize = 20, ylabelsize = 20, xticklabelsize = 18, yticklabelsize = 18)

    # error bars: the lower half is clipped so it stays > 0 on the log axes
    errorbars!(ax, G, z, min.(δz, 0.999 .* z), δz; color = :black, whiskerwidth = 5, linewidth = 1)
    errorbars!(ax, G, z, min.(δG, 0.999 .* G), δG; direction = :x, color = :black, whiskerwidth = 5, linewidth = 1)
    scatter!(ax, G, z; marker = :circle, markersize = 11, color = :white,
        strokecolor = :black, strokewidth = 1.5, label = "Experiment")   # strokewidth: marker outline thickness

    lines!(ax, G_qm[keep_qm], z_qm[keep_qm]; QM_LINE...,
        label = "Existing models")
    lines!(ax, G_cqd[keep_cqd], z_cqd[keep_cqd]; CQD_LINE...,
        label = "Coquantum dynamics")

    xlims!(ax, G_LIMS...)
    ylims!(ax, Z_LIMS...)              # fixed y range (mm): Z_LIMS in settings
    SHOW_DIR_LABEL && text!(ax, 0.985, 0.03; text = d, space = :relative, align = (:right, :bottom),
        font = :bold, color = DIR_LABEL_COLOR, fontsize = DIR_LABEL_SIZE)   # directory code name, bottom right
    axislegend(ax; position = :lt, labelsize = 16,
        framevisible = false, backgroundcolor = :transparent,
        patchsize = (45, 12))          # (width, height) of the legend symbols: longer line samples
    display(fig)

    outdir = joinpath(PLOT_DIR, d)
    mkpath(outdir)
    for ext in FIG_EXTS                 # png, pdf and svg (FIG_EXTS in settings)
        save(joinpath(outdir, "$(d)_vsG.$(ext)"), fig)
        println("saved ", joinpath(outdir, "$(d)_vsG.$(ext)"))
    end
end

## ---- per directory: vs magnetic field (<dir>_vsB) ------------------------------------
for d in directories
    g = "directories/" * d * "/"

    # measured points
    B  = f[g * "magnetic_field"]
    δB = f[g * "magnetic_field_Err"]
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_Err"]

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
        xminorticks = MINOR_DECADE_TICKS,
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

    lines!(ax, B_qm[keep_qm], z_qm[keep_qm]; QM_LINE...,
        label = "Existing models")
    lines!(ax, B_cqd[keep_cqd], z_cqd[keep_cqd]; CQD_LINE...,
        label = "Coquantum dynamics")

    xlims!(ax, B_LIMS...)
    ylims!(ax, Z_LIMS...)              # fixed y range (mm): Z_LIMS in settings
    SHOW_DIR_LABEL && text!(ax, 0.985, 0.03; text = d, space = :relative, align = (:right, :bottom),
        font = :bold, color = DIR_LABEL_COLOR, fontsize = DIR_LABEL_SIZE)   # directory code name, bottom right
    axislegend(ax; position = :lt, labelsize = 16,
        framevisible = false, backgroundcolor = :transparent,
        patchsize = (45, 12))          # (width, height) of the legend symbols: longer line samples
    display(fig)

    outdir = joinpath(PLOT_DIR, d)
    mkpath(outdir)
    for ext in FIG_EXTS                 # png, pdf and svg (FIG_EXTS in settings)
        save(joinpath(outdir, "$(d)_vsB.$(ext)"), fig)
        println("saved ", joinpath(outdir, "$(d)_vsB.$(ext)"))
    end
end

## ---- summary_all: all directories + CQD with the median fit values (vs SG current) ----
# one colour (viridis, without its pale-yellow end) and one marker shape per directory
n_dirs     = length(directories)
colors_dir = [get(cgrad(:viridis), t) for t in range(0.0, 0.85; length = max(n_dirs, 2))]
shapes_dir = [:circle, :rect, :diamond, :utriangle, :dtriangle, :pentagon, :hexagon, :star5, :ltriangle, :rtriangle]

# CQD curve with the medians of the fit parameters over the directories (global/median/cqd)
I_med   = f["global/median/cqd/current"]
z_med   = f["global/median/cqd/zf1_position"]
keep_med = (I_med .>= I_LINE_LO) .& (z_med .> 0)

fig = Figure(size = (1700, 900))
ax  = Axis(fig[1, 1];
    xscale = log10, yscale = log10,
    xlabel = "SG current (mA)", ylabel = L"Peak position $(\mathrm{mm})$",
    xticks = I_TICKS,
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xminorticks = MINOR_DECADE_TICKS,
    yminorticks = IntervalsBetween(9),
    xminorticksvisible = true, yminorticksvisible = true,
    xminorgridvisible = true, yminorgridvisible = true,
    xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),
    xlabelsize = 22, ylabelsize = 22, xticklabelsize = 17, yticklabelsize = 17)

for (j, d) in enumerate(directories)
    g  = "directories/" * d * "/"
    I  = f[g * "current"]
    z  = f[g * "zf1_position"]
    δz = f[g * "zf1_position_Err"]
    errorbars!(ax, I, z, min.(δz, 0.999 .* z), δz; color = (colors_dir[j], 0.8), whiskerwidth = 5, linewidth = 1)
    # open markers: transparent fill, the directory is told apart by stroke colour and shape
    scatter!(ax, I, z; marker = shapes_dir[mod1(j, length(shapes_dir))], markersize = 14,
        color = :transparent, strokecolor = colors_dir[j], strokewidth = 1.5, label = d)
end

lines!(ax, I_med[keep_med], z_med[keep_med]; CQD_SUM_LINE..., label = "Coquantum dynamics")

xlims!(ax, I_LIMS...)
ylims!(ax, Z_LIMS...)                  # fixed y range (mm): Z_LIMS in settings
axislegend(ax; position = :lt, labelsize = 16,
    framevisible = false, backgroundcolor = :transparent,
    patchsize = (45, 12))
display(fig)

for ext in FIG_EXTS                 # png, pdf and svg (FIG_EXTS in settings)
    save(joinpath(PLOT_DIR, "summary_all.$(ext)"), fig)
    println("saved ", joinpath(PLOT_DIR, "summary_all.$(ext)"))
end

## ---- summary_points_mean (mean ± std) and summary_points_median (median ± NMAD) of the points ----
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
for (z_c, z_s, lbl, fname) in ((z_mean,   z_std,  "mean ± std over $(n_dirs) directories",    "summary_points_mean"),
                               (z_median, z_nmad, "median ± NMAD over $(n_dirs) directories", "summary_points_median"))
    local fig = Figure(size = (1700, 900))      # local: a global `fig` already exists
    local ax  = Axis(fig[1, 1];
        xscale = log10, yscale = log10,
        xlabel = "SG current (mA)", ylabel = L"Peak position $(\mathrm{mm})$",
        xticks = I_TICKS,
        yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xminorticks = MINOR_DECADE_TICKS,
        yminorticks = IntervalsBetween(9),
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),
        xlabelsize = 22, ylabelsize = 22, xticklabelsize = 17, yticklabelsize = 17)

    band!(ax, I_grp, z_min, z_max; color = (:darkgreen, 0.25), label = "min–max over the directories")   # (colour, opacity)
    errorbars!(ax, I_grp, z_c, min.(z_s, 0.999 .* z_c), z_s; color = :black, whiskerwidth = 6, linewidth = 1.2)
    scatter!(ax, I_grp, z_c; marker = :circle, markersize = 12, color = :white,
        strokecolor = :black, strokewidth = 1.5, label = lbl)
    lines!(ax, I_med[keep_med], z_med[keep_med]; CQD_SUM_LINE..., label = "Coquantum dynamics")

    xlims!(ax, I_LIMS...)
    ylims!(ax, Z_LIMS...)              # fixed y range (mm): Z_LIMS in settings
    axislegend(ax; position = :lt, labelsize = 16,
        framevisible = false, backgroundcolor = :transparent,
        patchsize = (45, 12))
    display(fig)

    for ext in FIG_EXTS                 # png, pdf and svg (FIG_EXTS in settings)
        save(joinpath(PLOT_DIR, fname * ".$(ext)"), fig)
        println("saved ", joinpath(PLOT_DIR, fname * ".$(ext)"))
    end
end

## ---- summary_stacked_fitmedian / _fitmean: one panel per directory in a single column ----
# Shared current axis (tick labels only on the bottom panel), same y range in every
# panel, each panel 3:1 (width:height). Two figures, made by the same loop:
#   summary_stacked_fitmedian   CQD curve with the MEDIAN fit parameters (global/median/cqd)
#   summary_stacked_fitmean     CQD curve with the MEAN   fit parameters (global/mean/cqd)
# In both: the data of every directory as black open circles, the global CQD curve in
# solid blue, and the directory's own CQD fit (its kᵢ, z_init, scale) in dotted red.
PANEL_W = 1050                      # panel width (px); height = PANEL_W/3
for (stat, fname) in (("median", "summary_stacked_fitmedian"),
                      ("mean",   "summary_stacked_fitmean"))
    # CQD curve with the median or mean fit parameters over the directories
    I_glob    = f["global/$(stat)/cqd/current"]
    z_glob    = f["global/$(stat)/cqd/zf1_position"]
    keep_glob = (I_glob .>= I_LINE_LO) .& (z_glob .> 0)

    local fig = Figure(size = (1500, 420 * n_dirs))     # trimmed to the panels by resize_to_layout! below
    local axs = Axis[]
    for (j, d) in enumerate(directories)
        g  = "directories/" * d * "/"
        I  = f[g * "current"]
        z  = f[g * "zf1_position"]
        δz = f[g * "zf1_position_Err"]
        local ax = Axis(fig[j, 1];                  # local: a global `ax` already exists
            width = PANEL_W, height = PANEL_W / 3,      # panel width : height = 3 : 1
            xscale = log10, yscale = log10,
            xlabel = j == n_dirs ? "SG current (mA)" : "",
            xticks = I_TICKS,
            yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
            xminorticks = MINOR_DECADE_TICKS,
            yminorticks = IntervalsBetween(9),
            xminorticksvisible = true, yminorticksvisible = true,
            xminorgridvisible = true, yminorgridvisible = true,
            xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),
            xtickalign = 1, xminortickalign = 1,        # x ticks point inside: no room left for them between the panels
            xlabelsize = 22, xticklabelsize = 17, yticklabelsize = 17)
        # data: black open circles (the same in every panel)
        errorbars!(ax, I, z, min.(δz, 0.999 .* z), δz; color = :black, whiskerwidth = 5, linewidth = 1)
        scatter!(ax, I, z; marker = :circle, markersize = 14,
            color = :transparent, strokecolor = :black, strokewidth = 1.5)
        # CQD with the median / mean fit parameters: solid blue
        lines!(ax, I_glob[keep_glob], z_glob[keep_glob]; GLOBAL_LINE...)
        # this directory's own CQD fit, as reference: dotted red
        I_dir = f[g * "cqd/current"]
        z_dir = f[g * "cqd/zf1_position"]
        keep_dir = (I_dir .>= I_LINE_LO) .& (z_dir .> 0)
        lines!(ax, I_dir[keep_dir], z_dir[keep_dir]; OWN_LINE...)
        SHOW_DIR_LABEL && text!(ax, 0.985, 0.03; text = d, space = :relative, align = (:right, :bottom),
            font = :bold, color = DIR_LABEL_COLOR, fontsize = DIR_LABEL_SIZE)   # directory code name, bottom right
        push!(axs, ax)
    end
    linkaxes!(axs...)                                   # same x and y range in every panel
    for a in axs[1:end-1]                               # x tick labels only on the bottom panel
        hidexdecorations!(a; grid = false, minorgrid = false, ticks = false, minorticks = false)
    end
    xlims!(axs[end], I_LIMS...)
    ylims!(axs[end], Z_LIMS...)                         # fixed y range (mm): Z_LIMS in settings (linked: every panel)
    Label(fig[1:n_dirs, 0], L"Peak position $(\mathrm{mm})$"; rotation = π / 2, fontsize = 22)   # one y label for all panels
    rowgap!(fig.layout, 0)                              # panels touch: no space between them
    resize_to_layout!(fig)                              # figure size = panels + labels, no empty space
    display(fig)
    for ext in FIG_EXTS                 # png, pdf and svg (FIG_EXTS in settings)
        save(joinpath(PLOT_DIR, fname * ".$(ext)"), fig)
        println("saved ", joinpath(PLOT_DIR, fname * ".$(ext)"))
    end
end

##
close(f)
