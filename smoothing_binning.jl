# Kelvin Titimbo
# California Institute of Technology
# August 2025

############## EXPERIMENT ANALYSIS PREAMBLE ##############
# Headless/Windows-safe GR: set before using Plots
if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "100"  # offscreen; avoids popup windows/crashes
end

# Plotting backend and general appearance settings
using Plots; gr()
# Set default plot aesthetics
const IN_NOTEBOOK = isdefined(Main, :IJulia);
Plots.default(
    show=IN_NOTEBOOK, dpi=800, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
# Data I/O and numerical tools
using LinearAlgebra
using Statistics, StatsBase, OrderedCollections
# Aesthetics and output formatting
using Colors, ColorSchemes
using Printf, LaTeXStrings, PrettyTables
using CSV, DataFrames, DelimitedFiles, JLD2
# Time-stamping/logging
using Dates
using Alert
const T_START = Dates.now()
# Custom modules
include("./Modules/MyExperimentalAnalysis.jl");
using .MyExperimentalAnalysis;
include("./Modules/DataReading.jl");
# Set the working directory to the current location
cd(@__DIR__) 
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSS");
const OUTDIR    = joinpath(@__DIR__, "analysis_data", "smoothing_binning")
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
# For Plots
FIG_EXT = "png"   # could be "pdf", "svg", etc.
SAVE_FIG = false
MyExperimentalAnalysis.SAVE_FIG = SAVE_FIG;
MyExperimentalAnalysis.FIG_EXT  = FIG_EXT;
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

# Previous experiment data for comparison
data_JSF = OrderedDict(
    :exp => hcat(
    [0.0200, 0.0300, 0.0500, 0.1500, 0.2000, 0.2500, 0.3500, 0.5000, 0.7500], #mA
    [0.0229, 0.0610, 0.1107, 0.3901, 0.5122, 0.6315, 0.8139, 1.1201, 1.5738]),
    :model => hcat(
    [0.0150, 0.0200, 0.0250, 0.0300, 0.0400, 0.0500, 0.0700, 0.1000, 0.1500, 0.2000, 0.2500, 0.3500, 0.5000, 0.7500], #mA
    [0.0409, 0.0566, 0.0830, 0.1015, 0.1478, 0.1758, 0.2409, 0.3203, 0.4388, 0.5433, 0.6423, 0.8394, 1.1267, 1.5288], #CQD
    [0.0179, 0.0233, 0.0409, 0.0536, 0.0883, 0.1095, 0.1713, 0.2487, 0.3697, 0.4765, 0.5786, 0.7757, 1.0655, 1.4630]) #QM
);


parent_folder = joinpath(@__DIR__, "analysis_data")
data_directories = ["20250814", "20250820", "20250825","20250919"];
magnification_factor = 1.2697 ;

n_runs = length(data_directories)
I_all  = Vector{Vector{Float64}}(undef, n_runs)
dI_all = Vector{Vector{Float64}}(undef, n_runs)

for (i, dir) in enumerate(data_directories)
    d   = load(joinpath(@__DIR__, dir, "data_processed.jld2"), "data")
    I_all[i]  = Vector{Float64}(d[:Currents])
    dI_all[i] = Vector{Float64}(d[:CurrentsError])
end

fig_Is = plot(
        title = "Coil Currents",
        legend = :bottomright,
        xgrid=false,
        gridalpha = 0.25,
        gridstyle = :dot,
        minorgridalpha = 0.05,
        tickfontsize=11,
        guidefontsize=14,
    );
cols = palette(:darkrainbow, n_runs)
for (idx,data_directory) in enumerate(data_directories)
    scatter!(fig_Is,
        idx .* ones(length(I_all[idx])), 
        I_all[idx],
        yerror=dI_all[idx],
        label=false,
        marker = (:circle, :white, 2.5),
        markerstrokecolor = cols[idx],
        markerstrokewidth = 1.5,)
end
plot!(fig_Is,
    ylim = (1e-3,1.05),
    xlim=(-1,n_runs+2),
    yaxis = (:log10, L"$I_{0} \ (\mathrm{A})$"),
    xticks = (1:length(data_directories), data_directories),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xminorticks = false,
    xrotation=65,
    bottom_margin=-2mm,
    left_margin = 6mm,
    size=(350,720)
)
display(fig_Is)
saveplot(fig_Is, "currents_sampled")


# only load a few columns from each fw_data.csv
sel = [:Icoil_A, :Icoil_error_A, :F1_z_centroid_mm, :F1_z_centroid_se_mm]; 

for data_directory in data_directories
    # Data Directory
    # data_directory = "20250825" ;
    
    
    m = DataReading.collect_fw_map(parent_folder; 
                                    select=sel, 
                                    filename="fw_data.csv", 
                                    report_name="experiment_report.txt", 
                                    sort_on=:binning, 
                                    data_dir_filter=data_directory)
    

    pretty_table(hcat(collect(keys(m)),
                        [v.binning   for v in values(m)],
                        [v.smoothing for v in values(m)]); 
                title = "Analysis for $(data_directory)",
                column_labels=["Run Label","Binning","Smoothing"],
                alignment=:c,
                style = TextTableStyle(
                        first_line_column_label = crayon"yellow bold",
                        table_border  = crayon"blue bold",
                        # column_label  = crayon"yellow bold",
                ),
                # border_crayon = crayon"blue bold",
                table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
                # header_crayon = crayon"yellow bold",
                equal_data_column_widths= true,
    )

    key_labels = collect(keys(m))
    cols = palette(:darkrainbow, length(key_labels))
    fig=plot(title="Experimental Data : binning & spline smoothing factor",
    titlefontsize = 12)
    for (i,key) in enumerate(key_labels)
        plot!(fig,m[key][3][2:end,"Icoil_A"], abs.(m[key][3][2:end,"F1_z_centroid_mm"]/magnification_factor), 
        xerror = m[key][3][2:end,"Icoil_error_A"],
        yerror = m[key][3][2:end,"F1_z_centroid_se_mm"]/magnification_factor,
        label="n=$(m[key][1]) | λ=$(m[key][2])", 
        color=cols[i],
        marker=(:circle,cols[i],2),
        markerstrokewidth = 1,
        markerstrokecolor=cols[i]
        )
    end
    plot!(fig,
        xlabel="Current (A)",
        ylabel=L"$z_{F_{1}}$ (mm)",
        xaxis=:log10,
        yaxis=:log10,
        xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xlims=(1e-3,1.2),
        ylims=(5e-4,5),
        size=(800,600),
        legend=:outerright,
        legend_columns=1,
        legendfontsize=8,
        foreground_color_legend = nothing,
        left_margin=3mm,
        legend_title = data_directory,
    )
    saveplot(fig,"bin_vs_smoothing_$(data_directory)")

    println("\n")

end

println("Experiment analysis finished!")
alert("Experiment analysis finished!")


m_sets = map(d -> DataReading.collect_fw_map(
                 parent_folder;
                 select=sel,
                 filename="fw_data.csv",
                 report_name="experiment_report.txt",
                 sort_on=:binning,
                 data_dir_filter=d
             ), data_directories)


# desired values
selected_bin = 2
selected_spl = 0.01



# exact match (safe for Int; Float uses == here)
key_run = Vector{Union{Nothing,String}}(undef, length(m_sets))
for (midx,ms) in enumerate(m_sets)
    keys_match = [k for (k, nt) in ms if nt.binning == selected_bin && nt.smoothing == selected_spl]
    key_run[midx] = isempty(keys_match) ? nothing : first(keys_match)
end

# ---------- common axis + style ----------
xticks_vals = 10.0 .^ (-6:-1); xticks_vals = vcat(xticks_vals, 1.0)
yticks_vals = 10.0 .^ (-6:-1); yticks_vals = vcat(yticks_vals, 1.0)
xtick_labels = [L"10^{%$k}" for k in -6:-1]; xtick_labels = vcat(xtick_labels, L"10^{0}")
ytick_labels = [L"10^{%$k}" for k in -6:-1]; ytick_labels = vcat(ytick_labels, L"10^{0}")

# ---------- experimental series (4 runs) ----------
# pack your inputs to avoid repetition
runs    = key_run
dirs    = data_directories
colors  = [:black, :red, :blue, :purple]

using Dierckx

spl = Spline1D(m_sets[1][runs[1]][3][!,"Icoil_A"], m_sets[1][runs[1]][3][!,"F1_z_centroid_mm"]; k=3, s=0.0, bc="extrapolate")   # k=3 cubic; s=0 exact interpolate, s>0 smoothing
i_xx = range(10e-3,0.775,length=100)
plot(m_sets[1][runs[1]][3][5:end,"Icoil_A"], m_sets[1][runs[1]][3][5:end,"F1_z_centroid_mm"], xaxis=:log10,)
plot!(i_xx,spl.(i_xx))


fig1 = plot(
    xlabel = "Current (A)",
    ylabel = L"$z_{F_{1}}$ (mm)",
    xaxis  = :log10,
    yaxis  = :log10,
    xticks = (xticks_vals, xtick_labels),
    yticks = (yticks_vals, ytick_labels),
    xlims  = (8e-3, 1.2),
    ylims  = (1e-4, 5.0),
    legend = :outerright,
    legend_title = L"sim $n=%$(selected_bin)$",
    size   = (900, 420),
    left_margin = 4mm,
    bottom_margin = 3mm,
)
for (j, (M, r, d, c)) in enumerate(zip(m_sets, runs, dirs, colors))
    # columns and transforms
    I_A   = M[r][3][2:end, "Icoil_A"]            # mA -> A, abs
    δI_A  = M[r][3][2:end, "Icoil_error_A"]
    z_mm  = M[r][3][2:end, "F1_z_centroid_mm"] ./ magnification_factor
    δz_mm = M[r][3][2:end, "F1_z_centroid_se_mm"] ./ magnification_factor

    # guard for log10 axes: filter out non-positive values
    I_A   = ifelse.(I_A .> 0, I_A, missing)
    z_mm  = ifelse.(z_mm .> 0, z_mm, missing)

    plot!(fig1, I_A, z_mm;
        xerror = δI_A,
        yerror = δz_mm,
        label = "Experiment $(d): n=$(M[r][1]) | λ=$(M[r][2])",
        marker = (:circle,c,3),
        markerstrokewidth = 1,
        markerstrokecolor = c,
        line = (:solid,c,1) # pure markers; change to :solid if you want lines
    )

    # If you later want y-error bars, uncomment and make sure the column exists:
    # yerr = sqrt(30) .* M[r][3][3:end, "F1_z_centroid_se_mm"] ./ magnification_factor
    # plot!(fig1, I_A, z_mm; yerror = yerr, label = "", color = c, lw = 0)
end
# ---------- Alexander's data ----------
plot!(fig1,
    data_JSF[:exp][:, 1],
    data_JSF[:exp][:, 2],
    label = "Alexander's data",
    line = (:dash, :green, 2),
)
display(fig1)
saveplot(fig1, "bin_vs_smoothing_log")   # use explicit extension; pdf/png/svg as you like


fig2 = plot(
    xlabel = "Current (A)",
    ylabel = L"$z_{F_{1}}$ (mm)",
    # xticks = (xticks_vals, xtick_labels),
    # yticks = (yticks_vals, ytick_labels),
    xlims  = (8e-3, 1.2),
    ylims  = (1e-4, 2.0),
    legend = :outerright,
    legend_title = L"sim $n=%$(selected_bin)$",
    size   = (900, 420),
    left_margin = 4mm,
    bottom_margin = 3mm,
)
for (j, (M, r, d, c)) in enumerate(zip(m_sets, runs, dirs, colors))
    # columns and transforms
    I_A   = M[r][3][2:end, "Icoil_A"]            # mA -> A, abs
    δI_A  = M[r][3][2:end, "Icoil_error_A"]
    z_mm  = M[r][3][2:end, "F1_z_centroid_mm"] ./ magnification_factor
    δz_mm = M[r][3][2:end, "F1_z_centroid_se_mm"] ./ magnification_factor

    # guard for log10 axes: filter out non-positive values
    I_A   = ifelse.(I_A .> 0, I_A, missing)
    z_mm  = ifelse.(z_mm .> 0, z_mm, missing)

    plot!(fig2, I_A, z_mm;
        label = "Experiment $(d): n=$(M[r][1]) | λ=$(M[r][2])",
        marker = (:circle,c,3),
        markerstrokewidth = 1,
        markerstrokecolor = c,
        line = (:solid,c,1) # pure markers; change to :solid if you want lines
    )

    # If you later want y-error bars, uncomment and make sure the column exists:
    # yerr = sqrt(30) .* M[r][3][3:end, "F1_z_centroid_se_mm"] ./ magnification_factor
    # plot!(fig1, I_A, z_mm; yerror = yerr, label = "", color = c, lw = 0)
end
# ---------- Alexander's data ----------
plot!(fig2,
    data_JSF[:exp][:, 1],
    data_JSF[:exp][:, 2],
    label = "Alexander's data",
    line = (:dash, :green, 2),
)
display(fig2)
saveplot(fig2, "bin_vs_smoothing_lin")   # use explicit extension; pdf/png/svg as you like

