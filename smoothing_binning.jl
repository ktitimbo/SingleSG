# Kelvin Titimbo, Xukun Lin, S. Suleyman Kahraman, and Lihong V. Wang
# California Institute of Technology
# July 2025

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
using CSV, DataFrames, DelimitedFiles
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
data_directories = ["20250814", "20250820", "20250825"];
magnification_factor = 1.2697 ;
# only load a few columns from each fw_data.csv
sel = [:I_coil_mA, :F1_z_centroid_mm, :F1_z_centroid_se_mm]; 

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
                header=["Run Label","Binning","Smoothing"],
                alignment=:c,
                border_crayon = crayon"blue bold",
                tf = tf_unicode_rounded,
                header_crayon = crayon"yellow bold",
                equal_columns_width= true,
    )

    key_labels = collect(keys(m))
    cols = palette(:darkrainbow, length(key_labels))
    fig=plot(title="Experimental Data : binning & spline smoothing factor",
    titlefontsize = 12)
    for (i,key) in enumerate(key_labels)
        plot!(fig,abs.(m[key][3][2:end,"I_coil_mA"]/1000), abs.(m[key][3][2:end,"F1_z_centroid_mm"]/magnification_factor), 
        yerror = m[key][3][2:end,"F1_z_centroid_se_mm"],
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
    )
    saveplot(fig,"bin_vs_smoothing_$(data_directory)")

end




println("Experiment analysis finished!")
alert("Experiment analysis finished!")




I_coils_ssk = 1e-3 * [0,2,4,6,8,10,12,16,22,32,44,58,74,94,114,134,154,174,195,215,236,260,285,310,340,380,420,460,500,550,600,750,850];
kis         = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
# N=4e6 
cqd_set1 = CSV.read("./simulation_data/results_CQD_20250819T150057.csv",DataFrame; header=false);   # n_bins = 1
cqd_set2 = CSV.read("./simulation_data/results_CQD_20250820T180324.csv",DataFrame; header=false);   # n_bins = 2
cqd_set3 = CSV.read("./simulation_data/results_CQD_20250821T174824.csv",DataFrame; header=false);   # n_bins = 4
cqd_set4 = CSV.read("./simulation_data/results_CQD_20250825T114325.csv",DataFrame; header=false);   # n_bins = 8
cqd_set5 = CSV.read("./simulation_data/results_CQD_20250825T100550.csv",DataFrame; header=false);   # n_bins = 2

qm_set1  = CSV.read("./simulation_data/results_QM_20250819T150057.csv",DataFrame; header=false);    # n_bins = 1
qm_set2  = CSV.read("./simulation_data/results_QM_20250820T180324.csv",DataFrame; header=false);    # n_bins = 2
qm_set3  = CSV.read("./simulation_data/results_QM_20250821T174824.csv",DataFrame; header=false);    # n_bins = 4
qm_set4  = CSV.read("./simulation_data/results_QM_20250825T114325.csv",DataFrame; header=false);    # n_bins = 8
qm_set5  = CSV.read("./simulation_data/results_QM_20250825T100550.csv",DataFrame; header=false);    # n_bins = 2




m_1 = DataReading.collect_fw_map(parent_folder; 
                                select=sel, 
                                filename="fw_data.csv", 
                                report_name="experiment_report.txt", 
                                sort_on=:binning, 
                                data_dir_filter=data_directories[1]);

m_2 = DataReading.collect_fw_map(parent_folder; 
                                select=sel, 
                                filename="fw_data.csv", 
                                report_name="experiment_report.txt", 
                                sort_on=:binning, 
                                data_dir_filter=data_directories[2]);

m_3 = DataReading.collect_fw_map(parent_folder; 
                                select=sel, 
                                filename="fw_data.csv", 
                                report_name="experiment_report.txt", 
                                sort_on=:binning, 
                                data_dir_filter=data_directories[3]);

key_run = ["20250820T171704","20250825T183729","20250828T130751"]

# CO-QUANTUM DYNAMICS
fig1 = plot(
    xlabel="Current (A)",
    ylabel=L"$z_{F_{1}}$ (mm)"
)
cols = palette(:thermal, length(kis))
for i in eachindex(kis)
    plot!(fig1, cqd_set2[3:end,1], abs.(cqd_set2[3:end,23+i]), line=(:solid, cols[i], 1), label=L"$k_{i} = %$(kis[i])\times 10^{-6}$" )
end
plot!(fig1, 
    xaxis=:log10,
    yaxis=:log10,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(1e-3,1.2),
    ylims=(1e-4,5),
    legend=:outerright,
    legend_title = L"sim $n=2$",
    size=(800,400),
    left_margin = 4mm,
    bottom_margin = 3mm,
)
plot!(fig1,abs.(m_1[key_run[1]][3][2:end,"I_coil_mA"]/1000), abs.(m_1[key_run[1]][3][2:end,"F1_z_centroid_mm"]/magnification_factor), 
    # yerror = sqrt(30)*m[key][3][3:end,"F1_z_centroid_se_mm"],
    label="Experiment $(data_directories[1]): n=$(m_1[key_run[1]][1]) | λ=$(m_1[key_run[1]][2])", 
    color=:black,
    marker=(:circle,:black,2),
    markerstrokewidth = 1,
    markerstrokecolor=:black
)
plot!(fig1,abs.(m_2[key_run[2]][3][2:end,"I_coil_mA"]/1000), abs.(m_2[key_run[2]][3][2:end,"F1_z_centroid_mm"]/magnification_factor), 
    # yerror = sqrt(30)*m[key][3][3:end,"F1_z_centroid_se_mm"],
    label="Experiment $(data_directories[2]): n=$(m_2[key_run[2]][1]) | λ=$(m_2[key_run[2]][2])", 
    color=:red,
    marker=(:circle,:red,2),
    markerstrokewidth = 1,
    markerstrokecolor=:red
)
plot!(fig1,abs.(m_3[key_run[3]][3][2:end,"I_coil_mA"]/1000), abs.(m_3[key_run[3]][3][2:end,"F1_z_centroid_mm"]/magnification_factor), 
    # yerror = sqrt(30)*m[key][3][3:end,"F1_z_centroid_se_mm"],
    label="Experiment $(data_directories[3]): n=$(m_3[key_run[3]][1]) | λ=$(m_3[key_run[3]][2])", 
    color=:blue,
    marker=(:circle,:blue,2),
    markerstrokewidth = 1,
    markerstrokecolor=:blue
)
plot!(fig1,data_JSF[:exp][:,1],data_JSF[:exp][:,2],
    label="Alexander's data",
    line=(:dash,:green, 2),
)
display(fig1)
saveplot(fig1,"bin_vs_smoothing_log")


fig2 = plot(
    xlabel="Current (A)",
    ylabel=L"$z_{F_{1}}$ (mm)"
)
cols = palette(:thermal, length(kis))
for i in eachindex(kis)
    plot!(fig2, cqd_set2[3:end,1], abs.(cqd_set2[3:end,23+i]), line=(:solid, cols[i], 1), label=L"$k_{i} = %$(kis[i])\times 10^{-6}$" )
end
plot!(fig2, 
    # xaxis=:log10,
    # yaxis=:log10,
    # xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(1e-3,1.2),
    ylims=(1e-4,2),
    legend=:outerright,
    legend_title = L"sim $n=2$",
    size=(800,400),
    left_margin = 4mm,
    bottom_margin = 3mm,
)
plot!(fig2,abs.(m_1[key_run[1]][3][2:end,"I_coil_mA"]/1000), abs.(m_1[key_run[1]][3][2:end,"F1_z_centroid_mm"]/magnification_factor), 
    # yerror = sqrt(30)*m[key][3][3:end,"F1_z_centroid_se_mm"],
    label="Experiment $(data_directories[1]): n=$(m_1[key_run[1]][1]) | λ=$(m_1[key_run[1]][2])", 
    color=:black,
    marker=(:circle,:black,2),
    markerstrokewidth = 1,
    markerstrokecolor=:black
)
plot!(fig2,abs.(m_2[key_run[2]][3][2:end,"I_coil_mA"]/1000), abs.(m_2[key_run[2]][3][2:end,"F1_z_centroid_mm"]/magnification_factor), 
    # yerror = sqrt(30)*m[key][3][3:end,"F1_z_centroid_se_mm"],
    label="Experiment $(data_directories[2]): n=$(m_2[key_run[2]][1]) | λ=$(m_2[key_run[2]][2])", 
    color=:red,
    marker=(:circle,:red,2),
    markerstrokewidth = 1,
    markerstrokecolor=:red
)
plot!(fig2,abs.(m_3[key_run[3]][3][2:end,"I_coil_mA"]/1000), abs.(m_3[key_run[3]][3][2:end,"F1_z_centroid_mm"]/magnification_factor), 
    # yerror = sqrt(30)*m[key][3][3:end,"F1_z_centroid_se_mm"],
    label="Experiment $(data_directories[3]): n=$(m_3[key_run[3]][1]) | λ=$(m_3[key_run[3]][2])", 
    color=:blue,
    marker=(:circle,:blue,2),
    markerstrokewidth = 1,
    markerstrokecolor=:blue
)
plot!(fig2,data_JSF[:exp][:,1],data_JSF[:exp][:,2],
    label="Alexander's data",
    line=(:dash,:green, 2),
)
display(fig2)
saveplot(fig2,"bin_vs_smoothing_lin")
