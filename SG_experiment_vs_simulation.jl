# Simulation of atom trajectories in the Stern–Gerlach experiment
# Kelvin Titimbo
# California Institute of Technology
# August 2025

using Plots; gr()
Plots.default(
    show=true, dpi=800, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
FIG_EXT = "png"   # could be "pdf", "svg", etc.
SAVE_FIG = true
# Aesthetics and output formatting
using Colors, ColorSchemes
using LaTeXStrings, Printf, PrettyTables
# Time-stamping/logging
using Dates
const T_START = Dates.now() ; # Timestamp start for execution timing
# Numerical tools
using LinearAlgebra, DataStructures
using Interpolations, Roots, Loess, Optim
using BSplineKit
using DSP
using LambertW, PolyLog
using StatsBase
using Random, Statistics, NaNStatistics, Distributions, StaticArrays
using Alert
# Data manipulation
using OrderedCollections
using DelimitedFiles, CSV, DataFrames, JLD2
# include("./Modules/MyPolylogarithms.jl");
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Set the working directory to the current location
cd(@__DIR__) ;
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSS");
const OUTDIR    = joinpath(@__DIR__, "data_studies", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
include("./Modules/TheoreticalSimulation.jl");


# import experimental data
data_exp        = load(joinpath("analysis_data","20250827T133507","profiles.jld2"))["profiles"]
data_exp_fw     = CSV.read(joinpath("analysis_data","20250827T133507","fw_data.csv"),DataFrame)
data_exp_mean   = CSV.read(joinpath("analysis_data","20250827T133507","mean_data.csv"),DataFrame)

centroid = mean((data_exp_mean[!,"F1_z_peak_mm"]+ data_exp_mean[!,"F2_z_peak_mm"])/2)

# import simulated data
data_simulation         = load(joinpath("simulation_data", "20250822T203747","qm_2000000_valid_particles_data.jld2"))["data"]
data_simulation_top     = load(joinpath("simulation_data", "20250822T203747","zmax_profiles_top_32x1.jld2"))["data"]
data_simulation_bottom  = load(joinpath("simulation_data", "20250822T203747","zmax_profiles_bottom_32x1.jld2"))["data"]


z_mm = 1e3*(6.5e-6 .* (1:2560) .- 6.5e-6 / 2)
# z_mm = (1e3 * 6.5e-6) .* ((1:2560) .- (2560 + 1)/2)


I_pairs = [ 1 1 ; 2 2 ; 3 4 ; 9 8 ; 12 11 ; 14 15 ; 18 20 ; 19 22 ; 21 27 ; 22 30 ]
new_width = 0.300
for row in eachrow(I_pairs)
    icoil_idx_exp = row[1]
    icoil_idx_sim = row[2]
    


    y_conv_f1 = TheoreticalSimulation.smooth_profile(data_simulation_bottom[icoil_idx_sim][:z_profile][:,1], data_simulation_bottom[icoil_idx_sim][:z_profile][:,2], new_width)
    y_conv_f2 = TheoreticalSimulation.smooth_profile(data_simulation_top[icoil_idx_sim][:z_profile][:,1], data_simulation_top[icoil_idx_sim][:z_profile][:,2], new_width)


    f1 = plot(data_simulation_bottom[icoil_idx_sim][:z_profile][:,1], data_simulation_bottom[icoil_idx_sim][:z_profile][:,2]/maximum(data_simulation_bottom[icoil_idx_sim][:z_profile][:,2]),
        label="Simulation raw", seriestype=:scatter, marker=(:circle, :white,1), markerstrokecolor=:grey60)
    plot!(f1, data_simulation_bottom[icoil_idx_sim][:z_profile][:,1], data_simulation_bottom[icoil_idx_sim][:z_profile][:,3]/maximum(data_simulation_bottom[icoil_idx_sim][:z_profile][:,3]),
        label=L"Simulation: $150\mathrm{\mu m}$")
    plot!(f1, data_simulation_bottom[icoil_idx_sim][:z_profile][:,1], y_conv_f1/maximum(y_conv_f1),
        label=L"Simulation: $%$(1e3*new_width)\mathrm{\mu m}$")
    plot!(f1, z_mm .-  centroid, data_exp[:F1_profile][icoil_idx_exp,:]/maximum(data_exp[:F1_profile][icoil_idx_exp,:]),
        label="Experiment")
    plot!(f1, xlims=(-4,4),
        xlabel=L"$z$ (mm)",
        title =L"F1: $I_{c}=%$(round(data_exp[:Icoils][icoil_idx_exp], digits=3))\mathrm{A}$ | $I_{c}=%$(data_simulation[:Icoils][icoil_idx_sim])\mathrm{A}$",
        legend=:topleft,
        )


    f2 = plot(data_simulation_top[icoil_idx_sim][:z_profile][:,1], data_simulation_top[icoil_idx_sim][:z_profile][:,2]/maximum(data_simulation_top[icoil_idx_sim][:z_profile][:,2]),
        label="Simulation raw", seriestype=:scatter, marker=(:circle, :white,1), markerstrokecolor=:grey60)
    plot!(f2, data_simulation_top[icoil_idx_sim][:z_profile][:,1], data_simulation_top[icoil_idx_sim][:z_profile][:,3]/maximum(data_simulation_top[icoil_idx_sim][:z_profile][:,3]),
        label=L"Simulation: $150\mathrm{\mu m}$")
    plot!(f2, data_simulation_top[icoil_idx_sim][:z_profile][:,1], y_conv_f2/maximum(y_conv_f2),
        label=L"Simulation: $%$(1e3*new_width)\mathrm{\mu m}$")
    plot!(f2, z_mm .- centroid, data_exp[:F2_profile][icoil_idx_exp,:]/maximum(data_exp[:F2_profile][icoil_idx_exp,:]),
        label="Experiment")
    plot!(f2, xlims=(-4,4), 
        xlabel=L"$z$ (mm)",
        title =L"F2: $I_{c}=%$(round(data_exp[:Icoils][icoil_idx_exp], digits=3))\mathrm{A}$ | $I_{c}=%$(data_simulation[:Icoils][icoil_idx_sim])\mathrm{A}$"
    )

    fig=plot(f1,f2, 
    layout=(1,2),
    size=(800,400),
    legendfontsize=6)
    display(fig)
end









