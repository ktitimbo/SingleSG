# Simulation of atom trajectories in the Stern–Gerlach experiment
# Manipulation of quantum simulations
# Kelvin Titimbo
# California Institute of Technology
# December 2025

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
# using Interpolations #, Roots, Loess, Optim
# using BSplineKit
using Dierckx, Optim
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
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
const OUTDIR    = joinpath(@__DIR__, "data_studies", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
include("./Modules/TheoreticalSimulation.jl");
include("./Modules/DataReading.jl");
include("./Modules/MyExperimentalAnalysis.jl");


table_old = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_3M","qm_3000000_screen_profiles_table.jld2"))["table"];
table_qm_f1   = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_6M","qm_6000000_screen_profiles_f1_table.jld2"))["table"];
table_qm_f2   = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_6M","qm_6000000_screen_profiles_f2_table.jld2"))["table"];
ks = keys(table_qm_f1);
Ic = [table_old[(2,0.250,0.01)][v][:Icoil] for v=1:47]

zqm_old  = [table_old[(2,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:47]
zqm_F1  = [table_qm_f1[(2,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:47]
zqm_F2  = [table_qm_f2[(2,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:47]

plot(Ic[2:end],(zqm_F1 .+ zqm_F2)[2:end]/2,
    xscale=:log10,
    )


println("the nz-binning are: [ ", join(string.(sort(unique(first.(ks)))), ", ")," ]")
println("the σ-values for the smoothing convolution are (mm): [ ", join((@sprintf("%.3f", x) for x in sort(unique(getindex.(ks,2)))), ", ")," ]")
println("the spline fitting smoothing factors are: [ ", join((@sprintf("%.3f", x) for x in sort(unique(getindex.(ks,3)))), ", ")," ]")




fig = plot(xlims = (-2,2),
    xlabel=L"$z$ (mm)")
for i in sort(unique(getindex.(ks,1)))
    set1 = table_qm_f1[(i,0.001,0.001)][1][:z_profile]
    plot!(fig,
        set1[:,1],set1[:,2],
        label=L"$n_{z}=%$(i)$")
    display(fig)
end

fig = plot(xlims = (-2,2),
    xlabel=L"$z$ (mm)")
set0 = table_qm_f1[(2,0.001,0.001)][1][:z_profile]
plot!(fig,
    set0[:,1],set0[:,2],
    label=L"$\sigma_{z}=0$ μm (raw)")
for i in sort(unique(getindex.(ks,2)))
    set1 = table_qm_f1[(2,i,0.001)][1][:z_profile]
    plot!(fig,
        set1[:,1],set1[:,3],
        label=L"$\sigma_{z}=%$(1e3*i)$ μm")
    display(fig)
end

fig = plot(xlims = (-2,2),
    xlabel=L"$z$ (mm)")
set0 = table_qm_f1[(2,0.001,0.001)][1][:z_profile]
plot!(fig,
    set0[:,1],set0[:,2],
    label=L"$\sigma_{z}=0$ μm (raw)")
for i in sort(unique(getindex.(ks,3)))
    set1 = table_qm_f1[(2,0.001,i)][1][:z_profile]
    plot!(fig,
        set1[:,1],set1[:,3],
        label=L"$\lambda_{0}=%$(i)$")
    display(fig)
end



zqm  = [table_qm_f1[(2,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:47]
zold = [table_old[(4,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:47]

println(hcat(zqm,zold))

plot(Ic[2:end],zqm[2:end])
plot!(Ic[2:end],zold[2:end])
plot!(xscale=:log10,
yscale=:log10,
xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)

plot(Ic[2:end],abs.(1e3*(zqm[2:end]-zold[2:end])),
    label="6M – 3M",
    ylabel="Difference (μm)",
    xscale=:log10,
    yscale=:log10,
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    ylims=(1e-2,5))


fig = plot(ylabel=L"$z$ (mm)",
    xlabel=L"$I_{c}$ (A)")
for i in sort(unique(getindex.(ks,1)))
    zqm_F1  = [table_qm_f1[(i,0.150,0.001)][v][:z_max_smooth_spline_mm] for v=1:47]
    fig = plot!(Ic[10:end], zqm_F1[10:end],label=L"$n_{z}=%$(i)$")
end
plot!(xscale=:log10,
    yscale=:log10,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)

    sort(unique(getindex.(ks,2)))

data_exp = load(joinpath(@__DIR__,"20250820","data_processed.jld2"),"data")

fig = plot(xlabel="pixel")
zpix = 1e3*TheoreticalSimulation.pixel_coordinates(2560,1,6.5e-6)
for i =1:10
    plot!(fig,zpix,vec(mean(data_exp[:F1ProcessedImages][:,:,i,1],dims=1)), ls=:dash)
    plot!(fig,zpix,vec(mean(data_exp[:F2ProcessedImages][:,:,i,1],dims=1)), ls=:dot)
end
plot!(fig,zpix,vec(mean(data_exp[:F1ProcessedImages][:,:,:,1],dims=(3,1))), line=(:solid,:black))
plot!(fig,zpix,vec(mean(data_exp[:F2ProcessedImages][:,:,:,1],dims=(3,1))), line=(:solid,:black))
display(fig)

fig = plot(xlabel=L"$x$ (mm)")
xpix = 1e3*TheoreticalSimulation.pixel_coordinates(2160,4,6.5e-6)
for i =1:10
    plot!(fig,xpix,vec(mean(data_exp[:F1ProcessedImages][:,:,i,1],dims=2)), ls=:dash)
    plot!(fig,xpix,vec(mean(data_exp[:F2ProcessedImages][:,:,i,1],dims=2)), ls=:dot)
end
plot!(fig,xpix,vec(mean(data_exp[:F1ProcessedImages][:,:,:,1],dims=(3,2))), line=(:solid,:black))
plot!(fig,xpix,vec(mean(data_exp[:F2ProcessedImages][:,:,:,1],dims=(3,2))), line=(:solid,:black))
display(fig)




data_qm_profiles = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_6M","qm_6000000_screen_profiles.jld2"),"profiles")

load(joinpath("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_6M\\qm_6000000_screen_data.jld2"))