# Simulation of atom trajectories in the Stern–Gerlach experiment
# Interpolation of the grid for fitting the induction term
# Kelvin Titimbo
# California Institute of Technology
# September 2025

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
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSS");
const OUTDIR    = joinpath(@__DIR__, "data_studies", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
include("./Modules/TheoreticalSimulation.jl");
include("./Modules/DataReading.jl")



# import Suleyman's ki simulation:

sim_data = CSV.read("./simulation_data/results_CQD_20250829T005913.csv",DataFrame; header=false)
Ic_sim = sim_data[!,1]
ki_sim = collect(1.50:0.10:3.50) ./ 1_000_000
n_col = (1+2*21) + 1 
z_sim = Matrix(sim_data[1:end,n_col:end])

plot(xlabel=L"$I_{c}$ (A)")
for i=1:length(ki_sim)
    plot!(Ic_sim[6:end], abs.(z_sim[6:end,i]) , label = L"$k_{i} =%$(round(1e6*ki_sim[i], digits=2)) \times 10^{-6}$")
end
plot!(ylabel=L"$z$ (mm)",
    yaxis=:log10,
    legend=:outerright,
)

itp = Spline2D(Ic_sim, ki_sim, z_sim; kx=3, ky=3, s=0.00)

res = DataReading.find_report_data(joinpath(@__DIR__, "analysis_data");
        wanted_data_dir="20250825/",
        wanted_binning=2,
        wanted_smooth=0.02
)

if res === nothing
    @warn "No matching report found"
else
    @info "Matched" res.path res.data_dir res.name res.binning res.smoothing
    I_exp = sort(res.currents_mA / 1_000)
    z_exp = res.framewise_mm/res.magnification
end

xxx = hcat(I_exp, z_exp)
xxx = xxx[[10,12,14,22:25...],:]

function loss(ki)
    # ni=12
    z_pred = itp.(xxx[:,1], Ref(ki))
    return mean(abs2,log10.(z_pred) .- log10.(xxx[:,2]))
end


fit_param = optimize(loss, minimum(ki_sim), maximum(ki_sim),Brent())
k_fit = Optim.minimizer(fit_param)

# diagnostics
mse = loss(k_fit)
pred = itp.(I_exp, Ref(k_fit))
r2 = 1 - sum(abs2, pred .- z_exp) / sum(abs2, z_exp .- mean(z_exp))


logspace10(lo, hi; n=50) = 10.0 .^ range(log10(lo), log10(hi); length=n)

I_scan = logspace10(20e-3,1; n=30)
plot(
    xlabel=L"$I_{c}$ (A)",
    ylabel=L"$z$ (mm)"
)
plot!(I_exp[9:end], z_exp[9:end], 
    label="Experiment 0825",
    seriestype=:scatter,
    marker=(:circle,:white,3), 
    markerstrokecolor=:red, 
    markerstrokewidth=2
)
plot!(I_scan, itp.(I_scan, Ref(k_fit)),
    label=L"$k_{i}= %$(round(1e6*k_fit, digits=3))\times 10^{-6} $",
    line=(:solid,:blue,2),
    marker=(:xcross, :blue, 1),
)
plot!(xaxis=:log10,
    yaxis=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(5e-3,5),
    legend=:bottomright,
)