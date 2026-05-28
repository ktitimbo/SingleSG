# Comparison
# Kelvin Titimbo
# California Institute of Technology
# MAy 2026

#  Plotting Setup
# ENV["GKS_WSTYPE"] = "100"
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
using Polynomials
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
LinearAlgebra.BLAS.set_num_threads(1)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Set the working directory to the current location
cd(@__DIR__);
const BASE_PATH = raw"F:\SternGerlachExperiments"
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
const OUTDIR    = joinpath(@__DIR__, "simulation_data", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
const TEMP_DIR = joinpath(@__DIR__,"artifacts", "JuliaTemp")
isdir(TEMP_DIR) || mkpath(TEMP_DIR);
ENV["TMPDIR"] = TEMP_DIR
ENV["TEMP"]   = TEMP_DIR
ENV["TMP"]    = TEMP_DIR
@info "Temporary directory configured" TEMP_DIR
# General setup
HOSTNAME = gethostname();
@info "Running on host" HOSTNAME=HOSTNAME
# Random seeds
base_seed_set = 145;
rng_set = MersenneTwister(base_seed_set)
# rng_set = TaskLocalRNG();
# Custom modules
include("./Modules/atoms.jl");
include("./Modules/samplings.jl");
include("./Modules/JLD2_MyTools.jl")
include("./Modules/TheoreticalSimulation.jl");
using .TheoreticalSimulation;
TheoreticalSimulation.SAVE_FIG = SAVE_FIG;
TheoreticalSimulation.FIG_EXT  = FIG_EXT;
TheoreticalSimulation.OUTDIR   = OUTDIR;

println("\n\t\tRunning process on:\t $(RUN_STAMP) \n")

atom        = "39K"  ;
## PHYSICAL CONSTANTS from NIST
# RSU : Relative Standard Uncertainty
const kb    = 1.380649e-23 ;       # Boltzmann constant (J/K)
const ħ     = 6.62607015e-34/2π ;  # Reduced Planck constant (J s)
const μ₀    = 1.25663706127e-6;    # Vacuum permeability (Tm/A)
const μB    = 9.2740100657e-24 ;   # Bohr magneton (J/T)
const γₑ    = -1.76085962784e11 ;  # Electron gyromagnetic ratio  (1/sT). Relative Standard Uncertainty = 3.0e-10
const μₑ    = 9.2847646917e-24 ;   # Electron magnetic moment (J/T). RSU = 3.0e-10
const Sspin = 1/2 ;                # Electron spin
const gₑ    = -2.00231930436092 ;  # Electron g-factor

#_____________________________________________________________________________________________________________
# Coil currents
Icoils = [0.00,
            0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
            0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,
            0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
];
nI = length(Icoils);

σw = 0.150
ki = 2.0 # ×10^-6
nz = 2
λ0 = 0.01
#_____________________________________________________________________________________________________________

CQD_UP_T200_manual = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T200_8M","cqd_8M_up_profiles.jld2")
CQD_DW_T200_manual = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T200_8M","cqd_8M_dw_profiles.jld2")

CQD_UP_T200_manual_data = jldopen(CQD_UP_T200_manual,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:up,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end

CQD_DW_T200_manual_data = jldopen(CQD_DW_T200_manual,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:dw,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end


CQD_UP_T200_ΔG = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T200_6M_constG","cqd_6000000_up_profiles.jld2")
CQD_DW_T200_ΔG = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T200_6M_constG","cqd_6000000_dw_profiles.jld2")

CQD_UP_T200_ΔG_data = jldopen(CQD_UP_T200_ΔG,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:up,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end

CQD_DW_T200_ΔG_data = jldopen(CQD_DW_T200_ΔG,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:dw,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end



i0_idx = 47
for i0_idx=42:nI
FIGUP = plot(CQD_UP_T200_manual_data.z_profiles[i0_idx][:,1],
CQD_UP_T200_manual_data.z_profiles[i0_idx][:,3],
label="manual",
line=(:solid,2,:red)
)
plot!(CQD_UP_T200_ΔG_data.z_profiles[i0_idx][:,1],
CQD_UP_T200_ΔG_data.z_profiles[i0_idx][:,3],
label=L"$\Delta\mathcal{G}$",
line=(:dashdot,1.5,:dodgerblue3)
)

FIGDW = plot(CQD_DW_T200_manual_data.z_profiles[i0_idx][:,1],
CQD_DW_T200_manual_data.z_profiles[i0_idx][:,3],
label="manual",
line=(:solid,2,:orange)
)
plot!(CQD_DW_T200_ΔG_data.z_profiles[i0_idx][:,1],
CQD_DW_T200_ΔG_data.z_profiles[i0_idx][:,3],
label=L"$\Delta\mathcal{G}$",
line=(:dashdot,1.5,:purple)
)


FIG = plot(FIGUP, FIGDW,
    suptitle = L"CQD | $T=200 \degree \mathrm{C}$ | $I_{c}=%$(Int(1000*Icoils[i0_idx]))\mathrm{mA}$",
    layout=(2,1),
    link=:x,
)
plot!(FIG[1], xlabel="", xformatter=_->"", bottom_margin=-5mm)
display(FIG)
end




plot(xlabel="Currents (A)", ylabel=L"$z_{max} \ (\mathrm{mm})$")
plot!(CQD_UP_T200_manual_data.Ic, CQD_UP_T200_manual_data.z_max)
plot!(CQD_UP_T200_ΔG_data.Ic, CQD_UP_T200_ΔG_data.z_max)