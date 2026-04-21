# Kelvin Titimbo
# California Institute of Technology
# March 2026

#  Plotting Setup
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
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Set the working directory to the current location
cd(@__DIR__) ;
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
const OUTDIR    = joinpath(@__DIR__, "data_studies", RUN_STAMP);
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
include("./Modules/JLD2_MyTools.jl");
include("./Modules/DataReading.jl");
include("./Modules/FittingDataCQDQM.jl");
include("./Modules/MyExperimentalAnalysis.jl");
include("./Modules/TheoreticalSimulation.jl");
using .TheoreticalSimulation;
TheoreticalSimulation.SAVE_FIG = SAVE_FIG;
TheoreticalSimulation.FIG_EXT  = FIG_EXT;
TheoreticalSimulation.OUTDIR   = OUTDIR;

println("\n\t\tRunning process on:\t $(RUN_STAMP) \n")

# Math constants
const TWOπ = 2π;
const INV_E = exp(-1);
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
## ATOM INFORMATION: 
# atom_info       = AtomicSpecies.atoms(atom);
K39_params = AtomParams(atom); # [R μn γn Ispin Ahfs M ] 
# Math constants
const TWOπ = 2π;
const INV_E = exp(-1);
quantum_numbers = fmf_levels(K39_params);

# STERN--GERLACH EXPERIMENT
# Camera and pixel geometry : intrinsic properties
cam_pixelsize = 6.5e-6 ;  # Physical pixel size of camera [m]
nx_pixels , nz_pixels= (2160, 2560); # (Nx,Nz) pixels
# Simulation resolution
sim_bin_x, sim_bin_z = (1,1) ;  # Camera binning
sim_pixelsize_x, sim_pixelsize_z = (sim_bin_x, sim_bin_z).*cam_pixelsize ; # Effective pixel size after binning [m]
# Image dimensions (adjusted for binning)
x_pixels = Int(nx_pixels / sim_bin_x);  # Number of x-pixels after binning
z_pixels = Int(nz_pixels / sim_bin_z);  # Number of z-pixels after binning
# Spatial axes shifted to center the pixels
x_position = pixel_coordinates(x_pixels, sim_bin_x, sim_pixelsize_x);
z_position = pixel_coordinates(z_pixels, sim_bin_z, sim_pixelsize_z);
println("""
***************************************************
CAMERA FEATURES
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm

SIMULATION INFORMATION
    Binning                 : $(sim_bin_x) × $(sim_bin_z)
    Effective pixels        : $(x_pixels) × $(z_pixels)
    Pixel size              : $(1e6*sim_pixelsize_x)μm × $(1e6*sim_pixelsize_z)μm
    xlims                   : ($(round(minimum(1e6*x_position), digits=6)) μm, $(round(maximum(1e3*x_position), digits=4)) mm)
    zlims                   : ($(round(minimum(1e6*z_position), digits=6)) μm, $(round(maximum(1e3*z_position), digits=4)) mm)
***************************************************
""")
# Furnace
T_K = 273.15 + 200 ; # Furnace temperature (K)
# Furnace aperture
const x_furnace = 2.0e-3 ;
const z_furnace = 100e-6 ;
# Slit : Pre SG
const x_slit  = 4.0e-3 ;
const z_slit  = 300e-6 ;
# Circular Aperture : Post SG
const R_aper            = 5.8e-3/2 ;
const y_SGToAperture    = 42.0e-3 ;   
# Propagation distances
const y_FurnaceToSlit = 224.0e-3 ;
const y_SlitToSG      = 44.0e-3 ;
const y_SG            = 7.0e-2 ;
const y_SGToScreen    = 32.0e-2 ;
# Connecting pipes
const R_tube = 35e-3/2 ; # Radius of the connecting pipe (m)
effusion_params = BeamEffusionParams(x_furnace,z_furnace,x_slit,z_slit,y_FurnaceToSlit,T_K,K39_params);
println("""
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
""")
# Setting the variables for the module
TheoreticalSimulation.default_camera_pixel_size = cam_pixelsize;
TheoreticalSimulation.default_x_pixels          = nx_pixels;
TheoreticalSimulation.default_z_pixels          = nz_pixels;
TheoreticalSimulation.default_x_furnace         = x_furnace;
TheoreticalSimulation.default_z_furnace         = z_furnace;
TheoreticalSimulation.default_x_slit            = x_slit;
TheoreticalSimulation.default_z_slit            = z_slit;
TheoreticalSimulation.default_y_FurnaceToSlit   = y_FurnaceToSlit;
TheoreticalSimulation.default_y_SlitToSG        = y_SlitToSG;
TheoreticalSimulation.default_y_SG              = y_SG;
TheoreticalSimulation.default_y_SGToScreen      = y_SGToScreen;
TheoreticalSimulation.default_R_tube            = R_tube;
TheoreticalSimulation.default_c_aperture        = R_aper;
TheoreticalSimulation.default_y_SGToAperture    = y_SGToAperture;
##################################################################################################
function standard_error(x)
    return std(x; corrected=true) ./ sqrt.(length(x))
end

logspace10(lo, hi; n=50) = 10.0 .^ range(log10(lo), log10(hi); length=n)

function log_mask(x, y)
    (x .> 0) .& (y .> 0) .& isfinite.(x) .& isfinite.(y)
end

# helper: first index where column > threshold (skips missings; falls back to 1)
@inline function first_gt_idx(df::DataFrame, col::Symbol, thr::Real)
    v = df[!, col]
    idx = findfirst(x -> !ismissing(x) && x >= thr, v)
    return idx === nothing ? 1 : idx
end

function sigdigits_str(x, n=3)
    x == 0 && return "0"

    xr = round(x; sigdigits=n)

    # number of decimal places needed
    d = max(0, n - 1 - floor(Int, log10(abs(xr))))

    s = @sprintf("%.*f", d, xr)

    # remove trailing zeros and possible trailing dot
    s = replace(s, r"(\.\d*?)0+$" => s"\1")
    s = replace(s, r"\.$" => "")

    return s
end

function sci_label(x; n=3)
    x == 0 && return L"$0.0$"

    exp = floor(Int, log10(abs(x)))
    mant = x / 10.0^exp

    # round mantissa to (n sigdigits → typically 1 decimal for n=3)
    mant_r = round(mant; sigdigits=n)

    # force exactly ONE decimal place
    mant_str = @sprintf("%.1f", mant_r)

    return L"$%$(mant_str) \times 10^{%$exp}$"
end

function fit_poly2(x, y)
    X = hcat(ones(length(x)), x, x.^2)
    β = X \ y
    return β
end


##################################################################################################
# Coil currents
Icoils = [0.00,
            0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
            0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,
            0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
];
nI = length(Icoils);
colores_current = palette(:darktest,nI);

data_qmf1_path = joinpath(@__DIR__,"simulation_data",
    "QM_T200_8M",
    "qm_screen_profiles_f1_table.jld2");
JLD2_MyTools.summarize_meta_qm_jld2(data_qmf1_path)
data_qmf2_path = joinpath(@__DIR__,"simulation_data",
    "QM_T200_8M",
    "qm_screen_profiles_f2_table.jld2");
JLD2_MyTools.summarize_meta_qm_jld2(data_qmf2_path)
data_cqdup_path = joinpath(@__DIR__, "simulation_data",
    "CQD_T200_8M",
    "cqd_8M_up_profiles.jld2");
JLD2_MyTools.summarize_meta_cqd_jld2(data_cqdup_path)
data_cqddw_path = joinpath(@__DIR__, "simulation_data",
    "CQD_T200_8M",
    "cqd_8M_dw_profiles.jld2");
JLD2_MyTools.summarize_meta_cqd_jld2(data_cqddw_path)


data_directories = [
                    # "20260220", 
                    # "20260225", 
                    "20260226am",
                    # "20260226pm",
                    # "20260227", 
                    "20260303", 
                    # "20260306r1", 
                    "20260306r2",
]
n_data = length(data_directories);
colores_data = palette(:darkrainbow, n_data);

nz , λ0 = 2, 0.01;
σw_mm = 0.175;

## ################################################################################################
# MAGNETIC FIELD
############ phywe magnetic field measurement ##############################
# PhyWe Calibration
BvsI_phywe = sort(CSV.read("SG_BvsI_phywe.csv",DataFrame; header=["Ic","Bz"]),1);
phywe_shift = hcat(BvsI_phywe.Ic .- 0.11021, BvsI_phywe.Bz .- 0.00445);
phywe_shift = DataReading.subset_by_cols(phywe_shift,[1,2]; thr=0.0, include_equal=false)[3];

# Experimental measurements while acquiring SG1 data
exp_data = Vector{Matrix{Float64}}(undef, n_data)
for (idx, dir) in enumerate(data_directories)
    data = load(joinpath(@__DIR__,"EXPERIMENTS",dir,"data_processed.jld2"),"data")
    Ic = data[:Currents]
    Bz = data[:BzTesla]
    exp_data[idx] = hcat(Ic,Bz)
end

BvsI_optimal = CSV.read("SG_BzinGauss_Iscan.csv", DataFrame; header=["Ic","Bz"]);

# magnetic field: log-log scale
fig1 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    size=(900,500),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    left_margin=5mm,
    bottom_margin=3mm,);
# for (idx, dir) in enumerate(data_directories)
#     data = exp_data[idx]
#     Ic = data[:,1]
#     Bz = data[:,2]

#     plot!(fig1, Ic, 1000*Bz,
#         label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
#         marker=(:circle, 2, :white),
#         markerstrokecolor=colores[idx],
#         line=(:dot, colores[idx], 1)
#     )

# end
plot!(fig1,Icoils[2:end], 1e3*TheoreticalSimulation.BvsI.(Icoils)[2:end],
    label="SG manual",
    line=(:solid,2,:black));
# plot!(fig1,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
#     label="SG PHYWE calibration",
#     line=(:solid,2,:dodgerblue4));
# plot!(fig1,phywe_shift[:,1], 1000*phywe_shift[:,2],
#     label="SG PHYWE calibration (adjusted)",
#     line=(:dot,2,:dodgerblue4))
plot!(fig1,1e-3*BvsI_optimal.Ic[4:end], 1e3*1e-4*BvsI_optimal.Bz[4:end],
    line=(:dashdot,2,:purple),
    marker=(:diamond, 3, :white),
    markerstrokecolor=:purple,
    label="New setup arrangement",
);
plot!(fig1,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.02),
    ylims=(1e-2,1e3),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10, 100, 1000], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}", L"10^{3}"]),
    legendfontsize = 8,
);
display(fig1)

# magnetic field: linear scale
fig2 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    size=(900,500),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    left_margin=5mm,
    bottom_margin=3mm,
    legendfontsize = 8,);
# for (idx, dir) in enumerate(data_directories)
#     data = exp_data[idx]
#     Ic = data[:,1]
#     Bz = data[:,2]

#     plot!(fig2, Ic, 1000*Bz,
#         label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
#         marker=(:circle, 2, :white),
#         markerstrokecolor=colores[idx],
#         line=(:dot, colores[idx], 1)
#     )

# end
plot!(fig2,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
# plot!(fig2,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
#     label="SG PHYWE calibration",
#     line=(:solid,2,:dodgerblue4));
# plot!(fig2,phywe_shift[:,1], 1000*phywe_shift[:,2],
#     label="SG PHYWE calibration (adjusted)",
#     line=(:dot,2,:dodgerblue4));
plot!(fig2,1e-3*BvsI_optimal.Ic, 1e3*1e-4*BvsI_optimal.Bz,
    line=(:dashdot,2,:purple),
    marker=(:diamond, 3, :white),
    markerstrokecolor=:purple,
    label="New setup arrangement",
);
display(fig2)

# magnetic field: low currents
fig3 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    size=(900,500),);
for (idx, dir) in enumerate(data_directories)
    data = exp_data[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig3, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig3,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig3,1e-3*BvsI_optimal.Ic, 1e3*1e-4*BvsI_optimal.Bz,
    line=(:dashdot,2,:purple),
    marker=(:diamond, 3, :white),
    markerstrokecolor=:purple,
    label="New setup arrangement",
);
plot!(fig3,
    xlims=(0.0,15e-3),
    ylims=(-5,15),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig3)

# magnetic field: high currents
fig4 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    size=(900,500),);
for (idx, dir) in enumerate(data_directories)
    data = exp_data[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig4, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig4,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black))
plot!(fig4,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
    label="SG PHYWE calibration",
    line=(:solid,2,:dodgerblue4));
plot!(fig4,phywe_shift[:,1], 1000*phywe_shift[:,2],
    label="SG PHYWE calibration (adjusted)",
    line=(:dot,2,:dodgerblue4));
plot!(fig4,1e-3*BvsI_optimal.Ic, 1e3*1e-4*BvsI_optimal.Bz,
    line=(:dashdot,2,:purple),
    marker=(:diamond, 3, :white),
    markerstrokecolor=:purple,
    label="New setup arrangement",
);
plot!(fig4,
    xlims=(0.600,1.1),
    ylims=(300,900),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    bottom_margin=2mm,
    left_margin=3mm,
);
display(fig4)

# magnetic field measured vs magnetic field manual
BvsI_comparison = Vector{Matrix{Float64}}(undef, n_data);
for (idx, dir) in enumerate(data_directories)
    vs = exp_data[idx]

    mask, inds, rows_view = DataReading.subset_by_cols(vs, [1,2]; thr=1e-3, include_equal = true)
    BvsI_comparison[idx] = Matrix(rows_view)  # store a copy; or keep the view (see B)
end
fig5 = plot(xlabel="Currents (A)",
    ylabel=L"$B_{\mathrm{exp}} / B_{\mathrm{manual}}$");
for (idx, dir) in enumerate(data_directories)
    B_ratio = BvsI_comparison[idx][:,2] ./ TheoreticalSimulation.BvsI.(BvsI_comparison[idx][:,1])
    
    plot!(fig5,  BvsI_comparison[idx][:,1] , B_ratio,
        label=data_directories[idx],
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:solid,1,colores_data[idx])
    )
    y0 , σ0 = mean(B_ratio), standard_error(B_ratio)

    x = range(1e-3, 1.1, length=200)
    plot!(fig5, x, fill(y0, length(x)),
     ribbon = σ0,
     color = colores_data[idx],
     fillalpha = 0.25,
     line=(:dash,0.5,colores_data[idx]),
     label = "$(round(y0; digits=3)) ± $(round(σ0; sigdigits=1))")
    # hline!([y0], ine=(:dot,0.5,colores[idx]), label= "")
end
plot!(fig5,
    legend=:bottomright,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.05),
    ylims=(3e-1,2),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig5)

fig5a = plot(xlabel="Currents (A)",
    ylabel=L"$B_{\mathrm{exp}} / B_{\mathrm{manual}}$");
plot!(1e-3*BvsI_optimal.Ic[4:end] , (1e-4*(BvsI_optimal.Bz) ./ TheoreticalSimulation.BvsI.(1e-3*BvsI_optimal.Ic))[4:end],
    label=false,
    marker=(:square,2,:white),
    markerstrokecolor=:red,
    line=(:solid,1,:red));
plot!(fig5a,
    legend=:bottomright,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.05),
    ylims=(1e-1,2),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig5a)

# SHIFTED EXPERIMENTAL DATA TO ZERO FIELD
exp_data_corr = Vector{Matrix{Float64}}(undef, n_data);
for (idx, dir) in enumerate(data_directories)
    exp_data_corr[idx] = hcat(exp_data[idx][:,1] , exp_data[idx][:,2] .- exp_data[idx][1,2] )
end

fig1A = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    title="Shifted B field");
for (idx, dir) in enumerate(data_directories)
    data = exp_data_corr[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig1A, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig1A,Icoils[2:end], 1e3*TheoreticalSimulation.BvsI.(Icoils)[2:end],
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig1A,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.02),
    ylims=(1e-2,1e3),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10, 100], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig1A)

fig2A = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    title="Shifted B field");
for (idx, dir) in enumerate(data_directories)
    data = exp_data_corr[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig2A, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig2A,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig2A,
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig2A)

fig3A = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    title="Shifted B field");
for (idx, dir) in enumerate(data_directories)
    data = exp_data_corr[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig3A, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig3A,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig3A,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
    label="SG PHYWE calibration",
    line=(:solid,2,:dodgerblue4));
plot!(fig3A,phywe_shift[:,1], 1000*phywe_shift[:,2],
    label="SG PHYWE calibration (adjusted)",
    line=(:dot,2,:dodgerblue4));
plot!(fig3A,
    # xscale=:log10,
    # yscale=:log10,
    xlims=(0.0,15e-3),
    ylims=(-5,15),
    # xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # yticks = ([0.01, 0.1, 1, 10, 100], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig3A)

fig4A = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    title="Shifted B field");
for (idx, dir) in enumerate(data_directories)
    data = exp_data_corr[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig4A, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig4A,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig4A,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
    label="SG PHYWE calibration",
    line=(:solid,2,:dodgerblue4));
plot!(fig4A,phywe_shift[:,1], 1000*phywe_shift[:,2],
    label="SG PHYWE calibration (adjusted)",
    line=(:dot,2,:dodgerblue4));
plot!(fig4A,
    xlims=(0.600,1.10),
    ylims=(300,900),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig4A)


BvsI_comparison_corr = Vector{Matrix{Float64}}(undef, n_data)
for (idx, dir) in enumerate(data_directories)
    vs = exp_data_corr[idx]

    mask, inds, rows_view = DataReading.subset_by_cols(vs, [1,2]; thr=1e-3, include_equal = true)
    BvsI_comparison_corr[idx] = Matrix(rows_view)  # store a copy; or keep the view (see B)
end


fig5A = plot(xlabel="Currents (A)",
    ylabel=L"$B_{\mathrm{exp}} / B_{\mathrm{manual}}$",
    title="Shifted B field")
for (idx, dir) in enumerate(data_directories)
    B_ratio = BvsI_comparison_corr[idx][:,2] ./ TheoreticalSimulation.BvsI.(BvsI_comparison_corr[idx][:,1])
    
    plot!(fig5A,
        BvsI_comparison_corr[idx][:,1] , B_ratio,
        label=data_directories[idx],
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:solid,1,colores_data[idx])
    )
    y0 , σ0 = mean(B_ratio), standard_error(B_ratio)

    x = range(1e-3, 1.1, length=200)
    plot!(fig5A,
    x, fill(y0, length(x)),
    ribbon = σ0,
    color = colores_data[idx],
    fillalpha = 0.25,
    line=(:dash,0.5,colores_data[idx]),
    label = "$(round(y0; digits=3)) ± $(round(σ0; sigdigits=1))")
end
plot!(fig5A,legend=:bottomright,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.05),
    ylims=(3e-1,2),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    legendfontsize = 6,
    legend_columns=2,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
)
display(fig5A)

plot(fig5,fig5A,
layout=(1,2),
size=(1200,600),
left_margin=5mm,
bottom_margin=2mm,
)

plot(fig1,fig2, fig3,fig4,
layout=(2,2),
size=(1200,700),
left_margin=5mm,
bottom_margin=2mm,
)

plot(fig1A,fig2A, fig3A,fig4A,
layout=(2,2),
size=(1200,700),
left_margin=5mm,
bottom_margin=2mm,
)

########################################################################################
## ######################################################################################
#+++++++++++++++++++++++++++ Centroid +++++++++++++++++++++++++++++++++++++++++++++
centroid_fw = Matrix{Float64}(undef, n_data, 2)
pos_at_zero = Matrix{Float64}(undef, n_data, 4)
for (i,dir) in enumerate(data_directories)
    kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS","summary", dir, dir * "_report_summary.jld2")

    data = jldopen(kk_path, "r") do file
        dd = file[JLD2_MyTools.make_keypath_exp(dir,nz,λ0)]
    end

    # zc ± δzc (computed from the high currents)
    centroid_fw[i,1] = data[:centroid_fw_mm][1]
    centroid_fw[i,2] = round(data[:centroid_fw_mm][2], digits=3)

    # zc ± δzc (for the signal at zero current for F1 and F2)
    pos_at_zero[i,1] = data[:fw_F1_peak_pos_raw][1][1]
    pos_at_zero[i,2] = data[:fw_F1_peak_pos_raw][2][1]
    pos_at_zero[i,3] = data[:fw_F2_peak_pos_raw][1][1]
    pos_at_zero[i,4] = data[:fw_F2_peak_pos_raw][2][1]

end

fig = plot(data_directories, centroid_fw[:,1],
label="Computed centroid positions (high currents)",
 color=:blue,
 marker=(:circle,:white,4),
 markerstrokecolor=:blue,
 line=(:solid,2,:blue),
 yerror = centroid_fw[:,2],);
plot!(data_directories,pos_at_zero[:,1],
    ribbon=pos_at_zero[:,2],
    label=L"$F=1$ at ($I_{c}=0A$)",
    color=:orangered2,
    marker=(:circle,:white),
    markerstrokecolor=:orangered2,
    fillalpha=0.2);
plot!(data_directories,pos_at_zero[:,3],
    ribbon=pos_at_zero[:,4],
    label=L"$F=2$ at ($I_{c}=0A$)",
    color=:springgreen2,
    marker=(:circle,:white),
    markerstrokecolor=:springgreen2,
    fillalpha=0.2);
# plot!(["20260225","20260226am","20260226pm","20260227","20260303"],[8.8369,8.8440,8.8549,8.8284,8.8292],
#     label="Qihang's fitting 1",
#     marker=(:square,2,:white),
#     markerstrokecolor=:purple,
#     line=(:purple))
# plot!(["20260225","20260226am","20260226pm","20260227","20260303"], [8.811211551,8.790075956,8.796932977,8.84320487,8.8048355],
#     label="Qihang's fitting 2",
#     marker=(:square,2,:white),
#     markerstrokecolor=:green,
#     line=(:green))
plot!(
    # title = "Std.Dev=$(round(std(centroid_fw[:,1], corrected=false); digits=3))",
    legend=:bottomleft,
    ylims=(8.700,8.950),
    yformatter = y -> @sprintf("%.3f", y),
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    xminorticks=false,
    bottom_margin=5mm,
    ylabel="Position (mm)",
    xrotation=75,);
display(fig)


########################################################################################
########################################################################################
########################################################################################
#+++++++++++++++++++++++++++ Peak Position +++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++ QUANTUM MECHANICS +++++++++++++
chosen_qm_f1 =  jldopen(data_qmf1_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end;
chosen_qm_f2 =  jldopen(data_qmf2_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end;
data_QM = hcat(
    [v[:Icoil]                  for v in values(chosen_qm_f1)],
    [v[:z_max_smooth_spline_mm] for v in values(chosen_qm_f1)],
    [v[:z_max_smooth_spline_mm] for v in values(chosen_qm_f2)],
)
data_QM = hcat(data_QM, data_QM[:,2] .- data_QM[:,3]);
QM_df = DataFrame(data_QM, [:Ic, :F1, :F2, :Δ]);
pretty_table(QM_df; 
            column_labels = [["Ic", "F1", "F2", "Δ"],["(A)","(mm)","(mm)","(mm)"]],
            alignment =:c,
            row_label_column_alignment  = :c,
            row_group_label_alignment   = :c,
            title         = @sprintf("QUANTUM MECHANICS"),
            formatters = [fmt__printf("%8.3f", [1]), fmt__printf("%8.6f",2:4)],
            style = TextTableStyle(
                    first_line_column_label = crayon"yellow bold",
                    table_border  = crayon"blue bold",
                    title = crayon"bold red"
            ),
            table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
            equal_data_column_widths= true,
)

r = QM_df[!,:F2] ./ QM_df[!,:F1];
fig0a = plot(QM_df[!,:Ic], r,
    marker=(:square,2,:white),
    markerstrokecolor=:orangered,
    line=(:solid,1,:orangered),
    xlims=(1e-3,1.05),
    ylims=(-2,2),
    label=L"$I_{c}=%$(QM_df[1,:Ic])\mathrm{A}$, $r=%$(r[1])$",
);
plot!(fig0a, 
    xscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
);
plot!(fig0a,
    xlabel="Current (A)",
    ylabel=L"$z_{\mathrm{max}}(F=2)/z_{\mathrm{max}}(F=1)$"
);
display(fig0a)

rc = 0.5*(QM_df[!,:F1] .+ QM_df[!,:F2]);
fig0b=plot(QM_df[!,:Ic], 1000*rc,
    marker=(:square,2,:white),
    markerstrokecolor=:orangered,
    line=(:solid,1,:orangered),
    label=L"$I_{c}=%$(QM_df[1,:Ic])\mathrm{A}$, $z_{c}=%$(1e6 * round(rc[1]; sigdigits=1))\times 10^{-3}\mathrm{\mu m}$",
    xlims=(1e-3,1.05),
)
plot!(fig0b, 
    xscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
);
plot!(fig0b,
    xlabel="Current (A)",
    ylabel=L"$\frac{1}{2}\left(z_{\mathrm{max}}(F=1) + z_{\mathrm{max}}(F=2)\right) \ \mathrm{\mu m}$",
    left_margin=3mm,
    legend=:topleft,
);
display(fig0b)

fig0 = plot(fig0a, fig0b,
layout=(2,1),
link=:x,
size=(1000,700),
left_margin=4mm,)
plot!(fig0[1], xlabel="", xformatter=_->"", bottom_margin=-8mm)


fig00 = plot(xlabel=L"$z \ (\mathrm{mm})$")
for i=1:2:nI
    data = chosen_qm_f1[i][:z_profile][:,[1,3]]
    plot!(fig00,
        data[:,1], data[:,2],
        label=L"$I_{c}=%$(1000*Icoils[i])\mathrm{mA}$",
        line=(:solid,2,colores_current[i]))
end
plot!(fig00,
    yformatter = y -> @sprintf("%.1e", y),
    xlims=(-5,5),
    legend=:outertop,
    legend_columns=5,
    background_color_legend = nothing,
    foreground_color_legend = nothing,)
display(fig00)

fig01 = plot(xlabel=L"$z \ (\mathrm{mm})$")
for i=1:2:nI
    data = chosen_qm_f2[i][:z_profile][:,[1,3]]
    plot!(fig01,
        data[:,1], data[:,2],
        # label=L"$I_{c}=%$(1000*Icoils[i])\mathrm{mA}$",
        line=(:solid,2,colores_current[i]))
end
plot!(fig01,
    yformatter = y -> @sprintf("%.1e", y),
    xlims=(-5,5),
    legend=false,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig01)

fig=plot(fig00, fig01,
layout=(2,1),
size=(900,600),
)
plot!(fig[1], xlabel="", xformatter=_->"", bottom_margin=-5mm)
display(fig)

fig1 = plot(
    xlabel="Currents (A)",
    ylabel="Position (mm)",
);
plot!(fig1,
    QM_df[!,:Ic],QM_df[!,:F1],
    label=L"$F=1$",
    marker=(:circle,2,:white),
    markerstrokecolor=:red,
    line=(:red,1),
);
plot!(fig1, 
    QM_df[!,:Ic],QM_df[!,:F2],
    label=L"$F=2$",
    marker=(:circle,2,:white),
    markerstrokecolor=:blue,
    line=(:blue,1)
);
plot!(fig1,
    xlims=(1e-3,1.05),
    xscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:topleft,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
);
display(fig1)

fig2 = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-Peak (mm)",
);
plot!(fig2,
    QM_df[!,:Ic],QM_df[!,:Δ],
    label=L"$\Delta z$",
    marker=(:circle,2,:white),
    markerstrokecolor=:purple,
    line=(:purple,1),
);
plot!(fig2,
    xlims=(1e-3,1.05),
    ylims=(1e-4,5.0),
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:bottomright,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    
);
display(fig2)

fig = plot(fig1, fig2,
layout=(2,1),
link=:x,
labelfontsize=10)
plot!(fig[1], xlabel="", xformatter=_->"", bottom_margin=-8mm);
display(fig)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++ COQUANTUM DYNAMICS +++++++++++++
cqd_sim_data = JLD2_MyTools.list_keys_jld_cqd(data_cqdup_path)
n_ki = length(cqd_sim_data.ki);
@info "CQD simulation for $(n_ki) ki values"
colores_ki = palette(:darkrainbow, n_ki);

mixdw_cqd = FittingDataCQDQM.plot_combined_cqd_profiles_dict(
    cqd_sim_data[:ki],
    data_cqdup_path,
    data_cqddw_path;
    nz = nz,
    σw_mm = σw_mm,
    λ0 = λ0,
    nI = nI,
    Icoils = Icoils,
    colores_current = colores_current,
    show_plots = false,
    show_table = false,
);
dw_mix_cqd = hcat(values(mixdw_cqd)...);  # size (nI, nki)

up_cqd = Matrix{Float64}(undef, nI, n_ki);
dw_cqd = Matrix{Float64}(undef, nI, n_ki);
Δz_cqd = Matrix{Float64}(undef, nI, n_ki);
Δz_mix_cqd = Matrix{Float64}(undef, nI, n_ki);

for (i,ki) in enumerate(cqd_sim_data.ki)
    dataup = jldopen(data_cqdup_path,"r") do f
        f[JLD2_MyTools.make_keypath_cqd(:up,ki,nz,σw_mm,λ0)]
    end
    up_cqd[:,i] = [dataup[j][:z_max_smooth_spline_mm] for j=1:nI]

    datadw = jldopen(data_cqddw_path,"r") do f
        f[JLD2_MyTools.make_keypath_cqd(:dw,ki,nz,σw_mm,λ0)]
    end
    dw_cqd[:,i] = [datadw[j][:z_max_smooth_spline_mm] for j=1:nI]

    Δz_cqd[:,i] = up_cqd[:,i] .- dw_cqd[:,i]

    Δz_mix_cqd[:,i] = up_cqd[:,i] .- dw_mix_cqd[:,i] 
end

fig1 = plot(xlabel="Current (A)",
    ylabel="Position (mm)");
for (i,ki) in enumerate(cqd_sim_data.ki)
    ki_string = sigdigits_str(ki, 2)
    plot!(fig1,
        Icoils, up_cqd[:,i],
        label=L"$%$(ki_string)$",
        line=(:solid,1,colores_ki[i]),

    )
    plot!(fig1,
        Icoils, dw_cqd[:,i],
        label=false,
        line=(:dot,1,colores_ki[i])

    )
end
annotate!(fig1, [
    (0.05, 1.0, text(L"$\vec{\mu}_{e}=\Uparrow $", :black, 10)),
    (0.05, -1.0, text(L"$\vec{\mu}_{e}=\Downarrow$", :black, 10))
]);
plot!(fig1,
    xlims=(1e-3,1.05),
    xscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerbottom,
    legendtitle=L"$k_{i} \times 10^{-6}$",
    legendtitlefontsize=8,
    legendfontsize=6,
    legend_columns=7,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size=(800,600),
    left_margin=2mm,
)
display(fig1)


fig2 = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-Peak (mm)",
);
for (i,ki) in enumerate(cqd_sim_data.ki)
    # ki_string = @sprintf("%2.3f",ki)
    ki_string = sigdigits_str(ki, 2)
    plot!(fig2,
        Icoils, Δz_cqd[:,i],
        label=L"$%$(ki_string)$",
        line=(:solid,1,colores_ki[i]),

    )
end
annotate!(fig2, [
    (0.05, 1.0, text(L"$\Delta z_{\mathrm{p}-\mathrm{p}}$", :black, 10))
]);
plot!(fig2,
    xlims=(1e-3,1.05),
    ylims=(1e-3,5.0),
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerbottom,
    legendtitle=L"$k_{i} \times 10^{-6}$",
    legendtitlefontsize=8,
    legendfontsize=6,
    legend_columns=7,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size=(800,600),
    left_margin=2mm,
);
display(fig2)

fig3 = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-Peak (mm)",
);
for (i,ki) in enumerate(cqd_sim_data.ki)
    # ki_string = @sprintf("%2.3f",ki)
    ki_string = sigdigits_str(ki, 2)
    plot!(fig3,
        Icoils, Δz_mix_cqd[:,i],
        label=L"$%$(ki_string)$",
        line=(:solid,1,colores_ki[i]),

    )
end
annotate!(fig3, [
    (0.05, 1.0, text(L"$\Delta z^{\mathrm{mix}}_{\mathrm{p}-\mathrm{p}}$", :black, 10))
]);
plot!(fig3,
    xlims=(1e-3,1.05),
    ylims=(1e-3,5.0),
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerbottom,
    legendtitle=L"$k_{i} \times 10^{-6}$",
    legendtitlefontsize=8,
    legendfontsize=6,
    legend_columns=7,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size=(800,600),
    left_margin=2mm,
);
display(fig3)

fig4 = plot(xlabel="Current (A)",
    ylabel="Position (mm)");
for (i,ki) in enumerate(cqd_sim_data.ki)
    ki_string = sigdigits_str(ki, 2)
    plot!(fig4,
        Icoils, up_cqd[:,i],
        label=L"$%$(ki_string)$",
        line=(:solid,1,colores_ki[i]),

    )
end
annotate!(fig4, [
    (0.05, 1.0, text(L"$\vec{\mu}_{e}=\Uparrow $", :black, 10))
]);
plot!(fig4,
    xlims=(1e-3,1.05),
    ylims=(1e-3,5.0),
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerbottom,
    # legendtitle=L"$k_{i} \times 10^{-6}$",
    # legendtitlefontsize=8,
    legendfontsize=6,
    legend_columns=7,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size=(800,600),
    left_margin=2mm,
);
display(fig4)

fig = plot(fig1, fig4, fig2, fig3,
    size=(1600,1300),
    layout=(2,2),
    link=:x,
    left_margin=5mm,
    bottom_margin=5mm,
);
plot!(fig[1], xlabel="", xformatter=_->"",bottom_margin=-9mm,legend=false);
plot!(fig[2], xlabel="", xformatter=_->"",bottom_margin=-9mm,legend=false);
plot!(fig[3], bottom_margin=-10mm);
plot!(fig[4], bottom_margin=2mm,legend=false);
display(fig)




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Select a subset of kᵢ values for interpolation
using Dierckx
# ki_start , ki_stop = 1 , 109 ;
ki_start , ki_stop = 12 , 60 ;
kis_chosen = cqd_sim_data.ki[ki_start:ki_stop]
kmin, kmax = extrema(kis_chosen)
@info @sprintf(
    "Interpolation in the induction term goes from %.1e to %.1e",
    kmin / 1e6,
    kmax / 1e6,
)
# Build 2D cubic spline interpolant: z_max(I, kᵢ)
# s=0 => exact interpolation (no smoothing)
ki_up_itp   = Dierckx.Spline2D(Icoils, kis_chosen, up_cqd[:,ki_start:ki_stop]; kx=3, ky=3, s=0.00);
ki_dw_itp   = Dierckx.Spline2D(Icoils, kis_chosen, dw_cqd[:,ki_start:ki_stop]; kx=3, ky=3, s=0.00);
ki_Δ_itp    = Dierckx.Spline2D(Icoils, kis_chosen, Δz_cqd[:,ki_start:ki_stop]; kx=3, ky=3, s=0.00);
ki_Δmix_itp = Dierckx.Spline2D(Icoils, kis_chosen, Δz_mix_cqd[:,ki_start:ki_stop]; kx=3, ky=3, s=0.00);


# -----------------------------------------------------------------------------
# One small helper – takes raw Z, optionally log-transforms, returns a plot
# -----------------------------------------------------------------------------
function fitting_contour_ki(i_ax, ki_ax, Z, cbar_title; log_scale=true)
    Zp = log_scale ? log10.(max.(Z, 1e-15)) : Z
    lo, hi   = floor(minimum(Zp)), ceil(maximum(Zp))
    decades  = collect(lo:1:hi)
    labels   = [L"10^{%$k}" for k in decades]

    contourf(i_ax, ki_ax, Zp;
        levels         = 101,
        title          = "Fitting contour",
        xlabel         = L"$I_{c}$ (A)",
        ylabel         = L"$k_{i}\times 10^{-6}$",
        color          = :viridis,
        linewidth      = 0.2,
        linestyle      = :dash,
        xaxis          = :log10,
        yaxis          = :log10,
        xlims          = (10e-3, 1.05),
        xticks         = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        clims          = (lo, hi),
        colorbar_ticks = (decades, labels),
        colorbar_title = cbar_title,
    )
end

# -----------------------------------------------------------------------------
# Grid
# -----------------------------------------------------------------------------
i_surface  = range(10e-3, 1.0;    length=101);
ki_surface = range(kmin,  kmax;   length=101);

Zup      = [ki_up_itp(x, y)   for y in ki_surface, x in i_surface];
Zdw      = [ki_dw_itp(x, y)   for y in ki_surface, x in i_surface];
ΔZ_cqd   = [ki_Δ_itp(x, y)    for y in ki_surface, x in i_surface];
ΔZ_mix   = [ki_Δmix_itp(x, y) for y in ki_surface, x in i_surface];

# -----------------------------------------------------------------------------
# Four plots – only data + colorbar title change
# -----------------------------------------------------------------------------
panels = [
    fitting_contour_ki(i_surface, ki_surface, Zup,    L"$ \log(z_{\mathrm{peak}}^{\mathrm{up}}\ \mathrm{(mm)}) $"),
    fitting_contour_ki(i_surface, ki_surface, abs.(Zdw),    L"$ \log( |z_{\mathrm{peak}}^{\mathrm{dw}}| \ \mathrm{(mm)})$"),
    fitting_contour_ki(i_surface, ki_surface, ΔZ_cqd, L"$ \log(z_{\mathrm{p-p}}\ \mathrm{(mm)}) $"),
    fitting_contour_ki(i_surface, ki_surface, ΔZ_mix, L"$ \log(z^{\mathrm{mix}}_{\mathrm{p-p}}\ \mathrm{(mm)}) $"),
];

plot(panels...;
    layout       = @layout([a1 a2; a3 a4]),
    title        = "",
    size         = (800, 600),
    left_margin  = 4mm,
)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++ QUANTUM MECHANICS & COQUANTUM DYNAMICS +++++++++++++
fig1 = plot(
    xlabel="Currents (A)",
    ylabel="Position (mm)",
);
plot!(fig1,
    QM_df[!,:Ic],QM_df[!,:F1],
    label=L"$F=1$",
    seriestype=:scatter,
    marker=(:circle,2,:white),
    markerstrokecolor=:red,
    line=(:red,1),
);
plot!(fig1, 
    QM_df[!,:Ic],QM_df[!,:F2],
    label=L"$F=2$",
    seriestype=:scatter,
    marker=(:circle,2,:white),
    markerstrokecolor=:blue,
    line=(:blue,1)
);
for (i,ki) in enumerate(cqd_sim_data.ki)
    ki_string = sci_label(1e-6*ki; n=2)
    plot!(fig1,
        Icoils, up_cqd[:,i],
        label=L"$k_{i}=$%$(ki_string)",
        line=(:solid,1,colores_ki[i]),

    )
    plot!(fig1,
        Icoils, dw_cqd[:,i],
        label=false,
        line=(:dot,1,colores_ki[i])

    )
end
plot!(fig1,
    xlims=(1e-3,1.05),
    xscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    legendfontsize=6,
    legend_columns=2,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size=(1000,600),
    left_margin=5mm,
)
display(fig1)


fig2 = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-Peak (mm)",
);
for (i,ki) in enumerate(cqd_sim_data.ki)
    ki_string = sci_label(1e-6*ki; n=2)
    plot!(fig2,
        Icoils, Δz_cqd[:,i],
        label=L"$k_{i}=$%$(ki_string)",
        line=(:solid,1,colores_ki[i]),

    )
end
plot!(fig2,
    QM_df[!,:Ic],QM_df[!,:Δ],
    label=L"QM $\Delta z$",
    marker=(:circle,2,:white, 0.55),
    markerstrokecolor=:black,
    line=(:black,2),
);
plot!(fig2,
    xlims=(1e-3,1.05),
    ylims=(1e-4,5.0),
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    legend_columns = 2,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,  
    size=(1000,600);
    left_margin=4mm,
    bottom_margin=2mm,
);
display(fig2)


fig3 = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-Peak (mm)",
);
for (i,ki) in enumerate(cqd_sim_data.ki)
    ki_string = sci_label(1e-6*ki; n=2)
    plot!(fig3,
        Icoils, Δz_mix_cqd[:,i],
        label=L"$k_{i}=$%$(ki_string)",
        line=(:solid,1,colores_ki[i]),

    )
end
plot!(fig3,
    QM_df[!,:Ic],QM_df[!,:Δ],
    label=L"QM $\Delta z$",
    marker=(:circle,2,:white, 0.55),
    markerstrokecolor=:black,
    line=(:black,2),
);
plot!(fig3,
    xlims=(1e-3,1.05),
    ylims=(1e-4,5.0),
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    legend_columns = 2,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,  
    size=(1000,600);
    left_margin=4mm,
    bottom_margin=2mm,
);
display(fig3)


fig = plot(fig1, fig2, fig3,
size=(1200,1000),
layout=(3,1),
link=:x,
left_margin=5mm,
bottom_margin=2mm,
);
plot!(fig[1], xlabel="", xformatter=_->"",bottom_margin=-9mm,legend=false);
plot!(fig[2], xlabel="", xformatter=_->"",bottom_margin=-9mm,legend=false);
plot!(fig[3], xlabel="", xformatter=_->"",bottom_margin=-3mm,legend=:outerbottom, legend_columns=7);
display(fig)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++ EXPERIMENTS ++++++++++++++++++++
EXP_data = OrderedDict{String, NamedTuple}()
for dir in data_directories
    println(dir)
    kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS", "summary", dir, dir * "_report_summary.jld2")

    EXP_data[dir] = jldopen(kk_path, "r") do file
        ic = file["meta/Currents"]
        bz = file["meta/BzTesla"]

        dd = file[JLD2_MyTools.make_keypath_exp(dir, nz, λ0)]

        F1 = dd[:fw_F1_peak_pos_raw]
        F2 = dd[:fw_F2_peak_pos_raw]

        zc = 0.5 * (F1[1][1] + F2[1][1])
        δc = 0.5 * sqrt(F1[2][1]^2 + F2[2][1]^2)

        (
            Ic = ic,
            Bz = bz,
            F1 = F1,
            F2 = F2,
            C0 = dd[:centroid_fw_mm],
            C00 = [zc, δc]
        )
    end
end

fig1=plot(
    xlabel="Currents (A)",
    ylabel=L"$z^{F=1}_{max}$ (mm)",
)
hspan!(fig1,[-6.5e-3,6.5e-3], color=:gray68, fillalpha=0.8, label="pixel size")
for i = 1:n_data
    data = DataFrame(
            :Ic => EXP_data[data_directories[i]].Ic, 
            :F1 => EXP_data[data_directories[i]].F1[1] .- EXP_data[data_directories[i]].C00[1]
            )
    
    data_sub = data[data.Ic .< 0.015, :]
    β = fit_poly2(data_sub[!,:Ic], data_sub[!,:F1])
    println(β)
    # smooth curve for plotting
    xfit = range(0.0, 15e-3, length = 300)
    yfit = β[1] .+ β[2] .* xfit .+ β[3] .* xfit.^2

    plot!(fig1,data_sub[!,:Ic], data_sub[!,:F1] ,
    label=data_directories[i],
    marker=(:square,3,:white),
    markerstrokewidth=2,
    markerstrokecolor=colores_data[i],
    line=(:dash, 1, colores_data[i]))
    # fitted curve
    plot!(fig1,
        xfit,
        yfit,
        label = false,
        lw = 2,
        color = colores_data[i],
    )
end
plot!(fig1,legend=:outerright,
    xlims=(-0.1e-3,15e-3),
    ylims=(-10e-3,20e-3),
    # xscale=:log10,
    # yscale=:log10,
)

fig2=plot(
    xlabel="Currents (A)",
    ylabel=L"$z^{F=2}_{max}$ (mm)",
)
hspan!(fig2,[-6.5e-3,6.5e-3], color=:gray68, fillalpha=0.8, label="pixel size")
for i = 1:n_data
    data = DataFrame(
            :Ic => EXP_data[data_directories[i]].Ic, 
            :F2 => EXP_data[data_directories[i]].F2[1] .- EXP_data[data_directories[i]].C00[1]
            )
    
    data_sub = data[data.Ic .< 0.015, :]
    β = fit_poly2(data_sub[!,:Ic], data_sub[!,:F2])
    println(β)
    # smooth curve for plotting
    xfit = range(0.0, 15e-3, length = 300)
    yfit = β[1] .+ β[2] .* xfit .+ β[3] .* xfit.^2

    plot!(fig2,data_sub[!,:Ic], data_sub[!,:F2] ,
    label=data_directories[i],
    marker=(:square,3,:white),
    markerstrokewidth=2,
    markerstrokecolor=colores_data[i],
    line=(:dash, 1, colores_data[i]))
    # fitted curve
    plot!(fig2,
        xfit,
        yfit,
        label = false,
        lw = 2,
        color = colores_data[i],
    )
end
plot!(fig2,legend=:outerright,
    xlims=(-0.1e-3,15e-3),
    ylims=(-20e-3,12e-3),
    # xscale=:log10,
    # yscale=:log10,
)

fig3 = plot(xlabel="Currents (A)",
ylabel=L"$z_{F=1}-z_{F=2}$ (mm)")
hspan!(fig3,[-6.5e-3,6.5e-3], color=:gray68, fillalpha=0.8, label="pixel size")
for i = 1:n_data
    data = DataFrame(
            :Ic => EXP_data[data_directories[i]].Ic, 
            :Δ  => EXP_data[data_directories[i]].F1[1] .- EXP_data[data_directories[i]].F2[1]
            )
    
    data_sub = data[data.Ic .< 0.015, :]
    β = fit_poly2(data_sub[!,:Ic], data_sub[!,:Δ])
    println(β)
    # smooth curve for plotting
    xfit = range(0.0, 15e-3, length = 300)
    yfit = β[1] .+ β[2] .* xfit .+ β[3] .* xfit.^2

    plot!(fig3,data_sub[!,:Ic], data_sub[!,:Δ] ,
        label=data_directories[i],
        marker=(:square,3,:white),
        markerstrokewidth=2,
        markerstrokecolor=colores_data[i],
        line=(:dash, 1, colores_data[i]))

    plot!(fig3,
        xfit,
        yfit,
        label = false,
        lw = 2,
        color = colores_data[i],
    )
end
plot!(fig3,legend=:outerright,
    xlims=(-0.1e-3,15e-3),
    ylims=(-10e-3,20e-3),
    # xscale=:log10,
    # yscale=:log10,
)

fig = plot(fig1, fig2, fig3,
layout=(3,1),
size=(600,700),
link=:x,
left_margin=2mm,
bottom_margin=-2mm,)
plot!(fig[1], xlabel="", xformatter=_->"", bottom_margin=-7mm)
plot!(fig[2], xlabel="", xformatter=_->"", bottom_margin=-7mm)
display(fig)


EXP_data_processed = OrderedDict{String, DataFrame}()
for 𝓁 = 1:n_data
    dir = data_directories[𝓁]
    data_dir = DataFrame(hcat(
            EXP_data[dir].Ic,
            EXP_data[dir].F1[1] .- EXP_data[dir].C00[1],
            round.(sqrt.(EXP_data[dir].F1[2].^2 .+ EXP_data[dir].C00[2].^2); sigdigits=1),
            EXP_data[dir].F2[1] .- EXP_data[dir].C00[1],
            round.(sqrt.(EXP_data[dir].F2[2].^2 .+ EXP_data[dir].C00[2].^2); sigdigits=1),
            EXP_data[dir].F1[1] .- EXP_data[dir].F2[1],
            round.(sqrt.(EXP_data[dir].F1[2].^2 .+ EXP_data[dir].F2[2].^2); sigdigits=1),
        ),
        [:Ic, :F1, :ErrF1, :F2, :ErrF2, :Δ, :ErrΔ]
        )
    EXP_data_processed[dir] = data_dir

    pretty_table(data_dir; 
                alignment =:c,
                title         = "EXPERIMENT $(data_directories[𝓁])",
                formatters = [fmt__printf("%8.4f", [1]), fmt__printf("%8.4f",[2,4,6]), fmt__printf("%8.3f",[3,5,7])],
                style = TextTableStyle(
                        first_line_column_label = crayon"yellow bold",
                        table_border  = crayon"blue bold",
                        title = crayon"bold red"
                ),
                table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
                equal_data_column_widths= true,
    )


    data_f1 = data_dir[data_dir.F1 .> 0, :]

    figa = plot(xlabel="Currents (A)",
    ylabel=L"$F=1$ position (mm)");
    plot!(figa, data_f1.Ic, data_f1.F1,
        yerror = data_f1.ErrF1,
        label="$(data_directories[𝓁])",
        marker=(:square,2,:white),
        markerstrokecolor=:black,
        line=(:solid,:black)
    );
    plot!(figa,
        QM_df[!,:Ic],QM_df[!,:F1],
        label=L"$F=1$",
        marker=(:circle,2,:white),
        markerstrokecolor=:red,
        line=(:red,2),
    );
    for (i,ki) in enumerate(cqd_sim_data.ki)
        ki_string = sci_label(1e-6*ki; n=2)
        plot!(figa,
            Icoils, up_cqd[:,i],
            label=L"%$(ki_string)",
            line=(:solid,1,colores_ki[i], 0.5),
        )
    end
    plot!(figa,
        xlims=(5e-3,1.05),
        ylims=(1e-3,2.0),
        legend=:outerright,
        legend_columns = 2,
        legendfontsize=6,
        background_color_legend = nothing,
        foreground_color_legend = nothing,
        size=(1000,400),
        left_margin=5mm,
        bottom_margin=5mm
    );
    figb = deepcopy(figa);
    # display(figb)
    plot!(figa,
        xscale=:log10,
        yscale=:log10,
        xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    );
    # display(figa)

    fig = plot(figa,figb,
        layout=(2,1),
        size=(1200,1000),
        top_margin=5mm)
    display(fig)


    data_Δz = data_dir[data_dir.Δ .> 0, :];

    figc = plot(
        xlabel="Currents (A)",
        ylabel="Peak-to-Peak (mm)",
    );
    for (i,ki) in enumerate(cqd_sim_data.ki)
        ki_string = sci_label(1e-6*ki; n=2)
        plot!(figc,
            Icoils, Δz_mix_cqd[:,i],
            label=L"%$(ki_string)",
            line=(:solid,1,colores_ki[i]),

        )
    end
    plot!(figc,
        QM_df[!,:Ic],QM_df[!,:Δ],
        label=L"QM $\Delta z$",
        marker=(:circle,2,:white, 0.55),
        markerstrokecolor=:lime,
        line=(:lime,1),
    );
    plot!(figc, data_Δz.Ic, data_Δz.Δ,
        yerror = data_Δz.ErrΔ,
        label="$(data_directories[𝓁])",
        marker=(:square,2,:white),
        markerstrokecolor=:black,
        line=(:solid,:black)
    );
    plot!(figc,
        xlims=(5e-6,1.05),
        ylims=(1e-3,4.2),
        legend=:outerright,
        legend_columns = 2,
        legendfontsize=6,
        background_color_legend = nothing,
        foreground_color_legend = nothing,
        size=(1000,400),
        left_margin=5mm,
        bottom_margin=5mm
    );
    figd = deepcopy(figc);
    # display(figd)
    plot!(figc,
        xlims=(1e-3,1.05),
        ylims=(1e-4,5.0),
        xscale=:log10,
        yscale=:log10,
        xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        legend=:outerright,
        legend_columns = 2,
        legendfontsize=6,
        background_color_legend = nothing,
        foreground_color_legend = nothing,  
    );
    # display(figc)

    fig = plot(figc,figd,
        layout=(2,1),
        size=(1200,1000),
        top_margin=7mm)
    display(fig)

end

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++ DATA COMPARISON
figa = plot(
    xlabel="Currents (A)",
    ylabel="Position (mm)",
);
for (i,dir) in enumerate(data_directories)
    plot!(figa,
        EXP_data_processed[dir].Ic, EXP_data_processed[dir].F1,
        yerror = EXP_data_processed[dir].ErrF1,
        label= dir,
        marker=(:circle,2,:white,0.6),
        markerstrokecolor=colores_data[i],
        line=(:solid,1,colores_data[i])
    )
end
plot!(figa,
    QM_df[!,:Ic],QM_df[!,:F1],
    label=L"QM : $F=1$",
    seriestype=:scatter,
    marker=(:diamond,2,:white),
    markerstrokecolor=:black,
    line=(:black,1),
);
plot!(figa,
    legend=:bottomright,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    xlims=(10e-3, 1.05),
    ylims=(1e-3, 2.05)
);
figa1= deepcopy(figa);
plot!(figa,
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
);

figb = plot(
    xlabel="Currents (A)",
    ylabel="Position (mm)",
);
for (i,dir) in enumerate(data_directories)
    plot!(figb,
        EXP_data_processed[dir].Ic, EXP_data_processed[dir].F2,
        yerror = EXP_data_processed[dir].ErrF2,
        label= dir,
        marker=(:circle,2,:white,0.6),
        markerstrokecolor=colores_data[i],
        line=(:solid,1,colores_data[i])
    )
end
plot!(figb,
    QM_df[!,:Ic],QM_df[!,:F2],
    label=L"QM : $F=2$",
    seriestype=:scatter,
    marker=(:diamond,2,:white),
    markerstrokecolor=:black,
    line=(:black,1),
);
plot!(figb,
    xlims=(10e-3, 1.05),
    ylims=(-2.05, -6e-3),
);
plot!(figb,
    legend=:topright,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
);


figc = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-position (mm)"
);
for (i,dir) in enumerate(data_directories)
    plot!(figc,
        EXP_data_processed[dir].Ic, EXP_data_processed[dir].Δ ,
        yerror = EXP_data_processed[dir].ErrΔ,
        label= dir,
        marker=(:circle,2,:white,0.6),
        markerstrokecolor=colores_data[i],
        line=(:solid,1,colores_data[i])
    )
end
plot!(figc,
    QM_df[!,:Ic],QM_df[!,:Δ],
    label=L"QM : $\Delta z$",
    seriestype=:scatter,
    marker=(:diamond,2,:white),
    markerstrokecolor=:black,
    line=(:black,1),
);
plot!(figc,
    legend=:bottomright,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    xlims=(10e-3, 1.05),
    ylims=(1e-3, 4.05),
);
figc1 = deepcopy(figc);
plot!(figc,xscale=:log10, yscale=:log10,
 xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
 yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
);

fig = plot(figa,figa1, figb,figc, figc1,
labelfontsize = 10,
layout=@layout[ a1 a2 ; a3; a4 a5],
size=(1200,1000),
top_margin=2mm,
left_margin=5mm,
bottom_margin=3mm,
);
display(fig)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++
threshold = 0.030 ; # lower cut-off for experimental currents

tol_grouping = 0.04;
Ics = [EXP_data_processed[dir].Ic for dir in data_directories];
clusters = MyExperimentalAnalysis.cluster_by_tolerance(Ics; tol=tol_grouping);
for s in clusters.summary
    println("Value group ≈ $(@sprintf("%1.3f", s.mean_val)) ± $(round(s.std_val; sigdigits=1)) \t appears in datasets: ", s.datasets)
end
Ic_grouped  = round.(getproperty.(clusters.summary, :mean_val); digits=6);
δIc_grouped = round.(getproperty.(clusters.summary, :std_val);  sigdigits=2);
mask = Ic_grouped .>= threshold;
grouped_exp_current = DataFrame(
    Ic  = Ic_grouped[mask],
    δIc = δIc_grouped[mask],
)

fig=plot(grouped_exp_current.Ic, 
    ribbon=grouped_exp_current.δIc,
    marker=(:diamond, 2, :white),
    markerstrokecolor=:red,
    line=(:solid,1,:red),
    color=:red,
    fillalpha=0.2,
    label=false,
    xlabel="current number",
    ylabel="Current (A)",
    ylims=(7e-4,1.05),
);
plot!(
    yscale=:log10,
    legend=:bottomright,
);
display(fig)

CURRENT_ROW_START = [first_gt_idx(EXP_data_processed[t], :Ic, threshold) for t in keys(EXP_data_processed)];
Ic_sets  = [EXP_data_processed[t][i:end, :Ic]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];
F1_sets  = [EXP_data_processed[t][i:end, :F1]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];
F2_sets  = [EXP_data_processed[t][i:end, :F2]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];
Δ_sets   = [EXP_data_processed[t][i:end, :Δ]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];
σF1_sets = [EXP_data_processed[t][i:end, :ErrF1]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];
σF2_sets = [EXP_data_processed[t][i:end, :ErrF2]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];
σΔ_sets  = [EXP_data_processed[t][i:end, :ErrΔ]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];


# pick a log-spaced grid across the overall I-range (nice for decades-wide currents)
i_sampled_length = 65 ;
xlo, xhi = maximum([minimum(first.(Ic_sets)),threshold]) ,  maximum([maximum(last.(Ic_sets)),1.01]);
Ic_sampling  = exp10.(range(log10(xlo), log10(xhi), length=i_sampled_length))


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Monte Carlo analysis
MC_data_F1 = FittingDataCQDQM.combine_on_grid_mc_weighted(Ic_sets, F1_sets; σxsets=nothing, σysets=σF1_sets, B=5000, xq=grouped_exp_current.Ic);
MC_data_F2 = FittingDataCQDQM.combine_on_grid_mc_weighted(Ic_sets, F2_sets; σxsets=nothing, σysets=σF2_sets, B=5000, xq=grouped_exp_current.Ic);
MC_data_Δ  = FittingDataCQDQM.combine_on_grid_mc_weighted(Ic_sets, Δ_sets;  σxsets=nothing, σysets=σΔ_sets,  B=5000, xq=grouped_exp_current.Ic);

function build_peak_panel(
    exp_data_processed,
    data_directories,
    current_row_start,
    colors_data,
    qm_df,
    mc_data;
    exp_col::Symbol,
    err_col::Symbol,
    qm_col::Symbol,
    ylabel,
    threshold,
    ylims,
    take_abs::Bool=false,
    legend=:bottomright,
)
    f = take_abs ? abs : identity

    fig = plot(
        xlabel = "Current (A)",
        ylabel = ylabel,
        xlims  = (0.6 * threshold, 1.0),
        ylims  = ylims,
        xscale = :log10,
        yscale = :log10,
        legend = legend,
        foreground_color_legend = nothing,
    )

    for (i, dir) in enumerate(data_directories)
        df   = exp_data_processed[dir]
        rows = current_row_start[i]:nrow(df)

        xs  = df[rows, :Ic]
        ys  = f.(df[rows, exp_col])
        dys = df[rows, err_col]

        scatter!(
            fig, xs, ys;
            yerror            = dys,
            label             = dir,
            marker            = (:circle, :white, 3),
            markerstrokecolor = colors_data[i],
            markerstrokewidth = 1,
        )
    end

    plot!(
        fig,
        qm_df[!, :Ic],
        f.(qm_df[!, qm_col]);
        label = "QM",
        line  = (:black, :dashdot, 2),
    )

    plot!(
        fig,
        mc_data.xq,
        f.(mc_data.μ);
        ribbon = mc_data.σ_tot,
        label  = false,
        color  = :maroon1,
    )

    return fig
end

function plot_weighted_mc_panels(
    exp_data_processed,
    data_directories,
    current_row_start,
    colors_data,
    qm_df,
    mc_data_f1,
    mc_data_f2,
    mc_data_Δ;
    threshold,
)
    specs = (
        (
            exp_col  = :F1,
            err_col  = :ErrF1,
            qm_col   = :F1,
            mc_data  = mc_data_f1,
            ylabel   = L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
            ylims    = (1e-3, 2.0),
            take_abs = false,
        ),
        (
            exp_col  = :F2,
            err_col  = :ErrF2,
            qm_col   = :F2,
            mc_data  = mc_data_f2,
            ylabel   = L"$F_{2} : |z_{\mathrm{peak}}|$ (mm)",
            ylims    = (1e-3, 2.0),
            take_abs = true,
        ),
        (
            exp_col  = Symbol("Δ"),
            err_col  = Symbol("ErrΔ"),
            qm_col   = Symbol("Δ"),
            mc_data  = mc_data_Δ,
            ylabel   = L"$\Delta : z_{\mathrm{peak}}$ (mm)",
            ylims    = (1e-3, 4.0),
            take_abs = true,
        ),
    )

    figs = map(specs) do s
        build_peak_panel(
            exp_data_processed,
            data_directories,
            current_row_start,
            colors_data,
            qm_df,
            s.mc_data;
            exp_col   = s.exp_col,
            err_col   = s.err_col,
            qm_col    = s.qm_col,
            ylabel    = s.ylabel,
            threshold = threshold,
            ylims     = s.ylims,
            take_abs  = s.take_abs,
        )
    end

    fig1, fig2, fig3 = figs

    fig = plot(
        fig1, fig2, fig3;
        suptitle    = "Weighted Monte Carlo",
        layout      = @layout([a1 a2; a3]),
        link        = :y,
        size        = (900, 700),
        left_margin = 3mm,
    )

    return fig1, fig2, fig3, fig
end

fig1, fig2, fig3, fig = plot_weighted_mc_panels(
    EXP_data_processed,
    data_directories,
    CURRENT_ROW_START,
    colores_data,
    QM_df,
    MC_data_F1,
    MC_data_F2,
    MC_data_Δ;
    threshold = threshold,
)

# display(fig1)
# display(fig2)
# display(fig3)
display(fig)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Spline interpolation

# i_xx0 = unique(round.(sort(union(Ic_sampling,grouped_exp_current.Ic)); digits=6))
i_xx0 = unique(round.(sort(grouped_exp_current.Ic); digits=6))
i_sampled_length = length(i_xx0)

function compute_cubic_spline_summary(
    exp_data_processed,
    data_directories,
    current_row_start,
    i_xx0,
)
    n_data = length(data_directories)
    n_grid = length(i_xx0)

    z1_mat = Matrix{Float64}(undef, n_data, n_grid)
    z2_mat = Matrix{Float64}(undef, n_data, n_grid)
    Δz_mat = Matrix{Float64}(undef, n_data, n_grid)

    for (i, dir) in enumerate(data_directories)
        df   = exp_data_processed[dir]
        rows = current_row_start[i]:nrow(df)
        xs   = df[rows, :Ic]

        spl1 = BSplineKit.extrapolate(
            BSplineKit.interpolate(xs, df[rows, :F1], BSplineKit.BSplineOrder(4), BSplineKit.Natural()),
            BSplineKit.Linear(),
        )
        spl2 = BSplineKit.extrapolate(
            BSplineKit.interpolate(xs, df[rows, :F2], BSplineKit.BSplineOrder(4), BSplineKit.Natural()),
            BSplineKit.Linear(),
        )
        splΔ = BSplineKit.extrapolate(
            BSplineKit.interpolate(xs, df[rows, :Δ], BSplineKit.BSplineOrder(4), BSplineKit.Natural()),
            BSplineKit.Linear(),
        )

        z1_mat[i, :] = spl1.(i_xx0)
        z2_mat[i, :] = spl2.(i_xx0)
        Δz_mat[i, :] = splΔ.(i_xx0)
    end

    spl_zf1  = vec(mean(z1_mat, dims = 1))
    spl_δzf1 = vec(std(z1_mat; dims = 1, corrected = true)) ./ sqrt(n_data)

    spl_zf2  = vec(mean(z2_mat, dims = 1))
    spl_δzf2 = vec(std(z2_mat; dims = 1, corrected = true)) ./ sqrt(n_data)

    spl_Δz   = vec(mean(Δz_mat, dims = 1))
    spl_δzΔ  = vec(std(Δz_mat; dims = 1, corrected = true)) ./ sqrt(n_data)

    data_spl = OrderedDict(
        :Ic  => i_xx0,
        :F1  => spl_zf1,
        :σF1 => spl_δzf1,
        :F2  => spl_zf2,
        :σF2 => spl_δzf2,
        :Δ   => spl_Δz,
        :σΔ  => spl_δzΔ,
    )

    return data_spl
end

function plot_cubic_spline_summary(
    exp_data_processed,
    data_directories,
    current_row_start,
    colors_data,
    qm_df,
    spl;
    threshold,
)
    function build_panel(; data_col, err_col, qm_col, ylabel, ylims, mean_key, sem_key, take_abs=false)
        f = take_abs ? abs : identity

        fig = plot(
            xlabel = "Current (A)",
            ylabel = ylabel,
            xlims  = (0.6 * threshold, 1.0),
            ylims  = ylims,
            legend = :bottomright,
            foreground_color_legend = nothing,
            xaxis  = :log10,
            yaxis  = :log10,
            xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
            yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        )

        for (i, dir) in enumerate(data_directories)
            df   = exp_data_processed[dir]
            rows = current_row_start[i]:nrow(df)

            xs  = df[rows, :Ic]
            ys  = df[rows, data_col]
            dys = df[rows, err_col]

            spl_i = BSplineKit.extrapolate(
                BSplineKit.interpolate(
                    xs, ys,
                    BSplineKit.BSplineOrder(4),
                    BSplineKit.Natural(),
                ),
                BSplineKit.Linear(),
            )

            plot!(
                fig,
                spl[:Ic],
                f.(spl_i.(spl[:Ic]));
                label = false,
                line  = (colors_data[i], 1),
            )

            scatter!(
                fig,
                xs,
                f.(ys);
                yerror            = dys,
                label             = dir,
                marker            = (:circle, :white, 3),
                markerstrokecolor = colors_data[i],
                markerstrokewidth = 1,
            )
        end

        plot!(
            fig,
            qm_df[!, :Ic],
            f.(qm_df[!, qm_col]);
            label = "QM",
            line  = (:black, :dashdot, 2),
        )

        plot!(
            fig,
            spl[:Ic],
            f.(spl[mean_key]);
            ribbon    = spl[sem_key],
            fillalpha = 0.40,
            fillcolor = :green3,
            label     = false,
            line      = (:dash, :green3, 2),
        )

        return fig
    end

    fig1 = build_panel(
        data_col = :F1,
        err_col  = :ErrF1,
        qm_col   = :F1,
        ylabel   = L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
        ylims    = (1e-3, 2.0),
        mean_key = :F1,
        sem_key  = :σF1,
    )

    fig2 = build_panel(
        data_col = :F2,
        err_col  = :ErrF2,
        qm_col   = :F2,
        ylabel   = L"$F_{2} : |z_{\mathrm{peak}}|$ (mm)",
        ylims    = (1e-3, 2.0),
        mean_key = :F2,
        sem_key  = :σF2,
        take_abs = true,
    )

    fig3 = build_panel(
        data_col = :Δ,
        err_col  = Symbol("ErrΔ"),
        qm_col   = :Δ,
        ylabel   = L"$\Delta : z_{\mathrm{peak}}$ (mm)",
        ylims    = (1e-3, 4.0),
        mean_key = :Δ,
        sem_key  = :σΔ,
    )

    fig = plot(
        fig1, fig2, fig3;
        suptitle    = "Interpolation: cubic splines",
        layout      = @layout([a1 a2; a3]),
        link        = :y,
        size        = (900, 700),
        left_margin = 2mm,
    )

    return fig1, fig2, fig3, fig
end

data_spl = compute_cubic_spline_summary(
    EXP_data_processed,
    data_directories,
    CURRENT_ROW_START,
    i_xx0,
)

fig1, fig2, fig3, fig = plot_cubic_spline_summary(
    EXP_data_processed,
    data_directories,
    CURRENT_ROW_START,
    colores_data,
    QM_df,
    data_spl;
    threshold = threshold,
)

# display(fig1)
# display(fig2)
# display(fig3)
display(fig)



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Fit Spline interpolation
function compute_smoothing_spline_summary(
    exp_data_processed,
    data_directories,
    current_row_start,
    i_xx0;
    λ0::Real = 0.005,
)
    n_data = length(data_directories)
    n_grid = length(i_xx0)

    z1_mat = Matrix{Float64}(undef, n_data, n_grid)
    z2_mat = Matrix{Float64}(undef, n_data, n_grid)
    Δz_mat = Matrix{Float64}(undef, n_data, n_grid)

    for (i, dir) in enumerate(data_directories)
        df   = exp_data_processed[dir]
        rows = current_row_start[i]:nrow(df)
        xs   = df[rows, :Ic]

        ys1  = df[rows, :F1]
        δy1  = df[rows, :ErrF1]
        spl1 = BSplineKit.extrapolate(
            BSplineKit.fit(
                BSplineKit.BSplineOrder(4),
                xs,
                ys1,
                λ0,
                BSplineKit.Natural();
                weights = 1 ./ δy1.^2,
            ),
            BSplineKit.Smooth(),
        )

        ys2  = df[rows, :F2]
        δy2  = df[rows, :ErrF2]
        spl2 = BSplineKit.extrapolate(
            BSplineKit.fit(
                BSplineKit.BSplineOrder(4),
                xs,
                ys2,
                λ0,
                BSplineKit.Natural();
                weights = 1 ./ δy2.^2,
            ),
            BSplineKit.Smooth(),
        )

        ysΔ  = df[rows, :Δ]
        δyΔ  = df[rows, Symbol("ErrΔ")]
        splΔ = BSplineKit.extrapolate(
            BSplineKit.fit(
                BSplineKit.BSplineOrder(4),
                xs,
                ysΔ,
                λ0,
                BSplineKit.Natural();
                weights = 1 ./ δyΔ.^2,
            ),
            BSplineKit.Smooth(),
        )

        z1_mat[i, :] = spl1.(i_xx0)
        z2_mat[i, :] = spl2.(i_xx0)
        Δz_mat[i, :] = splΔ.(i_xx0)
    end

    zf1_fit  = vec(mean(z1_mat, dims=1))
    δzf1_fit = vec(std(z1_mat; dims=1, corrected=true)) ./ sqrt(n_data)

    zf2_fit  = vec(mean(z2_mat, dims=1))
    δzf2_fit = vec(std(z2_mat; dims=1, corrected=true)) ./ sqrt(n_data)

    Δz_fit   = vec(mean(Δz_mat, dims=1))
    δzΔ_fit  = vec(std(Δz_mat; dims=1, corrected=true)) ./ sqrt(n_data)

    data_fit = OrderedDict(
        :Ic  => i_xx0,
        :F1  => zf1_fit,
        :σF1 => δzf1_fit,
        :F2  => zf2_fit,
        :σF2 => δzf2_fit,
        :Δ   => Δz_fit,
        :σΔ  => δzΔ_fit,
    )

    return data_fit
end

function plot_smoothing_spline_summary(
    exp_data_processed,
    data_directories,
    current_row_start,
    colors_data,
    qm_df,
    fit_data;
    threshold,
    λ0::Real = 0.005,
)
    function build_panel(; data_col, err_col, qm_col, ylabel, ylims, mean_key, sem_key, take_abs=false, title=nothing)
        f = take_abs ? abs : identity

        fig = plot(
            xlabel = "Current (A)",
            ylabel = ylabel,
            xaxis  = :log10,
            yaxis  = :log10,
            xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
            yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
            xlims  = (0.6 * threshold, 1.0),
            ylims  = ylims,
            foreground_color_legend = nothing,
            title = isnothing(title) ? "" : title,
        )

        for (i, dir) in enumerate(data_directories)
            df   = exp_data_processed[dir]
            rows = current_row_start[i]:nrow(df)

            xs  = df[rows, :Ic]
            ys  = df[rows, data_col]
            dys = df[rows, err_col]

            spl_i = BSplineKit.extrapolate(
                BSplineKit.fit(
                    BSplineKit.BSplineOrder(4),
                    xs,
                    ys,
                    λ0,
                    BSplineKit.Natural();
                    weights = 1 ./ dys.^2,
                ),
                BSplineKit.Smooth(),
            )

            scatter!(
                fig,
                xs,
                f.(ys);
                label             = dir,
                marker            = (:circle, :white, 3),
                markerstrokecolor = colors_data[i],
                markerstrokewidth = 1,
            )

            plot!(
                fig,
                fit_data[:Ic],
                f.(spl_i.(fit_data[:Ic]));
                label = false,
                line  = (colors_data[i], 1),
            )
        end

        plot!(
            fig,
            qm_df[!, :Ic],
            f.(qm_df[!, qm_col]);
            label = "QM",
            line  = (:black, :dashdot, 2),
        )

        plot!(
            fig,
            fit_data[:Ic],
            f.(fit_data[mean_key]);
            ribbon    = fit_data[sem_key],
            fillalpha = 0.40,
            fillcolor = :goldenrod,
            label     = false,
            line      = (:dash, :goldenrod, 2),
        )

        return fig
    end

    fig1 = build_panel(
        data_col = :F1,
        err_col  = :ErrF1,
        qm_col   = :F1,
        ylabel   = L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
        ylims    = (8e-3, 2.0),
        mean_key = :F1,
        sem_key  = :σF1,
        title    = "Fit smoothing cubic spline",
    )

    fig2 = build_panel(
        data_col = :F2,
        err_col  = :ErrF2,
        qm_col   = :F2,
        ylabel   = L"$F_{2} : |z_{\mathrm{peak}}|$ (mm)",
        ylims    = (8e-3, 2.0),
        mean_key = :F2,
        sem_key  = :σF2,
        take_abs = true,
    )

    fig3 = build_panel(
        data_col = :Δ,
        err_col  = Symbol("ErrΔ"),
        qm_col   = :Δ,
        ylabel   = L"$\Delta : z_{\mathrm{peak}}$ (mm)",
        ylims    = (8e-3, 4.0),
        mean_key = :Δ,
        sem_key  = :σΔ,
        title    = "Fit smoothing cubic spline",
    )

    fig = plot(
        fig1, fig2, fig3;
        layout      = @layout([a1 a2; a3]),
        link        = :y,
        size        = (900, 700),
        left_margin = 2mm,
    )

    return fig1, fig2, fig3, fig
end

data_fit = compute_smoothing_spline_summary(
    EXP_data_processed,
    data_directories,
    CURRENT_ROW_START,
    i_xx0;
    λ0 = 0.005,
)

fig1, fig2, fig3, fig = plot_smoothing_spline_summary(
    EXP_data_processed,
    data_directories,
    CURRENT_ROW_START,
    colores_data,
    QM_df,
    data_fit;
    threshold = threshold,
    λ0 = 0.005,
)

# display(fig1)
# display(fig2)
# display(fig3)
display(fig)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Combined results

combined_result = OrderedDict(
    :Current => (
        Ic  = grouped_exp_current.Ic,
        σIc = grouped_exp_current.δIc,
    ),

    :MonteCarlo => (
        zF1      = MC_data_F1.μ,
        σzF1     = MC_data_F1.σ_tot,
        zF2      = MC_data_F2.μ,
        σzF2     = MC_data_F2.σ_tot,
        Δz       = MC_data_Δ.μ,
        σΔz      = MC_data_Δ.σ_tot,
        Δz_alt   = MC_data_F1.μ .- MC_data_F2.μ,
        σΔz_alt  = sqrt.( (MC_data_F1.σ_tot).^2 .+ (MC_data_F2.σ_tot).^2 ),
    ),

    :SplineInter => (
        zF1      = data_spl[:F1],
        σzF1     = data_spl[:σF1],
        zF2      = data_spl[:F2],
        σzF2     = data_spl[:σF2],
        Δz       = data_spl[:Δ],
        σΔz      = data_spl[:σΔ],
        Δz_alt   = data_spl[:F1] .- data_spl[:F2],
        σΔz_alt  = sqrt.( (data_spl[:σF1]).^2 .+ (data_spl[:σF2]).^2 ),
    ),

    :SplineFit => (
        zF1      = data_fit[:F1],
        σzF1     = data_fit[:σF1],
        zF2      = data_fit[:F2],
        σzF2     = data_fit[:σF2],
        Δz       = data_fit[:Δ],
        σΔz      = data_fit[:σΔ],
        Δz_alt   = data_fit[:F1] .- data_fit[:F2],
        σΔz_alt  = sqrt.( (data_fit[:σF1]).^2 .+ (data_fit[:σF2]).^2 ),
    ),

)

fig1 = plot(xlabel="Currents (A)", 
    ylabel=L"z_{\mathrm{peak}}^{F=1} \ (\mathrm{mm})")
plot!(fig1, combined_result[:Current].Ic, combined_result[:MonteCarlo].zF1,
    ribbon= combined_result[:MonteCarlo].σzF1,
    label= "Monte Carlo",
    color=:red )
plot!(fig1,
    combined_result[:Current].Ic, combined_result[:SplineInter].zF1 ,
    ribbon= combined_result[:SplineInter].σzF1,
    label="Spl. Interpolation",
    color=:blue)
plot!(fig1,
    combined_result[:Current].Ic, combined_result[:SplineFit].zF1 ,
    ribbon= combined_result[:SplineFit].σzF1,
    label="Spl. Fit",
    color=:darkgreen )
plot!(fig1,
    xlims=(0.80*threshold,1.05),
    ylims=(10e-3,2.05),
    xscale=:log10,yscale=:log10,
    foreground_color_legend=nothing,
    xticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),)

fig2 = plot(xlabel="Currents (A)", ylabel=L"z_{\mathrm{peak}}^{F=2}  \ (\mathrm{mm})")    
plot!(fig2,
    combined_result[:Current].Ic, combined_result[:MonteCarlo].zF2,
    ribbon= combined_result[:MonteCarlo].σzF2,
    label= "Monte Carlo",
    color=:red )
plot!(fig2,
    combined_result[:Current].Ic, combined_result[:SplineInter].zF2 ,
    ribbon= combined_result[:SplineInter].σzF2,
    label="Spl. Interpolation",
    color=:blue)
plot!(fig2,
    combined_result[:Current].Ic, combined_result[:SplineFit].zF2 ,
    ribbon= combined_result[:SplineFit].σzF2,
    label="Spl. Fit",
    color=:darkgreen )
plot!(fig2,
    legend=:bottomleft,
    foreground_color_legend=nothing,
    xlims=(0.80*threshold,1.05),
    xticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xscale=:log10)

fig3 = plot(xlabel="Currents (A)", ylabel=L"z_{\mathrm{peak-peak}} \ (\mathrm{mm})")    
plot!(fig3,
    combined_result[:Current].Ic, combined_result[:MonteCarlo].Δz ,
    ribbon= combined_result[:MonteCarlo].σΔz,
    label= "Monte Carlo",
    color=:red )
plot!(fig3,
    combined_result[:Current].Ic, combined_result[:SplineInter].Δz ,
    ribbon= combined_result[:SplineInter].σΔz,
    label="Spl. Interpolation",
    color=:blue)
plot!(fig3,
    combined_result[:Current].Ic, combined_result[:SplineFit].Δz ,
    ribbon= combined_result[:SplineFit].σΔz,
    label="Spl. Fit",
    color=:darkgreen )
plot!(fig3,
    legend=:topleft,
    foreground_color_legend=nothing,
    xlims=(0.80*threshold,1.05),
    ylims=(10e-3,4.05),
    xscale=:log10,yscale=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)

fig4 = plot(xlabel="Currents (A)", ylabel=L"z_{\mathrm{peak-peak}} \ (\mathrm{mm})")    
plot!(fig4,
    combined_result[:Current].Ic, combined_result[:MonteCarlo].Δz_alt ,
    ribbon= combined_result[:MonteCarlo].σΔz_alt,
    label= "Monte Carlo",
    color=:red )
plot!(fig4,
    combined_result[:Current].Ic, combined_result[:SplineInter].Δz_alt ,
    ribbon= combined_result[:SplineInter].σΔz_alt,
    label="Spl. Interpolation",
    color=:blue)
plot!(fig4,
    combined_result[:Current].Ic, combined_result[:SplineFit].Δz_alt ,
    ribbon= combined_result[:SplineFit].σΔz_alt,
    label="Spl. Fit",
    color=:darkgreen )
plot!(fig4,
    legend=:topleft,
    foreground_color_legend=nothing,
    xlims=(0.80*threshold,1.05),
    ylims=(10e-3,4.05),
    xscale=:log10,yscale=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)



fig = plot(fig1, fig2, fig3, fig4,
    layout=(2,2),
    link=:x,
    size=(800,600),
    left_margin=4mm,)
plot!(fig[1], xlabel="", xformatter=_->"", bottom_margin=-9mm)
plot!(fig[2], xlabel="", xformatter=_->"", bottom_margin=-9mm)
display(fig)




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FITTING
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Currents used for scan/plotting of fitted curves (log-spaced)
Iscan = logspace10(0.020, 1.00; n = 101);
QM_itp_zF1 = Spline1D(QM_df[!,:Ic],QM_df[!,:F1],k=3);
QM_itp_zF2 = Spline1D(QM_df[!,:Ic],QM_df[!,:F2],k=3);
QM_itp_Δz  = Spline1D(QM_df[!,:Ic],QM_df[!,:Δ],k=3);

combined_method = :SplineInter ; 

data_exp = DataFrame(
    Ic  = combined_result[:Current].Ic, 
    σIc = combined_result[:Current].σIc, 
    F1  = combined_result[combined_method].zF1, 
    σF1 = combined_result[combined_method].σzF1,
    F2  = combined_result[combined_method].zF2, 
    σF2 = combined_result[combined_method].σzF2,
    Δz  = combined_result[combined_method].Δz, 
    σΔz = combined_result[combined_method].σΔz,
    Δz_alt  = combined_result[combined_method].Δz_alt, 
    σΔz_alt = combined_result[combined_method].σΔz_alt

)

fig1 = plot(xlabel="Currents (A)", ylabel=L"$z^{F=1}_{p} \ (\mathrm{mm})$")
scatter!(fig1, data_exp.Ic, data_exp.F1; yerror=data_exp.σF1, 
    label=L"$F=1$ Experimental data (%$(length(data_exp.Ic)))",
    marker=(:circle, 3, :white),
    markerstrokecolor=:black)
for (i, (x, y)) in enumerate(zip(data_exp.Ic, data_exp.F1))
    annotate!(fig1, x, y-0.05, text(L"$\mathbf{%$i}$", :red, 8))
end
display(fig1)

fig2 = plot(xlabel="Currents (A)", ylabel=L"$z^{F=2}_{p} \ (\mathrm{mm})$")
scatter!(fig2,data_exp.Ic, data_exp.F2; yerror=data_exp.σF2, 
    label=L"$F=2$ Experimental data (%$(length(data_exp.Ic)))",
    marker=(:circle, 3, :white),
    markerstrokecolor=:black)
for (i, (x, y)) in enumerate(zip(data_exp.Ic, data_exp.F2))
    annotate!(fig2, x, y+0.05, text(L"$\mathbf{%$i}$", :red, 8))
end
display(fig2)

fig3 = plot(xlabel="Currents (A)", ylabel=L"$\Delta z_{p} \ (\mathrm{mm})$")
scatter!(fig3,data_exp.Ic, data_exp.Δz ; yerror=data_exp.σΔz, 
    label=L"$\Delta z$ Experimental data (%$(length(data_exp.Ic)))",
    marker=(:circle, 3, :white),
    markerstrokecolor=:black)
for (i, (x, y)) in enumerate(zip(data_exp.Ic, data_exp.Δz))
    annotate!(fig3, x, y+0.05, text(L"$\mathbf{%$i}$", :red, 8))
end
display(fig3)

fig4 = plot(xlabel="Currents (A)", ylabel=L"$\Delta z_{p} \ (\mathrm{mm})$")
scatter!(fig4,data_exp.Ic, data_exp.Δz_alt ; yerror=data_exp.σΔz_alt, 
    label=L"$\Delta z$ Experimental data (%$(length(data_exp.Ic)))",
    marker=(:circle, 3, :white),
    markerstrokecolor=:black)
for (i, (x, y)) in enumerate(zip(data_exp.Ic, data_exp.Δz))
    annotate!(fig4, x, y+0.05, text(L"$\mathbf{%$i}$", :red, 8))
end
display(fig4)

fig = plot(fig1, fig2, fig3, fig4,
    layout=(2,2),
    link=:x,
    size=(800,500),
    left_margin=3mm,
)
plot!(fig[1], xlabel="", xformatter=_->"", bottom_margin=-6mm)
plot!(fig[2], xlabel="", xformatter=_->"", bottom_margin=-6mm)
display(fig)



# fit_idx = vcat(28:30, 32:33)
# fit_idx = vcat(31:33,36)
fit_idx = vcat(27:30,33)    #** 25 26am 27 03 r2
fit_idx = vcat(23:26)       #** 26am 27 03 r2
fit_idx = vcat(23:27)       #** 26am 03 r2

# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# F1 FITTING
# ==================================================================================================

current_A      = data_exp.Ic
sigma_current  = data_exp.σIc
f1_exp_mm      = data_exp.F1
sigma_f1_exp   = data_exp.σF1

fitting_results = OrderedDict{Symbol, Any}()

fitting_results[:Current] = (
    Ic  = current_A,
    σIc = sigma_current,
)


# --------------------------------------------------------------------------------------------------
# QUANTUM MECHANICS vs EXPERIMENT
# --------------------------------------------------------------------------------------------------

qm_f1_mm   = QM_itp_zF1.(current_A)

qm_fit = FittingDataCQDQM.fit_QM_scale_model(
    current_A,
    f1_exp_mm,
    QM_itp_zF1;
    offset   = false,
    fitspace = :log10,
    σy       = sigma_f1_exp,
    idx      = fit_idx,
    project  = :model_to_y,
)

qm_alpha       = qm_fit.α
qm_sigma_alpha = qm_fit.σα
qm_beta        = qm_fit.β
qm_sigma_beta  = qm_fit.σβ

qm_f1_fit_mm = qm_alpha .* qm_f1_mm .+ qm_beta

qm_relerr_raw_pct = 100 .* (f1_exp_mm .- qm_f1_mm)     ./ qm_f1_mm
qm_relerr_fit_pct = 100 .* (f1_exp_mm .- qm_f1_fit_mm) ./ qm_f1_fit_mm

if qm_sigma_beta !== nothing
    @info @sprintf(
        "F1 QM fit: Experiment = (%.3f ± %.3f) * QM + (%.3f ± %.3f)",
        qm_alpha, qm_sigma_alpha, qm_beta, qm_sigma_beta
    )
else
    @info @sprintf(
        "F1 QM fit: Experiment = (%.3f ± %.3f) * QM",
        qm_alpha, qm_sigma_alpha
    )
end

pretty_table(
    hcat(current_A, f1_exp_mm, qm_f1_mm, qm_relerr_raw_pct, qm_f1_fit_mm, qm_relerr_fit_pct);
    column_labels = [
        ["Current", "F1 exp", "F1 QM", "Rel. Error", "F1 fitted QM", "Rel. Error"],
        ["[A]", "[mm]", "[mm]", "[%]", "[mm]", "[%]"],
    ],
    alignment                  = :c,
    row_label_column_alignment = :c,
    row_group_label_alignment  = :c,
    title                      = "QUANTUM MECHANICS",
    formatters = [
        fmt__printf("%8.3f", [1]),
        fmt__printf("%8.4f", [2, 3, 5]),
        fmt__printf("%8.1f", [4, 6]),
    ],
    style = TextTableStyle(
        first_line_column_label = crayon"yellow bold",
        table_border            = crayon"blue bold",
        title                   = crayon"bold red",
    ),
    table_format             = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths = true,
)

qm_fig = scatter(
    current_A,
    f1_exp_mm;
    yerror = sigma_f1_exp,
    label  = "Experimental data",
)
scatter!(
    qm_fig,
    current_A[qm_fit.used_mask],
    f1_exp_mm[qm_fit.used_mask];
    label  = "Used in fit",
    marker = :diamond,
)
plot!(qm_fig, current_A, qm_f1_mm;     lw = 2, label = "QM")
plot!(qm_fig, current_A, qm_f1_fit_mm; lw = 2, label = "$(@sprintf("%.3f", qm_alpha)) × QM + $(@sprintf("%.3f", qm_beta))")
plot!(qm_fig; xlabel = "Current (A)", ylabel = L"$z^{F=1}_{\mathrm{peak}}$ (mm)")

fitting_results[:QM_F1] = (
    F1     = qm_f1_fit_mm,
    mag    = (qm_alpha, qm_sigma_alpha),
    cutoff = (qm_beta, qm_sigma_beta),
)


# --------------------------------------------------------------------------------------------------
# CQD vs EXPERIMENT
# --------------------------------------------------------------------------------------------------

cqd_model = (I, k) -> ki_up_itp(I, k)

ki_bounds = (
    minimum(cqd_sim_data.ki[ki_start:ki_stop]),
    maximum(cqd_sim_data.ki[ki_start:ki_stop]),
)

cqd_k_fit_idx = findall(current_A .< 0.100)

cqd_fit = FittingDataCQDQM.fit_scale_and_k(
    current_A,
    f1_exp_mm,
    cqd_model;
    σy        = sigma_f1_exp,
    bounds    = ki_bounds,
    idx_k     = cqd_k_fit_idx,
    offset    = false,
    fit_alpha = true,
    idx_alpha = fit_idx,
    fitspace  = :log10,
)

cqd_k_best       = cqd_fit.k
cqd_alpha        = cqd_fit.α
cqd_sigma_alpha  = cqd_fit.σα
cqd_beta         = cqd_fit.β
cqd_sigma_beta   = cqd_fit.σβ

cqd_f1_fit_mm      = cqd_alpha .* cqd_model.(current_A, Ref(cqd_k_best)) .+ cqd_beta
cqd_relerr_fit_pct = 100 .* (f1_exp_mm .- cqd_f1_fit_mm) ./ cqd_f1_fit_mm

@info @sprintf(
    "F1 CQD fit: α = %.3f, β = %.3f, ki = %.6f",
    cqd_alpha, cqd_beta, cqd_k_best
)

pretty_table(
    hcat(current_A, f1_exp_mm, cqd_f1_fit_mm, cqd_relerr_fit_pct);
    column_labels = [
        ["Current", "F1 exp", "UP CQD", "Rel. Error"],
        ["[A]", "[mm]", "[mm]", "[%]"],
    ],
    alignment                  = :c,
    row_label_column_alignment = :c,
    row_group_label_alignment  = :c,
    title                      = "COQUANTUM DYNAMICS",
    formatters = [
        fmt__printf("%8.3f", [1]),
        fmt__printf("%8.4f", [2, 3]),
        fmt__printf("%8.1f", [4]),
    ],
    style = TextTableStyle(
        first_line_column_label = crayon"yellow bold",
        table_border            = crayon"blue bold",
        title                   = crayon"bold red",
    ),
    table_format             = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths = true,
)

cqd_fig_linear = scatter(
    current_A,
    f1_exp_mm;
    yerror            = sigma_f1_exp,
    marker            = (:circle, 2, :white),
    markerstrokecolor = :red,
    label             = "Experiment",
    xlabel            = L"$I$ (A)",
    ylabel            = L"$z_{\mathrm{max}}$ (mm)",
    legend            = :topleft,
)
plot!(
    cqd_fig_linear,
    current_A,
    cqd_f1_fit_mm;
    label = L"Fit: $\alpha f(I,k)$",
    lw    = 2,
    lc    = :blue,
)

cqd_fig_log = scatter(
    current_A,
    f1_exp_mm;
    yerror            = sigma_f1_exp,
    marker            = (:circle, 2, :white),
    markerstrokecolor = :red,
    xscale            = :log10,
    yscale            = :log10,
    label             = "Experiment",
    xlabel            = L"$I$ (A)",
    ylabel            = L"$z_{\mathrm{max}}$ (mm)",
)
plot!(
    cqd_fig_log,
    current_A,
    cqd_f1_fit_mm;
    lw    = 2,
    lc    = :blue,
    label = L"Fit: $\alpha=%$(round(cqd_alpha, digits=3))$, $k_{i}=%$(round(cqd_k_best; digits=2))\times 10^{-6}$",
)

plot(cqd_fig_linear, cqd_fig_log; layout = (2, 1), size = (800, 800))

fitting_results[:CQD_F1] = (
    F1     = cqd_f1_fit_mm,
    ki     = cqd_k_best,
    mag    = (cqd_alpha, cqd_sigma_alpha),
    cutoff = (cqd_beta, cqd_sigma_beta),
)


# --------------------------------------------------------------------------------------------------
# JOINT CQD + QM SCALING ANALYSIS
# --------------------------------------------------------------------------------------------------

joint_fit_mode = :full
n_front        = 8
n_back         = 8

joint_data = hcat(
    combined_result[:Current].Ic,
    combined_result[:Current].σIc,
    combined_result[combined_method].zF1,
    combined_result[combined_method].σzF1,
)

n_joint_rows = size(joint_data, 1)

low_range  = 1:n_front
high_range = (n_joint_rows - n_back + 1):n_joint_rows

@assert last(low_range) ≤ n_joint_rows
@assert first(high_range) ≥ 1

joint_fit_idx = begin
    if joint_fit_mode === :full
        Colon()
    elseif joint_fit_mode === :low
        low_range
    elseif joint_fit_mode === :high
        high_range
    elseif joint_fit_mode === :low_high
        vcat(low_range, high_range)
    else
        error("Unknown joint_fit_mode = $joint_fit_mode")
    end
end

joint_scale_result = FittingDataCQDQM.fit_ki_joint_scaling_fitsubset(
    joint_data,
    QM_itp_zF1,
    ki_up_itp,
    0.750,
    (cqd_sim_data[:ki][ki_start], cqd_sim_data[:ki][ki_stop]);
    fit_ki_mode = joint_fit_mode,
    n_front     = n_front,
    n_back      = n_back,
    w           = 0.50,
    ref_type    = :geom,
)

joint_data_scaled = copy(joint_data)
joint_data_scaled[:, 3] ./= joint_scale_result.scale_factor
joint_data_scaled[:, 4] ./= joint_scale_result.scale_factor

joint_data_fit_subset        = joint_data[joint_fit_idx, :]
joint_data_scaled_fit_subset = joint_data_scaled[joint_fit_idx, :]

joint_cqd_fit = FittingDataCQDQM.fit_ki_with_error(
    cqd_model,
    joint_data_scaled_fit_subset;
    bounds   = (cqd_sim_data[:ki][ki_start], cqd_sim_data[:ki][ki_stop]),
    conf     = 0.95,
    use_Zse  = false,
)

joint_fig = plot(title = L"Peak position ($F=1$)")
plot!(
    joint_fig,
    joint_data_scaled[:, 1],
    joint_data_scaled[:, 3];
    yerror            = joint_data_scaled[:, 4],
    label             = L"Experimental data (magnif.factor $m = %$(round(joint_scale_result.scale_factor, digits=4))$)",
    marker            = (:circle, 3, :white),
    markerstrokecolor = :darkgreen,
    line              = (:solid, 2, :darkgreen),
)
plot!(
    joint_fig,
    Iscan,
    QM_itp_zF1.(Iscan);
    label = "Quantum mechanical model",
    line  = (:solid, :red, 1.75),
)
plot!(
    joint_fig,
    Iscan,
    ki_up_itp.(Iscan, Ref(joint_cqd_fit.ki));
    label = L"CoQuantum dynamics: $k_{i}= \left( %$(round(joint_cqd_fit.ki; sigdigits=4)) \pm %$(round(joint_cqd_fit.ki_err, sigdigits=1)) \right) \times 10^{-6}$",
    line  = (:dot, :blue, 2),
    markerstrokewidth = 1,
)
plot!(
    joint_fig,
    Iscan,
    ki_up_itp.(Iscan, Ref(2.3));
    label = L"CoQuantum dynamics: $k_{i}= \left( %$(round(2.3; sigdigits=4)) \pm %$(round(joint_cqd_fit.ki_err, sigdigits=1)) \right) \times 10^{-6}$",
    line  = (:dot, :orangered, 2),
    markerstrokewidth = 1,
)
plot!(
    joint_fig;
    xlabel         = "Coil Current (A)",
    ylabel         = L"$z_{\mathrm{max}}$ (mm)",
    xaxis          = :log10,
    yaxis          = :log10,
    labelfontsize  = 14,
    tickfontsize   = 12,
    xticks         = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks         = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    size           = (900, 800),
    legendfontsize = 12,
    left_margin    = 3mm,
)
display(joint_fig)

joint_relerr_fit_pct = 100 .* (
    f1_exp_mm ./ joint_scale_result.scale_factor .- ki_up_itp.(current_A, Ref(joint_cqd_fit.ki))
) ./ ki_up_itp.(current_A, Ref(joint_cqd_fit.ki))

pretty_table(
    hcat(
        current_A,
        f1_exp_mm ./ joint_scale_result.scale_factor,
        ki_up_itp.(current_A, Ref(joint_cqd_fit.ki)),
        joint_relerr_fit_pct,
    );
    column_labels = [
        ["Current", "F1 exp", "UP CQD", "Rel. Error"],
        ["[A]", "[mm]", "[mm]", "[%]"],
    ],
    alignment                  = :c,
    row_label_column_alignment = :c,
    row_group_label_alignment  = :c,
    title                      = "COQUANTUM DYNAMICS",
    formatters = [
        fmt__printf("%8.3f", [1]),
        fmt__printf("%8.4f", [2, 3]),
        fmt__printf("%8.1f", [4]),
    ],
    style = TextTableStyle(
        first_line_column_label = crayon"yellow bold",
        table_border            = crayon"blue bold",
        title                   = crayon"bold red",
    ),
    table_format             = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths = true,
)


fitting_results[:CQD_QM_F1] = (
    CQD     = ki_up_itp.(current_A, Ref(joint_cqd_fit.ki)),
    ki     = (joint_cqd_fit.ki, joint_cqd_fit.ki_err),
    mag    = joint_scale_result.scale_factor,
)

# --------------------------------------------------------------------------------------------------
# GOODNESS OF FIT
# --------------------------------------------------------------------------------------------------

gof_current_A    = joint_data[:, 1];
gof_f1_exp_mm    = joint_data[:, 3];
gof_sigma_f1_exp = joint_data[:, 4];

gof_cqd = FittingDataCQDQM.goodness_of_fit(
    gof_current_A,
    gof_f1_exp_mm,
    cqd_f1_fit_mm;
    σ = gof_sigma_f1_exp,
    k = 2,
);

gof_qm = FittingDataCQDQM.goodness_of_fit(
    gof_current_A,
    gof_f1_exp_mm,
    qm_f1_fit_mm;
    σ = gof_sigma_f1_exp,
    k = 1,
);

metric_names = collect(String.(propertynames(gof_cqd)));

gof_table = hcat(
    [getproperty(gof_cqd, Symbol(name)) for name in metric_names],
    [getproperty(gof_qm,  Symbol(name)) for name in metric_names],
);

lower_is_better  = Set(["logMSE", "logRMSE", "chi2_log", "chi2_red", "AIC", "BIC", "NMAD"]);
higher_is_better = Set(["R2_log", "p_chi2"]);

best_metric_highlighter = TextHighlighter(
    (tbl, i, j) -> begin
        if !(j == 1 || j == 2)
            return false
        end

        metric_name = metric_names[i]
        cqd_value   = tbl[i, 1]
        qm_value    = tbl[i, 2]

        if !(isa(cqd_value, Number) && isa(qm_value, Number))
            return false
        end

        if !isfinite(cqd_value) || !isfinite(qm_value)
            return false
        end

        if metric_name in lower_is_better
            best_value = min(cqd_value, qm_value)
            return isapprox(tbl[i, j], best_value; atol = 0, rtol = sqrt(eps(Float64)))
        elseif metric_name in higher_is_better
            best_value = max(cqd_value, qm_value)
            return isapprox(tbl[i, j], best_value; atol = 0, rtol = sqrt(eps(Float64)))
        end

        return false
    end,
    crayon"fg:black bg:#fff7a1 bold",
);

pretty_table(
    gof_table;
    column_labels               = ["CQD", "QM"],
    row_labels                  = metric_names,
    row_label_column_alignment  = :l,
    highlighters                = [best_metric_highlighter],
    alignment                   = [:c, :c],
    style = TextTableStyle(
        first_line_column_label = crayon"yellow bold",
        table_border            = crayon"blue bold",
        column_label            = crayon"yellow bold",
    ),
    table_format             = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths = true,
)


# ==================================================================================================
# ==================================================================================================
# ==================================================================================================
# ΔZ FITTING
# ==================================================================================================

current_A      = data_exp.Ic
sigma_current  = data_exp.σIc
Δz_exp_mm      = data_exp.Δz
sigma_Δz_exp   = data_exp.σΔz

# --------------------------------------------------------------------------------------------------
# QUANTUM MECHANICS vs EXPERIMENT
# --------------------------------------------------------------------------------------------------

qm_Δz_mm   = QM_itp_Δz.(current_A)

qm_fit = FittingDataCQDQM.fit_QM_scale_model(
    current_A,
    Δz_exp_mm,
    QM_itp_Δz;
    offset   = false,
    fitspace = :log10,
    σy       = sigma_Δz_exp,
    idx      = fit_idx,
    project  = :model_to_y,
)

qm_alpha       = qm_fit.α
qm_sigma_alpha = qm_fit.σα
qm_beta        = qm_fit.β
qm_sigma_beta  = qm_fit.σβ

qm_Δz_fit_mm = qm_alpha .* qm_Δz_mm .+ qm_beta

qm_relerr_raw_pct = 100 .* (Δz_exp_mm .- qm_Δz_mm)     ./ qm_Δz_mm
qm_relerr_fit_pct = 100 .* (Δz_exp_mm .- qm_Δz_fit_mm) ./ qm_Δz_fit_mm

if qm_sigma_beta !== nothing
    @info @sprintf(
        "Δz QM fit: Experiment = (%.3f ± %.3f) * QM + (%.3f ± %.3f)",
        qm_alpha, qm_sigma_alpha, qm_beta, qm_sigma_beta
    )
else
    @info @sprintf(
        "Δz QM fit: Experiment = (%.3f ± %.3f) * QM",
        qm_alpha, qm_sigma_alpha
    )
end

pretty_table(
    hcat(current_A, Δz_exp_mm, qm_Δz_mm, qm_relerr_raw_pct, qm_Δz_fit_mm, qm_relerr_fit_pct);
    column_labels = [
        ["Current", "Δz exp", "Δz QM", "Rel. Error", "Δz fitted QM", "Rel. Error"],
        ["[A]", "[mm]", "[mm]", "[%]", "[mm]", "[%]"],
    ],
    alignment                  = :c,
    row_label_column_alignment = :c,
    row_group_label_alignment  = :c,
    title                      = "QUANTUM MECHANICS",
    formatters = [
        fmt__printf("%8.3f", [1]),
        fmt__printf("%8.4f", [2, 3, 5]),
        fmt__printf("%8.1f", [4, 6]),
    ],
    style = TextTableStyle(
        first_line_column_label = crayon"yellow bold",
        table_border            = crayon"blue bold",
        title                   = crayon"bold red",
    ),
    table_format             = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths = true,
)

qm_fig = scatter(
    current_A,
    Δz_exp_mm;
    yerror = sigma_Δz_exp,
    label  = "Experimental data",
)
scatter!(
    qm_fig,
    current_A[qm_fit.used_mask],
    Δz_exp_mm[qm_fit.used_mask];
    label  = "Used in fit",
    marker = :diamond,
)
plot!(qm_fig, current_A, qm_Δz_mm;     lw = 2, label = "QM")
plot!(qm_fig, current_A, qm_Δz_fit_mm; lw = 2, label = "$(@sprintf("%.3f", qm_alpha)) × QM + $(@sprintf("%.3f", qm_beta))")
plot!(qm_fig; xlabel = "Current (A)", ylabel = L"$\Delta z_{\mathrm{peak}}$ (mm)")

fitting_results[:QM_Δz] = (
    Δz     = qm_Δz_fit_mm,
    mag    = (qm_alpha, qm_sigma_alpha),
    cutoff = (qm_beta, qm_sigma_beta),
)


# --------------------------------------------------------------------------------------------------
# CQD vs EXPERIMENT
# --------------------------------------------------------------------------------------------------

cqd_model = (I, k) -> ki_Δ_itp(I, k)

ki_bounds = (
    minimum(cqd_sim_data.ki[ki_start:ki_stop]),
    maximum(cqd_sim_data.ki[ki_start:ki_stop]),
)

cqd_k_fit_idx = findall(current_A .< 0.100)

cqd_fit = FittingDataCQDQM.fit_scale_and_k(
    current_A,
    Δz_exp_mm,
    cqd_model;
    σy        = sigma_Δz_exp,
    bounds    = ki_bounds,
    idx_k     = cqd_k_fit_idx,
    offset    = false,
    fit_alpha = true,
    idx_alpha = fit_idx,
    fitspace  = :log10,
)

cqd_k_best       = cqd_fit.k
cqd_alpha        = cqd_fit.α
cqd_sigma_alpha  = cqd_fit.σα
cqd_beta         = cqd_fit.β
cqd_sigma_beta   = cqd_fit.σβ

cqd_Δz_fit_mm      = cqd_alpha .* cqd_model.(current_A, Ref(cqd_k_best)) .+ cqd_beta ;
cqd_relerr_fit_pct = 100 .* (Δz_exp_mm .- cqd_Δz_fit_mm) ./ cqd_Δz_fit_mm ;

@info @sprintf(
    "Δz CQD fit: α = %.3f, β = %.3f, ki = %.6f",
    cqd_alpha, cqd_beta, cqd_k_best
);

pretty_table(
    hcat(current_A, Δz_exp_mm, cqd_Δz_fit_mm, cqd_relerr_fit_pct);
    column_labels = [
        ["Current", "Δz exp", "Δz CQD", "Rel. Error"],
        ["[A]", "[mm]", "[mm]", "[%]"],
    ],
    alignment                  = :c,
    row_label_column_alignment = :c,
    row_group_label_alignment  = :c,
    title                      = "COQUANTUM DYNAMICS",
    formatters = [
        fmt__printf("%8.3f", [1]),
        fmt__printf("%8.4f", [2, 3]),
        fmt__printf("%8.1f", [4]),
    ],
    style = TextTableStyle(
        first_line_column_label = crayon"yellow bold",
        table_border            = crayon"blue bold",
        title                   = crayon"bold red",
    ),
    table_format             = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths = true,
)

cqd_fig_linear = scatter(
    current_A,
    Δz_exp_mm;
    yerror            = sigma_Δz_exp,
    marker            = (:circle, 2, :white),
    markerstrokecolor = :red,
    label             = "Experiment",
    xlabel            = L"$I$ (A)",
    ylabel            = L"$\Delta z_{\mathrm{max}}$ (mm)",
    legend            = :topleft,
);
plot!(
    cqd_fig_linear,
    current_A,
    cqd_Δz_fit_mm;
    label = L"Fit: $\alpha f(I,k)$",
    lw    = 2,
    lc    = :blue,
)

cqd_fig_log = scatter(
    current_A,
    Δz_exp_mm;
    yerror            = sigma_Δz_exp,
    marker            = (:circle, 2, :white),
    markerstrokecolor = :red,
    xscale            = :log10,
    yscale            = :log10,
    label             = "Experiment",
    xlabel            = L"$I$ (A)",
    ylabel            = L"$z_{\mathrm{max}}$ (mm)",
);
plot!(
    cqd_fig_log,
    current_A,
    cqd_Δz_fit_mm;
    lw    = 2,
    lc    = :blue,
    label = L"Fit: $\alpha=%$(round(cqd_alpha, digits=3))$, $k_{i}=%$(round(cqd_k_best; digits=2))\times 10^{-6}$",
)

plot(cqd_fig_linear, cqd_fig_log; layout = (2, 1), size = (800, 800))

fitting_results[:CQD_Δz] = (
    Δz     = cqd_Δz_fit_mm,
    ki     = cqd_k_best,
    mag    = (cqd_alpha, cqd_sigma_alpha),
    cutoff = (cqd_beta, cqd_sigma_beta),
)


# --------------------------------------------------------------------------------------------------
# JOINT CQD + QM SCALING ANALYSIS
# --------------------------------------------------------------------------------------------------

joint_fit_mode = :full
n_front        = 8
n_back         = 8

joint_data = hcat(
    combined_result[:Current].Ic,
    combined_result[:Current].σIc,
    combined_result[combined_method].Δz,
    combined_result[combined_method].σΔz,
);

n_joint_rows = size(joint_data, 1);

low_range  = 1:n_front;
high_range = (n_joint_rows - n_back + 1):n_joint_rows;

@assert last(low_range) ≤ n_joint_rows;
@assert first(high_range) ≥ 1;

joint_fit_idx = begin
    if joint_fit_mode === :full
        Colon()
    elseif joint_fit_mode === :low
        low_range
    elseif joint_fit_mode === :high
        high_range
    elseif joint_fit_mode === :low_high
        vcat(low_range, high_range)
    else
        error("Unknown joint_fit_mode = $joint_fit_mode")
    end
end

joint_scale_result = FittingDataCQDQM.fit_ki_joint_scaling_fitsubset(
    joint_data,
    QM_itp_Δz,
    ki_Δ_itp,
    0.750,
    (cqd_sim_data[:ki][ki_start], cqd_sim_data[:ki][ki_stop]);
    fit_ki_mode = joint_fit_mode,
    n_front     = n_front,
    n_back      = n_back,
    w           = 0.50,
    ref_type    = :geom,
)

joint_data_scaled = copy(joint_data);
joint_data_scaled[:, 3] ./= joint_scale_result.scale_factor;
joint_data_scaled[:, 4] ./= joint_scale_result.scale_factor;

joint_data_fit_subset        = joint_data[joint_fit_idx, :];
joint_data_scaled_fit_subset = joint_data_scaled[joint_fit_idx, :];

joint_cqd_fit = FittingDataCQDQM.fit_ki_with_error(
    cqd_model,
    joint_data_scaled_fit_subset;
    bounds   = (cqd_sim_data[:ki][ki_start], cqd_sim_data[:ki][ki_stop]),
    conf     = 0.95,
    use_Zse  = false,
)

joint_fig = plot(title = L"Peak position ($F=1$)");
plot!(
    joint_fig,
    joint_data_scaled[:, 1],
    joint_data_scaled[:, 3];
    yerror            = joint_data_scaled[:, 4],
    label             = L"Experimental data (magnif.factor $m = %$(round(joint_scale_result.scale_factor, digits=4))$)",
    marker            = (:circle, 3, :white),
    markerstrokecolor = :darkgreen,
    line              = (:solid, 2, :darkgreen),
);
plot!(
    joint_fig,
    Iscan,
    QM_itp_Δz.(Iscan);
    label = "Quantum mechanical model",
    line  = (:solid, :red, 1.75),
);
plot!(
    joint_fig,
    Iscan,
    ki_Δ_itp.(Iscan, Ref(joint_cqd_fit.ki));
    label = L"CoQuantum dynamics: $k_{i}= \left( %$(round(joint_cqd_fit.ki; sigdigits=4)) \pm %$(round(joint_cqd_fit.ki_err, sigdigits=1)) \right) \times 10^{-6}$",
    line  = (:dot, :blue, 2),
    markerstrokewidth = 1,
);
plot!(
    joint_fig,
    Iscan,
    ki_Δ_itp.(Iscan, Ref(2.3));
    label = L"CoQuantum dynamics: $k_{i}= \left( %$(round(2.3; sigdigits=4)) \pm %$(round(joint_cqd_fit.ki_err, sigdigits=1)) \right) \times 10^{-6}$",
    line  = (:dot, :orangered, 2),
    markerstrokewidth = 1,
);
plot!(
    joint_fig;
    xlabel         = "Coil Current (A)",
    ylabel         = L"$\Delta z_{\mathrm{max}}$ (mm)",
    xaxis          = :log10,
    yaxis          = :log10,
    labelfontsize  = 14,
    tickfontsize   = 12,
    xticks         = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks         = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    size           = (900, 800),
    legendfontsize = 12,
    left_margin    = 3mm,
);
display(joint_fig)

joint_relerr_fit_pct = 100 .* (
    Δz_exp_mm ./ joint_scale_result.scale_factor .- ki_Δ_itp.(current_A, Ref(joint_cqd_fit.ki))
) ./ ki_Δ_itp.(current_A, Ref(joint_cqd_fit.ki)) ;

pretty_table(
    hcat(
        current_A,
        Δz_exp_mm ./ joint_scale_result.scale_factor,
        ki_Δ_itp.(current_A, Ref(joint_cqd_fit.ki)),
        joint_relerr_fit_pct,
    );
    column_labels = [
        ["Current", "Δz exp", "Δz CQD", "Rel. Error"],
        ["[A]", "[mm]", "[mm]", "[%]"],
    ],
    alignment                  = :c,
    row_label_column_alignment = :c,
    row_group_label_alignment  = :c,
    title                      = "COQUANTUM DYNAMICS",
    formatters = [
        fmt__printf("%8.3f", [1]),
        fmt__printf("%8.4f", [2, 3]),
        fmt__printf("%8.1f", [4]),
    ],
    style = TextTableStyle(
        first_line_column_label = crayon"yellow bold",
        table_border            = crayon"blue bold",
        title                   = crayon"bold red",
    ),
    table_format             = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths = true,
)


fitting_results[:CQD_QM_Δz] = (
    CQD     = ki_Δ_itp.(current_A, Ref(joint_cqd_fit.ki)),
    ki     = (joint_cqd_fit.ki, joint_cqd_fit.ki_err),
    mag    = joint_scale_result.scale_factor,
);

# --------------------------------------------------------------------------------------------------
# GOODNESS OF FIT
# --------------------------------------------------------------------------------------------------

gof_current_A    = joint_data[:, 1];
gof_Δz_exp_mm    = joint_data[:, 3];
gof_sigma_Δz_exp = joint_data[:, 4];

gof_cqd = FittingDataCQDQM.goodness_of_fit(
    gof_current_A,
    gof_Δz_exp_mm,
    cqd_Δz_fit_mm;
    σ = gof_sigma_Δz_exp,
    k = 2,
);

gof_qm = FittingDataCQDQM.goodness_of_fit(
    gof_current_A,
    gof_Δz_exp_mm,
    qm_Δz_fit_mm;
    σ = gof_sigma_Δz_exp,
    k = 1,
);

metric_names = collect(String.(propertynames(gof_cqd)));

gof_table = hcat(
    [getproperty(gof_cqd, Symbol(name)) for name in metric_names],
    [getproperty(gof_qm,  Symbol(name)) for name in metric_names],
);

lower_is_better  = Set(["logMSE", "logRMSE", "chi2_log", "chi2_red", "AIC", "BIC", "NMAD"]);
higher_is_better = Set(["R2_log", "p_chi2"]);

best_metric_highlighter = TextHighlighter(
    (tbl, i, j) -> begin
        if !(j == 1 || j == 2)
            return false
        end

        metric_name = metric_names[i]
        cqd_value   = tbl[i, 1]
        qm_value    = tbl[i, 2]

        if !(isa(cqd_value, Number) && isa(qm_value, Number))
            return false
        end

        if !isfinite(cqd_value) || !isfinite(qm_value)
            return false
        end

        if metric_name in lower_is_better
            best_value = min(cqd_value, qm_value)
            return isapprox(tbl[i, j], best_value; atol = 0, rtol = sqrt(eps(Float64)))
        elseif metric_name in higher_is_better
            best_value = max(cqd_value, qm_value)
            return isapprox(tbl[i, j], best_value; atol = 0, rtol = sqrt(eps(Float64)))
        end

        return false
    end,
    crayon"fg:black bg:#fff7a1 bold",
);

pretty_table(
    gof_table;
    column_labels               = ["CQD", "QM"],
    row_labels                  = metric_names,
    row_label_column_alignment  = :l,
    highlighters                = [best_metric_highlighter],
    alignment                   = [:c, :c],
    style = TextTableStyle(
        first_line_column_label = crayon"yellow bold",
        table_border            = crayon"blue bold",
        column_label            = crayon"yellow bold",
    ),
    table_format             = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths = true,
)


















































#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

















#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Fitting for induction factor for each individual data set
fit_ki_mode = :full ; # ← change to :low, :high, or :low_high
n_front  = 6 ;
n_back   = 6 ;
for dir in data_directories

    dir_chosen = EXP_data_processed[dir]
    data_threshold = dir_chosen[(0.025 .< dir_chosen.Ic) .& (dir_chosen.Ic .< 0.850), :]
    data = hcat(data_threshold[!,:Ic], 0.02 * data_threshold[!,:Ic] , data_threshold[!,:F1], data_threshold[!,:ErrF1] )

    low_range  = 1:n_front ;
    high_range = (size(data, 1) - n_back + 1):size(data, 1);

    @assert last(low_range) ≤ size(data,1)
    @assert first(high_range) ≥ 1

    # Select rows according to the chosen fitting mode
    fit_ki_idx = begin
        if fit_ki_mode === :full
            Colon()
        elseif fit_ki_mode === :low
            low_range
        elseif fit_ki_mode === :high
            high_range
        elseif fit_ki_mode === :low_high
            vcat(low_range, high_range)
        else
            error("Unknown fit_ki_mode = $fit_ki_mode")
        end
    end

    # -----------------------------------------------------------------------------
    # Compute a global scaling factor for the experimental z-values 
    # with respect to QM
    #
    # Motivation:
    #   Experimental z may differ from simulated z by an overall scale factor
    #   (e.g., magnification calibration). We estimate a single multiplicative
    #   factor using only the highest-current tail, where SNR is typically best.
    #
    # Scaling convention used:
    #   scaled_mag = (yexp⋅yexp) / (yexp⋅ythe)
    # so that (yexp / scaled_mag) best matches ythe in a least-squares sense.
    # -----------------------------------------------------------------------------
    n_tail = 6  # number of tail points used for scaling

    @printf "For the scaling of the experimental data, we use the current range = %.3f A – %.3f A \n" first(last(data[:, 1], n_tail)) last(last(data[:, 1], n_tail))
    yexp = last(data[:, 3], n_tail)              # experimental z-values (tail)
    ythe = last(zqm.(data[:, 1]), n_tail)        # QM reference z-values at same currents
    scaled_mag = dot(yexp, yexp) / dot(yexp, ythe)

    # Apply scaling to both z and δz to preserve relative uncertainties
    data_scaled = copy(data);
    data_scaled[:, 3] ./= scaled_mag;
    data_scaled[:, 4] ./= scaled_mag;

    # @printf "The re-scaling factor of the experimental data with respect to Quantum Mechanics is %.3f" scaled_mag
    @printf "The re-scaling factor for the high current regime is %.3f" scaled_mag
    println("")

    # -----------------------------------------------------------------------------
    # 4) Build fitting subsets and fit kᵢ using CQD interpolant surface
    #
    # Note:
    #   fit_ki minimizes the log-space residual internally, but reports a 
    #   linear-space RMSE as `ki_err`.
    # -----------------------------------------------------------------------------
    data_fitting        = data[fit_ki_idx, :];
    data_scaled_fitting = data_scaled[fit_ki_idx, :];
    fit_original = fit_ki(data, data_fitting, cqd_sim_data.ki, (ki_start,ki_stop))
    fit_scaled   = fit_ki(data_scaled, data_scaled_fitting, cqd_sim_data.ki, (ki_start,ki_stop))

    @info "Induction term ki = $(round(fit_scaled.ki; sigdigits=4))"

    z_qm = zqm.(I_scan);
    m_qm = log_mask(I_scan, z_qm);
    fig = plot(
        I_scan[m_qm], z_qm[m_qm];
        label = "Quantum mechanics",
        line  = (:solid, :red, 1.75),
    )
    plot!(
        fig,
        data_scaled[:,1], data_scaled[:,3];
        yerror= data_scaled[:,4],
        color = :gray35,
        marker = (:circle, :gray35, 4),
        markerstrokecolor = :gray35,
        markerstrokewidth = 1,
        label = dir,
    )
    z_fit_orig = ki_up_itp.(I_scan, Ref(fit_scaled.ki));
    m_orig = log_mask(I_scan, z_fit_orig);
    plot!(
        fig,
        I_scan[m_orig], z_fit_orig[m_orig];
        label = L"CQD : $k_{i}= \left( %$(round(fit_original.ki, sigdigits=3)) \pm %$(round(fit_scaled.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
        line  = (:solid, :blue, 2),
        marker = (:xcross, :blue, 0.2),
        markerstrokewidth = 1,
    )
    plot!(
        fig;
        # title = dir,
        xlabel = "Current (A)",
        ylabel = L"$z_{\mathrm{max}}$ (mm)",
        xaxis  = :log10,
        yaxis  = :log10,
        labelfontsize = 14,
        tickfontsize  = 12,
        xticks = ([1e-3, 1e-2, 1e-1, 1.0],
                [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-3, 1e-2, 1e-1, 1.0],
                [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xlims = (0.010, 1.05),
        size  = (900, 800),
        # legendtitle = L"$n_{z} = %$(nz_fixed)$ | $\sigma_{\mathrm{conv}}=%$(1e3*σw_fixed)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_fixed)$",
        legendfontsize = 12,
        left_margin = 3mm,
    )
    display(fig)

end

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


fit_ki_mode = :full   # ← change to :low, :high, or :low_high
n_front  = 8
n_back   = 6

data = hcat(combined_result[:Current].Ic, combined_result[:Current].σIc , combined_result[:MonteCarlo].zF1 , combined_result[:MonteCarlo].σzF1 )
low_range  = 1:n_front ;
high_range = (size(data, 1) - n_back + 1):size(data, 1);

@assert last(low_range) ≤ size(data,1)
@assert first(high_range) ≥ 1

# Select rows according to the chosen fitting mode
fit_ki_idx = begin
    if fit_ki_mode === :full
        Colon()
    elseif fit_ki_mode === :low
        low_range
    elseif fit_ki_mode === :high
        high_range
    elseif fit_ki_mode === :low_high
        vcat(low_range, high_range)
    else
        error("Unknown fit_ki_mode = $fit_ki_mode")
    end
end

n_tail = 8  # number of tail points used for scaling
@printf "For the scaling of the experimental data, we use the current range = %.3f A – %.3f A \n" first(last(data[:, 1], n_tail)) last(last(data[:, 1], n_tail))
yexp = last(data[:, 3], n_tail)              # experimental z-values (tail)
ythe = last(zqm.(data[:, 1]), n_tail)        # QM reference z-values at same currents
scaled_mag = dot(yexp, yexp) / dot(yexp, ythe)

# Apply scaling to both z and δz to preserve relative uncertainties
data_scaled = copy(data);
data_scaled[:, 3] ./= 1 #scaled_mag;
data_scaled[:, 4] ./= 1 #scaled_mag;

# @printf "The re-scaling factor of the experimental data with respect to Quantum Mechanics is %.3f" scaled_mag
@printf "The re-scaling factor for the high current regime is %.3f" scaled_mag
println("")

# -----------------------------------------------------------------------------
# 4) Build fitting subsets and fit kᵢ using CQD interpolant surface
#
# Note:
#   fit_ki minimizes the log-space residual internally, but reports a 
#   linear-space RMSE as `ki_err`.
# -----------------------------------------------------------------------------
data_fitting        = data[fit_ki_idx, :];
data_scaled_fitting = data_scaled[fit_ki_idx, :];
fit_original = fit_ki(data, data_fitting, cqd_sim_data.ki, (ki_start,ki_stop))
fit_scaled   = fit_ki(data_scaled, data_scaled_fitting, cqd_sim_data.ki, (ki_start,ki_stop))

@info "Induction term ki = $(round(fit_scaled.ki; sigdigits=4)) × 10^-6"

z_qm = zqm.(I_scan);
m_qm = log_mask(I_scan, z_qm);
fig = plot(
    I_scan[m_qm], scaled_mag .* z_qm[m_qm];
    label = "Quantum mechanics",
    line  = (:solid, :red, 1.75),
)
plot!(
    fig,
    data_scaled[:,1], data_scaled[:,3];
    yerror= data_scaled[:,4],
    label="Combined experiment",
    color = :gray35,
    marker = (:circle, :gray35, 4),
    markerstrokecolor = :gray35,
    markerstrokewidth = 1,
)
z_fit_orig = ki_up_itp.(I_scan, Ref(fit_scaled.ki));
m_orig = log_mask(I_scan, z_fit_orig);
plot!(
    fig,
    I_scan[m_orig], z_fit_orig[m_orig];
    label = L"CQD : $k_{i}= \left( %$(round(fit_original.ki, sigdigits=3)) \pm %$(round(fit_scaled.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
    line  = (:solid, :blue, 2),
    marker = (:xcross, :blue, 0.2),
    markerstrokewidth = 1,
)
plot!(
    fig;
    # title = dir,
    xlabel = "Current (A)",
    ylabel = L"$z_{\mathrm{max}}$ (mm)",
    xaxis  = :log10,
    yaxis  = :log10,
    labelfontsize = 14,
    tickfontsize  = 12,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0],
              [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0],
              [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims = (0.010, 1.05),
    size  = (900, 800),
    # legendtitle = L"$n_{z} = %$(nz_fixed)$ | $\sigma_{\mathrm{conv}}=%$(1e3*σw_fixed)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_fixed)$",
    legendfontsize = 12,
    left_margin = 3mm,
)
display(fig)

















































fit_ki_with_error(ki_up_itp, data_fitting; bounds=(cqd_sim_data.ki[ki_start], cqd_sim_data.ki[ki_stop]),)
fit_ki_with_error(ki_up_itp, data_scaled_fitting; bounds=(cqd_sim_data.ki[ki_start], cqd_sim_data.ki[ki_stop]),)



TheoreticalSimulation.BvsI(1.0)
TheoreticalSimulation.GvsI(1.0)

6400/130 * 4


TheoreticalSimulation.BvsI(0.020)


0.0200/0.0275











cqd_sim_data = JLD2_MyTools.list_keys_jld_cqd(data_cqdup_path);
n_ki = length(cqd_sim_data.ki);
@info "CQD simulation for $(n_ki) ki values"
colores_ki = palette(:darkrainbow, n_ki)
up_cqd = Matrix{Float64}(undef, nI, n_ki);
dw_cqd = Matrix{Float64}(undef, nI, n_ki);
Δz_cqd = Matrix{Float64}(undef, nI, n_ki);



















dir = data_directories[1]
kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS", "summary", dir, dir * "_report_summary.jld2")
EXP_data_dd = jldopen(kk_path, "r") do file
    ic = file["meta/Currents"]
    bz = file["meta/BzTesla"]

    dd = file[JLD2_MyTools.make_keypath_exp(dir, nz, λ0)]

    F1 = dd[:fw_F1_peak_pos_raw]
    F1b = dd[:fw_F1_peak_pos]
    F2 = dd[:fw_F2_peak_pos_raw]

    zc = 0.5 * (F1[1][1] + F2[1][1])
    δc = 0.5 * sqrt(F1[2][1]^2 + F2[2][1]^2)

    (
        Ic = ic,
        Bz = bz,
        F1 = F1,
        F1b = F1b,
        F2 = F2,
        C0 = dd[:centroid_fw_mm],
        C00 = [zc, δc]
    )
end
hcat(EXP_data_dd.F1[1] .- EXP_data_dd.C0[1], EXP_data_dd.F1b[1])








2+2













chosen_cqd_up =  jldopen(data_cqdup_path,"r") do file
    file[JLD2_MyTools.make_keypath_cqd(:up,nz, σw_mm, λ0)]
end

chosen_cqd_dw =  jldopen(data_cqddw_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end




!fig = plot(
    xlabel="Currents (mA)",
    ylabel=L"$F=1$ position (mm)",
    title="($nz,$λ0)",
    size=(950,600),
    left_margin=3mm,
    bottom_margin=2mm,
    legendfontsize=10,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
for (idx,dir) in enumerate(data_directories)
    kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS","summary", dir, dir * "_report_summary.jld2")
    data = jldopen(kk_path, "r") do file
        ic = file["meta/Currents"]
        bz = file["meta/BzTesla"]

        dd = file[JLD2_MyTools.make_keypath_exp(dir,nz,λ0)]
        return ( Ic=ic, Bz=bz, F1 = dd[:fw_F1_peak_pos_raw], F2 = dd[:fw_F2_peak_pos_raw], C0 = dd[:centroid_fw_mm] )
    end

    plot!(fig,
        1000*data.Ic, data.F1[1],
        yerror = data.F1[2],
        seriestype=:scatter,
        label="$(dir) (degauss = $(round(1e3*data.Ic[1]; digits=6))mA , $(Int(round(1e4*data.Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
    )
    plot!(fig,
        1000*data.Ic, data.F1[1],
        seriestype=:line,
        label= nothing,
        line=(:solid,0.3,colores_data[idx]),
        color = colores_data[idx],
        fillalpha=0.10,
    )

    # plot!(fig,
    #     1000*data.Ic, data.F2,
    #     label="",
    #     marker=(:circle, 2, :white),
    #     markerstrokecolor=colores[idx],
    #     line=(:dash,1,colores[idx]))

    # hline!(fig, [data.C0[1]], label="",line=(:dash,1,colores[idx]))

end
display(fig)
plot!(fig,
    xlims=(0,50),
    ylims=(8.70,9.00),
);
display(fig)
plot!(fig,
    xlims=(750,1010),
    ylims=(10.30,10.90),

);
display(fig)

## Recalculation of the F=1 peak position with respect to a different centroid

new_data = Vector{NamedTuple}(undef, n_data)

for (idx, dir) in enumerate(data_directories)

    kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS","summary", dir, dir * "_report_summary.jld2")

    new_data[idx] = jldopen(kk_path, "r") do file
        ic = file["meta/Currents"]
        bz = file["meta/BzTesla"]

        dd = file[JLD2_MyTools.make_keypath_exp(dir,nz,λ0)]

        (Ic=ic, 
         Bz=bz,
         F1=dd[:fw_F1_peak_pos_raw],
         F2=dd[:fw_F2_peak_pos_raw],
         C0=dd[:centroid_fw_mm])
    end
end

new_centroid = [mean([new_data[v].F1[1][1],new_data[v].F2[1][1]]) for v=1:n_data]

plot(xlabel="Current (A)")
for (idx,dir) in enumerate(data_directories)

    z_table = hcat(new_data[idx].Ic , new_data[idx].F1[1] .- new_centroid[idx] )
    zz = DataReading.subset_by_cols( z_table , [1,2] ; thr = 1e-3, include_equal=false)[3]

    plot!(zz[:,1], zz[:,2],
    marker=(:circle,2,:white),
    markerstrokecolor=colores[idx],
    line=(colores[idx]),
    label=dir)
end
plot!(xlims=(0.030,0.050),
ylims=(10e-3, 120e-3))
plot!(Ic_QM_sim,zm_QM_sim,
    label="Quantum mechanics (T=200C)",
    line=(:solid,2,:blue))
for ki in [1.0,1.2,1.4,1.6,1.8,2.0]

    chosen_cqd =  jldopen(data_cqd_path,"r") do file
        file[JLD2_MyTools.make_keypath_cqd(:up, ki, nz, σw_mm, λ0)]
    end
    Ic_CQD_sim = [chosen_cqd[i][:Icoil] for i in eachindex(chosen_cqd)][2:end]
    zm_CQD_sim = [chosen_cqd[i][:z_max_smooth_spline_mm] for i in eachindex(chosen_cqd)][2:end]

    plot!(Ic_CQD_sim,zm_CQD_sim,
    label="CQD (T=200C | ki = $(ki)×10^-6)",
    line=(:solid,2))
end
plot!(
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.001, 0.01, 0.1, 1, 10], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    xlims=(8e-4,1.1),
    ylims=(8e-4,2.1),
    legend=:outerright,
    xscale=:log10,
    yscale=:log10,
)






dir = data_directories[6]
kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS","summary", dir, dir * "_report_summary.jld2")
JLD2_MyTools.show_exp_summary(kk_path, dir)

data = jldopen(kk_path, "r") do file
    ic = file["meta/Currents"]
    bz = file["meta/BzTesla"]

    dd = file[JLD2_MyTools.make_keypath_exp(dir,2,0.01)]
    return ( Ic=ic, Bz=bz, F1 = dd[:fw_F1_peak_pos_raw], F2 = dd[:fw_F2_peak_pos_raw], C0 = dd[:centroid_fw_mm] )
end
data.C0

data_centroid_fw  = 0.5 * (data.F1[1] + data.F2[1])
data_δcentroid_fw = round.(0.5 * sqrt.(data.F1[2].^2 + data.F2[2].^2) ; sigdigits=1)
post_threshold_mean(data_centroid_fw, data.Ic, data_δcentroid_fw;
                                 threshold=0.010,
                                 half_life=5,
                                 eps=1e-6,
                                 weighted=true)


MyExperimentalAnalysis.post_threshold_mean(data_centroid_fw, data.Ic, data_δcentroid_fw;
                                 threshold=0.010,
                                 half_life=5,
                                 eps=1e-6,
                                 weighted=false)







