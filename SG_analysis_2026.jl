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


data_directories = ["20260220", "20260225", "20260226am","20260226pm","20260227", "20260303", "20260306r1", "20260306r2"]
no = length(data_directories);
colores = palette(:darkrainbow, no);

nz , λ0 = 2, 0.01;
σw_mm = 0.200;

############ phywe magnetic field measurement ##############################
BvsI_phywe = sort(CSV.read("SG_BvsI_phywe.csv",DataFrame; header=["Ic","Bz"]),1);
phywe_shift = hcat(BvsI_phywe.Ic .- 0.11021, BvsI_phywe.Bz .- 0.00445);
phywe_shift = DataReading.subset_by_cols(phywe_shift,[1,2]; thr=0.0, include_equal=false)[3]

exp_data = Vector{Matrix{Float64}}(undef, no)
for (idx, dir) in enumerate(data_directories)
    data = load(joinpath(@__DIR__,"EXPERIMENTS",dir,"data_processed.jld2"),"data")
    Ic = data[:Currents]
    Bz = data[:BzTesla]
    exp_data[idx] = hcat(Ic,Bz)
end


## magnetic field: log-log scale
fig1 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    size=(900,500),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    left_margin=5mm,
    bottom_margin=3mm,);
for (idx, dir) in enumerate(data_directories)
    data = exp_data[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig1, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
    )

end
plot!(fig1,Icoils[2:end], 1e3*TheoreticalSimulation.BvsI.(Icoils)[2:end],
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig1,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
    label="SG PHYWE calibration",
    line=(:solid,2,:dodgerblue4));
plot!(fig1,phywe_shift[:,1], 1000*phywe_shift[:,2],
    label="SG PHYWE calibration (adjusted)",
    line=(:dot,2,:dodgerblue4));
plot!(fig1,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.02),
    ylims=(1e-2,1e3),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10, 100], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    legendfontsize = 6,

);
display(fig1)

## magnetic field: linear scale
fig2 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    size=(900,500),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    left_margin=5mm,
    bottom_margin=3mm,
    legendfontsize = 8,);
for (idx, dir) in enumerate(data_directories)
    data = exp_data[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig2, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
    )

end
plot!(fig2,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig2,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
    label="SG PHYWE calibration",
    line=(:solid,2,:dodgerblue4));
plot!(fig2,phywe_shift[:,1], 1000*phywe_shift[:,2],
    label="SG PHYWE calibration (adjusted)",
    line=(:dot,2,:dodgerblue4));
display(fig2)

##magnetic field: low currents
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
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
    )

end
plot!(fig3,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig3,
    xlims=(0.0,15e-3),
    ylims=(-5,15),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig3)

##magnetic field: high currents
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
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
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
BvsI_comparison = Vector{Matrix{Float64}}(undef, no)
for (idx, dir) in enumerate(data_directories)
    vs = exp_data[idx]

    mask, inds, rows_view = DataReading.subset_by_cols(vs, [1,2]; thr=1e-3, include_equal = true)
    BvsI_comparison[idx] = Matrix(rows_view)  # store a copy; or keep the view (see B)
end
fig5 = plot(xlabel="Currents (A)",
    ylabel=L"$B_{\mathrm{exp}} / B_{\mathrm{manual}}$")
for (idx, dir) in enumerate(data_directories)
    B_ratio = BvsI_comparison[idx][:,2] ./ TheoreticalSimulation.BvsI.(BvsI_comparison[idx][:,1])
    
    plot!(fig5,  BvsI_comparison[idx][:,1] , B_ratio,
        label=data_directories[idx],
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:solid,1,colores[idx])
    )
    y0 , σ0 = mean(B_ratio), standard_error(B_ratio)

    x = range(1e-3, 1.1, length=200)
    plot!(fig5, x, fill(y0, length(x)),
     ribbon = σ0,
     color = colores[idx],
     fillalpha = 0.25,
     line=(:dash,0.5,colores[idx]),
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
)
display(fig5)

# SHIFTED EXPERIMENTAL DATA TO ZERO FIELD
exp_data_corr = Vector{Matrix{Float64}}(undef, no)
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
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
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
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
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
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
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
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
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


BvsI_comparison_corr = Vector{Matrix{Float64}}(undef, no)
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
        markerstrokecolor=colores[idx],
        line=(:solid,1,colores[idx])
    )
    y0 , σ0 = mean(B_ratio), standard_error(B_ratio)

    x = range(1e-3, 1.1, length=200)
    plot!(fig5A,
    x, fill(y0, length(x)),
    ribbon = σ0,
    color = colores[idx],
    fillalpha = 0.25,
    line=(:dash,0.5,colores[idx]),
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
#+++++++++++++++++++++++++++ Centroid +++++++++++++++++++++++++++++++++++++++++++++
centroid_fw = Matrix{Float64}(undef, no, 2)
pos_at_zero = Matrix{Float64}(undef, no, 4)
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

plot(data_directories, centroid_fw[:,1],
label="Computed centroid positions (high currents)",
 color=:blue,
 marker=(:circle,:white,4),
 markerstrokecolor=:blue,
 line=(:solid,2,:blue),
 yerror = centroid_fw[:,2],)
plot!(data_directories,pos_at_zero[:,1],
    ribbon=pos_at_zero[:,2],
    label=L"$F=1$ at ($I_{c}=0A$)",
    color=:orangered2,
    marker=(:circle,:white),
    markerstrokecolor=:orangered2,
    fillalpha=0.2)
plot!(data_directories,pos_at_zero[:,3],
    ribbon=pos_at_zero[:,4],
    label=L"$F=2$ at ($I_{c}=0A$)",
    color=:springgreen2,
    marker=(:circle,:white),
    markerstrokecolor=:springgreen2,
    fillalpha=0.2)
plot!(["20260225","20260226am","20260226pm","20260227","20260303"],[8.8369,8.8440,8.8549,8.8284,8.8292],
    label="Qihang's fitting 1",
    marker=(:square,2,:white),
    markerstrokecolor=:purple,
    line=(:purple))
plot!(["20260225","20260226am","20260226pm","20260227","20260303"], [8.811211551,8.790075956,8.796932977,8.84320487,8.8048355],
    label="Qihang's fitting 2",
    marker=(:square,2,:white),
    markerstrokecolor=:green,
    line=(:green))
plot!(
    # title = "Std.Dev=$(round(std(centroid_fw[:,1], corrected=false); digits=3))",
    legend=:bottomleft,
    ylims=(8.700,8.950),
    yformatter = y -> @sprintf("%.3f", y),
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    xminorticks=false,
    bottom_margin=5mm,
    ylabel="peak position (mm)",
    xrotation=75,)


########################################################################################
#+++++++++++++++++++++++++++ Peak Position +++++++++++++++++++++++++++++++++++++++++++++

#+++++++++ QUANTUM MECHANICS +++++++++++++
chosen_qm_f1 =  jldopen(data_qmf1_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end
chosen_qm_f2 =  jldopen(data_qmf2_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end
data_QM = hcat(
    [v[:Icoil]                  for v in values(chosen_qm_f1)],
    [v[:z_max_smooth_spline_mm] for v in values(chosen_qm_f1)],
    [v[:z_max_smooth_spline_mm] for v in values(chosen_qm_f2)],
)
data_QM = hcat(data_QM, data_QM[:,2] .- data_QM[:,3]);
QM_df = DataFrame(data_QM, [:Ic, :F1, :F2, :Δ])
pretty_table(QM_df; 
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


fig1 = plot(
    xlabel="Currents (A)",
    ylabel="Peak position (mm)",
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
    ylabel="Peak-to-Peak distance (mm)",
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
link=:x,)
plot!(fig[1], xlabel="", xformatter=_->"");
display(fig)



#+++++++++ COQUANTUM DYNAMICS +++++++++++++
cqd_dt = JLD2_MyTools.list_keys_jld_cqd(data_cqdup_path);
n_ki = length(cqd_dt.ki);
@info "CQD simulation for $(n_ki) ki values"
colores_ki = palette(:darkrainbow, n_ki)
up_cqd = Matrix{Float64}(undef, nI, n_ki);
dw_cqd = Matrix{Float64}(undef, nI, n_ki);
Δz_cqd = Matrix{Float64}(undef, nI, n_ki);

for (i,ki) in enumerate(cqd_dt.ki)
    dataup = jldopen(data_cqdup_path,"r") do f
        f[JLD2_MyTools.make_keypath_cqd(:up,ki,nz,σw_mm,λ0)]
    end
    up_cqd[:,i] = [dataup[j][:z_max_smooth_spline_mm] for j=1:nI]

    datadw = jldopen(data_cqddw_path,"r") do f
        f[JLD2_MyTools.make_keypath_cqd(:dw,ki,nz,σw_mm,λ0)]
    end
    dw_cqd[:,i] = [datadw[j][:z_max_smooth_spline_mm] for j=1:nI]

    Δz_cqd[:,i] = up_cqd[:,i] .- dw_cqd[:,i]
end

fig1 = plot(xlabel="Current (A)",
    ylabel="Peak position (mm)");
for (i,ki) in enumerate(cqd_dt.ki)
    ki_string = @sprintf("%2.3f",ki)
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
plot!(fig1,
    xlims=(1e-3,1.05),
    xscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    legendtitle=L"$k_{i} \times 10^{-6}$",
    legendfontsize=6,
    legend_columns=3,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
<<<<<<< HEAD
);
=======
    size=(800,600),
)
>>>>>>> 41a6e1bd97275ed15d64dadbb23cfd6e082beb39
display(fig1)


fig2 = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-Peak distance (mm)",
);
for (i,ki) in enumerate(cqd_dt.ki)
    ki_string = @sprintf("%2.3f",ki)
    plot!(fig2,
        Icoils, Δz_cqd[:,i],
        label=L"$%$(ki_string)$",
        line=(:solid,1,colores_ki[i]),

    )
end
plot!(fig2,
    xlims=(1e-3,1.05),
    ylims=(1e-4,5.0),
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    legendtitle=L"$k_{i} \times 10^{-6}$",
    legend_columns = 3,
    legendfontsize=5,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
<<<<<<< HEAD
    
);
=======
    size=(800,600),
)
>>>>>>> 41a6e1bd97275ed15d64dadbb23cfd6e082beb39
display(fig2)

fig = plot(fig1, fig2,
size=(1000,800),
layout=(2,1),
link=:x,
left_margin=5mm,
bottom_margin=2mm,
<<<<<<< HEAD
);
display(fig)
=======
)
plot!(fig[2],legend=false)
>>>>>>> 41a6e1bd97275ed15d64dadbb23cfd6e082beb39

#+++++++++ QUANTUM MECHANICS & COQUANTUM DYNAMICS +++++++++++++
fig1 = plot(
    xlabel="Currents (A)",
    ylabel="Peak position (mm)",
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
for (i,ki) in enumerate(cqd_dt.ki)
    ki_string = @sprintf("%2.3f",ki)
    plot!(fig1,
        Icoils, up_cqd[:,i],
        label=L"$k_{i}=%$(ki_string)\times 10^{-6}$",
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
<<<<<<< HEAD
);
=======
    size=(1000,600)
)
>>>>>>> 41a6e1bd97275ed15d64dadbb23cfd6e082beb39
display(fig1)


fig2 = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-Peak distance (mm)",
);
for (i,ki) in enumerate(cqd_dt.ki)
    ki_string = @sprintf("%2.2f",ki)
    plot!(fig2,
        Icoils, Δz_cqd[:,i],
        label=L"$k_{i}=%$(ki_string)\times 10^{-6}$",
        line=(:solid,1,colores_ki[i]),

    )
end
plot!(fig2,
    QM_df[!,:Ic],QM_df[!,:Δ],
    label=L"QM $\Delta z$",
    marker=(:circle,2,:white, 0.55),
    markerstrokecolor=:lime,
    line=(:lime,1),
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
<<<<<<< HEAD
);
=======
    size=(1000,600)
)
>>>>>>> 41a6e1bd97275ed15d64dadbb23cfd6e082beb39
display(fig2)


fig = plot(fig1, fig2,
size=(1200,1000),
layout=(2,1),
link=:x,
left_margin=5mm,
bottom_margin=2mm,
);
display(fig)


#+++++++++++++ EXPERIMENTS ++++++++++++++++++++
EXP_data = OrderedDict{String, NamedTuple}()
data_directories
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



EXP_data_processed = OrderedDict{String, DataFrame}()
for 𝓁 = 1:8
    dir = data_directories[𝓁]
    println(dir)


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
                formatters = [fmt__printf("%8.3f", [1]), fmt__printf("%8.3f",[2,4,6]), fmt__printf("%8.3f",[3,5,7])],
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
    ylabel=L"$F=1$ peak position (mm)");
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
    for (i,ki) in enumerate(cqd_dt.ki)
        ki_string = @sprintf("%2.2f",ki)
        plot!(figa,
            Icoils, up_cqd[:,i],
            label=L"$k_{i}=%$(ki_string)\times 10^{-6}$",
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

<<<<<<< HEAD
    fig = plot(figa,figb,
        layout=(2,1),
        size=(1000,600),
        top_margin=5mm)
    display(fig)
=======
fig = plot(figa,figb,
    layout=(2,1),
    size=(1200,1000),
    top_margin=5mm)
display(fig)
>>>>>>> 41a6e1bd97275ed15d64dadbb23cfd6e082beb39


    data_Δz = data_dir[data_dir.Δ .> 0, :];

    figc = plot(
        xlabel="Currents (A)",
        ylabel="Peak-to-Peak distance (mm)",
    );
    for (i,ki) in enumerate(cqd_dt.ki)
        ki_string = @sprintf("%2.2f",ki)
        plot!(figc,
            Icoils, Δz_cqd[:,i],
            label=L"$k_{i}=%$(ki_string)\times 10^{-6}$",
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

<<<<<<< HEAD
    fig = plot(figc,figd,
        layout=(2,1),
        size=(1000,600),
        top_margin=7mm)
    display(fig)
=======
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
>>>>>>> 41a6e1bd97275ed15d64dadbb23cfd6e082beb39

end


#++++++ DATA COMPARISON
figa = plot(
    xlabel="Currents (A)",
    ylabel="Peak position (mm)"
)
for (i,dir) in enumerate(data_directories)
    plot!(figa,
        EXP_data_processed[dir].Ic, EXP_data_processed[dir].F1,
        yerror = EXP_data_processed[dir].ErrF1,
        label= dir,
        marker=(:circle,2,:white,0.6),
        markerstrokecolor=colores[i],
        line=(:solid,1,colores[i])
    )
end
plot!(figa,
    QM_df[!,:Ic],QM_df[!,:F1],
    label=L"QM : $F=1$",
    seriestype=:scatter,
    marker=(:diamond,2,:white),
    markerstrokecolor=:red,
    line=(:red,1),
)
plot!(figa,
    legend=:bottomright,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    xlims=(10e-3, 1.05),
    ylims=(1e-3, 2.05))
figa1= deepcopy(figa)
plot!(figa,
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)

figb = plot(
    xlabel="Currents (A)",
    ylabel="Peak position (mm)"
)
for (i,dir) in enumerate(data_directories)
    plot!(figb,
        EXP_data_processed[dir].Ic, EXP_data_processed[dir].F2,
        yerror = EXP_data_processed[dir].ErrF2,
        label= dir,
        marker=(:circle,2,:white,0.6),
        markerstrokecolor=colores[i],
        line=(:solid,1,colores[i])
    )
end
plot!(figb,
    QM_df[!,:Ic],QM_df[!,:F2],
    label=L"QM : $F=2$",
    seriestype=:scatter,
    marker=(:diamond,2,:white),
    markerstrokecolor=:red,
    line=(:red,1),
)
plot!(figb,
    xlims=(10e-3, 1.05),
    ylims=(-2.05, -6e-3))
plot!(figb,
    legend=:topright,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)


figc = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-Peak position (mm)"
)
for (i,dir) in enumerate(data_directories)
    plot!(figc,
        EXP_data_processed[dir].Ic, EXP_data_processed[dir].Δ ,
        yerror = EXP_data_processed[dir].ErrΔ,
        label= dir,
        marker=(:circle,2,:white,0.6),
        markerstrokecolor=colores[i],
        line=(:solid,1,colores[i])
    )
end
plot!(figc,
    QM_df[!,:Ic],QM_df[!,:Δ],
    label=L"QM : $\Delta z$",
    seriestype=:scatter,
    marker=(:diamond,2,:white),
    markerstrokecolor=:red,
    line=(:red,1),
)
plot!(figc,
    legend=:bottomright,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    xlims=(10e-3, 1.05),
    ylims=(1e-3, 4.05),
)
figc1 = deepcopy(figc)
plot!(figc,xscale=:log10, yscale=:log10,
 xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
 yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)


<<<<<<< HEAD
fig = plot(figa,figb,figc,
=======
plot(figa,figa1, figb,figc, figc1,
>>>>>>> 41a6e1bd97275ed15d64dadbb23cfd6e082beb39
labelfontsize = 10,
layout=@layout[ a1 a2 ; a3; a4 a5],
size=(1200,1000),
top_margin=2mm,
left_margin=5mm,
bottom_margin=3mm,
);
display(fig)

#++++++++++++++++++++++++++++++++++++++++++++++++++














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
    ylabel=L"$F=1$ Peak position (mm)",
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
        markerstrokecolor=colores[idx],
    )
    plot!(fig,
        1000*data.Ic, data.F1[1],
        seriestype=:line,
        label= nothing,
        line=(:solid,0.3,colores[idx]),
        color = colores[idx],
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

new_data = Vector{NamedTuple}(undef, no)

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

new_centroid = [mean([new_data[v].F1[1][1],new_data[v].F2[1][1]]) for v=1:no]

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







