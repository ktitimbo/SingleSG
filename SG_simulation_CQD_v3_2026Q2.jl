# ============================================================================
#   Stern–Gerlach Experiment: Atom Trajectory Simulation
#
#   Simulates ³⁹K atom beam trajectories under CQD (Co-Quantum Dynamics)
#   theory for a range of coil currents and induction coefficients.
#
#   Author : Kelvin Titimbo
#   Affiliation : California Institute of Technology
#   Date : August 2025
# ============================================================================

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
using LinearAlgebra
using DataStructures, OrderedCollections
using Interpolations, Roots, Loess, Optim
using BSplineKit
using Polynomials
using DSP
using LambertW, PolyLog
using StatsBase
using Random, Statistics, NaNStatistics, Distributions, StaticArrays
# Miscellaneous utilities
using Alert
# Data manipulation
using DelimitedFiles, CSV, DataFrames, JLD2
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(1)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Set the working directory to the current location
cd(@__DIR__) ;
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
const BASE_PATH = raw"F:\SternGerlachExperiments";
const OUTDIR    = joinpath(@__DIR__, "simulation_data", "CQD2025_" * RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# Redirect Julia's temp files to a project-local folder (avoids /tmp clutter)
const TEMP_DIR = joinpath(@__DIR__,"artifacts", "JuliaTemp")
isdir(TEMP_DIR) || mkpath(TEMP_DIR);
ENV["TMPDIR"] = TEMP_DIR
ENV["TEMP"]   = TEMP_DIR
ENV["TMP"]    = TEMP_DIR
@info "Temporary directory configured" TEMP_DIR
# General setup & Diagnostic: record which machine produced this run
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
# Propagate global settings into the module
TheoreticalSimulation.SAVE_FIG = SAVE_FIG;
TheoreticalSimulation.FIG_EXT  = FIG_EXT;
TheoreticalSimulation.OUTDIR   = OUTDIR;

println("\n\t\tRunning process on:\t $(RUN_STAMP) \n")

atom        = "39K"  ;
## PHYSICAL CONSTANTS from NIST
# RSU : Relative Standard Uncertainty
const kb    = 1.380649e-23 ;       # Boltzmann constant             (J/K)
const ħ     = 6.62607015e-34/2π ;  # Reduced Planck constant        (J s)
const μ₀    = 1.25663706127e-6;    # Vacuum permeability            (Tm/A)
const μB    = 9.2740100657e-24 ;   # Bohr magneton                  (J/T)
const γₑ    = -1.76085962784e11 ;  # Electron gyromagnetic ratio    (1/sT). Relative Standard Uncertainty = 3.0e-10
const μₑ    = 9.2847646917e-24 ;   # Electron magnetic moment       (J/T). RSU = 3.0e-10
const Sspin = 1/2 ;                # Electron spin
const gₑ    = -2.00231930436092 ;  # Electron g-factor
## ATOM INFORMATION: 
# atom_info       = AtomicSpecies.atoms(atom)
K39_params = AtomParams(atom); # [R μn γn Ispin Ahfs M ] 
# Math constants
const TWOπ = 2π;
const INV_E = exp(-1);


# STERN--GERLACH EXPERIMENT
# Camera and pixel geometry : intrinsic properties
CAM_PIXELSIZE           = 6.5e-6 ;  # Physical pixel size of camera [m]
NX_PIXELS , NZ_PIXELS   = (2160, 2560); # (Nx,Nz) pixels
# Simulation resolution
SIM_BIN_X, SIM_BIN_Z                = (1,1) ;  # Camera binning
SIM_PIXELSIZE_X, SIM_PIXELSIZE_Z    = (SIM_BIN_X, SIM_BIN_Z) .* CAM_PIXELSIZE ; # Effective pixel size after binning [m]
# Image dimensions (adjusted for binning)
X_PIXELS = Int(NX_PIXELS / SIM_BIN_X);  # Number of x-pixels after binning
Z_PIXELS = Int(NZ_PIXELS / SIM_BIN_Z);  # Number of z-pixels after binning
# Spatial axes shifted to center the pixels
x_position = pixel_coordinates(X_PIXELS, SIM_BIN_X, SIM_PIXELSIZE_X);
z_position = pixel_coordinates(Z_PIXELS, SIM_BIN_Z, SIM_PIXELSIZE_Z);
println("""
***************************************************
CAMERA FEATURES
    Number of pixels        : $(NX_PIXELS) × $(NZ_PIXELS)
    Pixel size              : $(1e6*CAM_PIXELSIZE) μm

SIMULATION INFORMATION
    Binning                 : $(SIM_BIN_X) × $(SIM_BIN_Z)
    Effective pixels        : $(X_PIXELS) × $(Z_PIXELS)
    Pixel size              : $(1e6*SIM_PIXELSIZE_X)μm × $(1e6*SIM_PIXELSIZE_Z)μm
    xlims                   : ($(round(minimum(1e6*x_position), digits=6)) μm, $(round(maximum(1e3*x_position), digits=4)) mm)
    zlims                   : ($(round(minimum(1e6*z_position), digits=6)) μm, $(round(maximum(1e3*z_position), digits=4)) mm)
***************************************************
""")
# Furnace
const TCelsius = 200
const T_K = 273.15 + TCelsius ; # Furnace temperature (K)
# Furnace aperture
const X_FURNACE = 2.0e-3 ;
const Z_FURNACE = 100e-6 ;
# Slit : Pre SG
const X_SLIT  = 4.0e-3 ;
const Z_SLIT  = 300e-6 ;
# Circular Aperture : Post SG
const R_aper            = 5.8e-3/2 ;
const y_SGToAperture    = 1.0e-3 ;
# Propagation distances
const y_FurnaceToSlit = 541.75e-3 ;
const y_SlitToSG      = 44.0e-3 ;
const y_SG            = 7.0e-2 ;
const y_SGToScreen    = 395.25e-3 ;
# Connecting pipes
const R_tube = 35e-3/2 ; # Radius of the connecting pipe (m)
effusion_params = BeamEffusionParams(X_FURNACE,Z_FURNACE,X_SLIT,Z_SLIT,y_FurnaceToSlit,T_K,K39_params);
println("""
***************************************************
SETUP FEATURES
    Temperature             : $(T_K)
    Furnace aperture (x,z)  : ($(1e3*X_FURNACE)mm , $(1e6*Z_FURNACE)μm)
    Slit (x,z)              : ($(1e3*X_SLIT)mm , $(1e6*Z_SLIT)μm)
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
TheoreticalSimulation.default_camera_pixel_size = CAM_PIXELSIZE;
TheoreticalSimulation.default_x_pixels          = NX_PIXELS;
TheoreticalSimulation.default_z_pixels          = NZ_PIXELS;
TheoreticalSimulation.default_x_furnace         = X_FURNACE;
TheoreticalSimulation.default_z_furnace         = Z_FURNACE;
TheoreticalSimulation.default_x_slit            = X_SLIT;
TheoreticalSimulation.default_z_slit            = Z_SLIT;
TheoreticalSimulation.default_y_FurnaceToSlit   = y_FurnaceToSlit;
TheoreticalSimulation.default_y_SlitToSG        = y_SlitToSG;
TheoreticalSimulation.default_y_SG              = y_SG;
TheoreticalSimulation.default_y_SGToScreen      = y_SGToScreen;
TheoreticalSimulation.default_R_tube            = R_tube;
TheoreticalSimulation.default_c_aperture        = R_aper;
TheoreticalSimulation.default_y_SGToAperture    = y_SGToAperture;
##################################################################################################
JLD2_MyTools.save_script_copy(OUTDIR; script_path=@__FILE__, timestamp=RUN_STAMP)
##################################################################################################
##################################################################################################

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
# Coil currents
const ICOILS = [0.00,
            0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
            0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,
            0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
];
nI = length(ICOILS);
@info "No of currents sampled : $(nI)"

# Sample size: number of atoms arriving to the screen
const Nss = 2_000 ; 
@info "Number of MonteCarlo particles : $(Nss)\n"

nx_bins , nz_bins = 32 , 2
gaussian_width_mm = 0.200
λ0_raw            = 0.01
λ0_spline         = 0.001

# ============================================================================
# MONTE CARLO BEAM GENERATION
# ============================================================================
# Monte Carlo generation of particles traversing the filtering slit [x0 y0 z0 v0x v0y v0z]
crossing_slit = generate_samples(Nss, effusion_params; v_pdf=:v3, rng = rng_set, multithreaded = false, base_seed = base_seed_set);
jldsave( joinpath(OUTDIR,"cross_slit_particles_$(Nss).jld2"), data = crossing_slit)

# Diagnostic plots (saved only when SAVE_FIG = true)
if SAVE_FIG
    plot_μeff(K39_params,"mm_effective")
    plot_SG_geometry("SG_geometry")
    plot_velocity_stats(crossing_slit, "Initial data" , "velocity_pdf")
    # plot_velocity_stats(pairs_UP, "data μ–up" , "velocity_pdf_up")
    # plot_velocity_stats(pairs_DOWN, "data μ–down" , "velocity_pdf_down")
end

##################################################################################################
# COQUANTUM DYNAMICS : Initial conditions
##################################################################################################

# Partition atoms into magnetic moment-up (μ ↑↑ ẑ) and magnetic moment-down (μ ↑↓ ẑ) populations.
# Each particle is assigned a polar angles (θₑ, θₙ) for the electron-nuclear magnetic moments.
data_UP, data_DOWN = generate_CQDinitial_conditions(Nss, crossing_slit, rng_set; mode=:partition);

# ============================================================================
# CQD TRAJECTORY INTEGRATION  (reference kᵢ)
# ============================================================================
ki_ref = 1.60e-6;
# --- Magnetic moment-up branch ---
@time CQD_up_particles_flag         = TheoreticalSimulation.CQD_flag_travelling_particles(ICOILS, data_UP, ki_ref, K39_params; y_length=5001,verbose=true);
@time CQD_up_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles(ICOILS, ki_ref, data_UP, CQD_up_particles_flag, K39_params);     # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]
# --- Magnetic moment-down branch ---
@time CQD_dw_particles_flag         = TheoreticalSimulation.CQD_flag_travelling_particles(ICOILS, data_DOWN, ki_ref, K39_params; y_length=5001,verbose=true);
@time CQD_dw_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles(ICOILS, ki_ref, data_DOWN, CQD_dw_particles_flag, K39_params);   # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]

# Print per-current survival statistics for each branch
TheoreticalSimulation.CQD_travelling_particles_summary(ICOILS,CQD_up_particles_trajectories, :up)
TheoreticalSimulation.CQD_travelling_particles_summary(ICOILS,CQD_dw_particles_trajectories, :down)

# Extract only the particles that reach the screen
CQD_up_screen = OrderedDict(:Icoils=>ICOILS, :data => TheoreticalSimulation.CQD_select_flagged(CQD_up_particles_trajectories,:screen ))
CQD_dw_screen = OrderedDict(:Icoils=>ICOILS, :data => TheoreticalSimulation.CQD_select_flagged(CQD_dw_particles_trajectories,:screen ))

jldsave(joinpath(OUTDIR,"cqd_$(Nss)_up_screen.jld2"), screen = CQD_up_screen )
jldsave(joinpath(OUTDIR,"cqd_$(Nss)_dw_screen.jld2"), screen = CQD_dw_screen )

# ============================================================================
# PROFILE ANALYSIS  (reference kᵢ)
# ============================================================================

mm_up = TheoreticalSimulation.CQD_analyze_profiles_to_dict(CQD_up_screen;
    n_bins = (nx_bins , nz_bins), width_mm = gaussian_width_mm, 
    add_plot = false, plot_xrange= :all, branch=:up,
    λ_raw = λ0_raw, λ_smooth = λ0_spline, mode = :probability);

mm_dw = TheoreticalSimulation.CQD_analyze_profiles_to_dict(CQD_dw_screen;
    n_bins = (nx_bins , nz_bins), width_mm = gaussian_width_mm, 
    add_plot = false, plot_xrange= :all, branch=:dw,
    λ_raw = λ0_raw, λ_smooth = λ0_spline, mode = :probability);

jldsave(
    joinpath(OUTDIR,"cqd_$(Nss)_screen_profiles.jld2"), 
    profile = OrderedDict(
    :nz_bins    => nz_bins,
    :gauss_w    => gaussian_width_mm,
    :smoothing   => (λ0_raw,λ0_spline),
    :ki         => ki_ref,
    :mmup       => mm_up,
    :mmdw       => mm_dw
    ) 
)

# ============================================================================
# ANIMATED PROFILE OVERVIEW  (reference kᵢ)
# ============================================================================
# Profiles : up and down
anim = @animate for j in eachindex(ICOILS)
    isodd(j) || continue        # keep only every other
    fig = plot(
        title=L"CQD profiles : $k_i = %$(round(1e6*ki_ref, sigdigits=2))\times 10^{-6}$",
        legend=:topleft,
        legendtitle=L"$I_{0}=%$(ICOILS[j])\mathrm{A}$",
        legendtitlefontsize=8,
        yformatter = val -> string(round(val * 1e4, digits = 2)),
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",)
    plot!(mm_up[j][:z_profile][:,1],mm_up[j][:z_profile][:,3],
        label=L"$\vec{\mu}\upuparrows \hat{z}$",
        line=(:solid,:orangered2,1),
        marker=(:circle,:white,2),
        markerstrokecolor=:orangered2,
        markerstrokewidth=1)
    vline!([mm_up[j][:z_max_smooth_spline_mm]], 
        line=(:orangered2,0.5), 
        label=L"$z_{\mathrm{max}}=%$(round(mm_up[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
    plot!(mm_dw[j][:z_profile][:,1],mm_dw[j][:z_profile][:,3],
        label=L"$\vec{\mu}\updownarrows \hat{z}$",
        line=(:solid,:dodgerblue3,1),
        marker=(:circle,:white,2),
        markerstrokecolor=:dodgerblue3,
        markerstrokewidth=1)
    vline!([mm_dw[j][:z_max_smooth_spline_mm]],
        line=(:dodgerblue3,0.5), 
        label=L"$z_{\mathrm{max}}=%$(round(mm_dw[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
    plot!(        
        background_color_legend = nothing,
        foreground_color_legend = nothing,)
    display(fig)
end
gif_path = joinpath(OUTDIR, "CQD_profiles.gif");
gif(anim, gif_path, fps=2)  # adjust fps 
@info "Saved GIF" gif_path ;
anim = nothing      # free memory

# ============================================================================
# SPATIAL DISTRIBUTION ANIMATION ALONG THE BEAMLINE
# ============================================================================
# Six-panel GIF showing the 2-D (x, z) atom density at each stage:
#   furnace → slit → SG entrance → SG exit → circular aperture → screen
 
# --- Static geometry overlays (computed once) ---
# SG pole-tip edge profile in the (x, z) plane
x_magnet_m  = 1e-3 .* range(-1.0, 1.0; length=1000)   # [m]
z_edge_μm   = 1e6 .* TheoreticalSimulation.z_magnet_edge.(x_magnet_m)
x_magnet_mm = 1e3 .* x_magnet_m                        # [mm]
 
# Post-SG circular aperture boundary centered at the main axis (0,0)
θ_circ     = range(0, 2π; length=361)
R_mm       = 1e3 * R_aper
x_circ_mm  = R_mm .* cos.(θ_circ)
z_circ_μm  = 1e3 .* R_mm .* sin.(θ_circ)   # 1e3 because plotting in μm
 
# --- Absolute y-positions of each beamline stage [m] ---
y_furn   = 0.0
y_slit   = y_FurnaceToSlit
y_sg_in  = y_FurnaceToSlit + y_SlitToSG
y_sg_out = y_sg_in + y_SG
y_aper   = y_sg_out + y_SGToAperture
y_scr    = y_sg_out + y_SGToScreen
 
# Iterate over currents; the last frame is always included via Iterators.flatten
iter = let r = 1:1:nI
    (isempty(r) || last(r) == nI) ? r : Iterators.flatten((r, (NI,)))
end
 
anim = @animate for j in iter
    data_set = CQD_up_screen[:data][j]
    n = Int(floor(size(data_set, 1)/10))
 
    # Pre-allocate position arrays at each stage (units: mm, mm, mm, mm, mm, mm / μm respectively)
    xs_furn  = Vector{Float64}(undef, n); zs_furn  = Vector{Float64}(undef, n)
    xs_slit  = Vector{Float64}(undef, n); zs_slit  = Vector{Float64}(undef, n)
    xs_sgin  = Vector{Float64}(undef, n); zs_sgin  = Vector{Float64}(undef, n)
    xs_sgout = Vector{Float64}(undef, n); zs_sgout = Vector{Float64}(undef, n)
    xs_aper  = Vector{Float64}(undef, n); zs_aper  = Vector{Float64}(undef, n)
    xs_scr   = Vector{Float64}(undef, n); zs_scr   = Vector{Float64}(undef, n)
 
    # Single pass over all particles: propagate to each stage
    @inbounds for i in 1:n
        x0, y0, z0   = data_set[i,1], data_set[i,2], data_set[i,3]
        v0x, v0y, v0z = data_set[i,4], data_set[i,5], data_set[i,6]
        θe, θn        = data_set[i,7], data_set[i,8]
        pos0 = [x0, y0, z0]
        vel0 = [v0x, v0y, v0z]
 
        # Record furnace plane (initial conditions)
        xs_furn[i] = 1e3 * x0
        zs_furn[i] = 1e6 * z0
 
        # Helper: propagate to a target y-plane and return (x,z) in (mm, μm)
        # Time of flight τ = Δy / v₀y; valid in the paraxial approximation.
        function propagate_xmm_zum(τ)
            r, _ = TheoreticalSimulation.CQD_EqOfMotion(
                τ, ICOILS[j], μₑ, pos0, vel0, θe, θn, ki_ref, K39_params)
            return 1e3*r[1], 1e6*r[3]
        end
 
        xs_slit[i],  zs_slit[i]  = propagate_xmm_zum(y_slit   / v0y)
        xs_sgin[i],  zs_sgin[i]  = propagate_xmm_zum(y_sg_in  / v0y)
        xs_sgout[i], zs_sgout[i] = propagate_xmm_zum(y_sg_out / v0y)
        xs_aper[i],  zs_aper[i]  = propagate_xmm_zum(y_aper   / v0y)
 
        # Screen: z in mm (not μm) to match experimental axis
        r, _ = TheoreticalSimulation.CQD_EqOfMotion(
            y_scr / v0y, ICOILS[j], μₑ, pos0, vel0, θe, θn, ki_ref, K39_params)
        xs_scr[i] = 1e3 * r[1]
        zs_scr[i] = 1e3 * r[3]
    end
 
    # Auto-bin widths via Freedman–Diaconis rule (robust to outliers)
    bins_furn  = (FreedmanDiaconisBins(xs_furn),  FreedmanDiaconisBins(zs_furn))
    bins_slit  = (FreedmanDiaconisBins(xs_slit),  FreedmanDiaconisBins(zs_slit))
    bins_sgin  = (FreedmanDiaconisBins(xs_sgin),  FreedmanDiaconisBins(zs_sgin))
    bins_sgout = (FreedmanDiaconisBins(xs_sgout), FreedmanDiaconisBins(zs_sgout))
    bins_aper  = (FreedmanDiaconisBins(xs_aper),  FreedmanDiaconisBins(zs_aper))
    bins_scr   = (FreedmanDiaconisBins(xs_scr),   FreedmanDiaconisBins(zs_scr))
 
    # --- Helper: white label box + annotation ---
    function annotate_panel!(p, xpos, ypos, dx, dy, label_str)
        plot!(p, Shape(
            [xpos-dx, xpos+dx, xpos+dx, xpos-dx],
            [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
            color=:white, opacity=0.65, linealpha=0, label=false)
        annotate!(p, xpos, ypos, text(label_str, 10, :black, :bold, :center, "Helvetica"))
    end
 
    # ---- Panel A : Furnace ----
    pA = histogram2d(xs_furn, zs_furn;
        bins=bins_furn, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x\,(\mathrm{mm})$", ylabel=L"$z\,(\mathrm{\mu m})$",
        xticks=-1.0:0.25:1.0, yticks=-50:25:50)
    annotate_panel!(pA, -0.75, 35, 0.15, 7, "Furnace")
 
    # ---- Panel B : Slit ----
    pB = histogram2d(xs_slit, zs_slit;
        bins=bins_slit, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x\,(\mathrm{mm})$", ylabel=L"$z\,(\mathrm{\mu m})$",
        xticks=-4.0:0.5:4.0, yticks=-200:50:200,
        xlims=(-4,4), ylims=(-200,200))
    annotate_panel!(pB, -3.5, 150, 0.4, 20, "Slit")
 
    # ---- Panel C : SG entrance ----
    pC = histogram2d(xs_sgin, zs_sgin;
        bins=bins_sgin, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x\,(\mathrm{mm})$", ylabel=L"$z\,(\mathrm{\mu m})$",
        xticks=-4.0:0.5:4.0, yticks=-1000:100:1000,
        xlims=(-4,4), ylims=(-250,250))
    annotate_panel!(pC, -3.0, 180, 0.8, 30, "SG entrance")
 
    # ---- Panel D : SG exit  (with pole-tip edge overlay) ----
    pD = histogram2d(xs_sgout, zs_sgout;
        bins=bins_sgout, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x\,(\mathrm{mm})$", ylabel=L"$z\,(\mathrm{\mu m})$",
        xticks=-4.0:0.5:4.0, yticks=-1000:200:1000,
        xlims=(-4,4), ylims=(-300,1000))
    plot!(pD, x_magnet_mm, z_edge_μm, line=(:dash, :black, 2), label=false)
    annotate_panel!(pD, -3.0, 700, 0.6, 160, "SG exit")
 
    # ---- Panel E : Circular aperture ----
    pE = histogram2d(xs_aper, zs_aper;
        bins=bins_aper, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x\,(\mathrm{mm})$", ylabel=L"$z\,(\mathrm{\mu m})$",
        xticks=-4.0:0.5:4.0, yticks=-1000:500:3000,
        xlims=(-4,4), ylims=(-300,3000))
    plot!(pE, x_circ_mm, z_circ_μm; linestyle=:dash, lw=2, color=:gray, legend=false)
    annotate_panel!(pE, -3.0, 2400, 0.7, 270, "⊚ Aperture")
 
    # ---- Panel F : Detection screen  (z in mm) ----
    pF = histogram2d(xs_scr, zs_scr;
        bins=bins_scr, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x\,(\mathrm{mm})$", ylabel=L"$z\,(\mathrm{mm})$",
        ylims=(-1, 17.5))
    annotate_panel!(pF, -4.0, 14, 0.9, 0.9, "Screen")
 
    # Compose 6-panel layout
    fig = plot(pA, pB, pC, pD, pE, pF;
        layout       = (6,1),
        suptitle     = L"$I_{0} = %$(Int(1000*ICOILS[j]))\,\mathrm{mA}$",
        size         = (750, 800),
        right_margin = 2mm,
        bottom_margin = -2mm,
    )
    # Remove redundant x-axis labels on all but the bottom panel
    for k in 1:5
        plot!(fig[k], xlabel="", bottom_margin=-3mm)
    end
    display(fig)
end
let gif_path = joinpath(OUTDIR, "CQD_time_evolution.gif")
    gif(anim, gif_path, fps=2)
    @info "Saved GIF" gif_path
end
anim = nothing


# ============================================================================
# kᵢ PARAMETER SWEEP
# ============================================================================
# Alternative: multi-decade logarithmic grid (uncomment to use)
# kis = round.([
#     [exp10(p) * x for p in -8:-8 for x in 1.0:1:9]; 
#     [exp10(p) * x for p in -7:-7 for x in 1.0:1:9]; 
#     [exp10(p) * x for p in -6:-6 for x in 1.0:0.1:9.9]; 
#     [exp10(p) * x for p in -5:-5 for x in 1.0:1:9]; 
#     exp10.(-4:0)
# ];sigdigits=4)
kis = unique(round.(vcat([x * exp10(p) for p in -6:-6 for x in 0.5:0.5:5.0],0.001);sigdigits=4))
@info "Number of ki sampled = $(length(kis))"

# Visual check: plot the sampled kᵢ values on a log scale
fig=scatter(2*ones(length(kis)), kis,
    marker=(:circle,2,:white,stroke(:red,1)),
    label=L"Induction coefficient $k_{i}$")
scatter!(xlims=(1.5,2.5),
xticks=nothing,
yscale=:log10,
yticks = ([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-8}", L"10^{-7}", L"10^{-6}",L"10^{-5}", L"10^{-4}",L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
size=(300,600),
left_margin=3mm,
legend=:outerbottom)
display(fig)

# Storage: rows = kᵢ values, columns = coil currents
dta_ki_up = zeros(length(kis), nI);
dta_ki_dw = zeros(length(kis), nI);
# Create sub-directories for raw screen data
isdir(joinpath(OUTDIR,"up")) || mkpath(joinpath(OUTDIR,"up"));
isdir(joinpath(OUTDIR,"dw")) || mkpath(joinpath(OUTDIR,"dw"));

# ---- Main sweep loop ----
for (i, ki) in enumerate(kis)
    @info "Running for kᵢ = $(round(1e6*ki,sigdigits=3))×10⁻⁶"
 
    # Integrate trajectories for this ki
    @time tmp_up_flags = TheoreticalSimulation.CQD_flag_travelling_particles(
        ICOILS, data_UP, ki, K39_params; y_length=5001, verbose=true)
    @time tmp_up_traj  = TheoreticalSimulation.CQD_build_travelling_particles(
        ICOILS, ki, data_UP, tmp_up_flags, K39_params)
 
    @time tmp_dw_flags = TheoreticalSimulation.CQD_flag_travelling_particles(
        ICOILS, data_DOWN, ki, K39_params; y_length=5001, verbose=true)
    @time tmp_dw_traj  = TheoreticalSimulation.CQD_build_travelling_particles(
        ICOILS, ki, data_DOWN, tmp_dw_flags, K39_params)
 
    TheoreticalSimulation.CQD_travelling_particles_summary(ICOILS, tmp_up_traj, :up)
    TheoreticalSimulation.CQD_travelling_particles_summary(ICOILS, tmp_dw_traj, :down)
 
    # Select screen arrivals and persist to disk
    tmp_up_screen = OrderedDict(
        :Icoils => ICOILS,
        :N      => Nss,
        :T      => T_K,
        :ki     => ki,
        :data   => TheoreticalSimulation.CQD_select_flagged(tmp_up_traj, :screen))
    tmp_dw_screen = OrderedDict(
        :Icoils => ICOILS,
        :N      => Nss,
        :T      => T_K,
        :ki     => ki,
        :data   => TheoreticalSimulation.CQD_select_flagged(tmp_dw_traj, :screen))
 
    jldsave(joinpath(OUTDIR, "up", "cqd$(RUN_STAMP)_$(Nss)_ki$(@sprintf("%03d",i))_up_screen.jld2"),
        screen = tmp_up_screen)
    jldsave(joinpath(OUTDIR, "dw", "cqd$(RUN_STAMP)_$(Nss)_ki$(@sprintf("%03d",i))_dw_screen.jld2"),
        screen = tmp_dw_screen)
 
    # Compute smoothed profiles and extract peak positions
    tmp_mm_up = TheoreticalSimulation.CQD_analyze_profiles_to_dict(tmp_up_screen;
        n_bins=(nx_bins, nz_bins), width_mm=gaussian_width_mm,
        add_plot=false, plot_xrange=:all, branch=:up,
        λ_raw=λ0_raw, λ_smooth=λ0_spline, mode=:probability)
 
    tmp_mm_dw = TheoreticalSimulation.CQD_analyze_profiles_to_dict(tmp_dw_screen;
        n_bins=(nx_bins, nz_bins), width_mm=gaussian_width_mm,
        add_plot=false, plot_xrange=:all, branch=:dw,
        λ_raw=λ0_raw, λ_smooth=λ0_spline, mode=:probability)
 
    dta_ki_up[i, :] = [tmp_mm_up[v][:z_max_smooth_spline_mm] for v in 1:nI]
    dta_ki_dw[i, :] = [tmp_mm_dw[v][:z_max_smooth_spline_mm] for v in 1:nI]
 
    # Explicitly free memory before the next iteration
    tmp_up_flags = tmp_up_traj = tmp_dw_flags = tmp_dw_traj = nothing
    tmp_up_screen = tmp_dw_screen = tmp_mm_up = tmp_mm_dw = nothing
    GC.gc()
end
jldsave( joinpath(OUTDIR, "cqd_$(Nss)_kis.jld2"), 
        data=OrderedDict(
            :Icoils     => ICOILS,
            :ki         => kis,
            :nx_bins    => nx_bins,
            :nz_bins    => nz_bins,
            :gauss_w    => gaussian_width_mm,
            :smoothing   => (λ0_raw,λ0_spline),
            :up         => dta_ki_up,
            :dw         => dta_ki_dw)
)

# ============================================================================
# Ref kᵢ SWEEP — COMPARISON PLOT
# ============================================================================

cls = palette(:darkrainbow, length(kis))
fig_sweep = plot(xlabel=L"$I_{c}\,(\mathrm{A})$", ylabel=L"$z_{\mathrm{max}} \, \ (\mathrm{mm})$")
Isim_start_idx = findfirst(>=(0.010),ICOILS)

for i in eachindex(kis)
    plot!(fig_sweep, ICOILS[Isim_start_idx:end], abs.(dta_ki_up[i,Isim_start_idx:end]),
    label=L"$k_{i} = %$(round(1e6*kis[i], sigdigits=2))\times 10^{-6}$",
    line=(cls[i],1))
end
plot!(fig_sweep, 
    size=(1200,800),
    xaxis=:log10,
    yaxis=:log10,
    xlims=(8e-3,2),
    ylims=(8e-3,2),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], 
            [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    legend_columns=3,
    background_color_legend=nothing,
    foreground_color_legend=nothing
)
display(fig_sweep)
savefig(fig_sweep, joinpath(OUTDIR,"cqd_$(Nss)_kis_comparison.$(FIG_EXT)"))


#########################################################################################
T_END = Dates.now()
T_RUN = Dates.canonicalize(T_END-T_START)
report = """
***************************************************
EXPERIMENT
    Single Stern–Gerlach Experiment
    CO-QUANTUM DYNAMICS SIMULATION
    atom                    : $(atom)
    Output directory        : $(OUTDIR)
    RUN_STAMP               : $(RUN_STAMP)

CAMERA FEATURES
    Number of pixels        : $(NX_PIXELS) × $(NZ_PIXELS)
    Pixel size              : $(1e6*CAM_PIXELSIZE) μm

SETUP FEATURES
    Temperature             : $(T_K)
    Furnace aperture (x,z)  : ($(1e3*X_FURNACE)mm , $(1e6*Z_FURNACE)μm)
    Slit (x,z)              : ($(1e3*X_SLIT)mm , $(1e6*Z_SLIT)μm)
    Furnace → Slit          : $(1e3*y_FurnaceToSlit)mm
    Slit → SG magnet        : $(1e3*y_SlitToSG)mm
    SG magnet               : $(1e3*y_SG)mm
    SG magnet → Screen      : $(1e3*y_SGToScreen)mm
    Tube radius             : $(1e3*R_tube)mm

SIMULATION INFORMATION
    Number of atoms         : $(Nss)
    Binning (nx,nz)         : ($(nx_bins),$(nz_bins))
    Gaussian width (mm)     : $(gaussian_width_mm)
    Smoothing raw           : $(λ0_raw)
    Smoothing spline        : $(λ0_spline)
    Currents (A)            : $(round.(ICOILS,sigdigits=3))
    No. of currents         : $(nI)
    Induction term (×10⁻⁶)  : $(round.(1e6*kis, sigdigits=3))

CODE
    Code name               : $(PROGRAM_FILE)
    Start date              : $(T_START)
    End data                : $(T_END)
    Run time                : $(T_RUN)
    Hostname                : $(HOSTNAME)

***************************************************
"""
# Print to terminal
println(report)

# Save to file
open(joinpath(OUTDIR,"simulation_report.txt"), "w") do io
    write(io, report)
end

println("DATA COLLECTED : script $RUN_STAMP has finished!")


################################################################################################
################################################################################################
# ============================================================================
#   kᵢ PARAMETER SWEEP — Profile Analysis
#
#   Runs after the trajectory data for all kᵢ values has been generated.
#   For each kᵢ and each (nz, σ, λ) hyperparameter combination, computes
#   smoothed 1-D intensity profiles and saves z_max to a JLD2 table.
#
#   Both magnetic moment branches (up / down) are handled by a single generic function
#   to avoid code duplication.
# ============================================================================

# =======================
# Global parameters
# =======================
Ns = Nss
@info "Number of ki sampled : $(length(kis))"


# ============================================================================
# PARAMETER GRID
# ============================================================================
induction_coeff_label     = round.(1e6 .* kis, sigdigits=4)
NX_BINS_SWEEP       = 32 # fixed nx bins
NZ_BINS_SWEEP       = [1, 2, 4]
GAUSSIAN_WIDTHS_MM  = [0.001, 0.010, 0.025, 0.050, 0.065, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.270, 0.275, 0.300, 0.350, 0.400, 0.450, 0.500 ]; # try different gaussian widths
λ0_RAW_LIST         = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10]; # try different smoothing factors for raw data
λ0_SPLINE_SWEEP      = 0.001

# ---------- precompute param grid ----------
const PARAM_GRID   = [(nz, gw, λ0_raw)
          for nz in NZ_BINS_SWEEP
          for gw in GAUSSIAN_WIDTHS_MM   
          for λ0_raw in λ0_RAW_LIST  ]


@info "Hyperparameter combinations per kᵢ file = $(length(PARAM_GRID))"
@info "Grand total profile computations (both branches) = $(2 * length(kis) * length(PARAM_GRID))"

# ============================================================================
# FILE DISCOVERY AND PRE-RUN SANITY CHECK
# ============================================================================
# Files are ~15 GB each — we do NOT load them just to verify metadata.
# Instead we (a) parse what we can from the filenames, and (b) print a
# summary table of what we *expect* every file to contain, based on the
# in-memory values of N, T, and kᵢ used when they were generated.
# If the files were produced and saved in the same run (or from the same
# parameter set), this is sufficient: wrong-folder mixing is impossible.
 
function discover_files(indir, label)
    files = sort(filter(readdir(indir)) do f
        isfile(joinpath(indir, f)) && endswith(f, ".jld2")
    end)
    @assert length(files) == length(kis) """
    [$label] File count mismatch:
        found   : $(length(files)) files in $indir
        expected: $(length(kis))   (one per kᵢ value)
    """
    @info "[$label] Files found" n=length(files) indir
    return files
end

const INDIR_UP = joinpath(OUTDIR, "up");
const INDIR_DW = joinpath(OUTDIR, "dw");

files_up = discover_files(INDIR_UP, "up")
files_dw = discover_files(INDIR_DW, "dw")

# --- Print expected metadata table (no file I/O) ---
# Compute column widths from actual content so separators always fit
w_idx  = max(6,  ndigits(length(induction_coeff_label)))
w_ki   = max(15, maximum(length(@sprintf("%.4g", k)) for k in induction_coeff_label))
w_n    = max(10, ndigits(Ns))
w_up   = max(7,  maximum(length.(files_up)))
w_dw   = max(7,  maximum(length.(files_dw)))
w_tot  = 2 + w_idx + 2 + w_ki + 2 + w_n + 2 + w_up + 2 + w_dw + 2
 
header = @sprintf("  %-*s  %-*s  %-*s  %-*s  %-*s",
    w_idx, "index",
    w_ki,  "kᵢ (×10⁻⁶)",
    w_n,   "N",
    w_up,  "up file",
    w_dw,  "dw file")
 
println("\n" * "="^w_tot)
println("  PRE-RUN PARAMETER CHECK  (values used during data generation)")
println("="^w_tot)
println(header)
println("-"^w_tot)
for (i, ki) in enumerate(induction_coeff_label)
    println(@sprintf("  %-*d  %-*.4g  %-*d  %-*s  %-*s",
        w_idx, i,
        w_ki,  ki,
        w_n,   Ns,
        w_up,  files_up[i],
        w_dw,  files_dw[i]))
end
println("-"^w_tot)
println(@sprintf("  T = %.2f K    N = %d    λ0_spline = %.4f", T_K, Ns, λ0_SPLINE_SWEEP))
println("="^w_tot * "\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
 
"""
    init_output_jld2(path, branch, files, kis) -> path
 
Create the output JLD2 table and write all run metadata.
"""
function init_output_jld2(path, branch::Symbol, files, kis)
    jldopen(path, "w") do f
        f["meta/N"]        = Ns
        f["meta/T"]        = T_K
        f["meta/branch"]   = string(branch)
        f["meta/s_spline"] = λ0_SPLINE_SWEEP
        f["meta/nx"]       = NX_BINS_SWEEP
        f["/meta/nz"]      = NZ_BINS_SWEEP
        f["/meta/σw"]      = GAUSSIAN_WIDTHS_MM
        f["/meta/λ0"]      = λ0_RAW_LIST
        f["/meta/ki"]      = kis
        f["/meta/files"]   = files
    end
    @info "Initialised output JLD2" path branch n_ki=length(kis)
    return path
end


"""
    process_branch(branch, indir, outjld, files, kis, params)
 
For each kᵢ file in `indir`, compute CQD intensity profiles over the full
hyperparameter grid in parallel (threaded) and write results to `outjld`.
Prints T, N, and kᵢ from the file's stored metadata before processing.
"""
function process_branch(branch::Symbol, indir, outjld, files, kis, params)
 
    nfiles = length(files)
    nt     = Threads.maxthreadid()
 
    for (j, fname) in pairs(files)
 
        ki_display = round(1e6 * kis[j], sigdigits=4)
        simpath    = joinpath(indir, fname)
 
        # --- Load file and echo stored metadata for visual verification ---
        data_sim = load(simpath, "screen")
 
        println("─"^60)
        @info "[$branch] File $j/$nfiles | T: $(get(data_sim,:T,"—")) / $(T_K) K  |  N: $(get(data_sim,:N,"—")) / $(Ns)  |  kᵢ: $(get(data_sim,:ki,"—")) / $(kis[j])" fname
        println("─"^60)
 
        # Thread-local result buffers — avoids lock contention during compute
        local_chunks = [Pair{Tuple{Float64,Int,Float64,Float64}, Any}[] for _ in 1:nt]
 
        # ---- Parallel profile computation over the hyperparameter grid ----
        @threads for (nz, gw, λ0_raw) in params
            profiles = TheoreticalSimulation.CQD_analyze_profiles_to_dict(
                data_sim;
                n_bins      = (NX_BINS_SWEEP, nz),
                width_mm    = gw,
                add_plot    = false,
                plot_xrange = :all,
                branch      = branch,
                λ_raw       = λ0_raw,
                λ_smooth    = λ0_SPLINE_SWEEP,
                mode        = :probability,
            )
            push!(local_chunks[threadid()], (ki_display, nz, gw, λ0_raw) => profiles)
        end
 
        # ---- Serial I/O — write all results for this file in one open ----
        jldopen(outjld, "r+") do f
            for chunk in local_chunks, (key, profiles) in chunk
                ki2, nz, gw, λ0_raw = key
                label_path = JLD2_MyTools.make_keypath_cqd(branch, ki2, nz, gw, λ0_raw)
                haskey(f, label_path) || (f[label_path] = profiles)   # resume-safe
            end
            JLD2_MyTools.jld_overwrite!(f, "/meta/progress/last_completed_j",    j)
            JLD2_MyTools.jld_overwrite!(f, "/meta/progress/last_completed_file", fname)
            JLD2_MyTools.jld_overwrite!(f, "/meta/progress/last_completed_ki",   ki_display)
        end
 
        data_sim = nothing
        GC.gc()
        @info "[$branch] Done $j/$nfiles" free_GiB=round(Sys.free_memory()/1024^3; digits=3)
    end
 
    @info "Completed $branch table" outjld
end

# ============================================================================
# RUN BOTH BRANCHES
# ============================================================================
for (branch, indir, files) in [(:up, INDIR_UP, files_up), (:dw, INDIR_DW, files_dw)]
    outjld = joinpath(OUTDIR,
        "cqd_$(Ns)_$(branch)_profiles_bykey.jld2")
    init_output_jld2(outjld, branch, files, kis)
    @time process_branch(branch, indir, outjld, files, kis, PARAM_GRID)
end


######################################################################
T_END = Dates.now()
T_RUN = Dates.canonicalize(T_END-T_START)
report = """
***************************************************
EXPERIMENT
    Single Stern–Gerlach Experiment
    atom                    : $(atom)
    Output directory        : $(OUTDIR)
    RUN_STAMP               : $(RUN_STAMP)

CAMERA FEATURES
    Number of pixels        : $(NX_PIXELS) × $(NZ_PIXELS)
    Pixel size              : $(1e6*CAM_PIXELSIZE) μm

SETUP FEATURES
    Temperature             : $(T_K)
    Furnace aperture (x,z)  : ($(1e3*X_FURNACE)mm , $(1e6*Z_FURNACE)μm)
    Slit (x,z)              : ($(1e3*X_SLIT)mm , $(1e6*Z_SLIT)μm)
    Post-SG aperture radius : $(1e3*R_aper)mm
    Furnace → Slit          : $(1e3*y_FurnaceToSlit)mm
    Slit → SG magnet        : $(1e3*y_SlitToSG)mm
    SG magnet               : $(1e3*y_SG)mm
    SG magnet → Screen      : $(1e3*y_SGToScreen)mm
    SG magnet → Aperture    : $(1e3*y_SGToAperture)mm
    Tube radius             : $(1e3*R_tube)mm

SIMULATION INFORMATION
    Number of atoms         : $(Ns)
    Induction term          : $(kis)
    Binning (nx,nz)         : ($(NX_BINS_SWEEP),$(NZ_BINS_SWEEP))
    Gaussian width (mm)     : $(GAUSSIAN_WIDTHS_MM   )
    Smoothing raw           : $(λ0_RAW_LIST  )
    Smoothing spline        : $(λ0_SPLINE_SWEEP    )
    Currents (A)            : $(round.(ICOILS,sigdigits=3))
    No. of currents         : $(nI)

CODE
    Code name               : $(PROGRAM_FILE)
    Start date              : $(T_START)
    End data                : $(T_END)
    Run time                : $(T_RUN)
    Hostname                : $(HOSTNAME)

***************************************************
"""
# Print to terminal
println(report)

# Save to file
open(joinpath(OUTDIR,"simulation_cqd_report.txt"), "w") do io
    write(io, report)
end

println("DATA ANALYZED : script $RUN_STAMP has finished!")
alert("script $RUN_STAMP has finished!")


# path1 = joinpath("W:\\SternGerlach\\cqd_T200_8M","cqd_8M_up_profiles.jld2")
# path2 = joinpath("W:\\cqd_8000000_up_profiles_1_29_bykey.jld2")
# pathout_up = joinpath("W:\\SternGerlach","cqd_T200_8M","cqd_8M_up_profiles.jld2")
# JLD2_MyTools.merge2_cqd_jld2(path1, path2, pathout_up)

# path1 = joinpath("W:\\cqd_8M_dw_profiles.jld2")
# path2 = joinpath("W:\\cqd_8000000_dw_profiles_1_29_bykey.jld2")
# pathout_dw = joinpath("W:\\SternGerlach","cqd_T200_8M","cqd_8M_dw_profiles.jld2")
# JLD2_MyTools.merge2_cqd_jld2(path1, path2, pathout_dw)


# JLD2_MyTools.summarize_meta_cqd_jld2(pathout_up)
# JLD2_MyTools.summarize_meta_cqd_jld2(pathout_dw)

# d1 = JLD2_MyTools.list_keys_jld_cqd(pathout_up)
# d2 = JLD2_MyTools.list_keys_jld_cqd(pathout_dw)

# colores = palette(:darkrainbow, length(d1.ki))
# fig = plot(xlabel="Current (A)",
#     ylabel="Peak position (mm)")
# split = Matrix{Float64}(undef, nI, length(d1.ki));
# for (i,ki) in enumerate(d1.ki)
#     dataup = jldopen(pathout_up,"r") do f
#         f[JLD2_MyTools.make_keypath_cqd(:up,ki,2,0.2,0.01)]
#     end
#     zmm_up = [dataup[j][:z_max_smooth_spline_mm] for j=1:nI]

#     datadw = jldopen(pathout_dw,"r") do f
#         f[JLD2_MyTools.make_keypath_cqd(:dw,ki,2,0.2,0.01)]
#     end
#     zmm_dw = [datadw[j][:z_max_smooth_spline_mm] for j=1:nI]

#     plot!(fig,
#         ICOILS,zmm_up,
#         label=L"$%$(ki)$",
#         line=(:solid,1,colores[i])  
#     )

#     split[:,i] = zmm_up .- zmm_dw
# end
# hspan!([1e-4,6e-3], color=:gray67, fillalpha=0.2, label="pixel size")
# plot!(fig,
# legend=:outerright,
# legendtitle=L"$k_{i}\times 10^{-6}$",
# legendtitlefont=8,
# legend_columns=3,
# legendfontsize=6,
# foreground_color_legend = nothing,
# background_color_legend = nothing,
# xlims=(1e-3,1.05),
# ylims=(1e-3,2.05),
# size=(1000,600),
# left_margin=4mm,
# bottom_margin=2mm,
# )
# display(fig)
# plot!(
# xticks = ([1e-3, 1e-2, 1e-1, 1.0], 
#         [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
# yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
#         [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
# xscale=:log10,
# yscale=:log10,
# )

# plot(xlabel="Current (A)")
# for (i,ki) in enumerate(d1.ki)
#     plot!(ICOILS, split[:,i],
#         label=L"$%$(ki)$",
#         line=(:solid,1,colores[i]),
#     ) 
# end
# hspan!([1e-4,6e-3], color=:gray67, fillalpha=0.2, label="pixel size")
# plot!(ylabel="peak-to-peak separation (mm)",
#     legend=:outerright,
#     legendtitle=L"$k_{i}\times 10^{-6}$",
#     legendtitlefont=8,
#     legend_columns=3,
#     legendfontsize=6,
#     foreground_color_legend = nothing, 
#     background_color_legend = nothing,
#     size=(1000,600),
#     xlims=(1e-3,1.05),
#     ylims=(1e-3,4.05),
#     xticks = ([1e-3, 1e-2, 1e-1, 1.0], 
#             [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
#     yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
#             [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
#     xscale=:log10,
#     yscale=:log10,
#     left_margin=4mm,
#     bottom_margin=2mm,
# )

# plot(ones(73), d1.ki*1e-6,
#     label=L"Sampled $k_{i}$",
#     legend=:outerbottom,
#     yscale=:log10,
#     seriestype=:scatter,
#     marker=(:circle,2,:white),
#     markerstrokecolor=:red,
#     xlims=(0.5,1.5),
#     foreground_color_legend = nothing,
#     background_color_legend = nothing,
#     # ylims=(9e-8,10e-6),
#     size=(200,600),
#     xticks=:none,
#     yticks = ([1e-9, 1e-8, 1e-7, 1e-6,1e-5, 1e-4, 1e-3, 1e-2], 
#         [ L"10^{-9}", L"10^{-8}", L"10^{-7}", L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}"]),
#     left_margin=5mm,
# )


# using JLD2
# # --- helper: extract kis from stored keypaths under /up/ki=<...>e-6/... ---
# function collect_kis_from_file(path::AbstractString)::Vector{Float64}
#     kis = Float64[]
#     jldopen(path, "r") do fin
#         for k in keys(fin)
#             s = String(k)
#             startswith(s, "/up/ki=") || continue

#             # parse the substring between "/up/ki=" and "e-6/"
#             m = match(r"^/up/ki=([^/]+)e-6/", s)
#             m === nothing && continue

#             ki_scaled = parse(Float64, m.captures[1])  # this is the number before "e-6"
#             push!(kis, ki_scaled * 1e-6)               # convert to kis
#         end
#     end
#     return kis
# end

# function merge_up_outputs_streaming(out1::AbstractString,
#                                     out2::AbstractString,
#                                     outmerged::AbstractString;
#                                     overwrite_from_out2::Bool = true)

#     # -------------------------------
#     # 1) Build updated kis = union
#     # -------------------------------
#     kis1 = collect_kis_from_file(out1)
#     kis2 = collect_kis_from_file(out2)

#     kis_new = sort!(unique!(round.(vcat(kis1, kis2); sigdigits=4)))

#     # -------------------------------
#     # 2) Create fresh output file
#     # -------------------------------
#     jldopen(outmerged, "w") do _ end

#     # -------------------------------
#     # 3) Copy META (ignore /meta/progress), update /meta/induction_coeff
#     #    Rule: use meta from out1 as the base (except induction_coeff)
#     # -------------------------------
#     jldopen(out1, "r") do fin1
#         jldopen(outmerged, "r+") do fout
#             # keep these meta entries (and ignore progress)
#             for k in keys(fin1)
#                 s = String(k)
#                 startswith(s, "/meta/") || continue
#                 startswith(s, "/meta/progress/") && continue
#                 s == "/meta/induction_coeff" && continue  # we'll replace

#                 fout[s] = fin1[k]  # one dataset at a time
#             end

#             # set updated induction coeff
#             jld_overwrite!(fout, "/meta/induction_coeff", kis_new)
#         end
#     end

#     # -------------------------------
#     # 4) Copy all /up datasets from out1 into merged
#     # -------------------------------
#     jldopen(out1, "r") do fin1
#         jldopen(outmerged, "r+") do fout
#             for k in keys(fin1)
#                 s = String(k)
#                 startswith(s, "/up/") || continue
#                 # write as-is
#                 fout[s] = fin1[k]
#             end
#         end
#     end

#     # -------------------------------
#     # 5) Copy all /up datasets from out2 into merged (overwrite if exists)
#     # -------------------------------
#     jldopen(out2, "r") do fin2
#         jldopen(outmerged, "r+") do fout
#             for k in keys(fin2)
#                 s = String(k)
#                 startswith(s, "/up/") || continue

#                 if overwrite_from_out2
#                     jld_overwrite!(fout, s, fin2[k])
#                 else
#                     # if you ever want "skip if present"
#                     haskey(fout, s) || (fout[s] = fin2[k])
#                 end
#             end
#         end
#     end

#     return outmerged
# end


# outmerged = merge_up_outputs_streaming("output1.jld2", "output2.jld2", "merged_up.jld2")
# @info "Merged file written" outmerged


# function count_up_overlaps(out1, out2)
#     s1 = Set{String}()
#     jldopen(out1, "r") do f
#         for k in keys(f)
#             s = String(k)
#             startswith(s, "/up/") && push!(s1, s)
#         end
#     end
#     c = 0
#     jldopen(out2, "r") do f
#         for k in keys(f)
#             s = String(k)
#             startswith(s, "/up/") && (c += (s in s1))
#         end
#     end
#     return c
# end




# using JLD2

# # delete+write helper (JLD2 won't overwrite datasets by default)
# function jld_overwrite!(f, path::AbstractString, value)
#     if haskey(f, path)
#         delete!(f, path)
#     end
#     f[path] = value
#     return nothing
# end

# """
#     merge_branch_outputs_streaming(out1, out2, outmerged; branch=:up)

# Merge two JLD2 outputs produced by the same pipeline, for a given `branch` (e.g. `:up` or `:dw`).

# Behavior:
# - Copies `/meta/*` from `out1` into `outmerged`, except ignores `/meta/progress/*`.
# - Sets `/meta/induction_coeff` to the union of both files' `/meta/induction_coeff` (rounded to sigdigits=4).
# - Adds `/meta/merged_from` containing `[out1, out2]` and the `branch`.
# - Copies all datasets under `/<branch>/` from `out1`, then from `out2` overwriting overlaps (so `out2` wins).
# - Streaming: copies dataset-by-dataset (does not load whole files at once).
# """
# function merge_branch_outputs_streaming(out1::AbstractString,
#                                        out2::AbstractString,
#                                        outmerged::AbstractString;
#                                        branch::Symbol = :up)

#     bprefix = "/" * String(branch) * "/"   # "/up/" or "/dw/"

#     # ---- union kis from meta (fast; no path parsing) ----
#     kis1 = jldopen(out1, "r") do f
#         f["/meta/induction_coeff"]
#     end
#     kis2 = jldopen(out2, "r") do f
#         f["/meta/induction_coeff"]
#     end
#     kis_new = sort!(unique!(round.(vcat(kis1, kis2); sigdigits=4)))

#     # ---- create/overwrite merged file ----
#     jldopen(outmerged, "w") do _ end

#     # ---- write META (from out1), ignoring /meta/progress, updating induction_coeff + merged_from ----
#     jldopen(out1, "r") do fin1
#         jldopen(outmerged, "r+") do fout
#             for k in keys(fin1)
#                 s = String(k)
#                 startswith(s, "/meta/") || continue
#                 startswith(s, "/meta/progress/") && continue
#                 (s == "/meta/induction_coeff") && continue
#                 (s == "/meta/merged_from") && continue
#                 fout[s] = fin1[k]
#             end

#             jld_overwrite!(fout, "/meta/induction_coeff", kis_new)

#             merged_from = Dict(
#                 "branch" => String(branch),
#                 "sources" => [String(out1), String(out2)]
#             )
#             jld_overwrite!(fout, "/meta/merged_from", merged_from)
#         end
#     end

#     # ---- copy branch datasets from out1 ----
#     jldopen(out1, "r") do fin1
#         jldopen(outmerged, "r+") do fout
#             for k in keys(fin1)
#                 s = String(k)
#                 startswith(s, bprefix) || continue
#                 fout[s] = fin1[k]
#             end
#         end
#     end

#     # ---- copy branch datasets from out2 (overwrite) ----
#     jldopen(out2, "r") do fin2
#         jldopen(outmerged, "r+") do fout
#             for k in keys(fin2)
#                 s = String(k)
#                 startswith(s, bprefix) || continue
#                 jld_overwrite!(fout, s, fin2[k])  # out2 wins
#             end
#         end
#     end

#     return outmerged
# end


# merge_branch_outputs_streaming("out1_up.jld2", "out2_up.jld2", "merged_up.jld2"; branch=:up)
# merge_branch_outputs_streaming("out1_dw.jld2", "out2_dw.jld2", "merged_dw.jld2"; branch=:dw)
