# Simulation of atom trajectories in the Stern–Gerlach experiment
# Kelvin Titimbo
# California Institute of Technology
# August 2025

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
cd(@__DIR__) ;
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
T_K = 273.15 + 205 ; # Furnace temperature (K)
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
avg_data = load(joinpath(@__DIR__, "analysis_data", "smoothing_binning","data_averaged_2.jld2"), "data" )
I_exp  = avg_data[:i_smooth];
z_exp  = avg_data[:z_smooth];
δz_exp = avg_data[:δz_smooth];
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

# Sample size: number of atoms arriving to the screen
const Nss = 80_000 ; 
@info "Number of MonteCarlo particles : $(Nss)\n"

nx_bins , nz_bins = 32 , 2
gaussian_width_mm = 0.200
λ0_raw            = 0.01
λ0_spline         = 0.001

# Monte Carlo generation of particles traersing the filtering slit [x0 y0 z0 v0x v0y v0z]
crossing_slit = generate_samples(Nss, effusion_params; v_pdf=:v3, rng = rng_set, multithreaded = false, base_seed = base_seed_set);
jldsave( joinpath(OUTDIR,"cross_slit_particles_$(Nss).jld2"), data = crossing_slit)

if SAVE_FIG
    plot_μeff(K39_params,"mm_effective")
    plot_SG_geometry("SG_geometry")
    plot_velocity_stats(crossing_slit, "Initial data" , "velocity_pdf")
    # plot_velocity_stats(pairs_UP, "data μ–up" , "velocity_pdf_up")
    # plot_velocity_stats(pairs_DOWN, "data μ–down" , "velocity_pdf_down")
end

##################################################################################################
#   COQUANTUM DYNAMICS
##################################################################################################

# Monte Carlo generation of particles traversing the filtering slit and assigning polar angles
data_UP, data_DOWN = generate_CQDinitial_conditions(Nss, crossing_slit, rng_set; mode=:partition);
ki = 1.6e-6;
@time CQD_up_particles_flag         = TheoreticalSimulation.CQD_flag_travelling_particles(Icoils, data_UP, ki, K39_params; y_length=5001,verbose=true);
@time CQD_up_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles(Icoils, ki, data_UP, CQD_up_particles_flag, K39_params);     # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]
@time CQD_dw_particles_flag         = TheoreticalSimulation.CQD_flag_travelling_particles(Icoils, data_DOWN, ki, K39_params; y_length=5001,verbose=true);
@time CQD_dw_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles(Icoils, ki, data_DOWN, CQD_dw_particles_flag, K39_params);   # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]

TheoreticalSimulation.CQD_travelling_particles_summary(Icoils,CQD_up_particles_trajectories, :up)
TheoreticalSimulation.CQD_travelling_particles_summary(Icoils,CQD_dw_particles_trajectories, :down)

CQD_up_screen = OrderedDict(:Icoils=>Icoils, :data => TheoreticalSimulation.CQD_select_flagged(CQD_up_particles_trajectories,:screen ))
CQD_dw_screen = OrderedDict(:Icoils=>Icoils, :data => TheoreticalSimulation.CQD_select_flagged(CQD_dw_particles_trajectories,:screen ))

jldsave(joinpath(OUTDIR,"cqd_$(Nss)_up_screen.jld2"), screen = CQD_up_screen )
jldsave(joinpath(OUTDIR,"cqd_$(Nss)_dw_screen.jld2"), screen = CQD_dw_screen )

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
    :smothing   => (λ0_raw,λ0_spline),
    :ki         => ki,
    :mmup       => mm_up,
    :mmdw       => mm_dw
    ) 
)

# Profiles : up and down
anim = @animate for j in eachindex(Icoils)
    fig = plot(
        title=L"CQD profiles : $k_i = %$(round(1e6*ki, sigdigits=2))\times 10^{-6}$",
        legend=:topleft,
        legendtitle=L"$I_{0}=%$(Icoils[j])\mathrm{A}$",
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
    display(fig)
end
gif_path = joinpath(OUTDIR, "CQD_profiles.gif");
gif(anim, gif_path, fps=2)  # adjust fps 
@info "Saved GIF" gif_path ;
anim = nothing

# Profiles COMPARISON : different contributions 
anim = @animate for j in 1:4:length(Icoils)
    isodd(j) || continue        # keep only every other
    fig = plot(title=L"$I_{0}=%$(Int(1e3*Icoils[j]))\mathrm{mA}$",
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        yformatter = val -> string(round(val * 1e4, digits = 2)),)
    plot!(mm_up[j][:z_profile][:,1],mm_up[j][:z_profile][:,3],
        label=L"CQD UP $k_{i}=%$(round(1e6*ki,sigdigits=2))\times 10^{-6}$",
        line=(:solid,:maroon3,1),
        marker=(:circle,:white,2),
        markerstrokecolor=:maroon3,
        markerstrokewidth=1)
    vline!([mm_up[j][:z_max_smooth_spline_mm]],
        line=(:maroon3,0.5), 
        label=L"$z_{\mathrm{max}}=%$(round(mm_up[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
    plot!(mm_dw[j][:z_profile][:,1],mm_dw[j][:z_profile][:,3],
        label=L"CQD DOWN $k_{i}=%$(round(1e6*ki,sigdigits=2))\times 10^{-6}$",
        line=(:solid,:darkcyan,1),
        marker=(:circle,:white,2),
        markerstrokecolor=:darkcyan,
        markerstrokewidth=1)
    vline!([mm_dw[j][:z_max_smooth_spline_mm]],
        line=(:darkcyan,0.5), 
        label=L"$z_{\mathrm{max}}=%$(round(mm_dw[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
    plot!(legend=:outerbottom,
        background_legend_color = nothing,
        foreground_legend_color = :red,
        legend_columns = 2,
        legendfontsize =6,
        left_margin=3mm,
        right_margin=3mm,)
    display(fig)
end
gif_path = joinpath(OUTDIR, "CQD_$(Nss)_profiles_comparison.gif");
gif(anim, gif_path, fps=2)  # adjust fps 
@info "Saved GIF" gif_path ;
anim = nothing


# ATOMS PROPAGATION
r = 1:1:nI;
iter = (isempty(r) || last(r) == nI) ? r : Iterators.flatten((r, (nI,)));
# =========================
# Precompute geometry overlays (constant)
# =========================
x_magnet    = 1e-3 .* range(-1.0, 1.0, length=1000);  # m
z_edge_um   = 1e6 .* TheoreticalSimulation.z_magnet_edge.(x_magnet); # μm
x_magnet_mm = 1e3 .* x_magnet;                      # mm

# Aperture circle (drawn on the "aperture" panel)
xc_mm, zc_um = 0.0, 0.0;
R_mm = 1e3 * R_aper;
θ = range(0, 2π, length=361);
x_circ_mm = xc_mm .+ R_mm .* cos.(θ);
z_circ_um = zc_um .+ (1e3*R_mm) .* sin.(θ); # mm -> μm

# =========================
# Precompute stage distances (so per-particle you only divide by v0y)
# =========================
y_furn   = 0.0 ;
y_slit   = y_FurnaceToSlit ;
y_sg_in  = y_FurnaceToSlit + y_SlitToSG ;
y_sg_out = y_sg_in + y_SG ;
y_aper   = y_sg_out + y_SGToAperture ;
y_scr    = y_sg_out + y_SGToScreen ;

anim = @animate for j in iter
    data_set = CQD_up_screen[:data][j]
    n = size(data_set, 1)
    
    # --- preallocate arrays for histograms (store already scaled units) ---
    xs_a = Vector{Float64}(undef, n); zs_a = Vector{Float64}(undef, n)  # furnace (mm, μm)
    xs_b = Vector{Float64}(undef, n); zs_b = Vector{Float64}(undef, n)  # slit (mm, μm)
    xs_c = Vector{Float64}(undef, n); zs_c = Vector{Float64}(undef, n)  # SG in (mm, μm)
    xs_d = Vector{Float64}(undef, n); zs_d = Vector{Float64}(undef, n)  # SG out (mm, μm)
    xs_f = Vector{Float64}(undef, n); zs_f = Vector{Float64}(undef, n)  # aperture (mm, μm)
    xs_e = Vector{Float64}(undef, n); zs_e = Vector{Float64}(undef, n)  # screen (mm, mm)


    # --- one pass: compute all stages for each selected particle ---
    @inbounds for i in 1:n
        # initial conditions
        x0  = data_set[i,1]; y0  = data_set[i,2]; z0  = data_set[i,3]
        v0x = data_set[i,4]; v0y = data_set[i,5]; v0z = data_set[i,6]
        θe  = data_set[i,7]; ϕe  = data_set[i,8]

        # Furnace (just the initial plane)
        xs_a[i] = 1e3 * x0
        zs_a[i] = 1e6 * z0

        # Propagation times (divide once per stage)
        τ_slit  = y_slit / v0y
        τ_sgin  = y_sg_in / v0y
        τ_sgout = y_sg_out / v0y
        τ_aper  = y_aper / v0y
        τ_scr   = y_scr  / v0y

        r, _ = TheoreticalSimulation.CQD_EqOfMotion(τ_slit,  Icoils[j], μₑ, [x0,y0,z0], [v0x,v0y,v0z], θe, ϕe, ki, K39_params)
        xs_b[i] = 1e3 * r[1]
        zs_b[i] = 1e6 * r[3]

        r, _ = TheoreticalSimulation.CQD_EqOfMotion(τ_sgin,  Icoils[j], μₑ, [x0,y0,z0], [v0x,v0y,v0z], θe, ϕe, ki, K39_params)
        xs_c[i] = 1e3 * r[1]
        zs_c[i] = 1e6 * r[3]

        r, _ = TheoreticalSimulation.CQD_EqOfMotion(τ_sgout, Icoils[j], μₑ, [x0,y0,z0], [v0x,v0y,v0z], θe, ϕe, ki, K39_params)
        xs_d[i] = 1e3 * r[1]
        zs_d[i] = 1e6 * r[3]

        r, _ = TheoreticalSimulation.CQD_EqOfMotion(τ_aper,  Icoils[j], μₑ, [x0,y0,z0], [v0x,v0y,v0z], θe, ϕe, ki, K39_params)
        xs_f[i] = 1e3 * r[1]
        zs_f[i] = 1e6 * r[3]

        r, _ = TheoreticalSimulation.CQD_EqOfMotion(τ_scr,   Icoils[j], μₑ, [x0,y0,z0], [v0x,v0y,v0z], θe, ϕe, ki, K39_params)
        xs_e[i] = 1e3 * r[1]
        zs_e[i] = 1e3 * r[3]  # mm
    end


    bins_furn  = (FreedmanDiaconisBins(xs_a), FreedmanDiaconisBins(zs_a))
    bins_slit  = (FreedmanDiaconisBins(xs_b), FreedmanDiaconisBins(zs_b))
    bins_sgin  = (FreedmanDiaconisBins(xs_c), FreedmanDiaconisBins(zs_c))
    bins_sgout = (FreedmanDiaconisBins(xs_d), FreedmanDiaconisBins(zs_d))
    bins_aper  = (FreedmanDiaconisBins(xs_f), FreedmanDiaconisBins(zs_f))
    bins_scr   = (FreedmanDiaconisBins(xs_e), FreedmanDiaconisBins(zs_e))


    # --- Furnace panel ---
    figa = histogram2d(xs_a, zs_a;
        bins = bins_furn,
        show_empty_bins = true, color = :plasma, normalize = :pdf,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -1.0:0.25:1.0, yticks = -50:25:50,
    );
    # Text position
    xpos, ypos = -0.75, 35;
    # Draw a small white rectangle behind the text
    dx, dy = 0.15, 7 ;  # adjust width and height
    plot!(figa, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                  [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
      color=:white, opacity=0.65, linealpha=0,
      label=false);
    annotate!(figa, xpos, ypos,  text("Furnace", 10, :black, :bold, :center, "Helvetica") );

    # --- Slit panel ---
    figb = histogram2d(xs_b, zs_b;
        bins = bins_slit,
        show_empty_bins = true, color = :plasma, normalize = :pdf,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -4.0:0.50:4.0, yticks = -200:50:200,
        xlims = (-4, 4), ylims = (-200, 200),
    );
    # Text position
    xpos, ypos = -3.5, 150;
    # Draw a small white rectangle behind the text
    dx, dy = 0.4, 20;   # adjust width and height
    plot!(figb, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                  [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
      color=:white, opacity=0.65, linealpha=0,
      label=false);
    annotate!(figb, xpos, ypos,  text("Slit", 10, :black, :bold, :center, "Helvetica") );

    # --- SG entrance panel ---
    figc = histogram2d(xs_c, zs_c;
        bins = bins_sgin,
        show_empty_bins = true, color = :plasma, normalize = :pdf,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -4.0:0.50:4.0, yticks = -1000:100:1000,
        xlims = (-4, 4), ylims = (-250, 250),
    );
    # Text position
    xpos, ypos = -3.0, 180;
    # Draw a small white rectangle behind the text
    dx, dy = 0.8, 30;  # adjust width and height
    plot!(figc, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                  [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
      color=:white, opacity=0.65, linealpha=0,
      label=false);
    annotate!(figc, xpos, ypos,  text("SG entrance", 10, :black, :bold, :center, "Helvetica") );

    # --- SG exit panel ---
    figd = histogram2d(xs_d, zs_d;
        bins = bins_sgout,
        show_empty_bins = true, color = :plasma, normalize = :pdf,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -4.0:0.50:4.0, yticks = -1000:200:1000,
        xlims = (-4, 4), ylims = (-300, 1000),
    );
    plot!(figd, x_magnet_mm, z_edge_um, line = (:dash, :black, 2), label=false)
    # Text position
    xpos, ypos = -3.0, 700;
    # Draw a small white rectangle behind the text
    dx, dy = 0.6, 160 ;  # adjust width and height
    plot!(figd, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                  [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
      color=:white, opacity=0.65, linealpha=0,
      label=false);
    annotate!(figd, xpos, ypos,  text("SG exit", 10, :black, :bold, :center, "Helvetica") );

    # --- Aperture panel ---
    figf = histogram2d(xs_f, zs_f;
        bins = bins_aper,
        show_empty_bins = true, color = :plasma, normalize = :pdf,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -4.0:0.50:4.0, yticks = -1000:500:3000,
        xlims = (-4, 4), ylims = (-300, 3000),
    );
    plot!(figf, x_circ_mm, z_circ_um; linestyle=:dash, lw=2, color=:gray, legend=false);
    # Text position
    xpos, ypos = -3.0, 2400;
    # Draw a small white rectangle behind the text
    dx, dy = 0.7, 270;   # adjust width and height
    plot!(figf, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                  [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
      color=:white, opacity=0.65, linealpha=0,
      label=false);
    annotate!(figf, xpos, ypos,  text("⊚ Aperture", 10, :black, :bold, :center, "Helvetica") );

    # Screen
    # --- Screen panel ---
    fige = histogram2d(xs_e, zs_e;
        bins = bins_scr,
        show_empty_bins = true, color = :plasma, normalize = :pdf,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{mm})$",
        ylims = (-1, 17.5),
    );
    # Text position
    xpos, ypos = -4.0, 14;
    # Draw a small white rectangle behind the text
    dx, dy = 0.9, 0.9;   # adjust width and height
    plot!(fige, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                  [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
      color=:white, opacity=0.65, linealpha=0,
      label=false);
    annotate!(fige, xpos, ypos,  text("Screen", 10, :black, :bold, :center, "Helvetica") );

    fig = plot(figa,figb,figc,figd,figf,fige,
    layout=(6,1),
    suptitle = L"$I_{0} = %$(Int(1000*Icoils[j]))\,\mathrm{mA}$",
    size=(750,800),
    right_margin=2mm,
    bottom_margin=-2mm,
    );
    plot!(fig[1], xlabel="", bottom_margin=-3mm);
    plot!(fig[2], xlabel="", bottom_margin=-3mm);
    plot!(fig[3], xlabel="", bottom_margin=-3mm);
    plot!(fig[4], xlabel="", bottom_margin=-3mm);
    plot!(fig[5], xlabel="", bottom_margin=-3mm);
    display(fig)
end
gif_path = joinpath(OUTDIR, "CQD_time_evolution.gif");
gif(anim, gif_path, fps=2)  # adjust fps
@info "Saved GIF" gif_path ;
anim = nothing

fig = plot(xlabel=L"$I_{c}$ (A)", ylabel=L"$z_{\mathrm{max}}$ (mm)") 
plot!(I_exp[2:end],z_exp[2:end],
    ribbon=δz_exp[5:end],
    label="Experiment (combined)",
    line=(:black,:dash,2),
    fillalpha=0.23, 
    fillcolor=:black, 
    )
Isim_start_idx = findall(>=(0.010), Icoils)[1]
plot!(fig,Icoils[Isim_start_idx:end], [mm_up[v][:z_max_smooth_spline_mm] for v in 1:nI][Isim_start_idx:end],
    label=L"CQD: $k_{i}=%$(1e6*ki)\times 10^{-6}$",
    line=(:solid,:red,2))
plot!(fig,xaxis=:log10,
    yaxis=:log10,
    xlims=(8e-3,2),
    ylims=(8e-3,2),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], 
            [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:topleft,
    left_margin =2mm,
)
display(fig)
savefig(fig, joinpath(OUTDIR,"CQD_results_comparison.$FIG_EXT"))


kis = round.([
    [exp10(p) * x for p in -8:-8 for x in 1.0:1:9]; 
    [exp10(p) * x for p in -7:-7 for x in 1.0:1:9]; 
    [exp10(p) * x for p in -6:-6 for x in 1.0:0.1:9.9]; 
    ## exp10(-5) * (1:0.1:10);
    exp10.(-5:-1)
];sigdigits=4)
@info "Number of ki sampled" length(kis)

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

dta_ki_up = zeros(length(kis),length(Icoils));
dta_ki_dw = zeros(length(kis),length(Icoils));
isdir(joinpath(OUTDIR,"up")) || mkpath(joinpath(OUTDIR,"up"));
isdir(joinpath(OUTDIR,"dw")) || mkpath(joinpath(OUTDIR,"dw"));
for (i,ki) in enumerate(kis)
    @info "Running for kᵢ = $(round(1e6*ki,sigdigits=3))×10⁻⁶"

    @time temp_CQD_up_particles_flag         = TheoreticalSimulation.CQD_flag_travelling_particles(Icoils, data_UP, ki, K39_params; y_length=5001,verbose=true);
    @time temp_CQD_up_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles(Icoils, ki, data_UP, temp_CQD_up_particles_flag, K39_params)      # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]
    @time temp_CQD_dw_particles_flag         = TheoreticalSimulation.CQD_flag_travelling_particles(Icoils, data_DOWN, ki, K39_params; y_length=5001,verbose=true);
    @time temp_CQD_dw_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles(Icoils, ki, data_DOWN, temp_CQD_dw_particles_flag, K39_params);   # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]

    TheoreticalSimulation.CQD_travelling_particles_summary(Icoils,temp_CQD_up_particles_trajectories, :up)
    TheoreticalSimulation.CQD_travelling_particles_summary(Icoils,temp_CQD_dw_particles_trajectories, :down)

    temp_CQD_up_screen = OrderedDict(:Icoils=>Icoils, :data => TheoreticalSimulation.CQD_select_flagged(temp_CQD_up_particles_trajectories,:screen ))
    temp_CQD_dw_screen = OrderedDict(:Icoils=>Icoils, :data => TheoreticalSimulation.CQD_select_flagged(temp_CQD_dw_particles_trajectories,:screen ))

    jldsave(joinpath(OUTDIR,"up","cqd_$(Nss)_ki$(@sprintf("%03d", i))_up_screen.jld2"), screen=temp_CQD_up_screen)
    jldsave(joinpath(OUTDIR,"dw","cqd_$(Nss)_ki$(@sprintf("%03d", i))_dw_screen.jld2"), screen=temp_CQD_dw_screen )

    temp_mm_up = TheoreticalSimulation.CQD_analyze_profiles_to_dict(temp_CQD_up_screen;
        n_bins = (nx_bins , nz_bins), width_mm = gaussian_width_mm, 
        add_plot = false, plot_xrange= :all, branch=:up,
        λ_raw = λ0_raw, λ_smooth = λ0_spline, mode = :probability)

    temp_mm_dw = TheoreticalSimulation.CQD_analyze_profiles_to_dict(temp_CQD_dw_screen;
        n_bins = (nx_bins , nz_bins), width_mm = gaussian_width_mm, 
        add_plot = false, plot_xrange= :all, branch=:dw,
        λ_raw = λ0_raw, λ_smooth = λ0_spline, mode = :probability)

    dta_ki_up[i,:] = [temp_mm_up[v][:z_max_smooth_spline_mm] for v in 1:nI][1:end]
    dta_ki_dw[i,:] = [temp_mm_dw[v][:z_max_smooth_spline_mm] for v in 1:nI][1:end]

    temp_CQD_up_particles_flag         = nothing
    temp_CQD_up_particles_trajectories = nothing
    temp_CQD_dw_particles_flag         = nothing
    temp_CQD_dw_particles_trajectories = nothing
    temp_CQD_up_screen = nothing
    temp_CQD_dw_screen = nothing
    temp_mm_up = nothing
    temp_mm_dw = nothing
    GC.gc()
end
jldsave( joinpath(OUTDIR, "cqd_$(Nss)_kis.jld2"), 
        data=OrderedDict(
            :Icoils     => Icoils,
            :ki         => kis,
            :nz_bins    => nz_bins,
            :gauss_w    => gaussian_width_mm,
            :smothing   => (λ0_raw,λ0_spline),
            :up         => dta_ki_up,
            :dw         => dta_ki_dw)
)


cls = palette(:darkrainbow, length(kis))
fig = plot(xlabel=L"$I_{c}$ (A)", ylabel=L"$z_{\mathrm{max}}$ (mm)")
plot!(I_exp[2:end],z_exp[2:end],
    ribbon=δz_exp[5:end],
    label="Experiment (combined)",
    line=(:black,:dash,2),
    fillalpha=0.23, 
    fillcolor=:black, 
    ) 
Isim_start_idx = findall(>=(0.010), Icoils)[1]   
for i in eachindex(kis)
    plot!(fig, Icoils[Isim_start_idx:end], abs.(dta_ki_up[i,Isim_start_idx:end]),
    label=L"$k_{i} = %$(round(1e6*kis[i], sigdigits=2))\times 10^{-6}$",
    line=(cls[i],1))
end
plot!(fig, 
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
    foreground_color_legend=nothing)
display(fig)
savefig(fig, joinpath(OUTDIR,"cqd_$(Nss)_kis_comparison.$(FIG_EXT)"))


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
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm

SETUP FEATURES
    Temperature             : $(T_K)
    Furnace aperture (x,z)  : ($(1e3*x_furnace)mm , $(1e6*z_furnace)μm)
    Slit (x,z)              : ($(1e3*x_slit)mm , $(1e6*z_slit)μm)
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
    Currents (A)            : $(round.(Icoils,sigdigits=3))
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
# =======================
# Global parameters
# =======================
Ns = 6_000_000

kis = round.([
    [exp10(p) * x for p in -8:-8 for x in 1.0:1:9]; 
    [exp10(p) * x for p in -7:-7 for x in 1.0:1:9]; 
    [exp10(p) * x for p in -6:-6 for x in 1.0:0.1:9.9]; 
    ## exp10(-5) * (1:0.1:10);
    exp10.(-5:-1)
];sigdigits=4)
@info "Number of ki sampled" length(kis)


induction_coeff     = 1e6 .* kis
nx_bins             = 64 # fixed nx bins
nz_bins             = [1, 2, 4]
gaussian_width_mm   = [0.001, 0.010, 0.065, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500]; # try different gaussian widths
λ0_raw_list         = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10]; # try different smoothing factors for raw data
λ0_spline           = 0.001

# Total combinations (diagnostic only)
Ntot = length(nz_bins) * length(gaussian_width_mm) * length(λ0_raw_list)
@info "Total combinations : Nnz × Nσ × Nλ0 × Nλs " Ntot

# ---------- precompute param grid ----------
params = [(nz, gw, λ0_raw)
          for nz in nz_bins
          for gw in gaussian_width_mm
          for λ0_raw in λ0_raw_list]

# ---------- helpers for hierarchical keys ----------
fmt(x) = @sprintf("%.12g", x)  # safer than %.6g to reduce collisions

function keypath(branch::Symbol, ki::Float64, nz::Int, gw::Float64, λ0_raw::Float64)
    return "/" * String(branch) *
           "/ki=" * fmt(ki) *"e-6" *
           "/nz=" * string(nz) *
           "/gw=" * fmt(gw) *
           "/lam=" * fmt(λ0_raw)
end

# JLD2 doesn't overwrite datasets by default; delete first.
function jld_overwrite!(f, path::AbstractString, value)
    if haskey(f, path)
        delete!(f, path)
    end
    f[path] = value
    return nothing
end

# =========================================================
# ======================== UP =============================
# =========================================================
# const INDIR_up = joinpath(OUTDIR,"up")
# const INDIR_up = joinpath("Z:\\SingleSternGerlachExperimentData\\simulation_data\\cqd_simulation_6M","up")
const INDIR_up = joinpath("/Volumes/My Passport/SternGerlach/cqd_simulation_6M","up")
# const INDIR_up = joinpath(dirname(OUTDIR),"cqd","up")
# --- Files ---
files = sort(filter(f -> isfile(joinpath(INDIR_up, f)) && endswith(f, ".jld2"),
               readdir(INDIR_up)))
ki_initial, ki_final = 1, 113
# files = [
#     @sprintf("cqd_%6d_ki%03d_up_screen.jld2", Ns, ki)
#     for ki in ki_initial:ki_final
# ]
nfiles = length(files)
@assert nfiles == length(induction_coeff) "Mismatch: files vs induction_coeff"

# Total combinations
Ntot = nfiles * length(nz_bins) * length(gaussian_width_mm) * length(λ0_raw_list)
@info "Total profiles to compute : Nfiles × Nnz × Nσ × Nλ0 × Nλs " Ntot

# ---------- output file + metadata ----------
outjld = joinpath(OUTDIR, "cqd_$(Ns)_up_profiles_$(ki_initial)_$(ki_final)_bykey.jld2")

jldopen(outjld, "w") do f
    f["/meta/nz_bins"]           = nz_bins
    f["/meta/gaussian_width_mm"] = gaussian_width_mm
    f["/meta/λ0_raw_list"]       = λ0_raw_list
    f["/meta/λ0_spline"]         = λ0_spline
    f["/meta/induction_coeff"]   = kis
    f["/meta/files"]             = files
end

# ---------- main loop ----------
@time for (j, fname) in pairs(files)
    ki      = round(induction_coeff[j]; sigdigits=3)
    simpath = joinpath(INDIR_up, fname)

    @info "Loading file $(j)/$(length(files))" fname=fname ki=ki
    data_sim = load(simpath, "screen")

    ## Allocate thread-local buffers safely
    nt = Threads.maxthreadid()

    # Safer to store Any as the value type
    local_chunks = Vector{Vector{Pair{Tuple{Float64,Int,Float64,Float64}, Any}}}(undef, nt)
    for t in 1:nt
        local_chunks[t] = Pair{Tuple{Float64,Int,Float64,Float64}, Any}[]
    end

    # ---- threaded compute ----
    @time begin
    @threads for pidx in eachindex(params)
        nz, gw, λ0_raw = params[pidx]

        profiles_up = TheoreticalSimulation.CQD_analyze_profiles_to_dict(
            data_sim;
            n_bins      = (nx_bins, nz),
            width_mm    = gw,
            add_plot    = false,
            plot_xrange = :all,
            branch      = :up,
            λ_raw       = λ0_raw,
            λ_smooth    = λ0_spline,
            mode        = :probability
        )

        push!(local_chunks[threadid()], ((ki, nz, gw, λ0_raw) => profiles_up))
    end

    # ---- serial I/O ONCE per file ----
    jldopen(outjld, "r+") do f
        for t in 1:nt
            for pair in local_chunks[t]
                k = pair.first
                profiles_up = pair.second

                ki2, nz, gw, λ0_raw = k
                label_path = keypath(:up, ki2, nz, gw, λ0_raw)

                # resume-friendly
                if !haskey(f, label_path)
                    f[label_path] = profiles_up
                end
            end
        end

    # progress markers (overwrite safely)
    jld_overwrite!(f, "/meta/progress/last_completed_j", j)
    jld_overwrite!(f, "/meta/progress/last_completed_file", fname)
    jld_overwrite!(f, "/meta/progress/last_completed_ki", ki)
    end
    end

    data_sim = nothing
    GC.gc()
    @info "Done file $(j)/$(length(files))" free_GiB = round(Sys.free_memory() / 1024^3,digits=3)
end
@info "Completed UP table"

# =========================================================
# ======================== DW =============================
# =========================================================
# const INDIR_dw = joinpath(OUTDIR,"dw")
# const INDIR_dw = joinpath("Z:\\SingleSternGerlachExperimentData\\simulation_data\\cqd_simulation_6M","dw")
# const INDIR_dw = joinpath("/Volumes/My Passport/SternGerlach/cqd_simulation_6M","dw")
const INDIR_dw = joinpath(dirname(OUTDIR),"cqd","dw")
# --- Files ---
files = sort(filter(f -> isfile(joinpath(INDIR_dw, f)) && endswith(f, ".jld2"),
               readdir(INDIR_dw)))
ki_initial, ki_final = 1, 60
files = [
    @sprintf("cqd_%6d_ki%03d_dw_screen.jld2",Ns, ki)
    for ki in ki_initial:ki_final
]
nfiles = length(files)
# @assert nfiles == length(induction_coeff) "Mismatch: files vs induction_coeff"

# Total combinations
Ntot = nfiles * length(nz_bins) * length(gaussian_width_mm) * length(λ0_raw_list)
@info "Total profiles to compute : Nfiles × Nnz × Nσ × Nλ0 × Nλs " Ntot

# ---------- output file + metadata ----------
outjld = joinpath(OUTDIR, "cqd_$(Ns)_dw_profiles_$(ki_initial)_$(ki_final)_bykey.jld2")

jldopen(outjld, "w") do f
    f["/meta/nz_bins"]           = nz_bins
    f["/meta/gaussian_width_mm"] = gaussian_width_mm
    f["/meta/λ0_raw_list"]       = λ0_raw_list
    f["/meta/λ0_spline"]         = λ0_spline
    f["/meta/induction_coeff"]   = kis
    f["/meta/files"]             = files
end

# ---------- main loop ----------
@time for (j, fname) in pairs(files)
    ki      = round(induction_coeff[j]; sigdigits=3)
    simpath = joinpath(INDIR_dw, fname)

    @info "Loading file $(j)/$(length(files))" fname=fname ki=ki
    data_sim = load(simpath, "screen")

    ## Allocate thread-local buffers safely
    nt = Threads.maxthreadid()

    # Safer to store Any as the value type
    local_chunks = Vector{Vector{Pair{Tuple{Float64,Int,Float64,Float64}, Any}}}(undef, nt)
    for t in 1:nt
        local_chunks[t] = Pair{Tuple{Float64,Int,Float64,Float64}, Any}[]
    end

    # ---- threaded compute ----
    @threads for pidx in eachindex(params)
        nz, gw, λ0_raw = params[pidx]

        profiles_dw = TheoreticalSimulation.CQD_analyze_profiles_to_dict(
            data_sim;
            n_bins      = (nx_bins, nz),
            width_mm    = gw,
            add_plot    = false,
            plot_xrange = :all,
            branch      = :dw,
            λ_raw       = λ0_raw,
            λ_smooth    = λ0_spline,
            mode        = :probability
        )

        push!(local_chunks[threadid()], ((ki, nz, gw, λ0_raw) => profiles_dw))
    end

    # ---- serial I/O ONCE per file ----
    jldopen(outjld, "r+") do f
        for t in 1:nt
            for pair in local_chunks[t]
                k = pair.first
                profiles_dw = pair.second

                ki2, nz, gw, λ0_raw = k
                label_path = keypath(:dw, ki2, nz, gw, λ0_raw)

                # resume-friendly
                if !haskey(f, label_path)
                    f[label_path] = profiles_dw
                end
            end
        end

    # progress markers (overwrite safely)
    jld_overwrite!(f, "/meta/progress/last_completed_j", j)
    jld_overwrite!(f, "/meta/progress/last_completed_file", fname)
    jld_overwrite!(f, "/meta/progress/last_completed_ki", ki)
    end

    data_sim = nothing
    GC.gc()
    @info "Done file $(j)/$(length(files))" free_GiB = round(Sys.free_memory() / 1024^3,digits=3)
end
@info "Completed DOWN table"

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
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm

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

SIMULATION INFORMATION
    Number of atoms         : $(Ns)
    Induction term          : ($(ki_initial),$(ki_final)) = ($(kis[ki_initial]), $(kis[ki_final]))
    Binning (nx,nz)         : ($(nx_bins),$(nz_bins))
    Gaussian width (mm)     : $(gaussian_width_mm)
    Smoothing raw           : $(λ0_raw_list)
    Smoothing spline        : $(λ0_spline)
    Currents (A)            : $(round.(Icoils,sigdigits=3))
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


# using JLD2

# # --- helper: delete+write (since JLD2 won't overwrite datasets by default) ---
# function jld_overwrite!(f, path::AbstractString, value)
#     if haskey(f, path)
#         delete!(f, path)
#     end
#     f[path] = value
#     return nothing
# end

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
