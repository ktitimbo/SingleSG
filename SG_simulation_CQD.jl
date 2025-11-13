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
LinearAlgebra.BLAS.set_num_threads(4)
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
# Slit
const x_slit  = 4.0e-3 ;
const z_slit  = 300e-6 ;
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
    Furnace → Slit          : $(1e3*y_FurnaceToSlit)mm
    Slit → SG magnet        : $(1e3*y_SlitToSG)mm
    SG magnet               : $(1e3*y_SG)mm
    SG magnet → Screen      : $(1e3*y_SGToScreen)mm
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
const Nss = 2_800_000 ; 
@info "Number of MonteCarlo particles : $(Nss)\n"

nx_bins , nz_bins = 32 , 2
gaussian_width_mm = 0.150
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
# ki = 0.75e-6;
# @time CQD_up_particles_flag         = TheoreticalSimulation.CQD_flag_travelling_particles(Icoils, data_UP, ki, K39_params; y_length=5001,verbose=true);
# @time CQD_up_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles(Icoils, ki, data_UP, CQD_up_particles_flag, K39_params);     # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]
# @time CQD_dw_particles_flag         = TheoreticalSimulation.CQD_flag_travelling_particles(Icoils, data_DOWN, ki, K39_params; y_length=5001,verbose=true);
# @time CQD_dw_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles(Icoils, ki, data_DOWN, CQD_dw_particles_flag, K39_params);   # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]

# TheoreticalSimulation.CQD_travelling_particles_summary(Icoils,CQD_up_particles_trajectories, :up)
# TheoreticalSimulation.CQD_travelling_particles_summary(Icoils,CQD_dw_particles_trajectories, :down)

# CQD_up_screen = OrderedDict(:Icoils=>Icoils, :data => TheoreticalSimulation.CQD_select_flagged(CQD_up_particles_trajectories,:screen ))
# CQD_dw_screen = OrderedDict(:Icoils=>Icoils, :data => TheoreticalSimulation.CQD_select_flagged(CQD_dw_particles_trajectories,:screen ))
# jldsave(joinpath(OUTDIR,"cqd_$(Nss)_up_screen.jld2"), screen = CQD_up_screen )
# jldsave(joinpath(OUTDIR,"cqd_$(Nss)_dw_screen.jld2"), screen = CQD_dw_screen )

# mm_up = TheoreticalSimulation.CQD_analyze_profiles_to_dict(CQD_up_screen;
#     n_bins = (nx_bins , nz_bins), width_mm = gaussian_width_mm, 
#     add_plot = false, plot_xrange= :all, branch=:up,
#     λ_raw = λ0_raw, λ_smooth = λ0_spline, mode = :probability);

# mm_dw = TheoreticalSimulation.CQD_analyze_profiles_to_dict(CQD_dw_screen;
#     n_bins = (nx_bins , nz_bins), width_mm = gaussian_width_mm, 
#     add_plot = false, plot_xrange= :all, branch=:dw,
#     λ_raw = λ0_raw, λ_smooth = λ0_spline, mode = :probability);

# jldsave(
#     joinpath(OUTDIR,"cqd_$(Nss)_screen_profiles.jld2"), 
#     profile = OrderedDict(
#     :nz_bins    => nz_bins,
#     :gauss_w    => gaussian_width_mm,
#     :smothing   => (λ0_raw,λ0_spline),
#     :ki         => ki,
#     :mmup       => mm_up,
#     :mmdw       => mm_dw
#     ) 
# )
                                                                 
# # Profiles : up and down
# anim = @animate for j in eachindex(Icoils)
#     fig = plot(
#         title=L"CQD profiles : $k_i = %$(round(1e6*ki, sigdigits=2))\times 10^{-6}$",
#         legend=:topleft,
#         legendtitle=L"$I_{0}=%$(Icoils[j])\mathrm{A}$",
#         legendtitlefontsize=8,
#         yformatter = val -> string(round(val * 1e4, digits = 2)),
#         xlabel=L"$z$ (mm)",
#         ylabel="Intensity (au)",)
#     plot!(mm_up[j][:z_profile][:,1],mm_up[j][:z_profile][:,3],
#         label=L"$\vec{\mu}\upuparrows \hat{z}$",
#         line=(:solid,:orangered2,1),
#         marker=(:circle,:white,2),
#         markerstrokecolor=:orangered2,
#         markerstrokewidth=1)
#     vline!([mm_up[j][:z_max_smooth_spline_mm]], 
#         line=(:orangered2,0.5), 
#         label=L"$z_{\mathrm{max}}=%$(round(mm_up[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
#     plot!(mm_dw[j][:z_profile][:,1],mm_dw[j][:z_profile][:,3],
#         label=L"$\vec{\mu}\updownarrows \hat{z}$",
#         line=(:solid,:dodgerblue3,1),
#         marker=(:circle,:white,2),
#         markerstrokecolor=:dodgerblue3,
#         markerstrokewidth=1)
#     vline!([mm_dw[j][:z_max_smooth_spline_mm]],
#         line=(:dodgerblue3,0.5), 
#         label=L"$z_{\mathrm{max}}=%$(round(mm_dw[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
#     display(fig)
# end
# gif_path = joinpath(OUTDIR, "CQD_profiles.gif");
# gif(anim, gif_path, fps=2)  # adjust fps 
# @info "Saved GIF" gif_path ;
# anim = nothing

# # Profiles COMPARISON : different contributions 
# anim = @animate for j in 1:4:length(Icoils)
#     isodd(j) || continue        # keep only every other
#     fig = plot(title=L"$I_{0}=%$(Int(1e3*Icoils[j]))\mathrm{mA}$",
#         xlabel=L"$z$ (mm)",
#         ylabel="Intensity (au)",
#         yformatter = val -> string(round(val * 1e4, digits = 2)),)
#     plot!(mm_up[j][:z_profile][:,1],mm_up[j][:z_profile][:,3],
#         label=L"CQD UP $k_{i}=%$(round(1e6*ki,sigdigits=2))\times 10^{-6}$",
#         line=(:solid,:maroon3,1),
#         marker=(:circle,:white,2),
#         markerstrokecolor=:maroon3,
#         markerstrokewidth=1)
#     vline!([mm_up[j][:z_max_smooth_spline_mm]],
#         line=(:maroon3,0.5), 
#         label=L"$z_{\mathrm{max}}=%$(round(mm_up[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
#     plot!(mm_dw[j][:z_profile][:,1],mm_dw[j][:z_profile][:,3],
#         label=L"CQD DOWN $k_{i}=%$(round(1e6*ki,sigdigits=2))\times 10^{-6}$",
#         line=(:solid,:darkcyan,1),
#         marker=(:circle,:white,2),
#         markerstrokecolor=:darkcyan,
#         markerstrokewidth=1)
#     vline!([mm_dw[j][:z_max_smooth_spline_mm]],
#         line=(:darkcyan,0.5), 
#         label=L"$z_{\mathrm{max}}=%$(round(mm_dw[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
#     plot!(legend=:outerbottom,
#         background_legend_color = nothing,
#         foreground_legend_color = :red,
#         legend_columns = 2,
#         legendfontsize =6,
#         left_margin=3mm,
#         right_margin=3mm,)
#     display(fig)
# end
# gif_path = joinpath(OUTDIR, "QM_CQD_$(Nss)_profiles_comparison.gif");
# gif(anim, gif_path, fps=2)  # adjust fps 
# @info "Saved GIF" gif_path ;
# anim = nothing


# # ATOMS PROPAGATION
# r = 1:4:nI;
# iter = (isempty(r) || last(r) == nI) ? r : Iterators.flatten((r, (nI,)));
# anim = @animate for j in iter
#     data_set = CQD_up_screen[:data][j]
    
#     #Furnace
#     xs_a = 1e3 .* data_set[:,1]; # mm
#     zs_a = 1e6 .* data_set[:,3]; # μm
#     figa = histogram2d(xs_a, zs_a;
#         bins = (FreedmanDiaconisBins(xs_a), FreedmanDiaconisBins(zs_a)),
#         show_empty_bins = true, color = :plasma, normalize=:pdf,
#         xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
#         xticks = -1.0:0.25:1.0, yticks = -50:25:50,
#     );

#     # Slit
#     r_at_slit = Matrix{Float64}(undef, size(data_set, 1), 3);
#     for i in axes(data_set,1)
#         v0y = data_set[i,5]
#         r , _ = TheoreticalSimulation.CQD_EqOfMotion(y_FurnaceToSlit ./ v0y, Icoils[j], μₑ, data_set[i,1:3], data_set[i,4:6], data_set[i,7], data_set[i,8], ki, K39_params)
#         r_at_slit[i,:] = r
#     end
#     xs_b = 1e3 .* r_at_slit[:,1]; # mm
#     zs_b = 1e6 .* r_at_slit[:,3]; # μm
#     figb = histogram2d(xs_b, zs_b;
#         bins = (FreedmanDiaconisBins(xs_b), FreedmanDiaconisBins(zs_b)),
#         show_empty_bins = true, color = :plasma, normalize=:pdf,
#         xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
#         xticks = -4.0:0.50:4.0, yticks = -200:50:200,
#         xlims=(-4,4),
#         ylims=(-200,200),
#     ) ;

#     # SG entrance
#     r_at_SG_entrance = Matrix{Float64}(undef, size(data_set, 1), 3);
#     for i in axes(data_set,1)
#         v0y = data_set[i,5]
#         r , _ = TheoreticalSimulation.CQD_EqOfMotion((y_FurnaceToSlit+y_SlitToSG) ./ v0y , Icoils[j], μₑ, data_set[i,1:3], data_set[i,4:6], data_set[i,7], data_set[i,8], ki, K39_params)
#         r_at_SG_entrance[i,:] = r
#     end
#     xs_c = 1e3 .* r_at_SG_entrance[:,1]; # mm
#     zs_c = 1e6 .* r_at_SG_entrance[:,3]; # μm
#     figc = histogram2d(xs_c, zs_c;
#         bins = (FreedmanDiaconisBins(xs_c), FreedmanDiaconisBins(zs_c)),
#         show_empty_bins = true, color = :plasma, normalize=:pdf,
#         xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
#         xticks = -4.0:0.50:4.0, yticks = -1000:100:1000,
#         xlims=(-4,4), ylims=(-250,250),
#     );

#     # SG exit
#     r_at_SG_exit = Matrix{Float64}(undef, size(data_set, 1), 3);
#     for i in axes(data_set,1)
#         v0y = data_set[i,5]
#         r , _ = TheoreticalSimulation.CQD_EqOfMotion((y_FurnaceToSlit+y_SlitToSG+y_SlitToSG) ./ v0y, Icoils[j], μₑ, data_set[i,1:3], data_set[i,4:6], data_set[i,7], data_set[i,8], ki, K39_params)
#         r_at_SG_exit[i,:] = r
#     end
#     xs_d = 1e3 .* r_at_SG_exit[:,1]; # mm
#     zs_d = 1e6 .* r_at_SG_exit[:,3]; # μm
#     figd = histogram2d(xs_d, zs_d;
#         bins = (FreedmanDiaconisBins(xs_d), FreedmanDiaconisBins(zs_d)),
#         show_empty_bins = true, color = :plasma, normalize=:pdf,
#         xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
#         xticks = -4.0:0.50:4.0, yticks = -1000:200:1000,
#         xlims=(-4,4), ylims=(-300,1000),
#     )
#     x_magnet = 1e-3*range(-1.0,1.0,length=1000)
#     plot!(figd,1e3*x_magnet,1e6*TheoreticalSimulation.z_magnet_edge.(x_magnet),line=(:dash,:black,2),label=false)

#     # Screen
#     r_at_screen = Matrix{Float64}(undef, size(data_set, 1), 3);
#     for i in axes(data_set,1)
#         v0y = data_set[i,5]
#         r , _ = TheoreticalSimulation.CQD_EqOfMotion((y_FurnaceToSlit+y_SlitToSG+y_SlitToSG+y_SGToScreen) ./ v0y, Icoils[j], μₑ, data_set[i,1:3], data_set[i,4:6], data_set[i,7], data_set[i,8], ki, K39_params)
#         r_at_screen[i,:] = r
#     end
#     xs_e = 1e3 .* r_at_screen[:,1]; # mm
#     zs_e = 1e3 .* r_at_screen[:,3]; # μm
#     fige = histogram2d(xs_e, zs_e;
#         bins = (FreedmanDiaconisBins(xs_e), FreedmanDiaconisBins(zs_e)),
#         show_empty_bins = true, color = :plasma, normalize=:pdf,
#         xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{mm})$",
#         ylims=(-1,17.5),
#         # xticks = -4.0:0.50:4.0, yticks = -1250:50:1250,
#     );

#     fig = plot(figa,figb,figc,figd,fige,
#     layout=(5,1),
#     suptitle = L"$I_{0} = %$(Int(1000*Icoils[j]))\,\mathrm{mA}$",
#     size=(750,800),
#     right_margin=2mm,
#     bottom_margin=-2mm,
#     )
#     plot!(fig[1], xlabel="", bottom_margin=-3mm),
#     plot!(fig[2], xlabel="", bottom_margin=-3mm),
#     plot!(fig[3], xlabel="", bottom_margin=-3mm),
#     plot!(fig[4], xlabel="", bottom_margin=-3mm),
#     display(fig)
# end
# gif_path = joinpath(OUTDIR, "CQD_time_evolution.gif");
# gif(anim, gif_path, fps=2)  # adjust fps
# @info "Saved GIF" gif_path ;
# anim = nothing


# fig = plot(xlabel=L"$I_{c}$ (A)", ylabel=L"$z_{\mathrm{max}}$ (mm)") 
# plot!(I_exp[2:end],z_exp[2:end],
#     ribbon=δz_exp[5:end],
#     label="Experiment (combined)",
#     line=(:black,:dash,2),
#     fillalpha=0.23, 
#     fillcolor=:black, 
#     )
# Isim_start_idx = findall(>=(0.010), Icoils)[1]
# plot!(fig,Icoils[Isim_start_idx:end], [mm_up[v][:z_max_smooth_spline_mm] for v in 1:nI][Isim_start_idx:end],
#     label=L"CQD: $k_{i}=%$(1e6*ki)\times 10^{-6}$",
#     line=(:solid,:red,2))
# plot!(fig,xaxis=:log10,
#     yaxis=:log10,
#     xlims=(8e-3,2),
#     ylims=(8e-3,2),
#     xticks = ([1e-3, 1e-2, 1e-1, 1.0], 
#             [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
#     yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
#             [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
#     legend=:topleft,
#     left_margin =2mm,
# )
# display(fig)
# savefig(fig, joinpath(OUTDIR,"CQD_results_comparison.$FIG_EXT"))


kis = vcat(
    collect(1e-7*range(0.1,1.0, length=10)),
    collect(1e-6*range(0.1,1.0, length=10)),
    collect(range(1.1,2.0, length=10)/1e6),
    collect(range(2.1,3.0, length=10)/1e6),
    collect(range(3.1,4.0, length=10)/1e6),
    collect(range(4.1,5.0, length=10)/1e6),
    collect(range(5.1,6.0, length=10)/1e6),
    collect(range(10,100, length=10)/1e6),
    )

dta_ki_up = zeros(length(kis),length(Icoils));
dta_ki_dw = zeros(length(kis),length(Icoils));
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

    jldsave(joinpath(OUTDIR,"cqd_$(Nss)_ki$(@sprintf("%02d", i))_up_screen.jld2"), screen=temp_CQD_up_screen)
    jldsave(joinpath(OUTDIR,"cqd_$(Nss)_ki$(@sprintf("%02d", i))_dw_screen.jld2"), screen=temp_CQD_dw_screen )

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
    legend=:bottomright,
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


println("script $RUN_STAMP has finished!")
alert("script $RUN_STAMP has finished!")

