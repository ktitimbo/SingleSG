# Simulation of atom trajectories in the Stern–Gerlach experiment
# Kelvin Titimbo
# California Institute of Technology
# August 2025

#  Plotting Setup
# ENV["GKS_WSTYPE"] = "101"
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
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
const OUTDIR    = joinpath(@__DIR__, "simulation_data", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
# Random seeds
base_seed_set = 145;
# rng_set = MersenneTwister(base_seed_set)
rng_set = TaskLocalRNG();
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
K39_params = AtomParams(atom);
# const R         = atom_info[1];
# const μₙ        = atom_info[2];
# const γₙ        = atom_info[3];
# const Ispin    = atom_info[4];
# const Ahfs     = atom_info[6];
# const M        = atom_info[7];
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
    Furnace aperture (x,z)  : ($(1e3*x_furnace)μm , $(1e6*z_furnace)μm)
    Slit (x,z)              : ($(1e3*x_slit)μm , $(1e6*z_slit)μm)
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

# Coil currents
Icoils = [0.00,
            0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,0.090,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.75,0.80,0.90,1.00
];
# Icoils = reverse([0.993, 0.739, 0.549, 0.01164, 0.0])
nI = length(Icoils);

# Sample size: number of atoms arriving to the screen
const Nss = 100_000
@info "Number of MonteCarlo particles : $(Nss)\n"

# Monte Carlo generation of particles traersing the filtering slit
crossing_slit = generate_samples(Nss, effusion_params; v_pdf=:v3, rng = rng_set, multithreaded = false, base_seed = base_seed_set);
# pairs_UP, pairs_DOWN = build_initial_conditions(Nss, crossing_slit, rng_set; mode=:bucket);

if SAVE_FIG
    plot_μeff(K39_params,"mm_effective")
    plot_SG_geometry("SG_geometry")
    plot_velocity_stats(crossing_slit, "data μ" , "velocity_pdf_up")
    # plot_velocity_stats(pairs_UP, "data μ–up" , "velocity_pdf_up")
    # plot_velocity_stats(pairs_DOWN, "data μ–down" , "velocity_pdf_down")
end

# @time particles_colliding  = QM_find_discarded_particles_multithreading(Icoils,crossing_slit,K39_params; t_length=1000, verbose=true)   # heavy loop: goes in series
# particles_reaching_screen = QM_build_alive_screen(Icoils,crossing_slit,particles_colliding,K39_params)   # [current_idx][μ_idx][x0 y0 z0 v0x v0y v0z x z vz]

@time particles_flag  = TheoreticalSimulation.QM_flag_travelling_particles(
                            Icoils, 
                            crossing_slit, 
                            K39_params; 
                            y_length=2500, 
                            verbose=true
);
@time particles_trajectories = TheoreticalSimulation.QM_build_travelling_particles(
        Icoils,
        crossing_slit,
        particles_flag,
        K39_params
);
TheoreticalSimulation.travelling_particles_summary(Icoils, quantum_numbers, particles_trajectories)
jldsave( joinpath(OUTDIR,"qm_$(Nss)_valid_particles_data.jld2"), data = OrderedDict(:Icoils => Icoils, :levels => fmf_levels(K39_params), :data => particles_trajectories))

alive_screen = OrderedDict(:Icoils=>Icoils, :levels => fmf_levels(K39_params), :data => TheoreticalSimulation.select_flagged(particles_trajectories,:screen ));
dead_crash   = OrderedDict(:Icoils=>Icoils, :levels => fmf_levels(K39_params), :data => TheoreticalSimulation.select_flagged(particles_trajectories,:crash ));

nx_bins , nz_bins = 32 , 2
println("F=$(K39_params.Ispin+0.5) profiles")
profiles_top = QM_analyze_profiles_to_dict(alive_screen, K39_params;
    manifold=:F_top, n_bins= (nx_bins , nz_bins), width_mm=0.150, add_plot=true, plot_xrange=:all, λ_raw=0.01, λ_smooth = 0.001, mode=:probability);
println("F=$(K39_params.Ispin-0.5) profiles")
profiles_bottom = QM_analyze_profiles_to_dict(alive_screen, K39_params;
    manifold=:F_bottom, n_bins= (nx_bins , nz_bins), width_mm=0.150, add_plot=true, plot_xrange=:all, λ_raw=0.01, λ_smooth = 0.001, mode=:pdf);

normalize_vec(v) = (m = maximum(v); m == 0 ? v : v ./ m);
anim = @animate for i in 1:nI
    # Monte Carlo profile (from your data)
    zprof = @views profiles_bottom[i][:z_profile]
    z_mc  = @views zprof[:, 1]                # (mm)
    s_mc  = @views zprof[:, 3]                # intensity
    s_mcN = normalize_vec(s_mc)

    # Closed-form profile
    zd     = range(-12.5, 12.5; length=20_001) .* 1e-3  # m
    dBzdz  = TheoreticalSimulation.GvsI(Icoils[i])
    dd1    = TheoreticalSimulation.getProbDist_v3(μF_effective(Icoils[i],1,-1,K39_params), dBzdz, zd, K39_params, effusion_params; npts=2001, pdf=:finite) 
    dd2    = TheoreticalSimulation.getProbDist_v3(μF_effective(Icoils[i],1,0,K39_params), dBzdz, zd, K39_params, effusion_params; npts=2001, pdf=:finite)
    dd3    = TheoreticalSimulation.getProbDist_v3(μF_effective(Icoils[i],1,1,K39_params), dBzdz, zd, K39_params, effusion_params; npts=2001, pdf=:finite)
    dd     = dd1 + dd2 + dd3
    ds_s   = TheoreticalSimulation.smooth_profile(zd, dd, 150e-6)
    ds_sN  = normalize_vec(ds_s)

    # Plot (x in mm, normalized y)
    fig = plot(z_mc, s_mcN;
         label="QM Monte Carlo", line=(:red, 2),
         xlabel=L"z~(\mathrm{mm})", ylabel="normalized intensity",
         legend=:topleft, size=(800, 520),
         left_margin=2mm,)

    plot!(1e3 .* zd, ds_sN; label="Closed-form", line=(:blue, 1.5))

    # Legend title with current
    plot!(legendtitle = L"$I_{0} = %$(round(Icoils[i]; digits=5))\,\mathrm{A}$")
    display(fig)
end
gif_path = joinpath(OUTDIR, "z_profiles_comparison.gif")
gif(anim, gif_path, fps=2)  # adjust fps as you like
@info "Saved GIF" gif_path


Icoils = alive_screen[:Icoils]
r = 1:3:length(Icoils)
for j in (r[end] == length(Icoils) ? r : Iterators.flatten((r, (length(Icoils),))))
# j=30
    data_set_5 = vcat((alive_screen[:data][j][k] for k in Int(2*K39_params.Ispin + 3):Int(4*K39_params.Ispin + 2))...)

    #Furnace
    xs_a = 1e3 .* data_set_5[:,1]; # mm
    zs_a = 1e6 .* data_set_5[:,3]; # μm
    figa = histogram2d(xs_a, zs_a;
        bins = (FreedmanDiaconisBins(xs_a), FreedmanDiaconisBins(zs_a)),
        show_empty_bins = true, color = :plasma, normalize=:pdf,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -1.0:0.25:1.0, yticks = -50:10:50,
        # clims = (0, 0.0003),
        # colorbar_position = :bottom,
    );

    # Slit
    r_at_slit = Matrix{Float64}(undef, size(data_set_5, 1), 3);
    for i in axes(data_set_5,1)
        r , _ = QM_EqOfMotion(y_FurnaceToSlit ./ data_set_5[i,5],Icoils[j],2,-2,data_set_5[i,1:3],data_set_5[i,4:6], K39_params)
        r_at_slit[i,:] = r
    end
    xs_b = 1e3 .* r_at_slit[:,1]; # mm
    zs_b = 1e6 .* r_at_slit[:,3]; # μm
    figb = histogram2d(xs_b, zs_b;
        bins = (FreedmanDiaconisBins(xs_b), FreedmanDiaconisBins(zs_b)),
        show_empty_bins = true, color = :plasma, normalize=:pdf,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -4.0:0.50:4.0, yticks = -200:50:200,
        xlims=(-4,4),
        # clims = (0, 0.0003),
        # colorbar_position = :bottom,
    ) ;

    # SG entrance
    r_at_SG_entrance = Matrix{Float64}(undef, size(data_set_5, 1), 3);
    for i in axes(data_set_5,1)
        r , _ = QM_EqOfMotion((y_FurnaceToSlit+y_SlitToSG) ./ data_set_5[i,5],Icoils[j],2,-2,data_set_5[i,1:3],data_set_5[i,4:6], K39_params)
        r_at_SG_entrance[i,:] = r
    end
    xs_c = 1e3 .* r_at_SG_entrance[:,1]; # mm
    zs_c = 1e6 .* r_at_SG_entrance[:,3]; # μm
    figc = histogram2d(xs_c, zs_c;
        bins = (FreedmanDiaconisBins(xs_c), FreedmanDiaconisBins(zs_c)),
        show_empty_bins = true, color = :plasma, normalize=:pdf,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -4.0:0.50:4.0, yticks = -1000:100:1000,
        xlims=(-4,4),
        # clims = (0, 0.0003),
        # colorbar_position = :bottom,
    );

    # SG exit
    r_at_SG_exit = Matrix{Float64}(undef, size(data_set_5, 1), 3);
    for i in axes(data_set_5,1)
        r , _ = QM_EqOfMotion((y_FurnaceToSlit+y_SlitToSG+y_SlitToSG) ./ data_set_5[i,5],Icoils[j],2,-2,data_set_5[i,1:3],data_set_5[i,4:6], K39_params)
        r_at_SG_exit[i,:] = r
    end
    xs_d = 1e3 .* r_at_SG_exit[:,1]; # mm
    zs_d = 1e6 .* r_at_SG_exit[:,3]; # μm
    figd = histogram2d(xs_d, zs_d;
        bins = (FreedmanDiaconisBins(xs_d), FreedmanDiaconisBins(zs_d)),
        show_empty_bins = true, color = :plasma, normalize=:pdf,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -4.0:0.50:4.0, 
        # yticks = -1000:200:1000,
        xlims=(-4,4),
        # clims = (0, 0.0003),
        # colorbar_position = :bottom,
    )
    x_magnet = 1e-3*range(-1.0,1.0,length=1000)
    plot!(figd,1e3*x_magnet,1e6*TheoreticalSimulation.z_magnet_edge.(x_magnet),line=(:dash,:black,2),label=false)

    # Screen
    r_at_screen = Matrix{Float64}(undef, size(data_set_5, 1), 3);
    for i in axes(data_set_5,1)
        r , _ = QM_EqOfMotion((y_FurnaceToSlit+y_SlitToSG+y_SlitToSG+y_SGToScreen) ./ data_set_5[i,5],Icoils[j],2,-2,data_set_5[i,1:3],data_set_5[i,4:6], K39_params)
        r_at_screen[i,:] = r
    end
    xs_e = 1e3 .* r_at_screen[:,1]; # mm
    zs_e = 1e3 .* r_at_screen[:,3]; # μm
    fige = histogram2d(xs_e, zs_e;
        bins = (FreedmanDiaconisBins(xs_e), FreedmanDiaconisBins(zs_e)),
        show_empty_bins = true, color = :plasma, normalize=:pdf,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{mm})$",
        # xticks = -4.0:0.50:4.0, yticks = -1250:50:1250,
        # clims = (0, 0.0003),
        # colorbar_position = :bottom,
    );


    fig = plot(figa,figb,figc,figd,fige,
    layout=(5,1),
    suptitle = L"$%$(1000*Icoils[j]) \mathrm{mA}$",
    size=(750,800),
    right_margin=2mm,
    bottom_margin=-2mm,
    )
    plot!(fig[1], xlabel="", bottom_margin=-3mm),
    plot!(fig[2], xlabel="", bottom_margin=-3mm),
    plot!(fig[3], xlabel="", bottom_margin=-3mm),
    plot!(fig[4], xlabel="", bottom_margin=-3mm),
    display(fig)

end

Bn_QM  = 2π*ħ*K39_params.Ahfs*(0.5+K39_params.Ispin) / 2 / μₑ
Bn_CQD = 11.8e-6
ix = range(10e-3,1,length=1001)
plot(ix,TheoreticalSimulation.BvsI.(ix),
    ribbon=Bn_QM*ones(length(ix)),
    xaxis=:log10,
    label=L"$B_{0} \pm B_{n}^{\mathrm{QM}}$")
plot(ix, abs.(Bn_QM ./  TheoreticalSimulation.BvsI.(ix)),
    label ="Relative magnitude" ,
    ylabel = L"B_{n}^{QM} / B_{0}",
    xaxis=:log10,)

plot(ix,TheoreticalSimulation.BvsI.(ix),
    ribbon=Bn_CQD*ones(length(ix)),
    label=L"$B_{0} \pm B_{n}^{\mathrm{CQD}}$",
    xaxis=:log10,)
plot(ix, abs.(Bn_CQD ./  TheoreticalSimulation.BvsI.(ix)),
    label ="Relative magnitude" ,
    ylabel = L"B_{n}^{CQD} / B_{0}",
    xaxis=:log10,)
















for i =1:nI
fig = plot(profiles_bottom[i][:z_profile][:,1],profiles_bottom[i][:z_profile][:,3]/maximum(profiles_bottom[i][:z_profile][:,3]),
    label="QM MonteCarlo",
    line=(:red,2))
zd = range(-12.5,12.5,20001)*1e-3
dBzdz = TheoreticalSimulation.GvsI(Icoils[i])
dd = TheoreticalSimulation.getProbDist_v3(μB, dBzdz,zd, K39_params, effusion_params; npts=2001, pdf=:finite)
dsds = TheoreticalSimulation.smooth_profile(zd,dd,150e-6)
plot!(1e3*zd,dsds/maximum(dsds), label="Closed-form", line=(:blue,1.5))
plot!(legendtitle=L"$I_{0}=%$(Icoils[i])\mathrm{A}$",
    legend=:topleft)
display(fig)
end



zd = range(-10,10,20001)*1e-3
dBzdz = 28.6/1.01 ;

plot(zd,getProbDist(zd, dBzdz, K39_params, effusion_params; wfurnace=0.200e-3, npts=401, pdf=:width))
plot!(zd,getProbDist(zd, dBzdz, K39_params, effusion_params; wfurnace=0.200e-3, npts=401, pdf=:point))















# ---- return a submatrix (copies the selected rows) ----
rows_pass(M) = M[(@view M[:,10]) .== 0.0, :]
rows_top(M)  = M[(@view M[:,10]) .== 1.0, :]
rows_bot(M)  = M[(@view M[:,10]) .== 2.0, :]
rows_scr(M)  = M[(@view M[:,10]) .== 3.0, :]
# generic:
rows_by_flag(M, flag::Real) = M[(@view M[:,10]) .== flag, :]


rows_by_flag(particles_travelling[30][5],0)





n_nonzero   = map(v -> count(!iszero, v), particles_colliding[30])
n_top       = map(v -> count(==(1), v), particles_colliding[30])
n_bot       = map(v -> count(==(3), v), particles_colliding[30])
n_tube      = map(v -> count(==(2), v), particles_colliding[30])
n_top+n_bot+n_tube == n_nonzero


#########################################################################################
T_END = Dates.now()
T_RUN = Dates.canonicalize(T_END-T_START)
report = """
***************************************************
EXPERIMENT
    Single Stern–Gerlach Experiment
    atom                    : $(atom)
    Output directory        : $(OUTDIR)

CAMERA FEATURES
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm

SETUP FEATURES
    Temperature             : $(T_K)
    Furnace aperture (x,z)  : ($(1e3*x_furnace)μm , $(1e6*z_furnace)μm)
    Slit (x,z)              : ($(1e3*x_slit)μm , $(1e6*z_slit)μm)
    Furnace → Slit          : $(1e3*y_FurnaceToSlit)mm
    Slit → SG magnet        : $(1e3*y_SlitToSG)mm
    SG magnet               : $(1e3*y_SG)mm
    SG magnet → Screen      : $(1e3*y_SGToScreen)mm
    Tube radius             : $(1e3*R_tube)mm

SIMULATION INFORMATION
    Number of atoms         : $(Nss)
    Binning                 : $(sim_bin_x) × $(sim_bin_z)
    Effective pixels        : $(x_pixels) × $(z_pixels)
    Pixel size              : $(1e6*sim_pixelsize_x)μm × $(1e6*sim_pixelsize_z)μm
    xlims                   : ($(round(minimum(1e6*x_position), digits=6)) μm, $(round(maximum(1e3*x_position), digits=4)) mm)
    zlims                   : ($(round(minimum(1e6*z_position), digits=6)) μm, $(round(maximum(1e3*z_position), digits=4)) mm)

    Currents (A)            : $(round.(Icoils,digits=5))
    No. of currents         : $(nI)

CODE
    Code name               : $(PROGRAM_FILE),
    Start date              : $(T_START)
    End data                : $(T_END)
    Run time                : $(T_RUN)
    Hostname                : $(hostname)

***************************************************
"""
# Print to terminal
println(report)

# Save to file
open(joinpath(OUTDIR,"simulation_report.txt"), "w") do io
    write(io, report)
end
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

dir_load_string = joinpath(@__DIR__, "simulation_data", "qm_analytic_sim")
data = load(joinpath(dir_load_string,"qm_2000000_valid_particles_data.jld2"))["data"]


data[:data]

for nz_iter in [1,2,4,8]
    println("\tCreates the z-profile with bin_nz = $(nz_iter)")
    nx_bins , nz_bins = 32 , nz_iter ;
    println("F=$(K39_params.Ispin+0.5) profiles")
    profiles_top = QM_analyze_profiles_to_dict(data, K39_params;
        manifold=:F_top, n_bins= (nx_bins,nz_bins), width_mm=0.150, add_plot=true, plot_xrange=:all, λ_raw=0.01, λ_smooth = 0.001, mode=:probability)
    println("F=$(K39_params.Ispin-0.5) profiles")
    profiles_bottom = QM_analyze_profiles_to_dict(data, K39_params;
        manifold=:F_bottom, n_bins= (nx_bins,nz_bins), width_mm=0.150, add_plot=true, plot_xrange=:all, λ_raw=0.01, λ_smooth = 0.001, mode=:pdf)

    jldsave( joinpath(dir_load_string, "zmax_profiles_top_$(nx_bins)x$(nz_bins).jld2"), data=profiles_top)
    jldsave( joinpath(dir_load_string, "zmax_profiles_bottom_$(nx_bins)x$(nz_bins).jld2"), data=profiles_bottom)
end


data[:Icoils]
data[:data][28][8]



data_num = load(joinpath(dir_load_string,"data_num_20250820.jld2"))["data"]
x1=load(joinpath(dir_load_string,"zmax_profiles_bottom_32x1.jld2"))["data"]
x2=load(joinpath(dir_load_string,"zmax_profiles_bottom_32x2.jld2"))["data"]
x4=load(joinpath(dir_load_string,"zmax_profiles_bottom_32x4.jld2"))["data"]
x8=load(joinpath(dir_load_string,"zmax_profiles_bottom_32x8.jld2"))["data"]

anim = @animate for i in eachindex(x1)
    fig = plot(
        xlabel=L"$z$ (mm)", 
        ylabel="Intensity (au)",
        xlims = (-5.0,12.5),
    )
    plot!(x1[i][:z_profile][:,1],x1[i][:z_profile][:,3], label=L"$n_{z}=1$")
    plot!(x2[i][:z_profile][:,1],x2[i][:z_profile][:,3], label=L"$n_{z}=2$")
    plot!(x4[i][:z_profile][:,1],x4[i][:z_profile][:,3], label=L"$n_{z}=4$")
    plot!(x8[i][:z_profile][:,1],x8[i][:z_profile][:,3], label=L"$n_{z}=8$")
    plot!(
        legendtitle= L"$I_{c}=%$(x1[i][:Icoil])\mathrm{A}$",
        legendtitlefontsize = 8,
    )
    display(fig)
end
gif(anim, joinpath(dir_load_string,"z_dw_profiles.gif"); fps=2) 

cols = palette(:darkrainbow,8)
fig = plot(xlabel="Current (A)", ylabel=L"$z_{max}$ (mm)")
plot!([4.5, 6.2, 7.8, 9.53, 12.93, 16.38, 21.6, 28.46, 35.39, 45.8, 57.86, 73.34, 92.38, 117.0, 148.1, 187.9, 239.9, 304.0, 387.0, 489.0, 623.0, 789.0]/1000, 
    [0.002286, 0.009615, 0.012025, 0.020023, 0.026224, 0.036606, 0.053704, 0.075875, 0.090899, 0.1277, 0.162333, 0.215772, 0.269977, 0.344652, 0.434583, 0.546819, 0.675259, 0.824128, 1.052232, 1.295606, 1.629635, 1.98437]/1.2697,
    label="Experimental data 20250814",
    seriestype=:scatter,
    marker=(:circle,:white,3),
    markerstrokewidth=2,
    markerstrokecolor=:brown,)
plot!([1.06, 2.81, 4.6, 6.39, 8.06, 11.64, 15.17, 20.47, 27.53, 35.94, 49.83, 67.16, 91.48, 121.5, 165.2, 222.9, 298.1, 407.0, 549.0, 739.0, 993.0]/1000, 
    [ 0.006027, 0.00683, 0.008836, 0.005707, 0.003933, 0.008094, 0.011108, 0.025917, 0.045916, 0.071526, 0.117163, 0.175239, 0.253787, 0.342267, 0.472704, 0.629466, 0.821751, 1.083423, 1.412981, 1.807272, 2.165861]/1.2697,
    label="Experimental data 20250820",
    seriestype=:scatter,
    marker=(:rect,:white,3.),
    markerstrokewidth=2,
    markerstrokecolor=:black,)
plot!([ 1.025, 2.835, 4.6, 6.34, 7.98, 9.8, 13.32, 16.83, 22.07, 27.4, 35.76, 46.22, 56.57, 72.12, 92.89, 117.7, 149.0, 189.1, 239.7, 308.0, 386.0, 494.0, 622.0, 787.0, 995.0]/1000, 
    [0.024406123, 0.025163373, 0.022424218, 0.023249073, 0.024653974, 0.023975278, 0.028969559, 0.033020055, 0.046727542, 0.062208333, 0.088430282, 0.121899207, 0.159090754, 0.212299474, 0.280313781, 0.354560784, 0.446607844, 0.556555544, 0.697296481, 0.865196889, 1.053177508, 1.328516385, 1.575994256, 1.949320167, 2.137558111]/1.2697,
    label="Experimental data 20250825",
    seriestype=:scatter,
    marker=(:diamond,:white,2.5),
    markerstrokewidth=2,
    markerstrokecolor=:gray36,)
#  [x8[i][s] for i in eachindex(x8), s in (:Icoil, :z_max_smooth_spline_mm)]
plot!([x1[i][:Icoil] for i =2:length(x1)] , [x1[i][:z_max_smooth_spline_mm] for i =2:length(x1)],
    label=L"Analytic QM ($n_{z}=1$)",
    line=(:dash,cols[1],1))
plot!([x2[i][:Icoil] for i =2:length(x2)] , [x2[i][:z_max_smooth_spline_mm] for i =2:length(x2)],
    label=L"Analytic QM ($n_{z}=2$)",
    line=(:dash,cols[2],1))
plot!([x4[i][:Icoil] for i =2:length(x4)] , [x4[i][:z_max_smooth_spline_mm] for i =2:length(x4)],
    label=L"Analytic QM ($n_{z}=4$)",
    line=(:dash,cols[3],1))
plot!([x8[i][:Icoil] for i =2:length(x8)] , [x8[i][:z_max_smooth_spline_mm] for i =2:length(x8)],
    label=L"Analytic QM ($n_{z}=8$)",
    line=(:dash,cols[4],1))
plot!(data_num[:runs][1][:data_QM][:,1], data_num[:runs][1][:data_QM][:,2],
    label=L"Numeric QM $(n_{z}=1$)",
    line=(:dot,cols[5],2))
plot!(data_num[:runs][2][:data_QM][:,1], data_num[:runs][2][:data_QM][:,2],
    label=L"Numeric QM $(n_{z}=2$)",
    line=(:dot,cols[6],2))
plot!(data_num[:runs][4][:data_QM][:,1], data_num[:runs][4][:data_QM][:,2],
    label=L"Numeric QM $(n_{z}=4$)",
    line=(:dot,cols[7],2))
plot!(data_num[:runs][8][:data_QM][:,1], data_num[:runs][8][:data_QM][:,2],
    label=L"Numeric QM $(n_{z}=8$)",
    line=(:dot,cols[8],2))
plot!(xlim=(12e-3,1.1))
plot!(xaxis=:log10, yaxis=:log10,
    legend=:bottomright)
savefig(fig, joinpath(dir_load_string,"qm_peaks.$(FIG_EXT)"))



# helper: symmetric bin centers/edges in mm
function symmetric_centers_edges(lims::Tuple{<:Real,<:Real}, bin_mm::Real)
    a, b = lims
    half = max(abs(a), abs(b))
    k = max(1, ceil(Int, half / bin_mm))
    centers = collect((-k:k) .* bin_mm)
    edges   = collect((-(k + 0.5)) * bin_mm : bin_mm : ((k + 0.5) * bin_mm))
    return centers, edges
end

# speed windows (change step to taste)
speed_edges = collect(range(400,800,10))           # 200–300, 300–400, …, 2000–2100
windows = zip(speed_edges[1:end-1], speed_edges[2:end])

# fixed analysis (mm)
xlim = (-8.0, 8.0); 
zlim = (-12.5, 12.5)
x_bin_mm = 1e3 * 32 * 6.5e-6                   # 32 px @ 6.5 µm → 0.208 mm
z_bin_mm = 1e3 *  4 * 6.5e-6                   #  4 px → 0.026 mm
centers_x, edges_x = symmetric_centers_edges(xlim, x_bin_mm)
centers_z, edges_z = symmetric_centers_edges(zlim, z_bin_mm)

for i in eachindex(data[:Icoils])
    # build once per current: columns [5,7,8] = [speed, x, z] (speed in m/s; x,z in meters)
    data_range = sortslices(
        vcat(data[:data][i][6][:, [5, 7, 8]],
             data[:data][i][7][:, [5, 7, 8]],
             data[:data][i][8][:, [5, 7, 8]]);
        dims = 1, by = row -> (row[1], row[3], row[2])
    )

    figh = plot(xlabel=L"$z$ (mm)", ylabel="mean counts (au)", legend=:topleft, legendtitle = @sprintf("%.3f A", data[:Icoils][i])  )
    for (lo, hi) in windows
        mask = (data_range[:, 1] .>= lo) .& (data_range[:, 1] .< hi)  # [lo, hi)
        if !any(mask)
            @info "no points in window [$lo, $hi) for i=$i"; continue
        end

        # x,z in meters -> mm for histogram (edges are in mm)
        x_mm = 1e3 .* @view(data_range[mask, 2])
        z_mm = 1e3 .* @view(data_range[mask, 3])

        h = fit(Histogram, (x_mm, z_mm), (edges_x, edges_z))   # raw counts
        counts = h.weights

        ttl = @sprintf("I = %.3f A   v ∈ [%g, %g) m/s", data[:Icoils][i], lo, hi)
        fig = heatmap(centers_x, centers_z, counts';
                      xlabel="x (mm)", ylabel="z (mm)",
                      title=ttl, 
                      # aspect_ratio=:equal,
                      # zscale=:log10,                 # uncomment if dynamic range is huge
                      colorbar_title="count")
        display(fig)
        # savefig(fig, joinpath(outdir, @sprintf("i%03d_v%04d-%04d.png", i, Int(round(lo)), Int(round(hi)))))
        # close(fig)

        s = analyze_screen_profile(data[:Icoils][i], hcat(x_mm,z_mm); 
            manifold=:F_bottom, 
            nx_bins = 1, nz_bins = 2, 
            add_plot=false, plot_xrange= :all,
            width_mm=0.150, λ_raw=0.01, λ_smooth = 1e-3, 
            mode=:density)
        plot!(figh,s.z_profile[:,1],s.z_profile[:,2], label=@sprintf("[%g,%g) m/s", lo, hi))
    end
    display(figh)
end



# for i in eachindex(data[:Icoils])
i=28
data_range = sortslices(vcat(data[:data][i][6][:,[5,7,8]],data[:data][i][7][:,[5,7,8]],data[:data][i][8][:,[5,7,8]]); dims = 1, by = row -> (row[1], row[3], row[2]))
lo, hi = 300.0, 500.0;
mask = (data_range[:, 1] .>= lo) .& (data_range[:, 1] .<= hi)
A_in = data_range[mask,:]

# Fixed analysis limits
xlim = (-8.0, 8.0)
zlim = (-12.5, 12.5)
xmin, xmax = xlim
zmin, zmax = zlim
x_bin_size = 1e3 * 32 * 6.5e-6
z_bin_size = 1e3 * 4 * 6.5e-6
x_half_range = max(abs(xmin), abs(xmax))
kx = max(1, ceil(Int, x_half_range / x_bin_size))
centers_x = collect((-kx:kx) .* x_bin_size)
edges_x = collect((-(kx + 0.5)) * x_bin_size : x_bin_size : ((kx + 0.5) * x_bin_size))
z_half_range = max(abs(zmin), abs(zmax))
kz = max(1, ceil(Int, z_half_range / z_bin_size))
centers_z = collect((-kz:kz) .* z_bin_size)
edges_z = collect((-(kz + 0.5)) * z_bin_size : z_bin_size : ((kz + 0.5) * z_bin_size))
x = @view A_in[:, 2]
z = @view A_in[:, 3]
h = fit(Histogram, (1e3*x, 1e3*z), (edges_x, edges_z))
counts = h.weights  # size: (length(centers_x), length(centers_z))
fig = heatmap(centers_x, centers_z, counts', xlabel="x (mm)", ylabel="z (mm)", title="2D Histogram")
display(fig)
# end

s = analyze_screen_profile(0.01, 1e3*A_in[:,[2,3]]; manifold=:lvl1, 
    nx_bins = 1, nz_bins = 2, 
    add_plot=true, plot_xrange= :all,
    width_mm=0.150, λ_raw=0.01, λ_smooth = 1e-3, 
    mode=:density)

plot(s.z_profile[:,1],s.z_profile[:,2])


analyze_screen_profile(Ix, data_mm::AbstractMatrix; 
    manifold::Symbol = :F_top, nx_bins::Integer = 2, nz_bins::Integer = 2, 
    add_plot::Bool = false, plot_xrange::Symbol = :all,
    width_mm::Float64 = 0.150, λ_raw::Float64=0.01, λ_smooth::Float64 = 1e-3, 
    mode::Symbol=:probability)

histogram(vcat(data[:data][30][6][:,9],data[:data][30][7][:,9],data[:data][30][8][:,9]), normalize=:probability, label=L"$v_{z}$ (m/s)")
histogram!(vcat(data[:data][1][6][:,6]), normalize=:probability, label=L"$v_{0,z}$ (m/s)")

data[:data][30][6][:,6]
[[length(data[:data][i][j][:,6]) for i=1:30] for j=1:8]



#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################


# load(joinpath(OUTDIR,"zmax_profiles_top.jld2"))["data"]

# bb = jldopen(joinpath(OUTDIR,"qm_data.jld2"))["data"]



filtered[1][1]

π



#################################################################################
# FUNCTIONS
#################################################################################















function CQD_find_bad_particles_ix(Ix, pairs, kx::Float64)
    No = size(pairs, 1)  # Number of particles
    ncurrents = length(Ix)

    # Indexed by idx, NOT threadid
    bad_particles_per_current = Vector{Vector{Int}}(undef, ncurrents)
    for i in 1:ncurrents
        bad_particles_per_current[i] = Int[]
    end

    Threads.@threads for idx in 1:ncurrents
    # for idx in 1:ncurrents
        i0 = Ix[idx]
        println("Analyzing current I₀ = $(@sprintf("%.3f", i0))A")

        local_bad_particles = Int[]  # local to this thread and current
        hits_SG = 0
        hits_post = 0

        for j = 1:No
            try
                @inbounds begin
                    v_y = pairs[j, 5]
                    t_in = (y_FurnaceToSlit + y_SlitToSG) / v_y
                    t_out = (y_FurnaceToSlit + y_SlitToSG + y_SG) / v_y
                    # t_screen = (y_FurnaceToSlit + y_SlitToSG + y_SG + y_SGToScreen) / v_y
                    t_length = 1000

                    r0 = @view pairs[j, 1:3]
                    v0 = @view pairs[j, 4:6]
                    θe0 = pairs[j, 7]
                    θn0 = pairs[j, 8]
                end

                t_sweep_sg  = range(t_in, t_out, length=t_length)
                z_val       = CQD_EqOfMotion_z.(t_sweep_sg, Ref(i0), Ref(μₑ), Ref(r0), Ref(v0), Ref(θe0), Ref(θn0), Ref(kx))
                z_top       = z_magnet_edge_time.(t_sweep_sg, Ref(r0), Ref(v0))
                z_bottom    = z_magnet_trench_time.(t_sweep_sg, Ref(r0), Ref(v0))

                inside_cavity = (z_bottom .< z_val) .& (z_val .< z_top)
                if !all(inside_cavity)
                    push!(local_bad_particles, j)
                    hits_SG += 1
                    continue
                end

                # Post-SG pipe check
                x_screen, _ ,  z_screen = CQD_Screen_position(i0, μₑ, r0, v0, θe0, θn0, kx)
                if x_screen^2 + z_screen^2 .>= R_tube^2
                    push!(local_bad_particles, j)
                    hits_post += 1 
                    continue
                end

            catch err
                @error "Thread $(Threads.threadid()), particle $j crashed" exception=err
            end
        end

        println("\t→ SG hits   = $hits_SG")
        println("\t→ Pipe hits = $hits_post\n")

        sort!(local_bad_particles)
        bad_particles_per_current[idx] = local_bad_particles
    end

    # Final result as Dict for compatibility
    bad_particles = Dict{Int8, Vector{Int}}()
    for idx in 1:ncurrents
        bad_particles[Int8(idx)] = bad_particles_per_current[idx]
    end

    return bad_particles
end

function compute_screen_xyz( Ix::Vector, valid_up::OrderedDict, valid_dw::OrderedDict, kx::Float64) 
    screen_up = OrderedDict{Int64, Matrix{Float64}}()
    screen_dw = OrderedDict{Int64, Matrix{Float64}}()

    
    @inbounds for i in eachindex(Ix)
        good_up = valid_up[i]
        good_dw = valid_dw[i]

        N_up = size(good_up, 1)
        N_dw = size(good_dw, 1)

        coords_up = Matrix{Float64}(undef, N_up, 3)
        coords_dw = Matrix{Float64}(undef, N_dw, 3)

        Threads.@threads for j = 1:N_up
        # for j = 1:N_up
            # r0 = @view good_up[j, 1:3]
            # v0 = @view good_up[j, 4:6]
            r0  = SVector{3,Float64}(good_up[j, 1], good_up[j, 2], good_up[j, 3])
            v0  = SVector{3,Float64}(good_up[j, 4], good_up[j, 5], good_up[j, 6])
            θe0 = good_up[j, 7]
            θn0 = good_up[j, 8]
            coords_up[j, :] = CQD_Screen_position(Ix[i], μₑ, r0, v0, θe0, θn0, kx)
        end

        # Threads.@threads for j = 1:N_dw
        for j = 1:N_dw
            # r0 = @view good_dw[j, 1:3]
            # v0 = @view good_dw[j, 4:6]
            r0  = SVector{3,Float64}(good_dw[j, 1], good_dw[j, 2], good_dw[j, 3])
            v0  = SVector{3,Float64}(good_dw[j, 4], good_dw[j, 5], good_dw[j, 6])
            θe0 = good_dw[j, 7]
            θn0 = good_dw[j, 8]
            coords_dw[j, :] = CQD_Screen_position(Ix[i], μₑ, r0, v0, θe0, θn0, kx)
        end

        screen_up[i] = coords_up
        screen_dw[i] = coords_dw
    end

    return screen_up, screen_dw
end





"""
    plot_SG_magneticfield(path_filename::AbstractString) -> Nothing

    Plot magnetic field properties of the Stern–Gerlach apparatus and save the result.

    The figure contains three vertically stacked panels:
    1. Magnetic field gradient vs coil current, with experimental data and fitted model.
    2. Magnetic field B_z vs coil current, with experimental data and fitted model.
    3. B_z vs gradient, using the model curves.

    # Arguments
    - `path_filename::AbstractString`: Output path for saving the figure (PNG, PDF, etc.).

    # Assumptions
    The following must be defined in the current scope:
    - `GradCurrents`, `GradGradient`: experimental gradient data vs current.
    - `GvsI(current::AbstractVector)`: function returning model gradient vs current.
    - `Bdata.dI`, `Bdata.Bz`: experimental B-field data vs current.
    - `BvsI(current::AbstractVector)`: function returning model B-field vs current.
"""
function plot_SG_magneticfield(path_filename::AbstractString)
    @assert isdefined(Main, :GradCurrents) && isdefined(Main, :GradGradient) "Missing gradient data."
    @assert isdefined(Main, :GvsI) && isdefined(Main, :BvsI) "Missing model functions."
    @assert isdefined(Main, :Bdata) "Missing B-field experimental data."

    icoils = collect(range(1e-6, 1.05, length=10_000))

    # Panel 1: Gradient vs current
    fig1a = plot(GradCurrents, GradGradient;
        seriestype = :scatter, marker = (:circle, :black, 2),
        label = false, xlabel = "Coil Current (A)",
        ylabel = "Magnetic field gradient (T/m)",
        yticks = 0:50:350
    )
    plot!(fig1a, icoils, GvsI(icoils);
        line = (:red, 2), label = L"$\partial_{z}B_{z}$"
    )

    # Panel 2: B-field vs current
    fig1b = plot(Bdata.dI, Bdata.Bz;
        seriestype = :scatter, marker = (:circle, :black, 2),
        label = false, xlabel = "Coil Current (A)",
        ylabel = "Magnetic field (T)",
        yticks = 0:0.1:1.0
    )
    plot!(fig1b, icoils, BvsI(icoils);
        line = (:orange, 2), label = L"$B_{z}$"
    )

    # Panel 3: B-field vs gradient
    fig1c = plot(GvsI(icoils), BvsI(icoils);
        label = false, line = (:blue, 2),
        xlabel = "Magnetic field gradient (T/m)",
        ylabel = "Magnetic field (T)",
        ylims = (0, 0.8), xticks = 0:50:350, yticks = 0:0.1:1.0
    )

    # Layout
    fig = plot(fig1a, fig1b, fig1c;
        layout = @layout([a1; a2; a3]),
        size = (400, 700),
        plot_title = "Magnetic field in the Stern–Gerlach apparatus",
        plot_titlefont = font(10, "Computer Modern"),
        guidefont = font(8, "Computer Modern"),
        left_margin = 5mm, bottom_margin = 0mm, right_margin = 0mm
    )

    savefig(fig, path_filename)
    return nothing
end



"""
    plot_polar_stats(Ix, data_up, data_dw, path_filename) -> Plot

    Generate a 2×2 grid of histograms showing electron (θₑ) and nuclear (θₙ) polar angle
    distributions for a randomly chosen coil current from `Ix`. The top row shows "up"
    spin data, and the bottom row shows "down" spin data, with electron angles in the
    left column and nuclear angles in the right column. Tick numbers are hidden on the
    top row’s x-axes and the right column’s y-axes. The figure is saved to `path_filename`
    and returned as a `Plots.Plot` object.

    # Arguments
    - `Ix::Vector{Float64}`: Coil current values in amperes.
    - `data_up`: Collection of 2D arrays for "up" spin data (≥8 columns).
    - `data_dw`: Collection of 2D arrays for "down" spin data (≥8 columns).
    - `path_filename::AbstractString`: Output path for saving the figure.
"""
function plot_polar_stats(Ix::Vector{Float64}, data_up, data_dw, path_filename::AbstractString)
    @assert !isempty(Ix) "Ix is empty."
    @assert length(data_up) == length(Ix) "data_up length must match Ix."
    @assert length(data_dw) == length(Ix) "data_dw length must match Ix."

    idxi0 = rand(1:length(Ix))
    up = data_up[idxi0]
    dw = data_dw[idxi0]
    @assert ndims(up) == 2 && size(up,2) ≥ 8 "data_up[idxi0] must be a 2D array with ≥ 8 columns."
    @assert ndims(dw) == 2 && size(dw,2) ≥ 8 "data_dw[idxi0] must be a 2D array with ≥ 8 columns."

    # FD_histograms should return a Plots.Plot
    figa = FD_histograms(up[:,7], L"\theta_{e}", :dodgerblue)
    figb = FD_histograms(up[:,8], L"\theta_{n}", :red)
    figc = FD_histograms(dw[:,7], L"\theta_{e}", :dodgerblue)
    figd = FD_histograms(dw[:,8], L"\theta_{n}", :red)
    
    fig = plot(figa, figb, figc, figd,
        layout = @layout([a1 a2 ; a3 a4]),
        size = (600, 600),
        plot_title = L"Initial polar angles for $I_{c}= %$(Ix[idxi0]) \mathrm{A}$",
        plot_titlefontsize = 10,
        guidefont = font(8, "Computer Modern"),
        link = :both,
        left_margin = 5mm, bottom_margin = 0mm, right_margin = 0mm,
    );

    # Remove x-axis numbers from top row
    plot!(fig[1]; xticks=(xticks(fig[1])[1], fill("", length(xticks(fig[1])[1]))), xlabel="",bottom_margin=-5mm)
    plot!(fig[2]; xticks=(xticks(fig[2])[1], fill("", length(xticks(fig[2])[1]))), xlabel="",bottom_margin=-5mm)

    # Remove y-axis numbers from right column
    plot!(fig[2]; yticks=(yticks(fig[2])[1], fill("", length(yticks(fig[2])[1]))), ylabel="", left_margin=-5mm)
    plot!(fig[4]; yticks=(yticks(fig[4])[1], fill("", length(yticks(fig[4])[1]))), ylabel="", left_margin=-5mm)

    savefig(fig,  path_filename)
        
    return fig
end

function get_valid_particles_per_current(pairs, bad_particles_dict)
    valid_dict = OrderedDict{Int, Matrix}()
    all_indices = 1:size(pairs, 1)
    for (idx, bad_indices) in bad_particles_dict
        good_indices = setdiff(all_indices, bad_indices)
        valid_dict[idx] = pairs[good_indices, :]
    end
    return valid_dict
end














save_fig = true
if save_fig == true
    plot_SG_geometry(joinpath(dir_path, "slit.png"));
    plot_SG_magneticfield(joinpath(dir_path, "SG_magneticfield.png"));
    plot_ueff(Ispin,joinpath(dir_path, "mu_effective.png"));
end






pairs_UP = build_init_cond(alive_slit, θesUP, θnsUP);
pairs_DOWN = build_init_cond(alive_slit, θesDOWN, θnsDOWN);
# Optionally clear memory
θesUP = θnsUP = θesDOWN = θnsDOWN = alive_slit = nothing
GC.gc()

ki = 3.96e-6

bad_particles_up = find_bad_particles_ix(Icoils, pairs_UP, ki)
bad_particles_up = OrderedDict(sort(collect(bad_particles_up); by=first))

valid_up = get_valid_particles_per_current(pairs_UP,   bad_particles_up)
# println("Particles with final θₑ=0")
# for (i0, content) in valid_up
#     println("Current $(@sprintf("%.3f", Icoils[i0]))A \t→   Good particles: ", size(content,1))
# end
pairs_UP = bad_particles_up =nothing

bad_particles_dw = find_bad_particles_ix(Icoils, pairs_DOWN, ki)
bad_particles_dw = OrderedDict(sort(collect(bad_particles_dw); by=first))

valid_dw = get_valid_particles_per_current(pairs_DOWN, bad_particles_dw)
# println("Particles with final θₑ=0")
# for (i0, content) in valid_dw
#     println("Current $(@sprintf("%.3f", Icoils[i0]))A \t→   Good particles: ", size(content,1))
# end
pairs_DOWN = bad_particles_dw = nothing

GC.gc()

println("Minimum number of valid particles for up-spin: $(minimum(size(valid_up[v],1) for v in eachindex(Icoils)))")
println("Minimum number of valid particles for down-spin: $(minimum(size(valid_dw[v],1) for v in eachindex(Icoils)))")

@save joinpath(dir_path, "data_up.jld2") valid_up Icoils
@save joinpath(dir_path, "data_dw.jld2") valid_dw Icoils

# ########################################################################################################################
# # data recovery
# data_path = ["./simulation_data/"] .* [
#     "20250807T163648/",
#     "20250807T180252/", 
#     "20250807T181304/",
# ]

# # data_u = JLD2.jldopen(joinpath(data_path[3], "data_up.jld2"), "r") do file
# #     return Dict(k => read(file, k) for k in keys(file))
# # end
# # data_d = JLD2.jldopen(joinpath(data_path[3], "data_dw.jld2"), "r") do file
# #     return Dict(k => read(file, k) for k in keys(file))
# # end
# # valid_up = data_u["valid_up"]
# # valid_dw = data_d["valid_dw"]

# # Load all valid_up dictionaries into a vector
# # valid_up_list = [
# #     JLD2.jldopen(joinpath(path, "data_up.jld2"), "r") do file
# #         read(file, "valid_up")
# #     end
# #     for path in data_path
# # ]

# # # Combine: same keys, so we vcat the matrices for each key
# # combined_valid_up = OrderedDict{Int64, Matrix{Float64}}()
# # for k in keys(valid_up_list[1])
# #     combined_valid_up[k] = vcat([vu[k] for vu in valid_up_list]...)
# # end

# # combined_valid_up

# function combine_valid_data(data_paths::Vector{String}; spin::Symbol = :up)
#     # Decide which key to load
#     key = spin === :up ? "valid_up" : "valid_dw"
#     file_name = spin === :up ? "data_up.jld2" : "data_dw.jld2"

#     # Load each dictionary from file
#     dict_list = Vector{OrderedDict{Int64, Matrix{Float64}}}(undef, length(data_paths))
#     for (i, path) in enumerate(data_paths)
#         filepath = joinpath(path, file_name)
#         dict_list[i] = JLD2.jldopen(filepath, "r") do file
#             read(file, key)
#         end
#     end

#     # Get the common keys
#     first_keys = collect(keys(dict_list[1]))
#     combined = OrderedDict{Int64, Matrix{Float64}}()

#     # Preallocate dictionary
#     for k in first_keys
#         combined[k] = Matrix{Float64}(undef, 0, size(dict_list[1][k], 2))
#     end

#     # Threaded concatenation
#     @threads for i in eachindex(first_keys)
#         k = first_keys[i]
#         combined[k] = vcat((d[k] for d in dict_list)...)
#     end

#     return combined
# end

# valid_up = combine_valid_data(data_path; spin = :up)
# valid_dw = combine_valid_data(data_path; spin = :dw)
# ########################################################################################################################

if save_fig
    display(plot_polar_stats(Icoils, valid_up, valid_dw, joinpath(dir_path, "polar_stats.png")))
end

screen_up, screen_dw = compute_screen_xyz(Icoils, valid_up, valid_dw, ki);

results_up = analyze_profiles_to_dict(
    Icoils, screen_up;
    m_mom=:up,
    n_bins=1, width_mm=0.10, add_plot=true, λ_raw=0.01, λ_smooth=1e-3,
    store_profiles=true
)

results_dw = analyze_profiles_to_dict(
    Icoils, screen_dw;
    m_mom=:dw,
    n_bins=1, width_mm=0.10, add_plot=true, λ_raw=0.01, λ_smooth=1e-3,
    store_profiles=true
)

@save joinpath(dir_path, "zpeak_up.jld2") results

data_comparison = zeros(Float64,nI,8)
for i=1:nI
    data_comparison[i,:] = [results_up[i][:z_max_raw_mm] , 
                            results_up[i][:z_max_raw_spline_mm], 
                            results_up[i][:z_max_smooth_mm] , 
                            results_up[i][:z_max_smooth_spline_mm],
                            results_dw[i][:z_max_raw_mm] , 
                            results_dw[i][:z_max_raw_spline_mm], 
                            results_dw[i][:z_max_smooth_mm] , 
                            results_dw[i][:z_max_smooth_spline_mm]
                            ]
end

data_centroid = (data_comparison[:,1:4] .+ data_comparison[:,5:8])/2
centroid_mean = mean(data_centroid, Weights(0:nI-1), dims=1)
centroid_std = permutedims(std.(eachcol(data_centroid), Ref(Weights(0:nI-1)))) ./ sqrt(nI)


plot(Icoils, abs.(data_comparison[:,1]), label="Raw (px)")
plot!(Icoils, data_comparison[:,2], label="Raw Spline (sub px)")
plot!(Icoils, data_comparison[:,3], label="Smoothed (px)")
plot!(Icoils, data_comparison[:,4], label="Smoothed Spline (sub px)")
# plot!(Icoils, data_comparison[:,5], label="Raw (px)")
# plot!(Icoils, data_comparison[:,6], label="Raw Spline (sub px)")
# plot!(Icoils, data_comparison[:,7], label="Smoothed (px)")
# plot!(Icoils, data_comparison[:,8], label="Smoothed Spline (sub px)")
plot!(
    yaxis=(:log10, L"$z_{\mathrm{max}} \ (\mathrm{mm})$"),
    xaxis = (:log10, L"$I_{0} \ (\mathrm{A})$"),
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),

)







rand_maxwell_chi(a, No; rng=rng_set) = a .* rand(rng, Chi(3), No)

rand_maxwell_gamma(a, No; rng=rng_set) = begin
    G = rand(rng, Gamma(3/2,1.0), No)
    a .* sqrt.(2 .* G)
end

arth=rand_maxwell_gamma(sqrt(kb*T/M), 80000000)
mean(arth)

histogram(rand_maxwell_gamma(sqrt(kb*T/M), 100000))

2*sqrt(2/π)*sqrt(kb*T/M)
2+2


# interpolation on the high current and low currents profiles for different velocity groups
# seek the function for the convlution low and high currents
# share with arthur peak position for f1 and f2: centroid folded and peak positions :::: DONE
# contact arthur for keep track of his codes : github compatible 
# experimental comparison symmetric f1 and f2 ::::
# include peak heights and compare ratios for the simulations
# process new data











n_bins

throw(error("valimos"))
# Example access:
results[5][:Icoil]
results[5][:z_profile][1900:2000,1]


for idxi0=1:nI
    data_up = 1e3*screen_up[idxi0][:,[1,3]] # [mm]
    res = analyze_screen_profile(Icoils[idxi0],data_up; n_bins=1, width_mm=0.10, add_plot=true, λ_raw=0.2 ,λ_smooth=1e-6)
    println(res.z_max_smooth_spline_mm)
end

screen_up

data_up = 1e3*screen_up[17][:,[1,3]] # [mm]
res = analyze_screen_profile(data_up; n_bins=8, width_mm=0.10, add_plot=true)
println(res.z_max_smooth_mm)

# global max position only
a,b, c = max_of_bspline(res.z_profile[:,1], res.z_profile[:,3])

plot(c.(collect(-5.0:0.01:5.0)))

2+2










extrema(data[:,1])
extrema(data[:,2])

sum(data[:,1].^2 + data[:,2].^2 .< (1e3*R_tube)^2)

n_bins = 2
xmin = -9.0
xmax =  9.0
zmin = -12.5
zmax =  12.5
bin_size = 1e3 * n_bins * cam_pixelsize

x_pixels = ceil(Int, (xmax - xmin) / bin_size)
z_pixels = ceil(Int, (zmax - zmin) / bin_size)

edges_x = xmin:bin_size:(xmin + x_pixels*bin_size)
edges_z = zmin:bin_size:(zmin + z_pixels*bin_size)

# Example usage:
centers_x = bin_centers(edges_x)
centers_z = bin_centers(edges_z)

# data is Nx2 matrix: columns are x and z positions
h = fit(Histogram, (data[:, 1], data[:, 2]), (edges_x, edges_z))
counts = h.weights

# heatmap expects x and y vectors, and a matrix of values (counts)
# heatmap(
#     centers_x,
#     centers_z,
#     counts',
#     xlabel = "x (mm)",
#     ylabel = "z (mm)",
#     title = "2D Histogram",
#     color = :inferno,
#     # aspect_ratio = :equal,
# );

z_profile = vec(mean(counts,dims=1))
# Raw max
zmax_idx = argmax(z_profile)
z_max_0 = centers_z[zmax_idx]
plot(centers_z, z_profile)
# Example usage:
wd = 0.1                   # kernel width (mm), adjust as needed
smoothed_pdf = smooth_profile(centers_z, z_profile, wd)
plot!(centers_z, smoothed_pdf)


error("does it work?")



































function analyze_2dhist(Ix::Float64, data::AbstractMatrix, n_bins::Int)

    @assert size(data, 2) ≥ 2 "Data must have at least two columns (x and z)."

    # Determine bounds
    sim_xmin, sim_xmax = extrema(data[:, 1])
    sim_zmin, sim_zmax = extrema(data[:, 2])

    # Number of bins without materializing all edges
    steps = 1e3 * n_bins * cam_pixelsize
    nbins_x = Int(cld(sim_xmax - sim_xmin, steps)) + 1
    nbins_z = Int(cld(sim_zmax - sim_zmin, steps)) + 1

    # Fit 2D histogram
    h0 = fit(
        Histogram,
        (data[:, 1], data[:, 2]),
        (range(sim_xmin, sim_xmax, length=nbins_x),
         range(sim_zmin, sim_zmax, length=nbins_z))
    )
    h0 = normalize(h0, mode=:pdf)

    # Bin centers
    # bin_centers_x = (h0.edges[1][1:end-1] .+ h0.edges[1][2:end]) ./ 2
    bin_centers_z = (h0.edges[2][1:end-1] .+ h0.edges[2][2:end]) ./ 2

    # Z-profile (mean along x-axis)
    z_profile = hcat(bin_centers_z, vec(mean(h0.weights, dims=1)))

    # Raw max
    zmax_idx = argmax(z_profile[:, 2])
    z_max_0 = z_profile[zmax_idx, 1]

    # Figure: 2D histogram
    fig_2dhist = histogram2d(
        data[:, 1], data[:, 2],
        nbins=(nbins_x, nbins_z),
        color=:inferno,
        title=L"Co Quantum Dynamics: $I_{c}=%$(Ix)\mathrm{A}$ $\vec{\mu}_{e} \upuparrows \hat{z}$",
        xlabel=L"$x \ (\mathrm{mm})$",
        ylabel=L"$z \ (\mathrm{mm})$",
        xlim=(sim_xmin, sim_xmax),
        show_empty_bins=true,
    )
    hline!([z_max_0], label=false, line=(:red, :dash, 1))

    # Figure: Z-profile with LOESS
    fig_prof = plot(
        z_profile[:, 1], z_profile[:, 2],
        label="Simulation",
        seriestype=:line,
        line=(:gray, 1),
        title="CoQuantum Dynamics",
        xlabel=L"$z \, (\mathrm{mm})$",
        legend=:best,
        # zlims=(0, :auto),
        legendtitle=(
            isempty(Icoils) ?
            "" :
            L"$I_{0}=%$(Ix)\,\mathrm{A}$"
        ),
        legendtitlefontsize=10,
    )
    vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$", line=(:red, :dash, 2))

    # LOESS fit
    zscan = range(minimum(z_profile[:, 1]), maximum(z_profile[:, 1]), step=0.001)
    model = loess(z_profile[:, 1], z_profile[:, 2], span=0.10)
    plot!(zscan, predict(model, zscan), label="Loess", line=(:purple4, 2, 0.5))

    # Optimization to refine z_max
    smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
    opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
    z_max_fit = Optim.minimizer(opt_result)[1]
    vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red, :dot, 2))

    return (
        # h0=h0,
        z_profile=z_profile,
        z_max_0=z_max_0,
        z_max_fit=z_max_fit,
        fig_2dhist=fig_2dhist,
        fig_prof=fig_prof
    )
end

result = analyze_2dhist(Icoils[22], data, 2)


result.fig_prof


function gaussian_kernel(x,wd)
    # Create Gaussian kernel around zero
    kernel = (1 / (sqrt(2π) * wd)) .* exp.(-x .^ 2 ./ (2 * wd^2))
    kernel ./= sum(kernel)  # normalize to sum to 1
    return kernel
end

function smooth_profile(z_vals, pdf_vals, wd)
    kernel = gaussian_kernel(z_vals,wd)
    # Convolve pdf values with kernel, pad=true means full convolution
    smoothed = DSP.conv(pdf_vals, kernel)
    # Trim convolution result to same length as input, like MATLAB 'same'
    n = length(pdf_vals)
    start_idx = div(length(kernel), 2) + 1
    return smoothed[start_idx:start_idx + n - 1]
end




# Create 2D histogram
n_bins=2
sim_xmin , sim_xmax = minimum(data[:,1])  , maximum(data[:,1])
sim_zmin , sim_zmax = minimum(data[:,2]) , maximum(data[:,2])
nbins_x = length(collect(sim_xmin:1e3*n_bins * cam_pixelsize:sim_xmax))+1
nbins_z = length(collect(sim_zmin:1e3*n_bins * cam_pixelsize:sim_zmax))+1
h0 = fit(Histogram,(data[:,1],data[:,2]),
    (range(sim_xmin,sim_xmax,length=nbins_x),range(sim_zmin,sim_zmax,length=nbins_z))
)
h0=normalize(h0,mode=:pdf)
bin_edges_x = collect(h0.edges[1])
bin_edges_z = collect(h0.edges[2])
bin_centers_x = (bin_edges_x[1:end-1] .+ bin_edges_x[2:end]) ./ 2  # Compute bin centers
bin_centers_z = (bin_edges_z[1:end-1] .+ bin_edges_z[2:end]) ./ 2  # Compute bin centers

z_profile = hcat(bin_centers_z, vec(mean(h0.weights,dims=1)))
# Find the index of the maximum value in the second column & Extract the corresponding value from the first column
zmax_idx = argmax(z_profile[:, 2])
z_max_0 = z_profile[zmax_idx, 1]

fig_2dhist = histogram2d(data[:,1],data[:,2],
    nbins=(nbins_x,nbins_z),
    # normalize=:pdf,
    color=:inferno,
    title=L"Co Quantum Dynamics: $\vec{\mu}_{e} \upuparrows \hat{z}$",
    xlabel=L"$x \ (\mathrm{mm})$",
    ylabel=L"$z \ (\mathrm{mm})$",
    xlim=(sim_xmin, sim_xmax),
    # ylim=(sim_zmin,3),
    show_empty_bins=true,
)
hline!([z_max_0],label=false,line=(:red,:dash,1))


# Example usage:
wd = 0.1                   # kernel width (mm), adjust as needed
smoothed_pdf = smooth_profile(z_profile[:,1], z_profile[:,2], wd)

fig_prof = plot(z_profile[:,1],z_profile[:,2],
    label="Simulation",
    seriestype=:line,
    line=(:gray,1),
    # marker=(:black,:circle,2),
    title="CoQuantum Dynamics",
    # zlims=(0,:auto),
    xlabel=L"$z \, (\mathrm{mm})$",
    legend=:best,
    legendtitle=L"$I_{0}=%$(Icoils[idxi0])\,\mathrm{A}$",
    legendtitlefontsize=10,
)
vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$",line=(:red,:dash,2))
zscan = collect(minimum(z_profile[:,1]):0.001:maximum(z_profile[:,1]))

model = loess(z_profile[:,1],z_profile[:,2], span=0.10)
plot!(zscan,predict(model,zscan),
    label="Loess",
    line=(:purple4,2,0.5),
)

# Define the smoothed function
smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
# Find minimum using optimization
opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
z_max_fit = Optim.minimizer(opt_result)[1]
vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red,:dot,2))
plot!(z_profile[:,1], smoothed_pdf)
2+2































s_bin = 2
data = 1e3*hcat(screen_coord[:,1,5],screen_coord[:,3,5])
data = permutedims(reduce(hcat, filter(row -> row[2] <= 5, eachrow(data)) |> collect ))

# Create 2D histogram
sim_xmin , sim_xmax = minimum(data[:,1])  , maximum(data[:,1])
sim_zmin , sim_zmax = minimum(data[:,2]) , maximum(data[:,2])
nbins_x = length(collect(sim_xmin:s_bin*cam_pixelsize:sim_xmax))+1
nbins_z = length(collect(sim_zmin:s_bin*cam_pixelsize:sim_zmax))+1
h0 = fit(Histogram,(data[:,1],data[:,2]),
    (range(sim_xmin,sim_xmax,length=nbins_x),range(sim_zmin,sim_zmax,length=nbins_z))
)
h0=normalize(h0,mode=:pdf)
bin_edges_x = collect(h0.edges[1])
bin_edges_z = collect(h0.edges[2])
bin_centers_x = (bin_edges_x[1:end-1] .+ bin_edges_x[2:end]) ./ 2  # Compute bin centers
bin_centers_z = (bin_edges_z[1:end-1] .+ bin_edges_z[2:end]) ./ 2  # Compute bin centers

z_profile = hcat(bin_centers_z, vec(mean(h0.weights,dims=1)))
# Find the index of the maximum value in the second column & Extract the corresponding value from the first column
zmax_idx = argmax(z_profile[:, 2])
z_max_0 = z_profile[zmax_idx, 1]

fig_2dhist = histogram2d(data[:,1],data[:,2],
    nbins=(nbins_x,nbins_z),
    normalize=:pdf,
    color=:inferno,
    title=L"Co Quantum Dynamics: $\vec{\mu}_{e} \upuparrows \hat{z}$",
    xlabel=L"$x \ (\mathrm{mm})$",
    ylabel=L"$z \ (\mathrm{mm})$",
    xlim=(sim_xmin, sim_xmax),
    ylim=(sim_zmin,8),
    show_empty_bins=true,
)
hline!([z_max_0],label=false,line=(:red,:dash,1))

fig_prof = plot(z_profile[:,1],z_profile[:,2],
    label="Simulation",
    seriestype=:line,
    line=(:gray,1),
    # marker=(:black,:circle,2),
    title="CoQuantum Dynamics",
    zlims=(0,:auto),
    xlabel=L"$z \, (\mathrm{mm})$",
    legend=:topright,
    legendtitle=L"$I_{0}=%$(21)\,\mathrm{A}$",
    legendtitlefontsize=10,
)
vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$",line=(:red,:dash,2))
zscan = collect(minimum(z_profile[:,1]):0.001:maximum(z_profile[:,1]))


model = loess(z_profile[:,1],z_profile[:,2], span=0.10)
plot!(zscan,predict(model,zscan),
    label="Loess",
    line=(:purple4,2,0.5),
)

# Define the smoothed function
smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
# Find minimum using optimization
opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
z_max_fit = Optim.minimizer(opt_result)[1]
vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red,:dot,2))

23

function CQD_analysis(Ix,cqd_data::AbstractMatrix; z_upper = 10 , s_bin = 1 , loess_factor = 0.10 )

    data = cqd_data[:,[9,11]]
    data = permutedims(reduce(hcat, filter(row -> row[2] <= z_upper, eachrow(data)) |> collect ))

    # Create 2D histogram
    sim_xmin , sim_xmax = minimum(data[:,1])  , maximum(data[:,1])
    sim_zmin , sim_zmax = minimum(data[:,2]) , maximum(data[:,2])
    nbins_x = length(collect(sim_xmin:s_bin*cam_pixelsize:sim_xmax))+1
    nbins_z = length(collect(sim_zmin:s_bin*cam_pixelsize:sim_zmax))+1
    h0 = fit(Histogram,(data[:,1],data[:,2]),
        (range(sim_xmin,sim_xmax,length=nbins_x),range(sim_zmin,sim_zmax,length=nbins_z))
    )
    h0=normalize(h0,mode=:pdf)
    bin_edges_x = collect(h0.edges[1])
    bin_edges_z = collect(h0.edges[2])
    bin_centers_x = (bin_edges_x[1:end-1] .+ bin_edges_x[2:end]) ./ 2  # Compute bin centers
    bin_centers_z = (bin_edges_z[1:end-1] .+ bin_edges_z[2:end]) ./ 2  # Compute bin centers

    z_profile = hcat(bin_centers_z, vec(mean(h0.weights,dims=1)))
    # Find the index of the maximum value in the second column & Extract the corresponding value from the first column
    zmax_idx = argmax(z_profile[:, 2])
    z_max_0 = z_profile[zmax_idx, 1]


    fig_2dhist = histogram2d(data[:,1],data[:,2],
        nbins=(nbins_x,nbins_z),
        normalize=:pdf,
        color=:inferno,
        title=L"Co Quantum Dynamics: $\vec{\mu}_{e} \upuparrows \hat{z}$",
        xlabel=L"$x \ (\mathrm{mm})$",
        ylabel=L"$z \ (\mathrm{mm})$",
        xlim=(sim_xmin, sim_xmax),
        ylim=(sim_zmin,sim_zmax),
        show_empty_bins=true,
    )
    hline!([z_max_0],label=false,line=(:red,:dash,1))

    fig_prof = plot(z_profile[:,1],z_profile[:,2],
        label="Simulation",
        seriestype=:line,
        line=(:gray,1),
        # marker=(:black,:circle,2),
        title="CoQuantum Dynamics",
        zlims=(0,:auto),
        xlabel=L"$z \, (\mathrm{mm})$",
        legend=:topright,
        legendtitle=L"$I_{0}=%$(Ix)\,\mathrm{A}$",
        legendtitlefontsize=10,
    )
    vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$",line=(:red,:dash,2))
    zscan = collect(minimum(z_profile[:,1]):0.001:maximum(z_profile[:,1]))
    ## Dierckx
    # fspline = Spline1D(z_profile[:,1],z_profile[:,2],k=3,s=0.01)
    ## Define and optimize the negative spline function
    # neg_spline(x) = -fspline(x[1])
    # opt_result = optimize(neg_spline, [minimum(bin_centers)], [maximum(bin_centers)], [bin_center_max], Fminbox(LBFGS()))
    # plot!(zscan,fspline(zscan))

    # Loess
    model = loess(z_profile[:,1],z_profile[:,2], span=loess_factor)
    plot!(zscan,predict(model,zscan),
        label="Loess",
        line=(:purple4,2,0.5),
    )

    # Define the smoothed function
    smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
    # Find minimum using optimization
    opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
    z_max_fit = Optim.minimizer(opt_result)[1]
    vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red,:dot,2))


    return fig_2dhist , fig_prof , z_max_0 , z_max_fit , z_profile
end

# function QM_analysis(Ix,dataqm::AbstractMatrix,ms,mf::AbstractVector; z_upper = 10, s_bin = 1 , loess_factor = 0.10 )
#     if ms==1/2
#         idx = [[2 1 0 -1],[9,10,11,12]]
#         # Find the indices where elements in idx[1] are in mf
#         valid_indices = findall(x -> x in mf, idx[1])
#         # Convert CartesianIndex to plain indices
#         flat_indices = [i[2] for i in valid_indices]
#         # Retrieve the corresponding elements from idx[2]
#         valid_columns = idx[2][flat_indices]
#     else
#         idx = [[-2 1 0 -1],[13,14,15,16]]
#         # Find the indices where elements in idx[1] are in mf
#         valid_indices = findall(x -> x in mf, idx[1])
#         # Convert CartesianIndex to plain indices
#         flat_indices = [i[2] for i in valid_indices]
#         # Retrieve the corresponding elements from idx[2]
#         valid_columns = idx[2][flat_indices]
#     end
#     data = dataqm[:, [7,valid_columns[1]]]  # Start with the 7th column
#     for i in valid_columns[2:end]
#         data = vcat(data, dataqm[:, [7, i]])  # Concatenate columns
#     end

#     data = permutedims(reduce(hcat, filter(row -> row[2] <= z_upper, eachrow(data)) |> collect ))

#     # Create 2D histogram
#     sim_xmin , sim_xmax = minimum(data[:,1])  , maximum(data[:,1])
#     sim_zmin , sim_zmax = minimum(data[:,2]) , maximum(data[:,2])
#     nbins_x = length(collect(sim_xmin:s_bin*cam_pixelsize:sim_xmax))+1
#     nbins_z = length(collect(sim_zmin:s_bin*cam_pixelsize:sim_zmax))+1
#     h0 = fit(Histogram,(data[:,1],data[:,2]),
#         (range(sim_xmin,sim_xmax,length=nbins_x),range(sim_zmin,sim_zmax,length=nbins_z))
#     )
#     h0=normalize(h0,mode=:pdf)
#     bin_edges_x = collect(h0.edges[1])
#     bin_edges_z = collect(h0.edges[2])
#     bin_centers_x = (bin_edges_x[1:end-1] .+ bin_edges_x[2:end]) ./ 2  # Compute bin centers
#     bin_centers_z = (bin_edges_z[1:end-1] .+ bin_edges_z[2:end]) ./ 2  # Compute bin centers

#     z_profile = hcat(bin_centers_z, vec(mean(h0.weights,dims=1)))
#     # Find the index of the maximum value in the second column & Extract the corresponding value from the first column
#     zmax_idx = argmax(z_profile[:, 2])
#     z_max_0 = z_profile[zmax_idx, 1]


#     fig_2dhist = histogram2d(data[:,1],data[:,2],
#         nbins=(nbins_x,nbins_z),
#         normalize=:pdf,
#         color=:inferno,
#         title=L"Quantum Mechanics: $m_{s} \updownarrows \hat{z}$",
#         xlabel=L"$x \ (\mathrm{mm})$",
#         ylabel=L"$z \ (\mathrm{mm})$",
#         xlim=(sim_xmin, sim_xmax),
#         ylim=(sim_zmin,sim_zmax),
#         show_empty_bins=true,
#     )
#     hline!([z_max_0],label=false,line=(:red,:dash,1))

#     fig_prof = plot(z_profile[:,1],z_profile[:,2],
#         label="Simulation",
#         title="Quantum mechanics",
#         seriestype=:line,
#         line=(:gray,1),
#         # marker=(:black,:circle,2),
#         zlims=(0,:auto),
#         xlabel=L"$z \, (\mathrm{mm})$",
#         legend=:topright,
#         legendtitle=L"$I_{0}=%$(Ix)\,\mathrm{A}$",
#         legendtitlefontsize=10,
#     )
#     vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$",line=(:red,:dash,2))
#     zscan = collect(minimum(z_profile[:,1]):0.001:maximum(z_profile[:,1]))
#     ## Dierckx
#     # fspline = Spline1D(z_profile[:,1],z_profile[:,2],k=3,s=0.01)
#     ## Define and optimize the negative spline function
#     # neg_spline(x) = -fspline(x[1])
#     # opt_result = optimize(neg_spline, [minimum(bin_centers)], [maximum(bin_centers)], [bin_center_max], Fminbox(LBFGS()))
#     # plot!(zscan,fspline(zscan))

#     # Loess
#     model = loess(z_profile[:,1],z_profile[:,2], span=loess_factor)
#     plot!(zscan,predict(model,zscan),
#         label="Loess",
#         line=(:purple4,2,0.5),
#     )

#     # Define the smoothed function
#     smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
#     # Find minimum using optimization
#     opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
#     z_max_fit = Optim.minimizer(opt_result)[1]
#     vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red,:dot,2))


#     return fig_2dhist , fig_prof , z_max_0 , z_max_fit , z_profile

# end


Icoils = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.75,0.80];
hist2d_cqd_up = Vector{Plots.Plot}()
histz_cqd_up = Vector{Plots.Plot}()
hist2d_qm = Vector{Plots.Plot}()
histz_qm = Vector{Plots.Plot}()
zpeak = zeros(Float64,length(Icoils),4)
@time for (idx,Io) in enumerate(Icoils)
    println("\t\tCurrent $(Io) A")
    # CO QUANTUM DYNAMICS
    # Add the final position according to CQD to each final projection 
    # [x0,y0,z0,vx0,vy0,vz0,θₑ,θₙ,xf,yf,zf]
    println("Atoms with magnetic moment going UP")
    atomsCQD_UP=[Vector{Float64}() for _ in 1:length(pairs_UP)]
    @time @threads for i=1:length(pairs_UP)
        atomsCQD_UP[i] = vcat(pairs_UP[i],
        CQD_Screen_position(Io,μₑ,pairs_UP[i][1:3],pairs_UP[i][4:6],pairs_UP[i][7])
        )
    end
    println("Atoms with magnetic moment going DOWN")
    atomsCQD_DOWN=[Vector{Float64}() for _ in 1:length(pairs_DOWN)]
    @time @threads for i=1:length(pairs_DOWN)
        atomsCQD_DOWN[i] = vcat(pairs_DOWN[i],
        CQD_Screen_position(Io,-μₑ,pairs_DOWN[i][1:3],pairs_DOWN[i][4:6],pairs_DOWN[i][7])
        )
    end

    # QUANTUM MECHANICS 
    # [x0,y0,z0, v0x,v0y,v0z, xf,yf,zf(2,2), zf(2,1),zf(2,0),zf(2,-1),zf(2,-2), zf(1,1),zf(1,0),zf(1,-1)]
    # println("Atoms in QM")
    # atomsQM=[Vector{Float64}() for _ in 1:Nss]
    # μF2p2 , μF2p1 , μF20 , μF2m1 , μF2m2 = μF_effective(Io,Ispin,2,2), μF_effective(Io,Ispin,2,1) , μF_effective(Io,Ispin,2,0) , μF_effective(Io,Ispin,2,-1) , μF_effective(Io,Ispin,2,-2)
    # μF1p1 , μF10 , μF1m1 = μF_effective(Io,Ispin,1,1) , μF_effective(Io,Ispin,1,0) , μF_effective(Io,Ispin,1,-1)
    # @time @threads for i=1:Nss
    #     atomsQM[i] = vcat(alive_slit[i],
    #     QM_Screen_position(Io,μF2p2, alive_slit[i][1:3], alive_slit[i][4:6]),
    #     QM_Screen_position(Io,μF2p1, alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF20,  alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF2m1, alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF2m2, alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF1p1, alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF10,  alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF1m1, alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     )
    # end

    atomsCQD_UP     = permutedims(reduce(hcat, atomsCQD_UP))
    atomsCQD_DOWN   = permutedims(reduce(hcat, atomsCQD_DOWN))
    # atomsQM     = permutedims(reduce(hcat, atomsQM))
    println("Data analysis : ",Io,"A")
    
    result_cqd_up   = CQD_analysis(Io,atomsCQD_UP;              z_upper = 8 , s_bin = 8 , loess_factor = 0.07)
    # result_qm_f1    = QM_analysis(Io,atomsQM,-0.5,[1,0,-1] ;   z_upper = 8 , s_bin = 8 , loess_factor = 0.07)

    push!(hist2d_cqd_up, result_cqd_up[1])
    # push!(hist2d_qm, result_qm_f1[1])

    display(result_cqd_up[1])
    # display(result_qm_f1[1])

    push!(histz_cqd_up, result_cqd_up[2])
    # push!(histz_qm, result_qm_f1[2])
    display(result_cqd_up[2])
    # display(result_qm_f1[2])

    writedlm(filename*"I$(idx)_cqd.csv",result_cqd_up[5],',')
    # writedlm(filename*"I$(idx)_qm.csv",result_qm_f1[5],',')

    # zpeak[idx,:] = [result_cqd_up[3],result_cqd_up[4],result_qm_f1[3],result_qm_f1[4]]
    zpeak[idx,:] = [result_cqd_up[3],result_cqd_up[4]]


end


Io= 0.0
s_bin=4
cam_pixelsize=0.0065

println("Atoms with magnetic moment going UP")
atomsCQD_UP=[Vector{Float64}() for _ in 1:length(pairs_UP)]
@time @threads for i=1:length(pairs_UP)
    atomsCQD_UP[i] = vcat(pairs_UP[i],
    CQD_Screen_position(Io,μₑ,pairs_UP[i][1:3],pairs_UP[i][4:6],pairs_UP[i][7])
    )
end

atomsCQD_UP     = permutedims(reduce(hcat, atomsCQD_UP))
data = atomsCQD_UP[:,[9,11]]
data = permutedims(reduce(hcat, filter(row -> -10 <= row[2] <= 10, eachrow(data)) |> collect ))

# Create 2D histogram
sim_xmin , sim_xmax = minimum(data[:,1])  , maximum(data[:,1])
sim_zmin , sim_zmax = minimum(data[:,2]) , maximum(data[:,2])
nbins_x = length(collect(sim_xmin:s_bin*cam_pixelsize:sim_xmax))+1
nbins_z = length(collect(sim_zmin:s_bin*cam_pixelsize:sim_zmax))+1
h0 = StatsBase.fit(Histogram,(data[:,1],data[:,2]),
    (range(sim_xmin,sim_xmax,length=nbins_x),range(sim_zmin,sim_zmax,length=nbins_z))
)
h0=normalize(h0,mode=:pdf)
bin_edges_x = collect(h0.edges[1])
bin_edges_z = collect(h0.edges[2])
bin_centers_x = (bin_edges_x[1:end-1] .+ bin_edges_x[2:end]) ./ 2  # Compute bin centers
bin_centers_z = (bin_edges_z[1:end-1] .+ bin_edges_z[2:end]) ./ 2  # Compute bin centers

z_profile = hcat(bin_centers_z, vec(mean(h0.weights,dims=1)))
# Find the index of the maximum value in the second column & Extract the corresponding value from the first column
zmax_idx = argmax(z_profile[:, 2])
z_max_0 = z_profile[zmax_idx, 1]


fig_2dhist = histogram2d(data[:,1],data[:,2],
    nbins=(nbins_x,nbins_z),
    normalize=:pdf,
    color=:inferno,
    title=L"Co Quantum Dynamics: $\vec{\mu}_{e} \upuparrows \hat{z}$",
    xlabel=L"$x \ (\mathrm{mm})$",
    ylabel=L"$z \ (\mathrm{mm})$",
    xlim=(sim_xmin, sim_xmax),
    ylim=(sim_zmin,sim_zmax),
    show_empty_bins=true,
)
hline!([z_max_0],label=false,line=(:red,:dash,1))

fig_prof = plot(z_profile[:,1],z_profile[:,2],
    label="Simulation",
    seriestype=:line,
    line=(:gray,1),
    # marker=(:black,:circle,2),
    title="CoQuantum Dynamics",
    zlims=(0,:auto),
    xlabel=L"$z \, (\mathrm{mm})$",
    legend=:topright,
    legendtitle=L"$I_{0}=%$(Io)\,\mathrm{A}$",
    legendtitlefontsize=10,
)
vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$",line=(:red,:dash,2))
zscan = collect(minimum(z_profile[:,1]):0.001:maximum(z_profile[:,1]))
## Dierckx
# fspline = Spline1D(z_profile[:,1],z_profile[:,2],k=3,s=0.01)
## Define and optimize the negative spline function
# neg_spline(x) = -fspline(x[1])
# opt_result = optimize(neg_spline, [minimum(bin_centers)], [maximum(bin_centers)], [bin_center_max], Fminbox(LBFGS()))
# plot!(zscan,fspline(zscan))

#BSplineKit
xs = z_profile[:,1]
ys = z_profile[:,2]
λ=0.01
weights = (1-λ)fill!(similar(xs), 1)
weights[zmax_idx]=2
S_fit = BSplineKit.fit(xs, ys,0.001; weights)
S_interp = BSplineKit.interpolate(xs, ys, BSplineOrder(4),BSplineKit.Natural())
scatter(xs, ys; label = "Data", marker = (:black,2))
plot!(xs, S_interp.(xs); label = "Interpolation", linewidth = 2)
plot!(xs, S_fit.(xs); label = "Fit (λ = $λ )", linewidth = 2)
neg_spline(x) = -S_fit(x[1])
opt_result = optimize(neg_spline, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
vline!([Optim.minimizer(opt_result)[1]])

# Loess
model = loess(z_profile[:,1],z_profile[:,2], span=0.1)
plot!(zscan,predict(model,zscan),
    label="Loess",
    line=(:purple4,2,0.5),
)

# Define the smoothed function
smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
# Find minimum using optimization
opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
z_max_fit = Optim.minimizer(opt_result)[1]
vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red,:dot,2))

rng = MersenneTwister(42)
Ndata = 20
xs = sort!(rand(rng, Ndata))



################################################################################################
################################################################################################
################################################################################################

Iexp = [ 0.0, 0.01, 0.02, 0.03, 0.05, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75 ]
zexp = [
    0.00124986,
    0.00900368,
    0.0227256,
    0.0629495,
    0.11486,
    0.390562,
    0.510494,
    0.631897,
    0.812013,
    1.12686,
    1.59759
]

sulqm=[0.0409
0.0566
0.0830
0.1015
0.1478
0.1758
0.2409
0.3203
0.4388
0.5433
0.6423
0.8394
1.1267
1.5288]

for i=1:length(sulqm)
    sulqm[i] = sulqm[i]+0.00625*rand(Uniform(-1,1))
end


sulcqd = [0.0179
0.0233
0.0409
0.0536
0.0883
0.1095
0.1713
0.2487
0.3697
0.4765
0.5786
0.7757
1.0655
1.4630]

for i=1:length(sulcqd)
    sulcqd[i] = sulcqd[i]+0.001*rand(Uniform(-1,1))
end

suli=   [ 0.0150
0.0200
0.0250
0.0300
0.0400
0.0500
0.0700
0.1000
0.1500
0.2000
0.2500
0.3500
0.5000
0.7500]


fig5=plot(Icoils[2:end],zpeak[2:end,2],
label="Coquantum dynamics",
# seriestype=:scatter,
marker=(:rect,:red,2),
markerstrokecolor=:red,
line=(:red,1,0.6),
xaxis=:log10,
yaxis=:log10,
xlims=(0.008,1),
legend=:topleft)
plot!(Icoils[2:end],zpeak[2:end,4],
label="Quantum mechanics",
# seriestype=:scatter,
marker=(:blue,:diamond,2),
markerstrokecolor=:blue,
line=(:blue,1))
plot!(Iexp[3:end], zexp[3:end],
label="COIL",
seriestype=:scatter,
marker=(:xcross,:black,3),
markeralpha=0.85,
markerstrokecolor=:black,
markerstrokewidth=3)     # mean(zpeak[:, 3:4], dims=2))
display(fig5)
savefig(fig5,filename*"_05.svg")



sulqm=[0.0409
0.0566
0.0830
0.1015
0.1478
0.1758
0.2409
0.3203
0.4388
0.5433
0.6423
0.8394
1.1267
1.5288]












t_run = Dates.canonicalize(Dates.now()-t_start)
# Create a dictionary with all the parameters
params = OrderedDict(
    "Experiment" => "FRISCH-SEGRÈ EXPERIMENT",
    "Equation" => "Bloch Equation ($equation)",
    "Filename" => filename,
    "Atom" => atom,
    "kᵢ:CQD" => "$ki",
    "B-field" => field,
    "Iw direction" => "$(Iw_direction)̂",
    "ODE system" => "$(θn_DiffEq)",
    "zₐ" => "$(1e6 .* zₐ)μm",
    "v" => "$(v)m/s",
    "Bᵣ" => "$b_remnant",
    "Bₑ" => "$(round(1e3*Be, digits=3))mT",
    "Bₙ" => "$(round(1e6*Bn, digits=3))μT",
    "Initial μₑ" => "$θe0_arrow [θₑ(tᵢ)=$(round(θe0/π, digits=4))π]",
    "Initial μₙ" => initial_μₙ,
    "θₙ(tᵢ)" => initial_μₙ == "CONSTANT" ? "$(round(θn_constant/π,digits=4))π" : "",
    "RNG" => string(rng)[1:findfirst(c -> c in ['{', '('], string(rng))-1],
    "N atoms" => "$N_atoms",
    "Time span" => "$(1e6 .* tspan)μs",
    "SG magnets" => "(BSG=$(BSG)T, ySG=±$(1e3 * ySG)mm)",
    "R²" => "$(round.(R_Squared; digits=4))",
    "δθ" => "$δθ",
    "Algorithm" => string(alg)[1:findfirst(c -> c in ['{', '('], string(alg))-1],
    "reltol" => "$reltol",
    "abstol" => "$abstol",
    "dtmin" => "$dtmin",
    "Start date" => Dates.format(t_start, "yyyy-mm-ddTHH-MM-SS"),
    "End date" => Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS"),
    "Run time" => "$t_run",
    "Hostname" => hostname,
    "Code name" => PROGRAM_FILE,
    "Iwire" => "$Iwire",
    "Prob(μₑ:↓)" => "$PSF_FS_global",
    "Prob(μₑ:↓|δt)" => "$(PSF_δt_avg[:,1])",
    "Prob(μₑ:↓|Bₑ>>B₀)" => "$PSF_FS_local",
    "Prob(μₑ:↓|Bₑ>>B₀|δt)" => "$(PSF_δt_avg[:,2])"
)
# Determine the maximum length of keys
max_key_length = maximum(length.(keys(params)))

open(filename * ".txt", "w") do file
    for (key, value) in params
        if value ≠ ""
            # Format each line with the key aligned
            write(file, @sprintf("%-*s = \t%s\n", max_key_length, key, value))
        end
    end
end

println("script   << $filename >>   has finished!")
println("$atom [ $experiment | $equation | $θe0_arrow | $initial_μₙ | $θn_DiffEq | $field | $(Int.(1e6.*tspan))μs | $(Int(1e6*zₐ))μm | $(v)m/s | $b_remnant | N=$N_atoms ]")
alert("script   << $filename >>   has finished!")

data = vcat(atomsQM[:,[7,14]],atomsQM[:,[7,15]],atomsQM[:,[7,16]])
data = permutedims(reduce(hcat, filter(row -> row[2] <= z_upper, eachrow(data)) |> collect ))

# Create 2D histogram
sim_xmin , sim_xmax = minimum(data[:,1])  , maximum(data[:,1])
sim_zmin , sim_zmax = minimum(data[:,2]) , maximum(data[:,2])
nbins_x = length(collect(sim_xmin:s_bin*cam_pixelsize:sim_xmax))+1
nbins_z = length(collect(sim_zmin:s_bin*cam_pixelsize:sim_zmax))+1
h0 = fit(Histogram,(data[:,1],data[:,2]),
    (range(sim_xmin,sim_xmax,length=nbins_x),range(sim_zmin,sim_zmax,length=nbins_z))
)
h0=normalize(h0,mode=:pdf)
bin_edges_x = collect(h0.edges[1])
bin_edges_z = collect(h0.edges[2])
bin_centers_x = (bin_edges_x[1:end-1] .+ bin_edges_x[2:end]) ./ 2  # Compute bin centers
bin_centers_z = (bin_edges_z[1:end-1] .+ bin_edges_z[2:end]) ./ 2  # Compute bin centers

z_profile = hcat(bin_centers_z, vec(mean(h0.weights,dims=1)))
# Find the index of the maximum value in the second column & Extract the corresponding value from the first column
zmax_idx = argmax(z_profile[:, 2])
z_max_0 = z_profile[zmax_idx, 1]


fig_2dhist = histogram2d(data[:,1],data[:,2],
    nbins=(nbins_x,nbins_z),
    normalize=:pdf,
    color=:inferno,
    title=L"Quantum mechanics: $F = 1$",
    xlabel=L"$x \ (\mathrm{mm})$",
    ylabel=L"$z \ (\mathrm{mm})$",
    xlim=(sim_xmin, sim_xmax),
    ylim=(sim_zmin,sim_zmax),
    show_empty_bins=true,
)
hline!([z_max_0],label=false,line=(:red,:dash,1))

fig_prof = plot(z_profile[:,1],z_profile[:,2],
    label="Simulation",
    seriestype=:line,
    line=(:gray,1),
    # marker=(:black,:circle,2),
    zlims=(0,:auto),
    xlabel=L"$z \, (\mathrm{mm})$",
    legend=:topright,
    legendtitle=L"$I_{0}=%$(Io)\,\mathrm{A}$",
    legendtitlefontsize=10,
)
vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$",line=(:red,:dash,2))
zscan = collect(minimum(z_profile[:,1]):0.001:maximum(z_profile[:,1]))
## Dierckx
# fspline = Spline1D(z_profile[:,1],z_profile[:,2],k=3,s=0.01)
## Define and optimize the negative spline function
# neg_spline(x) = -fspline(x[1])
# opt_result = optimize(neg_spline, [minimum(bin_centers)], [maximum(bin_centers)], [bin_center_max], Fminbox(LBFGS()))
# plot!(zscan,fspline(zscan))

# Loess
model = loess(z_profile[:,1],z_profile[:,2], span=loess_factor)
plot!(zscan,predict(model,zscan),
    label="Loess",
    line=(:purple4,2,0.5),
)

# Define the smoothed function
smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
# Find minimum using optimization
opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
z_max_fit = Optim.minimizer(opt_result)[1]
vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red,:dot,2))






plot(bin_centers_z,vec(mean(h0.weights,dims=1)),
    seriestype=:line,
    line=(:gray,1),
    marker=(:black,:circle,2),
    # xlims=(-1,8),
)
x = bin_centers_z[findall(x -> (-1 <= x <= 9), bin_centers_z)]
y = vec(mean(h0.weights,dims=1))[findall(x -> (-1 <= x <= 9), bin_centers_z)]
splfit = Spline1D(x,y,
    # w=ones(length(x)),
    k=3, 
    s=length(x)*0.5)

    model = loess(x, y, span=0.1)


zz=collect(minimum(bin_centers_z):0.013:maximum(bin_centers_z))
us = range(extrema(x)...; step = 0.1)
plot!(zz,(splfit(zz)))
plot!(us,predict(model,us),line=(:green,3))


histogram(vec(mean(h0.weights,dims=1)),nbins=588,normalize=:probability)


xs = 10 .* rand(100)
ys = sin.(xs) .+ 0.5 * rand(100)

model = loess(xs, ys, span=0.5)
vs = predict(model, us)

scatter(xs, ys)
plot!(us, vs, legend=false)


heatmap(h0.weights',nbins=nbins_z)



minimum(1e3*atomsCQD_UP[:,9]):0.026:maximum(1e3*atomsCQD_UP[:,9])
minimum(1e3*atomsCQD_UP[:,11]):0.026:maximum(1e3*atomsCQD_UP[:,11])




d1 = randn(10_000)
d2 = randn(10_000)

nbins1 = 25
nbins2 = 10
	
hist = fit(Histogram, (d1,d2),
		(range(minimum(d1), stop=maximum(d1), length=nbins1+1),
		range(minimum(d2), stop=maximum(d2), length=nbins2+1)))
plot(hist)

data = [
    0.0  0.074501;
    0.1  0.127343;
    0.2  0.187198;
    0.3  0.299073;
    0.4  0.435718;
    0.5  0.467139;
    0.6  0.62702;
    0.7  0.631098;
    0.8  0.774073;
    0.9  0.793128;
    1.0  0.84104;
    1.1  0.886343;
    1.2  0.93662;
    1.3  0.956826;
    1.4  0.966104;
    1.5  0.999325;
    1.6  0.993967;
    1.7  0.98652;
    1.8  0.989205;
    1.9  0.914493;
    2.0  0.894332;
    2.1  0.884692;
    2.2  0.835543;
    2.3  0.790565;
    2.4  0.668164;
    2.5  0.52381;
    2.6  0.591465;
    2.7  0.406899;
    2.8  0.260562;
    2.9  0.214678;
    3.0  0.181986;
    3.1  0.0490647
]

# Sample data (replace with your own data)
x = 0:0.1:10
y = sin.(x) + 0.1 * randn(length(x))  # Adding some noise to make it interesting


# Define the cost function
function cost_fn(x,y,smoothing_factor)
    # Fit the spline using cubic interpolation (without smoothing)
    spline = CubicSplineInterpolation(x, y, extrapolation_bc=Line())

    # Calculate the residual sum of squares (RSS)
    residuals = sum((spline(x) .- y).^2)
    
    # Approximate the second derivative for roughness penalty
    dx = diff(x)
    second_derivative = diff(spline(x)) ./ dx
    second_derivative_penalty = sum(second_derivative.^2)  # Roughness penalty

    # Define the cost function as a weighted sum of residuals and penalty
    p = smoothing_factor
    cost = p*residuals + (1-p) * second_derivative_penalty
    
    return cost
end

# Function to fit the smoothing spline with a given smoothing parameter
function fit_spline(x, y, smoothing_factor)
    # Minimize the cost function to get the optimal smoothing factor
    result = optimize(cost_fn, 0.0, 1.0, BFGS(), x, y, smoothing_factor)

    # Return the fitted spline
    optimal_smoothing_factor = Optim.minimizer(result)[1]
    spline = CubicSplineInterpolation(x, y, extrapolation_bc=Line())

    return spline
end


# Fit the spline
fitted_spline = fit_spline(x, y, 0.98)


# Use Optim.jl to minimize the cost function
result = optimize(cost_fn, 0.0, 1.0, BFGS())  # Optimization over smoothing factor (0 to 1)

# Get the optimal smoothing factor
optimal_smoothing_factor = Optim.minimizer(result)[1]
println("Optimal smoothing factor: ", optimal_smoothing_factor)

# Fit the spline using the optimal smoothing factor
spline_with_optimal_smoothing = CubicSplineInterpolation(x, y, extrapolation_bc=Line())

# Plot the original data and the fitted spline
plot(x, y, label="Original data", marker=:o)
plot!(x, spline_with_optimal_smoothing(x), label="Fitted spline with optimal smoothing", linewidth=2)







# Function to compute the smoothing spline
function (x, y, smoothing_factor)
    # Fit the spline using cubic interpolation
    spline = CubicSplineInterpolation(x, y, extrapolation_bc=Line())
    
    # Define the penalty term: the integral of the square of the second derivative (roughness)
    # This is an approximation of the smoothness of the spline
    dx = diff(x)
    second_derivative_penalty = sum((diff(spline(x)[2:end])./dx[2:end]).^2)
    
    # Calculate the residuals (least squares)
    residuals = sum((spline(x) .- y).^2)
    
    # Define the cost function: a weighted sum of residuals and penalty
    cost = residuals .+ smoothing_factor .* second_derivative_penalty
    
    return cost
end

# Define the cost function for optimization (smoothing_factor will be optimized)
function cost_fn(smoothing_factor)
    return fit_spline(x, y, smoothing_factor[1])
end

# Use Optim.jl to minimize the cost function
result = optimize(cost_fn, 0.0, 1.0)  # smoothing_factor is between 0 and 1

# Get the optimal smoothing factor
optimal_smoothing_factor = Optim.minimizer(result)[1]
println("Optimal smoothing factor: ", optimal_smoothing_factor)

# Fit the final spline with the optimal smoothing factor
final_spline = CubicSplineInterpolation(x, y, extrapolation_bc=Line())

# Plot the original data and the fitted spline
plot(x, y, label="Original data", marker=:o)
plot!(x, final_spline(x), label="Fitted spline", linewidth=2)
plot!(x,fit_spline)

# Fit the final spline with the optimal smoothing factor
final_spline = CubicSplineInterpolation(x, y, extrapolation_bc=Line())

sfitting = Spline1D(x,y,s=0.99)
listpi = collect(0:0.001:3π)

plot(x,y, seriestype=:scatter)
plot!(listpi,sfitting(listpi))






y_FurnaceToSlit+y_SlitToSG+y_SG+y_SGToScreen
