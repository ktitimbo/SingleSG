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
avg_data = load(joinpath(@__DIR__, "analysis_data", "smoothing_binning","data_averaged_2.jld2"), "data" );
I_exp  = avg_data[:i_smooth];
z_exp  = avg_data[:z_smooth];
δz_exp = avg_data[:δz_smooth];
##################################################################################################

# Coil currents
Icoils = [0.00,
            # 0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            # 0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
            # 0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
            # 0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,
            # 0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,
            1.00
];
nI = length(Icoils);

# Sample size: number of atoms arriving to the screen
const Nss = 1_000_000 ; 
@info "Number of MonteCarlo particles : $(Nss)\n"

nx_bins , nz_bins = 2 , 2
gaussian_width_mm = 0.250
λ0_raw            = 0.01
λ0_spline         = 0.001
norm_mode         = :probability

# Monte Carlo generation of particles traersing the filtering slit [x0 y0 z0 v0x v0y v0z]
crossing_slit = generate_samples(Nss, effusion_params; v_pdf=:v3, rng = rng_set, multithreaded = false, base_seed = base_seed_set);
jldsave( joinpath(OUTDIR,"cross_slit_particles_$(Nss).jld2"), data = crossing_slit)

if SAVE_FIG
    plot_μeff(K39_params,"mm_effective")
    plot_SG_geometry("SG_geometry")
    plot_velocity_stats(crossing_slit, "Initial data" , "velocity_pdf")
end


@inline function survives_kill_rect(x0,z0,vx,vy,vz, xmin,xmax,zmin,zmax)
    t_back  = (y_FurnaceToSlit + Δ_slit) / vy
    x1 = x0 + vx*t_back
    z1 = z0 + vz*t_back
    return (xmin <= x0 <= xmax) & (xmin <= x1 <= xmax) &
           (zmin <= z0 <= zmax) & (zmin <= z1 <= zmax)
end

@inline function fold_reflect(u, w)
    # Fold coordinate u onto [0,w] with specular reflection (triangular wave)
    tw = 2w
    m = mod(u, tw)              # m in [0, 2w)
    return (m <= w) ? m : (tw - m)
end

@inline function exit_state_elastic_rect(
    x0::T, y0::T, z0::T, vx::T, vy::T, vz::T,
    xmin::T, xmax::T, zmin::T, zmax::T) where {T}

    y1 = y_FurnaceToSlit
    y2 = y_FurnaceToSlit + Δ_slit

    # Advance from (x0, y0, z0) to the tunnel entrance at y1
    t_to_y1 = (y1 - y0) / vy
    x1 = x0 + vx * t_to_y1
    z1 = z0 + vz * t_to_y1

    # Time spent inside the tunnel (from y1 to y2)
    t_inside = (y2 - y1) / vy

    wx = xmax - xmin
    wz = zmax - zmin

    ux = (x1 - xmin) + vx * t_inside
    uz = (z1 - zmin) + vz * t_inside

    # Folded exit positions
    xf = xmin + fold_reflect(ux, wx)
    zf = zmin + fold_reflect(uz, wz)

    # Compute the number of full widths crossed (for velocity reflection)
    kx = floor(Int, (ux / wx))  # How many x-widths crossed
    kz = floor(Int, (uz / wz))  # How many z-widths crossed

    # Flip velocity signs if the number of reflections is odd
    vx_final = isodd(kx) ? -vx : vx
    vz_final = isodd(kz) ? -vz : vz

    return [xf, y2, zf, vx_final, vy, vz_final]
end


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----- no slit thickness ----------------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@time particles_flag = TheoreticalSimulation.QM_flag_travelling_particles(
                            Icoils, 
                            crossing_slit, 
                            K39_params; 
                            y_length=5001, 
                            verbose=true
);
@time particles_trajectories = TheoreticalSimulation.QM_build_travelling_particles(
                                Icoils,
                                crossing_slit,
                                particles_flag,
                                K39_params
);
TheoreticalSimulation.travelling_particles_summary(Icoils, quantum_numbers, particles_trajectories)
data_alive_screen = TheoreticalSimulation.QM_select_flagged(particles_trajectories,:screen);
alive_screen = OrderedDict(:Icoils=>Icoils, :levels => quantum_numbers , :data => data_alive_screen);

println("Profiles F=$(K39_params.Ispin+0.5)")
profiles_top = QM_analyze_profiles_to_dict(alive_screen, K39_params;
                    manifold=:F_top,    n_bins= (nx_bins , nz_bins), width_mm=gaussian_width_mm, add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth = λ0_spline, mode=norm_mode);
println("Profiles F=$(K39_params.Ispin-0.5)")
profiles_bottom = QM_analyze_profiles_to_dict(alive_screen, K39_params;
                    manifold=:F_bottom, n_bins= (nx_bins , nz_bins), width_mm=gaussian_width_mm, add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth = λ0_spline, mode=norm_mode);

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----- slit thickness + particles are removed after collision----------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Δ_slit = 400e-6
kill_slit1 = falses(Nss);
@threads for i=1:Nss
    x0  = crossing_slit[i,1]
    z0  = crossing_slit[i,3]
    v0x = crossing_slit[i,4]
    v0y = crossing_slit[i,5]
    v0z = crossing_slit[i,6]
    kill_slit1[i] = survives_kill_rect(x0,z0,v0x,v0y,v0z,-x_slit/2,x_slit/2,-z_slit/2,z_slit/2)
end
crossing_killed = crossing_slit[kill_slit1, :]
println("Particles removed after collisions with the slit walls (%) " , round(100*(1-size(crossing_killed,1)/Nss); sigdigits=2))

@time particles_flag_δslit = TheoreticalSimulation.QM_flag_travelling_particles(
                            Icoils, 
                            crossing_killed, 
                            K39_params; 
                            y_length=5001, 
                            verbose=true
);
@time particles_trajectories_δslit = TheoreticalSimulation.QM_build_travelling_particles(
                                Icoils,
                                crossing_killed,
                                particles_flag_δslit,
                                K39_params
);
TheoreticalSimulation.travelling_particles_summary(Icoils, quantum_numbers, particles_trajectories_δslit)
data_alive_screen_δslit = TheoreticalSimulation.QM_select_flagged(particles_trajectories_δslit,:screen);
alive_screen_δslit = OrderedDict(:Icoils=>Icoils, :levels => quantum_numbers , :data => data_alive_screen_δslit);

println("Profiles F=$(K39_params.Ispin+0.5)")
profiles_top_δslit = QM_analyze_profiles_to_dict(alive_screen_δslit, K39_params;
                    manifold=:F_top,    n_bins= (nx_bins , nz_bins), width_mm=gaussian_width_mm, add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth = λ0_spline, mode=norm_mode);
println("Profiles F=$(K39_params.Ispin-0.5)")
profiles_bottom_δslit = QM_analyze_profiles_to_dict(alive_screen_δslit, K39_params;
                    manifold=:F_bottom, n_bins= (nx_bins , nz_bins), width_mm=gaussian_width_mm, add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth = λ0_spline, mode=norm_mode);

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----- slit thickness + particles are reflected after collision--------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@inline function QM_EqOfMotion_slit_width(Ix,f,mf,r0::AbstractVector{<:Real},v0::AbstractVector{<:Real}, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"

    x0, y0, z0 = r0
    v0x, v0y, v0z = v0
    @assert v0y != 0.0 "y-velocity must be nonzero."

    μ =  TheoreticalSimulation.μF_effective(Ix,f,mf,p)
    acc_z    = μ * TheoreticalSimulation.GvsI(Ix) / p.M

    # Furnace - Slit
    x_f1_in = x0 + v0x/v0y * y_FurnaceToSlit
    y_f1_in = y0 + y_FurnaceToSlit
    z_f1_in = z0 + v0z/v0y * y_FurnaceToSlit
    vx_f1_in = v0x
    vy_f1_in = v0y
    vz_f1_in = v0z

    # Slit width
    x_f1_out , y_f1_out , z_f1_out , vx_f1_out , vy_f1_out , vz_f1_out = exit_state_elastic_rect(x0, y0, z0, v0x, v0y, v0z,-x_slit/2,x_slit/2,-z_slit/2,z_slit/2)

    # Slit back side to SG entrance
    x_SG_in = x_f1_out + vx_f1_out / vy_f1_out * (y_SlitToSG - Δ_slit)
    y_SG_in = y_f1_in + y_SlitToSG
    z_SG_in = z_f1_out + vz_f1_out / vy_f1_out * (y_SlitToSG - Δ_slit)
    vx_SG_in = vx_f1_out
    vy_SG_in = vy_f1_out
    vz_SG_in = vz_f1_out

    # SG magnet exit
    x_SG_out = x_SG_in + vx_SG_in / vy_SG_in * y_SG
    y_SG_out = y_SG_in + y_SG
    z_SG_out = z_SG_in + vz_SG_in / vy_SG_in * y_SG + 0.5 * acc_z * (y_SG / vy_SG_in)^2
    vx_SG_out = vx_SG_in
    vy_SG_out = vy_SG_in
    vz_SG_out = vz_SG_in + acc_z * y_SG / vy_SG_in

    # Circular aperture
    x_aperture = x_SG_out + vx_SG_out / vy_SG_out * y_SGToAperture
    y_aperture = y_SG_out + y_SGToAperture
    z_aperture = z_SG_out + vz_SG_out / vy_SG_out * y_SGToAperture
    vx_aperture = vx_SG_out
    vy_aperture = vy_SG_out
    vz_aperture = vz_SG_out
    if x_aperture*x_aperture + z_aperture*z_aperture >= R_aper^2
        r = SVector{3,Float64}(4.0, 4.0, 4.0)
        v = SVector{3,Float64}(4.0, 4.0, 4.0)
        return r, v
    end

    # Screen
    x_screen = x_SG_out + vx_SG_out / vy_SG_out * y_SGToScreen
    y_screen = y_SG_out + y_SGToScreen
    z_screen = z_SG_out + vz_SG_out / vy_SG_out * y_SGToScreen

    vx_screen = vx_SG_out
    vy_screen = vy_SG_out
    vz_screen = vz_SG_out

    if x_aperture*x_aperture + z_aperture*z_aperture >= R_aper^2
        r = SVector{3,Float64}(3.0, 3.0, 3.0)
        v = SVector{3,Float64}(3.0, 3.0, 3.0)
        return r, v
    end
    
    r = SVector{3,Float64}(x_screen, y_screen, z_screen)
    v = SVector{3,Float64}(vx_screen, vy_screen, vz_screen)
    return r, v
end

crossing_bounced = zeros(Nss,6,3)
k= 0
for (f,mf) in quantum_numbers[6:8]
    k+=1
    for i=1:Nss
        x0  = crossing_slit[i,1]
        y0  = crossing_slit[i,2]
        z0  = crossing_slit[i,3]
        v0x = crossing_slit[i,4]
        v0y = crossing_slit[i,5]
        v0z = crossing_slit[i,6]

        screen = QM_EqOfMotion_slit_width(Icoils[1],f,mf,[x0,y0,z0],[v0x,v0y,v0z],K39_params)
        crossing_bounced[i,1:3,k] = screen[1]
        crossing_bounced[i,4:6,k] = screen[2]
    end
end
A = vcat([crossing_bounced[:,:,l] for l=1:3]...)
mask = trues(size(A,1)) ;  # keep row?
@inbounds for i in axes(A,1)
    row = @view A[i, :]

    all4 = true
    all3 = true

    for x in row
        all4 &= (x == 4.0)
        all3 &= (x == 3.0)
    end

    mask[i] = !(all4 || all3)
end
crossing_reflection = A[mask, :]
x_ref_screen = 1e3*crossing_reflection[:,1]
z_ref_screen = 1e3*crossing_reflection[:,3]
extrema(z_ref_screen)

bins_aper  = (TheoreticalSimulation.FreedmanDiaconisBins(x_ref_screen), TheoreticalSimulation.FreedmanDiaconisBins(z_ref_screen))
# Aperture
fig_up = histogram2d(x_ref_screen, z_ref_screen;
    bins=bins_aper, show_empty_bins=true, color=:plasma, normalize=:pdf,
    xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{mm})$",
    # xticks=-4.0:0.50:4.0, yticks=-1000:500:3000,
    # xlims=(-3.5,3.5), ylims=(-3.5,3.5),
    # aspect_ratio=:equal
)

# Fixed analysis limits
xlim = (-8.0, 8.0);
zlim = (-12.5, 12.5);
xmin, xmax = xlim ;
zmin, zmax = zlim ;

# Bin size in mm (default_camera_pixel_size is assumed global in meters)
x_bin_size = 1e3 * nx_bins * cam_pixelsize ;
z_bin_size = 1e3 * nz_bins * cam_pixelsize ;

# --------------------------------------------------------
# X edges: force symmetric centers around 0
# --------------------------------------------------------
x_half_range = max(abs(xmin), abs(xmax)) ;
kx = max(1, ceil(Int, x_half_range / x_bin_size)) ;
centers_x = collect((-kx:kx) .* x_bin_size) ;
edges_x = collect((-(kx + 0.5)) * x_bin_size : x_bin_size : ((kx + 0.5) * x_bin_size)) ;
# --------------------------------------------------------
# Z edges: force symmetric centers around 0
# --------------------------------------------------------
z_half_range = max(abs(zmin), abs(zmax));
kz = max(1, ceil(Int, z_half_range / z_bin_size));
centers_z = collect((-kz:kz) .* z_bin_size);
edges_z = collect((-(kz + 0.5)) * z_bin_size : z_bin_size : ((kz + 0.5) * z_bin_size));

if norm_mode === :none
    h_qm_f1 = StatsBase.fit(Histogram, (x_ref_screen, z_ref_screen), (edges_x, edges_z))
elseif norm_mode in (:probability, :pdf, :density)
    h_qm_f1 = normalize(StatsBase.fit(Histogram, (x_ref_screen, z_ref_screen), (edges_x, edges_z)); mode=norm_mode)
end


counts_qm_f1 = h_qm_f1.weights ; # size: (length(centers_x), length(centers_z))
qm_profile_f1 = vec(mean(counts_qm_f1, dims=1));
z_max_qm_mm_f1 = centers_z[argmax(qm_profile_f1)]
z_max_qm_spline_mm_f1, Sfit_qm_f1 = TheoreticalSimulation.max_of_bspline_positions(centers_z,qm_profile_f1;λ0=λ0_raw)


plot(profiles_top[1][:z_profile][:,1],profiles_top[1][:z_profile][:,2],
    label="No slit thickness",
    seriestype=:scatter,
    marker=(:circle,2,:white),
    markerstrokecolor=:blue)
plot!(profiles_top_δslit[1][:z_profile][:,1],profiles_top_δslit[1][:z_profile][:,2],
    label="Remove after collision | thickness $(1e3*Δ_slit)mm",
    line=(:solid,2,:orangered))
plot!(xlims=(-1.5,1.5))
plot!(centers_z, qm_profile_f1,
    title=L"$z$–profile at the detection",
    label="Reflections after collision",
    xlabel=L"$z$ (mm)",
    line=(:solid,2,:green),
    legend=:outerbottom,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
    xticks=-4:1:4,
)


##################################################################################################
#   QUANTUM MECHANICS
##################################################################################################
@time particles_flag  = TheoreticalSimulation.QM_flag_travelling_particles(
                            Icoils, 
                            crossing_slit, 
                            K39_params; 
                            y_length=5001, 
                            verbose=true
);
@time particles_trajectories = TheoreticalSimulation.QM_build_travelling_particles(
                                Icoils,
                                crossing_slit,
                                particles_flag,
                                K39_params
);
TheoreticalSimulation.travelling_particles_summary(Icoils, quantum_numbers, particles_trajectories)
jldsave( joinpath(OUTDIR,"qm_$(Nss)_valid_particles_data.jld2"), data = OrderedDict(:Icoils => Icoils, :levels => quantum_numbers , :data => particles_trajectories))

data_alive_screen = TheoreticalSimulation.QM_select_flagged(particles_trajectories,:screen);
alive_screen = OrderedDict(:Icoils=>Icoils, :levels => quantum_numbers , :data => data_alive_screen);
jldsave(joinpath(OUTDIR,"qm_$(Nss)_screen_data.jld2"), alive = alive_screen)

############### data saved in block format for easier access ###############
jldopen(joinpath(OUTDIR,"qm_screen_data.jld2"), "w") do file
    file["meta/Icoils"] = Icoils
    file["meta/levels"] = quantum_numbers 
    for i in 1:nI
        file["screen/I$(i)"] = data_alive_screen[i]  
    end
end
############################################################################
# fname = joinpath(OUTDIR,"qm_$(Nss)_screen_data.jld2")
#  jldopen(fname, "r") do file
#         file["screen/i1"]  
# end
# jldopen(fname, "r") do f
#     println("top:   ", collect(keys(f)))
#     println("meta:  ", collect(keys(f["meta"])))
#     println("screen:", collect(keys(f["screen"])))
# end


# clean memory
crossing_slit           = nothing
particles_flag          = nothing
particles_trajectories  = nothing
GC.gc()
@info "Memory cleaned after QM data acquired"
@info "Free system memory $(round(Sys.free_memory() / 1024^3,digits=1)) GiB"


println("Profiles F=$(K39_params.Ispin+0.5)")
profiles_top = QM_analyze_profiles_to_dict(alive_screen, K39_params;
                    manifold=:F_top,    n_bins= (nx_bins , nz_bins), width_mm=gaussian_width_mm, add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth = λ0_spline, mode=:probability);
println("Profiles F=$(K39_params.Ispin-0.5)")
profiles_bottom = QM_analyze_profiles_to_dict(alive_screen, K39_params;
                    manifold=:F_bottom, n_bins= (nx_bins , nz_bins), width_mm=gaussian_width_mm, add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth = λ0_spline, mode=:probability);
println("Profiles F=$(K39_params.Ispin+0.5), mf=$(-(K39_params.Ispin+0.5))")
profiles_5      = QM_analyze_profiles_to_dict(alive_screen, K39_params;
                    manifold=5,         n_bins= (nx_bins , nz_bins), width_mm=gaussian_width_mm, add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth = λ0_spline, mode=:probability);
println("Profiles ms=$((K39_params.Ispin))")
profiles_Sup  = QM_analyze_profiles_to_dict(alive_screen, K39_params;
                    manifold=:S_up,     n_bins= (nx_bins , nz_bins), width_mm=gaussian_width_mm, add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth = λ0_spline, mode=:probability);
println("Profiles ms=$(-(K39_params.Ispin))")
profiles_Sdown  = QM_analyze_profiles_to_dict(alive_screen, K39_params;
                    manifold=:S_down,   n_bins= (nx_bins , nz_bins), width_mm=gaussian_width_mm, add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth = λ0_spline, mode=:probability);
jldsave(joinpath(OUTDIR,"qm_$(Nss)_screen_profiles.jld2"), profiles = OrderedDict(
                                                                    :nz_bins    => nz_bins,
                                                                    :gauss_w    => gaussian_width_mm,
                                                                    :smothing   => (λ0_raw,λ0_spline),
                                                                    :upper      => profiles_top, 
                                                                    :lower      => profiles_bottom, 
                                                                    :Sup        => profiles_Sup, 
                                                                    :Sdown      => profiles_Sdown, 
                                                                    :lvl5       => profiles_5) 
                                                                    )

# Profiles : different contributions
anim = @animate for j in eachindex(Icoils)
    pretty_table(permutedims(round.([μF_effective(Icoils[j],v[1],v[2],K39_params)/μB for v in quantum_numbers ],sigdigits=4)),
        column_label_alignment      = :c,
        column_labels               = quantum_numbers,
        formatters                  = [ fmt__printf("%2.4f", 1:8)],
        stubhead_label              = "μ/μ₀",
        alignment                   = :c,
        equal_data_column_widths    = true,
        row_label_column_alignment  = :c,
        row_group_label_alignment   = :c,
        title                       = "QUANTUM MECHANICS APPROACH μ/μB (I₀=$(@sprintf("%d",1000*Icoils[j]))mA)",
        table_format                = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style                       = TextTableStyle(
                                            first_line_merged_column_label  = crayon"light_red bold",
                                            first_line_column_label         = crayon"yellow bold",
                                            column_label                    = crayon"yellow",
                                            table_border                    = crayon"blue bold",
                                            title                           = crayon"red bold"
                                        ),
    
    )
    fig = plot(
        title="Quantum mechanics profiles",
        legend=:topleft,
        legendtitle=L"$I_{0}=%$(Icoils[j])\mathrm{A}$",
        legendtitlefontsize=8,
        yformatter = val -> string(round(val * 1e4, digits = 2)),
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",)
    plot!(profiles_top[j][:z_profile][:,1],profiles_top[j][:z_profile][:,3],
        label=L"$F=2$",
        line=(:solid,:grey66,1),
        marker=(:circle,:white,2),
        markerstrokecolor=:grey66,
        markerstrokewidth=1)
    vline!([profiles_top[j][:z_max_smooth_spline_mm]], 
        line=(:grey66,0.5), 
        label=L"$z_{\mathrm{max}}=%$(round(profiles_top[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
    plot!(profiles_Sup[j][:z_profile][:,1],profiles_Sup[j][:z_profile][:,3],
        label=L"$m_{s}=+1/2$",
        line=(:solid,:seagreen3,1),
        marker=(:circle,:white,2),
        markerstrokecolor=:seagreen3,
        markerstrokewidth=1)
    vline!([profiles_Sup[j][:z_max_smooth_spline_mm]],
        line=(:seagreen3,0.5), 
        label=L"$z_{\mathrm{max}}=%$(round(profiles_Sup[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
    plot!(profiles_bottom[j][:z_profile][:,1],profiles_bottom[j][:z_profile][:,3],
        label=L"$F=1$",
        line=(:solid,:blueviolet,1),
        marker=(:circle,:white,2),
        markerstrokecolor=:blueviolet,
        markerstrokewidth=1)
    vline!([profiles_bottom[j][:z_max_smooth_spline_mm]], 
        line=(:blueviolet,0.5), 
        label=L"$z_{\mathrm{max}}=%$(round(profiles_bottom[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
    plot!(profiles_Sdown[j][:z_profile][:,1],profiles_Sdown[j][:z_profile][:,3],
        label=L"$m_{s}=-1/2$",
        line=(:solid,:orangered2,1),
        marker=(:circle,:white,2),
        markerstrokecolor=:orangered2,
        markerstrokewidth=1)
    vline!([profiles_Sdown[j][:z_max_smooth_spline_mm]],
        line=(:orangered2,0.5), 
        label=L"$z_{\mathrm{max}}=%$(round(profiles_Sdown[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
    plot!(profiles_5[j][:z_profile][:,1],profiles_5[j][:z_profile][:,3],
        label=L"$F=2$, $m_{F}=-2$",
        line=(:solid,:dodgerblue3,1),
        marker=(:circle,:white,2),
        markerstrokecolor=:dodgerblue3,
        markerstrokewidth=1)
    vline!([profiles_5[j][:z_max_smooth_spline_mm]],
        line=(:dodgerblue3,0.5), 
        label=L"$z_{\mathrm{max}}=%$(round(profiles_5[j][:z_max_smooth_spline_mm],sigdigits=3)) \mathrm{mm}$")
    display(fig)
end
gif_path = joinpath(OUTDIR, "QM_profiles.gif");
gif(anim, gif_path, fps=2)  # adjust fps 
@info "Saved GIF" gif_path ;
anim = nothing

# Peak position (mm) : lower branch
fig=plot(xlabel=L"$I_{c}$ (A)", ylabel=L"$z_{\mathrm{max}}$ (mm)")
plot!(fig,Icoils[2:end], [profiles_5[i][:z_max_smooth_spline_mm] for i in eachindex(Icoils)][2:end],
    label=L"Trajectory $\vert 2,-2\rangle$",
    line=(:solid,:red,2))
plot!(fig,Icoils[2:end], [profiles_bottom[i][:z_max_smooth_spline_mm] for i in eachindex(Icoils)][2:end],
    label=L"Trajectory $F=1$",
    line=(:solid,:blue,2))
plot!(fig,Icoils[2:end], [profiles_Sdown[i][:z_max_smooth_spline_mm] for i in eachindex(Icoils)][2:end],
    label=L"Trajectory $m_{s}=-1/2$",
    line=(:solid,:purple,2))
plot!(fig,I_exp[2:end],z_exp[2:end],
    ribbon=δz_exp[5:end],
    label="Experiment (combined)",
    line=(:black,:dash,2),
    fillalpha=0.23, 
    fillcolor=:black, 
    )
plot!(fig,xaxis=:log10,
    yaxis=:log10,
    xlims=(0.8e-3,2),
    ylims=(0.8e-4,2),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], 
            [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:topleft,
    left_margin =2mm,
)
display(fig)
savefig(fig,joinpath(OUTDIR,"QM_results_comparison.$(FIG_EXT)"))

# ATOMS PROPAGATION
r = 1:1:nI;
iter = (isempty(r) || last(r) == nI) ? r : Iterators.flatten((r, (nI,)));
lvl = 5 #Int(4*K39_params.Ispin+2)
f,mf=quantum_numbers[lvl]
# =========================
# Precompute geometry overlays (constant)
# =========================
x_magnet    = 1e-3 .* range(-1.0, 1.0, length=1000)  # m
z_edge_um   = 1e6 .* TheoreticalSimulation.z_magnet_edge.(x_magnet) # μm
x_magnet_mm = 1e3 .* x_magnet                        # mm

R_mm = 1e3 * R_aper
θcirc = range(0, 2π, length=361)
x_circ_mm = R_mm .* cos.(θcirc)                      # mm
z_circ_um = (1e3 * R_mm) .* sin.(θcirc)              # μm

# =========================
# Precompute stage distances (divide by v0y inside loop)
# =========================
y_slit   = y_FurnaceToSlit
y_sg_in  = y_FurnaceToSlit + y_SlitToSG
y_sg_out = y_sg_in + y_SG
y_aper   = y_sg_out + y_SGToAperture
y_scr    = y_sg_out + y_SGToScreen

anim = @animate for j in iter
    data_set = alive_screen[:data][j][lvl];
    n = size(data_set, 1) ; 
    
    # --- preallocate arrays for histograms (scaled units) ---
    xs_a = Vector{Float64}(undef, n); zs_a = Vector{Float64}(undef, n)  # furnace (mm, μm)
    xs_b = Vector{Float64}(undef, n); zs_b = Vector{Float64}(undef, n)  # slit (mm, μm)
    xs_c = Vector{Float64}(undef, n); zs_c = Vector{Float64}(undef, n)  # SG in (mm, μm)
    xs_d = Vector{Float64}(undef, n); zs_d = Vector{Float64}(undef, n)  # SG out (mm, μm)
    xs_f = Vector{Float64}(undef, n); zs_f = Vector{Float64}(undef, n)  # aperture (mm, μm)
    xs_e = Vector{Float64}(undef, n); zs_e = Vector{Float64}(undef, n)  # screen (mm, mm)

    # --- one pass: compute all stages for each particle ---
    @inbounds for i in 1:n
        v0y = data_set[i,5]

        # Furnace (just the initial plane)
        xs_a[i] = 1e3 * data_set[i,1]  # mm
        zs_a[i] = 1e6 * data_set[i,3]  # μm

        τ_slit  = y_slit  / v0y
        τ_sgin  = y_sg_in / v0y
        τ_sgout = y_sg_out / v0y
        τ_aper  = y_aper  / v0y
        τ_scr   = y_scr   / v0y

        # NOTE: keep your original calling convention (views)
        rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(τ_slit,  Icoils[j], f, mf,
                                                      @view(data_set[i,1:3]),
                                                      @view(data_set[i,4:6]),
                                                      K39_params)
        xs_b[i] = 1e3 * rtmp[1] ; zs_b[i] = 1e6 * rtmp[3]

        rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(τ_sgin,  Icoils[j], f, mf,
                                                      @view(data_set[i,1:3]),
                                                      @view(data_set[i,4:6]),
                                                      K39_params)
        xs_c[i] = 1e3 * rtmp[1] ; zs_c[i] = 1e6 * rtmp[3]

        rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(τ_sgout, Icoils[j], f, mf,
                                                      @view(data_set[i,1:3]),
                                                      @view(data_set[i,4:6]),
                                                      K39_params)
        xs_d[i] = 1e3 * rtmp[1] ; zs_d[i] = 1e6 * rtmp[3]

        rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(τ_aper,  Icoils[j], f, mf,
                                                      @view(data_set[i,1:3]),
                                                      @view(data_set[i,4:6]),
                                                      K39_params)
        xs_f[i] = 1e3 * rtmp[1] ; zs_f[i] = 1e6 * rtmp[3]

        rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(τ_scr,   Icoils[j], f, mf,
                                                      @view(data_set[i,1:3]),
                                                      @view(data_set[i,4:6]),
                                                      K39_params)
        xs_e[i] = 1e3 * rtmp[1] ; zs_e[i] = 1e3 * rtmp[3]
    end

    bins_furn  = (FreedmanDiaconisBins(xs_a), FreedmanDiaconisBins(zs_a))
    bins_slit  = (FreedmanDiaconisBins(xs_b), FreedmanDiaconisBins(zs_b))
    bins_sgin  = (FreedmanDiaconisBins(xs_c), FreedmanDiaconisBins(zs_c))
    bins_sgout = (FreedmanDiaconisBins(xs_d), FreedmanDiaconisBins(zs_d))
    bins_aper  = (FreedmanDiaconisBins(xs_f), FreedmanDiaconisBins(zs_f))
    bins_scr   = (FreedmanDiaconisBins(xs_e), FreedmanDiaconisBins(zs_e))

    # Furnace
    figa = histogram2d(xs_a, zs_a;
        bins=bins_furn, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{\mu m})$",
        xticks=-1.0:0.25:1.0, yticks=-50:25:50,
    );
    xpos, ypos = -0.75, 35; dx, dy = 0.15, 7;
    plot!(figa, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                      [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
          color=:white, opacity=0.65, linealpha=0, label=false);
    annotate!(figa, xpos, ypos, text("Furnace", 10, :black, :bold, :center, "Helvetica"));

    # Slit
    figb = histogram2d(xs_b, zs_b;
        bins=bins_slit, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{\mu m})$",
        xticks=-4.0:0.50:4.0, yticks=-200:50:200,
        xlims=(-4,4), ylims=(-200,200),
    );
    xpos, ypos = -3.5, 150; dx, dy = 0.4, 20
    plot!(figb, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                      [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
          color=:white, opacity=0.65, linealpha=0, label=false);
    annotate!(figb, xpos, ypos, text("Slit", 10, :black, :bold, :center, "Helvetica"));

    # SG entrance
    figc = histogram2d(xs_c, zs_c;
        bins=bins_sgin, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{\mu m})$",
        xticks=-4.0:0.50:4.0, yticks=-1000:100:1000,
        xlims=(-4,4), ylims=(-250,250),
    )
    xpos, ypos = -3.0, 180; dx, dy = 0.8, 30
    plot!(figc, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                      [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
          color=:white, opacity=0.65, linealpha=0, label=false)
    annotate!(figc, xpos, ypos, text("SG entrance", 10, :black, :bold, :center, "Helvetica"))

    # SG exit
    figd = histogram2d(xs_d, zs_d;
        bins=bins_sgout, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{\mu m})$",
        xticks=-4.0:0.50:4.0, yticks=-1000:200:1000,
        xlims=(-4,4), ylims=(-300,1000),
    );
    plot!(figd, x_magnet_mm, z_edge_um, line=(:dash,:black,2), label=false);
    xpos, ypos = -3.0, 700; dx, dy = 0.6, 160
    plot!(figd, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                      [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
          color=:white, opacity=0.65, linealpha=0, label=false);
    annotate!(figd, xpos, ypos, text("SG exit", 10, :black, :bold, :center, "Helvetica"));

    # Aperture
    figf = histogram2d(xs_f, zs_f;
        bins=bins_aper, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{\mu m})$",
        xticks=-4.0:0.50:4.0, yticks=-1000:500:3000,
        xlims=(-4,4), ylims=(-300,3000),
    )
    plot!(figf, x_circ_mm, z_circ_um, line=(:dash,:gray,2), label=false)
    xpos, ypos = -3.0, 2400; dx, dy = 0.7, 270
    plot!(figf, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                      [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
          color=:white, opacity=0.65, linealpha=0, label=false)
    annotate!(figf, xpos, ypos, text("⊚ Aperture", 10, :black, :bold, :center, "Helvetica"))

    # Screen
    fige = histogram2d(xs_e, zs_e;
        bins=bins_scr, show_empty_bins=true, color=:plasma, normalize=:pdf,
        xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{mm})$",
        ylims=(-1,17.5),
    );
    xpos, ypos = -4.0, 14; dx, dy = 0.9, 0.9
    plot!(fige, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                      [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
          color=:white, opacity=0.65, linealpha=0, label=false);
    annotate!(fige, xpos, ypos, text("Screen", 10, :black, :bold, :center, "Helvetica"));

    fig = plot(figa,figb,figc,figd,figf,fige,
    layout=(6,1),
    suptitle = L"$I_{0} = %$(Int(1000*Icoils[j]))\,\mathrm{mA}$",
    size=(750,850),
    right_margin=2mm,
    bottom_margin=-2mm,
    )
    plot!(fig[1], xlabel="", bottom_margin=-3mm);
    plot!(fig[2], xlabel="", bottom_margin=-3mm);
    plot!(fig[3], xlabel="", bottom_margin=-3mm);
    plot!(fig[4], xlabel="", bottom_margin=-3mm);
    plot!(fig[5], xlabel="", bottom_margin=-3mm);
    display(fig)
end
gif_path = joinpath(OUTDIR, "QM_time_evolution.gif");
gif(anim, gif_path, fps=2)  # adjust fps
@info "Saved GIF" gif_path ;
anim = nothing

nz_bins_list = [1,2,4];
ls_list = [:solid,:dash,:dot];
gaussian_width_mm_list = [0.050, 0.065, 0.100, 0.150, 0.200, 0.300, 0.400, 0.500];
zmax_gaussian_width = zeros(length(nz_bins_list),nI,length(gaussian_width_mm_list));
for (jdx,nz) in enumerate(nz_bins_list)
    for (idx,val) in enumerate(gaussian_width_mm_list)

        profiles_bottom_temp = QM_analyze_profiles_to_dict(alive_screen, K39_params;
                        manifold=:F_bottom, n_bins= (nx_bins , nz), width_mm=val, add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth = λ0_spline, mode=:probability);


        zmax_gaussian_width[jdx,:,idx] = [profiles_bottom_temp[i][:z_max_smooth_spline_mm] for i in eachindex(Icoils)]

    end
end

fig=plot(title = L"Quantum Mechanics : $F=1$",
    xlabel="Currents (A)",
    ylabel=L"$z_{\mathrm{max}}$ (mm)",);
Ic_start_idx = findall(>(0.010), Icoils)[1];
cls = palette(:darkrainbow,length(gaussian_width_mm_list));
for (jdx,nz) in enumerate(nz_bins_list)
    for (idx,val) in enumerate(gaussian_width_mm_list)
        plot!(fig, 
        Icoils[Ic_start_idx:end], zmax_gaussian_width[jdx,Ic_start_idx:end,idx],
        label = L"$n_{z}=%$(nz_bins_list[jdx])$ | $w=%$(1e3*gaussian_width_mm_list[idx])\,\mathrm{\mu m}$",
        line=(ls_list[jdx],cls[idx],1.5),)
    end
end
plot!(fig,I_exp[2:end], z_exp[2:end],
    ribbon=δz_exp[5:end],
    label="Experiment (combined)",
    line=(:black,:dash,2),
    fillalpha=0.23, 
    fillcolor=:black, 
    )
plot!(fig,
    xaxis=:log10,
    yaxis=:log10,
    xlims=(8e-3,2),
    ylims=(8e-3,2),
    xticks = ([ 1e-2, 1e-1, 1.0], 
            [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([ 1e-2, 1e-1, 1.0], 
            [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    size=(850,650),
    rightmargin=5mm,
    legend=:outerright,
    legend_columns = 4,)
display(fig)
savefig(fig,joinpath(OUTDIR,"qm_comparison_w_n.$(FIG_EXT)"))


#########################################################################################
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
    Number of atoms         : $(Nss)
    Binning (nx,nz)         : ($(nx_bins),$(nz_bins))
    Gaussian width (mm)     : $(gaussian_width_mm)
    Smoothing raw           : $(λ0_raw)
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
open(joinpath(OUTDIR,"simulation_report.txt"), "w") do io
    write(io, report)
end


println("script $RUN_STAMP has finished!")
alert("script $RUN_STAMP has finished!")

GC.gc()
@info "Memory cleaned after QM data acquired"
println("Free memory: $(Sys.free_memory() / (1024)^3) GiB")

Ns = 7_000_000
const OUTDIR = joinpath(@__DIR__,"simulation_data","quantum_simulation_$(Int(1e-6*Ns))M")
data_exists = isfile(joinpath(OUTDIR,"qm_$(Ns)_screen_data.jld2"))

if !data_exists
    println("Analyzing data arriving to the screen")

    dataQM = load(joinpath(OUTDIR,"qm_$(Ns)_valid_particles_data.jld2"))["data"]
    @time alive_screen = OrderedDict(
                :Icoils => dataQM[:Icoils],
                :levels => dataQM[:levels], 
                :data   => TheoreticalSimulation.QM_select_flagged(dataQM[:data],:screen));
    jldsave(joinpath(OUTDIR,"qm_$(Ns)_screen_data.jld2"), alive = alive_screen)
    dataQM = nothing
    GC.gc()
    @info "Memory cleaned after QM data acquired"
else    
    # data analysis
    println("QM approach : peak position data analysis") 

    alive_screen = load(joinpath(OUTDIR,"qm_$(Ns)_screen_data.jld2"))["alive"];
    @info "file loaded"

    Icoils  = alive_screen[:Icoils];
    nI      = length(Icoils);
    quantum_numbers = alive_screen[:levels];


    # ATOMS PROPAGATION
    r = 1:1:nI;
    iter = (isempty(r) || last(r) == nI) ? r : Iterators.flatten((r, (nI,)));
    lvl = 5 #Int(4*K39_params.Ispin+2)
    f,mf=quantum_numbers[lvl]
    # =========================
    # Precompute geometry overlays (constant)
    # =========================
    x_magnet    = 1e-3 .* range(-1.0, 1.0, length=1000)  # m
    z_edge_um   = 1e6 .* TheoreticalSimulation.z_magnet_edge.(x_magnet) # μm
    x_magnet_mm = 1e3 .* x_magnet                        # mm

    R_mm = 1e3 * R_aper
    θcirc = range(0, 2π, length=361)
    x_circ_mm = R_mm .* cos.(θcirc)                      # mm
    z_circ_um = (1e3 * R_mm) .* sin.(θcirc)              # μm

    # =========================
    # Precompute stage distances (divide by v0y inside loop)
    # =========================
    y_slit   = y_FurnaceToSlit
    y_sg_in  = y_FurnaceToSlit + y_SlitToSG
    y_sg_out = y_sg_in + y_SG
    y_aper   = y_sg_out + y_SGToAperture
    y_scr    = y_sg_out + y_SGToScreen

    anim = @animate for j in iter
        data_set = alive_screen[:data][j][lvl];
        n = size(data_set, 1) ; 
        
        # --- preallocate arrays for histograms (scaled units) ---
        xs_a = Vector{Float64}(undef, n); zs_a = Vector{Float64}(undef, n)  # furnace (mm, μm)
        xs_b = Vector{Float64}(undef, n); zs_b = Vector{Float64}(undef, n)  # slit (mm, μm)
        xs_c = Vector{Float64}(undef, n); zs_c = Vector{Float64}(undef, n)  # SG in (mm, μm)
        xs_d = Vector{Float64}(undef, n); zs_d = Vector{Float64}(undef, n)  # SG out (mm, μm)
        xs_f = Vector{Float64}(undef, n); zs_f = Vector{Float64}(undef, n)  # aperture (mm, μm)
        xs_e = Vector{Float64}(undef, n); zs_e = Vector{Float64}(undef, n)  # screen (mm, mm)

        # --- one pass: compute all stages for each particle ---
        @inbounds for i in 1:n
            v0y = data_set[i,5]

            # Furnace (just the initial plane)
            xs_a[i] = 1e3 * data_set[i,1]  # mm
            zs_a[i] = 1e6 * data_set[i,3]  # μm

            τ_slit  = y_slit  / v0y
            τ_sgin  = y_sg_in / v0y
            τ_sgout = y_sg_out / v0y
            τ_aper  = y_aper  / v0y
            τ_scr   = y_scr   / v0y

            # NOTE: keep your original calling convention (views)
            rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(τ_slit,  Icoils[j], f, mf,
                                                        @view(data_set[i,1:3]),
                                                        @view(data_set[i,4:6]),
                                                        K39_params)
            xs_b[i] = 1e3 * rtmp[1] ; zs_b[i] = 1e6 * rtmp[3]

            rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(τ_sgin,  Icoils[j], f, mf,
                                                        @view(data_set[i,1:3]),
                                                        @view(data_set[i,4:6]),
                                                        K39_params)
            xs_c[i] = 1e3 * rtmp[1] ; zs_c[i] = 1e6 * rtmp[3]

            rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(τ_sgout, Icoils[j], f, mf,
                                                        @view(data_set[i,1:3]),
                                                        @view(data_set[i,4:6]),
                                                        K39_params)
            xs_d[i] = 1e3 * rtmp[1] ; zs_d[i] = 1e6 * rtmp[3]

            rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(τ_aper,  Icoils[j], f, mf,
                                                        @view(data_set[i,1:3]),
                                                        @view(data_set[i,4:6]),
                                                        K39_params)
            xs_f[i] = 1e3 * rtmp[1] ; zs_f[i] = 1e6 * rtmp[3]

            rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(τ_scr,   Icoils[j], f, mf,
                                                        @view(data_set[i,1:3]),
                                                        @view(data_set[i,4:6]),
                                                        K39_params)
            xs_e[i] = 1e3 * rtmp[1] ; zs_e[i] = 1e3 * rtmp[3]
        end

        bins_furn  = (FreedmanDiaconisBins(xs_a), FreedmanDiaconisBins(zs_a))
        bins_slit  = (FreedmanDiaconisBins(xs_b), FreedmanDiaconisBins(zs_b))
        bins_sgin  = (FreedmanDiaconisBins(xs_c), FreedmanDiaconisBins(zs_c))
        bins_sgout = (FreedmanDiaconisBins(xs_d), FreedmanDiaconisBins(zs_d))
        bins_aper  = (FreedmanDiaconisBins(xs_f), FreedmanDiaconisBins(zs_f))
        bins_scr   = (FreedmanDiaconisBins(xs_e), FreedmanDiaconisBins(zs_e))

        # Furnace
        figa = histogram2d(xs_a, zs_a;
            bins=bins_furn, show_empty_bins=true, color=:plasma, normalize=:pdf,
            xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{\mu m})$",
            xticks=-1.0:0.25:1.0, yticks=-50:25:50,
        );
        xpos, ypos = -0.75, 35; dx, dy = 0.15, 7;
        plot!(figa, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                        [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
            color=:white, opacity=0.65, linealpha=0, label=false);
        annotate!(figa, xpos, ypos, text("Furnace", 10, :black, :bold, :center, "Helvetica"));

        # Slit
        figb = histogram2d(xs_b, zs_b;
            bins=bins_slit, show_empty_bins=true, color=:plasma, normalize=:pdf,
            xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{\mu m})$",
            xticks=-4.0:0.50:4.0, yticks=-200:50:200,
            xlims=(-4,4), ylims=(-200,200),
        );
        xpos, ypos = -3.5, 150; dx, dy = 0.4, 20
        plot!(figb, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                        [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
            color=:white, opacity=0.65, linealpha=0, label=false);
        annotate!(figb, xpos, ypos, text("Slit", 10, :black, :bold, :center, "Helvetica"));

        # SG entrance
        figc = histogram2d(xs_c, zs_c;
            bins=bins_sgin, show_empty_bins=true, color=:plasma, normalize=:pdf,
            xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{\mu m})$",
            xticks=-4.0:0.50:4.0, yticks=-1000:100:1000,
            xlims=(-4,4), ylims=(-250,250),
        )
        xpos, ypos = -3.0, 180; dx, dy = 0.8, 30
        plot!(figc, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                        [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
            color=:white, opacity=0.65, linealpha=0, label=false)
        annotate!(figc, xpos, ypos, text("SG entrance", 10, :black, :bold, :center, "Helvetica"))

        # SG exit
        figd = histogram2d(xs_d, zs_d;
            bins=bins_sgout, show_empty_bins=true, color=:plasma, normalize=:pdf,
            xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{\mu m})$",
            xticks=-4.0:0.50:4.0, yticks=-1000:200:1000,
            xlims=(-4,4), ylims=(-300,1000),
        );
        plot!(figd, x_magnet_mm, z_edge_um, line=(:dash,:black,2), label=false);
        xpos, ypos = -3.0, 700; dx, dy = 0.6, 160
        plot!(figd, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                        [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
            color=:white, opacity=0.65, linealpha=0, label=false);
        annotate!(figd, xpos, ypos, text("SG exit", 10, :black, :bold, :center, "Helvetica"));

        # Aperture
        figf = histogram2d(xs_f, zs_f;
            bins=bins_aper, show_empty_bins=true, color=:plasma, normalize=:pdf,
            xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{\mu m})$",
            xticks=-4.0:0.50:4.0, yticks=-1000:500:3000,
            xlims=(-4,4), ylims=(-300,3000),
        )
        plot!(figf, x_circ_mm, z_circ_um, line=(:dash,:gray,2), label=false)
        xpos, ypos = -3.0, 2400; dx, dy = 0.7, 270
        plot!(figf, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                        [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
            color=:white, opacity=0.65, linealpha=0, label=false)
        annotate!(figf, xpos, ypos, text("⊚ Aperture", 10, :black, :bold, :center, "Helvetica"))

        # Screen
        fige = histogram2d(xs_e, zs_e;
            bins=bins_scr, show_empty_bins=true, color=:plasma, normalize=:pdf,
            xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{mm})$",
            ylims=(-1,17.5),
        );
        xpos, ypos = -4.0, 14; dx, dy = 0.9, 0.9
        plot!(fige, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                        [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
            color=:white, opacity=0.65, linealpha=0, label=false);
        annotate!(fige, xpos, ypos, text("Screen", 10, :black, :bold, :center, "Helvetica"));

        fig = plot(figa,figb,figc,figd,figf,fige,
        layout=(6,1),
        suptitle = L"$I_{0} = %$(Int(1000*Icoils[j]))\,\mathrm{mA}$",
        size=(750,850),
        right_margin=2mm,
        bottom_margin=-2mm,
        )
        plot!(fig[1], xlabel="", bottom_margin=-3mm);
        plot!(fig[2], xlabel="", bottom_margin=-3mm);
        plot!(fig[3], xlabel="", bottom_margin=-3mm);
        plot!(fig[4], xlabel="", bottom_margin=-3mm);
        plot!(fig[5], xlabel="", bottom_margin=-3mm);
        display(fig)
        savefig(fig, joinpath(OUTDIR,"QM_time_evolution_$(@sprintf("%02d", j)).$(FIG_EXT)"))
    end
    gif_path = joinpath(OUTDIR, "QM_time_evolution.gif");
    # gif(anim, gif_path, fps=2)  # adjust fps
    @info "Saved GIF" gif_path ;
    anim = nothing
    GC.gc()

    nx_bins = 128 ;
    nz_bins = [1,2,4,8];  # try different nz_bins
    gaussian_width_mm = [0.001, 0.010, 0.065, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500 ];  # try different gaussian widths
    λ0_raw_list       = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10]; # try different smoothing factors for raw data
    λ0_spline         = 0.001;


    # ============================== F=1 manifold ==============================
    println("QM approach : analyzing screen profiles for F=1 manifold")
    # three-dimensional dictionary keyed by (nz, gw, λ0_raw)
    table_f1 = OrderedDict{Tuple{Int, Float64, Float64},
                        OrderedDict{Int64, OrderedDict{Symbol, Any}}}()
    @time for nz in nz_bins, gw in gaussian_width_mm, λ0_raw in λ0_raw_list
        println("Profiles F=$(K39_params.Ispin-0.5), for nz_bin=$nz, Gaussian-convolution σ=$(Int(1e3*gw))μm, SplineFit smoothing factor λ₀=$(λ0_raw)")
        profiles_bottom_loop = QM_analyze_profiles_to_dict(
            alive_screen, K39_params;
            manifold=:F_bottom, n_bins=(nx_bins, nz), width_mm=gw,
            add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth=λ0_spline, mode=:probability
        )
        table_f1[(nz, gw, λ0_raw)] = profiles_bottom_loop
    end
    jldsave(joinpath(OUTDIR,"qm_$(Ns)_screen_profiles_f1_table.jld2"), table = table_f1)

    clrs = palette(:darkrainbow, length(nz_bins) * length(gaussian_width_mm) * length(λ0_raw_list) )
    fig = plot(
        xlabel = "Currents (A)",
        ylabel = L"$z_{\mathrm{max}}$ (mm)",
    )
    local color_idx = 1
    line_styles = [:solid, :dash, :dot, :dashdot]
    for λ0_raw in λ0_raw_list
        for gw in gaussian_width_mm
            nz_idx = 1
            for nz in nz_bins
                zvals = [table_f1[(nz, gw, λ0_raw)][i][:z_max_smooth_spline_mm] for i in eachindex(Icoils)]
                label = L"$n_{z}=%$(nz)$ | $w=%$(Int(round(1000*gw)))\,\mathrm{\mu m}$ | $\lambda_{0} = %$(λ0_raw)$"
                plot!(Icoils[2:end], abs.(zvals[2:end]),
                    line = (line_styles[nz_idx], clrs[color_idx], 2),
                    label = label)
                nz_idx      += 1
                color_idx   += 1
            end
        end
    end
    display(fig)
    plot!(
        xaxis = :log10,
        yaxis = :log10,
        xlims = (8e-3, 2),
        ylims = (8e-3, 2),
        xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        size = (850, 650),
        rightmargin = 5mm,
        legend = :outerright,
        legend_columns = length(nz_bins),
    )
    display(fig)
    savefig(fig, joinpath(OUTDIR, "qm_profiles_f1_table_comparison_w_n.$(FIG_EXT)"))
    table_f1 = nothing
    GC.gc()

    # ============================== F=2 manifold ==============================
    println("QM approach : analyzing screen profiles for F=2 manifold")
    # three-dimensional dictionary keyed by (nz, gw, λ0_raw)
    table_f2 = OrderedDict{Tuple{Int, Float64, Float64},
                        OrderedDict{Int64, OrderedDict{Symbol, Any}}}()
    @time for nz in nz_bins, gw in gaussian_width_mm, λ0_raw in λ0_raw_list
        println("Profiles F=$(K39_params.Ispin+0.5), for nz_bin=$nz, Gaussian-convolution σ=$(Int(1e3*gw))μm, SplineFit smoothing factor λ₀=$(λ0_raw)")
        profiles_bottom_loop = QM_analyze_profiles_to_dict(
            alive_screen, K39_params;
            manifold=:F_top, n_bins=(nx_bins, nz), width_mm=gw,
            add_plot=false, plot_xrange=:all, λ_raw=λ0_raw, λ_smooth=λ0_spline, mode=:probability
        )
        table_f2[(nz, gw, λ0_raw)] = profiles_bottom_loop
    end
    jldsave(joinpath(OUTDIR,"qm_$(Ns)_screen_profiles_f2_table.jld2"), table = table_f2)

    clrs = palette(:darkrainbow, length(nz_bins) * length(gaussian_width_mm) * length(λ0_raw_list) )
    fig = plot(
        xlabel = "Currents (A)",
        ylabel = L"$z_{\mathrm{max}}$ (mm)",
    )
    local color_idx = 1
    line_styles = [:solid, :dash, :dot, :dashdot]
    for λ0_raw in λ0_raw_list
        for gw in gaussian_width_mm
            nz_idx = 1
            for nz in nz_bins
                zvals = [table_f2[(nz, gw, λ0_raw)][i][:z_max_smooth_spline_mm] for i in eachindex(Icoils)]
                label = L"$n_{z}=%$(nz)$ | $w=%$(Int(round(1000*gw)))\,\mathrm{\mu m}$ | $\lambda_{0} = %$(λ0_raw)$"
                plot!(Icoils[2:end], abs.(zvals[2:end]),
                    line = (line_styles[nz_idx], clrs[color_idx], 2),
                    label = label)
                nz_idx      += 1
                color_idx   += 1
            end
        end
    end
    display(fig)
    plot!(
        xaxis = :log10,
        yaxis = :log10,
        xlims = (8e-3, 2),
        ylims = (8e-3, 2),
        xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        size = (850, 650),
        rightmargin = 5mm,
        legend = :outerright,
        legend_columns = length(nz_bins),
    )
    display(fig)
    savefig(fig, joinpath(OUTDIR, "qm_profiles_f2_table_comparison_w_n.$(FIG_EXT)"))
    table_f2 = nothing

    #########################################################################################
    GC.gc()
    @info "Memory cleaned after processing QM data"
    println("Free memory: $(Sys.free_memory() / 1e9) GB") 
    #########################################################################################
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
    open(joinpath(OUTDIR,"simqm_report.txt"), "w") do io
        write(io, report)
    end

    println("script $RUN_STAMP has finished!")
end