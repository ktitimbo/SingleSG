# Simulation of atom trajectories in the Stern–Gerlach experiment
# Kelvin Titimbo
# California Institute of Technology
# July 2025

#  Plotting Setup
# ENV["GKS_WSTYPE"] = "101"
using Plots; gr()
Plots.default(
    show=true, dpi=800, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
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
# Custom modules
include("./Modules/atoms.jl");
include("./Modules/samplings.jl");
include("./Modules/TheoreticalSimulation.jl");
using .TheoreticalSimulation;
# include("./Modules/MyPolylogarithms.jl");
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Set the working directory to the current location
cd(@__DIR__) ;
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSS");
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

FIG_EXT = "png"   # could be "pdf", "svg", etc.

println("\n\t\tRunning process on:\t $(Dates.format(T_START, "yyyymmddTHHMMSS")) \n")


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
TheoreticalSimulation.default_x_furnace     = x_furnace;
TheoreticalSimulation.default_z_furnace     = z_furnace;
TheoreticalSimulation.default_x_slit        = x_slit;
TheoreticalSimulation.default_z_slit        = z_slit;
TheoreticalSimulation.default_y_FurnaceToSlit   = y_FurnaceToSlit;
TheoreticalSimulation.default_y_SlitToSG        = y_SlitToSG;
TheoreticalSimulation.default_y_SG              = y_SG;
TheoreticalSimulation.default_y_SGToScreen      = y_SGToScreen;
TheoreticalSimulation.default_R_tube            = R_tube;

# Coil currents
Icoils = [0.001,0.002,0.003,0.005,0.007,
            0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.75,0.80
];
nI = length(Icoils);

plot_μeff(K39_params,joinpath(OUTDIR,"mm_effective.$(FIG_EXT)"))


# Sample size: number of atoms arriving to the screen
const Nss = 2000000

# Monte Carlo generation of particles traersing the filtering slit
alive_slit = generate_samples(Nss, effusion_params; v_pdf=:v3, rng = rng_set, multithreaded = true, base_seed = base_seed_set);
θesUP, θnsUP, θesDOWN, θnsDOWN = generate_matched_pairs(Nss, rng_set; mode=:total)

"""
    build_init_cond(alive, UPθe, UPθn, DOWNθe, DOWNθn; mode=:total, add_label=false)

Build paired arrays:
- `mode = :total`:
    returns `(pairsUP::Matrix, pairsDOWN::Matrix)` with sizes
    `length(UPθe) × (8 or 9)` and `length(DOWNθe) × (8 or 9)`.
- `mode = :bucket`:
    returns `(pairsUP::Matrix, pairsDOWN::Matrix)`, both `No × (8 or 9)`.

`alive[i,1:6]` is copied to the first 6 columns; columns 7–8 are θe, θn.
If `add_label=true`, column 9 is 1.0 for UP and 0.0 for DOWN.
"""
function build_init_cond(
    alive::AbstractMatrix{T},
    UPθe::AbstractVector{T}, UPθn::AbstractVector{T},
    DOWNθe::AbstractVector{T}, DOWNθn::AbstractVector{T};
    mode::Symbol = :total
) where {T<:Real}

    No = size(alive, 1)
    @assert length(UPθe)   == length(UPθn)   "UP θe/θn lengths must match"
    @assert length(DOWNθe) == length(DOWNθn) "DOWN θe/θn lengths must match"

    ncols = add_label ? 9 : 8

    if mode === :total
        n_up = length(UPθe)
        n_dn = length(DOWNθe)
        @assert n_up + n_dn == No "In :total, n_up + n_dn must equal size(alive,1)"

        pairsUP   = Matrix{T}(undef, n_up, ncols)
        pairsDOWN = Matrix{T}(undef, n_dn, ncols)

        @inbounds @views begin
            # UP block: rows 1:n_up from `alive`
            for i in 1:n_up
                pairsUP[i, 1:6] = alive[i, 1:6]
                pairsUP[i, 7]   = UPθe[i]
                pairsUP[i, 8]   = UPθn[i]
            end
            # DOWN block: rows (n_up+1):No from `alive`
            for j in 1:n_dn
                i_alive = n_up + j
                pairsDOWN[j, 1:6] = alive[i_alive, 1:6]
                pairsDOWN[j, 7]   = DOWNθe[j]
                pairsDOWN[j, 8]   = DOWNθn[j]
            end
        end

        return pairsUP, pairsDOWN

    elseif mode === :bucket
        @assert length(UPθe) == No == length(DOWNθe) "In :bucket, each θ list must have length No"

        pairsUP   = Matrix{T}(undef, No, ncols)
        pairsDOWN = Matrix{T}(undef, No, ncols)

        @inbounds @views for i in 1:No
            # UP
            pairsUP[i, 1:6] = alive[i, 1:6]
            pairsUP[i, 7]   = UPθe[i]
            pairsUP[i, 8]   = UPθn[i]
            if add_label; pairsUP[i, 9] = one(T); end
            # DOWN
            pairsDOWN[i, 1:6] = alive[i, 1:6]
            pairsDOWN[i, 7]   = DOWNθe[i]
            pairsDOWN[i, 8]   = DOWNθn[i]
            if add_label; pairsDOWN[i, 9] = zero(T); end
        end

        return pairsUP, pairsDOWN

    else
        error("Unknown mode=$mode. Use :total or :bucket.")
    end
end


if save_fig
    display(plot_velocity_stats(alive_slit, joinpath(dir_path, "init_vel_stats.png")))
end

#################################################################################
# FUNCTIONS
#################################################################################








# Magnet shape
function z_magnet_edge(x)
    a = 2.5e-3;
    z_center = 1.3*a 
    r_edge = a
    φ = π/6

    if x <= -r_edge
        z = z_center - tan(φ)*(x+r_edge)
    elseif x > -r_edge && x <= r_edge
        z = z_center - sqrt(r_edge^2 - x^2)
    elseif x > r_edge
        z = z_center + tan(φ)*(x-r_edge)
    else
        0
    end

    return z
end

function z_magnet_trench(x)
    a = 2.5e-3;
    z_center = 1.3*a 
    r_edge = 1.0*a
    r_trench = 1.362*a
    r_trench_center = z_center - 1.018*a
    lw = 1.58*a
    φ = π/6

    if x <= -r_trench - lw*cos(φ)
        z = r_trench_center + lw*sin(φ)
    elseif x > -r_trench-lw*cos(φ) && x <= -r_trench
        z = r_trench_center - tan(φ)*(x+r_trench)
    elseif x > -r_trench && x <= r_trench
        z = r_trench_center - sqrt( r_trench^2 - x^2 )
    elseif x > r_trench && x<= r_trench + lw*cos(φ)
        z = r_trench_center + tan(φ)*(x-r_trench)
    elseif x > r_trench + lw*cos(φ)
        z = r_trench_center + lw*sin(φ)
    else
        0
    end

    return z
end

@inline function z_magnet_edge_time(t, r0::AbstractVector{Float64}, v0::AbstractVector{Float64})
    a =2.5e-3;
    z_center = 1.3*a 
    r_edge = a
    φ = π/6

    x = r0[1] + v0[1]*t
    if x <= -r_edge
        z = z_center - tan(φ)*(x+r_edge)
    elseif x <= r_edge
        z = z_center - sqrt(r_edge^2 - x^2)
    else # x > r_edge
        z = z_center + tan(φ)*(x-r_edge)
    end

    return z
end

@inline function z_magnet_trench_time(t, r0::AbstractVector{Float64}, v0::AbstractVector{Float64})
    a = 2.5e-3;
    z_center = 1.3*a 
    r_edge = 1.0*a
    r_trench = 1.362*a
    r_trench_center = z_center - 1.018*a
    lw = 1.58*a
    φ = π/6

    x = r0[1] + v0[1]*t
    if x <= -r_trench - lw*cos(φ)
        z = r_trench_center + lw*sin(φ)
    elseif x <= -r_trench
        z = r_trench_center - tan(φ)*(x+r_trench)
    elseif x <= r_trench
        z = r_trench_center - sqrt( r_trench^2 - x^2 )
    elseif x<= r_trench + lw*cos(φ)
        z = r_trench_center + tan(φ)*(x-r_trench)
    else # x > r_trench + lw*cos(φ)
        z = r_trench_center + lw*sin(φ)
    end

    return z
end

function z_magnet_profile_time(t, r0::AbstractVector{Float64}, v0::AbstractVector{Float64}, side::String)
    a = 2.5e-3;
    z_center = 1.3*a 
    r_edge = 1.0*a
    r_trench = 1.362*a
    r_trench_center = z_center - 1.018*a
    lw = 1.58*a
    φ = π/6

    x = r0[1] + v0[1]*t

    if side=="top"
        if x <= -r_edge
           return z_center - tan(φ)*(x+r_edge)
        elseif x <= r_edge
            return z_center - sqrt(r_edge^2 - x^2)
        else # r> r_edge
            return z_center + tan(φ)*(x-r_edge)
        end
    elseif side=="bottom" 
        if x <= -r_trench - lw*cos(φ)
            return r_trench_center + lw*sin(φ)
        elseif x <= -r_trench
            return r_trench_center - tan(φ)*(x+r_trench)
        elseif x <= r_trench
            return r_trench_center - sqrt( r_trench^2 - x^2 )
        elseif x <= r_trench + lw*cos(φ)
            return r_trench_center + tan(φ)*(x-r_trench)
        else # x > r_trench + lw*cos(φ)
            return r_trench_center + lw*sin(φ)
        end
    else
        error("options are top and bottom")
    end
end



function build_init_cond(alive::Matrix{Float64}, θes::Vector{Float64}, θns::Vector{Float64})
    No = size(alive, 1)
    pairs = Matrix{Float64}(undef, No, 8)
    @inbounds for i in 1:No
        pairs[i, 1:6] = alive[i,:]
        pairs[i, 7] = θes[i]
        pairs[i, 8] = θns[i]
    end
    return pairs
end

function find_bad_particles_ix(Ix, pairs, kx::Float64)
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

                t_sweep_sg = range(t_in, t_out, length=t_length)
                z_val = CQDEqOfMotion_z.(t_sweep_sg, Ref(i0), Ref(μₑ), Ref(r0), Ref(v0), Ref(θe0), Ref(θn0), Ref(kx))
                z_top = z_magnet_edge_time.(t_sweep_sg, Ref(r0), Ref(v0))
                z_bottom = z_magnet_trench_time.(t_sweep_sg, Ref(r0), Ref(v0))

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

# Function to plot histogram using Freedman-Diaconis binning rule
function FD_histograms(data_list::Vector{Float64},Label::LaTeXString,color)
    # Calculate the interquartile range (IQR)
    Q1 = quantile(data_list, 0.25)
    Q3 = quantile(data_list, 0.75)
    IQR = Q3 - Q1

    # Calculate Freedman-Diaconis bin width
    n = length(data_list)
    bin_width = 2 * IQR / (n^(1/3))

    # Calculate the number of bins using the range of the data
    data_range = maximum(data_list) - minimum(data_list)
    bins = ceil(Int, data_range / bin_width)

    # Plot the histogram
    histogram(data_list, bins=bins, normalize=:pdf,
            label=Label,
            # xlabel="Polar angle", 
            color=color,
            alpha=0.8,
            xlim=(0,π),
            xticks=PlottingTools.pitick(0, π, 8; mode=:latex),)
end

"""
    plot_velocity_stats(alive::Matrix{Float64}, path_filename::AbstractString) -> Nothing

    Generate and save a multi-panel figure showing velocity statistics, angular
    distributions, and spatial distribution for a set of particles.

    The figure includes:
    1. Speed distribution with mean and RMS markers.
    2. Distributions for velocity components (vx, vy, vz).
    3. Polar (θ) and azimuthal (φ) velocity angle distributions with mean markers.
    4. 2D histogram of initial positions (x, z) in mm and μm.

    # Arguments
    - `alive::Matrix{Float64}`: Particle data matrix with columns:
        1. x-position (m)
        2. y-position (m)
        3. z-position (m)
        4. vx-velocity (m/s)
        5. vy-velocity (m/s)
        6. vz-velocity (m/s)
    - `path_filename::AbstractString`: Output file path for saving the figure.

    # Notes
    - Uses Freedman–Diaconis binning for all histograms.
    - Plots are normalized to probability density.
"""
function plot_velocity_stats(alive::Matrix{Float64}, path_filename::String)
    @assert size(alive, 2) ≥ 6 "Expected at least 6 columns (x, y, z, vx, vy, vz)."

    # --- Velocity magnitude and angles ---
    vxs, vys, vzs = eachcol(alive[:, 4:6])
    velocities = sqrt.(vxs.^2 .+ vys.^2 .+ vzs.^2)
    theta_vals = acos.(vzs ./ velocities)       # polar angle
    phi_vals   = atan.(vys, vxs)                 # azimuthal angle

    # Means
    mean_v, rms_v = mean(velocities), sqrt(mean(velocities.^2))
    mean_theta, mean_phi = mean(theta_vals), mean(phi_vals)


    # Histogram for velocities
    figa = histogram(velocities;
        bins = FreedmanDiaconisBins(velocities),
        label = L"$v_0$", normalize = :pdf,
        xlabel = L"v_{0} \ (\mathrm{m/s})",
        alpha = 0.70,
    )
    vline!([mean_v], label = L"$\langle v_{0} \rangle = %$(round(mean_v, digits=1))\ \mathrm{m/s}$",
           line = (:black, :solid, 2))
    vline!([rms_v], label = L"$\sqrt{\langle v_{0}^2 \rangle} = %$(round(rms_v, digits=1))\ \mathrm{m/s}$",
           line = (:red, :dash, 3))

    figb = histogram(theta_vals;
        bins = FreedmanDiaconisBins(theta_vals),
        label = L"$\theta_v$", normalize = :pdf,
        alpha = 0.70, xlabel = L"$\theta_{v}$"
    )
    vline!([mean_theta], label = L"$\langle \theta_{v} \rangle = %$(round(mean_theta/π, digits=3))\pi$",
           line = (:black, :solid, 2))

    figc = histogram(phi_vals;
        bins = FreedmanDiaconisBins(phi_vals),
        label = L"$\phi_v$", normalize = :pdf,
        alpha = 0.70, xlabel = L"$\phi_{v}$"
    )
    vline!([mean_phi], label = L"$\langle \phi_{v} \rangle = %$(round(mean_phi/π, digits=3))\pi$",
           line = (:black, :solid, 2))

    # 2D Histogram of position (x, z)
    # --- 2D position histogram ---
    xs, zs = 1e3 .* alive[:, 1], 1e6 .* alive[:, 3]  # mm, μm
    figd = histogram2d(xs, zs;
        bins = (FreedmanDiaconisBins(xs), FreedmanDiaconisBins(zs)),
        show_empty_bins = true, color = :plasma,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -1.0:0.25:1.0, yticks = -50:10:50,
        colorbar_position = :bottom,
    )

    # --- Velocity component histograms ---
    fige = histogram(vxs;
        bins = FreedmanDiaconisBins(vxs), normalize = :pdf,
        label = L"$v_{0,x}$", alpha = 0.65, color = :orange,
        xlabel = L"$v_{0,x} \ (\mathrm{m/s})$"
    )
    figf = histogram(vys;
        bins = FreedmanDiaconisBins(vys), normalize = :pdf,
        label = L"$v_{0,y}$", alpha = 0.65, color = :blue,
        xlabel = L"$v_{0,y} \ (\mathrm{m/s})$"
    )
    figg = histogram(vzs;
        bins = FreedmanDiaconisBins(vzs), normalize = :pdf,
        label = L"$v_{0,z}$", alpha = 0.65, color = :red,
        xlabel = L"$v_{0,z} \ (\mathrm{m/s})$"
    )

    # Combine plots
    fig = plot(
        figa, fige, figb, figf, figc, figg, figd,
        layout = @layout([a1 a2; a3 a4; a5 a6; a7]),
        size = (650, 800),
        legendfontsize = 8,
        left_margin = 3mm,
    );
    savefig(fig, path_filename)
    

    return fig
end

"""
    plot_SG_geometry(path_filename::AbstractString) -> Nothing

    Plot the cross-sectional geometry of a Stern–Gerlach (SG) magnet slit and save it to file.

    The plot shows:
    - The rounded top edge of the magnet.
    - The lower trench region.
    - The rectangular slit opening.

    # Arguments
    - `path_filename::AbstractString`: Output file path for the saved plot (PDF, PNG, etc.).

    # Assumptions
    - The functions `z_magnet_edge(x::Real)` and `z_magnet_trench(x::Real)` are defined
    and return vertical positions (in meters) of the magnet’s top and bottom surfaces
    for a given horizontal position `x`.
    - Global constants `x_slit` and `z_slit` (in meters) define the slit’s width and height.

    # Units
    - All distances in the plot are shown in millimeters.
"""
function plot_SG_geometry(path_filename::AbstractString)
    @assert isdefined(Main, :z_magnet_edge) "Function `z_magnet_edge` must be defined."
    @assert isdefined(Main, :z_magnet_trench) "Function `z_magnet_trench` must be defined."
    @assert isdefined(Main, :x_slit) && isdefined(Main, :z_slit) "Global `x_slit` and `z_slit` must be defined."

    # x positions for evaluation (in meters)
    x_line = 1e-3 .* collect(range(-10, 10, length=10_001))

    # Base figure
    fig = plot(
        xlabel = L"$x \ (\mathrm{mm})$",
        xlim = (-8, 8), xticks = -8:2:8,
        ylabel = L"$y \ (\mathrm{mm})$",
        ylim = (-3, 7), yticks = -3:1:7,
        aspect_ratio = :equal,
        legend = :bottomright,
        title = "Stern–Gerlach Slit Geometry"
    )

    # Top magnet edge shape
    x_fill = 1e3 .* x_line
    y_edge = 1e3 .* z_magnet_edge.(x_line)
    y_top  = fill(10.0, length(x_fill))
    plot!(fig, [x_fill; reverse(x_fill)], [y_edge; reverse(y_top)];
        seriestype = :shape, label = "Rounded edge",
        color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
    )

    # Bottom trench shape
    y_trench = 1e3 .* z_magnet_trench.(x_line)
    y_bottom = fill(-10.0, length(x_fill))
    plot!(fig, [x_fill; reverse(x_fill)], [y_bottom; reverse(y_trench)];
        seriestype = :shape, label = "Trench",
        color = :grey60, line = (:solid, :grey60), fillalpha = 0.75
    )

    # Slit rectangle
    plot!(fig,
        1e3 .* 0.5 .* [-x_slit, -x_slit, x_slit, x_slit, -x_slit],
        1e3 .* 0.5 .* [-z_slit,  z_slit,  z_slit, -z_slit, -z_slit];
        seriestype = :shape, label = "Slit",
        line = (:solid, :red, 1), color = :red, fillalpha = 0.2
    )

    savefig(fig, path_filename)
    
    return nothing
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

function bin_centers(edges::AbstractVector)
    return (edges[1:end-1] .+ edges[2:end]) ./ 2
end

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

"""
    max_of_bspline_positions(z, y; λ0=1e-3, order=4, n_peaks=1, n_scan=max(400, length(z)), sep=1e-6)

    Fit a smoothing B-spline to `(z, y)` data and return the positions of the most prominent local maxima.

    # Arguments
    - `z::AbstractVector`: Sorted vector of x-coordinates (domain points).
    - `y::AbstractVector`: Vector of y-values corresponding to `z`.
    - `λ0::Real=1e-3`: Smoothing parameter for the spline fit. Smaller values follow the data more closely; larger values yield smoother curves.
    - `order::Int=4`: B-spline order (4 = cubic).
    - `n_peaks::Int=1`: Number of top peaks to return, sorted by descending spline value.
    - `n_scan::Int=max(400, length(z))`: Number of points in the dense scan for detecting candidate peaks.
    - `sep::Real=1e-6`: Minimum separation between peaks (in `z` units) to consider them distinct.

    # Method
    1. Fits a smoothing B-spline `S(z)` using `BSplineKit.fit` with uniform weights from `compute_weights`.
    2. Performs a dense scan to find candidate peak locations by detecting sign changes in the finite-difference slope.
    3. Refines each candidate's position using a local Brent optimization in a small bracket around the candidate.
    4. Removes peaks closer than `sep` to each other.
    5. Sorts the remaining peaks by their spline height and returns the positions of the top `n_peaks`.

    # Returns
    - `Vector{Float64}`: Positions of the most prominent peaks in descending order of height.

    # Notes
    - Only peak positions are returned; heights can be obtained by evaluating the returned spline at those positions if needed.
    - The function is robust to multiple local maxima and will always include endpoints as candidates in case the maximum lies at the boundary.
"""
function max_of_bspline_positions(z::AbstractVector,y::AbstractVector;
    λ0::Real=0.01, order::Int=4, n_peaks::Int=1, n_scan::Int=max(400, length(z)),sep::Real=1e-6)

    @assert length(z) == length(y) && issorted(z)
    a, b = extrema(z)

    # Fit smoothing spline
    S = BSplineKit.fit(BSplineOrder(order), z, y, λ0; weights=compute_weights(z, λ0))

    # --- 1) Dense scan to find candidate peaks
    xs = range(a, b; length=n_scan)
    ys = S.(xs)
    dx = step(xs)
    dydx = diff(ys) ./ dx

    cand = Float64[]
    for i in 1:length(dydx)-1
        if (dydx[i] > 0) && (dydx[i+1] < 0)
            push!(cand, xs[i+1])              # near a local max
        end
    end
    push!(cand, a); push!(cand, b)            # endpoints too
    cand = unique(sort(cand))

    # --- 2) Refine each candidate in a small bracket with Brent
    negS(x) = -S(x)
    function refine(x0)
        δ = 2dx
        lo = max(a, x0 - δ)
        hi = min(b, x0 + δ)
        if lo == hi
            return x0
        end
        res = Optim.optimize(negS, lo, hi)
        return Optim.minimizer(res)
    end

    peaks = Float64[]
    for c in cand
        ẑ = refine(c)
        # de-duplicate close peaks
        if isempty(peaks) || all(abs(ẑ - p) > sep for p in peaks)
            push!(peaks, ẑ)
        end
    end

    # --- 3) Sort by actual spline height and return positions only
    ord = sortperm(S.(peaks), rev=true)
    peaks = peaks[ord]
    return peaks[1:min(n_peaks, length(peaks))] , S # positions only
end

"""
    analyze_screen_profile(
        Ix::Real,
        data_mm::AbstractMatrix;
        n_bins::Integer = 2,
        width_mm::Float64 = 0.1,
        add_plot::Bool = false,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3
    ) -> NamedTuple

    Analyze the vertical (z) profile of particle hits on a screen for a given
    coil current, using 2D histogramming, smoothing, and spline fitting to
    identify peak positions.

    This version is designed for both **single-current analysis** and **batch
    processing** via `analyze_profiles_to_dict`. The first argument `Ix` is
    the coil current (in A) and is used to annotate plots, allowing you to
    automatically generate titled plots for each current in a loop.

    The function:
    1. Builds a 2D histogram of hit positions (x, z) in **millimeters** with
    bin centers symmetric about zero.
    2. Extracts the mean **z-profile** (averaged over x) from the histogram.
    3. Finds the z-location of the maximum in the raw profile and in the raw
    profile fitted with a smoothing spline.
    4. Smooths the profile with a Gaussian kernel, then finds the z-location
    of the maximum from both the smoothed data and a smoothing spline fit.
    5. Optionally plots raw, smoothed, and spline-fitted profiles, annotated
    with the corresponding maxima and titled with the coil current.

    # Arguments
    - `Ix::Real`:  
        Coil current (in A) for labeling the plot (especially in batch runs).
    - `data_mm::AbstractMatrix`:  
        N×2 array of hit positions in **mm**, where column 1 = x, column 2 = z.
    - `n_bins::Integer`:  
        Binning multiplier; bin size in mm is  
        `bin_size = 1e3 * n_bins * cam_pixelsize`,  
        with `cam_pixelsize` assumed global in meters.
    - `width_mm::Float64`:  
        Gaussian kernel width σ (mm) for `smooth_profile`.
    - `add_plot::Bool`:  
        If `true`, plots the raw, smoothed, and spline profiles.
    - `λ_raw::Float64`:  
        Regularization parameter for spline fitting of raw profile.
    - `λ_smooth::Float64`:  
        Regularization parameter for spline fitting of smoothed profile.

    # Implementation details
    - Analysis limits: `x ∈ [-9.0, 9.0]` mm, `z ∈ [-12.5, 12.5]` mm.
    - Bin centers are symmetric around 0 with a center exactly at 0.
    - Maxima are identified using `argmax` for discrete data and
    `max_of_bspline_positions` for spline fits.

    # Returns
    A `NamedTuple` with:
    - `z_profile` :: Matrix (Nz × 3) → `[z_center, raw_counts, smoothed_counts]`
    - `z_max_raw_mm` :: z at raw profile maximum [mm]
    - `z_max_raw_spline_mm` :: z at spline-fitted raw maximum [mm]
    - `z_max_smooth_mm` :: z at smoothed profile maximum [mm]
    - `z_max_smooth_spline_mm` :: z at spline-fitted smoothed maximum [mm]

    # Notes
    - Pass `Ix` from a loop over coil currents in `analyze_profiles_to_dict` to
    automatically label each plot with its current.
    - Intended for integration into workflows that analyze many currents in
    sequence.
"""
function analyze_screen_profile(Ix, data_mm::AbstractMatrix; 
    m_mom::Symbol = :up, n_bins::Integer = 2, width_mm::Float64 = 0.1, add_plot::Bool = false, λ_raw::Float64=0.01, λ_smooth::Float64 = 1e-3)

    @assert size(data_mm,2) == 2 "data_mm must be N×2 (columns: x,z in mm)"
    @assert n_bins > 0 "n_bins must be > 0"
    @assert width_mm > 0 "width_mm must be positive"

    # Fixed analysis limits
    xlim = (-9.0, 9.0)
    zlim = (-12.5, 12.5)
    xmin, xmax = xlim
    zmin, zmax = zlim

    # Bin size in mm (cam_pixelsize is assumed global in meters)
    bin_size = 1e3 * n_bins * cam_pixelsize

    # --------------------------------------------------------
    # X edges: force symmetric centers around 0
    # --------------------------------------------------------
    x_half_range = max(abs(xmin), abs(xmax))
    kx = max(1, ceil(Int, x_half_range / bin_size))
    centers_x = collect((-kx:kx) .* bin_size)
    edges_x = collect((-(kx + 0.5)) * bin_size : bin_size : ((kx + 0.5) * bin_size))

    # --------------------------------------------------------
    # Z edges: force symmetric centers around 0
    # --------------------------------------------------------
    z_half_range = max(abs(zmin), abs(zmax))
    kz = max(1, ceil(Int, z_half_range / bin_size))
    centers_z = collect((-kz:kz) .* bin_size)
    edges_z = collect((-(kz + 0.5)) * bin_size : bin_size : ((kz + 0.5) * bin_size))

    # 2D histogram
    x = @view data_mm[:, 1]
    z = @view data_mm[:, 2]
    h = fit(Histogram, (x, z), (edges_x, edges_z))
    counts = h.weights  # size: (length(centers_x), length(centers_z))

    # z-profile = mean over x bins
    z_profile_raw = vec(mean(counts, dims = 1))
    z_max_raw_mm = centers_z[argmax(z_profile_raw)]
    z_max_raw_spline_mm, Sfit_raw = max_of_bspline_positions(centers_z,z_profile_raw;λ0=λ_raw)

    # Smoothing
    z_profile_smooth = smooth_profile(centers_z, z_profile_raw, width_mm)
    z_max_smooth_mm = centers_z[argmax(z_profile_smooth)]
    z_max_smooth_spline_mm, Sfit_smooth = max_of_bspline_positions(centers_z,z_profile_smooth;λ0=λ_smooth)

    # Combine into one matrix for convenience: [z raw smooth]
    z_profile = hcat(
        centers_z,
        z_profile_raw,
        z_profile_smooth,
    )

    if add_plot
        # Uncomment to visualize full 2D histogram:
        # heatmap(centers_x, centers_z, counts', xlabel="x (mm)", ylabel="z (mm)", title="2D Histogram")

        # Profiles
        z = range(zmin,zmax,length=max(2000,length(centers_z)))
        xlims_plot = m_mom == :up ? (zmin/4, zmax) : m_mom == :dw ? (zmin, zmax/4) : (zmin, zmax)
        fig=plot(       
            title =L"$I_{c}=%$(Ix)\mathrm{A}$",
            xlabel = L"$z$ (mm)", 
            ylabel = "mean counts (au)",
            xlims = xlims_plot,
        );
        plot!(z_profile[:, 1], z_profile[:, 2], 
            label = L"Raw $z=%$(round(z_max_raw_mm,digits=4))\mathrm{mm}$",
            line=(:solid,:gray90,2),
            marker=(:circle,:white,1),
            markerstrokecolor=:gray70
        );
        vline!([z_max_raw_mm], 
            label=false,
            line=(:solid,:black,1),
        );
        plot!(z,Sfit_raw.(z), 
            label=L"Spline Raw $z=%$(round(z_max_raw_spline_mm[1],digits=4))\mathrm{mm}$",
            line=(:forestgreen,:dot,2),
        );
        vline!(z_max_raw_spline_mm, 
            label=false,
            line=(:green,:solid,1))
        # Convolution
        plot!(z_profile[:, 1], z_profile[:, 3], 
            label = L"Smoothed $z=%$(round(z_max_smooth_mm,digits=4))\mathrm{mm}$",
            line=(:coral3,:dash,2),
        );
        vline!([z_max_smooth_mm], 
            label=false,
            line=(:red,:solid,1)
        );
        plot!(z,Sfit_smooth.(z), 
            label=L"Spline Smoothed $z=%$(round(z_max_smooth_spline_mm[1],digits=4))\mathrm{mm}$",
            line=(:royalblue3,:dot,2),
        );
        vline!(z_max_smooth_spline_mm, 
            label=false,
            line=(:blue,:solid,1)
        );
        savefig(fig, joinpath(dir_path, "profiles_"*replace(@sprintf("%d", 1e3*Ix), "." => "")*"uA.png"))
    end

    return (
        z_profile = z_profile,
        z_max_raw_mm = z_max_raw_mm,
        z_max_raw_spline_mm = z_max_raw_spline_mm[1],
        z_max_smooth_mm = z_max_smooth_mm,
        z_max_smooth_spline_mm = z_max_smooth_spline_mm[1]
    )
end

"""
    analyze_profiles_to_dict(
        Icoils::AbstractVector,
        screen_up::AbstractDict{<:Integer,<:AbstractMatrix};
        n_bins::Integer = 2,
        width_mm::Float64 = 0.1,
        add_plot::Bool = false,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3,
        store_profiles::Bool = true,
    ) -> OrderedDict{Int, OrderedDict{Symbol, Any}}

    Run `analyze_screen_profile` for each coil current `Icoils[i]` with its
    corresponding screen data `screen_up[i]`, collecting results in a nested
    dictionary keyed by the **index** `i` (1..length(Icoils)).

    This is the batch companion to `analyze_screen_profile(Ix, data_mm; ...)`.
    It converts each dataset from meters to millimeters (taking columns 1=x and
    3=z), calls the single-dataset analyzer (passing `Ix = Icoils[i]` so the
    plots are titled per current), and aggregates the outputs.

    # Arguments
    - `Icoils`: Vector of coil currents (A). Length must match `screen_up`.
    - `screen_up`: Dict-like container (e.g. `OrderedDict{Int, Matrix}`) whose
    keys include `1:length(Icoils)` and whose values are N×M matrices of hit
    positions in **meters** (columns: 1=x, 3=z).
    - `n_bins`: Histogram binning multiplier used by `analyze_screen_profile`.
    - `width_mm`: Gaussian kernel σ (mm) used by `smooth_profile`.
    - `add_plot`: If `true`, each call plots the profiles titled with `Icoils[i]`.
    - `λ_raw`, `λ_smooth`: Spline regularization parameters for raw and smoothed
    profiles, respectively.
    - `store_profiles`: If `true`, store the full `z_profile` array for each
    current (can be large). If `false`, omit it to save memory.

    # Returns
    An `OrderedDict{Int, OrderedDict{Symbol, Any}}` such that:
    - `out[i][:Icoil]`                       → `Icoils[i]`
    - `out[i][:z_max_raw_mm]`                → raw-profile maximum z [mm]
    - `out[i][:z_max_raw_spline_mm]`         → spline fit (raw) maximum z [mm]
    - `out[i][:z_max_smooth_mm]`             → smoothed-profile maximum z [mm]
    - `out[i][:z_max_smooth_spline_mm]`      → spline fit (smoothed) maximum z [mm]
    - `out[i][:z_profile]` (optional)        → Nz×3 matrix `[z, raw, smooth]`

    # Notes
    - Expects `screen_up[i]` to have at least 3 columns (x=col 1, z=col 3).
    - This function assumes `analyze_screen_profile(Ix, data_mm; ...)` accepts
    hit data in **mm** and will handle the histogramming/plotting.
    - If you prefer keys by current value instead of index, use a
    `Dict{Float64, ...}` and set `out[Icoils[i]] = inner`.
"""
function analyze_profiles_to_dict(Icoils::AbstractVector, screen_up::AbstractDict{<:Integer,<:AbstractMatrix};
    m_mom::Symbol = :up, n_bins::Integer = 2, width_mm::Float64 = 0.1, add_plot::Bool = false,
    λ_raw::Float64 = 0.01, λ_smooth::Float64 = 1e-3, store_profiles::Bool = true,)

    @assert length(Icoils) == length(screen_up)
    @assert all(haskey(screen_up, i) for i in 1:length(Icoils))
    @assert all(size(screen_up[i], 2) ≥ 3 for i in 1:length(Icoils)) "screen_up[i] must have at least 3 columns (x in col 1, z in col 3)"
    @assert eltype(Icoils) <: Real "Icoils must be numeric"

    out = OrderedDict{Int, OrderedDict{Symbol, Any}}()
    for i in eachindex(Icoils)
        data_mm = 1e3 .* screen_up[i][:, [1, 3]]

        res = analyze_screen_profile(
            Icoils[i],
            data_mm;
            m_mom=m_mom,
            n_bins=n_bins,
            width_mm=width_mm,
            add_plot=add_plot,
            λ_raw=λ_raw,
            λ_smooth=λ_smooth,
        )

        inner = OrderedDict{Symbol, Any}(
            :Icoil => Icoils[i],
            :z_max_raw_mm => res.z_max_raw_mm,
            :z_max_raw_spline_mm => res.z_max_raw_spline_mm,
            :z_max_smooth_mm => res.z_max_smooth_mm,
            :z_max_smooth_spline_mm => res.z_max_smooth_spline_mm,
        )
        if store_profiles
            inner[:z_profile] = res.z_profile
        end
        out[i] = inner
    end
    return out
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
