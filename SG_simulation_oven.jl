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
include("./Modules/JLD2_MyTools.jl");
include("./Modules/TheoreticalSimulation.jl");
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
K39_params = TheoreticalSimulation.AtomParams(atom); # [R μn γn Ispin Ahfs M ] 
# Math constants
const TWOπ = 2π;
const INV_E = exp(-1);
quantum_numbers = TheoreticalSimulation.fmf_levels(K39_params);

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
x_position = TheoreticalSimulation.pixel_coordinates(x_pixels, sim_bin_x, sim_pixelsize_x);
z_position = TheoreticalSimulation.pixel_coordinates(z_pixels, sim_bin_z, sim_pixelsize_z);
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
effusion_params = TheoreticalSimulation.BeamEffusionParams(x_furnace,z_furnace,x_slit,z_slit,y_FurnaceToSlit,T_K,K39_params);
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
@inline function AtomicBeamVelocity_v3(rng::AbstractRNG, p::TheoreticalSimulation.EffusionParams)::SVector{3,Float64}
    ϕ = TWOπ * rand(rng)
    θ = asin(sqrt(rand(rng)))
    v = sqrt(-2*p.α2 * (1 + lambertw((rand(rng)-1)*INV_E, -1)))
    sθ = sin(θ); cθ = cos(θ); sϕ = sin(ϕ); cϕ = cos(ϕ)
    return SVector(v*sθ*sϕ, v*cθ, v*sθ*cϕ)
end

@inline function InitialPositions(rng::AbstractRNG)
    x0 = x_furnace * (rand(rng) - 0.5)
    z0 = z_furnace * (rand(rng) - 0.5)
    return SVector(x0,0,z0)
end

@inline function sample_initial_conditions(N, rng, p)
    pos = Matrix{Float64}(undef, N, 3)
    vel = Matrix{Float64}(undef, N, 3)

    @inbounds for i in 1:N
        pos_i = InitialPositions(rng)
        vel_i = AtomicBeamVelocity_v3(rng, p)

        pos[i,1] = pos_i[1]
        pos[i,2] = pos_i[2]
        pos[i,3] = pos_i[3]

        vel[i,1] = vel_i[1]
        vel[i,2] = vel_i[2]
        vel[i,3] = vel_i[3]
    end

    return pos, vel
end


# --------------------------
# Geometry helpers
# --------------------------

@inline function time_to_plane(pos::SVector{3,Float64},
                               vel::SVector{3,Float64},
                               axis::Int, bound::Float64)
    """
    Computes the time at which a particle moving in a straight line intersects a plane of the form:
        x=bound, y=bound, or z=bound
    depending on axis.
    """
    v = vel[axis]
    if v == 0.0
        return Inf
    end
    t = (bound - pos[axis]) / v
    return (t > 0.0) ? t : Inf
end

function next_hit(pos::SVector{3,Float64}, vel::SVector{3,Float64},
                  xmin::Float64, xmax::Float64,
                  L::Float64,
                  zmin::Float64, zmax::Float64)
    """
    Determines which cavity wall the particle will hit next and how long it will take.
    """

    tx0 = time_to_plane(pos, vel, 1, xmin)
    tx1 = time_to_plane(pos, vel, 1, xmax)

    ty0 = time_to_plane(pos, vel, 2, 0.0)
    ty1 = time_to_plane(pos, vel, 2, L)

    tz0 = time_to_plane(pos, vel, 3, zmin)
    tz1 = time_to_plane(pos, vel, 3, zmax)

    tmin = tx0; hit_axis = 1; hit_side = 0
    if tx1 < tmin; tmin = tx1; hit_axis = 1; hit_side = 1; end
    if ty0 < tmin; tmin = ty0; hit_axis = 2; hit_side = 0; end
    if ty1 < tmin; tmin = ty1; hit_axis = 2; hit_side = 1; end
    if tz0 < tmin; tmin = tz0; hit_axis = 3; hit_side = 0; end
    if tz1 < tmin; tmin = tz1; hit_axis = 3; hit_side = 1; end

    return tmin, hit_axis, hit_side
end

# --------------------------
# Lambertian scattering
# --------------------------

function sample_lambertian_dir(rng::AbstractRNG, n::SVector{3,Float64})
    nn = n / norm(n)

    u1 = rand(rng)
    u2 = rand(rng)

    r = sqrt(u1)
    ϕ = 2π * u2

    # local coords, z aligned with nn
    x = r * cos(ϕ)
    y = r * sin(ϕ)
    z = sqrt(1.0 - u1)

    a  = (abs(nn[1]) < 0.9) ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    t1 = normalize(cross(nn, a))
    t2 = cross(nn, t1)

    return x*t1 + y*t2 + z*nn
end

@inline function inward_normal(hit_axis::Int, hit_side::Int)
    if hit_axis == 1
        return (hit_side == 0) ? SVector( 1.0, 0.0, 0.0) : SVector(-1.0, 0.0, 0.0)
    elseif hit_axis == 2
        return (hit_side == 0) ? SVector( 0.0, 1.0, 0.0) : SVector( 0.0,-1.0, 0.0)
    else
        return (hit_side == 0) ? SVector( 0.0, 0.0, 1.0) : SVector( 0.0, 0.0,-1.0)
    end
end

# --------------------------
# Simulator (centered x,z; open y)
# --------------------------

function simulate_cavity_centered(pos0::AbstractMatrix{<:Real},
                                  vel0::AbstractMatrix{<:Real},
                                  L::Real;
                                  x_furnace::Real = 2.0e-3,
                                  z_furnace::Real = 100e-6,
                                  max_bounces::Int = 50_000,
                                  eps_push::Float64 = 1e-12,
                                  collect_backscatter::Bool = true,
                                  rng::AbstractRNG = Random.default_rng())
    """
    Cavity:
    x ∈ [-x_furnace/2, +x_furnace/2]
    y ∈ [0, L]
    z ∈ [-z_furnace/2, +z_furnace/2]

    Open boundaries at y=0 and y=L (no scattering there).
    Lambertian scattering on x- and z-walls. Speed preserved.
    """
    N = size(pos0, 1)
    @assert size(pos0,2) == 3 && size(vel0,2) == 3 && size(vel0,1) == N

    xmin = -0.5 * Float64(x_furnace)
    xmax =  0.5 * Float64(x_furnace)
    zmin = -0.5 * Float64(z_furnace)
    zmax =  0.5 * Float64(z_furnace)
    L    = Float64(L)

    exited      = falses(N)  # y=L
    backscatter = falses(N)  # y=0

    exit_pos = zeros(Float64, N, 3)
    exit_vel = zeros(Float64, N, 3)
    exit_t   = zeros(Float64, N)

    nhit_x     = zeros(Int, N)
    nhit_z     = zeros(Int, N)
    nhit_total = zeros(Int, N)

    path_len = zeros(Float64, N)  # total arclength traveled inside cavity
    n_bounces = zeros(Int, N)

    for i in 1:N
        pos = SVector{3,Float64}(Float64(pos0[i,1]), Float64(pos0[i,2]), Float64(pos0[i,3]))
        vel = SVector{3,Float64}(Float64(vel0[i,1]), Float64(vel0[i,2]), Float64(vel0[i,3]))

        speed = norm(vel)
        if speed == 0.0
            continue
        end

        ttot = 0.0
        sacc = 0.0
        nb   = 0

        hx = 0
        hz = 0

        while nb <= max_bounces
            thit, hit_axis, hit_side = next_hit(pos, vel, xmin, xmax, L, zmin, zmax)
            if thit == Inf
                break
            end

            # distance traveled in this flight segment
            # (speed is constant along each segment)
            sacc += speed * thit

            pos_hit = pos + thit * vel
            ttot += thit

            # y planes: exit/backscatter
            if hit_axis == 2
                if hit_side == 1
                    exited[i] = true
                else
                    backscatter[i] = true
                    if !collect_backscatter
                        break
                    end
                end

                exit_pos[i, :] .= Tuple(pos_hit)
                exit_vel[i, :] .= Tuple(vel)
                exit_t[i] = ttot

                path_len[i] = sacc
                nhit_x[i] = hx
                nhit_z[i] = hz
                nhit_total[i] = hx + hz
                n_bounces[i] = nb
                break
            end

            # x or z wall hit: count + scatter
            nb += 1
            if hit_axis == 1
                hx += 1
            else
                hz += 1
            end

            n = inward_normal(hit_axis, hit_side)
            dir = sample_lambertian_dir(rng, n)
            vel = speed * dir

            # push into interior to avoid re-hit
            pos = pos_hit + eps_push * n
        end

        # if terminated by max_bounces/Inf without exiting, still store what we have
        if !(exited[i] || backscatter[i])
            path_len[i] = sacc
            nhit_x[i] = hx
            nhit_z[i] = hz
            nhit_total[i] = hx + hz
            n_bounces[i] = nb
        end
    end

    return (exited=exited,              # transmitted atoms
            backscatter=backscatter,    # atoms that returned to the oven
            exit_pos=exit_pos,          # Final position at the moment the particle leaves the cavity
            exit_vel=exit_vel,          # Velocity vector at exit moment
            exit_t=exit_t,              # Total time spent inside cavity before exiting
            nhit_x=nhit_x,              # Number of collisions with x-walls
            nhit_z=nhit_z,              # Number of collisions with z-walls
            nhit_total=nhit_total,      # Total number of side-wall collisions
            path_len=path_len,          # Total geometric distance traveled inside cavity
            n_bounces=n_bounces         # Internal counter of total scattering events
            )
end

function simulate_cavity_ballistic(pos0::AbstractMatrix{<:Real},
                                   vel0::AbstractMatrix{<:Real},
                                   L::Real;
                                   x_furnace::Real = 2.0e-3,
                                   z_furnace::Real = 100e-6,
                                   max_steps::Int = 10,
                                   rng::AbstractRNG = Random.default_rng())  # rng unused, kept for symmetry

    N = size(pos0, 1)
    @assert size(pos0,2) == 3 && size(vel0,2) == 3 && size(vel0,1) == N

    xmin = -0.5 * Float64(x_furnace)
    xmax =  0.5 * Float64(x_furnace)
    zmin = -0.5 * Float64(z_furnace)
    zmax =  0.5 * Float64(z_furnace)
    L    = Float64(L)

    exited      = falses(N)  # y=L
    backscatter = falses(N)  # y=0
    lost        = falses(N)  # hit x/z wall before exiting

    exit_pos = zeros(Float64, N, 3)
    exit_vel = zeros(Float64, N, 3)
    exit_t   = zeros(Float64, N)
    path_len = zeros(Float64, N)

    for i in 1:N
        pos = SVector{3,Float64}(Float64(pos0[i,1]), Float64(pos0[i,2]), Float64(pos0[i,3]))
        vel = SVector{3,Float64}(Float64(vel0[i,1]), Float64(vel0[i,2]), Float64(vel0[i,3]))

        speed = norm(vel)
        if speed == 0.0
            continue
        end

        # In pure ballistic, one hit decides everything, but we keep a loop
        # in case you later add extra surfaces.
        for _ in 1:max_steps
            thit, hit_axis, hit_side = next_hit(pos, vel, xmin, xmax, L, zmin, zmax)
            if thit == Inf
                break
            end

            pos_hit = pos + thit * vel
            ttot = thit
            sacc = speed * thit

            if hit_axis == 2
                if hit_side == 1
                    exited[i] = true
                else
                    backscatter[i] = true
                end
                exit_pos[i,:] .= Tuple(pos_hit)
                exit_vel[i,:] .= Tuple(vel)
                exit_t[i] = ttot
                path_len[i] = sacc
            else
                # x or z wall reached first → lost (geometric rejection)
                lost[i] = true
                # you can record where it hit if you want:
                # exit_pos[i,:] .= Tuple(pos_hit)
                # exit_vel[i,:] .= Tuple(vel)
                # exit_t[i] = ttot
                path_len[i] = sacc
            end
            break
        end
    end

    return (exited=exited,
            backscatter=backscatter,
            lost=lost,
            exit_pos=exit_pos,
            exit_vel=exit_vel,
            exit_t=exit_t,
            path_len=path_len)
end

@inline function exit_angles(out; transmitted_only::Bool=true)
    idx = transmitted_only ? findall(out.exited) : eachindex(out.exit_t)
    v = out.exit_vel[idx, :]
    vx = v[:,1]; vy = v[:,2]; vz = v[:,3]

    # keep only forward-going vy>0 (should be true for y=L exits, but safe)
    m = vy .> 0
    vx = vx[m]; vy = vy[m]; vz = vz[m]

    speed = sqrt.(vx.^2 .+ vy.^2 .+ vz.^2)

    θx = atan.(vx ./ vy)
    θz = atan.(vz ./ vy)

    θ  = acos.(vy ./ speed)      # polar from +y
    ϕ  = atan.(vz, vx)           # atan2(vz, vx)

    return (θx=θx, θz=θz, θ=θ, ϕ=ϕ, speed=speed)
end

@inline transmission(out) = count(out.exited) / length(out.exited)

function compare_ballistic_diffusive(out_ball, out_diff)
    Tball = transmission(out_ball)
    Tdiff = transmission(out_diff)
    K = (Tball > 0) ? (Tdiff / Tball) : NaN
    return (T_ballistic=Tball, T_diffusive=Tdiff, Clausing=K)
end

@inline solid_angle_cone(θmax) = 2π * (1 - cos(θmax))

function brightness_proxy(out; x_furnace, z_furnace, θmax = 10e-3)
    A = Float64(x_furnace) * Float64(z_furnace)
    Ω = solid_angle_cone(θmax)

    ang = exit_angles(out; transmitted_only=true)
    θ = ang.θ

    Nincone = count(<=(θmax), θ)
    N = length(out.exited)

    # proxy radiance ~ fraction in cone / (A * Ω)
    B = (Nincone / N) / (A * Ω)
    return (B=B, Nincone=Nincone, Ω=Ω, A=A)
end

function brightness_reduction(out_ball, out_diff; x_furnace, z_furnace, θmax=10e-3)
    Bb = brightness_proxy(out_ball; x_furnace=x_furnace, z_furnace=z_furnace, θmax=θmax).B
    Bd = brightness_proxy(out_diff; x_furnace=x_furnace, z_furnace=z_furnace, θmax=θmax).B
    red = (Bb > 0) ? (Bd / Bb) : NaN
    return (B_ballistic=Bb, B_diffusive=Bd, reduction=red, θmax=θmax)
end

function collect_transmitted_inout_table(
    N_t::Int,
    rng::AbstractRNG,
    p::TheoreticalSimulation.EffusionParams,
    L::Real;
    model::Symbol = :diffusive,
    batch::Int = 50_000,
    x_furnace::Real = 2.0e-3,
    z_furnace::Real = 100e-6,
    # diffusive options
    max_bounces::Int = 50_000,
    eps_push::Float64 = 1e-12,
    collect_backscatter::Bool = false,)
    @assert model === :diffusive || model === :ballistic "model must be :diffusive or :ballistic"

    tbl = Matrix{Float64}(undef, N_t, 12)
    filled = 0

    # hoist constants/conversions once
    Lf = Float64(L)
    xf = Float64(x_furnace)
    zf = Float64(z_furnace)
    eps = Float64(eps_push)

    while filled < N_t
        pos0, vel0 = sample_initial_conditions(batch, rng, p)

        out = if model === :diffusive
            simulate_cavity_centered(pos0, vel0, Lf;
                x_furnace=xf,
                z_furnace=zf,
                max_bounces=max_bounces,
                eps_push=eps,
                collect_backscatter=collect_backscatter,
                rng=rng,
            )
        else
            simulate_cavity_ballistic(pos0, vel0, Lf;
                x_furnace=xf,
                z_furnace=zf,
            )
        end

        ex = out.exited            # BitVector
        ep = out.exit_pos          # N×3
        ev = out.exit_vel          # N×3

        @inbounds for i in 1:batch
            (filled >= N_t) && break
            ex[i] || continue

            filled += 1

            # initial state
            tbl[filled, 1] = pos0[i, 1]
            tbl[filled, 2] = 0.0
            tbl[filled, 3] = pos0[i, 3]
            tbl[filled, 4] = vel0[i, 1]
            tbl[filled, 5] = vel0[i, 2]
            tbl[filled, 6] = vel0[i, 3]

            # exit state
            tbl[filled, 7]  = ep[i, 1]
            tbl[filled, 8]  = ep[i, 2]
            tbl[filled, 9]  = ep[i, 3]
            tbl[filled, 10] = ev[i, 1]
            tbl[filled, 11] = ev[i, 2]
            tbl[filled, 12] = ev[i, 3]
        end
    end

    return tbl
end



@inline function passes_slit_from_row(row, y_slit, xh, zh)
    xf  = row[7];  yf  = row[8];  zf  = row[9]
    vxf = row[10]; vyf = row[11]; vzf = row[12]

    if vyf <= 0.0
        return false
    end
    t = (y_slit - yf) / vyf
    if t <= 0.0
        return false
    end
    xs = xf + vxf * t
    zs = zf + vzf * t
    return (abs(xs) <= xh) & (abs(zs) <= zh)
end

function collect_exit_particles_passing_slit_via_exit_table(
    N_slit::Int,
    rng::AbstractRNG,
    p::TheoreticalSimulation.EffusionParams,
    L::Real;
    y_slit::Real,
    x_slit::Real = 4.0e-3,
    z_slit::Real = 300e-6,
    model::Symbol = :diffusive,

    # stage-1 (exit collection) knobs
    N_exit_chunk::Int = 200_000,   # how many exited particles to collect per chunk
    batch::Int = 50_000,
    x_furnace::Real = 2.0e-3,
    z_furnace::Real = 100e-6,
    max_bounces::Int = 50_000,
    eps_push::Float64 = 1e-12,
    collect_backscatter::Bool = false,)
    @assert y_slit > L
    y_slit = Float64(y_slit)
    xh = 0.5 * Float64(x_slit)
    zh = 0.5 * Float64(z_slit)

    out_tbl = Matrix{Float64}(undef, N_slit, 6)
    filled = 0

    while filled < N_slit
        # Stage 1: collect a chunk of exited particles (Nx12 table)
        tbl12 = collect_transmitted_inout_table(
            N_exit_chunk, rng, p, L;
            model=model,
            batch=batch,
            x_furnace=x_furnace,
            z_furnace=z_furnace,
            max_bounces=max_bounces,
            eps_push=eps_push,
            collect_backscatter=collect_backscatter,
        )

        # Stage 2: filter those exits by downstream slit
        @inbounds for k in 1:size(tbl12, 1)
            (filled >= N_slit) && break
            row = @view tbl12[k, :]

            if passes_slit_from_row(row, y_slit, xh, zh)
                filled += 1
                # write exit-only columns: xf yf zf vxf vyf vzf
                out_tbl[filled, 1] = row[7]
                out_tbl[filled, 2] = row[8]
                out_tbl[filled, 3] = row[9]
                out_tbl[filled, 4] = row[10]
                out_tbl[filled, 5] = row[11]
                out_tbl[filled, 6] = row[12]
            end
        end
    end

    return out_tbl
end

# multi - threading

@inline function passes_slit_from_cols(xf, yf, zf, vxf, vyf, vzf, y_slit, xh, zh)
    vyf <= 0.0 && return false
    t = (y_slit - yf) / vyf
    t <= 0.0 && return false
    xs = xf + vxf*t
    zs = zf + vzf*t
    return (abs(xs) <= xh) & (abs(zs) <= zh)
end

@inline function passes_slit_from_exit(xf, zf, vxf, vyf, vzf, d, xh, zh)
    vyf <= 0.0 && return false
    invvy = 1.0 / vyf
    xs = xf + vxf * (d * invvy)
    zs = zf + vzf * (d * invvy)
    return (abs(xs) <= xh) & (abs(zs) <= zh)
end

function collect_exit_particles_passing_slit_threaded(
    N_slit::Int,
    rng::AbstractRNG,
    p::TheoreticalSimulation.EffusionParams,
    L::Real;
    d::Real,                       # d = y_slit - L  (your 0.224)
    x_slit::Real = 4.0e-3,
    z_slit::Real = 300e-6,
    model::Symbol = :diffusive,
    batch::Int = 200_000,          # try larger batch
    x_furnace::Real = 2.0e-3,
    z_furnace::Real = 100e-6,
    max_bounces::Int = 50_000,
    eps_push::Float64 = 1e-12,
    collect_backscatter::Bool = false,)
    @assert model === :diffusive || model === :ballistic

    d  = Float64(d)
    xh = 0.5 * Float64(x_slit)
    zh = 0.5 * Float64(z_slit)

    out_tbl = Matrix{Float64}(undef, N_slit, 6)
    filled = 0

    nt = Threads.nthreads()
    seeds = rand(rng, UInt64, nt)
    rngs = [Random.Xoshiro(seeds[t]) for t in 1:nt]

    # thread-local buffers (grow-only)
    local_bufs = [Matrix{Float64}(undef, 0, 6) for _ in 1:nt]

    next_report = 0.1         # next milestone (10%)

    while filled < N_slit
        # each thread produces accepted exits from one batch
        @threads for t in 1:nt
            rt = rngs[t]

            pos0, vel0 = sample_initial_conditions(batch, rt, p)

            out = if model === :diffusive
                simulate_cavity_centered(pos0, vel0, L;
                    x_furnace=x_furnace,
                    z_furnace=z_furnace,
                    max_bounces=max_bounces,
                    eps_push=eps_push,
                    collect_backscatter=collect_backscatter,
                    rng=rt)
            else
                simulate_cavity_ballistic(pos0, vel0, L;
                    x_furnace=x_furnace,
                    z_furnace=z_furnace)
            end

            ex = out.exited
            ep = out.exit_pos
            ev = out.exit_vel

            # pessimistic prealloc: at most batch rows
            buf = Matrix{Float64}(undef, batch, 6)
            n = 0

            @inbounds for i in 1:batch
                ex[i] || continue

                xf  = ep[i,1]; zf = ep[i,3]
                vxf = ev[i,1]; vyf = ev[i,2]; vzf = ev[i,3]

                if passes_slit_from_exit(xf, zf, vxf, vyf, vzf, d, xh, zh)
                    n += 1
                    buf[n,1] = xf
                    buf[n,2] = ep[i,2]   # yf (~L)
                    buf[n,3] = zf
                    buf[n,4] = vxf
                    buf[n,5] = vyf
                    buf[n,6] = vzf
                end
            end

            # store only used prefix
            local_bufs[t] = @view buf[1:n, :]
        end

        # serial merge
        for t in 1:nt
            buf = local_bufs[t]
            n = size(buf, 1)
            n == 0 && continue

            take = min(n, N_slit - filled)
            out_tbl[filled+1:filled+take, :] .= buf[1:take, :]
            filled += take
            filled >= N_slit && break
        end
    end

    return out_tbl
end


##################################################################################################

## Coil currents
Icoils = [0.00,
            1.00
];
nI = length(Icoils);

# Sample size: number of atoms arriving to the screen
const Ns = 100_000 ; 
@info "Number of MonteCarlo particles : $(Ns)\n"
# Monte Carlo generation of particles traversing the filtering SG-slit [x0 y0 z0 v0x v0y v0z]
crossing_slit = TheoreticalSimulation.generate_samples(Ns, effusion_params; v_pdf=:v3, rng = rng_set, multithreaded = false, base_seed = base_seed_set);

r0, v0 = sample_initial_conditions(Ns, rng_set, effusion_params)
L=400e-6
y_slit = L + y_FurnaceToSlit

# Diffusive (Lambertian)
out_diff = simulate_cavity_centered(r0, v0, L; x_furnace=x_furnace, z_furnace=z_furnace,rng=rng_set)

# Ballistic (geometric): new function
out_ball = simulate_cavity_ballistic(r0, v0, L; x_furnace=x_furnace, z_furnace=z_furnace)

# 1) transmissions + Clausing
cmp = compare_ballistic_diffusive(out_ball, out_diff)

# 2) angular distributions at y=L (arrays of angles)
ang_diff = exit_angles(out_diff)
ang_ball = exit_angles(out_ball)

# 3) brightness reduction in a chosen forward cone (e.g. 5 mrad)
bright = brightness_reduction(out_ball, out_diff; x_furnace=x_furnace, z_furnace=z_furnace, θmax=asin(effusion_params.sinθmax))

@show cmp
@show bright
@show mean(ang_diff.θ), mean(abs.(ang_diff.θx)), mean(abs.(ang_diff.θz))

@time tbl_diff = collect_transmitted_inout_table(Ns, rng_set, effusion_params, L;
    model=:diffusive, batch=3*Ns, x_furnace=x_furnace, z_furnace=z_furnace);
@time tbl_ball = collect_transmitted_inout_table(Ns, rng_set, effusion_params, L;
    model=:ballistic, batch=2*Ns, x_furnace=x_furnace, z_furnace=z_furnace)

data_exit_diff = tbl_diff[:,7:end];
TheoreticalSimulation.plot_velocity_stats(data_exit_diff, "Oven: Diffusive" , "velocity_pdf")





plot!(suptitle="Released from oven")

histogram(data_exit[:,6])


tbl_ball = collect_transmitted_inout_table(Ns, rng_set, effusion_params, L;
    model=:ballistic, batch=2*Ns, x_furnace=x_furnace, z_furnace=z_furnace)





tbl_exit = collect_exit_particles_passing_slit_threaded(
    Ns, rng_set, effusion_params, L;
    d=y_FurnaceToSlit, model=:diffusive, batch=500_000)




tbl = collect_exit_particles_passing_slit_via_exit_table(Ns, rng_set, effusion_params, L;
    x_slit=x_slit,
    y_slit=y_slit,
    z_slit=z_slit,
    model=:diffusive,
    batch=10_000,
    N_exit_chunk = 20_000,
    x_furnace=x_furnace,
    z_furnace=z_furnace,
)
