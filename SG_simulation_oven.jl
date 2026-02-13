# Simulation of atom trajectories in the Stern–Gerlach experiment
# Kelvin Titimbo
# Caltech
# February 2026

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
# using Interpolations, Roots, Loess, Optim
# using BSplineKit
# using Polynomials
# using DSP
# using LambertW, PolyLog
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
const OUTDIR    = joinpath(@__DIR__, "simulation_data", "OvenCanal"*RUN_STAMP);
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

Lcanal=400.0e-6
y_slit = Lcanal + y_FurnaceToSlit


## Initial Conditions
################################################################################################
@inline function AtomicBeamVelocity_v3(rng::AbstractRNG, p::TheoreticalSimulation.EffusionParams)::SVector{3,Float64}
    ϕ = TWOπ * rand(rng)
    θ = asin(sqrt(rand(rng)))
    v = sqrt(-2*p.α2 * (1 + lambertw((rand(rng)-1)*INV_E, -1)))
    sθ = sin(θ); cθ = cos(θ); sϕ = sin(ϕ); cϕ = cos(ϕ)
    return SVector(v*sθ*sϕ, v*cθ, v*sθ*cϕ)
end

@inline function sample_initial_conditions!(pos::Matrix{Float64},
                                            vel::Matrix{Float64},
                                            rng::AbstractRNG,
                                            p::TheoreticalSimulation.EffusionParams)
    N = size(pos, 1)
    @inbounds for i in 1:N
        # position
        pos[i,1] = x_furnace * (rand(rng) - 0.5)
        pos[i,2] = 0.0
        pos[i,3] = z_furnace * (rand(rng) - 0.5)

        # velocity
        v = AtomicBeamVelocity_v3(rng, p)
        vel[i,1] = v[1]
        vel[i,2] = v[2]
        vel[i,3] = v[3]
    end
    return nothing
end

@inline function sample_initial_conditions(N::Int,
                                           rng::AbstractRNG,
                                           p::TheoreticalSimulation.EffusionParams)
    pos = Matrix{Float64}(undef, N, 3)
    vel = Matrix{Float64}(undef, N, 3)
    sample_initial_conditions!(pos, vel, rng, p)
    return pos, vel
end

function sci_latex(N::Int; digits::Int=2)
    e = floor(Int, log10(N))
    m = N / 10.0^e
    m_str = round(m, digits=digits)
    return L"%$(m_str)\times 10^{%$(e)}"
end

function plot_statistics(alive::Matrix{Float64}, title::String, filename::String)
    @assert size(alive, 2) ≥ 6 "Expected at least 6 columns (x, y, z, vx, vy, vz)."
    No = size(alive,1)

    # --- Velocity magnitude and angles ---
    vxs, vys, vzs = eachcol(alive[:, 4:6])
    velocities = sqrt.(vxs.^2 .+ vys.^2 .+ vzs.^2)
    theta_vals = acos.(vzs ./ velocities) # polar angle
    phi_vals   = atan.(vys, vxs)          # azimuthal angle

    # Means
    mean_v, rms_v = mean(velocities), sqrt(mean(velocities.^2))
    mean_theta, mean_phi = mean(theta_vals), mean(phi_vals)

    # Histogram for velocities
    figa = histogram(velocities;
        bins = TheoreticalSimulation.FreedmanDiaconisBins(velocities),
        label = L"$v_0$", normalize = :pdf,
        xlabel = L"v_{0} \ (\mathrm{m/s})",
        alpha = 0.70,
    )
    vline!([mean_v], label = L"$\langle v_{0} \rangle = %$(round(mean_v, digits=1))\ \mathrm{m/s}$",
        line = (:black, :solid, 2))
    vline!([rms_v], label = L"$\sqrt{\langle v_{0}^2 \rangle} = %$(round(rms_v, digits=1))\ \mathrm{m/s}$",
        line = (:red, :dash, 3))

    figb = histogram(theta_vals;
        bins = TheoreticalSimulation.FreedmanDiaconisBins(theta_vals),
        label = L"$\theta_v$", normalize = :pdf,
        alpha = 0.70, xlabel = L"$\theta_{v}$"
    )
    vline!([mean_theta], label = L"$\langle \theta_{v} \rangle = %$(round(mean_theta/π, digits=3))\pi$",
        line = (:black, :solid, 2))

    figc = histogram(phi_vals;
        bins = TheoreticalSimulation.FreedmanDiaconisBins(phi_vals),
        label = L"$\phi_v$", normalize = :pdf,
        alpha = 0.70, xlabel = L"$\phi_{v}$"
    )
    vline!([mean_phi], label = L"$\langle \phi_{v} \rangle = %$(round(mean_phi/π, digits=3))\pi$",
        line = (:black, :solid, 2))

    # 2D Histogram of position (x, z)
    # --- 2D position histogram ---
    xs, zs = 1e3 .* alive[:, 1], 1e6 .* alive[:, 3]  # mm, μm
    figd = histogram2d(xs, zs;
        bins = (TheoreticalSimulation.FreedmanDiaconisBins(xs), TheoreticalSimulation.FreedmanDiaconisBins(zs)),
        show_empty_bins = true, color = :plasma,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -1.0:0.25:1.0, yticks = -50:10:50,
        colorbar_position = :bottom,
    )

    # --- 2D position histogram (for plotting) ---
    xs = 1e3 .* alive[:, 1]   # mm
    zs = 1e6 .* alive[:, 3]   # μm

    xbins = TheoreticalSimulation.FreedmanDiaconisBins(xs)
    zbins = TheoreticalSimulation.FreedmanDiaconisBins(zs)

    figd = histogram2d(xs, zs;
        bins = (xbins, zbins),
        show_empty_bins = true, color = :plasma,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -1.0:0.25:1.0, yticks = -50:10:50,
        colorbar_position = :bottom,
    )

    # xs, zs already defined
    x_edges = StatsBase.histrange(xs, xbins)   # edges vector for FD rule
    z_edges = StatsBase.histrange(zs, zbins)

    h2 = StatsBase.fit(Histogram, (xs, zs), (x_edges, z_edges))  # works
    W  = h2.weights

    z_prof = vec(sum(W; dims=1)) # integrate over x
    x_prof = vec(sum(W; dims=2)) # integrate over z

    z_centers = @views 0.5 .* (z_edges[1:end-1] .+ z_edges[2:end])
    x_centers = @views 0.5 .* (x_edges[1:end-1] .+ x_edges[2:end])

    dz = diff(z_edges)
    z_prof_pdf = z_prof ./ sum(z_prof .* dz)

    dx = diff(x_edges)
    x_prof_pdf = x_prof ./ sum(x_prof .* dx)

    figz = plot(z_centers, z_prof_pdf;
        seriestype = :steppre,
        xlabel = L"$z \ (\mathrm{\mu m})$",
        ylabel = L"$I(z)$",
        label = L"$\int dx\,\rho(x,z)$",
        linewidth = 1,
        xticks = -50:25:50,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
    )

    figx = plot(x_centers, x_prof_pdf;
        seriestype = :steppre,
        xlabel = L"$x \ (\mathrm{mm})$",
        ylabel = L"$I(x)$",
        label = L"$\int dz\,\rho(x,z)$",
        linewidth = 1,
        xticks = -1.0:0.5:1.0,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
    )

    # --- Velocity component histograms ---
    fige = histogram(vxs;
        bins = TheoreticalSimulation.FreedmanDiaconisBins(vxs), normalize = :pdf,
        label = L"$v_{0,x}$", alpha = 0.65, color = :orange,
        xlabel = L"$v_{0,x} \ (\mathrm{m/s})$"
    )
    figf = histogram(vys;
        bins = TheoreticalSimulation.FreedmanDiaconisBins(vys), normalize = :pdf,
        label = L"$v_{0,y}$", alpha = 0.65, color = :blue,
        xlabel = L"$v_{0,y} \ (\mathrm{m/s})$"
    )
    figg = histogram(vzs;
        bins = TheoreticalSimulation.FreedmanDiaconisBins(vzs), normalize = :pdf,
        label = L"$v_{0,z}$", alpha = 0.65, color = :red,
        xlabel = L"$v_{0,z} \ (\mathrm{m/s})$"
    )

    # Combine plots
    fig = plot(
        figa, fige, figb, figf, figc, figg, figd, figz, figx,
        layout = @layout([a1 a2; a3 a4; a5 a6; a7; a8 a9]),
        plot_title = L"%$(title) | $N =$ %$(sci_latex(No))",
        size = (650, 800),
        legendfontsize = 8,
        left_margin = 3mm,
    );
    display(fig)
    savefig(fig, joinpath(OUTDIR,"$(filename).$(FIG_EXT)"))

    return nothing
end


## ##########################################################
# +++++++ Diffusive Compact version ++++++++++++++++++++++++
@inline function time_to_plane_scalar(x::Float64, y::Float64, z::Float64,
                                      vx::Float64, vy::Float64, vz::Float64,
                                      axis::Int, bound::Float64)
    v = (axis == 1) ? vx : (axis == 2 ? vy : vz)
    if v == 0.0
        return Inf
    end
    p = (axis == 1) ? x : (axis == 2 ? y : z)
    t = (bound - p) / v
    return (t > 0.0) ? t : Inf
end

@inline function next_hit_scalar_exact(x::Float64, y::Float64, z::Float64,
                                       vx::Float64, vy::Float64, vz::Float64,
                                       xmin::Float64, xmax::Float64,
                                       L::Float64,
                                       zmin::Float64, zmax::Float64)

    tx0 = time_to_plane_scalar(x,y,z, vx,vy,vz, 1, xmin)
    tx1 = time_to_plane_scalar(x,y,z, vx,vy,vz, 1, xmax)

    ty0 = time_to_plane_scalar(x,y,z, vx,vy,vz, 2, 0.0)
    ty1 = time_to_plane_scalar(x,y,z, vx,vy,vz, 2, L)

    tz0 = time_to_plane_scalar(x,y,z, vx,vy,vz, 3, zmin)
    tz1 = time_to_plane_scalar(x,y,z, vx,vy,vz, 3, zmax)

    tmin = tx0; hit_axis = 1; hit_side = 0
    if tx1 < tmin; tmin = tx1; hit_axis = 1; hit_side = 1; end
    if ty0 < tmin; tmin = ty0; hit_axis = 2; hit_side = 0; end
    if ty1 < tmin; tmin = ty1; hit_axis = 2; hit_side = 1; end
    if tz0 < tmin; tmin = tz0; hit_axis = 3; hit_side = 0; end
    if tz1 < tmin; tmin = tz1; hit_axis = 3; hit_side = 1; end

    return tmin, hit_axis, hit_side
end

@inline function sample_lambertian_dir_axis(rng::AbstractRNG, hit_axis::Int, hit_side::Int)
    u1 = rand(rng)
    u2 = rand(rng)
    r  = sqrt(u1)
    ϕ  = 2π * u2
    sϕ, cϕ = sincos(ϕ)

    lx = r * cϕ
    ly = r * sϕ
    lz = sqrt(1.0 - u1)

    if hit_axis == 1
        nx = (hit_side == 0) ? 1.0 : -1.0
        return (nx*lz, lx, ly)
    else
        nz = (hit_side == 0) ? 1.0 : -1.0
        return (lx, ly, nz*lz)
    end
end

function simulate_cavity_diffusive_exit!(exited::BitVector,
                                         exit_pos::Matrix{Float64},
                                         exit_vel::Matrix{Float64},
                                         pos0::AbstractMatrix{<:Real},
                                         vel0::AbstractMatrix{<:Real},
                                         L::Real;
                                         max_bounces::Int = 50_000,
                                         eps_push::Float64 = 1e-12,
                                         rng::AbstractRNG = Random.default_rng())

    N = size(pos0, 1)
    @assert size(pos0,2) == 3 && size(vel0,2) == 3 && size(vel0,1) == N
    @assert length(exited) == N
    @assert size(exit_pos) == (N, 3)
    @assert size(exit_vel) == (N, 3)

    xmin = -0.5 * Float64(x_furnace)
    xmax =  0.5 * Float64(x_furnace)
    zmin = -0.5 * Float64(z_furnace)
    zmax =  0.5 * Float64(z_furnace)
    Lf   = Float64(L)
    eps  = Float64(eps_push)

    fill!(exited, false)

    @inbounds for i in 1:N
        # if pos0/vel0 are Float64 matrices, these are already Float64
        x  = pos0[i,1]
        y  = pos0[i,2]
        z  = pos0[i,3]
        vx = vel0[i,1]
        vy = vel0[i,2]
        vz = vel0[i,3]

        s2 = vx*vx + vy*vy + vz*vz
        s2 == 0.0 && continue
        speed = sqrt(s2)

        nb = 0
        while nb <= max_bounces
            thit, hit_axis, hit_side = next_hit_scalar_exact(x,y,z, vx,vy,vz, xmin,xmax, Lf, zmin,zmax)
            thit == Inf && break

            # move to collision point
            xh = muladd(thit, vx, x)
            yh = muladd(thit, vy, y)
            zh = muladd(thit, vz, z)

            if hit_axis == 2
                if hit_side == 1
                    exited[i] = true
                    exit_pos[i,1] = xh; exit_pos[i,2] = yh; exit_pos[i,3] = zh
                    exit_vel[i,1] = vx; exit_vel[i,2] = vy; exit_vel[i,3] = vz
                end
                break
            end

            nb += 1
            dvx, dvy, dvz = sample_lambertian_dir_axis(rng, hit_axis, hit_side)
            vx = speed * dvx
            vy = speed * dvy
            vz = speed * dvz

            # push slightly inward
            if hit_axis == 1
                nx = (hit_side == 0) ? 1.0 : -1.0
                x = xh + eps * nx
                y = yh
                z = zh
            else
                nz = (hit_side == 0) ? 1.0 : -1.0
                x = xh
                y = yh
                z = zh + eps * nz
            end
        end
    end

    return nothing
end

function collect_transmitted_diffusive_compact(
    N_t::Int,
    rng::AbstractRNG,
    p::TheoreticalSimulation.EffusionParams,
    L::Real;
    batch::Int = 50_000,
    # diffusive options
    max_bounces::Int = 50_000,
    eps_push::Float64 = 1e-12,)
    tbl = Matrix{Float64}(undef, N_t, 6)
    filled = 0

    Lf  = Float64(L)
    eps = Float64(eps_push)

    # reusable batch buffers
    pos0 = Matrix{Float64}(undef, batch, 3)
    vel0 = Matrix{Float64}(undef, batch, 3)

    # reusable outputs for exit-only simulator
    exited   = falses(batch)
    exit_pos = Matrix{Float64}(undef, batch, 3)
    exit_vel = Matrix{Float64}(undef, batch, 3)

    while filled < N_t
        sample_initial_conditions!(pos0, vel0, rng, p)

        simulate_cavity_diffusive_exit!(exited, exit_pos, exit_vel, pos0, vel0, Lf;
                                        max_bounces=max_bounces,
                                        eps_push=eps,
                                        rng=rng)

        @inbounds for i in 1:batch
            filled >= N_t && break
            exited[i] || continue

            filled += 1
            tbl[filled,1] = exit_pos[i,1]
            tbl[filled,2] = exit_pos[i,2]
            tbl[filled,3] = exit_pos[i,3]
            tbl[filled,4] = exit_vel[i,1]
            tbl[filled,5] = exit_vel[i,2]
            tbl[filled,6] = exit_vel[i,3]
        end
    end

    return tbl
end

function collect_slitcrossing_diffusive_compact(n_t::Int,
                                                rng::AbstractRNG,
                                                p::TheoreticalSimulation.EffusionParams,
                                                L::Real;
                                                batch::Int = 50_000,
                                                # diffusive options (passed through)
                                                max_bounces::Int = 50_000,
                                                eps_push::Float64 = 1e-12,
                                                log_every::Int = 1)
    # Output rows are exit-state at cavity exit: [x y z vx vy vz]
    out = Matrix{Float64}(undef, n_t, 6)

    half_x = 0.5 * x_slit   # uses your const globals
    half_z = 0.5 * z_slit
    yFS    = Float64(y_FurnaceToSlit)

    filled = 0
    iter   = 0

    # Reuse buffers so we don't allocate per batch
    pos0 = Matrix{Float64}(undef, batch, 3)
    vel0 = Matrix{Float64}(undef, batch, 3)

    exited   = falses(batch)
    exit_pos = Matrix{Float64}(undef, batch, 3)
    exit_vel = Matrix{Float64}(undef, batch, 3)

    while filled < n_t
        iter += 1

        # generate batch
        sample_initial_conditions!(pos0, vel0, rng, p)

        # simulate diffusive, exit-only (fills exited/exit_pos/exit_vel)
        simulate_cavity_diffusive_exit!(exited, exit_pos, exit_vel, pos0, vel0, L;
                                        max_bounces=max_bounces,
                                        eps_push=eps_push,
                                        rng=rng)

        # filter by crossing the slit plane at y = y_FurnaceToSlit downstream
        @inbounds for i in 1:batch
            exited[i] || continue

            vy = exit_vel[i, 2]          # always > 0
            t  = yFS / vy

            # z first (narrower slit)
            z0  = exit_pos[i, 3]
            vz  = exit_vel[i, 3]
            z_at = muladd(vz, t, z0)

            if (-half_z < z_at < half_z)
                x0  = exit_pos[i, 1]
                vx  = exit_vel[i, 1]
                x_at = muladd(vx, t, x0)

                if (-half_x < x_at < half_x)
                    filled += 1
                    # write [x y z vx vy vz] into output
                    out[filled, 1] = exit_pos[i, 1]
                    out[filled, 2] = exit_pos[i, 2]
                    out[filled, 3] = exit_pos[i, 3]
                    out[filled, 4] = exit_vel[i, 1]
                    out[filled, 5] = exit_vel[i, 2]
                    out[filled, 6] = exit_vel[i, 3]
                    filled >= n_t && break
                end
            end
        end

        if log_every > 0 && (iter % log_every == 0)
            acc = filled / (iter * batch)
            @info "Accumulated $filled / $n_t (accept≈$(round(100*acc; digits=5))%)"
        end
    end

    return out
end

function collect_slitcrossing_diffusive_compact_threads(n_t::Int,
                                                             rng0::AbstractRNG,
                                                             p::TheoreticalSimulation.EffusionParams,
                                                             L::Real;
                                                             batch::Int = 50_000,
                                                             max_bounces::Int = 50_000,
                                                             eps_push::Float64 = 1e-12,
                                                             log_every::Int = 1)

    out = Matrix{Float64}(undef, n_t, 6)

    half_x = 0.5 * x_slit
    half_z = 0.5 * z_slit
    yFS    = Float64(y_FurnaceToSlit)
    Lf     = Float64(L)
    eps    = Float64(eps_push)

    nT = Threads.nthreads()

    # Per-thread RNGs (only rng0 used here, single-threaded)
    seeds = rand(rng0, UInt64, nT)
    rngs  = [MersenneTwister(seeds[t]) for t in 1:nT]

    filled = Threads.Atomic{Int}(0)

    # (Optional) stats for logging
    local_iters    = zeros(Int, nT)
    local_attempts = zeros(Int, nT)

    Threads.@threads for tid in 1:nT
        rng = rngs[tid]

        pos0 = Matrix{Float64}(undef, batch, 3)
        vel0 = Matrix{Float64}(undef, batch, 3)
        exited   = falses(batch)
        exit_pos = Matrix{Float64}(undef, batch, 3)
        exit_vel = Matrix{Float64}(undef, batch, 3)

        stop = false
        while !stop
            # global termination check (cheap)
            filled[] >= n_t && break

            local_iters[tid] += 1
            local_attempts[tid] += batch

            sample_initial_conditions!(pos0, vel0, rng, p)

            simulate_cavity_diffusive_exit!(exited, exit_pos, exit_vel, pos0, vel0, Lf;
                                            max_bounces=max_bounces,
                                            eps_push=eps,
                                            rng=rng)

            @inbounds for i in 1:batch
                exited[i] || continue

                # Slit crossing at y = y_FurnaceToSlit downstream from exit plane
                vy = exit_vel[i,2]
                t  = yFS / vy

                # z first (narrower)
                z_at = muladd(exit_vel[i,3], t, exit_pos[i,3])
                (-half_z < z_at < half_z) || continue

                x_at = muladd(exit_vel[i,1], t, exit_pos[i,1])
                (-half_x < x_at < half_x) || continue

                # Reserve a unique output row (1-based)
                idx = Threads.atomic_add!(filled, 1) + 1
                if idx <= n_t
                    out[idx,1] = exit_pos[i,1]
                    out[idx,2] = exit_pos[i,2]
                    out[idx,3] = exit_pos[i,3]
                    out[idx,4] = exit_vel[i,1]
                    out[idx,5] = exit_vel[i,2]
                    out[idx,6] = exit_vel[i,3]
                else
                    stop = true
                    break
                end
            end

            if log_every > 0 && tid == 1 && (local_iters[tid] % log_every == 0)
                total_filled   = min(filled[], n_t)
                total_attempts = sum(local_attempts)
                acc = total_filled / max(total_attempts, 1)
                @info "Accumulated $total_filled / $n_t (accept≈$(round(100*acc; digits=5))%)"
            end
        end
    end

    return out
end

## #########################################################
# +++++++ Reflective Compact version +++++++++++++++++++++++
# ---------------------------------------------------------------------------
# Ballistic / reflective (specular elastic) compact version
# Reuses:
#   time_to_plane_scalar(...)
#   next_hit_scalar_exact(...)
# ---------------------------------------------------------------------------

@inline function reflect_specular!(vx::Float64, vy::Float64, vz::Float64,
                                  hit_axis::Int)
    # hit_axis: 1 => x-wall, 2 => y-wall, 3 => z-wall
    if hit_axis == 1
        vx = -vx
    elseif hit_axis == 2
        vy = -vy
    else
        vz = -vz
    end
    return vx, vy, vz
end

function simulate_cavity_reflective_exit!(exited::BitVector,
                                                     exit_pos::Matrix{Float64},
                                                     exit_vel::Matrix{Float64},
                                                     pos0::AbstractMatrix{<:Real},
                                                     vel0::AbstractMatrix{<:Real},
                                                     L::Real;
                                                     max_bounces::Int = 50_000,
                                                     eps_push::Float64 = 1e-12)

    N = size(pos0, 1)
    @assert size(pos0,2) == 3 && size(vel0,2) == 3 && size(vel0,1) == N
    @assert length(exited) == N
    @assert size(exit_pos) == (N, 3)
    @assert size(exit_vel) == (N, 3)

    xmin = -0.5 * Float64(x_furnace)
    xmax =  0.5 * Float64(x_furnace)
    zmin = -0.5 * Float64(z_furnace)
    zmax =  0.5 * Float64(z_furnace)
    Lf   = Float64(L)
    eps  = Float64(eps_push)

    fill!(exited, false)

    @inbounds for i in 1:N
        x  = Float64(pos0[i,1])
        y  = Float64(pos0[i,2])
        z  = Float64(pos0[i,3])
        vx = Float64(vel0[i,1])
        vy = Float64(vel0[i,2])
        vz = Float64(vel0[i,3])

        s2 = vx*vx + vy*vy + vz*vz
        s2 == 0.0 && continue

        nb = 0
        while nb <= max_bounces
            thit, hit_axis, hit_side = next_hit_scalar_exact(
                x,y,z, vx,vy,vz, xmin,xmax, Lf, zmin,zmax
            )
            thit == Inf && break

            xh = muladd(thit, vx, x)
            yh = muladd(thit, vy, y)
            zh = muladd(thit, vz, z)

            if hit_axis == 2
                if hit_side == 1
                    # y = L : transmitted
                    exited[i] = true
                    exit_pos[i,1] = xh; exit_pos[i,2] = yh; exit_pos[i,3] = zh
                    exit_vel[i,1] = vx; exit_vel[i,2] = vy; exit_vel[i,3] = vz
                end
                # y = 0 : LOST (open inlet) OR y=L after recording
                break
            end

            nb += 1

            # specular reflection on x/z walls only
            if hit_axis == 1
                vx = -vx
                nx = (hit_side == 0) ? 1.0 : -1.0
                x = xh + eps * nx; y = yh; z = zh
            else
                # hit_axis == 3
                vz = -vz
                nz = (hit_side == 0) ? 1.0 : -1.0
                x = xh; y = yh; z = zh + eps * nz
            end
        end
    end

    return nothing
end

function collect_transmitted_reflective_compact(
    N_t::Int,
    rng::AbstractRNG,
    p::TheoreticalSimulation.EffusionParams,
    L::Real;
    batch::Int = 50_000,
    max_bounces::Int = 50_000,
    eps_push::Float64 = 1e-12)

    tbl = Matrix{Float64}(undef, N_t, 6)
    filled = 0

    Lf  = Float64(L)
    eps = Float64(eps_push)

    # reusable batch buffers
    pos0 = Matrix{Float64}(undef, batch, 3)
    vel0 = Matrix{Float64}(undef, batch, 3)

    exited   = falses(batch)
    exit_pos = Matrix{Float64}(undef, batch, 3)
    exit_vel = Matrix{Float64}(undef, batch, 3)

    while filled < N_t
        sample_initial_conditions!(pos0, vel0, rng, p)

        simulate_cavity_reflective_exit!(exited, exit_pos, exit_vel, pos0, vel0, Lf;
                                         max_bounces=max_bounces,
                                         eps_push=eps)

        @inbounds for i in 1:batch
            filled >= N_t && break
            exited[i] || continue

            filled += 1
            tbl[filled,1] = exit_pos[i,1]
            tbl[filled,2] = exit_pos[i,2]
            tbl[filled,3] = exit_pos[i,3]
            tbl[filled,4] = exit_vel[i,1]
            tbl[filled,5] = exit_vel[i,2]
            tbl[filled,6] = exit_vel[i,3]
        end
    end

    return tbl
end

function collect_slitcrossing_reflective_compact(n_t::Int,
                                                 rng::AbstractRNG,
                                                 p::TheoreticalSimulation.EffusionParams,
                                                 L::Real;
                                                 batch::Int = 50_000,
                                                 max_bounces::Int = 50_000,
                                                 eps_push::Float64 = 1e-12,
                                                 log_every::Int = 1)

    out = Matrix{Float64}(undef, n_t, 6)

    half_x = 0.5 * x_slit
    half_z = 0.5 * z_slit
    yFS    = Float64(y_FurnaceToSlit)

    filled = 0
    iter   = 0

    pos0 = Matrix{Float64}(undef, batch, 3)
    vel0 = Matrix{Float64}(undef, batch, 3)

    exited   = falses(batch)
    exit_pos = Matrix{Float64}(undef, batch, 3)
    exit_vel = Matrix{Float64}(undef, batch, 3)

    while filled < n_t
        iter += 1

        sample_initial_conditions!(pos0, vel0, rng, p)

        simulate_cavity_reflective_exit!(exited, exit_pos, exit_vel, pos0, vel0, L;
                                         max_bounces=max_bounces,
                                         eps_push=eps_push)

        @inbounds for i in 1:batch
            exited[i] || continue

            vy = exit_vel[i,2]
            # If your geometry guarantees exit implies vy>0, this is safe.
            # Otherwise, guard:
            vy <= 0.0 && continue

            t = yFS / vy

            z_at = muladd(exit_vel[i,3], t, exit_pos[i,3])
            if (-half_z < z_at < half_z)
                x_at = muladd(exit_vel[i,1], t, exit_pos[i,1])
                if (-half_x < x_at < half_x)
                    filled += 1
                    out[filled,1] = exit_pos[i,1]
                    out[filled,2] = exit_pos[i,2]
                    out[filled,3] = exit_pos[i,3]
                    out[filled,4] = exit_vel[i,1]
                    out[filled,5] = exit_vel[i,2]
                    out[filled,6] = exit_vel[i,3]
                    filled >= n_t && break
                end
            end
        end

        if log_every > 0 && (iter % log_every == 0)
            acc = filled / (iter * batch)
            @info "Accumulated $filled / $n_t (accept≈$(round(100*acc; digits=5))%)"
        end
    end

    return out
end

function collect_slitcrossing_reflective_compact_threads(n_t::Int,
                                                              rng0::AbstractRNG,
                                                              p::TheoreticalSimulation.EffusionParams,
                                                              L::Real;
                                                              batch::Int = 50_000,
                                                              max_bounces::Int = 50_000,
                                                              eps_push::Float64 = 1e-12,
                                                              log_every::Int = 1)

    out = Matrix{Float64}(undef, n_t, 6)

    half_x = 0.5 * x_slit
    half_z = 0.5 * z_slit
    yFS    = Float64(y_FurnaceToSlit)
    Lf     = Float64(L)
    eps    = Float64(eps_push)

    nT = Threads.nthreads()

    # Per-thread RNGs (reflective sim doesn't need rng, but sampler does)
    seeds = rand(rng0, UInt64, nT)
    rngs  = [MersenneTwister(seeds[t]) for t in 1:nT]

    filled = Threads.Atomic{Int}(0)

    # (Optional) stats for logging
    local_iters    = zeros(Int, nT)
    local_attempts = zeros(Int, nT)

    Threads.@threads for tid in 1:nT
        rng = rngs[tid]

        pos0 = Matrix{Float64}(undef, batch, 3)
        vel0 = Matrix{Float64}(undef, batch, 3)

        exited   = falses(batch)
        exit_pos = Matrix{Float64}(undef, batch, 3)
        exit_vel = Matrix{Float64}(undef, batch, 3)

        stop = false
        while !stop
            filled[] >= n_t && break

            local_iters[tid] += 1
            local_attempts[tid] += batch

            sample_initial_conditions!(pos0, vel0, rng, p)

            simulate_cavity_reflective_exit!(exited, exit_pos, exit_vel, pos0, vel0, Lf;
                                             max_bounces=max_bounces,
                                             eps_push=eps)

            @inbounds for i in 1:batch
                exited[i] || continue

                vy = exit_vel[i,2]
                t  = yFS / vy

                z_at = muladd(exit_vel[i,3], t, exit_pos[i,3])
                (-half_z < z_at < half_z) || continue

                x_at = muladd(exit_vel[i,1], t, exit_pos[i,1])
                (-half_x < x_at < half_x) || continue

                idx = Threads.atomic_add!(filled, 1) + 1
                if idx <= n_t
                    out[idx,1] = exit_pos[i,1]
                    out[idx,2] = exit_pos[i,2]
                    out[idx,3] = exit_pos[i,3]
                    out[idx,4] = exit_vel[i,1]
                    out[idx,5] = exit_vel[i,2]
                    out[idx,6] = exit_vel[i,3]
                else
                    stop = true
                    break
                end
            end

            if log_every > 0 && tid == 1 && (local_iters[tid] % log_every == 0)
                total_filled   = min(filled[], n_t)
                total_attempts = sum(local_attempts)
                acc = total_filled / max(total_attempts, 1)
                @info "Accumulated $total_filled / $n_t (accept≈$(round(100*acc; digits=5))%)"
            end
        end
    end

    return out
end

# ---------------------------------------------------------------------------
# Ballistic "kill-on-collision" (absorbing walls) model:
# - if first boundary hit is y=L => exit (accepted)
# - else => killed (rejected)
#
# Reuses:
#   time_to_plane_scalar(...)
#   next_hit_scalar_exact(...)
# ---------------------------------------------------------------------------

function simulate_cavity_kill_on_collision_exit!(exited::BitVector,
                                                exit_pos::Matrix{Float64},
                                                exit_vel::Matrix{Float64},
                                                pos0::AbstractMatrix{<:Real},
                                                vel0::AbstractMatrix{<:Real},
                                                L::Real)

    N = size(pos0, 1)
    @assert size(pos0,2) == 3 && size(vel0,2) == 3 && size(vel0,1) == N
    @assert length(exited) == N
    @assert size(exit_pos) == (N, 3)
    @assert size(exit_vel) == (N, 3)

    xmin = -0.5 * Float64(x_furnace)
    xmax =  0.5 * Float64(x_furnace)
    zmin = -0.5 * Float64(z_furnace)
    zmax =  0.5 * Float64(z_furnace)
    Lf   = Float64(L)

    fill!(exited, false)

    @inbounds for i in 1:N
        x  = Float64(pos0[i,1])
        y  = Float64(pos0[i,2])
        z  = Float64(pos0[i,3])
        vx = Float64(vel0[i,1])
        vy = Float64(vel0[i,2])
        vz = Float64(vel0[i,3])

        # must move downstream to ever exit
        vy <= 0.0 && continue

        s2 = vx*vx + vy*vy + vz*vz
        s2 == 0.0 && continue

        thit, hit_axis, hit_side = next_hit_scalar_exact(
            x,y,z, vx,vy,vz, xmin,xmax, Lf, zmin,zmax
        )
        thit == Inf && continue

        # move to first boundary hit point
        xh = muladd(thit, vx, x)
        yh = muladd(thit, vy, y)
        zh = muladd(thit, vz, z)

        # accept only if first hit is the exit plane y=L
        if hit_axis == 2 && hit_side == 1
            exited[i] = true
            exit_pos[i,1] = xh; exit_pos[i,2] = yh; exit_pos[i,3] = zh
            exit_vel[i,1] = vx; exit_vel[i,2] = vy; exit_vel[i,3] = vz
        end
        # else: killed => exited[i] remains false
    end

    return nothing
end

function collect_transmitted_kill_on_collision_compact(
    N_t::Int,
    rng::AbstractRNG,
    p::TheoreticalSimulation.EffusionParams,
    L::Real;
    batch::Int = 50_000)

    tbl = Matrix{Float64}(undef, N_t, 6)
    filled = 0

    Lf = Float64(L)

    pos0 = Matrix{Float64}(undef, batch, 3)
    vel0 = Matrix{Float64}(undef, batch, 3)

    exited   = falses(batch)
    exit_pos = Matrix{Float64}(undef, batch, 3)
    exit_vel = Matrix{Float64}(undef, batch, 3)

    while filled < N_t
        sample_initial_conditions!(pos0, vel0, rng, p)

        simulate_cavity_kill_on_collision_exit!(exited, exit_pos, exit_vel, pos0, vel0, Lf)

        @inbounds for i in 1:batch
            filled >= N_t && break
            exited[i] || continue

            filled += 1
            tbl[filled,1] = exit_pos[i,1]
            tbl[filled,2] = exit_pos[i,2]
            tbl[filled,3] = exit_pos[i,3]
            tbl[filled,4] = exit_vel[i,1]
            tbl[filled,5] = exit_vel[i,2]
            tbl[filled,6] = exit_vel[i,3]
        end
    end

    return tbl
end

function collect_slitcrossing_kill_on_collision_compact(n_t::Int,
                                                        rng::AbstractRNG,
                                                        p::TheoreticalSimulation.EffusionParams,
                                                        L::Real;
                                                        batch::Int = 50_000,
                                                        log_every::Int = 1)

    out = Matrix{Float64}(undef, n_t, 6)

    half_x = 0.5 * x_slit
    half_z = 0.5 * z_slit
    yFS    = Float64(y_FurnaceToSlit)
    Lf     = Float64(L)

    filled = 0
    iter   = 0

    pos0 = Matrix{Float64}(undef, batch, 3)
    vel0 = Matrix{Float64}(undef, batch, 3)

    exited   = falses(batch)
    exit_pos = Matrix{Float64}(undef, batch, 3)
    exit_vel = Matrix{Float64}(undef, batch, 3)

    while filled < n_t
        iter += 1

        sample_initial_conditions!(pos0, vel0, rng, p)

        simulate_cavity_kill_on_collision_exit!(exited, exit_pos, exit_vel, pos0, vel0, Lf)

        @inbounds for i in 1:batch
            exited[i] || continue

            vy = exit_vel[i,2]
            vy <= 0.0 && continue

            t = yFS / vy

            z_at = muladd(exit_vel[i,3], t, exit_pos[i,3])
            if (-half_z < z_at < half_z)
                x_at = muladd(exit_vel[i,1], t, exit_pos[i,1])
                if (-half_x < x_at < half_x)
                    filled += 1
                    out[filled,1] = exit_pos[i,1]
                    out[filled,2] = exit_pos[i,2]
                    out[filled,3] = exit_pos[i,3]
                    out[filled,4] = exit_vel[i,1]
                    out[filled,5] = exit_vel[i,2]
                    out[filled,6] = exit_vel[i,3]
                    filled >= n_t && break
                end
            end
        end

        if log_every > 0 && (iter % log_every == 0)
            acc = filled / (iter * batch)
            @info "Accumulated $filled / $n_t (accept≈$(round(100*acc; digits=5))%)"
        end
    end

    return out
end

function collect_slitcrossing_kill_on_collision_compact_threads(n_t::Int,
                                                                     rng0::AbstractRNG,
                                                                     p::TheoreticalSimulation.EffusionParams,
                                                                     L::Real;
                                                                     batch::Int = 50_000,
                                                                     log_every::Int = 1)

    out = Matrix{Float64}(undef, n_t, 6)

    half_x = 0.5 * x_slit
    half_z = 0.5 * z_slit
    yFS    = Float64(y_FurnaceToSlit)
    Lf     = Float64(L)

    nT = Threads.nthreads()

    # Per-thread RNGs (kill model doesn't need rng, but sampler does)
    seeds = rand(rng0, UInt64, nT)
    rngs  = [MersenneTwister(seeds[t]) for t in 1:nT]

    filled = Threads.Atomic{Int}(0)

    # (Optional) stats for logging
    local_iters    = zeros(Int, nT)
    local_attempts = zeros(Int, nT)

    Threads.@threads for tid in 1:nT
        rng = rngs[tid]

        pos0 = Matrix{Float64}(undef, batch, 3)
        vel0 = Matrix{Float64}(undef, batch, 3)

        exited   = falses(batch)
        exit_pos = Matrix{Float64}(undef, batch, 3)
        exit_vel = Matrix{Float64}(undef, batch, 3)

        stop = false
        while !stop
            filled[] >= n_t && break

            local_iters[tid] += 1
            local_attempts[tid] += batch

            sample_initial_conditions!(pos0, vel0, rng, p)

            simulate_cavity_kill_on_collision_exit!(exited, exit_pos, exit_vel, pos0, vel0, Lf)

            @inbounds for i in 1:batch
                exited[i] || continue

                vy = exit_vel[i,2]
                t  = yFS / vy

                z_at = muladd(exit_vel[i,3], t, exit_pos[i,3])
                (-half_z < z_at < half_z) || continue

                x_at = muladd(exit_vel[i,1], t, exit_pos[i,1])
                (-half_x < x_at < half_x) || continue

                idx = Threads.atomic_add!(filled, 1) + 1
                if idx <= n_t
                    out[idx,1] = exit_pos[i,1]
                    out[idx,2] = exit_pos[i,2]
                    out[idx,3] = exit_pos[i,3]
                    out[idx,4] = exit_vel[i,1]
                    out[idx,5] = exit_vel[i,2]
                    out[idx,6] = exit_vel[i,3]
                else
                    stop = true
                    break
                end
            end

            if log_every > 0 && tid == 1 && (local_iters[tid] % log_every == 0)
                total_filled   = min(filled[], n_t)
                total_attempts = sum(local_attempts)
                acc = total_filled / max(total_attempts, 1)
                @info "Accumulated $total_filled / $n_t (accept≈$(round(100*acc; digits=5))%)"
            end
        end
    end

    return out
end

## TEST
"""
Generic transmission estimator for any exit-only simulator of signature:
    sim!(exited, exit_pos, exit_vel, pos0, vel0, L; kwargs...)

Returns (T, σT, n_exit, n_attempts).
"""
function estimate_transmission_exit_only(sim!::Function,
                                         rng::AbstractRNG,
                                         p::TheoreticalSimulation.EffusionParams,
                                         L::Real;
                                         n_attempts::Int = 2_000_000,
                                         batch::Int = 50_000,
                                         kwargs...)

    Lf = Float64(L)

    pos0 = Matrix{Float64}(undef, batch, 3)
    vel0 = Matrix{Float64}(undef, batch, 3)

    exited   = falses(batch)
    exit_pos = Matrix{Float64}(undef, batch, 3)
    exit_vel = Matrix{Float64}(undef, batch, 3)

    n_exit = 0
    done = 0

    while done < n_attempts
        b = min(batch, n_attempts - done)

        sample_initial_conditions!(pos0, vel0, rng, p)

        sim!(exited, exit_pos, exit_vel,
             view(pos0, 1:b, :),
             view(vel0, 1:b, :),
             Lf; kwargs...)

        @inbounds for i in 1:b
            n_exit += exited[i]
        end

        done += b
    end

    T = n_exit / n_attempts
    σT = sqrt(T * (1 - T) / n_attempts)
    return T, σT, n_exit, n_attempts
end

Td, σd, _, _ = estimate_transmission_exit_only(simulate_cavity_diffusive_exit!,
                                               rng_set, effusion_params, Lcanal;
                                               n_attempts=1_000_000,
                                               max_bounces=500_000,
                                               eps_push=1e-12,
                                               rng=rng_set)

Tr, σr, _, _ = estimate_transmission_exit_only(simulate_cavity_reflective_exit!,
                                               rng_set, effusion_params, Lcanal;
                                               n_attempts=1_000_000,
                                               max_bounces=500_000,
                                               eps_push=1e-12)

Tk, σk, _, _ = estimate_transmission_exit_only(simulate_cavity_kill_on_collision_exit!,
                                               rng_set, effusion_params, Lcanal;
                                               n_attempts=1_000_000)

@info "Exit transmission summary:"
@info "  Diffusive   T=$(Td) ± $(σd)"
@info "  Reflective  T=$(Tr) ± $(σr)"
@info "  Kill-coll   T=$(Tk) ± $(σk)"
@info "  Ratios:  (diff/kill)=$(Td/Tk), (refl/kill)=$(Tr/Tk)"


## ##########################################################

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

function simulate_cavity_diffusive(pos0::AbstractMatrix{<:Real},
                                  vel0::AbstractMatrix{<:Real},
                                  L::Real;
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
                                   max_steps::Int = 10)

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

function brightness_proxy(out; θmax = 10e-3)
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

function brightness_reduction(out_ball, out_diff; θmax=10e-3)
    Bb = brightness_proxy(out_ball; θmax=θmax).B
    Bd = brightness_proxy(out_diff; θmax=θmax).B
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
    # diffusive options
    max_bounces::Int = 50_000,
    eps_push::Float64 = 1e-12,
    collect_backscatter::Bool = false,)
    @assert model === :diffusive || model === :ballistic "model must be :diffusive or :ballistic"

    tbl = Matrix{Float64}(undef, N_t, 12)
    filled = 0

    # hoist constants/conversions once
    Lf = Float64(L)
    eps = Float64(eps_push)

    while filled < N_t
        pos0, vel0 = sample_initial_conditions(batch, rng, p)

        out = if model === :diffusive
            simulate_cavity_diffusive(pos0, vel0, Lf;
                max_bounces=max_bounces,
                eps_push=eps,
                collect_backscatter=collect_backscatter,
                rng=rng,
            )
        else
            simulate_cavity_ballistic(pos0, vel0, Lf)
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

function collect_transmitted_exit_table(
    N_t::Int,
    rng::AbstractRNG,
    p::TheoreticalSimulation.EffusionParams,
    L::Real;
    model::Symbol = :diffusive,
    batch::Int = 50_000,
    # diffusive options
    max_bounces::Int = 50_000,
    eps_push::Float64 = 1e-12,
    collect_backscatter::Bool = false,
    )
    @assert model === :diffusive || model === :ballistic "model must be :diffusive or :ballistic"

    # Only exit state: [x y z vx vy vz]
    tbl = Matrix{Float64}(undef, N_t, 6)
    filled = 0

    # hoist constants/conversions once
    Lf  = Float64(L)
    eps = Float64(eps_push)

    while filled < N_t
        pos0, vel0 = sample_initial_conditions(batch, rng, p)

        out = if model === :diffusive
            simulate_cavity_diffusive(pos0, vel0, Lf;
                max_bounces=max_bounces,
                eps_push=eps,
                collect_backscatter=collect_backscatter,
                rng=rng,
            )
        else
            simulate_cavity_ballistic(pos0, vel0, Lf)
        end

        ex = out.exited   # BitVector
        ep = out.exit_pos # batch×3
        ev = out.exit_vel # batch×3

        @inbounds for i in 1:batch
            filled >= N_t && break
            ex[i] || continue

            filled += 1
            # exit state only
            tbl[filled, 1] = ep[i, 1]
            tbl[filled, 2] = ep[i, 2]
            tbl[filled, 3] = ep[i, 3]
            tbl[filled, 4] = ev[i, 1]
            tbl[filled, 5] = ev[i, 2]
            tbl[filled, 6] = ev[i, 3]
        end
    end

    return tbl
end


##################################################################################################

## Coil currents
Icoils = [0.00, 1.00 ];
nI = length(Icoils);
const Ns = 2_000_000 ; 
batch_size = 10*Ns
log_step = Int(round(0.1*Ns))

# Original sampling
@info "Number of MonteCarlo particles : $(Ns)\n"
# Monte Carlo generation of particles traversing the filtering SG-slit [x0 y0 z0 v0x v0y v0z]
crossing_slit = TheoreticalSimulation.generate_samples(Ns, effusion_params; v_pdf=:v3, rng = rng_set, multithreaded = false, base_seed = base_seed_set);
plot_statistics(crossing_slit, "Oven (no thickness): cos(θ)" , "histogram_original")

out_diff = collect_transmitted_diffusive_compact( Ns, rng_set, effusion_params, Lcanal; batch = batch_size)
plot_statistics(out_diff, "Oven (thickness): diffusive" , "histogram_diffusive")

out_refl = collect_transmitted_reflective_compact( Ns, rng_set, effusion_params, Lcanal; batch = batch_size)
plot_statistics(out_refl, "Oven (thickness): specular" , "histogram_specular")

out_kill = collect_transmitted_kill_on_collision_compact( Ns, rng_set, effusion_params, Lcanal; batch = batch_size)
plot_statistics(out_kill, "Oven (thickness): kill" , "histogram_kill")

jldopen(joinpath(OUTDIR,"slitcrossing_data.jld2"), "w") do file
    
    file["meta/N"] = Ns
    file["meta/L"] = Lcanal

    file["data/diffusive"] = collect_slitcrossing_diffusive_compact_threads(Ns,
                                        rng_set,
                                        effusion_params,
                                        Lcanal;
                                        batch = batch_size,
                                        log_every = log_step);
    plot_statistics(file["data/diffusive"], "Oven (thickness): diffusive" , "histogram_difussive_slit")
    

    file["data/specular"] = collect_slitcrossing_reflective_compact_threads(Ns,
                                     rng_set,
                                     effusion_params,
                                     Lcanal;
                                     batch = batch_size,
                                     log_every = log_step);

    plot_statistics(file["data/specular"], "Oven (thickness): specular" , "histogram_specular_slit")

    file["data/kill"] = collect_slitcrossing_kill_on_collision_compact_threads(Ns,
                                     rng_set,
                                     effusion_params,
                                     Lcanal;
                                     batch = batch_size,
                                     log_every = log_step);
    plot_statistics(file["data/kill"], "Oven (thickness): kill" , "histogram_kill_slit")
end

r0, v0 = sample_initial_conditions(Ns, rng_set, effusion_params)
# Diffusive (Lambertian)
out_diff = simulate_cavity_diffusive(r0, v0, Lcanal; rng=rng_set)
# Ballistic (geometric): new function
out_ball = simulate_cavity_ballistic(r0, v0, Lcanal)
# 1) transmissions + Clausing
cmp = compare_ballistic_diffusive(out_ball, out_diff)
# 2) angular distributions at y=L (arrays of angles)
ang_diff = exit_angles(out_diff)
ang_ball = exit_angles(out_ball)
# 3) brightness reduction in a chosen forward cone (e.g. 5 mrad)
bright = brightness_reduction(out_ball, out_diff; θmax=asin(effusion_params.sinθmax))
@show cmp;
@show bright;
@show mean(ang_diff.θ), mean(abs.(ang_diff.θx)), mean(abs.(ang_diff.θz));


# @time tbl_ball = collect_transmitted_inout_table(Ns, rng_set, effusion_params, L; model=:ballistic, batch=2*Ns)

# data_exit_ball = tbl_ball[:,7:end];
# plot_statistics(data_exit_ball, "Oven: Ballistic" , "velocity_pdf")


# Nss = 5_000_000
# @time tbl_diff = collect_transmitted_inout_table(Nss, rng_set, effusion_params, L;
#     model=:diffusive, batch=3*Nss, x_furnace=x_furnace, z_furnace=z_furnace);
# data_exit_diff = tbl_diff[:,7:end];
# plot_statistics(data_exit_diff, "Oven: Diffusive" , "velocity_pdf")

# x_at_slit = data_exit_diff[:,1] .+ data_exit_diff[:,4] ./ data_exit_diff[:,5] .* y_FurnaceToSlit
# y_at_slit = data_exit_diff[:,2] .+ y_FurnaceToSlit
# z_at_slit = data_exit_diff[:,3] .+ data_exit_diff[:,6] ./ data_exit_diff[:,5] .* y_FurnaceToSlit

# idx = (abs.(x_at_slit) .< x_slit/2) .& (abs.(z_at_slit) .< z_slit/2 )
# @info "Crossing slit $(sum(idx))/$(Nss)"
# data_crossing = data_exit_diff[idx,:]


# function collect_crossing_table_fast(n_t::Int,
#                                      rng_set,
#                                      effusion_params,
#                                      L;
#                                      model::Symbol = :diffusive,
#                                      batch_size::Int = 5_000_000,
#                                      batch_mult::Int = 3,   # matches your batch=3*Nss idea
#                                      log_every::Int = 1)


#     ncols = 6          # because you want tbl[:, 7:end]
#     out   = Matrix{Float64}(undef, n_t, ncols)

#     # Constants hoisted
#     half_x = 0.5 * x_slit
#     half_z = 0.5 * z_slit
#     yFS    = y_FurnaceToSlit

#     filled = 0
#     iter   = 0

#     while filled < n_t
#         iter += 1

#         tbl = collect_transmitted_inout_table(batch_size, rng_set, effusion_params, L;
#                                               model=model,
#                                               batch=batch_mult*batch_size)

#         nrows = size(tbl, 1)

#         @inbounds for i in 1:nrows
#             # Pull needed columns from tbl[:,7:end] WITHOUT creating that slice
#             # data_exit = tbl[:,7:end] so:
#             #   x0 = tbl[i,7], y0 = tbl[i,8], z0 = tbl[i,9],
#             #   vx = tbl[i,10], vy = tbl[i,11], vz = tbl[i,12]
#             x0 = tbl[i, 7]
#             z0 = tbl[i, 9]
#             vx = tbl[i,10]
#             vy = tbl[i,11]
#             vz = tbl[i,12]

#             t_y = yFS / vy
#             x_at_slit = muladd(vx, t_y, x0)
#             z_at_slit = muladd(vz, t_y, z0)

#             if abs(x_at_slit) < half_x && abs(z_at_slit) < half_z
#                 filled += 1
#                 # Copy entire row tbl[i,7:end] into out[filled,:] without slicing
#                 for j in 1:ncols
#                     out[filled, j] = tbl[i, 6 + j]
#                 end
#                 filled >= n_t && break
#             end
#         end

#         if log_every > 0 && (iter % log_every == 0)
#             @info "Accumulated $filled / $n_t (batch_size=$batch_size, transmission~$(round(100*filled/(batch_size*iter);digits=6))%)"
#         end
#     end

#     return out
# end


# function collect_crossing_table_fastest(n_t::Int,
#                                                rng_set,
#                                                effusion_params,
#                                                L;
#                                                model::Symbol = :diffusive,
#                                                batch_size::Int = 5_000_000,
#                                                batch_mult::Int = 3,
#                                                log_every::Int = 1)

#     out = Matrix{Float64}(undef, n_t, 6)

#     half_x = 0.5 * x_slit
#     half_z = 0.5 * z_slit
#     yFS    = y_FurnaceToSlit

#     filled = 0
#     iter   = 0

#     while filled < n_t
#         iter += 1

#         tbl = collect_transmitted_exit_table(batch_size, rng_set, effusion_params, L;
#                                               model=model,
#                                               batch=batch_mult*batch_size)

#         @inbounds for i in 1:batch_size
#             # vy > 0 always
#             vy = tbl[i, 5]
#             t  = yFS / vy

#             # z-test first (narrower slit -> reject earlier)
#             z0 = tbl[i,3]
#             vz = tbl[i,6]
#             z_at = muladd(vz, t, z0)  # z0 + vz*t

#             if (-half_z < z_at < half_z)
#                 x0 = tbl[i, 1]
#                 vx = tbl[i,4]
#                 x_at = muladd(vx, t, x0)

#                 if (-half_x < x_at < half_x)
#                     filled += 1
#                     copyto!(@view(out[filled, :]), @view(tbl[i, :]))
#                     filled >= n_t && break
#                 end
#             end
#         end

#         if log_every > 0 && (iter % log_every == 0)
#             acc = filled / (iter * batch_size)
#             @info "Accumulated $filled / $n_t (accept≈$(round(100*acc; digits=5))%)"
#         end
#     end

#     return out
# end

# @time collect_crossing_table_fast(1000,
#                                      rng_set,
#                                      effusion_params,
#                                      L;
#                                      model= :diffusive,
#                                      batch_size = 5_000_000,
#                                      batch_mult = 3,
#                                      log_every = 1);

# @time collect_crossing_table_fastest(1000,
#                                      rng_set,
#                                      effusion_params,
#                                      L;
#                                      model= :diffusive,
#                                      batch_size = 5_000_000,
#                                      batch_mult = 3,
#                                      log_every = 1);


# @time dd= collect_slitcrossing_diffusive_compact(1_000,
#                                      rng_set,
#                                      effusion_params,
#                                      L;
#                                      batch = 20_000_000,
#                                      log_every = 1);

# @time out = collect_slitcrossing_diffusive_compact_threads(1_000_000,
#                                      rng_set,
#                                      effusion_params,
#                                      L;
#                                      batch = 50_000_000,
#                                      log_every = 1);

# plot_statistics(out, "Oven: Diffusive" , "velocity_pdf")
# jldopen(joinpath(OUTDIR,"slitcrossing_data.jld2"), "w") do file
#     file["data/initial"] = out
#     file["meta/N"] = size(out,1)
#     file["meta/L"] = L
# end