# Simulation of atom trajectories in the Stern–Gerlach experiment
# Kelvin Titimbo
# California Institute of Technology
# July 2025

#  Plotting Setup
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
# Numerical tools
using LinearAlgebra, DataStructures
using Interpolations, Roots, Dierckx, Loess, Optim
using BSplineKit
using WignerD, LambertW, PolyLog
using StatsBase
using Random, Statistics, NaNStatistics, MLBase, Distributions, StaticArrays
using Alert
# Data manipulation
using OrderedCollections
using DelimitedFiles, CSV, DataFrames, JLD2
# Custom modules
include("./Modules/atoms.jl");
include("./Modules/samplings.jl");
# include("./Modules/MyPolylogarithms.jl");
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Set the working directory to the current location
cd(dirname(@__FILE__)) 
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
# Timestamp start for execution timing
t_start = Dates.now()
# Random seeds
base_seed_set = 145;
# rng_set = MersenneTwister(base_seed_set)
rng_set = TaskLocalRNG()

println("\n\t\tRunning process on:\t $(Dates.format(t_start, "yyyymmddTHHMMSS")) \n")
# Generate a timestamped directory name for output (e.g., "20250718T153245")
directoryname = Dates.format(t_start, "yyyymmddTHHMMSS") ;
# Construct the full directory path (relative to current working directory)
dir_path = "./simulation_data/$(directoryname)" ;
# Create the directory (and any necessary parent folders)
mkpath(dir_path) ;
@info "Created output directory" dir = dir_path

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
atom_info       = AtomicSpecies.atoms(atom);
const R         = atom_info[1];
const μₙ        = atom_info[2];
const γₙ        = atom_info[3];
const Ispin    = atom_info[4];
const Ahfs     = atom_info[6];
const M        = atom_info[7];
const ki = 2.1e-6

# STERN--GERLACH EXPERIMENT
# Image size
const cam_pixelsize = 0.0065 ;  # [mm]
n_bins = 4
exp_pixelsize = n_bins * cam_pixelsize ;   # [mm] for 20243014
# Furnace
const T = 273.15 + 200 ; # Furnace temperature (K)
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
# Sample size: number of atoms arriving to the screen
const Nss = 10000

# Coil currents
Icoils = [0.001,0.002,0.003,0.005,0.007,
            0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.75,0.80
];
nI = length(Icoils);

# Magnetic field gradient interpolation
GradCurrents = [0, 0.095, 0.2, 0.302, 0.405, 0.498, 0.6, 0.7, 0.75, 0.8, 0.902, 1.01];
GradGradient = [0, 25.6, 58.4, 92.9, 132.2, 164.2, 196.3, 226, 240, 253.7, 277.2, 298.6];
GvsI = Interpolations.LinearInterpolation(GradCurrents, GradGradient, extrapolation_bc=Line());
IvsG = Interpolations.LinearInterpolation(GradGradient, GradCurrents, extrapolation_bc=Line());

# Magnetic Field
Bdata = CSV.read("./SG_BvsI.csv",DataFrame; header=["dI","Bz"]);
BvsI = linear_interpolation(Bdata.dI, Bdata.Bz, extrapolation_bc=Line());


#################################################################################
# FUNCTIONS
#################################################################################
function clear_all()
    for name in names(Main, all=true)
        if name ∉ (:Base, :Core, :Main, Symbol("@__dot__"))
            if !isdefined(Main, name) || isconst(Main, name)
                continue  # Skip constants
            end
            @eval Main begin
                global $name = nothing
            end
        end
    end
    GC.gc()
    println("All user-defined variables (except constants) cleared.")
end

function polylog(s,z)
    # return MyPolylogarithms.polylog(s,z)
    return reli2(z)
end

function FreedmanDiaconisBins(data_list::Vector{Float64})
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

    return bins
end

# Quantum Magnetic Moment μF : electron(1/2)-nucleus(3/2)
function μF_effective(Ix,II,F,mF)
    ΔE = 2π*ħ*Ahfs*(II+1/2)
    normalized_B = (γₑ-γₙ)*ħ / ΔE * BvsI(Ix) 
    if F==II+1/2 
        if mF==F
            μF = gₑ/2 * ( 1 + 2*γₙ/γₑ*II)*μB
        elseif mF==-F
            μF = -gₑ/2 * ( 1 + 2*γₙ/γₑ*II)*μB
        else
            μF = gₑ*μB* ( mF*γₙ/γₑ + (1-γₙ/γₑ)/sqrt(1-4*mF/(2*II+1)*normalized_B+(normalized_B)^2) * ( mF/(2*II+1)-1/2*normalized_B ) )
        end
    elseif F==II-1/2
        μF = gₑ*μB* ( mF*γₙ/γₑ - (1-γₙ/γₑ)/sqrt(1-4*mF/(2*II+1)*normalized_B+(normalized_B)^2) * ( mF/(2*II+1)-1/2*normalized_B ) )
    end
    return μF
end

# Atomic beam velocity probability Distribution
p_furnace   = [-x_furnace/2,-z_furnace/2];
p_slit      = [x_slit/2, z_slit/2];
θv_max      = 1.25*atan(norm(p_furnace-p_slit) , y_FurnaceToSlit);
function AtomicBeamVelocity()
    ϕ = 2π*rand(rng_set)
    θ = asin(sin(θv_max)*sqrt(rand(rng_set)))
    v = sqrt(-2*kb*T/M*(1 + lambertw((rand(rng_set)-1)/exp(1),-1)))
    return [ v*sin(θ)*sin(ϕ) , v*cos(θ) , v*sin(θ)*cos(ϕ) ]
end

# CQD Equations of motion
function CQDEqOfMotion(t,Ix,μ,r0::Vector{Float64},v0::Vector{Float64},θe::Float64, θn::Float64, kx::Float64)
    tf1 = y_FurnaceToSlit / v0[2]
    tf2 = (y_FurnaceToSlit + y_SlitToSG ) / v0[2]
    tf3 = (y_FurnaceToSlit + y_SlitToSG + y_SG ) / v0[2]
    tF = (y_FurnaceToSlit + y_SlitToSG + y_SG + y_SGToScreen ) / v0[2]

    cqd_sign = sign(θn-θe) 
    ωL       = abs(γₑ * BvsI(Ix) )
    acc_0    = μ*GvsI(Ix)/M
    kω       = cqd_sign*kx*ωL

    if 0.00 <= t && t <= tf1     # Furnace to Slit
        x = r0[1] + v0[1]*t 
        y = r0[2] + v0[2]*t 
        z = r0[3] + v0[3]*t
        vx , vy , vz = v0[1] , v0[2] , v0[3]
    elseif tf1 < t && t <= tf2    # Slit to SG apparatus
        x = r0[1] + v0[1]*t 
        y = r0[2] + v0[2]*t
        z = r0[3] + v0[3]*t
        vx , vy , vz = v0[1] , v0[2] , v0[3]
    elseif tf2 < t && t <= tf3   # Crossing the SG apparatus
        vx = v0[1]
        vy = v0[2]
        vz = v0[3] + acc_0*(t-tf2) + acc_0/kω * log( cos(θe/2)^2 + exp(-2*kω*(t-tf2))*sin(θe/2)^2 )
        x = r0[1] + v0[1]*t 
        y = r0[2] + v0[2]*t
        z = r0[3] + v0[3]*t + acc_0/2*(t-tf2)^2 + acc_0/kω*log(cos(θe/2)^2)*(t-tf2) + 1/2/(kω)^2 * acc_0 * ( polylog(2,-exp(-2*kω*(t-tf2))*tan(θe/2)^2) - polylog(2,-tan(θe/2)^2) )
    elseif t > tf3 # Travel to the Screen
        x = r0[1] + v0[1]*t
        y = r0[2] + v0[2]*t
        z = r0[3] + v0[3]*t + acc_0/2*( (t-tf2)^2 - (t-tf3)^2) + acc_0/kω*y_SG/v0[2] * ( log(cos(θe/2)^2) + v0[2]/y_SG*log(cos(θe/2)^2+exp(-2*kω*y_SG/v0[2])*sin(θe/2)^2)*(t-tf3) ) + acc_0/2/kω^2*( polylog(2,-exp(-2*kω*y_SG/v0[2])*tan(θe/2)^2) - polylog(2,-tan(θe/2)^2) )
        vx = v0[1]
        vy = v0[2]
        vz = v0[3] + acc_0*y_SG/v0[2] + acc_0/kω*log(cos(θe/2)^2 + exp(-2*kω*y_SG/v0[2])*sin(θe/2)^2)
    end

    return [x,y,z]
end

# CQD equations of motion only along the z-coordinate
@inline function CQDEqOfMotion_z(t,Ix::Float64,μ::Float64,r0::AbstractVector{Float64},v0::AbstractVector{Float64},θe::Float64, θn::Float64, kx::Float64)
    vy = v0[2]
    vz = v0[3]
    z0 = r0[3]
    
    tf2 = (y_FurnaceToSlit + y_SlitToSG ) / vy
    tf3 = (y_FurnaceToSlit + y_SlitToSG + y_SG ) / vy

    cqd_sign = sign(θn-θe) 
    ωL       = abs( γₑ * BvsI(Ix) )
    acc_0    = μ*GvsI(Ix)/M
    kω       = cqd_sign*kx*ωL

    # Precompute angles
    θe_half = θe / 2
    tanθ = tan(θe_half)
    tanθ2 = tanθ^2
    cosθ2 = cos(θe_half)^2
    sinθ2 = sin(θe_half)^2
    log_cos2 = log(cosθ2)
    polylog_0 = polylog(2, -tanθ2)

    if t <= tf2
        return z0 + vz*t
    elseif t <= tf3   # Crossing the SG apparatus
        Δt = t - tf2
        exp_term = exp(-2 * kω * Δt)
        polylog_t = polylog(2, -exp_term * tanθ2)

        return z0 + vz*t + 0.5*acc_0*Δt^2 + acc_0 / kω * log_cos2 * Δt + 0.5 * acc_0 / kω^2 * ( polylog_t - polylog_0 )
    
    else # t > tf3 # Travel to the Screen
        Δt2 = t - tf2
        Δt3 = t - tf3
        τ_SG = y_SG / vy
        exp_SG = exp(-2 * kω * τ_SG)
        polylog_SG = polylog(2, -exp_SG * tanθ2)
        log_term = log(cosθ2 + exp_SG * sinθ2)

        return z0 + vz*t + 0.5*acc_0*( Δt2^2 - Δt3^2 ) + acc_0 / kω * τ_SG * (log_cos2 + log_term * Δt3 / τ_SG) + 0.5 * acc_0 / kω^2 * (polylog_SG - polylog_0)
    end
end

# CQD Screen position
function CQD_Screen_position(Ix,μ,r0::Vector{Float64},v0::Vector{Float64},θe::Float64, θn::Float64, kx::Float64)
    L1 = y_FurnaceToSlit 
    L2 = y_SlitToSG
    Lsg = y_SG
    Ld = y_SGToScreen

    cqd_sign = sign(θn-θe) 
    acc_0 = μ * GvsI(Ix) / M
    ωL = abs(γₑ * BvsI(Ix))
    kω = cqd_sign * kx * ωL

    x = r0[1] + (L1 + L2 + Lsg + Ld) * v0[1] / v0[2]
    y = r0[2] +  L1 + L2 + Lsg + Ld
    z = r0[3] + (L1 + L2 + Lsg + Ld) * v0[3] / v0[2] + acc_0/2/v0[2]^2*((Lsg+Ld)^2-Ld^2) + acc_0/kω*Lsg/v0[2]*( log(cos(θe/2)^2) + Ld/Lsg * log( cos(θe/2)^2 + exp(-2*kω*Lsg/v0[2])*sin(θe/2)^2 ) ) + acc_0/2/kω^2 * ( polylog(2, -exp(-2*kω*Lsg/v0[2])*tan(θe/2)^2) - polylog(2, -tan(θe/2)^2)  )
    return [x,y,z]
end

# Generate samples post-filtering by the slit
function _generate_samples_serial(No::Int, rng0)
    alive = Matrix{Float64}(undef, No, 6)
    iteration_count = 0
    count = 0

    @time while count < No
        iteration_count += 1

        x_initial = x_furnace * (rand(rng0) - 0.5)
        z_initial = z_furnace * (rand(rng0) - 0.5)
        v0_x, v0_y, v0_z = AtomicBeamVelocity()

        x_at_slit = x_initial + y_FurnaceToSlit * v0_x / v0_y
        z_at_slit = z_initial + y_FurnaceToSlit * v0_z / v0_y

        if -x_slit/2 <= x_at_slit <= x_slit/2 && -z_slit/2 <= z_at_slit <= z_slit/2
            count += 1
            alive[count,:] =  [x_initial, 0.0, z_initial, v0_x, v0_y, v0_z]
        end
    end

    println("Total iterations: ", iteration_count)
    return alive
end

function _generate_samples_multithreaded(No::Int, base_seed::Int)
    alive = Matrix{Float64}(undef, No, 6)
    sample_count = Threads.Atomic{Int}(0)
    iteration_count = Threads.Atomic{Int}(0)

    @time Threads.@threads for thread_id in 1:Threads.nthreads()
        rng0 = TaskLocalRNG()
        Random.seed!(rng0, hash((base_seed, thread_id)))
        # rng0 = MersenneTwister(hash((base_seed, thread_id)))   

        while true
            Threads.atomic_add!(iteration_count, 1)

            x_initial = x_furnace * (rand(rng0) - 0.5)
            z_initial = z_furnace * (rand(rng0) - 0.5)
            v0_x, v0_y, v0_z = AtomicBeamVelocity()

            x_at_slit = x_initial + y_FurnaceToSlit * v0_x / v0_y
            z_at_slit = z_initial + y_FurnaceToSlit * v0_z / v0_y

            if -x_slit/2 <= x_at_slit <= x_slit/2 && -z_slit/2 <= z_at_slit <= z_slit/2
                idx = Threads.atomic_add!(sample_count, 1)
                if idx <= No
                    @inbounds alive[idx, :] = [x_initial, 0.0, z_initial, v0_x, v0_y, v0_z]
                else
                    break
                end
            end
        end
    end

    println("Total iterations: ", iteration_count[])
    return alive
end

function generate_samples(No::Int; rng = Random.default_rng(), multithreaded::Bool = false, base_seed::Int = 1234)
    if multithreaded
        return _generate_samples_multithreaded(No, base_seed)
    else
        return _generate_samples_serial(No, rng)
    end
end

# Magnet shape
function z_magnet_edge(x)
    a =2.5e-3;
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

function generate_matched_pairs(No)
    θes_up_list = Float64[]
    θns_up_list = Float64[]
    θes_down_list = Float64[]
    θns_down_list = Float64[]
    
    count_less = 0
    count_greater = 0
    total_trials = 0
    
    @time while count_less < No || count_greater < No
        total_trials += 1
        θe = 2 * asin(sqrt(rand(rng_set)))
        θn = 2 * asin(sqrt(rand(rng_set)))

        if θe < θn && count_less < No
            push!(θes_up_list, θe)
            push!(θns_up_list, θn)
            count_less += 1
        elseif θe > θn && count_greater < No
            push!(θes_down_list, θe)
            push!(θns_down_list, θn)
            count_greater += 1
        end
    end
    
    println("Total angle pairs generated: $total_trials")

    return θes_up_list, θns_up_list, θes_down_list, θns_down_list
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

function find_good_particles(Ix, pairs)
    No = size(pairs, 1) # Number of particles (rows in pairs)
    good_particles = OrderedDict{Int8, Vector{Int}}()

    for (i0_idx,i0) in enumerate(Ix)
        println("Analyzing current I₀ = $(@sprintf("%.3f", i0))A")
        
        # Thread-local buffers to safely collect indices without locks
        thread_buffers = [Vector{Int}() for _ in 1:Threads.nthreads()]

        # Counters for particles hitting Stern-Gerlach cavity and post-SG pipe
        hits_SG_threads = zeros(Int, Threads.nthreads())
        hits_post_threads = zeros(Int, Threads.nthreads())

        # Parallel loop over all particles
        Threads.@threads for j = 1:No
        # for j = 1:No
            try
                tid = Threads.threadid()

                # travel times
                v_y = pairs[j, 5] # Extract particle's propagation speed
                t_in = (y_FurnaceToSlit + y_SlitToSG) / v_y
                t_out = (y_FurnaceToSlit + y_SlitToSG + y_SG) / v_y
                t_screen = (y_FurnaceToSlit + y_SlitToSG + y_SG + y_SGToScreen) / v_y
                t_length = 1000;

                # initial conditions of the particle
                # Extract initial position and velocity vectors (x,y,z components)
                r0 = [pairs[j, 1], pairs[j, 2], pairs[j, 3]]
                v0 = [pairs[j, 4], pairs[j, 5], pairs[j, 6]]
                # Extract angles θe and θn
                θe0 = pairs[j, 7]
                θn0 = pairs[j, 8]

                # --- SG cavity check ---
                # Sample times within SG cavity (t_length)
                t_sweep_sg = range(t_in, t_out, length=t_length)
                # Compute z position at each time sample using CQDEqOfMotion_z vectorized with broadcasting `.`
                z_val = CQDEqOfMotion_z.(t_sweep_sg, Ref(i0), Ref(μₑ), Ref(r0), Ref(v0), Ref(θe0), Ref(θn0), Ref(ki))
                # Compute top and bottom cavity boundary at each sampled time
                # z_top    = z_magnet_profile_time.(t_sweep_sg, Ref(r0), Ref(v0), Ref("top"))
                # z_bottom = z_magnet_profile_time.(t_sweep_sg, Ref(r0), Ref(v0), Ref("bottom"))
                z_top = z_magnet_edge_time.(t_sweep_sg, Ref(r0), Ref(v0))
                z_bottom = z_magnet_trench_time.(t_sweep_sg, Ref(r0), Ref(v0))


                # Logical vector where particle z position lies inside the cavity boundaries
                inside_cavity = (z_bottom .< z_val) .& (z_val .< z_top)
                # If particle ever crosses cavity boundaries, count it as hit and skip post-SG check
                if !all(inside_cavity)
                    hits_SG_threads[tid] += 1
                    continue # Skip post-SG check if already invalid
                end

                # --- post-SG circular pipe check ---
                # Sample times from end of SG to screen (t_length)
                t_sweep_screen = range(t_out, t_screen, length=t_length)
                # x positions move ballistically: x = x0 + vx * t
                xs = r0[1] .+ v0[1].* t_sweep_screen
                # z positions computed via CQDEqOfMotion_z
                zs = CQDEqOfMotion_z.(t_sweep_screen, Ref(i0), Ref(μₑ), Ref(r0), Ref(v0), Ref(θe0), Ref(θn0), Ref(ki))

                R_tube = 35e-3 / 2  # Radius of pipe aperture

                # Check if particle leaves pipe aperture at any time
                outside = any(xs.^2 .+ zs.^2 .> R_tube^2)
                # Count pipe hits and skip if outside aperture
                if outside
                    hits_post_threads[tid] += 1
                    continue  # Rejected by post-SG aperture
                end

                # If passed both checks, keep it
                push!(thread_buffers[tid], j)
            catch err
                @error "Thread $j crashed" exception=err
            end
        end

        # Combine results from all threads safely
        good_idx = reduce(vcat, thread_buffers)
        # Sort the indices for convenience and reproducibility
        sort!(good_idx)
        # Store sorted indices in output dictionary keyed by current index
        good_particles[i0_idx] = good_idx

        # Print diagnostics
        println("\t→ SG hits   = $(sum(hits_SG_threads))")
        println("\t→ Pipe hits = $(sum(hits_post_threads))\n")
    end

    return good_particles
end

function find_good_particles_ix(Ix, pairs)
    No = size(pairs, 1)  # Number of particles
    ncurrents = length(Ix)
    good_particles_per_thread = Vector{Dict{Int8, Vector{Int}}}(undef, Threads.nthreads())
    for i in 1:Threads.nthreads()
        good_particles_per_thread[i] = Dict{Int8, Vector{Int}}()
    end

    Threads.@threads for idx in 1:ncurrents
    # for idx in 1:ncurrents
        tid = Threads.threadid()
        i0 = Ix[idx]
        println("Analyzing current I₀ = $(@sprintf("%.3f", i0))A")

        # Thread-local buffers for this current
        thread_buffer = Vector{Int}()
        hits_SG = 0
        hits_post = 0

        for j = 1:No
            try
                @inbounds begin
                    # travel times
                    v_y = pairs[j, 5]
                    t_in = (y_FurnaceToSlit + y_SlitToSG) / v_y
                    t_out = (y_FurnaceToSlit + y_SlitToSG + y_SG) / v_y
                    t_screen = (y_FurnaceToSlit + y_SlitToSG + y_SG + y_SGToScreen) / v_y
                    t_length = 1000

                    # initial conditions
                    r0 = @view pairs[j, 1:3]
                    v0 = @view pairs[j, 4:6]
                    θe0 = pairs[j, 7]
                    θn0 = pairs[j, 8]
                end

                # SG cavity check
                t_sweep_sg = range(t_in, t_out, length=t_length)
                z_val = CQDEqOfMotion_z.(t_sweep_sg, Ref(i0), Ref(μₑ), Ref(r0), Ref(v0), Ref(θe0), Ref(θn0), Ref(ki))
                z_top = z_magnet_edge_time.(t_sweep_sg, Ref(r0), Ref(v0))
                z_bottom = z_magnet_trench_time.(t_sweep_sg, Ref(r0), Ref(v0))

                inside_cavity = (z_bottom .< z_val) .& (z_val .< z_top)
                if !all(inside_cavity)
                    hits_SG += 1
                    continue
                end

                # Post-SG pipe check
                t_sweep_screen = range(t_out, t_screen, length=t_length)
                xs = r0[1] .+ v0[1] .* t_sweep_screen
                zs = CQDEqOfMotion_z.(t_sweep_screen, Ref(i0), Ref(μₑ), Ref(r0), Ref(v0), Ref(θe0), Ref(θn0), Ref(ki))

                R_tube = 35e-3 / 2
                if any(xs.^2 .+ zs.^2 .> R_tube^2)
                    hits_post += 1
                    continue
                end

                push!(thread_buffer, j)

            catch err
                @error "Thread $tid, particle $j crashed" exception=err
            end
        end

        # Store per-thread result
        sort!(thread_buffer)
        good_particles_per_thread[tid][Int8(idx)] = thread_buffer

        println("\t→ SG hits   = $hits_SG")
        println("\t→ Pipe hits = $hits_post\n")
    end


    # Merge all thread-local dictionaries into a single Dict
    good_particles = Dict{Int8, Vector{Int}}()
    for d in good_particles_per_thread
        for (k, v) in d
            good_particles[k] = v
        end
    end
    return good_particles

end

function find_bad_particles_ix(Ix, pairs)
    No = size(pairs, 1)  # Number of particles
    ncurrents = length(Ix)

    # Indexed by idx, NOT threadid
    bad_particles_per_current = Vector{Vector{Int}}(undef, ncurrents)
    for i in 1:ncurrents
        bad_particles_per_current[i] = Int[]
    end

    Threads.@threads for idx in 1:ncurrents
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
                    t_screen = (y_FurnaceToSlit + y_SlitToSG + y_SG + y_SGToScreen) / v_y
                    t_length = 1000

                    r0 = @view pairs[j, 1:3]
                    v0 = @view pairs[j, 4:6]
                    θe0 = pairs[j, 7]
                    θn0 = pairs[j, 8]
                end

                t_sweep_sg = range(t_in, t_out, length=t_length)
                z_val = CQDEqOfMotion_z.(t_sweep_sg, Ref(i0), Ref(μₑ), Ref(r0), Ref(v0), Ref(θe0), Ref(θn0), Ref(ki))
                z_top = z_magnet_edge_time.(t_sweep_sg, Ref(r0), Ref(v0))
                z_bottom = z_magnet_trench_time.(t_sweep_sg, Ref(r0), Ref(v0))

                inside_cavity = (z_bottom .< z_val) .& (z_val .< z_top)
                if !all(inside_cavity)
                    push!(local_bad_particles, j)
                    hits_SG += 1
                    continue
                end

                t_sweep_screen = range(t_out, t_screen, length=t_length)
                xs = r0[1] .+ v0[1] .* t_sweep_screen
                zs = CQDEqOfMotion_z.(t_sweep_screen, Ref(i0), Ref(μₑ), Ref(r0), Ref(v0), Ref(θe0), Ref(θn0), Ref(ki))
                R_tube = 35e-3 / 2
                if any(xs.^2 .+ zs.^2 .> R_tube^2)
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

function compute_screen_xyz( Ix::Vector, valid_up::OrderedDict, valid_dw::OrderedDict ) 
    screen_up = OrderedDict{Int64, Matrix{Float64}}()
    screen_dw = OrderedDict{Int64, Matrix{Float64}}()

    for i in eachindex(Ix)
        good_up = valid_up[i]
        good_dw = valid_dw[i]

        N_up = size(good_up, 1)
        N_dw = size(good_dw, 1)

        coords_up = Matrix{Float64}(undef, N_up, 3)
        coords_dw = Matrix{Float64}(undef, N_dw, 3)

        Threads.@threads for j = 1:N_up
            r0 = @inbounds good_up[j, 1:3]
            v0 = @inbounds good_up[j, 4:6]
            θe0 = good_up[j, 7]
            θn0 = good_up[j, 8]
            @inbounds coords_up[j, :] = CQD_Screen_position(Ix[i], μₑ, r0, v0, θe0, θn0, ki)
        end

        Threads.@threads for j = 1:N_dw
            r0 = @inbounds good_dw[j, 1:3]
            v0 = @inbounds good_dw[j, 4:6]
            θe0 = good_dw[j, 7]
            θn0 = good_dw[j, 8]
            @inbounds coords_dw[j, :] = CQD_Screen_position(Ix[i], μₑ, r0, v0, θe0, θn0, ki)
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

function plot_velocity_stats(alive::Matrix{Float64}, path_filename::String)
    # Extract velocity components from columns 4:6
    velocities = sqrt.(sum(alive[:, 4:6].^2, dims=2))[:, 1]  # Vector of norms
    theta_vals = acos.(alive[:, 6] ./ velocities)
    phi_vals   = atan.(alive[:, 5], alive[:, 4])

    # Compute means
    mean_v = mean(velocities)
    rms_v  = sqrt(mean(velocities .^ 2))
    mean_theta = mean(theta_vals)
    mean_phi   = mean(phi_vals)

    # Histogram for velocities
    figa = histogram(
        velocities,
        bins = FreedmanDiaconisBins(velocities),
        label = L"$v_0$",
        normalize = :pdf,
        xlabel = L"v_{0} \ (\mathrm{m/s})",
        alpha = 0.70
    )
    vline!([mean_v], 
        label = L"$\langle v_{0} \rangle = %$(round(mean_v, digits=1)) \mathrm{m/s}$",
        line = (:black, :solid, 2),
    )
    vline!([rms_v], 
        label = L"$\sqrt{\langle v_{0}^2 \rangle} = %$(round(rms_v, digits=1)) \mathrm{m/s}$",
        line = (:red, :dash, 3)
    )

    # Histogram for theta (polar angle)
    figb = histogram(
        theta_vals,
        bins = FreedmanDiaconisBins(theta_vals),
        label = L"$\theta_v$",
        normalize = :pdf,
        alpha = 0.70,
        xlabel = L"$\theta_{v}$",
    )
    vline!([mean_theta], 
        label = L"$\langle \theta_{v} \rangle = %$(round(mean_theta/π, digits=3))\pi$",
        line = (:black, :solid, 2)
    )

    # Histogram for phi (azimuthal angle)
    figc = histogram(
        phi_vals,
        bins = FreedmanDiaconisBins(phi_vals),
        label = L"$\phi_v$",
        normalize = :pdf,
        alpha = 0.70,
        xlabel = L"$\phi_{v}$",
    )
    vline!([mean_phi], 
        label = L"$\langle \phi_{v} \rangle = %$(round(mean_phi/π, digits=3))\pi$",
        line = (:black, :solid, 2)
    )

    # 2D Histogram of position (x, z)
    xs = 1e3 .* alive[:, 1]
    zs = 1e6 .* alive[:, 3]
    figd = histogram2d(xs, zs,
        bins = (FreedmanDiaconisBins(xs), FreedmanDiaconisBins(zs)),
        show_empty_bins = true,
        color = :plasma,
        xlabel = L"$x \ (\mathrm{mm})$",
        ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -1.0:0.25:1.0,
        yticks = -50:10:50,
        colorbar_position = :bottom,
    )

    # Histograms for velocity components
    vxs = alive[:, 4]
    vys = alive[:, 5]
    vzs = alive[:, 6]

    fige = histogram(vxs,
        bins = FreedmanDiaconisBins(vxs),
        normalize = :pdf,
        label = L"$v_{0,x}$",
        alpha = 0.65,
        color = :orange,
        xlabel = L"$v_{0,x} \ (\mathrm{m/s})$",
    )

    figf = histogram(vys,
        bins = FreedmanDiaconisBins(vys),
        normalize = :pdf,
        label = L"$v_{0,y}$",
        alpha = 0.65,
        color = :blue,
        xlabel = L"$v_{0,y} \ (\mathrm{m/s})$",
    )

    figg = histogram(vzs,
        bins = FreedmanDiaconisBins(vzs),
        normalize = :pdf,
        label = L"$v_{0,z}$",
        alpha = 0.65,
        color = :red,
        xlabel = L"$v_{0,z} \ (\mathrm{m/s})$",
    )

    # Combine plots
    fig = plot(
        figa, fige, figb, figf, figc, figg, figd,
        layout = @layout([a1 a2; a3 a4; a5 a6; a7]),
        size = (650, 800),
        legendfontsize = 8,
        left_margin = 3mm,
    )

    display(fig)
    savefig(fig, path_filename)

    return fig
end

function plot_SG_geometry(path_filename::AbstractString)
    # Assumes:
    # - z_magnet_edge(x::Real)
    # - z_magnet_trench(x::Real)
    # - x_slit, z_slit (in meters)

    local x_line = 1e-3*collect(range(-10,10,10001));
    fig = plot(
        xlabel=L"$x \ (\mathrm{mm})$",
        xlim = (-8, 8),
        xticks = -8:2:8,
        ylabel=L"$y \ (\mathrm{mm})$",
        ylim = (-3, 7),
        yticks = -3:1:7,
        aspect_ratio = :equal,
        legend = :bottomright,
        title = "Stern–Gerlach Slit Geometry"
    )

    # Convert to mm
    x_fill = 1e3 .* x_line
    y_edge = 1e3 .* z_magnet_edge.(x_line)
    y_top = fill(10.0, length(x_fill))

    plot!(
        fig,
        [x_fill; reverse(x_fill)],
        [y_edge; reverse(y_top)],
        seriestype = :shape,
        label = "Rounded edge",
        color = :grey36,
        line = (:solid, :grey36),
        fillalpha = 0.75
    )

    y_trench = 1e3 .* z_magnet_trench.(x_line)
    y_bottom = fill(-10.0, length(x_fill))

    plot!(
        fig,
        [x_fill; reverse(x_fill)],
        [y_bottom; reverse(y_trench)],
        seriestype = :shape,
        color = :grey60,
        line = (:solid, :grey60),
        label = "Trench",
        fillalpha = 0.75
    )

    # Slit rectangle (assumes x_slit and z_slit are global)
    plot!(
        fig,
        1e3 .* 0.5 .* [-x_slit, -x_slit, x_slit, x_slit, -x_slit],
        1e3 .* 0.5 .* [-z_slit, z_slit, z_slit, -z_slit, -z_slit],
        label = "Slit",
        seriestype = :shape,
        line = (:solid, :red, 1),
        color = :red,
        fillalpha = 0.2
    )

    # Save and show
    display(fig)
    savefig(fig, path_filename)

    return fig
end

function plot_SG_magneticfield(path_filename::AbstractString)
    # Assumes the following are available in scope:
    # - GradCurrents, GradGradient
    # - GvsI(current::AbstractVector)
    # - Bdata.dI, Bdata.Bz
    # - BvsI(current::AbstractVector)

    local icoils = collect(range(1e-6, 1.05, 10000))

    # Panel 1: Gradient data vs current
    fig1a = plot(
        GradCurrents, GradGradient,
        seriestype = :scatter,
        marker = (:circle, :black, 2),
        label = false,
        xlabel = "Coil Current (A)",
        ylabel = "Magnetic field gradient (T/m)",
        yticks = 0:50:350
    )
    plot!(
        fig1a,
        icoils, GvsI(icoils),
        line = (:red, 2),
        label = L"$\partial_{z}B_{z}$"
    )

    # Panel 2: B field vs current
    fig1b = plot(
        Bdata.dI, Bdata.Bz,
        seriestype = :scatter,
        marker = (:circle, :black, 2),
        label = false,
        xlabel = "Coil Current (A)",
        ylabel = "Magnetic field (T)",
        yticks = 0:0.1:1.0
    )
    plot!(
        fig1b,
        icoils, BvsI(icoils),
        line = (:orange, 2),
        label = L"$B_{z}$"
    )

    # Panel 3: B vs gradient
    fig1c = plot(
        GvsI(icoils), BvsI(icoils),
        label = false,
        line = (:blue, 2),
        xlabel = "Magnetic field gradient (T/m)",
        ylabel = "Magnetic field (T)",
        ylims = (0, 0.8),
        xticks = 0:50:350,
        yticks = 0:0.1:1.0
    )

    # Compose layout
    fig1 = plot(
        fig1a, fig1b, fig1c,
        layout = @layout([a1; a2; a3]),
        size = (400, 700),
        plot_title = "Magnetic field in the Stern--Gerlach apparatus",
        plot_titlefontsize = 10,
        guidefont = font(8, "Computer Modern"),
        link = :none,
        left_margin = 5mm,
        bottom_margin = 0mm,
        right_margin = 0mm
    )

    # Save and show
    display(fig1)
    savefig(fig1, path_filename)

    return fig1
end

function plot_ueff(II,path_filename::AbstractString)
    F_up = II + 0.5
    mf_up = collect(F_up:-1:-F_up)
    F_down = II - 0.5
    mf_down = collect(-F_down:1:F_down)
    dimF = Int(4*II + 2)
        
    # Set color palette
    colorsF = palette(:phase, dimF)
    current_range = collect(0.00009:0.00002:1);

    # Initialize plot
    fig = plot(
        xlabel = L"Current ($\mathrm{A}$)",
        ylabel = L"$\mu_{F}/\mu_{B}$",
        legend = :right,
        background_color_legend = RGBA(0.85, 0.85, 0.85, 0.1),
        size = (800, 600),
    )

    # Define lines to plot: (F, mF, color index, style)
    lines_to_plot = vcat(
        [(F_up, mf, :solid) for mf in mf_up[1:end-1]],
        [(F_up, mf_up[end],:dash)],
        [(F_down, mf, :dash) for mf in mf_down],
    )

    # Plot all curves
    for ((f,mf,lstyle),color) in zip(lines_to_plot,colorsF)
        μ_vals = μF_effective.(current_range, II, f, mf) ./ μB
        label = L"$F=%$(f)$, $m_{F}=%$(mf)$"
        plot!(current_range, μ_vals, label=label, line=(color,lstyle, 2))
    end
        
    # Magnetic crossing point
    f(x) = BvsI(x) - 2π*ħ*Ahfs*(Ispin+1/2)/(2ħ)/(γₙ - γₑ)
    bcrossing = find_zero(f, (0.001, 0.02))

    # Annotated vertical line
    label_text = L"$I_{0} = %$(round(bcrossing, digits=5))\,\mathrm{A}$
     $\partial_{z}B_{z} = %$(round(GvsI(bcrossing), digits=2))\,\mathrm{T/m}$
     $B_{z} = %$(round(1e3 * BvsI(bcrossing), digits=3))\,\mathrm{mT}$"
    vline!([bcrossing], line=(:black, :dot, 2), label=label_text,xaxis = :log10,)
    
    display(fig)
    savefig(fig, path_filename)

    return fig
end

plot_SG_geometry(joinpath(dir_path, "slit.png"))
plot_SG_magneticfield(joinpath(dir_path, "SG_magneticfield.png"))
plot_ueff(Ispin,joinpath(dir_path, "mu_effective.png"))

# Monte Carlo generation
alive_slit = generate_samples(Nss; rng = rng_set, multithreaded = true, base_seed = base_seed_set);
plot_velocity_stats(alive_slit, joinpath(dir_path, "vel_stats.png"))

θesUP, θnsUP, θesDOWN, θnsDOWN = generate_matched_pairs(Nss);
pairs_UP = build_init_cond(alive_slit, θesUP, θnsUP)
pairs_DOWN = build_init_cond(alive_slit, θesDOWN, θnsDOWN)
# Optionally clear memory
θesUP = θnsUP = θesDOWN = θnsDOWN = alive_slit = nothing
GC.gc()

@time bad_particles_up = find_bad_particles_ix(Icoils, pairs_UP)
@time bad_particles_dw = find_bad_particles_ix(Icoils, pairs_DOWN)

bad_particles_up = OrderedDict(sort(collect(bad_particles_up); by=first))
bad_particles_dw = OrderedDict(sort(collect(bad_particles_dw); by=first))

println("Particles with final θₑ=0")
for (i0, indices) in bad_particles_up
    println("Current $(@sprintf("%.3f", Icoils[i0]))A \t→   Good particles: ", Nss-length(indices))
end
println("Particles with final θₑ=π")
for (i0, indices) in bad_particles_dw
    println("Current $(@sprintf("%.3f", Icoils[i0]))A \t→   Good particles: ", Nss-length(indices))
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

valid_up   = get_valid_particles_per_current(pairs_UP,   bad_particles_up)
valid_dw = get_valid_particles_per_current(pairs_DOWN, bad_particles_dw)

min_ups = minimum(size(valid_up[v],1) for v in eachindex(Icoils))
min_dws = minimum(size(valid_dw[v],1) for v in eachindex(Icoils))

@save joinpath(dir_path, "data_up.jld2") valid_up Icoils
@save joinpath(dir_path, "data_dw.jld2") valid_dw Icoils


# data recovery
data_path = "./simulation_data/20250807T163648/"
data_u = JLD2.jldopen(joinpath(data_path, "data_up.jld2"), "r") do file
    return Dict(k => read(file, k) for k in keys(file))
end
data_d = JLD2.jldopen(joinpath(data_path, "data_dw.jld2"), "r") do file
    return Dict(k => read(file, k) for k in keys(file))
end

valid_up = data_u["valid_up"]
valid_dw = data_d["valid_dw"]

valid_up[1]

idxi0 = rand(1:nI)
fig4a = FD_histograms(valid_up[idxi0][:,7],L"\theta_{e}",:dodgerblue);
fig4b = FD_histograms(valid_up[idxi0][:,8],L"\theta_{n}",:red);
fig4c = FD_histograms(valid_dw[idxi0][:,7],L"\theta_{e}",:dodgerblue);
fig4d = FD_histograms(valid_dw[idxi0][:,8],L"\theta_{n}",:red);
fig4= plot(fig4a,fig4b,fig4c,fig4d,
    layout = @layout([a1 a2 ; a3 a4]),
    size=(600,600),
    plot_title="Initial polar angles",
    # plot_titlefontcolor=:black,
    plot_titlefontsize=10,
    guidefont=font(8,"Computer Modern"),
    # tickfont=font(8, "Computer Modern"),
    link=:xy,
    # bottom_margin=-8mm, left_margin=-4mm, right_margin=-1mm
    left_margin=5mm,bottom_margin=0mm,right_margin=0mm,
);
plot!(fig4[1],xticks=(xticks(fig4[1])[1], []),xlabel="",bottom_margin=-5mm);
plot!(fig4[2],xticks=(xticks(fig4[2])[1], []), yticks=(yticks(fig4[2])[1], []), xlabel="",bottom_margin=-5mm, left_margin=-5mm);
plot!(fig4[4],yticks=(yticks(fig4[4])[1], fill("", length(yticks(fig4[4])[1]))), ylabel="",left_margin=-5mm);
display(fig4)
savefig(fig4, joinpath(dir_path, "polar_stats.png"))

valid_up
valid_dw


@time screen_up, screen_dw = compute_screen_xyz(Icoils, valid_up, valid_dw);


screen_up[1]

screen_coord = zeros(Nss,3, length(Icoils));

for j=1:length(Icoils)
    @time @threads for i=1:Nss
        screen_coord[i,:,j] = CQD_Screen_position(Icoils[j],μₑ,pairs_UP[i,1:3],pairs_UP[i,4:6],pairs_UP[i,7], pairs_UP[i,8],ki)
    end
end


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





using 
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