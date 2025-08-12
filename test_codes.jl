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
const Nss = 5000

# Magnetic field gradient interpolation
GradCurrents = [0, 0.095, 0.2, 0.302, 0.405, 0.498, 0.6, 0.7, 0.75, 0.8, 0.902, 1.01];
GradGradient = [0, 25.6, 58.4, 92.9, 132.2, 164.2, 196.3, 226, 240, 253.7, 277.2, 298.6];
GvsI = Interpolations.LinearInterpolation(GradCurrents, GradGradient, extrapolation_bc=Line());
IvsG = Interpolations.LinearInterpolation(GradGradient, GradCurrents, extrapolation_bc=Line());

# Magnetic Field
Bdata = CSV.read("./SG_BvsI.csv",DataFrame; header=["dI","Bz"]);
BvsI = linear_interpolation(Bdata.dI, Bdata.Bz, extrapolation_bc=Line());

function find_bad_particles_ix(Ix, pairs)
    No = size(pairs, 1)  # Number of particles
    ncurrents = length(Ix)
    bad_particles_per_thread = Vector{Dict{Int8, Vector{Int}}}(undef, Threads.nthreads())
    for i in 1:Threads.nthreads()
        bad_particles_per_thread[i] = Dict{Int8, Vector{Int}}()
    end

    Threads.@threads for idx in 1:ncurrents
    # for idx in 1:ncurrents
        tid = Threads.threadid()
        i0 = Ix[idx]
        println("Analyzing current I₀ = $(@sprintf("%.3f", i0))A")

        # Thread-local buffer for particles that collided
        bad_particles = Vector{Int}()
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
                    push!(bad_particles, j)
                    hits_SG += 1
                    continue
                end

                # Post-SG pipe check
                t_sweep_screen = range(t_out, t_screen, length=t_length)
                xs = r0[1] .+ v0[1] .* t_sweep_screen
                zs = CQDEqOfMotion_z.(t_sweep_screen, Ref(i0), Ref(μₑ), Ref(r0), Ref(v0), Ref(θe0), Ref(θn0), Ref(ki))

                R_tube = 35e-3 / 2
                if any(xs.^2 .+ zs.^2 .> R_tube^2)
                    push!(bad_particles, j)
                    hits_post += 1
                    continue
                end

            catch err
               @error "Thread $tid, particle $j crashed" exception=err
            end
        end

        # Save results from this thread
        sort!(bad_particles)
        bad_particles_per_thread[tid][Int8(idx)] = bad_particles

        println("\t→ SG hits   = $hits_SG")
        println("\t→ Pipe hits = $hits_post\n")
    end

    # Merge thread-local results into one dictionary
    bad_particles = Dict{Int8, Vector{Int}}()
    for d in bad_particles_per_thread
        for (k, v) in d
            if haskey(bad_particles, k)
                append!(bad_particles[k], v)
            else
                bad_particles[k] = copy(v)  # make sure to copy to avoid mutation issues
            end
        end
    end
    # Optional: sort & deduplicate entries (recommended!)
    for (k, v) in bad_particles
        sort!(v)
        unique!(v)
    end

    return bad_particles
end