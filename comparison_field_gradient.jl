# Comparison
# Kelvin Titimbo
# California Institute of Technology
# MAy 2026

#  Plotting Setup
# ENV["GKS_WSTYPE"] = "100"
# using Plots; gr()
# Plots.default(
#     show=true, dpi=800, fontfamily="Computer Modern", 
#     grid=true, minorgrid=true, framestyle=:box, widen=true,
# )
# using Plots.PlotMeasures
using CairoMakie
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
cd(@__DIR__);
const BASE_PATH = raw"F:\SternGerlachExperiments"
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
include("./Modules/JLD2_MyTools.jl")
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
const y_SGToAperture    = 42.0e-3 ;
# Propagation distances
const y_FurnaceToSlit = 224.0e-3 ;
const y_SlitToSG      = 44.0e-3 ;
const y_SG            = 7.0e-2 ;
const y_SGToScreen    = 32.0e-2 ;
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
calibration = TheoreticalSimulation.build_calibration(ICOILS; degree=3, span =0.12);

# Sample size: number of atoms arriving to the screen
const Nss = 2_000 ; 
@info "Number of MonteCarlo particles : $(Nss)\n"

# ============================================================================
# COQUANTUM DYNAMICS TRAJECTORIES
# ============================================================================
    # Monte Carlo generation of particles traersing the filtering slit [x0 y0 z0 v0x v0y v0z]
crossing_slit = generate_samples(Nss, effusion_params; v_pdf=:v3, rng = rng_set, multithreaded = false, base_seed = base_seed_set);
if SAVE_FIG
    plot_μeff(K39_params,"mm_effective")
    plot_SG_geometry("SG_geometry")
    plot_velocity_stats(crossing_slit, "Initial data" , "velocity_pdf")
end

##################################################################################################
#   COQUANTUM DYNAMICS
##################################################################################################
ki_ref = 2.30e-6;

# Monte Carlo generation of particles traversing the filtering slit and assigning polar angles
data_UP, data_DOWN = generate_CQDinitial_conditions(Nss, crossing_slit, rng_set; mode=:balanced);
data_UP_SG = TheoreticalSimulation.propagate_to_SG_entrance(data_UP)[:,1:3];
data_DOWN_SG = TheoreticalSimulation.propagate_to_SG_entrance(data_DOWN)[:,1:3];


# ============================================================================
# (1) CQD TRAJECTORY INTEGRATION  (reference kᵢ)
# ============================================================================

# --- Magnetic moment-up branch ---
@time CQD_up_particles_flag         = TheoreticalSimulation.CQD_flag_travelling_particles(ICOILS, data_UP, ki_ref, K39_params; y_length=5001,verbose=true);
CQD_up_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles(ICOILS, ki_ref, data_UP, CQD_up_particles_flag, K39_params);     # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]
TheoreticalSimulation.CQD_travelling_particles_summary(ICOILS,CQD_up_particles_trajectories, :up);
CQD_up_particles_at_screen    = TheoreticalSimulation.CQD_select_flagged(CQD_up_particles_trajectories,:screen );
CQD_up_particles_flag = CQD_up_particles_trajectories = nothing;
# --- Magnetic moment-down branch ---
@time CQD_dw_particles_flag         = TheoreticalSimulation.CQD_flag_travelling_particles(ICOILS, data_DOWN, ki_ref, K39_params; y_length=5001,verbose=true);
CQD_dw_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles(ICOILS, ki_ref, data_DOWN, CQD_dw_particles_flag, K39_params);   # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]
TheoreticalSimulation.CQD_travelling_particles_summary(ICOILS,CQD_dw_particles_trajectories, :down);
CQD_dw_particles_at_screen    = TheoreticalSimulation.CQD_select_flagged(CQD_dw_particles_trajectories,:screen );
CQD_dw_particles_flag = CQD_dw_particles_trajectories = nothing;

GC.gc()

# ============================================================================
# (2) CQD TRAJECTORY INTEGRATION  (reference kᵢ)
# ============================================================================

# --- Magnetic moment-up branch ---
@time CQD_up_particles_flag = TheoreticalSimulation.CQD_flag_travelling_particles_twowires(ICOILS, data_UP, data_UP_SG, ki_ref, K39_params, calibration; y_length=5001, verbose=true);
CQD_up_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles_twowires(ICOILS, ki_ref, data_UP, data_UP_SG, CQD_up_particles_flag, K39_params, calibration);   # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]
TheoreticalSimulation.CQD_travelling_particles_summary(ICOILS, CQD_up_particles_trajectories,"UP")
CQD_ΔG_up_particles_at_screen  = TheoreticalSimulation.CQD_select_flagged(CQD_up_particles_trajectories,:screen );
CQD_up_particles_flag = CQD_up_particles_trajectories = nothing;

# --- Magnetic moment-down branch ---
@time CQD_dw_particles_flag = TheoreticalSimulation.CQD_flag_travelling_particles_twowires(ICOILS, data_DOWN, data_DOWN_SG, ki_ref, K39_params, calibration; y_length=5001, verbose=true);
CQD_dw_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles_twowires(ICOILS, ki_ref, data_DOWN, data_DOWN_SG, CQD_dw_particles_flag, K39_params, calibration);   # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]
TheoreticalSimulation.CQD_travelling_particles_summary(ICOILS, CQD_dw_particles_trajectories,"DOWN")
CQD_ΔG_dw_particles_at_screen  = TheoreticalSimulation.CQD_select_flagged(CQD_dw_particles_trajectories,:screen );
CQD_dw_particles_flag = CQD_dw_particles_trajectories = nothing;

GC.gc()

# ============================================================================
"""
    plot_screen_comparison(dict1, dict2, idx; kwargs...) → Figure

Compare the final (x, z) positions of particles arriving at the screen plane
for two force configurations stored in OrderedDict{Int64, Matrix{Float64}}.

Each matrix has particles as rows. Columns 1:8 are used as particle identity
keys (e.g. initial conditions), and columns 9:10 are the (x, z) screen
coordinates that get plotted.

# Arguments
- `dict1`  : first  force dict, e.g. CQD_up_particles_at_screen
- `dict2`  : second force dict, e.g. CQD_ΔG_up_particles_at_screen
- `idx`    : key to look up in both dicts (e.g. 45)

# Keyword arguments
- `mode`          : `:matched` (default) keeps only particles present in both
                    datasets; `:all` plots every particle from each dataset.
- `force_labels`  : display names for the two force cases.
- `force_colors`  : Makie-compatible colours for each case.
- `key_cols`      : column range used to identify matching particles (default 1:8).
- `data_cols`     : column range holding the (x, z) screen coordinates (default 9:10).
- `n_bins`        : number of bins for the marginal histograms (default 80).
- `alpha_scatter` : opacity of the scatter points (default 0.25).
- `alpha_hull`    : opacity of the convex-hull fill (default 0.12).

# Returns
A CairoMakie `Figure` which can be displayed or saved with `save(path, fig)`.

# Example
    fig = plot_screen_comparison(
        CQD_up_particles_at_screen,
        CQD_ΔG_up_particles_at_screen,
        45;
        mode         = :matched,
        force_labels = ["Uniform", "xz dependent"],
    )
    save("screen_45.pdf", fig)
"""
function plot_screen_comparison(
        dict1, dict2, idx;
        mode          = :matched,
        force_labels  = ["Force 1", "Force 2"],
        force_colors  = [:steelblue, :tomato],
        key_cols      = 1:8,
        data_cols     = 9:10,
        n_bins        = 240,
        alpha_scatter = 0.55,
        alpha_hull    = 0.12,
)

    # ── Helper: cross2d ───────────────────────────────────────────────────────
    # Computes the 2-D cross product of vectors (o→a) and (o→b).
    # Used by convex_hull to determine the turn direction between three points:
    #   > 0  →  left turn  (keep the point)
    #   ≤ 0  →  right turn or collinear (discard the point)
    function cross2d(o, a, b)
        (a[1]-o[1])*(b[2]-o[2]) - (a[2]-o[2])*(b[1]-o[1])
    end

    # ── Helper: convex_hull ───────────────────────────────────────────────────
    # Returns the convex hull of a point cloud (x, z) using the
    # Andrew's monotone chain algorithm (O(n log n)).
    # Output is a 2×K matrix where row 1 = x coords, row 2 = z coords
    # of the hull vertices in counter-clockwise order.
    # Falls back to returning all points if fewer than 3 are given.
    function convex_hull(x, z)
        pts = sort(collect(zip(x, z)))   # sort lexicographically by (x, z)
        n   = length(pts)
        n < 3 && return hcat(x, z)'

        # build lower hull (left to right)
        lower = Tuple{Float64,Float64}[]
        for p in pts
            while length(lower) >= 2 && cross2d(lower[end-1], lower[end], p) <= 0
                pop!(lower)
            end
            push!(lower, p)
        end

        # build upper hull (right to left)
        upper = Tuple{Float64,Float64}[]
        for p in reverse(pts)
            while length(upper) >= 2 && cross2d(upper[end-1], upper[end], p) <= 0
                pop!(upper)
            end
            push!(upper, p)
        end

        # remove last point of each half (duplicate of first point of the other)
        pop!(lower); pop!(upper)
        hull = vcat(lower, upper)
        return hcat(first.(hull), last.(hull))'   # 2×K matrix
    end

    # ── Helper: prepare_data ──────────────────────────────────────────────────
    # Extracts and filters the (x, z) data columns from two raw matrices.
    #
    # :matched mode
    #   Builds a set of row-identity keys from key_cols for each matrix,
    #   computes the intersection, filters both matrices to common rows only,
    #   then sorts both by key so that row i in A and row i in B always
    #   correspond to the same particle. This is the correct mode when you
    #   want per-particle differences (ΔX, ΔZ) between the two force cases.
    #
    # :all mode
    #   Simply slices out data_cols from each matrix with no filtering.
    #   Use this when you want to compare the overall beam distributions
    #   regardless of which individual particles survived in each case.
    #
    # Returns: (A, B, n_lost_A, n_lost_B)
    #   A, B        : filtered/sorted data matrices (n_common × length(data_cols))
    #   n_lost_A/B  : number of rows dropped (0 in :all mode)
    function prepare_data(m1, m2)
        if mode == :matched
            n1 = size(m1, 1)
            n2 = size(m2, 1)

            # build row-identity keys from key_cols
            keys1  = Set(Tuple(m1[i, key_cols]) for i in 1:n1)
            keys2  = Set(Tuple(m2[i, key_cols]) for i in 1:n2)
            common = intersect(keys1, keys2)

            # boolean masks selecting only the common rows
            mask1 = [Tuple(m1[i, key_cols]) in common for i in 1:n1]
            mask2 = [Tuple(m2[i, key_cols]) in common for i in 1:n2]

            f1 = m1[mask1, :]
            f2 = m2[mask2, :]

            # sort both filtered matrices by key so rows are aligned
            sortby(M) = sortperm([Tuple(M[i, key_cols]) for i in 1:size(M,1)])
            A = f1[sortby(f1), data_cols]
            B = f2[sortby(f2), data_cols]

            # sanity check: key columns must be identical after sorting
            @assert all(
                abs.(f1[sortby(f1), key_cols] .- f2[sortby(f2), key_cols]) .< 1e-10
            ) "Keys don't align after sorting — check key_cols!"

            n_lost_A = n1 - size(A, 1)
            n_lost_B = n2 - size(B, 1)

            println("── Matched mode ──────────────────────────────")
            println("  force1: $n1 → $(size(A,1)) rows  ($n_lost_A lost)")
            println("  force2: $n2 → $(size(B,1)) rows  ($n_lost_B lost)")
            println("  common particles: $(length(common))")

            return A, B, n_lost_A, n_lost_B

        elseif mode == :all
            A = m1[:, data_cols]
            B = m2[:, data_cols]

            println("── All mode ──────────────────────────────────")
            println("  force1: $(size(A,1)) rows (unfiltered)")
            println("  force2: $(size(B,1)) rows (unfiltered)")

            return A, B, 0, 0

        else
            error("mode must be :matched or :all, got :$mode")
        end
    end

    # ── Section 1: Data extraction ────────────────────────────────────────────
    # Pull the two matrices for this idx and run the selected preparation mode.
    A, B, n_lost_A, n_lost_B = prepare_data(dict1[idx], dict2[idx])

    raw_data = [A, B]
    n_lost   = [n_lost_A, n_lost_B]

    # ── Section 2: Beam statistics ────────────────────────────────────────────
    # Compute centroid, standard deviations, and convex hull for each dataset.
    # Coordinates are converted to mm (×1e3) here and kept consistent
    # throughout the rest of the function.
    # Results are stored as named tuples to avoid struct redefinition issues
    # when re-running the function interactively.
    beam_stats = map(eachindex(raw_data)) do i
        x = 1e3 * raw_data[i][:, 1]   # convert to mm
        z = 1e3 * raw_data[i][:, 2]
        (
            label  = force_labels[i],
            n      = size(raw_data[i], 1),   # number of particles plotted
            n_lost = n_lost[i],               # particles dropped (0 if mode=:all)
            x̄      = mean(x),
            z̄      = mean(z),
            σx     = std(x),
            σz     = std(z),
            hull   = convex_hull(x, z),
        )
    end

    # ── Section 3: Figure layout ──────────────────────────────────────────────
    # Three-panel layout:
    #   [ax_main | ax_mz ]   ← scatter + hull + centroid  |  z marginal (horizontal)
    #   [ax_mx   |  —    ]   ← x marginal                 |  (empty corner)
    # The marginal panels are shrunk to ~22% of the figure to keep the main
    # scatter dominant.
    fig = Figure(size = (900, 760), backgroundcolor = :white)

    ax_main = Axis(fig[1, 1],
        xlabel     = "x  (mm)",
        ylabel     = "z  (mm)",
        title      = "Final particle positions on screen (xz plane)  [I=$(Int(1000*ICOILS[idx]))mA]",
        xgridcolor = (:black, 0.06),
        ygridcolor = (:black, 0.06),
        xticks     = LinearTicks(10),
        yticks     = LinearTicks(10),
    )

    ax_mx = Axis(fig[2, 1],
        xlabel       = "x  (mm)",
        ylabel       = "counts",
        title        = "x marginal",
        xgridvisible = false,
        xticks       = LinearTicks(10),
    )

    ax_mz = Axis(fig[1, 2],
        xlabel        = "counts",
        ylabel        = "z  (mm)",
        title         = "z marginal",
        yaxisposition = :right,
        ygridvisible  = false,
        yticks        = LinearTicks(10),
    )

    # shrink marginal panels relative to the main scatter
    rowsize!(fig.layout, 2, Relative(0.22))
    colsize!(fig.layout, 2, Relative(0.22))

    # ── Section 4: Plotting ───────────────────────────────────────────────────
    # For each force case:
    #   1. scatter of (x, z) positions with transparency
    #   2. convex hull polygon (dashed outline + light fill) showing beam extent
    #   3. centroid marker (cross) at (x̄, z̄)
    #   4. marginal histograms along x (bottom panel) and z (right panel)
    for i in eachindex(raw_data)
        c  = force_colors[i]
        s  = beam_stats[i]
        xs = 1e3 * raw_data[i][:, 1]   # mm
        zs = 1e3 * raw_data[i][:, 2]

        # scatter: individual particle final positions
        CairoMakie.scatter!(ax_main, xs, zs;
            color      = (c, alpha_scatter),
            markersize = 4,
        )

        # convex hull: close the polygon by repeating the first vertex
        hx = vcat(s.hull[1,:], s.hull[1,1])
        hz = vcat(s.hull[2,:], s.hull[2,1])
        poly!(ax_main, Point2f.(hx, hz);
            color       = (c, alpha_hull),
            strokecolor = (c, 0.7),
            strokewidth = 1.2,
            linestyle   = :dash,
        )

        # centroid cross marker
        CairoMakie.scatter!(ax_main, [s.x̄], [s.z̄];
            color       = c,
            marker      = :cross,
            markersize  = 14,
            strokewidth = 2,
        )

        # x marginal: project all particles onto the x axis
        hx_hist = StatsBase.fit(Histogram, xs; nbins = n_bins)
        stairs!(ax_mx, hx_hist.edges[1], vcat(hx_hist.weights, 0);
            color = (c, 0.5), step = :post)

        # z marginal: project onto z axis, plotted horizontally (counts on x)
        hz_hist = StatsBase.fit(Histogram, zs; nbins = n_bins)
        stairs!(ax_mz, vcat(hz_hist.weights, 0), hz_hist.edges[1];
            color = (c, 0.5), step = :post)
    end

    # ── Section 5: Legend ─────────────────────────────────────────────────────
    # Manual legend using MarkerElement so the marker size is independent of
    # the scatter markersize=4 used in the plot (which would make legend
    # symbols tiny if using axislegend directly).
    # Label includes particle count and, in :matched mode, the number lost.
    legend_labels = [
        mode == :matched ?
            "$(beam_stats[i].label)  [n=$(beam_stats[i].n), lost=$(beam_stats[i].n_lost)]" :
            "$(beam_stats[i].label)  [n=$(beam_stats[i].n)]"
        for i in eachindex(beam_stats)
    ]

    legend_elements = [
        MarkerElement(color = (force_colors[i], 0.8), marker = :circle, markersize = 16)
        for i in eachindex(force_labels)
    ]

    Legend(fig[1, 1], legend_elements, legend_labels;
        tellwidth    = false,
        tellheight   = false,
        halign       = :left,
        valign       = :top,
        fontsize     = 13,
        patchsize    = (20, 20),
        framevisible = false,
    )

    # ── Section 6: Stats annotation ───────────────────────────────────────────
    # Monospaced bold text box inside ax_main showing centroid and spread
    # for each force case. Printf formatting keeps columns aligned.
    stats_str = join([
        @sprintf("%-10s  <x>=%+6.1f  <z>=%+6.1f  σx=%5.1f  σz=%5.1f",
            s.label, s.x̄, s.z̄, s.σx, s.σz)
        for s in beam_stats
    ], "\n")

    text!(ax_main, -4.5, 0.25;
        text     = stats_str,
        fontsize = 14,
        font     = "Courier New Bold",
        color    = :black,
    )

    return fig
end
# ============================================================================

for idx=1:nI
    display(
        plot_screen_comparison(
            CQD_up_particles_at_screen, CQD_ΔG_up_particles_at_screen, idx;
            mode          = :all,
            force_labels  = ["Uniform", "G(x,z)"]
        )
    )
end

#_____________________________________________________________________________________________________________
σw = 0.150
ki = 2.0 # ×10^-6
nz = 2
λ0 = 0.01
#_____________________________________________________________________________________________________________
# T = 200C
CQD_UP_T200_manual = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T200_8M","cqd_8M_up_profiles.jld2")
CQD_DW_T200_manual = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T200_8M","cqd_8M_dw_profiles.jld2")
CQD_UP_T200_manual_data = jldopen(CQD_UP_T200_manual,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:up,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end
CQD_DW_T200_manual_data = jldopen(CQD_DW_T200_manual,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:dw,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end

CQD_UP_T200_ΔG = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T200_6M_constG","cqd_6M_up_profiles.jld2")
CQD_DW_T200_ΔG = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T200_6M_constG","cqd_6M_dw_profiles.jld2")
CQD_UP_T200_ΔG_data = jldopen(CQD_UP_T200_ΔG,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:up,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end
CQD_DW_T200_ΔG_data = jldopen(CQD_DW_T200_ΔG,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:dw,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end

# T = 205C
CQD_UP_T205_manual = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T205_7M","cqd_7M_up_profiles.jld2")
CQD_DW_T205_manual = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T205_7M","cqd_7M_dw_profiles.jld2")
CQD_UP_T205_manual_data = jldopen(CQD_UP_T205_manual,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:up,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end
CQD_DW_T205_manual_data = jldopen(CQD_DW_T205_manual,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:dw,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end

CQD_UP_T205_ΔG = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T205_6M_constG","cqd_6M_up_profiles.jld2")
CQD_DW_T205_ΔG = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T205_6M_constG","cqd_6M_dw_profiles.jld2")
CQD_UP_T205_ΔG_data = jldopen(CQD_UP_T205_ΔG,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:up,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end
CQD_DW_T205_ΔG_data = jldopen(CQD_DW_T205_ΔG,"r") do file
    data = file[JLD2_MyTools.make_keypath_cqd(:dw,ki,nz,σw,λ0)]
    nI = length(keys(data))
    currents = [data[x][:Icoil] for x=1:nI]

    return (; 
            Ic      = [data[x][:Icoil] for x=1:nI],
            z_max   = [data[x][:z_max_smooth_spline_mm] for x=1:nI],
            z_profiles = [data[x][:z_profile] for x=1:nI],
    )
end


#_____________________________________________________________________________________________________________
function plot_CQD_profiles(
    up_manual_data, up_ΔG_data,
    dw_manual_data, dw_ΔG_data,
    ICOILS, i0_idx;
    fig_size = (600, 700),
    title_temp = 200,
)
    fig = Figure(size = fig_size)

    axis_common = (
        xminorticksvisible = true,
        xminorticks = IntervalsBetween(5),
        xminorgridvisible = true,
        yminorgridvisible = false,
        xminorgridcolor = (:gray, 0.3),
        yminorgridcolor = (:gray, 0.3),
        xminorgridwidth = 0.5,
        yminorgridwidth = 0.5,
        ylabel = "counts (au)",
    )

    ax1 = Axis(fig[1, 1];
        axis_common...,
        xticklabelsvisible = false,
        bottomspinevisible = true,
        xticksvisible = true,
    )

    ax2 = Axis(fig[2, 1];
        axis_common...,
        xlabel = L"$z \ (\mathrm{mm})$",
    )

    linkxaxes!(ax1, ax2)

    # UP panel
    lines!(ax1,
        up_manual_data.z_profiles[i0_idx][:, 1],
        up_manual_data.z_profiles[i0_idx][:, 3];
        label = "manual", color = :red, linewidth = 2, linestyle = :solid,
    )
    lines!(ax1,
        up_ΔG_data.z_profiles[i0_idx][:, 1],
        up_ΔG_data.z_profiles[i0_idx][:, 3];
        label = L"$\Delta\mathcal{G}$", color = :dodgerblue3, linewidth = 1.5, linestyle = :dashdot,
    )

    # DW panel
    lines!(ax2,
        dw_manual_data.z_profiles[i0_idx][:, 1],
        dw_manual_data.z_profiles[i0_idx][:, 3];
        label = "manual", color = :orange, linewidth = 2, linestyle = :solid,
    )
    lines!(ax2,
        dw_ΔG_data.z_profiles[i0_idx][:, 1],
        dw_ΔG_data.z_profiles[i0_idx][:, 3];
        label = L"$\Delta\mathcal{G}$", color = :purple, linewidth = 1.5, linestyle = :dashdot,
    )

    axislegend(ax1, position = :rt)
    axislegend(ax2, position = :rt)

    Label(fig[0, 1],
        L"CQD $|$ $T=%$(title_temp)\degree\mathrm{C}$ $|$ $I_c = %$(Int(1000 * ICOILS[i0_idx]))\,\mathrm{mA}$";
        fontsize = 18, tellwidth = false,
    )

    rowgap!(fig.layout, 1, 5)

    return fig
end

for i0_idx = 1:nI
    fig = plot_CQD_profiles(
        CQD_UP_T200_manual_data, CQD_UP_T200_ΔG_data,
        CQD_DW_T200_manual_data, CQD_DW_T200_ΔG_data,
        ICOILS, i0_idx;
        title_temp = 200,
    )
    display(fig)
end

for i0_idx = 1:nI
    fig = plot_CQD_profiles(
        CQD_UP_T205_manual_data, CQD_UP_T205_ΔG_data,
        CQD_DW_T205_manual_data, CQD_DW_T205_ΔG_data,
        ICOILS, i0_idx;
        title_temp = 205,
    )
    display(fig)
end


function plot_zmax_summary(
    up_manual_data, up_ΔG_data,
    dw_manual_data, dw_ΔG_data,
    ICOILS;
    fig_size  = (1300, 800),
    title_temp = 200,
)
    fig = Figure(size = fig_size)

    # ── log mask ──────────────────────────────────────────────────────────────
    log_mask = ICOILS .> 0

    # ── axis templates ────────────────────────────────────────────────────────
    axis_common = (
        xminorticksvisible = true,
        xminorticks        = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorticks        = IntervalsBetween(5),
        xminorgridvisible  = true,
        yminorgridvisible  = true,
        xminorgridcolor    = (:gray, 0.3),
        yminorgridcolor    = (:gray, 0.3),
        xminorgridwidth    = 0.5,
        yminorgridwidth    = 0.5,
        xlabel             = L"$I_c \ (\mathrm{A})$",
    )

    axis_log = merge(axis_common, (
        xscale      = log10,
        yscale      = log10,
        xlabel      = L"$I_c \ (\mathrm{A})$",
        xminorticks = IntervalsBetween(9),
        yminorticks = IntervalsBetween(9),
    ))

    # ── helpers ───────────────────────────────────────────────────────────────
    function plot_pair!(axlin, axlog, x, y_manual, y_ΔG;
            color_manual = :red, color_ΔG = :dodgerblue3)
        lines!(axlin, x, y_manual;
            label = "manual", color = color_manual, linewidth = 2, linestyle = :solid)
        lines!(axlin, x, y_ΔG;
            label = L"$\Delta\mathcal{G}$", color = color_ΔG, linewidth = 1.5, linestyle = :dashdot)
        lines!(axlog, x[log_mask], abs.(y_manual[log_mask]);
            label = "manual", color = color_manual, linewidth = 2, linestyle = :solid)
        lines!(axlog, x[log_mask], abs.(y_ΔG[log_mask]);
            label = L"$\Delta\mathcal{G}$", color = color_ΔG, linewidth = 1.5, linestyle = :dashdot)
    end

    function plot_residual!(ax, x, y_resid)
        lines!(ax, x, 1e3 .* y_resid;
            color = :black, linewidth = 1.5)
        scatter!(ax, x, 1e3 .* y_resid;
            color = :black, markersize = 6)
        hlines!(ax, [0.0]; color = (:gray, 0.6), linewidth = 1, linestyle = :dash)
    end

    # ── axes ──────────────────────────────────────────────────────────────────
    ax1  = Axis(fig[1, 1]; axis_common...,
        ylabel = L"$z_\mathrm{max} \ (\mathrm{mm})$",                                          title = "UP")
    ax1l = Axis(fig[1, 2]; axis_log...,
        ylabel = L"$|z_\mathrm{max}| \ (\mathrm{mm})$",                                        title = "UP (log, abs)")
    ax2  = Axis(fig[1, 3]; axis_common...,
        ylabel = L"$\Delta z_\mathrm{max} \ (\mathrm{\mu m})$",                                title = "UP residual")

    ax3  = Axis(fig[2, 1]; axis_common...,
        ylabel = L"$z_\mathrm{max} \ (\mathrm{mm})$",                                          title = "DW")
    ax3l = Axis(fig[2, 2]; axis_log...,
        ylabel = L"$|z_\mathrm{max}| \ (\mathrm{mm})$",                                        title = "DW (log, abs)")
    ax4  = Axis(fig[2, 3]; axis_common...,
        ylabel = L"$\Delta z_\mathrm{max} \ (\mathrm{\mu m})$",                                title = "DW residual")

    ax5  = Axis(fig[3, 1]; axis_common...,
        ylabel = L"$z_\mathrm{max}^\mathrm{UP} - z_\mathrm{max}^\mathrm{DW} \ (\mathrm{mm})$", title = "UP − DW")
    ax5l = Axis(fig[3, 2]; axis_log...,
        ylabel = L"$|z_\mathrm{max}^\mathrm{UP} - z_\mathrm{max}^\mathrm{DW}| \ (\mathrm{mm})$", title = "UP − DW (log, abs)")
    ax6  = Axis(fig[3, 3]; axis_common...,
        ylabel = L"$\Delta(z_\mathrm{max}^\mathrm{UP} - z_\mathrm{max}^\mathrm{DW}) \ (\mathrm{\mu m})$", title = "UP − DW residual")

    # ── Row 1: UP ─────────────────────────────────────────────────────────────
    plot_pair!(ax1, ax1l, ICOILS,
        up_manual_data.z_max,
        up_ΔG_data.z_max)

    plot_residual!(ax2, ICOILS,
        up_manual_data.z_max .- up_ΔG_data.z_max)

    # ── Row 2: DW ─────────────────────────────────────────────────────────────
    plot_pair!(ax3, ax3l, ICOILS,
        dw_manual_data.z_max,
        dw_ΔG_data.z_max;
        color_manual = :orange, color_ΔG = :purple)

    plot_residual!(ax4, ICOILS,
        dw_manual_data.z_max .- dw_ΔG_data.z_max)

    # ── Row 3: UP − DW ────────────────────────────────────────────────────────
    plot_pair!(ax5, ax5l, ICOILS,
        up_manual_data.z_max .- dw_manual_data.z_max,
        up_ΔG_data.z_max     .- dw_ΔG_data.z_max)

    plot_residual!(ax6, ICOILS,
        (up_manual_data.z_max .- dw_manual_data.z_max) .-
        (up_ΔG_data.z_max     .- dw_ΔG_data.z_max))

    # ── legends ───────────────────────────────────────────────────────────────
    axislegend(ax1, position = :rb)
    axislegend(ax3, position = :rt)
    axislegend(ax5, position = :rb)

    # ── suptitle ──────────────────────────────────────────────────────────────
    Label(fig[0, 1:3],
        L"CQD $|$ $T = %$(title_temp)\degree\mathrm{C}$ $|$ $z_\mathrm{max}$ vs $I_c$";
        fontsize = 18, tellwidth = false)

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 10)

    return fig
end

fig = plot_zmax_summary(
    CQD_UP_T200_manual_data, CQD_UP_T200_ΔG_data,
    CQD_DW_T200_manual_data, CQD_DW_T200_ΔG_data,
    ICOILS;
    title_temp = 200,
)
display(fig)

fig = plot_zmax_summary(
    CQD_UP_T205_manual_data, CQD_UP_T205_ΔG_data,
    CQD_DW_T205_manual_data, CQD_DW_T205_ΔG_data,
    ICOILS;
    title_temp = 205,
)
display(fig)


function plot_field_maps(
    ICOILS, i_idx,
    calibration, TheoreticalSimulation;
    n        = 200,
    fig_size = (800, 1200),
    quant    = 0.96,
)
    Bmanual_reference = TheoreticalSimulation.BvsI(ICOILS[i_idx])
    Gmanual_reference = TheoreticalSimulation.GvsI(ICOILS[i_idx])
    Iw_eff            = calibration.I_eff_B(ICOILS[i_idx])
    S                 = calibration.grad_scale(ICOILS[i_idx])

    # ── shared x range, different z ranges ────────────────────────────────────
    xrng   = LinRange(-3.75e-3, 3.75e-3, n)
    zrng_B = LinRange(-2.5e-3,  2.5e-3,  n)
    zrng_G = LinRange(-2.5e-3,  1.5e-3,  n)

    # ── compute maps ──────────────────────────────────────────────────────────
    B0_map = [sqrt(sum(TheoreticalSimulation.B_total(x, 0.0, z; Iw=Iw_eff) .^ 2))
              for x in xrng, z in zrng_B]

    dBdz_map = [begin
        Bx, By, Bz = TheoreticalSimulation.B_total(x, 0.0, z; Iw=Iw_eff)
        _, _, dBdz = S .* TheoreticalSimulation.grad_normB(x, 0.0, z, Bx, By, Bz; Iw=Iw_eff)
        dBdz
    end for x in xrng, z in zrng_G]

    # ── color limits ──────────────────────────────────────────────────────────
    finite_B = vec(B0_map[.!isnan.(B0_map) .& .!isinf.(B0_map)])
    vmin_B   = minimum(finite_B)
    vmax_B   = maximum(finite_B)

    finite_G = vec(dBdz_map[.!isnan.(dBdz_map) .& .!isinf.(dBdz_map)])
    vmax_G   = Statistics.quantile(abs.(finite_G), quant)

    # ── colorbar ticks ────────────────────────────────────────────────────────
    tick_vals_B = collect(range(vmin_B, vmax_B, length=6))
    tick_labs_B = string.(round.(tick_vals_B, sigdigits=3))
    all_vals_B  = vcat(tick_vals_B, Bmanual_reference)
    all_labs_B  = vcat(tick_labs_B, [L"$B_\mathrm{ref}$"])

    tick_vals_G = collect(range(-vmax_G, vmax_G, length=7))
    tick_labs_G = string.(round.(tick_vals_G, sigdigits=3))
    all_vals_G  = vcat(tick_vals_G, clamp(Gmanual_reference, -vmax_G, vmax_G))
    all_labs_G  = vcat(tick_labs_G, [L"$G_\mathrm{ref}$"])

    # ── axis template ─────────────────────────────────────────────────────────
    axis_common = (
        xlabelsize         = 24,
        ylabelsize         = 24,
        xticklabelsize     = 16,
        yticklabelsize     = 16,
        titlesize          = 24,
        xminorticksvisible = true,
        yminorticksvisible = true,
        xminorticks        = IntervalsBetween(5),
        yminorticks        = IntervalsBetween(5),
        ylabel             = L"$z \ (\mathrm{mm})$",
    )

    # ── figure ────────────────────────────────────────────────────────────────
    fig = Figure(size = fig_size)

    # suptitle
    Label(fig[0, 1:2],
        L"$I_c = %$(round(1000*ICOILS[i_idx], digits=1))\,\mathrm{mA}$";
        fontsize = 28, tellwidth = false,
    )

    # ── Row 1: |B| ────────────────────────────────────────────────────────────
    ax1 = Axis(fig[1, 1]; axis_common...,
        xlabel = "",
        xticklabelsvisible = true,
        title  = L"$|\mathbf{B}(x,z)|$",
    )

    hm1 = heatmap!(ax1,
        xrng * 1e3, zrng_B * 1e3, B0_map;
        colormap = :inferno,
    )
    contour!(ax1,
        xrng * 1e3, zrng_B * 1e3, B0_map;
        levels = 60, color = (:white, 0.4), linewidth = 0.8,
    )
    contour!(ax1,
        xrng * 1e3, zrng_B * 1e3, B0_map;
        levels = [Bmanual_reference], color = :cyan, linewidth = 2.0, linestyle = :dash,
    )

    Colorbar(fig[1, 2], hm1;
        label         = L"$|\mathbf{B}|$ (T)",
        width         = 15,
        labelsize     = 18,
        ticklabelsize = 14,
        ticks         = (all_vals_B, all_labs_B),
    )

    # ── Row 2: ∂_z|B| ─────────────────────────────────────────────────────────
    ax2 = Axis(fig[2, 1]; axis_common...,
        xlabel = L"$x \ (\mathrm{mm})$",
        title  = L"$\partial_z |\mathbf{B}(x,z)|$",
    )

    hm2 = heatmap!(ax2,
        xrng * 1e3, zrng_G * 1e3, dBdz_map;
        colormap   = :RdBu,
        colorrange = (-vmax_G, vmax_G),
        highclip   = :darkred,
        lowclip    = :darkblue,
    )
    contour!(ax2,
        xrng * 1e3, zrng_G * 1e3, dBdz_map;
        levels = 60, color = (:black, 0.3), linewidth = 0.8,
    )
    contour!(ax2,
        xrng * 1e3, zrng_G * 1e3, dBdz_map;
        levels = [Gmanual_reference], color = :black, linewidth = 2.0, linestyle = :dash,
    )

    Colorbar(fig[2, 2], hm2;
        label         = L"$\partial_z |\mathbf{B}|$ (T/m)",
        width         = 15,
        labelsize     = 18,
        ticklabelsize = 14,
        ticks         = (all_vals_G, all_labs_G),
    )

    colgap!(fig.layout, 1, 7)
    rowgap!(fig.layout, 10)

    return fig
end

for i_idx=1:nI
    fig = plot_field_maps(ICOILS, i_idx, calibration, TheoreticalSimulation)
    display(fig)
end