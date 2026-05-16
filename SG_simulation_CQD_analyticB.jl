# Simulation of atom trajectories in the Stern–Gerlach experiment
# Kelvin Titimbo
# California Institute of Technology
# May 2026

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
using Interpolations, Roots, Loess, Optim, Polynomials
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
@info "\e[1;31mCreated output directory\e[0m" OUTDIR
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
# atom_info       = AtomicSpecies.atoms(atom);
K39_params = AtomParams(atom); # [R μn γn Ispin Ahfs M ] 
const μe_over_M = μₑ / K39_params.M
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
# SG magnets characteristic length
const 𝒶 = 2.5e-3 ;
const ℓ = 0.5*y_SG
const center_of_SG_magnet = y_FurnaceToSlit + y_SlitToSG + ℓ
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
    SG characteristic       : $(1e3*𝒶)mm
***************************************************
""")
# Setting the variables for the module
TheoreticalSimulation.default_camera_pixel_size     = cam_pixelsize;
TheoreticalSimulation.default_x_pixels              = nx_pixels;
TheoreticalSimulation.default_z_pixels              = nz_pixels;
TheoreticalSimulation.default_x_furnace             = x_furnace;
TheoreticalSimulation.default_z_furnace             = z_furnace;
TheoreticalSimulation.default_x_slit                = x_slit;
TheoreticalSimulation.default_z_slit                = z_slit;
TheoreticalSimulation.default_y_FurnaceToSlit       = y_FurnaceToSlit;
TheoreticalSimulation.default_y_SlitToSG            = y_SlitToSG;
TheoreticalSimulation.default_y_SG                  = y_SG;
TheoreticalSimulation.default_y_SGToScreen          = y_SGToScreen;
TheoreticalSimulation.default_R_tube                = R_tube;
TheoreticalSimulation.default_c_aperture            = R_aper;
TheoreticalSimulation.default_y_SGToAperture        = y_SGToAperture;
TheoreticalSimulation.default_𝒶                     = 𝒶;
TheoreticalSimulation.default_ℓ                     = ℓ;
TheoreticalSimulation.default_center_of_SG_magnet   = center_of_SG_magnet
##################################################################################################

function run_save_and_filter(data, Icoils, calibration, filepath_raw, filepath_passed;
                              k, μ_over_m,
                              θ0_col    = 7,
                              grad_mask = (0.0, 0.0, 1.0))

    jldopen(filepath_raw, "w") do f_raw
        jldopen(filepath_passed, "w") do f_pass

            # raw file keeps full metadata including initial conditions
            f_raw["meta/k"]       = k
            f_raw["meta/initial"] = data
            f_raw["meta/Iw"]      = collect(Icoils)

            # passed file only needs current list and k — no link to initial data
            f_pass["meta/k"]  = k
            f_pass["meta/Iw"] = collect(Icoils)

            for (idx, Iw) in enumerate(Icoils)
                screen = TheoreticalSimulation.run_ensemble(Iw, data, calibration;
                             μ_over_m  = μ_over_m,
                             k         = k,
                             θ0_col    = θ0_col,
                             grad_mask = grad_mask)

                f_raw["data/I$(idx)"]  = screen
                f_pass["data/I$(idx)"] = screen[screen[:, end] .== 1.0, 1:end-1]

                @info "$(idx)/$(length(Icoils))" Iw="$(Int(round(1000*Iw)))mA" z_mm="$(round.(extrema(1e3 .* screen[:,3]); digits=3))mm" x_mm="$(round.(extrema(1e3 .* screen[:,1]); digits=3))mm"
            end
        end
    end

    @info "Saved" raw=filepath_raw passed=filepath_passed currents=length(Icoils) particles=size(data,1)
end

function run_save_and_filter2(data, Icoils, calibration, filepath;
                              k, μ_over_m,
                              grad_mask = (0.0, 0.0, 1.0))

    jldopen(filepath, "w") do f

        # Global metadata
        f["meta/k"]       = k
        f["meta/initial"] = data
        f["meta/Iw"]      = collect(Icoils)

        for (idx, Iw) in enumerate(Icoils)

            screen = TheoreticalSimulation.run_ensemble2(
                Iw, data, calibration;
                μ_over_m  = μ_over_m,
                k         = k,
                grad_mask = grad_mask
            )

            # Mask for particles that reached / passed the camera
            passed_mask = screen[:, end] .== 1.0

            # Initial conditions of particles that passed
            data_initial_passed = data[passed_mask, :]

            # Final screen/camera data, dropping the pass/fail column
            screen_camera = screen[passed_mask, 1:end-1]

            # Save the corresponding initial particles
            f["data/initial/I$(idx)"] = data_initial_passed
            # Save only passed particles
            f["data/final/I$(idx)"] = screen_camera

            @info "$(idx)/$(length(Icoils))" Iw_mA = round(Int, 1000 * Iw) npassed = count(passed_mask) z_mm  = round.(extrema(1e3 .* screen_camera[:, 3]); digits=3) x_mm  = round.(extrema(1e3 .* screen_camera[:, 1]); digits=3)
        end
    end

    @info "\e[1;33mSaved\e[0m" filepath=filepath
end

# Coil currents
Icoils = [0.00,
            # 0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            # 0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
            # 0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
            # 0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,
            0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
];
nI = length(Icoils);
@info "No of currents : $(nI)"
calibration = TheoreticalSimulation.build_calibration(Icoils);

# Sample size: number of atoms arriving to the screen
const Nss = 200 ; 
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

kis                 = unique(round.([x * exp10(p) for p in -6:-6 for x in 0.5:0.5:1.5];sigdigits=4))
@info "Number of ki sampled = $(length(kis))" 
induction_coeff     = round.(1e6 .* kis; sigdigits=3)

isdir(joinpath(OUTDIR,"up")) || mkpath(joinpath(OUTDIR,"up"));
isdir(joinpath(OUTDIR,"dw")) || mkpath(joinpath(OUTDIR,"dw"));
for (i,ki) in enumerate(kis)
    @info "\e[1;31mRUNNING FOR kᵢ = $(round(1e6*ki,sigdigits=3))×10⁻⁶\e[0m"

    @info "\e[1;32mAnalyzing data for electron magnetic moment UP\e[0m"
    run_save_and_filter2(data_UP, Icoils, calibration,
        joinpath(OUTDIR, "up", "cqd$(RUN_STAMP)_ki$(@sprintf("%03d", i))_up_screen.jld2");
        k=ki, μ_over_m=μe_over_M)

    @info "\e[1;32mAnalyzing data for electron magnetic moment DOWN\e[0m"
    run_save_and_filter2(data_DOWN, Icoils, calibration,
        joinpath(OUTDIR, "dw", "cqd$(RUN_STAMP)_ki$(@sprintf("%03d", i))_dw_screen.jld2");
        k=-ki, μ_over_m=μe_over_M)

    GC.gc()
end

@info "\e[1;32mDATA COLLECTED : $RUN_STAMP\e[0m"

nx_bins             = 32; # fixed nx bins
nz_bins             = [1, 2];
gaussian_width_mm   = [0.001, 0.010, 0.025, 0.050, 0.065, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.270, 0.275, 0.300, 0.350, 0.400, 0.450, 0.500 ]; # try different gaussian widths
λ0_raw_list         = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10]; # try different smoothing factors for raw data
λ0_spline           = 0.001;

# ---------- precompute param grid ----------
params = [(nz, gw, λ0_raw)
          for nz in nz_bins
          for gw in gaussian_width_mm
          for λ0_raw in λ0_raw_list]



# =========================================================
# ======================== UP =============================
# =========================================================
const INDIR_up = joinpath(OUTDIR,"up")
# --- Files ---
files = sort(filter(f -> isfile(joinpath(INDIR_up, f)) && endswith(f, ".jld2"),
               readdir(INDIR_up)))
nfiles = length(files)
@assert nfiles == length(induction_coeff) "Mismatch: files vs induction_coeff"

# Total combinations
Ntot = nfiles * length(nz_bins) * length(gaussian_width_mm) * length(λ0_raw_list)
@info "\e[93mTotal profiles to compute : Nkᵢ × Nnz × Nσ × Nλ0 × Nλs = $(Ntot)\e[0m"

# ============================================================
# Output file + metadata
# ============================================================

outjld = joinpath(OUTDIR, "cqd_$(Nss)_up_profiles.jld2")

# ============================================================
# Main serial loop
# ============================================================

@time jldopen(outjld, "w") do f

    # ------------------------------------------------------------
    # Global metadata
    # ------------------------------------------------------------
    f["meta/N"]        = Nss
    f["meta/T"]        = T_K
    f["meta/branch"]   = "up"
    f["meta/s_spline"] = λ0_spline
    f["meta/nx"]       = nx_bins
    f["meta/nz"]       = nz_bins
    f["meta/σw"]       = gaussian_width_mm
    f["meta/λ0"]       = λ0_raw_list
    f["meta/ki"]       = kis
    f["meta/files"]    = files

    # ------------------------------------------------------------
    # Loop over simulation files
    # ------------------------------------------------------------
    for (j, fname) in pairs(files)

        ki      = induction_coeff[j]
        simpath = joinpath(INDIR_up, fname)

        @info "\e[96mProcessing file $(j)/$(length(files))\e[0m" fname=fname ki=ki

        # --------------------------------------------------------
        # Loop over all analysis parameter combinations
        # --------------------------------------------------------
        @time for pidx in eachindex(params)

            nz, gw, λ0_raw = params[pidx]

            # Build output key before doing expensive work.
            # This makes the loop resume-friendly if the file is opened with "r+" later.
            label_path = JLD2_MyTools.make_keypath_cqd(
                :up, ki, nz, gw, λ0_raw
            )

            @info "Analyzing" nz=nz σw=gw λ0=λ0_raw

            # Main expensive call.
            # This reads from simpath and returns the profile dictionary.
            profiles_up = TheoreticalSimulation.CQD_analyze_profiles_to_dict(
                simpath;
                n_bins      = (nx_bins, nz),
                width_mm    = gw,
                add_plot    = false,
                plot_xrange = :all,
                branch      = :up,
                λ_raw       = λ0_raw,
                λ_smooth    = λ0_spline,
                mode        = :probability
            )

            # Save immediately after each parameter set.
            # This avoids accumulating many large dictionaries in RAM.
            f[label_path] = profiles_up

            # Release large result from memory before the next iteration.
            profiles_up = nothing

            # Optional: collect only occasionally to avoid slowing every iteration.
            if pidx % 5 == 0
                GC.gc()
            end
        end

        GC.gc()

        @info "Done file $(j)/$(length(files))" free_GiB = round(Sys.free_memory() / 1024^3, digits=3)
    end
end
@info "\e[91mCompleted UP table\e[0m"



# =========================================================
# ======================== DOWN ===========================
# =========================================================
const INDIR_dw = joinpath(OUTDIR,"dw")
# --- Files ---
files = sort(filter(f -> isfile(joinpath(INDIR_dw, f)) && endswith(f, ".jld2"),
               readdir(INDIR_dw)))
nfiles = length(files)
@assert nfiles == length(induction_coeff) "Mismatch: files vs induction_coeff"

# Total combinations
Ntot = nfiles * length(nz_bins) * length(gaussian_width_mm) * length(λ0_raw_list)
@info "\e[93mTotal profiles to compute : Nkᵢ × Nnz × Nσ × Nλ0 × Nλs = $(Ntot)\e[0m"

# ============================================================
# Output file + metadata
# ============================================================

outjld = joinpath(OUTDIR, "cqd_$(Nss)_dw_profiles.jld2")

# ============================================================
# Main serial loop
# ============================================================

@time jldopen(outjld, "w") do f

    # ------------------------------------------------------------
    # Global metadata
    # ------------------------------------------------------------
    f["meta/N"]        = Nss
    f["meta/T"]        = T_K
    f["meta/branch"]   = "dw"
    f["meta/s_spline"] = λ0_spline
    f["meta/nx"]       = nx_bins
    f["meta/nz"]       = nz_bins
    f["meta/σw"]       = gaussian_width_mm
    f["meta/λ0"]       = λ0_raw_list
    f["meta/ki"]       = kis
    f["meta/files"]    = files

    # ------------------------------------------------------------
    # Loop over simulation files
    # ------------------------------------------------------------
    for (j, fname) in pairs(files)

        ki      = induction_coeff[j]
        simpath = joinpath(INDIR_dw, fname)

        @info "\e[96mProcessing file $(j)/$(length(files))\e[0m" fname=fname ki=ki

        # --------------------------------------------------------
        # Loop over all analysis parameter combinations
        # --------------------------------------------------------
        @time for pidx in eachindex(params)

            nz, gw, λ0_raw = params[pidx]

            # Build output key before doing expensive work.
            # This makes the loop resume-friendly if the file is opened with "r+" later.
            label_path = JLD2_MyTools.make_keypath_cqd(
                :dw, ki, nz, gw, λ0_raw
            )

            @info "Analyzing" nz=nz σw=gw λ0=λ0_raw

            # Main expensive call.
            # This reads from simpath and returns the profile dictionary.
            profiles_dw = TheoreticalSimulation.CQD_analyze_profiles_to_dict(
                simpath;
                n_bins      = (nx_bins, nz),
                width_mm    = gw,
                add_plot    = false,
                plot_xrange = :all,
                branch      = :dw,
                λ_raw       = λ0_raw,
                λ_smooth    = λ0_spline,
                mode        = :probability
            )

            # Save immediately after each parameter set.
            # This avoids accumulating many large dictionaries in RAM.
            f[label_path] = profiles_dw

            # Release large result from memory before the next iteration.
            profiles_dw = nothing

            # Optional: collect only occasionally to avoid slowing every iteration.
            if pidx % 5 == 0
                GC.gc()
            end
        end

        GC.gc()

        @info "Done file $(j)/$(length(files))" free_GiB = round(Sys.free_memory() / 1024^3, digits=3)
    end
end
@info "\e[91mCompleted DOWN table\e[0m"

# ii = rand(1:Nss)
# CQD_up_particles_flag         = TheoreticalSimulation.CQD_flag_travelling_particles(Icoils, data_UP, kI, K39_params; y_length=5001,verbose=true);
# CQD_up_particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles(Icoils, kI, data_UP, CQD_up_particles_flag, K39_params);     # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]
# jj = 19;
# Ij = Icoils[jj];

# TheoreticalSimulation.B_total(0.0001,0,0.0002; Iw=Ij) 
# TheoreticalSimulation.approx_B_total(0.0001,0,0.0002; Iw=Ij)

# TheoreticalSimulation.grad_normB(0.0001,0,0.0002; Iw=Ij)
# TheoreticalSimulation.approx_grad_normB(0.0001,0,0.0002; Iw=Ij)
# TheoreticalSimulation.approx_dnormBdz(0.0001,0.0002; Iw=Ij)


# scatter(Icoils, TheoreticalSimulation.BvsI.(Icoils), 
#     label="Magnetic field (target)",
#     marker=(:circle,2,:white));
# plot!(Icoils, [TheoreticalSimulation.B_total(0,0,0; Iw=calibration.I_eff_B(x))[3] for x in Icoils],
#     label="Magnetic field corrected" );
# plot!(xlabel="Current (A)", ylabel=L"Magnetic field $B_{z}$ (T)")

# scatter(Icoils, TheoreticalSimulation.GvsI.(Icoils), 
#     label="Gradient (target)",
#     marker=(:circle,2,:white));
# plot!(Icoils, [calibration.grad_scale(x) for x in Icoils].*[TheoreticalSimulation.grad_normB(0,0,0; Iw=calibration.I_eff_B(x))[1] for x in Icoils],
#     label=L"$\partial_{x}|B|$ corrected" );
# plot!(Icoils, [calibration.grad_scale(x) for x in Icoils].*[TheoreticalSimulation.grad_normB(0,0,0; Iw=calibration.I_eff_B(x))[2] for x in Icoils],
#     label=L"$\partial_{y}|B|$ corrected" );
# plot!(Icoils, [calibration.grad_scale(x) for x in Icoils].*[TheoreticalSimulation.grad_normB(0,0,0; Iw=calibration.I_eff_B(x))[3] for x in Icoils],
#     label=L"$\partial_{z}|B|$ corrected" );
# plot!(xlabel="Current (A)", ylabel=L"Magnetic field Gradient $\partial_{z}|B|$ (T/m)")


# r0 = data_UP[ii, 1:3]
# v0 = data_UP[ii, 4:6]
# θ0 = data_UP[ii, 7]          # initial polar angle from generate_CQDinitial_conditions

# result = TheoreticalSimulation.full_trajectory(Ij, r0, v0, calibration;
#     μ_over_m = μe_over_M,
#     k        = kI,
#     θ0       = θ0,
#     grad_mask = (0.0,0.0,1.0)
#     );

# println("""
# ***************************************************
# FINITE TWO-WIRE MODEL
#     r at furnace   = $(round.(1e3 .* r0; digits=3))          mm
#     v at furnace   = $(round.(v0; digits=3))      m/s

#     r at slit      = $(round.(1e3 .* result.sol_magnet.u[1][1:3]; digits=3))        mm
#     v at slit      = $(round.(result.sol_magnet.u[1][4:6]; digits=3))      m/s

#     r at SG-in     = $(round.(1e3 .* result.sol_magnet.u[2][1:3]; digits=3))        mm
#     v at SG-in     = $(round.(result.sol_magnet.u[2][4:6]; digits=3))      m/s

#     r at SG-centre = $(round.(1e3 .* result.sol_magnet.u[3][1:3]; digits=3))        mm
#     v at SG_centre = $(round.(result.sol_magnet.u[3][4:6]; digits=3))      m/s

#     r at SG-out    = $(round.(1e3 .* result.sol_magnet.u[4][1:3]; digits=3))         mm
#     v at SG_out    = $(round.(result.sol_magnet.u[4][4:6]; digits=3))       m/s

#     r at aperture  = $(round.(1e3 .* result.sol_magnet.u[5][1:3]; digits=3))        mm
#     v at aperture  = $(round.(result.sol_magnet.u[5][4:6]; digits=3))       m/s

#     r at screen    = $(round.(1e3 .* result.r_screen; digits=3))        mm
#     v at screen    = $(round.(result.v_screen; digits=3))       m/s
# ***************************************************
# MANUAL FIELD : SIMPLIFIED VERSION
#     r at furnace   = $(round.(1e3*CQD_up_particles_trajectories[19][ii,1:3]; digits=3)) mm
#     v at furnace   = $(round.(CQD_up_particles_trajectories[19][ii,4:6]; digits=3)) m/s
#     (x,z) at screen = $(round.(1e3*CQD_up_particles_trajectories[19][ii,9:10]; digits=3)) mm
#     vz at screen    = $(round.(CQD_up_particles_trajectories[19][ii,11]; digits=3)) m/s

# """)

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
    Number of atoms         : $(Nss)
    Induction term          : ($(kis))
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

@info "\e[1;32mDATA ANALYZED : script $RUN_STAMP has finished!\e[0m"
alert("script $RUN_STAMP has finished!")
