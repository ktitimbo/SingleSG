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
using Interpolations, Roots, Loess, Optim
using BSplineKit, Polynomials
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
const BASE_PATH = raw"F:\SternGerlachExperiments";
const OUTDIR    = joinpath(@__DIR__, "simulation_data", "CQD_" * RUN_STAMP);
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
const SUP  = Dict(zip("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
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
const TCelsius = 200
const T_K = 273.15 + TCelsius ; # Furnace temperature (K)
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
    Temperature             : $(T_K)K
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
JLD2_MyTools.save_script_copy(OUTDIR; script_path=@__FILE__, timestamp=RUN_STAMP)
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
@info "No of currents : $(nI)"
calibration = TheoreticalSimulation.build_calibration(Icoils; degree=3, span =0.12);

# Induction terms 
kis     = unique(round.(vcat([x * exp10(p) for p in -6:-6 for x in 0.5:0.5:4.0],0.001);sigdigits=4))
@info "Number of ki sampled = $(length(kis))" 
induction_coeff_for_label     = round.(1e6 .* kis; sigdigits=3)
ki_labels = [(e = floor(Int, log10(abs(k)) + 1e-9);
              string(round(k / exp10(e), sigdigits=3), "×10", join(SUP[c] for c in string(e))))
             for k in kis]


# Sample size: number of atoms arriving to the screen
const Nss = 6_000_000 ; 
@info "Number of MonteCarlo particles : $(Nss)\n"

const DATA_READY = false

if DATA_READY
    const OUTDIR_PATH = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T$(TCelsius)_6M_constG")
    # const OUTDIR_PATH = joinpath(dirname(OUTDIR),"20260521T142848073")
else
    const OUTDIR_PATH = OUTDIR
    # Monte Carlo generation of particles traersing the filtering slit [x0 y0 z0 v0x v0y v0z]
    crossing_slit = generate_samples(Nss, effusion_params; v_pdf=:v3, rng = rng_set, multithreaded = false, base_seed = base_seed_set);
    jldsave( joinpath(OUTDIR_PATH,"cross_slit_particles_$(Nss).jld2"), data = crossing_slit)

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

    data_UP_SG = TheoreticalSimulation.propagate_to_SG_entrance(data_UP);
    data_DOWN_SG = TheoreticalSimulation.propagate_to_SG_entrance(data_DOWN);

    isdir(joinpath(OUTDIR_PATH, "up")) || mkpath(joinpath(OUTDIR_PATH, "up"))
    isdir(joinpath(OUTDIR_PATH, "dw")) || mkpath(joinpath(OUTDIR_PATH, "dw"))
    for (i, kI) in enumerate(kis)
        @info "\e[1;31mRUNNING FOR kᵢ = $(ki_labels[i])\e[0m"


        for (label, label_str, data, data_SG) in ((:up, "UP", data_UP,   data_UP_SG),
                                    (:dw, "DOWN", data_DOWN, data_DOWN_SG))

            @info "\e[1;32mAnalyzing data for electron magnetic moment $(uppercase(String(label_str)))\e[0m"

            particles_flag = TheoreticalSimulation.CQD_flag_travelling_particles_twowires(
                Icoils, data, data_SG, kI, K39_params, calibration; y_length=5001, verbose=true)

            @time particles_trajectories = TheoreticalSimulation.CQD_build_travelling_particles_twowires(
                Icoils, kI, data, data_SG, particles_flag, K39_params, calibration)   # [x0 y0 z0 vx0 vy0 vz0 θe θn x z vz]

            TheoreticalSimulation.CQD_travelling_particles_summary(Icoils, particles_trajectories, label_str)

            particles_screen = TheoreticalSimulation.CQD_select_flagged(particles_trajectories, :screen)

            filepath = joinpath(OUTDIR_PATH, String(label),
                "cqd$(RUN_STAMP)_ki$(@sprintf("%03d", i))_$(label)_screen.jld2")

            jldopen(filepath, "w") do f
                # Global metadata
                f["meta/T"]  = T_K
                f["meta/N"]  = Nss
                f["meta/k"]  = kI
                f["meta/Iw"] = collect(Icoils)

                for (idx, Iw) in enumerate(Icoils)
                    # Save only passed particles arriving to the detector
                    screen_camera = particles_screen[idx]
                    f["data/final/I$(idx)"] = screen_camera

                    @info "$(idx)/$(length(Icoils))" Iw_mA = round(Int, 1000 * Iw) passed_pct = round(100 * size(screen_camera, 1) / size(data, 1); digits=3) z_mm = round.(extrema(1e3 .* screen_camera[:, 10]); digits=3) x_mm = round.(extrema(1e3 .* screen_camera[:, 9]); digits=3)
                end
            end

            @info "\e[1;33mSaved\e[0m" filepath=filepath
        end
    end

    ######################################################################
    T_END = Dates.now()
    T_RUN = Dates.canonicalize(T_END-T_START)
    report = """
    ***************************************************
    EXPERIMENT
        Single Stern–Gerlach Experiment
        atom                    : $(atom)
        Output directory        : $(OUTDIR_PATH)
        RUN_STAMP               : $(RUN_STAMP)

    CAMERA FEATURES
        Number of pixels        : $(nx_pixels) × $(nz_pixels)
        Pixel size              : $(1e6*cam_pixelsize) μm

    SETUP FEATURES
        Temperature             : $(T_K)K
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
        Currents (A)            : $(round.(Icoils,sigdigits=3))
        No. of currents         : $(nI)

    CODE
        Code name               : $(PROGRAM_FILE)
        Start date              : $(T_START)
        End date                : $(T_END)
        Run time                : $(T_RUN)
        Hostname                : $(HOSTNAME)

    ***************************************************
    """
    # Print to terminal
    println(report)

    # Save to file
    open(joinpath(OUTDIR_PATH,"simulation_cqd_report.txt"), "w") do io
        write(io, report)
    end

    @info "\e[1;32mDATA COLLECTED : $RUN_STAMP\e[0m"

end


######################################################################
######################################################################
######################################################################

nx_bins             = 32; # fixed nx bins
nz_bins             = [1, 2];
gaussian_width_mm   = [0.001, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300, 0.350, 0.400, 0.500 ]; # try different gaussian widths
λ0_raw_list         = [0.001, 0.005, 0.01, 0.02]#, 0.03, 0.04, 0.05, 0.10]; # try different smoothing factors for raw data
λ0_spline           = 0.001;

# ---------- precompute param grid ----------
params = [(nz, gw, λ0_raw)
          for nz in nz_bins
          for gw in gaussian_width_mm
          for λ0_raw in λ0_raw_list];


"""
    process_branch!(branch, OUTDIR, Nss, induction_coeff_for_label, nz_bins,
                    gaussian_width_mm, λ0_raw_list, params, nx_bins, kis,
                    T_K, λ0_spline)

Process all simulation files for a given branch (`:up` or `:dw`), computing
CQD profiles over all parameter combinations (nz, σw, λ0) and saving results
incrementally to a JLD2 file.

Skips parameter combinations that were already computed in a previous run,
making it safe to call repeatedly — progress is always preserved.

# Arguments
- `branch`                    : `:up` or `:dw`
- `OUTDIR`                    : root output directory; files are read from and written to `OUTDIR/branch/`
- `Nss`                       : number of samples (used in output filename and metadata)
- `induction_coeff_for_label` : vector of induction coefficients, one per simulation file
- `nz_bins`                   : vector of z-bin counts to sweep over
- `gaussian_width_mm`         : vector of Gaussian widths (mm) to sweep over
- `λ0_raw_list`               : vector of raw λ0 values to sweep over
- `params`                    : pre-expanded parameter grid (nz, gw, λ0) — e.g. from `Iterators.product`
- `nx_bins`                   : number of x bins (fixed across all runs)
- `kis`                       : vector of induction coefficient labels aligned with `files`
- `T_K`                       : temperature in Kelvin (metadata only)
- `λ0_spline`                 : smoothed spline of λ0 (metadata + passed to analyzer)
"""
function process_branch!(branch, OUTDIR, Nss, kis, induction_coeff_for_label, 
                         nx_bins, nz_bins, gaussian_width_mm, λ0_raw_list, λ0_spline,
                         params, T_K
)

    indir  = joinpath(OUTDIR, String(branch))
    outjld = joinpath(OUTDIR, String(branch), "cqd_$(Nss)_$(branch)_profiles.jld2")

    # Discover .jld2 simulation files present at this moment
    files  = sort(filter(f -> isfile(joinpath(indir, f)) && endswith(f, "$(branch)_screen.jld2"),
                         readdir(indir)))
    nfiles = length(files)

    # Nothing to do yet — caller decides whether to retry later
    nfiles == 0 && (@info "No files yet for $(branch), skipping."; return)

    Ntot = nfiles * length(nz_bins) * length(gaussian_width_mm) * length(λ0_raw_list)
    @info "\e[93m[$(uppercase(String(branch)))] Files found: $(nfiles), total profiles: $(Ntot)\e[0m"

    # Open output JLD2 in append mode so partial results from previous runs are preserved
    @time jldopen(outjld, "a+") do f

        # Write metadata once on first open; on subsequent opens only refresh
        # meta/files since new simulation files may have appeared since last run
        if !haskey(f, "meta/N")
            f["meta/N"]        = Nss
            f["meta/T"]        = T_K
            f["meta/branch"]   = String(branch)
            f["meta/s_spline"] = λ0_spline
            f["meta/nx"]       = nx_bins
            f["meta/nz"]       = nz_bins
            f["meta/σw"]       = gaussian_width_mm
            f["meta/λ0"]       = λ0_raw_list
            f["meta/ki"]       = kis
            f["meta/files"]    = files
        else
            # JLD2 doesn't support in-place mutation, so delete and rewrite
            delete!(f, "meta/files")
            f["meta/files"] = files
        end

        for (j, fname) in pairs(files)

            ki      = induction_coeff_for_label[j]
            simpath = joinpath(indir, fname)

            @info "\e[96m[$(uppercase(String(branch)))] Processing file $(j)/$(nfiles)\e[0m" file_name=fname ki=kis[j]

            # Sweep over all (nz, σw, λ0) combinations for this simulation file
            @time for pidx in eachindex(params)

                nz, gw, λ0_raw = params[pidx]
                label_path = JLD2_MyTools.make_keypath_cqd(branch, ki, nz, gw, λ0_raw)

                # Resume support: skip combinations already saved
                haskey(f, label_path) && (@info "Skipping (already computed)" key=label_path; continue)

                @info "\e[1;33mAnalyzing: nz=$(nz)  σw=$(@sprintf("%.3f",gw))  λ0=$(λ0_raw)\e[0m"

                # Main expensive call — reads simulation file, returns profile dict
                profiles = TheoreticalSimulation.CQD_analyze_profiles_to_dict(
                    simpath;
                    n_bins      = (nx_bins, nz),
                    width_mm    = gw,
                    add_plot    = false,
                    plot_xrange = :all,
                    branch      = branch,
                    λ_raw       = λ0_raw,
                    λ_smooth    = λ0_spline,
                    mode        = :probability
                )

                # Save immediately — if the run crashes, completed work is not lost
                f[label_path] = profiles

                # Explicitly release large result before next iteration
                profiles = nothing
            end

            @info "Done file $(j)/$(nfiles)" free_GiB = round(Sys.free_memory() / 1024^3, digits=3)
        end
    end

    @info "\e[91mCompleted $(uppercase(String(branch))) table\e[0m"
end

# Returns true if all expected simulation files are present for `branch`
files_ready(branch) = length(filter(
    f -> isfile(joinpath(OUTDIR_PATH, String(branch), f)) && endswith(f, "$(branch)_screen.jld2"),
    readdir(joinpath(OUTDIR_PATH, String(branch)))
)) == length(induction_coeff_for_label)

# --- Dispatch: single pass if all files are ready, polling loop otherwise ---
if all(files_ready(b) for b in (:up, :dw))
    @info "All files present, running single pass."
    for branch in (:up, :dw)
        process_branch!(branch, OUTDIR_PATH, Nss, kis, induction_coeff_for_label, 
                            nx_bins, nz_bins, gaussian_width_mm, λ0_raw_list, λ0_spline,
                            params, T_K)
    end
else
    # Some files are still being generated — keep looping until both branches are complete
    @info "Files still being generated, switching to polling mode."
    while true
        for branch in (:up, :dw)
            process_branch!(branch, OUTDIR_PATH, Nss, kis, induction_coeff_for_label, 
                            nx_bins, nz_bins, gaussian_width_mm, λ0_raw_list, λ0_spline,
                            params, T_K)
        end

        all(files_ready(b) for b in (:up, :dw)) && break

        @info "Not all files present yet, sleeping 30s..."
        sleep(30)
    end
end

JLD2_MyTools.list_keys_jld_cqd(joinpath(OUTDIR_PATH, "up", "cqd_$(Nss)_up_profiles.jld2"));
JLD2_MyTools.list_keys_jld_cqd(joinpath(OUTDIR_PATH, "dw", "cqd_$(Nss)_dw_profiles.jld2"));

######################################################################
T_END = Dates.now()
T_RUN = Dates.canonicalize(T_END-T_START)
report = """
***************************************************
EXPERIMENT
    Single Stern–Gerlach Experiment
    atom                    : $(atom)
    Base directory          : $(BASE_PATH)
    Output directory        : $(OUTDIR_PATH)
    RUN_STAMP               : $(RUN_STAMP)

CAMERA FEATURES
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm

SETUP FEATURES
    Temperature             : $(T_K)K
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
    End date                : $(T_END)
    Run time                : $(T_RUN)
    Hostname                : $(HOSTNAME)

***************************************************
"""
# Print to terminal
println(report)

# Save to file
open(joinpath(OUTDIR_PATH,"simulation_cqd_report.txt"), "w") do io
    write(io, report)
end

@info "\e[1;32mDATA ANALYZED : script $RUN_STAMP has finished!\e[0m"
alert("script $RUN_STAMP has finished!")