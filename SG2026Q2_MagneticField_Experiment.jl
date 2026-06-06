# ╔══════════════════════════════════════════════════════════════════════════════
# ║  Stern–Gerlach / CQD  —  Convergence study entry point
# ║  Loads experimental data and analysis summary, then produces a diagnostic
# ║  dual-axis plot of B-field calibration and peak separation vs. coil current.
# ╚══════════════════════════════════════════════════════════════════════════════
# ── Output format & persistence ───────────────────────────────────────────────
using CairoMakie
const FIG_EXT  = "png"   # Supported: "pdf" | "svg" | "png"
const SAVE_FIG = true    # Set false for interactive-only sessions
# ── Standard-library imports ──────────────────────────────────────────────────
using LinearAlgebra
using JLD2
using Dates
# ── Timing & run identification ───────────────────────────────────────────────
const T_START   = Dates.now()
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss")
# ── Working directory & output paths ─────────────────────────────────────────
cd(@__DIR__)   # Ensure relative paths resolve from the script's location
const BASE_PATH = raw"F:\SternGerlachExperiments"
# Each run gets its own timestamped subdirectory to avoid overwriting outputs.
const OUTDIR = joinpath(@__DIR__, "data_studies", "CONV_" * RUN_STAMP)
isdir(OUTDIR) || mkpath(OUTDIR)
@info "Output directory" path  = OUTDIR
@info "Hostname"         host  = gethostname()
@info "Run stamp"        stamp = RUN_STAMP
# ── Custom module loading ─────────────────────────────────────────────────────
include("./Modules/atoms.jl")
include("./Modules/samplings.jl")
include("./Modules/DataReading.jl")
include("./Modules/ProfileFitTools.jl")
include("./Modules/JLD2_MyTools.jl")
include("./Modules/TheoreticalSimulation.jl")
# Forward output settings into TheoreticalSimulation so figures are written
# to the same timestamped directory with the chosen extension.
TheoreticalSimulation.SAVE_FIG = SAVE_FIG
TheoreticalSimulation.FIG_EXT  = FIG_EXT
TheoreticalSimulation.OUTDIR   = OUTDIR

# ── Dataset selection ─────────────────────────────────────────────────────────
data_directories = ["20260529", "20260603"]
data_directory   = data_directories[1]   # Change index to switch dataset

# Canonical paths for raw experiment data and pre-computed analysis summaries
EXPERIMENT_PATH                  = joinpath(BASE_PATH, "EXPERIMENTS")
EXPERIMENT_ANALYSIS_SUMMARY_PATH = joinpath(BASE_PATH, "EXPDATA_ANALYSIS", "summary")

# ── Load raw experimental data ────────────────────────────────────────────────
exp_data = load(
    joinpath(EXPERIMENT_PATH, data_directory, "data.jld2"),
    "data",
)

# Build a 2-column matrix [current_A | B_field_T], sorted by coil current.
# Column 1: SG1 coil current  [A]
# Column 2: SG1 magnetic field [T]
current_field = let M = hcat(exp_data[:SG1currentInA], exp_data[:SG1BfieldInTesla])
    M[sortperm(M[:, 1]), :]
end

# ── Load analysis summary (peak positions) ────────────────────────────────────
# Retrieves the summary entry for this dataset at the canonical key path
# (run index 2, threshold 0.005).
exp_analysis = jldopen(
    joinpath(EXPERIMENT_ANALYSIS_SUMMARY_PATH, data_directory,
             data_directory * "_report_summary.jld2"),
    "r",
) do f
    f[JLD2_MyTools.make_keypath_exp(data_directory, 2, 0.005)]
end





data_directories = ["20260529", "20260603"]
data_directory = data_directories[1]
EXPERIMENT_PATH = joinpath(BASE_PATH, "EXPERIMENTS")
EXPERIMENT_ANALYSIS_SUMMARY_PATH = joinpath(BASE_PATH, "EXPDATA_ANALYSIS","summary")

exp_data = load(joinpath(EXPERIMENT_PATH, data_directory,"data.jld2"), "data")

current_field  = let M = hcat(exp_data[:SG1currentInA], exp_data[:SG1BfieldInTesla])
    M[sortperm(M[:, 1]), :]
end

exp_analysis = jldopen(joinpath(EXPERIMENT_ANALYSIS_SUMMARY_PATH, data_directory, data_directory * "_report_summary.jld2"), "r") do f
    f[JLD2_MyTools.make_keypath_exp(data_directory,2,0.005)]
end

Δz =  exp_analysis[:fw_F2_peak_pos_raw][1] .- exp_analysis[:fw_F1_peak_pos_raw][1]


