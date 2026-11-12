# =============================================================================
#  SG0_pattern_widths.jl
# -----------------------------------------------------------------------------
#  Kelvin Titimbo — California Institute of Technology — March 2026
#
#  PURPOSE
#    Measure the widths of the F1 / F2 atomic-beam patterns on the camera for
#    the single Stern–Gerlach (SG0) experiment, as a function of the SG0 coil
#    current (equivalently of the field gradient). This script will grow to
#    carry more tasks; at the moment it only covers data preparation and the
#    loading of one run for analysis.
#
#  PIPELINE
#    §1  Preamble        packages, plotting defaults, custom modules, B/G model
#    §2  Camera geometry binning, effective pixel size, spatial axes
#    §3  Run selection   list of acquisition directories (YYYYMMDD/RoundN)
#    §4  Data prep       per directory, cached on disk (nothing is rebuilt if
#                        the output file already exists):
#                            *.mat ──SG0_stack_data──▶ data.jld2
#                                  ──SG0_process_and_save(:simple)──▶ data_processed.jld2
#    §5  Analysis run    load the processed stack of one selected directory
#    §6  Peak positions  z_F1, z_F2, splitting Δz, centroid C₀ vs current
#                        (+ StartPosition = I=0 reference), two methods:
#        §6a             framewise: mean of per-frame spline maxima
#        §6b             mean profile: maximum of the frame-averaged profile
#        §6c             comparison 6b − 6a; output peaks_nz<nz>_<stamp>.jld2
#    §7  Figures         for every current: frame-averaged F1|F2 images with
#                        their profiles, and the z-profiles of all frames
#    §8  Pattern widths  HWHM along z on the §6 profiles/splines/maxima:
#                        every frame (mean ± SEM) and the mean profile
#                        (jackknife), raw and spline; widths_nz<nz>_<stamp>.jld2
#
#  FILES  (under BASE_PATH/EXPERIMENTS/<dir>/)
#    data.jld2            key "data" → OrderedDict, raw camera frames
#                           :F1_data, :F2_data, :BG_data :: (Nx, Nz, Nframes, Ncurrents)
#    data_processed.jld2  "meta/*" → scalars / per-current vectors (SI units)
#                         "data/F1ProcessedImages", "data/F2ProcessedImages"
#                           :: Array{Float32,4} (Nx, Nz, Nframes, Ncurrents)
#                           = F − BG, frame i paired with background frame i
#
#  CONVENTIONS
#    x : horizontal camera axis (binned ×4 → 540 px)
#    z : vertical camera axis = SG deflection direction (unbinned → 2560 px)
#    SI units internally (m, A, T, °C for the oven); mm/μm only for display.
# =============================================================================


# =============================================================================
# §1  PREAMBLE
# =============================================================================
cd(@__DIR__)                          # make relative paths (Modules/, outputs) predictable

# --- Plotting backend and defaults ---------------------------------------------
using Plots; gr()
using Plots.PlotMeasures
const IN_NOTEBOOK = isdefined(Main, :IJulia)
Plots.default(
    show=IN_NOTEBOOK, dpi=800, fontfamily="Computer Modern",
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
# Figures of THIS script use CairoMakie. It is imported under the alias `CM`
# (not `using`) because Plots and Makie export the same names (heatmap, lines,
# plot, ...). Plots stays loaded for the figures made inside MyExperimentalAnalysis.
import CairoMakie as CM
CM.activate!()
CM.set_theme!(CM.theme_latexfonts())  # Computer Modern, to match the Plots figures

# --- Data I/O and numerical tools ----------------------------------------------
using MAT, JLD2
using LinearAlgebra
using ImageFiltering, FFTW
using DataStructures
using Statistics, StatsBase, Distributions
using Interpolations, BSplineKit, Optim, Dierckx, LsqFit
using ProgressMeter

# --- Aesthetics and output formatting ------------------------------------------
using Colors, ColorSchemes
using Printf, LaTeXStrings, PrettyTables
using CSV, DataFrames, DelimitedFiles

# --- Time-stamping / logging ---------------------------------------------------
using Dates
using Alert

# --- Threads -------------------------------------------------------------------
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(2)
@info "BLAS threads"  count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()

# --- Custom modules ------------------------------------------------------------
include("./Modules/DataReading.jl")
include("./Modules/JLD2_MyTools.jl")
include("./Modules/TheoreticalSimulation.jl")
include("./Modules/MyExperimentalAnalysis.jl")
using .MyExperimentalAnalysis

# Field/gradient calibration used by BvsI, GvsI, IvsG (switches B and G together):
#   :manual       → apparatus manual (SG_BvsI.csv + built-in gradient table)
#   :calibration  → measured calibration (SG_BvsI_calibration.csv + SG_GvsI_calibration.csv)
TheoreticalSimulation.set_magnetic_field!(:calibration)
@info "Magnetic-field model" mode = TheoreticalSimulation.magnetic_field_mode()

# --- General setup -------------------------------------------------------------
const BASE_PATH = raw"F:\SternGerlachExperiments"
const HOSTNAME  = gethostname()
@info "Running on host" hostname = HOSTNAME

const T_START   = Dates.now()
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss")   # tag for output files

# Figure output (also pushed into the analysis module)
# (non-const on purpose: toggled between REPL re-runs)
FIG_EXT  = "png"     # "pdf", "svg", ...
SAVE_FIG = false
SAVE_TABLE = true   # write the results (peaks_*.jld2, widths_*.jld2) to the run's summary folder
MyExperimentalAnalysis.SAVE_FIG = SAVE_FIG
MyExperimentalAnalysis.FIG_EXT  = FIG_EXT


# =============================================================================
# §2  CAMERA GEOMETRY
# =============================================================================
# Intrinsic sensor properties
const CAM_PIXELSIZE        = 6.5e-6               # physical pixel pitch [m]
const NX_PIXELS, NZ_PIXELS = 2160, 2560           # sensor size (Nx, Nz) [px]

# Acquisition binning (x is binned, z — the deflection axis — is not)
const EXP_BIN_X, EXP_BIN_Z = 4, 1
const EXP_PIXELSIZE_X      = EXP_BIN_X * CAM_PIXELSIZE   # effective pixel size [m]
const EXP_PIXELSIZE_Z      = EXP_BIN_Z * CAM_PIXELSIZE

# Image dimensions after binning (÷ + check instead of Int(a/b) to fail loudly)
@assert NX_PIXELS % EXP_BIN_X == 0 && NZ_PIXELS % EXP_BIN_Z == 0 "Binning must divide sensor size"
const X_PIXELS = NX_PIXELS ÷ EXP_BIN_X            # 540
const Z_PIXELS = NZ_PIXELS ÷ EXP_BIN_Z            # 2560

# Pixel-centre coordinates [m], first pixel centred at pixelsize/2
const X_POSITION = MyExperimentalAnalysis.pixel_positions(X_PIXELS, 1, EXP_PIXELSIZE_X)
const Z_POSITION = MyExperimentalAnalysis.pixel_positions(Z_PIXELS, 1, EXP_PIXELSIZE_Z)

nz        = 1         # z-binning before the spline fit (must divide Z_PIXELS)
λ0        = 0.01      # spline smoothing parameter

# position
z_mm        = 1e3 .* MyExperimentalAnalysis.pixel_positions(Z_PIXELS, nz, EXP_PIXELSIZE_Z) 
z_mm_error  = 1/sqrt(12) * 1e3 * EXP_PIXELSIZE_Z * nz ;

println("""
***************************************************
CAMERA FEATURES
    Number of pixels        : $(NX_PIXELS) × $(NZ_PIXELS)
    Pixel size              : $(1e6*CAM_PIXELSIZE) μm

IMAGES INFORMATION
    Binning                 : $(EXP_BIN_X) × $(EXP_BIN_Z)
    Effective pixels        : $(X_PIXELS) × $(Z_PIXELS)
    Pixel size              : $(1e6*EXP_PIXELSIZE_X) μm × $(1e6*EXP_PIXELSIZE_Z) μm
    x range (pixel centres) : $(round(1e6*first(X_POSITION), digits=3)) μm → $(round(1e3*last(X_POSITION), digits=4)) mm
    z range (pixel centres) : $(round(1e6*first(Z_POSITION), digits=3)) μm → $(round(1e3*last(Z_POSITION), digits=4)) mm
***************************************************
""")

# Module-level defaults read by MyExperimentalAnalysis (e.g. SG0_mean_maxima)
MyExperimentalAnalysis.effective_cam_pixelsize_z = EXP_PIXELSIZE_Z
MyExperimentalAnalysis.DEFAULT_x_pixels          = X_PIXELS
MyExperimentalAnalysis.DEFAULT_z_pixels          = Z_PIXELS


# =============================================================================
# §3  RUN SELECTION
# =============================================================================
# Acquisition directories, relative to BASE_PATH/EXPERIMENTS.
# Comment/uncomment to choose which runs are prepared.
data_directories = [
    # "20260617/Round1", "20260617/Round2",
    # "20260618/Round1", "20260618/Round2",
    # "20260622/Round1", "20260622/Round2",
    # "20260623/Round1", "20260623/Round2",
    # "20260624/Round1", "20260624/Round2",
    # "20260626/Round1", "20260626/Round2",
    "20260629/Round1", "20260629/Round2",
    "20260701/Round1", "20260701/Round2",
    "20260702/Round1", "20260702/Round2",
    "20260706/Round1", "20260706/Round2",
    "20260709/Round1", "20260709/Round2",
    "20260710/Round1", "20260710/Round2", "20260710/Round3",
    # "20260806",
    # "20260807/Round1",
    # "20260923/Round1", "20260923/Round2",
    # "20261009/Round1", "20261009/Round2",
]
nd = length(data_directories)


# =============================================================================
# §4  DATA PREPARATION
# =============================================================================
# --- Path helpers --------------------------------------------------------------
"Folder holding the .mat files and the cached .jld2 files of a run."
experiment_dir(dir::AbstractString) = joinpath(BASE_PATH, "EXPERIMENTS", dir)
"Raw stacked frames (output of `SG0_stack_data`)."
raw_file(dir::AbstractString)       = joinpath(experiment_dir(dir), "data.jld2")
"Background-subtracted frames (output of `SG0_process_and_save`)."
processed_file(dir::AbstractString) = joinpath(experiment_dir(dir), "data_processed.jld2")
"Output folder for summaries/figures of a run (created on demand)."
summary_dir(dir::AbstractString)    = mkpath(joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS", "summary", dir))

"""
    prepare_directory(dir; force=false, mode=:simple) -> String

Make sure `data_processed.jld2` exists for the run `dir` and return its path.

Steps, each skipped when its output file already exists (unless `force=true`):
1. Stack all `.mat` files of the run into `data.jld2` (`SG0_stack_data`).
2. Background-subtract F1/F2 and write `data_processed.jld2`
   (`SG0_process_and_save`; `mode=:simple` pairs frame i with BG frame i,
   `mode=:permuted` uses every (F, BG) frame pair).

The raw dictionary is built (or read) once and reused for step 2; it is local
to this function, so its memory is released when the function returns.
"""
function prepare_directory(dir::AbstractString; force::Bool=false, mode::Symbol=:simple)
    f_raw, f_proc = raw_file(dir), processed_file(dir)

    if isfile(f_proc) && !force
        @info "Found $f_proc → skipping build"
        return f_proc
    end

    data_raw = if isfile(f_raw) && !force
        @info "Found $f_raw → loading"
        load(f_raw, "data")
    else
        @info "Building $f_raw"
        d = MyExperimentalAnalysis.SG0_stack_data("new", experiment_dir(dir))
        jldsave(f_raw; data = d)
        d
    end

    MyExperimentalAnalysis.SG0_process_and_save(data_raw, f_proc; mode = mode)
    return f_proc
end

# --- Run over all selected directories; one failure does not stop the batch ----
prep_failures = OrderedDict{String,String}()
for dir in data_directories
    printstyled("\t$dir\n"; color = :cyan, bold = true)
    try
        prepare_directory(dir)
        summary_dir(dir)
        println("\n$dir PROCESSING COMPLETED\n")
    catch e
        prep_failures[dir] = sprint(showerror, e)
        @error "Preparation failed" dir exception = (e, catch_backtrace())
    end
    GC.gc()                                   # raw stacks are several GB
end
isempty(prep_failures) ||
    @warn "$(length(prep_failures)) of $nd directories failed" dirs = collect(keys(prep_failures))


# =============================================================================
# §5  LOAD ONE RUN FOR ANALYSIS
# =============================================================================
"""
    load_processed(dir; images=true) -> Dict{Symbol,Any}

Read `data_processed.jld2` of run `dir`. Keys use the names expected by the
`MyExperimentalAnalysis` routines (e.g. `:SG0Currents`, `:F1ProcessedImages`).

| key                     | content                                  | units |
|:------------------------|:-----------------------------------------|:------|
| `:Directory`            | source folder of the .mat files          |       |
| `:TemperatureInCelsius` | oven temperature (scalar or per current) | °C    |
| `:AcquisitionStep`      | acquisition order of each current        |       |
| `:MGcurrentInA`         | guiding-magnet current                   | A     |
| `:MGfieldInTesla`       | guiding-magnet field                     | T     |
| `:SG0Currents`, `:SG0Bz`| SG0 coil current / field                 | A, T  |
| `:SG1Currents`, `:SG1Bz`| SG1 coil current / field                 | A, T  |
| `:F1ProcessedImages`    | F1 − BG, (Nx, Nz, Nframes, Ncurrents)    | counts|
| `:F2ProcessedImages`    | F2 − BG, (Nx, Nz, Nframes, Ncurrents)    | counts|

With `images=false` only the metadata are read (fast). Each image stack is
Float32, 540 × 2560 × 4 B ≈ 5.5 MB per frame, i.e. Nframes·Ncurrents × 5.5 MB
per state, so both stacks of a run can reach a few GB.
"""
function load_processed(dir::AbstractString; images::Bool=true)
    path = processed_file(dir)
    isfile(path) || error("Processed file not found: $path (run prepare_directory first)")
    jldopen(path, "r") do f
        d = Dict{Symbol,Any}(
            :Directory            => f["meta/Directory"],
            :TemperatureInCelsius => f["meta/TemperatureInCelsius"],
            :AcquisitionStep      => f["meta/AcquisitionStep"],
            :MGcurrentInA         => f["meta/MGcurrentInA"],
            :MGfieldInTesla       => f["meta/MGfieldInTesla"],
            :SG0Currents          => f["meta/SG0currentInA"],
            :SG0Bz                => f["meta/SG0BfieldInTesla"],
            :SG1Currents          => f["meta/SG1currentInA"],
            :SG1Bz                => f["meta/SG1BfieldInTesla"],
        )
        if images
            d[:F1ProcessedImages] = f["data/F1ProcessedImages"]
            d[:F2ProcessedImages] = f["data/F2ProcessedImages"]
        end
        d
    end
end

# Select the run by name (robust to reordering/commenting of the list above)
ANALYSIS_DIR = "20260706/Round2"
@assert ANALYSIS_DIR in data_directories "ANALYSIS_DIR=$ANALYSIS_DIR is not in data_directories"

data_processed = load_processed(ANALYSIS_DIR)
let F1 = data_processed[:F1ProcessedImages]
    nx, nz, nf, nI = size(F1)
    @info "Loaded $ANALYSIS_DIR" Nx = nx Nz = nz Nframes = nf Ncurrents = nI I_SG0 = data_processed[:SG0Currents]
end


# =============================================================================
# §6  PEAK POSITIONS, SPLITTING AND CENTROID
# =============================================================================
# The F1 and F2 peak positions along z are obtained in two ways, each from
# the x-averaged z-profile fitted with a cubic smoothing spline (nz, λ0 of §2):
#
#   §6a framewise : z_F = ⟨ argmax S_frame ⟩_frames      "mean of the maxima"
#                   (SG0_framewise_maxima; 5-px median along x first)
#   §6b mean      : z_F =   argmax S_⟨frame⟩             "maximum of the mean"
#                   (SG0_mean_maxima; no median filter)
#   §6c           : comparison of the two, and the JLD2 output file
#
# From z_F1, z_F2 (F1 and F2 treated as independent):
#   Δz  = |z_F1 − z_F2|,          σ_Δz = √(σ_F1² + σ_F2²)
#   C₀  = (z_F1 + z_F2)/2,        σ_C₀ = σ_Δz/2
# σ_pix = Δz_bin/√12 is the uniform position uncertainty within one z-bin.
# "StartPosition" = mean of the rows with I_SG0 = I_SG1 = 0 (field-free
# reference points of the scan), see `zero_current_reference`.

HALF_MAX  = false     # true → fit only the points above half maximum

# --- Shared tools ---------------------------------------------------------------
"""
    nanmean_sem(M) -> (μ, sem)

Column-wise mean and standard error of the mean of `M :: (Nframes, Ncurrents)`,
ignoring `NaN` entries (`SG0_framewise_maxima` pads with `NaN` when a current
has fewer frames). `sem` is `NaN` when a column has fewer than two values.
"""
function nanmean_sem(M::AbstractMatrix)
    μ   = fill(NaN, size(M, 2))
    sem = fill(NaN, size(M, 2))
    for j in axes(M, 2)
        v = filter(!isnan, view(M, :, j))
        n = length(v)
        n ≥ 1 && (μ[j]   = mean(v))
        n ≥ 2 && (sem[j] = std(v; corrected = true) / sqrt(n))
    end
    return μ, sem
end

"Return `v` as a vector of length `n` (scalars in the metadata are repeated)."
per_current(v, n::Int) = v isa AbstractArray ? vec(v) : fill(v, n)

"σ_pix [mm] for a z-binning of `nz` camera pixels."
sigma_pix_mm(nz::Integer) = 1e3 / sqrt(12) * EXP_PIXELSIZE_Z * nz

"""
    control_table(data) -> DataFrame

One row per current with the control parameters of the scan:
`I0, I1, IG` [A] (absolute values of the SG currents), `B0, B1, BG` [mT].
Shared by the §6 peak tables and the §8 width tables.
"""
function control_table(data)
    nI = length(data[:SG0Currents])
    return DataFrame(
        I0 = abs.(per_current(data[:SG0Currents], nI)),
        I1 = abs.(per_current(data[:SG1Currents], nI)),
        IG = per_current(data[:MGcurrentInA],     nI),
        B0 = 1e3 .* per_current(data[:SG0Bz],          nI),   # T → mT
        B1 = 1e3 .* per_current(data[:SG1Bz],          nI),
        BG = 1e3 .* per_current(data[:MGfieldInTesla], nI),
    )
end

"""
    build_peak_table(data, zf1, errzf1, zf2, errzf2) -> DataFrame

One row per current with the control parameters and the peak results:
`I0, I1, IG` [A] (absolute values), `B0, B1, BG` [mT],
`zf1, errzf1, zf2, errzf2, split, errsplit, c0, errc0` [mm] (full precision).
"""
function build_peak_table(data, zf1, errzf1, zf2, errzf2)
    errsplit = hypot.(errzf1, errzf2)
    return hcat(control_table(data), DataFrame(
        zf1      = zf1,
        errzf1   = errzf1,
        zf2      = zf2,
        errzf2   = errzf2,
        split    = abs.(zf1 .- zf2),
        errsplit = errsplit,
        c0       = (zf1 .+ zf2) ./ 2,
        errc0    = errsplit ./ 2,
    ))
end

"""
    zero_current_reference(table; tol=1e-9) -> DataFrame

One-row summary of the rows with `I0 ≈ 0` and `I1 ≈ 0` (the StartPosition).
Value columns are averaged, frame counts (`nfr*`) are summed (frames used in
total); the uncertainty columns (`err*`) are combined as
the error of a mean of independent points, `√(Σσᵢ²)/n`. The σ_pix part of
each error is systematic and does not really average down, so this is a lower
bound when σ_pix dominates. Returns an empty `DataFrame` (with a warning) if
there are no zero-current rows.
"""
function zero_current_reference(table::DataFrame; tol::Real = 1e-9)
    rows = table[(abs.(table.I0) .<= tol) .& (abs.(table.I1) .<= tol), :]
    n    = nrow(rows)
    n == 0 && (@warn "No zero-current rows found"; return similar(table, 0))
    reduce_col(col, name) = startswith(name, "err") ? sqrt(sum(abs2, col)) / n :
                            startswith(name, "nfr") ? sum(Int, col) : mean(col)
    return DataFrame([name => [reduce_col(rows[!, name], name)] for name in names(rows)])
end

"Common PrettyTables styling of all §6 tables."
pretty_peak_style() = (
    alignment     = :c,
    table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
    style         = TextTableStyle(
        first_line_column_label = crayon"yellow bold",
        column_label            = crayon"yellow",
        table_border            = crayon"blue bold",
        title                   = crayon"bold red",
    ),
    equal_data_column_widths          = true,
    fit_table_in_display_horizontally = false,   # never drop columns in narrow terminals
    fit_table_in_display_vertically   = false,   # never drop rows
)

const PEAK_TABLE_LABELS = [
    ["I0 Current", "I1 Current", "IG Current", "B0 field", "B1 field", "BG field",
     "F1", "F1 err", "F2", "F2 err", "Δz", "Δz err", "C₀", "C₀ err"],
    ["[A]", "[A]", "[A]", "[mT]", "[mT]", "[mT]",
     "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"],
]

"Pretty-print a table produced by `build_peak_table` / `zero_current_reference`."
show_peak_table(df::DataFrame; title::AbstractString = "") =
    pretty_table(df; title, column_labels = PEAK_TABLE_LABELS,
        formatters = [fmt__printf("%8.5f", 1:3),              # currents
                      fmt__printf("%8.3f", 4:6),              # fields
                      fmt__printf("%8.3f", [7, 9, 11, 13]),   # positions
                      fmt__printf("%8.4f", [8, 10, 12, 14])], # uncertainties
        pretty_peak_style()...)

T_oven = data_processed[:TemperatureInCelsius]


# -----------------------------------------------------------------------------
# §6a  FRAMEWISE: MEAN OF THE PER-FRAME MAXIMA
# -----------------------------------------------------------------------------
#   z_F = ⟨z_max⟩_frames,   σ_F = √( SEM_frames² + σ_pix² )
# σ_pix is added to each state; since both are read on the same pixel grid
# this is conservative for Δz and C₀.

"""
    peak_table(data; nz=1, λ0=0.01, half_max=false) -> (; table, zero, f1_max, f2_max, σ_pix)

§6a for a processed run: `table` (see `build_peak_table`), its zero-current
row `zero`, the framewise peak positions `f1_max`, `f2_max` [mm]
`(Nframes, Ncurrents)` and the σ_pix used [mm].
"""
function peak_table(data; nz::Integer = 1, λ0::Real = 0.01, half_max::Bool = false)
    σ_pix  = sigma_pix_mm(nz)
    f1_max = MyExperimentalAnalysis.SG0_framewise_maxima("F1", data, nz; half_max, λ0 = Float64(λ0))
    f2_max = MyExperimentalAnalysis.SG0_framewise_maxima("F2", data, nz; half_max, λ0 = Float64(λ0))
    zf1, sem1 = nanmean_sem(f1_max)
    zf2, sem2 = nanmean_sem(f2_max)
    table = build_peak_table(data, zf1, hypot.(sem1, σ_pix), zf2, hypot.(sem2, σ_pix))
    return (; table, zero = zero_current_reference(table), f1_max, f2_max, σ_pix)
end

peaks     = peak_table(data_processed; nz, λ0, half_max = HALF_MAX)
peak_data = peaks.table          # kept for §8
peak_zero = peaks.zero

show_peak_table(peak_data; title = "$ANALYSIS_DIR  T = $(T_oven) °C — FRAMEWISE (nz = $nz, λ0 = $λ0)")
nrow(peak_zero) > 0 && show_peak_table(peak_zero; title = "$ANALYSIS_DIR — StartPosition, framewise")


# -----------------------------------------------------------------------------
# §6b  MEAN PROFILE: MAXIMUM OF THE FRAME-AVERAGED PROFILE
# -----------------------------------------------------------------------------
# For each current the stack is averaged over x AND over all frames first; the
# peak is the tallest maximum of the spline fit to that single profile.
# One profile per current → no frame scatter: err = σ_pix only. This is a
# LOWER BOUND; the SEM of §6a is the realistic statistical error.

"""
    mean_peak_table(data; nz=1, λ0=0.01, half_max=false) -> (; table, zero, f1, f2, σ_pix)

§6b for a processed run. `table`/`zero` as in [`peak_table`](@ref) (errors =
σ_pix only); `f1`, `f2` are the full outputs of `SG0_mean_maxima`
(`peak_positions`, `mean_profile`, `mean_bin_profile` = `[z_mm  y₁ … y_nI]`).
"""
function mean_peak_table(data; nz::Integer = 1, λ0::Real = 0.01, half_max::Bool = false)
    σ_pix = sigma_pix_mm(nz)
    f1  = MyExperimentalAnalysis.SG0_mean_maxima("F1", data, nz; half_max, λ0 = Float64(λ0))
    f2  = MyExperimentalAnalysis.SG0_mean_maxima("F2", data, nz; half_max, λ0 = Float64(λ0))
    err = fill(σ_pix, length(f1.peak_positions))
    table = build_peak_table(data, f1.peak_positions, err, f2.peak_positions, err)
    return (; table, zero = zero_current_reference(table), f1, f2, σ_pix)
end

peaks_mean     = mean_peak_table(data_processed; nz, λ0, half_max = HALF_MAX)
peak_data_mean = peaks_mean.table
peak_zero_mean = peaks_mean.zero

show_peak_table(peak_data_mean;
    title = "$ANALYSIS_DIR  T = $(T_oven) °C — MEAN PROFILE (nz = $nz, λ0 = $λ0; err = σ_pix only)")
nrow(peak_zero_mean) > 0 &&
    show_peak_table(peak_zero_mean; title = "$ANALYSIS_DIR — StartPosition, mean profile")


# -----------------------------------------------------------------------------
# §6c  COMPARISON AND OUTPUT FILE
# -----------------------------------------------------------------------------
# Differences to expect between §6a and §6b:
# * §6a applies a 5-px median along x (hot-pixel removal), §6b does not.
# * Symmetric lobes: both agree on average. Asymmetric lobes with frame-to-
#   frame jitter: averaging blurs the profile and moves its maximum toward the
#   broad side (synthetic test: +1…4 μm for σ_L/σ_R = 0.25/0.40 mm, 50 μm jitter).
# * Noise.

"""
    compare_peak_tables(A, B) -> DataFrame

Difference B − A of two peak tables (same currents, same order), in μm, for
z_F1, z_F2, Δz and C₀, and the same differences in units of A's uncertainty
(`n…` columns). With A = §6a and B = §6b, |n| ≲ 1 means the two methods agree
within the framewise error.
"""
function compare_peak_tables(A::DataFrame, B::DataFrame)
    A.I0 ≈ B.I0 || error("Tables do not have the same currents")
    d(col)       = 1e3 .* (B[!, col] .- A[!, col])          # mm → μm
    n(col, ecol) = (B[!, col] .- A[!, col]) ./ A[!, ecol]
    return DataFrame(
        I0     = A.I0,
        B0     = A.B0,
        dzf1   = d(:zf1),   dzf2   = d(:zf2),
        dsplit = d(:split), dc0    = d(:c0),
        nzf1   = n(:zf1, :errzf1),     nzf2 = n(:zf2, :errzf2),
        nsplit = n(:split, :errsplit), nc0  = n(:c0, :errc0),
    )
end

"Pretty-print the output of `compare_peak_tables`."
show_peak_comparison(df::DataFrame; title::AbstractString = "") =
    pretty_table(df; title,
        column_labels = [
            ["I0 Current", "B0 field", "δF1", "δF2", "δΔz", "δC₀",
             "δF1/σ", "δF2/σ", "δΔz/σ", "δC₀/σ"],
            ["[A]", "[mT]", "[μm]", "[μm]", "[μm]", "[μm]", "", "", "", ""],
        ],
        formatters = [fmt__printf("%8.5f", [1]), fmt__printf("%8.3f", [2]),
                      fmt__printf("%8.2f", 3:6), fmt__printf("%6.2f", 7:10)],
        pretty_peak_style()...)

peak_compare = compare_peak_tables(peak_data, peak_data_mean)
show_peak_comparison(peak_compare;
    title = "$ANALYSIS_DIR — mean profile − framewise  (σ = framewise uncertainty)")

# --- Output file ---------------------------------------------------------------
# peaks_nz<nz>_<RUN_STAMP>.jld2 in the run's summary folder. Units: currents in A
# (signed in meta/, absolute values in StartPosition/), magnetic fields in T,
# positions in mm (*_mm). The §6c comparison is only printed (it can be
# recomputed from framewise/ and meanprofile/). Layout:
#
#   meta/        Directory, TemperatureInCelsius, CurrentSG0, MagneticFieldSG0,
#                CurrentSG1, MagneticFieldSG1, CurrentGuide, MagneticFieldGuide, Units
#   framewise/   F1_mm, ErrF1_mm, F2_mm, ErrF2_mm, Split_mm, ErrSplit_mm,
#                Centroid_mm, ErrCentroid_mm                 (vectors, one per current)
#                StartPosition/<the 6 control keys of meta/ + the 8 keys above>  (scalars)
#                PerFrame/F1_mm, PerFrame/F2_mm              (Nframes × Ncurrents)
#   meanprofile/ same keys as framewise/ (+ StartPosition/), and
#                Profile/z_mm, Profile/F1, Profile/F2        (mean z-profiles, Nz_bin × Ncurrents)
#   settings/    AnalysisDirectory, nz, lambda0, HalfMax, SigmaPix_mm, RunStamp
#
# A missing StartPosition (no zero-current rows) is stored as NaN.

const PEAK_KEYS = (zf1 = "F1_mm", errzf1 = "ErrF1_mm", zf2 = "F2_mm", errzf2 = "ErrF2_mm",
                   split = "Split_mm", errsplit = "ErrSplit_mm",
                   c0 = "Centroid_mm", errc0 = "ErrCentroid_mm")
const CONTROL_KEYS = (I0 = "CurrentSG0", B0 = "MagneticFieldSG0",
                      I1 = "CurrentSG1", B1 = "MagneticFieldSG1",
                      IG = "CurrentGuide", BG = "MagneticFieldGuide")

"""
Write one result group (e.g. `framewise`): per-current vectors + StartPosition scalars.
`keys` maps table columns to file keys (§6: `PEAK_KEYS`, §8: `WIDTH_KEYS`).
"""
function _write_peak_group!(f, group::AbstractString, table::DataFrame, zero::DataFrame;
                            keys::NamedTuple = PEAK_KEYS)
    has0 = nrow(zero) == 1
    for (col, key) in pairs(keys)
        f["$group/$key"] = Vector{Float64}(table[!, col])
        f["$group/StartPosition/$key"] = has0 ? Float64(zero[1, col]) : NaN
    end
    for (col, key) in pairs(CONTROL_KEYS)
        v = has0 ? Float64(zero[1, col]) : NaN
        f["$group/StartPosition/$key"] = startswith(key, "MagneticField") ? 1e-3 * v : v  # mT → T
    end
    return f
end

"""
    save_peak_results(path, data, fw, mp; nz, λ0, half_max) -> path

Write the §6 results to `path` with the layout described above. `fw` and `mp`
are the outputs of [`peak_table`](@ref) and [`mean_peak_table`](@ref).
"""
function save_peak_results(path::AbstractString, data, fw, mp;
                           nz::Integer, λ0::Real, half_max::Bool)
    jldopen(path, "w") do f
        _write_meta!(f, data)

        # §6a
        _write_peak_group!(f, "framewise", fw.table, fw.zero)
        f["framewise/PerFrame/F1_mm"] = fw.f1_max
        f["framewise/PerFrame/F2_mm"] = fw.f2_max

        # §6b
        _write_peak_group!(f, "meanprofile", mp.table, mp.zero)
        f["meanprofile/Profile/z_mm"] = mp.f1.mean_bin_profile[:, 1]
        f["meanprofile/Profile/F1"]   = mp.f1.mean_bin_profile[:, 2:end]
        f["meanprofile/Profile/F2"]   = mp.f2.mean_bin_profile[:, 2:end]

        # analysis settings (needed to reproduce the numbers)
        f["settings/AnalysisDirectory"] = ANALYSIS_DIR
        f["settings/nz"]                = Int(nz)
        f["settings/lambda0"]           = Float64(λ0)
        f["settings/HalfMax"]           = half_max
        f["settings/SigmaPix_mm"]       = fw.σ_pix
        f["settings/RunStamp"]          = RUN_STAMP
    end
    return path
end

"`meta/` group shared by the §6 and §8 files: control parameters of the scan (SI, signed)."
function _write_meta!(f, data)
    nI = length(data[:SG0Currents])
    f["meta/Directory"]            = String(data[:Directory])
    f["meta/TemperatureInCelsius"] = data[:TemperatureInCelsius]
    f["meta/CurrentSG0"]           = Float64.(per_current(data[:SG0Currents],    nI))
    f["meta/MagneticFieldSG0"]     = Float64.(per_current(data[:SG0Bz],          nI))
    f["meta/CurrentSG1"]           = Float64.(per_current(data[:SG1Currents],    nI))
    f["meta/MagneticFieldSG1"]     = Float64.(per_current(data[:SG1Bz],          nI))
    f["meta/CurrentGuide"]         = Float64.(per_current(data[:MGcurrentInA],   nI))
    f["meta/MagneticFieldGuide"]   = Float64.(per_current(data[:MGfieldInTesla], nI))
    f["meta/Units"] = Dict("Current*" => "A", "MagneticField*" => "T",
                           "*_mm" => "mm", "TemperatureInCelsius" => "°C")
    return f
end

if SAVE_TABLE
    peaks_file = joinpath(summary_dir(ANALYSIS_DIR), "peaks_nz$(nz)_$(RUN_STAMP).jld2")
    save_peak_results(peaks_file, data_processed, peaks, peaks_mean;
                      nz, λ0, half_max = HALF_MAX)
    @info "Saved peak results" peaks_file
    JLD2_MyTools.list_keys_jld_peaks(peaks_file)     # print the saved layout
end

# =============================================================================
# §7  FRAME-AVERAGED F1 | F2 IMAGES AND PROFILES FOR EVERY CURRENT
# =============================================================================
# One figure per current. F1 occupies columns 1–2, F2 columns 3–4:
#
#        col 1 (3)          col 2 (4)
#   r1 ┌──────────────┐
#      │  ⟨I⟩_z (x)   │                     mean image averaged over z
#   r2 ├──────────────┼──────────┐
#      │  mean image  │ ⟨I⟩_x(z) │          mean over frames; ⟨I⟩_x(z) = §6b profile
#   r3 ├──────────────┤          │
#      │  colour bar  │                     horizontal, under the image
#   r4 ├──────────────┴──────────┐
#      │  ⟨I⟩_x(z) of every frame │          one colour per frame + mean (dashed)
#      └─────────────────────────┘          + §6a peak of each frame (dotted)
#
# All averages are plain means (no median filter, no baseline removal), so the
# dashed mean in row 4 equals the ⟨I⟩_x(z) panel of row 2. Coordinates are
# stored in metres and shown in mm.

PLOT_MEAN_ALL = true      # false → skip (one figure per current)

"""
    _state_block!(fig, col, stack, x_mm, z_mm; zmax=nothing, label="",
                  colorrange=automatic, colorbar=true, zticks=true, zwindow=2.0)
        -> NamedTuple

Draw the four rows of one state (`stack :: (Nx, Nz, Nframes)`, one current) in
columns `col`, `col+1` of `fig`: row 1 ⟨I⟩_z(x), row 2 mean image + ⟨I⟩_x(z),
row 3 horizontal colour bar (if `colorbar`), row 4 all frames' z-profiles,
zoomed to ±`zwindow` mm around the peak of the mean (`nothing` → full camera).
`zmax` = peak position of each frame [mm] (NaN allowed), drawn as dotted lines.
Returns `(; hm, ax_x, ax_im, ax_z, ax_f)` for linking axes.
"""
function _state_block!(fig, col::Int, stack::AbstractArray{<:Real,3}, x_mm, z_mm;
                       zmax = nothing, label::AbstractString = "",
                       colorrange = CM.Makie.automatic, colorbar::Bool = true,
                       zticks::Bool = true, zwindow = 2.0)
    size(stack)[1:2] == (length(x_mm), length(z_mm)) ||
        throw(DimensionMismatch("stack is $(size(stack)); expected ($(length(x_mm)), $(length(z_mm)), Nframes)"))

    img    = dropdims(mean(stack; dims = 3); dims = 3)        # mean image, Nx × Nz
    P      = dropdims(mean(stack; dims = 1); dims = 1)        # frame z-profiles, Nz × Nframes
    prof_z = vec(mean(P; dims = 2))                           # = mean over x of img
    prof_x = vec(mean(img; dims = 2))
    nf     = size(P, 2)
    cols   = CM.Makie.resample_cmap(:viridis, max(nf, 2))[1:nf]

    # rows 1–2: mean image and its two profiles
    ax_x  = CM.Axis(fig[1, col]; ylabel = L"\langle I \rangle_z", title = label,
                    xticklabelsvisible = false)
    ax_im = CM.Axis(fig[2, col]; xlabel = L"x \; (\mathrm{mm})",
                    ylabel = zticks ? L"z \; (\mathrm{mm})" : "", yticklabelsvisible = zticks)
    ax_z  = CM.Axis(fig[2, col + 1]; xlabel = L"\langle I \rangle_x",
                    yticklabelsvisible = false, xticklabelrotation = π/4)
    hm = CM.heatmap!(ax_im, x_mm, z_mm, img; colormap = :inferno, colorrange)
    CM.lines!(ax_x, x_mm, prof_x; color = :black, linewidth = 1)
    CM.lines!(ax_z, prof_z, z_mm; color = :black, linewidth = 1)
    CM.linkxaxes!(ax_im, ax_x)
    CM.linkyaxes!(ax_im, ax_z)
    CM.limits!(ax_im, extrema(x_mm)..., extrema(z_mm)...)

    # row 3: colour bar under the image (ticks and label below the bar)
    colorbar && CM.Colorbar(fig[3, col], hm; label = "counts (F − BG)",
                            vertical = false, flipaxis = false, height = 12)

    # row 4: every frame's z-profile, their mean, and the §6a peak of each frame
    ax_f = CM.Axis(fig[4, col:col+1]; xlabel = L"z \; (\mathrm{mm})",
                   ylabel = L"\langle I \rangle_x", title = "$label — all frames")
    for i in 1:nf
        CM.lines!(ax_f, z_mm, view(P, :, i); color = cols[i], linewidth = 0.9, label = string(i))
    end
    CM.lines!(ax_f, z_mm, prof_z; color = :black, linewidth = 1.6, linestyle = :dash, label = "mean")
    if !isnothing(zmax)
        ok = findall(!isnan, collect(zmax))
        isempty(ok) || CM.vlines!(ax_f, collect(zmax)[ok]; color = cols[ok], linestyle = :dot, linewidth = 0.9)
    end
    if !isnothing(zwindow)
        zc = z_mm[argmax(prof_z)]
        CM.xlims!(ax_f, max(first(z_mm), zc - zwindow), min(last(z_mm), zc + zwindow))
    end
    CM.axislegend(ax_f; position = :rt, nbanks = 2, labelsize = 10, patchsize = (14, 6),
                  rowgap = 0, colgap = 6, padding = (4, 4, 2, 2), framevisible = false)

    return (; hm, ax_x, ax_im, ax_z, ax_f)
end

"""
    plot_mean_F1F2(data, j; peaks=nothing, common_scale=false, zwindow=2.0,
                   x=X_POSITION, z=Z_POSITION) -> Figure

Figure of current index `j` (layout at the top of §7), F1 left and F2 right.
`peaks` = §6a output (`peaks.f1_max`, `peaks.f2_max`) to mark each frame's peak.

* `common_scale=false` : independent colour ranges, one colour bar under each
  image, independent intensity axes in the frame row — best to see shapes.
* `common_scale=true`  : one colour range and one colour bar for both images,
  shared intensity axes in the frame row — compares F1/F2 brightness.

Image axes and the row-1/row-2 profile axes of F1 and F2 are always linked
(zoom one, zoom all). The frame-row z windows are not linked: each is centred
on its own peak.
"""
function plot_mean_F1F2(data, j::Integer; peaks = nothing, common_scale::Bool = false,
                        zwindow = 2.0, x::AbstractVector = X_POSITION, z::AbstractVector = Z_POSITION)
    S1 = view(data[:F1ProcessedImages], :, :, :, j)
    S2 = view(data[:F2ProcessedImages], :, :, :, j)
    x_mm, z_mm = 1e3 .* x, 1e3 .* z
    nf, nI = size(S1, 3), size(data[:F1ProcessedImages], 4)
    I_SG0  = per_current(data[:SG0Currents], nI)[j]
    B_SG0  = per_current(data[:SG0Bz], nI)[j]
    zmax(M) = isnothing(peaks) ? nothing : view(M, :, j)

    crange = CM.Makie.automatic
    if common_scale
        m1 = extrema(mean(S1; dims = 3)); m2 = extrema(mean(S2; dims = 3))
        crange = (min(m1[1], m2[1]), max(m1[2], m2[2]))
    end

    fig = CM.Figure(size = (1250, 1180))
    b1 = _state_block!(fig, 1, S1, x_mm, z_mm; label = "F1", colorrange = crange, zwindow,
                       zmax = isnothing(peaks) ? nothing : zmax(peaks.f1_max),
                       colorbar = !common_scale)
    b2 = _state_block!(fig, 3, S2, x_mm, z_mm; label = "F2", colorrange = crange, zwindow,
                       zmax = isnothing(peaks) ? nothing : zmax(peaks.f2_max),
                       colorbar = !common_scale, zticks = !common_scale)
    common_scale && CM.Colorbar(fig[3, 1:3], b1.hm; label = "counts (F − BG)",
                                vertical = false, flipaxis = false, height = 12)

    # link: same geometry everywhere; same intensity scale in the profile panels
    CM.linkxaxes!(b1.ax_im, b2.ax_im, b1.ax_x, b2.ax_x)
    CM.linkyaxes!(b1.ax_im, b2.ax_im, b1.ax_z, b2.ax_z)
    CM.linkyaxes!(b1.ax_x, b2.ax_x)
    CM.linkxaxes!(b1.ax_z, b2.ax_z)
    common_scale && CM.linkyaxes!(b1.ax_f, b2.ax_f)

    # sizes and gaps (set BEFORE the title, which inserts a row at the top)
    CM.rowsize!(fig.layout, 1, CM.Relative(0.12))
    CM.rowsize!(fig.layout, 4, CM.Relative(0.24))
    CM.colsize!(fig.layout, 2, CM.Relative(0.13))
    CM.colsize!(fig.layout, 4, CM.Relative(0.13))
    CM.rowgap!(fig.layout, 1, 6); CM.rowgap!(fig.layout, 2, 10); CM.rowgap!(fig.layout, 3, 18)
    CM.colgap!(fig.layout, 1, 6); CM.colgap!(fig.layout, 2, 30); CM.colgap!(fig.layout, 3, 6)

    CM.Label(fig[0, :], @sprintf("%s : mean of %d frames — current %d/%d:  I_SG0 = %.4f A,  B_SG0 = %.2f mT",
                                 ANALYSIS_DIR, nf, j, nI, I_SG0, 1e3 * B_SG0); fontsize = 16)
    return fig
end

"""
    plot_mean_F1F2_all(data; peaks=nothing, currents=all, common_scale=false,
                       zwindow=2.0, show=true, save=SAVE_FIG) -> Vector{String}

[`plot_mean_F1F2`](@ref) for every current index in `currents`. Each figure is
displayed (`show`) and/or saved in the run's summary folder as
`F1F2_mean_I<jj>_<RUN_STAMP>.<FIG_EXT>` (`save`); figures are not kept in
memory. Returns the paths of the saved files.
"""
function plot_mean_F1F2_all(data; peaks = nothing,
                            currents = axes(data[:F1ProcessedImages], 4),
                            common_scale::Bool = false, zwindow = 2.0,
                            show::Bool = true, save::Bool = SAVE_FIG)
    saved = String[]
    for j in currents
        fig = plot_mean_F1F2(data, j; peaks, common_scale, zwindow)
        show && display(fig)
        if save
            fname = joinpath(summary_dir(ANALYSIS_DIR), @sprintf("F1F2_mean_I%02d_%s.%s", j, RUN_STAMP, FIG_EXT))
            CM.save(fname, fig; px_per_unit = 4)
            push!(saved, fname)
        end
    end
    isempty(saved) || @info "Saved $(length(saved)) frame-averaged F1|F2 figures" folder = summary_dir(ANALYSIS_DIR)
    return saved
end

PLOT_MEAN_ALL && plot_mean_F1F2_all(data_processed; peaks,       # §6a peaks per frame
                                    common_scale = false,        # independent colour bars
                                    zwindow = 2.0)               # ±2 mm around the peak in the frame row


# =============================================================================
# §8  PATTERN WIDTHS ALONG z  (HWHM)
# =============================================================================
# Width of the pattern along the deflection axis z, measured on EXACTLY the
# profiles (same nz) and maxima (same λ0) that gave the peak positions of §6.
# The width itself is read on a second, lighter spline of the same profile,
# smoothing λ_w = WIDTH_λW, because one λ cannot serve both purposes
# (λ0 scan, 20260706/Round1, 3 currents × F1/F2, all frames + mean profile):
#
#      λ                1e-4     1e-3     3e-3     1e-2     3e-2     1e-1
#      spline HWHM     +0.5–0.9  +1.3–1.8 +2.6–3.1 +6.9–7.8  +17      +38   %
#      SEM z_max        2–16     1–3      0.2–1.4  0.2–0.7  0.2–0.5  0.2–0.4 μm
#
#   flat-topped lobes: a light spline follows the top → z_max wanders across
#   it from frame to frame; a stiff spline rounds the top → the width grows.
#   Hence z_max from λ0 = 0.01 (§6), widths from λ_w = 1e-3 (≈ +1.5 %, ≈ 4 μm).
#   Set WIDTH_λW = λ0 to measure on the §6 splines themselves.
#
#   (a) framewise   : every frame of every current — profile and spline of
#                     `SG0_framewise_maxima` (5-px median along x, mean over x,
#                     nz-binning), peak = §6a z_max of that frame;
#                     then mean ± SEM over the frames of each current.
#   (b) meanprofile : the frame-averaged profile of `SG0_mean_maxima` (mean over
#                     x and frames, nz-binning, no median), peak = §6b z_max;
#                     error = jackknife over frames (leave one frame out,
#                     re-average, refit, re-find the peak, re-measure).
#
# Each profile gives two HWHMs, both around the same peak z_max (§6, λ0) and
# the same half-maximum level:
#   level  = b + (S_w(z_max) − b)/2    S_w = module spline with λ_w, b = baseline
#   spline : z_L, z_R = crossings of `level` by the spline S_w
#            (¼-bin grid walk outward from z_max, then bisection)
#   raw    : z_L, z_R = crossings of `level` by the binned data points
#            (walk outward bin by bin from the bin nearest z_max; linear
#            interpolation between the last bin ≥ level and the first < level)
#   HWHM = (z_R − z_L)/2,   HWHM_L = z_max − z_L,   HWHM_R = z_R − z_max
#
# Baseline b: `:edges` = median of the outer WIDTH_EDGE_FRAC of the z-bins on
# both sides (checked on 20260706/Round1: the edge windows hold only flat floor,
# ≥ 22 HWHM from the peak; `:zero` widens the HWHM by 1–4.5 μm there).
#
# ┌──────────────────────────────── MAINTENANCE ────────────────────────────────┐
# │ The functions `replica_profiles`, `module_spline` and `module_peak` below   │
# │ are LOCAL COPIES of the profile / spline / peak-search steps inside          │
# │ MyExperimentalAnalysis.SG0_framewise_maxima and .SG0_mean_maxima (those      │
# │ functions only return the peak positions, and many other scripts use them,  │
# │ so they are not modified here).                                              │
# │ If either module function changes (median window, binning, weights, λ0      │
# │ handling, initial guesses, optimiser, …) these copies MUST be updated.       │
# │ `width_analysis` verifies the copies on every run: it recomputes every peak  │
# │ and the mean profiles and compares them with the §6 results; the comparison  │
# │ is printed and a warning is raised if anything differs by more than          │
# │ WIDTH_MATCH_TOL. Do not trust §8 if that warning appears.                    │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# Caveats
# * The SPLINE HWHM keeps the λ_w smoothing bias of the table above (+1.5 %
#   at 1e-3). The RAW HWHM has no smoothing bias of its own but is noisier
#   (two noisy points per edge); its level uses the spline height S_w(z_max),
#   because the tallest raw bin is biased high by noise, so a stiff λ_w also
#   widens it slightly (≤ ±2 μm for λ ≤ 3e-3, +1…5 μm at 1e-2).
#   spline − raw ≈ smoothing bias.
# * The framewise profiles are median-filtered along x, the mean profile is not
#   (as in the module functions); compare (a) and (b) with that in mind.
# * The mean profile also contains the frame-to-frame jitter σ_jit of the peak:
#   HWHM_b² ≈ HWHM_a² + 2 ln2 σ_jit² for Gaussian lobes.
# * Errors are statistical only (SEM over frames / jackknife); no σ_pix term,
#   the crossings are interpolated, not read on the pixel grid.
# * HWHM_L / HWHM_R depend on where z_max sits; treat them as an asymmetry
#   indicator. A crossing not found inside the camera gives NaN; NaN frames are
#   left out of the averages (`nfr_*` counts the valid frames).
# * Requires HALF_MAX = false: with half_max the module fits the spline only
#   above half maximum, so the half-maximum crossings would be the fit edges.

WIDTH_λW        = 1e-3     # spline smoothing for the widths (peaks keep the λ0 of §2/§6)
WIDTH_BASELINE  = :edges   # :edges → median of the outer z-bins, :zero → b = 0
WIDTH_EDGE_FRAC = 0.05     # fraction of z-bins used on EACH side for :edges
WIDTH_JACKKNIFE = true     # jackknife error of the mean-profile HWHM (Nframes extra fits per current)
WIDTH_MATCH_TOL = 1e-9     # [mm] / [counts] tolerance of the replica check against §6

# --- Local copies of the module steps (see MAINTENANCE) --------------------------
const MEA = MyExperimentalAnalysis

"z-bin centres [mm], same call as in `SG0_framewise_maxima` / `SG0_mean_maxima`."
module_z_mm(nz::Integer) = 1e3 .* MEA.pixel_positions(MEA.DEFAULT_z_pixels, nz, MEA.effective_cam_pixelsize_z)

"""
    replica_profiles(data, state, j; nz) -> (; z, fw, plain, mean)

Binned z-profiles of `state` (`:F1`/`:F2`) at current index `j`, built as in
the module functions:
* `fw    :: (Nz÷nz, Nframes)` — `SG0_framewise_maxima` (5-px median along x,
  mean over x, binning);
* `mean  :: Nz÷nz` — `SG0_mean_maxima` (mean over x and frames, binning);
* `plain :: (Nz÷nz, Nframes)` — per-frame profiles without median (the frames
  that `mean` averages; used for the jackknife).
"""
function replica_profiles(data, state::Symbol, j::Integer; nz::Integer)
    stack = Float64.(data[Symbol(state, :ProcessedImages)][:, :, :, j])
    size(stack, 2) % nz == 0 || error("Nz = $(size(stack, 2)) is not divisible by nz = $nz")
    bin(p) = dropdims(mean(reshape(p, nz, :, size(p, 2)); dims = 1); dims = 1)

    # SG0_mean_maxima
    full  = vec(dropdims(mean(stack, dims = (1, 3)), dims = (1, 3)))
    pmean = vec(mean(reshape(full, nz, :), dims = 1))
    plain = bin(dropdims(mean(stack, dims = 1), dims = 1))
    # SG0_framewise_maxima
    fw    = bin(dropdims(mean(mapwindow(median, stack, (5, 1, 1)), dims = 1), dims = 1))

    return (; z = module_z_mm(nz), fw, plain, mean = pmean)
end

"Cubic smoothing spline, same call as in the module functions."
module_spline(z, y, λ0::Real) =
    BSplineKit.fit(BSplineOrder(4), z, y, Float64(λ0); weights = MEA.compute_weights(z, Float64(λ0)))

# initial-guess quantiles of the module peak searches (+ the tallest data point)
const MODULE_GUESS_Q = (framewise = (0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
                        mean      = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90))

"""
    module_peak(S, z, y; method) -> z_max

Tallest maximum of the spline `S`, found as in the module (`method = :framewise`
or `:mean`, which differ only in the initial guesses): bounded LBFGS from every
guess, duplicates removed, highest spline value kept.
"""
function module_peak(S, z::AbstractVector, y::AbstractVector; method::Symbol)
    guesses = sort(vcat([quantile(z, q) for q in MODULE_GUESS_Q[method]], z[argmax(y)]))
    lo, hi  = minimum(z), maximum(z)
    cand = sort([Optim.minimizer(optimize(x -> -S(x[1]), [lo], [hi], [g], Fminbox(LBFGS())))[1]
                 for g in guesses])
    dedup = [cand[1]]
    for c in cand[2:end]
        all(abs(c - x) > 1e-9 for x in dedup) && push!(dedup, c)
    end
    return dedup[argmax(S.(dedup))]
end

# --- Half-maximum crossings ---------------------------------------------------------
"Baseline of a profile: `:zero` → 0, `:edges` → median of the outer `frac` of the bins on both sides."
function profile_baseline(y::AbstractVector; mode::Symbol = :edges, frac::Real = 0.05)
    mode === :zero && return 0.0
    mode === :edges || throw(ArgumentError("baseline mode must be :edges or :zero"))
    n = max(1, round(Int, frac * length(y)))
    return median(vcat(y[1:n], y[end-n+1:end]))
end

"""
    spline_crossing(S, z0, level, zlim, dir; step) -> z

First `z` beyond `z0` in direction `dir = ±1` where the spline drops below
`level` (grid walk of `step`, then 50 bisections). `NaN` if `zlim` is reached.
"""
function spline_crossing(S, z0::Real, level::Real, zlim::Real, dir::Int; step::Real)
    a = float(z0)
    while true
        b = a + dir * step
        dir * (b - zlim) > 0 && return NaN
        if S(b) < level
            for _ in 1:50
                m = (a + b) / 2
                S(m) ≥ level ? (a = m) : (b = m)
            end
            return (a + b) / 2
        end
        a = b
    end
end

"""
    raw_crossing(z, y, z0, level, dir) -> z

First crossing of `level` by the data points, walking from the bin nearest
`z0` in direction `dir = ±1`; linear interpolation between the last bin
≥ `level` and the first one below. `NaN` if the camera edge is reached.
"""
function raw_crossing(z::AbstractVector, y::AbstractVector, z0::Real, level::Real, dir::Int)
    k = argmin(abs.(z .- z0))
    while true
        k2 = k + dir
        checkbounds(Bool, y, k2) || return NaN
        if y[k2] < level
            return y[k] == y[k2] ? z[k2] : z[k] + (level - y[k]) * (z[k2] - z[k]) / (y[k2] - y[k])
        end
        k = k2
    end
end

_halfwidths(zL, zR, zp) = (; zL, zR, hwhm = (zR - zL) / 2, hwhm_L = zp - zL, hwhm_R = zR - zp)

"""
    profile_hwhm(z, y, S, zpeak; baseline=:edges, frac=0.05) -> NamedTuple

Raw and spline HWHM of one profile `y(z)` with spline `S` around `zpeak`
(see §8 header). Returns `(; zpeak, height, base, level, raw, spl)` with
`raw`, `spl` = `(; zL, zR, hwhm, hwhm_L, hwhm_R)` [mm].
"""
function profile_hwhm(z::AbstractVector, y::AbstractVector, S, zpeak::Real;
                      baseline::Symbol = :edges, frac::Real = 0.05)
    h, b = S(zpeak), profile_baseline(y; mode = baseline, frac)
    lvl  = b + (h - b) / 2
    nanw = _halfwidths(NaN, NaN, zpeak)
    h > b || return (; zpeak, height = h, base = b, level = lvl, raw = nanw, spl = nanw)
    step = (z[2] - z[1]) / 4
    spl  = _halfwidths(spline_crossing(S, zpeak, lvl, first(z), -1; step),
                       spline_crossing(S, zpeak, lvl, last(z),  +1; step), zpeak)
    raw  = _halfwidths(raw_crossing(z, y, zpeak, lvl, -1), raw_crossing(z, y, zpeak, lvl, +1), zpeak)
    return (; zpeak, height = h, base = b, level = lvl, raw, spl)
end

# --- Analysis --------------------------------------------------------------------------
const WIDTH_METHODS = (raw = "Raw", spl = "Spline")
const WIDTH_QTYS    = (hwhm = "HWHM", hwhm_L = "HWHM_L", hwhm_R = "HWHM_R")
_wcol(q, m) = Symbol(q, "_", m)          # e.g. :hwhm_L_raw
_ecol(q, m) = Symbol(:err, q, "_", m)    # e.g. :errhwhm_L_raw

"Jackknife standard error of the leave-one-out estimates `v` (NaN ignored)."
function jackknife_err(v::AbstractVector)
    u = filter(isfinite, v); n = length(u)
    n < 2 && return NaN
    return sqrt((n - 1) / n * sum(abs2, u .- mean(u)))
end

"""
    width_analysis(data, peaks, peaks_mean; nz, λ0, λw=λ0, half_max=false, baseline=:edges,
                   frac=0.05, jackknife=true, tol=WIDTH_MATCH_TOL) -> NamedTuple

§8 for a processed run, using the §6a (`peaks`) and §6b (`peaks_mean`) results,
which must come from the same `nz`, `λ0`, `half_max`. Peaks use the λ0
splines; widths use a second spline of the same profile with smoothing `λw`. Returns
`(; F1, F2, check)` with, for each state,

* `framewise.table`   — control columns of [`control_table`](@ref) + for
  m ∈ {raw, spl}, q ∈ {hwhm, hwhm_L, hwhm_R}: `q_m` (mean over frames), `errq_m`
  (SEM) and `nfr_raw`, `nfr_spl` (valid frames);  `framewise.zero` its StartPosition;
* `framewise.perframe` — `(Nframes, Ncurrents)` matrices: `q_m`, `zpeak`, `height`, `base`;
* `meanprofile.table` — control columns + `zpeak`, `height`, `base`, `q_m`,
  `errq_m` (jackknife, NaN if `jackknife=false`);  `meanprofile.zero`;

and `check` = largest replica-vs-§6 differences (peaks [mm], mean profile [counts]),
`ok` = all within `tol`, `λw` = width smoothing used.
"""
function width_analysis(data, peaks, peaks_mean; nz::Integer, λ0::Real, λw::Real = λ0,
                        half_max::Bool = false,
                        baseline::Symbol = :edges, frac::Real = 0.05,
                        jackknife::Bool = true, tol::Real = WIDTH_MATCH_TOL)
    half_max && error("§8 needs HALF_MAX = false (with half_max the module spline only covers the points above half maximum)")
    nI   = length(data[:SG0Currents])
    ctrl = control_table(data)
    MQ   = [(m, q) for m in keys(WIDTH_METHODS) for q in keys(WIDTH_QTYS)]
    res  = Dict{Symbol,Any}()
    check = DataFrame(state = String[], max_dz_framewise_mm = Float64[],
                      max_dz_meanprofile_mm = Float64[], max_dprofile_counts = Float64[])

    for (state, fwmax, mp) in ((:F1, peaks.f1_max, peaks_mean.f1), (:F2, peaks.f2_max, peaks_mean.f2))
        nf = size(fwmax, 1)
        PF = Dict(_wcol(q, m) => fill(NaN, nf, nI) for (m, q) in MQ)
        zp, H, B = copy(fwmax), fill(NaN, nf, nI), fill(NaN, nf, nI)
        MPv = Dict(_wcol(q, m) => fill(NaN, nI) for (m, q) in MQ)
        MPe = Dict(_ecol(q, m) => fill(NaN, nI) for (m, q) in MQ)
        mH, mB = fill(NaN, nI), fill(NaN, nI)
        dz_fw = dz_mp = dprof = 0.0

        @showprogress "HWHM $state: " for j in 1:nI
            r = replica_profiles(data, state, j; nz)
            dprof = max(dprof, maximum(abs, r.mean .- mp.mean_bin_profile[:, j + 1]))

            # (a) framewise
            for i in 1:min(nf, size(r.fw, 2))
                isnan(fwmax[i, j]) && continue
                y = r.fw[:, i]
                S = module_spline(r.z, y, λ0)
                dz_fw = max(dz_fw, abs(module_peak(S, r.z, y; method = :framewise) - fwmax[i, j]))
                Sw = λw == λ0 ? S : module_spline(r.z, y, λw)
                w = profile_hwhm(r.z, y, Sw, fwmax[i, j]; baseline, frac)
                H[i, j], B[i, j] = w.height, w.base
                for (m, q) in MQ
                    PF[_wcol(q, m)][i, j] = getproperty(getproperty(w, m), q)
                end
            end

            # (b) mean profile, at the §6b peak
            S = module_spline(r.z, r.mean, λ0)
            dz_mp = max(dz_mp, abs(module_peak(S, r.z, r.mean; method = :mean) - mp.peak_positions[j]))
            Sw = λw == λ0 ? S : module_spline(r.z, r.mean, λw)
            w = profile_hwhm(r.z, r.mean, Sw, mp.peak_positions[j]; baseline, frac)
            mH[j], mB[j] = w.height, w.base
            for (m, q) in MQ
                MPv[_wcol(q, m)][j] = getproperty(getproperty(w, m), q)
            end

            # (b) jackknife over frames
            n = size(r.plain, 2)
            if jackknife && n ≥ 3
                J = Dict((m, q) => fill(NaN, n) for (m, q) in MQ)
                for k in 1:n
                    yk = vec(mean(view(r.plain, :, [1:k-1; k+1:n]); dims = 2))
                    Sk = module_spline(r.z, yk, λ0)
                    zk = module_peak(Sk, r.z, yk; method = :mean)
                    wk = profile_hwhm(r.z, yk, λw == λ0 ? Sk : module_spline(r.z, yk, λw), zk; baseline, frac)
                    for (m, q) in MQ
                        J[(m, q)][k] = getproperty(getproperty(wk, m), q)
                    end
                end
                for (m, q) in MQ
                    MPe[_ecol(q, m)][j] = jackknife_err(J[(m, q)])
                end
            end
        end

        # tables
        fwt = copy(ctrl)
        for (m, q) in MQ
            μ, sem = nanmean_sem(PF[_wcol(q, m)])
            fwt[!, _wcol(q, m)] = μ
            fwt[!, _ecol(q, m)] = sem
        end
        fwt.nfr_raw = vec(count(isfinite, PF[:hwhm_raw]; dims = 1))
        fwt.nfr_spl = vec(count(isfinite, PF[:hwhm_spl]; dims = 1))

        mpt = copy(ctrl)
        mpt.zpeak, mpt.height, mpt.base = copy(mp.peak_positions), mH, mB
        for (m, q) in MQ
            mpt[!, _wcol(q, m)] = MPv[_wcol(q, m)]
            mpt[!, _ecol(q, m)] = MPe[_ecol(q, m)]
        end

        perframe = (; (k => v for (k, v) in PF)..., zpeak = zp, height = H, base = B)
        res[state] = (; framewise   = (; table = fwt, zero = zero_current_reference(fwt), perframe),
                        meanprofile = (; table = mpt, zero = zero_current_reference(mpt)))
        push!(check, (String(state), dz_fw, dz_mp, dprof))
    end

    ok = all(<=(tol), Matrix(check[:, 2:end]))
    ok || @warn "§8 replica does NOT match MyExperimentalAnalysis (or §6 was run with other nz/λ0/half_max): update replica_profiles / module_spline / module_peak" check
    return (; F1 = res[:F1], F2 = res[:F2], check, ok, λw = Float64(λw))
end

# --- Printing ----------------------------------------------------------------------------
"""
    width_summary(fw, mp) -> DataFrame

Compact view of one state: framewise (FW) and mean-profile (MP) HWHM, raw and
spline, with their errors [mm], valid frames, and spline − raw [μm]. `fw`, `mp`
are tables (or StartPosition rows) of [`width_analysis`](@ref).
"""
width_summary(fw::DataFrame, mp::DataFrame) = DataFrame(
    I0 = fw.I0, B0 = fw.B0,
    fw_raw = fw.hwhm_raw, fw_raw_err = fw.errhwhm_raw,
    fw_spl = fw.hwhm_spl, fw_spl_err = fw.errhwhm_spl, nfr = Int.(fw.nfr_raw),
    mp_raw = mp.hwhm_raw, mp_raw_err = mp.errhwhm_raw,
    mp_spl = mp.hwhm_spl, mp_spl_err = mp.errhwhm_spl,
    d_fw = 1e3 .* (fw.hwhm_spl .- fw.hwhm_raw), d_mp = 1e3 .* (mp.hwhm_spl .- mp.hwhm_raw))

show_width_summary(df::DataFrame; title::AbstractString = "") =
    pretty_table(df; title,
        column_labels = [
            ["I0 Current", "B0 field", "FW raw", "± SEM", "FW spline", "± SEM", "frames",
             "MP raw", "± jackknife", "MP spline", "± jackknife", "spl−raw FW", "spl−raw MP"],
            ["[A]", "[mT]", "[mm]", "[mm]", "[mm]", "[mm]", "",
             "[mm]", "[mm]", "[mm]", "[mm]", "[μm]", "[μm]"],
        ],
        formatters = [fmt__printf("%8.5f", [1]), fmt__printf("%8.3f", [2]),
                      fmt__printf("%8.4f", [3, 5, 8, 10]), fmt__printf("%7.4f", [4, 6, 9, 11]),
                      fmt__printf("%d", [7]), fmt__printf("%6.1f", 12:13)],
        pretty_peak_style()...)

# --- Figures ------------------------------------------------------------------------------
"""
    plot_width_check(data, widths, j; iframe=1, nz, baseline=:edges, frac=0.05, title="") -> Figure

2 × 2 check of the HWHM extraction at current index `j`: rows F1 / F2, columns
frame `iframe` (framewise profile) / mean profile. Grey: binned data; red:
module spline with the width smoothing `widths.λw`; dotted: baseline;
dashed: z_max (§6, λ0); blue: spline full width at
half maximum; green circles: raw crossings.
"""
function plot_width_check(data, widths, j::Integer; iframe::Integer = 1, nz::Integer,
                          baseline::Symbol = :edges, frac::Real = 0.05, title::AbstractString = "")
    fig = CM.Figure(size = (1100, 720))
    for (row, state) in enumerate((:F1, :F2))
        r   = replica_profiles(data, state, j; nz)
        W   = getproperty(widths, state)
        cases = ((r.fw[:, iframe], W.framewise.perframe.zpeak[iframe, j], "frame $iframe"),
                 (r.mean,          W.meanprofile.table.zpeak[j],          "mean profile"))
        for (col, (y, zp, what)) in enumerate(cases)
            S = module_spline(r.z, y, widths.λw)
            w = profile_hwhm(r.z, y, S, zp; baseline, frac)
            ax = CM.Axis(fig[row, col]; xlabel = L"z \; (\mathrm{mm})", ylabel = L"\langle I \rangle_x",
                         title = @sprintf("%s, %s:  HWHM raw %.4f / spline %.4f mm", state, what,
                                          w.raw.hwhm, w.spl.hwhm))
            CM.scatter!(ax, r.z, y; markersize = 3, color = (:gray40, 0.5))
            zz = range(first(r.z), last(r.z); length = 4000)
            CM.lines!(ax, zz, S.(zz); color = :red, linewidth = 1.5)
            CM.hlines!(ax, [w.base]; color = :gray50, linestyle = :dot)
            CM.vlines!(ax, [zp]; color = :black, linestyle = :dash, linewidth = 1)
            if isfinite(w.spl.hwhm)
                CM.lines!(ax, [w.spl.zL, w.spl.zR], [w.level, w.level]; color = :dodgerblue, linewidth = 2.5)
                CM.xlims!(ax, zp - 5w.spl.hwhm_L, zp + 5w.spl.hwhm_R)
            end
            ok = filter(isfinite, [w.raw.zL, w.raw.zR])
            isempty(ok) || CM.scatter!(ax, ok, fill(w.level, length(ok)); color = :transparent,
                                       strokecolor = :green, strokewidth = 2, markersize = 12)
        end
    end
    isempty(title) || CM.Label(fig[0, :], title; fontsize = 16)
    return fig
end

"""
    plot_width_vs_current(widths; title="") -> Figure

HWHM [μm] vs |I_SG0|, one panel per state: framewise (filled) and mean
profile (open), raw (circles) and spline (squares), with their errors.
"""
function plot_width_vs_current(widths; title::AbstractString = "")
    fig = CM.Figure(size = (1200, 520))
    axs = [CM.Axis(fig[1, c]; title = s, xlabel = L"|I_{\mathrm{SG0}}| \; (\mathrm{A})",
                   ylabel = c == 1 ? L"\mathrm{HWHM}_z \; (\mu\mathrm{m})" : "",
                   xminorticksvisible = true, yminorticksvisible = true,
                   xminorticks = CM.IntervalsBetween(5), yminorticks = CM.IntervalsBetween(5))
           for (c, s) in enumerate(("F1", "F2"))]
    CM.linkyaxes!(axs...)
    series = ((:framewise, :raw, :circle, true, :dodgerblue, "framewise, raw"),
              (:framewise, :spl, :rect,   true, :orangered,  "framewise, spline"),
              (:meanprofile, :raw, :circle, false, :dodgerblue, "mean profile, raw"),
              (:meanprofile, :spl, :rect,   false, :orangered,  "mean profile, spline"))
    for (ax, state) in zip(axs, (:F1, :F2)), (grp, m, marker, filled, color, label) in series
        t = getproperty(getproperty(widths, state), grp).table
        x = t.I0                                                  # true currents, no offset
        y, e = 1e3 .* t[!, _wcol(:hwhm, m)], 1e3 .* t[!, _ecol(:hwhm, m)]   # mm → μm
        CM.errorbars!(ax, x, y, map(v -> isfinite(v) ? v : 0.0, e); color, whiskerwidth = 5)
        CM.scatter!(ax, x, y; marker, label, markersize = 9,
                    color = filled ? color : :white, strokecolor = color, strokewidth = 1.5)
    end
    CM.Legend(fig[2, 1:2], axs[1]; orientation = :horizontal, framevisible = false)
    isempty(title) || CM.Label(fig[0, :], title; fontsize = 16)
    return fig
end

# --- Output file -------------------------------------------------------------------------
# widths_nz<nz>_<RUN_STAMP>.jld2 in the run's summary folder (units as in §6):
#
#   meta/        as in the peaks file
#   framewise/F1/  HWHM_Raw_mm, ErrHWHM_Raw_mm, HWHM_L_Raw_mm, ErrHWHM_L_Raw_mm,
#                  HWHM_R_Raw_mm, ErrHWHM_R_Raw_mm, and the same with _Spline_
#                  (mean ± SEM over frames, one per current), NFrames_Raw, NFrames_Spline
#                  StartPosition/<6 control keys + the 12 keys above>
#                  PerFrame/<the 6 HWHM keys>, Peak_mm, Height, Baseline  (Nframes × Ncurrents)
#   framewise/F2/  same
#   meanprofile/F1/ the 12 HWHM keys (Err = jackknife), Peak_mm, Height, Baseline,
#                  StartPosition/...
#   meanprofile/F2/ same
#   settings/    AnalysisDirectory, nz, lambda0 (peaks), lambdaWidth (λ_w), HalfMax, Baseline, EdgeFrac,
#                Jackknife, RunStamp, ReplicaCheck/{F1,F2}/MaxDzFramewise_mm,
#                MaxDzMeanProfile_mm, MaxDProfile_counts, ReplicaCheck/OK

"Column → key map of the width tables for `_write_peak_group!`."
const WIDTH_KEYS = (; (k => v for (m, M) in pairs(WIDTH_METHODS) for (q, Q) in pairs(WIDTH_QTYS)
    for (k, v) in ((_wcol(q, m), "$(Q)_$(M)_mm"), (_ecol(q, m), "Err$(Q)_$(M)_mm")))...)

"""
    save_width_results(path, data, widths; nz, λ0, half_max, baseline, frac, jackknife) -> path

Write the §8 results to `path` with the layout described above.
"""
function save_width_results(path::AbstractString, data, widths; nz::Integer, λ0::Real,
                            half_max::Bool, baseline::Symbol, frac::Real, jackknife::Bool)
    jldopen(path, "w") do f
        _write_meta!(f, data)
        for state in (:F1, :F2)
            W = getproperty(widths, state)
            g = "framewise/$state"
            _write_peak_group!(f, g, W.framewise.table, W.framewise.zero; keys = WIDTH_KEYS)
            f["$g/NFrames_Raw"]    = W.framewise.table.nfr_raw
            f["$g/NFrames_Spline"] = W.framewise.table.nfr_spl
            pf = W.framewise.perframe
            for m in keys(WIDTH_METHODS), q in keys(WIDTH_QTYS)
                f["$g/PerFrame/$(WIDTH_KEYS[_wcol(q, m)])"] = getproperty(pf, _wcol(q, m))
            end
            f["$g/PerFrame/Peak_mm"]  = pf.zpeak
            f["$g/PerFrame/Height"]   = pf.height
            f["$g/PerFrame/Baseline"] = pf.base

            g = "meanprofile/$state"
            t = W.meanprofile.table
            _write_peak_group!(f, g, t, W.meanprofile.zero; keys = WIDTH_KEYS)
            f["$g/Peak_mm"]  = t.zpeak
            f["$g/Height"]   = t.height
            f["$g/Baseline"] = t.base

            c = widths.check[widths.check.state .== String(state), :][1, :]
            f["settings/ReplicaCheck/$state/MaxDzFramewise_mm"]   = c.max_dz_framewise_mm
            f["settings/ReplicaCheck/$state/MaxDzMeanProfile_mm"] = c.max_dz_meanprofile_mm
            f["settings/ReplicaCheck/$state/MaxDProfile_counts"]  = c.max_dprofile_counts
        end
        f["settings/ReplicaCheck/OK"]   = widths.ok
        f["settings/AnalysisDirectory"] = ANALYSIS_DIR
        f["settings/nz"]                = Int(nz)
        f["settings/lambda0"]           = Float64(λ0)        # peaks (§6)
        f["settings/lambdaWidth"]       = widths.λw          # widths
        f["settings/HalfMax"]           = half_max
        f["settings/Baseline"]          = String(baseline)
        f["settings/EdgeFrac"]          = Float64(frac)
        f["settings/Jackknife"]         = jackknife
        f["settings/RunStamp"]          = RUN_STAMP
    end
    return path
end

# --- Run -----------------------------------------------------------------------------------
widths = width_analysis(data_processed, peaks, peaks_mean; nz, λ0, λw = WIDTH_λW, half_max = HALF_MAX,
                        baseline = WIDTH_BASELINE, frac = WIDTH_EDGE_FRAC, jackknife = WIDTH_JACKKNIFE)

pretty_table(widths.check; title = "§8 replica check against §6 (all must be ≤ $WIDTH_MATCH_TOL)",
             formatters = [fmt__printf("%.2e", 2:4)], pretty_peak_style()...)

for state in (:F1, :F2)
    W = getproperty(widths, state)
    show_width_summary(width_summary(W.framewise.table, W.meanprofile.table);
        title = "$ANALYSIS_DIR — $state HWHM along z  (nz = $nz, λ0 = $λ0 peaks, λ_w = $WIDTH_λW widths, baseline = $WIDTH_BASELINE)")
    nrow(W.framewise.zero) > 0 &&
        show_width_summary(width_summary(W.framewise.zero, W.meanprofile.zero);
            title = "$ANALYSIS_DIR — $state HWHM, StartPosition")
end

# Frame and current shown in the HWHM check figure
iframe   = 1           # frame index within the current setting
icurrent = 16          # SG0 current index
let (_, _, nf, nI) = size(data_processed[:F1ProcessedImages])
    1 ≤ iframe ≤ nf   || error("iframe=$iframe out of range 1:$nf")
    1 ≤ icurrent ≤ nI || error("icurrent=$icurrent out of range 1:$nI")
end

fig_wcheck = plot_width_check(data_processed, widths, icurrent; iframe, nz,
                              baseline = WIDTH_BASELINE, frac = WIDTH_EDGE_FRAC,
                              title = @sprintf("%s — HWHM check, frame %d, SG0 current = %.3f A",
                                               ANALYSIS_DIR, iframe, data_processed[:SG0Currents][icurrent]))
display(fig_wcheck)

fig_wI = plot_width_vs_current(widths;
    title = L"%$(ANALYSIS_DIR) — HWHM along $z$  ($n_{z} = %$(nz)$, $\lambda_{0} = %$(λ0)$, $\lambda_{w} = %$(WIDTH_λW)$)")
display(fig_wI)

if SAVE_FIG
    for (fig, name) in ((fig_wcheck, "HWHMcheck_frame$(iframe)_I$(icurrent)"), (fig_wI, "HWHM_vs_I_nz$(nz)"))
        fname = joinpath(summary_dir(ANALYSIS_DIR), "$(name)_$(RUN_STAMP).$(FIG_EXT)")
        CM.save(fname, fig; px_per_unit = 4)
        @info "Saved" fname
    end
end

if SAVE_TABLE
    widths_file = joinpath(summary_dir(ANALYSIS_DIR), "widths_nz$(nz)_$(RUN_STAMP).jld2")
    save_width_results(widths_file, data_processed, widths; nz, λ0, half_max = HALF_MAX,
                       baseline = WIDTH_BASELINE, frac = WIDTH_EDGE_FRAC, jackknife = WIDTH_JACKKNIFE)
    @info "Saved width results" widths_file
    JLD2_MyTools.list_keys_jld_peaks(widths_file)
end
