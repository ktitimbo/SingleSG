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
#    §6  Peak positions  framewise spline maxima of F1/F2 → z_F1, z_F2,
#                        splitting Δz and centroid C₀ vs current (+ I=0 reference)
#    §6b                 same from the maximum of the frame-averaged profile
#                        (SG0_mean_maxima) and its difference to §6
#    §7  Figures         single frame / F1|F2 images with averaged profiles
#    §8  Pattern widths  HWHM along z per frame and of the frame-averaged
#                        profile (left/right half widths, vs current)
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
SAVE_TABLE = false   # write result tables (CSV + JLD2) to the run's summary folder
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
const X_POSITION = pixel_positions(X_PIXELS, 1, EXP_PIXELSIZE_X)
const Z_POSITION = pixel_positions(Z_PIXELS, 1, EXP_PIXELSIZE_Z)

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
ANALYSIS_DIR = "20260706/Round1"
@assert ANALYSIS_DIR in data_directories "ANALYSIS_DIR=$ANALYSIS_DIR is not in data_directories"

data_processed = load_processed(ANALYSIS_DIR)
let F1 = data_processed[:F1ProcessedImages]
    nx, nz, nf, nI = size(F1)
    @info "Loaded $ANALYSIS_DIR" Nx = nx Nz = nz Nframes = nf Ncurrents = nI I_SG0 = data_processed[:SG0Currents]
end


# =============================================================================
# §6  PEAK POSITIONS, SPLITTING AND CENTROID
# =============================================================================
# For every frame and every current, the x-averaged z-profile is fitted with a
# cubic smoothing spline and its tallest maximum z_max is taken as the peak
# position (`MyExperimentalAnalysis.SG0_framewise_maxima`; it also applies a
# 5-px median filter along x to remove hot pixels). Then, per current:
#
#   z_F       = ⟨z_max⟩_frames                       (F = F1, F2)
#   σ_F       = √( SEM_frames² + σ_pix² ),   σ_pix = Δz_bin/√12
#                (σ_pix: uniform position uncertainty within one z-bin)
#   Δz        = |z_F1 − z_F2|,               σ_Δz = √(σ_F1² + σ_F2²)
#   C₀        = (z_F1 + z_F2)/2,             σ_C₀ = σ_Δz/2
#
# F1 and F2 are treated as independent. σ_pix is added to each state; since
# both are read on the same pixel grid this is conservative for Δz and C₀.

nz        = 1         # z-binning before the spline fit (must divide Z_PIXELS)
λ0        = 0.01      # spline smoothing parameter
HALF_MAX  = false     # true → fit only the points above half maximum

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

"""
    peak_table(data; nz=1, λ0=0.01, half_max=false) -> (; table, f1_max, f2_max, σ_pix)

Peak positions of F1 and F2 for every current of a processed run (see the
formulas at the top of §6). Returns:

* `table`  — `DataFrame`, one row per current, columns
  `I0, I1, IG` [A] (absolute values of SG0, SG1 and guiding-magnet currents),
  `B0, B1, BG` [mT], `zf1, errzf1, zf2, errzf2, split, errsplit, c0, errc0` [mm].
  Numbers are stored at full precision (rounding is only applied when printing).
* `f1_max`, `f2_max` — framewise peak positions [mm], `(Nframes, Ncurrents)`.
* `σ_pix`  — the bin-quantisation uncertainty used [mm].
"""
function peak_table(data; nz::Integer = 1, λ0::Real = 0.01, half_max::Bool = false)
    nI    = length(data[:SG0Currents])
    σ_pix = 1e3 / sqrt(12) * EXP_PIXELSIZE_Z * nz

    f1_max = MyExperimentalAnalysis.SG0_framewise_maxima("F1", data, nz; half_max, λ0 = Float64(λ0))
    f2_max = MyExperimentalAnalysis.SG0_framewise_maxima("F2", data, nz; half_max, λ0 = Float64(λ0))

    zf1, sem1 = nanmean_sem(f1_max)
    zf2, sem2 = nanmean_sem(f2_max)
    errzf1    = hypot.(sem1, σ_pix)
    errzf2    = hypot.(sem2, σ_pix)
    errsplit  = hypot.(errzf1, errzf2)

    table = DataFrame(
        I0       = abs.(per_current(data[:SG0Currents],   nI)),
        I1       = abs.(per_current(data[:SG1Currents],   nI)),
        IG       = per_current(data[:MGcurrentInA],       nI),
        B0       = 1e3 .* per_current(data[:SG0Bz],          nI),   # T → mT
        B1       = 1e3 .* per_current(data[:SG1Bz],          nI),
        BG       = 1e3 .* per_current(data[:MGfieldInTesla], nI),
        zf1      = zf1,
        errzf1   = errzf1,
        zf2      = zf2,
        errzf2   = errzf2,
        split    = abs.(zf1 .- zf2),
        errsplit = errsplit,
        c0       = (zf1 .+ zf2) ./ 2,
        errc0    = errsplit ./ 2,
    )
    return (; table, f1_max, f2_max, σ_pix)
end

"""
    zero_current_reference(table; tol=1e-9) -> DataFrame

One-row summary of the rows with `I0 ≈ 0` and `I1 ≈ 0` (the field-free
reference points taken during the scan). Value columns are averaged; the
uncertainty columns (`err*`) are combined as the error of a mean of
independent points, `√(Σσᵢ²)/n`. Note that the σ_pix part of each error is
systematic and does not really average down, so this is a lower bound when
σ_pix dominates. Returns an empty `DataFrame` (with a warning) if there are no
zero-current rows.
"""
function zero_current_reference(table::DataFrame; tol::Real = 1e-9)
    rows = table[(abs.(table.I0) .<= tol) .& (abs.(table.I1) .<= tol), :]
    n    = nrow(rows)
    n == 0 && (@warn "No zero-current rows found"; return similar(table, 0))
    reduce_col(col, name) = startswith(name, "err") ? sqrt(sum(abs2, col)) / n : mean(col)
    return DataFrame([name => [reduce_col(rows[!, name], name)] for name in names(rows)])
end

# Column headers and number formats shared by all peak tables
const PEAK_TABLE_LABELS = [
    ["I0 Current", "I1 Current", "IG Current", "B0 field", "B1 field", "BG field",
     "F1", "F1 err", "F2", "F2 err", "Δz", "Δz err", "C₀", "C₀ err"],
    ["[A]", "[A]", "[A]", "[mT]", "[mT]", "[mT]",
     "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"],
]

"Pretty-print a table produced by `peak_table` / `zero_current_reference`."
function show_peak_table(df::DataFrame; title::AbstractString = "")
    pretty_table(df;
        title,
        formatters    = [fmt__printf("%8.5f", 1:3),             # currents
                         fmt__printf("%8.3f", 4:6),             # fields
                         fmt__printf("%8.3f", [7, 9, 11, 13]),  # positions
                         fmt__printf("%8.4f", [8, 10, 12, 14])],# uncertainties
        alignment     = :c,
        column_labels = PEAK_TABLE_LABELS,
        table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style         = TextTableStyle(
            first_line_column_label = crayon"yellow bold",
            column_label            = crayon"yellow",
            table_border            = crayon"blue bold",
            title                   = crayon"bold red",
        ),
        equal_data_column_widths = true,
        fit_table_in_display_horizontally = false,   # never drop columns in narrow terminals
        fit_table_in_display_vertically   = false,   # never drop rows
    )
end

# --- Run -----------------------------------------------------------------------
peaks      = peak_table(data_processed; nz, λ0, half_max = HALF_MAX)
peak_data  = peaks.table
peak_zero  = zero_current_reference(peak_data)

T_oven = data_processed[:TemperatureInCelsius]
show_peak_table(peak_data; title = "$ANALYSIS_DIR  T = $(T_oven) °C  (nz = $nz, λ0 = $λ0)")
nrow(peak_zero) > 0 && show_peak_table(peak_zero; title = "$ANALYSIS_DIR — zero-current reference")


# -----------------------------------------------------------------------------
# §6b  ALTERNATIVE: MAXIMUM OF THE MEAN PROFILE  (for comparison with §6)
# -----------------------------------------------------------------------------
# `MyExperimentalAnalysis.SG0_mean_maxima`: for each current the stack is
# averaged over x AND over all frames first, and the peak is the tallest
# maximum of the spline fit to that single mean profile (same nz, λ0).
#
#   §6  (framewise)  z_F = ⟨ argmax S_frame ⟩_frames   — "mean of the maxima"
#   §6b (mean)       z_F =   argmax S_⟨frame⟩          — "maximum of the mean"
#
# Differences to expect between the two:
# * §6 applies a 5-px median along x before averaging (hot-pixel removal);
#   §6b does not, so a spike in any frame enters the mean profile directly.
# * For symmetric lobes both agree on average. For asymmetric lobes with
#   frame-to-frame jitter they differ: averaging the frames blurs the profile
#   and moves its maximum toward the broad side.
# * §6b has one profile per current, so there is no frame scatter to give a
#   statistical error: err = σ_pix only. This is a LOWER BOUND; the SEM of §6
#   is the realistic statistical error for both methods.

"""
    mean_peak_table(data; nz=1, λ0=0.01, half_max=false) -> NamedTuple

Same output layout as [`peak_table`](@ref), but with the peak of the
frame-averaged profile (`SG0_mean_maxima`). Returns
`(; table, f1, f2, σ_pix)`, where `table` has the columns of `peak_table`
(errors = σ_pix only, see §6b) and `f1`, `f2` are the full outputs of
`SG0_mean_maxima` (`peak_positions`, `mean_profile`, `mean_bin_profile`).
"""
function mean_peak_table(data; nz::Integer = 1, λ0::Real = 0.01, half_max::Bool = false)
    nI    = length(data[:SG0Currents])
    σ_pix = 1e3 / sqrt(12) * EXP_PIXELSIZE_Z * nz

    f1 = MyExperimentalAnalysis.SG0_mean_maxima("F1", data, nz; half_max, λ0 = Float64(λ0))
    f2 = MyExperimentalAnalysis.SG0_mean_maxima("F2", data, nz; half_max, λ0 = Float64(λ0))
    zf1, zf2 = f1.peak_positions, f2.peak_positions
    err      = fill(σ_pix, nI)

    table = DataFrame(
        I0       = abs.(per_current(data[:SG0Currents],   nI)),
        I1       = abs.(per_current(data[:SG1Currents],   nI)),
        IG       = per_current(data[:MGcurrentInA],       nI),
        B0       = 1e3 .* per_current(data[:SG0Bz],          nI),
        B1       = 1e3 .* per_current(data[:SG1Bz],          nI),
        BG       = 1e3 .* per_current(data[:MGfieldInTesla], nI),
        zf1      = zf1,
        errzf1   = err,
        zf2      = zf2,
        errzf2   = err,
        split    = abs.(zf1 .- zf2),
        errsplit = sqrt(2) .* err,
        c0       = (zf1 .+ zf2) ./ 2,
        errc0    = err ./ sqrt(2),
    )
    return (; table, f1, f2, σ_pix)
end

"""
    compare_peak_tables(A, B) -> DataFrame

Difference B − A of two peak tables (same currents, same order), in μm, for
z_F1, z_F2, Δz and C₀, and the same differences in units of A's uncertainty
(`n…` columns). With A = framewise (§6) and B = mean profile (§6b), |n| ≲ 1
means the two methods agree within the framewise error.
"""
function compare_peak_tables(A::DataFrame, B::DataFrame)
    A.I0 ≈ B.I0 || error("Tables do not have the same currents")
    d(col) = 1e3 .* (B[!, col] .- A[!, col])          # mm → μm
    n(col, ecol) = (B[!, col] .- A[!, col]) ./ A[!, ecol]
    return DataFrame(
        I0     = A.I0,
        B0     = A.B0,
        dzf1   = d(:zf1),   dzf2   = d(:zf2),
        dsplit = d(:split), dc0    = d(:c0),
        nzf1   = n(:zf1, :errzf1),   nzf2 = n(:zf2, :errzf2),
        nsplit = n(:split, :errsplit), nc0 = n(:c0, :errc0),
    )
end

"Pretty-print the output of `compare_peak_tables`."
function show_peak_comparison(df::DataFrame; title::AbstractString = "")
    pretty_table(df;
        title,
        formatters    = [fmt__printf("%8.5f", [1]), fmt__printf("%8.3f", [2]),
                         fmt__printf("%8.2f", 3:6), fmt__printf("%6.2f", 7:10)],
        alignment     = :c,
        column_labels = [
            ["I0 Current", "B0 field", "δF1", "δF2", "δΔz", "δC₀",
             "δF1/σ", "δF2/σ", "δΔz/σ", "δC₀/σ"],
            ["[A]", "[mT]", "[μm]", "[μm]", "[μm]", "[μm]", "", "", "", ""],
        ],
        table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style         = TextTableStyle(
            first_line_column_label = crayon"yellow bold",
            column_label            = crayon"yellow",
            table_border            = crayon"blue bold",
            title                   = crayon"bold red",
        ),
        equal_data_column_widths = true,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically   = false,
    )
end

# --- Run -----------------------------------------------------------------------
peaks_mean     = mean_peak_table(data_processed; nz, λ0, half_max = HALF_MAX)
peak_data_mean = peaks_mean.table
peak_zero_mean = zero_current_reference(peak_data_mean)
peak_compare   = compare_peak_tables(peak_data, peak_data_mean)

show_peak_table(peak_data_mean;
    title = "$ANALYSIS_DIR  T = $(T_oven) °C — MEAN PROFILE (nz = $nz, λ0 = $λ0; err = σ_pix only)")
nrow(peak_zero_mean) > 0 &&
    show_peak_table(peak_zero_mean; title = "$ANALYSIS_DIR — zero-current reference, mean profile")
show_peak_comparison(peak_compare;
    title = "$ANALYSIS_DIR — mean profile − framewise  (σ = framewise uncertainty)")


if SAVE_TABLE
    stem = joinpath(summary_dir(ANALYSIS_DIR), "peaks_nz$(nz)_$(RUN_STAMP)")
    CSV.write(stem * ".csv", peak_data)
    CSV.write(stem * "_meanprofile.csv", peak_data_mean)
    CSV.write(stem * "_compare.csv", peak_compare)
    jldsave(stem * ".jld2";
            table = peak_data, zero_reference = peak_zero,
            f1_max = peaks.f1_max, f2_max = peaks.f2_max,
            table_meanprofile = peak_data_mean, zero_reference_meanprofile = peak_zero_mean,
            f1_meanprofile = peaks_mean.f1.mean_bin_profile,   # [z_mm  y₁ … y_nI]
            f2_meanprofile = peaks_mean.f2.mean_bin_profile,
            comparison = peak_compare,
            nz, λ0, half_max = HALF_MAX, σ_pix_mm = peaks.σ_pix)
    @info "Saved peak tables" stem
end


# =============================================================================
# §7  SINGLE-FRAME IMAGE AND MARGINAL PROFILES
# =============================================================================
# Layout used by both figures below (one block per frame, all in ONE flat grid
# so that image rows/columns of different blocks stay aligned):
#
#        col c          col c+1
#   r1 ┌──────────────┐
#      │  ⟨I⟩_z (x)   │                 ← mean over z  (dims = 2)
#   r2 ├──────────────┼────────┐
#      │  image (x,z) │ ⟨I⟩_x  │       ← mean over x  (dims = 1), along z
#      └──────────────┴────────┘
#
# Profiles are plain means over the full other axis (no x/z window, no
# baseline removal). Coordinates are given in metres and shown in mm.

"Check that `img :: (Nx, Nz)` matches the coordinate vectors."
function _check_frame(img::AbstractMatrix, x::AbstractVector, z::AbstractVector)
    size(img) == (length(x), length(z)) ||
        throw(DimensionMismatch("img is $(size(img)); expected ($(length(x)), $(length(z)))"))
    return nothing
end

"""
    _frame_block!(fig, col, img, x_mm, z_mm; colormap, colorrange,
                  label="", zticks=true) -> NamedTuple

Draw one image+profiles block in rows 1–2, columns `col`, `col+1` of `fig`.
Returns `(; hm, ax_im, ax_x, ax_z)` so the caller can link axes / add colorbars.
"""
function _frame_block!(fig, col::Int, img::AbstractMatrix, x_mm, z_mm;
                       colormap = :inferno, colorrange = CM.Makie.automatic,
                       label::AbstractString = "", zticks::Bool = true)
    prof_x = vec(mean(img; dims = 2))       # average over z → function of x
    prof_z = vec(mean(img; dims = 1))       # average over x → function of z

    ax_x  = CM.Axis(fig[1, col]; ylabel = L"\langle I \rangle_z", title = label,
                    xticklabelsvisible = false)
    ax_im = CM.Axis(fig[2, col]; xlabel = L"x \; (\mathrm{mm})",
                    ylabel = zticks ? L"z \; (\mathrm{mm})" : "",
                    yticklabelsvisible = zticks)
    ax_z  = CM.Axis(fig[2, col + 1]; xlabel = L"\langle I \rangle_x",
                    yticklabelsvisible = false, xticklabelrotation = π/4)

    hm = CM.heatmap!(ax_im, x_mm, z_mm, img; colormap, colorrange)
    CM.lines!(ax_x, x_mm, prof_x; color = :black, linewidth = 1)
    CM.lines!(ax_z, prof_z, z_mm; color = :black, linewidth = 1)

    CM.linkxaxes!(ax_im, ax_x)
    CM.linkyaxes!(ax_im, ax_z)
    CM.limits!(ax_im, extrema(x_mm)..., extrema(z_mm)...)
    return (; hm, ax_im, ax_x, ax_z)
end

"Common sizes: profile row = 18 % of the figure height, each profile column = 13 % of its width."
function _style_layout!(fig, profile_cols)
    CM.rowsize!(fig.layout, 1, CM.Relative(0.18))
    for c in profile_cols
        CM.colsize!(fig.layout, c, CM.Relative(0.13))
    end
    CM.rowgap!(fig.layout, 1, 6)
    return fig
end

"""
    plot_frame_with_profiles(img; x=X_POSITION, z=Z_POSITION, title="",
                             colormap=:inferno, colorrange=CM.Makie.automatic) -> Figure

Plot one camera frame `img :: (Nx, Nz)` as a 2D image together with its two
marginal profiles:

* top panel   : ⟨I⟩_z(x) = mean of `img` over z, the horizontal profile;
* right panel : ⟨I⟩_x(z) = mean of `img` over x, the profile along the SG
  deflection axis.

The profile panels share their axes with the image (zooming the image zooms
the profiles).
"""
function plot_frame_with_profiles(img::AbstractMatrix;
                                  x::AbstractVector = X_POSITION,
                                  z::AbstractVector = Z_POSITION,
                                  title::AbstractString = "",
                                  colormap = :inferno,
                                  colorrange = CM.Makie.automatic)
    _check_frame(img, x, z)
    x_mm, z_mm = 1e3 .* x, 1e3 .* z

    fig = CM.Figure(size = (720, 760))
    b = _frame_block!(fig, 1, img, x_mm, z_mm; colormap, colorrange)
    CM.Colorbar(fig[2, 3], b.hm; label = "counts (F − BG)")

    _style_layout!(fig, (2,))
    CM.colgap!(fig.layout, 1, 6)
    isempty(title) || CM.Label(fig[0, :], title; fontsize = 16)
    return fig
end

"""
    plot_F1F2_with_profiles(img1, img2; x=X_POSITION, z=Z_POSITION, title="",
                            colormap=:inferno, common_scale=true) -> Figure

Side-by-side F1 | F2 version of [`plot_frame_with_profiles`](@ref) for the same
frame and current. All image axes are linked (zoom one, zoom all), and so are
the intensity axes of the profiles, so the two states are directly comparable.

* `common_scale=true`  : one colour range (min/max over both frames) and one
  colorbar — relative populations/brightness of F1 and F2 are visible.
* `common_scale=false` : each frame gets its own colour range and colorbar —
  better to see the shape of a weak pattern.
"""
function plot_F1F2_with_profiles(img1::AbstractMatrix, img2::AbstractMatrix;
                                 x::AbstractVector = X_POSITION,
                                 z::AbstractVector = Z_POSITION,
                                 title::AbstractString = "",
                                 colormap = :inferno,
                                 common_scale::Bool = true)
    _check_frame(img1, x, z)
    _check_frame(img2, x, z)
    x_mm, z_mm = 1e3 .* x, 1e3 .* z
    cblabel = "counts (F − BG)"

    fig = CM.Figure(size = (1350, 760))

    if common_scale
        lo1, hi1 = extrema(img1); lo2, hi2 = extrema(img2)
        crange = (min(lo1, lo2), max(hi1, hi2))
        # columns: F1 image | F1 ⟨I⟩_x | F2 image | F2 ⟨I⟩_x | colorbar
        b1 = _frame_block!(fig, 1, img1, x_mm, z_mm; colormap, colorrange = crange, label = "F1")
        b2 = _frame_block!(fig, 3, img2, x_mm, z_mm; colormap, colorrange = crange, label = "F2",
                           zticks = false)
        CM.Colorbar(fig[2, 5], b1.hm; label = cblabel)
        gaps = (6, 24, 6, 12)
        profile_cols = (2, 4)
    else
        # columns: F1 image | F1 ⟨I⟩_x | cb1 | F2 image | F2 ⟨I⟩_x | cb2
        b1 = _frame_block!(fig, 1, img1, x_mm, z_mm; colormap, label = "F1")
        CM.Colorbar(fig[2, 3], b1.hm; label = cblabel)
        b2 = _frame_block!(fig, 4, img2, x_mm, z_mm; colormap, label = "F2")
        CM.Colorbar(fig[2, 6], b2.hm; label = cblabel)
        gaps = (6, 12, 24, 6, 12)
        profile_cols = (2, 5)
    end

    # Same geometry in both blocks; same intensity scale in both sets of profiles
    CM.linkxaxes!(b1.ax_im, b2.ax_im, b1.ax_x, b2.ax_x)
    CM.linkyaxes!(b1.ax_im, b2.ax_im, b1.ax_z, b2.ax_z)
    CM.linkyaxes!(b1.ax_x, b2.ax_x)
    CM.linkxaxes!(b1.ax_z, b2.ax_z)

    _style_layout!(fig, profile_cols)
    for (i, g) in enumerate(gaps)
        CM.colgap!(fig.layout, i, g)
    end
    isempty(title) || CM.Label(fig[0, :], title; fontsize = 16)
    return fig
end

"Figure file name: `<tag>_frame<i>_I<j>_<RUN_STAMP>.<FIG_EXT>` in the run's summary folder."
figure_path(tag, iframe, icurrent) =
    joinpath(summary_dir(ANALYSIS_DIR), "$(tag)_frame$(iframe)_I$(icurrent)_$(RUN_STAMP).$(FIG_EXT)")

# --- Choose what to show -------------------------------------------------------
iframe   = 1          # frame index within the current setting
icurrent = 16          # SG0 current index
let (_, _, nf, nI) = size(data_processed[:F1ProcessedImages])
    1 ≤ iframe ≤ nf   || error("iframe=$iframe out of range 1:$nf")
    1 ≤ icurrent ≤ nI || error("icurrent=$icurrent out of range 1:$nI")
end

img_F1 = data_processed[:F1ProcessedImages][:, :, iframe, icurrent]
img_F2 = data_processed[:F2ProcessedImages][:, :, iframe, icurrent]
I0     = data_processed[:SG0Currents][icurrent]
fig_tag(state) = @sprintf("%s — %s, frame %d, SG0 current = %.3f A", ANALYSIS_DIR, state, iframe, I0)

# (a) one state at a time
state     = :F1        # :F1 or :F2
fig_frame = plot_frame_with_profiles(state === :F1 ? img_F1 : img_F2; title = fig_tag(state))
display(fig_frame)

# (b) F1 and F2 side by side, same frame and current
#     common_scale=true  → shared colour range: compare brightness of F1 vs F2
#     common_scale=false → independent ranges: compare pattern shapes
fig_F1F2 = plot_F1F2_with_profiles(img_F1, img_F2; title = fig_tag("F1 | F2"), common_scale = true)
display(fig_F1F2)

if SAVE_FIG
    for (fig, tag) in ((fig_frame, string(state)), (fig_F1F2, "F1F2"))
        fname = figure_path(tag, iframe, icurrent)
        CM.save(fname, fig; px_per_unit = 4)
        @info "Saved" fname
    end
end


# =============================================================================
# §8  PATTERN WIDTHS ALONG z  (HWHM)
# =============================================================================
# Width of the pattern along the deflection axis z, i.e. of the 2D image
# projected onto z: the profile is ⟨I⟩_x(z), the x-average of the frame.
#
# Profiles are built EXACTLY as in `SG0_framewise_maxima` (5-px median along x,
# mean over x, mean over `nz` z-pixels, cubic smoothing spline with λ0), so the
# widths refer to the same curves as the peak positions of §6.
#
# For one profile with spline S(z), peak z_max and baseline b:
#   level       = b + (S(z_max) − b)/2
#   z_L, z_R    = first crossings of `level` walking outward from z_max
#                 (spline evaluated on a ¼-bin grid, then bisection)
#   HWHM        = (z_R − z_L)/2,   HWHM_L = z_max − z_L,   HWHM_R = z_R − z_max
# HWHM_L ≠ HWHM_R measures the asymmetry of the pattern.
#
# Two estimates per current:
#   (a) framewise  : HWHM of every frame (z_max from §6), then mean ± SEM over
#                    frames;  σ = √(SEM² + σ_pix²/2)  (σ_pix per edge, 2 edges, /2)
#   (b) averaged   : HWHM of the frame-averaged profile (its own z_max).
# (b) ≥ (a) in general: the average also contains the frame-to-frame jitter of
# the peak position σ_jit. For Gaussian lobes  HWHM_b² ≈ HWHM_a² + 2 ln2 σ_jit².
#
# Caveats
# * If the profile has two lobes and the dip between them stays above half of
#   the tallest peak, the HWHM spans both lobes (that is the HWHM of the whole
#   pattern); otherwise it is the HWHM of the tallest lobe only.
# * The smoothing spline lowers and broadens narrow peaks. Measured on a
#   noise-free Gaussian (nz = 1), relative HWHM bias:
#       σ = 0.3 mm : λ0 = 0.1 → +17 %,  0.01 → +3.1 %,  1e-3 → +0.4 %,  1e-4 → +0.04 %
#       σ = 1.0 mm : λ0 = 0.1 → +0.3 %, 0.01 → +0.03 %
#   (the half-maximum search itself is exact: 0.00 % at λ0 = 0). The bias
#   grows quickly as the peak narrows, so widths use their own λ_w = WIDTH_λ0
#   (smaller than the λ0 of §6). The peak positions z_max of §6 are kept: a
#   symmetric smoothing does not move a peak. Check the sensitivity to λ_w and nz.
# * HWHM = (z_R − z_L)/2 does not depend on where z_max sits, but the split
#   HWHM_L / HWHM_R does. For asymmetric lobes, the stronger smoothing of §6
#   moves z_max toward the broad side (synthetic σ_L/σ_R = 0.25/0.40 mm: z_max
#   +35 μm), so treat HWHM_L/HWHM_R as an asymmetry indicator, not as exact.
# * A crossing that is not found inside the camera window gives NaN; NaN
#   frames are left out of the averages (column `nfr` counts the valid ones).

WIDTH_λ0        = 1e-3     # spline smoothing for the widths (see bias note above)
WIDTH_BASELINE  = :edges   # :edges → median of the outer z-bins, :zero → b = 0
WIDTH_EDGE_FRAC = 0.05     # fraction of z-bins used on EACH side for :edges

"""
    z_profiles(data, state, j; nz=1) -> (z_mm, P)

x-averaged, z-binned profiles of all frames of `state` (`:F1`/`:F2`) at current
index `j`. `P :: (Nz ÷ nz, Nframes)`, `z_mm` are the bin centres [mm].
Same preprocessing as `MyExperimentalAnalysis.SG0_framewise_maxima`.
"""
function z_profiles(data, state::Symbol, j::Integer; nz::Integer = 1)
    stack = Float64.(data[Symbol(state, :ProcessedImages)][:, :, :, j])
    stack = mapwindow(median, stack, (5, 1, 1))           # despike along x only
    prof  = dropdims(mean(stack; dims = 1); dims = 1)     # Nz × Nframes
    Nz, nf = size(prof)
    Nz % nz == 0 || error("Nz = $Nz is not divisible by nz = $nz")
    P    = dropdims(mean(reshape(prof, nz, :, nf); dims = 1); dims = 1)
    z_mm = 1e3 .* pixel_positions(Nz, nz, EXP_PIXELSIZE_Z)
    return z_mm, P
end

"Cubic smoothing spline used throughout (same call as in `SG0_framewise_maxima`)."
fit_profile(z, y, λ0::Real) =
    BSplineKit.fit(BSplineOrder(4), z, y, Float64(λ0);
                   weights = MyExperimentalAnalysis.compute_weights(z, Float64(λ0)))

"Baseline of a profile: `:zero` → 0, `:edges` → median of the outer `frac` of the bins on both sides."
function profile_baseline(y::AbstractVector; mode::Symbol = :edges, frac::Real = 0.05)
    mode === :zero && return 0.0
    mode === :edges || throw(ArgumentError("baseline mode must be :edges or :zero"))
    n = max(1, round(Int, frac * length(y)))
    return median(vcat(y[1:n], y[end-n+1:end]))
end

"Position of the tallest maximum of `S` on `[first(z), last(z)]` (grid search + Brent)."
function spline_peak(S, z::AbstractVector; step::Real)
    zz = first(z):step:last(z)
    zg = zz[argmax(S.(zz))]
    lo, hi = max(first(z), zg - step), min(last(z), zg + step)
    return Optim.minimizer(optimize(t -> -S(t), lo, hi, Brent()))
end

"""
    half_crossing(S, z0, level, zlim, dir; step) -> Float64

First `z` beyond `z0` in direction `dir = ±1` where `S(z)` drops below `level`
(grid walk of size `step`, then 50 bisection steps). `NaN` if `zlim` is reached.
"""
function half_crossing(S, z0::Real, level::Real, zlim::Real, dir::Int; step::Real)
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
    hwhm_of_profile(z, y; λ0, zpeak=nothing, baseline=:edges, frac=0.05) -> NamedTuple

HWHM of one profile `y(z)` (see §8 header). If `zpeak` is given it is used as
the peak position (framewise case: z_max of §6), otherwise the spline maximum
is searched. Returns `(; zpeak, height, base, level, zL, zR, hwhm, hwhm_L, hwhm_R, S)`
in mm / counts; `S` is the fitted spline (for plotting).
"""
function hwhm_of_profile(z::AbstractVector, y::AbstractVector; λ0::Real,
                         zpeak = nothing, baseline::Symbol = :edges, frac::Real = 0.05)
    step = (z[2] - z[1]) / 4
    S    = fit_profile(z, y, λ0)
    zp   = isnothing(zpeak) ? spline_peak(S, z; step) : float(zpeak)
    h    = S(zp)
    b    = profile_baseline(y; mode = baseline, frac)
    lvl  = b + (h - b) / 2
    if !(h > b)
        return (; zpeak = zp, height = h, base = b, level = lvl,
                  zL = NaN, zR = NaN, hwhm = NaN, hwhm_L = NaN, hwhm_R = NaN, S)
    end
    zL = half_crossing(S, zp, lvl, first(z), -1; step)
    zR = half_crossing(S, zp, lvl, last(z),  +1; step)
    return (; zpeak = zp, height = h, base = b, level = lvl, zL, zR,
              hwhm = (zR - zL) / 2, hwhm_L = zp - zL, hwhm_R = zR - zp, S)
end

"""
    width_table(data, peaks; nz=1, λ0=0.01, baseline=:edges, frac=0.05) -> NamedTuple

HWHM along z of F1 and F2 for every current, using the framewise peak
positions `peaks.f1_max`, `peaks.f2_max` from §6 (they must come from the same
`nz`; `λ0` here is the smoothing used for the widths and may differ from the
one used for the peaks). Returns `(; F1, F2, σ_pix)`, F1/F2 being NamedTuples with

* `table` — `DataFrame`, one row per current:
  `I0` [A], `B0` [mT], `zpk` (mean framewise peak), `hwhm`, `errhwhm`,
  `hwhm_L`, `hwhm_R` (framewise means), `nfr` (valid frames),
  `zpk_avg`, `hwhm_avg`, `hwhm_L_avg`, `hwhm_R_avg` (frame-averaged profile) — all in mm;
* `hwhm`, `hwhm_L`, `hwhm_R` — framewise values [mm], `(Nframes, Ncurrents)`.
"""
function width_table(data, peaks; nz::Integer = 1, λ0::Real = 0.01,
                     baseline::Symbol = :edges, frac::Real = 0.05)
    nI    = length(data[:SG0Currents])
    σ_pix = 1e3 / sqrt(12) * EXP_PIXELSIZE_Z * nz
    out   = Dict{Symbol,Any}()

    for (state, zmax) in ((:F1, peaks.f1_max), (:F2, peaks.f2_max))
        nf = size(zmax, 1)
        W, WL, WR = fill(NaN, nf, nI), fill(NaN, nf, nI), fill(NaN, nf, nI)
        avg = Vector{NamedTuple}(undef, nI)

        @showprogress "HWHM $state: " for j in 1:nI
            z, P = z_profiles(data, state, j; nz)
            for i in 1:min(nf, size(P, 2))
                isnan(zmax[i, j]) && continue
                r = hwhm_of_profile(z, view(P, :, i); λ0, zpeak = zmax[i, j], baseline, frac)
                W[i, j], WL[i, j], WR[i, j] = r.hwhm, r.hwhm_L, r.hwhm_R
            end
            avg[j] = hwhm_of_profile(z, vec(mean(P; dims = 2)); λ0, baseline, frac)
        end

        w,  sem = nanmean_sem(W)
        wL, _   = nanmean_sem(WL)
        wR, _   = nanmean_sem(WR)
        zpk, _  = nanmean_sem(zmax)
        table = DataFrame(
            I0         = abs.(per_current(data[:SG0Currents], nI)),
            B0         = 1e3 .* per_current(data[:SG0Bz], nI),
            zpk        = zpk,
            hwhm       = w,
            errhwhm    = hypot.(sem, σ_pix / sqrt(2)),
            hwhm_L     = wL,
            hwhm_R     = wR,
            nfr        = vec(count(!isnan, W; dims = 1)),
            zpk_avg    = [a.zpeak  for a in avg],
            hwhm_avg   = [a.hwhm   for a in avg],
            hwhm_L_avg = [a.hwhm_L for a in avg],
            hwhm_R_avg = [a.hwhm_R for a in avg],
        )
        out[state] = (; table, hwhm = W, hwhm_L = WL, hwhm_R = WR)
    end
    return (; F1 = out[:F1], F2 = out[:F2], σ_pix)
end

const WIDTH_TABLE_LABELS = [
    ["I0 Current", "B0 field", "z peak", "HWHM", "HWHM err", "HWHM L", "HWHM R", "frames",
     "z peak avg", "HWHM avg", "HWHM L avg", "HWHM R avg"],
    ["[A]", "[mT]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "",
     "[mm]", "[mm]", "[mm]", "[mm]"],
]

"Pretty-print one table of `width_table` (framewise columns 3–8, frame-averaged 9–12)."
function show_width_table(df::DataFrame; title::AbstractString = "")
    pretty_table(df;
        title,
        formatters    = [fmt__printf("%8.5f", [1]), fmt__printf("%8.3f", [2]),
                         fmt__printf("%8.4f", [3, 4, 5, 6, 7, 9, 10, 11, 12])],
        alignment     = :c,
        column_labels = WIDTH_TABLE_LABELS,
        table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style         = TextTableStyle(
            first_line_column_label = crayon"yellow bold",
            column_label            = crayon"yellow",
            table_border            = crayon"blue bold",
            title                   = crayon"bold red",
        ),
        equal_data_column_widths = true,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically   = false,
    )
end

"Draw profile points, spline, baseline, half-maximum segment and peak of one HWHM result."
function _draw_hwhm!(ax, z, y, r; zoom::Real = 5)
    CM.scatter!(ax, z, y; markersize = 3, color = (:gray40, 0.5))
    zz = range(first(z), last(z); length = 4000)
    CM.lines!(ax, zz, r.S.(zz); color = :red, linewidth = 1.5)
    CM.hlines!(ax, [r.base]; color = :gray50, linestyle = :dot)
    CM.vlines!(ax, [r.zpeak]; color = :black, linestyle = :dash, linewidth = 1)
    if isfinite(r.hwhm)
        CM.lines!(ax, [r.zL, r.zR], [r.level, r.level]; color = :dodgerblue, linewidth = 2.5)
        CM.xlims!(ax, r.zpeak - zoom * r.hwhm_L, r.zpeak + zoom * r.hwhm_R)
    end
    return ax
end

"""
    plot_width_diagnostic(data, peaks, j; iframe=1, nz=1, λ0=0.01, baseline=:edges,
                          frac=0.05, title="") -> Figure

2 × 2 check of the HWHM extraction at current index `j`: rows F1 / F2,
columns frame `iframe` / frame-averaged profile. Grey: data; red: spline;
dotted: baseline; dashed: peak; blue: full width at half maximum.
"""
function plot_width_diagnostic(data, peaks, j::Integer; iframe::Integer = 1,
                               nz::Integer = 1, λ0::Real = 0.01,
                               baseline::Symbol = :edges, frac::Real = 0.05,
                               title::AbstractString = "")
    fig = CM.Figure(size = (1000, 700))
    for (row, (state, zmax)) in enumerate(((:F1, peaks.f1_max), (:F2, peaks.f2_max)))
        z, P = z_profiles(data, state, j; nz)
        y_fr = P[:, iframe]
        y_av = vec(mean(P; dims = 2))
        r_fr = hwhm_of_profile(z, y_fr; λ0, zpeak = zmax[iframe, j], baseline, frac)
        r_av = hwhm_of_profile(z, y_av; λ0, baseline, frac)
        for (col, (y, r, what)) in enumerate(((y_fr, r_fr, "frame $iframe"),
                                              (y_av, r_av, "frame average")))
            ax = CM.Axis(fig[row, col];
                         xlabel = L"z \; (\mathrm{mm})", ylabel = L"\langle I \rangle_x",
                         title = @sprintf("%s, %s:  HWHM = %.4f mm  (L %.4f, R %.4f)",
                                          state, what, r.hwhm, r.hwhm_L, r.hwhm_R))
            _draw_hwhm!(ax, z, y, r)
        end
    end
    isempty(title) || CM.Label(fig[0, :], title; fontsize = 16)
    return fig
end

"""
    plot_width_vs_current(widths; title="") -> Figure

HWHM vs |I_SG0| for F1 and F2: framewise mean ± error (markers with error bars)
and frame-averaged profile (open markers).
"""
function plot_width_vs_current(widths; title::AbstractString = "")
    fig = CM.Figure(size = (800, 500))
    ax  = CM.Axis(fig[1, 1]; xlabel = L"|I_{\mathrm{SG0}}| \; (\mathrm{A})",
                  ylabel = L"\mathrm{HWHM}_z \; (\mathrm{mm})", title)
    for (state, color) in ((:F1, :dodgerblue), (:F2, :orangered))
        t = getproperty(widths, state).table
        CM.errorbars!(ax, t.I0, t.hwhm, t.errhwhm; color, whiskerwidth = 6)
        CM.scatter!(ax, t.I0, t.hwhm; color, label = "$state framewise")
        CM.scatter!(ax, t.I0, t.hwhm_avg; color = :transparent, strokecolor = color,
                    strokewidth = 1.5, marker = :diamond, label = "$state averaged")
    end
    CM.Legend(fig[1, 2], ax; framevisible = false)   # outside: never hides points
    return fig
end

# --- Run -----------------------------------------------------------------------
widths = width_table(data_processed, peaks; nz, λ0 = WIDTH_λ0,
                     baseline = WIDTH_BASELINE, frac = WIDTH_EDGE_FRAC)

for state in (:F1, :F2)
    show_width_table(getproperty(widths, state).table;
        title = "$ANALYSIS_DIR — $state HWHM along z  (nz = $nz, λ_w = $WIDTH_λ0, baseline = $WIDTH_BASELINE)")
end

fig_wdiag = plot_width_diagnostic(data_processed, peaks, icurrent; iframe, nz, λ0 = WIDTH_λ0,
                                  baseline = WIDTH_BASELINE, frac = WIDTH_EDGE_FRAC,
                                  title = fig_tag("HWHM check"))
display(fig_wdiag)

fig_wI = plot_width_vs_current(widths; title = "$ANALYSIS_DIR — HWHM along z")
display(fig_wI)

if SAVE_FIG
    for (fig, fname) in ((fig_wdiag, figure_path("HWHMcheck", iframe, icurrent)),
                         (fig_wI, joinpath(summary_dir(ANALYSIS_DIR),
                                           "HWHM_vs_I_nz$(nz)_$(RUN_STAMP).$(FIG_EXT)")))
        CM.save(fname, fig; px_per_unit = 4)
        @info "Saved" fname
    end
end

if SAVE_TABLE
    stem = joinpath(summary_dir(ANALYSIS_DIR), "hwhm_nz$(nz)_$(RUN_STAMP)")
    CSV.write(stem * "_F1.csv", widths.F1.table)
    CSV.write(stem * "_F2.csv", widths.F2.table)
    jldsave(stem * ".jld2";
            F1_table = widths.F1.table, F2_table = widths.F2.table,
            F1_hwhm = widths.F1.hwhm, F1_hwhm_L = widths.F1.hwhm_L, F1_hwhm_R = widths.F1.hwhm_R,
            F2_hwhm = widths.F2.hwhm, F2_hwhm_L = widths.F2.hwhm_L, F2_hwhm_R = widths.F2.hwhm_R,
            nz, λ0_peaks = λ0, λ0_widths = WIDTH_λ0,
            baseline = String(WIDTH_BASELINE), edge_frac = WIDTH_EDGE_FRAC,
            σ_pix_mm = widths.σ_pix)
    @info "Saved width tables" stem
end
