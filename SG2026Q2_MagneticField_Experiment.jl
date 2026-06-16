# ╔══════════════════════════════════════════════════════════════════════════════
# ║  Stern–Gerlach / CQD  —  Convergence study entry point
# ║  Loads experimental data and analysis summary, then produces a diagnostic
# ║  dual-axis plot of B-field calibration and peak separation vs. coil current.
# ╚══════════════════════════════════════════════════════════════════════════════
# ── Output format & persistence ───────────────────────────────────────────────
using CairoMakie
set_theme!(fonts = (
    size        = 16,
    regular     = "Arial",
    bold        = "Arial Bold",
    italic      = "Arial Italic",
    bold_italic = "Arial Bold Italic",
))
const FIG_EXT  = "png"   # Supported: "pdf" | "svg" | "png"
const SAVE_FIG = true    # Set false for interactive-only sessions
# ── Standard-library imports ──────────────────────────────────────────────────
using LinearAlgebra
using Polynomials
using Statistics
using JLD2, DataFrames
using Dates
using PrettyTables
# ── Timing & run identification ───────────────────────────────────────────────
const T_START   = Dates.now()
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss")
# ── Working directory & output paths ─────────────────────────────────────────
cd(@__DIR__)   # Ensure relative paths resolve from the script's location
const BASE_PATH = raw"F:\SternGerlachExperiments"
# Each run gets its own timestamped subdirectory to avoid overwriting outputs.
const OUTDIR = joinpath(@__DIR__, "data_studies", "CONV_" * RUN_STAMP)
isdir(OUTDIR) || mkpath(OUTDIR)
const HOSTNAME    = gethostname()
const PROGRAM_FILE = @__FILE__
@info "Output directory" path  = OUTDIR
@info "Hostname"         host  = HOSTNAME
@info "Program File"     program_file = PROGRAM_FILE 
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


"""
    find_and_collapse_duplicates(data::Matrix{Float64}; col=1, atol=1e-3) 
        -> (collapsed_data, index_groups)

Detects clusters of near-identical values in `col` of `data`, collapses each
cluster to its mean row, and returns the collapsed matrix plus the original
row index groups (for applying the same operation to associated data).

`atol` is the absolute tolerance for grouping values as "similar".
Returns:
  - `collapsed`    : Matrix with duplicate rows replaced by their mean
  - `index_groups` : Vector of index vectors; singletons for unique rows,
                     multi-element for collapsed groups
"""
function find_and_collapse_duplicates(data::Matrix{Float64};
                                      col::Int    = 1,
                                      rtol::Real  = 0.01)   # 1% default
    N = size(data, 1)
    index_groups = Vector{Vector{Int}}()
    i = 1
    while i <= N
        j = i
        ref = data[i, col]
        # Extend group while next value is within rtol of group start
        while j + 1 <= N && abs(data[j+1, col] - ref) <= rtol * max(abs(ref), abs(data[j+1, col]))
            j += 1
        end
        push!(index_groups, collect(i:j))
        i = j + 1
    end

    collapsed = vcat([mean(data[grp, :], dims=1) for grp in index_groups]...)
    mult_groups = filter(grp -> length(grp) > 1, index_groups)

    return collapsed, index_groups, mult_groups
end



# --- Parameter Selection ---- 
Nz_binning = 2
λ0_smoothing = 0.001
σw_conv = 0.100
ki = 1.5


# ── Load QM simulation peak positions ─────────────────────────────────────────
# F1 and F2 are stored in separate JLD2 files; both are indexed by the same
# canonical key (Nz_binning, σw_conv, λ0_smoothing).

function _load_qm_table(path, z_key)
    jldopen(path, "r") do f
        data = f[JLD2_MyTools.make_keypath_qm(Nz_binning, σw_conv, λ0_smoothing)]
        N    = length(keys(data))
        Ic   = [data[x][:Icoil]   for x in 1:N]
        z    = [data[x][z_key]    for x in 1:N]
        return Ic, z
    end
end

const QM_DIR = joinpath(BASE_PATH, "SIMULATIONS", "2026Q2_SETUP", "QM_T205_8M")

Ic_f1, zf1 = _load_qm_table(joinpath(QM_DIR, "qm_screen_profiles_f1_table.jld2"),
                              :z_max_smooth_spline_mm)
Ic_f2, zf2 = _load_qm_table(joinpath(QM_DIR, "qm_screen_profiles_f2_table.jld2"),
                              :z_max_smooth_spline_mm)

# ── Sanity check: current grids must match exactly ────────────────────────────
if Ic_f1 ≈ Ic_f2
    @info "QM simulation current grids match  ($(length(Ic_f1)) points)"
else
    mismatches = findall(.!(Ic_f1 .≈ Ic_f2))
    @warn "Current grid mismatch at $(length(mismatches)) indices" indices=mismatches
end

# ── Build summary DataFrame ───────────────────────────────────────────────────
df_qm = DataFrame(
    Ic_A   = Ic_f1,
    F2_mm  = zf2,
    F1_mm  = zf1,
    Δ_mm   = zf1 .- zf2,
)

# ── Display ───────────────────────────────────────────────────────────────────
pretty_table(
    df_qm;
    show_row_number_column  = true,
    row_number_column_label = "#",
    column_labels           = ["Ic [A]", "F2 [mm]", "F1 [mm]", "Δ = F1−F2 [mm]"],
    title                   = "QM SIMULATION — Peak positions",
    formatters              = [fmt__printf("%+.4f", [1]), fmt__printf("%+.3f", 4:11)],
    alignment               = :c,
    table_format            = TextTableFormat(borders = text_table_borders__unicode_rounded),
    style                   = TABLE_STYLE,
    equal_data_column_widths = true,
)


# ── Load CQD simulation peak positions ────────────────────────────────────────
# Up (F2) and down (F1) deflections are stored in separate JLD2 files, indexed
# by the canonical CQD key (ki, Nz_binning, λ0_smoothing, σw_conv).

function _load_cqd_table(path, branch, z_key)
    jldopen(path, "r") do f
        data = f[JLD2_MyTools.make_keypath_cqd(branch, ki, Nz_binning, σw_conv, λ0_smoothing,)]
        N    = length(keys(data))
        Ic   = [data[x][:Icoil] for x in 1:N]
        z    = [data[x][z_key]  for x in 1:N]
        return Ic, z
    end
end

const CQD_DIR = joinpath(BASE_PATH, "SIMULATIONS", "2026Q2_SETUP", "CQD_T205_8M")

Ic_cqd_dw, zdw_cqd = _load_cqd_table(
    joinpath(CQD_DIR, "cqd_8000000_dw_profiles.jld2"), :dw, :z_max_smooth_spline_mm)
Ic_cqd_up, zup_cqd = _load_cqd_table(
    joinpath(CQD_DIR, "cqd_8000000_up_profiles.jld2"), :up, :z_max_smooth_spline_mm)

# ── Sanity check: current grids must match ────────────────────────────────────
if Ic_cqd_dw ≈ Ic_cqd_up
    @info "CQD simulation current grids match  ($(length(Ic_cqd_dw)) points)"
else
    mismatches = findall(.!(Ic_cqd_dw .≈ Ic_cqd_up))
    @warn "CQD current grid mismatch at $(length(mismatches)) indices" indices=mismatches
end

# ── Build summary DataFrame ───────────────────────────────────────────────────
df_cqd = DataFrame(
    Ic_A  = Ic_cqd_dw,
    dw_mm = zdw_cqd,
    up_mm = zup_cqd,
    Δ_mm  = zup_cqd .- zdw_cqd,
)

# ── Display ───────────────────────────────────────────────────────────────────
pretty_table(
    df_cqd;
    show_row_number_column   = true,
    row_number_column_label  = "#",
    column_labels            = ["Ic [A]", "dw [mm]", "up [mm]", "Δ = up−dw [mm]"],
    title                    = "CQD SIMULATION — Peak positions",
    formatters               = [fmt__printf("%+.4f", [1]), fmt__printf("%+.3f", 2:4)],
    alignment                = :c,
    table_format             = TextTableFormat(borders = text_table_borders__unicode_rounded),
    style                    = TABLE_STYLE,
    equal_data_column_widths = true,
)


# ── Dataset selection ─────────────────────────────────────────────────────────
data_directories = ["20260529", "20260603"]
data_directory   = data_directories[2]   # Change index to switch dataset

# Canonical paths for raw experiment data and pre-computed analysis summaries
EXPERIMENT_PATH                  = joinpath(BASE_PATH, "EXPERIMENTS")
EXPERIMENT_ANALYSIS_SUMMARY_PATH = joinpath(BASE_PATH, "EXPDATA_ANALYSIS", "summary")

# ── 1. RAW EXPERIMENTAL DATA ───────────────────────────────────────────────────
# ── Load raw experimental data ────────────────────────────────────────────────
exp_data = load(
    joinpath(EXPERIMENT_PATH, data_directory, "data.jld2"),
    "data",
)

# Build a 2-column matrix [current_A | B_field_T], sorted by coil current.
# Column 1: SG1 coil current  [A]
# Column 2: SG1 magnetic field [T] magnetometer 1
# Column 3: SG1 magnetic field [T] magnetometer 2
current_field = let M = hcat(
     exp_data[:SG1currentInA], 
     exp_data[:SG1BfieldInTesla], 
     exp_data[:SG0BfieldInTesla],
    )
    M[sortperm(M[:, 1]), :]
end

# ── 1a. Plot B₀/B₁ field ratio on a log-current axis ─────────────────────────
fig = Figure(size = (700, 420))
ax = Axis(fig[1, 1];
    xlabel             = "Coil current  [A]",
    ylabel             = L"B_\mathrm{SG0}\,/\,B_\mathrm{SG1}",
    title              = "Magnetic field ratio — $(data_directory)",
    xlabelsize         = 16,
    ylabelsize         = 16,
    titlesize          = 15,
    xticklabelsize     = 14,
    yticklabelsize     = 14,
    xscale             = log10,
    xtickformat        = xs -> [L"10^{%$(Int(round(log10(x))))}" for x in xs],
    ytickformat        = xs -> [L"%$(round.(x; sigdigits=2))" for x in xs],
    xminorticksvisible = true,
    yminorticksvisible = true,
    xminorticks        = IntervalsBetween(9),
    yminorticks        = IntervalsBetween(5),
    xgridcolor         = (:gray, 0.35),
    ygridcolor         = (:gray, 0.35),
    xminorgridvisible  = true,
    yminorgridvisible  = true,
    xminorgridcolor    = (:gray, 0.20),
    yminorgridcolor    = (:gray, 0.20),
)
# Mask non-positive currents (log axis requires I > 0).
let m = current_field[:, 1] .> 0
    scatterlines!(ax, current_field[m, 1], current_field[m, 3] ./ current_field[m, 2];
        color      = :steelblue,
        markersize = 8,
        linewidth  = 1.5,
    )
end
fig


# ── 2. COLLAPSE DUPLICATE CURRENT SETPOINTS ───────────────────────────────────
# `find_and_collapse_duplicates` groups rows whose col-1 values agree within
# rtol=1 % and returns the collapsed matrix, the group indices, and a list of
# groups that contained more than one row (mult_groups).
current_field_reduced, index_groups, mult_groups  = find_and_collapse_duplicates(current_field; col=1, rtol=0.01) ;
for grp in mult_groups
    @info "
N  = $(size.(mult_groups,1))
Ic = $(round.(mean(1000*current_field[grp,1]) ;sigdigits=3)) ± $(round.(std(1000*current_field[grp,1]); sigdigits=1)) mA 
B₁ = $(round.(mean(1000*current_field[grp,2]) ;sigdigits=5)) ± $(round.(std(1000*current_field[grp,2]); sigdigits=1)) mT 
B₁ = $(round.(mean(1000*current_field[grp,3]) ;sigdigits=5)) ± $(round.(std(1000*current_field[grp,3]); sigdigits=1)) mT
───────────────────────────────────" 
end


# ── 3. LOAD PEAK-POSITION SUMMARY ─────────────────────────────────────────────
# The summary JLD2 stores framewise peak statistics indexed by a canonical key
# composed of (data_directory, Nz_binning, λ0_smoothing).
exp_analysis = jldopen(
    joinpath(EXPERIMENT_ANALYSIS_SUMMARY_PATH, data_directory,
             data_directory * "_report_summary.jld2"),
    "r",
) do f
    f[JLD2_MyTools.make_keypath_exp(data_directory, Nz_binning, λ0_smoothing)]
end


# Unpack the relevant arrays from the summary entry.
ic     = exp_analysis[:Currents];
f1_raw = exp_analysis[:fw_F1_peak_pos_raw][1];
f2_raw = exp_analysis[:fw_F2_peak_pos_raw][1];
Δ_raw  = exp_analysis[:fw_p2p_sep_raw][1];
center = 0.5 .* (f1_raw .+ f2_raw);

# Report mean ± std of peak positions for repeated setpoints.
for grp in mult_groups
    @info "PEAK POSITION
I₀     = $(round.(mean(1000*ic[grp]) ;sigdigits=4)) ± $(round.(std(1000*ic[grp]); sigdigits=1)) mA
Center = $(round.(mean(center[grp]) ;sigdigits=4)) ± $(round.(std(center[grp]); sigdigits=1)) mm
F1     = $(round.(mean(f1_raw[grp]) ;sigdigits=4)) ± $(round.(std(f1_raw[grp]); sigdigits=1)) mm
F2     = $(round.(mean(f2_raw[grp]) ;sigdigits=4)) ± $(round.(std(f2_raw[grp]); sigdigits=1)) mm
Δ      = $(round.(mean(Δ_raw[grp]) ;sigdigits=4)) ± $(round.(std(Δ_raw[grp]); sigdigits=1)) mm" 
end


# ── 4. MERGE & DISPLAY RAW TABLE ──────────────────────────────────────────────
# Combine coil current, both field readings, and all peak-position statistics
# (with errors) into a single n×11 matrix.  Columns:
#   1  Ic [A]        2  B1 [T]          3  B0 [T]
#   4  F1 [mm]       5  σF1 [mm]
#   6  F2 [mm]       7  σF2 [mm]
#   8  Δ [mm]        9  σΔ [mm]
#  10  Centroid [mm] 11 σCentroid [mm]
data = let ea = exp_analysis
    F1, σF1 = ea[:fw_F1_peak_pos_raw]
    F2, σF2 = ea[:fw_F2_peak_pos_raw]
    p2p, σp2p = ea[:fw_p2p_sep_raw]
    hcat(ea[:Currents], 
         exp_data[:SG1BfieldInTesla], 
         exp_data[:SG0BfieldInTesla],
         F1, σF1,
         F2, σF2, 
         -p2p, σp2p,
         0.5 .* (F1 .+ F2),
         0.5 .* sqrt.(σF1.^2 .+ σF2.^2))
end;

# Collapse duplicates using the same 1 % tolerance as for current_field.
data_red, _, _ = find_and_collapse_duplicates(data; rtol = 0.01);

df_red = DataFrame(
    Ic_A           = data_red[:, 1],
    B1_T           = data_red[:, 2],
    B0_T           = data_red[:, 3],
    F1_mm          = data_red[:, 4],
    σF1_mm         = data_red[:, 5],
    F2_mm          = data_red[:, 6],
    σF2_mm         = data_red[:, 7],
    Δ_mm           = data_red[:, 8],
    σΔ_mm          = data_red[:, 9],
    Centroid_mm    = data_red[:, 10],
    σCentroid_mm   = data_red[:, 11],
);

const COL_LABELS = [
    "Ic [A]", "B1 [T]", "B0 [T]",
    "F1 [mm]", "σF1 [mm]",
    "F2 [mm]", "σF2 [mm]",
    "Δ [mm]",  "σΔ [mm]",
    "Centroid [mm]", "σCentroid [mm]",
];

const TABLE_STYLE = TextTableStyle(
    title                   = crayon"red bold",
    first_line_column_label = crayon"yellow bold",
    column_label            = crayon"yellow",
    table_border            = crayon"blue bold",
);

# Helper: render the n×11 matrix with consistent formatting.
function _print_data_table(mat; title = "EXPERIMENTAL DATA")
    pretty_table(
        mat;
        show_row_number_column   = true,
        row_number_column_label  = "#",
        column_labels            = COL_LABELS,
        title                    = title,
        formatters               = [fmt__printf("%+.4f", 1:3), fmt__printf("%+.3f", 4:11)],
        alignment                = :c,
        table_format             = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style                    = TABLE_STYLE,
        equal_data_column_widths = true,
    )
end

_print_data_table(data_red; title = "EXPERIMENTAL DATA (raw peak positions)")

# ── 5. ZERO-FIELD CENTROID CALIBRATION ────────────────────────────────────────
# The first row of data_red corresponds to the zero-current (I≈0) shot.
# Its centroid gives the beam axis in screen coordinates c₀; all F1/F2 peaks
# are shifted by −c₀ so that positions are relative to the beam axis.
# The calibration uncertainty σc₀ is added in quadrature to σF1 and σF2.
 
c0, c0Err = data_red[1, 10:11] ; 
@info "ZERO-FIELD CALIBRATION
I₀ = $(round.(1000*data_red[1,1]; digits=4)) mA 
B₁ = $(round.(1000*data_red[1,2]; digits=4)) mT 
B₁ = $(round.(1000*data_red[1,3]; digits=4)) mT 
→  c₀ = $(round(c0, digits=3)) ± $(round(c0Err, sigdigits=1)) mm"

# Apply the shift and propagate uncertainties (in-place on a copy).
data_cal = copy(data_red)

data_cal[:,1] = data_cal[:,1] .+ abs(data_cal[1,1])
data_cal[:,2] = data_cal[:,2] .+ abs(data_cal[1,2])
data_cal[:,3] = data_cal[:,3] .+ abs(data_cal[1,3])
# F1 shift + quadrature error propagation
data_cal[:, 4] = data_cal[:, 4] .- c0
data_cal[:, 5] = sqrt.(data_cal[:, 5].^2 .+ c0Err^2)
# F2 shift + quadrature error propagation
data_cal[:, 6] = data_cal[:, 6] .- c0
data_cal[:, 7] = sqrt.(data_cal[:, 7].^2 .+ c0Err^2)

_print_data_table(data_cal; title = "EXPERIMENTAL DATA (calibrated, axis-centred)")

# ── 6. CALIBRATED DATAFRAME ───────────────────────────────────────────────────
# Expose the calibrated result as a tidy DataFrame for downstream analyses
# (plotting, fitting, export, etc.).
 
df_cal = DataFrame(
    Ic_A           = data_cal[:, 1],
    B1_T           = data_cal[:, 2],
    B0_T           = data_cal[:, 3],
    F1_mm          = data_cal[:, 4],
    σF1_mm         = data_cal[:, 5],
    F2_mm          = data_cal[:, 6],
    σF2_mm         = data_cal[:, 7],
    Δ_mm           = data_cal[:, 8],
    σΔ_mm          = data_cal[:, 9],
    Centroid_mm    = data_cal[:, 10],
    σCentroid_mm   = data_cal[:, 11],
);

# ── Global x limits shared by all log-current plots ───────────────────────────
# Only currents where both I > 0 and B1 > 0 (points actually plottable on log axes).
const I_ALL_POS   = vcat(
    df_red.Ic_A[(df_red.Ic_A .> 0) .& (df_red.B1_T .> 0)],
    df_cal.Ic_A[(df_cal.Ic_A .> 0) .& (df_cal.B1_T .> 0)],
)
const X_LO_GLOBAL = floor(Int, log10(minimum(I_ALL_POS)))
const X_HI_GLOBAL = ceil( Int, log10(maximum(I_ALL_POS)))
const X_MIN       = 10.0^X_LO_GLOBAL
const X_MAX       = 10.0^X_HI_GLOBAL/4.5

# ── 7. DIAGNOSTIC DUAL-AXIS PLOT: B-FIELD CALIBRATION & Δz vs. CURRENT ───────
# Left  y-axis : SG1 B-field [T]   — blue markers
# Right y-axis : Peak separation Δz — red markers
# Both share the same x-axis: coil current [A]

let
    # ── Unpack from dataframes ────────────────────────────────────────────────
    # Uncalibrated (raw collapsed)
    I_red      = df_red.Ic_A
    B1_red     = df_red.B1_T
    B0_red     = df_red.B0_T
    Δz_red     = df_red.Δ_mm
    err_dz_red = df_red.σΔ_mm

    # Calibrated (axis-centred, I/B offset-corrected)
    I_cal      = df_cal.Ic_A
    B1_cal     = df_cal.B1_T
    B0_cal     = df_cal.B0_T
    Δz_cal     = df_cal.Δ_mm
    err_dz_cal = df_cal.σΔ_mm

    # ── Guard: include B0 series only if the column is not identically zero ──
    # A column is considered "empty" when every entry rounds to zero within
    # double precision.  If either dataframe has valid B0 data, both are shown
    # (they share the same magnetometer, so all-zero in one implies all-zero
    # in the other, but we check independently to be safe).
    B0_red_valid = !all(iszero, B0_red)
    B0_cal_valid = !all(iszero, B0_cal)
    SHOW_B0      = B0_red_valid || B0_cal_valid
    SHOW_B0 || @warn "B0 column is all-zero in both dataframes — omitting from plot"

    # ── Positive-value masks ──────────────────────────────────────────────────
    m_B1_red  = (I_red .> 0) .& (B1_red .> 0)
    m_B1_cal  = (I_cal .> 0) .& (B1_cal .> 0)
    m_Δz_red  = (I_red .> 0) .& (Δz_red .> 0) .& (Δz_red .- err_dz_red .> 0)
    m_Δz_cal  = (I_cal .> 0) .& (Δz_cal .> 0) .& (Δz_cal .- err_dz_cal .> 0)

    # B0 masks: only positive AND non-zero entries
    m_B0_red  = SHOW_B0 ? ((I_red .> 0) .& (B0_red .> 0)) : falses(length(I_red))
    m_B0_cal  = SHOW_B0 ? ((I_cal .> 0) .& (B0_cal .> 0)) : falses(length(I_cal))

    # ── Tick helpers ──────────────────────────────────────────────────────────
    pow10_fmt = xs -> [L"10^{%$(Int(round(log10(x))))}" for x in xs]

    function decade_range(vals)
        lo = floor(Int, log10(minimum(vals)))
        hi = ceil( Int, log10(maximum(vals)))
        return lo, hi
    end

    function decade_ticks(lo, hi)
        major = 10.0 .^ (lo:hi)
        minor = [b * m for b in major for m in 2:9]
        return major, minor
    end

    # ── x-axis: global decade range shared with all other plots ──────────────
    x_lo, x_hi       = X_LO_GLOBAL, X_HI_GLOBAL
    x_major, x_minor = decade_ticks(x_lo, x_hi)

    # ── Align B and Δz log spans so decade grids coincide ────────────────────
    # Collect all B values (B1 always; B0 only when available) to set a
    # stable left-axis range that accommodates every field series.
    B_vals_list = [B1_red[m_B1_red], B1_cal[m_B1_cal]]
    SHOW_B0 && any(m_B0_red) && push!(B_vals_list, B0_red[m_B0_red])
    SHOW_B0 && any(m_B0_cal) && push!(B_vals_list, B0_cal[m_B0_cal])
    B_vals  = vcat(B_vals_list...)
    Δz_vals = vcat(Δz_red[m_Δz_red], Δz_cal[m_Δz_cal])

    B_lo,  B_hi  = decade_range(B_vals)
    Δz_lo, Δz_hi = decade_range(Δz_vals)

    n_common  = max(B_hi - B_lo, Δz_hi - Δz_lo)

    pad_B_lo  = (n_common - (B_hi  - B_lo))  ÷ 2
    pad_B_hi  =  n_common - (B_hi  - B_lo)  - pad_B_lo
    pad_Δz_lo = (n_common - (Δz_hi - Δz_lo)) ÷ 2
    pad_Δz_hi =  n_common - (Δz_hi - Δz_lo) - pad_Δz_lo

    B_lo_a  = B_lo  - pad_B_lo;   B_hi_a  = B_hi  + pad_B_hi
    Δz_lo_a = Δz_lo - pad_Δz_lo;  Δz_hi_a = Δz_hi + pad_Δz_hi

    B_major,  B_minor  = decade_ticks(B_lo_a,  B_hi_a)
    Δz_major, Δz_minor = decade_ticks(Δz_lo_a, Δz_hi_a)

    # ── Map Δz log-space → B log-space for co-plotting on ax_B ───────────────
    log_offset        = B_lo_a - Δz_lo_a
    scale             = 10.0^log_offset

    Δz_red_mapped     = Δz_red     .* scale
    err_dz_red_mapped = err_dz_red .* scale
    Δz_cal_mapped     = Δz_cal     .* scale
    err_dz_cal_mapped = err_dz_cal .* scale

    Δz_major_mapped = Δz_major .* scale

    # ── Global font sizes (tune these two numbers and everything follows) ─────────
    LABEL_SIZE = 18
    TICK_SIZE  = 15
    LEGEND_SIZE = 15

    update_theme!(
        Axis = (
            xlabelsize      = LABEL_SIZE,
            ylabelsize      = LABEL_SIZE,
            titlesize       = LABEL_SIZE,
            xticklabelsize  = TICK_SIZE,
            yticklabelsize  = TICK_SIZE,
        ),
        Legend = (
            labelsize   = LEGEND_SIZE,
            titlesize   = LEGEND_SIZE,
        ),
    )


    with_theme(
        Theme(
            Axis = (
                xlabelsize     = 26,
                ylabelsize     = 26,
                titlesize      = 30,
                xticklabelsize = 24,
                yticklabelsize = 24,
            ),
            Legend = (
                labelsize  = 18,
                titlesize  = 18,
            ),
        )
    ) do
        let
            # ─────────────────────────────────────────────────────────────────────
            fig = Figure(size = (1500, 750))

            ax_B = Axis(fig[1, 1];
                xlabel             = "Coil current  I  [A]",
                ylabel             = "B-field  [T]",
                xscale             = log10,
                yscale             = log10,
                yticklabelcolor    = :steelblue,
                ylabelcolor        = :steelblue,
                title              = "SG1 calibration & peak separation — $(data_directory)",
                xticks             = x_major,
                xtickformat        = pow10_fmt,
                xticksvisible      = true,
                xminorticksvisible = true,
                xminorticks        = x_minor,
                xminorgridvisible  = true,
                xminorgridcolor    = (:gray, 0.20),
                yticks             = B_major,
                ytickformat        = pow10_fmt,
                yticksvisible      = true,
                yminorticksvisible = true,
                yminorticks        = B_minor,
                ygridvisible       = true,
                ygridcolor         = (:gray, 0.35),
                yminorgridvisible  = true,
                yminorgridcolor    = (:gray, 0.20),
            )

            # ── B1: raw (open, dashed) then calibrated (filled, solid) ───────────
            scatterlines!(ax_B, I_red[m_B1_red], B1_red[m_B1_red];
                color       = (:steelblue, 0.45),
                markersize  = 14,
                linewidth   = 1.2,
                linestyle   = :dash,
                strokewidth = 1.2,
                strokecolor = :steelblue,
            )
            scatterlines!(ax_B, I_cal[m_B1_cal], B1_cal[m_B1_cal];
                color      = :steelblue,
                markersize = 13,
                linewidth  = 1.8,
            )

            # ── B0: raw then calibrated — only when the column has real data ──────
            if SHOW_B0
                any(m_B0_red) && scatterlines!(ax_B, I_red[m_B0_red], B0_red[m_B0_red];
                    color       = (:teal, 0.45),
                    markersize  = 14,
                    linewidth   = 1.2,
                    linestyle   = :dash,
                    marker      = :utriangle,
                    strokewidth = 1.2,
                    strokecolor = :teal,
                )
                any(m_B0_cal) && scatterlines!(ax_B, I_cal[m_B0_cal], B0_cal[m_B0_cal];
                    color      = :teal,
                    markersize = 13,
                    linewidth  = 1.8,
                    marker     = :utriangle,
                )
            end

            # ── Δz: raw (open, dashed) then calibrated (filled, dashed) ──────────
            errorbars!(ax_B, I_red[m_Δz_red], Δz_red_mapped[m_Δz_red], err_dz_red_mapped[m_Δz_red];
                color = (:crimson, 0.35), whiskerwidth = 5)
            scatterlines!(ax_B, I_red[m_Δz_red], Δz_red_mapped[m_Δz_red];
                color       = (:crimson, 0.45),
                markersize  = 14,
                linewidth   = 1.2,
                linestyle   = :dash,
                strokewidth = 1.2,
                strokecolor = :crimson,
            )
            errorbars!(ax_B, I_cal[m_Δz_cal], Δz_cal_mapped[m_Δz_cal], err_dz_cal_mapped[m_Δz_cal];
                color = (:crimson, 0.55), whiskerwidth = 6)
            scatterlines!(ax_B, I_cal[m_Δz_cal], Δz_cal_mapped[m_Δz_cal];
                color      = :crimson,
                markersize = 13,
                linewidth  = 1.8,
                linestyle  = :dash,
            )

            # ── Right spine: phantom axis carrying Δz tick labels ────────────────
            ax_Δz = Axis(fig[1, 1];
                ylabel             = "Δz  [mm]",
                xscale             = log10,
                yscale             = log10,
                yticklabelcolor    = :crimson,
                ylabelcolor        = :crimson,
                yaxisposition      = :right,
                yticks             = (Δz_major_mapped,
                                     [L"10^{%$(Int(round(log10(v))))}" for v in Δz_major]),
                yticksvisible      = true,
                yminorticksvisible = true,
                yminorticks        = Δz_minor .* scale,
                ygridvisible       = false,
                yminorgridvisible  = false,
                xgridvisible       = false,
                xminorgridvisible  = false,
                xticksvisible      = false,
                xticklabelsvisible = false,
                xlabelvisible      = false,
                backgroundcolor    = (:white, 0),
            )

            linkxaxes!(ax_B, ax_Δz)
            xlims!(ax_B,  X_MIN, X_MAX)
            ylims!(ax_B,  10.0^B_lo_a,  10.0^B_hi_a)
            ylims!(ax_Δz, 10.0^B_lo_a,  10.0^B_hi_a)

            # ── Legend: build entries conditionally ───────────────────────────────
            legend_elements = [
                [LineElement(color = :steelblue,           linewidth = 1.8),
                 MarkerElement(color = :steelblue,          marker = :circle,    markersize = 9)],
                [LineElement(color = (:steelblue, 0.45),   linewidth = 1.2, linestyle = :dash),
                 MarkerElement(color = (:steelblue, 0.45), marker = :circle,    markersize = 8,
                              strokewidth = 1.2, strokecolor = :steelblue)],
            ]
            legend_labels = ["B1 (cal.)", "B1 (raw)"]

            if SHOW_B0
                push!(legend_elements,
                    [LineElement(color = :teal,           linewidth = 1.8),
                     MarkerElement(color = :teal,          marker = :utriangle, markersize = 9)])
                push!(legend_elements,
                    [LineElement(color = (:teal, 0.45),   linewidth = 1.2, linestyle = :dash),
                     MarkerElement(color = (:teal, 0.45), marker = :utriangle, markersize = 8,
                                  strokewidth = 1.2, strokecolor = :teal)])
                push!(legend_labels, "B0 (cal.)", "B0 (raw)")
            end

            push!(legend_elements,
                [LineElement(color = :crimson,           linewidth = 1.8, linestyle = :dash),
                 MarkerElement(color = :crimson,          marker = :circle,    markersize = 9)],
                [LineElement(color = (:crimson, 0.45),   linewidth = 1.2, linestyle = :dash),
                 MarkerElement(color = (:crimson, 0.45), marker = :circle,    markersize = 8,
                              strokewidth = 1.2, strokecolor = :crimson)],
            )
            push!(legend_labels, "Δz (cal.)", "Δz (raw)")

            Legend(fig[1, 2], legend_elements, legend_labels; framevisible = false)

            if SAVE_FIG
                fname = joinpath(OUTDIR,
                    "calibration_Bfield_Deltaz_loglog_$(data_directory).$(FIG_EXT)")
                save(fname, fig; px_per_unit = 2)
                @info "Saved figure" path = fname
            end

            fig
        end
    end
end

# ── 8. MAGNETIC FIELD vs. GRADIENT ───────────────────────────────────────────
# Three data sources are overlaid:
#   · Theoretical curve  BvsI / GvsI  from TheoreticalSimulation (dense sweep)
#   · Experimental B1    from df_cal, gradient = man_Gconstant × B1
#   · Experimental B0    from df_cal, gradient = man_Gconstant × B0
#     (B0 is only shown when the column is non-zero — same guard as above)

# ── Dense theoretical sweep ───────────────────────────────────────────────────
const I_SWEEP       = collect(range(0.001, 1.01, 100))
const MAN_GCONSTANT = 386.6171   # [m⁻¹]  apparatus geometry constant

B_theory = TheoreticalSimulation.BvsI.(I_SWEEP)
G_theory = TheoreticalSimulation.GvsI.(I_SWEEP)

# ── Experimental B → gradient via linear calibration ─────────────────────────
B1_exp = df_cal.B1_T
G1_exp = MAN_GCONSTANT .* B1_exp

B0_exp = df_cal.B0_T
G0_exp = MAN_GCONSTANT .* B0_exp

# ── Positive-value masks (log axes require strictly positive values) ──────────
m_theory = (B_theory .> 0) .& (G_theory .> 0)
m_B1     = (B1_exp   .> 0) .& (G1_exp   .> 0)
m_B0     = (B0_exp   .> 0) .& (G0_exp   .> 0)

SHOW_B0_grad = any(m_B0)
SHOW_B0_grad || @warn "B0 column yields no positive gradient values — omitting from B vs G plot"

# ── Linear fit: G = α · B  (origin-constrained, no intercept) ───────────────
B_fit = B_theory[m_theory]
G_fit = G_theory[m_theory]

# Least-squares solution for α with zero intercept: α = (B'B)⁻¹ B'G
α_fit = (B_fit ⋅ G_fit) / (B_fit ⋅ B_fit)

@info "Linear fit  G = α · B" α = round(α_fit, sigdigits=6) units = "T m⁻¹ / T = m⁻¹"

# ── R² for zero-intercept model G = α·B ──────────────────────────────────────
G_pred  = α_fit .* B_fit
SS_res  = sum((G_fit .- G_pred).^2)   # residual sum of squares
SS_tot  = sum(G_fit.^2)               # total SS around zero (no intercept)
R²      = 1.0 - SS_res / SS_tot

@info "Goodness of fit" α=round(α_fit, sigdigits=6) R²=round(R², sigdigits=6)

# Fitted line for plotting
G_fitted = α_fit .* B_fit;


# ── Log-log plot ──────────────────────────────────────────────────────────────
let
    pow10_fmt = xs -> [L"10^{%$(Int(round(log10(x))))}" for x in xs]

    function decade_range(vals)
        lo = floor(Int, log10(minimum(vals)))
        hi = ceil( Int, log10(maximum(vals)))
        return lo, hi
    end

    function decade_ticks(lo, hi)
        major = 10.0 .^ (lo:hi)
        minor = [b * m for b in major for m in 2:9]
        return major, minor
    end

    # Axis limits from the union of all visible series
    B_all = vcat(B_theory[m_theory], B1_exp[m_B1])
    G_all = vcat(G_theory[m_theory], G1_exp[m_B1])
    SHOW_B0_grad && (B_all = vcat(B_all, B0_exp[m_B0]); G_all = vcat(G_all, G0_exp[m_B0]))

    x_lo, x_hi       = decade_range(B_all)
    y_lo, y_hi       = decade_range(G_all)
    x_major, x_minor = decade_ticks(x_lo, x_hi)
    y_major, y_minor = decade_ticks(y_lo, y_hi)

    fig = Figure(size = (1125, 780))

    ax = Axis(fig[1, 1];
        xlabel             = "B-field  [T]",
        ylabel             = L"\nabla B\ \ \mathrm{[T\,m^{-1}]}",
        title              = "Magnetic field vs. gradient — $(data_directory)",
        xlabelsize         = 24,
        ylabelsize         = 24,
        titlesize          = 22,
        xticklabelsize     = 20,
        yticklabelsize     = 20,
        xscale             = log10,
        yscale             = log10,
        xticks             = x_major,
        yticks             = y_major,
        xtickformat        = pow10_fmt,
        ytickformat        = pow10_fmt,
        xminorticks        = x_minor,
        yminorticks        = y_minor,
        xminorticksvisible = true,
        yminorticksvisible = true,
        xminorgridvisible  = true,
        yminorgridvisible  = true,
        xminorgridcolor    = (:gray, 0.20),
        yminorgridcolor    = (:gray, 0.20),
        xgridcolor         = (:gray, 0.35),
        ygridcolor         = (:gray, 0.35),
    )

    # Theoretical curve (no markers — dense enough to read as a line)
    lines!(ax, B_theory[m_theory], G_theory[m_theory];
        color     = :black,
        linewidth = 2.0,
        label     = L"Theory ($\alpha = %$(round(α_fit; digits=4))\mathrm{T}\mathrm{m}^{-1}$, $R^{2}=%$(round(R²; digits=6))$)",
    )

    # B1 experimental points
    scatter!(ax, B1_exp[m_B1], G1_exp[m_B1];
        color       = :transparent,
        strokecolor = :steelblue,
        strokewidth = 2,
        markersize  = 18,
        marker      = :circle,
        label       = L"B_1\ \mathrm{(cal.)}",
    )

    # B0 experimental points — conditional
    if SHOW_B0_grad
        scatter!(ax, B0_exp[m_B0], G0_exp[m_B0];
            color       = :transparent,
            strokecolor = :orange,
            strokewidth = 2,
            markersize  = 18,
            marker      = :utriangle,
            label       = L"B_0\ \mathrm{(cal.)}",
        )
    end

    axislegend(ax; position = :lt, framevisible = false, labelsize = 20)

    if SAVE_FIG
        fname = joinpath(OUTDIR,
            "Bfield_vs_gradient_loglog_$(data_directory).$(FIG_EXT)")
        save(fname, fig; px_per_unit = 2)
        @info "Saved figure" path = fname
    end

    fig
end


# ── 9. PEAK-POSITION & SEPARATION DIAGNOSTIC PLOT ────────────────────────────
# Layout:  [F1 centred (top-left) | F2 centred (top-right)]
#          [      peak-to-peak separation Δz (bottom)      ]
let
    X_MIN = 10e-3
    # ── Unpack from df_cal (experimental) ────────────────────────────────────
    ic     = df_cal.Ic_A
    f1     = -df_cal.F1_mm
    err_f1 = df_cal.σF1_mm
    f2     = df_cal.F2_mm
    err_f2 = df_cal.σF2_mm
    dz     = df_cal.Δ_mm
    err_dz = df_cal.σΔ_mm

    # ── Unpack simulation data ────────────────────────────────────────────────
    mag_qm = 1.20
    ic_qm  = df_qm.Ic_A
    f1_qm  = df_qm.F1_mm ./ mag_qm
    f2_qm  = -df_qm.F2_mm ./ mag_qm
    dz_qm  = abs.(df_qm.Δ_mm) ./ mag_qm

    mag_cqd = 1.17
    ic_cqd = df_cqd.Ic_A
    up_cqd = df_cqd.up_mm ./ mag_cqd
    dw_cqd = -df_cqd.dw_mm ./ mag_cqd
    dz_cqd = abs.(df_cqd.Δ_mm) ./ mag_cqd

    i_threshold = 20e-3   # minimum current [A] included in all masks

    # ── Positive-value masks (required for log-log axes) ─────────────────────
    m1 = (ic .> i_threshold) .& (f1 .> 0) .& (f1 .- err_f1 .> 0)
    m2 = (ic .> i_threshold) .& (f2 .> 0) .& (f2 .- err_f2 .> 0)
    m3 = (ic .> i_threshold) .& (dz .> 0) .& (dz .- err_dz .> 0)

    m1_qm  = (ic_qm  .> i_threshold) .& (f1_qm  .> 0)
    m2_qm  = (ic_qm  .> i_threshold) .& (f2_qm  .> 0)
    m3_qm  = (ic_qm  .> i_threshold) .& (dz_qm  .> 0)
    m_cqd1 = (ic_cqd .> i_threshold) .& (up_cqd .> 0)
    m_cqd2 = (ic_cqd .> i_threshold) .& (dw_cqd .> 0)
    m_cqd3 = (ic_cqd .> i_threshold) .& (dz_cqd .> 0)

    # ── x ticks from global decade range ─────────────────────────────────────
    x_lo, x_hi = X_LO_GLOBAL, X_HI_GLOBAL
    x_major    = 10.0 .^ (x_lo:x_hi)
    x_minor    = [b * m for b in x_major for m in 2:9]

    # LaTeX power-of-10 tick formatter
    pow10_fmt = xs -> [L"10^{%$(Int(round(log10(x))))}" for x in xs]

    # ── Shared log-axis keyword builder ──────────────────────────────────────
    function log_axis_kw(; extra...)
        return (
            xscale             = log10,
            yscale             = log10,
            xticks             = x_major,
            xticksvisible      = true,
            xticklabelsvisible = true,
            xtickformat        = pow10_fmt,
            ytickformat        = pow10_fmt,
            xminorticks        = x_minor,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible  = true,
            yminorgridvisible  = true,
            xminorgridcolor    = (:gray, 0.20),
            yminorgridcolor    = (:gray, 0.20),
            extra...
        )
    end

    FS_tick  = 18   # tick-label font size
    FS_label = 24   # axis-label font size
    FS_title = 14   # panel title font size
    FS_leg   = 22   # legend font size
    C_EXP    = :black   # unified experimental data color

    fig = Figure(size = 1.5 .* (900, 650))

    # Shared x-axis label sits below both bottom panels.
    Label(fig[3, 1:2], "Coil current  [A]"; tellwidth = false, fontsize = FS_label)

    # ── Top-left: F1 peak position ────────────────────────────────────────────
    ax1 = Axis(fig[1, 1];
        log_axis_kw(
            title              = "F1 peak position",
            ylabel             = L"F_1 - c_0\ \mathrm{[mm]}",
            titlesize          = FS_title,
            ylabelsize         = FS_label,
            xticklabelsize     = FS_tick,
            yticklabelsize     = FS_tick,
        )...
    )
    errorbars!(ax1, ic[m1], f1[m1], err_f1[m1];
        color = (C_EXP, 0.5), whiskerwidth = 6)
    scatterlines!(ax1, ic[m1], f1[m1];
        color = C_EXP, markercolor = :transparent,
        strokecolor = C_EXP, strokewidth = 2,
        markersize = 12, linewidth = 1.5, label = "Exp.")
    lines!(ax1, ic_qm[m1_qm], f1_qm[m1_qm];
        color = :blue, linewidth = 1.8, label = "QM")
    lines!(ax1, ic_cqd[m_cqd1], up_cqd[m_cqd1];
        color = :red, linewidth = 1.8, linestyle = :dash, label = "CQD")

    # ── Top-right: F2 peak position ───────────────────────────────────────────
    ax2 = Axis(fig[1, 2];
        log_axis_kw(
            title              = "F2 peak position",
            ylabel             = L"F_2 - c_0\ \mathrm{[mm]}",
            titlesize          = FS_title,
            ylabelsize         = FS_label,
            xticklabelsize     = FS_tick,
            yticklabelsize     = FS_tick,
        )...
    )
    errorbars!(ax2, ic[m2], f2[m2], err_f2[m2];
        color = (C_EXP, 0.5), whiskerwidth = 6)
    scatterlines!(ax2, ic[m2], f2[m2];
        color = C_EXP, markercolor = :transparent,
        strokecolor = C_EXP, strokewidth = 2,
        markersize = 12, linewidth = 1.5, label = "Exp.")
    lines!(ax2, ic_qm[m2_qm], f2_qm[m2_qm];
        color = :blue, linewidth = 1.8, label = "QM")
    lines!(ax2, ic_cqd[m_cqd2], dw_cqd[m_cqd2];
        color = :red, linewidth = 1.8, linestyle = :dash, label = "CQD")

    # ── Bottom: peak-to-peak separation ──────────────────────────────────────
    ax3 = Axis(fig[2, 1:2];
        log_axis_kw(
            title              = "Peak-to-peak separation",
            ylabel             = L"\Delta z\ \mathrm{[mm]}",
            titlesize          = FS_title,
            ylabelsize         = FS_label,
            xticklabelsize     = FS_tick,
            yticklabelsize     = FS_tick,
        )...
    )
    errorbars!(ax3, ic[m3], dz[m3], err_dz[m3];
        color = (C_EXP, 0.5), whiskerwidth = 6)
    scatterlines!(ax3, ic[m3], dz[m3];
        color = C_EXP, markercolor = :transparent,
        strokecolor = C_EXP, strokewidth = 2,
        markersize = 12, linewidth = 1.5, label = "Exp.")
    lines!(ax3, ic_qm[m3_qm], dz_qm[m3_qm];
        color = :blue, linewidth = 1.8, label = "QM")
    lines!(ax3, ic_cqd[m_cqd3], dz_cqd[m_cqd3];
        color = :red, linewidth = 1.8, linestyle = :dash, label = "CQD")

    axislegend(ax1; position = :lt, framevisible = false, labelsize = FS_leg)
    axislegend(ax2; position = :lt, framevisible = false, labelsize = FS_leg)
    axislegend(ax3; position = :lt, framevisible = false, labelsize = FS_leg)

    xlims!(ax1, X_MIN, X_MAX)
    xlims!(ax2, X_MIN, X_MAX)
    xlims!(ax3, X_MIN, X_MAX)

    rowgap!(fig.layout, 1, 8)
    colgap!(fig.layout, 12)

    if SAVE_FIG
        fname = joinpath(OUTDIR,
            "peak_positions_sep_loglog_$(data_directory).$(FIG_EXT)")
        save(fname, fig; px_per_unit = 2)
        @info "Saved figure" path = fname
    end

    fig
end

# ── 10. PEAK-POSITION & SEPARATION — LINEAR SCALE ─────────────────────────────
# Same layout as section 9 but on linear axes; all data points included (no masks).
let
    X_MAX = 1.1
    # ── Unpack from df_cal (experimental) ────────────────────────────────────
    ic     = df_cal.Ic_A
    f1     = -df_cal.F1_mm
    err_f1 = df_cal.σF1_mm
    f2     = df_cal.F2_mm
    err_f2 = df_cal.σF2_mm
    dz     = df_cal.Δ_mm
    err_dz = df_cal.σΔ_mm

    # ── Unpack simulation data ────────────────────────────────────────────────
    mag_qm = 1.20
    ic_qm  = df_qm.Ic_A
    f1_qm  = df_qm.F1_mm  ./ mag_qm
    f2_qm  = -df_qm.F2_mm ./ mag_qm
    dz_qm  = abs.(df_qm.Δ_mm) ./ mag_qm

    mag_cqd = 1.17
    ic_cqd  = df_cqd.Ic_A
    up_cqd  = df_cqd.up_mm  ./ mag_cqd
    dw_cqd  = -df_cqd.dw_mm ./ mag_cqd
    dz_cqd  = abs.(df_cqd.Δ_mm) ./ mag_cqd

    FS_tick  = 18
    FS_label = 24
    FS_title = 14
    FS_leg   = 22
    C_EXP    = :black   # unified experimental data color

    # ── Shared linear-axis keyword builder ───────────────────────────────────
    function lin_axis_kw(; extra...)
        return (
            xticks             = 0:0.2:1.0,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorticks        = IntervalsBetween(5),
            yminorticks        = IntervalsBetween(5),
            xminorgridvisible  = true,
            yminorgridvisible  = true,
            xminorgridcolor    = (:gray, 0.20),
            yminorgridcolor    = (:gray, 0.20),
            extra...
        )
    end

    fig = Figure(size = 1.5 .* (900, 650))

    Label(fig[3, 1:2], "Coil current  [A]"; tellwidth = false, fontsize = FS_label)

    # ── Top-left: F1 peak position ────────────────────────────────────────────
    ax1 = Axis(fig[1, 1];
        lin_axis_kw(
            title          = "F1 peak position",
            ylabel         = L"F_1 - c_0\ \mathrm{[mm]}",
            titlesize      = FS_title,
            ylabelsize     = FS_label,
            xticklabelsize = FS_tick,
            yticklabelsize = FS_tick,
        )...
    )
    errorbars!(ax1, ic, f1, err_f1; color = (C_EXP, 0.5), whiskerwidth = 6)
    scatterlines!(ax1, ic, f1;
        color = C_EXP, markercolor = :transparent,
        strokecolor = C_EXP, strokewidth = 2,
        markersize = 12, linewidth = 1.5, label = "Exp.")
    lines!(ax1, ic_qm, f1_qm; color = :blue, linewidth = 1.8, label = "QM")
    lines!(ax1, ic_cqd, up_cqd;
        color = :red, linewidth = 1.8, linestyle = :dash, label = "CQD")

    # ── Top-right: F2 peak position ───────────────────────────────────────────
    ax2 = Axis(fig[1, 2];
        lin_axis_kw(
            title          = "F2 peak position",
            ylabel         = L"F_2 - c_0\ \mathrm{[mm]}",
            titlesize      = FS_title,
            ylabelsize     = FS_label,
            xticklabelsize = FS_tick,
            yticklabelsize = FS_tick,
        )...
    )
    errorbars!(ax2, ic, f2, err_f2; color = (C_EXP, 0.5), whiskerwidth = 6)
    scatterlines!(ax2, ic, f2;
        color = C_EXP, markercolor = :transparent,
        strokecolor = C_EXP, strokewidth = 2,
        markersize = 12, linewidth = 1.5, label = "Exp.")
    lines!(ax2, ic_qm, f2_qm; color = :blue, linewidth = 1.8, label = "QM")
    lines!(ax2, ic_cqd, dw_cqd;
        color = :red, linewidth = 1.8, linestyle = :dash, label = "CQD")

    # ── Bottom: peak-to-peak separation ──────────────────────────────────────
    ax3 = Axis(fig[2, 1:2];
        lin_axis_kw(
            title          = "Peak-to-peak separation",
            ylabel         = L"\Delta z\ \mathrm{[mm]}",
            titlesize      = FS_title,
            ylabelsize     = FS_label,
            xticklabelsize = FS_tick,
            yticklabelsize = FS_tick,
        )...
    )
    errorbars!(ax3, ic, dz, err_dz; color = (C_EXP, 0.5), whiskerwidth = 6)
    scatterlines!(ax3, ic, dz;
        color = C_EXP, markercolor = :transparent,
        strokecolor = C_EXP, strokewidth = 2,
        markersize = 12, linewidth = 1.5, label = "Exp.")
    lines!(ax3, ic_qm, dz_qm; color = :blue, linewidth = 1.8, label = "QM")
    lines!(ax3, ic_cqd, dz_cqd;
        color = :red, linewidth = 1.8, linestyle = :dash, label = "CQD")

    axislegend(ax1; position = :lt, framevisible = false, labelsize = FS_leg)
    axislegend(ax2; position = :lt, framevisible = false, labelsize = FS_leg)
    axislegend(ax3; position = :lt, framevisible = false, labelsize = FS_leg)

    xlims!(ax1, X_MIN, X_MAX)
    xlims!(ax2, X_MIN, X_MAX)
    xlims!(ax3, X_MIN, X_MAX)

    rowgap!(fig.layout, 1, 8)
    colgap!(fig.layout, 12)

    if SAVE_FIG
        fname = joinpath(OUTDIR,
            "peak_positions_sep_linear_$(data_directory).$(FIG_EXT)")
        save(fname, fig; px_per_unit = 2)
        @info "Saved figure" path = fname
    end

    fig
end
