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


# --- Parameter Selection ---- 
Nz_binning = 2
λ0_smoothing = 0.01


# ── Dataset selection ─────────────────────────────────────────────────────────
data_directories = ["20260529", "20260603"]
data_directory   = data_directories[2]   # Change index to switch dataset

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
    f[JLD2_MyTools.make_keypath_exp(data_directory, Nz_binning, λ0_smoothing)]
end


C0 = 0.5 * (exp_analysis[:fw_F1_peak_pos_raw][1][1] .+
        exp_analysis[:fw_F2_peak_pos_raw][1][1])
ic     = exp_analysis[:Currents]
f1     = -(exp_analysis[:fw_F1_peak_pos_raw][1] .- C0)
err_f1 = abs.(exp_analysis[:fw_F1_peak_pos_raw][2])
f2     =   exp_analysis[:fw_F2_peak_pos_raw][1] .- C0
err_f2 = abs.(exp_analysis[:fw_F2_peak_pos_raw][2])
dz     = - exp_analysis[:fw_p2p_sep_raw][1]
err_dz = abs.(exp_analysis[:fw_p2p_sep_raw][2])

# ── Diagnostic dual-axis plot: B-field calibration & Δz vs. current ──────────
# Left  y-axis : SG1 B-field [T]   — blue markers
# Right y-axis : Peak separation Δz — red markers
# Both share the same x-axis: coil current [A]
let
    I  = current_field[:, 1]
    B  = current_field[:, 2]

    Δz_vec     = dz     isa AbstractVector ? dz     : fill(dz,     length(I))
    err_dz_vec = err_dz isa AbstractVector ? err_dz : fill(err_dz, length(I))

    # ── Positive-value masks ──────────────────────────────────────────────────
    m_B  = (I .> 0) .& (B .> 0)
    m_Δz = (I .> 0) .& (Δz_vec .> 0) .& (Δz_vec .- err_dz_vec .> 0)

    # ── Tick formatter ────────────────────────────────────────────────────────
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

    # ── x axis ───────────────────────────────────────────────────────────────
    x_lo, x_hi       = decade_range(I[m_B])
    x_major, x_minor = decade_ticks(x_lo, x_hi)

    # ── Align y axes in log space ─────────────────────────────────────────────
    B_lo,  B_hi  = decade_range(B[m_B])
    Δz_lo, Δz_hi = decade_range(Δz_vec[m_Δz])

    span_B   = B_hi  - B_lo
    span_Δz  = Δz_hi - Δz_lo
    n_common = max(span_B, span_Δz)

    pad_B_lo  = (n_common - span_B)  ÷ 2;  pad_B_hi  = n_common - span_B  - pad_B_lo
    pad_Δz_lo = (n_common - span_Δz) ÷ 2;  pad_Δz_hi = n_common - span_Δz - pad_Δz_lo

    B_lo_a  = B_lo  - pad_B_lo;   B_hi_a  = B_hi  + pad_B_hi
    Δz_lo_a = Δz_lo - pad_Δz_lo;  Δz_hi_a = Δz_hi + pad_Δz_hi

    B_major,  B_minor  = decade_ticks(B_lo_a,  B_hi_a)
    Δz_major, Δz_minor = decade_ticks(Δz_lo_a, Δz_hi_a)

    # ── Normalize Δz into B log-space for plotting on ax_B ───────────────────
    # Map Δz log-space [Δz_lo_a, Δz_hi_a] → B log-space [B_lo_a, B_hi_a]
    # so both series share the same y axis physically.
    # Transformation: log10(Δz) → B_lo_a + (log10(Δz) - Δz_lo_a) / n_common * n_common
    # Since both spans equal n_common, this simplifies to a linear shift in log space:
    #   log10(Δz_mapped) = log10(Δz) - Δz_lo_a + B_lo_a
    log_offset = B_lo_a - Δz_lo_a          # additive shift in log10 space
    Δz_mapped     = Δz_vec     .* 10.0^log_offset
    err_dz_mapped = err_dz_vec .* 10.0^log_offset

    # ── Right-spine tick labels: show original Δz values ─────────────────────
    # These are the B-axis values that correspond to each Δz decade label
    Δz_major_mapped = Δz_major .* 10.0^log_offset

    # ─────────────────────────────────────────────────────────────────────────
    fig = Figure(size = (900, 480))

    # ── Single axis: both series share log x and log y ───────────────────────
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

    scatterlines!(ax_B, I[m_B], B[m_B];
        color      = :steelblue,
        markersize = 8,
        linewidth  = 1.5,
    )

    # Plot Δz on the same axis using the log-shifted values
    errorbars!(ax_B, I[m_Δz], Δz_mapped[m_Δz], err_dz_mapped[m_Δz];
        color        = (:crimson, 0.5),
        whiskerwidth = 6,
    )
    scatterlines!(ax_B, I[m_Δz], Δz_mapped[m_Δz];
        color      = :crimson,
        markersize = 8,
        linewidth  = 1.5,
        linestyle  = :dash,
    )

    # ── Right spine: Δz tick labels via overlay axis (no grid, no data) ───────
    ax_Δz = Axis(fig[1, 1];
        ylabel             = "Δz  [mm]",
        xscale             = log10,
        yscale             = log10,
        yticklabelcolor    = :crimson,
        ylabelcolor        = :crimson,
        yaxisposition      = :right,
        # Ticks at mapped positions but labelled with original Δz values
        yticks             = (Δz_major_mapped,
                              [L"10^{%$(Int(round(log10(v))))}" for v in Δz_major]),
        yticksvisible      = true,
        yminorticksvisible = true,
        yminorticks        = Δz_minor .* 10.0^log_offset,
        ygridvisible       = false,
        yminorgridvisible  = false,
        xticksvisible      = false,
        xticklabelsvisible = false,
        xlabelvisible      = false,
        backgroundcolor    = (:white, 0),
    )

    # ── Lock limits ───────────────────────────────────────────────────────────
    ylims!(ax_B,  10.0^B_lo_a,  10.0^B_hi_a)
    ylims!(ax_Δz, 10.0^B_lo_a,  10.0^B_hi_a)   # same physical limits as ax_B

    # ── Legend ────────────────────────────────────────────────────────────────
    Legend(fig[1, 2],
        [
            [LineElement(color = :steelblue, linewidth = 1.5),
             MarkerElement(color = :steelblue, marker = :circle, markersize = 8)],
            [LineElement(color = :crimson, linewidth = 1.5, linestyle = :dash),
             MarkerElement(color = :crimson, marker = :circle, markersize = 8)],
        ],
        ["B-field [T]", "Δz  [mm]"],
        framevisible = false,
    )

    if SAVE_FIG
        fname = joinpath(OUTDIR,
            "calibration_Bfield_Deltaz_loglog_$(data_directory).$(FIG_EXT)")
        save(fname, fig; px_per_unit = 2)
        @info "Saved figure" path = fname
    end

    fig
end

# ── Peak-position & separation diagnostic plot ────────────────────────────────
# Layout:  [F1 centred | F2 centred]
#          [   p2p separation  ]
let
    # ── Positive-value masks (required for log-log axes) ─────────────────────
    m1 = (ic .> 0) .& (f1 .> 0) .& (f1 .- err_f1 .> 0)
    m2 = (ic .> 0) .& (f2 .> 0) .& (f2 .- err_f2 .> 0)
    m3 = (ic .> 0) .& (dz .> 0) .& (dz .- err_dz .> 0)

    # ── Explicit x tick positions (computed before masking) ───────────────────
    ic_pos  = ic[ic .> 0]
    x_lo    = floor(log10(minimum(ic_pos)))
    x_hi    = ceil( log10(maximum(ic_pos)))
    x_major = 10.0 .^ (x_lo:x_hi)                        # one tick per decade
    x_minor = [b * m for b in x_major for m in 2:9]      # 2×–9× within each decade

    # ── Tick formatter: renders values as LaTeX powers of 10 ─────────────────
    # e.g. 0.01 → "10^{-2}",  1.0 → "10^{0}",  100.0 → "10^{2}"
    pow10_fmt = xs -> [L"10^{%$(Int(round(log10(x))))}" for x in xs]

    fig = Figure(size = (900, 650))

    Label(fig[3, 1:2], "Coil current  [A]";
          tellwidth = false, fontsize = 14)

    # ── Shared log-axis keyword builder ──────────────────────────────────────
    function log_axis_kw(; extra...)
        return (
            xscale             = log10,
            yscale             = log10,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible  = true,
            yminorgridvisible  = true,
            xminorticks        = IntervalsBetween(9),
            yminorticks        = IntervalsBetween(9),
            xminorgridcolor    = (:gray, 0.20),
            yminorgridcolor    = (:gray, 0.20),
            xtickformat        = pow10_fmt,   # LaTeX powers of 10 on x
            ytickformat        = pow10_fmt,   # LaTeX powers of 10 on y — matches x font
            extra...
        )
    end

    # ── Top-left: F1 peak position (centred) ─────────────────────────────────
    ax1 = Axis(fig[1, 1];
        log_axis_kw(
            title              = "F1 peak position (centred)",
            ylabel             = "z − c₀  [mm]",
            xticks             = x_major,
            xticksvisible      = true,
            xticklabelsvisible = true,
            xtickformat        = pow10_fmt,
            xminorticks        = x_minor,
            xminorticksvisible = true,
        )...
    )
    errorbars!(ax1, ic[m1], f1[m1], err_f1[m1];
        color = (:steelblue, 0.5), whiskerwidth = 6)
    scatterlines!(ax1, ic[m1], f1[m1];
        color = :steelblue, markersize = 7, linewidth = 1.5)

    # ── Top-right: F2 peak position (centred) ────────────────────────────────
    ax2 = Axis(fig[1, 2];
        log_axis_kw(
            title              = "F2 peak position (centred)",
            ylabel             = "z − c₀  [mm]",
            xticks             = x_major,
            xticksvisible      = true,
            xticklabelsvisible = true,
            xtickformat        = pow10_fmt,
            xminorticks        = x_minor,
            xminorticksvisible = true,
        )...
    )
    errorbars!(ax2, ic[m2], f2[m2], err_f2[m2];
        color = (:crimson, 0.5), whiskerwidth = 6)
    scatterlines!(ax2, ic[m2], f2[m2];
        color = :crimson, markersize = 7, linewidth = 1.5)

    # ── Bottom: peak-to-peak separation ──────────────────────────────────────
    ax3 = Axis(fig[2, 1:2];
        log_axis_kw(
            title       = "Peak-to-peak separation",
            ylabel      = "Δz  [mm]",
            xticks      = x_major,
            xticksvisible      = true,
            xticklabelsvisible = true,
            xtickformat = pow10_fmt,
            xminorticks = x_minor,
        )...
    )
    errorbars!(ax3, ic[m3], dz[m3], err_dz[m3];
        color = (:darkorange, 0.5), whiskerwidth = 6)
    scatterlines!(ax3, ic[m3], dz[m3];
        color = :darkorange, markersize = 7, linewidth = 1.5)

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



data_directories = ["20260529", "20260603"]
data_directory = data_directories[2]

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


