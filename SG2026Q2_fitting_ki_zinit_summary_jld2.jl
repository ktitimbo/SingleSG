# =============================================================================
#  Stern–Gerlach experiment — fitting the CQD induction coefficient kᵢ
#  from the per-directory summary JLD2s (loop over DATA_DIRECTORIES)
#  — WITH A VERTICAL OFFSET z_init IN THE MODEL  (v01, from SG2026Q2_fitting_ki_summary_jld2.jl v02)
# -----------------------------------------------------------------------------
#  Kelvin Titimbo — California Institute of Technology — November 2026
#
#  WHAT IS NEW WITH RESPECT TO SG2026Q2_fitting_ki_summary_jld2.jl
#  ------------------------------------------------------------------
#  Both calibrated models get one extra free parameter, a vertical shift:
#
#      z_model(I) = (z_init + m(I))/s     m = CQD ki_itp(I, kᵢ)  or  QM zqm(I)
#
#  z_init (mm) absorbs an origin that is not the true zero-deflection position,
#  e.g. the deflection already present at the lowest (non-zero) current where
#  the origin is set (zero-current midpoint), or a residual camera
#  offset. It is fitted separately for CQD and QM (same freedom for both):
#    • s(kᵢ, z_init) closed form on the tail with the shifted model  z_init + m ≈ s·z;
#    • z_init and kᵢ minimise the log10 loss on the fit subset (nested Brent);
#    • QM: z_init from the fit subset, then s on the tail (shifted model).
#  Settings: FIT_ZINIT, ZINIT_BOUNDS. FIT_ZINIT = false reproduces the
#  no-offset script. Parameter counts in χ²_red/AIC/BIC: CQD 3, QM 2.
#  z_init is added to fig002 (scan), the figures' labels, the tables, the
#  archives and the combined table/figure.
#  All figures are drawn with CairoMakie (not Plots/GR): PNG at PX_PER_UNIT
#  pixels per unit, LaTeX labels via MathTeXEngine, theme set in section 0.
#
#  PURPOSE
#  -------
#  Same methodology as `SG2026Q2_fitting_ki_independent.jl`, but the
#  experimental curve is NOT the combined/averaged one (`data_averaged_*.jld2`).
#  Instead the F=1 peak position z(I) is read directly from the analysis
#  report of one experiment directory,
#
#      EXPDATA_ANALYSIS/summary/<DATA_DIR>/<DATA_DIR>_report_summary.jld2
#
#  for a chosen profile-extraction configuration (NZ_EXP, λ0_EXP), stored under
#  the key  "<DATA_DIR>/nz=<NZ_EXP>/λ0=<λ0_EXP>"  (see JLD2_MyTools.make_keypath_exp).
#
#  The analysis is repeated for every directory in DATA_DIRECTORIES (function
#  `analyze_directory`, one sub-folder of results per directory); a directory
#  that fails is reported and skipped. The simulation grids and the (I, kᵢ)
#  interpolant are built once. Finally all kᵢ and scale factors are collected
#  in one table with their inverse-variance weighted mean (section ALL).
#
#  CQD predictions exist only on a discrete grid of (current, kᵢ). The script
#  builds a smooth 2D interpolant over that grid, fits (kᵢ, s) to the measured
#  points — s being a global calibration (magnification) factor fixed on the
#  high-current tail — and fits the same kind of s for QM.
#
#  CALIBRATION CONVENTION (used everywhere below)
#  ----------------------------------------------
#      model(I) / s  ≈  experiment(I)
#  The experimental data are never rescaled; each model is divided by its own
#  tail-fitted factor:  s_CQD (from `fit_cqd_ki_scale*`) and s_QM (from
#  `fit_qm_scale*`). `scale_inv = 1/s` is the factor that would multiply the
#  data instead (old convention), kept for reference only.
#
#  FROM THE SUMMARY FILE TO z(I)
#  -----------------------------
#  (identical to the per-directory treatment in SG2026Q2_experiment_summary.jl)
#    1. raw F=1 / F=2 peak positions x ± δx (camera frame, `PEAK_KEY_F1/F2`)
#       are converted to mm with the magnification M ± δM of the directory:
#           z = x/M,   δz² = (δx/M)² + (x·δM/M²)²
#    2. origin z0 = mean F1/F2 midpoint of the repeats at I = 0 (lowest current
#       if there is none), as in SG2026Q2_experiment_summary.jl; error = larger
#       of the propagated √(Σδ²)/n and the scatter std/√n of the repeats.
#       The file's `centroid_fw_mm` (midpoint averaged over I ≥ 20 mA) is not used.
#    3. F=1 deflection  z_F1 = −(z_F1,raw − z0),  δz² = δz_F1² + δz0²
#    4. sorted by current, cut at I ≥ I_THRESHOLD, non-positive z removed
#    5. δI from the file (`ErrorCurrentsPhys`, δI_MODE = :phys) or a fraction of
#       I (δI_MODE = :frac, as in the averaged-curve script), and propagated:
#           δz_total² = δz² + (dz/dI · δI)²
#       with dz/dI from a weighted log–log smoothing spline of the points.
#  Unlike the averaged-curve script there is no dense "continuous" curve: the
#  same table of measured points is used for every fit and figure.
#
#  WORKFLOW (run top-to-bottom in one Julia session)
#  --------------------------------------------------
#    0. Setup: packages, output directory, local modules, helper functions.
#    1. Load simulation grids
#         • QM  : z_max(I) for one analysis configuration (nz, σw, λ0).
#         • CQD : z_max(I, kᵢ) for every available kᵢ, same configuration.
#    2. Build interpolants (once, common to all directories)
#         • zqm(I)        : 1D cubic spline of the QM curve.
#         • ki_itp(I, kᵢ) : 2D cubic spline of the CQD grid → fit surface.
#       Sanity figures fig001–fig002.
#    LOOP over DATA_DIRECTORIES → analyze_directory(dir), sections 3–A:
#    3. Load the experimental points z(I) ± δz from the summary JLD2; fig001.
#    4. Tail-convergence study: (kᵢ, s_CQD) and s_QM vs tail length; spread
#       over the plateau quoted as tail-choice systematic.           fig002
#    5. Final fits with formal uncertainties at the chosen `N_TAIL`
#       (`fit_cqd_ki_scale_with_error`, `fit_qm_scale_with_error`).
#    6. Diagnostics: overlay (fig003), relative error + pulls (fig004),
#       publication overlay (fig005).
#    7. Publication figures vs current and vs gradient, goodness-of-fit table
#       (χ², AIC/BIC, …) and diagnostic panels (fig006).
#    S/A. Summary and per-directory archive.
#    ALL. Combined table of kᵢ and s over the directories, weighted mean of kᵢ,
#         fig_ki_all_directories, fig_summary_all_directories (all data + median curves).
#
#  INPUTS  (external — this script does not create them)
#  ----------------------------------------------------
#    Local modules (./Modules/):
#         TheoreticalSimulation.jl   — physics helpers (GvsI gradient map)
#         DataReading.jl             — experimental analysis report locator
#         MyExperimentalAnalysis.jl  — magnification factors per dataset
#         JLD2_MyTools.jl            — JLD2 key/keypath helpers for QM, CQD, EXP
#    Data files (under BASE_PATH):
#         SIMULATIONS/2026Q2_SETUP/QM_T205_8M_Bexp/qm_screen_profiles_f1_table.jld2
#         SIMULATIONS/2026Q2_SETUP/CQD_T205_8M_Bexp/cqd_8M_up_profiles.jld2
#         EXPDATA_ANALYSIS/summary/<DATA_DIR>/<DATA_DIR>_report_summary.jld2
#
#  OUTPUTS  (RUN_OUTDIR = ./data_studies/FIT2026_ki_scale_zinit_dirs_nz<NZ>_l<λ0>_<timestamp>/)
#  ------------------------------------------------------------------
#    In RUN_OUTDIR (once):
#      fig001  CQD z_max(kᵢ) slices vs QM at selected currents
#      fig002  interpolated (I, kᵢ) surface and contour
#      fig_ki_all_directories       kᵢ, s and z_init per directory, weighted mean of kᵢ
#      fig_summary_all_directories  all directories' data + CQD/QM curves at the median kᵢ, z_init, s
#      ki_all_directories.csv       one row per directory (kᵢ, s, errors, χ², ΔAIC, …)
#      SG_ki_scale_all_directories.jld2  the same table + weighted mean
#    In RUN_OUTDIR/<DATA_DIR>/ (per directory; figures .<FIG_EXT>, default png):
#      fig001  CQD family + QM + experiment (log–log)
#      fig002_tail_convergence  kᵢ, s, z_init and R² vs calibration-tail length
#      fig003  data + calibrated QM + calibrated CQD (diagnostic)
#      fig004  relative error (%) and pulls vs current, both models
#      fig005  data + calibrated models (publication style)
#      fig006  goodness-of-fit diagnostic panels
#      single_SG_comparison(.png|.svg)      points vs current
#      single_SG_comparison_vsg(.png|.svg)  points vs field gradient
#    Tables (.csv)
#      tail_convergence_cqd.csv   scan of (kᵢ, s_CQD, tail RMSE, R²) vs n_tail
#      tail_convergence_qm.csv    scan of (s_QM, tail RMSE, R²) vs n_tail
#      model_residuals_scaled.csv data, both calibrated models, relative
#                                 errors and pulls
#      data_exp.csv               experiment (I, δI, G, δG, z, δz)
#      data_sim.csv               calibrated model curves on the dense scan
#      goodness_of_fit.csv        metric table of section 7
#      model_data.csv             data and calibrated models at the measured currents
#    Archive: SG_ki_scale_results.jld2 (same layout as the averaged-curve script,
#             plus meta/data_dir, settings/nz_exp, settings/λ0_exp, …)
#
#  CONVENTIONS
#  -----------
#    • kᵢ is stored in "micro-units": physical value = (reported kᵢ) × 10⁻⁶.
#    • Experimental data matrices are N×4 with columns [I, δI, z, δz]
#      (I in A, z in mm), sorted by increasing I.
#    • All kᵢ fits minimise a residual in log10(z). "Errors" are documented
#      per function: `ki_err`/`scale_err` are Student-t half-widths from the
#      `_with_error` fitters; `tail_rmse`/`ki_err` of the plain fitters are
#      goodness-of-fit RMSEs in mm, not error bars.
#    • Residual sign convention everywhere: model − experiment.
#    • Tail lengths (N_TAIL, NTAIL_LIST) and fit windows (N_FRONT, N_BACK) are
#      counted in measured POINTS (≈ 25 above 20 mA), not in dense-grid rows.
# =============================================================================

# -----------------------------------------------------------------------------
# 0) Packages and global setup
# -----------------------------------------------------------------------------
# Figures: CairoMakie (vector-quality PNG/PDF/SVG). LaTeX labels (L"...") are
# rendered by MathTeXEngine; theme_latexfonts() gives Computer Modern text.
using CairoMakie
const PX_PER_UNIT = 3            # PNG resolution: pixels per figure unit (size=(900,800) → 2700×2400 px)
CairoMakie.activate!(type = "png", px_per_unit = PX_PER_UNIT)
set_theme!(merge(Theme(
    fontsize = 15,
    Axis = (
        xgridvisible = true, ygridvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        xminorticksvisible = true, yminorticksvisible = true,
        xgridcolor = (:black, 0.12), ygridcolor = (:black, 0.12),
        xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05),
        titlefont = :regular,
    ),
    Axis3 = (titlefont = :regular,),
    Legend = (framevisible = true, framecolor = :transparent,
              backgroundcolor = (:white, 0.85), titlefont = :regular,
              patchsize = (22, 12), rowgap = 0),
    Colorbar = (minorticksvisible = false,),
), theme_latexfonts()))
FIG_EXT = "png"   # "png" | "pdf" | "svg"
# Aesthetics and output formatting
using Colors, ColorSchemes
using LaTeXStrings, Printf, PrettyTables
# Time-stamping / logging
using Dates
const T_START = Dates.now();
# Numerical tools
using LinearAlgebra
using Dierckx                                  # Spline1D / Spline2D
using Optim                                    # Brent 1D bounded minimisation
using Statistics, StatsBase, Distributions     # mean/median/var, TDist, Chisq
# Data handling
using OrderedCollections
using CSV, DataFrames, JLD2
# BLAS threads (the script itself is single-threaded)
LinearAlgebra.BLAS.set_num_threads(2)
@info "BLAS threads"  count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()


cd(@__DIR__);
const BASE_PATH = raw"F:\SternGerlachExperiments";

# -----------------------------------------------------------------------------
# User-adjustable analysis settings (collected here; referenced below)
# -----------------------------------------------------------------------------
# Experimental data sets: each directory is analysed on its own (sections 3–A),
# then all results are collected in one table (section ALL).
#   EXPDATA_ANALYSIS/summary/<dir>/<dir>_report_summary.jld2
const DATA_DIRECTORIES = [
                        #   "20260819",
                        #   "20260821", "20260826", "20260827", "20260831", "20260902",
                          "20260903",
                        #   "20261105", "20261111", "20261112", "20261113", "20261116", "20261118"
                        ]
const NZ_EXP   = 2              # z-bins of the experimental profile extraction   (file: meta/nz)
const λ0_EXP   = 0.01           # raw smoothing parameter of the experiment       (file: meta/λ0)
const PEAK_KEY_F1 = :fw_F1_peak_pos_raw   # raw F=1 peak position (camera frame) and its error
const PEAK_KEY_F2 = :fw_F2_peak_pos_raw   # raw F=2 peak position (used for the origin)
# Analysis configuration shared by the QM and CQD tables
const NX_FIXED  = 128       # x-bins (bookkeeping only)
const NZ_FIXED  = 2         # z-bins used in profile extraction (simulation)
const σW_FIXED  = 0.125     # (mm) Gaussian smoothing width of the profiles
const λ0_FIXED  = 0.01      # raw smoothing parameter (simulation)
const λ0_SPLINE = 0.001     # spline smoothing parameter (bookkeeping only)
# Experimental data
const I_THRESHOLD  = 0.020  # (A) lowest current kept for all fits
const δI_MODE = :frac       # :frac → δI = δI_FRAC·I | :phys → δI = ErrorCurrentsPhys from the file
const δI_FRAC = 0.001       # δI placeholder used when δI_MODE = :frac
# kᵢ-loss subset (counted in points; capped at the number of points of each directory)
const FIT_KI_MODE   = :full # :full | :low | :high | :low_high
const N_FRONT_PTS   = 12    # points in the low-current window
const N_BACK_PTS    = 8     # points in the high-current window
# Calibration tail
const TAIL_SCALE_MODE = :log     # :linear | :log | :legacy — same for CQD and QM
const I_PLATEAU       = 0.650    # (A) plateau = tails confined to I_min ≥ I_PLATEAU;
                                 #     the final tail = all points with I ≥ I_PLATEAU
# Vertical offset of the models:  z_model(I) = (z_init + m(I))/s
const FIT_ZINIT    = true             # false → z_init ≡ 0 (reproduces the no-offset script)
const ZINIT_BOUNDS = (-0.009, 0.009)  # (mm) search bracket of z_init; the lower side is
                                      #      also limited by z_init + m > 0 on the fit points
const ZINIT_FIT_BOUNDS = FIT_ZINIT ? ZINIT_BOUNDS : nothing
const N_PAR_CQD = FIT_ZINIT ? 3 : 2   # free parameters (kᵢ, s, z_init) for χ²_red, AIC, BIC
const N_PAR_QM  = FIT_ZINIT ? 2 : 1   # (s, z_init)
# Figures
const SHOW_FIGS = true      # display every figure (false: only save them — faster for many directories)

const RUN_STAMP  = Dates.format(T_START, "yyyymmddTHHMMSSsss");
const RUN_OUTDIR = joinpath(@__DIR__, "data_studies",
                            "FIT2026_ki_scale_zinit_dirs_nz$(NZ_EXP)_l$(@sprintf("%.3f", λ0_EXP))_" * RUN_STAMP);
isdir(RUN_OUTDIR) || mkpath(RUN_OUTDIR);
@info "Created output directory" RUN_OUTDIR
@info "Running on host" hostname = gethostname()

# Local modules (see INPUTS in the header)
include("./Modules/TheoreticalSimulation.jl");
TheoreticalSimulation.set_magnetic_field!(:calibration)       # back to the default (apparatus manual)
@info "Magnetic-field model" mode = TheoreticalSimulation.magnetic_field_mode()              # which one is active
include("./Modules/DataReading.jl");
include("./Modules/MyExperimentalAnalysis.jl");
include("./Modules/JLD2_MyTools.jl");


##################################################################################################
JLD2_MyTools.save_script_copy(RUN_OUTDIR; script_path=@__FILE__, timestamp=RUN_STAMP)
##################################################################################################

# =============================================================================
# -----------------------------------------------------------------------------
TheoreticalSimulation.set_magnetic_field!(:calibration)  # Calibration/Experimental field
table_qm_path = joinpath(BASE_PATH, "SIMULATIONS", "2026Q2_SETUP", "QM_T205_8M_Bexp",
                         "qm_screen_profiles_f1_table.jld2");
table_cqd_path = joinpath(BASE_PATH, "SIMULATIONS", "2026Q2_SETUP", "CQD_T205_8M_Bexp",
                          "cqd_8M_up_profiles.jld2");
summary_path(dir) = joinpath(BASE_PATH, "EXPDATA_ANALYSIS", "summary", dir, dir * "_report_summary.jld2")
missing_dirs = [d for d in DATA_DIRECTORIES if !isfile(summary_path(d))]
isempty(missing_dirs) || @warn "Summary file missing — these directories will be skipped" missing_dirs
# -----------------------------------------------------------------------------
# =============================================================================

# =============================================================================
# 0a) Small numerical helpers
# =============================================================================

"""
    logspace10(lo, hi; n=50)
 
Return `n` points logarithmically spaced (base 10) between `lo` and `hi`,
inclusive. Used for dense current scans on log axes.
"""
logspace10(lo, hi; n=50) = 10.0 .^ range(log10(lo), log10(hi); length=n)
 
"""
    log_mask(x, y)
 
Boolean mask selecting entries safe for log–log plotting/fitting: both `x` and
`y` strictly positive and finite. Applied element-wise.
"""
log_mask(x, y) = (x .> 0) .& (y .> 0) .& isfinite.(x) .& isfinite.(y)
 
"""
    relerr(model, exp)
 
Dimensionless relative (fractional) error `(model - exp) / exp`, element-wise.
Positive = model above experiment. Measures the *size* of a discrepancy.
"""
relerr(model, exp) = (model .- exp) ./ exp
 
"""
    pull(model, exp, σ_exp; σ_model=nothing)
 
Normalised residual `(model - exp) / σ_tot`, element-wise, with
`σ_tot = sqrt(σ_exp² + σ_model²)` (`σ_model` may be a scalar, a vector, or
`nothing`). Same sign convention as `relerr`. Measures the *significance* of
a discrepancy: for a correct model and correct errors the pulls have zero
mean, unit width and no structure vs the independent variable;
`mean(pull.^2)` is the reduced χ² (only if the points are independent).
"""
function pull(model, exp, σ_exp; σ_model = nothing)
    σ_tot = σ_model === nothing ? σ_exp : sqrt.(σ_exp .^ 2 .+ σ_model .^ 2)
    (model .- exp) ./ σ_tot
end
 
"""
    fmt(v, d)
 
Round `v` to `d` significant digits (for figure labels).
"""
fmt(v, d) = round(v, sigdigits = d)


# =============================================================================
# 0b) Fitting routines — models with a vertical offset z_init
#
#   Every model curve enters the fits as
#
#       z_model(I) = (z_init + m(I)) / s      m = CQD ki_itp(I, kᵢ)  or  QM zqm(I)
#
#   i.e. the offset is added to the SIMULATED curve before the calibration
#   factor is applied (s·z ≈ z_init + m). In data units the shift is z_init/s.
#   z_init (mm, model units) is a rigid vertical shift of the F=1 deflection. It
#   absorbs an origin z0 that is not the true zero-deflection position — e.g.
#   if a directory has no I = 0 point the origin is taken at the lowest
#   *non-zero* current, where the beam is already (slightly) deflected — or any
#   residual offset of the camera frame. It is relevant at low currents only
#   (z of a few |z_init|); at high current it is negligible next to m.
#
#   fit_cqd_ki_scale            — (kᵢ, z_init, s_CQD), point estimates only (fast; used in the scan)
#   fit_qm_scale                — (z_init, s_QM),      point estimates only (fast; used in the scan)
#   fit_cqd_ki_scale_with_error — (kᵢ, z_init, s_CQD) with t- and profile intervals
#   fit_qm_scale_with_error     — (z_init, s_QM)      with t- and profile intervals
#
# Common ingredients:
#   • tail       = last `n_tail` rows of the full data matrix (high currents).
#                  For a given z_init, s is closed form on the tail with the
#                  SHIFTED MODEL  m' = z_init + m ≈ s·z  (data unchanged;
#                  scale_mode, identical for CQD and QM):
#         :log     log10 s = ⟨log10 m' − log10 z⟩
#         :linear  s = ⟨m',m'⟩/⟨z, m'⟩
#         :legacy  s = ⟨m', z⟩/⟨z, z⟩
#   • loss       = log10-space (weighted) residual of  (z_init + m)/s  vs z on the
#                  selected fit subset; a non-positive prediction → loss = Inf.
#   • s is profiled out (closed form), z_init by an inner bounded Brent search
#     in `zinit_bounds`, kᵢ by the outer Brent search: the joint
#     (kᵢ, z_init, s) problem is a nested pair of 1D searches.
#   • z_init has to be fixed by the LOW-current points (it is invisible on the
#     tail), so the QM fit now also uses the fit subset, not only the tail.
#   • `zinit_bounds = nothing` → z_init ≡ 0: the fitters then reproduce the
#     no-offset script (SG2026Q2_fitting_ki_summary_jld2.jl) exactly.
# =============================================================================

"""
    _scale_from_tail(m, z; scale_mode, w=nothing) -> (s, var_log10_s)

Closed-form calibration factor `s` such that `m/s ≈ z` on the tail points
(`m` model — already shifted, `z_init + m` —, `z` raw data), and the conditional
variance of `u = log10 s` from the scatter of the tail.

- `:log`    → `u = ⟨log10 m − log10 z⟩` (weighted by `w` if given); `Var(u) = var(d)/n` or `1/Σw`.
- `:linear` → `z ≈ a·m`, `s = 1/a`; `Var(a)` from the regression residuals.
- `:legacy` → `m ≈ s·z`; `Var(s)` from the residuals.
`w` is used by `:log` only. With a single point the variance is `NaN`.
"""
function _scale_from_tail(m, z; scale_mode::Symbol, w = nothing)
    ln10 = log(10.0)
    n    = length(m)
    if scale_mode === :log
        d = log10.(m) .- log10.(z)
        if w === nothing
            u = mean(d);               var_u = n > 1 ? var(d) / n : NaN
        else
            u = sum(w .* d) / sum(w);  var_u = 1 / sum(w)
        end
        return (10.0^u, var_u)
    elseif scale_mode === :linear                  # z ≈ a·m , s = 1/a
        a     = dot(z, m) / dot(m, m)
        var_a = n > 1 ? sum(abs2, z .- a .* m) / (n - 1) / dot(m, m) : NaN
        return (1 / a, var_a / (a * ln10)^2)
    elseif scale_mode === :legacy                  # m ≈ s·z
        s     = dot(m, z) / dot(z, z)
        var_s = n > 1 ? sum(abs2, m .- s .* z) / (n - 1) / dot(z, z) : NaN
        return (s, var_s / (s * ln10)^2)
    else
        error("scale_mode must be :log, :linear or :legacy, got $scale_mode")
    end
end

"""
    _minimize_1d(f, lo, hi) -> (x̂, f(x̂), result)

Bounded Brent minimisation of a scalar function on `[lo, hi]`.
"""
function _minimize_1d(f, lo, hi)
    r = optimize(f, float(lo), float(hi), Brent())
    return (Optim.minimizer(r), Optim.minimum(r), r)
end

"""
    _zinit_step(ẑ, Z)

Central-difference step for z_init (mm): `cbrt(eps)·max(|ẑ|, min Z)`. Not
clipped to `zinit_bounds` (the model is defined outside them).
"""
_zinit_step(ẑ, Z) = cbrt(eps(Float64)) * max(abs(ẑ), minimum(Z))

"""
    fit_cqd_ki_scale(data_org, selected_points, ki_list, ki_range;
                     n_tail, itp=ki_itp, scale_mode=:linear, zinit_bounds=nothing)

Three-parameter fit of the induction coefficient `kᵢ`, the vertical offset
`z_init` (mm) and the calibration factor `s`, so that the **calibrated CQD
model** `(z_init + m)/s` matches the raw experiment `z`.

For candidate `(kᵢ, z_init)`, `s(kᵢ, z_init)` is the closed-form factor that
makes `(z_init + itp(I,kᵢ))/s` agree with the tail `z` (last `n_tail` rows of
`data_org`); `(kᵢ, z_init)` then minimise the log10-space MSE between
`z` and `(z_init + itp(I,kᵢ))/s` on `selected_points` — z_init by an inner Brent
search in `zinit_bounds`, kᵢ by the outer one. `zinit_bounds = nothing` fixes
`z_init = 0` (no-offset fit).

# Returns
NamedTuple: `ki`, `z_init`, `scale`, `scale_inv`, `ki_err` (linear-space RMSE in
mm on `selected_points`, goodness of fit — **not** an error bar), `tail_rmse`
(mm), `r2_coeff` (linear R² on the full `data_org`), `loss_log`, `tail_range`,
`converged`, `result`.
"""
function fit_cqd_ki_scale(data_org, selected_points, ki_list, ki_range;
                          n_tail::Int,
                          itp = ki_itp,
                          scale_mode::Symbol = :linear,
                          zinit_bounds = nothing)

    ki_start, ki_stop = ki_range
    N = size(data_org, 1)
    @assert 1 ≤ n_tail ≤ N "n_tail must be between 1 and size(data_org,1) = $N"
    fitz = zinit_bounds !== nothing

    # tail used for the calibration (last n_tail rows of the full data)
    I_tail = data_org[end-n_tail+1:end, 1]
    z_tail = data_org[end-n_tail+1:end, 3]

    # subset used for the (kᵢ, z_init) loss
    I_fit = selected_points[:, 1]
    z_fit = selected_points[:, 3]

    # closed-form scale s(kᵢ, z_init):  (z_init + model_tail) / s ≈ z_tail
    function scale_for(ki, zi)
        m  = zi .+ itp.(I_tail, Ref(ki))
        ok = isfinite.(m) .& (m .> 0) .& (z_tail .> 0)
        any(ok) || return NaN
        first(_scale_from_tail(m[ok], z_tail[ok]; scale_mode = scale_mode))
    end

    # composite objective: log-space MSE of (z_init + m)/s on the fit subset
    function loss(ki, zi)
        s = scale_for(ki, zi)
        (isfinite(s) && s > 0) || return Inf
        m  = itp.(I_fit, Ref(ki))
        ok = isfinite.(m)
        any(ok) || return Inf
        p = (zi .+ m[ok]) ./ s
        all(>(0), p) || return Inf
        mean(abs2, log10.(p) .- log10.(z_fit[ok]))
    end

    # z_init profiled out for every kᵢ (inner search), then the outer search over kᵢ
    best_zi(ki) = fitz ? _minimize_1d(zi -> loss(ki, zi), zinit_bounds[1], zinit_bounds[2])[1:2] :
                         (0.0, loss(ki, 0.0))

    res    = optimize(ki -> best_zi(ki)[2], ki_list[ki_start], ki_list[ki_stop], Brent())
    k_fit  = Optim.minimizer(res)
    zi_fit = best_zi(k_fit)[1]
    s_fit  = scale_for(k_fit, zi_fit)
    model(Ivec) = (zi_fit .+ itp.(Ivec, Ref(k_fit))) ./ s_fit

    # diagnostics: calibrated model vs raw data
    rmse_fit  = sqrt(mean(abs2, model(I_fit)  .- z_fit))          # mm
    rmse_tail = sqrt(mean(abs2, model(I_tail) .- z_tail))         # mm

    z_all = data_org[:, 3]
    p_all = model(data_org[:, 1])
    r2    = 1 - sum(abs2, p_all .- z_all) / sum(abs2, z_all .- mean(z_all))

    return (
        ki         = k_fit,
        z_init     = zi_fit,
        scale      = s_fit,
        scale_inv  = 1 / s_fit,
        ki_err     = rmse_fit,
        tail_rmse  = rmse_tail,
        r2_coeff   = r2,
        loss_log   = Optim.minimum(res),
        tail_range = (first(I_tail), last(I_tail)),
        converged  = Optim.converged(res),
        result     = res,
    )
end

"""
    fit_qm_scale(data_org, selected_points, zqm; n_tail, scale_mode=:linear, zinit_bounds=nothing)

Offset `z_init` and calibration factor `s` such that the **calibrated QM
model** `(z_init + zqm(I))/s` matches the raw experiment. `s(z_init)` comes from
the tail with the shifted model (last `n_tail` rows of `data_org`), `z_init` minimises the
log10-space MSE on `selected_points` (Brent in `zinit_bounds`).
`zinit_bounds = nothing` → `z_init = 0` and `s` from the tail alone (as in the
no-offset script).

Returns `(z_init, scale, scale_inv, tail_rmse, r2_coeff, loss_log, tail_range)`.
"""
function fit_qm_scale(data_org, selected_points, zqm;
                      n_tail::Int, scale_mode::Symbol = :linear, zinit_bounds = nothing)
    N = size(data_org, 1)
    @assert 1 ≤ n_tail ≤ N "n_tail must be between 1 and size(data_org,1) = $N"
    fitz = zinit_bounds !== nothing

    I_tail = data_org[end-n_tail+1:end, 1]
    z_tail = data_org[end-n_tail+1:end, 3]
    m_tail = zqm.(I_tail)
    I_fit  = selected_points[:, 1]
    z_fit  = selected_points[:, 3]
    m_fit  = zqm.(I_fit)
    okf    = isfinite.(m_fit)

    function scale_for(zi)
        mt = zi .+ m_tail
        ok = isfinite.(mt) .& (mt .> 0) .& (z_tail .> 0)
        any(ok) || return NaN
        first(_scale_from_tail(mt[ok], z_tail[ok]; scale_mode = scale_mode))
    end
    function loss(zi)
        s = scale_for(zi)
        (isfinite(s) && s > 0 && any(okf)) || return Inf
        p = (zi .+ m_fit[okf]) ./ s
        all(>(0), p) || return Inf
        mean(abs2, log10.(p) .- log10.(z_fit[okf]))
    end

    zi = fitz ? _minimize_1d(loss, zinit_bounds[1], zinit_bounds[2])[1] : 0.0
    s  = scale_for(zi)
    model(Ivec) = (zi .+ zqm.(Ivec)) ./ s

    rmse_tail = sqrt(mean(abs2, model(I_tail) .- z_tail))
    z_all = data_org[:, 3]
    p_all = model(data_org[:, 1])
    r2    = 1 - sum(abs2, p_all .- z_all) / sum(abs2, z_all .- mean(z_all))

    return (z_init = zi, scale = s, scale_inv = 1/s, tail_rmse = rmse_tail,
            r2_coeff = r2, loss_log = loss(zi), tail_range = (first(I_tail), last(I_tail)))
end

# -----------------------------------------------------------------------------
# Shared building blocks of the two `_with_error` fitters
# -----------------------------------------------------------------------------

"""
    _fd_step(k, lo, hi; rel=cbrt(eps()), absmin=1e-12)

Central-difference step for a parameter `k` inside `[lo, hi]`: relative step
`rel·max(|k|,1)`, never larger than half the distance to the nearest bound.
"""
function _fd_step(k, lo, hi; rel = cbrt(eps(Float64)), absmin = 1e-12)
    hh   = max(absmin, rel * max(abs(k), 1.0))
    room = min(k - lo, hi - k)
    room > 0 ? min(hh, 0.5 * room) : absmin
end

"""
    _profile_interval(loss, x̂, xmin, xmax, target; grid=400)

Likelihood-profile interval of a 1-parameter loss: starting from the optimum
`x̂`, walk towards each bound on a grid of `grid` points until `loss` first
exceeds `target`, then locate the crossing by bisection. If no crossing is
found on a side, that side returns the bound itself.

Returns `(x_lo, x_hi)`.
"""
function _profile_interval(loss, x̂, xmin, xmax, target; grid::Int = 400)
    function bracket_side(dir::Int)
        g     = range(x̂, dir > 0 ? xmax : xmin; length = grid)
        prevx = first(g)
        prevL = loss(prevx)
        for x in Iterators.drop(g, 1)
            L = loss(x)
            if isfinite(L) && (L > target) && isfinite(prevL) && (prevL <= target)
                return (prevx, x)
            end
            prevx, prevL = x, L
        end
        return nothing
    end
    function bisect_cross(a, b; maxiter = 80, tol = 1e-10)
        lo, hi = a, b
        for _ in 1:maxiter
            mid  = (lo + hi) / 2
            fmid = loss(mid) - target
            if !isfinite(fmid)
                hi = mid
                continue
            end
            fmid > 0 ? (hi = mid) : (lo = mid)
            abs(hi - lo) <= tol * max(1.0, abs(mid)) && return (lo + hi) / 2
        end
        return (lo + hi) / 2
    end
    left_br  = bracket_side(-1)
    right_br = bracket_side(+1)
    x_lo = left_br  === nothing ? xmin : bisect_cross(left_br[1],  left_br[2])
    x_hi = right_br === nothing ? xmax : bisect_cross(right_br[1], right_br[2])
    return (x_lo, x_hi)
end

"""
    fit_cqd_ki_scale_with_error(itp, data_org, selected_points;
                                n_tail, bounds, zinit_bounds=nothing, scale_mode=:log,
                                conf=0.95, use_Zse=false, profile=true, profile_grid=400)

Uncertainty-aware companion of `fit_cqd_ki_scale`: fits `kᵢ`, the offset
`z_init` and the calibration factor `s` (calibrated model `(z_init + m)/s` ≈ raw
data `z`) and returns formal uncertainties on all three.

Model. For candidate `(kᵢ, z_init)` the scale `s(kᵢ, z_init)` is fixed by the
tail with the shifted model `z_init + m ≈ s·z` (closed form, see `scale_mode`). `(kᵢ, z_init)`
minimise the (optionally weighted) log10-space RSS on `selected_points`,
`RSS = Σ wᵢ (log10[(z_init + itp(Iᵢ,kᵢ))/s] − log10 Zᵢ)²` (nested Brent
searches). With `use_Zse=true`, `wᵢ = (Zᵢ ln10 / σZᵢ)²`; otherwise all 1.

Uncertainties (composite residuals, so the Jacobian includes ∂s/∂kᵢ and ∂s/∂z_init):
- covariance of `(kᵢ, z_init)`: `σ² (JᵀWJ)⁻¹`, Jacobian `J = [∂r/∂kᵢ ∂r/∂z_init]`
  by central finite differences; Student-t intervals `ci_t`, `z_init_ci_t`.
  Degrees of freedom `dof = n − p − p_s`: `p = 2` (kᵢ, z_init; 1 if z_init is
  fixed) and `p_s = 1` when the calibration tail lies inside the fit subset
  (`n_overlap > 0`, e.g. FIT_KI_MODE = :full), since s is then fitted on the
  same points; `p_s = 0` for a disjoint tail. `n_params = p + p_s`;
- profile intervals `ci_profile` (kᵢ, z_init re-minimised) and
  `z_init_ci_profile` (z_init, kᵢ re-minimised) where `RSS` rises by
  `χ²₁(conf)` (× `σ²` if unweighted);
- scale, in `u = log10 s`: `Var(u) = Var(u | kᵢ, z_init)_tail + gᵀ Cov g`, with
  `g = (∂u/∂kᵢ, ∂u/∂z_init)`.

# Returns
NamedTuple with
- kᵢ block     : `ki`, `ki_err` (= t × `se`), `se`, `ci_t`, `ci_profile`,
                 `delta_target`, `delta_rss`, `profile_note`, `rss`, `sigma2`,
                 `dof`, `n_params`, `n_used`, `r2_coeff` (weighted R² in log10
                 space), `converged`, `result`;
- z_init block : `z_init`, `z_init_err` (= t × `z_init_se`), `z_init_se`,
                 `z_init_ci_t`, `z_init_ci_profile`, `corr_ki_zinit`, `cov`;
- scale block  : `scale`, `scale_inv`, `scale_err`, `scale_se`, `scale_ci_t`,
                 `scale_ci_from_profile`, `se_log10_scale`, `dlog10s_dki`,
                 `dlog10s_dzinit`, `corr_ki_scale`, `corr_zinit_scale`,
                 `tail_rmse` (mm), `tail_range`, `n_tail_used`, `n_overlap`.
"""
function fit_cqd_ki_scale_with_error(itp, data_org, selected_points;
    n_tail::Int,
    bounds::Tuple{<:Real,<:Real},
    zinit_bounds = nothing,
    scale_mode::Symbol = :log,
    conf::Real = 0.95,
    use_Zse::Bool = false,
    profile::Bool = true,
    profile_grid::Int = 400)

    ln10 = log(10.0)
    N = size(data_org, 1)
    @assert 1 ≤ n_tail ≤ N "n_tail must be between 1 and size(data_org,1) = $N"
    fitz     = zinit_bounds !== nothing
    zlo, zhi = fitz ? (float(zinit_bounds[1]), float(zinit_bounds[2])) : (0.0, 0.0)

    # ---- tail (calibration) --------------------------------------------------
    I_t  = collect(Float64, data_org[end-n_tail+1:end, 1])
    Z_t  = collect(Float64, data_org[end-n_tail+1:end, 3])
    σZ_t = collect(Float64, data_org[end-n_tail+1:end, 4])
    mt = isfinite.(I_t) .& isfinite.(Z_t) .& (Z_t .> 0)
    use_Zse && (mt .&= isfinite.(σZ_t) .& (σZ_t .> 0))
    I_t, Z_t, σZ_t = I_t[mt], Z_t[mt], σZ_t[mt]
    @assert length(I_t) ≥ 2 "Need at least 2 valid tail points"

    # ---- fit subset (kᵢ, z_init loss) ---------------------------------------
    I  = collect(Float64, selected_points[:, 1])
    Z  = collect(Float64, selected_points[:, 3])
    σZ = collect(Float64, selected_points[:, 4])
    m0 = isfinite.(I) .& isfinite.(Z) .& (Z .> 0)
    use_Zse && (m0 .&= isfinite.(σZ) .& (σZ .> 0))
    I, Z, σZ = I[m0], Z[m0], σZ[m0]
    w = use_Zse ? ((Z .* ln10) ./ σZ) .^ 2 : ones(length(I))

    ki_min, ki_max = float(bounds[1]), float(bounds[2])

    # ---- scale s(kᵢ, z_init) and Var(log10 s | kᵢ, z_init) on the tail:
    #      shifted model z_init + m ≈ s·Z (data unchanged)
    function scale_for(ki, zi)
        m  = zi .+ itp.(I_t, Ref(ki))
        ok = isfinite.(m) .& (m .> 0)
        count(ok) ≥ 2 || return (NaN, NaN)
        wt = use_Zse ? ((Z_t[ok] .* ln10) ./ σZ_t[ok]) .^ 2 : nothing     # 1/σ²(log10 Z)
        _scale_from_tail(m[ok], Z_t[ok]; scale_mode = scale_mode, w = wt)
    end

    # ---- composite residuals on the fit subset -------------------------------
    function residuals_for(ki, zi)
        s, _ = scale_for(ki, zi)
        (isfinite(s) && s > 0) || return nothing
        mI = itp.(I, Ref(ki))
        m  = isfinite.(mI)
        any(m) || return nothing
        zpred = (zi .+ mI[m]) ./ s
        all(>(0), zpred) || return nothing
        r = log10.(zpred) .- log10.(Z[m])
        return (r = r, w = w[m], m = m, s = s)
    end
    function loss(ki, zi)
        o = residuals_for(ki, zi)
        o === nothing && return Inf
        sum(o.w .* (o.r .^ 2))
    end

    # profile losses: one parameter fixed, the other re-minimised
    best_zi(ki) = fitz ? _minimize_1d(zi -> loss(ki, zi), zlo, zhi)[1:2] : (0.0, loss(ki, 0.0))
    best_ki(zi) = _minimize_1d(ki -> loss(ki, zi), ki_min, ki_max)[1:2]
    loss_ki(ki) = best_zi(ki)[2]
    loss_zi(zi) = best_ki(zi)[2]

    # ---- optimise (kᵢ, z_init) ------------------------------------------------
    res = optimize(loss_ki, ki_min, ki_max, Brent())
    k̂  = Optim.minimizer(res)
    ẑ  = best_zi(k̂)[1]

    out0 = residuals_for(k̂, ẑ)
    out0 === nothing && error("Residuals invalid at the optimum (kᵢ, z_init)")
    r0, w0, mfit0 = out0.r, out0.w, out0.m

    # Parameters estimated from the points entering the RSS: kᵢ (+ z_init), and
    # also s when the calibration tail lies inside the fit subset (e.g.
    # FIT_KI_MODE = :full) — then the residuals have lost one more degree of
    # freedom. With a disjoint tail (:low) s comes from independent points.
    n_overlap = count(I .>= first(I_t))        # fit points that also lie in the tail
    p   = fitz ? 2 : 1                         # parameters in the Jacobian (kᵢ, z_init)
    p_s = n_overlap > 0 ? 1 : 0                # s, if fitted on these same points
    n = length(r0)
    @assert n > p + p_s "Not enough valid points to estimate uncertainty"
    RSS0 = sum(w0 .* (r0 .^ 2))
    dof  = n - p - p_s
    σ²   = RSS0 / dof

    # ---- finite-difference Jacobian [∂r/∂kᵢ  ∂r/∂z_init] (composite) ----------
    h₀ = _fd_step(k̂, ki_min, ki_max)
    hz = _zinit_step(ẑ, Z)
    shifts = fitz ? [(h₀, 0.0), (-h₀, 0.0), (0.0, hz), (0.0, -hz)] : [(h₀, 0.0), (-h₀, 0.0)]
    outs   = [residuals_for(k̂ + dk, ẑ + dz) for (dk, dz) in shifts]
    any(isnothing, outs) &&
        error("Derivative evaluation failed near the optimum; widen bounds or check model positivity")
    mJ = mfit0
    for o in outs
        mJ = mJ .& o.m
    end
    @assert count(mJ) > p "Not enough common points to compute the derivative"

    function r_on_mask(ki, zi, mask)
        s, _ = scale_for(ki, zi)
        zpred = (zi .+ itp.(I[mask], Ref(ki))) ./ s
        log10.(zpred) .- log10.(Z[mask])
    end
    rJ   = r_on_mask(k̂, ẑ, mJ)
    wJ   = w[mJ]
    drdk = (r_on_mask(k̂ + h₀, ẑ, mJ) .- r_on_mask(k̂ - h₀, ẑ, mJ)) ./ (2h₀)
    J    = if fitz
        drdz = (r_on_mask(k̂, ẑ + hz, mJ) .- r_on_mask(k̂, ẑ - hz, mJ)) ./ (2hz)
        hcat(drdk, drdz)
    else
        reshape(drdk, :, 1)
    end
    F   = J' * (wJ .* J)
    Cov = σ² .* inv(F)
    se      = sqrt(Cov[1, 1])
    se_z    = fitz ? sqrt(Cov[2, 2]) : 0.0
    corr_kz = fitz ? Cov[1, 2] / (se * se_z) : NaN

    tcrit  = quantile(TDist(dof), 0.5 + conf/2)
    k_err  = tcrit * se
    ci_t   = (k̂ - k_err, k̂ + k_err)
    z_err  = tcrit * se_z
    z_ci_t = (ẑ - z_err, ẑ + z_err)

    # weighted R² in log10 space on mJ
    y   = log10.(Z[mJ])
    ŷ   = y .+ rJ
    ȳw  = sum(wJ .* y) / sum(wJ)
    TSS = sum(wJ .* (y .- ȳw) .^ 2)
    R2  = TSS > 0 ? 1 - sum(wJ .* (y .- ŷ) .^ 2) / TSS : NaN

    # ---- profile intervals for kᵢ and z_init ----------------------------------
    ci_profile   = nothing
    z_ci_profile = nothing
    Δtarget      = profile ? quantile(Chisq(1), conf) : nothing
    Δrss         = nothing
    profile_note = nothing
    if profile
        if use_Zse
            Δrss = Δtarget
        else
            profile_note = :profile_interval_scaled_for_unweighted
            Δrss = σ² * Δtarget
        end
        ci_profile = _profile_interval(loss_ki, k̂, ki_min, ki_max, RSS0 + Δrss; grid = profile_grid)
        fitz && (z_ci_profile = _profile_interval(loss_zi, ẑ, zlo, zhi, RSS0 + Δrss; grid = profile_grid))
    end

    # ---- scale and its uncertainty ------------------------------------------
    ŝ, var_u_tail = scale_for(k̂, ẑ)
    û    = log10(ŝ)
    u_at(ki, zi) = log10(scale_for(ki, zi)[1])
    dudk = (u_at(k̂ + h₀, ẑ) - u_at(k̂ - h₀, ẑ)) / (2h₀)
    dudz = fitz ? (u_at(k̂, ẑ + hz) - u_at(k̂, ẑ - hz)) / (2hz) : 0.0
    g    = fitz ? [dudk, dudz] : [dudk]
    Cg   = Cov * g

    var_u      = var_u_tail + dot(g, Cg)           # tail scatter ⊕ propagated (kᵢ, z_init)
    se_u       = sqrt(var_u)
    scale_se   = ŝ * ln10 * se_u                   # delta method, in units of s
    scale_err  = tcrit * scale_se
    scale_ci_t = (ŝ * 10.0^(-tcrit * se_u), ŝ * 10.0^(tcrit * se_u))
    corr_ks    = se_u > 0 ? Cg[1] / (se * se_u) : NaN
    corr_zs    = (fitz && se_u > 0) ? Cg[2] / (se_z * se_u) : NaN

    scale_ci_from_profile = ci_profile === nothing ? nothing :
        extrema((scale_for(ci_profile[1], best_zi(ci_profile[1])[1])[1],
                 scale_for(ci_profile[2], best_zi(ci_profile[2])[1])[1]))

    m_tail    = (ẑ .+ itp.(I_t, Ref(k̂))) ./ ŝ
    okt       = isfinite.(m_tail)
    tail_rmse = sqrt(mean(abs2, m_tail[okt] .- Z_t[okt]))

    return (
        # --- kᵢ ---
        ki           = k̂,
        ki_err       = k_err,
        se           = se,
        ci_t         = ci_t,
        ci_profile   = ci_profile,
        delta_target = Δtarget,
        delta_rss    = Δrss,
        profile_note = profile_note,
        rss          = RSS0,
        sigma2       = σ²,
        dof          = dof,
        n_params     = p + p_s,          # parameters subtracted from the dof (kᵢ, z_init, s)
        n_used       = length(rJ),
        r2_coeff     = R2,
        converged    = Optim.converged(res),
        result       = res,
        # --- z_init ---
        z_init            = ẑ,
        z_init_err        = z_err,
        z_init_se         = se_z,
        z_init_ci_t       = z_ci_t,
        z_init_ci_profile = z_ci_profile,
        corr_ki_zinit     = corr_kz,
        cov               = Cov,
        # --- scale ---
        scale                 = ŝ,
        scale_inv             = 1 / ŝ,
        scale_err             = scale_err,
        scale_se              = scale_se,
        scale_ci_t            = scale_ci_t,
        scale_ci_from_profile = scale_ci_from_profile,
        se_log10_scale        = se_u,
        dlog10s_dki           = dudk,
        dlog10s_dzinit        = dudz,
        corr_ki_scale         = corr_ks,
        corr_zinit_scale      = corr_zs,
        tail_rmse             = tail_rmse,
        tail_range            = (first(I_t), last(I_t)),
        n_tail_used           = length(I_t),
        n_overlap             = n_overlap,
    )
end

"""
    fit_qm_scale_with_error(zqm, data_org, selected_points;
                            n_tail, zinit_bounds=nothing, scale_mode=:log, conf=0.95,
                            use_Zse=false, profile=true, profile_grid=400)

Uncertainty-aware companion of `fit_qm_scale`: offset `z_init` and calibration
factor `s` such that the **calibrated QM model** `(z_init + zqm(I))/s` matches the
raw experiment, with formal uncertainties on both.

- `z_init` minimises the log10-space RSS on `selected_points` (s(z_init) from
  the tail, shifted model z_init + zqm ≈ s·z); linearised SE from `∂r/∂z_init` (central differences),
  Student-t interval `z_init_ci_t` and profile interval `z_init_ci_profile`,
  with `z_init_dof = n − 1 − p_s` (`p_s = 1` when the tail lies inside the fit
  subset, as in the CQD fitter).
- `s` is estimated on the tail at `ẑ` exactly as in the no-offset script, and
  its variance gains the part propagated from z_init:
  `Var(u) = Var(u | z_init)_tail + (∂u/∂z_init)² Var(ẑ)`, `u = log10 s`.
  `scale_ci_profile` is the profile in `u` at fixed `ẑ` (consistency check).

With `zinit_bounds = nothing` → `z_init = 0` and the result equals the
no-offset `fit_qm_scale_with_error`.

# Returns
NamedTuple: `z_init`, `z_init_err`, `z_init_se`, `z_init_ci_t`,
`z_init_ci_profile`, `z_init_rss`, `z_init_dof`, `scale`, `scale_inv`,
`scale_err`, `scale_se`, `scale_ci_t`, `scale_ci_profile`, `log10_scale`,
`se_log10_scale`, `dlog10s_dzinit`, `corr_zinit_scale`, `delta_target`,
`delta_rss`, `profile_note`, `rss`, `sigma2`, `dof` (tail fit in `u`),
`n_params`, `n_tail_used`, `tail_rmse` (mm), `tail_range`, `r2_coeff`
(linear-space R² on the full `data_org`).
"""
function fit_qm_scale_with_error(zqm, data_org, selected_points;
    n_tail::Int,
    zinit_bounds = nothing,
    scale_mode::Symbol = :log,
    conf::Real = 0.95,
    use_Zse::Bool = false,
    profile::Bool = true,
    profile_grid::Int = 400)

    ln10 = log(10.0)
    N = size(data_org, 1)
    @assert 1 ≤ n_tail ≤ N "n_tail must be between 1 and size(data_org,1) = $N"
    fitz     = zinit_bounds !== nothing
    zlo, zhi = fitz ? (float(zinit_bounds[1]), float(zinit_bounds[2])) : (0.0, 0.0)

    # ---- tail (calibration) --------------------------------------------------
    I_t  = collect(Float64, data_org[end-n_tail+1:end, 1])
    Z_t  = collect(Float64, data_org[end-n_tail+1:end, 3])
    σZ_t = collect(Float64, data_org[end-n_tail+1:end, 4])
    m_t  = zqm.(I_t)
    mt = isfinite.(I_t) .& isfinite.(Z_t) .& (Z_t .> 0) .& isfinite.(m_t) .& (m_t .> 0)
    use_Zse && (mt .&= isfinite.(σZ_t) .& (σZ_t .> 0))
    I_t, Z_t, σZ_t, m_t = I_t[mt], Z_t[mt], σZ_t[mt], m_t[mt]
    n = length(I_t)
    @assert n ≥ 2 "Need at least 2 valid tail points"

    # ---- fit subset (z_init loss) --------------------------------------------
    I   = collect(Float64, selected_points[:, 1])
    Z   = collect(Float64, selected_points[:, 3])
    σZ  = collect(Float64, selected_points[:, 4])
    m_f = zqm.(I)
    mf  = isfinite.(I) .& isfinite.(Z) .& (Z .> 0) .& isfinite.(m_f) .& (m_f .> 0)
    use_Zse && (mf .&= isfinite.(σZ) .& (σZ .> 0))
    I, Z, σZ, m_f = I[mf], Z[mf], σZ[mf], m_f[mf]
    w = use_Zse ? ((Z .* ln10) ./ σZ) .^ 2 : ones(length(I))

    # tail: shifted model z_init + m ≈ s·Z (data unchanged); log10-space weights 1/σ²(log10 Z)
    w_tail = use_Zse ? ((Z_t .* ln10) ./ σZ_t) .^ 2 : nothing
    function scale_for(zi)
        mt = zi .+ m_t
        all(>(0), mt) || return (NaN, NaN)
        _scale_from_tail(mt, Z_t; scale_mode = scale_mode, w = w_tail)
    end
    function residuals_for(zi)
        s, _ = scale_for(zi)
        (isfinite(s) && s > 0) || return nothing
        zpred = (zi .+ m_f) ./ s
        all(>(0), zpred) || return nothing
        log10.(zpred) .- log10.(Z)
    end
    function loss_zi(zi)
        r = residuals_for(zi)
        r === nothing ? Inf : sum(w .* r .^ 2)
    end

    # ---- z_init and its uncertainty -----------------------------------------
    ẑ      = fitz ? _minimize_1d(loss_zi, zlo, zhi)[1] : 0.0
    z_rss  = loss_zi(ẑ)
    # dof of the fit-subset residuals: minus z_init, and minus s when the tail
    # lies inside the fit subset (same rule as the CQD fitter)
    n_overlap = count(I .>= first(I_t))
    z_dof  = length(Z) - (fitz ? 1 : 0) - (n_overlap > 0 ? 1 : 0)
    hz     = _zinit_step(ẑ, Z)
    z_se, z_err, z_ci_t, z_ci_profile = 0.0, 0.0, (ẑ, ẑ), nothing
    if fitz
        @assert z_dof ≥ 1 "Not enough valid points to estimate the z_init uncertainty"
        rp = residuals_for(ẑ + hz)
        rm = residuals_for(ẑ - hz)
        (rp === nothing || rm === nothing) &&
            error("Derivative evaluation failed near the optimum z_init; check model positivity")
        drdz   = (rp .- rm) ./ (2hz)
        σ²_z   = z_rss / z_dof
        z_se   = sqrt(σ²_z / sum(w .* drdz .^ 2))
        z_err  = quantile(TDist(z_dof), 0.5 + conf/2) * z_se
        z_ci_t = (ẑ - z_err, ẑ + z_err)
        if profile
            Δz = quantile(Chisq(1), conf) * (use_Zse ? 1.0 : σ²_z)
            z_ci_profile = _profile_interval(loss_zi, ẑ, zlo, zhi, z_rss + Δz; grid = profile_grid)
        end
    end

    # ---- scale at ẑ: tail estimator ⊕ part propagated from z_init -----------
    w_t = use_Zse ? w_tail : ones(n)
    d   = log10.(ẑ .+ m_t) .- log10.(Z_t)    # log10 ratios (z_init + model)/data on the tail
    ŝ, var_u_tail = scale_for(ẑ)
    û    = log10(ŝ)
    dudz = fitz ? (log10(scale_for(ẑ + hz)[1]) - log10(scale_for(ẑ - hz)[1])) / (2hz) : 0.0
    var_u   = var_u_tail + dudz^2 * z_se^2
    se_u    = sqrt(var_u)
    corr_zs = (fitz && se_u > 0) ? dudz * z_se / se_u : NaN

    # ---- log10-space RSS in u (σ², dof, profile) — conditional on ẑ ----------
    loss(u) = sum(w_t .* (d .- u) .^ 2)
    RSS0 = loss(û)
    dof  = n - 1
    σ²   = RSS0 / dof

    tcrit      = quantile(TDist(dof), 0.5 + conf/2)
    scale_se   = ŝ * ln10 * se_u
    scale_err  = tcrit * scale_se
    scale_ci_t = (ŝ * 10.0^(-tcrit * se_u), ŝ * 10.0^(tcrit * se_u))

    scale_ci_profile = nothing
    Δtarget      = profile ? quantile(Chisq(1), conf) : nothing
    Δrss         = nothing
    profile_note = nothing
    if profile
        if use_Zse
            Δrss = Δtarget
        else
            profile_note = :profile_interval_scaled_for_unweighted
            Δrss = σ² * Δtarget
        end
        halfw = max(10 * tcrit * se_u, 1e-6)      # generous search window
        u_lo, u_hi = _profile_interval(loss, û, û - halfw, û + halfw, RSS0 + Δrss; grid = profile_grid)
        scale_ci_profile = (10.0^u_lo, 10.0^u_hi)
    end

    # ---- diagnostics: calibrated model vs raw data ---------------------------
    tail_rmse = sqrt(mean(abs2, (ẑ .+ m_t) ./ ŝ .- Z_t))
    z_all = data_org[:, 3]
    p_all = (ẑ .+ zqm.(data_org[:, 1])) ./ ŝ
    r2    = 1 - sum(abs2, p_all .- z_all) / sum(abs2, z_all .- mean(z_all))

    return (
        z_init            = ẑ,
        z_init_err        = z_err,
        z_init_se         = z_se,
        z_init_ci_t       = z_ci_t,
        z_init_ci_profile = z_ci_profile,
        z_init_rss        = z_rss,
        z_init_dof        = z_dof,
        scale             = ŝ,
        scale_inv         = 1 / ŝ,
        scale_err         = scale_err,
        scale_se          = scale_se,
        scale_ci_t        = scale_ci_t,
        scale_ci_profile  = scale_ci_profile,
        log10_scale       = û,
        se_log10_scale    = se_u,
        dlog10s_dzinit    = dudz,
        corr_zinit_scale  = corr_zs,
        delta_target      = Δtarget,
        delta_rss         = Δrss,
        profile_note      = profile_note,
        rss               = RSS0,
        sigma2            = σ²,
        dof               = dof,
        n_params          = fitz ? 2 : 1,     # (s, z_init)
        n_overlap         = n_overlap,
        n_tail_used       = n,
        tail_rmse         = tail_rmse,
        tail_range        = (first(I_t), last(I_t)),
        r2_coeff          = r2,
    )
end
"""
    compute_metrics(A, X)
 
Battery of agreement metrics between a reference vector `A` (experiment) and a
comparison vector `X` (a model) for data spanning several orders of magnitude.
All inputs must be strictly positive.
 
Returns a NamedTuple with `log_MSE`, `log_RMSE`, `max_log_error` (log10-space
errors), `rel_mean`/`rel_median`/`rel_max` (fractional errors), `MAPE`/`sMAPE`
(percent), `L2_norm`/`L2_log_norm` (relative L2 norms), `chi2_log` (χ² of the
log residuals scaled by their own std), `KS_distance` (Kolmogorov–Smirnov
distance between normalised cumulative sums) and the raw `log_err` vector.
 
Used by `compare_datasets`.
"""
function compute_metrics(A,X)
    LA = log10.(A)
    LX = log10.(X)
    log_err = LX .- LA
 
    log_MSE  = mean(abs2, log_err)
    log_RMSE = sqrt(log_MSE)
    max_log_error = maximum(abs.(log_err))
 
    rel_err = abs.((X .- A) ./ A)
    rel_mean   = mean(rel_err)
    rel_median = median(rel_err)
    rel_max    = maximum(rel_err)
 
    MAPE  = 100 * mean(rel_err)
    sMAPE = 100 * mean(abs.(A .- X) ./ ((abs.(A) .+ abs.(X)) ./ 2))
 
    L2_norm     = norm(X .- A) / norm(A)
    L2_log_norm = norm(LX .- LA) / norm(LA)
 
    σlog = std(log_err)
    chi2_log = sum((log_err ./ σlog).^2)
 
    A_norm = cumsum(A ./ sum(A))
    X_norm = cumsum(X ./ sum(X))
    KS_distance = maximum(abs.(X_norm .- A_norm))
 
    return (
        log_MSE = log_MSE,
        log_RMSE = log_RMSE,
        max_log_error = max_log_error,
        rel_mean = rel_mean,
        rel_median = rel_median,
        rel_max = rel_max,
        MAPE = MAPE,
        sMAPE = sMAPE,
        L2_norm = L2_norm,
        L2_log_norm = L2_log_norm,
        chi2_log = chi2_log,
        KS_distance = KS_distance,
        log_err = log_err
    )
end
 
"""
    compare_datasets(x_ref, A, B, C; plot_errors=true)
 
Compare two models against experiment on a common current grid `x_ref`:
`A` = experimental z, `B` = CQD prediction, `C` = QM prediction (all > 0).
 
Prints a PrettyTables summary of `compute_metrics(A, B)` and
`compute_metrics(A, C)` side by side (the better value of each row is
highlighted in red) and, if `plot_errors`, displays a scatter of the log10
errors of both models versus current.
 
Returns `(CQD = metrics_B, QM = metrics_C)`.
"""
function compare_datasets(x_ref::AbstractVector, # current
                            A::AbstractVector,   # Experimental
                            B::AbstractVector,   # CQD
                            C::AbstractVector;   # QM
                            plot_errors=true)
    @assert length(x_ref) == length(A) == length(B) == length(C) "All vectors must have same length."
    @assert all(A .> 0) && all(B .> 0) && all(C .> 0) "All values must be > 0 for log comparison."
 
    # --- compute for both models ---
    R_B = compute_metrics(A, B);
    R_C = compute_metrics(A, C);
 
    # ------- Pretty Table ---------
    header = ["Metric", "CQD vs Exp", "QM vs Exp"];
    tbl = [
        "log_MSE"        R_B.log_MSE        R_C.log_MSE
        "log_RMSE"       R_B.log_RMSE       R_C.log_RMSE
        "max_log_error"  R_B.max_log_error  R_C.max_log_error
        "rel_mean"       R_B.rel_mean       R_C.rel_mean
        "rel_median"     R_B.rel_median     R_C.rel_median
        "rel_max"        R_B.rel_max        R_C.rel_max
        "MAPE (%)"       R_B.MAPE           R_C.MAPE
        "sMAPE (%)"      R_B.sMAPE          R_C.sMAPE
        "L2_norm"        R_B.L2_norm        R_C.L2_norm
        "L2_log_norm"    R_B.L2_log_norm    R_C.L2_log_norm
        "chi2_log"       R_B.chi2_log       R_C.chi2_log
        "KS_distance"    R_B.KS_distance    R_C.KS_distance
    ];
    hl_min_red = TextHighlighter(
        (tbl, i, j) -> begin
            # Only highlight numeric entries in columns 2 or 3
            if j == 2 || j == 3
                v1 = tbl[i, 2]
                v2 = tbl[i, 3]
 
                # Be safe: only operate if both are numbers
                if isa(v1, Number) && isa(v2, Number)
                    row_min = v1 <= v2 ? v1 : v2
                    return tbl[i, j] == row_min
                else
                    return false
                end
            else
                return false
            end
        end,
        crayon"fg:red bold"  # red, bold; you can add bg if you like
    );
 
    pretty_table(tbl,
        alignment     = [:l,:c,:c],
        column_labels = header,
        formatters    = [fmt__printf("%8.6f", 2:3)],
        highlighters  = [hl_min_red],
        style         = TextTableStyle(
                        first_line_column_label = crayon"yellow bold",
                        table_border  = crayon"blue bold",
                        column_label  = crayon"yellow bold",
                        ),
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
        equal_data_column_widths= true,
    )
 
    # ---- Plot of log-errors ----
    if plot_errors
        fig = Figure(size = (800, 500))
        ax  = Axis(fig[1, 1]; title = "Log Errors for CQD and QM",
                   xlabel = "Current", ylabel = "Log Error")
        scatter!(ax, x_ref, R_B.log_err; markersize = 7, color = :red,
                 label = "log10(cqd) - log10(exp)")
        scatter!(ax, x_ref, R_C.log_err; markersize = 7, color = :blue,
                 label = "log10(qm) - log10(exp)")
        axislegend(ax; position = :rt)
        SHOW_FIGS && display(fig)
    end
 
    return (CQD = R_B, QM = R_C)
end
 
"""
    LOGX, LOGY

Keyword bundles for a log10 axis in CairoMakie (`Axis(fig[1,1]; LOGX..., LOGY...)`):
the scale plus minor ticks at 2…9 × 10ⁿ.
"""
const LOGX = (xscale = log10, xminorticks = IntervalsBetween(9))
const LOGY = (yscale = log10, yminorticks = IntervalsBetween(9))

"""
    decade_ticks(v)

`(values, labels)` tick specification with `10^{n}` LaTeX labels for decade
values `v` (e.g. `decade_ticks([1e-2, 1e-1, 1.0])`).
"""
decade_ticks(v) = (v, [L"10^{%$(round(Int, log10(x)))}" for x in v])

"""
    logsafe(v)

Copy of `v` with the non-positive / non-finite entries replaced by `NaN`. On a
log axis Makie throws a `DomainError` for values ≤ 0 (Plots dropped them
silently); `NaN` is simply not drawn.
"""
logsafe(v) = map(t -> (isfinite(t) && t > 0) ? float(t) : NaN, v)

"""
    yerr_log!(ax, x, y, σ; kw...)
    xerr_log!(ax, x, y, σ; kw...)

Symmetric error bars that remain valid on a log axis: the lower half-length is
clipped to `0.999·y` (resp. `0.999·x`) so the lower end stays > 0; the upper end
is unchanged. Also fine on a linear axis (where it only matters if σ > |y|).
"""
function yerr_log!(ax, x, y, σ; color = :black, whiskerwidth = 5, linewidth = 1, kw...)
    lo = min.(σ, 0.999 .* abs.(y))
    errorbars!(ax, x, y, lo, σ; color, whiskerwidth, linewidth, kw...)
end
function xerr_log!(ax, x, y, σ; color = :black, whiskerwidth = 5, linewidth = 1, kw...)
    lo = min.(σ, 0.999 .* abs.(x))
    errorbars!(ax, x, y, lo, σ; direction = :x, color, whiskerwidth, linewidth, kw...)
end

"""
    save_fig(fig, path; show = SHOW_FIGS)

Display the CairoMakie figure (if `show`) and save it to `path`; the format
follows the extension (png at `PX_PER_UNIT` pixels per unit, or pdf / svg).
"""
function save_fig(fig, path; show = SHOW_FIGS)
    show && display(fig)
    save(path, fig)
    return path
end

"""
    plot_cqd_vs_qm(ZCQD, ZQM, Icurrent, ki_list, I_compared;
                   palette_name=:rainbow, size=(1100, 1250))

Diagnostic figure showing, for a handful of coil currents, how the CQD
prediction varies with kᵢ against the (kᵢ-independent) QM value.

`I_compared` is a list of three lists of currents (A), one per panel, e.g.
`[[0.0], [0.001, 0.005, 0.010, 0.015], [0.850, 0.900, 0.950, 1.000]]`. Each
current is looked up by value in `Icurrent` (error if absent). For each one,
`ZCQD[row, :]` is drawn versus kᵢ on a **log10 kᵢ axis** (one point per grid
kᵢ) and the QM value `ZQM[row]` as a dashed horizontal line. Three stacked
panels share the kᵢ axis:
1. `I_compared[1]` on a **linear** y axis in **μm**, with the signed values
   (no `abs`, so a negative z_max shows up); lines **and markers**, a
   different marker symbol for each series (CQD and QM of each current);
   symmetric y range ±`ylim_top` (μm), automatic when `nothing` (largest
   |value| rounded up to 1, 2 or 5 × 10ⁿ); legend inside, right, at z = 0;
2. `I_compared[2]`, log10 y axis in mm, range taken from the plotted data;
3. `I_compared[3]`, log10 y axis in mm.

Colours: one colour per current over the whole figure (no repeats), sampled
from `palette_name` with as many colours as there are currents in total. The
**reddest** (last) colour goes to the top panel; panels 2 and 3 take the
remaining colours in order. Each panel has its own legend, inside the plot
area. On the log panels the non-positive CQD values are not drawn (`logsafe`).

Returns the CairoMakie `Figure`.
"""
function plot_cqd_vs_qm(ZCQD, ZQM, Icurrent, ki_list, I_compared;
        palette_name = :rainbow,
        ylim_top = nothing,          # (μm) top panel y range ±ylim_top; nothing = automatic
        size = (1100, 1250),
    )
    @assert length(I_compared) == 3 "I_compared must hold three lists (one per panel)"
    # rows of Icurrent for every requested current (by value)
    function row_of(Ix)
        r = findfirst(I -> isapprox(I, Ix; atol = 1e-9), Icurrent)
        r === nothing && error("plot_cqd_vs_qm: current $(Ix) A is not in the simulated currents")
        return r
    end
    rows = [row_of.(Ip) for Ip in I_compared]

    # one colour per current; the reddest (last in the palette) for the top panel
    n_tot = sum(length, rows)
    cmap  = Makie.resample_cmap(palette_name, n_tot)
    n1    = length(rows[1])
    cols  = [cmap[end-n1+1:end],                                   # panel 1: red end
             cmap[1:length(rows[2])],                              # panel 2
             cmap[length(rows[2])+1 : length(rows[2])+length(rows[3])]]   # panel 3

    kx  = collect(float.(ki_list))                                 # kᵢ (×10⁻⁶), log x axis
    mA(r) = 1000 * Icurrent[r]
    # marker symbols of the upper panel, one per series (CQD, QM, CQD, QM, …)
    mk_cycle = [:circle, :rect, :diamond, :utriangle, :dtriangle, :pentagon, :hexagon, :star5, :cross, :xcross]
    mk(n)    = mk_cycle[mod1(n, length(mk_cycle))]

    fig = Figure(; size)
    # larger axis labels and tick labels (every panel goes through xkw)
    xkw = (LOGX..., xlabelsize = 22, ylabelsize = 22, xticklabelsize = 17, yticklabelsize = 17,
           xticks = decade_ticks(10.0 .^ (floor(Int, log10(minimum(kx))):ceil(Int, log10(maximum(kx))))))

    # ---- panel 1: linear y in μm, signed z_max, lines + markers ----
    ax_l = Axis(fig[1, 1]; xkw..., ylabel = L"$z_{max}$ ($\mathrm{\mu m}$)")
    hlines!(ax_l, 0.0; color = :black, linestyle = :dot, linewidth = 1)
    for (n, (r, c)) in enumerate(zip(rows[1], cols[1]))
        scatterlines!(ax_l, kx, 1e3 .* ZCQD[r, :]; color = c, linewidth = 2,           # mm → μm
            marker = mk(2n - 1), markersize = 10, markercolor = c, strokecolor = c, strokewidth = 0.8,
            label = "CQD $(mA(r)) mA")
        scatterlines!(ax_l, kx, fill(1e3 * ZQM[r], length(kx)); color = c, linewidth = 1.5, linestyle = :dash,
            marker = mk(2n), markersize = 9, markercolor = :white, strokecolor = c, strokewidth = 1.2,
            label = "QM $(mA(r)) mA")
    end
    # symmetric y range about 0: ±ylim_top (μm) if given, otherwise the largest
    # |value| drawn rounded up to 1, 2 or 5 × 10ⁿ
    ymax_l = maximum(abs, vcat([1e3 .* vec(ZCQD[r, :]) for r in rows[1]]..., [1e3 * ZQM[r] for r in rows[1]]))
    if ylim_top === nothing
        dec    = 10.0^floor(log10(ymax_l))
        ylim_l = dec * first(filter(m -> m * dec >= ymax_l, [1, 2, 5, 10]))
    else
        ylim_l = ylim_top
    end
    ylims!(ax_l, -ylim_l, ylim_l)
    axislegend(ax_l; position = :rc)          # inside, right edge, at mid-height (z = 0)

    # ---- panel 2: log y; range from the data actually drawn (CQD > 0 and |QM|):
    #      ×0.5 below, ×4 above so the upper-left legend does not cover the curves
    vals_a = filter(v -> isfinite(v) && v > 0,
                    vcat([vec(ZCQD[r, :]) for r in rows[2]]..., [abs(ZQM[r]) for r in rows[2]]))
    ya_lo, ya_hi = 0.5 * minimum(vals_a), 4.0 * maximum(vals_a)
    ax_a = Axis(fig[2, 1]; xkw..., LOGY..., ylabel = L"$z_{max}$ (mm)",
        yticks = decade_ticks(10.0 .^ (floor(Int, log10(ya_lo)):ceil(Int, log10(ya_hi)))))
    for (r, c) in zip(rows[2], cols[2])
        lines!(ax_a, kx, logsafe(ZCQD[r, :]); color = c, linewidth = 2, label = "CQD $(mA(r)) mA")
        hlines!(ax_a, abs(ZQM[r]); color = c, linewidth = 1.5, linestyle = :dash, label = "QM")
    end
    ylims!(ax_a, ya_lo, ya_hi)
    axislegend(ax_a; position = :lt, nbanks = 4)

    # ---- panel 3: log y ----
    ax_b = Axis(fig[3, 1]; xkw..., LOGY..., ylabel = L"$z_{max}$ (mm)",
        yticks = decade_ticks([1e-1, 1.0]),
        xlabel = L"$k_{i} \quad (\,\times 10^{-6})$")
    for (r, c) in zip(rows[3], cols[3])
        lines!(ax_b, kx, logsafe(ZCQD[r, :]); color = c, linewidth = 2, label = "CQD $(mA(r)) mA")
        hlines!(ax_b, abs(ZQM[r]); color = c, linewidth = 1.5, linestyle = :dash, label = "QM")
    end
    ylims!(ax_b, 5e-1, 3)
    # lower right is empty: at large kᵢ the curves sit at the top
    axislegend(ax_b; position = :rb, nbanks = 4)

    # shared x axis: identical kᵢ range, labels only at the bottom, panels close together
    linkxaxes!(ax_l, ax_a, ax_b)
    for a in (ax_l, ax_a)
        hidexdecorations!(a; grid = false, minorgrid = false, ticks = false, minorticks = false)
    end
    rowgap!(fig.layout, 6)
    return fig
end
 

# =============================================================================
# 0c) Goodness-of-fit tools (used on the scattered points, section 7)
# =============================================================================
 
"""
    FitStats
 
Container for the log-space goodness-of-fit metrics returned by
`goodness_of_fit`: `logMSE`, `logRMSE`, `R2_log`, `chi2_log`, `chi2_red`
(reduced χ²), `p_chi2` (χ² tail p-value), `AIC`, `BIC` and `NMAD`
(normalised median absolute deviation of the residuals).
"""
struct FitStats
    logMSE::Float64
    logRMSE::Float64
    R2_log::Float64
    chi2_log::Float64
    chi2_red::Float64
    p_chi2::Float64
    AIC::Float64
    BIC::Float64
    NMAD::Float64
end
 
"""
    goodness_of_fit(x, y, ypred; σ=nothing, k=0)
 
Evaluate how well model predictions `ypred` match observations `y` (over
support `x`), working in **natural-log** space where the SG curves are roughly
power-law. `k` is the number of fitted model parameters (used for the degrees
of freedom and by AIC/BIC). Residual sign convention: model − data.
 
Always returns `logMSE`, `logRMSE`, `R2_log` and the robust scatter `NMAD`.
When per-point uncertainties `σ` are given they are propagated to log space
(`σ_log ≈ σ/y`) and the χ² statistic, reduced χ² (`χ²/(N−k)`), χ² p-value and
χ²-based AIC/BIC are computed; otherwise those are `NaN` and AIC/BIC fall back
to a `logMSE`-based surrogate.
 
Returns a `FitStats`.
"""
function goodness_of_fit(x, y, ypred; σ = nothing, k::Int = 0)
    @assert length(x) == length(y) == length(ypred)
    N = length(y)
 
    logy    = log.(y)
    logpred = log.(ypred)
    r       = logpred .- logy
 
    logMSE  = mean(r .^ 2)
    logRMSE = sqrt(logMSE)
    R2_log  = 1 - sum(r .^ 2) / sum((logy .- mean(logy)) .^ 2)
    NMAD    = 1.4826 * median(abs.(r))
 
    if isnothing(σ)
        chi2_log = NaN; chi2_red = NaN; p_chi2 = NaN
        AIC = 2k + N * log(logMSE)          # logMSE as variance surrogate
        BIC = k * log(N) + N * log(logMSE)
    else
        @assert length(σ) == N
        σlog     = σ ./ y                   # δ(ln y) ≈ σ/y
        chi2_log = sum((r ./ σlog) .^ 2)
        dof      = max(N - k, 1)
        chi2_red = chi2_log / dof
        p_chi2   = ccdf(Chisq(dof), chi2_log)
        AIC = 2k + chi2_log                 # Gaussian likelihood, up to a constant
        BIC = k * log(N) + chi2_log
    end
 
    return FitStats(logMSE, logRMSE, R2_log, chi2_log, chi2_red, p_chi2, AIC, BIC, NMAD)
end
 
"""
    make_diagnostic_plots(x, y, y_CQD, y_QM, stats_CQD, stats_QM; σ=nothing,
                          size=(1250, 1000))

Diagnostic figure for the goodness of fit of the two *calibrated* models
(CQD: `(z_init + ki_itp)/s_CQD`, QM: `(z_init + zqm)/s_QM`) against the raw
scattered experiment.

Panels (2 × 2 grid of one CairoMakie `Figure`)
1. data (with y-error bars if `σ` is given) vs both models, log–log;
2. natural-log residuals `log(model) − log(exp)` vs current, annotated with
   logRMSE and R²_log;
3. pulls `(model − exp)/σ` vs current with ±1σ/±2σ bands, annotated with the
   reduced χ² (only if `σ` is given; otherwise an empty placeholder);
4. histogram of the log residuals, annotated with NMAD and logRMSE.

Returns the `Figure`.
"""
function make_diagnostic_plots(x, y, y_CQD, y_QM, stats_CQD::FitStats, stats_QM::FitStats;
                               σ = nothing, size = (1250, 1000))
    # residuals in natural-log space, sign convention model − data
    r_CQD = log.(y_CQD) .- log.(y)
    r_QM  = log.(y_QM)  .- log.(y)

    xt  = decade_ticks([1e-2, 1e-1, 1.0])
    fig = Figure(; size)
    # marker style of the residual scatters (fill with alpha, darker stroke)
    mk_cqd = (marker = :circle, markersize = 10, color = (:salmon3, 0.70), strokecolor = :red4,  strokewidth = 0.8)
    mk_qm  = (marker = :circle, markersize = 10, color = (:royalblue3, 0.70), strokecolor = :blue4, strokewidth = 0.8)

    # 1) Data vs calibrated models (log–log)
    ax1 = Axis(fig[1, 1]; LOGX..., LOGY..., xticks = xt,
        yticks = decade_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]),
        xlabel = "Coil Current (A)", ylabel = "Peak position (mm)",
        title = "Data vs calibrated models")
    σ === nothing || yerr_log!(ax1, x, y, σ)
    scatter!(ax1, x, logsafe(y); label = "Experiment",
        marker = :circle, markersize = 7, color = :white, strokecolor = :black, strokewidth = 0.8)
    lines!(ax1, x, y_CQD; label = "CQD", color = :red,  linewidth = 1.5)
    lines!(ax1, x, y_QM;  label = "QM",  color = :blue, linewidth = 2, linestyle = :dot)
    axislegend(ax1; position = :rb)

    # 2) Log residuals vs current
    ax2 = Axis(fig[1, 2]; LOGX..., xticks = xt,
        xlabel = "Coil Current (A)",
        ylabel = L"\ln(y_{\mathrm{model}}) - \ln(y_{\mathrm{exp}})",
        title = "Log-space residuals")
    hlines!(ax2, 0.0; color = :black, linestyle = :dash)
    scatter!(ax2, x, r_CQD; label = "CQD", mk_cqd...)
    scatter!(ax2, x, r_QM;  label = "QM",  mk_qm...)
    txt_CQD = L"CQD: logRMSE $= %$(fmt(stats_CQD.logRMSE, 3))$, $R^{2}_{\log} = %$(round(stats_CQD.R2_log, digits=4))$"
    txt_QM  = L"QM: logRMSE $= %$(fmt(stats_QM.logRMSE,  3))$, $R^{2}_{\log} = %$(round(stats_QM.R2_log,  digits=4))$"
    text!(ax2, 0.5, 0.15; text = txt_CQD, space = :relative, align = (:center, :center), fontsize = 12)
    text!(ax2, 0.5, 0.05; text = txt_QM,  space = :relative, align = (:center, :center), fontsize = 12)
    axislegend(ax2; position = :rt)

    # 3) Pulls vs current (needs σ)
    if σ === nothing
        ax3 = Axis(fig[2, 1]; title = "Pulls (no σ supplied)")
        hidedecorations!(ax3); hidespines!(ax3)
    else
        pu_CQD = pull(y_CQD, y, σ)
        pu_QM  = pull(y_QM,  y, σ)
        ax3 = Axis(fig[2, 1]; LOGX..., xticks = xt,
            xlabel = "Coil Current (A)",
            ylabel = L"(y_{\mathrm{model}} - y_{\mathrm{exp}})/\sigma_{\mathrm{exp}}",
            title = "Normalised residuals (pulls)")
        hspan!(ax3, -2, 2; color = (:gray, 0.08), label = L"\pm 2\sigma")
        hspan!(ax3, -1, 1; color = (:gray, 0.15), label = L"\pm 1\sigma")
        hlines!(ax3, 0.0; color = :black, linestyle = :dash)
        scatter!(ax3, x, pu_CQD; label = L"CQD: $\chi^{2}_{\mathrm{red}} = %$(fmt(stats_CQD.chi2_red, 3))$", mk_cqd...)
        scatter!(ax3, x, pu_QM;  label = L"QM: $\chi^{2}_{\mathrm{red}} = %$(fmt(stats_QM.chi2_red, 3))$",  mk_qm...)
        axislegend(ax3; position = :rt)
    end

    # 4) Histogram of log residuals
    ax4 = Axis(fig[2, 2];
        xlabel = "log-space residual", ylabel = "Normalised count",
        title = "Distribution of log-space residuals")
    hist!(ax4, r_CQD; normalization = :pdf, color = (:red,  0.4), label = "CQD")
    hist!(ax4, r_QM;  normalization = :pdf, color = (:blue, 0.4), label = "QM")
    vlines!(ax4, 0.0; color = :black, linestyle = :dash, linewidth = 1)
    txt2_CQD = @sprintf "CQD: NMAD = %.3g, logRMSE = %.3g" stats_CQD.NMAD stats_CQD.logRMSE
    txt2_QM  = @sprintf "QM:  NMAD = %.3g, logRMSE = %.3g" stats_QM.NMAD  stats_QM.logRMSE
    text!(ax4, 0.03, 0.95; text = txt2_CQD, space = :relative, align = (:left, :top), fontsize = 12)
    text!(ax4, 0.03, 0.88; text = txt2_QM,  space = :relative, align = (:left, :top), fontsize = 12)
    axislegend(ax4; position = :rt)

    return fig
end

# =============================================================================
# 0d) Console styling helpers
#
# Plain ANSI escape codes wrapped around a string, so that parts of an `@info`
# / `println` message stand out in the terminal (REPL, VS Code, Windows
# Terminal all support them). No package dependency; the codes are written
# verbatim if the log is redirected to a file, which is harmless but visible.
# For richer styling (256 colours, backgrounds) `PrettyTables.Crayons` is
# already available: `c = crayon"yellow bold"; "$(c)text$(inv(c))"`.
# =============================================================================

"""
    _ANSI

ANSI escape sequences keyed by style name. Colours: `:red`, `:green`,
`:yellow`, `:blue`, `:magenta`, `:cyan`, `:white`; attributes: `:bold`,
`:underline`. `"\\e[0m"` (used by `cstr`) resets every style.
"""
const _ANSI = Dict(
    :red => "\e[31m", :green => "\e[32m", :yellow => "\e[33m", :blue => "\e[34m",
    :magenta => "\e[35m", :cyan => "\e[36m", :white => "\e[37m",
    :bold => "\e[1m", :underline => "\e[4m",
)

"""
    cstr(s, styles...)

Return `string(s)` wrapped in the ANSI codes of the given `styles` (symbols
from `_ANSI`, applied together) and followed by a reset, so it prints in
colour/bold when interpolated into `@info`, `@warn`, `println`, …

# Examples
```julia
@info "kᵢ range: " * cstr("(1.0 – 4.1)×10⁻⁶", :yellow, :bold)
@info cstr("Selected configuration: nz = ", :cyan) * cstr(NZ_FIXED, :yellow, :bold)
```
"""
cstr(s, styles::Symbol...) = join(_ANSI[k] for k in styles) * string(s) * "\e[0m"

"""
    section(label, title)

Print a coloured banner marking the start of a script section, with the
elapsed time since `T_START`, so the REPL output can be navigated when the
script is run top-to-bottom. `label` is the section number ("4"), `title` its
name (the same text as the comment banner above the call).
"""
function section(label, title)
    elapsed = Dates.canonicalize(round(Dates.now() - T_START, Dates.Second))
    line = "═"^96
    println("\n", cstr(line, :blue, :bold))
    println(cstr("  $(label))  $(title)", :blue, :bold),
            cstr("   [t = $(elapsed)]", :blue))
    println(cstr(line, :blue, :bold))
end

"""
    list_jld2_entries(path; io = stdout, show_type = true)

Print the full tree of groups and datasets stored in the JLD2 file `path`
(names only, no values) as a quick reference of *what* is in the archive and
*where* to find it. Groups are printed in bold blue, datasets in green; with
`show_type = true` each dataset is annotated with its element type and, for
arrays, its size (e.g. `Vector{Float64} (47)`), so the layout can be checked
without loading anything into the workspace.

Works on any JLD2 file, not only the run archive, e.g.

    list_jld2_entries(archive_path)
    list_jld2_entries(raw"F:\\...\\SG_ki_scale_results.jld2"; show_type = false)

Returns the vector of full dataset paths (e.g. `"fit/cqd/ki"`), which can be
used directly as keys: `jldopen(path) do f; f["fit/cqd/ki"]; end`.
"""
function list_jld2_entries(path::AbstractString; io::IO = stdout, show_type::Bool = true)
    paths = String[]

    # recursive walk: `g` is the file or a group, `prefix` its full path
    function _walk(g, prefix, depth)
        for k in keys(g)            # insertion (write) order, as stored by JLD2
            full = isempty(prefix) ? k : prefix * "/" * k
            v    = g[k]
            pad  = "  "^depth
            if v isa JLD2.Group
                println(io, pad, cstr(k * "/", :blue, :bold))
                _walk(v, full, depth + 1)
            else
                push!(paths, full)
                info = ""
                if show_type
                    info = v isa AbstractArray ?
                           "  " * cstr("$(typeof(v)) ($(join(size(v), "×")))", :white) :
                           "  " * cstr(string(typeof(v)), :white)
                end
                println(io, pad, cstr(k, :green), info)
            end
        end
    end

    println(io, cstr("Contents of ", :bold), cstr(path, :magenta))
    jldopen(path, "r") do f
        _walk(f, "", 1)
    end
    println(io, cstr("  $(length(paths)) datasets", :white))
    return paths
end


"""
    loglog_slope(I, z, σz)

dz/dI at the points `(I, z)` from a weighted cubic smoothing spline of
`log z` vs `log I` (weights `z/σz`, smoothing `s = n`). Repeated currents are
merged first (weighted mean), since Dierckx needs strictly increasing x.
"""
function loglog_slope(I, z, σz)
    # merge repeated currents
    groups = Vector{Vector{Int}}()
    for i in eachindex(I)
        if !isempty(groups) && isapprox(I[i], I[last(groups)[1]]; rtol = 1e-9, atol = 1e-12)
            push!(last(groups), i)
        else
            push!(groups, [i])
        end
    end
    lx = [mean(log.(I[g])) for g in groups]
    w  = [(z[g] ./ σz[g]) .^ 2 for g in groups]
    ly = [sum(w[k] .* log.(z[g])) / sum(w[k]) for (k, g) in enumerate(groups)]
    wk = [sqrt(sum(wi)) for wi in w]
    k  = min(3, length(lx) - 1)
    spl = Spline1D(lx, ly; w = wk ./ mean(wk), k = k, s = float(length(lx)), bc = "extrapolate")
    return z ./ I .* derivative(spl, log.(I); nu = 1)       # dz/dI = (z/I)·dln z/dln I
end


##################################################################################################
#  1) SIMULATION GRIDS
section("1", "SIMULATION GRIDS — QM table, CQD table, analysis configuration")
##################################################################################################

# -----------------------------------------------------------------------------
# Simulated coil currents (A) — common to the QM and CQD tables; non-uniform.
# -----------------------------------------------------------------------------
Icoils = [0.00,
        0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
        0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
        0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
        0.100,0.150,0.200,0.250,0.300,0.350,0.400,0.450,0.500,0.550,
        0.600,0.650,0.700,0.750,0.800,0.850,0.900,0.950,1.00
];
nI = length(Icoils);
 
# -----------------------------------------------------------------------------
# QM table: dictionary keyed by (nz_bins, gaussian_width_mm, λ0_raw); each entry
# holds the screen-profile analysis for all currents in `Icoils`.
# -----------------------------------------------------------------------------
JLD2_MyTools.summarize_meta_qm_jld2(table_qm_path)
qm_meta = JLD2_MyTools.list_keys_jld_qm(table_qm_path);
@info "QM table" n_keys = length(qm_meta.keys) nz = qm_meta.nz σw = qm_meta.σw λ0 = qm_meta.λ0
 
# -----------------------------------------------------------------------------
# CQD table: JLD2 group hierarchy /<branch>/ki=…/nz=…/gw=…/lam=… plus a `meta`
# group. `list_keys_jld_cqd` walks the hierarchy once, so the parameter lists
# below reflect what the file actually contains (not only what `meta` declares).
#   ki : induction coefficients in micro-units (×10⁻⁶ is applied in labels only)
#   nz : z-bins, σw : Gaussian smoothing width (mm), λ0 : raw smoothing,
#   λs : spline smoothing (present in `meta` only)
# -----------------------------------------------------------------------------
cqd_info = JLD2_MyTools.list_keys_jld_cqd(table_cqd_path);
cqd_meta = OrderedDict{Symbol,Any}(
    :ki => round.(cqd_info.ki, digits=3),
    :nz => cqd_info.nz,
    :σw => round.(cqd_info.σw, digits=3),
    :λ0 => round.(cqd_info.λ0, digits=3),
    :λs => haskey(cqd_info.meta, "meta/λ0_spline") ?
           round.(cqd_info.meta["meta/λ0_spline"], digits=3) : nothing,
);
n_ki = length(cqd_meta[:ki]);

# kᵢ grid indices used for the 2D interpolant / Brent bracket
const KI_START, KI_STOP = 1, 30#n_ki

ki_lo, ki_hi = cqd_meta[:ki][KI_START], cqd_meta[:ki][KI_STOP];
@info cstr("kᵢ search bracket for the fit (2D interpolant range): ", :cyan) *
      cstr("$(ki_lo)×10⁻⁶", :yellow, :bold) * cstr(" ≤ kᵢ ≤ ", :yellow) * cstr("$(ki_hi)×10⁻⁶", :yellow, :bold) *
      cstr("   (grid indices $(KI_START):$(KI_STOP), $(KI_STOP - KI_START + 1) of $(n_ki) kᵢ values)", :cyan)

# -----------------------------------------------------------------------------
# Analysis configuration: must exist in BOTH tables
# -----------------------------------------------------------------------------
meta_nz = Int.(intersect(qm_meta.nz, cqd_meta[:nz]));
meta_σw = intersect(qm_meta.σw, cqd_meta[:σw]);
meta_λ0 = intersect(qm_meta.λ0, cqd_meta[:λ0]);
@info "Common parameter grid" meta_nz meta_σw meta_λ0
@info "Selected configuration" nx_bins=NX_FIXED nz_bins=NZ_FIXED gw=σW_FIXED λ0_raw=λ0_FIXED λ0_spline=λ0_SPLINE
 
@assert NZ_FIXED in meta_nz "NZ_FIXED = $NZ_FIXED not in common nz set: $meta_nz"
@assert σW_FIXED in meta_σw "σW_FIXED = $σW_FIXED not in common gw set: $meta_σw"
@assert λ0_FIXED in meta_λ0 "λ0_FIXED = $λ0_FIXED not in common λ0 set: $meta_λ0"
missing_ki = [ki for ki in cqd_info.ki if (:up, ki, NZ_FIXED, σW_FIXED, λ0_FIXED) ∉ cqd_info.keys]
@assert isempty(missing_ki) "CQD file lacks ki = $missing_ki for the selected configuration"
 

# -----------------------------------------------------------------------------
# QM reference curve z_max(I) → cubic spline zqm(I)  (mm)
# -----------------------------------------------------------------------------
data_qm = jldopen(table_qm_path, "r") do file
    file[JLD2_MyTools.make_keypath_qm(NZ_FIXED, σW_FIXED, λ0_FIXED)]
end;
Ic_QM   = [data_qm[i][:Icoil] for i in eachindex(data_qm)];
zmax_QM = [data_qm[i][:z_max_smooth_spline_mm] for i in eachindex(data_qm)];
zqm     = Spline1D(Ic_QM, zmax_QM, k=3);


# -----------------------------------------------------------------------------
# CQD matrix z_up_ki[j, i] = z_max(Icoils[j], ki[i])  (mm), size nI × n_ki
# -----------------------------------------------------------------------------
z_up_ki = Matrix{Float64}(undef, nI, n_ki);
for (i, ki) in enumerate(cqd_meta[:ki])
    println("\t($(@sprintf("%03d", i))/$(n_ki)) loading ki=$(@sprintf("%2.1e", 1e-6*ki))")
    data_up = jldopen(table_cqd_path, "r") do file
        file[JLD2_MyTools.make_keypath_cqd(:up, ki, NZ_FIXED, σW_FIXED, λ0_FIXED)]
    end
    z_up_ki[:, i] = [data_up[l][:z_max_smooth_spline_mm] for l in 1:nI]
end



##################################################################################################
#  2) THE (I, kᵢ) INTERPOLANT AND DIRECTORY-INDEPENDENT FIGURES (saved in RUN_OUTDIR)
section("2", "THE (I, kᵢ) INTERPOLANT — fig001, fig002 (common to all directories)")
##################################################################################################

# -----------------------------------------------------------------------------
# fig001 — CQD z_max(kᵢ) slices vs QM at selected currents
# -----------------------------------------------------------------------------
# currents (A) of the three panels: top (linear, μm) | middle (log) | bottom (log).
# One colour per current from the :rainbow palette; the reddest goes to the top panel.
I_compared = [[0.0], [0.001, 0.005, 0.010, 0.015], [0.850, 0.900, 0.950, 1.000]];
fig = plot_cqd_vs_qm(z_up_ki, zmax_QM, Icoils, cqd_meta[:ki], I_compared);
save_fig(fig, joinpath(RUN_OUTDIR, "fig001.$(FIG_EXT)"); show = true)




# -----------------------------------------------------------------------------
# 2D cubic spline ki_itp(I, kᵢ) → z_max (mm) on the kᵢ sub-range
# KI_START:KI_STOP (exact interpolation, s = 0). Dierckx expects
# Spline2D(x, y, z) with z[i, j] = z(x[i], y[j]), matching z_up_ki's layout.
# -----------------------------------------------------------------------------
ki_bounds = (ki_lo, ki_hi);
@info "kᵢ interpolation range (micro-units)" ki_min = ki_bounds[1] ki_max = ki_bounds[2] n = KI_STOP - KI_START + 1
ki_itp = Spline2D(Icoils, cqd_meta[:ki][KI_START:KI_STOP], z_up_ki[:, KI_START:KI_STOP];
                  kx=3, ky=3, s=0.0);
 
# -----------------------------------------------------------------------------
# fig002 — interpolated surface and contour (log10 z)
# -----------------------------------------------------------------------------
# log-spaced currents: the I axes below are logarithmic, so a linear grid would
# give a few very wide cells (vertical bands) at low current
i_surface  = 10 .^ range(-3, 0; length = 301);           # 1 mA … 1 A
ki_surface = range(ki_bounds[1], ki_bounds[2]; length = 301);
Z = [ki_itp(x, y) for y in ki_surface, x in i_surface];
 
logZ    = log10.(max.(abs.(Z), 1e-12));            # clamp away from log10(0)
lo, hi  = floor(minimum(logZ)), ceil(maximum(logZ));
decades = collect(lo:1:hi);
# fixed colour range of fig002 (mm): values outside it take the end colours
Z_CRANGE   = (1e-3, 5.0);
crange     = log10.(Z_CRANGE);
cbar_vals  = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0];

# Makie wants z[i, j] at (x[i], y[j]); Z is indexed [kᵢ, I] → transpose
fit_figs = Figure(size = (1200, 1400));
ax_surf = Axis3(fit_figs[1, 1:2];
    title = "Fitting surface", titlesize = 18,
    xlabelsize = 22, ylabelsize = 22, zlabelsize = 22, xticklabelsize = 16, yticklabelsize = 16, zticklabelsize = 16,
    xlabel = L"I_{c}", ylabel = L"$k_{i}\times 10^{-6}$", zlabel = L"$z\ (\mathrm{mm})$",
    xticks = (log10.([1e-3, 1e-2, 1e-1, 1.0]), [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    zticks = (log10.([1e-3, 1e-2, 1e-1, 1.0, 10.0]), [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    azimuth = 1.30π, elevation = deg2rad(25),
    xgridcolor = (:black, 0.15), ygridcolor = (:black, 0.15), zgridcolor = (:black, 0.15));
surface!(ax_surf, log10.(i_surface), ki_surface, permutedims(logZ); colormap = :viridis, colorrange = crange);
xlims!(ax_surf, log10(8e-4), log10(2.05));
zlims!(ax_surf, log10(2e-4), log10(10.0));

# Lower panel: log10 kᵢ axis from the smallest grid kᵢ up to KI_MAX_PLOT. ki_itp
# only covers the fit bracket (≤ ki_hi), so the panel uses its own plotting-only
# spline over every grid kᵢ ≤ KI_MAX_PLOT, built in (I, log10 kᵢ) because the kᵢ
# grid is log-spaced. The fit itself is unaffected.
KI_MAX_PLOT = 100.0;                                   # (×10⁻⁶) top of the kᵢ axis
sel_plot    = findall(<=(KI_MAX_PLOT), cqd_meta[:ki]);
itp_plot    = Spline2D(Icoils, log10.(cqd_meta[:ki][sel_plot]), z_up_ki[:, sel_plot];
                       kx = 3, ky = 3, s = 0.0);
ki_cont     = 10 .^ range(log10(minimum(cqd_meta[:ki])), log10(KI_MAX_PLOT); length = 301);
logZc       = log10.(max.(abs.([itp_plot(x, log10(y)) for x in i_surface, y in ki_cont]), 1e-12));  # [I, kᵢ]

ax_cont = Axis(fit_figs[2, 1]; LOGX..., LOGY...,
    title = "Fitting contour", titlesize = 18,
    xlabelsize = 22, ylabelsize = 22, xticklabelsize = 17, yticklabelsize = 17,
    xlabel = L"$I_{c}$ (A)", ylabel = L"$k_{i}\times 10^{-6}$",
    xticks = decade_ticks([1e-3, 1e-2, 1e-1, 1.0]),
    yticks = decade_ticks(10.0 .^ (floor(Int, log10(minimum(ki_cont))):round(Int, log10(KI_MAX_PLOT)))),
    xgridvisible = false, ygridvisible = false, xminorgridvisible = false, yminorgridvisible = false);
# smooth heatmap (301 × 301 cells, no seams in the Cairo output) plus white
# labelled iso-lines at 1, 2, 5 × 10ⁿ mm: z can be read off directly, and the
# tilt of the lines shows where z depends on kᵢ
cf = heatmap!(ax_cont, collect(i_surface), ki_cont, logZc;
    colormap = :viridis, colorrange = crange);
decades_c  = floor(minimum(logZc)):ceil(maximum(logZc));
iso_levels = sort(vec([log10(m * 10.0^d) for d in decades_c, m in (1, 2, 5)]));
iso_levels = filter(l -> minimum(logZc) < l < maximum(logZc), iso_levels);
contour!(ax_cont, collect(i_surface), ki_cont, logZc;
    levels = iso_levels, color = :white, linewidth = 1.0,
    labels = true, labelsize = 15, labelfont = :bold, labelcolor = :white,
    labelformatter = l -> string(round(10^l, sigdigits = 1)));
xlims!(ax_cont, extrema(i_surface)...);           # no empty strips beside the heatmap
ylims!(ax_cont, extrema(ki_cont)...);
Colorbar(fit_figs[2, 2], cf; labelsize = 22, ticklabelsize = 16,
    ticks = (log10.(cbar_vals), string.(cbar_vals)), minorticksvisible = false,
    label = L"$ z \ \mathrm{(mm)}$");
rowsize!(fit_figs.layout, 1, Relative(0.55));
save_fig(fit_figs, joinpath(RUN_OUTDIR, "fig002.$(FIG_EXT)"); show = true)



##################################################################################################
#  PER-DIRECTORY ANALYSIS (sections 3–A)
#
#  `analyze_directory(DATA_DIR)` runs, for one experimental directory, the whole
#  chain of the single-directory analysis: load z(I) from the summary JLD2,
#  fig001, tail-convergence study (fig002), final fits, diagnostics
#  (fig003–fig006), CSV tables and the run archive, all written to
#  RUN_OUTDIR/<DATA_DIR>/. Uses the globals of sections 0–2 (zqm, ki_itp,
#  cqd_meta, …). Returns one NamedTuple row of results for the combined table.
##################################################################################################
function analyze_directory(DATA_DIR::AbstractString)
    local OUTDIR = joinpath(RUN_OUTDIR, DATA_DIR)
    isdir(OUTDIR) || mkpath(OUTDIR)
    experiment_path = summary_path(DATA_DIR)
    @assert isfile(experiment_path) "Experimental summary not found: $experiment_path"


    ##################################################################################################
    #  3) EXPERIMENTAL DATA (single directory, summary JLD2) — and fig001
    ##################################################################################################
    # The entry "<DATA_DIR>/nz=<NZ_EXP>/λ0=<λ0_EXP>" of the summary file holds, for
    # every measured current, the raw F=1 / F=2 peak positions (camera frame) with
    # their errors, the currents and their physical uncertainties. They are turned
    # into the F=1 deflection table `data` = [I, δI, z, δz] (see header, steps 1–5).
    section("3", "[$(DATA_DIR)] EXPERIMENTAL DATA — $(DATA_DIR) summary, nz = $(NZ_EXP), λ0 = $(λ0_EXP); fig001")
    ##################################################################################################

    JLD2_MyTools.show_exp_summary(experiment_path, DATA_DIR)

    exp_meta = jldopen(experiment_path, "r") do f
        (T_K = f["meta/TemperatureK"], n_I = f["meta/n_Currents"],
         nz = collect(f["meta/nz"]), λ0 = collect(f["meta/λ0"]))
    end
    @info "Experimental summary" file = experiment_path T_K = exp_meta.T_K n_currents = exp_meta.n_I nz = exp_meta.nz λ0 = exp_meta.λ0
    @assert NZ_EXP in exp_meta.nz "NZ_EXP = $NZ_EXP not in the summary file: nz = $(exp_meta.nz)"
    @assert any(x -> isapprox(x, λ0_EXP; atol = 1e-9), exp_meta.λ0) "λ0_EXP = $λ0_EXP not in the summary file: λ0 = $(exp_meta.λ0)"
    (NZ_EXP == NZ_FIXED && isapprox(λ0_EXP, λ0_FIXED; atol = 1e-9)) ||
        @warn "Experimental (nz, λ0) differs from the simulation configuration" exp = (NZ_EXP, λ0_EXP) sim = (NZ_FIXED, λ0_FIXED)

    exp_key   = JLD2_MyTools.make_keypath_exp(DATA_DIR, NZ_EXP, λ0_EXP)
    exp_entry = jldopen(experiment_path, "r") do f
        f[exp_key]
    end;
    @info "Experimental entry loaded" key = exp_key runstamp = get(exp_entry, :RUNSTAMP, "n/a")

    # ---- raw columns --------------------------------------------------------------
    ic_raw   = Float64.(exp_entry[:Currents])            # coil current (A)
    δic_phys = Float64.(exp_entry[:ErrorCurrentsPhys])   # its physical uncertainty (A)
    x1, δx1  = Float64.(exp_entry[PEAK_KEY_F1][1]), Float64.(exp_entry[PEAK_KEY_F1][2])
    x2, δx2  = Float64.(exp_entry[PEAK_KEY_F2][1]), Float64.(exp_entry[PEAK_KEY_F2][2])
    @assert length(ic_raw) == length(δic_phys) == length(x1) == length(x2) "Length mismatch in $exp_key"

    # ---- 1. peak positions in mm (magnification of the directory) -----------------
    M, δM = MyExperimentalAnalysis.mag_factor(DATA_DIR)
    zf1   = x1 ./ M
    δzf1  = sqrt.((δx1 ./ M) .^ 2 .+ (x1 .* δM ./ M^2) .^ 2)
    zf2   = x2 ./ M
    δzf2  = sqrt.((δx2 ./ M) .^ 2 .+ (x2 .* δM ./ M^2) .^ 2)

    # ---- 2. origin z0 from the zero-current repeats (as in SG2026Q2_experiment_summary.jl)
    # At I = 0 the F=1 and F=2 peaks coincide, so the midpoint of each repeat is the
    # undeflected beam position:
    #   (a) select the repeats at I = 0 (fall back to the lowest current, with a
    #       warning, if a directory has no zero-current point);
    #   (b) midpoint of each repeat   c_k = (z_F1,k + z_F2,k)/2,
    #       error                     δc_k = √(δz_F1,k² + δz_F2,k²)/2   (peak + magnification errors);
    #   (c) origin                    z0 = mean(c_k);
    #   (d) its error = the LARGER of
    #         propagated  √(Σ δc_k²)/n     (the SG2026Q2_experiment_summary.jl definition)
    #         scatter     std(c_k)/√n      (spread of the repeats, n ≥ 2)
    #       so that repeats scattering more than their own errors are not underestimated.
    # The summary file's `centroid_fw_mm` is NOT used: it averages the midpoint over
    # I ≥ 20 mA (weighted toward the highest currents), a different definition.
    rep_idx = findall(x -> abs(x) < 1e-9, ic_raw)                                # (a)
    I_ref   = 0.0
    if isempty(rep_idx)
        I_ref   = minimum(ic_raw)
        rep_idx = findall(x -> isapprox(x, I_ref; atol = 1e-9), ic_raw)
        @warn "No zero-current measurement in $(DATA_DIR): origin taken from the lowest current" I_ref n = length(rep_idx)
    end
    n_rep        = length(rep_idx)
    centroid     = (zf1 .+ zf2) ./ 2                                            # (b)
    err_centroid = sqrt.(δzf1 .^ 2 .+ δzf2 .^ 2) ./ 2
    z0           = mean(centroid[rep_idx])                                      # (c)
    δz0_prop     = sqrt(sum(abs2, err_centroid[rep_idx])) / n_rep               # (d)
    δz0_scat     = n_rep > 1 ? std(centroid[rep_idx]) / sqrt(n_rep) : NaN
    δz0, z0_err_method = (n_rep > 1 && δz0_scat > δz0_prop) ?
        (δz0_scat, @sprintf("scatter std/√n of the %d repeats (> propagated %.5f mm)", n_rep, δz0_prop)) :
        (δz0_prop, n_rep > 1 ?
            @sprintf("propagated peak + magnification errors, √(Σδ²)/n of %d repeats (≥ scatter %.5f mm)", n_rep, δz0_scat) :
            "propagated peak + magnification errors (single point: no scatter estimate)")
    # F1–F2 separation at I_ref: ≈ 0 if the peaks really coincide there
    Δz_ref, δΔz_ref = mean(zf2[rep_idx] .- zf1[rep_idx]), sqrt(sum(δzf1[rep_idx] .^ 2 .+ δzf2[rep_idx] .^ 2)) / n_rep
    @info cstr("Origin in the camera frame (zero-current midpoint): z0 = ", :cyan) *
          cstr(@sprintf("%.5f ± %.5f mm", z0, δz0), :yellow, :bold) *
          cstr("   (M = $(M) ± $(δM); $(n_rep) repeats at I = $(I_ref) A)", :cyan)

    # ---- 3. F=1 deflection (positive for log–log) -------------------------------------
    zF1_all  = -(zf1 .- z0)
    δzF1_all = sqrt.(δzf1 .^ 2 .+ δz0^2)

    # ---- 4. sort, threshold, positivity ------------------------------------------------
    ord  = sortperm(ic_raw)
    I_s, δIphys_s, z_s, δz_s = ic_raw[ord], δic_phys[ord], zF1_all[ord], δzF1_all[ord]

    # ---- 4a. tables of the points NOT used in the fits (printed before the main table)
    # same style as the main table below
    show_points(M, title, labels, units, fmts) = pretty_table(M;
        alignment     = :c,
        title         = title,
        column_labels = [labels, units],
        formatters    = fmts,
        style         = TextTableStyle(first_line_column_label = crayon"yellow bold",
                                       table_border = crayon"blue bold",
                                       column_label = crayon"yellow bold",
                                       title = crayon"bold red"),
        table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
        equal_data_column_widths = true,
        show_row_number_column = true,
        fit_table_in_display_horizontally = false, fit_table_in_display_vertically = false)
    zf1_s, zf2_s = zf1[ord], zf2[ord]

    # (i) zero-current repeats: raw F=1 / F=2 positions, their midpoint, and the
    #     F=1 deflection relative to the origin z0
    i0 = findall(x -> abs(x) < 1e-9, I_s)
    if isempty(i0)
        println(cstr("  No zero-current points in $(DATA_DIR) (lowest current: $(minimum(I_s)) A)", :cyan))
    else
        show_points(hcat(I_s[i0], δIphys_s[i0], zf1_s[i0], zf2_s[i0], (zf1_s[i0] .+ zf2_s[i0]) ./ 2, z_s[i0], δz_s[i0]),
            "ZERO-CURRENT DATA — $(DATA_DIR)  ($(length(i0)) point(s))",
            ["Ic", "δIc (phys)", "z F=1 raw", "z F=2 raw", "midpoint", "z", "δz (pos.)"],
            ["(A)", "(A)", "(mm)", "(mm)", "(mm)", "(mm)", "(mm)"],
            [fmt__printf("%1.5f", 1:2), fmt__printf("%1.5f", 3:7)])
        println("    z F=1/F=2 raw : peak positions in the camera frame / M;  midpoint = (F=1 + F=2)/2")
        println("    z = −(z F=1 raw − z0) : F=1 deflection relative to the origin;  ideally ≈ 0 at I = 0")
        println("    origin z0 = mean of the $(n_rep) midpoints = $(@sprintf("%.5f ± %.5f", z0, δz0)) mm")
        println("      error: ", z0_err_method)
        println("      F=2 − F=1 separation at I = 0: $(@sprintf("%.5f ± %.5f", Δz_ref, δΔz_ref)) mm  (≈ 0 if the peaks coincide)")
    end

    # (ii) currents 0 < I < I_THRESHOLD: measured but discarded from all fits
    ib = findall(x -> 1e-9 <= abs(x) < I_THRESHOLD, I_s)
    if isempty(ib)
        println(cstr("  No points with 0 < I < $(I_THRESHOLD) A in $(DATA_DIR)", :cyan))
    else
        show_points(hcat(I_s[ib], δIphys_s[ib], z_s[ib], δz_s[ib], 100 .* δz_s[ib] ./ abs.(z_s[ib])),
            "DISCARDED DATA (0 < Ic < I_THRESHOLD = $(I_THRESHOLD) A) — $(DATA_DIR)  ($(length(ib)) point(s))",
            ["Ic", "δIc (phys)", "z", "δz (pos.)", "δz/|z|"],
            ["(A)", "(A)", "(mm)", "(mm)", "(%)"],
            [fmt__printf("%1.5f", 1:2), fmt__printf("%1.5f", 3:4), fmt__printf("%1.1f", [5])])
        println("    not used in any fit or goodness-of-fit statistic; δz/|z| shows how poorly")
        println("    resolved the deflection is at these currents (negative z: peak on the wrong side of z0)")
    end

    keep = (I_s .>= I_THRESHOLD) .& isfinite.(z_s) .& (z_s .> 0) .& isfinite.(δz_s)
    n_dropped_pos = count((I_s .>= I_THRESHOLD) .& .!((z_s .> 0) .& isfinite.(z_s)))
    n_dropped_pos > 0 && @warn "$(n_dropped_pos) point(s) above I_THRESHOLD dropped (non-positive or non-finite z)"
    I_s, δIphys_s, z_s, δz_s = I_s[keep], δIphys_s[keep], z_s[keep], δz_s[keep]

    # ---- 5. current uncertainty and propagation into δz -----------------------------
    δI_s = δI_MODE === :frac ? δI_FRAC .* I_s :
           δI_MODE === :phys ? δIphys_s :
           error("δI_MODE must be :frac or :phys, got $δI_MODE")


    dzdI_s   = loglog_slope(I_s, z_s, δz_s)
    δz_total = sqrt.((dzdI_s .* δI_s) .^ 2 .+ δz_s .^ 2)

    data    = hcat(I_s, δI_s, z_s, δz_total);     # [I, δI, z, δz], sorted, I ≥ I_THRESHOLD
    # kept for the final summary figure (all directories together)
    DIR_DATA[String(DATA_DIR)] = (I = copy(I_s), δI = copy(δI_s), z = copy(z_s), δz = copy(δz_total))
    N_data  = size(data, 1);
    I_scan  = logspace10(I_THRESHOLD, 1.00; n = 801);

    pretty_table(hcat(data, δz_s, dzdI_s);
        alignment     = :c,
        title         = "EXPERIMENTAL DATA — $(DATA_DIR), nz = $(NZ_EXP), λ0 = $(λ0_EXP)  (δI mode: $(δI_MODE))",
        column_labels = [["Ic", "δIc", "z", "δz total", "δz (pos.)", "dz/dI"],
                         ["(A)", "(A)", "(mm)", "(mm)", "(mm)", "(mm/A)"]],
        formatters    = [fmt__printf("%1.5f", 1:2), fmt__printf("%1.4f", 3:5), fmt__printf("%1.3f", [6])],
        style         = TextTableStyle(first_line_column_label = crayon"yellow bold",
                                       table_border = crayon"blue bold",
                                       column_label = crayon"yellow bold",
                                       title = crayon"bold red"),
        table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
        equal_data_column_widths = true,
        show_row_number_column = true)
    # brief legend of the table above
    δI_note = δI_MODE === :frac ? "δI = $(δI_FRAC) × Ic (fixed fraction, δI_MODE = :frac)" :
                                  "δI = ErrorCurrentsPhys from the summary file (δI_MODE = :phys)"
    println(cstr("  Columns:", :cyan, :bold))
    println("    Ic         coil current; points sorted, Ic ≥ $(I_THRESHOLD) A, non-positive z removed")
    println("    δIc        current uncertainty: ", δI_note)
    println("    z          F=1 deflection = −(x/M − z0), z0 = mean zero-current midpoint (see table above)")
    println("    δz (pos.)  position error only: peak error ⊕ magnification error ⊕ origin error")
    println("    dz/dI      local slope of z(I) from a weighted log–log smoothing spline")
    println("    δz total   √(δz(pos.)² + (dz/dI · δIc)²)  → used in every fit, error bar, pull and χ²")
    println(cstr("  nz, λ0 in the title: profile-extraction setting of the experiment (NZ_EXP, λ0_EXP)", :cyan))
    @info "Experimental points kept" N = N_data I_range = extrema(data[:, 1]) z_range = extrema(data[:, 3])


    # -----------------------------------------------------------------------------
    # fig001 — CQD family (one curve per kᵢ), QM curve and experiment
    # -----------------------------------------------------------------------------
    color_list = Makie.resample_cmap(:darkrainbow, n_ki);
    fig = Figure(size = (1350, 850));
    ax  = Axis(fig[1, 1]; LOGX..., LOGY...,
        xlabel = "Current (A)", ylabel = L"$z_{\mathrm{max}}$ (mm)",
        xticks = decade_ticks([1e-3, 1e-2, 1e-1, 1.0]),
        yticks = decade_ticks([1e-3, 1e-2, 1e-1, 1.0]));
    for (i, ki) in enumerate(cqd_meta[:ki])
        mask_cqd = log_mask(Icoils, z_up_ki[:, i])
        lines!(ax, Icoils[mask_cqd], z_up_ki[mask_cqd, i];
            color = color_list[i], linewidth = 1,
            label = L"$k_{i}=%$(round(ki, sigdigits=2))\times 10^{-6}$")
    end
    mask_qm = log_mask(Icoils, zmax_QM);
    lines!(ax, Icoils[mask_qm], zmax_QM[mask_qm]; color = :black, linestyle = :dashdot, linewidth = 2, label = "QM");
    mask_d = log_mask(data[:, 1], data[:, 3]);
    yerr_log!(ax, data[mask_d, 1], data[mask_d, 3], data[mask_d, 4]);
    scatter!(ax, data[mask_d, 1], data[mask_d, 3];
        marker = :circle, markersize = 9, color = :gold, strokecolor = :black, strokewidth = 0.8,
        label = "Experiment $(DATA_DIR)");
    text!(ax, 1e-2, 1;
        text = L"$n_{z} = %$(NZ_FIXED)$ | $\sigma_{\mathrm{conv}}=%$(Int(1e3*σW_FIXED))\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_FIXED)$",
        align = (:center, :center), fontsize = 16);
    Legend(fig[1, 2], ax; nbanks = 2, labelsize = 10, patchsize = (16, 8));
    save_fig(fig, joinpath(OUTDIR, "fig001.$(FIG_EXT)"))




    ##################################################################################################
    #  4) FIT SUBSET, TAIL-CONVERGENCE STUDY
    section("4", "[$(DATA_DIR)] FIT SUBSET, TAIL-CONVERGENCE STUDY (fig002)")
    ##################################################################################################

    # -----------------------------------------------------------------------------
    # Rows of `data` entering the kᵢ loss (FIT_KI_MODE), counted in points:
    #   :full      all points above I_THRESHOLD
    #   :low       first N_FRONT points   (small-deflection regime)
    #   :high      last  N_BACK  points   (asymptotic regime)
    #   :low_high  both windows, mid-current region excluded
    # The calibration tail is chosen separately (N_TAIL) and may overlap.
    # -----------------------------------------------------------------------------
    N_FRONT     = min(N_FRONT_PTS, N_data)   # points in the low-current window
    N_BACK      = min(N_BACK_PTS,  N_data)   # points in the high-current window

    low_range  = 1:N_FRONT;
    high_range = (N_data - N_BACK + 1):N_data;
    @assert last(low_range) ≤ N_data && first(high_range) ≥ 1

    fit_ki_idx = FIT_KI_MODE === :full     ? Colon() :
                 FIT_KI_MODE === :low      ? low_range :
                 FIT_KI_MODE === :high     ? high_range :
                 FIT_KI_MODE === :low_high ? vcat(low_range, high_range) :
                 error("Unknown FIT_KI_MODE = $FIT_KI_MODE")
    data_fitting = data[fit_ki_idx, :];

    # human-readable description of the selected rows: "rows a:b (I = x – y A)"
    fmtI(i) = @sprintf("%.3f", data[i, 1])
    window(r) = "rows $(first(r)):$(last(r)) (I = $(fmtI(first(r))) – $(fmtI(last(r))) A)"
    subset_desc = FIT_KI_MODE === :full     ? "all points above threshold, " * window(1:N_data) :
                  FIT_KI_MODE === :low      ? "the low-current window only, "    * window(low_range) :
                  FIT_KI_MODE === :high     ? "the high-current window only, "   * window(high_range) :
                                              "the low- and high-current windows, " *
                                              window(low_range) * " and " * window(high_range) *
                                              ", mid-current region excluded";

    @info cstr("kᵢ loss evaluated on ", :cyan) * cstr(String(FIT_KI_MODE), :yellow, :bold) *
          cstr(": " * subset_desc, :cyan) *
          cstr("  [$(size(data_fitting, 1)) of $(N_data) points]", :cyan)

    # -----------------------------------------------------------------------------
    # Tail-convergence study
    #
    # For each tail length n_tail in NTAIL_LIST (last n_tail points of `data`):
    #   • CQD : (kᵢ, s_CQD) jointly with `fit_cqd_ki_scale`  (kᵢ loss on data_fitting)
    #   • QM  :  s_QM        with `fit_qm_scale`
    # recorded vs I_min, the lowest current inside the tail. The plateau is the set
    # of tails confined to I_min ≥ I_PLATEAU; the spread of each quantity over the
    # plateau is quoted as a tail-choice systematic σ_sys.
    # -----------------------------------------------------------------------------
    n_plateau_pts = count(data[:, 1] .>= I_PLATEAU)            # points at I ≥ I_PLATEAU
    N_TAIL          = n_plateau_pts   # tail length (points) for the final fits: whole plateau region
    N_TAIL_MAX      = N_data          # longest tail in the convergence scan
    NTAIL_LIST      = 2:1:N_TAIL_MAX
    @assert 2 ≤ N_TAIL ≤ N_data "N_TAIL = $N_TAIL must be between 2 and the $N_data points above I_THRESHOLD"

    nt_lo, nt_hi = first(NTAIL_LIST), last(NTAIL_LIST)
    I_tail_lo    = data[end - nt_hi + 1, 1]     # start of the longest tail
    I_tail_hi    = data[end - nt_lo + 1, 1]     # start of the shortest tail
    I_tail_end   = data[end, 1]                 # every tail ends at the last point

    @info cstr("Tail-convergence scan: ", :cyan) *
          cstr("$(nt_lo) ≤ n_tail ≤ $(nt_hi)", :yellow, :bold) *
          cstr(" points (step $(step(NTAIL_LIST)), $(length(NTAIL_LIST)) tails); tails start between ", :cyan) *
          cstr(@sprintf("%.3f", I_tail_lo), :yellow, :bold) * cstr(" A ", :yellow) * cstr("and ", :cyan) *
          cstr(@sprintf("%.3f", I_tail_hi), :yellow, :bold) *
          cstr(" A ", :yellow) * cstr("and all end at ", :cyan) * cstr(@sprintf("%.3f", I_tail_end), :yellow, :bold) * cstr(" A", :yellow)

    scan_cqd = DataFrame(n_tail = Int[], I_min = Float64[], ki = Float64[], z_init = Float64[],
                         scale = Float64[], tail_rmse = Float64[], r2 = Float64[]);
    scan_qm  = DataFrame(n_tail = Int[], I_min = Float64[],                 z_init = Float64[],
                         scale = Float64[], tail_rmse = Float64[], r2 = Float64[]);
    for nt in NTAIL_LIST
        fc = fit_cqd_ki_scale(data, data_fitting, cqd_meta[:ki], (KI_START, KI_STOP);
                              n_tail = nt, scale_mode = TAIL_SCALE_MODE, zinit_bounds = ZINIT_FIT_BOUNDS)
        fq = fit_qm_scale(data, data_fitting, zqm;
                          n_tail = nt, scale_mode = TAIL_SCALE_MODE, zinit_bounds = ZINIT_FIT_BOUNDS)
        push!(scan_cqd, (nt, fc.tail_range[1], fc.ki, fc.z_init, fc.scale, fc.tail_rmse, fc.r2_coeff))
        push!(scan_qm,  (nt, fq.tail_range[1],        fq.z_init, fq.scale, fq.tail_rmse, fq.r2_coeff))
    end
    CSV.write(joinpath(OUTDIR, "tail_convergence_cqd.csv"), scan_cqd)
    CSV.write(joinpath(OUTDIR, "tail_convergence_qm.csv"),  scan_qm)

    # plateau statistics (centre: median for kᵢ and R², mean for s; spread: std)
    plateau_cqd = scan_cqd[scan_cqd.I_min .>= I_PLATEAU, :]
    plateau_qm  = scan_qm[ scan_qm.I_min  .>= I_PLATEAU, :]
    @assert nrow(plateau_cqd) ≥ 3 "Fewer than 3 scan points have I_min ≥ $I_PLATEAU A; lower I_PLATEAU"

    ki_ref, ki_sys  = median(plateau_cqd.ki),  std(plateau_cqd.ki)
    s_cqd,  s_cqd_σ = mean(plateau_cqd.scale), std(plateau_cqd.scale)
    s_qm,   s_qm_σ  = mean(plateau_qm.scale),  std(plateau_qm.scale)
    r2_cqd, r2_qm   = median(plateau_cqd.r2),  median(plateau_qm.r2)
    zi_cqd, zi_cqd_σ = median(plateau_cqd.z_init), std(plateau_cqd.z_init)   # mm
    zi_qm,  zi_qm_σ  = median(plateau_qm.z_init),  std(plateau_qm.z_init)    # mm

    @info "Tail-choice systematics (plateau: I_min ≥ $(I_PLATEAU) A, n = $(nrow(plateau_cqd)), scale_mode = $(TAIL_SCALE_MODE))" #=
        =# ki_CQD = (ki_ref, ki_sys) z_init_CQD_mm = (zi_cqd, zi_cqd_σ) s_CQD = (s_cqd, s_cqd_σ) #=
        =# z_init_QM_mm = (zi_qm, zi_qm_σ) s_QM = (s_qm, s_qm_σ) #=
        =# R2_CQD = r2_cqd R2_QM = r2_qm ki_scale_cov = cov(plateau_cqd.ki, plateau_cqd.scale)

    # fig002 — kᵢ (CQD) | s (CQD & QM) | R² (CQD & QM), shared x
    xlab = L"Lowest current in the calibration tail $I_{\min}$ (A)";
    # marker + line series: white-filled markers stroked in the line colour
    sl!(ax, x, y, c; marker = :circle, linestyle = :solid, label = nothing) =
        scatterlines!(ax, x, y; color = c, linestyle, linewidth = 1.5, label,
                      marker, markersize = 8, markercolor = :white, strokecolor = c, strokewidth = 0.8)
    mark_plateau!(ax) = vlines!(ax, I_PLATEAU; color = :gray, linestyle = :dot, linewidth = 1)

    fig = Figure(size = (950, 1250));
    Label(fig[0, 1], L"Convergence of $k_{i}$, $s$, $z_{\mathrm{init}}$ vs calibration-tail length ($%$(TAIL_SCALE_MODE)$ scale)";
          fontsize = 17, tellwidth = false)

    p_ki = Axis(fig[1, 1]; ylabel = L"$k_{i}\ \left(\times 10^{-6}\right)$")
    hspan!(p_ki, ki_ref - ki_sys, ki_ref + ki_sys; color = (:blue, 0.15))
    hlines!(p_ki, ki_ref; color = :blue, linestyle = :dash, linewidth = 1)
    mark_plateau!(p_ki)
    sl!(p_ki, scan_cqd.I_min, scan_cqd.ki, :blue;
        label = L"CQD: $k_{i} \pm \sigma_{\mathrm{sys}} = %$(fmt(ki_ref,4)) \pm %$(fmt(ki_sys,1))$")
    axislegend(p_ki; position = :lb, labelsize = 13)

    p_s = Axis(fig[2, 1]; ylabel = L"scale $s$ (model$/s \approx$ data)")
    hspan!(p_s, s_cqd - s_cqd_σ, s_cqd + s_cqd_σ; color = (:red, 0.15))
    hlines!(p_s, s_cqd; color = :red, linestyle = :dash, linewidth = 1)
    hspan!(p_s, s_qm - s_qm_σ, s_qm + s_qm_σ; color = (:purple, 0.15))
    hlines!(p_s, s_qm; color = :purple, linestyle = :dash, linewidth = 1)
    mark_plateau!(p_s)
    sl!(p_s, scan_cqd.I_min, scan_cqd.scale, :red;
        label = L"CQD: $\bar{s} \pm \sigma_{\mathrm{sys}} = %$(fmt(s_cqd,4)) \pm %$(fmt(s_cqd_σ,1))$")
    sl!(p_s, scan_qm.I_min, scan_qm.scale, :purple; marker = :diamond, linestyle = :dashdot,
        label = L"QM: $\bar{s} \pm \sigma_{\mathrm{sys}} = %$(fmt(s_qm,4)) \pm %$(fmt(s_qm_σ,1))$")
    axislegend(p_s; position = :lc, labelsize = 13)

    # z_init (μm) of both models vs tail length
    p_zi = Axis(fig[3, 1]; ylabel = L"$z_{\mathrm{init}}$ ($\mathrm{\mu m}$)")
    hlines!(p_zi, 0.0; color = :black, linestyle = :dash, linewidth = 1)
    mark_plateau!(p_zi)
    sl!(p_zi, scan_cqd.I_min, 1e3 .* scan_cqd.z_init, :red;
        label = L"CQD: $z_{\mathrm{init}} = %$(fmt(1e3*zi_cqd,3)) \pm %$(fmt(1e3*zi_cqd_σ,1))\ \mathrm{\mu m}$")
    sl!(p_zi, scan_qm.I_min, 1e3 .* scan_qm.z_init, :purple; marker = :diamond, linestyle = :dashdot,
        label = L"QM: $z_{\mathrm{init}} = %$(fmt(1e3*zi_qm,3)) \pm %$(fmt(1e3*zi_qm_σ,1))\ \mathrm{\mu m}$")
    axislegend(p_zi; position = :rt, labelsize = 13)

    p_r2 = Axis(fig[4, 1]; ylabel = L"$R^{2}$ (full data, calibrated model)", xlabel = xlab)
    mark_plateau!(p_r2)
    sl!(p_r2, scan_cqd.I_min, scan_cqd.r2, :red;
        label = L"CQD: $\langle R^{2}\rangle = %$(fmt(r2_cqd,5))$")
    sl!(p_r2, scan_qm.I_min, scan_qm.r2, :purple; marker = :diamond, linestyle = :dashdot,
        label = L"QM: $\langle R^{2}\rangle = %$(fmt(r2_qm,5))$")
    axislegend(p_r2; position = :lb, labelsize = 13)

    linkxaxes!(p_ki, p_s, p_zi, p_r2)
    foreach(a -> hidexdecorations!(a; grid = false, minorgrid = false, ticks = false, minorticks = false), (p_ki, p_s, p_zi))
    rowgap!(fig.layout, 6)
    save_fig(fig, joinpath(OUTDIR, "fig002_tail_convergence.$(FIG_EXT)"))



    ##################################################################################################
    #  5) FINAL FITS WITH UNCERTAINTIES (tail = last N_TAIL points)
    section("5", "[$(DATA_DIR)] FINAL FITS WITH UNCERTAINTIES — tail = last N_TAIL points")
    ##################################################################################################
    I_tail_start = data[end - N_TAIL + 1, 1]     # first current inside the calibration tail
    @info cstr("Calibration tail for the final fits: last ", :cyan) * cstr(N_TAIL, :yellow, :bold) *
          cstr(" points, ", :cyan) *
          cstr(@sprintf("%.3f", I_tail_start), :yellow, :bold) * cstr(" A ≤ I ≤ ", :cyan) *
          cstr(@sprintf("%.3f", data[end, 1]), :yellow, :bold) * cstr(" A", :cyan) *
          cstr("   (scale_mode = $(TAIL_SCALE_MODE); plateau cut I_min ≥ $(I_PLATEAU) A ", :cyan) *
          cstr(I_tail_start ≥ I_PLATEAU ? "✓ inside plateau" : "✗ OUTSIDE plateau", I_tail_start ≥ I_PLATEAU ? :green : :red, :bold) *
          cstr(")", :cyan)

    fit_cs = fit_cqd_ki_scale_with_error(ki_itp, data, data_fitting;
                 n_tail = N_TAIL, bounds = ki_bounds, zinit_bounds = ZINIT_FIT_BOUNDS,
                 scale_mode = TAIL_SCALE_MODE, use_Zse = false);
    @info "CQD: kᵢ + z_init + scale (with errors)" ki = (fit_cs.ki, fit_cs.ki_err) ci_profile = fit_cs.ci_profile #=
        =# z_init_mm = (fit_cs.z_init, fit_cs.z_init_err) z_init_ci_profile = fit_cs.z_init_ci_profile #=
        =# scale = (fit_cs.scale, fit_cs.scale_err) scale_ci_t = fit_cs.scale_ci_t #=
        =# corr_ki_scale = fit_cs.corr_ki_scale corr_ki_zinit = fit_cs.corr_ki_zinit corr_zinit_scale = fit_cs.corr_zinit_scale #=
        =# n_overlap = fit_cs.n_overlap converged = fit_cs.converged
    (fit_cs.ki - ki_bounds[1] < 1e-3 * (ki_bounds[2] - ki_bounds[1]) ||
     ki_bounds[2] - fit_cs.ki < 1e-3 * (ki_bounds[2] - ki_bounds[1])) &&
        @warn "Best-fit kᵢ sits on the edge of the search bracket; widen KI_START:KI_STOP" ki = fit_cs.ki bounds = ki_bounds

    fit_qs = fit_qm_scale_with_error(zqm, data, data_fitting;
                 n_tail = N_TAIL, zinit_bounds = ZINIT_FIT_BOUNDS, scale_mode = TAIL_SCALE_MODE);
    @info "QM: z_init + scale (with errors)" z_init_mm = (fit_qs.z_init, fit_qs.z_init_err) #=
        =# z_init_ci_profile = fit_qs.z_init_ci_profile scale = (fit_qs.scale, fit_qs.scale_err) #=
        =# scale_ci_t = fit_qs.scale_ci_t corr_zinit_scale = fit_qs.corr_zinit_scale tail_rmse_mm = fit_qs.tail_rmse

    # z_init stuck on the edge of its bracket → the bracket (not the data) decides it
    if FIT_ZINIT
        zw = ZINIT_BOUNDS[2] - ZINIT_BOUNDS[1]
        for (name, zi) in (("CQD", fit_cs.z_init), ("QM", fit_qs.z_init))
            min(zi - ZINIT_BOUNDS[1], ZINIT_BOUNDS[2] - zi) < 1e-3 * zw &&
                @warn "$(name): best-fit z_init sits on the edge of ZINIT_BOUNDS; widen it" z_init = zi bounds = ZINIT_BOUNDS
        end
    end

    # calibrated model curves used by every figure below:  (z_init + model)/s
    m_qm_scan  = (fit_qs.z_init .+ zqm.(I_scan))                    ./ fit_qs.scale;
    m_cqd_scan = (fit_cs.z_init .+ ki_itp.(I_scan, Ref(fit_cs.ki))) ./ fit_cs.scale;
    mask_qm_scan  = log_mask(I_scan, m_qm_scan);      # z_init < 0 can make the lowest currents ≤ 0
    mask_cqd_scan = log_mask(I_scan, m_cqd_scan);

    cfg_legend = L"%$(DATA_DIR) | $n_{z} = %$(NZ_EXP)$, $\lambda_{0} = %$(λ0_EXP)$ (exp) | $\sigma_{\mathrm{conv}}=%$(1e3*σW_FIXED)\mathrm{\mu m}$";
    tail_legend = L"tail: $I \geq %$(round(fit_cs.tail_range[1], digits=3))$ A ($n = %$(N_TAIL)$, %$(TAIL_SCALE_MODE))";
    log_ticks   = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]);



    ##################################################################################################
    #  6) DIAGNOSTICS ON THE MEASURED POINTS
    section("6", "[$(DATA_DIR)] DIAGNOSTICS ON THE MEASURED POINTS — fig003, fig004, fig005")
    ##################################################################################################

    I_d, δI_d, z_d, σ_d = data[:, 1], data[:, 2], data[:, 3], data[:, 4];
    m_qm_d  = (fit_qs.z_init .+ zqm.(I_d))                    ./ fit_qs.scale;
    m_cqd_d = (fit_cs.z_init .+ ki_itp.(I_d, Ref(fit_cs.ki))) ./ fit_cs.scale;
    # log-space metrics below need positive predictions (guaranteed on the fit
    # subset; with FIT_KI_MODE ≠ :full a negative z_init may break it elsewhere)
    @assert all(m_qm_d .> 0) && all(m_cqd_d .> 0) "Calibrated model ≤ 0 at a measured current (z_init too negative); use FIT_KI_MODE = :full or narrow ZINIT_BOUNDS"

    # -----------------------------------------------------------------------------
    # fig003 — data with error bars + calibrated QM + calibrated CQD
    # -----------------------------------------------------------------------------
    fig = Figure(size = (900, 800));
    ax  = Axis(fig[1, 1]; LOGX..., LOGY...,
        title = "Calibrated models vs data",
        xlabel = "Current (A)", ylabel = L"$z_{\mathrm{max}}$ (mm)",
        xticks = log_ticks, yticks = log_ticks,
        xlabelsize = 18, ylabelsize = 18, xticklabelsize = 16, yticklabelsize = 16);
    lines!(ax, I_scan[mask_qm_scan], m_qm_scan[mask_qm_scan]; color = :blue, linewidth = 1.75,
        label = L"QM ($s = %$(fmt(fit_qs.scale, 4))$, $z_{\mathrm{init}} = %$(fmt(1e3*fit_qs.z_init, 3))\ \mathrm{\mu m}$)")
    yerr_log!(ax, I_d, z_d, σ_d)
    scatter!(ax, I_d, logsafe(z_d); marker = :circle, markersize = 7, color = :gray35,
        strokecolor = :black, strokewidth = 0.8, label = "Data $(DATA_DIR)")
    lines!(ax, I_scan[mask_cqd_scan], m_cqd_scan[mask_cqd_scan]; color = :red, linewidth = 2,
        label = L"CQD: $k_{i}= \left( %$(fmt(fit_cs.ki, 3)) \pm %$(fmt(fit_cs.ki_err, 1)) \right) \times 10^{-6}$, $s = %$(fmt(fit_cs.scale, 4))$, $z_{\mathrm{init}} = %$(fmt(1e3*fit_cs.z_init, 3))\ \mathrm{\mu m}$")
    xlims!(ax, 0.010, 1.05)
    axislegend(ax; position = :rb, title = cfg_legend, titlesize = 13, labelsize = 13)
    save_fig(fig, joinpath(OUTDIR, "fig003.$(FIG_EXT)"))

    # -----------------------------------------------------------------------------
    # fig004 — relative error (%) and pulls of both calibrated models
    #
    # Both models are divided by their own tail-fitted factor; the data are
    # untouched. Two complementary views of (model − exp):
    #   • relative error (model − exp)/exp   → SIZE of the discrepancy
    #   • pull (model − exp)/σ_exp           → SIGNIFICANCE in units of δz
    # Here the points are individual measurements (approximately independent),
    # so χ²_red is meaningful.
    # -----------------------------------------------------------------------------
    lbl_qm  = L"QM ($s = %$(fmt(fit_qs.scale, 4))$, $z_{\mathrm{init}} = %$(fmt(1e3*fit_qs.z_init, 3))\ \mathrm{\mu m}$)";
    lbl_cqd = L"CQD ($k_{i}=%$(fmt(fit_cs.ki, 4)) \times10^{-6}$, $s = %$(fmt(fit_cs.scale, 4))$, $z_{\mathrm{init}} = %$(fmt(1e3*fit_cs.z_init, 3))\ \mathrm{\mu m}$)";
    axis_kw = (LOGX...,
        xticks = decade_ticks([1e-2, 1e-1, 1.0]),
        xlabelsize = 18, ylabelsize = 18, xticklabelsize = 16, yticklabelsize = 16, titlesize = 18)
    # filled markers stroked in their own colour, connecting line semi-transparent
    sl_fill!(ax, x, y, c; marker = :circle, label = nothing) =
        scatterlines!(ax, x, y; color = (c, 0.4), linewidth = 1.5, label,
                      marker, markersize = 8, markercolor = c, strokecolor = c, strokewidth = 0.8)

    println("Model comparison — log-space metrics (calibrated models vs raw data)")
    compare_datasets(I_d, z_d, m_cqd_d, m_qm_d; plot_errors = true);

    re_qm  = 100 .* relerr(m_qm_d,  z_d);
    re_cqd = 100 .* relerr(m_cqd_d, z_d);

    pull_qm  = pull(m_qm_d,  z_d, σ_d);
    pull_cqd = pull(m_cqd_d, z_d, σ_d);
    # reduced χ² with the fitted parameters subtracted: χ²/(N − k),
    # k = N_PAR_CQD (kᵢ, s, z_init) or N_PAR_QM (s, z_init) — same as section 7
    chi2red(p, k) = sum(abs2, p) / max(length(p) - k, 1)
    pull_summary(p, k) = (mean = mean(p), std = std(p), chi2 = sum(abs2, p), dof = length(p) - k,
                          chi2_red = chi2red(p, k),
                          frac_within_1σ = mean(abs.(p) .<= 1), frac_within_2σ = mean(abs.(p) .<= 2))
    @info "Pulls — QM  (k = $(N_PAR_QM))"  pull_summary(pull_qm,  N_PAR_QM)...
    @info "Pulls — CQD (k = $(N_PAR_CQD))" pull_summary(pull_cqd, N_PAR_CQD)...

    fig = Figure(size = (1350, 850));
    fig_rel = Axis(fig[1, 1]; axis_kw..., title = "Relative error of the calibrated models",
        xlabel = "Current (A)", ylabel = L"(\mathrm{model}-\mathrm{exp})/\mathrm{exp}\ (\%)")
    hlines!(fig_rel, 0.0; color = :black, linestyle = :dash, linewidth = 1)
    sl_fill!(fig_rel, I_d, re_qm,  :blue; label = lbl_qm)
    sl_fill!(fig_rel, I_d, re_cqd, :red;  marker = :diamond, label = lbl_cqd)
    Legend(fig[1, 2], fig_rel; title = cfg_legend, titlesize = 12, labelsize = 13)

    fig_pull = Axis(fig[2, 1]; axis_kw..., title = "Normalised residuals (pulls)",
        xlabel = "Current (A)", ylabel = L"(\mathrm{model}-\mathrm{exp})/\sigma_{\mathrm{exp}}")
    hspan!(fig_pull, -2, 2; color = (:gray, 0.08), label = L"\pm 2\sigma")
    hspan!(fig_pull, -1, 1; color = (:gray, 0.15), label = L"\pm 1\sigma")
    hlines!(fig_pull, 0.0; color = :black, linestyle = :dash, linewidth = 1)
    sl_fill!(fig_pull, I_d, pull_qm, :blue;
        label = L"QM ($\chi^{2}_{\mathrm{red}} = %$(fmt(chi2red(pull_qm, N_PAR_QM), 3))$, $k = %$(N_PAR_QM)$)")
    sl_fill!(fig_pull, I_d, pull_cqd, :red; marker = :diamond,
        label = L"CQD ($\chi^{2}_{\mathrm{red}} = %$(fmt(chi2red(pull_cqd, N_PAR_CQD), 3))$, $k = %$(N_PAR_CQD)$)")
    Legend(fig[2, 2], fig_pull; title = cfg_legend, titlesize = 12, labelsize = 13)
    linkxaxes!(fig_rel, fig_pull)
    xlims!(fig_rel, 0.010, 1.05)
    save_fig(fig, joinpath(OUTDIR, "fig004.$(FIG_EXT)"))

    CSV.write(joinpath(OUTDIR, "model_residuals_scaled.csv"), DataFrame(
        Ic = I_d, z_exp = z_d, dz_exp = σ_d, z_QM = m_qm_d, z_CQD = m_cqd_d,
        relQM = re_qm ./ 100, relCQD = re_cqd ./ 100, pullQM = pull_qm, pullCQD = pull_cqd))

    # -----------------------------------------------------------------------------
    # fig005 — publication-style overlay
    # -----------------------------------------------------------------------------
    fig = Figure(size = (1000, 850));
    ax  = Axis(fig[1, 1]; LOGX..., LOGY...,
        title = L"Peak position ($F=1$) — %$(DATA_DIR)",
        xlabel = "Coil Current (A)", ylabel = L"$z_{\mathrm{max}}$ (mm)",
        xticks = log_ticks, yticks = log_ticks,
        xlabelsize = 18, ylabelsize = 18, xticklabelsize = 16, yticklabelsize = 16);
    xerr_log!(ax, I_d, z_d, δI_d)
    yerr_log!(ax, I_d, z_d, σ_d)
    scatter!(ax, I_d, logsafe(z_d); marker = :circle, markersize = 9, color = :darkgreen,
        strokecolor = :black, strokewidth = 0.6, label = "Experimental data")
    lines!(ax, I_scan[mask_qm_scan], m_qm_scan[mask_qm_scan]; color = :blue, linewidth = 1.75,
        label = L"Quantum mechanics: $s = %$(fmt(fit_qs.scale, 4)) \pm %$(fmt(fit_qs.scale_err, 1))$, $z_{\mathrm{init}} = (%$(fmt(1e3*fit_qs.z_init, 3)) \pm %$(fmt(1e3*fit_qs.z_init_err, 1)))\ \mathrm{\mu m}$")
    lines!(ax, I_scan[mask_cqd_scan], m_cqd_scan[mask_cqd_scan]; color = :red, linewidth = 2, linestyle = :dot,
        label = L"CoQuantum dynamics: $k_{i} = \left( %$(fmt(fit_cs.ki, 4)) \pm %$(fmt(fit_cs.ki_err, 1)) \right) \times 10^{-6}$, $s = %$(fmt(fit_cs.scale, 4)) \pm %$(fmt(fit_cs.scale_err, 1))$, $z_{\mathrm{init}} = (%$(fmt(1e3*fit_cs.z_init, 3)) \pm %$(fmt(1e3*fit_cs.z_init_err, 1)))\ \mathrm{\mu m}$")
    axislegend(ax; position = :lt, title = tail_legend, titlesize = 13, labelsize = 12)
    save_fig(fig, joinpath(OUTDIR, "fig005.$(FIG_EXT)"))



    ##################################################################################################
    #  7) PUBLICATION FIGURES AND GOODNESS OF FIT
    section("7", "[$(DATA_DIR)] PUBLICATION FIGURES, GOODNESS OF FIT (fig006)")
    ##################################################################################################
    # Measured points (propagated δz) vs the two calibrated models. The points are
    # approximately independent, so χ², p-values and AIC/BIC are meaningful here.

    # columns of the experimental table, named once for the whole section
    I_sc  = data[:, 1];      # coil current (A)
    δI_sc = data[:, 2];      # current uncertainty (A)
    z_sc  = data[:, 3];      # F=1 peak position (mm)
    δz_sc = data[:, 4];      # propagated position uncertainty (mm)

    # field-gradient axis: G(I) and δG ≈ |G(I+δI) − G(I−δI)|/2
    # (G is not linear through the origin, so applying G to δI directly would be wrong)
    gradvsI(x) = TheoreticalSimulation.GvsI(x)
    G_sc   = gradvsI.(I_sc);
    δG_sc  = abs.(gradvsI.(I_sc .+ δI_sc) .- gradvsI.(max.(I_sc .- δI_sc, 0.0))) ./ 2;
    G_scan = gradvsI.(I_scan);

    # calibrated models evaluated at the measured currents (for the GOF block)
    m_qm_sc  = m_qm_d;
    m_cqd_sc = m_cqd_d;

    # residual views (same definitions as section 6)
    re_qm_sc    = relerr(m_qm_sc,  z_sc);          # (model − exp)/exp, dimensionless
    re_cqd_sc   = relerr(m_cqd_sc, z_sc);
    pull_qm_sc  = pull(m_qm_sc,  z_sc, δz_sc);     # (model − exp)/δz
    pull_cqd_sc = pull(m_cqd_sc, z_sc, δz_sc);

    # -----------------------------------------------------------------------------
    # single_SG_comparison — vs coil current
    # -----------------------------------------------------------------------------
    # styles shared by the two single_SG_comparison figures
    sg_axis = (LOGX..., LOGY..., ylabel = L"$F=1$ peak position (mm)", yticks = log_ticks,
               xlabelsize = 20, ylabelsize = 20, xticklabelsize = 18, yticklabelsize = 18)
    function sg_series!(ax, xd, δxd, xm)
        xerr_log!(ax, xd, z_sc, δxd)
        yerr_log!(ax, xd, z_sc, δz_sc)
        scatter!(ax, xd, logsafe(z_sc); marker = :circle, markersize = 9, color = :white,
                 strokecolor = :black, strokewidth = 0.5, label = "Experimental data")
        lines!(ax, xm[mask_qm_scan], m_qm_scan[mask_qm_scan]; color = :blue, linewidth = 1.75,
               linestyle = :dash, label = "Existing models")
        lines!(ax, xm[mask_cqd_scan], m_cqd_scan[mask_cqd_scan]; color = :red, linewidth = 2,
               label = L"Coquantum dynamics: $k_{i} \approx %$(fmt(fit_cs.ki, 2)) \times 10^{-6}$")
    end

    fig = Figure(size = (900, 800));
    ax  = Axis(fig[1, 1]; sg_axis..., xlabel = "Coil Current (A)", xticks = log_ticks)
    sg_series!(ax, I_sc, δI_sc, I_scan)
    axislegend(ax; position = :rb, labelsize = 16)
    save_fig(fig, joinpath(OUTDIR, "single_SG_comparison.png"))
    save(joinpath(OUTDIR, "single_SG_comparison.svg"), fig)

    # -----------------------------------------------------------------------------
    # single_SG_comparison_vsg — vs magnetic-field gradient G(I)
    # -----------------------------------------------------------------------------
    fig = Figure(size = (900, 800));
    ax  = Axis(fig[1, 1]; sg_axis..., xlabel = "Magnetic field gradient (T/m)",
               xticks = ([5, 10, 20, 50, 100, 200, 400], ["5", "10", "20", "50", "100", "200", "400"]))
    sg_series!(ax, G_sc, δG_sc, G_scan)
    xlims!(ax, 5, 400)
    axislegend(ax; position = :lt, labelsize = 16)
    save_fig(fig, joinpath(OUTDIR, "single_SG_comparison_vsg.png"))
    save(joinpath(OUTDIR, "single_SG_comparison_vsg.svg"), fig)

    # -----------------------------------------------------------------------------
    # Tables: experiment (both axes) and calibrated model curves
    # -----------------------------------------------------------------------------
    df_exp = DataFrame(Ic = I_sc, sIc = δI_sc, G = G_sc, sG = δG_sc, zmax = z_sc, szmax = δz_sc)
    df_sim = DataFrame(Ic = I_scan, G = G_scan, QM = m_qm_scan, CQD = m_cqd_scan)
    CSV.write(joinpath(OUTDIR, "data_exp.csv"), df_exp)
    CSV.write(joinpath(OUTDIR, "data_sim.csv"), df_sim)

    # -----------------------------------------------------------------------------
    # Goodness of fit: calibrated CQD (k = N_PAR_CQD: kᵢ, s, z_init) vs
    # calibrated QM (k = N_PAR_QM: s, z_init)
    # -----------------------------------------------------------------------------
    stats_CQD = goodness_of_fit(I_sc, z_sc, m_cqd_sc; σ = δz_sc, k = N_PAR_CQD)
    stats_QM  = goodness_of_fit(I_sc, z_sc, m_qm_sc;  σ = δz_sc, k = N_PAR_QM)
    @info "Goodness of fit on $(length(z_sc)) points" ΔAIC_CQD_minus_QM = stats_CQD.AIC - stats_QM.AIC ΔBIC_CQD_minus_QM = stats_CQD.BIC - stats_QM.BIC

    metrics   = ["logMSE", "logRMSE", "R2_log", "chi2_log", "chi2_red", "p_chi2", "AIC", "BIC", "NMAD"]
    gof_table = hcat([getfield(stats_CQD, Symbol(m)) for m in metrics],
                     [getfield(stats_QM,  Symbol(m)) for m in metrics])
    lower_is_better  = Set(["logMSE", "logRMSE", "chi2_log", "chi2_red", "AIC", "BIC", "NMAD"])
    higher_is_better = Set(["R2_log", "p_chi2"])

    hl_best = TextHighlighter(      # highlight, per row, the better of the two models
        (tbl, i, j) -> begin
            (j == 1 || j == 2) || return false
            v_CQD, v_QM = tbl[i, 1], tbl[i, 2]
            (isa(v_CQD, Number) && isa(v_QM, Number)) || return false
            metric = metrics[i]
            metric in lower_is_better  && return tbl[i, j] == min(v_CQD, v_QM)
            metric in higher_is_better && return tbl[i, j] == max(v_CQD, v_QM)
            return false
        end,
        crayon"fg:black bg:#fff7a1");

    pretty_table(gof_table;
        title         = "Goodness of fit — calibrated models vs data $(DATA_DIR) (k: CQD = $(N_PAR_CQD), QM = $(N_PAR_QM))",
        column_labels = ["CQD", "QM"],
        row_labels    = metrics,
        row_label_column_alignment = :l,
        highlighters  = [hl_best],
        alignment     = [:c, :c],
        style         = TextTableStyle(first_line_column_label = crayon"yellow bold",
                                       table_border = crayon"blue bold",
                                       column_label = crayon"yellow bold",
                                       title = crayon"bold red"),
        table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
        equal_data_column_widths = true)
    CSV.write(joinpath(OUTDIR, "goodness_of_fit.csv"),
              DataFrame(metric = metrics, CQD = gof_table[:, 1], QM = gof_table[:, 2]))

    # fig006 — diagnostic panels
    fig = make_diagnostic_plots(I_sc, z_sc, m_cqd_sc, m_qm_sc, stats_CQD, stats_QM; σ = δz_sc)
    save_fig(fig, joinpath(OUTDIR, "fig006.$(FIG_EXT)"))


    ##################################################################################################
    #  SUMMARY
    #  Final numbers in one place: value ± statistical error (t-interval half-width
    #  from the `_with_error` fitters) ± tail-choice systematic (plateau spread of
    #  the convergence scan). Text in green, values in bold magenta.
    section("S", "[$(DATA_DIR)] SUMMARY — results and run archive")
    ##################################################################################################
    res(v, d = 4) = cstr(fmt(v, d), :magenta, :bold)
    pm(v, e_stat, e_sys; d = 4) = res(v, d) * cstr(" ± ", :green) * res(e_stat, 2) *
                                  cstr(" (stat) ± ", :green) * res(e_sys, 2) * cstr(" (sys)", :green)
    row(name, val) = "\n  " * cstr(rpad(name, 12), :green) * val      # "  name……  value"

    @info cstr("DATA", :green, :bold) *
          row("directory",  cstr(DATA_DIR, :magenta, :bold)) *
          row("exp config", cstr("nz = $(NZ_EXP), λ0 = $(λ0_EXP), $(PEAK_KEY_F1), origin zero-current midpoint (n = $(n_rep)), δI $(δI_MODE)", :green)) *
          row("points",     cstr("$(N_data) with I ≥ $(I_THRESHOLD) A", :green))

    @info cstr("RESULT — CQD", :green, :bold) *
          row("kᵢ",         pm(fit_cs.ki, fit_cs.ki_err, ki_sys) * cstr(" × 10⁻⁶", :green)) *
          row("z_init",     pm(1e3 * fit_cs.z_init, 1e3 * fit_cs.z_init_err, 1e3 * zi_cqd_σ; d = 3) * cstr(" μm", :green)) *
          row("scale",      pm(fit_cs.scale, fit_cs.scale_err, s_cqd_σ)) *
          row("corr(kᵢ,s)", res(fit_cs.corr_ki_scale, 2)) *
          row("corr(kᵢ,z)", res(fit_cs.corr_ki_zinit, 2)) *
          row("corr(z,s)",  res(fit_cs.corr_zinit_scale, 2)) *
          row("tail",       cstr("I ≥ $(fmt(fit_cs.tail_range[1], 3)) A, n = $(N_TAIL), $(TAIL_SCALE_MODE)", :green))

    @info cstr("RESULT — QM", :green, :bold) *
          row("z_init",     pm(1e3 * fit_qs.z_init, 1e3 * fit_qs.z_init_err, 1e3 * zi_qm_σ; d = 3) * cstr(" μm", :green)) *
          row("scale",      pm(fit_qs.scale, fit_qs.scale_err, s_qm_σ)) *
          row("tail",       cstr("I ≥ $(fmt(fit_qs.tail_range[1], 3)) A, n = $(N_TAIL), $(TAIL_SCALE_MODE)", :green))

    @info cstr("Done", :green, :bold) *
          row("elapsed",    cstr(Dates.canonicalize(Dates.now() - T_START), :magenta, :bold)) *
          row("OUTDIR",     cstr(OUTDIR, :magenta))



    # -----------------------------------------------------------------------------
    # Single-file archive of the run (JLD2)
    #
    # Same layout as the averaged-curve script, plus the experimental selection
    # (data directory, nz/λ0 of the experiment, origin, magnification). Only plain
    # scalars, strings and Float64 vectors are stored.
    #
    #   meta/                 provenance: script, run stamp, host, Julia version,
    #                         input files, experimental key
    #   settings/             analysis configuration and fit choices
    #   fit/cqd/, fit/qm/     kᵢ (SI units) and calibration factors s with
    #                         statistical (t-interval) and systematic (tail-choice) errors
    #   experiment/           measured points: I, G(I), z_F1 and uncertainties,
    #                         plus both calibrated models at the same currents
    #   model/                calibrated model curves on the dense current scan
    #   statistical_analysis/ relative errors and pulls on the measured points
    #   gof/                  goodness-of-fit table (same rows as goodness_of_fit.csv)
    section("A", "[$(DATA_DIR)] RUN ARCHIVE — single JLD2 file: settings, fits, data, models, GOF")
    # -----------------------------------------------------------------------------
    archive_path = joinpath(OUTDIR, "SG_ki_scale_results.jld2")
    jldopen(archive_path, "w") do f
        # --- provenance ---
        f["meta/script"]          = @__FILE__
        f["meta/run_stamp"]       = RUN_STAMP
        f["meta/host"]            = gethostname()
        f["meta/julia"]           = string(VERSION)
        f["meta/experiment_file"] = experiment_path
        f["meta/experiment_key"]  = exp_key
        f["meta/data_dir"]        = DATA_DIR
        f["meta/qm_file"]         = table_qm_path
        f["meta/cqd_file"]        = table_cqd_path

        # --- settings ---
        f["settings/nx"]            = NX_FIXED
        f["settings/nz"]            = NZ_FIXED
        f["settings/σw_mm"]         = σW_FIXED
        f["settings/λ0"]            = λ0_FIXED
        f["settings/λ_spline"]      = λ0_SPLINE
        f["settings/nz_exp"]        = NZ_EXP
        f["settings/λ0_exp"]        = λ0_EXP
        f["settings/peak_key"]      = String(PEAK_KEY_F1)
        f["settings/origin_mode"]   = "zero_current_midpoint"     # mean F1/F2 midpoint of the I = 0 repeats
        f["settings/z0_n_repeats"]  = n_rep
        f["settings/z0_I_ref_A"]    = I_ref
        f["settings/z0_err_method"] = z0_err_method
        f["settings/z0_mm"]         = z0
        f["settings/z0Err_mm"]      = δz0
        f["settings/magnification"]    = M
        f["settings/magnificationErr"] = δM
        f["settings/I_threshold_A"] = I_THRESHOLD
        f["settings/dI_mode"]       = String(δI_MODE)
        f["settings/dI_frac"]       = δI_FRAC
        f["settings/n_tail"]        = N_TAIL
        f["settings/ki_mode"]       = String(FIT_KI_MODE)
        f["settings/I_plateau_A"]   = I_PLATEAU
        f["settings/fit_zinit"]     = FIT_ZINIT
        f["settings/zinit_bounds_mm"] = collect(Float64, ZINIT_BOUNDS)
        f["settings/n_par_cqd"]     = N_PAR_CQD
        f["settings/n_par_qm"]      = N_PAR_QM

        # --- vertical offsets z_init (mm) of the calibrated models ---
        f["fit/cqd/z_init_mm"]         = fit_cs.z_init
        f["fit/cqd/z_init_ErrStat_mm"] = fit_cs.z_init_err
        f["fit/cqd/z_init_ErrSys_mm"]  = zi_cqd_σ
        f["fit/cqd/corr_ki_zinit"]     = fit_cs.corr_ki_zinit
        f["fit/cqd/corr_zinit_scale"]  = fit_cs.corr_zinit_scale
        f["fit/qm/z_init_mm"]          = fit_qs.z_init
        f["fit/qm/z_init_ErrStat_mm"]  = fit_qs.z_init_err
        f["fit/qm/z_init_ErrSys_mm"]   = zi_qm_σ
        f["fit/qm/corr_zinit_scale"]   = fit_qs.corr_zinit_scale

        # --- fit results (physical units: kᵢ in SI, i.e. micro-units × 1e-6) ---
        f["fit/cqd/ki"]            = fit_cs.ki      * 1e-6
        f["fit/cqd/ki_ErrStat"]    = fit_cs.ki_err  * 1e-6
        f["fit/cqd/ki_ErrSys"]     = ki_sys         * 1e-6
        f["fit/cqd/scale"]         = fit_cs.scale
        f["fit/cqd/scale_ErrStat"] = fit_cs.scale_err
        f["fit/cqd/scale_ErrSys"]  = s_cqd_σ
        f["fit/qm/scale"]          = fit_qs.scale
        f["fit/qm/scale_ErrStat"]  = fit_qs.scale_err
        f["fit/qm/scale_ErrSys"]   = s_qm_σ

        # --- experiment (I ≥ I_THRESHOLD), both x-axes ---
        f["experiment/Current_A"]      = I_sc
        f["experiment/CurrentErr_A"]   = δI_sc
        f["experiment/Gradient_Tm"]    = G_sc
        f["experiment/GradientErr_Tm"] = δG_sc
        f["experiment/zF1_mm"]         = z_sc
        f["experiment/zF1Err_mm"]      = δz_sc
        # calibrated models at the same currents (inputs of the GOF table)
        f["experiment/CQD_up_mm"]      = m_cqd_sc
        f["experiment/QM_zF1_mm"]      = m_qm_sc

        # --- calibrated model curves ((z_init + model)/s) on the dense scan I_scan ---
        f["model/Current_A"]   = I_scan
        f["model/Gradient_Tm"] = G_scan
        f["model/CQD_up_mm"]   = m_cqd_scan
        f["model/QM_zF1_mm"]   = m_qm_scan

        # --- residual views on the measured points (inputs of fig006) ---
        f["statistical_analysis/Current_A"]   = I_sc
        f["statistical_analysis/relErr_CQD"]  = re_cqd_sc         # already dimensionless
        f["statistical_analysis/relErr_QM"]   = re_qm_sc
        f["statistical_analysis/pull_CQD"]    = pull_cqd_sc
        f["statistical_analysis/pull_QM"]     = pull_qm_sc

        # --- goodness of fit (k: CQD = N_PAR_CQD, QM = N_PAR_QM) ---
        f["gof/metric"]    = metrics
        f["gof/CQD"]       = gof_table[:, 1]
        f["gof/QM"]        = gof_table[:, 2]
    end
    @info cstr("Run archived to ", :green) * cstr(archive_path, :magenta)

    archive_keys = list_jld2_entries(archive_path)     # names-only reference of the archive layout

    df = jldopen(archive_path, "r") do f
        I0 = f["experiment/Current_A"]
        DataFrame(
            Current_A   = I0,
            B_T         = TheoreticalSimulation.BvsI.(I0),
            G_Tm        = TheoreticalSimulation.GvsI.(I0),
            Experiment      = f["experiment/zF1_mm"],
            ErrorExperiment = f["experiment/zF1Err_mm"],
            CQD_up_mm       = f["experiment/CQD_up_mm"],
            QM_F1_mm        = f["experiment/QM_zF1_mm"],
        )
    end
    CSV.write(joinpath(OUTDIR,"model_data.csv"), df)


    @info cstr("Directory $(DATA_DIR) done → ", :green) * cstr(OUTDIR, :magenta)
    return (
        dir               = String(DATA_DIR),
        n_points          = N_data,
        ki                = fit_cs.ki,              # ×10⁻⁶
        ki_err            = fit_cs.ki_err,          # stat (t half-width), ×10⁻⁶
        ki_sys            = ki_sys,                 # tail-choice systematic, ×10⁻⁶
        ki_ci_lo          = fit_cs.ci_profile[1],
        ki_ci_hi          = fit_cs.ci_profile[2],
        scale_cqd         = fit_cs.scale,
        scale_cqd_err     = fit_cs.scale_err,
        scale_cqd_sys     = s_cqd_σ,
        scale_qm          = fit_qs.scale,
        scale_qm_err      = fit_qs.scale_err,
        scale_qm_sys      = s_qm_σ,
        zinit_cqd         = fit_cs.z_init,          # mm
        zinit_cqd_err     = fit_cs.z_init_err,      # stat (t half-width), mm
        zinit_cqd_sys     = zi_cqd_σ,               # tail-choice systematic, mm
        zinit_qm          = fit_qs.z_init,
        zinit_qm_err      = fit_qs.z_init_err,
        zinit_qm_sys      = zi_qm_σ,
        corr_ki_zinit     = fit_cs.corr_ki_zinit,
        corr_ki_scale     = fit_cs.corr_ki_scale,
        chi2red_cqd       = stats_CQD.chi2_red,
        chi2red_qm        = stats_QM.chi2_red,
        dAIC_cqd_minus_qm = stats_CQD.AIC - stats_QM.AIC,
        dBIC_cqd_minus_qm = stats_CQD.BIC - stats_QM.BIC,
        n_tail            = N_TAIL,
        I_tail_min        = fit_cs.tail_range[1],
        z0_mm             = z0,
        outdir            = OUTDIR,
    )
end


##################################################################################################
#  LOOP OVER THE EXPERIMENTAL DIRECTORIES
#  Each directory runs sections 3–A inside `analyze_directory`; a failure in one
#  directory is reported and the loop moves on to the next.
section("LOOP", "ANALYSIS OF $(length(DATA_DIRECTORIES)) DIRECTORIES — nz = $(NZ_EXP), λ0 = $(λ0_EXP)")
##################################################################################################
dir_results = NamedTuple[]
# measured points of every directory [I, δI, z, δz], filled by analyze_directory,
# used by the final summary figure
const DIR_DATA = OrderedDict{String,NamedTuple}()
dir_failed  = Pair{String,String}[]
for (idx, dir) in enumerate(DATA_DIRECTORIES)
    println("\n", cstr("▶▶▶  DIRECTORY [$(idx)/$(length(DATA_DIRECTORIES))]  $(dir)", :magenta, :bold))
    if !isfile(summary_path(dir))
        push!(dir_failed, dir => "summary file not found")
        continue
    end
    try
        push!(dir_results, analyze_directory(dir))
    catch err
        @error "Analysis failed for $(dir)" exception = (err, catch_backtrace())
        push!(dir_failed, dir => sprint(showerror, err))
    end
end



##################################################################################################
#  ALL) COMBINED RESULTS — one row per directory
section("ALL", "COMBINED RESULTS — kᵢ and scale factors for every directory")
##################################################################################################
isempty(dir_failed) || @warn "Directories not analysed" dir_failed
@assert !isempty(dir_results) "No directory could be analysed"

df_all = DataFrame([r for r in dir_results])    # concrete NamedTuple eltype
df_all.ki_err_tot = sqrt.(df_all.ki_err .^ 2 .+ df_all.ki_sys .^ 2)    # stat ⊕ sys (×10⁻⁶)
CSV.write(joinpath(RUN_OUTDIR, "ki_all_directories.csv"), df_all)

# inverse-variance weighted mean of kᵢ over the directories (stat ⊕ sys), and
# its consistency χ²/dof (≫ 1 → directory-to-directory scatter beyond the errors)
w_ki      = 1 ./ df_all.ki_err_tot .^ 2
ki_wmean  = sum(w_ki .* df_all.ki) / sum(w_ki)
ki_wmerr  = 1 / sqrt(sum(w_ki))
ki_chi2   = sum(w_ki .* (df_all.ki .- ki_wmean) .^ 2)
ki_dof    = max(nrow(df_all) - 1, 1)
ki_mean, ki_std = mean(df_all.ki), (nrow(df_all) > 1 ? std(df_all.ki) : NaN)

tbl_all = hcat(Matrix(df_all[:, [:ki, :ki_err, :ki_sys, :scale_cqd, :scale_cqd_err, :scale_qm, :scale_qm_err]]),
               1e3 .* Matrix(df_all[:, [:zinit_cqd, :zinit_cqd_err, :zinit_qm, :zinit_qm_err]]),     # μm
               Matrix(df_all[:, [:chi2red_cqd, :chi2red_qm, :dAIC_cqd_minus_qm, :n_points, :n_tail, :I_tail_min]]))
pretty_table(tbl_all;
    title         = "kᵢ FIT PER DIRECTORY (with z_init) — nz = $(NZ_EXP), λ0 = $(λ0_EXP), tail I ≥ $(I_PLATEAU) A ($(TAIL_SCALE_MODE))",
    row_labels    = df_all.dir,
    row_label_column_alignment = :l,
    column_labels = [["kᵢ", "δkᵢ stat", "δkᵢ sys", "s CQD", "δs CQD", "s QM", "δs QM",
                      "z_init CQD", "δz_init CQD", "z_init QM", "δz_init QM",
                      "χ²red CQD", "χ²red QM", "ΔAIC", "N", "n tail", "I tail"],
                     ["×10⁻⁶", "×10⁻⁶", "×10⁻⁶", "", "", "", "", "(μm)", "(μm)", "(μm)", "(μm)",
                      "", "", "CQD−QM", "", "", "(A)"]],
    formatters    = [fmt__printf("%.4f", 1:7), fmt__printf("%.2f", 8:11), fmt__printf("%.3g", 12:14),
                     fmt__printf("%d", 15:16), fmt__printf("%.3f", [17])],
    alignment     = :c,
    style         = TextTableStyle(first_line_column_label = crayon"yellow bold",
                                   table_border = crayon"blue bold",
                                   column_label = crayon"yellow bold",
                                   title = crayon"bold red"),
    table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths = true)

@info cstr("kᵢ over $(nrow(df_all)) directories", :green, :bold) *
      "\n  " * cstr(rpad("weighted", 12), :green) * cstr(@sprintf("%.4f ± %.4f × 10⁻⁶", ki_wmean, ki_wmerr), :magenta, :bold) *
      cstr(@sprintf("   (χ²/dof = %.2f / %d)", ki_chi2, ki_dof), :green) *
      "\n  " * cstr(rpad("mean ± std", 12), :green) * cstr(@sprintf("%.4f ± %.4f × 10⁻⁶", ki_mean, ki_std), :magenta, :bold)

# -----------------------------------------------------------------------------
# FINAL TABLE — kᵢ, s_CQD and s_QM per directory as  value ± stat ± sys
#   stat : Student-t half-width from the `_with_error` fitters (95 %)
#   sys  : tail-choice systematic (plateau spread of the convergence scan)
# Last row: inverse-variance weighted mean over the directories (weights
# 1/(stat² + sys²)), quoted with its error and the consistency χ²/dof.
# Printed to the console and written to final_table.txt.
# -----------------------------------------------------------------------------
"""
    weighted_mean(v, σ) -> (mean, err, chi2, dof)

Inverse-variance weighted mean of `v` with per-point errors `σ`, its error
`1/√Σw`, and the consistency `χ² = Σ w (v − mean)²` with `dof = n − 1`.
"""
function weighted_mean(v, σ)
    w  = 1 ./ σ .^ 2
    μ  = sum(w .* v) / sum(w)
    return (mean = μ, err = 1 / sqrt(sum(w)), chi2 = sum(w .* (v .- μ) .^ 2), dof = max(length(v) - 1, 1))
end

pm3(v, s, y; d = 4) = @sprintf("%.*f ± %.*f ± %.*f", d, v, d, s, d, y)     # value ± stat ± sys
pm2(v, e; d = 4)    = @sprintf("%.*f ± %.*f", d, v, d, e)                   # value ± total

wm_ki  = weighted_mean(df_all.ki,        sqrt.(df_all.ki_err .^ 2        .+ df_all.ki_sys .^ 2))
wm_scq = weighted_mean(df_all.scale_cqd, sqrt.(df_all.scale_cqd_err .^ 2 .+ df_all.scale_cqd_sys .^ 2))
wm_sqm = weighted_mean(df_all.scale_qm,  sqrt.(df_all.scale_qm_err .^ 2  .+ df_all.scale_qm_sys .^ 2))
# z_init in μm; with FIT_ZINIT = false all errors are 0 → no weighted mean
zi_ok   = FIT_ZINIT
wm_zcq  = zi_ok ? weighted_mean(1e3 .* df_all.zinit_cqd, 1e3 .* sqrt.(df_all.zinit_cqd_err .^ 2 .+ df_all.zinit_cqd_sys .^ 2)) : nothing
wm_zqm  = zi_ok ? weighted_mean(1e3 .* df_all.zinit_qm,  1e3 .* sqrt.(df_all.zinit_qm_err .^ 2  .+ df_all.zinit_qm_sys .^ 2))  : nothing
chi2s(w) = w === nothing ? "—" : @sprintf("χ²/dof = %.1f/%d", w.chi2, w.dof)
pm2w(w; d = 4) = w === nothing ? "—" : pm2(w.mean, w.err; d = d)

final_tbl = vcat(
    hcat(df_all.dir,
         pm3.(df_all.ki,        df_all.ki_err,        df_all.ki_sys),
         pm3.(df_all.scale_cqd, df_all.scale_cqd_err, df_all.scale_cqd_sys),
         pm3.(1e3 .* df_all.zinit_cqd, 1e3 .* df_all.zinit_cqd_err, 1e3 .* df_all.zinit_cqd_sys; d = 2),
         pm3.(df_all.scale_qm,  df_all.scale_qm_err,  df_all.scale_qm_sys),
         pm3.(1e3 .* df_all.zinit_qm,  1e3 .* df_all.zinit_qm_err,  1e3 .* df_all.zinit_qm_sys;  d = 2)),
    permutedims(["Weighted mean", pm2(wm_ki.mean, wm_ki.err), pm2(wm_scq.mean, wm_scq.err), pm2w(wm_zcq; d = 2),
                 pm2(wm_sqm.mean, wm_sqm.err), pm2w(wm_zqm; d = 2)]),
    permutedims(["Consistency",   chi2s(wm_ki), chi2s(wm_scq), chi2s(wm_zcq), chi2s(wm_sqm), chi2s(wm_zqm)]),
)
final_title  = "FINAL RESULTS (with z_init) — nz = $(NZ_EXP), λ0 = $(λ0_EXP) | tail I ≥ $(I_PLATEAU) A ($(TAIL_SCALE_MODE)) | value ± stat ± sys"
final_labels = [["Directory", "CQD  kᵢ", "CQD  scale s", "CQD  z_init", "QM  scale s", "QM  z_init"],
                ["",          "(×10⁻⁶)", "",             "(μm)",        "",            "(μm)"]]
pretty_table(final_tbl;
    title         = final_title,
    column_labels = final_labels,
    alignment     = [:l, :c, :c, :c, :c, :c],
    highlighters  = [TextHighlighter((tbl, i, j) -> i > nrow(df_all), crayon"bold magenta")],
    style         = TextTableStyle(first_line_column_label = crayon"yellow bold",
                                   table_border = crayon"blue bold",
                                   column_label = crayon"yellow bold",
                                   title = crayon"bold red"),
    table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
    fit_table_in_display_horizontally = false,
    fit_table_in_display_vertically   = false)
zb_um = 1e3 .* ZINIT_BOUNDS
final_legend = """
Uncertainties:  value ± stat ± sys
  stat : statistical error of the final fit (tail I ≥ $(I_PLATEAU) A), from the `_with_error` fitters:
         Student-t half-width at 95 % confidence = t-quantile(dof) × standard error, the
         standard error coming from the covariance σ²(JᵀWJ)⁻¹ of the fit (finite-difference Jacobian).
  sys  : tail-choice systematic = standard deviation of the parameter over the plateau of the
         tail-convergence scan (all tails with I_min ≥ $(I_PLATEAU) A), i.e. how much the result
         moves when the number of high-current points used in the fit is changed.
  Weighted mean : Σwᵢvᵢ/Σwᵢ with wᵢ = 1/(statᵢ² + sysᵢ²); its error is 1/√Σwᵢ (single total ±).
  Consistency   : χ² = Σwᵢ(vᵢ − mean)², dof = n − 1; χ²/dof ≫ 1 means the directories scatter
                  more than their quoted errors.
  z_init = $(@sprintf("%.2f", zb_um[1])) or $(@sprintf("%.2f", zb_um[2])) μm means the fit stopped at the edge of ZINIT_BOUNDS
         (sys ≈ 0 then only says every tail hit the same bound, not that z_init is well determined).
"""
println(final_legend)
open(joinpath(RUN_OUTDIR, "final_table.txt"), "w") do io
    pretty_table(io, final_tbl; title = final_title, column_labels = final_labels,
                 alignment = [:l, :c, :c, :c, :c, :c],
                 table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
                 fit_table_in_display_horizontally = false,
                 fit_table_in_display_vertically   = false)
    println(io, final_legend)
end

# fig_all — kᵢ, calibration factors and z_init per directory (shared x)
xs  = 1:nrow(df_all)
fig = Figure(size = (1100, 1150))
Label(fig[0, 1], L"$n_{z} = %$(NZ_EXP)$, $\lambda_{0} = %$(λ0_EXP)$ | tail $I \geq %$(I_PLATEAU)$ A"; fontsize = 17, tellwidth = false)
dir_ticks = (collect(xs), string.(df_all.dir))
ax_kw_all = (xticks = dir_ticks, xminorgridvisible = false, xminorticksvisible = false)

p_kall = Axis(fig[1, 1]; ax_kw_all..., ylabel = L"$k_{i}\ \left(\times 10^{-6}\right)$")
hspan!(p_kall, ki_wmean - ki_wmerr, ki_wmean + ki_wmerr; color = (:blue, 0.2))
hlines!(p_kall, ki_wmean; color = :blue, linestyle = :dash, linewidth = 1,
    label = L"weighted mean $%$(fmt(ki_wmean, 4)) \pm %$(fmt(ki_wmerr, 2))$")
errorbars!(p_kall, xs, df_all.ki, df_all.ki_err_tot; color = :blue, whiskerwidth = 8)
errorbars!(p_kall, xs, df_all.ki, df_all.ki_err; color = :blue, linewidth = 2, whiskerwidth = 4)
scatter!(p_kall, xs, df_all.ki; marker = :circle, markersize = 13, color = :white, strokecolor = :blue,
    strokewidth = 1, label = L"$k_{i}$ (stat $\oplus$ sys)")
scatter!(p_kall, xs, df_all.ki; marker = :circle, markersize = 9, color = :blue, label = L"$k_{i}$ (stat)")
axislegend(p_kall; position = :rt)

p_sall = Axis(fig[2, 1]; ax_kw_all..., ylabel = L"scale $s$")
errorbars!(p_sall, xs, df_all.scale_cqd, df_all.scale_cqd_err; color = :red, whiskerwidth = 6)
scatter!(p_sall, xs, df_all.scale_cqd; marker = :circle, markersize = 11, color = :red, label = L"$s_{\mathrm{CQD}}$")
errorbars!(p_sall, xs, df_all.scale_qm, df_all.scale_qm_err; color = :purple, whiskerwidth = 6)
scatter!(p_sall, xs, df_all.scale_qm; marker = :diamond, markersize = 11, color = :purple, label = L"$s_{\mathrm{QM}}$")
axislegend(p_sall; position = :rt)

p_zall = Axis(fig[3, 1]; ax_kw_all..., ylabel = L"$z_{\mathrm{init}}$ ($\mathrm{\mu m}$)", xticklabelrotation = π / 4)
hlines!(p_zall, 0.0; color = :black, linestyle = :dash, linewidth = 1)
errorbars!(p_zall, xs, 1e3 .* df_all.zinit_cqd, 1e3 .* df_all.zinit_cqd_err; color = :red, whiskerwidth = 6)
scatter!(p_zall, xs, 1e3 .* df_all.zinit_cqd; marker = :circle, markersize = 11, color = :red, label = L"$z_{\mathrm{init}}$ CQD")
errorbars!(p_zall, xs, 1e3 .* df_all.zinit_qm, 1e3 .* df_all.zinit_qm_err; color = :purple, whiskerwidth = 6)
scatter!(p_zall, xs, 1e3 .* df_all.zinit_qm; marker = :diamond, markersize = 11, color = :purple, label = L"$z_{\mathrm{init}}$ QM")
axislegend(p_zall; position = :rt)

linkxaxes!(p_kall, p_sall, p_zall)
xlims!(p_zall, 0.5, nrow(df_all) + 0.5)
foreach(a -> hidexdecorations!(a; grid = false, ticks = false), (p_kall, p_sall))
rowgap!(fig.layout, 6)
save_fig(fig, joinpath(RUN_OUTDIR, "fig_ki_all_directories.$(FIG_EXT)"))

# -----------------------------------------------------------------------------
# fig_summary_all_directories — every directory's measured points together with
# ONE calibrated CQD curve and ONE calibrated QM curve built from the MEDIANS of
# the per-directory fits:
#     CQD: (z̃_init,CQD + ki_itp(I, k̃ᵢ)) / s̃_CQD        QM: (z̃_init,QM + zqm(I)) / s̃_QM
# (each directory was fitted with its own s and z_init; the median curves show the
#  typical calibrated model, not a joint fit of all the data)
# -----------------------------------------------------------------------------
ki_med    = median(df_all.ki);        zi_cqd_med = median(df_all.zinit_cqd);  s_cqd_med = median(df_all.scale_cqd)
zi_qm_med = median(df_all.zinit_qm);  s_qm_med   = median(df_all.scale_qm)
@info "Summary curves (medians over $(nrow(df_all)) directories)" ki_med zi_cqd_med_um = 1e3 * zi_cqd_med s_cqd_med zi_qm_med_um = 1e3 * zi_qm_med s_qm_med

I_sum      = logspace10(I_THRESHOLD, 1.00; n = 801)
m_cqd_sum  = (zi_cqd_med .+ ki_itp.(I_sum, Ref(ki_med))) ./ s_cqd_med
m_qm_sum   = (zi_qm_med  .+ zqm.(I_sum))                 ./ s_qm_med
mk_cqd_sum = log_mask(I_sum, m_cqd_sum)
mk_qm_sum  = log_mask(I_sum, m_qm_sum)

dirs_ok   = [d for d in df_all.dir if haskey(DIR_DATA, d)]
cols_dir  = Makie.resample_cmap(:viridis, max(length(dirs_ok), 2))[1:length(dirs_ok)]
mks_dir   = [:circle, :rect, :diamond, :utriangle, :dtriangle, :pentagon, :hexagon, :star5, :ltriangle, :rtriangle]

fig = Figure(size = (1250, 900))
ax  = Axis(fig[1, 1]; LOGX..., LOGY...,
    title  = L"All directories — $n_{z} = %$(NZ_EXP)$, $\lambda_{0} = %$(λ0_EXP)$, $\sigma_{\mathrm{conv}} = %$(1e3*σW_FIXED)\,\mathrm{\mu m}$",
    xlabel = "Coil Current (A)", ylabel = L"$z_{\mathrm{max}}$ (mm)",
    xticks = decade_ticks([1e-3, 1e-2, 1e-1, 1.0]),
    xminorticksvisible = true, yminorticksvisible = true,       # 2…9 × 10ⁿ (IntervalsBetween(9) from LOGX/LOGY)
    xminorticksize = 4, yminorticksize = 4,
    yticks = decade_ticks([1e-3, 1e-2, 1e-1, 1.0, 10.0]),
    xlabelsize = 22, ylabelsize = 22, xticklabelsize = 17, yticklabelsize = 17, titlesize = 18)
for (j, d) in enumerate(dirs_ok)
    dd = DIR_DATA[d]
    ok = log_mask(dd.I, dd.z)
    yerr_log!(ax, dd.I[ok], dd.z[ok], dd.δz[ok]; color = (cols_dir[j], 0.8))
    scatter!(ax, dd.I[ok], dd.z[ok]; marker = mks_dir[mod1(j, length(mks_dir))], markersize = 9,
             color = (cols_dir[j], 0.85), strokecolor = :black, strokewidth = 0.4, label = d)
end
lines!(ax, I_sum[mk_qm_sum], m_qm_sum[mk_qm_sum]; color = :blue, linewidth = 2.5, linestyle = :dash,
    label = L"QM: $\tilde{z}_{\mathrm{init}} = %$(fmt(1e3*zi_qm_med, 3))\ \mathrm{\mu m}$, $\tilde{s} = %$(fmt(s_qm_med, 4))$")
lines!(ax, I_sum[mk_cqd_sum], m_cqd_sum[mk_cqd_sum]; color = :red, linewidth = 2.5,
    label = L"CQD: $\tilde{k}_{i} = %$(fmt(ki_med, 3))\times 10^{-6}$, $\tilde{z}_{\mathrm{init}} = %$(fmt(1e3*zi_cqd_med, 3))\ \mathrm{\mu m}$, $\tilde{s} = %$(fmt(s_cqd_med, 4))$")
xlims!(ax, min(1e-2, 0.9 * I_THRESHOLD), 1.05)       # start at a decade so 10⁻² is labelled
Legend(fig[1, 2], ax; labelsize = 13)                     # ~ : median over the directories
save_fig(fig, joinpath(RUN_OUTDIR, "fig_summary_all_directories.$(FIG_EXT)"))

# combined archive (plain vectors; one entry per analysed directory)
all_path = joinpath(RUN_OUTDIR, "SG_ki_scale_all_directories.jld2")
jldopen(all_path, "w") do f
    f["meta/script"]        = @__FILE__
    f["meta/run_stamp"]     = RUN_STAMP
    f["meta/directories"]   = String.(df_all.dir)
    f["meta/failed"]        = first.(dir_failed)
    f["settings/nz_exp"]    = NZ_EXP
    f["settings/λ0_exp"]    = λ0_EXP
    f["settings/nz"]        = NZ_FIXED
    f["settings/λ0"]        = λ0_FIXED
    f["settings/σw_mm"]     = σW_FIXED
    f["settings/I_plateau_A"] = I_PLATEAU
    for c in names(df_all)
        c == "dir" && continue
        f["results/" * c] = collect(df_all[!, c])
    end
    f["combined/ki_wmean"]  = ki_wmean * 1e-6
    f["combined/ki_wmerr"]  = ki_wmerr * 1e-6
    f["combined/ki_chi2"]   = ki_chi2
    f["combined/ki_dof"]    = ki_dof
end
@info cstr("Combined results written to ", :green) * cstr(RUN_OUTDIR, :magenta) *
      cstr("   (elapsed $(Dates.canonicalize(round(Dates.now() - T_START, Dates.Second))))", :green)

