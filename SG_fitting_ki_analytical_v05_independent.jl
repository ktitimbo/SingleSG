# =============================================================================
#  Stern–Gerlach experiment — fitting the CQD induction coefficient kᵢ   (v05)
# -----------------------------------------------------------------------------
#  Kelvin Titimbo — California Institute of Technology — September 2026
#
#  PURPOSE
#  -------
#  Determine the Co-Quantum-Dynamics (CQD) induction coefficient kᵢ that best
#  reproduces the measured F=1 peak position z_max(I) of the Stern–Gerlach beam
#  as a function of coil current I, and compare the calibrated CQD prediction
#  with the quantum-mechanical (QM) reference curve on an equal footing.
#
#  CQD predictions exist only on a discrete grid of (current, kᵢ). The script
#  builds a smooth 2D interpolant over that grid, fits (kᵢ, s) to the combined
#  experimental curve — s being a global calibration (magnification) factor
#  fixed on the high-current tail — and fits the same kind of s for QM.
#
#  CALIBRATION CONVENTION (used everywhere below)
#  ----------------------------------------------
#      model(I) / s  ≈  experiment(I)
#  The experimental data are never rescaled; each model is divided by its own
#  tail-fitted factor:  s_CQD (from `fit_cqd_ki_scale*`) and s_QM (from
#  `fit_qm_scale*`). `scale_inv = 1/s` is the factor that would multiply the
#  data instead (old convention), kept for reference only.
#
#  WORKFLOW (run top-to-bottom in one Julia session)
#  --------------------------------------------------
#    0. Setup: packages, output directory, local modules, helper functions.
#    1. Load simulation grids
#         • QM  : z_max(I) for one analysis configuration (nz, σw, λ0).
#         • CQD : z_max(I, kᵢ) for every available kᵢ, same configuration.
#    2. Load the combined/averaged experimental curve z(I) ± δz (`exp_avg`)
#       and derive a "scattered" table at the grouped currents.
#    3. Build interpolants
#         • zqm(I)        : 1D cubic spline of the QM curve.
#         • ki_itp(I, kᵢ) : 2D cubic spline of the CQD grid → fit surface.
#       Sanity figures fig001–fig003.
#    4. Tail-convergence study: (kᵢ, s_CQD) and s_QM vs tail length; spread
#       over the plateau quoted as tail-choice systematic.           fig008
#    5. Final fits with formal uncertainties at the chosen `n_tail`
#       (`fit_cqd_ki_scale_with_error`, `fit_qm_scale_with_error`).
#    6. Diagnostics on the continuous curve: overlay (fig004), relative
#       error + pulls (fig005), publication overlay (fig006).
#    7. Scattered points: publication figures vs current and vs gradient,
#       goodness-of-fit table (χ², AIC/BIC, …) and diagnostic panels (fig007).
#
#  INPUTS  (external — this script does not create them)
#  ----------------------------------------------------
#    Local modules (./Modules/):
#         TheoreticalSimulation.jl   — physics helpers (GvsI gradient map)
#         DataReading.jl             — experimental analysis report locator
#         MyExperimentalAnalysis.jl  — magnification factors per dataset
#         JLD2_MyTools.jl            — JLD2 key/keypath helpers for QM & CQD
#    Data files (under BASE_PATH):
#         SIMULATIONS/2025_SETUP/QM_T205_8M/qm_screen_profiles_f1_table.jld2
#         SIMULATIONS/2025_SETUP/CQD_T205_7M/cqd_7M_up_profiles.jld2
#         EXPDATA_ANALYSIS/smoothing_binning_2025/data_averaged_2.jld2
#
#  OUTPUTS  (OUTDIR = ./data_studies/FIT2025_ki_scale_<timestamp>/)
#  ------------------------------------------------------------------
#    Figures (.<FIG_EXT>, default png)
#      fig001  CQD family + QM + combined experiment (log–log)
#      fig002  CQD z_max(kᵢ) slices vs QM at selected currents
#      fig003  interpolated (I, kᵢ) surface and contour
#      fig004  continuous data + calibrated QM + calibrated CQD (diagnostic)
#      fig005  relative error (%) and pulls vs current, both models
#      fig006  continuous data + calibrated models (publication style)
#      fig007  goodness-of-fit diagnostic panels (scattered points)
#      fig008_tail_convergence  kᵢ, s and R² vs calibration-tail length
#      single_SG_comparison(.png|.svg)      scattered points vs current
#      single_SG_comparison_vsg(.png|.svg)  scattered points vs field gradient
#    Tables (.csv)
#      tail_convergence_cqd.csv   scan of (kᵢ, s_CQD, tail RMSE, R²) vs n_tail
#      tail_convergence_qm.csv    scan of (s_QM, tail RMSE, R²) vs n_tail
#      model_residuals_scaled.csv continuous curve: data, both calibrated
#                                 models, relative errors and pulls
#      data_exp.csv               scattered experiment (I, δI, G, δG, z, δz)
#      data_sim.csv               calibrated model curves on the dense scan
#      goodness_of_fit.csv        metric table of section 7
#    Console: parameter tables, tail-scan summary, fit results with
#             statistical and tail-choice systematic errors
#
#  CONVENTIONS
#  -----------
#    • kᵢ is stored in "micro-units": physical value = (reported kᵢ) × 10⁻⁶.
#    • Experimental data matrices are N×4 with columns [I, δI, z, δz]
#      (I in A, z in mm). δI is a fraction-of-I placeholder (see δI_FRAC_*).
#    • All kᵢ fits minimise a residual in log10(z). "Errors" are documented
#      per function: `ki_err`/`scale_err` are Student-t half-widths from the
#      `_with_error` fitters; `tail_rmse`/`ki_err` of the plain fitters are
#      goodness-of-fit RMSEs in mm, not error bars.
#    • Residual sign convention everywhere: model − experiment.
# =============================================================================

# -----------------------------------------------------------------------------
# 0) Packages and global setup
# -----------------------------------------------------------------------------
using Plots; gr()
Plots.default(
    # show=true, 
    dpi=600, fontfamily="Computer Modern",
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
FIG_EXT = "png"   # "png" | "pdf" | "svg"
using Plots.PlotMeasures
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
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
const OUTDIR    = joinpath(@__DIR__, "data_studies", "FIT2025_ki_scale_" * RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
@info "Running on host" hostname = gethostname()

# Local modules (see INPUTS in the header)
include("./Modules/TheoreticalSimulation.jl");
include("./Modules/DataReading.jl");
include("./Modules/MyExperimentalAnalysis.jl");
include("./Modules/JLD2_MyTools.jl");

##################################################################################################
JLD2_MyTools.save_script_copy(OUTDIR; script_path=@__FILE__, timestamp=RUN_STAMP)
##################################################################################################

# -----------------------------------------------------------------------------
# User-adjustable analysis settings (collected here; referenced below)
# -----------------------------------------------------------------------------
# Analysis configuration shared by the QM and CQD tables
const NX_FIXED  = 128       # x-bins (bookkeeping only)
const NZ_FIXED  = 2         # z-bins used in profile extraction
const σW_FIXED  = 0.200     # (mm) Gaussian smoothing width of the profiles
const λ0_FIXED  = 0.01      # raw smoothing parameter
const λ0_SPLINE = 0.001     # spline smoothing parameter (bookkeeping only)
# Experimental data
const I_THRESHOLD  = 0.025  # (A) lowest current kept for all fits on the combined curve
const δI_FRAC = 0.001       # δI placeholder for the δI curve
# Calibration tail
const TAIL_SCALE_MODE = :log     # :linear | :log | :legacy — same for CQD and QM
const I_PLATEAU       = 0.60     # (A) plateau = tails confined to I_min ≥ I_PLATEAU



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
# 0b) Fitting routines
#
#   fit_cqd_ki_scale            — (kᵢ, s_CQD), point estimates only (fast; used in the scan)
#   fit_qm_scale                — s_QM, point estimate only (fast; used in the scan)
#   fit_cqd_ki_scale_with_error — (kᵢ, s_CQD) with t- and profile intervals
#   fit_qm_scale_with_error     — s_QM with t- and profile intervals
#
# Common ingredients:
#   • tail       = last `n_tail` rows of the full data matrix (high currents)
#   • scale_mode = how s is estimated on the tail (identical for CQD and QM):
#         :log     log10 s = ⟨log10 m − log10 z⟩       (least squares in log space)
#         :linear  s = ⟨m,m⟩/⟨z,m⟩                     (min Σ(z − m/s)², data-space residuals)
#         :legacy  s = ⟨m,z⟩/⟨z,z⟩                     (min Σ(m − s·z)², model-space residuals)
#   • kᵢ loss    = log10-space MSE of m/s vs z on the selected fit subset;
#                  s(kᵢ) is closed-form, so the joint problem is a 1D Brent search.
# =============================================================================
 
"""
    fit_cqd_ki_scale(data_org, selected_points, ki_list, ki_range;
                     n_tail, itp=ki_itp, scale_mode=:linear)
 
Two-parameter fit of the induction coefficient `kᵢ` and the calibration factor
`s`, defined so that the **scaled CQD model** `m/s` matches the raw experiment
`z`.
 
For a candidate `kᵢ`, `s(kᵢ)` is the least-squares factor that makes
`itp(I,kᵢ)/s` agree with the last `n_tail` rows of `data_org`; `kᵢ` then
minimises the log10-space MSE between `z` and `itp(I,kᵢ)/s(kᵢ)` on
`selected_points` (bounded Brent search). Because `s(kᵢ)` is closed-form the
joint (kᵢ, s) problem reduces to a 1D search.
 
# Arguments
- `data_org`        : N×4 matrix `[I, δI, z, δz]`, full data set. Its last
                      `n_tail` rows define the calibration; all rows enter R².
- `selected_points` : same layout; the subset used for the kᵢ loss.
- `ki_list`, `ki_range` : candidate kᵢ values and `(ki_start, ki_stop)`
                      indices giving the Brent bracket `ki_list[ki_start:ki_stop]`.
- `n_tail`          : number of tail points of `data_org` used for the scale.
- `itp`             : callable `(I, kᵢ) -> z_model` (default: global `ki_itp`).
- `scale_mode`      : `:linear` | `:log` | `:legacy` (see section header).
 
# Returns
NamedTuple:
- `ki`, `scale`     : best-fit pair; `m/scale` overlays the raw data.
                      `scale_inv = 1/scale` multiplies the *data* instead.
- `ki_err`          : linear-space RMSE (mm) of `m/s` vs `z` on `selected_points`
                      (goodness of fit, **not** an error bar on kᵢ)
- `tail_rmse`       : linear-space RMSE (mm) on the tail after scaling
- `r2_coeff`        : R² in linear space of `m/s` vs `z` on the full `data_org`
- `loss_log`        : log10-space MSE at the optimum
- `tail_range`      : `(I_min, I_max)` of the tail actually used
- `converged`, `result` : Optim status and raw result
"""
function fit_cqd_ki_scale(data_org, selected_points, ki_list, ki_range;
                          n_tail::Int,
                          itp = ki_itp,
                          scale_mode::Symbol = :linear)
 
    ki_start, ki_stop = ki_range
    N = size(data_org, 1)
    @assert 1 ≤ n_tail ≤ N "n_tail must be between 1 and size(data_org,1) = $N"
 
    # tail used for the calibration (last n_tail rows of the full data)
    I_tail = data_org[end-n_tail+1:end, 1]
    z_tail = data_org[end-n_tail+1:end, 3]
 
    # subset used for the kᵢ loss
    I_fit = selected_points[:, 1]
    z_fit = selected_points[:, 3]
 
    # closed-form scale s(kᵢ):  model_tail / s ≈ z_tail
    function scale_for(ki)
        m  = itp.(I_tail, Ref(ki))
        ok = isfinite.(m) .& (m .> 0)
        any(ok) || return NaN
        if scale_mode === :linear
            dot(m[ok], m[ok]) / dot(z_tail[ok], m[ok])
        elseif scale_mode === :log
            10.0 ^ mean(log10.(m[ok]) .- log10.(z_tail[ok]))
        elseif scale_mode === :legacy
            dot(m[ok], z_tail[ok]) / dot(z_tail[ok], z_tail[ok])
        else
            error("scale_mode must be :linear, :log or :legacy, got $scale_mode")
        end
    end
 
    # composite objective: log-space MSE of the scaled model on the fit subset
    function loss(ki)
        s = scale_for(ki)
        (isfinite(s) && s > 0) || return Inf
        m  = itp.(I_fit, Ref(ki))
        ok = isfinite.(m) .& (m .> 0)
        any(ok) || return Inf
        mean(abs2, log10.(m[ok] ./ s) .- log10.(z_fit[ok]))
    end
 
    # 1D bounded search over kᵢ (scale is profiled out)
    res   = optimize(loss, ki_list[ki_start], ki_list[ki_stop], Brent())
    k_fit = Optim.minimizer(res)
    s_fit = scale_for(k_fit)
 
    # diagnostics: scaled model vs raw data
    m_fit     = itp.(I_fit,  Ref(k_fit)) ./ s_fit
    rmse_fit  = sqrt(mean(abs2, m_fit .- z_fit))                  # mm
    m_tail    = itp.(I_tail, Ref(k_fit)) ./ s_fit
    rmse_tail = sqrt(mean(abs2, m_tail .- z_tail))                # mm
 
    z_all = data_org[:, 3]
    p_all = itp.(data_org[:, 1], Ref(k_fit)) ./ s_fit
    r2    = 1 - sum(abs2, p_all .- z_all) / sum(abs2, z_all .- mean(z_all))
 
    return (
        ki         = k_fit,
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
    fit_qm_scale(data_org, zqm; n_tail, scale_mode=:linear)
 
Calibration factor `s` such that the **scaled QM model** `zqm(I)/s` agrees with
the raw experiment `z` on the last `n_tail` rows of `data_org` (QM has no free
parameter, so this is the whole fit). `scale_mode` as in `fit_cqd_ki_scale`.
 
Returns `(scale, scale_inv, tail_rmse, r2_coeff, tail_range)`; `tail_rmse` and
`r2_coeff` compare `zqm/s` with the raw data.
"""
function fit_qm_scale(data_org, zqm; n_tail::Int, scale_mode::Symbol = :linear)
    N = size(data_org, 1)
    @assert 1 ≤ n_tail ≤ N "n_tail must be between 1 and size(data_org,1) = $N"
    I_tail = data_org[end-n_tail+1:end, 1]
    z_tail = data_org[end-n_tail+1:end, 3]
    m_tail = zqm.(I_tail)
 
    s = if scale_mode === :linear
        dot(m_tail, m_tail) / dot(z_tail, m_tail)
    elseif scale_mode === :log
        10.0 ^ mean(log10.(m_tail) .- log10.(z_tail))
    elseif scale_mode === :legacy
        dot(m_tail, z_tail) / dot(z_tail, z_tail)
    else
        error("scale_mode must be :linear, :log or :legacy, got $scale_mode")
    end
 
    rmse_tail = sqrt(mean(abs2, m_tail ./ s .- z_tail))
    z_all = data_org[:, 3]
    p_all = zqm.(data_org[:, 1]) ./ s
    r2    = 1 - sum(abs2, p_all .- z_all) / sum(abs2, z_all .- mean(z_all))
 
    return (scale = s, scale_inv = 1/s, tail_rmse = rmse_tail,
            r2_coeff = r2, tail_range = (first(I_tail), last(I_tail)))
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
                                n_tail, bounds, scale_mode=:log, conf=0.95,
                                use_Zse=false, profile=true, profile_grid=400)
 
Uncertainty-aware companion of `fit_cqd_ki_scale`: fits `kᵢ` and the
calibration factor `s` (convention: **scaled model** `m/s` ≈ raw data `z`) and
returns formal uncertainties on both.
 
Model. For a candidate `kᵢ` the scale `s(kᵢ)` is fixed by the last `n_tail`
rows of `data_org` (closed form, see `scale_mode`). `kᵢ` minimises the
(optionally weighted) log10-space residual sum of squares on `selected_points`,
`RSS(kᵢ) = Σ wᵢ (log10[itp(Iᵢ,kᵢ)/s(kᵢ)] − log10 Zᵢ)²`, by bounded Brent search.
With `use_Zse=true`, `wᵢ = (Zᵢ ln10 / σZᵢ)²`; otherwise all weights are 1.
 
Uncertainty on `kᵢ` (on the composite residuals, so `dr/dk` includes the
tail-driven `ds/dk`):
- linearised SE `se` from `Var(k̂) ≈ σ²/(JᵀWJ)`, Jacobian by central finite
  differences, Student-t interval `ci_t`;
- profile interval `ci_profile` where `RSS` rises by `χ²₁(conf)` (× `σ²` if
  unweighted), located by bracketing + bisection.
 
Uncertainty on `s` (in `u = log10 s`, then mapped back):
`Var(u) = Var(u | kᵢ)_tail + (du/dk)² Var(k̂)` — sampling variance of the tail
estimator at fixed `kᵢ` plus the part propagated from `kᵢ`. `scale_ci_t` is
the t-interval `s·10^(±t·se_u)`, `scale_ci_from_profile` maps `ci_profile`
through `s(kᵢ)` (propagation only), and `corr_ki_scale` is the correlation
between `k̂` and `û` induced by the tail.
 
# Arguments
- `itp`             : callable `(I, kᵢ) -> z_model` (the CQD interpolant).
- `data_org`        : N×4 matrix `[I, δI, Z, σZ]`; its last `n_tail` rows
                      define the calibration.
- `selected_points` : same layout; the subset entering the kᵢ loss.
- `n_tail`          : tail length (points) used for the scale.
- `bounds`          : `(ki_min, ki_max)` search bracket.
- `scale_mode`      : `:log` (weighted mean if `use_Zse`) | `:linear` | `:legacy`.
                      The linear modes ignore `use_Zse` for the scale and use
                      the regression residual variance instead.
- `conf`, `use_Zse`, `profile`, `profile_grid` : confidence level, weighting,
                      profile-interval switch and bracketing grid size.
 
# Returns
NamedTuple with
- kᵢ block   : `ki`, `ki_err` (= t × `se`), `se`, `ci_t`, `ci_profile`,
               `delta_target`, `delta_rss`, `profile_note`, `rss`, `sigma2`,
               `dof`, `n_used`, `r2_coeff` (weighted R² in log10 space),
               `converged`, `result`;
- scale block: `scale`, `scale_inv`, `scale_err` (= t × `scale_se`), `scale_se`,
               `scale_ci_t`, `scale_ci_from_profile`, `se_log10_scale`,
               `dlog10s_dki`, `corr_ki_scale`, `tail_rmse` (mm), `tail_range`,
               `n_tail_used`, `n_overlap` (fit points that also lie in the tail —
               if large, the two error sources are not independent and the
               scale error is optimistic).
"""
function fit_cqd_ki_scale_with_error(itp, data_org, selected_points;
    n_tail::Int,
    bounds::Tuple{<:Real,<:Real},
    scale_mode::Symbol = :log,
    conf::Real = 0.95,
    use_Zse::Bool = false,
    profile::Bool = true,
    profile_grid::Int = 400)
 
    ln10 = log(10.0)
    N = size(data_org, 1)
    @assert 1 ≤ n_tail ≤ N "n_tail must be between 1 and size(data_org,1) = $N"
 
    # ---- tail (calibration) --------------------------------------------------
    I_t  = collect(Float64, data_org[end-n_tail+1:end, 1])
    Z_t  = collect(Float64, data_org[end-n_tail+1:end, 3])
    σZ_t = collect(Float64, data_org[end-n_tail+1:end, 4])
    mt = isfinite.(I_t) .& isfinite.(Z_t) .& (Z_t .> 0)
    use_Zse && (mt .&= isfinite.(σZ_t) .& (σZ_t .> 0))
    I_t, Z_t, σZ_t = I_t[mt], Z_t[mt], σZ_t[mt]
    @assert length(I_t) ≥ 2 "Need at least 2 valid tail points"
    w_t = use_Zse ? ((Z_t .* ln10) ./ σZ_t) .^ 2 : ones(length(I_t))
 
    # ---- fit subset (kᵢ loss) -----------------------------------------------
    I  = collect(Float64, selected_points[:, 1])
    Z  = collect(Float64, selected_points[:, 3])
    σZ = collect(Float64, selected_points[:, 4])
    m0 = isfinite.(I) .& isfinite.(Z) .& (Z .> 0)
    use_Zse && (m0 .&= isfinite.(σZ) .& (σZ .> 0))
    I, Z, σZ = I[m0], Z[m0], σZ[m0]
    w = use_Zse ? ((Z .* ln10) ./ σZ) .^ 2 : ones(length(I))
 
    ki_min, ki_max = float(bounds[1]), float(bounds[2])
 
    # ---- scale s(kᵢ) and its conditional variance Var(log10 s | kᵢ) --------
    function scale_for(ki)
        m  = itp.(I_t, Ref(ki))
        ok = isfinite.(m) .& (m .> 0)
        count(ok) ≥ 2 || return (NaN, NaN)
        mo, zo, wo = m[ok], Z_t[ok], w_t[ok]
        if scale_mode === :log
            d = log10.(mo) .- log10.(zo)
            if use_Zse
                u = sum(wo .* d) / sum(wo);  var_u = 1 / sum(wo)
            else
                u = mean(d);                 var_u = var(d) / length(d)
            end
            return (10.0 ^ u, var_u)
        elseif scale_mode === :linear          # z ≈ a·m , s = 1/a
            a     = dot(zo, mo) / dot(mo, mo)
            var_a = sum(abs2, zo .- a .* mo) / (length(zo) - 1) / dot(mo, mo)
            return (1 / a, var_a / (a * ln10)^2)
        elseif scale_mode === :legacy          # m ≈ s·z
            s     = dot(mo, zo) / dot(zo, zo)
            var_s = sum(abs2, mo .- s .* zo) / (length(zo) - 1) / dot(zo, zo)
            return (s, var_s / (s * ln10)^2)
        else
            error("scale_mode must be :log, :linear or :legacy, got $scale_mode")
        end
    end
 
    # ---- composite residuals on the fit subset -------------------------------
    function residuals_for(ki)
        s, _ = scale_for(ki)
        (isfinite(s) && s > 0) || return nothing
        zpred = itp.(I, Ref(ki)) ./ s
        m = isfinite.(zpred) .& (zpred .> 0)
        any(m) || return nothing
        r = log10.(zpred[m]) .- log10.(Z[m])
        return (r = r, w = w[m], m = m, s = s)
    end
    function loss(ki)
        o = residuals_for(ki)
        o === nothing && return Inf
        sum(o.w .* (o.r .^ 2))
    end
 
    # ---- optimise kᵢ ---------------------------------------------------------
    res = optimize(loss, ki_min, ki_max, Brent())
    k̂  = Optim.minimizer(res)
 
    out0 = residuals_for(k̂)
    out0 === nothing && error("Residuals invalid at the optimum kᵢ")
    r0, w0, mfit0 = out0.r, out0.w, out0.m
 
    p = 1
    n = length(r0)
    @assert n > p "Not enough valid points to estimate uncertainty"
    RSS0 = sum(w0 .* (r0 .^ 2))
    dof  = n - p
    σ²   = RSS0 / dof
 
    # ---- finite-difference Jacobian dr/dk (composite) ------------------------
    h₀   = _fd_step(k̂, ki_min, ki_max)
    outp = residuals_for(k̂ + h₀)
    outm = residuals_for(k̂ - h₀)
    (outp === nothing || outm === nothing) &&
        error("Derivative evaluation failed near the optimum; widen bounds or check model positivity")
    mJ = mfit0 .& outp.m .& outm.m
    @assert count(mJ) > p "Not enough common points to compute the derivative"
 
    function r_on_mask(ki, mask)
        s, _ = scale_for(ki)
        zpred = itp.(I, Ref(ki)) ./ s
        log10.(zpred[mask]) .- log10.(Z[mask])
    end
    r⁺   = r_on_mask(k̂ + h₀, mJ)
    r⁻   = r_on_mask(k̂ - h₀, mJ)
    rJ   = r_on_mask(k̂,      mJ)
    wJ   = w[mJ]
    drdk = (r⁺ .- r⁻) ./ (2h₀)
 
    SJJ = sum(wJ .* (drdk .^ 2))
    se  = sqrt(σ² / SJJ)
 
    tcrit = quantile(TDist(dof), 0.5 + conf/2)
    k_err = tcrit * se
    ci_t  = (k̂ - k_err, k̂ + k_err)
 
    # weighted R² in log10 space on mJ
    y   = log10.(Z[mJ])
    ŷ   = y .+ rJ
    ȳw  = sum(wJ .* y) / sum(wJ)
    TSS = sum(wJ .* (y .- ȳw) .^ 2)
    R2  = TSS > 0 ? 1 - sum(wJ .* (y .- ŷ) .^ 2) / TSS : NaN
 
    # ---- profile interval for kᵢ ---------------------------------------------
    ci_profile   = nothing
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
        ci_profile = _profile_interval(loss, k̂, ki_min, ki_max, RSS0 + Δrss; grid = profile_grid)
    end
 
    # ---- scale and its uncertainty ------------------------------------------
    ŝ, var_u_tail = scale_for(k̂)
    û    = log10(ŝ)
    dudk = (log10(scale_for(k̂ + h₀)[1]) - log10(scale_for(k̂ - h₀)[1])) / (2h₀)
 
    var_u      = var_u_tail + dudk^2 * se^2        # tail scatter ⊕ propagated kᵢ
    se_u       = sqrt(var_u)
    scale_se   = ŝ * ln10 * se_u                   # delta method, in units of s
    scale_err  = tcrit * scale_se
    scale_ci_t = (ŝ * 10.0^(-tcrit * se_u), ŝ * 10.0^(tcrit * se_u))
    corr_ks    = se_u > 0 ? (dudk * se^2) / (se * se_u) : NaN
 
    scale_ci_from_profile = ci_profile === nothing ? nothing :
        extrema((scale_for(ci_profile[1])[1], scale_for(ci_profile[2])[1]))
 
    m_tail    = itp.(I_t, Ref(k̂)) ./ ŝ
    okt       = isfinite.(m_tail)
    tail_rmse = sqrt(mean(abs2, m_tail[okt] .- Z_t[okt]))
    n_overlap = count(I .>= first(I_t))
 
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
        n_used       = length(rJ),
        r2_coeff     = R2,
        converged    = Optim.converged(res),
        result       = res,
        # --- scale ---
        scale                 = ŝ,
        scale_inv             = 1 / ŝ,
        scale_err             = scale_err,
        scale_se              = scale_se,
        scale_ci_t            = scale_ci_t,
        scale_ci_from_profile = scale_ci_from_profile,
        se_log10_scale        = se_u,
        dlog10s_dki           = dudk,
        corr_ki_scale         = corr_ks,
        tail_rmse             = tail_rmse,
        tail_range            = (first(I_t), last(I_t)),
        n_tail_used           = length(I_t),
        n_overlap             = n_overlap,
    )
end
 
"""
    fit_qm_scale_with_error(zqm, data_org; n_tail, scale_mode=:log, conf=0.95,
                            use_Zse=false, profile=true, profile_grid=400)
 
Uncertainty-aware companion of `fit_qm_scale`: calibration factor `s` such
that the **scaled QM model** `zqm(I)/s` matches the raw experiment `z` on the
last `n_tail` rows of `data_org`, with a formal uncertainty on `s`. QM has no
free parameter, so the scale is the whole fit and its error comes from the
tail alone.
 
Estimator (`scale_mode`, as in `fit_cqd_ki_scale_with_error`):
- `:log`    → `u = log10 s = ⟨log10 m − log10 z⟩_tail` (weighted if `use_Zse`);
              `Var(u) = var(d)/n` (or `1/Σw`).
- `:linear` → `z ≈ a·m`, `s = 1/a`; `Var(a)` from the regression residuals.
- `:legacy` → `m ≈ s·z`; `Var(s)` from the residuals.
 
Uncertainty is reported two ways:
- Student-t interval `scale_ci_t = s·10^(±t·se_u)` at confidence `conf`, with
  `scale_se = s·ln10·se_u` and `scale_err = t·scale_se`.
- Profile interval `scale_ci_profile` of the log10-space weighted RSS in `u`
  (exactly quadratic in `u`, so mainly a consistency check).
 
# Arguments
- `zqm`      : callable `I -> z_QM(I)` (the QM spline).
- `data_org` : N×4 matrix `[I, δI, Z, σZ]`; `σZ` used only if `use_Zse`.
- `n_tail`, `scale_mode`, `conf`, `use_Zse`, `profile`, `profile_grid` :
  as in `fit_cqd_ki_scale_with_error`.
 
# Returns
NamedTuple: `scale`, `scale_inv`, `scale_err`, `scale_se`, `scale_ci_t`,
`scale_ci_profile`, `log10_scale`, `se_log10_scale`, `delta_target`,
`delta_rss`, `profile_note`, `rss`, `sigma2`, `dof`, `n_tail_used`,
`tail_rmse` (mm), `tail_range`, `r2_coeff` (linear-space R² of `zqm/s` vs `z`
on the full `data_org`).
"""
function fit_qm_scale_with_error(zqm, data_org;
    n_tail::Int,
    scale_mode::Symbol = :log,
    conf::Real = 0.95,
    use_Zse::Bool = false,
    profile::Bool = true,
    profile_grid::Int = 400)
 
    ln10 = log(10.0)
    N = size(data_org, 1)
    @assert 1 ≤ n_tail ≤ N "n_tail must be between 1 and size(data_org,1) = $N"
 
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
    w_t = use_Zse ? ((Z_t .* ln10) ./ σZ_t) .^ 2 : ones(n)
 
    d = log10.(m_t) .- log10.(Z_t)          # log10 ratios model/data on the tail
 
    # ---- point estimate and conditional variance -----------------------------
    ŝ, var_u = if scale_mode === :log
        if use_Zse
            u = sum(w_t .* d) / sum(w_t);  (10.0^u, 1 / sum(w_t))
        else
            u = mean(d);                   (10.0^u, var(d) / n)
        end
    elseif scale_mode === :linear                  # z ≈ a·m , s = 1/a
        a     = dot(Z_t, m_t) / dot(m_t, m_t)
        var_a = sum(abs2, Z_t .- a .* m_t) / (n - 1) / dot(m_t, m_t)
        (1 / a, var_a / (a * ln10)^2)
    elseif scale_mode === :legacy                  # m ≈ s·z
        s     = dot(m_t, Z_t) / dot(Z_t, Z_t)
        var_s = sum(abs2, m_t .- s .* Z_t) / (n - 1) / dot(Z_t, Z_t)
        (s, var_s / (s * ln10)^2)
    else
        error("scale_mode must be :log, :linear or :legacy, got $scale_mode")
    end
    û    = log10(ŝ)
    se_u = sqrt(var_u)
 
    # ---- log10-space RSS in u (σ², dof, profile) ----------------------------
    loss(u) = sum(w_t .* (d .- u) .^ 2)
    p    = 1
    RSS0 = loss(û)
    dof  = n - p
    σ²   = RSS0 / dof
 
    tcrit      = quantile(TDist(dof), 0.5 + conf/2)
    scale_se   = ŝ * ln10 * se_u
    scale_err  = tcrit * scale_se
    scale_ci_t = (ŝ * 10.0^(-tcrit * se_u), ŝ * 10.0^(tcrit * se_u))
 
    # ---- profile interval in u, mapped to s ----------------------------------
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
 
    # ---- diagnostics: scaled model vs raw data -------------------------------
    tail_rmse = sqrt(mean(abs2, m_t ./ ŝ .- Z_t))
    z_all = data_org[:, 3]
    p_all = zqm.(data_org[:, 1]) ./ ŝ
    r2    = 1 - sum(abs2, p_all .- z_all) / sum(abs2, z_all .- mean(z_all))
 
    return (
        scale            = ŝ,
        scale_inv        = 1 / ŝ,
        scale_err        = scale_err,
        scale_se         = scale_se,
        scale_ci_t       = scale_ci_t,
        scale_ci_profile = scale_ci_profile,
        log10_scale      = û,
        se_log10_scale   = se_u,
        delta_target     = Δtarget,
        delta_rss        = Δrss,
        profile_note     = profile_note,
        rss              = RSS0,
        sigma2           = σ²,
        dof              = dof,
        n_tail_used      = n,
        tail_rmse        = tail_rmse,
        tail_range       = (first(I_t), last(I_t)),
        r2_coeff         = r2,
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
        fig = plot( x_ref,
            R_B.log_err,
            seriestype=:scatter,
            label="log10(cqd) - log10(exp)",
            title="Log Errors for CQD and QM",
            xlabel="Current", ylabel="Log Error",
            markersize=3,
            markerstrokewidth=0.01,
        )
        plot!( x_ref,
            R_C.log_err,
            seriestype=:scatter,
            markersize=3,
            markerstrokewidth=0.01,
            label="log10(qm) - log10(exp)"
        )
        display(fig)
    end
 
    return (CQD = R_B, QM = R_C)
end
 
"""
    plot_cqd_vs_qm(ZCQD, ZQM, Icurrent, ki_list;
                   idx_top=[1,2,3,12], idx_bottom=[-1,-2,-3,-4],
                   palette_name=:rainbow)
 
Diagnostic figure showing, for a handful of coil currents, how the CQD
prediction varies with kᵢ against the (kᵢ-independent) QM value.
 
For each selected current index, `ZCQD[idx, :]` is drawn versus the kᵢ axis and
the QM value `ZQM[idx]` is overlaid as a dashed horizontal line. Two stacked
panels are produced: `idx_top` (small-deflection currents) and `idx_bottom`
(large-deflection currents). Negative indices count from the end of `Icurrent`.
 
Returns the combined `Plots.Plot` (shared x-axis).
"""
function plot_cqd_vs_qm(ZCQD, ZQM, Icurrent, ki_list;
        idx_top = [1, 2, 3, 12],
        idx_bottom = [-1, -2, -3, -4],
        palette_name = :rainbow
    )
 
    cls = palette(palette_name, max(length(idx_top), length(idx_bottom)))
 
    # ---------------------------
    # FIGURE A (top selected currents)
    # ---------------------------
    figa = plot()
    for (j, idx) in enumerate(idx_top)
        idx2 = idx > 0 ? idx : length(Icurrent) + idx + 1  # allow negatives
        plot!(figa, ZCQD[idx2, :],
            label = "CQD $(1000*Icurrent[idx2]) mA",
            line = (:solid, cls[j], 2))
        hline!(figa, abs.([ZQM[idx2]]),
            label = "QM",
            line = (:dash, cls[j], 1.5))
    end
    plot!(figa,
        size = (1050,500),
        yaxis = :log10,
        ylabel = L"$z_{max}$ (mm)",
        ylims = (1e-4, 1e-1),
        yticks = ([1e-4, 1e-3, 1e-2, 1e-1],
                  [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
        xlabel = L"$k_{i} \quad (\,\times 10^{-6})$",
        xtickfont = font(4),
        xticks = (1:length(ki_list), round.(ki_list, sigdigits=2)),
        xminorticks = false,
        xrotation = 88,
        bottom_margin = 4mm,
        left_margin = 3mm,
        legend = :outerright,
    )
 
    # ---------------------------
    # FIGURE B (bottom selected currents)
    # ---------------------------
    figb = plot()
    for (j, idx) in enumerate(idx_bottom)
        idx2 = idx > 0 ? idx : length(Icurrent) + idx + 1
        plot!(figb, ZCQD[idx2, :],
            label = "CQD $(1000*Icurrent[idx2]) mA",
            line = (:solid, cls[j], 2))
        hline!(figb, abs.([ZQM[idx2]]),
            label = "QM",
            line = (:dash, cls[j], 1.5))
    end
 
    plot!(figb,
        size = (1050,500),
        yaxis = :log10,
        ylabel = L"$z_{max}$ (mm)",
        ylims = (5e-1, 3),
        yticks = ([1e-1, 1.0],
                  [L"10^{-1}", L"10^{0}"]),
        xlabel = L"$k_{i} \quad (\,\times 10^{-6})$",
        xtickfont = font(4),
        xticks = (1:length(ki_list), round.(ki_list, sigdigits=2)),
        xminorticks = false,
        xrotation = 88,
        bottom_margin = 4mm,
        left_margin = 3mm,
        legend = :outerright,
    )
 
    # ---------------------------
    # Combined figure
    # ---------------------------
    fig = plot(figa, figb, layout = (2, 1),
    link=:x,
    )
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
    make_diagnostic_plots(x, y, y_CQD, y_QM, stats_CQD, stats_QM; σ=nothing)
 
Diagnostic figure for the goodness of fit of the two *calibrated* models
(CQD: `ki_itp/s_CQD`, QM: `zqm/s_QM`) against the raw scattered experiment.
 
Panels
1. data (with y-error bars if `σ` is given) vs both models, log–log;
2. natural-log residuals `log(model) − log(exp)` vs current, annotated with
   logRMSE and R²_log;
3. pulls `(model − exp)/σ` vs current with ±1σ/±2σ bands, annotated with the
   reduced χ² (only if `σ` is given; otherwise an empty placeholder);
4. histogram of the log residuals, annotated with NMAD and logRMSE.
 
Returns `(p_data, p_resid, p_pull, p_hist)`.
"""
function make_diagnostic_plots(x, y, y_CQD, y_QM, stats_CQD::FitStats, stats_QM::FitStats; σ = nothing)
    # residuals in natural-log space, sign convention model − data
    r_CQD = log.(y_CQD) .- log.(y)
    r_QM  = log.(y_QM)  .- log.(y)
 
    xt = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"])
 
    # 1) Data vs calibrated models (log–log)
    p_data = plot(x, y;
        yerror = σ,
        seriestype = :scatter,
        marker = (:circle, :white, 3, stroke(:black, 0.8)),
        xscale = :log10, yscale = :log10, xticks = xt,
        label = "Experiment",
        xlabel = "Coil Current (A)",
        ylabel = "Peak position (mm)",
        title = "Data vs calibrated models",
        legend = :bottomright,
    )
    plot!(p_data, x, y_CQD; label = "CQD", line = (:solid, :red, 1.5))
    plot!(p_data, x, y_QM;  label = "QM",  line = (:dot,   :blue, 2))
 
    # 2) Log residuals vs current
    p_resid = plot(x, r_CQD;
        seriestype = :scatter,
        marker = (:circle, 5, 0.70, :salmon3, stroke(0.8, :red4)),
        xscale = :log10, xticks = xt,
        xlabel = "Coil Current (A)",
        ylabel = L"\ln(y_{\mathrm{model}}) - \ln(y_{\mathrm{exp}})",
        title = "Log-space residuals",
        label = "CQD",
        legend = :topright,
    )
    scatter!(p_resid, x, r_QM; label = "QM",
        marker = (:circle, 5, 0.70, :royalblue3, stroke(0.8, :blue4)))
    hline!(p_resid, [0.0]; c = :black, ls = :dash, label = false)
    txt_CQD = L"CQD: logRMSE $= %$(fmt(stats_CQD.logRMSE, 3))$, $R^{2}_{\log} = %$(round(stats_CQD.R2_log, digits=4))$"
    txt_QM  = L"QM: logRMSE $= %$(fmt(stats_QM.logRMSE,  3))$, $R^{2}_{\log} = %$(round(stats_QM.R2_log,  digits=4))$"
    x_annot = x[argmin(abs.(x .- median(x)))]
    rmin, rmax = extrema(vcat(r_CQD, r_QM))
    annotate!(p_resid, (x_annot, rmin + 0.15(rmax - rmin), Plots.text(txt_CQD, 8)))
    annotate!(p_resid, (x_annot, rmin + 0.05(rmax - rmin), Plots.text(txt_QM,  8)))
 
    # 3) Pulls vs current (needs σ)
    if σ === nothing
        p_pull = plot(; title = "Pulls (no σ supplied)", framestyle = :none)
    else
        pu_CQD = pull(y_CQD, y, σ)
        pu_QM  = pull(y_QM,  y, σ)
        p_pull = plot(;
            xscale = :log10, xticks = xt,
            xlabel = "Coil Current (A)",
            ylabel = L"(y_{\mathrm{model}} - y_{\mathrm{exp}})/\sigma_{\mathrm{exp}}",
            title = "Normalised residuals (pulls)",
            legend = :topright,
        )
        hspan!(p_pull, [-2, 2]; fillalpha = 0.08, color = :gray, linealpha = 0, label = L"\pm 2\sigma")
        hspan!(p_pull, [-1, 1]; fillalpha = 0.15, color = :gray, linealpha = 0, label = L"\pm 1\sigma")
        hline!(p_pull, [0.0]; c = :black, ls = :dash, label = false)
        scatter!(p_pull, x, pu_CQD;
            label = L"CQD: $\chi^{2}_{\mathrm{red}} = %$(fmt(stats_CQD.chi2_red, 3))$",
            marker = (:circle, 5, 0.70, :salmon3, stroke(0.8, :red4)))
        scatter!(p_pull, x, pu_QM;
            label = L"QM: $\chi^{2}_{\mathrm{red}} = %$(fmt(stats_QM.chi2_red, 3))$",
            marker = (:circle, 5, 0.70, :royalblue3, stroke(0.8, :blue4)))
    end
 
    # 4) Histogram of log residuals
    p_hist = histogram(r_CQD;
        normalize = true, color = :red, alpha = 0.4,
        label = "CQD",
        xlabel = "log-space residual",
        ylabel = "Normalised count",
        title = "Distribution of log-space residuals",
        legend = :topright,
    )
    histogram!(p_hist, r_QM; normalize = true, color = :blue, alpha = 0.4, label = "QM")
    vline!(p_hist, [0.0]; c = :black, ls = :dash, lw = 1, label = false)
    txt2_CQD = @sprintf "CQD: NMAD = %.3g, logRMSE = %.3g" stats_CQD.NMAD stats_CQD.logRMSE
    txt2_QM  = @sprintf "QM:  NMAD = %.3g, logRMSE = %.3g" stats_QM.NMAD  stats_QM.logRMSE
    hx_lo, hx_hi = Plots.xlims(p_hist)
    hy_hi        = Plots.ylims(p_hist)[2]
    annotate!(p_hist, (hx_lo + 0.03(hx_hi - hx_lo), 0.90hy_hi, Plots.text(txt2_CQD, 8, :left)))
    annotate!(p_hist, (hx_lo + 0.03(hx_hi - hx_lo), 0.82hy_hi, Plots.text(txt2_QM,  8, :left)))
 
    return p_data, p_resid, p_pull, p_hist
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
table_qm_path = joinpath(BASE_PATH, "SIMULATIONS", "2025_SETUP", "QM_T205_8M",
                         "qm_screen_profiles_f1_table.jld2");
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
table_cqd_path = joinpath(BASE_PATH, "SIMULATIONS", "2025_SETUP", "CQD_T205_7M",
                          "cqd_7M_up_profiles.jld2");
cqd_info = JLD2_MyTools.list_keys_jld_cqd(table_cqd_path)
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
const KI_START, KI_STOP = 1, 41

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
#  2) EXPERIMENTAL DATA (combined / averaged curve)
##################################################################################################
# `exp_avg` holds the smoothed peak position z(I) on a dense grid (:i_smooth,
# :z_smooth, :δz_smooth) and the grouped currents (:Ic_grouped) at which the
# raw data were binned. Two views are derived:
#   • continuous curve  `data`             (built in section 4, I ≥ I_THRESHOLD)
#   • scattered table   `data_exp_scattered` at the grouped currents, with
#     δz_total² = (dz/dI · δI)² + δz²  propagated from the placeholder δI.
section("2", "EXPERIMENTAL DATA — combined curve and scattered table")
##################################################################################################


experiment_path = joinpath(BASE_PATH, "EXPDATA_ANALYSIS", "smoothing_binning_2025",
                        "data_averaged_2.jld2")
exp_avg = load(experiment_path)["data"];
@info "Experimental data loaded" n_smooth = length(exp_avg[:i_smooth]) n_grouped = size(exp_avg[:Ic_grouped], 1)
 
# rows of the smooth grid that coincide with the grouped currents
mask = [any(abs(a - b) ≤ 1e-12 for a in exp_avg[:Ic_grouped][:, 1]) for b in exp_avg[:i_smooth]];
 
# weighted interpolating spline of the smoothed curve (for the slope dz/dI)
data_experiment = Spline1D(exp_avg[:i_smooth], exp_avg[:z_smooth];
                           k=3, bc="extrapolate", s=0.0, w = 1 ./ exp_avg[:δz_smooth].^2);
 
Ichosen  = exp_avg[:i_smooth][mask];
δIchosen = δI_FRAC .* Ichosen;
dz       = derivative(data_experiment, Ichosen; nu=1);       # dz/dI at the grouped currents
δz       = exp_avg[:δz_smooth][mask];
δz_total = sqrt.((dz .* δIchosen).^2 .+ δz.^2);
data_exp_scattered = hcat(Ichosen, δIchosen, exp_avg[:z_smooth][mask], δz_total);   # [I, δI, z, δz]
 
pretty_table(data_exp_scattered;
    alignment     = :c,
    title         = "EXPERIMENTAL DATA (scattered)",
    column_labels = ["Ic (A)", "δIc (A)", "z (mm)", "δz (mm)"],
    formatters    = [fmt__printf("%1.3f", [1]), fmt__printf("%1.4f", [2]), fmt__printf("%1.3f", 3:4)],
    style         = TextTableStyle(first_line_column_label = crayon"yellow bold",
                                   table_border = crayon"blue bold",
                                   column_label = crayon"yellow bold",
                                   title = crayon"bold red"),
    table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths = true)


##################################################################################################
#  3) SANITY FIGURES AND THE (I, kᵢ) INTERPOLANT
 section("3", "SANITY FIGURES AND THE (I, kᵢ) INTERPOLANT — fig001–fig003")
##################################################################################################

# -----------------------------------------------------------------------------
# fig001 — CQD family (one curve per kᵢ), QM curve and combined experiment
# -----------------------------------------------------------------------------
color_list = palette(:darkrainbow, n_ki);
fig = plot(xlabel = "Current (A)", ylabel = L"$z_{\mathrm{max}}$ (mm)");
for (i, ki) in enumerate(cqd_meta[:ki])
    mask_cqd = log_mask(Icoils, z_up_ki[:, i])
    plot!(fig, Icoils[mask_cqd], z_up_ki[mask_cqd, i],
        label = L"$k_{i}=%$(round(ki, sigdigits=2))\times 10^{-6}$",
        line = (:solid, color_list[i]))
end
mask_qm = log_mask(Icoils, zmax_QM);
plot!(fig, Icoils[mask_qm], zmax_QM[mask_qm], label = "QM", line = (:dashdot, :black, 2));
plot!(fig, exp_avg[:i_smooth], exp_avg[:z_smooth],
    ribbon = exp_avg[:δz_smooth], color = :gold,
    label = "Combined experiments", line = (:solid, :gold, 3), fillalpha = 0.3);
plot!(fig,
    size = (1350, 850),
    xaxis = :log10, yaxis = :log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend = :outerright, legend_columns = 2, legendfontsize = 7, legendtitlefontsize = 8,
    left_margin = 6mm, bottom_margin = 5mm, foreground_color_legend = nothing)
annotate!(fig, 1e-2, 1,
    text(L"$n_{z} = %$(NZ_FIXED)$ | $\sigma_{\mathrm{conv}}=%$(Int(1e3*σW_FIXED))\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_FIXED)$", :black, 12));
display(fig)
savefig(fig, joinpath(OUTDIR, "fig001.$(FIG_EXT)"))
 
# -----------------------------------------------------------------------------
# fig002 — CQD z_max(kᵢ) slices vs QM at selected currents
# -----------------------------------------------------------------------------
fig = plot_cqd_vs_qm(z_up_ki, zmax_QM, Icoils, cqd_meta[:ki]);
display(fig)
savefig(fig, joinpath(OUTDIR, "fig002.$(FIG_EXT)"))
 
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
# fig003 — interpolated surface and contour (log10 z)
# -----------------------------------------------------------------------------
i_surface  = range(10e-3, 1.0; length = 101);
ki_surface = range(ki_bounds[1], ki_bounds[2]; length = 101);
Z = [ki_itp(x, y) for y in ki_surface, x in i_surface];
 
fit_surface = surface(log10.(i_surface), ki_surface, log10.(abs.(Z));
    title = "Fitting surface",
    xlabel = L"I_{c}", ylabel = L"$k_{i}\times 10^{-6}$", zlabel = L"$z\ (\mathrm{mm})$",
    legend = false, color = :viridis,
    xticks = (log10.([1e-3, 1e-2, 1e-1, 1.0]), [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    zticks = (log10.([1e-3, 1e-2, 1e-1, 1.0, 10.0]), [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    camera = (20, 25),
    xlims = log10.((8e-4, 2.05)), zlims = log10.((2e-4, 10.0)),
    gridalpha = 0.3,
)
 
logZ    = log10.(max.(abs.(Z), 1e-12));            # clamp away from log10(0)
lo, hi  = floor(minimum(logZ)), ceil(maximum(logZ));
decades = collect(lo:1:hi);
fit_contour = contourf(i_surface, ki_surface, logZ;
    levels = 101, title = "Fitting contour",
    xlabel = L"$I_{c}$ (A)", ylabel = L"$k_{i}\times 10^{-6}$",
    color = :viridis, linewidth = 0.2, linestyle = :dash,
    xaxis = :log10, xlims = (9e-3, 1.05),
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    clims = (lo, hi),
    colorbar_ticks = (decades, [L"10^{%$k}" for k in decades]),
    colorbar_title = L"$ z \ \mathrm{(mm)}$",
);
 
fit_figs = plot(fit_surface, fit_contour;
    layout = @layout([a; b]), size = (1800, 750), 
    left_margin=5mm, bottom_margin = 8mm, top_margin = 3mm);
display(fit_figs)
savefig(fit_figs, joinpath(OUTDIR, "fig003.$(FIG_EXT)"))



##################################################################################################
#  4) CONTINUOUS CURVE — fit subset, tail-convergence study
section("4", "CONTINUOUS CURVE — fit subset, tail-convergence study (fig008)")

##################################################################################################

# -----------------------------------------------------------------------------
# Continuous data matrix [I, δI, z, δz] above I_THRESHOLD, and the dense
# current scan used to draw model curves.
# -----------------------------------------------------------------------------
i_start = searchsortedfirst(exp_avg[:i_smooth], I_THRESHOLD);
I_scan  = logspace10(I_THRESHOLD, 1.00; n = 801);
data    = hcat(exp_avg[:i_smooth], δI_FRAC .* exp_avg[:i_smooth],
               exp_avg[:z_smooth], exp_avg[:δz_smooth])[i_start:end, :];
N_data  = size(data, 1);

pretty_table(data;
    alignment     = :c,
    title         = "EXPERIMENTAL DATA (continuous)",
    column_labels = ["Ic (A)", "δIc (A)", "z (mm)", "δz (mm)"],
    formatters    = [fmt__printf("%1.4f", 1:3), fmt__printf("%1.3f", [4])],
    style         = TextTableStyle(first_line_column_label = crayon"yellow bold",
                                   table_border = crayon"blue bold",
                                   column_label = crayon"yellow bold",
                                   title = crayon"bold red"),
    table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths = true)
 
# -----------------------------------------------------------------------------
# Rows entering the kᵢ loss (FIT_KI_MODE):
#   :full      whole post-threshold range
#   :low       first N_FRONT points   (small-deflection regime)
#   :high      last  N_BACK  points   (asymptotic regime)
#   :low_high  both windows, mid-current region excluded
# The calibration tail is chosen separately (N_TAIL) and may overlap.
# -----------------------------------------------------------------------------
# kᵢ-loss subset on the continuous curve
const FIT_KI_MODE = :full   # :full | :low | :high | :low_high
const N_FRONT     = 30      # points in the low-current window
const N_BACK      = 200     # points in the high-current window

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
subset_desc = FIT_KI_MODE === :full     ? "the whole post-threshold curve, " * window(1:N_data) :
              FIT_KI_MODE === :low      ? "the low-current window only, "    * window(low_range) :
              FIT_KI_MODE === :high     ? "the high-current window only, "   * window(high_range) :
                                          "the low- and high-current windows, " *
                                          window(low_range) * " and " * window(high_range) *
                                          ", mid-current region excluded";

@info cstr("kᵢ loss evaluated on ", :cyan) * cstr(String(FIT_KI_MODE), :yellow, :bold) *
      cstr(": " * subset_desc, :cyan) *
      cstr("  [$(size(data_fitting, 1)) of $(N_data) points]", :cyan)


@info "kᵢ-loss subset" mode = FIT_KI_MODE n_points = size(data_fitting, 1) I_range = extrema(data_fitting[:, 1])
 
# -----------------------------------------------------------------------------
# Tail-convergence study
#
# For each tail length n_tail in NTAIL_LIST (last n_tail rows of `data`):
#   • CQD : (kᵢ, s_CQD) jointly with `fit_cqd_ki_scale`  (kᵢ loss on data_fitting)
#   • QM  :  s_QM        with `fit_qm_scale`
# recorded vs I_min, the lowest current inside the tail. The plateau is the set
# of tails confined to I_min ≥ I_PLATEAU (tails that do not reach into the
# kᵢ-sensitive low-current region); the spread of each quantity over the
# plateau is quoted as a tail-choice systematic σ_sys.
# -----------------------------------------------------------------------------
const N_TAIL          = 400      # tail length (points) for the final fits
const N_TAIL_MAX      = 1000     # longest tail in the convergence scan
const NTAIL_LIST      = 1:50:N_TAIL_MAX
@assert N_TAIL_MAX ≤ N_data "N_TAIL_MAX = $N_TAIL_MAX exceeds the $N_data rows above I_THRESHOLD"

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

scan_cqd = DataFrame(n_tail = Int[], I_min = Float64[], ki = Float64[],
                     scale = Float64[], tail_rmse = Float64[], r2 = Float64[])
scan_qm  = DataFrame(n_tail = Int[], I_min = Float64[],
                     scale = Float64[], tail_rmse = Float64[], r2 = Float64[])
for nt in NTAIL_LIST
    fc = fit_cqd_ki_scale(data, data_fitting, cqd_meta[:ki], (KI_START, KI_STOP);
                          n_tail = nt, scale_mode = TAIL_SCALE_MODE)
    fq = fit_qm_scale(data, zqm; n_tail = nt, scale_mode = TAIL_SCALE_MODE)
    push!(scan_cqd, (nt, fc.tail_range[1], fc.ki, fc.scale, fc.tail_rmse, fc.r2_coeff))
    push!(scan_qm,  (nt, fq.tail_range[1],        fq.scale, fq.tail_rmse, fq.r2_coeff))
end
CSV.write(joinpath(OUTDIR, "tail_convergence_cqd.csv"), scan_cqd)
CSV.write(joinpath(OUTDIR, "tail_convergence_qm.csv"),  scan_qm)

# plateau statistics (centre: median for kᵢ and R², mean for s; spread: std)
plateau_cqd = scan_cqd[scan_cqd.I_min .>= I_PLATEAU, :]
plateau_qm  = scan_qm[ scan_qm.I_min  .>= I_PLATEAU, :]
@assert nrow(plateau_cqd) ≥ 3 "Fewer than 3 scan points have I_min ≥ $I_PLATEAU A; lower I_PLATEAU or shorten the tails"
 
ki_ref, ki_sys  = median(plateau_cqd.ki),  std(plateau_cqd.ki)
s_cqd,  s_cqd_σ = mean(plateau_cqd.scale), std(plateau_cqd.scale)
s_qm,   s_qm_σ  = mean(plateau_qm.scale),  std(plateau_qm.scale)
r2_cqd, r2_qm   = median(plateau_cqd.r2),  median(plateau_qm.r2)
 
@info "Tail-choice systematics (plateau: I_min ≥ $(I_PLATEAU) A, n = $(nrow(plateau_cqd)), scale_mode = $(TAIL_SCALE_MODE))" #=
    =# ki_CQD = (ki_ref, ki_sys) s_CQD = (s_cqd, s_cqd_σ) s_QM = (s_qm, s_qm_σ) #=
    =# R2_CQD = r2_cqd R2_QM = r2_qm ki_scale_cov = cov(plateau_cqd.ki, plateau_cqd.scale)
 
# fig008 — kᵢ (CQD) | s (CQD & QM) | R² (CQD & QM), shared x
xlab = L"Lowest current in the calibration tail $I_{\min}$ (A)";
 
p_ki = plot(scan_cqd.I_min, scan_cqd.ki;
    marker = (:circle, 3, :white, stroke(0.8, :blue)), line = (:solid, :blue, 1.5),
    ylabel = L"$k_{i}\ \left(\times 10^{-6}\right)$",
    label  = L"CQD: $k_{i} \pm \sigma_{\mathrm{sys}} = %$(fmt(ki_ref,4)) \pm %$(fmt(ki_sys,1))$",
    legend = :bottomleft)
hspan!(p_ki, [ki_ref - ki_sys, ki_ref + ki_sys]; fillalpha = 0.15, color = :blue, linealpha = 0, label = false)
hline!(p_ki, [ki_ref]; line = (:dash, :blue, 1), label = false)
 
p_s = plot(scan_cqd.I_min, scan_cqd.scale;
    marker = (:circle, 3, :white, stroke(0.8, :red)), line = (:solid, :red, 1.5),
    ylabel = L"scale $s$ (model$/s \approx$ data)",
    label  = L"CQD: $\bar{s} \pm \sigma_{\mathrm{sys}} = %$(fmt(s_cqd,4)) \pm %$(fmt(s_cqd_σ,1))$",
    legend = :left)
hspan!(p_s, [s_cqd - s_cqd_σ, s_cqd + s_cqd_σ]; fillalpha = 0.15, color = :red, linealpha = 0, label = false)
hline!(p_s, [s_cqd]; line = (:dash, :red, 1), label = false)
plot!(p_s, scan_qm.I_min, scan_qm.scale;
    marker = (:diamond, 3, :white, stroke(0.8, :purple)), line = (:dashdot, :purple, 1.5),
    label  = L"QM: $\bar{s} \pm \sigma_{\mathrm{sys}} = %$(fmt(s_qm,4)) \pm %$(fmt(s_qm_σ,1))$")
hspan!(p_s, [s_qm - s_qm_σ, s_qm + s_qm_σ]; fillalpha = 0.15, color = :purple, linealpha = 0, label = false)
hline!(p_s, [s_qm]; line = (:dash, :purple, 1), label = false)
 
p_r2 = plot(scan_cqd.I_min, scan_cqd.r2;
    marker = (:circle, 3, :white, stroke(0.8, :red)), line = (:solid, :red, 1.5),
    ylabel = L"$R^{2}$ (full data, calibrated model)", xlabel = xlab,
    label  = L"CQD: $\langle R^{2}\rangle = %$(fmt(r2_cqd,5))$")
plot!(p_r2, scan_qm.I_min, scan_qm.r2;
    marker = (:diamond, 3, :white, stroke(0.8, :purple)), line = (:dashdot, :purple, 1.5),
    label  = L"QM: $\langle R^{2}\rangle = %$(fmt(r2_qm,5))$",
    legend = :bottomleft)
 
fig = plot(p_ki, p_s, p_r2;
    layout = (3, 1), link = :x,
    size = (900, 950), left_margin = 5mm, bottom_margin = 3mm,
    legendfontsize = 9,
    plot_title = L"Convergence of $k_{i}$ and $s$ vs calibration-tail length ($%$(TAIL_SCALE_MODE)$ scale)",
)
display(fig)
savefig(fig, joinpath(OUTDIR, "fig008_tail_convergence.$(FIG_EXT)"))



##################################################################################################
#  5) FINAL FITS WITH UNCERTAINTIES (tail = last N_TAIL points)
section("5", "FINAL FITS WITH UNCERTAINTIES — tail = last N_TAIL points")

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
             n_tail = N_TAIL, bounds = ki_bounds, scale_mode = TAIL_SCALE_MODE, use_Zse = false);
@info "CQD: kᵢ + scale (with errors)" ki = (fit_cs.ki, fit_cs.ki_err) ci_profile = fit_cs.ci_profile #=
    =# scale = (fit_cs.scale, fit_cs.scale_err) scale_ci_t = fit_cs.scale_ci_t #=
    =# corr_ki_scale = fit_cs.corr_ki_scale n_overlap = fit_cs.n_overlap converged = fit_cs.converged
 
fit_qs = fit_qm_scale_with_error(zqm, data; n_tail = N_TAIL, scale_mode = TAIL_SCALE_MODE);
@info "QM: scale (with error)" scale = (fit_qs.scale, fit_qs.scale_err) scale_ci_t = fit_qs.scale_ci_t tail_rmse_mm = fit_qs.tail_rmse
 
# calibrated model curves used by every figure below
m_qm_scan  = zqm.(I_scan)                    ./ fit_qs.scale;
m_cqd_scan = ki_itp.(I_scan, Ref(fit_cs.ki)) ./ fit_cs.scale;
 
cfg_legend = L"$n_{z} = %$(NZ_FIXED)$ | $\sigma_{\mathrm{conv}}=%$(1e3*σW_FIXED)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_FIXED)$";
tail_legend = L"tail: $I \geq %$(round(fit_cs.tail_range[1], digits=3))$ A ($n = %$(N_TAIL)$, %$(TAIL_SCALE_MODE))";
log_ticks   = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]);
 


##################################################################################################
#  6) DIAGNOSTICS ON THE CONTINUOUS CURVE
section("6", "DIAGNOSTICS ON THE CONTINUOUS CURVE — fig004, fig005, fig006")
##################################################################################################
 
# -----------------------------------------------------------------------------
# fig004 — subsampled data with ribbon + calibrated QM + calibrated CQD
# -----------------------------------------------------------------------------
mask_qm_scan = log_mask(I_scan, m_qm_scan);
fig = plot(I_scan[mask_qm_scan], m_qm_scan[mask_qm_scan];
    label = L"QM ($s = %$(fmt(fit_qs.scale, 4))$)", line = (:solid, :red, 1.75))
 
I_exp, z_exp, dz_exp = data[1:2:end, 1], data[1:2:end, 3], data[1:2:end, 4];
m_exp = log_mask(I_exp, z_exp) .& isfinite.(dz_exp) .& (dz_exp .>= 0);
plot!(fig, I_exp[m_exp], z_exp[m_exp];
    ribbon = dz_exp[m_exp], color = :gray35,
    marker = (:circle, :gray35, 1), markerstrokecolor = :gray35, markerstrokewidth = 1,
    label = "Combined data")
 
mask_cqd_scan = log_mask(I_scan, m_cqd_scan);
plot!(fig, I_scan[mask_cqd_scan], m_cqd_scan[mask_cqd_scan];
    label = L"CQD: $k_{i}= \left( %$(fmt(fit_cs.ki, 3)) \pm %$(fmt(fit_cs.ki_err, 1)) \right) \times 10^{-6}$, $s = %$(fmt(fit_cs.scale, 4))$",
    line = (:solid, :blue, 2), marker = (:xcross, :blue, 0.2), markerstrokewidth = 1)
 
plot!(fig;
    title = "Calibrated models vs combined data",
    xlabel = "Current (A)", ylabel = L"$z_{\mathrm{max}}$ (mm)",
    xaxis = :log10, yaxis = :log10, xticks = log_ticks, yticks = log_ticks,
    labelfontsize = 14, tickfontsize = 12,
    xlims = (0.010, 1.05), size = (900, 800),
    legendtitle = cfg_legend, legendfontsize = 12, left_margin = 3mm)
display(fig)
savefig(fig, joinpath(OUTDIR, "fig004.$(FIG_EXT)"))
 
# -----------------------------------------------------------------------------
# fig005 — relative error (%) and pulls of both calibrated models
#
# Both models are divided by their own tail-fitted factor; the data are
# untouched. Two complementary views of (model − exp):
#   • relative error (model − exp)/exp   → SIZE of the discrepancy
#   • pull (model − exp)/σ_exp           → SIGNIFICANCE in units of δz
# NOTE: `data` is the smoothed combined curve, so neighbouring points are
# correlated; the pull pattern is meaningful, the χ²_red only indicative
# (the scattered-point table in section 7 gives a proper χ²).
# -----------------------------------------------------------------------------
I_d, z_d, σ_d = data[:, 1], data[:, 3], data[:, 4];
m_qm_d  = zqm.(I_d)                    ./ fit_qs.scale;
m_cqd_d = ki_itp.(I_d, Ref(fit_cs.ki)) ./ fit_cs.scale;
 
lbl_qm  = L"QM ($s = %$(fmt(fit_qs.scale, 4))$)";
lbl_cqd = L"CQD ($k_{i}=%$(fmt(fit_cs.ki, 4)) \times10^{-6}$, $s = %$(fmt(fit_cs.scale, 4))$)";
axis_kw = (
    xaxis = :log10,
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims = (0.010, 1.05),
    labelfontsize = 14, tickfontsize = 12,
    legend = :outerright, legendfontsize = 11, legendtitle = cfg_legend,
    left_margin = 3mm, bottom_margin = 3mm,
)
 
println("Model comparison — log-space metrics (calibrated models vs raw data)")
compare_datasets(I_d, z_d, m_cqd_d, m_qm_d; plot_errors = true);
 
re_qm  = 100 .* relerr(m_qm_d,  z_d);
re_cqd = 100 .* relerr(m_cqd_d, z_d);
fig_rel = plot(; title = "Relative error of the calibrated models", titlefontsize = 14,
    xlabel = "Current (A)", ylabel = L"(\mathrm{model}-\mathrm{exp})/\mathrm{exp}\ (\%)", axis_kw...)
hline!(fig_rel, [0.0]; line = (:dash, :black, 1), label = false)
plot!(fig_rel, I_d, re_qm;  label = lbl_qm,  line = (:solid, :red,  2))
plot!(fig_rel, I_d, re_cqd; label = lbl_cqd, line = (:solid, :blue, 2))
 
pull_qm  = pull(m_qm_d,  z_d, σ_d);
pull_cqd = pull(m_cqd_d, z_d, σ_d);
pull_summary(p) = (mean = mean(p), std = std(p), chi2_red = mean(abs2, p),
                   frac_within_1σ = mean(abs.(p) .<= 1), frac_within_2σ = mean(abs.(p) .<= 2))
@info "Pulls — QM  (continuous curve)" pull_summary(pull_qm)...
@info "Pulls — CQD (continuous curve)" pull_summary(pull_cqd)...
 
fig_pull = plot(; title = "Normalised residuals (pulls)", titlefontsize = 14,
    xlabel = "Current (A)", ylabel = L"(\mathrm{model}-\mathrm{exp})/\sigma_{\mathrm{exp}}", axis_kw...)
hspan!(fig_pull, [-2, 2]; fillalpha = 0.08, color = :gray, linealpha = 0, label = L"\pm 2\sigma")
hspan!(fig_pull, [-1, 1]; fillalpha = 0.15, color = :gray, linealpha = 0, label = L"\pm 1\sigma")
hline!(fig_pull, [0.0]; line = (:dash, :black, 1), label = false)
plot!(fig_pull, I_d, pull_qm;
    label = L"QM ($\chi^{2}_{\mathrm{red}} = %$(fmt(mean(abs2, pull_qm), 3))$)", line = (:solid, :red, 2))
plot!(fig_pull, I_d, pull_cqd;
    label = L"CQD ($\chi^{2}_{\mathrm{red}} = %$(fmt(mean(abs2, pull_cqd), 3))$)", line = (:solid, :blue, 2))
 
fig = plot(fig_rel, fig_pull; layout = (2, 1), link = :x, size = (1100, 750))
display(fig)
savefig(fig, joinpath(OUTDIR, "fig005.$(FIG_EXT)"))
 
CSV.write(joinpath(OUTDIR, "model_residuals_scaled.csv"), DataFrame(
    Ic = I_d, z_exp = z_d, dz_exp = σ_d, z_QM = m_qm_d, z_CQD = m_cqd_d,
    relQM = re_qm ./ 100, relCQD = re_cqd ./ 100, pullQM = pull_qm, pullCQD = pull_cqd))
 
# -----------------------------------------------------------------------------
# fig006 — publication-style overlay on the continuous curve
# -----------------------------------------------------------------------------
fig = plot(title = L"Peak position ($F=1$)")
plot!(fig, data[:, 1], data[:, 3];
    ribbon = data[:, 4], label = "Experimental data",
    line = (:dash, :darkgreen, 3), fillcolor = :darkgreen, fillalpha = 0.35)
plot!(fig, I_scan, m_qm_scan;
    label = L"Quantum mechanics: $s = %$(fmt(fit_qs.scale, 4)) \pm %$(fmt(fit_qs.scale_err, 1))$",
    line = (:solid, :red, 1.75))
plot!(fig, I_scan, m_cqd_scan;
    label = L"CoQuantum dynamics: $k_{i} = \left( %$(fmt(fit_cs.ki, 4)) \pm %$(fmt(fit_cs.ki_err, 1)) \right) \times 10^{-6}$, $s = %$(fmt(fit_cs.scale, 4)) \pm %$(fmt(fit_cs.scale_err, 1))$",
    line = (:dot, :blue, 2), markerstrokewidth = 1)
plot!(fig;
    xlabel = "Coil Current (A)", ylabel = L"$z_{\mathrm{max}}$ (mm)",
    xaxis = :log10, yaxis = :log10, xticks = log_ticks, yticks = log_ticks,
    labelfontsize = 14, tickfontsize = 12, size = (900, 800),
    legend = :topleft, legendtitle = tail_legend, legendtitlefontsize = 10, legendfontsize = 11,
    left_margin = 3mm)
display(fig)
savefig(fig, joinpath(OUTDIR, "fig006.$(FIG_EXT)"))



##################################################################################################
#  7) SCATTERED POINTS — publication figures and goodness of fit
section("7", "SCATTERED POINTS — publication figures, goodness of fit (fig007)")
##################################################################################################
# Raw scattered experimental points (spline at the grouped currents, propagated
# δz) vs the two calibrated models. The points are approximately independent,
# so χ², p-values and AIC/BIC are meaningful here.
i_sc    = searchsortedfirst(data_exp_scattered[:, 1], I_THRESHOLD);
data_sc = data_exp_scattered[i_sc:end, :];          # [I, δI, z, δz]
 
# columns of the scattered table, named once for the whole section
I_sc  = data_sc[:, 1];      # coil current (A)
δI_sc = data_sc[:, 2];      # current uncertainty (A), placeholder δI_FRAC·I
z_sc  = data_sc[:, 3];      # F=1 peak position (mm)
δz_sc = data_sc[:, 4];      # propagated position uncertainty (mm)

# field-gradient axis: G(I) and δG ≈ |G(I+δI) − G(I−δI)|/2
# (G is not linear through the origin, so applying G to δI directly would be wrong)
gradvsI(x) = TheoreticalSimulation.GvsI(x)
G_sc   = gradvsI.(I_sc);
δG_sc  = abs.(gradvsI.(I_sc .+ δI_sc) .- gradvsI.(I_sc .- δI_sc)) ./ 2;
G_scan = gradvsI.(I_scan);

# calibrated models evaluated at the scattered currents (for the GOF block)
m_qm_sc  = zqm.(I_sc)                    ./ fit_qs.scale;
m_cqd_sc = ki_itp.(I_sc, Ref(fit_cs.ki)) ./ fit_cs.scale;

# residual views on the scattered points (same definitions as section 6)
re_qm_sc   = relerr(m_qm_sc,  z_sc);          # (model − exp)/exp, dimensionless
re_cqd_sc  = relerr(m_cqd_sc, z_sc);
pull_qm_sc = pull(m_qm_sc,  z_sc, δz_sc);     # (model − exp)/δz
pull_cqd_sc = pull(m_cqd_sc, z_sc, δz_sc);

# -----------------------------------------------------------------------------
# single_SG_comparison — vs coil current
# -----------------------------------------------------------------------------
fig = plot()
plot!(fig, I_sc, z_sc;
    xerr = δI_sc, yerr = δz_sc,
    label = "Experimental data", seriestype = :scatter,
    marker = (:circle, 4, :white, stroke(0.5, :black)))
plot!(fig, I_scan, m_qm_scan;  label = "Existing models", line = (:dash, :blue, 1.75))
plot!(fig, I_scan, m_cqd_scan;
    label = L"Coquantum dynamics: $k_{i} \approx %$(fmt(fit_cs.ki, 2)) \times 10^{-6}$",
    line = (:solid, :red, 2), markerstrokewidth = 1)
plot!(fig;
    xlabel = "Coil Current (A)", ylabel = L"$F=1$ peak position (mm)",
    xaxis = :log10, yaxis = :log10, xticks = log_ticks, yticks = log_ticks,
    labelfontsize = 16, tickfontsize = 14, size = (900, 800),
    legendfontsize = 12, left_margin = 3mm)
display(fig)
savefig(fig, joinpath(OUTDIR, "single_SG_comparison.png"))
savefig(fig, joinpath(OUTDIR, "single_SG_comparison.svg"))
 
# -----------------------------------------------------------------------------
# single_SG_comparison_vsg — vs magnetic-field gradient G(I)
# -----------------------------------------------------------------------------
fig = plot()
plot!(fig, G_sc, z_sc;
    xerr = δG_sc, yerr = δz_sc,
    label = "Experimental data", seriestype = :scatter,
    marker = (:circle, 4, :white, stroke(0.5, :black)))
plot!(fig, G_scan, m_qm_scan;  label = "Existing models", line = (:dash, :blue, 1.75))
plot!(fig, G_scan, m_cqd_scan;
    label = L"Coquantum dynamics: $k_{i} \approx %$(fmt(fit_cs.ki, 2)) \times 10^{-6}$",
    line = (:solid, :red, 2), markerstrokewidth = 1)
plot!(fig;
    xlabel = "Magnetic field gradient (T/m)", ylabel = L"$F=1$ peak position (mm)",
    xaxis = :log10, yaxis = :log10, yticks = log_ticks, xlims = (5, 400),
    labelfontsize = 16, tickfontsize = 14, size = (900, 800),
    legend = :topleft, legendfontsize = 12, left_margin = 3mm)
display(fig)
savefig(fig, joinpath(OUTDIR, "single_SG_comparison_vsg.png"))
savefig(fig, joinpath(OUTDIR, "single_SG_comparison_vsg.svg"))
 
# -----------------------------------------------------------------------------
# Tables: raw scattered experiment (both axes) and calibrated model curves
# -----------------------------------------------------------------------------
df_exp = DataFrame(Ic = I_sc, sIc = δI_sc, G = G_sc, sG = δG_sc, zmax = z_sc, szmax = δz_sc)
df_sim = DataFrame(Ic = I_scan, G = G_scan, QM = m_qm_scan, CQD = m_cqd_scan)
CSV.write(joinpath(OUTDIR, "data_exp.csv"), df_exp)
CSV.write(joinpath(OUTDIR, "data_sim.csv"), df_sim)
 
# -----------------------------------------------------------------------------
# Goodness of fit: calibrated CQD (k = 2: kᵢ, s) vs calibrated QM (k = 1: s)
# -----------------------------------------------------------------------------
stats_CQD = goodness_of_fit(I_sc, z_sc, m_cqd_sc; σ = δz_sc, k = 2)   # kᵢ + s
stats_QM  = goodness_of_fit(I_sc, z_sc, m_qm_sc;  σ = δz_sc, k = 1)   # s only
@info "Goodness of fit on $(length(z_sc)) scattered points" ΔAIC_CQD_minus_QM = stats_CQD.AIC - stats_QM.AIC ΔBIC_CQD_minus_QM = stats_CQD.BIC - stats_QM.BIC
 
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
    title         = "Goodness of fit — calibrated models vs raw scattered data (k: CQD = 2, QM = 1)",
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
 
# fig007 — diagnostic panels
p1, p2, p3, p4 = make_diagnostic_plots(I_sc, z_sc, m_cqd_sc, m_qm_sc, stats_CQD, stats_QM; σ = δz_sc)
fig = plot(p1, p2, p3, p4; layout = (2, 2), size = (1200, 1000), left_margin = 4mm, bottom_margin = 3mm)
display(fig)
savefig(fig, joinpath(OUTDIR, "fig007.$(FIG_EXT)"))
 

##################################################################################################
#  SUMMARY
#  Final numbers in one place: value ± statistical error (t-interval half-width
#  from the `_with_error` fitters) ± tail-choice systematic (plateau spread of
#  the convergence scan). Text in green, values in bold magenta.
section("S", "SUMMARY — results and run archive")
##################################################################################################
res(v, d = 4) = cstr(fmt(v, d), :magenta, :bold)
pm(v, e_stat, e_sys; d = 4) = res(v, d) * cstr(" ± ", :green) * res(e_stat, 2) *
                              cstr(" (stat) ± ", :green) * res(e_sys, 2) * cstr(" (sys)", :green)
row(name, val) = "\n  " * cstr(rpad(name, 12), :green) * val      # "  name……  value"

@info cstr("RESULT — CQD", :green, :bold) *
      row("kᵢ",         pm(fit_cs.ki, fit_cs.ki_err, ki_sys) * cstr(" × 10⁻⁶", :green)) *
      row("scale",      pm(fit_cs.scale, fit_cs.scale_err, s_cqd_σ)) *
      row("corr(kᵢ,s)", res(fit_cs.corr_ki_scale, 2)) *
      row("tail",       cstr("I ≥ $(fmt(fit_cs.tail_range[1], 3)) A, n = $(N_TAIL), $(TAIL_SCALE_MODE)", :green))

@info cstr("RESULT — QM", :green, :bold) *
      row("scale",      pm(fit_qs.scale, fit_qs.scale_err, s_qm_σ)) *
      row("tail",       cstr("I ≥ $(fmt(fit_qs.tail_range[1], 3)) A, n = $(N_TAIL), $(TAIL_SCALE_MODE)", :green))

@info cstr("Done", :green, :bold) *
      row("elapsed",    cstr(Dates.canonicalize(Dates.now() - T_START), :magenta, :bold)) *
      row("OUTDIR",     cstr(OUTDIR, :magenta))



# -----------------------------------------------------------------------------
# Single-file archive of the run (JLD2)
#
# One self-describing file with everything needed to re-quote the numbers or
# redraw the publication figures without re-running the fits. Only plain
# scalars, strings and Float64 vectors are stored (no DataFrames, no Optim
# objects), so the file is readable from any Julia version or from Python/h5py.
#
#   meta/                 provenance: script, run stamp, host, Julia version,
#                         experimental input file
#   settings/             analysis configuration (nz, σw, λ0, …) and fit
#                         choices (I threshold, δI fraction, kᵢ-loss mode,
#                         calibration tail, plateau cut)
#   fit/cqd/, fit/qm/     kᵢ (SI units) and calibration factors s with
#                         statistical (t-interval) and systematic (tail-choice)
#                         errors
#   experiment/           scattered points: I, G(I), z_F1 and uncertainties,
#                         plus both calibrated models at the same currents
#   model/                calibrated model curves on the dense current scan
#   statistical_analysis/ relative errors and pulls on the scattered points
#   gof/                  goodness-of-fit table (same rows as goodness_of_fit.csv)
#
# Within a group every vector has the length of its Current_A, so each group
# rebuilds as one table: e.g. DataFrame(f["experiment"]) after `f = load(path)`.
section("A", "RUN ARCHIVE — single JLD2 file: settings, fits, data, models, GOF")
# -----------------------------------------------------------------------------
archive_path = joinpath(OUTDIR, "SG_ki_scale_results.jld2")
jldopen(archive_path, "w") do f
    # --- provenance ---
    f["meta/script"]          = @__FILE__
    f["meta/run_stamp"]       = RUN_STAMP
    f["meta/host"]            = gethostname()
    f["meta/julia"]           = string(VERSION)
    f["meta/experiment_file"] = experiment_path

    # --- settings ---
    f["settings/nx"]            = NX_FIXED
    f["settings/nz"]            = NZ_FIXED
    f["settings/σw_mm"]         = σW_FIXED
    f["settings/λ0"]            = λ0_FIXED
    f["settings/λ_spline"]      = λ0_SPLINE
    f["settings/I_threshold_A"] = I_THRESHOLD
    f["settings/dI_frac"]       = δI_FRAC
    f["settings/n_tail"]        = N_TAIL
    f["settings/ki_mode"]       = String(FIT_KI_MODE)
    f["settings/I_plateau_A"]   = I_PLATEAU

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

    # --- experiment, scattered points (I ≥ I_THRESHOLD), both x-axes ---
    f["experiment/Current_A"]      = I_sc
    f["experiment/CurrentErr_A"]   = δI_sc
    f["experiment/Gradient_Tm"]    = G_sc
    f["experiment/GradientErr_Tm"] = δG_sc
    f["experiment/zF1_mm"]         = z_sc
    f["experiment/zF1Err_mm"]      = δz_sc
    # calibrated models at the same currents (inputs of the GOF table)
    f["experiment/CQD_up_mm"]      = m_cqd_sc
    f["experiment/QM_zF1_mm"]      = m_qm_sc

    # --- calibrated model curves (model/s) on the dense scan I_scan ---
    f["model/Current_A"]   = I_scan
    f["model/Gradient_Tm"] = G_scan
    f["model/CQD_up_mm"]   = m_cqd_scan
    f["model/QM_zF1_mm"]   = m_qm_scan

    # --- residual views on the scattered points (inputs of fig007) ---
    f["statistical_analysis/Current_A"]   = I_sc
    f["statistical_analysis/relErr_CQD"]  = re_cqd_sc         # already dimensionless
    f["statistical_analysis/relErr_QM"]   = re_qm_sc
    f["statistical_analysis/pull_CQD"]    = pull_cqd_sc
    f["statistical_analysis/pull_QM"]     = pull_qm_sc

    # --- goodness of fit on the scattered points (k: CQD = 2, QM = 1) ---
    f["gof/metric"]    = metrics
    f["gof/CQD"]       = gof_table[:, 1]
    f["gof/QM"]        = gof_table[:, 2]
end
@info cstr("Run archived to ", :green) * cstr(archive_path, :magenta)

archive_keys = list_jld2_entries(archive_path)     # names-only reference of the archive layout





