# Simulation of atom trajectories in the Stern–Gerlach experiment
# Interpolation of the grid for fitting the induction term
# Kelvin Titimbo
# California Institute of Technology
# November 2025

using Plots; gr()
Plots.default(
    show=true, dpi=600, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
# Aesthetics and output formatting
using Colors, ColorSchemes
using LaTeXStrings, Printf, PrettyTables
# Time-stamping/logging
using Dates
const T_START = Dates.now() ; # Timestamp start for execution timing
# Numerical tools
using LinearAlgebra, DataStructures
using Dierckx, Optim
using DSP
using LambertW, PolyLog
using StatsBase
using Random, Statistics, NaNStatistics, Distributions, StaticArrays
using Alert
# Data manipulation
using OrderedCollections
using DelimitedFiles, CSV, DataFrames, JLD2
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(2)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Set the working directory to the current location
cd(@__DIR__) ;
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
const OUTDIR    = joinpath(@__DIR__, "data_studies", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
include("./Modules/TheoreticalSimulation.jl");
include("./Modules/DataReading.jl");
include("./Modules/MyExperimentalAnalysis.jl");

logspace10(lo, hi; n=50) = 10.0 .^ range(log10(lo), log10(hi); length=n)

function log_mask(x, y)
    (x .> 0) .& (y .> 0) .& isfinite.(x) .& isfinite.(y)
end

# Relative error helper (dimensionless)
relerr(model, exp) = (model .- exp) ./ exp

function fit_ki(data_org, selected_points, ki_list, ki_range)
    """
        fit_ki(data_org, selected_points, ki_list, ki_range)

    Fit the induction coefficient `kᵢ` by minimizing a mean-squared error in **log10 space**
    between the interpolated prediction `ki_itp(x, kᵢ)` and a selected subset of data points.
    The fit is therefore sensitive to *relative (fractional) deviations* across orders of
    magnitude.

    Although the optimization is performed in log space, the reported error is evaluated
    **in linear space** at the best-fit value of `kᵢ`, and is returned as a root-mean-square
    error (RMSE) in the same physical units as the dependent variable (e.g. millimeters).

    # Arguments
    - `data_org` :: 2-column array `(x, y)`  
    Full data set used to compute the coefficient of determination R² in linear space.

    - `selected_points` :: 2-column array `(x, y)`  
    Subset of points used for the fit. All `y` values must be strictly positive
    (required for log10 evaluation).

    - `ki_list` :: AbstractVector  
    Vector of candidate `kᵢ` values defining the search interval.

    - `ki_range` :: Tuple{Int,Int}  
    Index range `(ki_start, ki_stop)` selecting the portion of `ki_list` used in
    the bounded 1D optimization.

    # Returns
    NamedTuple with fields:
    - `ki`        : Best-fit value of the induction coefficient `kᵢ`
    - `ki_err`    : Root-mean-square error (RMSE) in **linear space**, evaluated on
                    `selected_points` at the fitted `kᵢ`
    - `r2_coeff`  : Coefficient of determination R² computed in linear space on `data_org`

    # Notes
    - The fit minimizes an error in log10 space, but no uncertainty on `kᵢ` is estimated.
    The returned `ki_err` is **not** an error bar on `kᵢ`, but a goodness-of-fit measure
    in real space.
    - If the dependent variable spans several orders of magnitude, the log-space fit
    prevents large-amplitude points from dominating the optimization, while the linear
    RMSE provides a physically interpretable error metric.
    """
    ki_start, ki_stop = ki_range

    Ic_fit = selected_points[:, 1]
    z_fit  = selected_points[:, 3]

    # # --- log-space loss (used ONLY for fitting) ---
    loss_log(ki) = begin
        z_pred = ki_itp.(Ic_fit, Ref(ki))
        mean(abs2, log10.(z_pred) .- log10.(z_fit))
    end

    # 1D optimization over ki
    fit_param = optimize(loss_log,
                         ki_list[ki_start], ki_list[ki_stop],
                         Brent())

    k_fit = Optim.minimizer(fit_param)

    # --- linear-space error (reported) ---
    z_pred_sel = ki_itp.(Ic_fit, Ref(k_fit))
    z_obs_sel  = z_fit

    mse_lin  = mean(abs2, z_pred_sel .- z_obs_sel)
    rmse_lin = sqrt(mse_lin)   # same units as z (e.g. mm)

    # predictions for the full data set
    Ic = data_org[:, 1]
    pred = ki_itp.(Ic, Ref(k_fit))
    y    = data_org[:, 3]
    coef_r2 = 1 - sum(abs2, pred .- y) / sum(abs2, y .- mean(y))

    return (ki = k_fit, ki_err = rmse_lin, r2_coeff = coef_r2)
end

function fit_ki_with_error(itp, data;
                           bounds::Tuple{<:Real,<:Real},
                           conf::Real=0.95,
                           weights::Union{Nothing,AbstractVector}=nothing,
                           h::Union{Nothing,Real}=nothing)
    """
        fit_ki_with_error(itp, data; bounds, conf=0.95, weights=nothing, h=nothing)

    Fit a single parameter `ki` by minimizing the (optionally weighted) mean squared error
    in **log10 space** between the model prediction and measurements:

        r_i(ki) = log10(itp(I_i, ki)) - log10(Z_i)

    Then estimate an approximate standard error for `ki` using a local linearization
    of the residuals around the optimum `k̂`, via a finite-difference Jacobian dr/dki.

    Arguments
    - `itp`    : callable model/interpolant such that `itp(I, ki) -> z_pred` (broadcastable)
    - `data`   : array/table with at least:
                * column 1: `I`  (currents)
                * column 3: `Z`  (measured peak positions; must be > 0 for log10)
                (Note: this matches your code using `data[:,3]`.)
    - `bounds` : `(ki_min, ki_max)` search interval for bounded 1D optimization (required)

    Keywords
    - `conf`     : confidence level for the interval (default 0.95)
    - `weights`  : optional per-point weights applied in log-space loss
                  (if provided, must match length of data rows)
    - `h`        : finite-difference step for ∂r/∂ki; if omitted, chosen automatically

    Returns
    NamedTuple with fields:
    - `k_hat`     : best-fit `ki`
    - `k_err`     : half-width of the (approx.) `conf` confidence interval (t * SE)
    - `se`        : standard error estimate for `k_hat`
    - `ci`        : confidence interval `(low, high)` at level `conf`
    - `rss`       : weighted residual sum of squares in log-space at optimum
    - `sigma2`    : estimated residual variance (rss / dof)
    - `dof`       : degrees of freedom (n_used - 1)
    - `n_used`    : number of points used after filtering invalid values
    - `converged` : optimizer convergence flag
    - `result`    : Optim.jl result object
    """

    # -----------------------------
    # 1) Unpack the relevant columns
    # -----------------------------
    I = collect(Float64, data[:, 1])   # currents
    Z = collect(Float64, data[:, 3])   # measured peaks (note: col 3 by convention here)

    # -----------------------------------------------------------
    # 2) Filter invalid points (log10 requires Z > 0 and finite)
    # -----------------------------------------------------------
    mask0 = isfinite.(I) .& isfinite.(Z) .& (Z .> 0)
    I, Z = I[mask0], Z[mask0]

    # -----------------------------------------------------------
    # 3) Prepare weights (or default to uniform)
    #    Weights are applied to squared log-residuals.
    # -----------------------------------------------------------
    w = weights === nothing ? ones(length(I)) : collect(weights)[mask0]
    @assert length(w) == length(I) "weights length must match number of valid points"

    # -----------------------------------------------------------
    # 4) Define the objective function:
    #    weighted mean of squared residuals in log10 space.
    # -----------------------------------------------------------
    function loss(ki)
        zpred = itp.(I, Ref(ki))
        r = log10.(zpred) .- log10.(Z)

        # If interpolation produces NaNs/Infs at some points, drop them
        m = isfinite.(r)
        r = r[m]; ww = w[m]

        return mean(ww .* (r .^ 2))
        # Note: this is a weighted *mean* (not normalized by sum(w)).
        # If you want normalized weighted MSE: sum(ww .* r.^2) / sum(ww).
    end

    # ----------------------------------------
    # 5) Solve bounded 1D minimization for ki
    # ----------------------------------------
    ki_min, ki_max = float(bounds[1]), float(bounds[2])
    res = optimize(loss, ki_min, ki_max, Brent())
    k̂  = Optim.minimizer(res)

    # ---------------------------------------------------
    # 6) Compute residuals at the optimum and re-filter
    #    (keeps I, Z, w aligned with usable residuals).
    # ---------------------------------------------------
    ẑ   = itp.(I, Ref(k̂))
    r    = log10.(ẑ) .- log10.(Z)
    mres = isfinite.(r)
    r, I, Z, w = r[mres], I[mres], Z[mres], w[mres]

    # For a single-parameter fit, degrees of freedom = n - 1
    n = length(r)
    p = 1
    @assert n > p "Not enough valid points to estimate uncertainty"

    # --------------------------------------------------------------------
    # 7) Estimate the Jacobian dr/dki using a central finite difference
    #    We choose a step size that is:
    #      - relative to |k̂| (≈ cbrt(eps)*|k̂|)
    #      - with an absolute minimum floor
    #      - reduced if k̂ is too close to the bounds
    # --------------------------------------------------------------------
    fd_step(k̂, lo, hi; rel=cbrt(eps(Float64)), absmin=1e-12) = begin
        h = max(absmin, rel * abs(k̂))
        if isfinite(lo) && isfinite(hi)
            room = min(k̂ - lo, hi - k̂)
            h = room > 0 ? min(h, 0.5 * room) : absmin
        end
        h
    end

    h₀ = isnothing(h) ? fd_step(k̂, ki_min, ki_max) : float(h)

    # Evaluate residuals at k̂ ± h₀
    z⁺ = itp.(I, Ref(k̂ + h₀))
    z⁻ = itp.(I, Ref(k̂ - h₀))
    r⁺ = log10.(z⁺) .- log10.(Z)
    r⁻ = log10.(z⁻) .- log10.(Z)

    # Keep only points where both sides are finite (safe central difference)
    mJ   = isfinite.(r⁺) .& isfinite.(r⁻)
    r, w = r[mJ], w[mJ]

    # Central difference derivative
    drdk = (r⁺[mJ] .- r⁻[mJ]) ./ (2h₀)

    n_used = length(r)
    @assert n_used > p "Not enough valid points after derivative filtering"

    # --------------------------------------------------------------------
    # 8) Standard error estimate from weighted least squares linearization:
    #    Var(k̂) ≈ σ² / (J'J), with J = dr/dk (scalar parameter).
    # --------------------------------------------------------------------
    RSS = sum(w .* (r .^ 2))         # weighted residual sum of squares
    dof = n_used - p                 # degrees of freedom
    σ²  = RSS / dof                  # residual variance estimate

    # J'J for scalar parameter with weights: sum( (sqrt(w_i)*J_i)^2 )
    SJJ = sum((sqrt.(w) .* drdk) .^ 2)

    # Standard error of k̂
    se  = sqrt(σ² / SJJ)

    # --------------------------------------------------------------------
    # 9) Confidence interval using Student-t critical value
    # --------------------------------------------------------------------
    tcrit = quantile(TDist(dof), 0.5 + conf/2)
    ci    = (k̂ - tcrit*se, k̂ + tcrit*se)

    return (
        k_hat     = k̂,
        k_err     = tcrit * se,
        se        = se,
        ci        = ci,
        rss       = RSS,
        sigma2    = σ²,
        dof       = dof,
        n_used    = n_used,
        converged = Optim.converged(res),
        result    = res
    )
end

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

function compare_datasets(x_ref::AbstractVector, # current 
                            A::AbstractVector,   # Experimental
                            B::AbstractVector,   # CQD
                            C::AbstractVector;   # QM
                            plot_errors=true)
    """
        compare_datasets(A, B; plot_errors=true)

    Extended comparison of datasets A and B for log-scale analysis.
    Returns:
    - NamedTuple of metrics
    - PrettyTables summary
    - Optional log-error plot

    Metrics:
    log_MSE, log_RMSE
    rel_mean, rel_median, rel_max
    MAPE, sMAPE
    max_log_error
    L2_norm, L2_log_norm
    chi2_log
    KS_distance
    """
    @assert length(x_ref) == length(A) == length(B) == length(C) "All vectors must have same length."
    @assert all(A .> 0) && all(B .> 0) && all(C .> 0) "All values must be > 0 for log comparison."

    # --- compute for both models ---
    R_B = compute_metrics(A, B);
    R_C = compute_metrics(A, C);

    # ------- Pretty Table ---------
        # PrettyTable
    header = ["Metric", "CQD vs Exp", "QM vs Exp"];
    data = [
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
        (data, i, j) -> begin
            # Only highlight numeric entries in columns 2 or 3
            if j == 2 || j == 3
                v1 = data[i, 2]
                v2 = data[i, 3]

                # Be safe: only operate if both are numbers
                if isa(v1, Number) && isa(v2, Number)
                    row_min = v1 <= v2 ? v1 : v2
                    return data[i, j] == row_min
                else
                    return false
                end
            else
                return false
            end
        end,
        crayon"fg:red bold"  # red, bold; you can add bg if you like
    );

    pretty_table(data, 
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

function keypath(branch::Symbol, ki::Float64, nz::Int, gw::Float64, λ0_raw::Float64)
    fmt(x) = @sprintf("%.12g", x)  # safer than %.6g to reduce collisions
    return "/" * String(branch) *
           "/ki=" * fmt(ki) *"e-6" *
           "/nz=" * string(nz) *
           "/gw=" * fmt(gw) *
           "/lam=" * fmt(λ0_raw)
end

# =============================================================================
# Simulated coil currents (in Amperes)
#
# These are the discrete current values at which both QM and CQD simulations
# were performed. The spacing is non-uniform.
# =============================================================================
Icoils = [0.00,
        0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
        0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
        0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
        0.100,0.150,0.200,0.250,0.300,0.350,0.400,0.450,0.500,0.550,
        0.600,0.650,0.700,0.750,0.800,0.850,0.900,0.950,1.00
];
nI = length(Icoils); # Number of simulated current points


# =============================================================================
# Quantum-mechanical (QM) simulation data
#
# The QM data is stored as a dictionary indexed by tuples:
#     (nz_bins, gaussian_width_mm, λ0_raw)
#
# Each entry contains the corresponding screen-profile analysis results
# for all currents in `Icoils`.
# =============================================================================
table_qm_path = joinpath(@__DIR__,
    "simulation_data",
    "qm_simulation_7M",
    "qm_7000000_screen_profiles_f1_table.jld2");
table_qm      = load(table_qm_path)["table"];
@info "QM data loaded"
qm_meta = let
    keys_qm = collect(keys(table_qm))  # make it an indexable Vector of tuples

    nz_qm = sort(unique(getindex.(keys_qm, 1)))
    gw_qm = sort(unique(getindex.(keys_qm, 2)))
    λ0_qm = sort(unique(getindex.(keys_qm, 3)))

    # --- pretty print aligned ---
    labels = ["nz_qm", "gw_qm", "λ0_qm"]
    w = maximum(length.(labels))

    # println(rpad("nz_qm", w), " = ", nz_qm);
    # println(rpad("gw_qm", w), " = ", gw_qm);
    # println(rpad("λ0_qm", w), " = ", λ0_qm);

    # --- return renamed container ---
    OrderedDict(
        :nz => nz_qm,
        :gw => gw_qm,
        :λ0 => λ0_qm,
        :λs => 0.001
    );
end

# =============================================================================
# CoQuantum Dynamics (CQD) simulation data
#
# The CQD results are stored in a JLD2 file indexed by parameter-dependent
# key paths. A dedicated "meta" group records all available values of:
#
#   - ki  : induction coefficients (dimensionless, scaled later as ×10⁻⁶)
#   - nz  : number of bins in z
#   - gw  : Gaussian smoothing width (mm)
#   - λ0  : raw spline smoothing parameter
#   - λs  : spline smoothing parameter used internally
# =============================================================================
table_cqd_path = joinpath(@__DIR__,
    "simulation_data",
    "cqd_simulation_6M",
    "cqd_6000000_up_profiles_bykey.jld2");
cqd_meta = jldopen(table_cqd_path, "r") do file
    meta = file["meta"]

    # mapping: original key (String) → new Symbol
    rename = OrderedDict(
        "induction_coeff"   => :ki,
        "nz_bins"           => :nz,
        "gaussian_width_mm" => :gw,
        "λ0_raw_list"       => :λ0,
        "λ0_spline"         => :λs,
    )

    # alignment width (use original names for printing)
    w = maximum(length.(keys(rename)))

    out = OrderedDict{Symbol,Any}()

    for (k_old, k_new) in rename
        val = round.(meta[k_old], digits=3)

        # println(rpad(k_old, w), " = ", val)

        out[k_new] = val
    end
    out;
end

# =============================================================================
# Experimental data (combined / averaged)
#
# This dataset contains a smoothed experimental peak position z(I) and its
# uncertainty. We build:
#   1) A weighted cubic spline z_spline(I) fitted to the smoothed data
#   2) A set of "grouped" current points xq with uncertainties δxq
#   3) Propagated z-uncertainty at xq:
#        δz_total^2 = ( (dz/dI)*δI )^2  +  (δz_interp)^2
#
# where:
#   - dz/dI is the spline derivative evaluated at xq
#   - δz_interp is the interpolated z-uncertainty at xq
# =============================================================================
exp_avg = load(joinpath(@__DIR__,"analysis_data","smoothing_binning","data_averaged_2.jld2"))["data"];
@info "Experimental data loaded"
# 1. Fit spline for the experiment data
data_experiment = Spline1D(
    exp_avg[:i_smooth], 
    exp_avg[:z_smooth], 
    k=3, 
    bc="extrapolate", 
    s=0.0, 
    w = 1 ./ exp_avg[:δz_smooth].^2
);
# xq and δxq from grouped data
xq  = exp_avg[:Ic_grouped][:,1];
δxq = exp_avg[:Ic_grouped][:,2];
# 2. Spline derivative at xq : 
# Evaluate derivative dz/dI at the grouped current points xq.
# This is needed to propagate δI into δz via slope*δI.
dyq = derivative(data_experiment, xq; nu=1);
# 3. Interpolate δy uncertainty to xq
err_spline = Spline1D(exp_avg[:i_smooth], exp_avg[:δz_smooth], k=3, bc="extrapolate");
δy_interp = err_spline.(xq);   # # σ_z at xq
# 4. Total propagated uncertainty σ_total = sqrt( (dz/dI * σ_I)^2 + σ_z(I)^2 )
δyq = sqrt.( (dyq .* δxq).^2 .+ δy_interp.^2 );
# Pack into a convenient table:
# columns = [I, δI, z_spline(I), σ_total]
data_exp_scattered = hcat(xq,δxq,data_experiment.(xq),δyq);
pretty_table(data_exp_scattered;
        alignment     = :c,
        title         = @sprintf("EXPERIMENTAL DATA (scattered)"),
        column_labels = ["Ic (A)","δIc (A)", "z (mm)", "δz (mm)"],
        formatters    = ([fmt__printf("%1.3f", [1]),fmt__printf("%1.4f", [2]),fmt__printf("%1.3f", 3:4)]),
        style         = TextTableStyle(
                        first_line_column_label = crayon"yellow bold",
                        table_border  = crayon"blue bold",
                        column_label  = crayon"yellow bold",
                        title = crayon"bold red"
                        ),
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
        equal_data_column_widths= true,)
# =============================================================================

# =============================================================================
# General analysis parameters
#
# The QM and CQD datasets may not share the exact same grid of analysis parameters.
# Here we compute the *intersection* (common values) for:
#   - nz : number of z-bins used in profile extraction
#   - gw : Gaussian smoothing width (mm)
#   - λ0 : raw smoothing parameter
#
# We then pick a single analysis configuration (nx_bins, nz_bins, gw, λ0, λs)
# and assert that it exists in BOTH QM and CQD metadata.
# =============================================================================

# ---- common parameter sets across QM and CQD ----
meta_nz = Int.(intersect(qm_meta[:nz],cqd_meta[:nz]));
meta_σw = intersect(qm_meta[:gw],cqd_meta[:gw]);
meta_λ0 = intersect(qm_meta[:λ0],cqd_meta[:λ0]);
@info "Common parameter grid" meta_nz=meta_nz meta_gw=meta_σw meta_λ0=meta_λ0
# number of CQD induction coefficients available
n_ki    = length(cqd_meta[:ki]);

# ---- chosen working point for this run ----
nx_bins , nz_bins = 128 , 2;
gaussian_width_mm = 0.200;
λ0_raw            = 0.01;
λ0_spline         = 0.001;
@info "Selected parameters" nx_bins=nx_bins nz_bins=nz_bins gw=gaussian_width_mm λ0_raw=λ0_raw λ0_spline=λ0_spline

# -----------------------------------------------------------------------------
# Sanity checks:
# Ensure the chosen parameters exist in the *common* QM ∩ CQD sets.
# -----------------------------------------------------------------------------
@assert nz_bins in meta_nz "nz_bins = $nz_bins not in common nz set: $meta_nz"
@assert gaussian_width_mm in meta_σw "gaussian_width_mm = $gaussian_width_mm not in common gw set: $meta_σw"
@assert λ0_raw in meta_λ0 "λ0_raw = $λ0_raw not in common λ0 set: $meta_λ0"

# =============================================================================
# Quantum-mechanical (QM) reference curve z_max(I)
#
# For the selected analysis parameters (nz_bins, gaussian_width_mm, λ0_raw),
# extract the QM-predicted maximum deflection z_max as a function of coil
# current I, and construct a smooth interpolant z_qm(I).
# =============================================================================
data_qm = table_qm[(nz_bins,gaussian_width_mm,λ0_raw)];
Ic_QM   = [data_qm[i][:Icoil] for i in eachindex(data_qm)];
zmax_QM = [data_qm[i][:z_max_smooth_spline_mm] for i in eachindex(data_qm)];
zqm = Spline1D(Ic_QM,zmax_QM,k=3);
table_qm = nothing
# =============================================================================
# Build CQD matrix z_max(I, kᵢ)
#
# Goal:
#   Construct a matrix `z_up_ki` of size (nI × n_ki), where:
#     - rows correspond to coil currents Icoils (index 1:nI)
#     - columns correspond to the induction coefficient values cqd_meta[:ki]
#
# Each entry is the CQD-predicted peak position:
#     z_up_ki[j, i] = z_max_smooth_spline_mm at current index j and ki index i
# =============================================================================
z_up_ki = Matrix{Float64}(undef, nI, n_ki);
for (i,ki) in enumerate(cqd_meta[:ki])
    # Progress print (ki is stored in "micro-units"; display it as ×10⁻⁶)
    println("\t($(@sprintf("%03d", i))/$(n_ki)) Running ki=$(@sprintf("%2.1e",1e-6*ki))")
    # Load the CQD profile data for this ki and analysis configuration.
    # The keypath encodes the branch (:up), ki, nz_bins, gaussian_width_mm, λ0_raw.
    data_up = jldopen(table_cqd_path, "r") do file
        file[keypath(:up,ki,nz_bins,gaussian_width_mm,λ0_raw)]
    end
    # Extract z_max (in mm) for each simulated current index l = 1:nI
    # and store as the i-th column of z_up_ki.
    z_up_ki[:,i] = [data_up[l][:z_max_smooth_spline_mm] for l in 1:nI]

end


# =============================================================================
# Visual sanity checks BEFORE building the (I, kᵢ) interpolation surface
#
# Goal:
#   Confirm that:
#   1) CQD z_max(I) curves vary smoothly with current and with kᵢ
#   2) CQD family brackets the experimental curve in the region of interest
#   3) QM reference curve is in the expected range (for comparison / scaling)
#
# Plot 1: z_max vs I (log-log), overlaying:
#   - CQD curves for each kᵢ (colored lines)
#   - QM curve (dash-dot black)
#   - Combined experimental spline with uncertainty ribbon (gold)
#
# Plot 2: helper figure comparing CQD z_max(kᵢ) slices against QM at selected currents
# =============================================================================
color_list = palette(:darkrainbow, n_ki);
fig = plot(xlabel="Current (A)",
    ylabel=L"$z_{\mathrm{max}}$ (mm)",
);
for (i,ki) in enumerate(cqd_meta[:ki])
    # Keep only points valid for log-log plotting
    mask_cqd = log_mask(Icoils, z_up_ki[:, i]);
    plot!(fig,Icoils[mask_cqd], z_up_ki[mask_cqd,i],
        label = L"$k_{i}=%$(round(ki, sigdigits=2))\times 10^{-6}$",
        line=(:solid,color_list[i]),
    )
end
mask_qm = log_mask(Icoils, zmax_QM);
plot!(Icoils[mask_qm],zmax_QM[mask_qm],
    label="QM",
    line=(:dashdot,:black,2),);
plot!(fig, exp_avg[:i_smooth], exp_avg[:z_smooth],
    ribbon=exp_avg[:δz_smooth],
    color=:gold,
    label="Combined experiments",
    line=(:solid,:gold,3),
    fillalpha=0.3,);
plot!(fig, 
    size=(1350,850),
    xaxis=:log10, 
    yaxis=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    # legend_title = L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(1e3*gaussian_width_mm)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",
    legendtitlefontsize = 8,
    legend_columns = 2,
    legendfontsize=7,
    left_margin=6mm,
    bottom_margin=5mm,
    foreground_color_legend=nothing);
annotate!(fig, 1e-2,1, 
    text(L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(Int(1e3*gaussian_width_mm))\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",:black,12));
display(fig)

fig = plot_cqd_vs_qm(z_up_ki, zmax_QM, Icoils, cqd_meta[:ki]);
display(fig)
# =============================================================================

# =============================================================================
# Interpolated kᵢ surface: z_max = f(I, kᵢ)
#
# We have CQD predictions on a discrete grid:
#   - I ∈ Icoils              (length nI)
#   - kᵢ ∈ cqd_meta[:ki]      (length n_ki)
# with z_up_ki[j, i] = z_max(Icoils[j], ki[i])  (units: mm)
#
# Here we build a smooth 2D interpolant:
#   ki_itp(I, kᵢ) -> z_max (mm)
#
# Important note on axis ordering:
#   Dierckx.Spline2D(x, y, z) expects z values on the x–y grid.
#   With z_up_ki sized (length(Icoils), length(ki_list)), the natural call is:
#       Spline2D(Icoils, ki_list, z_up_ki)
# which matches your storage convention z_up_ki[:, i] for fixed ki.
# =============================================================================

# Select a subset of kᵢ values for interpolation (e.g., exclude tails if needed)
ki_start , ki_stop = 1 , 109 ;
println("Interpolation in the induction term goes from ",
    (cqd_meta[:ki][ki_start]),
    "×10⁻⁶ to ",
    (round(cqd_meta[:ki][ki_stop]*1e-6, sigdigits=2)))
# Build 2D cubic spline interpolant: z_max(I, kᵢ)
# s=0 => exact interpolation (no smoothing)
ki_itp = Spline2D(Icoils, cqd_meta[:ki][ki_start:ki_stop], z_up_ki[:,ki_start:ki_stop]; kx=3, ky=3, s=0.00);

# -----------------------------------------------------------------------------
# Create a dense grid for visualization:
#   - currents from 10 mA to 1 A
#   - ki from chosen min to max
# -----------------------------------------------------------------------------
i_surface = range(10e-3,1.0; length = 101);
ki_surface = range(cqd_meta[:ki][ki_start],cqd_meta[:ki][ki_stop]; length = 101);
# Evaluate surface on a grid.
Z = [ki_itp(x, y) for y in ki_surface, x in i_surface] ;

# -----------------------------------------------------------------------------
# 3D surface plot (log10 axes for I and z)
# -----------------------------------------------------------------------------
fit_surface = surface(log10.(i_surface), ki_surface, log10.(abs.(Z));
    title = "Fitting surface",
    xlabel = L"I_{c}",
    ylabel = L"$k_{i}\times 10^{-6}$",
    zlabel = L"$z\ (\mathrm{mm})$",
    legend = false,
    color = :viridis,
    xticks = (log10.([1e-3, 1e-2, 1e-1, 1.0]), [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    zticks = (log10.([1e-3, 1e-2, 1e-1, 1.0, 10.0]), [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    camera = (20, 25),     # (azimuth, elevation)
    xlims = log10.((8e-4,2.05)),
    zlims = log10.((2e-4,10.0)),
    gridalpha = 0.3,
)

# -----------------------------------------------------------------------------
# Contour plot uses log10(z) as the displayed quantity.
# We clamp |Z| away from zero to avoid log10(0) and produce stable color limits.
# -----------------------------------------------------------------------------
Zp   = max.(abs.(Z), 1e-12);
logZ = log10.(Zp);
# Choose "decade" ticks for the colorbar based on min/max of logZ
lo , hi  = floor(minimum(logZ)) , ceil(maximum(logZ)); 
decades = collect(lo:1:hi) ; # [-4,-3,-2,-1,0] 
labels = [L"10^{%$k}" for k in decades];
fit_contour = contourf(i_surface, ki_surface, logZ; 
    levels=101,
    title="Fitting contour",
    xlabel=L"$I_{c}$ (A)", 
    ylabel=L"$k_{i}\times 10^{-6}$", 
    color=:viridis, 
    linewidth=0.2,
    linestyle=:dash,
    xaxis=:log10,
    xlims = (9e-3,1.05),
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    clims = (lo, hi),   # optional explicit range
    colorbar_ticks = (decades, labels),      # show ticks as 10^k
    colorbar_title = L"$ z \ \mathrm{(mm)}$",   # what the values mean
);

# Combined display: surface on top, contour below
fit_figs = plot(fit_surface, fit_contour,
    layout=@layout([a ; b]),
    size = (1800,750),
    bottom_margin = 8mm,
    top_margin = 3mm,
);
display(fit_figs)

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
# --- Analysis : Combined experimental data ---
#
# This section:
#   1) selects a current range from the combined experimental dataset
#   2) optionally restricts the fit to low+high current windows (use_range)
#   3) computes a global scaling factor vs QM (to match magnification / amplitude)
#   4) fits kᵢ using the interpolated CQD surface, for:
#        - original experimental data
#        - scaled experimental data
##################################################################################################
# -----------------------------------------------------------------------------
# 1) Select experimental data above a current threshold
#
# Rationale:
#   Low currents can be noisier, and log-space fitting becomes sensitive to any
#   near-zero / unstable values. We therefore start from a minimum current.
# -----------------------------------------------------------------------------
i_threshold = 0.025 ; 
i_start = searchsortedfirst(exp_avg[:i_smooth], i_threshold) ;

# Currents used for scan/plotting of fitted curves (log-spaced)
I_scan = logspace10(i_threshold, 1.00; n = 501);

# Build a convenient N×4 array: [I, δI, z, δz] and keep only I ≥ i_threshold
data = hcat(exp_avg[:i_smooth],0.02*exp_avg[:i_smooth], exp_avg[:z_smooth], exp_avg[:δz_smooth])[i_start:end, :];
pretty_table(data;
        alignment     = :c,
        title         = @sprintf("EXPERIMENTAL DATA (continuous)"),
        column_labels = ["Ic (A)","δIc (A)", "z (mm)", "δz (mm)"],
        formatters    = ([fmt__printf("%1.4f", [1]),fmt__printf("%1.4f", [2]),fmt__printf("%1.4f", [3]),fmt__printf("%1.3f", [4])]),
        style         = TextTableStyle(
                        first_line_column_label = crayon"yellow bold",
                        table_border  = crayon"blue bold",
                        column_label  = crayon"yellow bold",
                        title = crayon"bold red"
                        ),
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
        equal_data_column_widths= true,)
# -----------------------------------------------------------------------------
# 2) Choose which rows to use for the kᵢ fit
#
# Available modes:
#   - fit_ki_mode = :full
#       Use the full post-threshold current range.
#
#   - fit_ki_mode = :low
#       Use only the low-current window (small-deflection regime).
#
#   - fit_ki_mode = :high
#       Use only the high-current window (asymptotic / large-deflection regime).
#
#   - fit_ki_mode = :low_high
#       Use both low- and high-current windows, excluding the mid-current region.
#
# This flexibility allows the fit to emphasize different physical regimes,
# depending on whether sensitivity to low-current behavior, high-current
# behavior, or both is desired.
# -----------------------------------------------------------------------------
fit_ki_mode = :full   # ← change to :low, :high, or :low_high
n_front  = 30
n_back   = 200

low_range  = 1:n_front ;
high_range = (size(data, 1) - n_back + 1):size(data, 1);

@assert last(low_range) ≤ size(data,1)
@assert first(high_range) ≥ 1

# Select rows according to the chosen fitting mode
fit_ki_idx = begin
    if fit_ki_mode === :full
        Colon()
    elseif fit_ki_mode === :low
        low_range
    elseif fit_ki_mode === :high
        high_range
    elseif fit_ki_mode === :low_high
        vcat(low_range, high_range)
    else
        error("Unknown fit_ki_mode = $fit_ki_mode")
    end
end

# Informative logging
if fit_ki_mode === :full
    println("Using FULL data range for kᵢ fitting")
elseif fit_ki_mode === :low
    println("Using LOW-current range for kᵢ fitting: ",
            extrema(data[low_range, 1]), " A")
elseif fit_ki_mode === :high
    println("Using HIGH-current range for kᵢ fitting: ",
            extrema(data[high_range, 1]), " A")
elseif fit_ki_mode === :low_high
    println("Using LOW + HIGH current ranges for kᵢ fitting: ",
            extrema(data[low_range, 1]), " A & ",
            extrema(data[high_range, 1]), " A")
end

# -----------------------------------------------------------------------------
# 3) Compute a global scaling factor for the experimental z-values 
#   with respect to QM
#
# Motivation:
#   Experimental z may differ from simulated z by an overall scale factor
#   (e.g., magnification calibration). We estimate a single multiplicative
#   factor using only the highest-current tail, where SNR is typically best.
#
# Scaling convention used:
#   scaled_mag = (yexp⋅yexp) / (yexp⋅ythe)
# so that (yexp / scaled_mag) best matches ythe in a least-squares sense.
# -----------------------------------------------------------------------------
n_tail = 200  # number of tail points used for scaling

@printf "For the scaling of the experimental data, we use the current range = %.3f A – %.3f A \n" first(last(data[:, 1], n_tail)) last(last(data[:, 1], n_tail))
yexp = last(data[:, 3], n_tail)              # experimental z-values (tail)
ythe = last(zqm.(data[:, 1]), n_tail)        # QM reference z-values at same currents
scaled_mag = dot(yexp, yexp) / dot(yexp, ythe)

# Apply scaling to both z and δz to preserve relative uncertainties
data_scaled = copy(data);
data_scaled[:, 3] ./= scaled_mag;
data_scaled[:, 4] ./= scaled_mag;

@printf "The re-scaling factor of the experimental data with respect to Quantum Mechanics is %.3f" scaled_mag

# -----------------------------------------------------------------------------
# 4) Build fitting subsets and fit kᵢ using CQD interpolant surface
#
# Note:
#   fit_ki minimizes the log-space residual internally, but reports a 
#   linear-space RMSE as `ki_err`.
# -----------------------------------------------------------------------------
data_fitting        = data[fit_ki_idx, :];
data_scaled_fitting = data_scaled[fit_ki_idx, :];
fit_original = fit_ki(data, data_fitting, cqd_meta[:ki], (ki_start,ki_stop))
fit_scaled   = fit_ki(data_scaled, data_scaled_fitting, cqd_meta[:ki], (ki_start,ki_stop))

fit_ki_with_error(ki_itp, data_fitting; bounds=(cqd_meta[:ki][ki_start], cqd_meta[:ki][ki_stop]),)
fit_ki_with_error(ki_itp, data_scaled_fitting; bounds=(cqd_meta[:ki][ki_start], cqd_meta[:ki][ki_stop]),)
# =============================================================================
# Plot: QM reference + experimental data (original & scaled) + best-fit CQD curves
#
# Overlays:
#   1) QM reference curve zqm(I)
#   2) Combined experimental data (subsampled for readability)
#   3) Scaled experimental curve with uncertainty ribbon
#   4) CQD best-fit curve using kᵢ from original data fit
#   5) CQD best-fit curve using kᵢ from scaled data fit
#
# Notes:
#   - We set log-log axes at the end.
#   - Any nonpositive (I or z) values must be excluded for log plots.
# =============================================================================
# -----------------------------------------------------------------------------
# 1) QM reference curve
# -----------------------------------------------------------------------------
z_qm = zqm.(I_scan);
m_qm = log_mask(I_scan, z_qm);
fig = plot(
    I_scan[m_qm], z_qm[m_qm];
    label = "Quantum mechanics",
    line  = (:solid, :red, 1.75),
)
# -----------------------------------------------------------------------------
# 2) Combined experimental data (subsampled points for clarity)
# -----------------------------------------------------------------------------
I_exp  = data[1:2:end, 1];
z_exp  = data[1:2:end, 3];
m_exp  = log_mask(I_exp, z_exp);
plot!(
    fig,
    I_exp[m_exp], z_exp[m_exp];
    color = :gray35,
    marker = (:circle, :gray35, 1),
    markerstrokecolor = :gray35,
    markerstrokewidth = 1,
    label = "Combined data",
)
# -----------------------------------------------------------------------------
# 3) Scaled experimental curve + uncertainty ribbon
# -----------------------------------------------------------------------------
I_s  = data_scaled[:, 1];
z_s  = data_scaled[:, 3];
dz_s = data_scaled[:, 4];
m_s  = log_mask(I_s, z_s) .& isfinite.(dz_s) .& (dz_s .>= 0)
plot!(
    fig,
    I_s[m_s], z_s[m_s];
    ribbon    = dz_s[m_s],
    label     = L"Combined data (scaled $m_{p} = %$(round(scaled_mag, digits=4))$ )",
    line      = (:dash, :darkgreen, 1.2),
    fillcolor = :darkgreen,
    fillalpha = 0.35,
)
# -----------------------------------------------------------------------------
# 4) Best-fit CQD curve (fit to original experimental data)
# -----------------------------------------------------------------------------
z_fit_orig = ki_itp.(I_scan, Ref(fit_original.ki));
m_orig = log_mask(I_scan, z_fit_orig);
plot!(
    fig,
    I_scan[m_orig], z_fit_orig[m_orig];
    label = L"Original : $k_{i}= \left( %$(round(fit_original.ki, sigdigits=3)) \pm %$(round(fit_original.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
    line  = (:solid, :blue, 2),
    marker = (:xcross, :blue, 0.2),
    markerstrokewidth = 1,
)
# -----------------------------------------------------------------------------
# 5) Best-fit CQD curve (fit to scaled experimental data)
# -----------------------------------------------------------------------------
z_fit_scaled = ki_itp.(I_scan, Ref(fit_scaled.ki));
m_scaled = log_mask(I_scan, z_fit_scaled);
plot!(
    fig,
    I_scan[m_scaled], z_fit_scaled[m_scaled];
    label = L"Scaled: $k_{i}= \left( %$(round(fit_scaled.ki, sigdigits=3)) \pm %$(round(fit_scaled.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
    line  = (:solid, :lawngreen, 2),
    marker = (:xcross, :lawngreen, 0.2),
    markerstrokewidth = 1,
)
# -----------------------------------------------------------------------------
# Global plot formatting (apply once, then display once)
# -----------------------------------------------------------------------------
plot!(
    fig;
    xlabel = "Current (A)",
    ylabel = L"$z_{\mathrm{max}}$ (mm)",
    xaxis  = :log10,
    yaxis  = :log10,
    labelfontsize = 14,
    tickfontsize  = 12,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0],
              [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0],
              [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims = (0.010, 1.05),
    size  = (900, 800),
    legendtitle = L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(1e3*gaussian_width_mm)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",
    legendfontsize = 12,
    left_margin = 3mm,
)
display(fig)


# =============================================================================
# Post-fit diagnostics
#
# For BOTH datasets:
#   (A) scaled experimental data
#   (B) original experimental data
#
# we compute and compare:
#   - QM prediction vs experiment
#   - CQD best-fit prediction vs experiment
#
# Diagnostics produced:
#   1) Summary metrics + log-error scatter via compare_datasets(...)
#   2) Relative error curves: (model - exp) / exp as a function of current
#   3) CSV export of relative error curves for later analysis/plotting
# =============================================================================
# -----------------------------------------------------------------------------
# A) Scaled data diagnostics
# -----------------------------------------------------------------------------
println("Re-scaled data")
compare_datasets(data_scaled[:,1], data_scaled[:,3], ki_itp.(data_scaled[:,1], Ref(fit_scaled.ki)), zqm.(data_scaled[:,1]); plot_errors=true);
fig1 = plot(data_scaled[:,1], relerr(zqm.(data_scaled[:,1]) , data_scaled[:,3]),
    label="QM",
    line=(:solid,:red,2));
plot!(data_scaled[:,1], relerr(ki_itp.(data_scaled[:,1], Ref(fit_scaled.ki)) , data_scaled[:,3]),
    label=L"CQD ($k_{i}=%$(round(fit_scaled.ki,sigdigits=4)) \times10^{-6}$)",
    line=(:solid,:blue,2));
plot!(
    title="Relative Error - Scaled data",
    titlefontsize=24,
    xlabel = "Current (A)",
    ylabel = "Rel.Error",
    xaxis=:log10,
    # yaxis=:log10,
    labelfontsize=14,
    tickfontsize=12,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(0.020,1.05),
    # ylims=(-0.10,0.25),
    size=(800,500),
    legendtitle=L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(1e3*gaussian_width_mm)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",
    legendfontsize=12,
    left_margin=3mm,
    bottom_margin=3mm,
)

df_scaled = DataFrame(hcat(data_scaled[:,1],
    relerr(zqm.(data_scaled[:,1]) , data_scaled[:,3]),
    relerr(ki_itp.(data_scaled[:,1], Ref(fit_scaled.ki)) , data_scaled[:,3])
),
[:Ic, :eQM, :eCQD]
)
CSV.write(joinpath(OUTDIR,"rel_error_scaled.csv"),df_scaled)

# -----------------------------------------------------------------------------
# B) Original data diagnostics
# -----------------------------------------------------------------------------
println("Original data")
compare_datasets(data[:,1], data[:,3], ki_itp.(data[:,1], Ref(fit_original.ki)), zqm.(data[:,1]); plot_errors=true);
fig2=plot(data[:,1], relerr(zqm.(data[:,1]) , data[:,3]) ,
    label="QM",
    line=(:solid,:red,2))
plot!(data[:,1], relerr(ki_itp.(data[:,1], Ref(fit_original.ki)) , data[:,3]) ,
    label=L"CQD ($k_{i}=%$(round(fit_original.ki,sigdigits=4)) \times10^{-6}$)",
    line=(:solid,:blue,2))
plot!(
    title="Relative Error - Original data",
    titlefontsize=24,
    xlabel = "Current (A)",
    ylabel = "Rel.Error",
    xaxis=:log10,
    # yaxis=:log10,
    labelfontsize=14,
    tickfontsize=12,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(0.020,1.05),
    # ylims=(-0.10,0.25),
    size=(800,500),
    legendtitle=L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(1e3*gaussian_width_mm)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",
    legendfontsize=12,
    left_margin=3mm,
    bottom_margin=3mm,
)

df_orig = DataFrame(hcat(data[:,1],
    relerr(zqm.(data[:,1]) , data[:,3]) ,
    relerr(ki_itp.(data[:,1], Ref(fit_original.ki)) , data[:,3]) 
),
[:Ic, :eQM, :eCQD]
)
CSV.write(joinpath(OUTDIR,"rel_error_original.csv"),df_orig)

# -----------------------------------------------------------------------------
# Combined figure (original on top, scaled on bottom)
# -----------------------------------------------------------------------------
fig= plot(fig2,fig1,
    layout=(2,1),
    size=(1000,600))


# ==============================================================================
# ==============================================================================
# ==============================================================================

function plot_zmax_vs_current(
        data_exp,
        ki_list;
        Icurrent,
        zmm_cqd,
        zmax_QM,
        data_label = "experiment",
        p = 1.0,
        scale_exp = false,
        axis_scale = :loglog,   # :linear, :loglog, :semilogx, :semilogy
        figsize = (850, 600),
    )

    """
        plot_zmax_vs_current(data_exp, ki_list; Icurrent, zmm_cqd, zmax_QM,
                             data_label="experiment", p=1.0, scale_exp=false,
                             axis_scale=:loglog, figsize=(850,600), warn_drop=true)

    Plot `z_max` vs coil current for:
      - CQD curves (one per `ki_list[i]`)
      - Experimental data with uncertainty ribbon
      - QM reference curve

    Masking behavior (always applied, independent of axis_scale):
      - Points are removed if x or y is non-finite (NaN/Inf)
      - Points are removed if x ≤ 0 or y ≤ 0
      This makes the function safe for log plotting without manual slicing.

    Notes:
      - Each CQD curve is masked independently (curve-by-curve).
      - Experimental and QM curves are also masked independently.
    """

    # --- checks ---
    @assert size(zmm_cqd, 1) == length(Icurrent) "zmm_cqd must have size (length(Icurrent), n_ki)"
    @assert length(zmax_QM)  == length(Icurrent) "zmax_QM must match length(Icurrent)"
    @assert size(zmm_cqd, 2) >= length(ki_list)  "zmm_cqd must have ≥ length(ki_list) columns"

    # --- Axis scaling ---
    if axis_scale == :loglog
        xscale = :log10
        yscale = :log10
    elseif axis_scale == :semilogx
        xscale = :log10
        yscale = :identity
    elseif axis_scale == :semilogy
        xscale = :identity
        yscale = :log10
    else
        xscale = :identity
        yscale = :identity
    end

    # Colors
    color_list = palette(:darkrainbow, length(ki_list))

    # --- Figure setup ---
    fig = plot(
        xlabel = "Currents (A)",
        ylabel = L"$z_{\mathrm{max}}$ (mm)",
        legend = :outerright,
    )

    # -------------------------------------------------------------------------
    # CQD curves: each curve gets its own mask (ALWAYS requiring x>0 and y>0)
    # -------------------------------------------------------------------------
    for (j, ki) in enumerate(ki_list)
        x = Icurrent
        y = view(zmm_cqd, :, j)

        mask_cqd = log_mask(x,y)

        plot!(
            fig,
            x[mask_cqd],
            y[mask_cqd],
            label = L"$k_{i}=%$(round(ki, sigdigits=3))\times 10^{-6}$",
            line  = (:solid, color_list[j]),
        )
    end

    # -------------------------------------------------------------------------
    # Experimental data: own mask, also requiring x>0 and y>0
    # -------------------------------------------------------------------------
    I_exp  = data_exp[!, :Ic]
    z_exp  = scale_exp ? data_exp[!, :z]  ./ p : data_exp[!, :z]
    dz_exp = scale_exp ? data_exp[!, :δz] ./ p : data_exp[!, :δz]

    mask_exp = log_mask(I_exp,z_exp)

    plot!(
        fig,
        I_exp[mask_exp],
        z_exp[mask_exp],
        ribbon    = dz_exp[mask_exp],
        line      = (:black, :dash, 2),
        fillalpha = 0.25,
        fillcolor = :gray13,
        label     = data_label,
    )

    # -------------------------------------------------------------------------
    # QM curve: own mask, requiring x>0 and y>0 (use Icurrent, not global Icoils)
    # -------------------------------------------------------------------------
    xQ = Icurrent
    yQ = zmax_QM

    mask_qm = log_mask(xQ,yQ)

    plot!(
        fig,
        xQ[mask_qm],
        yQ[mask_qm],
        label = "QM",
        line  = (:dashdot, :magenta, 3),
    )

    # --- Layout ---
    plot!(
        fig;
        size = figsize,
        xaxis = xscale,
        yaxis = yscale,
        legend = :outerright,
        legend_columns = 2,
        legendfontsize = 4,
        left_margin = 3mm,
    )

    return fig
end

function plot_scaling_factor(n, data_exp, wanted_data_dir, mag; zqm)
    """
        plot_scaling_factor(n, data_exp, wanted_data_dir, mag; zqm)

    Compute a multiplicative scaling factor `p` from the last `n` points and
    visualize:

    - Experimental data: z_exp(I)
    - QM curve:          z_qm(I)
    - Scaled data:       z_exp(I)/p

    The factor is computed by a 1-parameter least-squares match between the
    experimental tail and the QM tail:

        p = (yexp⋅yexp) / (yexp⋅ythe)

    so that yexp/p best matches ythe in the dot-product sense.

    Inputs
    - `n`               : number of tail points used to compute scaling
    - `data_exp`        : DataFrame (must include columns :Ic and :z)
    - `wanted_data_dir` : string label (used in plot legend)
    - `mag`             : original magnification value (for displaying scaled mag)
    - `zqm`             : callable; maps current I -> z_qm(I)

    Returns
    - `p`   : scaling factor (dimensionless)
    - `fig` : plot showing original/ theory / scaled curves
    """

    # --- Compute scaling factor ---
    yexp = last(data_exp[!, :z], n)
    ythe = last(zqm.(data_exp[!, :Ic]), n)
    p = dot(yexp, yexp) / dot(yexp, ythe)

    # Scaled magnification
    scaled_mag = mag * p

    # --- Build plot ---
    fig = plot(
        data_exp[!, :Ic], data_exp[!, :z],
        label = wanted_data_dir,
        marker = (:circle, :white, 2),
        markerstrokecolor = :red,
        line = (:solid, :red, 1),
        ylabel = L"$z_{max}$ (mm)",
        xlabel = "Current (A)",
        title = "Magnification: $(@sprintf("%1.4f", mag)) → $(@sprintf("%1.4f", scaled_mag))",
    )

    plot!(
        fig,
        data_exp[!, :Ic],
        zqm.(data_exp[!, :Ic]),
        label = "QM",
        marker = (:xcross, :blue, 2),
        markerstrokecolor = :red,
        line = (:solid, :blue, 1),
    )

    plot!(
        fig,
        data_exp[!, :Ic],
        data_exp[!, :z] ./ p,
        label = "data / $(@sprintf("%1.3f", p))",
        marker = (:circle, :white, 2),
        markerstrokecolor = :orangered,
        line = (:solid, :orangered, 1),
    )

    return p, fig
end

function fit_k_parameter(data_fitting, p, ki_list, ki_start, ki_stop;
                         ki_itp,
                         I_exp,
                         z_exp)
    """
    fit_k_parameter(data_fitting, p, ki_list, ki_start, ki_stop; ki_itp, I_exp, z_exp)

    Fit the induction parameter `kᵢ` using a log-space MSE objective after scaling
    experimental data by a factor `p`.

    Workflow
    1) Copy `data_fitting` and scale its last two columns by `p`
    (by convention: these are [z, δz] or similar)
    2) Define a log-space loss:
        mean( (log10(z_pred) - log10(z_obs))^2 )
    3) Minimize over `kᵢ` using Brent on [ki_list[ki_start], ki_list[ki_stop]]
    4) Compute diagnostics:
    - `mse` = loss at optimum (log-space)
    - `coef_r2` = R² in linear space on (I_exp, z_exp)

    Inputs
    - `data_fitting` : numeric matrix; expected to include columns:
                    col 1 = I, col 3 = z (used in loss), last two columns are scaled by p
    - `p`            : scaling factor applied to last two columns
    - `ki_list`      : vector of candidate kᵢ values
    - `ki_start/stop`: indices selecting the fitting bounds within `ki_list`

    Keywords
    - `ki_itp` : callable (I, ki) -> z_pred
    - `I_exp`  : currents for R² evaluation
    - `z_exp`  : experimental z values for R² evaluation

    Returns
    - `(k_fit, mse, coef_r2)`
    """

    # --- Scale the data using p ---
    data_scaled = copy(data_fitting)
    data_scaled[:, end-1:end] ./= p   # scale last two columns

    # --- Define loss function (same as yours) ---
    loss_scaled(ki) = begin
        z_pred = ki_itp.(data_scaled[:, 1], Ref(ki))
        mean(abs2, log10.(z_pred) .- log10.(data_scaled[:, 3]))
    end

    # --- Brent optimization ---
    fit_param = optimize(loss_scaled,
                         ki_list[ki_start],
                         ki_list[ki_stop],
                         Brent())

    k_fit = Optim.minimizer(fit_param)

    # --- Diagnostics ---
    mse = loss_scaled(k_fit)

    pred = ki_itp.(I_exp, Ref(k_fit))

    coef_r2 = 1 - sum(abs2, pred .- z_exp) / sum(abs2, z_exp .- mean(z_exp))

    return k_fit, mse, coef_r2
end

function plot_full_ki_fit(
        data_exp, data_fitting,
        p0,p, 
        k_fit, loss_scaled;
        wanted_data_dir,
        wanted_binning,
        wanted_smooth,
        ki_itp,
        zqm,
        out
    )
    """
    Summary plot (log–log) showing:
      - Raw experimental data with y-error bars
      - Subset used for fitting (raw)
      - CQD best-fit (unscaled) + CI band from `out.ci`
      - Scaled experimental data (divide by `p`) with y-error bars
      - Subset used for fitting (scaled)
      - CQD best-fit for scaled fit (`k_fit`)
      - QM reference curve `zqm(I)`

    Inputs
      - data_exp     : DataFrame with columns :Ic, :z, :δz
      - data_fitting : Matrix with columns [I, δI, z, δz] (your convention)
      - p0, p        : magnification factors (annotation only; scaling uses p)
      - k_fit        : best-fit kᵢ for scaled fit
      - loss_scaled  : fit loss value for scaled fit (NOT an uncertainty)

    Keywords
      - ki_itp : (I, ki) -> z_CQD
      - zqm    : I -> z_QM
      - out    : output from fit_ki_with_error (expects k_hat, k_err, ci)
    """

    # --- Scale the data using p ---
    data_fitting_scaled = copy(data_fitting)
    data_fitting_scaled[:, end-1:end] ./= p   # scale last two columns

    Iscan = logspace10(10e-3, 1.00; n=201);
    # --- Main figure ---
    fig = plot(
        size = (850,600),
        xlabel = L"Coil current $I_{c}$ (A)",
        ylabel = L"$z$ (mm)",
        left_margin = 2mm,
    )

    # --- Raw experimental data ---
    plot!(
        fig,
        data_exp[!, :Ic], data_exp[!, :z],
        label = "Experiment $(wanted_data_dir): n=$(wanted_binning) | λ=$(wanted_smooth)",
        seriestype = :scatter,
        yerror = data_exp[!, :δz],
        marker = (:circle, :white, 1.8),
        markerstrokecolor = :red,
        markerstrokewidth = 2,
    )

    # --- Points used for fitting (unscaled) ---
    plot!(
        fig,
        data_fitting[:, 1], data_fitting[:, 3],
        seriestype = :scatter,
        label = "Used for fitting",
        marker = (:xcross, :black, 2),
        markerstrokecolor = :black,
        markerstrokewidth = 2,
    )

    # --- Confidence interval band (unscaled) ---
    z_lo = ki_itp.(Iscan, Ref(out.ci[1]))
    z_hi = ki_itp.(Iscan, Ref(out.ci[2]))
    mb   = log_mask(Iscan, z_lo) .& log_mask(Iscan, z_hi)
    plot!(
        fig,
        Iscan[mb],
        z_hi[mb],
        color = :royalblue1,
        label = false,
        linewidth = 0,
        fillrange = z_lo[mb],
        fillcolor = :royalblue1,
        fillalpha = 0.35,
    )

    # --- Best-fit curve (unscaled) ---
    plot!(
        fig,
        Iscan,
        ki_itp.(Iscan, Ref(out.k_hat)),
        label = L"$k_{i}= \left( %$(round(out.k_hat, digits=4)) \pm %$(round(out.k_err, digits=4)) \right) \times 10^{-6} $",
        line = (:solid, :blue, 2),
        marker = (:xcross, :blue, 1),
    )

    # Dummy entry for legend separation
    plot!([1,1], label = "Scaled Magnification m=($(round(p,sigdigits=5))×$(round(p0, sigdigits=5))) ", color = :white)

    # --- Scaled experimental data ---
    plot!(
        fig,
        data_exp[!, :Ic], data_exp[!, :z] ./ p,
        label = "Experiment $(wanted_data_dir): n=$(wanted_binning) | λ=$(wanted_smooth)",
        seriestype = :scatter,
        yerror = data_exp[!, :δz] ./ p,
        marker = (:circle, :white, 1.8),
        markerstrokecolor = :orangered2,
        markerstrokewidth = 2,
    )

    # --- Points used for fitting (scaled) ---
    plot!(
        fig,
        data_fitting_scaled[:, 1], data_fitting_scaled[:, 3],
        seriestype = :scatter,
        label = "Used for fitting (scaled)",
        marker = (:xcross, :grey26, 2),
        markerstrokecolor = :grey26,
        markerstrokewidth = 2,
    )

    # --- Best-fit curve for scaled data ---
    plot!(
        fig,
        Iscan,
        ki_itp.(Iscan, Ref(k_fit)),
        label = L"$k_{i}= \left( %$(round(k_fit, digits=4)) \pm %$(round(loss_scaled, sigdigits=1)) \right) \times 10^{-6} $",
        line = (:solid, :seagreen, 2.2),
        marker = (:xcross, :seagreen, 1),
    )

    # --- Axes, ticks, limits ---
    plot!(
        fig;
        xaxis = :log10,
        yaxis = :log10,
        xticks = ([1e-3, 1e-2, 1e-1, 1.0],
                  [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-3, 1e-2, 1e-1, 1.0],
                  [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xlims = (8e-3, 1.5),
        ylims = (8e-3, 2),
        legend = :bottomright,
    )

    # --- QM curve ---
    xQ = Iscan
    yQ=zqm.(Iscan)
    maskQ=log_mask(xQ,yQ)
    plot!(
        fig,
        xQ[maskQ], yQ[maskQ],
        label = "QM",
        line = (:dashdot, :black, 2),
    )

    return fig
end

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Batch analysis: process each experimental dataset directory
#
# For each dataset:
#   1) find the corresponding analysis report / metadata (binning + smoothing)
#   2) load per-frame data and propagate uncertainties into z_exp_error
#   3) apply current threshold (drop low-I region)
#   4) plot raw (unscaled) experiment vs CQD family vs QM
#   5) estimate global scaling factor p from the last n points (magnification correction)
#   6) plot scaled experiment (z/p) vs CQD family vs QM
#   7) choose fitting subset and fit kᵢ (scaled + unscaled uncertainty estimate)
#   8) create a summary plot with fit curves and confidence band
# ------------------------------------------------------------------------------
# List of experimental data directories to process
wanted_data_dirs = [
    "20250814", "20250820",
    "20250825", "20250919",
    "20251002", "20251003",
    "20251006", "20251109"
    # ...
]

wanted_binning = 2
wanted_smooth  = 0.01
current_threshold = 0.020

for wanted_data_dir in wanted_data_dirs
    @info "Processing dataset" wanted_data_dir

    # --------------------------------------------------------------------------
    # 1) Locate the matching experimental analysis output (report + paths)
    # --------------------------------------------------------------------------
    res = DataReading.find_report_data(
        joinpath(@__DIR__, "analysis_data");
        wanted_data_dir = wanted_data_dir,
        wanted_binning  = wanted_binning,
        wanted_smooth   = wanted_smooth,
    )

    if res === nothing
        @warn "No matching report found" wanted_data_dir
        continue
    else
        @info "Imported experimental data" "Path\t\t" = res.path "Date label\t\t"  = res.data_dir "Analysis label\t" = res.name "Binning\t\t" = res.binning "Smoothing\t\t" =res.smoothing

    end

    mag, δmag = MyExperimentalAnalysis.mag_factor(wanted_data_dir)

    # --------------------------------------------------------------------------
    # 2) Load framewise data and propagate uncertainties into z_exp_error
    # --------------------------------------------------------------------------
    load_data   = CSV.read(joinpath(dirname(res.path), "fw_data.csv"), DataFrame; header = true)
    I_exp       = load_data[!, "Icoil_A"]
    I_exp_error = load_data[!, "Icoil_error_A"]
    z_exp       = load_data[!, "F1_z_centroid_mm"] / mag
    # Propagate uncertainty: combine centroid SEM and magnification uncertainty
    # δz = |z| * sqrt( (δz_centroid / z_centroid)^2 + (δmag/mag)^2 )
    z_exp_error = abs.(z_exp) .* sqrt.(
        (load_data[!, "F1_z_centroid_se_mm"] ./ load_data[!, "F1_z_centroid_mm"]).^2 .+
        (δmag / mag)^2
    )

    # --------------------------------------------------------------------------
    # 3) Apply current threshold: ignore low-current region
    # --------------------------------------------------------------------------
    i_start  = searchsortedfirst(I_exp, current_threshold)
    # Store a numeric matrix for fitting and a DataFrame for plotting convenience
    data     = hcat(I_exp, I_exp_error, z_exp, z_exp_error)[i_start:end, :]
    data_exp = DataFrame(data, [:Ic, :δIc, :z, :δz])

    # --------------------------------------------------------------------------
    # 4) Plot raw experiment vs CQD family vs QM (unscaled)
    # --------------------------------------------------------------------------
    fig = plot_zmax_vs_current(
        data_exp, cqd_meta[:ki];
        Icurrent   = Icoils,
        zmm_cqd    = z_up_ki,
        zmax_QM    = zmax_QM,
        axis_scale = :identity,
        data_label = wanted_data_dir,
    )
    display(fig)


    # --------------------------------------------------------------------------
    # 5) Estimate scaling factor p from the last n points (magnification correction)
    # --------------------------------------------------------------------------
    p, fig = plot_scaling_factor(
        2,
        data_exp,
        wanted_data_dir,
        mag;
        zqm = zqm,
    )
    display(fig)
    println("Scaled Magnification factor 𝓂 = $(@sprintf("%2.4f", mag * p))")

    # --------------------------------------------------------------------------
    # 6) Plot scaled experiment (z/p) vs CQD family vs QM
    # --------------------------------------------------------------------------
    fig = plot_zmax_vs_current(
        data_exp, cqd_meta[:ki];
        zmm_cqd    = z_up_ki,
        Icurrent   = Icoils,
        zmax_QM    = zmax_QM,
        p          = p,
        scale_exp  = true,
        data_label = wanted_data_dir,
        axis_scale = :identity,
    )
    display(fig)

    # --------------------------------------------------------------------------
    # 7) Choose fitting subset and fit kᵢ
    #
    # Current choice: first 4 and last 4 points after threshold.
    # (You might later replace this with your fit_mode logic.)
    # --------------------------------------------------------------------------
    data_fitting = data[[1:4; (end-3):end], :]

    # 7a) Fit kᵢ using scaled data (your custom fit_k_parameter)
    k_fit, mse, r2 = fit_k_parameter(
        data_fitting,
        p,
        cqd_meta[:ki],
        ki_start,
        ki_stop;
        ki_itp = ki_itp,
        I_exp  = I_exp,
        z_exp  = z_exp,
    )
    @info "Fitting for rescaled data (𝓂 = $(p*mag))" "kᵢ\t\t" = k_fit "Err kᵢ\t" = mse "R²\t\t" = r2

    # 7b) Fit kᵢ with uncertainty estimate (unscaled magnification, using fit_ki_with_error)
    out = fit_ki_with_error(
        ki_itp,
        data_fitting;
        bounds = (cqd_meta[:ki][ki_start], cqd_meta[:ki][ki_stop]),
    )
    @info "Fitting (𝓂 = $(mag))" "kᵢ\t\t" = out.k_hat "Err kᵢ\t" = out.k_err "kᵢ interval\t" = out.ci

    # --------------------------------------------------------------------------
    # 8) Summary plot: raw + scaled data, fit curves, CI band, QM curve
    # --------------------------------------------------------------------------
    fig = plot_full_ki_fit(
        data_exp, data_fitting,
        mag, p, k_fit, mse;
        wanted_data_dir = wanted_data_dir,
        wanted_binning  = wanted_binning,
        wanted_smooth   = wanted_smooth,
        ki_itp          = ki_itp,
        zqm             = zqm,
        out             = out,
    )
    display(fig)

    println("Finished processing dataset $wanted_data_dir\n" * "-"^60 * "\n")
end
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

function fit_ki_joint_scaling_fitsubset(
    data,
    zqm,
    ki_itp,
    thresholdI::Float64,
    ki_range;
    fit_ki_mode::Symbol = :low_high,     # :full | :low | :high | :low_high
    n_front::Int = n_front,
    n_back::Int  = n_back,
    w::Float64   = 0.5,
    ref_type::Symbol = :arith,        # :arith or :geom
    )
    """
        fit_ki_joint_scaling_fitsubset(
            data, zqm, ki_itp, thresholdI, ki_range;
            fit_mode=:low_high, n_front=50, n_back=50,
            w=0.7, ref_type=:arith
        )

    Fit the CQD parameter `ki` while dynamically rescaling the experimental data
    using a high-current tail region and a combined QM+CQD reference.

    Inputs
    ------
    data :: AbstractMatrix{<:Real}
        N×4 (or wider) array. The function uses:
            - `data[:,1]` : Iexp  experimental current (A)
            - `data[:,3]` : yexp  experimental z_max (mm)
        (Any other columns are ignored here.)

    zqm :: callable
        Interpolated QM model: `zqm(I) -> z_QM(I)`.

    ki_itp :: callable
        Interpolated CQD model: `ki_itp(I, ki) -> z_CQD(I; ki)`.

    thresholdI :: Real
        Current threshold (A) defining the tail region used ONLY for scaling.
        Only points with `Iexp ≥ thresholdI` contribute to the scale factor.

    ki_range :: Tuple{<:Real,<:Real}
        (kmin, kmax) bracket for the 1D Brent optimizer over `ki`.

    Keyword arguments
    -----------------
    fit_mode :: Symbol = :low_high
        Select which experimental points are used for the *ki fit* (NOT for scaling):
            :full      → use all points
            :low       → use only the first `n_front` points
            :high      → use only the last  `n_back` points
            :low_high  → use both first `n_front` and last `n_back` points

    n_front :: Int = 50
        Number of lowest-current points used when `fit_mode` includes low-current.

    n_back :: Int = 50
        Number of highest-current points used when `fit_mode` includes high-current.

    w :: Real = 0.7
        Weight for the QM model in the combined tail reference model.

    ref_type :: Symbol = :arith
        Tail reference type:
            :arith → z_ref = w*zQM + (1-w)*zCQD
            :geom  → z_ref = zQM^w * zCQD^(1-w)   (weighted in log space)

    Method summary
    --------------
    1) Identify tail points (I ≥ thresholdI) used ONLY to compute a global scale factor.
    2) For each trial `ki`, build a tail reference curve z_ref_tail(ki) from QM and CQD.
    3) Compute scale(ki) by least-squares projection of y_tail onto z_ref_tail.
            s(ki) = <y_tail, z_ref_tail> / <z_ref_tail, z_ref_tail>
    4) Scale the entire experimental dataset by 1/scale(ki).
    5) Fit `ki` by minimizing mean squared log-residuals between scaled experiment and CQD
    restricted to the subset selected by `fit_mode`.
    6) Return best-fit ki, final scale factor, and the minimum loss value.
    """
    # --- Unpack experimental current and z_max ---
    Iexp = data[:, 1]
    yexp = data[:, 3]
    N = length(Iexp)

    # ------------------------------
    # 1) Tail region (for scaling)
    # Use only points with Iexp ≥ thresholdI to compute the scale factor.
    # ------------------------------
    tail_idx = findall(Iexp .>= thresholdI)
    isempty(tail_idx) && error("No experimental points with current ≥ $thresholdI A")

    I_tail = Iexp[tail_idx]
    y_tail = yexp[tail_idx]

    # -----------------------------------------
    # 2) Fitting region (for ki optimization)
    # -----------------------------------------
    # Only use a subset of points for the ki fit:
    # - first n_front points (low current)
    # - last  n_back points (high current)
    low_range  = 1:min(n_front, N)
    high_range = max(1, N - n_back + 1):N

    fit_idx = begin
        if fit_ki_mode === :full
            collect(1:N)
        elseif fit_ki_mode === :low
            collect(low_range)
        elseif fit_ki_mode === :high
            collect(high_range)
        elseif fit_ki_mode === :low_high
            vcat(collect(low_range), collect(high_range))
        else
            error("Invalid fit_ki_mode = $fit_ki_mode. Use :full, :low, :high, or :low_high.")
        end
    end

    # Informative logging
    if fit_ki_mode === :full
        println("Using FULL data range for kᵢ fitting")
    elseif fit_ki_mode === :low
        println("Using LOW-current range for kᵢ fitting: ", extrema(Iexp[low_range]), " A")
    elseif fit_ki_mode === :high
        println("Using HIGH-current range for kᵢ fitting: ", extrema(Iexp[high_range]), " A")
    elseif fit_ki_mode === :low_high
        println("Using LOW + HIGH current ranges for kᵢ fitting: ",
                extrema(Iexp[low_range]), " A & ",
                extrema(Iexp[high_range]), " A")
    end

    # ------------------------------------------------
    # 3) Reference model in the tail: z_ref(I; ki)
    # ------------------------------------------------
    # Given a trial ki, construct the tail-region reference curve that
    # will be used to define the scale factor. This can be:
    #
    #  - arithmetic blend:  z_ref = w*zQM + (1-w)*zCQD
    #  - geometric blend:   z_ref = zQM^w * zCQD^(1-w)
    #
    # Note: we only evaluate this in the tail region.
    function zref_tail_for(ki)
        zqm_tail  = zqm.(I_tail)
        zcqd_tail = ki_itp.(I_tail, ki)

        if ref_type === :arith
            return w .* zqm_tail .+ (1 - w) .* zcqd_tail
        elseif ref_type === :geom
            return zqm_tail .^ w .* zcqd_tail .^ (1 - w)
        else
            error("Invalid ref_type = $ref_type. Use :arith or :geom.")
        end
    end

    # ------------------------------------------------
    # 4) Loss function over ki
    # For each trial ki, we:
    #  - build CQD prediction y_cqd(I; ki)
    #  - compute the tail reference z_ref_tail(ki)
    #  - compute the optimal scale s(ki) so that
    #       y_tail ≈ s(ki) * z_ref_tail
    #    in a least-squares sense:
    #       s(ki) = <y_tail, z_ref_tail> / <z_ref_tail, z_ref_tail>
    #  - scale the *full* experimental data by 1/s(ki)
    #  - compute log-residuals between scaled experiment and CQD model
    #    only for the chosen fit_idx points
    #  - return mean squared log-residuals as the loss.
    # ------------------------------------------------
    function loss(ki)
        # CQD prediction at all experimental currents
        y_cqd = ki_itp.(Iexp, ki)

        # Tail reference curve (QM/CQD blend) and projection scale
        zref_tail = zref_tail_for(ki)
        # Scale factor: least-squares projection of y_tail onto zref_tail
        scale = dot(y_tail, zref_tail) / dot(zref_tail, zref_tail)

        # Apply global scaling to experimental data
        yexp_scaled = yexp ./ scale

        # Log-residuals only on the selected fitting subset
        r = log.(yexp_scaled[fit_idx]) .- log.(y_cqd[fit_idx])

        return mean(abs2, r)
    end

    # ------------------------------------------------
    # 5) 1D optimization over ki (Brent)
    # ------------------------------------------------
    kmin, kmax = ki_range
    opt = optimize(loss, kmin, kmax, Brent())

    ki_fit = Optim.minimizer(opt)   # best-fit ki
    mse    = Optim.minimum(opt)     # minimum loss value (mean squared log-residual)

    # ------------------------------------------------
    # 6) Final scale factor at ki_fit (for output)
    # Recompute the reference tail with the best-fit ki and get the final
    # scale factor that defines the global rescaling of the experiment.
    # ------------------------------------------------
    zref_tail_final = zref_tail_for(ki_fit)
    scale_final = dot(y_tail, zref_tail_final) / dot(zref_tail_final, zref_tail_final)

    # ------------------------------------------------
    # 7) Return results
    # ------------------------------------------------
    return (
        ki_fit       = ki_fit,        # best-fit ki
        scale_factor = scale_final,   # global magnification correction
        mse          = mse,           # mean squared log-residual at optimum
    )
end

# =============================================================================
# Joint scaling + kᵢ fit on the COMBINED experimental curve (exp_avg)
# =============================================================================

# Build the combined dataset matrix expected by the pipeline:
#   col1 = I (A)
#   col2 = δI (A)        (not used by fit_ki_joint_scaling_fitsubset, but kept)
#   col3 = z (mm)
#   col4 = δz (mm)
#
# NOTE: you currently set δI = 0.02*I as a placeholder.
i_start = searchsortedfirst(exp_avg[:i_smooth], i_threshold)
data     = hcat(
    exp_avg[:i_smooth],
    0.02*exp_avg[:i_smooth],
    exp_avg[:z_smooth],
    exp_avg[:δz_smooth]
)[i_start:end,:]
# -----------------------------------------------------------------------------
# 1) Fit ki with joint scaling from high-current tail
# -----------------------------------------------------------------------------
result = fit_ki_joint_scaling_fitsubset(
    data,
    zqm,
    ki_itp,
    0.750,                                  # tail threshold
    (cqd_meta[:ki][ki_start], cqd_meta[:ki][ki_stop]); # bracket
    fit_ki_mode=fit_ki_mode,
    n_front = n_front,
    n_back  = n_back,
    w       = 0.50,
    ref_type=:arith,
)

# -----------------------------------------------------------------------------
# 2) Apply the fitted scale factor to the experimental data
# -----------------------------------------------------------------------------
# `scale_factor` is the best-fit global scaling s such that:
#   y_tail ≈ s * z_ref_tail
# We want "scaled experiment" -> y_scaled = y / s
scale_mag = result.scale_factor
 
data_scaled = copy(data)
data_scaled[:, 3] ./= scale_mag    # scale z
data_scaled[:, 4] ./= scale_mag    # scale δz

# If you want to report the *global* magnification (instrument mag × scaling)
mag0, _ = MyExperimentalAnalysis.mag_factor("20250814")   # baseline dataset choice
global_mag_factor = scale_mag * mag0

# -----------------------------------------------------------------------------
# 3) Define the exact fitting subset indices (so plotting matches fit_mode)
# -----------------------------------------------------------------------------
data_fitting        = data[fit_ki_idx, :]
data_scaled_fitting = data_scaled[fit_ki_idx, :]

# -----------------------------------------------------------------------------
# 4) (Optional) Re-fit ki on the scaled data using your `fit_ki` utility
#    - This is separate from the joint fit above (it uses log-space objective internally)
#    - Here we report a linear-space error (if your fit_ki returns RMSE in mm)
# -----------------------------------------------------------------------------
fit_scaled = fit_ki(
    data_scaled,            # full dataset for R² (I,z)
    data_scaled_fitting,    # fitting subset for the loss
    cqd_meta[:ki],
    (ki_start, ki_stop),
)

# -----------------------------------------------------------------------------
# 5) Plot: scaled experiment + QM + CQD best-fit
# -----------------------------------------------------------------------------
fig=plot(    
    title = L"Peak position ($F=1$)",)
# Scaled experimental curve with ribbon
plot!(fig,
    data_scaled[:,1],data_scaled[:,3],
    ribbon = data_scaled[:,4],
    label=L"Experimental data (magnif.factor $m = %$(round(global_mag_factor, digits=4))$)",
    line=(:dash,:darkgreen,3),
    fillcolor = :darkgreen,
    fillalpha = 0.35,
)
# QM reference curve
plot!(fig,I_scan, zqm.(I_scan),
    label="Quantum mechanical model",
    line=(:solid,:red,1.75)
)
# CQD best-fit curve (from scaled refit)
plot!(fig,
    I_scan, ki_itp.(I_scan, Ref(fit_scaled.ki)),
    label=L"CoQuantum dynamics: $k_{i}= \left( %$(round(fit_scaled.ki; sigdigits=4)) \pm %$(round(fit_scaled.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
    line=(:dot,:blue,2),
    # marker=(:xcross, :blue, 0.2),
    markerstrokewidth=1
)
# Global formatting
plot!(fig,
    xlabel = "Coil Current (A)",
    ylabel = L"$z_{\mathrm{max}}$ (mm)",
    xaxis=:log10,
    yaxis=:log10,
    labelfontsize=14,
    tickfontsize=12,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # xlims=(0.010,1.05),
    size=(900,800),
    # legendtitle=L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(1e3*gaussian_width_mm)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",
    legendfontsize=12,
    left_margin=3mm,
)
display(fig)



# =============================================================================
# SCATTERED DATA
# Plot scaled scattered experiment vs QM and CQD
#   (1) x-axis = current
#   (2) x-axis = gradient
# =============================================================================

# -------------------------------------------------------------------------
# 1) Prepare scattered experimental points (from spline + propagated errors)
# -------------------------------------------------------------------------
i_start = searchsortedfirst(data_exp_scattered[:,1], i_threshold)
# columns: I, δI, z, δz
data = data_exp_scattered[i_start:end, :]

# -------------------------------------------------------------------------
# 2) Apply the global scaling factor found earlier (from joint scaling fit)
# -------------------------------------------------------------------------
data_scaled = copy(data)
data_scaled[:, 3] ./= scaled_mag
data_scaled[:, 4] ./= scaled_mag

fig=plot(    
    # title = L"Peak position ($F=1$)",
)
plot!(fig,
    data_scaled[:,1],data_scaled[:,3],
    xerr = data_scaled[:,2],
    yerr = data_scaled[:,4],
    label=L"Experimental data (magnif.factor $m = %$(round(global_mag_factor, digits=4))$)",
    seriestype=:scatter,
    marker = (:circle,4,:white,stroke(0.5,:black) )
    # line=(:dash,:darkgreen,3),
    # fillcolor = :darkgreen,
    # fillalpha = 0.35,
)
plot!(fig,I_scan, zqm.(I_scan),
    label="Existing models",
    line=(:dash,:blue,1.75)
)
plot!(fig,
    I_scan, ki_itp.(I_scan, Ref(fit_scaled.ki)),
    label=L"Coquantum dynamics: $k_{i}= \left( %$(round(fit_scaled.ki, sigdigits=3)) \pm %$(round(fit_scaled.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
    line=(:solid,:red,2),
    # marker=(:xcross, :blue, 0.2),
    markerstrokewidth=1
)
plot!(fig,
    xlabel = "Coil Current (A)",
    ylabel = L"$F=1$ peak position (mm)",
    xaxis=:log10,
    yaxis=:log10,
    labelfontsize=14,
    tickfontsize=12,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # xlims=(0.010,1.05),
    size=(900,800),
    # legendtitle=L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(1e3*gaussian_width_mm)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",
    legendfontsize=12,
    left_margin=3mm,
)
display(fig)
savefig(fig,joinpath(OUTDIR,"single_SG_comparison.png"))
savefig(fig,joinpath(OUTDIR,"single_SG_comparison.svg"))

# vs Gradient
gradvsI(x) = TheoreticalSimulation.GvsI(x)

fig=plot(    
    # title = L"Peak position ($F=1$)",
)
plot!(fig,
    gradvsI.(data_scaled[:,1]),data_scaled[:,3],
    xerr = gradvsI.(data_scaled[:,2]),
    yerr = data_scaled[:,4],
    label=L"Experiment (mag. $m = %$(round(global_mag_factor, digits=1))$)",
    seriestype=:scatter,
    marker = (:circle,4,:white,stroke(0.5,:black) )
    # line=(:dash,:darkgreen,3),
    # fillcolor = :darkgreen,
    # fillalpha = 0.35,
)
plot!(fig,gradvsI.(I_scan), zqm.(I_scan),
    label="Existing models",
    line=(:dash,:blue,1.75)
)
plot!(fig,
    gradvsI.(I_scan), ki_itp.(I_scan, Ref(fit_scaled.ki)),
    # label=L"Coquantum dynamics: $k_{i}= \left( %$(round(result.ki_fit, digits=4)) \pm %$(round(result.mse, sigdigits=1)) \right) \times 10^{-6} $",
    label=L"Coquantum dynamics: $k_{i}=  %$(round(fit_scaled.ki, digits=2)) \times 10^{-6} $",
    line=(:solid,:red,2),
    # marker=(:xcross, :blue, 0.2),
    markerstrokewidth=1
)
plot!(fig,
    xlabel = "Magnetic field gradient (T/m)",
    ylabel = L"$F=1$ peak position (mm)",
    xaxis=:log10,
    yaxis=:log10,
    labelfontsize=14,
    tickfontsize=12,
    # xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(5,400),
    size=(900,800),
    # legendtitle=L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(1e3*gaussian_width_mm)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",
    legend=:topleft,
    legendfontsize=12,
    left_margin=3mm,
)
display(fig)
savefig(fig,joinpath(OUTDIR,"single_SG_comparison_vsg.svg"))
savefig(fig,joinpath(OUTDIR,"single_SG_comparison_vsg.png"))



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


function goodness_of_fit(x, y, ypred; σ = nothing, k::Int = 0)
    @assert length(x) == length(y) == length(ypred)

    N = length(y)

    # --- residuals in log-space ---
    logy    = log.(y)
    logpred = log.(ypred)
    r       = logy .- logpred

    # --- core metrics in log-space ---
    logMSE  = mean(r .^ 2)
    logRMSE = sqrt(logMSE)
    R2_log  = 1 - sum(r .^ 2) / sum((logy .- mean(logy)) .^ 2)

    # --- robust scatter (NMAD) ---
    NMAD = 1.4826 * median(abs.(r))

    # --- χ², p-value, AIC, BIC ---
    if isnothing(σ)
        # No uncertainties: we cannot do a proper χ² test
        chi2_log = NaN
        chi2_red = NaN
        p_chi2   = NaN
        # Use logMSE as a surrogate for variance in "likelihood"
        AIC = 2k + N * log(logMSE)
        BIC = k * log(N) + N * log(logMSE)
    else
        @assert length(σ) == N
        # Propagate σ into log-space: σ_log ≈ σ / y
        σlog    = σ ./ y
        chi2_log = sum((r ./ σlog) .^ 2)
        dof      = max(N - k, 1)  # degrees of freedom
        chi2_red = chi2_log / dof

        # p-value: P(χ² >= observed χ² | dof)
        dist   = Chisq(dof)
        p_chi2 = ccdf(dist, chi2_log)  # 1 - cdf(dist, chi2_log)

        # AIC/BIC using χ² (Gaussian likelihood)
        AIC = 2k + chi2_log
        BIC = k * log(N) + chi2_log
    end

    return FitStats(logMSE, logRMSE, R2_log, chi2_log, chi2_red, p_chi2, AIC, BIC, NMAD)
end


x_exp = data[:,1]
y_exp = data[:,3] ./ scaled_mag
σ_exp = data[:,4] ./ scaled_mag
y_CQD = ki_itp.(x_exp, Ref(fit_scaled.ki))
y_QM  = zqm.(x_exp) 

stats_CQD = goodness_of_fit(x_exp, y_exp, y_CQD; σ = σ_exp, k = 2)
stats_QM  = goodness_of_fit(x_exp, y_exp, y_QM;  σ = σ_exp, k = 1)

metrics = [
    "logMSE",
    "logRMSE",
    "R2_log",
    "chi2_log",
    "chi2_red",
    "p_chi2",
    "AIC",
    "BIC",
    "NMAD",
]

data = [
    stats_CQD.logMSE   stats_QM.logMSE
    stats_CQD.logRMSE  stats_QM.logRMSE
    stats_CQD.R2_log   stats_QM.R2_log
    stats_CQD.chi2_log stats_QM.chi2_log
    stats_CQD.chi2_red stats_QM.chi2_red
    stats_CQD.p_chi2   stats_QM.p_chi2
    stats_CQD.AIC      stats_QM.AIC
    stats_CQD.BIC      stats_QM.BIC
    stats_CQD.NMAD     stats_QM.NMAD
]

lower_is_better  = Set(["logMSE", "logRMSE", "chi2_log", "chi2_red", "AIC", "BIC", "NMAD"])
higher_is_better = Set(["R2_log", "p_chi2"])

hl_best = TextHighlighter(
    (tbl, i, j) -> begin
        # Only evaluate columns 1 (CQD) and 2 (QM)
        if !(j == 1 || j == 2)
            return false
        end

        metric = metrics[i]   # row label from your vector
        v_CQD = tbl[i, 1]
        v_QM  = tbl[i, 2]

        # safety: both numeric
        if !(isa(v_CQD, Number) && isa(v_QM, Number))
            return false
        end

        if metric in lower_is_better
            best = min(v_CQD, v_QM)
            return tbl[i, j] == best

        elseif metric in higher_is_better
            best = max(v_CQD, v_QM)
            return tbl[i, j] == best
        end

        return false
    end,
    crayon"fg:black bg:#fff7a1"
);


pretty_table(
    data;
    column_labels = ["CQD", "QM"],
    row_labels    = metrics,
    row_label_column_alignment = :l,
    highlighters  = [hl_best],
    alignment     = [:c,:c],
    style         = TextTableStyle(
                first_line_column_label = crayon"yellow bold",
                table_border  = crayon"blue bold",
                column_label  = crayon"yellow bold",
                ),
    table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
    equal_data_column_widths= true,
)



function make_diagnostic_plots(x, y, y_CQD, y_QM, stats_CQD::FitStats, stats_QM::FitStats; σ = nothing)
"""
    make_diagnostic_plots(x, y, y_CQD, y_QM, stats_CQD, stats_QM; σ = nothing)

Create a set of diagnostic plots illustrating the goodness-of-fit metrics
for two models (CQD and QM) against experimental data.

Plots:
1. Data vs models in log-log space.
2. Log-space residuals vs x.
3. Histogram of log-space residuals with NMAD and logRMSE annotated.
4. Bar chart comparing key scalar metrics (logRMSE, R2_log, chi2_red, AIC, BIC, NMAD).

Returns a tuple of plots: (p_data, p_residuals, p_hist, p_bars).
"""
    # --- residuals in log-space ---
    logy     = log.(y)
    logCQD   = log.(y_CQD)
    logQM    = log.(y_QM)
    r_CQD    = logCQD .- logy
    r_QM     = logQM .- logy
    yerr = σ

    # ---------------------------------------------------
    # 1) Data vs models in log-log
    # ---------------------------------------------------
    p_data = plot(
        x, y;
        yerror = yerr, 
        seriestype = :scatter,
        marker=(:circle,:white,3,stroke(:black,0.8)),
        xscale = :log10,
        yscale = :log10,
        label = "Experiment",
        xlabel = "Coil Current (A)",
        ylabel = "Peak position (mm)",
        title = "Data vs Models (log-log space)",
        legend = :bottomright,
    )
    plot!(p_data, x, y_CQD; label="CQD model", line = (:solid,:red,1.5))
    plot!(p_data, x, y_QM;  label="QM model",  line =(:dot,:blue,2))

    # ---------------------------------------------------
    # 2) Log residuals vs x
    # ---------------------------------------------------
    p_resid = plot(
        x, r_CQD;
        seriestype = :scatter,
        marker = (:circle,5,0.70,:salmon3, stroke(0.8,:red4)),
        xlabel = "Coil Current (A)",
        ylabel = L"\mathrm{log}(y_{model}) - \mathrm{log}(y_{exp})",
        title = "Log-space Residuals",
        label = "CQD residuals",
        xscale = :log10,
    )
    scatter!(p_resid, x, r_QM; label="QM residuals",
        marker = (:circle,5,0.70,:royalblue3, stroke(0.8,:blue4)),
    )
    hline!(p_resid, [0.0]; c=:black, ls=:dash, label="perfect fit")
    # Annotate with global metrics
    txt_CQD = @sprintf "CQD: logRMSE = %.3g, R2_log = %.4f" stats_CQD.logRMSE stats_CQD.R2_log
    txt_QM  = @sprintf "QM:  logRMSE = %.3g, R2_log = %.4f" stats_QM.logRMSE  stats_QM.R2_log
    x_annot = x[argmin(abs.(x .- median(x)))]  # roughly middle x
    ymin, ymax = extrema(vcat(r_CQD, r_QM))
    annotate!(p_resid, (x_annot, 0.8*ymax, Plots.text(txt_CQD, 8)))
    annotate!(p_resid, (x_annot, 0.8*ymax - 0.1*(ymax-ymin), Plots.text(txt_QM, 8)))

    # ---------------------------------------------------
    # 3) Histogram of residuals with NMAD / logRMSE
    # ---------------------------------------------------
    p_hist = histogram(
        r_CQD;
        normalize = true,
        color=:red,
        alpha = 0.4,
        label = "CQD residuals",
        xlabel = "log-space residual r",
        ylabel = "Normalized count",
        title = "Distribution of log-space residuals",
    )
    histogram!(p_hist, r_QM; 
        normalize = true, 
        color=:blue,
        alpha = 0.4, 
        label="QM residuals")
    vline!(p_hist, [0.0]; c=:black, ls=:dash, lw=1, label="r = 0")
    # annotate NMAD and logRMSE
    txt2_CQD = @sprintf "CQD: NMAD = %.3g, logRMSE = %.3g" stats_CQD.NMAD stats_CQD.logRMSE
    txt2_QM  = @sprintf "QM:  NMAD = %.3g, logRMSE = %.3g" stats_QM.NMAD  stats_QM.logRMSE
    x_hist_min, x_hist_max = extrema(vcat(r_CQD, r_QM))
    y_hist_max = Plots.ylims(p_hist)[2]
    annotate!(p_hist, (-0.6x_hist_min, 0.7y_hist_max, Plots.text(txt2_CQD, 8, :left)))
    annotate!(p_hist, (-0.6x_hist_min, 0.6y_hist_max, Plots.text(txt2_QM, 8, :left)))

    return p_data, p_resid, p_hist
end

p1, p2, p3 = make_diagnostic_plots(x_exp, y_exp, y_CQD, y_QM, stats_CQD, stats_QM; σ=σ_exp)
plot(p1, p2, p3; 
    layout = (2, 2), 
    size = (1000, 1000),
    left_margin=3mm,
)