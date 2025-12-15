# Simulation of atom trajectories in the Stern–Gerlach experiment
# Interpolation of the grid for fitting the induction term
# Kelvin Titimbo
# California Institute of Technology
# November 2025

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
# using Interpolations #, Roots, Loess, Optim
# using BSplineKit
using Dierckx, Optim
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
LinearAlgebra.BLAS.set_num_threads(4)
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

function fit_ki(data_org, selected_points, ki_list, ki_range)
    """
        fit_ki(data_org, selected_points, ki_list, ki_range)

    Fit the induction coefficient `kᵢ` by minimizing the mean squared error in log10-space
    between the interpolated prediction `ki_itp(x, kᵢ)` and the selected fitting points.

    Arguments
    ---------
    - `data_org`        : full data set as a 2-column array (x, y) used to compute R²
    - `selected_points` : subset of points (x, y) used for the fit
    - `ki_list`         : vector of candidate `kᵢ` values defining the search interval
    - `ki_range`        : tuple `(ki_start, ki_stop)` with indices into `ki_list`

    Returns
    -------
    - `k_fit`   : best-fit value of `kᵢ`
    - `mse`     : mean squared error (in log10-space) on `selected_points`
    - `coef_r2` : coefficient of determination R² on `data_org`
    """
    ki_start, ki_stop = ki_range

    # loss function (uses only the fitting subset)
    loss_scaled(ki) = begin
        z_pred = ki_itp.(selected_points[:, 1], Ref(ki))
        mean(abs2, log10.(z_pred) .- log10.(selected_points[:, 2]))
    end

    # 1D optimization over ki
    fit_param = optimize(loss_scaled,
                         ki_list[ki_start], ki_list[ki_stop],
                         Brent())

    k_fit = Optim.minimizer(fit_param)

    # diagnostics on the fitting subset
    mse = loss_scaled(k_fit)

    # predictions for the full data set
    pred = ki_itp.(data_org[:, 1], Ref(k_fit))
    y    = data_org[:, 2]
    coef_r2 = 1 - sum(abs2, pred .- y) / sum(abs2, y .- mean(y))

    return (ki = k_fit, ki_err = mse, r2_coeff = coef_r2)
end

function fit_ki_with_error(itp, data;
                           bounds::Tuple{<:Real,<:Real},
                           conf::Real=0.95,
                           weights::Union{Nothing,AbstractVector}=nothing,
                           h::Union{Nothing,Real}=nothing)
    """
        fit_ki_with_error(itp, data; bounds, conf=0.95, weights=nothing, h=nothing)

    Fit a single parameter `ki` by minimizing the mean squared error of
    `log10.(itp.(I, ki))` vs `log10.(Z)`, then estimate the standard error of `ki`
    from the (finite-difference) Jacobian.

    Arguments
    - `itp`    : callable s.t. `itp(Ic, ki) -> predicted z` (broadcastable)
    - `data`   : 2-column table/array with columns `[I, Z]` (currents, measured peaks)
    - `bounds` : `(ki_min, ki_max)` search interval (required)

    Keywords
    - `conf`     : confidence level for the interval (default 0.95)
    - `weights`  : optional weights for each point (Vector; applied in log-space)
    - `h`        : finite-difference step for ∂r/∂ki. Default scales with `k̂`.

    Returns
    NamedTuple:
    `(k_hat, se, ci, rss, sigma2, dof, n_used, converged, result)`
    """
    # --- unpack data ---
    I = collect(Float64, data[:, 1])
    Z = collect(Float64, data[:, 3])

    # drop nonpositive / nonfinite (log10 requires > 0)
    mask0 = isfinite.(I) .& isfinite.(Z) .& (Z .> 0)
    I, Z = I[mask0], Z[mask0]

    # optional weights (in log-space); broadcast & subset
    w = weights === nothing ? ones(length(I)) : collect(weights)[mask0]
    @assert length(w) == length(I) "weights length must match number of valid points"

    # loss in log-space (weighted MSE)
    function loss(ki)
        zpred = itp.(I, Ref(ki))
        r = log10.(zpred) .- log10.(Z)
        m = isfinite.(r)
        r = r[m]; ww = w[m]
        return mean(ww .* (r .^ 2))
    end

    # --- 1D bounded minimization ---
    ki_min, ki_max = float(bounds[1]), float(bounds[2])
    res = optimize(loss, ki_min, ki_max, Brent())
    k̂  = Optim.minimizer(res)

    # residuals at k̂
    ẑ   = itp.(I, Ref(k̂))
    r    = log10.(ẑ) .- log10.(Z)
    mres = isfinite.(r)
    r, I, Z, w = r[mres], I[mres], Z[mres], w[mres]

    n = length(r); p = 1
    @assert n > p "Not enough valid points to estimate uncertainty"

    # finite-difference Jacobian dr/dk at k̂
    # choose a numerically sensible central-diff step for k
    # uses relative step ≈ cbrt(eps) * |k| with a tiny absolute floor,
    # and shrinks if we're near the bounds.
    fd_step(k̂, lo, hi; rel=cbrt(eps(Float64)), absmin=1e-12) = begin
        h = max(absmin, rel * abs(k̂))             # ~6e-6 * |k̂|, floored at 1e-12
        if isfinite(lo) && isfinite(hi)
            room = min(k̂ - lo, hi - k̂)
            h = room > 0 ? min(h, 0.5 * room) : absmin
        end
        h
    end
    h₀ = isnothing(h) ? fd_step(k̂, ki_min, ki_max) : float(h)
    z⁺   = itp.(I, Ref(k̂ + h₀))
    z⁻   = itp.(I, Ref(k̂ - h₀))
    r⁺   = log10.(z⁺) .- log10.(Z)
    r⁻   = log10.(z⁻) .- log10.(Z)

    mJ   = isfinite.(r⁺) .& isfinite.(r⁻)
    r, w = r[mJ], w[mJ]
    drdk = (r⁺[mJ] .- r⁻[mJ]) ./ (2h₀)

    n_used = length(r)
    @assert n_used > p "Not enough valid points after derivative filtering"

    # weighted LS variance and SE(k)
    RSS   = sum(w .* (r .^ 2))
    dof   = n_used - p
    σ²    = RSS / dof
    SJJ   = sum((sqrt.(w) .* drdk) .^ 2)   # J'J for scalar param (weighted)
    se    = sqrt(σ² / SJJ)

    # confidence interval
    tcrit = quantile(TDist(dof), 0.5 + conf/2)
    ci    = (k̂ - tcrit*se, k̂ + tcrit*se)

    return (
        k_hat=k̂, 
        k_err=tcrit*se, 
        se=se, 
        ci=ci, 
        rss=RSS, 
        sigma2=σ², 
        dof=dof,
        n_used=n_used, 
        converged=Optim.converged(res), 
        result=res)
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

# --- helper ---
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

function plot_cqd_vs_qm(zmm_cqd, zmax_QM, Icoils_cqd, ki_list;
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
        idx2 = idx > 0 ? idx : length(Icoils_cqd) + idx + 1  # allow negatives
        plot!(figa, zmm_cqd[idx2, :],
            label = "CQD $(1000*Icoils_cqd[idx2]) mA",
            line = (:solid, cls[j], 2))
        hline!(figa, [zmax_QM[idx2]],
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
        idx2 = idx > 0 ? idx : length(Icoils_cqd) + idx + 1
        plot!(figb, zmm_cqd[idx2, :],
            label = "CQD $(1000*Icoils_cqd[idx2]) mA",
            line = (:solid, cls[j], 2))
        hline!(figb, [zmax_QM[idx2]],
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

# Simulated currents
Icoils_cqd = [0.00,
            0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
            0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,
            0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
];
nI_sim = length(Icoils_cqd);

nruns= 10;
induction_coeff = 1e-6 * [
    # 20251113T102859450
    range(0.01,0.10,length=nruns),
    range(0.1,1.0,length=nruns),
    range(1.1,2.0,length=nruns),
    range(2.1,3.0,length=nruns),
    range(3.1,4.0,length=nruns),
    range(4.1,5.0,length=nruns),
    range(5.1,6.0,length=nruns),
    range(10,100,length=nruns),
    # 20251116T164054691
    [0.001,0.01],
    # 20251117T200727431
    range(6.1,10.0,length=4*nruns),
    [100.0,1000.0,10000.0]

]
ki_list = round.(1e6*vcat(induction_coeff...), sigdigits=2)
groups = Dict{Any, Vector{Int}}();
for (i, v) in pairs(ki_list)
    push!(get!(groups, v, Int[]), i)
end
filter!(kv -> length(kv[2]) > 1, groups);
to_remove = sort!(vcat([idxs[2:end] for idxs in values(groups)]...), rev=true)
for i in to_remove
    deleteat!(ki_list, i)
end
sort!(ki_list);
n_ki = length(ki_list)

# --- CoQuantum Dynamics ---
table_cqd =load(joinpath(@__DIR__,"simulation_data","cqd_simulation_2.8m","cqd_2800000_screen_profiles_table_thread.jld2"),"table");
@info "CQD data loaded"
keys_vec = collect(keys(table_cqd)) ; # Vector of tuples
ki_set  = sort(unique(first.(keys_vec)))
nz_set  = sort(unique(getindex.(keys_vec, 2)))
gw_set  = sort(unique(getindex.(keys_vec, 3)))
λ0_set  = sort(unique(getindex.(keys_vec, 4)))
# --- Quantum mechanics data ---
table_qm   = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_3m","qm_3000000_screen_profiles_table.jld2"))["table"];
@info "QM data loaded"
# --- Experiment combined ---
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
)
# xq and δxq from grouped data
xq  = exp_avg[:Ic_grouped][:,1]
δxq = exp_avg[:Ic_grouped][:,2]

# 2. Spline derivative at xq
dyq = derivative(data_experiment, xq; nu=1)

# 3. Interpolate δy uncertainty to xq
err_spline = Spline1D(exp_avg[:i_smooth], exp_avg[:δz_smooth], k=3, bc="extrapolate")
δy_interp = err_spline.(xq)   # now same length as xq

# 4. Total propagated uncertainty
δyq = sqrt.( (dyq .* δxq).^2 .+ δy_interp.^2 )

# --------------------------------

nx_bins , nz_bins = 128 , 2
gaussian_width_mm = 0.200
λ0_raw            = 0.01
λ0_spline         = 0.001

# ------------------------------
data_qm = table_qm[(nz_bins,gaussian_width_mm,λ0_raw)];
Ic_QM   = [data_qm[i][:Icoil] for i in eachindex(data_qm)];
zmax_QM = [data_qm[i][:z_max_smooth_spline_mm] for i in eachindex(data_qm)];
zqm = Spline1D(Ic_QM,zmax_QM,k=3);
# ------------------------------

z_mm_ki = Matrix{Float64}(undef, nI_sim, length(ki_list));
for (i,ki) in enumerate(ki_list)
    println("\t($(@sprintf("%02d", i))/$(length(ki_list))) Running ki=$(@sprintf("%2.1e",1e-6*ki))")
    cqd_data = table_cqd[(ki,nz_bins,gaussian_width_mm,λ0_raw)]
    I_inst   = [cqd_data[j][:Icoil] for j=1:length(cqd_data)]    
    nI_inst  = length(I_inst)

        # ✅ Sanity check: ensure Icoil matches the rsimulated
    if nI_sim != nI_inst || !isapprox(I_inst, Icoils_cqd; atol=1e-8)
        @warn "Icoil vector differs in run $j!"
    end

    z_mm_ki[:,i] = [cqd_data[v][:z_max_smooth_spline_mm] for v in 1:nI_inst]
end

color_list = palette(:darkrainbow, n_ki);
fig = plot(xlabel="Current (A)",
    ylabel=L"$z_{\mathrm{max}}$ (mm)",
    );
for (i,ki) in enumerate(ki_list)
        plot!(fig,Icoils_cqd[2:end], z_mm_ki[2:end,i],
            label = L"$k_{i}=%$(round(ki, sigdigits=2))\times 10^{-6}$",
            line=(:solid,color_list[i]),
        )
end
plot!(Ic_QM[2:end],zmax_QM[2:end],
    label="QM",
    line=(:dashdot,:black,2));
plot!(fig, exp_avg[:i_smooth], exp_avg[:z_smooth],
    ribbon=exp_avg[:δz_smooth],
    color=:gold,
    label="Combined experiments",
    line=(:solid,:gold,3),
    fillalpha=0.3,);
annotate!(fig, 1e-2,1, text(L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(1e3*gaussian_width_mm)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",:black,16));
plot!(fig, 
    size=(1250,600),
    xaxis=:log10, 
    yaxis=:log10,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    # legend_title = L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(1e3*gaussian_width_mm)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",
    legendtitlefontsize = 8,
    legend_columns = 2,
    legendfontsize=6,
    left_margin=6mm,
    bottom_margin=5mm);
display(fig)

fig = plot_cqd_vs_qm(z_mm_ki, zmax_QM, Icoils_cqd, ki_list);
display(fig)


# Interpolated kᵢ surface
ki_start , ki_stop = 1 , 119 #length(ki_sim)
println("Interpolation in the induction term goes from $(ki_list[ki_start])×10⁻⁶ to $(ki_list[ki_stop])×10⁻⁶")
ki_itp = Spline2D(Icoils_cqd, ki_list[ki_start:ki_stop], z_mm_ki[:,ki_start:ki_stop]; kx=3, ky=3, s=0.00);

i_surface = range(10e-3,1.0; length = 101);
ki_surface = range(ki_list[ki_start],ki_list[ki_stop]; length = 101);
Z = [ki_itp(x, y) for y in ki_surface, x in i_surface] ;

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
);

Zp   = max.(abs.(Z), 1e-12);      # guard against zeros
logZ = log10.(Zp);
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
# --- Analysis : Combined experimental data --- 
i_threshold = 0.025;
i_start  = searchsortedfirst(exp_avg[:i_smooth],i_threshold);
data     = hcat(exp_avg[:i_smooth],exp_avg[:z_smooth],exp_avg[:δz_smooth])[i_start:end,:];

fig=plot(data[:,1], zqm.(data[:,1]),
    label="Quantum mechanics",
    line=(:solid,:red,1.75)
);
plot!(data[1:2:end,1],data[1:2:end,2],
    color=:gray35,
    marker=(:circle,:gray35,1),
    markerstrokecolor=:gray35,
    markerstrokewidth=1,
    # ribbon = data[:,3],
    label="Combined data",
);
# --- Compute scaling factor ---
n=50
yexp = last(data[:, 2], n)
ythe = last(zqm.(data[:, 1]), n)
# Scaled magnification
scaled_mag = dot(yexp, yexp) / dot(yexp, ythe)
data_scaled = copy(data)
data_scaled[:, 2] ./= scaled_mag
data_scaled[:, 3] ./= scaled_mag
plot!(data_scaled[:,1],data_scaled[:,2],
    ribbon = data_scaled[:,3],
    label=L"Combined data (scaled $m_{p} = %$(round(scaled_mag, digits=4))$ )",
    line=(:dash,:darkgreen,1.2),
    fillcolor = :darkgreen,
    fillalpha = 0.35,
);
data_fitting = data[[1:6; (end-5):(end)], :]
data_scaled_fitting = data_scaled[[1:6; (end-5):(end)], :]
fit_original = fit_ki(data, data_fitting, ki_list, (ki_start,ki_stop))
fit_scaled   = fit_ki(data_scaled, data_scaled_fitting, ki_list, (ki_start,ki_stop))
I_scan = logspace10(i_threshold,1.00; n=101);
plot!(
    I_scan, ki_itp.(I_scan, Ref(fit_original.ki)),
    label=L"Original : $k_{i}= \left( %$(round(fit_original.ki, digits=4)) \pm %$(round(fit_original.ki_err, digits=4)) \right) \times 10^{-6} $",
    line=(:solid,:blue,2),
    marker=(:xcross, :blue, 0.2),
    markerstrokewidth=1
);
plot!(
    I_scan, ki_itp.(I_scan, Ref(fit_scaled.ki)),
    label=L"Scaled: $k_{i}= \left( %$(round(fit_scaled.ki, digits=4)) \pm %$(round(fit_scaled.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
    line=(:solid,:lawngreen,2),
    marker=(:xcross, :lawngreen, 0.2),
    markerstrokewidth=1
);
plot!(
    xlabel = "Current (A)",
    ylabel = L"$z_{\mathrm{max}}$ (mm)",
    xaxis=:log10,
    yaxis=:log10,
    labelfontsize=14,
    tickfontsize=12,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # xlims=(0.010,1.05),
    size=(900,800),
    legendtitle=L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(1e3*gaussian_width_mm)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",
    legendfontsize=12,
    left_margin=3mm,
);
display(fig)

compare_datasets(data_scaled[:,1], data_scaled[:,2], ki_itp.(data_scaled[:,1], Ref(fit_scaled.ki)), zqm.(data_scaled[:,1]); plot_errors=true);
fig1 = plot(data_scaled[:,1], (zqm.(data_scaled[:,1]) .- data_scaled[:,2]) ./ data_scaled[:,2],
    label="QM",
    line=(:solid,:red,2))
plot!(data_scaled[:,1], (ki_itp.(data_scaled[:,1], Ref(fit_scaled.ki)) .- data_scaled[:,2]) ./ data_scaled[:,2],
    label=L"CQD ($k_{i}=%$(round(fit_scaled.ki,sigdigits=4)) \times10^{-6}$)",
    line=(:solid,:blue,2))
plot!(
    title="Relative Error - Scaled data",
    titlefontsize=24,
    xlabel = "Current (A)",
    ylabel = L"$z_{\mathrm{max}}$ (mm)",
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

compare_datasets(data[:,1], data[:,2], ki_itp.(data[:,1], Ref(fit_original.ki)), zqm.(data[:,1]); plot_errors=true);
fig2=plot(data[:,1], ((zqm.(data[:,1]) .- data[:,2]) ./ data[:,2]) ,
    label="QM",
    line=(:solid,:red,2))
plot!(data[:,1], (ki_itp.(data[:,1], Ref(fit_original.ki)) .- data[:,2]) ./ data[:,2],
    label=L"CQD ($k_{i}=%$(round(fit_original.ki,sigdigits=4)) \times10^{-6}$)",
    line=(:solid,:blue,2))
plot!(
    title="Relative Error - Original data",
    titlefontsize=24,
    xlabel = "Current (A)",
    ylabel = L"$z_{\mathrm{max}}$ (mm)",
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

fig= plot(fig2,fig1,
    layout=(2,1),
    size=(1000,600))

df = DataFrame(hcat(data_scaled[:,1],
    (zqm.(data_scaled[:,1]) .- data_scaled[:,2]) ./ data_scaled[:,2],
    (ki_itp.(data_scaled[:,1], Ref(fit_scaled.ki)) .- data_scaled[:,2]) ./ data_scaled[:,2]
),
[:Ic, :eQM, :eCQD]
)
CSV.write(joinpath(OUTDIR,"rel_error_scaled.csv"),df)
df = DataFrame(hcat(data[:,1],
    (zqm.(data[:,1]) .- data[:,2]) ./ data[:,2],
    (ki_itp.(data[:,1], Ref(fit_original.ki)) .- data[:,2]) ./ data[:,2]
),
[:Ic, :eQM, :eCQD]
)
CSV.write(joinpath(OUTDIR,"rel_error_original.csv"),df)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

function plot_zmax_vs_current(
        data_exp,
        ki_list;
        Icoils_cqd,
        zmm_cqd,
        Ic_QM,
        zmax_QM,
        data_label = "experiment",
        p = 1.0,
        scale_exp = false,
        axis_scale = :loglog,   # :linear, :loglog, :semilogx, :semilogy
        figsize = (850,600)
    )

    # Colors
    color_list = palette(:darkrainbow, length(ki_list))

    # --- Figure setup ---
    fig = plot(
        xlabel = "Currents (A)",
        ylabel = L"$z_{\mathrm{max}}$ (mm)",
        legend = :outerright,
    )

    # --- CQD curves ---
    for i in eachindex(ki_list)
        plot!(
            fig,
            Icoils_cqd[2:end],
            zmm_cqd[2:end, i],
            label = L"$k_{i}=%$(round(ki_list[i], sigdigits=3))\times 10^{-6}$",
            line = (:solid, color_list[i]),
        )
    end

    # --- Experimental data ---
    z_exp   = scale_exp ? data_exp[!, :z] ./ p   : data_exp[!, :z]
    dz_exp  = scale_exp ? data_exp[!, :δz] ./ p  : data_exp[!, :δz]

    plot!(
        fig,
        data_exp[!, :Ic],
        z_exp,
        ribbon = dz_exp,
        line = (:black, :dash, 2),
        fillalpha = 0.25,
        fillcolor = :gray13,
        label = data_label,
    )

    # --- QM curve ---
    plot!(
        fig,
        Ic_QM[2:end],
        zmax_QM[2:end],
        label = "QM",
        line = (:dashdot, :magenta, 3),
    )

    # --- Axis scaling ---
    xscale = :identity
    yscale = :identity

    if axis_scale == :loglog
        xscale = :log10
        yscale = :log10
    elseif axis_scale == :semilogx
        xscale = :log10
    elseif axis_scale == :semilogy
        yscale = :log10
    end

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
    Compute scaling factor p from the last n points and plot:
      - Experimental data
      - QM theory curve
      - Scaled experimental curve (data_exp/p)

    Inputs:
        n               :: Int
        data_exp        :: DataFrame (must contain :Ic, :z)
        wanted_data_dir :: String (label for experiment)
        mag             :: Float (original magnification)
        zqm             :: Function Ic -> z_qm(Ic)

    Outputs:
        p :: Float
        fig :: Plot
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
        k_fit, mse;
        wanted_data_dir,
        wanted_binning,
        wanted_smooth,
        ki_itp,
        out,
        Ic_QM,
        zmax_QM,
    )

    # --- Scale the data using p ---
    data_fitting_scaled = copy(data_fitting)
    data_fitting_scaled[:, end-1:end] ./= p   # scale last two columns

    I_scan = logspace10(10e-3, 1.00; n=30);
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
    plot!(
        fig,
        I_scan,
        ki_itp.(I_scan, Ref(out.ci[2])),
        color = :royalblue1,
        label = false,
        linewidth = 0,
        fillrange = ki_itp.(I_scan, Ref(out.ci[1])),
        fillcolor = :royalblue1,
        fillalpha = 0.35,
    )

    # --- Best-fit curve (unscaled) ---
    plot!(
        fig,
        I_scan,
        ki_itp.(I_scan, Ref(out.k_hat)),
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
        I_scan,
        ki_itp.(I_scan, Ref(k_fit)),
        label = L"$k_{i}= \left( %$(round(k_fit, digits=4)) \pm %$(round(mse, digits=4)) \right) \times 10^{-6} $",
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
    plot!(
        fig,
        Ic_QM[2:end], zmax_QM[2:end],
        label = "QM",
        line = (:dashdot, :black, 2),
    )

    return fig
end

# ------------------------------------------------------------------------------
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

    # Data loading
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

    # Load framewise data
    load_data   = CSV.read(joinpath(dirname(res.path), "fw_data.csv"), DataFrame; header = true)
    I_exp       = load_data[!, "Icoil_A"]
    I_exp_error = load_data[!, "Icoil_error_A"]
    z_exp       = load_data[!, "F1_z_centroid_mm"] / mag
    z_exp_error = abs.(z_exp) .* sqrt.(
        (load_data[!, "F1_z_centroid_se_mm"] ./ load_data[!, "F1_z_centroid_mm"]).^2 .+
        (δmag / mag)^2
    )

    i_start  = searchsortedfirst(I_exp, current_threshold)
    data     = hcat(I_exp, I_exp_error, z_exp, z_exp_error)[i_start:end, :]
    data_exp = DataFrame(data, [:Ic, :δIc, :z, :δz])

    # Plot 1: raw magnification
    fig = plot_zmax_vs_current(
        data_exp, ki_list;
        Icoils_cqd = Icoils_cqd,
        zmm_cqd    = z_mm_ki,
        Ic_QM      = Ic_QM,
        zmax_QM    = zmax_QM,
        axis_scale = :loglog,
        data_label = wanted_data_dir,
    )
    display(fig)

    # Scaling factor fit
    p, fig = plot_scaling_factor(
        2,
        data_exp,
        wanted_data_dir,
        mag;
        zqm = zqm,
    )
    display(fig)
    println("Scaled Magnification factor 𝓂 = $(@sprintf("%2.4f", mag * p))")

    # Plot 2: scaled experimental data
    fig = plot_zmax_vs_current(
        data_exp, ki_list;
        Icoils_cqd = Icoils_cqd,
        zmm_cqd    = z_mm_ki,
        Ic_QM      = Ic_QM,
        zmax_QM    = zmax_QM,
        p          = p,
        scale_exp  = true,
        data_label = wanted_data_dir,
        axis_scale = :loglog,
    )
    display(fig)

    # Choose fitting subset (first 4 and last 4 points)
    data_fitting = data[[1:4; (end-3):end], :]

    # Fit ki using rescaled data
    k_fit, mse, r2 = fit_k_parameter(
        data_fitting,
        p,
        ki_list,
        ki_start,
        ki_stop;
        ki_itp = ki_itp,
        I_exp  = I_exp,
        z_exp  = z_exp,
    )
    @info @info "Fitting for rescaled data (𝓂 = $(p*mag))" "kᵢ\t\t" = k_fit "Err kᵢ\t" = mse "R²\t\t" = r2

    # Fit ki with error estimation (unscaled magnification)
    out = fit_ki_with_error(
        ki_itp,
        data_fitting;
        bounds = (ki_list[ki_start], ki_list[ki_stop]),
    )
    @info "Fitting (𝓂 = $(mag))" "kᵢ\t\t" = out.k_hat "Err kᵢ\t" = out.k_err "kᵢ interval\t" = out.ci

    # Full ki-fit plot
    fig = plot_full_ki_fit(
        data_exp, data_fitting,
        mag, p, k_fit, mse;
        wanted_data_dir = wanted_data_dir,
        wanted_binning  = wanted_binning,
        wanted_smooth   = wanted_smooth,
        ki_itp          = ki_itp,
        out             = out,
        Ic_QM           = Ic_QM,
        zmax_QM         = zmax_QM,
    )
    display(fig)

    println("Finished processing dataset $wanted_data_dir\n" * "-"^60 * "\n")
end
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

"""
Fit the CQD parameter `ki` while dynamically rescaling the experimental data
using a high-current tail region and a combined QM+CQD reference.

Inputs
------
data      :: Matrix{<:Real}
    Nx3 array with columns:
        1. Iexp  - experimental current (A)
        2. yexp  - experimental z_max (mm)
        3. (ignored here, usually y-error)

zqm       :: callable
    Interpolated QM model: zqm(I) → z_QM(I)

ki_itp    :: callable
    Interpolated CQD model: ki_itp(I, ki) → z_CQD(I; ki)

thresholdI :: Float64
    Current threshold (A) defining the "tail" region used for scaling.
    Only points with Iexp ≥ thresholdI contribute to the scale factor.

ki_range :: Tuple{<:Real,<:Real}
    (kmin, kmax) bracket for the 1D Brent optimizer over ki.

Keyword arguments
-----------------
n_front :: Int = 6
    Number of lowest-current points used in the ki fit.

n_back  :: Int = 6
    Number of highest-current points used in the ki fit.

w       :: Float64 = 0.7
    Weight for the QM model in the combined reference:
        ref = w * QM + (1-w) * CQD   (arithmetic)
        ref = QM^w * CQD^(1-w)       (geometric)

ref_type :: Symbol = :arith
    Type of combination for the tail reference model:
        :arith  → arithmetic blend  z_ref = w*zQM + (1-w)*zCQD
        :geom   → geometric blend   z_ref = zQM^w * zCQD^(1-w)

Method summary
--------------
1. Select a tail region (I ≥ thresholdI) used ONLY to determine a global
   scale factor s(ki), which maps experimental data to theory.
2. Define a reference tail curve z_ref_tail(ki) as an arithmetic or geometric
   combination of QM and CQD in that tail region.
3. For each trial ki, compute the least-squares projection scale:
       s(ki) = <y_tail, z_ref_tail> / <z_ref_tail, z_ref_tail>
4. Scale the *entire* experimental dataset by 1/s(ki).
5. Fit ki by minimizing the mean squared log-residual between the scaled
   experiment and the CQD model, but using only a subset of points:
       - the first n_front points (low current)
       - the last  n_back  points (high current).
6. Return the best-fit ki, the final scale factor s(ki_fit), and diagnostics.
"""
function fit_ki_joint_scaling_fitsubset(
    data,
    zqm,
    ki_itp,
    thresholdI::Float64,
    ki_range;
    n_front::Int = 6,
    n_back::Int  = 6,
    w::Float64   = 0.7,
    ref_type::Symbol = :arith,   # :arith or :geom)

    # Unpack experimental current and z_max
    Iexp = data[:,1]
    yexp = data[:,3]

    N = length(Iexp)

    # ------------------------------
    # 1) Tail region (for scaling)
    # ------------------------------
    # Use only points with Iexp ≥ thresholdI to compute the scale factor.
    tail_idx = findall(Iexp .>= thresholdI)
    if isempty(tail_idx)
        error("No experimental points with current ≥ $thresholdI A")
    end

    I_tail  = Iexp[tail_idx]
    y_tail  = yexp[tail_idx]

    # -----------------------------------------
    # 2) Fitting region (for ki optimisation)
    # -----------------------------------------
    # Only use a subset of points for the ki fit:
    # - first n_front points (low current)
    # - last  n_back points (high current)
    fit_idx = vcat(1:n_front, (N-n_back+1):N)

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
        zqm_tail  = zqm.(I_tail)         # QM prediction in tail
        zcqd_tail = ki_itp.(I_tail, ki)  # CQD prediction in tail

        if ref_type == :arith
            # Arithmetic blend in linear space
            return w .* zqm_tail .+ (1 - w) .* zcqd_tail

        elseif ref_type == :geom
            # Geometric blend (equivalent to weighted average in log-space)
            # z_ref = zQM^w * zCQD^(1-w)
            return zqm_tail .^ w .* zcqd_tail .^ (1 - w)

        else
            error("Invalid ref_type = $ref_type. Use :arith or :geom.")
        end
    end

    # ------------------------------------------------
    # 4) Loss function over ki
    # ------------------------------------------------
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
    function loss(ki)
        # CQD model evaluated at all experimental currents
        y_cqd = ki_itp.(Iexp, ki)

        # Tail reference model for this ki
        zref_tail = zref_tail_for(ki)

        # Scale factor: least-squares projection of y_tail onto zref_tail
        scale = dot(y_tail, zref_tail) / dot(zref_tail, zref_tail)

        # Apply global scaling to experimental data
        yexp_scaled = yexp ./ scale

        # Log-space residuals restricted to fitting subset
        r = log.(yexp_scaled[fit_idx]) .- log.(y_cqd[fit_idx])

        # Mean squared residuals (this is the objective being minimized)
        return mean(abs2, r)
    end

    # ------------------------------------------------
    # 5) 1D optimisation over ki (Brent)
    # ------------------------------------------------
    kmin, kmax = ki_range
    opt = optimize(loss, kmin, kmax, Brent())

    ki_fit = Optim.minimizer(opt)   # best-fit ki
    mse    = Optim.minimum(opt)     # minimum loss value (mean squared log-residual)

    # ------------------------------------------------
    # 6) Final scale factor at ki_fit (for output)
    # ------------------------------------------------
    # Recompute the reference tail with the best-fit ki and get the final
    # scale factor that defines the global rescaling of the experiment.
    zref_tail_final = zref_tail_for(ki_fit)
    scale_final = dot(y_tail, zref_tail_final) / dot(zref_tail_final, zref_tail_final)

    # ------------------------------------------------
    # 7) Return results and configuration parameters
    # ------------------------------------------------
    return (
        ki_fit       = ki_fit,        # best-fit ki
        scale_factor = scale_final,   # global magnification correction
        mse          = mse,           # mean squared log-residual at optimum
    )
end

i_threshold = 0.025;
i_start  = searchsortedfirst(exp_avg[:i_smooth],i_threshold);
data     = hcat(exp_avg[:i_smooth],0.02*exp_avg[:i_smooth],exp_avg[:z_smooth],exp_avg[:δz_smooth])[i_start:end,:];
result = fit_ki_joint_scaling_fitsubset(
    data,
    zqm,
    ki_itp,
    0.750,                                  # tail threshold
    (ki_list[ki_start], ki_list[ki_stop]); # bracket
    n_front = 7*2,
    n_back  = 6*2,
    w       = 0.50,
    ref_type=:geom,
    # ref_type=:arith,
)


fig=plot(    
    title = L"Peak position ($F=1$)",)
# Scaled magnification
scaled_mag = result.scale_factor
data_scaled = copy(data)
data_scaled[:, 3] ./= scaled_mag
data_scaled[:, 4] ./= scaled_mag
global_mag_factor = scaled_mag*MyExperimentalAnalysis.mag_factor("20250814")[1]
plot!(fig,
    data_scaled[:,1],data_scaled[:,3],
    ribbon = data_scaled[:,4],
    label=L"Experimental data (magnif.factor $m = %$(round(global_mag_factor, digits=4))$)",
    line=(:dash,:darkgreen,3),
    fillcolor = :darkgreen,
    fillalpha = 0.35,
)
data_fitting = data[[1:6; (end-5):(end)], :]
data_scaled_fitting = data_scaled[[1:6; (end-5):(end)], :]
fit_scaled   = fit_ki(data_scaled[:,[1,3]], data_scaled_fitting[:,[1,3]], ki_list, (ki_start,ki_stop))
I_scan = logspace10(i_threshold,1.00; n=101);
plot!(fig,I_scan, zqm.(I_scan),
    label="Quantum mechanical model",
    line=(:solid,:red,1.75)
)
plot!(fig,
    I_scan, ki_itp.(I_scan, Ref(fit_scaled.ki)),
    label=L"CoQuantum dynamics: $k_{i}= \left( %$(round(result.ki_fit, digits=4)) \pm %$(round(result.mse, sigdigits=1)) \right) \times 10^{-6} $",
    line=(:dot,:blue,2),
    # marker=(:xcross, :blue, 0.2),
    markerstrokewidth=1
)
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


# Scattered data
i_threshold = 0.025;
i_start  = searchsortedfirst(xq,i_threshold)
data     = hcat(xq, δxq, data_experiment.(xq), δyq)[i_start:end,:]

fig=plot(    
    # title = L"Peak position ($F=1$)",
)
# Scaled magnification
scaled_mag = result.scale_factor
data_scaled = copy(data)
data_scaled[:, 3] ./= scaled_mag
data_scaled[:, 4] ./= scaled_mag
global_mag_factor = scaled_mag*MyExperimentalAnalysis.mag_factor("20250814")[1]
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
I_scan = logspace10(i_threshold,1.00; n=101);
plot!(fig,I_scan, zqm.(I_scan),
    label="Existing models",
    line=(:dash,:blue,1.75)
)
plot!(fig,
    I_scan, ki_itp.(I_scan, Ref(fit_scaled.ki)),
    label=L"Coquantum dynamics: $k_{i}= \left( %$(round(result.ki_fit, digits=4)) \pm %$(round(result.mse, sigdigits=1)) \right) \times 10^{-6} $",
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
# Scaled magnification
scaled_mag = result.scale_factor
data_scaled = copy(data)
data_scaled[:, 3] ./= scaled_mag
data_scaled[:, 4] ./= scaled_mag
global_mag_factor = scaled_mag*MyExperimentalAnalysis.mag_factor("20250814")[1]
plot!(fig,
    gradvsI.(data_scaled[:,1]),data_scaled[:,3],
    xerr = gradvsI.(data_scaled[:,2]),
    yerr = data_scaled[:,4],
    label=L"Experimental data (magnif.factor $m = %$(round(global_mag_factor, digits=4))$)",
    seriestype=:scatter,
    marker = (:circle,4,:white,stroke(0.5,:black) )
    # line=(:dash,:darkgreen,3),
    # fillcolor = :darkgreen,
    # fillalpha = 0.35,
)
I_scan = logspace10(i_threshold,1.00; n=101);
plot!(fig,gradvsI.(I_scan), zqm.(I_scan),
    label="Existing models",
    line=(:dash,:blue,1.75)
)
plot!(fig,
    gradvsI.(I_scan), ki_itp.(I_scan, Ref(fit_scaled.ki)),
    label=L"Coquantum dynamics: $k_{i}= \left( %$(round(result.ki_fit, digits=4)) \pm %$(round(result.mse, sigdigits=1)) \right) \times 10^{-6} $",
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


data     = hcat(xq, δxq, data_experiment.(xq), δyq)[i_start:end,:]

x_exp = data[:,1]
y_exp = data[:,3] ./ scaled_mag
σ_exp = data[:,4] ./ scaled_mag
y_CQD = ki_itp.(x_exp, Ref(result.ki_fit))
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

result = [
    [exp10(p) * x for p in -7:-6 for x in 1.0:0.1:9.9]; # Decades -7 to -6
    exp10(-5) * (1:0.1:10);                             # Decade -5
    exp10.(-3:0)                                        # Decades -3 to 0
]

using Plots
plot(result,
    yaxis=:log10)
