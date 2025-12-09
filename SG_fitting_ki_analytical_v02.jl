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
include("./Modules/DataReading.jl")

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

function mag_factor(directory::String)
    if directory == "20251109"
        values = (0.996,0.0047)
    else
        values = (1.1198,0.0061) 
    end
    return values
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

    mag, δmag = mag_factor(wanted_data_dir)

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


i_threshold = 0.025;
i_start  = searchsortedfirst(exp_avg[:i_smooth],i_threshold);
data     = hcat(exp_avg[:i_smooth],exp_avg[:z_smooth],exp_avg[:δz_smooth])[i_start:end,:];

using Optim

"""
Fit ki using:
  • Scaling from the high-current tail (I >= thresholdI)
  • Reference model = w*QM + (1-w)*CQD
  • Fitting region = first n_front points + last n_back points

Inputs:
  data        :: Nx3 matrix  (I, z_exp, δz)
  zqm         :: callable: zqm(I)
  ki_itp      :: callable: ki_itp(I, ki)
  thresholdI  :: Float, threshold for tail region
  ki_range    :: (kmin, kmax)
  n_front     :: number of points to use at beginning
  n_back      :: number of points to use at end
  w           :: QM weight in combined reference (0–1)

Returns:
  (ki_fit, scale_factor, mse)
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
)

    Iexp = data[:,1]
    yexp = data[:,2]

    N = length(Iexp)

    # Tail region (for scaling)
    tail_idx = findall(Iexp .>= thresholdI)
    if isempty(tail_idx)
        error("No experimental points with current ≥ $thresholdI A")
    end

    I_tail  = Iexp[tail_idx]
    y_tail  = yexp[tail_idx]

    # Fitting region (first n_front and last n_back)
    fit_idx = vcat(1:n_front, (N-n_back+1):N)

    function loss(ki)
        # CQD model for all points
        y_cqd = ki_itp.(Iexp, ki)

        # reference model in tail
        zref_tail = w .* zqm.(I_tail) .+
                    (1-w) .* ki_itp.(I_tail, ki)

        # scale factor
        scale = dot(y_tail, zref_tail) /
                dot(zref_tail, zref_tail)

        # scale experimental data
        yexp_scaled = yexp ./ scale

        # loss only on selected fit indices
        r = log.(yexp_scaled[fit_idx]) .-
            log.(y_cqd[fit_idx])

        return mean(abs2, r)
    end

    # optimize over ki
    kmin, kmax = ki_range
    opt = optimize(loss, kmin, kmax, Brent())

    ki_fit = Optim.minimizer(opt)
    mse    = Optim.minimum(opt)

    # final scale factor
    zref_tail_final = w .* zqm.(I_tail) .+
                      (1-w) .* ki_itp.(I_tail, ki_fit)
    scale_final = dot(y_tail, zref_tail_final) /
                  dot(zref_tail_final, zref_tail_final)

    return (
        ki_fit       = ki_fit,
        scale_factor = scale_final,
        mse          = mse,
        n_front      = n_front,
        n_back       = n_back,
        w            = w
    )
end


i_threshold = 0.025;
i_start  = searchsortedfirst(exp_avg[:i_smooth],i_threshold);
data     = hcat(exp_avg[:i_smooth],exp_avg[:z_smooth],exp_avg[:δz_smooth])[i_start:end,:];
result = fit_ki_joint_scaling_fitsubset(
    data,
    zqm,
    ki_itp,
    0.92,                                  # tail threshold
    (ki_list[ki_start], ki_list[ki_stop]); # bracket
    n_front = 6,
    n_back  = 6,
    w       = 0.55
)


fig=plot(data[:,1], zqm.(data[:,1]),
    label="Quantum mechanics",
    line=(:solid,:red,1.75)
)
# Scaled magnification
scaled_mag = result.scale_factor
data_scaled = copy(data)
data_scaled[:, 2] ./= scaled_mag
data_scaled[:, 3] ./= scaled_mag
plot!(data_scaled[:,1],data_scaled[:,2],
    ribbon = data_scaled[:,3],
    label=L"Combined data (scaled $m_{p} = %$(round(scaled_mag, digits=4))$ )",
    line=(:dash,:darkgreen,1.2),
    fillcolor = :darkgreen,
    fillalpha = 0.35,
)
data_fitting = data[[1:6; (end-5):(end)], :]
data_scaled_fitting = data_scaled[[1:6; (end-5):(end)], :]
fit_scaled   = fit_ki(data_scaled, data_scaled_fitting, ki_list, (ki_start,ki_stop))
I_scan = logspace10(i_threshold,1.00; n=101);
plot!(
    I_scan, ki_itp.(I_scan, Ref(fit_scaled.ki)),
    label=L"Scaled: $k_{i}= \left( %$(round(fit_scaled.ki, digits=4)) \pm %$(round(fit_scaled.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
    line=(:solid,:blue,2),
    # marker=(:xcross, :blue, 0.2),
    markerstrokewidth=1
)
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
)
display(fig)

























# 20250814
data_fitting = data[[1:4; (end-3):(end)], :]
# 20250820
data_fitting = data[[1:4; (end-3):(end)], :]
# 20250825
data_fitting = data[[1:3; (end-3):(end)], :]
# 20250919
data_fitting = data[[1:2; (end-3):(end)], :]
# 20251002
data_fitting = data[[2:3; (end):(end)], :]
# 20251003
data_fitting = data[[2:4; (end):(end)], :]
data_fitting = data[2:4, :]
# 20251006
data_fitting = data[[2:4; (end):(end)], :]
# 20251109
data_fitting = data[[2:4; (end-1):(end)], :]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

## choose a few points for low currents and high currents
if wanted_data_dir == "20250815"
    data_fitting = data[[9,10,11,15,19:22...],:] # for fitting purposes
elseif wanted_data_dir == "20250820"
    data_fitting = data[[2:4; (end-3):(end-1)], :]
elseif wanted_data_dir == "20250825"
    data_fitting = data[[2:4; (end-3):(end-1)], :]
elseif wanted_data_dir == "20250919"
    data_fitting = data[[2:4; (end-3):(end-1)], :]
else
    data_fitting = data[[10,11,12,14,22:25...],:] # for fitting purposes
end