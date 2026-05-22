module FittingDataCQDQM

using Plots; gr()
using Plots.PlotMeasures
using LinearAlgebra
using Interpolations
using Optim
using Random
using Statistics
using Distributions
using PrettyTables
using OrderedCollections
using JLD2
using LaTeXStrings
using Printf
include("./TheoreticalSimulation.jl");
include("./JLD2_MyTools.jl");

"""
    combine_on_grid_mc_weighted(xsets, ysets;
                                σxsets=nothing,
                                σysets=nothing,
                                xq=:union,
                                B::Int=800,
                                outside::Symbol=:mask,
                                rel_x::Bool=false,
                                min_datasets::Int=1,
                                rng=Random.default_rng())

Weighted Monte-Carlo combination of multiple noisy 1D datasets onto a common grid,
propagating uncertainties in both x and y.

For each Monte-Carlo replicate, each dataset is perturbed according to its supplied
uncertainties, linearly interpolated onto a shared query grid, and then combined
pointwise across datasets using inverse-variance weights.

This is a weighted version of the user's earlier `average_on_grid_mc(...)`, which
performed an unweighted average after interpolation. Here, datasets with smaller
uncertainty contribute more strongly to the final combined curve. :contentReference[oaicite:1]{index=1}

# Arguments
- `xsets`, `ysets`:
    Collections of x- and y-vectors, one per dataset.
    Must satisfy `length(xsets) == length(ysets)` and each pair must have matching lengths.

# Keyword arguments
- `σxsets=nothing`:
    Per-dataset x uncertainties. If `nothing`, x is not perturbed.
- `σysets=nothing`:
    Per-dataset y uncertainties. If `nothing`, all datasets are combined with equal weights.
    If provided, these are also propagated onto the query grid and used for weighting.
- `rel_x=false`:
    If `true`, interpret `σxsets[i]` as relative uncertainties, i.e. `Δx = σx .* x`.
    Otherwise treat them as absolute.
- `xq=:union`:
    Query grid specification.
    - `:union`  → sorted union of all x-values
    - vector    → explicit query grid
- `B=400`:
    Number of Monte-Carlo replicates.
- `outside=:mask`:
    Extrapolation policy:
    - `:mask`   → outside each dataset's x-range return `NaN` and weight 0
    - `:linear` → linear extrapolation
    - `:flat`   → constant extrapolation
- `min_datasets=1`:
    Minimum number of contributing datasets required at a grid point in a replicate.
    If fewer contribute, that replicate gives `NaN` there.
- `rng`:
    Random number generator.

# Returns
A named tuple with fields:
- `xq`      : common query grid
- `μ`       : final Monte-Carlo mean combined curve
- `σ_mc`    : replicate-to-replicate Monte-Carlo standard deviation of the combined curve
- `σ_w`     : average weighted standard error from the inverse-variance combine step
- `σ_tot`   : total uncertainty, `sqrt(σ_mc^2 + σ_w^2)`
- `n_eff`   : average number of datasets contributing at each query point
- `preds`   : matrix of combined predictions, size `(B, length(xq))`

# Notes
- Linear interpolation is used for both `y` and `σy`.
- If `σysets` is provided, the local variance used for weighting is obtained by
  linearly interpolating `σy^2` onto the query grid.
- Repeated x-values after x-jittering are merged before interpolation.
- `σ_tot` is often the most useful final uncertainty to plot/use.

# Interpretation of uncertainties
At each grid point, the final uncertainty is split into two pieces:

1. `σ_mc`:
   variation of the final combined curve across Monte-Carlo replicates
2. `σ_w`:
   the typical inverse-variance combination uncertainty within each replicate

These are combined in quadrature as

    σ_tot = sqrt(σ_mc^2 + σ_w^2)

This is usually more informative than only reporting the Monte-Carlo spread.
"""
function combine_on_grid_mc_weighted(xsets, ysets;
                                     σxsets=nothing,
                                     σysets=nothing,
                                     xq=:union,
                                     B::Int=400,
                                     outside::Symbol=:mask,
                                     rel_x::Bool=false,
                                     min_datasets::Int=1,
                                     rng=Random.default_rng())

    @assert length(xsets) == length(ysets) "xsets and ysets must have the same number of datasets"
    nset = length(xsets)
    @assert nset > 0 "At least one dataset is required"
    @assert B > 0 "B must be positive"
    @assert outside in (:mask, :linear, :flat) "outside must be :mask, :linear, or :flat"
    @assert min_datasets ≥ 1 "min_datasets must be at least 1"


    function get_σx_for_dataset(σxsets, x, i)
        if σxsets === nothing
            return nothing
        elseif σxsets isa Real
            # same scalar for every point in every dataset
            return fill(Float64(σxsets), length(x))
        else
            # original behavior: one entry per dataset
            return collect(σxsets[i])
        end
    end

    if σxsets !== nothing && !(σxsets isa Real)
        @assert length(σxsets) == nset "σxsets must be either: nothing , a scalar, or one entry per dataset"
    end
    if σysets !== nothing
        @assert length(σysets) == nset "σysets must have one entry per dataset"
    end

    # ----------------------------
    # Build common query grid
    # ----------------------------
    xq_vec = xq === :union ? sort!(unique(vcat(map(collect, xsets)...))) : collect(xq)
    m = length(xq_vec)
    @assert m > 0 "Query grid is empty"

    # Combined prediction for each MC replicate
    preds   = fill(NaN, B, m)

    # Weighted standard error inside each replicate:
    #   σ_w(b, j) = sqrt(1 / sum_i w_i(j))
    σw_repl = fill(NaN, B, m)

    # Number of datasets contributing per replicate / grid point
    neff_repl = zeros(Int, B, m)

    # ----------------------------
    # Helper: merge repeated x after jittering
    # ----------------------------
    function merge_duplicate_x(x::AbstractVector, y::AbstractVector, σy::Union{Nothing,AbstractVector})
        p = sortperm(x)
        xs = collect(x[p])
        ys = collect(y[p])
        σs = σy === nothing ? nothing : collect(σy[p])

        xout = Float64[]
        yout = Float64[]
        σout = σs === nothing ? nothing : Float64[]

        i = 1
        n = length(xs)
        while i ≤ n
            j = i
            xi = xs[i]
            while j < n && isapprox(xs[j+1], xi; atol=0.0, rtol=0.0)
                j += 1
            end

            # block i:j has same x
            push!(xout, xi)

            if σs === nothing
                push!(yout, mean(@view ys[i:j]))
            else
                # inverse-variance merge at identical x
                v = (@view σs[i:j]).^2
                w = 1.0 ./ v
                ȳ = sum((@view ys[i:j]) .* w) / sum(w)
                σ̄ = sqrt(1.0 / sum(w))
                push!(yout, ȳ)
                push!(σout, σ̄)
            end

            i = j + 1
        end

        return xout, yout, σout
    end

    # ----------------------------
    # Helper: build interpolation / extrapolation
    # ----------------------------
    function build_interp(xb::AbstractVector, yb::AbstractVector)
        itp = Interpolations.interpolate((xb,), yb, Gridded(Interpolations.Linear()))
        if outside === :linear
            return Interpolations.extrapolate(itp, Line())
        elseif outside === :flat
            return Interpolations.extrapolate(itp, Flat())
        else
            return itp
        end
    end

    # ----------------------------
    # Helper: evaluate y and local variance on xq
    # ----------------------------
    function eval_dataset_on_grid(xb, yb, σyb, xqv)
        xlo, xhi = first(xb), last(xb)

        yvals = fill(NaN, length(xqv))
        vvals = fill(NaN, length(xqv))

        yitp = build_interp(xb, yb)

        vitp = nothing
        if σyb !== nothing
            # interpolate variance, not σ itself
            vb = σyb.^2
            vitp = build_interp(xb, vb)
        end

        if outside === :mask
            mask = (xqv .>= xlo) .& (xqv .<= xhi)
            yvals[mask] .= yitp.(xqv[mask])
            if vitp !== nothing
                vvals[mask] .= vitp.(xqv[mask])
            else
                vvals[mask] .= 1.0
            end
        else
            yvals .= yitp.(xqv)
            if vitp !== nothing
                vvals .= vitp.(xqv)
            else
                vvals .= 1.0
            end
        end

        # Numerical safety
        @inbounds for j in eachindex(vvals)
            if !isnan(vvals[j]) && vvals[j] ≤ 0
                vvals[j] = NaN
            end
        end

        return yvals, vvals
    end

    # ----------------------------
    # Monte-Carlo loop
    # ----------------------------
    for b in 1:B
        curves = Vector{Vector{Float64}}(undef, nset)
        vars   = Vector{Vector{Float64}}(undef, nset)

        for i in 1:nset
            x = collect(xsets[i])
            y = collect(ysets[i])
            @assert length(x) == length(y) "Dataset $i has mismatched x/y lengths"

            σx = get_σx_for_dataset(σxsets, x, i)
            σy = σysets === nothing ? nothing : collect(σysets[i])

            if σx !== nothing
                @assert length(σx) == length(x) "Dataset $i has mismatched x/σx lengths"
            end
            if σy !== nothing
                @assert length(σy) == length(y) "Dataset $i has mismatched y/σy lengths"
            end

            # Jitter x
            xb = if σx === nothing
                copy(x)
            else
                dx = rel_x ? σx .* x : σx
                x .+ randn(rng, length(x)) .* dx
            end

            # Jitter y
            yb = if σy === nothing
                copy(y)
            else
                y .+ randn(rng, length(y)) .* σy
            end

            # Need at least 2 points for interpolation
            if length(xb) < 2
                curves[i] = fill(NaN, m)
                vars[i]   = fill(NaN, m)
                continue
            end

            # Merge exact duplicates after jitter/sort
            xb2, yb2, σy2 = merge_duplicate_x(xb, yb, σy)

            if length(xb2) < 2
                curves[i] = fill(NaN, m)
                vars[i]   = fill(NaN, m)
                continue
            end

            curves[i], vars[i] = eval_dataset_on_grid(xb2, yb2, σy2, xq_vec)
        end

        # Weighted combine across datasets at each xq
        for j in 1:m
            num = 0.0
            den = 0.0
            ncontrib = 0

            @inbounds for i in 1:nset
                yij = curves[i][j]
                vij = vars[i][j]

                if !isnan(yij) && !isnan(vij) && vij > 0
                    wij = 1.0 / vij
                    num += wij * yij
                    den += wij
                    ncontrib += 1
                end
            end

            neff_repl[b, j] = ncontrib

            if ncontrib ≥ min_datasets && den > 0
                preds[b, j]   = num / den
                σw_repl[b, j] = sqrt(1.0 / den)
            else
                preds[b, j]   = NaN
                σw_repl[b, j] = NaN
            end
        end
    end

    # ----------------------------
    # Final summary across MC replicates
    # ----------------------------
    μ      = fill(NaN, m)
    σ_mc   = fill(NaN, m)
    σ_w    = fill(NaN, m)
    σ_tot  = fill(NaN, m)
    n_eff  = fill(NaN, m)

    for j in 1:m
        vals_pred = [v for v in @view(preds[:, j]) if !isnan(v)]
        vals_σw   = [v for v in @view(σw_repl[:, j]) if !isnan(v)]
        vals_neff = [v for v in @view(neff_repl[:, j]) if v > 0]

        if !isempty(vals_pred)
            μ[j] = mean(vals_pred)
            σ_mc[j] = length(vals_pred) > 1 ? std(vals_pred; corrected=true) : 0.0
        end

        if !isempty(vals_σw)
            # Typical within-replicate weighted SE
            σ_w[j] = mean(vals_σw)
        end

        if !isnan(σ_mc[j]) && !isnan(σ_w[j])
            σ_tot[j] = hypot(σ_mc[j], σ_w[j])
        elseif !isnan(σ_mc[j])
            σ_tot[j] = σ_mc[j]
        elseif !isnan(σ_w[j])
            σ_tot[j] = σ_w[j]
        end

        if !isempty(vals_neff)
            n_eff[j] = mean(vals_neff)
        end
    end

    return (
        xq    = xq_vec,
        μ     = μ,
        σ_mc  = σ_mc,
        σ_w   = σ_w,
        σ_tot = σ_tot,
        n_eff = n_eff,
        preds = preds,
    )
end


"""
    fit_ki(itp, data_original_set, data_selected_points, ki_list, ki_range)

Fit the induction coefficient `kᵢ` by minimizing a mean-squared error in **log10 space**
between the interpolated prediction `itp(x, kᵢ)` and a selected subset of data points.

Arguments
- `itp`
    Callable interpolated/model function such that `itp(Ic, ki)` returns the predicted value.
    It must accept scalar inputs and be broadcastable.

- `data_original_set`
    Full dataset, assumed here to have current in column 1 and observed values in column 3.

- `data_selected_points`
    Subset of points used for the fit, with current in column 1 and observed values in column 3.

- `ki_list`
    Vector of candidate `kᵢ` values defining the search interval.

- `ki_range`
    Tuple `(ki_start, ki_stop)` selecting the range inside `ki_list` used as optimization bounds.

Returns
- `(ki, ki_err, r2_coeff)`
"""
function fit_ki(itp, data_original_set, data_selected_points, ki_list, ki_range)

    ki_start, ki_stop = ki_range

    Ic_fit = data_selected_points[:, 1]
    z_fit  = data_selected_points[:, 3]

    # basic validation for log-space fit
    @assert all(isfinite, Ic_fit) "Ic_fit contains non-finite values"
    @assert all(isfinite, z_fit)  "z_fit contains non-finite values"
    @assert all(z_fit .> 0)       "z_fit must be strictly positive for log10 fitting"

    klo = ki_list[ki_start]
    khi = ki_list[ki_stop]
    @assert isfinite(klo) && isfinite(khi) "Optimization bounds must be finite"
    @assert klo < khi "Lower bound must be smaller than upper bound"

    # log-space loss used for fitting
    function loss_log(ki)
        z_pred = itp.(Ic_fit, Ref(ki))

        if any(.!isfinite.(z_pred)) || any(z_pred .<= 0)
            return Inf
        end

        return mean(abs2, log10.(z_pred) .- log10.(z_fit))
    end

    fit_param = Optim.optimize(loss_log, klo, khi, Brent())
    k_fit = Optim.minimizer(fit_param)

    # linear-space RMSE on selected points
    z_pred_sel = itp.(Ic_fit, Ref(k_fit))
    z_obs_sel  = z_fit
    mse_lin    = mean(abs2, z_pred_sel .- z_obs_sel)
    rmse_lin   = sqrt(mse_lin)

    # R² on the full dataset
    Ic   = data_original_set[:, 1]
    y    = data_original_set[:, 3]
    pred = itp.(Ic, Ref(k_fit))

    @assert all(isfinite, Ic)   "data_original_set current column contains non-finite values"
    @assert all(isfinite, y)    "data_original_set observed column contains non-finite values"
    @assert all(isfinite, pred) "Model prediction contains non-finite values"

    ss_tot = sum(abs2, y .- mean(y))
    coef_r2 = ss_tot > 0 ? 1 - sum(abs2, pred .- y) / ss_tot : NaN

    return (
        ki       = k_fit,
        ki_err   = rmse_lin,
        r2_coeff = coef_r2,
    )
end

"""
    fit_ki_with_error(itp, data;
        bounds::Tuple{<:Real,<:Real},
        conf::Real = 0.95,
        use_Zse::Bool = false,
        profile::Bool = true,
        profile_grid::Int = 400)

Fit a single induction parameter `kᵢ` by minimizing a residual sum of squares in
**log10 space**, and estimate its uncertainty from a local linearization around
the optimum.

The model `itp` is assumed to be callable as

    itp(I, ki)

where `I` is the independent variable and `ki` is the fitted parameter.

The fit minimizes

    RSS(ki) = Σ wᵢ [log10(itp(Iᵢ, ki)) - log10(Zᵢ)]²

using a 1D bounded optimization over the interval specified by `bounds`.

If `use_Zse=true`, the fit is weighted using the propagated variance in log10-space,

    Var(log10 Z) ≈ (σZ / (Z ln 10))²,

so that points with smaller uncertainty contribute more strongly to the fit.
If `use_Zse=false`, all valid points are weighted equally.

After the optimum is found, the function estimates the standard error of `kᵢ`
from a finite-difference approximation to the Jacobian of the log-residuals,
and returns both a symmetric t-based confidence interval and, optionally, a
profile-likelihood interval.

# Arguments
- `itp`:
    Callable model or interpolant satisfying `itp(I, ki) -> Zpred`.

- `data`:
    Array-like object with at least:
    - column 1: independent variable `I`
    - column 3: observed positive response `Z`
    - column 4: uncertainty `σZ` on `Z` (used only if `use_Zse=true`)

# Keyword arguments
- `bounds::Tuple{<:Real,<:Real}`:
    Lower and upper bounds `(ki_min, ki_max)` for the bounded 1D optimization.

- `conf::Real=0.95`:
    Confidence level used for the t-based interval `ci_t`, and for the
    profile interval `ci_profile` when `profile=true`.

- `use_Zse::Bool=false`:
    If `true`, use the uncertainties in column 4 to define inverse-variance
    weights in log10-space. If `false`, use equal weights.

- `profile::Bool=true`:
    If `true`, also compute a profile-style confidence interval by locating the
    parameter values where the objective exceeds the minimum by a target amount.

- `profile_grid::Int=400`:
    Number of grid points used on each side of the optimum to bracket the
    profile-interval crossings.

# Returns
A named tuple with fields:

- `ki`:
    Best-fit value of `kᵢ`.

- `ki_err`:
    Half-width of the symmetric t-based confidence interval for `kᵢ`.

- `se`:
    Estimated standard error of `kᵢ` from local linearization.

- `ci_t`:
    Symmetric t-based confidence interval `(k_lo, k_hi)`.

- `ci_profile`:
    Profile-style confidence interval `(k_lo, k_hi)`, or `nothing` if
    `profile=false`.

- `delta_target`:
    The χ² threshold used to define the profile interval, or `nothing` if
    `profile=false`.

- `delta_rss`:
    Increment added to the minimum RSS to define the profile interval.
    For weighted fits (`use_Zse=true`), this is `delta_target`.
    For unweighted fits, this is scaled by `sigma2`.

- `profile_note`:
    `nothing` for weighted fits, or a note indicating that the unweighted
    profile interval used a scaled threshold.

- `rss`:
    Residual sum of squares at the optimum.

- `sigma2`:
    Estimated residual variance, `rss / dof`.

- `dof`:
    Degrees of freedom used for uncertainty estimation.

- `n_used`:
    Number of points retained after filtering and derivative-validity checks.

- `r2_coeff`:
    Weighted coefficient of determination in log10-space, evaluated on the
    points used in the Jacobian-based uncertainty calculation.

- `converged`:
    Boolean flag indicating whether the optimizer reported convergence.

- `result`:
    Full optimizer result object returned by `Optim.optimize`.

# Data filtering
The function automatically excludes invalid rows before fitting:
- non-finite `I`
- non-finite `Z`
- non-positive `Z`
- and, if `use_Zse=true`, non-finite or non-positive `σZ`

In addition, uncertainty estimation excludes points for which the model or its
finite-difference perturbations are non-finite or non-positive.

# Notes
- The fit is performed in **log10-space**, so it is most appropriate when the
  dependent variable spans multiple orders of magnitude and relative deviations
  are more meaningful than absolute ones.

- The returned `ki_err` is derived from a local linear approximation and should
  be interpreted as an approximate uncertainty estimate near the optimum.

- The profile interval is usually more robust to local nonlinearity than the
  symmetric t-based interval, but it is still limited by the chosen bounds and
  the numerical behavior of the objective function.

- This function assumes that `itp(I, ki)` returns positive predictions in the
  region relevant to the fit. Non-positive or non-finite predictions are
  automatically discarded in the objective and derivative calculations.

# Examples
julia
fit = fit_ki_with_error(my_itp, data;
    bounds = (1e-7, 1e-5),
    conf = 0.95,
    use_Zse = true,
    profile = true
)

fit.ki
fit.ci_t
fit.ci_profile
"""
function fit_ki_with_error(itp, data;
    bounds::Tuple{<:Real,<:Real},
    conf::Real = 0.95,
    use_Zse::Bool = false,
    profile::Bool = true,
    profile_grid::Int = 400)
    I  = collect(Float64, data[:, 1])
    Z  = collect(Float64, data[:, 3])
    σZ = collect(Float64, data[:, 4])   # only used if use_Zse=true

    # Valid mask (log requires Z>0; σZ>0 if used)
    m0 = isfinite.(I) .& isfinite.(Z) .& (Z .> 0)
    if use_Zse
        m0 .&= isfinite.(σZ) .& (σZ .> 0)
    end
    I, Z, σZ = I[m0], Z[m0], σZ[m0]

    # weights in log10 space: Var(log10 Z) ≈ (σZ/(Z ln10))^2
    w = use_Zse ? ((Z .* log(10.0)) ./ σZ) .^ 2 : ones(length(I))

    ki_min, ki_max = float(bounds[1]), float(bounds[2])

    # RSS objective in log10 space
    function loss(ki)
        zpred = itp.(I, Ref(ki))
        m = isfinite.(zpred) .& (zpred .> 0)
        any(m) || return Inf
        r  = log10.(zpred[m]) .- log10.(Z[m])
        ww = w[m]
        sum(ww .* (r .^ 2))
    end

    res = Optim.optimize(loss, ki_min, ki_max, Brent())
    k̂  = Optim.minimizer(res)

    # residuals at optimum (define RSS0 consistently)
    ẑ = itp.(I, Ref(k̂))
    m = isfinite.(ẑ) .& (ẑ .> 0)
    Zu, wu = Z[m], w[m]
    r0 = log10.(ẑ[m]) .- log10.(Zu)

    p = 1
    n = length(r0)
    @assert n > p "Not enough valid points to estimate uncertainty"

    RSS0 = sum(wu .* (r0 .^ 2))
    dof  = n - p
    σ²   = RSS0 / dof

    # finite-difference step (always computed)
    fd_step(k, lo, hi; rel=cbrt(eps(Float64)), absmin=1e-12) = begin
        hh = max(absmin, rel * max(abs(k), 1.0))
        room = min(k - lo, hi - k)
        room > 0 ? min(hh, 0.5 * room) : absmin
    end
    h₀ = fd_step(k̂, ki_min, ki_max)

    # derivative dr/dk (central diff, common-valid points)
    z⁺ = itp.(I[m], Ref(k̂ + h₀))
    z⁻ = itp.(I[m], Ref(k̂ - h₀))
    mJ = isfinite.(z⁺) .& isfinite.(z⁻) .& (z⁺ .> 0) .& (z⁻ .> 0)
    @assert count(mJ) > p "Not enough valid points after derivative filtering"

    Zu2 = Zu[mJ]
    w2  = wu[mJ]
    rJ  = r0[mJ]

    r⁺ = log10.(z⁺[mJ]) .- log10.(Zu2)
    r⁻ = log10.(z⁻[mJ]) .- log10.(Zu2)
    drdk = (r⁺ .- r⁻) ./ (2h₀)

    # SE from linearization: Var(k̂) ≈ σ² / (J'WJ)
    SJJ = sum(w2 .* (drdk .^ 2))
    se  = sqrt(σ² / SJJ)

    tcrit = quantile(TDist(dof), 0.5 + conf/2)
    k_err = tcrit * se
    ci_t  = (k̂ - k_err, k̂ + k_err)

    # R² in log10 space (weighted)
    y  = log10.(Zu2)
    ŷ  = log10.(ẑ[m][mJ])
    ȳw = sum(w2 .* y) / sum(w2)
    TSS = sum(w2 .* (y .- ȳw).^2)
    R2  = TSS > 0 ? 1 - sum(w2 .* (y .- ŷ).^2) / TSS : NaN

    # Profile interval: ΔRSS = χ²(1,conf) if weighted; else scale by σ²
    ci_profile = nothing
    Δtarget = profile ? quantile(Chisq(1), conf) : nothing
    Δrss = nothing
    profile_note = nothing

    if profile
        if use_Zse
            Δrss = Δtarget
        else
            profile_note = :profile_interval_scaled_for_unweighted
            Δrss = σ² * Δtarget
        end
        target = RSS0 + Δrss

        function bracket_side(dir::Int)
            grid = range(k̂, dir > 0 ? ki_max : ki_min; length=profile_grid)
            prevk = first(grid)
            prevL = loss(prevk)
            for k in Iterators.drop(grid, 1)
                L = loss(k)
                if isfinite(L) && (L > target) && isfinite(prevL) && (prevL <= target)
                    return (prevk, k)
                end
                prevk, prevL = k, L
            end
            return nothing
        end

        function bisect_cross(a, b; maxiter=80, tol=1e-10)
            lo, hi = a, b
            for _ in 1:maxiter
                mid = (lo + hi)/2
                fmid = loss(mid) - target
                if !isfinite(fmid)
                    hi = mid
                    continue
                end
                if fmid > 0
                    hi = mid
                else
                    lo = mid
                end
                if abs(hi - lo) <= tol*max(1.0, abs(mid))
                    return (lo + hi)/2
                end
            end
            return (lo + hi)/2
        end

        left_br  = bracket_side(-1)
        right_br = bracket_side(+1)
        k_lo = left_br  === nothing ? ki_min : bisect_cross(left_br[1], left_br[2])
        k_hi = right_br === nothing ? ki_max : bisect_cross(right_br[1], right_br[2])
        ci_profile = (k_lo, k_hi)
    end

    return (
        ki = k̂,
        ki_err = k_err,
        se = se,
        ci_t = ci_t,
        ci_profile = ci_profile,
        delta_target = Δtarget,
        delta_rss = Δrss,
        profile_note = profile_note,
        rss = RSS0,
        sigma2 = σ²,
        dof = dof,
        n_used = length(rJ),
        r2_coeff = R2,
        converged = Optim.converged(res),
        result = res
    )
end

"""
    plot_combined_cqd_profiles_dict(
        ki_values,
        data_cqdup_path::AbstractString,
        data_cqddw_path::AbstractString;
        nz::Integer,
        σw_mm::Real,
        λ0::Real,
        nI::Integer,
        Icoils,
        colores_current,
        λ0_peak::Real = 1e-6,
        show_plots::Bool = true,
        show_table::Bool = true,
    )

Load CQD simulation data from JLD2 files, combine upward and downward
profiles, extract peak positions, and optionally visualize and tabulate
the results for a set of `kᵢ` values.

For each value of `kᵢ`, the function:
1. Loads CQD data for upward (`:up`) and downward (`:dw`) branches.
2. Combines amplitudes as:
       amp_combined = amp_up + 0.25 * amp_dw
3. Computes the peak position using a spline-based maximum finder.
4. Optionally plots the profiles and prints a formatted summary table.
5. Stores the resulting peak positions for each current.

# Arguments
- `ki_values`:
    Iterable collection of `kᵢ` values to process.

- `data_cqdup_path`:
    Path to the JLD2 file containing upward branch data.

- `data_cqddw_path`:
    Path to the JLD2 file containing downward branch data.

# Keyword arguments
- `nz::Integer`:
    Number of z-bins used in the stored data.

- `σw_mm::Real`:
    Gaussian smoothing width (in mm) used in the stored data.

- `λ0::Real`:
    Spline smoothing parameter used in the stored data.

- `nI::Integer`:
    Number of current values.

- `Icoils`:
    Vector of current values (in amperes).

- `colores_current`:
    Collection of colors for plotting, one per current.

- `λ0_peak::Real=1e-6`:
    Smoothing parameter used when extracting the peak position.

- `show_plots::Bool=true`:
    If `true`, display plots of the downward, upward, and combined profiles.

- `show_table::Bool=true`:
    If `true`, print a formatted table summarizing peak positions and relative errors.

# Returns
- `results::OrderedDict{Float64, Vector{Float64}}`:
    Dictionary mapping each `kᵢ` value to a vector of peak positions
    (one per current).

# Data structure assumptions
The JLD2 files are expected to contain entries indexed by a key constructed via

    JLD2_MyTools.make_keypath_cqd(branch, ki, nz, σw_mm, λ0)

Each entry must provide:
- `:z_profile`      → matrix with columns `[z, ..., amplitude]`
- `:z_max_smooth_spline_mm` → reference peak position

# Notes
- Only odd-indexed currents are plotted to reduce visual clutter.
- Peak extraction is performed using
      TheoreticalSimulation.max_of_bspline_positions
- The relative error shown in the table is:
      100 * (z_new / z_old - 1)
- Plot layout uses three stacked panels: downward, upward, and combined profiles.

# Example
julia
results = plot_combined_cqd_profiles_dict(
    ki_values,
    "cqdup.jld2",
    "cqddw.jld2";
    nz = 128,
    σw_mm = 0.05,
    λ0 = 0.01,
    nI = length(Icoils),
    Icoils = Icoils,
    colores_current = colors,
)
"""
function plot_combined_cqd_profiles_dict(
    ki_values,
    data_cqdup_path::AbstractString,
    data_cqddw_path::AbstractString;
    nz::Integer,
    σw_mm::Real,
    λ0::Real,
    nI::Integer,
    Icoils,
    colores_current,
    λ0_peak::Real = 1e-6,
    show_plots::Bool = true,
    show_table::Bool = true,
)
    results = OrderedDict{Float64, Vector{Float64}}()

    @assert length(Icoils) == nI
    @assert length(colores_current) >= nI

    function sci_label(x; n=3)
        x == 0 && return L"$0.0$"

        exp = floor(Int, log10(abs(x)))
        mant = x / 10.0^exp

        # round mantissa to (n sigdigits → typically 1 decimal for n=3)
        mant_r = round(mant; sigdigits=n)

        # force exactly ONE decimal place
        mant_str = @sprintf("%.1f", mant_r)

        return L"$%$(mant_str) \times 10^{%$exp}$"
    end

    make_profile_plot(; title::Union{Nothing,String}=nothing) = plot(
        xlabel = L"$z \ (\mathrm{mm})$",
        title = isnothing(title) ? "" : title,
        yformatter = y -> @sprintf("%.1e", y),
        xlims = (-5, 5),
    )

    @inline function maybe_plot_profile!(fig, ii, z, amp, Icoils, colores_current)
        if isodd(ii)
            plot!(
                fig,
                z,
                amp,
                label = "$(Int(round(1000 * Icoils[ii]))) mA",
                color = colores_current[ii],
            )
        end
        return nothing
    end

    function load_branch(path::AbstractString, branch::Symbol, ki, nz, σw_mm, λ0)
        key = JLD2_MyTools.make_keypath_cqd(branch, ki, nz, σw_mm, λ0)
        return jldopen(path, "r") do f
            f[key]
        end
    end

    nlegend = count(isodd, eachindex(Icoils))

    for ki_test in ki_values
        dat_branch_up = load_branch(data_cqddw_path, :dw, ki_test, nz, σw_mm, λ0)
        dat_branch_dw = load_branch(data_cqdup_path, :up, ki_test, nz, σw_mm, λ0)

        fig0 = fig1 = fig2 = nothing
        if show_plots
            fig0 = make_profile_plot(title = L"$k_{i}=$ " * sci_label(ki_test / 1e6, n=6))
            fig1 = make_profile_plot()
            fig2 = make_profile_plot()
        end

        zmax_vec = Vector{Float64}(undef, nI)
        z_old    = Vector{Float64}(undef, nI)

        amp_combined = nothing

        for ii in eachindex(Icoils)
            up_prof = dat_branch_up[ii][:z_profile]
            dw_prof = dat_branch_dw[ii][:z_profile]

            z_up   = @views up_prof[:, 1]
            amp_up = @views up_prof[:, 3]

            z_dw   = @views dw_prof[:, 1]
            amp_dw = @views dw_prof[:, 3]

            z_old[ii] = dat_branch_up[ii][:z_max_smooth_spline_mm]

            if amp_combined === nothing
                amp_combined = similar(amp_up)
            end
            @. amp_combined = amp_up + 0.25 * amp_dw

            z_max_new, _ = TheoreticalSimulation.max_of_bspline_positions(
                z_up, amp_combined; λ0 = λ0_peak
            )
            zmax_vec[ii] = z_max_new[1]

            if show_plots
                maybe_plot_profile!(fig0, ii, z_dw, amp_dw, Icoils, colores_current)
                maybe_plot_profile!(fig1, ii, z_up, amp_up, Icoils, colores_current)
                maybe_plot_profile!(fig2, ii, z_up, amp_combined, Icoils, colores_current)
            end
        end

        if show_plots
            plot!(fig0; legend = false)
            plot!(fig1; legend = false)
            plot!(
                fig2;
                legend = :outerbottom,
                legend_columns = min(7, nlegend),
                legend_fontsize = 6,
                background_color_legend = nothing,
                foreground_color_legend = nothing,
            )

            fig = plot(
                fig0, fig1, fig2;
                layout = (3, 1),
                labelfontsize = 10,
                size = (900, 1000),
                left_margin=2mm,
                bottom_margin=-12mm,
            )
            plot!(fig[1], xlabel = "", xformatter = _ -> "", bottom_margin=-7.5mm)
            plot!(fig[2], xlabel = "", xformatter = _ -> "", bottom_margin=-7.5mm)
            display(fig)
        end

        if show_table
            pct = 100 .* (zmax_vec ./ z_old .- 1)

            pretty_table(
                hcat(
                    Int.(round.(1000 .* Icoils)),
                    round.(zmax_vec; digits = 3),
                    round.(z_old; digits = 3),
                    round.(pct; digits = 1),
                );
                column_labels = [
                    ["Ic", "CQD F2", "CQD down", "RelError"],
                    ["(mA)", "(mm)", "(mm)", "(%)"]
                ],
                alignment = :c,
                row_label_column_alignment = :c,
                row_group_label_alignment = :c,
                title = @sprintf("CQD EQUIVALENT TO F2: %.3fe-6", ki_test),
                formatters = [
                    fmt__printf("%d", [1]),
                    fmt__printf("%8.3f", 2:3),
                    fmt__printf("%8.1f", [4]),
                ],
                style = TextTableStyle(
                    first_line_column_label = crayon"yellow bold",
                    table_border = crayon"blue bold",
                    title = crayon"bold red",
                ),
                table_format = TextTableFormat(
                    borders = text_table_borders__unicode_rounded
                ),
                equal_data_column_widths = true,
            )
        end

        results[ki_test] = copy(zmax_vec)
    end

    return results
end


"""
    fit_QM_scale_model(x, y, model;
        σy=nothing,
        mask=nothing,
        idx=nothing,
        xmin=nothing,
        xmax=nothing,
        offset::Bool=false,
        fitspace::Symbol=:log10,
        project::Symbol=:model_to_y)

Fit a scaled model to data, with optional offset, subset selection, and
choice of fitting space.

The model is first evaluated on the experimental x-grid:

    z = model.(x)

and then one of the following relations is fit.

Zero-offset modes
-----------------
If `offset=false`:

- `project=:model_to_y`
    Fits
        y ≈ α z
- `project=:y_to_model`
    Fits
        z ≈ α y

For zero offset, both `fitspace=:log10` and `fitspace=:linear` are supported.

Offset modes
------------
If `offset=true`:

- `project=:model_to_y`
    Fits
        y ≈ α z + β
- `project=:y_to_model`
    Fits
        z ≈ α y + β

For nonzero offset, only `fitspace=:linear` is supported.

Subset selection
----------------
You may restrict the fit using any combination of:
- `mask` : boolean mask of same length as `x`
- `idx`  : vector/range of selected indices
- `xmin` : keep only `x >= xmin`
- `xmax` : keep only `x <= xmax`

Important note on uncertainties
-------------------------------
The weighted fit is statistically meaningful only when the supplied uncertainty
matches the residual being minimized.

Therefore:

- `project=:model_to_y` naturally uses `σy`
- `project=:y_to_model` should use `σy = nothing`

In other words, for `project=:y_to_model`, this function enforces `σy === nothing`.

Arguments
---------
- `x`     : experimental x-values
- `y`     : experimental y-values
- `model` : callable object so that `model.(x)` returns model values on the x-grid

Keyword arguments
-----------------
- `σy`       : uncertainty on `y`; only allowed for `project=:model_to_y`
- `mask`     : boolean mask for selecting points
- `idx`      : indices for selecting points
- `xmin`     : minimum x to include
- `xmax`     : maximum x to include
- `offset`   : if `false`, fit only a scale factor; if `true`, fit scale + offset
- `fitspace` : `:log10` or `:linear`
- `project`  : `:model_to_y` or `:y_to_model`

Returns
-------
A named tuple containing the fitted parameters and diagnostics. Depending on the
mode, this may include:
- `α`, `β`
- `σα`, `σβ`
- `used_mask`
- `x_used`, `y_used`, `z_used`
- residuals in linear and/or log space
- `rss_linear`, `rss_log`
- `χ2`, `χ2red`, `χ2_log`, `χ2red_log`
- `dof`

Examples
--------
Zero-offset log-space fit, model projected onto data:

julia
Zero-offset linear fit, data projected onto model:
fit = fit_QM_scale_model(Ic, F1, zqm;
    σy=σF1,
    offset=false,
    fitspace=:log10,
    project=:model_to_y)

Linear fit with offset:
fit = fit_QM_scale_model(Ic, F1, zqm;
    σy=σF1,
    offset=true,
    fitspace=:linear,
    project=:model_to_y)
"""
function fit_QM_scale_model(x, y, model;
    σy=nothing,
    mask=nothing,
    idx=nothing,
    xmin=nothing,
    xmax=nothing,
    offset::Bool=false,
    fitspace::Symbol=:log10,              # :log10 or :linear
    project::Symbol=:model_to_y,        # :model_to_y or :y_to_model
    )

    @assert fitspace in (:log10, :linear) "fitspace must be :log10 or :linear"
    @assert project in (:model_to_y, :y_to_model) "project must be :model_to_y or :y_to_model"

    # IMPORTANT:
    # If project = :y_to_model, the residual is defined in z-space,
    # not in y-space. Therefore σy must be nothing.
    @assert !(project == :y_to_model && σy !== nothing) "project=:y_to_model must use σy = nothing"

    z = model.(x)

    # -----------------------------
    # Build selection mask
    # -----------------------------
    sel = trues(length(x))

    if mask !== nothing
        @assert length(mask) == length(x)
        sel .&= mask
    end

    if idx !== nothing
        tmp = falses(length(x))
        tmp[idx] .= true
        sel .&= tmp
    end

    if xmin !== nothing
        sel .&= x .>= xmin
    end

    if xmax !== nothing
        sel .&= x .<= xmax
    end

    # ============================================================
    # ZERO OFFSET: y ≈ αz   or   z ≈ αy
    # ============================================================
    if !offset

        # =========================
        # LOG-SPACE
        # =========================
        if fitspace == :log10

            sel_fit = sel .& (y .> 0) .& (z .> 0)
            if σy !== nothing
                sel_fit .&= (σy .> 0)
            end

            xx = x[sel_fit]
            yy = y[sel_fit]
            zz = z[sel_fit]

            @assert length(xx) > 1

            if project == :model_to_y
                Δ = log10.(yy) .- log10.(zz)

                if σy === nothing
                    logα = mean(Δ)
                    α = 10^(logα)
                    σα = nothing
                else
                    ss = σy[sel_fit]
                    w = yy.^2 ./ ss.^2
                    logα = sum(w .* Δ) / sum(w)
                    σα = 10^logα * sqrt(1 / sum(w))
                    α = 10^logα
                end

                pred = α .* zz
                rlog = log10.(yy) .- log10.(pred)
                rlin = yy .- pred

            else
                Δ = log10.(zz) .- log10.(yy)
                logα = mean(Δ)
                α = 10^(logα)
                σα = nothing

                pred = α .* yy
                rlog = log10.(zz) .- log10.(pred)
                rlin = zz .- pred
            end

            return (
                α=α, β=0.0, σα=σα, σβ=nothing,
                mode=:zero_offset_log10,
                fitspace=:log10,
                project=project,
                used_mask=sel_fit,
                x_used=xx, y_used=yy, z_used=zz,
                residuals_log=rlog,
                residuals_linear=rlin,
                rss_log=sum(rlog.^2),
                rss_linear=sum(rlin.^2),
                dof=length(yy)-1
            )

        # =========================
        # LINEAR SPACE
        # =========================
        else
            xx = x[sel]
            yy = y[sel]
            zz = z[sel]

            @assert length(xx) > 1

            if project == :model_to_y
                if σy === nothing
                    α = dot(yy, zz) / dot(zz, zz)
                    σα = nothing
                else
                    ss = σy[sel]
                    w = 1.0 ./ ss.^2
                    α = sum(w .* yy .* zz) / sum(w .* zz.^2)
                    σα = sqrt(1 / sum(w .* zz.^2))
                end

                pred = α .* zz
                rlin = yy .- pred

            else
                α = dot(zz, yy) / dot(yy, yy)
                σα = nothing

                pred = α .* yy
                rlin = zz .- pred
            end

            return (
                α=α, β=0.0, σα=σα, σβ=nothing,
                mode=:zero_offset_linear,
                fitspace=:linear,
                project=project,
                used_mask=sel,
                x_used=xx, y_used=yy, z_used=zz,
                residuals_linear=rlin,
                rss_linear=sum(rlin.^2),
                dof=length(yy)-1
            )
        end

    # ============================================================
    # OFFSET: y ≈ αz + β  or  z ≈ αy + β
    # ============================================================
    else
        @assert fitspace == :linear "offset=true only supports linear fit"

        xx = x[sel]
        yy = y[sel]
        zz = z[sel]

        @assert length(xx) > 2

        if project == :model_to_y
            A = hcat(zz, ones(length(zz)))
            target = yy
        else
            A = hcat(yy, ones(length(yy)))
            target = zz
        end

        if σy === nothing
            p = A \ target
            α, β = p
            pred = A * p
            r = target .- pred

            return (
                α=α, β=β,
                σα=nothing, σβ=nothing,
                mode=:linear_offset,
                fitspace=:linear,
                project=project,
                used_mask=sel,
                x_used=xx, y_used=yy, z_used=zz,
                residuals_linear=r,
                rss_linear=sum(r.^2),
                dof=length(target)-2
            )
        else
            ss = σy[sel]
            w = 1.0 ./ ss.^2
            W = Diagonal(w)

            ATA = A' * W * A
            ATb = A' * W * target

            p = ATA \ ATb
            α, β = p

            Cov = inv(ATA)
            σα = sqrt(Cov[1,1])
            σβ = sqrt(Cov[2,2])

            pred = A * p
            r = target .- pred

            return (
                α=α, β=β,
                σα=σα, σβ=σβ,
                mode=:linear_offset,
                fitspace=:linear,
                project=project,
                used_mask=sel,
                x_used=xx, y_used=yy, z_used=zz,
                residuals_linear=r,
                χ2=sum(w .* r.^2),
                χ2red=sum(w .* r.^2)/(length(target)-2),
                rss_linear=sum(r.^2),
                dof=length(target)-2
            )
        end
    end
end

"""
    fit_scale_and_k(x, y, model;
        bounds,
        σy=nothing,
        mask=nothing,
        idx_k=nothing,
        idx_alpha=nothing,
        xmin=nothing,
        xmax=nothing,
        offset=false,
        fitspace=:log10,
        project=:model_to_y,
        fit_alpha=true)

Fit a model of the form

    model(x, k)

to experimental data by optimizing the parameter `k`, while optionally
fitting a scale factor `α` and offset `β`.

The fitting procedure separates:
- optimization of `k` (global nonlinear parameter)
- estimation of `α` and `β` (linear or log-linear parameters)

This allows flexible fitting strategies where different subsets of the data
can be used for estimating `k` and for estimating `α` and `β`.

Arguments
---------
- `x`:
    Vector of independent variable values.

- `y`:
    Vector of observed data values.

- `model`:
    Callable function such that `model(x, k)` returns predictions.

Keyword arguments
-----------------
- `bounds`:
    Tuple `(kmin, kmax)` specifying the optimization interval for `k`.

- `σy`:
    Optional uncertainties on `y`. Used only for weighted fits when
    `project = :model_to_y`.

- `mask`:
    Boolean mask selecting a subset of data.

- `idx_k`:
    Indices used for fitting the parameter `k`.

- `idx_alpha`:
    Indices used for estimating `α` and `β`. If not provided, defaults to `idx_k`.

- `xmin`, `xmax`:
    Restrict the data to `xmin ≤ x ≤ xmax`.

- `offset`:
    If `false`, fit only a scale factor `α`.
    If `true`, fit both `α` and offset `β` (linear space only).

- `fitspace`:
    `:linear` or `:log10`. Determines the fitting metric.

- `project`:
    `:model_to_y` or `:y_to_model`.
    Controls whether the model is projected onto the data or vice versa.

- `fit_alpha`:
    If `false`, fixes `α = 1` and only fits `k` (and optionally `β`).

Returns
-------
A named tuple containing:
- `k`:
    Best-fit parameter value.

- `α`, `β`:
    Fitted scale and offset.

- `σα`, `σβ`:
    Estimated uncertainties on `α` and `β` (if available).

- `mode`:
    Symbol describing the fit type.

- `fitspace`, `project`:
    Fitting configuration.

- `used_mask`, `used_mask_k`, `used_mask_alpha`:
    Masks defining which data points were used.

- `x_used`, `y_used`, `z_used`:
    Data used in the final evaluation.

- `y_fit` or `z_fit`:
    Model prediction over the full dataset.

- `residuals_linear`, `residuals_log`:
    Residuals in linear and/or log space.

- `rss_linear`, `rss_log`:
    Residual sums of squares.

- `χ2`, `χ2red`:
    Weighted chi-square statistics (if `σy` is provided).

- `dof`:
    Degrees of freedom.

- `optimizer`:
    Full optimization result object.

Notes
-----
- Log-space fitting requires strictly positive `y` and model predictions.
- When `project = :y_to_model`, uncertainties `σy` must not be provided.
- Offset fitting is only supported in linear space.
- The optimization of `k` is performed using a 1D Brent method.

"""
function fit_scale_and_k(x, y, model;
    bounds::Tuple,
    σy=nothing,
    mask=nothing,
    idx_k=nothing,
    idx_alpha=nothing,
    xmin=nothing,
    xmax=nothing,
    offset::Bool=false,
    fitspace::Symbol=:log10,          # :log10 or :linear
    project::Symbol=:model_to_y,      # :model_to_y or :y_to_model
    fit_alpha::Bool=true,
)

    @assert fitspace in (:log10, :linear) "fitspace must be :log10 or :linear"
    @assert project in (:model_to_y, :y_to_model) "project must be :model_to_y or :y_to_model"
    @assert !(project == :y_to_model && σy !== nothing) "project=:y_to_model must use σy = nothing"
    @assert !(offset && fitspace != :linear) "offset=true only supports linear fit"

    n = length(x)
    @assert length(y) == n
    if σy !== nothing
        @assert length(σy) == n
    end
    if mask !== nothing
        @assert length(mask) == n
    end

    # --------------------------------------------------------
    # Base selection mask, same spirit as fit_QM_scale_model
    # --------------------------------------------------------
    sel = trues(n)

    if mask !== nothing
        sel .&= mask
    end
    if xmin !== nothing
        sel .&= x .>= xmin
    end
    if xmax !== nothing
        sel .&= x .<= xmax
    end

    # indices used for k-objective
    sel_k = copy(sel)
    if idx_k !== nothing
        tmp = falses(n)
        tmp[idx_k] .= true
        sel_k .&= tmp
    end

    # indices used for alpha/beta estimation
    sel_a = copy(sel)
    if idx_alpha !== nothing
        tmp = falses(n)
        tmp[idx_alpha] .= true
        sel_a .&= tmp
    else
        sel_a .= sel_k
    end

    @assert count(sel_k) > 1 "Need at least 2 points for k-fit selection."
    @assert count(sel_a) > 0 "Need at least 1 point for alpha selection."

    xk = x[sel_k]
    yk = y[sel_k]
    σk = σy === nothing ? nothing : σy[sel_k]

    xa = x[sel_a]
    ya = y[sel_a]
    σa = σy === nothing ? nothing : σy[sel_a]

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    z_of_k_all(k) = model.(x, Ref(k))
    z_of_k_k(k)   = model.(xk, Ref(k))
    z_of_k_a(k)   = model.(xa, Ref(k))

    function solve_alpha_beta_linear(k)
        za = z_of_k_a(k)

        if project == :model_to_y
            target = ya
            basis  = za
            ssa    = σa
        else
            target = za
            basis  = ya
            ssa    = nothing
        end

        if !offset
            if !fit_alpha
                α = 1.0
                β = 0.0
                σα = nothing
                σβ = nothing
            else
                if ssa === nothing
                    α = dot(target, basis) / dot(basis, basis)
                    σα = nothing
                else
                    w = 1.0 ./ ssa.^2
                    α = sum(w .* target .* basis) / sum(w .* basis.^2)
                    σα = sqrt(1 / sum(w .* basis.^2))
                end
                β = 0.0
                σβ = nothing
            end
            return (α=α, β=β, σα=σα, σβ=σβ)
        end

        # offset=true, linear only
        if project == :model_to_y
            A = hcat(za, ones(length(za)))
        else
            A = hcat(ya, ones(length(ya)))
        end

        if !fit_alpha
            # α fixed to 1, fit only β
            if project == :model_to_y
                resid0 = ya .- za
            else
                resid0 = za .- ya
            end

            if ssa === nothing
                β = mean(resid0)
                σβ = nothing
            else
                w = 1.0 ./ ssa.^2
                β = sum(w .* resid0) / sum(w)
                σβ = sqrt(1 / sum(w))
            end
            return (α=1.0, β=β, σα=nothing, σβ=σβ)
        end

        if ssa === nothing
            p = A \ target
            α, β = p
            return (α=α, β=β, σα=nothing, σβ=nothing)
        else
            w = 1.0 ./ ssa.^2
            W = Diagonal(w)
            ATA = A' * W * A
            ATb = A' * W * target
            p = ATA \ ATb
            α, β = p
            Cov = inv(ATA)
            σα = sqrt(Cov[1,1])
            σβ = sqrt(Cov[2,2])
            return (α=α, β=β, σα=σα, σβ=σβ)
        end
    end

    function solve_alpha_log(k)
        @assert !offset "offset=true is not supported in log fit"
        za = z_of_k_a(k)

        if project == :model_to_y
            valid = (ya .> 0) .& (za .> 0)
            if σa !== nothing
                valid .&= (σa .> 0)
            end
            @assert count(valid) > 1 "Need >1 positive points for log fit."
            yy = ya[valid]
            zz = za[valid]

            if !fit_alpha
                α = 1.0
                σα = nothing
            else
                Δ = log10.(yy) .- log10.(zz)
                if σa === nothing
                    logα = mean(Δ)
                    α = 10.0^logα
                    σα = nothing
                else
                    ss = σa[valid]
                    w = yy.^2 ./ ss.^2
                    logα = sum(w .* Δ) / sum(w)
                    α = 10.0^logα
                    σα = 10.0^logα * sqrt(1 / sum(w))
                end
            end
            return (α=α, β=0.0, σα=σα, σβ=nothing, valid=valid)
        else
            valid = (ya .> 0) .& (za .> 0)
            @assert count(valid) > 1 "Need >1 positive points for log fit."
            yy = ya[valid]
            zz = za[valid]

            if !fit_alpha
                α = 1.0
            else
                Δ = log10.(zz) .- log10.(yy)
                logα = mean(Δ)
                α = 10.0^logα
            end
            return (α=α, β=0.0, σα=nothing, σβ=nothing, valid=valid)
        end
    end

    function objective(k)
        if fitspace == :linear
            pars = solve_alpha_beta_linear(k)
            α, β = pars.α, pars.β
            zk = z_of_k_k(k)

            if project == :model_to_y
                pred = α .* zk .+ β
                r = yk .- pred
                if σk === nothing
                    return sum(r.^2)
                else
                    w = 1.0 ./ σk.^2
                    return sum(w .* r.^2)
                end
            else
                pred = α .* yk .+ β
                r = zk .- pred
                return sum(r.^2)
            end

        else
            pars = solve_alpha_log(k)
            α = pars.α
            zk = z_of_k_k(k)

            if project == :model_to_y
                valid = (yk .> 0) .& (zk .> 0)
                if σk !== nothing
                    valid .&= (σk .> 0)
                end
                if count(valid) <= 1
                    return Inf
                end
                yy = yk[valid]
                zz = zk[valid]
                pred = α .* zz
                r = log10.(yy) .- log10.(pred)

                if σk === nothing
                    return sum(r.^2)
                else
                    ss = σk[valid]
                    w = yy.^2 ./ ss.^2
                    return sum(w .* r.^2)
                end
            else
                valid = (yk .> 0) .& (zk .> 0)
                if count(valid) <= 1
                    return Inf
                end
                yy = yk[valid]
                zz = zk[valid]
                pred = α .* yy
                r = log10.(zz) .- log10.(pred)
                return sum(r.^2)
            end
        end
    end

    # --------------------------------------------------------
    # Optimize k
    # --------------------------------------------------------
    res = Optim.optimize(objective, bounds[1], bounds[2], Brent())
    kbest = Optim.minimizer(res)

    # --------------------------------------------------------
    # Final parameters at optimum
    # --------------------------------------------------------
    if fitspace == :linear
        pars = solve_alpha_beta_linear(kbest)
        αbest, βbest = pars.α, pars.β
        σα, σβ = pars.σα, pars.σβ
        zall = z_of_k_all(kbest)

        if project == :model_to_y
            pred_all = αbest .* zall .+ βbest
            rlin_all = y .- pred_all

            used_mask = sel
            xx = x[used_mask]
            yy = y[used_mask]
            zz = zall[used_mask]
            pred_used = αbest .* zz .+ βbest
            rlin = yy .- pred_used

            out = (
                k=kbest,
                α=αbest, β=βbest, σα=σα, σβ=σβ,
                mode=offset ? :linear_offset_k : :zero_offset_linear_k,
                fitspace=:linear,
                project=project,
                used_mask=used_mask,
                used_mask_k=sel_k,
                used_mask_alpha=sel_a,
                x_used=xx, y_used=yy, z_used=zz,
                y_fit=pred_all,
                residuals_linear=rlin,
                residuals_linear_all=rlin_all,
                rss_linear=sum(rlin.^2),
                dof=length(yy) - (offset ? 3 : (fit_alpha ? 2 : 1)),
                optimizer=res,
            )

            if σy !== nothing
                ss = σy[used_mask]
                w = 1.0 ./ ss.^2
                χ2 = sum(w .* rlin.^2)
                dof = length(yy) - (offset ? 3 : (fit_alpha ? 2 : 1))
                out = merge(out, (
                    χ2=χ2,
                    χ2red=χ2 / dof,
                ))
            end
            return out
        else
            pred_all = αbest .* y .+ βbest
            rlin_all = zall .- pred_all

            used_mask = sel
            xx = x[used_mask]
            yy = y[used_mask]
            zz = zall[used_mask]
            pred_used = αbest .* yy .+ βbest
            rlin = zz .- pred_used

            return (
                k=kbest,
                α=αbest, β=βbest, σα=σα, σβ=σβ,
                mode=offset ? :linear_offset_k : :zero_offset_linear_k,
                fitspace=:linear,
                project=project,
                used_mask=used_mask,
                used_mask_k=sel_k,
                used_mask_alpha=sel_a,
                x_used=xx, y_used=yy, z_used=zz,
                z_fit=pred_all,
                residuals_linear=rlin,
                residuals_linear_all=rlin_all,
                rss_linear=sum(rlin.^2),
                dof=length(yy) - (offset ? 3 : (fit_alpha ? 2 : 1)),
                optimizer=res,
            )
        end

    else
        pars = solve_alpha_log(kbest)
        αbest = pars.α
        βbest = 0.0
        σα, σβ = pars.σα, pars.σβ
        zall = z_of_k_all(kbest)

        if project == :model_to_y
            sel_fit = sel .& (y .> 0) .& (zall .> 0)
            if σy !== nothing
                sel_fit .&= (σy .> 0)
            end

            xx = x[sel_fit]
            yy = y[sel_fit]
            zz = zall[sel_fit]

            pred = αbest .* zz
            rlog = log10.(yy) .- log10.(pred)
            rlin = yy .- pred

            return (
                k=kbest,
                α=αbest, β=0.0, σα=σα, σβ=σβ,
                mode=:zero_offset_log10_k,
                fitspace=:log10,
                project=project,
                used_mask=sel_fit,
                used_mask_k=sel_k,
                used_mask_alpha=sel_a,
                x_used=xx, y_used=yy, z_used=zz,
                y_fit=αbest .* zall,
                residuals_log=rlog,
                residuals_linear=rlin,
                rss_log=sum(rlog.^2),
                rss_linear=sum(rlin.^2),
                dof=length(yy) - (fit_alpha ? 2 : 1),
                optimizer=res,
            )
        else
            sel_fit = sel .& (y .> 0) .& (zall .> 0)

            xx = x[sel_fit]
            yy = y[sel_fit]
            zz = zall[sel_fit]

            pred = αbest .* yy
            rlog = log10.(zz) .- log10.(pred)
            rlin = zz .- pred

            return (
                k=kbest,
                α=αbest, β=0.0, σα=σα, σβ=σβ,
                mode=:zero_offset_log10_k,
                fitspace=:log10,
                project=project,
                used_mask=sel_fit,
                used_mask_k=sel_k,
                used_mask_alpha=sel_a,
                x_used=xx, y_used=yy, z_used=zz,
                z_fit=αbest .* y,
                residuals_log=rlog,
                residuals_linear=rlin,
                rss_log=sum(rlog.^2),
                rss_linear=sum(rlin.^2),
                dof=length(yy) - (fit_alpha ? 2 : 1),
                optimizer=res,
            )
        end
    end
end


"""
    fit_ki_joint_scaling_fitsubset(
        data::AbstractMatrix,
        zqm,
        ki_itp,
        thresholdI::Real,
        ki_range::Tuple{<:Real,<:Real};
        fit_ki_mode::Symbol = :low_high,
        n_front::Integer = 3,
        n_back::Integer = 3,
        w::Real = 0.5,
        ref_type::Symbol = :arith,
        conf::Real = 0.95,
        use_Zse::Bool = false,
        profile::Bool = true,
        profile_grid::Integer = 400,
    )

Fit the parameter `ki` by comparing experimental data against a CQD interpolant in
log10-space, while determining a separate scale factor from the high-current tail.

The fit can be restricted to different subsets of the data using `fit_ki_mode`:
`:full`, `:low`, `:high`, or `:low_high`.

Arguments
- `data`:
    Matrix whose columns are interpreted as:
    - column 1: current `I`
    - column 3: measured signal `y`
    - column 4: uncertainty `σy`
- `zqm`:
    Callable reference model `zqm(I)`.
- `ki_itp`:
    Callable CQD interpolant `ki_itp(I, ki)`.
- `thresholdI`:
    Current threshold used to define the tail region for scaling.
- `ki_range`:
    Tuple `(kmin, kmax)` giving the optimization bounds for `ki`.

Keyword arguments
- `fit_ki_mode`:
    Which subset of points to use for the `ki` optimization.
- `n_front`, `n_back`:
    Number of points used from the beginning and end of the dataset when
    `fit_ki_mode = :low`, `:high`, or `:low_high`.
- `w`:
    Mixing weight between QM and CQD tail references.
- `ref_type`:
    `:arith` for arithmetic mixing or `:geom` for geometric mixing.
- `conf`:
    Confidence level for uncertainty intervals.
- `use_Zse`:
    If `true`, uses uncertainties from column 4 to build weighted residuals in log10-space.
- `profile`:
    If `true`, computes a profile-likelihood-like interval.
- `profile_grid`:
    Number of grid points used to bracket the profile interval.

Returns
A named tuple containing the fitted `ki`, scaling diagnostics, uncertainty
estimates, profile interval, goodness-of-fit metrics, and optimizer result.

Notes
- The fit is performed in log10-space, so only strictly positive model/data values
  are retained where needed.
- If `use_Zse == false`, the profile interval threshold is rescaled to match the
  unweighted RSS convention.
"""
function fit_ki_joint_scaling_fitsubset(
    data::AbstractMatrix,
    zqm,
    ki_itp,
    thresholdI::Real,
    ki_range::Tuple{<:Real,<:Real};
    fit_ki_mode::Symbol = :low_high,   # :full | :low | :high | :low_high
    n_front::Integer = 3,
    n_back::Integer = 3,
    w::Real = 0.5,
    ref_type::Symbol = :arith,         # :arith or :geom
    conf::Real = 0.95,
    use_Zse::Bool = false,
    profile::Bool = true,
    profile_grid::Integer = 400,
)

    # ------------------------------
    # Basic validation
    # ------------------------------
    size(data, 2) >= 4 || throw(ArgumentError(
        "data must have at least 4 columns: I, ..., y, σy"
    ))
    0 < conf < 1 || throw(ArgumentError("conf must satisfy 0 < conf < 1"))
    profile_grid ≥ 2 || throw(ArgumentError("profile_grid must be at least 2"))
    0.0 ≤ w ≤ 1.0 || throw(ArgumentError("w must lie in [0, 1]"))

    kmin, kmax = float.(ki_range)
    kmin < kmax || throw(ArgumentError("ki_range must satisfy kmin < kmax"))

    # ------------------------------
    # Unpack experimental data
    # ------------------------------
    Iexp = collect(Float64, data[:, 1])
    yexp = collect(Float64, data[:, 3])
    σy   = collect(Float64, data[:, 4])

    # Base validity (log requires y > 0; σy > 0 if used)
    mbase = isfinite.(Iexp) .& isfinite.(yexp) .& (yexp .> 0)
    if use_Zse
        mbase .&= isfinite.(σy) .& (σy .> 0)
    end

    Iexp = Iexp[mbase]
    yexp = yexp[mbase]
    σy   = σy[mbase]

    N = length(Iexp)
    N ≥ 2 || throw(ArgumentError("Not enough valid data points after filtering."))

    # ------------------------------
    # Tail region (for scaling)
    # ------------------------------
    tail_mask = Iexp .>= thresholdI
    n_tail = count(tail_mask)
    n_tail > 0 || throw(ArgumentError(
        "No experimental points with current ≥ $(thresholdI) A"
    ))

    I_tail = Iexp[tail_mask]
    y_tail = yexp[tail_mask]

    # ------------------------------
    # Fitting region (subset for ki optimization)
    # ------------------------------
    low_range  = 1:min(n_front, N)
    high_range = max(1, N - n_back + 1):N

    fit_idx = if fit_ki_mode === :full
        collect(1:N)
    elseif fit_ki_mode === :low
        collect(low_range)
    elseif fit_ki_mode === :high
        collect(high_range)
    elseif fit_ki_mode === :low_high
        unique!(sort!(vcat(collect(low_range), collect(high_range))))
    else
        throw(ArgumentError(
            "Invalid fit_ki_mode = $fit_ki_mode. Use :full, :low, :high, or :low_high."
        ))
    end
    n_fit_idx = length(fit_idx)

    # Var(log10 y) ≈ (σy / (y ln 10))^2
    log_weights(y, σ) = ((y .* log(10.0)) ./ σ) .^ 2

    # ------------------------------
    # Tail reference model z_ref(I; ki)
    # ------------------------------
    function zref_tail_for(ki)
        zqm_tail  = zqm.(I_tail)
        zcqd_tail = ki_itp.(I_tail, ki)

        if ref_type === :arith
            return w .* zqm_tail .+ (1 - w) .* zcqd_tail
        elseif ref_type === :geom
            return zqm_tail .^ w .* zcqd_tail .^ (1 - w)
        else
            throw(ArgumentError(
                "Invalid ref_type = $ref_type. Use :arith or :geom."
            ))
        end
    end

    # ------------------------------
    # Scale factor s(ki) from tail projection
    # ------------------------------
    function scale_for(ki)
        zref_tail = zref_tail_for(ki)
        m = isfinite.(zref_tail) .& isfinite.(y_tail) .& (zref_tail .!= 0)
        any(m) || return NaN

        denom = dot(zref_tail[m], zref_tail[m])
        denom == 0 && return NaN

        return dot(y_tail[m], zref_tail[m]) / denom
    end

    # ------------------------------
    # Residual builder in log10 space on fit subset
    # r_i = log10(yexp / scale) - log10(y_cqd)
    # ------------------------------
    function residuals_for(ki)
        scale = scale_for(ki)
        if !isfinite(scale) || scale == 0
            return nothing
        end

        y_cqd = ki_itp.(Iexp, ki)

        mfit = falses(N)
        mfit[fit_idx] .= true

        m = mfit .&
            isfinite.(y_cqd) .& (y_cqd .> 0) .&
            isfinite.(yexp) .& (yexp .> 0) .&
            (yexp ./ scale .> 0)

        any(m) || return nothing

        y_scaled = yexp[m] ./ scale
        r = log10.(y_scaled) .- log10.(y_cqd[m])

        ww = if use_Zse
            log_weights(yexp[m], σy[m])
        else
            ones(length(r))
        end

        return (r = r, w = ww, m = m, scale = scale)
    end

    # ------------------------------
    # Loss = weighted RSS in log10 space
    # ------------------------------
    function loss(ki)
        out = residuals_for(ki)
        out === nothing && return Inf
        return sum(out.w .* (out.r .^ 2))
    end

    # ------------------------------
    # Optimize ki (Brent)
    # ------------------------------
    opt = Optim.optimize(loss, kmin, kmax, Optim.Brent())
    ki_fit = Optim.minimizer(opt)

    out0 = residuals_for(ki_fit)
    out0 === nothing && error("Unexpected: residuals invalid at optimum ki.")

    r0 = out0.r
    w0 = out0.w
    m0 = out0.m
    scale_final = out0.scale

    p = 1
    n_used0 = length(r0)
    n_used0 > p || error("Not enough valid points to estimate uncertainty.")

    RSS0 = sum(w0 .* (r0 .^ 2))
    mse0 = RSS0 / n_used0

    # ------------------------------
    # Curvature-based SE via finite-difference Jacobian dr/dki
    # ------------------------------
    fd_step(k, lo, hi; rel = cbrt(eps(Float64)), absmin = 1e-12) = begin
        hh = max(absmin, rel * max(abs(k), 1.0))
        room = min(k - lo, hi - k)
        return room > 0 ? min(hh, 0.5 * room) : absmin
    end

    h0 = fd_step(ki_fit, kmin, kmax)

    outp = residuals_for(ki_fit + h0)
    outm = residuals_for(ki_fit - h0)
    (outp === nothing || outm === nothing) &&
        error("Derivative evaluation failed near optimum; try widening bounds or check model positivity.")

    mJ = m0 .& outp.m .& outm.m
    count(mJ) > p || error("Not enough common points to compute derivative.")

    function r_on_mask(ki, m)
        scale = scale_for(ki)
        y_cqd = ki_itp.(Iexp, ki)
        y_scaled = yexp[m] ./ scale
        return log10.(y_scaled) .- log10.(y_cqd[m])
    end

    r_plus  = r_on_mask(ki_fit + h0, mJ)
    r_minus = r_on_mask(ki_fit - h0, mJ)
    drdk = (r_plus .- r_minus) ./ (2h0)

    wJ = use_Zse ? log_weights(yexp[mJ], σy[mJ]) : ones(count(mJ))
    rJ = r_on_mask(ki_fit, mJ)

    RSS = sum(wJ .* (rJ .^ 2))
    dof = length(rJ) - p
    σ2  = RSS / dof
    SJJ = sum(wJ .* (drdk .^ 2))
    se  = sqrt(σ2 / SJJ)

    tcrit = Distributions.quantile(Distributions.TDist(dof), 0.5 + conf / 2)
    k_err = tcrit * se
    ci_t  = (ki_fit - k_err, ki_fit + k_err)

    # ------------------------------
    # R² in log10 space on mJ
    # ------------------------------
    scale0 = scale_for(ki_fit)
    y_cqd0 = ki_itp.(Iexp, ki_fit)
    y_obs  = log10.(yexp[mJ] ./ scale0)
    y_hat  = log10.(y_cqd0[mJ])

    ybar_w = sum(wJ .* y_obs) / sum(wJ)
    TSS = sum(wJ .* (y_obs .- ybar_w) .^ 2)
    R2 = TSS > 0 ? 1 - RSS / TSS : NaN

    # ------------------------------
    # Profile interval with proper scaling
    # ------------------------------
    ci_profile = nothing
    Δtarget = nothing
    Δrss = nothing
    profile_note = nothing

    if profile
        Δtarget = Distributions.quantile(Distributions.Chisq(1), conf)

        if use_Zse
            Δrss = Δtarget
            profile_note = nothing
        else
            profile_note = :profile_interval_scaled_for_unweighted
            dof0 = n_used0 - p
            σ2hat0 = RSS0 / dof0
            Δrss = σ2hat0 * Δtarget
        end

        target = RSS0 + Δrss

        function bracket_side(dir::Int)
            grid = range(ki_fit, dir > 0 ? kmax : kmin; length = profile_grid)
            prevk = first(grid)
            prevL = loss(prevk)

            for k in Iterators.drop(grid, 1)
                L = loss(k)
                if isfinite(L) && (L > target) && isfinite(prevL) && (prevL <= target)
                    return (prevk, k)
                end
                prevk, prevL = k, L
            end
            return nothing
        end

        function bisect_cross(a, b; maxiter = 80, tol = 1e-10)
            lo, hi = a, b
            for _ in 1:maxiter
                mid = (lo + hi) / 2
                fmid = loss(mid) - target

                if !isfinite(fmid)
                    hi = mid
                    continue
                end

                if fmid > 0
                    hi = mid
                else
                    lo = mid
                end

                if abs(hi - lo) <= tol * max(1.0, abs(mid))
                    return (lo + hi) / 2
                end
            end
            return (lo + hi) / 2
        end

        left_br  = bracket_side(-1)
        right_br = bracket_side(+1)

        k_lo = left_br  === nothing ? kmin : bisect_cross(left_br[1], left_br[2])
        k_hi = right_br === nothing ? kmax : bisect_cross(right_br[1], right_br[2])

        ci_profile = (k_lo, k_hi)
    end

    # ------------------------------
    # Extra diagnostics
    # ------------------------------
    rmse_log10 = sqrt(mse0)
    mult_rmse  = 10.0 ^ rmse_log10

    frac_err = abs.(10.0 .^ r0 .- 1.0)
    med_abs_frac = Statistics.median(frac_err)
    p90_abs_frac = Distributions.quantile(frac_err, 0.90)
    p99_abs_frac = Distributions.quantile(frac_err, 0.99)
    max_abs_frac = maximum(frac_err)
    n_bad_2pct   = count(>(0.02), frac_err)
    n_bad_5pct   = count(>(0.05), frac_err)

    scale_inv = 1 / scale_final
    scale_pct = (scale_inv - 1) * 100

    return (
        ki_fit       = ki_fit,
        scale_factor = scale_final,
        scale_inv    = scale_inv,
        scale_pct    = scale_pct,

        rss          = RSS0,
        mse          = mse0,
        rmse_log10   = rmse_log10,

        se           = se,
        k_err        = k_err,
        ci_t         = ci_t,

        ci_profile   = ci_profile,
        delta_target = Δtarget,
        delta_rss    = Δrss,
        profile_note = profile_note,

        R2           = R2,

        mult_rmse    = mult_rmse,
        med_abs_frac = med_abs_frac,
        p90_abs_frac = p90_abs_frac,
        p99_abs_frac = p99_abs_frac,
        max_abs_frac = max_abs_frac,
        n_bad_2pct   = n_bad_2pct,
        n_bad_5pct   = n_bad_5pct,

        n_total      = N,
        n_tail       = n_tail,
        n_fit_idx    = n_fit_idx,
        n_used       = length(rJ),

        converged    = Optim.converged(opt),
        result       = opt,
    )
end



"""
    FitStats{T}

Container for goodness-of-fit diagnostics computed in log-space.

Fields
- `logMSE`:
    Mean squared residual in log-space.
- `logRMSE`:
    Root mean squared residual in log-space.
- `R2_log`:
    Coefficient of determination computed in log-space.
- `chi2_log`:
    Chi-square statistic in log-space, if uncertainties are provided.
    Otherwise `NaN`.
- `chi2_red`:
    Reduced chi-square, if uncertainties are provided.
    Otherwise `NaN`.
- `p_chi2`:
    Upper-tail p-value for the chi-square statistic, if uncertainties are provided.
    Otherwise `NaN`.
- `AIC`:
    Akaike information criterion.
- `BIC`:
    Bayesian information criterion.
- `NMAD`:
    Normalized median absolute deviation of the log-residuals.
"""
struct FitStats{T<:AbstractFloat}
    logMSE::T
    logRMSE::T
    R2_log::T
    chi2_log::T
    chi2_red::T
    p_chi2::T
    AIC::T
    BIC::T
    NMAD::T
end

"""
    goodness_of_fit(y, ypred; σ = nothing, k::Integer = 0)

Compute goodness-of-fit diagnostics in log-space for a model prediction `ypred`
compared against observed data `y`.

Arguments
- `y`:
    Observed dependent-variable values.
- `ypred`:
    Predicted dependent-variable values.

Keyword arguments
- `σ`:
    Optional uncertainties on `y` in linear space. If provided, they are propagated
    approximately into log-space using `σ_log ≈ σ / y`.
- `k`:
    Number of fitted parameters used in the model. This is used for the reduced
    chi-square, AIC, and BIC.

Returns
A `FitStats` object containing log-space fit diagnostics.

Notes
- This function requires `y > 0` and `ypred > 0` for all retained points, since
  logarithms are used.
- If `σ === nothing`, no formal chi-square test is performed, and `chi2_log`,
  `chi2_red`, and `p_chi2` are returned as `NaN`.
- `AIC` and `BIC` are computed from `logMSE` when uncertainties are not provided,
  and from `χ²` when uncertainties are available.
"""
function goodness_of_fit(
    y::AbstractVector,
    ypred::AbstractVector;
    σ::Union{Nothing,AbstractVector} = nothing,
    k::Integer = 0,
)
    length(y) == length(ypred) || throw(ArgumentError(
        "y and ypred must have the same length."
    ))
    k ≥ 0 || throw(ArgumentError("k must be non-negative."))

    N = length(y)
    N > 0 || throw(ArgumentError("Input vectors must be non-empty."))

    T = float(promote_type(eltype(y), eltype(ypred)))
    yv  = collect(T, y)
    ypv = collect(T, ypred)

    all(isfinite, yv)  || throw(ArgumentError("y contains non-finite values."))
    all(isfinite, ypv) || throw(ArgumentError("ypred contains non-finite values."))
    all(>(zero(T)), yv)  || throw(ArgumentError("All y values must be > 0 for log-space metrics."))
    all(>(zero(T)), ypv) || throw(ArgumentError("All ypred values must be > 0 for log-space metrics."))

    logy    = log.(yv)
    logpred = log.(ypv)
    r       = logy .- logpred

    logMSE  = Statistics.mean(r .^ 2)
    logRMSE = sqrt(logMSE)

    tss = sum((logy .- Statistics.mean(logy)) .^ 2)
    R2_log = tss > zero(T) ? one(T) - sum(r .^ 2) / tss : T(NaN)

    rmed = Statistics.median(r)
    NMAD = T(1.4826) * Statistics.median(abs.(r .- rmed))

    if isnothing(σ)
        chi2_log = T(NaN)
        chi2_red = T(NaN)
        p_chi2   = T(NaN)

        mse_safe = max(logMSE, eps(T))
        AIC = T(2) * T(k) + T(N) * log(mse_safe)
        BIC = T(k) * log(T(N)) + T(N) * log(mse_safe)
    else
        length(σ) == N || throw(ArgumentError(
            "σ must have the same length as y and ypred."
        ))

        σv = collect(T, σ)
        all(isfinite, σv) || throw(ArgumentError("σ contains non-finite values."))
        all(>(zero(T)), σv) || throw(ArgumentError(
            "All σ values must be > 0 when provided."
        ))

        σlog = σv ./ yv
        all(>(zero(T)), σlog) || throw(ArgumentError(
            "Propagated log-space uncertainties must be > 0."
        ))

        chi2_log = sum((r ./ σlog) .^ 2)
        dof = max(N - k, 1)
        chi2_red = chi2_log / T(dof)

        dist = Distributions.Chisq(dof)
        p_chi2 = Distributions.ccdf(dist, chi2_log)

        AIC = T(2) * T(k) + chi2_log
        BIC = T(k) * log(T(N)) + chi2_log
    end

    return FitStats{T}(logMSE, logRMSE, R2_log, chi2_log, chi2_red, p_chi2, AIC, BIC, NMAD)
end

"""
    goodness_of_fit(x, y, ypred; σ = nothing, k::Integer = 0)

Compatibility method that accepts `x` but uses it only to verify matching lengths.
"""
function goodness_of_fit(
    x::AbstractVector,
    y::AbstractVector,
    ypred::AbstractVector;
    σ::Union{Nothing,AbstractVector} = nothing,
    k::Integer = 0,
)
    length(x) == length(y) == length(ypred) || throw(ArgumentError(
        "x, y, and ypred must have the same length."
    ))
    return goodness_of_fit(y, ypred; σ = σ, k = k)
end

function Base.propertynames(::FitStats; private::Bool = false)
    return (
        :logMSE,
        :logRMSE,
        :R2_log,
        :chi2_log,
        :chi2_red,
        :p_chi2,
        :AIC,
        :BIC,
        :NMAD,
    )
end

function Base.show(io::IO, s::FitStats)
    println(io, "FitStats")
    println(io, "  logMSE   = ", s.logMSE)
    println(io, "  logRMSE  = ", s.logRMSE)
    println(io, "  R2_log   = ", s.R2_log)
    println(io, "  chi2_log = ", s.chi2_log)
    println(io, "  chi2_red = ", s.chi2_red)
    println(io, "  p_chi2   = ", s.p_chi2)
    println(io, "  AIC      = ", s.AIC)
    println(io, "  BIC      = ", s.BIC)
    print(io,   "  NMAD     = ", s.NMAD)
end

function Base.show(io::IO, ::MIME"text/plain", s::FitStats)
    println(io, "FitStats")
    println(io, "  logMSE   = ", round(s.logMSE; sigdigits = 6))
    println(io, "  logRMSE  = ", round(s.logRMSE; sigdigits = 6))
    println(io, "  R2_log   = ", round(s.R2_log; sigdigits = 6))
    println(io, "  chi2_log = ", round(s.chi2_log; sigdigits = 6))
    println(io, "  chi2_red = ", round(s.chi2_red; sigdigits = 6))
    println(io, "  p_chi2   = ", round(s.p_chi2; sigdigits = 6))
    println(io, "  AIC      = ", round(s.AIC; sigdigits = 6))
    println(io, "  BIC      = ", round(s.BIC; sigdigits = 6))
    print(io,   "  NMAD     = ", round(s.NMAD; sigdigits = 6))
end


end