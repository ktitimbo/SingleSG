# Simulation of atom trajectories in the Stern–Gerlach experiment
# Interpolation of the grid for fitting the induction term
# Kelvin Titimbo
# California Institute of Technology
# September 2025

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
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSS");
const OUTDIR    = joinpath(@__DIR__, "data_studies", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
include("./Modules/TheoreticalSimulation.jl");
include("./Modules/DataReading.jl")

logspace10(lo, hi; n=50) = 10.0 .^ range(log10(lo), log10(hi); length=n)

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
function fit_ki_with_error(itp, data;
                           bounds::Tuple{<:Real,<:Real},
                           conf::Real=0.95,
                           weights::Union{Nothing,AbstractVector}=nothing,
                           h::Union{Nothing,Real}=nothing)

    # --- unpack data ---
    I = collect(Float64, data[:, 1])
    Z = collect(Float64, data[:, 2])

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
    println(h₀)
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


# import Suleyman's ki simulation:
# 20250829T005913: no binning
sim_data = CSV.read("./simulation_data/results_CQD_20250829T005913.csv",DataFrame; header=false)
sim_QMdata = CSV.read("./simulation_data/results_QM_20250829T005913.csv",DataFrame; header=false)
Ic_sim = sim_data[!,1]
ki_sim = collect(1.50:0.10:3.50) ./ 1_000_000
n_col = (1+2*length(ki_sim)) + 1 
z_sim = Matrix(sim_data[1:end,n_col:end])
z_QMsim = vec(mean(Matrix(sim_QMdata[1:end,n_col:end]), dims=2))
z_QMsim_err = vec(std(Matrix(sim_QMdata[1:end,n_col:end]), dims=2; corrected=true))/sqrt(length(ki_sim))

cols = palette(:darkrainbow, length(ki_sim));
fig = plot(xlabel=L"$I_{c}$ (A)",ylabel=L"$z$ (mm)",)
for i=1:length(ki_sim)
    plot!(fig,Ic_sim[6:end], abs.(z_sim[6:end,i]) , 
        label = L"$k_{i} =%$(round(1e6*ki_sim[i], digits=2)) \times 10^{-6}$",
        line=(:solid,cols[i],1)
    )
end
plot!(fig, Ic_sim[6:end], abs.(z_QMsim[6:end]),
    label="QM + Class.Trajs.",
    ribbon = z_QMsim_err,
    line=(:dash,:green,2) )
plot!(fig, 
    title="CQD Simulation",
    xlims=(8e-3,1.5),
    ylims=(8e-3,2.5),
    xaxis=:log10,
    yaxis=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    legendfontsize=7,
    size=(800,600),
)
display(fig)


# Interpolated surface
itp = Spline2D(Ic_sim, ki_sim, z_sim; kx=3, ky=3, s=0.00);

# Select data
wanted_data_dir = "20250825/"
wanted_binning  = 2
wanted_smooth   = 0.02 

res = DataReading.find_report_data(
        joinpath(@__DIR__, "analysis_data");
        wanted_data_dir=wanted_data_dir,
        wanted_binning=wanted_binning,
        wanted_smooth=wanted_smooth
)

if res === nothing
    @warn "No matching report found"
else
    @info "Matched" res.path res.data_dir res.name res.binning res.smoothing
    # I_exp = sort(res.currents_mA / 1_000);
    # z_exp = res.framewise_mm/res.magnification;
end

load_data = CSV.read(joinpath(dirname(res.path),"fw_data.csv"),DataFrame; header=true);
I_exp = load_data[!,"I_coil_mA"]/1_000
z_exp = load_data[!,"F1_z_centroid_mm"]/res.magnification
z_exp_stde = load_data[!,"F1_z_centroid_se_mm"]

# choose a few points for low currents and high currents
data = hcat(I_exp, z_exp, z_exp_stde)
data = data[[10,12,14,22:25...],:] # for fitting purposes
# data = data[[9,10,11,15,19:22...],:] # for fitting purposes

function loss(ki) # loss function
    # ni=12
    z_pred = itp.(data[:,1], Ref(ki))
    return mean(abs2,log10.(z_pred) .- log10.(data[:,2]))
end


fit_param = optimize(loss, minimum(ki_sim), maximum(ki_sim),Brent())
k_fit = Optim.minimizer(fit_param)

# diagnostics
mse = loss(k_fit);
pred = itp.(I_exp, Ref(k_fit));
coef_r2 = 1 - sum(abs2, pred .- z_exp) / sum(abs2, z_exp .- mean(z_exp))

# given: itp, data (N×2), ki_sim
out = fit_ki_with_error(itp, data; bounds=(minimum(ki_sim), maximum(ki_sim)))
@info "Fitting" out.k_hat out.k_err out.ci

I_scan = logspace10(20e-3,1; n=30)
fig= plot(
    title =L"$R^{2}=%$(round(coef_r2,digits=4))$",
    xlabel=L"$I_{c}$ (A)",
    ylabel=L"$z$ (mm)"
)
plot!(fig,I_exp[9:end], z_exp[9:end], 
    label="Experiment $(wanted_data_dir): n=$(wanted_binning) | λ=$(wanted_smooth)",
    seriestype=:scatter,
    yerror = z_exp_stde[9:end],
    marker=(:circle,:white,1.8), 
    markerstrokecolor=:red, 
    markerstrokewidth=2
)
plot!(fig,data[:,1], data[:,2],
    seriestype=:scatter,
    label="Used for fitting",
    marker=(:xcross,:black,2),
    markerstrokecolor=:black,
    markerstrokewidth=2,)
plot!(fig,I_scan,itp.(I_scan, Ref(out.ci[2])),
    color=:royalblue1,
    label=false,
    linewidth=0,
    fillrange= itp.(I_scan, Ref(out.ci[1])),
    fillcolor=:royalblue1,
    fillalpha=0.35,
    )
plot!(fig,I_scan, itp.(I_scan, Ref(k_fit)),
    label=L"$k_{i}= \left( %$(round(1e6*k_fit, digits=4)) \pm %$(round(1e6*out.k_err, digits=4)) \right) \times 10^{-6} $",
    line=(:solid,:blue,1),
    marker=(:xcross, :blue, 1),
)
plot!(fig, Ic_sim[6:end], abs.(z_QMsim[6:end]),
    label="QM + Class.Trajs.",
    ribbon = z_QMsim_err,
    line=(:dash,:green,2),
    fillalpha=0.23, 
    fillcolor=:green, 
)
plot!(fig,
    xaxis=:log10,
    yaxis=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(8e-3,1.5),
    ylims=(8e-3,2),
    legend=:bottomright,
)
display(fig)
