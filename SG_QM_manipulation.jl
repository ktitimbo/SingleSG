# Simulation of atom trajectories in the Stern–Gerlach experiment
# Manipulation of quantum simulations
# Kelvin Titimbo
# California Institute of Technology
# December 2025

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

function keypath(branch::Symbol, ki::Float64, nz::Int, gw::Float64, λ0_raw::Float64)
    fmt(x) = @sprintf("%.12g", x)  # safer than %.6g to reduce collisions
    return "/" * String(branch) *
           "/ki=" * fmt(ki) *"e-6" *
           "/nz=" * string(nz) *
           "/gw=" * fmt(gw) *
           "/lam=" * fmt(λ0_raw)
end

"""
Compute per-particle fluorescence weights for a laser sheet.

Inputs:
- vz, vy: arrays (m/s) at detection (same length)
- λ: wavelength (m), e.g. 770e-9 for K D1
- Γν: natural FWHM linewidth in Hz, e.g. 5.956e6
- δlaser: laser detuning from unshifted resonance in Hz
- s0: saturation parameter (dimensionless). If unknown, treat as a tunable parameter.
- Δy: sheet thickness in m (50e-6)
Returns:
- weights w (arbitrary units)
"""
function doppler_weights(vz::AbstractVector, vy::AbstractVector;
    λ::Float64 = 770e-9,
    Γν::Float64 = 5.956e6,
    δlaser::Float64 = 0.0,
    s0::Float64 = 30.3,
    Δy::Float64 = 50e-6)

    N = length(vz)
    length(vy) == N || error("vz and vy must have same length")

    w = Vector{Float64}(undef, N)

    @inbounds for i in 1:N
        # Detuning seen by particle i (Hz)
        Δ = δlaser - vz[i]/λ

        # Lorentzian factor (dimensionless)
        denom = 1 + s0 + (2*Δ/Γν)^2

        # Scattering rate ∝ s0/denom (prefactor cancels if you normalize)
        R = s0 / denom

        # Transit time in sheet
        τ = Δy / vy[i]

        w[i] = R * τ
    end

    return w
end

"""
Build Doppler-corrected image as a weighted 2D histogram.

coords_u, coords_v: arrays defining image coordinates (e.g. xf, yf) at detector
weights: per-particle weights (same length)
edges_u, edges_v: bin edges (ranges)
"""
function weighted_image(coords_u, coords_v, w, edges_u, edges_v; norm_mode::Symbol= :probability)
    if norm_mode === :none
        h = fit(Histogram, (coords_u, coords_v), weights(w), (edges_u, edges_v))                    # raw counts (no normalization)
    elseif norm_mode in (:probability, :pdf, :density)
        h = normalize(fit(Histogram, (coords_u, coords_v), weights(w), (edges_u, edges_v)); mode=norm_mode)
    else
        throw(ArgumentError("mode must be one of :pdf, :density, :probability, :none, got $norm_mode"))
    end
    return h
end

cam_pixelsize = 6.5e-6 ;  # Physical pixel size of camera [m]

# Coil currents
Icoils = [0.00,
            0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
            0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,
            0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
];
nI = length(Icoils)

nx_bins, nz_bins = 64, 2 ;
gaussian_width_mm = 0.200;
λ0_raw            = 0.01;
λ0_spline         = 0.001;
ki_idx = 27;
Ic_idx = 1;
norm_type = :none


data_CQD_up  = load(joinpath(@__DIR__,"simulation_data","cqd_simulation_6M","cqd_6000000_ki0$(ki_idx)_up_screen.jld2"),"screen")[:data]
data = data_CQD_up[Ic_idx]

xf = 1e3* view(data,:,9)
zf = 1e3* view(data,:,10)

vxf = view(data,:,4)
vyf = view(data,:,5)
vzf =view(data,:,11)

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
ki = cqd_meta[:ki][ki_idx]

data_CQD_up_profiles = jldopen(table_cqd_path, "r") do file
    file[keypath(:up,ki,nz_bins,gaussian_width_mm,λ0_raw)]
end

# Fixed analysis limits
xlim = (-8.0, 8.0)
zlim = (-12.5, 12.5)
xmin, xmax = xlim
zmin, zmax = zlim

# Bin size in mm (default_camera_pixel_size is assumed global in meters)
x_bin_size = 1e3 * nx_bins * cam_pixelsize
z_bin_size = 1e3 * nz_bins * cam_pixelsize

# --------------------------------------------------------
# X edges: force symmetric centers around 0
# --------------------------------------------------------
x_half_range = max(abs(xmin), abs(xmax))
kx = max(1, ceil(Int, x_half_range / x_bin_size))
centers_x = collect((-kx:kx) .* x_bin_size)
edges_x = collect((-(kx + 0.5)) * x_bin_size : x_bin_size : ((kx + 0.5) * x_bin_size))

# --------------------------------------------------------
# Z edges: force symmetric centers around 0
# --------------------------------------------------------
z_half_range = max(abs(zmin), abs(zmax))
kz = max(1, ceil(Int, z_half_range / z_bin_size))
centers_z = collect((-kz:kz) .* z_bin_size)
edges_z = collect((-(kz + 0.5)) * z_bin_size : z_bin_size : ((kz + 0.5) * z_bin_size))

if norm_type === :none
    h = fit(Histogram, (xf, zf), (edges_x, edges_z))                    # raw counts (no normalization)
elseif norm_type in (:probability, :pdf, :density)
    h = normalize(fit(Histogram, (xf, zf), (edges_x, edges_z)); mode=norm_type)
else
    throw(ArgumentError("mode must be one of :pdf, :density, :probability, :none, got $norm_type"))
end

counts = h.weights  # size: (length(centers_x), length(centers_z))

# heatmap(
#     centers_x,
#     centers_z,
#     counts';
#     colormap = :viridis,
#     xlabel = "x (mm)",
#     ylabel = "z (mm)",
#     xlims = (-8,8),
#     ylims = (-3,12),
#     aspect_ratio = :equal,
#     colorbar = true
# )

# z-profile = mean over x bins
z_profile_raw = vec(mean(counts, dims = 1))
z_max_raw_mm = centers_z[argmax(z_profile_raw)]
z_max_raw_spline_mm, Sfit_raw = TheoreticalSimulation.max_of_bspline_positions(centers_z,z_profile_raw;λ0=λ0_raw)

# 1) Choose laser detuning (Hz). You can set δlaser = 0 or optimize it later.
δlaser = 60e6
# 2) s0: saturation parameter (dimensionless)
s0 = 0.9
# 3) Compute Doppler weights
w = doppler_weights(vzf, vyf; Γν=5.956e6, δlaser=δlaser, s0=s0)

img = weighted_image(xf, zf, w./ sum(w), edges_x, edges_z; norm_mode = norm_type)
# img.weights is your Doppler-corrected image intensity array

counts_doppler = img.weights
# heatmap(
#     centers_x,
#     centers_z,
#     counts_doppler';
#     colormap = :viridis,
#     xlabel = "x (mm)",
#     ylabel = "z (mm)",
#     xlims = (-8,8),
#     ylims = (-3,12),
#     aspect_ratio = :equal,
#     colorbar = true
# )


# z-profile = mean over x bins
z_profile_raw_doppler = vec(mean(counts_doppler, dims = 1))
z_max_raw_mm_doppler = centers_z[argmax(z_profile_raw_doppler)]
z_max_raw_spline_mm_doppler, Sfit_raw_doppler = TheoreticalSimulation.max_of_bspline_positions(centers_z,z_profile_raw_doppler;λ0=0.005)

plot(xlabel=L"$z$ (mm)")
plot!(centers_z,z_profile_raw, label="Raw profile",
    xlims=(-2,10))
plot!(centers_z,z_profile_raw_doppler, label="Doppler correction")
plot!(data_CQD_up_profiles[Ic_idx][:z_profile][:,1],
    data_CQD_up_profiles[Ic_idx][:z_profile][:,3], label="Smoothing")

2+2



fig = plot(xlabel=L"$z$ (mm)")
for δlaser in -6:6
w = doppler_weights(vzf, vyf; Γν=5.956e6, δlaser=1e6*δlaser, s0=s0)
img = weighted_image(xf, zf, w ./ sum(w), edges_x, edges_z; norm_mode = norm_type)
counts_doppler = img.weights
z_profile_raw_doppler = vec(mean(counts_doppler, dims = 1))
plot!(fig,centers_z,z_profile_raw_doppler, label=L"$\delta_{L}=%$(δlaser)$MHz")
end
plot!(xlims=(-2,+2))
display(fig)





mean(vzf/770e-9)






















data_CQD

ki_l = round.(1e6*[
    [exp10(p) * x for p in -8:-8 for x in 1.0:1:9]; 
    [exp10(p) * x for p in -7:-7 for x in 1.0:1:9]; 
    [exp10(p) * x for p in -6:-6 for x in 1.0:0.1:9.9]; 
    ## exp10(-5) * (1:0.1:10);
    exp10.(-5:-1)
];sigdigits=4)



jldopen(joinpath(OUTDIR,"qm_screen_data.jld2"), "w") do file
    file["meta/Icoils"] = Icoils
    file["meta/levels"] = quantum_numbers 
    for i in 1:nI
        file["screen/I$(i)"] = data_alive_screen[i]  
    end
end







table_old     = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_3M","qm_3000000_screen_profiles_table.jld2"))["table"];
table_qm_f1   = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_6M","qm_6000000_screen_profiles_f1_table.jld2"))["table"];
table_qm_f2   = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_6M","qm_6000000_screen_profiles_f2_table.jld2"))["table"];
table_qm_f1_1 = load(joinpath(@__DIR__,"simulation_data","qm_simulation_7M","qm_7000000_screen_profiles_f1_table.jld2"))["table"];
table_qm_f2_2 = load(joinpath(@__DIR__,"simulation_data","qm_simulation_7M","qm_7000000_screen_profiles_f2_table.jld2"))["table"];

ks = keys(table_qm_f1);

zqm_old  = [table_old[(1,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:nI]
zqm_F1   = [table_qm_f1[(1,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:nI]
zqm_F2   = [table_qm_f2[(1,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:nI]
zqm_F1_1 = [table_qm_f1_1[(1,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:nI]
zqm_F2_2 = [table_qm_f2_2[(1,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:nI]

hcat(Icoils, zqm_old, zqm_F1, zqm_F1_1, 100*(zqm_F1_1 .- zqm_old)./zqm_F1_1, 100*(zqm_F1_1 .- zqm_old) )

hcat(Icoils, zqm_F2, zqm_F2_2, 100*(zqm_F2_2 .- zqm_F2)./zqm_F2_2, 100*(zqm_F2_2 .- zqm_F2) )


plot(Icoils[3:end], zqm_old[3:end], label="3M")
plot!(Icoils[3:end], zqm_F1[3:end], label="6M")
plot!(Icoils[3:end], zqm_F1_1[3:end], label="7M")
plot!(
    title="QM = (1,0.200,0.02)",
    xscale=:log10,
    yscale=:log10,
    xlabel="Coil current (A)",
    ylabel=L"$F=1$ Peak position",
)


jldopen(joinpath("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_7M", "qm_screen_data.jld2"), "r") do file
    keys(file)                # top-level groups
    keys(file["screen"])      # should show I1, I2, ...
end

dataI1 = jldopen(joinpath("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_7M", "qm_screen_data.jld2"), "r") do file
    file["screen/i1"]
end

dataI47 = jldopen(joinpath("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_7M", "qm_screen_data.jld2"), "r") do file
    file["screen/i47"]
end

vy_final_I1  = vcat(dataI1[6],dataI1[7],dataI1[8])[:,5]
vz_final_I1  = vcat(dataI1[6],dataI1[7],dataI1[8])[:,9]
vy_final_I47 = vcat(dataI47[6],dataI47[7],dataI47[8])[:,5]
vz_final_I47 = vcat(dataI47[6],dataI47[7],dataI47[8])[:,9]



mean(vy_final_I47)
sqrt(mean(vy_final_I47.^2))

mean(vz_final_I1)
sqrt(mean(vz_final_I47.^2))

sqrt(mean(vz_final_I47.^2)-mean(vz_final_I47)^2)


function doppler_peak_from_vz(vzlist::AbstractVector;
    weights::Union{Nothing,AbstractVector}=nothing,
    λ_m::Float64 = 770e-9,
    Γν_MHz::Float64 = 5.956,
    pad_MHz::Float64 = 20.0,
    Ngrid::Int = 4001
)
    vz = collect(Float64.(vzlist))
    isempty(vz) && error("vzlist is empty.")

    w = weights === nothing ? ones(Float64, length(vz)) : collect(Float64.(weights))
    length(w) == length(vz) || error("weights must match vzlist length.")
    wsum = sum(w)
    wsum > 0 || error("sum(weights) must be > 0.")

    # Doppler detunings in MHz
    Δ = @. (-vz / λ_m) / 1e6

    # weighted mean Doppler shift (MHz)
    δ_mean = sum(w .* Δ) / wsum

    @inline L(x) = 1.0 / (1.0 + (2.0*x/Γν_MHz)^2)

    δmin = minimum(Δ) - pad_MHz
    δmax = maximum(Δ) + pad_MHz
    δgrid = range(δmin, δmax, length=Ngrid)

    S = Vector{Float64}(undef, length(δgrid))
    for (i, δ) in enumerate(δgrid)
        acc = 0.0
        @inbounds for j in eachindex(Δ)
            acc += w[j] * L(δ - Δ[j])
        end
        S[i] = acc / wsum
    end

    imax = argmax(S)
    δ_peak = δgrid[imax]

    return (; δ_mean, δ_peak, δ_peak_minus_mean = δ_peak - δ_mean, δgrid, S, Δ)
end

out = doppler_peak_from_vz(vz_final_I47)
@show out.δ_mean out.δ_peak out.δ_peak_minus_mean ; 



1.8e-3*650.94/0.32

histogram(vy_final_I1;
    bins = TheoreticalSimulation.FreedmanDiaconisBins(vy_final_I1), 
    normalize = :pdf,
    label = L"$v_{0,y}$", 
    alpha = 0.65, 
    color = :orange,
    xlabel = L"$v_{0,y} \ (\mathrm{m/s})$"
)
histogram!(vy_final_I47;
    bins = TheoreticalSimulation.FreedmanDiaconisBins(vy_final_I47), 
    normalize = :pdf,
    label = L"$v_{0,y}$", 
    alpha = 0.65, 
    color = :blue,
    xlabel = L"$v_{0,y} \ (\mathrm{m/s})$"
)


plot(Icoils[2:end],(zqm_F1 .+ zqm_F2)[2:end]/2,
    xscale=:log10,
    )


println("the nz-binning are: [ ", join(string.(sort(unique(first.(ks)))), ", ")," ]")
println("the σ-values for the smoothing convolution are (mm): [ ", join((@sprintf("%.3f", x) for x in sort(unique(getindex.(ks,2)))), ", ")," ]")
println("the spline fitting smoothing factors are: [ ", join((@sprintf("%.3f", x) for x in sort(unique(getindex.(ks,3)))), ", ")," ]")

data_profile = table_qm_f1_1[(1,0.001,0.001)][47][:z_profile]
zprof = vec(data_profile[:, 1])[1500:end]
Aprof = vec(data_profile[:, 2])[1500:end]

plot(zprof,Aprof)
hline!([maximum(Aprof)/2])
vline!([0.95], line=(:black,:dash,1))
vline!([3.70], line=(:black,:dash,1))
3.70-0.95

using LsqFit, Statistics

# Gaussian + constant background
# model(z, p) is the safest convention for LsqFit
function gauss_bg(z, p)
    A, z0, σ, C = p
    @. A * exp(-0.5 * ((z - z0) / σ)^2) + 0*C
end
# p = [A, z0, sigma, C]

function fit_gaussian_sigma(z::AbstractVector, I::AbstractVector;
                            use_window::Bool=true, window_factor::Float64=2.5)

    z = collect(Float64.(z))
    I = collect(Float64.(I))
    length(z) == length(I) || error("z and I must have the same length")

    # sort by z
    idx = sortperm(z)
    z = z[idx]; I = I[idx]

    # peak guess
    imax = argmax(I)
    z0_guess = z[imax]

    # background guess: median of lowest 20%
    n = length(I)
    lo = max(1, round(Int, 0.2n))
    C_guess = median(sort(I)[1:lo])

    # amplitude guess
    A_guess = max(I[imax] - C_guess, 1e-12)

    # rough sigma guess via weighted second moment
    w = max.(I .- C_guess, 0.0)
    if sum(w) > 0
        sigma_guess = sqrt(sum(w .* (z .- z0_guess).^2) / sum(w))
    else
        sigma_guess = 0.1 * (maximum(z) - minimum(z))
    end
    sigma_guess = max(sigma_guess, 1e-12)

    # optional window around peak
    z_fit = z; I_fit = I
    if use_window
        halfw = window_factor * sigma_guess
        m = (z .>= z0_guess - halfw) .& (z .<= z0_guess + halfw)
        if count(m) ≥ 6
            z_fit = z[m]; I_fit = I[m]
        end
    end

    p0 = [A_guess, z0_guess, sigma_guess, C_guess]

    fit = curve_fit(gauss_bg, z_fit, I_fit, p0)
    p = coef(fit)                # [A, z0, sigma, C]

    # covariance & 1-sigma errors (approx)
    # cov = estimate_covar(fit)
    # perr = sqrt.(diag(cov))
    sigma = p[3]
    # sigma_err = perr[3]

    fwhm = 2 * sqrt(2 * log(2)) * sigma
    # fwhm_err = 2 * sqrt(2 * log(2)) * sigma_err

    return (; A=p[1], z0=p[2], sigma=sigma, C=p[4],
            # sigma_err=sigma_err, 
            fwhm=fwhm, 
            # fwhm_err=fwhm_err, 
            # cov=cov, 
            fit=fit
            )
end

function halfmax_data(z::AbstractVector, I::AbstractVector; pad::Int=0)
    z = vec(Float64.(z))
    I = vec(Float64.(I))
    length(z) == length(I) || error("z and I must have same length")

    # sort by z so the kept region is contiguous
    idx = sortperm(z)
    z = z[idx]; I = I[idx]

    # same background estimate as your fitting code
    n  = length(I)
    lo = max(1, round(Int, 0.2n))
    C  = median(sort(I)[1:lo])

    imax = argmax(I)
    Imax = I[imax]

    Ihalf = C + 0.90*(Imax - C)

    m = I .>= Ihalf
    if count(m) < 6
        error("Too few points above half max (got $(count(m))).")
    end

    # keep a contiguous block from first..last above half max
    inds = findall(m)
    i1 = max(1, first(inds) - pad)
    i2 = min(n, last(inds) + pad)

    return z[i1:i2], I[i1:i2], Ihalf, C
end



z_half, I_half, Ihalf, C = halfmax_data(zprof,Iprof; pad=0)

out = fit_gaussian_sigma(z_half,I_half)
# @show out.sigma out.sigma_err out.fwhm out.fwhm_err
@show out.sigma out.fwhm 

zfit = range(minimum(zprof), maximum(zprof), length=5000)
Ifit = gauss_bg(zfit, coef(out.fit))

scatter(zprof, Iprof; ms=3, label="data", xlabel="z", ylabel="Intensity",marker=(:white,:circle,2))
plot!(zfit, Ifit; lw=2, label="fit")






data_profile[:,1]










fig = plot(xlims = (-2,2),
    xlabel=L"$z$ (mm)")
for i in sort(unique(getindex.(ks,1)))
    set1 = table_qm_f1[(i,0.001,0.001)][1][:z_profile]
    plot!(fig,
        set1[:,1],set1[:,2],
        label=L"$n_{z}=%$(i)$")
    display(fig)
end

fig = plot(xlims = (-2,2),
    xlabel=L"$z$ (mm)")
set0 = table_qm_f1[(2,0.001,0.001)][1][:z_profile]
plot!(fig,
    set0[:,1],set0[:,2],
    label=L"$\sigma_{z}=0$ μm (raw)")
for i in sort(unique(getindex.(ks,2)))
    set1 = table_qm_f1[(2,i,0.001)][1][:z_profile]
    plot!(fig,
        set1[:,1],set1[:,3],
        label=L"$\sigma_{z}=%$(1e3*i)$ μm")
    display(fig)
end

fig = plot(xlims = (-2,2),
    xlabel=L"$z$ (mm)")
set0 = table_qm_f1[(2,0.001,0.001)][1][:z_profile]
plot!(fig,
    set0[:,1],set0[:,2],
    label=L"$\sigma_{z}=0$ μm (raw)")
for i in sort(unique(getindex.(ks,3)))
    set1 = table_qm_f1[(2,0.001,i)][1][:z_profile]
    plot!(fig,
        set1[:,1],set1[:,3],
        label=L"$\lambda_{0}=%$(i)$")
    display(fig)
end



zqm  = [table_qm_f1[(2,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:47]
zold = [table_old[(4,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:47]

println(hcat(zqm,zold))

plot(Ic[2:end],zqm[2:end])
plot!(Ic[2:end],zold[2:end])
plot!(xscale=:log10,
yscale=:log10,
xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)

plot(Ic[2:end],abs.(1e3*(zqm[2:end]-zold[2:end])),
    label="6M – 3M",
    ylabel="Difference (μm)",
    xscale=:log10,
    yscale=:log10,
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    ylims=(1e-2,5))


fig = plot(ylabel=L"$z$ (mm)",
    xlabel=L"$I_{c}$ (A)")
for i in sort(unique(getindex.(ks,1)))
    zqm_F1  = [table_qm_f1[(i,0.150,0.001)][v][:z_max_smooth_spline_mm] for v=1:47]
    fig = plot!(Ic[10:end], zqm_F1[10:end],label=L"$n_{z}=%$(i)$")
end
plot!(xscale=:log10,
    yscale=:log10,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)

    sort(unique(getindex.(ks,2)))

data_exp = load(joinpath(@__DIR__,"20250820","data_processed.jld2"),"data")

fig = plot(xlabel="pixel")
zpix = 1e3*TheoreticalSimulation.pixel_coordinates(2560,1,6.5e-6)
for i =1:10
    plot!(fig,zpix,vec(mean(data_exp[:F1ProcessedImages][:,:,i,1],dims=1)), ls=:dash)
    plot!(fig,zpix,vec(mean(data_exp[:F2ProcessedImages][:,:,i,1],dims=1)), ls=:dot)
end
plot!(fig,zpix,vec(mean(data_exp[:F1ProcessedImages][:,:,:,1],dims=(3,1))), line=(:solid,:black))
plot!(fig,zpix,vec(mean(data_exp[:F2ProcessedImages][:,:,:,1],dims=(3,1))), line=(:solid,:black))
display(fig)

fig = plot(xlabel=L"$x$ (mm)")
xpix = 1e3*TheoreticalSimulation.pixel_coordinates(2160,4,6.5e-6)
for i =1:10
    plot!(fig,xpix,vec(mean(data_exp[:F1ProcessedImages][:,:,i,1],dims=2)), ls=:dash)
    plot!(fig,xpix,vec(mean(data_exp[:F2ProcessedImages][:,:,i,1],dims=2)), ls=:dot)
end
plot!(fig,xpix,vec(mean(data_exp[:F1ProcessedImages][:,:,:,1],dims=(3,2))), line=(:solid,:black))
plot!(fig,xpix,vec(mean(data_exp[:F2ProcessedImages][:,:,:,1],dims=(3,2))), line=(:solid,:black))
display(fig)




data_qm_profiles = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_6M","qm_6000000_screen_profiles.jld2"),"profiles")

joinpath("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_6M\\qm_6000000_screen_data.jld2")

data_i0 = jldopen("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_6M\\qm_6000000_screen_data.jld2", "r") do f
    f["alive"][:data][1]
end


jldopen("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_6M\\qm_6000000_screen_data.jld2","r") do f
    println(typeof(f["alive"]))
end
