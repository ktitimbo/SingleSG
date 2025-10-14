# Fitting experimental profile
# Kelvin Titimbo — California Institute of Technology — October 2025

#  Plotting Setup
# ENV["GKS_WSTYPE"] = "101"
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
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
# Numerical tools
using LinearAlgebra, DataStructures
using LsqFit
using BSplineKit
using Polynomials
using StatsBase
using Statistics, Distributions, StaticArrays
# using Alert
# Data manipulation
using OrderedCollections
using JLD2
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Set the working directory to the current location
cd(@__DIR__) ;
const OUTDIR    = joinpath(@__DIR__, "data_studies", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
const HOSTNAME = gethostname()
@info "Running on host" hostname = HOSTNAME
# Custom modules
include("./Modules/atoms.jl");
include("./Modules/samplings.jl");
include("./Modules/DataReading.jl");
include("./Modules/TheoreticalSimulation.jl");
# Propagate output settings to TheoreticalSimulation
TheoreticalSimulation.SAVE_FIG = SAVE_FIG;
TheoreticalSimulation.FIG_EXT  = FIG_EXT;
TheoreticalSimulation.OUTDIR   = OUTDIR;

@info "Run stamp initialized" RUN_STAMP = RUN_STAMP
println("\n\t\tRunning process on:\t $(RUN_STAMP) \n")

const ATOM        = "39K"  ;
## PHYSICAL CONSTANTS from NIST
# RSU : Relative Standard Uncertainty
const kb    = 1.380649e-23 ;       # Boltzmann constant (J/K)
const ħ     = 6.62607015e-34/2π ;  # Reduced Planck constant (J s)
const μ₀    = 1.25663706127e-6;    # Vacuum permeability (Tm/A)
const μB    = 9.2740100657e-24 ;   # Bohr magneton (J/T)
const γₑ    = -1.76085962784e11 ;  # Electron gyromagnetic ratio  (1/sT). Relative Standard Uncertainty = 3.0e-10
const μₑ    = 9.2847646917e-24 ;   # Electron magnetic moment (J/T). RSU = 3.0e-10
const Sspin = 1/2 ;                # Electron spin
const gₑ    = -2.00231930436092 ;  # Electron g-factor
## ATOM INFORMATION: 
K39_params = TheoreticalSimulation.AtomParams(ATOM);

# STERN--GERLACH EXPERIMENT
# Camera and pixel geometry : intrinsic properties
cam_pixelsize = 6.5e-6 ;  # Physical pixel size of camera [m]
nx_pixels , nz_pixels= (2160, 2560); # (Nx,Nz) pixels
println("""
***************************************************
CAMERA FEATURES
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm
***************************************************
""")
# Furnace
T_K = 273.15 + 205 ; # Furnace temperature (K)
# Furnace aperture
const x_furnace = 2.0e-3 ;
const z_furnace = 100e-6 ;
# Slit
const x_slit  = 4.0e-3 ;
const z_slit  = 300e-6 ;
# Propagation distances
const y_FurnaceToSlit = 224.0e-3 ;
const y_SlitToSG      = 44.0e-3 ;
const y_SG            = 7.0e-2 ;
const y_SGToScreen    = 32.0e-2 ;
# Connecting pipes
const R_tube = 35e-3/2 ; # Radius of the connecting pipe (m)
effusion_params = TheoreticalSimulation.BeamEffusionParams(x_furnace,z_furnace,x_slit,z_slit,y_FurnaceToSlit,T_K,K39_params);
println("""
***************************************************
SETUP FEATURES
    Temperature (K          : $(T_K)
    Furnace aperture (x,z)  : ($(1e3*x_furnace)mm , $(1e6*z_furnace)μm)
    Slit (x,z)              : ($(1e3*x_slit)mm , $(1e6*z_slit)μm)
    Furnace → Slit          : $(1e3*y_FurnaceToSlit)mm
    Slit → SG magnet        : $(1e3*y_SlitToSG)mm
    SG magnet               : $(1e3*y_SG)mm
    SG magnet → Screen      : $(1e3*y_SGToScreen)mm
    Tube radius             : $(1e3*R_tube)mm
***************************************************
""")
# Setting the variables for the module
TheoreticalSimulation.default_camera_pixel_size = cam_pixelsize;
TheoreticalSimulation.default_x_pixels          = nx_pixels;
TheoreticalSimulation.default_z_pixels          = nz_pixels;
TheoreticalSimulation.default_x_furnace         = x_furnace;
TheoreticalSimulation.default_z_furnace         = z_furnace;
TheoreticalSimulation.default_x_slit            = x_slit;
TheoreticalSimulation.default_z_slit            = z_slit;
TheoreticalSimulation.default_y_FurnaceToSlit   = y_FurnaceToSlit;
TheoreticalSimulation.default_y_SlitToSG        = y_SlitToSG;
TheoreticalSimulation.default_y_SG              = y_SG;
TheoreticalSimulation.default_y_SGToScreen      = y_SGToScreen;
TheoreticalSimulation.default_R_tube            = R_tube;

"""
    normalize_vec(v; by = :max, atol = 0)

Normalize `v` by `maximum(v)` (`:max`), `sum(v)` (`:sum`), or return unchanged (`:none`).
If the chosen denominator has magnitude ≤ `atol`, returns `v` unchanged.
"""
@inline function normalize_vec(v::AbstractArray; by::Symbol = :max, atol::Real = 0)
    denom = by === :max  ? maximum(v) :
            by === :sum  ? sum(v)      :
            by === :none ? one(eltype(v)) :
            throw(ArgumentError("`by` must be :max, :sum, or :none"))
    (by === :none || abs(denom) ≤ atol) && return v
    v ./ denom
end

"""
    std_sample(a, N)

Compute `a * sqrt(N*(N+1) / (3*(N-1)^2))` for odd `N = 2n+1`.
"""
@inline function std_sample(a::Real, N::Integer)
    @assert N ≥ 3 "N must be ≥ 3"
    @assert isodd(N) "N must be odd (N = 2n + 1)"
    a * sqrt(N*(N+1) / (3*(N-1)^2))
end

"""
    robust_se_and_cov(fit; rcond=1e-12, ridge=0.0, model=nothing, x=nothing, y=nothing, p̂=nothing)

Return `(se, cov)`.

- Try `vcov(fit)` first.
- Else build covariance from a Jacobian:
  * Prefer `fit.jacobian` (if present).
  * Else try `LsqFit.jacobian(fit)`.
  * Else, if `model, x, y, p̂` are provided, recompute J with ForwardDiff at `p̂`.
- If `ridge > 0`, use σ² * (J'J + λI)^(-1) for stabilization.
"""
function robust_se_and_cov(fit; rcond=1e-12, ridge=0.0, model=nothing, x=nothing, y=nothing, p̂=nothing)
    # 1) Try the built-in covariance
    try
        cov = LsqFit.vcov(fit)
        return sqrt.(diag(cov)), cov
    catch
        # 2) Get a Jacobian
        J = try
            getfield(fit, :jacobian)  # stored J from LsqFit
        catch
            nothing
        end
        if J === nothing
            @assert model !== nothing && x !== nothing && y !== nothing && p̂ !== nothing "Need (model,x,y,p̂) to recompute Jacobian"
            # recompute J at the solution p̂ as ∂/∂p (model(x,p) - y)
            # Uncomment ForwardDiff import above if you use this path.
            g(p) = model(x, p) .- y
            FD = try
                getfield(Main, :ForwardDiff)
            catch
                error("ForwardDiff is required to recompute the Jacobian; please `using ForwardDiff`.")
            end
            J = FD.jacobian(g, p̂)
        end

        # 3) Build a covariance from J
        r   = fit.resid
        p   = size(J, 2)
        dof = max(length(r) - p, 1)
        σ²  = sum(abs2, r) / dof

        if ridge > 0
            # cov ≈ σ² * (J'J + λI)^(-1)
            JTJ = J' * J              # CHANGED: explicit `*` (was `J'J`)
            # CHANGED: factorize with Cholesky for stability/speed; avoid explicit inv
            F = cholesky!(Symmetric(JTJ) + ridge * I)
            cov = σ² * (F \ I)
            return sqrt.(diag(cov)), cov
        else
            # SVD pseudo-inverse on singular directions
            S = svd(J)
            if isempty(S.S) || maximum(S.S) == 0
                return fill(NaN, p), zeros(p, p)
            end
            thr  = rcond * maximum(S.S)
            wInv = map(s -> s > thr ? 1/s : 0.0, S.S)
            cov  = σ² * (S.Vt' * Diagonal(wInv.^2) * S.Vt)
            return sqrt.(diag(cov)), cov
        end
    end
end


###############
# UTILITIES
###############

"""
    orthonormal_basis_on(z; n)

Build an orthonormal polynomial basis on `t = (z-μ)/σ` up to degree `n`.

Returns `(μ, σ, t, Qthin, R)` where:
- `X = [t.^0 t.^1 … t.^n]` (size npts×(n+1))
- `X = Qthin * R` with `Qthin` npts×(n+1), `R` (n+1)×(n+1) upper-triangular.
"""
function orthonormal_basis_on(z::AbstractVector{<:Real}; n::Integer)
    @assert n ≥ 0
    μ = (first(z) + last(z)) / 2
    σ = std(z)
    @assert σ > 0 "z has zero variance"
    invσ = inv(σ)

    t = @. (z - μ) * invσ
    X = hcat((t .^ k for k in 0:n)...)           # npts×(n+1)
    F = qr(X)
    k = n + 1
    R = F.R[1:k, :]                               # (n+1)×(n+1)
    Qthin = X / R                                  # thin Q via solve (no full Q)
    return μ, σ, t, Qthin, R
end

"""
    horner(z, c)

Evaluate a polynomial with coefficients `c` (c[1] + c[2] z + … + c[end] z^(m))
for scalar or array `z`. Works with Dual numbers.
"""
function horner(z::Union{Real,AbstractArray}, c::AbstractVector)
    m = length(c)
    m == 0 && return zero(eltype(c))
    acc = c[end]
    @inbounds for j in (m-1):-1:1
        acc = @. muladd(z, acc, c[j])
    end
    return acc
end

"""
    background_poly_any(z, c)

Convenience wrapper around `horner`. `c` is `[c0, c1, …, c_n]`.
"""
background_poly_any(z, c) = horner(z, c)

"""
    t_affine_poly(μ, σ)

Return a `Polynomial` p(z) such that `p(z) = (z - μ)/σ`.
(Requires Polynomials.jl)
"""
function t_affine_poly(μ::Real, σ::Real)
    @assert σ > 0
    return Polynomial([-μ/σ, 1/σ])
end

###########################################
# GENERALIZED ORTHONORMAL FITTER (ORDER n)
###########################################

"""
    fit_pdf_ortho_n(
        z, pdf_exp, pdf_th; 
        n::Integer=3, Qthin=nothing, R=nothing,  # optional precomputed basis on z
        w0::Real, A0::Real=1.0, d0::AbstractVector=zeros(n+1),
        progress_every::Int=10,
        rcond::Real=1e-12, ridge::Real=0.0
    )

Fit `pdf_exp(z)` with `A * ProbDist_convolved(z, pdf_th, w) + Pₙ(t)` where
`t = (z - μ)/σ`, and `Pₙ` is represented in an orthonormal basis with
coefficients `d` (columns of `Qthin`). If `Qthin`/`R` are not given, they are
built on the training grid `z`.

Parameters:
- `n`: polynomial degree (≥0). The number of polynomial params is `n+1`.
- Start params: `w0`, `A0`, `d0` (length n+1).
- Covariance is returned both in `d`-space and mapped to regular coefficients
  `c = R \\ d`.

Returns:
  (fit_data,
   params,            # Named tuple: w, A, c::Vector{Float64} (length n+1)
   param_se,          # Named tuple: δw, δA, δc::Vector{Float64} (length n+1)
   modelfun,          # x -> model(x, p̂)
   model_on_z,        # model(z, p̂)
   meta,              # evals, best_probe, μ, σ, n
   extras)            # d, R, cov_d, cov_c
"""
function fit_pdf_ortho_n(
    z::AbstractVector,
    pdf_exp::AbstractVector,
    pdf_th::AbstractVector;
    n::Integer=3,
    Qthin::Union{AbstractMatrix,Nothing}=nothing,
    R::Union{AbstractMatrix,Nothing}=nothing,
    w0::Real,
    A0::Real=1.0,
    d0::AbstractVector=zeros(n+1),
    progress_every::Int=10,
    rcond::Real=1e-12,
    ridge::Real=0.0,
    model_name::AbstractString="A*f(Ic,w;z)+Pₙ(z)",
    )
    @assert length(z) == length(pdf_exp) == length(pdf_th)
    @assert n ≥ 0
    @assert length(d0) == n+1


    # basis (on training grid)
    if Qthin === nothing || R === nothing
        μ, σ, t, Qthin_, R_ = orthonormal_basis_on(z; n=n)
    else
        μ = (first(z) + last(z)) / 2
        σ = std(z);  @assert σ > 0
        t = @. (z - μ) / σ
        Qthin_ = Qthin
        R_ = R
    end

    # (Optional) sanity checks to catch mismatches early
    @assert size(Qthin_, 1) == length(z)
    @assert size(Qthin_, 2) == n + 1
    @assert size(R_, 1) == n + 1 && size(R_, 2) == n + 1
    invσ = inv(σ)

    # helper: Float64 extraction that tolerates Duals or wrappers
    toflt(x) = try
        Float64(x)
    catch
        try getfield(Main,:ForwardDiff).value(x) |> Float64 catch
            try getfield(x,:value) |> Float64 catch; NaN end
        end
    end

    # parameter vector: [logw, logA, d0..dn]
    p0 = vcat(log(float(w0)), log(float(A0)), float.(d0))
    calls = Ref(0)
    best  = Ref((rss=Inf, p=copy(p0)))

    # model factory that can evaluate on any grid
    make_model = function (pdf_th_, μ_, invσ_; QthinT::Union{AbstractMatrix,Nothing}, RT)
        return function (zz::AbstractVector{<:Real}, p::AbstractVector{<:Real})
            logw, logA = p[1], p[2]
            d          = @view p[3:end]
            A, w       = exp(logA), exp(logw)

            conv = TheoreticalSimulation.ProbDist_convolved(zz, pdf_th_, w)

            poly = if QthinT !== nothing && (zz === z)   # fast path on training grid
                QthinT * d
            else
                c  = RT \ d                               # length n+1
                tt = @. (zz - μ_) * invσ_
                horner(tt, c)
            end

            @. A * conv + poly
        end
    end
    model = make_model(pdf_th, μ, invσ; QthinT=Qthin_, RT=R_)

    function pdf_model(zz::AbstractVector{<:Real}, p::AbstractVector{<:Real})
        yhat = model(zz, p)
        if progress_every > 0
            calls[] += 1
            if calls[] % progress_every == 0
                rss_val = toflt(sum(abs2, yhat .- pdf_exp))
                p_val   = map(toflt, p)
                if rss_val < best[].rss
                    best[] = (rss=rss_val, p=p_val)
                end
                @printf(stderr,
                    "eval %3d | rss≈%.6g \t| w≈%.6g A≈%.6g d₀…dₙ≈%s\n",
                    calls[], rss_val, exp(p_val[1]), exp(p_val[2]),
                    sprint(show, round.(p_val[3:end][1:min(end,6)], sigdigits=6)))
            end
        end
        return yhat
    end

    # bounds
    lower = vcat(log(1e-9),  log(1e-12), fill(-Inf, n+1))
    upper = vcat(log(1.0),   log(1e3),   fill( Inf, n+1))

    fit_data = LsqFit.curve_fit(pdf_model, z, pdf_exp, p0; autodiff=:forward, lower=lower, upper=upper)

    # unpack solution
    p̂    = coef(fit_data)
    ŵ, Â = exp(p̂[1]), exp(p̂[2])
    d̂     = @view p̂[3:end]                 # length n+1
    ĉ     = R_ \ d̂                         # map to ordinary coeffs (in t)

    # covariance in parameter space [logw, logA, d...]
    se_p, cov_p = robust_se_and_cov(fit_data; rcond=rcond, ridge=ridge)

    # propagate to c = R \\ d
    cov_d = (cov_p === nothing || isempty(cov_p)) ? nothing : cov_p[3:end, 3:end]
    cov_c, se_c = if cov_d === nothing
        (nothing, fill(NaN, n+1))
    else
        C = R_ \ (cov_d / R_')
        (C, sqrt.(diag(C)))
    end

    model_on_z = model(z, p̂)
    modelfun   = x -> model(x, p̂)

    params   = (w = ŵ, A = Â, c = collect(ĉ))
    param_se = (δw = ŵ * se_p[1], δA = Â * se_p[2], δc = collect(se_c))

    meta   = (evals=calls[], best_probe=(rss=best[].rss, p=best[].p))
    extras = (d = collect(d̂), R = R_, cov_d = cov_d, cov_c = cov_c)

    return fit_data, params, param_se, modelfun, model_on_z, meta, extras
end

# Select experimental data
wanted_data_dir = "20250919" ;
wanted_binning  = 2 ; 
wanted_smooth   = 0.01 ;

# Data loading
read_exp_info = DataReading.find_report_data(
        joinpath(@__DIR__, "analysis_data");
        wanted_data_dir=wanted_data_dir,
        wanted_binning=wanted_binning,
        wanted_smooth=wanted_smooth
);
[(String(k), getfield(read_exp_info, k)) for k in propertynames(read_exp_info)];
if isnothing(read_exp_info)
    @warn "No matching report found" wanted_data_dir wanted_binning wanted_smooth
else
    msg = join([
    "Imported experimental data:",
    "  directory     : $(read_exp_info.directory)",
    "  path          : $(read_exp_info.path)",
    "  data_dir      : $(read_exp_info.data_dir)",
    "  analysis_name : $(read_exp_info.name)",
    "  binning       : $(read_exp_info.binning)",
    "  smoothing     : $(read_exp_info.smoothing)",
    "  magnification : $(read_exp_info.magnification)",
    ], "\n")
    @info msg
end

exp_data = load(joinpath(read_exp_info.directory,"profiles_mean.jld2"))["profiles"]
Ic_sampled = exp_data[:Icoils];

STEP        = 10 ;
THRESH_A    = 0.020 ;
P_DEGREE    = 5 ;
ncols_bg    = P_DEGREE + 1 ;

chosen_currents_idx = sort(unique([firstindex(Ic_sampled);
        @view(findall(>(THRESH_A), Ic_sampled)[1:STEP:end]);
        @view(findall(>(THRESH_A), Ic_sampled)[end-2:end]);
        lastindex(Ic_sampled)
        ]
));

println("Target currents in A: (", 
        join(map(x -> @sprintf("%.3f", x), Ic_sampled[chosen_currents_idx]), ", "),
        ")"
)

norm_modes = (:none,:sum,:max) ;
magnification_factor = read_exp_info.magnification ;
λ0_exp     = 0.001 ;

z_exp   = (exp_data[:z_mm] .- exp_data[:Centroid_mm][1]) ./ magnification_factor
range_z    = floor(minimum([maximum(z_exp),abs(minimum(z_exp))]),digits=1);
nrange_z   = 20001;
z_theory  = collect(range(-range_z,range_z,length=nrange_z));

@assert isapprox(mean(z_theory), 0.0; atol= 10 * eps(float(range_z)) ) "μz=$(μz) not ~ 0 within atol=$(10 * eps(float(range_z)) )"
@assert isapprox(std(z_theory), std_sample(range_z, nrange_z); atol= eps(float(range_z))) "σz=$(σz) is not defined for a symmetric range"

μz, σz, t, Qthin, R = orthonormal_basis_on(z_theory; n=P_DEGREE);

tpoly = t_affine_poly(μz, σz) ;            # p(z) = (z-μ)/σ
rl = length(chosen_currents_idx) ;
nl = length(norm_modes) ;
fitting_params = zeros(nl,rl,1+2+ncols_bg); # (norm_modes x currents x [res, params])

# --- Column headers ---
# map ASCII digits to Unicode subscripts
const _sub = Dict(
    '0'=>'₀','1'=>'₁','2'=>'₂','3'=>'₃','4'=>'₄',
    '5'=>'₅','6'=>'₆','7'=>'₇','8'=>'₈','9'=>'₉','-'=>'₋'
)
sub(k::Integer) = join((_sub[c] for c in string(k)))  # "12" -> "₁₂"

hdr_top = Any[
    "Residuals",
    MultiColumn(2, "Theoretical PDF"),
    MultiColumn(ncols_bg, "Background P$(sub(P_DEGREE))(z)")
];
hdr_bot = vcat(["(exp-model)²", "A", "w [mm]"], ["c" * sub(k) for k in 0:P_DEGREE]);

@time for (n_idx, norm_mode) in enumerate(norm_modes)
    println("\n\n\t\t\tNORMALIZATION MODE = $(string(norm_mode))")
    p_prev = zeros(2+ncols_bg)
    for (j,i_idx) in enumerate(chosen_currents_idx)

        I0 = Ic_sampled[i_idx]
        println("\n\t\tANALYZING BACKGROUND FOR I₀=$(round(1000*I0,digits=3))mA")

        𝒢  = TheoreticalSimulation.GvsI(I0)
        _ℬ = abs.(TheoreticalSimulation.BvsI(I0))
        
        μ_eff = [TheoreticalSimulation.μF_effective(I0,v[1],v[2],K39_params) for v in TheoreticalSimulation.fmf_levels(K39_params,Fsel=1)]
        println("Effective magnetic moments (μF/μ₀) : ", μ_eff/μB)

        amp_exp = @view exp_data[:F1_profile][i_idx,:]

        Spl_exp = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp; 
                    weights=TheoreticalSimulation.compute_weights(z_exp, λ0_exp));

        pdf_exp = Spl_exp.(z_theory)
        pdf_exp = normalize_vec(pdf_exp; by=norm_mode)

        pdf_theory = mapreduce(μ -> TheoreticalSimulation.getProbDist_v3(μ, 𝒢, 1e-3 .* z_theory, K39_params, effusion_params),
                            +, μ_eff)    
        pdf_theory = normalize_vec(pdf_theory;by=norm_mode)

        # --- Quick diagnostic plots (raw vs spline vs closed-form) ---
        fig1= plot(z_exp , amp_exp, 
            label="Experiment (raw)", 
            seriestype=:scatter, 
            marker=(:hexagon,:white,2),
            xlabel=L"$z$ (mm)",
            ylabel="Intensity (au)",
            xlims=(-8,8),
        );
        fig2 = plot(z_theory, pdf_exp, 
            label="Experiment (spl. fit | $(norm_mode))", 
            line=(:black,2),
            xlabel=L"$z$ (mm)",
            ylabel="Intensity (au)",
            xlims=(-8,8),);
        plot!(z_theory , pdf_theory, label="Closed-form | $(norm_mode)", line=(:red,1.5));
        plot!(z_theory, TheoreticalSimulation.ProbDist_convolved(z_theory, pdf_theory, 150e-3), 
            label="Closed-form + Conv | $(norm_mode)", 
            line=(:dodgerblue2,1.2));

        fig=plot(fig1,fig2, layout=(2,1))
        display(fig)

        if j == 1
            A0 = 0.63
            w0 = 0.409
            d0 = zeros(ncols_bg)
        else
            A0 = exp(p_prev[2])
            w0 = exp(p_prev[1])
            d0 = p_prev[3:end]
        end

        # @time fit_data, params, δparams, modelfun, model_on_z , progress =
        #     fit_pdf(z_theory, pdf_exp, pdf_theory; w0=w0, A0=A0, c0=c0);

        # do the fit:
        @time fit_data, params, δparams, modelfun, model_on_z, meta, extras =
            fit_pdf_ortho_n(z_theory, pdf_exp, pdf_theory;
                            n=P_DEGREE, Qthin=Qthin, R=R, A0=A0, w0=w0, d0=d0,
                            progress_every=10)

        bg_poly = sum(params.c[k] * tpoly^(k-1) for k in 1:length(params.c))

        fig=plot(z_theory , pdf_exp, 
            label="Experiment $(wanted_data_dir)", 
            xlabel=L"$z$ (mm)",
            ylabel="Intensity (au)",
            seriestype=:scatter, 
            marker=(:hexagon,:white,1),
            markerstrokewidth=0.5,
            legend=:topleft,
            legendtitle=L"$I_{0}=%$(round(1000*I0,digits=3))\mathrm{mA}$",
            legendtitlefontsize=8,
            legendfontsize=8,);
        plot!(z_theory, pdf_theory, 
            label="ClosedForm",
            line=(:purple3,1) );
        plot!(z_theory, TheoreticalSimulation.ProbDist_convolved(z_theory, pdf_theory, params.w), 
            label="ClosedForm+Conv",
            line=(:dodgerblue2,1.2));
        plot!(z_theory,modelfun(z_theory), 
            label=L"Fit: $A f(I_{c},w;z) + P_{%$(P_DEGREE)}(z)$", 
            line=(:red,:dash,2),);
        plot!(z_theory,bg_poly.(z_theory),
            label="Background",
            line=(:green4,:dash,1.5));
        display(fig)
        savefig(fig,joinpath(OUTDIR,"$(wanted_data_dir)_$(@sprintf("%02d", i_idx))_$(string(norm_mode))_$(P_DEGREE).$(FIG_EXT)"))

        fitting_params[n_idx,j,:]  = vcat(meta.best_probe.rss,params.A,params.w,[bg_poly[dg] for dg in 0:P_DEGREE])

        pretty_table(
            fitting_params[n_idx,:,:];
            column_label_alignment      = :c,
            column_labels               = [hdr_top, hdr_bot],
            row_labels                  = round.(1000*Ic_sampled[chosen_currents_idx], sigdigits=4),
            formatters                  = [fmt__printf("%8.5e", [1]), fmt__printf("%8.5f", 2:3), fmt__printf("%8.5e", 4:(1+2+ncols_bg))],
            alignment                   = :c,
            equal_data_column_widths    = true,
            stubhead_label              = "I₀ [mA]",
            row_label_column_alignment  = :c,
            title                       = "FITTING ANALYSIS : $(wanted_data_dir)",
            table_format                = TextTableFormat(borders = text_table_borders__unicode_rounded),
            style                       = TextTableStyle(
                                                first_line_merged_column_label  = crayon"light_red bold",
                                                first_line_column_label         = crayon"yellow bold",
                                                column_label                    = crayon"yellow",
                                                table_border                    = crayon"blue bold",
                                                title                           = crayon"red bold"
                                            )
        )
        # update the carry for the next iteration
        p_prev = meta.best_probe.p
    end
end

starts      = collect(range(1; step=rl, length=nl)) ;
rg_labels   = Pair{Int,String}.(starts, string.(norm_modes)) ;

pretty_table(
    reduce(vcat, (@view fitting_params[i, :, :] for i in 1:nl));
    column_label_alignment      = :c,
    column_labels               = [hdr_top, hdr_bot],
    row_labels                  = repeat(round.(1000*Ic_sampled[chosen_currents_idx], sigdigits=4),3),
    formatters                  = [fmt__printf("%8.5e", [1]), fmt__printf("%8.5e", [2]), fmt__printf("%5.6f", [3]), fmt__printf("%8.5e", 4:(1+2+ncols_bg))],
    alignment                   = :c,
    equal_data_column_widths    = true,
    stubhead_label              = "I₀ [mA]",
    row_label_column_alignment  = :c,
    row_group_labels            = rg_labels,
    row_group_label_alignment   = :c,
    title                       = "FITTING ANALYSIS : $(wanted_data_dir)",
    table_format                = TextTableFormat(borders = text_table_borders__unicode_rounded),
    style                       = TextTableStyle(
                                        first_line_merged_column_label  = crayon"light_red bold",
                                        first_line_column_label         = crayon"yellow bold",
                                        column_label                    = crayon"yellow",
                                        table_border                    = crayon"blue bold",
                                        title                           = crayon"red bold"
                                    )
)

cols = palette(:darkrainbow, rl);
plot_list_bg = Vector{Plots.Plot}(undef, nl);

for (i,val) in enumerate(norm_modes)
    plot_list_bg[i] = plot(xlabel=L"$z$ (mm)", ylabel="Intensity (au)")
    for (j,idx) in enumerate(chosen_currents_idx)
        val_mA = 1000 * Ic_sampled[idx]

        plot!(z_theory,background_poly_any(z_theory, @view fitting_params[i,j,4:(1+2+ncols_bg)]),
            line=(cols[j],2),
            label= L"$I_{0}=" * @sprintf("%.1f", val_mA) * L"\,\mathrm{mA}$",
            )
    end
    plot!(legend=:bottom,
        legendtitle=string(val),
        legendtitlefontsize=8,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        legend_columns = 2,
    )

end
fig=plot(plot_list_bg..., 
    layout=(nl,1), 
    suptitle=L"Background $P_{%$(P_DEGREE)}$",
    left_margin=4mm,
    size=(500,700))
savefig(fig,joinpath(OUTDIR,"$(wanted_data_dir)_background_$(P_DEGREE).$(FIG_EXT)"))


plot_list_fit = Vector{Plots.Plot}(undef, nl)
for (i,val) in enumerate(norm_modes)
    fig=plot(xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)")
    for (j,i_idx) in enumerate(chosen_currents_idx)
        amp_exp = exp_data[:F1_profile][i_idx,:]
        Spl_exp = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp; weights=TheoreticalSimulation.compute_weights(z_exp, λ0_exp));
        pdf_exp = Spl_exp.(z_theory)
        pdf_exp = normalize_vec(pdf_exp; by=val)
        val_mA = 1000 * Ic_sampled[i_idx]
        plot!(z_theory, pdf_exp,
            line=(:solid,cols[j],2),
            label= L"%$(val) | $I_{0}=" * @sprintf("%.1f", val_mA) * L"\,\mathrm{mA}$",)

        𝒢  = TheoreticalSimulation.GvsI(Ic_sampled[i_idx])
            
        μ_eff = [TheoreticalSimulation.μF_effective(Ic_sampled[i_idx],v[1],v[2],K39_params) for v in TheoreticalSimulation.fmf_levels(K39_params,Fsel=1)]



        pdf_theory = mapreduce(μ -> TheoreticalSimulation.getProbDist_v3(μ, 𝒢, 1e-3 .* z_theory, K39_params, effusion_params),
                                +, μ_eff)    
        pdf_theory = normalize_vec(pdf_theory;by=val)
        f_fit = fitting_params[i,j,2] .* TheoreticalSimulation.ProbDist_convolved(z_theory, pdf_theory, fitting_params[i,j,3])+background_poly_any(z_theory, @view fitting_params[i,j,4:(1+2+ncols_bg)])
        plot!(z_theory,f_fit,
            line=(:dash,cols[j],1.5),
            label= false,)

        # plot!(z_theory,background_poly_any(z_theory, @view fitting_params[1,j, 3:(ncols_bg)]),
        #     line=(:dot,cols[j],1.5),
        #     label= false,
        #     )
    end
    plot!(legend=:topleft,
        legendfontsize=6,)

    plot_list_fit[i] = fig
end
fig=plot(plot_list_fit..., 
    layout=(nl,1), 
    suptitle=L"Fitting $P_{%$(P_DEGREE)}$",
    left_margin=4mm,
    size=(500,700))
savefig(fig,joinpath(OUTDIR,"$(wanted_data_dir)_fitting_$(P_DEGREE).$(FIG_EXT)"))

plot_list_exp = Vector{Plots.Plot}(undef, nl)
for (i,val) in enumerate(norm_modes)
    fig=plot(xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)")
    for (j,i_idx) in enumerate(chosen_currents_idx)
        amp_exp = exp_data[:F1_profile][i_idx,:]
        Spl_exp = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp; weights=TheoreticalSimulation.compute_weights(z_exp, λ0_exp));
        pdf_exp = Spl_exp.(z_theory)
        pdf_exp = normalize_vec(pdf_exp; by=val) - background_poly_any(z_theory, @view fitting_params[i,j,4:(1+2+ncols_bg)])
        val_mA = 1000 * Ic_sampled[i_idx]
        plot!(z_theory, pdf_exp,
            line=(:solid,cols[j],2),
            label= L"%$(val) | $I_{0}=" * @sprintf("%.1f", val_mA) * L"\,\mathrm{mA}$",)

        𝒢  = TheoreticalSimulation.GvsI(Ic_sampled[i_idx])
            
        μ_eff = [TheoreticalSimulation.μF_effective(Ic_sampled[i_idx],v[1],v[2],K39_params) for v in TheoreticalSimulation.fmf_levels(K39_params,Fsel=1)]

        pdf_theory = mapreduce(μ -> TheoreticalSimulation.getProbDist_v3(μ, 𝒢, 1e-3 .* z_theory, K39_params, effusion_params),
                                +, μ_eff)    
        pdf_theory = normalize_vec(pdf_theory;by=val)
        f_fit = fitting_params[i,j,2] .* TheoreticalSimulation.ProbDist_convolved(z_theory, pdf_theory, fitting_params[i,j,3])
        plot!(z_theory,f_fit,
            line=(:dash,cols[j],1.5),
            label= false,)

        # plot!(z_theory,background_poly_any(z_theory, @view fitting_params[1,j, 3:(ncols_bg)]),
        #     line=(:dot,cols[j],1.5),
        #     label= false,
        #     )
    end
    plot!(legend=:topleft,
        legendfontsize=6,)

    plot_list_exp[i] = fig
end
fig=plot(plot_list_exp..., 
    layout=(nl,1), 
    suptitle=L"Experiment $-$ Background $P_{%$(P_DEGREE)}$",
    left_margin=4mm,
    size=(500,700))

jldsave( joinpath(OUTDIR,"fitting_params_$(wanted_data_dir)_$(P_DEGREE).jld2"), 
        data = OrderedDict(
                :data_dir       => wanted_data_dir,
                :data_name      => read_exp_info.name,
                :nz_bin         => wanted_binning,
                :smooth_spline  => wanted_smooth,
                :magn_factor    => magnification_factor,
                :Icoil_A        => Ic_sampled[chosen_currents_idx],
                :normalization  => norm_modes,
                :fit_params     => fitting_params))

aa = load(joinpath(OUTDIR,"fitting_params_20250919_3.jld2"))["data"]


plot(Ic_sampled[chosen_currents_idx], fitting_params[1,:,1])
plot(Ic_sampled[chosen_currents_idx], fitting_params[1,:,2])
plot(Ic_sampled[chosen_currents_idx], fitting_params[1,:,3])
plot(Ic_sampled[chosen_currents_idx], fitting_params[1,:,4])
plot(Ic_sampled[chosen_currents_idx], fitting_params[1,:,5])
plot(Ic_sampled[chosen_currents_idx], fitting_params[1,:,6])




#########################################################################################
#########################################################################################

T_END = Dates.now()
T_RUN = Dates.canonicalize(T_END-T_START)


report = """
***************************************************
EXPERIMENT
    Single Stern–Gerlach Experiment
    Output directory            : $(OUTDIR)
    Run label                   : $(RUN_STAMP)
    

EXPERIMENT ANALYSIS PROPERTIES    
    Data directory              : $(wanted_data_dir)
    Analysis name               : $(read_exp_info.name)
    Analysis Binning            : $(wanted_binning)
    Analysis spline smoothing   : $(wanted_smooth)
    Magnification factor        : $magnification_factor

CAMERA FEATURES
    Number of pixels            : $(nx_pixels) × $(nz_pixels)
    Pixel size                  : $(1e6*cam_pixelsize) μm

FITTING INFORMATION
    Currents                    : $(round.(Ic_sampled[chosen_currents_idx],sigdigits=4))
    Normalization mode          : $(norm_modes)
    No z-divisions              : $(nrange_z)
    Coordinate z range (mm)     : ($(-range_z), $(range_z))
    Polynomial degree           : $(P_DEGREE)

CODE
    Code name                   : $(PROGRAM_FILE),
    Start date                  : $(T_START)
    End data                    : $(T_END)
    Run time                    : $(T_RUN)
    Hostname                    : $(HOSTNAME)

***************************************************
"""

# Print to terminal
println(report)

# Save to file
open(joinpath(OUTDIR,"analysis_report.txt"), "w") do io
    write(io, report)
end

println("Experiment analysis finished!")
alert("Experiment analysis finished!")