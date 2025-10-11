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
using Alert
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
    robust_stderror(fit; rcond=1e-12, ridge=0.0)

Return parameter standard errors for `fit`. Uses `stderror(fit)` if available;
otherwise falls back to a Jacobian-based covariance with either SVD
(pseudoinverse, threshold `rcond`) or ridge regularization (`ridge > 0`).
"""
function robust_stderror(fit; rcond=1e-12, ridge=0.0)
    # 1) Try the built-in way first
    try
        return stderror(fit)
    catch
        # 2) Fallback via Jacobian-based covariance
        J = try
            fit.jacobian                # prefer the stored Jacobian
        catch
            nothing
        end
        J === nothing && error("No Jacobian stored on the fit result; recompute it or pass (model,x,y,p̂).")

        r   = fit.resid
        p   = size(J, 2)
        dof = max(length(r) - p, 1)
        σ²  = sum(abs2, r) / dof

        cov = if ridge > 0
            # cov ≈ σ² * (J'J + λI)^(-1)
            JTJ = J' * J
            F   = cholesky!(Symmetric(JTJ) + ridge * I)
            σ² * (F \ I)  # solve for the full inverse once; small p so OK
        else
            # cov ≈ σ² * V * diag(1/s^2) * V'
            S = svd(J)
            (isempty(S.S) || maximum(S.S) == 0) && return fill(NaN, p)
            thr  = rcond * maximum(S.S)
            wInv = map(s -> s > thr ? 1/s : 0.0, S.S)
            σ² * (S.Vt' * Diagonal(wInv.^2) * S.Vt)
        end

        return sqrt.(diag(cov))
    end
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

"""
    fit_pdf(z, pdf_exp, pdf_theory; w0, A0=1.0, c0=[0,0,0,0], progress_every=10)

Fit `pdf_exp(z)` to `A * ProbDist_convolved(z, pdf_theory, w) + cubic((z-μz)/σz)`.
Returns: `(fit_data, params, param_se, modelfun, model_on_z, meta)`.
"""
function fit_pdf(
    z::AbstractVector,
    pdf_exp::AbstractVector,
    pdf_theory::AbstractVector;
    w0::Float64,
    A0::Float64 = 1.0,
    c0::AbstractVector = [0.0, 0.0, 0.0, 0.0],
    progress_every::Int = 10,)

    @assert length(z) == length(pdf_exp) == length(pdf_theory)
    @assert length(c0) == 4  

    μz = (first(z) + last(z)) / 2
    σz = std(z)
    @assert σz > 0 "z has zero variance"
    invσz = inv(σz);

    # Helper so printing works even when numbers are Duals
    toflt(x) = try
        Float64(x)
    catch
        try getfield(Main, :ForwardDiff).value(x) |> Float64 catch
            try getfield(x, :value) |> Float64 catch; NaN end
        end
    end

    # Parameters: p = [logw, logA, c0, c1, c2, c3]
    p0 = [
        log(float(w0)),
        log(float(A0)),
        float(c0[1]),
        float(c0[2]),
        float(c0[3]),
        float(c0[4]),
    ]

    """
    make_model(pdf_theory)

    Return `model(zz, p)` evaluating
    A * ProbDist_convolved(zz, pdf_theory, w) + cubic((zz-μz)*invσz; c0..c3).
    p = [logw, logA, c0, c1, c2, c3]
    """
    make_model(pdf_theory) = function (zz::AbstractVector{<:Real}, p::AbstractVector{<:Real})
        logw, logA, c₀, c₁, c₂, c₃ = p
        A, w  = exp(logA), exp(logw)
        conv  = TheoreticalSimulation.ProbDist_convolved(zz, pdf_theory, w)     # alloc-returning
        tt_z = (zz .- μz) .* invσz
        poly  = @. muladd(tt_z, muladd(tt_z, muladd(tt_z, c₃, c₂), c₁), c₀)
        @. A * conv + poly
    end

    calls = Ref(0)
    best  = Ref((rss = Inf, p = copy(p0)))  # track best (numeric) probe

    # Create the model ONCE; then call it inside pdfmix_model
    model = make_model(pdf_theory)

    # --- Model: LsqFit expects model(x, p) ---
    function pdf_model(zz::AbstractVector{<:Real}, p::AbstractVector{<:Real})

        yhat = model(zz, p)

        if progress_every > 0
            calls[] += 1
            if calls[] % progress_every == 0
                rss_val = toflt(sum(abs2, yhat .- pdf_exp))
                p_val   = map(toflt, p)
                if rss_val < best[].rss
                    best[] = (rss = rss_val, p = p_val)
                end
                @printf(
                    stderr,
                    "eval %6d | rss≈%.6g \t| w≈%.6g\t A≈%.6g\t c≈(%.3g, %.3g, %.3g, %.3g)\n",
                    calls[], rss_val,
                    exp(p_val[1]), exp(p_val[2]),
                    p_val[3], p_val[4], p_val[5], p_val[6],
                )
            end
        end
        return yhat
    end


    lower = [log(1e-9), log(1e-12), -Inf, -Inf, -Inf, -Inf]
    upper = [log(1),    log(1e3),    Inf,  Inf,  Inf,  Inf]

    fit_data = LsqFit.curve_fit(pdf_model, z, pdf_exp, p0; autodiff=:forward, lower=lower, upper=upper)

    p̂ = coef(fit_data)
    logw, logA, c₀, c₁, c₂, c₃ = p̂
    A, w = exp(logA), exp(logw)

    se = robust_stderror(fit_data)
    sw, sA = w * se[1], A * se[2]
    sc0, sc1, sc2, sc3 = se[3], se[4], se[5], se[6]

    model_on_z = model(z, p̂)

    # Callable on arbitrary grids
    modelfun = x -> model(x, p̂)

    return fit_data,
           (w = w, A = A, c0 = c₀, c1 = c₁, c2 = c₂, c3 = c₃),
           (w = sw, A = sA, c0 = sc0, c1 = sc1, c2 = sc2, c3 = sc3),
           modelfun,
           model_on_z,
           (evals = calls[],
            best_probe = (rss = best[].rss,
                          w = exp(best[].p[1]), A = exp(best[].p[2]),
                          c0 = best[].p[3], c1 = best[].p[4], c2 = best[].p[5], c3 = best[].p[6]))
end


"""
    fit_pdf_ortho(z, pdf_exp, pdf_theory; Qthin, R11, w0, A0=1.0, d0=[0,0,0,0], progress_every=10)

Fit `pdf_exp(z)` to `A * ProbDist_convolved(z, pdf_theory, w) + cubic(t)` with `t = (z-μz)/σz`,
parameterizing the cubic in an orthonormal basis (columns of `Qthin`).
Returns `(fit_data, params, param_se, modelfun, model_on_z, meta, extras)`.
"""
function fit_pdf_ortho(
    z::AbstractVector,
    pdf_exp::AbstractVector,
    pdf_theory::AbstractVector;
    Qthin::AbstractMatrix,        # <-- n×4 dense thin Q
    R11::AbstractMatrix,          # <-- 4×4 R block
    w0::Float64,
    A0::Float64 = 1.0,
    d0::AbstractVector = [0,0,0,0],
    progress_every::Int = 10,)

    @assert size(Qthin,2) == 4
    @assert size(Qthin,1) == length(z)
    @assert size(R11) == (4,4)
    @assert length(d0) == 4

    μz  = (first(z) + last(z)) / 2  
    σz  = std(z) 
    @assert σz > 0 "z has zero variance"   
    invσz = inv(σz)    

    toflt(x) = try
        Float64(x)
    catch
        try getfield(Main,:ForwardDiff).value(x) |> Float64 catch
            try getfield(x,:value) |> Float64 catch; NaN end
        end
    end

    p0 = [log(float(w0)), log(float(A0)), float.(d0)...]
    calls = Ref(0)
    best  = Ref((rss=Inf, p=copy(p0)))

    # make_model_ortho: returns a model(zz, p) that works on any evaluation grid
    # Fast path: if zz === z (training grid), use Qthin * d.
    # Else: convert orthonormal coeffs d -> cubic coeffs c via R11 \ d and
    #       evaluate cubic in standardized t = (zz - μz) * invσz.
    make_model_ortho = function (pdf_theory_, μz_, invσz_; Qthin_::Union{AbstractMatrix,Nothing}, R11_)
        return function (zz::AbstractVector{<:Real}, p::AbstractVector{<:Real})
            logw, logA = p[1], p[2]
            d          = @view p[3:6]                   # works with Duals
            A, w       = exp(logA), exp(logw)

            conv = TheoreticalSimulation.ProbDist_convolved(zz, pdf_theory_, w)

            poly = if Qthin_ !== nothing && (zz === z)  # fast path on training grid
                Qthin_ * d
            else
                c  = R11_ \ d                           # c0..c3
                tt = @. (zz - μz_) * invσz_
                @. muladd(tt, muladd(tt, muladd(tt, c[4], c[3]), c[2]), c[1])
            end

            @. A * conv + poly
        end
    end
    # Build the reusable model once (closes over μz, invσz, pdf_theory, Qthin, R11)
    model = make_model_ortho(pdf_theory, μz, invσz; Qthin_=Qthin, R11_=R11)
    # ============================================================================

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
                    "eval %6d | rss≈%.6g \t| w≈%.6g\t A≈%.6g\t d≈(%.3g, %.3g, %.3g, %.3g)\n",
                    calls[], rss_val, exp(p_val[1]), exp(p_val[2]),
                    p_val[3], p_val[4], p_val[5], p_val[6])
            end
        end
        yhat
    end

    lower = [log(1e-9), log(1e-12), -Inf, -Inf, -Inf, -Inf]
    upper = [log(1),    log(1e3),    Inf,  Inf,  Inf,  Inf]

    fit_data = LsqFit.curve_fit(pdf_model, z, pdf_exp, p0; autodiff=:forward, lower=lower, upper=upper)

    p̂ = coef(fit_data)
    logw, logA = p̂[1], p̂[2]
    dvec       = @view p̂[3:6]
    w, A       = exp(logw), exp(logA)

    # Map orthonormal coeffs d -> regular cubic-in-t coeffs c:  d = R11 * c  => c = R11 \ d
    c = R11 \ dvec

    # Standard errors: for p = [logw, logA, d0..d3]
    se, cov_p = robust_se_and_cov(fit_data; rcond=1e-12, ridge=0.0)  # set ridge>0 if needed

    # Transform covariance from d to c = R11 \ d
    @views begin                              # CHANGED: @views to avoid copies
        if cov_p !== nothing && !isempty(cov_p)
            cov_d = cov_p[3:6, 3:6]
            # Stable transforms (triangular solves under the hood)
            cov_c = R11 \ (cov_d / R11')      # CHANGED: explicit solve form (no inv)
            se_c  = sqrt.(diag(cov_c))
        else
            cov_c = nothing
            se_c  = fill(NaN, 4)
        end
    end

    model_on_z = model(z, p̂)
    modelfun   = x -> model(x, p̂)

    return fit_data,
           (w=w, A=A, c0=c[1], c1=c[2], c2=c[3], c3=c[4]),
           (w=w*se[1], A=A*se[2], c0=se_c[1], c1=se_c[2], c2=se_c[3], c3=se_c[4]),
           modelfun,
           model_on_z,
           (evals=calls[], best_probe=(rss=best[].rss, w=exp(best[].p[1]), A=exp(best[].p[2]),
                                       d0=best[].p[3], d1=best[].p[4], d2=best[].p[5], d3=best[].p[6])),
           (d=dvec, R=R11, cov_c=cov_c)
end

"""
    background_poly(z, c)

Evaluate the cubic background `c0 + c1*z + c2*z^2 + c3*z^3`.
`c` must have length 4 (ordered as `[c0, c1, c2, c3]`). Works with scalars or arrays `z`.
"""
function background_poly(z::Union{Real,AbstractArray}, c::AbstractVector{<:Real})
    @assert length(c) == 4
    c0, c1, c2, c3 = c
    @. muladd(z, muladd(z, muladd(z, c3, c2), c1), c0)
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

STEP        = 10
THRESH_A    = 0.020

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

# ---- Axes / normalization ----
norm_modes = (:none,:sum,:max) ;
range_z    = 6.1;
nrange_z   = 20001;
λ0_exp     = 0.001;

z_exp   = (exp_data[:z_mm] .- exp_data[:Centroid_mm][1]) ./ read_exp_info.magnification
z_theory  = collect(range(-range_z,range_z,length=nrange_z));

μz, σz = mean(z_theory) , std(z_theory);
@assert isapprox(μz, 0.0; atol= 10 * eps(float(range_z)) ) "μz=$(μz) not ~ 0 within atol=$(10 * eps(float(range_z)) )"
@assert isapprox(σz, std_sample(range_z, nrange_z); atol= eps(float(range_z))) "σz=$(σz) is not defined for a symmetric range"
# --- Orthonormal polynomial basis (QR of Vandermonde on t) ---
t = (z_theory .- μz) ./ σz
X = hcat(ones(length(t)), t, t.^2, t.^3)   # n×4 Vandermonde on t

F    = qr(X)              # QR factorization
k    = size(X, 2)         # 4
R11  = F.R[1:k, :]        # 4×4 upper-triangular
Qthin = X / R11           # n×4 thin Q via solve (no full Q)
# -------------------------------------------------------------------

tpoly = Polynomial([0, 1/σz]) ;       # t = (-μ/σ) + (1/σ) z


rl = length(chosen_currents_idx) 
nl = length(norm_modes)
fitting_params = zeros(nl,rl,6);

@time for (n_idx, norm_mode) in enumerate(norm_modes)
    println("\n\n\t\t\tNORMALIZATION MODE = $(string(norm_mode))")
    for (j,i_idx) in enumerate(chosen_currents_idx)

        I0 = Ic_sampled[i_idx]
        println("\n\t\tANALYZING BACKGROUND FOR I₀=$(round(1000*I0,digits=3))mA")

        𝒢  = TheoreticalSimulation.GvsI(I0)
        _ℬ = abs.(TheoreticalSimulation.BvsI(I0))
        
        μ_eff = [TheoreticalSimulation.μF_effective(I0,v[1],v[2],K39_params) for v in TheoreticalSimulation.fmf_levels(K39_params,Fsel=1)]

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
            w0, A0, c0  = 409.417e-3, 0.63, [0.529, -0.0143, -0.0984, 0.0118];
        else

            w0 = fitting_params[n_idx,j-1,2]
            A0 = fitting_params[n_idx,j-1,1]
            c0 = @view fitting_params[n_idx,j-1, 3:6]
        end

        # @time fit_data, params, δparams, modelfun, model_on_z , progress =
        #     fit_pdf(z_theory, pdf_exp, pdf_theory; w0=w0, A0=A0, c0=c0);

        @time fit_data, params, δparams, modelfun, model_on_z, progress, ortho_info =
            fit_pdf_ortho(z_theory, pdf_exp, pdf_theory;
                        Qthin=Qthin, R11=R11, w0=w0, A0=A0, d0=zeros(4), progress_every=10)

        bg_poly = params.c0 + params.c1*tpoly + params.c2*tpoly^2 + params.c3*tpoly^3

        fig=plot(z_theory , pdf_exp, 
            label="Experiment", 
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
            label=L"Fit: $A f(I_{c},w;z) + P_{3}(z)$", 
            line=(:red,:dash,2),);
        plot!(z_theory,bg_poly.(z_theory),
            label="Background",
            line=(:green4,:dash,1.5));
        display(fig)
        savefig(fig,joinpath(OUTDIR,"$(wanted_data_dir)_$(@sprintf("%02d", i_idx))_$(string(norm_mode)).$(FIG_EXT)"))

        fitting_params[n_idx,j,:]  = vcat(params.A,params.w,[bg_poly[dg] for dg in 0:3])

        pretty_table(
            fitting_params[n_idx,:,:];
            column_label_alignment      = :c,
            column_labels               = [[MultiColumn(2, "Theoretical PDF"), MultiColumn(4, "Background P₃(z)") ],
                                            ["A", "w [mm]", "c₀", "c₁", "c₂", "c₃"]],
            row_labels                  = round.(1000*Ic_sampled[chosen_currents_idx], sigdigits=4),
            formatters                  = [fmt__printf("%8.5f", 1:2), fmt__printf("%8.4f", [3]), fmt__printf("%8.5e", 4:6)],
            alignment                   = :c,
            equal_data_column_widths    = true,
            stubhead_label              = "I₀ [mA]",
            row_label_column_alignment  = :c,
            title                       = "FITTING ANALYSIS",
            table_format                = TextTableFormat(borders = text_table_borders__unicode_rounded),
            style                       = TextTableStyle(
                                                first_line_merged_column_label  = crayon"light_red bold",
                                                first_line_column_label         = crayon"yellow bold",
                                                column_label                    = crayon"yellow",
                                                table_border                    = crayon"blue bold",
                                                title                           = crayon"red bold"
                                            )
        )

    end
end


starts      = collect(range(1; step=rl, length=nl)) 
rg_labels   = Pair{Int,String}.(starts, string.(norm_modes))

pretty_table(
    reduce(vcat, (@view fitting_params[i, :, :] for i in 1:nl));
    column_label_alignment      = :c,
    column_labels               = [[MultiColumn(2, "Theoretical PDF"), MultiColumn(4, "Background P₃(z)") ],
                                    ["A", "w [mm]", "c₀", "c₁", "c₂", "c₃"]],
    row_labels                  = repeat(round.(1000*Ic_sampled[chosen_currents_idx], sigdigits=4),3),
    formatters                  = [fmt__printf("%8.5f", 1:2), fmt__printf("%8.4f", [3]), fmt__printf("%8.5e", 4:6)],
    alignment                   = :c,
    equal_data_column_widths    = true,
    stubhead_label              = "I₀ [mA]",
    row_label_column_alignment  = :c,
    row_group_labels            = rg_labels,
    row_group_label_alignment   = :c,
    title                       = "FITTING ANALYSIS",
    table_format                = TextTableFormat(borders = text_table_borders__unicode_rounded),
    style                       = TextTableStyle(
                                        first_line_merged_column_label = crayon"light_red bold",
                                        first_line_column_label = crayon"yellow bold",
                                        column_label  = crayon"yellow",
                                        table_border  = crayon"blue bold",
                                        title = crayon"red bold"
                                    )
)

cols = palette(:darkrainbow, rl)
plot_list = Vector{Plots.Plot}(undef, nl)

for (i,val) in enumerate(norm_modes)

    plot_list[i] = plot(xlabel=L"$z$ (mm)", ylabel="Intensity (au)")
    for (j,idx) in enumerate(chosen_currents_idx)
        val_mA = 1000 * Ic_sampled[idx]

        plot!(z_theory,background_poly(z_theory, @view fitting_params[i,j,3:6]),
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

plot(plot_list..., 
layout=(nl,1), 
suptitle="Background",
size=(500,700))



fig=plot(xlabel=L"$z$ (mm)",
    ylabel="Intensity (au)")
for (j,i_idx) in enumerate(chosen_currents_idx)
    amp_exp = exp_data[:F1_profile][i_idx,:]
    Spl_exp = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp; weights=TheoreticalSimulation.compute_weights(z_exp, λ0_exp));
    pdf_exp = Spl_exp.(z_theory)
    pdf_exp = normalize_vec(pdf_exp; by=:none)
    val_mA = 1000 * Ic_sampled[i_idx]
    plot!(z_theory, pdf_exp,
        line=(:solid,cols[j],2),
        label= L"$I_{0}=" * @sprintf("%.1f", val_mA) * L"\,\mathrm{mA}$",)
    

    plot!(z_theory,background_poly(z_theory, @view fitting_params[1,j, 3:6]),
        line=(:dash,cols[j],1.5),
        label= false,
        )
end
plot!(legend=:best,)
savefig(fig,joinpath(OUTDIR,"summary_$(wanted_data_dir)_$(string(norm_mode)).$(FIG_EXT)"))


jldsave( joinpath(OUTDIR,"fitting_params_$(wanted_data_dir)_$(string(norm_mode)).jld2"), data = fitting_params)

aa = load(joinpath(OUTDIR,"fitting_params_20250919_max.jld2"))["data"]


plot(Ic_sampled[chosen_currents_idx], fitting_params[:,1])
plot(Ic_sampled[chosen_currents_idx], fitting_params[:,2])
plot(Ic_sampled[chosen_currents_idx], fitting_params[:,3])
plot(Ic_sampled[chosen_currents_idx], fitting_params[:,4])
plot(Ic_sampled[chosen_currents_idx], fitting_params[:,5])
plot(Ic_sampled[chosen_currents_idx], fitting_params[:,6])


# [TheoreticalSimulation.μF_effective(I0,v[1],v[2],K39_params) for v in TheoreticalSimulation.fmf_levels(K39_params,Fsel=2)][end]
