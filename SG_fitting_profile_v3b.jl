# Fitting experimental profile
# Kelvin Titimbo — California Institute of Technology — October 2025
# Fitting for common profiles: shift z-axis before interpolation

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
    anypoly_eval(z, c)

Convenience wrapper around `horner`. `c` is `[c0, c1, …, c_n]`.
"""
anypoly_eval(z, c) = horner(z, c)

"""
    t_affine_poly(μ, σ)

Return a `Polynomial` p(z) such that `p(z) = (z - μ)/σ`.
(Requires Polynomials.jl)
"""
function t_affine_poly(μ::Real, σ::Real)
    @assert σ > 0
    return Polynomial([-μ/σ, 1/σ])
end

function bg_function(z::AbstractVector,c::AbstractVector)
    @assert isapprox(mean(z), 0.0; atol= 10 * eps() ) "μz not ~ 0 within atol=$(10 * eps())"
    μ   = mean(z)
    σ   = std(z)
    n   = length(c)
    t   = Polynomial([μ/σ, 1/σ]) # t=(z-μ)/σ
    bg  = sum(c[k] * t^(k-1) for k in 1:n)
    return bg
end

function predict_profile(z::AbstractVector,profile_theory::AbstractVector,A::Float64,w::Float64,c::AbstractVector)
    @assert length(z) == length(profile_theory) "z and pdf must have the same length"
    bg = anypoly_eval(z,c)
    first_term = A * TheoreticalSimulation.ProbDist_convolved(z, profile_theory, w)
    return first_term + bg
end

###########################################
# GENERALIZED ORTHONORMAL FITTER (ORDER n)
###########################################

function fit_pdf_joint(
    z_list::Vector{<:AbstractVector},
    y_list::Vector{<:AbstractVector},
    pdf_th_list::Vector{<:AbstractVector};
    n::Integer,
    Q_list::Vector{<:AbstractMatrix},
    R_list::Vector{<:AbstractMatrix},
    μ_list::Vector{<:Real},
    σ_list::Vector{<:Real},
    # inits / fixed
    w0::Real, A0::Real=1.0, d0=nothing,
    w_mode::Symbol = :global,            # :per_profile, :global, :fixed
    A_mode::Symbol = :per_profile,       # :per_profile, :global, :fixed
    d_mode::Symbol = :per_profile,       # NEW: :per_profile, :global
    w_fixed::Real = w0,                  # used if w_mode == :fixed
    A_fixed::Real = A0,                  # used if A_mode == :fixed
    progress_every::Int=25,
    rcond::Real=1e-12, ridge::Real=0.0,)
    M = length(z_list)
    @assert length(y_list) == M == length(pdf_th_list) == length(Q_list) ==
            length(R_list) == length(μ_list) == length(σ_list)
    @assert n ≥ 0
    @assert w_mode in (:per_profile, :global, :fixed)
    @assert A_mode in (:per_profile, :global, :fixed)
    @assert d_mode in (:per_profile, :global)

    # -------- ensure basis matches current n and grids --------
    ncoef = n + 1
    QL = Vector{Matrix{Float64}}(undef, M)
    RL = Vector{Matrix{Float64}}(undef, M)
    μL = Vector{Float64}(undef, M)
    σL = Vector{Float64}(undef, M)
    for i in 1:M
        needs_rebuild = size(Q_list[i],1) != length(z_list[i]) ||
                        size(Q_list[i],2) != ncoef ||
                        size(R_list[i],1) != ncoef ||
                        size(R_list[i],2) != ncoef
        if needs_rebuild
            μi, σi, _t, Qi, Ri = orthonormal_basis_on(z_list[i]; n=n)
            μL[i], σL[i], QL[i], RL[i] = μi, σi, Qi, Ri
        else
            μL[i], σL[i], QL[i], RL[i] = μ_list[i], σ_list[i], Q_list[i], R_list[i]
        end
    end

    # -------- init d / c --------
    # For :per_profile we store dᵢ (orthonormal-basis coeffs). For :global we store a single c (monomial in t).
    d0vec = d0 === nothing ? zeros(ncoef) : collect(float.(d0))
    @assert length(d0vec) == ncoef

    # If global, prefer initializing c0 from the first profile's R: c0 = R⁻¹ d0
    c0vec = RL[1] \ d0vec

    # -------- dynamic parameter packing --------
    # order: [ maybe logw(_i), maybe A(_i), then d/c block(s) ]
    idx_logw_global = nothing::Union{Int,Nothing}
    idx_logw_vec    = nothing::Union{Vector{Int},Nothing}
    idx_A_global    = nothing::Union{Int,Nothing}
    idx_A_vec       = nothing::Union{Vector{Int},Nothing}

    k = 1
    if w_mode == :global
        idx_logw_global = k; k += 1
    elseif w_mode == :per_profile
        idx_logw_vec = collect(k:(k+M-1)); k += M
    end

    if A_mode == :global
        idx_A_global = k; k += 1
    elseif A_mode == :per_profile
        idx_A_vec = collect(k:(k+M-1)); k += M
    end

    # d/c indices
    idx_c_global = nothing::Union{UnitRange{Int},Nothing}
    d_ranges     = nothing::Union{Vector{UnitRange{Int}},Nothing}

    if d_mode == :global
        idx_c_global = k:(k+ncoef-1)
        k += ncoef
    else
        d_start  = k
        d_ranges = [ (d_start + (i-1)*ncoef) : (d_start + i*ncoef - 1) for i in 1:M ]
        k += M*ncoef
    end
    total_len = k - 1

    # initial parameter vector
    p0 = Vector{Float64}(undef, total_len)
    if w_mode == :global
        p0[idx_logw_global] = log(float(w0))
    elseif w_mode == :per_profile
        @inbounds for i in 1:M
            p0[idx_logw_vec[i]] = log(float(w0))
        end
    end
    if A_mode == :global
        p0[idx_A_global] = float(A0)
    elseif A_mode == :per_profile
        @inbounds for i in 1:M
            p0[idx_A_vec[i]] = float(A0)
        end
    end
    if d_mode == :global
        p0[idx_c_global] .= c0vec
    else
        @inbounds for i in 1:M
            p0[d_ranges[i]] .= d0vec
        end
    end

    # -------- utilities --------
    calls    = Ref(0)
    best_rss = Ref{Float64}(Inf)
    best_p   = Ref{Vector{Float64}}(copy(p0))

    @inline function toflt(x)
        if Base.hasproperty(x, :value)
            return Float64(getfield(x, :value))
        elseif x isa Real
            return Float64(x)
        else
            return Float64(float(x))
        end
    end
    @inline promote_to_p(val, p) = one(p[1]) * val

    get_logw = function (i::Int, p)
        if w_mode == :global
            return p[idx_logw_global]
        elseif w_mode == :per_profile
            return p[idx_logw_vec[i]]
        else
            return promote_to_p(log(w_fixed), p)
        end
    end
    get_A = function (i::Int, p)
        if A_mode == :global
            return p[idx_A_global]
        elseif A_mode == :per_profile
            return p[idx_A_vec[i]]
        else
            return promote_to_p(A_fixed, p)
        end
    end

    # Accessors for d/c
    get_c = function (i::Int, p)
        if d_mode == :global
            return view(p, idx_c_global)  # same for all i
        else
            # cᵢ = Rᵢ⁻¹ dᵢ
            dview = view(p, d_ranges[i])
            return RL[i] \ dview
        end
    end
    get_d = function (i::Int, p)
        if d_mode == :global
            # dᵢ = Rᵢ c (when Q basis on grid is needed)
            cview = view(p, idx_c_global)
            return RL[i] * cview
        else
            return view(p, d_ranges[i])
        end
    end

    # -------- model pieces --------
    function model_i(i::Int, zz, p)
        logw = get_logw(i, p)
        w    = exp(logw)
        A    = get_A(i, p)

        conv = TheoreticalSimulation.ProbDist_convolved(zz, pdf_th_list[i], w)

        poly = if zz === z_list[i]
            # Evaluate via grid Q * dᵢ for numerical stability on provided grid
            d = get_d(i, p)
            QL[i] * d
        else
            # Off-grid: use Horner in standardized t with c
            c  = get_c(i, p)
            t  = @. (zz - μL[i]) / σL[i]
            horner(t, c)
        end
        @. A * conv + poly
    end

    function joint_model(dummy_x, p)
        totalN = 0
        @inbounds for i in 1:M
            totalN += length(z_list[i])
        end
        T = eltype(p)
        res = Vector{T}(undef, totalN)
        pos = 1
        @inbounds for i in 1:M
            yi = model_i(i, z_list[i], p)
            ni = length(yi)
            copyto!(res, pos, yi, 1, ni)
            pos += ni
        end
        res
    end

    # concatenated observations
    y_concat = reduce(vcat, y_list)

    # -------- bounds --------
    lower = fill(-Inf, length(p0))
    upper = fill( Inf, length(p0))
    if w_mode == :global
        lower[idx_logw_global] = log(1e-9); upper[idx_logw_global] = log(1.0)
    elseif w_mode == :per_profile
        @inbounds for i in 1:M
            lower[idx_logw_vec[i]] = log(1e-9)
            upper[idx_logw_vec[i]] = log(1.0)
        end
    end
    if A_mode == :global
        lower[idx_A_global] = 0.0
    elseif A_mode == :per_profile
        @inbounds for i in 1:M
            lower[idx_A_vec[i]] = 0.0
        end
    end
    # (No bounds on d/c)

    # -------- progress wrapper --------
    function joint_model_for_fit(x, p)
        yhat = joint_model(x, p)
        if progress_every > 0
            calls[] += 1
            if calls[] % progress_every == 0
                rss_val = sum(abs2, yhat .- y_concat)
                rss_f   = toflt(rss_val)
                if rss_f < best_rss[]
                    best_rss[] = rss_f
                    best_p[]   = map(toflt, p)
                end
                # display a representative w
                w_show = begin
                    if w_mode == :fixed
                        w_fixed
                    elseif w_mode == :global
                        exp(toflt(p[idx_logw_global]))
                    else
                        exp(toflt(p[idx_logw_vec[1]]))
                    end
                end
                @printf(stderr, "eval %6d | joint rss≈%.6g | w≈%.6g\n", calls[], rss_f, w_show)
            end
        end
        yhat
    end

    # -------- fit --------
    fit_data = LsqFit.curve_fit(joint_model_for_fit, similar(y_concat, 0), y_concat, p0;
                                autodiff=:forward, lower=lower, upper=upper)

    # -------- unpack solution --------
    p̂ = coef(fit_data)

    ŵ = if w_mode == :global
        exp(p̂[idx_logw_global])
    elseif w_mode == :per_profile
        [exp(p̂[idx_logw_vec[i]]) for i in 1:M]
    else
        w_fixed
    end

    Â = if A_mode == :global
        p̂[idx_A_global]
    elseif A_mode == :per_profile
        [p̂[idx_A_vec[i]] for i in 1:M]
    else
        A_fixed
    end

    # d̂ and ĉ lists (always return per-profile lists for convenience)
    if d_mode == :global
        c_global = collect(p̂[idx_c_global])
        d̂ = [RL[i] * c_global for i in 1:M]
        ĉ = [c_global for _ in 1:M]
    else
        d̂ = [collect(p̂[d_ranges[i]]) for i in 1:M]
        ĉ = [RL[i] \ d̂[i] for i in 1:M]
    end

    # covariance / SEs (robust)
    se_all, cov_all = robust_se_and_cov(fit_data; rcond=rcond, ridge=ridge)

    δw = if w_mode == :global
        (ŵ isa Number ? ŵ : error("logic")) * se_all[idx_logw_global]
    elseif w_mode == :per_profile
        [exp(p̂[idx_logw_vec[i]]) * se_all[idx_logw_vec[i]] for i in 1:M]
    else
        0.0
    end

    δA = if A_mode == :global
        se_all[idx_A_global]
    elseif A_mode == :per_profile
        [se_all[idx_A_vec[i]] for i in 1:M]
    else
        0.0
    end

    # per-profile covariance blocks for dᵢ → cᵢ
    cov_d = Vector{Union{Nothing,Matrix{Float64}}}(undef, M)
    cov_c = Vector{Union{Nothing,Matrix{Float64}}}(undef, M)
    se_c  = Vector{Vector{Float64}}(undef, M)

    if cov_all === nothing || isempty(cov_all)
        fill!(cov_d, nothing); fill!(cov_c, nothing)
        se_c .= [fill(NaN, ncoef) for _ in 1:M]
    else
        if d_mode == :global
            # one global c-block
            Ic = idx_c_global
            cov_c_global = Matrix(cov_all[Ic, Ic])
            for i in 1:M
                cov_c[i] = cov_c_global
                cov_d[i] = RL[i] * cov_c_global * RL[i]'
                se_c[i]  = sqrt.(diag(cov_c_global))
            end
        else
            # independent d-blocks
            for i in 1:M
                Id       = d_ranges[i]
                cov_d[i] = Matrix(cov_all[Id, Id])
                C        = RL[i] \ (cov_d[i] / RL[i]')  # cov in c-space
                cov_c[i] = C
                se_c[i]  = sqrt.(diag(C))
            end
        end
    end

    model_on_z = [model_i(i, z_list[i], p̂) for i in 1:M]
    modelfun   = (i, zz) -> model_i(i, zz, p̂)

    fit_params   = (w = ŵ, A = Â, c = ĉ)
    param_se = (δw = δw, δA = δA, δc = se_c)

    meta   = (evals=calls[], best_probe=(rss=best_rss[], p=best_p[]),
              w_mode=w_mode, A_mode=A_mode, d_mode=d_mode)
    extras = (
        d = d̂, cov_all = cov_all, cov_d = cov_d, cov_c = cov_c,
        d_mode = d_mode,
        d_ranges = d_mode == :per_profile ? d_ranges : nothing,
        idx_logw_global = idx_logw_global, idx_logw_vec = idx_logw_vec,
        idx_A_global = idx_A_global, idx_A_vec = idx_A_vec,
        idx_c_global = idx_c_global
    )

    return fit_data, fit_params, param_se, modelfun, model_on_z, meta, extras
end

# --- Column headers ---
const _sub = Dict( # map ASCII digits to Unicode subscripts
    '0'=>'₀','1'=>'₁','2'=>'₂','3'=>'₃','4'=>'₄',
    '5'=>'₅','6'=>'₆','7'=>'₇','8'=>'₈','9'=>'₉','-'=>'₋'
);
sub(k::Integer) = join((_sub[c] for c in string(k)));  # "12" -> "₁₂"

# Select experimental data
wanted_data_dir = "20250814" ;
wanted_binning  = 2 ; 
wanted_smooth   = 0.01 ;

# Data loading
read_exp_info = DataReading.find_report_data(
        joinpath(@__DIR__, "EXPDATA_ANALYSIS");
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
nI = length(Ic_sampled)

STEP        = 26 ;
THRESH_A    = 0.020 ;
n_start = findall(>(THRESH_A), Ic_sampled)[1]
P_DEGREE    = 5 ;
ncols_bg    = P_DEGREE + 1 ;

chosen_currents_idx = sort(unique([
        # firstindex(Ic_sampled);
        # @view(findall(>(THRESH_A), Ic_sampled)[1:STEP:end]);
        @view(findall(>(THRESH_A), Ic_sampled)[end-2:end]);
        lastindex(Ic_sampled)
        ]
));

println("Target currents in A: (", 
        join(map(x -> @sprintf("%.3f", x), Ic_sampled[chosen_currents_idx]), ", "),
        ")"
)

norm_mode = :none ;
magnification_factor = read_exp_info.magnification ;
λ0_exp     = 0.0001 ;

z_exp    = (exp_data[:z_mm] .- exp_data[:Centroid_mm][1]) ./ magnification_factor ;
range_z  = floor(minimum([maximum(z_exp),abs(minimum(z_exp))]),digits=1);
nrange_z = 20001;
z_theory = collect(range(-range_z,range_z,length=nrange_z));

@assert isapprox(mean(z_theory), 0.0; atol= 10 * eps(float(range_z)) ) "μz=$(μz) not ~ 0 within atol=$(10 * eps(float(range_z)) )"
@assert isapprox(std(z_theory), std_sample(range_z, nrange_z); atol= eps(float(range_z))) "σz=$(σz) is not defined for a symmetric range"

rl   = length(chosen_currents_idx) ;
cols = palette(:darkrainbow, rl);

hdr_top = Any[
    "Residuals",
    MultiColumn(2, "Theoretical PDF"),
    MultiColumn(ncols_bg, "Background P$(sub(P_DEGREE))(z)")
];
hdr_bot = vcat(["(exp-model)²", "A", "w [mm]"], ["c" * sub(k) for k in 0:P_DEGREE]);

# Preallocate containers
exp_list     = Vector{Vector{Float64}}(undef, rl);   # splined/normalized experiment on z_theory
pdf_th_list  = Vector{Vector{Float64}}(undef, rl) ;  # closed-form theory on z_theory
z_list       = fill(z_theory, rl) ;                  # same grid for all (read-only is fine)

# precompute for this grid:
μ, σ, _t, Q, R = orthonormal_basis_on(z_theory; n=P_DEGREE);
μ_list = fill(μ, rl);  σ_list = fill(σ, rl);
Q_list = fill(Q, rl);  R_list = fill(R, rl);


for (j,i_idx) in enumerate(chosen_currents_idx)
    
    I0 = Ic_sampled[i_idx]

    # EXPERIMENT
    amp_exp     = @view exp_data[:F1_profile][i_idx, :]
    Spl_exp     = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp;
                             weights = TheoreticalSimulation.compute_weights(z_exp, λ0_exp))
    pdf_exp     = Spl_exp.(z_theory)
    exp_list[j] = normalize_vec(pdf_exp; by = norm_mode)

    # THEORY
    𝒢           = TheoreticalSimulation.GvsI(I0)
    μ_eff       = [TheoreticalSimulation.μF_effective(I0, v[1], v[2], K39_params)
             for v in TheoreticalSimulation.fmf_levels(K39_params; Fsel=1)]
    pdf_theory  = mapreduce(μF -> TheoreticalSimulation.getProbDist_v3(
                               μF, 𝒢, 1e-3 .* z_theory, K39_params, effusion_params),
                           +, μ_eff)
    pdf_th_list[j] = normalize_vec(pdf_theory; by = norm_mode)
end

#########################################################################################################
# (1) w = :global & A = :global & Pn =:per_profile
#########################################################################################################
fit_data, fit_params, δparams, modelfun, model_on_z, meta, extras = fit_pdf_joint(z_list, exp_list, pdf_th_list;
              n=P_DEGREE, Q_list, R_list, μ_list, σ_list,
              w_mode=:global, A_mode=:global, d_mode =:per_profile,
              w0=0.25, A0=1.0);

c_poly_coeffs = [Vector{Float64}(undef, ncols_bg) for _ in 1:rl]
for i=1:rl
    fit_poly = bg_function(z_theory,fit_params.c[i])
    c_poly_coeffs[i] = [fit_poly[dg] for dg in 0:P_DEGREE]
end

w_fit = fit_params.w
A_fit = fit_params.A

c_fit_mean = vec(mean(hcat(c_poly_coeffs...); dims=2))

fig_a = plot(
        title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        legend=:topleft,
        legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$ | $A=%$(round(A_fit,sigdigits=5))$",
        legendtitlefontsize = 8
        )
for (i,val) in enumerate(chosen_currents_idx)
    plot!(fig_a,z_theory,exp_list[i], 
        label=L"Experiment ($I_{0}=%$(round(1000*Ic_sampled[val], sigdigits=3))\mathrm{mA}$)",
        line=(cols[i],:solid,1.50))
    plot!(fig_a,z_theory, predict_profile(z_theory,pdf_th_list[i],A_fit,w_fit,c_poly_coeffs[i]),
        label="Fitting function",
        line=(:dash,cols[i],1.2))
end
display(fig_a)

fig_b = plot(
        title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        legend=:topleft,
        legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$ | $A=%$(round(A_fit,sigdigits=5))$",
        legendtitlefontsize = 8
        )
for (i,val) in enumerate(chosen_currents_idx)
    plot!(fig_b,z_theory,exp_list[i], 
        label=L"Experiment ($I_{0}=%$(round(1000*Ic_sampled[val], sigdigits=3))\mathrm{mA}$)",
        line=(cols[i],:solid,1.50))
    plot!(fig_b,z_theory, anypoly_eval(z_theory, c_poly_coeffs[i]),
        label="Background",
        line=(:dash,cols[i],1.2))
end
display(fig_b)

exp_list_all    = Vector{Vector{Float64}}(undef, nI);
pdf_th_list_all = Vector{Vector{Float64}}(undef, nI);
for (j,I0) in enumerate(Ic_sampled)
    
    # EXPERIMENT
    amp_exp     = @view exp_data[:F1_profile][j, :]
    Spl_exp     = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp;
                             weights = TheoreticalSimulation.compute_weights(z_exp, λ0_exp))
    pdf_exp     = Spl_exp.(z_theory)
    exp_list_all[j] = normalize_vec(pdf_exp; by = norm_mode) - anypoly_eval(z_theory,c_fit_mean)

    # THEORY
    𝒢           = TheoreticalSimulation.GvsI(I0)
    μ_eff       = [TheoreticalSimulation.μF_effective(I0, v[1], v[2], K39_params)
             for v in TheoreticalSimulation.fmf_levels(K39_params; Fsel=1)]
    pdf_theory  = mapreduce(μF -> TheoreticalSimulation.getProbDist_v3(
                               μF, 𝒢, 1e-3 .* z_theory, K39_params, effusion_params),
                           +, μ_eff)
    pdf_theory = TheoreticalSimulation.ProbDist_convolved(z_theory, pdf_theory, w_fit)
    pdf_th_list_all[j] = normalize_vec(pdf_theory; by = norm_mode)
end

z_max_mm = zeros(nI,2);
for i=1:nI
    z, S = TheoreticalSimulation.max_of_bspline_positions(z_theory, exp_list_all[i], λ0=wanted_smooth)
    z_max_mm[i,1] = z[1] 

    z, S = TheoreticalSimulation.max_of_bspline_positions(z_theory, pdf_th_list_all[i], λ0=wanted_smooth)
    z_max_mm[i,2] = z[1] 
end

fig_c = plot(       
    title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
    xlabel=L"$z$ (mm)",
    ylabel="Intensity (au)",
    legend=:topleft,
    legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$ | $A=%$(round(A_fit,sigdigits=5))$",
    legendtitlefontsize = 8
)
cols_ni = palette(:rainbow, nI);
for i in n_start:2:nI
    plot!(fig_c,z_theory,exp_list_all[i], label=L"$%$(round(1000*Ic_sampled[i], sigdigits=3))\,\mathrm{mA}$", line=(:solid, cols_ni[i]))
    plot!(fig_c,z_theory,A_fit*pdf_th_list_all[i], label=false, line=(:dash, cols_ni[i]))
end
display(fig_c)


fig_d=plot(Ic_sampled[n_start:end],read_exp_info.framewise_mm[n_start:end]/magnification_factor,
    label="Original",
    line=(:red,2),
    xaxis=:log10,
    yaxis=:log10,
    xlabel="Current (A)",
    ylabel=L"$z$ (mm)",
    xlim=(10e-3,2),
    ylim=(1e-2,2),
    legend=:bottomright,
    legend_foreground_color = nothing,
    legend_background_color = nothing,
)
plot!(fig_d,Ic_sampled[n_start:end],z_max_mm[n_start:end,1],
    label="Current spline",
    line=(:blue,1.8))
plot!(fig_d,Ic_sampled[n_start:end],z_max_mm[n_start:end,2],
    label=L"QM closed formula ($w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$)",
    line=(:green,1.5),
    # size=(400,400),
    )
display(fig_d)


fig = plot(fig_a,fig_b,fig_c,fig_d,
    layout=@layout([a1 a2; a3 a4]),
    size=(1200,800),
    left_margin=4mm,
    bottom_margin=2mm,)
savefig(fig,joinpath(OUTDIR,"fig_01.$(FIG_EXT)"))


#########################################################################################################
# (2) w = :global & A = :per_profile & Pn = :per_profile
#########################################################################################################
@time fit_data, fit_params, δparams, modelfun, model_on_z, meta, extras = fit_pdf_joint(z_list, exp_list, pdf_th_list;
              n=P_DEGREE, Q_list, R_list, μ_list, σ_list,
              w_mode=:global, A_mode=:per_profile, d_mode=:per_profile,
              w0=0.25, A0=1.0)

c_poly_coeffs = [Vector{Float64}(undef, ncols_bg) for _ in 1:rl]
for i=1:rl
    fit_poly = bg_function(z_theory,fit_params.c[i])
    c_poly_coeffs[i] = [fit_poly[dg] for dg in 0:P_DEGREE]
end
c_poly_coeffs

w_fit = fit_params.w
A_fit = fit_params.A

c_fit_mean = vec(mean(hcat(c_poly_coeffs...); dims=2))

fig_a = plot(
        title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        legend=:topleft,
        legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$",
        legendtitlefontsize = 8
        )
for (i,val) in enumerate(chosen_currents_idx)
    plot!(fig_a,z_theory,exp_list[i], 
        label=L"Experiment ($I_{0}=%$(round(1000*Ic_sampled[val], sigdigits=3))\mathrm{mA}$)",
        line=(cols[i],:solid,1.50))
    plot!(fig_a,z_theory, predict_profile(z_theory,pdf_th_list[i],A_fit[i],w_fit,c_poly_coeffs[i]),
        label="Fitting function",
        line=(:dash,cols[i],1.2))
end
display(fig_a)

fig_b = plot(
        title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        legend=:topleft,
        legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$",
        legendtitlefontsize = 8
        )
for (i,val) in enumerate(chosen_currents_idx)
    plot!(fig_b,z_theory,exp_list[i], 
        label=L"Experiment ($I_{0}=%$(round(1000*Ic_sampled[val], sigdigits=3))\mathrm{mA}$)",
        line=(cols[i],:solid,1.50))
    plot!(fig_b,z_theory, anypoly_eval(z_theory, c_poly_coeffs[i]),
        label="Background",
        line=(:dash,cols[i],1.2))
end
display(fig_b)

exp_list_all    = Vector{Vector{Float64}}(undef, nI);
pdf_th_list_all = Vector{Vector{Float64}}(undef, nI);
for (j,I0) in enumerate(Ic_sampled)
    
    # EXPERIMENT
    amp_exp     = @view exp_data[:F1_profile][j, :]
    Spl_exp     = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp;
                             weights = TheoreticalSimulation.compute_weights(z_exp, λ0_exp))
    pdf_exp     = Spl_exp.(z_theory)
    exp_list_all[j] = normalize_vec(pdf_exp; by = norm_mode) - anypoly_eval(z_theory,c_fit_mean)

    # THEORY
    𝒢           = TheoreticalSimulation.GvsI(I0)
    μ_eff       = [TheoreticalSimulation.μF_effective(I0, v[1], v[2], K39_params)
             for v in TheoreticalSimulation.fmf_levels(K39_params; Fsel=1)]
    pdf_theory  = mapreduce(μF -> TheoreticalSimulation.getProbDist_v3(
                               μF, 𝒢, 1e-3 .* z_theory, K39_params, effusion_params),
                           +, μ_eff)
    pdf_theory = TheoreticalSimulation.ProbDist_convolved(z_theory, pdf_theory, w_fit)
    pdf_th_list_all[j] = normalize_vec(pdf_theory; by = norm_mode)
end


z_max_mm = zeros(nI,2);
for i=1:nI
    z, S = TheoreticalSimulation.max_of_bspline_positions(z_theory, exp_list_all[i], λ0=wanted_smooth)
    z_max_mm[i,1] = z[1] 

    z, S = TheoreticalSimulation.max_of_bspline_positions(z_theory, pdf_th_list_all[i], λ0=wanted_smooth)
    z_max_mm[i,2] = z[1] 
end

fig_c = plot(       
    title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
    xlabel=L"$z$ (mm)",
    ylabel="Intensity (au)",
    legend=:topleft,
    legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$ ",
    legendtitlefontsize = 8
)
cols_ni = palette(:rainbow, nI);
for i in n_start:2:nI
    plot!(fig_c,z_theory,exp_list_all[i], label=L"$%$(round(1000*Ic_sampled[i], sigdigits=3))\,\mathrm{mA}$", line=(:solid, cols_ni[i]))
    plot!(fig_c,z_theory,mean(A_fit)*pdf_th_list_all[i], label=false, line=(:dash, cols_ni[i]))
end
display(fig_c)


fig_d=plot(Ic_sampled[n_start:end],read_exp_info.framewise_mm[n_start:end]/magnification_factor,
    label="Original",
    line=(:red,2),
    xaxis=:log10,
    yaxis=:log10,
    xlabel="Current (A)",
    ylabel=L"$z$ (mm)",
    xlim=(10e-3,2),
    ylim=(1e-2,2),
    legend=:bottomright,
    legend_foreground_color = nothing,
    legend_background_color = nothing,
)
plot!(fig_d,Ic_sampled[n_start:end],z_max_mm[n_start:end,1],
    label="Current spline",
    line=(:blue,1.8))
plot!(fig_d,Ic_sampled[n_start:end],z_max_mm[n_start:end,2],
    label=L"QM closed formula ($w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$)",
    line=(:green,1.5),
    # size=(400,400),
    )
display(fig_d)


fig = plot(fig_a,fig_b,fig_c,fig_d,
    layout=@layout([a1 a2; a3 a4]),
    size=(1200,800),
    left_margin=4mm,
    bottom_margin=2mm,)
savefig(fig,joinpath(OUTDIR,"fig_02.$(FIG_EXT)"))



#########################################################################################################
# (3) w = :global & A = :global & Pn = :global
#########################################################################################################
@time fit_data, fit_params, δparams, modelfun, model_on_z, meta, extras = fit_pdf_joint(z_list, exp_list, pdf_th_list;
              n=P_DEGREE, Q_list, R_list, μ_list, σ_list,
              w_mode=:global, A_mode=:global, d_mode=:global,
              w0=0.25, A0=1.0)

c_poly_coeffs = [Vector{Float64}(undef, ncols_bg) for _ in 1:rl]
for i=1:rl
    fit_poly = bg_function(z_theory,fit_params.c[i])
    c_poly_coeffs[i] = [fit_poly[dg] for dg in 0:P_DEGREE]
end
c_poly_coeffs

w_fit = fit_params.w
A_fit = fit_params.A
c_fit = vec(mean(hcat(c_poly_coeffs...); dims=2))

fig_a = plot(
        title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        legend=:topleft,
        legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$",
        legendtitlefontsize = 8
        )
for (i,val) in enumerate(chosen_currents_idx)
    plot!(fig_a,z_theory,exp_list[i], 
        label=L"Experiment ($I_{0}=%$(round(1000*Ic_sampled[val], sigdigits=3))\mathrm{mA}$)",
        line=(cols[i],:solid,1.50))
    plot!(fig_a,z_theory, predict_profile(z_theory,pdf_th_list[i],A_fit,w_fit,c_fit),
        label="Fitting function",
        line=(:dash,cols[i],1.2))
end
display(fig_a)

fig_b = plot(
        title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        legend=:topleft,
        legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$",
        legendtitlefontsize = 8
        )
for (i,val) in enumerate(chosen_currents_idx)
    plot!(fig_b,z_theory,exp_list[i], 
        label=L"Experiment ($I_{0}=%$(round(1000*Ic_sampled[val], sigdigits=3))\mathrm{mA}$)",
        line=(cols[i],:solid,1.50))
    plot!(fig_b,z_theory, anypoly_eval(z_theory, c_poly_coeffs[i]),
        label="Background",
        line=(:dash,cols[i],1.2))
end
display(fig_b)

exp_list_all    = Vector{Vector{Float64}}(undef, nI);
pdf_th_list_all = Vector{Vector{Float64}}(undef, nI);
for (j,I0) in enumerate(Ic_sampled)
    
    # EXPERIMENT
    amp_exp     = @view exp_data[:F1_profile][j, :]
    Spl_exp     = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp;
                             weights = TheoreticalSimulation.compute_weights(z_exp, λ0_exp))
    pdf_exp     = Spl_exp.(z_theory)
    exp_list_all[j] = normalize_vec(pdf_exp; by = norm_mode) - anypoly_eval(z_theory,c_fit)

    # THEORY
    𝒢           = TheoreticalSimulation.GvsI(I0)
    μ_eff       = [TheoreticalSimulation.μF_effective(I0, v[1], v[2], K39_params)
             for v in TheoreticalSimulation.fmf_levels(K39_params; Fsel=1)]
    pdf_theory  = mapreduce(μF -> TheoreticalSimulation.getProbDist_v3(
                               μF, 𝒢, 1e-3 .* z_theory, K39_params, effusion_params),
                           +, μ_eff)
    pdf_theory = TheoreticalSimulation.ProbDist_convolved(z_theory, pdf_theory, w_fit)
    pdf_th_list_all[j] = normalize_vec(pdf_theory; by = norm_mode)
end


z_max_mm = zeros(nI,2);
for i=1:nI
    z, S = TheoreticalSimulation.max_of_bspline_positions(z_theory, exp_list_all[i], λ0=wanted_smooth)
    z_max_mm[i,1] = z[1] 

    z, S = TheoreticalSimulation.max_of_bspline_positions(z_theory, pdf_th_list_all[i], λ0=wanted_smooth)
    z_max_mm[i,2] = z[1] 
end

fig_c = plot(       
    title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
    xlabel=L"$z$ (mm)",
    ylabel="Intensity (au)",
    legend=:topleft,
    legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$ ",
    legendtitlefontsize = 8
)
cols_ni = palette(:rainbow, nI);
for i in n_start:2:nI
    plot!(fig_c,z_theory,exp_list_all[i], label=L"$%$(round(1000*Ic_sampled[i], sigdigits=3))\,\mathrm{mA}$", line=(:solid, cols_ni[i]))
    plot!(fig_c,z_theory,A_fit*pdf_th_list_all[i], label=false, line=(:dash, cols_ni[i]))
end
display(fig_c)


fig_d=plot(Ic_sampled[n_start:end],read_exp_info.framewise_mm[n_start:end]/magnification_factor,
    label="Original",
    line=(:red,2),
    xaxis=:log10,
    yaxis=:log10,
    xlabel="Current (A)",
    ylabel=L"$z$ (mm)",
    xlim=(10e-3,2),
    ylim=(1e-2,2),
    legend=:bottomright,
    legend_foreground_color = nothing,
    legend_background_color = nothing,
)
plot!(fig_d,Ic_sampled[n_start:end],z_max_mm[n_start:end,1],
    label="Current spline",
    line=(:blue,1.8))
plot!(fig_d,Ic_sampled[n_start:end],z_max_mm[n_start:end,2],
    label=L"QM closed formula ($w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$)",
    line=(:green,1.5),
    # size=(400,400),
    )
display(fig_d)


fig = plot(fig_a,fig_b,fig_c,fig_d,
    layout=@layout([a1 a2; a3 a4]),
    size=(1200,800),
    left_margin=4mm,
    bottom_margin=2mm,)
savefig(fig,joinpath(OUTDIR,"fig_03.$(FIG_EXT)"))



#########################################################################################################
# (4) w = :global & A = :per_profile & Pn = :global
#########################################################################################################
@time fit_data, fit_params, δparams, modelfun, model_on_z, meta, extras = fit_pdf_joint(z_list, exp_list, pdf_th_list;
              n=P_DEGREE, Q_list, R_list, μ_list, σ_list,
              w_mode=:global, A_mode=:per_profile, d_mode=:global,
              w0=0.25, A0=1.0)

c_poly_coeffs = [Vector{Float64}(undef, ncols_bg) for _ in 1:rl]
for i=1:rl
    fit_poly = bg_function(z_theory,fit_params.c[i])
    c_poly_coeffs[i] = [fit_poly[dg] for dg in 0:P_DEGREE]
end
c_poly_coeffs

w_fit = fit_params.w
A_fit = fit_params.A
c_fit_mean = vec(mean(hcat(c_poly_coeffs...); dims=2))

fig_a = plot(
        title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        legend=:topleft,
        legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$",
        legendtitlefontsize = 8
        )
for (i,val) in enumerate(chosen_currents_idx)
    plot!(fig_a,z_theory,exp_list[i], 
        label=L"Experiment ($I_{0}=%$(round(1000*Ic_sampled[val], sigdigits=3))\mathrm{mA}$)",
        line=(cols[i],:solid,1.50))
    plot!(fig_a,z_theory, predict_profile(z_theory,pdf_th_list[i],A_fit[i],w_fit,c_poly_coeffs[i]),
        label="Fitting function",
        line=(:dash,cols[i],1.2))
end
display(fig_a)

fig_b = plot(
        title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        legend=:topleft,
        legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$",
        legendtitlefontsize = 8
        )
for (i,val) in enumerate(chosen_currents_idx)
    plot!(fig_b,z_theory,exp_list[i], 
        label=L"Experiment ($I_{0}=%$(round(1000*Ic_sampled[val], sigdigits=3))\mathrm{mA}$)",
        line=(cols[i],:solid,1.50))
    plot!(fig_b,z_theory, anypoly_eval(z_theory, c_poly_coeffs[i]),
        label="Background",
        line=(:dash,cols[i],1.2))
end
display(fig_b)

exp_list_all    = Vector{Vector{Float64}}(undef, nI);
pdf_th_list_all = Vector{Vector{Float64}}(undef, nI);
for (j,I0) in enumerate(Ic_sampled)
    
    # EXPERIMENT
    amp_exp     = @view exp_data[:F1_profile][j, :]
    Spl_exp     = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp;
                             weights = TheoreticalSimulation.compute_weights(z_exp, λ0_exp))
    pdf_exp     = Spl_exp.(z_theory)
    exp_list_all[j] = normalize_vec(pdf_exp; by = norm_mode) - anypoly_eval(z_theory,c_fit_mean)

    # THEORY
    𝒢           = TheoreticalSimulation.GvsI(I0)
    μ_eff       = [TheoreticalSimulation.μF_effective(I0, v[1], v[2], K39_params)
             for v in TheoreticalSimulation.fmf_levels(K39_params; Fsel=1)]
    pdf_theory  = mapreduce(μF -> TheoreticalSimulation.getProbDist_v3(
                               μF, 𝒢, 1e-3 .* z_theory, K39_params, effusion_params),
                           +, μ_eff)
    pdf_theory = TheoreticalSimulation.ProbDist_convolved(z_theory, pdf_theory, w_fit)
    pdf_th_list_all[j] = normalize_vec(pdf_theory; by = norm_mode)
end


z_max_mm = zeros(nI,2);
for i=1:nI
    z, S = TheoreticalSimulation.max_of_bspline_positions(z_theory, exp_list_all[i], λ0=wanted_smooth)
    z_max_mm[i,1] = z[1] 

    z, S = TheoreticalSimulation.max_of_bspline_positions(z_theory, pdf_th_list_all[i], λ0=wanted_smooth)
    z_max_mm[i,2] = z[1] 
end

fig_c = plot(       
    title=L"$w\rightarrow$ %$(meta.w_mode) | $A\rightarrow$ %$(meta.A_mode) | $P_{%$(P_DEGREE)}\rightarrow$ %$(meta.d_mode)",
    xlabel=L"$z$ (mm)",
    ylabel="Intensity (au)",
    legend=:topleft,
    legendtitle = L"$w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$ ",
    legendtitlefontsize = 8
)
cols_ni = palette(:rainbow, nI);
for i in n_start:2:nI
    plot!(fig_c,z_theory,exp_list_all[i], label=L"$%$(round(1000*Ic_sampled[i], sigdigits=3))\,\mathrm{mA}$", line=(:solid, cols_ni[i]))
    plot!(fig_c,z_theory,mean(A_fit)*pdf_th_list_all[i], label=false, line=(:dash, cols_ni[i]))
end
display(fig_c)


fig_d=plot(Ic_sampled[n_start:end],read_exp_info.framewise_mm[n_start:end]/magnification_factor,
    label="Original",
    line=(:red,2),
    xaxis=:log10,
    yaxis=:log10,
    xlabel="Current (A)",
    ylabel=L"$z$ (mm)",
    xlim=(10e-3,2),
    ylim=(1e-2,2),
    legend=:bottomright,
    legend_foreground_color = nothing,
    legend_background_color = nothing,
)
plot!(fig_d,Ic_sampled[n_start:end],z_max_mm[n_start:end,1],
    label="Current spline",
    line=(:blue,1.8))
plot!(fig_d,Ic_sampled[n_start:end],z_max_mm[n_start:end,2],
    label=L"QM closed formula ($w=%$(round(1000*w_fit,sigdigits=5))\mathrm{\mu m}$)",
    line=(:green,1.5),
    # size=(400,400),
    )
display(fig_d)


fig = plot(fig_a,fig_b,fig_c,fig_d,
    layout=@layout([a1 a2; a3 a4]),
    size=(1200,800),
    left_margin=4mm,
    bottom_margin=2mm,)
savefig(fig,joinpath(OUTDIR,"fig_04.$(FIG_EXT)"))


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
    Normalization mode          : $(norm_mode)
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
