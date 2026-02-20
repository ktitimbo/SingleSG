module ProfileFitTools

"""
    ProfileFitTools

Small utility module collecting helper functions for profile normalization,
polynomial bases/evaluation, robust covariance extraction from `LsqFit` fits,
and a joint multi-profile fitter (`fit_pdf_joint`) used to fit experimental
profiles to convolved theoretical PDFs plus polynomial backgrounds.

## Dependencies

### Standard libraries
- `LinearAlgebra`
- `Statistics`
- `Printf`

### External packages (required for specific functionality)
- `LsqFit` (**required** for `robust_se_and_cov` and `fit_pdf_joint`)
- `Polynomials` (**required** for `t_affine_poly`, `bg_function`)

### Optional packages / modules
- `ForwardDiff` (**optional**) used only if `robust_se_and_cov` is asked to
  recompute the Jacobian (when the fit object does not store a Jacobian).
- `TheoreticalSimulation` (**required by you / your project**) for
  `TheoreticalSimulation.ProbDist_convolved` used in `predict_profile` and
  `fit_pdf_joint`.

If you want this module to be *standalone*, replace `TheoreticalSimulation.*`
calls with a function argument (dependency injection), or move the convolver
into this module.
"""

# -------------------------
# Imports / dependencies
# -------------------------
using LinearAlgebra
using Statistics
using Printf
using DSP

import LsqFit
import Polynomials: Polynomial
include("./TheoreticalSimulation.jl")

# -------------------------
# Exports
# -------------------------
# export normalize_vec,
#        std_sample,
#        robust_se_and_cov,
#        orthonormal_basis_on,
#        horner,
#        anypoly_eval,
#        t_affine_poly,
#        bg_function,
#        predict_profile,
#        fit_pdf_joint,
#        sub

# =========================
# 1) BASIC UTILITIES
# =========================

"""
    normalize_vec(v; by=:max, atol=0)

Normalize an array-like `v` by a chosen scalar denominator.

- `by = :max` divides by `maximum(v)`
- `by = :sum` divides by `sum(v)`
- `by = :none` returns `v` unchanged

If the chosen denominator has magnitude `≤ atol`, returns `v` unchanged
(to avoid blowing up noise or dividing by ~0).

### Notes
- Returns `v ./ denom`, i.e. a new array (same shape as `v`).
- If you want an in-place version, define `normalize_vec!(v; ...)`.

### Examples
julia
normalize_vec([1,2,3])                  # -> [1/3, 2/3, 1]
normalize_vec([1,2,3]; by=:sum)         # -> [1/6, 2/6, 3/6]
normalize_vec([0,0,0]; by=:max, atol=0) # -> unchanged (denom==0)
```
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
    normalize_pdf!(y, Δz; nonneg=true)

In-place normalization so that `sum(y)*Δz == 1` (if possible).
Optionally clips negative values to 0 before normalizing.
Returns the modified vector.
"""
function normalize_pdf!(y::AbstractVector, Δz; nonneg::Bool=true)
    if nonneg
        @inbounds for i in eachindex(y)
            yi = y[i]
            y[i] = yi < 0 ? 0.0 : yi
        end
    end
    area = sum(y) * Δz
    if area > 0
        y ./= area
    end
    return y
end

"""
    std_sample(a, N)

Compute the factor
`a * sqrt(N*(N+1) / (3*(N-1)^2))` for odd `N = 2n+1` (with `N ≥ 3`).

This appears in discretized / finite-sample corrections when mapping
a half-width `a` and an odd number of samples `N` to an equivalent
standard-deviation-like scale.

### Requirements
- `N ≥ 3`
- `N` must be odd

### Example
julia
std = std_sample(1.0, 5)
```
"""
@inline function std_sample(a::Real, N::Integer)
    @assert N ≥ 3 "N must be ≥ 3"
    @assert isodd(N) "N must be odd (N = 2n + 1)"
    a * sqrt(N*(N+1) / (3*(N-1)^2))
end

# =========================
# 2) ROBUST SE / COVARIANCE
# =========================

"""
    robust_se_and_cov(fit; rcond=1e-12, ridge=0.0,
                      model=nothing, x=nothing, y=nothing, p̂=nothing)

Return `(se, cov)` where:
- `cov` is an estimated parameter covariance matrix,
- `se = sqrt.(diag(cov))` are parameter standard errors.

This helper tries to be robust across different `LsqFit` usage patterns:

1. **Preferred path:** call `LsqFit.vcov(fit)` directly.
2. If that fails, build a covariance estimate from a Jacobian `J`:
   - First try `fit.jacobian` (common in `LsqFit.LsqFitResult`).
   - Otherwise, if `model, x, y, p̂` are provided, recompute `J` using
     Forward-mode AD via `ForwardDiff.jacobian`.

Given residual vector `r`, Jacobian `J` (size `N×p`), degrees of freedom
`dof = max(N - p, 1)`, and `σ² = sum(abs2,r)/dof`:

- If `ridge > 0`:
  `cov ≈ σ² * (J'J + ridge*I)^{-1}` using a Cholesky solve (no explicit inv).
- Else:
  compute an SVD-based pseudo-inverse of `J` with threshold `rcond*max(s)`.

### Keyword arguments
- `rcond`: SVD truncation threshold (relative to max singular value).
- `ridge`: Tikhonov/ridge stabilization strength (λ ≥ 0).
- `model, x, y, p̂`: only needed if we must recompute `J`.

### Dependencies
- Requires `LsqFit`.
- Recomputing `J` requires `ForwardDiff` *to be loaded* (see error message).

### Returns
- `se::Vector`
- `cov::Matrix`

### Example
julia
fit = LsqFit.curve_fit(model, x, y, p0)
se, cov = robust_se_and_cov(fit; rcond=1e-12)
```
"""
function robust_se_and_cov(
    fit;
    rcond::Real = 1e-12,
    ridge::Real = 0.0,
    model = nothing,
    x = nothing,
    y = nothing,
    p̂ = nothing,
)
    # 1) Try the built-in covariance
    try
        cov = LsqFit.vcov(fit)
        return sqrt.(diag(cov)), cov
    catch
        # 2) Get or recompute Jacobian
        J = try
            getfield(fit, :jacobian)
        catch
            nothing
        end

        if J === nothing
            @assert model !== nothing && x !== nothing && y !== nothing && p̂ !== nothing \
                "Need (model, x, y, p̂) to recompute Jacobian"
            # Jacobian of g(p) = model(x,p) - y
            g(p) = model(x, p) .- y

            # Require ForwardDiff only for this path
            FD = try
                getfield(Main, :ForwardDiff)
            catch
                error("ForwardDiff is required to recompute the Jacobian; please `using ForwardDiff`.")
            end
            J = FD.jacobian(g, p̂)
        end

        # 3) Build covariance from J
        r   = fit.resid
        p   = size(J, 2)
        dof = max(length(r) - p, 1)
        σ²  = sum(abs2, r) / dof

        if ridge > 0
            JTJ = J' * J
            F = cholesky!(Symmetric(JTJ) + ridge * I)
            cov = σ² * (F \ I)
            return sqrt.(diag(cov)), cov
        else
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

# =========================
# 3) POLYNOMIAL BASES / EVAL
# =========================

"""
    orthonormal_basis_on(z; n)

Build an orthonormal polynomial basis (via QR) on the standardized coordinate
`t = (z - μ)/σ`, up to degree `n`.

Returns `(μ, σ, t, Qthin, R)` such that:
- `X = [t.^0  t.^1  …  t.^n]` has size `length(z) × (n+1)`,
- `X = Qthin * R`, where `Qthin` has orthonormal columns (thin-Q),
  and `R` is upper triangular.

### Why this exists
Direct monomial fitting can be ill-conditioned for moderate-to-high `n`.
Representing polynomials in the orthonormal basis (columns of `Qthin`) greatly
improves numerical stability.

### Notes
- `μ` is taken as `(first(z)+last(z))/2` (cheap and stable for monotonic grids).
- `σ = std(z)`; `σ` must be nonzero.

### Example
julia
μ, σ, t, Q, R = orthonormal_basis_on(z; n=3)
# If d are orthonormal-basis coefficients, polynomial values on grid:
polyvals = Q * d
# Convert to monomial-in-t coefficients c:
c = R \\ d
```
"""
function orthonormal_basis_on(z::AbstractVector{<:Real}; n::Integer)
    @assert n ≥ 0
    μ = (first(z) + last(z)) / 2
    σ = std(z)
    @assert σ > 0 "z has zero variance"
    invσ = inv(σ)

    t = @. (z - μ) * invσ
    X = hcat((t .^ k for k in 0:n)...)  # npts×(n+1)
    F = qr(X)
    k = n + 1
    R = F.R[1:k, :]                     # (n+1)×(n+1)
    Qthin = X / R                       # thin Q via solve
    return μ, σ, t, Qthin, R
end

"""
    horner(z, c)

Evaluate a polynomial with coefficients `c` using Horner's method:

`p(z) = c[1] + c[2]z + … + c[end] z^(m-1)`

Works for scalar `z` or array `z`, and is compatible with AD types
(e.g. `ForwardDiff.Dual`) because it uses `muladd` and broadcasting.

### Arguments
- `z::Real` or `z::AbstractArray`
- `c::AbstractVector` coefficient vector `[c0, c1, ...]`

### Returns
- If `z` is scalar: scalar
- If `z` is array: array of same shape as `z`

### Example
julia
p = horner(2.0, [1.0, 3.0, 4.0])   # 1 + 3*2 + 4*2^2 = 23
```
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

Alias for [`horner`](@ref), where `c = [c0, c1, …, c_n]`.
"""
anypoly_eval(z, c) = horner(z, c)

"""
    t_affine_poly(μ, σ)

Return a `Polynomials.Polynomial` `p(z)` such that:
`p(z) = (z - μ)/σ`.

This is convenient for building background polynomials in standardized
coordinates.

### Requires
- `Polynomials.jl`

### Example
julia
t = t_affine_poly(μ, σ)
bg = 1.0 + 0.1*t + 0.01*t^2
```
"""
function t_affine_poly(μ::Real, σ::Real)
    @assert σ > 0
    return Polynomial([-μ/σ, 1/σ])
end

"""
    bg_function(z, c)

Construct a polynomial background using `Polynomials.jl` and return it as a
`Polynomial` object.

- `z` is used only to infer `(μ, σ)` for standardization.
- `c[k]` multiplies `t^(k-1)` where `t = (z - μ)/σ`.

### Notes / gotchas
- This function currently asserts `mean(z) ≈ 0`. That matches your current
  workflow, but if you want a general routine, remove that assertion.

### Returns
- `bg::Polynomial`

If you want numeric background values on a grid, use `anypoly_eval`
or evaluate the returned `Polynomial` on your `z` grid.

### Example
julia
bg = bg_function(z, [c0,c1,c2])  # c0 + c1*t + c2*t^2
```
"""
function bg_function(z::AbstractVector, c::AbstractVector)
    @assert isapprox(mean(z), 0.0; atol = 10 * eps()) "μz not ~ 0 within atol=$(10 * eps())"
    μ = mean(z)
    σ = std(z)
    n = length(c)
    t = Polynomial([μ/σ, 1/σ]) # t=(z-μ)/σ
    bg = sum(c[k] * t^(k-1) for k in 1:n)
    return bg
end

"""
    predict_profile(z, profile_theory, A, w, c)

Predict a model profile on grid `z`:

`ŷ(z) = A * conv(profile_theory, w) + bg(z)`

where:
- the convolution is computed by `TheoreticalSimulation.ProbDist_convolved(z, profile_theory, w)`
- the background is `bg(z) = anypoly_eval(z, c)` with monomial coefficients `c`.

### Requirements
- `TheoreticalSimulation.ProbDist_convolved` must exist in your project/module.
  If `TheoreticalSimulation` is not in scope, you must `using TheoreticalSimulation`
  before using this.

### Arguments
- `z::AbstractVector`
- `profile_theory::AbstractVector` (same length as `z`)
- `A::Float64` amplitude
- `w::Float64` Gaussian width (or your convolver’s width parameter)
- `c::AbstractVector` polynomial coefficients in `z` (monomial form)

### Returns
- `Vector` (same length as `z`)
"""
function predict_profile(
    z::AbstractVector,
    profile_theory::AbstractVector,
    A::Float64,
    w::Float64,
    c::AbstractVector,
)
    @assert length(z) == length(profile_theory) "z and pdf must have the same length"
    bg = anypoly_eval(z, c)
    first_term = A * TheoreticalSimulation.ProbDist_convolved(z, profile_theory, w)
    return first_term + bg
end

# =========================
# 4) JOINT MULTI-PROFILE FIT
# =========================

"""
    fit_pdf_joint(z_list, y_list, pdf_th_list;
                  n, Q_list, R_list, μ_list, σ_list,
                  w0, A0=1.0, d0=nothing,
                  w_mode=:global, A_mode=:per_profile, d_mode=:per_profile,
                  w_fixed=w0, A_fixed=A0,
                  progress_every=25,
                  rcond=1e-12, ridge=0.0)

Joint nonlinear least-squares fit of **multiple profiles** (`i = 1,…,M`) to:

`ŷᵢ(z) = Aᵢ * (pdf_thᵢ ⊗ G_{wᵢ})(z) + Pᵢ(z)` for `i = 1..M`

where:
- `pdf_thᵢ` is the theoretical profile on `zᵢ`,
- `⊗` denotes convolution (implemented by your `TheoreticalSimulation.ProbDist_convolved`),
- `G_wᵢ` is a Gaussian kernel with width `wᵢ`
- `Aᵢ` is an amplitude (shared or per-profile or fixed),
- `Pᵢ(z)` is an `n`-degree polynomial background.

### Polynomial representation (stability)
On each grid `zᵢ`, the polynomial is represented in an orthonormal basis:
- Build `Xᵢ = [tᵢ.^0 … tᵢ.^n]`, with standardized `tᵢ = (zᵢ-μᵢ)/σᵢ`
- QR: `Xᵢ = Qᵢ Rᵢ`
- Store orthonormal coefficients `dᵢ` such that polynomial values on-grid are `Qᵢ*dᵢ`
- Convert between monomial-in-`t` coefficients `cᵢ` and `dᵢ` via `dᵢ = Rᵢ*cᵢ`
For off-grid evaluation (when `zz !== z_list[i]`), the model uses Horner on `t`.

### Parameter sharing modes
- `w_mode ∈ (:global, :per_profile, :fixed)`
- `A_mode ∈ (:global, :per_profile, :fixed)`
- `d_mode ∈ (:global, :per_profile)`:
  - `:per_profile`: independent polynomials per profile (stored as `dᵢ`)
  - `:global`: a single polynomial in `t` shared across profiles (stored as global `c`)

### Numerical details
- Width parameters are optimized in log-space (`logw`) to enforce positivity.
- Bounds:
  - `w` bounded to `[1e-9, 1.0]` (in your units)
  - `A ≥ 0` if fitted
- Covariance and standard errors are computed via [`robust_se_and_cov`](@ref).

### Inputs
- `z_list, y_list, pdf_th_list`: vectors of length `M`, each element a vector.
- `z_list::Vector{<:AbstractVector}`  
  Grids `zᵢ` on which each profile is sampled.
- `y_list::Vector{<:AbstractVector}`  
  Observed data values `yᵢ(zᵢ)`.
- `pdf_th_list::Vector{<:AbstractVector}`  
  Theoretical PDFs evaluated on `z_list`.
- `Q_list, R_list, μ_list, σ_list`: basis data per profile.
  If the dimensions don’t match the requested `n`, they are rebuilt internally.

### Required keyword arguments
- `n::Integer`  
  Polynomial degree (`n ≥ 0`). Number of coefficients is `n+1`.
- `Q_list::Vector{<:AbstractMatrix}`  
  Orthonormal basis matrices for each grid.
- `R_list::Vector{<:AbstractMatrix}`  
  Upper-triangular factors for basis transforms.
- `μ_list::Vector{<:Real}`  
  Mean values used for standardizing each grid.
- `σ_list::Vector{<:Real}`  
  Standard deviations for standardization.

### Initial values / fixed parameters
- `w0::Real`  
  Initial guess for Gaussian width.
- `A0::Real = 1.0`  
  Initial guess for amplitude.
- `d0 = nothing`  
  Initial polynomial coefficients in orthonormal basis.
  Defaults to zeros.

### Parameter sharing modes
- `w_mode::Symbol`  
  How Gaussian width is treated:
  - `:global` — one width for all profiles
  - `:per_profile` — separate width per profile
  - `:fixed` — fixed at `w_fixed`

- `A_mode::Symbol`  
  How amplitudes are treated:
  - `:global`
  - `:per_profile`
  - `:fixed`

- `d_mode::Symbol`  
  How polynomial backgrounds are treated:
  - `:global` — shared polynomial
  - `:per_profile` — independent polynomials

### Fixed values (used if mode == :fixed)
- `w_fixed::Real`
- `A_fixed::Real

### Fitting / diagnostics
- `progress_every::Int = 25`  
  Print progress every N model evaluations. Set ≤0 to disable.
- `rcond::Real = 1e-12`  
  Conditioning threshold for covariance estimation.
- `ridge::Real = 0.0`  
  Ridge regularization added to covariance matrix.

### Returns
`(fit_data, fit_params, param_se, modelfun, model_on_z, meta, extras)`

- `fit_data`: `LsqFit.LsqFitResult`
- `fit_params`: `(w=..., A=..., c=...)` (always returns `c` per-profile list for convenience)
- `param_se`: `(δw=..., δA=..., δc=...)`
- `modelfun`: `(i, zz) -> ŷᵢ(zz)`
- `model_on_z`: fitted values on each original grid
- `meta`: run diagnostics (eval count, best probe, modes)
- `extras`: internal bookkeeping (cov blocks, indices, d-coeffs, etc.)

### 1. `fit_data`
**Type:** `LsqFit.LsqFitResult`
Raw result from `LsqFit.curve_fit`, containing:
- packed parameter vector
- residuals
- Jacobian
- convergence diagnostics
Useful for advanced inspection and diagnostics.

### 2. `fit_params`
**Type:** NamedTuple
    (w = ŵ, A = Â, c = ĉ)
- `ŵ` — fitted Gaussian widths  
  (scalar or vector depending on `w_mode`)
- `Â` — fitted amplitudes  
  (scalar or vector depending on `A_mode`)
- `ĉ` — polynomial coefficients (monomial/Horner form)  
  Vector of length `M`, each of length `n+1`
These are the primary physical fit parameters.

### 3. `param_se`
**Type:** NamedTuple
    (δw = δw, δA = δA, δc = se_c)
Standard errors of the fitted parameters:
- `δw` — uncertainty of widths
- `δA` — uncertainty of amplitudes
- `δc` — per-profile uncertainties of polynomial coefficients
Derived from a robust covariance estimate.

### 4. `modelfun`
**Type:** Function
    (i, zz) -> Vector
Evaluates the fitted model for profile `i` on grid `zz`.
Example:
    yfit = modelfun(2, z_new)
Useful for plotting and interpolation.

### 5. `model_on_z`
**Type:** `Vector{Vector}`
Model evaluated on the original grids:
    model_on_z[i] == modelfun(i, z_list[i])
Useful for residual analysis and plotting.

### 6. `meta`
**Type:** NamedTuple
Contains diagnostic information:
    (evals, best_probe, w_mode, A_mode, d_mode)
- `evals` — number of model evaluations
- `best_probe` — best sampled RSS and parameters
- `*_mode` — modes used
Used for logging and reproducibility.

### 7. `extras`
**Type:** NamedTuple
Contains advanced/internal information:
- `d` — orthonormal-basis polynomial coefficients
- `cov_all` — full covariance matrix
- `cov_d` — per-profile covariance in d-space
- `cov_c` — per-profile covariance in c-space
- `d_ranges` — parameter packing ranges
- `idx_*` — indices in packed parameter vector
Intended for debugging and advanced uncertainty propagation.

---

### Required dependencies
- `LsqFit`
- Your `TheoreticalSimulation.ProbDist_convolved`
"""
function fit_pdf_joint(
    z_list::Vector{<:AbstractVector},
    y_list::Vector{<:AbstractVector},
    pdf_th_list::Vector{<:AbstractVector};
    n::Integer,
    Q_list::Vector{<:AbstractMatrix},
    R_list::Vector{<:AbstractMatrix},
    μ_list::Vector{<:Real},
    σ_list::Vector{<:Real},
    w0::Real, A0::Real=1.0, d0=nothing,
    w_mode::Symbol = :global,       # :per_profile, :global, :fixed
    A_mode::Symbol = :per_profile,  # :per_profile, :global, :fixed
    d_mode::Symbol = :per_profile,  # :per_profile, :global
    w_fixed::Real = w0,             # used if w_mode == :fixed
    A_fixed::Real = A0,             # used if A_mode == :fixed
    progress_every::Int=25,
    rcond::Real=1e-12,
    ridge::Real=0.0,
)
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
        needs_rebuild = size(Q_list[i], 1) != length(z_list[i]) ||
                        size(Q_list[i], 2) != ncoef ||
                        size(R_list[i], 1) != ncoef ||
                        size(R_list[i], 2) != ncoef
        if needs_rebuild
            μi, σi, _t, Qi, Ri = orthonormal_basis_on(z_list[i]; n=n)
            μL[i], σL[i], QL[i], RL[i] = μi, σi, Qi, Ri
        else
            μL[i], σL[i], QL[i], RL[i] = float(μ_list[i]), float(σ_list[i]),
                                         Matrix{Float64}(Q_list[i]), Matrix{Float64}(R_list[i])
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
            return view(p, idx_c_global)
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
            c = get_c(i, p)
            t = @. (zz - μL[i]) / σL[i]
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
    fit_data = LsqFit.curve_fit(
        joint_model_for_fit,
        similar(y_concat, 0),
        y_concat,
        p0;
        autodiff = :forward,
        lower = lower,
        upper = upper,
    )
    
    p̂ = LsqFit.coef(fit_data)

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
        exp(p̂[idx_logw_global]) * se_all[idx_logw_global]
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
            Ic = idx_c_global
            cov_c_global = Matrix(cov_all[Ic, Ic])
            for i in 1:M
                cov_c[i] = cov_c_global
                cov_d[i] = RL[i] * cov_c_global * RL[i]'
                se_c[i]  = sqrt.(diag(cov_c_global))
            end
        else
            for i in 1:M
                Id       = d_ranges[i]
                cov_d[i] = Matrix(cov_all[Id, Id])
                C        = RL[i] \ (cov_d[i] / RL[i]')
                cov_c[i] = C
                se_c[i]  = sqrt.(diag(C))
            end
        end
    end

    model_on_z = [model_i(i, z_list[i], p̂) for i in 1:M]
    modelfun   = (i, zz) -> model_i(i, zz, p̂)

    fit_params = (w = ŵ, A = Â, c = ĉ)
    param_se   = (δw = δw, δA = δA, δc = se_c)

    meta = (
        evals = calls[],
        best_probe = (rss = best_rss[], p = best_p[]),
        w_mode = w_mode,
        A_mode = A_mode,
        d_mode = d_mode,
    )

    extras = (
        d = d̂,
        cov_all = cov_all,
        cov_d = cov_d,
        cov_c = cov_c,
        d_mode = d_mode,
        d_ranges = d_mode == :per_profile ? d_ranges : nothing,
        idx_logw_global = idx_logw_global,
        idx_logw_vec = idx_logw_vec,
        idx_A_global = idx_A_global,
        idx_A_vec = idx_A_vec,
        idx_c_global = idx_c_global,
    )

    return fit_data, fit_params, param_se, modelfun, model_on_z, meta, extras
end

# =========================
# 5) SMALL STRING UTILITY
# =========================

"""
    sub(k::Integer) -> String

Convert an integer into a unicode-subscript string.

Examples:
julia
sub(12)    # "₁₂"
sub(-3)    # "₋₃"
```
"""
const _sub = Dict(
    '0'=>'₀','1'=>'₁','2'=>'₂','3'=>'₃','4'=>'₄',
    '5'=>'₅','6'=>'₆','7'=>'₇','8'=>'₈','9'=>'₉','-'=>'₋'
)
sub(k::Integer) = join((_sub[c] for c in string(k)))


# ============================================================
# Window / convolution / deconvolution utilities
# ============================================================

"""
    unitbox_scaled(z, scale; soft=false, ϵ=1e-3)

Centered unit-box window of total width `scale`.

The window is centered at zero and has half-width `scale/2`.

Hard version (`soft=false`)
    Returns 1.0 if |z/scale| ≤ 0.5, otherwise 0.0.

Soft version (`soft=true`)
    Uses a smooth tanh transition controlled by `ϵ`.
    Smaller ϵ produces sharper edges.

Works with scalars or arrays via broadcasting.

Typical use:
    aperture = unitbox_scaled(z, slit_width)
"""
@inline function unitbox_scaled(z, scale; soft::Bool=false, ϵ::Real=1e-3)
    invscale = inv(scale)
    if soft
        return @. 0.5 * (tanh((0.5 - abs(z * invscale)) / ϵ) + 1.0)
    else
        return @. ifelse(abs(z * invscale) ≤ 0.5, 1.0, 0.0)
    end
end


"""
    conv_centered(a, b, Δz)

Centered same-length convolution scaled by grid spacing Δz.

Computes DSP.conv(a,b) and extracts the central region so the
kernel center aligns with the signal grid.

Arguments
---------
a : signal vector
b : kernel vector
Δz : grid spacing

Returns
-------
Vector of length(a).

Note
----
DSP.conv allocates memory.
"""
@inline function conv_centered(a, b, Δz::Real)
    g = DSP.conv(a, b)
    n = length(a)
    m = length(b)
    c = (m ÷ 2) + 1
    y = @view g[c : c + n - 1]
    return Δz .* y
end


"""
    convT_centered(r, b, Δz)

Adjoint (transpose) of conv_centered with respect to the first argument.

Mathematically equivalent to:
    conv_centered(r, reverse(b), Δz)

Used when computing gradients in deconvolution problems.
"""
@inline function convT_centered(r, b, Δz::Real)
    return conv_centered(r, Base.reverse(b), Δz)
end


"""
    D2!(out, x)

Compute the discrete second difference (curvature operator).

Interior stencil:
    out[i] = x[i-1] - 2x[i] + x[i+1]

Boundary points use one-sided versions.

Used as a smoothness regularizer.
"""
function D2!(out::Vector{Float64}, x::Vector{Float64})
    N = length(x)
    @assert N ≥ 3
    @assert length(out) == N
    @inbounds begin
        out[1] = x[1] - 2x[2] + x[3]
        for i in 2:N-1
            out[i] = x[i-1] - 2x[i] + x[i+1]
        end
        out[N] = x[N-2] - 2x[N-1] + x[N]
    end
    return out
end


"""
    deconv_kernel(g, k, z;
                  λ=1e-1, stepsize=1e-1, maxiter=3000,
                  nonneg=true, normalize=true,
                  support_mask=nothing,
                  sym_weight=0.0,
                  verbose_every=200,
                  return_meta=false)

Estimate an unknown 1D kernel `x(z)` from a measured signal `g(z)` and a known kernel `k(z)`
under the forward model:

    g ≈ conv_centered(k, x, Δz)

This is a *regularized, constrained* deconvolution tailored for physically meaningful kernels
(e.g. blur/PSF or probability distributions): smooth, nonnegative, normalized, possibly symmetric.

We solve the optimization problem:

    minimize_x  0.5*||Kx - g||₂² + 0.5*λ*||D²x||₂² + 0.5*sym_weight*||x - reverse(x)||₂²
    subject to  x ≥ 0 (if `nonneg=true`)
                ∑ x Δz = 1 (if `normalize=true`)
                x[i]=0 outside support_mask (if provided)

where:
- `Kx` is the centered "same-length" linear convolution with kernel `k`,
- `D²` is the discrete second-difference (curvature) operator (encourages smoothness),
- the symmetry term is a *soft* penalty (not a hard constraint).

Inputs
------
- `g::Vector{Float64}`: measured signal on grid `z`.
- `k::Vector{Float64}`: known kernel on the same grid `z`. (For your case: k = F = furnace * slit,
  or k = slit to estimate the intermediate furnace-blurred kernel.)
- `z::Vector{Float64}`: grid (must be uniform-ish). `Δz = mean(diff(z))` is used.

Keyword options
---------------
- `λ`: curvature regularization strength. Larger → smoother `x` (less ringing).
- `stepsize`: gradient step size. If unstable/diverges, reduce by ×10.
- `maxiter`: number of projected gradient iterations.
- `nonneg`: enforce x ≥ 0 (recommended for PDFs/PSFs).
- `normalize`: enforce ∑ x Δz = 1 (recommended for PDFs/PSFs).
- `support_mask`: optional `BitVector` of length N; if provided, forces x[i]=0 where mask is false.
- `sym_weight`: symmetry penalty weight (0 disables). Helps if you expect x(z)≈x(-z).
- `verbose_every`: print progress every N iterations (0 disables).
- `return_meta`: if true, return `(x, meta)` where meta includes objective trace.

Returns
-------
- `x::Vector{Float64}` (estimated kernel), or `(x, meta)` if `return_meta=true`.

Notes
-----
- This implementation preallocates all buffers for *everything except* `DSP.conv`, which allocates.
  If you need convolution to be allocation-free too, we can replace it with a planned FFT-based
  operator (recommended if you solve many times with the same `k`).
"""
function deconv_kernel(g::Vector{Float64}, k::Vector{Float64}, z::Vector{Float64};
                      λ::Float64=1e-1,
                      stepsize::Float64=1e-1,
                      maxiter::Int=3000,
                      nonneg::Bool=true,
                      normalize::Bool=true,
                      support_mask::Union{Nothing,BitVector}=nothing,
                      sym_weight::Float64=0.0,
                      verbose_every::Int=1000,
                      return_meta::Bool=false)

    @assert length(g) == length(k) == length(z)
    N  = length(g)
    Δz = mean(diff(z))

    # -------- helper: reversed view (no allocation) --------
    @inline revview(v::AbstractVector) = @view v[end:-1:1]

    # -------- centered "same" conv (allocates due to DSP.conv) --------
    @inline function local_conv_centered(a::Vector{Float64}, b::Vector{Float64})
        full = DSP.conv(a, b)                # allocates (length 2N-1)
        c = (length(b) ÷ 2) + 1
        return @view(full[c : c + N - 1])    # view into full
    end

    # Adjoint w.r.t. x (second argument): Kᵀ r = local_conv_centered(r, reverse(k))
    # We'll implement with reversed view (no allocation of reverse(k)).
    # (Still allocates in DSP.conv.)
    @inline function local_convT_centered(r::Vector{Float64})
        kr = revview(k)
        full = DSP.conv(r, kr)              # allocates
        c = (length(kr) ÷ 2) + 1
        return @view(full[c : c + N - 1])
    end

    # -------- second difference, in-place --------
    function local_D2!(out::Vector{Float64}, x::Vector{Float64})
        @inbounds begin
            out[1] = x[1] - 2x[2] + x[3]
            for i in 2:N-1
                out[i] = x[i-1] - 2x[i] + x[i+1]
            end
            out[N] = x[N-2] - 2x[N-1] + x[N]
        end
        return out
    end

    # -------- buffers (preallocated) --------
    x      = zeros(Float64, N)
    y      = zeros(Float64, N)
    r      = zeros(Float64, N)
    grad   = zeros(Float64, N)
    tmp1   = zeros(Float64, N)
    tmp2   = zeros(Float64, N)

    # optional objective trace
    obj_trace = return_meta ? Float64[] : nothing

    # -------- projection (in-place) --------
    function local_project!(x::Vector{Float64})
        if support_mask !== nothing
            @inbounds for i in 1:N
                if !support_mask[i]
                    x[i] = 0.0
                end
            end
        end
        if nonneg
            @inbounds for i in 1:N
                xi = x[i]
                x[i] = xi < 0 ? 0.0 : xi
            end
        end
        if normalize
            s = sum(x) * Δz
            if s > 0
                invs = inv(s)
                @inbounds for i in 1:N
                    x[i] *= invs
                end
            end
        end
        return x
    end

    # -------- initialize x from g --------
    @inbounds for i in 1:N
        x[i] = nonneg ? max(g[i], 0.0) : g[i]
    end
    local_project!(x)

    # -------- objective (mostly allocation-free; conv allocates) --------
    function local_objective(x::Vector{Float64})
        # y = Kx
        yview = local_conv_centered(k, x)
        @inbounds for i in 1:N
            y[i] = Δz * yview[i]
        end

        # data term
        data = 0.0
        @inbounds for i in 1:N
            di = y[i] - g[i]
            data += 0.5 * di * di
        end

        # reg term
        local_D2!(tmp1, x)
        reg = 0.0
        @inbounds for i in 1:N
            reg += 0.5 * λ * tmp1[i] * tmp1[i]
        end

        # symmetry term
        sym = 0.0
        if sym_weight > 0
            xr = revview(x)
            @inbounds for i in 1:N
                di = x[i] - xr[i]
                sym += 0.5 * sym_weight * di * di
            end
        end

        return data + reg + sym
    end

    fprev = local_objective(x)
    return_meta && push!(obj_trace, fprev)

    # --- add these BEFORE the main loop (right after fprev is computed is fine) ---
    x_prev   = copy(x)                 # snapshot for step-size / movement diagnostics
    stall_ct = 0                       # consecutive “no progress” counter
    stall_max = 3                      # stop after 5 consecutive stalls

    # thresholds (tune)
    tol_rel_obj = 1e-4                 # objective relative change threshold (per verbose check)
    tol_rel_x   = 1e-6                 # parameter relative change threshold (per verbose check)

    # -------- main loop --------
    for it in 1:maxiter
        # y = Kx  (scaled by Δz)
        yview = local_conv_centered(k, x)
        @inbounds for i in 1:N
            y[i] = Δz * yview[i]
        end

        # r = y - g
        @inbounds for i in 1:N
            r[i] = y[i] - g[i]
        end

        # grad_data = Kᵀ r  (then scale by Δz because forward had Δz)
        # convT gives view into allocated conv result; copy into grad
        gview = local_convT_centered(r)
        @inbounds for i in 1:N
            grad[i] = Δz * gview[i]
        end

        # grad_reg = λ * D2ᵀD2 x ≈ λ * D2(D2(x))
        local_D2!(tmp1, x)
        local_D2!(tmp2, tmp1)
        @inbounds for i in 1:N
            grad[i] += λ * tmp2[i]
        end

        # symmetry gradient: sym_weight*(x - reverse(x))
        if sym_weight > 0
            xr = revview(x)
            @inbounds for i in 1:N
                grad[i] += sym_weight * (x[i] - xr[i])
            end
        end

        # x <- x - stepsize*grad
        @inbounds for i in 1:N
            x[i] -= stepsize * grad[i]
        end

        # project
        local_project!(x)

        # if verbose_every > 0 && (it % verbose_every == 0)
        #     fnow = local_objective(x)
        #     rel = abs(fnow - fprev) / max(fprev, 1e-12)
        #     @printf("iter %5d | obj=%.6g | relΔ=%.3g\n", it, fnow, rel)
        #     fprev = fnow
        #     return_meta && push!(obj_trace, fnow)
        # end
        # --- replace your existing verbose block with THIS ---
        if verbose_every > 0 && (it % verbose_every == 0)

            # Objective and relative improvement
            fnow = local_objective(x)
            rel_obj = abs(fnow - fprev) / max(fprev, 1e-12)

            # How much x changed since last verbose checkpoint
            dx = x .- x_prev
            rel_x = norm(dx) / max(norm(x_prev), 1e-12)

            # Optional: track individual terms (data / reg / sym) without extra allocations
            # (This recomputes Kx and D2x, so it's extra work, but only every verbose_every)
            # data term:
            yview = local_conv_centered(k, x)
            @inbounds for i in 1:N
                y[i] = Δz * yview[i]
            end
            data_term = 0.0
            @inbounds for i in 1:N
                di = y[i] - g[i]
                data_term += 0.5 * di * di
            end

            # reg term:
            local_D2!(tmp1, x)
            reg_term = 0.0
            @inbounds for i in 1:N
                reg_term += 0.5 * λ * tmp1[i] * tmp1[i]
            end

            # symmetry term:
            sym_term = 0.0
            if sym_weight > 0
                xr = revview(x)
                @inbounds for i in 1:N
                    di = x[i] - xr[i]
                    sym_term += 0.5 * sym_weight * di * di
                end
            end

            @printf(
                "iter %7d | obj=%.6g | relΔobj=%.3g | relΔx=%.3g | data=%.6g | reg=%.6g | sym=%.6g\n",
                it, fnow, rel_obj, rel_x, data_term, reg_term, sym_term
            )

            # --- early stopping logic ---
            # Consider it “stalled” only if BOTH objective and x changes are small
            if (rel_obj < tol_rel_obj) && (rel_x < tol_rel_x)
                stall_ct += 1
            else
                stall_ct = 0
            end

            if stall_ct ≥ stall_max
                @printf("Early stop: stalled for %d checks (relΔobj<%.1e and relΔx<%.1e)\n",
                        stall_max, tol_rel_obj, tol_rel_x)
                fprev = fnow
                return_meta && push!(obj_trace, fnow)
                break
            end

            # update checkpoints
            fprev = fnow
            x_prev .= x
            return_meta && push!(obj_trace, fnow)
        end
    end

    if return_meta
        meta = (Δz=Δz, λ=λ, stepsize=stepsize, maxiter=maxiter,
                nonneg=nonneg, normalize=normalize,
                sym_weight=sym_weight,
                obj_trace=obj_trace)
        return x, meta
    else
        return x
    end
end


# ============================================================
# Peak models and fitting helpers
# ============================================================

""" Gaussian peak with constant background.
Parameters p = [A, μ, σ, c]. """
gauss(x, p) = @. p[1] * exp(-0.5*((x - p[2]) / p[3])^2) + p[4]


""" Lorentzian (Cauchy) peak with background.
Parameters p = [A, μ, γ, c], γ is HWHM. """
lorentz(x, p) = @. p[1] * (p[3]^2 / ((x - p[2])^2 + p[3]^2)) + p[4]


""" Pseudo-Voigt profile (Gaussian–Lorentzian mixture).
p = [A, μ, σ, γ, η, c], η ∈ [0,1]. """
function pvoigt(x, p)
    A, μ, σ, γ, η, c = p
    @. A * ( η*(γ^2/((x-μ)^2+γ^2)) +
             (1-η)*exp(-0.5*((x-μ)/σ)^2) ) + c
end


"""
    init_peak(x,y)

Robust initialization for peak fitting.

Returns NamedTuple:
(A0, μ0, σ0, γ0, c0)
"""
function init_peak(x, y)
    # baseline guess = median (robust)
    c0 = median(y)
    y0 = y .- c0
    # peak location
    im = argmax(y0)
    # amplitude
    μ0 = x[im]
    A0 = y0[im]
    # width guess from second moment around peak (robust-ish)
    w = max.(y0, 0.0)
    s = sum(w)
    if s > 0
        μw = sum(x .* w) / s
        σ0 = sqrt(sum(((x .- μw).^2) .* w) / s)
    else
        σ0 = (maximum(x) - minimum(x)) / 20
    end
    # Lorentz HWHM guess from σ (rough)
    σ0 = max(σ0, eps(Float64))
    γ0 = σ0
    return (A0=A0, μ0=μ0, σ0=σ0, γ0=γ0, c0=c0)
end


""" Fit Gaussian model using LsqFit. """
function fit_gaussian(x, y)
    g = init_peak(x, y)
    p0 = [g.A0, g.μ0, g.σ0, g.c0]
    lower = [0.0, minimum(x), 1e-12, -Inf]
    upper = [Inf, maximum(x), Inf, Inf]
    return LsqFit.curve_fit(gauss, x, y, p0; lower=lower, upper=upper)
end


""" Fit Lorentzian model using LsqFit. """
function fit_lorentzian(x, y)
    g = init_peak(x, y)
    p0 = [g.A0, g.μ0, g.γ0, g.c0]
    lower = [0.0, minimum(x), 1e-12, -Inf]
    upper = [Inf, maximum(x), Inf, Inf]
    return LsqFit.curve_fit(lorentz, x, y, p0; lower=lower, upper=upper)
end


""" Fit pseudo-Voigt model using LsqFit. """
function fit_pvoigt(x, y)
    g = init_peak(x, y)
    η0 = 0.5
    p0 = [g.A0, g.μ0, g.σ0, g.γ0, η0, g.c0]
    lower = [0.0, minimum(x), 1e-12, 1e-12, 0.0, -Inf]
    upper = [Inf, maximum(x), Inf, Inf, 1.0, Inf]
    return LsqFit.curve_fit(pvoigt, x, y, p0; lower=lower, upper=upper)
end


""" Residual sum of squares. """
rss(y, yhat) = sum(abs2, yhat .- y)


""" Akaike Information Criterion using RSS. """
function aic(y, yhat, k)
    n = length(y)
    r = rss(y, yhat)
    return n*log(r/n) + 2k
end

########### END COPY ###########









end # module
