# ============================================================================
#  Pure-CQD forward model of the SG1 F=1 (up-branch) peak position
#
#  No QM reference curve and no phenomenological dose offset.  For every
#  current the up-branch spatial profile at the screen is built from
#    • CQD electron dynamics  c(τ) = tanh[artanh c₀ + u τ],  u = kᵢ γ B L₁ / v
#    • heart-shaped up-branch weighting  P(c₀|up) = (1 + c₀)/2
#    • effusive flux speed distribution  f(v) ∝ v³ exp(−M v²/2k_BT)
#    • geometric beam image of furnace aperture + collimating slit
#    • Gaussian detector/analysis blur σ
#  and its maximum is the predicted peak z_peak(I; kᵢ).  The field is taken
#  uniform over the effective magnet length L₁ (B, G from model_data.csv).
#
#  Fits:  kᵢ alone (A = 1)  and  kᵢ + magnification A  (z_data ≈ A z_peak).
#  Objective weighted by the experimental errors σ_j, as in the analytical
#  script SG2026Q4_analytic_ki_weighted.jl:
#     χ² = Σ [ ln(z_j^mod / z_j) / s_j ]²,   s_j = σ_j / z_j
#  A is profiled analytically (weighted geometric mean); kᵢ uncertainty from
#  χ²(kᵢ) = χ²_min + max(1, χ²_red).  The check against the Monte Carlo column
#  is unweighted, since that column has no errors.
#
#  Appendix script of SG1_analytic_CQD_fit_v3.tex (same folder).
#  All outputs stay in this folder: figures in figs_forward/, the tables
#  forward_summary.csv and forward_points.csv next to this script.
#  Run with threads, e.g.  julia -t auto SG2026Q4_cqd_forward_fit.jl
# ============================================================================
using CSV, DataFrames, Optim, LinearAlgebra, Printf, Plots, Distributions, FFTW
using Base.Threads

cd(@__DIR__)
const ROOT   = normpath(joinpath(@__DIR__, "..", ".."))
const CSV_IN = joinpath(ROOT, "data_studies", "FIT2026_ki_scale_20261007T172140485", "model_data.csv")
const OUTDIR = @__DIR__
const FIGDIR = joinpath(OUTDIR, "figs_forward")
mkpath(FIGDIR)

# ---------------------------------------------------------------- constants
const γe   = 1.76085963023e11           # rad s⁻¹ T⁻¹
const μe   = 9.2847647043e-24           # J T⁻¹
const kB   = 1.380649e-23
const M    = 38.9637064864 * 1.66053906660e-27
const T_ov = 478.15
const a_fl = M / (2kB * T_ov)

# ---------------------------------------------------------------- geometry (m)
const z_furnace = 100e-6                # furnace aperture height
const z_slit    = 300e-6                # collimating slit height
const y_FS      = 0.54175               # furnace → slit (2026Q2 setup)
const y_SSG     = 0.044                 # slit → SG entrance
const L1        = 0.070                 # SG length
const L2_LIST   = [0.39525]             # SG exit → screen (requested)
const σ_BLUR    = 0.125                 # mm, Gaussian profile smoothing (as in the MC pipeline)

# ---------------------------------------------------------------- data
df   = CSV.read(CSV_IN, DataFrame)
I    = df.Current_A; B = df.B_T; G = df.G_Tm; z = df.Experiment; σz = df.ErrorExperiment
zCQD = df.CQD_up_mm; zQM = df.QM_F1_mm
n    = length(z)

# ---------------------------------------------------------------- quadrature
function gauss_legendre(N)
    β = [k / sqrt(4k^2 - 1) for k in 1:N-1]
    F = eigen(SymTridiagonal(zeros(N), β))
    return F.values, 2 .* F.vectors[1, :] .^ 2
end
glmap(x, w, a, b) = ((b - a) / 2 .* x .+ (a + b) / 2, (b - a) / 2 .* w)

const τx, τw = glmap(gauss_legendre(40)..., 0.0, 1.0)
const cx, cw0 = gauss_legendre(160)
const cw = cw0 .* (1 .+ cx) ./ 2                              # heart weighting (1+c₀)/2, Σ = 1
const vmax = 6 / sqrt(a_fl)
const vx, vw0 = glmap(gauss_legendre(400)..., 0.0, vmax)
const fv = @. 2a_fl^2 * vx^3 * exp(-a_fl * vx^2)              # normalized flux density
const vw = vw0 .* fv

# ---------------------------------------------------------------- beam kernel (mm)
# box(W1) ⊗ box(W2) ⊗ N(0,σ²) in closed form;  Ψ(x) = ∫_{-∞}^x Φ(t/σ) dt
Ψ(x, σ) = x * cdf(Normal(), x / σ) + σ * pdf(Normal(), x / σ)
function beam_widths(L2)
    y_scr = y_FS + y_SSG + L1 + L2
    m = y_scr / y_FS
    return 1e3 * z_slit * m, 1e3 * z_furnace * (m - 1)         # mm
end
function kernel(x, W1, W2, σ)
    a, b = W1 / 2, W2 / 2
    return (Ψ(x + a + b, σ) - Ψ(x - a + b, σ) - Ψ(x + a - b, σ) + Ψ(x - a - b, σ)) / (W1 * W2)
end

# ---------------------------------------------------------------- atom deflections
"""Deflection (mm) and weight of every (v, c₀) node for field B, gradient Gr."""
function deflections(ki, Bj, Gj, L2)
    r  = L2 / L1
    nv, nc, nτ = length(vx), length(cx), length(τx)
    Δ  = Vector{Float64}(undef, nv * nc)
    w  = Vector{Float64}(undef, nv * nc)
    t  = Vector{Float64}(undef, nτ)
    lev = @. τw * (r + 1 - τx)
    for i in 1:nv
        T1 = L1 / vx[i]
        u  = ki * γe * Bj * T1
        s  = 1e3 * μe * Gj * T1^2 / M                          # mm
        @inbounds for k in 1:nτ
            t[k] = tanh(u * τx[k])
        end
        @inbounds for l in 1:nc
            c = cx[l]; acc = 0.0
            for k in 1:nτ
                acc += lev[k] * (c + t[k]) / (1 + c * t[k])
            end
            idx = (i - 1) * nc + l
            Δ[idx] = s * acc
            w[idx] = vw[i] * cw[l]
        end
    end
    return Δ, w
end

# ---------------------------------------------------------------- profile peak
const DZ   = 0.002                       # mm, coarse histogram step
const ZLO, ZHI = -3.0, 9.0
const ZG   = collect(ZLO:DZ:ZHI)

function peak_position(Δ, w, W1, W2, σ)
    # coarse: cloud-in-cell histogram ⊗ sampled kernel via FFT
    nb = length(ZG)
    h  = zeros(nb)
    @inbounds for i in eachindex(Δ)
        x = (Δ[i] - ZLO) / DZ
        (x < 0 || x >= nb - 1) && continue
        j = floor(Int, x); f = x - j
        h[j+1] += w[i] * (1 - f); h[j+2] += w[i] * f
    end
    half = ceil(Int, ((W1 + W2) / 2 + 6σ) / DZ)
    kx   = (-half:half) .* DZ
    ks   = [kernel(x, W1, W2, σ) for x in kx]
    L    = nextprod([2, 3, 5], nb + length(ks))
    P    = real(ifft(fft([h; zeros(L - nb)]) .* fft([ks; zeros(L - length(ks))])))[half+1:half+nb]
    jmax = argmax(P)
    z0   = ZG[jmax]
    # refine: maximize the exact profile Σ w K(z − Δ)
    prof(zz) = -sum(w[i] * kernel(zz - Δ[i], W1, W2, σ) for i in eachindex(Δ) if abs(zz - Δ[i]) < (W1 + W2) / 2 + 6σ)
    res = optimize(prof, z0 - 2DZ, z0 + 2DZ, Brent(); abs_tol = 1e-9)
    return Optim.minimizer(res)
end

function zpeak_all(ki, L2; σ = σ_BLUR, idx = 1:n)
    W1, W2 = beam_widths(L2)
    out = zeros(length(idx))
    @threads for k in eachindex(idx)
        j = idx[k]
        Δ, w = deflections(ki, B[j], G[j], L2)
        out[k] = peak_position(Δ, w, W1, W2, σ)
    end
    return out
end

relrms(zm, zd) = sqrt(sum(abs2, zm ./ zd .- 1) / length(zd))
const s_log = σz ./ z                    # log-space standard deviation

# weighted χ², with magnification A profiled analytically (weighted geometric mean) or fixed to 1
function objective(ki, L2; freeA = false, σ = σ_BLUR, idx = 1:n)
    zm  = zpeak_all(ki, L2; σ = σ, idx = idx)
    w   = 1 ./ s_log[idx] .^ 2
    r   = log.(zm) .- log.(z[idx])
    lnA = freeA ? -sum(w .* r) / sum(w) : 0.0
    return sum(w .* (r .+ lnA) .^ 2), exp(lnA), zm
end

"""kᵢ where χ² first exceeds χ²_min + Δ on one side (sgn = ±1), by bracketing and bisection in ln kᵢ."""
function crossing(f, lmin, target, sgn)
    a, b = lmin, lmin + sgn * 0.05
    for _ in 1:60
        f(b) > target && break
        a, b = b, b + sgn * 0.05
    end
    f(b) > target || return NaN
    for _ in 1:20
        m = (a + b) / 2
        f(m) > target ? (b = m) : (a = m)
    end
    return exp((a + b) / 2)
end

function fit_ki(L2; freeA = false, σ = σ_BLUR, idx = 1:n, grid = 10 .^ range(-7, -4.7, length = 24), interval = false)
    χ(l) = objective(exp(l), L2; freeA = freeA, σ = σ, idx = idx)[1]
    Qs = [χ(log(k)) for k in grid]
    i  = argmin(Qs)
    lo, hi = log(grid[max(i - 1, 1)]), log(grid[min(i + 1, end)])
    res = optimize(χ, lo, hi, Brent(); abs_tol = 1e-5)
    lmin = Optim.minimizer(res); ki = exp(lmin)
    Q, A, zm = objective(ki, L2; freeA = freeA, σ = σ, idx = idx)
    ν  = length(idx) - (freeA ? 2 : 1)
    χr = Q / ν
    Δ  = max(1.0, χr)                    # inflate when χ²_red > 1
    klo, khi = interval ? (crossing(χ, lmin, Q + Δ, -1), crossing(χ, lmin, Q + Δ, 1)) : (NaN, NaN)
    return (ki = ki, ki_lo = klo, ki_hi = khi, A = A, chi2 = Q, nu = ν, chi2red = χr,
            rms = relrms(A .* zm, z[idx]), zm = A .* zm, grid = grid, Qgrid = Qs)
end

# ---------------------------------------------------------------- run
println("threads = ", nthreads())
t0 = time()
zpeak_all(1.5e-6, 0.39525)
@printf("one forward evaluation (28 currents): %.2f s\n", time() - t0)

# validation: fit the forward model (kᵢ, scale) to the supplied MC CQD column,
# which was generated with the same 2026Q2 geometry (L2 = 0.39525 m) and is
# stored as model/scale.  Recovering kᵢ ≈ 1.5e-6 validates the forward model.
println("\nValidation against supplied CQD_up_mm column (full Monte Carlo, same geometry):")
function valobj(k)
    r = log.(zpeak_all(k, 0.39525)) .- log.(zCQD)
    sum(abs2, r .- sum(r) / n), exp(-sum(r) / n)
end
val_grid = 10 .^ range(-6.5, -5.3, length = 13)
iv   = argmin([valobj(k)[1] for k in val_grid])
resv = optimize(l -> valobj(exp(l))[1], log(val_grid[max(iv-1,1)]), log(val_grid[min(iv+1,end)]), Brent(); abs_tol = 1e-5)
ki_val = exp(Optim.minimizer(resv))
A_val  = valobj(ki_val)[2]
z_val  = A_val .* zpeak_all(ki_val, 0.39525)
@printf("  kᵢ reproducing the MC column = %.4e (scale %.4f, 1/scale %.4f) ; rel RMS forward-vs-MC = %.3f%%\n",
        ki_val, A_val, 1 / A_val, 100 * relrms(z_val, zCQD))

summary = DataFrame()
push!(summary, (L2_m = 0.39525, fit = "validation: unweighted fit to CQD_up_mm column", ki = ki_val,
                ki_lo = NaN, ki_hi = NaN, A = A_val, chi2 = NaN, nu = n - 2, chi2_red = NaN,
                rel_rms_pct = 100 * relrms(z_val, zCQD)))
row(L2, lab, f) = (L2_m = L2, fit = lab, ki = f.ki, ki_lo = f.ki_lo, ki_hi = f.ki_hi, A = f.A,
                   chi2 = f.chi2, nu = f.nu, chi2_red = f.chi2red, rel_rms_pct = 100 * f.rms)
show_fit(lab, f) = @printf("  %-24s kᵢ = %.4e  [%.4e, %.4e]  A = %.4f  χ² = %8.1f  ν = %2d  χ²_red = %6.2f  rel RMS = %.3f%%\n",
                           lab, f.ki, f.ki_lo, f.ki_hi, f.A, f.chi2, f.nu, f.chi2red, 100 * f.rms)
fits = Dict{Symbol,Any}()
for L2 in L2_LIST
    println("\n", "="^72)
    W1, W2 = beam_widths(L2)
    @printf("L2 = %.5f m   beam image: slit %.3f mm ⊗ furnace %.3f mm ⊗ σ %.3f mm\n", L2, W1, W2, σ_BLUR)
    println("  error-weighted fits; kᵢ intervals from χ²_min + max(1, χ²_red)")

    f1 = fit_ki(L2; freeA = false, interval = true)
    f2 = fit_ki(L2; freeA = true, interval = true)
    χk, Ak, zk = objective(1.5e-6, L2; freeA = true)
    fk = (ki = 1.5e-6, ki_lo = NaN, ki_hi = NaN, A = Ak, chi2 = χk, nu = n - 1, chi2red = χk / (n - 1),
          rms = relrms(Ak .* zk, z))
    fl = fit_ki(L2; freeA = false, idx = 1:6, grid = 10 .^ range(-7, -4.5, length = 26), interval = true)
    fits[:pure] = f1; fits[:A] = f2; fits[:low] = fl
    show_fit("pure CQD (A = 1)", f1)
    show_fit("CQD + magnification A", f2)
    show_fit("kᵢ = 1.5e-6, A free", fk)
    show_fit("low field 6 pts (A = 1)", fl)
    for (lab, f) in (("pure CQD (A=1)", f1), ("CQD + magnification A", f2), ("ki=1.5e-6, A free", fk),
                     ("low field 6 pts (A=1)", fl))
        push!(summary, row(L2, lab, f))
    end

    # blur sensitivity
    for σ in (0.100, 0.150)
        fs = fit_ki(L2; freeA = false, σ = σ, grid = 10 .^ range(-6.6, -5.2, length = 12))
        show_fit(@sprintf("σ = %.3f mm (A = 1)", σ), fs)
        push!(summary, row(L2, @sprintf("pure CQD, sigma=%.3f", σ), fs))
    end

    CSV.write(joinpath(OUTDIR, "forward_points.csv"),
              DataFrame(I_A = I, B_T = B, G_Tm = G, z_exp = z, sigma_exp = σz, z_cqd_pure = f1.zm, z_cqd_A = f2.zm,
                        resid_pure_pct = 100 .* (f1.zm ./ z .- 1), resid_A_pct = 100 .* (f2.zm ./ z .- 1),
                        z_MCcol = zCQD, z_forward_valid = z_val))
end
CSV.write(joinpath(OUTDIR, "forward_summary.csv"), summary)

# ---------------------------------------------------------------- figures
gr()
f1, f2 = fits[:pure], fits[:A]
p1 = scatter(B, z, xscale = :log10, yscale = :log10, mc = :white, msc = :black, label = "Experiment",
             xlabel = "B (T)", ylabel = "F=1 peak (mm)", title = "Pure-CQD forward fit", legend = :topleft)
plot!(p1, B, f1.zm, lw = 2, label = @sprintf("A = 1, kᵢ = %.3g", f1.ki))
plot!(p1, B, f2.zm, lw = 2, ls = :dash, label = @sprintf("A = %.3f, kᵢ = %.3g", f2.A, f2.ki))
display(p1)
p2 = plot(B, 100 .* (f1.zm ./ z .- 1), xscale = :log10, marker = :circle, ms = 3, label = "A = 1",
          xlabel = "B (T)", ylabel = "relative residual (%)", title = "Residuals")
plot!(p2, B, 100 .* (f2.zm ./ z .- 1), marker = :circle, ms = 3, label = "A fitted")
hline!(p2, [0], c = :black, label = "")
display(p2)
p3 = plot(1e6 .* f1.grid, f1.Qgrid, xscale = :log10, yscale = :log10, marker = :circle, ms = 2,
          xlabel = "kᵢ (×10⁻⁶)", ylabel = "weighted χ²", title = "Objective vs kᵢ", label = "A = 1")
plot!(p3, 1e6 .* f2.grid, f2.Qgrid, marker = :circle, ms = 2, label = "A profiled")
display(p3)

p4 = plot(B, z ./ G, xscale = :log10, marker = :circle, c = :black, label = "data",
          xlabel = "B (T)", ylabel = "z / G (mm per T/m)", title = "Displacement per gradient")
plot!(p4, B, f1.zm ./ G, lw = 2, label = "pure CQD fit")
for (name, p) in (("fig_cqd_fit", p1), ("fig_cqd_residuals", p2), ("fig_cqd_objective", p3), ("fig_cqd_zG", p4))
    plot!(p, size = (600, 420), margin = 3Plots.mm)
    savefig(p, joinpath(FIGDIR, name * ".pdf")); savefig(p, joinpath(FIGDIR, name * ".png"))
end
println("\nData written to ", OUTDIR)
println("Figures written to ", FIGDIR)
