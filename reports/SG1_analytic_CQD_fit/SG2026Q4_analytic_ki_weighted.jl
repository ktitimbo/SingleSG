# ============================================================================
#  Analytical-dose CQD model fitted to the EXPERIMENTAL data only
#
#     z_j = S_G · G_j · Z(D_j, DIR),     D_j = kᵢ |γe| L1 B_j / v
#     Z(D,DIR) = 1/(DIR+1/2) ∫₀¹ (DIR+1−τ) m(Dτ) dτ
#     m(x)     = coth x − csch²x + x(coth x − 1)csch²x
#
#  Free parameters: kᵢ (analytical induction coefficient) and S_G.
#  The QM_F1_mm and CQD_up_mm columns are NOT used in any fit (reference only).
#
#  Objective (weighted by the experimental errors σ_j):
#     χ² = Σ [ ln(z_j^mod / z_j) / s_j ]²,   s_j = σ_j / z_j
#  S_G is profiled analytically (weighted geometric mean); kᵢ uncertainty
#  from the profile χ²(kᵢ) = χ²_min + 1, also inflated by √χ²_red.
#  Validation: equal weights at L2 = 0.403 m reproduce the PDF tables
#  (L. Wang, 2026-10-07).
#
#  Appendix script of SG1_analytic_CQD_fit_v2.tex (same folder).
#  All outputs stay in this folder: figures in figs_v2/, the window-scan
#  table next to this script.  Only the input is read from data_studies/.
# ============================================================================
using CSV, DataFrames, Optim, LinearAlgebra, Printf, Plots

cd(@__DIR__)
const ROOT   = normpath(joinpath(@__DIR__, "..", ".."))
const CSV_IN = joinpath(ROOT, "data_studies", "FIT2026_ki_scale_20261007T172140485", "model_data.csv")
const OUTDIR = @__DIR__
const FIGDIR = joinpath(OUTDIR, "figs_v2")
mkpath(FIGDIR)

const N_FIT  = 5                        # points in the main low-field fit (from 0.020 A)

# ---------------------------------------------------------------- constants
const γe   = 1.76085963023e11
const kB   = 1.380649e-23
const m_a  = 38.9637064864 * 1.66053906660e-27
const T_ov = 478.15
const a_fl = m_a / (2kB * T_ov)
const v_a  = 3sqrt(π) / (4sqrt(a_fl))
const L1   = 0.070
const L2   = 0.39525
const DIR  = L2 / L1
const κ    = γe * L1 / v_a

# ---------------------------------------------------------------- data
df  = CSV.read(CSV_IN, DataFrame)
I   = df.Current_A; B = df.B_T; G = df.G_Tm
z   = df.Experiment; σz = df.ErrorExperiment
s_log = σz ./ z                         # log-space standard deviation
n   = length(z)

# ---------------------------------------------------------------- response
function gauss_legendre(N)
    β = [k / sqrt(4k^2 - 1) for k in 1:N-1]
    F = eigen(SymTridiagonal(zeros(N), β))
    return F.values, 2 .* F.vectors[1, :] .^ 2
end
const GLx, GLw = gauss_legendre(128)
const τn, τw = (GLx .+ 1) ./ 2, GLw ./ 2
const Cx, Cw = gauss_legendre(64)

function m_ang(x)
    x == 0 && return 1/3
    if x < 0.05
        t = tanh(x)
        return 0.5 * sum(Cw[i] * (1 + Cx[i]) * (Cx[i] + t) / (1 + Cx[i] * t) for i in eachindex(Cx))
    end
    e = exp(-2x); cth = (1 + e) / (1 - e); csh2 = 4e / (1 - e)^2
    return cth - csh2 + x * (2e / (1 - e)) * csh2
end
Zresp(D, dir = DIR) = sum(τw[i] * (dir + 1 - τn[i]) * m_ang(D * τn[i]) for i in eachindex(τn)) / (dir + 0.5)

# ---------------------------------------------------------------- objective
"""Profiled χ² for window idx; weighted=false gives the PDF's equal-weight Q."""
function chi2(ki, idx; weighted = true, dir = DIR)
    w  = weighted ? 1 ./ s_log[idx] .^ 2 : ones(length(idx))
    d  = log.(G[idx] .* Zresp.(ki * κ .* B[idx], dir)) .- log.(z[idx])
    lnS = -sum(w .* d) / sum(w)
    return sum(w .* (d .+ lnS) .^ 2), exp(lnS)
end

relrms(zm, zd) = sqrt(sum(abs2, zm ./ zd .- 1) / length(zd))

"""All local minima of the profile over kᵢ, refined, with Δχ²=1 intervals."""
function fit_window(idx; weighted = true, dir = DIR)
    lk = range(log(1e-8), log(1e-3), length = 500)
    Q  = [chi2(exp(l), idx; weighted = weighted, dir = dir)[1] for l in lk]
    sols = NamedTuple[]
    for i in 2:length(lk)-1
        (Q[i] < Q[i-1] && Q[i] < Q[i+1]) || continue
        res = optimize(l -> chi2(exp(l), idx; weighted = weighted, dir = dir)[1], lk[i-1], lk[i+1], Brent(); abs_tol = 1e-12)
        lmin = Optim.minimizer(res); Qmin = Optim.minimum(res)
        ν    = length(idx) - 2
        # Δχ² = 1 crossing on each side (bounded by the neighbouring barrier)
        function cross(sgn, Δ)
            l = lmin
            for step in 1:4000
                l += sgn * 0.002
                (l < lk[1] || l > lk[end]) && return NaN
                chi2(exp(l), idx; weighted = weighted, dir = dir)[1] > Qmin + Δ && return exp(l)
            end
            NaN
        end
        χr = ν > 0 ? Qmin / ν : NaN
        Δs = max(1.0, χr)                         # inflate when χ²_red > 1
        ki = exp(lmin); S = chi2(ki, idx; weighted = weighted, dir = dir)[2]
        zm = S .* G[idx] .* Zresp.(ki * κ .* B[idx], dir)
        push!(sols, (ki = ki, SG = S, chi2 = Qmin, nu = ν, chi2red = χr,
                     lo1 = cross(-1, 1.0), hi1 = cross(1, 1.0), lo_s = cross(-1, Δs), hi_s = cross(1, Δs),
                     rms = relrms(zm, z[idx]), Dlo = ki * κ * B[idx[1]], Dhi = ki * κ * B[idx[end]]))
    end
    return sols, exp.(lk), Q
end

# ---------------------------------------------------------------- run
@printf("L1 = %.3f m, L2 = %.5f m, DIR = %.4f, v = %.2f m/s, κ = %.6e T⁻¹\n", L1, L2, DIR, v_a, κ)
@printf("relative experimental errors: %.2f%% – %.2f%%\n", 100minimum(s_log), 100maximum(s_log))

rows = DataFrame()
println("\nWindow scan (all start at 0.020 A), weighted by experimental errors:")
println(" pts  I_max   kᵢ (×1e-6)  [Δχ²=1]           [scaled]          S_G       χ²     ν  χ²_red  RMS%")
for np in 4:n
    sols, _, _ = fit_window(collect(1:np))
    for (k, s) in enumerate(sols)
        @printf(" %2d  %.3f  %s %8.4f  [%6.3f, %6.3f]  [%6.3f, %6.3f]  %.6f %8.2f %3d %7.2f %5.2f\n",
                np, I[np], length(sols) > 1 ? "#$k" : "  ", 1e6s.ki, 1e6s.lo1, 1e6s.hi1, 1e6s.lo_s, 1e6s.hi_s,
                s.SG, s.chi2, s.nu, s.chi2red, 100 * s.rms)
        push!(rows, (points = np, I_max = I[np], solution = k, n_solutions = length(sols), ki = s.ki,
                     ki_lo_1sigma = s.lo1, ki_hi_1sigma = s.hi1, ki_lo_scaled = s.lo_s, ki_hi_scaled = s.hi_s,
                     S_G = s.SG, chi2 = s.chi2, nu = s.nu, chi2_red = s.chi2red, rel_rms_pct = 100 * s.rms,
                     D_min = s.Dlo, D_max = s.Dhi))
    end
end
CSV.write(joinpath(OUTDIR, "window_scan_weighted.csv"), rows)

# unweighted (PDF objective) for the N_FIT-point window and whole field, for comparison
for (lab, idx) in (("$N_FIT points, unweighted (PDF Q)", 1:N_FIT), ("all $n, unweighted", 1:n))
    sols, _, _ = fit_window(collect(idx); weighted = false)
    for s in sols
        @printf("%-30s kᵢ = %.4e  S_G = %.6f  rel RMS = %.3f%%\n", lab, s.ki, s.SG, 100 * s.rms)
    end
end

# validation: equal weights at the reference geometry (L2 = 0.403 m) reproduce the PDF tables;
# kept at 6 points (independent of N_FIT) because that is the PDF's window
println("\nValidation against the PDF, L2 = 0.403 m, first 6 points, unweighted:")
for s in fit_window(collect(1:6); weighted = false, dir = 0.403 / L1)[1]
    @printf("  kᵢ = %.4e  S_G = %.7f  D = %.4f–%.4f  Q = %.4e  rel RMS = %.3f%%\n",
            s.ki, s.SG, s.Dlo, s.Dhi, s.chi2, 100 * s.rms)
end

# ---------------------------------------------------------------- figures
gr()
sF, kF, QF = fit_window(collect(1:N_FIT))
sA, kA, QA = fit_window(collect(1:n))
p1 = plot(1e6 .* kF, QF, xscale = :log10, yscale = :log10, c = :black, label = "first $N_FIT points",
          xlabel = "kᵢ (×10⁻⁶)", ylabel = "weighted χ² (S_G profiled)", title = "Profile χ²(kᵢ)")
plot!(p1, 1e6 .* kA, QA, c = :gray, ls = :dash, label = "all $n points")
for s in sF; scatter!(p1, [1e6s.ki], [s.chi2], label = @sprintf("kᵢ = %.3g", s.ki)); end
display(p1)

p2 = scatter(B, z ./ G, yerr = σz ./ G, xscale = :log10, mc = :white, msc = :black, label = "data",
             xlabel = "B (T)", ylabel = "z / G (mm per T/m)", title = "Analytical model vs data")
Bf = 10 .^ range(log10(B[1]), log10(B[end]), length = 200)
for s in sF
    plot!(p2, Bf, s.SG .* Zresp.(s.ki * κ .* Bf), lw = 2, label = @sprintf("%d-pt fit kᵢ=%.3g", N_FIT, s.ki))
end
for s in sA
    plot!(p2, Bf, s.SG .* Zresp.(s.ki * κ .* Bf), lw = 2, ls = :dash, label = @sprintf("%d-pt fit kᵢ=%.3g", n, s.ki))
end
vline!(p2, [B[N_FIT]], c = :gray, ls = :dot, label = "")
display(p2)

sub = filter(r -> r.points <= 14, rows)
p3 = plot(xlabel = "upper current of window (A)", ylabel = "kᵢ (×10⁻⁶)", yscale = :log10,
          title = "Window dependence (weighted)", legend = :right)
for sol in unique(sub.solution)
    r = filter(x -> x.solution == sol && x.n_solutions == 2, sub)
    isempty(r) && continue
    plot!(p3, r.I_max, 1e6 .* r.ki, yerror = (1e6 .* (r.ki .- r.ki_lo_scaled), 1e6 .* (r.ki_hi_scaled .- r.ki)),
          marker = :circle, label = sol == 1 ? "low-dose minimum" : "high-dose minimum")
end
r1 = filter(x -> x.n_solutions == 1, sub)
isempty(r1) || scatter!(p3, r1.I_max, 1e6 .* r1.ki, label = "single minimum")
hline!(p3, [1.5], ls = :dash, c = :gray, label = "1.5×10⁻⁶ (numerical CQD)")
display(p3)

p4 = plot(xlabel = "upper current of window (A)", ylabel = "χ²_red", yscale = :log10,
          title = "Goodness of fit vs window")
for sol in (1, 2)
    r = filter(x -> x.solution == sol, rows)
    plot!(p4, r.I_max, r.chi2_red, marker = :circle, ms = 3, label = sol == 1 ? "minimum 1" : "minimum 2")
end
hline!(p4, [1], c = :black, ls = :dot, label = "")
display(p4)


p5 = plot(xscale = :log10, xlabel = "dose u", ylabel = "R(u)", title = "Response functions", legend = :topleft)
uu = 10 .^ range(-2, 2, length = 300)
plot!(p5, uu, m_ang.(uu), label = "p(u): instantaneous alignment", lw = 2)
plot!(p5, uu, Zresp.(uu), label = @sprintf("R(u): time-averaged, r = %.3f", DIR), lw = 2)
hline!(p5, [1/3, 1], ls = :dot, c = :gray, label = "")
display(p5)

for (name, p) in (("fig_profile", p1), ("fig_zG_fits", p2), ("fig_window_ki", p3), ("fig_window_chi2", p4),
                  ("fig_response", p5))
    plot!(p, size = (620, 430), margin = 3Plots.mm)
    savefig(p, joinpath(FIGDIR, name * ".pdf")); savefig(p, joinpath(FIGDIR, name * ".png"))
end
println("\nData written to ", OUTDIR)
println("Figures written to ", FIGDIR)
