# ============================================================================
#  Analytical CQD fit of the SG1 F=1 peak positions (model_data.csv)
#
#  Re-implements the reduction of "Fits to Kelvin's August–September 2026 SG1
#  Data" (L. Wang, 2026-10-07):
#     Z(D,DIR) = 1/(DIR+1/2) ∫₀¹ (DIR+1-τ) m(Dτ) dτ                 (5)
#     m(x)     = coth x − csch²x + x(coth x − 1)csch²x               (7)
#     D_j      = kᵢ |γe| ℓ1 B_j / v_a                                 (9)
#  Low field : z = S_G G Z(D_j, DIR),           S_G profiled         (9,12)
#  Whole field: z = A_z z_QM Z(D_b + D_j, DIR),  A_z profiled          (18)
#  Objective : Q = Σ [ln(z_mod/z)]²                                   (11)
#
#  The script is run for every drift distance in LD_LIST; the first entry
#  (0.403 m) reproduces the PDF tables as a validation of the implementation.
# ============================================================================
using CSV, DataFrames, Optim, LinearAlgebra, Printf, Plots, Dates

cd(@__DIR__)
const CSV_IN  = joinpath(@__DIR__, "data_studies", "FIT2026_ki_scale_20261007T172140485", "model_data.csv")
const RUN_STAMP = Dates.format(now(), "yyyymmddTHHMMSS")
const OUTDIR  = joinpath(@__DIR__, "data_studies", "FIT2026_analytic_CQD_" * RUN_STAMP)
mkpath(OUTDIR)

# ---------------------------------------------------------------- constants
const γe   = 1.76085963023e11           # rad s⁻¹ T⁻¹
const μe   = 9.2847647043e-24           # J T⁻¹ (electron moment magnitude)
const kB   = 1.380649e-23
const u    = 1.66053906660e-27
const m_a  = 38.9637064864u             # ³⁹K
const T_ov = 478.15                     # K (205 °C)
const ℓ1   = 0.070                      # m, effective SG1 length
const a_fl = m_a / (2kB * T_ov)
const v_a  = 3sqrt(π) / (4sqrt(a_fl))   # effusive flux-weighted mean speed
const κ    = γe * ℓ1 / v_a              # field-to-dose factor (T⁻¹)

const LD_LIST = [0.403, 0.39525]        # validation, requested geometry
const N_LOW   = 6                       # low-field window (0.020–0.045 A)
const KI_FIX  = 1.5e-6                  # Kelvin's approximate value

# ---------------------------------------------------------------- data
df   = CSV.read(CSV_IN, DataFrame)
I    = df.Current_A
B    = df.B_T
G    = df.G_Tm
z    = df.Experiment
zQM  = df.QM_F1_mm
zCQD = df.CQD_up_mm
n    = length(z)

# ---------------------------------------------------------------- quadrature
function gauss_legendre(N)
    β = [k / sqrt(4k^2 - 1) for k in 1:N-1]
    F = eigen(SymTridiagonal(zeros(N), β))
    x = F.values
    w = 2 .* F.vectors[1, :] .^ 2
    return x, w
end
const GLx, GLw = gauss_legendre(128)
const τn = (GLx .+ 1) ./ 2              # nodes on [0,1]
const τw = GLw ./ 2
const Cx, Cw = gauss_legendre(64)       # for small-x angular average

# heart-conditioned angular mean, eq. (6)/(7)
function m_ang(x)
    x == 0 && return 1/3
    if x < 0.05                         # direct quadrature avoids cancellation
        t = tanh(x)
        return 0.5 * sum(Cw[i] * (1 + Cx[i]) * (Cx[i] + t) / (1 + Cx[i] * t) for i in eachindex(Cx))
    end
    e  = exp(-2x)
    cth  = (1 + e) / (1 - e)
    csh2 = 4e / (1 - e)^2
    return cth - csh2 + x * (2e / (1 - e)) * csh2
end

Zresp(D, DIR) = sum(τw[i] * (DIR + 1 - τn[i]) * m_ang(D * τn[i]) for i in eachindex(τn)) / (DIR + 0.5)

relrms(zm, zd) = sqrt(sum(abs2, zm ./ zd .- 1) / length(zd))

# ---------------------------------------------------------------- low-field model
function lowfield_profile(ki, idx, DIR)
    Zj = [Zresp(ki * κ * B[j], DIR) for j in idx]
    r  = log.(G[idx] .* Zj) .- log.(z[idx])
    lnS = -sum(r) / length(r)                       # eq. (12)
    Q  = sum(abs2, r .+ lnS)
    return Q, exp(lnS), Zj
end

function lowfield_minima(idx, DIR)
    lk = range(log(1e-8), log(1e-3), length = 600)
    Qs = [lowfield_profile(exp(l), idx, DIR)[1] for l in lk]
    sols = NamedTuple[]
    for i in 2:length(lk)-1
        if Qs[i] < Qs[i-1] && Qs[i] < Qs[i+1]
            res = optimize(l -> lowfield_profile(exp(l), idx, DIR)[1], lk[i-1], lk[i+1], Brent(); abs_tol = 1e-14)
            ki = exp(Optim.minimizer(res))
            Q, S, Zj = lowfield_profile(ki, idx, DIR)
            zm = S .* G[idx] .* Zj
            push!(sols, (ki = ki, SG = S, Dlo = ki * κ * B[idx[1]], Dhi = ki * κ * B[idx[end]],
                         Q = Q, rms = relrms(zm, z[idx])))
        end
    end
    return sols, exp.(lk), Qs
end

# ---------------------------------------------------------------- whole-field hybrid
# θ = (ln kᵢ, √D_b) keeps D_b ≥ 0 without bounds
hyb_Z(ki, Db, DIR, idx) = [Zresp(Db + ki * κ * B[j], DIR) for j in idx]

function hyb_logQ(ki, Db, DIR, idx; Az = nothing)
    r = log.(zQM[idx] .* hyb_Z(ki, Db, DIR, idx)) .- log.(z[idx])
    lnA = Az === nothing ? -sum(r) / length(r) : log(Az)
    return sum(abs2, r .+ lnA), exp(lnA)
end

function best_of_starts(f, starts)
    best = nothing
    for s in starts
        res = optimize(f, s, NelderMead(), Optim.Options(g_tol = 1e-14, iterations = 20_000))
        res = optimize(f, Optim.minimizer(res), NelderMead(), Optim.Options(g_tol = 1e-15, iterations = 20_000))
        (best === nothing || Optim.minimum(res) < Optim.minimum(best)) && (best = res)
    end
    return best
end

const STARTS2 = [[log(k), sqrt(d)] for k in (3e-7, 1e-6, 3e-6, 1e-5) for d in (0.1, 1.0, 2.0)]

function hyb_fit(DIR, idx; Az = nothing)
    f(θ) = hyb_logQ(exp(θ[1]), θ[2]^2, DIR, idx; Az = Az)[1]
    res = best_of_starts(f, STARTS2)
    ki, Db = exp(Optim.minimizer(res)[1]), Optim.minimizer(res)[2]^2
    Q, A = hyb_logQ(ki, Db, DIR, idx; Az = Az)
    zm = A .* zQM[idx] .* hyb_Z(ki, Db, DIR, idx)
    return (ki = ki, Az = A, Db = Db, Q = Q, rms = relrms(zm, z[idx]), zm = zm)
end

function hyb_fit_kifixed(ki, DIR, idx)
    res = optimize(s -> hyb_logQ(ki, s^2, DIR, idx)[1], 0.0, 3.0, Brent())
    Db = Optim.minimizer(res)^2
    Q, A = hyb_logQ(ki, Db, DIR, idx)
    zm = A .* zQM[idx] .* hyb_Z(ki, Db, DIR, idx)
    return (ki = ki, Az = A, Db = Db, Q = Q, rms = relrms(zm, z[idx]), zm = zm)
end

# alternative objectives (A_z fitted numerically): :rel or :mm
function hyb_fit_alt(DIR, idx, kind)
    function f(θ)
        zm = exp(θ[3]) .* zQM[idx] .* hyb_Z(exp(θ[1]), θ[2]^2, DIR, idx)
        kind === :rel ? sum(abs2, zm ./ z[idx] .- 1) : sum(abs2, zm .- z[idx])
    end
    res = best_of_starts(f, [[s[1], s[2], log(1.04)] for s in STARTS2])
    θ = Optim.minimizer(res)
    ki, Db, A = exp(θ[1]), θ[2]^2, exp(θ[3])
    zm = A .* zQM[idx] .* hyb_Z(ki, Db, DIR, idx)
    return (ki = ki, Az = A, Db = Db, rms = relrms(zm, z[idx]),
            rms_mm = sqrt(sum(abs2, zm .- z[idx]) / length(idx)))
end

# ---------------------------------------------------------------- run
@printf("v_a = %.3f m/s   |γe|ℓ1/v_a = %.9e T⁻¹\n", v_a, κ)
@printf("B/G = %.9e m (std/mean %.1e)\n", sum(B ./ G) / n, (maximum(B ./ G) - minimum(B ./ G)) / (sum(B ./ G) / n))

summary = DataFrame()
allidx  = collect(1:n)
lowidx  = collect(1:N_LOW)
results = Dict{Float64,Any}()

for Ld in LD_LIST
    DIR = Ld / ℓ1
    println("\n", "="^72)
    @printf("L_d = %.5f m   DIR = %.6f   T_d = %.4e s   T_1 = %.4e s\n", Ld, DIR, Ld / v_a, ℓ1 / v_a)
    SGmech = 1e3 * μe * ℓ1 * (Ld + ℓ1 / 2) / (m_a * v_a^2)
    @printf("S_G^mech = %.7f mm/(T m⁻¹)\n", SGmech)
    @printf("Z(0)=%.6f  C1=%.6f  C2=%.6f\n", Zresp(0.0, DIR), (3DIR + 1) / (9(DIR + 0.5)), (4DIR + 1) / (90(DIR + 0.5)))

    # --- low field
    sols, kgrid, Qgrid = lowfield_minima(lowidx, DIR)
    println("\nLow field (first $N_LOW points):")
    for s in sols
        @printf("  kᵢ = %.4e  S_G = %.7f  D = %.4f–%.4f  Q = %.4e  rel RMS = %.3f%%\n",
                s.ki, s.SG, s.Dlo, s.Dhi, s.Q, 100s.rms)
    end

    # --- window sensitivity
    println("\nWindow sensitivity (lower / higher dose kᵢ, ×1e-6):")
    win = DataFrame(points = Int[], I_upper = Float64[], ki_low = Float64[], ki_high = Float64[], smaller = String[])
    for np in 4:10
        ss, _, _ = lowfield_minima(collect(1:np), DIR)
        if length(ss) >= 2
            lo, hi = ss[1], ss[end]
            push!(win, (np, I[np], lo.ki, hi.ki, lo.Q < hi.Q ? "lower" : "higher"))
            @printf("  %2d  %.3f A   %.3f   %.3f   %s\n", np, I[np], 1e6lo.ki, 1e6hi.ki, lo.Q < hi.Q ? "lower" : "higher")
        else
            @printf("  %2d  %.3f A   only %d minimum\n", np, I[np], length(ss))
        end
    end

    # --- whole field
    hf  = hyb_fit(DIR, allidx)
    hf1 = hyb_fit(DIR, allidx; Az = 1.0)
    hfk = hyb_fit_kifixed(KI_FIX, DIR, allidx)
    hr  = hyb_fit_alt(DIR, allidx, :rel)
    hm  = hyb_fit_alt(DIR, allidx, :mm)
    println("\nWhole field hybrid z = A_z z_QM Z(D_b + D_j):")
    for (lab, h) in (("fitted A_z", hf), ("A_z = 1", hf1), ("kᵢ = 1.5e-6", hfk))
        @printf("  %-12s kᵢ = %.5e  A_z = %.5f  D_b = %.5f  Q = %.7f  rel RMS = %.3f%%\n",
                lab, h.ki, h.Az, h.Db, h.Q, 100h.rms)
    end
    @printf("  dose span D_j = %.4f–%.4f ; D_b+D_j = %.4f–%.4f\n",
            hf.ki * κ * B[1], hf.ki * κ * B[end], hf.Db + hf.ki * κ * B[1], hf.Db + hf.ki * κ * B[end])
    @printf("  rel-resid objective: kᵢ = %.4e  A_z = %.5f  rel RMS = %.3f%%\n", hr.ki, hr.Az, 100hr.rms)
    @printf("  mm objective       : kᵢ = %.4e  A_z = %.5f  rel RMS = %.3f%%  RMS = %.6f mm\n",
            hm.ki, hm.Az, 100hm.rms, hm.rms_mm)

    results[Ld] = (DIR = DIR, sols = sols, kgrid = kgrid, Qgrid = Qgrid, hf = hf, hf1 = hf1, hfk = hfk,
                   hr = hr, hm = hm, win = win, SGmech = SGmech)

    tag = @sprintf("Ld%05d", round(Int, 1e5Ld))
    CSV.write(joinpath(OUTDIR, "window_sensitivity_$tag.csv"), win)
    CSV.write(joinpath(OUTDIR, "wholefield_points_$tag.csv"),
              DataFrame(I_A = I, B_T = B, z_exp = z, z_hyb = hf.zm, z_hyb_Az1 = hf1.zm,
                        z_hyb_ki15 = hfk.zm, resid_pct = 100 .* (hf.zm ./ z .- 1)))

    lowrows = [(fit = "Low field " * (k == 1 ? "lower dose" : "higher dose"), ki = s.ki, Az_or_SG = s.SG, Db = NaN,
                Q = s.Q, rel_rms_pct = 100s.rms) for (k, s) in enumerate(sols)]
    for r in vcat(lowrows,
                  [(fit = "Whole field, fitted A_z", ki = hf.ki, Az_or_SG = hf.Az, Db = hf.Db, Q = hf.Q, rel_rms_pct = 100hf.rms),
                   (fit = "Whole field, A_z = 1", ki = hf1.ki, Az_or_SG = hf1.Az, Db = hf1.Db, Q = hf1.Q, rel_rms_pct = 100hf1.rms),
                   (fit = "Whole field, ki = 1.5e-6", ki = hfk.ki, Az_or_SG = hfk.Az, Db = hfk.Db, Q = hfk.Q, rel_rms_pct = 100hfk.rms),
                   (fit = "Whole field, rel. residuals", ki = hr.ki, Az_or_SG = hr.Az, Db = hr.Db, Q = NaN, rel_rms_pct = 100hr.rms),
                   (fit = "Whole field, mm residuals", ki = hm.ki, Az_or_SG = hm.Az, Db = hm.Db, Q = NaN, rel_rms_pct = 100hm.rms)])
        push!(summary, merge((Ld_m = Ld, DIR = DIR), r))
    end
end

CSV.write(joinpath(OUTDIR, "summary.csv"), summary)

# ---------------------------------------------------------------- figures (requested L_d)
Ld  = LD_LIST[end]
R   = results[Ld]
lab = @sprintf("L_d = %.5f m", Ld)
gr()
p1 = scatter(B, z, xscale = :log10, yscale = :log10, label = "Experiment", mc = :white, msc = :black,
             xlabel = "B (T)", ylabel = "F=1 peak (mm)", title = "Whole field  ($lab)", legend = :topleft)
plot!(p1, B, zCQD, label = "Supplied CQD", lw = 1.5)
plot!(p1, B, zQM, label = "Supplied QM", ls = :dash, c = :gray)
plot!(p1, B, R.hf.zm, label = @sprintf("Hybrid fit, kᵢ=%.3g", R.hf.ki), lw = 2)

p2 = plot(R.kgrid .* 1e6, R.Qgrid, xscale = :log10, yscale = :log10, label = "Q(kᵢ), S_G profiled",
          xlabel = "kᵢ (×10⁻⁶)", ylabel = "Σ ln² residuals", title = "Low-field profile (first $N_LOW points)", c = :black)
for s in R.sols
    scatter!(p2, [1e6s.ki], [s.Q], label = @sprintf("kᵢ=%.3g", s.ki))
end
vline!(p2, [1.5], ls = :dash, label = "1.5×10⁻⁶")

p3 = plot(B, 100 .* (R.hf.zm ./ z .- 1), xscale = :log10, marker = :circle, ms = 3, label = "fitted A_z",
          xlabel = "B (T)", ylabel = "relative residual (%)", title = "Whole-field residuals")
plot!(p3, B, 100 .* (R.hf1.zm ./ z .- 1), marker = :circle, ms = 3, label = "A_z = 1")
hline!(p3, [0], c = :black, label = "")

p4 = scatter(B[lowidx], z[lowidx], label = "First $N_LOW points", mc = :white, msc = :black,
             xlabel = "B (T)", ylabel = "F=1 peak (mm)", title = "Low-field solutions")
for s in R.sols
    Bf = range(B[1], B[N_LOW], length = 100)
    plot!(p4, Bf, [s.SG * (b / (B[1] / G[1])) * Zresp(s.ki * κ * b, R.DIR) for b in Bf] .* 1.0,
          label = @sprintf("kᵢ=%.3g", s.ki))
end
fig = plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900), margin = 5Plots.mm)
savefig(fig, joinpath(OUTDIR, "fit_overview_Ld39525.png"))

# individual panels for the report
p5 = plot(xscale = :log10, xlabel = "dose u", ylabel = "R(u)", title = "Response functions", legend = :topleft)
uu = 10 .^ range(-2, 2, length = 300)
plot!(p5, uu, m_ang.(uu), label = "p(u): instantaneous alignment", lw = 2)
plot!(p5, uu, [Zresp(x, R.DIR) for x in uu], label = @sprintf("R(u): time-averaged, r = %.3f", R.DIR), lw = 2)
hline!(p5, [1/3, 1], ls = :dot, c = :gray, label = "")

p6 = plot(B, z ./ G, xscale = :log10, marker = :circle, c = :black, label = "z / G (data)",
          xlabel = "B (T)", ylabel = "z / G  (mm per T/m)", title = "Displacement per unit gradient")
p7 = scatter(B, z ./ zQM, xscale = :log10, mc = :white, msc = :black, label = "data / QM",
             xlabel = "B (T)", ylabel = "ratio to QM curve", title = "Ratio to the QM reference", legend = :bottomright)
plot!(p7, B, zCQD ./ zQM, label = "supplied CQD / QM", lw = 1.5)
plot!(p7, B, R.hf.zm ./ zQM, label = "fitted hybrid factor A·R(u₀+u)", lw = 1.5)

p8 = plot(xscale = :log10, xlabel = "kᵢ (×10⁻⁶)", ylabel = "χ² (profiled)", title = "Whole-field profile over kᵢ")
kk = 10 .^ range(log10(2e-7), log10(1e-5), length = 80)
χk = [hyb_fit_kifixed(k, R.DIR, allidx).Q for k in kk]
plot!(p8, 1e6 .* kk, χk, label = "A, u₀ optimized at each kᵢ", c = :black)
scatter!(p8, [1e6R.hf.ki], [R.hf.Q], label = @sprintf("best kᵢ = %.3g", R.hf.ki))
scatter!(p8, [1.5], [R.hfk.Q], label = "kᵢ = 1.5×10⁻⁶")

for (name, p) in (("fig_wholefield", p1), ("fig_lowfield_profile", p2), ("fig_residuals", p3),
                  ("fig_lowfield_fits", p4), ("fig_response", p5), ("fig_z_over_G", p6),
                  ("fig_ratio_QM", p7), ("fig_wholefield_profile", p8))
    plot!(p, size = (600, 420), margin = 3Plots.mm)
    savefig(p, joinpath(OUTDIR, name * ".pdf"))
end
println("\nOutputs written to ", OUTDIR)
