# Spin-resolved beam separation vs SG0 current (CairoMakie version).
# Packages: ] add JLD2 CSV DataFrames CairoMakie
# Run in a fresh Julia session (not one where Plots was loaded: it clashes with Makie names).
using JLD2, CSV, DataFrames, CairoMakie
using LinearAlgebra
# Set the working directory to the current location
cd(@__DIR__) ;


set_theme!(theme_latexfonts())   # Computer Modern (New Computer Modern, bundled with Makie)

# ---------- Loaders: both return a Vector of (label, I, y, yerror) ----------

# JLD2: every group named "idx_N" holds one curve (fields: label, I, y, yerror)
function load_jld2(file)
    jldopen(file, "r") do f
        keys_ = sort(filter(k -> startswith(k, "idx_"), collect(keys(f))))
        [(label = f[k]["label"], I = f[k]["I"], y = f[k]["y"], yerror = f[k]["yerror"])
         for k in keys_]
    end
end

# CSV: long format with columns  label, I, y, yerror  (one row per point)
function load_csv(file)
    df = CSV.read(file, DataFrame)
    [(label = g.label[1], I = g.I, y = g.y, yerror = g.yerror)
     for g in groupby(df, :label)]
end

# Helper to create a CSV from the curves (handy to test load_csv)
function save_csv(curves, file)
    df = vcat([DataFrame(label = c.label, I = c.I, y = c.y, yerror = c.yerror) for c in curves]...)
    CSV.write(file, df)
end

# Fit-curves CSV (wide format, one column per quantity): returns a DataFrame and prints the headers
function load_fit_csv(file)
    df = CSV.read(file, DataFrame)
    println("Columns in ", basename(file), " (", nrow(df), " rows):")
    for (i, name) in enumerate(names(df))
        println("  ", i, ". ", name)
    end
    return df
end

# Extract the current I0_A and the dD_parallel_px / dD_antiparallel_px columns from the fit DataFrame
function extract_fit(df)
    I = df.I0_A
    cols = ["dD_parallel_px", "dD_antiparallel_px"]
    dD = [(label = c, y = df[!, c]) for c in cols]
    println("Extracted I0_A and: ", join(cols, ", "))
    return I, dD
end

# ---------- CQD model and fit ----------
# Double-stage (SG0 -> SG1) CQD model, fitted directly to the loaded `curves`.
# Eq. n  = fit_20260706/cqd_sg0_fit_20260706.pdf;  [n] = source note cqd_sg0_analytical_model_2026_10_13F.pdf.
# Same algorithm as fit_20260706/cqd_model.js + run_fits.js ("parallel first, then borrow k_i", PDF Sec. 4).
# SG0 is a uniform field: standard QM ("Existing models") predicts DeltaD = 0 (source note, Sec. 23).

# Fixed calibration inputs (PDF Sec. 3)
const D0_PX = 13.0      # absolute zero-current separation (px): sets the scale of the model, Eq. 18
const I_STAR = 1.03869  # B1 l1 / (beta0 l0) in A: SG0/SG1 dose link, Eq. 9
const K_PIX = 6.47      # 2 mu_e G1 T1^2 / (M_a d_pix) in px, Eq. 19
const V_A = 600.5       # beam speed (m/s)
const GAMMA_E = 1.76086e11          # |gamma_e| (rad s^-1 T^-1)
const BL1 = 0.018931 * 0.070        # SG1 field-length product B1 l1 (T m)
ki_from_dose(D1) = D1 * V_A / (GAMMA_E * BL1)   # k_i = Dc1 v_a / (|gamma_e| BL1), Eq. 7

# Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration on P_n)
function gauss_legendre(n)
    x, w = zeros(n), zeros(n)
    for i in 1:n
        z = cos(pi * (i - 0.25) / (n + 0.5)); dp = 0.0
        for _ in 1:100
            p1, p2 = 1.0, 0.0
            for j in 1:n
                p1, p2 = ((2j - 1) * z * p1 - (j - 1) * p2) / j, p1
            end
            dp = n * (z * p1 - p2) / (z^2 - 1)
            dz = p1 / dp; z -= dz
            abs(dz) < 1e-15 && break
        end
        x[i], w[i] = z, 2 / ((1 - z^2) * dp^2)
    end
    return x, w
end
const GL48 = gauss_legendre(48)   # ensemble integrals over the entrance cosine c
const GL24 = gauss_legendre(24)   # SG1 time integral of Eq. 13
function integ(f, a, b, (X, W) = GL48)
    h, m = (b - a) / 2, (b + a) / 2
    return h * sum(W[i] * f(m + h * X[i]) for i in eachindex(X))
end

# CQD transfer map (Eq. 8; [145]): translation of artanh(c) by q*D
transfer(c, q, D) = (t = tanh(D); (c + q * t) / (1 + q * c * t))

# Normalized SG1 response of one atom with entrance cosine c, q1 = +1, rho = DIR*Dc1 (Eq. 13; [204],[211])
function z_ex(c, D1, rho)
    C = atanh(clamp(c, -1 + 1e-15, 1 - 1e-15))
    return integ(l -> (D1 + rho - l) * tanh(C + l), 0.0, D1, GL24) / (rho * D1 + D1^2 / 2)
end

# Heart-weighted ensemble responses (Eqs. 14-16; [217]-[221], [251])
Z_par(D0, D1, rho) = 0.5 * integ(c -> (1 + c) * z_ex(transfer(c, 1, D0), D1, rho), -1.0, 1.0)
Z_ret(D0, D1, rho) = 0.5 * integ(c -> (1 + c) * z_ex(-transfer(c, 1, D0), D1, rho), -1.0, 1.0)
function Z_geom(D0, D1, rho)   # strict mirror-and-rebranch rule; L+- have kinks at c = -/+ tanh(D0/2)
    g = function (c)
        Tp, Tm = transfer(c, 1, D0), transfer(c, -1, D0)
        Lp, Lm = 1 + min(c, -Tp), max(0.0, -Tm - c)
        return Lp * z_ex(-Tp, D1, rho) + (Lm > 0 ? Lm * z_ex(-Tm, D1, rho) : 0.0)
    end
    k = tanh(D0 / 2)
    return 0.5 * (integ(g, -1.0, -k) + integ(g, -k, k) + integ(g, k, 1.0))
end

# DIR = Td/T1 from the 13-px baseline, 13 = K_pix (DIR + 1/2) Z0(Dc1, DIR)  (Eq. 19; [249]), by bisection
function solve_DIR(D1)
    g(DIR) = K_PIX * (DIR + 0.5) * Z_par(0.0, D1, DIR * D1) - D0_PX
    a, b = 0.01, 200.0
    for _ in 1:80
        m = (a + b) / 2
        g(a) * g(m) <= 0 ? (b = m) : (a = m)
    end
    return (a + b) / 2
end

# Responses at currents I for SG1 dose D1 and SG0 divisor f; Dc0 = Dc1 I0 / (f I_star) (Eqs. 9, 21)
function cqd_responses(D1, f, I)
    DIR = solve_DIR(D1); rho = DIR * D1
    Z0 = Z_par(0.0, D1, rho)
    D0 = D1 .* I ./ (f * I_STAR)
    return (; DIR, Z0, D0, zp = [Z_par(d, D1, rho) for d in D0],
              zg = [Z_geom(d, D1, rho) for d in D0], zr = [Z_ret(d, D1, rho) for d in D0])
end
# Model curves DeltaD = 13 [Z/Z0 - 1] (Eq. 18), antiparallel mixture with weight w (Eq. 17)
dD_par(r) = D0_PX .* (r.zp ./ r.Z0 .- 1)
dD_anti(r, w) = D0_PX .* (((1 .- w) .* r.zg .+ w .* r.zr) ./ r.Z0 .- 1)
w_ret(D0, p, form) = form === :lin ? min.(1.0, p .* D0) : 1 .- exp.(-p .* D0)   # Eq. 24

chi2(model, y, s) = sum(((model .- y) ./ s) .^ 2)   # Eq. 22

# Golden-section minimization of a 1-D function on [a, b]
function golden(fn, a, b; n = 60)
    r = (sqrt(5) - 1) / 2
    c, d = b - r * (b - a), a + r * (b - a); fc, fd = fn(c), fn(d)
    for _ in 1:n
        if fc < fd
            b, d, fd = d, c, fc; c = b - r * (b - a); fc = fn(c)
        else
            a, c, fc = c, d, fd; d = a + r * (b - a); fd = fn(d)
        end
    end
    return fc < fd ? (c, fc) : (d, fd)
end

"""
    fit_cqd(curves; f = 7.0, wform = :lin, Igrid = 0:0.01:3.7)

Two-step pure-CQD fit of the spin-resolved separations (PDF Sec. 4):
1. parallel set only: minimize chi2 over Dc1 (log scan on [1e-3, 5], then golden section) -> k_i,
   with the profile Delta-chi2 = 1 interval (Eqs. 22-23);
2. antiparallel set with k_i fixed: fit only the retained-label weight
   (`wform = :lin`: w = kappa*Dc0, `:exp`: w = 1 - exp(-RLC*Dc0)), Eq. 24.
`f` is the fixed SG0 field divisor (f = 7: Revision F). Returns the fit summary and the model curves
on `Igrid` in the same shape as `extract_fit`: `(I_fit, dD)`.
"""
function fit_cqd(curves; f = 7.0, wform = :lin, Igrid = 0:0.01:3.7)
    ia = findfirst(c -> occursin("antiparallel", c.label), curves)
    ip = findfirst(c -> !occursin("antiparallel", c.label), curves)
    par, anti = curves[ip], curves[ia]

    # Step 1: parallel set, one parameter Dc1
    chi_par(D1) = chi2(dD_par(cqd_responses(D1, f, par.I)), par.y, par.yerror)
    xs = range(log(1e-3), log(5.0); length = 41)
    cs = [chi_par(exp(x)) for x in xs]
    ib = argmin(cs)
    x1, chi_p = golden(x -> chi_par(exp(x)), xs[max(1, ib - 1)], xs[min(end, ib + 1)]; n = 50)
    D1 = exp(x1)
    function side(up)   # profile Delta-chi2 = 1 bound by bisection in log(Dc1)
        a, b = D1, up ? 4D1 : D1 / 4
        chi_par(b) - chi_p < 1 && return NaN
        for _ in 1:50
            m = sqrt(a * b); chi_par(m) - chi_p < 1 ? (a = m) : (b = m)
        end
        return sqrt(a * b)
    end
    D1_lo, D1_hi = side(false), side(true)

    # Step 2: antiparallel set with Dc1 (k_i) fixed; pure CQD (w = 0) for reference
    ra = cqd_responses(D1, f, anti.I)
    chi_anti(p) = chi2(dD_anti(ra, w_ret(ra.D0, p, wform)), anti.y, anti.yerror)
    p, chi_a = golden(chi_anti, 0.0, 5.0)
    h = max(1e-4, 1e-3 * p)
    p_err = sqrt(2 / ((chi_anti(p + h) - 2chi_anti(p) + chi_anti(p - h)) / h^2))
    chi_pure = chi2(dD_anti(ra, zeros(length(anti.I))), anti.y, anti.yerror)

    pname = wform === :lin ? "kappa" : "RLC"
    println("CQD fit (f = $f, $(length(par.I)) + $(length(anti.I)) points)")
    println("  step 1, parallel : Dc1 = ", round(D1; sigdigits = 4), ",  k_i = ", round(ki_from_dose(D1); sigdigits = 3),
            " (", round(ki_from_dose(D1_lo); sigdigits = 3), " - ", round(ki_from_dose(D1_hi); sigdigits = 3),
            "),  chi2 = ", round(chi_p; digits = 2), ",  DIR = ", round(ra.DIR; digits = 3))
    println("  step 2, antipar. : $pname = ", round(p; digits = 4), " +- ", round(p_err; digits = 4),
            ",  chi2 = ", round(chi_a; digits = 2), "   (pure CQD: chi2 = ", round(chi_pure; digits = 1), ")")

    # Model curves on the plotting grid
    I_fit = collect(Igrid)
    rg = cqd_responses(D1, f, I_fit)
    dD = [(label = "dD_parallel_px", y = dD_par(rg)),
          (label = "dD_antiparallel_px", y = dD_anti(rg, w_ret(rg.D0, p, wform))),
          (label = "dD_antiparallel_pureCQD_px", y = dD_anti(rg, zeros(length(I_fit))))]
    summary = (; f, D1, ki = ki_from_dose(D1), ki_lo = ki_from_dose(D1_lo), ki_hi = ki_from_dose(D1_hi),
               chi2_par = chi_p, DIR = ra.DIR, wform, p, p_err, chi2_anti = chi_a, chi2_anti_pure = chi_pure)
    return summary, I_fit, dD
end

# ---------- Plot ----------

# xlog = true: log-scale current axis; points with I <= 0 are masked out
function plot_curves(curves; fit = nothing, xlog = false)   # fit = (I_fit, dD) from extract_fit
    colors  = [:red, :blue]
    markers = [:diamond, :rect]

    keep(I) = xlog ? I .> 0 : trues(length(I))
    xmin = xlog ? 0.7 * minimum(minimum(c.I[c.I .> 0]) for c in curves) : -0.15

    fig = Figure(size = (1000, 560), fontsize = 20)
    ax = Axis(fig[1, 1];
              xlabel = "SG0 current (A)", ylabel = "Relative beam separation (pixels)",
              xscale = xlog ? log10 : identity,
              xticks = xlog ? Makie.automatic : (0:3), yticks = -2:5,
              limits = (xmin, 3.8, -2.7, 5.4),
              xminorticksvisible = true, yminorticksvisible = true,
              xminorgridvisible = true, yminorgridvisible = true,
              xminorticks = IntervalsBetween(xlog ? 9 : 5), yminorticks = IntervalsBetween(2),
              xgridcolor = (:black, 0.12), ygridcolor = (:black, 0.12),
              xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05))

    zero_line = hlines!(ax, 0; color = :black, linestyle = :dash, linewidth = 1.8)

    # fit curves in black (drawn first so the data sit on top)
    fit_lines = []
    if fit !== nothing
        I_fit, dD = fit
        m = keep(I_fit)
        for d in dD
            push!(fit_lines, lines!(ax, I_fit[m], d.y[m]; color = :black, linewidth = 2.5))
        end
    end

    handles, labels = [], String[]
    for (i, c) in enumerate(curves)
        m = keep(c.I)
        I, y, yerr = c.I[m], c.y[m], c.yerror[m]
        errorbars!(ax, I, y, yerr; color = (colors[i], 0.7), whiskerwidth = 10, linewidth = 2)
        sc = Makie.scatter!(ax, I, y; color = :transparent, marker = markers[i],
                 strokecolor = colors[i], strokewidth = 2, markersize = 16)
        push!(handles, sc); push!(labels, c.label)
    end

    # legend order: data first, then a single "Coquantum dynamics" entry, then the zero line
    if !isempty(fit_lines)
        push!(handles, fit_lines[1]); push!(labels, "Coquantum dynamics")
    end
    push!(handles, zero_line); push!(labels, "Existing models")
    axislegend(ax, handles, labels; position = :lt, framevisible = false,
               patchsize = (40, 20))   # wider legend patches so the solid/dashed lines read clearly
    return fig
end

# ---------- Usage ----------

curves = load_jld2(joinpath(@__DIR__,"spin_resolved_sep_20260706.jld2"))
# Fit the CQD model to the loaded data (f = 7 fixed; parallel first, then k_i borrowed)
fitres, I_fit, dD = fit_cqd(curves; f = 7.0)   # dD[1] = parallel, dD[2] = antiparallel, dD[3] = pure CQD antiparallel
dD = dD[1:2]                                   # plot the two fitted curves
# To plot the precomputed fit instead:
# I_fit, dD = extract_fit(load_fit_csv(joinpath(@__DIR__, "fit_20260706", "fit_curves_f7_20260706.csv")))
fig1 = plot_curves(curves; fit = (I_fit, dD))                     # both fits, linear
fig2 = plot_curves(curves; fit = (I_fit, dD[1:1]))                # dD_parallel_px fit only, linear
fig3 = plot_curves(curves; fit = (I_fit, dD), xlog = true)        # both fits, semilog
fig4 = plot_curves(curves; fit = (I_fit, dD[1:1]), xlog = true)   # dD_parallel_px fit only, semilog

save("spin_resolved_sep.pdf", fig1)
save("spin_resolved_sep.svg", fig1)
save("spin_resolved_sep_parallel.pdf", fig2)
save("spin_resolved_sep_semilog.pdf", fig3)
save("spin_resolved_sep_parallel_semilog.pdf", fig4)

# save_csv(curves, "spin_resolved_sep.csv")
# plot_curves(load_csv("spin_resolved_sep.csv"))