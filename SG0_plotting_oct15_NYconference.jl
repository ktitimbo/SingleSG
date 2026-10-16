# Spin-resolved beam separation vs SG0 current (CairoMakie version).
# Packages: ] add JLD2 CSV DataFrames CairoMakie
# Run in a fresh Julia session (not one where Plots was loaded: it clashes with Makie names).
using JLD2, CSV, DataFrames, CairoMakie

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

    hlines!(ax, 0; color = :black, linestyle = :dash, linewidth = 1.8)

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

    # legend order: data first, then a single "Coquantum dynamics" entry last
    if !isempty(fit_lines)
        push!(handles, fit_lines[1]); push!(labels, "Coquantum dynamics")
    end
    axislegend(ax, handles, labels; position = :lt, framevisible = false)
    return fig
end

# ---------- Usage ----------

curves = load_jld2("spin_resolved_sep.jld2")
fit = load_fit_csv("fit_curves_f7.csv")
I_fit, dD = extract_fit(fit)   # dD[1] = parallel, dD[2] = antiparallel (each has .label and .y)
fig1 = plot_curves(curves; fit = (I_fit, dD))                     # both fits, linear
fig2 = plot_curves(curves; fit = (I_fit, dD[1:1]))                # dD_parallel_px fit only, linear
fig3 = plot_curves(curves; fit = (I_fit, dD), xlog = true)        # both fits, semilog
fig4 = plot_curves(curves; fit = (I_fit, dD[1:1]), xlog = true)   # dD_parallel_px fit only, semilog

save("spin_resolved_sep.pdf", fig1)
save("spin_resolved_sep_parallel.pdf", fig2)
save("spin_resolved_sep_semilog.pdf", fig3)
save("spin_resolved_sep_parallel_semilog.pdf", fig4)

# save_csv(curves, "spin_resolved_sep.csv")
# plot_curves(load_csv("spin_resolved_sep.csv"))