# Kelvin Titimbo, Xukun Lin, S. Suleyman Kahraman, and Lihong V. Wang
# California Institute of Technology
# December 2025

############## EXPERIMENT ANALYSIS PREAMBLE ##############
# Headless/Windows-safe GR: set before using Plots
# if !haskey(ENV, "GKSwstype")
#     ENV["GKSwstype"] = "100"  # offscreen; avoids popup windows/crashes
# end
# Plotting backend and general appearance settings
using Plots; gr()
# Set default plot aesthetics
const IN_NOTEBOOK = isdefined(Main, :IJulia);
Plots.default(
    show=IN_NOTEBOOK, dpi=800, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
# Data I/O and numerical tools
using JLD2
using LinearAlgebra
using DataStructures, OrderedCollections
using Statistics, StatsBase
using BSplineKit, Optim
# Aesthetics and output formatting
using Colors, ColorSchemes
using Printf, LaTeXStrings, PrettyTables
using CSV, DataFrames, DelimitedFiles
# Time-stamping/logging
using Dates
using Alert
const T_START = Dates.now()
# Custom modules
include("./Modules/DataReading.jl");
include("./Modules/MyExperimentalAnalysis.jl");
using .MyExperimentalAnalysis;
# Set the working directory to the current location
cd(@__DIR__) ;
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
# For Plots
FIG_EXT  = "png"   # could be "pdf", "svg", etc.
SAVE_FIG = false
MyExperimentalAnalysis.SAVE_FIG = SAVE_FIG;
MyExperimentalAnalysis.FIG_EXT  = FIG_EXT;
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
const OUTDIR    = joinpath(@__DIR__, "data_studies", "peak_vs_intensity", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# ---------------------------------------------------------------------------
effective_cam_pixelsize_z = 6.5e-6;
z_pixels = 2560;
# Data processing
nz_binning = 2;
λ0 = 0.01;
half_max = true;
# Precompute z-axes (mm)
z_full_mm   = 1e3 .* pixel_positions(z_pixels, 1,  effective_cam_pixelsize_z);
z_binned_mm = 1e3 .* pixel_positions(z_pixels, nz_binning, effective_cam_pixelsize_z);
# ---------------------------------------------------------------------------
data_directories = ["20250814", "20250820", "20250825","20250919","20251002","20251003","20251006"];

data_results = Dict{String, NamedTuple{(:fw_centroid_mm, :framewise_mm, :δframewise_mm), Tuple{Any, Any, Any}}}()
for dir in data_directories
    res = DataReading.find_report_data(
        joinpath(@__DIR__, "analysis_data");
        wanted_data_dir = dir,
        wanted_binning  = nz_binning,
        wanted_smooth   = λ0,
    )

    data_results[dir] = (
        fw_centroid_mm = res.fw_centroid_mm,
        framewise_mm   = res.framewise_mm,
        δframewise_mm  = res.δframewise_mm,
    )
end
plot(data_directories,[data_results[dir].fw_centroid_mm[1] for dir in data_directories],
    yerr = [data_results[dir].fw_centroid_mm[2] for dir in data_directories],
    seriestype=:scatter,
    marker=(:circle,:white,3, stroke(1,:red)),
    label="Centroid"
)
plot!(
    yaxis=(L"$z_{c}$ (mm)",:identity),
    xlim=(-1,length(data_directories)+2),
    xticks = (1:length(data_directories), data_directories),
    xminorticks = false,
    xrotation=65,
    bottom_margin=-2mm,
    left_margin = 6mm,
    size=(350,720)
)

Ics = Vector{Vector{Float64}}(undef, length(data_directories));
for (i, dir) in enumerate(data_directories)
    data = load(joinpath(@__DIR__, dir, "data_processed.jld2"), "data")
    Ics[i] = data[:Currents]
end

function cluster_by_tolerance(Ics; tol=0.10)
    # Flatten values, dataset ids, and index-in-dataset
    vals  = Float64[]
    vidx  = Int[]
    iidx  = Int[]

    for (j, vec) in enumerate(Ics)
        for (k, v) in enumerate(vec)
            push!(vals, v)
            push!(vidx, j)
            push!(iidx, k)
        end
    end

    # Sort all flattened values
    p = sortperm(vals)
    v  = vals[p]
    g  = vidx[p]
    idx = iidx[p]

    # Clustering
    clusters = Vector{Vector{NamedTuple{(:val, :set, :idx), Tuple{Float64,Int,Int}}}}()

    current = [(val=v[1], set=g[1], idx=idx[1])]

    for i in 2:length(v)
        prev = v[i-1]
        curr = v[i]

        if abs(curr - prev) ≤ tol * min(curr, prev)
            push!(current, (val=curr, set=g[i], idx=idx[i]))
        else
            push!(clusters, current)
            current = [(val=curr, set=g[i], idx=idx[i])]
        end
    end

    push!(clusters, current)

    for c in clusters
        sort!(c, by = x -> x.set)
    end

    # Keep only clusters appearing in ≥2 datasets
    multi = [c for c in clusters if length(unique(getfield.(c, :set))) ≥ 2]

    return multi
end

clusters = cluster_by_tolerance(Ics; tol=0.08);
summaries = [
    (
        mean_val = mean(getfield.(c, :val)),
        std_val  = std(getfield.(c, :val)),
        currents = getfield.(c, :val),
        datasets = getfield.(c, :set),
        indices  = getfield.(c, :idx)
    )
    for c in clusters
];
for s in summaries
    println("Value group ≈ $(@sprintf("%1.3f", s.mean_val)) ± $(round(s.std_val;sigdigits=1)) \t appears in datasets: ", s.datasets)
end

plt = plot();
color   = palette(:darkrainbow, length(data_directories));
for (i, Ic) in enumerate(Ics)
    xpos = fill(0.85+(i-1)*0.05, length(Ic))
    scatter!(plt,
        Ic, xpos;
        marker  = (:circle, 4, 0.5, color[i], stroke(1, 1, color[i])),
        label   = "$(data_directories[i])"
    )
end
for s in summaries
    vline!([s.mean_val], line=(:dash, 0.80, 1, :grey26), label= nothing)
end
plot!(plt;
    yaxis = ("Dataset",(0,2),([0,1,2],["","",""])),
    xlabel  = "Current (A)",
    xaxis   = :log10,
    xlims   = (1e-4, 1.05),
    legend  = :outerright,
    title   = "Clusters of Similar Currents Across Datasets",
    size    = (800,180),
    left_margin = 4mm,
    bottom_margin = 6mm,
    top_margin=2mm,
    foreground_color_legend = nothing,
);
display(plt)


peak_positions_dict = OrderedDict{Tuple{Int,String}, Matrix{Float64}}()
for i in eachindex(summaries)
    println("\nSummary $(@sprintf("%1.3f",summaries[i].mean_val))")

    for (j, jset) in enumerate(summaries[i].datasets)
        datadir = data_directories[jset]
        println("\t data set $(datadir)")

        data = load(joinpath(@__DIR__, datadir, "data_processed.jld2"), "data")
        F1  = data[:F1ProcessedImages]
        nx, nz, nframes, ncurr = size(F1)

        frameidx = summaries[i].indices[j]
        stack_z     = Vector{Vector{Float64}}(undef, nframes)
        stack_z_bin = Vector{Vector{Float64}}(undef, nframes)

        for k in 1:nframes
            # mean over x dimension → gives 1×nz
            prof = @views mean(F1[:,:,k,frameidx], dims=1)
            vec_prof = vec(prof)              # ensure 1D Vector
            stack_z[k] = vec_prof

            # binning into groups of nz_binning
            resh = reshape(vec_prof, nz_binning, :)
            stack_z_bin[k] = vec(mean(resh, dims=1))
        end

        peak_positions = zeros(Float64,nframes,2)
        # colors = palette(:darkrainbow, nframes)
        for l in 1:nframes
            # --- Optional half-maximum window
            z_fit = z_binned_mm
            y_fit = stack_z_bin[l]
            if half_max
                ymax    = maximum(y_fit)
                mask = y_fit .> (ymax * 0.5)
                z_fit = z_fit[mask]
                y_fit = y_fit[mask]
            end
            # --- Spline fit (cubic) on (z_fit, y_fit)
            S_fit = BSplineKit.fit(BSplineOrder(4), z_fit, y_fit, λ0; weights=compute_weights(z_fit, λ0))
            # --- Maxima via minimizing negative spline from multiple guesses
            negative_spline(x) = -S_fit(x[1])
            initial_guesses = sort([
                first(z_fit),
                quantile(z_fit, 0.40),
                z_fit[argmax(y_fit)],
                quantile(z_fit, 0.65),
                quantile(z_fit, 0.75),
                quantile(z_fit, 0.90),
                last(z_fit),
            ])

            minima_candidates = Float64[]
            for g in initial_guesses
                res = optimize(negative_spline, [minimum(z_fit)], [maximum(z_fit)], [g], Fminbox(LBFGS()))
                push!(minima_candidates, Optim.minimizer(res)[1])
            end
            sort!(minima_candidates)
            filtered = [minima_candidates[1]]
            for m in minima_candidates[2:end]
                if all(abs(m - x) > 1.0e-9 for x in filtered)
                    push!(filtered, m)
                end
            end
            vals   = -S_fit.(filtered)
            order  = sortperm(vals)
            maxima = filtered[order]

            peak_positions[l,1] = maxima[1]
            peak_positions[l,2] = S_fit(maxima[1])

            fig=plot(z_full_mm,stack_z[l],
                label="Experiment $(datadir) ($(@sprintf("%02d",l)))",
                seriestype=:scatter,
                marker=(:circle,:white,2,stroke(:gray26,0.75,1))
            )
            plot!(fig,z_binned_mm, stack_z_bin[l],
                label="Experiment (binned)",
                seriestype=:scatter,
                marker=(:diamond,:white,2, stroke(:dodgerblue4,0.6,1))
            )
            # plot!(fig,z_fit, y_fit)
            plot!(fig,z_fit, S_fit.(z_fit), 
                line=(:solid,:red,2),
                label="Spline fitting",
            )
            vline!(fig,[peak_positions[l,1]],
                line= (:dash,:black,2.5),
                label= L"$(z_{\mathrm{max}}, A_{\mathrm{max}}) = (%$(round(peak_positions[l,1]; sigdigits=5))\mathrm{mm}, %$(round(peak_positions[l,2]; sigdigits=2)))$",
                )
            plot!(
                title  = L"$I_{c} = (%$(round.(summaries[i].mean_val,digits=3)) \vert %$(round(summaries[i].currents[j], digits=3))) \mathrm{A}$",
                xlabel = L"$z \ (\mathrm{mm})$",
                ylabel = "(ON-OFF)/(FLAT-DARK)", 
                legend=:topleft,

            )
            display(fig)
        end
        peak_positions_dict[(i,datadir)] = peak_positions
        
    end
end

peak_positions_dict[(18,"20250814")]

s = 18
summaries[s]
summaries[s].currents
filtered_data = filter(kv -> kv.first[1] == s, peak_positions_dict)
filtered_label = [k[2] for k in keys(filtered_data)]
fig = plot(
    xaxis=(L"$z_{\mathrm{max}} \ (\mathrm{mm})$",:identity),
    yaxis=("Amplitude", :identity),
)
for f_dir in filtered_label
    # f_dir = filtered_label[1]
    z0_dir = data_results[f_dir].fw_centroid_mm
    println(z0_dir)
    peak_amp_dir = filtered_data[(s,f_dir)]
    println(peak_amp_dir[:,1])
    peak_amp_dir[:,1] = peak_amp_dir[:,1] .- z0_dir[1]
    println(peak_amp_dir[:,1])
plot!(fig, peak_amp_dir[:,1], peak_amp_dir[:,2],
    seriestype=:scatter,
    )
end
display(fig)


filtered

peak_positions_dict

plot(stack_z)

plot(z_fit, y_fit)
plot!(z_fit, S_fit.(z_fit))
plot!(z_binned_mm, stack_z_bin[1])



peak_positions = maxima[1]


Ic = data[:Currents]
F1data = ss["data"][:F1ProcessedImages]


ss["data"]
ss["data"][:Currents]
size(ss["data"][:F1ProcessedImages])
ss["data"][:F1ProcessedImages]

[:,:,30,20]

[:F1_data])