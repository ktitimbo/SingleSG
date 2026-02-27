# Kelvin Titimbo
# California Institute of Technology
# February 2026

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
using LinearAlgebra, Random
using Statistics, StatsBase, OrderedCollections, Interpolations
# Aesthetics and output formatting
using Colors, ColorSchemes
using Printf, LaTeXStrings, PrettyTables
using CSV, DataFrames, DelimitedFiles, JLD2
# Time-stamping/logging
using Dates
using Alert
const T_START = Dates.now()
# Custom modules
include("./Modules/MyExperimentalAnalysis.jl");
using .MyExperimentalAnalysis;
include("./Modules/DataReading.jl");
include("./Modules/JLD2_MyTools.jl");
# Set the working directory to the current location
cd(@__DIR__) 
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSS");
const OUTDIR    = joinpath(@__DIR__, "analysis_data", "smoothing_binning")
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
# For Plots
FIG_EXT = "png"   # could be "pdf", "svg", etc.
SAVE_FIG = false
MyExperimentalAnalysis.SAVE_FIG = SAVE_FIG;
MyExperimentalAnalysis.FIG_EXT  = FIG_EXT;
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

# Previous experiment data for comparison
data_JSF = OrderedDict(
    :exp => hcat(
    [0.0200, 0.0300, 0.0500, 0.1500, 0.2000, 0.2500, 0.3500, 0.5000, 0.7500], #mA
    [0.0229, 0.0610, 0.1107, 0.3901, 0.5122, 0.6315, 0.8139, 1.1201, 1.5738]),
    :model => hcat(
    [0.0150, 0.0200, 0.0250, 0.0300, 0.0400, 0.0500, 0.0700, 0.1000, 0.1500, 0.2000, 0.2500, 0.3500, 0.5000, 0.7500], #mA
    [0.0409, 0.0566, 0.0830, 0.1015, 0.1478, 0.1758, 0.2409, 0.3203, 0.4388, 0.5433, 0.6423, 0.8394, 1.1267, 1.5288], #CQD
    [0.0179, 0.0233, 0.0409, 0.0536, 0.0883, 0.1095, 0.1713, 0.2487, 0.3697, 0.4765, 0.5786, 0.7757, 1.0655, 1.4630]) #QM
);

nz_fix, σ_fix, λ0_fix = (2,0.200,0.01);
data_qm_path = joinpath(@__DIR__,"simulation_data","qm_simulation_8M","qm_screen_profiles_f1_table.jld2");
chosen_qm = jldopen(data_qm_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz_fix,σ_fix,λ0_fix)]
end
Ic_qm     = [chosen_qm[i][:Icoil] for i in eachindex(chosen_qm)][2:end];
zm_qm     = [chosen_qm[i][:z_max_smooth_spline_mm] for i in eachindex(chosen_qm)][2:end];

parent_folder = joinpath(@__DIR__, "analysis_data");
data_directories = [
    "20250814", "20250820", "20250825","20250919","20251002","20251003","20251006",
    # "20260211", "20260213", 
    # "20260220", "20260225", "20260226am"
];

n_runs = length(data_directories);
I_all  = Vector{Vector{Float64}}(undef, n_runs);
dI_all = Vector{Vector{Float64}}(undef, n_runs);
cols = palette(:darkrainbow, n_runs);

for (i, dir) in enumerate(data_directories)
    d   = load(joinpath(@__DIR__, dir, "data_processed.jld2"), "data");
    I_all[i]  = Vector{Float64}(d[:Currents]);
    dI_all[i] = Vector{Float64}(d[:CurrentsError]);
end

fig_Is = plot(
        title = "Coil Currents",
        legend = :bottomright,
        xgrid=false,
        gridalpha = 0.25,
        gridstyle = :dot,
        minorgridalpha = 0.05,
        tickfontsize=11,
        guidefontsize=14,
    );
for (idx,data_directory) in enumerate(data_directories)
    scatter!(fig_Is,
        idx .* ones(length(I_all[idx])), 
        I_all[idx],
        yerror=dI_all[idx],
        label=false,
        marker = (:circle, :white, 2.5),
        markerstrokecolor = cols[idx],
        markerstrokewidth = 1.5,)
end
plot!(fig_Is,
    ylim = (1e-5,1.05),
    xlim=(-1,n_runs+2),
    yaxis = (:log10, L"$I_{0} \ (\mathrm{A})$"),
    xticks = (1:n_runs, data_directories),
    yticks = ([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xminorticks = false,
    xrotation=75,
    bottom_margin=-2mm,
    left_margin = 6mm,
    size=(350,720)
)
display(fig_Is)
saveplot(fig_Is, "currents_sampled")

sel = [:Icoil_A, :Icoil_error_A, :F1_z_centroid_mm, :F1_z_centroid_se_mm]; 
for data_directory in data_directories
    # Data Directory
    # data_directory = "20250814" ;

    magnification_factor = mag_factor(data_directory) ;
        
    m = DataReading.collect_fw_map(parent_folder; 
                                    select=sel, 
                                    filename="fw_data.csv", 
                                    report_name="experiment_report.txt", 
                                    sort_on=:binning, 
                                    data_dir_filter=data_directory);
    
    pretty_table(hcat(collect(keys(m)),
                        [v.binning   for v in values(m)],
                        [v.smoothing for v in values(m)]); 
                title = "Analysis for $(data_directory)",
                column_labels=["Run Label","Binning","Smoothing"],
                alignment=:c,
                style = TextTableStyle(
                        first_line_column_label = crayon"yellow bold",
                        table_border  = crayon"blue bold",
                        # column_label  = crayon"yellow bold",
                ),
                # border_crayon = crayon"blue bold",
                table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
                # header_crayon = crayon"yellow bold",
                equal_data_column_widths= true,
    )

    summary_path = joinpath(@__DIR__,"analysis_data","summary",data_directory, data_directory*"_report_summary.jld2")
    Icoils = jldopen(summary_path,"r") do mfile
            abs.(mfile["meta/Currents"])
    end

    nz_list = [1,2]
    λ0_list = [0.001, 0.005, 0.01, 0.02]
    param_grid = vec(collect(Iterators.product(nz_list, λ0_list)))
    sort!(param_grid, by = x -> (x[1], x[2]))
    N_labels = length(param_grid);
    cols_k = palette(:darkrainbow, N_labels)
    
    fig=plot(title="Experimental Data : binning & spline smoothing factor",
        titlefontsize = 12)
    i = 1
    for (nz,λ0) in param_grid
        # Check experimental data
        data_exp = jldopen(summary_path,"r") do mfile
                mfile[JLD2_MyTools.make_keypath_exp(data_directory,nz,λ0)]
        end
        
        ic = Icoils
        δic = data_exp[:ErrorCurrentsPhys]
        zf1 = data_exp[:fw_F1_peak_pos][1] / magnification_factor[1]
        δzf1 = abs.(zf1) .* sqrt.( (data_exp[:fw_F1_peak_pos][2] ./ data_exp[:fw_F1_peak_pos][1]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )

        plot!(fig,
        ic, zf1, 
        xerror = δic,
        yerror = δzf1,
        label="n=$(nz) | λ=$(λ0)", 
        color=cols_k[i],
        marker=(:circle,cols_k[i],2),
        markerstrokewidth = 1,
        markerstrokecolor=cols_k[i]
        )

        chosen_qm_i  = jldopen(data_qm_path,"r") do file
                            file[JLD2_MyTools.make_keypath_qm(nz,σ_fix,λ0)]
        end       
        Ic_qm_i      = [chosen_qm_i[i][:Icoil] for i in eachindex(chosen_qm_i)]
        zm_qm_i      = [chosen_qm_i[i][:z_max_smooth_spline_mm] for i in eachindex(chosen_qm_i)]
        if nz == 1
            qm_color = :grey28
        elseif nz == 2
            qm_color = :grey42
        elseif nz ==4
            qm_color = :grey56
        else
            qm_color = :grey70
        end
        plot!(Ic_qm_i,zm_qm_i,
            label=false,
            line=(qm_color,:dash,1.5)
        )
        i+=1
    end
    display(fig)
    plot!(fig,Ic_qm,zm_qm, label=L"QM: 
        $(n_{z},\sigma,λ_{0})=(%$(nz_fix),%$(Int(1000*σ_fix))\mathrm{\mu m},%$(λ0_fix))$", line=(:solid,:black,2), marker=(:square,:grey66,2))
    plot!(fig,
        xlabel="Current (A)",
        ylabel=L"$z_{F_{1}}$ (mm)",
        xaxis=:log10,
        yaxis=:log10,
        xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xlims=(1e-4,1.2),
        ylims=(1e-4,5),
        size=(1050,600),
        legend=:outerright,
        legend_columns=1,
        legendfontsize=8,
        foreground_color_legend = nothing,
        left_margin=5mm,
        bottom_margin=3mm,
        legend_title = data_directory,
    )
    saveplot(fig,"bin_vs_smoothing_$(data_directory)")
    display(fig)
    println("\n")
end

println("Experiment analysis finished!\n\n")

#########################################################################################
# Choose a particular configuration for comparison purposes 
#########################################################################################

# desired values
selected_bin = nz_fix
selected_spl = λ0_fix

cols = palette(:darkrainbow, n_runs)

# ---------- common axis + style ----------
xticks_vals = 10.0 .^ (-6:-1); xticks_vals = vcat(xticks_vals, 1.0)
yticks_vals = 10.0 .^ (-6:-1); yticks_vals = vcat(yticks_vals, 1.0)
xtick_labels = [L"10^{%$k}" for k in -6:-1]; xtick_labels = vcat(xtick_labels, L"10^{0}")
ytick_labels = [L"10^{%$k}" for k in -6:-1]; ytick_labels = vcat(ytick_labels, L"10^{0}")

fig1 = plot(
    xlabel = "Current (A)",
    ylabel = L"$z_{F_{1}}$ (mm)",
    xaxis  = :log10,
    yaxis  = :log10,
    xticks = (xticks_vals, xtick_labels),
    yticks = (yticks_vals, ytick_labels),
    xlims  = (1e-3, 1.2),
    ylims  = (1e-4, 3.0),
    legend = :outerright,
    legend_title = L"$n=%$(selected_bin)$ & $\lambda_{0}=%$(selected_spl)$",
    size   = (900, 420),
    left_margin = 4mm,
    bottom_margin = 3mm,
)
for (idx,data_directory) in enumerate(data_directories)
    magnification_factor = mag_factor(data_directory) ;

    summary_path = joinpath(@__DIR__,"analysis_data","summary",data_directory, data_directory*"_report_summary.jld2")

    Icoils = jldopen(summary_path,"r") do mfile
            mfile["meta/Currents"]
    end

    # Check experimental data
    data_exp = jldopen(summary_path,"r") do mfile
            mfile[JLD2_MyTools.make_keypath_exp(data_directory,selected_bin,selected_spl)]
    end

    ic = Icoils
    δic = data_exp[:ErrorCurrentsPhys]
    zf1 = data_exp[:fw_F1_peak_pos][1] / magnification_factor[1]
    δzf1 = zf1 .* sqrt.( (data_exp[:fw_F1_peak_pos][2] ./ data_exp[:fw_F1_peak_pos][1]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )

    # guard for log10 axes: filter out non-positive values
    ic   = ifelse.(ic .> 0, ic, missing)
    zf1  = ifelse.(zf1 .> 0, zf1, missing)
    plot!(fig1, ic, zf1;
        xerror = δic,
        yerror = δzf1,
        label = "Experiment $(data_directory)",
        marker = (:circle,cols[idx],3),
        markerstrokewidth = 1,
        markerstrokecolor = cols[idx],
        line = (:solid,cols[idx],1) # pure markers; change to :solid if you want lines
    )
    display(fig1)
end
plot!(fig1, # ---------- Alexander's data ----------
    data_JSF[:exp][:, 1],
    data_JSF[:exp][:, 2],
    label = "Alexander's data",
    line = (:dash, :green, 2),
)
plot!(fig1, Ic_qm, zm_qm, label=L"QM $(\sigma_{w}=%$(Int(1000*σ_fix))\mathrm{\mu m})$", line=(:black,2))
display(fig1)
saveplot(fig1, "bin_vs_smoothing_log")   # use explicit extension; pdf/png/svg as you like


fig2 = plot(
    xlabel = "Current (A)",
    ylabel = L"$z_{F_{1}}$ (mm)",
    xlims  = (1e-3, 1.1),
    ylims  = (1e-4, 2.0),
    legend = :outerright,
    legend_title = L"$n=%$(selected_bin)$ & $\lambda_{0}=%$(selected_spl)$",
    size   = (900, 420),
    left_margin = 4mm,
    bottom_margin = 3mm,
)
for (idx,data_directory) in enumerate(data_directories)
    magnification_factor = mag_factor(data_directory) ;

    summary_path = joinpath(@__DIR__,"analysis_data","summary",data_directory, data_directory*"_report_summary.jld2")

    Icoils = jldopen(summary_path,"r") do mfile
            mfile["meta/Currents"]
    end

    # Check experimental data
    data_exp = jldopen(summary_path,"r") do mfile
            mfile[JLD2_MyTools.make_keypath_exp(data_directory,selected_bin,selected_spl)]
    end

    ic = Icoils
    δic = data_exp[:ErrorCurrentsPhys]
    zf1 = data_exp[:fw_F1_peak_pos][1] / magnification_factor[1]
    δzf1 = abs.(zf1) .* sqrt.( (data_exp[:fw_F1_peak_pos][2] ./ data_exp[:fw_F1_peak_pos][1]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )

    # guard for log10 axes: filter out non-positive values
    ic   = ifelse.(ic .> 0, ic, missing)
    zf1  = ifelse.(zf1 .> 0, zf1, missing)
    plot!(fig2, ic, zf1;
        xerror = δic,
        yerror = δzf1,
        label = "Experiment $(data_directory)",
        marker = (:circle,cols[idx],3),
        markerstrokewidth = 1,
        markerstrokecolor = cols[idx],
        line = (:solid,cols[idx],1) # pure markers; change to :solid if you want lines
    )
    display(fig2)
end
plot!(fig2, # ---------- Alexander's data ----------
    data_JSF[:exp][:, 1],
    data_JSF[:exp][:, 2],
    label = "Alexander's data",
    line = (:dash, :green, 2),
)
plot!(fig2, Ic_qm, zm_qm, label=L"QM $(\sigma_{w}=%$(Int(1000*σ_fix))\mathrm{\mu m})$", line=(:black,2))
display(fig2)
saveplot(fig2, "bin_vs_smoothing_lin")   # use explicit extension; pdf/png/svg as you like

println("\nComparison of differente experiments finished!\n\n")

#######################################################################################################################
######################################### AVERAGING ###################################################################
#######################################################################################################################
Ics = Vector{Vector{Float64}}(undef, n_runs);
tol_grouping = 0.05
for (i, dir) in enumerate(data_directories)
    data = load(joinpath(@__DIR__, dir, "data_processed.jld2"), "data")
    Ics[i] = data[:Currents]
end
clusters = MyExperimentalAnalysis.cluster_by_tolerance(Ics; tol=tol_grouping);
for s in clusters.summary
    println("Value group ≈ $(@sprintf("%1.3f", s.mean_val)) ± $(round(s.std_val;sigdigits=1)) \t appears in datasets: ", s.datasets)
end
Ic_grouped  = round.([clusters.summary[i].mean_val for i in 1:length(clusters.summary)]; digits=3)
δIc_grouped = round.([clusters.summary[i].std_val for i in 1:length(clusters.summary)]; sigdigits=1)

magnification_factor_ith =  [mag_factor(d)[1] for d in data_directories]
magnification_factor_error_ith =  [mag_factor(d)[2] for d in data_directories]
"""
    average_on_grid_mc(xsets, ysets;
                       σxsets=nothing, σysets=nothing,
                       xq=:union, B=400, outside=:mask, rel_x=false,
                       rng=Random.default_rng()) -> (xq_vec, μ, σ)

Monte-Carlo average of multiple noisy curves onto a common 1D grid, propagating
uncertainties in both the x- and y-coordinates.

Each input dataset `i` is a pair `(xsets[i], ysets[i])`. For each Monte-Carlo replicate,
the function perturbs `x` and/or `y` according to the provided uncertainties, interpolates
the perturbed curve onto a shared query grid `xq_vec` using linear gridded interpolation,
and then averages across datasets at each grid point (ignoring missing values). The output
mean `μ` and standard deviation `σ` are computed pointwise across the `B` replicates.

# Arguments
- `xsets::AbstractVector{<:AbstractVector}`: Collection of x-vectors, one per dataset.
- `ysets::AbstractVector{<:AbstractVector}`: Collection of y-vectors, one per dataset.
  Must satisfy `length(xsets) == length(ysets)` and each pair must have matching lengths.

# Keyword Arguments
- `σxsets::Union{Nothing,AbstractVector}=nothing`:
  Per-dataset vectors of 1σ uncertainties for `x`. If `nothing`, `x` is not perturbed.
  Each `σxsets[i]` must match `length(xsets[i])`.
- `σysets::Union{Nothing,AbstractVector}=nothing`:
  Per-dataset vectors of 1σ uncertainties for `y`. If `nothing`, `y` is not perturbed.
  Each `σysets[i]` must match `length(ysets[i])`.
- `rel_x::Bool=false`:
  If `true`, interpret `σxsets[i]` as *relative* uncertainties so that `Δx = σx .* x`.
  If `false`, interpret `σxsets[i]` as absolute uncertainties.
- `xq::Union{Symbol,AbstractVector}=:union`:
  Query grid specification. If `:union`, uses `sort!(unique(vcat(xsets...)))`.
  Otherwise, uses `collect(xq)` as the query grid.
- `B::Integer=400`:
  Number of Monte-Carlo replicates.
- `outside::Symbol=:mask`:
  Policy for evaluating outside each dataset's x-range:
  - `:mask`  → return `NaN` outside `[minimum(x), maximum(x)]` (excluded from averages)
  - `:linear` → linear extrapolation
  - `:flat`   → constant (flat) extrapolation
- `rng::AbstractRNG=Random.default_rng()`:
  Random number generator used for the perturbations.

# Returns
- `xq_vec::Vector{Float64}`: The common query grid.
- `μ::Vector{Float64}`: Pointwise Monte-Carlo mean on `xq_vec`.
- `σ::Vector{Float64}`: Pointwise Monte-Carlo standard deviation (sample std, `corrected=true`)
  on `xq_vec`. Entries may be `NaN` where no dataset covered that grid point (under `:mask`).

# Notes
- Within each replicate, each dataset is interpolated with `Gridded(Linear())` after sorting by `x`.
- When `outside == :mask`, points with no coverage across all datasets remain `NaN` in the output.
- This routine performs an *unweighted* mean across datasets at each grid point; if you need weighting
  (e.g. by `σy`), modify the combine step accordingly.
"""
function average_on_grid_mc(xsets, ysets;
                            σxsets=nothing, σysets=nothing,
                            xq=:union, B::Int=400, outside::Symbol=:mask, rel_x::Bool=false,
                            rng = Random.default_rng())

    @assert length(xsets) == length(ysets) "xsets and ysets must have the same number of datasets"
    nset = length(xsets)
    
    @assert outside in (:mask, :linear, :flat) "outside must be :mask, :linear, or :flat"
    @assert B > 0 "B must be positive"

    # Build common grid
    xq_vec = xq === :union ? sort!(unique(vcat(map(collect, xsets)...))) : collect(xq)
    m = length(xq_vec)

    preds = Matrix{Float64}(undef, B, m)
    fill!(preds, NaN)

    # small helper: eval with chosen outside policy
    function eval_on_grid(xb, yb, xq)
        p = sortperm(xb); xb = xb[p]; yb = yb[p]
        itp = Interpolations.interpolate((xb,), yb, Gridded(Interpolations.Linear()))
        ext = outside === :linear ? Interpolations.extrapolate(itp, Line()) :
              outside === :flat   ? Interpolations.extrapolate(itp, Flat()) :
                                    Interpolations.extrapolate(itp, Throw())
        vals = similar(xq, Float64); fill!(vals, NaN)
        if outside === :mask
            mask = (xq .>= first(xb)) .& (xq .<= last(xb))
            vals[mask] = itp.(xq[mask])  # safe since on-grid
        else
            vals .= ext.(xq)
        end
        return vals
    end

    # Monte-Carlo
    for b in 1:B
        # gather each set’s curve on xq for this replicate
        curves = Vector{Vector{Float64}}(undef, nset)
        for i in 1:nset
            x = collect(xsets[i])
            y = collect(ysets[i])

            # jitter x
            if σxsets === nothing
                xb = x
            else
                σx = σxsets[i]
                dx = rel_x ? (σx .* x) : σx                      # abs σ from relative if requested
                xb = x .+ randn(rng, length(x)) .* dx
            end

            # jitter y
            if σysets === nothing
                yb = y
            else
                σy = σysets[i]
                yb = y .+ randn(rng, length(y)) .* σy
            end

            curves[i] = eval_on_grid(xb, yb, xq_vec)
        end

        # combine across sets at each xq (ignore NaNs)
        for j in 1:m
            s = 0.0; k = 0
            @inbounds for i in 1:nset
                v = curves[i][j]
                if !isnan(v); s += v; k += 1; end
            end
            preds[b, j] = k == 0 ? NaN : (s / k)
        end
    end

    # MC mean & std at each xq (ignore NaNs if some points had no coverage)
    μ  = similar(xq_vec, Float64)
    σ  = similar(xq_vec, Float64)
    for j in 1:m
        col = @view preds[:, j]
        vals = [v for v in col if !isnan(v)]
        if isempty(vals)
            μ[j] = NaN; σ[j] = NaN
        else
            μ[j] = mean(vals)
            σ[j] = std(vals; corrected=true)
        end
    end
    return xq_vec, μ, σ
end

# helper: first index where column > threshold (skips missings; falls back to 1)
@inline function first_gt_idx(df::DataFrame, col::Symbol, thr::Real)
    v = df[!, col]
    idx = findfirst(x -> !ismissing(x) && x >= thr, v)
    return idx === nothing ? 1 : idx
end

tables = Vector{DataFrame}(undef, n_runs)
for (idx,data_directory) in enumerate(data_directories)
    magnification_factor = mag_factor(data_directory) ;

    summary_path = joinpath(@__DIR__,"analysis_data","summary",data_directory, data_directory*"_report_summary.jld2")

    Icoils = jldopen(summary_path,"r") do mfile
            mfile["meta/Currents"]
    end

    # Check experimental data
    data_exp = jldopen(summary_path,"r") do mfile
            mfile[JLD2_MyTools.make_keypath_exp(data_directory,selected_bin,selected_spl)]
    end

    ic = Icoils
    δic = data_exp[:ErrorCurrentsPhys]
    zf1 = data_exp[:fw_F1_peak_pos][1] / magnification_factor[1]
    δzf1 = abs.(zf1) .* sqrt.( (data_exp[:fw_F1_peak_pos][2] ./ data_exp[:fw_F1_peak_pos][1]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )
    zf2 = data_exp[:fw_F2_peak_pos][1] / magnification_factor[1]
    δzf2 = abs.(zf2) .* sqrt.( (data_exp[:fw_F2_peak_pos][2] ./ data_exp[:fw_F2_peak_pos][1]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )

    tables[idx] = DataFrame(hcat(ic,δic,zf1,δzf1,zf2,δzf2),[:x,:sx,:y1,:sy1,:y2,:sy2])
end

threshold = 0.000 # lower cut-off for experimental currents
CURRENT_ROW_START = [first_gt_idx(t, :x, threshold) for t in tables]

xsets  = [ t[i:end, :x]  for (t,i) in zip(tables, CURRENT_ROW_START)]
y1sets = [ t[i:end, :y1] for (t,i) in zip(tables, CURRENT_ROW_START)]
y2sets = [ t[i:end, :y2] for (t,i) in zip(tables, CURRENT_ROW_START)]
σxsets = [ t[i:end, :sx] for (t,i) in zip(tables, CURRENT_ROW_START)]
σy1sets = [ t[i:end, :sy1] for (t, i) in zip(tables, CURRENT_ROW_START)]
σy2sets = [ t[i:end, :sy2] for (t, i) in zip(tables, CURRENT_ROW_START)]

# pick a log-spaced grid across the overall x-range (nice for decades-wide currents)
i_sampled_length = 20001
xlo = maximum([minimum(first.(xsets)),1e-9])
xhi = maximum([maximum(last.(xsets)),1.000])
xq  = exp10.(range(log10(xlo), log10(xhi), length=i_sampled_length))

xi1, μ1, σ1 = average_on_grid_mc(xsets, y1sets; σxsets=σxsets, σysets=σy1sets,
                              xq=:union, B=500, outside=:mask, rel_x=true)

# xq, μ, σ_xy = average_on_grid_mc(xsets, y1sets; σxsets=σxsets, σy1sets=σy1sets)
# _,  _, σ_y  = average_on_grid_mc(xsets, y1sets; σxsets=nothing,   σy1sets=σy1sets)
# _,  _, σ_x  = average_on_grid_mc(xsets, y1sets; σxsets=σxsets,    σy1sets=nothing)
## If x and y errors are independent, typically:
# σ_quad = sqrt.(σ_x.^2 .+ σ_y.^2)  # should be close to σ_xy
# hcat(σ_xy, σ_quad )

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
    xlims = (1e-3,1.0),
    ylims = (1e-3, 2),
    legend=:bottomright,
)
for i=1:n_runs
    xs = tables[i][CURRENT_ROW_START[i]:end,:x]
    ys = tables[i][CURRENT_ROW_START[i]:end,:y1]
    scatter!(fig,xs,ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
end
plot!(fig, Ic_qm, zm_qm, label="QM", line=(:red,:dash,2))
plot!(fig, xi1, μ1; 
    ribbon=σ1,
    # yerror=σ1,
    label=false,
)
plot!(fig,
    xscale=:log10,
    yscale=:log10, 
    title = "Interpolation MC",
    color=:black,
)
display(fig)
saveplot(fig, "MC_interpolation")


# using Dierckx
# spl = Spline1D(m_sets[1][runs[1]][3][!,"Icoil_A"], m_sets[1][runs[1]][3][!,"F1_z_centroid_mm"]; k=3, s=0.5, bc="extrapolate")   # k=3 cubic; s=0 exact interpolate, s>0 smoothing

using BSplineKit
# i_sampled_length = 2*i_sampled_length
# i_xx = round.(range(threshold,1.000,length=i_sampled_length); digits=5)
i_xx0 = unique(round.(sort(union(xq,Ic_grouped)); digits=9))
i_sampled_length = length(i_xx0)

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
    xlims = (10e-3,1.0),
    ylims = (8e-3, 2),
)
z_final = zeros(n_runs,i_sampled_length)
cols = palette(:darkrainbow, n_runs);
for i=1:n_runs
    xs = tables[i][CURRENT_ROW_START[i]:end,:x]
    ys = tables[i][CURRENT_ROW_START[i]:end,:y1]
    spl = BSplineKit.extrapolate(BSplineKit.interpolate(xs,ys, BSplineKit.BSplineOrder(4),BSplineKit.Natural()),BSplineKit.Linear())
    z_final[i,:] = spl.(i_xx0)
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx0,spl.(i_xx0),
        label=false,
        line=(cols[i],1))
end
plot!(fig, Ic_qm, zm_qm, label="QM", line=(:red,:dash,2))
display(fig)
plot!(fig,
title="Interpolation: cubic splines",
xaxis=:log10, 
yaxis=:log10,
xticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)
zf1 = vec(mean(z_final, dims=1))
δzf1 = vec(std(z_final; dims=1, corrected=true)/sqrt(n_runs))
plot!(fig, i_xx0, zf1,
    ribbon = δzf1,
    fillalpha=0.40, 
    fillcolor=:gray36, 
    label="Mean",
    line=(:dash,:black,:2))
display(fig)
saveplot(fig, "interpolation")

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
)
z_final_fit = zeros(n_runs,i_sampled_length)
cols = palette(:darkrainbow, n_runs)
for i=1:n_runs
    xs = tables[i][CURRENT_ROW_START[i]:end,:x]
    ys = tables[i][CURRENT_ROW_START[i]:end,:y1]
    δys = tables[i][CURRENT_ROW_START[i]:end,:sy1]
    spl = BSplineKit.extrapolate(BSplineKit.fit(BSplineKit.BSplineOrder(4),xs,ys, 0.002, BSplineKit.Natural(); weights=1 ./ δys.^2),BSplineKit.Smooth())
    z_final_fit[i,:] = spl.(i_xx0)
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx0,spl.(i_xx0),
        label=false,
        line=(cols[i],1))
end
plot!(fig, Ic_qm, zm_qm, label="QM", line=(:red,:dash,2))
display(fig)
plot!(fig,
title = "Fit smoothing cubic spline",
xaxis=:log10, 
yaxis=:log10,
xticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlims = (10e-3,1.0),
ylims = (8e-3, 2),
)
display(fig)
zf1_fit = vec(mean(z_final_fit, dims=1))
δzf1_fit = vec(std(z_final_fit; dims=1, corrected=true)/sqrt(n_runs))
plot!(fig,i_xx0, zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:gray36, 
    label="Mean",
    line=(:dash,:black,:2))
display(fig)
saveplot(fig, "smoothing_interpolation")


z2_final_fit = zeros(n_runs,i_sampled_length)
for i=1:n_runs
    xs = tables[i][CURRENT_ROW_START[i]:end,:x]
    ys = tables[i][CURRENT_ROW_START[i]:end,:y2]
    δys = tables[i][CURRENT_ROW_START[i]:end,:sy2]
    spl = BSplineKit.extrapolate(BSplineKit.fit(BSplineKit.BSplineOrder(4),xs,ys, 0.002, BSplineKit.Natural(); weights=1 ./ δys.^2),BSplineKit.Smooth())
    z2_final_fit[i,:] = spl.(i_xx0)
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx0,spl.(i_xx0),
        label=false,
        line=(cols[i],1))
end
zf2_fit = vec(mean(z2_final_fit, dims=1))
δzf2_fit = vec(std(z2_final_fit; dims=1, corrected=true)/sqrt(n_runs))
fig_c = plot(i_xx0, zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:gray36, 
    label="Mean F1",
    line=(:dash,:black,1))
plot!(fig_c,
    i_xx0, zf2_fit,
    ribbon = δzf2_fit,
    fillalpha=0.40, 
    fillcolor=:gray36, 
    label="Mean F2",
    line=(:dash,:black,1)
)
plot!(fig_c,
    xlabel="Current (A)",
    ylabel="Centered Peak position (mm)",
    # xlims=(0,0.020)
    )
saveplot(fig_c, "fit_interpol_centroid")

Ic_around_0 = filter(v -> v <= 0.010, i_xx0)
ni_0  = length(Ic_around_0)
δi, eδi, m, i0, σd0 = curr_error_physical(
                i_xx0, 0*i_xx0,
                zf1_fit, zf2_fit;
                δz1 = δzf1_fit,
                δz2 = δzf2_fit,
                use_mismatch = false,
                nfit = ni_0, order = 3,
                weight = :gaussian, h = nothing
            );
@show round(eδi,digits=6)
@info "Channel disagreement at Ic=$(i_xx0[i0])A is $(round(1000*abs.((zf1_fit[i0] - zf2_fit[i0]) / 2 );digits=3))μm"
@info "Channel error measured at Ic=$(i_xx0[i0])A is $(round(1000*abs.( 0.5 * sqrt( δzf1_fit[i0]^2 + δzf2_fit[i0]^2 ) );digits=3))μm"

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
)
z_final_fit = zeros(n_runs,i_sampled_length)
cols = palette(:darkrainbow, n_runs)
for i=1:n_runs
    xs = tables[i][CURRENT_ROW_START[i]:end,:x]
    ys = tables[i][CURRENT_ROW_START[i]:end,:y1]
    δys = tables[i][CURRENT_ROW_START[i]:end,:sy1]
    spl = BSplineKit.extrapolate(BSplineKit.fit(BSplineKit.BSplineOrder(4),xs,ys, 0.002, BSplineKit.Natural(); weights=1 ./ δys.^2),BSplineKit.Smooth())
    z_final_fit[i,:] = spl.(i_xx0)
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx0,spl.(i_xx0),
        label=false,
        line=(cols[i],1))
end
plot!(fig, Ic_qm, zm_qm, label="QM", line=(:red,:dash,2))
display(fig)
plot!(fig,
title = "Fit smoothing cubic spline",
xaxis=:log10, 
yaxis=:log10,
xticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlims = (10e-3,1.0),
ylims = (8e-3, 2),
)
plot!(fig,i_xx0[1:200:end], zf1_fit[1:200:end],
    xerror = δi[1:200:end],
    yerror = δzf1_fit[1:200:end],
    marker=(:square,1,:black),
    fillalpha=0.40, 
    fillcolor=:gray36, 
    label="Mean",
    line=(:solid,:black,2))
display(fig)
saveplot(fig, "smoothing_interpolation_err")


fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
)
plot!(fig,i_xx0, zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:green, 
    label="Average: smoothing cubic spline",
    line=(:dot,:green,:2))
plot!(fig, i_xx0, zf1,
    ribbon = δzf1,
    fillalpha=0.40, 
    fillcolor=:dodgerblue, 
    label="Average: interpolation cubic spline",
    line=(:dash,:dodgerblue,:2))
plot!(fig, xi1, μ1; 
    ribbon=σ1, 
    label="Interpolation MC",
    color=:orangered2
)
plot!(fig, Ic_qm, zm_qm, label=L"QM $(n_{z},\sigma,λ_{0})=(%$(nz_fix),%$(Int(1000*σ_fix))\mathrm{\mu m},%$(λ0_fix))$", line=(:red,:dash,2))
plot!(fig,
xaxis=:log10, 
yaxis=:log10,
xticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlims = (15e-3,1.0),
ylims = (1e-3, 2),
legend=:bottomright,
)
display(fig)
saveplot(fig, "inter_vs_mc_vs_fit")



include("./Modules/TheoreticalSimulation.jl");
fig = plot(
    xlabel="Magnetic field gradient  (T/m)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
)
plot!(fig,TheoreticalSimulation.GvsI(i_xx0), zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:green, 
    label="Average: smoothing cubic spline",
    line=(:dot,:green,:2))
plot!(fig, TheoreticalSimulation.GvsI(i_xx0), zf1,
    ribbon = δzf1,
    fillalpha=0.40, 
    fillcolor=:dodgerblue, 
    label="Average: interpolation cubic spline",
    line=(:dash,:dodgerblue,:2))
plot!(fig, TheoreticalSimulation.GvsI(xi1), μ1; 
    ribbon=σ1, 
    label="Interpolation MC",
    color=:orangered2
)
plot!(fig, TheoreticalSimulation.GvsI.(Ic_qm), zm_qm, label=L"QM $(n_{z},\sigma,λ_{0})=(%$(nz_fix),%$(Int(1000*σ_fix))\mathrm{\mu m},%$(λ0_fix))$", line=(:red,:dash,2))
plot!(fig,
xaxis=:log10, 
yaxis=:log10,
xticks = ([ 1.0, 10, 100, 1000], [L"10^{0}", L"10^{1}", L"10^{2}", L"10^{3}"]),
yticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlims = (3,400,),
ylims = (8e-3, 2),
legend=:bottomright,
)
display(fig)
saveplot(fig, "g_inter_vs_mc_vs_fit")


jldsave(joinpath(OUTDIR,"data_averaged_$(selected_bin).jld2"), 
    data=OrderedDict(
        :nz_bin         => selected_bin,
        :σw_um          => round(1000*σ_fix; sigdigits=6),
        :λ0_spl         => selected_spl,
        :dir            => data_directories,
        :mag_factor     => hcat(magnification_factor_ith ,magnification_factor_error_ith),
        :tol_grouping   => tol_grouping,
        :Ic_grouped     => hcat(Ic_grouped , δIc_grouped),
        # Interpolated data => Mean
        :i_interp       => i_xx0,
        :z_interp       => zf1,
        :δz_interp      => δzf1,
        # Fitting data => Mean
        :i_smooth       => i_xx0,
        :δi_smooth      => δi,
        :z_smooth       => zf1_fit,
        :δz_smooth      => δzf1_fit,
        :z2_smooth      => zf2_fit,
        :δz2_smooth     => δzf2_fit,
        # MonteCarlo sampling => Mean
        :i_mc           => xi1,
        :z_mc           => μ1,
        :δz_mc          => σ1
    )
)

T_END = Dates.now()
T_RUN = Dates.canonicalize(T_END-T_START)
println("\nEXPERIMENTS ANALYSIS FINISHED! $(T_RUN)")
alert("EXPERIMENTS ANALYSIS FINISHED!")


# using Optim

# zQM_itpl = BSplineKit.extrapolate(BSplineKit.interpolate(Ic_qm, zm_qm, BSplineKit.BSplineOrder(4),BSplineKit.Natural()),BSplineKit.Linear())
# # index cutoff
# idx = 8



# # -------------------------------------------------------------
# # 1. Scaling model:   z_scaled = X/s + r
# # -------------------------------------------------------------
# scale_model(X, r, s) = @. X/s + r 

# # -------------------------------------------------------------
# # 2. Log-error function with positivity constraints
# # -------------------------------------------------------------
# function log_error(X::Vector, Y::Vector)
#     function f(x)
#         r, s = x
#         s <= 0 && return Inf

#         vals = scale_model(X, r, s)
#         any(vals .<= 0) && return Inf  # log safety

#         diff = log10.(Y) .- log10.(vals)
#         return sum(diff .^ 2)
#     end
#     return f
# end

# # -------------------------------------------------------------
# # 3. Fit (r, s) using Nelder–Mead
# # -------------------------------------------------------------
# function fit_rs(X::Vector, Y::Vector; x0=[0.0, 1.0])
#     f = log_error(X, Y)
#     res = optimize(f, x0, NelderMead())
#     return Optim.minimizer(res) 
# end

# # -------------------------------------------------------------
# # Plot (QM vs Experiment)
# # -------------------------------------------------------------
# function plotting_qm_fixed(X::Vector,Y::Vector; idx::Integer = 1, title::String = "title" , yscale::Symbol = :identity)
#     z0_fit, m_fit = fit_rs(X, Y; x0=[0.0, 1.0])

#     X_scaled = scale_model(X, z0_fit, m_fit)

#     fig1 = plot(Ic_qm, zm_qm,
#         label="Model: QM",
#         line=(:dash,:blue,2));
#     plot!(i_xx0[idx:end], X_scaled,
#         label="Experiment: Scaled ($(@sprintf("%2.2f",m_fit*mean(magnification_factor_ith))), $(@sprintf("%2.2f",1000*z0_fit))μm)",
#         line=(:solid,0.75,2,:red)
#     );
#     plot!(
#         title=title,
#         xaxis = (L"$I_{c} \ (\mathrm{A})$",
#                 (10e-3,1),
#                 ([1e-3, 1e-2, 1e-1, 1.0], 
#                     [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
#                 :log10,),
#         yaxis=(L"$z_{\mathrm{max}} \ (\mathrm{mm})$",yscale),
#         legend=:bottomright,
#     );
#     display(fig1)

#     fig2 = plot(i_xx0[idx:end], 100 .*( Y ./ X  .- 1),
#         label="Experiment : original",
#         line=(:solid,:red,2)
#     );
#     plot!(i_xx0[idx:end], 100*(Y ./ X_scaled .- 1),
#         label  = "Experiment : scaled",
#         line=(:solid,:dodgerblue4,2),
#         ylabel = "Relative Error (%)",
#         xaxis = (L"$I_{c} \ (\mathrm{A})$",
#                 (10e-3,1),
#                 ([1e-3, 1e-2, 1e-1, 1.0], 
#                     [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
#                 :log10,),
#     );
#     hline!([0], line=(:dash,:black,1), label=nothing)

#     fig = plot(fig1,fig2,
#         layout=(2,1),
#         size=(800,500),
#         left_margin=3mm,)
#     display(fig)

#     return (m_fit = m_fit, z0_fit = z0_fit, fig=fig)
# end

# # plotting_qm_fixed(zf1_fit[idx:end],zQM_itpl.(i_xx0[idx:end]); idx=idx, title="Spline fitting", yscale=:log10)
# # plotting_qm_fixed(zf1_fit[idx:end],zQM_itpl.(i_xx0[idx:end]); idx=idx, title="Spline fitting", yscale=:identity)

# # plotting_qm_fixed(zf1[idx:end],zQM_itpl.(i_xx0[idx:end]); idx=idx, title="Spline interpolation", yscale=:log10)
# # plotting_qm_fixed(zf1[idx:end],zQM_itpl.(i_xx0[idx:end]); idx=idx, title="Spline interpolation", yscale=:identity)

# # ss = load(joinpath(@__DIR__,"20250820","data_processed.jld2"))

# # ss["data"]
# # ss["data"][:Currents]
# # size(ss["data"][:F1ProcessedImages])
# # ss["data"][:F1ProcessedImages]


# jldopen(joinpath(@__DIR__,"analysis_data","summary","20260225","20260225_report_summary.jld2"),"r") do file
#     println(file["meta/Currents"])
#     file[JLD2_MyTools.make_keypath_exp("20260225",2,0.10)]
# end

# f["meta"]
# f[JLD2_MyTools.make_keypath_exp("20260211",2,0.01)]