# Kelvin Titimbo
# California Institute of Technology
# August 2025

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

# To be generalized
data_qm   = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_3m","qm_3000000_screen_profiles_table.jld2"))["table"]
nz_fix, σ_fix, λ0_fix = (2,0.200,0.01)
chosen_qm = data_qm[(nz_fix, σ_fix, λ0_fix)]
Ic_qm     = [chosen_qm[i][:Icoil] for i in eachindex(chosen_qm)][2:end]
zm_qm     = [chosen_qm[i][:z_max_smooth_spline_mm] for i in eachindex(chosen_qm)][2:end]
# Ic_qm =  [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# zm_qm = [ 0.07000392151832582, 0.12103624215126021,
#  0.1637624935340883, 0.19920462520122545, 0.23292316199064259, 0.25916178963661196,
#  0.2878604881906509, 0.31263042984724054, 0.3374424611139296, 0.4497195784807205,
#  0.5458749029445646, 0.641772232809067, 0.7316863846147059, 0.830490922909975,
#  0.9258644851398463, 1.0140804073005922, 1.0981745467764124, 1.2656789418584087,
#  1.4240360171699535, 1.5814155127000782, 1.6859927244621518, 1.8169193762660025]
#

parent_folder = joinpath(@__DIR__, "analysis_data");
data_directories = ["20250814", "20250820", "20250825","20250919","20251002","20251003","20251006"];

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
    ylim = (1e-3,1.05),
    xlim=(-1,n_runs+2),
    yaxis = (:log10, L"$I_{0} \ (\mathrm{A})$"),
    xticks = (1:length(data_directories), data_directories),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xminorticks = false,
    xrotation=65,
    bottom_margin=-2mm,
    left_margin = 6mm,
    size=(350,720)
)
display(fig_Is)
saveplot(fig_Is, "currents_sampled")

# only load a few columns from each fw_data.csv
sel = [:Icoil_A, :Icoil_error_A, :F1_z_centroid_mm, :F1_z_centroid_se_mm]; 

for data_directory in data_directories
    # Data Directory
    # data_directory = "20250825" ;

    magnification_factor = mag_factor(data_directory) ;
        
    m = DataReading.collect_fw_map(parent_folder; 
                                    select=sel, 
                                    filename="fw_data.csv", 
                                    report_name="experiment_report.txt", 
                                    sort_on=:binning, 
                                    data_dir_filter=data_directory)
    

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

    key_labels = collect(keys(m))
    cols_k = palette(:darkrainbow, length(key_labels))
    fig=plot(title="Experimental Data : binning & spline smoothing factor",
    titlefontsize = 12)
    for (i,key) in enumerate(key_labels)
        z_real  = abs.(m[key][3][2:end,"F1_z_centroid_mm"]/magnification_factor[1])
        δz_real = z_real .* sqrt.( (m[key][3][2:end,"F1_z_centroid_se_mm"] ./ m[key][3][2:end,"F1_z_centroid_mm"]).^2 .+ (magnification_factor[2]./magnification_factor[1]).^2  ) 

        plot!(fig,m[key][3][2:end,"Icoil_A"], z_real, 
        xerror = m[key][3][2:end,"Icoil_error_A"],
        yerror = δz_real,
        label="n=$(m[key][1]) | λ=$(m[key][2])", 
        color=cols_k[i],
        marker=(:circle,cols_k[i],2),
        markerstrokewidth = 1,
        markerstrokecolor=cols_k[i]
        )
        chosen_qm_i  = data_qm[(m[key][1],σ_fix,m[key][2])]
        Ic_qm_i      = [chosen_qm_i[i][:Icoil] for i in eachindex(chosen_qm_i)][2:end]
        zm_qm_i      = [chosen_qm_i[i][:z_max_smooth_spline_mm] for i in eachindex(chosen_qm_i)][2:end]
        if m[key][1] == 1
            qm_color = :grey28
        elseif m[key][1] == 2
            qm_color = :grey42
        elseif m[key][1] ==3
            qm_color = :grey56
        else
            qm_color = :grey70
        end

        plot!(Ic_qm_i,zm_qm_i,
            label=false,
            line=(qm_color,:dash,1.5))


    end
    plot!(fig,Ic_qm,zm_qm, label=L"QM: 
    $(n_{z},\sigma,λ_{0})=(%$(nz_fix),%$(Int(1000*σ_fix))\mathrm{\mu m},%$(λ0_fix))$", line=(:solid,:black,2), marker=(:square,:grey66,2))
    plot!(fig,
        xlabel="Current (A)",
        ylabel=L"$z_{F_{1}}$ (mm)",
        xaxis=:log10,
        yaxis=:log10,
        xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xlims=(1e-3,1.2),
        ylims=(5e-4,5),
        size=(850,600),
        legend=:outerright,
        legend_columns=1,
        legendfontsize=8,
        foreground_color_legend = nothing,
        left_margin=3mm,
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
m_sets = map(d -> DataReading.collect_fw_map(
                 parent_folder;
                 select=sel,
                 filename="fw_data.csv",
                 report_name="experiment_report.txt",
                 sort_on=:binning,
                 data_dir_filter=d
             ), data_directories)

# desired values
selected_bin = 2
selected_spl = 0.01

# exact match (safe for Int; Float uses == here)
key_run = Vector{Union{Nothing,String}}(undef, length(m_sets))
for (midx,ms) in enumerate(m_sets)
    keys_match = [k for (k, nt) in ms if nt.binning == selected_bin && nt.smoothing == selected_spl]
    key_run[midx] = isempty(keys_match) ? nothing : first(keys_match)
end

# ---------- common axis + style ----------
xticks_vals = 10.0 .^ (-6:-1); xticks_vals = vcat(xticks_vals, 1.0)
yticks_vals = 10.0 .^ (-6:-1); yticks_vals = vcat(yticks_vals, 1.0)
xtick_labels = [L"10^{%$k}" for k in -6:-1]; xtick_labels = vcat(xtick_labels, L"10^{0}")
ytick_labels = [L"10^{%$k}" for k in -6:-1]; ytick_labels = vcat(ytick_labels, L"10^{0}")

# ---------- experimental series (4 runs) ----------
# pack your inputs to avoid repetition
runs = key_run
dirs = data_directories
cols = palette(:darkrainbow, n_runs)

fig1 = plot(
    xlabel = "Current (A)",
    ylabel = L"$z_{F_{1}}$ (mm)",
    xaxis  = :log10,
    yaxis  = :log10,
    xticks = (xticks_vals, xtick_labels),
    yticks = (yticks_vals, ytick_labels),
    xlims  = (8e-3, 1.2),
    ylims  = (1e-4, 5.0),
    legend = :outerright,
    legend_title = L"sim $n=%$(selected_bin)$ & $\lambda_{0}=%$(selected_spl)$",
    size   = (900, 420),
    left_margin = 4mm,
    bottom_margin = 3mm,
)
for (j, (M, r, d, c)) in enumerate(zip(m_sets, runs, dirs, cols))
    magnification_factor = mag_factor(d)
    # columns and transforms
    I_A   = M[r][3][2:end, "Icoil_A"]            # mA -> A, abs
    δI_A  = M[r][3][2:end, "Icoil_error_A"]
    z_mm  = M[r][3][2:end, "F1_z_centroid_mm"] ./ magnification_factor[1]
    δz_mm = abs.(z_mm) .* sqrt.( ( M[r][3][2:end, "F1_z_centroid_se_mm"] ./ M[r][3][2:end, "F1_z_centroid_mm"]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )
    
    # guard for log10 axes: filter out non-positive values
    I_A   = ifelse.(I_A .> 0, I_A, missing)
    z_mm  = ifelse.(z_mm .> 0, z_mm, missing)

    plot!(fig1, I_A, z_mm;
        xerror = δI_A,
        yerror = δz_mm,
        label = "Experiment $(d): n=$(M[r][1]) | λ=$(M[r][2])",
        marker = (:circle,c,3),
        markerstrokewidth = 1,
        markerstrokecolor = c,
        line = (:solid,c,1) # pure markers; change to :solid if you want lines
    )
end
# ---------- Alexander's data ----------
plot!(fig1,
    data_JSF[:exp][:, 1],
    data_JSF[:exp][:, 2],
    label = "Alexander's data",
    line = (:dash, :green, 2),
)
plot!(fig1, Ic_qm, zm_qm, label=L"QM $(n_{z},\sigma,λ_{0})=(%$(nz_fix),%$(Int(1000*σ_fix))\mathrm{\mu m},%$(λ0_fix))$", line=(:black,2))
display(fig1)
saveplot(fig1, "bin_vs_smoothing_log")   # use explicit extension; pdf/png/svg as you like



fig2 = plot(
    xlabel = "Current (A)",
    ylabel = L"$z_{F_{1}}$ (mm)",
    # xticks = (xticks_vals, xtick_labels),
    # yticks = (yticks_vals, ytick_labels),
    xlims  = (8e-3, 1.2),
    ylims  = (1e-4, 2.0),
    legend = :outerright,
    legend_title = L"sim $n=%$(selected_bin)$",
    size   = (900, 420),
    left_margin = 4mm,
    bottom_margin = 3mm,
)
for (j, (M, r, d, c)) in enumerate(zip(m_sets, runs, dirs, cols))
    magnification_factor = mag_factor(d)
    # columns and transforms
    I_A   = M[r][3][2:end, "Icoil_A"]            # mA -> A, abs
    δI_A  = M[r][3][2:end, "Icoil_error_A"]
    z_mm  = M[r][3][2:end, "F1_z_centroid_mm"] ./ magnification_factor[1]
    δz_mm = abs.(z_mm) .* sqrt.( ( M[r][3][2:end, "F1_z_centroid_se_mm"] ./ M[r][3][2:end, "F1_z_centroid_mm"]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )

    # guard for log10 axes: filter out non-positive values
    I_A   = ifelse.(I_A .> 0, I_A, missing)
    z_mm  = ifelse.(z_mm .> 0, z_mm, missing)

    plot!(fig2, I_A, z_mm;
        label = "Experiment $(d): n=$(M[r][1]) | λ=$(M[r][2])",
        marker = (:circle,c,3),
        markerstrokewidth = 1,
        markerstrokecolor = c,
        line = (:solid,c,1) # pure markers; change to :solid if you want lines
    )

    # If you later want y-error bars, uncomment and make sure the column exists:
    # yerr = sqrt(30) .* M[r][3][3:end, "F1_z_centroid_se_mm"] ./ magnification_factor
    # plot!(fig1, I_A, z_mm; yerror = yerr, label = "", color = c, lw = 0)
end
# ---------- Alexander's data ----------
plot!(fig2,
    data_JSF[:exp][:, 1],
    data_JSF[:exp][:, 2],
    label = "Alexander's data",
    line = (:dash, :green, 2),
)
plot!(fig2, Ic_qm, zm_qm, label=L"QM $(n_{z},\sigma,λ_{0})=(%$(nz_fix),%$(Int(1000*σ_fix))\mathrm{\mu m},%$(λ0_fix))$", line=(:black,2))
display(fig2)
saveplot(fig2, "bin_vs_smoothing_lin")   # use explicit extension; pdf/png/svg as you like

println("\nComparison of differente experiments finished!\n\n")


#######################################################################################################################
######################################### AVERAGING ###################################################################
#######################################################################################################################

magnification_factor_ith =  [mag_factor(d)[1] for d in data_directories]
magnification_factor_error_ith =  [mag_factor(d)[2] for d in data_directories]
"""
Monte-Carlo average of multiple (x,y) sets onto a common grid, propagating x- and y-uncertainties.

Inputs:
  xsets :: Vector{<:AbstractVector}        # e.g. [x1,x2,x3,x4]
  ysets :: Vector{<:AbstractVector}        #       [y1,y2,y3,y4]
Keywords:
  σxsets::Union{Nothing,Vector} = nothing  # [σx1, σx2, ...] (per-point abs. σ; vectors match xsets)
  σysets::Union{Nothing,Vector} = nothing  # [σy1, σy2, ...] (optional)
  xq     :: Union{Symbol,AbstractVector} = :union  # :union or provide custom grid vector
  B      :: Int = 400                      # MC replicates
  outside::Symbol = :mask                  # :mask | :linear | :flat handling outside each set’s range
  rel_x   :: Bool = false                  # if true, σx interpreted as relative (fractional) error

Returns: (xq, μ, σ) where μ is the MC mean and σ the pointwise std (≈ 1σ band).
"""
function average_on_grid_mc(xsets, ysets;
                            σxsets=nothing, σysets=nothing,
                            xq=:union, B=400, outside=:mask, rel_x=false,
                            rng = Random.default_rng())

    @assert length(xsets) == length(ysets)
    nset = length(xsets)

    # Build common grid
    xq_vec = xq === :union ? sort!(unique(vcat(xsets...))) : collect(xq)
    m = length(xq_vec)
    preds = Matrix{Float64}(undef, B, m); fill!(preds, NaN)

    # small helper: eval with chosen outside policy
    function eval_on_grid(xb, yb, xq)
        p = sortperm(xb); xb = xb[p]; yb = yb[p]
        itp = Interpolations.interpolate((xb,), yb, Gridded(Linear()))
        ext = outside === :linear ? extrapolate(itp, Line()) :
              outside === :flat   ? extrapolate(itp, Flat()) :
                                    extrapolate(itp, Throw())
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
    idx = findfirst(x -> !ismissing(x) && x > thr, v)
    return idx === nothing ? 1 : idx
end

# Grab the table once per (M, run)
tables = [M[r][3] for (M, r) in zip(m_sets, runs)]
col = Dict(
    :x  => :Icoil_A,
    :y  => :F1_z_centroid_mm,
    :sx => :Icoil_error_A,
    :sy => :F1_z_centroid_se_mm,
)

scale_mag_factor = inv.([mag_factor(d)[1] for d in data_directories])   # = 1 / magnification_factor
threshold = 0.010 # lower cut-off for experimental currents
CURRENT_ROW_START = [first_gt_idx(t, col[:x], threshold) for t in tables]

xsets  = [t[i:end, col[:x]]                      for (t,i) in zip(tables, CURRENT_ROW_START)]
ysets  = [smf .* t[i:end, col[:y]] for (t, i, smf) in zip(tables, CURRENT_ROW_START, scale_mag_factor)]
σxsets = [t[i:end, col[:sx]]                     for (t,i) in zip(tables, CURRENT_ROW_START)]
σysets = [abs.(smf .* t[i:end, col[:y]]) .* sqrt.(
              (t[i:end, col[:sy]] ./ t[i:end, col[:y]]).^2 .+
              (magfac_err .* smf).^2
          )
          for (t, i, smf,magfac_err) in zip(tables, CURRENT_ROW_START, scale_mag_factor, magnification_factor_error_ith)]
i_sampled_length = 401

# pick a log-spaced grid across the overall x-range (nice for decades-wide currents)
xlo = minimum(first.(xsets))
xhi = maximum(last.(xsets))
xq  = exp10.(range(log10(xlo), log10(xhi), length=i_sampled_length))

xq, μ, σ = average_on_grid_mc(xsets, ysets; σxsets=σxsets, σysets=σysets,
                              xq=:union, B=500, outside=:mask, rel_x=false)

# xq, μ, σ_xy = average_on_grid_mc(xsets, ysets; σxsets=σxsets, σysets=σysets)
# _,  _, σ_y  = average_on_grid_mc(xsets, ysets; σxsets=nothing,   σysets=σysets)
# _,  _, σ_x  = average_on_grid_mc(xsets, ysets; σxsets=σxsets,    σysets=nothing)
# # If x and y errors are independent, typically:
# σ_quad = sqrt.(σ_x.^2 .+ σ_y.^2)  # should be close to σ_xy
# hcat(σ_xy, σ_quad )

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
    xlims = (10e-3,1.0),
    ylims = (8e-3, 2),
    legend=:bottomright,
)
for i=1:length(data_directories)
    xs = m_sets[i][runs[i]][3][CURRENT_ROW_START[i]:end,"Icoil_A"]
    ys = m_sets[i][runs[i]][3][CURRENT_ROW_START[i]:end,"F1_z_centroid_mm"]/magnification_factor_ith[i]
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
end
plot!(fig, Ic_qm, zm_qm, label="QM", line=(:red,:dash,2))
plot!(fig, xq, μ; 
    ribbon=σ, 
    xscale=:log10,
    yscale=:log10, 
    title = "Interpolation MC",
    label=false,
    color=:black,
)
display(fig)
saveplot(fig, "MC_interpolation")


# using Dierckx
# spl = Spline1D(m_sets[1][runs[1]][3][!,"Icoil_A"], m_sets[1][runs[1]][3][!,"F1_z_centroid_mm"]; k=3, s=0.5, bc="extrapolate")   # k=3 cubic; s=0 exact interpolate, s>0 smoothing

using BSplineKit
i_xx = range(0.005,1.000,length=i_sampled_length)

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
    xlims = (10e-3,1.0),
    ylims = (8e-3, 2),
)
z_final = zeros(length(data_directories),i_sampled_length)
cols = palette(:darkrainbow, length(data_directories))
for i=1:length(data_directories)
    xs = m_sets[i][runs[i]][3][CURRENT_ROW_START[i]:end,"Icoil_A"]
    ys = m_sets[i][runs[i]][3][CURRENT_ROW_START[i]:end,"F1_z_centroid_mm"]*scale_mag_factor[i]
    spl = BSplineKit.extrapolate(BSplineKit.interpolate(xs,ys, BSplineKit.BSplineOrder(4),BSplineKit.Natural()),BSplineKit.Linear())
    z_final[i,:] = spl.(i_xx)
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx,spl.(i_xx),
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
δzf1 = vec(std(z_final; dims=1, corrected=true)/sqrt(length(data_directories)))
plot!(fig, i_xx, zf1,
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
z_final_fit = zeros(length(data_directories),i_sampled_length)
cols = palette(:darkrainbow, length(data_directories))
for i=1:length(data_directories)
    xs = m_sets[i][runs[i]][3][CURRENT_ROW_START[i]:end,"Icoil_A"]
    ys = m_sets[i][runs[i]][3][CURRENT_ROW_START[i]:end,"F1_z_centroid_mm"]*scale_mag_factor[i]
    # δys = m_sets[i][runs[i]][3][CURRENT_ROW_START[i]:end,"F1_z_centroid_se_mm"]*scale_mag_factor
    δys = abs.(ys) .* sqrt.( (m_sets[i][runs[i]][3][CURRENT_ROW_START[i]:end,"F1_z_centroid_se_mm"] ./ m_sets[i][runs[i]][3][CURRENT_ROW_START[i]:end,"F1_z_centroid_mm"]).^2 .+ (magnification_factor_error_ith[i]*scale_mag_factor[i]).^2 )
    spl = BSplineKit.extrapolate(BSplineKit.fit(BSplineKit.BSplineOrder(4),xs,ys, 0.002, BSplineKit.Natural(); weights=1 ./ δys.^2),BSplineKit.Smooth())
    z_final_fit[i,:] = spl.(i_xx)
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx,spl.(i_xx),
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
δzf1_fit = vec(std(z_final_fit; dims=1, corrected=true)/sqrt(length(data_directories)))
plot!(fig, i_xx, zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:gray36, 
    label="Mean",
    line=(:dash,:black,:2))
display(fig)
saveplot(fig, "smoothing_interpolation")

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
)
plot!(fig,i_xx, zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:green, 
    label="Average: smoothing cubic spline",
    line=(:dot,:green,:2))
plot!(fig, i_xx, zf1,
    ribbon = δzf1,
    fillalpha=0.40, 
    fillcolor=:dodgerblue, 
    label="Average: interpolation cubic spline",
    line=(:dash,:dodgerblue,:2))
plot!(fig, xq, μ; 
    ribbon=σ, 
    label="Interpolation MC",
    color=:orangered2
)
plot!(fig, Ic_qm, zm_qm, label=L"QM $(n_{z},\sigma,λ_{0})=(%$(nz_fix),%$(Int(1000*σ_fix))\mathrm{\mu m},%$(λ0_fix))$", line=(:red,:dash,2))
plot!(fig,
xaxis=:log10, 
yaxis=:log10,
xticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlims = (10e-3,1.0),
ylims = (8e-3, 2),
legend=:bottomright,
)
display(fig)
saveplot(fig, "inter_vs_mc_vs_fit")


include("./Modules/TheoreticalSimulation.jl");
fig = plot(
    xlabel="Magnetic field gradient  (T/m)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
)
plot!(fig,TheoreticalSimulation.GvsI(i_xx), zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:green, 
    label="Average: smoothing cubic spline",
    line=(:dot,:green,:2))
plot!(fig, TheoreticalSimulation.GvsI(i_xx), zf1,
    ribbon = δzf1,
    fillalpha=0.40, 
    fillcolor=:dodgerblue, 
    label="Average: interpolation cubic spline",
    line=(:dash,:dodgerblue,:2))
plot!(fig, TheoreticalSimulation.GvsI(xq), μ; 
    ribbon=σ, 
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
        :nz_bin     => selected_bin,
        :λ0_spl     => selected_spl,
        # Interpolated data => Mean
        :i_interp   => i_xx,
        :z_interp   => zf1,
        :δz_interp  => δzf1,
        # Fitting data => Mean
        :i_smooth   => i_xx,
        :z_smooth   => zf1_fit,
        :δz_smooth  => δzf1_fit,
        # MonteCarlo sampling => Mean
        :i_mc       => xq,
        :z_mc       => μ,
        :δz_mc      => σ
    )
)

T_END = Dates.now()
T_RUN = Dates.canonicalize(T_END-T_START)
println("\nEXPERIMENTS ANALYSIS FINISHED! $(T_RUN)")
alert("EXPERIMENTS ANALYSIS FINISHED!")


using Optim

zQM_itpl = BSplineKit.extrapolate(BSplineKit.interpolate(Ic_qm, zm_qm, BSplineKit.BSplineOrder(4),BSplineKit.Natural()),BSplineKit.Linear())
# index cutoff
idx = 8



# -------------------------------------------------------------
# 1. Scaling model:   z_scaled = X/s + r
# -------------------------------------------------------------
scale_model(X, r, s) = @. X/s + r 

# -------------------------------------------------------------
# 2. Log-error function with positivity constraints
# -------------------------------------------------------------
function log_error(X::Vector, Y::Vector)
    function f(x)
        r, s = x
        s <= 0 && return Inf

        vals = scale_model(X, r, s)
        any(vals .<= 0) && return Inf  # log safety

        diff = log10.(Y) .- log10.(vals)
        return sum(diff .^ 2)
    end
    return f
end

# -------------------------------------------------------------
# 3. Fit (r, s) using Nelder–Mead
# -------------------------------------------------------------
function fit_rs(X::Vector, Y::Vector; x0=[0.0, 1.0])
    f = log_error(X, Y)
    res = optimize(f, x0, NelderMead())
    return Optim.minimizer(res) 
end

# -------------------------------------------------------------
# Plot (QM vs Experiment)
# -------------------------------------------------------------
function plotting_qm_fixed(X::Vector,Y::Vector; idx::Integer = 1, title::String = "title" , yscale::Symbol = :identity)
    z0_fit, m_fit = fit_rs(X, Y; x0=[0.0, 1.0])

    X_scaled = scale_model(X, z0_fit, m_fit)

    fig1 = plot(Ic_qm, zm_qm,
        label="Model: QM",
        line=(:dash,:blue,2));
    plot!(i_xx[idx:end], X_scaled,
        label="Experiment: Scaled ($(@sprintf("%2.2f",m_fit*mean(magnification_factor_ith))), $(@sprintf("%2.2f",1000*z0_fit))μm)",
        line=(:solid,0.75,2,:red)
    );
    plot!(
        title=title,
        xaxis = (L"$I_{c} \ (\mathrm{A})$",
                (10e-3,1),
                ([1e-3, 1e-2, 1e-1, 1.0], 
                    [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
                :log10,),
        yaxis=(L"$z_{\mathrm{max}} \ (\mathrm{mm})$",yscale),
        legend=:bottomright,
    );
    display(fig1)

    fig2 = plot(i_xx[idx:end], 100 .*( Y ./ X  .- 1),
        label="Experiment : original",
        line=(:solid,:red,2)
    );
    plot!(i_xx[idx:end], 100*(Y ./ X_scaled .- 1),
        label  = "Experiment : scaled",
        line=(:solid,:dodgerblue4,2),
        ylabel = "Relative Error (%)",
        xaxis = (L"$I_{c} \ (\mathrm{A})$",
                (10e-3,1),
                ([1e-3, 1e-2, 1e-1, 1.0], 
                    [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
                :log10,),
    );
    hline!([0], line=(:dash,:black,1), label=nothing)

    fig = plot(fig1,fig2,
        layout=(2,1),
        size=(800,500),
        left_margin=3mm,)
    display(fig)

    return (m_fit = m_fit, z0_fit = z0_fit, fig=fig)
end

plotting_qm_fixed(zf1_fit[idx:end],zQM_itpl.(i_xx[idx:end]); idx=idx, title="Spline fitting", yscale=:log10)
plotting_qm_fixed(zf1_fit[idx:end],zQM_itpl.(i_xx[idx:end]); idx=idx, title="Spline fitting", yscale=:identity)

plotting_qm_fixed(zf1[idx:end],zQM_itpl.(i_xx[idx:end]); idx=idx, title="Spline interpolation", yscale=:log10)
plotting_qm_fixed(zf1[idx:end],zQM_itpl.(i_xx[idx:end]); idx=idx, title="Spline interpolation", yscale=:identity)

ss = load(joinpath(@__DIR__,"20250820","data_processed.jld2"))

ss["data"]
ss["data"][:Currents]
size(ss["data"][:F1ProcessedImages])
ss["data"][:F1ProcessedImages]

[:,:,30,20]

[:F1_data]