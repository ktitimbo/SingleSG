# Kelvin Titimbo, Xukun Lin, S. Suleyman Kahraman, and Lihong V. Wang
# California Institute of Technology
# July 2025

############## EXPERIMENT ANALYSIS PREAMBLE ##############
# Headless/Windows-safe GR: set before using Plots
if !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "100"  # offscreen; avoids popup windows/crashes
end

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
using MAT, JLD2
using LinearAlgebra
using ImageFiltering, FFTW
using DataStructures
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
include("./Modules/MyExperimentalAnalysis.jl");
using .MyExperimentalAnalysis;
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

# Data Directory
data_directory = "20250814/" ;

parent_folder = joinpath(@__DIR__, "analysis_data")

# --- 1) List only subfolders starting with "2025"
"""
    folders_2025(parent) -> Vector{String}

Return the names of immediate subfolders of `parent` that start with "2025".
"""
function folders_2025(parent::AbstractString)
    filter(f -> isdir(joinpath(parent, f)) && startswith(f, "2025"), readdir(parent))
end

# --- 2) Parse report.txt → (binning::Union{Int,Missing}, smoothing::Union{Float64,Missing})
"""
    extract_info(report_path) -> (binning::Union{Int,Missing}, smoothing::Union{Float64,Missing})

Reads a `report.txt` and returns:
- `binning`: the integer from the LAST "Binning : ..." occurrence on the line (handles "4 × 1", "4 x 1", "1", etc.)
- `smoothing`: the Float64 after "Smoothing parameter : ..."
"""
function extract_info(report_path)::Tuple{Union{Int,Missing},Union{Float64,Missing}}
    txt = read(report_path, String)

    # Take the LAST "Binning : ..." occurrence (captures the whole right-hand side of the line)
    bins = collect(eachmatch(r"Binning\s*:\s*([^\r\n]+)", txt))
    binning_val::Union{Int,Missing} = missing
    if !isempty(bins)
        bin_str = strip(bins[end].captures[1])          # e.g. "4 × 1" or "1"
        # Get the LAST integer on that line (handles "4 × 1", "4 x 1", "1", etc.)
        m = match(r"(\d+)\s*$", bin_str)
        if m === nothing
            # fallback: any integer anywhere on the line; take the last one
            ints = collect(eachmatch(r"\d+", bin_str))
            if !isempty(ints)
                m = ints[end]
            end
        end
        if m !== nothing
            binning_val = parse(Int, m.match)
        end
    end

    # Smoothing parameter (allow decimals/scientific)
    sm = match(r"Smoothing\s*parameter\s*:\s*([0-9.eE+-]+)", txt)
    smoothing_val::Union{Float64,Missing} =
        sm === nothing ? missing : (tryparse(Float64, sm.captures[1]) === nothing ? missing : tryparse(Float64, sm.captures[1]))

    return (binning_val, smoothing_val)
end

# --- 3) Find fw_data CSV inside a folder
"""
    find_fw_data_csv(folder; filename="fw_data.csv") -> Union{String,Nothing}

Prefers `filename` if present; otherwise searches for any CSV whose basename matches /\\bfw[_-]?data\\b/i.
Returns the full path, or `nothing` if not found.
"""
function find_fw_data_csv(folder::AbstractString; filename::AbstractString="fw_data.csv")::Union{String,Nothing}
    exact = joinpath(folder, filename)
    if isfile(exact)
        return exact
    end
    entries = readdir(folder; join=true)
    cands = filter(p -> endswith(lowercase(p), ".csv") &&
                        occursin(r"(?i)\bfw[_\-]?data\b", basename(p)),
                   entries)
    return isempty(cands) ? nothing : cands[1]
end

# --- 4) Load CSV with optional column selection/drop
"""
    load_fw_data_csv(path; select=nothing, drop=nothing, normalizenames=true) -> DataFrame

Wrapper over CSV.read that lets you load *only* certain columns.
- `select`: Vector of Symbols/Strings/Ints or a Regex to keep
- `drop`:   Vector or Regex to drop
- `normalizenames=true`: turn headers like "time (s)" into :time_s
"""
function load_fw_data_csv(path::AbstractString; select=nothing, drop=nothing, normalizenames::Bool=true)
    kwargs = (; normalizenames)
    select === nothing || (kwargs = merge(kwargs, (; select=select)))
    drop   === nothing || (kwargs = merge(kwargs, (; drop=drop)))
    return CSV.read(path, DataFrame; kwargs...)
end


# --- 5) Main collector: returns (meta=DataFrame, tables=Dict{String,DataFrame})
"""
    collect_fw(parent; select=nothing, drop=nothing, filename="fw_data.csv")
      -> (meta::DataFrame, tables::Dict{String,DataFrame})

Scans `parent/2025*` folders.
- Parses each `report.txt` to get (Binning, Smoothing).
- Finds and loads `fw_data.csv` (or a close match).
- Returns:
    * `meta`: one row per folder with Folder, Binning, Smoothing, fw_path
    * `tables`: Dict mapping folder name => DataFrame (columns possibly filtered via `select`/`drop`)
"""
"""
    collect_fw_map(parent; select=nothing, drop=nothing, filename="fw_data.csv", skip_missing=true)
      -> Dict{String, NamedTuple{(:binning,:smoothing,:df),Tuple{Union{Int,Missing},Union{Float64,Missing},DataFrame}}}

For each subfolder `parent/2025*`, returns:
  map["20250814"] => (binning=1, smoothing=0.03, df=<DataFrame from fw_data.csv>)

- `select`/`drop` are passed to CSV.read to load only needed columns.
- If `skip_missing` is true, folders missing report or fw_data are skipped.
"""
function collect_fw_map(parent::AbstractString; select=nothing, drop=nothing,
                        filename::AbstractString="fw_data.csv", skip_missing::Bool=true)
    out = OrderedDict{String,NamedTuple{(:binning,:smoothing,:df),
           Tuple{Union{Int,Missing},Union{Float64,Missing},DataFrame}}}()
    for f in folders_2025(parent)
        folder_path = joinpath(parent, f)
        report_path = joinpath(folder_path, "experiment_report.txt")
        fw_path = find_fw_data_csv(folder_path; filename=filename)

        # handle missing pieces
        if !(isfile(report_path) && fw_path !== nothing)
            if skip_missing
                @warn "Skipping (missing report or fw_data)" folder=f
                continue
            end
        end

        binning, smoothing = isfile(report_path) ? extract_info(report_path) : (missing, missing)
        df = fw_path === nothing ? DataFrame() : load_fw_data_csv(fw_path; select=select, drop=drop)
        out[f] = (binning=binning, smoothing=smoothing, df=df)
    end
    return out
end


# only load a few columns from each fw_data.csv
sel = [:I_coil_mA, :F1_z_centroid_mm, :F1_z_centroid_se_mm]  # works with normalizenames=true
m = collect_fw_map(parent_folder; select=sel)

key_labels = sort(collect(keys(m)))

plot(title="Binning 7 Smoothing")
for k in key_labels
    plot!(abs.(m[k][3][3:end,"I_coil_mA"]/1000), abs.(m[k][3][3:end,"F1_z_centroid_mm"]), label="$(m[k][1])| $(m[k][2])")
end
plot!(xlabel="Current (mA)",xaxis=:log10,yaxis=:log10)

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

# STERN–GERLACH EXPERIMENT SETUP
# Camera and pixel geometry : intrinsic properties
cam_pixelsize = 6.5e-6 ;  # Physical pixel size of camera [m]
nx_pixels , nz_pixels= (2160, 2560); # (Nx,Nz) pixels
magnification_factor = 1.2697 ;
# Experiment resolution
exp_bin_x, exp_bin_z = (4,1) ;  # Camera binning
exp_pixelsize_x, exp_pixelsize_z = (exp_bin_x, exp_bin_z).*cam_pixelsize ; # Effective pixel size after binning [m]
# Furnace 
Temperature = 273+205
# Image dimensions (adjusted for binning)
x_pixels = Int(nx_pixels / exp_bin_x);  # Number of x-pixels after binning
z_pixels = Int(nz_pixels / exp_bin_z);  # Number of z-pixels after binning
# Spatial axes shifted to center the pixels
x_position = pixel_positions(x_pixels, 1, exp_pixelsize_x)
z_position = pixel_positions(z_pixels, 1, exp_pixelsize_z)
println("""
***************************************************
CAMERA FEATURES
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm

IMAGES INFORMATION
    Magnification factor    : $magnification_factor
    Binning                 : $(exp_bin_x) × $(exp_bin_z)
    Effective pixels        : $(x_pixels) × $(z_pixels)
    Pixel size              : $(1e6*exp_pixelsize_x)μm × $(1e6*exp_pixelsize_z)μm
    xlims                   : ($(round(minimum(1e6*x_position), digits=6)) μm, $(round(maximum(1e3*x_position), digits=4)) mm)
    zlims                   : ($(round(minimum(1e6*z_position), digits=6)) μm, $(round(maximum(1e3*z_position), digits=4)) mm)
***************************************************
""")
# Setting the variables for the module
MyExperimentalAnalysis.effective_cam_pixelsize_z = exp_pixelsize_z;
MyExperimentalAnalysis.x_pixels = x_pixels;
MyExperimentalAnalysis.z_pixels = z_pixels;
      
# Binning for the analysis
λ0 = 0.05; # Fitting factor
n_bins = 4
z_mm = 1e3 .* pixel_positions(z_pixels, n_bins, exp_pixelsize_z)


# Importing data
if !isfile(outfile2) # check if the processed images exists
    if !isfile(outfile) # check if the raw data exists
        @info "Not found → building $outfile"
        data_raw = stack_data(data_directory; order=:desc, keynames=("BG","F1","F2"))
        jldsave(outfile, data=data_raw)
        data_raw = nothing
    else
        @info "Found $outfile → skipping build"
    end

    # Background and Flat with no binning
    img_dk = matread(joinpath(data_directory, "img_dk.mat"))["DKMean"];
    img_fl = matread(joinpath(data_directory, "img_fl.mat"))["FLMean"];
    # Binning to match acquired data
    img_dk = bin_x_mean(img_dk,exp_bin_x);
    img_fl = bin_x_mean(img_fl,exp_bin_x);

    p1 = heatmap(1e3*z_position, 1e3*x_position, img_dk; 
        title=L"$\langle$ Dark Frame $\rangle$",
        xlabel=L"$z\ \mathrm{(mm)}$",
        ylabel=L"$x\ \mathrm{(mm)}$");
    p2 = heatmap(1e3*z_position, 1e3*x_position, img_fl; 
        title=L"$\langle$ Flat Frame $\rangle$",
        xlabel=L"$z\ \mathrm{(mm)}$",
        ylabel=L"$x\ \mathrm{(mm)}$");
    fig = plot(p1, p2; layout=(1,2), link=:both, size=(1000,400), 
    left_margin=4mm,
    bottom_margin=3mm)
    saveplot(fig, "dark_flat")

    data_raw = load(outfile)["data"]
    data_processed = build_processed_dict(data_raw, img_dk,img_fl)
    jldsave(outfile2, data=data_processed)
    data_processed = nothing
    data_raw = nothing
else
    @info "Found $outfile2 → skipping build"
end
data_processed = load(outfile2)["data"]

Icoils = data_processed[:Currents]
nI = length(Icoils)

fig_I0 = plot(abs.(reverse(Icoils)), 
    seriestype=:scatter,
    label = "Currents sampled",
    marker = (:circle, :white, 2),
    markerstrokecolor = :orangered,
    markerstrokewidth = 1,
);
plot!(    
    yaxis = (:log10, L"$I_{0} \ (\mathrm{A})$", :log),
    ylim = (1e-6,1),
    title = "Coil Currents",
    label = "20250725",
    legend = :bottomright,
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    yticks = :log10,
    framestyle = :box,
    size=(600,400),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
);
vspan!([0, findlast(<(0), reverse(Icoils))+0.25], color=:gray, alpha=0.30,label="zero" );
display(fig_I0) 
saveplot(fig_I0, "current_range")

########################################################################
############# MEAN ANALYSIS ###########################################
########################################################################


# --- Compute F1 / F2 mean profiles ----------------------------------------
profiles_F1 = extract_profiles(data_processed, :F1ProcessedImages, nI, z_pixels; n_bin=n_bins)
profiles_F2 = extract_profiles(data_processed, :F2ProcessedImages, nI, z_pixels; n_bin=n_bins)

# --- Plot ------------------------------------------------------------------
fig1 = plot_profiles(z_mm, profiles_F1, Icoils; title="F1 processed data")
display(fig1)
# saveplot(fig1, "mean_f1_processed")
fig2 = plot_profiles(z_mm, profiles_F2, Icoils; title="F2 processed data")
display(fig2)
# saveplot(fig2, "mean_f2_processed")
fig = plot(fig1, fig2;
    layout=(2,1),
    size=(1000,600),
    share=:both,
    legend_columns=2,
    left_margin=5mm,
)
display(fig)
saveplot(fig, "mean_profiles_processed")

f1_mean_max = my_process_mean_maxima("F1", data_processed, n_bins; half_max=true, λ0=λ0)
f2_mean_max = my_process_mean_maxima("F2", data_processed, n_bins; half_max=true, λ0=λ0)

data_centroid = (f1_mean_max .+ f2_mean_max)/2
centroid_mean = mean(data_centroid, Weights(nI-1:-1:0))
centroid_std = std(data_centroid, Weights(nI-1:-1:0); corrected=false) / sqrt(nI)
fig = plot(abs.(Icoils), data_centroid,
    label=false,
    color=:purple,
    marker=(:cross,5),
    line=(:solid,1),
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$"),
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
                [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlim=(1e-3,1),
    yaxis = L"$z_{0} \ (\mathrm{mm})$",
    title="Centroid",
    legend=:topleft
)
hline!([centroid_mean], label=L"Centroid $z=%$(round(centroid_mean,digits=3))$mm")
hspan!( [centroid_mean - centroid_std,centroid_mean + centroid_std], color=:orangered, alpha=0.30, label=L"St.Err. = $\pm%$(round(centroid_std,digits=3))$mm")
saveplot(fig,"mean_centroid")


df_mean = DataFrame(
    I_coil_mA          = 1000 .* Icoils,
    F1_z_peak_mm       = f1_mean_max,
    F2_z_peak_mm       = f2_mean_max,
    Δz_mm              = f1_mean_max .- f2_mean_max,
    F1_z_centroid_mm   = f1_mean_max .- centroid_mean,
    F2_z_centroid_mm   = f2_mean_max .- centroid_mean,
)
sort!(df_mean, :I_coil_mA)
CSV.write(joinpath(OUTDIR, "mean_data.csv"), df_mean);

hl_Ic = Highlighter(
           (data, i, j) -> data[i, 1] == minimum(data[:, 1]),
           crayon"fg:white bold bg:dark_gray"
       );
hl_F1 = Highlighter(
           (data, i, j) -> data[i,5]<0,
           crayon"fg:red bold bg:dark_gray"
       );
hl_F2 = Highlighter(
           (data, i, j) -> data[i,6]>0,
           crayon"fg:green bold bg:dark_gray"
       );
pretty_table(
    df_mean;
    formatters    = (ft_printf("%8.3f",1), ft_printf("%8.5f",2:6)),
    alignment=:c,
    header        = (
        ["Current", "F1 z", "F2 z", "Δz", "Centroid F1 z","Centroid F2 z"], 
        ["[mA]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
        ),
    border_crayon = crayon"blue bold",
    tf            = tf_unicode_rounded,
    header_crayon = crayon"yellow bold",
    equal_columns_width= true,
    highlighters  = (hl_Ic,hl_F1,hl_F2),
)

fig_01 = plot(abs.(df_mean[!,:I_coil_mA]/1000), df_mean[!,:F1_z_peak_mm],
    label=L"$F_{1}$",
    line=(:solid,:red,2),
);
plot!(abs.(df_mean[!,:I_coil_mA]/1000), df_mean[!,:F2_z_peak_mm],
    label=L"$F_{2}$",
    line=(:solid,:blue,2),
);
plot!(
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
    xlims = (1e-5,1.0),
    title = "Peak position",
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    size=(800,600),
    legend=:topleft,
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
);
vspan!(
    [1e-8, abs(df_mean.I_coil_mA[findlast(<(0), df_mean.I_coil_mA)+1] / 1000)],
    color = :gray,
    alpha = 0.30,
    label = "zero"
);
plot!(abs.(df_mean[!,:I_coil_mA]/1000), df_mean[!,:F1_z_peak_mm], fillrange=df_mean[!,:F2_z_peak_mm],
    fillalpha=0.2,
    color=:purple,
    label = false,
);
hline!([centroid_mean], line=(:dot,:black,2), label="Centroid");

fig_02 = plot(abs.(df_mean[!,:I_coil_mA]/1000), df_mean[!,:F1_z_peak_mm] ,
    label=L"$F_{1}$",
    line=(:solid,:red,2),
);
plot!(abs.(df_mean[!,:I_coil_mA]/1000), 2*centroid_mean .- df_mean[!,:F2_z_peak_mm],
    label=L"Centroid Mirrored $F_{2}$",
    line=(:solid,:blue,2),
);
plot!(
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
    xlims = (1e-5,1.0),
    title = "Peak position",
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    size=(800,600),
    legend=:topleft,
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
);
vspan!(
    [1e-8, abs(df_mean.I_coil_mA[findlast(<(0), df_mean.I_coil_mA)+1] / 1000)],
    color = :gray,
    alpha = 0.30,
    label = "zero"
);
# Fill between y1 and y2
plot!(abs.(df_mean[!,:I_coil_mA]/1000), df_mean[!,:F1_z_peak_mm], fillrange=2*centroid_mean .- df_mean[!,:F2_z_peak_mm],
    fillalpha=0.2,
    color=:purple,
    label = false,
);
hline!([centroid_mean], line=(:dot,:black,2), label="Centroid");

fig_03 = plot(abs.(df_mean[!,:I_coil_mA]/1000), df_mean[!,:F1_z_centroid_mm] ,
    label=L"$F_{1}$",
    line=(:solid,:red,2),
);
plot!(abs.(df_mean[!,:I_coil_mA]/1000), df_mean[!,:F2_z_centroid_mm] ,
    label=L"$F_{2}$",
    line=(:solid,:blue,2),
);
plot!(
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
    xlims = (1e-5,1.0),
    title = "Peak position - Centered at Centroid",
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    size=(800,600),
    legend=:topleft,
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
)
vspan!(
    [1e-8, abs(df_mean.I_coil_mA[findlast(<(0), df_mean.I_coil_mA)+1] / 1000)],
    color = :gray,
    alpha = 0.30,
    label = "zero"
);
# Fill between y1 and y2
plot!(abs.(df_mean[!,:I_coil_mA]/1000), df_mean[!,:F1_z_centroid_mm], fillrange=df_mean[!,:F2_z_centroid_mm],
    fillalpha=0.2,
    color=:purple,
    label = false,
);

fig=plot(fig_01, fig_02, fig_03, 
layout=@layout([a ; b ; c]),
share=:x,
);
plot!(fig[1], xlabel="", xformatter=_->"");
plot!(fig[2], xlabel="", xformatter=_->"", title = "", top_margin = -9mm);
plot!(fig[3], title="", top_margin = -9mm);
display(fig)
saveplot(fig, "mean_peak_centroid") 

fig000=plot(
    abs.(df_mean[!,:I_coil_mA]/1000), abs.(df_mean[!,:F1_z_centroid_mm])/magnification_factor,
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-4,1.5),
    title = "F=1 Peak Position vs Current",
    label = "08142025",
    seriestype = :scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    legend = :bottomright,
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = :log10,
    framestyle = :box,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
) 
hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Pixel size" )
plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
marker=(:cross, :purple, 6),
line=(:purple, :dash, 2, 0.5),
markerstrokewidth=2,
label="10142024"
)
plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
line=(:dash, :blue, 3),
markerstrokewidth=2,
label="10142024: QM"
)
plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
line=(:dot, :red, 3),
markerstrokewidth=2,
label="10142024: CQD"
)
saveplot(fig000, "mean_000")

fig001=plot(
    abs.(df_mean[!,:I_coil_mA]/1000), abs.(df_mean[!,:F1_z_centroid_mm])/magnification_factor,
    xaxis = (L"$I_{c} \ (\mathrm{A})$"),
    yaxis = (L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$"),
    title = "F=1 Peak Position vs Current",
    label = "08142025",
    seriestype = :scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    legend = :bottomright,
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    framestyle = :box,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
) 
hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Pixel size" )
plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
marker=(:cross, :purple, 6),
line=(:purple, :dash, 2, 0.5),
markerstrokewidth=2,
label="10142024"
)
plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
line=(:dash, :blue, 3),
markerstrokewidth=2,
label="10142024: QM"
)
plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
line=(:dot, :red, 3),
markerstrokewidth=2,
label="10142024: CQD"
)
# saveplot(fig001, "mean_001")

fig0 = plot(fig000, fig001,
    layout=(1,2),
    size=(1000,400),
    left_margin=8mm,
    bottom_margin=5mm,
)
saveplot(fig0, "mean_00")


# Compute absolute values for plotting
y = df_mean[!,:F1_z_centroid_mm];
y_abs = abs.(y);
# Create masks for negative and non-negative values
neg_mask = y .< 0;
pos_mask = .!neg_mask;
fig100=plot(
    abs.(df_mean[pos_mask,:I_coil_mA]/1000), y_abs[pos_mask]/magnification_factor,
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-4,1.5),
    title = "F=1 Peak Position vs Current",
    label = "08142025",
    seriestype = :scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    legend = :bottomright,
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = :log10,
    framestyle = :box,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
) ;
plot!(abs.(df_mean[neg_mask,:I_coil_mA]/1000), y_abs[neg_mask]/magnification_factor, 
    label=false, 
    seriestype=:scatter,
    marker = (:xcross, :orangered2, 4),
    markerstrokecolor = :orangered2,
    markerstrokewidth = 2,
);
hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Pixel size" )
plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
marker=(:cross, :purple, 6),
line=(:purple, :dash, 2, 0.5),
markerstrokewidth=2,
label="10142024"
);
plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
line=(:dash, :blue, 3),
markerstrokewidth=2,
label="10142024: QM"
);
plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
line=(:dot, :red, 3),
markerstrokewidth=2,
label="10142024: CQD"
);
# saveplot(fig100, "mean_100")

fig101=plot(
    abs.(df_mean[pos_mask,:I_coil_mA]/1000), y_abs[pos_mask]/magnification_factor,
    xaxis = (L"$I_{c} \ (\mathrm{A})$"),
    yaxis = (L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$"),
    title = "F=1 Peak Position vs Current",
    label = "08142025",
    seriestype = :scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    legend = :bottomright,
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    framestyle = :box,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
) ;
plot!(abs.(df_mean[neg_mask,:I_coil_mA]/1000), y_abs[neg_mask]/magnification_factor, 
    label=false, 
    seriestype=:scatter,
    marker = (:xcross, :orangered2, 4),
    markerstrokecolor = :orangered2,
    markerstrokewidth = 2,
);
hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Pixel size" )
plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
marker=(:cross, :purple, 6),
line=(:purple, :dash, 2, 0.5),
markerstrokewidth=2,
label="10142024"
);
plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
line=(:dash, :blue, 3),
markerstrokewidth=2,
label="10142024: QM"
);
plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
line=(:dot, :red, 3),
markerstrokewidth=2,
label="10142024: CQD"
);
# saveplot(fig101, "mean_101")

fig1 = plot(fig100, fig101,
    layout=(1,2),
    size=(1000,400),
    left_margin=8mm,
    bottom_margin=5mm,
)
saveplot(fig1, "mean_01")


########################################################################
############# FRAMEWISE ANALYSIS #######################################
########################################################################

f1_max = my_process_framewise_maxima("F1", data_processed, n_bins; half_max=true,λ0=λ0)
f2_max = my_process_framewise_maxima("F2", data_processed, n_bins; half_max=true,λ0=λ0)

f1_z_mm , f1_zstd_mm  = (vec(mean(f1_max, dims=1)) , vec(std(f1_max, dims=1)));
f2_z_mm , f2_zstd_mm  = (vec(mean(f2_max, dims=1)) , vec(std(f2_max, dims=1)));
 
data_centroid = (f1_z_mm .+ f2_z_mm)/2
centroid_fw = mean(data_centroid, Weights(nI-1:-1:0))
centroid_std_err = std(data_centroid, Weights(nI-1:-1:0); corrected=false) / sqrt(nI)
fig = plot(abs.(Icoils), data_centroid,
    label=false,
    color=:purple,
    marker=(:cross,5),
    line=(:solid,1),
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$"),
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
                [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlim=(5e-6,1),
    yaxis = L"$z_{0} \ (\mathrm{mm})$",
    title="Centroid",
    legend=:topleft,
)
hline!([centroid_fw], label=L"Centroid $z=%$(round(centroid_fw,digits=3))$mm")
hspan!([centroid_fw - centroid_std_err,centroid_fw + centroid_std_err], color=:orangered, alpha=0.30, label=L"St.Err. = $\pm%$(round(centroid_std_err,digits=3))$mm")
saveplot(fig,"fw_centroid")


res = summarize_framewise(f1_max, f2_max, Icoils, centroid_fw, centroid_std_err; rev_order=true);
df_fw = DataFrame(
    I_coil_mA           = -res.I_coil_mA,

    F1_z_peak_mm        = res.F1_z_peak_mm,
    F1_z_peak_se_mm     = res.F1_z_se_mm , 

    F2_z_peak_mm        = res.F2_z_peak_mm,
    F2_z_peak_se_mm     = res.F2_z_se_mm ,

    Δz_mm               = res.Δz_mm,
    Δz_se_mm            = res.Δz_se_mm,

    F1_z_centroid_mm    = res.F1_z_centroid_mm, 
    F1_z_centroid_se_mm = res.F1_z_centroid_se_mm,
    F2_z_centroid_mm    = res.F2_z_centroid_mm, 
    F2_z_centroid_se_mm = res.F2_z_centroid_se_mm
)
CSV.write(joinpath(OUTDIR, "fw_data.csv"), df_fw);

hl_Ic = Highlighter(
           (data, i, j) -> data[i, 1] == minimum(data[:, 1]),
           crayon"fg:white bold bg:dark_gray"
       );
hl_F1 = Highlighter(
           (data, i, j) -> data[i,8]<0,
           crayon"fg:red bold bg:dark_gray"
       );
hl_F2 = Highlighter(
           (data, i, j) -> data[i,10]>0,
           crayon"fg:green bold bg:dark_gray"
       );
pretty_table(
    df_fw;
    formatters    = (ft_printf("%8.3f",1), ft_printf("%8.5f",2:6)),
    alignment=:c,
    header        = (
        ["Current", "F1 z", "Std.Err.",  "F2 z", "Std.Err.", "Δz", "Std.Err.", "Centroid F1 z", "Std.Err.", "Centroid F2 z", "Std.Err."], 
        ["[mA]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
        ),
    border_crayon = crayon"blue bold",
    tf            = tf_unicode_rounded,
    header_crayon = crayon"yellow bold",
    equal_columns_width= true,
    highlighters  = (hl_Ic,hl_F1,hl_F2),
)



fig_01 = plot(abs.(df_fw[!,:I_coil_mA]/1000), df_fw[!,:F1_z_peak_mm ],
    ribbon= df_fw[!, :F1_z_peak_se_mm ],
    label=L"$F_{1}$",
    line=(:solid,:red,1),
    fillalpha=0.23, 
    fillcolor=:red,  
)
plot!(abs.(df_fw[!,:I_coil_mA]/1000), df_fw[!,:F1_z_peak_mm ], fillrange=df_fw[!,:F2_z_peak_mm ],
    fillalpha=0.05,
    color=:purple,
    label = false,
)
plot!(abs.(df_fw[!,:I_coil_mA]/1000), df_fw[!,:F2_z_peak_mm ],
    ribbon= df_fw[!, :F2_z_peak_se_mm ],
    label=L"$F_{2}$",
    line=(:solid,:blue,1),
    fillalpha=0.23, 
    fillcolor=:blue,  
)
plot!(
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
    xlims = (1e-5,1.0),
    title = "Peak position",
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    size=(800,600),
    legend=:topleft,
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
)
vspan!(
    [1e-8, abs(df_fw.I_coil_mA[findlast(<(0), df_fw.I_coil_mA)+1] / 1000)],
    color = :gray,
    alpha = 0.30,
    label = "zero"
)
hline!([centroid_fw], line=(:dot,:black,2), label="Centroid")

fig_02 = plot(abs.(df_fw[!,:I_coil_mA]/1000), df_fw[!,:F1_z_peak_mm] ,
    ribbon= df_fw[!,:F1_z_peak_se_mm],
    label=L"$F_{1}$",
    line=(:solid,:red,2),
    fillalpha=0.23, 
    fillcolor=:red,  
)
plot!(abs.(df_fw[!,:I_coil_mA]/1000), 2*centroid_fw .- df_fw[!,:F2_z_peak_mm] ,
    ribbon= df_fw[!,:F2_z_peak_se_mm],
    label=L"Mirrored $F_{2}$",
    line=(:solid,:blue,2),
    fillalpha=0.23, 
    fillcolor=:blue,  
)
plot!(
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
    xlims = (1e-5,1.0),
    title = "Peak position",
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    size=(800,600),
    legend=:topleft,
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
)
vspan!(
    [1e-8, abs(df_fw.I_coil_mA[findlast(<(0), df_fw.I_coil_mA)+1] / 1000)],
    color = :gray,
    alpha = 0.30,
    label = "zero"
);
# Fill between y1 and y2
plot!(abs.(df_fw[!,:I_coil_mA]/1000), df_fw[!,:F1_z_peak_mm], fillrange=2*centroid_fw .- df_fw[!,:F2_z_peak_mm],
    fillalpha=0.2,
    color=:purple,
    label = false,
);
hline!([centroid_fw], line=(:dot,:black,2), label="Centroid")

fig_03 = plot(abs.(df_fw[!,:I_coil_mA]/1000), df_fw[!,:F1_z_centroid_mm] ,
    ribbons=df_fw[!,:F1_z_centroid_se_mm],
    label=L"$F_{1}$",
    line=(:solid,:red,2),
);
plot!(abs.(df_fw[!,:I_coil_mA]/1000), df_fw[!,:F2_z_centroid_mm] ,
    label=L"$F_{2}$",
    line=(:solid,:blue,2),
);
plot!(
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
    xlims = (1e-5,1.0),
    title = "Peak position - Centered at Centroid",
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    size=(800,600),
    legend=:topleft,
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
)
vspan!(
    [1e-8, abs(df_fw.I_coil_mA[findlast(<(0), df_fw.I_coil_mA)+1] / 1000)],
    color = :gray,
    alpha = 0.30,
    label = "zero"
);
# Fill between y1 and y2
plot!(abs.(df_fw[!,:I_coil_mA]/1000), df_fw[!,:F1_z_centroid_mm], fillrange=df_fw[!,:F2_z_centroid_mm],
    fillalpha=0.2,
    color=:purple,
    label = false,
);

fig=plot(fig_01, fig_02, fig_03, 
layout=@layout([a ; b ; c]),
share=:x,
);
plot!(fig[1], xlabel="", xformatter=_->"");
plot!(fig[2], xlabel="", xformatter=_->"", title = "", top_margin = -9mm);
plot!(fig[3], title="", top_margin = -9mm);
display(fig)
saveplot(fig, "fw_peak_centroid")


fig_log=plot(
    abs.(df_fw[!,:I_coil_mA]/1000), abs.(df_fw[!,:F1_z_centroid_mm])/magnification_factor,
    yerror = df_fw[!,:F1_z_centroid_se_mm],
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-6,3.5),
    title = "F=1 Peak Position vs Current",
    label = "08142025",
    seriestype = :scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    legend = :bottomright,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = :log10,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
    left_margin=3mm,
) 
hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Pixel size" )
plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
marker=(:cross, :purple, 6),
line=(:purple, :dash, 2, 0.5),
markerstrokewidth=2,
label="10142024"
)
plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
line=(:dash, :blue, 3),
markerstrokewidth=2,
label="10142024: QM"
)
plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
line=(:dot, :red, 3),
markerstrokewidth=2,
label="10142024: CQD"
)
display(fig_log)
saveplot(fig_log, "fw_000")

fig_lin=plot(
    abs.(df_fw[!,:I_coil_mA]/1000), abs.(df_fw[!,:F1_z_centroid_mm])/magnification_factor,
    yerror = df_fw[!,:F1_z_centroid_se_mm],
    xaxis = (L"$I_{c} \ (\mathrm{A})$"),
    yaxis = (L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$"),
    title = "F=1 Peak Position vs Current",
    label = "08142025",
    seriestype = :scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    legend = :bottomright,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
    left_margin=3mm,
) 
hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Pixel size" )
plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
marker=(:cross, :purple, 6),
line=(:purple, :dash, 2, 0.5),
markerstrokewidth=2,
label="10142024"
)
plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
line=(:dash, :blue, 3),
markerstrokewidth=2,
label="10142024: QM"
)
plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
line=(:dot, :red, 3),
markerstrokewidth=2,
label="10142024: CQD"
)
display(fig_lin)

fig = plot(fig_log, fig_lin,
    layout=(1,2),
    size=(1000,400),
    left_margin=5mm,
    bottom_margin=5mm
)
display(fig)
saveplot(fig,"fw_00")



# Compute absolute values for plotting
y = df_fw[!,:F1_z_centroid_mm];
y_abs = abs.(y);
# Create masks for negative and non-negative values
neg_mask = y .< 0;
pos_mask = .!neg_mask;
fig10=plot(
    abs.(df_fw[pos_mask,:I_coil_mA]/1000), y_abs[pos_mask],
    yerr = df_fw[pos_mask,:F1_z_centroid_se_mm],
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-4,1.5),
    title = "F=1 Peak Position vs Current",
    label = "08142025",
    seriestype = :scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    legend = :bottomright,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = :log10,
    framestyle = :box,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
) ;
plot!(abs.(df_fw[neg_mask,:I_coil_mA]/1000), y_abs[neg_mask],
    yerr = df_fw[neg_mask,:F1_z_centroid_se_mm],
    label=false, 
    seriestype=:scatter,
    marker = (:xcross, :orangered2, 4),
    markerstrokecolor = :orangered2,
    markerstrokewidth = 2,
);
hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Pixel size" )
plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
marker=(:cross, :purple, 6),
line=(:purple, :dash, 2, 0.5),
markerstrokewidth=2,
label="10142024"
);
plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
line=(:dash, :blue, 3),
markerstrokewidth=2,
label="10142024: QM"
);
plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
line=(:dot, :red, 3),
markerstrokewidth=2,
label="10142024: CQD"
);
display(fig10)

fig11=plot(
    abs.(df_fw[pos_mask,:I_coil_mA]/1000), y_abs[pos_mask],
    yerr = df_fw[pos_mask,:F1_z_centroid_se_mm],
    xaxis = ( L"$I_{c} \ (\mathrm{A})$"),
    yaxis = (L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$"),
    title = "F=1 Peak Position vs Current",
    label = "08142025",
    seriestype = :scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    legend = :bottomright,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
) ;
plot!(abs.(df_fw[neg_mask,:I_coil_mA]/1000), y_abs[neg_mask],
    yerr = df_fw[neg_mask,:F1_z_centroid_se_mm],
    label=false, 
    seriestype=:scatter,
    marker = (:xcross, :orangered2, 4),
    markerstrokecolor = :orangered2,
    markerstrokewidth = 2,
);
hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Pixel size" )
plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
marker=(:cross, :purple, 6),
line=(:purple, :dash, 2, 0.5),
markerstrokewidth=2,
label="10142024"
);
plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
line=(:dash, :blue, 3),
markerstrokewidth=2,
label="10142024: QM"
);
plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
line=(:dot, :red, 3),
markerstrokewidth=2,
label="10142024: CQD"
);
display(fig11)

fig = plot(fig10, fig11,
    layout=(1,2),
    size=(1000,400),
    left_margin=5mm,
    bottom_margin=5mm
)
display(fig)
saveplot(fig,"fw_01")



#########################################################################################
#########################################################################################
############################## COMPARISON ###############################################
#########################################################################################
#########################################################################################
fig_comp=plot(
    abs.(df_fw[4:end,:I_coil_mA]/1000), abs.(df_fw[4:end,:F1_z_centroid_mm])/magnification_factor,
    yerror = df_fw[!,:F1_z_centroid_se_mm],
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    # xlims = (1e-3,1.0),
    ylims = (0.9e-3,3.5),
    title = "F=1 Peak Position vs Current",
    label = "08142025",
    seriestype = :scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    legend = :bottomright,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
    left_margin=3mm,
) 
plot!(fig_comp,data_JSF[:model][:,1], data_JSF[:model][:,2],
line=(:dash, :blue, 3),
markerstrokewidth=2,
label="10142024: QM"
)
plot!(fig_comp,data_JSF[:model][:,1], data_JSF[:model][:,3],
line=(:dot, :red, 3),
markerstrokewidth=2,
label="10142024: CQD"
)
saveplot(fig_comp,"comparison_zoom")

fig_comp=plot(
    abs.(df_fw[!,:I_coil_mA]/1000), abs.(df_fw[!,:F1_z_centroid_mm])/magnification_factor,
    yerror = df_fw[!,:F1_z_centroid_se_mm],
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-6,3.5),
    title = "F=1 Peak Position vs Current",
    label = "08142025",
    seriestype = :scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    legend = :bottomright,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = :log10,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
    left_margin=3mm,
) 
hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Pixel size" )
plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
marker=(:cross, :purple, 6),
line=(:purple, :dash, 2, 0.5),
markerstrokewidth=2,
label="10142024"
)
plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
line=(:dash, :blue, 3),
markerstrokewidth=2,
label="10142024: QM"
)
plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
line=(:dot, :red, 3),
markerstrokewidth=2,
label="10142024: CQD"
)
display(fig_log)
sim_data = CSV.read("./simulation_data/results_CQD_20250818T160111.csv",DataFrame; header=false)
# simqm    = CSV.read("./simulation_data/results_QM_20250728T105702.csv",DataFrame; header=false)
kis = [1.50,1.80,2.00,2.10,2.20,2.25,2.30,2.40,2.50,2.60] # ×10^-6
colors = palette(:phase, length(kis) );
for i=1:length(kis)
    plot!(sim_data[:,1],abs.(sim_data[:,21+i]), 
    label=L"CQD $k_{i}=%$(kis[i])\times10^{-6}$",
    line=(:dash,colors[i],2))
end
display(fig_comp)
saveplot(fig_comp,"comparison")


#########################################################################################
#########################################################################################

T_END = Dates.now()
T_RUN = Dates.canonicalize(T_END-T_START)


report = """
***************************************************
EXPERIMENT
    Single Stern–Gerlach Experiment
    Data directory          : $(data_directory)
    Output directory        : $(OUTDIR)

CAMERA FEATURES
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm

IMAGES INFORMATION
    Magnification factor    : $magnification_factor
    Binning                 : $(exp_bin_x) × $(exp_bin_z)
    Effective pixels        : $(x_pixels) × $(z_pixels)
    Pixel size              : $(1e6*exp_pixelsize_x)μm × $(1e6*exp_pixelsize_z)μm
    xlims                   : ($(round(minimum(1e6*x_position), digits=6)) μm, $(round(maximum(1e3*x_position), digits=4)) mm)
    zlims                   : ($(round(minimum(1e6*z_position), digits=6)) μm, $(round(maximum(1e3*z_position), digits=4)) mm)

EXPERIMENT CONDITIONS
    Currents (mA)           : $(round.(1e3*Icoils,digits=5))
    No. of current          : $(nI)
    Temperature (K)         : $(Temperature)

ANALYSIS PROPERTIES
    Binning                 : $(n_bins)
    Smoothing parameter     : $(λ0)
    Mean F1 peak (mm)       : $(round.(df_mean[!,:F1_z_centroid_mm],digits=6))
    Framewise F1 peak (mm)  : $(round.(df_fw[!,:F1_z_centroid_mm], digits=6))

CODE
    Code name               : $(PROGRAM_FILE),
    Start date              : $(T_START)
    End data                : $(T_END)
    Run time                : $(T_RUN)
    Hostname                : $(hostname)

***************************************************
"""

# Print to terminal
println(report)

# Save to file
open(joinpath(OUTDIR,"experiment_report.txt"), "w") do io
    write(io, report)
end

println("Experiment analysis finished!")
alert("Experiment analysis finished!")



