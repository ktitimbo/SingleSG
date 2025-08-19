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
const T_START = Dates.now()
# Custom modules
include("./Modules/MyExperimentalAnalysis.jl");
using .MyExperimentalAnalysis;
# Set the working directory to the current location
cd(@__DIR__) ;
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSS");
const OUTDIR    = joinpath(@__DIR__, "analysis_data", RUN_STAMP);
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
outfile = joinpath(data_directory, "data.jld2")

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
magnification_factor = 1.3 ;
# Experiment resolution
exp_bin_x, exp_bin_z = (4,1) ;  # Camera binning
exp_pixelsize_x, exp_pixelsize_z = (exp_bin_x, exp_bin_z).*cam_pixelsize ; # Effective pixel size after binning [m]
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
n_bins = 1

if !isfile(outfile)
    @info "Not found → building $outfile"
    data = stack_data(data_directory; order=:desc, keynames=("BG","F1","F2"))
    jldsave(outfile, data=data)
    data = nothing

else
    @info "Found $outfile → skipping build"
end
raw_data = load(outfile)["data"]
Icoils = raw_data[:Currents]
nI = length(Icoils)

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


data_processed = build_processed_dict(raw_data, img_dk,img_fl)

########################################################################
############# MEAN ANALYSIS #######################################
########################################################################
profiles_mean_analysis = zeros(Float64, nI, z_pixels)
for j in 1:nI
        # --- Load stack (Nx × Nz × Nframes at j-th current)
        stack = Float64.(data_processed[:F1ProcessedImages][:,:,:, j])
        n_frames = size(stack, 3) # Number of frames in the signal

        # --- Per-frame z-profiles (mean over x → 1×Nz, then vec)
        frame_profiles = [vec(mean(stack[:, :, i], dims=1)) for i in 1:n_frames]
        frame_profiles_mat = reduce(hcat, frame_profiles)'  # (Nframes × Nz)

        # --- Mean profile over all frames and x
        mean_over_frames = mean(stack, dims=3)                # (Nx × Nz × 1)
        mean_over_frames = dropdims(mean_over_frames; dims=3) # (Nx × Nz)
        # --- Mean profile over x : overall mean signal
        mean_profile = mean(mean_over_frames, dims=1)         # (1 × Nz)
        mean_profile = vec(mean_profile)                      # (Nz)
        profiles_mean_analysis[j,:] = 1e3 * mean_profile
end

cols = palette(:darkrainbow, nI)
fig = plot(title="F1 processed data",
    xlabel=L"$z$ (mm)",
    ylabel="Intensity (au)")
for i=1:nI
    plot!(fig,1e3*z_position,profiles_mean_analysis[i,:], line=(:solid,cols[i],1), label=L"$I_{c}=%$(round(1e3*Icoils[i], digits=3))\mathrm{mA}$")
end
plot!(fig,legend=:outerright, legend_columns=1, foreground_color_legend=nothing)
display(fig)
saveplot(fig,"mean_f1_processed")

profiles_mean_analysis = zeros(Float64, nI, z_pixels)
for j in 1:nI
        # --- Load stack (Nx × Nz × Nframes at j-th current)
        stack = Float64.(data_processed[:F2ProcessedImages][:,:,:, j])
        n_frames = size(stack, 3) # Number of frames in the signal

        # --- Per-frame z-profiles (mean over x → 1×Nz, then vec)
        frame_profiles = [vec(mean(stack[:, :, i], dims=1)) for i in 1:n_frames]
        frame_profiles_mat = reduce(hcat, frame_profiles)'  # (Nframes × Nz)

        # --- Mean profile over all frames and x
        mean_over_frames = mean(stack, dims=3)                # (Nx × Nz × 1)
        mean_over_frames = dropdims(mean_over_frames; dims=3) # (Nx × Nz)
        # --- Mean profile over x : overall mean signal
        mean_profile = mean(mean_over_frames, dims=1)         # (1 × Nz)
        mean_profile = vec(mean_profile)                      # (Nz)
        profiles_mean_analysis[j,:] = 1e3 * mean_profile
end

cols = palette(:darkrainbow, nI)
fig = plot(title="F2 processed data",
    xlabel=L"$z$ (mm)",
    ylabel="Intensity (au)")
for i=1:nI
    plot!(fig,1e3*z_position,profiles_mean_analysis[i,:], line=(:solid,cols[i],1), label=L"$I_{c}=%$(round(1e3*Icoils[i], digits=3))\mathrm{mA}$")
end
plot!(fig,legend=:outerright, legend_columns=1, foreground_color_legend=nothing)
display(fig)
saveplot(fig,"mean_f2_processed")


########################################################################
############# FRAMEWISE ANALYSIS #######################################
########################################################################

f1_max = my_process_framewise_maxima("F1", data_processed, n_bins; half_max=true,λ0=0.03)
f2_max = my_process_framewise_maxima("F2", data_processed, n_bins; half_max=true,λ0=0.03)

f1_z_mm , f1_zstd_mm  = (vec(mean(f1_max, dims=1)) , vec(std(f1_max, dims=1)));
f2_z_mm , f2_zstd_mm  = (vec(mean(f2_max, dims=1)) , vec(std(f2_max, dims=1)));
 
data_centroid = (f1_z_mm .+ f2_z_mm)/2
centroid_fw = mean(data_centroid, Weights(nI:-1:1))
centroid_std_err = std(data_centroid, Weights(nI:-1:1); corrected=false) / sqrt(nI)
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

fig_log=plot(
    abs.(df_fw[!,:I_coil_mA]/1000), abs.(df_fw[!,:F1_z_centroid_mm])/magnification_factor,
    yerror = df_fw[!,:F1_z_centroid_se_mm],
    # xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    # yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-6,3.5),
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
plot!(left_margin=2mm,)
sim_data = CSV.read("./simulation_data/results_CQD_20250807T135817.csv",DataFrame; header=false)
# simqm    = CSV.read("./simulation_data/results_QM_20250728T105702.csv",DataFrame; header=false)
kis = [1.50,1.80,2.00,2.10,2.20,2.25,2.30,2.40,2.50,2.60] # ×10^-6
colors = palette(:phase, length(kis) );
for i=1:length(kis)
    plot!(sim_data[:,1],abs.(sim_data[:,21+i]), 
    label=L"CQD $k_{i}=%$(kis[i])\times10^{-6}$",
    line=(:dash,colors[i],2))
end
plot!(right_margin=1mm)







n_frames    = size(raw_data[:F1_data],3)
n_currents  = size(raw_data[:F1_data],4)

raw_data[:F1_data]


F1_profiles = dropdims(mean(raw_data[:BG_data]; dims=1), dims=1)
cols=palette(:phase, n_frames*n_currents)
fig = plot(
    title = "Background profile",
    legend=false
)
for j=1:n_currents, i=1:n_frames
    k = (j-1)*n_frames + i
    plot!(fig, z_position, F1_profiles[:,i,j], line=(:solid,cols[k],1))
end
display(fig)

F1_profiles = dropdims(mean(F1_profiles; dims=2);dims=2)
cols=palette(:darktest, n_currents)
fig = plot(
    title = "Background profile",
    legend=:best
)
for j=1:4:n_currents
    print(j)
    plot!(fig, z_position, F1_profiles[:,j], line=(:solid,cols[j],1),label=L"$I_{c}=%$(round(1e3*Icoils[j], digits=3))\mathrm{mA}$")
end
display(fig)


data = matread(joinpath(data_directory, "data.mat")) ;

mean(raw_data[:F1_data][:,:,1,1],dims=1)

for i=1:size(data[:F1_data],3)
    img = heatmap(raw_data[:F1_data][:,:,i,1])
    display(img)
end

old = matread(joinpath("20250725/", "data.mat"))
old["data"]["F1ProcessedImages"][1,41]


["data"]

dd = matread(files[1])

dd["BG"]

# Read data
data = matread(joinpath(data_directory, "data.mat")) ;

# Extract coil currents (in mA)
Icoils = vec(data["data"]["Current_mA"]/1000)
nI = length(Icoils)     # Number of valid current files found

fig_I0 = plot(abs.(reverse(-Icoils)), 
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
vspan!([0, findlast(<(0), reverse(-Icoils))+0.25], color=:gray, alpha=0.30,label="unresolved" );
display(fig_I0) 
saveplot(fig_I0, "current_range")


##########################################################################################
##########################################################################################
# Run for F1 and F2 signals: MEAN OF FRAMES
##########################################################################################
##########################################################################################

f1_data_mean = process_maxima(:mean,"F1", data, n_bins )
f2_data_mean = process_maxima(:mean,"F2", data, n_bins )

data_centroid = (f1_data_mean .+ f2_data_mean)/2
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
)
hline!([centroid_mean], label=L"Centroid $z=%$(round(centroid_mean,digits=3))$mm")
hspan!( [centroid_mean - centroid_std,centroid_mean + centroid_std], color=:orangered, alpha=0.30, label=L"St.Err. = $\pm%$(round(centroid_std,digits=3))$mm")
saveplot(fig,"mean_centroid")

df_mean = DataFrame(
    I_coil_mA          = -1000 .* Icoils,
    F1_z_peak_mm       = f1_data_mean,
    F2_z_peak_mm       = f2_data_mean,
    Δz_mm              = f1_data_mean .- f2_data_mean,
    F1_z_centroid_mm   = f1_data_mean .- centroid_mean,
    F2_z_centroid_mm   = f2_data_mean .- centroid_mean,
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

fig=plot(
    abs.(df_mean[!,:I_coil_mA]/1000), abs.(df_mean[!,:F1_z_centroid_mm]),
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-4,1.5),
    title = "F=1 Peak Position vs Current",
    label = "07122025",
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
saveplot(fig, "mean_00")


# Compute absolute values for plotting
y = df_mean[!,:F1_z_centroid_mm];
y_abs = abs.(y);
# Create masks for negative and non-negative values
neg_mask = y .< 0;
pos_mask = .!neg_mask;
fig=plot(
    abs.(df_mean[pos_mask,:I_coil_mA]/1000), y_abs[pos_mask],
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-4,1.5),
    title = "F=1 Peak Position vs Current",
    label = "07122025",
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
plot!(abs.(df_mean[neg_mask,:I_coil_mA]/1000), y_abs[neg_mask], 
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
saveplot(fig, "mean_01")


##########################################################################################
##########################################################################################
# Run for F1 and F2 signals: : FRAMEWISE
##########################################################################################
##########################################################################################

f1_data_framewise = process_maxima(:framewise,"F1", data, n_bins )
f2_data_framewise = process_maxima(:framewise,"F2", data, n_bins )

data_centroid = vec(mean((f1_data_framewise .+ f2_data_framewise)/2, dims=1))
centroid_fw = mean(data_centroid, Weights(nI-1:-1:0))
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
    legend=:topleft,
)
hline!([centroid_fw], label=L"Centroid $z=%$(round(centroid_fw,digits=3))$mm")
hspan!( [centroid_fw - centroid_std,centroid_fw + centroid_std], color=:orangered, alpha=0.30, label=L"St.Err. = $\pm%$(round(centroid_std,digits=3))$mm")
saveplot(fig,"fw_centroid")

res = summarize_framewise(f1_data_framewise, f2_data_framewise, Icoils, centroid_fw, centroid_std; rev_order=true);
df_framewise = DataFrame(
    I_coil_mA           = res.I_coil_mA,

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
CSV.write(joinpath(OUTDIR, "fw_data.csv"), df_framewise);

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
    df_framewise;
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



fig_01 = plot(abs.(df_framewise[!,:I_coil_mA]/1000), df_framewise[!,:F1_z_peak_mm ],
    ribbon= df_framewise[!, :F1_z_peak_se_mm ],
    label=L"$F_{1}$",
    line=(:solid,:red,1),
    fillalpha=0.23, 
    fillcolor=:red,  
)
plot!(abs.(df_framewise[!,:I_coil_mA]/1000), df_framewise[!,:F1_z_peak_mm ], fillrange=df_framewise[!,:F2_z_peak_mm ],
    # abs.(data_framewise[:,1]/1000), data_framewise[:,2], fillrange=data_framewise[:,4],
    fillalpha=0.05,
    color=:purple,
    label = false,
)
plot!(abs.(df_framewise[!,:I_coil_mA]/1000), df_framewise[!,:F2_z_peak_mm ],
    ribbon= df_framewise[!, :F2_z_peak_se_mm ],
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
    [1e-8, abs(df_framewise.I_coil_mA[findlast(<(0), df_mean.I_coil_mA)+1] / 1000)],
    color = :gray,
    alpha = 0.30,
    label = "zero"
)
hline!([centroid_fw], line=(:dot,:black,2), label="Centroid")


fig_02 = plot(abs.(df_framewise[!,:I_coil_mA]/1000), df_framewise[!,:F1_z_peak_mm] ,
    ribbon= df_framewise[!,:F1_z_peak_se_mm],
    label=L"$F_{1}$",
    line=(:solid,:red,2),
)
plot!(abs.(df_framewise[!,:I_coil_mA]/1000), 2*centroid_fw .- df_framewise[!,:F2_z_peak_mm] ,
    ribbon= ribbon= df_framewise[!,:F2_z_peak_se_mm],
    label=L"Mirrored $F_{2}$",
    line=(:solid,:blue,2),
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
    [1e-8, abs(df_framewise.I_coil_mA[findlast(<(0), df_framewise.I_coil_mA)+1] / 1000)],
    color = :gray,
    alpha = 0.30,
    label = "zero"
);
# Fill between y1 and y2
plot!(abs.(df_framewise[!,:I_coil_mA]/1000), df_framewise[!,:F1_z_peak_mm], fillrange=2*centroid_mean .- df_framewise[!,:F2_z_peak_mm],
    fillalpha=0.2,
    color=:purple,
    label = false,
);
hline!([centroid_fw], line=(:dot,:black,2), label="Centroid")

fig_03 = plot(abs.(df_framewise[!,:I_coil_mA]/1000), df_framewise[!,:F1_z_centroid_mm] ,
    ribbons=df_framewise[!,:F1_z_centroid_se_mm],
    label=L"$F_{1}$",
    line=(:solid,:red,2),
);
plot!(abs.(df_framewise[!,:I_coil_mA]/1000), df_framewise[!,:F2_z_centroid_mm] ,
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
    [1e-8, abs(df_framewise.I_coil_mA[findlast(<(0), df_framewise.I_coil_mA)+1] / 1000)],
    color = :gray,
    alpha = 0.30,
    label = "zero"
);
# Fill between y1 and y2
plot!(abs.(df_framewise[!,:I_coil_mA]/1000), df_framewise[!,:F1_z_centroid_mm], fillrange=df_framewise[!,:F2_z_centroid_mm],
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

fig=plot(
    abs.(df_framewise[!,:I_coil_mA]/1000), abs.(df_framewise[!,:F1_z_centroid_mm]),
    yerror = df_framewise[!,:F1_z_centroid_se_mm],
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-4,1.5),
    title = "F=1 Peak Position vs Current",
    label = "07122025",
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
saveplot(fig, "fw_00")


# Compute absolute values for plotting
y = df_framewise[!,:F1_z_centroid_mm];
y_abs = abs.(y);
# Create masks for negative and non-negative values
neg_mask = y .< 0;
pos_mask = .!neg_mask;
fig=plot(
    abs.(df_framewise[pos_mask,:I_coil_mA]/1000), y_abs[pos_mask],
    yerr = df_framewise[pos_mask,:F1_z_centroid_se_mm],
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-4,1.5),
    title = "F=1 Peak Position vs Current",
    label = "07122025",
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
plot!(abs.(df_framewise[neg_mask,:I_coil_mA]/1000), y_abs[neg_mask],
    yerr = df_framewise[neg_mask,:F1_z_centroid_se_mm],
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
display(fig)
saveplot(fig, "fw_01")


###########################################################################################################
###########################################################################################################
# Comparison with Xukun's analysis
fig=plot(
    abs.(df_framewise[!,:I_coil_mA]/1000), abs.(df_framewise[!,:Δz_mm]),
    yerror = df_framewise[!,:Δz_se_mm],
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$\Delta z \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-4,3.0),
    title = "Peak Separation vs Current",
    label = "KT",
    # seriestype = :scatter,
    line=(:solid,:red,2),
    marker = (:circle, :white, 1),
    markerstrokecolor = :red,
    markerstrokewidth = 3,
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
plot!(abs.(vec(data["data"]["Current_mA"])/1000), vec(data["data"]["PeakSeparationMean_um"]/1000),
yerr=vec(data["data"]["PeakSeparationStd_um"])/1000,
label="XL",
# seriestype=:scatter,
line=(:solid,:purple,1),
marker=(:square,:white,1),
markerstrokewidth=1,
markerstrokecolor=:purple
)
###########################################################################################################
###########################################################################################################
t_run = Dates.canonicalize(Dates.now()-T_START);
println("Running time : ", t_run)
############################################################################################################################################################################
############################################################################################################################################################################

data_20250712 = [
    0.0    8.44791  0.000518525  8.42793  0.00118161;
    0.002  8.44287  0.000938761  8.42958  0.00233814;
    0.004  8.43723  0.000910976  8.42722  0.0027258;
    0.006  8.43482  0.000990039  8.42604  0.0018639;
    0.008  8.43537  0.00114152   8.42256  0.000477835;
    0.01   8.44025  0.000706939  8.42801  0.0014779;
    0.012  8.43449  0.00169919   8.42741  0.00162732;
    0.016  8.43596  0.00175432   8.43447  0.000601924;
    0.022  8.44658  0.000916083  8.43388  0.000597619;
    0.032  8.46016  0.000435286  8.42367  0.000669231;
    0.044  8.48834  0.00132164   8.39899  0.00107894;
    0.058  8.53888  0.00401241   8.36016  0.00421937;
    0.074  8.58065  0.00165977   8.30529  0.0031774;
    0.094  8.64399  0.00258139   8.24529  0.00294799;
    0.114  8.70363  0.00274223   8.20613  0.000977637;
    0.134  8.74965  0.00140882   8.13372  0.00068276;
    0.154  8.79234  0.00346408   8.0876   0.00190484;
    0.174  8.85616  0.00228569   8.01507  0.00452789;
    0.195  8.89328  0.00255722   7.96818  0.00401482;
    0.215  8.94618  0.00363949   7.90005  0.00386873;
    0.236  9.00505  0.00196132   7.84441  0.00613796;
    0.26   9.05515  0.00203986   7.83049  0.00222426;
    0.285  9.1162   0.00694458   7.72989  0.00238672;
    0.308  9.15401  0.0087926    7.71359  0.00904536;
    0.341  9.27811  0.0156376    7.65668  0.00283654;
    0.381  9.29719  0.00758248   7.52245  0.00229699;
    0.42   9.37437  0.00336082   7.43683  0.00277513;
    0.46   9.45703  0.00700836   7.37241  0.00276939;
    0.5    9.55574  0.015104     7.28498  0.0137936;
    0.55   9.6742   0.0112162    7.19503  0.012108;
    0.609  9.76674  0.0125186    7.04633  0.00839444;
]

plot(data_JSF[:exp][:,1], abs.(data_JSF[:exp][:,2]),
marker=(:cross, :purple, 6),
line=(:purple, :dash, 2, 0.5),
markerstrokewidth=2,
label="10142024"
)
plot!(data_JSF[:model][:,1], abs.(data_JSF[:model][:,2]),
line=(:dash, :blue, 3),
markerstrokewidth=2,
label="10142024: QM"
)
plot!(data_JSF[:model][:,1], abs.(data_JSF[:model][:,3]),
line=(:dot, :red, 3),
markerstrokewidth=2,
label="10142024: CQD"
)
plot!(data_20250712[:,1], abs.(data_20250712[:,2] .- data_20250712[1,2]),
ribbon = data_20250712[:,3] ,
label="07122025",
marker=(:xcross,:green,2),
line=(:solid,:green,2),
fillalpha=0.23, 
fillcolor=:green,  
)
plot!(abs.(df_framewise[!,:I_coil_mA]/1000), abs.(df_framewise[!,:F1_z_centroid_mm]),
    yerror = df_framewise[!,:F1_z_centroid_se_mm],
    marker=(:circle, :white, 3),
    markerstrokewidth=2,
    markerstrokecolor=:black,
    line=(:solid,:black,2),
    label = "20250725",
    fillalpha=0.25, 
    fillcolor=:gray,  
)
plot!(
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$\Delta z \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-4,2.0),
    title = "F1 Peak positions vs Current",
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

sim_data = CSV.read("./simulation_data/results_CQD_20250807T135817.csv",DataFrame; header=false)
simqm    = CSV.read("./simulation_data/results_QM_20250728T105702.csv",DataFrame; header=false)

kis = [1.50,1.80,2.00,2.10,2.20,2.25,2.30,2.40,2.50,2.60] # ×10^-6
# Compute absolute values for plotting
y = df_framewise[!,:F1_z_centroid_mm];
y_abs = abs.(y)
# Create masks for negative and non-negative values
neg_mask = y .< 0
pos_mask = .!neg_mask
fig=plot(
    abs.(df_framewise[pos_mask,:I_coil_mA]/1000), y_abs[pos_mask],
    yerr = df_framewise[pos_mask,:F1_z_centroid_se_mm],
    label = "20250725",
    seriestype = :scatter,
    marker = (:circle, :white, 2),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
)
plot!(
    abs.(df_framewise[neg_mask,:I_coil_mA]/1000), y_abs[neg_mask],
    yerr = df_framewise[neg_mask,:F1_z_centroid_se_mm],
    label=false, 
    seriestype=:scatter,
    marker = (:circle, :white, 2),
    markerstrokecolor = :chocolate4,
    markerstrokewidth = 2,
)
plot!(
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.015,1.0),
    ylims = (1e-3,1.5),
    title = "F=1 Peak Position vs Current",
    legend = :bottomright,
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    framestyle = :box,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
) 
colors = palette(:phase, length(kis) );
for i=1:length(kis)
    plot!(sim_data[:,1],abs.(sim_data[:,21+i]), 
    label=L"CQD $k_{i}=%$(kis[i])\times10^{-6}$",
    line=(:dash,colors[i],2))
end
hspan!([1e-6,1000*n_bins* cam_pixelsize], color=:gray, alpha=0.30, label="Pixel size" )