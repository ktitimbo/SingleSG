# Kelvin Titimbo
# California Institute of Technology
# March 2026

############## EXPERIMENT ANALYSIS PREAMBLE ##############
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
using BSplineKit, Optim, Dierckx
# Aesthetics and output formatting
using Colors, ColorSchemes
using Printf, LaTeXStrings, PrettyTables
using CSV, DataFrames, DelimitedFiles
# Time-stamping/logging
using Dates
using Alert
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(2)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Custom modules
include("./Modules/DataReading.jl")
include("./Modules/JLD2_MyTools.jl");
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
# STERN–GERLACH EXPERIMENT SETUP
# Camera and pixel geometry : intrinsic properties
cam_pixelsize           = 6.5e-6 ;  # Physical pixel size of camera [m]
nx_pixels , nz_pixels   = (2160, 2560); # (Nx,Nz) pixels
# Experiment resolution
exp_bin_x, exp_bin_z    = (4,1) ;  # Camera binning
exp_pixelsize_x, exp_pixelsize_z = (exp_bin_x, exp_bin_z).*cam_pixelsize ; # Effective pixel size after binning [m]
# Image dimensions (adjusted for binning)
x_pixels = Int(nx_pixels / exp_bin_x);  # Number of x-pixels after binning
z_pixels = Int(nz_pixels / exp_bin_z);  # Number of z-pixels after binning
# Spatial axes shifted to center the pixels
x_position = pixel_positions(x_pixels, 1, exp_pixelsize_x);
z_position = pixel_positions(z_pixels, 1, exp_pixelsize_z);
println("""
***************************************************
CAMERA FEATURES
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm

IMAGES INFORMATION
    Binning                 : $(exp_bin_x) × $(exp_bin_z)
    Effective pixels        : $(x_pixels) × $(z_pixels)
    Pixel size              : $(1e6*exp_pixelsize_x)μm × $(1e6*exp_pixelsize_z)μm
    xlims                   : ($(round(minimum(1e6*x_position), digits=6)) μm, $(round(maximum(1e3*x_position), digits=4)) mm)
    zlims                   : ($(round(minimum(1e6*z_position), digits=6)) μm, $(round(maximum(1e3*z_position), digits=4)) mm)
***************************************************
""")
# Setting the variables for the module
MyExperimentalAnalysis.effective_cam_pixelsize_z    = exp_pixelsize_z;
MyExperimentalAnalysis.x_pixels                     = x_pixels;
MyExperimentalAnalysis.z_pixels                     = z_pixels;

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nz = 2
λ0 = 0.005

# position
z_mm        = 1e3 .* pixel_positions(z_pixels, nz, exp_pixelsize_z)
z_mm_error  = 1e3 * 0.5 * exp_pixelsize_z * nz # half of the pixel size
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_directory = "20260311"

T_START   = Dates.now()
RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
OUTDIR    = joinpath(@__DIR__, "SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

outfile_raw         = joinpath("EXPERIMENTS",data_directory, "data.jld2")
outfile_processed   = joinpath("EXPERIMENTS",data_directory, "data_processed.jld2")
data_summary_path   = joinpath(@__DIR__, "SG0_EXPDATA_ANALYSIS","summary",data_directory)
isdir(data_summary_path) || mkpath(data_summary_path);


if !isfile(outfile_processed) # check if the processed images exists
    if !isfile(outfile_raw) # check if the raw data exists
        @info "Not found → building $outfile_raw"
        data_in = joinpath(@__DIR__,"EXPERIMENTS", data_directory)
        data_raw = MyExperimentalAnalysis.SG0_stack_data(data_in)
        jldsave(outfile_raw, data=data_raw)
        data_raw = nothing
    else
        @info "Found $outfile_raw → skipping build"
    end

    data_raw = load(outfile_raw)["data"]
    data_processed = MyExperimentalAnalysis.SG0_build_processed_dict(data_raw)
    jldsave(outfile_processed, data=data_processed)
    data_processed = nothing
    data_raw = nothing
else
    @info "Found $outfile_processed → skipping build"
end
data_processed = load(outfile_processed)["data"]


SG0_current = data_processed[:SG0Currents]
SG1_current = data_processed[:SG1Currents]

f1_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F1", data_processed, nz; half_max=true, λ0=λ0);
f2_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F2", data_processed, nz; half_max=true, λ0=λ0);

f1_max = MyExperimentalAnalysis.SG0_framewise_maxima("F1", data_processed, nz ; half_max=true,λ0=λ0);
f2_max = MyExperimentalAnalysis.SG0_framewise_maxima("F2", data_processed, nz ; half_max=true,λ0=λ0);

f1_z_mm , f1_z_sem_mm  = vec(mean(f1_max, dims=1)) , sqrt.(vec(std(f1_max, dims=1; corrected=true) ./ sqrt(size(f1_max,1))).^2 .+ z_mm_error^2 );
f2_z_mm , f2_z_sem_mm  = vec(mean(f2_max, dims=1)) , sqrt.(vec(std(f2_max, dims=1; corrected=true) ./ sqrt(size(f2_max,1))).^2 .+ z_mm_error^2 );


data = hcat(SG0_current,SG1_current, f1_z_mm, f1_z_sem_mm)
cols = palette(:darkrainbow, size(data,1))   # generate colors


data01 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
fig01 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data01,1)
    plot!(fig01,
        [1000*data01[i,2]], [data01[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c0} = %$(round(data01[i,1];digits=3))\mathrm{A}$")
end
plot!(fig01,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig01)


data02 = DataReading.subset_by_cols(data01,[2]; thr = 1e-6, include_equal=true )[3]
fig02 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data02,1)
    plot!(fig02,
        [1000*data02[i,2]], [data02[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data02[i,4]],
        label=L"$I_{c0} = %$(round(data02[i,1];digits=3))\mathrm{A}$")
end
plot!(fig02,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig02)


data03 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
zcenter = data03[1,3]
data03[:,3] = data03[:,3] .- zcenter
data03 = DataReading.subset_by_cols(data03,[2]; thr = 1e-6, include_equal=true )[3]
fig03 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data03,1)
    plot!(fig03,
        [1000*data03[i,2]], [data03[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data03[i,4]],
        label=L"$I_{c0} = %$(round(data03[i,1];digits=3))\mathrm{A}$")
end
plot!(fig03,
    yscale=:log10,
    ylims=(1e-2,1e-1),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig03)

p1 = plot(fig01,fig02,fig03,
layout=@layout([a ; b ; c]),
size=(900,800)
)
saveplot(p1,"zvssg1")


data04 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
fig04 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data04,1)
    plot!(fig04,
        [data04[i,1]], [data04[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c1} = %$(round(1000*data04[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig04,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig04)


data05 = DataReading.subset_by_cols(data04,[1]; thr = 1e-6, include_equal=true )[3]
fig05 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data05,1)
    plot!(fig05,
        [data05[i,1]], [data05[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data05[i,4]],
        label=L"$I_{c1} = %$(round(1000*data05[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig05,
    xscale=:log10,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig05)


data06 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
zcenter = data06[1,3]
data06[:,3] = data06[:,3] .- zcenter
data06 = DataReading.subset_by_cols(data06,[1]; thr = 1e-6, include_equal=true )[3]
fig06 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data06,1)
    plot!(fig06,
        [data06[i,1]], [data06[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data06[i,4]],
        label=L"$I_{c1} = %$(round(1000*data06[i,2];digits=3))\mathrm{mA}$")
end
plot!(fig06,
    xscale=:log10,
    yscale=:log10,
    ylims=(3e-2,1e-1),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig06)

p2 = plot(fig04,fig05,fig06,
layout=@layout([a ; b ; c]),
size=(900,800)
)
saveplot(p2,"zvssg0")




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_directory = "20260312A"

T_START   = Dates.now()
RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
OUTDIR    = joinpath(@__DIR__, "SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

outfile_raw         = joinpath("EXPERIMENTS",data_directory, "data.jld2")
outfile_processed   = joinpath("EXPERIMENTS",data_directory, "data_processed.jld2")
data_summary_path   = joinpath(@__DIR__, "SG0_EXPDATA_ANALYSIS","summary",data_directory)
isdir(data_summary_path) || mkpath(data_summary_path);


if !isfile(outfile_processed) # check if the processed images exists
    if !isfile(outfile_raw) # check if the raw data exists
        @info "Not found → building $outfile_raw"
        data_in = joinpath(@__DIR__,"EXPERIMENTS", data_directory)
        data_raw = MyExperimentalAnalysis.SG0_stack_data(data_in)
        jldsave(outfile_raw, data=data_raw)
        data_raw = nothing
    else
        @info "Found $outfile_raw → skipping build"
    end

    data_raw = load(outfile_raw)["data"]
    data_processed = MyExperimentalAnalysis.SG0_build_processed_dict(data_raw)
    jldsave(outfile_processed, data=data_processed)
    data_processed = nothing
    data_raw = nothing
else
    @info "Found $outfile_processed → skipping build"
end
data_processed = load(outfile_processed)["data"]


SG0_current = data_processed[:SG0Currents]
SG1_current = data_processed[:SG1Currents]

f1_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F1", data_processed, nz; half_max=true, λ0=λ0);
f2_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F2", data_processed, nz; half_max=true, λ0=λ0);

f1_max = MyExperimentalAnalysis.SG0_framewise_maxima("F1", data_processed, nz ; half_max=true,λ0=λ0);
f2_max = MyExperimentalAnalysis.SG0_framewise_maxima("F2", data_processed, nz ; half_max=true,λ0=λ0);

f1_z_mm , f1_z_sem_mm  = vec(mean(f1_max, dims=1)) , sqrt.(vec(std(f1_max, dims=1; corrected=true) ./ sqrt(size(f1_max,1))).^2 .+ z_mm_error^2 );
f2_z_mm , f2_z_sem_mm  = vec(mean(f2_max, dims=1)) , sqrt.(vec(std(f2_max, dims=1; corrected=true) ./ sqrt(size(f2_max,1))).^2 .+ z_mm_error^2 );


data = hcat(SG0_current,SG1_current, f1_z_mm, f1_z_sem_mm)
cols = palette(:darkrainbow, size(data,1))   # generate colors


data01 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
fig01 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data01,1)
    plot!(fig01,
        [1000*data01[i,2]], [data01[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c0} = %$(round(data01[i,1];digits=3))\mathrm{A}$")
end
plot!(fig01,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig01)


data02 = DataReading.subset_by_cols(data01,[2]; thr = 1e-6, include_equal=true )[3]
fig02 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data02,1)
    plot!(fig02,
        [1000*data02[i,2]], [data02[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data02[i,4]],
        label=L"$I_{c0} = %$(round(data02[i,1];digits=3))\mathrm{A}$")
end
plot!(fig02,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig02)


data03 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
zcenter = data03[1,3]
data03[:,3] = data03[:,3] .- zcenter
data03 = DataReading.subset_by_cols(data03,[2]; thr = 1e-6, include_equal=true )[3]
fig03 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data03,1)
    plot!(fig03,
        [1000*data03[i,2]], [data03[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data03[i,4]],
        label=L"$I_{c0} = %$(round(data03[i,1];digits=3))\mathrm{A}$")
end
plot!(fig03,
    yscale=:log10,
    ylims=(1e-2,1e-1),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig03)

p1 = plot(fig01,fig02,fig03,
layout=@layout([a ; b ; c]),
size=(900,800)
)
saveplot(p1,"zvssg1")


data04 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
fig04 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data04,1)
    plot!(fig04,
        [data04[i,1]], [data04[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c1} = %$(round(1000*data04[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig04,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig04)


data05 = DataReading.subset_by_cols(data04,[1]; thr = 1e-6, include_equal=true )[3]
fig05 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data05,1)
    plot!(fig05,
        [data05[i,1]], [data05[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data05[i,4]],
        label=L"$I_{c1} = %$(round(1000*data05[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig05,
    xscale=:log10,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig05)


data06 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
zcenter = data06[1,3]
data06[:,3] = data06[:,3] .- zcenter
data06 = DataReading.subset_by_cols(data06,[1]; thr = 1e-6, include_equal=true )[3]
fig06 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data06,1)
    plot!(fig06,
        [data06[i,1]], [data06[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data06[i,4]],
        label=L"$I_{c1} = %$(round(1000*data06[i,2];digits=3))\mathrm{mA}$")
end
plot!(fig06,
    xscale=:log10,
    yscale=:log10,
    ylims=(3e-2,1e-1),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig06)

p2 = plot(fig04,fig05,fig06,
layout=@layout([a ; b ; c]),
size=(900,800)
)
saveplot(p2,"zvssg0")




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_directory = "20260312B"

T_START   = Dates.now()
RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
OUTDIR    = joinpath(@__DIR__, "SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

outfile_raw         = joinpath("EXPERIMENTS",data_directory, "data.jld2")
outfile_processed   = joinpath("EXPERIMENTS",data_directory, "data_processed.jld2")
data_summary_path   = joinpath(@__DIR__, "SG0_EXPDATA_ANALYSIS","summary",data_directory)
isdir(data_summary_path) || mkpath(data_summary_path);


if !isfile(outfile_processed) # check if the processed images exists
    if !isfile(outfile_raw) # check if the raw data exists
        @info "Not found → building $outfile_raw"
        data_in = joinpath(@__DIR__,"EXPERIMENTS", data_directory)
        data_raw = MyExperimentalAnalysis.SG0_stack_data(data_in)
        jldsave(outfile_raw, data=data_raw)
        data_raw = nothing
    else
        @info "Found $outfile_raw → skipping build"
    end

    data_raw = load(outfile_raw)["data"]
    data_processed = MyExperimentalAnalysis.SG0_build_processed_dict(data_raw)
    jldsave(outfile_processed, data=data_processed)
    data_processed = nothing
    data_raw = nothing
else
    @info "Found $outfile_processed → skipping build"
end
data_processed = load(outfile_processed)["data"]


SG0_current = data_processed[:SG0Currents]
SG1_current = data_processed[:SG1Currents]

f1_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F1", data_processed, nz; half_max=true, λ0=λ0);
f2_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F2", data_processed, nz; half_max=true, λ0=λ0);

f1_max = MyExperimentalAnalysis.SG0_framewise_maxima("F1", data_processed, nz ; half_max=true,λ0=λ0);
f2_max = MyExperimentalAnalysis.SG0_framewise_maxima("F2", data_processed, nz ; half_max=true,λ0=λ0);

f1_z_mm , f1_z_sem_mm  = vec(mean(f1_max, dims=1)) , sqrt.(vec(std(f1_max, dims=1; corrected=true) ./ sqrt(size(f1_max,1))).^2 .+ z_mm_error^2 );
f2_z_mm , f2_z_sem_mm  = vec(mean(f2_max, dims=1)) , sqrt.(vec(std(f2_max, dims=1; corrected=true) ./ sqrt(size(f2_max,1))).^2 .+ z_mm_error^2 );


data = hcat(SG0_current,SG1_current, f1_z_mm, f1_z_sem_mm)
cols = palette(:darkrainbow, size(data,1))   # generate colors


data01 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
fig01 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data01,1)
    plot!(fig01,
        [1000*data01[i,2]], [data01[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c0} = %$(round(data01[i,1];digits=3))\mathrm{A}$")
end
plot!(fig01,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig01)

data01
data02 = DataReading.subset_by_cols(data01,[2]; thr = 1e-6, include_equal=true )[3]
fig02 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data02,1)
    plot!(fig02,
        [1000*data02[i,2]], [data02[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data02[i,4]],
        label=L"$I_{c0} = %$(round(data02[i,1];digits=3))\mathrm{A}$")
end
plot!(fig02,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig02)


data03 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
zcenter = data03[1,3]
data03[:,3] = data03[:,3] .- zcenter
data03 = DataReading.subset_by_cols(data03,[2]; thr = 1e-6, include_equal=true )[3]
fig03 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data03,1)
    plot!(fig03,
        [1000*data03[i,2]], [data03[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data03[i,4]],
        label=L"$I_{c0} = %$(round(data03[i,1];digits=3))\mathrm{A}$")
end
plot!(fig03,
    yscale=:log10,
    # ylims=(1e-2,1e-1),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig03)

p1 = plot(fig01,fig02,fig03,
layout=@layout([a ; b ; c]),
size=(900,800)
)
saveplot(p1,"zvssg1")


data04 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
fig04 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data04,1)
    plot!(fig04,
        [data04[i,1]], [data04[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c1} = %$(round(1000*data04[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig04,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig04)


data05 = DataReading.subset_by_cols(data04,[1]; thr = 1e-6, include_equal=true )[3]
fig05 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data05,1)
    plot!(fig05,
        [data05[i,1]], [data05[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data05[i,4]],
        label=L"$I_{c1} = %$(round(1000*data05[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig05,
    xscale=:log10,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig05)


data06 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
zcenter = data06[1,3]
data06[:,3] = data06[:,3] .- zcenter
data06 = DataReading.subset_by_cols(data06,[1]; thr = 1e-6, include_equal=true )[3]
fig06 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data06,1)
    plot!(fig06,
        [data06[i,1]], [data06[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data06[i,4]],
        label=L"$I_{c1} = %$(round(1000*data06[i,2];digits=3))\mathrm{mA}$")
end
plot!(fig06,
    xscale=:log10,
    yscale=:log10,
    # ylims=(3e-2,1e-1),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig06)

p2 = plot(fig04,fig05,fig06,
layout=@layout([a ; b ; c]),
size=(900,800)
)
saveplot(p2,"zvssg0")








plot(
    f1_mean_max.mean_profile[:,1],
    f1_mean_max.mean_profile[:,2],
seriestype=:scatter,
marker=(:circle,:white,3));
plot!(
    f1_mean_max.mean_bin_profile[:,1],
    f1_mean_max.mean_bin_profile[:,2],
seriestype=:scatter,
marker=(:circle,:white,3),
markerstrokecolor=:red,);
plot!(xlims=(0.0,0.2))

