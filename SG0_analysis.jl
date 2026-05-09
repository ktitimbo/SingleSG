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

BASE_PATH = raw"F:\SternGerlachExperiments"

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
T_START   = Dates.now()
RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");

nz = 2 ;
λ0 = 0.005 ;

# position
z_mm        = 1e3 .* pixel_positions(z_pixels, nz, exp_pixelsize_z) ; 
z_mm_error  = 1/sqrt(12) * 1e3 * exp_pixelsize_z * nz ;

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_directories = [
        # "20260318/Round1",
        # "20260318/Round2",
        # "20260318/Round3",
        # "20260318/Round4",
        # "20260415A", 
        # "20260415B", 
        # "20260416A", 
        # "20260416B",
        # "20260427A",
        # "20260427B",
        # "20260427C",
        # "20260429A",
        # "20260429B",
        # "20260429C",
        # "20260429D",
        # "20260429E",
        # "20260429F",
        "20260504/Round1_SG1=0mA_SG0=+MG=3A",
        "20260504/Round2_SG1=0mA_SG0=-MG=3A",
        "20260504/Round3_SG1=40mA_SG0=+MG=3A",
        "20260504/Round4_SG1=40mA_SG0=-MG=3A",
        "20260504/Round5_SG1=500mA_SG0=+MG=3A",
        "20260504/Round6_SG1=500mA_SG0=-MG=3A",
        ];
nd = length(data_directories);

for data_directory in data_directories
    printstyled("\t" * data_directory * "\n"; color=:cyan, bold=true)
    outfile_raw         = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data.jld2")
    outfile_processed   = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2")
    data_summary_path   = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS","summary",data_directory)
    isdir(data_summary_path) || mkpath(data_summary_path);


    if !isfile(outfile_processed) # check if the processed images exists
        if !isfile(outfile_raw) # check if the raw data exists
            @info "Not found → building $outfile_raw"
            data_in = joinpath(BASE_PATH,"EXPERIMENTS", data_directory)
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

    println("\n$(data_directory) PROCESSING COMPLETED\n\n")
end


tables = Vector{DataFrame}(undef, nd);
for (idx,data_directory) in enumerate(data_directories)
    # idx = 2
    # data_directory = data_directories[idx]
    printstyled("\t" * data_directory * "\n"; color=:cyan, bold=true);
    data_processed = load(joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2"))["data"];

    OUTDIR    = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
    isdir(OUTDIR) || mkpath(OUTDIR);
    @info "Created output directory" OUTDIR
    MyExperimentalAnalysis.OUTDIR   = OUTDIR;

    # ── Currents & fields ────────────────────────────────────────────────────
    SG0_current = data_processed[:SG0Currents];
    SG1_current = data_processed[:SG1Currents];
    MG_current  = data_processed[:MGCurrents];
    Bz0 = data_processed[:SG0Bz];
    Bz1 = data_processed[:SG1Bz];

    # ── Framewise maxima & statistics ────────────────────────────────────────
    f1_max = MyExperimentalAnalysis.SG0_framewise_maxima("F1", data_processed, nz ; half_max=false,λ0=λ0);
    f2_max = MyExperimentalAnalysis.SG0_framewise_maxima("F2", data_processed, nz ; half_max=false,λ0=λ0);

    # MyExperimentalAnalysis.SG0_mean_maxima("F1", data_processed, nz ; half_max=false,λ0=λ0);
    # MyExperimentalAnalysis.SG0_mean_maxima("F2", data_processed, nz ; half_max=false,λ0=λ0);

    f1_z_mm , f1_z_sem_mm  = vec(mean(f1_max, dims=1)) , sqrt.(vec(std(f1_max, dims=1; corrected=true) ./ sqrt(size(f1_max,1))).^2 .+ z_mm_error^2 );
    f2_z_mm , f2_z_sem_mm  = vec(mean(f2_max, dims=1)) , sqrt.(vec(std(f2_max, dims=1; corrected=true) ./ sqrt(size(f2_max,1))).^2 .+ z_mm_error^2 );

    Δz_mm = -(f1_z_mm .- f2_z_mm);
    Δz_sem_mm = sqrt.( (f1_z_sem_mm).^2 .+ (f2_z_sem_mm).^2 );

    # ── Summary DataFrame ────────────────────────────────────────────────────
    data = DataFrame(
        Ig        = MG_current,
        I0        = SG0_current,
        I1        = SG1_current,
        B0        = Bz0,
        B1        = Bz1,
        zf1       = f1_z_mm,
        errzf1    = f1_z_sem_mm,
        zf2       = f2_z_mm,
        errzf2    = f2_z_sem_mm,
        split     = Δz_mm,
        errsplit  = round.(Δz_sem_mm; sigdigits=1)
    )
    tables[idx] = data
    data_pos = data[data.I0 .> 0, :]
    pretty_table(data;
            title         = data_directory,
            formatters    = [ fmt__printf("%8.3f", [1]), fmt__printf("%8.5f", 2:5), fmt__printf("%8.3f", [6,8,10]), fmt__printf("%8.4f", [7,9,11])],
            alignment     = :c,
            column_labels  = [
                ["Ig current", "I0 Current", "I1 Current", "B0 field", "B1 field", "F1", "F1 err", "F2", "F2 err", "Δz", "Δz err"], 
                ["[A]", "[A]", "[A]" ,"[T]", "[T]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
            ],
            table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
            style = TextTableStyle(
                        first_line_column_label = crayon"yellow bold",
                        column_label  = crayon"yellow",
                        table_border  = crayon"blue bold",
                        title = crayon"bold red"
                        ),
            equal_data_column_widths = true,
    )

    # ── Colour palette (one colour per SG0 current step) ─────────────────────
    colors_sg0 = palette(:darkrainbow, size(data,1));

    sg1_label = "$(round(1000 * data.I1[end], digits=2))mA"   # reused in titles

    # ── Overlay of all SG0 currents: z-position vs current ───────────────────
    common_scatter_kw = (seriestype=:scatter, marker=(:circle, 2, :white))

    fig1 = plot(data.I0, data.zf1;
        common_scatter_kw..., yerror=data.errzf1,
        label=L"$F=1$", markerstrokecolor=:red);
    plot!(fig1, data.I0, data.zf2;
        common_scatter_kw..., yerror=data.errzf2,
        label=L"$F=2$", markerstrokecolor=:blue);
    plot!(fig1, data.I0, mean([data.zf1, data.zf2]);
        label="Centre", marker=(:diamond, :white),
        markerstrokecolor=:gray47, line=(:dash, 1, :gray47));
    plot!(fig1;
        xlabel="SG0 Current (A)",
        foreground_color_legend=nothing, background_color_legend=nothing,
        yformatter=y -> @sprintf("%.3f", y));

    fig2 = plot(data_pos.I0, data_pos.zf1;
        common_scatter_kw..., yerror=data_pos.errzf1,
        label=L"$F=1$", markerstrokecolor=:red);
    plot!(fig2, data_pos.I0, data_pos.zf2;
        common_scatter_kw..., yerror=data_pos.errzf2,
        label=L"$F=2$", markerstrokecolor=:blue);
    plot!(fig2, data_pos.I0, mean([data_pos.zf1, data_pos.zf2]),
        label="Centre", marker=(:diamond, :white),
        markerstrokecolor=:gray47, line=(:dash, 1, :gray47));
    plot!(fig2;
        xscale=:log10,
        foreground_color_legend=nothing, background_color_legend=nothing,
        xlabel="SG0 Current (A)",
        yformatter=y -> @sprintf("%.3f", y));

    fig3 = plot(data.I0, data.split ./ 6.5e-3;
        common_scatter_kw..., yerror=data.errsplit ./ 6.5e-3,
        label=L"$\Delta z$", markerstrokecolor=:darkgreen);
    plot!(fig3;
        foreground_color_legend=nothing, background_color_legend=nothing,
        xlabel="SG0 Current (A)", ylabel="Separation (px)", yminorticks=false);

    fig4 = plot(data_pos.I0, data_pos.split ./ 6.5e-3;
        common_scatter_kw..., yerror=data_pos.errsplit ./ 6.5e-3,
        label=L"$\Delta z$", markerstrokecolor=:darkgreen);
    plot!(fig4;
        xscale=:log10,
        foreground_color_legend=nothing, background_color_legend=nothing,
        xlabel="SG0 Current (A)", ylabel="Separation (px)", yminorticks=false);

    fig = plot(fig1, fig2, fig3, fig4;
        suptitle     = "$(data_directory) | SG1: $sg1_label",
        layout       = (2, 2),
        size         = (1000, 600),
        link         = :x,
        left_margin  = 5mm,
        bottom_margin= 3mm,
    )
    plot!(fig[1]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
    plot!(fig[2]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
    display(fig)
    
    # ── Extract image arrays once, outside all loops ─────────────────────────
    f1imgs = data_processed[:F1ProcessedImages];
    f2imgs = data_processed[:F2ProcessedImages];
    sg1_label    = "$(data_directory) | SG1: $(round(1000*data.I1[end], digits=2))mA";
    current_label(i) = L"$%$(round(1000*SG0_current[i]; digits=3))\mathrm{mA}$";

    legend_kw = (
        legend                  = :topleft,
        legend_title            = "SG0",
        background_color_legend = :white,
        foreground_color_legend = nothing,
    );

    # ── Pre-compute cleaned images and profiles for all currents ──────────────
    # Store so each quantity is computed once and reused in both plot blocks
    f1_images   = Vector{Matrix{Float64}}(undef, length(SG0_current));
    f2_images   = Vector{Matrix{Float64}}(undef, length(SG0_current));
    f1_profiles = Vector{Vector{Float64}}(undef, length(SG0_current));
    f2_profiles = Vector{Vector{Float64}}(undef, length(SG0_current));


    for i in eachindex(SG0_current)
        # Average over frames (dim 3), then mask
        f1img = dropdims(mean(f1imgs[:,:,:,i], dims=3), dims=3)
        f1img .*= isfinite.(f1img) .& (f1img .>= -10) .& (f1img .<= 1000)

        f2img = dropdims(mean(f2imgs[:,:,:,i], dims=3), dims=3)
        f2img .*= isfinite.(f2img) .& (f2img .>=  -5) .& (f2img .<= 500)

        f1_images[i]   = f1img
        f2_images[i]   = f2img

        # Profile: average over x (dim 1) after frame-averaging and masking
        f1_profiles[i] = vec(mean(f1img, dims=1))
        f2_profiles[i] = vec(mean(f2img, dims=1))

    end

    # ── Per-current panels: heatmap (left) + z-profile (right) ───────────────
    for i in eachindex(SG0_current)
        f1vmax = Statistics.quantile(vec(f1_images[i]), 0.999)
        f2vmax = Statistics.quantile(vec(f2_images[i]), 0.999)

        plt1 = heatmap(f1_images[i];
            xlabel = L"$z\ \ (\mathrm{px})$",
            ylabel = L"$x\ \ (\mathrm{px})$",
            cbar   = true, clims = (0, f1vmax),
        )
        plt2 = plot(f1_profiles[i];
            line = (:solid, 1, colors_sg0[i]), label = current_label(i),
            xlabel = L"$z\ \ (\mathrm{px})$", legend_kw...,
        )
        plt3 = heatmap(f2_images[i];
            xlabel = L"$z\ \ (\mathrm{px})$",
            ylabel = L"$x\ \ (\mathrm{px})$",
            cbar   = true, clims = (0, f2vmax),
        )
        plt4 = plot(f2_profiles[i];
            line = (:solid, 1, colors_sg0[i]), label = current_label(i),
            xlabel = L"$z\ \ (\mathrm{px})$", legend_kw...,
        )

        plt = plot(plt1, plt2, plt3, plt4;
            suptitle      = sg1_label,
            layout        = (2, 2),
            link          = :x,
            size          = (800, 450),
            left_margin   = 3mm,
            bottom_margin = 2mm,
        )
        plot!(plt[1]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
        plot!(plt[2]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
        display(plt)
    end

    # ── Overlay: all currents on one z-profile plot per state ─────────────────
    overlay_kw = (
        legend                  = :topleft,
        legend_font             = 8,
        foreground_color_legend = nothing,
        legend_columns          = 2,
        xlabel                  = L"$z \ \ (\mathrm{px})$",
    );

    plt_f1 = plot(; overlay_kw...);
    plt_f2 = plot(; overlay_kw...);

    for i in eachindex(SG0_current)
        plot!(plt_f1, f1_profiles[i]; line=(:solid, 1, colors_sg0[i]), label=current_label(i))
        plot!(plt_f2, f2_profiles[i]; line=(:solid, 1, colors_sg0[i]), label=current_label(i))
    end
    
    plot!(plt_f1; legend_title=L"$F=1$");
    plot!(plt_f2; legend_title=L"$F=2$");

    fig = plot(plt_f1, plt_f2;
        suptitle      = sg1_label,
        layout        = (2, 1),
        size          = (900, 850),
        left_margin   = 3mm,
        bottom_margin = 2mm,
    )
    plot!(fig[1]; xlabel="", xformatter=_->"", bottom_margin=-6mm)
    display(fig)

    GC.gc()
end


mark_symbol = [:circle, :rect]
colores = (:darkgreen, :indigo )
current_polarity = ("+","–")

fig1a = plot(xlabel="SG0 Current (A)",
    ylabel=L"$\Delta z$ (px)");
for (i,dir) in enumerate(data_directories[1:2])
    data = tables[i]
    data_pos = data[data.I0 .> 0, :]
    plot!(fig1a,
            data_pos.I0, data_pos.split/6.5e-3,
            yerror = data_pos.errsplit/6.5e-3,
            label="($(current_polarity[i])) "*dir,
            line=(:solid, 1, colores[i],0.5),
            marker=(mark_symbol[i],2,:white),
            markerstrokecolor=colores[i]
    )
end
plot!(fig1a,
    title = L"SG1 $\sim 0.0\mathrm{mA}$",
    xscale=:log10,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=false,)


fig1b=plot(
    xlabel="SG0 Current (mA)",
    ylabel=L"$\Delta z$ (px)");
for (i,dir) in enumerate(data_directories[1:2])
    data = tables[i]
    data_pos = data[data.I0 .< 0.050, :]
    plot!(fig1b,
            1000*data_pos.I0, data_pos.split/6.5e-3,
            label=false,
            line=(:dashdot, 1, colores[i],0.5),
    )
    plot!(fig1b,
            1000*data_pos.I0, data_pos.split/6.5e-3,
            yerror = data_pos.errsplit/6.5e-3,
            seriestype=:scatter,
            label="($(current_polarity[i])) "*dir,
            marker=(mark_symbol[i],2,:white),
            lw=2,
            markerstrokecolor=colores[i]
    )
end
plot!(fig1b,
    title = L"SG1 $\sim 0\mathrm{mA}$",
    xlims=(-1,10),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=5,)

fig1c=plot(
    xlabel="SG0 Current (mA)",
    ylabel=L"$\Delta z$ (px)");
for (i,dir) in enumerate(data_directories[1:2])
    data = tables[i]
    data_pos = data
    plot!(fig1c,
            1000*data_pos.I0, data_pos.split/6.5e-3,
            label=false,
            line=(:dashdot, 1, colores[i],0.5),
    )
    plot!(fig1c,
            1000*data_pos.I0, data_pos.split/6.5e-3,
            yerror = data_pos.errsplit/6.5e-3,
            seriestype=:scatter,
            label="($(current_polarity[i])) "*dir,
            marker=(mark_symbol[i],2,:white),
            lw=2,
            markerstrokecolor=colores[i]
    )
end
plot!(fig1c,
    title = L"SG1 $\sim 0\mathrm{mA}$",
    xlims=(-1,10),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=5,)


fig1 = plot(fig1a,fig1b,fig1c,
    layout=(1,3),
    size=(900,400),
    link=:y,
    left_margin=3mm,
    bottom_margin=3mm,);
plot!(fig1[2], ylabel="", yformatter=_->"", left_margin=-4mm);
display(fig1)


fig2a = plot(
    xlabel="SG0 Current (A)",
    ylabel=L"$\Delta z$ (px)",
    title = L"SG1 $\sim 40\mathrm{mA}$",
);
for (i,dir) in enumerate(data_directories[3:4])
    data = tables[i+2]
    data_pos = data[data.I0 .> 0, :]
    plot!(fig2a,
            data_pos.I0, data_pos.split/6.5e-3,
            yerror = data_pos.errsplit/6.5e-3,
            label="($(current_polarity[i])) "*dir,
            line=(:solid, 1, colores[i],0.5),
            marker=(mark_symbol[i],2,:white),
            markerstrokecolor=colores[i]
    )
end
plot!(fig2a,
    xscale=:log10,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=false,
);


fig2b=plot(
    xlabel="SG0 Current (mA)",
    ylabel=L"$\Delta z$ (px)");
for (i,dir) in enumerate(data_directories[3:4])
    data = tables[i+2]
    data_pos = data[(data.I0 .< 0.050) .& (data.I1 .> 0), :]
    plot!(fig2b,
            1000*data_pos.I0, data_pos.split/6.5e-3,
            label=false,
            line=(:dashdot, 1, colores[i],0.5),
    )
    plot!(fig2b,
            1000*data_pos.I0, data_pos.split/6.5e-3,
            yerror = data_pos.errsplit/6.5e-3,
            seriestype=:scatter,
            label="($(current_polarity[i])) "*dir,
            marker=(mark_symbol[i],2,:white),
            lw=2,
            markerstrokecolor=colores[i]
    )
end
plot!(fig2b,
    title = L"SG1 $\sim 40\mathrm{mA}$",
    xlims=(-1,10),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=5,)

fig2c=plot(
    xlabel="SG0 Current (mA)",
    ylabel=L"$\Delta z$ (px)");
for (i,dir) in enumerate(data_directories[3:4])
    data = tables[i+2]
    data_pos = data
    plot!(fig2c,
            1000*data_pos.I0, data_pos.split/6.5e-3,
            label=false,
            line=(:dashdot, 1, colores[i],0.5),
    )
    plot!(fig2c,
            1000*data_pos.I0, data_pos.split/6.5e-3,
            yerror = data_pos.errsplit/6.5e-3,
            seriestype=:scatter,
            label="($(current_polarity[i])) "*dir,
            marker=(mark_symbol[i],2,:white),
            lw=2,
            markerstrokecolor=colores[i]
    )
end
plot!(fig2c,
    title = L"SG1 $\sim 40\mathrm{mA}$",
    xlims=(-1,10),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=5,)

fig2 = plot(fig2a,fig2b,fig2c,
    layout=(1,3),
    size=(950,400),
    link=:y,
    left_margin=3mm,
    bottom_margin=4mm,);
plot!(fig2[2], ylabel="", yformatter=_->"", left_margin=-4mm);
display(fig2)



fig3a = plot(
    xlabel="SG0 Current (A)",
    ylabel=L"$\Delta z$ (px)",
    title = L"SG1 $\sim 500\mathrm{mA}$",
);
for (i,dir) in enumerate(data_directories[5:6])
    data = tables[i+4]
    data_pos = data[data.I0 .> 0, :]
    plot!(fig3a,
            data_pos.I0, data_pos.split/6.5e-3,
            yerror = data_pos.errsplit/6.5e-3,
            label="($(current_polarity[i])) "*dir,
            line=(:solid, 1, colores[i],0.5),
            marker=(mark_symbol[i],2,:white),
            markerstrokecolor=colores[i]
    )
end
plot!(fig3a,
    xscale=:log10,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=5,
)


fig3b=plot(
    xlabel="SG0 Current (mA)",
    ylabel=L"$\Delta z$ (px)");
for (i,dir) in enumerate(data_directories[5:6])
    data = tables[i+4]
    data_pos = data[(data.I0 .< 0.050) .& (data.I1 .> 0), :]
    plot!(fig3b,
            1000*data_pos.I0, data_pos.split/6.5e-3,
            label=false,
            line=(:dashdot, 1, colores[i],0.5),
    )
    plot!(fig3b,
            1000*data_pos.I0, data_pos.split/6.5e-3,
            yerror = data_pos.errsplit/6.5e-3,
            seriestype=:scatter,
            label="($(current_polarity[i])) "*dir,
            marker=(mark_symbol[i],2,:white),
            lw=2,
            markerstrokecolor=colores[i]
    )
end
plot!(fig3b,
    title = L"SG1 $\sim 500\mathrm{mA}$",
    xlims=(-1,10),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=5,)


fig3c=plot(
    xlabel="SG0 Current (mA)",
    ylabel=L"$\Delta z$ (px)");
for (i,dir) in enumerate(data_directories[5:6])
    data = tables[i+4]
    data_pos = data[3:end,:]
    plot!(fig3c,
            1000*data_pos.I0, -data_pos.split/6.5e-3,
            label=false,
            line=(:dashdot, 1, colores[i],0.5),
    )
    plot!(fig3c,
            1000*data_pos.I0, -data_pos.split/6.5e-3,
            yerror = data_pos.errsplit/6.5e-3,
            seriestype=:scatter,
            label="($(current_polarity[i])) "*dir,
            marker=(mark_symbol[i],2,:white),
            lw=2,
            markerstrokecolor=colores[i]
    )
end
plot!(fig3c,
    title = L"SG1 $\sim 500\mathrm{mA}$",
    # xlims=(-1,10),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=5,)

fig3 = plot(fig3a,fig3b,fig3c,
    layout=(1,3),
    size=(950,400),
    link=:y,
    left_margin=3mm,
    bottom_margin=4mm,);
plot!(fig3[2], ylabel="", yformatter=_->"", left_margin=-4mm);
display(fig3)















plot(xlabel="SG0 Current (A)",
    ylabel=L"$\Delta z$ (px)",
    title = L"SG1 $\sim 500\mathrm{mA}$",);
for (i,dir) in enumerate(data_directories[5:6])
    data = tables[i+4]
    data_pos = data[data.I0 .> 0, :]
    plot!(data_pos.I0, data_pos.split/6.5e-3,
            yerror = data_pos.errsplit/6.5e-3,
            label="($(current_polarity[i])) "*dir,
            line=(:solid, 1, colores[i],0.5),
            marker=(mark_symbol[i],2,:white),
            markerstrokecolor=colores[i]
    )
end
plot!(
    xscale=:log10,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=false,)




mark_symbol = (:circle, :rect, :diamond, :utriangle )
colores = ((:red,:blue), (:orangered, :dodgerblue), (:darkgreen, :indigo) , (:sienna, :lime))
plot(xlabel="SG0 Current (A)",
    ylabel="position (mm)")
for (i,dir) in enumerate(data_directories)
    data = tables[i]
    data_pos = data[data.I0 .>= 0, :]
    plot!(data_pos.I0, data_pos.zf1, 
        yerror = data_pos.errzf1,
        label=L"$F=1$",
        seriestype=:scatter,
        marker=(mark_symbol[i],2,:white),
        markerstrokecolor=colores[i][1])
    plot!(data_pos.I0, data_pos.zf2,
        yerror = data_pos.errzf2,
        label=L"$F=2$",
        seriestype=:scatter,
        marker=(mark_symbol[i],2,:white),
        markerstrokecolor=colores[i][2])
    plot!(data_pos.I0, mean([data_pos.zf1,data_pos.zf2]),
        label="Centre",
        marker=(:xcross,:gray47),
        line=(:dash,1,:gray47)  
    )
end
plot!(
    # xscale=:log10,
    yformatter = y -> @sprintf("%.3f", y),
    legend=:outerright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=false,)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_directory = "20260415A"

OUTDIR    = joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

outfile_raw         = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data.jld2")
outfile_processed   = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2")
data_summary_path   = joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS","summary",data_directory)
isdir(data_summary_path) || mkpath(data_summary_path);


if !isfile(outfile_processed) # check if the processed images exists
    if !isfile(outfile_raw) # check if the raw data exists
        @info "Not found → building $outfile_raw"
        data_in = joinpath(BASE_PATH,"EXPERIMENTS", data_directory)
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

data = hcat(SG0_current,SG1_current, f1_z_mm, f1_z_sem_mm, f2_z_mm, f2_z_sem_mm)
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
    legend=:bottomleft,
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
    ylims=(5e-2,1e-1),
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
    ylims=(5e-2,1e-1),
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

OUTDIR    = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

outfile_raw         = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data.jld2")
outfile_processed   = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2")
data_summary_path   = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS","summary",data_directory)
isdir(data_summary_path) || mkpath(data_summary_path);


if !isfile(outfile_processed) # check if the processed images exists
    if !isfile(outfile_raw) # check if the raw data exists
        @info "Not found → building $outfile_raw"
        data_in = joinpath(BASE_PATH,"EXPERIMENTS", data_directory)
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


data = hcat(SG0_current,SG1_current, f1_z_mm, f1_z_sem_mm, f2_z_mm, f2_z_sem_mm)
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

OUTDIR    = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

outfile_raw         = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data.jld2")
outfile_processed   = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2")
data_summary_path   = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS","summary",data_directory)
isdir(data_summary_path) || mkpath(data_summary_path);


if !isfile(outfile_processed) # check if the processed images exists
    if !isfile(outfile_raw) # check if the raw data exists
        @info "Not found → building $outfile_raw"
        data_in = joinpath(BASE_PATH,"EXPERIMENTS", data_directory)
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


data = hcat(SG0_current,SG1_current, f1_z_mm, f1_z_sem_mm, f2_z_mm, f2_z_sem_mm)
cols = palette(:darkrainbow, size(data,1))   # generate colors

dataA = sortslices(data; dims=1, by = r -> (r[1], r[2]))[1:8,:]
dataB = sortslices(data; dims=1, by = r -> (r[2], r[1]))[9:end,:]



data01 = sortslices(dataA; dims=1, by = r -> (r[1], r[2]))
zcenter = data01[1,3]
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
    xlims=(-0.1,0.1),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig01)

data02 = sortslices(dataA; dims=1, by = r -> (r[1], r[2]))
zcenter = data02[1,3]
data02[:,3] = (data02[:,3] .- zcenter)
data02 = data02[2:end, :]
fig02 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data02,1)
    plot!(fig02,
        [1000*data02[i,2]], [abs.(data02[i,3])],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        # yerror=[data03[i,4]],
        label=L"$I_{c0} = %$(round(data02[i,1];digits=3))\mathrm{A}$")
end
plot!(fig02,
    yscale=:log10,
    ylims=(1e-3,1e-2),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig02)

data03 = sortslices(dataB; dims=1, by = r -> (r[2], r[1]))
zcenter = dataA[1,3]
fig03 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
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
    # xlims=(-0.1,0.1),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig03)

data04 = sortslices(dataB; dims=1, by = r -> (r[2], r[1]))
zcenter = dataA[1,3]
data04[:,3] = (data04[:,3] .- zcenter)
fig04 = plot(xlabel="Current SG1 (A)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data04,1)
    plot!(fig04,
        [data04[i,2]], [abs.(data04[i,3])],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        # yerror=[data03[i,4]],
        label=L"$I_{c0} = %$(round(data04[i,1];digits=3))\mathrm{A}$")
end
plot!(fig04,
    xscale=:identity,
    yscale=:log10,
    # ylims=(1e-3,1e-2),
    # yticks = ([1e-3,1e-2, 1e-1], 
    #         [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig04)


p1 = plot(fig01,fig02, fig03, fig04,
layout=@layout([a  b ; c  d ]),
size=(900,800)
)
saveplot(p1,"zvssg1")


data04 = sortslices(dataB; dims=1, by = r -> (r[1], r[2]))
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

