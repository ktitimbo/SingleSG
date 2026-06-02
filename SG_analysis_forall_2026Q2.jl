# Kelvin Titimbo
# California Institute of Technology
# June 2026

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
# Custom modules
include("./Modules/JLD2_MyTools.jl");
include("./Modules/MyExperimentalAnalysis.jl");
using .MyExperimentalAnalysis;
# Set the working directory to the current location
cd(@__DIR__) ;
# General setup
const BASE_PATH = raw"F:\SternGerlachExperiments";
hostname = gethostname();
@info "Running on host" hostname=hostname
# For Plots
FIG_EXT  = "png"   # could be "pdf", "svg", etc.
SAVE_FIG = false
MyExperimentalAnalysis.SAVE_FIG = SAVE_FIG;
MyExperimentalAnalysis.FIG_EXT  = FIG_EXT;

# Data Directory
data_directories =  ["20260529"]
data_directory      = data_directories[1] ;
# Furnace 
TCelsius = 205
const Temperature = 273.15 + TCelsius
# Blurring (gaussian) width
σw_um = 0.200

outfile_raw         = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data.jld2")
outfile_processed   = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2")
data_summary_path   = joinpath(BASE_PATH, "EXPDATA_ANALYSIS","summary",data_directory)
isdir(data_summary_path) || mkpath(data_summary_path);


# STERN–GERLACH EXPERIMENT SETUP
# Camera and pixel geometry : intrinsic properties
CAM_PIXELSIZE           = 6.5e-6 ;  # Physical pixel size of camera [m]
NX_PIXELS , NZ_PIXELS   = (2160, 2560); # (Nx,Nz) pixels
magnification_factor    = mag_factor(data_directory)[1] ;
# Experiment resolution
EXP_BIN_X, EXP_BIN_Z    = (4,1) ;  # Camera binning
EXP_PIXELSIZE_X, EXP_PIXELSIZE_Z = (EXP_BIN_X, EXP_BIN_Z).*CAM_PIXELSIZE ; # Effective pixel size after binning [m]
# Image dimensions (adjusted for binning)
x_pixels = Int(NX_PIXELS / EXP_BIN_X);  # Number of x-pixels after binning
z_pixels = Int(NZ_PIXELS / EXP_BIN_Z);  # Number of z-pixels after binning
# Spatial axes shifted to center the pixels
x_position = pixel_positions(x_pixels, 1, EXP_PIXELSIZE_X);
z_position = pixel_positions(z_pixels, 1, EXP_PIXELSIZE_Z);
println("""
***************************************************
CAMERA FEATURES
    Number of pixels        : $(NX_PIXELS) × $(NZ_PIXELS)
    Pixel size              : $(1e6*CAM_PIXELSIZE) μm

IMAGES INFORMATION
    Magnification factor    : $magnification_factor
    Binning                 : $(EXP_BIN_X) × $(EXP_BIN_Z)
    Effective pixels        : $(x_pixels) × $(z_pixels)
    Pixel size              : $(1e6*EXP_PIXELSIZE_X)μm × $(1e6*EXP_PIXELSIZE_Z)μm
    xlims                   : ($(round(minimum(1e6*x_position), digits=6)) μm, $(round(maximum(1e3*x_position), digits=4)) mm)
    zlims                   : ($(round(minimum(1e6*z_position), digits=6)) μm, $(round(maximum(1e3*z_position), digits=4)) mm)
***************************************************
""")
# Setting the variables for the module
MyExperimentalAnalysis.effective_cam_pixelsize_z = EXP_PIXELSIZE_Z;
MyExperimentalAnalysis.default_x_pixels          = x_pixels;
MyExperimentalAnalysis.default_z_pixels          = z_pixels;

# Previous experiment data for comparison
data_JSF = OrderedDict(
    :exp => hcat(
    [0.0200, 0.0300, 0.0500, 0.1500, 0.2000, 0.2500, 0.3500, 0.5000, 0.7500], #A
    [0.0229, 0.0610, 0.1107, 0.3901, 0.5122, 0.6315, 0.8139, 1.1201, 1.5738]),
    :model => hcat(
    [0.0150, 0.0200, 0.0250, 0.0300, 0.0400, 0.0500, 0.0700, 0.1000, 0.1500, 0.2000, 0.2500, 0.3500, 0.5000, 0.7500], #A
    [0.0409, 0.0566, 0.0830, 0.1015, 0.1478, 0.1758, 0.2409, 0.3203, 0.4388, 0.5433, 0.6423, 0.8394, 1.1267, 1.5288], #CQD
    [0.0179, 0.0233, 0.0409, 0.0536, 0.0883, 0.1095, 0.1713, 0.2487, 0.3697, 0.4765, 0.5786, 0.7757, 1.0655, 1.4630]) #QM
);

data_qm_f1_path = joinpath(BASE_PATH,"SIMULATIONS","2026Q2_SETUP","QM_T$(TCelsius)_8M","qm_screen_profiles_f1_table.jld2")
data_qm_f2_path = joinpath(BASE_PATH,"SIMULATIONS","2026Q2_SETUP","QM_T$(TCelsius)_8M","qm_screen_profiles_f2_table.jld2")

# Importing data
if isfile(outfile_processed)
    @info "Found $outfile_processed → skipping build"
else
    # Build raw file if needed
    if isfile(outfile_raw)
        @info "Found $outfile_raw → skipping build"
    else
        @info "Not found → building $outfile_raw"
        data_in = joinpath(BASE_PATH, "EXPERIMENTS", data_directory)
        data_raw = MyExperimentalAnalysis.SG0_stack_data("2026",data_in; order=:asc)
        jldsave(outfile_raw, data=data_raw)
        data_raw = nothing
    end

    # Dark & flat frames (load, bin, plot, save)
    img_dk = bin_x_mean(matread(joinpath(BASE_PATH, "EXPERIMENTS", "img_dk.mat"))["DKMean"], EXP_BIN_X);
    img_fl = bin_x_mean(matread(joinpath(BASE_PATH, "EXPERIMENTS", "img_fl.mat"))["FLMean"], EXP_BIN_X);

    p1 = heatmap(1e3*z_position, 1e3*x_position, img_dk;
        title=L"$\langle$ Dark Frame $\rangle$",
        xlabel=L"$z\ \mathrm{(mm)}$", ylabel=L"$x\ \mathrm{(mm)}$")
    p2 = heatmap(1e3*z_position, 1e3*x_position, img_fl;
        title=L"$\langle$ Flat Frame $\rangle$",
        xlabel=L"$z\ \mathrm{(mm)}$", ylabel=L"$x\ \mathrm{(mm)}$")
    savefig(
        plot(p1, p2; layout=(1,2), link=:both, size=(1000,400), left_margin=4mm, bottom_margin=3mm),
        joinpath(BASE_PATH, "EXPERIMENTS", "dark_flat.$FIG_EXT")
    )

    # Process and save
    data_raw = load(outfile_raw)["data"];
    MyExperimentalAnalysis.SG0_process_and_save(data_raw, outfile_processed; mode=:simple);
    @info "data ready"
end

jldopen(outfile_processed, "r") do file
    if haskey(file["meta"], "SG1BfieldInTesla")
        fig = plot(abs.(file["meta"]["SG1currentInA"]), file["meta"]["SG1BfieldInTesla"];
            seriestype=:scatter,
            label=false,
            marker=(:circle,2,:white),
            markerstrokecolor=:blue,
            xlabel="Coil current (A)",
            ylabel="Magnetic field (T)"
            )
        saveplot(fig, joinpath(BASE_PATH,"EXPERIMENTS",data_directory,"mag_field_exp.$(FIG_EXT)"))
        display(fig)
    end

    global Iexp_coil  = file["meta"]["SG1currentInA"]

end
nI      = length(Iexp_coil)
ΔIexp_coil = 0.01 * Iexp_coil
sort_perm = sortperm(Iexp_coil)

fig_I0 = plot(Iexp_coil, 
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
    label = data_directory,
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
display(fig_I0) 
saveplot(fig_I0, joinpath(BASE_PATH,"EXPERIMENTS",data_directory,"current_range.$(FIG_EXT)"))


# Binning for the analysis
nbins_list  = (1, 2, 4)
λ0_list     = (0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10)
PARAM_GRID = [(nz, λ0) for λ0 in λ0_list, nz in nbins_list] |> vec
const Cell = Union{Missing, String, Int, Float64}
summary_table = Matrix{Cell}(undef, length(PARAM_GRID), 3)
# ── Shared plot attributes ─────────────────────────────────────────────
log_xaxis  = (:log10, L"$I_{c} \ (\mathrm{A})$", :log10);
log_xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]);
grid_attrs = (grid=true, minorgrid=true, gridalpha=0.5, gridstyle=:dot,
            minorgridalpha=0.05, framestyle=:box);
scatter_attrs = (seriestype=:scatter, marker=(:circle,:white,4),
                markerstrokecolor=:black, markerstrokewidth=2);
base_plot_attrs = (size=(800,600), tickfontsize=11, guidefontsize=14, legendfontsize=10);
log_axis_attrs = (;
    xaxis=log_xaxis, xticks=log_xticks, xlims=(1e-3,1.0))
jldopen(joinpath(data_summary_path, data_directory * "_report_summary.jld2"), "w") do file
    file["meta/Currents"]      = Iexp_coil
    file["meta/ErrorCurrents"] = ΔIexp_coil
    file["meta/n_Currents"]    = nI
    file["meta/TemperatureK"]  = Temperature
    file["meta/nz"]            = nbins_list
    file["meta/λ0"]            = λ0_list

    for (row, (nz,λ0)) in enumerate(PARAM_GRID)
        T_START   = Dates.now()
        RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
        OUTDIR    = joinpath(BASE_PATH, "EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
        isdir(OUTDIR) || mkpath(OUTDIR);
        @info "Created output directory" OUTDIR
        MyExperimentalAnalysis.OUTDIR   = OUTDIR;
        summary_table[row,:] = Cell[RUN_STAMP, nz, λ0]

        chosen_qm =  jldopen(data_qm_f1_path,"r") do file
            file[JLD2_MyTools.make_keypath_qm(nz, σw_um, λ0)]
        end
        Ic_QM_sim = [chosen_qm[i][:Icoil] for i in eachindex(chosen_qm)][2:end]
        zm_QM_sim = [chosen_qm[i][:z_max_smooth_spline_mm] for i in eachindex(chosen_qm)][2:end]

        # position
        z_mm        = 1e3 .* pixel_positions(z_pixels, nz, EXP_PIXELSIZE_Z)
        z_mm_error  = 1e3 / sqrt(12) * EXP_PIXELSIZE_Z * nz # assuming uniform distribution within the bin/pixel size

        # --- Compute F1 / F2 mean profiles ----------------------------------------
        profiles_F1 = extract_profiles(outfile_processed, :F1ProcessedImages, nI, z_pixels; n_bin=nz, with_error=true);
        profiles_F2 = extract_profiles(outfile_processed, :F2ProcessedImages, nI, z_pixels; n_bin=nz, with_error=true);

        # --- Plot ------------------------------------------------------------------
        fig1 = plot_profiles(z_mm, profiles_F1, Iexp_coil; title="F1 processed data")
        fig2 = plot_profiles(z_mm, profiles_F2, Iexp_coil; title="F2 processed data")
        fig = plot(fig1, fig2;
            layout=(2,1),
            size=(1000,600),
            share=:both,
            legend_columns=3,
            left_margin=5mm,
        )
        display(fig)
        saveplot(fig, "mean_profiles_processed")

        ########################################################################
        ############# MEAN ANALYSIS ###########################################
        ########################################################################
        f1_mean_max , f1_mean_spline_profile = my_process_mean_maxima( outfile_processed, "F1", nz; half_max=true, λ0=λ0);
        f2_mean_max , f2_mean_spline_profile = my_process_mean_maxima( outfile_processed, "F2", nz; half_max=true, λ0=λ0);

        # ── Centroid ──────────────────────────────────────────────────────────
        data_centroid_mean       = 0.5 .* (f1_mean_max .+ f2_mean_max)
        data_centroid_mean_error = fill(0.5 * sqrt(2) * z_mm_error, length(data_centroid_mean))
        centroid_mean = post_threshold_mean(data_centroid_mean, Iexp_coil, data_centroid_mean_error;
                            threshold=0.020, half_life=5, eps=1e-6, weighted=true)
        saveplot(centroid_mean.plot, "mean_centroid_diagnose")

        # ── DataFrame ─────────────────────────────────────────────────────────
        df_mean = sort!(DataFrame(
            Icoil_A              = Iexp_coil,
            Icoil_error_A        = ΔIexp_coil,
            F1_z_peak_mm         = f1_mean_max,
            F2_z_peak_mm         = f2_mean_max,
            Δz_mm                = f1_mean_max .- f2_mean_max,
            F1_z_centroid_mm     = f1_mean_max .- centroid_mean.mean,
            F1_z_centroid_mm_sem = sqrt.(z_mm_error^2 .+ centroid_mean.sem^2),
            F2_z_centroid_mm     = f2_mean_max .- centroid_mean.mean,
            F2_z_centroid_mm_sem = sqrt.(z_mm_error^2 .+ centroid_mean.sem^2),
        ), :Icoil_A)
        CSV.write(joinpath(OUTDIR, "mean_data.csv"), df_mean)

        # ── Save profiles ──────────────────────────────────────────────────────
        jldsave(joinpath(OUTDIR, "profiles_mean.jld2"),
            profiles = OrderedDict(
                :Icoils      => Iexp_coil,
                :Icoils_err  => ΔIexp_coil,
                :Centroid_mm => (centroid_mean.mean, centroid_mean.sem),
                :z_mm        => z_mm,
                :F1_profile  => profiles_F1.mean,
                :F1_err      => profiles_F1.sem,
                :F2_profile  => profiles_F2.mean,
                :F2_err      => profiles_F2.sem,
            )
        )

        # ── Pretty table ───────────────────────────────────────────────────────
        pretty_table(df_mean;
            title        = "Mean analysis",
            formatters   = [fmt__printf("%8.3f", [1]), fmt__printf("%8.5f", 3:9)],
            alignment    = :c,
            column_labels = [
                ["Current","Current Error","F1 z","F2 z","Δz","Centroid F1 z","Centroid F1 z Error","Centroid F2 z","Centroid F2 z Error"],
                ["[A]","[A]","[mm]","[mm]","[mm]","[mm]","[mm]","[mm]","[mm]"]
            ],
            table_format = TextTableFormat(borders=text_table_borders__unicode_rounded),
            style = TextTableStyle(
                first_line_column_label = crayon"yellow bold",
                column_label            = crayon"yellow",
                table_border            = crayon"blue bold"),
            equal_data_column_widths = true,
            highlighters = [
                TextHighlighter((d,i,j) -> d[i,1] == minimum(d[:,1]), crayon"fg:white bold bg:dark_gray"),
                TextHighlighter((d,i,j) -> d[i,6] < 0,                crayon"fg:red bold bg:dark_gray"),
                TextHighlighter((d,i,j) -> d[i,8] > 0,                crayon"fg:green bold bg:dark_gray"),
            ],
        )

        # ── Centroid figure ────────────────────────────────────────────────────
        fig_centroid = plot(Iexp_coil, data_centroid_mean;
            yerror           = data_centroid_mean_error,
            label            = false,
            color            = :purple,
            marker           = (:circle,3),
            markerstrokecolor = :purple,
            line             = (:solid,1),
            xlim             = (1e-3,1),
            yaxis            = L"$z_{0} \ (\mathrm{mm})$",
            title            = "Centroid",
            legend           = :topleft,
        )
        plot!(fig_centroid,
            xaxis            = log_xaxis,
            xticks           = log_xticks,
        )
        hline!(fig_centroid, [centroid_mean.mean];
            label = L"Centroid $z=%$(round(centroid_mean.mean, digits=3))$mm")
        hspan!(fig_centroid,
            [centroid_mean.mean - centroid_mean.sem, centroid_mean.mean + centroid_mean.sem];
            color=:orangered, alpha=0.30,
            label=L"Error = $\pm%$(round(centroid_mean.sem, digits=3))$mm")
        saveplot(fig_centroid, "mean_centroid")

        # ── Peak position figures (fig_01, fig_02, fig_03) ────────────────────
        Ic   = abs.(df_mean[!,:Icoil_A])
        z_c  = centroid_mean.mean
        F1   = df_mean[!,:F1_z_peak_mm]
        F2   = df_mean[!,:F2_z_peak_mm]
        F1c  = df_mean[!,:F1_z_centroid_mm]
        F2c  = df_mean[!,:F2_z_centroid_mm]

        peak_plot_attrs = (;
            yaxis=L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
            legend=:topleft,
            grid_attrs..., base_plot_attrs...)

        fig_01 = plot(Ic, F1; label=L"$F_{1}$", line=(:solid,:red,2), peak_plot_attrs..., title="Peak position")
        plot!(fig_01, Ic, F2;  label=L"$F_{2}$",  line=(:solid,:blue,2))
        plot!(fig_01, Ic, F1;  fillrange=F2,       fillalpha=0.2, color=:purple, label=false)
        hline!(fig_01, [z_c];  line=(:dot,:black,2), label="Centroid")
        plot!(fig_01; log_axis_attrs...)   # apply log scale last

        fig_02 = plot(Ic, F1;          label=L"$F_{1}$",                   line=(:solid,:red,2), peak_plot_attrs..., title="Peak position")
        plot!(fig_02, Ic, 2z_c .- F2;  label=L"Centroid Mirrored $F_{2}$", line=(:solid,:blue,2))
        plot!(fig_02, Ic, F1;          fillrange=2z_c .- F2,                fillalpha=0.2, color=:purple, label=false)
        hline!(fig_02, [z_c];          line=(:dot,:black,2), label="Centroid")
        plot!(fig_02; log_axis_attrs...)

        fig_03 = plot(Ic, F1c; label=L"$F_{1}$", line=(:solid,:red,2), peak_plot_attrs..., title="Peak position - Centered at Centroid")
        plot!(fig_03, Ic, F2c; label=L"$F_{2}$", line=(:solid,:blue,2))
        plot!(fig_03, Ic, F1c; fillrange=F2c,    fillalpha=0.2, color=:purple, label=false)
        plot!(fig_03; log_axis_attrs...)

        fig_peaks = plot(fig_01, fig_02, fig_03; layout=@layout([a; b; c]), share=:x)
        plot!(fig_peaks[1], xlabel="", xformatter=_->"")
        plot!(fig_peaks[2], xlabel="", xformatter=_->"", title="", top_margin=-9mm)
        plot!(fig_peaks[3], title="", top_margin=-9mm)
        display(fig_peaks)
        saveplot(fig_peaks, "mean_peak_centroid")

        # ── Absolute centroid figures (log/linear pairs) ───────────────────────
        neg_mask = F1c .< 0
        pos_mask = .!neg_mask
        y_abs    = abs.(F1c) ./ magnification_factor
        y_err    = df_mean[!,:F1_z_centroid_mm_sem]
        x_err    = df_mean[!,:Icoil_error_A]
        pix_span = [1e-6, 1000 * nz * EXP_PIXELSIZE_Z]

        _add_reference_overlays! = let pix_span = pix_span, Ic_QM_sim = Ic_QM_sim, zm_QM_sim = zm_QM_sim
            function(fig)
                hspan!(fig, pix_span; color=:gray, alpha=0.30, label="Effective pixel size")
                plot!(fig, data_JSF[:exp][:,1], data_JSF[:exp][:,2];
                    marker=(:cross,:purple,6), line=(:purple,:dash,2,0.5),
                    markerstrokewidth=2, label="10142024")
                plot!(fig, Ic_QM_sim, zm_QM_sim;
                    line=(:dash,:darkgreen,2.5),
                    label=L"Analytic QM ($\sigma_{w}=%$(Int(1000*σw_um))\mathrm{\mu m}$)")
            end
        end

        # Shared attrs for all-data plots (fig_000/fig_001)
        all_scatter_attrs = (;
            xerror=x_err, yerror=y_err,
            label=data_directory, scatter_attrs...,
            xaxis=L"$I_{c} \ (\mathrm{A})$",
            yaxis=L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$",
            title="F=1 Peak Position vs Current",
            legend=:bottomright, grid_attrs..., base_plot_attrs...)

        # Shared attrs for sign-masked plots (fig_100/fig_101)
        abs_plot_base = (;
            xerror=x_err[pos_mask], yerror=y_err[pos_mask],
            label=data_directory, scatter_attrs...,
            xaxis=L"$I_{c} \ (\mathrm{A})$",
            yaxis=L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$",
            title="F=1 Peak Position vs Current",
            legend=:bottomright, grid_attrs..., base_plot_attrs...)

        neg_scatter_attrs = (seriestype=:scatter, marker=(:xcross,:orangered2,4),
                            markerstrokecolor=:orangered2, markerstrokewidth=2, label=false)

        # ── fig_000 / fig_001 : all data, no sign masking ─────────────────────
        fig_000 = plot(df_mean[!,:Icoil_A], y_abs; all_scatter_attrs...,
                    xlims=(0.001,1.0), ylims=(1e-4,2.5), xticks=log_xticks, yticks=:log10)
        _add_reference_overlays!(fig_000)
        plot!(fig_000, xaxis=(:log10, L"$I_{c} \ (\mathrm{A})$", :log),
                    yaxis=(:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log))
        fig_001 = plot(df_mean[!,:Icoil_A], y_abs; all_scatter_attrs...)
        _add_reference_overlays!(fig_001)

        fig_00 = plot(fig_000, fig_001; layout=(1,2), size=(1000,400), left_margin=8mm, bottom_margin=5mm)
        saveplot(fig_00, "mean_00")

        # ── fig_100 / fig_101 : sign-coded markers ────────────────────────────
        fig_100 = plot(df_mean[pos_mask,:Icoil_A], y_abs[pos_mask]; abs_plot_base...,
                    xlims=(0.001,1.0), ylims=(1e-4,2.5), xticks=log_xticks, yticks=:log10)
        plot!(fig_100, df_mean[neg_mask,:Icoil_A], y_abs[neg_mask]; neg_scatter_attrs...,
            xerror=x_err[neg_mask], yerror=y_err[neg_mask])
        _add_reference_overlays!(fig_100)
        plot!(fig_100, xaxis=(:log10, L"$I_{c} \ (\mathrm{A})$", :log),
                    yaxis=(:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log))

        fig_101 = plot(df_mean[pos_mask,:Icoil_A], y_abs[pos_mask]; abs_plot_base...)
        plot!(fig_101, df_mean[neg_mask,:Icoil_A], y_abs[neg_mask]; neg_scatter_attrs...,
            xerror=x_err[neg_mask], yerror=y_err[neg_mask])
        _add_reference_overlays!(fig_101)

        fig_1 = plot(fig_100, fig_101; layout=(1,2), size=(1000,400), left_margin=8mm, bottom_margin=5mm)
        saveplot(fig_1, "mean_01")


        ########################################################################
        ############# FRAMEWISE ANALYSIS ######################################
        ########################################################################

        f1_max = my_process_framewise_maxima(outfile_processed, "F1", nz; peak_threshold=0.1, half_max=true, λ0=λ0)
        f2_max = my_process_framewise_maxima(outfile_processed, "F2", nz; peak_threshold=0.1, half_max=true, λ0=λ0)


        # ── Per-current statistics ─────────────────────────────────────────────
        f1_z_mm, f1_z_sem_mm = vec(mean(f1_max; dims=1)),
                                sqrt.(vec(std(f1_max; dims=1, corrected=true) ./ sqrt(size(f1_max,1))).^2 .+ z_mm_error^2)
        f2_z_mm, f2_z_sem_mm = vec(mean(f2_max; dims=1)),
                                sqrt.(vec(std(f2_max; dims=1, corrected=true) ./ sqrt(size(f2_max,1))).^2 .+ z_mm_error^2)

        # ── Centroid ───────────────────────────────────────────────────────────
        data_centroid_fw       = 0.5 .* (f1_z_mm .+ f2_z_mm)
        data_centroid_fw_error = 0.5 .* sqrt.(f1_z_sem_mm.^2 .+ f2_z_sem_mm.^2)
        centroid_fw = post_threshold_mean(data_centroid_fw, Iexp_coil, data_centroid_fw_error;
                            threshold=0.020, half_life=5, eps=1e-6, weighted=true)
        saveplot(centroid_fw.plot, "fw_centroid_diagnose")

        # ── DataFrame ──────────────────────────────────────────────────────────
        res   = summarize_framewise(f1_max, f2_max, Iexp_coil, centroid_fw, z_mm_error)
        df_fw = sort!(DataFrame(
            Icoil_A             = res.Icoil_A,
            Icoil_error_A       = ΔIexp_coil,
            F1_z_peak_mm        = res.F1_z_peak_mm,
            F1_z_peak_se_mm     = res.F1_z_se_mm,
            F2_z_peak_mm        = res.F2_z_peak_mm,
            F2_z_peak_se_mm     = res.F2_z_se_mm,
            Δz_mm               = res.Δz_mm,
            Δz_se_mm            = res.Δz_se_mm,
            F1_z_centroid_mm    = res.F1_z_centroid_mm,
            F1_z_centroid_se_mm = res.F1_z_centroid_se_mm,
            F2_z_centroid_mm    = res.F2_z_centroid_mm,
            F2_z_centroid_se_mm = res.F2_z_centroid_se_mm,
        ), :Icoil_A)
        CSV.write(joinpath(OUTDIR, "fw_data.csv"), df_fw)

        # ── Pretty table ───────────────────────────────────────────────────────
        pretty_table(df_fw;
            title        = "Framewise Analysis",
            formatters   = [fmt__printf("%8.3f", [1]), fmt__printf("%8.5f", 2:6)],
            alignment    = :c,
            column_labels = [
                ["Current","Current Error","F1 z","Std.Err.","F2 z","Std.Err.","Δz","Std.Err.","Centroid F1 z","Std.Err.","Centroid F2 z","Std.Err."],
                ["[mA]","[mA]","[mm]","[mm]","[mm]","[mm]","[mm]","[mm]","[mm]","[mm]","[mm]","[mm]"]
            ],
            table_format = TextTableFormat(borders=text_table_borders__unicode_rounded),
            style = TextTableStyle(
                first_line_column_label = crayon"yellow bold",
                column_label            = crayon"yellow",
                table_border            = crayon"blue bold"),
            equal_data_column_widths = true,
            highlighters = [
                TextHighlighter((d,i,j) -> d[i,1] == minimum(d[:,1]), crayon"fg:white bold bg:dark_gray"),
                TextHighlighter((d,i,j) -> d[i,9] < 0,                crayon"fg:red bold bg:dark_gray"),
                TextHighlighter((d,i,j) -> d[i,11] > 0,               crayon"fg:green bold bg:dark_gray"),
            ],
        )


        # ── Centroid figure ────────────────────────────────────────────────────
        fig_centroid_fw = plot(Iexp_coil, data_centroid_fw;
            xerror           = ΔIexp_coil,
            yerror           = data_centroid_fw_error,
            label            = false,
            color            = :purple,
            marker           = (:circle,3),
            markerstrokecolor = :purple,
            line             = (:solid,1),
            xlim             = (1e-3,1),
            yaxis            = L"$z_{0} \ (\mathrm{mm})$",
            title            = "Centroid",
            legend           = :topleft,
        )
        hline!(fig_centroid_fw, [centroid_fw.mean];
            label=L"Centroid $z=%$(round(centroid_fw.mean, digits=3))$mm")
        hspan!(fig_centroid_fw,
            [centroid_fw.mean - centroid_fw.sem, centroid_fw.mean + centroid_fw.sem];
            color=:orangered, alpha=0.30,
            label=L"Error = $\pm%$(round(centroid_fw.sem, digits=3))$mm")
        plot!(fig_centroid_fw; xaxis=log_xaxis, xticks=log_xticks)
        saveplot(fig_centroid_fw, "fw_centroid")

        # ── Peak position figures (fig_01, fig_02, fig_03) ────────────────────
        Ic  = abs.(df_fw[!,:Icoil_A])
        z_c = centroid_fw.mean
        F1  = df_fw[!,:F1_z_peak_mm];   F1e = df_fw[!,:F1_z_peak_se_mm]
        F2  = df_fw[!,:F2_z_peak_mm];   F2e = df_fw[!,:F2_z_peak_se_mm]
        F1c = df_fw[!,:F1_z_centroid_mm]; F1ce = df_fw[!,:F1_z_centroid_se_mm]
        F2c = df_fw[!,:F2_z_centroid_mm]

        fw_peak_attrs = (;
            yaxis=L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
            legend=:topleft, grid_attrs..., base_plot_attrs...)

        fig_01 = plot(Ic, F1; ribbon=F1e, label=L"$F_{1}$", line=(:solid,:red,1),   fillalpha=0.23, fillcolor=:red,  fw_peak_attrs..., title="Peak position")
        plot!(fig_01, Ic, F2; ribbon=F2e, label=L"$F_{2}$", line=(:solid,:blue,1),  fillalpha=0.23, fillcolor=:blue)
        plot!(fig_01, Ic, F1; fillrange=F2,                  label=false,            fillalpha=0.05, color=:purple)
        hline!(fig_01, [z_c]; line=(:dot,:black,2), label="Centroid")
        plot!(fig_01; log_axis_attrs...)

        fig_02 = plot(Ic, F1;             ribbon=F1e, label=L"$F_{1}$",          line=(:solid,:red,2),  fillalpha=0.23, fillcolor=:red,  fw_peak_attrs..., title="Peak position")
        plot!(fig_02, Ic, 2z_c .- F2;    ribbon=F2e, label=L"Mirrored $F_{2}$", line=(:solid,:blue,2), fillalpha=0.23, fillcolor=:blue)
        plot!(fig_02, Ic, F1;             fillrange=2z_c .- F2,                   label=false,           fillalpha=0.2,  color=:purple)
        hline!(fig_02, [z_c]; line=(:dot,:black,2), label="Centroid")
        plot!(fig_02; log_axis_attrs...)

        fig_03 = plot(Ic, F1c; ribbon=F1ce, label=L"$F_{1}$", line=(:solid,:red,2),  fw_peak_attrs..., title="Peak position - Centered at Centroid",
                    yaxis=L"$z_{\mathrm{max}} \ (\mathrm{mm})$")
        plot!(fig_03, Ic, F2c;              label=L"$F_{2}$", line=(:solid,:blue,2))
        plot!(fig_03, Ic, F1c; fillrange=F2c, fillalpha=0.2, color=:purple, label=false)
        plot!(fig_03; log_axis_attrs...)

        fig_peaks_fw = plot(fig_01, fig_02, fig_03; layout=@layout([a; b; c]), share=:x)
        plot!(fig_peaks_fw[1], xlabel="", xformatter=_->"")
        plot!(fig_peaks_fw[2], xlabel="", xformatter=_->"", title="", top_margin=-9mm)
        plot!(fig_peaks_fw[3], title="", top_margin=-9mm)
        display(fig_peaks_fw)
        saveplot(fig_peaks_fw, "fw_peak_centroid")

        # ── Absolute centroid figures (log/linear pairs) ───────────────────────
        y_fw     = df_fw[!,:F1_z_centroid_mm] ./ magnification_factor
        y_abs_fw = abs.(y_fw)
        y_err_fw = df_fw[!,:F1_z_centroid_se_mm]
        x_err_fw = df_fw[!,:Icoil_error_A]
        neg_mask_fw = y_fw .< 0
        pos_mask_fw = .!neg_mask_fw

        fw_all_scatter = (;
            xerror=x_err_fw, yerror=y_err_fw,
            label=data_directory, scatter_attrs...,
            xaxis=L"$I_{c} \ (\mathrm{A})$",
            yaxis=L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$",
            title="F=1 Peak Position vs Current",
            legend=:bottomright, grid_attrs..., base_plot_attrs...)

        fw_pos_scatter = (;
            xerror=x_err_fw[pos_mask_fw], yerror=y_err_fw[pos_mask_fw],
            label=data_directory, scatter_attrs...,
            xaxis=L"$I_{c} \ (\mathrm{A})$",
            yaxis=L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$",
            title="F=1 Peak Position vs Current",
            legend=:bottomright, grid_attrs..., base_plot_attrs...)

        fw_neg_scatter = (seriestype=:scatter, marker=(:xcross,:orangered2,4),
                        markerstrokecolor=:orangered2, markerstrokewidth=2, label=false)

        # fig_log / fig_lin : all data
        fig_log = plot(df_fw[!,:Icoil_A], y_abs_fw; fw_all_scatter...,
                    xlims=(0.001,1.0), ylims=(1e-6,3.5), xticks=log_xticks, yticks=:log10)
        _add_reference_overlays!(fig_log)
        plot!(fig_log; xaxis=(:log10, L"$I_{c} \ (\mathrm{A})$", :log),
                    yaxis=(:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log))
        saveplot(fig_log, "fw_000")

        fig_lin = plot(df_fw[!,:Icoil_A], y_abs_fw; fw_all_scatter...)
        _add_reference_overlays!(fig_lin)

        fig_fw_00 = plot(fig_log, fig_lin; layout=(1,2), size=(1000,400), left_margin=5mm, bottom_margin=5mm)
        display(fig_fw_00)
        saveplot(fig_fw_00, "fw_00")

        # fig_10 / fig_11 : sign-coded markers
        fig_10 = plot(df_fw[pos_mask_fw,:Icoil_A], y_abs_fw[pos_mask_fw]; fw_pos_scatter...,
                    xlims=(0.001,1.0), ylims=(1e-4,2.5), xticks=log_xticks, yticks=:log10)
        plot!(fig_10, df_fw[neg_mask_fw,:Icoil_A], y_abs_fw[neg_mask_fw]; fw_neg_scatter...,
            xerror=x_err_fw[neg_mask_fw], yerror=y_err_fw[neg_mask_fw])
        _add_reference_overlays!(fig_10)
        plot!(fig_10; xaxis=(:log10, L"$I_{c} \ (\mathrm{A})$", :log),
                    yaxis=(:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log))

        fig_11 = plot(df_fw[pos_mask_fw,:Icoil_A], y_abs_fw[pos_mask_fw]; fw_pos_scatter...)
        plot!(fig_11, df_fw[neg_mask_fw,:Icoil_A], y_abs_fw[neg_mask_fw]; fw_neg_scatter...,
            xerror=x_err_fw[neg_mask_fw], yerror=y_err_fw[neg_mask_fw])
        _add_reference_overlays!(fig_11)

        fig_fw_01 = plot(fig_10, fig_11; layout=(1,2), size=(1000,400), left_margin=5mm, bottom_margin=5mm)
        display(fig_fw_01)
        saveplot(fig_fw_01, "fw_01")


        ########################################################################
        ############# COMPARISON ##############################################
        ########################################################################

        comp_scatter_base = (;
            xerror=x_err_fw, yerror=y_err_fw,
            label=data_directory, scatter_attrs...,
            legend=:bottomright, grid_attrs..., base_plot_attrs...,
            left_margin=3mm)

        log_axis_comp = (;
            xaxis=(:log10, L"$I_{c} \ (\mathrm{A})$", :log),
            yaxis=(:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
            xticks=log_xticks,
            title="F=1 Peak Position vs Current")

        # ── Zoomed (skip first point) ──────────────────────────────────────────
        fig_comp_zoom = plot(df_fw[2:end,:Icoil_A], y_abs_fw[2:end];
            comp_scatter_base...,
            xerror=x_err_fw[2:end], yerror=y_err_fw[2:end],
            ylims=(0.9e-3, 3.5), yticks=log_xticks)
        plot!(fig_comp_zoom, Ic_QM_sim, zm_QM_sim;
            line=(:dash,:darkgreen,2.5),
            label=L"Analytic QM ($\sigma_{w}=%$(Int(1000*σw_um))\mathrm{\mu m}$)")
        plot!(fig_comp_zoom; log_axis_comp...)
        saveplot(fig_comp_zoom, "comparison_zoom")

        # ── Full range ─────────────────────────────────────────────────────────
        fig_comp = plot(df_fw[!,:Icoil_A], y_abs_fw;
            comp_scatter_base...,
            xlims=(0.001,1.0), ylims=(1e-6,3.5), yticks=:log10)
        _add_reference_overlays!(fig_comp)
        plot!(fig_comp; log_axis_comp...)
        display(fig_comp)
        saveplot(fig_comp, "comparison")

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
            Run label               : $(RUN_STAMP)

        CAMERA FEATURES
            Number of pixels        : $(NX_PIXELS) × $(NZ_PIXELS)
            Pixel size              : $(1e6*CAM_PIXELSIZE) μm

        IMAGES INFORMATION
            Magnification factor    : $magnification_factor
            Camera Binning          : $(EXP_BIN_X) × $(EXP_BIN_Z)
            Effective pixels        : $(x_pixels) × $(z_pixels)
            Pixel size              : $(1e6*EXP_PIXELSIZE_X)μm × $(1e6*EXP_PIXELSIZE_Z)μm
            xlims                   : ($(round(minimum(1e6*x_position), digits=6)) μm, $(round(maximum(1e3*x_position), digits=4)) mm)
            zlims                   : ($(round(minimum(1e6*z_position), digits=6)) μm, $(round(maximum(1e3*z_position), digits=4)) mm)

        EXPERIMENT CONDITIONS
            Currents (A)            : $(Iexp_coil)
            Currents Error (A)      : $(ΔIexp_coil)
            No. of currents         : $(nI)
            Temperature (K)         : $(Temperature)

        ANALYSIS PROPERTIES
            Binning                 : $(nz)
            Smoothing parameter     : $(λ0)
            Error px size (mm)      : $(z_mm_error)
            Centroid Mean (mm)      : $(round.(centroid_mean.mean, digits=6)) ± $(round.(centroid_mean.sem, digits=6))
            Centroid FW (mm)        : $(round.(centroid_fw.mean, digits=6)) ± $(round.(centroid_fw.sem, digits=6))
            Mean F1 peak (mm)       : $(round.(df_mean[!,:F1_z_centroid_mm],digits=9))
            Framewise F1 peak (mm)  : $(round.(df_fw[!,:F1_z_centroid_mm], digits=9))
            Framewise F1 STDE (mm)  : $(round.(df_fw[!,:F1_z_centroid_se_mm], digits=9))

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

        Ic_around_0 = filter(v -> v <= 0.010, Iexp_coil)
        ni_0  = length(Ic_around_0)
        DeltaI = if ni_0 > 4
            δi, _, _, _, _ = curr_error_physical(
                Iexp_coil, ΔIexp_coil,
                collect(df_fw.F1_z_centroid_mm), collect(df_fw.F2_z_centroid_mm);
                δz1 = collect(df_fw.F1_z_centroid_se_mm),
                δz2 = collect(df_fw.F2_z_centroid_se_mm),
                use_mismatch = false,
                nfit = ni_0, order = 3,
                weight = :gaussian, h = nothing
            )
            δi
        else
            ΔIexp_coil
        end 
        @assert df_mean.Icoil_A == df_fw.Icoil_A "Electric Current mismatch"
        file[JLD2_MyTools.make_keypath_exp(data_directory, nz, λ0)] = OrderedDict(
            :RUNSTAMP               => RUN_STAMP,
            :Currents               => df_fw.Icoil_A,
            :ErrorCurrentsPhys      => DeltaI,
            :z_mm                   => z_mm,
            :z_mm_error             => z_mm_error,

            # Mean profiles
            :F1_profile             => profiles_F1.mean[sort_perm],
            :F1_err                 => profiles_F1.sem[sort_perm],
            :F1_profile_spline      => f1_mean_spline_profile[sort_perm],
            :F2_profile             => profiles_F2.mean[sort_perm],
            :F2_err                 => profiles_F2.sem[sort_perm],
            :F2_profile_spline      => f2_mean_spline_profile[sort_perm],

            # Mean analysis
            :mean_F1_peak_pos_raw   => df_mean.F1_z_peak_mm,
            :mean_F2_peak_pos_raw   => df_mean.F2_z_peak_mm,
            :centroid_mean_mm       => (centroid_mean.mean, centroid_mean.sem),
            :mean_F1_peak_pos       => (collect(df_mean.F1_z_centroid_mm), collect(df_mean.F1_z_centroid_mm_sem)),
            :mean_F2_peak_pos       => (collect(df_mean.F2_z_centroid_mm), collect(df_mean.F2_z_centroid_mm_sem)),

            # Framewise analysis
            :fw_F1_peak_pos_raw     => (collect(df_fw.F1_z_peak_mm), collect(df_fw.F1_z_peak_se_mm) ),
            :fw_F2_peak_pos_raw     => (collect(df_fw.F2_z_peak_mm), collect(df_fw.F2_z_peak_se_mm) ),
            :centroid_fw_mm         => (centroid_fw.mean, centroid_fw.sem),         
            :fw_F1_peak_pos         => (collect(df_fw.F1_z_centroid_mm), collect(df_fw.F1_z_centroid_se_mm)),
            :fw_F2_peak_pos         => (collect(df_fw.F2_z_centroid_mm), collect(df_fw.F2_z_centroid_se_mm)),
        )

        pretty_table(summary_table;
            title         = data_directory,
            alignment     = :c,
            column_labels = [ "Filename", "n" ,  "λ₀" ],
            formatters    = [fmt__printf("%d", [2]), fmt__printf("%5.3f", [3])],
            table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
            style         = TextTableStyle(first_line_column_label = crayon"yellow bold",
                            column_label  = crayon"yellow",
                            table_border  = crayon"blue bold",
                            ),
            equal_data_column_widths = true
        )

        println("Experiment analysis finished!\n\n")

    end
    
end

path = joinpath(data_summary_path, data_directory * "_report_summary.jld2")
JLD2_MyTools.show_exp_summary(path, data_directory)

println("EXPERIMENTAL ANALYSIS COMPLETED!")
alert("EXPERIMENTAL ANALYSIS COMPLETED!")

joinpath(data_summary_path, data_directory * "_report_summary.jld2")

jldopen(outfile_processed, "r") do file
    if haskey(file["meta"], "SG1BfieldInTesla")
        fig = plot(abs.(file["meta"]["SG1currentInA"]), file["meta"]["SG1BfieldInTesla"];
            seriestype=:scatter,
            label=false,
            marker=(:circle,2,:white),
            markerstrokecolor=:blue,
            xlabel="Coil current (A)",
            ylabel="Magnetic field (T)"
            )
        saveplot(fig, joinpath(BASE_PATH,"EXPERIMENTS",data_directory,"mag_field_exp.$(FIG_EXT)"))
        display(fig)
    end

    global Iexp_coil  = file["meta"]["SG1currentInA"]

end
nI      = length(Iexp_coil)


data_exp = jldopen(joinpath(data_summary_path, data_directory * "_report_summary.jld2"),"r") do file
    label = JLD2_MyTools.make_keypath_exp(data_directory,2,0.001)
    println(file["meta/Currents"])
    return file[label]
end
c0 = 0.5*(data_exp[:fw_F1_peak_pos_raw][1][1] + data_exp[:fw_F2_peak_pos_raw][1][1])

data_QM = jldopen(data_qm_f1_path, "r") do file
    file[JLD2_MyTools.make_keypath_qm(2,0.100,0.01)]
end

data_QM_old = jldopen(joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","QM_T$(TCelsius)_8M","qm_screen_profiles_f1_table.jld2"), "r") do file
    file[JLD2_MyTools.make_keypath_qm(2,0.100,0.01)]
end

plot(Iexp_coil, abs.(data_exp[:fw_F1_peak_pos_raw][1] .- c0 ), label=L"Experiment $F=1$",
    ribbon =abs.(data_exp[:fw_F1_peak_pos_raw][2])  )
plot!(Iexp_coil, abs.(data_exp[:fw_F2_peak_pos_raw][1] .- c0 ), label=L"Experiment $F=2$",
    ribbon =abs.(data_exp[:fw_F2_peak_pos_raw][2]))
plot!([data_QM[x][:Icoil] for x=1:47], [data_QM[x][:z_max_smooth_spline_mm] for x=1:47], label="QM_new" )
plot!([data_QM_old[x][:Icoil] for x=1:47], [data_QM_old[x][:z_max_smooth_spline_mm] for x=1:47], label="QM_old" )
plot!(xlims=(0.010,1.02),
        ylims=(0.005,3),
    legend=:bottomright,)
plot!(xscale=:log10)
plot!(yscale=:log10)

lastI = abs.(data_exp[:fw_F1_peak_pos_raw][1] .- c0 )[end-9:end]
mean(lastI)
std(lastI)

Icoils




data_exp[:fw_F1_peak_pos_raw][2]
data_exp[:fw_F2_peak_pos_raw]
