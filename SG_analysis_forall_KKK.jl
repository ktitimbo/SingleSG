# Kelvin Titimbo, Xukun Lin, S. Suleyman Kahraman, and Lihong V. Wang
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
# Custom modules
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

# Data Directory
data_directory      = "20260211" ;
outfile_raw         = joinpath(data_directory, "data.jld2")
outfile_processed   = joinpath(data_directory, "data_processed.jld2")

# STERN–GERLACH EXPERIMENT SETUP
# Camera and pixel geometry : intrinsic properties
cam_pixelsize           = 6.5e-6 ;  # Physical pixel size of camera [m]
nx_pixels , nz_pixels   = (2160, 2560); # (Nx,Nz) pixels
magnification_factor    = mag_factor(data_directory)[1] ;
# Experiment resolution
exp_bin_x, exp_bin_z    = (4,1) ;  # Camera binning
exp_pixelsize_x, exp_pixelsize_z = (exp_bin_x, exp_bin_z).*cam_pixelsize ; # Effective pixel size after binning [m]
# Furnace 
Temperature = 273.15+205
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
    Magnification factor    : $magnification_factor
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

data_qm_path = joinpath(@__DIR__,"simulation_data","qm_simulation_7M","qm_screen_profiles_f1_table.jld2")

# Importing data
if !isfile(outfile_processed) # check if the processed images exists
    if !isfile(outfile_raw) # check if the raw data exists
        @info "Not found → building $outfile_raw"
        data_raw = stack_data(data_directory; order=:asc, keynames=("BG","F1","F2"))
        jldsave(outfile_raw, data=data_raw)
        data_raw = nothing
    else
        @info "Found $outfile_raw → skipping build"
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
    local fig = plot(p1, p2; layout=(1,2), link=:both, size=(1000,400), 
    left_margin=4mm,
    bottom_margin=3mm)
    savefig(fig, joinpath(data_directory,"dark_flat.$(FIG_EXT)"))

    data_raw = load(outfile_raw)["data"]
    data_processed = build_processed_dict(data_raw, img_dk,img_fl)
    jldsave(outfile_processed, data=data_processed)
    data_processed = nothing
    data_raw = nothing
else
    @info "Found $outfile_processed → skipping build"
end
data_processed = load(outfile_processed)["data"]

plot(data_processed[:Currents], abs.(data_processed[:BzTesla]))

# Binning for the analysis
nbins_list  = (1, 2, 4, 8)
λ0_list     = (0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10)
const Cell = Union{Missing, String, Int, Float64}
summary_table = Matrix{Cell}(undef, length(nbins_list)*length(λ0_list), 3);
for (row, (λ0,n_bins)) in enumerate(Iterators.product(λ0_list, nbins_list))
    T_START   = Dates.now()
    RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
    OUTDIR    = joinpath(@__DIR__, "analysis_data", RUN_STAMP);
    isdir(OUTDIR) || mkpath(OUTDIR);
    @info "Created output directory" OUTDIR
    MyExperimentalAnalysis.OUTDIR   = OUTDIR;
    summary_table[row,:] = Cell[RUN_STAMP, n_bins, λ0]

    chosen_qm =  jldopen(data_qm_path,"r") do file
        file[JLD2_MyTools.make_keypath_qm(n_bins,0.270, λ0)]
    end
    Ic_QM_sim = [chosen_qm[i][:Icoil] for i in eachindex(chosen_qm)][3:end]
    zm_QM_sim = [chosen_qm[i][:z_max_smooth_spline_mm] for i in eachindex(chosen_qm)][3:end]

    # position
    z_mm        = 1e3 .* pixel_positions(z_pixels, n_bins, exp_pixelsize_z)
    z_mm_error  = 1e3 * 0.5 * exp_pixelsize_z * n_bins # half of the pixel size

    Icoils  = data_processed[:Currents]
    ΔIcoils = data_processed[:CurrentsError]
    nI      = length(Icoils)

    fig_I0 = plot(Icoils, 
        yerror = ΔIcoils,
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
    idx = findlast(<(0), Icoils)
    if isnothing(idx)
        @warn "No negative values in Icoils, skipping vspan"
    else
        vspan!([0, idx + 0.25], color=:gray, alpha=0.30, label="zero")
    end
    display(fig_I0) 
    saveplot(fig_I0, "current_range")

    ########################################################################
    ############# MEAN ANALYSIS ###########################################
    ########################################################################

    # --- Compute F1 / F2 mean profiles ----------------------------------------
    profiles_F1 = extract_profiles(data_processed, :F1ProcessedImages, nI, z_pixels; n_bin=n_bins, with_error=true);
    profiles_F2 = extract_profiles(data_processed, :F2ProcessedImages, nI, z_pixels; n_bin=n_bins, with_error=true);

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

    data_centroid_mean  = 0.5 * (f1_mean_max .+ f2_mean_max)
    data_centroid_mean_error = 0.5 * sqrt(2)*z_mm_error*ones(length(data_centroid_mean))
    centroid_mean = post_threshold_mean(data_centroid_mean, Icoils, data_centroid_mean_error; 
                        threshold=0.010,
                        half_life=5, # in samples
                        eps=1e-6,
                        weighted=true)
    fig = plot(Icoils, data_centroid_mean,
        xerror = ΔIcoils,
        yerror = data_centroid_mean_error,
        label=false,
        color=:purple,
        marker=(:circle,3),
        markerstrokecolor = :purple,
        line=(:solid,1),
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$"),
        xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
                    [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xlim=(1e-3,1),
        yaxis = L"$z_{0} \ (\mathrm{mm})$",
        title="Centroid",
        legend=:topleft
    )
    hline!([centroid_mean.mean], label=L"Centroid $z=%$(round(centroid_mean.mean,digits=3))$mm")
    hspan!( [centroid_mean.mean - centroid_mean.sem,centroid_mean.mean + centroid_mean.sem], color=:orangered, alpha=0.30, label=L"Error = $\pm%$(round(centroid_mean.sem,digits=3))$mm")
    saveplot(fig,"mean_centroid")

    jldsave(joinpath(OUTDIR, "profiles_mean.jld2"),
        profiles = OrderedDict(:Icoils      => Icoils,
                               :Icoils_err  => ΔIcoils,
                               :Centroid_mm => (centroid_mean.mean, centroid_mean.sem), 
                               :z_mm        => z_mm,
                               :F1_profile  => profiles_F1.mean,
                               :F1_err      => profiles_F1.sem,
                               :F2_profile  => profiles_F2.mean,
                               :F2_err      => profiles_F2.sem,
                    )
    )

    df_mean = DataFrame(
        Icoil_A                 =  Icoils,
        Icoil_error_A           =  ΔIcoils, 

        F1_z_peak_mm            = f1_mean_max,
        F2_z_peak_mm            = f2_mean_max,
        
        Δz_mm                   = f1_mean_max .- f2_mean_max,
        
        F1_z_centroid_mm        = f1_mean_max .- centroid_mean.mean,
        F1_z_centroid_mm_sem    = sqrt.( z_mm_error^2 .+ centroid_mean.sem^2),
        
        F2_z_centroid_mm        = f2_mean_max .- centroid_mean.mean,
        F2_z_centroid_mm_sem    = sqrt.( z_mm_error^2 .+ centroid_mean.sem^2),
    )
    sort!(df_mean, :Icoil_A)
    CSV.write(joinpath(OUTDIR, "mean_data.csv"), df_mean);

    hl_Ic = TextHighlighter(
            (data, i, j) -> data[i, 1] == minimum(data[:, 1]),
            crayon"fg:white bold bg:dark_gray"
        );
    hl_F1 = TextHighlighter(
            (data, i, j) -> data[i,6]<0,
            crayon"fg:red bold bg:dark_gray"
        );
    hl_F2 = TextHighlighter(
            (data, i, j) -> data[i,8]>0,
            crayon"fg:green bold bg:dark_gray"
        );
    pretty_table(
        df_mean;
        title         = "Mean analysis",
        formatters    = [fmt__printf("%8.3f", [1]), fmt__printf("%8.5f", 3:9)],
        alignment     = :c,
        column_labels  = [
            ["Current", "Current Error", "F1 z", "F2 z", "Δz", "Centroid F1 z","Centroid F1 z Error","Centroid F2 z", "Centroid F1 z Error"], 
            ["[A]", "[A]" ,"[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
        ],
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style = TextTableStyle(first_line_column_label = crayon"yellow bold",
                        column_label  = crayon"yellow",
                        table_border  = crayon"blue bold",
                    ),
        equal_data_column_widths = true,
        highlighters  = [hl_Ic,hl_F1,hl_F2],
    )

    fig_01 = plot(abs.(df_mean[!,:Icoil_A]), df_mean[!,:F1_z_peak_mm],
        label=L"$F_{1}$",
        line=(:solid,:red,2),
    );
    plot!(abs.(df_mean[!,:Icoil_A]), df_mean[!,:F2_z_peak_mm],
        label=L"$F_{2}$",
        line=(:solid,:blue,2),
    );
    plot!(
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
        xlims = (1e-3,1.0),
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
    idx = findlast(<(0), df_mean.Icoil_A)
    if isnothing(idx) || idx == length(df_mean.Icoil_A)
        @warn "No valid negative Icoil_A (or no next element) → skipping vspan"
    else
        vspan!(
            [1e-8, abs.(df_mean.Icoil_A[idx + 1]) ],
            color = :gray,
            alpha = 0.30,
            label = "zero"
        )
    end
    plot!(abs.(df_mean[!,:Icoil_A]), df_mean[!,:F1_z_peak_mm], fillrange=df_mean[!,:F2_z_peak_mm],
        fillalpha=0.2,
        color=:purple,
        label = false,
    );
    hline!([centroid_mean.mean], line=(:dot,:black,2), label="Centroid");

    fig_02 = plot(abs.(df_mean[!,:Icoil_A]), df_mean[!,:F1_z_peak_mm] ,
        label=L"$F_{1}$",
        line=(:solid,:red,2),
    );
    plot!(abs.(df_mean[!,:Icoil_A]), 2*centroid_mean.mean .- df_mean[!,:F2_z_peak_mm],
        label=L"Centroid Mirrored $F_{2}$",
        line=(:solid,:blue,2),
    );
    plot!(
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
        xlims = (1e-3,1.0),
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
    idx = findlast(<(0), df_mean.Icoil_A)
    if isnothing(idx) || idx == length(df_mean.Icoil_A)
        @warn "No valid negative Icoil_A (or no next element) → skipping vspan"
    else
        vspan!(
            [1e-8, abs.(df_mean.Icoil_A[idx + 1])],
            color = :gray,
            alpha = 0.30,
            label = "zero"
        )
    end
    # Fill between y1 and y2
    plot!(abs.(df_mean[!,:Icoil_A]), df_mean[!,:F1_z_peak_mm], fillrange=2*centroid_mean.mean .- df_mean[!,:F2_z_peak_mm],
        fillalpha=0.2,
        color=:purple,
        label = false,
    );
    hline!([centroid_mean.mean], line=(:dot,:black,2), label="Centroid");

    fig_03 = plot(abs.(df_mean[!,:Icoil_A]), df_mean[!,:F1_z_centroid_mm] ,
        label=L"$F_{1}$",
        line=(:solid,:red,2),
    );
    plot!(abs.(df_mean[!,:Icoil_A]), df_mean[!,:F2_z_centroid_mm] ,
        label=L"$F_{2}$",
        line=(:solid,:blue,2),
    );
    plot!(
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
        xlims = (1e-3,1.0),
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
    idx = findlast(<(0), df_mean.Icoil_A)
    if isnothing(idx) || idx == length(df_mean.Icoil_A)
        @warn "No valid negative Icoil_A (or no next element) → skipping vspan"
    else
        vspan!(
            [1e-8, abs.(df_mean.Icoil_A[idx + 1]) ],
            color = :gray,
            alpha = 0.30,
            label = "zero"
        )
    end
    # Fill between y1 and y2
    plot!(abs.(df_mean[!,:Icoil_A]), df_mean[!,:F1_z_centroid_mm], fillrange=df_mean[!,:F2_z_centroid_mm],
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
        df_mean[!,:Icoil_A], abs.(df_mean[!,:F1_z_centroid_mm])/magnification_factor,
        xerror = df_mean[!,:Icoil_error_A],
        yerror = df_mean[!,:F1_z_centroid_mm_sem],
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
        xlims = (0.001,1.0),
        ylims = (1e-4,2.5),
        title = "F=1 Peak Position vs Current",
        label = data_directory,
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
    hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size" )
    plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
    marker=(:cross, :purple, 6),
    line=(:purple, :dash, 2, 0.5),
    markerstrokewidth=2,
    label="10142024"
    )
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
    # line=(:dash, :blue, 3),
    # markerstrokewidth=2,
    # label="10142024: QM"
    # )
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    )
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
    # line=(:dot, :red, 3),
    # markerstrokewidth=2,
    # label="10142024: CQD"
    # )
    saveplot(fig000, "mean_000")

    fig001=plot(
        df_mean[!,:Icoil_A], abs.(df_mean[!,:F1_z_centroid_mm])/magnification_factor,
        xerror = df_mean[!,:Icoil_error_A],
        yerror = df_mean[!,:F1_z_centroid_mm_sem],
        xaxis = (L"$I_{c} \ (\mathrm{A})$"),
        yaxis = (L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$"),
        title = "F=1 Peak Position vs Current",
        label = data_directory,
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
    hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size")
    plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
    marker=(:cross, :purple, 6),
    line=(:purple, :dash, 2, 0.5),
    markerstrokewidth=2,
    label="10142024"
    )
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
    # line=(:dash, :blue, 3),
    # markerstrokewidth=2,
    # label="10142024: QM"
    # )
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    )
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
    # line=(:dot, :red, 3),
    # markerstrokewidth=2,
    # label="10142024: CQD"
    # )
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
        df_mean[pos_mask,:Icoil_A], y_abs[pos_mask]/magnification_factor,
        xerror = df_mean[pos_mask,:Icoil_error_A],
        yerror = df_mean[pos_mask,:F1_z_centroid_mm_sem],
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
        xlims = (0.001,1.0),
        ylims = (1e-4,2.5),
        title = "F=1 Peak Position vs Current",
        label = data_directory,
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
    plot!(df_mean[neg_mask,:Icoil_A], y_abs[neg_mask]/magnification_factor, 
        xerror = df_mean[neg_mask,:Icoil_error_A],
        yerror = df_mean[neg_mask,:F1_z_centroid_mm_sem],
        label=false, 
        seriestype=:scatter,
        marker = (:xcross, :orangered2, 4),
        markerstrokecolor = :orangered2,
        markerstrokewidth = 2,
    );
    hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size" )
    plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
    marker=(:cross, :purple, 6),
    line=(:purple, :dash, 2, 0.5),
    markerstrokewidth=2,
    label="10142024"
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
    # line=(:dash, :blue, 3),
    # markerstrokewidth=2,
    # label="10142024: QM"
    # );
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
    # line=(:dot, :red, 3),
    # markerstrokewidth=2,
    # label="10142024: CQD"
    # );
    # saveplot(fig100, "mean_100")

    fig101=plot(
        df_mean[pos_mask,:Icoil_A], y_abs[pos_mask]/magnification_factor,
        xerror = df_mean[pos_mask,:Icoil_error_A],
        yerror = df_mean[pos_mask,:F1_z_centroid_mm_sem],
        xaxis = (L"$I_{c} \ (\mathrm{A})$"),
        yaxis = (L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$"),
        title = "F=1 Peak Position vs Current",
        label = data_directory,
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
    plot!(df_mean[neg_mask,:Icoil_A], y_abs[neg_mask]/magnification_factor,
        xerror = df_mean[neg_mask,:Icoil_error_A],
        yerror = df_mean[neg_mask,:F1_z_centroid_mm_sem],
        label=false, 
        seriestype=:scatter,
        marker = (:xcross, :orangered2, 4),
        markerstrokecolor = :orangered2,
        markerstrokewidth = 2,
    );
    hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size" )
    plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
    marker=(:cross, :purple, 6),
    line=(:purple, :dash, 2, 0.5),
    markerstrokewidth=2,
    label="10142024"
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
    # line=(:dash, :blue, 3),
    # markerstrokewidth=2,
    # label="10142024: QM"
    # );
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
    # line=(:dot, :red, 3),
    # markerstrokewidth=2,
    # label="10142024: CQD"
    # );
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

    f1_z_mm , f1_z_sem_mm  = vec(mean(f1_max, dims=1)) , sqrt.(vec(std(f1_max, dims=1; corrected=true)/sqrt(size(f1_max,1))).^2 .+ z_mm_error^2 );
    f2_z_mm , f2_z_sem_mm  = vec(mean(f2_max, dims=1)) , sqrt.(vec(std(f2_max, dims=1; corrected=true)/sqrt(size(f2_max,1))).^2 .+ z_mm_error^2 );

    data_centroid_fw       = 0.5 * (f1_z_mm .+ f2_z_mm)
    data_centroid_fw_error = 0.5 * sqrt.(f1_z_sem_mm.^2 + f2_z_sem_mm.^2) / 2 
    # centroid_fw = mean(data_centroid_fw, Weights(nI-1:-1:0)) 
    # centroid_std_err = std(data_centroid_fw, Weights(nI-1:-1:0); corrected=false) / sqrt(nI)
    centroid_fw = post_threshold_mean(data_centroid_fw, Icoils, data_centroid_fw_error; 
                        threshold=0.010,
                        half_life=5, # in samples
                        eps=1e-6,
                        weighted=true)

    fig = plot(Icoils, data_centroid_fw, 
        xerror=ΔIcoils,
        yerror= data_centroid_fw_error,
        label=false,
        color=:purple,
        marker=(:circle,3),
        markerstrokecolor=:purple,
        line=(:solid,1),
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$"),
        xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
                    [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xlim=(1e-3,1),
        yaxis = L"$z_{0} \ (\mathrm{mm})$",
        title="Centroid",
        legend=:topleft,
    )
    hline!([centroid_fw.mean], label=L"Centroid $z=%$(round(centroid_fw.mean,digits=3))$mm")
    hspan!([centroid_fw.mean - centroid_fw.sem, centroid_fw.mean + centroid_fw.sem], color=:orangered, alpha=0.30, label=L"Error = $\pm%$(round(centroid_fw.sem,digits=3))$mm")
    saveplot(fig,"fw_centroid")


    res = summarize_framewise(f1_max, f2_max, Icoils, centroid_fw, z_mm_error)
    df_fw = DataFrame(
        Icoil_A             = res.Icoil_A,
        Icoil_error_A       = ΔIcoils,

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

    hl_Ic = TextHighlighter(
            (data, i, j) -> data[i, 1] == minimum(data[:, 1]),
            crayon"fg:white bold bg:dark_gray"
        );
    hl_F1 = TextHighlighter(
            (data, i, j) -> data[i,9]<0,
            crayon"fg:red bold bg:dark_gray"
        );
    hl_F2 = TextHighlighter(
            (data, i, j) -> data[i,11]>0,
            crayon"fg:green bold bg:dark_gray"
        );
    pretty_table(
        df_fw;
        title         = "Framewise Analysis",
        formatters    = [fmt__printf("%8.3f",[1]), fmt__printf("%8.5f",2:6)],
        alignment     = :c,
        column_labels = [
            ["Current", "Current Error", "F1 z", "Std.Err.",  "F2 z", "Std.Err.", "Δz", "Std.Err.", "Centroid F1 z", "Std.Err.", "Centroid F2 z", "Std.Err."], 
            ["[mA]", "[mA]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
        ],
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style = TextTableStyle(first_line_column_label = crayon"yellow bold",
                        column_label  = crayon"yellow",
                        table_border  = crayon"blue bold",
        ),
        equal_data_column_widths = true,
        highlighters  = [hl_Ic,hl_F1,hl_F2],
    )


    fig_01 = plot(df_fw[!,:Icoil_A], df_fw[!,:F1_z_peak_mm],
        ribbon= df_fw[!, :F1_z_peak_se_mm ],
        label=L"$F_{1}$",
        line=(:solid,:red,1),
        fillalpha=0.23, 
        fillcolor=:red,  
    )
    plot!(df_fw[!,:Icoil_A], df_fw[!,:F1_z_peak_mm ], 
        fillrange=df_fw[!,:F2_z_peak_mm ],
        fillalpha=0.05,
        color=:purple,
        label = false,
    )
    plot!(df_fw[!,:Icoil_A], df_fw[!,:F2_z_peak_mm ],
        ribbon= df_fw[!, :F2_z_peak_se_mm ],
        label=L"$F_{2}$",
        line=(:solid,:blue,1),
        fillalpha=0.23, 
        fillcolor=:blue,  
    )
    plot!(
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
        xlims = (1e-3,1.0),
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
    idx = findlast(<(0), df_fw.Icoil_A)
    if isnothing(idx) || idx == length(df_fw.Icoil_A)
        @warn "No valid negative Icoil_A (or no next element) → skipping vspan"
    else
        vspan!(
            [1e-8, abs.(df_fw.Icoil_A[idx + 1])],
            color = :gray,
            alpha = 0.30,
            label = "zero"
        )
    end
    hline!([centroid_fw.mean], line=(:dot,:black,2), label="Centroid")

    fig_02 = plot(df_fw[!,:Icoil_A], df_fw[!,:F1_z_peak_mm] ,
        ribbon= df_fw[!,:F1_z_peak_se_mm],
        label=L"$F_{1}$",
        line=(:solid,:red,2),
        fillalpha=0.23, 
        fillcolor=:red,  
    )
    plot!(df_fw[!,:Icoil_A], 2*centroid_fw.mean .- df_fw[!,:F2_z_peak_mm] ,
        ribbon= df_fw[!,:F2_z_peak_se_mm],
        label=L"Mirrored $F_{2}$",
        line=(:solid,:blue,2),
        fillalpha=0.23, 
        fillcolor=:blue,  
    )
    plot!(
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
        xlims = (1e-3,1.0),
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
    idx = findlast(<(0), df_fw.Icoil_A)
    if isnothing(idx) || idx == length(df_fw.Icoil_A)
        @warn "No valid negative Icoil_A (or no next element) → skipping vspan"
    else
        vspan!(
            [1e-8, abs.(df_fw.Icoil_A[idx + 1])],
            color = :gray,
            alpha = 0.30,
            label = "zero"
        )
    end
    # Fill between y1 and y2
    plot!(df_fw[!,:Icoil_A], df_fw[!,:F1_z_peak_mm], fillrange=2*centroid_fw.mean .- df_fw[!,:F2_z_peak_mm],
        fillalpha=0.2,
        color=:purple,
        label = false,
    );
    hline!([centroid_fw.mean], line=(:dot,:black,2), label="Centroid")

    fig_03 = plot(df_fw[!,:Icoil_A], df_fw[!,:F1_z_centroid_mm] ,
        ribbons=df_fw[!,:F1_z_centroid_se_mm],
        label=L"$F_{1}$",
        line=(:solid,:red,2),
    );
    plot!(df_fw[!,:Icoil_A], df_fw[!,:F2_z_centroid_mm] ,
        label=L"$F_{2}$",
        line=(:solid,:blue,2),
    );
    plot!(
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = L"$z_{\mathrm{max}} \ (\mathrm{mm})$",
        xlims = (1e-3,1.0),
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
    idx = findlast(<(0), df_fw.Icoil_A)
    if isnothing(idx) || idx == length(df_fw.Icoil_A)
        @warn "No valid negative Icoil_A (or no next element) → skipping vspan"
    else
        vspan!(
            [1e-8, abs.(df_fw.Icoil_A[idx + 1])],
            color = :gray,
            alpha = 0.30,
            label = "zero"
        )
    end
    # Fill between y1 and y2
    plot!(df_fw[!,:Icoil_A], df_fw[!,:F1_z_centroid_mm], fillrange=df_fw[!,:F2_z_centroid_mm],
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
        df_fw[!,:Icoil_A], abs.(df_fw[!,:F1_z_centroid_mm])/magnification_factor,
        xerror = df_fw[!,:Icoil_error_A],
        yerror = df_fw[!,:F1_z_centroid_se_mm],
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
        xlims = (0.001,1.0),
        ylims = (1e-6,3.5),
        title = "F=1 Peak Position vs Current",
        label = data_directory,
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
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
    # line=(:dash, :blue, 3),
    # markerstrokewidth=2,
    # label="10142024: QM"
    # );
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
    # line=(:dot, :red, 3),
    # markerstrokewidth=2,
    # label="10142024: CQD"
    # );
    display(fig_log)
    saveplot(fig_log, "fw_000")

    fig_lin=plot(
        df_fw[!,:Icoil_A], abs.(df_fw[!,:F1_z_centroid_mm])/magnification_factor,
        xerror = df_fw[!,:Icoil_error_A],
        yerror = df_fw[!,:F1_z_centroid_se_mm],
        xaxis = (L"$I_{c} \ (\mathrm{A})$"),
        yaxis = (L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$"),
        title = "F=1 Peak Position vs Current",
        label = data_directory,
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
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
    # line=(:dash, :blue, 3),
    # markerstrokewidth=2,
    # label="10142024: QM"
    # )
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM",
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
    # line=(:dot, :red, 3),
    # markerstrokewidth=2,
    # label="10142024: CQD"
    # )
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
    y = df_fw[!,:F1_z_centroid_mm]/magnification_factor;
    y_abs = abs.(y);
    # Create masks for negative and non-negative values
    neg_mask = y .< 0;
    pos_mask = .!neg_mask;
    fig10=plot(
        df_fw[pos_mask,:Icoil_A], y_abs[pos_mask],
        xerror = df_fw[pos_mask,:Icoil_error_A],
        yerror = df_fw[pos_mask,:F1_z_centroid_se_mm],
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
        xlims = (0.001,1.0),
        ylims = (1e-4,2.5),
        title = "F=1 Peak Position vs Current",
        label = data_directory,
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
    plot!(df_fw[neg_mask,:Icoil_A], y_abs[neg_mask],
        yerror = df_fw[neg_mask,:F1_z_centroid_se_mm],
        xerror = df_fw[neg_mask,:Icoil_error_A],
        label=false, 
        seriestype=:scatter,
        marker = (:xcross, :orangered2, 4),
        markerstrokecolor = :orangered2,
        markerstrokewidth = 2,
    );
    hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size" )
    plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
    marker=(:cross, :purple, 6),
    line=(:purple, :dash, 2, 0.5),
    markerstrokewidth=2,
    label="10142024"
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
    # line=(:dash, :blue, 3),
    # markerstrokewidth=2,
    # label="10142024: QM"
    # );
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
    # line=(:dot, :red, 3),
    # markerstrokewidth=2,
    # label="10142024: CQD"
    # );
    display(fig10)

    fig11=plot(
        df_fw[pos_mask,:Icoil_A], y_abs[pos_mask],
        xerror = df_fw[pos_mask,:Icoil_error_A],
        yerror = df_fw[pos_mask,:F1_z_centroid_se_mm],
        xaxis = ( L"$I_{c} \ (\mathrm{A})$"),
        yaxis = (L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$"),
        title = "F=1 Peak Position vs Current",
        label = data_directory,
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
    plot!(df_fw[neg_mask,:Icoil_A], y_abs[neg_mask],
        xerror = df_fw[neg_mask,:Icoil_error_A],
        yerror = df_fw[neg_mask,:F1_z_centroid_se_mm],
        label=false, 
        seriestype=:scatter,
        marker = (:xcross, :orangered2, 4),
        markerstrokecolor = :orangered2,
        markerstrokewidth = 2,
    );
    hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size" )
    plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
    marker=(:cross, :purple, 6),
    line=(:purple, :dash, 2, 0.5),
    markerstrokewidth=2,
    label="10142024"
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
    # line=(:dash, :blue, 3),
    # markerstrokewidth=2,
    # label="10142024: QM"
    # );
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
    # line=(:dot, :red, 3),
    # markerstrokewidth=2,
    # label="10142024: CQD"
    # );
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
        df_fw[4:end,:Icoil_A], abs.(df_fw[4:end,:F1_z_centroid_mm])/magnification_factor,
        xerror = df_fw[4:end,:Icoil_error_A],
        yerror = df_fw[!,:F1_z_centroid_se_mm],
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
        # xlims = (1e-3,1.0),
        ylims = (0.9e-3,3.5),
        title = "F=1 Peak Position vs Current",
        label = data_directory,
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
    ) ;
    # plot!(fig_comp,data_JSF[:model][:,1], data_JSF[:model][:,2],
    # line=(:dash, :blue, 3),
    # markerstrokewidth=2,
    # label="10142024: QM"
    # );
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    );
    # plot!(fig_comp,data_JSF[:model][:,1], data_JSF[:model][:,3],
    # line=(:dot, :red, 3),
    # markerstrokewidth=2,
    # label="10142024: CQD"
    # );
    saveplot(fig_comp,"comparison_zoom")

    fig_comp=plot(
        df_fw[!,:Icoil_A], abs.(df_fw[!,:F1_z_centroid_mm])/magnification_factor,
        xerror = df_fw[!,:Icoil_error_A],
        yerror = df_fw[!,:F1_z_centroid_se_mm],
        xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
        yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
        xlims = (0.001,1.0),
        ylims = (1e-6,3.5),
        title = "F=1 Peak Position vs Current",
        label = data_directory,
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
    hspan!([1e-6,1000*n_bins* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size" )
    plot!(data_JSF[:exp][:,1], data_JSF[:exp][:,2],
    marker=(:cross, :purple, 6),
    line=(:purple, :dash, 2, 0.5),
    markerstrokewidth=2,
    label="10142024"
    )
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,2],
    # line=(:dash, :blue, 3),
    # markerstrokewidth=2,
    # label="10142024: QM"
    # )
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    );
    # plot!(data_JSF[:model][:,1], data_JSF[:model][:,3],
    # line=(:dot, :red, 3),
    # markerstrokewidth=2,
    # label="10142024: CQD"
    # )
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
        Run label               : $(RUN_STAMP)

    CAMERA FEATURES
        Number of pixels        : $(nx_pixels) × $(nz_pixels)
        Pixel size              : $(1e6*cam_pixelsize) μm

    IMAGES INFORMATION
        Magnification factor    : $magnification_factor
        Camera Binning          : $(exp_bin_x) × $(exp_bin_z)
        Effective pixels        : $(x_pixels) × $(z_pixels)
        Pixel size              : $(1e6*exp_pixelsize_x)μm × $(1e6*exp_pixelsize_z)μm
        xlims                   : ($(round(minimum(1e6*x_position), digits=6)) μm, $(round(maximum(1e3*x_position), digits=4)) mm)
        zlims                   : ($(round(minimum(1e6*z_position), digits=6)) μm, $(round(maximum(1e3*z_position), digits=4)) mm)

    EXPERIMENT CONDITIONS
        Currents (A)            : $(Icoils)
        Currents Error (A)      : $(ΔIcoils)
        No. of currents         : $(nI)
        Temperature (K)         : $(Temperature)

    ANALYSIS PROPERTIES
        Binning                 : $(n_bins)
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

    println("Experiment analysis finished!")
    alert("Experiment analysis finished!")

    pretty_table(summary_table;
    title         = data_directory,
    alignment     = :c,
    column_labels = [ "Filename", "n" ,  "λ₀" ],
    table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
    style         = TextTableStyle(first_line_column_label = crayon"yellow bold",
                    column_label  = crayon"yellow",
                    table_border  = crayon"blue bold",
                    ),
    equal_data_column_widths = true
    )
end

println("EXPERIMENTAL ANALYSIS COMPLETED!")
alert("EXPERIMENTAL ANALYSIS COMPLETED!")