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
include("./Modules/MyExperimentalAnalysis.jl");
using .MyExperimentalAnalysis;
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
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
data_directory      = "20251207" ;

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
MyExperimentalAnalysis.effective_cam_pixelsize_z    = exp_pixelsize_z;
MyExperimentalAnalysis.x_pixels                     = x_pixels;
MyExperimentalAnalysis.z_pixels                     = z_pixels;


# ------- Functions ------------------------- #

"""
    extract_all_profiles(
        data, key::String, nI::Int, z_pixels::Int;
        T::Type{<:Real}=Float64, n_bin::Int=1
    ) -> OrderedDict{Int, NamedTuple}

Compute per-frame z-profiles and their statistical summaries for each current.
For every current j, this function reads the image stack data[key][j], extracts
the z-profile of each frame, computes the standard error of the mean (SEM) over
x for each frame, and produces the mean profile over frames with full error
propagation.

----------------------------------------------------------------------------------------------------
Input Structure
----------------------------------------------------------------------------------------------------

data[key][j] must be a 3D array of size (Nx, Nz, Nf_j):

• Nx   = number of x-pixels  
• Nz   = number of z-pixels (equal to z_pixels)  
• Nf_j = number of frames for current j (may vary with j)

The function processes j = 1:nI currents.

----------------------------------------------------------------------------------------------------
Keyword Arguments
----------------------------------------------------------------------------------------------------

T::Type          Floating-point type for computations (default Float64).

n_bin::Int       Number of z-pixels per bin.
                 z_pixels must be divisible by n_bin.
                 If n_bin == 1, no binning is applied.
                 If n_bin > 1, contiguous groups of n_bin z-pixels are averaged,
                 producing z_out = z_pixels ÷ n_bin bins.

----------------------------------------------------------------------------------------------------
Per-Frame Computation
----------------------------------------------------------------------------------------------------

For each frame f and each z-bin:

1. Compute the x-mean profile:
       mean over Nx pixels if n_bin == 1,
       mean over Nx * n_bin pixels if n_bin > 1.

2. Compute the standard deviation over those pixels.

3. Compute the per-frame SEM over x:
       SEM_x = std / sqrt(number of samples)
       number of samples = Nx if n_bin == 1
                          = Nx * n_bin if n_bin > 1

Results are stored as:
    z_profiles[f, zbin]  = mean value
    z_errors[f, zbin]    = SEM_x for that frame and z-bin

----------------------------------------------------------------------------------------------------
Statistics Over Frames
----------------------------------------------------------------------------------------------------

For each current j:

Let y_f(z)       = z_profiles[f, z]
Let σ_f(z)       = z_errors[f, z]

Mean profile:
    mean_profile[z] = mean over frames f of y_f(z)

(A) Propagated SEM_x contribution:
    propagated_error[z] = sqrt( sum_f σ_f(z)^2 ) / Nf_j

(B) Frame-to-frame scatter:
    frame_var[z]    = sum_f ( y_f(z) - mean_profile[z] )^2 / (Nf_j - 1)
    frame_scatter[z] = sqrt( frame_var[z] / Nf_j )

Total SEM of the mean profile:
    total_error[z] = sqrt( propagated_error[z]^2 + frame_scatter[z]^2 )

----------------------------------------------------------------------------------------------------
Return Value
----------------------------------------------------------------------------------------------------

The function returns an OrderedDict{Int, NamedTuple}.
For each current j, the entry out[j] contains:

    nf               :: Int            # number of frames Nf_j
    z_profiles       :: Matrix{T}      # size (Nf_j, z_out)
    z_errors         :: Matrix{T}      # per-frame SEM_x, size (Nf_j, z_out)
    mean_profile     :: Vector{T}      # mean over frames, size (z_out,)
    propagated_error :: Vector{T}      # SEM_x contribution, size (z_out,)
    frame_scatter    :: Vector{T}      # frame-to-frame SEM contribution
    total_error      :: Vector{T}      # combined error of the mean profile

----------------------------------------------------------------------------------------------------
Example Usage
----------------------------------------------------------------------------------------------------

    profiles = extract_all_profiles(data, "F1", nI, 2560; T=Float64, n_bin=4)
    pf = profiles[5]

    mean_z = pf.mean_profile
    err_z  = pf.total_error
    zprof  = pf.z_profiles    # per-frame profiles
    zsem   = pf.z_errors      # per-frame SEM_x

    plot(z_mm, mean_z; ribbon=err_z)

"""
function extract_all_profiles(
    data, key::String, nI::Int, z_pixels::Int;
    T::Type{<:Real}=Float64, nz_bin::Int=1)

    @assert z_pixels % nz_bin == 0
    z_out = div(z_pixels, nz_bin)

    out = OrderedDict{Int, NamedTuple}()

    @inbounds for j in 1:nI

        # stack_raw :: (Nx, Nz, Nf_j)
        stack_raw = data[key][j]
        Nx, Nz, Nf_j = size(stack_raw)
        @assert Nz == z_pixels

        # Convert only if needed
        stack = T <: eltype(stack_raw) ? stack_raw : T.(stack_raw)

        # Per-frame outputs
        z_profiles = Matrix{T}(undef, Nf_j, z_out)
        z_errors   = Matrix{T}(undef, Nf_j, z_out)

        if nz_bin == 1
            # ---- No Z-binning ----
            @inbounds for f in 1:Nf_j
                frame = stack[:, :, f]    # Nx × Nz

                # x-mean
                xm = mean(frame; dims=1)        # 1 × Nz

                # std over x
                xs = std(frame; dims=1, corrected=true)

                # SEM over x: std(x) / sqrt(Nx)
                sem = xs ./ sqrt(Nx)

                z_profiles[f, :] = dropdims(xm;  dims=1)
                z_errors[f, :]   = dropdims(sem; dims=1)
            end

        else
            # ---- With Z-binning ----
            # reshape: Nx × nz_bin × z_out × Nf_j
            B = reshape(stack, Nx, nz_bin, z_out, Nf_j)

            @inbounds for f in 1:Nf_j
                slice = B[:,:,:,f]        # Nx × nz_bin × z_out

                # x-bin mean → 1×1×z_out
                xm = mean(slice; dims=(1,2))

                # std over x and bin
                xs = std(slice; dims=(1,2), corrected=true)

                # SEM_x = std / sqrt(Nx * nz_bin)
                sem = xs ./ sqrt(Nx * nz_bin)

                z_profiles[f, :] = dropdims(xm;  dims=(1,2))
                z_errors[f, :]   = dropdims(sem; dims=(1,2))
            end
        end

        # ============================================================
        #         STATISTICS OVER FRAMES (MEAN + ERROR PROP)
        # ============================================================

        # Mean profile over frames → size (z_out,)
        mean_profile = vec(mean(z_profiles; dims=1))

        # ---- (A) propagate per-frame SEM_x ----
        # propagated_error[z] = sqrt( sum_f (σ_f(z)^2) ) / Nf
        propagated_error = vec( sqrt.(sum(z_errors.^2; dims=1)) ./ Nf_j )

        # ---- (B) frame-to-frame scatter ----
        # diffs[f,z] = z_profiles[f,z] - mean_profile[z]
        diffs = z_profiles .- mean_profile'

        # unbiased variance across frames
        frame_var = sum(diffs.^2; dims=1) ./ (Nf_j - 1)

        # SEM of the mean from frame scatter
        frame_scatter = vec( sqrt.(frame_var ./ Nf_j) )

        # ---- Total SEM (quadrature sum) ----
        total_error = sqrt.(propagated_error.^2 .+ frame_scatter.^2)

        # ============================================================
        #                       SAVE EVERYTHING
        # ============================================================

        out[j] = (
            key              = key,
            nf               = Nf_j,
            z_profiles       = z_profiles,
            z_errors         = z_errors,
            mean_profile     = mean_profile,
            propagated_error = propagated_error,   # from SEM_x
            frame_scatter    = frame_scatter,      # frame variability
            total_error      = total_error         # combined SEM
        )
    end

    return out
end

function maxima_clustered_select(x, xprev; tol=0.5)
    # --- 1. Cluster by relative tolerance ---
    xs = sort(x)
    clusters = [[xs[1]]]

    for i in 2:length(xs)
        if abs((xs[i] - xs[i-1]) / xs[i-1]) <= tol
            push!(clusters[end], xs[i])
        else
            push!(clusters, [xs[i]])
        end
    end

    # --- 2. Per-cluster mean and std (singleton std = Inf) ---
    μ = mean.(clusters)
    σ = [length(c) > 1 ? std(c) : Inf for c in clusters]
    stats = hcat(μ, σ)  # columns: [mean  std]

    # --- 3. Select cluster closest to xprev (relative difference) ---
    if xprev == 0
        # Avoid divide-by-zero: fall back to absolute distance
        d = abs.(μ .- xprev)
    else
        d = abs.((μ .- xprev) ./ xprev)
    end

    idx = argmin(d)

    return (
        clusters        = clusters,
        stats           = stats,          # [mean  std] per cluster
        idx             = idx,            # index of selected cluster
        selected_values = clusters[idx],  # the cluster itself
        selected_mean   = μ[idx],
        selected_std    = σ[idx]
    )
end

function compute_peak_positions_mean(data;
    currents::Vector,
    nz_bin::Integer,
    λ0::Float64=0.01,
    dir::String)
    # --------- Prepare constants ---------
    nI = length(currents)

    # --------- Validate signal_key -------
    signal_key = union([data[i].key for i=1:nI])
    @assert length(signal_key) == 1
    signal_label = signal_key[1] == "F1ProcessedImages" ? "F1" :
                signal_key[1] == "F2ProcessedImages" ? "F2" :
                error("Invalid data:  it must be either F1 or F2")
    @info "Processing mean maxima" Signal=signal_label


    # z-axes (mm)
    z_binned_mm = 1e3 .* pixel_positions(z_pixels, nz_bin, cam_pixelsize)

    midpoint = sum(extrema(z_binned_mm)) / 2
    peak_positions = zeros(Float64, nI)

    # --------- Main loop over currents ---------
    anim = @animate for j in 1:nI

        mean_profile = data[j].mean_profile

        # --- Ensure binning consistency ---
        @assert length(mean_profile) % nz_bin == 0

        # Apply optional half-maximum filtering (disabled for now)
        z_fit = z_binned_mm
        y_fit = mean_profile
        half_max = false
        if half_max
            ymax = maximum(y_fit)
            mask = y_fit .> ymax * 0.5
            z_fit = z_fit[mask]
            y_fit = y_fit[mask]
        end

        # --------- Spline fit ---------
        S_fit = BSplineKit.fit(
            BSplineOrder(4), z_fit, y_fit, λ0;
            weights = compute_weights(z_fit, λ0)
        )

        negative_spline(x) = -S_fit(x[1])

        # --------- Generate initial guesses for maximum finder ---------
        zmax_prev = (j == 1 ? midpoint : mean(peak_positions[j-1]) )

        initial_guesses = sort([
            quantile(z_fit, 0.15),
            quantile(z_fit, 0.25),
            quantile(z_fit, 0.40),
            midpoint,
            zmax_prev,
            z_fit[argmax(y_fit)],
            quantile(z_fit, 0.65),
            quantile(z_fit, 0.75),
            quantile(z_fit, 0.90)
        ])

        # --------- Optimize ---------
        minima_candidates = Float64[]
        for g in initial_guesses
            res = optimize(
                negative_spline,
                [minimum(z_fit)], [maximum(z_fit)], [g],
                Fminbox(LBFGS())
            )
            push!(minima_candidates, Optim.minimizer(res)[1])
        end

        # Remove duplicates
        sort!(minima_candidates)
        filtered = [minima_candidates[1]]
        for m in minima_candidates[2:end]
            if all(abs(m - x) > 1e-9 for x in filtered)
                push!(filtered, m)
            end
        end

        # Order by spline height
        vals   = -S_fit.(filtered)
        order  = sortperm(vals)
        maxima = filtered[order]

        # --------- Cluster maxima & pick closest to previous one ---------
        res_cluster = maxima_clustered_select(maxima, zmax_prev; tol=0.01)
        peak_positions[j] = res_cluster.selected_mean

        # --------- Plotting ---------
        fig = plot(
            z_binned_mm, mean_profile,
            ribbon = data[j].total_error,
            marker=(:circle,1.2,:white,stroke(:black,1)),
            line=(:solid,0.001,0.1,:black),
            fillcolor=:gray26, fillalpha=0.8,
            label="Raw data (mean)"
        )

        plot!(fig, z_fit, S_fit.(z_fit),
            line=(:solid,0.70,2,:dodgerblue),
            label="Interpolation (s=$(λ0))"
        )

        vline!(fig, [peak_positions[j]],
            label = L"$z_{\mathrm{max}} = %$(round(peak_positions[j]; sigdigits=5)) \mathrm{mm}$",
            line = (:dash,2,0.80,:darkgreen)
        )

        plot!(fig,
            xlabel=L"$z$ (mm)", ylabel="Counts",
            legend=:outerbottom,
            legend_columns = 3,
            legend_title=L"$I_{c} = %$(round(currents[j], digits=3)) \mathrm{A}$",
            legendtitlefontsize=8,
            legendfontsize=6,
            legend_foreground_color = nothing,
            legend_background_color = nothing,
            bottom_margin=-5mm,
        )
        display(fig)

    end

    gif(anim, joinpath(dir,"peak_positions_mean_$(signal_label).gif"), fps=1)
    return peak_positions
end

function compute_peak_positions_perframe(data;
    currents::Vector,
    nz_bin::Integer,
    λ0::Float64=0.01,
    display_figs::Bool=false)

    # --------- Prepare constants ---------
    nI = length(currents)

    # --------- Validate signal_key -------
    signal_key = union([data[i].key for i=1:nI])
    @assert length(signal_key) == 1
    signal_label = signal_key[1] == "F1ProcessedImages" ? "F1" :
                signal_key[1] == "F2ProcessedImages" ? "F2" :
                error("Invalid data:  it must be either F1 or F2")
    @info "Processing mean maxima" Signal=signal_label


    # z-axes (mm)
    z_binned_mm = 1e3 .* pixel_positions(z_pixels, nz_bin, cam_pixelsize)

    midpoint = sum(extrema(z_binned_mm)) / 2
    peak_positions = Vector{Vector{Float64}}(undef, nI)

    # --------- Main loop over currents ---------
    for j in 1:nI

        perframe_profiles = data[j].z_profiles
        nf_j, nz_j = size(perframe_profiles)
        nf_peak_positions = zeros(Float64,nf_j)
        # --- Ensure binning consistency ---
        @assert nz_j % nz_bin == 0
        for k in 1:nf_j
            # Apply optional half-maximum filtering (disabled for now)
            z_fit = z_binned_mm
            y_fit = perframe_profiles[k,:]
            half_max = false
            if half_max
                ymax = maximum(y_fit)
                mask = y_fit .> ymax * 0.5
                z_fit = z_fit[mask]
                y_fit = y_fit[mask]
            end

            # --------- Spline fit ---------
            S_fit = BSplineKit.fit(
                BSplineOrder(4), z_fit, y_fit, λ0;
                weights = compute_weights(z_fit, λ0)
            )

            negative_spline(x) = -S_fit(x[1])

            # --------- Generate initial guesses for maximum finder ---------
            zmax_prev = (j == 1 ? midpoint : mean(peak_positions[j-1]))

            initial_guesses = sort([
                quantile(z_fit, 0.15),
                quantile(z_fit, 0.25),
                quantile(z_fit, 0.40),
                midpoint,
                zmax_prev,
                z_fit[argmax(y_fit)],
                quantile(z_fit, 0.65),
                quantile(z_fit, 0.75),
                quantile(z_fit, 0.90)
            ])

            # --------- Optimize ---------
            minima_candidates = Float64[]
            for g in initial_guesses
                res = optimize(
                    negative_spline,
                    [minimum(z_fit)], [maximum(z_fit)], [g],
                    Fminbox(LBFGS())
                )
                push!(minima_candidates, Optim.minimizer(res)[1])
            end

            # Remove duplicates
            sort!(minima_candidates)
            filtered = [minima_candidates[1]]
            for m in minima_candidates[2:end]
                if all(abs(m - x) > 1e-9 for x in filtered)
                    push!(filtered, m)
                end
            end

            # Order by spline height
            vals   = -S_fit.(filtered)
            order  = sortperm(vals)
            maxima = filtered[order]

            # --------- Cluster maxima & pick closest to previous one ---------
            res_cluster = maxima_clustered_select(maxima, zmax_prev; tol=0.01)

            nf_peak_positions[k] = res_cluster.selected_mean

            # --------- Plotting ---------
            if display_figs
                fig = plot(
                    z_binned_mm, perframe_profiles[k,:],
                    ribbon = data[j].z_errors[k,:],
                    marker=(:circle,1.2,:white,stroke(:black,1)),
                    line=(:solid,0.001,0.1,:black),
                    fillcolor=:gray26, fillalpha=0.8,
                    label="Raw data (frame $k)"
                )
                plot!(fig, z_fit, S_fit.(z_fit),
                    line=(:solid,0.70,2,:dodgerblue),
                    label="Interpolation (s=$(λ0))"
                )
                vline!(fig, [nf_peak_positions[k]],
                    label = "$(round(nf_peak_positions[k]; sigdigits=5)) mm",
                    line = (:dash,2,0.80,:darkgreen)
                )
                plot!(fig,
                    xlabel=L"$z$ (mm)", ylabel="Counts",
                    legend=:outerbottom,
                    legend_columns=3,
                    legend_title=L"$%$(round(currents[j], digits=3)) \mathrm{A}$",
                    legendtitlefontsize=8,
                    legendfontsize=6,
                    legend_foreground_color = nothing,
                    legend_background_color = nothing,
                    bottom_margin = -5mm,
                )
                display(fig)
            end
        end
        peak_positions[j] = nf_peak_positions

    end
   
    return peak_positions
end

# --------------------------------------------------- #

# data for comparison
data_qm  = load(joinpath(@__DIR__,"simulation_data","qm_simulation_7M","qm_7000000_screen_profiles_f1_table.jld2"))["table"]
unique(first.(keys(data_qm)))
unique(getindex.(keys(data_qm),2))
unique(last.(keys(data_qm)))

# load experimental data processed
data = matread(joinpath("W:\\SternGerlach\\experiments",data_directory,"data.mat"))["data"]

Icoils = vec(data["Current_mA"])/1000
Icoils[1] = 1.0e-6
nI = length(Icoils)
ΔIcoils = vec(data["AmmeterRange_mA"])/1000*0.015

@time data_f1 = extract_all_profiles(data, "F1ProcessedImages", nI, z_pixels; nz_bin=1);
@time data_f2 = extract_all_profiles(data, "F2ProcessedImages", nI, z_pixels; nz_bin=1);

for j = 1:nI
    fig = plot()
    colors = palette(:darkrainbow,data_f1[j].nf);
    for i=1:5:data_f1[j].nf
        plot!(data_f1[j].z_profiles[i,:],
            ribbon=data_f1[j].z_errors[i,:],
            color=colors[i],
            label=nothing,)
    end
    plot!(title=L"$I_{c} =%$(round(Icoils[j];digits=3))\mathrm{A}$",
        xlabel=L"$z$ pixels",
        ylabel="counts")
    display(fig)
end

fig = plot(legend=:outerright,)
colors = palette(:darkrainbow,nI);
for j = 1:nI
    plot!(data_f1[j].mean_profile,
        ribbon=data_f1[j].total_error,
        color=colors[j],
        label=L"$I_{c} =%$(round(Icoils[j];digits=3))\mathrm{A}$",)
    plot!(
        size=(800,500),
        xlabel=L"$z$ pixels",
        ylabel="counts")
end
display(fig)


# iterative parameters
nbins_list  = (1, 2, 4, 8)
λ0_list     = (0.001,0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10)
const Cell = Union{Missing, String, Int, Float64}
summary_table = Matrix{Cell}(undef, length(nbins_list)*length(λ0_list), 3);
for (row, (λ0,nz_binning)) in enumerate(Iterators.product(λ0_list, nbins_list))
    # λ0 = λ0_list[1]
    # nz_binning = nbins_list[1]
    # row = 1

    chosen_qm = data_qm[(nz_binning, 0.200, λ0)]
    Ic_QM_sim = [chosen_qm[i][:Icoil] for i in eachindex(chosen_qm)][2:end]
    zm_QM_sim = [chosen_qm[i][:z_max_smooth_spline_mm] for i in eachindex(chosen_qm)][2:end]

    T_START   = Dates.now()
    RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
    OUTDIR    = joinpath(@__DIR__, "analysis_data", RUN_STAMP);
    isdir(OUTDIR) || mkpath(OUTDIR);
    @info "Created output directory" OUTDIR
    MyExperimentalAnalysis.OUTDIR   = OUTDIR;
    summary_table[row,:] = Cell[RUN_STAMP, nz_binning, λ0]

    # position
    z_mm        = 1e3 .* pixel_positions(z_pixels, nz_binning, exp_pixelsize_z)
    z_mm_error  = 1e3 * 0.5 * exp_pixelsize_z * nz_binning # half of the pixel size

    @time data_f1 = extract_all_profiles(data, "F1ProcessedImages", nI, z_pixels; nz_bin=nz_binning);
    @time data_f2 = extract_all_profiles(data, "F2ProcessedImages", nI, z_pixels; nz_bin=nz_binning);

    zf1_mean = compute_peak_positions_mean(data_f1;
        currents=Icoils,
        nz_bin=nz_binning,
        λ0=λ0,
        dir=OUTDIR
    );
    zf2_mean = compute_peak_positions_mean(data_f2;
        currents=Icoils,
        nz_bin=nz_binning,
        λ0=λ0,
        dir=OUTDIR
    );

    zf1_perframe = compute_peak_positions_perframe(data_f1;
        currents=Icoils,
        nz_bin=nz_binning,
        λ0=λ0
    )
    zf2_perframe = compute_peak_positions_perframe(data_f2;
        currents=Icoils,
        nz_bin=nz_binning,
        λ0=λ0
    )

    fig = plot(Icoils[2:end],zf1_mean[2:end],
        label=L"$F=1$ : mean",
        marker=(:circle,4,0.5,:blue,stroke(:blue,1)),
        line=(:solid,1,:blue))
    plot!(fig,Icoils[2:end],zf2_mean[2:end],
        label=L"$F=2$ : mean",
        marker=(:circle,4,0.5,:red,stroke(:red,1)),
        line=(:solid,1,:red))
    plot!(fig,Icoils[2:end],[mean(zf1_perframe[i]) for i=2:nI],
        yerror = [std(zf1_perframe[i]) for i=2:nI] ./ sqrt.([length(zf1_perframe[i]) for i=2:nI]),
        label=L"$F=1$ : per-frame",
        marker=(:xcross,4,0.5,:darkgreen,stroke(:darkgreen,2,0.9)),
        line=(:dash,1,:darkgreen)
    )
    plot!(fig,Icoils[2:end],[mean(zf2_perframe[i]) for i=2:nI],
        yerror = [std(zf2_perframe[i]; corrected=true) for i=2:nI] ./ sqrt.([length(zf2_perframe[i]) for i=2:nI]),
        label=L"$F=2$ : per-frame",
        marker=(:xcross,4,0.5,:orangered2,stroke(:orangered2,2,0.9)),
        line=(:dash,1,:orangered2)
    )
    plot!(fig,
        title = data_directory,
        xaxis=("Current (A)",:log10),
        yaxis=("Peak position (mm)",:identity))
    plot!(fig,title="Peak positions")
    display(fig)
    savefig(fig,joinpath(OUTDIR,"peak_position_comparison.$(FIG_EXT)"))


    # ==================================================================== #
    # ------------- Peak position for the mean profile ------------------- #

    f1_mean_max = zf1_mean
    f2_mean_max = zf2_mean
    data_centroid_mean  = 0.5 * (f1_mean_max .+ f2_mean_max)[1:end-6]
    data_centroid_mean_error = 0.5 * sqrt(2)*z_mm_error*ones(length(data_centroid_mean))
    centroid_mean = post_threshold_mean(data_centroid_mean, Icoils[1:end-6], data_centroid_mean_error; 
                        threshold=0.000,
                        half_life=8, # in samples
                        eps=1e-6,
                        weighted=true)

    fig = plot(Icoils[1:end-6], data_centroid_mean,
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
        title="Mean - Centroid",
        legend=:topleft
    )
    hline!([centroid_mean.mean], label=L"Centroid $z=%$(round(centroid_mean.mean,digits=3))$mm")
    hspan!( [centroid_mean.mean - centroid_mean.sem,centroid_mean.mean + centroid_mean.sem], color=:orangered, alpha=0.30, label=L"Error = $\pm%$(round(centroid_mean.sem,digits=3))$mm")
    display(fig)
    saveplot(fig,"mean_centroid")

    jldsave(joinpath(OUTDIR, "profiles_mean.jld2"),
        profiles = OrderedDict(:Icoils      => Icoils,
                               :Icoils_err  => ΔIcoils,
                               :Centroid_mm => (centroid_mean.mean, centroid_mean.sem), 
                               :z_mm        => z_mm,
                               :F1_profile  => [data_f1[i].mean_profile for i =1:nI],
                               :F1_err      => [data_f1[i].total_error for i =1:nI],
                               :F2_profile  => [data_f2[i].mean_profile for i =1:nI],
                               :F2_err      => [data_f2[i].total_error for i =1:nI],
                    )
    )

    df_mean = DataFrame(
        Icoil_A                 =  Icoils,
        Icoil_error_A           =  ΔIcoils, 

        F1_z_peak_mm            = f1_mean_max,
        F2_z_peak_mm            = f2_mean_max,
        
        Δz_mm                   = f2_mean_max .- f1_mean_max,
        
        F1_z_centroid_mm        = - (f1_mean_max .- centroid_mean.mean),
        F1_z_centroid_mm_sem    = sqrt.( z_mm_error^2 .+ centroid_mean.sem^2),
        
        F2_z_centroid_mm        = f2_mean_max .- centroid_mean.mean,
        F2_z_centroid_mm_sem    = sqrt.( z_mm_error^2 .+ centroid_mean.sem^2),
    )
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
            (data, i, j) -> data[i,8]<0,
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
    # Fill between y1 and y2
    plot!(abs.(df_mean[!,:Icoil_A]), df_mean[!,:F1_z_centroid_mm], fillrange=df_mean[!,:F2_z_centroid_mm],
        fillalpha=0.2,
        color=:purple,
        label = false,
    );

    fig=plot(fig_01, fig_02, fig_03, 
    layout=@layout([a ; b ; c]),
    share=:x,
    left_margin=2mm,
    );
    plot!(fig[1], xlabel="", xformatter=_->"");
    plot!(fig[2], xlabel="", xformatter=_->"", title = "", top_margin = -9mm);
    plot!(fig[3], title="", top_margin = -9mm);
    display(fig)
    saveplot(fig, "mean_peak_centroid") 

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
    hspan!([1e-6,1000*nz_binning* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size" )
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    );
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
    hspan!([1e-6,1000*nz_binning* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size" )
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    );
    # saveplot(fig101, "mean_101")

    fig1 = plot(fig100, fig101,
        layout=(1,2),
        size=(1000,400),
        left_margin=8mm,
        bottom_margin=5mm,
    )
    saveplot(fig1, "mean_01")



    # ==================================================================== #
    # -------- Peak position for each acquired frame profile ------------- #

    f1_z_mm , f1_z_sem_mm  = [mean(zf1_perframe[i]) for i=1:nI]  , sqrt.( [std(zf1_perframe[i]; corrected=true)./sqrt.(length(zf1_perframe[i])) for i=1:nI].^2 .+ z_mm_error^2 );
    f2_z_mm , f2_z_sem_mm  = [mean(zf2_perframe[i]) for i=1:nI] , sqrt.( [std(zf2_perframe[i]; corrected=true)./sqrt.(length(zf2_perframe[i])) for i=1:nI].^2 .+ z_mm_error^2 );
    data_centroid_fw       = 0.5 * (f1_z_mm .+ f2_z_mm)[1:end-5]
    data_centroid_fw_error = (0.5 * sqrt.(f1_z_sem_mm.^2 + f2_z_sem_mm.^2) / 2 )[1:end-5]
    # centroid_fw = mean(data_centroid_fw, Weights(nI-1:-1:0)) 
    # centroid_std_err = std(data_centroid_fw, Weights(nI-1:-1:0); corrected=false) / sqrt(nI)
    centroid_fw = post_threshold_mean(data_centroid_fw, Icoils[1:end-5], data_centroid_fw_error; 
                        threshold=0.010,
                        half_life=7, # in samples
                        eps=1e-6,
                        weighted=true)

    fig = plot(Icoils[1:end-5], data_centroid_fw, 
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
        title="Per-frame - Centroid",
        legend=:topleft,
    )
    hline!([centroid_fw.mean], label=L"Centroid $z=%$(round(centroid_fw.mean,digits=3))$mm")
    hspan!([centroid_fw.mean - centroid_fw.sem, centroid_fw.mean + centroid_fw.sem], color=:orangered, alpha=0.30, label=L"Error = $\pm%$(round(centroid_fw.sem,digits=3))$mm")


    df_fw = DataFrame(
        Icoil_A             = Icoils,
        Icoil_error_A       = ΔIcoils,

        F1_z_peak_mm        = f1_z_mm , 
        F1_z_peak_se_mm     = f1_z_sem_mm , 

        F2_z_peak_mm        = f2_z_mm , 
        F2_z_peak_se_mm     = f2_z_sem_mm ,

        Δz_mm               = f2_z_mm  .- f1_z_mm ,
        Δz_se_mm            = sqrt.( (f1_z_sem_mm).^2 .+ (f2_z_sem_mm).^2 ),

        F1_z_centroid_mm    = -( f1_z_mm .- centroid_fw.mean ), 
        F1_z_centroid_se_mm = sqrt.( (f1_z_sem_mm).^2 .+ (centroid_fw.sem).^2 ),
        F2_z_centroid_mm    = f2_z_mm .- centroid_fw.mean, 
        F2_z_centroid_se_mm = sqrt.( (f2_z_sem_mm).^2 .+ (centroid_fw.sem).^2 ),
    )
    CSV.write(joinpath(OUTDIR, "fw_data.csv"), df_mean);

    hl_Ic = TextHighlighter(
            (data, i, j) -> data[i, 1] == minimum(data[:, 1]),
            crayon"fg:white bold bg:dark_gray"
        );
    hl_F1 = TextHighlighter(
            (data, i, j) -> data[i,9]<0,
            crayon"fg:red bold bg:dark_gray"
        );
    hl_F2 = TextHighlighter(
            (data, i, j) -> data[i,11]<0,
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

    fig_01 = plot(df_fw[!,:Icoil_A], df_fw[!,:F2_z_peak_mm],
    ribbon= df_fw[!, :F2_z_peak_se_mm ],
    label=L"$F_{2}$",
    line=(:solid,:red,1),
    fillalpha=0.23, 
    fillcolor=:red,  
    )
    plot!(df_fw[!,:Icoil_A], df_fw[!,:F2_z_peak_mm ], 
        fillrange=df_fw[!,:F1_z_peak_mm ],
        fillalpha=0.05,
        color=:purple,
        label = false,
    )
    plot!(df_fw[!,:Icoil_A], df_fw[!,:F1_z_peak_mm ],
        ribbon= df_fw[!, :F1_z_peak_se_mm ],
        label=L"$F_{1}$",
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
        color=:red,
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
    hspan!([1e-6,1000*nz_binning* exp_pixelsize_z], color=:gray, alpha=0.30, label="Pixel size" )
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    );
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
    hspan!([1e-6,1000*nz_binning* exp_pixelsize_z], color=:gray, alpha=0.30, label="Pixel size" )
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM",
    );
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
    hspan!([1e-6,1000*nz_binning* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size" )
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
    );
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
    hspan!([1e-6,1000*nz_binning* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size" )
    plot!(Ic_QM_sim, zm_QM_sim,
        line=(:dash,:darkgreen,2.5),
        label="Analytic QM"    ,
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





    # ==================================================================== #



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
    GC.gc()

end





























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
plot!(Ic_QM_sim, zm_QM_sim,
    line=(:dash,:darkgreen,2.5),
    label="Analytic QM"    ,
);
display(fig_comp)
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
hspan!([1e-6,1000*nz_binning* exp_pixelsize_z], color=:gray, alpha=0.30, label="Effective pixel size" )
plot!(Ic_QM_sim, zm_QM_sim,
    line=(:dash,:darkgreen,2.5),
    label="Analytic QM"    ,
);
display(fig_comp)
saveplot(fig_comp,"comparison")
