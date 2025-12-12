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
data_qm   = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_3m","qm_3000000_screen_profiles_table.jld2"))["table"]


data = matread(joinpath("Z:\\SingleSternGerlachExperimentData\\experiments\\20251207","data.mat"))["data"]

T_START   = Dates.now()
RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
OUTDIR    = joinpath(@__DIR__, "analysis_data", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);

Currents = vec(data["Current_mA"])/1000
nI = length(Currents)


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
    peak_positions = Vector{Vector{Float64}}(undef, nI)

    Nframes_total = sum([data[i].nf for i=1:nI])
    idx = 0 
    plots_anim = Vector{Plots.Plot}(undef, Nframes_total)

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
            # display(fig)
            idx += 1
            plots_anim[idx] = fig   # store in preallocated slot
        end
        peak_positions[j] = nf_peak_positions

    end

    anim = @animate for i=1:Nframes_total
        display(plots_anim[i])
    end
    gif(anim, joinpath(dir,"peak_positions_perframe_$(signal_label).gif"), fps=3)
    
    return peak_positions
end

nz_binning = 2
@time data_f1 = extract_all_profiles(data, "F1ProcessedImages", nI, z_pixels; nz_bin=nz_binning);
@time data_f2 = extract_all_profiles(data, "F2ProcessedImages", nI, z_pixels; nz_bin=nz_binning);


for j = 1:nI
fig = plot()
colors = palette(:darkrainbow,data_f1[j].nf);
for i=1:5:data_f1[j].nf
plot!(data_f1[j].z_profiles[i,:],
ribbon=data_f1[j].z_errors[i,:],
color=colors[i],
label=nothing,)
end
plot!(xlabel="$(Currents[j])")
display(fig)
end

fig = plot()
colors = palette(:darkrainbow,nI);
for j = 1:nI
plot!(data_f1[j].mean_profile,
ribbon=data_f1[j].total_error,
color=colors[j],
label=nothing,)
plot!(xlabel="$(Currents[j])")
end
display(fig)


zf1_mean = compute_peak_positions_mean(data_f1;
    currents=Currents,
    nz_bin=nz_binning,
    λ0=0.01,
    dir=OUTDIR
);
zf2_mean = compute_peak_positions_mean(data_f2;
    currents=Currents,
    nz_bin=nz_binning,
    λ0=0.01,
    dir=OUTDIR
);




zf1_perframe = compute_peak_positions_perframe(data_f1;
    currents=Currents,
    nz_bin=nz_binning,
    λ0=0.01,
    dir=OUTDIR
)
zf2_perframe = compute_peak_positions_perframe(data_f2;
    currents=Currents,
    nz_bin=nz_binning,
    λ0=0.01,
)


plot(Currents[2:end],zf1_mean[2:end])
plot!(Currents[2:end],zf2_mean[2:end])
plot!(Currents[2:end],[mean(zf1_perframe[i]) for i=2:nI])
plot!(Currents[2:end],[mean(zf2_perframe[i]) for i=2:nI])
plot!(xscale=:log10)


