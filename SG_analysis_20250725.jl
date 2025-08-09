# Kelvin Titimbo,Xukun Lin, S. Suleyman Kahraman, and Lihong V. Wang
# California Institute of Technology
# July 2025

# Plotting backend and general appearance settings
using Plots; gr()
# Set default plot aesthetics
Plots.default(
    show=true, dpi=800, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
# Data I/O and numerical tools
using MAT
using LinearAlgebra
using ImageFiltering, FFTW
using DataStructures
using Statistics, StatsBase
using BSplineKit, Optim
# Aesthetics and output formatting
using Colors, ColorSchemes
using Printf, LaTeXStrings, PrettyTables
using CSV, DataFrames
# Time-stamping/logging
using Dates
# Set the working directory to the current location
cd(dirname(@__FILE__)) 
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
# Timestamp start for execution timing
t_start = Dates.now()

# -------------------------------
# FUNCTIONS
# -------------------------------
function compute_weights(x_array, λ0)
    """
    For BSplineKit fitting, compute weights for the B-spline fit.
    Compute uniform weights scaled by (1 - λ0). Returns an array of the same size as `x_array`.
    """
    return (1 - λ0) * fill!(similar(x_array), 1)
end

function normalize_image(img; method::String = "none")
    """
        normalize_image(img; method="none")

    Normalize a 2D image using the specified method.

    # Arguments
    - `img::AbstractMatrix{<:Real}`: Input image
    - `method::String`: Normalization method:
        - `"none"`     → return original image
        - `"zscore"`   → zero mean, unit variance normalization
        - `"contrast"` → subtract local mean (high-pass) and normalize

    # Returns
    - Normalized image of type `Matrix{Float64}`
    """

    img_f = float.(img)  # Ensure Float64

    if method == "none"
        return img_f

    elseif method == "zscore"
        μ = mean(img_f)
        σ = std(img_f)
        return (img_f .- μ) ./ (σ + eps())  # avoid division by 0

    elseif method == "contrast"
        if ndims(img_f) == 2
            # 2D Gaussian kernel
            kernel = Kernel.gaussian(5)
        elseif ndims(img_f) == 1
            # 1D Gaussian kernel
            kernel = KernelFactor.gaussian(5)
        else
            error("Unsupported input dimension: $(ndims(img_f))")
        end

        blurred = imfilter(img_f, kernel)
        highpass = img_f .- blurred
        μ = mean(highpass)
        σ = std(highpass)
        return (highpass .- μ) ./ (σ + eps())

    else
        error("Unknown normalization method: '$method'. Use 'none', 'zscore', or 'contrast'.")
    end
end

function estimate_shift_fft(img1, img2; Nmethod::String = "none")
    """
    estimate_shift(img1::AbstractMatrix{<:Real}, img2::AbstractMatrix{<:Real})

    Estimate the relative displacement (dx, dy) between two images using 
    phase correlation via FFT and subpixel refinement using quadratic fitting.

    Returns a tuple (dx, dy) in pixels representing how much `img2` is shifted
    relative to `img1`.
    """
    @assert size(img1) == size(img2) "Images must have the same dimensions"
    # Convert images to Float64 for numerical accuracy
    img1_f = float.(img1)
    img2_f = float.(img2)

    # Normalize: zero mean, unit variance
    img1_f = normalize_image(img1_f; method=Nmethod)
    img2_f = normalize_image(img2_f; method=Nmethod)

    # Compute cross-correlation via FFT:
    # Cross-correlation in frequency domain: inverse FFT of f1 * conj(f2)
    xcorr = fftshift(real(ifft(fft(img1_f) .* conj(fft(img2_f)))))
    # xcorr = imfilter(img1_f, reverse(img2_f), "reflect")

    # Visualize the cross-correlation surface
    fig1=heatmap(img1_f, colorbar=true, title="2D Normalized Signal 1")
    fig2=heatmap(img2_f, colorbar=true, title="2D Normalized Signal 2")
    fig3=heatmap(xcorr, colorbar=true, title="2D Normalized Cross-Correlation")
    fig = plot(fig1,fig2,fig3,
       layout=@layout([a1 a2 a3]),
       size=(1200,350),
       ) |> display

    # Find the peak (maximum correlation value), indicating best alignment
    max_idx = argmax(xcorr)  # linear index of peak
    peak_y, peak_x = Tuple(CartesianIndices(xcorr)[max_idx])  # convert to (row, col)

    dx_sub = dy_sub = 0.0  # initialize subpixel offsets

    # If the peak is not too close to the image edge, perform subpixel refinement
    if 2 ≤ peak_y ≤ size(xcorr, 1) - 1 && 2 ≤ peak_x ≤ size(xcorr, 2) - 1
        # Extract a 3×3 patch around the peak
        patch = @view xcorr[peak_y-1:peak_y+1, peak_x-1:peak_x+1]

        # Construct design matrix A and vector b for quadratic surface fit
        # We fit a quadratic f(x, y) = ax² + by² + cxy + dx + ey + f
        A = zeros(9, 6)
        b = zeros(9)
        k = 1
        for j in -1:1, i in -1:1
            A[k, :] .= (i^2, j^2, i*j, i, j, 1.0)  # coordinates and polynomial terms
            b[k] = patch[j+2, i+2]                # center of patch is at (2, 2)
            k += 1
        end

        # Solve least squares problem A * coeffs ≈ b
        coeffs = A \ b
        a, b2, c, d, e, _ = coeffs  # unpack coefficients of quadratic fit

        # Compute the gradient and Hessian of the fitted quadratic surface
        H = [2a c; c 2b2]           # Hessian matrix of second derivatives
        g = [-d; -e]                # Gradient vector (first derivatives)

        # Solve for subpixel offset: H⁻¹ * g gives the location of the peak
        sub_offset = H \ g
        dx_sub, dy_sub = sub_offset
    else
        @warn "Peak too close to border for subpixel refinement"
    end

    # Center of cross-correlation image (after fftshift) is considered zero shift
    center_y = size(xcorr, 1) ÷ 2 + 1
    center_x = size(xcorr, 2) ÷ 2 + 1

    # Total shift = (peak position + subpixel offset) - center
    dy_total = (peak_y + dy_sub) - center_y
    dx_total = (peak_x + dx_sub) - center_x

    return dx_total, dy_total
end

function estimate_1d_shift_fft(signal1::AbstractVector, signal2::AbstractVector; Nmethod::String = "none")
    """
    Estimate relative shift between two 1D signals using FFT-based cross-correlation.
    Returns subpixel shift (float), representing how much signal2 is shifted relative to signal1.
    """
    @assert length(signal1) == length(signal2) "Signals must be the same length"
    N = length(signal1)
    # Convert images to Float64 for numerical accuracy
    signal1 = float.(signal1)
    signal2 = float.(signal2)

    # Normalize signals: zero mean, unit variance
    signal1 = normalize_image(signal1; method=Nmethod)
    signal2 = normalize_image(signal2; method=Nmethod)

    # FFT-based cross-correlation
    f1 = fft(signal1)
    f2 = fft(signal2)

    xcorr = fftshift(real(ifft(f1 .* conj(f2))))

    fig1 = plot(signal1, label=false, title="Profile Signal 1");
    fig2 = plot(signal2, label=false, title="Profile Signal 2") ;
    fig3 = plot(xcorr,title="1D Normalized Cross-correlation",label=false) ;
    fig = plot(fig1,fig2,fig3,
       layout=@layout([a1 a2 a3]),
       size=(1200,350),
       ) |> display

    # Find integer shift (peak)
    max_idx = argmax(xcorr)
    peak = max_idx
    center = N ÷ 2 + 1

    # Subpixel quadratic interpolation using 3-point parabolic fit
    if 2 ≤ peak ≤ N-1
        y1, y2, y3 = xcorr[peak-1], xcorr[peak], xcorr[peak+1]
        denom = y1 - 2*y2 + y3
        subpixel_offset = abs(denom) > eps() ? 0.5 * (y1 - y3) / denom : 0.0
    else
        @warn "Peak at edge — subpixel refinement not possible"
        subpixel_offset = 0.0
    end

    # Total shift from center
    total_shift = (peak + subpixel_offset) - center
    return total_shift
end

function linear_fit(data::AbstractMatrix)
    """
    Fit a linear model y ≈ a * x + b 

    Returns: fit_function
    """
    # Extract last n rows for selected columns
    x = data[:, 1]
    y = data[:, 2]

    # Build design matrix X = [x 1]
    X = [x ones(length(x))]

    # Solve least squares: X * [a; b] ≈ y
    a, b = X \ y

    # Return a, b, and a callable function fit(x) = a*x + b
    return a,b,x -> a * x + b
end

# Compute pixel positions after binning, centered
function pixel_positions(img_size::Int, bin_size::Int, pixel_size::Float64)
    @assert img_size % bin_size == 0 "img_size must be divisible by bin_size"
    n_pixels = div(img_size,bin_size)
    effective_size = bin_size * pixel_size
    return effective_size .* (1:n_pixels) .- effective_size / 2
end

function process_mean_maxima(signal_key::String, data, n_bins; save=false, half_max=false)
    """
    Processes the mean z-profile of either the F1 or F2 signal across current settings.
    Arguments:
    - signal_key: "F1" or "F2"
    - data: structured dataset containing processed images
    - n_bins: number of bins for z-position averaging
    - save: whether to save the plots to disk

    Steps:
    - Averages the intensity per frame
    - Fits a smoothing spline
    - Locates the primary peak
    - Plots both raw and processed signals with spline fit
    """
    λ0 = 0.01  # Smoothing parameter for B-spline fitting
    Icoils = vec(data["data"]["Current_mA"])
    nI = length(Icoils)
    
    # Validate signal_key
    signal_label = signal_key == "F1" ? "F1ProcessedImages" :
                   signal_key == "F2" ? "F2ProcessedImages" :
                   error("Invalid signal_key: choose 'F1' or 'F2'")

    peak_positions = zeros(nI)

    for j in 1:nI
        # Get spatial z-coordinates (in mm)
        z_coord = 1e3 .* pixel_positions(2560,n_bins,cam_pixelsize)  

        # Load 3D image data for current j
        signal = Float64.(data["data"][signal_label][1,j])
        n_sig = size(signal, 3)  # Number of frames in the signal

        # Compute signal profile for each frame
        signal_profiles = [vec(mean(signal[:, :, i], dims=1)) for i in 1:n_sig]
        signal_profiles = reduce(hcat, signal_profiles)';  # Each row corresponds to one frame

        # Compute overall mean signal
        signal_mean = vec(mean(dropdims(mean(signal,dims=3); dims=3), dims=1))

        # Bin signal by reshaping and averaging
        @assert length(signal_mean) % n_bins == 0 "Signal length not divisible by n_bins"
        binned = reshape(signal_mean, n_bins, :)  # now shape is (2, 1280)
        signal_mean = vec(mean(binned, dims=1))

        # Optional background subtraction (currently placeholder)
        processed_signal = signal_mean  # Modify here if subtracting background

        # --- Plot raw signal per frame + mean ---
        fig_00 = plot(
            xlabel = L"$z$ (mm)",
            ylabel = "Intensity (a.u.)",
            title = L"%$(signal_key) Raw Signal: $I_{c} = %$(round(Icoils[j],digits=3))\mathrm{mA}$",
        );
        # Color palette for plotting frames
        colors = palette(:phase, n_sig);
        for i in 1:n_sig
            plot!( 1e3 .* pixel_positions(2560,1,cam_pixelsize) , signal_profiles[i, :], label=false, line=(:solid, colors[i], 1))
        end
        plot!(z_coord, processed_signal, label="mean (bins=$(n_bins))", line=(:solid, :black, 2));
        # display(fig_00);
        # savefig(fig_00, joinpath(dir_path, "$(signal_key)_I$( @sprintf("%02d", j))_raw.png" ))

        if half_max == true
            ymax = maximum(processed_signal)
            indices = findall(yi -> yi > ymax / 2, processed_signal)
            z_coord = z_coord[indices]
            processed_signal = processed_signal[indices]
        end

        # --- Fit smoothing spline to processed signal ---
        S_fit = BSplineKit.fit(BSplineOrder(4), z_coord, processed_signal, λ0; weights=compute_weights(z_coord,λ0))

        # Define negative spline function for finding maxima via minimization
        negative_spline(x) = -S_fit(x[1])

        # # Initial guesses for maxima (to increase robustness using quantiles and extrema)
        initial_guesses = sort([
            ceil(minimum(z_coord)),
            quantile(z_coord, 0.4),
            z_coord[argmax(processed_signal)],
            quantile(z_coord, 0.65),
            quantile(z_coord, 0.75),
            quantile(z_coord, 0.90),
            floor(maximum(z_coord))
        ])

        # Optimize to find maxima from initial guesses
        minima_candidates = Float64[]
        for guess in initial_guesses
            opt_result = optimize(negative_spline, [minimum(z_coord)], [maximum(z_coord)], [guess], Fminbox(LBFGS()))
            push!(minima_candidates, Optim.minimizer(opt_result)[1])
        end
        # Sort and filter close minima (tolerance 1e-9)
        sort!(minima_candidates)
        filtered_minima = [minima_candidates[1]]
        for min in minima_candidates[2:end]
            if all(abs(min - x) > 1.0e-9 for x in filtered_minima)
                push!(filtered_minima, min)
            end
        end

        # Evaluate spline at filtered minima and sort by descending function value (maxima)
        minima_values = -S_fit.(filtered_minima)
        sorted_indices = sortperm(minima_values)
        minima = filtered_minima[sorted_indices]
        minima_values = minima_values[sorted_indices]

        # Store primary peak position (max) 
        peak_positions[j] = minima[1]

        # --- Plot processed signal with spline fit and maximum ---
        fig_01 = plot(
            z_coord,
            processed_signal,
            title = L"%$(signal_key) Processed Signal: $I_{c} = %$(round(Icoils[j],digits=3))\mathrm{mA}$",
            label = "$(signal_key) processed",
            seriestype = :scatter,
            marker = (:white, 2),
            markerstrokecolor = :gray36,
            markerstrokewidth = 0.8,
            xlabel = L"$z \ (\mathrm{mm})$",
            ylabel = "Intensity (a.u.)",
            legend = :topleft,
        );
        xxs = collect(range(minimum(z_coord), maximum(z_coord), length=2000));
        plot!(xxs, S_fit.(xxs), line=(:solid, :red, 2), label="Spline fitting");
        vline!([minima[1]], line=(:dash, :black, 1), label=L"$z_{\mathrm{max}}= %$(round(minima[1], digits=3))\mathrm{mm}$");
        # display(fig_01);
        # savefig(fig_01, joinpath(dir_path, "$(signal_key)_I$(@sprintf("%02d", j))_processed.png" ))

        # --- Display side-by-side plot ---
        fig =plot(fig_00,fig_01,
        layout=@layout([a1 a2]),
        size=(900,400),
        left_margin=3mm,
        bottom_margin=3mm,
        # link=:x,
        )
        display(fig)
        
        # --- Save if needed ---
        save && savefig(fig, joinpath(dir_path, "m_$(signal_key)_I$(@sprintf("%02d", j)).png" ))

    end
    return peak_positions
end

function process_framewise_maxima(signal_key::String, data, n_bins; save=false, half_max= false)
    """
    Process per-frame z-position of intensity maxima for a given signal key (e.g., "F1", "F2").
    - Fits B-splines to the averaged intensity profiles (along x-direction).
    - Searches for local maxima using multi-start optimization.
    - Returns a (n_runs_max × nI) matrix where each entry contains the position of the most prominent maximum in mm,
    and is NaN if no data exists for that run/current.
    """
    λ0 = 0.01                                # Smoothing parameter for B-spline
    Icoils = vec(data["data"]["Current_mA"])
    nI = length(Icoils)                      # Number of current settings
    # Get spatial z-coordinates (in mm)
    z_coord = 1e3 .* pixel_positions(2560,n_bins,cam_pixelsize)

    # Validate signal_key
    signal_label = signal_key == "F1" ? "F1ProcessedImages" :
                   signal_key == "F2" ? "F2ProcessedImages" :
                   error("Invalid signal_key: choose 'F1' or 'F2'")


    # Determine maximum number of runs across all files
    n_runs_max = maximum([size(data["data"][signal_label][1,i], 3) for i=1:nI])
    max_position_data = fill(NaN, n_runs_max, nI)

    @info "Processing per-frame maxima" signal_label=signal_label
    for j in 1:nI

        signal = Float64.(data["data"][signal_label][1,j])
        n_runs = size(signal, 3)  # Number of frames in the signal

        colors = palette(:phase, n_runs)

        for i in 1:n_runs

            signal_profile = vec(mean(signal[:, :, i], dims=1))

            # Bin signal by reshaping and averaging
            @assert length(signal_profile) % n_bins == 0 "Signal length not divisible by n_bins"
            binned = reshape(signal_profile, n_bins, :)  # now shape is (2, 1280)
            signal_profile = vec(mean(binned, dims=1))

            processed = signal_profile

            if half_max == true
                ymax = maximum(processed_signal)
                indices = findall(yi -> yi > ymax / 2, processed_signal)
                z_coord = z_coord[indices]
                processed_signal = processed_signal[indices]
            end

            # Fit B-spline to the processed signal
            S_fit = BSplineKit.fit(
                BSplineOrder(4), z_coord, processed, λ0;
                weights=compute_weights(z_coord, λ0)
            )

            # Define negative spline for maxima search
            negative_spline(x) = -S_fit(x[1])

            initial_guesses = sort([
                ceil(minimum(z_coord)),
                quantile(z_coord, 0.4),
                z_coord[argmax(processed)],
                quantile(z_coord, 0.65),
                quantile(z_coord, 0.75),
                quantile(z_coord, 0.90),
                floor(maximum(z_coord))
            ])

            # Perform local optimizations to find candidate maxima
            maxima_candidates = Float64[]
            for guess in initial_guesses
                opt_result = optimize(
                    negative_spline,
                    [minimum(z_coord)], [maximum(z_coord)], [guess],
                    Fminbox(LBFGS())
                )
                push!(maxima_candidates, Optim.minimizer(opt_result)[1])
            end
            sort!(maxima_candidates)

            # Filter out near-duplicates (within 1e-9)
            filtered_candidates = [maxima_candidates[1]]
            for val in maxima_candidates[2:end]
                if all(abs(val - x) > 1e-9 for x in filtered_candidates)
                    push!(filtered_candidates, val)
                end
            end

            # Rank maxima candidates by actual spline value
            candidate_values = -S_fit.(filtered_candidates)
            best_index = argmin(candidate_values)
            max_z = filtered_candidates[best_index]

            # Store result
            max_position_data[i, j] = max_z

            # Plot
            fig_signal = plot(
                z_coord, processed,
                title = L"%$(signal_key) Signal: $I_{c} = %$(round(Icoils[j], digits=3))\mathrm{mA}$",
                label = "Run $i",
                seriestype = :scatter,
                marker = (:white, 2),
                markerstrokecolor = :gray36,
                markerstrokewidth = 0.8,
                xlabel = L"$z \ (\mathrm{mm})$",
                ylabel = "Intensity (a.u.)",
                legend = :topleft,
            )
            z_fit = range(extrema(z_coord)..., length=2000)
            plot!(z_fit, S_fit.(z_fit), line=(:solid, :red, 2), label="Spline fitting")
            vline!([ max_z], line=(:dash, :black, 1), label=L"$z_{\mathrm{max}}= %$(round( max_z, digits=3))\mathrm{mm}$")
            display(fig_signal)
            save && savefig(fig_signal, joinpath(dir_path, "fw$(i)_$(signal_key)_I$(@sprintf("%02d", j)).png"))


        end
    end

    return max_position_data
end

# Previous experiment data for comparison
data_JSF = OrderedDict(
    :exp => hcat(
    [0.0200, 0.0300, 0.0500, 0.1500, 0.2000, 0.2500, 0.3500, 0.5000, 0.7500],
    [0.0229, 0.0610, 0.1107, 0.3901, 0.5122, 0.6315, 0.8139, 1.1201, 1.5738]),
    :model => hcat(
    [0.0150, 0.0200, 0.0250, 0.0300, 0.0400, 0.0500, 0.0700, 0.1000, 0.1500, 0.2000, 0.2500, 0.3500, 0.5000, 0.7500],
    [0.0409, 0.0566, 0.0830, 0.1015, 0.1478, 0.1758, 0.2409, 0.3203, 0.4388, 0.5433, 0.6423, 0.8394, 1.1267, 1.5288],
    [0.0179, 0.0233, 0.0409, 0.0536, 0.0883, 0.1095, 0.1713, 0.2487, 0.3697, 0.4765, 0.5786, 0.7757, 1.0655, 1.4630])
)

# Data Directory
data_directory = "20250725/" ;

# Generate a timestamped directory name for output (e.g., "20250718T153245")
directoryname = Dates.format(t_start, "yyyymmddTHHMMSS") ;
# Construct the full directory path (relative to current working directory)
dir_path = "./results_data/$(directoryname)" ;
# Create the directory (and any necessary parent folders)
mkpath(dir_path) ;
@info "Created output directory" dir = dir_path

# STERN–GERLACH EXPERIMENT SETUP
# Camera and pixel geometry
cam_pixelsize = 6.5e-6 ;            # Physical pixel size of camera [m]
exp_bin = 1 ;                       # Camera binning
exp_pixelsize = exp_bin*cam_pixelsize ; # Effective pixel size after binning [m]
# Image dimensions (adjusted for binning)
x_pixels = Int(2160 / exp_bin)  # Number of x-pixels after binning
z_pixels = Int(2560 / exp_bin)  # Number of z-pixels after binning
# Spatial axes shifted to center the pixels
x_position = pixel_positions(2160, exp_bin, cam_pixelsize)
z_position = pixel_positions(2560, exp_bin, cam_pixelsize)
# Binning for the analysis
n_bins = 1
# Read data
data = matread(joinpath(data_directory, "data.mat")) 

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
vspan!([0, findlast(<(0), reverse(-Icoils))+0.5 ], color=:gray, alpha=0.30,label="unresolved" );
display(fig_I0)


# SIGNAL
##########################################################################################
##########################################################################################
# Run for F1 and F2 signals: MEAN OF FRAMES
##########################################################################################
##########################################################################################

f1_data_mean = process_mean_maxima("F1", data, n_bins )
f2_data_mean = process_mean_maxima("F2", data, n_bins )

data_centroid = (f1_data_mean .+ f2_data_mean)/2
centroid_mean = mean(data_centroid, Weights(nI-1:-1:0))
centroid_std = std(data_centroid, Weights(nI-1:-1:0); corrected=false) / sqrt(nI)
plot(abs.(Icoils), data_centroid,
label=false,
color=:purple,
marker=(:cross,5),
line=(:solid,1),
xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$"),
xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlim=(1e-3,1),
yaxis = L"$z_{0} \ (\mathrm{mm})$",
)
hline!([centroid_mean], label=L"Centroid $z=%$(round(centroid_mean,digits=3))$mm")
hspan!( [centroid_mean - centroid_std,centroid_mean + centroid_std], color=:orangered, alpha=0.30, label=L"St.Err. = $\pm%$(round(centroid_std,digits=3))$mm")


data_mean = hcat( # [I_coil (mA), F1_z_peak (mm), F2_z_peak (mm), Δz (mm), F1_z_centered (mm), F2_z_centered (mm)]
    -1000*Icoils, 
    f1_data_mean, 
    f2_data_mean, 
    f1_data_mean .- f2_data_mean, 
    f1_data_mean .- f1_data_mean[end], 
    f2_data_mean .- f2_data_mean[end],
    f1_data_mean .- centroid_mean, 
    f2_data_mean .- centroid_mean
)  
reverse!(data_mean, dims=1)

pretty_table(
    data_mean;
    formatters    = (ft_printf("%8.3f",1), ft_printf("%8.5f",2:6)),
    alignment=:c,
    header        = (
        ["Current", "F1 z", "F2 z", "Δz", "Centered F1 z","Centered F2 z", "Centroid F1 z","Centroid F2 z"], 
        ["[mA]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
        ),
    border_crayon = crayon"blue bold",
    tf            = tf_unicode_rounded,
    header_crayon = crayon"yellow bold",
    equal_columns_width= true,
)


fig_01 = plot(abs.(data_mean[:,1]/1000), data_mean[:,2],
    label=L"$F_{1}$",
    line=(:solid,:red,2),
)
plot!(abs.(data_mean[:,1]/1000), data_mean[:,3],
    label=L"$F_{2}$",
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
vspan!([1e-8, abs(data_mean[findlast(<(0), data_mean[:,1])+1,1]/1000)], color=:gray, alpha=0.30,label="zero" )
plot!(abs.(data_mean[:,1]/1000), data_mean[:,2], fillrange=data_mean[:,3],
    fillalpha=0.2,
    color=:purple,
    label = false,
)
hline!([centroid_mean], line=(:dot,:black,2), label="Centroid")

fig_02 = plot(abs.(data_mean[:,1]/1000), data_mean[:,2],
    label=L"$F_{1}$",
    line=(:solid,:red,2),
)
plot!(abs.(data_mean[:,1]/1000), 2*centroid_mean .- data_mean[:,3],
    label=L"Centroid Mirrored $F_{2}$",
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
vspan!([1e-8, abs(data_mean[findlast(<(0), data_mean[:,1])+1,1]/1000)], color=:gray, alpha=0.30,label="zero" )
# Fill between y1 and y2
plot!(abs.(data_mean[:,1]/1000), data_mean[:,2], fillrange=2*centroid_mean .- data_mean[:,3],
    fillalpha=0.2,
    color=:purple,
    label = false,
)
hline!([centroid_mean], line=(:dot,:black,2), label="Centroid")

fig=plot(fig_01, fig_02,
layout=@layout([a ; b]),
share=:x,
)
plot!(fig[1], xlabel = "", xformatter=_->"")
plot!(fig[2], title = "", top_margin = -9mm)
display(fig)

fig=plot(
    data_mean[2:end, 1]/1000, abs.(data_mean[2:end, 5]),
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
hspan!([1e-6,1000*n_bins* cam_pixelsize], color=:gray, alpha=0.30, label="Pixel size" )
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
savefig(fig, joinpath(dir_path, "mean.png"))


# Compute absolute values for plotting
y = data_mean[:,7]
y_abs = abs.(y)
# Create masks for negative and non-negative values
neg_mask = y .< 0
pos_mask = .!neg_mask
fig=plot(
    abs.(data_mean[pos_mask, 1]/1000), y_abs[pos_mask],
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
plot!(abs.(data_mean[neg_mask,1]/1000), y_abs[neg_mask], 
    label=false, 
    seriestype=:scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :chocolate4,
    markerstrokewidth = 2,
)
hspan!([1e-6,1000*n_bins* cam_pixelsize], color=:gray, alpha=0.30, label="Pixel size" )
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



##########################################################################################
##########################################################################################
# Run for F1 and F2 signals: : FRAMEWISE
##########################################################################################
##########################################################################################

F1_data_framewise = process_framewise_maxima("F1", data, n_bins)
F2_data_framewise = process_framewise_maxima("F2", data, n_bins)

data_centroid = vec(mean((F1_data_framewise .+ F2_data_framewise)/2, dims=1))
centroid_mean = mean(data_centroid, Weights(nI-1:-1:0))
centroid_std = std(data_centroid, Weights(nI-1:-1:0); corrected=false) / sqrt(nI)
plot(abs.(Icoils), data_centroid,
label=false,
color=:purple,
marker=(:cross,5),
line=(:solid,1),
xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$"),
xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
              [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlim=(1e-3,1),
yaxis = L"$z_{0} \ (\mathrm{mm})$",
)
hline!([centroid_mean], label=L"Centroid $z=%$(round(centroid_mean,digits=3))$mm")
hspan!( [centroid_mean - centroid_std,centroid_mean + centroid_std], color=:orangered, alpha=0.30, label=L"St.Err. = $\pm%$(round(centroid_std,digits=3))$mm")


data_framewise = hcat( 
    # [I_coil (mA), F1_z_peak (mm), Error, F2_z_peak (mm), Error, Δz (mm), Error, F1_z_centered (mm), Error, F2_z_centered (mm), Error]
    -1000*Icoils, 
    vec(mean(F1_data_framewise,dims=1)), 
    vec(std(F1_data_framewise,dims=1)) ./ sqrt(size(F1_data_framewise,1)) , 

    vec(mean(F2_data_framewise,dims=1)), 
    vec(std(F2_data_framewise,dims=1)) ./ sqrt(size(F2_data_framewise,1)) ,
    
    vec(mean(F1_data_framewise,dims=1)) .- vec(mean(F2_data_framewise,dims=1)),
    sqrt.( vec(std(F1_data_framewise,dims=1)).^2 .+ vec(std(F2_data_framewise,dims=1)).^2 ) ./ sqrt(size(F1_data_framewise,1)) ,
    
    vec(mean(F1_data_framewise,dims=1)) .- vec(mean(F1_data_framewise,dims=1))[end],    
    sqrt.( vec(std(F1_data_framewise,dims=1)).^2 .+ vec(std(F1_data_framewise,dims=1))[end].^2 ) ./ sqrt(size(F1_data_framewise,1)),
        
    vec(mean(F2_data_framewise,dims=1)) .- vec(mean(F2_data_framewise,dims=1))[1],
    sqrt.( vec(std(F2_data_framewise,dims=1)).^2 .+ vec(std(F2_data_framewise,dims=1))[end].^2 ) ./sqrt(size(F2_data_framewise,1)),

    vec(mean(F1_data_framewise,dims=1)) .- centroid_mean,
    sqrt.( vec(std(F1_data_framewise,dims=1)/size(F1_data_framewise,1)).^2  .+ centroid_std.^2  ),

    vec(mean(F2_data_framewise,dims=1)) .- centroid_mean,
    sqrt.( vec(std(F2_data_framewise,dims=1)./size(F2_data_framewise,1)).^2  .+ centroid_std.^2  )
 ) 
reverse!(data_framewise, dims=1)

pretty_table(
    data_framewise;
    formatters    = (ft_printf("%8.3f",1), ft_printf("%8.5f",2:11)),
    alignment=:c,
    header        = (
        ["Current", "F1 z", "Std.Dev.",  "F2 z", "Std.Dev.", "Δz", "Std.Dev.", "Centered F1 z", "Std.Dev.", "Centered F2 z", "Std.Dev.", "Centroid F1 z", "Std.Dev.", "Centroid F2 z", "Std.Dev."], 
        ["[mA]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
        ),
    border_crayon = crayon"green bold",
    tf            = tf_unicode_rounded,
    header_crayon = crayon"yellow bold",
    equal_columns_width= true,
)


fig_01 = plot(abs.(data_framewise[:,1]/1000), data_framewise[:,2],
    ribbon= data_framewise[:, 3],
    label=L"$F_{1}$",
    line=(:solid,:red,1),
    fillalpha=0.23, 
    fillcolor=:red,  
)
plot!(abs.(data_framewise[:,1]/1000), data_framewise[:,2], fillrange=data_framewise[:,4],
    fillalpha=0.2,
    color=:orange,
    label = false,
)
plot!(abs.(data_framewise[:,1]/1000), data_framewise[:,4],
    ribbon= data_framewise[:,5],
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
vspan!([1e-8, abs(data_framewise[findlast(<(0), data_framewise[:,1])+1,1]/1000)], color=:gray, alpha=0.30,label="zero" )
hline!([centroid_mean], line=(:dot,:black,2), label="Centroid")


fig_02 = plot(abs.(data_framewise[:,1]/1000), data_framewise[:,2],
    ribbon= data_framewise[:, 3],
    label=L"$F_{1}$",
    line=(:solid,:red,2),
)
plot!(abs.(data_framewise[:,1]/1000), 2*centroid_mean .- data_framewise[:,4],
    ribbon= data_framewise[:, 5],
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
vspan!([1e-8, abs(data_framewise[findlast(<(0), data_mean[:,1])+1,1]/1000)], color=:gray, alpha=0.30,label="zero" )
# Fill between y1 and y2
plot!(abs.(data_framewise[:,1]/1000), data_framewise[:,2], fillrange=2*centroid_mean .- data_framewise[:,4],
    fillalpha=0.2,
    color=:purple,
    label = false,
)
hline!([centroid_mean], line=(:dot,:black,2), label="Centroid")

fig=plot(fig_01, fig_02,
layout=@layout([a ; b]),
share=:x,
)
plot!(fig[1], xlabel = "", xformatter=_->"")
plot!(fig[2], title = "", top_margin = -9mm)
display(fig)

fig=plot(
    data_framewise[2:end, 1]/1000, abs.(data_framewise[2:end, 8]),
    yerror = data_framewise[2:end, 9],
    label = "20250725",
    seriestype = :scatter,
    marker = (:circle, :white, 2),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
)
plot!(
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-5,1.5),
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
hspan!([1e-6,1000*n_bins* cam_pixelsize], color=:gray, alpha=0.30, label="Pixel size" )
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
savefig(fig, joinpath(dir_path, "framewise.png"))


# Compute absolute values for plotting
y = data_framewise[:,12]
y_abs = abs.(y)
# Create masks for negative and non-negative values
neg_mask = y .< 0
pos_mask = .!neg_mask
fig=plot(
    abs.(data_framewise[pos_mask, 1]/1000), y_abs[pos_mask],
    yerror = data_framewise[pos_mask, 13],
    label = "20250725",
    seriestype = :scatter,
    marker = (:circle, :white, 2),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
)
plot!(abs.(data_framewise[neg_mask,1]/1000), y_abs[neg_mask],
    yerror = data_framewise[neg_mask, 13],
    label=false, 
    seriestype=:scatter,
    marker = (:circle, :white, 2),
    markerstrokecolor = :chocolate4,
    markerstrokewidth = 2,
)
plot!(
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
    ylims = (1e-5,1.5),
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
hspan!([1e-6,1000*n_bins* cam_pixelsize], color=:gray, alpha=0.30, label="Pixel size" )
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


# Comparison with Xukun's analysis
fig=plot(
    abs.(data_framewise[1:end, 1]/1000), abs.(data_framewise[1:end, 6]),
    yerror = data_framewise[2:end, 7],
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
plot!(abs.(vec(data["data"]["Current_mA"])/1000), vec(data["data"]["PeakSeparationMean_um"]/1000),
yerr=vec(data["data"]["PeakSeparationStd_um"])/1000,
label="XL",
# seriestype=:scatter,
line=(:solid,:purple,1),
marker=(:square,:white,1),
markerstrokewidth=1,
markerstrokecolor=:purple
)

t_run = Dates.canonicalize(Dates.now()-t_start)
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
plot!(abs.(data_framewise[:, 1]/1000), abs.(data_framewise[:, 8]),
    yerror = data_framewise[:, 9],
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
kis = [1.50,1.80,2.00,2.10,2.20,2.25,2.30,2.40,2.50,2.60] # ×10^-6
# Compute absolute values for plotting
y = data_framewise[:,12]
y_abs = abs.(y)
# Create masks for negative and non-negative values
neg_mask = y .< 0
pos_mask = .!neg_mask
fig=plot(
    abs.(data_framewise[pos_mask, 1]/1000), y_abs[pos_mask],
    yerror = data_framewise[pos_mask, 13],
    label = "20250725",
    seriestype = :scatter,
    marker = (:circle, :white, 2),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
)
plot!(abs.(data_framewise[neg_mask,1]/1000), y_abs[neg_mask],
    yerror = data_framewise[neg_mask, 13],
    label=false, 
    seriestype=:scatter,
    marker = (:circle, :white, 2),
    markerstrokecolor = :chocolate4,
    markerstrokewidth = 2,
)
plot!(
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    # yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
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


simqm = CSV.read("./simulation_data/results_QM_20250728T105702.csv",DataFrame; header=false)
############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
