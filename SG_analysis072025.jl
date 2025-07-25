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
using Statistics
using BSplineKit, Optim
# Aesthetics and output formatting
using Colors, ColorSchemes
using Printf, LaTeXStrings, PrettyTables
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
        a, b_, c, d, e, _ = coeffs  # unpack coefficients of quadratic fit

        # Compute the gradient and Hessian of the fitted quadratic surface
        H = [2a c; c 2b_]           # Hessian matrix of second derivatives
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

function process_mean_maxima(signal_key::String, Icoils, matched_files, z_position, bg_signal; save=false)
    """
    This function processes the mean z-profile of either the F1 or F2 signal across current settings.
    - Averages the intensity per frame
    - Subtracts the background
    - Fits a smoothing spline
    - Locates the primary peak
    - Plots both raw and processed signals with spline fit
    """
    λ0 = 0.01  # Smoothing parameter for B-spline fitting
    nI = length(Icoils)
    z_coord = 1e3 .* collect(z_position)  # Convert z_position to mm for fitting

    peak_positions = zeros(nI)

    for j in 1:nI
        signal = matread(matched_files[j])
        signal_data = Float64.(signal[signal_key])

        n_sig = size(signal_data, 3)  # Number of frames in the signal

        # @info "Processing data" coil_current = Icoils[j], signal_key = signal_key

        # Mean intensity along z for each frame (dim 1 and 2 averaged, leaving dim 3 as frames)
        sig_mean_profiles = [vec(mean(signal_data[:, :, i], dims=1)) for i in 1:n_sig]
        sig_mean_profiles = reduce(hcat, sig_mean_profiles)';  # Each row corresponds to one frame
        mean_signal = vec(mean(sig_mean_profiles, dims=1));    # Mean over frames

        # Color palette for plotting frames
        colors = palette(:phase, n_sig)

        # Plot raw signals per frame + mean
        fig_00 = plot(
            xlabel = L"$z$ (mm)",
            ylabel = "Intensity (a.u.)",
            title = L"%$(signal_key) Raw Signal: $I_{c} = %$(1000*Icoils[j])\mathrm{mA}$",
        )
        for i in 1:n_sig
            plot!( z_coord, sig_mean_profiles[i, :], label=false, line=(:solid, colors[i], 1))
        end
        plot!(z_coord, mean_signal, label="mean", line=(:solid, :black, 2))
        display(fig_00)
        # savefig(fig_00, joinpath(dir_path, "$(signal_key)_I$( @sprintf("%02d", j))_raw.png" ))

        # Background subtraction
        processed_signal = mean_signal .- bg_signal
        

        # Fit B-spline with smoothing parameter and weights
        S_fit = BSplineKit.fit(BSplineOrder(4), z_coord, processed_signal, λ0; weights=compute_weights(z_coord,λ0))

        # Define negative spline function for finding maxima via minimization
        negative_spline(x) = -S_fit(x[1])

        # Initial guesses for minima search (using quantiles and extrema)
        initial_guesses = sort([
            ceil(minimum(z_coord)),
            quantile(z_coord, 0.4),
            z_coord[argmax(processed_signal)],
            quantile(z_coord, 0.65),
            quantile(z_coord, 0.75),
            quantile(z_coord, 0.90),
            floor(maximum(z_coord))
        ])

        # Find local maxima by minimizing negative spline with bounds
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

        # Plot processed signal with spline fit and peak positions
        fig_01 = plot(
            z_coord,
            processed_signal,
            title = L"%$(signal_key) Processed Signal: $I_{c} = %$(1000*Icoils[j])\mathrm{mA}$",
            label = "$(signal_key) processed",
            seriestype = :scatter,
            marker = (:white, 2),
            markerstrokecolor = :gray36,
            markerstrokewidth = 0.8,
            xlabel = L"$z \ (\mathrm{mm})$",
            ylabel = "Intensity (a.u.)",
            legend = :topleft,
        )
        xxs = collect(range(minimum(z_coord), maximum(z_coord), length=2000))
        plot!(xxs, S_fit.(xxs), line=(:solid, :red, 2), label="Spline fitting")
        vline!([minima[1]], line=(:dash, :black, 1), label=L"$z_{\mathrm{max}}= %$(round(minima[1], digits=3))\mathrm{mm}$")
        display(fig_01)
        # savefig(fig_01, joinpath(dir_path, "$(signal_key)_I$(@sprintf("%02d", j))_processed.png" ))

        fig =plot(fig_00,fig_01,
        layout=@layout([a1 a2]),
        size=(900,400),
        left_margin=3mm,
        bottom_margin=3mm,
        # link=:x,
        )
        display(fig)
        save && savefig(fig, joinpath(dir_path, "m_$(signal_key)_I$(@sprintf("%02d", j)).png" ))

    end
    return peak_positions
end

function process_framewise_maxima(signal_key::String, Icoils, matched_files, z_position, bg_signal; save=false)
    """
    Process per-frame z-position of intensity maxima for a given signal key (e.g., "F1", "F2").
    - Fits B-splines to the averaged intensity profiles (along x-direction).
    - Searches for local maxima using multi-start optimization.
    - Returns a (n_runs_max × nI) matrix where each entry contains the position of the most prominent maximum in mm,
    and is NaN if no data exists for that run/current.
    """
    λ0 = 0.01                                # Smoothing parameter for B-spline
    nI = length(Icoils)                      # Number of current settings
    z_coord = 1e3 .* collect(z_position)     # z-position in mm for readability in plots

    # Determine maximum number of runs across all files
    n_runs_max = maximum(size(matread(file)[signal_key], 3) for file in matched_files)
    max_position_data = fill(NaN, n_runs_max, nI)

    @info "Processing per-frame maxima" signal_key=signal_key
    for j in 1:nI
        signal = matread(matched_files[j])
        signal_data = Float64.(signal[signal_key])
        n_runs = size(signal_data, 3)  # Number of frames in the signal

        colors = palette(:phase, n_runs)

        for i in 1:n_runs
            signal_profile = vec(mean(signal_data[:, :, i], dims=1))
            processed = signal_profile .- bg_signal

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
                1e3*z_position, processed,
                title = L"%$(signal_key) Signal: $I_{c} = %$(1000*Icoils[j])\mathrm{mA}$",
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
data_directory = "20250712/" ;

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
exp_bin = 4 ;                       # Camera binning
exp_pixelsize = exp_bin*cam_pixelsize ; # Effective pixel size after binning [m]
# Image dimensions (adjusted for binning)
x_pixels = Int(2160 / exp_bin)  # Number of x-pixels after binning
z_pixels = Int(2560 / exp_bin)  # Number of z-pixels after binning
# Spatial axes shifted to center the pixels
x_position = exp_bin*cam_pixelsize*(1:1:x_pixels) .- exp_pixelsize/2
z_position = exp_bin*cam_pixelsize*(1:1:z_pixels) .- exp_pixelsize/2

# Use a regex to match filenames
files = readdir(data_directory; join=true)
# Extract coil currents (in mA) from filenames using regex
Icoils = collect(skipmissing(map(files) do f # (mA)
    m = match(r"I-(\d+)mA", f)
    m !== nothing ? parse(Int, m.captures[1]) : missing
end))
Icoils = Icoils / 1000 ; # Convert to A
nI = length(Icoils) ;    # Number of valid current files found


# BACKGROUND SIGNAL
# Load background data
background_file = matread(joinpath(data_directory, "background.mat")) ;     # Read background data from .mat file
bg = Float64.(background_file["Background"]) ;  # Convert to Float64
bg = dropdims(bg, dims=3) ;     # Removes the 3rd dimension
n_bg = size(bg, 3) ;          # Number of background frames

# Generate a perceptual color palette for plotting
colors = palette(:phase, n_bg) ;

# Preallocate and compute mean intensity along z for each frame
bg_mean_profiles = [vec(mean(bg[:, :, i], dims=1)) for i in 1:n_bg] ;
bg_mean_profiles = reduce(hcat, bg_mean_profiles)' ; # Stack as rows
bg_signal = vec(mean(bg_mean_profiles, dims=1)) ;    # Mean across frames

# Plot Background Signal
fig_bg = plot(
    xlabel=L"$z$ (mm)",
    ylabel="Intensity (a.u.)", 
    title="Background")
for i=1:n_bg
    plot!(1e3*z_position, bg_mean_profiles[i,:], label=false, line=(:solid, colors[i], 1))
end
plot!(1e3*z_position, bg_signal, label="Mean", line=(:dot, :black, 2))
relative_background_variation = (maximum(bg_signal) - minimum(bg_signal)) / maximum(bg_signal)
variation_str = @sprintf "Relative variation: %.2f%%" 100 * relative_background_variation
annotate!(fig_bg, 12, minimum(bg_signal), text(variation_str, :black, 10))
display(fig_bg)
savefig(fig_bg, joinpath(dir_path, "background.png"))

# SIGNAL
# Filter relevant .mat files with coil current info in filename
matched_files = filter(f -> endswith(f, ".mat") && occursin(r"I-\d+mA", basename(f)), files) ; 

##########################################################################################
##########################################################################################
# Run for F1 and F2 signals: MEAN OF FRAMES
##########################################################################################
##########################################################################################

f1_data_mean = process_mean_maxima("F1", Icoils, matched_files, z_position, bg_signal)
f2_data_mean = process_mean_maxima("F2", Icoils, matched_files, z_position, bg_signal)

data_mean = hcat( # [I_coil (mA), F1_z_peak (mm), F2_z_peak (mm), Δz (mm), F1_z_centered (mm), F2_z_centered (mm)]
    Icoils, 
    f1_data_mean, 
    f2_data_mean, 
    f1_data_mean .- f2_data_mean, 
    f1_data_mean .- f1_data_mean[1], 
    f2_data_mean .- f2_data_mean[1]
)  

pretty_table(
    data_mean;
    formatters    = (ft_printf("%8.3f",1), ft_printf("%8.5f",2:6)),
    alignment=:c,
    header        = (
        ["Current", "F1 z", "F2 z", "Δz", "Centered F1 z","Centered F2 z"], 
        ["[A]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
        ),
    border_crayon = crayon"blue bold",
    tf            = tf_unicode_rounded,
    header_crayon = crayon"yellow bold",
    equal_columns_width= true,
)

fig=plot(
    data_mean[2:end, 1], abs.(data_mean[2:end, 5]),
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.001,1.0),
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
    xticks = :log10,
    yticks = :log10,
    framestyle = :box,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=10,
)
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
a0,b0,fit_JSF = linear_fit(data_JSF[:exp][end-3:end,:])
a1,b1,fit_datamean = linear_fit(data_mean[end-9:end,[1,5]])
label1_str = @sprintf("z = %.3f \\, I + %.3f", a0, b0)
label2_str = @sprintf("z = %.3f \\, I + %.3f", a1, b1)
plot!(Icoils,fit_JSF.(Icoils), label=L"10142024: %$(label1_str)" )
plot!(Icoils,fit_datamean.(Icoils), label=L"07122025: %$(label2_str)" )
vspan!([0.0001,0.020], color=:gray, alpha=0.30,label="unresolved")
savefig(fig, joinpath(dir_path, "mean.png"))

# p = plot(
#     xlims = (0.01,1.0),
#     title = "F=1 Peak Position vs Current",
#     legend = :topleft,
#     grid = true,
#     minorgrid = true,
#     gridalpha = 0.5,
#     gridstyle = :dot,
#     minorgridalpha = 0.05,
#     xticks = :log10,
#     yticks = :log10,
#     framestyle = :box,
#     size=(800,600),
#     tickfontsize=11,
#     guidefontsize=14,
#     legendfontsize=12,
# )
# colors = palette(:viridis, 9) # Generate a color palette for the 9 currents
# for i = 1:8
#     data_mean = hcat( # [I_coil (mA), F1_z_peak (mm), F2_z_peak (mm), Δz (mm), F1_z_centered (mm), F2_z_centered (mm)]
#     Icoils, 
#     f1_data_mean, 
#     f2_data_mean, 
#     f1_data_mean .- f2_data_mean, 
#     f1_data_mean .- f1_data_mean[i], 
#     f2_data_mean .- f2_data_mean[i]
#     )
#     p = plot!(
#     data_mean[10:end, 1], data_mean[10:end, 5],
#     xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
#     yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
#     label = "07122025 : $( @sprintf("%.3f",Icoils[i]) ) A",
#     seriestype = :scatter,
#     marker = (:circle, :white, 4),
#     markerstrokecolor = colors[i],
#     markerstrokewidth = 2,

# )
# display(p)
# end  

##########################################################################################
##########################################################################################
# Run for F1 and F2 signals: : FRAMEWISE
##########################################################################################
##########################################################################################

F1_data_framewise = process_framewise_maxima("F1", Icoils, matched_files, z_position, bg_signal)
F2_data_framewise = process_framewise_maxima("F2", Icoils, matched_files, z_position, bg_signal)

data_framewise = hcat( 
    # [I_coil (mA), F1_z_peak (mm), Erro, F2_z_peak (mm), Error, Δz (mm), Error, F1_z_centered (mm), F2_z_centered (mm)]
    Icoils, 
    vec(mean(F1_data_framewise,dims=1)), 
    vec(std(F1_data_framewise,dims=1)), 
    vec(mean(F2_data_framewise,dims=1)), 
    vec(std(F2_data_framewise,dims=1)),
    vec(mean(F1_data_framewise,dims=1)) .- vec(mean(F2_data_framewise,dims=1)),
    sqrt.(vec(std(F1_data_framewise,dims=1)).^2 .+ vec(std(F2_data_framewise,dims=1)).^2),
    vec(mean(F1_data_framewise,dims=1)) .- vec(mean(F1_data_framewise,dims=1))[1],    
    sqrt.(vec(std(F1_data_framewise,dims=1)).^2 .+ vec(std(F1_data_framewise,dims=1))[1]^2),
    vec(mean(F2_data_framewise,dims=1)) .- vec(mean(F2_data_framewise,dims=1))[1],
    sqrt.(vec(std(F2_data_framewise,dims=1)).^2 .+ vec(std(F2_data_framewise,dims=1))[1]^2),
) 

pretty_table(
    data_framewise;
    formatters    = (ft_printf("%8.3f",1), ft_printf("%8.5f",2:11)),
    alignment=:c,
    header        = (
        ["Current", "F1 z", "Std.Dev.",  "F2 z", "Std.Dev.", "Δz", "Std.Dev.", "Centered F1 z", "Std.Dev.", "Centered F2 z", "Std.Dev."], 
        ["[A]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
        ),
    border_crayon = crayon"green bold",
    tf            = tf_unicode_rounded,
    header_crayon = crayon"yellow bold",
    equal_columns_width= true,
)


fig=plot(
    data_framewise[10:end, 1], data_framewise[10:end, 8],
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.01,1.0),
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
    xticks = :log10,
    yticks = :log10,
    framestyle = :box,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=12,
)
plot!(data_framewise[10:end, 1], data_framewise[10:end, 8],
    ribbon= data_framewise[10:end, 9],
    label= "Std", 
    fillalpha=0.5, fillcolor=:grey36, line=(:solid, :grey36, 0.2)
)
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
a0,b0,fit_JSF = linear_fit(data_JSF[:exp][end-3:end,:])
a1,b1,fit_dataframewise = linear_fit(data_framewise[end-9:end,[1,8]])
label1_str = @sprintf("z = %.3f \\, I + %.3f", a0, b0)
label2_str = @sprintf("z = %.3f \\, I + %.3f", a1, b1)
plot!(Icoils,fit_JSF.(Icoils), label=L"10142024: %$(label1_str)" )
plot!(Icoils,fit_dataframewise.(Icoils), label=L"07122025: %$(label2_str)")
vspan!([0.0001,0.020], color=:gray, alpha=0.30,label="unresolved")
savefig(fig, joinpath(dir_path, "framewise.png"))

##########################################################################################
##########################################################################################
# CROSS-CORRELATION
##########################################################################################
##########################################################################################

xcorr_data = zeros(nI,2)
for k=1:nI
    ccbg = dropdims(mean(bg, dims=3),dims=3)
    ccf1 = dropdims(mean(Float64.(matread(matched_files[k])["F1"]), dims=3), dims=3)
    ccf2 = dropdims(mean(Float64.(matread(matched_files[k])["F2"]), dims=3), dims=3)

    img1 = ccf1 .- ccbg
    img2 = ccf2 .- ccbg

    profile1 = mean(img1, dims=1) |> vec 
    profile2 = mean(img2, dims=1) |> vec

    fig1=heatmap(1e3*z_position,1e3*x_position, img1, 
    colorbar=true, 
    # aspect_ratio=1, 
    title="Mean F=1 Signal",
    xlabel=L"$z$ (mm)", 
    ylabel=L"$x$ (mm)",
    )
    fig2=heatmap(1e3*z_position,1e3*x_position,img2, 
    colorbar=true, 
    # aspect_ratio=1, 
    title="Mean F=2 Signal",
    xlabel=L"$z$ (mm)", 
    ylabel=L"$x$ (mm)",
    )
    fig3=plot(1e3*z_position, profile1,
    label=L"Profile $F=1$",
    xlabel=L"$z$ (mm)", 
    ylabel="Intensity (a.u.)",
    line=(:blue,:dot,3)
    )
    fig4=plot(1e3*z_position, profile2,
    label=L"Profile $F=2$",
    xlabel=L"$z$ (mm)", 
    ylabel="Intensity (a.u.)",
    line=(:blue,:dot,3)
    )
    plot(fig1,fig2,fig3,fig4,
    layout=@layout([a1 a2; a3 a4]),
    size=(1000,450),
    left_margin=4mm,
    bottom_margin=3mm,
    )|>display

    println("Coil currrent $(@sprintf("%.3f",1e3*Icoils[k]))mA")
    dz, dx = estimate_shift_fft(img1, img2; Nmethod="none")
    println("Estimated shift 2D: dz = $(round(dz, digits=2)) px = $(round(1e3*cam_pixelsize*dz, digits=4)) mm \t dx = $(round(dx, digits=2)) px =  $(round(1e3*cam_pixelsize*dx, digits=4)) mm")

    shift = estimate_1d_shift_fft(profile1, profile2; Nmethod="none");
    println("Estimated shift 1D: $(round(shift,digits=2)) px = $(round(1e3*cam_pixelsize*shift, digits=4)) mm ")

    xcorr_data[k,1] = cam_pixelsize*dz
    xcorr_data[k,2] = cam_pixelsize*shift
end


pretty_table(
    hcat(Icoils,1e3*xcorr_data);
    formatters    = (ft_printf("%8.3f",1), ft_printf("%8.5f",2:6)),
    alignment=:c,
    header        = (
        ["Current", "2D : Δz", "1D : Δz"], 
        ["[A]", "[mm]", "[mm]"]
        ),
    border_crayon = crayon"blue bold",
    tf            = tf_unicode_rounded,
    header_crayon = crayon"yellow bold",
    equal_columns_width= true,
)


fig = plot(Icoils[2:end], abs.(1e3*xcorr_data[2:end,1]),
seriestype=:scatter, 
label="2D", 
marker=(:circle,:white,3), 
markerstrokecolor=:blue,
markerstrokewidth=2,
)
plot!(Icoils[2:end], abs.(1e3*xcorr_data[2:end,2]), 
seriestype=:scatter,
label="1D", 
marker=(:circle,:white,3), 
markerstrokecolor=:red,
markerstrokewidth=2,
)
plot!(    
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.002,1.0),
    title = "F=1 Peak Position vs Current",
    label = "07122025",
    seriestype = :scatter,
    marker = (:circle, :white, 4),
    markerstrokecolor = :black,
    markerstrokewidth = 2,
    legend = :topleft,
    grid = true,
    minorgrid = true,
    gridalpha = 0.5,
    gridstyle = :dot,
    minorgridalpha = 0.05,
    xticks = :log10,
    yticks = :log10,
    framestyle = :box,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=12,
)
vspan!([0.0001,0.025], color=:gray, alpha=0.30,label="unresolved", legend_columns=2)
savefig(fig, joinpath(dir_path, "crosscorrelation.png"))


t_run = Dates.canonicalize(Dates.now()-t_start)
println("Running time : ", t_run)

############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################
############################################################################################################################################################################




fig=plot(
    data_framewise[10:end, 1], data_framewise[10:end, 8],
    xaxis = (:log10, L"$I_{c} \ (\mathrm{A})$", :log),
    yaxis = (:log10, L"$z_{\mathrm{F}_{1}} \ (\mathrm{mm})$", :log),
    xlims = (0.01,1.0),
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
    xticks = :log10,
    yticks = :log10,
    framestyle = :box,
    size=(800,600),
    tickfontsize=11,
    guidefontsize=14,
    legendfontsize=12,
)
plot!(data_framewise[10:end, 1], data_framewise[10:end, 8],
    ribbon= data_framewise[10:end, 9],
    label= "Std", 
    fillalpha=0.5, fillcolor=:grey36, line=(:solid, :grey36, 0.2)
)
# from matlab
Ic_matlab = [0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, .016, 0.022, 0.032, 0.044, 0.058, 
                0.074, 0.094, 0.114,  0.134,  0.154,  0.174,  0.195,  0.215,  0.236, 
                0.26, 0.285,  0.308,  0.341,  0.381,  0.42,  0.46,  0.5,  0.55,  0.609
]
pos0_matlab = [ -0.0075, 0.0486, -0.0065, 0.0021, -0.0165, -0.0018, -0.0260, -0.0009, -0.0056, 
                0.0268, 0.0557, 0.0619, 0.1082, 0.1380, 0.1837, 0.2404, 0.2793, 0.3283, 0.3691, 
                0.4018, 0.4628, 0.5015, 0.5492, 0.6106, 0.6546, 0.7344, 0.8241, 0.8804, 0.9341, 1.0199, 1.1314
] # ki=1.58e-6
plot!(abs.(Ic_matlab[1:end]),abs.(pos0_matlab[1:end]), label=L"$k_{i} = 1.58\times 10^{-6}$")
vspan!([0.0001,Ic_matlab[9]], color=:yellow, alpha=0.30,label="unresolved")


Ic_matlab[10:end]
pos0_matlab[10:end]





function ndgrid(y, x)
    Y = reshape(y, :, 1) .* ones(1, length(x))
    X = ones(length(y), 1) .* reshape(x, 1, :)
    return Y, X
end
function generate_gaussian_image(size::Tuple{Int, Int}, amp::Float64, center::Tuple{Float64, Float64}, 
                                 sigma::Float64 = 5.0, noise_level::Float64 = 0.05)
    """
    Create a 2D Gaussian blob image with additive white Gaussian noise.

    Arguments:
        size: Tuple (height, width) of image
        center: Tuple (y, x) location of the Gaussian center
        sigma: Standard deviation of the Gaussian
        noise_level: Standard deviation of the additive noise (relative to peak value = 1)

    Returns:
        img: 2D array with a Gaussian peak plus noise
    """
    height, width = size
    y0, x0 = center

    # Generate coordinate grid
    y = 1:height
    x = 1:width
    Y, X = ndgrid(y, x)

    # Create 2D Gaussian
    gaussian = @. amp*exp(-((X - x0)^2 + (Y - y0)^2) / (2*sigma^2))

    # Normalize to peak = 1
    gaussian ./= maximum(gaussian)

    # Add white Gaussian noise
    noise = noise_level * randn(size...)
    img = gaussian .+ noise

    return img
end

# Image parameters
gsize = (540, 640)
sigma = 150.0
noise_level = 0.05

# Generate two images: second is shifted version
img1 = generate_gaussian_image(gsize,500.0, (440.0, 350.0), sigma, noise_level)
img2 = generate_gaussian_image(gsize,900.0, (215.0, 285.0), sigma, noise_level)  # ~ (2.5, -2.5) shift

heatmap(img1, colorbar=true, title="Image 1 (Gaussian Blob)") |> display
heatmap(img2, colorbar=true, title="Image 1 (Gaussian Blob)") |> display
heatmap(img1.+img2, colorbar=true, title="Image 1 (Gaussian Blob)") |> display

# Estimate shift
dx, dy = estimate_shift_fft(img1, img1; Nmethod="contrast")
println("Estimated shift: dx = $(round(dx, digits=3)) px, dy = $(round(dy, digits=3)) px")

profile1 = sum(img1, dims=1) |> vec 
profile2 = sum(img2, dims=1) |> vec

shift = estimate_1d_shift_fft(profile1, profile2; Nmethod="zscore")
println("Estimated shift: $shift pixels")
