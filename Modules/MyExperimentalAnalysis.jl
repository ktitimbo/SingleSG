module MyExperimentalAnalysis
    # Plotting backend and general appearance settings
    using Plots; gr()
    using Plots.PlotMeasures
    using MAT
    using LinearAlgebra
    using ImageFiltering, FFTW
    using Statistics, StatsBase
    using BSplineKit, Optim
    using Colors, ColorSchemes
    using Printf, LaTeXStrings
    using Dates
    # Corresponding to data acquired with no previous binning
    default_camera_pixel_size = 6.5e-6 ; # (mm)
    default_x_pixels = 2160;
    default_z_pixels = 2560;

    effective_cam_pixelsize_z = default_camera_pixel_size # along the z axis
    x_pixels = default_x_pixels
    z_pixels = default_z_pixels

    # runtime helper: is this name defined *and* truthy?
    _has(name::Symbol) = isdefined(@__MODULE__, name) && getfield(@__MODULE__, name) === true

    # Defaults only if not already present in this module
    if !isdefined(@__MODULE__, :OUTDIR)
        RUN_STAMP = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
        OUTDIR    = joinpath(pwd(), "artifacts", RUN_STAMP)
        isdir(OUTDIR) || mkpath(OUTDIR)
    end

    if !isdefined(@__MODULE__, :FIG_EXT)
        FIG_EXT = "svg"
    end

    if !isdefined(@__MODULE__, :SAVE_FIG)
        SAVE_FIG = false
    end
    # -------------------------------
    # FUNCTIONS
    # -------------------------------
    """
        saveplot(fig, name::AbstractString; ext::AbstractString=FIG_EXT) -> String

    Saves a plot figure to a file in the output directory.

    # Arguments
    - `fig`: The figure object to be saved (typically a plot generated using a plotting library).
    - `name`: The base name of the output file (without extension).
    - `ext`: (Optional) The file extension/type (e.g., `"png"`, `"pdf"`). Defaults to `FIG_EXT`.

    # Returns
    - `String`: The full path to the saved figure file.

    # Example
    """
    saveplot(fig, name::AbstractString; ext::AbstractString=FIG_EXT) = begin
        path = joinpath(OUTDIR, "$(name).$(ext)")
        savefig(fig, path)
        @info "Saved figure" path
        return nothing
    end

    """
        For BSplineKit fitting, compute weights for the B-spline fit.
        Compute uniform weights scaled by (1 - λ0). Returns an array of the same size as `x_array`.
    """
    function compute_weights(x_array, λ0::Float64)
        λ = clamp(float(λ0), 0.0, 1.0)
        w = similar(float.(x_array))
        fill!(w, (1 - λ))
        return w
    end

    """
        normalize_image(img; method="none")

    Normalize a 1D/2D image (or vector) using one of:
    - `"none"`:     return `Float64` copy
    - `"zscore"`:   global zero-mean, unit-variance (safe for constant images)
    - `"contrast"`: local high-pass (subtract Gaussian blur) then z-score

    Returns an `Array{Float64}` matching `img`’s shape.
    """
    function normalize_image(img; method::String = "none")
        img_f = float.(img)  # Ensure Float64

        if method == "none"
            return Array{Float64}(img_f)

        elseif method == "zscore"
            μ = mean(img_f)
            σg = std(img_f)
            σg = σg ≤ eps(σg) ? one(σg) : σg   # guard against tiny variance
            return (img_f .- μ) ./ σg

        elseif method == "contrast"
        kernel = ndims(img_f) == 2  ? Kernel.gaussian(5) : # 2D Gaussian kernel, size 5×5
                ndims(img_f) == 1  ? KernelFactors.gaussian(5) : # 1D Gaussian kernel, length 5
                error("Unsupported input dimension $(ndims(img_f)); expected 1D or 2D.")

            blurred = imfilter(img_f, kernel)
            highpass = img_f .- blurred
            μ = mean(highpass)
            σh = std(highpass)
            σh = σh ≤ eps(σh) ? one(σh) : σh   # guard against tiny variance
            return (highpass .- μ) ./ σh 

        else
            error("Unknown normalization method: '$method'. Use 'none', 'zscore', or 'contrast'.")
        end
    end

    """
        estimate_shift_fft(img1, img2; Nmethod::String="none", filename::String="2Dxcorr") -> Tuple{Float64,Float64}

    Estimate the relative shift between two 2D images using FFT-based cross-correlation with subpixel refinement via a 3×3 quadratic surface fit. Both images must have the same dimensions.

    Arguments:
    - `img1`, `img2`: Real-valued 2D arrays (matrices) of identical size.
    - `Nmethod`: Normalization method passed to `normalize_image` before correlation:
        - `"none"`     → no normalization (only converts to `Float64`)
        - `"zscore"`   → zero mean, unit variance
        - `"contrast"` → subtract local mean via Gaussian smoothing, then z-score
    - `filename`: Base name (without extension) for saving the visualization if `SAVE_FIG` is defined and true.

    Returns:
    - `(dx, dy)::Tuple{Float64,Float64}`: Signed shift in pixels representing how much `img2` is shifted relative to `img1`:
        - Positive `dx` → `img2` is shifted right
        - Positive `dy` → `img2` is shifted down

    Method:
    1. Convert inputs to `Float64` and optionally normalize according to `Nmethod`.
    2. Compute raw FFT-based cross-correlation:
    `xcorr = fftshift(real(ifft(fft(img1) .* conj(fft(img2)))))`
    3. Visualize the two normalized images and their cross-correlation. If `SAVE_FIG` is defined and true, save the figure to `filename`.
    4. Find the integer-pixel peak location in `xcorr`.
    5. If the peak is not on the border, extract a 3×3 neighborhood and fit a quadratic surface:
    `f(x, y) = a x² + b y² + c x y + d x + e y + f₀`
    - Solve for subpixel offsets from the peak using the Hessian and gradient of the fitted surface.
    6. Compute the total shift as `(peak + subpixel_offset) - center`, where `center = (nx ÷ 2 + 1, ny ÷ 2 + 1)` corresponds to zero lag after `fftshift`.

    Notes:
    - If the peak lies on the image border, subpixel refinement is skipped and only the integer shift is returned.
    - Uses raw correlation (phase + amplitude). For phase-only correlation, normalize the cross-power spectrum before the inverse FFT.
    - Assumes periodic extension due to FFT; apply windowing if images are not periodic to reduce wrap-around artifacts.
    """
    function estimate_shift_fft(img1, img2; Nmethod::String = "none", filename::String="2Dxcorr")

        @assert size(img1) == size(img2) "Images must have the same dimensions"

        # 1) Promote to Float64 and (optionally) normalize
        img1_f = normalize_image(float.(img1); method=Nmethod)
        img2_f = normalize_image(float.(img2); method=Nmethod)

        # 2) Raw cross-correlation 
        # Cross-correlation in frequency domain: inverse FFT of f1 * conj(f2)
        xcorr = fftshift(real(ifft(fft(img1_f) .* conj(fft(img2_f)))))

        # 3) Visualize the cross-correlation surface
        fig1 = heatmap(img1_f, colorbar=true, title="2D Normalized Signal 1")
        fig2 = heatmap(img2_f, colorbar=true, title="2D Normalized Signal 2")
        fig3 = heatmap(xcorr,  colorbar=true, title="2D Normalized Cross-Correlation")
        plot(fig1, fig2, fig3, layout=@layout([a b c]), size=(1200, 350)) |> display
        if _has(:SAVE_FIG)
            saveplot(fig,filename)
        end
        

        # 4) Coarse peak
        # Find the peak (maximum correlation value), indicating best alignment
        _, max_idx = findmax(xcorr)
        peak_y, peak_x = Tuple(CartesianIndices(xcorr)[max_idx])  # convert to (row, col)

        # 5) Subpixel refinement (3×3 quadratic), with light guards
        ny, nx = size(xcorr)
        dx_sub = dy_sub = 0.0  # initialize subpixel offsets

        # If the peak is not too close to the image edge, perform subpixel refinement
        if 2 ≤ peak_y ≤ size(xcorr, 1) - 1 && 2 ≤ peak_x ≤ size(xcorr, 2) - 1
            # Extract a 3×3 patch around the peak
            patch = @view xcorr[peak_y-1:peak_y+1, peak_x-1:peak_x+1]

            # Construct design matrix A and vector b for quadratic surface fit
            # We fit a quadratic f(x, y) = ax² + by² + cxy + dx + ey + f
            A = zeros(9, 6)
            bvec = zeros(9)
            k = 1
            for j in -1:1, i in -1:1
                A[k, :] .= (i^2, j^2, i*j, i, j, 1.0)  # coordinates and polynomial terms
                bvec[k] = patch[j+2, i+2]                # center of patch is at (2, 2)
                k += 1
            end

            # Solve least squares problem A * coeffs ≈ b
            coeffs = A \ bvec
            a, b, c, d, e, _ = coeffs  # unpack coefficients of quadratic fit

            # Compute the gradient and Hessian of the fitted quadratic surface
            H = [2a c; c 2b]           # Hessian matrix of second derivatives
            g = [-d; -e]                # Gradient vector (first derivatives)

            # Solve for subpixel offset: H⁻¹ * g gives the location of the peak
            sub_offset = H \ g
            dx_sub, dy_sub = sub_offset
        else
            @warn "Peak too close to border for subpixel refinement"
        end

        # 6) Center of cross-correlation image (after fftshift) is considered zero shift
        center_y = ny ÷ 2 + 1
        center_x = nx ÷ 2 + 1
        # Total shift = (peak position + subpixel offset) - center
        dy_total = (peak_y + dy_sub) - center_y
        dx_total = (peak_x + dx_sub) - center_x

        return dx_total, dy_total
    end

    """
        estimate_1d_shift_fft(signal1::AbstractVector, signal2::AbstractVector;
                            Nmethod::String="none") -> Float64

    Estimate the relative shift between two 1D signals using FFT-based cross-correlation with subpixel refinement via a 3-point parabolic fit. Both signals must have the same length.

    Arguments:
    - `signal1`, `signal2`: Real-valued vectors of equal length.
    - `Nmethod`: Normalization method passed to `normalize_image` before correlation:
        - `"none"`     → no normalization (only converts to `Float64`)
        - `"zscore"`   → zero mean, unit variance
        - `"contrast"` → subtract local mean via Gaussian smoothing, then z-score

    Returns:
    - `shift::Float64`: Signed shift in samples representing how much `signal2` is shifted relative to `signal1`:
        - Positive → `signal2` is shifted right (delayed)
        - Negative → `signal2` is shifted left (advanced)

    Method:
    1. Convert inputs to `Float64` and optionally normalize.
    2. Compute raw FFT-based cross-correlation:
    `xcorr = fftshift(real(ifft(fft(signal1) .* conj(fft(signal2)))))`
    3. Find the integer shift from the index of the peak in `xcorr`.
    4. Refine to subpixel resolution by fitting a parabola to the peak and its immediate neighbors:
    `sub = 0.5 * (y_left - y_right) / (y_left - 2y_center + y_right)`
    This step is skipped if the peak is at the edge or the fit is numerically unstable.
    5. Return `(peak_index + subpixel_offset) - center_index`, where `center_index = N ÷ 2 + 1` corresponds to zero lag after `fftshift`.

    Notes:
    - If either signal has near-zero variance after normalization, the function returns `0.0` and emits a warning.
    - Uses raw correlation (phase + amplitude). For phase-only correlation, normalize the cross-power spectrum before inverse FFT.
    - Assumes periodic extension of the signals due to FFT; apply a window before FFT if the signals are not periodic to reduce wrap-around artifacts.
    - Always displays plots of the input signals and correlation; if `SAVE_FIG` is defined and true, also saves them using `saveplot`.
    """
    function estimate_1d_shift_fft(signal1::AbstractVector, signal2::AbstractVector; Nmethod::String="none")
        """
        Estimate relative 1D shift using FFT-based cross-correlation with subpixel
        refinement via a 3-point parabolic fit. Returns the shift (Float64) telling
        how much `signal2` is shifted relative to `signal1` (right = positive).
        """
        @assert length(signal1) == length(signal2) "Signals must be the same length"
        N = length(signal1)

        # Promote & (optionally) normalize (works with your normalize_image for 1D)
        s1 = normalize_image(float.(signal1); method=Nmethod)
        s2 = normalize_image(float.(signal2); method=Nmethod)

        # Early-out for degenerate inputs
        if std(s1) ≤ eps() || std(s2) ≤ eps()
            @warn "Near-constant signals — returning zero shift."
            return 0.0
        end

        # FFT-based cross-correlation (raw, like your 2D version)
        F1 = fft(s1)
        F2 = fft(s2)
        xcorr = fftshift(real(ifft(F1 .* conj(F2))))

        # Optional visualization
        fig1 = plot(s1, label=false, title="Signal 1 ($Nmethod)")
        fig2 = plot(s2, label=false, title="Signal 2 ($Nmethod)")
        fig3 = plot(xcorr, label=false, title="1D Cross-correlation")
        fig  = plot(fig1, fig2, fig3, layout=@layout([a b c]), size=(1200, 350))
        display(fig)
        if _has(:SAVE_FIG)
            saveplot(fig, "xcorr1d_debug")
        end


        # Peak (integer)
        _, peak = findmax(xcorr)          # linear index since it's 1D
        center = N ÷ 2 + 1

        # Subpixel quadratic interpolation around the peak
        sub = 0.0
        if 2 ≤ peak ≤ N - 1
            y_l, y_c, y_r = xcorr[peak-1], xcorr[peak], xcorr[peak+1]
            denom = (y_l - 2y_c + y_r)
            # robust guard: compare to magnitude scale of the peak
            if abs(denom) > eps()*max(1.0, abs(y_c))
                sub = 0.5 * (y_l - y_r) / denom   # in (-1, +1)
            end
        else
            @warn "Peak at edge — subpixel refinement skipped."
        end

        # Signed shift relative to center (fftshift puts zero-lag at center)
        total_shift = (peak + sub) - center
        return total_shift
    end

    """
        linear_fit(data::AbstractMatrix) -> (a::Float64, b::Float64, fit::Function)

    Perform a simple linear least-squares fit of the form `y ≈ a * x + b` using two columns from `data`.

    Arguments:
    - `data`: A two-column matrix or array-like object where:
        - Column 1 contains the independent variable `x`
        - Column 2 contains the dependent variable `y`

    Returns:
    - `a::Float64`: Estimated slope of the best-fit line.
    - `b::Float64`: Estimated intercept of the best-fit line.
    - `fit::Function`: A callable function `fit(x)` that evaluates the fitted line at given `x` values.

    Method:
    1. Extract `x` = `data[:,1]` and `y` = `data[:,2]`.
    2. Build the design matrix `X = [x  ones(length(x))]`.
    3. Solve the least-squares problem `X * [a; b] ≈ y` using the backslash operator.
    4. Return `(a, b, fit)` where `fit` is a closure implementing `a * x + b`.

    Notes:
    - Uses ordinary least squares without weighting. For weighted fits, modify `X` and `y` accordingly.
    - The returned function `fit` works with scalars and arrays.
    """
    function linear_fit(data::AbstractMatrix)
        x = data[:, 1]
        y = data[:, 2]
        X = [x ones(length(x))] # Build design matrix X = [x 1]
        a, b = X \ y # Solve least squares: X * [a; b] ≈ y
        return a, b, x -> a * x + b # Return a, b, and a callable function fit(x) = a*x + b
    end

    """
        pixel_positions(img_size::Int, bin_size::Int, pixel_size::Float64) -> Vector{Float64}

    Compute the physical positions of pixel centers after binning, assuming the coordinates are centered within each binned pixel. The first returned position corresponds to the center of the first binned pixel, located at `pixel_size * bin_size / 2` from the start of the axis.

    Arguments:
    - `img_size::Int`: Original image size in pixels along one dimension.
    - `bin_size::Int`: Number of original pixels combined into one binned pixel. Must evenly divide `img_size`.
    - `pixel_size::Float64`: Physical size of a single original pixel (e.g., in mm or μm).

    Returns:
    - `Vector{Float64}`: Positions of binned pixel centers in the same physical units as `pixel_size`.

    Method:
    1. Verify that `img_size` is divisible by `bin_size`.
    2. Compute `n_pixels = img_size / bin_size`, the number of binned pixels.
    3. Compute the physical size of a binned pixel: `effective_size = bin_size * pixel_size`.
    4. Return the positions as `effective_size .* (1:n_pixels) .- effective_size / 2`.

    Notes:
    - Assumes a 1D coordinate system along the axis of binning.
    - To center positions around zero instead of the first pixel, subtract the mean of the returned positions.
    """
    function pixel_positions(img_size::Int, bin_size::Int, pixel_size::Float64)
    @assert img_size % bin_size == 0 "img_size must be divisible by bin_size"
    n_pixels = div(img_size, bin_size)
    effective_size = bin_size * pixel_size
    return effective_size .* (1:n_pixels) .- effective_size / 2
    end

    """
        process_mean_maxima(signal_key::String, data, n_bins;
                            half_max::Bool=false, λ0::Float64=0.01) -> Vector{Float64}

    Compute the primary peak position (in mm) of the mean z‑profile for either F1 or F2 across all coil currents.
    For each current index, the function forms per‑frame z‑profiles by averaging over x, averages across frames,
    optionally bins along z, optionally truncates to points above half‑maximum, fits a cubic smoothing B‑spline,
    and extracts the dominant peak position.

    Arguments:
    - `signal_key`: `"F1"` or `"F2"`. Selects `data["data"]["F1ProcessedImages"]` or `["F2ProcessedImages"]`.
    - `data`: Structured dataset with keys `"Current_mA"` and `"<F1|F2>ProcessedImages"`, where each entry `[1, j]`
    is a 3‑D stack `(Nx × Nz × Nframes)`.
    - `n_bins`: Binning factor along z (must evenly divide the z length).
    - `half_max`: If `true`, restricts the spline fit to samples with intensity > (max/2).
    - `λ0`: Smoothing parameter for the B‑spline fit (`BSplineOrder(4)`).

    Returns:
    - `peak_positions::Vector{Float64}`: For each current setting, the estimated z‑position of the primary peak (mm).

    Method (per current index):
    1. **Per‑frame profiles:** For each frame `i`, compute a z‑profile as `mean(stack[:, :, i], dims=1)` (average over y), then `vec(...)`.
    2. **Mean profile:** Average the stack over frames: `mean_over_frames = dropdims(mean(stack, dims=3); dims=3)`, then
    average over y: `mean_profile = vec(mean(mean_over_frames, dims=1))`.
    3. **Bin along z:** Assert `length(mean_profile) % n_bins == 0`; reshape to `(n_bins, :)` and average rows to obtain a
    lower‑resolution `processed_signal`.
    4. **Half‑max (optional):** Keep only indices where `processed_signal > maximum(processed_signal)/2`.
    5. **Spline fit:** Fit a cubic smoothing spline on `(z_fit, y_fit)` with weights `compute_weights(z_fit, λ0)`.
    6. **Peak selection:** Minimize the negative spline from several initial guesses, deduplicate candidate maxima, rank by value,
    and select the highest as the primary peak.
    7. **Plotting:** Plot raw per‑frame profiles + the binned mean, and the processed signal + spline + peak. If `save=true` (or
    your global `SAVE_FIG` is true), save the figure via `saveplot`.

    Notes:
    - Requires global `effective_cam_pixelsize_z` (meters) and `pixel_positions` for building the z‑axis; positions are converted to mm via `1e3`.
    - Ensure the z length matches `pixel_positions(z_pixels, …)`. If your detector width differs, pass the correct pixel count.
    """
    function process_mean_maxima(signal_key::String, data, n_bins::Integer; half_max=false, λ0::Float64=0.01)
        I_current = vec(data["data"]["Current_mA"])
        nI = length(I_current)
        
        # Validate signal_key
        signal_label = signal_key == "F1" ? "F1ProcessedImages" :
                    signal_key == "F2" ? "F2ProcessedImages" :
                    error("Invalid signal_key: choose 'F1' or 'F2'")
        @info "Processing mean maxima" signal_label=signal_label

        # Precompute z-axes (mm)
        z_full_mm   = 1e3 .* pixel_positions(z_pixels, 1,  effective_cam_pixelsize_z)
        z_binned_mm = 1e3 .* pixel_positions(z_pixels, n_bins, effective_cam_pixelsize_z)

        peak_positions = zeros(Float64,nI)

        for j in 1:nI
            # --- Load stack (Nx × Nz × Nframes)
            stack = Float64.(data["data"][signal_label][1, j])
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

            # --- Bin along z by reshaping and averaging
            @assert length(mean_profile) % n_bins == 0 "Signal length not divisible by n_bins"
            binned           = reshape(mean_profile, n_bins, :)
            processed_signal = vec(mean(binned, dims=1))           # (Nz / n_bins)

            # --- Optional half-maximum window
            z_fit = z_binned_mm
            y_fit = processed_signal
            if half_max
                ymax    = maximum(y_fit)
                keep_ix = findall(yi -> yi > ymax/2, y_fit)
                z_fit   = z_fit[keep_ix]
                y_fit   = y_fit[keep_ix]
            end

            # --- Spline fit (cubic) on (z_fit, y_fit)
            S_fit = BSplineKit.fit(BSplineOrder(4), z_fit, y_fit, λ0; weights=compute_weights(z_fit, λ0))

            # --- Maxima via minimizing negative spline from multiple guesses
            negative_spline(x) = -S_fit(x[1])
            initial_guesses = sort([
                ceil(minimum(z_fit)),
                quantile(z_fit, 0.40),
                z_fit[argmax(y_fit)],
                quantile(z_fit, 0.65),
                quantile(z_fit, 0.75),
                quantile(z_fit, 0.90),
                floor(maximum(z_fit)),
            ])

            minima_candidates = Float64[]
            for g in initial_guesses
                res = optimize(negative_spline, [minimum(z_fit)], [maximum(z_fit)], [g], Fminbox(LBFGS()))
                push!(minima_candidates, Optim.minimizer(res)[1])
            end
            sort!(minima_candidates)
            filtered = [minima_candidates[1]]
            for m in minima_candidates[2:end]
                if all(abs(m - x) > 1.0e-9 for x in filtered)
                    push!(filtered, m)
                end
            end

            vals   = -S_fit.(filtered)
            order  = sortperm(vals)
            maxima = filtered[order]

            # Store primary peak (mm)
            peak_positions[j] = maxima[1]

            # --- Plots
            fig_raw = plot(
                xlabel=L"$z$ (mm)", ylabel="Intensity (a.u.)",
                title=L"%$(signal_key) Raw: $I_c = %$(round(I_current[j], digits=3))\ \mathrm{mA}$",
            )
            cols = palette(:phase, n_frames)
            for i in 1:n_frames
                plot!(fig_raw, z_full_mm, frame_profiles_mat[i, :], label=false, line=(:dot, cols[i], 1))
            end
            plot!(fig_raw, z_binned_mm, processed_signal, label="mean (bins=$(n_bins))", line=(:solid, :black, 2))

            fig_fit = plot(
                z_fit, y_fit,
                seriestype=:scatter, marker=(:circle,:white, 2), markerstrokecolor=:gray36, markerstrokewidth=0.8,
                xlabel=L"$z\ (\mathrm{mm})$", ylabel="Intensity (a.u.)",
                title=L"%$(signal_key) Processed: $I_c = %$(round(I_current[j], digits=3))\ \mathrm{mA}$",
                label="$(signal_key) processed", legend=:topleft,
            )
            xs = collect(range(minimum(z_fit), maximum(z_fit), length=2000))
            plot!(fig_fit, xs, S_fit.(xs), line=(:solid, :red, 2), label="Spline fit")
            vline!(fig_fit, [maxima[1]], line=(:dash, :black, 1), label=L"$z_{\max}=%$(round(maxima[1], digits=3))\ \mathrm{mm}$")

            fig = plot(fig_raw, fig_fit; layout=@layout([a b]), size=(900, 400), left_margin=3mm, bottom_margin=3mm)
            display(fig)
            if _has(:SAVE_FIG)
                saveplot(fig, "m_$(signal_key)_I$(@sprintf("%02d", j))")
            end
        end

        return peak_positions
    end

    """
        process_framewise_maxima(signal_key::String, data, n_bins;
                                half_max::Bool=false, λ0::Float64=0.01) -> Matrix{Float64}

    Compute per‑frame z‑positions (in mm) of the dominant intensity maximum for F1 or F2 stacks, across all coil currents.

    Arguments:
    - `signal_key`: `"F1"` or `"F2"`. Selects `data["data"]["F1ProcessedImages"]` or `["F2ProcessedImages"]`.
    - `data`: Structured dataset with keys `"Current_mA"` and `"<F1|F2>ProcessedImages"`. Each entry `[1, j]` is a 3‑D stack `(Nx × Nz × Nframes)`.
    - `n_bins`: Binning factor along z (must evenly divide the z length).
    - `half_max`: If `true`, restricts the spline fit to samples with intensity > (max/2).
    - `λ0`: Smoothing parameter for the cubic B‑spline fit.

    Returns:
    - `max_position_data::Matrix{Float64}` of size `(n_runs_max × nI)`. Row `i`, column `j` is the estimated peak position (mm) for frame `i` of current index `j`; entries with no data are `NaN`.

    Method (per current `j`, per frame `i`):
    1. Form a z‑profile by averaging the frame over **x**: `frame_profile = vec(mean(stack[:, :, i], dims=1))`.
    2. Bin along z by reshaping into `(n_bins, :)` and averaging rows to obtain `processed_profile`.
    3. (Optional) restrict to points above half‑maximum.
    4. Fit a cubic smoothing spline on `(z_fit, y_fit)` with weights `compute_weights(z_fit, λ0)`.
    5. Run multi‑start optimization on the negative spline, deduplicate candidates, rank by value, and select the top as the frame’s maximum.
    6. Plot the processed profile, spline, and detected maximum; save if requested.

    Notes:
    - Requires global `effective_cam_pixelsize_z` and `pixel_positions`; positions are converted to mm via `1e3`.
    - `n_bins` must evenly divide the z length (asserted).
    """
    function process_framewise_maxima(signal_key::String, data, n_bins::Integer; half_max::Bool=false, λ0::Float64=0.01)
        I_current = vec(data["data"]["Current_mA"])
        nI = length(I_current) # Number of current settings

        # Validate signal_key → dataset key
        signal_label = signal_key == "F1" ? "F1ProcessedImages" :
                    signal_key == "F2" ? "F2ProcessedImages" :
                    error("Invalid signal_key: choose 'F1' or 'F2'")

        # z-axes (mm)
        z_binned_mm = 1e3 .* pixel_positions(z_pixels, n_bins, effective_cam_pixelsize_z)

        # Determine max number of frames across currents
        n_runs_max = maximum(size(Float64.(data["data"][signal_label][1, i]), 3) for i in 1:nI)
        max_position_data = fill(NaN, n_runs_max, nI)  # (n_runs_max × nI)

        @info "Processing per-frame maxima" signal_label=signal_label

        for j in 1:nI
            # Load stack (Ny × Nz × Nframes)
            stack    = Float64.(data["data"][signal_label][1, j])
            n_frames = size(stack, 3)
            cols     = palette(:phase, n_frames)

            for i in 1:n_frames
                # --- Per-frame z-profile (average over x → 1×Nz → vec)
                frame_profile = vec(mean(stack[:, :, i], dims=1))

                # --- Bin along z
                @assert length(frame_profile) % n_bins == 0 "Signal length not divisible by n_bins"
                binned            = reshape(frame_profile, n_bins, :)
                processed_profile = vec(mean(binned, dims=1))  # length = Nz / n_bins

                # --- Optional half-maximum window
                z_fit = z_binned_mm
                y_fit = processed_profile
                if half_max
                    y_max  = maximum(y_fit)
                    keep   = findall(yi -> yi > y_max/2, y_fit)
                    z_fit  = z_fit[keep]
                    y_fit  = y_fit[keep]
                end

                # --- Spline fit (cubic) on (z_fit, y_fit)
                S_fit = BSplineKit.fit(BSplineOrder(4), z_fit, y_fit, λ0; weights=compute_weights(z_fit, λ0))

                # --- Maxima via minimizing negative spline from multiple guesses
                negative_spline(x) = -S_fit(x[1])
                initial_guesses = sort([
                    ceil(minimum(z_fit)),
                    quantile(z_fit, 0.40),
                    z_fit[argmax(y_fit)],
                    quantile(z_fit, 0.65),
                    quantile(z_fit, 0.75),
                    quantile(z_fit, 0.90),
                    floor(maximum(z_fit)),
                ])

                candidates = Float64[]
                for g in initial_guesses
                    res = optimize(negative_spline, [minimum(z_fit)], [maximum(z_fit)], [g], Fminbox(LBFGS()))
                    push!(candidates, Optim.minimizer(res)[1])
                end
                sort!(candidates)

                # Deduplicate (within 1e-9)
                dedup = [candidates[1]]
                for v in candidates[2:end]
                    if all(abs(v - x) > 1e-9 for x in dedup)
                        push!(dedup, v)
                    end
                end

                # Rank candidates by actual spline height (largest peak first)
                @assert !isempty(dedup) "No peak candidates found"
                vals     = S_fit.(dedup)        # evaluate spline (not negated)
                best_ix  = argmax(vals)         # tallest peak index
                max_z    = dedup[best_ix]       # z of tallest peak

                # Store result for this frame/current
                max_position_data[i, j] = max_z

                # --- Plot per-frame processed profile + spline + peak
                fig = plot(
                    z_fit, y_fit,
                    seriestype=:scatter, marker=(:circle, :white, 2),
                    markerstrokecolor=:gray36, markerstrokewidth=0.8,
                    xlabel=L"$z\ (\mathrm{mm})$", ylabel="Intensity (a.u.)",
                    title=L"%$(signal_key) Frame %$(i): $I_c=%$(round(I_current[j], digits=3))\ \mathrm{mA}$",
                    label="$(signal_key) processed", legend=:topleft,
                );
                xs = collect(range(minimum(z_fit), maximum(z_fit), length=2001));
                plot!(fig, xs, S_fit.(xs), line=(:solid, :red, 2), label="Spline fit");
                vline!(fig, [max_z], line=(:dash, :black, 1), label=L"$z_{\max}=%$(round(max_z, digits=3))\ \mathrm{mm}$");
                display(fig)
                if _has(:SAVE_FIG)
                    saveplot(fig, "fw$(i)_$(signal_key)_I$(@sprintf("%02d", j))")
                end
                
            end
        end

        return max_position_data
    end

    """
        process_maxima(mode::Symbol, signal_key::String, data, n_bins::Integer;
                    half_max::Bool=false, λ0::Float64=0.01)

    Run either [`process_mean_maxima`](@ref) or [`process_framewise_maxima`](@ref) depending on `mode`.

    # Arguments
    - `mode`: `:mean` → run `process_mean_maxima`; `:framewise` → run `process_framewise_maxima`.
    - `signal_key`: `"F1"` or `"F2"` — selects which processed image set to analyze.
    - `data`: structured dataset with `"Current_mA"` and `"<F1|F2>ProcessedImages"`.
    - `n_bins`: number of consecutive z-pixels to average when binning.
    - `half_max`: if `true`, restrict fit to points above half the maximum.
    - `λ0`: smoothing parameter for cubic B-spline fitting.

    # Returns
    - See the return type of the called function:
    - `:mean` → `Vector{Float64}` of peak positions per current setting.
    - `:framewise` → `Matrix{Float64}` of peak positions per frame and current setting.
    """
    function process_maxima(mode::Symbol, signal_key::String, data, n_bins::Integer; half_max::Bool=false, λ0::Float64=0.01 )
        if mode == :mean
            return process_mean_maxima(signal_key, data, n_bins; half_max=half_max, λ0=λ0)
        elseif mode ==:framewise
            return process_framewise_maxima(signal_key, data, n_bins; half_max=half_max, λ0=λ0)
        else
            return error("Invalid mode: choose :mean or :framewise")
        end   
    end

    """
        summarize_framewise(f1, f2, Icoils, centroid_mean, centroid_std; rev_order=false)

    Given frame-wise data matrices `f1` and `f2` (rows = samples, columns = current settings),
    compute per-column means, standard errors, differences, and centered versions.

    Arguments
    - `f1`, `f2` : AbstractMatrix (size (n_samples, n_currents)), same size
    - `Icoils` : AbstractVector of coil currents (length = n_currents)
    - `centroid_mean` : scalar (or vector length n_currents) centroid value to subtract
    - `centroid_std`  : scalar (or vector length n_currents) uncertainty of centroid
    - `rev_order` : if true, reverse the order of all returned vectors

    Returns (NamedTuple of Vectors)
    - `I_coil_mA`
    - `F1_z_peak_mm`, `F1_z_sem_mm`
    - `F2_z_peak_mm`, `F2_z_sem_mm`
    - `Δz_mm`, `Δz_sem_mm`
    - `F1_z_centroid_mm`, `F1_z_centroid_sem_mm`
    - `F2_z_centroid_mm`, `F2_z_centroid_sem_mm`
    """
    function summarize_framewise(f1, f2, Icoils, centroid_mean, centroid_std; rev_order::Bool=false)
        @assert size(f1) == size(f2) "f1 and f2 must have the same size"
        n = size(f1, 1)
        @assert size(f1, 2) == length(Icoils) "length(Icoils) must match number of columns"

        # Means / std / SEMs per column
        m1   = vec(mean(f1, dims=1));  s1   = vec(std(f1,  dims=1));  se1 = s1 ./ sqrt(n)
        m2   = vec(mean(f2, dims=1));  s2   = vec(std(f2,  dims=1));  se2 = s2 ./ sqrt(n)

        # Differences and propagated SEMs
        Δz       = m1 .- m2
        Δz_se   = sqrt.(s1.^2 .+ s2.^2) ./ sqrt(n)              # SEM(μ1-μ2)

        # Center relative to global centroid (allow centroid_* to be scalar or length-N vectors)
        F1_centroid      = m1 .- centroid_mean
        F1_centroid_se  = sqrt.( (se1).^2 .+ centroid_std.^2 )

        F2_centroid      = m2 .- centroid_mean
        F2_centroid_se  = sqrt.( (se2).^2 .+ centroid_std.^2 )

        I_coil_mA = -1000 .* Icoils

        out = (; I_coil_mA,
                F1_z_peak_mm=m1, F1_z_se_mm=se1,
                F2_z_peak_mm=m2, F2_z_se_mm=se2,
                Δz_mm=Δz, Δz_se_mm=Δz_se,
                F1_z_centroid_mm=F1_centroid, F1_z_centroid_se_mm=F1_centroid_se,
                F2_z_centroid_mm=F2_centroid, F2_z_centroid_se_mm=F2_centroid_se)

        if rev_order
            # reverse every vector in the NamedTuple
            return map(v -> reverse(v), out)
        else
            return out
        end
    end

    # Public API
    export saveplot, 
           compute_weights, 
           normalize_image, 
           estimate_shift_fft, 
           estimate_1d_shift_fft, 
           linear_fit, 
           pixel_positions, 
           process_mean_maxima,
           process_framewise_maxima, 
           process_maxima

end