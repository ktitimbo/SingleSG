module MyExperimentalAnalysis
    # Plotting backend and general appearance settings
    using Plots; gr()
    using Plots.PlotMeasures
    using MAT, JLD2
    using LinearAlgebra
    using ImageFiltering, FFTW
    using Statistics, StatsBase
    using BSplineKit, Optim
    using Colors, ColorSchemes
    using Printf, LaTeXStrings
    using OrderedCollections
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
            # Load stack (Nx × Nz × Nframes)
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
    function summarize_framewise(f1, f2, Icoils, centroid_data, δz ; rev_order::Bool=false)
        @assert size(f1) == size(f2) "f1 and f2 must have the same size"
        n = size(f1, 1)
        @info "Number of frames : $(n)"
        @assert size(f1, 2) == length(Icoils) "length(Icoils) must match number of columns"

        # Means / std / SEMs per column
        m1   = vec(mean(f1, dims=1));  s1   = vec(std(f1,  dims=1));  se1 = sqrt.( (s1 ./ sqrt(n)).^2 .+ δz.^2 )
        m2   = vec(mean(f2, dims=1));  s2   = vec(std(f2,  dims=1));  se2 = sqrt.( (s2 ./ sqrt(n)).^2 .+ δz.^2 )

        # Differences and propagated SEMs
        Δz       = m1 .- m2
        Δz_se   = sqrt.(s1.^2 .+ s2.^2)               # SEM(μ1-μ2)

        # Center relative to global centroid (allow centroid_* to be scalar or length-N vectors)
        F1_centroid      = m1 .- centroid_data.mean
        F1_centroid_se  = sqrt.( (se1).^2 .+ centroid_data.sem.^2 )

        F2_centroid      = m2 .- centroid_data.mean
        F2_centroid_se  = sqrt.( (se2).^2 .+ centroid_data.sem.^2 )

        Icoil_A = Icoils

        out = (; Icoil_A,
                F1_z_peak_mm=m1, F1_z_se_mm=se1,
                F2_z_peak_mm=m2, F2_z_se_mm=se2,
                Δz_mm=Δz, Δz_se_mm=Δz_se,
                F1_z_centroid_mm=F1_centroid, 
                F1_z_centroid_se_mm=F1_centroid_se,
                F2_z_centroid_mm=F2_centroid, 
                F2_z_centroid_se_mm=F2_centroid_se)

        if rev_order
            # reverse every vector in the NamedTuple
            return map(v -> reverse(v), out)
        else
            return out
        end
    end

    """
        stack_data(data_directory::AbstractString;
                pattern = r"Cur(.*?)A",
                ampmeter_scale = r"Ran(.*?)A",
                error_factor = 0.015,
                order::Symbol = :desc,
                keynames = ("BG","F1","F2"),
                verbose::Bool = true)

    Load and stack `F1`, `F2`, and `BG` image cubes from `.mat` files in `data_directory`,
    parsing coil currents from filenames and computing a per-file current uncertainty from
    an ammeter range token.

    # Filename expectations
    Each filename must contain **both**:
    - a current token matching `pattern` (default: `r"Cur(.*?)A"`), and
    - a range token matching `ampmeter_scale` (default: `r"Ran(.*?)A"`).

    In each regex, the **first capture group** `(.*?)` must yield a numeric token that may end
    with a unit suffix:
    - `"…u"` → micro (×1e-6 A)
    - `"…m"` → milli (×1e-3 A)
    - no suffix → amps (×1 A)

    Examples: `Cur373mA_Bin1x4_Exp2000ms_Ran10mA`, `Cur990uA_…_Ran1000uA`.

    # Arguments
    - `data_directory`: Folder containing the `.mat` files to load.
    - `pattern`: Regex used to extract the coil current from each filename.
    - `ampmeter_scale`: Regex used to extract the ammeter range (used for uncertainty).
    - `error_factor`: Multiplicative factor applied to the parsed range to form the current
    uncertainty; `CurrentsError[i] = error_factor * Ran[i]` (default `0.015`).
    - `order`: Sort order for the stacked data by current; `:asc` or `:desc` (default).
    - `keynames`: Tuple `(BG, F1, F2)` giving the MAT variable names to read from each file.
    - `verbose`: If `true`, logs progress every 5 files.

    # Returns
    An `OrderedDict` with:
    - `:Directory::String` — the input directory.
    - `:Files::Vector{String}` — file paths in sorted order.
    - `:Currents::Vector{Float64}` — parsed currents in amperes.
    - `:CurrentsError::Vector{Float64}` — per-file uncertainties in amperes,
    computed as `error_factor * Ran` (units converted as above).
    - `:F1_data::Array{T,4}`, `:F2_data::Array{T,4}`, `:BG_data::Array{T,4}` —
    arrays of shape `(Nx, Nz, Nframes, Ncurrents)`.

    # Behavior & assumptions
    - All files must have identical shapes for `BG`, `F1`, and `F2`; the first file defines
    `(Nx, Nz, Nframes)` and element type `T`.
    - Sorting is performed by `:Currents`, and `:Files`/`:CurrentsError` are permuted to match.
    - Throws an error if no files match, tokens cannot be parsed, required keys are missing,
    or array sizes differ.

    # Example
    ```julia
    st = stack_data("20250814"; order = :asc)
    st[:Currents]        # => e.g., [9.90e-4, 3.73e-1, …]
    st[:CurrentsError]   # => 0.015 .* parsed Ran values (A)
    size(st[:F1_data])   # => (Nx, Nz, Nframes, length(st[:Currents]))
    """
    function stack_data(data_directory::AbstractString;
                        pattern = r"Cur(.*?)A",
                        ampmeter_scale = r"Ran(.*?)A",
                        error_factor = 0.015,
                        order::Symbol = :desc,
                        keynames = ("BG","F1","F2"),
                        verbose::Bool = true)

        # ----------------------------- -----------------------------------------
        parse_amp(tok::AbstractString) =
            endswith(tok, "u") ? parse(Float64, chop(tok; tail=1)) * 1e-6 :
            endswith(tok, "m") ? parse(Float64, chop(tok; tail=1)) * 1e-3 :
                                parse(Float64, tok)
        # -----------------------------------------------------------------------
                        
        # 1) Collect only .mat files that match the pattern
        all_files = readdir(data_directory; join=true)
        files = [f for f in all_files if endswith(lowercase(f), ".mat") && occursin(pattern, f)]
        @assert !isempty(files) "No matching .mat files found in $data_directory"

        # 2a) Extract current tokens and convert to amperes
        tokens = Vector{SubString{String}}(undef, length(files))
        for (i, f) in enumerate(files)
            m = match(pattern, f)
            @assert m !== nothing "Couldn't parse current from: $f"
            tokens[i] = m.captures[1]  # e.g. "-7u", "117m", "12930u"
        end

        # 2b) Extract uncertainties of the currents
        ran_tokens = Vector{SubString{String}}(undef, length(files))
        for (i, f) in enumerate(files)
            mr = match(ampmeter_scale, f)
            @assert mr !== nothing "Couldn't parse range (Ran...A) from: $f"
            ran_tokens[i] = mr.captures[1]  # e.g. "10m", "1000m", "500u"
        end

        # Convert both Cur and Ran to amps (use the helper)
        currents = parse_amp.(tokens)
        ran_vals = parse_amp.(ran_tokens)   # amplitude in A (from Ran...)

        # 3) Sort by current (asc/desc)
        p = order === :desc ? sortperm(currents; rev=true) :
            order === :asc  ? sortperm(currents) :
            throw(ArgumentError("order must be :asc or :desc"))
        files    = files[p]
        currents = Float64.(currents[p])
        currents_errors = error_factor * Float64.(ran_vals[p])

        # 4) Probe sizes & eltype from first file
        keyBG, keyF1, keyF2 = keynames
        sz, T = matopen(files[1]) do fh
            f1 = read(fh, keyF1)
            ((size(f1)), eltype(f1))
        end

        # Sanity check the first file has all keys
        matopen(files[1]) do fh
            @assert haskey(fh, keyBG) && haskey(fh, keyF1) && haskey(fh, keyF2) "Missing keys in $(files[1])"
        end

        N = length(files)
        F1 = Array{T}(undef, sz[1], sz[2], sz[3], N)
        F2 = similar(F1)
        BG = similar(F1)
        
        println("\nEach component is organized as (Nx,Nz,Nframes,Ncurrents)\n")
        
        # 5) Load & stack
        for (i, f) in enumerate(files)
            matopen(f) do fh
                f1 = read(fh, keyF1); f2 = read(fh, keyF2); bg = read(fh, keyBG)
                @assert size(f1) == sz && size(f2) == sz && size(bg) == sz "Inconsistent sizes in $f"
                F1[:,:,:,i] = f1
                F2[:,:,:,i] = f2
                BG[:,:,:,i] = bg
            end
            verbose && i % 5 == 0 && @info "Loaded $i / $N" file=f
        end

        return OrderedDict(
            :Directory      => String(data_directory),
            :Files          => files,
            :Currents       => currents,        # amperes
            :CurrentsError  => currents_errors,  # amperes (0.015 * Ran in A)
            :F1_data        => F1,              # 540×2560×number of images× number of currents
            :F2_data        => F2,
            :BG_data        => BG
        )
    end

    """
        bin_x_mean(A::AbstractMatrix, binsize::Integer) -> AbstractMatrix

    Downsample (bin) the **rows** of `A` by averaging contiguous blocks of length `binsize`.
    Each column is processed independently. The result has size
    `(size(A,1) ÷ binsize, size(A,2))`.

    # Arguments
    - `A`: Input 2D array. If `A` has an integer element type (e.g. `UInt16`), consider
    converting first (e.g. `Float32.(A)`) to avoid overflow and to control the output type.
    - `binsize`: Positive integer that must divide `size(A,1)` exactly.

    # Returns
    A matrix where every `binsize` consecutive rows in `A` have been replaced by their mean.

    # Throws
    - `AssertionError` if `size(A,1) % binsize != 0`.

    # Notes
    - Internally reshapes to `(binsize, nblocks, ncols)`, takes `mean(...; dims=1)`, and
    removes the singleton dimension with `dropdims`.
    - Make sure `Statistics.mean` is available: `using Statistics`.
    (Qualify as `Statistics.mean` if other packages also define `mean`.)
    """
    function bin_x_mean(A::AbstractMatrix, binsize::Integer)
        nrows, ncols = size(A)
        @assert nrows % binsize == 0 "nrows ($nrows) must be divisible by binsize ($binsize)"

        B = reshape(A, binsize, div(nrows, binsize), ncols)  # (binsize, nblocks, ncols)
        M = dropdims(mean(B, dims=1), dims=1)                # (1, nblocks, ncols) -> (nblocks, ncols)
        return M 
    end


    """
        my_process_framewise_maxima(signal_key::String, data, n_bins::Integer;
                                    half_max::Bool=false, λ0::Float64=0.01) 
                                    -> Matrix{Float64}

    Extract per-frame peak positions (in mm) from processed Stern–Gerlach image stacks.
    Each frame’s z-profile is binned, optionally windowed at half-maximum, spline-fitted,
    and its primary maximum recorded. Returns a `(n_runs_max × nI)` matrix of peak positions

    Process processed Stern–Gerlach image stacks to extract the **per-frame peak
    positions** (along z, in mm) for each coil current setting.

    # Arguments
    - `signal_key::String` : Selects which signal to analyze. Must be `"F1"` or `"F2"`.
    - `data` : Dictionary containing processed image stacks with keys 
    `:F1ProcessedImages`, `:F2ProcessedImages`, and `:Currents`.
    - `n_bins::Integer` : Number of pixels to average together along z (binning factor).

    # Keywords
    - `half_max::Bool = false` : If `true`, restricts the spline fit to the region
    above half the maximum, focusing on the peak neighborhood.
    - `λ0::Float64 = 0.01` : Regularization parameter for cubic spline smoothing.

    # Returns
    - `Matrix{Float64}` of size `(n_runs_max, nI)` where:
    - `n_runs_max` = maximum number of frames across all currents,
    - `nI` = number of coil currents.
    Each entry `[i, j]` contains the z-position (in mm) of the primary peak for
    frame `i` at current `j`, or `NaN` if no frame exists.

    # Method
    For each current index `j` and each frame `i`:
    1. Load stack `(Nx × Nz × Nframes)` of processed images.
    2. Compute z-profile by averaging over x.
    3. Bin the profile along z by `n_bins`.
    4. (Optional) Restrict to half-maximum window.
    5. Fit a cubic smoothing spline `(z_fit, y_fit)` with regularization `λ0`.
    6. Find candidate maxima by minimizing the negative spline from several initial guesses.
    7. Deduplicate candidates and select the tallest peak.
    8. Store the z-position of the peak in mm.
    9. Generate diagnostic plots for each frame (profile + spline + peak), saving if `SAVE_FIG` is defined.

    # Example
    ```julia
    framewise_peaks_F1 = my_process_framewise_maxima("F1", processed_data, 2)
    framewise_peaks_F2 = my_process_framewise_maxima("F2", processed_data, 2; half_max=true)
    """
    function my_process_framewise_maxima(signal_key::String, data, n_bins::Integer; half_max::Bool=false, λ0::Float64=0.01)
        I_current = vec(data[:Currents])
        nI = length(I_current) # Number of current settings

        # Validate signal_key → dataset key
        signal_label = signal_key == "F1" ? :F1ProcessedImages :
                    signal_key == "F2" ? :F2ProcessedImages :
                    error("Invalid signal_key: choose 'F1' or 'F2'")

        # z-axes (mm)
        z_binned_mm = 1e3 .* pixel_positions(z_pixels, n_bins, effective_cam_pixelsize_z)

        # Determine max number of frames across currents
	    n_runs_max = maximum(size(Float64.(data[signal_label][:,:,:,i]), 3) for i in 1:nI)
        max_position_data = fill(NaN, n_runs_max, nI)  # (n_runs_max × nI)

        @info "Processing per-frame maxima" signal_label=signal_label

        for j in 1:nI
            # Load stack (Nx × Nz × Nframes)
            stack    = Float64.(data[signal_label][:,:,:,j])
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
                    # ceil(minimum(z_fit)),
                    quantile(z_fit, 0.40),
                    z_fit[argmax(y_fit)],
                    quantile(z_fit, 0.65),
                    quantile(z_fit, 0.75),
                    quantile(z_fit, 0.90),
                    # floor(maximum(z_fit)),
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
                    xlims=extrema(z_binned_mm),
                    title=L"%$(signal_key) Frame %$(i): $I_c=%$(round(1e3*I_current[j], digits=3))\ \mathrm{mA}$",
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
        my_process_mean_maxima(signal_key::String, data, n_bins::Integer; 
                            half_max=false, λ0::Float64=0.01) -> Vector{Float64}

    Extract mean peak positions (in mm) from processed Stern–Gerlach stacks.
    Profiles are averaged over frames, binned, optionally half-max windowed,
    spline-fitted, and the primary maximum is returned for each current.

    Process processed Stern–Gerlach image stacks to extract the primary peak 
    position (along z, in mm) for each coil current setting.

    # Arguments
    - `signal_key::String` : Selects which signal to analyze. Must be `"F1"` or `"F2"`.
    - `data` : Dictionary containing processed image stacks with keys 
    `:F1ProcessedImages`, `:F2ProcessedImages`, and `:Currents`.
    - `n_bins::Integer` : Number of pixels to average together along z (binning factor).

    # Keywords
    - `half_max::Bool = false` : If `true`, restricts the spline fit to the 
    region where the profile is above half of its maximum (focus on peak).
    - `λ0::Float64 = 0.01` : Regularization parameter for smoothing spline fit.

    # Returns
    - `Vector{Float64}` of length `nI`, where `nI` is the number of current values.  
    Each entry is the location (in mm) of the primary maximum of the mean profile 
    for the corresponding current.

    # Method
    For each current index `j`:
    1. Load stack `(Nx × Nz × Nframes)` of processed images.
    2. Compute per-frame z-profiles (average over x).
    3. Average over frames to get an overall mean profile.
    4. Bin the mean profile along z by `n_bins`.
    5. (Optional) Apply a half-maximum window to isolate the central peak.
    6. Fit a cubic smoothing spline `(z_fit, y_fit)` with parameter `λ0`.
    7. Locate maxima by minimizing the negative spline, starting from several 
    initial guesses. Filter duplicates and take the highest peak.
    8. Store the z-position of the primary peak in mm.
    9. Generate diagnostic plots:
    - Raw frame profiles and mean binned profile.
    - Spline fit, scatter of binned data, and annotated maximum.
    - Save plots if global `SAVE_FIG` is defined.

    # Example
    ```julia
    peaks_F1 = my_process_mean_maxima("F1", processed_data, 2; half_max=true, λ0=1e-3)
    peaks_F2 = my_process_mean_maxima("F2", processed_data, 2)
    """
    function my_process_mean_maxima(signal_key::String, data, n_bins::Integer; half_max=false, λ0::Float64=0.01)
        I_current = vec(data[:Currents])
        nI = length(I_current)
        
        # Validate signal_key
        signal_label = signal_key == "F1" ? :F1ProcessedImages :
                    signal_key == "F2" ? :F2ProcessedImages :
                    error("Invalid signal_key: choose 'F1' or 'F2'")
        @info "Processing mean maxima" signal_label=signal_label

        # Precompute z-axes (mm)
        z_full_mm   = 1e3 .* pixel_positions(z_pixels, 1,  effective_cam_pixelsize_z)
        z_binned_mm = 1e3 .* pixel_positions(z_pixels, n_bins, effective_cam_pixelsize_z)

        peak_positions = zeros(Float64,nI)

        for j in 1:nI
            # --- Load stack (Nx × Nz × Nframes at j-th current)
            stack = Float64.(data[signal_label][:,:,:, j])
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
                # ceil(minimum(z_fit)),
                quantile(z_fit, 0.40),
                z_fit[argmax(y_fit)],
                quantile(z_fit, 0.65),
                quantile(z_fit, 0.75),
                quantile(z_fit, 0.90),
                # floor(maximum(z_fit)),
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
                title=L"%$(signal_key) Processed: $I_c = %$(round(1e3*I_current[j], digits=3))\ \mathrm{mA}$",
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
        build_processed_dict(raw_data::OrderedDict{Symbol,Any},
                         DK::AbstractMatrix, FL::AbstractMatrix;
                         T = Float32, epsval = T(1e-12)) -> OrderedDict{Symbol,Any}

    Construct a dictionary of background-subtracted and flat-field–corrected
    images from raw Stern–Gerlach experimental data.

    # Arguments
    - `raw_data::OrderedDict{Symbol,Any}` : Dictionary containing raw image stacks.
    Expected keys:
    - `:F1_data` — 4D array of raw F1 signal images `(Nx × Nz × Nframes × Ncurrents)`.
    - `:F2_data` — 4D array of raw F2 signal images `(Nx × Nz × Nframes × Ncurrents)`.
    - `:BG_data` — 4D array of background images `(Nx × Nz × Nframes × Ncurrents)`.
    - `:Currents` — Vector of coil current values.
    - `DK::AbstractMatrix` : Dark-field (camera offset) image of size `(Nx × Nz)`.
    - `FL::AbstractMatrix` : Flat-field (uniform illumination) image of size `(Nx × Nz)`.

    # Keywords
    - `T::Type{<:Real} = Float32` : Numeric type used for computation and storage.
    - `epsval::Real = T(1e-12)` : Minimum allowed value in the flat-field to avoid
    division by zero during correction.

    # Processing steps
    1. Compute per-pixel flat-field correction: `flat = FL - DK`, clamped below by `epsval`.
    2. Convert all arrays to type `T`.
    3. Background-subtract each frame: `F1 - BG` and `F2 - BG`.
    4. Apply flat-field correction by dividing by `flat`.
    5. Expand `flat` to match the dimensionality of the image stacks.

    # Returns
    - `OrderedDict` with keys:
    - `:Currents` — coil currents (copied from `raw_data`).
    - `:F1ProcessedImages` — processed F1 image stack `(Nx × Nz × Nframes × Ncurrents)`.
    - `:F2ProcessedImages` — processed F2 image stack `(Nx × Nz × Nframes × Ncurrents)`.

    # Example
    ```julia
    processed = build_processed_dict(raw_data, DK_image, FL_image)

    F1proc = processed[:F1ProcessedImages]
    F2proc = processed[:F2ProcessedImages]
    currents = processed[:Currents]
    """
    function build_processed_dict(raw_data::OrderedDict{Symbol,Any},
                              DK::AbstractMatrix, FL::AbstractMatrix;
                              T = Float32, epsval = T(1e-12))

    # Per-pixel flat field, clamp to avoid zeros
    flat = max.(T.(FL) .- T.(DK), epsval)
    flat4 = reshape(flat, size(flat,1), size(flat,2), 1, 1)  # expand to 4D

    # Promote to Float32 once
    F1 = T.(raw_data[:F1_data])
    F2 = T.(raw_data[:F2_data])
    BG = T.(raw_data[:BG_data])

    # Background subtract then flat-field correct
    F1proc = (F1 .- BG) #./ flat4
    F2proc = (F2 .- BG) #./ flat4

    return OrderedDict(
        :Currents            => raw_data[:Currents],
        :CurrentsError       => raw_data[:CurrentsError],
        :F1ProcessedImages   => F1proc,   # size: 540 × 2560 × number of images× number of currents
        :F2ProcessedImages   => F2proc,   # size: 540 × 2560 × number of images× number of currents
    )
    end

    """
        mean_z_profile(stack::AbstractArray) -> Vector{<:Real}

    Compute the mean profile along the z-axis of a 3D image stack.

    Compute the mean z-profile from a stack `(Nx × Nz × Nframes)` by
    averaging over x and frames. Returns a vector of length `Nz`.

    # Arguments
    - `stack` : A 3D array of size `(Nx, Nz, Nframes)`, where
    - `Nx` = pixels along the x-axis,
    - `Nz` = pixels along the z-axis,
    - `Nframes` = number of frames.

    # Returns
    - `Vector{<:Real}` of length `Nz`, containing the mean intensity along `z`.
    This is obtained by averaging over both the x-dimension and all frames.

    # Notes
    - The output is dimensionless (arbitrary units).
    - Intended for use within `extract_profiles` to condense a stack into
    a single per-current profile.

    # Example
    ```julia
    jth_stack = data_processed[:F1ProcessedImages][:,:,:,j]  # (Nx × Nz × Nframes)
    profile = mean_z_profile(jth_stack)                     # length = Nz
    """
    mean_z_profile(stack) = vec(dropdims(mean(stack; dims=(1,3)); dims=(1,3)))

    """
        extract_profiles(data_processed, key::Symbol, nI::Integer, z_pixels::Integer;
                        T::Type{<:Real}=Float64, n_bin::Integer=1) -> Matrix{T}

    Compute mean `z`-profiles for each current index from a processed image dataset.
    If `n_bin > 1`, bin along `z` by averaging every `n_bin` consecutive entries.
    Returns an `(nI × z_out)` matrix, where `z_out = z_pixels ÷ n_bin`.

    # Arguments
    - `data_processed` : Dict-like container with a 4D array at `key`, sized `(Nx, Nz, Nframes, nI)`.
    - `key::Symbol`    : Dataset key (e.g. `:F1ProcessedImages`).
    - `nI::Integer`    : Number of current settings (size along 4th dimension).
    - `z_pixels::Integer` : Number of z pixels (`Nz`).
    - `T::Type{<:Real}` : Output element type (default `Float64`).
    - `n_bin::Integer` : Z-binning factor (default `1` = no binning).

    # Returns
    - `Matrix{T}` of size `(nI, z_pixels ÷ n_bin)`.

    # Notes
    - Requires `n_bin ≥ 1` and `z_pixels % n_bin == 0`.
    - Binning is along z (the 2nd dimension).
    """
    function extract_profiles(data_processed, key::Symbol, nI::Integer, z_pixels::Integer;
                            T::Type{<:Real}=Float64, n_bin::Integer=1, with_error::Bool = false)
        @assert n_bin ≥ 1 "n_bin must be ≥ 1"
        @assert z_pixels % n_bin == 0 "z_pixels ($z_pixels) must be divisible by n_bin ($n_bin)"
        z_out = div(z_pixels, n_bin)
        P = Matrix{T}(undef, nI, z_out)
        Q = with_error ? Matrix{T}(undef, nI, z_out) : nothing  # SEM (optional)

        @inbounds @views for j in 1:nI
            # stack_raw :: (Nx, Nz, Nframes)
            stack_raw = data_processed[key][:,:,:,j]
            stack = T <: eltype(stack_raw) ? stack_raw : T.(stack_raw)

            Nx, Nz, Nf = size(stack)
            @assert Nz == z_pixels

            if n_bin == 1
                # Average over x_pixels → (1,Nz,Nf)
                xmean = mean(stack; dims=1)
                # Mean across frames → (Nz,)
                prof = dropdims(mean(xmean; dims=3); dims=(1,3))
                P[j, :] = prof
                if with_error
                    σ_std = dropdims(std(xmean; dims=3, corrected=true); dims=(1,3))
                    σ_sem = σ_std ./ sqrt(Nf) 
                    Q[j, :] = σ_sem 
                end 
            else
                # Bin z first: reshape to (Nx, n_bin, z_out, Nframes)
                B = reshape(stack, Nx, n_bin, z_out, Nf)
                # Average over x and bin group → (1, 1, z_out, Nf)
                xbmean = mean(B; dims=(1,2))
                # Mean across frames → (z_out,)
                prof_binned = dropdims(mean(xbmean; dims=4); dims=(1,2,4))
                P[j, :] = prof_binned
                if with_error
                    σ_std = dropdims(std(xbmean; dims=4, corrected=true); dims=(1,2,4))
                    σ_sem = σ_std ./ sqrt(Nf) 
                    Q[j, :] = σ_sem  
                end
            end
        end

        return with_error ? (mean = P, sem = Q::Matrix{T}) : P
    end

    """
        plot_profiles(z_mm, profiles, Icoils; title::AbstractString)

    Plot one curve per row of `profiles` against `z_mm`. Labels show Ic in mA.
    Returns the figure.
    """
    function plot_profiles(z_mm, profiles, Icoils; title::AbstractString)
        nI = size(profiles.mean, 1)
        cols = palette(:darkrainbow, nI)
        fig = plot(title=title, xlabel=L"$z$ (mm)", ylabel="Intensity (au)")
        @inbounds for i in 1:nI
            plot!(fig, 
                z_mm, profiles.mean[i, :],
                ribbon = profiles.sem[i, :],
                line = (:solid, cols[i], 1),
                fillcolor = cols[i],
                fillalpha = 0.25,
                label = L"$I_{c}=%$(round(1e3*Icoils[i]; digits=3))\,\mathrm{mA}$")
        end
        plot!(fig, legend=:outerright, legend_columns=2, foreground_color_legend=nothing)
        return fig
    end


    function post_threshold_mean(x, Icoils, δx ; threshold,
                                   half_life::Real=5, # in samples
                                   eps::Real=1e-6,
                                   weighted::Bool=true)
        @assert length(x) == length(Icoils)
        @assert eps ≥ 0


        if !weighted
            # Plain mean over ALL entries (no threshold)
            N   = length(x)
            μ   = mean(x)
            s   = std(x; corrected=true)
            sem_data = (N > 1) ? s / sqrt(N) : 0.0
            # independent per-sample measurement errors δx_i:
            sem_meas = sqrt(sum(δx.^2)) / N              # = sqrt( (1/N^2)∑δx_i^2 )
            sem = sqrt(sem_data^2 + sem_meas^2)
            return (mean=μ, sem=sem)
        end

        # Weighted branch:
        idx0 = findfirst(>(threshold), Icoils)
        @assert idx0 !== nothing "No Icoils entry greater than threshold."

        # Weighted branch
        n  = length(x)
        τ  = half_life / log(2)                 # convert half-life to exp scale

        # Build weights for ALL entries:
        # - pre-threshold: tiny eps
        # - post-threshold: exponentially increasing (stabilized to avoid overflow)
        w = fill(eps, n)
        w[idx0:end] .= exp.( (0:(n - idx0)) ./ τ )
        # println("weights = $w")
        
        μ = mean(x, Weights(w))

        # normalize to probabilities ω (sum=1) for simple formulas
        ω = w ./ sum(w)

        # println("The result from the function is $(μ), whereas force is $(sum(ω .* x))")

        # process SEM (from residual scatter, scaled by Kish n_eff)
        res2    = (x .- μ).^2
        s_w2    = sum(ω .* res2)             # weighted residual variance (biased form)
        n_eff   = 1 / sum(ω.^2)              # Kish effective N
        sem_proc = sqrt(s_w2 / n_eff)

        # measurement SEM (from known per-sample errors δx_i)
        sem_meas = sqrt(sum((ω.^2) .* (δx.^2)))

        sem = sqrt(sem_proc^2 + sem_meas^2)

        return (mean=μ, sem=sem)
    end


    function mag_factor(directory::String)
        if directory >= "20260211"
            values = (1.00, 0.01)
        elseif directory >= "20251101"
            # values = (0.996,0.0047)
            values = (1.08,0.03)
        else
            # values = (1.1198,0.0061) 
            values = (1.28,0.01) 
        end
        return values
    end


    """
        cluster_by_tolerance(Ics; tol = 0.08)

    Cluster values from multiple datasets when they are numerically close within a
    relative tolerance. This is useful when several datasets contain floating-point
    values (for example, peak positions or inferred parameters) and you want to
    identify common values across datasets.

    # Arguments
    - `Ics`: A vector of vectors. Each element `Ics[j]` is a dataset containing
    numerical values.
    - `tol`: Relative tolerance (default = 0.08). Two values `x` and `y` belong to
    the same cluster when `abs(x - y) ≤ tol * min(x, y)`.

    A named tuple with two fields:

    ### 1. `raw`
    A vector of clusters, where each cluster is a vector of named tuples

        (val = Float64, set = Int, idx = Int)

    with:
    - `val`: the numerical value  
    - `set`: the dataset index (1-based)  
    - `idx`: the index of the value inside that dataset (1-based)

    Only clusters that include values from at least two distinct datasets are
    included in `raw`.

    ### 2. `summary`
    A vector of summaries, one per cluster in `raw`.  
    Each summary is a named tuple:

        (
            mean_val = Float64,
            std_val  = Float64,
            currents = Vector{Float64},
            datasets = Vector{Int},
            indices  = Vector{Int}
        )

    Fields:
    - `mean_val`: mean of the clustered values  
    - `std_val`: standard deviation of clustered values  
    - `currents`: all values that form the cluster  
    - `datasets`: dataset IDs of those values  
    - `indices`: within-dataset indices of those values  

    This provides a compact, analysis-ready representation of each cluster.

    # Algorithm
    1. Flatten all values across datasets while recording dataset index (`set`) and
    within-dataset index (`idx`).
    2. Sort the flattened values.
    3. Group consecutive sorted values into clusters if they satisfy the relative
    tolerance condition.
    4. Sort each cluster by dataset index.
    5. Discard clusters appearing in fewer than two datasets.
    6. Construct both:
    - the `raw` cluster representation, and
    - the `summary` representation with aggregated statistics.

    # Example
        A = [1.00, 1.05, 2.10]
        B = [0.97, 2.05, 4.00]
        C = [1.08, 2.00, 3.90]

        out = cluster_by_tolerance([A, B, C]; tol = 0.08)

        out.raw      # vector of clusters
        out.summary  # vector of summary statistics

    # Notes
    - Tolerance is relative, scaled by the smaller of the two values.
    - Complexity is O(N log N) due to sorting.
    - The function is deterministic for fixed inputs.
    """
    function cluster_by_tolerance(Ics; tol=0.08)
        # Flatten values, dataset ids, and index-in-dataset
        vals  = Float64[]
        vidx  = Int[]
        iidx  = Int[]

        for (j, vec) in enumerate(Ics)
            for (k, v) in enumerate(vec)
                push!(vals, v)
                push!(vidx, j)
                push!(iidx, k)
            end
        end

        # Sort all flattened values
        p = sortperm(vals)
        v  = vals[p]
        g  = vidx[p]
        idx = iidx[p]

        # Clustering
        clusters = Vector{Vector{NamedTuple{(:val, :set, :idx), Tuple{Float64,Int,Int}}}}()

        current = [(val=v[1], set=g[1], idx=idx[1])]

        for i in 2:length(v)
            prev = v[i-1]
            curr = v[i]

            if abs(curr - prev) ≤ tol * min(curr, prev)
                push!(current, (val=curr, set=g[i], idx=idx[i]))
            else
                push!(clusters, current)
                current = [(val=curr, set=g[i], idx=idx[i])]
            end
        end

        push!(clusters, current)

        for c in clusters
            sort!(c, by = x -> x.set)
        end

        # Keep only clusters appearing in ≥2 datasets
        multi = [c for c in clusters if length(unique(getfield.(c, :set))) ≥ 2]

        summary = [
        (
            mean_val = mean(getfield.(c, :val)),
            std_val  = std(getfield.(c, :val)),
            currents = getfield.(c, :val),
            datasets = getfield.(c, :set),
            indices  = getfield.(c, :idx)
        )
        for c in multi
    ];

        return (raw = multi, summary = summary)
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
           process_maxima,
           summarize_framewise,
           stack_data,
           bin_x_mean,
           my_process_framewise_maxima,
           my_process_mean_maxima,
           build_processed_dict,
           mean_z_profile,
           extract_profiles,
           plot_profiles,
           post_threshold_mean,
           mag_factor

end