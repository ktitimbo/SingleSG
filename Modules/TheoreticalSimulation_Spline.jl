# ==============================================================================
# Spline-based peak finding and Stern–Gerlach screen-profile analysis
#
# This file implements four layers, used together to go from raw (x, z) hit
# positions on the detector screen to sub-bin-accurate peak locations:
#
#   1. Generic peak finding
#      max_of_bspline_positions
#      Fit a smoothing B-spline to any (z, y) curve and return the locations
#      of its most prominent local maxima, refined via a 1-D Brent search.
#
#   2. Single-dataset screen-profile analysis
#      analyze_screen_profile / CQD_analyze_screen_profile
#      Bin a set of (x, z) hits into a 2D histogram, average over x to get a
#      1D z-profile, smooth it, and locate both raw and spline-refined maxima
#      for a single coil current. The two functions are functionally
#      identical; `CQD_analyze_screen_profile` exists so CQD up/down branch
#      results can be labeled and saved separately from QM manifold results
#      (`branch` vs `manifold` only affect plot/file labeling, not the math).
#
#   3. QM batch / dictionary wrappers
#      QM_analyze_profiles_to_dict (4 methods)
#      Repeatedly call `analyze_screen_profile` across many coil currents,
#      handling hyperfine-manifold selection and meters→mm conversion from
#      one of several input formats (an in-memory OrderedDict, a JLD2 file
#      path, or pre-extracted (Ix, img) pairs).
#
#   4. CQD batch / dictionary wrappers
#      CQD_analyze_profiles_to_dict (2 methods)
#      Same role as (3), but for CQD up/down branch data (no manifold
#      selection needed — each dataset already corresponds to one branch),
#      reading from either an in-memory OrderedDict or a JLD2 file path.
# ==============================================================================
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 1. Generic peak finding
# ──────────────────────────────────────────────────────────────────────────────
"""
    max_of_bspline_positions(z, y; λ0=0.01, order=4, n_peaks=1,
                             n_scan=max(400, length(z)), sep=1e-6)
 
Fit a smoothing B-spline `S(z)` to `(z, y)` and return the locations of the most
prominent local maxima of the fitted curve.
 
# Arguments
- `z::AbstractVector`:
  Sorted vector of abscissae (domain points). Must satisfy `issorted(z)`.
- `y::AbstractVector`:
  Ordinates corresponding to `z`. Must have `length(y) == length(z)`.
- `λ0::Real = 0.01`:
  Smoothing parameter for the spline fit (smaller = follows data more closely;
  larger = smoother curve).
- `order::Int = 4`:
  B-spline order (`order = degree + 1`; e.g. `4` → cubic).
- `n_peaks::Int = 1`:
  Number of peak positions to return (sorted by descending spline height).
- `n_scan::Int = max(400, length(z))`:
  Number of points in a dense scan used to locate candidate maxima.
- `sep::Real = 1e-6`:
  Minimum separation between reported peaks (same units as `z`).
 
# Method
1. Fit a smoothing spline `S` via `BSplineKit.fit(BSplineOrder(order), z, y, λ0; weights=compute_weights(z, λ0))`.
   (Assumes a helper `compute_weights(z, λ0)` is available; adapt if using
   uniform weights.)
2. Densely sample `S` on `n_scan` points over `extrema(z)` and detect candidate
   maxima by sign changes in the finite-difference slope.
3. Refine each candidate position with a 1-D Brent search (`Optim.optimize`)
   in a small bracket around the candidate.
4. Remove near-duplicates closer than `sep`.
5. Sort remaining candidates by `S` height (descending) and keep the top `n_peaks`.
 
# Returns
- `(positions::Vector{Float64}, S::Function)`:
  Peak positions (highest first) and the callable fitted spline `S`.
 
# Notes
- Endpoints are included as candidates so a boundary maximum can be returned.
- Throws an `AssertionError` if `z` is unsorted or `length(z) != length(y)`.
- Peak heights can be obtained as `S.(positions)` if needed.
- Even with the default `n_peaks=1`, `positions` is always returned as a
  `Vector` (length 1 in that case) — every caller in this file unwraps it via
  `positions[1]` rather than treating it as a scalar.
 
# Example
julia
pos, S = max_of_bspline_positions(z, y; λ0=0.005, order=4, n_peaks=2)
heights = S.(pos)
 
# See also
[`analyze_screen_profile`](@ref), [`QM_analyze_profiles_to_dict`](@ref), [`smooth_profile`](@ref)
 
"""
function max_of_bspline_positions(z::AbstractVector, y::AbstractVector;
    λ0::Real=0.01, order::Int=4, n_peaks::Int=1, n_scan::Int=max(400, length(z)), sep::Real=1e-6)
 
    @assert length(z) == length(y) && issorted(z)
    a, b = extrema(z)
 
    # Fit smoothing spline
    S = BSplineKit.fit(BSplineOrder(order), z, y, λ0; weights=compute_weights(z, λ0))
 
    # --- 1) Dense scan to find candidate peaks ----------------------------
    # Sample S finely and look for sign changes in the finite-difference
    # slope (positive → negative = a local max between xs[i] and xs[i+1]).
    xs = range(a, b; length=n_scan)
    ys = S.(xs)
    dx = step(xs)
    dydx = diff(ys) ./ dx
 
    cand = Float64[]
    for i in 1:length(dydx)-1
        if (dydx[i] > 0) && (dydx[i+1] < 0)
            push!(cand, xs[i+1])              # near a local max
        end
    end
    push!(cand, a); push!(cand, b)            # endpoints too, in case the true max sits at a boundary
    cand = unique(sort(cand))
 
    # --- 2) Refine each candidate in a small bracket with Brent -----------
    # The dense scan above only localizes maxima to within one scan step;
    # Brent's method then polishes each candidate to high precision.
    negS(x) = -S(x)   # Optim.optimize minimizes, so negate to search for a maximum
    function refine(x0)
        δ = 2*dx
        lo = max(a, x0 - δ)
        hi = min(b, x0 + δ)
        if lo == hi
            return x0
        end
        res = Optim.optimize(negS, lo, hi)
        return Optim.minimizer(res)
    end
 
    peaks = Float64[]
    for c in cand
        ẑ = refine(c)
        # de-duplicate close peaks (two distinct scan candidates can refine
        # to essentially the same point, e.g. near a broad/shallow maximum)
        if isempty(peaks) || all(abs(ẑ - p) > sep for p in peaks)
            push!(peaks, ẑ)
        end
    end
 
    # --- 3) Sort by actual spline height and return positions only --------
    # `ord[1:k]` selects the top-k indices directly, instead of materializing
    # the fully reordered `peaks` array and then slicing that.
    heights = S.(peaks)
    ord = sortperm(heights, rev=true)
    k = min(n_peaks, length(peaks))
    return peaks[ord[1:k]], S # positions only
end

# ──────────────────────────────────────────────────────────────────────────────
# 2. Single-dataset screen-profile analysis
# ──────────────────────────────────────────────────────────────────────────────
"""
    analyze_screen_profile(
        Ix::Real,
        data_mm::AbstractMatrix;
        manifold::Symbol = :F_top,
        nx_bins::Integer = 2,
        nz_bins::Integer = 2,
        add_plot::Bool = false,
        plot_xrange::Symbol = :all,
        width_mm::Float64 = 0.150,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3,
        mode::Symbol = :probability
    ) -> NamedTuple
 
Analyze the vertical **z** profile of particle hits on a screen for a given coil
current. The routine builds a 2D histogram in **mm** over fixed x/z windows,
averages over x to obtain a z-profile, smooths it, and finds peak locations
using both discrete maxima and spline-fitted maxima.
 
This function supports single-current analysis and batch workflows (see the
companion batch function). The `Ix` argument is used for plot labels/titles.
 
# Arguments
- `Ix::Real`:
  Coil current (A). Used for plot annotation; does not affect the analysis.
- `data_mm::AbstractMatrix`:
  `N×2` array of hit positions **in millimeters** with columns:
  `(:, 1) = x`, `(:, 2) = z`.
 
# Keyword Arguments
- `manifold::Symbol = :F_top`:
  Label forwarded to plot titles/filenames (no effect on computation).
- `nx_bins::Integer = 2`, `nz_bins::Integer = 2`:
  Binning multipliers. The physical bin sizes (mm) are
  `x_bin_size = 1e3 * nx_bins * DEFAULT_camera_pixel_size` and
  `z_bin_size = 1e3 * nz_bins * DEFAULT_camera_pixel_size`,
  where `DEFAULT_camera_pixel_size` is a **global** in meters.
- `add_plot::Bool = false`:
  If `true`, plot raw/smoothed/spline profiles and mark their maxima.
- `plot_xrange::Symbol = :all`:
  Limits for the x-axis of the profile plot.
  `:all` → full z window; `:left` → left quarter; `:right` → right quarter.
- `width_mm::Float64 = 0.150`:
  Gaussian kernel σ (mm) used by `smooth_profile`.
- `λ_raw::Float64 = 0.01`, `λ_smooth::Float64 = 1e-3`:
  Spline regularization parameters for the raw and smoothed profiles.
- `mode::Symbol = :probability`:
  Normalization mode passed to `StatsBase.normalize` for the 2D histogram
  (e.g., `:probability`, `:pdf`, `:density`).
 
# Method (overview)
1. Fixed analysis windows: `x ∈ [-8.0, 8.0]` mm, `z ∈ [-12.5, 12.5]` mm.
   Bin **centers** are symmetric about 0 (with a center exactly at 0).
2. Build a 2D histogram in `(x, z)`; average over x to get a 1D `z` profile.
3. Find raw discrete maximum (`argmax`).
4. Smooth the profile with a Gaussian (σ = `width_mm`) and find its discrete maximum.
5. Fit smoothing splines to both raw and smoothed profiles and extract peak
   positions via `max_of_bspline_positions`.
 
# Returns
A `NamedTuple` with fields:
- `z_profile::Matrix{Float64}` — `Nz × 3` matrix: `[z_center  raw  smooth]`
- `z_max_raw_mm::Float64` — z at raw-profile maximum (mm)
- `z_max_raw_spline_mm::Float64` — z at spline-fitted raw maximum (mm)
- `z_max_smooth_mm::Float64` — z at smoothed-profile maximum (mm)
- `z_max_smooth_spline_mm::Float64` — z at spline-fitted smoothed maximum (mm)
 
# Notes
- Requires a global `DEFAULT_camera_pixel_size::Real` (meters).
- Uses `max_of_bspline_positions` to obtain sub-bin-accurate maxima.
- The `manifold` only affects labels/filenames, not the numerical results.
- [`CQD_analyze_screen_profile`](@ref) is the functionally identical
  counterpart for CQD up/down branch data (same algorithm; `branch` replaces
  `manifold` purely as a plot/filename label).
 
# Throws
- `AssertionError` if `data_mm` is not `N×2` (x,z in mm), if `nx_bins/nz_bins ≤ 0`,
  if `width_mm ≤ 0`, or if `DEFAULT_camera_pixel_size` is not defined.
 
# Example
julia
res = analyze_screen_profile(0.125, hits_mm;
    manifold=:F_top, nx_bins=1, nz_bins=4,
    width_mm=0.15, λ_raw=0.01, λ_smooth=1e-3, add_plot=true)
 
@show res.z_max_smooth_spline_mm
 
# See also
[`max_of_bspline_positions`](@ref), [`QM_analyze_profiles_to_dict`](@ref),
[`CQD_analyze_screen_profile`](@ref), [`smooth_profile`](@ref)
 
"""
function analyze_screen_profile(Ix, data_mm::AbstractMatrix;
    manifold::Union{Symbol,Integer} = :F_top, nx_bins::Integer = 2, nz_bins::Integer = 2,
    add_plot::Bool = false, plot_xrange::Symbol = :all,
    width_mm::Float64 = 0.150, λ_raw::Float64=0.01, λ_smooth::Float64 = 1e-3,
    mode::Symbol=:probability)
 
    @assert size(data_mm,2) == 2 "data_mm must be N×2 (columns: x,z in mm)"
    @assert nx_bins > 0 "nx_bins must be > 0"
    @assert nz_bins > 0 "nz_bins must be > 0"
    @assert width_mm > 0 "width_mm must be positive"
    @assert @isdefined(DEFAULT_camera_pixel_size) "define `DEFAULT_camera_pixel_size` (meters)"
 
    # Fixed analysis limits (mm) — the detector region of interest is the
    # same for every call, regardless of binning choices.
    xlim = (-8.0, 8.0)
    zlim = (-12.5, 12.5)
    xmin, xmax = xlim
    zmin, zmax = zlim
 
    # Bin size in mm (DEFAULT_camera_pixel_size is assumed global in meters)
    x_bin_size = 1e3 * nx_bins * DEFAULT_camera_pixel_size
    z_bin_size = 1e3 * nz_bins * DEFAULT_camera_pixel_size
 
    # --------------------------------------------------------
    # X edges: force symmetric centers around 0
    # --------------------------------------------------------
    # kx = number of bins needed on EACH side of 0 to cover the analysis
    # window; centers then run from -kx*x_bin_size to +kx*x_bin_size with a
    # bin centered exactly at 0, and edges sit half a bin width off each center.
    x_half_range = max(abs(xmin), abs(xmax))
    kx = max(1, ceil(Int, x_half_range / x_bin_size))
    centers_x = collect((-kx:kx) .* x_bin_size)
    edges_x = collect((-(kx + 0.5)) * x_bin_size : x_bin_size : ((kx + 0.5) * x_bin_size))
 
    # --------------------------------------------------------
    # Z edges: force symmetric centers around 0 (same construction as X)
    # --------------------------------------------------------
    z_half_range = max(abs(zmin), abs(zmax))
    kz = max(1, ceil(Int, z_half_range / z_bin_size))
    centers_z = collect((-kz:kz) .* z_bin_size)
    edges_z = collect((-(kz + 0.5)) * z_bin_size : z_bin_size : ((kz + 0.5) * z_bin_size))
 
    # 2D histogram of raw (x, z) hits
    x = @view data_mm[:, 1]
    z = @view data_mm[:, 2]
 
    if mode === :none
        h = fit(Histogram, (x, z), (edges_x, edges_z))                    # raw counts (no normalization)
    elseif mode in (:probability, :pdf, :density)
        h = normalize(fit(Histogram, (x, z), (edges_x, edges_z)); mode=mode)
    else
        throw(ArgumentError("mode must be one of :pdf, :density, :probability, :none, got $mode"))
    end
 
    counts = h.weights  # size: (length(centers_x), length(centers_z))
 
    # z-profile = mean over x bins (collapse the x dimension)
    z_profile_raw = vec(mean(counts, dims = 1))
    z_max_raw_mm = centers_z[argmax(z_profile_raw)]
    z_max_raw_spline_mm, Sfit_raw = max_of_bspline_positions(centers_z,z_profile_raw;λ0=λ_raw)
 
    # Smoothing: Gaussian-convolve the raw profile, then repeat the
    # discrete-max and spline-max extraction on the smoothed curve.
    z_profile_smooth = smooth_profile(centers_z, z_profile_raw, width_mm)
    z_max_smooth_mm = centers_z[argmax(z_profile_smooth)]
    z_max_smooth_spline_mm, Sfit_smooth = max_of_bspline_positions(centers_z,z_profile_smooth;λ0=λ_smooth)
 
    # Combine into one matrix for convenience: [z raw smooth]
    z_profile = hcat(
        centers_z,
        z_profile_raw,
        z_profile_smooth,
    )
 
    if add_plot
        # Uncomment to visualize full 2D histogram:
        # heatmap(centers_x, centers_z, counts', xlabel="x (mm)", ylabel="z (mm)", title="2D Histogram")
 
        # Dense z-grid for plotting the smooth spline curves (independent of
        # the coarser histogram bin centers used for the data points/markers).
        z = range(zmin,zmax,length=max(2000,length(centers_z)))
        xlims_plot = plot_xrange== :right ? (zmin/4, zmax) : plot_xrange == :left ? (zmin, zmax/4) : (zmin, zmax)
        fig=plot(
            title =L"$I_{c}=%$(Ix)\mathrm{A}$",
            xlabel = L"$z$ (mm)",
            ylabel = "mean counts (au)",
            xlims = xlims_plot,
        );
        # Raw profile (data points) and its discrete argmax marker
        plot!(z_profile[:, 1], z_profile[:, 2],
            label = L"Raw $z=%$(round(z_max_raw_mm,digits=4))\mathrm{mm}$",
            line=(:solid,:gray90,2),
            marker=(:circle,:white,1),
            markerstrokecolor=:gray70
        );
        vline!([z_max_raw_mm],
            label=false,
            line=(:solid,:black,1),
        );
        # Spline fit to the raw profile and its refined maximum
        plot!(z,Sfit_raw.(z),
            label=L"Spline Raw $z=%$(round(z_max_raw_spline_mm[1],digits=4))\mathrm{mm}$",
            line=(:forestgreen,:dot,2),
        );
        vline!(z_max_raw_spline_mm,
            label=false,
            line=(:green,:solid,1))
        # Smoothed (Gaussian-convolved) profile and its discrete argmax marker
        plot!(z_profile[:, 1], z_profile[:, 3],
            label = L"Smoothed $z=%$(round(z_max_smooth_mm,digits=4))\mathrm{mm}$",
            line=(:coral3,:dash,2),
        );
        vline!([z_max_smooth_mm],
            label=false,
            line=(:red,:solid,1)
        );
        # Spline fit to the smoothed profile and its refined maximum
        plot!(z,Sfit_smooth.(z),
            label=L"Spline Smoothed $z=%$(round(z_max_smooth_spline_mm[1],digits=4))\mathrm{mm}$",
            line=(:royalblue3,:dot,2),
        );
        vline!(z_max_smooth_spline_mm,
            label=false,
            line=(:blue,:solid,1)
        );
        savefig(fig, joinpath(OUTDIR, "profiles_$(string(manifold))_"*replace(@sprintf("%d", 1e3*Ix), "." => "")*"mA.png"))
    end
 
    return (
        z_profile = z_profile,
        z_max_raw_mm = z_max_raw_mm,
        z_max_raw_spline_mm = z_max_raw_spline_mm[1],
        z_max_smooth_mm = z_max_smooth_mm,
        z_max_smooth_spline_mm = z_max_smooth_spline_mm[1]
    )
end

"""
    CQD_analyze_screen_profile(
        Ix::Real,
        data_mm::AbstractMatrix;
        nx_bins::Integer = 2,
        nz_bins::Integer = 2,
        branch::Symbol = :up,
        add_plot::Bool = false,
        plot_xrange::Symbol = :all,
        width_mm::Float64 = 0.150,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3,
        mode::Symbol = :probability
    ) -> NamedTuple
 
CQD counterpart of [`analyze_screen_profile`](@ref): analyze the vertical **z**
profile of particle hits on a screen for a given coil current, for one CQD
branch (`:up` or `:down`).
 
The algorithm is **identical** to `analyze_screen_profile` — same fixed
analysis windows, same binning/histogram/smoothing/spline-fitting steps, same
return fields. The only difference is the `branch` keyword (replacing
`manifold`), which is used purely to label plot titles and saved-figure
filenames (`profiles_cqd_<branch>_...png` instead of `profiles_<manifold>_...png`)
and has no effect on the numerical analysis.
 
# Arguments
- `Ix::Real`:
  Coil current (A). Used for plot annotation; does not affect the analysis.
- `data_mm::AbstractMatrix`:
  `N×2` array of hit positions **in millimeters** with columns:
  `(:, 1) = x`, `(:, 2) = z`.
 
# Keyword Arguments
- `nx_bins::Integer = 2`, `nz_bins::Integer = 2`:
  Binning multipliers, same role as in `analyze_screen_profile`.
- `branch::Symbol = :up`:
  Which CQD branch this dataset corresponds to (`:up` or `:down`). Forwarded
  to plot titles/filenames only; does not affect computation.
- `add_plot::Bool = false`:
  If `true`, plot raw/smoothed/spline profiles and mark their maxima.
- `plot_xrange::Symbol = :all`:
  `:all` → full z window; `:left` → left quarter; `:right` → right quarter.
- `width_mm::Float64 = 0.150`:
  Gaussian kernel σ (mm) used by `smooth_profile`.
- `λ_raw::Float64 = 0.01`, `λ_smooth::Float64 = 1e-3`:
  Spline regularization parameters for the raw and smoothed profiles.
- `mode::Symbol = :probability`:
  Normalization mode passed to `StatsBase.normalize` for the 2D histogram.
 
# Returns
A `NamedTuple` with fields:
- `z_profile::Matrix{Float64}` — `Nz × 3` matrix: `[z_center  raw  smooth]`
- `z_max_raw_mm::Float64` — z at raw-profile maximum (mm)
- `z_max_raw_spline_mm::Float64` — z at spline-fitted raw maximum (mm)
- `z_max_smooth_mm::Float64` — z at smoothed-profile maximum (mm)
- `z_max_smooth_spline_mm::Float64` — z at spline-fitted smoothed maximum (mm)
 
# Notes
- See [`analyze_screen_profile`](@ref) for the full method description — the
  two functions share the same algorithm verbatim; only the labeling keyword differs.
- Requires a global `DEFAULT_camera_pixel_size::Real` (meters).
 
# Throws
- `AssertionError` if `data_mm` is not `N×2` (x,z in mm), if `nx_bins/nz_bins ≤ 0`,
  if `width_mm ≤ 0`, or if `DEFAULT_camera_pixel_size` is not defined.
 
# See also
[`analyze_screen_profile`](@ref), [`CQD_analyze_profiles_to_dict`](@ref),
[`max_of_bspline_positions`](@ref), [`smooth_profile`](@ref)
 
"""
function CQD_analyze_screen_profile(Ix, data_mm::AbstractMatrix;
    nx_bins::Integer = 2, nz_bins::Integer = 2,
    branch::Symbol = :up, add_plot::Bool = false, plot_xrange::Symbol = :all,
    width_mm::Float64 = 0.150, λ_raw::Float64=0.01, λ_smooth::Float64 = 1e-3,
    mode::Symbol=:probability)
 
    @assert size(data_mm,2) == 2 "data_mm must be N×2 (columns: x,z in mm)"
    @assert nx_bins > 0 "nx_bins must be > 0"
    @assert nz_bins > 0 "nz_bins must be > 0"
    @assert width_mm > 0 "width_mm must be positive"
    @assert @isdefined(DEFAULT_camera_pixel_size) "define `DEFAULT_camera_pixel_size` (meters)"
 
    # Fixed analysis limits (mm) — same window as analyze_screen_profile.
    xlim = (-8.0, 8.0)
    zlim = (-12.5, 12.5)
    xmin, xmax = xlim
    zmin, zmax = zlim
 
    # Bin size in mm (DEFAULT_camera_pixel_size is assumed global in meters)
    x_bin_size = 1e3 * nx_bins * DEFAULT_camera_pixel_size
    z_bin_size = 1e3 * nz_bins * DEFAULT_camera_pixel_size
 
    # --------------------------------------------------------
    # X edges: force symmetric centers around 0
    # --------------------------------------------------------
    x_half_range = max(abs(xmin), abs(xmax))
    kx = max(1, ceil(Int, x_half_range / x_bin_size))
    centers_x = collect((-kx:kx) .* x_bin_size)
    edges_x = collect((-(kx + 0.5)) * x_bin_size : x_bin_size : ((kx + 0.5) * x_bin_size))
 
    # --------------------------------------------------------
    # Z edges: force symmetric centers around 0
    # --------------------------------------------------------
    z_half_range = max(abs(zmin), abs(zmax))
    kz = max(1, ceil(Int, z_half_range / z_bin_size))
    centers_z = collect((-kz:kz) .* z_bin_size)
    edges_z = collect((-(kz + 0.5)) * z_bin_size : z_bin_size : ((kz + 0.5) * z_bin_size))
 
    # 2D histogram of raw (x, z) hits
    x = @view data_mm[:, 1]
    z = @view data_mm[:, 2]
 
    if mode === :none
        h = fit(Histogram, (x, z), (edges_x, edges_z))                    # raw counts (no normalization)
    elseif mode in (:probability, :pdf, :density)
        h = normalize(fit(Histogram, (x, z), (edges_x, edges_z)); mode=mode)
    else
        throw(ArgumentError("mode must be one of :pdf, :density, :probability, :none, got $mode"))
    end
 
    counts = h.weights  # size: (length(centers_x), length(centers_z))
 
    # z-profile = mean over x bins
    z_profile_raw = vec(mean(counts, dims = 1))
    z_max_raw_mm = centers_z[argmax(z_profile_raw)]
    z_max_raw_spline_mm, Sfit_raw = max_of_bspline_positions(centers_z,z_profile_raw;λ0=λ_raw)
 
    # Smoothing
    z_profile_smooth = smooth_profile(centers_z, z_profile_raw, width_mm)
    z_max_smooth_mm = centers_z[argmax(z_profile_smooth)]
    z_max_smooth_spline_mm, Sfit_smooth = max_of_bspline_positions(centers_z,z_profile_smooth;λ0=λ_smooth)
 
    # Combine into one matrix for convenience: [z raw smooth]
    z_profile = hcat(
        centers_z,
        z_profile_raw,
        z_profile_smooth,
    )
 
    if add_plot
        # Uncomment to visualize full 2D histogram:
        # heatmap(centers_x, centers_z, counts', xlabel="x (mm)", ylabel="z (mm)", title="2D Histogram")
 
        # Profiles
        z = range(zmin,zmax,length=max(2000,length(centers_z)))
        xlims_plot = plot_xrange== :right ? (zmin/4, zmax) : plot_xrange == :left ? (zmin, zmax/4) : (zmin, zmax)
        fig=plot(
            title =L"$I_{c}=%$(Ix)\mathrm{A}$",
            xlabel = L"$z$ (mm)",
            ylabel = "mean counts (au)",
            xlims = xlims_plot,
        );
        plot!(z_profile[:, 1], z_profile[:, 2],
            label = L"Raw $z=%$(round(z_max_raw_mm,digits=4))\mathrm{mm}$",
            line=(:solid,:gray90,2),
            marker=(:circle,:white,1),
            markerstrokecolor=:gray70
        );
        vline!([z_max_raw_mm],
            label=false,
            line=(:solid,:black,1),
        );
        plot!(z,Sfit_raw.(z),
            label=L"Spline Raw $z=%$(round(z_max_raw_spline_mm[1],digits=4))\mathrm{mm}$",
            line=(:forestgreen,:dot,2),
        );
        vline!(z_max_raw_spline_mm,
            label=false,
            line=(:green,:solid,1))
        # Convolution
        plot!(z_profile[:, 1], z_profile[:, 3],
            label = L"Smoothed $z=%$(round(z_max_smooth_mm,digits=4))\mathrm{mm}$",
            line=(:coral3,:dash,2),
        );
        vline!([z_max_smooth_mm],
            label=false,
            line=(:red,:solid,1)
        );
        plot!(z,Sfit_smooth.(z),
            label=L"Spline Smoothed $z=%$(round(z_max_smooth_spline_mm[1],digits=4))\mathrm{mm}$",
            line=(:royalblue3,:dot,2),
        );
        vline!(z_max_smooth_spline_mm,
            label=false,
            line=(:blue,:solid,1)
        );
        savefig(fig, joinpath(OUTDIR, "profiles_cqd_$(string(branch))_"*replace(@sprintf("%d", 1e3*Ix), "." => "_")*"mA.png"))
    end
 
    return (
        z_profile = z_profile,
        z_max_raw_mm = z_max_raw_mm,
        z_max_raw_spline_mm = z_max_raw_spline_mm[1],
        z_max_smooth_mm = z_max_smooth_mm,
        z_max_smooth_spline_mm = z_max_smooth_spline_mm[1]
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# 3. QM batch / dictionary wrappers
# ──────────────────────────────────────────────────────────────────────────────

# Shared helpers used by the batch/dictionary wrappers below (sections 3 & 4).
# Factored out because the same blocks were previously repeated verbatim
# across six methods (the result-dict construction) and two methods (the
# manifold-selection branch) — centralizing them here means a future field
# rename or a new manifold group only needs to be edited once.
 
"""
    _result_to_dict(Icoil, result) -> OrderedDict{Symbol,Any}
 
Build the standard per-current result dictionary from a coil current `Icoil`
and the `NamedTuple` returned by [`analyze_screen_profile`](@ref) or
[`CQD_analyze_screen_profile`](@ref).
 
This is the single place the `QM_analyze_profiles_to_dict` and
`CQD_analyze_profiles_to_dict` methods get this schema from — if you ever
add, rename, or drop a field, this is the only function that needs to change.
 
# Returns
`OrderedDict{Symbol,Any}` with keys `:Icoil`, `:z_max_raw_mm`,
`:z_max_raw_spline_mm`, `:z_max_smooth_mm`, `:z_max_smooth_spline_mm`, `:z_profile`.
"""
@inline function _result_to_dict(Icoil, result)
    return OrderedDict{Symbol, Any}(
        :Icoil                  => Icoil,
        :z_max_raw_mm           => result.z_max_raw_mm,
        :z_max_raw_spline_mm    => result.z_max_raw_spline_mm,
        :z_max_smooth_mm        => result.z_max_smooth_mm,
        :z_max_smooth_spline_mm => result.z_max_smooth_spline_mm,
        :z_profile              => result.z_profile,
    )
end

"""
    _select_manifold_img(container, manifold, is_group::Bool, Ispin::Real) -> Matrix{Float64}
 
Select and vertically concatenate hit-position data (columns 7:8, converted
from meters to millimeters) from a level-indexable `container`, according to
`manifold`.
 
`container` must support `container[lvl]` returning a matrix whose columns
7:8 are `[x, z]` in meters, for every level index `lvl` referenced below.
`Ispin` is the nuclear spin quantum number `I`, used for the named-group
boundaries.
 
`manifold` may be:
- `:F_top`    → levels `1 : (2I + 2)`
- `:F_bottom` → levels `(2I + 3) : (4I + 2)`
- `:S_up`     → levels `1 : (2I + 1)`
- `:S_down`   → levels `(2I + 2) : (4I + 2)`
- a list/tuple of level indices, when `is_group` is `true`
- a single numeric level, as an `Integer` or a numeric-looking `Symbol` (e.g. `:3`)
 
Shared by both `QM_analyze_profiles_to_dict` methods that need manifold
selection (the `OrderedDict`-based and JLD2-path-based ones) — `container` is
`data[:data][i]` for the former and `data_i = file["screen/I\$i"]` for the
latter; the selection logic itself is otherwise identical between the two.
 
# Throws
- `AssertionError` if `manifold` is a list/tuple but empty, or if a non-numeric
  symbol is given for the single-level case.
"""
function _select_manifold_img(container, manifold, is_group::Bool, Ispin::Real)
    if manifold === :F_top
        return 1e3 .* vcat((xz[:, 7:8] for xz in container[1:Int(2*Ispin+2)])...)
    elseif manifold === :F_bottom
        return 1e3 .* vcat((xz[:, 7:8] for xz in container[Int(2*Ispin+3):Int(4*Ispin+2)])...)
    elseif manifold === :S_up
        return 1e3 .* vcat((xz[:, 7:8] for xz in container[1:Int(2*Ispin+1)])...)
    elseif manifold === :S_down
        return 1e3 .* vcat((xz[:, 7:8] for xz in container[Int(2*Ispin+2):Int(4*Ispin+2)])...)
    elseif is_group
        lvls = collect(manifold)
        @assert !isempty(lvls) "Empty level list passed as manifold"
        return 1e3 .* vcat((container[lvl][:, 7:8] for lvl in lvls)...)
    else
        lvl = manifold isa Integer ? Int(manifold) :
              begin
                  v = tryparse(Int, string(manifold))
                  @assert v !== nothing "Non-numeric manifold '$manifold' (expected e.g. :1, :2, ...)"
                  v
              end
        return 1e3 .* container[lvl][:, 7:8]
    end
end


"""
    QM_analyze_profiles_to_dict(
        data::OrderedDict{Symbol,Any},
        p::AtomParams;
        manifold::Union{Symbol,Integer} = :F_bottom,
        n_bins::Tuple = (1, 4),
        width_mm::Float64 = 0.150,
        add_plot::Bool = false,
        plot_xrange::Symbol = :all,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3,
        mode::Symbol = :probability
    ) -> OrderedDict{Int, OrderedDict{Symbol,Any}}
 
Batch-process screen-hit datasets across multiple coil currents by repeatedly
calling [`analyze_screen_profile`](@ref). For each current `Ix = data[:Icoils][i]`,
this function extracts the (x, z) hit positions from `data[:data][i]`, converts
them from **meters** to **millimeters**, analyzes the vertical z-profile, and
stores the peak summaries in a nested dictionary keyed by the dataset index `i`.
 
This is one of four `QM_analyze_profiles_to_dict` methods, differing only in
how the (current, hit-position) data is supplied — see "See also" below for
the others.
 
# Inputs
- `data::OrderedDict{Symbol,Any}`:
  Must contain:
  - `:Icoils :: AbstractVector{<:Real}` — coil currents (A), length `N`.
  - `:data` — a length-`N` container; each `data[:data][i]` is an indexable
    collection of matrices whose **columns 7:8** are `[x, z]` in **meters**.
- `p::AtomParams`:
  Used to determine level grouping via `p.Ispin` when selecting manifolds.
 
# Keyword Arguments
- `manifold::Union{Symbol,Integer} = :F_bottom`:
  Which manifold(s) to aggregate:
  - `:F_top`    → vertically concatenate levels `1 : (2I + 2)`
  - `:F_bottom` → vertically concatenate levels `(2I + 3) : (4I + 2)`
  - `:1`, `:2`, … (numeric-like symbol) → use that single level
  Here `I = p.Ispin`. In all cases, columns `7:8` are taken and multiplied by
  `1e3` to convert to **mm**.
- `n_bins::Tuple = (1, 4)`:
  `(nx_bins, nz_bins)` binning multipliers forwarded to `analyze_screen_profile`.
- `width_mm::Float64 = 0.150`:
  Gaussian σ (mm) used for profile smoothing.
- `add_plot::Bool = false`, `plot_xrange::Symbol = :all`:
  Plot options forwarded to `analyze_screen_profile`.
- `λ_raw::Float64 = 0.01`, `λ_smooth::Float64 = 1e-3`:
  Spline regularization parameters forwarded downstream.
- `mode::Symbol = :probability`:
  Histogram normalization mode (`:probability`, `:pdf`, etc.).
 
# Returns
`OrderedDict{Int, OrderedDict{Symbol,Any}}` where, for each index `i`:
- `:Icoil`                   → current `data[:Icoils][i]` (A)
- `:z_max_raw_mm`            → z at raw-profile maximum (mm)
- `:z_max_raw_spline_mm`     → z at spline-fitted raw maximum (mm)
- `:z_max_smooth_mm`         → z at smoothed-profile maximum (mm)
- `:z_max_smooth_spline_mm`  → z at spline-fitted smoothed maximum (mm)
- `:z_profile`               → `Nz × 3` matrix `[z_center  raw  smooth]`
 
# Notes
- Requires that `analyze_screen_profile` is available and that the global
  `DEFAULT_camera_pixel_size` (meters) is defined for bin sizing in that
  routine.
- Level selection uses `p.Ispin`; e.g., for `I = 3/2`, `:F_top` picks levels
  `1:5` and `:F_bottom` picks `6:8` (1-based).
- Throws an `AssertionError` if `data` is missing `:Icoils` or `:data`, or if
  `manifold` is a non-numeric symbol not equal to `:F_top`/`:F_bottom`.
 
# Example
julia
out = QM_analyze_profiles_to_dict(run_data, atom_params;
    manifold=:F_top, n_bins=(1,4), width_mm=0.15,
    add_plot=true, plot_xrange=:right, λ_raw=0.01, λ_smooth=1e-3)
 
z_first = out[1][:z_max_smooth_spline_mm]
 
# See also
[`analyze_screen_profile`](@ref), [`max_of_bspline_positions`](@ref), [`smooth_profile`](@ref)
 
"""
function QM_analyze_profiles_to_dict(data::OrderedDict{Symbol,Any}, p::AtomParams;
    manifold::Union{Symbol,Integer,AbstractVector{<:Integer},Tuple{Vararg{Integer}}} = :F_bottom,
    n_bins::Tuple = (1,4), width_mm::Float64 = 0.150,
    add_plot::Bool = false, plot_xrange::Symbol = :all,
    λ_raw::Float64 = 0.01, λ_smooth::Float64 = 1e-3, mode::Symbol = :probability)
 
    @assert haskey(data, :Icoils) "missing :Icoils"
    @assert haskey(data, :data)   "missing :data"
 
    nx_bins = n_bins[1]
    nz_bins = n_bins[2]
 
    Ix = data[:Icoils]
    out = OrderedDict{Int, OrderedDict{Symbol, Any}}()
 
    # `manifold` can be a named group (:F_top/:F_bottom/:S_up/:S_down), an
    # explicit list/tuple of level indices, or a single numeric level. For
    # plot/filename labeling, anything list-like collapses to :custom.
    is_group = manifold isa AbstractVector{<:Integer} || manifold isa Tuple{Vararg{Integer}}
    manifold_tag = is_group ? :custom : manifold
 
    for i in eachindex(Ix)
        img_F = _select_manifold_img(data[:data][i], manifold, is_group, p.Ispin)
 
        result = analyze_screen_profile(Ix[i], img_F;
                manifold=manifold_tag,
                nx_bins=nx_bins, nz_bins=nz_bins,
                width_mm=width_mm,
                add_plot=add_plot,
                plot_xrange=plot_xrange,
                λ_raw=λ_raw, λ_smooth=λ_smooth,mode=mode)
 
        out[i] = _result_to_dict(Ix[i], result)
    end
    return out
end

"""
    QM_analyze_profiles_to_dict(
        jld_path::AbstractString,
        p::AtomParams;
        manifold::Union{Symbol,Integer} = :F_bottom,
        n_bins::Tuple = (1, 4),
        width_mm::Float64 = 0.150,
        add_plot::Bool = false,
        plot_xrange::Symbol = :all,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3,
        mode::Symbol = :probability
    ) -> OrderedDict{Int, OrderedDict{Symbol,Any}}
 
JLD2-file variant of [`QM_analyze_profiles_to_dict`](@ref): identical behavior
and return format, but reads coil currents and per-level screen-hit data
directly from a saved JLD2 file rather than an in-memory `OrderedDict`.
 
# Inputs
- `jld_path::AbstractString`:
  Path to a JLD2 file containing:
  - `"meta/Icoils"` — coil currents (A), length `N`.
  - `"meta/levels"` — level metadata (read but currently unused here).
  - `"screen/I\$i"` for `i in 1:N` — indexable by level, where each level's
    matrix has **columns 7:8** equal to `[x, z]` in **meters** (same layout
    as `data[:data][i]` in the `OrderedDict` method).
- `p::AtomParams`:
  Used to determine level grouping via `p.Ispin` when selecting manifolds.
 
# Keyword Arguments
Identical to the `OrderedDict` method — see
[`QM_analyze_profiles_to_dict(data::OrderedDict{Symbol,Any}, p::AtomParams)`](@ref)
for the full description of `manifold`, `n_bins`, `width_mm`, `add_plot`,
`plot_xrange`, `λ_raw`, `λ_smooth`, and `mode`.
 
# Returns
Same structure as the `OrderedDict` method: `OrderedDict{Int, OrderedDict{Symbol,Any}}`
keyed by dataset index, with fields `:Icoil`, `:z_max_raw_mm`,
`:z_max_raw_spline_mm`, `:z_max_smooth_mm`, `:z_max_smooth_spline_mm`, `:z_profile`.
 
# Notes
- The file is opened read-only (`jldopen(jld_path, "r")`) and closed
  automatically via the `do`-block form.
- Throws an `AssertionError` if `"meta/Icoils"`, `"meta/levels"`, or any
  `"screen/I\$i"` key is missing from the file.
 
# Example
julia
out = QM_analyze_profiles_to_dict("run42.jld2", atom_params; manifold=:F_top)
 
# See also
[`QM_analyze_profiles_to_dict(data::OrderedDict{Symbol,Any}, p::AtomParams)`](@ref),
[`analyze_screen_profile`](@ref)
 
"""
function QM_analyze_profiles_to_dict(jld_path::AbstractString, p::AtomParams;
    manifold::Union{Symbol,Integer,AbstractVector{<:Integer},Tuple{Vararg{Integer}}} = :F_bottom,
    n_bins::Tuple = (1,4), width_mm::Float64 = 0.150,
    add_plot::Bool = false, plot_xrange::Symbol = :all,
    λ_raw::Float64 = 0.01, λ_smooth::Float64 = 1e-3, mode::Symbol = :probability)
 
    out = OrderedDict{Int, OrderedDict{Symbol, Any}}()
 
    nx_bins = n_bins[1]
    nz_bins = n_bins[2]
 
    is_group = manifold isa AbstractVector{<:Integer} || manifold isa Tuple{Vararg{Integer}}
    manifold_tag = is_group ? :custom : manifold
 
    jldopen(jld_path, "r") do file
        @assert haskey(file, "meta/Icoils") "missing JLD2 key: meta/Icoils"
        @assert haskey(file, "meta/levels") "missing JLD2 key: meta/levels"
 
        Ix = file["meta/Icoils"]   # vector of currents
        # levels = file["meta/levels"]  # if you need it later
 
        nI = length(Ix)
 
        for i in 1:nI
            screen_key = "screen/I$(i)"
            @assert haskey(file, screen_key) "missing JLD2 key: $screen_key"
 
            # This should be exactly what used to be data_alive_screen[i]
            data_i = file[screen_key]
 
            # data_i is expected to be indexable by level: data_i[lvl] is a matrix,
            # with columns including 7:8 = (x,z) in meters, converted to mm below.
            img_F = _select_manifold_img(data_i, manifold, is_group, p.Ispin)
 
            result = analyze_screen_profile(Ix[i], img_F;
                manifold=manifold_tag,
                nx_bins=nx_bins, nz_bins=nz_bins,
                width_mm=width_mm,
                add_plot=add_plot,
                plot_xrange=plot_xrange,
                λ_raw=λ_raw, λ_smooth=λ_smooth, mode=mode)
 
            out[i] = _result_to_dict(Ix[i], result)
        end
 
        return out
    end
end

"""
    QM_analyze_profiles_to_dict(
        Ix::AbstractVector,
        imgs::Vector,
        p::AtomParams;
        manifold = :F_bottom,
        n_bins::Tuple = (1, 4),
        width_mm::Float64 = 0.150,
        add_plot::Bool = false,
        plot_xrange::Symbol = :all,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3,
        mode::Symbol = :probability
    ) -> OrderedDict{Int, OrderedDict{Symbol,Any}}
 
Pre-extracted-data variant of [`QM_analyze_profiles_to_dict`](@ref): batch-process
a vector of coil currents `Ix` against a matching vector of already-extracted,
already-converted hit-position matrices `imgs`, with **no manifold-based level
selection or meters→mm conversion performed here** — both must already be done
by the caller before calling this method (use this when your data layout isn't
covered by the `OrderedDict` or JLD2-file methods).
 
# Inputs
- `Ix::AbstractVector`:
  Coil currents (A), length `N`.
- `imgs::Vector`:
  Length-`N` vector where `imgs[i]` is an `M×2` matrix of `[x, z]` hit
  positions **already in millimeters** for current `Ix[i]` (passed straight
  through to [`analyze_screen_profile`](@ref), which expects mm).
- `p::AtomParams`:
  Accepted for call-signature consistency with the other
  `QM_analyze_profiles_to_dict` methods, but **not used** in this method body —
  manifold/level selection (the only place `p.Ispin` matters) is assumed to
  have already happened upstream, since `imgs` is supplied pre-extracted.
 
# Keyword Arguments
- `manifold = :F_bottom`:
  Forwarded only as a plot/filename label (via `analyze_screen_profile`'s
  `manifold` keyword) — no level selection happens here, so this has no effect
  on which data is analyzed, only on labeling.
- `n_bins::Tuple = (1, 4)`, `width_mm`, `add_plot`, `plot_xrange`, `λ_raw`,
  `λ_smooth`, `mode`: forwarded to `analyze_screen_profile`; see
  [`QM_analyze_profiles_to_dict(data::OrderedDict{Symbol,Any}, p::AtomParams)`](@ref)
  for the full description of each.
 
# Returns
`OrderedDict{Int, OrderedDict{Symbol,Any}}` keyed by `i = eachindex(Ix)`, with
the same fields as the other `QM_analyze_profiles_to_dict` methods: `:Icoil`,
`:z_max_raw_mm`, `:z_max_raw_spline_mm`, `:z_max_smooth_mm`,
`:z_max_smooth_spline_mm`, `:z_profile`.
 
# Example
julia
# Ix and imgs prepared elsewhere (already mm, already manifold-selected)
out = QM_analyze_profiles_to_dict(Ix, imgs, atom_params; manifold=:F_top)
 
# See also
[`QM_analyze_profiles_to_dict(Ix_i::Real, img_i::Matrix, p::AtomParams)`](@ref)
(single-current version of this method), [`analyze_screen_profile`](@ref)
 
"""
function QM_analyze_profiles_to_dict(Ix::AbstractVector, imgs::Vector, p::AtomParams;
    manifold = :F_bottom,
    n_bins::Tuple = (1,4), width_mm::Float64 = 0.150,
    add_plot::Bool = false, plot_xrange::Symbol = :all,
    λ_raw::Float64 = 0.01, λ_smooth::Float64 = 1e-3, mode::Symbol = :probability)
 
    is_group     = manifold isa AbstractVector{<:Integer} || manifold isa Tuple{Vararg{Integer}}
    manifold_tag = is_group ? :custom : manifold
    nx_bins, nz_bins = n_bins
    out = OrderedDict{Int, OrderedDict{Symbol, Any}}()
 
    for i in eachindex(Ix)
        result = analyze_screen_profile(Ix[i], imgs[i];
            manifold     = manifold_tag,
            nx_bins      = nx_bins,
            nz_bins      = nz_bins,
            width_mm     = width_mm,
            add_plot     = add_plot,
            plot_xrange  = plot_xrange,
            λ_raw        = λ_raw,
            λ_smooth     = λ_smooth,
            mode         = mode)
 
        out[i] = _result_to_dict(Ix[i], result)
    end
    return out
end
 
"""
    QM_analyze_profiles_to_dict(
        Ix_i::Real,
        img_i::Matrix,
        p::AtomParams;
        manifold = :F_bottom,
        n_bins::Tuple = (1, 4),
        width_mm::Float64 = 0.150,
        add_plot::Bool = false,
        plot_xrange::Symbol = :all,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3,
        mode::Symbol = :probability
    ) -> OrderedDict{Symbol,Any}
 
Single-current variant of [`QM_analyze_profiles_to_dict`](@ref): analyzes one
already-extracted, already-converted hit-position matrix for one coil current
and returns a single flat `OrderedDict` (not nested by index), since there's
only one dataset.
 
Same role as the `(Ix::AbstractVector, imgs::Vector, p::AtomParams)` method,
but for a single `(current, image)` pair rather than a batch — useful for
ad-hoc single-current checks without constructing length-1 vectors.
 
# Inputs
- `Ix_i::Real`: a single coil current (A).
- `img_i::Matrix`: `M×2` matrix of `[x, z]` hit positions **already in
  millimeters**, passed straight through to [`analyze_screen_profile`](@ref).
- `p::AtomParams`:
  Accepted for call-signature consistency with the other
  `QM_analyze_profiles_to_dict` methods, but **not used** in this method body
  (see the batch `(Ix::AbstractVector, imgs::Vector, ...)` method for the same note).
 
# Keyword Arguments
Same as the batch pre-extracted-data method — `manifold`, `n_bins`, `width_mm`,
`add_plot`, `plot_xrange`, `λ_raw`, `λ_smooth`, `mode` — all forwarded to
[`analyze_screen_profile`](@ref).
 
# Returns
A single (non-nested) `OrderedDict{Symbol,Any}` with fields `:Icoil`,
`:z_max_raw_mm`, `:z_max_raw_spline_mm`, `:z_max_smooth_mm`,
`:z_max_smooth_spline_mm`, `:z_profile` — the same fields used per-entry by the
batch methods, just unwrapped from the outer index dictionary.
 
# Example
julia
res = QM_analyze_profiles_to_dict(0.125, img_mm, atom_params; manifold=:F_top)
res[:z_max_smooth_spline_mm]
 
# See also
[`QM_analyze_profiles_to_dict(Ix::AbstractVector, imgs::Vector, p::AtomParams)`](@ref)
(batch version of this method), [`analyze_screen_profile`](@ref)
 
"""
function QM_analyze_profiles_to_dict(Ix_i::Real, img_i::Matrix, p::AtomParams;
    manifold = :F_bottom,
    n_bins::Tuple = (1,4), width_mm::Float64 = 0.150,
    add_plot::Bool = false, plot_xrange::Symbol = :all,
    λ_raw::Float64 = 0.01, λ_smooth::Float64 = 1e-3, mode::Symbol = :probability)
 
    is_group     = manifold isa AbstractVector{<:Integer} || manifold isa Tuple{Vararg{Integer}}
    manifold_tag = is_group ? :custom : manifold
    nx_bins, nz_bins = n_bins
 
    result = analyze_screen_profile(Ix_i, img_i;
        manifold    = manifold_tag,
        nx_bins     = nx_bins,
        nz_bins     = nz_bins,
        width_mm    = width_mm,
        add_plot    = add_plot,
        plot_xrange = plot_xrange,
        λ_raw       = λ_raw,
        λ_smooth    = λ_smooth,
        mode        = mode)
 
    return _result_to_dict(Ix_i, result)
end


# ──────────────────────────────────────────────────────────────────────────────
# 4. CQD batch / dictionary wrappers
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# 4. CQD batch / dictionary wrappers
# ──────────────────────────────────────────────────────────────────────────────
"""
    CQD_analyze_profiles_to_dict(
        data::OrderedDict{Symbol,Any};
        n_bins::Tuple = (1, 4),
        width_mm::Float64 = 0.150,
        branch::Symbol = :up,
        add_plot::Bool = false,
        plot_xrange::Symbol = :all,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3,
        mode::Symbol = :probability
    ) -> OrderedDict{Int, OrderedDict{Symbol,Any}}
 
CQD counterpart of [`QM_analyze_profiles_to_dict`](@ref): batch-process a single
CQD branch's (`:up` or `:down`) screen-hit datasets across multiple coil
currents, by repeatedly calling [`CQD_analyze_screen_profile`](@ref).
 
Unlike the QM version, there is no manifold/level selection here — each CQD
dataset already corresponds to exactly one branch, so the hit positions are
read directly from a fixed pair of columns.
 
# Inputs
- `data::OrderedDict{Symbol,Any}`:
  Must contain:
  - `:Icoils :: AbstractVector{<:Real}` — coil currents (A), length `N`.
  - `:data` — a length-`N` container; each `data[:data][i]` is a matrix whose
    **columns 9:10** are `[x, z]` in **meters** (note: columns 9:10, not 7:8 as
    in the QM methods — CQD trajectory data carries two extra columns, the
    drawn electron/nuclear angles `(θe, θn)`, ahead of the screen position
    columns; see [`generate_CQDinitial_conditions`](@ref)).
 
# Keyword Arguments
- `n_bins::Tuple = (1, 4)`:
  `(nx_bins, nz_bins)` binning multipliers forwarded to `CQD_analyze_screen_profile`.
- `width_mm::Float64 = 0.150`:
  Gaussian σ (mm) used for profile smoothing.
- `branch::Symbol = :up`:
  Which CQD branch this dataset represents; forwarded purely as a plot/filename
  label (see [`CQD_analyze_screen_profile`](@ref)) — does not affect computation.
- `add_plot::Bool = false`, `plot_xrange::Symbol = :all`:
  Plot options forwarded to `CQD_analyze_screen_profile`.
- `λ_raw::Float64 = 0.01`, `λ_smooth::Float64 = 1e-3`:
  Spline regularization parameters forwarded downstream.
- `mode::Symbol = :probability`:
  Histogram normalization mode (`:probability`, `:pdf`, etc.).
 
# Returns
`OrderedDict{Int, OrderedDict{Symbol,Any}}` where, for each index `i`:
- `:Icoil`                   → current `data[:Icoils][i]` (A)
- `:z_max_raw_mm`            → z at raw-profile maximum (mm)
- `:z_max_raw_spline_mm`     → z at spline-fitted raw maximum (mm)
- `:z_max_smooth_mm`         → z at smoothed-profile maximum (mm)
- `:z_max_smooth_spline_mm`  → z at spline-fitted smoothed maximum (mm)
- `:z_profile`               → `Nz × 3` matrix `[z_center  raw  smooth]`
 
# Notes
- Requires that `CQD_analyze_screen_profile` is available and that the global
  `DEFAULT_camera_pixel_size` (meters) is defined for bin sizing in that routine.
- Throws an `AssertionError` if `data` is missing `:Icoils` or `:data`.
 
# Example
julia
out_up   = CQD_analyze_profiles_to_dict(up_branch_data;   branch=:up)
out_down = CQD_analyze_profiles_to_dict(down_branch_data; branch=:down)
 
# See also
[`CQD_analyze_screen_profile`](@ref),
[`CQD_analyze_profiles_to_dict(filepath::String)`](@ref) (JLD2-file variant),
[`QM_analyze_profiles_to_dict`](@ref)
 
"""
function CQD_analyze_profiles_to_dict(data::OrderedDict{Symbol,Any};
    n_bins::Tuple = (1,4), width_mm::Float64 = 0.150,
    branch::Symbol = :up, add_plot::Bool = false, plot_xrange::Symbol = :all,
    λ_raw::Float64 = 0.01, λ_smooth::Float64 = 1e-3, mode::Symbol = :probability)
 
    @assert haskey(data, :Icoils) "missing :Icoils"
    @assert haskey(data, :data)   "missing :data"
 
    nx_bins = n_bins[1]
    nz_bins = n_bins[2]
 
    Ix = data[:Icoils]
    out = OrderedDict{Int, OrderedDict{Symbol, Any}}()
    for i in eachindex(Ix)
        img = 1e3 .* data[:data][i][:,9:10] # mm — columns 9:10 = [x, z] for CQD trajectory data
 
        result = CQD_analyze_screen_profile(Ix[i], img;
                nx_bins=nx_bins, nz_bins=nz_bins,
                width_mm=width_mm,
                add_plot=add_plot,
                branch=branch,
                plot_xrange=plot_xrange,
                λ_raw=λ_raw, λ_smooth=λ_smooth,mode=mode)
 
        out[i] = _result_to_dict(Ix[i], result)
    end
    return out
end
 
"""
    CQD_analyze_profiles_to_dict(
        filepath::String;
        n_bins::Tuple = (1, 4),
        width_mm::Float64 = 0.150,
        branch::Symbol = :up,
        add_plot::Bool = false,
        plot_xrange::Symbol = :all,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3,
        mode::Symbol = :probability
    ) -> OrderedDict{Int, OrderedDict{Symbol,Any}}
 
JLD2-file variant of [`CQD_analyze_profiles_to_dict`](@ref): identical behavior
and return format, but reads coil currents and per-current screen-hit data
directly from a saved JLD2 file rather than an in-memory `OrderedDict`.
 
# Inputs
- `filepath::String`:
  Path to a JLD2 file containing:
  - `"meta/Iw"` — coil currents (A), length `N`.
  - `"data/final/I\$idx"` for `idx in 1:N` — a matrix whose **columns 9:10**
    are `[x, z]` in **meters**. The inline comment at the read site notes
    columns 1:6 as `x y z vx vy vz` (the pass-flag column has already been
    dropped upstream); columns 9:10 holding the final screen position is
    consistent with the `OrderedDict` variant of this function and with
    [`generate_CQDinitial_conditions`](@ref)'s output layout (columns 7:8 =
    the drawn CQD angles `θe, θn`).
 
# Keyword Arguments
Identical to the `OrderedDict` method — see
[`CQD_analyze_profiles_to_dict(data::OrderedDict{Symbol,Any})`](@ref) for the
full description of `n_bins`, `width_mm`, `branch`, `add_plot`, `plot_xrange`,
`λ_raw`, `λ_smooth`, and `mode`.
 
# Returns
Same structure as the `OrderedDict` method: `OrderedDict{Int, OrderedDict{Symbol,Any}}`
keyed by dataset index, with fields `:Icoil`, `:z_max_raw_mm`,
`:z_max_raw_spline_mm`, `:z_max_smooth_mm`, `:z_max_smooth_spline_mm`, `:z_profile`.
 
# Notes
- The file is opened read-only (`jldopen(filepath, "r")`) and closed
  automatically via the `do`-block form.
 
# Example
julia
out = CQD_analyze_profiles_to_dict("cqd_run42.jld2"; branch=:down)
 
# See also
[`CQD_analyze_profiles_to_dict(data::OrderedDict{Symbol,Any})`](@ref)
(in-memory variant), [`CQD_analyze_screen_profile`](@ref)
 
"""
function CQD_analyze_profiles_to_dict(filepath::String;
    n_bins::Tuple       = (1,4),
    width_mm::Float64   = 0.150,
    branch::Symbol      = :up,
    add_plot::Bool      = false,
    plot_xrange::Symbol = :all,
    λ_raw::Float64      = 0.01,
    λ_smooth::Float64   = 1e-3,
    mode::Symbol        = :probability)
 
    nx_bins, nz_bins = n_bins
 
    jldopen(filepath, "r") do f
        Icoils = f["meta/Iw"]
        out    = OrderedDict{Int, OrderedDict{Symbol, Any}}()
 
        for (idx, Iw) in enumerate(Icoils)
            screen = f["data/final/I$(idx)"]          # columns: x y z vx vy vz (pass already dropped)
            img    = 1e3 .* screen[:, [9, 10]]   # x and z in mm
 
            result = CQD_analyze_screen_profile(Iw, img;
                         nx_bins     = nx_bins,
                         nz_bins     = nz_bins,
                         width_mm    = width_mm,
                         add_plot    = add_plot,
                         branch      = branch,
                         plot_xrange = plot_xrange,
                         λ_raw       = λ_raw,
                         λ_smooth    = λ_smooth,
                         mode        = mode)
 
            out[idx] = _result_to_dict(Iw, result)
        end
 
        return out
    end
end