"""
    max_of_bspline_positions(z, y; λ0=1e-3, order=4, n_peaks=1, n_scan=max(400, length(z)), sep=1e-6)

    Fit a smoothing B-spline to `(z, y)` data and return the positions of the most prominent local maxima.

    # Arguments
    - `z::AbstractVector`: Sorted vector of x-coordinates (domain points).
    - `y::AbstractVector`: Vector of y-values corresponding to `z`.
    - `λ0::Real=1e-3`: Smoothing parameter for the spline fit. Smaller values follow the data more closely; larger values yield smoother curves.
    - `order::Int=4`: B-spline order (4 = cubic).
    - `n_peaks::Int=1`: Number of top peaks to return, sorted by descending spline value.
    - `n_scan::Int=max(400, length(z))`: Number of points in the dense scan for detecting candidate peaks.
    - `sep::Real=1e-6`: Minimum separation between peaks (in `z` units) to consider them distinct.

    # Method
    1. Fits a smoothing B-spline `S(z)` using `BSplineKit.fit` with uniform weights from `compute_weights`.
    2. Performs a dense scan to find candidate peak locations by detecting sign changes in the finite-difference slope.
    3. Refines each candidate's position using a local Brent optimization in a small bracket around the candidate.
    4. Removes peaks closer than `sep` to each other.
    5. Sorts the remaining peaks by their spline height and returns the positions of the top `n_peaks`.

    # Returns
    - `Vector{Float64}`: Positions of the most prominent peaks in descending order of height.

    # Notes
    - Only peak positions are returned; heights can be obtained by evaluating the returned spline at those positions if needed.
    - The function is robust to multiple local maxima and will always include endpoints as candidates in case the maximum lies at the boundary.
"""
function max_of_bspline_positions(z::AbstractVector,y::AbstractVector;
    λ0::Real=0.01, order::Int=4, n_peaks::Int=1, n_scan::Int=max(400, length(z)),sep::Real=1e-6)

    @assert length(z) == length(y) && issorted(z)
    a, b = extrema(z)

    # Fit smoothing spline
    S = BSplineKit.fit(BSplineOrder(order), z, y, λ0; weights=compute_weights(z, λ0))

    # --- 1) Dense scan to find candidate peaks
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
    push!(cand, a); push!(cand, b)            # endpoints too
    cand = unique(sort(cand))

    # --- 2) Refine each candidate in a small bracket with Brent
    negS(x) = -S(x)
    function refine(x0)
        δ = 2dx
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
        ẑ = refine(c)
        # de-duplicate close peaks
        if isempty(peaks) || all(abs(ẑ - p) > sep for p in peaks)
            push!(peaks, ẑ)
        end
    end

    # --- 3) Sort by actual spline height and return positions only
    ord = sortperm(S.(peaks), rev=true)
    peaks = peaks[ord]
    return peaks[1:min(n_peaks, length(peaks))] , S # positions only
end

"""
    analyze_screen_profile(
        Ix::Real,
        data_mm::AbstractMatrix;
        nz_bins::Integer = 2,
        width_mm::Float64 = 0.1,
        add_plot::Bool = false,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3
    ) -> NamedTuple

    Analyze the vertical (z) profile of particle hits on a screen for a given
    coil current, using 2D histogramming, smoothing, and spline fitting to
    identify peak positions.

    This version is designed for both **single-current analysis** and **batch
    processing** via `analyze_profiles_to_dict`. The first argument `Ix` is
    the coil current (in A) and is used to annotate plots, allowing you to
    automatically generate titled plots for each current in a loop.

    The function:
    1. Builds a 2D histogram of hit positions (x, z) in **millimeters** with
    bin centers symmetric about zero.
    2. Extracts the mean **z-profile** (averaged over x) from the histogram.
    3. Finds the z-location of the maximum in the raw profile and in the raw
    profile fitted with a smoothing spline.
    4. Smooths the profile with a Gaussian kernel, then finds the z-location
    of the maximum from both the smoothed data and a smoothing spline fit.
    5. Optionally plots raw, smoothed, and spline-fitted profiles, annotated
    with the corresponding maxima and titled with the coil current.

    # Arguments
    - `Ix::Real`:  
        Coil current (in A) for labeling the plot (especially in batch runs).
    - `data_mm::AbstractMatrix`:  
        N×2 array of hit positions in **mm**, where column 1 = x, column 2 = z.
    - `nz_bins::Integer`:  
        Binning multiplier; bin size in mm is  
        `bin_size = 1e3 * nz_bins * default_camera_pixel_size`,  
        with `default_camera_pixel_size` assumed global in meters.
    - `width_mm::Float64`:  
        Gaussian kernel width σ (mm) for `smooth_profile`.
    - `add_plot::Bool`:  
        If `true`, plots the raw, smoothed, and spline profiles.
    - `λ_raw::Float64`:  
        Regularization parameter for spline fitting of raw profile.
    - `λ_smooth::Float64`:  
        Regularization parameter for spline fitting of smoothed profile.

    # Implementation details
    - Analysis limits: `x ∈ [-9.0, 9.0]` mm, `z ∈ [-12.5, 12.5]` mm.
    - Bin centers are symmetric around 0 with a center exactly at 0.
    - Maxima are identified using `argmax` for discrete data and
    `max_of_bspline_positions` for spline fits.

    # Returns
    A `NamedTuple` with:
    - `z_profile` :: Matrix (Nz × 3) → `[z_center, raw_counts, smoothed_counts]`
    - `z_max_raw_mm` :: z at raw profile maximum [mm]
    - `z_max_raw_spline_mm` :: z at spline-fitted raw maximum [mm]
    - `z_max_smooth_mm` :: z at smoothed profile maximum [mm]
    - `z_max_smooth_spline_mm` :: z at spline-fitted smoothed maximum [mm]

    # Notes
    - Pass `Ix` from a loop over coil currents in `analyze_profiles_to_dict` to
    automatically label each plot with its current.
    - Intended for integration into workflows that analyze many currents in
    sequence.
"""
function analyze_screen_profile(Ix, data_mm::AbstractMatrix; 
    m_mom::Symbol = :qm, manifold::Symbol = :F_top, nx_bins::Integer = 2, nz_bins::Integer = 2, width_mm::Float64 = 0.150, add_plot::Bool = false, λ_raw::Float64=0.01, λ_smooth::Float64 = 1e-3)

    @assert size(data_mm,2) == 2 "data_mm must be N×2 (columns: x,z in mm)"
    @assert nx_bins > 0 "nx_bins must be > 0"
    @assert nz_bins > 0 "nz_bins must be > 0"
    @assert width_mm > 0 "width_mm must be positive"

    # Fixed analysis limits
    xlim = (-8.0, 8.0)
    zlim = (-10.5, 10.5)
    xmin, xmax = xlim
    zmin, zmax = zlim

    # Bin size in mm (default_camera_pixel_size is assumed global in meters)
    x_bin_size = 1e3 * nx_bins * default_camera_pixel_size
    z_bin_size = 1e3 * nz_bins * default_camera_pixel_size

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

    # 2D histogram
    x = @view data_mm[:, 1]
    z = @view data_mm[:, 2]
    h = fit(Histogram, (x, z), (edges_x, edges_z))
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
        xlims_plot = m_mom == :up ? (zmin/4, zmax) : m_mom == :dw ? (zmin, zmax/4) : (zmin, zmax)
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
    analyze_profiles_to_dict(
        Icoils::AbstractVector,
        screen_up::AbstractDict{<:Integer,<:AbstractMatrix};
        n_bins::Integer = 2,
        width_mm::Float64 = 0.1,
        add_plot::Bool = false,
        λ_raw::Float64 = 0.01,
        λ_smooth::Float64 = 1e-3,
        store_profiles::Bool = true,
    ) -> OrderedDict{Int, OrderedDict{Symbol, Any}}

    Run `analyze_screen_profile` for each coil current `Icoils[i]` with its
    corresponding screen data `screen_up[i]`, collecting results in a nested
    dictionary keyed by the **index** `i` (1..length(Icoils)).

    This is the batch companion to `analyze_screen_profile(Ix, data_mm; ...)`.
    It converts each dataset from meters to millimeters (taking columns 1=x and
    3=z), calls the single-dataset analyzer (passing `Ix = Icoils[i]` so the
    plots are titled per current), and aggregates the outputs.

    # Arguments
    - `Icoils`: Vector of coil currents (A). Length must match `screen_up`.
    - `screen_up`: Dict-like container (e.g. `OrderedDict{Int, Matrix}`) whose
    keys include `1:length(Icoils)` and whose values are N×M matrices of hit
    positions in **meters** (columns: 1=x, 3=z).
    - `n_bins`: Histogram binning multiplier used by `analyze_screen_profile`.
    - `width_mm`: Gaussian kernel σ (mm) used by `smooth_profile`.
    - `add_plot`: If `true`, each call plots the profiles titled with `Icoils[i]`.
    - `λ_raw`, `λ_smooth`: Spline regularization parameters for raw and smoothed
    profiles, respectively.
    - `store_profiles`: If `true`, store the full `z_profile` array for each
    current (can be large). If `false`, omit it to save memory.

    # Returns
    An `OrderedDict{Int, OrderedDict{Symbol, Any}}` such that:
    - `out[i][:Icoil]`                       → `Icoils[i]`
    - `out[i][:z_max_raw_mm]`                → raw-profile maximum z [mm]
    - `out[i][:z_max_raw_spline_mm]`         → spline fit (raw) maximum z [mm]
    - `out[i][:z_max_smooth_mm]`             → smoothed-profile maximum z [mm]
    - `out[i][:z_max_smooth_spline_mm]`      → spline fit (smoothed) maximum z [mm]
    - `out[i][:z_profile]` (optional)        → Nz×3 matrix `[z, raw, smooth]`

    # Notes
    - Expects `screen_up[i]` to have at least 3 columns (x=col 1, z=col 3).
    - This function assumes `analyze_screen_profile(Ix, data_mm; ...)` accepts
    hit data in **mm** and will handle the histogramming/plotting.
    - If you prefer keys by current value instead of index, use a
    `Dict{Float64, ...}` and set `out[Icoils[i]] = inner`.
"""
function QM_analyze_profiles_to_dict(data::OrderedDict{Symbol,Any}, p::AtomParams;
    manifold::Symbol =:F_bottom, n_bins::Tuple = (1,4), width_mm::Float64 = 0.150, add_plot::Bool = false,
    λ_raw::Float64 = 0.01, λ_smooth::Float64 = 1e-3)

    @assert haskey(data, :Icoils) "missing :Icoils"
    @assert haskey(data, :data)   "missing :data"

    nx_bins = n_bins[1]
    nz_bins = n_bins[2]

    Ix = data[:Icoils]
    out = OrderedDict{Int, OrderedDict{Symbol, Any}}()
    for i in eachindex(Ix)
        if manifold===:F_top
            img_F = 1e3 .* vcat((xz[:, 7:8] for xz in data[:data][i][1:Int(2*p.Ispin+2)])...) # 7:8 is [x,z] in mm from 1:2I+2

        elseif manifold===:F_bottom
            img_F = 1e3 .* vcat((xz[:, 7:8] for xz in data[:data][i][Int(2*p.Ispin+3):Int(4*p.Ispin+2)])...) # 7:8 is [x,z] in mm from 2I+3:4I+2
        else
            img_F = data[:data][i][tryparse(Int, string(manifold))][:,7:8] # 7:8 is [x,z] in mm for the manifold: level
        end
    
        result = analyze_screen_profile(Ix[i], img_F;
                m_mom=:qm, manifold=manifold, 
                nx_bins=nx_bins, nz_bins=nz_bins, 
                width_mm=width_mm, 
                add_plot=add_plot, 
                λ_raw=λ_raw, λ_smooth=λ_smooth)
    
        inner = OrderedDict{Symbol, Any}(
        :Icoil                  => Ix[i],
        :z_max_raw_mm           => result.z_max_raw_mm,
        :z_max_raw_spline_mm    => result.z_max_raw_spline_mm,
        :z_max_smooth_mm        => result.z_max_smooth_mm,
        :z_max_smooth_spline_mm => result.z_max_smooth_spline_mm,
        :z_profile              => result.z_profile,
        )
        
        out[i] = inner 
    end
    return out
end