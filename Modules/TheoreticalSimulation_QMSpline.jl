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

# Example
```julia
pos, S = max_of_bspline_positions(z, y; λ0=0.005, order=4, n_peaks=2)
heights = S.(pos)

# See also
[`analyze_screen_profile`](@ref), [`QM_analyze_profiles_to_dict`](@ref), [`smooth_profile`](@ref)

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
  `x_bin_size = 1e3 * nx_bins * default_camera_pixel_size` and
  `z_bin_size = 1e3 * nz_bins * default_camera_pixel_size`,
  where `default_camera_pixel_size` is a **global** in meters.
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
- Requires a global `default_camera_pixel_size::Real` (meters).
- Uses `max_of_bspline_positions` to obtain sub-bin-accurate maxima.
- The `manifold` only affects labels/filenames, not the numerical results.

# Throws
- `AssertionError` if `data_mm` is not `N×2` (x,z in mm), if `nx_bins/nz_bins ≤ 0`,
  if `width_mm ≤ 0`, or if `default_camera_pixel_size` is not defined.

# Example
```julia
res = analyze_screen_profile(0.125, hits_mm;
    manifold=:F_top, nx_bins=1, nz_bins=4,
    width_mm=0.15, λ_raw=0.01, λ_smooth=1e-3, add_plot=true)

@show res.z_max_smooth_spline_mm

# See also
[`max_of_bspline_positions`](@ref), [`QM_analyze_profiles_to_dict`](@ref), [`smooth_profile`](@ref)

"""
function analyze_screen_profile(Ix, data_mm::AbstractMatrix; 
    manifold::Symbol = :F_top, nx_bins::Integer = 2, nz_bins::Integer = 2, 
    add_plot::Bool = false, plot_xrange::Symbol = :all,
    width_mm::Float64 = 0.150, λ_raw::Float64=0.01, λ_smooth::Float64 = 1e-3, 
    mode::Symbol=:probability)

    @assert size(data_mm,2) == 2 "data_mm must be N×2 (columns: x,z in mm)"
    @assert nx_bins > 0 "nx_bins must be > 0"
    @assert nz_bins > 0 "nz_bins must be > 0"
    @assert width_mm > 0 "width_mm must be positive"
    @assert @isdefined(default_camera_pixel_size) "define `default_camera_pixel_size` (meters)"

    # Fixed analysis limits
    xlim = (-8.0, 8.0)
    zlim = (-12.5, 12.5)
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
    h = normalize(fit(Histogram, (x, z), (edges_x, edges_z)); mode=mode)
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
    QM_analyze_profiles_to_dict(
        data::OrderedDict{Symbol,Any},
        p::AtomParams;
        manifold::Symbol = :F_bottom,
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

# Inputs
- `data::OrderedDict{Symbol,Any}`:
  Must contain:
  - `:Icoils :: AbstractVector{<:Real}` — coil currents (A), length `N`.
  - `:data` — a length-`N` container; each `data[:data][i]` is an indexable
    collection of matrices whose **columns 7:8** are `[x, z]` in **meters**.
- `p::AtomParams`:
  Used to determine level grouping via `p.Ispin` when selecting manifolds.

# Keyword Arguments
- `manifold::Symbol = :F_bottom`:
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
  `default_camera_pixel_size` (meters) is defined for bin sizing in that
  routine.
- Level selection uses `p.Ispin`; e.g., for `I = 3/2`, `:F_top` picks levels
  `1:5` and `:F_bottom` picks `6:8` (1-based).
- Throws an `AssertionError` if `data` is missing `:Icoils` or `:data`, or if
  `manifold` is a non-numeric symbol not equal to `:F_top`/`:F_bottom`.

# Example
```julia
out = QM_analyze_profiles_to_dict(run_data, atom_params;
    manifold=:F_top, n_bins=(1,4), width_mm=0.15,
    add_plot=true, plot_xrange=:right, λ_raw=0.01, λ_smooth=1e-3)

z_first = out[1][:z_max_smooth_spline_mm]

# See also
[`analyze_screen_profile`](@ref), [`max_of_bspline_positions`](@ref), [`smooth_profile`](@ref)

"""
function QM_analyze_profiles_to_dict(data::OrderedDict{Symbol,Any}, p::AtomParams;
    manifold::Symbol =:F_bottom, n_bins::Tuple = (1,4), width_mm::Float64 = 0.150, 
    add_plot::Bool = false, plot_xrange::Symbol = :all,
    λ_raw::Float64 = 0.01, λ_smooth::Float64 = 1e-3, mode::Symbol = :probability)

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
            lvl = tryparse(Int, string(manifold))
            @assert lvl !== nothing "Non-numeric manifold '$manifold' (expected e.g. :1, :2, ...)"
            img_F = 1e3 .* data[:data][i][lvl][:, 7:8]   # convert to mm to match other branches
        end
    
        result = analyze_screen_profile(Ix[i], img_F;
                manifold=manifold, 
                nx_bins=nx_bins, nz_bins=nz_bins, 
                width_mm=width_mm, 
                add_plot=add_plot, 
                plot_xrange=plot_xrange,
                λ_raw=λ_raw, λ_smooth=λ_smooth,mode=mode)
    
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