# Kelvin Titimbo
# California Institute of Technology
# March 2026

#  Plotting Setup
using Plots; gr()
Plots.default(
    show=true, dpi=800, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
FIG_EXT = "png"   # could be "pdf", "svg", etc.
SAVE_FIG = true
# Aesthetics and output formatting
using Colors, ColorSchemes
using LaTeXStrings, Printf, PrettyTables
# Time-stamping/logging
using Dates
const T_START = Dates.now() ; # Timestamp start for execution timing
# Numerical tools
using LinearAlgebra, DataStructures
using Interpolations, Roots, Loess, Optim
using BSplineKit
using Polynomials
using DSP
using LambertW, PolyLog
using StatsBase
using Random, Statistics, NaNStatistics, Distributions, StaticArrays
using Alert
# Data manipulation
using OrderedCollections
using DelimitedFiles, CSV, DataFrames, JLD2
# include("./Modules/MyPolylogarithms.jl");
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Set the working directory to the current location
cd(@__DIR__) ;
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
const OUTDIR    = joinpath(@__DIR__, "data_studies", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
const TEMP_DIR = joinpath(@__DIR__,"artifacts", "JuliaTemp")
isdir(TEMP_DIR) || mkpath(TEMP_DIR);
ENV["TMPDIR"] = TEMP_DIR
ENV["TEMP"]   = TEMP_DIR
ENV["TMP"]    = TEMP_DIR
@info "Temporary directory configured" TEMP_DIR
# General setup
HOSTNAME = gethostname();
@info "Running on host" HOSTNAME=HOSTNAME
# Random seeds
base_seed_set = 145;
rng_set = MersenneTwister(base_seed_set)
# rng_set = TaskLocalRNG();
# Custom modules
include("./Modules/atoms.jl");
include("./Modules/samplings.jl");
include("./Modules/JLD2_MyTools.jl");
include("./Modules/DataReading.jl");
include("./Modules/MyExperimentalAnalysis.jl");
include("./Modules/TheoreticalSimulation.jl");
using .TheoreticalSimulation;
TheoreticalSimulation.SAVE_FIG = SAVE_FIG;
TheoreticalSimulation.FIG_EXT  = FIG_EXT;
TheoreticalSimulation.OUTDIR   = OUTDIR;

println("\n\t\tRunning process on:\t $(RUN_STAMP) \n")

# Math constants
const TWOπ = 2π;
const INV_E = exp(-1);
atom        = "39K"  ;
## PHYSICAL CONSTANTS from NIST
# RSU : Relative Standard Uncertainty
const kb    = 1.380649e-23 ;       # Boltzmann constant (J/K)
const ħ     = 6.62607015e-34/2π ;  # Reduced Planck constant (J s)
const μ₀    = 1.25663706127e-6;    # Vacuum permeability (Tm/A)
const μB    = 9.2740100657e-24 ;   # Bohr magneton (J/T)
const γₑ    = -1.76085962784e11 ;  # Electron gyromagnetic ratio  (1/sT). Relative Standard Uncertainty = 3.0e-10
const μₑ    = 9.2847646917e-24 ;   # Electron magnetic moment (J/T). RSU = 3.0e-10
const Sspin = 1/2 ;                # Electron spin
const gₑ    = -2.00231930436092 ;  # Electron g-factor
## ATOM INFORMATION: 
# atom_info       = AtomicSpecies.atoms(atom);
K39_params = AtomParams(atom); # [R μn γn Ispin Ahfs M ] 
# Math constants
const TWOπ = 2π;
const INV_E = exp(-1);
quantum_numbers = fmf_levels(K39_params);

# STERN--GERLACH EXPERIMENT
# Camera and pixel geometry : intrinsic properties
cam_pixelsize = 6.5e-6 ;  # Physical pixel size of camera [m]
nx_pixels , nz_pixels= (2160, 2560); # (Nx,Nz) pixels
# Simulation resolution
sim_bin_x, sim_bin_z = (1,1) ;  # Camera binning
sim_pixelsize_x, sim_pixelsize_z = (sim_bin_x, sim_bin_z).*cam_pixelsize ; # Effective pixel size after binning [m]
# Image dimensions (adjusted for binning)
x_pixels = Int(nx_pixels / sim_bin_x);  # Number of x-pixels after binning
z_pixels = Int(nz_pixels / sim_bin_z);  # Number of z-pixels after binning
# Spatial axes shifted to center the pixels
x_position = pixel_coordinates(x_pixels, sim_bin_x, sim_pixelsize_x);
z_position = pixel_coordinates(z_pixels, sim_bin_z, sim_pixelsize_z);
println("""
***************************************************
CAMERA FEATURES
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm

SIMULATION INFORMATION
    Binning                 : $(sim_bin_x) × $(sim_bin_z)
    Effective pixels        : $(x_pixels) × $(z_pixels)
    Pixel size              : $(1e6*sim_pixelsize_x)μm × $(1e6*sim_pixelsize_z)μm
    xlims                   : ($(round(minimum(1e6*x_position), digits=6)) μm, $(round(maximum(1e3*x_position), digits=4)) mm)
    zlims                   : ($(round(minimum(1e6*z_position), digits=6)) μm, $(round(maximum(1e3*z_position), digits=4)) mm)
***************************************************
""")
# Furnace
T_K = 273.15 + 200 ; # Furnace temperature (K)
# Furnace aperture
const x_furnace = 2.0e-3 ;
const z_furnace = 100e-6 ;
# Slit : Pre SG
const x_slit  = 4.0e-3 ;
const z_slit  = 300e-6 ;
# Circular Aperture : Post SG
const R_aper            = 5.8e-3/2 ;
const y_SGToAperture    = 42.0e-3 ;   
# Propagation distances
const y_FurnaceToSlit = 224.0e-3 ;
const y_SlitToSG      = 44.0e-3 ;
const y_SG            = 7.0e-2 ;
const y_SGToScreen    = 32.0e-2 ;
# Connecting pipes
const R_tube = 35e-3/2 ; # Radius of the connecting pipe (m)
effusion_params = BeamEffusionParams(x_furnace,z_furnace,x_slit,z_slit,y_FurnaceToSlit,T_K,K39_params);
println("""
***************************************************
SETUP FEATURES
    Temperature             : $(T_K)
    Furnace aperture (x,z)  : ($(1e3*x_furnace)mm , $(1e6*z_furnace)μm)
    Slit (x,z)              : ($(1e3*x_slit)mm , $(1e6*z_slit)μm)
    Post-SG aperture radius : $(1e3*R_aper)mm
    Furnace → Slit          : $(1e3*y_FurnaceToSlit)mm
    Slit → SG magnet        : $(1e3*y_SlitToSG)mm
    SG magnet               : $(1e3*y_SG)mm
    SG magnet → Screen      : $(1e3*y_SGToScreen)mm
    SG magnet → Aperture    : $(1e3*y_SGToAperture)mm
    Tube radius             : $(1e3*R_tube)mm
***************************************************
""")
# Setting the variables for the module
TheoreticalSimulation.default_camera_pixel_size = cam_pixelsize;
TheoreticalSimulation.default_x_pixels          = nx_pixels;
TheoreticalSimulation.default_z_pixels          = nz_pixels;
TheoreticalSimulation.default_x_furnace         = x_furnace;
TheoreticalSimulation.default_z_furnace         = z_furnace;
TheoreticalSimulation.default_x_slit            = x_slit;
TheoreticalSimulation.default_z_slit            = z_slit;
TheoreticalSimulation.default_y_FurnaceToSlit   = y_FurnaceToSlit;
TheoreticalSimulation.default_y_SlitToSG        = y_SlitToSG;
TheoreticalSimulation.default_y_SG              = y_SG;
TheoreticalSimulation.default_y_SGToScreen      = y_SGToScreen;
TheoreticalSimulation.default_R_tube            = R_tube;
TheoreticalSimulation.default_c_aperture        = R_aper;
TheoreticalSimulation.default_y_SGToAperture    = y_SGToAperture;
##################################################################################################
function standard_error(x)
    return std(x; corrected=true) ./ sqrt.(length(x))
end

logspace10(lo, hi; n=50) = 10.0 .^ range(log10(lo), log10(hi); length=n)

function log_mask(x, y)
    (x .> 0) .& (y .> 0) .& isfinite.(x) .& isfinite.(y)
end

# helper: first index where column > threshold (skips missings; falls back to 1)
@inline function first_gt_idx(df::DataFrame, col::Symbol, thr::Real)
    v = df[!, col]
    idx = findfirst(x -> !ismissing(x) && x >= thr, v)
    return idx === nothing ? 1 : idx
end

"""
    combine_on_grid_mc_weighted(xsets, ysets;
                                σxsets=nothing,
                                σysets=nothing,
                                xq=:union,
                                B::Int=400,
                                outside::Symbol=:mask,
                                rel_x::Bool=false,
                                min_datasets::Int=1,
                                rng=Random.default_rng())

Weighted Monte-Carlo combination of multiple noisy 1D datasets onto a common grid,
propagating uncertainties in both x and y.

For each Monte-Carlo replicate, each dataset is perturbed according to its supplied
uncertainties, linearly interpolated onto a shared query grid, and then combined
pointwise across datasets using inverse-variance weights.

This is a weighted version of the user's earlier `average_on_grid_mc(...)`, which
performed an unweighted average after interpolation. Here, datasets with smaller
uncertainty contribute more strongly to the final combined curve. :contentReference[oaicite:1]{index=1}

# Arguments
- `xsets`, `ysets`:
    Collections of x- and y-vectors, one per dataset.
    Must satisfy `length(xsets) == length(ysets)` and each pair must have matching lengths.

# Keyword arguments
- `σxsets=nothing`:
    Per-dataset x uncertainties. If `nothing`, x is not perturbed.
- `σysets=nothing`:
    Per-dataset y uncertainties. If `nothing`, all datasets are combined with equal weights.
    If provided, these are also propagated onto the query grid and used for weighting.
- `rel_x=false`:
    If `true`, interpret `σxsets[i]` as relative uncertainties, i.e. `Δx = σx .* x`.
    Otherwise treat them as absolute.
- `xq=:union`:
    Query grid specification.
    - `:union`  → sorted union of all x-values
    - vector    → explicit query grid
- `B=400`:
    Number of Monte-Carlo replicates.
- `outside=:mask`:
    Extrapolation policy:
    - `:mask`   → outside each dataset's x-range return `NaN` and weight 0
    - `:linear` → linear extrapolation
    - `:flat`   → constant extrapolation
- `min_datasets=1`:
    Minimum number of contributing datasets required at a grid point in a replicate.
    If fewer contribute, that replicate gives `NaN` there.
- `rng`:
    Random number generator.

# Returns
A named tuple with fields:
- `xq`      : common query grid
- `μ`       : final Monte-Carlo mean combined curve
- `σ_mc`    : replicate-to-replicate Monte-Carlo standard deviation of the combined curve
- `σ_w`     : average weighted standard error from the inverse-variance combine step
- `σ_tot`   : total uncertainty, `sqrt(σ_mc^2 + σ_w^2)`
- `n_eff`   : average number of datasets contributing at each query point
- `preds`   : matrix of combined predictions, size `(B, length(xq))`

# Notes
- Linear interpolation is used for both `y` and `σy`.
- If `σysets` is provided, the local variance used for weighting is obtained by
  linearly interpolating `σy^2` onto the query grid.
- Repeated x-values after x-jittering are merged before interpolation.
- `σ_tot` is often the most useful final uncertainty to plot/use.

# Interpretation of uncertainties
At each grid point, the final uncertainty is split into two pieces:

1. `σ_mc`:
   variation of the final combined curve across Monte-Carlo replicates
2. `σ_w`:
   the typical inverse-variance combination uncertainty within each replicate

These are combined in quadrature as

    σ_tot = sqrt(σ_mc^2 + σ_w^2)

This is usually more informative than only reporting the Monte-Carlo spread.
"""
function combine_on_grid_mc_weighted(xsets, ysets;
                                     σxsets=nothing,
                                     σysets=nothing,
                                     xq=:union,
                                     B::Int=400,
                                     outside::Symbol=:mask,
                                     rel_x::Bool=false,
                                     min_datasets::Int=1,
                                     rng=Random.default_rng())

    @assert length(xsets) == length(ysets) "xsets and ysets must have the same number of datasets"
    nset = length(xsets)
    @assert nset > 0 "At least one dataset is required"
    @assert B > 0 "B must be positive"
    @assert outside in (:mask, :linear, :flat) "outside must be :mask, :linear, or :flat"
    @assert min_datasets ≥ 1 "min_datasets must be at least 1"


    function get_σx_for_dataset(σxsets, x, i)
        if σxsets === nothing
            return nothing
        elseif σxsets isa Real
            # same scalar for every point in every dataset
            return fill(Float64(σxsets), length(x))
        else
            # original behavior: one entry per dataset
            return collect(σxsets[i])
        end
    end

    if σxsets !== nothing && !(σxsets isa Real)
        @assert length(σxsets) == nset "σxsets must be either: nothing , a scalar, or one entry per dataset"
    end
    if σysets !== nothing
        @assert length(σysets) == nset "σysets must have one entry per dataset"
    end

    # ----------------------------
    # Build common query grid
    # ----------------------------
    xq_vec = xq === :union ? sort!(unique(vcat(map(collect, xsets)...))) : collect(xq)
    m = length(xq_vec)
    @assert m > 0 "Query grid is empty"

    # Combined prediction for each MC replicate
    preds   = fill(NaN, B, m)

    # Weighted standard error inside each replicate:
    #   σ_w(b, j) = sqrt(1 / sum_i w_i(j))
    σw_repl = fill(NaN, B, m)

    # Number of datasets contributing per replicate / grid point
    neff_repl = zeros(Int, B, m)

    # ----------------------------
    # Helper: merge repeated x after jittering
    # ----------------------------
    function merge_duplicate_x(x::AbstractVector, y::AbstractVector, σy::Union{Nothing,AbstractVector})
        p = sortperm(x)
        xs = collect(x[p])
        ys = collect(y[p])
        σs = σy === nothing ? nothing : collect(σy[p])

        xout = Float64[]
        yout = Float64[]
        σout = σs === nothing ? nothing : Float64[]

        i = 1
        n = length(xs)
        while i ≤ n
            j = i
            xi = xs[i]
            while j < n && isapprox(xs[j+1], xi; atol=0.0, rtol=0.0)
                j += 1
            end

            # block i:j has same x
            push!(xout, xi)

            if σs === nothing
                push!(yout, mean(@view ys[i:j]))
            else
                # inverse-variance merge at identical x
                v = (@view σs[i:j]).^2
                w = 1.0 ./ v
                ȳ = sum((@view ys[i:j]) .* w) / sum(w)
                σ̄ = sqrt(1.0 / sum(w))
                push!(yout, ȳ)
                push!(σout, σ̄)
            end

            i = j + 1
        end

        return xout, yout, σout
    end

    # ----------------------------
    # Helper: build interpolation / extrapolation
    # ----------------------------
    function build_interp(xb::AbstractVector, yb::AbstractVector)
        itp = Interpolations.interpolate((xb,), yb, Gridded(Interpolations.Linear()))
        if outside === :linear
            return Interpolations.extrapolate(itp, Line())
        elseif outside === :flat
            return Interpolations.extrapolate(itp, Flat())
        else
            return itp
        end
    end

    # ----------------------------
    # Helper: evaluate y and local variance on xq
    # ----------------------------
    function eval_dataset_on_grid(xb, yb, σyb, xqv)
        xlo, xhi = first(xb), last(xb)

        yvals = fill(NaN, length(xqv))
        vvals = fill(NaN, length(xqv))

        yitp = build_interp(xb, yb)

        vitp = nothing
        if σyb !== nothing
            # interpolate variance, not σ itself
            vb = σyb.^2
            vitp = build_interp(xb, vb)
        end

        if outside === :mask
            mask = (xqv .>= xlo) .& (xqv .<= xhi)
            yvals[mask] .= yitp.(xqv[mask])
            if vitp !== nothing
                vvals[mask] .= vitp.(xqv[mask])
            else
                vvals[mask] .= 1.0
            end
        else
            yvals .= yitp.(xqv)
            if vitp !== nothing
                vvals .= vitp.(xqv)
            else
                vvals .= 1.0
            end
        end

        # Numerical safety
        @inbounds for j in eachindex(vvals)
            if !isnan(vvals[j]) && vvals[j] ≤ 0
                vvals[j] = NaN
            end
        end

        return yvals, vvals
    end

    # ----------------------------
    # Monte-Carlo loop
    # ----------------------------
    for b in 1:B
        curves = Vector{Vector{Float64}}(undef, nset)
        vars   = Vector{Vector{Float64}}(undef, nset)

        for i in 1:nset
            x = collect(xsets[i])
            y = collect(ysets[i])
            @assert length(x) == length(y) "Dataset $i has mismatched x/y lengths"

            σx = get_σx_for_dataset(σxsets, x, i)
            σy = σysets === nothing ? nothing : collect(σysets[i])

            if σx !== nothing
                @assert length(σx) == length(x) "Dataset $i has mismatched x/σx lengths"
            end
            if σy !== nothing
                @assert length(σy) == length(y) "Dataset $i has mismatched y/σy lengths"
            end

            # Jitter x
            xb = if σx === nothing
                copy(x)
            else
                dx = rel_x ? σx .* x : σx
                x .+ randn(rng, length(x)) .* dx
            end

            # Jitter y
            yb = if σy === nothing
                copy(y)
            else
                y .+ randn(rng, length(y)) .* σy
            end

            # Need at least 2 points for interpolation
            if length(xb) < 2
                curves[i] = fill(NaN, m)
                vars[i]   = fill(NaN, m)
                continue
            end

            # Merge exact duplicates after jitter/sort
            xb2, yb2, σy2 = merge_duplicate_x(xb, yb, σy)

            if length(xb2) < 2
                curves[i] = fill(NaN, m)
                vars[i]   = fill(NaN, m)
                continue
            end

            curves[i], vars[i] = eval_dataset_on_grid(xb2, yb2, σy2, xq_vec)
        end

        # Weighted combine across datasets at each xq
        for j in 1:m
            num = 0.0
            den = 0.0
            ncontrib = 0

            @inbounds for i in 1:nset
                yij = curves[i][j]
                vij = vars[i][j]

                if !isnan(yij) && !isnan(vij) && vij > 0
                    wij = 1.0 / vij
                    num += wij * yij
                    den += wij
                    ncontrib += 1
                end
            end

            neff_repl[b, j] = ncontrib

            if ncontrib ≥ min_datasets && den > 0
                preds[b, j]   = num / den
                σw_repl[b, j] = sqrt(1.0 / den)
            else
                preds[b, j]   = NaN
                σw_repl[b, j] = NaN
            end
        end
    end

    # ----------------------------
    # Final summary across MC replicates
    # ----------------------------
    μ      = fill(NaN, m)
    σ_mc   = fill(NaN, m)
    σ_w    = fill(NaN, m)
    σ_tot  = fill(NaN, m)
    n_eff  = fill(NaN, m)

    for j in 1:m
        vals_pred = [v for v in @view(preds[:, j]) if !isnan(v)]
        vals_σw   = [v for v in @view(σw_repl[:, j]) if !isnan(v)]
        vals_neff = [v for v in @view(neff_repl[:, j]) if v > 0]

        if !isempty(vals_pred)
            μ[j] = mean(vals_pred)
            σ_mc[j] = length(vals_pred) > 1 ? std(vals_pred; corrected=true) : 0.0
        end

        if !isempty(vals_σw)
            # Typical within-replicate weighted SE
            σ_w[j] = mean(vals_σw)
        end

        if !isnan(σ_mc[j]) && !isnan(σ_w[j])
            σ_tot[j] = hypot(σ_mc[j], σ_w[j])
        elseif !isnan(σ_mc[j])
            σ_tot[j] = σ_mc[j]
        elseif !isnan(σ_w[j])
            σ_tot[j] = σ_w[j]
        end

        if !isempty(vals_neff)
            n_eff[j] = mean(vals_neff)
        end
    end

    return (
        xq    = xq_vec,
        μ     = μ,
        σ_mc  = σ_mc,
        σ_w   = σ_w,
        σ_tot = σ_tot,
        n_eff = n_eff,
        preds = preds,
    )
end

function fit_ki(data_org, selected_points, ki_list, ki_range)
    """
        fit_ki(data_org, selected_points, ki_list, ki_range)

    Fit the induction coefficient `kᵢ` by minimizing a mean-squared error in **log10 space**
    between the interpolated prediction `ki_up_itp(x, kᵢ)` and a selected subset of data points.
    The fit is therefore sensitive to *relative (fractional) deviations* across orders of
    magnitude.

    Although the optimization is performed in log space, the reported error is evaluated
    **in linear space** at the best-fit value of `kᵢ`, and is returned as a root-mean-square
    error (RMSE) in the same physical units as the dependent variable (e.g. millimeters).

    # Arguments
    - `data_org` :: 2-column array `(x, y)`  
    Full data set used to compute the coefficient of determination R² in linear space.

    - `selected_points` :: 2-column array `(x, y)`  
    Subset of points used for the fit. All `y` values must be strictly positive
    (required for log10 evaluation).

    - `ki_list` :: AbstractVector  
    Vector of candidate `kᵢ` values defining the search interval.

    - `ki_range` :: Tuple{Int,Int}  
    Index range `(ki_start, ki_stop)` selecting the portion of `ki_list` used in
    the bounded 1D optimization.

    # Returns
    NamedTuple with fields:
    - `ki`        : Best-fit value of the induction coefficient `kᵢ`
    - `ki_err`    : Root-mean-square error (RMSE) in **linear space**, evaluated on
                    `selected_points` at the fitted `kᵢ`
    - `r2_coeff`  : Coefficient of determination R² computed in linear space on `data_org`

    # Notes
    - The fit minimizes an error in log10 space, but no uncertainty on `kᵢ` is estimated.
    The returned `ki_err` is **not** an error bar on `kᵢ`, but a goodness-of-fit measure
    in real space.
    - If the dependent variable spans several orders of magnitude, the log-space fit
    prevents large-amplitude points from dominating the optimization, while the linear
    RMSE provides a physically interpretable error metric.
    """
    ki_start, ki_stop = ki_range

    Ic_fit = selected_points[:, 1]
    z_fit  = selected_points[:, 3]

    # # --- log-space loss (used ONLY for fitting) ---
    loss_log(ki) = begin
        z_pred = ki_up_itp.(Ic_fit, Ref(ki))
        mean(abs2, log10.(z_pred) .- log10.(z_fit))
    end

    # 1D optimization over ki
    fit_param = optimize(loss_log,
                         ki_list[ki_start], ki_list[ki_stop],
                         Brent())

    k_fit = Optim.minimizer(fit_param)

    # --- linear-space error (reported) ---
    z_pred_sel = ki_up_itp.(Ic_fit, Ref(k_fit))
    z_obs_sel  = z_fit

    mse_lin  = mean(abs2, z_pred_sel .- z_obs_sel)
    rmse_lin = sqrt(mse_lin)   # same units as z (e.g. mm)

    # predictions for the full data set
    Ic = data_org[:, 1]
    pred = ki_up_itp.(Ic, Ref(k_fit))
    y    = data_org[:, 3]
    coef_r2 = 1 - sum(abs2, pred .- y) / sum(abs2, y .- mean(y))

    return (ki = k_fit, ki_err = rmse_lin, r2_coeff = coef_r2)
end

function fit_ki_with_error(itp, data;
    bounds::Tuple{<:Real,<:Real},
    conf::Real = 0.95,
    use_Zse::Bool = false,
    profile::Bool = true,
    profile_grid::Int = 400)
    I  = collect(Float64, data[:, 1])
    Z  = collect(Float64, data[:, 3])
    σZ = collect(Float64, data[:, 4])   # only used if use_Zse=true

    # Valid mask (log requires Z>0; σZ>0 if used)
    m0 = isfinite.(I) .& isfinite.(Z) .& (Z .> 0)
    if use_Zse
        m0 .&= isfinite.(σZ) .& (σZ .> 0)
    end
    I, Z, σZ = I[m0], Z[m0], σZ[m0]

    # weights in log10 space: Var(log10 Z) ≈ (σZ/(Z ln10))^2
    w = use_Zse ? ((Z .* log(10.0)) ./ σZ) .^ 2 : ones(length(I))

    ki_min, ki_max = float(bounds[1]), float(bounds[2])

    # RSS objective in log10 space
    function loss(ki)
        zpred = itp.(I, Ref(ki))
        m = isfinite.(zpred) .& (zpred .> 0)
        any(m) || return Inf
        r  = log10.(zpred[m]) .- log10.(Z[m])
        ww = w[m]
        sum(ww .* (r .^ 2))
    end

    res = optimize(loss, ki_min, ki_max, Brent())
    k̂  = Optim.minimizer(res)

    # residuals at optimum (define RSS0 consistently)
    ẑ = itp.(I, Ref(k̂))
    m = isfinite.(ẑ) .& (ẑ .> 0)
    Zu, wu = Z[m], w[m]
    r0 = log10.(ẑ[m]) .- log10.(Zu)

    p = 1
    n = length(r0)
    @assert n > p "Not enough valid points to estimate uncertainty"

    RSS0 = sum(wu .* (r0 .^ 2))
    dof  = n - p
    σ²   = RSS0 / dof

    # finite-difference step (always computed)
    fd_step(k, lo, hi; rel=cbrt(eps(Float64)), absmin=1e-12) = begin
        hh = max(absmin, rel * max(abs(k), 1.0))
        room = min(k - lo, hi - k)
        room > 0 ? min(hh, 0.5 * room) : absmin
    end
    h₀ = fd_step(k̂, ki_min, ki_max)

    # derivative dr/dk (central diff, common-valid points)
    z⁺ = itp.(I[m], Ref(k̂ + h₀))
    z⁻ = itp.(I[m], Ref(k̂ - h₀))
    mJ = isfinite.(z⁺) .& isfinite.(z⁻) .& (z⁺ .> 0) .& (z⁻ .> 0)
    @assert count(mJ) > p "Not enough valid points after derivative filtering"

    Zu2 = Zu[mJ]
    w2  = wu[mJ]
    rJ  = r0[mJ]

    r⁺ = log10.(z⁺[mJ]) .- log10.(Zu2)
    r⁻ = log10.(z⁻[mJ]) .- log10.(Zu2)
    drdk = (r⁺ .- r⁻) ./ (2h₀)

    # SE from linearization: Var(k̂) ≈ σ² / (J'WJ)
    SJJ = sum(w2 .* (drdk .^ 2))
    se  = sqrt(σ² / SJJ)

    tcrit = quantile(TDist(dof), 0.5 + conf/2)
    k_err = tcrit * se
    ci_t  = (k̂ - k_err, k̂ + k_err)

    # R² in log10 space (weighted)
    y  = log10.(Zu2)
    ŷ  = log10.(ẑ[m][mJ])
    ȳw = sum(w2 .* y) / sum(w2)
    TSS = sum(w2 .* (y .- ȳw).^2)
    R2  = TSS > 0 ? 1 - sum(w2 .* (y .- ŷ).^2) / TSS : NaN

    # Profile interval: ΔRSS = χ²(1,conf) if weighted; else scale by σ²
    ci_profile = nothing
    Δtarget = profile ? quantile(Chisq(1), conf) : nothing
    Δrss = nothing
    profile_note = nothing

    if profile
        if use_Zse
            Δrss = Δtarget
        else
            profile_note = :profile_interval_scaled_for_unweighted
            Δrss = σ² * Δtarget
        end
        target = RSS0 + Δrss

        function bracket_side(dir::Int)
            grid = range(k̂, dir > 0 ? ki_max : ki_min; length=profile_grid)
            prevk = first(grid)
            prevL = loss(prevk)
            for k in Iterators.drop(grid, 1)
                L = loss(k)
                if isfinite(L) && (L > target) && isfinite(prevL) && (prevL <= target)
                    return (prevk, k)
                end
                prevk, prevL = k, L
            end
            return nothing
        end

        function bisect_cross(a, b; maxiter=80, tol=1e-10)
            lo, hi = a, b
            for _ in 1:maxiter
                mid = (lo + hi)/2
                fmid = loss(mid) - target
                if !isfinite(fmid)
                    hi = mid
                    continue
                end
                if fmid > 0
                    hi = mid
                else
                    lo = mid
                end
                if abs(hi - lo) <= tol*max(1.0, abs(mid))
                    return (lo + hi)/2
                end
            end
            return (lo + hi)/2
        end

        left_br  = bracket_side(-1)
        right_br = bracket_side(+1)
        k_lo = left_br  === nothing ? ki_min : bisect_cross(left_br[1], left_br[2])
        k_hi = right_br === nothing ? ki_max : bisect_cross(right_br[1], right_br[2])
        ci_profile = (k_lo, k_hi)
    end

    return (
        ki = k̂,
        ki_err = k_err,
        se = se,
        ci_t = ci_t,
        ci_profile = ci_profile,
        delta_target = Δtarget,
        delta_rss = Δrss,
        profile_note = profile_note,
        rss = RSS0,
        sigma2 = σ²,
        dof = dof,
        n_used = length(rJ),
        r2_coeff = R2,
        converged = Optim.converged(res),
        result = res
    )
end

function sigdigits_str(x, n=3)
    x == 0 && return "0"

    xr = round(x; sigdigits=n)

    # number of decimal places needed
    d = max(0, n - 1 - floor(Int, log10(abs(xr))))

    s = @sprintf("%.*f", d, xr)

    # remove trailing zeros and possible trailing dot
    s = replace(s, r"(\.\d*?)0+$" => s"\1")
    s = replace(s, r"\.$" => "")

    return s
end

function sci_label(x; n=3)
    x == 0 && return L"$0.0$"

    exp = floor(Int, log10(abs(x)))
    mant = x / 10.0^exp

    # round mantissa to (n sigdigits → typically 1 decimal for n=3)
    mant_r = round(mant; sigdigits=n)

    # force exactly ONE decimal place
    mant_str = @sprintf("%.1f", mant_r)

    return L"$%$(mant_str) \times 10^{%$exp}$"
end

##################################################################################################
# Coil currents
Icoils = [0.00,
            0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
            0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,
            0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
];
nI = length(Icoils);
colores_current = palette(:darktest,nI);

data_qmf1_path = joinpath(@__DIR__,"simulation_data",
    "QM_T200_8M",
    "qm_screen_profiles_f1_table.jld2");
JLD2_MyTools.summarize_meta_qm_jld2(data_qmf1_path)
data_qmf2_path = joinpath(@__DIR__,"simulation_data",
    "QM_T200_8M",
    "qm_screen_profiles_f2_table.jld2");
JLD2_MyTools.summarize_meta_qm_jld2(data_qmf2_path)
data_cqdup_path = joinpath(@__DIR__, "simulation_data",
    "CQD_T200_8M",
    "cqd_8M_up_profiles.jld2");
JLD2_MyTools.summarize_meta_cqd_jld2(data_cqdup_path)
data_cqddw_path = joinpath(@__DIR__, "simulation_data",
    "CQD_T200_8M",
    "cqd_8M_dw_profiles.jld2");
JLD2_MyTools.summarize_meta_cqd_jld2(data_cqddw_path)


data_directories = [
                    "20260220", 
                    "20260225", 
                    "20260226am",
                    "20260226pm",
                    "20260227", 
                    "20260303", 
                    "20260306r1", 
                    "20260306r2",
]
n_data = length(data_directories);
colores_data = palette(:darkrainbow, n_data);

nz , λ0 = 2, 0.01;
σw_mm = 0.175;

## ################################################################################################
# MAGNETIC FIELD
############ phywe magnetic field measurement ##############################
# PhyWe Calibration
BvsI_phywe = sort(CSV.read("SG_BvsI_phywe.csv",DataFrame; header=["Ic","Bz"]),1);
phywe_shift = hcat(BvsI_phywe.Ic .- 0.11021, BvsI_phywe.Bz .- 0.00445);
phywe_shift = DataReading.subset_by_cols(phywe_shift,[1,2]; thr=0.0, include_equal=false)[3];

# Experimental measurements while acquiring SG1 data
exp_data = Vector{Matrix{Float64}}(undef, n_data)
for (idx, dir) in enumerate(data_directories)
    data = load(joinpath(@__DIR__,"EXPERIMENTS",dir,"data_processed.jld2"),"data")
    Ic = data[:Currents]
    Bz = data[:BzTesla]
    exp_data[idx] = hcat(Ic,Bz)
end

BvsI_optimal = CSV.read("SG_BzinGauss_Iscan.csv", DataFrame; header=["Ic","Bz"]);

# magnetic field: log-log scale
fig1 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    size=(900,500),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    left_margin=5mm,
    bottom_margin=3mm,);
# for (idx, dir) in enumerate(data_directories)
#     data = exp_data[idx]
#     Ic = data[:,1]
#     Bz = data[:,2]

#     plot!(fig1, Ic, 1000*Bz,
#         label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
#         marker=(:circle, 2, :white),
#         markerstrokecolor=colores[idx],
#         line=(:dot, colores[idx], 1)
#     )

# end
plot!(fig1,Icoils[2:end], 1e3*TheoreticalSimulation.BvsI.(Icoils)[2:end],
    label="SG manual",
    line=(:solid,2,:black));
# plot!(fig1,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
#     label="SG PHYWE calibration",
#     line=(:solid,2,:dodgerblue4));
# plot!(fig1,phywe_shift[:,1], 1000*phywe_shift[:,2],
#     label="SG PHYWE calibration (adjusted)",
#     line=(:dot,2,:dodgerblue4))
plot!(fig1,1e-3*BvsI_optimal.Ic[4:end], 1e3*1e-4*BvsI_optimal.Bz[4:end],
    line=(:dashdot,2,:purple),
    marker=(:diamond, 3, :white),
    markerstrokecolor=:purple,
    label="New setup arrangement",
);
plot!(fig1,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.02),
    ylims=(1e-2,1e3),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10, 100, 1000], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}", L"10^{3}"]),
    legendfontsize = 8,
);
display(fig1)

# magnetic field: linear scale
fig2 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    size=(900,500),
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    left_margin=5mm,
    bottom_margin=3mm,
    legendfontsize = 8,);
# for (idx, dir) in enumerate(data_directories)
#     data = exp_data[idx]
#     Ic = data[:,1]
#     Bz = data[:,2]

#     plot!(fig2, Ic, 1000*Bz,
#         label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
#         marker=(:circle, 2, :white),
#         markerstrokecolor=colores[idx],
#         line=(:dot, colores[idx], 1)
#     )

# end
plot!(fig2,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
# plot!(fig2,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
#     label="SG PHYWE calibration",
#     line=(:solid,2,:dodgerblue4));
# plot!(fig2,phywe_shift[:,1], 1000*phywe_shift[:,2],
#     label="SG PHYWE calibration (adjusted)",
#     line=(:dot,2,:dodgerblue4));
plot!(fig2,1e-3*BvsI_optimal.Ic, 1e3*1e-4*BvsI_optimal.Bz,
    line=(:dashdot,2,:purple),
    marker=(:diamond, 3, :white),
    markerstrokecolor=:purple,
    label="New setup arrangement",
);
display(fig2)

# magnetic field: low currents
fig3 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    size=(900,500),);
for (idx, dir) in enumerate(data_directories)
    data = exp_data[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig3, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig3,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig3,1e-3*BvsI_optimal.Ic, 1e3*1e-4*BvsI_optimal.Bz,
    line=(:dashdot,2,:purple),
    marker=(:diamond, 3, :white),
    markerstrokecolor=:purple,
    label="New setup arrangement",
);
plot!(fig3,
    xlims=(0.0,15e-3),
    ylims=(-5,15),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig3)

# magnetic field: high currents
fig4 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    size=(900,500),);
for (idx, dir) in enumerate(data_directories)
    data = exp_data[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig4, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig4,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black))
plot!(fig4,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
    label="SG PHYWE calibration",
    line=(:solid,2,:dodgerblue4));
plot!(fig4,phywe_shift[:,1], 1000*phywe_shift[:,2],
    label="SG PHYWE calibration (adjusted)",
    line=(:dot,2,:dodgerblue4));
plot!(fig4,1e-3*BvsI_optimal.Ic, 1e3*1e-4*BvsI_optimal.Bz,
    line=(:dashdot,2,:purple),
    marker=(:diamond, 3, :white),
    markerstrokecolor=:purple,
    label="New setup arrangement",
);
plot!(fig4,
    xlims=(0.600,1.1),
    ylims=(300,900),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    bottom_margin=2mm,
    left_margin=3mm,
);
display(fig4)

# magnetic field measured vs magnetic field manual
BvsI_comparison = Vector{Matrix{Float64}}(undef, n_data);
for (idx, dir) in enumerate(data_directories)
    vs = exp_data[idx]

    mask, inds, rows_view = DataReading.subset_by_cols(vs, [1,2]; thr=1e-3, include_equal = true)
    BvsI_comparison[idx] = Matrix(rows_view)  # store a copy; or keep the view (see B)
end
fig5 = plot(xlabel="Currents (A)",
    ylabel=L"$B_{\mathrm{exp}} / B_{\mathrm{manual}}$");
for (idx, dir) in enumerate(data_directories)
    B_ratio = BvsI_comparison[idx][:,2] ./ TheoreticalSimulation.BvsI.(BvsI_comparison[idx][:,1])
    
    plot!(fig5,  BvsI_comparison[idx][:,1] , B_ratio,
        label=data_directories[idx],
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:solid,1,colores_data[idx])
    )
    y0 , σ0 = mean(B_ratio), standard_error(B_ratio)

    x = range(1e-3, 1.1, length=200)
    plot!(fig5, x, fill(y0, length(x)),
     ribbon = σ0,
     color = colores_data[idx],
     fillalpha = 0.25,
     line=(:dash,0.5,colores_data[idx]),
     label = "$(round(y0; digits=3)) ± $(round(σ0; sigdigits=1))")
    # hline!([y0], ine=(:dot,0.5,colores[idx]), label= "")
end
plot!(fig5,
    legend=:bottomright,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.05),
    ylims=(3e-1,2),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig5)

fig5a = plot(xlabel="Currents (A)",
    ylabel=L"$B_{\mathrm{exp}} / B_{\mathrm{manual}}$");
plot!(1e-3*BvsI_optimal.Ic[4:end] , (1e-4*(BvsI_optimal.Bz) ./ TheoreticalSimulation.BvsI.(1e-3*BvsI_optimal.Ic))[4:end],
    label=false,
    marker=(:square,2,:white),
    markerstrokecolor=:red,
    line=(:solid,1,:red));
plot!(fig5a,
    legend=:bottomright,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.05),
    ylims=(1e-1,2),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig5a)

# SHIFTED EXPERIMENTAL DATA TO ZERO FIELD
exp_data_corr = Vector{Matrix{Float64}}(undef, n_data);
for (idx, dir) in enumerate(data_directories)
    exp_data_corr[idx] = hcat(exp_data[idx][:,1] , exp_data[idx][:,2] .- exp_data[idx][1,2] )
end

fig1A = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    title="Shifted B field");
for (idx, dir) in enumerate(data_directories)
    data = exp_data_corr[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig1A, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig1A,Icoils[2:end], 1e3*TheoreticalSimulation.BvsI.(Icoils)[2:end],
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig1A,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.02),
    ylims=(1e-2,1e3),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10, 100], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig1A)

fig2A = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    title="Shifted B field");
for (idx, dir) in enumerate(data_directories)
    data = exp_data_corr[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig2A, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig2A,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig2A,
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig2A)

fig3A = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    title="Shifted B field");
for (idx, dir) in enumerate(data_directories)
    data = exp_data_corr[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig3A, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig3A,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig3A,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
    label="SG PHYWE calibration",
    line=(:solid,2,:dodgerblue4));
plot!(fig3A,phywe_shift[:,1], 1000*phywe_shift[:,2],
    label="SG PHYWE calibration (adjusted)",
    line=(:dot,2,:dodgerblue4));
plot!(fig3A,
    # xscale=:log10,
    # yscale=:log10,
    xlims=(0.0,15e-3),
    ylims=(-5,15),
    # xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # yticks = ([0.01, 0.1, 1, 10, 100], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig3A)

fig4A = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    title="Shifted B field");
for (idx, dir) in enumerate(data_directories)
    data = exp_data_corr[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig4A, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:dot, colores_data[idx], 1)
    )

end
plot!(fig4A,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig4A,BvsI_phywe.Ic, 1000*BvsI_phywe.Bz,
    label="SG PHYWE calibration",
    line=(:solid,2,:dodgerblue4));
plot!(fig4A,phywe_shift[:,1], 1000*phywe_shift[:,2],
    label="SG PHYWE calibration (adjusted)",
    line=(:dot,2,:dodgerblue4));
plot!(fig4A,
    xlims=(0.600,1.10),
    ylims=(300,900),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig4A)


BvsI_comparison_corr = Vector{Matrix{Float64}}(undef, n_data)
for (idx, dir) in enumerate(data_directories)
    vs = exp_data_corr[idx]

    mask, inds, rows_view = DataReading.subset_by_cols(vs, [1,2]; thr=1e-3, include_equal = true)
    BvsI_comparison_corr[idx] = Matrix(rows_view)  # store a copy; or keep the view (see B)
end


fig5A = plot(xlabel="Currents (A)",
    ylabel=L"$B_{\mathrm{exp}} / B_{\mathrm{manual}}$",
    title="Shifted B field")
for (idx, dir) in enumerate(data_directories)
    B_ratio = BvsI_comparison_corr[idx][:,2] ./ TheoreticalSimulation.BvsI.(BvsI_comparison_corr[idx][:,1])
    
    plot!(fig5A,
        BvsI_comparison_corr[idx][:,1] , B_ratio,
        label=data_directories[idx],
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
        line=(:solid,1,colores_data[idx])
    )
    y0 , σ0 = mean(B_ratio), standard_error(B_ratio)

    x = range(1e-3, 1.1, length=200)
    plot!(fig5A,
    x, fill(y0, length(x)),
    ribbon = σ0,
    color = colores_data[idx],
    fillalpha = 0.25,
    line=(:dash,0.5,colores_data[idx]),
    label = "$(round(y0; digits=3)) ± $(round(σ0; sigdigits=1))")
end
plot!(fig5A,legend=:bottomright,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.05),
    ylims=(3e-1,2),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    legendfontsize = 6,
    legend_columns=2,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
)
display(fig5A)

plot(fig5,fig5A,
layout=(1,2),
size=(1200,600),
left_margin=5mm,
bottom_margin=2mm,
)

plot(fig1,fig2, fig3,fig4,
layout=(2,2),
size=(1200,700),
left_margin=5mm,
bottom_margin=2mm,
)

plot(fig1A,fig2A, fig3A,fig4A,
layout=(2,2),
size=(1200,700),
left_margin=5mm,
bottom_margin=2mm,
)

########################################################################################
## ######################################################################################
#+++++++++++++++++++++++++++ Centroid +++++++++++++++++++++++++++++++++++++++++++++
centroid_fw = Matrix{Float64}(undef, n_data, 2)
pos_at_zero = Matrix{Float64}(undef, n_data, 4)
for (i,dir) in enumerate(data_directories)
    kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS","summary", dir, dir * "_report_summary.jld2")

    data = jldopen(kk_path, "r") do file
        dd = file[JLD2_MyTools.make_keypath_exp(dir,nz,λ0)]
    end

    # zc ± δzc (computed from the high currents)
    centroid_fw[i,1] = data[:centroid_fw_mm][1]
    centroid_fw[i,2] = round(data[:centroid_fw_mm][2], digits=3)

    # zc ± δzc (for the signal at zero current for F1 and F2)
    pos_at_zero[i,1] = data[:fw_F1_peak_pos_raw][1][1]
    pos_at_zero[i,2] = data[:fw_F1_peak_pos_raw][2][1]
    pos_at_zero[i,3] = data[:fw_F2_peak_pos_raw][1][1]
    pos_at_zero[i,4] = data[:fw_F2_peak_pos_raw][2][1]

end

fig = plot(data_directories, centroid_fw[:,1],
label="Computed centroid positions (high currents)",
 color=:blue,
 marker=(:circle,:white,4),
 markerstrokecolor=:blue,
 line=(:solid,2,:blue),
 yerror = centroid_fw[:,2],);
plot!(data_directories,pos_at_zero[:,1],
    ribbon=pos_at_zero[:,2],
    label=L"$F=1$ at ($I_{c}=0A$)",
    color=:orangered2,
    marker=(:circle,:white),
    markerstrokecolor=:orangered2,
    fillalpha=0.2);
plot!(data_directories,pos_at_zero[:,3],
    ribbon=pos_at_zero[:,4],
    label=L"$F=2$ at ($I_{c}=0A$)",
    color=:springgreen2,
    marker=(:circle,:white),
    markerstrokecolor=:springgreen2,
    fillalpha=0.2);
# plot!(["20260225","20260226am","20260226pm","20260227","20260303"],[8.8369,8.8440,8.8549,8.8284,8.8292],
#     label="Qihang's fitting 1",
#     marker=(:square,2,:white),
#     markerstrokecolor=:purple,
#     line=(:purple))
# plot!(["20260225","20260226am","20260226pm","20260227","20260303"], [8.811211551,8.790075956,8.796932977,8.84320487,8.8048355],
#     label="Qihang's fitting 2",
#     marker=(:square,2,:white),
#     markerstrokecolor=:green,
#     line=(:green))
plot!(
    # title = "Std.Dev=$(round(std(centroid_fw[:,1], corrected=false); digits=3))",
    legend=:bottomleft,
    ylims=(8.700,8.950),
    yformatter = y -> @sprintf("%.3f", y),
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    xminorticks=false,
    bottom_margin=5mm,
    ylabel="Position (mm)",
    xrotation=75,);
display(fig)


########################################################################################
#+++++++++++++++++++++++++++ Peak Position +++++++++++++++++++++++++++++++++++++++++++++

#+++++++++ QUANTUM MECHANICS +++++++++++++
chosen_qm_f1 =  jldopen(data_qmf1_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end;
chosen_qm_f2 =  jldopen(data_qmf2_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end;
data_QM = hcat(
    [v[:Icoil]                  for v in values(chosen_qm_f1)],
    [v[:z_max_smooth_spline_mm] for v in values(chosen_qm_f1)],
    [v[:z_max_smooth_spline_mm] for v in values(chosen_qm_f2)],
)
data_QM = hcat(data_QM, data_QM[:,2] .- data_QM[:,3]);
QM_df = DataFrame(data_QM, [:Ic, :F1, :F2, :Δ]);
pretty_table(QM_df; 
            column_labels = [["Ic", "F1", "F2", "Δ"],["(A)","(mm)","(mm)","(mm)"]],
            alignment =:c,
            row_label_column_alignment  = :c,
            row_group_label_alignment   = :c,
            title         = @sprintf("QUANTUM MECHANICS"),
            formatters = [fmt__printf("%8.3f", [1]), fmt__printf("%8.6f",2:4)],
            style = TextTableStyle(
                    first_line_column_label = crayon"yellow bold",
                    table_border  = crayon"blue bold",
                    title = crayon"bold red"
            ),
            table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
            equal_data_column_widths= true,
)

r = QM_df[!,:F2] ./ QM_df[!,:F1];
fig0a = plot(QM_df[!,:Ic], r,
    marker=(:square,2,:white),
    markerstrokecolor=:orangered,
    line=(:solid,1,:orangered),
    xlims=(1e-3,1.05),
    label=L"$I_{c}=%$(QM_df[1,:Ic])\mathrm{A}$, $r=%$(r[1])$",
);
plot!(fig0a, 
    xscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
);
plot!(fig0a,
    xlabel="Current (A)",
    ylabel=L"$z_{\mathrm{max}}(F=2)/z_{\mathrm{max}}(F=1)$"
);
display(fig0a)

rc = 0.5*(QM_df[!,:F1] .+ QM_df[!,:F2]);
fig0b=plot(QM_df[!,:Ic], 1000*rc,
    marker=(:square,2,:white),
    markerstrokecolor=:orangered,
    line=(:solid,1,:orangered),
    label=L"$I_{c}=%$(QM_df[1,:Ic])\mathrm{A}$, $z_{c}=%$(1e6 * round(rc[1]; sigdigits=1))\times 10^{-3}\mathrm{\mu m}$",
    xlims=(1e-3,1.05),
)
plot!(fig0b, 
    xscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
);
plot!(fig0b,
    xlabel="Current (A)",
    ylabel=L"$\frac{1}{2}\left(z_{\mathrm{max}}(F=1) + z_{\mathrm{max}}(F=2)\right) \ \mathrm{\mu m}$",
    left_margin=3mm,
    legend=:topleft,
);
display(fig0b)

fig0 = plot(fig0a, fig0b,
layout=(2,1),
link=:x,
size=(1000,700),
left_margin=4mm,)
plot!(fig0[1], xlabel="", xformatter=_->"", bottom_margin=-8mm)


fig00 = plot(xlabel=L"$z \ (\mathrm{mm})$")
for i=1:2:nI
    data = chosen_qm_f1[i][:z_profile][:,[1,3]]
    plot!(fig00,
        data[:,1], data[:,2],
        label=L"$I_{c}=%$(1000*Icoils[i])\mathrm{mA}$",
        line=(:solid,2,colores_current[i]))
end
plot!(fig00,
    yformatter = y -> @sprintf("%.1e", y),
    xlims=(-5,5),
    legend=:outertop,
    legend_columns=5,
    background_color_legend = nothing,
    foreground_color_legend = nothing,)
display(fig00)

fig01 = plot(xlabel=L"$z \ (\mathrm{mm})$")
for i=1:2:nI
    data = chosen_qm_f2[i][:z_profile][:,[1,3]]
    plot!(fig01,
        data[:,1], data[:,2],
        # label=L"$I_{c}=%$(1000*Icoils[i])\mathrm{mA}$",
        line=(:solid,2,colores_current[i]))
end
plot!(fig01,
    yformatter = y -> @sprintf("%.1e", y),
    xlims=(-5,5),
    legend=false,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig01)

fig=plot(fig00, fig01,
layout=(2,1),
size=(900,600),
)
plot!(fig[1], xlabel="", xformatter=_->"", bottom_margin=-5mm)
display(fig)

fig1 = plot(
    xlabel="Currents (A)",
    ylabel="Position (mm)",
);
plot!(fig1,
    QM_df[!,:Ic],QM_df[!,:F1],
    label=L"$F=1$",
    marker=(:circle,2,:white),
    markerstrokecolor=:red,
    line=(:red,1),
);
plot!(fig1, 
    QM_df[!,:Ic],QM_df[!,:F2],
    label=L"$F=2$",
    marker=(:circle,2,:white),
    markerstrokecolor=:blue,
    line=(:blue,1)
);
plot!(fig1,
    xlims=(1e-3,1.05),
    xscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:topleft,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
);
display(fig1)

fig2 = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-Peak (mm)",
);
plot!(fig2,
    QM_df[!,:Ic],QM_df[!,:Δ],
    label=L"$\Delta z$",
    marker=(:circle,2,:white),
    markerstrokecolor=:purple,
    line=(:purple,1),
);
plot!(fig2,
    xlims=(1e-3,1.05),
    ylims=(1e-4,5.0),
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:bottomright,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    
);
display(fig2)

fig = plot(fig1, fig2,
layout=(2,1),
link=:x,
labelfontsize=10)
plot!(fig[1], xlabel="", xformatter=_->"", bottom_margin=-8mm);
display(fig)



#+++++++++ COQUANTUM DYNAMICS +++++++++++++
cqd_sim_data = JLD2_MyTools.list_keys_jld_cqd(data_cqdup_path)
n_ki = length(cqd_sim_data.ki);
@info "CQD simulation for $(n_ki) ki values"
colores_ki = palette(:darkrainbow, n_ki);
up_cqd = Matrix{Float64}(undef, nI, n_ki);
dw_cqd = Matrix{Float64}(undef, nI, n_ki);
Δz_cqd = Matrix{Float64}(undef, nI, n_ki);

for (i,ki) in enumerate(cqd_sim_data.ki)
    dataup = jldopen(data_cqdup_path,"r") do f
        f[JLD2_MyTools.make_keypath_cqd(:up,ki,nz,σw_mm,λ0)]
    end
    up_cqd[:,i] = [dataup[j][:z_max_smooth_spline_mm] for j=1:nI]

    datadw = jldopen(data_cqddw_path,"r") do f
        f[JLD2_MyTools.make_keypath_cqd(:dw,ki,nz,σw_mm,λ0)]
    end
    dw_cqd[:,i] = [datadw[j][:z_max_smooth_spline_mm] for j=1:nI]

    Δz_cqd[:,i] = up_cqd[:,i] .- dw_cqd[:,i]
end

fig1 = plot(xlabel="Current (A)",
    ylabel="Position (mm)");
for (i,ki) in enumerate(cqd_sim_data.ki)
    ki_string = sigdigits_str(ki, 2)
    plot!(fig1,
        Icoils, up_cqd[:,i],
        label=L"$%$(ki_string)$",
        line=(:solid,1,colores_ki[i]),

    )
    plot!(fig1,
        Icoils, dw_cqd[:,i],
        label=false,
        line=(:dot,1,colores_ki[i])

    )
end
plot!(fig1,
    xlims=(1e-3,1.05),
    xscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    legendtitle=L"$k_{i} \times 10^{-6}$",
    legendtitlefontsize=8,
    legendfontsize=6,
    legend_columns=3,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size=(800,600),
    left_margin=2mm,
)
display(fig1)


fig2 = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-Peak (mm)",
);
for (i,ki) in enumerate(cqd_sim_data.ki)
    # ki_string = @sprintf("%2.3f",ki)
    ki_string = sigdigits_str(ki, 2)
    plot!(fig2,
        Icoils, Δz_cqd[:,i],
        label=L"$%$(ki_string)$",
        line=(:solid,1,colores_ki[i]),

    )
end
plot!(fig2,
    xlims=(1e-3,1.05),
    ylims=(1e-3,5.0),
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    legendtitle=L"$k_{i} \times 10^{-6}$",
    legendtitlefontsize=8,
    legendfontsize=6,
    legend_columns=2,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size=(800,600),
    left_margin=2mm,
);
display(fig2)

fig = plot(fig1, fig2,
    size=(1100,900),
    layout=(2,1),
    link=:x,
    left_margin=5mm,
    bottom_margin=2mm,
);
plot!(fig[1], xlabel="", xformatter=_->"",bottom_margin=-9mm,legend=false);
plot!(fig[2], bottom_margin=-10mm,legend=:outerbottom, legend_columns=7);
display(fig)



function plot_combined_cqd_profiles_dict(
    ki_values,
    data_cqdup_path::AbstractString,
    data_cqddw_path::AbstractString;
    nz::Integer,
    σw_mm::Real,
    λ0::Real,
    nI::Integer,
    Icoils,
    colores_current,
    λ0_peak::Real = 1e-6,
)
    results = OrderedDict{Float64, Vector{Float64}}()

    for ki_test in ki_values

        dat_branch_up = jldopen(data_cqddw_path, "r") do f
            f[JLD2_MyTools.make_keypath_cqd(:dw, ki_test, nz, σw_mm, λ0)]
        end

        dat_branch_dw = jldopen(data_cqdup_path, "r") do f
            f[JLD2_MyTools.make_keypath_cqd(:up, ki_test, nz, σw_mm, λ0)]
        end

        fig1 = plot(xlabel = L"$z \ (\mathrm{mm})$",
            title = L"$k_{i}=$ "*sci_label(ki_test/1e6, n=6),
        )
        for ii in 1:nI
            z_new     = dat_branch_up[ii][:z_profile][:, 1]
            ampli_new = dat_branch_up[ii][:z_profile][:, 3]
            plot!(
                fig1,
                z_new,
                ampli_new,
                label = "$(Int(1000 * Icoils[ii])) mA",
                color = colores_current[ii],
            )
        end
        plot!(
            fig1;
            yformatter = y -> @sprintf("%.1e", y),
            xlims = (-5, 5),
            legend = false,
            legend_columns = 6,
            background_color_legend = nothing,
            foreground_color_legend = nothing,
        )
        # display(fig1)



        zmax_vec = Vector{Float64}(undef, nI)

        fig2 = plot(
            xlabel = L"$z \ (\mathrm{mm})$",
            # title = L"$k_{i}=$ "*sci_label(ki_test/1e6, n=6),
        );
        for ii in 1:nI
            z_new     = dat_branch_up[ii][:z_profile][:, 1]
            ampli_new = dat_branch_up[ii][:z_profile][:, 3] .+
                        0.25 .* dat_branch_dw[ii][:z_profile][:, 3]

            z_max_new, _ = TheoreticalSimulation.max_of_bspline_positions(
                z_new, ampli_new; λ0 = λ0_peak
            )

            zmax_vec[ii] = z_max_new[1]         

            plot!(
                fig2,
                z_new,
                ampli_new,
                label = "$(Int(1000 * Icoils[ii])) mA",
                color = colores_current[ii],
            )
        end        
        plot!(
            fig2;
            yformatter = y -> @sprintf("%.1e", y),
            xlims = (-5, 5),
            legend = :outerbottom,
            legend_columns = 7,
            legend_fontsize = 6,
            background_color_legend = nothing,
            foreground_color_legend = nothing,
        )
        # display(fig2)




        fig = plot(fig1, fig2,
        layout=(2,1),
        # link=:x,
        labelfontsize=10,
        size=(900,600),
        )
        plot!(fig[1], xlabel="", xformatter=_->"", bottom_margin=-2mm);
        display(fig)

        z_old = [dat_branch_up[ix][:z_max_smooth_spline_mm] for ix=1:nI]
        pct   = 100 * (zmax_vec ./ z_old .- 1)
        pretty_table( hcat(
            Int.(1000 * Icoils), 
            round.(zmax_vec; digits = 3), 
            round.(z_old; digits = 3),
            round.(pct; digits = 1)
        );
            column_labels = [["Ic", "CQD F2", "CQD down ", "RelError"],["(A)","(mm)","(mm)","(mm)"]],
            alignment =:c,
            row_label_column_alignment  = :c,
            row_group_label_alignment   = :c,
            title         = @sprintf("CQD EQUIVALENT TO F2"),
            formatters = [fmt__printf("%d", [1]), fmt__printf("%8.3f",2:3), fmt__printf("%8.3f", [4])],
            style = TextTableStyle(
                    first_line_column_label = crayon"yellow bold",
                    table_border  = crayon"blue bold",
                    title = crayon"bold red"
            ),
            table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
            equal_data_column_widths= true,
        )

        results[ki_test] = zmax_vec
    end

    return results
end

zmax_dict = plot_combined_cqd_profiles_dict(
    cqd_sim_data[:ki][1:2],
    data_cqdup_path,
    data_cqddw_path;
    nz = nz,
    σw_mm = σw_mm,
    λ0 = λ0,
    nI = nI,
    Icoils = Icoils,
    colores_current = colores_current,
)




for ki_test in cqd_sim_data[:ki]
dat_branch_dw = jldopen(data_cqdup_path,"r") do f
    f[JLD2_MyTools.make_keypath_cqd(:up,ki_test,nz,σw_mm,λ0)]
end

dat_branch_up = jldopen(data_cqddw_path,"r") do f
    f[JLD2_MyTools.make_keypath_cqd(:dw,ki_test,nz,σw_mm,λ0)]
end

fig = plot(xlabel=L"$z \ (\mathrm{mm})$")
for ii = 1:nI
z_new = dat_branch_up[ii][:z_profile][:,1]
ampli_new = dat_branch_up[ii][:z_profile][:,3] .+ 0.25*dat_branch_dw[ii][:z_profile][:,3]
z_max_new, _ = TheoreticalSimulation.max_of_bspline_positions(z_new,ampli_new;λ0=1e-6)


println(Int(1000*Icoils[ii]),"\t\t",
    round(z_max_new[1]; digits=3), 
    "\t\t ", 
    round(dat_branch_up[ii][:z_max_smooth_spline_mm]; digits=3),
    "\t\t",
    round(100 * (z_max_new[1] ./ dat_branch_up[ii][:z_max_smooth_spline_mm] .- 1) ; digits=1)
)

plot!(z_new, ampli_new,
    label="$(Int(1000*Icoils[ii])) mA",
    color=colores_current[ii])
end
plot!(
    xlims=(-5,5),
    legend=:outerright,
    legend_columns=2)
display(fig)
end

#+++++++++ QUANTUM MECHANICS & COQUANTUM DYNAMICS +++++++++++++
fig1 = plot(
    xlabel="Currents (A)",
    ylabel="Position (mm)",
);
plot!(fig1,
    QM_df[!,:Ic],QM_df[!,:F1],
    label=L"$F=1$",
    seriestype=:scatter,
    marker=(:circle,2,:white),
    markerstrokecolor=:red,
    line=(:red,1),
);
plot!(fig1, 
    QM_df[!,:Ic],QM_df[!,:F2],
    label=L"$F=2$",
    seriestype=:scatter,
    marker=(:circle,2,:white),
    markerstrokecolor=:blue,
    line=(:blue,1)
);
for (i,ki) in enumerate(cqd_sim_data.ki)
    ki_string = sci_label(1e-6*ki; n=2)
    plot!(fig1,
        Icoils, up_cqd[:,i],
        label=L"$k_{i}=$%$(ki_string)",
        line=(:solid,1,colores_ki[i]),

    )
    plot!(fig1,
        Icoils, dw_cqd[:,i],
        label=false,
        line=(:dot,1,colores_ki[i])

    )
end
plot!(fig1,
    xlims=(1e-3,1.05),
    xscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    legendfontsize=6,
    legend_columns=2,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size=(1000,600),
    left_margin=5mm,
)
display(fig1)


fig2 = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-Peak (mm)",
);
for (i,ki) in enumerate(cqd_sim_data.ki)
    ki_string = sci_label(1e-6*ki; n=2)
    plot!(fig2,
        Icoils, Δz_cqd[:,i],
        label=L"$k_{i}=$%$(ki_string)",
        line=(:solid,1,colores_ki[i]),

    )
end
plot!(fig2,
    QM_df[!,:Ic],QM_df[!,:Δ],
    label=L"QM $\Delta z$",
    marker=(:circle,2,:white, 0.55),
    markerstrokecolor=:black,
    line=(:black,2),
);
plot!(fig2,
    xlims=(1e-3,1.05),
    ylims=(1e-4,5.0),
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:outerright,
    legend_columns = 2,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,  
    size=(1000,600);
    left_margin=4mm,
    bottom_margin=2mm,
)
display(fig2)

fig = plot(fig1, fig2,
size=(1200,1000),
layout=(2,1),
link=:x,
left_margin=5mm,
bottom_margin=2mm,
);
plot!(fig[1], xlabel="", xformatter=_->"",bottom_margin=-9mm,legend=false);
plot!(fig[2], xlabel="", xformatter=_->"",bottom_margin=-3mm,legend=:outerbottom, legend_columns=7);
display(fig)


#+++++++++++++ EXPERIMENTS ++++++++++++++++++++
EXP_data = OrderedDict{String, NamedTuple}()
for dir in data_directories
    println(dir)
    kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS", "summary", dir, dir * "_report_summary.jld2")

    EXP_data[dir] = jldopen(kk_path, "r") do file
        ic = file["meta/Currents"]
        bz = file["meta/BzTesla"]

        dd = file[JLD2_MyTools.make_keypath_exp(dir, nz, λ0)]

        F1 = dd[:fw_F1_peak_pos_raw]
        F2 = dd[:fw_F2_peak_pos_raw]

        zc = 0.5 * (F1[1][1] + F2[1][1])
        δc = 0.5 * sqrt(F1[2][1]^2 + F2[2][1]^2)

        (
            Ic = ic,
            Bz = bz,
            F1 = F1,
            F2 = F2,
            C0 = dd[:centroid_fw_mm],
            C00 = [zc, δc]
        )
    end
end

plot(xlabel="Currents (A)",
ylabel=L"$z_{1}-z_{2}$ (mm)")
hspan!([-6.5e-3,6.5e-3], color=:gray68, fillalpha=0.8, label="pixel size")
for i = 1:n_data
plot!(EXP_data[data_directories[i]].Ic, EXP_data[data_directories[i]].F1[1] .- EXP_data[data_directories[i]].F2[1],
    label=data_directories[i],
    marker=(:square,3,:white),
    markerstrokewidth=2,
    markerstrokecolor=colores_data[i],
    line=(:dash, 1, colores_data[i]))
end
plot!(legend=:outerright,
    xlims=(-0.1e-3,5e-3),
    ylims=(-10e-3,20e-3),
    # xscale=:log10,
    # yscale=:log10,
)


EXP_data_processed = OrderedDict{String, DataFrame}()
for 𝓁 = 1:n_data
    dir = data_directories[𝓁]
    data_dir = DataFrame(hcat(
            EXP_data[dir].Ic,
            EXP_data[dir].F1[1] .- EXP_data[dir].C00[1],
            round.(sqrt.(EXP_data[dir].F1[2].^2 .+ EXP_data[dir].C00[2].^2); sigdigits=1),
            EXP_data[dir].F2[1] .- EXP_data[dir].C00[1],
            round.(sqrt.(EXP_data[dir].F2[2].^2 .+ EXP_data[dir].C00[2].^2); sigdigits=1),
            EXP_data[dir].F1[1] .- EXP_data[dir].F2[1],
            round.(sqrt.(EXP_data[dir].F1[2].^2 .+ EXP_data[dir].F2[2].^2); sigdigits=1),
        ),
        [:Ic, :F1, :ErrF1, :F2, :ErrF2, :Δ, :ErrΔ]
        )
    EXP_data_processed[dir] = data_dir

    pretty_table(data_dir; 
                alignment =:c,
                title         = "EXPERIMENT $(data_directories[𝓁])",
                formatters = [fmt__printf("%8.4f", [1]), fmt__printf("%8.4f",[2,4,6]), fmt__printf("%8.3f",[3,5,7])],
                style = TextTableStyle(
                        first_line_column_label = crayon"yellow bold",
                        table_border  = crayon"blue bold",
                        title = crayon"bold red"
                ),
                table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
                equal_data_column_widths= true,
    )


    data_f1 = data_dir[data_dir.F1 .> 0, :]

    figa = plot(xlabel="Currents (A)",
    ylabel=L"$F=1$ position (mm)");
    plot!(figa, data_f1.Ic, data_f1.F1,
        yerror = data_f1.ErrF1,
        label="$(data_directories[𝓁])",
        marker=(:square,2,:white),
        markerstrokecolor=:black,
        line=(:solid,:black)
    );
    plot!(figa,
        QM_df[!,:Ic],QM_df[!,:F1],
        label=L"$F=1$",
        marker=(:circle,2,:white),
        markerstrokecolor=:red,
        line=(:red,2),
    );
    for (i,ki) in enumerate(cqd_sim_data.ki)
        ki_string = @sprintf("%2.2f",ki)
        plot!(figa,
            Icoils, up_cqd[:,i],
            label=L"$k_{i}=%$(ki_string)\times 10^{-6}$",
            line=(:solid,1,colores_ki[i], 0.5),
        )
    end
    plot!(figa,
        xlims=(5e-3,1.05),
        ylims=(1e-3,2.0),
        legend=:outerright,
        legend_columns = 2,
        legendfontsize=6,
        background_color_legend = nothing,
        foreground_color_legend = nothing,
        size=(1000,400),
        left_margin=5mm,
        bottom_margin=5mm
    );
    figb = deepcopy(figa);
    # display(figb)
    plot!(figa,
        xscale=:log10,
        yscale=:log10,
        xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    );
    # display(figa)

    fig = plot(figa,figb,
        layout=(2,1),
        size=(1200,1000),
        top_margin=5mm)
    display(fig)


    data_Δz = data_dir[data_dir.Δ .> 0, :];

    figc = plot(
        xlabel="Currents (A)",
        ylabel="Peak-to-Peak (mm)",
    );
    for (i,ki) in enumerate(cqd_sim_data.ki)
        ki_string = @sprintf("%2.2f",ki)
        plot!(figc,
            Icoils, Δz_cqd[:,i],
            label=L"$k_{i}=%$(ki_string)\times 10^{-6}$",
            line=(:solid,1,colores_ki[i]),

        )
    end
    plot!(figc,
        QM_df[!,:Ic],QM_df[!,:Δ],
        label=L"QM $\Delta z$",
        marker=(:circle,2,:white, 0.55),
        markerstrokecolor=:lime,
        line=(:lime,1),
    );
    plot!(figc, data_Δz.Ic, data_Δz.Δ,
        yerror = data_Δz.ErrΔ,
        label="$(data_directories[𝓁])",
        marker=(:square,2,:white),
        markerstrokecolor=:black,
        line=(:solid,:black)
    );
    plot!(figc,
        xlims=(5e-6,1.05),
        ylims=(1e-3,4.2),
        legend=:outerright,
        legend_columns = 2,
        legendfontsize=6,
        background_color_legend = nothing,
        foreground_color_legend = nothing,
        size=(1000,400),
        left_margin=5mm,
        bottom_margin=5mm
    );
    figd = deepcopy(figc);
    # display(figd)
    plot!(figc,
        xlims=(1e-3,1.05),
        ylims=(1e-4,5.0),
        xscale=:log10,
        yscale=:log10,
        xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        legend=:outerright,
        legend_columns = 2,
        legendfontsize=6,
        background_color_legend = nothing,
        foreground_color_legend = nothing,  
    );
    # display(figc)

    fig = plot(figc,figd,
        layout=(2,1),
        size=(1200,1000),
        top_margin=7mm)
    display(fig)

end


#++++++ DATA COMPARISON
figa = plot(
    xlabel="Currents (A)",
    ylabel="Position (mm)"
)
for (i,dir) in enumerate(data_directories)
    plot!(figa,
        EXP_data_processed[dir].Ic, EXP_data_processed[dir].F1,
        yerror = EXP_data_processed[dir].ErrF1,
        label= dir,
        marker=(:circle,2,:white,0.6),
        markerstrokecolor=colores_data[i],
        line=(:solid,1,colores_data[i])
    )
end
plot!(figa,
    QM_df[!,:Ic],QM_df[!,:F1],
    label=L"QM : $F=1$",
    seriestype=:scatter,
    marker=(:diamond,2,:white),
    markerstrokecolor=:black,
    line=(:black,1),
)
plot!(figa,
    legend=:bottomright,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    xlims=(10e-3, 1.05),
    ylims=(1e-3, 2.05))
figa1= deepcopy(figa)
plot!(figa,
    xscale=:log10,
    yscale=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)

figb = plot(
    xlabel="Currents (A)",
    ylabel="Position (mm)"
)
for (i,dir) in enumerate(data_directories)
    plot!(figb,
        EXP_data_processed[dir].Ic, EXP_data_processed[dir].F2,
        yerror = EXP_data_processed[dir].ErrF2,
        label= dir,
        marker=(:circle,2,:white,0.6),
        markerstrokecolor=colores_data[i],
        line=(:solid,1,colores_data[i])
    )
end
plot!(figb,
    QM_df[!,:Ic],QM_df[!,:F2],
    label=L"QM : $F=2$",
    seriestype=:scatter,
    marker=(:diamond,2,:white),
    markerstrokecolor=:black,
    line=(:black,1),
)
plot!(figb,
    xlims=(10e-3, 1.05),
    ylims=(-2.05, -6e-3))
plot!(figb,
    legend=:topright,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)


figc = plot(
    xlabel="Currents (A)",
    ylabel="Peak-to-position (mm)"
)
for (i,dir) in enumerate(data_directories)
    plot!(figc,
        EXP_data_processed[dir].Ic, EXP_data_processed[dir].Δ ,
        yerror = EXP_data_processed[dir].ErrΔ,
        label= dir,
        marker=(:circle,2,:white,0.6),
        markerstrokecolor=colores_data[i],
        line=(:solid,1,colores_data[i])
    )
end
plot!(figc,
    QM_df[!,:Ic],QM_df[!,:Δ],
    label=L"QM : $\Delta z$",
    seriestype=:scatter,
    marker=(:diamond,2,:white),
    markerstrokecolor=:black,
    line=(:black,1),
)
plot!(figc,
    legend=:bottomright,
    legendfontsize=6,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    xlims=(10e-3, 1.05),
    ylims=(1e-3, 4.05),
)
figc1 = deepcopy(figc)
plot!(figc,xscale=:log10, yscale=:log10,
 xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
 yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)


fig = plot(figa,figa1, figb,figc, figc1,
labelfontsize = 10,
layout=@layout[ a1 a2 ; a3; a4 a5],
size=(1200,1000),
top_margin=2mm,
left_margin=5mm,
bottom_margin=3mm,
);
display(fig)

#++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++
threshold = 0.030 ; # lower cut-off for experimental currents

tol_grouping = 0.04;
Ics = [EXP_data_processed[dir].Ic for dir in data_directories];
clusters = MyExperimentalAnalysis.cluster_by_tolerance(Ics; tol=tol_grouping);
for s in clusters.summary
    println("Value group ≈ $(@sprintf("%1.3f", s.mean_val)) ± $(round(s.std_val; sigdigits=1)) \t appears in datasets: ", s.datasets)
end
Ic_grouped  = round.(getproperty.(clusters.summary, :mean_val); digits=6);
δIc_grouped = round.(getproperty.(clusters.summary, :std_val);  sigdigits=2);
mask = Ic_grouped .>= threshold;
grouped_exp_current = DataFrame(
    Ic  = Ic_grouped[mask],
    δIc = δIc_grouped[mask],
)

fig=plot(grouped_exp_current.Ic, 
    ribbon=grouped_exp_current.δIc,
    marker=(:diamond, 2, :white),
    markerstrokecolor=:red,
    line=(:solid,1,:red),
    color=:red,
    fillalpha=0.2,
    label=false,
    xlabel="current number",
    ylabel="Current (A)",
    ylims=(7e-4,1.05),
);
plot!(
    yscale=:log10,
    legend=:bottomright,
);
display(fig)

CURRENT_ROW_START = [first_gt_idx(EXP_data_processed[t], :Ic, threshold) for t in keys(EXP_data_processed)];
Ic_sets  = [EXP_data_processed[t][i:end, :Ic]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];
F1_sets  = [EXP_data_processed[t][i:end, :F1]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];
F2_sets  = [EXP_data_processed[t][i:end, :F2]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];
σF1_sets = [EXP_data_processed[t][i:end, :ErrF1]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];
σF2_sets = [EXP_data_processed[t][i:end, :ErrF2]  for (t,i) in zip(keys(EXP_data_processed), CURRENT_ROW_START)];

# pick a log-spaced grid across the overall I-range (nice for decades-wide currents)
i_sampled_length = 65 ;
xlo, xhi = maximum([minimum(first.(Ic_sets)),threshold]) ,  maximum([maximum(last.(Ic_sets)),1.01]);
Ic_sampling  = exp10.(range(log10(xlo), log10(xhi), length=i_sampled_length))


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Monte Carlo analysis
MC_data_F1 = combine_on_grid_mc_weighted(Ic_sets, F1_sets; σxsets=nothing, σysets=σF1_sets, B=5000, xq=grouped_exp_current.Ic);
MC_data_F2 = combine_on_grid_mc_weighted(Ic_sets, F2_sets; σxsets=nothing, σysets=σF2_sets, B=5000, xq=grouped_exp_current.Ic);

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
    xlims = (10e-3,1.0),
    ylims = (1e-3, 2),
    legend=:bottomright,
)
for (i,dir) in enumerate(data_directories)
    xs = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:Ic]
    ys = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:F1]
    δys = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:ErrF1]
    scatter!(fig,xs,ys,
        yerror= δys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=colores_data[i],
        markerstrokewidth=1,
        )
end
plot!(fig, QM_df[!,:Ic], QM_df[!,:F1], label="QM", line=(:black,:dashdot,2));
plot!(fig, MC_data_F1.xq, MC_data_F1.μ; 
    ribbon=MC_data_F1.σ_tot,
    label=false,
    color=:maroon1,
)
plot!(fig,
    xscale=:log10,
    yscale=:log10, 
    title = "Weighted Monte Carlo combiner",
    color=:black,
)
display(fig)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Spline interpolation

# i_xx0 = unique(round.(sort(union(Ic_sampling,grouped_exp_current.Ic)); digits=6))
i_xx0 = unique(round.(sort(grouped_exp_current.Ic); digits=6))
i_sampled_length = length(i_xx0)

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
    xlims = (10e-3,1.0),
    ylims = (1e-3, 2),
)
z1_final = zeros(n_data,i_sampled_length)
z2_final = zeros(n_data,i_sampled_length)
for (i,dir) in enumerate(data_directories)
    xs = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:Ic]
    ys = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:F1]
    δys = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:ErrF1]
    splF1 = BSplineKit.extrapolate(BSplineKit.interpolate(xs,ys, BSplineKit.BSplineOrder(4),BSplineKit.Natural()),BSplineKit.Linear())
    z1_final[i,:] = splF1.(i_xx0)
    scatter!(fig,xs, ys,
        yerror= δys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=colores_data[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx0,splF1.(i_xx0),
        label=false,
        line=(colores_data[i],1))

    xs = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:Ic]
    ys = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:F2]
    δys = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:ErrF2]
    splF2 = BSplineKit.extrapolate(BSplineKit.interpolate(xs,ys, BSplineKit.BSplineOrder(4),BSplineKit.Natural()),BSplineKit.Linear())
    z2_final[i,:] = splF2.(i_xx0)
end
plot!(fig, QM_df[!,:Ic], QM_df[!,:F1], label="QM", line=(:black,:dashdot,2));
display(fig)
plot!(fig,
title="Interpolation: cubic splines",
legend=:bottomright,
xaxis=:log10, 
yaxis=:log10,
xticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)
spl_zf1 = vec(mean(z1_final, dims=1))
spl_δzf1 = vec(std(z1_final; dims=1, corrected=true))/sqrt(n_data)
spl_zf2 = vec(mean(z2_final, dims=1))
spl_δzf2 = vec(std(z2_final; dims=1, corrected=true))/sqrt(n_data)
data_spl = hcat(i_xx0, spl_zf1, spl_δzf1, spl_zf2, spl_δzf2)
plot!(fig, data_spl[:,1], data_spl[:,2],
    ribbon = data_spl[:,3],
    fillalpha=0.40, 
    fillcolor=:green3, 
    label=false,
    line=(:dash,:green3,2))
display(fig)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Fit Spline interpolation
fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
)
z1_final_fit = zeros(n_data,i_sampled_length)
z2_final_fit = zeros(n_data,i_sampled_length)
for (i,dir) in enumerate(data_directories)
    xs = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:Ic]
    ys = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:F1]
    δys = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:ErrF1]
    splF1 = BSplineKit.extrapolate(BSplineKit.fit(BSplineKit.BSplineOrder(4),xs,ys, 0.005, BSplineKit.Natural(); weights=1 ./ δys.^2),BSplineKit.Smooth())
    z1_final_fit[i,:] = splF1.(i_xx0)
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=colores_data[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx0,splF1.(i_xx0),
        label=false,
        line=(colores_data[i],1))

    xs = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:Ic]
    ys = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:F2]
    δys = EXP_data_processed[dir][CURRENT_ROW_START[i]:end,:ErrF2]
    splF2 = BSplineKit.extrapolate(BSplineKit.fit(BSplineKit.BSplineOrder(4),xs,ys, 0.005, BSplineKit.Natural(); weights=1 ./ δys.^2),BSplineKit.Smooth())
    z2_final_fit[i,:] = splF2.(i_xx0)

end
plot!(fig, QM_df[!,:Ic], QM_df[!,:F1], label="QM", line=(:black,:dashdot,2));
display(fig)
plot!(fig,
title = "Fit smoothing cubic spline",
xaxis=:log10, 
yaxis=:log10,
xticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlims = (10e-3,1.0),
ylims = (8e-3, 2),
)
display(fig)
zf1_fit = vec(mean(z1_final_fit, dims=1))
δzf1_fit = vec(std(z1_final_fit; dims=1, corrected=true)/sqrt(n_data))
zf2_fit = vec(mean(z2_final_fit, dims=1))
δzf2_fit = vec(std(z2_final_fit; dims=1, corrected=true)/sqrt(n_data))
data_fit = hcat(i_xx0, zf1_fit, δzf1_fit, zf2_fit, δzf2_fit)
plot!(fig,i_xx0, zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:goldenrod, 
    label=false,
    line=(:dash,:goldenrod,2))
display(fig)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Combined results

combined_result = OrderedDict(
    :Current => (
        Ic  = grouped_exp_current.Ic,
        σIc = grouped_exp_current.δIc,
    ),

    :MonteCarlo => (
        zF1      = MC_data_F1.μ,
        σzF1     = MC_data_F1.σ_tot,
        zF2      = MC_data_F2.μ,
        σzF2     = MC_data_F2.σ_tot,
        Δz       = MC_data_F1.μ .- MC_data_F2.μ,
        σΔz      = sqrt.( (MC_data_F1.σ_tot).^2 .+ (MC_data_F2.σ_tot).^2 )
    ),

    :SplineInter => (
        zF1      = data_spl[:,2],
        σzF1     = data_spl[:,3],
        zF2      = data_spl[:,4],
        σzF2     = data_spl[:,5],
        Δz       = data_spl[:,2] .- data_spl[:,4],
        σΔz      = sqrt.( (data_spl[:,3]).^2 .+ (data_spl[:,5]).^2 )
    ),

    :SplineFit => (
        zF1      = data_fit[:,2],
        σzF1     = data_fit[:,3],
        zF2      = data_fit[:,4],
        σzF2     = data_fit[:,5],
        Δz       = data_fit[:,2] .- data_fit[:,4],
        σΔz      = sqrt.( (data_fit[:,3]).^2 .+ (data_fit[:,5]).^2 )
    ),

)

fig1 = plot(xlabel="Currents (A)", 
    ylabel=L"z_{\mathrm{peak}}^{F=1} \qquad (\mathrm{mm})")
plot!(fig1, combined_result[:Current].Ic, combined_result[:MonteCarlo].zF1,
    ribbon= combined_result[:MonteCarlo].σzF1,
    label= "Monte Carlo",
    color=:red )
plot!(fig1,
    combined_result[:Current].Ic, combined_result[:SplineInter].zF1 ,
    ribbon= combined_result[:SplineInter].σzF1,
    label="Spl. Interpolation",
    color=:blue)
plot!(fig1,
    combined_result[:Current].Ic, combined_result[:SplineFit].zF1 ,
    ribbon= combined_result[:SplineFit].σzF1,
    label="Spl. Fit",
    color=:darkgreen )
plot!(fig1,
    xlims=(10e-3,1.05),
    ylims=(10e-3,2.05),
    xscale=:log10,yscale=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),)

fig2 = plot(xlabel="Currents (A)", ylabel=L"z_{\mathrm{peak}}^{F=2}  \qquad (\mathrm{mm})")    
plot!(fig2,
    combined_result[:Current].Ic, combined_result[:MonteCarlo].zF2,
    ribbon= combined_result[:MonteCarlo].σzF2,
    label= "Monte Carlo",
    color=:red )
plot!(fig2,
    combined_result[:Current].Ic, combined_result[:SplineInter].zF2 ,
    ribbon= combined_result[:SplineInter].σzF2,
    label="Spl. Interpolation",
    color=:blue)
plot!(fig2,
    combined_result[:Current].Ic, combined_result[:SplineFit].zF2 ,
    ribbon= combined_result[:SplineFit].σzF2,
    label="Spl. Fit",
    color=:darkgreen )
plot!(fig2,
    legend=:bottomleft,
    xlims=(10e-3,1.05),
    xticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xscale=:log10)

fig3 = plot(xlabel="Currents (A)", ylabel=L"z_{\mathrm{peak-peak}} \qquad (\mathrm{mm})")    
plot!(fig3,
    combined_result[:Current].Ic, combined_result[:MonteCarlo].Δz ,
    ribbon= combined_result[:MonteCarlo].σΔz,
    label= "Monte Carlo",
    color=:red )
plot!(fig3,
    combined_result[:Current].Ic, combined_result[:SplineInter].Δz ,
    ribbon= combined_result[:SplineInter].σΔz,
    label="Spl. Interpolation",
    color=:blue)
plot!(fig3,
    combined_result[:Current].Ic, combined_result[:SplineFit].Δz ,
    ribbon= combined_result[:SplineFit].σΔz,
    label="Spl. Fit",
    color=:darkgreen )
plot!(fig3,
    legend=:topleft,
    xlims=(10e-3,1.05),
    ylims=(10e-3,4.05),
    xscale=:log10,yscale=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-2, 1e-1, 1.0], [ L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)

fig = plot(fig1, fig2, fig3,
    layout=(3,1),
    link=:x,
    size=(800,600),
    left_margin=4mm,)
plot!(fig[1], xlabel="", xformatter=_->"", bottom_margin=-9mm)
plot!(fig[2], xlabel="", xformatter=_->"", bottom_margin=-9mm)
display(fig)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Select a subset of kᵢ values for interpolation
using Dierckx
# ki_start , ki_stop = 1 , 109 ;
ki_start , ki_stop = 12 , 60 ;
println("Interpolation in the induction term goes from ",
    round(cqd_sim_data.ki[ki_start]/1e6; sigdigits=3),
    " to ",
    (round(cqd_sim_data.ki[ki_stop]/1e6, sigdigits=3)),
    "")
# Build 2D cubic spline interpolant: z_max(I, kᵢ)
# s=0 => exact interpolation (no smoothing)
ki_up_itp = Spline2D(Icoils, cqd_sim_data.ki[ki_start:ki_stop], up_cqd[:,ki_start:ki_stop]; kx=3, ky=3, s=0.00);
ki_dw_itp = Spline2D(Icoils, cqd_sim_data.ki[ki_start:ki_stop], dw_cqd[:,ki_start:ki_stop]; kx=3, ky=3, s=0.00);
ki_Δ_itp  = Spline2D(Icoils, cqd_sim_data.ki[ki_start:ki_stop], Δz_cqd[:,ki_start:ki_stop]; kx=3, ky=3, s=0.00);

# -----------------------------------------------------------------------------
# Create a dense grid for visualization:
#   - currents from 10 mA to 1 A
#   - ki from chosen min to max
# -----------------------------------------------------------------------------
i_surface = range(10e-3,1.0; length = 101);
ki_surface = range(cqd_sim_data.ki[ki_start],cqd_sim_data.ki[ki_stop]; length = 101);
# Evaluate surface on a grid.
Zup     = [ki_up_itp(x, y) for y in ki_surface, x in i_surface] ;
Zdw     = [ki_dw_itp(x, y) for y in ki_surface, x in i_surface] ;
ΔZ_cqd  = [ki_Δ_itp(x, y) for y in ki_surface, x in i_surface] ;

# -----------------------------------------------------------------------------
# 3D surface plot (log10 axes for I and z)
# -----------------------------------------------------------------------------
# fit_surface = surface(log10.(i_surface), ki_surface, log10.(abs.(Z));
#     title = "Fitting surface",
#     xlabel = L"I_{c}",
#     ylabel = L"$k_{i}\times 10^{-6}$",
#     zlabel = L"$z\ (\mathrm{mm})$",
#     legend = false,
#     color = :viridis,
#     xticks = (log10.([1e-3, 1e-2, 1e-1, 1.0]), [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
#     zticks = (log10.([1e-3, 1e-2, 1e-1, 1.0, 10.0]), [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
#     camera = (20, 25),     # (azimuth, elevation)
#     xlims = log10.((8e-4,2.05)),
#     zlims = log10.((2e-4,10.0)),
#     gridalpha = 0.3,
# );

# -----------------------------------------------------------------------------
# Contour plot uses log10(z) as the displayed quantity.
# We clamp |Z| away from zero to avoid log10(0) and produce stable color limits.
# -----------------------------------------------------------------------------
Zp   = max.(Zup, 1e-12);
logZ = log10.(Zp);
# Choose "decade" ticks for the colorbar based on min/max of logZ
lo , hi  = floor(minimum(logZ)) , ceil(maximum(logZ)); 
decades = collect(lo:1:hi) ; # [-4,-3,-2,-1,0] 
labels = [L"10^{%$k}" for k in decades];
fit_contour_up = contourf(i_surface, ki_surface, logZ; 
    levels=101,
    title="Fitting contour",
    xlabel=L"$I_{c}$ (A)", 
    ylabel=L"$k_{i}\times 10^{-6}$", 
    color=:viridis, 
    linewidth=0.2,
    linestyle=:dash,
    xaxis=:log10,
    yaxis=:log10,
    xlims = (10e-3,1.05),
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    clims = (lo, hi),   # optional explicit range
    colorbar_ticks = (decades, labels),      # show ticks as 10^k
    colorbar_title = L"$ \log(z_{\mathrm{peak}}^{\mathrm{up}}\ \mathrm{(mm)}) $",   # what the values mean
);
display(fit_contour_up)

Zp   = Zdw;
# Choose "decade" ticks for the colorbar based on min/max of logZ
lo , hi  = floor(minimum(Zp)) , ceil(maximum(Zp))
decades = collect(lo:1:hi) ; # [-4,-3,-2,-1,0] 
labels = [L"10^{%$k}" for k in decades];
fit_contour_dw = contourf(i_surface, ki_surface, Zp; 
    levels=101,
    title="Fitting contour",
    xlabel=L"$I_{c}$ (A)", 
    ylabel=L"$k_{i}\times 10^{-6}$", 
    color= cgrad(:viridis, rev=true), 
    linewidth=0.2,
    linestyle=:dash,
    xaxis=:log10,
    yaxis=:log10,
    xlims = (10e-3,1.05),
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    clims = (lo, hi),   # optional explicit range
    colorbar_ticks = (decades, labels),      # show ticks as 10^k
    colorbar_title = L"$ z_{\mathrm{peak}}^{\mathrm{dw}} \ \mathrm{(mm)}$",   # what the values mean
);
display(fit_contour_dw)

Zp   = max.(ΔZ_cqd, 1e-12);
logZ = log10.(Zp);
# Choose "decade" ticks for the colorbar based on min/max of logZ
lo , hi  = floor(minimum(logZ)) , ceil(maximum(logZ)); 
decades = collect(lo:1:hi) ; # [-4,-3,-2,-1,0] 
labels = [L"10^{%$k}" for k in decades];
fit_contour_Δ = contourf(i_surface, ki_surface, logZ; 
    levels=101,
    title="Fitting contour",
    xlabel=L"$I_{c}$ (A)", 
    ylabel=L"$k_{i}\times 10^{-6}$", 
    color=:viridis, 
    linewidth=0.2,
    linestyle=:dash,
    xaxis=:log10,
    yaxis=:log10,
    xlims = (10e-3,1.05),
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    clims = (lo, hi),   # optional explicit range
    colorbar_ticks = (decades, labels),      # show ticks as 10^k
    colorbar_title = L"$ \log(z_{\mathrm{p-p}}\ \mathrm{(mm)}) $",   # what the values mean
);
display(fit_contour_Δ )

plot(fit_contour_up, fit_contour_dw, fit_contour_Δ,
    layout=@layout([a1 a2; a3]);
    title="",
    size=(900,500),
    left_margin=4mm,
)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Currents used for scan/plotting of fitted curves (log-spaced)
Iscan = logspace10(0.020, 1.00; n = 101);
QM_itp_zF1 = Spline1D(QM_df[!,:Ic],QM_df[!,:F1],k=3);

combined_method = :SplineInter ; 

data_exp = DataFrame(
    Ic  = combined_result[:Current].Ic, 
    σIc = combined_result[:Current].σIc, 
    F1  = combined_result[combined_method].zF1, 
    σF1 = combined_result[combined_method].σzF1,
    F2  = combined_result[combined_method].zF2, 
    σF2 = combined_result[combined_method].σzF2
)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FITTING : QUANTUM MECHANICS vs EXPERIMENTAL DATA

"""
    fit_QM_scale_model(x, y, model;
        σy=nothing,
        mask=nothing,
        idx=nothing,
        xmin=nothing,
        xmax=nothing,
        offset::Bool=false,
        fitspace::Symbol=:log10,
        project::Symbol=:model_to_y)

Fit a scaled model to data, with optional offset, subset selection, and
choice of fitting space.

The model is first evaluated on the experimental x-grid:

    z = model.(x)

and then one of the following relations is fit.

Zero-offset modes
-----------------
If `offset=false`:

- `project=:model_to_y`
    Fits
        y ≈ α z
- `project=:y_to_model`
    Fits
        z ≈ α y

For zero offset, both `fitspace=:log10` and `fitspace=:linear` are supported.

Offset modes
------------
If `offset=true`:

- `project=:model_to_y`
    Fits
        y ≈ α z + β
- `project=:y_to_model`
    Fits
        z ≈ α y + β

For nonzero offset, only `fitspace=:linear` is supported.

Subset selection
----------------
You may restrict the fit using any combination of:
- `mask` : boolean mask of same length as `x`
- `idx`  : vector/range of selected indices
- `xmin` : keep only `x >= xmin`
- `xmax` : keep only `x <= xmax`

Important note on uncertainties
-------------------------------
The weighted fit is statistically meaningful only when the supplied uncertainty
matches the residual being minimized.

Therefore:

- `project=:model_to_y` naturally uses `σy`
- `project=:y_to_model` should use `σy = nothing`

In other words, for `project=:y_to_model`, this function enforces `σy === nothing`.

Arguments
---------
- `x`     : experimental x-values
- `y`     : experimental y-values
- `model` : callable object so that `model.(x)` returns model values on the x-grid

Keyword arguments
-----------------
- `σy`       : uncertainty on `y`; only allowed for `project=:model_to_y`
- `mask`     : boolean mask for selecting points
- `idx`      : indices for selecting points
- `xmin`     : minimum x to include
- `xmax`     : maximum x to include
- `offset`   : if `false`, fit only a scale factor; if `true`, fit scale + offset
- `fitspace` : `:log10` or `:linear`
- `project`  : `:model_to_y` or `:y_to_model`

Returns
-------
A named tuple containing the fitted parameters and diagnostics. Depending on the
mode, this may include:
- `α`, `β`
- `σα`, `σβ`
- `used_mask`
- `x_used`, `y_used`, `z_used`
- residuals in linear and/or log space
- `rss_linear`, `rss_log`
- `χ2`, `χ2red`, `χ2_log`, `χ2red_log`
- `dof`

Examples
--------
Zero-offset log-space fit, model projected onto data:

julia
Zero-offset linear fit, data projected onto model:
fit = fit_QM_scale_model(Ic, F1, zqm;
    σy=σF1,
    offset=false,
    fitspace=:log10,
    project=:model_to_y)

Linear fit with offset:
fit = fit_QM_scale_model(Ic, F1, zqm;
    σy=σF1,
    offset=true,
    fitspace=:linear,
    project=:model_to_y)
"""
function fit_QM_scale_model(x, y, model;
    σy=nothing,
    mask=nothing,
    idx=nothing,
    xmin=nothing,
    xmax=nothing,
    offset::Bool=false,
    fitspace::Symbol=:log10,              # :log10 or :linear
    project::Symbol=:model_to_y,        # :model_to_y or :y_to_model
    )

    @assert fitspace in (:log10, :linear) "fitspace must be :log10 or :linear"
    @assert project in (:model_to_y, :y_to_model) "project must be :model_to_y or :y_to_model"

    # IMPORTANT:
    # If project = :y_to_model, the residual is defined in z-space,
    # not in y-space. Therefore σy must be nothing.
    @assert !(project == :y_to_model && σy !== nothing) "project=:y_to_model must use σy = nothing"

    z = model.(x)

    # -----------------------------
    # Build selection mask
    # -----------------------------
    sel = trues(length(x))

    if mask !== nothing
        @assert length(mask) == length(x)
        sel .&= mask
    end

    if idx !== nothing
        tmp = falses(length(x))
        tmp[idx] .= true
        sel .&= tmp
    end

    if xmin !== nothing
        sel .&= x .>= xmin
    end

    if xmax !== nothing
        sel .&= x .<= xmax
    end

    # ============================================================
    # ZERO OFFSET: y ≈ αz   or   z ≈ αy
    # ============================================================
    if !offset

        # =========================
        # LOG-SPACE
        # =========================
        if fitspace == :log10

            sel_fit = sel .& (y .> 0) .& (z .> 0)
            if σy !== nothing
                sel_fit .&= (σy .> 0)
            end

            xx = x[sel_fit]
            yy = y[sel_fit]
            zz = z[sel_fit]

            @assert length(xx) > 1

            if project == :model_to_y
                Δ = log10.(yy) .- log10.(zz)

                if σy === nothing
                    logα = mean(Δ)
                    α = 10^(logα)
                    σα = nothing
                else
                    ss = σy[sel_fit]
                    w = yy.^2 ./ ss.^2
                    logα = sum(w .* Δ) / sum(w)
                    σα = 10^logα * sqrt(1 / sum(w))
                    α = 10^logα
                end

                pred = α .* zz
                rlog = log10.(yy) .- log10.(pred)
                rlin = yy .- pred

            else
                Δ = log10.(zz) .- log10.(yy)
                logα = mean(Δ)
                α = 10^(logα)
                σα = nothing

                pred = α .* yy
                rlog = log10.(zz) .- log10.(pred)
                rlin = zz .- pred
            end

            return (
                α=α, β=0.0, σα=σα, σβ=nothing,
                mode=:zero_offset_log10,
                fitspace=:log10,
                project=project,
                used_mask=sel_fit,
                x_used=xx, y_used=yy, z_used=zz,
                residuals_log=rlog,
                residuals_linear=rlin,
                rss_log=sum(rlog.^2),
                rss_linear=sum(rlin.^2),
                dof=length(yy)-1
            )

        # =========================
        # LINEAR SPACE
        # =========================
        else
            xx = x[sel]
            yy = y[sel]
            zz = z[sel]

            @assert length(xx) > 1

            if project == :model_to_y
                if σy === nothing
                    α = dot(yy, zz) / dot(zz, zz)
                    σα = nothing
                else
                    ss = σy[sel]
                    w = 1.0 ./ ss.^2
                    α = sum(w .* yy .* zz) / sum(w .* zz.^2)
                    σα = sqrt(1 / sum(w .* zz.^2))
                end

                pred = α .* zz
                rlin = yy .- pred

            else
                α = dot(zz, yy) / dot(yy, yy)
                σα = nothing

                pred = α .* yy
                rlin = zz .- pred
            end

            return (
                α=α, β=0.0, σα=σα, σβ=nothing,
                mode=:zero_offset_linear,
                fitspace=:linear,
                project=project,
                used_mask=sel,
                x_used=xx, y_used=yy, z_used=zz,
                residuals_linear=rlin,
                rss_linear=sum(rlin.^2),
                dof=length(yy)-1
            )
        end

    # ============================================================
    # OFFSET: y ≈ αz + β  or  z ≈ αy + β
    # ============================================================
    else
        @assert fitspace == :linear "offset=true only supports linear fit"

        xx = x[sel]
        yy = y[sel]
        zz = z[sel]

        @assert length(xx) > 2

        if project == :model_to_y
            A = hcat(zz, ones(length(zz)))
            target = yy
        else
            A = hcat(yy, ones(length(yy)))
            target = zz
        end

        if σy === nothing
            p = A \ target
            α, β = p
            pred = A * p
            r = target .- pred

            return (
                α=α, β=β,
                σα=nothing, σβ=nothing,
                mode=:linear_offset,
                fitspace=:linear,
                project=project,
                used_mask=sel,
                x_used=xx, y_used=yy, z_used=zz,
                residuals_linear=r,
                rss_linear=sum(r.^2),
                dof=length(target)-2
            )
        else
            ss = σy[sel]
            w = 1.0 ./ ss.^2
            W = Diagonal(w)

            ATA = A' * W * A
            ATb = A' * W * target

            p = ATA \ ATb
            α, β = p

            Cov = inv(ATA)
            σα = sqrt(Cov[1,1])
            σβ = sqrt(Cov[2,2])

            pred = A * p
            r = target .- pred

            return (
                α=α, β=β,
                σα=σα, σβ=σβ,
                mode=:linear_offset,
                fitspace=:linear,
                project=project,
                used_mask=sel,
                x_used=xx, y_used=yy, z_used=zz,
                residuals_linear=r,
                χ2=sum(w .* r.^2),
                χ2red=sum(w .* r.^2)/(length(target)-2),
                rss_linear=sum(r.^2),
                dof=length(target)-2
            )
        end
    end
end

# fit_idx = vcat(28:30, 32:33)
# fit_idx = vcat(31:33,36)
# fit_idx = vcat(27:30,33)
fit_idx = vcat(21:26)
Ic, yexp, σy = data_exp.Ic, data_exp.F1, data_exp.σF1
yqm = QM_itp_zF1.(Ic)

fitQM = fit_QM_scale_model(
    Ic,
    yexp,
    QM_itp_zF1;
    offset = false,
    fitspace = :log10,
    σy = σy,
    idx = fit_idx,
    project = :model_to_y,
)

α, σα, β = fitQM.α, fitQM.σα, fitQM.β;
ymod = α .* yqm .+ β;
relerr1 = 100 .* (yexp .- yqm) ./ yqm;
relerr2 = 100 .* (yexp .- ymod) ./ ymod;

@info @sprintf("Fitting parameters F1: Experiment = (%.3f ± %.3f ) * QM + %.3f", α, σα, β)
pretty_table(
    hcat(Ic, yexp, yqm, relerr1, ymod, relerr2);
    column_labels = [
        ["Current", "F1 exp", "F1 QM", "Rel. Error", "F1 model QM", "Rel. Error"],
        ["[A]", "[mm]", "[mm]", "[%]", "[mm]", "[%]"]
    ],
    alignment=:c,
    row_label_column_alignment=:c,
    row_group_label_alignment=:c,
    title="QUANTUM MECHANICS",
    formatters=[
        fmt__printf("%8.3f", [1]),
        fmt__printf("%8.4f", [2,3,5]),
        fmt__printf("%8.1f", [4,6])
    ],
    style=TextTableStyle(
        first_line_column_label=crayon"yellow bold",
        table_border=crayon"blue bold",
        title=crayon"bold red"
    ),
    table_format=TextTableFormat(borders=text_table_borders__unicode_rounded),
    equal_data_column_widths=true,
)

scatter(Ic, yexp; yerror=σy, label="Experimental data")
scatter!(Ic[fitQM.used_mask], yexp[fitQM.used_mask]; label="Used in fit", marker=:diamond)
plot!(Ic, yqm; lw=2, label="QM")
plot!(Ic, ymod; lw=2, label="α × QM + β")
plot!(xlabel="Current (A)", ylabel=L"$z^{F=1}_{\mathrm{peak}}$ (mm)")


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FITTING : COQUANTUM DYNAMICS vs EXPERIMENTAL DATA

function fit_scale_and_k(x, y, model;
    bounds::Tuple,
    σy=nothing,
    mask=nothing,
    idx_k=nothing,
    idx_alpha=nothing,
    xmin=nothing,
    xmax=nothing,
    offset::Bool=false,
    fitspace::Symbol=:log10,          # :log10 or :linear
    project::Symbol=:model_to_y,      # :model_to_y or :y_to_model
    fit_alpha::Bool=true,
)

    @assert fitspace in (:log10, :linear) "fitspace must be :log10 or :linear"
    @assert project in (:model_to_y, :y_to_model) "project must be :model_to_y or :y_to_model"
    @assert !(project == :y_to_model && σy !== nothing) "project=:y_to_model must use σy = nothing"
    @assert !(offset && fitspace != :linear) "offset=true only supports linear fit"

    n = length(x)
    @assert length(y) == n
    if σy !== nothing
        @assert length(σy) == n
    end
    if mask !== nothing
        @assert length(mask) == n
    end

    # --------------------------------------------------------
    # Base selection mask, same spirit as fit_QM_scale_model
    # --------------------------------------------------------
    sel = trues(n)

    if mask !== nothing
        sel .&= mask
    end
    if xmin !== nothing
        sel .&= x .>= xmin
    end
    if xmax !== nothing
        sel .&= x .<= xmax
    end

    # indices used for k-objective
    sel_k = copy(sel)
    if idx_k !== nothing
        tmp = falses(n)
        tmp[idx_k] .= true
        sel_k .&= tmp
    end

    # indices used for alpha/beta estimation
    sel_a = copy(sel)
    if idx_alpha !== nothing
        tmp = falses(n)
        tmp[idx_alpha] .= true
        sel_a .&= tmp
    else
        sel_a .= sel_k
    end

    @assert count(sel_k) > 1 "Need at least 2 points for k-fit selection."
    @assert count(sel_a) > 0 "Need at least 1 point for alpha selection."

    xk = x[sel_k]
    yk = y[sel_k]
    σk = σy === nothing ? nothing : σy[sel_k]

    xa = x[sel_a]
    ya = y[sel_a]
    σa = σy === nothing ? nothing : σy[sel_a]

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    z_of_k_all(k) = model.(x, Ref(k))
    z_of_k_k(k)   = model.(xk, Ref(k))
    z_of_k_a(k)   = model.(xa, Ref(k))

    function solve_alpha_beta_linear(k)
        za = z_of_k_a(k)

        if project == :model_to_y
            target = ya
            basis  = za
            ssa    = σa
        else
            target = za
            basis  = ya
            ssa    = nothing
        end

        if !offset
            if !fit_alpha
                α = 1.0
                β = 0.0
                σα = nothing
                σβ = nothing
            else
                if ssa === nothing
                    α = dot(target, basis) / dot(basis, basis)
                    σα = nothing
                else
                    w = 1.0 ./ ssa.^2
                    α = sum(w .* target .* basis) / sum(w .* basis.^2)
                    σα = sqrt(1 / sum(w .* basis.^2))
                end
                β = 0.0
                σβ = nothing
            end
            return (α=α, β=β, σα=σα, σβ=σβ)
        end

        # offset=true, linear only
        if project == :model_to_y
            A = hcat(za, ones(length(za)))
        else
            A = hcat(ya, ones(length(ya)))
        end

        if !fit_alpha
            # α fixed to 1, fit only β
            if project == :model_to_y
                resid0 = ya .- za
            else
                resid0 = za .- ya
            end

            if ssa === nothing
                β = mean(resid0)
                σβ = nothing
            else
                w = 1.0 ./ ssa.^2
                β = sum(w .* resid0) / sum(w)
                σβ = sqrt(1 / sum(w))
            end
            return (α=1.0, β=β, σα=nothing, σβ=σβ)
        end

        if ssa === nothing
            p = A \ target
            α, β = p
            return (α=α, β=β, σα=nothing, σβ=nothing)
        else
            w = 1.0 ./ ssa.^2
            W = Diagonal(w)
            ATA = A' * W * A
            ATb = A' * W * target
            p = ATA \ ATb
            α, β = p
            Cov = inv(ATA)
            σα = sqrt(Cov[1,1])
            σβ = sqrt(Cov[2,2])
            return (α=α, β=β, σα=σα, σβ=σβ)
        end
    end

    function solve_alpha_log(k)
        @assert !offset "offset=true is not supported in log fit"
        za = z_of_k_a(k)

        if project == :model_to_y
            valid = (ya .> 0) .& (za .> 0)
            if σa !== nothing
                valid .&= (σa .> 0)
            end
            @assert count(valid) > 1 "Need >1 positive points for log fit."
            yy = ya[valid]
            zz = za[valid]

            if !fit_alpha
                α = 1.0
                σα = nothing
            else
                Δ = log10.(yy) .- log10.(zz)
                if σa === nothing
                    logα = mean(Δ)
                    α = 10.0^logα
                    σα = nothing
                else
                    ss = σa[valid]
                    w = yy.^2 ./ ss.^2
                    logα = sum(w .* Δ) / sum(w)
                    α = 10.0^logα
                    σα = 10.0^logα * sqrt(1 / sum(w))
                end
            end
            return (α=α, β=0.0, σα=σα, σβ=nothing, valid=valid)
        else
            valid = (ya .> 0) .& (za .> 0)
            @assert count(valid) > 1 "Need >1 positive points for log fit."
            yy = ya[valid]
            zz = za[valid]

            if !fit_alpha
                α = 1.0
            else
                Δ = log10.(zz) .- log10.(yy)
                logα = mean(Δ)
                α = 10.0^logα
            end
            return (α=α, β=0.0, σα=nothing, σβ=nothing, valid=valid)
        end
    end

    function objective(k)
        if fitspace == :linear
            pars = solve_alpha_beta_linear(k)
            α, β = pars.α, pars.β
            zk = z_of_k_k(k)

            if project == :model_to_y
                pred = α .* zk .+ β
                r = yk .- pred
                if σk === nothing
                    return sum(r.^2)
                else
                    w = 1.0 ./ σk.^2
                    return sum(w .* r.^2)
                end
            else
                pred = α .* yk .+ β
                r = zk .- pred
                return sum(r.^2)
            end

        else
            pars = solve_alpha_log(k)
            α = pars.α
            zk = z_of_k_k(k)

            if project == :model_to_y
                valid = (yk .> 0) .& (zk .> 0)
                if σk !== nothing
                    valid .&= (σk .> 0)
                end
                if count(valid) <= 1
                    return Inf
                end
                yy = yk[valid]
                zz = zk[valid]
                pred = α .* zz
                r = log10.(yy) .- log10.(pred)

                if σk === nothing
                    return sum(r.^2)
                else
                    ss = σk[valid]
                    w = yy.^2 ./ ss.^2
                    return sum(w .* r.^2)
                end
            else
                valid = (yk .> 0) .& (zk .> 0)
                if count(valid) <= 1
                    return Inf
                end
                yy = yk[valid]
                zz = zk[valid]
                pred = α .* yy
                r = log10.(zz) .- log10.(pred)
                return sum(r.^2)
            end
        end
    end

    # --------------------------------------------------------
    # Optimize k
    # --------------------------------------------------------
    res = optimize(objective, bounds[1], bounds[2], Brent())
    kbest = Optim.minimizer(res)

    # --------------------------------------------------------
    # Final parameters at optimum
    # --------------------------------------------------------
    if fitspace == :linear
        pars = solve_alpha_beta_linear(kbest)
        αbest, βbest = pars.α, pars.β
        σα, σβ = pars.σα, pars.σβ
        zall = z_of_k_all(kbest)

        if project == :model_to_y
            pred_all = αbest .* zall .+ βbest
            rlin_all = y .- pred_all

            used_mask = sel
            xx = x[used_mask]
            yy = y[used_mask]
            zz = zall[used_mask]
            pred_used = αbest .* zz .+ βbest
            rlin = yy .- pred_used

            out = (
                k=kbest,
                α=αbest, β=βbest, σα=σα, σβ=σβ,
                mode=offset ? :linear_offset_k : :zero_offset_linear_k,
                fitspace=:linear,
                project=project,
                used_mask=used_mask,
                used_mask_k=sel_k,
                used_mask_alpha=sel_a,
                x_used=xx, y_used=yy, z_used=zz,
                y_fit=pred_all,
                residuals_linear=rlin,
                residuals_linear_all=rlin_all,
                rss_linear=sum(rlin.^2),
                dof=length(yy) - (offset ? 3 : (fit_alpha ? 2 : 1)),
                optimizer=res,
            )

            if σy !== nothing
                ss = σy[used_mask]
                w = 1.0 ./ ss.^2
                χ2 = sum(w .* rlin.^2)
                dof = length(yy) - (offset ? 3 : (fit_alpha ? 2 : 1))
                out = merge(out, (
                    χ2=χ2,
                    χ2red=χ2 / dof,
                ))
            end
            return out
        else
            pred_all = αbest .* y .+ βbest
            rlin_all = zall .- pred_all

            used_mask = sel
            xx = x[used_mask]
            yy = y[used_mask]
            zz = zall[used_mask]
            pred_used = αbest .* yy .+ βbest
            rlin = zz .- pred_used

            return (
                k=kbest,
                α=αbest, β=βbest, σα=σα, σβ=σβ,
                mode=offset ? :linear_offset_k : :zero_offset_linear_k,
                fitspace=:linear,
                project=project,
                used_mask=used_mask,
                used_mask_k=sel_k,
                used_mask_alpha=sel_a,
                x_used=xx, y_used=yy, z_used=zz,
                z_fit=pred_all,
                residuals_linear=rlin,
                residuals_linear_all=rlin_all,
                rss_linear=sum(rlin.^2),
                dof=length(yy) - (offset ? 3 : (fit_alpha ? 2 : 1)),
                optimizer=res,
            )
        end

    else
        pars = solve_alpha_log(kbest)
        αbest = pars.α
        βbest = 0.0
        σα, σβ = pars.σα, pars.σβ
        zall = z_of_k_all(kbest)

        if project == :model_to_y
            sel_fit = sel .& (y .> 0) .& (zall .> 0)
            if σy !== nothing
                sel_fit .&= (σy .> 0)
            end

            xx = x[sel_fit]
            yy = y[sel_fit]
            zz = zall[sel_fit]

            pred = αbest .* zz
            rlog = log10.(yy) .- log10.(pred)
            rlin = yy .- pred

            return (
                k=kbest,
                α=αbest, β=0.0, σα=σα, σβ=σβ,
                mode=:zero_offset_log10_k,
                fitspace=:log10,
                project=project,
                used_mask=sel_fit,
                used_mask_k=sel_k,
                used_mask_alpha=sel_a,
                x_used=xx, y_used=yy, z_used=zz,
                y_fit=αbest .* zall,
                residuals_log=rlog,
                residuals_linear=rlin,
                rss_log=sum(rlog.^2),
                rss_linear=sum(rlin.^2),
                dof=length(yy) - (fit_alpha ? 2 : 1),
                optimizer=res,
            )
        else
            sel_fit = sel .& (y .> 0) .& (zall .> 0)

            xx = x[sel_fit]
            yy = y[sel_fit]
            zz = zall[sel_fit]

            pred = αbest .* yy
            rlog = log10.(zz) .- log10.(pred)
            rlin = zz .- pred

            return (
                k=kbest,
                α=αbest, β=0.0, σα=σα, σβ=σβ,
                mode=:zero_offset_log10_k,
                fitspace=:log10,
                project=project,
                used_mask=sel_fit,
                used_mask_k=sel_k,
                used_mask_alpha=sel_a,
                x_used=xx, y_used=yy, z_used=zz,
                z_fit=αbest .* y,
                residuals_log=rlog,
                residuals_linear=rlin,
                rss_log=sum(rlog.^2),
                rss_linear=sum(rlin.^2),
                dof=length(yy) - (fit_alpha ? 2 : 1),
                optimizer=res,
            )
        end
    end
end

model = (I, k) -> ki_up_itp(I, k)
Ic, yexp, σy = data_exp.Ic, data_exp.F1, data_exp.σF1
kmin = minimum(cqd_sim_data.ki[ki_start:ki_stop])
kmax = maximum(cqd_sim_data.ki[ki_start:ki_stop])
idx_k = findall(data_exp.Ic .< 0.060)

fit = fit_scale_and_k(
    Ic,
    yexp,
    model,;
    σy = σy,
    bounds      = (kmin, kmax),
    idx_k       = idx_k,
    offset      = false,   
    fit_alpha   = true,
    idx_alpha   = nothing,
    fitspace    = :linear
)

kbest = fit.k
αbest = fit.α
βbest = fit.β

y_model = αbest .* model.(data_exp.Ic, Ref(kbest)) .+ βbest
relerr = 100 .* (yexp .- y_model) ./ y_model;

@info @sprintf("Fitting parameters F1: α = %.3f , β = %.3f,  ki = %.3f", αbest, βbest, kbest)
pretty_table(
    hcat(Ic, yexp, y_model, relerr);
    column_labels = [
        ["Current", "F1 exp", "UP CQD", "Rel. Error"],
        ["[A]", "[mm]", "[mm]", "[%]"]
    ],
    alignment=:c,
    row_label_column_alignment=:c,
    row_group_label_alignment=:c,
    title="COQUANTUM DYNAMICS",
    formatters=[
        fmt__printf("%8.3f", [1]),
        fmt__printf("%8.4f", [2,3,5]),
        fmt__printf("%8.1f", [4,6])
    ],
    style=TextTableStyle(
        first_line_column_label=crayon"yellow bold",
        table_border=crayon"blue bold",
        title=crayon"bold red"
    ),
    table_format=TextTableFormat(borders=text_table_borders__unicode_rounded),
    equal_data_column_widths=true,
)

fig = scatter(
    data_exp.Ic,
    data_exp.F1;
    yerror = data_exp.σF1,
    marker=(:circle,2,:white),
    markerstrokecolor=:red,
    label = "Experiment",
    xlabel = L"$I$ (A)",
    ylabel = L"$z_{\mathrm{max}}$ (mm)",
    legend = :topleft
)
plot!(
    fig,
    data_exp.Ic,
    y_model;
    label = L"Fit: $\alpha f(I,k)$",
    lw = 2,
    lc = :blue,
)

fig_log = scatter(
    data_exp.Ic,
    data_exp.F1;
    yerror = data_exp.σF1,
    marker=(:circle,2,:white),
    markerstrokecolor=:red,
    xscale = :log10,
    yscale = :log10,
    label  = "Experiment",
    xlabel = L"$I$ (A)",
    ylabel = L"$z_{\mathrm{max}}$ (mm)"
)
plot!(
    fig_log,
    data_exp.Ic,
    y_model;
    lw = 2,
    lc = :blue,
    label = L"Fit: $\alpha=%$(round(αbest, digits=3))$, $k_{i}=%$(round(kbest; digits=2))\times 10^{-6}$ "
)

plot(fig, fig_log,
    layout=(2,1))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function fit_ki_joint_scaling_fitsubset(
    data,
    zqm,
    ki_itp,
    thresholdI::Float64,
    ki_range;
    fit_ki_mode::Symbol = :low_high,     # :full | :low | :high | :low_high
    n_front::Int = n_front,
    n_back::Int  = n_back,
    w::Float64   = 0.5,
    ref_type::Symbol = :arith,           # :arith or :geom

    # --- uncertainty knobs ---
    conf::Real = 0.95,
    use_Zse::Bool = false,
    profile::Bool = true,
    profile_grid::Int = 400,)

    # --- Unpack experimental current and z_max ---
    Iexp = collect(Float64, data[:, 1])
    yexp = collect(Float64, data[:, 3])
    σy   = collect(Float64, data[:, 4])   # used only if use_Zse=true

    # base validity (log requires y>0; σy>0 if used)
    mbase = isfinite.(Iexp) .& isfinite.(yexp) .& (yexp .> 0)
    if use_Zse
        mbase .&= isfinite.(σy) .& (σy .> 0)
    end
    Iexp, yexp, σy = Iexp[mbase], yexp[mbase], σy[mbase]
    N = length(Iexp)
    @assert N ≥ 2 "Not enough valid data points."

    # ------------------------------
    # 1) Tail region (for scaling)
    # ------------------------------
    tail_mask = Iexp .>= thresholdI
    n_tail = count(tail_mask)
    n_tail > 0 || error("No experimental points with current ≥ $thresholdI A")

    I_tail = Iexp[tail_mask]
    y_tail = yexp[tail_mask]

    # -----------------------------------------
    # 2) Fitting region (subset for ki optimization)
    # -----------------------------------------
    low_range  = 1:min(n_front, N)
    high_range = max(1, N - n_back + 1):N

    fit_idx = if fit_ki_mode === :full
        collect(1:N)
    elseif fit_ki_mode === :low
        collect(low_range)
    elseif fit_ki_mode === :high
        collect(high_range)
    elseif fit_ki_mode === :low_high
        vcat(collect(low_range), collect(high_range))
    else
        error("Invalid fit_ki_mode = $fit_ki_mode. Use :full, :low, :high, or :low_high.")
    end
    n_fit_idx = length(fit_idx)

    # weights in log10 space from σy:
    # Var(log10 y) ≈ (σy/(y ln 10))^2 => w = (y ln 10 / σy)^2
    log_weights(y, σ) = ((y .* log(10.0)) ./ σ) .^ 2

    # ------------------------------------------------
    # 3) Tail reference model z_ref(I; ki)
    # ------------------------------------------------
    function zref_tail_for(ki)
        zqm_tail  = zqm.(I_tail)
        zcqd_tail = ki_itp.(I_tail, ki)

        if ref_type === :arith
            zref = w .* zqm_tail .+ (1 - w) .* zcqd_tail
        elseif ref_type === :geom
            # require positivity for fractional powers
            zref = zqm_tail .^ w .* zcqd_tail .^ (1 - w)
        else
            error("Invalid ref_type = $ref_type. Use :arith or :geom.")
        end
        zref
    end

    # ------------------------------------------------
    # 4) Scale factor s(ki) from tail projection
    # ------------------------------------------------
    function scale_for(ki)
        zref_tail = zref_tail_for(ki)
        m = isfinite.(zref_tail) .& isfinite.(y_tail) .& (zref_tail .!= 0)
        any(m) || return NaN
        denom = dot(zref_tail[m], zref_tail[m])
        denom == 0 && return NaN
        dot(y_tail[m], zref_tail[m]) / denom
    end

    # ------------------------------------------------
    # 5) Residual builder in log10 space on fit subset
    #     r_i = log10(yexp/scale) - log10(y_cqd)
    # ------------------------------------------------
    function residuals_for(ki)
        scale = scale_for(ki)
        if !isfinite(scale) || scale == 0
            return nothing
        end

        y_cqd = ki_itp.(Iexp, ki)

        # mask only selected indices + positivity for logs
        mfit = falses(N); mfit[fit_idx] .= true
        m = mfit .&
            isfinite.(y_cqd) .& (y_cqd .> 0) .&
            isfinite.(yexp) .& (yexp .> 0) .&
            (yexp ./ scale .> 0)

        any(m) || return nothing

        y_scaled = yexp[m] ./ scale
        r = log10.(y_scaled) .- log10.(y_cqd[m])

        ww = if use_Zse
            log_weights(yexp[m], σy[m])
        else
            ones(length(r))
        end

        return (r=r, w=ww, m=m, scale=scale)
    end

    # ------------------------------------------------
    # 6) Loss = weighted RSS in log10 space
    # ------------------------------------------------
    function loss(ki)
        out = residuals_for(ki)
        out === nothing && return Inf
        sum(out.w .* (out.r .^ 2))
    end

    # ------------------------------------------------
    # 7) Optimize ki (Brent)
    # ------------------------------------------------
    kmin, kmax = ki_range
    opt = optimize(loss, float(kmin), float(kmax), Brent())
    ki_fit = Optim.minimizer(opt)

    out0 = residuals_for(ki_fit)
    out0 === nothing && error("Unexpected: residuals invalid at optimum ki.")
    r0, w0, m0 = out0.r, out0.w, out0.m
    scale_final = out0.scale

    p = 1
    n_used0 = length(r0)
    @assert n_used0 > p "Not enough valid points to estimate uncertainty"

    RSS0 = sum(w0 .* (r0 .^ 2))
    mse0 = RSS0 / n_used0

    # ------------------------------------------------
    # 8) Curvature-based SE via finite-difference Jacobian dr/dki
    # ------------------------------------------------
    fd_step(k, lo, hi; rel=cbrt(eps(Float64)), absmin=1e-12) = begin
        hh = max(absmin, rel * max(abs(k), 1.0))
        room = min(k - lo, hi - k)
        room > 0 ? min(hh, 0.5 * room) : absmin
    end
    h₀ = fd_step(ki_fit, float(kmin), float(kmax))

    outp = residuals_for(ki_fit + h₀)
    outm = residuals_for(ki_fit - h₀)
    (outp === nothing || outm === nothing) &&
        error("Derivative evaluation failed near optimum; try widening bounds or check model positivity.")

    mJ = out0.m .& outp.m .& outm.m
    @assert count(mJ) > p "Not enough common points to compute derivative."

    function r_on_mask(ki, m)
        scale = scale_for(ki)
        y_cqd = ki_itp.(Iexp, ki)
        y_scaled = yexp[m] ./ scale
        log10.(y_scaled) .- log10.(y_cqd[m])
    end

    r_plus  = r_on_mask(ki_fit + h₀, mJ)
    r_minus = r_on_mask(ki_fit - h₀, mJ)
    drdk = (r_plus .- r_minus) ./ (2h₀)

    wJ = use_Zse ? log_weights(yexp[mJ], σy[mJ]) : ones(count(mJ))
    rJ = r_on_mask(ki_fit, mJ)

    RSS = sum(wJ .* (rJ .^ 2))
    dof = length(rJ) - p
    σ²  = RSS / dof
    SJJ = sum(wJ .* (drdk .^ 2))
    se  = sqrt(σ² / SJJ)

    tcrit = quantile(TDist(dof), 0.5 + conf/2)
    k_err = tcrit * se
    ci_t  = (ki_fit - k_err, ki_fit + k_err)

    # ------------------------------------------------
    # 9) R² in log10 space on mJ
    # ------------------------------------------------
    # Rebuild y_hat on mJ
    scale0 = scale_for(ki_fit)
    y_cqd0 = ki_itp.(Iexp, ki_fit)
    y_obs  = log10.(yexp[mJ] ./ scale0)
    y_hat  = log10.(y_cqd0[mJ])

    ȳw  = sum(wJ .* y_obs) / sum(wJ)
    TSS = sum(wJ .* (y_obs .- ȳw).^2)
    R2  = TSS > 0 ? 1 - RSS/TSS : NaN

    # ---------------------------------------------------------
    # 10) Profile interval with proper scaling
    #
    # If use_Zse=true: ΔRSS = χ²(1,conf)
    # If use_Zse=false: ΔRSS = σ²_hat * χ²(1,conf), with σ²_hat = RSS0/dof0
    # (this makes the threshold match the loss scale)
    # ---------------------------------------------------------
    ci_profile = nothing
    Δtarget = nothing
    Δrss = nothing
    profile_note = nothing

    if profile
        Δtarget = quantile(Chisq(1), conf)

        if use_Zse
            Δrss = Δtarget
            profile_note = nothing
        else
            profile_note = :profile_interval_scaled_for_unweighted
            dof0 = n_used0 - p
            σ²hat0 = RSS0 / dof0
            Δrss = σ²hat0 * Δtarget
        end

        target = RSS0 + Δrss

        function bracket_side(dir::Int)
            grid = range(ki_fit, dir > 0 ? float(kmax) : float(kmin); length=profile_grid)
            prevk = first(grid)
            prevL = loss(prevk)
            for k in Iterators.drop(grid, 1)
                L = loss(k)
                if isfinite(L) && (L > target) && isfinite(prevL) && (prevL <= target)
                    return (prevk, k)
                end
                prevk, prevL = k, L
            end
            return nothing
        end

        function bisect_cross(a, b; maxiter=80, tol=1e-10)
            lo, hi = a, b
            for _ in 1:maxiter
                mid = (lo + hi)/2
                fmid = loss(mid) - target
                if !isfinite(fmid)
                    hi = mid
                    continue
                end
                if fmid > 0
                    hi = mid
                else
                    lo = mid
                end
                if abs(hi - lo) <= tol*max(1.0, abs(mid))
                    return (lo + hi)/2
                end
            end
            return (lo + hi)/2
        end

        left_br  = bracket_side(-1)
        right_br = bracket_side(+1)

        k_lo = left_br  === nothing ? float(kmin) : bisect_cross(left_br[1], left_br[2])
        k_hi = right_br === nothing ? float(kmax) : bisect_cross(right_br[1], right_br[2])

        ci_profile = (k_lo, k_hi)
    end

    # ------------------------------------------------
    # 11) Extra diagnostics (linear-space error from log10 residuals)
    # ------------------------------------------------
    rmse_log10 = sqrt(mse0)
    mult_rmse  = 10.0 ^ rmse_log10

    frac_err = abs.(10.0 .^ r0 .- 1.0)
    med_abs_frac = median(frac_err)
    p90_abs_frac = quantile(frac_err, 0.90)
    p99_abs_frac = quantile(frac_err, 0.99)
    max_abs_frac = maximum(frac_err)
    n_bad_2pct   = count(>(0.02), frac_err)
    n_bad_5pct   = count(>(0.05), frac_err)

    scale_inv = 1 / scale_final
    scale_pct = (scale_inv - 1) * 100

    return (
        ki_fit       = ki_fit,
        scale_factor = scale_final,
        scale_inv    = scale_inv,
        scale_pct    = scale_pct,

        rss          = RSS0,
        mse          = mse0,
        rmse_log10   = rmse_log10,

        se           = se,
        k_err        = k_err,
        ci_t         = ci_t,

        ci_profile   = ci_profile,
        delta_target = Δtarget,
        delta_rss    = Δrss,
        profile_note = profile_note,

        R2           = R2,

        mult_rmse    = mult_rmse,
        med_abs_frac = med_abs_frac,
        p90_abs_frac = p90_abs_frac,
        p99_abs_frac = p99_abs_frac,
        max_abs_frac = max_abs_frac,
        n_bad_2pct   = n_bad_2pct,
        n_bad_5pct   = n_bad_5pct,

        n_total      = N,
        n_tail       = n_tail,
        n_fit_idx    = n_fit_idx,
        n_used       = length(rJ),

        converged    = Optim.converged(opt),
        result       = opt
    )
end

M_data_exp = Matrix(data_exp)[:,1:4]

result = fit_ki_joint_scaling_fitsubset(
    M_data_exp,
    QM_itp_zF1,
    ki_up_itp,
    0.750,                                  # tail threshold
    (cqd_sim_data[:ki][ki_start], cqd_sim_data[:ki][ki_stop]); # bracket
    fit_ki_mode=:full,
    n_front = 8,
    n_back  = 8,
    w       = 0.50,
    ref_type=:geom,
)

data_scaled = copy(M_data_exp)
data_scaled[:, 3] ./= result.scale_factor    # scale z
data_scaled[:, 4] ./= result.scale_factor    # scale δz

global_mag_factor = scale_mag 

data_fitting        = M_data_exp[fit_ki_idx, :]
data_scaled_fitting = data_scaled[fit_ki_idx, :]

fit_scaled = fit_ki(
    data_scaled,            # full dataset for R² (I,z)
    data_scaled_fitting,    # fitting subset for the loss
    cqd_sim_data[:ki],
    (ki_start, ki_stop),
)

fit_scaled = fit_ki_with_error(ki_up_itp, data_scaled_fitting; bounds=(cqd_sim_data[:ki][ki_start], cqd_sim_data[:ki][ki_stop]), conf=0.95, use_Zse=false)

fig=plot(    
    title = L"Peak position ($F=1$)",)
# Scaled experimental curve with ribbon
plot!(fig,
    data_scaled[:,1],data_scaled[:,3],
    yerror = data_scaled[:,4],
    label=L"Experimental data (magnif.factor $m = %$(round(global_mag_factor, digits=4))$)",
    marker=(:circle,3,:white),
    markerstrokecolor=:darkgreen,
    line=(:solid,2,:darkgreen),
    # fillcolor = :darkgreen,
    # fillalpha = 0.35,
)
# QM reference curve
plot!(fig,Iscan, QM_itp_zF1.(Iscan),
    label="Quantum mechanical model",
    line=(:solid,:red,1.75)
)
# CQD best-fit curve (from scaled refit)
plot!(fig,
    Iscan, ki_up_itp.(Iscan, Ref(fit_scaled.ki)),
    label=L"CoQuantum dynamics: $k_{i}= \left( %$(round(fit_scaled.ki; sigdigits=4)) \pm %$(round(fit_scaled.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
    line=(:dot,:blue,2),
    # marker=(:xcross, :blue, 0.2),
    markerstrokewidth=1
)
plot!(fig,
    Iscan, ki_up_itp.(Iscan, Ref(2.3)),
    label=L"CoQuantum dynamics: $k_{i}= \left( %$(round(2.3; sigdigits=4)) \pm %$(round(fit_scaled.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
    line=(:dot,:orangered,2),
    # marker=(:xcross, :blue, 0.2),
    markerstrokewidth=1
)
# Global formatting
plot!(fig,
    xlabel = "Coil Current (A)",
    ylabel = L"$z_{\mathrm{max}}$ (mm)",
    xaxis=:log10,
    yaxis=:log10,
    labelfontsize=14,
    tickfontsize=12,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # xlims=(0.010,1.05),
    size=(900,800),
    # legendtitle=L"$n_{z} = %$(nz_bins)$ | $\sigma_{\mathrm{conv}}=%$(1e3*gaussian_width_mm)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_raw)$",
    legendfontsize=12,
    left_margin=3mm,
)
display(fig)

relerr = 100*(yexp./scale_mag .- ki_up_itp.(Ic, Ref(fit_scaled.ki))) ./ ki_up_itp.(Ic, Ref(fit_scaled.ki))
pretty_table(
    hcat(Ic, yexp ./ scale_mag, ki_up_itp.(Ic, Ref(fit_scaled.ki)), relerr);
    column_labels = [
        ["Current", "F1 exp", "UP CQD", "Rel. Error"],
        ["[A]", "[mm]", "[mm]", "[%]"]
    ],
    alignment=:c,
    row_label_column_alignment=:c,
    row_group_label_alignment=:c,
    title="COQUANTUM DYNAMICS",
    formatters=[
        fmt__printf("%8.3f", [1]),
        fmt__printf("%8.4f", [2,3,5]),
        fmt__printf("%8.1f", [4,6])
    ],
    style=TextTableStyle(
        first_line_column_label=crayon"yellow bold",
        table_border=crayon"blue bold",
        title=crayon"bold red"
    ),
    table_format=TextTableFormat(borders=text_table_borders__unicode_rounded),
    equal_data_column_widths=true,
)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Fitting for induction factor for each individual data set
fit_ki_mode = :full ; # ← change to :low, :high, or :low_high
n_front  = 6 ;
n_back   = 6 ;
for dir in data_directories

    dir_chosen = EXP_data_processed[dir]
    data_threshold = dir_chosen[(0.025 .< dir_chosen.Ic) .& (dir_chosen.Ic .< 0.850), :]
    data = hcat(data_threshold[!,:Ic], 0.02 * data_threshold[!,:Ic] , data_threshold[!,:F1], data_threshold[!,:ErrF1] )

    low_range  = 1:n_front ;
    high_range = (size(data, 1) - n_back + 1):size(data, 1);

    @assert last(low_range) ≤ size(data,1)
    @assert first(high_range) ≥ 1

    # Select rows according to the chosen fitting mode
    fit_ki_idx = begin
        if fit_ki_mode === :full
            Colon()
        elseif fit_ki_mode === :low
            low_range
        elseif fit_ki_mode === :high
            high_range
        elseif fit_ki_mode === :low_high
            vcat(low_range, high_range)
        else
            error("Unknown fit_ki_mode = $fit_ki_mode")
        end
    end

    # -----------------------------------------------------------------------------
    # Compute a global scaling factor for the experimental z-values 
    # with respect to QM
    #
    # Motivation:
    #   Experimental z may differ from simulated z by an overall scale factor
    #   (e.g., magnification calibration). We estimate a single multiplicative
    #   factor using only the highest-current tail, where SNR is typically best.
    #
    # Scaling convention used:
    #   scaled_mag = (yexp⋅yexp) / (yexp⋅ythe)
    # so that (yexp / scaled_mag) best matches ythe in a least-squares sense.
    # -----------------------------------------------------------------------------
    n_tail = 6  # number of tail points used for scaling

    @printf "For the scaling of the experimental data, we use the current range = %.3f A – %.3f A \n" first(last(data[:, 1], n_tail)) last(last(data[:, 1], n_tail))
    yexp = last(data[:, 3], n_tail)              # experimental z-values (tail)
    ythe = last(zqm.(data[:, 1]), n_tail)        # QM reference z-values at same currents
    scaled_mag = dot(yexp, yexp) / dot(yexp, ythe)

    # Apply scaling to both z and δz to preserve relative uncertainties
    data_scaled = copy(data);
    data_scaled[:, 3] ./= scaled_mag;
    data_scaled[:, 4] ./= scaled_mag;

    # @printf "The re-scaling factor of the experimental data with respect to Quantum Mechanics is %.3f" scaled_mag
    @printf "The re-scaling factor for the high current regime is %.3f" scaled_mag
    println("")

    # -----------------------------------------------------------------------------
    # 4) Build fitting subsets and fit kᵢ using CQD interpolant surface
    #
    # Note:
    #   fit_ki minimizes the log-space residual internally, but reports a 
    #   linear-space RMSE as `ki_err`.
    # -----------------------------------------------------------------------------
    data_fitting        = data[fit_ki_idx, :];
    data_scaled_fitting = data_scaled[fit_ki_idx, :];
    fit_original = fit_ki(data, data_fitting, cqd_sim_data.ki, (ki_start,ki_stop))
    fit_scaled   = fit_ki(data_scaled, data_scaled_fitting, cqd_sim_data.ki, (ki_start,ki_stop))

    @info "Induction term ki = $(round(fit_scaled.ki; sigdigits=4))"

    z_qm = zqm.(I_scan);
    m_qm = log_mask(I_scan, z_qm);
    fig = plot(
        I_scan[m_qm], z_qm[m_qm];
        label = "Quantum mechanics",
        line  = (:solid, :red, 1.75),
    )
    plot!(
        fig,
        data_scaled[:,1], data_scaled[:,3];
        yerror= data_scaled[:,4],
        color = :gray35,
        marker = (:circle, :gray35, 4),
        markerstrokecolor = :gray35,
        markerstrokewidth = 1,
        label = dir,
    )
    z_fit_orig = ki_up_itp.(I_scan, Ref(fit_scaled.ki));
    m_orig = log_mask(I_scan, z_fit_orig);
    plot!(
        fig,
        I_scan[m_orig], z_fit_orig[m_orig];
        label = L"CQD : $k_{i}= \left( %$(round(fit_original.ki, sigdigits=3)) \pm %$(round(fit_scaled.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
        line  = (:solid, :blue, 2),
        marker = (:xcross, :blue, 0.2),
        markerstrokewidth = 1,
    )
    plot!(
        fig;
        # title = dir,
        xlabel = "Current (A)",
        ylabel = L"$z_{\mathrm{max}}$ (mm)",
        xaxis  = :log10,
        yaxis  = :log10,
        labelfontsize = 14,
        tickfontsize  = 12,
        xticks = ([1e-3, 1e-2, 1e-1, 1.0],
                [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-3, 1e-2, 1e-1, 1.0],
                [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xlims = (0.010, 1.05),
        size  = (900, 800),
        # legendtitle = L"$n_{z} = %$(nz_fixed)$ | $\sigma_{\mathrm{conv}}=%$(1e3*σw_fixed)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_fixed)$",
        legendfontsize = 12,
        left_margin = 3mm,
    )
    display(fig)

end

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


fit_ki_mode = :full   # ← change to :low, :high, or :low_high
n_front  = 8
n_back   = 6

data = hcat(combined_result[:Current].Ic, combined_result[:Current].σIc , combined_result[:MonteCarlo].zF1 , combined_result[:MonteCarlo].σzF1 )
low_range  = 1:n_front ;
high_range = (size(data, 1) - n_back + 1):size(data, 1);

@assert last(low_range) ≤ size(data,1)
@assert first(high_range) ≥ 1

# Select rows according to the chosen fitting mode
fit_ki_idx = begin
    if fit_ki_mode === :full
        Colon()
    elseif fit_ki_mode === :low
        low_range
    elseif fit_ki_mode === :high
        high_range
    elseif fit_ki_mode === :low_high
        vcat(low_range, high_range)
    else
        error("Unknown fit_ki_mode = $fit_ki_mode")
    end
end

n_tail = 8  # number of tail points used for scaling
@printf "For the scaling of the experimental data, we use the current range = %.3f A – %.3f A \n" first(last(data[:, 1], n_tail)) last(last(data[:, 1], n_tail))
yexp = last(data[:, 3], n_tail)              # experimental z-values (tail)
ythe = last(zqm.(data[:, 1]), n_tail)        # QM reference z-values at same currents
scaled_mag = dot(yexp, yexp) / dot(yexp, ythe)

# Apply scaling to both z and δz to preserve relative uncertainties
data_scaled = copy(data);
data_scaled[:, 3] ./= 1 #scaled_mag;
data_scaled[:, 4] ./= 1 #scaled_mag;

# @printf "The re-scaling factor of the experimental data with respect to Quantum Mechanics is %.3f" scaled_mag
@printf "The re-scaling factor for the high current regime is %.3f" scaled_mag
println("")

# -----------------------------------------------------------------------------
# 4) Build fitting subsets and fit kᵢ using CQD interpolant surface
#
# Note:
#   fit_ki minimizes the log-space residual internally, but reports a 
#   linear-space RMSE as `ki_err`.
# -----------------------------------------------------------------------------
data_fitting        = data[fit_ki_idx, :];
data_scaled_fitting = data_scaled[fit_ki_idx, :];
fit_original = fit_ki(data, data_fitting, cqd_sim_data.ki, (ki_start,ki_stop))
fit_scaled   = fit_ki(data_scaled, data_scaled_fitting, cqd_sim_data.ki, (ki_start,ki_stop))

@info "Induction term ki = $(round(fit_scaled.ki; sigdigits=4)) × 10^-6"

z_qm = zqm.(I_scan);
m_qm = log_mask(I_scan, z_qm);
fig = plot(
    I_scan[m_qm], scaled_mag .* z_qm[m_qm];
    label = "Quantum mechanics",
    line  = (:solid, :red, 1.75),
)
plot!(
    fig,
    data_scaled[:,1], data_scaled[:,3];
    yerror= data_scaled[:,4],
    label="Combined experiment",
    color = :gray35,
    marker = (:circle, :gray35, 4),
    markerstrokecolor = :gray35,
    markerstrokewidth = 1,
)
z_fit_orig = ki_up_itp.(I_scan, Ref(fit_scaled.ki));
m_orig = log_mask(I_scan, z_fit_orig);
plot!(
    fig,
    I_scan[m_orig], z_fit_orig[m_orig];
    label = L"CQD : $k_{i}= \left( %$(round(fit_original.ki, sigdigits=3)) \pm %$(round(fit_scaled.ki_err, sigdigits=1)) \right) \times 10^{-6} $",
    line  = (:solid, :blue, 2),
    marker = (:xcross, :blue, 0.2),
    markerstrokewidth = 1,
)
plot!(
    fig;
    # title = dir,
    xlabel = "Current (A)",
    ylabel = L"$z_{\mathrm{max}}$ (mm)",
    xaxis  = :log10,
    yaxis  = :log10,
    labelfontsize = 14,
    tickfontsize  = 12,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0],
              [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0],
              [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims = (0.010, 1.05),
    size  = (900, 800),
    # legendtitle = L"$n_{z} = %$(nz_fixed)$ | $\sigma_{\mathrm{conv}}=%$(1e3*σw_fixed)\mathrm{\mu m}$ | $\lambda_{\mathrm{fit}}=%$(λ0_fixed)$",
    legendfontsize = 12,
    left_margin = 3mm,
)
display(fig)

















































fit_ki_with_error(ki_up_itp, data_fitting; bounds=(cqd_sim_data.ki[ki_start], cqd_sim_data.ki[ki_stop]),)
fit_ki_with_error(ki_up_itp, data_scaled_fitting; bounds=(cqd_sim_data.ki[ki_start], cqd_sim_data.ki[ki_stop]),)



TheoreticalSimulation.BvsI(1.0)
TheoreticalSimulation.GvsI(1.0)

6400/130 * 4


TheoreticalSimulation.BvsI(0.020)


0.0200/0.0275











cqd_sim_data = JLD2_MyTools.list_keys_jld_cqd(data_cqdup_path);
n_ki = length(cqd_sim_data.ki);
@info "CQD simulation for $(n_ki) ki values"
colores_ki = palette(:darkrainbow, n_ki)
up_cqd = Matrix{Float64}(undef, nI, n_ki);
dw_cqd = Matrix{Float64}(undef, nI, n_ki);
Δz_cqd = Matrix{Float64}(undef, nI, n_ki);



















dir = data_directories[1]
kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS", "summary", dir, dir * "_report_summary.jld2")
EXP_data_dd = jldopen(kk_path, "r") do file
    ic = file["meta/Currents"]
    bz = file["meta/BzTesla"]

    dd = file[JLD2_MyTools.make_keypath_exp(dir, nz, λ0)]

    F1 = dd[:fw_F1_peak_pos_raw]
    F1b = dd[:fw_F1_peak_pos]
    F2 = dd[:fw_F2_peak_pos_raw]

    zc = 0.5 * (F1[1][1] + F2[1][1])
    δc = 0.5 * sqrt(F1[2][1]^2 + F2[2][1]^2)

    (
        Ic = ic,
        Bz = bz,
        F1 = F1,
        F1b = F1b,
        F2 = F2,
        C0 = dd[:centroid_fw_mm],
        C00 = [zc, δc]
    )
end
hcat(EXP_data_dd.F1[1] .- EXP_data_dd.C0[1], EXP_data_dd.F1b[1])








2+2













chosen_cqd_up =  jldopen(data_cqdup_path,"r") do file
    file[JLD2_MyTools.make_keypath_cqd(:up,nz, σw_mm, λ0)]
end

chosen_cqd_dw =  jldopen(data_cqddw_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end




!fig = plot(
    xlabel="Currents (mA)",
    ylabel=L"$F=1$ position (mm)",
    title="($nz,$λ0)",
    size=(950,600),
    left_margin=3mm,
    bottom_margin=2mm,
    legendfontsize=10,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
for (idx,dir) in enumerate(data_directories)
    kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS","summary", dir, dir * "_report_summary.jld2")
    data = jldopen(kk_path, "r") do file
        ic = file["meta/Currents"]
        bz = file["meta/BzTesla"]

        dd = file[JLD2_MyTools.make_keypath_exp(dir,nz,λ0)]
        return ( Ic=ic, Bz=bz, F1 = dd[:fw_F1_peak_pos_raw], F2 = dd[:fw_F2_peak_pos_raw], C0 = dd[:centroid_fw_mm] )
    end

    plot!(fig,
        1000*data.Ic, data.F1[1],
        yerror = data.F1[2],
        seriestype=:scatter,
        label="$(dir) (degauss = $(round(1e3*data.Ic[1]; digits=6))mA , $(Int(round(1e4*data.Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores_data[idx],
    )
    plot!(fig,
        1000*data.Ic, data.F1[1],
        seriestype=:line,
        label= nothing,
        line=(:solid,0.3,colores_data[idx]),
        color = colores_data[idx],
        fillalpha=0.10,
    )

    # plot!(fig,
    #     1000*data.Ic, data.F2,
    #     label="",
    #     marker=(:circle, 2, :white),
    #     markerstrokecolor=colores[idx],
    #     line=(:dash,1,colores[idx]))

    # hline!(fig, [data.C0[1]], label="",line=(:dash,1,colores[idx]))

end
display(fig)
plot!(fig,
    xlims=(0,50),
    ylims=(8.70,9.00),
);
display(fig)
plot!(fig,
    xlims=(750,1010),
    ylims=(10.30,10.90),

);
display(fig)

## Recalculation of the F=1 peak position with respect to a different centroid

new_data = Vector{NamedTuple}(undef, n_data)

for (idx, dir) in enumerate(data_directories)

    kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS","summary", dir, dir * "_report_summary.jld2")

    new_data[idx] = jldopen(kk_path, "r") do file
        ic = file["meta/Currents"]
        bz = file["meta/BzTesla"]

        dd = file[JLD2_MyTools.make_keypath_exp(dir,nz,λ0)]

        (Ic=ic, 
         Bz=bz,
         F1=dd[:fw_F1_peak_pos_raw],
         F2=dd[:fw_F2_peak_pos_raw],
         C0=dd[:centroid_fw_mm])
    end
end

new_centroid = [mean([new_data[v].F1[1][1],new_data[v].F2[1][1]]) for v=1:n_data]

plot(xlabel="Current (A)")
for (idx,dir) in enumerate(data_directories)

    z_table = hcat(new_data[idx].Ic , new_data[idx].F1[1] .- new_centroid[idx] )
    zz = DataReading.subset_by_cols( z_table , [1,2] ; thr = 1e-3, include_equal=false)[3]

    plot!(zz[:,1], zz[:,2],
    marker=(:circle,2,:white),
    markerstrokecolor=colores[idx],
    line=(colores[idx]),
    label=dir)
end
plot!(xlims=(0.030,0.050),
ylims=(10e-3, 120e-3))
plot!(Ic_QM_sim,zm_QM_sim,
    label="Quantum mechanics (T=200C)",
    line=(:solid,2,:blue))
for ki in [1.0,1.2,1.4,1.6,1.8,2.0]

    chosen_cqd =  jldopen(data_cqd_path,"r") do file
        file[JLD2_MyTools.make_keypath_cqd(:up, ki, nz, σw_mm, λ0)]
    end
    Ic_CQD_sim = [chosen_cqd[i][:Icoil] for i in eachindex(chosen_cqd)][2:end]
    zm_CQD_sim = [chosen_cqd[i][:z_max_smooth_spline_mm] for i in eachindex(chosen_cqd)][2:end]

    plot!(Ic_CQD_sim,zm_CQD_sim,
    label="CQD (T=200C | ki = $(ki)×10^-6)",
    line=(:solid,2))
end
plot!(
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.001, 0.01, 0.1, 1, 10], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    xlims=(8e-4,1.1),
    ylims=(8e-4,2.1),
    legend=:outerright,
    xscale=:log10,
    yscale=:log10,
)






dir = data_directories[6]
kk_path = joinpath(@__DIR__, "EXPDATA_ANALYSIS","summary", dir, dir * "_report_summary.jld2")
JLD2_MyTools.show_exp_summary(kk_path, dir)

data = jldopen(kk_path, "r") do file
    ic = file["meta/Currents"]
    bz = file["meta/BzTesla"]

    dd = file[JLD2_MyTools.make_keypath_exp(dir,2,0.01)]
    return ( Ic=ic, Bz=bz, F1 = dd[:fw_F1_peak_pos_raw], F2 = dd[:fw_F2_peak_pos_raw], C0 = dd[:centroid_fw_mm] )
end
data.C0

data_centroid_fw  = 0.5 * (data.F1[1] + data.F2[1])
data_δcentroid_fw = round.(0.5 * sqrt.(data.F1[2].^2 + data.F2[2].^2) ; sigdigits=1)
post_threshold_mean(data_centroid_fw, data.Ic, data_δcentroid_fw;
                                 threshold=0.010,
                                 half_life=5,
                                 eps=1e-6,
                                 weighted=true)


MyExperimentalAnalysis.post_threshold_mean(data_centroid_fw, data.Ic, data_δcentroid_fw;
                                 threshold=0.010,
                                 half_life=5,
                                 eps=1e-6,
                                 weighted=false)







