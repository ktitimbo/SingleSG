module TheoreticalSimulation
# ============================================================================
# DEPENDENCIES
# ============================================================================
# Plotting backend: GR via Plots.jl. `gr()` activates the GR backend
# explicitly (rather than relying on whatever the default happens to be),
# which matters for reproducibility since Plots.jl can switch backends
# depending on what's loaded/available in the environment.
using Plots; gr()
using Plots.PlotMeasures            # gives mm/inch/px units for plot margins, e.g. `left_margin = 5mm`
 
# ----- Aesthetics & output formatting -----
using Colors, ColorSchemes                      # named colors + perceptually-uniform colormaps for multi-series plots
using LaTeXStrings, Printf, PrettyTables         # LaTeX-rendered axis/legend labels, C-style formatted printing, console tables
 
# ----- Time-stamping / logging -----
using Dates                                      # used below to build a unique, run-stamped output directory
 
# ----- Numerical tools -----
using LinearAlgebra, DataStructures
using DifferentialEquations, ProgressMeter       # ODE/SDE solvers (atom trajectory integration) + progress bars for long batch runs
using Interpolations, Roots, Loess, Optim, DataInterpolations  # interpolation, root-finding, local regression smoothing, optimization
using BSplineKit                                 # penalized/weighted B-spline smoothing (see `compute_weights` below)
using DSP                                        # digital signal processing — used here for kernel convolution (`smooth_profile`)
using LambertW, PolyLog                          # special functions: Lambert-W and polylogarithms (dilogarithm Li₂, see below)
using StatsBase
using Random, Statistics, NaNStatistics, Distributions, StaticArrays
 
# Multithreading setup
using Base.Threads                               # enables `@threads` parallel loops used elsewhere in the pipeline
 
# Data manipulation
using OrderedCollections
using DelimitedFiles, CSV, DataFrames, JLD2      # JLD2 = binary, HDF5-based serialization for simulation results / DataFrames
 
# Custom (project-local) modules — atom physical/spin data and sampling utilities
include("./atoms.jl");
include("./samplings.jl");


# ============================================================================
# RUN-SCOPED OUTPUT CONFIGURATION
# ============================================================================
# The `if !isdefined(...)` guards below are a common Julia pattern for
# modules/scripts that may be `include`-d more than once in the same
# session (e.g. iterating in the REPL). Top-level `const`/binding
# re-declaration either errors or warns in Julia, so this pattern:
#   1. Lets a *calling* script pre-define OUTDIR/FIG_EXT/SAVE_FIG before
#      including this module, in which case those values are respected.
#   2. Falls back to sensible defaults otherwise, without crashing on
#      re-inclusion.
if !isdefined(@__MODULE__, :OUTDIR)
    # Millisecond-resolution timestamp guarantees a unique folder even if
    # the script/session is restarted twice within the same second.
    RUN_STAMP = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSSsss")
    OUTDIR    = joinpath(pwd(), "artifacts", RUN_STAMP)
    isdir(OUTDIR) || mkpath(OUTDIR)   # mkpath also creates intermediate dirs; no-op if already present
end
 
if !isdefined(@__MODULE__, :FIG_EXT)
    FIG_EXT = "svg"      # default saved-figure format (vector graphics; swap to "png" for raster)
end
 
if !isdefined(@__MODULE__, :SAVE_FIG)
    SAVE_FIG = false     # global toggle consumed by plotting functions: whether to actually write figures to OUTDIR
end

# ============================================================================
# PHYSICAL CONSTANTS (NIST / CODATA, SI units)
# ============================================================================
# RSU : Relative Standard Uncertainty
# Declared as `const` so Julia can specialize/inline on these values (faster
# numerics) and so any accidental later reassignment raises a warning/error
# instead of silently corrupting downstream physics.
const kb    = 1.380649e-23 ;       # Boltzmann constant (J/K) — exact, fixed by the 2019 SI redefinition
const ħ     = 6.62607015e-34/2π ;  # Reduced Planck constant (J·s) = h/2π, with h exact
const μ₀    = 1.25663706127e-6;    # Vacuum permeability (T·m/A) — measured (no longer exact post-2019 SI)
const μB    = 9.2740100657e-24 ;   # Bohr magneton (J/T)
const γₑ    = -1.76085962784e11 ;  # Electron gyromagnetic ratio (1/(s·T)); negative sign reflects the electron's negative charge. RSU = 3.0e-10
const μₑ    = 9.2847646917e-24 ;   # Electron magnetic moment (J/T), magnitude. RSU = 3.0e-10
const Sspin = 1/2 ;                # Electron spin quantum number S
const gₑ    = -2.00231930436092 ;  # Electron g-factor (dimensionless), includes the QED anomalous-moment correction
 
# ----- Math constants -----
const TWOπ  = 2π;          # cached 2π — avoids recomputing in hot loops, slightly clearer intent than writing `2*pi` inline
const INV_E = exp(-1);     # 1/e — appears in several effusive-beam / Maxwell–Boltzmann flux expressions

# ============================================================================
# STERN–GERLACH EXPERIMENT GEOMETRY (SI units, all lengths in meters)
# ============================================================================
# These are the nominal/default apparatus dimensions used to seed simulations
# unless a calling script overrides them. Prefixed `DEFAULT_` to make clear
# they're starting points, not hard physical constants.
DEFAULT_camera_pixel_size   = 6.5e-6 ;     # physical size of one camera pixel
DEFAULT_x_pixels            = 2160;        # camera sensor width, in pixels
DEFAULT_z_pixels            = 2560;        # camera sensor height, in pixels
 
# Furnace aperture (atomic-beam source slit)
DEFAULT_x_furnace           = 2.0e-3 ;
DEFAULT_z_furnace           = 100e-6 ;
 
# Collimating slit downstream of the furnace
DEFAULT_x_slit              = 4.0e-3 ;
DEFAULT_z_slit              = 300e-6 ;
 
# Circular aperture after the SG magnet pole pieces
DEFAULT_c_aperture          = 5.8e-3/2;    # stored as a *radius* (hence the /2 on the diameter-looking literal)
 
# Propagation distances along the beam axis (y)
DEFAULT_y_FurnaceToSlit     = 224.0e-3 ;
DEFAULT_y_SlitToSG          = 44.0e-3 ;
DEFAULT_y_SG                = 7.0e-2 ;     # length of the SG magnet itself
DEFAULT_y_SGToAperture      = 42.0e-3 ;
DEFAULT_y_SGToScreen        = 32.0e-2 ;
 
# Connecting vacuum pipe radius (m)
DEFAULT_R_tube = 35e-3/2 ;
 
# Characteristic SG pole-piece geometry (used in the analytic field model)
DEFAULT_𝒶                   = 2.5e-3 ;                                   # pole-piece separation parameter
DEFAULT_ℓ                   = 0.5*DEFAULT_y_SG;                          # half the magnet length (-ℓ,ℓ )
DEFAULT_center_of_SG_magnet = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_ℓ      # y-coordinate of the magnet's midpoint
DEFAULT_SG_magnet_entrance  = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG                   # y-coordinate where the beam enters the magnet

# CQD values
# for potassium
DEFAULT_CQD_Bn = 0.016475308514918384

# ============================================================================
# SESSION / WORKSPACE UTILITIES
# ============================================================================
"""
    clear_all() -> Nothing
 
Set to `nothing` every **non-const** binding in `Main`, except a small skip-list
(`:Base`, `:Core`, `:Main`, and `Symbol("@__dot__")`). This effectively clears
user-defined variables and functions from the current session without restarting Julia.
 
What it does
- Iterates `names(Main; all=true)` and, for each name:
- Skips if it is one of `:Base`, `:Core`, `:Main`, or `Symbol("@__dot__")`.
- Skips if the binding is **not defined** or is **const**.
- Otherwise sets the binding to `nothing` in `Main`.
- Triggers a `GC.gc()` afterward.
- Prints a summary message.
 
Notes & caveats
- This will clear **user functions** too (they're non-const bindings).
- Type names and imported modules are usually `const` in `Main`, so they are **not** cleared.
- This does not unload packages or reset the environment; it only nukes non-const globals.
- There is no undo; you'll need to re-run definitions after clearing.
 
Example
julia
julia> x = 1; y = "hi"; f(x) = x+1;
 
julia> clear_all()
All user-defined variables (except constants) cleared.
 
julia> x, y, f
(nothing, nothing, nothing)
"""
function clear_all()
    # `names(Main, all=true)` returns every symbol bound in Main, including
    # ones that would normally be hidden from `names(Main)` (e.g. names not
    # exported by a module, or starting with `#`/internal compiler symbols).
    for name in names(Main, all=true)
        if name ∉ (:Base, :Core, :Main, Symbol("@__dot__"))
            # `isdefined` guards against stale/placeholder symbols that
            # `names(..., all=true)` can surface but that aren't actually
            # bound to anything yet; `isconst` protects `const` bindings
            # (Julia disallows redefining those anyway, so skipping avoids
            # an error here).
            if !isdefined(Main, name) || isconst(Main, name)
                continue  # Skip constants
            end
            # `@eval Main begin ... end` is required because `name` is a
            # runtime `Symbol`, not a literal identifier — this is the
            # standard idiom for assigning to a *dynamically chosen*
            # top-level binding by name.
            @eval Main begin
                global $name = nothing
            end
        end
    end
    GC.gc()   # encourage immediate reclamation of whatever those bindings pointed to (large arrays, etc.)
    println("All user-defined variables (except constants) cleared.")
end

 
# ============================================================================
# SPECIAL FUNCTIONS — dilogarithm (Li₂) helpers
# ============================================================================
"""Return the real dilogarithm `Li₂(z)` via `reli2(z)`; `s` is ignored."""
function polylogarithm(s,z)
    # `s` (the polylog *order*) is accepted but ignored: only the
    # second-order polylogarithm — the dilogarithm
    #     Li₂(z) = -∫₀^z ln(1-t)/t dt
    # — is actually evaluated. The `s` parameter is kept purely so call
    # sites read like a generic `polylog(s, z)` interface, making it a
    # drop-in replacement if a general-order package is substituted later.
    # return MyPolylogarithms.polylog(s,z)   # earlier/alternative implementation, kept for reference
    return PolyLog.reli2(z)
end
 
const _PI2_OVER_6 = (pi*pi) / 6.0   # π²/6 = Li₂(1) = ζ(2); precomputed once so it isn't recomputed every call below
 
"""
    li2_negexp(x::Float64) -> Float64
 
Numerically robust evaluation of `Li₂(-exp(x))` for **any** real `x`,
including large positive `x` where `exp(x)` would otherwise overflow or
push the dilogarithm argument outside its well-conditioned range.
 
Mathematical basis — dilogarithm inversion identity:
 
    Li₂(-e^x) + Li₂(-e^{-x}) = -π²/6 - x²/2
 
Rearranged for `x > 0`:
 
    Li₂(-e^x) = -x²/2 - π²/6 - Li₂(-e^{-x})
 
so the exponential that actually gets formed is `exp(-x) ∈ (0, 1]`
(always safe), rather than `exp(x)`, which blows up for `x ≳ 700`.
For `x ≤ 0`, `exp(x) ≤ 1` is already safe and the identity isn't needed.
"""
@inline function li2_negexp(x::Float64)::Float64
    if x > 0.0
        exm = exp(-x)  # safe (in (0,1])
        # muladd(a, b, c) = a*b + c computed as a single rounded operation
        # (often lowered to a fused-multiply-add CPU instruction): marginally
        # faster and more accurate than writing the subtraction/sum out by hand.
        return muladd(-0.5, x*x, -_PI2_OVER_6 - polylogarithm(2, -exm))
    else
        ex  = exp(x)   # safe (<= 1), since x <= 0 here
        return polylogarithm(2, -ex)
    end
end
 

# ============================================================================
# CURVE FITTING / SMOOTHING HELPERS
# ============================================================================
"""
    For BSplineKit fitting, compute weights for the B-spline fit.
    Compute uniform weights scaled by (1 - λ0). Returns an array of the same size as `x_array`.
"""
function compute_weights(x_array, λ0)
    # Gives a uniform (all data points trusted equally) weight
    # vector. Scaling by (1 - λ0) ties the overall weight magnitude to the
    # smoothing parameter λ0 used elsewhere in the BSplineKit fit:
    #   λ0 → 0  ⇒ weights ≈ 1 (trust the data; little smoothing)
    #   λ0 → 1  ⇒ weights ≈ 0 (data barely constrains the fit; smoothing dominates)
    return fill(1 - λ0, size(x_array))
end

"""
    FreedmanDiaconisBins(data::AbstractVector{<:Real}) -> Int
 
Return the optimal number of histogram bins for `data` using the
**Freedman–Diaconis rule**:
 
    bin_width = 2 * IQR / n^(1/3)
    bins      = ceil( range / bin_width )
 
where:
- `IQR` is the interquartile range (Q3 − Q1).
- `n` is the number of samples.
- `range` is `maximum(data) - minimum(data)`.
 
This rule balances resolution with statistical noise and is robust to outliers.
 
# Arguments
- `data::AbstractVector{<:Real}`: 1D array of real numeric values.
 
# Returns
- `Int`: Number of bins.
 
# Notes
- If `IQR` is zero (e.g., all values identical), returns `1`.
- Assumes `data` has at least one element.
- Automatically promotes input to `Float64` for calculations.
"""
function FreedmanDiaconisBins(data::AbstractVector{<:Real})
    @assert !isempty(data) "data must not be empty"
    # Promote to Float64 up front so integer/Float32 inputs don't cause
    # truncated/inexact arithmetic in the quantile and bin-width calculations.
    data_f = eltype(data) === Float64 ? data : Float64.(data)
 
    # Interquartile range — robust spread measure, unlike e.g. std(),
    # which is sensitive to outliers/heavy tails.
    Q1, Q3 = quantile(data_f, (0.25, 0.75))
    IQR = Q3 - Q1
 
    # Edge case: zero spread (degenerate/constant data) would otherwise
    # divide by zero below.
    if IQR == 0
        return 1
    end
 
    # Freedman–Diaconis bin width: shrinks (more bins) as sample size n grows,
    # at rate n^(-1/3), trading resolution against per-bin statistical noise.
    n = length(data_f)
    bin_width = 2 * IQR / (n^(1/3))
 
    # Number of bins needed to cover the full data range at that bin width.
    data_range = maximum(data_f) - minimum(data_f)
    bins = max(1, ceil(Int, data_range / bin_width))   # `max(1, ...)` guards against degenerate ranges
 
    return bins
end

# Plot a histogram of `data_list` using the Freedman–Diaconis binning rule.
function FD_histograms(data_list::AbstractVector{<:Real}, Label::LaTeXString, color)
    bins = FreedmanDiaconisBins(data_list)
    histogram(data_list, 
        bins=bins, 
        normalize=:pdf, 
        label=Label, 
        color=color, 
        alpha=0.8,
        xlim=(0,π), 
        xticks=PlottingTools.pitick(0, π, 8; mode=:latex)
    )
end


# ============================================================================
# GEOMETRY / SIGNAL-PROCESSING HELPERS (camera binning, profile smoothing)
# ============================================================================
"""Return the midpoints between consecutive histogram/bin `edges`."""
function bin_centers(edges::AbstractVector)
    # Standard "average adjacent edges" midpoint formula; works for both
    # uniform and non-uniform bin edges.
    # can be replaced by StatsBase.midpoints(edges) since it is 
    # the same as (edges[1:end-1] .+ edges[2:end]) ./ 2
    return (edges[1:end-1] .+ edges[2:end]) ./ 2
end

"""
Centers of binned pixels (1D) in physical units. First center at (bin_size*pixel_size)/2. Requires img_size % bin_size == 0.
"""
@inline function pixel_coordinates(img_size::Integer, bin_size::Integer, pixel_size::Real)
    # Binning groups `bin_size` raw camera pixels into one "super-pixel";
    # this only makes sense if the sensor dimension divides evenly.
    @assert img_size % bin_size == 0 "img_size must be divisible by bin_size"
    n  = img_size ÷ bin_size            # number of super-pixels along this axis
    Δ  = bin_size * float(pixel_size)   # physical width of one super-pixel
    # Centers are placed at Δ/2, 3Δ/2, 5Δ/2, ... — i.e. Δ*(1:n) shifted back
    # by half a bin width so the first center sits in the middle of the
    # first super-pixel rather than at its right edge.
    return range(Δ/2, step=Δ, length=n)
end
 
"""Build a zero-centered Gaussian convolution kernel of standard deviation `wd`, sampled at the points `x`, normalized to sum to 1."""
function gaussian_kernel(x,wd::Number)
    # Standard (unnormalized-by-construction-but-then-renormalized) Gaussian
    # density evaluated at each point of `x`. NOTE: `x` is assumed to already
    # be a coordinate grid centered on zero (i.e. this is *not* re-centered
    # internally) — the kernel literally is exp(-x²/2wd²), so if `x` is not
    # symmetric about 0 the resulting kernel will not be properly centered.
    kernel = (1 / (sqrt(2π) * wd)) .* exp.(-x .^ 2 ./ (2 * wd^2))
    kernel ./= sum(kernel)  # discrete renormalization: forces Σkernel = 1 exactly,
                             # compensating for the fact that a finite/discrete
                             # sample of a Gaussian doesn't sum to 1 analytically.
    return kernel
end

"""Smooth `pdf_vals` (sampled at `z_vals`) by convolving with a Gaussian kernel of width `wd`, returning a vector the same length as `pdf_vals`."""
function smooth_profile(z_vals::AbstractVector{<:Real}, pdf_vals::AbstractVector{<:Real}, wd::Number)
    # Build the smoothing kernel on the same grid spacing as the data.
    # (z_vals is used only to determine the kernel's support/length here;
    # see the centering caveat noted in `gaussian_kernel`.)
    kernel = gaussian_kernel(z_vals,wd)
 
    # `DSP.conv` performs *full* linear convolution, so its output is longer
    # than the input: length(pdf_vals) + length(kernel) - 1.
    smoothed = DSP.conv(pdf_vals, kernel)
 
    # Trim the full convolution back down to `length(pdf_vals)` samples,
    # aligned the same way MATLAB's `conv(...,'same')` mode would: drop
    # (kernel_length-1)/2 samples from each end so index i of the output
    # still corresponds to z_vals[i] / pdf_vals[i].
    n = length(pdf_vals)
    start_idx = div(length(kernel), 2) + 1
    return smoothed[start_idx:start_idx + n - 1]
end


# ============================================================================
# SUBMODULE FILES
# ============================================================================
# The bulk of the physics pipeline lives in separate `include`-d files,
# split by concern so each piece stays reviewable on its own. All of them
# share this module's namespace (constants, `using` imports, and the helper
# functions defined above are all visible inside them):
#
#   TheoreticalSimulation_Params.jl              -- parameter/struct definitions (AtomParams, EffusionParams, BeamEffusionParams, ...)
#   TheoreticalSimulation_AnalyticMagneticField.jl -- closed-form/analytic SG magnetic field model
#   TheoreticalSimulation_MagneticField.jl       -- numerical magnetic field handling (e.g. field maps, gradients)
#   TheoreticalSimulation_muF.jl                 -- effective magnetic-moment (μF) / Zeeman-sublevel logic
#   TheoreticalSimulation_EquationsOfMotion.jl   -- ODE right-hand sides for atom trajectories through the SG apparatus
#   TheoreticalSimulation_VelocityPDF.jl         -- thermal/effusive beam velocity distribution functions
#   TheoreticalSimulation_Sampling.jl            -- Monte Carlo initial-condition / trajectory sampling
#   TheoreticalSimulation_DiscardedParticles.jl  -- bookkeeping for atoms that hit apertures/walls and don't reach the screen
#   TheoreticalSimulation_Spline.jl              -- spline-based post-processing of simulated/experimental profiles
#   TheoreticalSimulation_Plots.jl               -- higher-level diagnostic plotting routines built on the helpers above
    include(joinpath(@__DIR__, "TheoreticalSimulation_Params.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_AnalyticMagneticField.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_MagneticField.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_muF.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_EquationsOfMotion.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_VelocityPDF.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_Sampling.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_DiscardedParticles.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_Spline.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_Plots.jl"))
 
 
# ============================================================================
# PUBLIC API
# ============================================================================
# Only the names below are exported (i.e. usable as bare `AtomParams`,
# `clear_all`, etc. after `using TheoreticalSimulation` — everything else
# defined in this module or its `include`-d files is still *accessible*
# via `TheoreticalSimulation.something`, just not brought into the caller's
# namespace automatically). Grouped here by role:
#
#   Parameter/config structs : AtomParams, EffusionParams, BeamEffusionParams
#   Session utilities         : clear_all, compute_weights, FreedmanDiaconisBins
#   Geometry helpers           : pixel_coordinates
#   Physics                    : fmf_levels, μF_effective
#   Monte Carlo sampling       : generate_samples, generate_CQDinitial_conditions
#   Diagnostic plotting        : plot_μeff, plot_SG_geometry, plot_velocity_stats
#   Profile analysis           : QM_analyze_profiles_to_dict, CQD_analyze_profiles_to_dict
    export AtomParams, EffusionParams, BeamEffusionParams,
            clear_all, compute_weights, FreedmanDiaconisBins,
            pixel_coordinates,
            fmf_levels, μF_effective,
            generate_samples, generate_CQDinitial_conditions,
            QM_analyze_profiles_to_dict,
            CQD_analyze_profiles_to_dict
 

end