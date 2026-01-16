module TheoreticalSimulation
    using Plots; gr()
    using Plots.PlotMeasures
    # Aesthetics and output formatting
    using Colors, ColorSchemes
    using LaTeXStrings, Printf, PrettyTables
    # Time-stamping/logging
    using Dates
    # Numerical tools
    using LinearAlgebra, DataStructures
    using Interpolations, Roots, Loess, Optim, DataInterpolations
    using BSplineKit
    using DSP
    using LambertW, PolyLog
    using StatsBase
    using Random, Statistics, NaNStatistics, Distributions, StaticArrays
    # Multithreading setup
    using Base.Threads
    # Data manipulation
    using OrderedCollections
    using DelimitedFiles, CSV, DataFrames, JLD2
    # Customo modules
    include("./atoms.jl");
    include("./samplings.jl");

    # Defaults only if not already present in this module
    if !isdefined(@__MODULE__, :OUTDIR)
        RUN_STAMP = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSSsss")
        OUTDIR    = joinpath(pwd(), "artifacts", RUN_STAMP)
        isdir(OUTDIR) || mkpath(OUTDIR)
    end

    if !isdefined(@__MODULE__, :FIG_EXT)
        FIG_EXT = "svg"
    end

    if !isdefined(@__MODULE__, :SAVE_FIG)
        SAVE_FIG = false
    end

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

    # Math constants
    const TWOπ = 2π;
    const INV_E = exp(-1);

    # STERN--GERLACH EXPERIMENT
    default_camera_pixel_size   = 6.5e-6 ;
    default_x_pixels            = 2160;
    default_z_pixels            = 2560;
    # Furnace aperture
    default_x_furnace           = 2.0e-3 ;
    default_z_furnace           = 100e-6 ;
    # Slit
    default_x_slit              = 4.0e-3 ;
    default_z_slit              = 300e-6 ;
    # Circular aperture
    default_c_aperture          = 5.8e-3/2; # radius of the circular aperture after the pole pieces
    # Propagation distances
    default_y_FurnaceToSlit     = 224.0e-3 ;
    default_y_SlitToSG          = 44.0e-3 ;
    default_y_SG                = 7.0e-2 ;
    default_y_SGToAperture      = 42.0e-3 ; 
    default_y_SGToScreen        = 32.0e-2 ;
    # Connecting pipes
    default_R_tube = 35e-3/2 ; # Radius of the connecting pipe (m)


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
    - This will clear **user functions** too (they’re non-const bindings).
    - Type names and imported modules are usually `const` in `Main`, so they are **not** cleared.
    - This does not unload packages or reset the environment; it only nukes non-const globals.
    - There is no undo; you’ll need to re-run definitions after clearing.

    Example
    ```julia
    julia> x = 1; y = "hi"; f(x) = x+1;

    julia> clear_all()
    All user-defined variables (except constants) cleared.

    julia> x, y, f
    (nothing, nothing, nothing)
    """
    function clear_all()
        for name in names(Main, all=true)
            if name ∉ (:Base, :Core, :Main, Symbol("@__dot__"))
                if !isdefined(Main, name) || isconst(Main, name)
                    continue  # Skip constants
                end
                @eval Main begin
                    global $name = nothing
                end
            end
        end
        GC.gc()
        println("All user-defined variables (except constants) cleared.")
    end

    """Return the real dilogarithm `Li₂(z)` via `reli2(z)`; `s` is ignored."""
    function polylogarithm(s,z)
        # return MyPolylogarithms.polylog(s,z)
        return PolyLog.reli2(z)
    end

    const _PI2_OVER_6 = (pi*pi) / 6.0

    @inline function li2_negexp(x::Float64)::Float64
        # Returns Li₂(-exp(x)) robustly for any real x without forming exp(x) when x is large positive.
        # For x > 0: Li₂(-e^x) = -x^2/2 - π^2/6 - Li₂(-e^{-x})
        if x > 0.0
            exm = exp(-x)  # safe (in (0,1])
            return muladd(-0.5, x*x, -_PI2_OVER_6 - polylogarithm(2, -exm))
        else
            ex  = exp(x)   # safe (<= 1)
            return polylogarithm(2, -ex)
        end
    end


    """
        For BSplineKit fitting, compute weights for the B-spline fit.
        Compute uniform weights scaled by (1 - λ0). Returns an array of the same size as `x_array`.
    """
    function compute_weights(x_array, λ0)
        return (1 - λ0) * fill!(similar(x_array), 1)
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
        # Promote to Float64 for calculations
        data_f = Float64.(data)

        # Interquartile range
        Q1 = quantile(data_f, 0.25)
        Q3 = quantile(data_f, 0.75)
        IQR = Q3 - Q1

        # Edge case: no spread in data
        if IQR == 0
            return 1
        end

        # Freedman–Diaconis bin width
        n = length(data_f)
        bin_width = 2 * IQR / (n^(1/3))

        # Number of bins
        data_range = maximum(data_f) - minimum(data_f)
        bins = max(1, ceil(Int, data_range / bin_width))

        return bins
    end

    # Function to plot histogram using Freedman-Diaconis binning rule
    function FD_histograms(data_list::Vector{Float64},Label::LaTeXString,color)
        # Calculate the interquartile range (IQR)
        Q1 = quantile(data_list, 0.25)
        Q3 = quantile(data_list, 0.75)
        IQR = Q3 - Q1

        # Calculate Freedman-Diaconis bin width
        n = length(data_list)
        bin_width = 2 * IQR / (n^(1/3))

        # Calculate the number of bins using the range of the data
        data_range = maximum(data_list) - minimum(data_list)
        bins = ceil(Int, data_range / bin_width)

        # Plot the histogram
        histogram(data_list, bins=bins, normalize=:pdf,
                label=Label,
                # xlabel="Polar angle", 
                color=color,
                alpha=0.8,
                xlim=(0,π),
                xticks=PlottingTools.pitick(0, π, 8; mode=:latex),)
    end

    function bin_centers(edges::AbstractVector)
        return (edges[1:end-1] .+ edges[2:end]) ./ 2
    end


    """
    Centers of binned pixels (1D) in physical units. First center at (bin_size*pixel_size)/2. Requires img_size % bin_size == 0.
    """
    @inline function pixel_coordinates(img_size::Integer, bin_size::Integer, pixel_size::Real)
        @assert img_size % bin_size == 0 "img_size must be divisible by bin_size"
        n  = img_size ÷ bin_size
        Δ  = bin_size * float(pixel_size)
        return Δ .* (1:n) .- Δ/2
    end


    function gaussian_kernel(x,wd::Number)
        # Create Gaussian kernel around zero
        kernel = (1 / (sqrt(2π) * wd)) .* exp.(-x .^ 2 ./ (2 * wd^2))
        kernel ./= sum(kernel)  # normalize to sum to 1
        return kernel
    end

    function smooth_profile(z_vals::AbstractVector{<:Real}, pdf_vals::AbstractVector{<:Real}, wd::Number)
        kernel = gaussian_kernel(z_vals,wd)
        # Convolve pdf values with kernel, pad=true means full convolution
        smoothed = DSP.conv(pdf_vals, kernel)
        # Trim convolution result to same length as input, like MATLAB 'same'
        n = length(pdf_vals)
        start_idx = div(length(kernel), 2) + 1
        return smoothed[start_idx:start_idx + n - 1]
    end


    include(joinpath(@__DIR__, "TheoreticalSimulation_Params.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_MagneticField.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_muF.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_EquationsOfMotion.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_VelocityPDF.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_Sampling.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_DiscardedParticles.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_Spline.jl"))
    include(joinpath(@__DIR__, "TheoreticalSimulation_Plots.jl"))


    

    export AtomParams, EffusionParams, BeamEffusionParams,
            clear_all, compute_weights, FreedmanDiaconisBins,
            pixel_coordinates,
            fmf_levels, μF_effective, 
            generate_samples, generate_CQDinitial_conditions,
            plot_μeff, plot_SG_geometry, plot_velocity_stats,
            QM_analyze_profiles_to_dict,
            CQD_analyze_profiles_to_dict

end