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
    using Interpolations, Roots, Loess, Optim
    using BSplineKit
    using DSP
    using LambertW, PolyLog
    using StatsBase
    using Random, Statistics, NaNStatistics, Distributions, StaticArrays
    using Alert
    # Data manipulation
    using OrderedCollections
    using DelimitedFiles, CSV, DataFrames, JLD2
    # Customo modules
    include("./atoms.jl");

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
    # Furnace aperture
    default_x_furnace = 2.0e-3 ;
    default_z_furnace = 100e-6 ;
    # Slit
    default_x_slit  = 4.0e-3 ;
    default_z_slit  = 300e-6 ;
    # Propagation distances
    default_y_FurnaceToSlit = 224.0e-3 ;
    default_y_SlitToSG      = 44.0e-3 ;
    default_y_SG            = 7.0e-2 ;
    default_y_SGToScreen    = 32.0e-2 ;
    # Connecting pipes
    default_R_tube = 35e-3/2 ; # Radius of the connecting pipe (m)

    Base.@kwdef struct AtomParams{T<:Real}
        name::Symbol = :unknown
        R::T         # van der Waals radius
        μn::T        # nuclear magneton
        γn::T        # nuclear gyromagnetic ratio
        Ispin::T     # nuclear spin I
        Ahfs::T      # hyperfine constant
        M::T         # mass
    end

    # Build from a tuple/array in positions (1,2,3,4,6,7)
    AtomParams(atom; T=Float64) = begin
        ai = AtomicSpecies.atoms(atom)
        AtomParams{T}(
            name = Symbol(atom),
            R    = T(ai[1]),
            μn   = T(ai[2]),
            γn   = T(ai[3]),
            Ispin= T(ai[4]),
            Ahfs = T(ai[6]),
            M    = T(ai[7]),
        )
    end

    # ---- Gradient ↔ current tables (hardcoded) ----
    const GRAD_CURRENTS = [0, 0.095, 0.2, 0.302, 0.405, 0.498, 0.6, 0.7, 0.75, 0.8, 0.902, 1.01]
    const GRAD_GRADIENT = [0, 25.6, 58.4, 92.9, 132.2, 164.2, 196.3, 226, 240, 253.7, 277.2, 298.6]

    const _GvsI = Interpolations.LinearInterpolation(GRAD_CURRENTS, GRAD_GRADIENT; extrapolation_bc=Line())
    const _IvsG = Interpolations.LinearInterpolation(GRAD_GRADIENT, GRAD_CURRENTS; extrapolation_bc=Line())

    # Public wrappers (stable to call inside other functions)
    GvsI(x) = _GvsI(x)
    IvsG(x) = _IvsG(x)

    # ---- Magnetic Field vs current from CSV ----
    const B_TABLE_PATH = joinpath(@__DIR__, "SG_BvsI.csv")
    println("Importing file from $(B_TABLE_PATH)")
    const _BvsI = Ref{Any}(nothing)  # will hold the interpolation object
    
    function __init__()
        if isfile(B_TABLE_PATH)
            df = CSV.read(B_TABLE_PATH, DataFrame; header=["dI","Bz"])
            _BvsI[] = linear_interpolation(df.dI, df.Bz; extrapolation_bc=Line())
        else
            @warn "B table not found at $B_TABLE_PATH; call set_B_table! first."
        end
    end

    # Public accessor
    @inline function BvsI(I::Real)
        itp = _BvsI[]
        itp === nothing && error("BvsI not initialized. Load the table first.")
        return itp(float(I))
    end

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

    """
    Centers of binned pixels (1D) in physical units. First center at (bin_size*pixel_size)/2. Requires img_size % bin_size == 0.
    """
    @inline function pixel_coordinates(img_size::Integer, bin_size::Integer, pixel_size::Real)
        @assert img_size % bin_size == 0 "img_size must be divisible by bin_size"
        n  = img_size ÷ bin_size
        Δ  = bin_size * float(pixel_size)
        return Δ .* (1:n) .- Δ/2
    end

    """
        EffusionParams{T}

    Container of precomputed beam-sampling parameters.

    Fields
    - `sinθmax::T` — max sine of the polar angle (0 ≤ sinθmax ≤ 1)
    - `α2::T`      — speed scale `kB*T/M` (m²/s²)
    """
    Base.@kwdef struct EffusionParams
        sinθmax::Float64
        α2::Float64
    end

    """
        BeamEffusionParams(default_x_furnace, default_z_furnace, default_x_slit, default_z_slit, default_y_FurnaceToSlit, T, M) -> EffusionParams

    Build beam parameters from geometry (furnace/slit and spacing) and thermals.
    Returns `EffusionParams(sinθmax, α2)` with `α2 = kB*T/M` and
    `θvmax = 1.25 * atan(norm(Δxz), default_y_FurnaceToSlit)`, where
    `Δxz = (-default_x_furnace/2, -default_z_furnace/2) − (default_x_slit/2, default_z_slit/2)`.
    Assumes `kb` is in scope.
    """
    @inline function BeamEffusionParams(xx_furnace, zz_furnace, xx_slit, zz_slit, yy_FurnaceToSlit, T, p::AtomParams )
        Δxz   = SVector(-xx_furnace/2, -zz_furnace/2) - SVector(xx_slit/2, zz_slit/2)
        θvmax = 1.25 * atan(norm(Δxz), yy_FurnaceToSlit)
        return EffusionParams(sin(θvmax), kb*T/p.M)
    end

    """
        AtomicBeamVelocity_v3(rng, p::EffusionParams) -> SVector{3,Float64}

    Sample a velocity `(vx, vy, vz)` for an effusive beam within a cone of
    half-angle `asin(p.sinθmax)`. Speed PDF proportional to v^3 
    """
    @inline function AtomicBeamVelocity_v3(rng::AbstractRNG, p::EffusionParams)::SVector{3,Float64}
        ϕ = TWOπ * rand(rng)
        θ = asin(p.sinθmax * sqrt(rand(rng)))
        v = sqrt(-2*p.α2 * (1 + lambertw((rand(rng)-1)*INV_E, -1)))
        sθ = sin(θ); cθ = cos(θ); sϕ = sin(ϕ); cϕ = cos(ϕ)
        return SVector(v*sθ*sϕ, v*cθ, v*sθ*cϕ)
    end

    """
        AtomicBeamVelocity_v2(rng, p::EffusionParams) -> SVector{3,Float64}

    Sample a velocity `(vx, vy, vz)` for an effusive beam within a cone of
    half-angle `asin(p.sinθmax)`. Speed PDF proportional to v^3 
    """
    @inline function AtomicBeamVelocity_v2(rng,p::EffusionParams)::SVector{3,Float64} 
        ϕ = TWOπ * rand(rng)
        θ = asin(p.sinθmax * sqrt(rand(rng)))
        v = sqrt(2 .* p.α2 .* rand(rng, Gamma(3/2,1.0)))
        sθ = sin(θ); cθ = cos(θ); sϕ = sin(ϕ); cϕ = cos(ϕ)
        return SVector(v*sθ*sϕ, v*cθ, v*sθ*cϕ)
    end

    # Quantum Magnetic Moment μF : electron(1/2)-nucleus(3/2)
    """
        μF_effective(Ix, II, F, mF) -> Float64

        Effective magnetic moment μ_F for a given hyperfine manifold and Zeeman sublevel,
        based on the (Breit–Rabi–style) expression you coded.

        Inputs
        - `Ix`  : Coil current (units consistent with `BvsI(Ix)` → magnetic field).
        - `II`  : Nuclear spin quantum number (I). Can be integer or half-integer.
        - `F`   : Total angular momentum (must be `I ± 1/2`).
        - `mF`  : Magnetic quantum number (must satisfy `-F ≤ mF ≤ F`).

        Assumptions / Globals
        - Uses global constants: `ħ, Ahfs, γₑ, γₙ, μB, gₑ`.
        - Uses global field map/function: `BvsI(Ix)` returning B (same units used in Δ).
        - Defines the adimensional field parameter
        `normalized_B = (γₑ - γₙ) * ħ / ΔE * BvsI(Ix)`,
        where `ΔE = 2π * ħ * Ahfs * (I + 1/2)`.

        Details
        - For the upper manifold `F = I + 1/2`, the `mF = ±F` edges use the simplified
        analytic form `μF = ± gₑ/2 * (1 + 2*γₙ/γₑ * I) * μB`.
        - For other `mF` and for the lower manifold `F = I - 1/2`, uses the full expressions
        with the square‑root denominator
        `sqrt(1 - 4*mF/(2I+1)*normalized_B + normalized_B^2)`; the argument is clamped
        to ≥ 0 to avoid numerical noise causing `NaN`.

        Returns
        - `Float64` effective magnetic moment (units of μB if you keep the constants consistent).
    """
    function μF_effective(Ix,F,mF,p::AtomParams)
        # Promote to Float64 to avoid mixed-type arithmetic issues
        II = float(p.Ispin)
        γₙ = p.γn

        Ix, F, mF = promote(float(Ix), float(F), float(mF))

        # Energy scale and adimensional field
        ΔE = 2π * ħ * p.Ahfs * ( II + 1/2 )
        normalized_B = (γₑ-γₙ)*ħ / ΔE * BvsI(Ix) 
        
        # Validate quantum numbers
        is_F_upper = isapprox(F, II + 0.5; atol = 1e-12)
        is_F_lower = isapprox(F, II - 0.5; atol = 1e-12)
        if !(is_F_upper || is_F_lower)
            throw(ArgumentError("F must be I±1/2; got F=$F for I=$II"))
        end
        if mF < -F - 1e-12 || mF > F + 1e-12
            throw(ArgumentError("mF must be in [-F, F]; got mF=$mF for F=$F"))
        end

        # Common pieces
        ratio = γₙ / γₑ
        denom_arg = 1 - 4*mF/(2*II + 1) * normalized_B + normalized_B^2
        # Clamp tiny negative due to rounding
        denom = sqrt(max(denom_arg, 0.0))

        μF::Float64 = NaN  # <-- initialize
        if is_F_upper 
            if isapprox(mF,  F; atol = 1e-12) || isapprox(mF, -F; atol = 1e-12)
                s = sign(mF)
                μF = s * (gₑ/2) * (1 + 2*ratio*II) * μB
            else
                μF = gₑ * μB * ( mF*ratio + (1 - ratio)/denom * ( mF/(2*II + 1) - 0.5*normalized_B ) )
            end
        else # is_F_lower
            μF = gₑ * μB * ( mF*ratio - (1 - ratio)/denom * ( mF/(2*II + 1) - 0.5*normalized_B ) )
        end

        return μF
    end

"""
    plot_ueff(II, path_filename::AbstractString) -> Plot

    Plot the effective magnetic moment μ_F/μ_B versus coil current for all hyperfine
    levels (F, m_F) of a spin-I system, and annotate the magnetic crossing point.

    # Arguments
    - `II`: Nuclear spin quantum number (e.g., 3/2, 4, etc.).
    - `path_filename::AbstractString`: Output file path for saving the figure.

    # Behavior
    - Computes and plots μ_F/μ_B curves for all (F, m_F) states using `μF_effective`.
    - Uses solid lines for most F = I + 1/2 states, a dashed line for the lowest m_F
    in F = I + 1/2, and dashed lines for all F = I – 1/2 states.
    - Colors each curve using the `:phase` palette.
    - Finds the magnetic crossing current `I₀` by solving `BvsI(I) = …` and annotates
    the plot with:
        - I₀ in A
        - ∂ₓBₓ at I₀ in T/m
        - B_z at I₀ in mT
    - Plots current on a logarithmic x-axis.

    # Returns
    - The `Plots.Plot` object for the generated figure.

    # Notes
    - Requires `μF_effective`, `μB`, `BvsI`, `GvsI`, `ħ`, `Ahfs`, `Ispin`,
    `γₙ`, and `γₑ` to be defined in scope.
"""
function plot_μeff(p::AtomParams, path_filename::AbstractString)  
    F_up = p.Ispin + 0.5
    mf_up = collect(F_up:-1:-F_up)
    F_down = p.Ispin - 0.5
    mf_down = collect(-F_down:1:F_down)
    dimF = Int(4*p.Ispin + 2)
        
    # Set color palette
    colorsF = palette(:phase, dimF)
    current_range = collect(0.00009:0.00002:1);

    # Initialize plot
    fig = plot(
        xlabel = L"Current ($\mathrm{A}$)",
        ylabel = L"$\mu_{F}/\mu_{B}$",
        legend = :right,
        background_color_legend = RGBA(0.85, 0.85, 0.85, 0.1),
        size = (800, 600),
    );

    # Define lines to plot: (F, mF, color index, style)
    lines_to_plot = vcat(
        [(F_up, mf, :solid) for mf in mf_up[1:end-1]],
        [(F_up, mf_up[end],:dash)],
        [(F_down, mf, :dash) for mf in mf_down],
    );

    # Plot all curves
    for ((f,mf,lstyle),color) in zip(lines_to_plot,colorsF)
        μ_vals = μF_effective.(current_range, f, mf, Ref(p)) ./ μB
        label = L"$F=%$(f)$, $m_{F}=%$(mf)$"
        plot!(fig,current_range, μ_vals, label=label, line=(color,lstyle, 2))
    end
        
    # Magnetic crossing point
    f(x) = BvsI(x) - 2π*ħ*p.Ahfs*(p.Ispin+1/2)/(2ħ)/(p.γn - γₑ)
    bcrossing = find_zero(f, (0.001, 0.02))

    # Annotated vertical line
    label_text = L"$I_{0} = %$(round(bcrossing, digits=5))\,\mathrm{A}$
     $\partial_{z}B_{z} = %$(round(GvsI(bcrossing), digits=2))\,\mathrm{T/m}$
     $B_{z} = %$(round(1e3 * BvsI(bcrossing), digits=3))\,\mathrm{mT}$"
    vline!(fig, [bcrossing], line=(:black, :dot, 2), label=label_text,xaxis = :log10,);
    
    display(fig)
    savefig(fig, path_filename)
    
    return nothing
end

    # CQD Equations of motion
    @inline function CQD_EqOfMotion(t,Ix,μ,r0::Vector{Float64},v0::Vector{Float64},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)

        x0, y0, z0 = r0
        v0x, v0y, v0z = v0

        # Key times
        tf1 =  default_y_FurnaceToSlit / v0y
        tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG) / v0y
        tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v0y
        # tF  = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG + default_y_SGToScreen) / v0y  # (unused here)

        cqd_sign = sign(θn-θe) 
        ωL       = abs(γₑ * BvsI(Ix) )
        acc_0    = μ*GvsI(Ix)/p.M
        kω       = cqd_sign*kx*ωL

        θe_half = θe / 2
        cosθ2 = cos(θe_half)^2
        sinθ2 = sin(θe_half)^2
        tanθ2 = tan(θe_half)^2

        if t <= tf1     # Furnace to Slit
            x = x0 + v0x*t 
            y = y0 + v0y*t 
            z = z0 + v0z*t
            vx , vy , vz = v0x , v0y , v0z
        elseif t <= tf2    # Slit to SG apparatus
            x = x0 + v0x*t 
            y = y0 + v0y*t
            z = z0 + v0z*t
            vx , vy , vz = v0x , v0y , v0z
        elseif t <= tf3   # Crossing the SG apparatus
            vx = v0x
            vy = v0y
            vz = v0z + acc_0*(t-tf2) + acc_0/kω * log( cosθ2 + exp(-2*kω*(t-tf2))*sinθ2 )
            x = x0 + v0x*t 
            y = y0 + v0y*t
            z = z0 + v0z*t + 0.5*acc_0*(t-tf2)^2 + acc_0/kω*log(cosθ2)*(t-tf2) + 0.5/(kω)^2 * acc_0 * ( polylogarithm(2,-exp(-2*kω*(t-tf2))*tanθ2) - polylogarithm(2,-tanθ2) )
        elseif t > tf3 # Travel to the Screen
            x = x0 + v0x*t
            y = y0 + v0y*t
            z = z0 + v0z*t + 0.5*acc_0*( (t-tf2)^2 - (t-tf3)^2) + acc_0/kω*default_y_SG/v0y * ( log(cosθ2) + v0y/default_y_SG*log(cosθ2+exp(-2*kω*default_y_SG/v0y)*sinθ2)*(t-tf3) ) + 0.5*acc_0/kω^2*( polylogarithm(2,-exp(-2*kω*default_y_SG/v0y)*tanθ2) - polylogarithm(2,-tanθ2) )
            vx = v0x
            vy = v0y
            vz = v0z + acc_0*default_y_SG/v0y + acc_0/kω*log(cosθ2 + exp(-2*kω*default_y_SG/v0y)*sinθ2)
        end

        r = SVector{3,Float64}(x, y, z)
        v = SVector{3,Float64}(vx, vy, vz)
        return r, v
    end

    # QM equations of motion
    @inline function QM_EqOfMotion(t,Ix,μ,r0::Vector{Float64},v0::Vector{Float64},θ::Float64, p::AtomParams)
        @assert length(r0) == 3 "r0 must have length 3"
        @assert length(v0) == 3 "v0 must have length 3"

        x0, y0, z0 = r0
        v0x, v0y, v0z = v0
        @assert v0y != 0.0 "y-velocity must be nonzero."
        @assert t >= 0 "time t must be ≥ 0"  # optional; remove if you allow negative t


        # Segment times (in seconds)
        tf1 =  default_y_FurnaceToSlit / v0y                               # slit entrance
        tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG) / v0y                 # SG entrance
        tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v0y          # SG exit
        tF  = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG + default_y_SGToScreen) / v0y  # screen

        acc_0    = μ*cos(θ)*GvsI(Ix)/p.M

        if t <= tf2     # Furnace to Slit and Slit to SG apparatus
            x = x0 + v0x*t 
            y = y0 + v0y*t 
            z = z0 + v0z*t
            vx , vy , vz = v0x , v0y , v0z
        elseif t <= tf3   # Crossing the SG apparatus
            vx = v0x
            vy = v0y
            vz = v0z + acc_0*(t-tf2)
            x = x0 + v0x*t 
            y = y0 + v0y*t
            z = z0 + v0z*t + 0.5*acc_0*(t-tf2)^2
        elseif t > tf3 # Travel to the Screen
            x = x0 + v0x*t
            y = y0 + v0y*t
            z = z0 + v0z*t + acc_0 * default_y_SG / v0y * (t - 0.5*(tf2+tf3))
            vx = v0x
            vy = v0y
            vz = v0z + acc_0*default_y_SG/v0y 
        end

        r = SVector{3,Float64}(x, y, z)
        v = SVector{3,Float64}(vx, vy, vz)
        return r, v
    end

    # CQD equations of motion only along the z-coordinate
    @inline function CQD_EqOfMotion_z(t,Ix::Float64,μ::Float64,r0::AbstractVector{Float64},v0::AbstractVector{Float64},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
        v0y = v0[2]
        v0z = v0[3]
        z0 = r0[3]
        
        tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG ) / v0y
        tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG ) / v0y

        cqd_sign = sign(θn-θe) 
        ωL       = abs( γₑ * BvsI(Ix) )
        acc_0    = μ*GvsI(Ix)/p.M
        kω       = cqd_sign*kx*ωL

        # Precompute angles
        θe_half = θe / 2
        tanθ = tan(θe_half)
        tanθ2 = tanθ^2
        cosθ2 = cos(θe_half)^2
        sinθ2 = sin(θe_half)^2
        log_cos2 = log(cosθ2)
        polylog_0 = polylogarithm(2, -tanθ2)

        if t <= tf2
            return z0 + v0z*t
        elseif t <= tf3   # Crossing the SG apparatus
            Δt = t - tf2
            exp_term = exp(-2 * kω * Δt)
            polylog_t = polylogarithm(2, -exp_term * tanθ2)

            return z0 + v0z*t + 0.5 * acc_0 * Δt^2 + acc_0 / kω * log_cos2 * Δt + 0.5 * acc_0 / kω^2 * ( polylog_t - polylog_0 )
        
        else # t > tf3 # Travel to the Screen
            Δt2 = t - tf2
            Δt3 = t - tf3
            τ_SG = default_y_SG / v0y
            exp_SG = exp(-2 * kω * τ_SG)
            polylog_SG = polylogarithm(2, -exp_SG * tanθ2)
            log_term = log(cosθ2 + exp_SG * sinθ2)

            return z0 + v0z*t + 0.5*acc_0*( Δt2^2 - Δt3^2 ) + acc_0 / kω * τ_SG * (log_cos2 + log_term * Δt3 / τ_SG) + 0.5 * acc_0 / kω^2 * (polylog_SG - polylog_0)
        end
    end

    @inline function QM_EqOfMotion_z(t,Ix::Float64,μ::Float64,r0::AbstractVector{Float64},v0::AbstractVector{Float64},θ::Float64, p::AtomParams)
        v0y = v0[2]
        v0z = v0[3]
        z0 = r0[3]
        
        tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG ) / v0y
        tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG ) / v0y

        acc_0    = μ*cos(θ)*GvsI(Ix)/p.M

        if t <= tf2
            return z0 + v0z*t
        elseif t <= tf3   # Crossing the SG apparatus
            Δt = t - tf2
            return  z0 + v0z*t + 0.5*acc_0*Δt^2
        else # t > tf3 # Travel to the Screen
            τ_SG = default_y_SG / v0y
            return z0 + v0z*t + acc_0 * τ_SG * (t - 0.5*(tf2+tf3))
        end
    end



    # CQD Screen position
    function CQD_Screen_position(Ix,μ::Float64,r0::AbstractVector{Float64},v0::AbstractVector{Float64},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)

        x0, y0, z0 = r0
        v0x, v0y, v0z = v0

        L1 = default_y_FurnaceToSlit 
        L2 = default_y_SlitToSG
        Lsg = default_y_SG
        Ld = default_y_SGToScreen
        Ltot = L1 + L2 + Lsg + Ld

        # Physics parameters
        cqd_sign = sign(θn-θe) 
        acc_0 = μ * GvsI(Ix) / p.M
        ωL = abs(γₑ * BvsI(Ix))
        kω = cqd_sign * kx * ωL

        # Common trig values
        θe_half = θe / 2
        cos2 = cos(θe_half)^2
        sin2 = sin(θe_half)^2
        tan2 = tan(θe_half)^2
        exp_term = exp(-2 * kω * Lsg / v0y)

        x = x0 + Ltot * v0x / v0y
        y = y0 + Ltot
        z = z0 + Ltot * v0z / v0y + 0.5*acc_0/v0y^2*((Lsg+Ld)^2-Ld^2) + acc_0/kω*Lsg/v0y*( log(cos2) + Ld/Lsg * log( cos2 + exp_term*sin2 ) ) + 0.5*acc_0/kω^2 * ( polylogarithm(2, -exp_term*tan2) - polylogarithm(2, -tan2) )
        return SVector(x,y,z)
    end


    function QM_Screen_position(Ix,μ::Float64,r0::AbstractVector{Float64},v0::AbstractVector{Float64},θ::Float64, p::AtomParams)

        x0, y0, z0 = r0
        v0x, v0y, v0z = v0

        # Geometry
        Lsg = default_y_SG
        Ld = default_y_SGToScreen
        Ltot = default_y_FurnaceToSlit  + default_y_SlitToSG + Lsg + Ld

        # Physics parameters
        acc_0 = μ * cos(θ) * GvsI(Ix) / p.M

        x = x0 + Ltot * v0x / v0y
        y = y0 + Ltot
        z = z0 + Ltot * v0z / v0y + 0.5*acc_0/v0y^2*((Lsg+Ld)^2-Ld^2)
        return SVector{3,Float64}(x, y, z)
    end

    # Generate samples post-filtering by the slit
    function _generate_samples_serial(No::Int, rng, p::EffusionParams; v_pdf::Symbol=:v3)
        @assert No > 0
        alive = Matrix{Float64}(undef, No, 6)
        iteration_count = 0
        count = 0

        # precompute a few constants
        hx = default_x_slit/2
        hz = default_z_slit/2
        epsvy = 1e-18

        @time while count < No
            iteration_count += 1

            # initial transverse position (uniform over furnace rectangle)
            x0 = default_x_furnace * (rand(rng) - 0.5)
            z0 = default_z_furnace * (rand(rng) - 0.5)

            if v_pdf === :v3
                v = AtomicBeamVelocity_v3(rng,p)
            elseif v_pdf === :v2
                v = AtomicBeamVelocity_v2(rng,p)
            else
                @warn "No Velocity PDF chosen, got $v_pdf"
                v = SVector{3,Float64}(0,800,0)
            end

            v0_x, v0_y, v0_z = v

            # avoid near-zero v_y
            if abs(v0_y) ≤ epsvy
                continue
            end

            x_at_slit = x0 + default_y_FurnaceToSlit * v0_x / v0_y
            z_at_slit = z0 + default_y_FurnaceToSlit * v0_z / v0_y

            if (abs(x_at_slit) <= hx) & (abs(z_at_slit) <= hz)
                count += 1
                @inbounds alive[count,:] =  [x0, 0.0, z0, v0_x, v0_y, v0_z]
            end
        end

        println("Total iterations: ", iteration_count)
        return alive
    end

    function _generate_samples_multithreaded(No::Int, base_seed::Int, p::EffusionParams; v_pdf::Symbol = :v3)
        alive = Matrix{Float64}(undef, No, 6)

        sample_count = Threads.Atomic{Int}(0)
        iteration_count = Threads.Atomic{Int}(0)

        # Precomputed constants
        hx = default_x_slit/2
        hz = default_z_slit/2
        epsvy = 1e-18

        @time Threads.@threads for thread_id in 1:Threads.nthreads()
            rng0 = TaskLocalRNG()
            Random.seed!(rng0, hash((base_seed, thread_id)))
            # rng0 = MersenneTwister(hash((base_seed, thread_id)))   

            while true
                Threads.atomic_add!(iteration_count, 1)

                x0 = default_x_furnace * (rand(rng0) - 0.5)
                z0 = default_z_furnace * (rand(rng0) - 0.5)

                # Velocity sample (zero-alloc SVector)
                if v_pdf === :v3
                    v = AtomicBeamVelocity_v3(rng0,p)
                elseif v_pdf === :v2
                    v = AtomicBeamVelocity_v2(rng0,p)
                else
                    @warn "No Velocity PDF chosen, got $v_pdf"
                    v = SVector{3,Float64}(0,800,0)
                end
                v0_x, v0_y, v0_z = v

                # Avoid divide-by-zero / huge times
                if abs(v0_y) ≤ epsvy
                    continue
                end

                x_at_slit = x0 + default_y_FurnaceToSlit * v0_x / v0_y
                z_at_slit = z0 + default_y_FurnaceToSlit * v0_z / v0_y

                if -hx ≤ x_at_slit ≤ hx && -hz ≤ z_at_slit ≤ hz
                    idx = Threads.atomic_add!(sample_count, 1)
                    if idx <= No
                        @inbounds alive[idx, :] = [x0, 0.0, z0, v0_x, v0_y, v0_z]
                    else
                        break
                    end
                end

            end
        end

        println("Total iterations: ", iteration_count[])
        return alive
    end

    function generate_samples(No::Int, p::EffusionParams; v_pdf::Symbol =:v3, rng = Random.default_rng(), multithreaded::Bool = false, base_seed::Int = 1234)
        if multithreaded
            return _generate_samples_multithreaded(No, base_seed, p; v_pdf=v_pdf)
        else
            return _generate_samples_serial(No, rng, p; v_pdf=v_pdf)
        end
    end

    function _generate_matched_pairs(No::Integer, rng; mode::Symbol = :total)
        @assert No > 0
        θes_up = Float64[]; θns_up = Float64[]
        θes_dn = Float64[]; θns_dn = Float64[]

        if mode === :total
            sizehint!(θes_up, No ÷ 2); sizehint!(θns_up, No ÷ 2)
            sizehint!(θes_dn, No ÷ 2); sizehint!(θns_dn, No ÷ 2)

            kept = 0
            while kept < No
                θe = 2asin(sqrt(rand(rng))); θn = 2asin(sqrt(rand(rng)))
                if θe < θn
                    push!(θes_up, θe); push!(θns_up, θn); kept += 1
                elseif θe > θn
                    push!(θes_dn, θe); push!(θns_dn, θn); kept += 1
                end
            end

        elseif mode === :bucket
            sizehint!(θes_up, No); sizehint!(θns_up, No)
            sizehint!(θes_dn, No); sizehint!(θns_dn, No)

            nup = 0; ndn = 0
            while (nup < No) || (ndn < No)
                θe = 2asin(sqrt(rand(rng))); θn = 2asin(sqrt(rand(rng)))
                if (θe < θn) && (nup < No)
                    push!(θes_up, θe); push!(θns_up, θn); nup += 1
                elseif (θe > θn) && (ndn < No)
                    push!(θes_dn, θe); push!(θns_dn, θn); ndn += 1
                end
            end

        else
            error("Unknown mode=$mode. Use :total or :bucket.")
        end

        return θes_up, θns_up, θes_dn, θns_dn
    end


    function _build_init_conditions(
        alive::AbstractMatrix{T},
        UPθe::AbstractVector{T}, UPθn::AbstractVector{T},
        DOWNθe::AbstractVector{T}, DOWNθn::AbstractVector{T};
        mode::Symbol = :total
    ) where {T<:Real}

        No = size(alive, 1)
        @assert length(UPθe)   == length(UPθn)   "UP θe/θn lengths must match"
        @assert length(DOWNθe) == length(DOWNθn) "DOWN θe/θn lengths must match"

        if mode === :total
            n_up = length(UPθe)
            n_dn = length(DOWNθe)
            @assert n_up + n_dn == No "In :total, n_up + n_dn must equal size(alive,1)"

            pairsUP   = Matrix{T}(undef, n_up, 8)
            pairsDOWN = Matrix{T}(undef, n_dn, 8)

            @inbounds @views begin
                # UP block: rows 1:n_up from `alive`
                for i in 1:n_up
                    pairsUP[i, 1:6] = alive[i, 1:6]
                    pairsUP[i, 7]   = UPθe[i]
                    pairsUP[i, 8]   = UPθn[i]
                end
                # DOWN block: rows (n_up+1):No from `alive`
                for j in 1:n_dn
                    i_alive = n_up + j
                    pairsDOWN[j, 1:6] = alive[i_alive, 1:6]
                    pairsDOWN[j, 7]   = DOWNθe[j]
                    pairsDOWN[j, 8]   = DOWNθn[j]
                end
            end

            return pairsUP, pairsDOWN

        elseif mode === :bucket
            @assert length(UPθe) == No == length(DOWNθe) "In :bucket, each θ list must have length No"

            pairsUP   = Matrix{T}(undef, No, 8)
            pairsDOWN = Matrix{T}(undef, No, 8)

            @inbounds @views for i in 1:No
                # UP
                pairsUP[i, 1:6] = alive[i, 1:6]
                pairsUP[i, 7]   = UPθe[i]
                pairsUP[i, 8]   = UPθn[i]
                if add_label; pairsUP[i, 9] = one(T); end
                # DOWN
                pairsDOWN[i, 1:6] = alive[i, 1:6]
                pairsDOWN[i, 7]   = DOWNθe[i]
                pairsDOWN[i, 8]   = DOWNθn[i]
                if add_label; pairsDOWN[i, 9] = zero(T); end
            end

            return pairsUP, pairsDOWN

        else
            error("Unknown mode=$mode. Use :total or :bucket.")
        end
    end


    function build_initial_conditions(No::Integer, alive::AbstractMatrix{T}, rng::AbstractRNG; mode::Symbol = :total) where {T<:Real}
    @assert No > 0 "No must be > 0"
    @assert No == size(alive,1) "Total number of particles $No"

    if mode === :total
        # Two-pass: count UP with a cloned RNG → allocate exact sizes → fill.
        @assert hasmethod(copy, Tuple{typeof(rng)}) "RNG must support copy() for two-pass mode"
        rng1 = copy(rng)
        n_up = 0
        @inbounds for _ in 1:No
            θe = T(2asin(sqrt(rand(rng1))))
            θn = T(2asin(sqrt(rand(rng1))))
            n_up += (θe < θn)
        end
        n_dn = No - n_up

        UP   = Matrix{T}(undef, n_up, 8)
        DOWN = Matrix{T}(undef, n_dn, 8)

        iu = 0; id = 0
        @inbounds @views for i in 1:No
            θe = T(2asin(sqrt(rand(rng))))
            θn = T(2asin(sqrt(rand(rng))))
            if θe < θn
                iu += 1
                UP[iu, 1:6] = alive[i, 1:6]
                UP[iu, 7]   = θe
                UP[iu, 8]   = θn
            else
                id += 1
                DOWN[id, 1:6] = alive[i, 1:6]
                DOWN[id, 7]   = θe
                DOWN[id, 8]   = θn
            end
        end
        return UP, DOWN

    elseif mode === :bucket
        # --- Single pass: preallocate No×8 for both; write angles as we generate.
        UP   = Matrix{T}(undef, No, 8)
        DOWN = Matrix{T}(undef, No, 8)
        nup = 0; ndn = 0
        @inbounds while (nup < No) || (ndn < No)
            θe = T(2asin(sqrt(rand(rng))))
            θn = T(2asin(sqrt(rand(rng))))
            if (θe < θn) && (nup < No)
                nup += 1
                UP[nup, 7] = θe
                UP[nup, 8] = θn
            elseif (θe > θn) && (ndn < No)
                ndn += 1
                DOWN[ndn, 7] = θe
                DOWN[ndn, 8] = θn
            end
        end
        # now copy alive rows once
        @inbounds @views for i in 1:No
            UP[i,   1:6] = alive[i, 1:6]
            DOWN[i, 1:6] = alive[i, 1:6]
        end
        return UP, DOWN

    else
        error("Unknown mode=$mode. Use :total or :bucket.")
    end

end

    export AtomParams,
            clear_all,
            polylog,
            compute_weights,
            FreedmanDiaconisBins,
            pixel_coordinates,
            EffusionParams, BeamEffusionParams,
            AtomicBeamVelocity_v3, AtomicBeamVelocity_v2,
            μF_effective, plot_μeff,
            generate_samples,
            CQD_EqOfMotion, QM_EqOfMotion,
            CQD_EqOfMotion_z, QM_EqOfMotion_z,
            CQD_Screen_position, QM_Screen_position,
            build_initial_conditions

end