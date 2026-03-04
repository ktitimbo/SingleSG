# Simulation of atom trajectories in the Stern–Gerlach experiment
# Kelvin Titimbo
# California Institute of Technology
# February 2026

#  Plotting Setup
# ENV["GKS_WSTYPE"] = "100"
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
T_K = 273.15 + 205 ; # Furnace temperature (K)
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


x  = 0:0.1:10
y1 = sin.(x)
y2 = 100 .* cos.(x)

p = plot(x, y1;
    label="y1",
    xlabel="x",
    ylabel="sin(x)",
    legend=:topright,
)
plot!(twinx(p), x, y2;
    label="y2",
    ylabel="100 cos(x)",
)

p



ki_6M = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 
        3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 
        4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 
        5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 
        6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 
        7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 
        8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 
        9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 
        100.0, 1000.0, 10000.0, 100000.0]

ki_fit = 2.4e-6

cqd_ki_path = joinpath("Y://SternGerlach//simulations/cqd_simulation_6M","up","cqd_6000000_ki034_up_screen.jld2");
cqd_ki = load(cqd_ki_path,"screen")[:data]

Ic = Icoils[2:end]

number_precessions = round(1/(TWOπ*ki_fit))
collapse_time = inv.(ki_fit * abs(γₑ)* TheoreticalSimulation.BvsI.(Ic))

plot(Ic,1e6*collapse_time,
    label=L"Collapse time $\tau_{c}$",
    line=(:blue,2),
    xlabel="Current (A)",
    ylabel="Collapse time (μs)",
    xaxis=:log10,
    yaxis=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1, 10, 100], [L"10^{0}", L"10^{1}", L"10^{2}"]),
    xlims=(1e-2,1),
    ylims=(1,200)
)

travel_times = [mean(inv.(cqd_ki[ic][:,5] / y_SG)) for ic=2:47]
plot(Ic,
    1e6*travel_times,
    ribbon=1e6*[standard_error(inv.(cqd_ki[ic][:,5] / y_SG)) for ic=2:47],
    label=L"Time of flight $\Delta t_{\mathrm{SG}}$",
    line=(:solid,:dodgerblue3,2),
    marker=(:circle,:white,2),
    markerstrokecolor=:dodgerblue3,
    xlabel="Current (A)",
    ylabel="Time of flight (μs)",
    xaxis=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(1e-2,1),
    fillalpha=0.1,
)
@info @sprintf("The mean time of flight is (%d ± %d) μs ",mean(1e6*travel_times),std(1e6*travel_times))




plot(Ic,
    1e6*travel_times,
    ribbon=1e6*[standard_error(inv.(cqd_ki[ic][:,5] / 0.07)) for ic=2:47],
    line=(:solid,:dodgerblue3,2),
    label=L"Time of flight $\Delta t_{\mathrm{SG}}$",
    xaxis=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(1e-2,1),
    fillalpha=0.1,
)
plot!(Ic,1e6*collapse_time,
    label=L"Collapse time $\tau_{c}$",
    line=(:darkgreen,2),
    xlabel="Current (A)",
    ylabel="Time (μs)",
    xaxis=:log10,
    yaxis=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1, 10, 100], [L"10^{0}", L"10^{1}", L"10^{2}"]),
    xlims=(1e-2,1),
    ylims=(1,200),
    legend=:bottomleft
)

plot(Ic,travel_times ./ collapse_time,
    label=L"$\Delta t_{\mathrm{SG}} / \tau_{c}$",
    line=(:red,2),
    xaxis=:log10,
    yaxis=:log10,
    xlims=(1e-2,1),
    ylims=(0.7,40),
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlabel="Current (A)",
    ylabel="Number of collapse times",
    size=(800,600),
    left_margin=3mm,
    legendfontsize=12,
)
hspan!([0.1,1], color=:black, fillalpha=0.2, label=false)

plot(Ic, inv.(travel_times ./ collapse_time),
    label=L"$\tau_{c} / \Delta t_{\mathrm{SG}} $",
    line=(:red,2),
    xaxis=:log10,
    yaxis=:log10,
    xlims=(1e-2,1),
    # ylims=(0.7,40),
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlabel="Current (A)",
    ylabel= nothing,
    size=(800,600),
    left_margin=3mm,
    legendfontsize=12,
)


###########################################################################################
#******************************************************************************************
#+++++++++++++++++++++ Magnetic field measurements ++++++++++++++++++++++++++++++++++++++++
"""
    vector_subset(v::AbstractVector{T}; thr::Real = zero(T)) where {T<:Real}
        -> (mask, inds, values_view)

Return the subset of elements of `v` that are greater than or equal to a
given threshold `thr`, together with the corresponding boolean mask and indices.

This function is useful when filtering numerical data while preserving
alignment across multiple related vectors (e.g. values, uncertainties,
coordinates, weights).

# Arguments
- `v::AbstractVector{T}`:
    Input numeric vector.
- `thr::Real = zero(T)` (keyword):
    Threshold value. Elements satisfying `v[i] ≥ thr` are selected.
    The threshold is internally converted to the element type `T`.

# Returns
A tuple `(mask, inds, values_view)` where:

- `mask::BitVector`:
    Boolean mask of length `length(v)` with `true` at positions where
    `v[i] ≥ thr`.
- `inds::Vector{Int}`:
    Indices `i` such that `v[i] ≥ thr`.
- `values_view`:
    A `SubArray` view of the selected elements `v[mask]`
    (no data copy is performed).

# Notes
- The threshold is converted to type `T` via `convert(T, thr)` to ensure
  type stability and consistent comparisons.
- Returning a mask is often preferable when multiple vectors must be
  filtered consistently:
  julia
  mask, inds, vals = vector_subset(v; thr=0.1)
  x_filt  = x[mask]
  σx_filt = σx[mask]
- The returned values are provided as a view (@view v[mask]) to avoid
unnecessary memory allocation.
"""
function vector_subset(v::AbstractVector{T}; thr::Real = zero(T), include_equal::Bool = true) where {T<:Real}
    thrT = convert(T, thr)  # or: thrT = T(thr)
    if include_equal
        mask = v .>= thrT
    else
        mask = v .> thrT
    end
    inds = findall(mask)
    return mask, inds, @view v[mask]
end


"""
    subset_by_cols(A::AbstractMatrix{T}, cols::AbstractVector{<:Integer};
                        thr::Real = zero(T), include_equal::Bool = true)
        -> (mask, inds, rows_view)

Return rows of `A` for which all selected columns satisfy the threshold condition.

A row `i` is selected if, for every `c ∈ cols`:

    A[i, c] ≥ thr   (default)
or
    A[i, c] > thr   (if `include_equal=false`)

# Arguments
- `A::AbstractMatrix{T}`:
    Input matrix (`N×s`).
- `cols`:
    Vector of column indices to test.
- `thr::Real = zero(T)`:
    Threshold value (converted internally to type `T`).
- `include_equal::Bool = true`:
    If true use `≥`, otherwise use `>`.

# Returns
- `mask::BitVector` of length `N`
- `inds::Vector{Int}` selected row indices
- `rows_view` a view `A[mask, :]` (no copy)

# Example
julia
A = [0.1  2.0  0.5  4.0;
     0.8  1.0  0.7  2.0;
     0.9  0.3  0.2  1.0]

mask, inds, rows = subset_by_cols(A, [1,3]; thr=0.6)
# Only row 2 satisfies:
# A[2,1]=0.8 ≥ 0.6 AND A[2,3]=0.7 ≥ 0.6
"""
function subset_by_cols(A::AbstractMatrix{T},
        cols::AbstractVector{<:Integer};
        thr::Real = zero(T),
        include_equal::Bool = true) where {T<:Real}
    thrT = convert(T, thr)

    subA = @view A[:, cols]

    if include_equal
        mask = vec(all(subA .>= thrT, dims=2))
    else
        mask = vec(all(subA .> thrT, dims=2))
    end

    inds = findall(mask)

    return mask, inds, @view(A[mask, :])

end


data_directories = ["20260220", "20260225", "20260226am","20260226pm","20260227", "20260303"]
no = length(data_directories);
colores = palette(:darkrainbow, no);

exp_data = Vector{Matrix{Float64}}(undef, no)
for (idx, dir) in enumerate(data_directories)
    data = load(joinpath(@__DIR__,dir,"data_processed.jld2"),"data")
    Ic = data[:Currents]
    Bz = data[:BzTesla]
    exp_data[idx] = hcat(Ic,Bz)
end

fig1 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)");
for (idx, dir) in enumerate(data_directories)
    data = exp_data[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig1, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
    )

end
plot!(fig1,Icoils[2:end], 1e3*TheoreticalSimulation.BvsI.(Icoils)[2:end],
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig1,
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
display(fig1)

fig2 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)");
for (idx, dir) in enumerate(data_directories)
    data = exp_data[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig2, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
    )

end
plot!(fig2,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig2,
    # xscale=:log10,
    # yscale=:log10,
    # xlims=(1e-3,1.02),
    # ylims=(1e-2,1e3),
    # xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # yticks = ([0.01, 0.1, 1, 10, 100], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig2)

fig3 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)");
for (idx, dir) in enumerate(data_directories)
    data = exp_data[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig3, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
    )

end
plot!(fig3,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig3,
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
display(fig3)


BvsI_comparison = Vector{Matrix{Float64}}(undef, no)
for (idx, dir) in enumerate(data_directories)
    vs = exp_data[idx]

    mask, inds, rows_view = subset_rows_by_cols(vs, [1,2]; thr=1e-3, include_equal = true)
    BvsI_comparison[idx] = Matrix(rows_view)  # store a copy; or keep the view (see B)
end


fig4 = plot(xlabel="Currents (A)",
    ylabel=L"$B_{\mathrm{exp}} / B_{\mathrm{manual}}$")
for (idx, dir) in enumerate(data_directories)
    B_ratio = BvsI_comparison[idx][:,2] ./ TheoreticalSimulation.BvsI.(BvsI_comparison[idx][:,1])
    
    plot!(fig4,  BvsI_comparison[idx][:,1] , B_ratio,
        label=data_directories[idx],
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:solid,1,colores[idx])
    )
    y0 , σ0 = mean(B_ratio), standard_error(B_ratio)

    x = range(1e-3, 1.1, length=200)
    plot!(fig4, x, fill(y0, length(x)),
     ribbon = σ0,
     color = colores[idx],
     fillalpha = 0.25,
     line=(:dash,0.5,colores[idx]),
     label = "$(round(y0; digits=3)) ± $(round(σ0; sigdigits=1))")
    # hline!([y0], ine=(:dot,0.5,colores[idx]), label= "")
end
plot!(fig4,
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
)
display(fig4)

exp_data_corr = Vector{Matrix{Float64}}(undef, no)
for (idx, dir) in enumerate(data_directories)
    exp_data_corr[idx] = hcat(exp_data[idx][:,1] , exp_data[idx][:,2] .- exp_data[idx][1,2] )
end

fig5 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    title="Shifted B field");
for (idx, dir) in enumerate(data_directories)
    data = exp_data_corr[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig5, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
    )

end
plot!(fig5,Icoils[2:end], 1e3*TheoreticalSimulation.BvsI.(Icoils)[2:end],
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig5,
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
display(fig5)

fig6 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    title="Shifted B field");
for (idx, dir) in enumerate(data_directories)
    data = exp_data_corr[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig6, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
    )

end
plot!(fig6,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig6,
    # xscale=:log10,
    # yscale=:log10,
    # xlims=(1e-3,1.02),
    # ylims=(1e-2,1e3),
    # xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # yticks = ([0.01, 0.1, 1, 10, 100], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    legend=:bottomright,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
);
display(fig6)

fig7 = plot(xlabel="Currents (A)",
    ylabel="Magnetic field (mT)",
    title="Shifted B field");
for (idx, dir) in enumerate(data_directories)
    data = exp_data_corr[idx]
    Ic = data[:,1]
    Bz = data[:,2]

    plot!(fig7, Ic, 1000*Bz,
        label="$(dir) (degauss = $(round(1e3*Ic[1]; digits=6))mA , $(Int(round(1e4*Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:dot, colores[idx], 1)
    )

end
plot!(fig7,Icoils, 1e3*TheoreticalSimulation.BvsI.(Icoils),
    label="SG manual",
    line=(:solid,2,:black));
plot!(fig7,
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
display(fig7)

BvsI_comparison_corr = Vector{Matrix{Float64}}(undef, no)
for (idx, dir) in enumerate(data_directories)
    vs = exp_data_corr[idx]

    mask, inds, rows_view = subset_rows_by_cols(vs, [1,2]; thr=1e-3, include_equal = true)
    BvsI_comparison_corr[idx] = Matrix(rows_view)  # store a copy; or keep the view (see B)
end


fig8 = plot(xlabel="Currents (A)",
    ylabel=L"$B_{\mathrm{exp}} / B_{\mathrm{manual}}$",
    title="Shifted B field")
for (idx, dir) in enumerate(data_directories)
    B_ratio = BvsI_comparison_corr[idx][:,2] ./ TheoreticalSimulation.BvsI.(BvsI_comparison_corr[idx][:,1])
    
    plot!(fig8,
        BvsI_comparison_corr[idx][:,1] , B_ratio,
        label=data_directories[idx],
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:solid,1,colores[idx])
    )
    y0 , σ0 = mean(B_ratio), standard_error(B_ratio)

    x = range(1e-3, 1.1, length=200)
    plot!(fig8,
    x, fill(y0, length(x)),
    ribbon = σ0,
    color = colores[idx],
    fillalpha = 0.25,
    line=(:dash,0.5,colores[idx]),
    label = "$(round(y0; digits=3)) ± $(round(σ0; sigdigits=1))")
end
plot!(fig8,legend=:bottomright,
    xscale=:log10,
    yscale=:log10,
    xlims=(1e-3,1.05),
    ylims=(3e-1,2),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.01, 0.1, 1, 10], [L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
)
display(fig8)


plot(fig1,fig5,
layout=(1,2),
size=(1000,600),
left_margin=5mm,
bottom_margin=2mm,
)

plot(fig2,fig6, fig3,fig7,
layout=(2,2),
size=(1000,600),
left_margin=5mm,
bottom_margin=2mm,
)

plot(fig4,fig8,
layout=(1,2),
size=(1000,600),
left_margin=5mm,
bottom_margin=2mm,
)



########################################################################################
#+++++++++++++++++++++++++++ Kahraman & Ku +++++++++++++++++++++++++++++++++++++++++++++
data_directories = ["20260220", "20260225", "20260226am","20260226pm","20260227", "20260303"]
bzsign = [-1,1,1,-1,1,1]
colores = palette(:darkrainbow,6)

fig = plot(
    xlabel="Currents (mA)",
    ylabel=L"$F=1$ Peak position (mm)"
);
for (idx,dir) in enumerate(data_directories)
    kk_path = joinpath(@__DIR__, "analysis_data","summary", dir, dir * "_report_summary.jld2")
    data = jldopen(kk_path, "r") do file
        ic = file["meta/Currents"]
        bz = file["meta/BzTesla"]

        dd = file[JLD2_MyTools.make_keypath_exp(dir,2,0.01)]
        return ( Ic=ic, Bz=bz, F1 = dd[:fw_F1_peak_pos_raw], F2 = dd[:fw_F2_peak_pos_raw], C0 = dd[:centroid_fw_mm] )
    end

    plot!(fig,
        1000*data.Ic, data.F1,
        label="$(dir) (degauss = $(round(1e3*data.Ic[1]; digits=6))mA , $(Int(round(1e4*bzsign[idx]*data.Bz[1])))G )",
        marker=(:circle, 2, :white),
        markerstrokecolor=colores[idx],
        line=(:solid,1,colores[idx])
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
    xlims=(0,25),
    ylims=(8.83,8.90),
    legendfontsize=6,
    foreground_color_legend = nothing,
)
plot!(fig,
    xlims=(750,1010),
    ylims=(10.30,10.90),
    legendfontsize=6,
    foreground_color_legend = nothing,
)

JLD2_MyTools.show_exp_summary(kk_path, dir)




data_directories = ["20260220","20260225","20260226am","20260226pm","20260227","20260303"]
dir = data_directories[6]
bzsign = [-1,1,1,-1,1,1]
colores = palette(:darkrainbow,6)
kk_path = joinpath(@__DIR__, "analysis_data","summary", dir, dir * "_report_summary.jld2")

data = jldopen(kk_path, "r") do file
    dd = file[JLD2_MyTools.make_keypath_exp(dir,2,0.01)]
end

data[:fw_F1_peak_pos_raw]
data[:fw_F2_peak_pos_raw]



