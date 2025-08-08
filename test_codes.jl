#  Plotting Setup
using Plots; gr()
Plots.default(
    show=true, dpi=800, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
# Aesthetics and output formatting
using Colors, ColorSchemes
using LaTeXStrings, Printf, PrettyTables
# Time-stamping/logging
using Dates
# Numerical tools
using LinearAlgebra, DataStructures
using Interpolations, Roots, Dierckx, Loess, Optim
using BSplineKit
using WignerD, LambertW, PolyLog
using StatsBase
using Random, Statistics, NaNStatistics, MLBase, Distributions, StaticArrays
using Alert
# Data manipulation
using OrderedCollections
using DelimitedFiles, CSV, DataFrames, JLD2
# Custom modules
include("./Modules/atoms.jl");
include("./Modules/samplings.jl");
# include("./Modules/MyPolylogarithms.jl");
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Set the working directory to the current location
cd(dirname(@__FILE__)) 
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
# Timestamp start for execution timing
t_start = Dates.now()
# Random seeds
base_seed_set = 145;
# rng_set = MersenneTwister(base_seed_set)
rng_set = TaskLocalRNG()

println("\n\t\tRunning process on:\t $(Dates.format(t_start, "yyyymmddTHHMMSS")) \n")
# Generate a timestamped directory name for output (e.g., "20250718T153245")
directoryname = Dates.format(t_start, "yyyymmddTHHMMSS") ;
# Construct the full directory path (relative to current working directory)
dir_path = "./simulation_data/$(directoryname)" ;
# Create the directory (and any necessary parent folders)
mkpath(dir_path) ;
@info "Created output directory" dir = dir_path

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
atom_info       = AtomicSpecies.atoms(atom);
const R         = atom_info[1];
const μₙ        = atom_info[2];
const γₙ        = atom_info[3];
const Ispin    = atom_info[4];
const Ahfs     = atom_info[6];
const M        = atom_info[7];
const ki = 2.1e-6

# STERN--GERLACH EXPERIMENT
# Image size
const cam_pixelsize = 0.0065 ;  # [mm]
n_bins = 4
exp_pixelsize = n_bins * cam_pixelsize ;   # [mm] for 20243014
# Furnace
const T = 273.15 + 200 ; # Furnace temperature (K)
# Furnace aperture
const x_furnace = 2.0e-3 ;
const z_furnace = 100e-6 ;
# Slit
const x_slit  = 4.0e-3 ;
const z_slit  = 300e-6 ;
# Propagation distances
const y_FurnaceToSlit = 224.0e-3 ;
const y_SlitToSG      = 44.0e-3 ;
const y_SG            = 7.0e-2 ;
const y_SGToScreen    = 32.0e-2 ;
# Sample size: number of atoms arriving to the screen
const Nss = 5000

II = Ispin
    F_up = II + 0.5
    mf_up = collect(F_up:-1:-F_up)
    F_down = II - 0.5
    mf_down = collect(-F_down:1:F_down)
    dimF = Int8(4*II + 2)
    
    # Set color palette
    color8 = palette(:rainbow, dimF)
    current_range = collect(0.00009:0.00002:1);

    # Initialize plot
    fig = plot(
        xlabel = L"Current ($\mathrm{A}$)",
        ylabel = L"$\mu_{F}/\mu_{B}$",
        xaxis = :log10,
        legend = :right,
        background_color_legend = RGBA(0.85, 0.85, 0.85, 0.1),
        size = (800, 600),
    )

    # Define lines to plot: (F, mF, color index, style)
    lines_to_plot = vcat(
        [(F_up, mf, :solid) for mf in mf_up[1:end-1]],
        [(F_up, mf_up[end],:dash)]
        [(F_down, mf, :dash) for mf in mf_down],
    )

    # Plot all curves
    for i=1:dimF
        μ_vals = μF_effective.(current_range, II, lines_to_plot[], mF) ./ μB

    end

    for (F, mF, cidx, lstyle) in lines_to_plot
        μ_vals = μF_effective.(Irange, Ispin, F, mF) ./ μB
        label = L"\$F=$(F)\$, \$m_{F}=$(mF >= 0 ? "+$mF" : "$mF")\$"
        plot!(Irange, μ_vals, label=label, line=(color8[cidx], lstyle, 2))
    end

    # Magnetic crossing point
    f(x) = BvsI(x) - 2π*ħ*Ahfs*(Ispin+1/2)/(2ħ)/(γₙ - γₑ)
    bcrossing = find_zero(f, (0.001, 0.02))

    # Annotated vertical line
    label_text = L"\$I_{0} = %$(round(bcrossing, digits=5))\,\mathrm{A}$\n" *
                 L"\partial_{z}B_{z} = %$(round(GvsI(bcrossing), digits=2))\,\mathrm{T/m}$\n" *
                 L"\$B_{z} = %$(round(1e3 * BvsI(bcrossing), digits=3))\,\mathrm{mT}$"
    vline!([bcrossing], line=(:black, :dot, 2), label=label_text)

    display(fig)
    if savepath !== nothing
        savefig(fig, savepath)
    end
    return fig
end
