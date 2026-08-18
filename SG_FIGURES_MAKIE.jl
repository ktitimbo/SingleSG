# PREPARATION OF FIGURES FOR FINAL MANUSCRIPT
# Kelvin Titimbo
# California Institute of Technology
# JULY 2026

#  Plotting Setup
# ENV["GKS_WSTYPE"] = "100"
using CairoMakie
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
const BASE_PATH = raw"F:\SternGerlachExperiments";
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
TheoreticalSimulation.DEFAULT_camera_pixel_size = cam_pixelsize;
TheoreticalSimulation.DEFAULT_x_pixels          = nx_pixels;
TheoreticalSimulation.DEFAULT_z_pixels          = nz_pixels;
TheoreticalSimulation.DEFAULT_x_furnace         = x_furnace;
TheoreticalSimulation.DEFAULT_z_furnace         = z_furnace;
TheoreticalSimulation.DEFAULT_x_slit            = x_slit;
TheoreticalSimulation.DEFAULT_z_slit            = z_slit;
TheoreticalSimulation.DEFAULT_y_FurnaceToSlit   = y_FurnaceToSlit;
TheoreticalSimulation.DEFAULT_y_SlitToSG        = y_SlitToSG;
TheoreticalSimulation.DEFAULT_y_SG              = y_SG;
TheoreticalSimulation.DEFAULT_y_SGToScreen      = y_SGToScreen;
TheoreticalSimulation.DEFAULT_R_tube            = R_tube;
TheoreticalSimulation.DEFAULT_c_aperture        = R_aper;
TheoreticalSimulation.DEFAULT_y_SGToAperture    = y_SGToAperture;
##################################################################################################


##################################################################################################
##################################################################################################
## STERN--GERLACH GEOMETRY
x_line = range(-10e-3, 10e-3; length=10_001)
x_mm   = 1e3 .* x_line

fig = Figure(figure_padding = (10, 10, 10, 10))  # (left, right, bottom, top)
ax  = Axis(fig[1, 1];
    xlabel         = L"x \ (\mathrm{mm})",
    ylabel         = L"z \ (\mathrm{mm})",
    # title          = "Stern–Gerlach Slit Geometry",
    xlabelsize     = 20,
    ylabelsize     = 20,
    xticklabelsize = 16,
    yticklabelsize = 16,
    limits         = (-8, 8, -4, 7),
    xticks         = -8:2:8,
    yticks         = -4:1:7,
    xtickformat    = xs -> [L"%$(Int(round(Int, x)))" for x in xs],
    ytickformat    = ys -> [L"%$(Int(round(Int, y)))" for y in ys],
    aspect         = DataAspect(),
)

# Filled polygon: top magnet edge → fill to +∞ (capped at +10 mm)
y_edge = 1e3 .* TheoreticalSimulation.z_magnet_edge.(x_line)
y_top  = fill(10.0, length(x_mm))
poly!(ax, Point2f.(vcat(x_mm, reverse(x_mm)), vcat(y_edge, reverse(y_top)));
    color = (RGBf(10/255, 10/255, 200/255), 0.85), strokecolor = RGBf(10/255, 10/255, 200/255), strokewidth = 2,
    # label = "Rounded edge",
)

# Filled polygon: bottom trench → fill to −∞ (capped at −10 mm)
y_trench = 1e3 .* TheoreticalSimulation.z_magnet_trench.(x_line)
y_bottom = fill(-10.0, length(x_mm))
poly!(ax, Point2f.(vcat(x_mm, reverse(x_mm)), vcat(y_bottom, reverse(y_trench)));
    color = (RGBf(200/255, 10/255, 10/255), 0.85), strokecolor = RGBf(200/255, 10/255, 10/255), strokewidth = 2,
    # label = "Trench",
)

# Horizontal band representing the slit opening extent in z
hspan!(ax, -3.0, 3.0; color = (:gray36, 0.55))

# Slit rectangle (closed polygon, 5 vertices) — drawn on top of hspan
hw_x = 1e3 * x_slit / 2
hw_z = 1e3 * z_slit / 2
poly!(ax, Point2f.(
        [-hw_x, -hw_x, hw_x,  hw_x, -hw_x],
        [-hw_z,  hw_z, hw_z, -hw_z, -hw_z]);
    color = (:white, 0.99), strokecolor = :black, strokewidth = 1.5,
    linestyle = :solid,
    # label = "Slit",
)
# axislegend(ax; position = :rb)
resize_to_layout!(fig)
display(fig)
save(joinpath(OUTDIR, "SG_geometry.png"), fig; px_per_unit = 2)
save(joinpath(OUTDIR, "SG_geometry.pdf"), fig; px_per_unit = 2)

##################################################################################################
##################################################################################################
## POTASSIUM 39 - ZEEMAN DIAGRAM

function clipped_cmap(scheme::Symbol, n::Int; lo=0.25, hi=1.0)
    # Sample only the [lo, hi] portion of the colormap, skipping the
    # washed-out/near-white end that sits below `lo`.
    return get(colorschemes[scheme], range(lo, hi, length=n))
end
# colors_up   = clipped_cmap(:solar,  length(mf_up);   lo=0.3)
# colors_down = clipped_cmap(:deep, length(mf_down); lo=0.3)
# colorsF = vcat(colors_up, colors_down)

function anchored_cmap(c1, c2, n)
    g = cgrad([c1, c2])
    return [g[t] for t in range(0.0, 1.0, length=n)]
end

function anchored_cmap3(c1, c2, c3, n)
    g = cgrad([c1, c2, c3])
    return [g[t] for t in range(0.0, 1.0, length=n)]
end

F_up   = K39_params.Ispin + 0.5
F_down = K39_params.Ispin - 0.5
mf_up   = F_up:-1.0:-F_up
mf_down = -F_down:1.0:F_down
dimF = Int(4*K39_params.Ispin + 2)

colors_up   = anchored_cmap3(colorant"darkred", colorant"orangered", colorant"chocolate4", length(mf_up))
colors_down = anchored_cmap(colorant"navy", colorant"deepskyblue3", length(mf_down))
colorsF = vcat(colors_up, colors_down)

current_range = exp10.(range(log10(0.0009), log10(1.01), length=600))

ZM_LABEL_SIZE       = 24   # axis label font size
ZM_TICK_LABEL_SIZE  = 18   # tick number font size
ZM_LEGEND_LABEL_SIZE = 16  # legend entry font size

fig = Figure(size = (1120, 600))

# ── Top axis created FIRST so its gridlines render behind the data ──────
ax_top = Axis(fig[1, 1];
    xlabel         = L"Magnetic field $B_z\ (\mathrm{T})$",
    xlabelsize     = ZM_LABEL_SIZE,
    xticklabelsize = ZM_TICK_LABEL_SIZE,
    xaxisposition  = :top,
    xscale         = log10,
    xtickalign          = 0.5,
    xminortickalign     = 0.5,
    xticksize           = 10,
    xminorticksize      = 5,
    backgroundcolor    = :transparent,
    xminorticksvisible = true,
    xgridvisible       = false,
    xminorgridvisible  = false,
    xminorticks        = IntervalsBetween(9),
)
hideydecorations!(ax_top)
hidespines!(ax_top, :l, :r, :b)

# ── Bottom axis: current (data lives here) ───────────────────────────────
ax = Axis(fig[1, 1];
    xlabel         = L"Current $(\mathrm{A})$",
    ylabel         = L"$\mu_{F}/\mu_{B}$",
    xlabelsize     = ZM_LABEL_SIZE,
    ylabelsize     = ZM_LABEL_SIZE,
    xticklabelsize = ZM_TICK_LABEL_SIZE,
    yticklabelsize = ZM_TICK_LABEL_SIZE,
    xtickformat    = xs -> [L"10^{%$(round(Int, log10(x)))}" for x in xs],
    # ytickformat    = ys -> [isinteger(y) ? L"%$(Int(y))" : L"%$(y)" for y in ys],
    ytickformat    = ys -> [iszero(y) ? L"0" : L"%$(round(y, digits=1))" for y in ys],
    xtickalign          = 0.5,
    ytickalign          = 0.5,
    xminortickalign     = 0.5,
    yminortickalign     = 0.5,
    xticksize           = 10,
    yticksize           = 10,
    xminorticksize      = 5,
    yminorticksize      = 5,
    xscale             = log10,
    backgroundcolor    = :transparent,
    xminorticksvisible = true,
    xgridvisible       = false,
    xminorgridvisible  = false,
    xminorticks        = IntervalsBetween(9),
)

LONG_DASH = Linestyle([0.0, 8.0, 10.0])  # 0→12 on, 12→18 off, period=18

lines_to_plot = vcat(
    [(F_up, mf, :solid) for mf in mf_up[1:end-1]],
    [(F_up, mf_up[end], :dash)],
    [(F_down, mf, :dash) for mf in mf_down],
)

for ((f, mf, lstyle), color) in zip(lines_to_plot, colorsF)
    μ_vals = TheoreticalSimulation.μF_effective.(current_range, f, mf, Ref(K39_params)) ./ μB
    mf_str = mf > 0 ? "+$(Int(mf))" : "$(Int(mf))"
    label  = L"$F=%$(Int(f))$, $m_{F}=%$(mf_str)$"
    actual_style = lstyle === :dash ? LONG_DASH : lstyle
    lines!(ax, current_range, μ_vals; label=label, color=color, linestyle=actual_style, linewidth=3.0)
end

# ── Magnetic crossing point ────────────────────────────────────────
B_cross_target = 2π * K39_params.Ahfs * (K39_params.Ispin + 0.5) / (K39_params.γn - γₑ)
f_cross(x)     = TheoreticalSimulation.BvsI(x) - B_cross_target
I₀             = find_zero(f_cross, (0.001, 0.050))

@info "Magnetic crossing point" I₀_mA=round(1000*I₀, digits=3) ∂zBz_Tperm=round(TheoreticalSimulation.GvsI(I₀), digits=2) Bz_mT=round(1e3 * TheoreticalSimulation.BvsI(I₀), digits=3)

vlines!(ax, [I₀]; color=:black, linestyle=:dot, linewidth=2.5)

axislegend(ax; position=:rc, backgroundcolor=:white, patchsize=(60, 2), labelsize=ZM_LEGEND_LABEL_SIZE)
xlims!(ax, current_range[1], current_range[end])

linkxaxes!(ax, ax_top)

# ── Build B-axis ticks at pure decades (10^n T) ───────────────────────────
B_lo, B_hi = extrema((TheoreticalSimulation.BvsI(current_range[1]),
                       TheoreticalSimulation.BvsI(current_range[end])))
decade_lo, decade_hi = floor(Int, log10(B_lo)), ceil(Int, log10(B_hi))
candidate_B = [10.0^d for d in decade_lo:decade_hi]

tick_I = Float64[]
tick_labels = Vector{LaTeXString}()
for b in candidate_B
    try
        I_at_b = find_zero(x -> TheoreticalSimulation.BvsI(x) - b,
                            (current_range[1], current_range[end]))
        push!(tick_I, I_at_b)
        push!(tick_labels, L"10^{%$(round(Int, log10(b)))}")
    catch
        # skip if root-finding fails (non-monotonic or out of bracket)
    end
end

ax_top.xticks = (tick_I, tick_labels)

display(fig)
save(joinpath(OUTDIR, "SG_mm_effective.png"), fig; px_per_unit = 3)
save(joinpath(OUTDIR, "SG_mm_effective.pdf"), fig; px_per_unit = 3)

##################################################################################################
##################################################################################################
## Experimental pattern

experiment_path = joinpath(BASE_PATH,"EXPERIMENTS","20260220","data_processed.jld2");

function print_experiment_table(filepath::AbstractString)
    data = load(filepath, "data");

    currents      = vec(data[:Currents])
    currents_err  = vec(data[:CurrentsError])
    bz_tesla      = 1000* vec(data[:BzTesla]) # mT

    lengths = length.((currents, currents_err, bz_tesla))
    all(==(lengths[1]), lengths) ||
        throw(DimensionMismatch("Columns have different lengths: $lengths"))

    table = hcat(currents, currents_err, bz_tesla);

    pretty_table(
        table;
        title         = joinpath(splitpath(filepath)[end-1:end]...),
        formatters    = [fmt__printf("%8.4f", [1]), fmt__printf("%8.4f", [2]), fmt__printf("%8.4f", [3])],
        alignment     = :c,
        column_labels  = [
            ["I0 Current", "I0 CurrentError", "Bz field"], 
            ["[A]", "[A]", "[mT]"]
        ],
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style = TextTableStyle(
                    first_line_column_label = crayon"yellow bold",
                    column_label  = crayon"yellow",
                    table_border  = crayon"blue bold",
                    title = crayon"bold red"
                    ),
        equal_data_column_widths = true,
        show_row_number_column = true,
        row_number_column_label = "No.",
        row_number_column_alignment = :c,
    )
    return data
end

exp_data = print_experiment_table(experiment_path);

nI_idx = 19

F1_mean = dropdims(
    mean(@view(exp_data[:F1ProcessedImages][:, :, :, nI_idx]), dims=3),
    dims=3,
)

F2_mean = dropdims(
    mean(@view(exp_data[:F2ProcessedImages][:, :, :, nI_idx]), dims=3),
    dims=3,
)

function plot_heatmap_with_profile(
        data;
        colormap    = :viridis,
        colorrange  = extrema(filter(isfinite, data)),
        size        = (600, 550),
        cb_label    = "Mean intensity",
        profile_label = "Intensity (arb. units)",
        aspect      = 0.5,   # width:height ratio, e.g. 0.5 → twice as tall as wide
)
    fig = Figure(; size=size, backgroundcolor=:white)

    layout = GridLayout(fig[1, 1])

    x = axes(data, 1)
    y = axes(data, 2)
    xlims = (minimum(x), maximum(x))
    ylims = (minimum(y), maximum(y))

    mean_over_x = vec(mean(data, dims=1))   # profile vs y, shown on the left

    ax_right = Axis(
        layout[1, 1];
        ylabel = "z (pixels)",
        xlabel = profile_label,
        yreversed = true,
        xreversed=true,
        limits = (nothing, ylims),
        xautolimitmargin = (0, 0),
        yautolimitmargin = (0, 0),
        yticksmirrored = true,
    )

    ax_heatmap = Axis(
        layout[1, 2];
        xlabel = "x (pixels)",
        aspect = AxisAspect(aspect),
        yreversed = true,
        limits = (xlims, ylims),
        xautolimitmargin = (0, 0),
        yautolimitmargin = (0, 0),
        yticksmirrored = true,
    )

    hm = heatmap!(
        ax_heatmap,
        x,
        y,
        data;
        colormap=colormap,
        colorrange=colorrange,
    )

    lines!(
        ax_right,
        mean_over_x,
        y;
        color=:darkorange,
        linewidth=2,
    )

    linkyaxes!(ax_right, ax_heatmap)

    # heatmap is now in the middle: hide its z-axis label/ticklabels,
    # since ax_right (on the left) already carries the z-axis labels
    hideydecorations!(
        ax_heatmap;
        label = true,
        ticklabels = true,
        ticks = false,
        grid = false,
        minorgrid = false,
        minorticks = false,
    )

    # Tie column 2's width to row 1's height via the aspect ratio, so the
    # cell is exactly as wide as the axis needs — no leftover whitespace.
    colsize!(layout, 1, Relative(0.25))
    colsize!(layout, 2, Aspect(1, aspect))

    Colorbar(
        layout[1, 3],
        hm;
        label = cb_label,
        vertical = true,
        flipaxis = true,
    )

    colgap!(layout, 1, 8)
    colgap!(layout, 2, 2)

    Makie.trim!(layout)

    return fig
end

# ── Generate F1 and F2 independently ──────────────────────────────────────
fig_F1 = plot_heatmap_with_profile(
    F1_mean;
    colorrange = extrema(filter(isfinite, F1_mean)),
)

fig_F2 = plot_heatmap_with_profile(
    F2_mean;
    colorrange = extrema(filter(isfinite, F2_mean)),
)

F1_mean_norm = (x -> max(x, 0)).(F1_mean) ./ maximum(max.(F1_mean, 0))
F2_mean_norm = (x -> max(x, 0)).(F2_mean) ./ maximum(max.(F2_mean, 0))

function plot_heatmap_with_top_profile(
        data;
        colormap      = :viridis,
        size          = (600, 400),
        save_name     = "SG_img_profile",
        profile_label = L"Intensity ($\mathrm{a.u.}$)",
        aspect        = 4.75,
        label_size      = 18,
        ticklabel_size  = 14,
        y_scale         = 4,
        x_tick_step     = 400,
)
    fig = Figure(; size=size, backgroundcolor=:white)

    layout = GridLayout(fig[1, 1])

    x = axes(data, 1)
    y = axes(data, 2) 
    xlims = (minimum(x), maximum(x))
    ylims = (minimum(y), maximum(y))

    data_norm = (x -> max(x, 0)).(data) ./ maximum(max.(data, 0))

    mean_over_y = vec(mean(data, dims=2))   # transverse profile vs x, shown on top

    _latexfmt(vs) = [L"%$(Int(round(Int, v)))" for v in vs]
    _yfmt   = isnothing(y_scale) ? _latexfmt :
                  (vs -> [L"%$(Int(round(Int, y_scale * v)))" for v in vs])
    _xticks = isnothing(x_tick_step) ? Makie.automatic :
                  range(0, xlims[2]; step=x_tick_step)

    # Upper y-limit: ceil to the next multiple of the leading decade
    # e.g. 750 → 800 (decade=100),  32 → 40 (decade=10)
    _y_max  = max(maximum(filter(isfinite, mean_over_y)), 1.0)
    _decade = 10.0^floor(log10(_y_max))
    y_upper = ceil(_y_max / _decade) * _decade

    ax_top = Axis(
        layout[1, 1];
        ylabel         = profile_label,
        ylabelsize     = label_size,
        yticklabelsize = ticklabel_size,
        limits         = (xlims, (0, y_upper)),
        xautolimitmargin = (0, 0),
        yautolimitmargin = (0, 0),
        xticksvisible  = true,
        xtickalign     = 0.5,
        xticks         = _xticks,
        ytickformat    = _latexfmt,
    )

    ax_heatmap = Axis(
        layout[2, 1];
        xlabel         = L"$z$ (pixels)",
        ylabel         = L"$x$ (pixels)",
        xlabelsize     = label_size,
        ylabelsize     = label_size,
        xticklabelsize = ticklabel_size,
        yticklabelsize = ticklabel_size,
        aspect         = AxisAspect(aspect),
        yreversed      = false,
        limits         = (xlims, ylims),
        xautolimitmargin = (0, 0),
        yautolimitmargin = (0, 0),
        xtickalign     = 0.5,
        xticks         = _xticks,
        xtickformat    = _latexfmt,
        ytickformat    = _yfmt,
    )

    colorrange    = extrema(filter(isfinite, data_norm))
    hm = heatmap!(
        ax_heatmap,
        x,
        y,
        data_norm;
        colormap=colormap,
        colorrange=colorrange,
    )

    lines!(
        ax_top,
        x,
        mean_over_y;
        color=:red,
        linewidth=2,
    )

    linkxaxes!(ax_top, ax_heatmap)

    # profile is above the heatmap: hide its x tick labels; ticks
    # themselves are already off via xticksvisible=false above
    hidexdecorations!(
        ax_top;
        label      = true,
        ticklabels = true,
        ticks      = false,    # keep ticks visible with xtickalign=0.5
        grid       = false,
        minorgrid  = false,
        minorticks = false,
    )

    rowsize!(layout, 1, Relative(0.35))
    rowsize!(layout, 2, Aspect(1, 1/aspect))

    rowgap!(layout, 1, 12)

    save(joinpath(OUTDIR, "$(save_name).png"), fig; px_per_unit = 3)
    save(joinpath(OUTDIR, "$(save_name).pdf"), fig; px_per_unit = 3)

    return display(fig)
end


# fig_F1_transverse = plot_heatmap_with_top_profile(F1_mean)

fig_F1_transverse = plot_heatmap_with_top_profile(F1_mean'; save_name="SG_img_profile_f1")
fig_F2_transverse = plot_heatmap_with_top_profile(F2_mean'; save_name="SG_img_profile_f2")


extrema(filter(isfinite, F1_mean))


##################################################################################################
##################################################################################################
## Main plot

BASE_PATH

load(joinpath(BASE_PATH,"EXPDATA_ANALYSIS","smoothing_binning_xkl","data_averaged_2.jld2"),"data")
load(joinpath(BASE_PATH,"EXPDATA_ANALYSIS","smoothing_binning_2025","data_averaged_2.jld2"),"data")