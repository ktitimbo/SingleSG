##################################################################################################
#  PREPARATION OF FIGURES FOR FINAL MANUSCRIPT
#  Kelvin Titimbo - California Institute of Technology - July 2026
##################################################################################################
#
#  ORGANISATION
#  ----------------------------------------------------------------------------------------------
#  §0  Environment, paths, physical constants, apparatus geometry, and ALL shared infrastructure
#      (saving, tick formatters, interpolation, JLD2 inspection, model colours/labels, theme).
#  §1  Stern-Gerlach magnet geometry
#  §2  K-39 effective magnetic moment (Breit-Rabi) vs coil current
#  §3  Experimental patterns: F=1 / F=2 mean images with profiles
#  §4  Coil-current grid and the CQD induction-term run     (inputs for §5 and §6)
#  §5  Collapse time vs time of flight
#  §6  Model-experiment agreement vs current and vs collapse cycles
#  §7  Beam splitting: measured F=1 position vs CQD and QM
#  §8  Run manifest
#
#  Every section defines the functions it needs immediately above the figures that use them, and
#  each figure lives in its own `let ... end` block. After §0 has run, §1, §2, §3, §6 and §7 can
#  be evaluated independently; §5 needs §4, and §6 additionally needs `cqd_times` from §5 for the
#  collapse-cycle abscissa. Nothing else is shared between sections.
#
#  OPEN ITEMS
#  ----------------------------------------------------------------------------------------------
#   * §1: the ±3 mm `hspan!` band is not the slit (z_slit = 0.3 mm). Name what it represents and
#     derive it from a constant (pole-gap half-height?) before writing the caption.
#   * §5: confirm that column 5 of the CQD screen array is v_y in m/s (`V_COL`).
#   * §0: ~15 packages are loaded but unused here; prune once the `include`d module files are
#     confirmed to carry their own `using` statements.
#
#  FIGURE SIZING FOR A ONE-COLUMN LETTER-PAPER MANUSCRIPT
#  ----------------------------------------------------------------------------------------------
#  CairoMakie's `size` is in points for vector output (`pt_per_unit = 1`), and 1 pt = 1/72 in, so
#  a figure is printed at its true size only when `size[1]` equals the LaTeX line width in points
#  and it is included WITHOUT scaling. US Letter with 1 in margins gives a 6.5 in text block:
#
#      \includegraphics[width=\linewidth]   → 6.5 in  = 468 pt
#      \includegraphics[width=0.9\linewidth] → 5.85 in = 421 pt
#      \includegraphics[width=0.8\linewidth] → 5.2 in  = 374 pt
#
#  Set the figure width to the number you will actually use and DO NOT scale in LaTeX; then the
#  theme's `fontsize` is the printed point size, so 9-10 pt keeps labels legible and consistent
#  with the body text. A figure authored at 800 pt and included at \linewidth is scaled by 0.59,
#  which turns 16 pt labels into 9.4 pt; authored at 1120 pt it is scaled by 0.42, turning the
#  same labels into 6.7 pt. That is why panels of different widths look typographically different
#  in the same paper. The sizes below are left as they are - see FIG_WIDTH_PT when standardising.
##################################################################################################
 
##################################################################################################
## §0  ENVIRONMENT, PATHS, CONSTANTS, APPARATUS GEOMETRY, SHARED INFRASTRUCTURE
##################################################################################################
 
# ── Plotting ────────────────────────────────────────────────────────────────────────────────────
using CairoMakie
using Colors, ColorSchemes
using LaTeXStrings, Printf, PrettyTables
using MathTeXEngine

SAVE_FIG        = true
FIG_EXTS        = ("png", "pdf", "svg")   # every figure is written in all of these
FIG_PX_PER_UNIT = 3                # raster resolution; ignored for pdf/svg

# ── Time-stamping ───────────────────────────────────────────────────────────────────────────────
using Dates
const T_START = Dates.now()

# ── Numerics ────────────────────────────────────────────────────────────────────────────────────
using LinearAlgebra, Roots, StatsBase
using Random, Statistics, NaNStatistics
using Interpolations, Loess, Optim, BSplineKit, Polynomials, DSP, LambertW, PolyLog, Alert
using DataStructures, Distributions, StaticArrays

# ── Data I/O ────────────────────────────────────────────────────────────────────────────────────
using OrderedCollections, JLD2
using Pkg
using DelimitedFiles, CSV, DataFrames   # REVIEW: unused in this file

# ── Threads ─────────────────────────────────────────────────────────────────────────────────────
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads"  count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()

# ── Paths ───────────────────────────────────────────────────────────────────────────────────────
cd(@__DIR__)
const BASE_PATH = raw"F:\SternGerlachExperiments"
const STUDIES_DIR = joinpath(@__DIR__, "data_studies")

const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss")   # REVIEW: const + timestamp → not re-includable
const OUTDIR    = joinpath(STUDIES_DIR, "FINAL_IMAGES_" * RUN_STAMP)
isdir(OUTDIR) || mkpath(OUTDIR)
@info "Created output directory" OUTDIR

# Input archives. FIT_DIR is pinned to one fit run; to always take the most recent instead:
#   FIT_DIR = joinpath(STUDIES_DIR, last(sort(filter(startswith("FIT2025_ki_scale_"), readdir(STUDIES_DIR)))))
const FIT_DIR         = joinpath(STUDIES_DIR, "FIT2025_ki_scale_20260911T164352057")
const FIT_ARCHIVE     = joinpath(FIT_DIR, "SG_ki_scale_results.jld2")
const EXPERIMENT_FILE = joinpath(BASE_PATH, "EXPERIMENTS", "20260220", "data_processed.jld2")

const TEMP_DIR = joinpath(@__DIR__,"artifacts", "JuliaTemp")
isdir(TEMP_DIR) || mkpath(TEMP_DIR);
ENV["TMPDIR"] = ENV["TEMP"] = ENV["TMP"] = TEMP_DIR
@info "Temporary directory configured" TEMP_DIR

HOSTNAME = gethostname()
@info "Running on host" HOSTNAME

# ── RNG ─────────────────────────────────────────────────────────────────────────────────────────
base_seed_set = 145
rng_set = MersenneTwister(base_seed_set)   # REVIEW: unused in this file

# ── Custom modules ──────────────────────────────────────────────────────────────────────────────
include("./Modules/atoms.jl")
include("./Modules/samplings.jl")
include("./Modules/JLD2_MyTools.jl")
include("./Modules/MyExperimentalAnalysis.jl")
include("./Modules/TheoreticalSimulation.jl")
using .TheoreticalSimulation
TheoreticalSimulation.SAVE_FIG = SAVE_FIG
TheoreticalSimulation.FIG_EXT  = first(FIG_EXTS)
TheoreticalSimulation.OUTDIR   = OUTDIR

println("\n\t\tRunning process on:\t $(RUN_STAMP) \n")

# ── Math constants ──────────────────────────────────────────────────────────────────────────────
const TWOπ  = 2π 
const INV_E = exp(-1)

# ── Physical constants (CODATA 2022 via NIST) ───────────────────────────────────────────────────
const kb    = 1.380649e-23         # Boltzmann constant (J/K)
const ħ     = 6.62607015e-34 / 2π  # Reduced Planck constant (J s)
const μ₀    = 1.25663706127e-6     # Vacuum permeability (T m/A)
const μB    = 9.2740100657e-24     # Bohr magneton (J/T)
const γₑ    = -1.76085962784e11    # Electron gyromagnetic ratio (rad/(s T)), RSU 3.0e-10
const μₑ    = 9.2847646917e-24     # Electron magnetic moment (J/T), RSU 3.0e-10
const Sspin = 1/2                  # Electron spin
const gₑ    = -2.00231930436092    # Electron g-factor
# RSU : Relative Standard Uncertainty

# ── Atom ────────────────────────────────────────────────────────────────────────────────────────
atom            = "39K"
K39_params      = AtomParams(atom)          # [R μn γn Ispin Ahfs M]
quantum_numbers = fmf_levels(K39_params)

# ── Camera and pixel geometry ───────────────────────────────────────────────────────────────────
# Intrinsic camera properties
cam_pixelsize        = 6.5e-6 ;          # Physical pixel size of camera [m]
nx_pixels, nz_pixels = (2160, 2560) ;    # (Nx, Nz) pixels
# Simulation resolution
sim_bin_x, sim_bin_z = (1, 1) ;          # Camera binning
sim_pixelsize_x, sim_pixelsize_z = (sim_bin_x, sim_bin_z) .* cam_pixelsize ;  # Effective pixel size after binning [m]
# Image dimensions (adjusted for binning)
x_pixels = Int(nx_pixels / sim_bin_x) ;  # Number of x-pixels after binning
z_pixels = Int(nz_pixels / sim_bin_z) ;  # Number of z-pixels after binning
# Spatial axes shifted to center the pixels
x_position = pixel_coordinates(x_pixels, sim_bin_x, sim_pixelsize_x) ;
z_position = pixel_coordinates(z_pixels, sim_bin_z, sim_pixelsize_z) ;
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

# ── Apparatus ───────────────────────────────────────────────────────────────────────────────────
T_K = 273.15 + 205                 # furnace temperature (K)
const x_furnace = 2.0e-3           # furnace aperture (m)
const z_furnace = 100e-6
const x_slit    = 4.0e-3           # pre-SG slit (m)
const z_slit    = 300e-6
const R_aper         = 5.8e-3/2    # post-SG circular aperture radius (m)
const y_SGToAperture = 42.0e-3
const y_FurnaceToSlit = 224.0e-3   # propagation distances (m)
const y_SlitToSG      = 44.0e-3
const y_SG            = 7.0e-2
const y_SGToScreen    = 32.0e-2
const R_tube = 35e-3/2             # connecting pipe radius (m)
effusion_params = BeamEffusionParams(x_furnace, z_furnace, x_slit, z_slit, y_FurnaceToSlit, T_K, K39_params)
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

TheoreticalSimulation.DEFAULT_camera_pixel_size = cam_pixelsize
TheoreticalSimulation.DEFAULT_x_pixels          = nx_pixels
TheoreticalSimulation.DEFAULT_z_pixels          = nz_pixels
TheoreticalSimulation.DEFAULT_x_furnace         = x_furnace
TheoreticalSimulation.DEFAULT_z_furnace         = z_furnace
TheoreticalSimulation.DEFAULT_x_slit            = x_slit
TheoreticalSimulation.DEFAULT_z_slit            = z_slit
TheoreticalSimulation.DEFAULT_y_FurnaceToSlit   = y_FurnaceToSlit
TheoreticalSimulation.DEFAULT_y_SlitToSG        = y_SlitToSG
TheoreticalSimulation.DEFAULT_y_SG              = y_SG
TheoreticalSimulation.DEFAULT_y_SGToScreen      = y_SGToScreen
TheoreticalSimulation.DEFAULT_R_tube            = R_tube
TheoreticalSimulation.DEFAULT_c_aperture        = R_aper
TheoreticalSimulation.DEFAULT_y_SGToAperture    = y_SGToAperture


##################################################################################################
## §0.1  SHARED FIGURE INFRASTRUCTURE
##       Everything below is used by more than one section, so it lives here rather than in
##       whichever section happened to need it first. Section-specific helpers stay in their
##       own section, immediately above the figures that use them.
##################################################################################################
 
"""
    savefig(fig, name; px_per_unit = FIG_PX_PER_UNIT, exts = FIG_EXTS)
 
Write `fig` to `OUTDIR/name.<ext>` for every extension in `exts`. No-op when `SAVE_FIG == false`.
 
`px_per_unit` only affects raster formats; vector formats use `pt_per_unit` (CairoMakie default
1), so for PDF/SVG the figure's `size` is its size in points. Pass `exts` explicitly to skip a
format for one figure — e.g. `exts = ("png", "pdf")` for the §3 heatmaps, whose SVG would embed a
multi-megabyte raster for no gain.
"""
function savefig(fig, name::AbstractString; px_per_unit = FIG_PX_PER_UNIT, exts = FIG_EXTS)
    SAVE_FIG || return nothing
    for ext in exts
        save(joinpath(OUTDIR, "$name.$ext"), fig; px_per_unit)
    end
    return nothing
end

# ── Tick formatting ─────────────────────────────────────────────────────────────────────────────
"Integer tick labels typeset by MathTeXEngine, e.g. `-2` → `L\"-2\"`."
latex_int(vs) = [L"%$(round(Int, v))" for v in vs]
 
"Decade ticks `(positions, labels)` for a log10 axis, e.g. `latex_log_ticks(-3:0)`."
latex_log_ticks(pows) = (exp10.(pows), [L"10^{%$p}" for p in pows])
 
"""
    decade_ticks(vs...) -> (positions, labels)
 
Decade tick positions and LaTeX labels spanning every finite positive value in `vs`, for a
`log10` axis. Built from the data, so a panel gets correct ticks without a hard-coded range.
"""
function decade_ticks(vs...)
    v = filter(x -> isfinite(x) && x > 0, vcat(vs...))
    return latex_log_ticks(floor(Int, log10(minimum(v))):ceil(Int, log10(maximum(v))))
end

"""
    snapped_ticks(vals...; step) -> Vector
 
Tick positions covering every value in `vals`, snapped outward to multiples of `step`, so a
linear axis ends on round numbers and the zero line always carries a tick.
"""
function snapped_ticks(vals...; step)
    all_v = filter(isfinite, vcat(vals...))
    lo = floor(minimum(all_v) / step) * step
    hi = ceil(maximum(all_v) / step) * step
    return round.(lo:step:hi, digits = 6)
end

# ── Interpolation ───────────────────────────────────────────────────────────────────────────────
"""
    interp_loglog(xq, x, y; logy = true) -> Vector
 
Piecewise-linear interpolation of `y(x)` evaluated at `xq`, performed in log10(x) and (when
`logy`) log10(y). Appropriate when both variables span decades — a power law becomes a straight
line in those coordinates, so the interpolation is exact for one and nearly so in general.
 
Returns `NaN` outside the range of `x`: a query grid should lie inside the simulated grid, and a
NaN here is a signal that it does not, rather than a silent extrapolation.
"""
function interp_loglog(xq, x, y; logy::Bool = true)
    p  = sortperm(x)
    lx = log10.(x[p])
    ys = logy ? log10.(y[p]) : float.(y[p])
 
    out = similar(float.(xq))
    for (k, q) in enumerate(log10.(xq))
        if q < lx[1] || q > lx[end]
            out[k] = NaN
        else
            j = min(searchsortedlast(lx, q), length(lx) - 1)
            t = (q - lx[j]) / (lx[j+1] - lx[j])
            v = ys[j] + t * (ys[j+1] - ys[j])
            out[k] = logy ? exp10(v) : v
        end
    end
    return out
end

# ── Terminal colouring and JLD2 inspection ──────────────────────────────────────────────────────
"""
    _ANSI
 
ANSI escape sequences keyed by style name. Colours: `:red`, `:green`, `:yellow`, `:blue`,
`:magenta`, `:cyan`, `:white`; attributes: `:bold`, `:underline`. `"\\e[0m"` (used by `cstr`)
resets every style.
"""
const _ANSI = Dict(
    :red => "\e[31m", :green => "\e[32m", :yellow => "\e[33m", :blue => "\e[34m",
    :magenta => "\e[35m", :cyan => "\e[36m", :white => "\e[37m",
    :bold => "\e[1m", :underline => "\e[4m",
)
 
"""
    cstr(s, styles...)
 
Return `string(s)` wrapped in the ANSI codes of the given `styles` (symbols from `_ANSI`, applied
together) and followed by a reset, so it prints in colour/bold when interpolated into `@info`,
`@warn`, `println`, …
 
# Examples
```julia
@info "kᵢ range: " * cstr("(1.0 – 4.1)×10⁻⁶", :yellow, :bold)
@info cstr("Selected configuration: nz = ", :cyan) * cstr(NZ_FIXED, :yellow, :bold)
```
"""
cstr(s, styles::Symbol...) = join(_ANSI[k] for k in styles) * string(s) * "\e[0m"
 
"""
    list_jld2_entries(path; io = stdout, show_type = true)
 
Print the full tree of groups and datasets stored in the JLD2 file `path` (names only, no values)
as a quick reference of *what* is in the archive and *where* to find it. Groups are printed in
bold blue, datasets in green; with `show_type = true` each dataset is annotated with its element
type and, for arrays, its size (e.g. `Vector{Float64} (47)`), so the layout can be checked
without loading anything into the workspace.
 
Returns the vector of full dataset paths (e.g. `"fit/cqd/ki"`), usable directly as keys:
`jldopen(path) do f; f["fit/cqd/ki"]; end`. Assign that result to its own name — assigning it
back onto the path variable replaces the path with a `Vector{String}`.
 
Note: `show_type` reads each dataset to query its type, so it materialises the archive. Pass
`show_type = false` for a fast listing of a large file.
"""
function list_jld2_entries(path::AbstractString; io::IO = stdout, show_type::Bool = true)
    isfile(path) || throw(ArgumentError("JLD2 file not found: $path"))
    paths = String[]
 
    # recursive walk: `g` is the file or a group, `prefix` its full path
    function _walk(g, prefix, depth)
        for k in keys(g)            # insertion (write) order, as stored by JLD2
            full = isempty(prefix) ? k : prefix * "/" * k
            v    = g[k]
            pad  = "  "^depth
            if v isa JLD2.Group
                println(io, pad, cstr(k * "/", :blue, :bold))
                _walk(v, full, depth + 1)
            else
                push!(paths, full)
                info = ""
                if show_type
                    info = v isa AbstractArray ?
                           "  " * cstr("$(typeof(v)) ($(join(size(v), "×")))", :white) :
                           "  " * cstr(string(typeof(v)), :white)
                end
                println(io, pad, cstr(k, :green), info)
            end
        end
    end
 
    println(io, cstr("Contents of ", :bold), cstr(path, :magenta))
    jldopen(path, "r") do f
        _walk(f, "", 1)
    end
    println(io, cstr("  $(length(paths)) datasets", :white))
    return paths
end


# ── Model identity, used by every comparison panel ──────────────────────────────────────────────
# One colour and one spelling per model across the whole manuscript: a reader tracking CQD from
# figure to figure must never have to check whether the colours were swapped.
const COLOR_CQD = :crimson
const COLOR_QM  = :royalblue
const LABEL_EXP = "Experiment"
const LABEL_QM  = "Existing models"
const LABEL_CQD = "Co-quantum dynamics"


# ── Typography ──────────────────────────────────────────────────────────────────────────────────
# Plain strings and LaTeXStrings are rendered by different engines, so without a shared font
# family an axis labelled `L"..."` and one labelled `"..."` disagree typographically. Loading the
# MathTeXEngine faces as the regular fonts makes every label Computer Modern, matching the
# manuscript body text.
const TEX_FONTS = (
    regular     = texfont(:regular),
    bold        = texfont(:bold),
    italic      = texfont(:italic),
    bold_italic = texfont(:bolditalic),
)

# When standardising figure widths (see the header), set FIG_WIDTH_PT to the LaTeX line width in
# points and raise `fontsize` to the printed size you want — 9–10 pt for a one-column letter page.
const FIG_WIDTH_PT = 468        # 6.5 in text block; unused until the sizes are standardised
 
const PUB_THEME = Theme(
    fonts    = TEX_FONTS,
    fontsize = 16,
    Axis = (
        xlabelsize = 20, ylabelsize = 20,
        xticklabelsize = 18, yticklabelsize = 18,

        # minor ticks on by default, both axes
        # xminorticksvisible = true, yminorticksvisible = true,
 
        # tick geometry: majors and minors point the same way, minors shorter
        xtickalign = 0.5, ytickalign = 0.5,
        xminortickalign = 0.5, yminortickalign = 0.5,
        xticksize = 8, yticksize = 8,
        xminorticksize = 4, yminorticksize = 4,
        xtickwidth = 1.2, ytickwidth = 1.2,
        xminortickwidth = 1.0, yminortickwidth = 1.0,
    ),
    Legend = (labelsize = 16, framevisible = false),
)
set_theme!(PUB_THEME)


# ── Run record ──────────────────────────────────────────────────────────────────────────────────
"""
    write_manifest(; extras...)
 
Write `OUTDIR/MANIFEST.txt` recording what produced this set of figures: run stamp, host, Julia
version, the input archives, the environment status, and any `extras` passed as keywords
(selected indices, fitted parameters, …).
 
Without this the provenance of a figure set lives only in the current state of this script; with
it, "which simulation produced Figure 5?" is answerable months later from the output directory
alone.
"""
function write_manifest(; extras...)
    open(joinpath(OUTDIR, "MANIFEST.txt"), "w") do io
        println(io, "Figure set : ", basename(OUTDIR))
        println(io, "Run stamp  : ", RUN_STAMP)
        println(io, "Finished   : ", Dates.now())
        println(io, "Elapsed    : ", Dates.canonicalize(Dates.now() - T_START))
        println(io, "Host       : ", HOSTNAME)
        println(io, "Julia      : ", VERSION, "  (", Threads.nthreads(), " threads)")
        println(io, "Script     : ", @__FILE__)
        println(io)
        println(io, "INPUTS")
        println(io, "  experiment : ", EXPERIMENT_FILE)
        println(io, "  fit archive: ", FIT_ARCHIVE)
        if @isdefined(cqd_ki_path)
            println(io, "  CQD screen : ", cqd_ki_path)
        end
        println(io)
        if !isempty(extras)
            println(io, "PARAMETERS")
            for (k, v) in pairs(extras)
                println(io, "  ", rpad(string(k), 11), ": ", v)
            end
            println(io)
        end
        println(io, "FIGURES")
        for f in sort(filter(!=("MANIFEST.txt"), readdir(OUTDIR)))
            println(io, "  ", f)
        end
        println(io)
        println(io, "ENVIRONMENT")
        try
            Pkg.status(; io = io)
        catch err
            println(io, "  (Pkg.status unavailable: ", err, ")")
        end
    end
    @info "Manifest written" file = joinpath(OUTDIR, "MANIFEST.txt")
    return nothing
end


##################################################################################################
## §1  STERN–GERLACH MAGNET GEOMETRY  →  SG_geometry
##################################################################################################
let
    x_line = range(-10e-3, 10e-3; length = 10_001)
    x_mm   = 1e3 .* x_line
    CLIP_MM = 10.0                       # polygons are closed at ±CLIP_MM, outside the axis limits
    blue, red = RGBf(10/255, 10/255, 200/255), RGBf(200/255, 10/255, 10/255)
 
    fig = Figure(figure_padding = 10)
    ax  = Axis(fig[1, 1];
        xlabel = L"x \ (\mathrm{mm})", 
        ylabel = L"z \ (\mathrm{mm})",
        limits = (-8, 8, -4, 7),
        xticks = -8:2:8, 
        yticks = -4:1:7,
        xtickformat = latex_int, ytickformat = latex_int,
        aspect = DataAspect(),
    )
 
    # Top magnet (rounded edge): fill from the edge profile up to +∞
    z_edge = 1e3 .* TheoreticalSimulation.z_magnet_edge.(x_line)
    poly!(ax, Point2f.(vcat(x_mm, reverse(x_mm)), vcat(z_edge, fill(CLIP_MM, length(x_mm))));
        color = (blue, 0.85), strokecolor = blue, strokewidth = 2)
 
    # Bottom magnet (trench): fill from −∞ up to the trench profile
    z_trench = 1e3 .* TheoreticalSimulation.z_magnet_trench.(x_line)
    poly!(ax, Point2f.(vcat(x_mm, reverse(x_mm)), vcat(fill(-CLIP_MM, length(x_mm)), reverse(z_trench)));
        color = (red, 0.85), strokecolor = red, strokewidth = 2)
 
    # REVIEW: ±3 mm is NOT the slit extent, it refers to the piece of metal where the slit is.
    hspan!(ax, -3.0, 3.0; color = (:gray36, 0.55))
 
    # Pre-SG slit, drawn on top of the band
    hw_x, hw_z = 1e3 * x_slit / 2, 1e3 * z_slit / 2
    poly!(ax, Point2f.([-hw_x, -hw_x, hw_x, hw_x], [-hw_z, hw_z, hw_z, -hw_z]);
        color = :white, strokecolor = :black, strokewidth = 1.5)
 
    resize_to_layout!(fig)
    display(fig)
    savefig(fig, "SG_geometry"; px_per_unit = FIG_PX_PER_UNIT)
end
 
##################################################################################################
## §2  ³⁹K EFFECTIVE MAGNETIC MOMENT (BREIT–RABI) vs COIL CURRENT  →  SG_mm_effective
##################################################################################################
"Sample a `cgrad` through the given anchor colours at `n` evenly spaced positions."
anchored_cmap(anchors, n) = (g = cgrad(collect(anchors)); [g[t] for t in range(0, 1; length = n)])
 
let
    F_up, F_down = K39_params.Ispin + 0.5, K39_params.Ispin - 0.5
    mf_up   = F_up:-1.0:-F_up
    mf_down = -F_down:1.0:F_down
    colorsF = vcat(
        anchored_cmap((colorant"darkred", colorant"orangered", colorant"chocolate4"), length(mf_up)),
        anchored_cmap((colorant"navy", colorant"deepskyblue3"), length(mf_down)),
    )
    current_range = exp10.(range(log10(0.0009), log10(1.01); length = 600))
    LEGEND_LABEL_SIZE = 16
 
    # Common tick styling for both x axes
    ticks_kw = (xtickalign = 0.5, xminortickalign = 0.5, xticksize = 10, xminorticksize = 5,
                xscale = log10, xminorticksvisible = true, xminorticks = IntervalsBetween(9),
                xgridvisible = false, xminorgridvisible = false, backgroundcolor = :transparent)
 
    fig = Figure(size = (1120, 600))
 
    # Top axis (B_z) created first so it sits behind the data axis
    ax_top = Axis(fig[1, 1]; xlabel = L"Magnetic field $B_z\ (\mathrm{T})$",
        xaxisposition = :top, ticks_kw...)
    hideydecorations!(ax_top)
    hidespines!(ax_top, :l, :r, :b)
 
    # Bottom axis (current): data lives here
    ax = Axis(fig[1, 1];
        xlabel = L"SG current $(\mathrm{A})$", ylabel = L"$\mu_{F}/\mu_{B}$",
        xticks = latex_log_ticks(-3:0),               # explicit decades: no surprise minor labels
        ytickformat = ys -> [iszero(y) ? L"0" : L"%$(round(y, digits = 1))" for y in ys],
        ytickalign = 0.5, yminortickalign = 0.5, yticksize = 10, yminorticksize = 5,
        ticks_kw...)
 
    # Dash pattern in units of linewidth: [start, on, off] — 8 on, 2 off, period 10
    LONG_DASH = Linestyle([0.0, 8.0, 10.0])
 
    # Solid: F = I+½ manifold except its lowest m_F; dashed: the stretched m_F = −F and all F = I−½
    lines_to_plot = vcat(
        [(F_up, mf, :solid) for mf in mf_up[1:end-1]],
        [(F_up, mf_up[end], LONG_DASH)],
        [(F_down, mf, LONG_DASH) for mf in mf_down],
    )
    for ((f, mf, lstyle), color) in zip(lines_to_plot, colorsF)
        μ_vals = TheoreticalSimulation.μF_effective.(current_range, f, mf, Ref(K39_params)) ./ μB
        mf_str = mf > 0 ? "+$(Int(mf))" : "$(Int(mf))"
        lines!(ax, current_range, μ_vals; color, linestyle = lstyle, linewidth = 3,
            label = L"$F=%$(Int(f))$, $m_{F}=%$(mf_str)$")
    end
 
    # Field at which the F = I+½, m_F = −F moment crosses zero: μ_B B = A(I+½)·2πħ/(γn−γe)·...
    B_cross = 2π * K39_params.Ahfs * (K39_params.Ispin + 0.5) / (K39_params.γn - γₑ)
    I₀      = find_zero(I -> TheoreticalSimulation.BvsI(I) - B_cross, (0.001, 0.050))
    @info "Magnetic crossing point" I₀_mA = round(1e3 * I₀, digits = 3) ∂zBz_Tperm = round(TheoreticalSimulation.GvsI(I₀), digits = 2) Bz_mT = round(1e3 * TheoreticalSimulation.BvsI(I₀), digits = 3)
    vlines!(ax, [I₀]; color = :black, linestyle = :dot, linewidth = 2.5)
 
    axislegend(ax; position = :rc, backgroundcolor = :white, patchsize = (60, 2), labelsize = LEGEND_LABEL_SIZE)
    xlims!(ax, current_range[1], current_range[end])
    linkxaxes!(ax, ax_top)
 
    # Top-axis ticks: invert B(I) at pure decades of B. BvsI is monotone on this range, so the
    # bracket cannot fail; the try/catch only guards decades outside the current range.
    B_lo, B_hi = extrema(TheoreticalSimulation.BvsI.((current_range[1], current_range[end])))
    tick_I, tick_labels = Float64[], LaTeXString[]
    for d in floor(Int, log10(B_lo)):ceil(Int, log10(B_hi))
        b = 10.0^d
        try
            push!(tick_I, find_zero(I -> TheoreticalSimulation.BvsI(I) - b, (current_range[1], current_range[end])))
            push!(tick_labels, L"10^{%$d}")
        catch
            # decade outside the plotted current range
        end
    end
    ax_top.xticks = (tick_I, tick_labels)
 
    display(fig)
    savefig(fig, "SG_mm_effective")
end
 

##################################################################################################
## §3  EXPERIMENTAL PATTERN: F=1 / F=2 IMAGES WITH PROFILES
##      →  SG_img_f1, SG_img_f2                  (image vertical, profile on the left + colorbar)
##      →  SG_img_profile_f1, SG_img_profile_f2  (image horizontal, profile on top)
##################################################################################################
# The processed images are stored binned 4× along x (the non-dispersive direction); z (the
# Stern–Gerlach dispersive direction) is unbinned. Rather than rescale the data, the tick labels
# of the binned axis are multiplied by the binning factor (`y_scale`), so both panel styles are
# read in unbinned camera pixels while the arrays stay exactly as they come out of the pipeline.
#
# Array orientation differs between the two panel functions and is the caller's responsibility:
#   plot_heatmap_with_profile(img)    expects (x × z)  — image drawn with z vertical
#   plot_image_with_z_profile(img')   expects (z × x)  — image drawn with z horizontal
# In both cases the profile shown is the mean over x, i.e. the intensity distribution along z,
# which is the quantity the splitting is measured from.

"""
    load_experiment(filepath) -> data
 
Load the processed Stern–Gerlach dataset from a JLD2 file and print the coil-current / field
table for the run.
 
The returned dictionary carries at least `:Currents`, `:CurrentsError`, `:BzTesla` (one entry per
coil-current setting) and the image stacks `:F1ProcessedImages`, `:F2ProcessedImages`, each of
shape (x × z × repetition × current). The three current/field columns are checked for equal
length so a truncated or mis-saved file is caught here rather than at plotting time.
"""
function load_experiment(filepath::AbstractString)
    data = load(filepath, "data")
    currents     = vec(data[:Currents])
    currents_err = vec(data[:CurrentsError])
    bz_mT        = 1e3 .* vec(data[:BzTesla])

    lengths = length.((currents, currents_err, bz_mT))
    all(==(lengths[1]), lengths) || throw(DimensionMismatch("Columns have different lengths: $lengths"))
 
    pretty_table(hcat(currents, currents_err, bz_mT);
        title         = joinpath(splitpath(filepath)[end-1:end]...),
        formatters    = [fmt__printf("%8.4f", [1]), fmt__printf("%8.4f", [2]), fmt__printf("%8.4f", [3])],
        alignment     = :c,
        column_labels = [["I0 Current", "I0 CurrentError", "Bz field"], ["[A]", "[A]", "[mT]"]],
        table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style = TextTableStyle(
            first_line_column_label = crayon"yellow bold",
            column_label = crayon"yellow",
            table_border = crayon"blue bold",
            title        = crayon"bold red"),
        equal_data_column_widths = true,
        show_row_number_column = true, 
        row_number_column_label = "No.", 
        row_number_column_alignment = :c,
    )
    return data
end

"""
    mean_image(stack, idx) -> Matrix
 
Mean over the repetition axis (dim 3) of the (x × z × rep × current) image `stack` at
current index `idx`, returning the (x × z) mean frame.
 
NaN-safe: pixels masked out in some repetitions are averaged over the remaining ones, so a
single bad frame does not blank a pixel in the result. The `@view` avoids copying the slice.
"""
mean_image(stack, idx) = dropdims(nanmean(@view(stack[:, :, :, idx]); dims = 3); dims = 3)

"""
    plot_heatmap_with_profile(data; kwargs...) -> Figure
 
Vertical rendering of an (x × z) mean image: heatmap in the centre with z increasing downwards
(camera orientation), the x-averaged intensity profile along z on the left, and a colorbar on
the right. The three panels share the z axis, so a feature in the profile lines up with the
corresponding band in the image.
 
The profile's intensity axis is reversed (`xreversed = true`) so the curve grows toward the
image, and the heatmap's own z decorations are hidden because the profile axis already carries
that scale.
 
# Keyword arguments
- `colormap`, `colorrange`: colour mapping; the default range spans the finite data, i.e. the
  raw intensity scale (not normalised), which is what the colorbar then reports.
- `figsize`: figure size in points.
- `cb_label`, `profile_label`: colorbar and profile-axis labels.
- `aspect`: width:height of the heatmap panel; 0.5 makes the image twice as tall as wide.
  Also used as the column-width constraint, so the cell is exactly as wide as the axis needs.
"""
function plot_heatmap_with_profile(
        data;
        colormap    = :viridis,
        colorrange  = extrema(filter(isfinite, data)),
        figsize     = (600, 550),
        cb_label    = "Mean intensity",
        profile_label = "Intensity (arb. units)",
        aspect      = 0.5,   # width:height ratio, e.g. 0.5 → twice as tall as wide
)
    fig = Figure(; size=figsize, backgroundcolor=:white)
 
    layout = GridLayout(fig[1, 1])
 
    # data is (x × z): dim 1 → x (transverse), dim 2 → z (dispersive)
    x = axes(data, 1)
    y = axes(data, 2)
    xlims = (minimum(x), maximum(x))
    ylims = (minimum(y), maximum(y))
 
    mean_over_x = vec(nanmean(data, dims=1))   # profile vs y, shown on the left
 
    # ── Left panel: intensity profile along z ────────────────────────────────
    ax_right = Axis(
        layout[1, 1];
        ylabel = "z (pixels)",
        xlabel = profile_label,
        yreversed = true,      # z = 0 at the top, matching the camera frame
        xreversed=true,        # intensity grows toward the heatmap
        limits = (nothing, ylims),   # z limits fixed; intensity range left automatic
        xautolimitmargin = (0, 0),   # no padding, so the profile touches the panel edges
        yautolimitmargin = (0, 0),
        yticksmirrored = true,
    )
 
    # ── Centre panel: the mean image ─────────────────────────────────────────
    ax_heatmap = Axis(
        layout[1, 2];
        xlabel = "x (pixels)",
        aspect = AxisAspect(aspect),
        yreversed = true,      # same z direction as the profile axis
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
 
    # Shared z axis: panning/limits stay consistent between profile and image
    linkyaxes!(ax_right, ax_heatmap)
 
    # heatmap is now in the middle: hide its z-axis label/ticklabels,
    # since ax_right (on the left) already carries the z-axis labels
    hideydecorations!(
        ax_heatmap;
        label = true,
        ticklabels = true,
        ticks = false,       # keep the tick marks as a visual scale
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
        flipaxis = true,     # ticks and label on the right-hand side of the bar
    )
 
    colgap!(layout, 1, 8)    # profile ↔ heatmap
    colgap!(layout, 2, 2)    # heatmap ↔ colorbar
 
    # Drop the unused layout space left by trimmed decorations
    Makie.trim!(layout)
 
    return fig
end

"""
    plot_image_with_z_profile(data; kwargs...) -> Figure
 
Horizontal rendering of a (z × x) mean image — pass the transpose of the (x × z) frame — with
the x-averaged intensity profile along z drawn directly above it, sharing the z axis.
 
Intended as the compact, full-width panel for the manuscript: z runs left to right along the
long side, so the split components are separated horizontally and the profile above reads as
their line shape. The image is clipped at zero and normalised to its maximum, so the colour
scale is 0–1 in arbitrary units (no colorbar is drawn); the profile is computed from the raw
data and keeps its own intensity scale.
 
# Keyword arguments
- `colormap`: colour map applied to the normalised image.
- `figsize`: figure size in points.
- `save_name`: base filename, used by the caller when saving.
- `profile_label`: y-axis label of the profile panel.
- `aspect`: width:height of the heatmap panel (4.75 ≈ the frame's z:x ratio once the 4× x
  binning is accounted for). Also sets the image row height via `Aspect(1, 1/aspect)`.
- `label_size`, `ticklabel_size`: font sizes for this panel.
- `y_scale`: x-axis binning factor of the stored images; tick labels on the binned axis are
  multiplied by it so they read as unbinned camera pixels. Pass `nothing` to label raw indices.
- `x_tick_step`: spacing of the shared z ticks, in stored pixels. Pass `nothing` for automatic.
"""
function plot_image_with_z_profile(data;
        colormap      = :viridis,
        figsize       = (800, 450),
        profile_label = L"Intensity ($\mathrm{a.u.}$)",
        aspect        = 4.75,
        label_size      = 18,
        ticklabel_size  = 14,
        y_scale         = 4, # binning
        x_tick_step     = 400,
)
    fig = Figure(; size=figsize, backgroundcolor=:white)
 
    layout = GridLayout(fig[1, 1])
 
    # data is (z × x) here — the caller passes the transposed frame — so dim 1 (named `x`
    # below, the plot's horizontal axis) is z, and dim 2 (`y`) is the binned x direction.
    x = axes(data, 1)
    y = axes(data, 2)
    xlims = (minimum(x), maximum(x))
    ylims = (minimum(y), maximum(y))
 
    # Clip negatives (background subtraction can undershoot) and normalise to peak → colour scale 0–1
    data_norm = (x -> max(x, 0)).(data) ./ maximum(filter(isfinite, max.(data, 0)))
 
    mean_over_y = vec(nanmean(data, dims=2))   # transverse profile vs x, shown on top
 
    # Tick label helpers: integers typeset by MathTeXEngine so they match the LaTeX axis labels
    _latexfmt(vs) = [L"%$(Int(round(Int, v)))" for v in vs]
    # …and the same, rescaled by the binning factor, for the binned axis
    _yfmt   = isnothing(y_scale) ? _latexfmt :
                  (vs -> [L"%$(Int(round(Int, y_scale * v)))" for v in vs])
    _xticks = isnothing(x_tick_step) ? Makie.automatic :
                  range(0, xlims[2]; step=x_tick_step)
 
    # Upper y-limit: ceil to the next multiple of the leading decade
    # e.g. 750 → 800 (decade=100),  32 → 40 (decade=10)
    # Keeps the top tick round and the profile from touching the panel edge.
    _y_max  = max(maximum(filter(isfinite, mean_over_y)), 1.0)
    _decade = 10.0^floor(log10(_y_max))
    y_upper = ceil(_y_max / _decade) * _decade
 
    # ── Top panel: intensity profile along z ─────────────────────────────────
    ax_top = Axis(
        layout[1, 1];
        ylabel         = profile_label,
        ylabelsize     = label_size,
        yticklabelsize = ticklabel_size,
        limits         = (xlims, (0, y_upper)),   # baseline pinned at zero
        xautolimitmargin = (0, 0),
        yautolimitmargin = (0, 0),
        xticksvisible  = true,
        xtickalign     = 0.5,      # ticks centred on the spine, pointing both ways
        xticks         = _xticks,  # same z ticks as the image below
        ytickformat    = _latexfmt,
    )
 
    # ── Bottom panel: the mean image ─────────────────────────────────────────
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
        ytickformat    = _yfmt,    # ×y_scale → unbinned camera pixels
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
 
    # Shared z axis: profile and image columns stay registered
    linkxaxes!(ax_top, ax_heatmap)
 
    # profile sits above the heatmap: hide its x label and tick labels, which the
    # heatmap axis below already carries; the tick marks themselves stay visible
    hidexdecorations!(
        ax_top;
        label      = true,
        ticklabels = true,
        ticks      = false,    # keep ticks visible with xtickalign=0.5
        grid       = false,
        minorgrid  = false,
        minorticks = false,
    )
 
    rowsize!(layout, 1, Relative(0.35))        # profile height, fraction of the figure
    rowsize!(layout, 2, Aspect(1, 1/aspect))   # image row keeps the panel's z:x shape
 
    rowgap!(layout, 1, 12)
 
    return fig
end


exp_data = load_experiment(EXPERIMENT_FILE)
 # Coil-current index shown in the manuscript figure; top-level so §8 can record it.
NI_IDX = 16

let
    I_sel  = exp_data[:Currents][NI_IDX]
    @info "Experimental pattern" NI_IDX I0_A = I_sel Bz_mT = 1e3 * exp_data[:BzTesla][NI_IDX]
 
    F1_mean = mean_image(exp_data[:F1ProcessedImages], NI_IDX)
    F2_mean = mean_image(exp_data[:F2ProcessedImages], NI_IDX)
    @info "Image ranges" F1 = extrema(filter(isfinite, F1_mean)) F2 = extrema(filter(isfinite, F2_mean))
 
    for (img, tag) in ((F1_mean, "f1"), (F2_mean, "f2"))
        fig = plot_heatmap_with_profile(img)
        display(fig)
        savefig(fig, "SG_img_$tag")
 
        fig = plot_image_with_z_profile(img')
        display(fig)
        savefig(fig, "SG_img_profile_$tag")
    end
end
 

##################################################################################################
## §4  COIL CURRENTS AND CQD INDUCTION-TERM RUN (shared by §5 and §6)
##################################################################################################
Icoils = [0.00,
          0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
          0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050,
          0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095,
          0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
          0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
nI = length(Icoils)
Ic = Icoils[2:end]                 # non-zero currents (log axes)

# Induction-term grids, one per simulation batch. Naming both (rather than assigning `ki_list`
# twice) keeps it unambiguous which grid `KI_IDX` indexes into.
const KI_LIST_6M = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
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
    100.0, 1000.0, 10000.0, 100000.0]              # 6M-trajectory batch
const KI_LIST_7M = collect(1.0:0.1:5.0)            # 7M-trajectory batch (the one used below)

ki_list = KI_LIST_7M

KI_IDX  = 11                       # → ki = 2.0e-6 ; file suffix "ki011"
ki_fit  = ki_list[KI_IDX] * 1e-6
@info "Induction term" ki = ki_fit  # FIX: was `ki_selected`, undefined
              
cqd_ki_path = joinpath(BASE_PATH, "SIMULATIONS", "2025_SETUP", "CQD_T205_7M", "up",
                       "cqd_7000000_ki$(lpad(KI_IDX, 3, '0'))_up_screen.jld2");   # 7M batch → KI_LIST_7M
cqd_ki = load(cqd_ki_path,"screen")[:data];

standard_error(x) = std(x; corrected = true) / sqrt(length(x))


##################################################################################################
## §5  COLLAPSE TIME vs TIME OF FLIGHT  →  collapse_time, time_flight, collapse_cycles
##################################################################################################
cqd_times = let
    # τ_c = 1 / (k_i |γ_e| B_z(I)) ; Δt_SG = L_SG / v_y, column 5 of the screen array
    # REVIEW: confirm column 5 is v_y (m/s) in the screen-array layout.
    V_COL = 5;
    collapse_time = inv.(ki_fit * abs(γₑ) * TheoreticalSimulation.BvsI.(Ic));
    tof_samples   = [y_SG ./ cqd_ki[ic][:, V_COL] for ic in 2:nI]; 
    travel_times  = mean.(tof_samples);
    travel_err    = standard_error.(tof_samples);
    n_collapses   = travel_times ./ collapse_time;
 
    @info @sprintf("Mean time of flight in the SG region: (%d ± %d) μs", 1e6 * mean(travel_times), 1e6 * std(travel_times))
 
    log_ticks_I = latex_log_ticks(-3:0)
 
    # ── collapse time ───────────────────────────────────────────────────────────────────────
    fig = Figure(size=(800,450))
    ax  = Axis(fig[1, 1]; xlabel = "SG current (A)", ylabel = "Collapse time (μs)",
        xscale = log10, yscale = log10, xticks = log_ticks_I, yticks = latex_log_ticks(0:2))
    lines!(ax, Ic, 1e6 .* collapse_time; color = :blue, linewidth = 2, label = L"Collapse time $\tau_{c}$")
    limits!(ax, 1e-3, 1, 1, 3500)
    axislegend(ax)
    display(fig)
    savefig(fig, "collapse_time")
 
    # ── time of flight ──────────────────────────────────────────────────────────────────────
    tof, err = 1e6 .* travel_times, 1e6 .* travel_err
    fig = Figure(size=(800,450))
    ax  = Axis(fig[1, 1]; xlabel = "SG current (A)", ylabel = "Time of flight (μs)",
        xscale = log10, xticks = log_ticks_I)
    band!(ax, Ic, tof .- err, tof .+ err; color = (:dodgerblue3, 0.1))
    lines!(ax, Ic, tof; color = :dodgerblue3, linewidth = 2, label = L"Time of flight $\Delta t_{\mathrm{SG}}$")
    scatter!(ax, Ic, tof; color = :white, strokecolor = :dodgerblue3, strokewidth = 1, markersize = 6)
    xlims!(ax, 1e-3, 1)
    axislegend(ax; position = :lb)
    display(fig)
    savefig(fig, "time_flight")
 
    # ── both on one axis (display only) ─────────────────────────────────────────────────────
    fig = Figure(size=(800,450))
    ax  = Axis(fig[1, 1]; xlabel = "SG current (A)", ylabel = "Time (μs)",
        xscale = log10, yscale = log10, xticks = log_ticks_I, yticks = latex_log_ticks(0:2))
    band!(ax, Ic, tof .- err, tof .+ err; color = (:dodgerblue3, 0.1))
    lines!(ax, Ic, tof; color = :dodgerblue3, linewidth = 2, label = L"Time of flight $\Delta t_{\mathrm{SG}}$")
    lines!(ax, Ic, 1e6 .* collapse_time; color = :darkgreen, linewidth = 2, label = L"Collapse time $\tau_{c}$")
    limits!(ax, 1e-3, 1, 1, 4000)
    axislegend(ax; position = :lb)
    display(fig)
 
    # ── number of collapse cycles Δt_SG / τ_c ───────────────────────────────────────────────
    fig = Figure(size = (800, 450))
    ax  = Axis(fig[1, 1]; xlabel = "SG current (A)", ylabel = "Interaction time / collapse time",
        xscale = log10, yscale = log10,
        xticks = latex_log_ticks(-2:0), yticks = latex_log_ticks(-1:2),
        xgridvisible = true, xminorgridvisible = true, xminorticksvisible = true, xminorticks = IntervalsBetween(9),
        ygridvisible = true, yminorgridvisible = true, yminorticksvisible = true, yminorticks = IntervalsBetween(9),
    )
    hspan!(ax, 0.1, 1; color = (:black, 0.2))          # fewer than one collapse per transit
    lines!(ax, Ic, n_collapses; color = :red, linewidth = 2, label = L"$\Delta t_{\mathrm{SG}} / \tau_{c}$")
    limits!(ax, 1e-2, 1, 0.7, 40)
    axislegend(ax; position = :lt)
    display(fig)
    savefig(fig, "collapse_cycles")
 
    (; collapse_time, travel_times, n_collapses)
end



##################################################################################################
## §6  MODEL–EXPERIMENT AGREEMENT vs COIL CURRENT AND vs COLLAPSE CYCLES
##      →  relerr_vs_current, pull_vs_current, relerr_vs_collapsecycles, pull_vs_collapsecycles
##################################################################################################
# Two complementary measures of the same disagreement, read from the fit archive rather than
# hard-coded, so the figures track whatever the analysis last wrote:
#
#   Fractional deviation : (z_model − z_exp) / z_exp   — normalised by the value, unit-free
#   Normalized residual  : (z_model − z_exp) / σ_exp   — normalised by the uncertainty, so
#                                                        |value| ≲ 1 means "agrees within errors"
#
# The collapse-cycle abscissa comes from §5: N(I) = Δt_SG(I)/τ_c(I) is known on the `Ic` grid and
# is interpolated (log–log) onto the archive's current grid. Both N and I span decades, so
# interpolating log N against log I is far more faithful than a linear interpolation would be.

"""
    read_stats(path) -> NamedTuple
 
Read the `statistical_analysis/` group of the k_i-scale fit archive and return
`(; I, relErr_QM, relErr_CQD, pull_QM, pull_CQD)`, sorted by increasing current.
 
All five datasets are read in a single `jldopen` so the file is opened once, and their lengths
are checked against each other: a mismatch means the archive was written from runs with
different current grids and any plot made from it would silently pair the wrong points.
"""
function read_stats(path::AbstractString)
    isfile(path) || throw(ArgumentError("JLD2 archive not found: $path"))
 
    I, relErr_QM, relErr_CQD, pull_QM, pull_CQD = jldopen(path, "r") do f
        (vec(f["statistical_analysis/Current_A"]),
         vec(f["statistical_analysis/relErr_QM"]),
         vec(f["statistical_analysis/relErr_CQD"]),
         vec(f["statistical_analysis/pull_QM"]),
         vec(f["statistical_analysis/pull_CQD"]))
    end
 
    lengths = length.((I, relErr_QM, relErr_CQD, pull_QM, pull_CQD))
    all(==(lengths[1]), lengths) ||
        throw(DimensionMismatch("statistical_analysis columns have different lengths: $lengths"))
 
    p = sortperm(I)   # guarantee monotone x for line plots regardless of write order
    return (; I = I[p], relErr_QM = relErr_QM[p], relErr_CQD = relErr_CQD[p],
              pull_QM = pull_QM[p], pull_CQD = pull_CQD[p])
end

# Axis labels shared by the current- and collapse-cycle versions of each panel, so the two
# abscissae never disagree about what the ordinate means.
const LABEL_RELERR = L"Fractional deviation, $(z_{\mathrm{model}}-z_{\mathrm{exp}})/z_{\mathrm{exp}}$"
const LABEL_PULL   = L"Normalized residual, $(z_{\mathrm{model}}-z_{\mathrm{exp}})/\sigma_{\mathrm{exp}}$"

list_jld2_entries(FIT_ARCHIVE);          # printed reference; keys are returned, not captured here
 
stats = read_stats(FIT_ARCHIVE);
@info "Fit archive" n_points = length(stats.I) I_range = extrema(stats.I)

# The pair reads as fractional deviation (normalized by the value) 
# and normalized residual (normalized by the uncertainty), which makes their relationship obvious at a glance.


# Axis styling shared by all four panels of this section: a log abscissa with nine minor
# intervals per decade, and a two-level grid — solid at the labelled ticks, dotted between them.
# Splatted into every Axis below, so the four panels cannot drift apart typographically.
const AX_COMMON = (
    xscale = log10,
    xminorticksvisible = true, xminorticks = IntervalsBetween(9),

    xgridvisible = true, ygridvisible = true,
    xgridcolor = (:black, 0.12), ygridcolor = (:black, 0.12),
    xgridwidth = 0.65, ygridwidth = 0.65,

    xminorgridvisible = true,
    xminorgridcolor = (:black, 0.10), yminorgridcolor = (:black, 0.10),
    xminorgridwidth = 0.5, yminorgridwidth = 0.5,
    xminorgridstyle = :dot, yminorgridstyle = :dot,
)

# Ordinate minor structure. The deviation panels subdivide each 0.05 step into fifths; the
# residual panels leave it off, because the integer-σ major gridlines already carry the reading.
const AX_YMINOR    = (yminorticksvisible = true,  yminorticks = IntervalsBetween(5),
                      yminorgridvisible = true)
const AX_NO_YMINOR = (yminorticksvisible = false, yminorgridvisible = false)

# `limits` is pinned to the outermost major ticks in every panel: IntervalsBetween only fills the
# gaps *between* visible major ticks, so a tick outside the limits leaves that side of the axis
# without minor ticks or minor gridlines.

# ── Fractional deviation vs coil current ────────────────────────────────────────────────────
let
    yt = snapped_ticks(stats.relErr_QM, stats.relErr_CQD; step = 0.05)

    fig = Figure(size = (800, 450))
    ax  = Axis(fig[1, 1];
        xlabel = "SG current (A)", ylabel = LABEL_RELERR,
        xticks = latex_log_ticks(-2:0), yticks = yt,
        limits = ((18e-3, 1.1), (first(yt), last(yt))),
        AX_COMMON..., AX_YMINOR...,
    )
    hlines!(ax, 0; color = (:black, 0.4), linestyle = :dash)
    scatterlines!(ax, stats.I, stats.relErr_QM;  color = COLOR_QM,  markersize = 8, label = LABEL_QM)
    scatterlines!(ax, stats.I, stats.relErr_CQD; color = COLOR_CQD, markersize = 8, label = LABEL_CQD)
    axislegend(ax; position = :rt)
    display(fig)
    savefig(fig, "relerr_vs_current")
end

# ── Normalized residual vs coil current ─────────────────────────────────────────────────────
let
    yt = snapped_ticks(stats.pull_QM, stats.pull_CQD; step = 1.0)

    fig = Figure(size = (800, 450))
    ax  = Axis(fig[1, 1];
        xlabel = "SG current (A)", ylabel = LABEL_PULL,
        xticks = latex_log_ticks(-2:0), yticks = yt,
        limits = ((18e-3, 1.1), (first(yt), last(yt))),
        AX_COMMON..., AX_NO_YMINOR...,
    )
    # ±1σ band: points inside it agree with the measurement within its uncertainty
    # hspan!(ax, -1, 1; color = (:black, 0.10))
    hlines!(ax, 0; color = (:black, 0.4), linestyle = :dash)
    scatterlines!(ax, stats.I, stats.pull_QM;  color = COLOR_QM,  markersize = 8, label = LABEL_QM)
    scatterlines!(ax, stats.I, stats.pull_CQD; color = COLOR_CQD, markersize = 8, label = LABEL_CQD)
    axislegend(ax; position = :lt)
    display(fig)
    savefig(fig, "pull_vs_current")
end

# ── Same quantities against the number of collapse cycles ───────────────────────────────────
# Requires `Ic` (§4) and `cqd_times.n_collapses` (§5).
let
    N_at_I = interp_loglog(stats.I, Ic, cqd_times.n_collapses)

    keep = .!isnan.(N_at_I)
    all(keep) || @warn "Currents outside the simulated grid were dropped" n_dropped = count(!, keep) I_sim = extrema(Ic)

    o  = sortperm(N_at_I[keep])          # monotone x so the connecting line does not double back
    dτ = N_at_I[keep][o]
    eQM, eCQD = stats.relErr_QM[keep][o], stats.relErr_CQD[keep][o]
    pQM, pCQD = stats.pull_QM[keep][o],   stats.pull_CQD[keep][o]

    pow_lo, pow_hi = floor(Int, log10(minimum(dτ))), ceil(Int, log10(maximum(dτ)))
    xt = latex_log_ticks(pow_lo:pow_hi)
    xmt = [k * exp10(p) for p in pow_lo:pow_hi for k in 2:9]
    dlo, dhi = extrema(dτ)
    dec_lo, dec_hi = exp10(floor(log10(dlo))), exp10(floor(log10(dhi)))
    xlo = floor(dlo / dec_lo) * dec_lo        # 0.82 → 0.8
    xhi = ceil( dhi / dec_hi) * dec_hi        # 32   → 40
    # Half a grid unit of breathing room, so the end ticks are inside the panel, not on its edge.
    # When the lower bound lands exactly on a decade, use the finer unit below it (1 → 0.95,
    # not 0.5, which on a log axis is half a decade of empty space).
    unit_lo = xlo ≈ dec_lo ? dec_lo / 10 : dec_lo
    xl = (xlo - unit_lo / 4, xhi + dec_hi / 4)   # 0.75 … 45

    # Fractional deviation vs collapse cycles
    yt = snapped_ticks(eQM, eCQD; step = 0.05)
    fig = Figure(size = (800, 450))
    ax  = Axis(fig[1, 1];
        xlabel = "Number of collapse cycles", ylabel = LABEL_RELERR,
        xticks = xt, yticks = yt,
        limits = (xl, (first(yt), last(yt))),
        AX_COMMON..., AX_YMINOR...,
    )
    hlines!(ax, 0; color = (:black, 0.4), linestyle = :dash)
    scatterlines!(ax, dτ, eQM;  color = COLOR_QM,  markersize = 8, label = LABEL_QM)
    scatterlines!(ax, dτ, eCQD; color = COLOR_CQD, markersize = 8, label = LABEL_CQD)
    axislegend(ax; position = :rt)
    display(fig)
    savefig(fig, "relerr_vs_collapsecycles")

    # Normalized residual vs collapse cycles
    yt = snapped_ticks(pQM, pCQD; step = 1.0)
    fig = Figure(size = (800, 450))
    ax  = Axis(fig[1, 1];
        xlabel = "Number of collapse cycles", ylabel = LABEL_PULL,
        xticks = xt, yticks = yt,
        limits = (xl, (first(yt), last(yt))),
        AX_COMMON..., AX_NO_YMINOR...,
    )
    # hspan!(ax, -1, 1; color = (:black, 0.10))
    hlines!(ax, 0; color = (:black, 0.4), linestyle = :dash)
    scatterlines!(ax, dτ, pQM;  color = COLOR_QM,  markersize = 8, label = LABEL_QM)
    scatterlines!(ax, dτ, pCQD; color = COLOR_CQD, markersize = 8, label = LABEL_CQD)
    axislegend(ax; position = :lt)
    display(fig)
    savefig(fig, "pull_vs_collapsecycles")
end


##################################################################################################
## §7  BEAM SPLITTING: MEASURED F=1 POSITION vs CQD AND QM PREDICTIONS
##      →  zF1_vs_current, zF1_vs_gradient
##################################################################################################
# The manuscript's main comparison. The measured centroid of the F = 1 component, z_F1, is shown
# with its uncertainty at each of the 26 measured settings; the CQD and QM predictions are drawn
# as continuous curves on the model's dense 801-point grid, so the models read as theory curves
# through data points rather than as another series of markers.
#
# The two panels show the same data against the two equivalent control variables: the coil
# current, which is what is actually set in the laboratory, and the field gradient ∂zBz, which is
# what the physics depends on. Since the gradient is obtained from the current through the magnet
# calibration, the panels are not independent evidence — the gradient axis is the physical one,
# the current axis the operational one.
 
"""
    read_comparison(path) -> (experiment, model)
 
Read the `experiment/` and `model/` groups of the k_i-scale archive and return two NamedTuples.
 
`experiment` carries `(; I, Ierr, G, Gerr, z, zerr, cqd, qm)` — the 26 measured settings with
their uncertainties and the model values evaluated at exactly those settings — and `model`
carries `(; I, G, cqd, qm)` on the dense grid used for the theory curves. Both are sorted by
increasing current, and the columns within each group are length-checked so a partially written
archive is caught here rather than producing silently mispaired points.
"""
function read_comparison(path::AbstractString)
    isfile(path) || throw(ArgumentError("JLD2 archive not found: $path"))
 
    e, m = jldopen(path, "r") do f
        exp_cols = (; I    = vec(f["experiment/Current_A"]),
                      Ierr = vec(f["experiment/CurrentErr_A"]),
                      G    = vec(f["experiment/Gradient_Tm"]),
                      Gerr = vec(f["experiment/GradientErr_Tm"]),
                      z    = vec(f["experiment/zF1_mm"]),
                      zerr = vec(f["experiment/zF1Err_mm"]),
                      cqd  = vec(f["experiment/CQD_up_mm"]),
                      qm   = vec(f["experiment/QM_zF1_mm"]))
        mod_cols = (; I   = vec(f["model/Current_A"]),
                      G   = vec(f["model/Gradient_Tm"]),
                      cqd = vec(f["model/CQD_up_mm"]),
                      qm  = vec(f["model/QM_zF1_mm"]))
        (exp_cols, mod_cols)
    end
 
    for (name, nt) in (("experiment", e), ("model", m))
        lengths = length.(values(nt))
        all(==(first(lengths)), lengths) ||
            throw(DimensionMismatch("$name columns have different lengths: $lengths"))
    end
 
    # Sort both grids by current so the theory curves are drawn monotonically and the
    # experimental markers are in a predictable order.
    pe, pm = sortperm(e.I), sortperm(m.I)
    e, m = map(v -> v[pe], e), map(v -> v[pm], m)
 
    # The gradient panel plots against G while the arrays are ordered by I; that is only a valid
    # ordering if the magnet calibration G(I) is monotone. Fail loudly rather than draw a curve
    # that doubles back on itself.
    issorted(m.G) || @warn "Model gradient is not monotone in current — check the calibration"
 
    return (e, m)
end
 

 
 
"""
    plot_splitting(x_exp, xerr, z, zerr, x_model, cqd, qm; kwargs...) -> Figure
 
Measured F = 1 position with error bars against `x_exp`, overlaid with the CQD and QM curves
evaluated on the dense model grid `x_model`.
 
Both coordinates of the measurement carry an uncertainty, so horizontal and vertical error bars
are drawn; the markers are open (white fill, coloured stroke) so that error bars and overlapping
theory curves stay visible underneath them. The model curves are drawn first and the data last,
so no curve hides a data point.
 
Both axes are logarithmic by default: the abscissa spans decades, and so does the splitting
itself, so a log–log frame shows the small-current behaviour that a linear ordinate compresses
into the bottom of the panel, and turns a power law into a straight line.
 
# Keyword arguments
- `xlabel`, `ylabel`: axis labels (LaTeX strings).
- `xscale`, `yscale`: `log10` by default. Points a log axis cannot represent (non-positive
  abscissa or ordinate) are dropped per series, with a warning naming how many.
- `figsize`: figure size in points.
- `exp_color`: colour of the measured points; CQD and QM use the section-wide constants.
- `legend_position`: passed to `axislegend`.
- `label_exp`, `label_qm`, `label_cqd`: legend entries, listed in that order regardless of the
  order in which the series are drawn.
- `legend_patchsize`: size of the legend's line/marker swatch, (width, height) in points — a
  wide patch makes the line styles easy to tell apart.
- `marker_fill`: fill of the measured markers, independent of their outline (`exp_color`).
  `:transparent` leaves the disc empty so a theory curve remains readable through it;
  `(:white, α)` masks the curve by the fraction `α`, which keeps dense clusters of points
  legible. Opaque `:white` hides whatever passes underneath.
"""
function plot_splitting(x_exp, xerr, z, zerr, x_model, cqd, qm;
        xlabel,
        ylabel          = L"${F=1}$ Peak position (mm)",
        xscale          = log10,
        yscale          = log10,
        figsize         = (800, 600),
        exp_color       = :black,
        legend_position = :lt,
        label_exp       = LABEL_EXP,
        label_qm        = LABEL_QM,
        label_cqd       = LABEL_CQD,
        legend_patchsize = (45, 12),
        marker_fill      = (:white, 0.45),
)
    # A log axis cannot show non-positive values (e.g. a zero-current reference point, or a
    # model prediction that crosses zero). Mask each series on its own, so one unplottable
    # point in a model curve does not remove the other curve or the data.
    _mask(v, scale) = scale === log10 ? v .> 0 : trues(length(v))
    _report(m, what) = all(m) || @warn "Points dropped: not representable on a log axis" series = what n = count(!, m)
 
    keep_e   = _mask(x_exp, xscale)   .& _mask(z,   yscale)
    keep_qm  = _mask(x_model, xscale) .& _mask(qm,  yscale)
    keep_cqd = _mask(x_model, xscale) .& _mask(cqd, yscale)
    _report(keep_e,   "experiment")
    _report(keep_qm,  "QM")
    _report(keep_cqd, "CQD")
 
    fig = Figure(size = figsize)
    ax  = Axis(fig[1, 1];
        xlabel, ylabel,
        xscale, yscale,
        xticks = decade_ticks(x_exp[keep_e], x_model[keep_qm]),
        yticks = decade_ticks(z[keep_e], qm[keep_qm], cqd[keep_cqd]),
        xminorticksvisible = true, xminorticks = IntervalsBetween(9),
        yminorticksvisible = true, yminorticks = IntervalsBetween(9),
        # xticklabelsize = 18, yticklabelsize = 18,

        # major grid at the decades
        xgridvisible = true, ygridvisible = true,
        xgridcolor = (:black, 0.15), ygridcolor = (:black, 0.15),
        xgridwidth = 0.65, ygridwidth = 0.65,

        # minor grid at 2…9 × each decade
        xminorgridvisible = true, yminorgridvisible = true,
        xminorgridcolor = (:black, 0.12), yminorgridcolor = (:black, 0.12),
        xminorgridwidth = 0.5, yminorgridwidth = 0.5,
        xminorgridstyle = :dot, yminorgridstyle = :dot,
    )
 
    # Theory first, so the measurements sit on top of the curves. The plot handles are kept so
    # the legend can be ordered independently of this drawing order (see axislegend below).
    p_qm  = lines!(ax, x_model[keep_qm],  qm[keep_qm];   color = COLOR_QM,  linewidth = 2.5)
    p_cqd = lines!(ax, x_model[keep_cqd], cqd[keep_cqd]; color = COLOR_CQD, linewidth = 2.5)
 
    # Measurement: both error bars, then open markers.
    # On a log ordinate a symmetric bar would reach z - zerr ≤ 0 when the error is comparable to
    # the value; the lower whisker is clipped just inside the axis so the bar stays drawable.
    zlo = yscale === log10 ? min.(zerr[keep_e], z[keep_e] .* 0.999) : zerr[keep_e]
    p_err = errorbars!(ax, x_exp[keep_e], z[keep_e], zlo, zerr[keep_e];
                       color = exp_color, whiskerwidth = 6)
 
    xlo = xscale === log10 ? min.(xerr[keep_e], x_exp[keep_e] .* 0.999) : xerr[keep_e]
    errorbars!(ax, x_exp[keep_e], z[keep_e], xlo, xerr[keep_e]; color = exp_color, whiskerwidth = 6,
               direction = :x)
 
    # `color` is the disc fill, `strokecolor` the outline: a semi-transparent fill lets the
    # theory curves stay visible through the markers without losing the point outlines.
    p_exp = scatter!(ax, x_exp[keep_e], z[keep_e];
        color = marker_fill, strokecolor = exp_color, strokewidth = 1.5, markersize = 9)
 
    # Explicit entries: the legend reads Experiment → QM → CQD whatever the draw order, and the
    # first entry combines the error bar with the marker so the swatch matches what is plotted.
    axislegend(ax,
        [[p_err, p_exp], p_qm, p_cqd],
        [label_exp, label_qm, label_cqd];
        position     = legend_position,
        framevisible = false,              # no box around the legend
        patchsize    = legend_patchsize,   # longer line swatches
    )
    return fig
end
 
experiment, model = read_comparison(FIT_ARCHIVE);
@info "Splitting comparison" n_exp = length(experiment.I) n_model = length(model.I) I_range = extrema(experiment.I) G_range = extrema(experiment.G)
 
# ── z_F1 vs coil current (the operational control variable) ─────────────────────────────────
let
    fig = plot_splitting(experiment.I, experiment.Ierr, experiment.z, experiment.zerr,
                         model.I, model.cqd, model.qm;
                         xlabel = "SG current (A)")
    display(fig)
    savefig(fig, "zF1_vs_current")
end
 
# ── z_F1 vs field gradient (the physical control variable) ──────────────────────────────────
let
    fig = plot_splitting(experiment.G, experiment.Gerr, experiment.z, experiment.zerr,
                         model.G, model.cqd, model.qm;
                         xlabel = "SG magnetic field gradient (T/m)")
    display(fig)
    savefig(fig, "zF1_vs_gradient")
end


##################################################################################################
## §8  RUN MANIFEST
##################################################################################################
# Written last so the figure listing is complete. Parameters are passed only if their section was
# evaluated, which keeps every section independently runnable.
let
    extras = Dict{Symbol,Any}()
    @isdefined(ki_fit)  && (extras[:ki_fit]  = ki_fit)
    @isdefined(KI_IDX)  && (extras[:ki_index] = KI_IDX)
    @isdefined(NI_IDX)  && (extras[:image_current_index] = NI_IDX)
    @isdefined(T_K)     && (extras[:furnace_K] = T_K)
    write_manifest(; extras...)
end
 
@info "Figures written to" OUTDIR elapsed = Dates.canonicalize(Dates.now() - T_START)