##################################################################################################
#  PREPARATION OF FIGURES FOR FINAL MANUSCRIPT
#  Kelvin Titimbo — California Institute of Technology — July 2026
##################################################################################################
#
#  REVIEW SUMMARY (search the file for "REVIEW:" and "FIX:" to find each item in context)
#  ----------------------------------------------------------------------------------------------
#  Structure
#   * Every figure now lives in its own `let ... end` block with a numbered banner. Each block is
#     self-contained (creates its own `fig`/`ax`, saves, displays) and does not leak globals, so
#     you can evaluate any section on its own after §0 has run. Sections that must share results
#     (collapse time → relative-error plot) hand them over through a single NamedTuple.
#   * All `save` calls go through one `savefig` helper that (a) honours SAVE_FIG, which was
#     defined but never used, and (b) writes every extension in FIG_EXTS, instead of each figure
#     hard-coding its own .png/.pdf pair and its own px_per_unit.
#   * A shared publication Theme carries the font sizes that were copy-pasted into every Axis.
#     Sections now only state what is figure-specific.
#   * Two small formatters (`latex_int`, `latex_log_ticks`) replace six copies of the same
#     tick-label lambda.
#
#  Bugs fixed (marked FIX:)
#   1. `@info "Induction term" ki=ki_selected` — `ki_selected` was never defined; script would
#      abort there. Now logs `ki_fit`.
#   2. `for ic = 2:47` hard-coded the length of `Icoils`; now `2:nI`.
#   3. `const TWOπ`, `const INV_E` were defined twice.
#   4. `err2` was recomputed with the literal 0.07 instead of `y_SG` (same value, but a
#      maintenance trap); the duplicate is gone.
#   5. Bare `fig` at the end of a block does nothing in a script (only in the REPL); replaced by
#      `display(fig)`.
#   6. The first `ki_list` (the 6M list) was dead code, overwritten immediately.
#
#  Scientific / manuscript concerns (marked REVIEW:, not changed)
#   A. §6 relative-error curves are hand-digitized AND several CQD points are multiplied by
#      ad-hoc factors (−0.8×, 0.6×, ...). For a published figure this must be recomputed from the
#      actual ℰ_exp, ℰ_QM, ℰ_CQD arrays, or the provenance documented. As written it cannot be
#      reproduced.
#   B. §6 legend/colour mismatch: comments say QM = red, CQD = blue; the plot draws QM blue,
#      CQD red. Numerator uses `\mathcal{z}` (lowercase \mathcal is undefined in LaTeX) while
#      the denominator uses `\mathcal{E}`.
#   C. §1 `hspan!(ax, -3, 3)` is labelled "slit opening extent in z" but z_slit = 0.3 mm; the
#      ±3 mm band is something else (magnet gap? beam envelope?). Name it.
#   D. §3 the "top profile" is the mean over x, i.e. the profile ALONG z — the Stern–Gerlach
#      splitting direction — not a transverse profile. Comment corrected; check the caption.
#   E. §3 `y_scale = 4` silently multiplies the x-pixel tick labels by 4 (binning?). Make it a
#      named constant tied to the actual processing binning, or read it from the JLD2 file.
#   F. §3 two functions draw the same image with opposite `yreversed`; make sure the orientation
#      in the manuscript is consistent across panels.
#   G. §7 is unfinished: one file loaded from ".../smoothing_binning_xkl", another from
#      ".../smoothing_binning_2025", neither plotted.
#
#  Housekeeping
#   * ~15 packages are loaded but unused in this file (Interpolations, Loess, Optim, BSplineKit,
#     Polynomials, DSP, LambertW, PolyLog, Alert, DataStructures, CSV, DataFrames, DelimitedFiles,
#     Distributions, StaticArrays, ...). They cost tens of seconds of load time. I left them in
#     place because the `include`d module files may rely on them being in Main — prune once you
#     have confirmed those files carry their own `using` statements.
#   * `const` bindings that depend on time (RUN_STAMP, OUTDIR) make the script non-re-includable
#     in the same session (redefining a const with a new value is an error). Plain globals are
#     safer for a script.
#   * `rng_set`, `dimF`, `number_precessions`, `F1_mean_norm`, `F2_mean_norm`, `fig_F1`, `fig_F2`
#     were computed but never used.
#   * For vector output CairoMakie ignores `px_per_unit`; `pt_per_unit` (default 1) governs
#     PDF/SVG. Figure `size` is therefore in points for the PDF, i.e. (1120, 600) ≈ 15.5 × 8.3 in.
#     Consider sizing figures to the column width you will use in LaTeX (e.g. 3.4 in = 245 pt)
#     with a larger `fontsize`, instead of shrinking a large figure in `\includegraphics`.
##################################################################################################
 
 
##################################################################################################
## §0  ENVIRONMENT, OUTPUT, PHYSICAL CONSTANTS, APPARATUS GEOMETRY
##################################################################################################

# ── Plotting ────────────────────────────────────────────────────────────────────────────────────
using CairoMakie
using Colors, ColorSchemes
using LaTeXStrings, Printf, PrettyTables

SAVE_FIG        = true
FIG_EXTS        = ("png", "pdf", "svg")   # every figure is written in all of these
FIG_PX_PER_UNIT = 3                # raster resolution; ignored for pdf/svg

# ── Time-stamping ───────────────────────────────────────────────────────────────────────────────
using Dates
const T_START = Dates.now()

# ── Numerics ────────────────────────────────────────────────────────────────────────────────────
using LinearAlgebra, Roots, StatsBase
using Random, Statistics, NaNStatistics
# REVIEW: unused in this file — Interpolations, Loess, Optim, BSplineKit, Polynomials, DSP,
#         LambertW, PolyLog, Alert, DataStructures, Distributions, StaticArrays. Keep only if the
#         included module files need them in Main.
using Interpolations, Loess, Optim, BSplineKit, Polynomials, DSP, LambertW, PolyLog, Alert
using DataStructures, Distributions, StaticArrays

# ── Data I/O ────────────────────────────────────────────────────────────────────────────────────
using OrderedCollections, JLD2
using DelimitedFiles, CSV, DataFrames   # REVIEW: unused in this file

# ── Threads ─────────────────────────────────────────────────────────────────────────────────────
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads"  count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()

# ── Paths ───────────────────────────────────────────────────────────────────────────────────────
cd(@__DIR__)
const BASE_PATH = raw"F:\SternGerlachExperiments"
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss")   # REVIEW: const + timestamp → not re-includable
const OUTDIR    = joinpath(@__DIR__, "data_studies", "FINAL_IMAGES_" * RUN_STAMP)
isdir(OUTDIR) || mkpath(OUTDIR)
@info "Created output directory" OUTDIR

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
const TWOπ  = 2π          # FIX: was defined twice
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


# ── Shared figure infrastructure ────────────────────────────────────────────────────────────────
"""
    savefig(fig, name; px_per_unit = FIG_PX_PER_UNIT, exts = FIG_EXTS)
 
Write `fig` to `OUTDIR/name.<ext>` for every extension in `exts`. No-op when `SAVE_FIG == false`.
`px_per_unit` only affects raster formats; vector formats use `pt_per_unit` (CairoMakie default 1).
"""
function savefig(fig, name::AbstractString; px_per_unit = FIG_PX_PER_UNIT, exts = FIG_EXTS)
    SAVE_FIG || return nothing
    for ext in exts
        save(joinpath(OUTDIR, "$name.$ext"), fig; px_per_unit)
    end
    return nothing
end

"Integer tick labels typeset by MathTeXEngine, e.g. `-2` → `L\"-2\"`."
latex_int(vs) = [L"%$(round(Int, v))" for v in vs]
 
"Decade ticks `(positions, labels)` for a log10 axis, e.g. `latex_log_ticks(-3:0)`."
latex_log_ticks(pows) = (exp10.(pows), [L"10^{%$p}" for p in pows])

# Manuscript-wide typography. Per-figure overrides below only state what differs.
# REVIEW: if the journal has a column width, set `size` per figure to that width in pt and raise
#         `fontsize` so text is ≥ 8 pt after \includegraphics, rather than scaling down large figures.
const PUB_THEME = Theme(
    fontsize = 16,
    Axis = (
        xlabelsize = 20, ylabelsize = 20,
        xticklabelsize = 16, yticklabelsize = 16,
    ),
    Legend = (labelsize = 16,),
)
set_theme!(PUB_THEME)


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
 
    # REVIEW (C): ±3 mm is NOT the slit extent (z_slit = 0.3 mm). Name what this band represents
    #             and derive it from a constant (e.g. the pole-gap half-height).
    hspan!(ax, -3.0, 3.0; color = (:gray36, 0.55))
 
    # Pre-SG slit, drawn on top of the band
    hw_x, hw_z = 1e3 * x_slit / 2, 1e3 * z_slit / 2
    poly!(ax, Point2f.([-hw_x, -hw_x, hw_x, hw_x], [-hw_z, hw_z, hw_z, -hw_z]);
        color = :white, strokecolor = :black, strokewidth = 1.5)
 
    resize_to_layout!(fig)
    display(fig)
    savefig(fig, "SG_geometry"; px_per_unit = 2)
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
        xlabel = L"Current $(\mathrm{A})$", ylabel = L"$\mu_{F}/\mu_{B}$",
        xticks = latex_log_ticks(-3:0),               # explicit decades: no surprise minor labels
        ytickformat = ys -> [iszero(y) ? L"0" : L"%$(round(y, digits = 1))" for y in ys],
        ytickalign = 0.5, yminortickalign = 0.5, yticksize = 10, yminorticksize = 5,
        ticks_kw...)
 
    # Dash pattern in units of linewidth: [start, on, off] — 8 on, 2 off, period 10
    LONG_DASH = Linestyle([0.0, 8.0, 10.0])   # REVIEW: old comment said 12 on / 6 off — did not match the numbers
 
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
        figsize       = (600, 400),
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


exp_data = load_experiment(joinpath(BASE_PATH, "EXPERIMENTS", "20260220", "data_processed.jld2"))
 
let
    nI_idx = 16
    I_sel  = exp_data[:Currents][nI_idx]
    @info "Experimental pattern" nI_idx I0_A = I_sel Bz_mT = 1e3 * exp_data[:BzTesla][nI_idx]
 
    F1_mean = mean_image(exp_data[:F1ProcessedImages], nI_idx)
    F2_mean = mean_image(exp_data[:F2ProcessedImages], nI_idx)
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

ki_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
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
    100.0, 1000.0, 10000.0, 100000.0];#6M
ki_list = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 
            5.0] # 7M

KI_IDX  = 11                       # → ki = 2.0e-6 ; file suffix "ki011"
ki_fit  = ki_list[KI_IDX] * 1e-6
@info "Induction term" ki = ki_fit  # FIX: was `ki_selected`, undefined
        
cqd_ki_path = joinpath(BASE_PATH, "SIMULATIONS", "2025_SETUP", "CQD_T205_7M", "up",
                       "cqd_7000000_ki$(lpad(KI_IDX, 3, '0'))_up_screen.jld2");
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
    tof_samples   = [y_SG ./ cqd_ki[ic][:, V_COL] for ic in 2:nI];      # FIX: was 2:47
    travel_times  = mean.(tof_samples);
    travel_err    = standard_error.(tof_samples);
    n_collapses   = travel_times ./ collapse_time;
 
    @info @sprintf("Mean time of flight in the SG region: (%d ± %d) μs", 1e6 * mean(travel_times), 1e6 * std(travel_times))
 
    log_ticks_I = latex_log_ticks(-3:0)
 
    # ── collapse time ───────────────────────────────────────────────────────────────────────
    fig = Figure()
    ax  = Axis(fig[1, 1]; xlabel = "Current (A)", ylabel = "Collapse time (μs)",
        xscale = log10, yscale = log10, xticks = log_ticks_I, yticks = latex_log_ticks(0:2))
    lines!(ax, Ic, 1e6 .* collapse_time; color = :blue, linewidth = 2, label = L"Collapse time $\tau_{c}$")
    limits!(ax, 1e-3, 1, 1, 3500)
    axislegend(ax)
    display(fig)
    savefig(fig, "collapse_time")
 
    # ── time of flight ──────────────────────────────────────────────────────────────────────
    tof, err = 1e6 .* travel_times, 1e6 .* travel_err
    fig = Figure()
    ax  = Axis(fig[1, 1]; xlabel = "Current (A)", ylabel = "Time of flight (μs)",
        xscale = log10, xticks = log_ticks_I)
    band!(ax, Ic, tof .- err, tof .+ err; color = (:dodgerblue3, 0.1))
    lines!(ax, Ic, tof; color = :dodgerblue3, linewidth = 2, label = L"Time of flight $\Delta t_{\mathrm{SG}}$")
    scatter!(ax, Ic, tof; color = :white, strokecolor = :dodgerblue3, strokewidth = 1, markersize = 6)
    xlims!(ax, 1e-3, 1)
    axislegend(ax; position = :lb)
    display(fig)
    savefig(fig, "time_flight")
 
    # ── both on one axis (display only) ─────────────────────────────────────────────────────
    fig = Figure()
    ax  = Axis(fig[1, 1]; xlabel = "Current (A)", ylabel = "Time (μs)",
        xscale = log10, yscale = log10, xticks = log_ticks_I, yticks = latex_log_ticks(0:2))
    band!(ax, Ic, tof .- err, tof .+ err; color = (:dodgerblue3, 0.1))
    lines!(ax, Ic, tof; color = :dodgerblue3, linewidth = 2, label = L"Time of flight $\Delta t_{\mathrm{SG}}$")
    lines!(ax, Ic, 1e6 .* collapse_time; color = :darkgreen, linewidth = 2, label = L"Collapse time $\tau_{c}$")
    limits!(ax, 1e-3, 1, 1, 4000)
    axislegend(ax; position = :lb)
    display(fig)
 
    # ── number of collapse cycles Δt_SG / τ_c ───────────────────────────────────────────────
    fig = Figure(size = (800, 600))
    ax  = Axis(fig[1, 1]; xlabel = "Current (A)", ylabel = "Interaction time / collapse time",
        xscale = log10, yscale = log10,
        xticks = latex_log_ticks(-2:0), yticks = latex_log_ticks(-1:2),   # REVIEW: was mixed L"1", L"10", L"100"
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
## §6  RELATIVE ERROR vs NUMBER OF COLLAPSE CYCLES  →  relerr_vs_collapsecycles
##################################################################################################








# Current (A) — log-spaced sample grid
I_A = [0.0266, 0.0292, 0.0320, 0.0352, 0.0386, 0.0424, 0.0465, 0.0510,
       0.0560, 0.0614, 0.0674, 0.0740, 0.0812, 0.0891, 0.0978, 0.1073,
       0.1178, 0.1293, 0.1419, 0.1557, 0.1709, 0.1875, 0.2058, 0.2258,
       0.2478, 0.2720, 0.2985, 0.3276, 0.3595, 0.3946, 0.4330, 0.4752,
       0.5215, 0.5724, 0.6281, 0.6894, 0.7565, 0.8303, 0.9112, 1.0000]

# Red curve — Rel.Error (ℰ_QM − ℰ_exp)/ℰ_exp
rel_err_QM = [0.1457, 0.1717, 0.1935, 0.2054, 0.2115, 0.2112, 0.2066, 0.2000,
              0.1921, 0.1804, 0.1696, 0.1587, 0.1500, 0.1430, 0.1370, 0.1304,
              0.1196, 0.1045, 0.0913, 0.0826, 0.0777, 0.0699, 0.0609, 0.0511,
              0.0413, 0.0326, 0.0294, 0.0294, 0.0326, 0.0370, 0.0370, 0.0320,
              0.0239, 0.0130, 0.0033, -0.0022, 0.0005, 0.0065, 0.0043, -0.0087]

# Blue curve — Rel.Error (ℰ_CQD − ℰ_exp)/ℰ_exp
rel_err_CQD = [-0.0087, 0.0000, 0.0056, 0.0076, 0.0076, 0.0057, 0.0033, 0.0000,
               -0.0022, -0.0043, -0.0065, -0.0087, -0.0065, -0.0043, 0.0000, 0.0022,
                0.0022, -0.0032, -0.0076, -0.0087, -0.0065, -0.0065, -0.0087, -0.0130,
               -0.8*0.0174, -0.8*0.0196, -0.8*0.0196, -0.8*0.0152, -0.8*0.0087, -0.8*0.0011, 0.8*0.0022, 0.8*0.0000,
               -0.6*0.0042, -0.6*0.030, 0.6*0.00407, 0.6*0.0029, 0.6*0.0046, -0.6*0.0069, -0.6*0.0109, -0.6*0.0217]

N_collapses = travel_times ./ collapse_time

# --- digitized error curves live on I_A; N_collapse lives on Ic ---
# (I_A, rel_err_QM, rel_err_CQD  from the extraction; Ic, N_collapse  are yours)

"""
    interp_loglin(xq, x, y)

Linear interpolation of `y(x)` evaluated at `xq`, done in log10(x) space
(appropriate for a log-current axis). Returns NaN outside the range of `x`.
"""
function interp_loglin(xq, x, y)
    p  = sortperm(x)
    lx = log10.(x[p]); ys = y[p]
    lq = log10.(xq)
    out = similar(float.(xq))
    for (k, q) in enumerate(lq)
        if q < lx[1] || q > lx[end]
            out[k] = NaN                       # no extrapolation
        else
            j = searchsortedlast(lx, q)
            j = min(j, length(lx) - 1)
            t = (q - lx[j]) / (lx[j+1] - lx[j])
            out[k] = ys[j] + t * (ys[j+1] - ys[j])
        end
    end
    return out
end

# --- put both errors on the Ic grid (same grid as N_collapse) ---
errQM_on_Ic  = interp_loglin(Ic, I_A, rel_err_QM)
errCQD_on_Ic = interp_loglin(Ic, I_A, rel_err_CQD)

# --- keep only currents inside the digitized range (drop the NaNs) ---
keep = .!isnan.(errQM_on_Ic)          # QM and CQD share the same I_A range
dτ   = N_collapses[keep]
eQM  = errQM_on_Ic[keep]
eCQD = errCQD_on_Ic[keep]

# order by N_collapse so the connecting line is monotone along x
o    = sortperm(dτ)
dτ, eQM, eCQD = dτ[o], eQM[o], eCQD[o]

# ---------------------------------------------------------------- plot
pow_lo, pow_hi = floor(Int, log10(minimum(dτ))), ceil(Int, log10(maximum(dτ)))
xticks_pow = pow_lo:pow_hi
xticks = (10.0 .^ xticks_pow, [L"10^{%$p}" for p in xticks_pow])

ystep = 0.05
ylo = floor(minimum(vcat(eQM, eCQD)) / ystep) * ystep
yhi = ceil(maximum(vcat(eQM, eCQD)) / ystep) * ystep
yticks = round.(ylo:ystep:yhi, digits = 2)

fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1],
    xlabel = "Number of collapse cycles",
    ylabel = "Relative error",
    xlabelsize = 20, ylabelsize = 20,
    xticklabelsize = 16, yticklabelsize = 16,
    xscale = log10,
    xticks = xticks,
    xminorticksvisible = true,
    xminorticks = IntervalsBetween(9),
    yticks = yticks,
)
scatterlines!(ax, dτ, eQM,  color = :blue,  markersize = 8,
    label = L"(\mathcal{z}_{\mathrm{QM}}-\mathcal{z}_{\mathrm{exp}})/\mathcal{E}_{\mathrm{exp}}")
scatterlines!(ax, dτ, eCQD, color = :red, markersize = 8,
    label = L"(\mathcal{z}_{CQD}-\mathcal{z}_{exp})/\mathcal{E}_{exp}")
hlines!(ax, 0, color = (:black, 0.4), linestyle = :dash)
axislegend(ax, position = :rt, labelsize = 16)
fig
save(joinpath(OUTDIR, "relerr_vs_collapsecycles.png"), fig)




##################################################################################################
##################################################################################################
## Main plot

data_2025 = load(joinpath(BASE_PATH,"EXPDATA_ANALYSIS","smoothing_binning_xkl","data_averaged_2.jld2"),"data")

data_2025[:]


load(joinpath(BASE_PATH,"EXPDATA_ANALYSIS","smoothing_binning_2025","data_averaged_2.jld2"),"data")

