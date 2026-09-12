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
const OUTDIR    = joinpath(@__DIR__, "data_studies", "FINALIMAGES_" * RUN_STAMP)
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
##      →  SG_img_f1, SG_img_f2  (image + profile on the left + colorbar)
##      →  SG_img_profile_f1, SG_img_profile_f2  (image + profile on top)
##################################################################################################
# Processing binning of the stored images. The camera frames were binned 4× along x during
# processing; the plots below use unbinned camera-pixel coordinates so both panel styles show the
# image as recorded, and ticks / aspect ratio are honest rather than relabelled.
const IMG_BIN_X = 4
const IMG_BIN_Z = 1

"""
    unbinned_pixel_centers(n, bin) -> Vector{Float64}
 
Camera-pixel coordinate of the centre of each of `n` binned pixels, where every binned pixel
covers `bin` camera pixels. Binned pixel `i` spans camera pixels `bin*(i-1)+1 … bin*i`.
"""
unbinned_pixel_centers(n, bin) = bin .* ((1:n) .- 0.5) .+ 0.5

"""
    load_experiment(filepath) -> data
 
Load the processed JLD2 experiment file and print the coil-current / field table.
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
        show_row_number_column = true, row_number_column_label = "No.", row_number_column_alignment = :c,
    )
    return data
end

"""
    mean_image(stack, idx) -> Matrix
 
Mean over the repetition axis (dim 3) of the (x × z × rep × current) image stack at current
index `idx`. NaN-safe: pixels that are NaN in some repetitions are averaged over the others.
"""
mean_image(stack, idx) = dropdims(nanmean(@view(stack[:, :, :, idx]); dims = 3); dims = 3)



"""
    plot_heatmap_with_profile(img; kwargs...) -> Figure
 
Heatmap of an (x × z) image in unbinned camera-pixel coordinates, with the x-averaged intensity
profile along z drawn on the LEFT (intensity axis reversed so it grows toward the image) and a
colorbar on the right. The colour map spans the raw data range; `yreversed = true` puts z = 0 at
the top, matching the camera frame orientation.
"""
function plot_heatmap_with_profile(img;
        colormap      = :viridis,
        colorrange    = extrema(filter(isfinite, img)),
        figsize       = (600, 550),            # REVIEW: was `size`, which shadows Base.size inside the function
        cb_label      = L"Mean intensity ($\mathrm{a.u.}$)",
        profile_label = L"Intensity ($\mathrm{a.u.}$)",
        profile_width = 0.25,                  # fraction of the layout width given to the profile
        yreversed     = true,
)
    x = unbinned_pixel_centers(size(img, 1), IMG_BIN_X)
    z = unbinned_pixel_centers(size(img, 2), IMG_BIN_Z)
    xlims, zlims = extrema(x), extrema(z)
    profile_z = vec(nanmean(img; dims = 1))    # mean over x → profile along z (splitting direction)
 
    fig    = Figure(; size = figsize, backgroundcolor = :white)
    layout = GridLayout(fig[1, 1])
 
    ax_profile = Axis(layout[1, 1];
        xlabel = profile_label, ylabel = L"$z$ (pixels)",
        xreversed = true, yreversed,
        limits = (nothing, zlims),
        xautolimitmargin = (0, 0), yautolimitmargin = (0, 0),
        yticksmirrored = true,
        xtickformat = latex_int, ytickformat = latex_int,
    )
    ax_img = Axis(layout[1, 2];
        xlabel = L"$x$ (pixels)",
        aspect = DataAspect(),                 # true image shape in camera pixels
        yreversed,
        limits = (xlims, zlims),
        xautolimitmargin = (0, 0), yautolimitmargin = (0, 0),
        yticksmirrored = true,
        xtickformat = latex_int,
    )
 
    hm = heatmap!(ax_img, x, z, img; colormap, colorrange)
    lines!(ax_profile, profile_z, z; color = :darkorange, linewidth = 2)
    linkyaxes!(ax_profile, ax_img)
    hideydecorations!(ax_img; ticks = false, grid = false, minorgrid = false, minorticks = false)
 
    Colorbar(layout[1, 3], hm; label = cb_label, vertical = true, flipaxis = true)
 
    colsize!(layout, 1, Relative(profile_width))
    colgap!(layout, 1, 8)
    colgap!(layout, 2, 2)
    Makie.trim!(layout)
    return fig
end

"""
    plot_image_with_z_profile(img; kwargs...) -> Figure
 
Heatmap of an (x × z) image in unbinned camera-pixel coordinates, drawn with z horizontal, and the
x-averaged intensity profile along z above it. Values below zero are clipped and the image is
normalised to its maximum for the colour map; the profile is drawn from the raw data.
"""
function plot_image_with_z_profile(img;
        colormap      = :viridis,
        figsize       = (600, 400),
        profile_label = L"Intensity ($\mathrm{a.u.}$)",
        profile_height = 0.35,                 # fraction of the layout height given to the profile
        z_tick_step   = 400,
        yreversed     = false,                 # REVIEW (F): choose one orientation for both panel styles
)
    x = unbinned_pixel_centers(size(img, 1), IMG_BIN_X)
    z = unbinned_pixel_centers(size(img, 2), IMG_BIN_Z)
    xlims, zlims = extrema(x), extrema(z)
 
    img_norm  = max.(img, 0) ./ maximum(filter(isfinite, max.(img, 0)))
    profile_z = vec(nanmean(img; dims = 1))    # mean over x → profile along z (splitting direction)
 
    # Upper profile limit rounded up to the leading decade: 750 → 800, 32 → 40
    p_max   = max(maximum(filter(isfinite, profile_z)), 1.0)
    decade  = exp10(floor(log10(p_max)))
    p_upper = ceil(p_max / decade) * decade
 
    fig    = Figure(; size = figsize, backgroundcolor = :white)
    layout = GridLayout(fig[1, 1])
    zticks = range(0, zlims[2]; step = z_tick_step)
 
    ax_top = Axis(layout[1, 1];
        ylabel = profile_label,
        limits = (zlims, (0, p_upper)),
        xautolimitmargin = (0, 0), yautolimitmargin = (0, 0),
        xtickalign = 0.5, xticks = zticks, ytickformat = latex_int,
    )
    ax_img = Axis(layout[2, 1];
        xlabel = L"$z$ (pixels)", ylabel = L"$x$ (pixels)",
        aspect = DataAspect(),                 # true image shape in camera pixels
        yreversed,
        limits = (zlims, xlims),
        xautolimitmargin = (0, 0), yautolimitmargin = (0, 0),
        xtickalign = 0.5, xticks = zticks,
        xtickformat = latex_int, ytickformat = latex_int,
    )
 
    hm = heatmap!(ax_img, z, x, permutedims(img_norm); colormap, colorrange = (0, 1))
    lines!(ax_top, z, profile_z; color = :red, linewidth = 2)
    linkxaxes!(ax_top, ax_img)
    hidexdecorations!(ax_top; ticks = false, grid = false, minorgrid = false, minorticks = false)
 
    rowsize!(layout, 1, Relative(profile_height))
    rowgap!(layout, 1, 12)
    Makie.trim!(layout)
    return fig
end


exp_data = load_experiment(joinpath(BASE_PATH, "EXPERIMENTS", "20260220", "data_processed.jld2"))
 
let
    nI_idx = 19
    I_sel  = exp_data[:Currents][nI_idx]
    @info "Experimental pattern" nI_idx I0_A = I_sel Bz_mT = 1e3 * exp_data[:BzTesla][nI_idx]
 
    F1_mean = mean_image(exp_data[:F1ProcessedImages], nI_idx)
    F2_mean = mean_image(exp_data[:F2ProcessedImages], nI_idx)
    @info "Image ranges" F1 = extrema(filter(isfinite, F1_mean)) F2 = extrema(filter(isfinite, F2_mean))
 
    for (img, tag) in ((F1_mean, "f1"), (F2_mean, "f2"))
        fig = plot_heatmap_with_profile(img)
        display(fig)
        savefig(fig, "SG_img_$tag")
 
        fig = plot_image_with_z_profile(img)
        display(fig)
        savefig(fig, "SG_img_profile_$tag")
    end
end
 

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
);

F2_mean = dropdims(
    mean(@view(exp_data[:F2ProcessedImages][:, :, :, nI_idx]), dims=3),
    dims=3,
);

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
        y_scale         = 4, # binning
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
# Coil currents
Icoils = [0.00,
            0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
            0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,
            0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
];
nI = length(Icoils);

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
ki_fit = ki_list[11]*1e-6
@info "Induction term" ki=ki_selected

cqd_ki_path = joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T205_7M","up","cqd_7000000_ki011_up_screen.jld2");
cqd_ki = load(cqd_ki_path,"screen")[:data]

function standard_error(x)
    return std(x; corrected=true) ./ sqrt.(length(x))
end

Ic = Icoils[2:end]

number_precessions = round(1/(TWOπ*ki_fit))
collapse_time = inv.(ki_fit * abs(γₑ) * TheoreticalSimulation.BvsI.(Ic))

# ---------------------------------------------------------------- collapse time
fig = Figure()
ax = Axis(fig[1, 1],
    xlabel = "Current (A)",
    ylabel = "Collapse time (μs)",
    xscale = log10,
    yscale = log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1, 10, 100], [L"10^{0}", L"10^{1}", L"10^{2}"]),
)
lines!(ax, Ic, 1e6 .* collapse_time,
    color = :blue, linewidth = 2, label = L"Collapse time $\tau_{c}$")
xlims!(ax, 1e-3, 1)
ylims!(ax, 1, 3500)
axislegend(ax)
display(fig)
save(joinpath(OUTDIR, "collapse_time.png"), fig)

# ---------------------------------------------------------------- time of flight
travel_times = [mean(inv.(cqd_ki[ic][:, 5] / y_SG)) for ic = 2:47]
tof = 1e6 .* travel_times
err = 1e6 .* [standard_error(inv.(cqd_ki[ic][:, 5] / y_SG)) for ic = 2:47]

fig = Figure()
ax = Axis(fig[1, 1],
    xlabel = "Current (A)",
    ylabel = "Time of flight (μs)",
    xscale = log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)
band!(ax, Ic, tof .- err, tof .+ err, color = (:dodgerblue3, 0.1))
lines!(ax, Ic, tof,
    color = :dodgerblue3, linewidth = 2, label = L"Time of flight $\Delta t_{\mathrm{SG}}$")
scatter!(ax, Ic, tof,
    color = :white, strokecolor = :dodgerblue3, strokewidth = 1, markersize = 6)
xlims!(ax, 1e-3, 1)
axislegend(ax, position = :lb)
fig
save(joinpath(OUTDIR, "time_flight.png"), fig)

@info @sprintf("The mean time of flight is (%d ± %d) μs ",
    mean(1e6*travel_times), std(1e6*travel_times))

# ---------------------------------------------------------------- combined (display only)
err2 = 1e6 .* [standard_error(inv.(cqd_ki[ic][:, 5] / 0.07)) for ic = 2:47]

fig = Figure()
ax = Axis(fig[1, 1],
    xlabel = "Current (A)",
    ylabel = "Time (μs)",
    xscale = log10,
    yscale = log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1, 10, 100], [L"10^{0}", L"10^{1}", L"10^{2}"]),
)
band!(ax, Ic, tof .- err2, tof .+ err2, color = (:dodgerblue3, 0.1))
lines!(ax, Ic, tof,
    color = :dodgerblue3, linewidth = 2, label = L"Time of flight $\Delta t_{\mathrm{SG}}$")
lines!(ax, Ic, 1e6 .* collapse_time,
    color = :darkgreen, linewidth = 2, label = L"Collapse time $\tau_{c}$")
xlims!(ax, 1e-3, 1)
ylims!(ax, 1, 4000)
axislegend(ax, position = :lb)
fig

# ---------------------------------------------------------------- collapse cycles
fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1],
    xlabel = "Current (A)",
    ylabel = "Number of collapse times",
    xscale = log10,
    yscale = log10,
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([0.1, 1, 10, 100], [L"10^{-1}", L"1", L"10", L"100"]),

    # --- axis label font sizes ---
    xlabelsize = 20,
    ylabelsize = 20,

    # --- tick label font sizes ---
    xticklabelsize = 16,
    yticklabelsize = 16,

    # x/y gridlines as before …
    xgridvisible = true, xminorticksvisible = true,
    xminorgridvisible = true, xminorticks = IntervalsBetween(9),
    ygridvisible = true, yminorticksvisible = true,
    yminorgridvisible = true, yminorticks = IntervalsBetween(9),
)
hspan!(ax, 0.1, 1, color = (:black, 0.2))                       # behind the curve
lines!(ax, Ic, travel_times ./ collapse_time,
    color = :red, linewidth = 2, label = L"$\Delta t_{\mathrm{SG}} / \tau_{c}$")
xlims!(ax, 1e-2, 1)
ylims!(ax, 0.7, 40)
axislegend(ax, position = :lt, labelsize = 16)
fig
save(joinpath(OUTDIR, "collapse_cycles.png"), fig)





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

