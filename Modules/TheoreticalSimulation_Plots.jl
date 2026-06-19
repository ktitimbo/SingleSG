# ==============================================================================
# Diagnostic plotting routines for the SG beamline simulation
#
# Three public functions:
#   plot_μeff            — μF/μB vs. coil current for all hyperfine levels
#   plot_SG_geometry     — 2D cross-section of the SG slit/magnet geometry
#   plot_velocity_stats  — 7-panel dashboard of beam velocity/position statistics
#
# All functions:
#   - Display the figure interactively via `display`.
#   - Save to `joinpath(OUTDIR, "filename.FIG_EXT")` only when `SAVE_FIG = true`.
#   - Return the assembled `Plots.Plot` object for further modification in the REPL.
# ==============================================================================


"""
    plot_μeff(p::AtomParams, filename::AbstractString) -> Plots.Plot

Plot μF/μB versus coil current for all hyperfine (F, mF) levels of the species
described by `p`, and annotate the magnetic crossing point.

# Arguments
- `p        :: AtomParams`        — atomic parameters (Ispin, Ahfs, γn, ...).
- `filename :: AbstractString`    — base name for the saved figure (no extension).

# Behavior
- Curves are colored with the `:phase` palette, one color per (F, mF) state.
- Upper manifold (F = I + 1/2): solid lines, except the lowest mF which is dashed.
- Lower manifold (F = I − 1/2): all dashed.
- Current axis is logarithmic; samples are log-spaced for uniform visual density.
- The magnetic crossing current I₀ (where the normalized field x = 1) is found
  with `Roots.find_zero` and annotated with I₀, ∂zBz, and Bz at that point.

# Returns
The assembled `Plots.Plot` object.
"""
function plot_μeff(p::AtomParams, filename::AbstractString)
    F_up   = p.Ispin + 0.5
    F_down = p.Ispin - 0.5
    # Ranges are directly indexable in Julia — no collect needed
    mf_up   = F_up:-1.0:-F_up
    mf_down = -F_down:1.0:F_down

    # Total number of (F, mF) states: (2F_up+1) + (2F_down+1) = 4I + 2
    dimF = Int(4*p.Ispin + 2)
    colorsF = palette(:phase, dimF)

    # Log-spaced current samples: visually uniform on the log x-axis and
    # denser at low currents where μF curves are most nonlinear.
    current_range = exp10.(range(log10(0.00009), log10(1.0), length=600))

    # Base figure — log axis set here, not retroactively inside a later call
    fig = plot(
        xlabel  = L"Current ($\mathrm{A}$)",
        ylabel  = L"$\mu_{F}/\mu_{B}$",
        legend  = :right,
        background_color_legend = RGBA(0.85, 0.85, 0.85, 0.1),
        size    = (800, 600),
    )

    # Define (F, mF, linestyle) triples for every state.
    # Upper manifold: solid for all but the lowest mF (which gets dashed to
    # visually distinguish it as the only upper state that crosses mF=0 at B=0).
    # Lower manifold: all dashed (conventional in Breit–Rabi diagrams).
    lines_to_plot = vcat(
        [(F_up, mf, :solid) for mf in mf_up[1:end-1]],
        [(F_up, mf_up[end], :dash)],
        [(F_down, mf, :dash) for mf in mf_down],
    )

    for ((f, mf, lstyle), color) in zip(lines_to_plot, colorsF)
        μ_vals = μF_effective.(current_range, f, mf, Ref(p)) ./ μB
        label  = L"$F=%$(f)$, $m_{F}=%$(mf)$"
        plot!(fig, current_range, μ_vals; label=label, line=(color, lstyle, 2))
    end

    # ── Magnetic crossing point ────────────────────────────────────────────
    # The crossing occurs when the normalized field x = 1, i.e.
    #   (γₑ - γₙ)·ħ / ΔE · B = 1  →  B = ΔE / ((γₑ - γₙ)·ħ)
    # where ΔE = 2π·ħ·Ahfs·(I + 1/2).
    # The ħ cancels exactly: B_cross = 2π·Ahfs·(I+1/2) / (γₑ - γₙ)
    # Rewritten in terms of (γₙ - γₑ) to keep the sign explicit:
    #   B_cross = -2π·Ahfs·(I+1/2) / (γₙ - γₑ)
    # which is positive because (γₙ - γₑ) < 0 for K39 (γₑ ≪ 0, γₙ > 0).
    B_cross_target = 2π * p.Ahfs * (p.Ispin + 0.5) / (p.γn - γₑ)
    f_cross(x)     = BvsI(x) - B_cross_target
    I₀             = find_zero(f_cross, (0.001, 0.050))

    label_text = L"$I_{0} = %$(round(1000*I₀, digits=3))\,\mathrm{mA}$
 $\partial_{z}B_{z} = %$(round(GvsI(I₀), digits=2))\,\mathrm{T/m}$
 $B_{z} = %$(round(1e3 * BvsI(I₀), digits=3))\,\mathrm{mT}$"
    vline!(fig, [I₀]; line=(:black, :dot, 2), label=label_text)
    plot!(fig, xaxis   = :log10,)

    display(fig)
    savefig(fig, joinpath(OUTDIR, "$(filename).$(FIG_EXT)"))
    return fig
end


"""
    plot_SG_geometry(filename::AbstractString) -> Plots.Plot

Render and save a 2D cross-section of the Stern–Gerlach slit geometry.

# What is drawn
- **Top magnet edge**: shaded region above `z_magnet_edge(x)`.
- **Bottom trench**:   shaded region below `z_magnet_trench(x)`.
- **Slit aperture**:   centered rectangle of width `DEFAULT_x_slit` × `DEFAULT_z_slit`.

# Axes & units
- Both axes in millimetres. `x ∈ [−8, 8] mm`, `z ∈ [−3, 7] mm`. Aspect ratio 1:1.

# Assumptions / dependencies
- `z_magnet_edge(x)` and `z_magnet_trench(x)` return z in metres.
- Globals `DEFAULT_x_slit`, `DEFAULT_z_slit`, `OUTDIR`, `FIG_EXT`, `SAVE_FIG`.

# Returns
The assembled `Plots.Plot` object.
"""
function plot_SG_geometry(filename::AbstractString)
    # x positions in metres — range broadcasts directly, no collect needed
    x_line = range(-10e-3, 10e-3; length=10_001)
    x_mm   = 1e3 .* x_line     # convert to mm for plotting

    fig = plot(
        xlabel       = L"$x \ (\mathrm{mm})$",
        xlim         = (-8, 8),  xticks = -8:2:8,
        ylabel       = L"$z \ (\mathrm{mm})$",
        ylim         = (-3, 7),  yticks = -3:1:7,
        aspect_ratio = :equal,
        legend       = :bottomright,
        title        = "Stern–Gerlach Slit Geometry",
    )

    # Filled polygon: top magnet edge → fill to +∞ (capped at +10 mm)
    y_edge = 1e3 .* z_magnet_edge.(x_line)
    y_top  = fill(10.0, length(x_mm))
    plot!(fig, [x_mm; reverse(x_mm)], [y_edge; reverse(y_top)];
        seriestype = :shape, label = "Rounded edge",
        color = :grey36, line = (:solid, :grey36), fillalpha = 0.75,
    )

    # Filled polygon: bottom trench → fill to −∞ (capped at −10 mm)
    y_trench = 1e3 .* z_magnet_trench.(x_line)
    y_bottom = fill(-10.0, length(x_mm))
    plot!(fig, [x_mm; reverse(x_mm)], [y_bottom; reverse(y_trench)];
        seriestype = :shape, label = "Trench",
        color = :grey60, line = (:solid, :grey60), fillalpha = 0.75,
    )

    # Slit rectangle (closed polygon, 5 vertices)
    hw_x = 1e3 * DEFAULT_x_slit / 2     # half-width in mm
    hw_z = 1e3 * DEFAULT_z_slit / 2     # half-height in mm
    plot!(fig,
        [-hw_x, -hw_x,  hw_x, hw_x, -hw_x],
        [-hw_z,  hw_z,  hw_z, -hw_z, -hw_z];
        seriestype = :shape, label = "Slit",
        line = (:solid, :red, 1), color = :red, fillalpha = 0.2,
    )

    display(fig)
    savefig(fig, joinpath(OUTDIR, "$(filename).$(FIG_EXT)"))
    return fig
end


"""
    plot_velocity_stats(alive::Matrix{Float64}, title::String, filename::String)
        -> Plots.Plot

Build, display, and save a 7-panel dashboard of beam velocity and position statistics.

# Input
- `alive   :: Matrix{Float64}` — N × ≥6 matrix; columns are x, y, z, vₓ, vᵧ, vz (SI).
- `title   :: String`          — plot title (appended with particle count).
- `filename :: String`         — base name for the saved figure.

# Layout (7 panels)
| Panel | Content                                                          |
|-------|------------------------------------------------------------------|
| 1     | Speed ‖v‖ histogram, with ⟨v⟩ and √⟨v²⟩ marked               |
| 2     | vₓ histogram                                                     |
| 3     | Polar angle θ = acos(vz/‖v‖) histogram                          |
| 4     | vy histogram                                                     |
| 5     | Azimuth φ = atan(vy, vx) histogram                              |
| 6     | vz histogram                                                     |
| 7     | 2D position histogram (x in mm, z in μm) — full width           |

# Returns
The assembled `Plots.Plot` object.
"""
function plot_velocity_stats(alive::Matrix{Float64}, title::String, filename::String)
    @assert size(alive, 2) ≥ 6 "Expected at least 6 columns: x, y, z, vx, vy, vz"
    N = size(alive, 1)

    # ── Views into alive — zero allocation, no copy ────────────────────────
    # Velocity columns
    vxs = @view alive[:, 4]
    vys = @view alive[:, 5]
    vzs = @view alive[:, 6]

    # ── Speed and angles ───────────────────────────────────────────────────
    # @. fuses the three squares, the sum, and the sqrt into a single loop
    # (one allocation for `velocities`, no intermediate arrays)
    velocities = @. sqrt(vxs^2 + vys^2 + vzs^2)
    theta_vals = acos.(vzs ./ velocities)   # polar angle θ ∈ [0, π]
    phi_vals   = atan.(vys, vxs)            # azimuth φ ∈ (−π, π]

    mean_v     = mean(velocities)
    rms_v      = sqrt(mean(velocities .^ 2))
    mean_theta = mean(theta_vals)
    mean_phi   = mean(phi_vals)

    # ── Panel 1: speed ─────────────────────────────────────────────────────
    figa = histogram(velocities;
        bins      = FreedmanDiaconisBins(velocities),
        normalize = :pdf, alpha = 0.70,
        label     = L"$v_0$",
        xlabel    = L"$v_{0}\ (\mathrm{m/s})$",
    )
    vline!([mean_v];
        label = L"$\langle v_{0}\rangle = %$(round(mean_v, digits=1))\ \mathrm{m/s}$",
        line  = (:black, :solid, 2),
    )
    vline!([rms_v];
        label = L"$\sqrt{\langle v_{0}^{2}\rangle} = %$(round(rms_v, digits=1))\ \mathrm{m/s}$",
        line  = (:red, :dash, 3),
    )

    # ── Panel 3: polar angle ───────────────────────────────────────────────
    figb = histogram(theta_vals;
        bins      = FreedmanDiaconisBins(theta_vals),
        normalize = :pdf, alpha = 0.70,
        label     = L"$\theta_v$",
        xlabel    = L"$\theta_{v}$",
    )
    vline!([mean_theta];
        label = L"$\langle\theta_{v}\rangle = %$(round(mean_theta/π, digits=3))\pi$",
        line  = (:black, :solid, 2),
    )

    # ── Panel 5: azimuth ───────────────────────────────────────────────────
    figc = histogram(phi_vals;
        bins      = FreedmanDiaconisBins(phi_vals),
        normalize = :pdf, alpha = 0.70,
        label     = L"$\phi_v$",
        xlabel    = L"$\phi_{v}$",
    )
    vline!([mean_phi];
        label = L"$\langle\phi_{v}\rangle = %$(round(mean_phi/π, digits=3))\pi$",
        line  = (:black, :solid, 2),
    )

    # ── Panel 7: 2D position histogram ────────────────────────────────────
    # Unit conversions require new arrays (can't view a scaled column)
    xs_mm = 1e3 .* @view alive[:, 1]
    zs_μm = 1e6 .* @view alive[:, 3]
    figd = histogram2d(xs_mm, zs_μm;
        bins               = (FreedmanDiaconisBins(xs_mm), FreedmanDiaconisBins(zs_μm)),
        show_empty_bins    = true,
        color              = :plasma,
        xlabel             = L"$x\ (\mathrm{mm})$",
        ylabel             = L"$z\ (\mathrm{\mu m})$",
        xticks             = -1.0:0.25:1.0,
        yticks             = -50:10:50,
        colorbar_position  = :bottom,
    )

    # ── Panels 2, 4, 6: velocity components ───────────────────────────────
    fige = histogram(vxs;
        bins=FreedmanDiaconisBins(vxs), normalize=:pdf, alpha=0.65,
        label=L"$v_{0,x}$", color=:orange, xlabel=L"$v_{0,x}\ (\mathrm{m/s})$",
    )
    figf = histogram(vys;
        bins=FreedmanDiaconisBins(vys), normalize=:pdf, alpha=0.65,
        label=L"$v_{0,y}$", color=:blue,   xlabel=L"$v_{0,y}\ (\mathrm{m/s})$",
    )
    figg = histogram(vzs;
        bins=FreedmanDiaconisBins(vzs), normalize=:pdf, alpha=0.65,
        label=L"$v_{0,z}$", color=:red,    xlabel=L"$v_{0,z}\ (\mathrm{m/s})$",
    )

    # ── Assemble dashboard ─────────────────────────────────────────────────
    # Layout: three rows of paired panels (speed|vx, θ|vy, φ|vz),
    # then the 2D histogram spanning the full width.
    fig = plot(
        figa, fige,
        figb, figf,
        figc, figg,
        figd,
        layout      = @layout([a1 a2; a3 a4; a5 a6; a7]),
        plot_title  = title * " | N=$N particles",
        size        = (650, 800),
        legendfontsize = 8,
        left_margin    = 3mm,
    )

    display(fig)
    savefig(fig, joinpath(OUTDIR, "$(filename).$(FIG_EXT)"))
    return fig
end


"""
    plot_BreitRabi_energy(p::AtomParams, Brange::Tuple, filename::AbstractString;
                           xaxis_scale::Symbol=:log10) -> Plots.Plot

Plot the Breit–Rabi energy eigenvalues E(F, mF, B) / ΔE versus magnetic field
for all hyperfine levels of the species described by `p`, and annotate the
inflection-point field B× where the normalized field parameter x = 1.

# Arguments
- `p        :: AtomParams`     — atomic parameters (Ispin, Ahfs, γn, ...).
- `Brange   :: Tuple`          — `(Bmin, Bmax)` field range in Tesla over which
                                  the curves are sampled.
- `filename :: AbstractString` — base name for the saved figure (no extension).

# Keywords
- `xaxis_scale :: Symbol` — axis scale and sample spacing for the field axis:
    - `:log10`    (default) — logarithmic axis, log-spaced samples (denser at
      low field, where the Breit–Rabi curves are most nonlinear). Requires
      `Brange[1] > 0`.
    - `:identity` — linear axis, linearly spaced samples. `Brange[1]` may be 0.

# Axes & normalization
- x-axis: magnetic field B in Tesla, over `Brange`, scaled per `xaxis_scale`.
- y-axis: energy normalized by the zero-field hyperfine splitting
  `ΔE = 2π·ħ·Ahfs·(I + 1/2)`, so the plot is dimensionless and
  species-independent in shape (only the B× position shifts).

  At B = 0:   upper manifold sits at  +I/(2I+1),
               lower manifold sits at −(I+1)/(2I+1).
  At B → ∞:  levels fan out linearly (Paschen–Back regime).

# Line style convention (same as `plot_μeff`)
- Upper manifold (F = I + 1/2): solid lines, except the lowest mF which is dashed.
- Lower manifold (F = I − 1/2): all dashed.
- Colors: `:phase` palette, one color per (F, mF) state.

# Crossing annotation
B× is computed directly (no root-finding needed — field is the axis variable):

    B× = 2π·Ahfs·(I + 1/2) / (γₙ − γₑ)

# Returns
The assembled `Plots.Plot` object.

# Throws
- `ArgumentError` if `xaxis_scale` is neither `:log10` nor `:identity`.
- `ArgumentError` if `xaxis_scale == :log10` and `Brange[1] ≤ 0`
  (the log of a non-positive field is undefined).
"""
function plot_BreitRabi_energy(p::AtomParams, Brange::Tuple, filename::AbstractString;
                                xaxis_scale::Symbol=:log10)

    xaxis_scale in (:log10, :identity) ||
        throw(ArgumentError("xaxis_scale must be :log10 or :identity; got $xaxis_scale"))

    F_up   = p.Ispin + 0.5
    F_down = p.Ispin - 0.5
    # Ranges are directly indexable in Julia — no collect needed
    mf_up   = F_up:-1.0:-F_up
    mf_down = -F_down:1.0:F_down

    # Total number of (F, mF) states: (2F_up+1) + (2F_down+1) = 4I + 2
    dimF    = Int(4*p.Ispin + 2)
    colorsF = palette(:phase, dimF)

    Bmin, Bmax = Brange

    # Field samples: log-spaced under :log10 (uniform visual density on a log
    # axis, denser at low fields where the curves are most curved); linearly
    # spaced under :identity (uniform density on a linear axis instead).
    if xaxis_scale === :log10
        Bmin > 0 || throw(ArgumentError(
            "Brange[1] must be > 0 for xaxis_scale=:log10 (log of a non-positive field is undefined); got Bmin=$Bmin"))
        B_range = exp10.(range(log10(Bmin), log10(Bmax), length=600))
    else # :identity
        B_range = range(Bmin, Bmax, length=600)
    end

    # Zero-field hyperfine splitting — used to normalize the energy axis so
    # the plot is dimensionless and directly comparable across species.
    ΔE = 2π * ħ * p.Ahfs * (p.Ispin + 0.5)

    # Same line style convention as plot_μeff:
    # upper manifold solid (except lowest mF dashed), lower manifold all dashed.
    lines_to_plot = vcat(
        [(F_up, mf, :solid) for mf in mf_up[1:end-1]],
        [(F_up, mf_up[end], :dash)],
        [(F_down, mf, :dash) for mf in mf_down],
    )

    # Base figure — xaxis scale and xlim are set together here, not added
    # retroactively after data/annotations. With a log axis in particular,
    # xlim must be explicit: without it GR computes ticks on the default
    # empty-plot range [0,1], whose log is undefined and produces NaN ticks
    # (the same failure mode fixed earlier in plot_μeff).
    fig = plot(
        xlabel  = L"$B\ (\mathrm{T})$",
        ylabel  = L"$E\,/\,\Delta E$",
        xaxis   = xaxis_scale,
        xlim    = (B_range[1], B_range[end]),
        legend  = :right,
        background_color_legend = RGBA(0.85, 0.85, 0.85, 0.1),
        size    = (800, 600),
    )

    for ((f, mf, lstyle), color) in zip(lines_to_plot, colorsF)
        E_vals = BreitRabi_energy.(B_range, f, mf, Ref(p)) ./ ΔE
        label  = L"$F=%$(f)$, $m_{F}=%$(mf)$"
        plot!(fig, B_range, E_vals; label=label, line=(color, lstyle, 2))
    end

    # ── Inflection-point field B× ─────────────────────────────────────────
    # B× is where x = 1, i.e. the magnetic energy equals the hyperfine splitting.
    # Unlike plot_μeff (where we root-find BvsI), here B is the direct axis
    # variable so B× is computed analytically — no root-finding needed.
    B_cross = 2π * p.Ahfs * (p.Ispin + 0.5) / (p.γn - γₑ)

    label_cross = L"$B_{\times} = %$(round(1e3*B_cross, digits=3))\ \mathrm{mT}$"
    vline!(fig, [B_cross]; line=(:black, :dot, 2), label=label_cross)

    display(fig)
    savefig(fig, joinpath(OUTDIR, "$(filename).$(FIG_EXT)"))
    return fig
end