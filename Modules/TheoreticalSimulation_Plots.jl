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
function plot_μeff(p::AtomParams, filename::AbstractString)  
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
    savefig(fig, joinpath(OUTDIR,"$(filename).$(FIG_EXT)"))
    
    return nothing
end


"""
    plot_SG_geometry(filename::AbstractString) -> Nothing

Render and save a 2D cross-section of the Stern–Gerlach slit geometry.

What is drawn
- **Top magnet edge**: shaded region above `z_magnet_edge(x)`.
- **Bottom trench**: shaded region below `z_magnet_trench(x)`.
- **Slit aperture**: centered rectangle of width `default_x_slit` and height `default_z_slit`.

Axes & units
- Plots use millimetres on both axes (internally samples `x` in metres).
- Limits: `x ∈ [-8, 8] mm`, `y ∈ [-3, 7] mm`. Aspect ratio is 1:1.
- Labels use LaTeX (`L"...")`.

Sampling
- `x` is sampled uniformly over `[-10, 10] mm` with 10_001 points for smooth shapes.

Side effects
- Displays the figure and saves it to `joinpath(OUTDIR, "filename.FIG_EXT")`.

Assumptions / dependencies
- Functions `z_magnet_edge(x)` and `z_magnet_trench(x)` are in scope and return `z(x)` in metres.
- Globals `default_x_slit`, `default_z_slit`, `OUTDIR`, and `FIG_EXT` are defined.
- Uses `Plots.jl` (with LaTeXStrings) and an initialized backend.

Example
julia
plot_SG_geometry("sg_geometry")
# writes OUTDIR/sg_geometry.<FIG_EXT>
"""
function plot_SG_geometry(filename::AbstractString)
    # x positions for evaluation (in meters)
    x_line = 1e-3 .* collect(range(-10, 10, length=10_001))

    # Base figure
    fig = plot(
        xlabel = L"$x \ (\mathrm{mm})$",
        xlim = (-8, 8), xticks = -8:2:8,
        ylabel = L"$z \ (\mathrm{mm})$",
        ylim = (-3, 7), yticks = -3:1:7,
        aspect_ratio = :equal,
        legend = :bottomright,
        title = "Stern–Gerlach Slit Geometry"
    )

    # Top magnet edge shape
    x_fill = 1e3 .* x_line
    y_edge = 1e3 .* z_magnet_edge.(x_line)
    y_top  = fill(10.0, length(x_fill))
    plot!(fig, [x_fill; reverse(x_fill)], [y_edge; reverse(y_top)];
        seriestype = :shape, label = "Rounded edge",
        color = :grey36, line = (:solid, :grey36), fillalpha = 0.75
    )

    # Bottom trench shape
    y_trench = 1e3 .* z_magnet_trench.(x_line)
    y_bottom = fill(-10.0, length(x_fill))
    plot!(fig, [x_fill; reverse(x_fill)], [y_bottom; reverse(y_trench)];
        seriestype = :shape, label = "Trench",
        color = :grey60, line = (:solid, :grey60), fillalpha = 0.75
    )

    # Slit rectangle
    plot!(fig,
        1e3 .* 0.5 .* [-default_x_slit, -default_x_slit, default_x_slit,  default_x_slit, -default_x_slit],
        1e3 .* 0.5 .* [-default_z_slit,  default_z_slit, default_z_slit, -default_z_slit, -default_z_slit];
        seriestype = :shape, label = "Slit",
        line = (:solid, :red, 1), color = :red, fillalpha = 0.2
    )
    display(fig)
    savefig(fig, joinpath(OUTDIR,"$(filename).$(FIG_EXT)"))
    
    return nothing
end


"""
    plot_velocity_stats(alive::Matrix{Float64}, title::String, filename::String) -> Plots.Plot

Build, display, and save a dashboard of velocity/position statistics for a set of
particles.

Input
- `alive`: `N × ≥6` matrix with columns
  1: `x` (m), 2: `y` (m), 3: `z` (m), 4: `vₓ` (m/s), 5: `vᵧ` (m/s), 6: `v_z` (m/s).
- `title`: Plot title (shown atop the dashboard).
- `filename`: Basename for saving the figure (written to `joinpath(OUTDIR, "filename.FIG_EXT")`).

What it plots (7 panels)
1. **Speed histogram** of `‖v‖`, with vertical lines at the mean `⟨v₀⟩` and RMS `√⟨v₀²⟩`.
2. **Polar angle** `θ_v = acos(v_z/‖v‖)` histogram (radians).
3. **Azimuth** `φ_v = atan(v_y, vₓ)` histogram (radians).
4. **2D position histogram** of `(x, z)` with axes in mm (x) and μm (z).
5–7. **Component histograms** for `vₓ`, `vᵧ`, `v_z` (PDF-normalized).

Details
- Histogram bin counts are chosen via `FreedmanDiaconisBins`.
- Axes use LaTeX labels; units: mm for x, μm for z, m/s for velocities.
- The figure is displayed and saved; the function returns the assembled `Plots.Plot`.

Assumptions / dependencies
- Globals `OUTDIR` and `FIG_EXT` are defined.
- Functions/packages in scope: `FreedmanDiaconisBins`, `Plots`, `LaTeXStrings`.
- Assumes nonzero speeds for angle calculations (`‖v‖ > 0`).

Example
julia
fig = plot_velocity_stats(alive, "Beam velocity statistics", "vel_stats_run42")
"""
function plot_velocity_stats(alive::Matrix{Float64}, title::String, filename::String)
    @assert size(alive, 2) ≥ 6 "Expected at least 6 columns (x, y, z, vx, vy, vz)."
    No = size(alive,1)

    # --- Velocity magnitude and angles ---
    vxs, vys, vzs = eachcol(alive[:, 4:6])
    velocities = sqrt.(vxs.^2 .+ vys.^2 .+ vzs.^2)
    theta_vals = acos.(vzs ./ velocities) # polar angle
    phi_vals   = atan.(vys, vxs)          # azimuthal angle

    # Means
    mean_v, rms_v = mean(velocities), sqrt(mean(velocities.^2))
    mean_theta, mean_phi = mean(theta_vals), mean(phi_vals)

    # Histogram for velocities
    figa = histogram(velocities;
        bins = FreedmanDiaconisBins(velocities),
        label = L"$v_0$", normalize = :pdf,
        xlabel = L"v_{0} \ (\mathrm{m/s})",
        alpha = 0.70,
    )
    vline!([mean_v], label = L"$\langle v_{0} \rangle = %$(round(mean_v, digits=1))\ \mathrm{m/s}$",
        line = (:black, :solid, 2))
    vline!([rms_v], label = L"$\sqrt{\langle v_{0}^2 \rangle} = %$(round(rms_v, digits=1))\ \mathrm{m/s}$",
        line = (:red, :dash, 3))

    figb = histogram(theta_vals;
        bins = FreedmanDiaconisBins(theta_vals),
        label = L"$\theta_v$", normalize = :pdf,
        alpha = 0.70, xlabel = L"$\theta_{v}$"
    )
    vline!([mean_theta], label = L"$\langle \theta_{v} \rangle = %$(round(mean_theta/π, digits=3))\pi$",
        line = (:black, :solid, 2))

    figc = histogram(phi_vals;
        bins = FreedmanDiaconisBins(phi_vals),
        label = L"$\phi_v$", normalize = :pdf,
        alpha = 0.70, xlabel = L"$\phi_{v}$"
    )
    vline!([mean_phi], label = L"$\langle \phi_{v} \rangle = %$(round(mean_phi/π, digits=3))\pi$",
        line = (:black, :solid, 2))

    # 2D Histogram of position (x, z)
    # --- 2D position histogram ---
    xs, zs = 1e3 .* alive[:, 1], 1e6 .* alive[:, 3]  # mm, μm
    figd = histogram2d(xs, zs;
        bins = (FreedmanDiaconisBins(xs), FreedmanDiaconisBins(zs)),
        show_empty_bins = true, color = :plasma,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{\mu m})$",
        xticks = -1.0:0.25:1.0, yticks = -50:10:50,
        colorbar_position = :bottom,
    )

    # --- Velocity component histograms ---
    fige = histogram(vxs;
        bins = FreedmanDiaconisBins(vxs), normalize = :pdf,
        label = L"$v_{0,x}$", alpha = 0.65, color = :orange,
        xlabel = L"$v_{0,x} \ (\mathrm{m/s})$"
    )
    figf = histogram(vys;
        bins = FreedmanDiaconisBins(vys), normalize = :pdf,
        label = L"$v_{0,y}$", alpha = 0.65, color = :blue,
        xlabel = L"$v_{0,y} \ (\mathrm{m/s})$"
    )
    figg = histogram(vzs;
        bins = FreedmanDiaconisBins(vzs), normalize = :pdf,
        label = L"$v_{0,z}$", alpha = 0.65, color = :red,
        xlabel = L"$v_{0,z} \ (\mathrm{m/s})$"
    )

    # Combine plots
    fig = plot(
        figa, fige, figb, figf, figc, figg, figd,
        layout = @layout([a1 a2; a3 a4; a5 a6; a7]),
        plot_title = title*" | N=$(No) particles",
        size = (650, 800),
        legendfontsize = 8,
        left_margin = 3mm,
    );
    display(fig)
    savefig(fig, joinpath(OUTDIR,"$(filename).$(FIG_EXT)"))

    return nothing
end