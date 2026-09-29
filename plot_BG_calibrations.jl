# Compare the two SG magnetic-field calibrations (:manual vs :calibration):
#   top    : B(I) and G(I), linear axes
#   bottom : B(I) and G(I), log-log axes
using Plots; gr()
using Plots.PlotMeasures
using LaTeXStrings

include("./Modules/TheoreticalSimulation.jl");
using .TheoreticalSimulation;

# ── Current grids (A) ──
I_lin = range(0.001, 1.01; length = 1000)
I_log = exp10.(range(log10(0.001), log10(1.01); length = 1000))

# ── Evaluate B(I) and G(I) for each calibration ──
modes  = (:manual, :calibration)
labels = Dict(:manual => "Manual", :calibration => "Calibration")
styles = Dict(:manual => (:solid, :dodgerblue), :calibration => (:dash, :orangered))

curves = Dict(mode => TheoreticalSimulation.with_magnetic_field(mode) do
                  (B_lin = TheoreticalSimulation.BvsI.(I_lin),
                   G_lin = TheoreticalSimulation.GvsI.(I_lin),
                   B_log = TheoreticalSimulation.BvsI.(I_log),
                   G_log = TheoreticalSimulation.GvsI.(I_log))
              end
              for mode in modes)

# ── Panels ──
p_B_lin = plot(xlabel = L"I\ \mathrm{(A)}", ylabel = L"B\ \mathrm{(T)}", title = "Magnetic field")
p_G_lin = plot(xlabel = L"I\ \mathrm{(A)}", ylabel = L"\partial_z B\ \mathrm{(T/m)}", title = "Field gradient")
p_B_log = plot(xlabel = L"I\ \mathrm{(A)}", ylabel = L"B\ \mathrm{(T)}", xscale = :log10, yscale = :log10)
p_G_log = plot(xlabel = L"I\ \mathrm{(A)}", ylabel = L"\partial_z B\ \mathrm{(T/m)}", xscale = :log10, yscale = :log10)

for mode in modes
    ls, c = styles[mode]
    kw = (label = labels[mode], linestyle = ls, color = c, linewidth = 2)
    plot!(p_B_lin, I_lin, curves[mode].B_lin; kw...)
    plot!(p_G_lin, I_lin, curves[mode].G_lin; kw...)
    plot!(p_B_log, I_log, curves[mode].B_log; kw...)
    plot!(p_G_log, I_log, curves[mode].G_log; kw...)
end

fig = plot(p_B_lin, p_G_lin, p_B_log, p_G_log;
           layout = (2, 2), size = (1100, 850),
           legend = :topleft, left_margin = 5mm, bottom_margin = 5mm)
display(fig)

savefig(fig, joinpath(TheoreticalSimulation.OUTDIR, "BG_calibration_comparison.png"))
@info "Saved figure" joinpath(TheoreticalSimulation.OUTDIR, "BG_calibration_comparison.png")
