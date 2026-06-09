# ==============================================================================
# Stern–Gerlach Experiment — Experimental Profile Fitting
# Author  : Kelvin Titimbo — Caltech
# Date    : June 2026
# Purpose : Fit the experimental atomic beam profile at zero (or minimum) coil
#           current, using a theoretical PDF plus a polynomial background.
# ==============================================================================

# ── Plotting ──────────────────────────────────────────────────────────────────
# ENV["GKS_WSTYPE"] = "101"   # uncomment for headless/offscreen rendering
using Plots; gr()
Plots.default(
    show       = true,
    dpi        = 800,
    fontfamily = "Computer Modern",
    grid       = true,
    minorgrid  = true,
    framestyle = :box,
    widen      = true,
)
using Plots.PlotMeasures
const FIG_EXT  = "png"   # "pdf" | "svg" | "png"
const SAVE_FIG = true
# ── Aesthetics & formatting ───────────────────────────────────────────────────
using Colors, ColorSchemes
using LaTeXStrings, Printf, PrettyTables
# ── Timing & logging ──────────────────────────────────────────────────────────
using Dates
const T_START   = Dates.now()
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss")
# ── Numerics ──────────────────────────────────────────────────────────────────
using LinearAlgebra, DataStructures
using LsqFit, DSP, FFTW
using BSplineKit, Polynomials
using StatsBase, Statistics, Distributions, StaticArrays
using Alert
# ── Data I/O ──────────────────────────────────────────────────────────────────
using OrderedCollections, JLD2
# ── Threading ─────────────────────────────────────────────────────────────────
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads"  count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# ── Working directory & output ────────────────────────────────────────────────
cd(@__DIR__)
const HOSTNAME    = gethostname()
const PROGRAM_FILE = @__FILE__
const BASE_PATH = raw"F:\SternGerlachExperiments"
const OUTDIR    = joinpath(@__DIR__, "data_studies", "CONV_" * RUN_STAMP)
isdir(OUTDIR) || mkpath(OUTDIR)
@info "Output directory" path = OUTDIR
@info "Hostname"         host = HOSTNAME
@info "Program File"     program_file = PROGRAM_FILE 
@info "Run stamp"        stamp = RUN_STAMP
# ── Custom modules ────────────────────────────────────────────────────────────
include("./Modules/atoms.jl")
include("./Modules/samplings.jl")
include("./Modules/DataReading.jl")
include("./Modules/ProfileFitTools.jl")
include("./Modules/JLD2_MyTools.jl")
include("./Modules/TheoreticalSimulation.jl")
# Forward output settings into TheoreticalSimulation
TheoreticalSimulation.SAVE_FIG = SAVE_FIG
TheoreticalSimulation.FIG_EXT  = FIG_EXT
TheoreticalSimulation.OUTDIR   = OUTDIR

@info "Run stamp initialized" RUN_STAMP = RUN_STAMP

# ==============================================================================
# PHYSICAL CONSTANTS  (NIST 2022 CODATA)
# ==============================================================================
const kb  = 1.380649e-23       # Boltzmann constant            [J K⁻¹]
const ħ   = 6.62607015e-34/2π  # Reduced Planck constant       [J s]
const μ₀  = 1.25663706127e-6   # Vacuum permeability           [T m A⁻¹]
const μB  = 9.2740100657e-24   # Bohr magneton                 [J T⁻¹]
const γₑ  = -1.76085962784e11  # Electron gyromagnetic ratio   [s⁻¹ T⁻¹]  RSU = 3.0e-10
const μₑ  = 9.2847646917e-24   # Electron magnetic moment      [J T⁻¹]    RSU = 3.0e-10
const Sspin = 1/2              # Electron spin quantum number
const gₑ  = -2.00231930436092  # Electron g-factor
const TWOπ  = 2π
const INV_E = exp(-1)

# ==============================================================================
# ATOMIC SPECIES
# ==============================================================================
const ATOM      = "39K"
const K39_params = TheoreticalSimulation.AtomParams(ATOM)   # [R, μn, γn, Ispin, Ahfs, M]

# ==============================================================================
# CAMERA & SIMULATION GEOMETRY
# ==============================================================================
# Camera intrinsics
const CAM_PIXELSIZE         = 6.5e-6        # Physical pixel size        [m]
const NX_PIXELS, NZ_PIXELS  = (2160, 2560)  # Sensor dimensions (Nx, Nz) [px]
# Simulation binning
const SIM_BIN_X, SIM_BIN_Z = (1, 1)
const SIM_PIXELSIZE_X       = SIM_BIN_X * CAM_PIXELSIZE   # Effective pixel size x [m]
const SIM_PIXELSIZE_Z       = SIM_BIN_Z * CAM_PIXELSIZE   # Effective pixel size z [m]
# Pixel counts after binning
const X_PIXELS = Int(NX_PIXELS / SIM_BIN_X)
const Z_PIXELS = Int(NZ_PIXELS / SIM_BIN_Z)
# Centred spatial axes
const X_POSITION = TheoreticalSimulation.pixel_coordinates(X_PIXELS, SIM_BIN_X, SIM_PIXELSIZE_X)
const Z_POSITION = TheoreticalSimulation.pixel_coordinates(Z_PIXELS, SIM_BIN_Z, SIM_PIXELSIZE_Z)

println("""
╔═══════════════════════════════════════════════════╗
  CAMERA
  Sensor pixels     : $(NX_PIXELS) × $(NZ_PIXELS)
  Physical px size  : $(1e6*CAM_PIXELSIZE) μm
  ───────────────────────────────────────────────
  SIMULATION
  Binning           : $(SIM_BIN_X) × $(SIM_BIN_Z)
  Effective pixels  : $(X_PIXELS) × $(Z_PIXELS)
  Eff. px size      : $(1e6*SIM_PIXELSIZE_X) μm × $(1e6*SIM_PIXELSIZE_Z) μm
  x range           : [$(round(1e6*minimum(X_POSITION), digits=2)) μm,  $(round(1e3*maximum(X_POSITION), digits=4)) mm]
  z range           : [$(round(1e6*minimum(Z_POSITION), digits=2)) μm,  $(round(1e3*maximum(Z_POSITION), digits=4)) mm]
╚═══════════════════════════════════════════════════╝
""")

# ==============================================================================
# BEAMLINE GEOMETRY & THERMAL SOURCE
# ==============================================================================

# Thermal source
const T_CELSIUS   = 200
const T_K         = 273.15 + T_CELSIUS      # Furnace temperature      [K]
# Apertures
# Furnace aperture
const X_FURNACE   = 2.0e-3                  # Furnace aperture (x)     [m]
const Z_FURNACE   = 100e-6                  # Furnace aperture (z)     [m]
# Slit : Pre SG
const X_SLIT      = 4.0e-3                  # Pre-SG slit (x)          [m]
const Z_SLIT      = 300e-6                  # Pre-SG slit (z)          [m]
# Circular Aperture : Post SG
const R_APERTURE  = 5.8e-3 / 2             # Post-SG circular aperture [m]
# Propagation distances along beam axis (y)
const Y_FURNACETOSLIT  = 541.75e-3          # Furnace → slit           [m]
const Y_SLITTOSG       = 44.0e-3            # Slit → SG magnet         [m]
const y_SG             = 7.0e-2            # SG magnet length          [m]
const Y_SGTOSCREEN     = 395.25e-3          # SG magnet → screen       [m]
const Y_SGTOAPERTURE   = 1.0e-3            # SG magnet → post aperture [m]
# Vacuum pipe
const R_TUBE = 35e-3 / 2                   # Connecting pipe radius    [m]

const effusion_params = TheoreticalSimulation.BeamEffusionParams(
    X_FURNACE, Z_FURNACE,
    X_SLIT,    Z_SLIT,
    Y_FURNACETOSLIT, T_K,
    K39_params,
)

println("""
╔═══════════════════════════════════════════════════╗
  BEAMLINE SETUP
  Temperature          : $(T_K) K  ($(T_CELSIUS) °C)
  Furnace aperture     : $(1e3*X_FURNACE) mm × $(1e6*Z_FURNACE) μm
  Pre-SG slit          : $(1e3*X_SLIT) mm × $(1e6*Z_SLIT) μm
  Post-SG aperture     : r = $(1e3*R_APERTURE) mm
  ─────────────────────────────────────────
  Furnace → slit       : $(1e3*Y_FURNACETOSLIT) mm
  Slit → SG            : $(1e3*Y_SLITTOSG) mm
  SG length            : $(1e3*y_SG) mm
  SG → screen          : $(1e3*Y_SGTOSCREEN) mm
  SG → aperture        : $(1e3*Y_SGTOAPERTURE) mm
  Tube radius          : $(1e3*R_TUBE) mm
╚═══════════════════════════════════════════════════╝
""")

# Setting the variables for the module
# ── Push geometry into TheoreticalSimulation module ───────────────────────────
let ts = TheoreticalSimulation
    ts.default_camera_pixel_size  = CAM_PIXELSIZE
    ts.default_x_pixels           = NX_PIXELS
    ts.default_z_pixels           = NZ_PIXELS
    ts.default_x_furnace          = X_FURNACE
    ts.default_z_furnace          = Z_FURNACE
    ts.default_x_slit             = X_SLIT
    ts.default_z_slit             = Z_SLIT
    ts.default_y_FurnaceToSlit    = Y_FURNACETOSLIT
    ts.default_y_SlitToSG         = Y_SLITTOSG
    ts.default_y_SG               = y_SG
    ts.default_y_SGToScreen       = Y_SGTOSCREEN
    ts.default_R_tube             = R_TUBE
    ts.default_c_aperture         = R_APERTURE
    ts.default_y_SGToAperture     = Y_SGTOAPERTURE
end
##################################################################################################

# ==============================================================================
# FIT CONFIGURATION
# ==============================================================================

# Experimental data selection
const WANTED_ZBINNING = 1
const WANTED_SMOOTH   = 0.001

# Polynomial background degree
const P_DEGREE   = 5
const NCOLS_BG   = P_DEGREE + 1   # number of polynomial coefficients

const NORM_MODE  = :none
const λ0_EXP     = 0.0001

const NRANGE_Z   = 10_001

const DIR_LIST = [
    "20260529",
    "20260603"
]

# ── PrettyTables header construction ──────────────────────────────────────────
const HDR_TOP = Any[
    "Residuals",
    MultiColumn(2,        "Theoretical PDF"),
    MultiColumn(NCOLS_BG, "Background P$(ProfileFitTools.sub(P_DEGREE))(z)"),
]
const HDR_BOT = vcat(
    ["(exp−model)²", "A", "w [mm]"],
    ["c" * ProfileFitTools.sub(k) for k in 0:P_DEGREE],
)

# ── Results container ─────────────────────────────────────────────────────────
# Key   → date string
# Value → (Ic_lower, [A_fit, w_fit], c_fit_mean, full_matrix)
const results = OrderedDict{String, Tuple{
    Float64,
    Vector{Float64},
    Vector{Float64},
    Matrix{Float64},
}}()

const results_dict = OrderedDict{Any, NamedTuple}()

# ==============================================================================
# DATA LOADING & PROFILE FITTING LOOP
# ==============================================================================

for wanted_data_dir in DIR_LIST
    # ── Load summary JLD2 ─────────────────────────────────────────────────────
    exp_result_path = joinpath(
        BASE_PATH, "EXPDATA_ANALYSIS", "summary",
        wanted_data_dir, wanted_data_dir * "_report_summary.jld2",
    )

    exp_result = jldopen(exp_result_path, "r") do file
        
        data = file[JLD2_MyTools.make_keypath_exp(wanted_data_dir, WANTED_ZBINNING, WANTED_SMOOTH)];

        @info """Imported experimental data:
          directory     : $(wanted_data_dir)
          analysis      : $(data[:RUNSTAMP])
          nz            : $(WANTED_ZBINNING)
          λ₀            : $(WANTED_SMOOTH)
          """

        Ic = data[:Currents];
        Δz =  (data[:mean_F2_peak_pos_raw] .- data[:mean_F1_peak_pos_raw]);

        pretty_table(
            hcat(Ic, Δz)[1:20, :];
            column_labels   = ["Ic [A]", "Δz [mm]"],
            title           = "PEAK SEPARATION DIAGNOSTICS",
            formatters      = [fmt__printf("%+.4f",[1]), fmt__printf("%+.3f",[2])],
            alignment       = :c,
            table_format    = TextTableFormat(borders = text_table_borders__unicode_rounded),
            style           = TextTableStyle(
                                title                   = crayon"red bold",
                                first_line_column_label = crayon"yellow bold",
                                column_label            = crayon"yellow",
                                table_border            = crayon"blue bold"),
            equal_data_column_widths = true,
        )

        # Centre coordinates on the midpoint of the two F-peak positions
        C00 = 0.5 * (data[:mean_F1_peak_pos_raw][1] + data[:mean_F2_peak_pos_raw][1])

        F1_profile = data[:F1_profile_spline]; F1_profile[:, 1] .-= C00
        F2_profile = data[:F2_profile_spline]; F2_profile[:, 1] .-= C00

        return (
            Ic         = Ic,
            z          = collect(data[:z_mm]) .- C00,
            Δz         = Δz,
            F1         = data[:F1_profile],        # raw
            F2         = data[:F2_profile],        # raw
            F1_profile = F1_profile,               # spline-smoothed
            F2_profile = F2_profile,               # spline-smoothed
        )
    end;


    idx = findmin(abs.(exp_result.Ic .- 0.0)) |> ((d, m),) -> d ≤ 0.0005 ? m : nothing
   
    fig_F1 = plot(exp_result.z, exp_result.F1[idx, :];
        label             = "Experimental data",
        seriestype        = :scatter,
        marker            = (:circle, 2, :white),
        markerstrokecolor = :gray26,
    )
    plot!(fig_F1, exp_result.z, exp_result.F1_profile[:, 1 + idx];
        label  = "Interpolated",
        line   = (:solid, 1.5, :red),
    )
    plot!(fig_F1,
        xlabel=L"$z \quad (\mathrm{mm})$",
        legend=:topleft,  
        legend_title=L"$F=1 : I_{c} = %$(1000*exp_result.Ic[idx])\mathrm{mA}$")

    fig_F2 = plot(exp_result.z, exp_result.F2[idx, :];
        label             = "Experimental data",
        seriestype        = :scatter,
        marker            = (:circle, 2, :white),
        markerstrokecolor = :gray26,
    )
    plot!(fig_F2, exp_result.z, exp_result.F2_profile[:, 1 + idx];
        label  = "Interpolated",
        line   = (:solid, 1.5, :blue),
    )
    plot!(fig_F2,
        xlabel=L"$z \quad (\mathrm{mm})$",
        legend=:topleft,  
        legend_title=L"$F=2 : I_{c} = %$(1000*exp_result.Ic[idx])\mathrm{mA}$")

    fig_F1F2 = plot(fig_F1, fig_F2;
        layout = (2, 1),
        size   = (800, 600),
    )
    display(fig_F1F2)

    # ── Current selection ─────────────────────────────────────────────────────
    Ic_sampled = abs.(exp_result.Ic)
    nI         = length(Ic_sampled)

    chosen_currents_idx = [argmin(Ic_sampled)]  # lowest/zero current only
    @info "\e[1;94mTarget current\e[0m  idx=\e[96m$(only(chosen_currents_idx))\e[0m  I₀=\e[93m$(@sprintf("%.4f", only(Ic_sampled[chosen_currents_idx]))) A\e[0m"

    # ── z-grids ───────────────────────────────────────────────────────────────
    z_exp    = exp_result.z;
    range_z  = floor(minimum([maximum(z_exp), abs(minimum(z_exp))]), digits=1)
    z_theory = collect(range(-range_z, range_z; length=NRANGE_Z))

    @assert isapprox(mean(z_theory), 0.0; atol = 10eps(float(range_z))) "mean(z_theory) = $(mean(z_theory)); expected ≈ 0 within atol=$(10eps(float(range_z)))"
    @assert isapprox(std(z_theory), ProfileFitTools.std_sample(range_z, NRANGE_Z); atol = eps(float(range_z))) "std(z_theory) inconsistent with symmetric range"

    # ── Pre-allocate fit containers ───────────────────────────────────────────
    rl   = length(chosen_currents_idx)
    cols = palette(:darkrainbow, rl);

    exp_list    = Vector{Vector{Float64}}(undef, rl)   # splined experiment on z_theory
    pdf_th_list = Vector{Vector{Float64}}(undef, rl)   # theoretical PDF on z_theory
    z_list      = fill(z_theory, rl)                   # shared grid (read-only)

    # Orthonormal polynomial basis (computed once for this grid)
    μ_poly, σ_poly, _t, Q, R = ProfileFitTools.orthonormal_basis_on(z_theory; n=P_DEGREE)
    μ_list = fill(μ_poly, rl)
    σ_list = fill(σ_poly, rl)
    Q_list = fill(Q, rl)
    R_list = fill(R, rl)

    # ── Build experimental & theoretical PDFs ─────────────────────────────────
    for (j, i_idx) in enumerate(chosen_currents_idx)
        I0 = Ic_sampled[i_idx]

        # Experimental profile: spline-interpolate onto z_theory
        amp_exp     = @view exp_result.F1_profile[:, 1+i_idx]   # smoothed profile
        amp_exp     = @view exp_result.F1[i_idx,:]              # raw profile
        spl_exp     = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_EXP;
                          weights = TheoreticalSimulation.compute_weights(z_exp, λ0_EXP))
        pdf_exp     = spl_exp.(z_theory)
        exp_list[j] = ProfileFitTools.normalize_vec(pdf_exp; by = NORM_MODE)

        

        # Theoretical profile: sum over mF sublevels of F=1 manifold
        𝒢       = TheoreticalSimulation.GvsI(I0)
        μ_eff   = [TheoreticalSimulation.μF_effective(I0, v[1], v[2], K39_params)
                   for v in TheoreticalSimulation.fmf_levels(K39_params; Fsel=1)]
        pdf_th  = mapreduce(
            μF -> TheoreticalSimulation.getProbDist_v3(
                      μF, 𝒢, 1e-3 .* z_theory, K39_params, effusion_params; pdf=:finite),
            +, μ_eff,
        )
        pdf_th_list[j] = ProfileFitTools.normalize_vec(pdf_th; by = NORM_MODE)

    end

    # ── Joint fit: global A & w, per-profile polynomial background ────────────
    # Model: G(z) = A · F_theory(z; w) + P_n(z)
    fit_data, fit_params, δparams, modelfun, model_on_z, meta, extras =
        ProfileFitTools.fit_pdf_joint(
            z_list, exp_list, pdf_th_list;
            n      = P_DEGREE,
            Q_list, R_list, μ_list, σ_list,
            w_mode = :global,
            A_mode = :global,
            # d_mode = :per_profile,
            w0     = 0.050,
            A0     = 15.0,
        );

    # Extract polynomial background coefficients (standard basis)
    c_poly_coeffs = [
        let fit_poly = ProfileFitTools.bg_function(z_theory, fit_params.c[i])
            [fit_poly[dg] for dg in 0:P_DEGREE]
        end
        for i in 1:rl
    ]

    w_fit = fit_params.w
    A_fit = fit_params.A

    # Coefficient of determination R²
    ss_res = sum((exp_list[1] .- model_on_z[1]) .^ 2)
    ss_tot = sum((exp_list[1] .- mean(exp_list[1])) .^ 2)
    R²     = 1 - ss_res / ss_tot

    fitting_params = reduce(vcat, hcat(R², A_fit, w_fit, c_poly_coeffs))'

    pretty_table(
        fitting_params;
        column_label_alignment   = :c,
        column_labels            = [HDR_TOP, HDR_BOT],
        row_labels               = round.(1000 * Ic_sampled[chosen_currents_idx]; sigdigits=4),
        formatters               = [
            fmt__printf("%2.4f",  [1]),
            fmt__printf("%4.6f",  2:3),
            fmt__printf("%4.6e",  4:(3 + NCOLS_BG)),
        ],
        alignment                = :c,
        equal_data_column_widths = true,
        stubhead_label           = "I₀ [mA]",
        row_label_column_alignment = :c,
        title                    = "FITTING ANALYSIS — RAW PROFILE : $wanted_data_dir",
        table_format             = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style                    = TextTableStyle(
            first_line_merged_column_label = crayon"light_red bold",
            first_line_column_label        = crayon"yellow bold",
            column_label                   = crayon"yellow",
            table_border                   = crayon"blue bold",
            title                          = crayon"red bold",
        ),
    )

    # Store results for downstream analysis
    results[wanted_data_dir] = (
        Ic_sampled[1],
        [A_fit, w_fit],
        c_poly_coeffs[1],
        hcat(z_theory, exp_list[1], model_on_z[1]),
    )
end

# ── Persist fitting results ───────────────────────────────────────────────────
jldopen(joinpath(OUTDIR, "baseline_results_P$(P_DEGREE).jld2"), "w") do f
    f["fit_results"]  = results
    f["meta/date"]    = RUN_STAMP
    f["meta/Pdegree"] = P_DEGREE
end


# ==============================================================================
# DECONVOLUTION & BLUR KERNEL ANALYSIS LOOP
#
# For each dataset:
#   (1) subtract polynomial baseline
#   (2) build theoretical and geometric PDFs on the screen
#   (3) infer extra blur kernel H by regularised deconvolution: G = F ⊗ H
#   (4) fit parametric shapes to H (Gaussian / Lorentzian / Pseudo-Voigt)
#   (5) propagate best-fit Gaussian blur back into the forward model
# ==============================================================================

for dir_chosen in DIR_LIST
    @info "Processing dataset: $dir_chosen"

    data       = results[dir_chosen]
    p_baseline = Polynomial(data[3])   # background polynomial P_n(z)

    z_range              = data[4][:, 1]   # z grid [mm]
    data_exp             = data[4][:, 2]   # splined experiment on z_range
    base_line            = p_baseline.(z_range)
    data_exp_no_baseline = data_exp .- base_line
    data_exp_normalized  = data_exp_no_baseline ./ data[2][1]   # divide by A_fit

    # ── Geometric beam PDFs ───────────────────────────────────────────────────
    ΔL    = Y_FURNACETOSLIT + Y_SLITTOSG + y_SG + Y_SGTOSCREEN
    δslit = Y_FURNACETOSLIT
    z_m   = 1e-3 .* z_range   # [m]
    Δz    = mean(diff(z_m))

    pdf_oven = ProfileFitTools.unitbox_scaled(z_m, Z_FURNACE * (ΔL - δslit) / δslit; soft=true, ϵ=0.007)
    pdf_slit = ProfileFitTools.unitbox_scaled(z_m, Z_SLIT    *  ΔL          / δslit; soft=true, ϵ=0.007)
    # “furnace ⊗ slit” baseline kernel
    pdf_conv = ProfileFitTools.conv_centered(pdf_oven, pdf_slit, Δz)
    pdf_conv ./= sum(pdf_conv) * Δz

    # Furnace and slit windows
    fig_geom = plot(z_range, pdf_oven;
        label  = "Furnace aperture",
        line   = (:solid, 2, :orangered2),
        xlabel = L"$z$ (mm)",
        xlims  = (-1.5, 1.5),
        title  = "Geometric beam kernels",
    )
    plot!(fig_geom, z_range, pdf_slit;
        label = "Pre-SG slit",
        line  = (:dash, 2, :darkgreen),
    )
    display(fig_geom)

    # ── Theoretical SG profile ────────────────────────────────────────────────
    𝒢     = TheoreticalSimulation.GvsI(data[1])
    μ_eff = [TheoreticalSimulation.μF_effective(data[1], v[1], v[2], K39_params)
             for v in TheoreticalSimulation.fmf_levels(K39_params; Fsel=1)]
    pdf_theory = mapreduce(
        μF -> TheoreticalSimulation.getProbDist_v3(
                  μF, 𝒢, z_m, K39_params, effusion_params; pdf=:finite),
        +, μ_eff,
    )
    pdf_theory ./= sum(pdf_theory) * Δz

    # compares physics PDF vs geometric kernel
    fig_theory = plot(z_range, pdf_theory;
        label  = "SG profile (screen)",
        line   = (:black, 1.5),
        xlims  = (-2, 2),
        title  = "Theory vs. geometric kernel",
    )
    plot!(fig_theory, z_range, pdf_conv;
        label = "Furnace ⊗ Slit",
        line  = (:dash, 1.2, :orangered),
    )
    display(fig_theory)

    # ── Experimental profile: raw, baseline, subtracted ───────────────────────
    fig_exp = plot(z_range[1:8:end], data_exp[1:8:end];
        seriestype       = :scatter,
        marker           = (:circle, 2, :white),
        markerstrokecolor = :black,
        label            = "Experimental data ($dir_chosen)",
        xlabel           = L"$z$ (mm)",
        ylabel           = "Intensity (a.u.)",
        title            = "Experiment & baseline",
    )
    plot!(fig_exp, z_range, base_line;
        label = L"Baseline $P_{%$(P_DEGREE)}(z)$",
        line  = (:dash, 2, :red),
    )
    plot!(fig_exp, z_range[1:8:end], data_exp_no_baseline[1:8:end];
        seriestype        = :scatter,
        marker            = (:circle, 2, :white),
        markerstrokecolor = :gray25,
        label             = "Raw − Baseline",
    )
    display(fig_exp)

    # ── Normalised comparison: experiment vs. theory ──────────────────────────
    fig_norm = plot(z_range[1:8:end], data_exp_normalized[1:8:end];
        seriestype        = :scatter,
        marker            = (:circle, 2, :white),
        markerstrokecolor = :black,
        label             = "Experiment ($dir_chosen), normalised",
        xlabel            = L"$z$ (mm)",
        ylabel            = "Intensity (a.u.)",
        title             = "Normalised experiment vs. theory",
    )
    plot!(fig_norm, z_range, pdf_theory ./ maximum(pdf_theory);
        label  = "Theoretical model",
        line   = (:dash, 1.5, :orangered),
        legend = :outerbottom,
    )
    display(fig_norm)

    # ── Regularised deconvolution: G = F ⊗ H ─────────────────────────────────
    # Forward model :  G(z) = A · F(z) + P_n(z)
    #   G(z)  — raw experimental profile
    #   F(z)  — theoretical SG profile (normalised PDF)
    #   P_n(z)— polynomial baseline
    #
    # Cleaned profile:  𝒢(z) = (G(z) − P_n(z)) / A  ≥ 0,  ∫𝒢 dz = 1
    #
    # Blur model:  𝒢(z) = F(z) ⊗ H(z)
    #   H(z)  — unknown blur kernel, inferred by constrained deconvolution
    G  = max.(data_exp_normalized, 0.0); G  ./= sum(G)  * Δz
    F  = copy(pdf_theory);               F  ./= sum(F)  * Δz


    # It’s doing a regularized deconvolution to infer an unknown kernel H with constraints:
    # nonneg=true → physically meaningful (no negative probability)
    # normalize=true → PDF-like kernel
    # λ smoothness via curvature penalty ||D²H||²
    # sym_weight encourages symmetry of H around 0 (softly)
    # So H_est is your inferred “instrument + unmodelled physics” blur.
    H_est = ProfileFitTools.deconv_kernel(G, F, z_m;
        λ           = 1e-2,
        stepsize    = 1e-2,
        nonneg      = true,
        normalize   = true,
        maxiter     = 50_000,
        verbose_every = 5_000,
        sym_weight  = 1e-6,
    )

    fig_deconv = plot(xlabel = L"$z$ (mm)", title = "Deconvolution result")
    plot!(fig_deconv, z_range, G;
        seriestype        = :scatter,
        marker            = (:circle, 2, :white),
        markerstrokecolor = :black,
        label             = "Experiment ($dir_chosen)",
    )
    plot!(fig_deconv, z_range, F; label = "Theory",        line = (:solid, 2, :blue))
    plot!(fig_deconv, z_range, H_est; label = "Blur kernel H", line = (:solid, 2, :forestgreen))
    display(fig_deconv)

    # ── Reconvolution validation: G ≈ F ⊗ H ──────────────────────────────────
    signal_predicted = ProfileFitTools.conv_centered(F, H_est, Δz)

    fig_reconv = plot(xlabel = L"$z$ (mm)", title = "Reconvolution check")
    plot!(fig_reconv, z_range, G;
        seriestype        = :scatter,
        marker            = (:circle, 3, :white),
        markerstrokewidth = 0.2,
        markerstrokecolor = :black,
        label             = "Experiment ($dir_chosen)",
    )
    plot!(fig_reconv, z_range, signal_predicted;
        label = "F ⊗ H",
        line  = (:solid, 1.3, :red),
    )
    display(fig_reconv)

    # ── Parametric fits to blur kernel H ──────────────────────────────────────
    fit_G  = ProfileFitTools.fit_gaussian(z_m, H_est);   p_G  = coef(fit_G)
    fit_L  = ProfileFitTools.fit_lorentzian(z_m, H_est); p_L  = coef(fit_L)
    fit_PV = ProfileFitTools.fit_pvoigt(z_m, H_est);     p_PV = coef(fit_PV)

    yhat_G  = ProfileFitTools.gauss(z_m, p_G)
    yhat_L  = ProfileFitTools.lorentz(z_m, p_L)
    yhat_PV = ProfileFitTools.pvoigt(z_m, p_PV)

    for (name, yhat, k) in (
            ("Gaussian",      yhat_G,  4),
            ("Lorentzian",    yhat_L,  4),
            ("Pseudo-Voigt",  yhat_PV, 6),
        )
        @info "$name fit  RSS=$(ProfileFitTools.rss(H_est, yhat))  AIC=$(ProfileFitTools.aic(H_est, yhat, k))"
    end

    σ_G_μm = round(1e6 * p_G[3]; sigdigits=6)

    fig_blur = plot(xlabel = L"$z$ (mm)", xlims = (-2.5, 2.5), title = "Blur kernel — parametric fits")
    plot!(fig_blur, z_range, H_est;  label = "H_est ($dir_chosen)",        line = (:solid, 1.8, :forestgreen))
    plot!(fig_blur, z_range, yhat_G;  label = L"Gaussian ($\sigma_w = %$(σ_G_μm)$ μm)", line = (:dash, 1.5, :purple))
    plot!(fig_blur, z_range, yhat_L;  label = "Lorentzian",                line = (:dash, 1.5, :pink))
    plot!(fig_blur, z_range, yhat_PV; label = "Pseudo-Voigt",              line = (:dash, 1.5, :dodgerblue3))
    display(fig_blur)

    # ── Forward model with Gaussian blur ──────────────────────────────────────
    HH = ProfileFitTools.conv_centered(pdf_theory, yhat_G, Δz)

    fig_forward = plot(xlabel = L"$z$ (mm)", xlims = (-3, 3),
        title = "Forward model: Theory ⊗ Gauss blur")
    plot!(fig_forward, z_range, G ./ maximum(G);
        seriestype        = :scatter,
        marker            = (:circle, 3, :white),
        markerstrokewidth = 0.2,
        markerstrokecolor = :black,
        label             = "Experiment ($dir_chosen)",
    )
    plot!(fig_forward, z_range, HH ./ maximum(HH);
        label = L"F $\otimes$ Gauss($%$(round(1e6*p_G[3]; digits=2))$ μm)",
        line  = (:solid, 2, :red),
    )
    display(fig_forward)

    # ── Store per-dataset results ─────────────────────────────────────────────
    results_dict[dir_chosen] = (
        current_A      = data[1],
        blurrGwidth_um = 1e6 * p_G[3],
        zmax_mm        = TheoreticalSimulation.max_of_bspline_positions(z_range, HH; λ0=λ0_EXP )[1][1],
    )


    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS — Convolution/deconvolution pipeline quality checks
    #
    #  G      : cleaned experimental PDF
    #  F      : theoretical PDF
    #  H_est  : estimated blur kernel
    #  LL     : reconstructed profile = F ⊗ H_est  (should ≈ G)
    # ══════════════════════════════════════════════════════════════════════════
    @info "Running pipeline diagnostics for $dir_chosen"

    # (1) Re-normalise all three as PDFs on the same grid
    Gpdf = copy(G);     ProfileFitTools.normalize_pdf!(Gpdf, Δz; nonneg=true)
    Fpdf = copy(F);     ProfileFitTools.normalize_pdf!(Fpdf, Δz; nonneg=true)
    Hpdf = copy(H_est); ProfileFitTools.normalize_pdf!(Hpdf, Δz; nonneg=true)

    @info "PDF norms" ∫G=round(sum(Gpdf)*Δz; digits=3) ∫F=round(sum(Fpdf)*Δz; digits=3) ∫H=round(sum(Hpdf)*Δz; digits=3)

    # (2) Forward reconstruction
    LL = ProfileFitTools.conv_centered(Fpdf, Hpdf, Δz)
    ProfileFitTools.normalize_pdf!(LL, Δz; nonneg=true)

    # (3) Pointwise residual  (oscillations → regularisation issues;
    #                          antisymmetric pattern → centering shift)
    res = LL .- Gpdf

    fig_res  = plot(z_range, res;      label = "Residual",   line = (:blue,  2), xlabel = L"$z$ (mm)")
    fig_abs  = plot(z_range, abs.(res); label = "|Residual|", line = (:black, 2), xlabel = L"$z$ (mm)")
    display(plot(fig_res, fig_abs; layout=(1,2), title="Residual reconstruction"))

    # (4) CDF comparison  (shifts appear as an S-shaped ΔCDF)
    cdf_G  = cumsum(Gpdf) * Δz;  cdf_G  ./= cdf_G[end]
    cdf_LL = cumsum(LL)   * Δz;  cdf_LL ./= cdf_LL[end]

    fig_cdf = plot(z_range, cdf_G;
        label  = "CDF — Experiment ($dir_chosen)",
        line   = (:black, 2),
        xlabel = L"$z$ (mm)",
        ylabel = "Cumulative integral",
        title  = "CDF comparison (centering diagnostic)",
    )
    plot!(fig_cdf, z_range, cdf_LL;
        label = "CDF — Model",
        line  = (:red, 2, :dash),
    )
    display(fig_cdf)

    fig_dcdf = plot(z_range, cdf_LL .- cdf_G;
        label  = "ΔCDF",
        line   = (:purple, 2),
        xlabel = L"$z$ (mm)",
        ylabel = "CDF(Model) − CDF(Experiment)",
        title  = "ΔCDF  (sensitive to small shifts)",
    )
    display(fig_dcdf)

    # (5) Numeric summary
    rss_val = sum(abs2, res)
    l1_val  = sum(abs,  res) * Δz
    μ_G_cm  = sum(z_m .* Gpdf) * Δz
    μ_LL    = sum(z_m .* LL)   * Δz
    @info "Reconstruction quality" RSS=rss_val L1=l1_val Δμ=(μ_LL - μ_G_cm)

    diag = ProfileFitTools.sg_width_diagnostic(z_m, Gpdf, LL, Δz)
    @info "SG width diagnostic" Δμ=diag.Δμ Δσ²=diag.Δσ² σ_G=diag.σG σ_F=diag.σF

    # ══════════════════════════════════════════════════════════════════════════
    # END DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════
end


jldopen(joinpath(OUTDIR,"blur_conv.jld2"), "w") do f
    f["convolution"]    = results_dict
end


T_END = Dates.now()
T_RUN = Dates.canonicalize(T_END-T_START)

report = """
***************************************************
EXPERIMENT
    Single Stern–Gerlach Experiment
    Output directory            : $(OUTDIR)
    Run label                   : $(RUN_STAMP)
    

EXPERIMENT ANALYSIS PROPERTIES    
    Analysis Binning            : $(WANTED_ZBINNING)
    Analysis spline smoothing   : $(WANTED_SMOOTH)
    Analysis directories        : $(DIR_LIST)

CAMERA FEATURES
    Number of pixels            : $(NX_PIXELS) × $(NZ_PIXELS)
    Pixel size                  : $(1e6*CAM_PIXELSIZE) μm

FITTING INFORMATION
    Normalization mode          : $(NORM_MODE)
    No z-divisions              : $(NRANGE_Z)
    Polynomial degree           : $(P_DEGREE)

CODE
    Code name                   : $(PROGRAM_FILE),
    Start date                  : $(T_START)
    End data                    : $(T_END)
    Run time                    : $(T_RUN)
    Hostname                    : $(HOSTNAME)

***************************************************
"""

# Print to terminal
println(report)

# Save to file
open(joinpath(OUTDIR,"convolution_report.txt"), "w") do io
    write(io, report)
end

println("Experiment analysis finished!")
alert("Experiment analysis finished!")


