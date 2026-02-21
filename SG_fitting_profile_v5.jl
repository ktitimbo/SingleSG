# Fitting experimental profile
# Kelvin Titimbo — Caltech — January 2026
# Fitting for the zero -or lowest recorded- current

#  Plotting Setup
# ENV["GKS_WSTYPE"] = "101"
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
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
# Numerical tools
using LinearAlgebra, DataStructures
using LsqFit, DSP, FFTW
using BSplineKit
using Polynomials
using StatsBase
using Statistics, Distributions, StaticArrays
using Alert
# Data manipulation
using OrderedCollections
using JLD2
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Set the working directory to the current location
cd(@__DIR__) ;
const OUTDIR    = joinpath(@__DIR__, "data_studies", "CONV"*RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
const HOSTNAME = gethostname()
@info "Running on host" hostname = HOSTNAME
# Custom modules
include("./Modules/atoms.jl");
include("./Modules/samplings.jl");
include("./Modules/DataReading.jl");
include("./Modules/ProfileFitTools.jl");
include("./Modules/JLD2_MyTools.jl");
include("./Modules/TheoreticalSimulation.jl");
# Propagate output settings to TheoreticalSimulation
TheoreticalSimulation.SAVE_FIG = SAVE_FIG;
TheoreticalSimulation.FIG_EXT  = FIG_EXT;
TheoreticalSimulation.OUTDIR   = OUTDIR;

@info "Run stamp initialized" RUN_STAMP = RUN_STAMP
println("\n\t\tRunning process on:\t $(RUN_STAMP) \n")

const ATOM        = "39K"  ;
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
K39_params = TheoreticalSimulation.AtomParams(ATOM);

# STERN--GERLACH EXPERIMENT
# Camera and pixel geometry : intrinsic properties
cam_pixelsize = 6.5e-6 ;  # Physical pixel size of camera [m]
nx_pixels , nz_pixels= (2160, 2560); # (Nx,Nz) pixels
println("""
***************************************************
CAMERA FEATURES
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm
***************************************************
""")
# Furnace
T_K = 273.15 + 205 ; # Furnace temperature (K)
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
# Connecting pipes
const R_tube = 35e-3/2 ; # Radius of the connecting pipe (m)
effusion_params = TheoreticalSimulation.BeamEffusionParams(x_furnace,z_furnace,x_slit,z_slit,y_FurnaceToSlit,T_K,K39_params);
println("""
***************************************************
SETUP FEATURES
    Temperature (K          : $(T_K)
    Furnace aperture (x,z)  : ($(1e3*x_furnace)mm , $(1e6*z_furnace)μm)
    Slit (x,z)              : ($(1e3*x_slit)mm , $(1e6*z_slit)μm)
    Furnace → Slit          : $(1e3*y_FurnaceToSlit)mm
    Slit → SG magnet        : $(1e3*y_SlitToSG)mm
    SG magnet               : $(1e3*y_SG)mm
    SG magnet → Screen      : $(1e3*y_SGToScreen)mm
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

# Select experimental data
dict = OrderedDict{String, Tuple{
    Float64,           # Ic[lower]
    Vector{Float64},   # [A_fit, w_fit]
    Vector{Float64},   # c_fit_mean
    Matrix{Float64}    # hcat(...)
}}()

wanted_binning  = 2 ; 
wanted_smooth   = 0.01 ;

P_DEGREE    = 5 ;
ncols_bg    = P_DEGREE + 1 ;

norm_mode = :none ;
λ0_exp     = 0.001 ;

nrange_z = 20001;

dir_list = [
    "20250814" , "20250820" , "20250825" , 
    "20250919" , 
    "20251002" , "20251003", "20251006",
    "20260211" , "20260213"
]

hdr_top = Any[
    "Residuals",
    MultiColumn(2, "Theoretical PDF"),
    MultiColumn(ncols_bg, "Background P$(ProfileFitTools.sub(P_DEGREE))(z)")
];
hdr_bot = vcat(["(exp-model)²", "A", "w [mm]"], ["c" * ProfileFitTools.sub(k) for k in 0:P_DEGREE]);

for wanted_data_dir in dir_list
    # wanted_data_dir = dir_list[1]

    # Data loading
    read_exp_info = DataReading.find_report_data(
            joinpath(@__DIR__, "analysis_data");
            wanted_data_dir=wanted_data_dir,
            wanted_binning=wanted_binning,
            wanted_smooth=wanted_smooth
    );
    [(String(k), getfield(read_exp_info, k)) for k in propertynames(read_exp_info)];
    if isnothing(read_exp_info)
        @warn "No matching report found" wanted_data_dir wanted_binning wanted_smooth
    else
        msg = join([
        "Imported experimental data:",
        "  directory     : $(read_exp_info.directory)",
        "  path          : $(read_exp_info.path)",
        "  data_dir      : $(read_exp_info.data_dir)",
        "  analysis_name : $(read_exp_info.name)",
        "  binning       : $(read_exp_info.binning)",
        "  smoothing     : $(read_exp_info.smoothing)",
        "  magnification : $(read_exp_info.magnification)",
        ], "\n")
        @info msg
    end

    exp_data = load(joinpath(read_exp_info.directory,"profiles_mean.jld2"))["profiles"];
    # jldopen(joinpath(@__DIR__, "analysis_data", wanted_data_dir, wanted_data_dir*"_report_summary.jld2"),"r") do file
    #     file[JLD2_MyTools.make_keypath_exp(wanted_data_dir, wanted_binning, wanted_smooth)]
    # end
    Ic_sampled = abs.(exp_data[:Icoils]);
    nI = length(Ic_sampled);

    chosen_currents_idx = [1]

    println("Target currents in A: (", 
            join(map(x -> @sprintf("%.3f", x), Ic_sampled[chosen_currents_idx]), ", "),
            ")"
    )

    magnification_factor = read_exp_info.magnification ;

    z_exp    = (exp_data[:z_mm] .- exp_data[:Centroid_mm][1]) ./ magnification_factor ;
    range_z  = floor(minimum([maximum(z_exp),abs(minimum(z_exp))]),digits=1);
    z_theory = collect(range(-range_z,range_z,length=nrange_z));

    @assert isapprox(mean(z_theory), 0.0; atol= 10 * eps(float(range_z)) ) "μz=$(μz) not ~ 0 within atol=$(10 * eps(float(range_z)) )"
    @assert isapprox(std(z_theory), ProfileFitTools.std_sample(range_z, nrange_z); atol= eps(float(range_z))) "σz=$(σz) is not defined for a symmetric range"

    rl   = length(chosen_currents_idx) ;
    cols = palette(:darkrainbow, rl);

    # Preallocate containers
    exp_list     = Vector{Vector{Float64}}(undef, rl);   # splined/normalized experiment on z_theory
    pdf_th_list  = Vector{Vector{Float64}}(undef, rl) ;  # closed-form theory on z_theory
    z_list       = fill(z_theory, rl) ;                  # same grid for all (read-only is fine)

    # precompute for this grid:
    μ, σ, _t, Q, R = ProfileFitTools.orthonormal_basis_on(z_theory; n=P_DEGREE);
    μ_list = fill(μ, rl);  σ_list = fill(σ, rl);
    Q_list = fill(Q, rl);  R_list = fill(R, rl);


    for (j,i_idx) in enumerate(chosen_currents_idx)
        
        I0 = Ic_sampled[i_idx]

        # EXPERIMENT
        amp_exp     = @view exp_data[:F1_profile][i_idx, :]
        Spl_exp     = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp;
                                weights = TheoreticalSimulation.compute_weights(z_exp, λ0_exp))
        pdf_exp     = Spl_exp.(z_theory)
        exp_list[j] = ProfileFitTools.normalize_vec(pdf_exp; by = norm_mode)

        # THEORY
        𝒢           = TheoreticalSimulation.GvsI(I0)
        μ_eff       = [TheoreticalSimulation.μF_effective(I0, v[1], v[2], K39_params)
                for v in TheoreticalSimulation.fmf_levels(K39_params; Fsel=1)]
        pdf_theory  = mapreduce(μF -> TheoreticalSimulation.getProbDist_v3(
                                μF, 𝒢, 1e-3 .* z_theory, K39_params, effusion_params; pdf=:finite),
                            +, μ_eff)
        pdf_th_list[j] = ProfileFitTools.normalize_vec(pdf_theory; by = norm_mode)
    end

    #########################################################################################################
    # (1) w = :global & A = :global & Pn =:per_profile
    #########################################################################################################
    fit_data, fit_params, δparams, modelfun, model_on_z, meta, extras = ProfileFitTools.fit_pdf_joint(z_list, exp_list, pdf_th_list;
                n=P_DEGREE, Q_list, R_list, μ_list, σ_list,
                w_mode=:global, A_mode=:global, d_mode =:global,
                w0=0.25, A0=1.0);

    c_poly_coeffs = [Vector{Float64}(undef, ncols_bg) for _ in 1:rl]
    for i=1:rl
        fit_poly = ProfileFitTools.bg_function(z_theory,fit_params.c[i])
        c_poly_coeffs[i] = [fit_poly[dg] for dg in 0:P_DEGREE]
    end

    w_fit = fit_params.w
    A_fit = fit_params.A

    # coefficient of determination
    ss_res = sum((exp_list[1] .- model_on_z[1]).^2);
    ss_tot = sum((exp_list[1] .- mean(exp_list[1])).^2);
    R2 = 1 - ss_res / ss_tot

    fitting_params  = reduce(vcat,hcat(R2, A_fit, w_fit, c_poly_coeffs))'

    pretty_table(
        fitting_params;
        column_label_alignment      = :c,
        column_labels               = [hdr_top, hdr_bot],
        row_labels                  = round.(1000*Ic_sampled[chosen_currents_idx], sigdigits=4),
            formatters                  = [fmt__printf("%2.4f", [1]), fmt__printf("%4.6f", 2:3), fmt__printf("%4.6e", 4:(1+2+2+ncols_bg))],
        alignment                   = :c,
        equal_data_column_widths    = true,
        stubhead_label              = "I₀ [mA]",
        row_label_column_alignment  = :c,
        title                       = "FITTING ANALYSIS - RAW PROFILE : $(wanted_data_dir)",
        table_format                = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style                       = TextTableStyle(
                                            first_line_merged_column_label  = crayon"light_red bold",
                                            first_line_column_label         = crayon"yellow bold",
                                            column_label                    = crayon"yellow",
                                            table_border                    = crayon"blue bold",
                                            title                           = crayon"red bold"
                                        )
    )
    dict[wanted_data_dir] =
        (Ic_sampled[1], [A_fit, w_fit], c_poly_coeffs[1], hcat(z_theory, exp_list[1],model_on_z[1]))
end

jldopen(joinpath(OUTDIR,"baseline_results_P$(P_DEGREE).jld2"), "w") do f
    f["fit_results"]    = dict
    f["meta/date"]      = RUN_STAMP
    f["meta/Pdegree"]   = P_DEGREE
end

# dir_chosen = "20260211" ;
results_dict = Dict{Any,NamedTuple}()
# For each dataset
# (1) subtract a polynomial baseline, 
# (2) build theoretical and geometric PDFs on the screen, 
# (3) infer an extra blur kernel by regularized deconvolution, 
# (4) parametrize that blur kernel (Gaussian/Lorentz/PV) and pick the best, 
# (5) refit experimental data using the inferred Gaussian width as a fixed parameter.
for dir_chosen in dir_list
    @info "ANALYSIS $(dir_chosen)" 
    data = dict[dir_chosen]
    p_baseline = Polynomial(data[3]); # background polynomial

    z_range   = data[4][:,1];     # z-values
    data_exp  = data[4][:,2];    # interpolated experiment evaluated in the given z
    base_line = p_baseline.(z_range);   # evaluate the background polynomial at z
    data_exp_no_baseline = (data_exp .- base_line);     # experiment minus background
    data_exp_normalized  = data_exp_no_baseline ./ data[2][1]; # normalized profile

    ΔL      = y_FurnaceToSlit + y_SlitToSG + y_SG + y_SGToScreen
    δslit   = y_FurnaceToSlit
    z_m     = 1e-3*z_range
    pdf_oven = ProfileFitTools.unitbox_scaled(z_m, z_furnace*(ΔL-δslit)/δslit ; soft=true, ϵ=0.007)
    pdf_slit = ProfileFitTools.unitbox_scaled(z_m, z_slit* ΔL/δslit ; soft=true, ϵ=0.007)
    Δz = mean(diff(z_m))   # in mm (or whatever your z units are)
    # “furnace ⊗ slit” baseline kernel
    pdf_conv = ProfileFitTools.conv_centered(pdf_oven, pdf_slit, Δz)
    pdf_conv ./= (sum(pdf_conv) * Δz)

    # Furnace and slit windows
    figA = plot(z_range, pdf_oven,
    label="Furnace",
    line=(:solid,2,:orangered2),
    xlabel=L"$z$ (mm)",
    xlims = (-1.5,1.5));
    plot!(figA, z_range, pdf_slit,
    line=(:dash,2,:darkgreen),
    label="Slit");
    plot!(title="Theoretical");
    display(figA)

    𝒢     = TheoreticalSimulation.GvsI(data[1])
    μ_eff = [TheoreticalSimulation.μF_effective(data[1], v[1], v[2], K39_params) for v in TheoreticalSimulation.fmf_levels(K39_params; Fsel=1)]
    pdf_theory  = mapreduce(μF -> TheoreticalSimulation.getProbDist_v3(
                            μF, 𝒢, z_m, K39_params, effusion_params; pdf=:finite),
                        +, μ_eff)
    pdf_theory ./= (sum(pdf_theory) * Δz)

    # compares physics PDF vs geometric kernel
    figB = plot(z_range, pdf_theory,
        label="Profile at the screen",
        line=(:black,1.5),
        xlims=(-2,2)
    );
    plot!(figB, z_range, pdf_conv,
        label="Conv(furnace,slit)",
        line=(:dash,:orangered,1.2)
    );
    plot!(title="Simulation & Model");
    display(figB)

    # raw exp + baseline + baseline-subtracted
    figC = plot(z_range[1:8:end], data_exp[1:8:end],
        seriestype=:scatter,
        marker=(:circle,2,:white),
        markerstrokecolor=:black,
        label="Experimental data ($(dir_chosen))",
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)"
    );
    plot!(figC, z_range, base_line,
        label=L"Baseline $P_{%$(P_DEGREE)}(z)$",
        line=(:dash,2,:red)
    );
    plot!(figC, z_range,  data_exp_no_baseline,
        label="Raw–Baseline",
        seriestype=:scatter,
        marker=(:circle,2,:white),
        markerstrokecolor=:gray25,
    );
    plot!(title="Experiment & background");
    display(figC)

    # normalized exp vs normalized theory
    figD = plot(z_range[1:8:end], data_exp_normalized[1:8:end],
        seriestype=:scatter,
        marker=(:circle,2,:white),
        markerstrokecolor=:black,
        label="Experimental data ($(dir_chosen)): normalized",
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)"
    );
    plot!(figD, z_range, pdf_theory/maximum(pdf_theory),
        label="Theoretical model",
        line=(:dash,1.5,:orangered),
        legend=:outerbottom,
    );
    display(figD)

    # G is the experimental profile
    # F is the theoretical model of the profile
    # G(z) = A F(z) + background(z)
    # We define 𝒢(z) = (G(z) - background(z))/A : experimental cleaned profile, clipped nonnegative, normalized as a PDF.
    # Our model considers that 𝒢(z) = F(z) ⊗ H(z)
    # H(z) is the unknown blurring function
    G = max.(data_exp_normalized, 0.0)
    G ./= (sum(G) * Δz)
    F = pdf_theory
    F ./= (sum(F) * Δz)   # if F is meant as a PDF kernel too

    # It’s doing a regularized deconvolution to infer an unknown kernel H with constraints:
    # nonneg=true → physically meaningful (no negative probability)
    # normalize=true → PDF-like kernel
    # λ smoothness via curvature penalty ||D²H||²
    # sym_weight encourages symmetry of H around 0 (softly)
    # So H_est is your inferred “instrument + unmodelled physics” blur.
    H_est = ProfileFitTools.deconv_kernel(G, F, z_m;
                        λ=1e-2, stepsize=1e-2, 
                        nonneg=true, normalize=true,
                        maxiter=50000, verbose_every = 5000,
                        sym_weight=1e-6)
    
    figE = plot(xlabel=L"$z$ (mm)")
    plot!(figE, z_range,G,
        label="Experiment ($dir_chosen)",
        seriestype=:scatter,
        marker=(:circle,2,:white),
        markerstrokecolor=:black,
    );
    plot!(figE, z_range,F,
        label="Theoretical",
        line=(:solid,2,:blue)
    );
    plot!(figE, z_range,H_est,
        label="Blurring function",
        line=(:solid,2,:forestgreen)
    );
    plot!(title="Normalized comparison")
    display(figE)

    # Validate the deconvolution by reconvolution
    # signal_predicted = F ⊗ H : Conv(theory,Smoothing)
    signal_predicted = ProfileFitTools.conv_centered(F, H_est, Δz);
    figF = plot(xlabel=L"$z$ (mm)");
    plot!(figF, z_range, G,
        label="Experiment ($dir_chosen)",
        seriestype=:scatter,
        marker=(:circle,3,:white),
        markerstrokewidth=0.2,
        markerstrokecolor=:black
    );
    plot!(figF, z_range, signal_predicted,
        label="Conv(Theory,Blurring)",
        line=(:solid,1.3,:red)
    );
    display(figF)

    # Fit parametric shapes to the inferred blur kernel H_est
    fitG = ProfileFitTools.fit_gaussian(z_m, H_est);
    pG   = coef(fitG)

    fitL = ProfileFitTools.fit_lorentzian(z_m, H_est);
    pL   = coef(fitL)

    fitPV = ProfileFitTools.fit_pvoigt(z_m, H_est);
    pPV   = coef(fitPV)

    yhatG  = ProfileFitTools.gauss(z_m, pG);
    yhatL  = ProfileFitTools.lorentz(z_m, pL);
    yhatPV = ProfileFitTools.pvoigt(z_m, pPV);

    @info "Gaussian fitting statistics"
    @show  ProfileFitTools.rss(H_est, yhatG) ProfileFitTools.aic(H_est, yhatG, 4);
    @info "Lorentzian fitting statistics"
    @show ProfileFitTools.rss(H_est, yhatL) ProfileFitTools.aic(H_est, yhatL, 4);
    @info "PseudoVoigt fitting statistics"
    @show ProfileFitTools.rss(H_est, yhatPV) ProfileFitTools.aic(H_est, yhatPV, 6);

    figH = plot(xlabel=L"$z$ (mm)",
        xlims=(-2.5,2.5));
    plot!(figH, z_range, H_est,
        label="Blurring function for $(dir_chosen)",
        line=(:solid,1.8,:forestgreen)
    );
    plot!(figH, z_range, yhatG,
        label=L"Gaussian fit $(\sigma_{w}=%$(round(1e6*pG[3];sigdigits=6)))\mathrm{\mu m}$",
        line=(:dash,1.5,:purple)
    );
    display(figH)
    plot!(figH, z_range, yhatL,
        label="Lorentzian fit",
        line=(:dash,1.5,:pink)
    );
    plot!(figH, z_range, yhatPV,
        label="Pseudo-Voigt fit",
        line=(:dash,1.5,:dodgerblue3)
    );
    display(figH)

    # Propagate the Gaussian blur back into the forward model
    HH = ProfileFitTools.conv_centered(pdf_theory, yhatG, Δz)
    figI = plot(z_range, G / maximum(G),
        label="Experiment ($(dir_chosen))",
        seriestype=:scatter,
        marker=(:circle,3,:white),
        markerstrokewidth=0.2,
        markerstrokecolor=:black
    );
    plot!(figI, z_range, HH / maximum(HH),
        label=L"Conv(Theory, Gauss($%$(round(1e6*pG[3];digits=2))\mathrm{\mu m}$)",
        line=(:red,2,:solid),
        xlabel=L"$z$ (mm)",
        xlims=(-3,3),
    );
    display(figI)
 
    # --- store results ---
    results_dict[dir_chosen] = (
    current_A      = data[1],
    blurrGwidth_um = 1e6 * pG[3],
    zmax_mm        = TheoreticalSimulation.max_of_bspline_positions(z_range, HH ; λ0=0.001)[1][1]
    )


    # ============================================================
    # Diagnostics for convolution/deconvolution pipeline
    #   Goal:
    #     - Check reconstruction error:      (signal_predicted - G)
    #     - Check subtle centering shifts:   compare cumulative integrals (CDF-like)
    #
    #   Definitions (your notation):
    #     G      : experimental profile without baseline (as PDF)
    #     F      : theoretical profile (as PDF)
    #     H_est  : estimated blur kernel (as PDF)
    #     signal_predicted     : reconstructed profile = F ⊗ H_est
    #
    #   Assumptions:
    #     - z_m is your z-grid (meters) with ~uniform spacing
    #     - Δz = mean(diff(z_m))
    #     - ProfileFitTools.conv_centered implements centered "same" convolution
    # ============================================================
    @info "Running diagnostic"
    # ----------------------------
    # 1) Ensure G, F, H_est are PDFs on the same grid
    # ----------------------------
    Gpdf = copy(G);
    Fpdf = copy(F);
    Hpdf = copy(H_est);

    ProfileFitTools.normalize_pdf!(Gpdf, Δz; nonneg=true);
    ProfileFitTools.normalize_pdf!(Fpdf, Δz; nonneg=true);
    ProfileFitTools.normalize_pdf!(Hpdf, Δz; nonneg=true);

    @show sum(Gpdf)*Δz sum(Fpdf)*Δz sum(Hpdf)*Δz;

    # ----------------------------
    # 2) Forward reconstruction: LL = F ⊗ H
    # ----------------------------
    LL = ProfileFitTools.conv_centered(Fpdf, Hpdf, Δz);
    LL = ProfileFitTools.normalize_pdf!(LL, Δz; nonneg=true);  # keep it a PDF for fair comparisons

    # ----------------------------
    # 3) Residual diagnostic: (LL - G)
    #     - If residual shows oscillations: deconv/regularization issues
    #     - If residual shows antisymmetry / systematic sign change: centering shift
    # ----------------------------
    res = LL .- Gpdf

    fig_res = plot(
        z_range, res,
        xlabel = L"$z$ (mm)",
        label  = "Residual",
        line   = (:blue, 2),
    );
    fig_abs = plot(
        z_range, abs.(res),
        xlabel = L"$z$ (mm)",
        label  = "|Residual|",
        line   = (:black, 2),
    );
    plot(fig_res, fig_abs,
        suptitle="Residual reconstruction",
        layour=(1,2))

    # ----------------------------
    # 4) Cumulative integral diagnostic (CDF-like)
    #     Why it helps:
    #       - Even a tiny shift in LL vs G creates a clearly visible offset in CDFs
    #       - CDF comparison is much less sensitive to pointwise noise
    # ----------------------------
    cdf_G  = cumsum(Gpdf) * Δz;
    cdf_LL = cumsum(LL)   * Δz;

    # Force exact endpoints to reduce tiny numerical drift (optional)
    cdf_G  ./= cdf_G[end];
    cdf_LL ./= cdf_LL[end];

    fig_cdf = plot(
        z_range, cdf_G,
        xlabel = L"$z$ (mm)",
        ylabel = "Cumulative integral",
        label  = "CDF(Experiment $(dir_chosen))",
        line   = (:black, 2),
        title  = "CDF comparison (centering diagnostic)",
    );
    plot!(fig_cdf, z_range, cdf_LL,
        label = "CDF(Model)",
        line  = (:red, 2, :dash),
    );
    display(fig_cdf)

    # Difference of CDFs is an even sharper centering detector:
    # - If LL is shifted right, CDF_LL - CDF_G tends to show a characteristic S-shape.
    dcdf = cdf_LL .- cdf_G;
    fig_dcdf = plot(
        z_range, dcdf,
        xlabel = L"$z$ (m)",
        ylabel = "CDF(Model) - CDF(Experiment)",
        label  = "ΔCDF",
        line   = (:purple, 2),
        title  = "ΔCDF (very sensitive to small shifts)",
    );
    display(fig_dcdf)

    # ----------------------------
    # 5) Quick numeric summaries (optional but useful)
    # ----------------------------
    rss_val = sum(abs2, res)              # pointwise mismatch (PDF units)
    l1_val  = sum(abs.(res)) * Δz         # integrated absolute error
    @show rss_val l1_val;

    # A small “center-of-mass” shift estimate (if it exists):
    μG  = sum(z_m .* Gpdf) * Δz ;
    μLL = sum(z_m .* LL)   * Δz ;
    @show μG μLL (μLL - μG)  ;

    diag = ProfileFitTools.sg_width_diagnostic(z_m, Gpdf, LL, Δz)
    @info "SG diagnostic"
    @show diag.Δμ diag.Δσ² diag.σG diag.σF;

    # ============================================================
    # End diagnostics
    # ============================================================

    # """
    # Estimate mixture model: G ≈ (1-ε)F + ε(F * H), with H≥0, ∫H=1.
    # Returns (ε, H, recon, residual).
    # """
    # G0 = data_exp_normalized; 
    # G0 ./= sum(G0)*Δz
    # F0 = pdf_theory;           
    # F0 ./= sum(F0)*Δz;

    # ε, Hmix, recon, resid = ProfileFitTools.fit_mixture_blur(G0, F0, z_m; ε0=0.85, nouter=15, λ=1e-2, sym_weight=1e-6)
    # plot(z_range, G0, label="Exp.Signal($dir_chosen)");
    # plot!(z_range, recon, label="Mixture recombined");
    # plot!(z_range, Hmix, label="Blurring");
    # plot(z_m, resid, label="residual")
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
    Analysis Binning            : $(wanted_binning)
    Analysis spline smoothing   : $(wanted_smooth)
    Analysis directories        : $(dir_list)

CAMERA FEATURES
    Number of pixels            : $(nx_pixels) × $(nz_pixels)
    Pixel size                  : $(1e6*cam_pixelsize) μm

FITTING INFORMATION
    Normalization mode          : $(norm_mode)
    No z-divisions              : $(nrange_z)
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
