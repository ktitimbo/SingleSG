# Fitting experimental profile
# Kelvin Titimbo
# California Institute of Technology
# October 2025

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
# Numerical tools
using LinearAlgebra, DataStructures
# using Interpolations, Roots, Loess, Optim
using LsqFit
using BSplineKit
using Polynomials
# using DSP
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
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");
const OUTDIR    = joinpath(@__DIR__, "data_studies", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
# Custom modules
include("./Modules/atoms.jl");
include("./Modules/samplings.jl");
include("./Modules/DataReading.jl");
include("./Modules/TheoreticalSimulation.jl");
TheoreticalSimulation.SAVE_FIG = SAVE_FIG;
TheoreticalSimulation.FIG_EXT  = FIG_EXT;
TheoreticalSimulation.OUTDIR   = OUTDIR;

println("\n\t\tRunning process on:\t $(RUN_STAMP) \n")

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
K39_params = TheoreticalSimulation.AtomParams(atom);

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
x_position = TheoreticalSimulation.pixel_coordinates(x_pixels, sim_bin_x, sim_pixelsize_x);
z_position = TheoreticalSimulation.pixel_coordinates(z_pixels, sim_bin_z, sim_pixelsize_z);
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
    Temperature             : $(T_K)
    Furnace aperture (x,z)  : ($(1e3*x_furnace)μm , $(1e6*z_furnace)μm)
    Slit (x,z)              : ($(1e3*x_slit)μm , $(1e6*z_slit)μm)
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


normalize_vec(v; by::Symbol = :max, atol = 0) = begin
    denom = by === :max  ? maximum(v) :
            by === :sum  ? sum(v)      :
            by === :none ? 1           :
            throw(ArgumentError("by must be :max, :sum, or :none"))
    (by === :none || abs(denom) ≤ atol) ? v : v ./ denom
end

std_sample(a, N) = a * sqrt(N*(N+1) / (3*(N-1)^2))  # N = 2n+1 (odd count of points)

function fit_pdf(
    z::AbstractVector,
    pdf_exp::AbstractVector,
    pdf_theory::AbstractVector;
    w0::Float64,
    A0::Float64 = 1.0,
    c0::AbstractVector = [0.0, 0.0, 0.0, 0.0],
    progress_every::Int = 10,)

    @assert length(z) == length(pdf_exp) == length(pdf_theory)
    μz = (first(z) + last(z)) / 2
    σz = std(z)
    @assert σz > 0 "z has zero variance"
    invσz = inv(σz);

    # Helper so printing works even when numbers are Duals
    toflt(x) = try
        Float64(x)
    catch
        try getfield(Main, :ForwardDiff).value(x) |> Float64 catch
            try getfield(x, :value) |> Float64 catch; NaN end
        end
    end

    # Parameters: p = [logw, logA, c0, c1, c2, c3]
    p0 = [
        log(float(w0)),
        log(float(A0)),
        float(c0[1]),
        float(c0[2]),
        float(c0[3]),
        float(c0[4]),
    ]

    """
    make_pdfmix_model(μz, invσz, pdf_theory)

    Return a function `model(zz, p)` that computes
    A * ProbDist_convolved(zz, pdf_theory, w) + cubic((zz-μz)*invσz; c0..c3).
    p = [logw, logA, c0, c1, c2, c3]
    """
    tt_z = (z .- μz) .* invσz
    make_model(tt_z, pdf_theory) = function (zz::AbstractVector{<:Real}, p::AbstractVector{<:Real})
        logw, logA, c₀, c₁, c₂, c₃ = p
        A, w  = exp(logA), exp(logw)
        conv  = TheoreticalSimulation.ProbDist_convolved(zz, pdf_theory, w)     # alloc-returning
        poly  = @. muladd(tt_z, muladd(tt_z, muladd(tt_z, c₃, c₂), c₁), c₀)
        @. A * conv + poly
    end

    calls = Ref(0)
    best  = Ref((rss = Inf, p = copy(p0)))  # track best (numeric) probe

    # Create the model ONCE; then call it inside pdfmix_model
    model = make_model(tt_z, pdf_theory)

    # --- Model: LsqFit expects model(x, p) ---
    function pdf_model(zz::AbstractVector{<:Real}, p::AbstractVector{<:Real})
        # logw, logA, c₀, c₁, c₂, c₃ = p
        # A, w = exp(logA), exp(logw)

        # tt   = (zz .- μz) .* invσz
        # conv = TheoreticalSimulation.ProbDist_convolved(zz, pdf_theory, w)
        # poly = muladd.(tt, muladd.(tt, muladd.(tt, c₃, c₂), c₁), c₀)
        # yhat = @. A * conv + poly
        yhat = model(zz, p)

        if progress_every > 0
            calls[] += 1
            if calls[] % progress_every == 0
                # numeric copies for printing / best-tracking
                rss_val = toflt(sum(abs2, yhat .- pdf_exp))
                p_val   = map(toflt, p)  # Vector{Float64}
                if rss_val < best[].rss
                    best[] = (rss = rss_val, p = p_val)
                end
                @printf(
                    stderr,
                    "eval %6d | rss≈%.6g \t| w≈%.6g\t A≈%.6g\t c≈(%.3g, %.3g, %.3g, %.3g)\n",
                    calls[], rss_val,
                    exp(p_val[1]), exp(p_val[2]),
                    p_val[3], p_val[4], p_val[5], p_val[6],
                )
            end
        end

        return yhat
    end

    fit_data = LsqFit.curve_fit(pdf_model, z, pdf_exp, p0; autodiff = :forward)

    p̂ = coef(fit_data)
    logw, logA, c₀, c₁, c₂, c₃ = p̂
    A, w = exp(logA), exp(logw)

    se = stderror(fit_data)
    sw, sA = w * se[1], A * se[2]
    sc0, sc1, sc2, sc3 = se[3], se[4], se[5], se[6]

    model_on_z = model(z, p̂)

    # return a callable that evaluates on an arbitrary grid using the in-place kernel
    modelfun = x -> model(x, p̂)

    return fit_data,
           (w = w, A = A, c0 = c₀, c1 = c₁, c2 = c₂, c3 = c₃),
           (w = sw, A = sA, c0 = sc0, c1 = sc1, c2 = sc2, c3 = sc3),
           modelfun,
           model_on_z,
           (evals = calls[],
            best_probe = (rss = best[].rss,
                          w = exp(best[].p[1]), A = exp(best[].p[2]),
                          c0 = best[].p[3], c1 = best[].p[4], c2 = best[].p[5], c3 = best[].p[6]))
end

@inline function background_poly(z, c::AbstractVector{<:Real})
    @assert length(c) == 4
    ((c[4] .* z .+ c[3]) .* z .+ c[2]) .* z .+ c[1]
end

# Select experimental data
wanted_data_dir = "20250919" ;
wanted_binning  = 2 ; 
wanted_smooth   = 0.01 ;

# Data loading
read_exp_info = DataReading.find_report_data(
        joinpath(@__DIR__, "analysis_data");
        wanted_data_dir=wanted_data_dir,
        wanted_binning=wanted_binning,
        wanted_smooth=wanted_smooth
);
[(String(k), getfield(read_exp_info, k)) for k in propertynames(read_exp_info)];
if isnothing(read_exp_info)
    @warn "No matching report found"
else
    @info "Imported experimental data" "Directory\t\t" = read_exp_info.directory "Path\t\t" = read_exp_info.path "Date label\t\t"  = read_exp_info.data_dir "Analysis label\t" = read_exp_info.name "Binning\t\t" = read_exp_info.binning "Smoothing\t\t" =read_exp_info.smoothing "Magnitfication\t" =read_exp_info.magnification
    # I_exp = sort(read_exp_info.currents_mA / 1_000);
    # z_exp = read_exp_info.framewise_mm/read_exp_info.magnification;
end

exp_data = load(joinpath(read_exp_info.directory,"profiles_mean.jld2"))["profiles"]
Ic_sampled = exp_data[:Icoils];

valid_currents = sort(unique([firstindex(Ic_sampled);
        @view(findall(>(0.020), Ic_sampled)[1:10:end]);
        @view(findall(>(0.020), Ic_sampled)[end-2:end]);
        lastindex(Ic_sampled)
        ]
));

println("Target currents in A: (", join(map(x -> @sprintf("%.3f", x), Ic_sampled[valid_currents]), ", "),")")

norm_mode = :none;
range_z   = 7.8;
nrange_z  = 20001;
λ0_exp    = 0.001;
z_exp   = exp_data[:z_mm] .- exp_data[:Centroid_mm][1];
z_theory  = collect(range(-range_z,range_z,length=nrange_z));
μz, σz = mean(z_theory) , std(z_theory);
@assert isapprox(μz, 0.0; atol= 10 * eps(float(range_z)) ) "μz=$(μz) not ~ 0 within atol=$(10 * eps(float(range_z)) )"
@assert isapprox(σz, std_sample(range_z, nrange_z); atol= eps(float(range_z))) "σz=$(σz) is not defined for a symmetric range"
tpoly = Polynomial([0, 1/σz]) ;       # t = (-μ/σ) + (1/σ) z

fitting_params = zeros(length(valid_currents),6);

@time for (j,i_idx) in enumerate(valid_currents)

    I0 = Ic_sampled[i_idx]
    println("\n\t\tANALYZING BACKGROUND FOR I₀=$(round(1000*I0,digits=3))mA")
    𝒢  = TheoreticalSimulation.GvsI(I0)
    ℬ = abs.(TheoreticalSimulation.BvsI(I0))
    μ_eff = [TheoreticalSimulation.μF_effective(I0,v[1],v[2],K39_params) for v in TheoreticalSimulation.fmf_levels(K39_params,Fsel=1)]

    amp_exp = exp_data[:F1_profile][i_idx,:]

    Spl_exp = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp; weights=TheoreticalSimulation.compute_weights(z_exp, λ0_exp));

    pdf_exp = Spl_exp.(z_theory)
    pdf_exp = normalize_vec(pdf_exp; by=norm_mode)

    pdf_theory = reduce(+,[TheoreticalSimulation.getProbDist_v3(μ, 𝒢, 1e-3*z_theory, K39_params, effusion_params) for μ in μ_eff])
    pdf_theory = normalize_vec(pdf_theory;by=norm_mode)

    fig1= plot(z_exp , amp_exp, 
        label="Experiment (raw)", 
        seriestype=:scatter, 
        marker=(:hexagon,:white,2),
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        xlims=(-8,8),
    );
    fig2 = plot(z_theory, pdf_exp, 
        label="Experiment (spl. fit | $(norm_mode))", 
        line=(:black,2),
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        xlims=(-8,8),);
    plot!(z_theory , pdf_theory, label="Closed-form | $(norm_mode)", line=(:red,1.5));
    plot!(z_theory, TheoreticalSimulation.ProbDist_convolved(z_theory, pdf_theory, 150e-3), label="Closed-form + Conv | $(norm_mode)", line=(:dodgerblue2,1.2));

    fig=plot(fig1,fig2,
        layout=(2,1))
    display(fig)

    if j == 1
        w0, A0, c0  = 409.417e-3, 0.63, [0.529, -0.0143, -0.0984, 0.0118];
    else

        w0 = fitting_params[j-1,1]
        A0 = fitting_params[j-1,2]
        c0 = [fitting_params[j-1,v] for v in 3:6]
    end

    @time fit_data, params, δparams, modelfun, model_on_z , progress =
        fit_pdf(z_theory, pdf_exp, pdf_theory; w0=w0, A0=A0, c0=c0);

    bg_poly = params.c0 + params.c1*tpoly + params.c2*tpoly^2 + params.c3*tpoly^3

    fig=plot(z_theory , pdf_exp, 
        label="Experiment", 
        xlabel=L"$z$ (mm)",
        ylabel="Intensity (au)",
        seriestype=:scatter, 
        marker=(:hexagon,:white,1),
        markerstrokewidth=0.5,
        legend=:topleft,
        legendtitle=L"$I_{0}=%$(round(1000*I0,digits=3))\mathrm{mA}$",
        legendtitlefontsize=8,
        legendfontsize=8,);
    plot!(z_theory, pdf_theory, 
        label="ClosedForm",
        line=(:purple3,1) );
    plot!(z_theory, TheoreticalSimulation.ProbDist_convolved(z_theory, pdf_theory, params.w), 
        label="ClosedForm+Conv",
        line=(:dodgerblue2,1.2));
    plot!(z_theory,modelfun(z_theory), 
        label=L"Fit: $A f(I_{c},w;z) + P_{3}(z)$", 
        line=(:red,:dash,2),);
    plot!(z_theory,bg_poly.(z_theory),
        label="Background",
        line=(:green4,:dash,1.5));
    display(fig)
    savefig(fig,joinpath(OUTDIR,"$(wanted_data_dir)_$(@sprintf("%02d", i_idx))_$(string(norm_mode)).$(FIG_EXT)"))

    fitting_params[j,:]  = vcat(params.A,params.w,[bg_poly[dg] for dg in 0:3])

    pretty_table(
        fitting_params;
        column_label_alignment      = :c,
        column_labels               = [[MultiColumn(2, "Theoretical PDF"), MultiColumn(4, "Background P₃(z)") ],
                                        ["A", "w [mm]", "c₀", "c₁", "c₂", "c₃"]],
        row_labels                  = round.(1000*Ic_sampled[valid_currents], sigdigits=4),
        formatters                  = [fmt__printf("%8.5f", 1:2), fmt__printf("%8.4f", [3]), fmt__printf("%8.5e", 4:6)],
        alignment                   = :c,
        equal_data_column_widths    = true,
        stubhead_label              = "I₀ [mA]",
        row_label_column_alignment  = :c,
        title                       = "FITTING ANALYSIS",
        table_format                = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style                       = TextTableStyle(
                                            first_line_merged_column_label = crayon"light_red bold",
                                            first_line_column_label = crayon"yellow bold",
                                            column_label  = crayon"yellow",
                                            table_border  = crayon"blue bold",
                                            title = crayon"red bold"
                                        )
    )

end

pretty_table(
    fitting_params;
    column_label_alignment      = :c,
    column_labels               = [[MultiColumn(2, "Theoretical PDF"), MultiColumn(4, "Background P₃(z)") ],
                                    ["A", "w [mm]", "c₀", "c₁", "c₂", "c₃"]],
    row_labels                  = round.(1000*Ic_sampled[valid_currents], sigdigits=4),
    formatters                  = [fmt__printf("%8.5f", 1:2), fmt__printf("%8.4f", [3]), fmt__printf("%8.5e", 4:6)],
    alignment                   = :c,
    equal_data_column_widths    = true,
    stubhead_label              = "I₀ [mA]",
    row_label_column_alignment  = :c,
    title                       = "FITTING ANALYSIS",
    table_format                = TextTableFormat(borders = text_table_borders__unicode_rounded),
    style                       = TextTableStyle(
                                        first_line_merged_column_label = crayon"light_red bold",
                                        first_line_column_label = crayon"yellow bold",
                                        column_label  = crayon"yellow",
                                        table_border  = crayon"blue bold",
                                        title = crayon"red bold"
                                    )
)

cols = palette(:darkrainbow, length(valid_currents))
plot(xlabel=L"$z$ (mm)",
    ylabel="Intensity (au)")
for (i,idx) in enumerate(valid_currents)
    val_mA = 1000 * Ic_sampled[idx]

    plot!(z_theory,background_poly(z_theory, @view fitting_params[i, 3:6]),
        line=(cols[i],2),
        label= L"$I_{0}=" * @sprintf("%.1f", val_mA) * L"\,\mathrm{mA}$",
        )
end
plot!(legend=:best,)

plot(xlabel=L"$z$ (mm)",
    ylabel="Intensity (au)")
for (j,i_idx) in enumerate(valid_currents)
    amp_exp = exp_data[:F1_profile][i_idx,:]
    Spl_exp = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp; weights=TheoreticalSimulation.compute_weights(z_exp, λ0_exp));
    pdf_exp = Spl_exp.(z_theory)
    pdf_exp = normalize_vec(pdf_exp; by=norm_mode)
    val_mA = 1000 * Ic_sampled[i_idx]
    plot!(z_theory, pdf_exp,
        line=(:solid,cols[j],2),
        label= L"$I_{0}=" * @sprintf("%.1f", val_mA) * L"\,\mathrm{mA}$",)
    

    plot!(z_theory,background_poly(z_theory, @view fitting_params[j, 3:6]),
        line=(:dash,cols[j],1.5),
        label= false,
        )
end
plot!(legend=:best,)
savefig(fig,joinpath(OUTDIR,"summary_$(wanted_data_dir)_$(string(norm_mode)).$(FIG_EXT)"))




jldsave( joinpath(OUTDIR,"fitting_params_$(wanted_data_dir)_$(string(norm_mode)).jld2"), data = fitting_params)

aa = load(joinpath(OUTDIR,"fitting_params_20250919_max.jld2"))["data"]


plot(Ic_sampled[valid_currents], fitting_params[:,1])
plot(Ic_sampled[valid_currents], fitting_params[:,2])
plot(Ic_sampled[valid_currents], fitting_params[:,3])
plot(Ic_sampled[valid_currents], fitting_params[:,4])
plot(Ic_sampled[valid_currents], fitting_params[:,5])
plot(Ic_sampled[valid_currents], fitting_params[:,6])


# [TheoreticalSimulation.μF_effective(I0,v[1],v[2],K39_params) for v in TheoreticalSimulation.fmf_levels(K39_params,Fsel=2)][end]
