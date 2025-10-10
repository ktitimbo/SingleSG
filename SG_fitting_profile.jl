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
using Interpolations, Roots, Loess, Optim
using BSplineKit
using Polynomials
using DSP

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

# normalize_vec(v) = (m = maximum(v); m == 0 ? v : v ./ m);
normalize_vec(v; by::Symbol = :max, atol = 0) = begin
    denom = by === :max  ? maximum(v) :
            by === :sum  ? sum(v)      :
            by === :none ? 1           :
            throw(ArgumentError("by must be :max, :sum, or :none"))
    (by === :none || abs(denom) ≤ atol) ? v : v ./ denom
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

i_idx = 26

I0 = exp_data[:Icoils][i_idx]
𝒢  = TheoreticalSimulation.GvsI(I0)
ℬ = TheoreticalSimulation.BvsI(I0)
μ_eff = TheoreticalSimulation.μF_effective(I0,1,1,K39_params)

z_exp = exp_data[:z_mm] .- exp_data[:Centroid_mm][1]
amp_exp = normalize_vec(exp_data[:F1_profile][i_idx,:]; by=:none)
λ0_exp = 0.001
Spl_exp = BSplineKit.fit(BSplineOrder(4), z_exp, amp_exp, λ0_exp; weights=TheoreticalSimulation.compute_weights(z_exp, λ0_exp));

z_theory = collect(range(-7.80,7.80,length=20001))
pdf_exp = Spl_exp.(z_theory)
pdf_theory = TheoreticalSimulation.getProbDist_v3(μ_eff, 𝒢, 1e-3*z_theory, K39_params, effusion_params)
pdf_theory = normalize_vec(pdf_theory;by=:none)

plot(z_exp , amp_exp, label="Experiment", seriestype=:scatter, marker=(:hexagon,:white,2))
plot!(z_theory, pdf_exp, label="Experiment (spline fitting)", lw=2,)
plot!(z_theory , pdf_theory, label="Closed-form")
plot!(z_theory, TheoreticalSimulation.ProbDist_convolved(z_theory, pdf_theory, 150e-3), label="Closed-form + Conv")

using LsqFit


function fit_pdf_mix(z::AbstractVector, pdf_exp::AbstractVector, pdf_theory::AbstractVector;
                     ww0::Float64, A0::Float64=1.0, 
                     c0::AbstractVector=[0.0, 0.0, 0.0, 0.0],
                     normalize::Bool=false,
                     progress_every::Int=5,)

    @assert length(z) == length(pdf_exp) == length(pdf_theory)
    μz = mean(z); σz = std(z); @assert σz > 0 "z has zero variance"

    conv0 = TheoreticalSimulation.ProbDist_convolved(z, pdf_theory, ww0)
    conv_scale, poly_scale = 1.0, 1.0
    if normalize
        s_exp   = sum(pdf_exp)
        s_conv0 = max(sum(conv0), eps())
        conv_scale = s_exp / s_conv0
        poly_scale = s_exp / length(z)
    end

    # tiny helper so printing works even when numbers are Duals
    toflt(x) = try
        Float64(x)
    catch
        try getfield(Main, :ForwardDiff).value(x) |> Float64 catch
            try getfield(x, :value) |> Float64 catch; NaN end
        end
    end

    # --- move p0 BEFORE progress/bookkeeping so we know param length ---
    p0 = [log(float(ww0)), log(float(A0)),
          float(c0[1]), float(c0[2]), float(c0[3]), float(c0[4])]

    calls = Ref(0)
    # initialize 'best' with a Float64 vector of the right length
    best  = Ref((rss=Inf, p=copy(p0)))  # p::Vector{Float64}

    # LsqFit expects model(x, p)
    function pdfmix_model(zz::AbstractVector{<:Real}, p::AbstractVector{<:Real})
        logw, logA, c₀, c₁, c₂, c₃ = p
        A, w = exp(logA), exp(logw)

        tt = (zz .- μz) ./ σz
        conv = TheoreticalSimulation.ProbDist_convolved(zz, pdf_theory, w) .* conv_scale
        poly = (c₀ .+ c₁ .* tt .+ c₂ .* (tt.^2) .+ c₃ .* (tt.^3)) .* poly_scale

        yhat = @. A*conv + poly

        if progress_every > 0
            calls[] += 1
            if calls[] % progress_every == 0
                # make numeric copies for printing / best-tracking
                rss_val = toflt(sum(abs2, yhat .- pdf_exp))
                p_val   = map(toflt, p)                 # Vector{Float64}
                if rss_val < best[].rss
                    best[] = (rss=rss_val, p=p_val)     # matches best’s type
                end
                @printf(stderr,
                        "eval %6d | rss≈%.6g | w≈%.6g  A≈%.6g  c≈(%.3g, %.3g, %.3g, %.3g)\n",
                        calls[], rss_val, exp(p_val[1]), exp(p_val[2]), exp(p_val[3]), 
                        p_val[4], p_val[5], p_val[6])
            end
        end

        return yhat
    end

    fit_data = LsqFit.curve_fit(pdfmix_model, z, pdf_exp, p0; autodiff=:forward)

    logw, logA, c₀, c₁, c₂, c₃ = coef(fit_data)
    A, w = exp(logA), exp(logw)

    se = stderror(fit_data)
    sw,  sA = w*se[1], A*se[2]
    sc0, sc1, sc2, sc3 = se[3], se[4], se[5], se[6]

    model_on_z = pdfmix_model(z, coef(fit_data))

    return fit_data,
           (w=w, A=A, c0=c₀, c1=c₁, c2=c₂, c3=c₃),
           (w=sw, A=sA, c0=sc0, c1=sc1, c2=sc2, c3=sc3),
           (x -> pdfmix_model(x, coef(fit_data))),
           model_on_z,
           (evals=calls[],
            best_probe=(rss=best[].rss,
                        w=exp(best[].p[1]), A=exp(best[].p[2]),  
                        c0=best[].p[3], c1=best[].p[4], c2=best[].p[5], c3=best[].p[6]))
end



# --- 3) Fit with reasonable initial guesses ---
ww0 = 409e-3;
A0  = 0.043;
c0  = [0.012, -0.003, -0.0022, 0.00027];

fit_data, params, δparams, modelfun, model_on_z =
    fit_pdf_mix(z_theory, pdf_exp, pdf_theory; ww0=ww0, A0=A0, c0=c0);

println(params)
println(δparams)


plot(z_exp , amp_exp, label="Experiment", seriestype=:scatter, marker=(:hexagon,:white,2))
plot!(z_theory,modelfun(z_theory), label="Fit", line=(:red,:dash,2))

fit_data