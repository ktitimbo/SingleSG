# Kelvin Titimbo
# California Institute of Technology
# March 2026

############## EXPERIMENT ANALYSIS PREAMBLE ##############
# Plotting backend and general appearance settings
using Plots; gr()
# Set default plot aesthetics
const IN_NOTEBOOK = isdefined(Main, :IJulia);
Plots.default(
    show=IN_NOTEBOOK, dpi=800, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
# Data I/O and numerical tools
using MAT, JLD2
using LinearAlgebra
using ImageFiltering, FFTW
using DataStructures
using Statistics, StatsBase
using Interpolations, BSplineKit, Optim, Dierckx, LsqFit
# Aesthetics and output formatting
using Colors, ColorSchemes
using Printf, LaTeXStrings, PrettyTables
using CSV, DataFrames, DelimitedFiles
# Time-stamping/logging
using Dates
using Alert
# Multithreading setup
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(2)
@info "BLAS threads" count = BLAS.get_num_threads()
@info "Julia threads" count = Threads.nthreads()
# Custom modules
include("./Modules/DataReading.jl")
include("./Modules/JLD2_MyTools.jl");
include("./Modules/TheoreticalSimulation.jl")
include("./Modules/MyExperimentalAnalysis.jl");
using .MyExperimentalAnalysis;
# Set the working directory to the current location
cd(@__DIR__) ;
# General setup
const BASE_PATH = raw"F:\SternGerlachExperiments";
hostname = gethostname();
@info "Running on host" hostname=hostname
# For Plots
FIG_EXT  = "png"   # could be "pdf", "svg", etc.
SAVE_FIG = false
MyExperimentalAnalysis.SAVE_FIG = SAVE_FIG;
MyExperimentalAnalysis.FIG_EXT  = FIG_EXT;
# STERN–GERLACH EXPERIMENT SETUP
# Camera and pixel geometry : intrinsic properties
cam_pixelsize           = 6.5e-6 ;  # Physical pixel size of camera [m]
nx_pixels , nz_pixels   = (2160, 2560); # (Nx,Nz) pixels
# Experiment resolution
exp_bin_x, exp_bin_z    = (4,1) ;  # Camera binning
exp_pixelsize_x, exp_pixelsize_z = (exp_bin_x, exp_bin_z).*cam_pixelsize ; # Effective pixel size after binning [m]
# Image dimensions (adjusted for binning)
x_pixels = Int(nx_pixels / exp_bin_x);  # Number of x-pixels after binning
z_pixels = Int(nz_pixels / exp_bin_z);  # Number of z-pixels after binning
# Spatial axes shifted to center the pixels
x_position = pixel_positions(x_pixels, 1, exp_pixelsize_x);
z_position = pixel_positions(z_pixels, 1, exp_pixelsize_z);
println("""
***************************************************
CAMERA FEATURES
    Number of pixels        : $(nx_pixels) × $(nz_pixels)
    Pixel size              : $(1e6*cam_pixelsize) μm

IMAGES INFORMATION
    Binning                 : $(exp_bin_x) × $(exp_bin_z)
    Effective pixels        : $(x_pixels) × $(z_pixels)
    Pixel size              : $(1e6*exp_pixelsize_x)μm × $(1e6*exp_pixelsize_z)μm
    xlims                   : ($(round(minimum(1e6*x_position), digits=6)) μm, $(round(maximum(1e3*x_position), digits=4)) mm)
    zlims                   : ($(round(minimum(1e6*z_position), digits=6)) μm, $(round(maximum(1e3*z_position), digits=4)) mm)
***************************************************
""")
# Setting the variables for the module
MyExperimentalAnalysis.effective_cam_pixelsize_z    = exp_pixelsize_z;
MyExperimentalAnalysis.DEFAULT_x_pixels             = x_pixels;
MyExperimentalAnalysis.DEFAULT_z_pixels             = z_pixels;

data_2025 = CSV.read(joinpath(@__DIR__,"data_studies","FITki20260519T160646328","data_exp.csv"), DataFrame)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
T_START   = Dates.now()
RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSSsss");


K39_params = TheoreticalSimulation.AtomParams("39K"); # [R μn γn Ispin Ahfs M ] 
nz = 1 ;
λ0 = 0.01 ;
σw = 0.200;
ki = 2.3;

# position
z_mm        = 1e3 .* pixel_positions(z_pixels, nz, exp_pixelsize_z) 
z_mm_error  = 1/sqrt(12) * 1e3 * exp_pixelsize_z * nz ;

## #################################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
d_qm_f1 = jldopen(joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","QM_T205_8M","qm_screen_profiles_f1_table.jld2"),"r") do f
    select_key = JLD2_MyTools.make_keypath_qm(nz,0.200,λ0)
    data = f[select_key]
    nI = length(keys(data))
    return (;
        Ic = [data[x][:Icoil] for x=1:nI], 
        F1 = [data[x][:z_max_smooth_mm] for x=1:nI], 
        F1s = [data[x][:z_max_smooth_spline_mm] for x=1:nI] 
        )
end

d_qm_f2 = jldopen(joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","QM_T205_8M","qm_screen_profiles_f2_table.jld2"),"r") do f
    select_key = JLD2_MyTools.make_keypath_qm(nz,0.200,λ0)
    data = f[select_key]
    nI = length(keys(data))
    return (;
        Ic = [data[x][:Icoil] for x=1:nI], 
        F2 = [data[x][:z_max_smooth_mm] for x=1:nI], 
        F2s = [data[x][:z_max_smooth_spline_mm] for x=1:nI] )
end

profiles_1    = TheoreticalSimulation.QM_analyze_profiles_to_dict(joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","QM_T205_8M","qm_screen_data.jld2"), K39_params;
                    manifold=1,    n_bins= (32 , nz), width_mm=σw, add_plot=false, plot_xrange=:all, λ_raw=λ0, λ_smooth = 0.001, mode=:probability);
profiles_5    = TheoreticalSimulation.QM_analyze_profiles_to_dict(joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","QM_T205_8M","qm_screen_data.jld2"), K39_params;
                    manifold=5,    n_bins= (32 , nz), width_mm=σw, add_plot=false, plot_xrange=:all, λ_raw=λ0, λ_smooth = 0.001, mode=:probability);

d_qm_lvl1 = [profiles_1[x][:z_max_smooth_spline_mm] for x=1:47]
d_qm_lvl5 = [profiles_5[x][:z_max_smooth_spline_mm] for x=1:47]

d_cqd_up = jldopen(joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T205_7M","cqd_7M_up_profiles.jld2"),"r") do f
    select_key = JLD2_MyTools.make_keypath_cqd(:up,ki,nz,0.200,λ0)
    data = f[select_key]
    nI = length(keys(data))
    return (;
        Ic = [data[x][:Icoil] for x=1:nI], 
        up = [data[x][:z_max_smooth_mm] for x=1:nI], 
        ups = [data[x][:z_max_smooth_spline_mm] for x=1:nI] )
end

d_cqd_dw = jldopen(joinpath(BASE_PATH,"SIMULATIONS","2025_SETUP","CQD_T205_7M","cqd_7M_dw_profiles.jld2"),"r") do f
    select_key = JLD2_MyTools.make_keypath_cqd(:dw,ki,nz,0.200,λ0)
    data = f[select_key]
    nI = length(keys(data))
    return (;
        Ic = [data[x][:Icoil] for x=1:nI], 
        dw = [data[x][:z_max_smooth_mm] for x=1:nI], 
        dws = [data[x][:z_max_smooth_spline_mm] for x=1:nI] )
end

Icoils = d_cqd_up.Ic

QM_Δz = d_qm_f1.F1s .- d_qm_f2.F2s

nocollapse_Δz = d_qm_lvl5 .- d_qm_lvl1
CQD_Δz = d_cqd_up.ups .- d_cqd_dw.dws

"""
    interpolate_and_evaluate(x_list, y_list, x_exp)
 
Interpolates the data given by `x_list` and `y_list` using a cubic spline,
then evaluates the interpolated function at each point in `x_exp`.
 
# Arguments
- `x_list::AbstractVector{<:Real}`: Known x data points (must be strictly increasing).
- `y_list::AbstractVector{<:Real}`: Known y data points corresponding to `x_list`.
- `x_exp::AbstractVector{<:Real}`: Query points at which to evaluate the interpolant.
 
# Returns
- `Vector{Float64}`: Interpolated y values at each point in `x_exp`.
 
# Notes
- Uses cubic spline interpolation (k=3) via Dierckx.jl.
- s=0 forces the spline to pass exactly through every data point.
- `x_list` must be sorted in strictly increasing order.
"""
function interpolate_and_evaluate(
    x_list::AbstractVector{<:Real},
    y_list::AbstractVector{<:Real},
    x_exp::AbstractVector{<:Real}
)
    length(x_list) == length(y_list) ||
        throw(ArgumentError("x_list and y_list must have the same length."))
    issorted(x_list; lt = <) ||
        throw(ArgumentError("x_list must be strictly increasing."))
 
    # Build a cubic spline (k=3); s=0 forces the spline through every data point
    spl = Spline1D(Float64.(x_list), Float64.(y_list); k=3, s=0)
 
    # Broadcast call syntax works without needing to import evaluate explicitly
    return spl.(Float64.(x_exp))
end

"""
    find_spline_maximum(CNR; n_grid=10_000, log_x=false)
 
Given a two-column matrix `CNR` where column 1 is x (e.g. Ic) and column 2 is y,
fits a cubic spline and finds the x and y values at the global maximum.
 
Strategy:
  1. Evaluate the spline on a fine grid (n_grid points) to locate the global peak.
  2. Refine the best grid point with BFGS to get a precise answer.
 
# Keyword arguments
- `n_grid::Int=10_000`: number of points in the coarse search grid.
- `log_x::Bool=false`: if true, the coarse grid is spaced logarithmically (useful when
  x spans several decades, as with current sweeps on a log scale).
 
# Returns
- `(x_max, y_max)`: x location of the maximum and the corresponding spline value.
"""
function find_spline_maximum(
    CNR::AbstractMatrix{<:Real};
    n_grid::Int = 10_000,
    log_x::Bool = false
)
    x = Float64.(CNR[:, 1])
    y = Float64.(CNR[:, 2])
 
    # Sort by x in case data is unordered
    idx = sortperm(x)
    x, y = x[idx], y[idx]
 
    spl = Spline1D(x, y; k=3, s=0)
 
    # 1. Coarse grid search — log-spaced if requested (matches log-scale plots)
    x_grid = log_x ? exp.(range(log(x[1]), log(x[end]); length=n_grid)) :
                     range(x[1], x[end]; length=n_grid)
    y_grid = spl.(x_grid)
    x0 = x_grid[argmax(y_grid)]   # best coarse estimate
 
    # 2. Refine with BFGS starting from the coarse peak
    result = optimize(t -> -spl(t[1]), [x0], BFGS(),
                      Optim.Options(; x_abstol=1e-12))
 
    x_max = clamp(Optim.minimizer(result)[1], x[1], x[end])
    y_max = spl(x_max)
 
    return x_max, y_max
end
 
QM_Δz_on_exp         = interpolate_and_evaluate(Icoils, QM_Δz, data_2025.Ic)
nocollapse_Δz_on_exp = interpolate_and_evaluate(Icoils, nocollapse_Δz, data_2025.Ic)
CQD_Δz_on_exp        = interpolate_and_evaluate(Icoils, CQD_Δz, data_2025.Ic)

CNR = hcat(data_2025.Ic, abs.((nocollapse_Δz_on_exp .- CQD_Δz_on_exp)) ./ data_2025.szmax )
cnr_x_max, cnr_y_max = find_spline_maximum(CNR; log_x=true)

fig_CNR = plot(CNR[:,1], CNR[:,2],
    label=L"$\mathrm{No \ collapse}-\mathrm{CQD}(%$(ki)\times10^{-6}$)",
    line=(:solid,2,:red),
    marker=(:circle,2,:white),
    markerstrokecolor=:red,
    xlabel="Current (A)",
    ylabel="CNR",
    xticks = ([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], 
            [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    # yticks = ([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], 
    #         [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    xscale=:log10,
    # yscale=:log10,
    legend=:topright,
    foreground_color_legend=nothing,
    background_color_legend=nothing,
)
vline!(fig_CNR, [cnr_x_max], label=L"$I_{c}=%$(round(1000*cnr_x_max; digits=2))\mathrm{mA}$ ", line=(:dash,1,:black) )


Ithreshold = 0.025
mask = d_qm_f1.Ic .>= Ithreshold

plot(TheoreticalSimulation.BvsI.(Icoils[mask]), QM_Δz[mask]./(1e3*cam_pixelsize), label="QM",
line=(:solid,2,:red))
plot!(TheoreticalSimulation.BvsI.(Icoils[mask]), CQD_Δz[mask]./(1e3*cam_pixelsize), label="CQD",
line=(:solid,2,:blue))
plot!(TheoreticalSimulation.BvsI.(Icoils[mask]), nocollapse_Δz[mask]./(1e3*cam_pixelsize), label="CQD - No collapse",
line=(:solid,2,:green))
plot!(
    ylabel=L"$\Delta z$ (px)",
    xlabel="Magnetic field (T)",
    xlims=(0.001,1),
    ylims=(0.1,1000),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], 
            [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    xscale=:log10,
    yscale=:log10,
    legend=:topleft,
    foreground_color_legend=nothing,
    background_color_legend=nothing,
)

fig_z = plot(Icoils[mask], QM_Δz[mask]./(1e3*cam_pixelsize), label="QM",
line=(:solid,2,:red))
plot!(fig_z, Icoils[mask], CQD_Δz[mask]./(1e3*cam_pixelsize), label="CQD",
line=(:solid,2,:blue))
plot!(fig_z, Icoils[mask], nocollapse_Δz[mask]./(1e3*cam_pixelsize), label="CQD - No collapse",
line=(:solid,2,:green))
plot!(fig_z,
    ylabel=L"$\Delta z$ (px)",
    xlabel="Current (A)",
    xlims=(Ithreshold,1),
    ylims=(10,1000),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], 
            [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}", L"10^{2}"]),
    yticks = ([10.0, 100.0], 
            [ L"10^{1}", L"10^{2}"]),
    xscale=:log10,
    yscale=:log10,
    legend=:topleft,
    foreground_color_legend=nothing,
    background_color_legend=nothing)
vline!(fig_z, [cnr_x_max], label=L"$I_{c}=%$(round(1000*cnr_x_max; digits=2))\mathrm{mA}$ ", line=(:dash,1,:black) )

plot(fig_z, fig_CNR,
    layout=(2,1))



function find_optimal_I(Ic, QM, CQD; 
                        Ithreshold = 0.020)

    # ── discrete case (your existing approach) ────────────────────────────
    mask     = Ic .> Ithreshold
    Ic_m     = Ic[mask]; QM_m = QM[mask]; CQD_m = CQD[mask]

    log_diff = log10.(QM_m) .- log10.(CQD_m)
    imax     = argmax(log_diff)
    I_opt_discrete = Ic_m[imax]
    δ_discrete     = QM_m[imax] - CQD_m[imax]

    @info "Discrete optimum" Ic_A=I_opt_discrete  δ=round(δ_discrete; sigdigits=3)

    # ── continuous case (interpolation + optimization) ────────────────────
    # Work in log10(I) space so the interpolation is uniform
    logI    = log10.(Ic_m)
    itp_QM  = linear_interpolation(logI, log10.(QM_m))
    itp_CQD = linear_interpolation(logI, log10.(CQD_m))

    # Objective: negative log-diff (Optim minimizes)
    objective(logI_val) = -(itp_QM(logI_val[1]) - itp_CQD(logI_val[1]))

    result         = optimize(objective, [logI[1]], [logI[end]], [log10(I_opt_discrete)], Fminbox(BFGS()))
    I_opt_cont     = 10^Optim.minimizer(result)[1]
    δ_cont         = 10^itp_QM(log10(I_opt_cont)) - 10^itp_CQD(log10(I_opt_cont))

    @info "Continuous optimum" Ic_A=round(I_opt_cont; digits=3)  δ=round(δ_cont; sigdigits=3)

    return (;
        I_opt_discrete, δ_discrete,
        I_opt_cont,     δ_cont,
    )
end

res = find_optimal_I(d_qm_f1.Ic, QM_Δz, CQD_Δz; Ithreshold=Ithreshold);

plot(d_qm_f1.Ic[mask], QM_Δz[mask], label=L"QM: $F=2(m_{F}=2,1,0,-1) - F=1$",
    line=(:solid,2,:red))
plot!(d_qm_f1.Ic[mask], CQD_Δz[mask], label=L"CQD: $up - down$",
    line=(:solid,2,:blue))
vline!([res.I_opt_discrete], line=(:dash,1.2,:black), label=L"$I_{c}=%$(round(1000*res.I_opt_cont; sigdigits=3))\mathrm{mA}$ | $δ=%$(round(1000*res.δ_cont;sigdigits=3))\mathrm{\mu m}$")
plot!(
    xlabel="Current (A)",
    ylabel=L"$Δz \ (\mathrm{mm})$",
    xscale=:log10,
    yscale=:log10,
    ylims=(0.09,4),
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], 
            [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:topleft,
    foreground_color_legend=nothing,
    background_color_legend=nothing)




##xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX MARCH DATA XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
data_directories = [
        "20260318/Round1",
        "20260318/Round2",
        # "20260318/Round3",
        # "20260318/Round4",
        ];
nd = length(data_directories);

for data_directory in data_directories
    printstyled("\t" * data_directory * "\n"; color=:cyan, bold=true)
    outfile_raw         = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data.jld2")
    outfile_processed   = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2")
    data_summary_path   = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS","summary",data_directory)
    isdir(data_summary_path) || mkpath(data_summary_path);


    if !isfile(outfile_processed) # check if the processed images exists
        if !isfile(outfile_raw) # check if the raw data exists
            @info "Not found → building $outfile_raw"
            data_in = joinpath(BASE_PATH,"EXPERIMENTS", data_directory)
            data_raw = MyExperimentalAnalysis.SG0_stack_data(data_in)
            jldsave(outfile_raw, data=data_raw)
            data_raw = nothing
        else
            @info "Found $outfile_raw → skipping build"
        end

        data_raw = load(outfile_raw)["data"]
        data_processed = MyExperimentalAnalysis.SG0_build_processed_dict(data_raw)
        jldsave(outfile_processed, data=data_processed)
        data_processed = nothing
        data_raw = nothing
    else
        @info "Found $outfile_processed → skipping build"
    end

    println("\n$(data_directory) PROCESSING COMPLETED\n\n")
end



tables = Vector{DataFrame}(undef, nd);
jldopen(joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS", dirname.(data_directories[1]), "data_analysis_$(RUN_STAMP).jld2"), "w") do f
    f["meta/nz"]               = nz
    f["meta/λ0"]               = λ0
    f["meta/data_directories"] = data_directories

    for (idx,data_directory) in enumerate(data_directories)
        # idx = 2
        # data_directory = data_directories[idx]
        printstyled("\t" * data_directory * "\n"; color=:cyan, bold=true);
        data_processed = load(joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2"))["data"];

        OUTDIR    = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
        isdir(OUTDIR) || mkpath(OUTDIR);
        @info "Created output directory" OUTDIR
        MyExperimentalAnalysis.OUTDIR   = OUTDIR;

        # ── Currents & fields ────────────────────────────────────────────────────
        SG0_current = data_processed[:SG0Currents];
        SG1_current = data_processed[:SG1Currents];
        
        Bz0 = 1e3 .* data_processed[:SG0Bz];
        Bz1 = 1e3 .* data_processed[:SG1Bz];

        # ── Framewise maxima & statistics ────────────────────────────────────────
        f1_max = MyExperimentalAnalysis.SG0_framewise_maxima("F1", data_processed, nz ; half_max=false, λ0=λ0);
        f2_max = MyExperimentalAnalysis.SG0_framewise_maxima("F2", data_processed, nz ; half_max=false, λ0=λ0);

        f1_z_mm , f1_z_sem_mm  = vec(mean(f1_max, dims=1)) , sqrt.(vec(std(f1_max, dims=1; corrected=true) ./ sqrt(size(f1_max,1))).^2 .+ z_mm_error^2 );
        f2_z_mm , f2_z_sem_mm  = vec(mean(f2_max, dims=1)) , sqrt.(vec(std(f2_max, dims=1; corrected=true) ./ sqrt(size(f2_max,1))).^2 .+ z_mm_error^2 );

        Δz_mm = (f1_z_mm .- f2_z_mm);
        Δz_sem_mm = sqrt.( (f1_z_sem_mm).^2 .+ (f2_z_sem_mm).^2 );

        centroid_mm     = 0.5 * (f1_z_mm .+ f2_z_mm);
        centroid_sem_mm = 0.5 * sqrt.( (f1_z_sem_mm).^2 .+ (f2_z_sem_mm).^2 );

        # ── Summary DataFrame ────────────────────────────────────────────────────
        data = DataFrame(
            I0        = SG0_current,
            I1        = SG1_current,
            B0        = Bz0,
            B1        = Bz1,
            zf1       = f1_z_mm,
            errzf1    = f1_z_sem_mm,
            zf2       = f2_z_mm,
            errzf2    = f2_z_sem_mm,
            split     = Δz_mm,
            errsplit  = round.(Δz_sem_mm; sigdigits=2),
            c0        = centroid_mm,
            errc0     = round.(centroid_sem_mm; sigdigits=2)
        )
        tables[idx] = data
        f[data_directory] = data  # ← save immediately
        @info "Saved table" data_directory

        pretty_table(data;
                title         = data_directory,
                formatters    = [ fmt__printf("%8.5f", 1:2), fmt__printf("%8.3f", 3:4), fmt__printf("%8.3f", [5,7,9,11]), fmt__printf("%8.4f", [6,8,10,12])],
                alignment     = :c,
                column_labels  = [
                    ["I0 Current", "I1 Current", "B0 field", "B1 field", "F1", "F1 err", "F2", "F2 err", "Δz", "Δz err", "C₀", "C₀ err "], 
                    ["[A]", "[A]", "[mT]" ,"[mT]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
                ],
                table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
                style = TextTableStyle(
                            first_line_column_label = crayon"yellow bold",
                            column_label  = crayon"yellow",
                            table_border  = crayon"blue bold",
                            title = crayon"bold red"
                            ),
                equal_data_column_widths = true,
        )

        tol = 1e-9
        mask = (abs.(data.I0) .<= tol) .&& (abs.(data.I1) .<= tol)
        rows_zero = data[mask, :]
        mean_zero = DataFrame(
                    [name => mean(rows_zero[!, name]) for name in names(rows_zero)]
                )

        pretty_table(mean_zero;
                title         = data_directory,
                formatters    = [ fmt__printf("%8.5f", 1:2), fmt__printf("%8.3f", 3:4), fmt__printf("%8.3f", [5,7,9,11]), fmt__printf("%8.4f", [6,8,10,12])],
                alignment     = :c,
                column_labels  = [
                    ["I0 Current", "I1 Current", "B0 field", "B1 field", "F1", "F1 err", "F2", "F2 err", "Δz", "Δz err", "C₀", "C₀ err "], 
                    ["[A]", "[A]", "[mT]" ,"[mT]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
                ],
                table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
                style = TextTableStyle(
                            first_line_column_label = crayon"yellow bold",
                            column_label  = crayon"yellow",
                            table_border  = crayon"blue bold",
                            title = crayon"bold red"
                            ),
                equal_data_column_widths = true,
        )


        data_pos = data[data.I0 .> 0, :]
        # ── Colour palette (one colour per SG0 current step) ─────────────────────
        colors_sg0 = palette(:darkrainbow, size(data,1));
        sg1_label = "$(round(1000 * data.I1[end], digits=2))mA"   # reused in titles

        # ── Overlay of all SG0 currents: z-position vs current ───────────────────
        common_scatter_kw = (seriestype=:scatter, marker=(:circle, 2, :white))

        fig1 = plot(data.I0, data.zf1;
            common_scatter_kw..., yerror=data.errzf1,
            label=L"$F=1$", markerstrokecolor=:red);
        plot!(fig1, data.I0, data.zf2;
            common_scatter_kw..., yerror=data.errzf2,
            label=L"$F=2$", markerstrokecolor=:blue);
        plot!(fig1, data.I0, mean([data.zf1, data.zf2]);
            label="Centre", marker=(:diamond, :white),
            markerstrokecolor=:gray47, line=(:dash, 1, :gray47));
        plot!(fig1;
            xlabel="SG0 Current (A)",
            foreground_color_legend=nothing, background_color_legend=nothing,
            yformatter=y -> @sprintf("%.3f", y));

        fig2 = plot(data_pos.I0, data_pos.zf1;
            common_scatter_kw..., yerror=data_pos.errzf1,
            label=L"$F=1$", markerstrokecolor=:red);
        plot!(fig2, data_pos.I0, data_pos.zf2;
            common_scatter_kw..., yerror=data_pos.errzf2,
            label=L"$F=2$", markerstrokecolor=:blue);
        plot!(fig2, data_pos.I0, mean([data_pos.zf1, data_pos.zf2]),
            label="Centre", marker=(:diamond, :white),
            markerstrokecolor=:gray47, line=(:dash, 1, :gray47));
        plot!(fig2;
            xscale=:log10,
            foreground_color_legend=nothing, background_color_legend=nothing,
            xlabel="SG0 Current (A)",
            yformatter=y -> @sprintf("%.3f", y));

        fig3 = plot(data.I0, data.split ./ 6.5e-3;
            common_scatter_kw..., yerror=data.errsplit ./ 6.5e-3,
            label=L"$\Delta z$", markerstrokecolor=:darkgreen);
        plot!(fig3;
            foreground_color_legend=nothing, background_color_legend=nothing,
            xlabel="SG0 Current (A)", ylabel="Separation (px)", yminorticks=false);

        fig4 = plot(data_pos.I0, data_pos.split ./ 6.5e-3;
            common_scatter_kw..., yerror=data_pos.errsplit ./ 6.5e-3,
            label=L"$\Delta z$", markerstrokecolor=:darkgreen);
        plot!(fig4;
            xscale=:log10,
            foreground_color_legend=nothing, background_color_legend=nothing,
            xlabel="SG0 Current (A)", ylabel="Separation (px)", yminorticks=false);

        fig = plot(fig1, fig2, fig3, fig4;
            suptitle     = "$(data_directory) | SG1: $sg1_label",
            layout       = (2, 2),
            size         = (1000, 600),
            link         = :x,
            left_margin  = 5mm,
            bottom_margin= 3mm,
        )
        plot!(fig[1]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
        plot!(fig[2]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
        display(fig)
        
        # ── Extract image arrays once, outside all loops ─────────────────────────
        f1imgs = data_processed[:F1ProcessedImages];
        f2imgs = data_processed[:F2ProcessedImages];
        sg1_label    = "$(data_directory) | SG1: $(round(1000*data.I1[end], digits=2))mA";
        current_label(i) = L"$%$(round(1000*SG0_current[i]; digits=3))\mathrm{mA}$";

        legend_kw = (
            legend                  = :topleft,
            legend_title            = "SG0",
            background_color_legend = :white,
            foreground_color_legend = nothing,
        );

        # ── Pre-compute cleaned images and profiles for all currents ──────────────
        # Store so each quantity is computed once and reused in both plot blocks
        f1_images   = Vector{Matrix{Float64}}(undef, length(SG0_current));
        f2_images   = Vector{Matrix{Float64}}(undef, length(SG0_current));
        f1_profiles = Vector{Vector{Float64}}(undef, length(SG0_current));
        f2_profiles = Vector{Vector{Float64}}(undef, length(SG0_current));


        for i in eachindex(SG0_current)
            # Average over frames (dim 3), then mask
            f1img = dropdims(mean(f1imgs[:,:,:,i], dims=3), dims=3)
            f1img = mapwindow(median, f1img, (3, 3))
            # f1img .*= isfinite.(f1img) .& (f1img .>= -10) .& (f1img .<= 1000)

            f2img = dropdims(mean(f2imgs[:,:,:,i], dims=3), dims=3)
            # f2img .*= isfinite.(f2img) .& (f2img .>=  -5) .& (f2img .<= 500)

            f1_images[i]   = f1img
            f2_images[i]   = f2img

            # Profile: average over x (dim 1) after frame-averaging and masking
            f1_profiles[i] = vec(mean(f1img, dims=1))
            f2_profiles[i] = vec(mean(f2img, dims=1))
        end

        # ── Per-current panels: heatmap (left) + z-profile (right) ───────────────
        camera_z_mm = 1e3 .* pixel_positions(z_pixels, 1, exp_pixelsize_z) 
        for i in eachindex(SG0_current)
            f1vmax = Statistics.quantile(vec(f1_images[i]), 0.999)
            f2vmax = Statistics.quantile(vec(f2_images[i]), 0.999)

            plt1 = heatmap(f1_images[i];
                xlabel = L"$z\ \ (\mathrm{px})$",
                ylabel = L"$x\ \ (\mathrm{px})$",
                cbar   = true, clims = (0, f1vmax),
            )
            plt2 = plot(camera_z_mm, f1_profiles[i];
                line = (:solid, 1, colors_sg0[i]), label = current_label(i),
                xlabel = L"$z\ \ (\mathrm{px})$", legend_kw...,
            )
            vline!(plt2, [f1_max[i]], line = (:dot, 1, colors_sg0[i]), label = false)
            plt3 = heatmap(f2_images[i];
                xlabel = L"$z\ \ (\mathrm{px})$",
                ylabel = L"$x\ \ (\mathrm{px})$",
                cbar   = true, clims = (0, f2vmax),
            )
            plt4 = plot(camera_z_mm, f2_profiles[i];
                line = (:solid, 1, colors_sg0[i]), label = current_label(i),
                xlabel = L"$z\ \ (\mathrm{px})$", legend_kw...,
            )
            vline!(plt4, [f2_max[i]], line = (:dot, 1, colors_sg0[i]), label = false)

            plt = plot(plt1, plt2, plt3, plt4;
                suptitle      = "$(data_directory) | SG1: $(round(1000*data.I1[i], digits=2))mA",
                layout        = (2, 2),
                link          = :x,
                size          = (800, 450),
                left_margin   = 3mm,
                bottom_margin = 2mm,
            )
            plot!(plt[1]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
            plot!(plt[2]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
            display(plt)
        end

        # ── Overlay: all currents on one z-profile plot per state ─────────────────
        overlay_kw = (
            legend                  = :topleft,
            legend_font             = 8,
            foreground_color_legend = nothing,
            legend_columns          = 2,
            xlabel                  = L"$z \ \ (\mathrm{px})$",
        );

        plt_f1 = plot(; overlay_kw...);
        plt_f2 = plot(; overlay_kw...);

        for i in eachindex(SG0_current)
            plot!(plt_f1, f1_profiles[i]; line=(:solid, 1, colors_sg0[i]), label=current_label(i))
            plot!(plt_f2, f2_profiles[i]; line=(:solid, 1, colors_sg0[i]), label=current_label(i))
        end
        
        plot!(plt_f1; legend_title=L"$F=1$");
        plot!(plt_f2; legend_title=L"$F=2$");

        fig = plot(plt_f1, plt_f2;
            suptitle      = sg1_label,
            layout        = (2, 1),
            size          = (900, 850),
            left_margin   = 3mm,
            bottom_margin = 2mm,
        )
        plot!(fig[1]; xlabel="", xformatter=_->"", bottom_margin=-6mm)
        display(fig)

        GC.gc()
    end
end


scale_factor = 6.5e-3

mark_symbol      = [:circle, :rect]
colores          = (:darkgreen, :indigo)
current_polarity = ("+", "–")

tol = 1e-9

plot_specs = [
    (:split, :errsplit, L"$\Delta z$ (px)", scale_factor),
    (:c0,    :errc0,    L"$c_{0}$ (mm)",    1.0),
    (:zf1,   :errzf1,   L"$F=1$ (mm)",      1.0),
    (:zf2,   :errzf2,   L"$F=2$ (mm)",      1.0),
]

figs = []

for (ycol, errcol, ylabel_text, sf) in plot_specs

    fig = plot(
        xlabel = "SG0 Current (A)",
        ylabel = ylabel_text,
    )

    for (k, data) in enumerate(tables)

        data_pos = data[data.I0 .> 0, :]

        mask = isapprox.(data.I0, 0.0; atol=tol) .&& .!isapprox.(data.I1, 0.0; atol=tol)
        rows_zero = data[mask, :]
        rows_zero = DataFrame(
            [name => mean(rows_zero[!, name]) for name in names(rows_zero)]
        )

        plot!(fig,
            data_pos.I0,
            data_pos[!, ycol] ./ sf,
            yerror            = data_pos[!, errcol] ./ sf,
            label             = "($(current_polarity[k])) " * data_directories[k],
            line              = (:solid, 1, colores[k], 0.5),
            marker            = (mark_symbol[k], 2, :white),
            markerstrokecolor = colores[k],
        )
        y0  = only(rows_zero[!, ycol]) / sf
        dy0 = only(rows_zero[!, errcol]) / sf
        hline!(fig, [y0],
            label = false,
            line  = (:dash, 1, colores[k], 0.5),
        )
        hspan!(fig, [y0 - dy0, y0 + dy0],
            label     = false,
            color     = colores[k],
            fillalpha = 0.15,
            linealpha = 0,
        )
    end
    plot!(fig,
        xlims = (0.95e-3, 5.0),
        xticks = (
            [1e-3, 1e-2, 1e-1, 1.0],
            [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]
        ),
        xscale                  = :log10,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        yminorticks             = 5,
        legend                  = :topleft,
    )
    push!(figs, fig)
end
fig1a, fig1b, fig1c, fig1d = figs
xtick_vals   = [1e-3, 1e-2, 1e-1, 1.0]
xtick_labels = [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]
empty_labels = fill("", length(xtick_vals))
fig1 = plot(
    fig1a, fig1b, fig1c, fig1d,
    size   = (800, 600),
    layout = (2, 2),
    link   = :x,
)
# plot!(fig1[1], xlabel = "", xformatter = _ -> "", bottom_margin = -8mm)
# plot!(fig1[2], xlabel = "", xformatter = _ -> "", bottom_margin = -8mm)
plot!(fig1, subplot = 1,
    xlabel = "",
    xticks = (xtick_vals, empty_labels),
    bottom_margin = -5mm,
)
plot!(fig1, subplot = 2,
    xlabel = "",
    xticks = (xtick_vals, empty_labels),
    bottom_margin = -5mm,
)

plot!(fig1, subplot = 3,
    xticks = (xtick_vals, xtick_labels),
)

plot!(fig1, subplot = 4,
    xticks = (xtick_vals, xtick_labels),
)
display(fig1)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_directories = [
        # "20260617/Round1",
        # "20260617/Round2",
        # "20260618/Round1",
        # "20260618/Round2",
        # "20260622/Round1",
        # "20260622/Round2",
        # "20260623/Round1",
        # "20260623/Round2",
        # "20260624/Round1",
        # "20260624/Round2",
        # "20260626/Round1",
        # "20260626/Round2",
        "20260629/Round1",
        "20260629/Round2",
        "20260701/Round1",
        "20260701/Round2",
        "20260702/Round1",
        "20260702/Round2",
        "20260706/Round1",
        "20260706/Round2",
]
nd = length(data_directories);

for data_directory in data_directories
    printstyled("\t" * data_directory * "\n"; color=:cyan, bold=true)
    outfile_raw         = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data.jld2")
    outfile_processed   = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2")
    data_summary_path   = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS","summary",data_directory)
    isdir(data_summary_path) || mkpath(data_summary_path);


    if !isfile(outfile_processed) # check if the processed images exists
        if !isfile(outfile_raw) # check if the raw data exists
            @info "Not found → building $outfile_raw"
            data_in = joinpath(BASE_PATH,"EXPERIMENTS", data_directory)
            data_raw = MyExperimentalAnalysis.SG0_stack_data("new", data_in)
            jldsave(outfile_raw, data=data_raw)
            data_raw = nothing
        else
            @info "Found $outfile_raw → skipping build"
        end

        data_raw = load(outfile_raw)["data"]
        data_processed = MyExperimentalAnalysis.SG0_process_and_save(data_raw,outfile_processed)
        # jldsave(outfile_processed, data=data_processed)
        data_processed = nothing
        data_raw = nothing
    else
        @info "Found $outfile_processed → skipping build"
    end

    println("\n$(data_directory) PROCESSING COMPLETED\n\n")
end


# Group directories by their common root (the date)
grouped = OrderedDict{String, Vector{String}}()
for d in data_directories
    root = first(splitpath(d))  # "20260617/Round1" -> "20260617"
    push!(get!(grouped, root, String[]), d)
end

tables = Vector{DataFrame}(undef, nd)  # nd = total number of round-directories (flat count)
flat_idx = Ref(0)  # global counter across all groups
for (dirname, dirs_in_group) in grouped
    printstyled("== $dirname ==\n"; color=:yellow, bold=true)

    out_path = joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS", dirname, "data_analysis_$(RUN_STAMP).jld2")
    mkpath(Base.dirname(out_path))
    jldopen(out_path, "w") do f
        f["meta/nz"]               = nz
        f["meta/λ0"]               = λ0
        f["meta/data_directories"] = dirs_in_group

        for data_directory in dirs_in_group
            flat_idx[] += 1  # increment the global table index
            printstyled("\t" * data_directory * "\n"; color=:cyan, bold=true)

            data_processed = jldopen(joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2"), "r") do file
                Dict(
                    :Directory            => file["meta/Directory"],
                    :TemperatureInCelsius => file["meta/TemperatureInCelsius"],
                    :AcquisitionStep      => file["meta/AcquisitionStep"],

                    :MGcurrentInA         => file["meta/MGcurrentInA"],
                    :MGfieldInTesla       => file["meta/MGfieldInTesla"],

                    :SG0Currents          => file["meta/SG0currentInA"],
                    :SG0Bz                => file["meta/SG0BfieldInTesla"],

                    :SG1Currents          => file["meta/SG1currentInA"],
                    :SG1Bz                => file["meta/SG1BfieldInTesla"],

                    :F1ProcessedImages    => file["data/F1ProcessedImages"],
                    :F2ProcessedImages    => file["data/F2ProcessedImages"],
                )
            end

            # ── Currents & fields ────────────────────────────────────────────────────────
            SG0_current = abs.(data_processed[:SG0Currents])
            SG1_current = abs.(data_processed[:SG1Currents])

            Bz0 = 1e3 .* data_processed[:SG0Bz]   # T → mT
            Bz1 = 1e3 .* data_processed[:SG1Bz]   # T → mT

            # ── Framewise maxima & statistics ────────────────────────────────────────────
            f1_max = MyExperimentalAnalysis.SG0_framewise_maxima(
                "F1", data_processed, nz;
                half_max = false,
                λ0 = λ0,
            )

            f2_max = MyExperimentalAnalysis.SG0_framewise_maxima(
                "F2", data_processed, nz;
                half_max = false,
                λ0 = λ0,
            )

            f1_z_mm = vec(mean(f1_max, dims = 1))
            f2_z_mm = vec(mean(f2_max, dims = 1))

            f1_z_sem_mm = sqrt.(
                vec(std(f1_max, dims = 1; corrected = true) ./ sqrt(size(f1_max, 1))).^2 .+
                z_mm_error^2
            )

            f2_z_sem_mm = sqrt.(
                vec(std(f2_max, dims = 1; corrected = true) ./ sqrt(size(f2_max, 1))).^2 .+
                z_mm_error^2
            )

            # ── Splitting and centroid ───────────────────────────────────────────────────
            Δz_mm = f1_z_mm .- f2_z_mm

            Δz_sem_mm = sqrt.(
                f1_z_sem_mm.^2 .+
                f2_z_sem_mm.^2
            )

            centroid_mm = 0.5 .* (f1_z_mm .+ f2_z_mm)

            centroid_sem_mm = 0.5 .* sqrt.(
                f1_z_sem_mm.^2 .+
                f2_z_sem_mm.^2
            )

            # ── Summary DataFrame ────────────────────────────────────────────────────────
            data = DataFrame(
                I0        = SG0_current,
                I1        = SG1_current,
                B0        = Bz0,
                B1        = Bz1,
                zf1       = f1_z_mm,
                errzf1    = f1_z_sem_mm,
                zf2       = f2_z_mm,
                errzf2    = f2_z_sem_mm,
                split     = Δz_mm,
                errsplit  = round.(Δz_sem_mm; sigdigits = 2),
                c0        = centroid_mm,
                errc0     = round.(centroid_sem_mm; sigdigits = 2),
            )
            tables[flat_idx[]] = data
            f[data_directory]  = data  # ← save immediately
            @info "Saved table" data_directory

            pretty_table(data;
                title         = data_directory * " T=$(data_processed[:TemperatureInCelsius])°C",
                formatters    = [ fmt__printf("%8.5f", 1:2), fmt__printf("%8.3f", 3:4), fmt__printf("%8.3f", [5,7,9,11]), fmt__printf("%8.4f", [6,8,10,12])],
                alignment     = :c,
                column_labels = [
                    ["I0 Current", "I1 Current", "B0 field", "B1 field", "F1", "F1 err", "F2", "F2 err", "Δz", "Δz err", "C₀", "C₀ err "],
                    ["[A]", "[A]", "[mT]" ,"[mT]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
                ],
                table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
                style         = TextTableStyle(
                    first_line_column_label = crayon"yellow bold",
                    column_label            = crayon"yellow",
                    table_border            = crayon"blue bold",
                    title                   = crayon"bold red",
                ),
                equal_data_column_widths = true,
            )

            tol = 1e-9
            mask = (abs.(data.I0) .<= tol) .&& (abs.(data.I1) .<= tol)
            rows_zero = data[mask, :]
            mean_zero = DataFrame(
                [name => mean(rows_zero[!, name]) for name in names(rows_zero)]
            )

            pretty_table(mean_zero;
                title         = data_directory,
                formatters    = [ fmt__printf("%8.5f", 1:2), fmt__printf("%8.3f", 3:4), fmt__printf("%8.3f", [5,7,9,11]), fmt__printf("%8.4f", [6,8,10,12])],
                alignment     = :c,
                column_labels = [
                    ["I0 Current", "I1 Current", "B0 field", "B1 field", "F1", "F1 err", "F2", "F2 err", "Δz", "Δz err", "C₀", "C₀ err "],
                    ["[A]", "[A]", "[mT]" ,"[mT]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
                ],
                table_format  = TextTableFormat(borders = text_table_borders__unicode_rounded),
                style         = TextTableStyle(
                    first_line_column_label = crayon"yellow bold",
                    column_label            = crayon"yellow",
                    table_border            = crayon"blue bold",
                    title                   = crayon"bold red",
                ),
                equal_data_column_widths = true,
            )


            data_pos = data[data.I0 .> 0, :]
            # ── Colour palette (one colour per SG0 current step) ─────────────────────
            colors_sg0 = palette(:darkrainbow, size(data,1));
            sg1_label = "$(round(1000 * data.I1[end], digits=2))mA"   # reused in titles

            # ── Overlay of all SG0 currents: z-position vs current ───────────────────
            common_scatter_kw = (seriestype=:scatter, marker=(:circle, 2, :white))

            fig1 = plot(data.I0, data.zf1;
                common_scatter_kw..., yerror=data.errzf1,
                label=L"$F=1$", markerstrokecolor=:red);
            plot!(fig1, data.I0, data.zf2;
                common_scatter_kw..., yerror=data.errzf2,
                label=L"$F=2$", markerstrokecolor=:blue);
            plot!(fig1, data.I0, mean([data.zf1, data.zf2]);
                label="Centre", marker=(:diamond, :white),
                markerstrokecolor=:gray47, line=(:dash, 1, :gray47));
            plot!(fig1;
                xlabel="SG0 Current (A)",
                foreground_color_legend=nothing, background_color_legend=nothing,
                yformatter=y -> @sprintf("%.3f", y));

            fig2 = plot(data_pos.I0, data_pos.zf1;
                common_scatter_kw..., yerror=data_pos.errzf1,
                label=L"$F=1$", markerstrokecolor=:red);
            plot!(fig2, data_pos.I0, data_pos.zf2;
                common_scatter_kw..., yerror=data_pos.errzf2,
                label=L"$F=2$", markerstrokecolor=:blue);
            plot!(fig2, data_pos.I0, mean([data_pos.zf1, data_pos.zf2]),
                label="Centre", marker=(:diamond, :white),
                markerstrokecolor=:gray47, line=(:dash, 1, :gray47));
            plot!(fig2;
                xscale=:log10,
                foreground_color_legend=nothing, background_color_legend=nothing,
                xlabel="SG0 Current (A)",
                yformatter=y -> @sprintf("%.3f", y));

            fig3 = plot(data.I0, data.split ./ 6.5e-3;
                common_scatter_kw..., yerror=data.errsplit ./ 6.5e-3,
                label=L"$\Delta z$", markerstrokecolor=:darkgreen);
            plot!(fig3;
                foreground_color_legend=nothing, background_color_legend=nothing,
                xlabel="SG0 Current (A)", ylabel="Separation (px)", yminorticks=false);

            fig4 = plot(data_pos.I0, data_pos.split ./ 6.5e-3;
                common_scatter_kw..., yerror=data_pos.errsplit ./ 6.5e-3,
                label=L"$\Delta z$", markerstrokecolor=:darkgreen);
            plot!(fig4;
                xscale=:log10,
                foreground_color_legend=nothing, background_color_legend=nothing,
                xlabel="SG0 Current (A)", ylabel="Separation (px)", yminorticks=false);

            fig = plot(fig1, fig2, fig3, fig4;
                suptitle     = "$(data_directory) | SG1: $sg1_label",
                layout       = (2, 2),
                size         = (1000, 600),
                link         = :x,
                left_margin  = 5mm,
                bottom_margin= 3mm,
            )
            plot!(fig[1]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
            plot!(fig[2]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
            display(fig)
            
            # ── Extract image arrays once, outside all loops ─────────────────────────
            f1imgs = data_processed[:F1ProcessedImages];
            f2imgs = data_processed[:F2ProcessedImages];
            sg1_label    = "$(data_directory) | SG1: $(round(1000*data.I1[end], digits=2))mA";
            current_label(i) = L"$%$(round(1000*SG0_current[i]; digits=3))\mathrm{mA}$";

            legend_kw = (
                legend                  = :topleft,
                legend_title            = "SG0",
                background_color_legend = :white,
                foreground_color_legend = nothing,
            );

            # ── Pre-compute cleaned images and profiles for all currents ──────────────
            # Store so each quantity is computed once and reused in both plot blocks
            f1_images   = Vector{Matrix{Float64}}(undef, length(SG0_current));
            f2_images   = Vector{Matrix{Float64}}(undef, length(SG0_current));
            f1_profiles = Vector{Vector{Float64}}(undef, length(SG0_current));
            f2_profiles = Vector{Vector{Float64}}(undef, length(SG0_current));


            for i in eachindex(SG0_current)
                # Average over frames (dim 3), then mask
                f1img = dropdims(mean(f1imgs[:,:,:,i], dims=3), dims=3)
                f1img = mapwindow(median, f1img, (3, 3))
                # f1img .*= isfinite.(f1img) .& (f1img .>= -10) .& (f1img .<= 1000)

                f2img = dropdims(mean(f2imgs[:,:,:,i], dims=3), dims=3)
                # f2img .*= isfinite.(f2img) .& (f2img .>=  -5) .& (f2img .<= 500)

                f1_images[i]   = f1img
                f2_images[i]   = f2img

                # Profile: average over x (dim 1) after frame-averaging and masking
                f1_profiles[i] = vec(mean(f1img, dims=1))
                f2_profiles[i] = vec(mean(f2img, dims=1))
            end

            # ── Per-current panels: heatmap (left) + z-profile (right) ───────────────
            camera_z_mm = 1e3 .* pixel_positions(z_pixels, 1, exp_pixelsize_z) 
            for i in eachindex(SG0_current)
                f1vmax = Statistics.quantile(vec(f1_images[i]), 0.999)
                f2vmax = Statistics.quantile(vec(f2_images[i]), 0.999)

                plt1 = heatmap(f1_images[i];
                    xlabel = L"$z\ \ (\mathrm{px})$",
                    ylabel = L"$x\ \ (\mathrm{px})$",
                    cbar   = true, clims = (0, f1vmax),
                )
                plt2 = plot(camera_z_mm, f1_profiles[i];
                    line = (:solid, 1, colors_sg0[i]), label = current_label(i),
                    xlabel = L"$z\ \ (\mathrm{px})$", legend_kw...,
                )
                vline!(plt2, [f1_max[i]], line = (:dot, 1, colors_sg0[i]), label = false)
                plt3 = heatmap(f2_images[i];
                    xlabel = L"$z\ \ (\mathrm{px})$",
                    ylabel = L"$x\ \ (\mathrm{px})$",
                    cbar   = true, clims = (0, f2vmax),
                )
                plt4 = plot(camera_z_mm, f2_profiles[i];
                    line = (:solid, 1, colors_sg0[i]), label = current_label(i),
                    xlabel = L"$z\ \ (\mathrm{px})$", legend_kw...,
                )
                vline!(plt4, [f2_max[i]], line = (:dot, 1, colors_sg0[i]), label = false)

                plt = plot(plt1, plt2, plt3, plt4;
                    suptitle      = "$(data_directory) | SG1: $(round(1000*data.I1[i], digits=2))mA",
                    layout        = (2, 2),
                    link          = :x,
                    size          = (800, 450),
                    left_margin   = 3mm,
                    bottom_margin = 2mm,
                )
                plot!(plt[1]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
                plot!(plt[2]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
                display(plt)
            end

            # ── Overlay: all currents on one z-profile plot per state ─────────────────
            overlay_kw = (
                legend                  = :topleft,
                legend_font             = 8,
                foreground_color_legend = nothing,
                legend_columns          = 2,
                xlabel                  = L"$z \ \ (\mathrm{px})$",
            );

            plt_f1 = plot(; overlay_kw...);
            plt_f2 = plot(; overlay_kw...);

            for i in eachindex(SG0_current)
                plot!(plt_f1, f1_profiles[i]; line=(:solid, 1, colors_sg0[i]), label=current_label(i))
                plot!(plt_f2, f2_profiles[i]; line=(:solid, 1, colors_sg0[i]), label=current_label(i))
            end
            
            plot!(plt_f1; legend_title=L"$F=1$");
            plot!(plt_f2; legend_title=L"$F=2$");

            fig = plot(plt_f1, plt_f2;
                suptitle      = sg1_label,
                layout        = (2, 1),
                size          = (900, 850),
                left_margin   = 3mm,
                bottom_margin = 2mm,
            )
            plot!(fig[1]; xlabel="", xformatter=_->"", bottom_margin=-6mm)
            display(fig)

            GC.gc()
        end
    end
end

function make_pair_plots(tables, data_directories; xmode = :log10)

    scale_factor = 6.5e-3

    mark_symbol      = [:circle, :rect]
    colores          = (:darkgreen, :indigo)
    current_polarity = ("+", "–")

    tol = 1e-9

    plot_specs = [
        (:split, :errsplit, L"$\Delta z$ (px)", scale_factor),
        (:c0,    :errc0,    L"$c_{0}$ (mm)",    1.0),
        (:zf1,   :errzf1,   L"$F=1$ (mm)",      1.0),
        (:zf2,   :errzf2,   L"$F=2$ (mm)",      1.0),
    ]

    pair_indices = [(1, 2), (3, 4), (5, 6), (7, 8)]

    fig_pairs = []

    for pair in pair_indices

        figs = []

        for (ycol, errcol, ylabel_text, sf) in plot_specs

            fig = plot(
                xlabel = "SG0 Current (A)",
                ylabel = ylabel_text,
            )

            for (j, k) in enumerate(pair)

                data = tables[k]

                data_pos = data[data.I0 .> 0, :]

                mask = isapprox.(data.I0, 0.0; atol=tol) .&&
                       .!isapprox.(data.I1, 0.0; atol=tol)

                rows_zero = data[mask, :]

                if nrow(rows_zero) > 0
                    rows_zero = DataFrame(
                        [name => mean(rows_zero[!, name]) for name in names(rows_zero)]
                    )
                end

                plot!(fig,
                    data_pos.I0,
                    data_pos[!, ycol] ./ sf,
                    yerror            = data_pos[!, errcol] ./ sf,
                    label             = "($(current_polarity[j])) " * data_directories[k],
                    line              = (:solid, 1, colores[j], 0.5),
                    marker            = (mark_symbol[j], 2, :white),
                    markerstrokecolor = colores[j],
                )

                if nrow(rows_zero) > 0
                    y0  = only(rows_zero[!, ycol]) / sf
                    dy0 = only(rows_zero[!, errcol]) / sf

                    hline!(fig, [y0],
                        label = false,
                        line  = (:dash, 1, colores[j], 0.5),
                    )

                    hspan!(fig, [y0 - dy0, y0 + dy0],
                        label     = false,
                        color     = colores[j],
                        fillalpha = 0.15,
                        linealpha = 0,
                    )
                end
            end

            if xmode == :log10
                plot!(fig,
                    xlims = (0.95e-3, 5.0),
                    xticks = (
                        [1e-3, 1e-2, 1e-1, 1.0],
                        [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]
                    ),
                    xscale = :log10,
                )
            elseif xmode == :linear
                plot!(fig,
                    xlims = (-0.2,4.0),
                    xscale = :identity,
                )
            else
                error("xmode must be either :log10 or :linear")
            end

            plot!(fig,
                foreground_color_legend = nothing,
                background_color_legend = nothing,
                yminorticks             = 5,
                legend                  = :topleft,
            )

            push!(figs, fig)
        end

        fig1a, fig1b, fig1c, fig1d = figs

        fig_pair = plot(
            fig1a, fig1b, fig1c, fig1d,
            size   = (800, 600),
            layout = (2, 2),
            link   = :x,
        )

        if xmode == :log10
            xtick_vals   = [1e-3, 1e-2, 1e-1, 1.0]
            xtick_labels = [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]
            empty_labels = fill("", length(xtick_vals))

            plot!(fig_pair, subplot = 1,
                xticks = (xtick_vals, empty_labels),
                xlabel = "",
                bottom_margin = -5mm,
            )

            plot!(fig_pair, subplot = 2,
                xticks = (xtick_vals, empty_labels),
                xlabel = "",
                bottom_margin = -5mm,
            )

            plot!(fig_pair, subplot = 3,
                xticks = (xtick_vals, xtick_labels),
                xlabel = "SG0 Current (A)",
            )

            plot!(fig_pair, subplot = 4,
                xticks = (xtick_vals, xtick_labels),
                xlabel = "SG0 Current (A)",
            )

        elseif xmode == :linear

            plot!(fig_pair, subplot = 1,
                xlabel = "",
                xformatter = _ -> "",
                bottom_margin = -5mm,
            )

            plot!(fig_pair, subplot = 2,
                xformatter = _ -> "",
                xlabel = "",
                bottom_margin = -5mm,
            )

            plot!(fig_pair, subplot = 3,
                xlabel = "SG0 Current (A)",
            )

            plot!(fig_pair, subplot = 4,
                xlabel = "SG0 Current (A)",
            )
        end

        ymin, ymax = ylims(fig1a)

        plot!(fig_pair, subplot = 1,
            # xlabel = "",
            # xformatter = _ -> "",
            yticks = floor(ymin):1:ceil(ymax),
        )

        plot!(fig_pair, subplot = 2,
            # xlabel = "",
            # xformatter = _ -> "",
            yformatter = y -> @sprintf("%.3f", y),
        )

        plot!(fig_pair, subplot = 3,
            yformatter = y -> @sprintf("%.3f", y),
        )

        plot!(fig_pair, subplot = 4,
            yformatter = y -> @sprintf("%.3f", y),
        )

        push!(fig_pairs, fig_pair)

        display(fig_pair)
    end

    return fig_pairs
end




scale_factor = 6.5e-3;

load_tables = Vector{DataFrame}(undef, nd)
for (i,data_directory) in enumerate(data_directories)
    load_path = joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS", dirname(data_directory), "data_analysis_20260706T145140446.jld2")
    load_data = load(load_path)
    load_tables[i] = load_data[data_directory]
end
tables = load_tables;

make_pair_plots(tables, data_directories; xmode = :log10)
make_pair_plots(tables, data_directories; xmode = :linear)

for s_table in tables
    show(stdout, s_table, allcols = true)
    println("\n")
end


# Reference / Baseline
sg0_ref = OrderedDict{Int, DataFrame}()
for (i, idx) in enumerate(3:4)

    @info "Data set : $(data_directories[idx])"

    ref00 = copy(tables[idx])
    show(stdout, ref00, allcols = true, allrows = true)
    println()

    group_cols = [:I0, :I1]

    err_cols = Symbol.(filter(name -> startswith(String(name), "err"), names(ref00)))
    mean_cols = setdiff(Symbol.(names(ref00)), vcat(group_cols, err_cols))

    grouped = groupby(ref00, group_cols, sort = false)

    df_reduced = combine(grouped) do sdf

        out = (;)

        for col in mean_cols
            out = merge(out, (; col => mean(sdf[!, col])))
        end

        for col in err_cols
            n = nrow(sdf)
            propagated_err = round(sqrt(sum(sdf[!, col].^2)) / n; digits=4)
            out = merge(out, (; col => propagated_err))
        end

        return out
    end

    # Restore original column order
    df_reduced = df_reduced[:, names(ref00)]

    sg0_ref[idx] = df_reduced

    println()
    println("Reduced dataframe for data set : $(data_directories[idx])")
    println("-"^100)
    show(stdout, df_reduced, allcols = true, allrows = true)
    println("\n\n")

end

# Experiment data
selected_indices = vcat(1:2, 5:8)
sg0_data = OrderedDict{Int, DataFrame}()
for idx in selected_indices
    tol = 1e-9;
    @info "Data set : $(data_directories[idx])"

    df = copy(tables[idx])

    println()
    println("Original dataframe for data set $(idx): $(data_directories[idx])")
    println("-"^100)
    show(stdout, df, allcols = true, allrows = true)
    println("\n")

    group_cols = [:I0, :I1]

    # Error columns: errzf1, errzf2, errsplit, errc0, etc.
    err_cols = Symbol.(filter(name -> startswith(String(name), "err"), names(df)))

    # Ordinary columns: averaged normally
    mean_cols = setdiff(Symbol.(names(df)), vcat(group_cols, err_cols))

    # ------------------------------------------------------------
    # Create groups only for consecutive repeated I0, I1 blocks
    # ------------------------------------------------------------
    same_as_previous = [
        false;
        (isapprox.(df.I0[2:end], df.I0[1:end-1]; atol=tol) .&&
         isapprox.(df.I1[2:end], df.I1[1:end-1]; atol=tol))
    ]

    df.block_id = cumsum(.!same_as_previous)

    grouped = groupby(df, :block_id, sort = false)

    df_reduced = combine(grouped) do sdf

        out = (;)

        # Keep representative I0 and I1.
        # Since rows inside each block have the same I0/I1,
        # mean and first are equivalent up to tolerance.
        for col in group_cols
            out = merge(out, (; col => mean(sdf[!, col])))
        end

        # Average ordinary columns
        for col in mean_cols
            out = merge(out, (; col => mean(sdf[!, col])))
        end

        # Propagate independent errors:
        # σ_mean = sqrt(σ₁² + σ₂² + ... + σₙ²) / n
        for col in err_cols
            n = nrow(sdf)
            propagated_err = sqrt(sum(sdf[!, col].^2)) / n
            out = merge(out, (; col => propagated_err))
        end

        return out
    end

    # Remove block_id if present
    if :block_id in propertynames(df_reduced)
        select!(df_reduced, Not(:block_id))
    end

    # Restore original column order
    df_reduced = df_reduced[:, names(tables[idx])]

    # Store using the original table index as the key
    sg0_data[idx] = df_reduced

    println()
    println("Reduced consecutive-block dataframe for data set $(idx): $(data_directories[idx])")
    println("-"^100)
    show(stdout, df_reduced, allcols = true, allrows = true)
    println("\n")

end

## SPLITTING
#linear plots
ref_idx = 3
sg0_indices = [1, 6, 8]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlin_1 = plot(
    title  = "Antiparallel (↑↓) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$\Delta z = | z_{F=1} - z_{F=2} |  \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlin_1,
    sg0_ref[ref_idx].I0,
    abs.(sg0_ref[ref_idx].split ./ scale_factor),
    yerror = sg0_ref[ref_idx].errsplit ./ scale_factor,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= tol)

    plot!(fig_linlin_1,
        df.I0[mask_I1_nonzero],
        abs.(df.split[mask_I1_nonzero] ./ scale_factor),
        yerror = df.errsplit[mask_I1_nonzero] ./ scale_factor,
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlin_1,
            df.I0[mask_both_zero],
            abs.(df.split[mask_both_zero] ./ scale_factor),
            yerror = df.errsplit[mask_both_zero] ./ scale_factor,
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = ylims(fig_linlin_1) ./ 10
plot!(fig_linlin_1,
    xlims=(-0.02,4),
    ylims= (0,10*ymax),
    yticks = floor(ymin):5:ceil(ymax)*10,
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig_linlin_1)


ref_idx = 4
sg0_indices = [2, 5, 7]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlin_2 = plot(
    title  = "Parallel (↑↑) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$\Delta z = | z_{F=1} - z_{F=2} | \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlin_2,
    sg0_ref[ref_idx].I0,
    abs.(sg0_ref[ref_idx].split ./ scale_factor),
    yerror = sg0_ref[ref_idx].errsplit ./ scale_factor,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> 2e3*tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= 2e3*tol)

    plot!(fig_linlin_2,
        df.I0[mask_I1_nonzero],
        abs.(df.split[mask_I1_nonzero] ./ scale_factor),
        yerror = df.errsplit[mask_I1_nonzero] ./ scale_factor,
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlin_2,
            df.I0[mask_both_zero],
            abs.(df.split[mask_both_zero] ./ scale_factor),
            yerror = df.errsplit[mask_both_zero] ./ scale_factor,
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = ylims(fig_linlin_2) ./ 10
plot!(fig_linlin_2,
    xlims=(-0.02,4),
    ylims= (0,10*ymax),
    yticks = 0:5:ceil(ymax)*10,
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig_linlin_2)



#linear-log plots
ref_idx = 3
sg0_indices = [1, 6, 8]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlog_1 = plot(
    title  = "Antiparallel (↑↓) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$\Delta z = | z_{F=1} - z_{F=2} |  \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlog_1,
    sg0_ref[ref_idx].I0,
    abs.(sg0_ref[ref_idx].split ./ scale_factor),
    yerror = sg0_ref[ref_idx].errsplit ./ scale_factor,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= tol)

    plot!(fig_linlog_1,
        df.I0[mask_I1_nonzero],
        abs.(df.split[mask_I1_nonzero] ./ scale_factor),
        yerror = df.errsplit[mask_I1_nonzero] ./ scale_factor,
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlog_1,
            df.I0[mask_both_zero],
            abs.(df.split[mask_both_zero] ./ scale_factor),
            yerror = df.errsplit[mask_both_zero] ./ scale_factor,
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = ylims(fig_linlog_1) ./ 10
plot!(fig_linlog_1,
    xlims=(3e-3,4),
    ylims= (0,10*ymax),
    yticks = floor(ymin):5:ceil(ymax)*10,
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
plot!(fig_linlog_1,
    xscale=:log10,                    
    xticks = (
        [1e-2, 1e-1, 1.0],
        [L"10^{-2}", L"10^{-1}", L"10^{0}"]
    ),
)
display(fig_linlog_1)


ref_idx = 4
sg0_indices = [2, 5, 7]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlog_2 = plot(
    title  = "Parallel (↑↑) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$\Delta z = | z_{F=1} - z_{F=2} | \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlog_2,
    sg0_ref[ref_idx].I0,
    abs.(sg0_ref[ref_idx].split ./ scale_factor),
    yerror = sg0_ref[ref_idx].errsplit ./ scale_factor,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> 2e3*tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= 2e3*tol)

    plot!(fig_linlog_2,
        df.I0[mask_I1_nonzero],
        abs.(df.split[mask_I1_nonzero] ./ scale_factor),
        yerror = df.errsplit[mask_I1_nonzero] ./ scale_factor,
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlog_2,
            df.I0[mask_both_zero],
            abs.(df.split[mask_both_zero] ./ scale_factor),
            yerror = df.errsplit[mask_both_zero] ./ scale_factor,
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = ylims(fig_linlog_2) ./ 10
plot!(fig_linlog_2,
    xlims=(3e-3,4),
    ylims= (0,10*ymax),
    yticks = 0:5:ceil(ymax)*10,
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
plot!(fig_linlog_2,
    xscale=:log10,                    
    xticks = (
        [1e-2, 1e-1, 1.0],
        [L"10^{-2}", L"10^{-1}", L"10^{0}"]
    ),
)
display(fig_linlog_2)


#log-log plots
ref_idx = 3
sg0_indices = [1, 6, 8]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_loglog_1 = plot(
    title  = "Antiparallel (↑↓) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$\Delta z = | z_{F=1} - z_{F=2} |  \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_loglog_1,
    sg0_ref[ref_idx].I0,
    abs.(sg0_ref[ref_idx].split ./ scale_factor),
    yerror = sg0_ref[ref_idx].errsplit ./ scale_factor,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6
    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= tol)

    plot!(fig_loglog_1,
        df.I0[mask_I1_nonzero],
        abs.(df.split[mask_I1_nonzero] ./ scale_factor),
        yerror = df.errsplit[mask_I1_nonzero] ./ scale_factor,
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_loglog_1,
            df.I0[mask_both_zero],
            abs.(df.split[mask_both_zero] ./ scale_factor),
            yerror = df.errsplit[mask_both_zero] ./ scale_factor,
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = ylims(fig_loglog_1) ./ 10
plot!(fig_loglog_1,
    xlims=(3e-3,4),
    ylims= (0.5,10*ymax),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
plot!(fig_loglog_1,
    xscale=:log10,                    
    xticks = (
        [1e-2, 1e-1, 1.0],
        [L"10^{-2}", L"10^{-1}", L"10^{0}"]
    ),
    yscale=:log10,
    yticks = (
        [1e-1, 1.0, 10],
        [L"10^{-1}", L"10^{0}", L"10^{1}"]
    ),

)
display(fig_loglog_1)


ref_idx = 4
sg0_indices = [2, 5, 7]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_loglog_2 = plot(
    title  = "Parallel (↑↑) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$\Delta z = | z_{F=1} - z_{F=2} | \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_loglog_2,
    sg0_ref[ref_idx].I0,
    abs.(sg0_ref[ref_idx].split ./ scale_factor),
    yerror = sg0_ref[ref_idx].errsplit ./ scale_factor,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6
    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> 2e3*tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= 2e3*tol)

    plot!(fig_loglog_2,
        df.I0[mask_I1_nonzero],
        abs.(df.split[mask_I1_nonzero] ./ scale_factor),
        yerror = df.errsplit[mask_I1_nonzero] ./ scale_factor,
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_loglog_2,
            df.I0[mask_both_zero],
            abs.(df.split[mask_both_zero] ./ scale_factor),
            yerror = df.errsplit[mask_both_zero] ./ scale_factor,
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = ylims(fig_loglog_2) ./ 10
plot!(fig_loglog_2,
    xlims=(3e-3,4),
    ylims= (0.5,10*ymax),
    yticks = 0:5:ceil(ymax)*10,
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
plot!(fig_loglog_2,
    xscale=:log10,                    
    xticks = (
        [1e-2, 1e-1, 1.0],
        [L"10^{-2}", L"10^{-1}", L"10^{0}"]
    ),
    yscale=:log10,
    yticks = (
        [1e-1, 1.0, 10],
        [L"10^{-1}", L"10^{0}", L"10^{1}"]
    ),

)
display(fig_loglog_2)


## CENTROID
#linear plots
ref_idx = 3
sg0_indices = [1, 6, 8]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlin_1 = plot(
    title  = "Antiparallel (↑↓) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$c_{0} \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlin_1,
    sg0_ref[ref_idx].I0,
    abs.(sg0_ref[ref_idx].c0),
    yerror = sg0_ref[ref_idx].errsplit,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6
    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= tol)

    plot!(fig_linlin_1,
        df.I0[mask_I1_nonzero],
        df.c0[mask_I1_nonzero],
        yerror = df.errc0[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlin_1,
            df.I0[mask_both_zero],
            df.c0[mask_both_zero],
            yerror = df.errc0[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_1); digits=3)
plot!(fig_linlin_1,
    xlims=(-0.02,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig_linlin_1)


ref_idx = 4
sg0_indices = [2, 5, 7]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlin_2 = plot(
    title  = "Parallel (↑↑) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$c_{0} \quad (\mathrm{mm})$",
)
# Reference curve
sg0_ref[ref_idx]
plot!(fig_linlin_2,
    sg0_ref[ref_idx].I0,
    sg0_ref[ref_idx].c0,
    yerror = sg0_ref[ref_idx].errc0,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> 2e3*tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= 2e3*tol)

    plot!(fig_linlin_2,
        df.I0[mask_I1_nonzero],
        df.c0[mask_I1_nonzero],
        yerror = df.errc0[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlin_2,
            df.I0[mask_both_zero],
            df.c0[mask_both_zero],
            yerror = df.errc0[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_2); digits=3)
plot!(fig_linlin_2,
    xlims=(-0.02,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig_linlin_2)


#linear-log plots
ref_idx = 3
sg0_indices = [1, 6, 8]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlog_1 = plot(
    title  = "Antiparallel (↑↓) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$c_{0} \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlog_1,
    sg0_ref[ref_idx].I0,
    sg0_ref[ref_idx].c0,
    yerror = sg0_ref[ref_idx].errc0,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= tol)

    plot!(fig_linlog_1,
        df.I0[mask_I1_nonzero],
        df.c0[mask_I1_nonzero],
        yerror = df.errc0[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlog_1,
            df.I0[mask_both_zero],
            df.c0[mask_both_zero],
            yerror = df.errc0[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_1); digits=3)
plot!(fig_linlog_1,
    xlims=(3e-3,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
plot!(fig_linlog_1,
    xscale=:log10,                    
    xticks = (
        [1e-2, 1e-1, 1.0],
        [L"10^{-2}", L"10^{-1}", L"10^{0}"]
    ),
)
display(fig_linlog_1)


ref_idx = 4
sg0_indices = [2, 5, 7]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlog_2 = plot(
    title  = "Parallel (↑↑) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$c_{0} \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlog_2,
    sg0_ref[ref_idx].I0,
    sg0_ref[ref_idx].c0,
    yerror = sg0_ref[ref_idx].errc0,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> 2e3*tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= 2e3*tol)

    plot!(fig_linlog_2,
        df.I0[mask_I1_nonzero],
        df.c0[mask_I1_nonzero],
        yerror = df.errc0[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlog_2,
            df.I0[mask_both_zero],
            df.c0[mask_both_zero],
            yerror = df.errc0[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_2); digits=3)
plot!(fig_linlog_2,
    xlims=(3e-3,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
plot!(fig_linlog_2,
    xscale=:log10,                    
    xticks = (
        [1e-2, 1e-1, 1.0],
        [L"10^{-2}", L"10^{-1}", L"10^{0}"]
    ),
)
display(fig_linlog_2)




## F1
#linear plots
ref_idx = 3
sg0_indices = [1, 6, 8]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlin_1 = plot(
    title  = "Antiparallel (↑↓) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"Peak position $F=1 \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlin_1,
    sg0_ref[ref_idx].I0,
    sg0_ref[ref_idx].zf1,
    yerror = sg0_ref[ref_idx].errzf1,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6
    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= tol)

    plot!(fig_linlin_1,
        df.I0[mask_I1_nonzero],
        df.zf1[mask_I1_nonzero],
        yerror = df.errzf1[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlin_1,
            df.I0[mask_both_zero],
            df.zf1[mask_both_zero],
            yerror = df.errzf1[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_1); digits=3)
plot!(fig_linlin_1,
    xlims=(-0.02,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig_linlin_1)


ref_idx = 4
sg0_indices = [2, 5, 7]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlin_2 = plot(
    title  = "Parallel (↑↑) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$c_{0} \quad (\mathrm{mm})$",
)
# Reference curve
sg0_ref[ref_idx]
plot!(fig_linlin_2,
    sg0_ref[ref_idx].I0,
    sg0_ref[ref_idx].zf1,
    yerror = sg0_ref[ref_idx].errzf1,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> 2e3*tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= 2e3*tol)

    plot!(fig_linlin_2,
        df.I0[mask_I1_nonzero],
        df.zf1[mask_I1_nonzero],
        yerror = df.errzf1[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlin_2,
            df.I0[mask_both_zero],
            df.zf1[mask_both_zero],
            yerror = df.errzf1[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_2); digits=3)
plot!(fig_linlin_2,
    xlims=(-0.02,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig_linlin_2)


#linear-log plots
ref_idx = 3
sg0_indices = [1, 6, 8]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlog_1 = plot(
    title  = "Antiparallel (↑↓) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$c_{0} \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlog_1,
    sg0_ref[ref_idx].I0,
    sg0_ref[ref_idx].zf1,
    yerror = sg0_ref[ref_idx].errzf1,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6
    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= tol)

    plot!(fig_linlog_1,
        df.I0[mask_I1_nonzero],
        df.zf1[mask_I1_nonzero],
        yerror = df.errzf1[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlog_1,
            df.I0[mask_both_zero],
            df.zf1[mask_both_zero],
            yerror = df.errzf1[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_1); digits=3)
plot!(fig_linlog_1,
    xlims=(3e-3,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
plot!(fig_linlog_1,
    xscale=:log10,                    
    xticks = (
        [1e-2, 1e-1, 1.0],
        [L"10^{-2}", L"10^{-1}", L"10^{0}"]
    ),
)
display(fig_linlog_1)


ref_idx = 4
sg0_indices = [2, 5, 7]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlog_2 = plot(
    title  = "Parallel (↑↑) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$c_{0} \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlog_2,
    sg0_ref[ref_idx].I0,
    sg0_ref[ref_idx].zf1,
    yerror = sg0_ref[ref_idx].errzf1,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> 2e3*tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= 2e3*tol)

    plot!(fig_linlog_2,
        df.I0[mask_I1_nonzero],
        df.zf1[mask_I1_nonzero],
        yerror = df.errzf1[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlog_2,
            df.I0[mask_both_zero],
            df.zf1[mask_both_zero],
            yerror = df.errzf1[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_2); digits=3)
plot!(fig_linlog_2,
    xlims=(3e-3,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
plot!(fig_linlog_2,
    xscale=:log10,                    
    xticks = (
        [1e-2, 1e-1, 1.0],
        [L"10^{-2}", L"10^{-1}", L"10^{0}"]
    ),
)
display(fig_linlog_2)


## F2
#linear plots
ref_idx = 3
sg0_indices = [1, 6, 8]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlin_1 = plot(
    title  = "Antiparallel (↑↓) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"Peak position $F=2 \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlin_1,
    sg0_ref[ref_idx].I0,
    sg0_ref[ref_idx].zf2,
    yerror = sg0_ref[ref_idx].errzf2,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= tol)

    plot!(fig_linlin_1,
        df.I0[mask_I1_nonzero],
        df.zf2[mask_I1_nonzero],
        yerror = df.errzf2[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlin_1,
            df.I0[mask_both_zero],
            df.zf2[mask_both_zero],
            yerror = df.errzf2[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_1); digits=3)
plot!(fig_linlin_1,
    xlims=(-0.02,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig_linlin_1)

ref_idx = 4
sg0_indices = [2, 5, 7]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlin_2 = plot(
    title  = "Parallel (↑↑) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$c_{0} \quad (\mathrm{mm})$",
)
# Reference curve
sg0_ref[ref_idx]
plot!(fig_linlin_2,
    sg0_ref[ref_idx].I0,
    sg0_ref[ref_idx].zf2,
    yerror = sg0_ref[ref_idx].errzf2,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> 2e3*tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= 2e3*tol)

    plot!(fig_linlin_2,
        df.I0[mask_I1_nonzero],
        df.zf2[mask_I1_nonzero],
        yerror = df.errzf2[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlin_2,
            df.I0[mask_both_zero],
            df.zf2[mask_both_zero],
            yerror = df.errzf2[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_2); digits=3)
plot!(fig_linlin_2,
    xlims=(-0.02,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig_linlin_2)


#linear-log plots
ref_idx = 3
sg0_indices = [1, 6, 8]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlog_1 = plot(
    title  = "Antiparallel (↑↓) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$c_{0} \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlog_1,
    sg0_ref[ref_idx].I0,
    sg0_ref[ref_idx].zf2,
    yerror = sg0_ref[ref_idx].errzf2,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= tol)

    plot!(fig_linlog_1,
        df.I0[mask_I1_nonzero],
        df.zf2[mask_I1_nonzero],
        yerror = df.errzf2[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlog_1,
            df.I0[mask_both_zero],
            df.zf2[mask_both_zero],
            yerror = df.errzf2[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_1); digits=3)
plot!(fig_linlog_1,
    xlims=(3e-3,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
plot!(fig_linlog_1,
    xscale=:log10,                    
    xticks = (
        [1e-2, 1e-1, 1.0],
        [L"10^{-2}", L"10^{-1}", L"10^{0}"]
    ),
)
display(fig_linlog_1)


ref_idx = 4
sg0_indices = [2, 5, 7]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlog_2 = plot(
    title  = "Parallel (↑↑) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$c_{0} \quad (\mathrm{mm})$",
)
# Reference curve
plot!(fig_linlog_2,
    sg0_ref[ref_idx].I0,
    sg0_ref[ref_idx].zf2,
    yerror = sg0_ref[ref_idx].errzf2,
    label  = data_directories[ref_idx],
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)
# SG0 curves
for (idx, color) in zip(sg0_indices, sg0_colors)
    tol = 1e-6

    df = sg0_data[idx]

    # Optional special trimming for idx = 8
    if idx == 8
        df = df[1:end-1, :]
    end

    mask_I1_nonzero = abs.(df.I1) .> 2e3*tol
    mask_both_zero  = (abs.(df.I0) .<= tol) .&& (abs.(df.I1) .<= 2e3*tol)

    plot!(fig_linlog_2,
        df.I0[mask_I1_nonzero],
        df.zf2[mask_I1_nonzero],
        yerror = df.errzf2[mask_I1_nonzero],
        label  = data_directories[idx],
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )

    if any(mask_both_zero)
        scatter!(fig_linlog_2,
            df.I0[mask_both_zero],
            df.zf2[mask_both_zero],
            yerror = df.errzf2[mask_both_zero],
            # label = data_directories[idx] * L" $(I_{0}=I_{1}=0\mathrm{A})$",
            label=false,
            marker = (:square, 2, :white),
            markerstrokecolor = color,
            color = color,
        )
    end
end
ymin, ymax = round.(ylims(fig_linlin_2); digits=3)
plot!(fig_linlog_2,
    xlims=(3e-3,4),
    ylims= (ymin,ymax),
    yformatter = y -> @sprintf("%.3f", y),
    legend = :outerright,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
plot!(fig_linlog_2,
    xscale=:log10,                    
    xticks = (
        [1e-2, 1e-1, 1.0],
        [L"10^{-2}", L"10^{-1}", L"10^{0}"]
    ),
)
display(fig_linlog_2)



###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
## SPLITTING
#linear plots
tol = 1e-6
ref_idx = 3
sg0_indices = [1, 6, 8]
sg0_colors  = [:purple, :dodgerblue, :orange]

fig_linlin_1 = plot(
    # title  = "Antiparallel (↑↓) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$\Delta z = | z_{F=1} - z_{F=2} |  \quad (\mathrm{mm})$",
)
# ------------------------------------------------------------
# Reference curve
# ------------------------------------------------------------
df = copy(sg0_ref[ref_idx])
sort!(df, :I0)
df[!, :split] = df[!, :split] .- df[1, :split]
df[!, :errsplit] = sqrt.(df[!, :errsplit].^2 .+ df[1, :errsplit].^2)
plot!(fig_linlin_1,
    df.I0,
    df.split ./ scale_factor,
    yerror = df.errsplit ./ scale_factor,
    label = dirname(data_directories[ref_idx]) *
            L" $I_{1} = %$(round(1000 * mean(df.I1); digits=1)) \mathrm{mA}$",
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)

# ------------------------------------------------------------
# SG0 curves
# ------------------------------------------------------------
for (idx, color) in zip(sg0_indices, sg0_colors)

    df = copy(sg0_data[idx])

    # Keep only rows with nonzero I1
    df = df[abs.(df.I1) .> tol, :]

    # Optional row removal for dataset 1
    if idx == 1
        df = df[Not(14), :]
    end

    # Subtract first point as reference
    df[!, :split] = df[!, :split] .- df[1, :split]
    df[!, :errsplit] = sqrt.(df[!, :errsplit].^2 .+ df[1, :errsplit].^2)

    # Sort by I0
    sort!(df, :I0)

    plot!(fig_linlin_1,
        df.I0,
        df.split ./ scale_factor,
        yerror = df.errsplit ./ scale_factor,
        label = dirname(data_directories[idx]) *
                L" $I_{1} = %$(round(1000 * mean(df.I1); digits=1)) \mathrm{mA}$",
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )
end
plot!(fig_linlin_1,
    legend = :outerright,
    legend_columns = 1,
    legend_title = "Antiparallel (↑↓)",
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig_linlin_1)


tol = 1.5e-3
ref_idx = 4
sg0_indices = [2, 5, 7]
sg0_colors  = [:purple, :dodgerblue, :orange]
fig_linlin_2 = plot(
    # title  = "Parallel (↑↑) configuration",
    xlabel = "SG0 Current (A)",
    ylabel = L"$\Delta z = | z_{F=1} - z_{F=2} |  \quad (\mathrm{mm})$",
)
# ------------------------------------------------------------
# Reference curve
# ------------------------------------------------------------
df = copy(sg0_ref[ref_idx])
sort!(df, :I0)
df[!, :split] = df[!, :split] .- df[1, :split]
df[!, :errsplit] = sqrt.(df[!, :errsplit].^2 .+ df[1, :errsplit].^2)
plot!(fig_linlin_2,
    df.I0,
    df.split ./ scale_factor,
    yerror = df.errsplit ./ scale_factor,
    label = dirname(data_directories[ref_idx]) *
            L" $I_{1} = %$(round(1000 * mean(df.I1); digits=1)) \mathrm{mA}$",
    marker = (:circle, 2, :white),
    markerstrokecolor = :darkgreen,
    line = (:solid, 1, :darkgreen),
)

# ------------------------------------------------------------
# SG0 curves
# ------------------------------------------------------------
for (idx, color) in zip(sg0_indices, sg0_colors)

    df = copy(sg0_data[idx])

    # Keep only rows with nonzero I1
    df = df[abs.(df.I1) .> tol, :]

    # Subtract first point as reference
    df[!, :split] = df[!, :split] .- df[1, :split]
    df[!, :errsplit] = sqrt.(df[!, :errsplit].^2 .+ df[1, :errsplit].^2)

    # Sort by I0
    sort!(df, :I0)

    plot!(fig_linlin_2,
        df.I0,
        df.split ./ scale_factor,
        yerror = df.errsplit ./ scale_factor,
        label = dirname(data_directories[idx]) *
                L" $I_{1} = %$(round(1000 * mean(df.I1); digits=1)) \mathrm{mA}$",
        marker = (:rect, 2, :white),
        markerstrokecolor = color,
        line = (:solid, 1, color),
    )
end
plot!(fig_linlin_2,
    legend = :outerright,
    legend_columns = 1,
    legend_title = "Parallel (↑↑)",
    background_color_legend = nothing,
    foreground_color_legend = nothing,
)
display(fig_linlin_2)


fig=plot(fig_linlin_1, fig_linlin_2,
layout=(2,1),
size=(1920,1080),
link=:x,
    markersize = 6,
    legendtitlefontsize = 18,
    legendfontsize = 14,
    guidefontsize = 20,   # xlabel and ylabel size
    tickfontsize  = 14,   # x and y tick label size
left_margin   = 12mm,
bottom_margin = 8mm,
)
plot!(fig[1]; 
    xlabel="", 
    xformatter=_->"", 
    bottom_margin=-6mm,
)
display(fig)



###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
## SPLITTING

fig_linlin_1 = plot(
    xlabel = "SG0 Current (A)",
    ylabel = L"$\Delta z = z_{F=1} - z_{F=2} \quad (\mathrm{px})$",
);
ref_idxs = (3, 4)
plot_list = [
    (3, :darkgreen,     :rect,    "(↑↓)", 1e-6),
    (4, :darkgreen,     :diamond, "(↑↑)", 1.5e-3),

    (1, :purple,     :rect,    "(↑↓)", 1e-6),
    (2, :purple,     :diamond, "(↑↑)", 1.5e-3),

    (6, :dodgerblue, :rect,    "(↑↓)", 1e-6),
    (5, :dodgerblue, :diamond, "(↑↑)", 1.5e-3),

    (8, :orange,     :rect, "(↑↓)", 1.5e-3),
    (7, :orange,     :diamond,    "(↑↑)", 1e-6),
];
for (idx, color, marker_symbol, config_label, tol) in plot_list

    df = copy(idx in ref_idxs ? sg0_ref[idx] : sg0_data[idx])

    @info "Processing dataset" idx
    println()
    show(stdout, df)
    println()

    # Keep only rows with nonzero I1, except for reference datasets
    if idx ∉ ref_idxs
        df = df[abs.(df.I1) .> tol, :]
    end

    # Dataset-specific row removals
    if idx == 1
        df = df[Not(14), :]
    elseif idx == 8
        df = df[1:end-1, :]
    end

    # Subtract first point as reference
    split0 = df[1, :split]
    err0   = df[1, :errsplit]

    df[!, :split]    .-= split0
    df[!, :errsplit] .= sqrt.(df[!, :errsplit].^2 .+ err0^2)

    sort!(df, :I0)

    println()
    show(stdout, df)
    println()

    plot!(fig_linlin_1,
        df.I0,
        df.split ./ scale_factor,
        yerror = df.errsplit ./ scale_factor,
        label = dirname(data_directories[idx]) * " " * config_label * " " *
                L" $I_{1} = %$(round(1000 * mean(df.I1); digits=1)) \mathrm{mA}$",
        marker = (marker_symbol, 8, color, 0.60),
        markerstrokecolor = color,
        line = (:solid, 2, color),
    )
end
ymin, ymax = ylims(fig_linlin_1)
plot!(fig_linlin_1,
    ylims = (floor(ymin), ceil(ymax)),
    yticks = floor(ymin):1:ceil(ymax),
);
plot!(fig_linlin_1,
    legendtitle = "",
    legend = :bottomleft,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size = (1920, 1080),
    legendtitlefontsize = 18,
    legendfontsize = 18,
    guidefontsize = 24,
    tickfontsize = 18,
    left_margin = 12mm,
    bottom_margin =12mm,
);
display(fig_linlin_1)
plot!(fig_linlin_1,
    xlims=(4e-3,4),
    xscale=:log10,
);
display(fig_linlin_1)




## INTERPOLATIONS
split_itp    = Dict{Int, Any}()
errsplit_itp = Dict{Int, Any}()
processed_df = Dict{Int, DataFrame}()

Itest = collect(range(0.001,3.55, length=1001)) ;

ref_idxs = (3, 4)
for (idx, color, marker_symbol, config_label, tol) in plot_list

    @info "Processing dataset" idx

    # Select source dataframe
    df = copy(idx in ref_idxs ? sg0_ref[idx] : sg0_data[idx])

    # Keep only rows with nonzero I1, except for reference datasets
    if idx ∉ ref_idxs
        df = df[abs.(df.I1) .> tol, :]
    end

    # Optional row removals
    if idx == 1
        df = df[Not(14), :]
    elseif idx == 8
        df = df[1:end-1, :]
    end

    # Subtract first point as reference
    split0 = df[1, :split]
    err0   = df[1, :errsplit]

    df[!, :split]    .-= split0
    df[!, :errsplit] .= sqrt.(df[!, :errsplit].^2 .+ err0^2)

    # Sort by I0 before interpolation
    sort!(df, :I0)

    show(stdout, df)

    # Store processed dataframe
    processed_df[idx] = df

    # Interpolation variables
    x  = df.I0
    y  = df.split ./ scale_factor
    dy = df.errsplit ./ scale_factor
    w  = 1.0 ./ dy

    # Cubic/quintic interpolations
    split_itp[idx] = Dierckx.Spline1D(x, y; w = w, k = 5, s = 8)

    errsplit_itp[idx] = Dierckx.Spline1D(x, dy; k = 5, s = 8)

    plt = scatter(
        df.I0,
        df.split ./ scale_factor,
        yerror = df.errsplit ./ scale_factor,
        label = "Experiment",
        marker = (marker_symbol, 8, color, 0.60),
        markerstrokecolor = color,
    )

    plot!(
        plt,
        Itest,
        split_itp[idx].(Itest),
        ribbon = errsplit_itp[idx].(Itest),
        label = "Interpolation",
        line = (:solid, 2, color),
    )

    plot!(
        plt,
        xlims = (1e-3, 4),
        xscale = :log10,
        title = "Data set idx : $(idx)",
        xlabel = "SG0 Current (A)",
        ylabel = L"$\Delta z$ (px)",
    )

    display(plt)
end

idxs = [1, 2, 3, 4, 6, 5, 8, 7]
asymptote_results = Dict{Int, NamedTuple}()
for idx in idxs

    # Choose low-current fitting window
    Imin_fit = 1e-3
    Imax_fit = 0.100

    # Dense current grid for evaluating the interpolated function
    Ifit = collect(range(Imin_fit, Imax_fit, length = 300))

    # Evaluate interpolated function
    yfit = split_itp[idx].(Ifit)

    # Model: y(I) = y0 + a I^p
    model(I, p) = @. p[1] + p[2] * I^p[3]

    # Initial guess
    p0 = [
        yfit[1],              # y0 guess
        yfit[end] - yfit[1],  # amplitude guess
        1.0                   # power guess
    ]

    fit = curve_fit(model, Ifit, yfit, p0)

    pars = coef(fit)
    errs = stderror(fit)

    y0   = pars[1]
    dy0  = errs[1]
    a    = pars[2]
    pexp = pars[3]

    asymptote_results[idx] = (
        y0 = y0,
        dy0 = dy0,
        a = a,
        p = pexp,
        Imin_fit = Imin_fit,
        Imax_fit = Imax_fit,
        npoints = length(Ifit),
    )

    println()
    println("idx = $idx")
    println("asymptotic y0 = $(round(y0; digits=4)) ± $(round(dy0; digits=4)) px")
    println("power p       = $(round(pexp; digits=3))")
    println("fit range     = $Imin_fit A ≤ I0 ≤ $Imax_fit A")
end

idx_pairs = [(1, 2), (3,4), (6, 5), (8,7)]
y0_pair_mean = Dict{Tuple{Int,Int}, NamedTuple}()
for pair in idx_pairs

    i, j = pair

    y0_i  = asymptote_results[i].y0
    y0_j  = asymptote_results[j].y0

    dy0_i = asymptote_results[i].dy0
    dy0_j = asymptote_results[j].dy0

    y0_mean = mean([y0_i, y0_j])

    # Error propagation for mean of two independent values:
    # σ_mean = sqrt(σ₁² + σ₂²) / 2
    dy0_mean = sqrt(dy0_i^2 + dy0_j^2) / 2

    y0_pair_mean[pair] = (
        y0_mean = y0_mean,
        dy0_mean = dy0_mean,
        y0_values = (y0_i, y0_j),
        dy0_values = (dy0_i, dy0_j),
    )

    println()
    println("pair = $pair")
    println("y0 mean = $(round(y0_mean; digits=4)) ± $(round(dy0_mean; digits=4)) px")
end

y0_pair_mean[(1, 2)].y0_mean
y0_pair_mean[(3, 4)].y0_mean
y0_pair_mean[(6, 5)].y0_mean
y0_pair_mean[(8, 7)].y0_mean

Iexp_sampled = sg0_data[2].I0[2:end]

plot(Iexp_sampled,split_itp[1].(Iexp_sampled) .- y0_pair_mean[(1, 2)].y0_mean,
    )
plot!(Iexp_sampled,split_itp[2].(Iexp_sampled).- y0_pair_mean[(1, 2)].y0_mean,
    )
plot!(Iexp_sampled,split_itp[3].(Iexp_sampled) .- y0_pair_mean[(3, 4)].y0_mean )
plot!(Iexp_sampled,split_itp[4].(Iexp_sampled).- y0_pair_mean[(3, 4)].y0_mean)
plot!(Iexp_sampled,split_itp[6].(Iexp_sampled) .- y0_pair_mean[(6, 5)].y0_mean )
plot!(Iexp_sampled,split_itp[5].(Iexp_sampled).- y0_pair_mean[(6, 5)].y0_mean)
plot!(Iexp_sampled,split_itp[8].(Iexp_sampled) .- y0_pair_mean[(8, 7)].y0_mean )
plot!(Iexp_sampled,split_itp[7].(Iexp_sampled).- y0_pair_mean[(8, 7)].y0_mean)
plot!(xscale=:log10)


scatter!(sg0_data[1].I0, (sg0_data[1].split .- sg0_data[1].split[1])./ scale_factor)
plot!(Itest, split_itp[2].(Itest))
2+2



# Pairs used to define the common offset y0
pairs = [
    (3, 4), 
    (1, 2), 
    (6, 5), 
    (8, 7)]

# Map each index to its pair
pair_of_idx = Dict(idx => pair for pair in pairs for idx in pair)

plt = plot(
    xlabel = "SG0 current (A)",
    ylabel = "Spin-resolved beam separation (pixels)",
    yticks = -5:1:4,
    yminorticks=2,
    legend = :outerright,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size = (1920, 1080),
    legendtitlefontsize = 32,
    legendfontsize = 32,
    guidefontsize = 32,
    tickfontsize = 30,
    left_margin = 12mm,
    bottom_margin =12mm,
);
hline!([0.0], line=(:dot,1,:black), label=false);
for (idx, color, marker_symbol, config_label, tol) in plot_list

    pair = pair_of_idx[idx]
    y0   = y0_pair_mean[pair].y0_mean

    y = split_itp[idx].(Iexp_sampled) .- y0
    dy = errsplit_itp[idx].(Iexp_sampled)

    plot!(
        plt,
        Iexp_sampled,
        y,
        yerror = dy,
        label = data_directories[idx] * " $(config_label)",
        color = color,
        line = (:solid, 2),
        marker = (marker_symbol, 10, color, 0.6),
        markerstrokecolor = color,
    )


end
display(plt)
plot!(plt,
    xlims=(100e-3,4),
    xscale=:log10
);
display(plt)




# Discussed with Dr. Wamg

fig_linlin_1 = plot(
    xlabel = "SG0 Current (A)",
    ylabel = L"$\Delta z = z_{F=1} - z_{F=2} \quad (\mathrm{px})$",
);
ref_idxs = (3, 4)
plot_list = [
    # (3, :darkgreen,     :rect,    "(↑↓)", 1e-6),
    # (4, :darkgreen,     :diamond, "(↑↑)", 1.5e-3),

    # (1, :purple,     :rect,    "(↑↓)", 1e-6),
    # (2, :purple,     :diamond, "(↑↑)", 1.5e-3),

    # (6, :dodgerblue, :rect,    "(↑↓)", 1e-6),
    # (5, :dodgerblue, :diamond, "(↑↑)", 1.5e-3),

    (8, :orange,     :rect, "(↑↓)", 1.5e-3),
    (7, :orange,     :diamond,    "(↑↑)", 1e-6),
];
for (idx, color, marker_symbol, config_label, tol) in plot_list

    df = copy(idx in ref_idxs ? sg0_ref[idx] : sg0_data[idx])

    @info "Processing dataset" idx
    println()
    show(stdout, df)
    println()

    # Keep only rows with nonzero I1, except for reference datasets
    if idx ∉ ref_idxs
        df = df[abs.(df.I1) .> tol, :]
    end

    # Dataset-specific row removals
    if idx == 1
        df = df[Not(14), :]
    elseif idx == 8
        df = df[1:end-1, :]
    end

    # Subtract first point as reference
    split0 = df[1, :split]
    err0   = df[1, :errsplit]

    df[!, :split]    .-= split0
    df[!, :errsplit] .= sqrt.(df[!, :errsplit].^2 .+ err0^2)

    sort!(df, :I0)

    println()
    show(stdout, df)
    println()

    plot!(fig_linlin_1,
        df.I0,
        df.split ./ scale_factor,
        yerror = df.errsplit ./ scale_factor,
        label = dirname(data_directories[idx]) * " " * config_label * " " *
                L" $I_{1} = %$(round(1000 * mean(df.I1); digits=1)) \mathrm{mA}$",
        marker = (marker_symbol, 8, color, 0.60),
        markerstrokecolor = color,
        line = (:solid, 2, color),
    )
end
ymin, ymax = ylims(fig_linlin_1)
plot!(fig_linlin_1,
    ylims = (floor(ymin), ceil(ymax)),
    yticks = floor(ymin):1:ceil(ymax),
);
plot!(fig_linlin_1,
    legendtitle = "",
    legend = :bottomleft,
    legend_columns = 1,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size = (1920, 1080),
    legendtitlefontsize = 18,
    legendfontsize = 18,
    guidefontsize = 24,
    tickfontsize = 18,
    left_margin = 12mm,
    bottom_margin =12mm,
);
display(fig_linlin_1)
plot!(fig_linlin_1,
    xlims=(4e-3,4),
    xscale=:log10,
);
display(fig_linlin_1)



plot_list = [
    (7, :red,     :diamond, "(↑↑)", 1.5e-3),
    (8, :blue,    :rect,    "(↑↓)", 1e-6),
];
# Pairs used to define the common offset y0
pairs = [
    (8, 7)
];

# Map each index to its pair
pair_of_idx = Dict(idx => pair for pair in pairs for idx in pair)

plt = plot(
    xlabel = "SG0 current (A)",
    ylabel = "Spin-resolved beam separation (pixels)",
    yticks = -5:1:4,
    yminorticks=2,
    legend = :bottomleft,
    background_color_legend = nothing,
    foreground_color_legend = nothing,
    size = (1920, 1080),
    legendtitlefontsize = 32,
    legendfontsize = 24,
    guidefontsize = 32,
    tickfontsize = 30,
    left_margin = 12mm,
    bottom_margin =12mm,
);
hline!([0.0], line=(:dot,1,:black), label=false);
for (idx, color, marker_symbol, config_label, tol) in plot_list

    pair = pair_of_idx[idx]
    y0   = y0_pair_mean[pair].y0_mean

    y = split_itp[idx].(Iexp_sampled) .- y0
    dy = errsplit_itp[idx].(Iexp_sampled)

    if idx == 7
    plot!(
        plt,
        Iexp_sampled,
        y,
        yerror = dy,
        label = "↑↑ (parallel)",
        color = color,
        line = (:solid, 2),
        marker = (marker_symbol, 18, color, 0.6),
        markerstrokecolor = color,
    )
    else
    plot!(
        plt,
        Iexp_sampled,
        y,
        yerror = dy,
        label = "↑↓ (antiparallel)",
        color = color,
        line = (:solid, 2),
        marker = (marker_symbol, 10, color, 0.6),
        markerstrokecolor = color,
    )
    end

end
display(plt)
savefig(plt, "spin_resolved_sep.pdf")
savefig(plt, "spin_resolved_sep.svg")






2+2











































































































#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_directories = [
        "20260318/Round1",
        "20260318/Round2",
        "20260318/Round3",
        "20260318/Round4",
        # "20260415A", 
        # "20260415B", 
        # "20260416A", 
        # "20260416B",
        # "20260427A",
        # "20260427B",
        # "20260427C",
        # "20260429A",
        # "20260429B",
        # "20260429C",
        # "20260429D",
        # "20260429E",
        # "20260429F",
        # "20260504/Round1_SG1=0mA_SG0=+MG=3A",
        # "20260504/Round2_SG1=0mA_SG0=-MG=3A",
        # "20260504/Round3_SG1=40mA_SG0=+MG=3A",
        # "20260504/Round4_SG1=40mA_SG0=-MG=3A",
        # "20260504/Round5_SG1=500mA_SG0=+MG=3A",
        # "20260504/Round6_SG1=500mA_SG0=-MG=3A",
        # "20260507/Round1_SG1=278mA_SG0=+_MG=2300mA",
        # "20260507/Round2_SG1=278mA_SG0=-_MG=2300mA",
        # "20260507/Round3_SG1=279mA_SG0=+_MG=2020mA",
        # "20260507/Round4_SG1=279mA_SG0=-_MG=2020mA",
        # "20260507/Round5_SG1=279mA_SG0=+_MG=1530mA",
        # "20260507/Round6_SG1=279mA_SG0=-_MG=1530mA",
        # "20260507/Round7_SG1=278mA_SG0=+_MG=1070mA",
        # "20260507/Round8_SG1=278mA_SG0=-_MG=1070mA",
        # "20260507/Round9_SG1=279mA_SG0=+_MG=540mA",
        # "20260507/Round10_SG1=279mA_SG0=-_MG=540mA",
        # "20260513/Round1_SG1=40mA_SG0=+_MG=40G",
        # "20260513/Round2_SG1=40mA_SG0=-_MG=40G",
        # "20260515/Round2_SG1=223mA_SG0=+_MG=40G",
        # "20260515/Round3_SG1=223mA_SG0=-_MG=40G"
        ];
nd = length(data_directories);

for data_directory in data_directories
    printstyled("\t" * data_directory * "\n"; color=:cyan, bold=true)
    outfile_raw         = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data.jld2")
    outfile_processed   = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2")
    data_summary_path   = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS","summary",data_directory)
    isdir(data_summary_path) || mkpath(data_summary_path);


    if !isfile(outfile_processed) # check if the processed images exists
        if !isfile(outfile_raw) # check if the raw data exists
            @info "Not found → building $outfile_raw"
            data_in = joinpath(BASE_PATH,"EXPERIMENTS", data_directory)
            data_raw = MyExperimentalAnalysis.SG0_stack_data(data_in)
            jldsave(outfile_raw, data=data_raw)
            data_raw = nothing
        else
            @info "Found $outfile_raw → skipping build"
        end

        data_raw = load(outfile_raw)["data"]
        data_processed = MyExperimentalAnalysis.SG0_build_processed_dict(data_raw)
        jldsave(outfile_processed, data=data_processed)
        data_processed = nothing
        data_raw = nothing
    else
        @info "Found $outfile_processed → skipping build"
    end

    println("\n$(data_directory) PROCESSING COMPLETED\n\n")
end



tables = Vector{DataFrame}(undef, nd);
jldopen(joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS", "summary", "data_analysis_$(RUN_STAMP).jld2"), "w") do f
    f["meta/nz"]               = nz
    f["meta/λ0"]               = λ0
    f["meta/data_directories"] = data_directories

    for (idx,data_directory) in enumerate(data_directories)
        # idx = 2
        # data_directory = data_directories[idx]
        printstyled("\t" * data_directory * "\n"; color=:cyan, bold=true);
        data_processed = load(joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2"))["data"];

        OUTDIR    = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
        isdir(OUTDIR) || mkpath(OUTDIR);
        @info "Created output directory" OUTDIR
        MyExperimentalAnalysis.OUTDIR   = OUTDIR;

        # ── Currents & fields ────────────────────────────────────────────────────
        SG0_current = data_processed[:SG0Currents];
        SG1_current = data_processed[:SG1Currents];
        # MG_current  = data_processed[:MGCurrents];
        
        Bz0 = 1e3 .* data_processed[:SG0Bz];
        Bz1 = 1e3 .* data_processed[:SG1Bz];
        # MG_fields   = 1e3 * data_processed[:MGFields];

        # ── Framewise maxima & statistics ────────────────────────────────────────
        f1_max = MyExperimentalAnalysis.SG0_framewise_maxima("F1", data_processed, nz ; half_max=false, λ0=λ0);
        f2_max = MyExperimentalAnalysis.SG0_framewise_maxima("F2", data_processed, nz ; half_max=false, λ0=λ0);

        # MyExperimentalAnalysis.SG0_mean_maxima("F1", data_processed, nz ; half_max=false,λ0=λ0);
        # MyExperimentalAnalysis.SG0_mean_maxima("F2", data_processed, nz ; half_max=false,λ0=λ0);

        f1_z_mm , f1_z_sem_mm  = vec(mean(f1_max, dims=1)) , sqrt.(vec(std(f1_max, dims=1; corrected=true) ./ sqrt(size(f1_max,1))).^2 .+ z_mm_error^2 );
        f2_z_mm , f2_z_sem_mm  = vec(mean(f2_max, dims=1)) , sqrt.(vec(std(f2_max, dims=1; corrected=true) ./ sqrt(size(f2_max,1))).^2 .+ z_mm_error^2 );

        Δz_mm = -(f1_z_mm .- f2_z_mm);
        Δz_sem_mm = sqrt.( (f1_z_sem_mm).^2 .+ (f2_z_sem_mm).^2 );

        # ── Summary DataFrame ────────────────────────────────────────────────────
        data = DataFrame(
            Ig        = MG_current,
            I0        = SG0_current,
            I1        = SG1_current,
            Bg        = MG_fields,
            B0        = Bz0,
            B1        = Bz1,
            zf1       = f1_z_mm,
            errzf1    = f1_z_sem_mm,
            zf2       = f2_z_mm,
            errzf2    = f2_z_sem_mm,
            split     = Δz_mm,
            errsplit  = round.(Δz_sem_mm; sigdigits=1)
        )
        tables[idx] = data
        f[data_directory] = data  # ← save immediately
        @info "Saved table" data_directory

        pretty_table(data;
                title         = data_directory,
                formatters    = [ fmt__printf("%8.3f", [1]), fmt__printf("%8.5f", 2:3), fmt__printf("%8.3f", 4:6),  fmt__printf("%8.3f", [7,9,11]), fmt__printf("%8.4f", [8,10,12])],
                alignment     = :c,
                column_labels  = [
                    ["IG current", "I0 Current", "I1 Current", "BG field", "B0 field", "B1 field", "F1", "F1 err", "F2", "F2 err", "Δz", "Δz err"], 
                    ["[A]", "[A]", "[A]", "[mT]" ,"[mT]", "[mT]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]", "[mm]"]
                ],
                table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
                style = TextTableStyle(
                            first_line_column_label = crayon"yellow bold",
                            column_label  = crayon"yellow",
                            table_border  = crayon"blue bold",
                            title = crayon"bold red"
                            ),
                equal_data_column_widths = true,
        )


        data_pos = data[data.I0 .> 0, :]
        # ── Colour palette (one colour per SG0 current step) ─────────────────────
        colors_sg0 = palette(:darkrainbow, size(data,1));

        sg1_label = "$(round(1000 * data.I1[end], digits=2))mA"   # reused in titles

        # ── Overlay of all SG0 currents: z-position vs current ───────────────────
        common_scatter_kw = (seriestype=:scatter, marker=(:circle, 2, :white))

        fig1 = plot(data.I0, data.zf1;
            common_scatter_kw..., yerror=data.errzf1,
            label=L"$F=1$", markerstrokecolor=:red);
        plot!(fig1, data.I0, data.zf2;
            common_scatter_kw..., yerror=data.errzf2,
            label=L"$F=2$", markerstrokecolor=:blue);
        plot!(fig1, data.I0, mean([data.zf1, data.zf2]);
            label="Centre", marker=(:diamond, :white),
            markerstrokecolor=:gray47, line=(:dash, 1, :gray47));
        plot!(fig1;
            xlabel="SG0 Current (A)",
            foreground_color_legend=nothing, background_color_legend=nothing,
            yformatter=y -> @sprintf("%.3f", y));

        fig2 = plot(data_pos.I0, data_pos.zf1;
            common_scatter_kw..., yerror=data_pos.errzf1,
            label=L"$F=1$", markerstrokecolor=:red);
        plot!(fig2, data_pos.I0, data_pos.zf2;
            common_scatter_kw..., yerror=data_pos.errzf2,
            label=L"$F=2$", markerstrokecolor=:blue);
        plot!(fig2, data_pos.I0, mean([data_pos.zf1, data_pos.zf2]),
            label="Centre", marker=(:diamond, :white),
            markerstrokecolor=:gray47, line=(:dash, 1, :gray47));
        plot!(fig2;
            xscale=:log10,
            foreground_color_legend=nothing, background_color_legend=nothing,
            xlabel="SG0 Current (A)",
            yformatter=y -> @sprintf("%.3f", y));

        fig3 = plot(data.I0, data.split ./ 6.5e-3;
            common_scatter_kw..., yerror=data.errsplit ./ 6.5e-3,
            label=L"$\Delta z$", markerstrokecolor=:darkgreen);
        plot!(fig3;
            foreground_color_legend=nothing, background_color_legend=nothing,
            xlabel="SG0 Current (A)", ylabel="Separation (px)", yminorticks=false);

        fig4 = plot(data_pos.I0, data_pos.split ./ 6.5e-3;
            common_scatter_kw..., yerror=data_pos.errsplit ./ 6.5e-3,
            label=L"$\Delta z$", markerstrokecolor=:darkgreen);
        plot!(fig4;
            xscale=:log10,
            foreground_color_legend=nothing, background_color_legend=nothing,
            xlabel="SG0 Current (A)", ylabel="Separation (px)", yminorticks=false);

        fig = plot(fig1, fig2, fig3, fig4;
            suptitle     = "$(data_directory) | SG1: $sg1_label",
            layout       = (2, 2),
            size         = (1000, 600),
            link         = :x,
            left_margin  = 5mm,
            bottom_margin= 3mm,
        )
        plot!(fig[1]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
        plot!(fig[2]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
        display(fig)
        
        # ── Extract image arrays once, outside all loops ─────────────────────────
        f1imgs = data_processed[:F1ProcessedImages];
        f2imgs = data_processed[:F2ProcessedImages];
        current_label(i) = L"$%$(round(1000*SG0_current[i]; digits=3))\mathrm{mA}$";
        sg1_label    = "$(data_directory) | SG1: $(round(1000*data.I1[end], digits=2))mA";

        legend_kw = (
            legend                  = :topleft,
            legend_title            = "SG0",
            background_color_legend = :white,
            foreground_color_legend = nothing,
        );

        # ── Pre-compute cleaned images and profiles for all currents ──────────────
        # Store so each quantity is computed once and reused in both plot blocks
        f1_images   = Vector{Matrix{Float64}}(undef, length(SG0_current));
        f2_images   = Vector{Matrix{Float64}}(undef, length(SG0_current));
        f1_profiles = Vector{Vector{Float64}}(undef, length(SG0_current));
        f2_profiles = Vector{Vector{Float64}}(undef, length(SG0_current));


        for i in eachindex(SG0_current)
            # Average over frames (dim 3), then mask
            f1img = dropdims(mean(f1imgs[:,:,:,i], dims=3), dims=3)
            f1img .*= isfinite.(f1img) .& (f1img .>= -10) .& (f1img .<= 1000)

            f2img = dropdims(mean(f2imgs[:,:,:,i], dims=3), dims=3)
            f2img .*= isfinite.(f2img) .& (f2img .>=  -5) .& (f2img .<= 500)

            f1_images[i]   = f1img
            f2_images[i]   = f2img

            # Profile: average over x (dim 1) after frame-averaging and masking
            f1_profiles[i] = vec(mean(f1img, dims=1))
            f2_profiles[i] = vec(mean(f2img, dims=1))

        end

        # ── Per-current panels: heatmap (left) + z-profile (right) ───────────────
        for i in eachindex(SG0_current)
            f1vmax = Statistics.quantile(vec(f1_images[i]), 0.999)
            f2vmax = Statistics.quantile(vec(f2_images[i]), 0.999)

            plt1 = heatmap(f1_images[i];
                xlabel = L"$z\ \ (\mathrm{px})$",
                ylabel = L"$x\ \ (\mathrm{px})$",
                cbar   = true, clims = (0, f1vmax),
            )
            plt2 = plot(f1_profiles[i];
                line = (:solid, 1, colors_sg0[i]), label = current_label(i),
                xlabel = L"$z\ \ (\mathrm{px})$", legend_kw...,
            )
            plt3 = heatmap(f2_images[i];
                xlabel = L"$z\ \ (\mathrm{px})$",
                ylabel = L"$x\ \ (\mathrm{px})$",
                cbar   = true, clims = (0, f2vmax),
            )
            plt4 = plot(f2_profiles[i];
                line = (:solid, 1, colors_sg0[i]), label = current_label(i),
                xlabel = L"$z\ \ (\mathrm{px})$", legend_kw...,
            )

            plt = plot(plt1, plt2, plt3, plt4;
                suptitle      = "$(data_directory) | SG1: $(round(1000*data.I1[i], digits=2))mA",
                layout        = (2, 2),
                link          = :x,
                size          = (800, 450),
                left_margin   = 3mm,
                bottom_margin = 2mm,
            )
            plot!(plt[1]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
            plot!(plt[2]; xlabel="", xformatter=_->"", bottom_margin=-5mm)
            display(plt)
        end

        # ── Overlay: all currents on one z-profile plot per state ─────────────────
        overlay_kw = (
            legend                  = :topleft,
            legend_font             = 8,
            foreground_color_legend = nothing,
            legend_columns          = 2,
            xlabel                  = L"$z \ \ (\mathrm{px})$",
        );

        plt_f1 = plot(; overlay_kw...);
        plt_f2 = plot(; overlay_kw...);

        for i in eachindex(SG0_current)
            plot!(plt_f1, f1_profiles[i]; line=(:solid, 1, colors_sg0[i]), label=current_label(i))
            plot!(plt_f2, f2_profiles[i]; line=(:solid, 1, colors_sg0[i]), label=current_label(i))
        end
        
        plot!(plt_f1; legend_title=L"$F=1$");
        plot!(plt_f2; legend_title=L"$F=2$");

        fig = plot(plt_f1, plt_f2;
            suptitle      = sg1_label,
            layout        = (2, 1),
            size          = (900, 850),
            left_margin   = 3mm,
            bottom_margin = 2mm,
        )
        plot!(fig[1]; xlabel="", xformatter=_->"", bottom_margin=-6mm)
        display(fig)

        GC.gc()
    end
end


function plot_sg1_fig(
    indices::UnitRange{Int},
    sg1_current_mA::Float64;
    tables = tables,
    data_directories = data_directories,
    mark_symbol = [:circle, :rect],
    colores = (:darkgreen, :indigo),
    current_polarity = ("+", "–"),
    scale_factor = 6.5e-3,
)
    offset = first(indices) - 1
    local_dirs = data_directories[indices]

    # Helper: optionally filter by positive SG1 current
    sg1_filter(df) = iszero(sg1_current_mA) ? df : df[df.I1 .> 0, :]

    # ── fig1a : log scale, positive current only ──────────────────────────────
    fig1a = plot(xlabel = "SG0 Current (A)", ylabel = L"$\Delta z$ (px)")
    for (k, dir) in enumerate(local_dirs)
        data = tables[k + offset]
        data_pos = sg1_filter(data[data.I0 .> 0, :])
        plot!(fig1a,
            data_pos.I0, data_pos.split ./ scale_factor,
            yerror            = data_pos.errsplit ./ scale_factor,
            label             = "($(current_polarity[k])) " * dir,
            line              = (:solid, 1, colores[k], 0.5),
            marker            = (mark_symbol[k], 2, :white),
            markerstrokecolor = colores[k],
        )
    end
    plot!(fig1a,
        xscale                  = :log10,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        yminorticks             = 5,
    )

    # ── fig1b : linear, zoomed to I0 < 50 mA ─────────────────────────────────
    fig1b = plot(xlabel = "SG0 Current (mA)", ylabel = L"$\Delta z$ (px)")
    for (k, dir) in enumerate(local_dirs)
        data = tables[k + offset]
        data_pos = sg1_filter(data[data.I0 .< 0.050, :])
        plot!(fig1b,
            1000 .* data_pos.I0, data_pos.split ./ scale_factor,
            label = false,
            line  = (:dashdot, 1, colores[k], 0.5),
        )
        plot!(fig1b,
            1000 .* data_pos.I0, data_pos.split ./ scale_factor,
            yerror            = data_pos.errsplit ./ scale_factor,
            seriestype        = :scatter,
            label             = "($(current_polarity[k])) " * dir,
            marker            = (mark_symbol[k], 2, :white),
            lw                = 2,
            markerstrokecolor = colores[k],
        )
    end
    plot!(fig1b,
        xlims                   = (-1, 10),
        legend                  = :bottomright,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        yminorticks             = 5,
    )

    # ── fig1c : full current range ────────────────────────────────────────────
    fig1c = plot(xlabel = "SG0 Current (A)", ylabel = L"$\Delta z$ (px)")
    for (k, dir) in enumerate(local_dirs)
        data = tables[k + offset]
        data_pos = sg1_filter(data)
        plot!(fig1c,
            data_pos.I0, data_pos.split ./ scale_factor,
            label = false,
            line  = (:dashdot, 1, colores[k], 0.5),
        )
        plot!(fig1c,
            data_pos.I0, data_pos.split ./ scale_factor,
            yerror            = data_pos.errsplit ./ scale_factor,
            seriestype        = :scatter,
            label             = "($(current_polarity[k])) " * dir,
            marker            = (mark_symbol[k], 2, :white),
            lw                = 2,
            markerstrokecolor = colores[k],
        )
    end
    plot!(fig1c,
        legend                  = :bottomright,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        yminorticks             = 5,
    )

    # ── composite figure ──────────────────────────────────────────────────────
    fig = plot(fig1a, fig1b, fig1c,
        layout       = @layout([a1 a2; a3]),
        size         = (850, 550),
        link         = :y,
        left_margin  = 3mm,
    )
    plot!(fig[1], legend = false, title = "")
    plot!(fig[2], ylabel = "", yformatter = _ -> "", left_margin = -4mm,
          legend = false, title = "")
    plot!(fig[3],
        title          = "",
        bottom_margin  = -7mm,
        legend_title   = L"SG1 $\sim %$(sg1_current_mA)\mathrm{mA}$",
        legend_columns = 2,
        legend         = :outerbottom,
    )

    # display(fig)
    return fig
end



plot_sg1_fig(
    1:2,
    40.0;
)

plot_sg1_fig(
    3:4,
    223.0;
)

plot_sg1_fig(
    5:6,
    500.0;
)

plot_sg1_fig(
    7:8,
    278.0;
)

plot_sg1_fig(
    9:10,
    279.0;
)

plot_sg1_fig(
    11:12,
    279.0;
)

plot_sg1_fig(
    13:14,
    278.0;
)

plot_sg1_fig(
    15:16,
    279.0;
)


function plot_sg0_sweep(
    table_indices::AbstractVector{Int};
    tables        = tables,
    cam_pixelsize = cam_pixelsize,
    mark_symbols  = (:circle, :rect, :diamond, :utriangle, :dtriangle, :circle, :rect, :diamond, :utriangle, :dtriangle, :xcross, :cross),
    color_list    = palette(:darkrainbow, length(table_indices)),
)
    scale = 1e3 * cam_pixelsize

    function make_label(df)
        sg1 = Int(round(1000 * df.I1[1]))
        mg  = Int(round(1000 * df.Ig[1]))
        L"SG1=$%$(sg1)\mathrm{mA}$ - MG=$%$(mg)\mathrm{mA}$"
    end

    function plot_series!(ax, df, i; x_slice=Colon(), connect_slice=Colon())
        scatter!(ax,
            df.I0[x_slice], df.split[x_slice] ./ scale,
            yerror           = df.errsplit[x_slice] ./ scale,
            marker           = (mark_symbols[i], :white, 2),
            markerstrokecolor = color_list[i],
            label            = make_label(df),
        )
        plot!(ax,
            df.I0[connect_slice], df.split[connect_slice] ./ scale,
            line  = (:solid, 1, color_list[i], 0.25),
            label = false,
        )
    end

    legend_kw = (
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        yminorticks             = 5,
        legend                  = :outerright,
    )

    # ── a1 : full range ───────────────────────────────────────────────────────
    a1 = plot(xlabel = "SG0 (A)", ylabel = "Separation (px)")
    for (i, idx) in enumerate(table_indices)
        df = tables[idx]
        plot_series!(a1, df, i; x_slice = 2:nrow(df), connect_slice = 2:nrow(df))
    end
    plot!(a1; xscale=:identity, legend_kw...)

    # ── a2 : zoomed, I0 < 50 mA ───────────────────────────────────────────────
    a2 = plot(xlabel = "SG0 (A)", ylabel = "Separation (px)")
    for (i, idx) in enumerate(table_indices)
        df = tables[idx][tables[idx].I0 .< 0.050, :]
        plot_series!(a2, df, i)
    end
    plot!(a2; xscale = :identity, legend_kw...)

    # ── a3 : log scale, skip first point ─────────────────────────────────────
    a3 = plot(xlabel = "SG0 (A)", ylabel = "Separation (px)")
    for (i, idx) in enumerate(table_indices)
        df = tables[idx]
        plot_series!(a3, df, i)
    end
    plot!(a3; xscale = :identity, legend_kw...)

    # ── composite ─────────────────────────────────────────────────────────────
    # Build legend title from first table as representative
    df_first = tables[first(table_indices)]
    leg_title = make_label(df_first)

    fig = plot(a1, a2, a3,
        layout       = @layout([a1 a2; a3]),
        size         = (850, 550),
        link         = :y,
        left_margin  = 3mm,
    )
    plot!(fig[1], legend = false, title = "")
    plot!(fig[2], ylabel = "", yformatter = _ -> "", left_margin = -4mm,
          legend = false, title = "")
    plot!(fig[3],
        title          = "",
        bottom_margin  = -7mm,
        legend_columns = 3,
        legend         = :outerbottom,
    )

    display(fig)
    return fig
end

plot_sg0_sweep([1,2])
plot_sg0_sweep([3,4])

plot_sg0_sweep([7,9,11,13,15])

plot_sg0_sweep([8,10,12,14,16])

plot_sg0_sweep([7,8,9,10,11,12,13,14,15,16])

f = jldopen(joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS", "summary", "data_analysis_$(RUN_STAMP).jld2"), "r")
nz               = f["meta/nz"]
λ0               = f["meta/λ0"]
data_directories = f["meta/data_directories"]
tables2           = [f[dir] for dir in data_directories]
close(f)

tables2

table3 = jldopen(joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS", "summary", "data_analysis_$(RUN_STAMP).jld2"), "r") do f
    f["20260504/Round1_SG1=0mA_SG0=+MG=3A"]
end

tables4 = jldopen(joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS", "summary", "data_analysis_$(RUN_STAMP).jld2"), "r") do f
    # Recover metadata
    nz               = f["meta/nz"]
    λ0               = f["meta/λ0"]
    data_directories = f["meta/data_directories"]

    # Rebuild tables in the same order
    return tables = [f[dir] for dir in data_directories]
end






mark_symbol = (:circle, :rect, :diamond, :utriangle )
colores = ((:red,:blue), (:orangered, :dodgerblue), (:darkgreen, :indigo) , (:sienna, :lime))
plot(xlabel="SG0 Current (A)",
    ylabel="position (mm)")
for (i,dir) in enumerate(data_directories[1:4])
    data = tables[i]
    data_pos = data[data.I0 .>= 0, :]
    plot!(data_pos.I0, data_pos.zf1, 
        yerror = data_pos.errzf1,
        label=L"$F=1$",
        seriestype=:scatter,
        marker=(mark_symbol[i],2,:white),
        markerstrokecolor=colores[i][1])
    plot!(data_pos.I0, data_pos.zf2,
        yerror = data_pos.errzf2,
        label=L"$F=2$",
        seriestype=:scatter,
        marker=(mark_symbol[i],2,:white),
        markerstrokecolor=colores[i][2])
    plot!(data_pos.I0, mean([data_pos.zf1,data_pos.zf2]),
        label="Centre",
        marker=(:xcross,:gray47),
        line=(:dash,1,:gray47)  
    )
end
plot!(
    # xscale=:log10,
    yformatter = y -> @sprintf("%.3f", y),
    legend=:outerright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    yminorticks=false,)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_directory = "20260415A"

OUTDIR    = joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

outfile_raw         = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data.jld2")
outfile_processed   = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2")
data_summary_path   = joinpath(BASE_PATH, "SG0_EXPDATA_ANALYSIS","summary",data_directory)
isdir(data_summary_path) || mkpath(data_summary_path);


if !isfile(outfile_processed) # check if the processed images exists
    if !isfile(outfile_raw) # check if the raw data exists
        @info "Not found → building $outfile_raw"
        data_in = joinpath(BASE_PATH,"EXPERIMENTS", data_directory)
        data_raw = MyExperimentalAnalysis.SG0_stack_data(data_in)
        jldsave(outfile_raw, data=data_raw)
        data_raw = nothing
    else
        @info "Found $outfile_raw → skipping build"
    end

    data_raw = load(outfile_raw)["data"]
    data_processed = MyExperimentalAnalysis.SG0_build_processed_dict(data_raw)
    jldsave(outfile_processed, data=data_processed)
    data_processed = nothing
    data_raw = nothing
else
    @info "Found $outfile_processed → skipping build"
end
data_processed = load(outfile_processed)["data"]


SG0_current = data_processed[:SG0Currents]
SG1_current = data_processed[:SG1Currents]

f1_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F1", data_processed, nz; half_max=true, λ0=λ0);
f2_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F2", data_processed, nz; half_max=true, λ0=λ0);

f1_max = MyExperimentalAnalysis.SG0_framewise_maxima("F1", data_processed, nz ; half_max=true,λ0=λ0);
f2_max = MyExperimentalAnalysis.SG0_framewise_maxima("F2", data_processed, nz ; half_max=true,λ0=λ0);

f1_z_mm , f1_z_sem_mm  = vec(mean(f1_max, dims=1)) , sqrt.(vec(std(f1_max, dims=1; corrected=true) ./ sqrt(size(f1_max,1))).^2 .+ z_mm_error^2 );
f2_z_mm , f2_z_sem_mm  = vec(mean(f2_max, dims=1)) , sqrt.(vec(std(f2_max, dims=1; corrected=true) ./ sqrt(size(f2_max,1))).^2 .+ z_mm_error^2 );

data = hcat(SG0_current,SG1_current, f1_z_mm, f1_z_sem_mm, f2_z_mm, f2_z_sem_mm)
cols = palette(:darkrainbow, size(data,1))   # generate colors


data01 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
fig01 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data01,1)
    plot!(fig01,
        [1000*data01[i,2]], [data01[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c0} = %$(round(data01[i,1];digits=3))\mathrm{A}$")
end
plot!(fig01,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig01)


data02 = DataReading.subset_by_cols(data01,[2]; thr = 1e-6, include_equal=true )[3]
fig02 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data02,1)
    plot!(fig02,
        [1000*data02[i,2]], [data02[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data02[i,4]],
        label=L"$I_{c0} = %$(round(data02[i,1];digits=3))\mathrm{A}$")
end
plot!(fig02,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig02)


data03 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
zcenter = data03[1,3]
data03[:,3] = data03[:,3] .- zcenter
data03 = DataReading.subset_by_cols(data03,[2]; thr = 1e-6, include_equal=true )[3]
fig03 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data03,1)
    plot!(fig03,
        [1000*data03[i,2]], [data03[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data03[i,4]],
        label=L"$I_{c0} = %$(round(data03[i,1];digits=3))\mathrm{A}$")
end
plot!(fig03,
    yscale=:log10,
    ylims=(5e-2,1e-1),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig03)

p1 = plot(fig01,fig02,fig03,
layout=@layout([a ; b ; c]),
size=(900,800)
)
saveplot(p1,"zvssg1")


data04 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
fig04 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data04,1)
    plot!(fig04,
        [data04[i,1]], [data04[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c1} = %$(round(1000*data04[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig04,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig04)


data05 = DataReading.subset_by_cols(data04,[1]; thr = 1e-6, include_equal=true )[3]
fig05 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data05,1)
    plot!(fig05,
        [data05[i,1]], [data05[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data05[i,4]],
        label=L"$I_{c1} = %$(round(1000*data05[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig05,
    xscale=:log10,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig05)


data06 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
zcenter = data06[1,3]
data06[:,3] = data06[:,3] .- zcenter
data06 = DataReading.subset_by_cols(data06,[1]; thr = 1e-6, include_equal=true )[3]
fig06 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data06,1)
    plot!(fig06,
        [data06[i,1]], [data06[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data06[i,4]],
        label=L"$I_{c1} = %$(round(1000*data06[i,2];digits=3))\mathrm{mA}$")
end
plot!(fig06,
    xscale=:log10,
    yscale=:log10,
    ylims=(5e-2,1e-1),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig06)

p2 = plot(fig04,fig05,fig06,
layout=@layout([a ; b ; c]),
size=(900,800)
)
saveplot(p2,"zvssg0")



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_directory = "20260312A"

OUTDIR    = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

outfile_raw         = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data.jld2")
outfile_processed   = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2")
data_summary_path   = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS","summary",data_directory)
isdir(data_summary_path) || mkpath(data_summary_path);


if !isfile(outfile_processed) # check if the processed images exists
    if !isfile(outfile_raw) # check if the raw data exists
        @info "Not found → building $outfile_raw"
        data_in = joinpath(BASE_PATH,"EXPERIMENTS", data_directory)
        data_raw = MyExperimentalAnalysis.SG0_stack_data(data_in)
        jldsave(outfile_raw, data=data_raw)
        data_raw = nothing
    else
        @info "Found $outfile_raw → skipping build"
    end

    data_raw = load(outfile_raw)["data"]
    data_processed = MyExperimentalAnalysis.SG0_build_processed_dict(data_raw)
    jldsave(outfile_processed, data=data_processed)
    data_processed = nothing
    data_raw = nothing
else
    @info "Found $outfile_processed → skipping build"
end
data_processed = load(outfile_processed)["data"]


SG0_current = data_processed[:SG0Currents]
SG1_current = data_processed[:SG1Currents]

f1_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F1", data_processed, nz; half_max=true, λ0=λ0);
f2_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F2", data_processed, nz; half_max=true, λ0=λ0);

f1_max = MyExperimentalAnalysis.SG0_framewise_maxima("F1", data_processed, nz ; half_max=true,λ0=λ0);
f2_max = MyExperimentalAnalysis.SG0_framewise_maxima("F2", data_processed, nz ; half_max=true,λ0=λ0);

f1_z_mm , f1_z_sem_mm  = vec(mean(f1_max, dims=1)) , sqrt.(vec(std(f1_max, dims=1; corrected=true) ./ sqrt(size(f1_max,1))).^2 .+ z_mm_error^2 );
f2_z_mm , f2_z_sem_mm  = vec(mean(f2_max, dims=1)) , sqrt.(vec(std(f2_max, dims=1; corrected=true) ./ sqrt(size(f2_max,1))).^2 .+ z_mm_error^2 );


data = hcat(SG0_current,SG1_current, f1_z_mm, f1_z_sem_mm, f2_z_mm, f2_z_sem_mm)
cols = palette(:darkrainbow, size(data,1))   # generate colors


data01 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
fig01 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data01,1)
    plot!(fig01,
        [1000*data01[i,2]], [data01[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c0} = %$(round(data01[i,1];digits=3))\mathrm{A}$")
end
plot!(fig01,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig01)


data02 = DataReading.subset_by_cols(data01,[2]; thr = 1e-6, include_equal=true )[3]
fig02 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data02,1)
    plot!(fig02,
        [1000*data02[i,2]], [data02[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data02[i,4]],
        label=L"$I_{c0} = %$(round(data02[i,1];digits=3))\mathrm{A}$")
end
plot!(fig02,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig02)


data03 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
zcenter = data03[1,3]
data03[:,3] = data03[:,3] .- zcenter
data03 = DataReading.subset_by_cols(data03,[2]; thr = 1e-6, include_equal=true )[3]
fig03 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data03,1)
    plot!(fig03,
        [1000*data03[i,2]], [data03[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data03[i,4]],
        label=L"$I_{c0} = %$(round(data03[i,1];digits=3))\mathrm{A}$")
end
plot!(fig03,
    yscale=:log10,
    ylims=(1e-2,1e-1),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig03)

p1 = plot(fig01,fig02,fig03,
layout=@layout([a ; b ; c]),
size=(900,800)
)
saveplot(p1,"zvssg1")


data04 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
fig04 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data04,1)
    plot!(fig04,
        [data04[i,1]], [data04[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c1} = %$(round(1000*data04[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig04,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig04)


data05 = DataReading.subset_by_cols(data04,[1]; thr = 1e-6, include_equal=true )[3]
fig05 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data05,1)
    plot!(fig05,
        [data05[i,1]], [data05[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data05[i,4]],
        label=L"$I_{c1} = %$(round(1000*data05[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig05,
    xscale=:log10,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig05)


data06 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
zcenter = data06[1,3]
data06[:,3] = data06[:,3] .- zcenter
data06 = DataReading.subset_by_cols(data06,[1]; thr = 1e-6, include_equal=true )[3]
fig06 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data06,1)
    plot!(fig06,
        [data06[i,1]], [data06[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data06[i,4]],
        label=L"$I_{c1} = %$(round(1000*data06[i,2];digits=3))\mathrm{mA}$")
end
plot!(fig06,
    xscale=:log10,
    yscale=:log10,
    ylims=(3e-2,1e-1),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig06)

p2 = plot(fig04,fig05,fig06,
layout=@layout([a ; b ; c]),
size=(900,800)
)
saveplot(p2,"zvssg0")




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_directory = "20260312B"

OUTDIR    = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS", data_directory, RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

outfile_raw         = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data.jld2")
outfile_processed   = joinpath(BASE_PATH,"EXPERIMENTS",data_directory, "data_processed.jld2")
data_summary_path   = joinpath(BASE_PATH,"SG0_EXPDATA_ANALYSIS","summary",data_directory)
isdir(data_summary_path) || mkpath(data_summary_path);


if !isfile(outfile_processed) # check if the processed images exists
    if !isfile(outfile_raw) # check if the raw data exists
        @info "Not found → building $outfile_raw"
        data_in = joinpath(BASE_PATH,"EXPERIMENTS", data_directory)
        data_raw = MyExperimentalAnalysis.SG0_stack_data(data_in)
        jldsave(outfile_raw, data=data_raw)
        data_raw = nothing
    else
        @info "Found $outfile_raw → skipping build"
    end

    data_raw = load(outfile_raw)["data"]
    data_processed = MyExperimentalAnalysis.SG0_build_processed_dict(data_raw)
    jldsave(outfile_processed, data=data_processed)
    data_processed = nothing
    data_raw = nothing
else
    @info "Found $outfile_processed → skipping build"
end
data_processed = load(outfile_processed)["data"]


SG0_current = data_processed[:SG0Currents]
SG1_current = data_processed[:SG1Currents]

f1_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F1", data_processed, nz; half_max=true, λ0=λ0);
f2_mean_max = MyExperimentalAnalysis.SG0_mean_maxima("F2", data_processed, nz; half_max=true, λ0=λ0);

f1_max = MyExperimentalAnalysis.SG0_framewise_maxima("F1", data_processed, nz ; half_max=true,λ0=λ0);
f2_max = MyExperimentalAnalysis.SG0_framewise_maxima("F2", data_processed, nz ; half_max=true,λ0=λ0);

f1_z_mm , f1_z_sem_mm  = vec(mean(f1_max, dims=1)) , sqrt.(vec(std(f1_max, dims=1; corrected=true) ./ sqrt(size(f1_max,1))).^2 .+ z_mm_error^2 );
f2_z_mm , f2_z_sem_mm  = vec(mean(f2_max, dims=1)) , sqrt.(vec(std(f2_max, dims=1; corrected=true) ./ sqrt(size(f2_max,1))).^2 .+ z_mm_error^2 );


data = hcat(SG0_current,SG1_current, f1_z_mm, f1_z_sem_mm, f2_z_mm, f2_z_sem_mm)
cols = palette(:darkrainbow, size(data,1))   # generate colors

dataA = sortslices(data; dims=1, by = r -> (r[1], r[2]))[1:8,:]
dataB = sortslices(data; dims=1, by = r -> (r[2], r[1]))[9:end,:]



data01 = sortslices(dataA; dims=1, by = r -> (r[1], r[2]))
zcenter = data01[1,3]
fig01 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data01,1)
    plot!(fig01,
        [1000*data01[i,2]], [data01[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c0} = %$(round(data01[i,1];digits=3))\mathrm{A}$")
end
plot!(fig01, 
    xlims=(-0.1,0.1),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig01)

data02 = sortslices(dataA; dims=1, by = r -> (r[1], r[2]))
zcenter = data02[1,3]
data02[:,3] = (data02[:,3] .- zcenter)
data02 = data02[2:end, :]
fig02 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data02,1)
    plot!(fig02,
        [1000*data02[i,2]], [abs.(data02[i,3])],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        # yerror=[data03[i,4]],
        label=L"$I_{c0} = %$(round(data02[i,1];digits=3))\mathrm{A}$")
end
plot!(fig02,
    yscale=:log10,
    ylims=(1e-3,1e-2),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig02)

data03 = sortslices(dataB; dims=1, by = r -> (r[2], r[1]))
zcenter = dataA[1,3]
fig03 = plot(xlabel="Current SG1 (mA)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data03,1)
    plot!(fig03,
        [1000*data03[i,2]], [data03[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data03[i,4]],
        label=L"$I_{c0} = %$(round(data03[i,1];digits=3))\mathrm{A}$")
end
plot!(fig03, 
    # xlims=(-0.1,0.1),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig03)

data04 = sortslices(dataB; dims=1, by = r -> (r[2], r[1]))
zcenter = dataA[1,3]
data04[:,3] = (data04[:,3] .- zcenter)
fig04 = plot(xlabel="Current SG1 (A)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data04,1)
    plot!(fig04,
        [data04[i,2]], [abs.(data04[i,3])],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        # yerror=[data03[i,4]],
        label=L"$I_{c0} = %$(round(data04[i,1];digits=3))\mathrm{A}$")
end
plot!(fig04,
    xscale=:identity,
    yscale=:log10,
    # ylims=(1e-3,1e-2),
    # yticks = ([1e-3,1e-2, 1e-1], 
    #         [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:bottomright,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig04)


p1 = plot(fig01,fig02, fig03, fig04,
layout=@layout([a  b ; c  d ]),
size=(900,800)
)
saveplot(p1,"zvssg1")


data04 = sortslices(dataB; dims=1, by = r -> (r[1], r[2]))
fig04 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data04,1)
    plot!(fig04,
        [data04[i,1]], [data04[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data01[i,4]],
        label=L"$I_{c1} = %$(round(1000*data04[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig04,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:best,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig04)


data05 = DataReading.subset_by_cols(data04,[1]; thr = 1e-6, include_equal=true )[3]
fig05 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ Peak position (mm)")
for i in 1: size(data05,1)
    plot!(fig05,
        [data05[i,1]], [data05[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data05[i,4]],
        label=L"$I_{c1} = %$(round(1000*data05[i,2];digits=2))\mathrm{mA}$")
end
plot!(fig05,
    xscale=:log10,
    legend_title=data_directory,
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig05)


data06 = sortslices(data; dims=1, by = r -> (r[1], r[2]))
zcenter = data06[1,3]
data06[:,3] = data06[:,3] .- zcenter
data06 = DataReading.subset_by_cols(data06,[1]; thr = 1e-6, include_equal=true )[3]
fig06 = plot(xlabel="Current SG0 (A)",
    ylabel=L"$F=1$ $z-z_{c,o}$ (mm)")
for i in 1: size(data06,1)
    plot!(fig06,
        [data06[i,1]], [data06[i,3]],
        seriestype=:scatter,
        marker=(:circle,:white,2),
        markerstrokecolor=cols[i],
        markerstrokewidth = 1.5,
        yerror=[data06[i,4]],
        label=L"$I_{c1} = %$(round(1000*data06[i,2];digits=3))\mathrm{mA}$")
end
plot!(fig06,
    xscale=:log10,
    yscale=:log10,
    # ylims=(3e-2,1e-1),
    yticks = ([1e-3,1e-2, 1e-1], 
            [L"10^{-3}", L"10^{-2}", L"10^{-1}"]),
    legend_title=L"%$(data_directory): $z_{c}=%$(round(zcenter; digits=3))\mathrm{mm}$",
    legendtitlefontsize=8,
    legend=:bottomleft,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    legend_columns=2)
display(fig06)

p2 = plot(fig04,fig05,fig06,
layout=@layout([a ; b ; c]),
size=(900,800)
)
saveplot(p2,"zvssg0")








plot(
    f1_mean_max.mean_profile[:,1],
    f1_mean_max.mean_profile[:,2],
seriestype=:scatter,
marker=(:circle,:white,3));
plot!(
    f1_mean_max.mean_bin_profile[:,1],
    f1_mean_max.mean_bin_profile[:,2],
seriestype=:scatter,
marker=(:circle,:white,3),
markerstrokecolor=:red,);
plot!(xlims=(0.0,0.2))

