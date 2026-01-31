# Simulation of atom trajectories in the Stern–Gerlach experiment
# Manipulation of data
# Kelvin Titimbo
# California Institute of Technology
# January 2026

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
# using Interpolations #, Roots, Loess, Optim
# using BSplineKit
using Dierckx, Optim
using DSP
using LambertW, PolyLog
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
include("./Modules/TheoreticalSimulation.jl");
include("./Modules/DataReading.jl");
include("./Modules/MyExperimentalAnalysis.jl");

function keypath(branch::Symbol, ki::Float64, nz::Int, gw::Float64, λ0_raw::Float64)
    fmt(x) = @sprintf("%.12g", x)  # safer than %.6g to reduce collisions
    return "/" * String(branch) *
           "/ki=" * fmt(ki) *"e-6" *
           "/nz=" * string(nz) *
           "/gw=" * fmt(gw) *
           "/lam=" * fmt(λ0_raw)
end

"""
Compute per-particle fluorescence weights for a laser sheet,
including Doppler shift, saturation, and transit-time effects.

Arguments:
- vz, vy :: velocity components at detection (m/s)
- branch :: :up or :dw
    :up → Doppler detuning Δ = δlaser + vz/λ
    :dw → Doppler detuning Δ = δlaser - vz/λ
Keyword options:
- λ        :: laser wavelength (m)
- Γν       :: natural linewidth FWHM (Hz)
- δlaser   :: laser detuning from unshifted resonance (Hz)
- s0       :: effective saturation parameter
- Δy       :: laser sheet thickness (m)
- normalize :: if true, normalize weights to sum(w)=1

Returns:
- w :: vector of fluorescence weights
"""
function doppler_weights(
    vz::AbstractVector,
    vy::AbstractVector;
    branch::Symbol = :up,
    λ::Float64 = 770e-9,
    Γν::Float64 = 5.956e6,
    δlaser::Float64 = 0.0,
    s0::Float64 = 30.3,
    Δy::Float64 = 50e-6,
    normalize::Bool = false)

    N = length(vz)
    length(vy) == N || error("vz and vy must have same length")

    # Determine Doppler sign from branch
    doppler_sign = branch === :up  ? +1 :
                   branch === :dw  ? -1 :
                   error("branch must be :up or :dw (got $branch)")

    w = Vector{Float64}(undef, N)

    @inbounds for i in 1:N
        # Total detuning seen by atom i (Hz)
        Δ = δlaser + doppler_sign * vz[i] / λ

        # Steady-state Lorentzian factor (OBE result)
        denom = 1 + s0 + (2 * Δ / Γν)^2
        R = s0 / denom

        # Transit time through the laser sheet
        τ = Δy / vy[i]

        # Weight ∝ number of scattered photons
        w[i] = Γν * R * τ
    end

    if normalize
        s = sum(w)
        s > 0 || error("Sum of Doppler weights is zero")
        w ./= s
    end

    return w
end


"""
Build Doppler-corrected image as a weighted 2D histogram.

coords_u, coords_v: arrays defining image coordinates (e.g. xf, yf) at detector
weights: per-particle weights (same length)
edges_u, edges_v: bin edges (ranges)
"""
function weighted_image(coords_u, coords_v, w, edges_u, edges_v; norm_mode::Symbol= :probability)
    if norm_mode === :none
        h = fit(Histogram, (coords_u, coords_v), weights(w), (edges_u, edges_v))                    # raw counts (no normalization)
    elseif norm_mode in (:probability, :pdf, :density)
        h = normalize(fit(Histogram, (coords_u, coords_v), weights(w), (edges_u, edges_v)); mode=norm_mode)
    else
        throw(ArgumentError("mode must be one of :pdf, :density, :probability, :none, got $norm_mode"))
    end
    return h
end

cam_pixelsize = 6.5e-6 ;  # Physical pixel size of camera [m]

# Coil currents
Icoils = [0.00,
            0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
            0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,
            0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,
            0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
];
nI = length(Icoils)

nx_bins, nz_bins = 64, 2 ;
gaussian_width_mm = 0.200;
λ0_raw            = 0.01;
λ0_spline         = 0.001;
ki_idx = 27;
Ic_idx = 47;
norm_type = :none ;
@info "Electric current associated to the simulation $(Int(1000*Icoils[Ic_idx]))mA"

# Fixed analysis limits
xlim = (-8.0, 8.0);
zlim = (-12.5, 12.5);
xmin, xmax = xlim ;
zmin, zmax = zlim ;

# Bin size in mm (default_camera_pixel_size is assumed global in meters)
x_bin_size = 1e3 * nx_bins * cam_pixelsize ;
z_bin_size = 1e3 * nz_bins * cam_pixelsize ;

# --------------------------------------------------------
# X edges: force symmetric centers around 0
# --------------------------------------------------------
x_half_range = max(abs(xmin), abs(xmax)) ;
kx = max(1, ceil(Int, x_half_range / x_bin_size)) ;
centers_x = collect((-kx:kx) .* x_bin_size) ;
edges_x = collect((-(kx + 0.5)) * x_bin_size : x_bin_size : ((kx + 0.5) * x_bin_size)) ;

# --------------------------------------------------------
# Z edges: force symmetric centers around 0
# --------------------------------------------------------
z_half_range = max(abs(zmin), abs(zmax));
kz = max(1, ceil(Int, z_half_range / z_bin_size));
centers_z = collect((-kz:kz) .* z_bin_size);
edges_z = collect((-(kz + 0.5)) * z_bin_size : z_bin_size : ((kz + 0.5) * z_bin_size));

# ===================================================
# ++++++++++++++++++++ EXPERIMENT +++++++++++++++++++
# ===================================================
exp_path = joinpath(@__DIR__,"20250919");
exp_data = load(joinpath(exp_path,"data_processed.jld2"),"data");
Ic_idx_exp = 26 ;
@info "Electric current chosen from the experimental data $(round(1000*exp_data[:Currents][Ic_idx_exp]; sigdigits=4))mA"

exp_image = mean(exp_data[:F1ProcessedImages], dims=(3)) |> x -> dropdims(x, dims=(3));
exp_image = exp_image[:,:,Ic_idx_exp];
exp_size = size(exp_image);
exp_x = 1e3*TheoreticalSimulation.pixel_coordinates(2160, 4, cam_pixelsize);
exp_z = 1e3*TheoreticalSimulation.pixel_coordinates(2560, 1, cam_pixelsize);

exp_result = DataReading.find_report_data(
    joinpath(@__DIR__, "analysis_data");
    wanted_data_dir = "20250919",
    wanted_binning  = 1,
    wanted_smooth   = 0.01,
);
exp_profile = vec(mean(exp_image, dims=1));
exp_z = exp_z .- exp_result[:fw_centroid_mm][1];

img_exp = heatmap(
    exp_x,
    exp_z,
    exp_image',
    colormap = :viridis,
    title = "Experiment",
    xlabel = "x (mm)",
    ylabel = "z (mm)",
    xlims = extrema(exp_x),
    ylims = extrema(exp_z),
    aspect_ratio = :equal,
    colorbar = true
);

img_exp_prof = plot(exp_z, exp_profile,
    label="Experimental profile",
    seriestype=:scatter,
    marker=(:circle,2,:white),
    markerstrokecolor=:gray23,
    xlabel=L"$z$ (mm)",
    xlims=zlim,
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
);

# ===================================================
# ++++++++++++++++++++ QM DATA ++++++++++++++++++++++
# ===================================================
qm_path = joinpath(@__DIR__,"simulation_data","qm_simulation_7M","qm_screen_data.jld2");
qm_data = jldopen(qm_path, "r") do file
        file["screen"]["i$(Ic_idx)"]
end
qm_data = vcat([qm_data[lv]  for lv=6:8]...);

xf_qm = 1e3* view(qm_data,:,7);
zf_qm = 1e3* view(qm_data,:,8);

vxf_qm = view(qm_data,:,4);
vyf_qm = view(qm_data,:,5);
vzf_qm = view(qm_data,:,9);

if norm_type === :none
    h_qm = fit(Histogram, (xf_qm, zf_qm), (edges_x, edges_z))                    # raw counts (no normalization)
elseif norm_type in (:probability, :pdf, :density)
    h_qm = normalize(fit(Histogram, (xf_qm, zf_qm), (edges_x, edges_z)); mode=norm_type)
else
    throw(ArgumentError("mode must be one of :pdf, :density, :probability, :none, got $norm_type"))
end

counts_qm = h_qm.weights ; # size: (length(centers_x), length(centers_z))

qm_profile = vec(mean(counts_qm, dims=1));
z_max_qm_mm = centers_z[argmax(qm_profile)];
z_max_qm_spline_mm, Sfit_qm = TheoreticalSimulation.max_of_bspline_positions(centers_z,qm_profile;λ0=λ0_raw);

img_qm = heatmap(
    centers_x,
    centers_z,
    counts_qm';
    colormap = :viridis,
    title = "Quantum Mechanis",
    xlabel = "x (mm)",
    ylabel = "z (mm)",
    xlims = (-8,8),
    ylims = (-3,12),
    aspect_ratio = :equal,
    colorbar = true
);

img_qm_prof = plot(centers_z, qm_profile,
    label="Quantum mechanics",
    xlabel=L"$z$ (mm)",
    line=(:solid,2,:blue),
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
);

table_qm_path = joinpath(@__DIR__,
    "simulation_data",
    "qm_simulation_7M",
    "qm_7000000_screen_profiles_f1_table.jld2");
table_qm      = load(table_qm_path)["table"];
@info "QM data loaded"
qm_meta = let
    keys_qm = collect(keys(table_qm))  # make it an indexable Vector of tuples

    nz_qm = sort(unique(getindex.(keys_qm, 1)))
    gw_qm = sort(unique(getindex.(keys_qm, 2)))
    λ0_qm = sort(unique(getindex.(keys_qm, 3)))

    # --- pretty print aligned ---
    labels = ["nz_qm", "gw_qm", "λ0_qm"]
    w = maximum(length.(labels))

    # println(rpad("nz_qm", w), " = ", nz_qm);
    # println(rpad("gw_qm", w), " = ", gw_qm);
    # println(rpad("λ0_qm", w), " = ", λ0_qm);

    # --- return renamed container ---
    OrderedDict(
        :nz => nz_qm,
        :gw => gw_qm,
        :λ0 => λ0_qm,
        :λs => 0.001
    );
end    
data_qm = table_qm[(nz_bins,gaussian_width_mm,λ0_raw)];
Ic_QM   = [data_qm[i][:Icoil] for i in eachindex(data_qm)];
zmax_QM = [data_qm[i][:z_max_smooth_spline_mm] for i in eachindex(data_qm)];
# ===================================================
# +++++++++++++++++++++ CQD DATA ++++++++++++++++++++
# ===================================================
data_CQD_up  = load(joinpath(@__DIR__,"simulation_data","cqd_simulation_6M","cqd_6000000_ki0$(ki_idx)_up_screen.jld2"),"screen")[:data];
data_cqd = data_CQD_up[Ic_idx];

xf_cqd = 1e3* view(data_cqd,:,9);
zf_cqd = 1e3* view(data_cqd,:,10);

vxf_cqd = view(data_cqd,:,4);
vyf_cqd = view(data_cqd,:,5);
vzf_cqd =view(data_cqd,:,11);

table_cqd_path = joinpath(@__DIR__,
    "simulation_data",
    "cqd_simulation_6M",
    "cqd_6000000_up_profiles_bykey.jld2");
cqd_meta = jldopen(table_cqd_path, "r") do file
    meta = file["meta"]

    # mapping: original key (String) → new Symbol
    rename = OrderedDict(
        "induction_coeff"   => :ki,
        "nz_bins"           => :nz,
        "gaussian_width_mm" => :gw,
        "λ0_raw_list"       => :λ0,
        "λ0_spline"         => :λs,
    )

    # alignment width (use original names for printing)
    w = maximum(length.(keys(rename)))

    out = OrderedDict{Symbol,Any}()

    for (k_old, k_new) in rename
        val = round.(meta[k_old], digits=3)

        # println(rpad(k_old, w), " = ", val)

        out[k_new] = val
    end
    out;
end
ki = cqd_meta[:ki][ki_idx];
@info "Induction term kᵢ=$(ki)×10⁻⁶ "

data_CQD_up_zmax = jldopen(table_cqd_path, "r") do file
    file[keypath(:up,ki,nz_bins,gaussian_width_mm,λ0_raw)]
end
Ic_CQD   = [data_CQD_up_zmax[idx][:Icoil] for idx=1:nI];
zmax_CQD = [data_CQD_up_zmax[idx][:z_max_smooth_spline_mm] for idx=1:nI];

if norm_type === :none
    h_cqd = fit(Histogram, (xf_cqd, zf_cqd), (edges_x, edges_z));     # raw counts (no normalization)
elseif norm_type in (:probability, :pdf, :density)
    h_cqd = normalize(fit(Histogram, (xf_cqd, zf_cqd), (edges_x, edges_z)); mode=norm_type);
else
    throw(ArgumentError("mode must be one of :pdf, :density, :probability, :none, got $norm_type"))
end

counts_cqd = h_cqd.weights ; # size: (length(centers_x), length(centers_z))
cqd_profile = vec(mean(counts_cqd, dims=1));
z_max_cqd_mm = centers_z[argmax(cqd_profile)];
z_max_cqd_spline_mm, Sfit_cqd = TheoreticalSimulation.max_of_bspline_positions(centers_z,cqd_profile;λ0=λ0_raw);

img_cqd = heatmap(
    centers_x,
    centers_z,
    counts_cqd';
    colormap = :viridis,
    title="CoQuantum",
    xlabel = "x (mm)",
    ylabel = "z (mm)",
    xlims = (-8,8),
    ylims = (-3,12),
    aspect_ratio = :equal,
    colorbar = true
);

img_cqd_prof = plot(centers_z, cqd_profile,
    label=L"CQD ($k_{i}=%$(ki)\times 10^{-6}$)",
    xlabel=L"$z$ (mm)",
    line=(:solid,2,:red),
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
);

# ======================================================================================================
# ======================================================================================================

plot(img_exp, img_exp_prof, img_qm, img_qm_prof, img_cqd, img_cqd_prof, 
    layout=(3,2),
    suptitle = "Original",
    left_margin=10mm,
    size=(1000,1400)
)


# ======================================================================================================
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ======================================================================================================

# 1) Choose laser detuning (Hz). You can set δlaser = 0 or optimize it later.
δlaser = -5e6 ;
# 2) s0: saturation parameter (dimensionless)
s0 = 2.75 ;
# 3) Compute Doppler weights
w_cqd = doppler_weights(vzf_cqd, vyf_cqd; branch=:up, Γν=5.956e6, δlaser=δlaser, s0=s0, normalize = true);
w_qm = doppler_weights(vzf_qm, vyf_qm; branch=:up, Γν=5.956e6, δlaser=δlaser, s0=s0, normalize = true);

img_cqd = weighted_image(xf_cqd, zf_cqd, w_cqd, edges_x, edges_z; norm_mode = norm_type);
img_qm = weighted_image(xf_qm, zf_qm, w_qm, edges_x, edges_z; norm_mode = norm_type);

# img.weights is your Doppler-corrected image intensity array
counts_cqd_doppler = img_cqd.weights;
counts_qm_doppler = img_qm.weights;

# z-profile 
z_profile_cqd_doppler = vec(mean(counts_cqd_doppler, dims = 1));
z_max_cqd_mm_doppler = centers_z[argmax(z_profile_cqd_doppler)];
z_max_cqd_spline_mm_doppler, Sfit_cqd_doppler = TheoreticalSimulation.max_of_bspline_positions(centers_z,z_profile_cqd_doppler;λ0=λ0_raw );

z_profile_qm_doppler = vec(mean(counts_qm_doppler, dims = 1));
z_max_qm_mm_doppler = centers_z[argmax(z_profile_qm_doppler)];
z_max_qm_spline_mm_doppler, Sfit_qm_doppler = TheoreticalSimulation.max_of_bspline_positions(centers_z,z_profile_qm_doppler;λ0=λ0_raw );

img_qm_doppler = heatmap(
    centers_x,
    centers_z,
    counts_qm_doppler';
    colormap = :viridis,
    title = "Quantum Mechanis",
    xlabel = "x (mm)",
    ylabel = "z (mm)",
    xlims = (-8,8),
    ylims = (-3,12),
    aspect_ratio = :equal,
    colorbar = true
);

img_qm_prof_doppler = plot(centers_z, z_profile_qm_doppler,
    label="Quantum mechanics",
    xlabel=L"$z$ (mm)",
    line=(:solid,2,:blue),
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
);

img_cqd_doppler = heatmap(
    centers_x,
    centers_z,
    counts_cqd_doppler';
    colormap = :viridis,
    title="CoQuantum",
    xlabel = "x (mm)",
    ylabel = "z (mm)",
    xlims = (-8,8),
    ylims = (-3,12),
    aspect_ratio = :equal,
    colorbar = true
);

img_cqd_prof_doppler = plot(centers_z, z_profile_cqd_doppler,
    label=L"CQD ($k_{i}=%$(ki)\times 10^{-6}$)",
    xlabel=L"$z$ (mm)",
    line=(:solid,2,:red),
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
);

plot(img_exp, img_exp_prof, img_qm_doppler, img_qm_prof_doppler, img_cqd_doppler, img_cqd_prof_doppler, 
suptitle=L"Doppler: $\delta_{L}=%$(round(δlaser/1e6,sigdigits=4))$MHz, $s_{0}=%$(s0)$",
layout=(3,2),
left_margin=10mm,
size=(1000,1400))



fig1 = plot(exp_z[1:10:end], exp_profile[1:10:end]/maximum(exp_profile),
    title = L"$\delta_{L}=%$(round(δlaser/1e6,sigdigits=4))$MHz, $s_{0}=%$(s0)$",
    label="Experimental profile",
    seriestype=:scatter,
    marker=(:circle,2,:white),
    markerstrokecolor=:gray23,
    xlabel=L"$z$ (mm)",
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
);
plot!(centers_z, qm_profile/maximum(qm_profile),
    label="Quantum mechanics",
    xlabel=L"$z$ (mm)",
    line=(:solid,2,:blue),
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
);
plot!(centers_z, z_profile_qm_doppler/maximum(z_profile_qm_doppler),
    label="Quantum mechanics + Doppler",
    xlabel=L"$z$ (mm)",
    line=(:dash,2,:purple3),
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
);
plot!(data_qm[Ic_idx][:z_profile][:,1],data_qm[Ic_idx][:z_profile][:,2]/maximum(data_qm[Ic_idx][:z_profile][:,2]),
    label="Gaussian smoothing",
    line=(:orangered2,1,:dot)
);

fig2 = plot(exp_z[1:10:end], exp_profile[1:10:end]/maximum(exp_profile),
    title = L"$\delta_{L}=%$(round(δlaser/1e6,sigdigits=4))$MHz, $s_{0}=%$(s0)$",
    label="Experimental profile",
    seriestype=:scatter,
    marker=(:circle,2,:white),
    markerstrokecolor=:gray23,
    xlabel=L"$z$ (mm)",
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
);
plot!(centers_z, cqd_profile/maximum(cqd_profile),
    label=L"CQD ($k_{i}=%$(ki)\times 10^{-6}$)",
    xlabel=L"$z$ (mm)",
    line=(:solid,2,:red),
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
);
plot!(centers_z, z_profile_cqd_doppler/maximum(z_profile_cqd_doppler),
    label=L"CQD ($k_{i}=%$(ki)\times 10^{-6}$) + Doppler",
    xlabel=L"$z$ (mm)",
    line=(:dash,2,:seagreen4),
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
);
plot!(data_CQD_up_zmax[Ic_idx][:z_profile][:,1],data_CQD_up_zmax[Ic_idx][:z_profile][:,2]/maximum(data_CQD_up_zmax[Ic_idx][:z_profile][:,2]),
    label="Gaussian smoothing",
    line=(:green2,1,:dot)
);

plot(fig1,fig2,
layout = (2,1),
size=(600,1000),
left_margin = 5mm,
)



fig = plot(xlabel=L"$z$ (mm)");
Laser_detuning = collect(range(-15,15,11));
cols_i = palette(:darkrainbow,length(Laser_detuning));
Γ_K39 = 5.956e6 ;
s0 = 2.75 ;
# plot!(fig,centers_z,cqd_profile, label="CQD profile",
#     line=(:dash,2,:black));
for (i,δlaser) in enumerate(Laser_detuning)
    w = doppler_weights(vzf_cqd, vyf_cqd; branch=:up, Γν=Γ_K39, δlaser=1e6*δlaser, s0=s0, normalize=false)
    img = weighted_image(xf_cqd, zf_cqd, w , edges_x, edges_z; norm_mode = norm_type)
    counts_doppler = img.weights
    z_profile_doppler = vec(mean(counts_doppler, dims = 1))
    plot!(fig,
        centers_z,z_profile_doppler, 
        label=L"$\delta_{L}=%$(round(δlaser, sigdigits=3))$MHz",
        line = (:solid,1.2, cols_i[i]))
end
plot!(fig,xlims=(-1.5,+8.5),
    legend_columns=1,
    legend=:outerright,
    foreground_color_legend=nothing);
plot!(fig, title = L"$\Gamma = %$(round(Γ_K39/1e6; sigdigits=4))\mathrm{MHz}$ | $s_{0}=%$(s0)$");
display(fig)


# ======================================================================================================
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ======================================================================================================

out_qm = OrderedDict{Int, Tuple{Matrix{Float64}, Vector{Float64}}}();
for (idx,ic) in enumerate(Icoils)
    @info "Current " ic ; 
    qm_data = jldopen(qm_path, "r") do file
            file["screen"]["i$(idx)"]
    end
    qm_data = vcat([qm_data[lv]  for lv=6:8]...)

    xf_qm = 1e3* view(qm_data,:,7)
    zf_qm = 1e3* view(qm_data,:,8)

    vxf_qm = view(qm_data,:,4)
    vyf_qm = view(qm_data,:,5)
    vzf_qm = view(qm_data,:,9)

    if norm_type === :none
        h_qm = fit(Histogram, (xf_qm, zf_qm), (edges_x, edges_z))                    # raw counts (no normalization)
    elseif norm_type in (:probability, :pdf, :density)
        h_qm = normalize(fit(Histogram, (xf_qm, zf_qm), (edges_x, edges_z)); mode=norm_type)
    else
        throw(ArgumentError("mode must be one of :pdf, :density, :probability, :none, got $norm_type"))
    end

    counts_qm = h_qm.weights ; # size: (length(centers_x), length(centers_z))
    qm_profile = vec(mean(counts_qm, dims=1))
    z_max_qm_mm = centers_z[argmax(qm_profile)];
    z_max_qm_spline_mm, Sfit_qm = TheoreticalSimulation.max_of_bspline_positions(centers_z,qm_profile;λ0=λ0_raw);

    w_qm = doppler_weights(vzf_qm, vyf_qm; branch=:up, Γν=5.956e6, δlaser=δlaser, s0=s0, normalize = true)
    img_qm = weighted_image(xf_qm, zf_qm, w_qm, edges_x, edges_z; norm_mode = norm_type)
    counts_qm_doppler = img_qm.weights
    z_profile_qm_doppler = vec(mean(counts_qm_doppler, dims = 1))
    z_max_qm_mm_doppler = centers_z[argmax(z_profile_qm_doppler)]
    z_max_qm_spline_mm_doppler, Sfit_qm_doppler = TheoreticalSimulation.max_of_bspline_positions(centers_z,z_profile_qm_doppler;λ0=λ0_raw )

    z_hist = hcat(centers_z, qm_profile, z_profile_qm_doppler)
    z_hist_max = [z_max_qm_mm, z_max_qm_spline_mm[1] , z_max_qm_spline_mm_doppler[1] ]

    # ✅ STORE RESULT
    out_qm[idx] = (z_hist, z_hist_max)

end

exp_avg = load(joinpath(@__DIR__,"analysis_data","smoothing_binning","data_averaged_2.jld2"))["data"];
@info "Experimental data loaded"
# 1. Fit spline for the experiment data
data_experiment = Spline1D(
    exp_avg[:i_smooth], 
    exp_avg[:z_smooth], 
    k=3, 
    bc="extrapolate", 
    s=0.0, 
    w = 1 ./ exp_avg[:δz_smooth].^2
);
# xq and δxq from grouped data
xq  = exp_avg[:Ic_grouped][:,1];
δxq = exp_avg[:Ic_grouped][:,2];
# 2. Spline derivative at xq : 
# Evaluate derivative dz/dI at the grouped current points xq.
# This is needed to propagate δI into δz via slope*δI.
dyq = derivative(data_experiment, xq; nu=1);
# 3. Interpolate δy uncertainty to xq
err_spline = Spline1D(exp_avg[:i_smooth], exp_avg[:δz_smooth], k=3, bc="extrapolate");
δy_interp = err_spline.(xq);   # # σ_z at xq
# 4. Total propagated uncertainty σ_total = sqrt( (dz/dI * σ_I)^2 + σ_z(I)^2 )
δyq = sqrt.( (dyq .* δxq).^2 .+ δy_interp.^2 );
# Pack into a convenient table:
# columns = [I, δI, z_spline(I), σ_total]
data_exp_scattered = hcat(xq,δxq,data_experiment.(xq),δyq);
pretty_table(data_exp_scattered;
        alignment     = :c,
        title         = @sprintf("EXPERIMENTAL DATA (scattered)"),
        column_labels = ["Ic (A)","δIc (A)", "z (mm)", "δz (mm)"],
        formatters    = ([fmt__printf("%1.3f", [1]),fmt__printf("%1.4f", [2]),fmt__printf("%1.3f", 3:4)]),
        style         = TextTableStyle(
                        first_line_column_label = crayon"yellow bold",
                        table_border  = crayon"blue bold",
                        column_label  = crayon"yellow bold",
                        title = crayon"bold red"
                        ),
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
        equal_data_column_widths= true,
);



out_cqd = OrderedDict{Int, Tuple{Matrix{Float64}, Vector{Float64}}}();
for (idx,ic) in enumerate(Icoils)
    @info "Current " ic ; 
    data_cqd = data_CQD_up[idx]

    xf_cqd = 1e3* view(data_cqd,:,9)
    zf_cqd = 1e3* view(data_cqd,:,10)

    vxf_cqd = view(data_cqd,:,4)
    vyf_cqd = view(data_cqd,:,5)
    vzf_cqd =view(data_cqd,:,11)

    if norm_type === :none
        h_cqd = fit(Histogram, (xf_cqd, zf_cqd), (edges_x, edges_z))                    # raw counts (no normalization)
    elseif norm_type in (:probability, :pdf, :density)
        h_cqd = normalize(fit(Histogram, (xf_cqd, zf_cqd), (edges_x, edges_z)); mode=norm_type)
    else
        throw(ArgumentError("mode must be one of :pdf, :density, :probability, :none, got $norm_type"))
    end

    counts_cqd = h_cqd.weights ; # size: (length(centers_x), length(centers_z))
    cqd_profile = vec(mean(counts_cqd, dims=1))
    z_max_cqd_mm = centers_z[argmax(cqd_profile)];
    z_max_cqd_spline_mm, Sfit_cqd = TheoreticalSimulation.max_of_bspline_positions(centers_z,cqd_profile;λ0=λ0_raw);

    w_cqd = doppler_weights(vzf_cqd, vyf_cqd; branch=:up, Γν=5.956e6, δlaser=δlaser, s0=s0, normalize = true)
    img_cqd = weighted_image(xf_cqd, zf_cqd, w_cqd, edges_x, edges_z; norm_mode = norm_type)
    counts_cqd_doppler = img_cqd.weights
    z_profile_cqd_doppler = vec(mean(counts_cqd_doppler, dims = 1))
    z_max_cqd_mm_doppler = centers_z[argmax(z_profile_cqd_doppler)]
    z_max_cqd_spline_mm_doppler, Sfit_cqd_doppler = TheoreticalSimulation.max_of_bspline_positions(centers_z,z_profile_cqd_doppler;λ0=λ0_raw )

    z_hist = hcat(centers_z, cqd_profile, z_profile_cqd_doppler)
    z_hist_max = [z_max_cqd_mm, z_max_cqd_spline_mm[1] , z_max_cqd_spline_mm_doppler[1] ]

    # ✅ STORE RESULT
    out_cqd[idx] = (z_hist, z_hist_max)
end


plot(data_exp_scattered[12:end,1], data_exp_scattered[12:end,3]/0.93,
    seriestype=:scatter,
    marker=(:circle, 4, :white),
    markerstrokecolor =:black,
    markerstrokewidth=2,
    label="Combined Experiment");
plot!(Icoils[15:end], [out_qm[idx][2][2] for idx=15:nI],
    label="QM",
    line=(:darkred,1.8,:solid)
);
plot!(Icoils[15:end], zmax_QM[15:end],
    line=(:red,1,:dash),
    label=L"QM+Smoothing ($\sigma_{w} = %$(Int(1e3*gaussian_width_mm))\mathrm{\mu m}$)"
);
plot!(Icoils[15:end], [out_qm[idx][2][3] for idx=15:nI],
    line=(:orangered,1.2,:dot),
    label=L"QM+Doppler ($\delta_{L}=%$(round(1e-6*δlaser;sigdigits=4))$ MHz)"
);
plot!(Icoils[15:end], [out_cqd[idx][2][2] for idx=15:nI],
    line=(:navyblue,1.8,:solid),
    label="CQD"
);
plot!(Icoils[15:end], zmax_CQD[15:end],
    line=(:blue,1,:dash),
    label=L"CQD+Smoothing ($\sigma_{w} = %$(Int(1e3*gaussian_width_mm))\mathrm{\mu m}$)"
);
plot!(Icoils[15:end], [out_cqd[idx][2][3] for idx=15:nI],
    line=(:purple2,1.2,:dot),
    label=L"CQD+Doppler ($\delta_{L}=%$(round(1e-6*δlaser;sigdigits=4))$ MHz)"
);
plot!(
    xlabel="Current (A)",
    ylabel=L"$z_{max}$ (mm)",
    xscale=:log10,
    yscale=:log10,
    xlims=(0.015,1.05),
    xticks = ([1e-1, 1.0], [L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:bottomright,
    foreground_color_legend = nothing,
)


    mean(vzf/770e-9)

[out_qm[idx][2][3] for idx=12:nI]



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Dr. Ku has asked about the separation between the two branches at the circular aperture
# 20260126
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
# atom_info       = AtomicSpecies.atoms(atom);
K39_params = TheoreticalSimulation.AtomParams(atom); # [R μn γn Ispin Ahfs M ] 
# Furnace
T_K = 273.15 + 205 ; # Furnace temperature (K)
# Furnace aperture
const x_furnace = 2.0e-3 ;
const z_furnace = 100e-6 ;
# Slit : Pre SG
const x_slit  = 4.0e-3 ;
const z_slit  = 300e-6 ;
# Circular Aperture : Post SG
const R_aper            = 5.8e-3/2 ;
const y_SGToAperture    = 42.0e-3 ;   
# Propagation distances
const y_FurnaceToSlit = 224.0e-3 ;
const y_SlitToSG      = 44.0e-3 ;
const y_SG            = 7.0e-2 ;
const y_SGToScreen    = 32.0e-2 ;
# Connecting pipes
const R_tube = 35e-3/2 ; # Radius of the connecting pipe (m)

R_mm = 1e3 * R_aper
θcirc = range(0, 2π, length=361)
x_circ_mm = R_mm .* cos.(θcirc)    # mm
z_circ_um = R_mm .* sin.(θcirc)    # mm

effusion_params = TheoreticalSimulation.BeamEffusionParams(x_furnace,z_furnace,x_slit,z_slit,y_FurnaceToSlit,T_K,K39_params);
quantum_numbers = TheoreticalSimulation.fmf_levels(K39_params);
y_aper   = y_FurnaceToSlit + y_SlitToSG + y_SG + y_SGToAperture


Ic_idx = 47 ; 
@info "Current chosen Ic = $(Icoils[Ic_idx])A"
qm_path = joinpath(@__DIR__,"simulation_data","qm_simulation_7M","qm_screen_data.jld2");

levels_up = 1:4
levels_dw = 6:8

Ic = Icoils[Ic_idx]
mm = 1e3
λ0_raw = 0.000001

xs_up, zs_up, xs_dw, zs_dw = jldopen(qm_path, "r") do file
    screen = file["screen"]["i$(Ic_idx)"]

    n_atoms_up = sum(size(screen[lvl], 1) for lvl in levels_up)
    n_atoms_dw = sum(size(screen[lvl], 1) for lvl in levels_dw)

    xs_dw = Vector{Float64}(undef, n_atoms_dw)
    zs_dw = Vector{Float64}(undef, n_atoms_dw)
    xs_up = Vector{Float64}(undef, n_atoms_up)
    zs_up = Vector{Float64}(undef, n_atoms_up)

    k = 0
    @inbounds for lvl in levels_up
        data_lvl = screen[lvl]
        f, mf = quantum_numbers[lvl]
        @info "F=$(Int(f)) and mF=$(Int(mf))"
        n = size(data_lvl, 1)

        for j in 1:n
            k += 1
            v0y = data_lvl[j, 5]
            τ_aper = y_aper / v0y

            rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(
                τ_aper, Ic, f, mf,
                @view(data_lvl[j, 1:3]),
                @view(data_lvl[j, 4:6]),
                K39_params
            )

            xs_up[k] = mm * rtmp[1]
            zs_up[k] = mm * rtmp[3]
        end
    end

    k = 0
    @inbounds for lvl in levels_dw
        data_lvl = screen[lvl]
        f, mf = quantum_numbers[lvl]
        @info "F=$(Int(f)) and mF=$(Int(mf))"
        n = size(data_lvl, 1)

        for j in 1:n
            k += 1
            v0y = data_lvl[j, 5]
            τ_aper = y_aper / v0y

            rtmp, _ = TheoreticalSimulation.QM_EqOfMotion(
                τ_aper, Ic, f, mf,
                @view(data_lvl[j, 1:3]),
                @view(data_lvl[j, 4:6]),
                K39_params
            )

            xs_dw[k] = mm * rtmp[1]
            zs_dw[k] = mm * rtmp[3]
        end
    end

    # return arrays from the do-block if you want them outside:
    return xs_up, zs_up, xs_dw, zs_dw
end

bins_aper_up  = (TheoreticalSimulation.FreedmanDiaconisBins(xs_up), TheoreticalSimulation.FreedmanDiaconisBins(zs_up))
# Aperture
fig_up = histogram2d(xs_up, zs_up;
    bins=bins_aper_up, show_empty_bins=true, color=:plasma, normalize=:pdf,
    xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{\mu m})$",
    # xticks=-4.0:0.50:4.0, yticks=-1000:500:3000,
    xlims=(-3.5,3.5), ylims=(-3.5,3.5),
    aspect_ratio=:equal
)
plot!(fig_up, x_circ_mm, z_circ_um, line=(:dash,:gray,2), label=false)
xpos, ypos = -2.5, +3.2; dx, dy = 0.9, 0.3
plot!(fig_up, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                    [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
        color=:white, opacity=0.65, linealpha=0, label=false)
annotate!(fig_up, xpos, ypos, text("⊚ Aperture", 10, :black, :bold, :center, "Helvetica"))


bins_aper_dw  = (TheoreticalSimulation.FreedmanDiaconisBins(xs_dw), TheoreticalSimulation.FreedmanDiaconisBins(zs_dw))
# Aperture
fig_dw = histogram2d(xs_dw, zs_dw;
    bins=bins_aper_dw, show_empty_bins=true, color=:plasma, normalize=:pdf,
    xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{mm})$",
    # xticks=-4.0:0.50:4.0, yticks=-1000:500:3000,
    xlims=(-3.5,3.5), ylims=(-3.5,3.5),
    aspect_ratio=:equal
)
plot!(fig_dw, x_circ_mm, z_circ_um, line=(:dash,:gray,2), label=false)
xpos, ypos = -2.5, +3.2; dx, dy = 0.9, 0.3
plot!(fig_dw, Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                    [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
        color=:white, opacity=0.65, linealpha=0, label=false)
annotate!(fig_dw, xpos, ypos, text("⊚ Aperture", 10, :black, :bold, :center, "Helvetica"))


xs_all = vcat(xs_up,xs_dw);
zs_all = vcat(zs_up,zs_dw);
bins_aper_all  = (TheoreticalSimulation.FreedmanDiaconisBins(xs_all), TheoreticalSimulation.FreedmanDiaconisBins(zs_all))
histogram2d(xs_all, zs_all;
    bins=bins_aper_all, show_empty_bins=true, color=:plasma, normalize=:pdf,
    xlabel=L"$x \ (\mathrm{mm})$", ylabel=L"$z \ (\mathrm{mm})$",
    # xticks=-4.0:0.50:4.0, yticks=-1000:500:3000,
    xlims=(-3.5,3.5), ylims=(-3.5,3.5),
    aspect_ratio=:equal
)
plot!(x_circ_mm, z_circ_um, line=(:dash,:gray,2), label=false)
xpos, ypos = -2.5, +3.2; dx, dy = 0.9, 0.3
plot!(Shape([xpos-dx, xpos+dx, xpos+dx, xpos-dx],
                    [ypos-dy, ypos-dy, ypos+dy, ypos+dy]),
        color=:white, opacity=0.65, linealpha=0, label=false)
annotate!(xpos, ypos, text("⊚ Aperture", 10, :black, :bold, :center, "Helvetica"))


# Fixed analysis limits
xlim = (-3.0, 3.0);
zlim = (-3.0, 3.0);
xlim = (-8.0, 8.0);
zlim = (-12.5, 12.5);
xmin, xmax = xlim ;
zmin, zmax = zlim ;

# Bin size in mm (default_camera_pixel_size is assumed global in meters)
x_bin_size = 1e3 * 4 * cam_pixelsize ;
z_bin_size = 1e3 * 2 * cam_pixelsize ;

# --------------------------------------------------------
# X edges: force symmetric centers around 0
# --------------------------------------------------------
x_half_range = max(abs(xmin), abs(xmax)) ;
kx = max(1, ceil(Int, x_half_range / x_bin_size)) ;
centers_x = collect((-kx:kx) .* x_bin_size) ;
edges_x = collect((-(kx + 0.5)) * x_bin_size : x_bin_size : ((kx + 0.5) * x_bin_size)) ;

# --------------------------------------------------------
# Z edges: force symmetric centers around 0
# --------------------------------------------------------
z_half_range = max(abs(zmin), abs(zmax));
kz = max(1, ceil(Int, z_half_range / z_bin_size));
centers_z = collect((-kz:kz) .* z_bin_size);
edges_z = collect((-(kz + 0.5)) * z_bin_size : z_bin_size : ((kz + 0.5) * z_bin_size));

h_qm_f1 = fit(Histogram, (xs_dw, zs_dw), (edges_x, edges_z))
h_qm_f2 = fit(Histogram, (xs_up, zs_up), (edges_x, edges_z))

counts_qm_f1 = h_qm_f1.weights ; # size: (length(centers_x), length(centers_z))
qm_profile_f1 = vec(mean(counts_qm_f1, dims=1));
z_max_qm_mm_f1 = centers_z[argmax(qm_profile_f1)]
z_max_qm_spline_mm_f1, Sfit_qm_f1 = TheoreticalSimulation.max_of_bspline_positions(centers_z,qm_profile_f1;λ0=λ0_raw)

counts_qm_f2 = h_qm_f2.weights ; # size: (length(centers_x), length(centers_z))
qm_profile_f2 = vec(mean(counts_qm_f2, dims=1));
z_max_qm_mm_f2 = centers_z[argmax(qm_profile_f2)]
z_max_qm_spline_mm_f2, Sfit_qm_f2 = TheoreticalSimulation.max_of_bspline_positions(centers_z,qm_profile_f2;λ0=λ0_raw)


img_qm_prof = plot(centers_z, qm_profile_f1,
    title=L"$z$–profile at the circular aperture",
    label=L"QM ($F=1$)",
    xlabel=L"$z$ (mm)",
    line=(:solid,2,:red),
    legend=:topleft,
    background_color_legend = :transparent,
    foreground_color_legend = nothing,
    xlims=(-4,4),
    xticks=-4:1:4,
)
vline!(img_qm_prof, [z_max_qm_spline_mm_f1[1]], label= L"$z_{max}= %$(round(z_max_qm_spline_mm_f1[1];sigdigits=3))\mathrm{mm}$", line=(:dash, 1, :red))
plot!(img_qm_prof,centers_z, qm_profile_f2,
    label=L"QM ($F=2$, $m_{F}=2,1,0,-1$)",
    line=(:solid,2,:blue))
vline!(img_qm_prof, [z_max_qm_spline_mm_f2[1]], label= L"$z_{max}= %$(round(z_max_qm_spline_mm_f2[1];sigdigits=3))\mathrm{mm}$", line=(:dash, 1, :blue))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Suleyman : experiment = trapezoid + blurring function
# by deconvolution we can access the blurring function
# 20260128

















z_max_qm_spline_mm












data_CQD

ki_l = round.(1e6*[
    [exp10(p) * x for p in -8:-8 for x in 1.0:1:9]; 
    [exp10(p) * x for p in -7:-7 for x in 1.0:1:9]; 
    [exp10(p) * x for p in -6:-6 for x in 1.0:0.1:9.9]; 
    ## exp10(-5) * (1:0.1:10);
    exp10.(-5:-1)
];sigdigits=4)



jldopen(joinpath(OUTDIR,"qm_screen_data.jld2"), "w") do file
    file["meta/Icoils"] = Icoils
    file["meta/levels"] = quantum_numbers 
    for i in 1:nI
        file["screen/I$(i)"] = data_alive_screen[i]  
    end
end







table_old     = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_3M","qm_3000000_screen_profiles_table.jld2"))["table"];
table_qm_f1   = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_6M","qm_6000000_screen_profiles_f1_table.jld2"))["table"];
table_qm_f2   = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_6M","qm_6000000_screen_profiles_f2_table.jld2"))["table"];
table_qm_f1_1 = load(joinpath(@__DIR__,"simulation_data","qm_simulation_7M","qm_7000000_screen_profiles_f1_table.jld2"))["table"];
table_qm_f2_2 = load(joinpath(@__DIR__,"simulation_data","qm_simulation_7M","qm_7000000_screen_profiles_f2_table.jld2"))["table"];

ks = keys(table_qm_f1);

zqm_old  = [table_old[(1,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:nI]
zqm_F1   = [table_qm_f1[(1,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:nI]
zqm_F2   = [table_qm_f2[(1,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:nI]
zqm_F1_1 = [table_qm_f1_1[(1,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:nI]
zqm_F2_2 = [table_qm_f2_2[(1,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:nI]

hcat(Icoils, zqm_old, zqm_F1, zqm_F1_1, 100*(zqm_F1_1 .- zqm_old)./zqm_F1_1, 100*(zqm_F1_1 .- zqm_old) )

hcat(Icoils, zqm_F2, zqm_F2_2, 100*(zqm_F2_2 .- zqm_F2)./zqm_F2_2, 100*(zqm_F2_2 .- zqm_F2) )


plot(Icoils[3:end], zqm_old[3:end], label="3M")
plot!(Icoils[3:end], zqm_F1[3:end], label="6M")
plot!(Icoils[3:end], zqm_F1_1[3:end], label="7M")
plot!(
    title="QM = (1,0.200,0.02)",
    xscale=:log10,
    yscale=:log10,
    xlabel="Coil current (A)",
    ylabel=L"$F=1$ Peak position",
)


jldopen(joinpath("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_7M", "qm_screen_data.jld2"), "r") do file
    keys(file)                # top-level groups
    keys(file["screen"])      # should show I1, I2, ...
end

dataI1 = jldopen(joinpath("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_7M", "qm_screen_data.jld2"), "r") do file
    file["screen/i1"]
end

dataI47 = jldopen(joinpath("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_7M", "qm_screen_data.jld2"), "r") do file
    file["screen/i47"]
end

vy_final_I1  = vcat(dataI1[6],dataI1[7],dataI1[8])[:,5]
vz_final_I1  = vcat(dataI1[6],dataI1[7],dataI1[8])[:,9]
vy_final_I47 = vcat(dataI47[6],dataI47[7],dataI47[8])[:,5]
vz_final_I47 = vcat(dataI47[6],dataI47[7],dataI47[8])[:,9]



mean(vy_final_I47)
sqrt(mean(vy_final_I47.^2))

mean(vz_final_I1)
sqrt(mean(vz_final_I47.^2))

sqrt(mean(vz_final_I47.^2)-mean(vz_final_I47)^2)


function doppler_peak_from_vz(vzlist::AbstractVector;
    weights::Union{Nothing,AbstractVector}=nothing,
    λ_m::Float64 = 770e-9,
    Γν_MHz::Float64 = 5.956,
    pad_MHz::Float64 = 20.0,
    Ngrid::Int = 4001
)
    vz = collect(Float64.(vzlist))
    isempty(vz) && error("vzlist is empty.")

    w = weights === nothing ? ones(Float64, length(vz)) : collect(Float64.(weights))
    length(w) == length(vz) || error("weights must match vzlist length.")
    wsum = sum(w)
    wsum > 0 || error("sum(weights) must be > 0.")

    # Doppler detunings in MHz
    Δ = @. (-vz / λ_m) / 1e6

    # weighted mean Doppler shift (MHz)
    δ_mean = sum(w .* Δ) / wsum

    @inline L(x) = 1.0 / (1.0 + (2.0*x/Γν_MHz)^2)

    δmin = minimum(Δ) - pad_MHz
    δmax = maximum(Δ) + pad_MHz
    δgrid = range(δmin, δmax, length=Ngrid)

    S = Vector{Float64}(undef, length(δgrid))
    for (i, δ) in enumerate(δgrid)
        acc = 0.0
        @inbounds for j in eachindex(Δ)
            acc += w[j] * L(δ - Δ[j])
        end
        S[i] = acc / wsum
    end

    imax = argmax(S)
    δ_peak = δgrid[imax]

    return (; δ_mean, δ_peak, δ_peak_minus_mean = δ_peak - δ_mean, δgrid, S, Δ)
end

out = doppler_peak_from_vz(vz_final_I47)
@show out.δ_mean out.δ_peak out.δ_peak_minus_mean ; 



1.8e-3*650.94/0.32

histogram(vy_final_I1;
    bins = TheoreticalSimulation.FreedmanDiaconisBins(vy_final_I1), 
    normalize = :pdf,
    label = L"$v_{0,y}$", 
    alpha = 0.65, 
    color = :orange,
    xlabel = L"$v_{0,y} \ (\mathrm{m/s})$"
)
histogram!(vy_final_I47;
    bins = TheoreticalSimulation.FreedmanDiaconisBins(vy_final_I47), 
    normalize = :pdf,
    label = L"$v_{0,y}$", 
    alpha = 0.65, 
    color = :blue,
    xlabel = L"$v_{0,y} \ (\mathrm{m/s})$"
)


plot(Icoils[2:end],(zqm_F1 .+ zqm_F2)[2:end]/2,
    xscale=:log10,
    )


println("the nz-binning are: [ ", join(string.(sort(unique(first.(ks)))), ", ")," ]")
println("the σ-values for the smoothing convolution are (mm): [ ", join((@sprintf("%.3f", x) for x in sort(unique(getindex.(ks,2)))), ", ")," ]")
println("the spline fitting smoothing factors are: [ ", join((@sprintf("%.3f", x) for x in sort(unique(getindex.(ks,3)))), ", ")," ]")

data_profile = table_qm_f1_1[(1,0.001,0.001)][47][:z_profile]
zprof = vec(data_profile[:, 1])[1500:end]
Aprof = vec(data_profile[:, 2])[1500:end]

plot(zprof,Aprof)
hline!([maximum(Aprof)/2])
vline!([0.95], line=(:black,:dash,1))
vline!([3.70], line=(:black,:dash,1))
3.70-0.95

using LsqFit, Statistics

# Gaussian + constant background
# model(z, p) is the safest convention for LsqFit
function gauss_bg(z, p)
    A, z0, σ, C = p
    @. A * exp(-0.5 * ((z - z0) / σ)^2) + 0*C
end
# p = [A, z0, sigma, C]

function fit_gaussian_sigma(z::AbstractVector, I::AbstractVector;
                            use_window::Bool=true, window_factor::Float64=2.5)

    z = collect(Float64.(z))
    I = collect(Float64.(I))
    length(z) == length(I) || error("z and I must have the same length")

    # sort by z
    idx = sortperm(z)
    z = z[idx]; I = I[idx]

    # peak guess
    imax = argmax(I)
    z0_guess = z[imax]

    # background guess: median of lowest 20%
    n = length(I)
    lo = max(1, round(Int, 0.2n))
    C_guess = median(sort(I)[1:lo])

    # amplitude guess
    A_guess = max(I[imax] - C_guess, 1e-12)

    # rough sigma guess via weighted second moment
    w = max.(I .- C_guess, 0.0)
    if sum(w) > 0
        sigma_guess = sqrt(sum(w .* (z .- z0_guess).^2) / sum(w))
    else
        sigma_guess = 0.1 * (maximum(z) - minimum(z))
    end
    sigma_guess = max(sigma_guess, 1e-12)

    # optional window around peak
    z_fit = z; I_fit = I
    if use_window
        halfw = window_factor * sigma_guess
        m = (z .>= z0_guess - halfw) .& (z .<= z0_guess + halfw)
        if count(m) ≥ 6
            z_fit = z[m]; I_fit = I[m]
        end
    end

    p0 = [A_guess, z0_guess, sigma_guess, C_guess]

    fit = curve_fit(gauss_bg, z_fit, I_fit, p0)
    p = coef(fit)                # [A, z0, sigma, C]

    # covariance & 1-sigma errors (approx)
    # cov = estimate_covar(fit)
    # perr = sqrt.(diag(cov))
    sigma = p[3]
    # sigma_err = perr[3]

    fwhm = 2 * sqrt(2 * log(2)) * sigma
    # fwhm_err = 2 * sqrt(2 * log(2)) * sigma_err

    return (; A=p[1], z0=p[2], sigma=sigma, C=p[4],
            # sigma_err=sigma_err, 
            fwhm=fwhm, 
            # fwhm_err=fwhm_err, 
            # cov=cov, 
            fit=fit
            )
end

function halfmax_data(z::AbstractVector, I::AbstractVector; pad::Int=0)
    z = vec(Float64.(z))
    I = vec(Float64.(I))
    length(z) == length(I) || error("z and I must have same length")

    # sort by z so the kept region is contiguous
    idx = sortperm(z)
    z = z[idx]; I = I[idx]

    # same background estimate as your fitting code
    n  = length(I)
    lo = max(1, round(Int, 0.2n))
    C  = median(sort(I)[1:lo])

    imax = argmax(I)
    Imax = I[imax]

    Ihalf = C + 0.90*(Imax - C)

    m = I .>= Ihalf
    if count(m) < 6
        error("Too few points above half max (got $(count(m))).")
    end

    # keep a contiguous block from first..last above half max
    inds = findall(m)
    i1 = max(1, first(inds) - pad)
    i2 = min(n, last(inds) + pad)

    return z[i1:i2], I[i1:i2], Ihalf, C
end



z_half, I_half, Ihalf, C = halfmax_data(zprof,Iprof; pad=0)

out = fit_gaussian_sigma(z_half,I_half)
# @show out.sigma out.sigma_err out.fwhm out.fwhm_err
@show out.sigma out.fwhm 

zfit = range(minimum(zprof), maximum(zprof), length=5000)
Ifit = gauss_bg(zfit, coef(out.fit))

scatter(zprof, Iprof; ms=3, label="data", xlabel="z", ylabel="Intensity",marker=(:white,:circle,2))
plot!(zfit, Ifit; lw=2, label="fit")






data_profile[:,1]










fig = plot(xlims = (-2,2),
    xlabel=L"$z$ (mm)")
for i in sort(unique(getindex.(ks,1)))
    set1 = table_qm_f1[(i,0.001,0.001)][1][:z_profile]
    plot!(fig,
        set1[:,1],set1[:,2],
        label=L"$n_{z}=%$(i)$")
    display(fig)
end

fig = plot(xlims = (-2,2),
    xlabel=L"$z$ (mm)")
set0 = table_qm_f1[(2,0.001,0.001)][1][:z_profile]
plot!(fig,
    set0[:,1],set0[:,2],
    label=L"$\sigma_{z}=0$ μm (raw)")
for i in sort(unique(getindex.(ks,2)))
    set1 = table_qm_f1[(2,i,0.001)][1][:z_profile]
    plot!(fig,
        set1[:,1],set1[:,3],
        label=L"$\sigma_{z}=%$(1e3*i)$ μm")
    display(fig)
end

fig = plot(xlims = (-2,2),
    xlabel=L"$z$ (mm)")
set0 = table_qm_f1[(2,0.001,0.001)][1][:z_profile]
plot!(fig,
    set0[:,1],set0[:,2],
    label=L"$\sigma_{z}=0$ μm (raw)")
for i in sort(unique(getindex.(ks,3)))
    set1 = table_qm_f1[(2,0.001,i)][1][:z_profile]
    plot!(fig,
        set1[:,1],set1[:,3],
        label=L"$\lambda_{0}=%$(i)$")
    display(fig)
end



zqm  = [table_qm_f1[(2,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:47]
zold = [table_old[(4,0.200,0.02)][v][:z_max_smooth_spline_mm] for v=1:47]

println(hcat(zqm,zold))

plot(Ic[2:end],zqm[2:end])
plot!(Ic[2:end],zold[2:end])
plot!(xscale=:log10,
yscale=:log10,
xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)

plot(Ic[2:end],abs.(1e3*(zqm[2:end]-zold[2:end])),
    label="6M – 3M",
    ylabel="Difference (μm)",
    xscale=:log10,
    yscale=:log10,
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    ylims=(1e-2,5))


fig = plot(ylabel=L"$z$ (mm)",
    xlabel=L"$I_{c}$ (A)")
for i in sort(unique(getindex.(ks,1)))
    zqm_F1  = [table_qm_f1[(i,0.150,0.001)][v][:z_max_smooth_spline_mm] for v=1:47]
    fig = plot!(Ic[10:end], zqm_F1[10:end],label=L"$n_{z}=%$(i)$")
end
plot!(xscale=:log10,
    yscale=:log10,
    xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 
        [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)

    sort(unique(getindex.(ks,2)))

data_exp = load(joinpath(@__DIR__,"20250820","data_processed.jld2"),"data")

fig = plot(xlabel="pixel")
zpix = 1e3*TheoreticalSimulation.pixel_coordinates(2560,1,6.5e-6)
for i =1:10
    plot!(fig,zpix,vec(mean(data_exp[:F1ProcessedImages][:,:,i,1],dims=1)), ls=:dash)
    plot!(fig,zpix,vec(mean(data_exp[:F2ProcessedImages][:,:,i,1],dims=1)), ls=:dot)
end
plot!(fig,zpix,vec(mean(data_exp[:F1ProcessedImages][:,:,:,1],dims=(3,1))), line=(:solid,:black))
plot!(fig,zpix,vec(mean(data_exp[:F2ProcessedImages][:,:,:,1],dims=(3,1))), line=(:solid,:black))
display(fig)

fig = plot(xlabel=L"$x$ (mm)")
xpix = 1e3*TheoreticalSimulation.pixel_coordinates(2160,4,6.5e-6)
for i =1:10
    plot!(fig,xpix,vec(mean(data_exp[:F1ProcessedImages][:,:,i,1],dims=2)), ls=:dash)
    plot!(fig,xpix,vec(mean(data_exp[:F2ProcessedImages][:,:,i,1],dims=2)), ls=:dot)
end
plot!(fig,xpix,vec(mean(data_exp[:F1ProcessedImages][:,:,:,1],dims=(3,2))), line=(:solid,:black))
plot!(fig,xpix,vec(mean(data_exp[:F2ProcessedImages][:,:,:,1],dims=(3,2))), line=(:solid,:black))
display(fig)




data_qm_profiles = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_6M","qm_6000000_screen_profiles.jld2"),"profiles")

joinpath("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_6M\\qm_6000000_screen_data.jld2")

data_i0 = jldopen("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_6M\\qm_6000000_screen_data.jld2", "r") do f
    f["alive"][:data][1]
end


jldopen("Y:\\SingleSternGerlach\\simulations\\quantum_simulation_6M\\qm_6000000_screen_data.jld2","r") do f
    println(typeof(f["alive"]))
end
