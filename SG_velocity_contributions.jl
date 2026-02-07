# Simulation of atom trajectories in the Stern–Gerlach experiment
# Kelvin Titimbo
# California Institute of Technology
# November 2025

#  Plotting Setup
# ENV["GKS_WSTYPE"] = "100"
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
const TEMP_DIR = joinpath(@__DIR__,"artifacts", "JuliaTemp")
isdir(TEMP_DIR) || mkpath(TEMP_DIR);
ENV["TMPDIR"] = TEMP_DIR
ENV["TEMP"]   = TEMP_DIR
ENV["TMP"]    = TEMP_DIR
@info "Temporary directory configured" TEMP_DIR
# General setup
HOSTNAME = gethostname();
@info "Running on host" HOSTNAME=HOSTNAME
# Random seeds
base_seed_set = 145;
rng_set = MersenneTwister(base_seed_set)
# rng_set = TaskLocalRNG();
# Custom modules
include("./Modules/atoms.jl");
include("./Modules/samplings.jl");
include("./Modules/TheoreticalSimulation.jl");
using .TheoreticalSimulation;
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
# atom_info       = AtomicSpecies.atoms(atom);
K39_params = AtomParams(atom); # [R μn γn Ispin Ahfs M ] 
# Math constants
const TWOπ = 2π;
const INV_E = exp(-1);
quantum_numbers = fmf_levels(K39_params);

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
x_position = pixel_coordinates(x_pixels, sim_bin_x, sim_pixelsize_x);
z_position = pixel_coordinates(z_pixels, sim_bin_z, sim_pixelsize_z);
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
effusion_params = BeamEffusionParams(x_furnace,z_furnace,x_slit,z_slit,y_FurnaceToSlit,T_K,K39_params);
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
TheoreticalSimulation.default_c_aperture        = R_aper;
TheoreticalSimulation.default_y_SGToAperture    = y_SGToAperture;
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
##################################################################################################
nI_chosen = 15;
nx_bins , nz_bins = 2, 2
width_mm = 0.200 ; 
λ_raw    = 0.01 ;
λ_smooth = 0.001 ;
hist_mode = :density ;

# Fixed analysis limits
xlim = (-8.0, 8.0);
zlim = (-13, 13);
xmin, xmax = xlim;
zmin, zmax = zlim;

# Bin size in mm (default_camera_pixel_size is assumed global in meters)
x_bin_size = 1e3 * nx_bins * cam_pixelsize;
z_bin_size = 1e3 * nz_bins * cam_pixelsize;

# --------------------------------------------------------
# X edges: force symmetric centers around 0
# --------------------------------------------------------
x_half_range = max(abs(xmin), abs(xmax));
kx = max(1, ceil(Int, x_half_range / x_bin_size));
centers_x = collect((-kx:kx) .* x_bin_size);
edges_x = collect((-(kx + 0.5)) * x_bin_size : x_bin_size : ((kx + 0.5) * x_bin_size));

# --------------------------------------------------------
# Z edges: force symmetric centers around 0
# --------------------------------------------------------
z_half_range = max(abs(zmin), abs(zmax));
kz = max(1, ceil(Int, z_half_range / z_bin_size));
centers_z = collect((-kz:kz) .* z_bin_size);
edges_z = collect((-(kz + 0.5)) * z_bin_size : z_bin_size : ((kz + 0.5) * z_bin_size));
##################################################################################################
qm_data  = load(joinpath(@__DIR__,"simulation_data","quantum_simulation_3m","qm_3000000_screen_data.jld2"))["alive"]

jldopen( joinpath(@__DIR__,"simulation_data","qm_simulation_7M","qm_screen_data.jld2"), "r" ) do file
    keys(file["screen"])
end

qm_at_nI = QM_analyze_profiles_to_dict(
    qm_data, K39_params;
    manifold=:F_bottom, n_bins=(nx_bins, nz_bins), width_mm=width_mm,
    add_plot=false, plot_xrange=:all, λ_raw=λ_raw, λ_smooth=λ_smooth, mode=hist_mode
)[nI_chosen]
##################################################################################################
cqd_data0 = load(joinpath(@__DIR__,"simulation_data","cqd_simulation_2.8m","cqd_2800000_ki32_up_screen.jld2"))["screen"]
function remove_post_SG(cqd_in)
    """
    Return a *new* CQD dictionary where, for each entry in `:data`,
    only the particles that pass the post-SG aperture are kept.

    Input:
        cqd_in[:data] :: Dict or OrderedDict
            Each value is an N×7 matrix with columns:
            x0, y0, z0, v0x, v0y, v0z, θe

    Output:
        cqd_out :: same structure as cqd_in, but with rows removed
                   for particles that did not pass the aperture.
    """

    # Make a deep copy so we don't mutate the original structure
    cqd_out   = deepcopy(cqd_in)
    data_dict = cqd_out[:data]

    # Collect keys once (could also just iterate over the dict)
    data_keys = collect(keys(data_dict))

    # --- Precompute geometry (same for all keys) ---
    L_total = y_FurnaceToSlit + y_SlitToSG + y_SG + y_SGToAperture
    L_SG    = y_SG
    L_to_ap = y_SGToAperture

    # --- Precompute magnetic / force parameters (same for all keys) ---
    B  = TheoreticalSimulation.BvsI(Icoils[nI_chosen])
    G  = TheoreticalSimulation.GvsI(Icoils[nI_chosen])
    a0 = μₑ * G / K39_params.M
    kw = abs(2.20e-6 * γₑ * B)

    for k in data_keys
        data_sampled = data_dict[k]
        nn = size(data_sampled, 1)

        pass = falses(nn)  # BitVector

        for i = 1:nn
            x0, y0, z0, v0x, v0y, v0z, θe = data_sampled[i, 1:7]

            # position at aperture plane
            x3 = x0 + (v0x / v0y) * L_total

            # Z displacement: classical SG + spin rotation physics
            term1 = z0 + (v0z / v0y) * L_total
            term2 = 0.5 * a0 / v0y^2 * ((L_SG + L_to_ap)^2 - L_to_ap^2)
            term3 = a0/kw * (L_SG / v0y) * (
                        log(cos(θe/2)^2) +
                        (L_to_ap/L_SG) * log(
                            cos(θe/2)^2 +
                            exp(-kw*L_SG/v0y) * sin(θe/2)^2
                        )
                    )
            term4 = 0.5/kw^2 * a0 * (
                        TheoreticalSimulation.polylogarithm(2,
                            -exp(-kw*L_SG/v0y) * tan(θe/2)^2
                        ) -
                        TheoreticalSimulation.polylogarithm(2,
                            -tan(θe/2)^2
                        )
                    )

            z3 = term1 + term2 + term3 + term4

            # radial check at the aperture
            r2 = x3^2 + z3^2

            if r2 < R_aper^2
                pass[i] = true
            end
        end
        println("Particles passing the post SG circular aperture for I=$(@sprintf("%1.3f", Icoils[k]))A : $(@sprintf("%2.2f",100*sum(pass)/nn))%")
        # Keep only passing particles
        data_dict[k] = data_sampled[pass, :]
    end

    return cqd_out
end
cqd_data = remove_post_SG(cqd_data0)
cqd_data0 = nothing
cqd_at_nI = CQD_analyze_profiles_to_dict(cqd_data;
    n_bins=(nx_bins, nz_bins), width_mm=width_mm,
    branch=:up, add_plot=false, plot_xrange=:all,
    λ_raw=λ_raw, λ_smooth=λ_smooth, mode=hist_mode)[nI_chosen]
##################################################################################################
qm_data[:Icoils] == cqd_data[:Icoils] == Icoils ? 
    (@info "QM and CQD are simulated for the same currents") : 
    (@warn "QM and CQD are simulated for different currents")
GC.gc()
##################################################################################################
qm_F1_pool = sortslices(
                vcat([qm_data[:data][nI_chosen][i] for i in Int(2*K39_params.Ispin+3):Int(4*K39_params.Ispin+2)]...), 
                dims=1, by=row->(row[8], row[5])
            )
cqd_up_pool = sortslices(cqd_data[:data][nI_chosen], 
                dims=1, by=row->(row[10], row[5])
            )

θmax = asin(effusion_params.sinθmax);
mean_vy_i0  = sqrt(π/2 *effusion_params.α2) * (cos(θmax) + 0.5*sec(θmax/2)^2);
mean_vy_qm  = mean(qm_F1_pool[:,5]);
mean_vy_cqd = mean(cqd_up_pool[:,5]);

function make_intervals(row, n::Integer)
    @assert n > 0 "n must be > 0"
    mn, mx = extrema(row)
    mn = floor(mn)
    mx = ceil(mx)
    edges  = range(mn, mx; length = n + 1)
    return collect(zip(edges[1:end-1], edges[2:end]))
end

n_v0y = 5
v0y_intervals = make_intervals(vcat(qm_F1_pool[:,5],cqd_up_pool[:,5]), n_v0y)

p = plot(
    title  = L"Velocity splitting for $I_{c}=%$(Icoils[nI_chosen])\mathrm{A}$",
    xlabel = L"$v_{0,y} \ (\mathrm{m}/\mathrm{s})$",
    ylabel = nothing,
    xticks = 0:250:2000,
    yticks = ([1,2],["CQD","QM"]),
    xlims = (0,2000),
    ylims = (0,3),
    size=(1000,150),
    widen=false,
    yminorticks = nothing,
    top_margin = 5mm,
    bottom_margin = 7mm,
);
vline!(p,[mean_vy_i0], 
    label=L"$\langle v_{0,y} \rangle = %$(round(mean_vy_i0, sigdigits=3))\mathrm{m/s}$",
    line=(:dash,:red,1.5));
vline!(p,[mean_vy_qm],
    label=L"$\langle v_{0,y} \rangle_{\mathrm{QM}} = %$(round(mean_vy_qm, sigdigits=3))\mathrm{m/s}$",
    line=(:dot,:grey36,1.0));
vline!(p,[mean_vy_cqd],
    label=L"$\langle v_{0,y} \rangle_{\mathrm{CQD}} = %$(round(mean_vy_cqd, sigdigits=3))\mathrm{m/s}$",
    line=(:dot,:gray36,1.0));
vspan!(p,[0,minimum(first.(v0y_intervals))], color=:black, fillalpha=0.60, label=false);
vspan!(p,[maximum(last.(v0y_intervals)),2000], color=:black, fillalpha=0.60, label=false);
# Color palette for the bands
colors = palette(:viridis, n_v0y);
pop_percentage_vi = zeros(2,n_v0y);
# Add vertical colored spans for each interval
for (i, (lo, hi)) in enumerate(v0y_intervals)
    n_qm  = length(findall(row -> lo <= row[5] <= hi, eachrow(qm_F1_pool))) / size(qm_F1_pool,1) 
    n_cqd = length(findall(row -> lo <= row[5] <= hi, eachrow(cqd_up_pool))) / size(cqd_up_pool,1)
    vspan!(p, [lo, hi];
        color = colors[i],
        alpha = 0.5,   # transparency
        label = ""
    )
    annotate!(p,(hi+lo)/2,2.0, text("$(@sprintf("%2.2f",100*n_qm))%", 6, :bold, "Helvetica"))
    annotate!(p,(hi+lo)/2,1.0, text("$(@sprintf("%2.2f",100*n_cqd))%",6, :bold, "Helvetica"))
    pop_percentage_vi[1,i] = n_qm;
    pop_percentage_vi[2,i] = n_cqd;
end
plot!(p,legend=:outerright,
    legend_columns = 1,
    foreground_color_legend=nothing,
    background_color_legend=nothing,)
display(p)



# ----------------------------------------------------------------------------------------------------------------
# QUANTUM MECHANICS
# ----------------------------------------------------------------------------------------------------------------
profiles_qm_v   = Matrix{Float64}(undef, length(centers_z), n_v0y);
zmax_qm_v       = zeros(n_v0y);
for i =1:n_v0y
    qm_idx = findall(row -> v0y_intervals[i][1] <= row[5] <= v0y_intervals[i][2], eachrow(qm_F1_pool))
    d_qm = qm_F1_pool[qm_idx,:]

        # 2D histogram
    x = 1e3*d_qm[:, 7]
    z = 1e3*d_qm[:, 8]

    if hist_mode === :none
        h = StatsBase.fit(Histogram, (x, z), (edges_x, edges_z))                    # raw counts (no normalization)
    elseif hist_mode in (:probability, :pdf, :density)
        h = normalize(StatsBase.fit(Histogram, (x, z), (edges_x, edges_z)); mode=hist_mode)
    else
        throw(ArgumentError("mode must be one of :pdf, :density, :probability, :none, got $hist_mode"))
    end
    counts = h.weights  # size: (length(centers_x), length(centers_z))

    hfig = histogram2d(x, z;
        bins = (FreedmanDiaconisBins(x), FreedmanDiaconisBins(z)),
        show_empty_bins = true, color = :plasma, normalize=hist_mode,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{mm})$",
        # xticks = -4.0:0.50:4.0, yticks = -1000:100:1000,
        # xlims=(-4,4), ylims=(-250,250),
    )
    display(hfig)

    # hmfig = heatmap(centers_x, centers_z, counts', 
    #     xlabel="x (mm)", 
    #     ylabel="z (mm)", 
    #     title="2D Histogram", 
    #     # xlims=(-6.5,6.5),
    #     # ylims=(0,2)
    # )
    # display(hmfig)

    # z-profile = mean over x bins
    z_profile_raw = vec(mean(counts, dims = 1))
    z_max_raw_mm = centers_z[argmax(z_profile_raw)]
    z_max_raw_spline_mm, Sfit_raw = TheoreticalSimulation.max_of_bspline_positions(centers_z,z_profile_raw;λ0=λ_raw)
    # Smoothing
    z_profile_smooth = TheoreticalSimulation.smooth_profile(centers_z, z_profile_raw, width_mm)
    z_max_smooth_mm = centers_z[argmax(z_profile_smooth)]
    z_max_smooth_spline_mm, Sfit_smooth = TheoreticalSimulation.max_of_bspline_positions(centers_z,z_profile_smooth;λ0=λ_smooth)
    # Combine into one matrix for convenience: [z raw smooth]
    z_profile = hcat(
        centers_z,
        z_profile_raw,
        z_profile_smooth,
    )

    out = (
        z_profile = z_profile,
        z_max_raw_mm = z_max_raw_mm,
        z_max_raw_spline_mm = z_max_raw_spline_mm[1],
        z_max_smooth_mm = z_max_smooth_mm,
        z_max_smooth_spline_mm = z_max_smooth_spline_mm[1]
    )

    p = plot(z_profile[:,1],z_profile[:,3],
        label=L"Profile $v_{y}=%$(v0y_intervals[i])\mathrm{m/s}$",
        line=(:dot,colors[i],2),
        legend=:topleft,
        background_color_legend=nothing,
        foreground_color_legend=nothing,)
    plot!(z_profile[:,1],z_profile[:,3],
        label=nothing,
        line=nothing,
        color=colors[i],
        fillrange=0,
        fillalpha=0.10,
        fillstyle=nothing,)
    # plot!(z_profile[:,1],z_profile[:,3])
    # plot!(xlims=[0,4])
    # vline!([out.z_max_raw_mm])
    # vline!([out.z_max_raw_spline_mm])
    # vline!([out.z_max_smooth_mm], label=out.z_max_smooth_mm)
    vline!([out.z_max_smooth_spline_mm], 
        label=L"$z_{\mathrm{max}} = %$(round(out.z_max_smooth_spline_mm, sigdigits=5)) \mathrm{mm}$",
        line=(:dash, :blue,1))
    vline!([qm_at_nI[:z_max_smooth_spline_mm]], 
        label=L"Global $z_{\mathrm{max}} = %$(round(qm_at_nI[:z_max_smooth_spline_mm], sigdigits=5)) \mathrm{mm}$",
        line=(:solid,:red,1.5))
    display(p)

    profiles_qm_v[:,i] = z_profile[:,3]
    zmax_qm_v[i] = out.z_max_smooth_spline_mm
end


f00 = plot(
    title="Quantum Mechanics Simulation",
    xlabel=L"$z$ (mm)",
    ylabel="Counts"
)
plot!(f00, qm_at_nI[:z_profile][:,1], qm_at_nI[:z_profile][:,3],
    label=L"Full profile $z_{\mathrm{max}} = %$(round(qm_at_nI[:z_max_smooth_spline_mm], sigdigits=5)) \mathrm{mm}$",
    line=(:solid,:red,2))
vline!(f00,[qm_at_nI[:z_max_smooth_spline_mm]], 
    label=nothing,
    line=(:dash,:red,1.5))
plot!(f00,qm_at_nI[:z_profile][:,1], qm_at_nI[:z_profile][:,3],
    label=nothing,
    line=nothing,
    color=:red,
    fillrange=0,
    fillalpha=0.15,
    fillstyle=nothing,)
for i=1:n_v0y
    plot!(f00,centers_z, profiles_qm_v[:,i],
            label=L"$v_{y}=%$(round.(v0y_intervals[i], sigdigits=4))\mathrm{m/s}$ | $z_{\mathrm{max}} = %$(round(zmax_qm_v[i], sigdigits=5)) \mathrm{mm}$",
            line=(:dot,colors[i],2),)
    plot!(f00,centers_z,profiles_qm_v[:,i],
        label=nothing,
        line=nothing,
        color=colors[i],
        fillrange=0,
        fillalpha=0.10,
        fillstyle=nothing,)
    vline!(f00,[zmax_qm_v[i]], 
        label=nothing,
        line=(:dash, colors[i],1))
end
plot!(f00,
    xlims=(-1,10),
    xlabel=L"$z$ (mm)",
    size=(1000,400),
    left_margin=4mm,
    bottom_margin=5mm,
    legend=:outerright,
    legend_columns=1,
    background_color_legend=nothing,
    foreground_color_legend=nothing,
)
display(f00)

# ----------------------------------------------------------------------------------------------------------------
# CO-QUANTUM DYNAMICS
# ----------------------------------------------------------------------------------------------------------------
profiles_cqd_v = Matrix{Float64}(undef, length(centers_z), n_v0y);
zmax_cqd_v = zeros(n_v0y)
for i =1:n_v0y
    cqd_idx = findall(row -> v0y_intervals[i][1] <= row[5] <= v0y_intervals[i][2], eachrow(cqd_up_pool))
    d_cqd = cqd_up_pool[cqd_idx,:]

    # 2D histogram
    x = 1e3*d_cqd[:, 9]
    z = 1e3*d_cqd[:, 10]

    if hist_mode === :none
        h = StatsBase.fit(Histogram, (x, z), (edges_x, edges_z))                    # raw counts (no normalization)
    elseif hist_mode in (:probability, :pdf, :density)
        h = normalize(StatsBase.fit(Histogram, (x, z), (edges_x, edges_z)); mode=hist_mode)
    else
        throw(ArgumentError("mode must be one of :pdf, :density, :probability, :none, got $hist_mode"))
    end
    counts = h.weights  # size: (length(centers_x), length(centers_z))

    hfig = histogram2d(x, z;
        bins = (FreedmanDiaconisBins(x), FreedmanDiaconisBins(z)),
        show_empty_bins = true, color = :plasma, normalize=hist_mode,
        xlabel = L"$x \ (\mathrm{mm})$", ylabel = L"$z \ (\mathrm{mm})$",
        # xticks = -4.0:0.50:4.0, yticks = -1000:100:1000,
        # xlims=(-4,4), ylims=(-250,250),
    )
    display(hfig)

    # hmfig = heatmap(centers_x, centers_z, counts', 
    #     xlabel="x (mm)", 
    #     ylabel="z (mm)", 
    #     title="2D Histogram", 
    #     # xlims=(-6.5,6.5),
    #     # ylims=(0,2)
    # )
    # display(hmfig)

    # z-profile = mean over x bins
    z_profile_raw = vec(mean(counts, dims = 1))
    z_max_raw_mm = centers_z[argmax(z_profile_raw)]
    z_max_raw_spline_mm, Sfit_raw = TheoreticalSimulation.max_of_bspline_positions(centers_z,z_profile_raw;λ0=λ_raw)
    # Smoothing
    z_profile_smooth = TheoreticalSimulation.smooth_profile(centers_z, z_profile_raw, width_mm)
    z_max_smooth_mm = centers_z[argmax(z_profile_smooth)]
    z_max_smooth_spline_mm, Sfit_smooth = TheoreticalSimulation.max_of_bspline_positions(centers_z,z_profile_smooth;λ0=λ_smooth)
    # Combine into one matrix for convenience: [z raw smooth]
    z_profile = hcat(
        centers_z,
        z_profile_raw,
        z_profile_smooth,
    )

    out = (
        z_profile = z_profile,
        z_max_raw_mm = z_max_raw_mm,
        z_max_raw_spline_mm = z_max_raw_spline_mm[1],
        z_max_smooth_mm = z_max_smooth_mm,
        z_max_smooth_spline_mm = z_max_smooth_spline_mm[1]
    )

    p = plot(z_profile[:,1],z_profile[:,3],
        label=L"Profile $v_{y}=%$(v0y_intervals[i])\mathrm{m/s}$",
        line=(:dot,colors[i],2),
        legend=:topleft,
        background_color_legend=nothing,
        foreground_color_legend=nothing,)
    plot!(z_profile[:,1],z_profile[:,3],
        label=nothing,
        line=nothing,
        color=colors[i],
        fillrange=0,
        fillalpha=0.10,
        fillstyle=nothing,)
    # plot!(z_profile[:,1],z_profile[:,3])
    # plot!(xlims=[0,4])
    # vline!([out.z_max_raw_mm])
    # vline!([out.z_max_raw_spline_mm])
    # vline!([out.z_max_smooth_mm], label=out.z_max_smooth_mm)
    vline!([out.z_max_smooth_spline_mm], 
        label=L"$z_{\mathrm{max}} = %$(round(out.z_max_smooth_spline_mm, sigdigits=5)) \mathrm{mm}$",
        line=(:dash, :blue,1))
    vline!([qm_at_nI[:z_max_smooth_spline_mm]], 
        label=L"Global $z_{\mathrm{max}} = %$(round(qm_at_nI[:z_max_smooth_spline_mm], sigdigits=5)) \mathrm{mm}$",
        line=(:solid,:red,1.5))
    display(p)

    profiles_cqd_v[:,i] = z_profile[:,3]
    zmax_cqd_v[i] = out.z_max_smooth_spline_mm
end


f01 = plot(
    title="Co-Quantum Dynamics Simulation",
    xlabel=L"$z$ (mm)",
    ylabel="Counts"
)
plot!(f01, cqd_at_nI[:z_profile][:,1], cqd_at_nI[:z_profile][:,3],
    label=L"Full profile $z_{\mathrm{max}} = %$(round(cqd_at_nI[:z_max_smooth_spline_mm], sigdigits=5)) \mathrm{mm}$",
    line=(:solid,:red,2))
vline!(f01,[cqd_at_nI[:z_max_smooth_spline_mm]], 
    label=nothing,
    line=(:dash,:red,1.5))
plot!(f01,cqd_at_nI[:z_profile][:,1], cqd_at_nI[:z_profile][:,3],
    label=nothing,
    line=nothing,
    color=:red,
    fillrange=0,
    fillalpha=0.15,
    fillstyle=nothing,)
for i=1:n_v0y
    plot!(f01,centers_z, profiles_cqd_v[:,i],
            label=L"$v_{y}=%$(round.(v0y_intervals[i], sigdigits=4))\mathrm{m/s}$ | $z_{\mathrm{max}} = %$(round(zmax_cqd_v[i], sigdigits=5)) \mathrm{mm}$",
            line=(:dot,colors[i],2),)
    plot!(f00,centers_z,profiles_cqd_v[:,i],
        label=nothing,
        line=nothing,
        color=colors[i],
        fillrange=0,
        fillalpha=0.10,
        fillstyle=nothing,)
    vline!(f01,[zmax_qm_v[i]], 
        label=nothing,
        line=(:dash, colors[i],1))
end
plot!(f01,
    xlims=(-1,10),
    xlabel=L"$z$ (mm)",
    size=(1000,400),
    left_margin=4mm,
    bottom_margin=5mm,
    legend=:outerright,
    legend_columns=1,
    background_color_legend=nothing,
    foreground_color_legend=nothing,
)
display(f01)



OrderedDict(:Ic             => Icoils[nI_chosen],
            :n_v0y          => n_v0y,
            :v0y_intervals  => v0y_intervals,
            :z_centers      => centers_z,
            :QM             => OrderedDict(
                                :z_max      => zmax_qm_v
                                :profiles   => profiles_qm_v
                                ),
            # :CQD            => OrderedDict(
            #                     :z_max      => zmax_cqd_v
            #                     :profiles   => profiles_cqd_v
            #                     )
)


histogram(vcat(qm_data[:data][47][6],qm_data[:data][1][7],qm_data[:data][1][8])[:,8], normalize=:pdf)
histogram!(qm_data[:data][47][6][:,8], normalize=:pdf)
histogram!(qm_data[:data][47][7][:,8], normalize=:pdf)
histogram!(qm_data[:data][47][8][:,8], normalize=:pdf)

# Normalized field/gradient profile G(y)  (max = 1)
function GProfile(y, win, wout; yin  = y_FurnaceToSlit+y_SlitToSG, yout = y_FurnaceToSlit+y_SlitToSG+y_SG )
    # Smooth logistic sigmoid
    σ(y) = 1 / (1 + exp(-y))

    # Smooth top-hat with sigmoid tails
    windowSigmoid(y, y1, y2, win, wout) = σ((y - y1)/win) * (1 - σ((y - y2)/wout))


    # grid for normalization
    ygrid = range(0.5*yin, 2*yout; length=20000)
    wvals = windowSigmoid.(ygrid, yin, yout, win, wout)
    norm  = maximum(wvals)

    return windowSigmoid(y, yin, yout, win, wout) / norm
end

# Parameters
y_in  = y_FurnaceToSlit+y_SlitToSG;
y_out = y_in + y_SG;
win  = 0.002
wout = 0.002

# The actual acceleration/field profile
G(y) = GProfile(y, win, wout)

plot(y -> G(y), 0.20, 0.40,
     title="Normalized Smooth Top-Hat Field Profile",
     label="Profile",
     xlabel=L"Propagation axis $y$ (m)",
     ylabel=L"Normalized profile $G(y)$",
     line=(:solid,:red,2.5))
vline!([y_in, y_out], line=(:dashdot,:black,2), label=false)
vspan!([y_in,y_out], label="SG magnet", color=:orangered2,fillalpha=0.2)




function QM_mynum(x0::Real, v0::Real, 
                    tspan::Tuple{Real,Real},
                    v0y::Real,
                    Ix::Real,
                    pk::AtomParams;
                    saveat=nothing, reltol=1e-18, abstol=1e-18)

    # --- Time markers ---
    t1 = (y_FurnaceToSlit + y_SlitToSG) / v0y
    t2 = (y_FurnaceToSlit + y_SlitToSG + y_SG) / v0y
    @assert tspan[1] ≤ t1 ≤ t2 ≤ tspan[2] "Require tspan[1] ≤ t1 ≤ t2 ≤ tspan[2]"
    
    # --- Magnetic moments for all sublevels ---
    μ_list = [μF_effective(Ix,f, mf,K39_params) for (f,mf) in fmf_levels(pk)]

    G = TheoreticalSimulation.GvsI(Ix)
    results = Vector{ODESolution}(undef, length(μ_list))

    # --- Integrate for each spin sublevel ---
    for (i, μ) in enumerate(μ_list)
        a_scale = (μ * G) / pk.M

        function f!(du, u, p, t)
            x, v = u
            du[1] = v
            du[2] = (t < t1 || t > t2) ? 0.0 : a_scale
        end

        u0 = (float(x0), float(v0))
        prob = ODEProblem(f!, collect(u0), (float(tspan[1]), float(tspan[2])))
        results[i] = solve(prob, RadauIIA5();
                           tstops=(t1, t2),
                           saveat=saveat,
                           reltol=reltol,
                           abstol=abstol)
    end

    return results
end


QM_mynum(z0, v0z, (0.0, 1.2*T), v0y, I_sk, K39_params; saveat=0:1e-9:1.2*T);




# ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------- #

function standard_error(x)
    return std(x; corrected=true) ./ sqrt.(length(x))
end

cqd_ki = load(joinpath(@__DIR__,"simulation_data","cqd_simulation_2.8m","cqd_2800000_ki0001_up_screen.jld2"))["screen"]

Icoils = cqd_ki[:Icoils][14:end]
ki_fit = 2.50e-6
number_precessions = round(1/(TWOπ*ki_fit))
collapse_time = inv.(ki_fit * abs(γₑ)* TheoreticalSimulation.BvsI.(Icoils))

plot(Icoils,1e6*collapse_time,
    label=L"Collapse time $\tau_{c}$",
    line=(:blue,2),
    xlabel="Current (A)",
    ylabel="Collapse time (μs)",
    xaxis=:log10,
    yaxis=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1, 10, 100], [L"10^{0}", L"10^{1}", L"10^{2}"]),
    xlims=(1e-2,1),
    ylims=(1,200)
)


 [length(cqd_ki[:data][ic][:,5]) for ic=14:47]

travel_times = [mean(inv.(cqd_ki[:data][ic][:,5] / y_SG)) for ic=14:47]
plot(Icoils,
    1e6*travel_times,
    ribbon=1e6*[standard_error(inv.(cqd_ki[:data][ic][:,5] / y_SG)) for ic=14:47],
    label=L"Time of flight $\Delta t_{\mathrm{SG}}$",
    line=(:solid,:dodgerblue3,2),
    marker=(:circle,:white,2),
    markerstrokecolor=:dodgerblue3,
    xlabel="Current (A)",
    ylabel="Time of flight (μs)",
    xaxis=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(1e-2,1),
    fillalpha=0.1,
)
@info @sprintf("The mean time of flight is (%d ± %d) μs ",mean(1e6*travel_times),std(1e6*travel_times))

plot(Icoils,
    1e6*travel_times,
    ribbon=1e6*[standard_error(inv.(cqd_ki[:data][ic][:,5] / 0.07)) for ic=14:47],
    line=(:solid,:dodgerblue3,2),
    label=L"Time of flight $\Delta t_{\mathrm{SG}}$",
    xaxis=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(1e-2,1),
    fillalpha=0.1,
)
plot!(Icoils,1e6*collapse_time,
    label=L"Collapse time $\tau_{c}$",
    line=(:darkgreen,2),
    xlabel="Current (A)",
    ylabel="Time (μs)",
    xaxis=:log10,
    yaxis=:log10,
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    yticks = ([1, 10, 100], [L"10^{0}", L"10^{1}", L"10^{2}"]),
    xlims=(1e-2,1),
    ylims=(1,200),
    legend=:bottomleft
)

plot(Icoils,travel_times ./ collapse_time,
    label=L"$\Delta t_{\mathrm{SG}} / \tau_{c}$",
    line=(:red,2),
    xaxis=:log10,
    yaxis=:log10,
    xlims=(1e-2,1),
    ylims=(0.7,40),
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlabel="Current (A)",
    ylabel="Number of collapse times",
    size=(800,600),
    left_margin=3mm,
    legendfontsize=12,
)
hspan!([0.1,1], color=:black, fillalpha=0.2, label=false)

plot(Icoils, inv.(travel_times ./ collapse_time),
    label=L"$\tau_{c} / \Delta t_{\mathrm{SG}} $",
    line=(:red,2),
    xaxis=:log10,
    yaxis=:log10,
    xlims=(1e-2,1),
    # ylims=(0.7,40),
    xticks = ([1e-2, 1e-1, 1.0], [L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlabel="Current (A)",
    ylabel= nothing,
    size=(800,600),
    left_margin=3mm,
    legendfontsize=12,
)
# hspan!([0.1,1], color=:black, fillalpha=0.2, label=false)


df_scl=CSV.read("d:\\titimbo\\SingleSG_Xukun2025\\data_studies\\20251201T121327660\\rel_error_scaled.csv", DataFrame)
df_org=CSV.read("d:\\titimbo\\SingleSG_Xukun2025\\data_studies\\20251201T121327660\\rel_error_original.csv", DataFrame)


plot(df_scl.Ic, df_scl.eQM ,
    label=L"Rel.Error : $\left( \mathcal{E}_{QM} - \mathcal{E}_{exp} \right) / \mathcal{E}_{exp}$",
    line=(:solid,:red,2))
plot!(df_scl.Ic, df_scl.eCQD ,
    label=L"Rel.Error : $\left( \mathcal{E}_{CQD} - \mathcal{E}_{exp} \right) / \mathcal{E}_{exp}$",
    line=(:solid,:blue,2))
plot!(Icoils,  collapse_time ./ travel_times,
    label=L"\tau_{c} / \Delta t_{SG}",
    line=(:dot,:darkgreen,2))
plot!(
    xlabel = "Current (A)",
    ylabel = nothing,
    xaxis=:log10,
    # yaxis=:log10,
    labelfontsize=14,
    tickfontsize=12,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(0.020,1.05),
    # ylims=(-0.10,0.25),
    size=(800,500),
    legendtitle="Scaled data",
    legendfontsize=12,
    left_margin=3mm,
    bottom_margin=3mm,
)

plot(df_org.Ic, df_org.eQM ,
    label=L"Rel.Error : $\left( \mathcal{E}_{QM} - \mathcal{E}_{exp} \right) / \mathcal{E}_{exp}$",
    line=(:solid,:red,2))
plot!(df_org.Ic, df_org.eCQD ,
    label=L"Rel.Error : $\left( \mathcal{E}_{CQD} - \mathcal{E}_{exp} \right) / \mathcal{E}_{exp}$",
    line=(:solid,:blue,2))
plot!(Icoils,  collapse_time ./ travel_times,
    label=L"\tau_{c} / \Delta t_{SG}",
    line=(:dot,:darkgreen,2))
plot!(
    xlabel = "Current (A)",
    ylabel = nothing,
    xaxis=:log10,
    # yaxis=:log10,
    labelfontsize=14,
    tickfontsize=12,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    # yticks = ([1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims=(0.020,1.05),
    # ylims=(-0.10,0.25),
    size=(800,500),
    legendtitle="Original data",
    legendfontsize=12,
    left_margin=3mm,
    bottom_margin=3mm,
)