# Kelvin Titimbo
# California Institute of Technology
# March 2026

#  Plotting Setup
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
const OUTDIR    = joinpath(@__DIR__, "data_studies", "QM"*RUN_STAMP);
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
include("./Modules/JLD2_MyTools.jl");
include("./Modules/DataReading.jl");
include("./Modules/MyExperimentalAnalysis.jl");
include("./Modules/TheoreticalSimulation.jl");
using .TheoreticalSimulation;
TheoreticalSimulation.SAVE_FIG = SAVE_FIG;
TheoreticalSimulation.FIG_EXT  = FIG_EXT;
TheoreticalSimulation.OUTDIR   = OUTDIR;

println("\n\t\tRunning process on:\t $(RUN_STAMP) \n")

# Math constants
const TWOπ = 2π;
const INV_E = exp(-1);
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
T_K = 273.15 + 200 ; # Furnace temperature (K)
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
function closest_index(A, x)
    return argmin(abs.(A .- x))
end
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

z_theory = collect(range(-0.00832,0.00832,length=4096))

data_directories =  ["20260220", "20260225", "20260226am","20260226pm","20260227", "20260303", "20260306r1", "20260306r2"];
n_data = length(data_directories);

selected_current = 1.000

ni0 = closest_index(Icoils,selected_current);
I0 = Icoils[ni0]

##################################################################################################
nz , σw_mm , λ0 = (1,0.001,0.001)

data_qmf1_path = joinpath(@__DIR__,"simulation_data",
    "QM_T200_8M",
    "qm_screen_profiles_f1_table.jld2");
JLD2_MyTools.summarize_meta_qm_jld2(data_qmf1_path)
qm_f1 =  jldopen(data_qmf1_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end;

data_qmf2_path = joinpath(@__DIR__,"simulation_data",
    "QM_T200_8M",
    "qm_screen_profiles_f2_table.jld2");
JLD2_MyTools.summarize_meta_qm_jld2(data_qmf2_path)
qm_f2 =  jldopen(data_qmf2_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end;

dir_selected = data_directories[8]
data_exp_path = joinpath(@__DIR__,"EXPDATA_ANALYSIS","summary",dir_selected, dir_selected * "_report_summary.jld2");
JLD2_MyTools.show_exp_summary(data_exp_path,dir_selected)
exp_data =  jldopen(data_exp_path,"r") do file
    Ic = file["meta/Currents"]
    data = file[JLD2_MyTools.make_keypath_exp(dir_selected,nz,λ0)];
    C00 = 0.5*(data[:mean_F1_peak_pos_raw ][1] + data[:mean_F2_peak_pos_raw ][1])

    return (Ic = Ic, z=collect(data[:z_mm]) .- C00, F1=data[:F1_profile], F2=data[:F2_profile], F1_profile = data[:F1_profile_spline], F2_profile = data[:F2_profile_spline] ) 
end

ni0_exp = closest_index(exp_data.Ic,selected_current)



##################################################################################################

μeff_1 = [TheoreticalSimulation.μF_effective(I0, v[1], v[2], K39_params)
        for v in TheoreticalSimulation.fmf_levels(K39_params; Fsel=1)]

μeff_2 = [TheoreticalSimulation.μF_effective(I0, v[1], v[2], K39_params)
        for v in TheoreticalSimulation.fmf_levels(K39_params; Fsel=2)]

plot( vcat(μeff_2, μeff_1)./μB,
    label=L"$\mu_{\mathrm{eff}}$ | $I_{0}=%$(Int(1000*I0))$mA",
    marker=(:diamond, 3, :white),
    markerstrokecolor=:black,
    line=(:dot,1,:black),
    xticks=([1,2,3,4,5,6,7,8], [L"%$(Int.(v))" for v in TheoreticalSimulation.fmf_levels(K39_params)]),
    xminorticks=false,
    xrotation=65,
    ylims=(-1.1,1.1),
    xlabel="Quantum levels",
    ylabel=L"$\mu_{F} / \mu_{B}$",
    bottom_margin=5mm,
)


pdf_F1 = TheoreticalSimulation.weighted_QM_PDF_profile(
    I0,
    z_theory,
    K39_params,
    effusion_params;
    Fsel=1,
    weights=nothing,
    normalize=false,
);

pdf_F2 = TheoreticalSimulation.weighted_QM_PDF_profile(
    I0,
    z_theory,
    K39_params,
    effusion_params;
    Fsel=2,
    weights=nothing,
    normalize=false,
);

plot(title=L"Original signal : $n_{z}=%$(nz)$ | $λ_{s}=%$(λ0)$",
    xlims=(-8,8),
    xlabel=L"$z \, (\mathrm{mm})$",
    legend=:outerright,
)
scatter!([0],[0], marker=(:circle, 0.001,:white), fillalpha=0.01, label=L"Analytic: $I_{c}=%$(Int(1000*Icoils[ni0]))\mathrm{mA}$")
plot!(1e3*z_theory, TheoreticalSimulation.normalize_profile(z_theory, pdf_F1; method=:max),
    label=L"$F=1$",
    color=:red,
    fill=true,
    fillalpha=0.25,
    )
plot!(1e3*z_theory, TheoreticalSimulation.normalize_profile(z_theory, pdf_F2; method=:max),
    label=L"$F=2$",
    color=:blue,
    fill=true,
    fillalpha=0.25,
)
scatter!([0],[0], marker=(:circle, 0.001,:white), fillalpha=0.01, label=L"Montecarlo: $I_{c}=%$(Int(1000*Icoils[ni0]))\mathrm{mA}$")
plot!(qm_f1[ni0][:z_profile][:,1] , TheoreticalSimulation.normalize_profile(qm_f1[ni0][:z_profile][:,1], qm_f1[ni0][:z_profile][:,2]; method=:max),
    label=L"$F=1$")
plot!(qm_f2[ni0][:z_profile][:,1] , TheoreticalSimulation.normalize_profile(qm_f2[ni0][:z_profile][:,1], qm_f2[ni0][:z_profile][:,2]; method=:max),
    label=L"$F=2$")
plot!(
    legend_columns=1,
    legendfontsize = 6,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
)
scatter!([0],[0], marker=(:circle, 0.001,:white), fillalpha=0.01, label=L"Experiment %$(dir_selected) : 
    $I_{c}=%$(round(1000*exp_data.Ic[ni0_exp], digits=2))\mathrm{mA}$")
plot!(exp_data.z , TheoreticalSimulation.normalize_profile(exp_data.z , exp_data.F1[ni0_exp,:]; method=:max ),
    label=L"%$(dir_selected): $F=1$",
    line=(:solid,1.0, :orangered2),
)
plot!(exp_data.z , TheoreticalSimulation.normalize_profile(exp_data.z , exp_data.F2[ni0_exp,:]; method=:max ),
    label=L"%$(dir_selected): $F=2$",
    line=(:solid,1.0, :dodgerblue),
)




##################################################################################################
selected_current = 0.100
ni0 = closest_index(Icoils,selected_current);
I0 = Icoils[ni0]

nz , σw_mm , λ0 = (2,0.200,0.005)
##################################################################################################
data_qmf1_path = joinpath(@__DIR__,"simulation_data",
    "QM_T200_8M",
    "qm_screen_profiles_f1_table.jld2");
JLD2_MyTools.summarize_meta_qm_jld2(data_qmf1_path)
qm_f1 =  jldopen(data_qmf1_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end;

data_qmf2_path = joinpath(@__DIR__,"simulation_data",
    "QM_T200_8M",
    "qm_screen_profiles_f2_table.jld2");
JLD2_MyTools.summarize_meta_qm_jld2(data_qmf2_path)
qm_f2 =  jldopen(data_qmf2_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz, σw_mm, λ0)]
end;

# for dir_selected in data_directories
    dir_selected = data_directories[8]

    data_exp_path = joinpath(@__DIR__,"EXPDATA_ANALYSIS","summary",dir_selected, dir_selected * "_report_summary.jld2");
    # JLD2_MyTools.show_exp_summary(data_exp_path,dir_selected)
    exp_data =  jldopen(data_exp_path,"r") do file
        Ic = file["meta/Currents"]
        data = file[JLD2_MyTools.make_keypath_exp(dir_selected,nz,λ0)];
        
        C00 = 0.5*(data[:fw_F1_peak_pos_raw ][1][1] + data[:fw_F2_peak_pos_raw ][1][1])

        zmax_1 = data[:fw_F1_peak_pos_raw ][1] .- C00
        δzmax_1 = data[:fw_F1_peak_pos_raw ][2]

        zmax_2 = data[:fw_F2_peak_pos_raw ][1] .- C00
        δzmax_2 = data[:fw_F2_peak_pos_raw ][2]

        z_binned = data[:F1_profile_spline][:,1] .- C00

        return (Ic = Ic, z=collect(data[:z_mm]) .- C00, F1=data[:F1_profile], F2=data[:F2_profile], z1 = hcat(zmax_1,δzmax_1), z2 = hcat(zmax_2,δzmax_2),  F1_profile = hcat(z_binned, data[:F1_profile_spline][:,2:end]), F2_profile = hcat(z_binned, data[:F2_profile_spline][:,2:end])) 
    end
    ni0_exp = closest_index(exp_data.Ic,selected_current)
    println(ni0_exp)

    pdf_F1 = TheoreticalSimulation.weighted_QM_PDF_profile_smooth(
        I0,
        z_theory,
        K39_params,
        effusion_params,
        1e-3*σw_mm ;
        Fsel=1,
        weights=nothing,
        normalize=false,
    );

    pdf_F2 = TheoreticalSimulation.weighted_QM_PDF_profile_smooth(
        I0,
        z_theory,
        K39_params,
        effusion_params,
        1e-3*σw_mm ;
        Fsel=2,
        weights=nothing,
        normalize=false,
    );


    fig = plot(
        title=L"$n_{z}=%$(nz)$ | $\sigma_{w}=%$(Int(1000*σw_mm))\mathrm{\mu m}$ | $\lambda_{s}=%$(λ0)$",
        xlims=(-1,1),
        xlabel=L"$z \, (\mathrm{mm})$",
        legend=:outerright,
    )
    scatter!([0],[0], marker=(:circle, 0.001,:white), fillalpha=0.01, label=L"Analytic: $I_{c}=%$(Int(1000*Icoils[ni0]))\mathrm{mA}$")
    plot!(fig,
        1e3*z_theory, TheoreticalSimulation.normalize_profile(z_theory, pdf_F1; method=:max),
        label=L"$F=1$ ",
        color=:red,
        fill=true,
        fillalpha=0.25,
    );
    plot!(fig,
        1e3*z_theory, TheoreticalSimulation.normalize_profile(z_theory, pdf_F2; method=:max),
        label=L"$F=2$",
        color=:blue,
        fill=true,
        fillalpha=0.25,
    );
    scatter!([0],[0], marker=(:circle, 0.001,:white), fillalpha=0.01, label=L"Montecarlo: $I_{c}=%$(Int(1000*Icoils[ni0]))\mathrm{mA}$")
    plot!(fig,
        qm_f1[ni0][:z_profile][:,1] , TheoreticalSimulation.normalize_profile(qm_f1[ni0][:z_profile][:,1], qm_f1[ni0][:z_profile][:,3]; method=:max),
        label=L"$F=1$",
        line=(:dash,2,:red)
    );
    plot!(fig,
        qm_f2[ni0][:z_profile][:,1] , TheoreticalSimulation.normalize_profile(qm_f2[ni0][:z_profile][:,1], qm_f2[ni0][:z_profile][:,3]; method=:max),
        label=L"$F=2$",
        line=(:dash,2,:blue,)
    );
    plot!(fig,
        legend_columns=1,
        legendfontsize = 8,
        foreground_color_legend = nothing,
        background_color_legend = nothing,
        size=(800,600),
    );
    scatter!([0],[0], marker=(:circle, 0.001,:white), fillalpha=0.01, label=L"Experiment %$(dir_selected) : 
    $I_{c}=%$(round(1000*exp_data.Ic[ni0_exp], digits=2))\mathrm{mA}$")
    plot!(fig,
        exp_data.z , TheoreticalSimulation.normalize_profile(exp_data.z , exp_data.F1[ni0_exp,:]; method=:max ),
        label=L"$F=1$ ",
        line=(:solid,2.5, :orangered2),
    )
    plot!(fig,
        exp_data.F1_profile[:,1], TheoreticalSimulation.normalize_profile(exp_data.F1_profile[:,1] , exp_data.F1_profile[:,ni0_exp+1]; method=:max ),
        label=L"$F=1$ ",
        line=(:dashdot,1.5, :orangered2),
    )
    vline!([exp_data.z1[ni0_exp,:][1]], 
        line=(:dash,1, :gray26), 
        label=L"$z_{1}=%$(round(exp_data.z1[ni0_exp,:][1];digits=4))\mathrm{mm}$",
    )
    vspan!([exp_data.z1[ni0_exp,:][1]-exp_data.z1[ni0_exp,:][2], exp_data.z1[ni0_exp,:][1]+exp_data.z1[ni0_exp,:][2]], 
        label=false,
        color=:gray26, 
        fillalpha=0.5,
    );
    plot!(fig,
        exp_data.z , TheoreticalSimulation.normalize_profile(exp_data.z , exp_data.F2[ni0_exp,:]; method=:max ),
        label=L"$F=2$",
        line=(:solid,2.5, :dodgerblue),
    );
    plot!(fig,
        exp_data.F2_profile[:,1], TheoreticalSimulation.normalize_profile(exp_data.F2_profile[:,1] , exp_data.F2_profile[:,ni0_exp+1]; method=:max ),
        label=L"$F=2$ ",
        line=(:dashdot,1.5, :dodgerblue),
    )
    vline!([exp_data.z2[ni0_exp,:][1]], 
        line=(:dash,1, :gray46), 
        label=L"$z_{2}=%$(round(exp_data.z2[ni0_exp,:][1];digits=4))\mathrm{mm}$",
    );
    vspan!([exp_data.z2[ni0_exp,:][1]-exp_data.z2[ni0_exp,:][2], exp_data.z2[ni0_exp,:][1]+exp_data.z2[ni0_exp,:][2]], 
        label=false,
        color=:gray46, 
        fillalpha=0.5,
    );
    display(fig)

end


for dir_selected in data_directories
    # dir_selected = data_directories[5]

    data_exp_path = joinpath(@__DIR__,"EXPDATA_ANALYSIS","summary",dir_selected, dir_selected * "_report_summary.jld2");
    JLD2_MyTools.show_exp_summary(data_exp_path,dir_selected)

    Ic_list, nz_list , λ0_list = jldopen(data_exp_path,"r") do file
        return (file["meta/Currents"], file["meta/nz"], file["meta/λ0"] )
    end
    nrows = length(nz_list)*length(λ0_list)

    vec3d = Matrix{Float64}(undef, nrows, 4)



    for Iidx in eachindex(Ic_list)
    # Iidx=25

        count = 0 
        for nz in nz_list
            for λ0 in λ0_list
                count += 1

                
                z_data =  jldopen(data_exp_path,"r") do file
                data = file[JLD2_MyTools.make_keypath_exp(dir_selected,nz,λ0)];
                
                C00 = 0.5*(data[:fw_F1_peak_pos_raw ][1][1] + data[:fw_F2_peak_pos_raw ][1][1])

                zmax_1 = data[:fw_F1_peak_pos_raw ][1][Iidx] .- C00
                δzmax_1 = data[:fw_F1_peak_pos_raw ][2][Iidx]

                zmax_2 = data[:fw_F2_peak_pos_raw ][1][Iidx] .- C00
                δzmax_2 = data[:fw_F2_peak_pos_raw ][2][Iidx]

                return (zF1 = hcat(zmax_1,δzmax_1), zF2 = hcat(zmax_2,δzmax_2)) 
                end

                vec3d[count,1] = nz
                vec3d[count,2] = λ0
                vec3d[count,3] = z_data.zF1[1]
                vec3d[count,4] = z_data.zF2[1]
            end
        end

        M = vec3d[:,1:3]
        nz_vals = collect(nz_list)
        λ0_vals = collect(λ0_list)

        Z = Matrix{Float64}(undef, length(nz_vals), length(λ0_vals))
        for row in eachrow(M)
            i = findfirst(==(row[1]), nz_vals)
            j = findfirst(==(row[2]), λ0_vals)
            Z[i,j] = row[3]
        end

        fig1= plot(
            xlabel = L"$\lambda_{s}$",
            ylabel = L"$F=1 : z_{\mathrm{max}} \ (\mathrm{mm})$",
        );
        for (i, nz) in enumerate(nz_vals)
            plot!(λ0_vals, Z[i, :], marker = :o, label = L"$n_{z} = %$(nz)$")
        end
        plot!(legend=:bottomright,
            xscale = :log10);

        fig2= heatmap(
            λ0_vals, nz_vals, Z;
            xscale = :log10,
            xlabel = L"$\lambda_{s}$",
            ylabel = L"$n_{z}$",
            cbar=:top,
            color=:magma,
            yticks=nz_vals,
            yminorticks=false,
        );

        tot= plot(fig1,fig2,
        suptitle=L"%$(dir_selected) : $I_{c}=%$(round(1000*Ic_list[Iidx];digits=2))\mathrm{mA}$",
        layout=(2,1),
        right_margin=6mm,
        )

        display(tot)

    end
end