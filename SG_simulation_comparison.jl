# Simulation of atom trajectories in the Stern–Gerlach experiment
# Kelvin Titimbo
# California Institute of Technology
# August 2025

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
using DifferentialEquations
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
const OUTDIR    = joinpath(@__DIR__, "simulation_data", RUN_STAMP);
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
const TEMP_DIR = joinpath(@__DIR__,"artifacts", "temp")
isdir(TEMP_DIR) || mkpath(TEMP_DIR);
ENV["TMPDIR"] = TEMP_DIR
@info "Created temporary directory" TEMP_DIR
# General setup
HOSTNAME = gethostname();
@info "Running on host" HOSTNAME=HOSTNAME
# Random seeds
base_seed_set = 145;
# rng_set = MersenneTwister(base_seed_set)
rng_set = TaskLocalRNG();
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
const ħ     = 6.62607015e-34/2π   # Reduced Planck constant (J s)
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

TWOπ*ħ*K39_params.Ahfs*(0.5+K39_params.Ispin) / (2*μₑ)


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

##################################################################################################
##################################################################################################

I_matlab = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010,
     0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100,
     0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.600, 0.700,
     0.800, 0.900, 1.00];
I_coils = [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010,
           0.015, 0.020, 0.025,	0.030, 0.035, 0.040, 0.045,	0.050, 0.055, 0.060, 0.065,	
           0.070, 0.075, 0.080,	0.085, 0.090, 0.095, 0.100,	0.150, 0.200, 0.250, 0.300,
           0.350, 0.400, 0.450,	0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850,
           0.900, 0.950, 1.000];
B0_matlab = [0.000000000000000, 0.000757240744579, 0.001511867743341, 0.002263756661471, 0.003012783164159, 0.003758822916592, 
             0.004501751583957,	0.005241444831444, 0.005977778324238, 0.006710627727529, 0.007439868706503, 0.011027595518207,
             0.014506461500176,	0.017860924800881, 0.021061203998997, 0.024104602602527, 0.027063556732162, 0.030012805215533, 
             0.033027086880274, 0.036137912163646, 0.039282554480295, 0.042463063647200, 0.045685090013413, 0.048956665511292,
             0.052297113702953,	0.055684689137129, 0.059088511309483, 0.062477084029270, 0.065826142622610, 0.101979590539433,
             0.139806136141056,	0.179560711908128, 0.222746532428795, 0.267955372898295, 0.315406692585403, 0.364520457067929,
             0.413871329177959,	0.460046065231226, 0.503368911318113, 0.541754419774573, 0.576165476654194, 0.611168191353271,
             0.643646574722572,	0.675004967851836, 0.702907241211002, 0.726826056270392, 0.748734155551346] ;      
df = CSV.read(joinpath(@__DIR__,"SG_BvsI.csv"), DataFrame; header=["dI","Bz"]);
G_matlab = [0, 0.245169023227448, 0.490961855002948, 0.737375053774835, 0.984405177991444, 1.232048786101112, 1.480302436552172,
            1.729162687792962, 1.978626098271814, 2.228689226437067, 2.479348630737054, 3.741469339949749, 5.017980719781120,
            6.308452576273089, 7.612454715467577, 8.929556943406510, 10.259329066131809, 11.601340889685392, 12.955162220109184,
            14.320362863445112, 15.696512625735096, 17.083181313021058, 18.479938731344916, 19.886354686748600, 21.301998985274025,
            22.726441432963117,	24.159251835857805, 25.600000000000001, 27.052826575207895, 42.327637639949899, 58.399999999999999,
            74.875021553278344, 92.184071488699999, 1.111355628242844e2, 1.303766442545645e2, 1.480779363146106e2, 1.648550253159278e2,
            1.808530748581974e2, 1.963000000000000e2, 2.113651411281627e2, 226,	240, 2.537000000000000e2, 2.657170746928492e2, 
            2.767759421141869e2, 2.871316062931601e2, 2.967826216759129e2];
GRAD_CURRENTS = [0, 0.095, 0.2, 0.302, 0.405, 0.498, 0.6, 0.7, 0.75, 0.8, 0.902, 1.01];
GRAD_GRADIENT = [0, 25.6, 58.4, 92.9, 132.2, 164.2, 196.3, 226, 240, 253.7, 277.2, 298.6];

fig1=plot(df.dI[2:end], df.Bz[2:end], 
    seriestype=:scatter, 
    marker=(:white, :circle,3),
    label="Manual");
plot!(fig1,I_coils[2:end], TheoreticalSimulation.BvsI.(I_coils)[2:end],
    label="Julia (Akima)",
    line=(:red,2));
plot!(fig1, I_coils[2:end], B0_matlab[2:end],
    label="Matlab (makima)",
    line=(:darkgreen,2));
plot!(fig1,
    xlabel="Current (A)",
    ylabel="Magnetic Field (T)",
    xaxis=:log10,
    yaxis=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"])
    );
display(fig1)

ΔB_inter = TheoreticalSimulation.BvsI.(I_coils) .- B0_matlab
fig2=plot(I_coils[2:end], 1e6*ΔB_inter[2:end], 
    line=(:red,1),
    marker=(:circle,:white,2),
    markerstrokecolor=:red,
    label="Residuals",
    xlabel="Currents (A)",
    ylabel="Residuals (μT)",
    xaxis=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:topleft)
display(fig2)

fig3=plot(GRAD_CURRENTS[2:end], GRAD_GRADIENT[2:end], 
    seriestype=:scatter, 
    marker=(:white, :circle,3),
    label="Manual")
plot!(fig3,I_coils[2:end], TheoreticalSimulation.GvsI.(I_coils)[2:end],
    label="Julia (Akima)",
    line=(:red,2))
plot!(fig3, I_coils[2:end], G_matlab[2:end],
    label="Matlab (makima)",
    line=(:darkgreen,2))
plot!(fig3,
    xlabel="Current (A)",
    ylabel="Gradient (T/m)",
    xaxis=:log10,
    yaxis=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    )
display(fig3)

ΔG_inter = TheoreticalSimulation.GvsI.(I_coils) .- G_matlab;
fig4=plot(I_coils[2:end], ΔG_inter[2:end], 
    line=(:red,1),
    marker=(:circle,:white,2),
    markerstrokecolor=:red,
    label="Residuals",
    xlabel="Currents (A)",
    ylabel="Residuals (T/m)",
    xaxis=:log10,
    xticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    legend=:topleft);
display(fig4);

plot(fig1,fig2,fig3,fig4,
    layout = (2,2),
    size=(800,600))

"""
Simulate 1D motion with three phases:
  Phase 1: 0 ≤ t < t1      → a(t) = 0
  Phase 2: t1 ≤ t ≤ t2     → a(t) = (μ*G/m) * cos(2atan(tan(θ/2)*exp(-kw*t)))
  Phase 3: t2 < t ≤ T      → a(t) = 0

All angles in **radians**. Returns DifferentialEquations.jl solution.
"""
function CQD_mynum(x0::Real, v0::Real, θ::Real,
                    tspan::Tuple{Real,Real},
                    v0y::Real,
                    Ix::Real,
                    kx::Real,
                    pk::AtomParams;
                    saveat=nothing, reltol=1e-18, abstol=1e-18)

    t1 = (y_FurnaceToSlit + y_SlitToSG) / v0y
    t2 = (y_FurnaceToSlit + y_SlitToSG + y_SG) / v0y

    @assert tspan[1] ≤ t1 ≤ t2 ≤ tspan[2] "Require tspan[1] ≤ t1 ≤ t2 ≤ tspan[2]"

    # Fast/stable form of your a(t): cos(2atan x) = (1 - x^2)/(1 + x^2)
    ωL          = abs(γₑ*TheoreticalSimulation.BvsI(Ix))
    a_scale     = (μₑ*TheoreticalSimulation.GvsI(Ix))/pk.M
    kω          = kx * ωL
    tan_halfθ   = tan(θ/2)
    
    a_mid(t) = begin
        τ = t - t1
        x = tan_halfθ * exp(-kω*τ )
        a_scale * ((1 - x^2) / (1 + x^2))
    end

    function a(t)
        (t < t1 || t > t2) ? 0.0 : a_mid(t)
    end

    function f!(du,u,p,t)
        x, v  = u
        du[1] = v
        du[2] = a(t)
    end

    u0   = (float(x0), float(v0))
    prob = ODEProblem(f!, collect(u0), (float(tspan[1]), float(tspan[2])))
    sol  = solve(prob, RadauIIA5();
                 tstops=(t1, t2),   # force steps at the phase boundaries
                 saveat=saveat,
                 reltol=reltol, abstol=abstol)
    return sol
end

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



[reverse([round(μF_effective(ix,F,mF,K39_params)/μB, sigdigits=5) for (F,mF) in quantum_numbers][6:8]) for ix in I_matlab]

ki_init = 1.90 ;
i_init  = 800 ;
data_sk_pos = CSV.read(joinpath(dirname(OUTDIR),"20251105T162102417","initialstates_zqm_zcqd_ki$(@sprintf("%.2f",ki_init))em6_I$(i_init)mA.CSV"),DataFrame; header=["x0","z0","v0x","v0y","v0z","θe","xD","zQM1","zQM2","zQM3","zCQD"]);
n_data = size(data_sk_pos,1);
I_sk    = i_init*1e-3 ;
ki_sk   = ki_init*1e-6 ;

sk_row = rand(1:n_data)

x0  = data_sk_pos[sk_row,"x0"];
y0  = 0.0;
z0  = data_sk_pos[sk_row,"z0"];
v0x = data_sk_pos[sk_row,"v0x"];
v0y = data_sk_pos[sk_row,"v0y"];
v0z = data_sk_pos[sk_row,"v0z"];
θe0 = data_sk_pos[sk_row,"θe"];
T   = (y_FurnaceToSlit+y_SlitToSG+y_SG+y_SGToScreen) / v0y ;

CQD_num_sol = CQD_mynum(z0, v0z, θe0, (0.0, 1.2*T), v0y, I_sk, ki_sk, K39_params; saveat=0:1e-9:1.2*T) ; 
QM_num_sol  = QM_mynum(z0, v0z, (0.0, 1.2*T), v0y, I_sk, K39_params; saveat=0:1e-9:1.2*T);

# Query state at arbitrary time:
z_at_screen = 1e3*CQD_num_sol(T)[1];

cqd_screen_kt = 1e3*TheoreticalSimulation.CQD_Screen_position(I_sk, μₑ, 
                                        [x0, y0, z0], [v0x,v0y,v0z],
                                        θe0, 3.2,
                                        ki_sk, K39_params);
cqd_screen_sk_pos = 1e3*[data_sk_pos[sk_row,"xD"], y_FurnaceToSlit+y_SlitToSG+y_SG+y_SGToScreen, data_sk_pos[sk_row,"zCQD"]];

pretty_table(transpose([cqd_screen_sk_pos[3],cqd_screen_kt[3], z_at_screen]),
        column_labels   = ["Fixed", "Analytical", "Numerical"],
        title           = "CQD : 𝓏 -position at the screen (mm)",  
        formatters      = [ fmt__printf("%5.6f", 1:4)],        
        alignment       = :c,
        table_format    = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style           = TextTableStyle(
                            first_line_merged_column_label  = crayon"light_red bold",
                            first_line_column_label         = crayon"yellow bold",
                            column_label                    = crayon"yellow",
                            table_border                    = crayon"blue bold",
                            title                           = crayon"red bold"
                        ),)

# qm_screen_sk_pre    = 1e3*[data_sk_pre[sk_row,"zQM3"], data_sk_pre[sk_row,"zQM2"], data_sk_pre[sk_row,"zQM1"]]
qm_screen_sk_pos    = 1e3*[data_sk_pos[sk_row,"zQM3"], data_sk_pos[sk_row,"zQM2"], data_sk_pos[sk_row,"zQM1"]]
qm_screen_num_kt    = 1e3*[sol(T)[1] for sol in QM_num_sol][6:8]
qm_screen_kt        = [1e3*TheoreticalSimulation.QM_Screen_position(I_sk,1,mf,[x0,y0,z0],[v0x,v0y,v0z], K39_params)[3] for mf=-1:1]

pretty_table(hcat([qm_screen_sk_pos,qm_screen_num_kt, qm_screen_kt ]...),
        column_labels   = ["Fixed", "Numerical", "Analytical"],
        title           = "QM : 𝓏 -position at the screen (mm)",  
        formatters      = [ fmt__printf("%5.6f", 1:4)],        
        alignment       = :c,
        table_format    = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style           = TextTableStyle(
                            first_line_merged_column_label  = crayon"light_red bold",
                            first_line_column_label         = crayon"yellow bold",
                            column_label                    = crayon"yellow",
                            table_border                    = crayon"blue bold",
                            title                           = crayon"red bold"
                        ),)


# COQUANTUM DYNAMICS 
cqd_comp = zeros(n_data,4);
for sk_row = 1:n_data

    x0  = data_sk_pos[sk_row,"x0"];
    y0  = 0.0;
    z0  = data_sk_pos[sk_row,"z0"];
    v0x = data_sk_pos[sk_row,"v0x"];
    v0y = data_sk_pos[sk_row,"v0y"];
    v0z = data_sk_pos[sk_row,"v0z"];
    θe0 = data_sk_pos[sk_row,"θe"];
    T  = (y_FurnaceToSlit+y_SlitToSG+y_SG+y_SGToScreen) / v0y ;

    sol = CQD_mynum(z0, v0z, θe0, (0.0, T), v0y, I_sk, ki_sk, K39_params; saveat=0:0.000001:T) ; 

    # Query state at arbitrary time:
    x_at_screen = 1e3*sol(T)[1];

    screen_kt = 1e3*TheoreticalSimulation.CQD_Screen_position(I_sk, μₑ, 
                                            [x0, y0, z0], [v0x,v0y,v0z],
                                            θe0, 3.2,
                                            ki_sk, K39_params);
    # screen_sk_pre = 1e3*[data_sk_pre[sk_row,"xD"], y_FurnaceToSlit+y_SlitToSG+y_SG+y_SGToScreen, data_sk_pre[sk_row,"zCQD"]];
    screen_sk_pos = 1e3*[data_sk_pos[sk_row,"xD"], y_FurnaceToSlit+y_SlitToSG+y_SG+y_SGToScreen, data_sk_pos[sk_row,"zCQD"]];

    cqd_comp[sk_row,:] = [screen_sk_pos[3],screen_kt[3], x_at_screen, abs(100*(screen_kt[3] .- screen_sk_pos[3])./screen_kt[3]) ]

    pretty_table(transpose(cqd_comp[sk_row,:]),
            column_labels   = ["Matlab", "Analytical", "Numerical", "Δ"],
            title           = "($sk_row/$(n_data)) CQD : 𝓏 -position at the screen (mm)",  
            formatters      = [ fmt__printf("%5.6f", 1:4)],        
            alignment       = :c,
            table_format    = TextTableFormat(borders = text_table_borders__unicode_rounded),
            style           = TextTableStyle(
                                first_line_merged_column_label  = crayon"light_red bold",
                                first_line_column_label         = crayon"yellow bold",
                                column_label                    = crayon"yellow",
                                table_border                    = crayon"blue bold",
                                title                           = crayon"red bold"
                            ),)
end

cqd_comp
minimum(cqd_comp[:,4])
maximum(cqd_comp[:,4])
argmax(cqd_comp[:,4])
cqd_comp[argmax(cqd_comp[:,4]),:]
data_sk_pos[argmax(cqd_comp[:,4]),:]

fig1=plot(abs.(cqd_comp[:,1]), 
    seriestype=:scatter,
    label="MATLAB num",
    marker=(:circle,:white,3),
    markerstrokecolor=:black,
    markerstrokewidth=1,)
plot!(abs.(cqd_comp[:,2]),
    seriestype=:scatter,
    label="analytical",
    marker=(:xcross,:red,3))
plot!(yaxis=:log10,
    xlabel="sample",
    ylabel=L"$z$ (mm)",
    # yticks = ([1e-3, 1e-2, 1e-1, 1.0, 10.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    legend=:bottomright,
    background_color_legend = nothing,)
fig2=plot(cqd_comp[:,4],
    label=L"CQD relative error $I_{c}=%$(i_init)\mathrm{mA}$",
    marker=(:circle,:white,2),
    markerstrokecolor=:red,
    line=(:red,1),
    xlabel="sample",
    ylabel="relative error (%)")
fig = plot(fig1, fig2,
    layout=(1,2),
    size=(800,400),
    bottom_margin = 3mm,
    left_margin = 3mm,
    plot_title= title=L"Screen $I_{c}=%$(i_init)\mathrm{mA}$",)

# QUANTUM MECHANICS          
qm_comp = zeros(Int(3*n_data),4) ;
for sk_row = 1:n_data
    x0  = data_sk_pos[sk_row,"x0"]
    y0  = 0.0
    z0  = data_sk_pos[sk_row,"z0"]
    v0x = data_sk_pos[sk_row,"v0x"]
    v0y = data_sk_pos[sk_row,"v0y"]
    v0z = data_sk_pos[sk_row,"v0z"]
    T  = (y_FurnaceToSlit+y_SlitToSG+y_SG+y_SGToScreen) / v0y ;

    QM_num_sol  = QM_mynum(z0, v0z, (0.0, 1.2*T), v0y, I_sk, K39_params; saveat=0:1e-9:1.2*T);
    qm_screen_num_kt    = 1e3*[sol(T)[1] for sol in QM_num_sol][6:8]

    row_sk = 1e3*[data_sk_pos[sk_row,"zQM3"], data_sk_pos[sk_row,"zQM2"], data_sk_pos[sk_row,"zQM1"]]
    row_kt = [1e3*TheoreticalSimulation.QM_Screen_position(I_sk,1,mf,[x0,y0,z0],[v0x,v0y,v0z], K39_params)[3] for mf=-1:1]
    rows = hcat(
                row_sk,
                qm_screen_num_kt,
                row_kt,
                100 .* abs.( (row_sk .- row_kt) ./ row_kt)
                )

    qm_comp[3*sk_row-2 : 3*sk_row,:] = rows
    pretty_table(rows,
        column_labels   = ["Matlab", "Numerical",  "Analytical", "Δ"],
        formatters      = [ fmt__printf("%5.6f", 1:4)],
        title           = "($(sk_row)/$(n_data)) QM : 𝓏 -position at the screen (mm)",          
        alignment       = :c,
        table_format    = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style           = TextTableStyle(
                            first_line_merged_column_label  = crayon"light_red bold",
                            first_line_column_label         = crayon"yellow bold",
                            column_label                    = crayon"yellow",
                            table_border                    = crayon"blue bold",
                            title                           = crayon"red bold"
                        )
    )
end

minimum(qm_comp[:,4])
maximum(qm_comp[:,4])
argmax(qm_comp[:,4])
fig1=plot(qm_comp[1:3:end,1], 
    seriestype=:scatter,
    label="MATLAB num",
    marker=(:circle,:white,3),
    markerstrokecolor=:black,
    markerstrokewidth=1,)
plot!(qm_comp[1:3:end,3],
    seriestype=:scatter,
    label="analytical",
    marker=(:xcross,:red,3))
plot!(
    ylabel=L"$z$ (mm)",
    # yticks = ([1e-3, 1e-2, 1e-1, 1.0, 10.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    legend=:bottomright,
    background_color_legend = nothing,)
fig2=plot(qm_comp[2:3:end,1], 
    seriestype=:scatter,
    label="MATLAB num",
    marker=(:circle,:white,3),
    markerstrokecolor=:black,
    markerstrokewidth=1,)
plot!(qm_comp[2:3:end,3],
    seriestype=:scatter,
    label="analytical",
    marker=(:xcross,:red,3))
plot!(ylabel=L"$z$ (mm)",
    # yticks = ([1e-3, 1e-2, 1e-1, 1.0, 10.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    legend=:bottomright,
    background_color_legend = nothing,)
fig3=plot(qm_comp[3:3:end,1], 
    seriestype=:scatter,
    label="MATLAB num",
    marker=(:circle,:white,3),
    markerstrokecolor=:black,
    markerstrokewidth=1,)
plot!(qm_comp[3:3:end,3],
    seriestype=:scatter,
    label="analytical",
    marker=(:xcross,:red,3))
plot!(xlabel="sample",
    ylabel=L"$z$ (mm)",
    # yticks = ([1e-3, 1e-2, 1e-1, 1.0, 10.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}", L"10^{1}"]),
    legend=:bottomright,
    background_color_legend = nothing,)
figa = plot(fig1,fig2,fig3,
    layout=(3,1),
    )
figb = plot(qm_comp[:,4],
    label="QM relative error",
    marker=(:circle,:white,2),
    markerstrokecolor=:red,
    line=(:red,1),
    xlabel="sample",
    ylabel="relative error (%)")
fig= plot(figa, figb,
    size=(800,400),
    left_margin =3mm,
    bottom_margin = 3mm,
    plot_title=L"Screen $I_{c}=%$(i_init)\mathrm{mA}$",
    layout=(1,2))

qm_comp[237,:]

ki      = 1.8e-6
v0y     = 165.12
v0z     = 0.12
yfinal  = y_FurnaceToSlit+y_SlitToSG+y_SG+y_SGToScreen
tspan   = collect(range(0,yfinal, length=10000 )) ./ v0y
Itest   = 800e-3



r_cqd = 1e6*TheoreticalSimulation.CQD_EqOfMotion_z.(tspan, Ref(Itest),Ref(μₑ),Ref([0,0,0]),Ref([0,v0y,v0z]), Ref(3π/4), Ref(3.14), Ref(ki), Ref(K39_params))

r_qm1 = 1e6*TheoreticalSimulation.QM_EqOfMotion_z.(tspan, Ref(Itest),Ref(1.0),Ref(1.0),Ref([0,0,0]),Ref([0,v0y,v0z]), Ref(K39_params))
r_qm2 = 1e6*TheoreticalSimulation.QM_EqOfMotion_z.(tspan, Ref(Itest),Ref(1.0),Ref(0.0),Ref([0,0,0]),Ref([0,v0y,v0z]), Ref(K39_params))
r_qm3 = 1e6*TheoreticalSimulation.QM_EqOfMotion_z.(tspan, Ref(Itest),Ref(1.0),Ref(-1.0),Ref([0,0,0]),Ref([0,v0y,v0z]), Ref(K39_params))

1e3*TheoreticalSimulation.QM_Screen_position(Itest,1,1,[0,0,0],[0,v0y,v0z], K39_params)
1e3*TheoreticalSimulation.QM_Screen_position(Itest,1,0,[0,0,0],[0,v0y,v0z], K39_params)
1e3*TheoreticalSimulation.QM_Screen_position(Itest,1,-1,[0,0,0],[0,v0y,v0z], K39_params)

1e3*TheoreticalSimulation.CQD_Screen_position(Itest,μₑ,[0,0,0],[0,v0y,v0z], 3π/4, 3.14, ki, K39_params)


TheoreticalSimulation.BvsI(Itest)
TheoreticalSimulation.GvsI(Itest)


[round(μF_effective(Itest,F,mF,K39_params)/μB, sigdigits=5) for (F,mF) in quantum_numbers]


TheoreticalSimulation.QM_EqOfMotion.(tspan,Ref(Itest),Ref(1),Ref(1),Ref([0,0,0]),Ref([0,v0y,v0z]), Ref(K39_params))
TheoreticalSimulation.QM_EqOfMotion.(tspan,Ref(Itest),Ref(1),Ref(0),Ref([0,0,0]),Ref([0,v0y,v0z]), Ref(K39_params))
TheoreticalSimulation.QM_EqOfMotion.(tspan,Ref(Itest),Ref(1),Ref(-1),Ref([0,0,0]),Ref([0,v0y,v0z]), Ref(K39_params))
TheoreticalSimulation.CQD_EqOfMotion.(tspan,Ref(Itest),Ref(μₑ),Ref([0,0,0]),Ref([0,v0y,v0z]), Ref(3π/4), Ref(3.14), Ref(ki), Ref(K39_params))
