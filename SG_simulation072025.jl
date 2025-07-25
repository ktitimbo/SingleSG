# Simulation of atom trajectories in the Stern–Gerlach experiment
# Kelvin Titimbo
# California Institute of Technology
# July 2025

##  Plotting Setup
using Plots; gr()
Plots.default(
    show=true, dpi=800, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
using Interpolations, Roots, Dierckx, Loess
using BSplineKit
using Optim, Interpolations, Roots
using LinearAlgebra, DataStructures
using WignerD, LambertW, PolyLog
using StatsBase
using Random, Statistics, NaNStatistics, MLBase, Distributions, StaticArrays
# using DifferentialEquations, OrdinaryDiffEq
# using ODEInterface, ODEInterfaceDiffEq
using Alert
using DelimitedFiles, CSV, DataFrames
using Colors, LaTeXStrings, Dates, Printf
using Base.Threads
LinearAlgebra.BLAS.set_num_threads(4)
BLAS.get_num_threads();
Threads.nthreads();
cd(dirname(@__FILE__)) # Set the working directory to the current location
## Custom modules
include("./Modules/atoms.jl");
include("./Modules/samplings.jl");
include("./Modules/MyPolylogarithms.jl");
# Define color palette
mypalette = palette(:tab10)
# Set the working directory to the current location
cd(dirname(@__FILE__)) 
## General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
## Timestamp start for execution timing
t_start = Dates.now()

rng = MersenneTwister(145) #TaskLocalRNG()

filename = "./simulation_data/$(Dates.format(t_start, "yyyymmddTHHMMSS"))" # filename
println("\n\t\tRunning process on:\t $(Dates.format(t_start, "yyyymmddTHHMMSS")) \n")

#################################################################################
# FUNCTIONS
#################################################################################
function clear_all()
    for name in names(Main, all=true)
        if name ∉ (:Base, :Core, :Main, Symbol("@__dot__"))
            if !isdefined(Main, name) || isconst(Main, name)
                continue  # Skip constants
            end
            @eval Main begin
                global $name = nothing
            end
        end
    end
    GC.gc()
    println("All user-defined variables (except constants) cleared.")
end

function polylog(s,z)
    # return MyPolylogarithms.polylog(s,z)
    return reli2(z)
end

function FreedmanDiaconisBins(data_list::Vector{Float64})
    # Calculate the interquartile range (IQR)
    Q1 = quantile(data_list, 0.25)
    Q3 = quantile(data_list, 0.75)
    IQR = Q3 - Q1

    # Calculate Freedman-Diaconis bin width
    n = length(data_list)
    bin_width = 2 * IQR / (n^(1/3))

    # Calculate the number of bins using the range of the data
    data_range = maximum(data_list) - minimum(data_list)
    bins = ceil(Int, data_range / bin_width)

    return bins
end

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
atom_info       = AtomicSpecies.atoms(atom);
const R         = atom_info[1];
const μₙ        = atom_info[2];
const γₙ        = atom_info[3];
const Ispin    = atom_info[4];
const Ahfs     = atom_info[6];
const M        = atom_info[7];
ki = 2.0e-6

# STERN--GERLACH EXPERIMENT
# Image size
exp_pixelsize = 0.0260 ;   # [mm] for 20243014
cam_pixelsize = 0.0065 ;  # [mm]
# Furnace
T = 273.15 + 200 ; # Furnace temperature (K)
# Furnace aperture
x_furnace = 2.0e-3 ;
z_furnace = 100e-6 ;
# Slit
x_slit  = 4.0e-3 ;
z_slit  = 300e-6 ;
# Propagation distances
y_FurnaceToSlit = 224.0e-3 ;
y_SlitToSG      = 44.0e-3 ;
y_SG            = 7.0e-2 ;
y_SGToScreen    = 32.0e-2 ;

# Magnetic field gradient interpolation
GradCurrents = [0, 0.095, 0.2, 0.302, 0.405, 0.498, 0.6, 0.7, 0.75, 0.8, 0.902, 1.01];
GradGradient = [0, 25.6, 58.4, 92.9, 132.2, 164.2, 196.3, 226, 240, 253.7, 277.2, 298.6];
GvsI = Interpolations.LinearInterpolation(GradCurrents, GradGradient, extrapolation_bc=Line());
IvsG = Interpolations.LinearInterpolation(GradGradient, GradCurrents, extrapolation_bc=Line());

# Magnetic Field
Bdata = CSV.read("./SG_BvsI.csv",DataFrame; header=["dI","Bz"]);
BvsI = linear_interpolation(Bdata.dI, Bdata.Bz, extrapolation_bc=Line());

icoils = collect(range(1e-6,1.05,10000));
fig1a = plot(GradCurrents,GradGradient, 
    seriestype=:scatter,
    marker=(:circle, :black,2),
    label=false,
    xlabel="Coil Current (A)",
    ylabel="Magnetic field gradient (T/m)",
    yticks=0:50:350,
);
plot!(icoils,GvsI(icoils), line=(:red,2), label=L"$\partial_{z}B_{z}$");
fig1b=plot(Bdata.dI, Bdata.Bz,
    seriestype=:scatter,
    marker=(:circle, :black,2), 
    label=false,
    xlabel="Coil Current (A)",
    ylabel="Magnetic field (T)",
    yticks=0:0.1:1.0
);
plot!(icoils,BvsI(icoils), line=(:orange,2), label=L"$B_{z}$");
fig1c = plot(GvsI(icoils),BvsI(icoils), 
    label=false,
    line=(:blue,2), 
    xlabel="Magnetic field gradient (T/m)",
    ylabel="Magnetic field (T)",
    ylims=(0,0.8),
    xticks=0:50:350,
    yticks=0:0.1:1.0,
);
fig1= plot(fig1a,fig1b,fig1c,
    layout = @layout([a1 ;a2 ;a3]),
    size=(400,700),
    plot_title="Magnetic field in the Stern--Gerlach apparatus",
    # plot_titlefontcolor=:black,
    plot_titlefontsize=10,
    guidefont=font(8,"Computer Modern"),
    # tickfont=font(8, "Computer Modern"),
    link=:none,
    # bottom_margin=-8mm, left_margin=-4mm, right_margin=-1mm
    left_margin=5mm,bottom_margin=0mm,right_margin=0mm,
)
display(fig1)
savefig(fig1,filename*"_01.svg")

# Quantum Magnetic Moment μF : electron(1/2)-nucleus(3/2)
function μF_effective(Ix,II,F,mF)
    ΔE = 2π*ħ*Ahfs*(II+1/2)
    normalized_B = (γₑ-γₙ)*ħ / ΔE * BvsI(Ix) 
    if F==II+1/2 
        if mF==F
            μF = gₑ/2 * ( 1 + 2*γₙ/γₑ*II)*μB
        elseif mF==-F
            μF = -gₑ/2 * ( 1 + 2*γₙ/γₑ*II)*μB
        else
            μF = gₑ*μB* ( mF*γₙ/γₑ + (1-γₙ/γₑ)/sqrt(1-4*mF/(2*II+1)*normalized_B+(normalized_B)^2) * ( mF/(2*II+1)-1/2*normalized_B ) )
        end
    elseif F==II-1/2
        μF = gₑ*μB* ( mF*γₙ/γₑ - (1-γₙ/γₑ)/sqrt(1-4*mF/(2*II+1)*normalized_B+(normalized_B)^2) * ( mF/(2*II+1)-1/2*normalized_B ) )
    end
    return μF
end

Irange = collect(0.00009:0.00002:1);
color8 = palette(:Set1_8);
fig2 = plot(Irange,μF_effective.(Irange,Ispin,2,2)/μB,
    label = L"$F=2$, $m_{F}=+2$",
    xlabel=L"Current ($\mathrm{A}$)",
    ylabel=L"$\mu_{F}/\mu_{B}$",
    line = (color8[1],2),
    xaxis=:log10,
    legend=:right,
    background_color_legend=RGBA(0.85, 0.85, 0.85, 0.1),
    size=(800,600),
);
plot!(Irange,μF_effective.(Irange,Ispin,2,1)/μB,
    label = L"$F=2$, $m_{F}=+1$",
    line = (color8[2],2),);
plot!(Irange,μF_effective.(Irange,Ispin,2,0)/μB,
    label = L"$F=2$, $m_{F}=0$",
    line = (color8[3],2),);
plot!(Irange,μF_effective.(Irange,Ispin,2,-1)/μB,
    label = L"$F=2$, $m_{F}=-1$",
    line = (color8[4],2),);
plot!(Irange,μF_effective.(Irange,Ispin,2,-2)/μB,
    label = L"$F=2$, $m_{F}=-2$",
    line = (color8[5],:dash,2),);
plot!(Irange,μF_effective.(Irange,Ispin,1,1)/μB,
    label = L"$F=1$, $m_{F}=+1$",
    line = (color8[6],:dash,2),);
plot!(Irange,μF_effective.(Irange,Ispin,1,0)/μB,
    label = L"$F=1$, $m_{F}=0$",
    line = (color8[7],:dash,2),);
plot!(Irange,μF_effective.(Irange,Ispin,1,-1)/μB,
    label = L"$F=1$, $m_{F}=-1$",
    line = (color8[8],:dash,2),
);
f(x) = BvsI(x) - 2π*ħ*Ahfs*(Ispin+1/2)/(2*ħ)/(γₙ-γₑ);
bcrossing = Roots.find_zero(f, (0.001, 0.02));
vline!([bcrossing], line=(:black,:dot,2), label=L"$I_{0} = %$(round(bcrossing,digits=5))\,\mathrm{A}$ 
$\partial_{z}B_{z} = %$(round(GvsI(bcrossing),digits=2))\,\mathrm{T/m}$
$B_{z} = %$(round(1e3*BvsI(bcrossing),digits=3))\,\mathrm{mT}$");
display(fig2)
savefig(fig2,filename*"_02.svg")

# Equations of motion
function CQDEqOfMotion(t,Ix,μ,r0::Vector{Float64},v0::Vector{Float64},θe::Float64, θn::Float64, ki)
    tf1 = y_FurnaceToSlit / v0[2]
    tf2 = (y_FurnaceToSlit + y_SlitToSG ) / v0[2]
    tf3 = (y_FurnaceToSlit + y_SlitToSG + y_SG ) / v0[2]
    tF = (y_FurnaceToSlit + y_SlitToSG + y_SG + y_SGToScreen ) / v0[2]

    cqd_sign = sign(θn-θe) 
    ωL       = abs(γₑ * BvsI(Ix) )
    acc_0    = μ*GvsI(Ix)/M
    kω       = cqd_sign*ki*ωL

    if 0.00 <= t && t <= tf1     # Furnace to Slit
        x = r0[1] + v0[1]*t 
        y = r0[2] + v0[2]*t 
        z = r0[3] + v0[3]*t
        vx , vy , vz = v0[1] , v0[2] , v0[3]
    elseif tf1 < t && t <= tf2    # Slit to SG apparatus
        x = r0[1] + v0[1]*t 
        y = r0[2] + v0[2]*t
        z = r0[3] + v0[3]*t
        vx , vy , vz = v0[1] , v0[2] , v0[3]
    elseif tf2 < t && t <= tf3   # Crossing the SG apparatus
        vx = v0[1]
        vy = v0[2]
        vz = v0[3] + acc_0*(t-tf2) + acc_0/kω * log( cos(θe/2)^2 + exp(-2*kω*(t-tf2))*sin(θe/2)^2 )
        x = r0[1] + v0[1]*t 
        y = r0[2] + v0[2]*t
        z = r0[3] + v0[3]*t + acc_0/2*(t-tf2)^2 + acc_0/kω*log(cos(θe/2)^2)*(t-tf2) + 1/2/(kω)^2 * acc_0 * ( polylog(2,-exp(-2*kω*(t-tf2))*tan(θe/2)^2) - polylog(2,-tan(θe/2)^2) )
    elseif t > tf3
        x = r0[1] + v0[1]*t
        y = r0[2] + v0[2]*t
        z = r0[3] + v0[3]*t + acc_0/2*( (t-tf2)^2 - (t-tf3)^2) + acc_0/kω*y_SG/v0[2] * ( log(cos(θe/2)^2) + v0[2]/y_SG*log(cos(θe/2)^2+exp(-2*kω*y_SG/v0[2])*sin(θe/2)^2)*(t-tf3) ) + acc_0/2/kω^2*( polylog(2,-exp(-2*kω*y_SG/v0[2])*tan(θe/2)^2) - polylog(2,-tan(θe/2)^2) )
        vx = v0[1]
        vy = v0[2]
        vz = v0[3] + acc_0*y_SG/v0[2] + acc_0/kω*log(cos(θe/2)^2 + exp(-2*kω*y_SG/v0[2])*sin(θe/2)^2)
    end

    return [x,y,z]
end

function CQD_Screen_position(Ix,μ,r0::Vector{Float64},v0::Vector{Float64},θe::Float64, θn::Float64, ki)
    L1 = y_FurnaceToSlit 
    L2 = y_SlitToSG
    Lsg = y_SG
    Ld = y_SGToScreen

    cqd_sign = sign(θn-θe) 
    acc_0 = μ * GvsI(Ix) / M
    ωL = abs(γₑ * BvsI(Ix))
    kω = cqd_sign * ki * ωL

    x = r0[1] + (L1 + L2 + Lsg + Ld) * v0[1] / v0[2]
    y = r0[2] +  L1 + L2 + Lsg + Ld
    z = r0[3] + (L1 + L2 + Lsg + Ld) * v0[3] / v0[2] + acc_0/2/v0[2]^2*((Lsg+Ld)^2-Ld^2) + acc_0/kω*Lsg/v0[2]*( log(cos(θe/2)^2) + Ld/Lsg * log( cos(θe/2)^2 + exp(-2*kω*Lsg/v0[2])*sin(θe/2)^2 ) ) + acc_0/2/kω^2 * ( polylog(2, -exp(-2*kω*Lsg/v0[2])*tan(θe/2)^2) - polylog(2, -tan(θe/2)^2)  )
    return [x,y,z]
end

# function QMEqOfMotion(t,Ix,μ,r0::Vector{Float64},v0::Vector{Float64})
#     tf1 = y_FurnaceToSlit / v0[2]
#     tf2 = (y_FurnaceToSlit + y_SlitToSG ) / v0[2]
#     tf3 = (y_FurnaceToSlit + y_SlitToSG + y_SG ) / v0[2]
#     tF = (y_FurnaceToSlit + y_SlitToSG + y_SG + y_SGToScreen ) / v0[2]

#     acc_0   = μ*GvsI(Ix)/M

#     if 0.00 <= t && t <= tf1     # Furnace to Slit
#         x = r0[1] + v0[1]*t 
#         y = r0[2] + v0[2]*t 
#         z = r0[3] + v0[3]*t
#         vx , vy , vz = v0[1] , v0[2] , v0[3]
#     elseif tf1 < t && t <= tf2    # Slit to SG apparatus
#         x = r0[1] + v0[1]*t 
#         y = r0[2] + v0[2]*t
#         z = r0[3] + v0[3]*t
#         vx , vy , vz = v0[1] , v0[2] , v0[3]
#     elseif tf2 < t && t <= tf3   # Crossing the SG apparatus
#         vx = v0[1]
#         vy = v0[2]
#         vz = v0[3] + acc_0*(t-tf2)
#         x = r0[1] + v0[1]*t 
#         y = r0[2] + v0[2]*t
#         z = r0[3] + v0[3]*t + acc_0/2*(t-tf2)^2
#     elseif t > tf3
#         x = r0[1] + v0[1]*t
#         y = r0[2] + v0[2]*t
#         z = r0[3] + v0[3]*t + acc_0/2/v0[2]^2*( (v0[2]*t-y_FurnaceToSlit-y_SlitToSG)^2 - (v0[2]*t-y_FurnaceToSlit-y_SlitToSG-y_SG)^2 ) 
#         vx = v0[1]
#         vy = v0[2]
#         vz = v0[3] + acc_0*y_SG/v0[2]
#     end

#     return [x,y,z]
# end

# function QM_Screen_position(Ix,μ,r0::Vector{Float64},v0::Vector{Float64})
#     L1 = y_FurnaceToSlit 
#     L2 = y_SlitToSG
#     Lsg = y_SG
#     Ld = y_SGToScreen

#     acc_0 = μ * GvsI(Ix) / M

#     x = r0[1] + (L1 + L2 + Lsg + Ld) * v0[1] / v0[2]
#     y = r0[2] +  L1 + L2 + Lsg + Ld
#     z = r0[3] + (L1 + L2 + Lsg + Ld) * v0[3] / v0[2] + acc_0 /2 / v0[2]^2 * ((Lsg + Ld)^2 - Ld^2) 
 
#     return 1e3*[x,y,z]
# end

# trj_sk = CSV.read("./trajectory_ode.csv",DataFrame; header=["dy","dz"])
# ic_sk = CSV.read("./z0_vz0_vy0_theta_e0_spin__ki_current.csv",DataFrame;header=["z0","vz0","vy0","θe","spin","ki","I0"])

# plot(trj_sk[!,"dy"],trj_sk[!,"dz"])
# vline!([y_FurnaceToSlit], label="Slit", line=(:dash,:black))
# vline!([y_FurnaceToSlit+y_SlitToSG], label="SG", line=(:dash,:black))
# vline!([y_FurnaceToSlit+y_SlitToSG+y_SG], label="SG", line=(:dash,:black))
# vline!([y_FurnaceToSlit+y_SlitToSG+y_SG+y_SGToScreen], label="Screen", line=(:dash,:black))
# time_dt = collect(range(0,(y_FurnaceToSlit+y_SlitToSG+y_SG+y_SGToScreen)/ic_sk.vy0[1], length=1000))
# rt = zeros(Float64,length(time_dt),3)
# rtqm = zeros(Float64,length(time_dt),3)
# for (i,dt) in enumerate(time_dt)
#     println(i)
#     rt[i,:] = CQDEqOfMotion(dt,ic_sk.I0[1],μB,[0.0,0.0,ic_sk.z0[1]],[0.0,ic_sk.vy0[1],ic_sk.vz0[1]],ic_sk.θe[1],π/3,ic_sk.ki[1] )
#     rtqm[i,:] = QMEqOfMotion(dt,ic_sk.I0[1],μB,[0.0,0.0,ic_sk.z0[1]],[0.0,ic_sk.vy0[1],ic_sk.vz0[1]] )
# end
# plot!(rt[:,2],rt[:,3],line=(:red,4))
# plot!(rtqm[:,2],rtqm[:,3])

# QM_Screen_position(ic_sk.I0[1],μB,[0.0,0.0,ic_sk.z0[1]],[0.0,ic_sk.vy0[1],ic_sk.vz0[1]])
# 1e3*rtqm[end,:]

# CQD_Screen_position(ic_sk.I0[1],μB,[0.0,0.0,ic_sk.z0[1]],[0.0,ic_sk.vy0[1],ic_sk.vz0[1]],ic_sk.θe[1],π/3,ic_sk.ki[1])
# 1e3*rt[end,:]

# Atomic beam velocity probability Distribution
p_furnace   = [-x_furnace/2,-z_furnace/2];
p_slit      = [x_slit/2, z_slit/2];
θv_max      = 1.25*atan(norm(p_furnace-p_slit) , y_FurnaceToSlit);
function AtomicBeamVelocity()
    ϕ = 2π*rand(rng)
    θ = asin(sin(θv_max)*sqrt(rand(rng)))
    v = sqrt(-2*kb*T/M*(1 + lambertw((rand(rng)-1)/exp(1),-1)))
    return [ v*sin(θ)*sin(ϕ) , v*cos(θ) , v*sin(θ)*cos(ϕ) ]
end

# Sample size: number of atoms arriving to the screen
Nss=2000000

# Initialize empty array to store valid rows
# Pre-allocate storage with size hint for performance
alive_slit = Vector{SVector{6, Float64}}()  # [x0,y0,z0,v0x,v0y,v0z]
sizehint!(alive_slit, Nss)
 
iteration_count = 0;
# Generate samples until we get 10 valid ones
@time while length(alive_slit) < Nss

    iteration_count += 1

    # Generate initial positions
    x_initial = x_furnace * (rand(rng) - 0.5)
    z_initial = z_furnace * (rand(rng) - 0.5)

    # Generate initial velocities
    v0_x , v0_y , v0_z = AtomicBeamVelocity()

    # Calculate positions at the slit
    x_at_slit = x_initial + y_FurnaceToSlit * v0_x / v0_y
    y_at_slit = y_FurnaceToSlit
    z_at_slit = z_initial + y_FurnaceToSlit * v0_z / v0_y

    # Check conditions
    if -x_slit/2 <= x_at_slit <= x_slit/2 && -z_slit/2 <= z_at_slit <= z_slit/2
        # Add valid data to the array
        push!(alive_slit, SVector(x_initial, 0.0, z_initial, v0_x, v0_y, v0_z))
        # println("Added valid sample #", length(alive_slit))
    end
end
# println("Valid samples:\n", alive_slit)
println("Total iterations: ", iteration_count)

# Precompute velocity magnitudes and angles
velocities = [sqrt(v[4]^2 + v[5]^2 + v[6]^2) for v in alive_slit];
theta_vals = [acos(v[6] / sqrt(v[4]^2 + v[5]^2 + v[6]^2)) for v in alive_slit];
phi_vals   = [atan(v[5], v[4]) for v in alive_slit];
# Compute means once
mean_v , rms_v = mean(velocities) , sqrt(mean([v^2 for v in velocities]));
mean_theta , mean_phi = mean(theta_vals) , mean(phi_vals);

# Histogram for velocities
fig3a = histogram(
    velocities,
    bins = FreedmanDiaconisBins(velocities),
    label = L"$v_0$",
    normalize = :pdf,
    xlabel=L"v_{0} \ (\mathrm{m/s})",
    alpha = 0.70
);
vline!([mean_v], 
    label = L"$\langle v_{0} \rangle = %$(round(mean_v, digits=1)) \mathrm{m/s}$",
    line = (:black, :solid, 2),
);
vline!([rms_v], 
    label = L"$\sqrt{\langle v_{0}^2 \rangle} = %$(round(rms_v, digits=1)) \mathrm{m/s}$",
    line = (:red, :dash, 3)
);
# Histogram for theta (polar angle)
fig3b = histogram(
    theta_vals,
    bins = FreedmanDiaconisBins(theta_vals),
    label = L"$\theta_v$",
    normalize = :pdf,
    alpha = 0.70,
    xlabel=L"$\theta_{v}$",
);
vline!([mean_theta], 
    label = L"$\langle \theta_{v} \rangle = %$(round(mean_theta/π, digits=3))\pi$",
    line = (:black, :solid, 2)
);
# Histogram for phi (azimuthal angle)
fig3c = histogram(
    phi_vals,
    bins = FreedmanDiaconisBins(phi_vals),
    label = L"$\phi_v$",
    normalize = :pdf,
    alpha = 0.70,
    xlabel=L"$\phi_{v}$",
);
vline!([mean_phi], 
    label = L"$\langle \phi_{v} \rangle = %$(round(mean_phi/π, digits=3))\pi$",
    line = (:black, :solid, 2)
);
fig3d=histogram2d([1e3*x[1] for x in alive_slit], [1e6*z[3] for z in alive_slit], 
    bins=(FreedmanDiaconisBins([1e3*x[1] for x in alive_slit]), FreedmanDiaconisBins([1e6*z[3] for z in alive_slit])), 
    show_empty_bins=true,
    # normalize=:pdf,  
    color=:plasma,
    # title="Histogram: Initial position",
    xlabel=L"$x \ (\mathrm{mm})$",
    ylabel=L"$z \ (\mathrm{\mu m})$",
    xticks=-1.0:0.25:1.0,
    yticks=-50:10:50,
    colorbar_position=:bottom,  # Set colorbar at the bottom
) ;
fig3e=histogram( [v[4] for v in alive_slit],
    bins=FreedmanDiaconisBins([v[4] for v in alive_slit]),
    normalize=:pdf,
    label=L"$v_{0,x}$",
    alpha=0.65,
    color=(:orange),
    xlabel=L"$v_{0,x} \ (\mathrm{m/s})$",
);
fig3f=histogram( [v[5] for v in alive_slit],
    bins=FreedmanDiaconisBins([v[5] for v in alive_slit]),
    normalize=:pdf,
    label=L"$v_{0,y}$",
    alpha=0.65,
    color=(:blue),
    xlabel=L"$v_{0,y} \ (\mathrm{m/s})$",
);
fig3g=histogram( [v[6] for v in alive_slit],
    bins=FreedmanDiaconisBins([v[6] for v in alive_slit]),
    normalize=:pdf,
    label=L"$v_{0,z}$",
    alpha=0.65,
    color=(:red),
    xlabel=L"$v_{0,z} \ (\mathrm{m/s})$",
);
# Combine plots
fig3 = plot(
    fig3a, fig3e, fig3b, fig3f, fig3c, fig3g,  fig3d ,
    layout = @layout([a1 a2; a3 a4; a5 a6 ; a7]),
    size=(650,800),
    legendfontsize=8,
    left_margin=3mm,
)
display(fig3)
savefig(fig3,filename*"_03.svg")

# # CQD Sampling: electron and nuclear magnetic moments
# θes = 2 * asin.(sqrt.(rand(rng,Nss)))
# θns = 2 * asin.(sqrt.(rand(rng,Nss)))
# mask = θes .< θns
# # Split the data in those going UP and DOWN according to CQD
# MMeUP = hcat(θes[mask], θns[mask])
# MMeDOWN = hcat(θes[.!mask], θns[.!mask])
# Appended [x0,y0,z0,vx0,vy0,vz0,θₑ,θₙ] for UP and DOWN independently
# pairs_UP = [vcat( alive_slit[i], MMeUP[i,:]) for i in 1:size(MMeUP,1)]
# pairs_DOWN = [vcat(alive_slit[size(MMeUP,1)+i], MMeDOWN[i,:]) for i in 1:size(MMeDOWN,1)]

function generate_matched_pairs(No)
    θes_up_list = Float64[]
    θns_up_list = Float64[]
    θes_down_list = Float64[]
    θns_down_list = Float64[]
    
    count_less = 0
    count_greater = 0
    
    while count_less < No || count_greater < No
        θe = 2 * asin(sqrt(rand(rng)))
        θn = 2 * asin(sqrt(rand(rng)))

        if θe < θn && count_less < No
            push!(θes_up_list, θe)
            push!(θns_up_list, θn)
            count_less += 1
        elseif θe > θn && count_greater < No
            push!(θes_down_list, θe)
            push!(θns_down_list, θn)
            count_greater += 1
        end
    end
    
    return θes_up_list, θns_up_list, θes_down_list, θns_down_list
end
θesUP , θnsUP , θesDOWN , θnsDOWN= generate_matched_pairs(Nss);
pairs_UP    = [vcat( alive_slit[i], θesUP[i], θnsUP[i]) for i in 1:Nss];
pairs_DOWN  = [vcat(alive_slit[i], θesDOWN[i], θnsDOWN[i]) for i in 1:Nss];
θesUP , θnsUP = nothing , nothing
θesDOWN , θnsDOWN = nothing , nothing

pairs_UP    = reshape(reinterpret(Float64, pairs_UP), 8, :)'
pairs_DOWN  = reshape(reinterpret(Float64, pairs_DOWN), 8, :)'


# Function to plot histogram using Freedman-Diaconis binning rule
function FD_histograms(data_list::Vector{Float64},Label::LaTeXString,color)
    # Calculate the interquartile range (IQR)
    Q1 = quantile(data_list, 0.25)
    Q3 = quantile(data_list, 0.75)
    IQR = Q3 - Q1

    # Calculate Freedman-Diaconis bin width
    n = length(data_list)
    bin_width = 2 * IQR / (n^(1/3))

    # Calculate the number of bins using the range of the data
    data_range = maximum(data_list) - minimum(data_list)
    bins = ceil(Int, data_range / bin_width)

    # Plot the histogram
    histogram(data_list, bins=bins, normalize=:pdf,
            label=Label,
            # xlabel="Polar angle", 
            color=color,
            alpha=0.8,
            xlim=(0,π),
            xticks=PlottingTools.pitick(0, π, 8; mode=:latex),)
end
fig4a = FD_histograms(pairs_UP[:,7],L"\theta_{e}",:dodgerblue);
fig4b = FD_histograms(pairs_UP[:,8],L"\theta_{n}",:red);
fig4c = FD_histograms(pairs_DOWN[:,7],L"\theta_{e}",:dodgerblue);
fig4d = FD_histograms(pairs_DOWN[:,8],L"\theta_{n}",:red);
fig4= plot(fig4a,fig4b,fig4c,fig4d,
    layout = @layout([a1 a2 ; a3 a4]),
    size=(600,600),
    plot_title="Initial polar angles",
    # plot_titlefontcolor=:black,
    plot_titlefontsize=10,
    guidefont=font(8,"Computer Modern"),
    # tickfont=font(8, "Computer Modern"),
    link=:xy,
    # bottom_margin=-8mm, left_margin=-4mm, right_margin=-1mm
    left_margin=5mm,bottom_margin=0mm,right_margin=0mm,
);
plot!(fig4[1],xticks=(xticks(fig4[1])[1], []),xlabel="",bottom_margin=-5mm);
plot!(fig4[2],xticks=(xticks(fig4[2])[1], []), yticks=(yticks(fig4[2])[1], []), xlabel="",bottom_margin=-5mm, left_margin=-5mm);
plot!(fig4[4],yticks=(yticks(fig4[4])[1], fill("", length(yticks(fig4[4])[1]))), ylabel="",left_margin=-5mm);
display(fig4)
savefig(fig4,filename*"_04.svg")



Icoils = [0.001,0.002,0.003,0.005,0.007,
            0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080,
            0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.75,0.80]
screen_coord = zeros(Nss,3, length(Icoils));

for j=1:length(Icoils)
    @time @threads for i=1:Nss
        screen_coord[i,:,j] = CQD_Screen_position(Icoils[j],μₑ,pairs_UP[i,1:3],pairs_UP[i,4:6],pairs_UP[i,7], pairs_UP[i,8],ki)
    end
end


s_bin = 2
data = 1e3*hcat(screen_coord[:,1,5],screen_coord[:,3,5])
data = permutedims(reduce(hcat, filter(row -> row[2] <= 5, eachrow(data)) |> collect ))

# Create 2D histogram
sim_xmin , sim_xmax = minimum(data[:,1])  , maximum(data[:,1])
sim_zmin , sim_zmax = minimum(data[:,2]) , maximum(data[:,2])
nbins_x = length(collect(sim_xmin:s_bin*cam_pixelsize:sim_xmax))+1
nbins_z = length(collect(sim_zmin:s_bin*cam_pixelsize:sim_zmax))+1
h0 = fit(Histogram,(data[:,1],data[:,2]),
    (range(sim_xmin,sim_xmax,length=nbins_x),range(sim_zmin,sim_zmax,length=nbins_z))
)
h0=normalize(h0,mode=:pdf)
bin_edges_x = collect(h0.edges[1])
bin_edges_z = collect(h0.edges[2])
bin_centers_x = (bin_edges_x[1:end-1] .+ bin_edges_x[2:end]) ./ 2  # Compute bin centers
bin_centers_z = (bin_edges_z[1:end-1] .+ bin_edges_z[2:end]) ./ 2  # Compute bin centers

z_profile = hcat(bin_centers_z, vec(mean(h0.weights,dims=1)))
# Find the index of the maximum value in the second column & Extract the corresponding value from the first column
zmax_idx = argmax(z_profile[:, 2])
z_max_0 = z_profile[zmax_idx, 1]

fig_2dhist = histogram2d(data[:,1],data[:,2],
    nbins=(nbins_x,nbins_z),
    normalize=:pdf,
    color=:inferno,
    title=L"Co Quantum Dynamics: $\vec{\mu}_{e} \upuparrows \hat{z}$",
    xlabel=L"$x \ (\mathrm{mm})$",
    ylabel=L"$z \ (\mathrm{mm})$",
    xlim=(sim_xmin, sim_xmax),
    ylim=(sim_zmin,8),
    show_empty_bins=true,
)
hline!([z_max_0],label=false,line=(:red,:dash,1))

fig_prof = plot(z_profile[:,1],z_profile[:,2],
    label="Simulation",
    seriestype=:line,
    line=(:gray,1),
    # marker=(:black,:circle,2),
    title="CoQuantum Dynamics",
    zlims=(0,:auto),
    xlabel=L"$z \, (\mathrm{mm})$",
    legend=:topright,
    legendtitle=L"$I_{0}=%$(21)\,\mathrm{A}$",
    legendtitlefontsize=10,
)
vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$",line=(:red,:dash,2))
zscan = collect(minimum(z_profile[:,1]):0.001:maximum(z_profile[:,1]))


model = loess(z_profile[:,1],z_profile[:,2], span=0.10)
plot!(zscan,predict(model,zscan),
    label="Loess",
    line=(:purple4,2,0.5),
)

# Define the smoothed function
smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
# Find minimum using optimization
opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
z_max_fit = Optim.minimizer(opt_result)[1]
vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red,:dot,2))

23

function CQD_analysis(Ix,cqd_data::AbstractMatrix; z_upper = 10 , s_bin = 1 , loess_factor = 0.10 )

    data = cqd_data[:,[9,11]]
    data = permutedims(reduce(hcat, filter(row -> row[2] <= z_upper, eachrow(data)) |> collect ))

    # Create 2D histogram
    sim_xmin , sim_xmax = minimum(data[:,1])  , maximum(data[:,1])
    sim_zmin , sim_zmax = minimum(data[:,2]) , maximum(data[:,2])
    nbins_x = length(collect(sim_xmin:s_bin*cam_pixelsize:sim_xmax))+1
    nbins_z = length(collect(sim_zmin:s_bin*cam_pixelsize:sim_zmax))+1
    h0 = fit(Histogram,(data[:,1],data[:,2]),
        (range(sim_xmin,sim_xmax,length=nbins_x),range(sim_zmin,sim_zmax,length=nbins_z))
    )
    h0=normalize(h0,mode=:pdf)
    bin_edges_x = collect(h0.edges[1])
    bin_edges_z = collect(h0.edges[2])
    bin_centers_x = (bin_edges_x[1:end-1] .+ bin_edges_x[2:end]) ./ 2  # Compute bin centers
    bin_centers_z = (bin_edges_z[1:end-1] .+ bin_edges_z[2:end]) ./ 2  # Compute bin centers

    z_profile = hcat(bin_centers_z, vec(mean(h0.weights,dims=1)))
    # Find the index of the maximum value in the second column & Extract the corresponding value from the first column
    zmax_idx = argmax(z_profile[:, 2])
    z_max_0 = z_profile[zmax_idx, 1]


    fig_2dhist = histogram2d(data[:,1],data[:,2],
        nbins=(nbins_x,nbins_z),
        normalize=:pdf,
        color=:inferno,
        title=L"Co Quantum Dynamics: $\vec{\mu}_{e} \upuparrows \hat{z}$",
        xlabel=L"$x \ (\mathrm{mm})$",
        ylabel=L"$z \ (\mathrm{mm})$",
        xlim=(sim_xmin, sim_xmax),
        ylim=(sim_zmin,sim_zmax),
        show_empty_bins=true,
    )
    hline!([z_max_0],label=false,line=(:red,:dash,1))

    fig_prof = plot(z_profile[:,1],z_profile[:,2],
        label="Simulation",
        seriestype=:line,
        line=(:gray,1),
        # marker=(:black,:circle,2),
        title="CoQuantum Dynamics",
        zlims=(0,:auto),
        xlabel=L"$z \, (\mathrm{mm})$",
        legend=:topright,
        legendtitle=L"$I_{0}=%$(Ix)\,\mathrm{A}$",
        legendtitlefontsize=10,
    )
    vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$",line=(:red,:dash,2))
    zscan = collect(minimum(z_profile[:,1]):0.001:maximum(z_profile[:,1]))
    ## Dierckx
    # fspline = Spline1D(z_profile[:,1],z_profile[:,2],k=3,s=0.01)
    ## Define and optimize the negative spline function
    # neg_spline(x) = -fspline(x[1])
    # opt_result = optimize(neg_spline, [minimum(bin_centers)], [maximum(bin_centers)], [bin_center_max], Fminbox(LBFGS()))
    # plot!(zscan,fspline(zscan))

    # Loess
    model = loess(z_profile[:,1],z_profile[:,2], span=loess_factor)
    plot!(zscan,predict(model,zscan),
        label="Loess",
        line=(:purple4,2,0.5),
    )

    # Define the smoothed function
    smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
    # Find minimum using optimization
    opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
    z_max_fit = Optim.minimizer(opt_result)[1]
    vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red,:dot,2))


    return fig_2dhist , fig_prof , z_max_0 , z_max_fit , z_profile
end

# function QM_analysis(Ix,dataqm::AbstractMatrix,ms,mf::AbstractVector; z_upper = 10, s_bin = 1 , loess_factor = 0.10 )
#     if ms==1/2
#         idx = [[2 1 0 -1],[9,10,11,12]]
#         # Find the indices where elements in idx[1] are in mf
#         valid_indices = findall(x -> x in mf, idx[1])
#         # Convert CartesianIndex to plain indices
#         flat_indices = [i[2] for i in valid_indices]
#         # Retrieve the corresponding elements from idx[2]
#         valid_columns = idx[2][flat_indices]
#     else
#         idx = [[-2 1 0 -1],[13,14,15,16]]
#         # Find the indices where elements in idx[1] are in mf
#         valid_indices = findall(x -> x in mf, idx[1])
#         # Convert CartesianIndex to plain indices
#         flat_indices = [i[2] for i in valid_indices]
#         # Retrieve the corresponding elements from idx[2]
#         valid_columns = idx[2][flat_indices]
#     end
#     data = dataqm[:, [7,valid_columns[1]]]  # Start with the 7th column
#     for i in valid_columns[2:end]
#         data = vcat(data, dataqm[:, [7, i]])  # Concatenate columns
#     end

#     data = permutedims(reduce(hcat, filter(row -> row[2] <= z_upper, eachrow(data)) |> collect ))

#     # Create 2D histogram
#     sim_xmin , sim_xmax = minimum(data[:,1])  , maximum(data[:,1])
#     sim_zmin , sim_zmax = minimum(data[:,2]) , maximum(data[:,2])
#     nbins_x = length(collect(sim_xmin:s_bin*cam_pixelsize:sim_xmax))+1
#     nbins_z = length(collect(sim_zmin:s_bin*cam_pixelsize:sim_zmax))+1
#     h0 = fit(Histogram,(data[:,1],data[:,2]),
#         (range(sim_xmin,sim_xmax,length=nbins_x),range(sim_zmin,sim_zmax,length=nbins_z))
#     )
#     h0=normalize(h0,mode=:pdf)
#     bin_edges_x = collect(h0.edges[1])
#     bin_edges_z = collect(h0.edges[2])
#     bin_centers_x = (bin_edges_x[1:end-1] .+ bin_edges_x[2:end]) ./ 2  # Compute bin centers
#     bin_centers_z = (bin_edges_z[1:end-1] .+ bin_edges_z[2:end]) ./ 2  # Compute bin centers

#     z_profile = hcat(bin_centers_z, vec(mean(h0.weights,dims=1)))
#     # Find the index of the maximum value in the second column & Extract the corresponding value from the first column
#     zmax_idx = argmax(z_profile[:, 2])
#     z_max_0 = z_profile[zmax_idx, 1]


#     fig_2dhist = histogram2d(data[:,1],data[:,2],
#         nbins=(nbins_x,nbins_z),
#         normalize=:pdf,
#         color=:inferno,
#         title=L"Quantum Mechanics: $m_{s} \updownarrows \hat{z}$",
#         xlabel=L"$x \ (\mathrm{mm})$",
#         ylabel=L"$z \ (\mathrm{mm})$",
#         xlim=(sim_xmin, sim_xmax),
#         ylim=(sim_zmin,sim_zmax),
#         show_empty_bins=true,
#     )
#     hline!([z_max_0],label=false,line=(:red,:dash,1))

#     fig_prof = plot(z_profile[:,1],z_profile[:,2],
#         label="Simulation",
#         title="Quantum mechanics",
#         seriestype=:line,
#         line=(:gray,1),
#         # marker=(:black,:circle,2),
#         zlims=(0,:auto),
#         xlabel=L"$z \, (\mathrm{mm})$",
#         legend=:topright,
#         legendtitle=L"$I_{0}=%$(Ix)\,\mathrm{A}$",
#         legendtitlefontsize=10,
#     )
#     vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$",line=(:red,:dash,2))
#     zscan = collect(minimum(z_profile[:,1]):0.001:maximum(z_profile[:,1]))
#     ## Dierckx
#     # fspline = Spline1D(z_profile[:,1],z_profile[:,2],k=3,s=0.01)
#     ## Define and optimize the negative spline function
#     # neg_spline(x) = -fspline(x[1])
#     # opt_result = optimize(neg_spline, [minimum(bin_centers)], [maximum(bin_centers)], [bin_center_max], Fminbox(LBFGS()))
#     # plot!(zscan,fspline(zscan))

#     # Loess
#     model = loess(z_profile[:,1],z_profile[:,2], span=loess_factor)
#     plot!(zscan,predict(model,zscan),
#         label="Loess",
#         line=(:purple4,2,0.5),
#     )

#     # Define the smoothed function
#     smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
#     # Find minimum using optimization
#     opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
#     z_max_fit = Optim.minimizer(opt_result)[1]
#     vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red,:dot,2))


#     return fig_2dhist , fig_prof , z_max_0 , z_max_fit , z_profile

# end


Icoils = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.75,0.80];
hist2d_cqd_up = Vector{Plots.Plot}()
histz_cqd_up = Vector{Plots.Plot}()
hist2d_qm = Vector{Plots.Plot}()
histz_qm = Vector{Plots.Plot}()
zpeak = zeros(Float64,length(Icoils),4)
@time for (idx,Io) in enumerate(Icoils)
    println("\t\tCurrent $(Io) A")
    # CO QUANTUM DYNAMICS
    # Add the final position according to CQD to each final projection 
    # [x0,y0,z0,vx0,vy0,vz0,θₑ,θₙ,xf,yf,zf]
    println("Atoms with magnetic moment going UP")
    atomsCQD_UP=[Vector{Float64}() for _ in 1:length(pairs_UP)]
    @time @threads for i=1:length(pairs_UP)
        atomsCQD_UP[i] = vcat(pairs_UP[i],
        CQD_Screen_position(Io,μₑ,pairs_UP[i][1:3],pairs_UP[i][4:6],pairs_UP[i][7])
        )
    end
    println("Atoms with magnetic moment going DOWN")
    atomsCQD_DOWN=[Vector{Float64}() for _ in 1:length(pairs_DOWN)]
    @time @threads for i=1:length(pairs_DOWN)
        atomsCQD_DOWN[i] = vcat(pairs_DOWN[i],
        CQD_Screen_position(Io,-μₑ,pairs_DOWN[i][1:3],pairs_DOWN[i][4:6],pairs_DOWN[i][7])
        )
    end

    # QUANTUM MECHANICS 
    # [x0,y0,z0, v0x,v0y,v0z, xf,yf,zf(2,2), zf(2,1),zf(2,0),zf(2,-1),zf(2,-2), zf(1,1),zf(1,0),zf(1,-1)]
    # println("Atoms in QM")
    # atomsQM=[Vector{Float64}() for _ in 1:Nss]
    # μF2p2 , μF2p1 , μF20 , μF2m1 , μF2m2 = μF_effective(Io,Ispin,2,2), μF_effective(Io,Ispin,2,1) , μF_effective(Io,Ispin,2,0) , μF_effective(Io,Ispin,2,-1) , μF_effective(Io,Ispin,2,-2)
    # μF1p1 , μF10 , μF1m1 = μF_effective(Io,Ispin,1,1) , μF_effective(Io,Ispin,1,0) , μF_effective(Io,Ispin,1,-1)
    # @time @threads for i=1:Nss
    #     atomsQM[i] = vcat(alive_slit[i],
    #     QM_Screen_position(Io,μF2p2, alive_slit[i][1:3], alive_slit[i][4:6]),
    #     QM_Screen_position(Io,μF2p1, alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF20,  alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF2m1, alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF2m2, alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF1p1, alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF10,  alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     QM_Screen_position(Io,μF1m1, alive_slit[i][1:3], alive_slit[i][4:6])[3],
    #     )
    # end

    atomsCQD_UP     = permutedims(reduce(hcat, atomsCQD_UP))
    atomsCQD_DOWN   = permutedims(reduce(hcat, atomsCQD_DOWN))
    # atomsQM     = permutedims(reduce(hcat, atomsQM))
    println("Data analysis : ",Io,"A")
    
    result_cqd_up   = CQD_analysis(Io,atomsCQD_UP;              z_upper = 8 , s_bin = 8 , loess_factor = 0.07)
    # result_qm_f1    = QM_analysis(Io,atomsQM,-0.5,[1,0,-1] ;   z_upper = 8 , s_bin = 8 , loess_factor = 0.07)

    push!(hist2d_cqd_up, result_cqd_up[1])
    # push!(hist2d_qm, result_qm_f1[1])

    display(result_cqd_up[1])
    # display(result_qm_f1[1])

    push!(histz_cqd_up, result_cqd_up[2])
    # push!(histz_qm, result_qm_f1[2])
    display(result_cqd_up[2])
    # display(result_qm_f1[2])

    writedlm(filename*"I$(idx)_cqd.csv",result_cqd_up[5],',')
    # writedlm(filename*"I$(idx)_qm.csv",result_qm_f1[5],',')

    # zpeak[idx,:] = [result_cqd_up[3],result_cqd_up[4],result_qm_f1[3],result_qm_f1[4]]
    zpeak[idx,:] = [result_cqd_up[3],result_cqd_up[4]]


end


Io= 0.0
s_bin=4
cam_pixelsize=0.0065

println("Atoms with magnetic moment going UP")
atomsCQD_UP=[Vector{Float64}() for _ in 1:length(pairs_UP)]
@time @threads for i=1:length(pairs_UP)
    atomsCQD_UP[i] = vcat(pairs_UP[i],
    CQD_Screen_position(Io,μₑ,pairs_UP[i][1:3],pairs_UP[i][4:6],pairs_UP[i][7])
    )
end

atomsCQD_UP     = permutedims(reduce(hcat, atomsCQD_UP))
data = atomsCQD_UP[:,[9,11]]
data = permutedims(reduce(hcat, filter(row -> -10 <= row[2] <= 10, eachrow(data)) |> collect ))

# Create 2D histogram
sim_xmin , sim_xmax = minimum(data[:,1])  , maximum(data[:,1])
sim_zmin , sim_zmax = minimum(data[:,2]) , maximum(data[:,2])
nbins_x = length(collect(sim_xmin:s_bin*cam_pixelsize:sim_xmax))+1
nbins_z = length(collect(sim_zmin:s_bin*cam_pixelsize:sim_zmax))+1
h0 = StatsBase.fit(Histogram,(data[:,1],data[:,2]),
    (range(sim_xmin,sim_xmax,length=nbins_x),range(sim_zmin,sim_zmax,length=nbins_z))
)
h0=normalize(h0,mode=:pdf)
bin_edges_x = collect(h0.edges[1])
bin_edges_z = collect(h0.edges[2])
bin_centers_x = (bin_edges_x[1:end-1] .+ bin_edges_x[2:end]) ./ 2  # Compute bin centers
bin_centers_z = (bin_edges_z[1:end-1] .+ bin_edges_z[2:end]) ./ 2  # Compute bin centers

z_profile = hcat(bin_centers_z, vec(mean(h0.weights,dims=1)))
# Find the index of the maximum value in the second column & Extract the corresponding value from the first column
zmax_idx = argmax(z_profile[:, 2])
z_max_0 = z_profile[zmax_idx, 1]


fig_2dhist = histogram2d(data[:,1],data[:,2],
    nbins=(nbins_x,nbins_z),
    normalize=:pdf,
    color=:inferno,
    title=L"Co Quantum Dynamics: $\vec{\mu}_{e} \upuparrows \hat{z}$",
    xlabel=L"$x \ (\mathrm{mm})$",
    ylabel=L"$z \ (\mathrm{mm})$",
    xlim=(sim_xmin, sim_xmax),
    ylim=(sim_zmin,sim_zmax),
    show_empty_bins=true,
)
hline!([z_max_0],label=false,line=(:red,:dash,1))

fig_prof = plot(z_profile[:,1],z_profile[:,2],
    label="Simulation",
    seriestype=:line,
    line=(:gray,1),
    # marker=(:black,:circle,2),
    title="CoQuantum Dynamics",
    zlims=(0,:auto),
    xlabel=L"$z \, (\mathrm{mm})$",
    legend=:topright,
    legendtitle=L"$I_{0}=%$(Io)\,\mathrm{A}$",
    legendtitlefontsize=10,
)
vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$",line=(:red,:dash,2))
zscan = collect(minimum(z_profile[:,1]):0.001:maximum(z_profile[:,1]))
## Dierckx
# fspline = Spline1D(z_profile[:,1],z_profile[:,2],k=3,s=0.01)
## Define and optimize the negative spline function
# neg_spline(x) = -fspline(x[1])
# opt_result = optimize(neg_spline, [minimum(bin_centers)], [maximum(bin_centers)], [bin_center_max], Fminbox(LBFGS()))
# plot!(zscan,fspline(zscan))

#BSplineKit
xs = z_profile[:,1]
ys = z_profile[:,2]
λ=0.01
weights = (1-λ)fill!(similar(xs), 1)
weights[zmax_idx]=2
S_fit = BSplineKit.fit(xs, ys,0.001; weights)
S_interp = BSplineKit.interpolate(xs, ys, BSplineOrder(4),BSplineKit.Natural())
scatter(xs, ys; label = "Data", marker = (:black,2))
plot!(xs, S_interp.(xs); label = "Interpolation", linewidth = 2)
plot!(xs, S_fit.(xs); label = "Fit (λ = $λ )", linewidth = 2)
neg_spline(x) = -S_fit(x[1])
opt_result = optimize(neg_spline, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
vline!([Optim.minimizer(opt_result)[1]])

# Loess
model = loess(z_profile[:,1],z_profile[:,2], span=0.1)
plot!(zscan,predict(model,zscan),
    label="Loess",
    line=(:purple4,2,0.5),
)

# Define the smoothed function
smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
# Find minimum using optimization
opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
z_max_fit = Optim.minimizer(opt_result)[1]
vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red,:dot,2))

rng = MersenneTwister(42)
Ndata = 20
xs = sort!(rand(rng, Ndata))



################################################################################################
################################################################################################
################################################################################################

Iexp = [ 0.0, 0.01, 0.02, 0.03, 0.05, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75 ]
zexp = [
    0.00124986,
    0.00900368,
    0.0227256,
    0.0629495,
    0.11486,
    0.390562,
    0.510494,
    0.631897,
    0.812013,
    1.12686,
    1.59759
]

sulqm=[0.0409
0.0566
0.0830
0.1015
0.1478
0.1758
0.2409
0.3203
0.4388
0.5433
0.6423
0.8394
1.1267
1.5288]

for i=1:length(sulqm)
    sulqm[i] = sulqm[i]+0.00625*rand(Uniform(-1,1))
end


sulcqd = [0.0179
0.0233
0.0409
0.0536
0.0883
0.1095
0.1713
0.2487
0.3697
0.4765
0.5786
0.7757
1.0655
1.4630]

for i=1:length(sulcqd)
    sulcqd[i] = sulcqd[i]+0.001*rand(Uniform(-1,1))
end

suli=   [ 0.0150
0.0200
0.0250
0.0300
0.0400
0.0500
0.0700
0.1000
0.1500
0.2000
0.2500
0.3500
0.5000
0.7500]


fig5=plot(Icoils[2:end],zpeak[2:end,2],
label="Coquantum dynamics",
# seriestype=:scatter,
marker=(:rect,:red,2),
markerstrokecolor=:red,
line=(:red,1,0.6),
xaxis=:log10,
yaxis=:log10,
xlims=(0.008,1),
legend=:topleft)
plot!(Icoils[2:end],zpeak[2:end,4],
label="Quantum mechanics",
# seriestype=:scatter,
marker=(:blue,:diamond,2),
markerstrokecolor=:blue,
line=(:blue,1))
plot!(Iexp[3:end], zexp[3:end],
label="COIL",
seriestype=:scatter,
marker=(:xcross,:black,3),
markeralpha=0.85,
markerstrokecolor=:black,
markerstrokewidth=3)     # mean(zpeak[:, 3:4], dims=2))
display(fig5)
savefig(fig5,filename*"_05.svg")



sulqm=[0.0409
0.0566
0.0830
0.1015
0.1478
0.1758
0.2409
0.3203
0.4388
0.5433
0.6423
0.8394
1.1267
1.5288]












t_run = Dates.canonicalize(Dates.now()-t_start)
# Create a dictionary with all the parameters
params = OrderedDict(
    "Experiment" => "FRISCH-SEGRÈ EXPERIMENT",
    "Equation" => "Bloch Equation ($equation)",
    "Filename" => filename,
    "Atom" => atom,
    "kᵢ:CQD" => "$ki",
    "B-field" => field,
    "Iw direction" => "$(Iw_direction)̂",
    "ODE system" => "$(θn_DiffEq)",
    "zₐ" => "$(1e6 .* zₐ)μm",
    "v" => "$(v)m/s",
    "Bᵣ" => "$b_remnant",
    "Bₑ" => "$(round(1e3*Be, digits=3))mT",
    "Bₙ" => "$(round(1e6*Bn, digits=3))μT",
    "Initial μₑ" => "$θe0_arrow [θₑ(tᵢ)=$(round(θe0/π, digits=4))π]",
    "Initial μₙ" => initial_μₙ,
    "θₙ(tᵢ)" => initial_μₙ == "CONSTANT" ? "$(round(θn_constant/π,digits=4))π" : "",
    "RNG" => string(rng)[1:findfirst(c -> c in ['{', '('], string(rng))-1],
    "N atoms" => "$N_atoms",
    "Time span" => "$(1e6 .* tspan)μs",
    "SG magnets" => "(BSG=$(BSG)T, ySG=±$(1e3 * ySG)mm)",
    "R²" => "$(round.(R_Squared; digits=4))",
    "δθ" => "$δθ",
    "Algorithm" => string(alg)[1:findfirst(c -> c in ['{', '('], string(alg))-1],
    "reltol" => "$reltol",
    "abstol" => "$abstol",
    "dtmin" => "$dtmin",
    "Start date" => Dates.format(t_start, "yyyy-mm-ddTHH-MM-SS"),
    "End date" => Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS"),
    "Run time" => "$t_run",
    "Hostname" => hostname,
    "Code name" => PROGRAM_FILE,
    "Iwire" => "$Iwire",
    "Prob(μₑ:↓)" => "$PSF_FS_global",
    "Prob(μₑ:↓|δt)" => "$(PSF_δt_avg[:,1])",
    "Prob(μₑ:↓|Bₑ>>B₀)" => "$PSF_FS_local",
    "Prob(μₑ:↓|Bₑ>>B₀|δt)" => "$(PSF_δt_avg[:,2])"
)
# Determine the maximum length of keys
max_key_length = maximum(length.(keys(params)))

open(filename * ".txt", "w") do file
    for (key, value) in params
        if value ≠ ""
            # Format each line with the key aligned
            write(file, @sprintf("%-*s = \t%s\n", max_key_length, key, value))
        end
    end
end

println("script   << $filename >>   has finished!")
println("$atom [ $experiment | $equation | $θe0_arrow | $initial_μₙ | $θn_DiffEq | $field | $(Int.(1e6.*tspan))μs | $(Int(1e6*zₐ))μm | $(v)m/s | $b_remnant | N=$N_atoms ]")
alert("script   << $filename >>   has finished!")

data = vcat(atomsQM[:,[7,14]],atomsQM[:,[7,15]],atomsQM[:,[7,16]])
data = permutedims(reduce(hcat, filter(row -> row[2] <= z_upper, eachrow(data)) |> collect ))

# Create 2D histogram
sim_xmin , sim_xmax = minimum(data[:,1])  , maximum(data[:,1])
sim_zmin , sim_zmax = minimum(data[:,2]) , maximum(data[:,2])
nbins_x = length(collect(sim_xmin:s_bin*cam_pixelsize:sim_xmax))+1
nbins_z = length(collect(sim_zmin:s_bin*cam_pixelsize:sim_zmax))+1
h0 = fit(Histogram,(data[:,1],data[:,2]),
    (range(sim_xmin,sim_xmax,length=nbins_x),range(sim_zmin,sim_zmax,length=nbins_z))
)
h0=normalize(h0,mode=:pdf)
bin_edges_x = collect(h0.edges[1])
bin_edges_z = collect(h0.edges[2])
bin_centers_x = (bin_edges_x[1:end-1] .+ bin_edges_x[2:end]) ./ 2  # Compute bin centers
bin_centers_z = (bin_edges_z[1:end-1] .+ bin_edges_z[2:end]) ./ 2  # Compute bin centers

z_profile = hcat(bin_centers_z, vec(mean(h0.weights,dims=1)))
# Find the index of the maximum value in the second column & Extract the corresponding value from the first column
zmax_idx = argmax(z_profile[:, 2])
z_max_0 = z_profile[zmax_idx, 1]


fig_2dhist = histogram2d(data[:,1],data[:,2],
    nbins=(nbins_x,nbins_z),
    normalize=:pdf,
    color=:inferno,
    title=L"Quantum mechanics: $F = 1$",
    xlabel=L"$x \ (\mathrm{mm})$",
    ylabel=L"$z \ (\mathrm{mm})$",
    xlim=(sim_xmin, sim_xmax),
    ylim=(sim_zmin,sim_zmax),
    show_empty_bins=true,
)
hline!([z_max_0],label=false,line=(:red,:dash,1))

fig_prof = plot(z_profile[:,1],z_profile[:,2],
    label="Simulation",
    seriestype=:line,
    line=(:gray,1),
    # marker=(:black,:circle,2),
    zlims=(0,:auto),
    xlabel=L"$z \, (\mathrm{mm})$",
    legend=:topright,
    legendtitle=L"$I_{0}=%$(Io)\,\mathrm{A}$",
    legendtitlefontsize=10,
)
vline!([z_max_0], label=L"$z_{\mathrm{max}} = %$(round(z_max_0,digits=6)) \, \mathrm{mm}$",line=(:red,:dash,2))
zscan = collect(minimum(z_profile[:,1]):0.001:maximum(z_profile[:,1]))
## Dierckx
# fspline = Spline1D(z_profile[:,1],z_profile[:,2],k=3,s=0.01)
## Define and optimize the negative spline function
# neg_spline(x) = -fspline(x[1])
# opt_result = optimize(neg_spline, [minimum(bin_centers)], [maximum(bin_centers)], [bin_center_max], Fminbox(LBFGS()))
# plot!(zscan,fspline(zscan))

# Loess
model = loess(z_profile[:,1],z_profile[:,2], span=loess_factor)
plot!(zscan,predict(model,zscan),
    label="Loess",
    line=(:purple4,2,0.5),
)

# Define the smoothed function
smooth_fn = x_val -> -Loess.predict(model, [x_val[1]])[1]
# Find minimum using optimization
opt_result = optimize(smooth_fn, [minimum(bin_centers_z)], [maximum(bin_centers_z)], [z_max_0], Fminbox(LBFGS()))
z_max_fit = Optim.minimizer(opt_result)[1]
vline!([z_max_fit], label=L"$z_{\mathrm{max}} = %$(round(z_max_fit,digits=6)) \, \mathrm{mm}$", line=(:red,:dot,2))





using 
plot(bin_centers_z,vec(mean(h0.weights,dims=1)),
    seriestype=:line,
    line=(:gray,1),
    marker=(:black,:circle,2),
    # xlims=(-1,8),
)
x = bin_centers_z[findall(x -> (-1 <= x <= 9), bin_centers_z)]
y = vec(mean(h0.weights,dims=1))[findall(x -> (-1 <= x <= 9), bin_centers_z)]
splfit = Spline1D(x,y,
    # w=ones(length(x)),
    k=3, 
    s=length(x)*0.5)

    model = loess(x, y, span=0.1)


zz=collect(minimum(bin_centers_z):0.013:maximum(bin_centers_z))
us = range(extrema(x)...; step = 0.1)
plot!(zz,(splfit(zz)))
plot!(us,predict(model,us),line=(:green,3))


histogram(vec(mean(h0.weights,dims=1)),nbins=588,normalize=:probability)


xs = 10 .* rand(100)
ys = sin.(xs) .+ 0.5 * rand(100)

model = loess(xs, ys, span=0.5)
vs = predict(model, us)

scatter(xs, ys)
plot!(us, vs, legend=false)


heatmap(h0.weights',nbins=nbins_z)



minimum(1e3*atomsCQD_UP[:,9]):0.026:maximum(1e3*atomsCQD_UP[:,9])
minimum(1e3*atomsCQD_UP[:,11]):0.026:maximum(1e3*atomsCQD_UP[:,11])




d1 = randn(10_000)
d2 = randn(10_000)

nbins1 = 25
nbins2 = 10
	
hist = fit(Histogram, (d1,d2),
		(range(minimum(d1), stop=maximum(d1), length=nbins1+1),
		range(minimum(d2), stop=maximum(d2), length=nbins2+1)))
plot(hist)

data = [
    0.0  0.074501;
    0.1  0.127343;
    0.2  0.187198;
    0.3  0.299073;
    0.4  0.435718;
    0.5  0.467139;
    0.6  0.62702;
    0.7  0.631098;
    0.8  0.774073;
    0.9  0.793128;
    1.0  0.84104;
    1.1  0.886343;
    1.2  0.93662;
    1.3  0.956826;
    1.4  0.966104;
    1.5  0.999325;
    1.6  0.993967;
    1.7  0.98652;
    1.8  0.989205;
    1.9  0.914493;
    2.0  0.894332;
    2.1  0.884692;
    2.2  0.835543;
    2.3  0.790565;
    2.4  0.668164;
    2.5  0.52381;
    2.6  0.591465;
    2.7  0.406899;
    2.8  0.260562;
    2.9  0.214678;
    3.0  0.181986;
    3.1  0.0490647
]

# Sample data (replace with your own data)
x = 0:0.1:10
y = sin.(x) + 0.1 * randn(length(x))  # Adding some noise to make it interesting


# Define the cost function
function cost_fn(x,y,smoothing_factor)
    # Fit the spline using cubic interpolation (without smoothing)
    spline = CubicSplineInterpolation(x, y, extrapolation_bc=Line())

    # Calculate the residual sum of squares (RSS)
    residuals = sum((spline(x) .- y).^2)
    
    # Approximate the second derivative for roughness penalty
    dx = diff(x)
    second_derivative = diff(spline(x)) ./ dx
    second_derivative_penalty = sum(second_derivative.^2)  # Roughness penalty

    # Define the cost function as a weighted sum of residuals and penalty
    p = smoothing_factor
    cost = p*residuals + (1-p) * second_derivative_penalty
    
    return cost
end

# Function to fit the smoothing spline with a given smoothing parameter
function fit_spline(x, y, smoothing_factor)
    # Minimize the cost function to get the optimal smoothing factor
    result = optimize(cost_fn, 0.0, 1.0, BFGS(), x, y, smoothing_factor)

    # Return the fitted spline
    optimal_smoothing_factor = Optim.minimizer(result)[1]
    spline = CubicSplineInterpolation(x, y, extrapolation_bc=Line())

    return spline
end


# Fit the spline
fitted_spline = fit_spline(x, y, 0.98)


# Use Optim.jl to minimize the cost function
result = optimize(cost_fn, 0.0, 1.0, BFGS())  # Optimization over smoothing factor (0 to 1)

# Get the optimal smoothing factor
optimal_smoothing_factor = Optim.minimizer(result)[1]
println("Optimal smoothing factor: ", optimal_smoothing_factor)

# Fit the spline using the optimal smoothing factor
spline_with_optimal_smoothing = CubicSplineInterpolation(x, y, extrapolation_bc=Line())

# Plot the original data and the fitted spline
plot(x, y, label="Original data", marker=:o)
plot!(x, spline_with_optimal_smoothing(x), label="Fitted spline with optimal smoothing", linewidth=2)







# Function to compute the smoothing spline
function (x, y, smoothing_factor)
    # Fit the spline using cubic interpolation
    spline = CubicSplineInterpolation(x, y, extrapolation_bc=Line())
    
    # Define the penalty term: the integral of the square of the second derivative (roughness)
    # This is an approximation of the smoothness of the spline
    dx = diff(x)
    second_derivative_penalty = sum((diff(spline(x)[2:end])./dx[2:end]).^2)
    
    # Calculate the residuals (least squares)
    residuals = sum((spline(x) .- y).^2)
    
    # Define the cost function: a weighted sum of residuals and penalty
    cost = residuals .+ smoothing_factor .* second_derivative_penalty
    
    return cost
end

# Define the cost function for optimization (smoothing_factor will be optimized)
function cost_fn(smoothing_factor)
    return fit_spline(x, y, smoothing_factor[1])
end

# Use Optim.jl to minimize the cost function
result = optimize(cost_fn, 0.0, 1.0)  # smoothing_factor is between 0 and 1

# Get the optimal smoothing factor
optimal_smoothing_factor = Optim.minimizer(result)[1]
println("Optimal smoothing factor: ", optimal_smoothing_factor)

# Fit the final spline with the optimal smoothing factor
final_spline = CubicSplineInterpolation(x, y, extrapolation_bc=Line())

# Plot the original data and the fitted spline
plot(x, y, label="Original data", marker=:o)
plot!(x, final_spline(x), label="Fitted spline", linewidth=2)
plot!(x,fit_spline)

# Fit the final spline with the optimal smoothing factor
final_spline = CubicSplineInterpolation(x, y, extrapolation_bc=Line())

sfitting = Spline1D(x,y,s=0.99)
listpi = collect(0:0.001:3π)

plot(x,y, seriestype=:scatter)
plot!(listpi,sfitting(listpi))






y_FurnaceToSlit+y_SlitToSG+y_SG+y_SGToScreen