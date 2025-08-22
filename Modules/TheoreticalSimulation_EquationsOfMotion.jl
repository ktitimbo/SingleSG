# EQUATIONS OF MOTION

# Co-Quantum Dynamics

# CQD Equations of motion
@inline function CQD_EqOfMotion(t,Ix,μ,r0::Vector{Float64},v0::Vector{Float64},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)

    x0, y0, z0 = r0
    v0x, v0y, v0z = v0

    # Key times
    tf1 =  default_y_FurnaceToSlit / v0y
    tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG) / v0y
    tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v0y
    # tF  = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG + default_y_SGToScreen) / v0y  # (unused here)

    cqd_sign = sign(θn-θe) 
    ωL       = abs(γₑ * BvsI(Ix) )
    acc_0    = μ*GvsI(Ix)/p.M
    kω       = cqd_sign*kx*ωL

    θe_half = θe / 2
    cosθ2 = cos(θe_half)^2
    sinθ2 = sin(θe_half)^2
    tanθ2 = tan(θe_half)^2

    if t <= tf1     # Furnace to Slit
        x = x0 + v0x*t 
        y = y0 + v0y*t 
        z = z0 + v0z*t
        vx , vy , vz = v0x , v0y , v0z
    elseif t <= tf2    # Slit to SG apparatus
        x = x0 + v0x*t 
        y = y0 + v0y*t
        z = z0 + v0z*t
        vx , vy , vz = v0x , v0y , v0z
    elseif t <= tf3   # Crossing the SG apparatus
        vx = v0x
        vy = v0y
        vz = v0z + acc_0*(t-tf2) + acc_0/kω * log( cosθ2 + exp(-2*kω*(t-tf2))*sinθ2 )
        x = x0 + v0x*t 
        y = y0 + v0y*t
        z = z0 + v0z*t + 0.5*acc_0*(t-tf2)^2 + acc_0/kω*log(cosθ2)*(t-tf2) + 0.5/(kω)^2 * acc_0 * ( polylogarithm(2,-exp(-2*kω*(t-tf2))*tanθ2) - polylogarithm(2,-tanθ2) )
    elseif t > tf3 # Travel to the Screen
        x = x0 + v0x*t
        y = y0 + v0y*t
        z = z0 + v0z*t + 0.5*acc_0*( (t-tf2)^2 - (t-tf3)^2) + acc_0/kω*default_y_SG/v0y * ( log(cosθ2) + v0y/default_y_SG*log(cosθ2+exp(-2*kω*default_y_SG/v0y)*sinθ2)*(t-tf3) ) + 0.5*acc_0/kω^2*( polylogarithm(2,-exp(-2*kω*default_y_SG/v0y)*tanθ2) - polylogarithm(2,-tanθ2) )
        vx = v0x
        vy = v0y
        vz = v0z + acc_0*default_y_SG/v0y + acc_0/kω*log(cosθ2 + exp(-2*kω*default_y_SG/v0y)*sinθ2)
    end

    r = SVector{3,Float64}(x, y, z)
    v = SVector{3,Float64}(vx, vy, vz)
    return r, v
end


# CQD equations of motion only along the z-coordinate
@inline function CQD_EqOfMotion_z(t,Ix::Float64,μ::Float64,r0::AbstractVector{Float64},v0::AbstractVector{Float64},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)
    v0y = v0[2]
    v0z = v0[3]
    z0 = r0[3]
    
    tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG ) / v0y
    tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG ) / v0y

    cqd_sign = sign(θn-θe) 
    ωL       = abs( γₑ * BvsI(Ix) )
    acc_0    = μ*GvsI(Ix)/p.M
    kω       = cqd_sign*kx*ωL

    # Precompute angles
    θe_half = θe / 2
    tanθ = tan(θe_half)
    tanθ2 = tanθ^2
    cosθ2 = cos(θe_half)^2
    sinθ2 = sin(θe_half)^2
    log_cos2 = log(cosθ2)
    polylog_0 = polylogarithm(2, -tanθ2)

    if t <= tf2
        return z0 + v0z*t
    elseif t <= tf3   # Crossing the SG apparatus
        Δt = t - tf2
        exp_term = exp(-2 * kω * Δt)
        polylog_t = polylogarithm(2, -exp_term * tanθ2)

        return z0 + v0z*t + 0.5 * acc_0 * Δt^2 + acc_0 / kω * log_cos2 * Δt + 0.5 * acc_0 / kω^2 * ( polylog_t - polylog_0 )
    
    else # t > tf3 # Travel to the Screen
        Δt2 = t - tf2
        Δt3 = t - tf3
        τ_SG = default_y_SG / v0y
        exp_SG = exp(-2 * kω * τ_SG)
        polylog_SG = polylogarithm(2, -exp_SG * tanθ2)
        log_term = log(cosθ2 + exp_SG * sinθ2)

        return z0 + v0z*t + 0.5*acc_0*( Δt2^2 - Δt3^2 ) + acc_0 / kω * τ_SG * (log_cos2 + log_term * Δt3 / τ_SG) + 0.5 * acc_0 / kω^2 * (polylog_SG - polylog_0)
    end
end

# CQD Screen position
function CQD_Screen_position(Ix,μ::Float64,r0::AbstractVector{Float64},v0::AbstractVector{Float64},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)

    x0, y0, z0 = r0
    v0x, v0y, v0z = v0

    L1 = default_y_FurnaceToSlit 
    L2 = default_y_SlitToSG
    Lsg = default_y_SG
    Ld = default_y_SGToScreen
    Ltot = L1 + L2 + Lsg + Ld

    # Physics parameters
    cqd_sign = sign(θn-θe) 
    acc_0 = μ * GvsI(Ix) / p.M
    ωL = abs(γₑ * BvsI(Ix))
    kω = cqd_sign * kx * ωL

    # Common trig values
    θe_half = θe / 2
    cos2 = cos(θe_half)^2
    sin2 = sin(θe_half)^2
    tan2 = tan(θe_half)^2
    exp_term = exp(-2 * kω * Lsg / v0y)

    x = x0 + Ltot * v0x / v0y
    y = y0 + Ltot
    z = z0 + Ltot * v0z / v0y + 0.5*acc_0/v0y^2*((Lsg+Ld)^2-Ld^2) + acc_0/kω*Lsg/v0y*( log(cos2) + Ld/Lsg * log( cos2 + exp_term*sin2 ) ) + 0.5*acc_0/kω^2 * ( polylogarithm(2, -exp_term*tan2) - polylogarithm(2, -tan2) )
    return SVector{3,Float64}(x,y,z)
end

function CQD_Screen_velocity(Ix,μ::Float64,v0::AbstractVector{Float64},θe::Float64, θn::Float64, kx::Float64, p::AtomParams)

    v0x, v0y, v0z = v0

    Lsg = default_y_SG

    # Physics parameters
    cqd_sign = sign(θn-θe) 
    acc_0 = μ * GvsI(Ix) / p.M
    ωL = abs(γₑ * BvsI(Ix))
    kω = cqd_sign * kx * ωL

    # Common trig values
    θe_half = θe / 2
    cos2 = cos(θe_half)^2
    sin2 = sin(θe_half)^2
    tan2 = tan(θe_half)^2
    exp_term = exp(-2 * kω * Lsg / v0y)

    vx = v0x
    vy = v0y
    vz = v0z + acc_0 * Lsy / v0y + acc_0 / kω * log( cos2 + exp_term*sin2) 
    return SVector{3,Float64}(vx,vy,vz)
end

# Quantum Mechanics : Classical Trajectories

# QM equations of motion
@inline function QM_EqOfMotion(t,Ix,f,mf,r0::Vector{Float64},v0::Vector{Float64}, p::AtomParams)
    @assert length(r0) == 3 "r0 must have length 3"
    @assert length(v0) == 3 "v0 must have length 3"

    x0, y0, z0 = r0
    v0x, v0y, v0z = v0
    @assert v0y != 0.0 "y-velocity must be nonzero."
    @assert t >= 0 "time t must be ≥ 0"  # optional; remove if you allow negative t

        # Segment times (in seconds)
    tf1 =  default_y_FurnaceToSlit / v0y                               # slit entrance
    tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG) / v0y                 # SG entrance
    tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v0y          # SG exit
    tF  = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG + default_y_SGToScreen) / v0y  # screen

    # acc_0    = μ*cos(θ)*GvsI(Ix)/p.M
    μ =  μF_effective(Ix,f,mf,p)
    acc_0    = μ * GvsI(Ix) / p.M

    if t <= tf2     # Furnace to Slit and Slit to SG apparatus
        x = x0 + v0x*t 
        y = y0 + v0y*t 
        z = z0 + v0z*t
        vx , vy , vz = v0x , v0y , v0z
    elseif t <= tf3   # Crossing the SG apparatus
        vx = v0x
        vy = v0y
        vz = v0z + acc_0*(t-tf2)
        x = x0 + v0x*t 
        y = y0 + v0y*t
        z = z0 + v0z*t + 0.5*acc_0*(t-tf2)^2
    elseif t > tf3 # Travel to the Screen
        x = x0 + v0x*t
        y = y0 + v0y*t
        z = z0 + v0z*t + acc_0 * default_y_SG / v0y * (t - 0.5*(tf2+tf3))
        vx = v0x
        vy = v0y
        vz = v0z + acc_0*default_y_SG/v0y 
    end

    r = SVector{3,Float64}(x, y, z)
    v = SVector{3,Float64}(vx, vy, vz)
    return r, v
end

@inline function QM_EqOfMotion_z(t,Ix::Float64,f,mf,r0::AbstractVector{Float64},v0::AbstractVector{Float64}, p::AtomParams)
    v0y = v0[2]
    v0z = v0[3]
    z0 = r0[3]
    
    tf2 = (default_y_FurnaceToSlit + default_y_SlitToSG ) / v0y
    tf3 = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG ) / v0y

    # acc_0    = μ*cos(θ)*GvsI(Ix)/p.M
    μ =  μF_effective(Ix,f,mf,p)
    acc_0    = μ * GvsI(Ix) / p.M

    if t <= tf2
        return z0 + v0z*t
    elseif t <= tf3   # Crossing the SG apparatus
        Δt = t - tf2
        return  z0 + v0z*t + 0.5*acc_0*Δt^2
    else # t > tf3 # Travel to the Screen
        τ_SG = default_y_SG / v0y
        return z0 + v0z*t + acc_0 * τ_SG * (t - 0.5*(tf2+tf3))
    end
end

function QM_Screen_position(Ix,f,mf,r0::AbstractVector{Float64},v0::AbstractVector{Float64}, p::AtomParams)

    x0, y0, z0 = r0
    v0x, v0y, v0z = v0

    # Geometry
    Lsg = default_y_SG
    Ld = default_y_SGToScreen
    Ltot = default_y_FurnaceToSlit  + default_y_SlitToSG + Lsg + Ld

    # Physics parameters
    # acc_0 = μ * cos(θ) * GvsI(Ix) / p.M
    μ =  μF_effective(Ix,f,mf,p)
    acc_0 = μ * GvsI(Ix) / p.M

    x = x0 + Ltot * v0x / v0y
    y = y0 + Ltot
    z = z0 + Ltot * v0z / v0y + 0.5*acc_0/v0y^2*((Lsg+Ld)^2-Ld^2)
    return SVector{3,Float64}(x, y, z)
end

function QM_Screen_velocity(Ix,f,mf,v0::AbstractVector{Float64}, p::AtomParams)
    v0x, v0y, v0z = v0

    # Geometry
    Lsg = default_y_SG

    # Physics parameters
    μ =  μF_effective(Ix,f,mf,p)
    acc_0 = μ * GvsI(Ix) / p.M

    vx = v0x
    vy = v0y
    vz = v0z + acc_0 * Lsg/v0y
    return SVector{3,Float64}(vx, vy, vz)
end
