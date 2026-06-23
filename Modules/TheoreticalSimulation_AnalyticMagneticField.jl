# Finite parallel wires

# Helpers
G(y, ρ) = (y+DEFAULT_ℓ)/sqrt((y+DEFAULT_ℓ)^2+ρ^2) - (y-DEFAULT_ℓ)/sqrt((y-DEFAULT_ℓ)^2+ρ^2)
F(y, ρ) = G(y, ρ) / ρ^2

Ap(y, ρ) = 1 / ((y + DEFAULT_ℓ)^2 + ρ^2)^(3/2)
Am(y, ρ) = 1 / ((y - DEFAULT_ℓ)^2 + ρ^2)^(3/2)
dFdy(y, ρ) = Ap(y, ρ) - Am(y, ρ)
dFdρ(y, ρ) = -((y+DEFAULT_ℓ)*Ap(y, ρ) - (y-DEFAULT_ℓ)*Am(y, ρ))/ρ - 2*G(y, ρ)/ρ^3


function B_total(x,y,z; z0=1.3*DEFAULT_𝒶 ,Iw=0.2)
    ρ1, ρ2 = hypot(x-DEFAULT_𝒶 , z-z0), hypot(x+DEFAULT_𝒶 , z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end
    F1, F2 = F(y, ρ1), F(y, ρ2)
    C = -μ₀*Iw/(4π)
    Bx = C*(z-z0)*(F2-F1)
    Bz = C*((x-DEFAULT_𝒶 )*F1 - (x+DEFAULT_𝒶 )*F2)
    return (Bx,0.0,Bz)
end

function grad_B(x, y, z; z0=1.3*DEFAULT_𝒶, Iw=0.2)
    ρ1 = hypot(x-DEFAULT_𝒶, z-z0)
    ρ2 = hypot(x+DEFAULT_𝒶, z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end
    F1 = F(y, ρ1)
    F2 = F(y, ρ2)
    C = -μ₀*Iw/(4π)
    Δz = z - z0

    # dρ/dx, dρ/dz
    dρ1dx = (x - DEFAULT_𝒶)/ρ1;   dρ2dx = (x + DEFAULT_𝒶)/ρ2
    dρ1dz = Δz/ρ1;        dρ2dz = Δz/ρ2


    # F partials
    dF1dρ = dFdρ(y, ρ1);  dF2dρ = dFdρ(y, ρ2)
    dF1dy = dFdy(y, ρ1);  dF2dy = dFdy(y, ρ2)

    # ∂ᵢBx
    dBxdx = C * Δz * ( dF2dρ*dρ2dx - dF1dρ*dρ1dx )
    dBxdy = C * Δz * ( dF2dy - dF1dy )
    dBxdz = C * (F2-F1 + Δz*(dF2dρ*dρ2dz - dF1dρ*dρ1dz))
    # ∂ᵢBy = 0 ∀ i
    # ∂ᵢBz
    dBzdx = C * ( F1 - F2 + (x-DEFAULT_𝒶)*dF1dρ*dρ1dx - (x+DEFAULT_𝒶)*dF2dρ*dρ2dx )
    dBzdy = C * ( (x-DEFAULT_𝒶)*dF1dy - (x+DEFAULT_𝒶)*dF2dy )
    dBzdz = C * ( (x-DEFAULT_𝒶)*dF1dρ*dρ1dz - (x+DEFAULT_𝒶)*dF2dρ*dρ2dz )

    return [
        dBxdx dBxdy dBxdz;
        0 0 0;
        dBzdx dBzdy dBzdz
    ]
end

function grad_normB(x::Real, y::Real, z::Real;
                    Iw::Real = 0.2,
                    z0::Real = 1.3 * DEFAULT_𝒶
)

    Bx, By, Bz = B_total(x, y, z; z0=z0, Iw=Iw)
    Bmag = sqrt(Bx^2 + By^2 + Bz^2)

    if iszero(Bmag)
        return (0.0, 0.0, 0.0)
    end

    # J[i,j] = ∂Bᵢ/∂xⱼ   →   shape (3,3), columns = [∂/∂x, ∂/∂y, ∂/∂z]
    J = grad_B(x, y, z; z0=z0, Iw=Iw)

    # ∂|B|/∂xⱼ = (B · ∂B/∂xⱼ) / |B|  =  (Bx*J[1,j] + By*J[2,j] + Bz*J[3,j]) / |B|
    dBdx = (Bx * J[1,1] + By * J[2,1] + Bz * J[3,1]) / Bmag
    dBdy = (Bx * J[1,2] + By * J[2,2] + Bz * J[3,2]) / Bmag
    dBdz = (Bx * J[1,3] + By * J[2,3] + Bz * J[3,3]) / Bmag

    return (dBdx, dBdy, dBdz)
end

# ── grad_normB accepting precomputed B ───────────────────────────────────────
# avoids a redundant B_total call inside the ODE — B is already known
function grad_normB(x::Real, y::Real, z::Real,
                    Bx::Real, By::Real, Bz::Real;
                    Iw::Real = 0.2,
                    z0::Real = 1.3 * DEFAULT_𝒶
)
    Bmag = sqrt(Bx^2 + By^2 + Bz^2)
    iszero(Bmag) && return (0.0, 0.0, 0.0)

    J    = grad_B(x, y, z; Iw = Iw, z0 = z0)

    dBdx = (Bx * J[1,1] + By * J[2,1] + Bz * J[3,1]) / Bmag
    dBdy = (Bx * J[1,2] + By * J[2,2] + Bz * J[3,2]) / Bmag
    dBdz = (Bx * J[1,3] + By * J[2,3] + Bz * J[3,3]) / Bmag

    return (dBdx, dBdy, dBdz)
end


# In the limit ℓ → ∞
function approx_B_total(x,y,z; z0=1.3*DEFAULT_𝒶,Iw=0.2)
    ρ1, ρ2 = hypot(x-DEFAULT_𝒶, z-z0), hypot(x+DEFAULT_𝒶, z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end
    inv_sq_ρ1 = 1/ρ1^2
    inv_sq_ρ2 = 1/ρ2^2
    C = μ₀*Iw/(2π)
    Bx = C*(z-z0)*(inv_sq_ρ2 - inv_sq_ρ1)
    Bz = C*((x-DEFAULT_𝒶)*inv_sq_ρ1 - (x+DEFAULT_𝒶)*inv_sq_ρ2)
    return (Bx,0.0,Bz)
end

# In the limit ℓ → ∞
function approx_grad_B(x, y, z; z0=1.3*DEFAULT_𝒶, Iw=0.2)
    ρ1 = hypot(x-DEFAULT_𝒶, z-z0)
    ρ2 = hypot(x+DEFAULT_𝒶, z-z0)
    if ρ1 == 0 || ρ2 == 0
        throw(DomainError("Point lies on a wire (ρ=0): field/gradient undefined."))
    end

    inv_sq_ρ1 = 1/ρ1^2
    inv_sq_ρ2 = 1/ρ2^2
    C = μ₀*Iw/(2π)
    Δz = z - z0


    # ∂ᵢBx
    dBxdx = -2 * C * Δz * ( (x+DEFAULT_𝒶)*inv_sq_ρ2^2 - (x-DEFAULT_𝒶)*inv_sq_ρ1^2 )
    dBxdz = C * (inv_sq_ρ2-inv_sq_ρ1) + 2 * C * Δz^2 * (inv_sq_ρ1^2-inv_sq_ρ2^2 )
    # ∂ᵢBy = 0 ∀ i
    # ∂ᵢBz
    dBzdx = C * (inv_sq_ρ1-inv_sq_ρ2) - 2 * C * ((x-DEFAULT_𝒶)^2*inv_sq_ρ1^2-(x+DEFAULT_𝒶)^2*inv_sq_ρ2^2 )
    dBzdz = -2* C * Δz * ( (x-DEFAULT_𝒶)*inv_sq_ρ1^2 - (x+DEFAULT_𝒶)*inv_sq_ρ2^2 )

    return [
        dBxdx 0.0 dBxdz;
        0.0 0.0 0.0;
        dBzdx 0.0 dBzdz
    ]
end

# In the limit ℓ → ∞
function approx_grad_normB(x::Real, y::Real, z::Real;
                    Iw::Real = 0.2,
                    z0::Real = 1.3 * DEFAULT_𝒶
)

    Bx, By, Bz = approx_B_total(x, y, z; z0=z0, Iw=Iw)
    Bmag = sqrt(Bx^2 + By^2 + Bz^2)

    if iszero(Bmag)
        return (0.0, 0.0, 0.0)
    end

    # J[i,j] = ∂Bᵢ/∂xⱼ   →   shape (3,3), columns = [∂/∂x, ∂/∂y, ∂/∂z]
    J = approx_grad_B(x, y, z; z0=z0, Iw=Iw)

    # ∂|B|/∂xⱼ = (B · ∂B/∂xⱼ) / |B|  =  (Bx*J[1,j] + By*J[2,j] + Bz*J[3,j]) / |B|
    dBdx = (Bx * J[1,1] + By * J[2,1] + Bz * J[3,1]) / Bmag
    dBdy = (Bx * J[1,2] + By * J[2,2] + Bz * J[3,2]) / Bmag
    dBdz = (Bx * J[1,3] + By * J[2,3] + Bz * J[3,3]) / Bmag

    return (dBdx, dBdy, dBdz)
end

# In the limit ℓ → ∞
function approx_dnormBdz(x,z; Iw=0.2, z0=1.3*DEFAULT_𝒶)
    ρ1 = hypot(x-DEFAULT_𝒶, z-z0)
    ρ2 = hypot(x+DEFAULT_𝒶, z-z0)

    Δz = z - z0
    C = μ₀*Iw/(2π)

    return -4 * DEFAULT_𝒶 * C * Δz / (ρ1^3 * ρ2^3) * ( x^2 + DEFAULT_𝒶^2 + Δz^2)
end

function calibrate_Ieff_for_Bz(I_list; plot_check=true)

    B_measured = BvsI.(I_list);
    B_model    = [B_total(0, 0, 0; Iw=x)[3]    for x in I_list];

    itp_B = DataInterpolations.AkimaInterpolation(I_list, B_model; extrapolation = ExtrapolationType.Linear);
    I_eff_B = (I) -> itp_B(BvsI(I));


    if plot_check
        B_corrected = [B_total(0, 0, 0; Iw=I_eff_B(x))[3]    for x in I_list];

        plt_B = plot(I_list, B_measured,   
            label="BvsI (target)",  
            marker=(:circle,2,:white), 
            seriestype=:scatter,
            title="Field calibration");
        plot!(plt_B, I_list, B_corrected,  label=L"B_total($I_{eff}$)", lw=2, ls=:dash);
        plot!(plt_B, I_list, B_model,      label=L"B_total($I_{w}$) raw", lw=2, ls=:dot);
        plot!(plt_B, xlabel="Current (A)", ylabel="Magnetic field (T)")



        display(plt_B)
    end

    return I_eff_B
end

function calibrate_gradient(
    I_list, I_eff_B;
    span::Real        = 0.12,
    degree::Int       = 3,
    plot_check::Bool  = true,
    epsG::Float64     = 1e-300,
    anchor_zero::Bool = true,
)
    I        = sort(collect(float.(I_list)))
    G_target = GvsI.(I)

    Gz_raw(Ival) = grad_normB(0, 0, 0; Iw = I_eff_B(Ival))[3]
    G_raw        = Gz_raw.(I)

    # exclude I=0 and any non-finite or same-sign violations from log-space fit
    mask = (I .> 0.0)            .&    # I=0 excluded from log fit
           isfinite.(G_target)   .&
           isfinite.(G_raw)      .&
           (abs.(G_target) .> epsG) .&
           (abs.(G_raw)    .> epsG) .&
           (sign.(G_target) .== sign.(G_raw))

    Ifit = I[mask]
    logy = log.(abs.(G_target[mask] ./ G_raw[mask]))

    # anchor at I=0: S(0)=1 → log(S)=0, since both field and gradient vanish together
    if anchor_zero
        Ifit = vcat(0.0, Ifit)
        logy = vcat(0.0, logy)    # S(0) = 1 exactly
    end

    xmin, xmax = extrema(Ifit)
    model      = loess(Ifit, logy; span=span, degree=degree)

    function scale_factor(Ival)
        isfinite(Ival) || return NaN
        return exp(predict(model, [clamp(float(Ival), xmin, xmax)])[1])
    end

    if plot_check
        Iplot       = range(first(I), last(I), length=500)
        G_raw_plot  = Gz_raw.(Iplot)
        G_corr_plot = scale_factor.(Iplot) .* G_raw_plot

        plt1 = plot(I, G_target,
                    seriestype=:scatter, marker=(:circle,3,:white), label="GvsI target",
                    xlabel="Current (A)", ylabel="Gradient (T/m)", title="Gradient calibration")
        plot!(plt1, Iplot, G_raw_plot,  lw=2, ls=:dot,  label=L"raw: $\nabla|B|(I_{\mathrm{eff},B})$")
        plot!(plt1, Iplot, G_corr_plot, lw=2, ls=:dash, label=L"corrected: $S(I)\nabla|B|(I_{\mathrm{eff},B})$")

        plt2 = plot(Ifit, exp.(logy),
                    seriestype=:scatter, marker=(:circle,3,:white), label="target / raw",
                    xlabel="Current (A)", ylabel=L"S(I)", title="Fitted gradient scale")
        plot!(plt2, Iplot, scale_factor.(Iplot), lw=2, label="LOESS fit, span=$span")

        display(plot(plt1, plt2; layout=(1,2), size=(1000,400), left_margin=5mm, bottom_margin=4mm))
    end

    return scale_factor
end

# ── Calibration struct ────────────────────────────────────────────────────────
struct SGCalibration
    I_eff_B    :: Function
    grad_scale :: Function
end

function build_calibration(I_list; span=0.12, degree=4, plot_check=true)
    I_eff_B    = calibrate_Ieff_for_Bz(I_list; plot_check=plot_check)
    grad_scale = calibrate_gradient(I_list, I_eff_B; span=span, degree=degree,
                                    plot_check=plot_check)
    return SGCalibration(I_eff_B, grad_scale)
end

#______________________________________________________________________________________
# ATOM PROPAGATION CQD

# propagates the particles from the oven to a final position with free motion
function propagate_to_SG_entrance(data;
                                y_SG_entrance = DEFAULT_SG_magnet_entrance)

    N    = size(data, 1)
    # output: [x y z vx vy vz θ0] at SG entrance
    data_SG = Matrix{Float64}(undef, N, 7)

    Threads.@threads for ii in 1:N
        r0 = @view data[ii, 1:3]
        v0 = @view data[ii, 4:6]
        vy = v0[2]

        # combined free flight: oven → slit → SG entrance
        Δt = (y_SG_entrance - r0[2]) / vy

        data_SG[ii, 1] = r0[1] + v0[1] * Δt
        data_SG[ii, 2] = y_SG_entrance
        data_SG[ii, 3] = r0[3] + v0[3] * Δt
        data_SG[ii, 4] = v0[1]
        data_SG[ii, 5] = vy
        data_SG[ii, 6] = v0[3]
        data_SG[ii, 7] = data[ii, 7]    # θ0 carried through unchanged
    end

    return data_SG
end


# ── Free flight ──────────────────────────────────────────────────────────────
@inline function free_flight(r, v, y_target)
    Δt = (y_target - r[2]) / v[2]
    return r .+ v .* Δt, v
end


# ── EOM ──────────────────────────────────────────────────────────────────────
# grad_mask: NTuple{3} of 0/1 to selectively enable x/y/z force components
function make_eom(Iw_eff::Float64, S::Float64,
                  μ_over_m::Float64, k::Float64, θ0::Float64,
                  t_in::Float64, y_SG_center::Float64,
                  grad_mask::NTuple{3,Float64})

    function eom!(du, u, p, t)
        x, y, z, vx, vy, vz = u
        y_loc = y - y_SG_center

        # single B_total call — result reused in grad_normB
        Bx, By, Bz = B_total(x, y_loc, z; Iw=Iw_eff)
        B0         = sqrt(Bx^2 + By^2 + Bz^2)

        # gradient with precomputed B, scaled by S
        dBdx, dBdy, dBdz = S .* grad_normB(x, y_loc, z, Bx, By, Bz; Iw=Iw_eff)

        # dissipative relaxation: θ(t,|B|)
        τ     = t - t_in
        ξ     = tan(θ0 / 2) * exp(-k * abs(γₑ) * B0 * τ)
        θ     = 2 * atan(ξ)
        cosθ  = cos(θ)
        sin²θ = sin(θ)^2

        prefactor = μ_over_m * (cosθ + k * abs(γₑ) * B0 * τ * sin²θ)

        du[1] = vx
        du[2] = vy
        du[3] = vz
        du[4] = grad_mask[1] * prefactor * dBdx
        du[5] = grad_mask[2] * prefactor * dBdy
        du[6] = grad_mask[3] * prefactor * dBdz
    end

    return eom!
end

# ── ODE leg: slit → aperture ─────────────────────────────────────────────────
function propagate_SG(Iw, r_in, v_in, cal::SGCalibration;
                      μ_over_m,
                      k,
                      θ0,
                      y_field_start = DEFAULT_y_FurnaceToSlit,
                      y_field_end   = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToAperture,
                      y_SG_center   = DEFAULT_center_of_SG_magnet,
                      grad_mask     = (0.0, 0.0, 1.0))

    # resolve calibration once per particle — never inside the ODE
    Iw_eff = cal.I_eff_B(Float64(Iw))
    S      = cal.grad_scale(Float64(Iw))

    # time stamps for saveat (constant vy assumption — valid for thermal beams)
    vy          = v_in[2]
    t_in        = y_field_start                                              / vy
    t_out       = y_field_end                                                / vy
    t_SG_in     = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)            / vy
    t_SG_center = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + 0.5 * DEFAULT_y_SG) / vy
    t_SG_out    = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG)        / vy

    eom! = make_eom(Iw_eff, S,
                    Float64(μ_over_m), Float64(k), Float64(θ0),
                    t_in, Float64(y_SG_center),
                    NTuple{3,Float64}(grad_mask))

    u0    = Float64[r_in[1], r_in[2], r_in[3], v_in[1], v_in[2], v_in[3]]
    tspan = (t_in, 1.02 * t_out)

    cb   = ContinuousCallback((u, t, i) -> u[2] - y_field_end, terminate!)
    prob = ODEProblem(eom!, u0, tspan)
    sol  = solve(prob, Vern7(),
                 callback = cb,
                 abstol   = 1e-14,
                 reltol   = 1e-12,
                 saveat   = [t_in, t_SG_in, t_SG_center, t_SG_out, t_out])

    u_end = sol.u[end]
    return sol, u_end[1:3], u_end[4:6]
end

# ── Full pipeline: oven → slit → field region → aperture → screen ────────────
function full_trajectory(Iw, r0, v0, cal::SGCalibration;
                         μ_over_m,
                         k,
                         θ0,
                         y_slit      = DEFAULT_y_FurnaceToSlit,
                         y_aperture  = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToAperture,
                         y_screen    = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToScreen,
                         y_SG_center = DEFAULT_center_of_SG_magnet,
                         R_aperture  = DEFAULT_c_aperture,
                         R_screen    = DEFAULT_R_tube,
                         grad_mask   = (0.0, 0.0, 1.0))

    # 1. free flight: oven → slit
    r_slit, v_slit = free_flight(r0, v0, y_slit)

    # 2. ODE: slit → aperture (fringe fields included)
    sol, r_ap, v_ap = propagate_SG(Iw, r_slit, v_slit, cal;
                                    μ_over_m      = μ_over_m,
                                    k             = k,
                                    θ0            = θ0,
                                    y_field_start = y_slit,
                                    y_field_end   = y_aperture,
                                    y_SG_center   = y_SG_center,
                                    grad_mask     = grad_mask)

    # 3. aperture check: transverse radius at y_aperture
    pass_aper = r_ap[1]^2 + r_ap[3]^2 ≤ R_aperture^2

    # 4. free flight: aperture → screen
    r_screen, v_screen = free_flight(r_ap, v_ap, y_screen)

    # 5. screen check: transverse radius at detector
    pass_screen = r_screen[1]^2 + r_screen[3]^2 ≤ R_screen^2

    return (;
        # r_slit      = r_slit,
        # r_aperture  = r_ap,      v_aperture  = v_ap,
        r_screen    = r_screen,  v_screen    = v_screen,
        pass        = pass_aper && pass_screen,
        sol_magnet  = sol)
end

# ── Ensemble run ──────────────────────────────────────────────────────────────

function run_ensemble(Iw, data, cal::SGCalibration;
                      μ_over_m,
                      k,
                      θ0_col      = 7,
                      y_slit      = DEFAULT_y_FurnaceToSlit,
                      y_aperture  = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToAperture,
                      y_screen    = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToScreen,
                      y_SG_center = DEFAULT_center_of_SG_magnet,
                      R_aperture  = DEFAULT_c_aperture,
                      R_screen    = DEFAULT_R_tube,
                      grad_mask   = (0.0, 0.0, 1.0))

    N           = size(data, 1)
    # columns: x y z vx vy vz pass  (1=passed, 0=blocked)
    screen_data = Matrix{Float64}(undef, N, 7)
    progress    = Progress(N; desc="Running ensemble... ", showspeed=true)

    t_elapsed = @elapsed Threads.@threads for ii in 1:N
        r0 = @view data[ii, 1:3]
        v0 = @view data[ii, 4:6]
        θ0 = data[ii, θ0_col]

        result = full_trajectory(Iw, r0, v0, cal;
                                 μ_over_m    = μ_over_m,
                                 k           = k,
                                 θ0          = θ0,
                                 y_slit      = y_slit,
                                 y_aperture  = y_aperture,
                                 y_screen    = y_screen,
                                 y_SG_center = y_SG_center,
                                 R_aperture  = R_aperture,
                                 R_screen    = R_screen,
                                 grad_mask   = grad_mask)

        screen_data[ii, 1:3] = result.r_screen
        screen_data[ii, 4:6] = result.v_screen
        screen_data[ii, 7]   = Float64(result.pass)

        next!(progress)
    end

    n_pass = Int(sum(screen_data[:, 7]))
    @info "ENSEMBLE COMPLETED" particles=N passed_pct=round(100*n_pass/N, digits=1) threads=Threads.nthreads() time_s=round(t_elapsed, digits=2)

    return screen_data
end

function run_ensemble2(Iw, data, cal::SGCalibration;
                      μ_over_m,
                      k,
                      y_slit      = DEFAULT_y_FurnaceToSlit,
                      y_aperture  = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToAperture,
                      y_screen    = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG + DEFAULT_y_SG + DEFAULT_y_SGToScreen,
                      y_SG_center = DEFAULT_center_of_SG_magnet,
                      R_aperture  = DEFAULT_c_aperture,
                      R_screen    = DEFAULT_R_tube,
                      grad_mask   = (0.0, 0.0, 1.0))

    N = size(data, 1)

    # ── resolve all scalars once — nothing below should call cal or convert types ──
    Iw_eff    = cal.I_eff_B(Float64(Iw))      # effective current for B_total
    S         = cal.grad_scale(Float64(Iw))    # gradient scale factor
    gmask     = NTuple{3,Float64}(grad_mask)   # force component selector
    μ_m       = Float64(μ_over_m)
    kf        = Float64(k)
    yc        = Float64(y_SG_center)
    R_ap_sq   = R_aperture^2                   # squared radii avoid sqrt in checks
    R_sc_sq   = R_screen^2
    Δy_screen = y_screen - y_aperture          # fixed free-flight after aperture

    # output: [x y z vx vy vz pass] at screen, one row per particle
    screen_data = Matrix{Float64}(undef, N, 7)

    # ── template ODEProblem built from particle 1 ─────────────────────────────
    # EnsembleProblem requires a template; prob_func replaces all fields via
    # remake() so the template values never appear in the actual integration
    prob_template = let r0 = @view(data[1, 1:3]), v0 = @view(data[1, 4:6])
        vy   = v0[2]
        t_in = y_slit / vy
        Δt   = (y_slit - r0[2]) / vy
        u0   = Float64[r0[1] + v0[1]*Δt, y_slit, r0[3] + v0[3]*Δt, v0[1], vy, v0[3]]
        eom! = make_eom(Iw_eff, S, μ_m, kf, Float64(data[1,7]), t_in, yc, gmask)
        ODEProblem(eom!, u0, (t_in, 1.01 * y_aperture/vy))
    end

    # DifferentialEquations.jl API (SciMLBase 2.x):
    # prob_func signature : (prob, context)  — particle index = context.sim_id
    # affect! index access: integrator.p     — set via remake(..., p=ii)
    # output_func        : (sol, context)    — return (nothing, false) to discard sol

    # ── prob_func: called once per particle to specialise the template ────────
    # θ0 varies per particle so eom! must be rebuilt each time;
    # everything else (Iw_eff, S, gmask, ...) is captured from outer scope
    function prob_func(prob, context)
        ii   = context.sim_id
        r0   = @view data[ii, 1:3]
        v0   = @view data[ii, 4:6]
        θ0   = Float64(data[ii, 7])
        vy   = v0[2]
        t_in = y_slit / vy
        Δt   = (y_slit - r0[2]) / vy

        u0   = Float64[r0[1] + v0[1]*Δt, y_slit, r0[3] + v0[3]*Δt, v0[1], vy, v0[3]]
        eom! = make_eom(Iw_eff, S, μ_m, kf, θ0, t_in, yc, gmask)

        remake(prob; f=eom!, u0=u0, tspan=(t_in, 1.01*y_aperture/vy), p=ii)
    end

    # ── affect!: fires at y = y_aperture (ContinuousCallback crossing) ───────
    # computes free-flight to screen, performs both geometry checks,
    # writes directly to screen_data[ii,:] — no allocations, no return value
    function affect!(integrator)
        u  = integrator.u
        ii = integrator.p          # particle index passed via p

        vx_ap, vy_ap, vz_ap = u[4], u[5], u[6]
        Δt   = Δy_screen / vy_ap        # time from aperture to screen

        x_sc = u[1] + vx_ap * Δt
        z_sc = u[3] + vz_ap * Δt

        screen_data[ii, 1] = x_sc
        screen_data[ii, 2] = y_screen
        screen_data[ii, 3] = z_sc
        screen_data[ii, 4] = vx_ap
        screen_data[ii, 5] = vy_ap
        screen_data[ii, 6] = vz_ap
        # pass = 1 only if particle clears both aperture and screen radii
        screen_data[ii, 7] = Float64(
            (u[1]^2 + u[3]^2 ≤ R_ap_sq) && (x_sc^2 + z_sc^2 ≤ R_sc_sq)
        )

        terminate!(integrator)
    end

    # stop integration exactly when particle crosses y_aperture
    cb = ContinuousCallback((u, t, i) -> u[2] - y_aperture, affect!)

    # output_func discards the solution object immediately after each trajectory
    # — critical for N~10^7: without this every sol would accumulate in memory
    output_func(sol, context) = (nothing, false)

    ensemble_prob = EnsembleProblem(prob_template;
                                    prob_func   = prob_func,
                                    output_func = output_func)

    # EnsembleThreads: particles distributed across Julia threads (set via
    # JULIA_NUM_THREADS or --threads auto at startup)
    t_elapsed = @elapsed solve(ensemble_prob, Vern7(), EnsembleThreads();
                               trajectories   = N,
                               callback       = cb,
                               abstol         = 1e-14,
                               reltol         = 1e-12,
                               save_everystep = false,   # no internal steps saved
                               save_end       = false)   # affect! handles the exit state

    n_pass = sum(screen_data[:, 7])
    @info "ENSEMBLE COMPLETED" particles=N passed_pct=round(100*n_pass/N, digits=1) threads=Threads.nthreads() time_s=round(t_elapsed, digits=2)

    return screen_data
end







