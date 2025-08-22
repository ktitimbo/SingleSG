# Generate samples post-filtering by the slit
function _generate_samples_serial(No::Int, rng, p::EffusionParams; v_pdf::Symbol=:v3)
    @assert No > 0
    alive = Matrix{Float64}(undef, No, 6)
    iteration_count = 0
    count = 0

    # precompute a few constants
    hx = default_x_slit/2
    hz = default_z_slit/2
    epsvy = 1e-18

    @time while count < No
        iteration_count += 1

        # initial transverse position (uniform over furnace rectangle)
        x0 = default_x_furnace * (rand(rng) - 0.5)
        z0 = default_z_furnace * (rand(rng) - 0.5)

        if v_pdf === :v3
            v = AtomicBeamVelocity_v3(rng,p)
        elseif v_pdf === :v2
            v = AtomicBeamVelocity_v2(rng,p)
        else
            @warn "No Velocity PDF chosen, got $v_pdf"
            v = SVector{3,Float64}(0,800,0)
        end

        v0_x, v0_y, v0_z = v

        # avoid near-zero v_y
        if abs(v0_y) ≤ epsvy
            continue
        end

        x_at_slit = x0 + default_y_FurnaceToSlit * v0_x / v0_y
        z_at_slit = z0 + default_y_FurnaceToSlit * v0_z / v0_y

        if (abs(x_at_slit) <= hx) & (abs(z_at_slit) <= hz)
            count += 1
            @inbounds alive[count,:] =  [x0, 0.0, z0, v0_x, v0_y, v0_z]
        end
    end

    println("Total iterations: ", iteration_count)
    return alive
end

function _generate_samples_multithreaded(No::Int, base_seed::Int, p::EffusionParams; v_pdf::Symbol = :v3)
    alive = Matrix{Float64}(undef, No, 6)

    sample_count = Threads.Atomic{Int}(0)
    iteration_count = Threads.Atomic{Int}(0)

    # Precomputed constants
    hx = default_x_slit/2
    hz = default_z_slit/2
    epsvy = 1e-18

    @time Threads.@threads for thread_id in 1:Threads.nthreads()
        rng0 = TaskLocalRNG()
        Random.seed!(rng0, hash((base_seed, thread_id)))
        # rng0 = MersenneTwister(hash((base_seed, thread_id)))   

        while true
            Threads.atomic_add!(iteration_count, 1)

            x0 = default_x_furnace * (rand(rng0) - 0.5)
            z0 = default_z_furnace * (rand(rng0) - 0.5)

            # Velocity sample (zero-alloc SVector)
            if v_pdf === :v3
                v = AtomicBeamVelocity_v3(rng0,p)
            elseif v_pdf === :v2
                v = AtomicBeamVelocity_v2(rng0,p)
            else
                @warn "No Velocity PDF chosen, got $v_pdf"
                v = SVector{3,Float64}(0,800,0)
            end
            v0_x, v0_y, v0_z = v

            # Avoid divide-by-zero / huge times
            if abs(v0_y) ≤ epsvy
                continue
            end

            x_at_slit = x0 + default_y_FurnaceToSlit * v0_x / v0_y
            z_at_slit = z0 + default_y_FurnaceToSlit * v0_z / v0_y

            if -hx ≤ x_at_slit ≤ hx && -hz ≤ z_at_slit ≤ hz
                idx = Threads.atomic_add!(sample_count, 1)
                if idx <= No
                    @inbounds alive[idx, :] = [x0, 0.0, z0, v0_x, v0_y, v0_z]
                else
                    break
                end
            end

        end
    end

    println("Total iterations: ", iteration_count[])
    return alive
end

function generate_samples(No::Int, p::EffusionParams; v_pdf::Symbol =:v3, rng = Random.default_rng(), multithreaded::Bool = false, base_seed::Int = 1234)
    if multithreaded
        return _generate_samples_multithreaded(No, base_seed, p; v_pdf=v_pdf)
    else
        return _generate_samples_serial(No, rng, p; v_pdf=v_pdf)
    end
end

function _generate_matched_pairs(No::Integer, rng; mode::Symbol = :total) # deprecated : unused
    @assert No > 0
    θes_up = Float64[]; θns_up = Float64[]
    θes_dn = Float64[]; θns_dn = Float64[]

    if mode === :total
        sizehint!(θes_up, No ÷ 2); sizehint!(θns_up, No ÷ 2)
        sizehint!(θes_dn, No ÷ 2); sizehint!(θns_dn, No ÷ 2)

        kept = 0
        while kept < No
            θe = 2asin(sqrt(rand(rng))); θn = 2asin(sqrt(rand(rng)))
            if θe < θn
                push!(θes_up, θe); push!(θns_up, θn); kept += 1
            elseif θe > θn
                push!(θes_dn, θe); push!(θns_dn, θn); kept += 1
            end
        end

    elseif mode === :bucket
        sizehint!(θes_up, No); sizehint!(θns_up, No)
        sizehint!(θes_dn, No); sizehint!(θns_dn, No)

        nup = 0; ndn = 0
        while (nup < No) || (ndn < No)
            θe = 2asin(sqrt(rand(rng))); θn = 2asin(sqrt(rand(rng)))
            if (θe < θn) && (nup < No)
                push!(θes_up, θe); push!(θns_up, θn); nup += 1
            elseif (θe > θn) && (ndn < No)
                push!(θes_dn, θe); push!(θns_dn, θn); ndn += 1
            end
        end

    else
        error("Unknown mode=$mode. Use :total or :bucket.")
    end

    return θes_up, θns_up, θes_dn, θns_dn
end


function _build_init_conditions(
    alive::AbstractMatrix{T},
    UPθe::AbstractVector{T}, UPθn::AbstractVector{T},
    DOWNθe::AbstractVector{T}, DOWNθn::AbstractVector{T};
    mode::Symbol = :total
) where {T<:Real} # deprecated :  unused

    No = size(alive, 1)
    @assert length(UPθe)   == length(UPθn)   "UP θe/θn lengths must match"
    @assert length(DOWNθe) == length(DOWNθn) "DOWN θe/θn lengths must match"

    if mode === :total
        n_up = length(UPθe)
        n_dn = length(DOWNθe)
        @assert n_up + n_dn == No "In :total, n_up + n_dn must equal size(alive,1)"

        pairsUP   = Matrix{T}(undef, n_up, 8)
        pairsDOWN = Matrix{T}(undef, n_dn, 8)

        @inbounds @views begin
            # UP block: rows 1:n_up from `alive`
            for i in 1:n_up
                pairsUP[i, 1:6] = alive[i, 1:6]
                pairsUP[i, 7]   = UPθe[i]
                pairsUP[i, 8]   = UPθn[i]
            end
            # DOWN block: rows (n_up+1):No from `alive`
            for j in 1:n_dn
                i_alive = n_up + j
                pairsDOWN[j, 1:6] = alive[i_alive, 1:6]
                pairsDOWN[j, 7]   = DOWNθe[j]
                pairsDOWN[j, 8]   = DOWNθn[j]
            end
        end

        return pairsUP, pairsDOWN

    elseif mode === :bucket
        @assert length(UPθe) == No == length(DOWNθe) "In :bucket, each θ list must have length No"

        pairsUP   = Matrix{T}(undef, No, 8)
        pairsDOWN = Matrix{T}(undef, No, 8)

        @inbounds @views for i in 1:No
            # UP
            pairsUP[i, 1:6] = alive[i, 1:6]
            pairsUP[i, 7]   = UPθe[i]
            pairsUP[i, 8]   = UPθn[i]
            if add_label; pairsUP[i, 9] = one(T); end
            # DOWN
            pairsDOWN[i, 1:6] = alive[i, 1:6]
            pairsDOWN[i, 7]   = DOWNθe[i]
            pairsDOWN[i, 8]   = DOWNθn[i]
            if add_label; pairsDOWN[i, 9] = zero(T); end
        end

        return pairsUP, pairsDOWN

    else
        error("Unknown mode=$mode. Use :total or :bucket.")
    end
end


function build_initial_conditions(No::Integer, alive::AbstractMatrix{T}, rng::AbstractRNG; mode::Symbol = :total) where {T<:Real}
    @assert No > 0 "No must be > 0"
    @assert No == size(alive,1) "Total number of particles $No"

    if mode === :total
        # Two-pass: count UP with a cloned RNG → allocate exact sizes → fill.
        @assert hasmethod(copy, Tuple{typeof(rng)}) "RNG must support copy() for two-pass mode"
        rng1 = copy(rng)
        n_up = 0
        @inbounds for _ in 1:No
            θe = T(2asin(sqrt(rand(rng1))))
            θn = T(2asin(sqrt(rand(rng1))))
            n_up += (θe < θn)
        end
        n_dn = No - n_up

        UP   = Matrix{T}(undef, n_up, 8)
        DOWN = Matrix{T}(undef, n_dn, 8)

        iu = 0; id = 0
        @inbounds @views for i in 1:No
            θe = T(2asin(sqrt(rand(rng))))
            θn = T(2asin(sqrt(rand(rng))))
            if θe < θn
                iu += 1
                UP[iu, 1:6] = alive[i, 1:6]
                UP[iu, 7]   = θe
                UP[iu, 8]   = θn
            else
                id += 1
                DOWN[id, 1:6] = alive[i, 1:6]
                DOWN[id, 7]   = θe
                DOWN[id, 8]   = θn
            end
        end
        return UP, DOWN

    elseif mode === :bucket
        # --- Single pass: preallocate No×8 for both; write angles as we generate.
        UP   = Matrix{T}(undef, No, 8)
        DOWN = Matrix{T}(undef, No, 8)
        nup = 0; ndn = 0
        @inbounds while (nup < No) || (ndn < No)
            θe = T(2asin(sqrt(rand(rng))))
            θn = T(2asin(sqrt(rand(rng))))
            if (θe < θn) && (nup < No)
                nup += 1
                UP[nup, 7] = θe
                UP[nup, 8] = θn
            elseif (θe > θn) && (ndn < No)
                ndn += 1
                DOWN[ndn, 7] = θe
                DOWN[ndn, 8] = θn
            end
        end
        # now copy alive rows once
        @inbounds @views for i in 1:No
            UP[i,   1:6] = alive[i, 1:6]
            DOWN[i, 1:6] = alive[i, 1:6]
        end
        return UP, DOWN

    else
        error("Unknown mode=$mode. Use :total or :bucket.")
    end

end
