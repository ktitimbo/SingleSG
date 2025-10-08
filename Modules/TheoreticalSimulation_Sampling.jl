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

function generate_CQDinitial_conditions(No::Integer, alive::AbstractMatrix{T}, rng::AbstractRNG;
                                        mode::Symbol = :partition) where {T<:Real}
    @assert No > 0 "No must be > 0"
    @assert No == size(alive,1) "Total number of particles $No"
    @assert size(alive,2) ≥ 6 "alive must have at least 6 columns (x0,y0,z0,v0x,v0y,v0z)"

    # one-liner to draw θ with your distribution, in the target element type T
    @inline sample_theta() = T(2asin(sqrt(rand(rng))))

    if mode === :partition
        # Two-pass: count UP with a cloned RNG → allocate exact sizes → fill.
        @assert hasmethod(copy, Tuple{typeof(rng)}) "RNG must support copy() for two-pass mode"
        rng1 = copy(rng)
        n_up = 0
        @inbounds for _ in 1:No
            θe = T(2asin(sqrt(rand(rng1))))
            θn = T(2asin(sqrt(rand(rng1))))
            n_up += (θe < θn)
        end
        n_down = No - n_up

        up_batch   = Matrix{T}(undef, n_up,   8)
        down_batch = Matrix{T}(undef, n_down, 8)

        iu = 0; id = 0
        @inbounds @views for i in 1:No
            θe = sample_theta()
            θn = sample_theta()
            if θe < θn
                iu += 1
                up_batch[iu, 1:6] = alive[i, 1:6]
                up_batch[iu, 7]   = θe
                up_batch[iu, 8]   = θn
            else
                id += 1
                down_batch[id, 1:6] = alive[i, 1:6]
                down_batch[id, 7]   = θe
                down_batch[id, 8]   = θn
            end
        end
        return up_batch, down_batch

    elseif mode === :balanced
        # Single pass: preallocate No×8 for both; write angles as we generate.
        up_batch   = Matrix{T}(undef, No, 8)
        down_batch = Matrix{T}(undef, No, 8)
        nup = 0; ndn = 0
        @inbounds while (nup < No) || (ndn < No)
            θe = sample_theta()
            θn = sample_theta()
            if (θe < θn) && (nup < No)
                nup += 1
                up_batch[nup, 7] = θe
                up_batch[nup, 8] = θn
            elseif (θe > θn) && (ndn < No)
                ndn += 1
                down_batch[ndn, 7] = θe
                down_batch[ndn, 8] = θn
            end
        end
        # copy kinematics once for both batches
        @inbounds @views begin
            up_batch[:,   1:6] .= alive[:, 1:6]
            down_batch[:, 1:6] .= alive[:, 1:6]
        end
        return up_batch, down_batch

    else
        error("Unknown mode=$mode. Use :partition or :balanced.")
    end
end