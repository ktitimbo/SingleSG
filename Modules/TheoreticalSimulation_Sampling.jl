# Generate samples post-filtering by the slit
"""
    _generate_samples_serial(No::Int, rng, p::EffusionParams; v_pdf::Symbol = :v3) -> Matrix{Float64}

Generate `No` accepted (“alive”) particles that pass the entrance slit by
rejection sampling initial positions and velocities, then checking the ray
intersection at the slit plane.

Each accepted row is `[x0, y0, z0, v0x, v0y, v0z]` with `y0 = 0.0` (SI units).

# Arguments
- `No::Int`: Number of accepted particles to return (must be > 0 and equal to `size(alive, 1)` conceptually).
- `rng::AbstractRNG`: Random-number generator used for all sampling.
- `p::EffusionParams`: Parameters forwarded to the velocity samplers.

# Keywords
- `v_pdf::Symbol = :v3`: Velocity PDF selector.
  - `:v3` → `AtomicBeamVelocity_v3(rng, p)`
  - `:v2` → `AtomicBeamVelocity_v2(rng, p)`
  - Any other value triggers a warning and falls back to `SVector(0, 800, 0)`.

# Algorithm
1. Sample transverse position uniformly over the furnace rectangle:
   - `x0 ∈ [-default_x_furnace/2, +default_x_furnace/2]`
   - `z0 ∈ [-default_z_furnace/2, +default_z_furnace/2]`
2. Sample velocity `v = (v0x, v0y, v0z)` using the chosen PDF.
3. Reject if `|v0y| ≤ 1e-18` to avoid singular drift.
4. Propagate to the slit at `y = default_y_FurnaceToSlit` (straight-line drift):
   - `x_at_slit = x0 + default_y_FurnaceToSlit * v0x / v0y`
   - `z_at_slit = z0 + default_y_FurnaceToSlit * v0z / v0y`
5. Accept if the ray intersects the rectangular aperture:
   - `|x_at_slit| ≤ default_x_slit/2` and `|z_at_slit| ≤ default_z_slit/2`
6. Repeat until `No` rows are accepted; store as `Float64`.

# Returns
- `Matrix{Float64}` of size `No × 6`, columns:
  1. `x0`, 2. `y0 (= 0.0)`, 3. `z0`, 4. `v0x`, 5. `v0y`, 6. `v0z`.

# Globals expected in scope
- `default_x_furnace`, `default_z_furnace`,
- `default_y_FurnaceToSlit`,
- `default_x_slit`, `default_z_slit`.

# Notes
- Prints total iteration count and timing due to `@time`.
- Rejection efficiency depends on geometry and the chosen velocity PDF.
"""
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

"""
    _generate_samples_multithreaded(No::Int, base_seed::Int, p::EffusionParams;
                                    v_pdf::Symbol = :v3) -> Matrix{Float64}

Generate `No` accepted (“alive”) particles **in parallel** by rejection sampling
initial positions and velocities and checking intersection with the entrance slit.

Each accepted row is `[x0, y0, z0, v0x, v0y, v0z]` with `y0 = 0.0` (SI units).

# Arguments
- `No::Int`: Number of particles to return (must be > 0).
- `base_seed::Int`: Base seed for per-thread RNGs. Reproducibility depends on
  thread count and seed (see *Parallelism & reproducibility*).
- `p::EffusionParams`: Parameters forwarded to the velocity sampler(s).

# Keywords
- `v_pdf::Symbol = :v3`: Velocity PDF selector.
  - `:v3` → `AtomicBeamVelocity_v3(rng, p)`
  - `:v2` → `AtomicBeamVelocity_v2(rng, p)`
  - Any other value triggers a warning and falls back to `SVector(0, 800, 0)`.

# Returns
- `Matrix{Float64}` of size `No × 6`, columns:
  1. `x0`, 2. `y0 (= 0.0)`, 3. `z0`, 4. `v0x`, 5. `v0y`, 6. `v0z`.

# Algorithm
- Spawns a `Threads.@threads` loop where each thread:
  1. Initializes a `TaskLocalRNG()` seeded with `hash((base_seed, thread_id))`.
  2. Samples `x0, z0` uniformly in the furnace rectangle:
     - `x0 ∈ [-default_x_furnace/2, +default_x_furnace/2]`
     - `z0 ∈ [-default_z_furnace/2, +default_z_furnace/2]`
  3. Samples velocity `v = (v0x, v0y, v0z)` from the chosen PDF.
  4. Rejects near-zero `v0y` (|v0y| ≤ 1e-18) to avoid singular drift.
  5. Propagates to the slit at `y = default_y_FurnaceToSlit`:
     - `x_at_slit = x0 + default_y_FurnaceToSlit * v0x / v0y`
     - `z_at_slit = z0 + default_y_FurnaceToSlit * v0z / v0y`
  6. Accepts if `|x_at_slit| ≤ default_x_slit/2` **and** `|z_at_slit| ≤ default_z_slit/2`.
  7. On accept, obtains a unique write index `idx = Threads.atomic_add!(sample_count, 1)`;
     if `idx ≤ No`, writes the row; otherwise breaks out.

Additionally tracks `iteration_count` (total trials) via an atomic counter.
The function prints timing for the threaded region and total iterations.

# Parallelism & reproducibility
- Each thread owns its RNG (`TaskLocalRNG()`), seeded from `base_seed` and `thread_id`.
- The **set of returned samples** is reproducible given the same `base_seed`
  **and the same `Threads.nthreads()`**. Row order is inherently non-deterministic
  due to concurrent acceptance and atomic indexing.
- If you require fully reproducible row order, sort after generation or use the
  serial sampler.

# Globals expected in scope
- Geometry: `default_x_furnace`, `default_z_furnace`,
  `default_y_FurnaceToSlit`, `default_x_slit`, `default_z_slit`.
- Velocity samplers: `AtomicBeamVelocity_v3`, `AtomicBeamVelocity_v2`.
- `SVector` (from StaticArrays) for fallback velocity.

# Notes
- Uses rectangular aperture half-widths `hx = default_x_slit/2`, `hz = default_z_slit/2`.
- Prints timing due to `@time` and reports total trial count at the end.
- You can swap `TaskLocalRNG()` for a different per-thread RNG if desired
  (e.g., `MersenneTwister(hash((base_seed, thread_id)))`).

# Example
julia
Threads.nthreads()  # ensure desired thread count is set before starting Julia
alive = _generate_samples_multithreaded(100_000, 12345, effusion_params; v_pdf = :v3)
size(alive)  # (100000, 6)
"""
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

"""
    generate_samples(No::Int, p::EffusionParams;
                     v_pdf::Symbol = :v3,
                     rng = Random.default_rng(),
                     multithreaded::Bool = false,
                     base_seed::Int = 1234) -> Matrix{Float64}

High-level entry point to generate `No` accepted (“alive”) particle states that pass
the entrance slit. Dispatches to a serial or multithreaded sampler.

Each returned row is `[x0, y0, z0, v0x, v0y, v0z]` with `y0 = 0.0` (SI units).

# Arguments
- `No::Int`: Number of accepted particles to return (must be > 0).
- `p::EffusionParams`: Parameters forwarded to the velocity sampler(s).

# Keywords
- `v_pdf::Symbol = :v3`: Velocity PDF selector (`:v3` → `AtomicBeamVelocity_v3`,
  `:v2` → `AtomicBeamVelocity_v2`; other values warn and fall back).
- `rng = Random.default_rng()`: RNG used **only in serial mode**.
- `multithreaded::Bool = false`: If `true`, uses the threaded sampler; otherwise serial.
- `base_seed::Int = 1234`: Per-thread seeding base used **only in multithreaded mode**.

# Behavior
- `multithreaded = false` → calls `_generate_samples_serial(No, rng, p; v_pdf)`.
- `multithreaded = true`  → calls `_generate_samples_multithreaded(No, base_seed, p; v_pdf)`.

# Returns
- `Matrix{Float64}` of size `No × 6` with accepted initial conditions.

# Reproducibility
- Serial mode reproducibility is controlled by `rng`.
- Multithreaded mode uses per-thread RNGs seeded from `base_seed`; the **set** of
  samples is reproducible given the same `base_seed` and `Threads.nthreads()`,
  but row order may vary due to concurrency.

# Example
julia
alive_serial = generate_samples(50_000, effusion_params; v_pdf=:v3, rng=MersenneTwister(42))
alive_mt     = generate_samples(50_000, effusion_params; multithreaded=true, base_seed=1234)
"""
function generate_samples(No::Int, p::EffusionParams; v_pdf::Symbol =:v3, rng = Random.default_rng(), multithreaded::Bool = false, base_seed::Int = 1234)
    if multithreaded
        return _generate_samples_multithreaded(No, base_seed, p; v_pdf=v_pdf)
    else
        return _generate_samples_serial(No, rng, p; v_pdf=v_pdf)
    end
end

"""
    generate_CQDinitial_conditions(No, alive, rng; mode=:partition)
        -> (up_batch::Matrix{T}, down_batch::Matrix{T})

Generate two sets of initial conditions by drawing per-particle angles and
classifying each draw as “up” (`θe < θn`) or “down” (`θe > θn`).

Each output matrix has 8 columns:
1–6: copied from `alive[:, 1:6]` (e.g. `x0,y0,z0,v0x,v0y,v0z`),  
7–8: the drawn angles `(θe, θn)`, where `θ = 2asin(sqrt(u))` with `u ~ U(0,1)`.

# Arguments
- `No::Integer`: number of particles (must equal `size(alive,1)`).
- `alive::AbstractMatrix{T}`: at least 6 columns with the base kinematic state;
  element type `T<:Real` is preserved in the outputs.
- `rng::AbstractRNG`: random-number generator used for the draws.
- `mode::Symbol = :partition`:
  - `:partition` → **two-pass exact split.**
    - Clones the RNG (`copy(rng)`) to count how many “up” samples will occur,
      allocates exact sizes (`n_up×8` and `n_down×8`), then fills using the
      original RNG so counts match.
  - `:balanced` → **single-pass balanced batches.**
    - Generates until both “up” and “down” reach size `No`, returning two
      `No×8` matrices (both reuse `alive[:,1:6]`).

# Returns
A tuple `(up_batch, down_batch)` of `Matrix{T}`.  
For `:partition`, sizes are `n_up×8` and `n_down×8` with `n_up + n_down = No`.  
For `:balanced`, both are `No×8`.

# Notes
- Preconditions: `No > 0`, `size(alive,1) == No`, and `size(alive,2) ≥ 6`.
- `:partition` requires `copy(rng)` to exist so the counting pass doesn’t
  advance the RNG used for filling. If your RNG can’t be copied, consider a
  pre-sampling approach instead.
- Equality `θe == θn` is effectively measure-zero with continuous RNGs and is
  not handled specially.
- All work is in-place friendly and uses `@inbounds`/`@views` in hot loops.

# Example
julia
using StableRNGs
rng = StableRNG(42)
No = size(alive, 1)

up, down = generate_CQDinitial_conditions(No, alive, rng; mode=:partition)
# or
upB, downB = generate_CQDinitial_conditions(No, alive, rng; mode=:balanced)
"""
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