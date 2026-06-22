# ==============================================================================
# Particle sampling for the SG beamline simulation
#
# This file implements two layers:
#
#   1. Slit-acceptance sampling (rejection sampling)
#      _generate_samples_serial / _generate_samples_multithreaded / generate_samples
#      Draw initial (position, velocity) states from the furnace aperture and
#      the chosen velocity PDF, propagate them ballistically to the entrance
#      slit, and keep only those whose straight-line trajectory passes through
#      the slit aperture. This is standard geometric rejection sampling for an
#      effusive atomic beam source.
#
#   2. CQD branch assignment
#      generate_CQDinitial_conditions
#      For each accepted particle, draws a pair of Co-Quantum-Dynamics angles
#      (θe, θn) and classifies the particle into an "up" or "down" branch
#      depending on which angle is larger, splitting the population into the
#      two batches used downstream by the CQD trajectory integrator.
# ==============================================================================
 
 
# Generate samples post-filtering by the slit
"""
    _generate_samples_serial(No::Int, rng::AbstractRNG, p::EffusionParams;
                             v_pdf::Symbol = :v3,
                             max_trial_multiplier::Int = 10_000) -> Matrix{Float64}
 
Generate `No` accepted (“alive”) particles that pass the entrance slit by
rejection sampling initial positions and velocities, then checking the ray
intersection at the slit plane.
 
Each accepted row is `[x0, y0, z0, v0x, v0y, v0z]` with `y0 = 0.0` (SI units).
 
# Arguments
- `No::Int`: Number of accepted particles to return (must be > 0 and equal to `size(alive, 1)` conceptually).
- `rng::AbstractRNG`: Random-number generator used for all sampling.
- `p::EffusionParams`: Parameters forwarded to the velocity samplers.
 
# Keywords
- `v_pdf::Symbol = :v3`: Velocity PDF selector, resolved **once** before the
  sampling loop starts (not re-checked on every trial):
  - `:v3` → `AtomicBeamVelocity_v3(rng, p)`
  - `:v2` → `AtomicBeamVelocity_v2(rng, p)`
  - Any other value throws an `ArgumentError` immediately.
- `max_trial_multiplier::Int = 10_000`: Safety cap on total proposal trials,
  expressed as a multiple of `No`. If `iteration_count` exceeds
  `No * max_trial_multiplier` before `No` particles have been accepted, the
  function throws an error rather than looping indefinitely — this guards
  against a misconfigured geometry/PDF combination silently hanging.
 
# Algorithm
1. Sample transverse position uniformly over the furnace rectangle:
   - `x0 ∈ [-DEFAULT_x_furnace/2, +DEFAULT_x_furnace/2]`
   - `z0 ∈ [-DEFAULT_z_furnace/2, +DEFAULT_z_furnace/2]`
2. Sample velocity `v = (v0x, v0y, v0z)` using the chosen PDF.
3. Reject if `|v0y| ≤ 1e-18` to avoid singular drift.
4. Propagate to the slit at `y = DEFAULT_y_FurnaceToSlit` (straight-line drift):
   - `x_at_slit = x0 + DEFAULT_y_FurnaceToSlit * v0x / v0y`
   - `z_at_slit = z0 + DEFAULT_y_FurnaceToSlit * v0z / v0y`
5. Accept if the ray intersects the rectangular aperture:
   - `|x_at_slit| ≤ DEFAULT_x_slit/2` and `|z_at_slit| ≤ DEFAULT_z_slit/2`
6. Repeat until `No` rows are accepted; store as `Float64`.
 
# Returns
- `Matrix{Float64}` of size `No × 6`, columns:
  1. `x0`, 2. `y0 (= 0.0)`, 3. `z0`, 4. `v0x`, 5. `v0y`, 6. `v0z`.
 
# Globals expected in scope
- `DEFAULT_x_furnace`, `DEFAULT_z_furnace`,
- `DEFAULT_y_FurnaceToSlit`,
- `DEFAULT_x_slit`, `DEFAULT_z_slit`.
 
# Notes
- Displays a live `ProgressMeter` progress bar tracking accepted particles
  (not total trials, which is unbounded), and prints total iteration count,
  acceptance rate, and timing (via `@time`) on completion.
- Rejection efficiency depends on geometry and the chosen velocity PDF.
 
# Throws
- `ArgumentError` if `v_pdf` is not `:v3` or `:v2`.
- `ErrorException` if `No` particles are not accepted within
  `No * max_trial_multiplier` trials (likely indicates near-zero acceptance
  probability from a geometry/PDF misconfiguration).
"""
function _generate_samples_serial(No::Int, rng::AbstractRNG, p::EffusionParams;
                                   v_pdf::Symbol=:v3, max_trial_multiplier::Int=10_000)
    @assert No > 0
 
    # Output buffer: one row per accepted ("alive") particle, 6 kinematic
    # columns (x0, y0, z0, v0x, v0y, v0z). Preallocated at full size since the
    # exact number of accepted rows (`No`) is known ahead of time — the
    # rejection loop below simply runs until it has filled every row.
    alive = Matrix{Float64}(undef, No, 6)
 
    # ── Resolve the velocity sampler ONCE, before the loop ────────────────
    # Avoids re-checking `v_pdf` on every single trial (it never changes
    # during sampling), and fails fast on an invalid value instead of
    # silently falling back and re-warning on every iteration.
    sampler = if v_pdf === :v3
        AtomicBeamVelocity_v3
    elseif v_pdf === :v2
        AtomicBeamVelocity_v2
    else
        throw(ArgumentError("Unknown v_pdf=$v_pdf; use :v3 or :v2"))
    end
 
    iteration_count = 0   # total proposal trials, accepted + rejected
    count           = 0   # number of accepted ("alive") particles so far
    max_trials      = No * max_trial_multiplier   # safety cap, see docstring
 
    # Rectangular slit aperture half-widths, and a small guard threshold on
    # |v0y| used below to avoid a division blow-up when projecting the ray
    # onto the slit plane.
    hx    = DEFAULT_x_slit / 2
    hz    = DEFAULT_z_slit / 2
    epsvy = 1e-18
 
    # Progress is tracked against `count` (bounded by `No`), not
    # `iteration_count` (unbounded) — the bar fills smoothly as particles
    # are accepted, regardless of the underlying acceptance rate.
    prog = Progress(No; desc="Sampling (serial): ")
 
    @time while count < No
        iteration_count += 1
 
        # Bail out with a clear error rather than spinning forever if
        # acceptance is unreasonably low (e.g. a geometry/PDF misconfiguration).
        if iteration_count > max_trials
            error("_generate_samples_serial: exceeded $max_trials trials " *
                  "($count/$No accepted) — check geometry and v_pdf=$v_pdf for a near-zero acceptance rate.")
        end
 
        # ── Step 1: initial transverse position ──────────────────────────
        # Uniform over the furnace aperture rectangle, centered at the origin:
        #   x0 ∈ [-DEFAULT_x_furnace/2, +DEFAULT_x_furnace/2]
        #   z0 ∈ [-DEFAULT_z_furnace/2, +DEFAULT_z_furnace/2]
        x0 = DEFAULT_x_furnace * (rand(rng) - 0.5)
        z0 = DEFAULT_z_furnace * (rand(rng) - 0.5)
 
        # ── Step 2: initial velocity ───────────────────────────────────
        # `sampler` was resolved once above — no branch on `v_pdf` here.
        v0_x, v0_y, v0_z = sampler(rng, p)
 
        # ── Step 3: reject near-singular drift ────────────────────────
        # If v0y ≈ 0, the projection below (which divides by v0y) would blow
        # up; such particles are effectively moving parallel to the slit
        # plane and never cross it in finite time, so simply resample.
        if abs(v0_y) ≤ epsvy
            continue
        end
 
        # ── Step 4: ballistic propagation to the slit plane ──────────
        # No forces act between the furnace and the slit, so the trajectory
        # is a straight line. By similar triangles, the transverse position
        # at y = DEFAULT_y_FurnaceToSlit is x(t) = x0 + v0x·t with
        # t = DEFAULT_y_FurnaceToSlit / v0y (and likewise for z).
        x_at_slit = x0 + DEFAULT_y_FurnaceToSlit * v0_x / v0_y
        z_at_slit = z0 + DEFAULT_y_FurnaceToSlit * v0_z / v0_y
 
        # ── Step 5: accept/reject against the rectangular slit aperture ──
        if (abs(x_at_slit) <= hx) && (abs(z_at_slit) <= hz)
            count += 1
            # Write fields directly instead of building a temporary Vector
            # and copying it in — avoids one small heap allocation per
            # accepted particle, which adds up across millions of samples.
            @inbounds begin
                alive[count, 1] = x0
                alive[count, 2] = 0.0
                alive[count, 3] = z0
                alive[count, 4] = v0_x
                alive[count, 5] = v0_y
                alive[count, 6] = v0_z
            end
            next!(prog)
        end
    end
 
    finish!(prog)
    acceptance_rate = No / iteration_count
    println("Total iterations: ", iteration_count, "  (acceptance rate: ", round(100*acceptance_rate, digits=3), "%)")
    return alive
end

"""
    _generate_samples_multithreaded(No::Int, base_seed::Int, p::EffusionParams;
                                    v_pdf::Symbol = :v3,
                                    max_trial_multiplier::Int = 10_000) -> Matrix{Float64}
 
Generate `No` accepted (“alive”) particles **in parallel** by rejection sampling
initial positions and velocities and checking intersection with the entrance slit.
 
Each accepted row is `[x0, y0, z0, v0x, v0y, v0z]` with `y0 = 0.0` (SI units).
 
# Arguments
- `No::Int`: Number of particles to return (must be > 0).
- `base_seed::Int`: Base seed for per-thread RNGs. Reproducibility depends on
  thread count and seed (see *Parallelism & reproducibility*).
- `p::EffusionParams`: Parameters forwarded to the velocity sampler(s).
 
# Keywords
- `v_pdf::Symbol = :v3`: Velocity PDF selector, resolved **once** before any
  thread starts sampling (not re-checked on every trial):
  - `:v3` → `AtomicBeamVelocity_v3(rng, p)`
  - `:v2` → `AtomicBeamVelocity_v2(rng, p)`
  - Any other value throws an `ArgumentError` immediately.
- `max_trial_multiplier::Int = 10_000`: Safety cap on total proposal trials
  (summed across all threads), expressed as a multiple of `No`. If the shared
  `iteration_count` exceeds `No * max_trial_multiplier` before `No` particles
  have been accepted, the function throws an error rather than looping
  indefinitely — same safeguard as in [`_generate_samples_serial`](@ref).
 
# Returns
- `Matrix{Float64}` of size `No × 6`, columns:
  1. `x0`, 2. `y0 (= 0.0)`, 3. `z0`, 4. `v0x`, 5. `v0y`, 6. `v0z`.
 
# Algorithm
- Spawns a `Threads.@threads` loop where each thread:
  1. Initializes a `TaskLocalRNG()` seeded with `hash((base_seed, thread_id))`.
  2. Samples `x0, z0` uniformly in the furnace rectangle:
     - `x0 ∈ [-DEFAULT_x_furnace/2, +DEFAULT_x_furnace/2]`
     - `z0 ∈ [-DEFAULT_z_furnace/2, +DEFAULT_z_furnace/2]`
  3. Samples velocity `v = (v0x, v0y, v0z)` from the (pre-resolved) chosen PDF.
  4. Rejects near-zero `v0y` (|v0y| ≤ 1e-18) to avoid singular drift.
  5. Propagates to the slit at `y = DEFAULT_y_FurnaceToSlit`:
     - `x_at_slit = x0 + DEFAULT_y_FurnaceToSlit * v0x / v0y`
     - `z_at_slit = z0 + DEFAULT_y_FurnaceToSlit * v0z / v0y`
  6. Accepts if `|x_at_slit| ≤ DEFAULT_x_slit/2` **and** `|z_at_slit| ≤ DEFAULT_z_slit/2`.
  7. On accept, obtains a unique 1-based write index
     `idx = Threads.atomic_add!(sample_count, 1) + 1` — the `+1` is required
     because `atomic_add!` returns the value of the counter *before* the
     addition, not after; if `idx ≤ No`, writes the row, otherwise breaks out.
 
Additionally tracks `iteration_count` (total trials) via an atomic counter.
The function prints timing for the threaded region, total iterations, and
the overall acceptance rate.
 
# Parallelism & reproducibility
- Each thread owns its RNG (`TaskLocalRNG()`), seeded from `base_seed` and `thread_id`.
- The **set of returned samples** is reproducible given the same `base_seed`
  **and the same `Threads.nthreads()`**. Row order is inherently non-deterministic
  due to concurrent acceptance and atomic indexing.
- If you require fully reproducible row order, sort after generation or use the
  serial sampler.
 
# Globals expected in scope
- Geometry: `DEFAULT_x_furnace`, `DEFAULT_z_furnace`,
  `DEFAULT_y_FurnaceToSlit`, `DEFAULT_x_slit`, `DEFAULT_z_slit`.
- Velocity samplers: `AtomicBeamVelocity_v3`, `AtomicBeamVelocity_v2`.
 
# Notes
- Uses rectangular aperture half-widths `hx = DEFAULT_x_slit/2`, `hz = DEFAULT_z_slit/2`.
- Displays a live `ProgressMeter` progress bar tracking accepted particles,
  updated under a lock since `next!` is not inherently safe to call
  concurrently from multiple threads. Contention is low since acceptance
  events are sparse relative to total trials.
- Prints timing due to `@time`, total trial count, and acceptance rate at the end.
- You can swap `TaskLocalRNG()` for a different per-thread RNG if desired
  (e.g., `MersenneTwister(hash((base_seed, thread_id)))`).
 
# Throws
- `ArgumentError` if `v_pdf` is not `:v3` or `:v2`.
- `ErrorException` if `No` particles are not accepted within
  `No * max_trial_multiplier` total trials across all threads.
 
# Example
julia
Threads.nthreads()  # ensure desired thread count is set before starting Julia
alive = _generate_samples_multithreaded(100_000, 12345, effusion_params; v_pdf = :v3)
size(alive)  # (100000, 6)
"""
function _generate_samples_multithreaded(No::Int, base_seed::Int, p::EffusionParams;
                                          v_pdf::Symbol=:v3, max_trial_multiplier::Int=10_000)
    # Output buffer, shared across all threads. Each thread writes only to
    # rows it has been granted an exclusive index for (via the atomic
    # `sample_count` counter below), so concurrent writes never collide.
    alive = Matrix{Float64}(undef, No, 6)
 
    # ── Resolve the velocity sampler ONCE, before spawning threads ────────
    # Shared (read-only) across all threads — avoids re-checking `v_pdf` on
    # every trial, and fails fast on an invalid value, same as the serial
    # version's optimization.
    sampler = if v_pdf === :v3
        AtomicBeamVelocity_v3
    elseif v_pdf === :v2
        AtomicBeamVelocity_v2
    else
        throw(ArgumentError("Unknown v_pdf=$v_pdf; use :v3 or :v2"))
    end
 
    # Atomics are required here because ordinary `Int` counters are not safe
    # to increment from multiple threads simultaneously: two threads could
    # read the same value before either writes back, silently dropping a count.
    sample_count    = Threads.Atomic{Int}(0)   # accepted particles so far (shared)
    iteration_count = Threads.Atomic{Int}(0)   # total proposal trials (shared)
    max_trials      = No * max_trial_multiplier   # safety cap, see docstring
 
    # Precomputed constants (identical role to the serial sampler above).
    hx    = DEFAULT_x_slit / 2
    hz    = DEFAULT_z_slit / 2
    epsvy = 1e-18
 
    # Progress bar tracking accepted particles (bounded by `No`), shared
    # across threads. `next!` is serialized through `prog_lock` since
    # ProgressMeter does not guarantee safety under concurrent calls.
    prog      = Progress(No; desc="Sampling (multithreaded): ")
    prog_lock = Threads.SpinLock()
 
    @time Threads.@threads for thread_id in 1:Threads.nthreads()
        # Each thread gets its own independent RNG stream, seeded
        # deterministically from (base_seed, thread_id) so the overall set of
        # draws is reproducible for a fixed thread count.
        rng0 = TaskLocalRNG()
        Random.seed!(rng0, hash((base_seed, thread_id)))
        # rng0 = MersenneTwister(hash((base_seed, thread_id)))   # alternative RNG, kept for reference
 
        while true
            # `+ 1` converts the pre-increment value `atomic_add!` returns
            # into the post-increment (current) trial count, matching the
            # serial version's `iteration_count += 1` semantics for the
            # `max_trials` check below.
            trial = Threads.atomic_add!(iteration_count, 1) + 1
 
            if trial > max_trials
                error("_generate_samples_multithreaded: exceeded $max_trials trials " *
                      "($(sample_count[])/$No accepted) — check geometry and v_pdf=$v_pdf for a near-zero acceptance rate.")
            end
 
            # ── Initial transverse position (uniform over furnace rectangle) ──
            x0 = DEFAULT_x_furnace * (rand(rng0) - 0.5)
            z0 = DEFAULT_z_furnace * (rand(rng0) - 0.5)
 
            # ── Initial velocity ─────────────────────────────────────────
            # `sampler` was resolved once above — no branch on `v_pdf` here.
            v0_x, v0_y, v0_z = sampler(rng0, p)
 
            # ── Reject near-zero v0y (avoid divide-by-zero / huge times) ──
            if abs(v0_y) ≤ epsvy
                continue
            end
 
            # ── Ballistic propagation to the slit plane (same geometry as serial) ──
            x_at_slit = x0 + DEFAULT_y_FurnaceToSlit * v0_x / v0_y
            z_at_slit = z0 + DEFAULT_y_FurnaceToSlit * v0_z / v0_y
 
            # ── Accept/reject against the rectangular slit aperture ──────
            if -hx ≤ x_at_slit ≤ hx && -hz ≤ z_at_slit ≤ hz
                # Each accepted particle claims a unique 1-based row index
                # atomically, so no two threads ever write the same row of
                # `alive`. `atomic_add!` returns the value BEFORE the
                # addition, so `+ 1` gives this particle's actual row index
                # (without it, the first accepted particle would get idx=0,
                # which is out of bounds for a 1-indexed array).
                idx = Threads.atomic_add!(sample_count, 1) + 1
                if idx <= No
                    # Write fields directly instead of building a temporary
                    # Vector and copying it in — avoids one small heap
                    # allocation per accepted particle, same as the serial version.
                    @inbounds begin
                        alive[idx, 1] = x0
                        alive[idx, 2] = 0.0
                        alive[idx, 3] = z0
                        alive[idx, 4] = v0_x
                        alive[idx, 5] = v0_y
                        alive[idx, 6] = v0_z
                    end
                    Threads.lock(prog_lock) do
                        next!(prog)
                    end
                else
                    # Enough particles have already been collected by other
                    # threads; stop this thread's sampling loop.
                    break
                end
            end
 
        end
    end
 
    finish!(prog)
    acceptance_rate = No / iteration_count[]
    println("Total iterations: ", iteration_count[], "  (acceptance rate: ", round(100*acceptance_rate, digits=3), "%)")
    return alive
end

"""
    generate_samples(No::Int, p::EffusionParams;
                     v_pdf::Symbol = :v3,
                     rng = Random.DEFAULT_rng(),
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
  `:v2` → `AtomicBeamVelocity_v2`; other values throw `ArgumentError`).
- `rng = Random.DEFAULT_rng()`: RNG used **only in serial mode**.
- `multithreaded::Bool = false`: If `true`, uses the threaded sampler; otherwise serial.
- `base_seed::Int = 1234`: Per-thread seeding base used **only in multithreaded mode**.
- `max_trial_multiplier::Int = 10_000`: Forwarded to whichever sampler is
  used; aborts with an error if `No` particles aren't accepted within
  `No * max_trial_multiplier` total trials.
 
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
function generate_samples(
    No::Int, p::EffusionParams;
    v_pdf::Symbol               = :v3,
    rng                          = Random.default_rng(),
    multithreaded::Bool         = false,
    base_seed::Int              = 1234,
    max_trial_multiplier::Int   = 100_000,
)
    # Thin dispatcher: the actual sampling logic lives entirely in the two
    # `_generate_samples_*` implementations above; this just picks one based
    # on the `multithreaded` flag and forwards the relevant arguments,
    # including the shared safety-cap keyword.
    if multithreaded
        return _generate_samples_multithreaded(No, base_seed, p; v_pdf=v_pdf, max_trial_multiplier=max_trial_multiplier)
    else
        return _generate_samples_serial(No, rng, p; v_pdf=v_pdf, max_trial_multiplier=max_trial_multiplier)
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
function generate_CQDinitial_conditions(
    No::Integer, alive::AbstractMatrix{T}, rng::AbstractRNG;
    mode::Symbol = :partition,
) where {T<:Real}
    @assert No > 0 "No must be > 0"
    @assert No == size(alive,1) "Total number of particles $No"
    @assert size(alive,2) ≥ 6 "alive must have at least 6 columns (x0,y0,z0,v0x,v0y,v0z)"
 
    # Draw one CQD angle θ ∈ [0, π] in the target element type T.
    # θ = 2·asin(√u) with u ~ U(0,1) is the inverse-CDF transform for a
    # polar angle distributed as f(θ) ∝ sin(θ) — i.e. the standard
    # "uniform on a sphere, restricted to the polar angle" distribution
    # used for the CQD electron/nuclear angle draws.
    @inline sample_theta() = T(2asin(sqrt(rand(rng))))
 
    if mode === :partition
        # ── Two-pass exact split ──────────────────────────────────────
        # Pass 1 (counting): clone the RNG so this pass doesn't consume any
        # of the draws the fill pass will need. Replaying the *same* draw
        # sequence (θe, θn) on the clone lets us count exactly how many
        # particles will classify as "up" before allocating any output.
        @assert hasmethod(copy, Tuple{typeof(rng)}) "RNG must support copy() for two-pass mode"
        rng1 = copy(rng)
        n_up = 0
        @inbounds for _ in 1:No
            θe = T(2asin(sqrt(rand(rng1))))
            θn = T(2asin(sqrt(rand(rng1))))
            n_up += (θe < θn)
        end
        n_down = No - n_up
 
        # Pass 2 (fill): now that the exact split sizes are known, allocate
        # tightly sized output matrices and fill them using the *original*
        # `rng` (untouched by the counting pass above, so it reproduces the
        # identical (θe, θn) sequence in the same order).
        up_batch   = Matrix{T}(undef, n_up,   8)
        down_batch = Matrix{T}(undef, n_down, 8)
 
        iu = 0; id = 0
        @inbounds @views for i in 1:No
            θe = sample_theta()
            θn = sample_theta()
            if θe < θn
                # "Up" branch: copy this particle's kinematics (cols 1-6)
                # plus its drawn angles (cols 7-8) into the next free row.
                iu += 1
                up_batch[iu, 1:6] = alive[i, 1:6]
                up_batch[iu, 7]   = θe
                up_batch[iu, 8]   = θn
            else
                # "Down" branch: same, into down_batch instead.
                id += 1
                down_batch[id, 1:6] = alive[i, 1:6]
                down_batch[id, 7]   = θe
                down_batch[id, 8]   = θn
            end
        end
        return up_batch, down_batch
 
    elseif mode === :balanced
        # ── Single-pass balanced batches ─────────────────────────────
        # Unlike :partition, this mode does NOT split the No original
        # particles between the two branches. Instead, it draws fresh
        # (θe, θn) pairs repeatedly until No "up" classifications AND No
        # "down" classifications have each been collected, producing two
        # full-size (No×8) batches — every original kinematic row appears
        # in BOTH up_batch and down_batch, just paired with different
        # independently-drawn angles in each.
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
        # Kinematic columns (1-6) are identical for every row in both
        # batches — copied once in bulk rather than per-row inside the loop.
        @inbounds @views begin
            up_batch[:,   1:6] .= alive[:, 1:6]
            down_batch[:, 1:6] .= alive[:, 1:6]
        end
        return up_batch, down_batch
 
    else
        error("Unknown mode=$mode. Use :partition or :balanced.")
    end
end