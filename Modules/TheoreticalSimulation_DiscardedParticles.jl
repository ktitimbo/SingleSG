########################################################################################################################################
# Quantum Mechanics : Classical Trajectories
########################################################################################################################################

"""
    QM_flag_travelling_particles(
        Ix, init_particles, p::AtomParams; y_length::Int=1000, verbose::Bool=false
    ) -> OrderedDict{Int, Vector{Vector{UInt8}}}

Parallel, per-current / per-(F,mF) classification of particle trajectories through
the Stern–Gerlach (SG) cavity.

For each coil current `I₀ ∈ Ix` and each hyperfine sublevel `(F, mF)`, the function
evaluates every particle trajectory given by `init_particles[j, 1:6]` across `y_length`
sample points spanning the cavity and assigns a 1-byte code from `QM_cavity_crash`:

Codes (`UInt8`):
- `0x00` — clears the cavity **and** passes the downstream tube/screen.
- `0x01` — hits the **top edge** (ceiling) somewhere inside the cavity.
- `0x02` — hits the **bottom trench** somewhere inside the cavity.
- `0x03` — clears the cavity but **misses the tube/screen** aperture.

# Arguments
- `Ix::AbstractVector{<:Real}`: Coil currents (A). Each is processed on a thread.
- `init_particles::AbstractMatrix{<:Real}`: `(No × 6)` matrix of initial states per particle
  with columns `(x0, y0, z0, v0x, v0y, v0z)` (SI units).
- `p::AtomParams`: Atomic/physics parameters (e.g. `p.M`), used via `μF_effective`/`GvsI`.

# Keyword Arguments
- `y_length::Int=1000`: Number of y-samples over the cavity span
  `y ∈ [y_in, y_out]`, where `y_in = DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG`
  and `y_out = y_in + DEFAULT_y_SG`.
- `verbose::Bool=false`: If `true`, prints a status line per current.

# Returns
`OrderedDict{Int, Vector{Vector{UInt8}}}` mapping **current index**
`Int(1:length(Ix))` → vector over `(F, mF)` levels → `Vector{UInt8}` of length `No`
(containing the code for each particle).

# Performance notes
- Work is `Threads.@threads` over currents; per-current work is independent.
  Results are collected into a plain pre-sized `Vector` inside the threaded
  region (safe — each thread writes to a unique index) and assembled into the
  `OrderedDict` afterward, single-threaded, since concurrent `OrderedDict`
  insertion is not thread-safe.
- Time ~ `length(Ix) * nlevels * No * y_length`; dominated by `QM_cavity_crash`
  itself (it scans `y_length` points per call) — nothing in this function
  changes how many times that kernel is invoked.
- Result memory ~ `ncurrents * nlevels * No` bytes (one `UInt8` per particle).
- Loop order is levels-outer/particles-inner, so `μF_effective` (which depends
  only on current+level, not on the particle) is computed once per level
  rather than once per particle — at the cost of re-reading `init_particles`'
  6 columns from scratch on every level iteration.
- Best performance when:
  - `Ix::Vector{Float64}` and `init_particles::Matrix{Float64}` (avoids per-element casts).
  - Geometry globals are set **before** running and not mutated during execution.
  - The inner kernel `QM_cavity_crash` is called positionally (no kwargs).

# Example
julia
Ix    = collect(range(-20.0, 20.0; length=5))
init_particles = randn(10_000, 6) .* [1e-3, 0.10, 1e-3, 1e-3, 100.0, 1e-3]'
out   = QM_flag_travelling_particles(Ix, init_particles, K39_params;
                                                               y_length=1024, verbose=true)

# Codes for current index 1, level 3:
codes  = out[Int(1)][3]               # Vector{UInt8} of length size(init_particles,1)
n_top  = count(==(0x01), codes)
n_bot  = count(==(0x02), codes)
n_scr  = count(==(0x03), codes)
n_pass = count(==(0x00), codes)
"""
function QM_flag_travelling_particles(Ix, init_particles, p::AtomParams;
                                                    y_length::Int=1000,
                                                    verbose::Bool=false)
    @info "Evaluating particle trajectories and assigning flags"
    No        = size(init_particles, 1) # number of particles
    ncurrents = length(Ix)              # number of currents
    fmf       = fmf_levels(p)           # (F, mF) pairs, ordered per fmf_levels' convention
    nlevels   = length(fmf)             # number of (F, mF) levels

    # --- Precompute and share the y-grid for the SG span ---
    # Read-only and shared across all threads — the same ygrid is reused for
    # every (current, level, particle) trajectory evaluation below.
    y_in  = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    y_out = (y_in + DEFAULT_y_SG)::Float64
    ygrid = range(y_in, y_out; length=y_length)

    # Thread-local results bucket. Each thread writes to a UNIQUE index `idx`
    # (one per current), so no locking is needed here — this is exactly what
    # makes the Threads.@threads loop below safe without further
    # synchronization on the physics results themselves.
    results = Vector{Vector{Vector{UInt8}}}(undef, ncurrents)

    # Optional: serialize prints to avoid interleaved output from different threads
    print_lock = ReentrantLock()

    # Global progress counter, safe across threads via atomic increment
    progress = Threads.Atomic{Int}(0)

    Threads.@threads for idx in 1:ncurrents
        i0 = Ix[idx]

        if verbose
            # increment progress *before* doing the work
            c = Threads.atomic_add!(progress, 1) + 1
            lock(print_lock); 
            try
                @printf "[%02d/%02d]\tAnalyzing I₀ = %.3f A \t (levels = %d)\n" c ncurrents i0 nlevels
            finally
                unlock(print_lock)
            end
        end

        # For this current: one UInt8[No] vector per (F,mF), initialized to 0x00 (pass)
        codes_for_levels = [fill(UInt8(0x00), No) for _ in 1:nlevels]

        # --- hoist G(I) once per current (shared by every level below) ---
        gI = GvsI(i0)

        # --- loop levels outer, particles inner (μ computed once per level) ---
        # μG depends only on (current, level), not on the particle, so this
        # ordering re-reads init_particles' 6 columns from scratch on every
        # level iteration — nlevels× more reads than strictly necessary.
        # (See chat for a "particles outer, levels inner" alternative.)
        @inbounds for k in 1:nlevels
            F, mF = fmf[k]
            μ     = μF_effective(i0, F, mF, p)   # once per (current, level)
            μG    = μ * gI

            @inbounds for j in 1:No
                # zero-alloc scalar loads (faster than building SVectors/views)
                x0  = Float64(init_particles[j,1]);  y0  = Float64(init_particles[j,2]);  z0  = Float64(init_particles[j,3])
                v0x = Float64(init_particles[j,4]);  v0y = Float64(init_particles[j,5]);  v0z = Float64(init_particles[j,6])

                code = QM_cavity_crash(μG,x0,y0,z0,v0x,v0y,v0z,p,ygrid,0.0)

                codes_for_levels[k][j] = code
            end
        end

        results[idx] = codes_for_levels
    end

    # Assemble as an OrderedDict keyed by current index (Int).
    # Done OUTSIDE the threaded region, single-threaded: concurrent
    # OrderedDict insertion isn't safe, so results are gathered into the
    # plain `results` Vector above first (safe under threading), then copied
    # in here sequentially. Note `@inbounds` only protects the `results[idx]`
    # read — it has no effect on the `out[Int(idx)]` dict insertion, since
    # that's hash-based, not a bounds-checked array access.
    out = OrderedDict{Int, Vector{Vector{UInt8}}}()
    @inbounds for idx in 1:ncurrents
        out[Int(idx)] = results[idx]
    end
    return out
end

"""
    QM_build_travelling_particles(Ix, init_particles, flagged_trajec, p) -> OrderedDict{Int, Vector{Matrix{Float64}}}

For each coil current index and each `(F, mF)` level, propagate all input particles
from the SG entrance to the screen and return a per-level matrix of results.

# Inputs
- `Ix::Vector{Float64}`: Coil currents (A). Indices `1:length(Ix)` label each "current".
- `init_particles::Matrix{Float64}`: `No × 6` matrix of initial conditions (rows = particles) with columns
  `[x0, y0, z0, v0x, v0y, v0z]` in SI units.
- `flagged_trajec::OrderedDict{Int, Vector{Vector{UInt8}}}`:
  For each current index `idx::Int`, a vector of length `nlevels` where element `k`
  is a `Vector{UInt8}` of length `No` with a per-particle **flag** for level `k`
  (e.g. 0=pass, 1=top, 2=bottom, 3=tube). These flags are written into column 10.
  This is exactly the return value of [`QM_flag_travelling_particles`](@ref).
- `p::AtomParams`: Atom/beam parameters. Used here via
  `fmf_levels(p)`, `μF_effective(I0, F, mF, p)`, and `p.M` (mass).

# Geometry & dynamics
Uses global geometry (all in meters):  
`DEFAULT_y_FurnaceToSlit`, `DEFAULT_y_SlitToSG`, `DEFAULT_y_SG`, `DEFAULT_y_SGToScreen`.  
Defines total drift `Ltot = y_in + LSG + Ld` and uses
`Δ = LSG^2 + 2 LSG*Ld` inside the kinematics helper `QM_screen_x_z_vz`.

The transverse acceleration for each (current, level) is
`a_z = μF_effective(I0, F, mF, p) * GvsI(I0) / p.M`.

# Output
Returns `OrderedDict{Int, Vector{Matrix{Float64}}}`. For each current index `idx`,
the value is a vector of length `nlevels`; its `k`-th matrix is `No × 10` with columns:
1. `x0`
2. `y0`
3. `z0`
4. `v0x`
5. `v0y`
6. `v0z`
7. `x_screen`
8. `z_screen`
9. `v_z_exit`  (after SG region)
10. `flag`     (from `flagged_trajec[idx][k][j]`, stored as `Float64`)

Row order is preserved.

# Notes
- Expects `length(flagged_trajec[idx][k]) == size(init_particles, 1)` for all `idx` and `k`.
  **This is not currently checked** — a mismatch will surface as a `BoundsError`
  on `flags_k[j]` deep inside the loop rather than a clear validation message
  near the top of the function.
- `QM_screen_x_z_vz` must be available in scope and uses SI units, and is
  expected to be a cheap closed-form calculation (no internal iteration),
  unlike `QM_cavity_crash` used in `QM_flag_travelling_particles`.

# Performance notes
- **Not currently threaded**, unlike [`QM_flag_travelling_particles`](@ref),
  even though the work here is structured identically: independent per-current
  computation, writing to a unique output key. See chat for how to add
  `Threads.@threads` here using the same "Vector during the loop, OrderedDict
  after" pattern already used in that sibling function.
- Loop order is levels-outer/particles-inner for the same reason as
  `QM_flag_travelling_particles`: `μ`/`a_z` depend only on `(current, level)`.
  Unlike that function, the per-level body here also re-copies all 6
  `init_particles` columns into a fresh matrix on every level iteration —
  since `QM_screen_x_z_vz` looks like an O(1) closed-form call (not an
  iterative kernel), this redundant copy is likely a larger fraction of this
  function's total cost than the analogous redundancy is in
  `QM_flag_travelling_particles`.
"""
function QM_build_travelling_particles(
    Ix::Vector{Float64},
    init_particles::Matrix{Float64},
    flagged_trajec::OrderedDict{Int, Vector{Vector{UInt8}}},
    p::AtomParams
)
    No      = size(init_particles, 1)
    fmf     = fmf_levels(p)
    nlevels = length(fmf)

    # Bind geometry once (typed)
    y_in = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    Lsg  = DEFAULT_y_SG::Float64
    Ld   = DEFAULT_y_SGToScreen::Float64
    Ltot = (y_in + Lsg + Ld)::Float64
    Δ    = (Lsg*Lsg + 2.0*Lsg*Ld)::Float64  # (Lsg+Ld)^2 - Ld^2

    out = OrderedDict{Int, Vector{Matrix{Float64}}}()

    # NOTE: this loop is sequential — no Threads.@threads, unlike the
    # otherwise-structurally-identical loop in QM_flag_travelling_particles.
    # Each iteration here is fully independent (writes only to out[Int(idx)]
    # and reads only from Ix/init_particles/flagged_trajec, none of which are
    # mutated), so it's a candidate for the same threading pattern — see chat.
    for (kcurr, codes_per_level) in flagged_trajec
        idx = Int(kcurr)
        I0  = Ix[idx]
        gI  = GvsI(I0)  # once per current

        mats = Vector{Matrix{Float64}}(undef, nlevels)

        @inbounds for k in 1:nlevels
            F, mF = fmf[k]
            μ     = μF_effective(I0, F, mF, p)    # once per (current, level)
            a_z  = (μ * gI) / p.M                # a_z

            flags_k = codes_per_level[k]          # Vector{UInt8} length = No
            # length(flags_k) == No is assumed but not asserted — see docstring Notes.

            M = Matrix{Float64}(undef, No, 10)

            # copy the 6 initial columns (column-wise is cheap, no views created)
            # NOTE: this is the SAME init_particles data, re-read from scratch
            # on every level iteration (nlevels total full re-reads) — see
            # docstring's Performance notes.
            @simd for j in 1:No; M[j,1] = init_particles[j,1]; end
            @simd for j in 1:No; M[j,2] = init_particles[j,2]; end
            @simd for j in 1:No; M[j,3] = init_particles[j,3]; end
            @simd for j in 1:No; M[j,4] = init_particles[j,4]; end
            @simd for j in 1:No; M[j,5] = init_particles[j,5]; end
            @simd for j in 1:No; M[j,6] = init_particles[j,6]; end

            # fill screen columns + flag with scalar math (no function calls, no slices)
            for j in 1:No
                x0  = M[j,1]; z0  = M[j,3]
                v0x = M[j,4]; v0y = M[j,5]; v0z = M[j,6]
                x,z,vz = QM_screen_x_z_vz(x0,z0,v0x,v0y,v0z, Lsg, Δ, Ltot, a_z)
                M[j,7]  = x;  
                M[j,8]  = z;  
                M[j,9]  = vz
                M[j,10] = Float64(flags_k[j])   # keep your layout
            end

            mats[k] = M
        end

        out[Int(idx)] = mats
    end

    return out
end

"""
    QM_flag_travelling_particles_twowires(
        Ix, init_particles, particles_at_sg, p::AtomParams, cal::SGCalibration;
        y_length::Int=1000, verbose::Bool=false
    ) -> OrderedDict{Int, Vector{Vector{UInt8}}}

Two-wire-magnet counterpart of [`QM_flag_travelling_particles`](@ref): parallel,
per-current / per-(F,mF) classification of particle trajectories through the SG
cavity, using a spatially-varying field computed from a two-wire magnet model
instead of a single scalar gradient `GvsI(I)`.

Unlike the simple model, the field (and hence the effective moment `μ`) here
depends on **where each particle actually enters the SG region**
(`particles_at_sg[j, :]`), not just on the coil current — so this function
evaluates the local field once per particle, then reuses it across all
hyperfine levels for that particle.

Return codes are identical to `QM_flag_travelling_particles`:
- `0x00` — clears the cavity **and** passes the downstream tube/screen.
- `0x01` — hits the **top edge** (ceiling) somewhere inside the cavity.
- `0x02` — hits the **bottom trench** somewhere inside the cavity.
- `0x03` — clears the cavity but **misses the tube/screen** aperture.

# Arguments
- `Ix::AbstractVector{<:Real}`: Coil currents (A). Each is processed on a thread.
- `init_particles::AbstractMatrix{<:Real}`: `(No × 6)` matrix of initial states
  per particle, columns `(x0, y0, z0, v0x, v0y, v0z)` (SI units), same layout
  as in `QM_flag_travelling_particles`.
- `particles_at_sg::AbstractMatrix{<:Real}`: `(No × ≥3)` matrix giving each
  particle's position at the SG entrance plane. Only columns 1 (`x`) and 3
  (`z`) are read here; the entrance plane is taken as `y = 0` in this local
  frame (hardcoded below, not read from column 2).
- `p::AtomParams`: Atomic/physics parameters (`p.M`), used via `μF_effective_B`.
- `cal::SGCalibration`: Calibration mapping nominal coil current to an
  effective two-wire current (`cal.I_eff_B`) and a gradient scale correction
  (`cal.grad_scale`), built from field-map/Hall-probe data.

# Keyword Arguments
- `y_length::Int=1000`: Number of y-samples over the cavity span, same role
  as in `QM_flag_travelling_particles`.
- `verbose::Bool=false`: If `true`, prints a status line per current.

# Returns
`OrderedDict{Int, Vector{Vector{UInt8}}}`, same shape and meaning as
`QM_flag_travelling_particles`'s return value.

# Performance notes
- Already threaded (`Threads.@threads` over currents), same safe pattern as
  `QM_flag_travelling_particles`: per-current results are written to a
  unique index of a plain `Vector` inside the threaded region, then copied
  into the `OrderedDict` afterward, single-threaded.
- Loop order is **particles outer, levels inner** — and here this is not a
  style choice, it's required: `B0`/`G` depend on the particle's SG-entrance
  position, so they must be computed per particle. Levels only vary `μ`
  (via `μF_effective_B`, which is cheap), reusing the same per-particle field.
- `cal.I_eff_B(I0)` / `cal.grad_scale(I0)` are already correctly hoisted
  outside the particle loop — computed once per current, not once per particle.
- `B_total` is called once per particle and its result is reused inside
  `grad_normB` (passed in directly), avoiding a redundant second field
  evaluation — already handled, no further hoisting available here.
- As with `QM_flag_travelling_particles`, total runtime is almost certainly
  dominated by `QM_cavity_crash`'s internal `y_length`-point scan, called
  `nlevels × No` times per current regardless of anything in this function.
"""
function QM_flag_travelling_particles_twowires(Ix, init_particles, particles_at_sg, 
                                         p::AtomParams,cal::SGCalibration;
                                         y_length::Int=1000,
                                         verbose::Bool=false)
    @info "Evaluating particle trajectories and assigning flags"
    No        = size(init_particles, 1) # number of particles
    ncurrents = length(Ix)              # number of currents
    fmf       = fmf_levels(p)           # quantum numbers — computed once, shared (read-only) across all threads
    nlevels   = length(fmf)             # number of (F, mF) levels

    # --- Precompute and share the y-grid for the SG span ---
    y_in  = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    y_out = (y_in + DEFAULT_y_SG)::Float64
    ygrid = range(y_in, y_out; length=y_length)

    # Thread-local results bucket (each idx is unique, so no locking needed)
    results = Vector{Vector{Vector{UInt8}}}(undef, ncurrents)

    # Optional: serialize prints to avoid interleaving
    print_lock = ReentrantLock()

    # NEW: global progress counter, safe across threads
    progress = Threads.Atomic{Int}(0)

    Threads.@threads for idx in 1:ncurrents
        I0 = Ix[idx]

        # ── Hoist calibration out of the particle loop ──────────────────────
        # Both depend only on the current I0, not on any particle, so computing
        # them once here (rather than inside the j-loop below) avoids No-fold
        # redundant calibration lookups per current.
        Iw_eff = cal.I_eff_B(I0)
        S      = cal.grad_scale(I0)
        # ────────────────────────────────────────────────────────────────────

        if verbose
            # increment progress *before* doing the work
            c = Threads.atomic_add!(progress, 1) + 1
            lock(print_lock); 
            try
                @printf "[%02d/%02d]\tAnalyzing I₀ = %.3f A \t (levels = %d)\n" c ncurrents I0 nlevels
            finally
                unlock(print_lock)
            end
        end

        # For this current: one Int[No] vector per (F,mF), initialized to 0
        codes_for_levels = [fill(UInt8(0x00), No) for _ in 1:nlevels]

        # --- loop particles outer, levels inner ---
        # Required (not just preferred) here: B0/G below depend on this
        # particle's SG-entrance position, so they can't be hoisted any
        # further out than this. Levels only vary μ, computed cheaply from
        # the already-known B0.
        @inbounds for j in 1:No
            # zero-alloc scalar loads (faster than building SVectors/views)
            x0  = Float64(init_particles[j,1]);  y0  = Float64(init_particles[j,2]);  z0  = Float64(init_particles[j,3])
            v0x = Float64(init_particles[j,4]);  v0y = Float64(init_particles[j,5]);  v0z = Float64(init_particles[j,6])

            # Particle's actual entry point into the SG region — the field is
            # NOT spatially uniform in the two-wire model, so this position
            # (not just the current) determines the local field/gradient.
            # y is hardcoded to 0.0: the SG-entrance plane is the local origin.
            xsg, ysg, zsg = particles_at_sg[j,1], 0.0, particles_at_sg[j,3]

            # single B_total call — result reused in grad_normB
            Bx, By, Bz = B_total(xsg, ysg, zsg; Iw=Iw_eff)
            B0 = sqrt(Bx^2 + By^2 + Bz^2)

            # gradient with precomputed B, scaled by S
            _, _, dBdz = grad_normB(xsg, ysg, zsg, Bx, By, Bz; Iw=Iw_eff)

            G = S * dBdz

            @inbounds for k in 1:nlevels
                F, mF = fmf[k]
                μ     = μF_effective_B(B0, F, mF, p)   # once per (current, level)
                μG    = μ * G

                code = QM_cavity_crash(μG,x0,y0,z0,v0x,v0y,v0z,p,ygrid,0.0)

                codes_for_levels[k][j] = code
            end
        end

        results[idx] = codes_for_levels
    end

    # Assemble as an OrderedDict keyed by current index (Int)
    out = OrderedDict{Int, Vector{Vector{UInt8}}}()
    @inbounds for idx in 1:ncurrents
        out[Int(idx)] = results[idx]
    end
    return out
end

"""
    QM_build_travelling_particles_twowires(
        Ix, init_particles, particles_at_sg, flagged_trajec, p::AtomParams, cal::SGCalibration
    ) -> OrderedDict{Int, Vector{Matrix{Float64}}}

Two-wire-magnet counterpart of [`QM_build_travelling_particles`](@ref): for each
coil current index and each `(F, mF)` level, propagate all input particles from
the SG entrance to the screen using a two-wire-calibrated, spatially-varying
field instead of the simple scalar gradient `GvsI(I)`.

As in [`QM_flag_travelling_particles_twowires`](@ref), the field at the SG
entrance (and hence the effective moment) depends on **each particle's actual
entry position** (`particles_at_sg[j, :]`), not just the coil current, so the
field is evaluated once per particle and reused across all hyperfine levels
for that particle.

# Inputs
- `Ix::Vector{Float64}`: Coil currents (A). Indices `1:length(Ix)` label each "current".
- `init_particles::Matrix{Float64}`: `No × 6` matrix of initial conditions
  (rows = particles), columns `[x0, y0, z0, v0x, v0y, v0z]`, SI units.
- `particles_at_sg::Matrix{Float64}`: `No × ≥3` matrix of each particle's
  position at the SG entrance plane. Only columns 1 (`x`) and 3 (`z`) are
  used; the entrance-plane `y` is hardcoded to `0.0`, not read from column 2.
- `flagged_trajec::OrderedDict{Int, Vector{Vector{UInt8}}}`: per-current,
  per-level particle flags — exactly the return value of
  [`QM_flag_travelling_particles_twowires`](@ref). Element `codes_per_level[k][j]`
  is written into column 10 of level `k`'s output matrix for particle `j`.
- `p::AtomParams`: Atom/beam parameters. Used via `fmf_levels(p)`,
  `μF_effective_B(B0, F, mF, p)`, and `p.M` (mass).
- `cal::SGCalibration`: Calibration providing `cal.I_eff_B(I0)` (effective
  two-wire current) and `cal.grad_scale(I0)` (gradient scale correction).

# Geometry & dynamics
Uses global geometry (all in meters), identical to `QM_build_travelling_particles`:
`DEFAULT_y_FurnaceToSlit`, `DEFAULT_y_SlitToSG`, `DEFAULT_y_SG`, `DEFAULT_y_SGToScreen`,
combined into `Ltot` and `Δ = Lsg² + 2·Lsg·Ld` for `QM_screen_x_z_vz`.

For each particle `j` and level `k`, the transverse acceleration is
`a_z = μF_effective_B(B0_j, F, mF, p) * (S * dBdz_j) / p.M`, where `B0_j` and
`dBdz_j` are the field and gradient at that particle's own SG-entrance position.

# Output
Returns `OrderedDict{Int, Vector{Matrix{Float64}}}`. For each current index `idx`,
a vector of length `nlevels`; its `k`-th matrix is `No × 10` with the same column
layout as `QM_build_travelling_particles`:
`[x0, y0, z0, v0x, v0y, v0z, x_screen, z_screen, v_z_exit, flag]`.

# Notes
- Expects `length(codes_per_level[k]) == No` and `size(particles_at_sg, 1) == No`
  for every current/level. **Neither is currently asserted** — a mismatch
  surfaces as a `BoundsError` inside the loop rather than a clear message.

# Performance notes
- **Not threaded**, same as `QM_build_travelling_particles` — and for the
  same reason there's no obstacle to threading it: each current's iteration
  here is independent, reading only from `Ix`/`init_particles`/`particles_at_sg`/
  `flagged_trajec` and writing only to its own `out[Int(idx)]` entry.
- The **physics loop** (particles-outer, levels-inner) is already optimal —
  it correctly mirrors `QM_flag_travelling_particles_twowires`: `B_total` and
  `grad_normB` are each called exactly once per particle, not once per
  (particle, level), and `B0`/`G` are computed before the level loop and
  reused across all `nlevels` iterations.
- The **column-copy step** (copying `init_particles`'s 6 columns into every
  level's output matrix) is separate from the physics loop and is
  level-outer — it re-reads all 6 `init_particles` columns from scratch on
  every level iteration, `nlevels`-fold more reads than necessary. Since
  `QM_screen_x_z_vz` looks like an O(1) closed-form call, this redundant copy
  is likely a non-trivial fraction of this function's total cost (same
  concern as in `QM_build_travelling_particles`).
"""
function QM_build_travelling_particles_twowires(
    Ix::Vector{Float64},
    init_particles::Matrix{Float64},
    particles_at_sg::Matrix{Float64},
    flagged_trajec::OrderedDict{Int, Vector{Vector{UInt8}}},
    p::AtomParams, cal::SGCalibration
)
    No      = size(init_particles, 1)
    fmf     = fmf_levels(p)
    nlevels = length(fmf)

    # Bind geometry once (typed)
    y_in = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    Lsg  = DEFAULT_y_SG::Float64
    Ld   = DEFAULT_y_SGToScreen::Float64
    Ltot = (y_in + Lsg + Ld)::Float64
    Δ    = (Lsg*Lsg + 2.0*Lsg*Ld)::Float64  # (Lsg+Ld)^2 - Ld^2

    out = OrderedDict{Int, Vector{Matrix{Float64}}}()

    # NOTE: sequential, like QM_build_travelling_particles — same missing-
    # threading opportunity, same fix shape (see chat).
    for (kcurr, codes_per_level) in flagged_trajec
        idx = Int(kcurr)
        I0  = Ix[idx]
        # ── Hoist calibration out of the particle loop ──────────────────────
        # Depends only on I0, computed once per current (correct).
        Iw_eff = cal.I_eff_B(I0)
        S      = cal.grad_scale(I0)
        # ────────────────────────────────────────────────────────────────────

        mats = [Matrix{Float64}(undef, No, 10) for _ in 1:nlevels]

        # copy the 6 initial columns into all level matrices
        # NOTE: level-outer — re-reads init_particles' 6 columns from scratch
        # on every level iteration (nlevels total full re-reads). This step is
        # purely for the OUTPUT layout (recording initial conditions alongside
        # results) — it is NOT a cache the physics loop below reads from; that
        # loop independently re-reads init_particles directly (see below),
        # which is already minimal (one read per particle, not per level).
        @inbounds for k in 1:nlevels
            @simd for j in 1:No; mats[k][j,1] = init_particles[j,1]; end
            @simd for j in 1:No; mats[k][j,2] = init_particles[j,2]; end
            @simd for j in 1:No; mats[k][j,3] = init_particles[j,3]; end
            @simd for j in 1:No; mats[k][j,4] = init_particles[j,4]; end
            @simd for j in 1:No; mats[k][j,5] = init_particles[j,5]; end
            @simd for j in 1:No; mats[k][j,6] = init_particles[j,6]; end
        end

        # --- physics loop: particles outer, levels inner (already optimal) ---
        # Required by the same reasoning as QM_flag_travelling_particles_twowires:
        # B0/G depend on this particle's SG-entrance position, so they're
        # computed once per particle and reused across all nlevels below.
        @inbounds for j in 1:No
            x0  = Float64(init_particles[j,1]); z0  = Float64(init_particles[j,3])
            v0x = Float64(init_particles[j,4]); v0y = Float64(init_particles[j,5]); v0z = Float64(init_particles[j,6])

            xsg, ysg, zsg = particles_at_sg[j,1], 0.0, particles_at_sg[j,3]

            # single B_total call — result reused in grad_normB
            Bx, By, Bz = B_total(xsg, ysg, zsg; Iw=Iw_eff)
            B0         = sqrt(Bx^2 + By^2 + Bz^2)

            # gradient with precomputed B, scaled by S
            _, _, dBdz = grad_normB(xsg, ysg, zsg, Bx, By, Bz; Iw=Iw_eff)

            G = S * dBdz

            @inbounds for k in 1:nlevels
                F, mF = fmf[k]
                μ     = μF_effective_B(B0, F, mF, p)    # once per (current, particle, level)
                a_z  = (μ * G) / p.M                    # a_z

                x,z,vz = QM_screen_x_z_vz(x0,z0,v0x,v0y,v0z, Lsg, Δ, Ltot, a_z)

                mats[k][j,7]  = x
                mats[k][j,8]  = z
                mats[k][j,9]  = vz
                mats[k][j,10] = Float64(codes_per_level[k][j])
            end

        end

        out[Int(idx)] = mats
    end

    return out
end

"""
    QM_travelling_particles_summary(Ixs, q_numbers, particles) -> nothing

Pretty-prints, for each coil current `I₀` in `Ixs`, a per–(F,mF) summary of
particle outcomes based on the **flag in column 10** of each particle matrix.

For every current `I₀` and every quantum level `(F, mF)`, the function counts:
- `Pass`     — number of rows with flag `0` in column 10
- `Top`      — number of rows with flag `1`
- `Bottom`   — number of rows with flag `2`
- `Aperture` — number of rows with flag `4`
- `Tube`     — number of rows with flag `3`

It then prints a table (via PrettyTables) with columns:
`F, mF, Pass, Top, Bottom, Aperture, Tube, Total, Pass %, Loss %`,
where `Loss % = 100 * (Top + Bottom + Aperture + Tube) / Total`.
When `Total == 0`, both percentages are shown as `0.0`.

# Arguments
- `Ixs::AbstractVector`: Coil currents `I₀` (A). One summary table is printed per entry.
- `q_numbers::AbstractVector{<:Tuple}`: Quantum numbers `(F, mF)` for the levels; its length
  defines `nlevels`.
- `particles`: A vector of length `length(Ixs)`. For each `i`, `particles[i]` is a
  vector of length `nlevels` containing the particle matrices for current `Ixs[i]`.
  Each matrix must be `N×10`; only **column 10** is used here and must contain the
  outcome flag (either `Int` {0,1,2,3,4} or `Float64` {0.0,1.0,2.0,3.0,4.0}).

# Behavior
- Prints one formatted table per `I₀` using PrettyTables (with rounded unicode borders,
  centered alignment, custom colors/styles as configured in the call).
- Inserts a blank line between tables.
- Returns `nothing`.

# Example
julia
Ixs = [0.10, 0.20]
q_numbers = [(2, 1), (2, 0)]
M1 = rand(1_000, 10); M1[:,10] .= rand([0,1,2,3,4], size(M1,1))
M2 = rand( 800, 10); M2[:,10] .= rand([0,1,2,3,4], size(M2,1))
M3 = rand(1200, 10); M3[:,10] .= rand([0,1,2,3,4], size(M3,1))
M4 = rand( 900, 10); M4[:,10] .= rand([0,1,2,3,4], size(M4,1))

particles = [
    [M1, M2],  # for Ixs[1]
    [M3, M4],  # for Ixs[2]
]

QM_travelling_particles_summary(Ixs, q_numbers, particles)
"""
function QM_travelling_particles_summary(Ixs, q_numbers, particles)

    nlevels = length(q_numbers)

    # count flags in column 10; supports Float or Integer flags
    @inline function counts_from_M(M::AbstractMatrix)
        @views col = M[:,end]
        if eltype(col) <: Integer
            pass = count(==(0),  col); top = count(==(1), col)
            bot  = count(==(2),  col); tub = count(==(3), col)
            aper = count(==(4),  col);
        else
            pass = count(==(0.0), col); 
            top = count(==(1.0), col)
            bot  = count(==(2.0), col); 
            tub = count(==(3.0), col)
            aper = count(==(4.0), col);
        end
        return (pass=pass, top=top, bot=bot, tub=tub, aper=aper)
    end

    for i in eachindex(Ixs)
        I0   = Float64(Ixs[i])
        mats = particles[i]  # Vector of No×10 matrices, one per level

        nrows = nlevels
        data  = Matrix{Float64}(undef, nrows, 10)

        # rows per level
        for j in 1:nlevels
            F, mF = q_numbers[j]
            M     = mats[j]
            c     = counts_from_M(M)
            tot   = c.pass + c.top + c.bot + c.tub + c.aper

            passp = tot == 0 ? 0.0 : 100.0 * c.pass / tot
            lossp = tot == 0 ? 0.0 : 100.0 * (c.top + c.bot + c.tub + c.aper) / tot

            data[j, :] = [Float64(F), Float64(mF),
                          c.pass, c.top, c.bot, c.aper, c.tub,
                          tot, passp, lossp]
        end

        pretty_table(
            data;
            column_labels = ["F","mF","Pass","Top","Bottom","Aperture","Tube","Total","Pass %","Loss %"],
            title = @sprintf("I₀ = %.3f A", I0),
            formatters    = [fmt__printf("%d", 3:8), fmt__printf("%5.1f", 9:10)],
            alignment = :c,
            table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
            style = TextTableStyle(first_line_column_label = crayon"yellow bold",
                                    table_border  = crayon"blue bold",
                                    title = crayon"bold red"),
            equal_data_column_widths = true,
        )
        println()
    end
    return nothing
end

"""
    QM_select_flagged(initial_by_current, which; flagcol=10) -> OrderedDict{K, Vector{Matrix{Float64}}}

Filter each per-level matrix in `initial_by_current` by the flag stored in
column `flagcol`, keeping only rows whose flag matches the set implied by
`which`. Returns matrices restricted to columns `1:(flagcol-1)` — the flag
column itself is dropped from the output.

This is the QM/per-level counterpart of [`CQD_select_flagged`](@ref): the two
do the same conceptual job the same way, differing only in container shape —
`Vector{Matrix}` per current here (one matrix per hyperfine level), vs a
single `Matrix` per current there (CQD has no levels to break out).

`which` → kept flags:
- `:screen`     → `{0}`           (passed)
- `:crash_SG`   → `{1, 2}`        (hit top or bottom inside the cavity)
- `:crash_tube` → `{3}`           (cleared cavity, missed downstream tube/screen)
- `:crash_aper` → `{4}`           (aperture crash)
- `:crash`      → `{1, 2, 3, 4}`  (any non-pass outcome)
- `:all`        → `{0, 1, 2, 3, 4}`

# Arguments
- `initial_by_current::OrderedDict{K, Vector{Matrix{Float64}}}` where `K<:Integer`:
  per-current vector of per-level `No × ≥flagcol` matrices, as produced by
  [`QM_build_travelling_particles`](@ref) / [`QM_build_travelling_particles_twowires`](@ref).
  Different levels may have different row counts; each is handled
  independently (see Notes).
- `which::Symbol`: one of the categories listed above.
- `flagcol::Int=10`: 1-based index of the flag column. Output matrices keep
  columns `1:(flagcol-1)`.

# Returns
`OrderedDict{K, Vector{Matrix{Float64}}}` — same key type `K` as the input,
same per-current/per-level nesting, with rows filtered and the flag column
dropped. Each level's output row count equals however many of that level's
rows matched `which`; this is independent across levels.

# Notes
- `which` values not in the list above raise a plain `error("Invalid which value")`.
- Validates `1 ≤ flagcol ≤ size(M, 2)` per matrix before use; a mismatch
  raises a clear `AssertionError` naming the offending size, rather than
  surfacing later as a `BoundsError` inside the filtering loop.
- `M`, `col`, and `keep` are all freshly computed within each level
  iteration, with nothing carried over from one level to the next — so
  levels with differing row counts (e.g. different pass/crash rates per
  hyperfine level) are handled correctly with no special-casing needed.

# Performance notes
- `findall(f -> f in flagset_set, col)` applies the predicate during the
  scan — no intermediate `Vector{Bool}` materialized first.
- One allocation for `keep` (sized to the match count, not `nrows`) and one
  for the output matrix per level; no scratch buffer retained between calls
  or between levels.
- `flagset_set = Set(flagset)` is computed once per call, shared across
  every `(idx, k)` pair — not recomputed per level.

# Example
julia
screen_only = QM_select_flagged(built_particles, :screen)
crashes     = QM_select_flagged(built_particles, :crash; flagcol=10)
"""
function QM_select_flagged(initial_by_current::OrderedDict{K, Vector{Matrix{Float64}}},
                        which::Symbol; flagcol::Int=10) where {K<:Integer}

    # Map the requested category to the set of flag values it keeps.
    flagset = which === :screen     ? (0,)            :
              which === :crash_SG   ? (1, 2)          :
              which === :crash_tube ? (3,)            :
              which === :crash_aper ? (4,)            :
              which === :crash      ? (1, 2, 3, 4)    :
              which === :all        ? (0, 1, 2, 3, 4) :
              error("Invalid which value")

    flagset_set = Set(flagset)   # computed once, shared across every (idx, k) below
    out = OrderedDict{K, Vector{Matrix{Float64}}}()   # preserves the input's key type, rather than forcing Int

    @inbounds for (idx, mats) in initial_by_current
        nlevels = length(mats)
        v = Vector{Matrix{Float64}}(undef, nlevels)

        for k in 1:nlevels
            M = mats[k]
            @assert 1 ≤ flagcol ≤ size(M, 2) "flagcol out of bounds (got $flagcol, size=$(size(M)))"
            ncols = flagcol - 1
            @views col = M[:, flagcol]

            keep = findall(f -> f in flagset_set, col)

            v[k] = M[keep, 1:ncols]   # copy rows into a dense Matrix
        end

        out[idx] = v
    end

    return out
end

########################################################################################################################################
# Co-Quantum Dynamics
########################################################################################################################################

"""
    CQD_flag_travelling_particles(Ix, init_particles, kx::Float64, p::AtomParams;
                                   y_length::Int=1000, verbose::Bool=false
    ) -> Vector{Vector{UInt8}}

CQD counterpart of [`QM_flag_travelling_particles`](@ref): parallel, per-current
classification of particle trajectories through the SG cavity under the
Co-Quantum-Dynamics model.

Structurally different from the QM version in two ways:
- **No hyperfine-level loop.** CQD doesn't classify particles by discrete
  (F, mF) sublevels; instead each particle's own drawn angles `(θe, θn)`
  (columns 7–8 of `init_particles`) already encode which CQD branch it
  belongs to. `CQD_cavity_crash` is called once per particle, not once per
  (particle, level).
- **Effective moment is the bare electron moment**, not `μF_effective`:
  `μG = μₑ * GvsI(I0)`, where `μₑ` is the fixed physical constant (electron
  magnetic moment), not a level-dependent value. The branch-dependent physics
  (governed by `kx` and the sign of `θn - θe`) is handled inside
  `CQD_cavity_crash` itself.

# Arguments
- `Ix::AbstractVector{<:Real}`: Coil currents (A). Each is processed on a thread.
- `init_particles::AbstractMatrix{<:Real}`: `(No × ≥8)` matrix of per-particle
  state. Columns `1:6` are `(x0, y0, z0, v0x, v0y, v0z)` (SI units, same as
  the QM functions); columns `7:8` are the CQD angles `(θe, θn)` drawn by
  [`generate_CQDinitial_conditions`](@ref).
- `kx::Float64`: CQD coupling constant. Passed through directly to
  `CQD_cavity_crash`; combined there with the Larmor frequency and the sign
  of `θn - θe` (see `CQD_build_travelling_particles`'s explicit `kω` for the
  analogous combination used downstream — full details live in
  `CQD_cavity_crash` itself, not reviewed yet).
- `p::AtomParams`: Atomic/physics parameters, forwarded to `CQD_cavity_crash`.

# Keyword Arguments
- `y_length::Int=1000`: Number of y-samples over the cavity span, same role
  as in `QM_flag_travelling_particles`.
- `verbose::Bool=false`: If `true`, prints a status line per current.

# Performance notes
- Already threaded (`Threads.@threads` over currents), same safe pattern as
  `QM_flag_travelling_particles`.
- No redundant work analogous to the QM functions' levels-outer/particles-inner
  question — there is no levels loop, so `CQD_cavity_crash` is called exactly
  `No` times per current, which is already the minimum possible.
- `μG`/`B0` are correctly hoisted once per current (not recomputed per particle).
- Input columns are read via a type **assertion** (`init_particles[j,1]::Float64`)
  rather than a conversion (`Float64(init_particles[j,1])`, the style used in
  the QM functions). Equivalent speed when the input genuinely is
  `Matrix{Float64}` (the common case); the difference only shows up if it
  isn't — this version throws immediately on a type mismatch, the QM style
  would silently convert. Worth knowing the two styles coexist in this file,
  not something to "fix."

# Notes
- The shape check (`size(init_particles,2) ≥ 8`) happens once, up front,
  before the `@inbounds` particle loop — exactly the right place for it,
  since it's what justifies skipping bounds-checks inside the threaded loop.
"""
function CQD_flag_travelling_particles(Ix, init_particles, kx::Float64, p::AtomParams;
                                                    y_length::Int=1000,
                                                    verbose::Bool=false)
    @info "Evaluating particle trajectories and assigning flags"
    @assert size(init_particles, 2) ≥ 8 "init_particles must have at least 8 columns"
    No        = size(init_particles, 1) # number of particles
    ncurrents = length(Ix)              # number of currents

    # --- Precompute and share the y-grid for the SG span ---
    y_in  = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    y_out = (y_in + DEFAULT_y_SG)::Float64
    ygrid = range(y_in, y_out; length=y_length)

    # Thread-local results bucket: one Vector{UInt8} per current
    results = Vector{Vector{UInt8}}(undef, ncurrents)

    # Optional: serialize prints to avoid interleaving
    print_lock = ReentrantLock()
    progress = Threads.Atomic{Int}(0)

    Threads.@threads for idx in eachindex(Ix)
        I0 = Ix[idx]

        if verbose
            # increment progress *before* doing the work
            c = Threads.atomic_add!(progress, 1) + 1
            lock(print_lock); 
            try
                @printf "[%02d/%d]\tAnalyzing I₀ = %.3f A \n" c ncurrents I0
            finally
                unlock(print_lock)
            end
        end

        # Hoist field/gradient for this current — depends only on I0, not on
        # the particle or any level (there is no level loop in CQD), so this
        # is already computed at the minimum possible frequency.
        μG = μₑ * GvsI(I0)
        B0 = BvsI(I0)

        # Codes for all particles at this current
        codes = Vector{UInt8}(undef, No)

        @inbounds for j in 1:No
            # zero-alloc scalar loads — `::Float64` is a type ASSERTION here
            # (errors if init_particles isn't really Matrix{Float64}), not a
            # conversion like Float64(...) used in the QM functions.
            x0  = init_particles[j, 1]::Float64
            y0  = init_particles[j, 2]::Float64
            z0  = init_particles[j, 3]::Float64
            v0x = init_particles[j, 4]::Float64
            v0y = init_particles[j, 5]::Float64
            v0z = init_particles[j, 6]::Float64
            θe  = init_particles[j, 7]::Float64
            θn  = init_particles[j, 8]::Float64

            # One call per particle — no per-level loop, since the CQD branch
            # is already determined by (θe, θn), not a discrete (F,mF) level.
            codes[j] = CQD_cavity_crash(μG,B0,x0,y0,z0,v0x,v0y,v0z,θe,θn,kx,p,ygrid,1e-18)
        end

        results[idx] = codes
    end

    return results
end

"""
    CQD_build_travelling_particles(
        Ix::Vector{Float64}, kx::Float64, init_particles::Matrix{Float64},
        flagged_trajec::Vector{Vector{UInt8}}, p::AtomParams
    ) -> OrderedDict{Int, Matrix{Float64}}

CQD counterpart of [`QM_build_travelling_particles`](@ref): for each coil
current, propagate all input particles from the SG entrance to the screen
under the CQD model and return a single per-current matrix of results.

Like [`CQD_flag_travelling_particles`](@ref), there is no per-(F,mF)-level
loop — each particle's own `(θe, θn)` already determines its CQD branch, so
exactly one result matrix is built per current (not one per current per level).

# Inputs
- `Ix::Vector{Float64}`: Coil currents (A). `eachindex(Ix)` labels each "current".
- `kx::Float64`: CQD coupling constant, combined with the Larmor frequency
  and `sign(θn0 - θe0)` per particle to form `kω`.
- `init_particles::Matrix{Float64}`: `No × ≥8` matrix; columns `1:6` are
  `(x0, y0, z0, v0x, v0y, v0z)` (SI units), columns `7:8` are the CQD angles
  `(θe, θn)`.
- `flagged_trajec::Vector{Vector{UInt8}}`: length-`ncurrents` vector of
  per-current flag vectors — exactly the return value of
  `CQD_flag_travelling_particles`. `flagged_trajec[idx][j]` is written into
  column 12 for particle `j` of current `idx`.
- `p::AtomParams`: Atom/beam parameters. Used via `p.M` only (no
  `fmf_levels`/`μF_effective` calls here, unlike the QM build functions —
  there are no hyperfine levels to look up in the CQD model).

!!! note "Argument order differs from `CQD_flag_travelling_particles`"
    Here the order is `(Ix, kx, init_particles, ...)`; in
    `CQD_flag_travelling_particles` it's `(Ix, init_particles, kx, p; ...)`
    — `kx` and `init_particles` swap positions between the two. Not a bug,
    just easy to trip over when calling both in sequence.

# Geometry & dynamics
Uses the same global geometry as `QM_build_travelling_particles`
(`DEFAULT_y_FurnaceToSlit`, `DEFAULT_y_SlitToSG`, `DEFAULT_y_SG`,
`DEFAULT_y_SGToScreen`), combined into `Ltot` and `ΔL = Lsg² + 2·Lsg·Ld`.

For each current, `a_z = (μₑ * GvsI(I0)) / p.M` (the bare electron moment,
not a level-dependent `μF_effective`) and `ωL = |γₑ · BvsI(I0)|`. For each
particle, `kω = sign(θn0 - θe0) * kx * ωL` selects the branch-dependent
contribution before calling `CQD_screen_x_z_vz`.

# Output
Returns `OrderedDict{Int, Matrix{Float64}}`. For each current index `idx`, a
single `No × 12` matrix with columns:
1–6. `x0, y0, z0, v0x, v0y, v0z`
7–8. `θe, θn`
9. `x_screen`
10. `z_screen`
11. `v_z_exit`
12. `flag` (from `flagged_trajec[idx][j]`, stored as `Float64`)
"""
function CQD_build_travelling_particles(
    Ix::Vector{Float64}, kx::Float64,
    init_particles::Matrix{Float64},
    flagged_trajec::Vector{Vector{UInt8}},
    p::AtomParams)::OrderedDict{Int, Matrix{Float64}}

    @assert size(init_particles,2) ≥ 8 "init_particles needs at least 8 columns"
    No        = size(init_particles, 1)
    ncurrents = length(Ix)
    @assert length(flagged_trajec) == ncurrents "flags length must match Ix"
    
    # Bind geometry once (typed)
    y_in = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    Lsg  = DEFAULT_y_SG::Float64
    Ld   = DEFAULT_y_SGToScreen::Float64
    Ltot = (y_in + Lsg + Ld)::Float64
    ΔL   = (Lsg*Lsg + 2.0*Lsg*Ld)::Float64  # (Lsg+Ld)^2 - Ld^2

    out = OrderedDict{Int, Matrix{Float64}}()

    # NOTE: sequential — no Threads.@threads, despite each `idx` iteration
    # being fully independent (writes only to out[idx], reads only from
    # Ix/init_particles/flagged_trajec). Same missing-threading situation as
    # the QM build functions.
    @inbounds for idx in eachindex(Ix)
        I0      = Ix[idx]
        flags_i = flagged_trajec[idx]
        @assert length(flags_i) == No "flags[idx] length must equal number of particles"

        # a_z for this current — depends only on I0, computed once (correct)
        μG   = μₑ * GvsI(I0)
        a_z  = μG / p.M
        ωL    = abs(γₑ * BvsI(I0))
        
        # No×12 matrix: 1–8 init, 9 x_scr, 10 z_scr, 11 vz_scr, 12 flag
        M = Matrix{Float64}(undef, No, 12)

        # Copy initial state columns (allocation-free, column-wise).
        # No per-level loop here (unlike the QM build functions), so each
        # column is read from init_particles exactly once — already minimal.
        @simd for j in 1:No; M[j,1] = init_particles[j,1]; end
        @simd for j in 1:No; M[j,2] = init_particles[j,2]; end
        @simd for j in 1:No; M[j,3] = init_particles[j,3]; end
        @simd for j in 1:No; M[j,4] = init_particles[j,4]; end
        @simd for j in 1:No; M[j,5] = init_particles[j,5]; end
        @simd for j in 1:No; M[j,6] = init_particles[j,6]; end
        @simd for j in 1:No; M[j,7] = init_particles[j,7]; end
        @simd for j in 1:No; M[j,8] = init_particles[j,8]; end

        for j in 1:No
            x0  = M[j,1];  z0  = M[j,3]
            v0x = M[j,4];  v0y = M[j,5];  v0z = M[j,6]
            θe0 = M[j,7];  θn0 = M[j,8]

            kω  = sign(θn0 - θe0) * kx * ωL

            x, z, vz = CQD_screen_x_z_vz(x0, z0, v0x, v0y, v0z, θe0, a_z, kω, Lsg, Ld, Ltot, ΔL)
            M[j,9]  = x
            M[j,10] = z
            M[j,11] = vz
            M[j,12] = Float64(flags_i[j])   # keep matrix eltype Float64
        end

        out[idx] = M
    end

    return out
end


"""
    CQD_flag_travelling_particles_twowires(
        Ix, init_particles, particles_at_sg, kx::Float64, p::AtomParams, cal::SGCalibration;
        y_length::Int=1000, verbose::Bool=false
    ) -> Vector{Vector{UInt8}}

Two-wire-magnet counterpart of [`CQD_flag_travelling_particles`](@ref): same CQD
branch-determination logic (no per-(F,mF)-level loop — each particle's own
`(θe, θn)` already encodes its branch), but with a spatially-varying field
computed from the two-wire calibration, exactly as in
[`QM_flag_travelling_particles_twowires`](@ref).

As in the QM two-wire function, the field at each particle's actual SG-entrance
position (`particles_at_sg[j, :]`) is evaluated once per particle (not just
once per current), since the field is not spatially uniform in this model.

# Arguments
- `Ix::AbstractVector{<:Real}`: Coil currents (A). Each is processed on a thread.
- `init_particles::AbstractMatrix{<:Real}`: `(No × ≥8)` matrix; columns `1:6`
  are `(x0, y0, z0, v0x, v0y, v0z)`, columns `7:8` are the CQD angles `(θe, θn)`.
- `particles_at_sg::AbstractMatrix{<:Real}`: `(No × ≥3)` matrix of each
  particle's position at the SG entrance. Only columns 1 (`x`) and 3 (`z`)
  are read; entrance-plane `y` is hardcoded to `0.0` (same convention as
  `QM_flag_travelling_particles_twowires`).
- `kx::Float64`: CQD coupling constant, forwarded to `CQD_cavity_crash`.
- `p::AtomParams`: Atomic/physics parameters, forwarded to `CQD_cavity_crash`.
- `cal::SGCalibration`: Calibration providing `cal.I_eff_B(I0)` and
  `cal.grad_scale(I0)`, same role as in `QM_flag_travelling_particles_twowires`.

!!! note "Argument order"
    `kx` sits between `particles_at_sg` and `p` here — matching
    `CQD_flag_travelling_particles`'s convention (`kx` before `p`), but
    **not** `CQD_build_travelling_particles`'s convention (`kx` right after
    `Ix`). Same cross-function ordering inconsistency noted for the build
    function; flagging again here since it's the same family.

# Keyword Arguments
- `y_length::Int=1000`, `verbose::Bool=false`: same role as in the other
  flag functions.

# Physics note
`μG = abs(μₑ * S * dBdz)` here, with an explicit `abs`, whereas the
single-wire `CQD_flag_travelling_particles` uses `μG = μₑ * GvsI(I0)` with
no `abs`. This looks like a deliberate difference (presumably because
`grad_normB`'s `dBdz` can come out negative depending on local field
direction, while `GvsI` is returned with a fixed sign convention already),
not an inconsistency to resolve — but worth confirming if you didn't
intend the asymmetry.

# Performance notes
- Already threaded, already hoists `cal.I_eff_B`/`cal.grad_scale` to
  once-per-current, already reuses the single `B_total` call inside
  `grad_normB` rather than recomputing the field — identical good pattern
  to `QM_flag_travelling_particles_twowires` and `CQD_flag_travelling_particles`.
- No per-level loop (none needed), so there's no analogous
  "redundant-recompute" question to ask here at all.
- Nothing in this function looks improvable without seeing inside
  `CQD_cavity_crash`/`B_total`/`grad_normB` (none reviewed yet).
"""
function CQD_flag_travelling_particles_twowires(Ix, 
                            init_particles, 
                            particles_at_sg, 
                            kx::Float64, 
                            p::AtomParams, 
                            cal::SGCalibration;
                            y_length::Int=1000,
                            verbose::Bool=false
)
    @info "Evaluating particle trajectories and assigning flags"
    @assert size(init_particles, 2) ≥ 8 "init_particles must have at least 8 columns"
    No        = size(init_particles, 1) # number of particles
    ncurrents = length(Ix)              # number of currents

    # --- Precompute and share the y-grid for the SG span ---
    y_in  = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    y_out = (y_in + DEFAULT_y_SG)::Float64
    ygrid = range(y_in, y_out; length=y_length)

    # Thread-local results bucket: one Vector{UInt8} per current
    results = Vector{Vector{UInt8}}(undef, ncurrents)

    # Optional: serialize prints to avoid interleaving
    print_lock = ReentrantLock()
    progress = Threads.Atomic{Int}(0)

    Threads.@threads for idx in eachindex(Ix)
        I0 = Ix[idx]

        # ── Hoist calibration out of the particle loop ──────────────────────
        # Depends only on I0, computed once per current (correct).
        Iw_eff = cal.I_eff_B(I0)
        S      = cal.grad_scale(I0)
        # ────────────────────────────────────────────────────────────────────

        if verbose
            # increment progress *before* doing the work
            c = Threads.atomic_add!(progress, 1) + 1
            lock(print_lock); 
            try
                @printf "[%02d/%d]\tAnalyzing I₀ = %.3f A \n" c ncurrents I0
            finally
                unlock(print_lock)
            end
        end

        # Codes for all particles at this current
        codes = Vector{UInt8}(undef, No)

        @inbounds for j in 1:No

            # Particle's actual SG-entrance position — the field is not
            # spatially uniform, so this (not just I0) determines the local
            # field/gradient. y hardcoded to 0.0, same convention as the QM
            # twowires function.
            xsg, ysg, zsg = particles_at_sg[j,1], 0.0, particles_at_sg[j,3]

            # single B_total call — result reused in grad_normB
            Bx, By, Bz = B_total(xsg, ysg, zsg; Iw=Iw_eff)
            B0         = sqrt(Bx^2 + By^2 + Bz^2)

            # gradient with precomputed B, scaled by S
            _, _, dBdz = grad_normB(xsg, ysg, zsg, Bx, By, Bz; Iw=Iw_eff)
            μG = abs(μₑ * S * dBdz)

            # zero-alloc scalar loads (type-assertion style, consistent with
            # CQD_flag_travelling_particles; QM functions use Float64(...)
            # conversion instead — both styles coexist in this file).
            x0  = init_particles[j, 1]::Float64
            y0  = init_particles[j, 2]::Float64
            z0  = init_particles[j, 3]::Float64
            v0x = init_particles[j, 4]::Float64
            v0y = init_particles[j, 5]::Float64
            v0z = init_particles[j, 6]::Float64
            θe  = init_particles[j, 7]::Float64
            θn  = init_particles[j, 8]::Float64

            codes[j] = CQD_cavity_crash(μG,B0,x0,y0,z0,v0x,v0y,v0z,θe,θn,kx,p,ygrid,1e-18)
        end

        results[idx] = codes
    end

    return results
end

"""
    CQD_build_travelling_particles_twowires(
        Ix::Vector{Float64}, kx::Float64, init_particles::Matrix{Float64},
        particles_at_sg::Matrix{Float64}, flagged_trajec::Vector{Vector{UInt8}},
        p::AtomParams, cal::SGCalibration
    ) -> OrderedDict{Int, Matrix{Float64}}

Two-wire-magnet counterpart of [`CQD_build_travelling_particles`](@ref): for
each coil current, propagate all input particles from the SG entrance to the
screen under the CQD model, using a spatially-varying field from the two-wire
calibration instead of the simple scalar gradient `GvsI(I)`.

As in [`CQD_flag_travelling_particles_twowires`](@ref), the field/gradient at
each particle's actual SG-entrance position (`particles_at_sg[j, :]`) is
evaluated once per particle, since the field is not spatially uniform.

# Inputs
- `Ix::Vector{Float64}`: Coil currents (A).
- `kx::Float64`: CQD coupling constant, combined per-particle into `kω`.
- `init_particles::Matrix{Float64}`: `No × ≥8` matrix; columns `1:6` are
  `(x0,y0,z0,v0x,v0y,v0z)`, columns `7:8` are the CQD angles `(θe,θn)`.
- `particles_at_sg::Matrix{Float64}`: `No × ≥3` matrix of each particle's
  SG-entrance position. Only columns 1 and 3 are read; entrance `y`
  hardcoded to `0.0`.
- `flagged_trajec::Vector{Vector{UInt8}}`: per-current flag vectors, the
  return value of `CQD_flag_travelling_particles_twowires`. Written into
  column 12.
- `p::AtomParams`: Used via `p.M` only (no hyperfine levels in CQD).
- `cal::SGCalibration`: Provides `cal.I_eff_B(I0)` and `cal.grad_scale(I0)`.

!!! note "Argument order, again"
    `(Ix, kx, init_particles, particles_at_sg, flagged_trajec, p, cal)` here,
    vs `(Ix, init_particles, particles_at_sg, kx, p, cal; ...)` in
    `CQD_flag_travelling_particles_twowires` — same kx-position swap between
    the flag/build pair as in the non-twowires functions.

# Geometry & dynamics
Same global geometry and `Ltot`/`ΔL` construction as
`CQD_build_travelling_particles`. Per particle: `μG = |μₑ·S·dBdz|`,
`a_z = μG/p.M`, `ωL = |γₑ·B0|`, `kω = sign(θn0-θe0)·kx·ωL` — identical
formulas to the single-wire version, but with `S`, `dBdz`, and `B0` all
evaluated from this particle's own SG-entrance field rather than from
`GvsI(I0)`/`BvsI(I0)`.

# Output
Returns `OrderedDict{Int, Matrix{Float64}}`, same `No × 12` column layout
as `CQD_build_travelling_particles`.

# Performance notes
- **Not threaded** — same gap as every other "build" function in this file
  (`QM_build_travelling_particles`, `QM_build_travelling_particles_twowires`,
  `CQD_build_travelling_particles`). Fourth occurrence of the identical fix
  shape: collect into a pre-sized `Vector` under `Threads.@threads`, then
  assemble the `OrderedDict` afterward.
- The **column-copy step** (8 columns, level-independent) and the
  **physics loop** are correctly separated, same structure as
  `CQD_build_travelling_particles` — no per-level redundancy applies here
  since there's no level loop at all.
- `B_total`'s result is reused in `grad_normB` (not recomputed) — same good
  pattern as `CQD_flag_travelling_particles_twowires`.
"""
function CQD_build_travelling_particles_twowires(
    Ix::Vector{Float64}, 
    kx::Float64,
    init_particles::Matrix{Float64},
    particles_at_sg::Matrix{Float64},
    flagged_trajec::Vector{Vector{UInt8}},
    p::AtomParams, 
    cal::SGCalibration
)::OrderedDict{Int, Matrix{Float64}}

    @assert size(init_particles,2) ≥ 8 "init_particles needs at least 8 columns"
    No        = size(init_particles, 1)
    ncurrents = length(Ix)
    @assert length(flagged_trajec) == ncurrents "flags length must match Ix"
    
    # Bind geometry once (typed)
    y_in = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    Lsg  = DEFAULT_y_SG::Float64
    Ld   = DEFAULT_y_SGToScreen::Float64
    Ltot = (y_in + Lsg + Ld)::Float64
    ΔL   = (Lsg*Lsg + 2.0*Lsg*Ld)::Float64  # (Lsg+Ld)^2 - Ld^2

    out = OrderedDict{Int, Matrix{Float64}}()

    # NOTE: sequential — same missing-threading situation as every other
    # "build" function in this file. Each idx iteration is independent.
    @inbounds for idx in eachindex(Ix)
        I0      = Ix[idx]

        # ── Hoist calibration out of the particle loop ──────────────────────
        Iw_eff = cal.I_eff_B(I0)
        S      = cal.grad_scale(I0)
        # ────────────────────────────────────────────────────────────────────

        flags_i = flagged_trajec[idx]
        @assert length(flags_i) == No "flags[idx] length must equal number of particles"
       
        # No×12 matrix: 1–8 init, 9 x_scr, 10 z_scr, 11 vz_scr, 12 flag
        M = Matrix{Float64}(undef, No, 12)

        # Copy initial state columns (allocation-free, column-wise).
        # No per-level loop here, so each column is read from init_particles
        # exactly once per current — already minimal, same as the
        # non-twowires CQD build function.
        @simd for j in 1:No; M[j,1] = init_particles[j,1]; end
        @simd for j in 1:No; M[j,2] = init_particles[j,2]; end
        @simd for j in 1:No; M[j,3] = init_particles[j,3]; end
        @simd for j in 1:No; M[j,4] = init_particles[j,4]; end
        @simd for j in 1:No; M[j,5] = init_particles[j,5]; end
        @simd for j in 1:No; M[j,6] = init_particles[j,6]; end
        @simd for j in 1:No; M[j,7] = init_particles[j,7]; end
        @simd for j in 1:No; M[j,8] = init_particles[j,8]; end

        for j in 1:No

            # Particle's actual SG-entrance position — drives the local
            # field/gradient (not spatially uniform in this model).
            xsg, ysg, zsg = particles_at_sg[j,1], 0.0, particles_at_sg[j,3]

            # single B_total call — result reused in grad_normB
            Bx, By, Bz = B_total(xsg, ysg, zsg; Iw=Iw_eff)
            B0         = sqrt(Bx^2 + By^2 + Bz^2)

            # gradient with precomputed B, scaled by S
            _, _, dBdz = grad_normB(xsg, ysg, zsg, Bx, By, Bz; Iw=Iw_eff)

            μG = abs(μₑ * S * dBdz)

            # a_z for this particle
            a_z  = μG / p.M
            ωL    = abs(γₑ * B0)


            x0  = M[j,1];  z0  = M[j,3]
            v0x = M[j,4];  v0y = M[j,5];  v0z = M[j,6]
            θe0 = M[j,7];  θn0 = M[j,8]

            kω  = sign(θn0 - θe0) * kx * ωL


            x, z, vz = CQD_screen_x_z_vz(x0, z0, v0x, v0y, v0z, θe0, a_z, kω, Lsg, Ld, Ltot, ΔL)
            M[j,9]  = x
            M[j,10] = z
            M[j,11] = vz
            M[j,12] = Float64(flags_i[j])   # keep matrix eltype Float64
        end

        out[idx] = M
    end

    return out
end

"""
    CQD_select_flagged(initial_by_current, which; flagcol=12)

Filter each matrix in an `OrderedDict{K, Matrix{Float64}}` by a flag column,
keeping only rows where the flag value matches the set implied by `which`.
Returns an `OrderedDict{K, Matrix{Float64}}` containing only columns `1:(flagcol-1)`.

`which` → kept flags:
- `:screen`     → `{0}`
- `:crash_SG`   → `{1, 2}`
- `:crash_tube` → `{3}`
- `:crash_aper` → `{4}`
- `:crash`      → `{1, 2, 3, 4}`
- `:all`        → `{0, 1, 2, 3, 4}`

This is the CQD/per-current counterpart of [`QM_select_flagged`](@ref) (which
operates on `Vector{Matrix}` per current instead of a single `Matrix`). The
`which`-category list, default `flagcol`, and filtering logic are otherwise
the same idea, just on a flatter container shape and via a simpler
implementation (no buffer pool — see Performance notes).

# Arguments
- `initial_by_current::OrderedDict{K, Matrix{Float64}}` where `K<:Integer`
- `which::Symbol`: one of the above
- `flagcol::Integer=12`: 1-based index of the flag column. Default differs
  from `QM_select_flagged`'s default of `10` — reflecting the extra two
  `(θe, θn)` columns CQD matrices carry that QM matrices don't.

# Returns
- `OrderedDict{K, Matrix{Float64}}` with filtered rows and columns before `flagcol`.

# Notes
- The error message (`"...or :cras_aper, :crash, or :all"`) has a typo:
  `cras_aper` should read `crash_aper`. Cosmetic only — doesn't affect which
  inputs are accepted, only what the error text says if `which` is invalid.

# Performance notes
- No buffer pool here, unlike `QM_select_flagged` — every call allocates a
  fresh `Set`, a fresh `keep` index vector (sized to the actual match count,
  not `nrows`), and a fresh output matrix via `M[keep, 1:ncols]`. Simpler
  code, and as discussed for `QM_select_flagged`, not obviously slower —
  `findall` here avoids the full-`nrows` scratch buffer the QM version uses,
  at the cost of allocating fresh each call instead of reusing one. Neither
  approach has been benchmarked against the other.
- `s = Set(flagset)` is correctly computed once per `(idx, M)` — wait, more
  precisely: once per **call** to this function, since `flagset`/`s` don't
  depend on `idx` or `M` at all. They're computed once before the loop and
  reused for every current in `initial_by_current` — already optimal
  placement, nothing to hoist further.
- `M[keep, 1:ncols]` does a row-gather from a column-major matrix — the same
  inherent "reading scattered elements out of column-major storage" cost
  noted for `QM_select_flagged`'s row-copy, not specific to this function.
"""
function CQD_select_flagged(initial_by_current::OrderedDict{K, Matrix{Float64}},which::Symbol; flagcol::Integer=12)  where {K<:Integer}
    # Map the requested category to the set of flag values it keeps.
    flagset = which === :screen     ? (0,)        :
              which === :crash_SG   ? (1, 2)      :
              which === :crash_tube ? (3,)        :
              which === :crash_aper ? (4,)        :
              which === :crash      ? (1,2,3,4)   :
              which === :all        ? (0,1,2,3,4) :
          error("which must be :screen, :crash_SG, :crash_tube, :cras_aper, :crash, or :all")

    out = OrderedDict{K, Matrix{Float64}}()
    s = Set(flagset)   # computed once, shared across every current below

    @inbounds for (idx, M) in initial_by_current
        @assert 1 ≤ flagcol ≤ size(M, 2) "flagcol out of bounds (got $flagcol, size=$(size(M)))"
        
        nrows = size(M,1)
        ncols = flagcol - 1
        @views col = M[:, flagcol]

        # keep rows where flag ∈ flagset. findall applies the predicate
        # directly during the scan — no intermediate Vector{Bool} is
        # materialized (contrast with QM_select_flagged's original
        # `findall(in.(col, Ref(s)))` form, which does build one).
        keep = findall(f -> f in s, col)

        out[idx] = M[keep, 1:ncols]   # copy rows into a dense Matrix
    end

    return out
end

"""
    CQD_travelling_particles_summary(Ixs, particles, branch) -> nothing

Pretty-prints, for each coil current `I₀` in `Ixs`, a one-row-per-current
summary of CQD particle outcomes based on the **flag in the last column**
of each particle matrix.

CQD counterpart of [`travelling_particles_summary`](@ref), but with one row
per **current** rather than one row per **(F, mF) level** — CQD has no
discrete hyperfine levels to break out, so there's exactly one table row
per `I₀`, not `nlevels` rows.

For every current, counts:
- `Pass`     — flag `0`
- `Top`      — flag `1`
- `Bottom`   — flag `2`
- `Aperture` — flag `4`
- `Tube`     — flag `3`

and prints `Loss % = 100 * (Top + Bottom + Aperture + Tube) / Total`.

# Arguments
- `Ixs::AbstractVector`: Coil currents `I₀` (A). One table row per entry.
- `particles`: A vector of length `length(Ixs)`. Each `particles[i]` may be
  either a single `No × ≥1` matrix, or a `Vector` of such matrices — in the
  latter case **only the first matrix is used** (`mat[1]`), with the rest
  silently ignored. Only the **last column** of whichever matrix is used is
  read; it must contain the outcome flag (`Int` or `Float64`, values 0–4).
- `branch::Union{Symbol,String}`: Display label only (shown uppercased in
  the table title via `uppercase(string(branch))`); does not affect counting.

# Behavior
- Prints **one combined table** (not one table per current, unlike
  `travelling_particles_summary` — here every current is a row of the same
  table rather than its own separate table).
- Row labels are the currents in mA, rounded to 3 significant digits.
- Returns `nothing`.

# Notes
- If `particles[i]` is a `Vector` of more than one matrix, only `mat[1]` is
  summarized — there's no indication in the output that the remaining
  matrices exist or were skipped.
- The comment `"First column is a string label → use Matrix{Any}"` in the
  body no longer matches what the code does: `data` is built entirely from
  numeric values (`c.pass`, `c.top`, ..., `passp`, `lossp`) with no string
  column anywhere. Looks like a leftover from an earlier version. See
  Performance notes for the cost of keeping `Matrix{Any}` regardless.
- `counts_from_M` here counts 5 categories (including `aper`).

# Performance notes
- `data = Matrix{Any}(undef, length(Ixs), 8)`: every value actually stored
  is numeric (the `[c.pass, ..., lossp]` literal promotes to `Vector{Float64}`
  before assignment), so `Matrix{Any}` buys nothing here — it just means each
  element gets boxed individually rather than stored inline, adding
  unnecessary indirection for `pretty_table` to read back out.
  `travelling_particles_summary` (the QM sibling) already uses
  `Matrix{Float64}` for the equivalent table; this function could match it.
  Given `length(Ixs)` is typically small (a sweep of currents, not
  particles), the practical impact is modest — but it's a clean, free fix.
- `counts_from_M` does 5 separate `count(==(v), col)` passes over the same
  `col` (length `No`, potentially very large) — same redundant-scan pattern
  flagged in `travelling_particles_summary`; a single pass with a small
  tally would replace 5×`No` work with 1×`No`. Here this runs once per
  *current* (not once per level), so the total redundant work scales with
  `length(Ixs) × No` rather than `nlevels × No`.
- This exact `counts_from_M` helper (5-category version) is defined as a
  separate nested function in both this function and
  `travelling_particles_summary` — identical code, two copies. A single
  shared helper (module-level, not nested) would remove the duplication;
  not a speed concern, just a maintenance one.
"""
function CQD_travelling_particles_summary(Ixs, particles, branch::Union{Symbol,String})
    # Small helper: count flags in last column (robust to Int/Float).
    # NOTE: same logic, separately defined, in travelling_particles_summary —
    # see docstring.
    @inline function counts_from_M(M::AbstractMatrix)
        @views col = M[:, end]
        if eltype(col) <: Integer
            pass = count(==(0),  col); top = count(==(1), col)
            bot  = count(==(2),  col); tub = count(==(3), col)
            aper = count(==(4),  col);
        else
            pass = count(==(0.0), col); top = count(==(1.0), col)
            bot  = count(==(2.0), col); tub = count(==(3.0), col)
            aper = count(==(4.0), col);
        end
        return (pass=pass, top=top, bot=bot, tub=tub, aper=aper)
    end

    # NOTE: Matrix{Any} — but every value placed into it below is numeric
    # (the literal `[c.pass, ..., lossp]` promotes to Float64 before
    # assignment). See docstring Performance notes.
    data = Matrix{Any}(undef, length(Ixs), 8)

    for i in eachindex(Ixs)
        I0 = Float64(Ixs[i])

        # Accept Matrix or Vector{Matrix}; use the first matrix if a vector is given.
        # NOTE: if a Vector with more than one matrix is passed, every matrix
        # after the first is silently ignored — no warning either way.
        mat = particles[i]
        M   = mat isa AbstractMatrix ? mat : mat[1]

        c   = counts_from_M(M)
        tot = c.pass + c.top + c.bot + c.tub + c.aper
        passp = tot == 0 ? 0.0 : 100.0 * c.pass / tot
        lossp = tot == 0 ? 0.0 : 100.0 * (c.top + c.bot + c.tub + c.aper) / tot

        # NOTE: stale comment below — there is no string-label column in
        # this row; see docstring Notes.
        # First column is a string label → use Matrix{Any}
        
        data[i, :] = [c.pass, c.top, c.bot, c.aper, c.tub, tot, passp, lossp]


    end

        pretty_table(
        data;
        column_labels               = ["Pass","Top","Bottom","Aperture","Tube","Total","Pass %","Loss %"],
        title                       = "CQD PARTICLE TRAJECTORIES STATISTICS ($(uppercase(string(branch))))",
        formatters                  = [fmt__printf("%d", 1:6), fmt__printf("%5.1f", 7:8)],
        alignment                   = :c,
        table_format                = TextTableFormat(borders = text_table_borders__unicode_rounded),
        style                       = TextTableStyle(first_line_column_label = crayon"yellow bold",
                                                    table_border = crayon"blue bold",
                                                    title = crayon"bold red"),
        equal_data_column_widths    = true,
        row_labels                  = Int.(round.(1000*Ixs, sigdigits=3)),
        stubhead_label              = "I₀ [mA]",
        row_label_column_alignment  = :c,
    )
    println()
    return nothing
end

########################################################################################################################################
# Co-Quantum Dynamics : B = B₀ + Bₙ*cos(θn)
########################################################################################################################################
# NOTE: No Bn-specific filter/summary functions — CQD_select_flagged and
# CQD_travelling_particles_summary (defined in the plain-CQD section above)
# are used for Bn results too. Both already support the full 0–4 flag range,
# so they work unmodified on Bn-produced matrices (same column-12 layout).

"""
    CQD_Bn_flag_travelling_particles(Ix, init_particles, kx::Float64, p::AtomParams;
                                      y_length::Int=1000, verbose::Bool=false
    ) -> Vector{Vector{UInt8}}

`Bₙ`-extended counterpart of [`CQD_flag_travelling_particles`](@ref), for the
CQD model with nuclear-field modulation `B = B₀ + Bₙ·cos(θn)` (per this
file's section header). Same role, same input/output shapes, same per-particle
loop — the only changes from the plain CQD version are which kernel function
is called and one numerical tolerance (see below).

# Arguments
Identical to `CQD_flag_travelling_particles`: `Ix`, `init_particles`
(`No × ≥8`, columns `1:6` kinematics + `7:8` = `(θe, θn)`), `kx`, `p`.

# Keyword Arguments
Identical to `CQD_flag_travelling_particles`: `y_length`, `verbose`.

# Returns
!!! note "Where does the Bₙ·cos(θn) term actually enter?"
    `B0 = BvsI(I0)` here is computed identically to the plain CQD version —
    there is no visible `+ Bn*cos(θn)` term in *this* function, unlike
    `CQD_Bn_build_travelling_particles` (later in this file), which
    explicitly computes `ωL = abs(γₑ * (BvsI(I0) + Bn*cos(θn0)))`. Since `θn`
    is passed directly into `CQD_Bn_cavity_crash` below, it's plausible (and
    I'd guess likely) that the nuclear-field modulation is applied *inside*
    that kernel rather than precomputed here — but I can't confirm this
    without seeing `CQD_Bn_cavity_crash`'s implementation (not yet reviewed).
    Worth checking explicitly rather than assuming either way.

!!! note "Differs from `CQD_flag_travelling_particles` in exactly two places"
    1. The kernel called: `CQD_Bn_cavity_crash` here vs `CQD_cavity_crash` there.
    2. The trailing tolerance argument: `0.0` here vs `1e-18` there.
    Every other line — threading, hoisting, loop structure, scalar-load
    style — is identical between the two functions. Worth double-checking
    that tolerance difference is deliberate and not a copy-paste slip,
    since it's easy to lose track of a single literal across two ~50-line,
    otherwise-identical functions.

# Performance notes
- Already threaded, already hoists `μG`/`B0` to once-per-current, no
  per-level loop to worry about — identical good structure to
  `CQD_flag_travelling_particles`. Nothing new to flag here that wasn't
  already flagged (or rather, not flagged, for lack of issues) there.
"""
function CQD_Bn_flag_travelling_particles(Ix, init_particles, kx::Float64, p::AtomParams;
                                                    y_length::Int=1000,
                                                    verbose::Bool=false)
    @info "Evaluating particle trajectories and assigning flags"
    @assert size(init_particles, 2) ≥ 8 "init_particles must have at least 8 columns"
    No        = size(init_particles, 1) # number of particles
    ncurrents = length(Ix)              # number of currents

    # --- Precompute and share the y-grid for the SG span ---
    y_in  = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    y_out = (y_in + DEFAULT_y_SG)::Float64
    ygrid = range(y_in, y_out; length=y_length)

    # Thread-local results bucket: one Vector{UInt8} per current
    results = Vector{Vector{UInt8}}(undef, ncurrents)

    # Optional: serialize prints to avoid interleaving
    print_lock = ReentrantLock()
    progress = Threads.Atomic{Int}(0)

    Threads.@threads for idx in eachindex(Ix)
        I0 = Ix[idx]

        if verbose
            # increment progress *before* doing the work
            c = Threads.atomic_add!(progress, 1) + 1
            lock(print_lock); 
            try
                @printf "[%02d/%d]\tAnalyzing I₀ = %.3f A \n" c ncurrents I0
            finally
                unlock(print_lock)
            end
        end

        # Hoist field/gradient for this current — depends only on I0.
        # NOTE: no visible Bn*cos(θn) term here; see docstring note on
        # whether that's applied inside CQD_Bn_cavity_crash instead.
        μG = μₑ * GvsI(I0)
        B0 = BvsI(I0)

        # Codes for all particles at this current
        codes = Vector{UInt8}(undef, No)

        @inbounds for j in 1:No
            # zero-alloc scalar loads
            x0  = init_particles[j, 1]::Float64
            y0  = init_particles[j, 2]::Float64
            z0  = init_particles[j, 3]::Float64
            v0x = init_particles[j, 4]::Float64
            v0y = init_particles[j, 5]::Float64
            v0z = init_particles[j, 6]::Float64
            θe  = init_particles[j, 7]::Float64
            θn  = init_particles[j, 8]::Float64

            # NOTE: tolerance argument is 0.0 here; CQD_flag_travelling_particles
            # uses 1e-18 for the structurally identical call — see docstring.
            codes[j] = CQD_Bn_cavity_crash(μG,B0,x0,y0,z0,v0x,v0y,v0z,θe,θn,kx,p,ygrid,0.0)
        end

        results[idx] = codes
    end

    return results
end

"""
    CQD_Bn_build_travelling_particles(
        Ix::Vector{Float64}, kx::Float64, init_particles::Matrix{Float64},
        flagged_trajec::Vector{Vector{UInt8}}, p::AtomParams
    ) -> OrderedDict{Int, Matrix{Float64}}

`Bₙ`-extended counterpart of [`CQD_build_travelling_particles`](@ref), for the
CQD model with nuclear-field modulation `B = B₀ + Bₙ·cos(θn)`.

The `Bₙ·cos(θn)` term is applied per-particle (`θn0` varies by particle),
while the `B₀ = BvsI(I0)` part — which depends only on the current, not the
particle — is computed once per current, same as `μG`/`a_z` just above it.

# Inputs
Identical to `CQD_build_travelling_particles`: `Ix`, `kx`, `init_particles`
(`No × ≥8`), `flagged_trajec` (return value of
`CQD_Bn_flag_travelling_particles`), `p`.

# Geometry & dynamics
Same `Ltot`/`ΔL` construction as `CQD_build_travelling_particles`. Per
current: `a_z = (μₑ·GvsI(I0))/p.M` and `B0 = BvsI(I0)` (no `θn` dependence,
hoisted once). Per particle: `ωL = |γₑ·(B0 + Bn·cos(θn0))|`,
`kω = sign(θn0-θe0)·kx·ωL`, then `CQD_Bn_screen_x_z_vz`.

# Output
Same `No × 12` column layout as `CQD_build_travelling_particles`.

# Performance notes
- `BvsI(I0)` is now hoisted to once per current (alongside `μG`/`a_z`),
  matching `CQD_build_travelling_particles`'s structure. Only the
  `θn0`-dependent `Bn*cos(θn0)` term remains inside the per-particle loop.
- **Not threaded** — same gap as every other build function in this file;
  unrelated to and unaffected by this edit.
"""
function CQD_Bn_build_travelling_particles(
    Ix::Vector{Float64}, kx::Float64,
    init_particles::Matrix{Float64},
    flagged_trajec::Vector{Vector{UInt8}},
    p::AtomParams)::OrderedDict{Int, Matrix{Float64}}

    @assert size(init_particles,2) ≥ 8 "init_particles needs at least 8 columns"
    No        = size(init_particles, 1)
    ncurrents = length(Ix)
    @assert length(flagged_trajec) == ncurrents "flags length must match Ix"
    
    # Bind geometry once (typed)
    y_in = (DEFAULT_y_FurnaceToSlit + DEFAULT_y_SlitToSG)::Float64
    Lsg  = DEFAULT_y_SG::Float64
    Ld   = DEFAULT_y_SGToScreen::Float64
    Ltot = (y_in + Lsg + Ld)::Float64
    ΔL   = (Lsg*Lsg + 2.0*Lsg*Ld)::Float64  # (Lsg+Ld)^2 - Ld^2

    out = OrderedDict{Int, Matrix{Float64}}()

    @inbounds for idx in eachindex(Ix)
        I0      = Ix[idx]
        flags_i = flagged_trajec[idx]
        @assert length(flags_i) == No "flags[idx] length must equal number of particles"

        # Per-current quantities — none depend on the particle, so all three
        # are computed exactly once per current, not once per particle.
        μG   = μₑ * GvsI(I0)
        a_z  = μG / p.M
        B0   = BvsI(I0)   # hoisted: only Bn*cos(θn0) below actually needs the per-particle loop
        
        # No×12 matrix: 1–8 init, 9 x_scr, 10 z_scr, 11 vz_scr, 12 flag
        M = Matrix{Float64}(undef, No, 12)

        # Copy initial state columns (allocation-free, column-wise)
        @simd for j in 1:No; M[j,1] = init_particles[j,1]; end
        @simd for j in 1:No; M[j,2] = init_particles[j,2]; end
        @simd for j in 1:No; M[j,3] = init_particles[j,3]; end
        @simd for j in 1:No; M[j,4] = init_particles[j,4]; end
        @simd for j in 1:No; M[j,5] = init_particles[j,5]; end
        @simd for j in 1:No; M[j,6] = init_particles[j,6]; end
        @simd for j in 1:No; M[j,7] = init_particles[j,7]; end
        @simd for j in 1:No; M[j,8] = init_particles[j,8]; end

        for j in 1:No
            x0  = M[j,1];  z0  = M[j,3]
            v0x = M[j,4];  v0y = M[j,5];  v0z = M[j,6]
            θe0 = M[j,7];  θn0 = M[j,8]

            ωL  = abs(γₑ * (B0 + DEFAULT_CQD_Bn*cos(θn0)))   # only the θn0 term is genuinely per-particle
            kω  = sign(θn0 - θe0) * kx * ωL

            x, z, vz = CQD_screen_x_z_vz(x0, z0, v0x, v0y, v0z, θe0, a_z, kω, Lsg, Ld, Ltot, ΔL)
            M[j,9]  = x
            M[j,10] = z
            M[j,11] = vz
            M[j,12] = Float64(flags_i[j])   # keep matrix eltype Float64
        end

        out[idx] = M
    end

    return out
end
