# """
#     QM_find_bad_particles_ix(Ix, pairs, p::AtomParams;
#                              t_length=1000, verbose=false)
#         -> Dict{Int8, Vector{Vector{Int}}}

# For each current `Ix[idx]`, determine which particles are **discarded** for each
# hyperfine state `(F,mF)` and return a nested container:

# - The result is a `Dict{Int8, Vector{Vector{Int}}}`.
# - For each current index `idx`, `result[Int8(idx)]` is a vector of length `nlevels`
#   (number of `(F,mF)` pairs). Entry `k` is a vector of particle indices `j` that
#   violate either the **SG cavity** constraint or the **post-SG pipe** clearance
#   for level `fmf_levels[k]`.

# Arguments
# - `Ix::AbstractVector`: currents to analyze.
# - `pairs::AbstractMatrix`: `N×6` matrix with columns `[x, y, z, vx, vy, vz]` (SI).
# - `p::AtomParams`: atomic parameters (mass, hyperfine constant, etc.).

# Keywords
# - `t_length::Int=1000`: number of time samples across the SG region per particle.
# - `verbose::Bool=false`: print progress per current.

# Returns
# - `Dict{Int8, Vector{Vector{Int}}}` as described above.

# Notes
# - Assumes in scope: `y_FurnaceToSlit`, `y_SlitToSG`, `y_SG`, `default_R_tube`,
#   and functions: `QM_EqOfMotion_z`, `z_magnet_edge_time`, `z_magnet_trench_time`,
#   `QM_Screen_position`
# - Geometry envelopes (`z_top`, `z_bottom`) are computed once per particle and
#   reused across all `(F,mF)` for that current to avoid redundant work.
# """
# function QM_find_discarded_particles(Ix, pairs, p::AtomParams;
#                                   t_length::Int=1000, verbose::Bool=false)

#     No        = size(pairs, 1)  # number of particles
#     ncurrents = length(Ix)      # number of currents
#     fmf       = fmf_levels(p)    
#     nlevels   = length(fmf)     # number of levels

#     result = OrderedDict{Int8, Vector{Vector{Int}}}()

#     @inbounds for idx in 1:ncurrents
#         i0 = Ix[idx]
#         verbose && println("Analyzing I₀ = $(round(i0, sigdigits=5))A \t (levels = $nlevels)")

#         # One vector-of-vectors per current; one Int[] per (F,mF)
#         bad_for_levels = [Int[] for _ in 1:nlevels]

#         # Sweep particles once; reuse geometry envelopes for all (F,mF)
#         for j in 1:No
#             # Unpack this particle
#             r0 = @view pairs[j, 1:3]
#             v0 = @view pairs[j, 4:6]
#             v_y = v0[2]

#             # Time window inside SG for this particle
#             t_in  = (default_y_FurnaceToSlit + default_y_SlitToSG) / v_y
#             t_out = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v_y
#             t_sweep = range(t_in, t_out; length=t_length)

#             # Geometry bounds (independent of (F,mF))
#             z_top    = z_magnet_edge_time.(t_sweep, Ref(r0), Ref(v0))
#             z_bottom = z_magnet_trench_time.(t_sweep, Ref(r0), Ref(v0))

#             # Evaluate each (F,mF) against this particle
#             for k in 1:nlevels
#                 F, mF = fmf[k]

#                 # Trajectory z(t) depends on (F,mF)
#                 z_val = QM_EqOfMotion_z.(t_sweep, Ref(i0), Ref(F), Ref(mF), Ref(r0), Ref(v0), Ref(p))

#                 # SG cavity check
#                 inside = (z_bottom .< z_val) .& (z_val .< z_top)
#                 if !all(inside)
#                     push!(bad_for_levels[k], j)
#                     continue
#                 end

#                 # Post-SG pipe (screen) check
#                 x_scr, _, z_scr = QM_Screen_position(i0, F, mF, r0, v0, p)
#                 if x_scr^2 + z_scr^2 ≥ default_R_tube^2
#                     push!(bad_for_levels[k], j)
#                     continue
#                 end
#             end
#         end

#         # Optional: sort each level’s list for reproducibility
#         for k in 1:nlevels
#             sort!(bad_for_levels[k])
#         end

#         result[Int8(idx)] = bad_for_levels
#     end

#     return result
# end

# function QM_find_discarded_particles_multithreading(Ix, pairs, p::AtomParams;
#                                         t_length::Int=1000, verbose::Bool=false)

#     No        = size(pairs, 1)            # number of particles
#     ncurrents = length(Ix)                # number of currents
#     fmf       = fmf_levels(p)
#     nlevels   = length(fmf)               # number of (F, mF) levels

#     # Thread-local results bucket (no locking needed because each idx is unique)
#     results = Vector{Vector{Vector{Int}}}(undef, ncurrents)

#     # Optional: serialize prints to avoid interleaving
#     print_lock = ReentrantLock()

#     @threads for idx in 1:ncurrents
#         i0 = Ix[idx]
#         if verbose
#             lock(print_lock); try
#                 println("Analyzing I₀ = $(round(i0, sigdigits=5))A \t (levels = $nlevels)")
#             finally
#                 unlock(print_lock)
#             end
#         end

#         # One vector-of-vectors per current; one Int[] per (F,mF)
#         bad_for_levels = [Int[] for _ in 1:nlevels]

#         # Sweep particles once; reuse geometry envelopes for all (F,mF)
#         @inbounds for j in 1:No
#             # Unpack this particle
#             r0 = @view pairs[j, 1:3]
#             v0 = @view pairs[j, 4:6]
#             v_y = v0[2]

#             # Time window inside SG for this particle
#             t_in   = (default_y_FurnaceToSlit + default_y_SlitToSG) / v_y
#             t_out  = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v_y
#             t_sweep = range(t_in, t_out; length=t_length)

#             # Geometry bounds (independent of (F,mF))
#             z_top    = z_magnet_edge_time.(t_sweep, Ref(r0), Ref(v0))
#             z_bottom = z_magnet_trench_time.(t_sweep, Ref(r0), Ref(v0))

#             # Evaluate each (F,mF) against this particle
#             @inbounds for k in 1:nlevels
#                 F, mF = fmf[k]

#                 # Trajectory z(t) depends on (F,mF)
#                 z_val = QM_EqOfMotion_z.(t_sweep, Ref(i0), Ref(F), Ref(mF), Ref(r0), Ref(v0), Ref(p))

#                 # SG cavity check
#                 inside = (z_bottom .< z_val) .& (z_val .< z_top)
#                 if !all(inside)
#                     push!(bad_for_levels[k], j)
#                     continue
#                 end

#                 # Post-SG pipe (screen) check
#                 x_scr, _, z_scr = QM_Screen_position(i0, F, mF, r0, v0, p)
#                 if x_scr^2 + z_scr^2 ≥ default_R_tube^2
#                     push!(bad_for_levels[k], j)
#                     continue
#                 end
#             end
#         end

#         # Optional: sort each level’s list for reproducibility
#         @inbounds for k in 1:nlevels
#             sort!(bad_for_levels[k])
#         end

#         results[idx] = bad_for_levels
#     end

#     # Stitch into the same OrderedDict shape you had before
#     out = OrderedDict{Int8, Vector{Vector{Int}}}()
#     @inbounds for idx in 1:ncurrents
#         out[Int8(idx)] = results[idx]
#     end
#     return out
# end



# """
#     build_filtered_pairs(pairs, bad; copy=false) -> Dict{Int8, Vector{<:AbstractMatrix}}

# From the original `pairs::N×6` matrix and the `bad` index lists
# `bad[Int8(idx)][k]` (particles to discard for current `idx` and level `k`),
# build a ragged container:

# - key:   current index as `Int8` (1..length(Ix))
# - value: `Vector` of length `nlevels`, where element `k` is a matrix of the
#          **kept** rows (particles) for level `k` at that current.

# By default it returns **views** into `pairs` (no copying). Set `copy=true` to
# materialize independent matrices.

# Notes
# - Works even if a level keeps zero rows (returns an empty `0×size(pairs,2)` matrix).
# - The relative order of rows is preserved.
# """
# function QM_build_filtered_pairs(pairs::AbstractMatrix{T},
#                               bad::OrderedDict{Int8, Vector{Vector{Int}}};
#                               copy::Bool=false) where {T}
#     No = size(pairs, 1)
#     out = OrderedDict{Int8, Vector{AbstractMatrix{T}}}()

#     for (idx, bad_levels) in bad
#         nlevels = length(bad_levels)
#         mats = Vector{AbstractMatrix{T}}(undef, nlevels)

#         @inbounds for k in 1:nlevels
#             b = bad_levels[k]                 # indices to drop (assumed sorted, but not required)
#             # Build a keep mask (faster than setdiff for large N)
#             keepmask = trues(No)
#             for j in b
#                 keepmask[j] = false
#             end
#             keep = findall(keepmask)

#             mats[k] = copy ? pairs[keep, :] : view(pairs, keep, :)
#         end

#         out[idx] = mats
#     end

#     return out
# end

# """
#     QM_build_alive_screen(Ix, pairs, bad, p) 
#         -> OrderedDict{Int8, Vector{Matrix{T}}}

# For each current index `idx` and each hyperfine level `(F,mF) = fmf[k]`, build the
# matrix of **alive** particles (rows from `pairs` not listed in `bad[idx][k]`), and
# append the **screen** quantities as three new columns:
# `x_scr`, `z_scr` from `QM_Screen_position`, and `vz_scr` from `QM_Screen_velocity`.

# Output shape
# - `OrderedDict{Int8, Vector{Matrix{T}}}`:
#   - key: current index `Int8` (1-based index into `Ix`)
#   - value: a vector of length `length(fmf)`; element `k` is a `(#alive, 9)` matrix with
#     columns `[x, y, z, vx, vy, vz, x_scr, z_scr, vz_scr]`.

# Arguments
# - `Ix::AbstractVector{<:Real}`            : currents (A), length = ncurrents.
# - `pairs::AbstractMatrix{T}`              : `N×6` base matrix (`x y z vx vy vz`) in SI units.
# - `bad::OrderedDict{Int8, Vector{Vector{Int}}}` : for each current index, a vector of
#   *drop* indices per level (your finder’s result).
# - `p::AtomParams`                          : atomic parameters.

# Notes
# - Uses `QM_Screen_position(I, F, mF, r0, v0, p)` and
#   `QM_Screen_velocity(I, F, mF, v0, p)`. Adapt the signatures here if yours differ.
# - Returns **fresh matrices** (not views), because of the hcat of computed columns.
# - If a level keeps zero rows, returns a `0×9` matrix for that slot.

# Columns
# `[x, y, z, vx, vy, vz, x_scr, z_scr, vz_scr]`

# Example
# ```julia
# alive_with_screen = QM_build_alive_screen(Icoils, pairs, bad, p)
# M = alive_with_screen[Int8(3)][1]  # current idx=3, level k=1; size == (#alive, 9)
# """
# function QM_build_alive_screen(Ix::AbstractVector{<:Real},
#             pairs::AbstractMatrix{T},
#             bad::OrderedDict{Int8, Vector{Vector{Int}}},
#             p::AtomParams) where {T}

#     No      = size(pairs, 1)
#     fmf     = fmf_levels(p)
#     nlevels = length(fmf)
#     out     = OrderedDict{Int8, Vector{Matrix{T}}}()

#     # Bind function handles locally (avoid global lookups in hot loops)
#     screen_pos = QM_Screen_position
#     screen_vel = QM_Screen_velocity

#     for (kcurr, drop_per_level) in bad
#         idx = Int(kcurr)  # 1-based into Ix
#         I0  = float(Ix[idx])

#         mats = Vector{Matrix{T}}(undef, nlevels)

#         @inbounds for k in 1:nlevels
#             F, mF = fmf[k]
#             # Build keep list for this (current, level)
#             keepmask = trues(No)
#             for j in drop_per_level[k]
#                 keepmask[j] = false
#             end
#             keep = findall(keepmask)
#             m = length(keep)

#             # Allocate output matrix: original 6 + 3 screen columns
#             Mout = Matrix{T}(undef, m, 9)
#             # Copy base columns
#             if m > 0
#                 Mout[:, 1:6] .= pairs[keep, :]
#                 # Compute screen cols per row
#                 for i in 1:m
#                     # r0 = x,y,z ; v0 = vx,vy,vz as 1×3 views
#                     @views r0 = Mout[i, 1:3]
#                     @views v0 = Mout[i, 4:6]

#                     # position at screen: (x_scr, _, z_scr)
#                     x_scr, _, z_scr = screen_pos(I0, F, mF, r0, v0, p)
#                     # velocity at screen: (_, _, vz_scr)
#                     _, _, vz_scr    = screen_vel(I0, F, mF, v0, p)

#                     Mout[i, 7] = T(x_scr)
#                     Mout[i, 8] = T(z_scr)
#                     Mout[i, 9] = T(vz_scr)
#                 end
#             else
#                 # no rows → make an empty 0×9 to keep ragged shape consistent
#                 Mout = Matrix{T}(undef, 0, 9)
#             end

#             mats[k] = Mout
#         end

#     out[Int8(idx)] = mats
#     end

#     return out
# end

#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################


"""
    QM_flag_travelling_particles(
        Ix, init_particles, p::AtomParams; y_length::Int=1000, verbose::Bool=false
    ) -> OrderedDict{Int8, Vector{Vector{UInt8}}}

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
  `y ∈ [y_in, y_out]`, where `y_in = default_y_FurnaceToSlit + default_y_SlitToSG`
  and `y_out = y_in + default_y_SG`.
- `verbose::Bool=false`: If `true`, prints a status line per current.

# Returns
`OrderedDict{Int8, Vector{Vector{UInt8}}}` mapping **current index**
`Int8(1:length(Ix))` → vector over `(F, mF)` levels → `Vector{UInt8}` of length `No`
(containing the code for each particle).

# Performance notes
- Work is `Threads.@threads` over currents; per-current work is independent.
- Time ~ `length(Ix) * nlevels * No * y_length`.
- Result memory ~ `ncurrents * nlevels * No` bytes (one `UInt8` per particle).
- Best performance when:
  - `Ix::Vector{Float64}` and `init_particles::Matrix{Float64}` (avoids per-element casts).
  - Geometry globals are set **before** running and not mutated during execution.
  - The inner kernel `QM_cavity_crash` is called positionally (no kwargs).

# Example
```julia
Ix    = collect(range(-20.0, 20.0; length=5))
init_particles = randn(10_000, 6) .* [1e-3, 0.10, 1e-3, 1e-3, 100.0, 1e-3]'
out   = QM_flag_travelling_particles(Ix, init_particles, K39_params;
                                                               y_length=1024, verbose=true)

# Codes for current index 1, level 3:
codes  = out[Int8(1)][3]               # Vector{UInt8} of length size(init_particles,1)
n_top  = count(==(0x01), codes)
n_bot  = count(==(0x02), codes)
n_scr  = count(==(0x03), codes)
n_pass = count(==(0x00), codes)
"""
function QM_flag_travelling_particles(Ix, init_particles, p::AtomParams;
                                                    y_length::Int=1000,
                                                    verbose::Bool=false)

    No        = size(init_particles, 1)      # number of particles
    ncurrents = length(Ix)          # number of currents
    fmf       = fmf_levels(p)
    nlevels   = length(fmf)         # number of (F, mF) levels

    # --- Precompute and share the y-grid for the SG span ---
    y_in  = (default_y_FurnaceToSlit + default_y_SlitToSG)::Float64
    y_out = (y_in + default_y_SG)::Float64
    ygrid = range(y_in, y_out; length=y_length)

    # Thread-local results bucket (each idx is unique, so no locking needed)
    results = Vector{Vector{Vector{UInt8}}}(undef, ncurrents)

    # Optional: serialize prints to avoid interleaving
    print_lock = ReentrantLock()

    # NEW: global progress counter, safe across threads
    progress = Threads.Atomic{Int}(0)

    Threads.@threads for idx in 1:ncurrents
        i0 = Ix[idx]

        if verbose
            # increment progress *before* doing the work
            c = Threads.atomic_add!(progress, 1) + 1
            lock(print_lock); 
            try
                # println("Analyzing I₀ = $(round(i0, sigdigits=5)) A \t (levels = $nlevels)")
                @printf "[%02d/%d]\tAnalyzing I₀ = %.3f A \t (levels = %d)\n" c ncurrents i0 nlevels
            finally
                unlock(print_lock)
            end
        end

        # For this current: one Int8[No] vector per (F,mF), initialized to 0
        codes_for_levels = [fill(UInt8(0x00), No) for _ in 1:nlevels]

        # --- hoist G(I) once per current ---
        gI = GvsI(i0)

        # --- loop levels outer, particles inner (μ computed once per level) ---
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

    # Assemble as an OrderedDict keyed by current index (Int8)
    out = OrderedDict{Int8, Vector{Vector{UInt8}}}()
    @inbounds for idx in 1:ncurrents
        out[Int8(idx)] = results[idx]
    end
    return out
end

"""
    QM_build_travelling_particles(Ix, init_particles, flagged_particles, p) -> OrderedDict{Int8, Vector{Matrix{Float64}}}

For each coil current index and each `(F, mF)` level, propagate all input particles
from the SG entrance to the screen and return a per-level matrix of results.

# Inputs
- `Ix::Vector{Float64}`: Coil currents (A). Indices `1:length(Ix)` label each "current".
- `init_particles::Matrix{Float64}`: `No × 6` matrix of initial conditions (rows = particles) with columns
  `[x0, y0, z0, v0x, v0y, v0z]` in SI units.
- `flagged_particles::OrderedDict{Int8, Vector{Vector{UInt8}}}`:
  For each current index `idx::Int8`, a vector of length `nlevels` where element `k`
  is a `Vector{UInt8}` of length `No` with a per-particle **flag** for level `k`
  (e.g. 0=pass, 1=top, 2=bottom, 3=tube). These flags are written into column 10.
- `p::AtomParams`: Atom/beam parameters. Used here via
  `fmf_levels(p)`, `μF_effective(I0, F, mF, p)`, and `p.M` (mass).

# Geometry & dynamics
Uses global geometry (all in meters):  
`default_y_FurnaceToSlit`, `default_y_SlitToSG`, `default_y_SG`, `default_y_SGToScreen`.  
Defines total drift `Ltot = y_in + LSG + Ld` and uses
`Δ = LSG^2 + 2 LSG*Ld` inside the kinematics helper `screen_x_z_vz`.

The transverse acceleration for each (current, level) is
`a_z = μF_effective(I0, F, mF, p) * GvsI(I0) / p.M`.

# Output
Returns `OrderedDict{Int8, Vector{Matrix{Float64}}}`. For each current index `idx`,
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
10. `flag`     (from `flagged_particles[idx][k][j]`, stored as `Float64`)

Row order is preserved.

# Notes
- Expects `length(flagged_particles[idx][k]) == size(init_particles, 1)` for all `idx` and `k`.
- `screen_x_z_vz` must be available in scope and uses SI units.
"""
function QM_build_travelling_particles(
    Ix::Vector{Float64},
    init_particles::Matrix{Float64},
    flagged_trajec::OrderedDict{Int8, Vector{Vector{UInt8}}},
    p::AtomParams
)
    No      = size(init_particles, 1)
    fmf     = fmf_levels(p)
    nlevels = length(fmf)

    # Bind geometry once (typed)
    y_in = (default_y_FurnaceToSlit + default_y_SlitToSG)::Float64
    Lsg  = default_y_SG::Float64
    Ld   = default_y_SGToScreen::Float64
    Ltot = (y_in + Lsg + Ld)::Float64
    Δ    = (Lsg*Lsg + 2.0*Lsg*Ld)::Float64  # (Lsg+Ld)^2 - Ld^2

    out = OrderedDict{Int8, Vector{Matrix{Float64}}}()

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

            M = Matrix{Float64}(undef, No, 10)

            # copy the 6 initial columns (column-wise is cheap, no views created)
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
                x,z,vz = screen_x_z_vz(x0,z0,v0x,v0y,v0z, Lsg, Δ, Ltot, a_z)
                M[j,7]  = x;  
                M[j,8]  = z;  
                M[j,9]  = vz
                M[j,10] = Float64(flags_k[j])   # keep your layout
            end

            mats[k] = M
        end

        out[Int8(idx)] = mats
    end

    return out
end


"""
    travelling_particles_summary(Ixs, q_numbers, particles) -> nothing

Pretty-prints, for each coil current `I₀` in `Ixs`, a per–(F,mF) summary of
particle outcomes based on the **flag in column 10** of each particle matrix.

For every current `I₀` and every quantum level `(F, mF)`, the function counts:
- `Pass`   — number of rows with flag `0` in column 10
- `Top`    — number of rows with flag `1`
- `Bottom` — number of rows with flag `2`
- `Tube`   — number of rows with flag `3`

It then prints a table (via PrettyTables) with columns:
`F, mF, Pass, Top, Bottom, Tube, Total, Pass %, Loss %`,
where `Loss % = 100 * (Top + Bottom + Tube) / Total`.
When `Total == 0`, both percentages are shown as `0.0`.

# Arguments
- `Ixs::AbstractVector`: Coil currents `I₀` (A). One summary table is printed per entry.
- `q_numbers::AbstractVector{<:Tuple}}`: Quantum numbers `(F, mF)` for the levels; its length
  defines `nlevels`.
- `particles`: A vector of length `length(Ixs)`. For each `i`, `particles[i]` is a
  vector of length `nlevels` containing the particle matrices for current `Ixs[i]`.
  Each matrix must be `N×10`; only **column 10** is used here and must contain the
  outcome flag (either `Int` {0,1,2,3} or `Float64` {0.0,1.0,2.0,3.0}).

# Behavior
- Prints one formatted table per `I₀` using PrettyTables (with rounded unicode borders,
  centered alignment, custom colors/styles as configured in the call).
- Inserts a blank line between tables.
- Returns `nothing`.

# Notes
- Column 10 may be `Integer` or `Float64`; both are supported.
- This routine assumes `length(particles[i]) == length(q_numbers)` for all `i`.
- Formatting helpers like `fmt__printf` are expected to be in scope (adapt if using
  PrettyTables v3 formatters/highlighters directly).

# Example
```julia
Ixs = [0.10, 0.20]
q_numbers = [(2, 1), (2, 0)]
M1 = rand(1_000, 10); M1[:,10] .= rand([0,1,2,3], size(M1,1))
M2 = rand( 800, 10); M2[:,10] .= rand([0,1,2,3], size(M2,1))
M3 = rand(1200, 10); M3[:,10] .= rand([0,1,2,3], size(M3,1))
M4 = rand( 900, 10); M4[:,10] .= rand([0,1,2,3], size(M4,1))

particles = [
    [M1, M2],  # for Ixs[1]
    [M3, M4],  # for Ixs[2]
]

travelling_particles_summary(Ixs, q_numbers, particles)
"""
function travelling_particles_summary(Ixs, q_numbers, particles)

    nlevels = length(q_numbers)

    # count flags in column 10; supports Float or Integer flags
    @inline function counts_from_M(M::AbstractMatrix)
        @views col = M[:,end]
        if eltype(col) <: Integer
            pass = count(==(0),  col); top = count(==(1), col)
            bot  = count(==(2),  col); tub = count(==(3), col)
        else
            pass = count(==(0.0), col); top = count(==(1.0), col)
            bot  = count(==(2.0), col); tub = count(==(3.0), col)
        end
        return (pass=pass, top=top, bot=bot, tub=tub)
    end

    for i in eachindex(Ixs)
        I0   = Float64(Ixs[i])
        mats = particles[i]  # Vector of No×10 matrices, one per level

        nrows = nlevels
        data  = Matrix{Float64}(undef, nrows, 9)

        # rows per level
        for j in 1:nlevels
            F, mF = q_numbers[j]
            M     = mats[j]
            c     = counts_from_M(M)
            tot   = c.pass + c.top + c.bot + c.tub

            passp = tot == 0 ? 0.0 : 100.0 * c.pass / tot
            lossp = tot == 0 ? 0.0 : 100.0 * (c.top + c.bot + c.tub) / tot

            data[j, :] = [Float64(F), Float64(mF),
                          c.pass, c.top, c.bot, c.tub,
                          tot, passp, lossp]
        end

        pretty_table(
            data;
            column_labels = ["F","mF","Pass","Top","Bottom","Tube","Total","Pass %","Loss %"],
            title = @sprintf("I₀ = %.3f A", I0),
            formatters    = [fmt__printf("%d", 3:7), fmt__printf("%5.1f", 8:9)],
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
    select_flagged(initial_by_current, which::Symbol; flagcol::Int=10)
        -> OrderedDict{Int8, Vector{Matrix{T}}}

Filter particle matrices by the flag in column `flagcol` (default 10),
returning a **new** dictionary with the same structure but **only** the rows
matching the requested category. Rows are **copied** (no views).

Categories:
- `:screen`     → keep rows with flag == 0
- `:crash_SG`   → keep rows with flag == 1 or 2
- `:crash_tube` → keep rows with flag == 3

# Arguments
- `initial_by_current::OrderedDict{Int8, Vector{<:AbstractMatrix{T}}}`:
  For each current index, a vector of per-level `N×10` matrices.
- `which::Symbol`: one of `:screen`, `:crash_SG`, `:crash_tube`.
- `flagcol::Int=10`: 1-based column index holding the flag.

# Returns
`OrderedDict{Int8, Vector{Matrix{T}}}` with the same keys and number of
levels, where each matrix contains **only** the kept rows. If no rows match,
the matrix will be `0×D`.

# Notes
- Works whether the flag column is `Int`, `UInt8`, or `Float64`
  (`0 == 0.0` is true in Julia).
- Copies are made (dense `Matrix{T}`), so subsequent mutations won’t affect the
  originals.

# Example
```julia
only_screen = select_flagged(particles_trajectories, :screen)      # flag 0
only_crash  = select_flagged(particles_trajectories, :crash_SG)    # flags 1 or 2
only_tube   = select_flagged(particles_trajectories, :crash_tube)  # flag 3

only_screen[1][3]  # rows for current 1, level 3 with flag == 0
"""
function select_flagged(initial_by_current::OrderedDict{Int8, Vector{Matrix{Float64}}},which::Symbol; flagcol::Int=10) 
    flagset = which === :screen     ? (0,)      :
              which === :crash_SG   ? (1, 2)    :
              which === :crash_tube ? (3,)      :
              which === :crash      ? (1,2,3)   :
              which === :all_in     ? (0,1,2,3) :
          error("which must be :screen, :crash_SG, :crash_tube, or :crash")

    out = OrderedDict{Int8, Vector{Matrix{Float64}}}()

    for (idx, mats) in initial_by_current
        nlevels = length(mats)
        v = Vector{Matrix{Float64}}(undef, nlevels)

        @inbounds for k in 1:nlevels
            M = mats[k]
            @assert 1 ≤ flagcol ≤ size(M,2) "flagcol out of bounds"
            col = @view M[:, flagcol]

            # keep rows where flag ∈ flagset (works for 1, 2, or 3 values)
            keep = findall(in.(col, Ref(flagset)))
            # Alternatively (avoids building index vector):
            # mask = in.(col, Ref(flagset))
            # v[k] = M[mask, :]

            v[k] = M[keep, 1:flagcol-1]   # copy rows into a dense Matrix
        end

        out[idx] = v
    end

    return out
end