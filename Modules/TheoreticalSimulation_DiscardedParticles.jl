"""
    QM_find_bad_particles_ix(Ix, pairs, p::AtomParams;
                             t_length=1000, verbose=false)
        -> Dict{Int8, Vector{Vector{Int}}}

For each current `Ix[idx]`, determine which particles are **discarded** for each
hyperfine state `(F,mF)` and return a nested container:

- The result is a `Dict{Int8, Vector{Vector{Int}}}`.
- For each current index `idx`, `result[Int8(idx)]` is a vector of length `nlevels`
  (number of `(F,mF)` pairs). Entry `k` is a vector of particle indices `j` that
  violate either the **SG cavity** constraint or the **post-SG pipe** clearance
  for level `fmf_levels[k]`.

Arguments
- `Ix::AbstractVector`: currents to analyze.
- `pairs::AbstractMatrix`: `N×6` matrix with columns `[x, y, z, vx, vy, vz]` (SI).
- `p::AtomParams`: atomic parameters (mass, hyperfine constant, etc.).

Keywords
- `t_length::Int=1000`: number of time samples across the SG region per particle.
- `verbose::Bool=false`: print progress per current.

Returns
- `Dict{Int8, Vector{Vector{Int}}}` as described above.

Notes
- Assumes in scope: `y_FurnaceToSlit`, `y_SlitToSG`, `y_SG`, `default_R_tube`,
  and functions: `QM_EqOfMotion_z`, `z_magnet_edge_time`, `z_magnet_trench_time`,
  `QM_Screen_position`
- Geometry envelopes (`z_top`, `z_bottom`) are computed once per particle and
  reused across all `(F,mF)` for that current to avoid redundant work.
"""
function QM_find_discarded_particles(Ix, pairs, p::AtomParams;
                                  t_length::Int=1000, verbose::Bool=false)

    No        = size(pairs, 1)  # number of particles
    ncurrents = length(Ix)      # number of currents
    fmf       = fmf_levels(p)    
    nlevels   = length(fmf)     # number of levels

    result = OrderedDict{Int8, Vector{Vector{Int}}}()

    @inbounds for idx in 1:ncurrents
        i0 = Ix[idx]
        verbose && println("Analyzing I₀ = $(round(i0, sigdigits=5))A \t (levels = $nlevels)")

        # One vector-of-vectors per current; one Int[] per (F,mF)
        bad_for_levels = [Int[] for _ in 1:nlevels]

        # Sweep particles once; reuse geometry envelopes for all (F,mF)
        for j in 1:No
            # Unpack this particle
            r0 = @view pairs[j, 1:3]
            v0 = @view pairs[j, 4:6]
            v_y = v0[2]

            # Time window inside SG for this particle
            t_in  = (default_y_FurnaceToSlit + default_y_SlitToSG) / v_y
            t_out = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v_y
            t_sweep = range(t_in, t_out; length=t_length)

            # Geometry bounds (independent of (F,mF))
            z_top    = z_magnet_edge_time.(t_sweep, Ref(r0), Ref(v0))
            z_bottom = z_magnet_trench_time.(t_sweep, Ref(r0), Ref(v0))

            # Evaluate each (F,mF) against this particle
            for k in 1:nlevels
                F, mF = fmf[k]

                # Trajectory z(t) depends on (F,mF)
                z_val = QM_EqOfMotion_z.(t_sweep, Ref(i0), Ref(F), Ref(mF), Ref(r0), Ref(v0), Ref(p))

                # SG cavity check
                inside = (z_bottom .< z_val) .& (z_val .< z_top)
                if !all(inside)
                    push!(bad_for_levels[k], j)
                    continue
                end

                # Post-SG pipe (screen) check
                x_scr, _, z_scr = QM_Screen_position(i0, F, mF, r0, v0, p)
                if x_scr^2 + z_scr^2 ≥ default_R_tube^2
                    push!(bad_for_levels[k], j)
                    continue
                end
            end
        end

        # Optional: sort each level’s list for reproducibility
        for k in 1:nlevels
            sort!(bad_for_levels[k])
        end

        result[Int8(idx)] = bad_for_levels
    end

    return result
end

function QM_find_discarded_particles_multithreading(Ix, pairs, p::AtomParams;
                                        t_length::Int=1000, verbose::Bool=false)

    No        = size(pairs, 1)            # number of particles
    ncurrents = length(Ix)                # number of currents
    fmf       = fmf_levels(p)
    nlevels   = length(fmf)               # number of (F, mF) levels

    # Thread-local results bucket (no locking needed because each idx is unique)
    results = Vector{Vector{Vector{Int}}}(undef, ncurrents)

    # Optional: serialize prints to avoid interleaving
    print_lock = ReentrantLock()

    @threads for idx in 1:ncurrents
        i0 = Ix[idx]
        if verbose
            lock(print_lock); try
                println("Analyzing I₀ = $(round(i0, sigdigits=5))A \t (levels = $nlevels)")
            finally
                unlock(print_lock)
            end
        end

        # One vector-of-vectors per current; one Int[] per (F,mF)
        bad_for_levels = [Int[] for _ in 1:nlevels]

        # Sweep particles once; reuse geometry envelopes for all (F,mF)
        @inbounds for j in 1:No
            # Unpack this particle
            r0 = @view pairs[j, 1:3]
            v0 = @view pairs[j, 4:6]
            v_y = v0[2]

            # Time window inside SG for this particle
            t_in   = (default_y_FurnaceToSlit + default_y_SlitToSG) / v_y
            t_out  = (default_y_FurnaceToSlit + default_y_SlitToSG + default_y_SG) / v_y
            t_sweep = range(t_in, t_out; length=t_length)

            # Geometry bounds (independent of (F,mF))
            z_top    = z_magnet_edge_time.(t_sweep, Ref(r0), Ref(v0))
            z_bottom = z_magnet_trench_time.(t_sweep, Ref(r0), Ref(v0))

            # Evaluate each (F,mF) against this particle
            @inbounds for k in 1:nlevels
                F, mF = fmf[k]

                # Trajectory z(t) depends on (F,mF)
                z_val = QM_EqOfMotion_z.(t_sweep, Ref(i0), Ref(F), Ref(mF), Ref(r0), Ref(v0), Ref(p))

                # SG cavity check
                inside = (z_bottom .< z_val) .& (z_val .< z_top)
                if !all(inside)
                    push!(bad_for_levels[k], j)
                    continue
                end

                # Post-SG pipe (screen) check
                x_scr, _, z_scr = QM_Screen_position(i0, F, mF, r0, v0, p)
                if x_scr^2 + z_scr^2 ≥ default_R_tube^2
                    push!(bad_for_levels[k], j)
                    continue
                end
            end
        end

        # Optional: sort each level’s list for reproducibility
        @inbounds for k in 1:nlevels
            sort!(bad_for_levels[k])
        end

        results[idx] = bad_for_levels
    end

    # Stitch into the same OrderedDict shape you had before
    out = OrderedDict{Int8, Vector{Vector{Int}}}()
    @inbounds for idx in 1:ncurrents
        out[Int8(idx)] = results[idx]
    end
    return out
end



"""
    build_filtered_pairs(pairs, bad; copy=false) -> Dict{Int8, Vector{<:AbstractMatrix}}

From the original `pairs::N×6` matrix and the `bad` index lists
`bad[Int8(idx)][k]` (particles to discard for current `idx` and level `k`),
build a ragged container:

- key:   current index as `Int8` (1..length(Ix))
- value: `Vector` of length `nlevels`, where element `k` is a matrix of the
         **kept** rows (particles) for level `k` at that current.

By default it returns **views** into `pairs` (no copying). Set `copy=true` to
materialize independent matrices.

Notes
- Works even if a level keeps zero rows (returns an empty `0×size(pairs,2)` matrix).
- The relative order of rows is preserved.
"""
function QM_build_filtered_pairs(pairs::AbstractMatrix{T},
                              bad::OrderedDict{Int8, Vector{Vector{Int}}};
                              copy::Bool=false) where {T}
    No = size(pairs, 1)
    out = OrderedDict{Int8, Vector{AbstractMatrix{T}}}()

    for (idx, bad_levels) in bad
        nlevels = length(bad_levels)
        mats = Vector{AbstractMatrix{T}}(undef, nlevels)

        @inbounds for k in 1:nlevels
            b = bad_levels[k]                 # indices to drop (assumed sorted, but not required)
            # Build a keep mask (faster than setdiff for large N)
            keepmask = trues(No)
            for j in b
                keepmask[j] = false
            end
            keep = findall(keepmask)

            mats[k] = copy ? pairs[keep, :] : view(pairs, keep, :)
        end

        out[idx] = mats
    end

    return out
end

"""
    QM_build_alive_screen(Ix, pairs, bad, p) 
        -> OrderedDict{Int8, Vector{Matrix{T}}}

For each current index `idx` and each hyperfine level `(F,mF) = fmf[k]`, build the
matrix of **alive** particles (rows from `pairs` not listed in `bad[idx][k]`), and
append the **screen** quantities as three new columns:
`x_scr`, `z_scr` from `QM_Screen_position`, and `vz_scr` from `QM_Screen_velocity`.

Output shape
- `OrderedDict{Int8, Vector{Matrix{T}}}`:
  - key: current index `Int8` (1-based index into `Ix`)
  - value: a vector of length `length(fmf)`; element `k` is a `(#alive, 9)` matrix with
    columns `[x, y, z, vx, vy, vz, x_scr, z_scr, vz_scr]`.

Arguments
- `Ix::AbstractVector{<:Real}`            : currents (A), length = ncurrents.
- `pairs::AbstractMatrix{T}`              : `N×6` base matrix (`x y z vx vy vz`) in SI units.
- `bad::OrderedDict{Int8, Vector{Vector{Int}}}` : for each current index, a vector of
  *drop* indices per level (your finder’s result).
- `p::AtomParams`                          : atomic parameters.

Notes
- Uses `QM_Screen_position(I, F, mF, r0, v0, p)` and
  `QM_Screen_velocity(I, F, mF, v0, p)`. Adapt the signatures here if yours differ.
- Returns **fresh matrices** (not views), because of the hcat of computed columns.
- If a level keeps zero rows, returns a `0×9` matrix for that slot.

Columns
`[x, y, z, vx, vy, vz, x_scr, z_scr, vz_scr]`

Example
```julia
alive_with_screen = QM_build_alive_screen(Icoils, pairs, bad, p)
M = alive_with_screen[Int8(3)][1]  # current idx=3, level k=1; size == (#alive, 9)
"""
function QM_build_alive_screen(Ix::AbstractVector{<:Real},
            pairs::AbstractMatrix{T},
            bad::OrderedDict{Int8, Vector{Vector{Int}}},
            p::AtomParams) where {T}

    No      = size(pairs, 1)
    fmf     = fmf_levels(p)
    nlevels = length(fmf)
    out     = OrderedDict{Int8, Vector{Matrix{T}}}()

    # Bind function handles locally (avoid global lookups in hot loops)
    screen_pos = QM_Screen_position
    screen_vel = QM_Screen_velocity

    for (kcurr, drop_per_level) in bad
        idx = Int(kcurr)  # 1-based into Ix
        I0  = float(Ix[idx])

        mats = Vector{Matrix{T}}(undef, nlevels)

        @inbounds for k in 1:nlevels
            F, mF = fmf[k]
            # Build keep list for this (current, level)
            keepmask = trues(No)
            for j in drop_per_level[k]
                keepmask[j] = false
            end
            keep = findall(keepmask)
            m = length(keep)

            # Allocate output matrix: original 6 + 3 screen columns
            Mout = Matrix{T}(undef, m, 9)
            # Copy base columns
            if m > 0
                Mout[:, 1:6] .= pairs[keep, :]
                # Compute screen cols per row
                for i in 1:m
                    # r0 = x,y,z ; v0 = vx,vy,vz as 1×3 views
                    @views r0 = Mout[i, 1:3]
                    @views v0 = Mout[i, 4:6]

                    # position at screen: (x_scr, _, z_scr)
                    x_scr, _, z_scr = screen_pos(I0, F, mF, r0, v0, p)
                    # velocity at screen: (_, _, vz_scr)
                    _, _, vz_scr    = screen_vel(I0, F, mF, v0, p)

                    Mout[i, 7] = T(x_scr)
                    Mout[i, 8] = T(z_scr)
                    Mout[i, 9] = T(vz_scr)
                end
            else
                # no rows → make an empty 0×9 to keep ragged shape consistent
                Mout = Matrix{T}(undef, 0, 9)
            end

            mats[k] = Mout
        end

    out[Int8(idx)] = mats
    end

    return out
end