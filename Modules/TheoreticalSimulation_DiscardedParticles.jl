########################################################################################################################################
# Quantum Mechanics : Classical Trajectories
########################################################################################################################################

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
    @info "Evaluating particle trajectories and assigning flags"
    No        = size(init_particles, 1) # number of particles
    ncurrents = length(Ix)              # number of currents
    fmf       = fmf_levels(p)           # quantum numbers
    nlevels   = length(fmf)             # number of (F, mF) levels

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
`Δ = LSG^2 + 2 LSG*Ld` inside the kinematics helper `QM_screen_x_z_vz`.

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
- `QM_screen_x_z_vz` must be available in scope and uses SI units.
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
                x,z,vz = QM_screen_x_z_vz(x0,z0,v0x,v0y,v0z, Lsg, Δ, Ltot, a_z)
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
            aper = count(==(4),  col);
        else
            pass = count(==(0.0), col); top = count(==(1.0), col)
            bot  = count(==(2.0), col); tub = count(==(3.0), col)
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
    select_flagged(initial_by_current, which::Symbol; flagcol::Int=10)
        -> OrderedDict{Int8, Vector{Matrix{Float64}}}

Filter particle matrices by the flag stored in column `flagcol` (default `10`),
returning a **new** dictionary with the same keys and number of levels but with
**only the rows** whose flag matches the requested category. Rows are **copied**
(dense matrices), not views.

!!! note
    The returned matrices **exclude** the flag column and any columns to its
    right: only columns `1:flagcol-1` are kept.

## Categories
- `:screen`      → keep rows with flag == 0
- `:crash_SG`    → keep rows with flag ∈ {1, 2}
- `:crash_tube`  → keep rows with flag == 3
- `:crash`       → keep rows with flag ∈ {1, 2, 3}
- `:all`         → keep rows with flag ∈ {0, 1, 2, 3}

## Arguments
- `initial_by_current::OrderedDict{Int8, Vector{Matrix{Float64}}}`:
  For each current (key `Int8`), a vector of per-level `N×D` matrices (typically `D ≥ flagcol`).
- `which::Symbol`: one of `:screen`, `:crash_SG`, `:crash_tube`, `:crash`, `:all`.
- `flagcol::Int=10`: 1-based column index holding the flag (must satisfy `1 ≤ flagcol ≤ D`).

## Returns
`OrderedDict{Int8, Vector{Matrix{Float64}}}` with the same keys and number of
levels. Each matrix contains only the **kept rows** and columns `1:flagcol-1`.
If no rows match for a given level, its matrix will be `0×(flagcol-1)`.

## Notes
- Works even if flags are stored as floating-point values (e.g., `0.0`, `1.0`),
  since equality like `0 == 0.0` holds in Julia.
- All outputs are fresh copies (dense `Matrix{Float64}`), so later mutations
  won’t affect the input matrices.

## Example
```julia
only_screen = select_flagged(particles_trajectories, :screen)      # flag == 0
only_crash  = select_flagged(particles_trajectories, :crash)       # flags 1,2,3
all_flags   = select_flagged(particles_trajectories, :all)         # flags 0–3

# Access rows for current key 1, level 3 (columns 1:flagcol-1 are kept)
only_screen[1][3]

"""
function select_flagged(initial_by_current::OrderedDict{K, Vector{Matrix{Float64}}},which::Symbol; flagcol::Integer=10) where {K<:Integer}
    flagset = which === :screen     ? (0,)          :
              which === :crash_SG   ? (1, 2)        :
              which === :crash_tube ? (3,)          :
              which === :crash_aper ? (4,)          :
              which === :crash      ? (1,2,3,4)     :
              which === :all        ? (0,1,2,3,4)   : 
          error("which must be :screen, :crash_SG, :crash_tube, :cras_aper, :crash, or :all")

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

########################################################################################################################################
# Co-Quantum Dynamics
########################################################################################################################################

function CQD_flag_travelling_particles(Ix, init_particles, kx::Float64, p::AtomParams;
                                                    y_length::Int=1000,
                                                    verbose::Bool=false)
    @info "Evaluating particle trajectories and assigning flags"
    @assert size(init_particles, 2) ≥ 8 "init_particles must have at least 8 columns"
    No        = size(init_particles, 1) # number of particles
    ncurrents = length(Ix)              # number of currents

    # --- Precompute and share the y-grid for the SG span ---
    y_in  = (default_y_FurnaceToSlit + default_y_SlitToSG)::Float64
    y_out = (y_in + default_y_SG)::Float64
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

        # Hoist field/gradient for this current (adjust BvsI/GvsI names to your codebase)
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

            codes[j] = CQD_cavity_crash(μG,B0,x0,y0,z0,v0x,v0y,v0z,θe,θn,kx,p,ygrid,0.0)
        end

        results[idx] = codes
    end

    return results
end

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
    y_in = (default_y_FurnaceToSlit + default_y_SlitToSG)::Float64
    Lsg  = default_y_SG::Float64
    Ld   = default_y_SGToScreen::Float64
    Ltot = (y_in + Lsg + Ld)::Float64
    ΔL   = (Lsg*Lsg + 2.0*Lsg*Ld)::Float64  # (Lsg+Ld)^2 - Ld^2

    out = OrderedDict{Int8, Matrix{Float64}}()

    @inbounds for idx in eachindex(Ix)
        I0      = Ix[idx]
        flags_i = flagged_trajec[idx]
        @assert length(flags_i) == No "flags[idx] length must equal number of particles"

        # a_z for this current
        μG   = μₑ * GvsI(I0)
        a_z  = μG / p.M
        ωL    = abs(γₑ * BvsI(I0))
        
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
- `:screen`     → {0}
- `:crash_SG`   → {1, 2}
- `:crash_tube` → {3}
- `:crash`      → {1, 2, 3}
- `:all`        → {0, 1, 2, 3}

# Arguments
- `initial_by_current::OrderedDict{K, Matrix{Float64}}` where `K<:Integer`
- `which::Symbol`: one of the above
- `flagcol::Integer=12`: 1-based index of the flag column

# Returns
- `OrderedDict{K, Matrix{Float64}}` with filtered rows and columns before `flagcol`.
"""
function CQD_select_flagged(initial_by_current::OrderedDict{K, Matrix{Float64}},which::Symbol; flagcol::Integer=12)  where {K<:Integer}
    flagset = which === :screen     ? (0,)      :
              which === :crash_SG   ? (1, 2)    :
              which === :crash_tube ? (3,)      :
              which === :crash      ? (1,2,3)   :
              which === :all        ? (0,1,2,3) :
          error("which must be :screen, :crash_SG, :crash_tube, :crash, or :all")

    out = OrderedDict{K, Matrix{Float64}}()
    s = Set(flagset)

    @inbounds for (idx, M) in initial_by_current
        @assert 1 ≤ flagcol ≤ size(M, 2) "flagcol out of bounds (got $flagcol, size=$(size(M)))"
        
        @views col = M[:, flagcol]

        # keep rows where flag ∈ flagset (works for 1, 2, or 3 values)
        keep = findall(in.(col, Ref(s)))

        out[idx] = M[keep, 1:flagcol-1]   # copy rows into a dense Matrix
    end

    return out
end

function CQD_travelling_particles_summary(Ixs, particles, branch::Symbol)
    # Normalize branch label for display
    branch_str = String(Symbol(branch))

    # Small helper: count flags in last column (robust to Int/Float)
    @inline function counts_from_M(M::AbstractMatrix)
        @views col = M[:, end]
        if eltype(col) <: Integer
            pass = count(==(0),  col); top = count(==(1), col)
            bot  = count(==(2),  col); tub = count(==(3), col)
        else
            pass = count(==(0.0), col); top = count(==(1.0), col)
            bot  = count(==(2.0), col); tub = count(==(3.0), col)
        end
        return (pass=pass, top=top, bot=bot, tub=tub)
    end

    data = Matrix{Any}(undef, length(Ixs), 7)

    for i in eachindex(Ixs)
        I0 = Float64(Ixs[i])

        # Accept Matrix or Vector{Matrix}; use the first matrix if a vector is given
        mat = particles[i]
        M   = mat isa AbstractMatrix ? mat : mat[1]

        c   = counts_from_M(M)
        tot = c.pass + c.top + c.bot + c.tub
        passp = tot == 0 ? 0.0 : 100.0 * c.pass / tot
        lossp = tot == 0 ? 0.0 : 100.0 * (c.top + c.bot + c.tub) / tot

        # First column is a string label → use Matrix{Any}
        
        data[i, :] = [c.pass, c.top, c.bot, c.tub, tot, passp, lossp]


    end

        pretty_table(
        data;
        column_labels               = ["Pass","Top","Bottom","Tube","Total","Pass %","Loss %"],
        title                       = "CQD PARTICLE TRAJECTORIES STATISTICS ($(uppercase(string(branch))))",
        formatters                  = [fmt__printf("%d", 1:5), fmt__printf("%5.1f", 6:7)],
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

function CQD_Bn_flag_travelling_particles(Ix, init_particles, kx::Float64, p::AtomParams;
                                                    y_length::Int=1000,
                                                    verbose::Bool=false)
    @info "Evaluating particle trajectories and assigning flags"
    @assert size(init_particles, 2) ≥ 8 "init_particles must have at least 8 columns"
    No        = size(init_particles, 1) # number of particles
    ncurrents = length(Ix)              # number of currents

    # --- Precompute and share the y-grid for the SG span ---
    y_in  = (default_y_FurnaceToSlit + default_y_SlitToSG)::Float64
    y_out = (y_in + default_y_SG)::Float64
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

        # Hoist field/gradient for this current (adjust BvsI/GvsI names to your codebase)
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

            codes[j] = CQD_Bn_cavity_crash(μG,B0,x0,y0,z0,v0x,v0y,v0z,θe,θn,kx,p,ygrid,0.0)
        end

        results[idx] = codes
    end

    return results
end

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
    y_in = (default_y_FurnaceToSlit + default_y_SlitToSG)::Float64
    Lsg  = default_y_SG::Float64
    Ld   = default_y_SGToScreen::Float64
    Ltot = (y_in + Lsg + Ld)::Float64
    ΔL   = (Lsg*Lsg + 2.0*Lsg*Ld)::Float64  # (Lsg+Ld)^2 - Ld^2

    out = OrderedDict{Int8, Matrix{Float64}}()

    @inbounds for idx in eachindex(Ix)
        I0      = Ix[idx]
        flags_i = flagged_trajec[idx]
        @assert length(flags_i) == No "flags[idx] length must equal number of particles"

        # a_z for this current
        μG   = μₑ * GvsI(I0)
        a_z  = μG / p.M
        
        
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

            ωL    = abs(γₑ * (BvsI(I0)+Bn*cos(θn0)))
            kω  = sign(θn0 - θe0) * kx * ωL


            x, z, vz = CQD_Bn_screen_x_z_vz(x0, z0, v0x, v0y, v0z, θe0, a_z, kω, Lsg, Ld, Ltot, ΔL)
            M[j,9]  = x
            M[j,10] = z
            M[j,11] = vz
            M[j,12] = Float64(flags_i[j])   # keep matrix eltype Float64
        end

        out[idx] = M
    end

    return out
end

function CQD_Bn_select_flagged(initial_by_current::OrderedDict{K, Matrix{Float64}},which::Symbol; flagcol::Integer=12)  where {K<:Integer}
    flagset = which === :screen     ? (0,)      :
              which === :crash_SG   ? (1, 2)    :
              which === :crash_tube ? (3,)      :
              which === :crash      ? (1,2,3)   :
              which === :all        ? (0,1,2,3) :
          error("which must be :screen, :crash_SG, :crash_tube, :crash, or :all")

    out = OrderedDict{K, Matrix{Float64}}()
    s = Set(flagset)

    @inbounds for (idx, M) in initial_by_current
        @assert 1 ≤ flagcol ≤ size(M, 2) "flagcol out of bounds (got $flagcol, size=$(size(M)))"
        
        @views col = M[:, flagcol]

        # keep rows where flag ∈ flagset (works for 1, 2, or 3 values)
        keep = findall(in.(col, Ref(s)))

        out[idx] = M[keep, 1:flagcol-1]   # copy rows into a dense Matrix
    end

    return out
end

function CQD_Bn_travelling_particles_summary(Ixs, particles, branch::Symbol)
    # Normalize branch label for display
    branch_str = String(Symbol(branch))

    # Small helper: count flags in last column (robust to Int/Float)
    @inline function counts_from_M(M::AbstractMatrix)
        @views col = M[:, end]
        if eltype(col) <: Integer
            pass = count(==(0),  col); top = count(==(1), col)
            bot  = count(==(2),  col); tub = count(==(3), col)
        else
            pass = count(==(0.0), col); top = count(==(1.0), col)
            bot  = count(==(2.0), col); tub = count(==(3.0), col)
        end
        return (pass=pass, top=top, bot=bot, tub=tub)
    end

    data = Matrix{Any}(undef, length(Ixs), 7)

    for i in eachindex(Ixs)
        I0 = Float64(Ixs[i])

        # Accept Matrix or Vector{Matrix}; use the first matrix if a vector is given
        mat = particles[i]
        M   = mat isa AbstractMatrix ? mat : mat[1]

        c   = counts_from_M(M)
        tot = c.pass + c.top + c.bot + c.tub
        passp = tot == 0 ? 0.0 : 100.0 * c.pass / tot
        lossp = tot == 0 ? 0.0 : 100.0 * (c.top + c.bot + c.tub) / tot

        # First column is a string label → use Matrix{Any}
        
        data[i, :] = [c.pass, c.top, c.bot, c.tub, tot, passp, lossp]

    end

        pretty_table(
        data;
        column_labels               = ["Pass","Top","Bottom","Tube","Total","Pass %","Loss %"],
        title                       = "CQD PARTICLE TRAJECTORIES STATISTICS ($(uppercase(string(branch))))",
        formatters                  = [fmt__printf("%d", 1:5), fmt__printf("%5.1f", 6:7)],
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