module JLD2_MyTools
    using Printf
    using JLD2
    using DataStructures: OrderedDict

    function make_keypath_qm(nz::Int, σw::Float64, λ0::Float64)
        # This generates the keys for files as qm_screen_profiles_f1_table

        fmt = Printf.Format("%.3f")

        σw_str = Printf.format(fmt, σw)
        λ0_str = Printf.format(fmt, λ0)

        return "table/nz=$(nz)/σw=$(σw_str)/λ0=$(λ0_str)"
    end

    """
        list_keys_jld_qm(path::AbstractString)

    Scan a JLD2 file organized as

        table/nz=.../σw=.../λ0=...

    and reconstruct the parameter keys stored in the hierarchy.

    The function walks the `table` group and extracts `(nz, σw, λ0)` values
    from dataset paths of the form:

        nz=<Int>/σw=<Float64>/λ0=<Float64>

    It returns both the full list of tuple keys and the unique values
    for each parameter.

    # Arguments
    - `path::AbstractString`:
    Path to the JLD2 file containing the `table` hierarchy.

    # Returns
    A `NamedTuple` with fields:

    - `keys :: Vector{Tuple{Int,Float64,Float64}}`  
    Sorted list of all `(nz, σw, λ0)` parameter combinations found.

    - `nz :: Vector{Int}`  
    Sorted list of unique `nz` values.

    - `σw :: Vector{Float64}`  
    Sorted list of unique `σw` values.

    - `λ0 :: Vector{Float64}`  
    Sorted list of unique `λ0` values.

    If the file does not contain a `"table"` group, all returned lists
    are empty.

    # Notes
    - This function parses parameter values from group names, so it assumes
    they were written using a consistent formatting scheme
    (e.g. `@sprintf("%.3f", x)`).
    - Floating-point values are reconstructed from strings and may differ
    slightly from their original in-memory representations.
    - The result is sorted lexicographically by `(nz, σw, λ0)`.

    # Example
    julia
    idx = list_keys_jld_qm("qm_screen_data_vvv.jld2")

    idx.keys[1:5]        # first few keys
    idx.nz          # available nz values
    idx.σw          # available σw values
    idx.λ0          # available λ0 values
    """
    function list_keys_jld_qm(path::AbstractString) 
        klist = Tuple{Int,Float64,Float64}[] 
        nz_set = Set{Int}() 
        σw_set = Set{Float64}() 
        λ0_set = Set{Float64}() 
        jldopen(path, "r") do file 
            if !haskey(file, "table") 
                return (keys = klist, nz_list = Int[], sigmaw_list = Float64[], lambda0_list = Float64[]) 
            end 
        table = file["table"] 
        for nz_grp in Base.keys(table) 
            # nz_grp like "nz=1" 
            nz = parse(Int, split(nz_grp, "=", limit=2)[2]) 
            push!(nz_set, nz) 
            
            nz_group = table[nz_grp] 
            for σw_grp in Base.keys(nz_group) 
                # σw_grp like "σw=0.250" 
                σw = parse(Float64, split(σw_grp, "=", limit=2)[2]) 
                push!(σw_set, σw) 
                
                σw_group = nz_group[σw_grp] 
                for λ0_grp in Base.keys(σw_group) 
                    # λ0_grp like "λ0=0.001" 
                    λ0 = parse(Float64, split(λ0_grp, "=", limit=2)[2]) 
                    push!(λ0_set, λ0) 
                    
                    push!(klist, (nz, σw, λ0)) 
                end 
            end 
        end 
    end 

    sort!(klist) 
    nz_list = sort!(collect(nz_set)) 
    sigmaw_list = sort!(collect(σw_set)) 
    lambda0_list = sort!(collect(λ0_set)) 

    return (keys = klist, nz = nz_list, σw = sigmaw_list, λ0 = lambda0_list) 
    end


    function make_keypath_cqd(branch::Symbol, ki::Float64, nz::Int, gw::Float64, λ0_raw::Float64)
        fmt(x) = @sprintf("%.12g", x)  # safer than %.6g to reduce collisions
        return "/" * String(branch) *
            "/ki=" * fmt(ki) *"e-6" *
            "/nz=" * string(nz) *
            "/σw=" * fmt(gw) *
            "/λ0=" * fmt(λ0_raw)
    end

    # JLD2 doesn't overwrite datasets by default; delete first.
    function jld_overwrite!(f, path::AbstractString, value)
        if haskey(f, path)
            delete!(f, path)
        end
        f[path] = value
        return nothing
    end

    function make_keypath_exp(data::AbstractString, nz::Int, λ0::Float64)
        # This generates the keys for files as qm_screen_profiles_f1_table
        fmt = Printf.Format("%.3f")
        λ0_str = Printf.format(fmt, λ0)

        return "$(data)/nz=$(nz)/λ0=$(λ0_str)"
    end


    parse_nz(nz_key::AbstractString) = parse(Int, split(nz_key, "=", limit=2)[2])
    parse_λ0(λ0_key::AbstractString) = parse(Float64, split(λ0_key, "=", limit=2)[2])
    function show_exp_summary(path::AbstractString, data::AbstractString; full_keys = Set(["BzTesla","Currents","ErrorCurrents"]))
        fmt6(x::Real) = @sprintf("%.6f", x)
        function format_value(v)
            if v isa AbstractVector{<:Real}
                return "[" * join(fmt6.(v), ", ") * "]"
            elseif v isa Real
                return fmt6(v)
            else
                return repr(v)
            end
        end
        jldopen(path, "r") do f
            println("====================================")
            println("FILE:  ", path)
            println("DATA:  ", data)
            println("====================================")

            println("\nMETA:")
            if haskey(f, "meta")
                for k in sort(String.(collect(keys(f["meta"]))))
                    v = f["meta/$k"]

                    if v isa AbstractVector
                        if k in full_keys
                            # show the full vector
                            # println(@sprintf("  %-18s : %s", k, repr(v)))
                            println(@sprintf("  %-18s : %s", k, format_value(v)))
                        else
                            # compact summary
                            n = length(v)
                            if n == 0
                                println(@sprintf("  %-18s : Vector(len=0)", k))
                            else
                                println(@sprintf("  %-18s : Vector(len=%d), first=%s, last=%s",
                                                k, n, repr(first(v)), repr(last(v))))
                            end
                        end
                    else
                        println(@sprintf("  %-18s : %s", k, repr(v)))
                    end
                end
            else
                println("  (no meta)")
            end

            # ---- RUNS ----
            println("\nSTORED PARAMETERS:")
            if !haskey(f, data)
                println("  (no group '$data')")
                println("  top-level keys: ", sort(String.(collect(keys(f)))))
                return
            end

            nz_vals = Int[]
            λ0_vals = Float64[]

            for nz_key in String.(collect(keys(f[data])))
                startswith(nz_key, "nz=") || continue
                nz = parse_nz(nz_key)

                for λ0_key in String.(collect(keys(f["$data/$nz_key"])))
                    startswith(λ0_key, "λ0=") || continue
                    λ0 = parse_λ0(λ0_key)

                    push!(nz_vals, nz)
                    push!(λ0_vals, λ0)
                end
            end

            nz_list = sort(unique(nz_vals))
            λ0_list = sort(unique(λ0_vals))

            println("  nz   : ", nz_list)
            println("  λ0   : ", [@sprintf("%.3f", x) for x in λ0_list])
            println("  runs : ", length(nz_vals))

            println("\nRUNS (nz, λ0):")
            combos = sort(unique(zip(nz_vals, λ0_vals)), by = x -> (x[1], x[2]))
            for (nz, λ0) in combos
                println(@sprintf("  nz=%-3d  λ0=%.3f", nz, λ0))
            end
        end
    end

"""
    parse_qm_key(key::String) -> (nz, σw, λ0)

Extract numeric parameters from a QM table key.
Expected format:
    table/nz=.../σw=.../λ0=...
"""
function parse_qm_key(key::AbstractString)
    parts = split(key, '/')
    @assert length(parts) ≥ 4 "Invalid QM key: $key"

    nz  = parse(Int,      split(parts[2], '=')[2])
    σw  = parse(Float64,  split(parts[3], '=')[2])
    λ0  = parse(Float64,  split(parts[4], '=')[2])

    return nz, σw, λ0
end

"""
    merge_qm_tables_reference(refpath, inpaths, outpath)

Merge multiple QM screen–profile JLD2 tables using a *reference file*
as the authoritative dataset.

The function copies all entries from `refpath` into a new output file
and then scans additional input files, adding only parameter combinations
that are **missing** from the reference. Parameter keys are reconstructed
using `make_keypath_qm(nz, σw, λ0)` to guarantee consistent formatting
across runs.

This procedure is designed for combining independent parameter sweeps
(e.g. different Gaussian widths or smoothing parameters) while ensuring
that incompatible datasets are not merged.

# Behaviour
- The reference file defines the required spatial binning (`meta/nx`).
- Any input file whose `meta/nx` differs from the reference is skipped.
- For each dataset stored under

      table/nz=.../σw=.../λ0=...

  the tuple `(nz, σw, λ0)` is extracted numerically.
- Entries already present in the reference are not overwritten.
- Missing parameter combinations are appended to the output.
- Metadata (`meta/nz`, `meta/σw`, `meta/λ0`) is rebuilt from the
  parameter values actually written to the merged file.

# Arguments
- `refpath::String`:
  Path to the reference JLD2 table. All its data are copied first
  and define the merge compatibility conditions.

- `inpaths::Vector{String}`:
  Additional JLD2 files to scan for missing parameter combinations.

- `outpath::String`:
  Destination path of the merged JLD2 file.

# Output
Creates a new JLD2 file at `outpath` containing:
- all reference datasets,
- newly discovered `(nz, σw, λ0)` realizations,
- rebuilt metadata describing the full parameter grid.

# Notes
- Keys are regenerated using `make_keypath_qm`, preventing duplicate
  entries caused by floating-point formatting differences
  (e.g. `"0.1"` vs `"0.100"`).
- The reference file is treated as authoritative: existing datasets
  are never replaced.
- The function assumes the internal hierarchy used by
  `qm_screen_profiles_f*_table.jld2`.

# Physics context
Each table entry corresponds to a reconstructed screen probability
distribution obtained from the quantum Stern–Gerlach propagation
pipeline for a specific parameter triple

    (n_z, σ_w, λ₀),

representing vertical binning, Gaussian smoothing width, and raw
regularization strength, respectively. The merge therefore enlarges
the explored parameter space without altering previously validated
results.

# Example
julia
merge_qm_tables_reference(
    "runA/qm_screen_profiles_f1_table.jld2",
    ["runB/qm_screen_profiles_f1_table.jld2",
     "runC/qm_screen_profiles_f1_table.jld2"],
    "qm_screen_profiles_f1_table_merged.jld2"
)
"""
function merge_qm_tables_reference(
        refpath::String,
        inpaths::Vector{String},
        outpath::String)

    # -----------------------------
    # Load reference nx
    # -----------------------------
    ref_nx = jldopen(refpath,"r") do f
        Int(f["meta/nx"])
    end

    @info "Reference nx = $ref_nx"

    written = Set{Tuple{Int,Float64,Float64}}()

    nz_set  = Set{Int}()
    gw_set  = Set{Float64}()
    λ0_set  = Set{Float64}()

    # =============================
    # Create output and copy reference
    # =============================
    jldopen(outpath,"w") do fout

        jldopen(refpath,"r") do fref
            for k in keys(fref)

                if startswith(k,"table/")
                    nz,gw,λ0 = parse_qm_key(k)

                    canon = make_keypath_qm(nz,gw,λ0)
                    fout[canon] = fref[k]

                    push!(written,(nz,gw,λ0))
                    push!(nz_set,nz)
                    push!(gw_set,gw)
                    push!(λ0_set,λ0)
                else
                    # copy meta and everything else
                    fout[k] = fref[k]
                end
            end
        end

        # =============================
        # Scan additional files
        # =============================
        for ip in inpaths
            ip == refpath && continue

            jldopen(ip,"r") do fin

                # ---- nx consistency check ----
                nx_in = Int(fin["meta/nx"])
                if nx_in != ref_nx
                    @warn "Skipping (nx mismatch)" file=ip nx_in ref_nx
                    return
                end

                added = 0

                for k in keys(fin)
                    startswith(k,"table/") || continue

                    nz,gw,λ0 = parse_qm_key(k)

                    tup = (nz,gw,λ0)

                    # add only missing combinations
                    if !(tup in written)

                        canon = make_keypath_qm(nz,gw,λ0)

                        fout[canon] = fin[k]

                        push!(written,tup)
                        push!(nz_set,nz)
                        push!(gw_set,gw)
                        push!(λ0_set,λ0)

                        added += 1
                    end
                end

                @info "Merged new entries" file=ip added
            end
        end

        # =============================
        # rebuild META (authoritative)
        # =============================
        fout["meta/nx"] = ref_nx
        fout["meta/nz"] = sort(collect(nz_set))
        fout["meta/σw"] = sort(collect(gw_set))
        fout["meta/λ0"] = sort(collect(λ0_set))
    end

    @info "Reference merge finished" outpath
end

end