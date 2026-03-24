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

Inspect a QM screen–profile JLD2 file and extract the available parameter
combinations stored under the hierarchical structure

    table/nz=.../σw=.../λ0=...

The function traverses the JLD2 file, collects all `(nz, σw, λ0)` tuples
present in the `table` group, and prints the metadata contained in the
`meta/` group.

# Behaviour
- Reads and prints all datasets located in the `meta` group.
- Walks the hierarchy `table → nz → σw → λ0`.
- Parses parameter values directly from group names.
- Returns sorted unique parameter lists together with the full tuple list.

If the file does not contain a `table` group, empty parameter lists are
returned.

# Arguments
- `path::AbstractString`:
  Path to the JLD2 file to inspect.

# Returns
A named tuple containing:

- `keys :: Vector{Tuple{Int,Float64,Float64}}`
    Sorted list of all parameter triples `(nz, σw, λ0)` found.

- `nz :: Vector{Int}`
    Sorted unique values of vertical binning indices.

- `σw :: Vector{Float64}`
    Sorted unique Gaussian widths (mm).

- `λ0 :: Vector{Float64}`
    Sorted unique raw smoothing parameters.

- `meta :: OrderedDict{String,Any}`
    Metadata datasets read from the `meta` group, stored using keys
    `"meta/<name>"`.

# Notes
- Large vectors in metadata are abbreviated when printed.
- Parameter values are inferred from group names, so the function assumes
  the naming convention produced by `make_keypath_qm`.
- The function performs no validation that the parameter grid is complete;
  it only reports entries that physically exist in the file.

# Example
julia
info = list_keys_jld_qm("qm_screen_profiles_f1_table.jld2")

info.nz
info.σw
info.λ0
"""
function list_keys_jld_qm(path::AbstractString)

    keylist  = Tuple{Int,Float64,Float64}[]
    nz_set = Set{Int}()
    σw_set = Set{Float64}()
    λ0_set = Set{Float64}()

    meta = OrderedDict{String,Any}()

    jldopen(path, "r") do file

        # -------------------------
        # Collect meta/* datasets
        # -------------------------
        if haskey(file, "meta")
            meta_group = file["meta"]

            for k in keys(meta_group)
                meta["meta/$k"] = meta_group[k]
            end
        end

        println("\n================ META ================")
        if isempty(meta)
            println("(meta group exists but contains no datasets)")
        else
            for k in collect(keys(meta))
                v = meta[k]
                if v isa AbstractVector && length(v) > 20
                    println(k, " = ", v[1:6], " … ", v[end-5:end],
                            "  (len=", length(v), ")")
                else
                    println(k, " = ", v)
                end
            end
        end
        println("======================================\n")

        # -------------------------
        # Walk the table hierarchy
        # -------------------------
        if !haskey(file, "table")
            return (keys=keylist, nz=Int[], σw=Float64[], λ0=Float64[], meta=meta)
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

                    push!(keylist, (nz, σw, λ0))
                end
            end
        end
    end

    sort!(keylist)
    nz_list     = sort!(collect(nz_set))
    sigmaw_list = sort!(collect(σw_set))
    lambda0_list= sort!(collect(λ0_set))

    return (keys = keylist,
            nz   = nz_list,
            σw   = sigmaw_list,
            λ0   = lambda0_list,
            meta = meta)
end


function make_keypath_cqd(branch::Symbol, ki::Float64, nz::Int, gw::Float64, λ0_raw::Float64)
    fmt(x) = @sprintf("%.12g", x)  # safer than %.6g to reduce collisions
    return "/" * String(branch) *
        "/ki=" * fmt(ki) *"e-6" *
        "/nz=" * string(nz) *
        "/σw=" * fmt(gw) *
        "/λ0=" * fmt(λ0_raw)
end

"""
    list_keys_jld_cqd(path::AbstractString)

Inspect a CQD JLD2 file and extract all parameter combinations stored under
the hierarchical structure defined by `make_keypath_cqd`, namely

    /<branch>/ki=.../nz=.../σw=.../λ0=...

The function traverses the file, parses parameter values from group names,
and collects all existing combinations of `(branch, ki, nz, σw, λ0)`. It also
reads and prints the metadata stored in the `meta` group.

# Behaviour
- Reads all datasets inside the `meta/` group and prints them.
- Walks the hierarchy: `branch → ki → nz → σw → λ0`.
- Parses numerical values directly from group names.
- Collects only parameter combinations that are physically present in the file.
- Returns sorted unique lists of all parameters.

# Arguments
- `path::AbstractString`:
  Path to the JLD2 file to inspect.

# Returns
A named tuple containing:

- `keys :: Vector{Tuple{Symbol,Float64,Int,Float64,Float64}}`
    Sorted list of all parameter tuples `(branch, ki, nz, σw, λ0)` found.

- `branch :: Vector{Symbol}`
    Sorted list of branch identifiers (e.g. `:up`, `:dw`).

- `ki :: Vector{Float64}`
    Sorted list of unique current parameters.

- `nz :: Vector{Int}`
    Sorted list of vertical binning values.

- `σw :: Vector{Float64}`
    Sorted list of Gaussian widths.

- `λ0 :: Vector{Float64}`
    Sorted list of smoothing parameters.

- `meta :: OrderedDict{String,Any}`
    Metadata read from the `meta` group, stored as `"meta/<key>"`.

# Notes
- The function assumes the naming convention produced by
  `make_keypath_cqd(branch, ki, nz, σw, λ0)`.
- The value of `ki` is parsed from strings of the form `"ki=...e-6"`.
  Depending on the implementation, this may represent a scaled quantity
  (e.g. micro-units).
- No assumption is made about completeness of the parameter grid; only
  existing entries are reported.
- Large metadata vectors are truncated when printed for readability.

# Example
julia
info = list_keys_jld_cqd("cqd_profiles.jld2")

info.keys
info.ki
info.nz
info.σw
info.λ0
"""
function list_keys_jld_cqd(path::AbstractString)

    keylist  = Tuple{Symbol,Float64,Int,Float64,Float64}[]
    ki_set = Set{Float64}()
    nz_set = Set{Int}()
    σw_set = Set{Float64}()
    λ0_set = Set{Float64}()
    branch_set = Set{Symbol}()

    meta = OrderedDict{String,Any}()

    jldopen(path, "r") do file

        # -------------------------
        # Collect meta/* datasets
        # -------------------------
        if haskey(file, "meta")
            meta_group = file["meta"]
            for k in keys(meta_group)
                meta["meta/$k"] = meta_group[k]
            end
        end

        println("\n================ META ================")
        if isempty(meta)
            println("(meta group exists but contains no datasets)")
        else
            for k in collect(keys(meta))
                v = meta[k]
                if v isa AbstractVector && length(v) > 20
                    println(k, " = ", v[1:6], " … ", v[end-5:end],
                            "  (len=", length(v), ")")
                else
                    println(k, " = ", v)
                end
            end
        end
        println("======================================\n")

        # -------------------------
        # Walk branch hierarchy
        # -------------------------
        top_groups = [k for k in keys(file) if k != "meta"]

        for branch_grp in top_groups
            branch = Symbol(branch_grp)
            push!(branch_set, branch)

            branch_group = file[branch_grp]

            for ki_grp in keys(branch_group)
                # ki_grp like "ki=2.2e-6"
                ki_str = split(ki_grp, "=", limit=2)[2]

                # remove the trailing "e-6" added by make_keypath_cqd
                ki_base = endswith(ki_str, "e-6") ? ki_str[1:end-3] : ki_str
                ki = parse(Float64, ki_base)

                push!(ki_set, ki)

                ki_group = branch_group[ki_grp]
                for nz_grp in keys(ki_group)
                    # nz_grp like "nz=4"
                    nz = parse(Int, split(nz_grp, "=", limit=2)[2])
                    push!(nz_set, nz)

                    nz_group = ki_group[nz_grp]
                    for σw_grp in keys(nz_group)
                        # σw_grp like "σw=0.2"
                        σw = parse(Float64, split(σw_grp, "=", limit=2)[2])
                        push!(σw_set, σw)

                        σw_group = nz_group[σw_grp]
                        for λ0_grp in keys(σw_group)
                            # λ0_grp like "λ0=0.01"
                            λ0 = parse(Float64, split(λ0_grp, "=", limit=2)[2])
                            push!(λ0_set, λ0)

                            push!(keylist, (branch, ki, nz, σw, λ0))
                        end
                    end
                end
            end
        end
    end

    sort!(keylist)
    branch_list = sort!(collect(branch_set))
    ki_list     = sort!(collect(ki_set))
    nz_list     = sort!(collect(nz_set))
    sigmaw_list = sort!(collect(σw_set))
    lambda0_list= sort!(collect(λ0_set))

    return (keys   = keylist,
            branch = branch_list,
            ki     = ki_list,
            nz     = nz_list,
            σw     = sigmaw_list,
            λ0     = lambda0_list,
            meta   = meta)
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
    merge_qm_two_runs(run1_path::AbstractString,
                      run2_path::AbstractString,
                      out_path::AbstractString;
                      meta_overrides::Dict{String,Any}=Dict())

Merge two QM table JLD2 files (run1 as reference, run2 as donor) into `out_path`.

Behavior (matches your script):
- Reads `meta/nx`, `meta/nz`, `meta/σw`, `meta/λ0` from both files.
- Builds parameter grids as `product(nz, σw, λ0)`.
- If `meta/nx` differs, aborts (warns and returns `nothing`).
- Computes `comb_add = setdiff(param_grid_2, param_grid_1)`.
- Builds `nz_add`, `σw_add`, `λ0_add`, then `param_grid_add = product(nz_add, σw_add, λ0_add)`.
- Prints diagnostics:
    - whether `Set(param_grid_add) == Set(comb_add)`
    - how many tuples were "invented" by the marginal-product reconstruction.
- Writes output meta and copies datasets:
    - all `param_grid_1` entries from run1
    - all `param_grid_add` entries from run2
"""
function merge_qm_two_runs(run1_path::AbstractString,
                           run2_path::AbstractString,
                           out_path::AbstractString)

    jldopen(run1_path, "r") do file1
        N1   = Int(file1["meta/N"])
        T1   = Float64(file1["meta/T"])
        s1   = Float64(file1["meta/s_spline"])
        nx1  = Int(file1["meta/nx"])
        nz1  = Vector{Int}(file1["meta/nz"])
        σw1  = Vector{Float64}(file1["meta/σw"])
        λ01  = Vector{Float64}(file1["meta/λ0"])
        param_grid_1 = Tuple{Int,Float64,Float64}.(collect(Iterators.product(nz1, σw1, λ01)))

        jldopen(run2_path, "r") do file2
            N2   = Int(file2["meta/N"])
            T2   = Float64(file2["meta/T"])
            s2   = Float64(file2["meta/s_spline"])
            nx2  = Int(file2["meta/nx"])
            nz2  = Vector{Int}(file2["meta/nz"])
            σw2  = Vector{Float64}(file2["meta/σw"])
            λ02  = Vector{Float64}(file2["meta/λ0"])
            param_grid_2 = Tuple{Int,Float64,Float64}.(collect(Iterators.product(nz2, σw2, λ02)))

            if nx1 != nx2
                @warn "Not recommended to merge: nₓ do not match $(nx1) ≠ $(nx2)"
                return nothing
            end
            if N1 != N2
                @warn "Not recommended to merge: N do not match $(N1) ≠ $(N2)"
                return nothing
            end
            if T1 != T2
                @warn "Not recommended to merge: T do not match $(T1) ≠ $(T2)"
                return nothing
            end
            if s1 != s2
                @warn "Not recommended to merge: spline do not match $(s1) ≠ $(s2)"
                return nothing
            end

            comb_add = setdiff(param_grid_2, param_grid_1)
            nz_add   = unique(first.(comb_add))
            σw_add   = unique(getindex.(comb_add, 2))
            λ0_add   = unique(last.(comb_add))
            param_grid_add = Tuple{Int,Float64,Float64}.(collect(Iterators.product(nz_add, σw_add, λ0_add)))

            println("same as sets? ", Set(param_grid_add) == Set(comb_add))
            println("invented? ", length(setdiff(param_grid_add, comb_add)))

            nz_list = sort(union(nz1, nz_add))
            σw_list = sort(union(σw1, σw_add))
            λ0_list = sort(union(λ01, λ0_add))

            jldopen(out_path, "w") do outfile
                # ---- default meta (your original) ----
                outfile["meta/N"]        = N1
                outfile["meta/T"]        = T1
                outfile["meta/s_spline"] = s1
                outfile["meta/nx"]       = nx1
                outfile["meta/nz"]       = nz_list
                outfile["meta/σw"]       = σw_list
                outfile["meta/λ0"]       = λ0_list

                # copy run1 entries
                for (nz, gw, λ0) in param_grid_1
                    println("nz=$nz, σ=$(Int(round(1e3*gw)))μm, λ₀=$λ0")
                    keypath = make_keypath_qm(nz, gw, λ0)
                    outfile[keypath] = file1[keypath]
                end

                # copy run2 additions
                for (nz, gw, λ0) in param_grid_add
                    println("nz=$nz, σ=$(Int(round(1e3*gw)))μm, λ₀=$λ0")
                    keypath = make_keypath_qm(nz, gw, λ0)
                    outfile[keypath] = file2[keypath]
                end
            end

            @info "Reference merge finished"
            return out_path
        end
    end
end


"""
    tree_jld(path; maxdepth=typemax(Int))

Print the hierarchical structure of a JLD2 file.

Example output:

meta/
  nx
  nz
table/
  nz=1/
    σw=0.100/
      λ0=0.001
"""
function tree_jld(path::AbstractString; maxdepth::Int=typemax(Int))

    function _walk(node, prefix::String, depth::Int)
        depth > maxdepth && return

        for name in sort(collect(keys(node)))
            obj = node[name]

            if obj isa JLD2.Group
                println(prefix, name, "/")
                _walk(obj, prefix * "  ", depth + 1)
            else
                # dataset
                T = typeof(obj)
                if obj isa AbstractArray
                    println(prefix, name, " :: ", eltype(obj),
                            "  size=", size(obj))
                else
                    println(prefix, name, " :: ", T)
                end
            end
        end
    end

    jldopen(path, "r") do file
        println("\nJLD2 tree: ", path)
        println("--------------------------------")
        _walk(file, "", 0)
        println("--------------------------------\n")
    end

    return nothing
end

end