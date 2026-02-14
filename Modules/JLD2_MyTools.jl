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




end