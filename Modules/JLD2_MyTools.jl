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


end