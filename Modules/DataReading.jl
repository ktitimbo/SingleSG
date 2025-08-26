module DataReading
    using CSV, DataFrames
    using OrderedCollections
    using Dates

# ─────────────────────────────────────────────────────────────────────────────
# 1) Subfolders starting with a prefix (default "2025")
# ─────────────────────────────────────────────────────────────────────────────
    """
        folder_read(parent; hint="2025") -> Vector{String}

    Return the names of immediate subfolders of `parent` that start with `hint`
    (default `"2025"`).

    Keyword arguments
    - `hint`: string that folder names must start with (default `"2025"`).
    """
    function folder_read(parent::AbstractString; hint::AbstractString="2025")
        flds = filter(f -> isdir(joinpath(parent, f)) && startswith(f, hint), readdir(parent))
        sort!(flds)
        return flds
    end

# ─────────────────────────────────────────────────────────────────────────────
# 2) Parse report.txt → (binning::Union{Int,Missing}, smoothing::Union{Float64,Missing})
# ─────────────────────────────────────────────────────────────────────────────
    """
        extract_info(report_path) -> (binning::Union{Int,Missing}, smoothing::Union{Float64,Missing})

    Read a report file (e.g. `report.txt`) and return:
    - `binning`  : the integer from the **last** line matching `Binning : ...`
                (works for `"4 × 1"`, `"4 x 1"`, or `"1"` by taking the last integer).
    - `smoothing`: the floating value after `Smoothing parameter : ...`
                (supports decimals and scientific notation).

    Returns `missing` for either field if not found or not parseable.
    """
    function extract_info(report_path::AbstractString)::Tuple{Union{Int,Missing},Union{Float64,Missing}}
        txt = read(report_path, String)

        # --- Binning: last "Binning : ..." line → last integer on that line
        binning_val::Union{Int,Missing} = missing
        if !isempty(txt)
            bins = collect(eachmatch(r"Binning\s*:\s*([^\r\n]+)", txt))
            if !isempty(bins)
                bin_str = strip(bins[end].captures[1])           # e.g. "4 × 1" or "1"
                m = match(r"(\d+)\s*$", bin_str)                 # prefer trailing int
                if m === nothing
                    ints = collect(eachmatch(r"\d+", bin_str))   # fallback: any int, take last
                    if !isempty(ints)
                        m = ints[end]
                    end
                end
                if m !== nothing
                    binning_val = parse(Int, m.match)
                end
            end
        end

        # --- Smoothing parameter (allow decimals/scientific)
        smoothing_val::Union{Float64,Missing} = missing
        if !isempty(txt)
            sm = match(r"Smoothing\s*parameter\s*:\s*([0-9.eE+-]+)", txt)
            if sm !== nothing
                parsed = tryparse(Float64, sm.captures[1])
                smoothing_val = parsed === nothing ? missing : parsed
            end
        end

        return (binning_val, smoothing_val)
    end

    # Read `experiment_report.txt` and return the "Data directory" value,
    # with whitespace trimmed and any trailing / or \ removed.
    function extract_data_dir(report_path::AbstractString)::Union{String,Missing}
        txt = read(report_path, String)
        m = match(r"Data\s*directory\s*:\s*([^\r\n]+)", txt)
        if m === nothing
            return missing
        else
            s = strip(m.captures[1])
            return replace(s, r"[\\/]+$" => "")  # drop trailing slash/backslash if present
        end
    end
# ─────────────────────────────────────────────────────────────────────────────
# 3) Find fw_data CSV inside a folder
# ─────────────────────────────────────────────────────────────────────────────
    """
        find_fw_data_csv(folder; filename="fw_data.csv") -> Union{String,Nothing}

    Return the full path to `filename` inside `folder` **iff** it exists.
    No pattern/fallback search is performed.

    - `folder`   : directory to look in
    - `filename` : CSV file name to find (default: `"fw_data.csv"`)

    Returns `nothing` if the file does not exist.
    """
    function find_fw_data_csv(folder::AbstractString;
                            filename::AbstractString="fw_data.csv")::Union{String,Nothing}
        path = joinpath(folder, filename)
        return isfile(path) ? path : nothing
    end

# ─────────────────────────────────────────────────────────────────────────────
# 4) Load CSV with optional column selection/drop
# ─────────────────────────────────────────────────────────────────────────────
    """
        load_fw_data_csv(path; select=nothing, drop=nothing, normalizenames=true) -> DataFrame

    Thin wrapper over `CSV.read` that lets you load *only* certain columns.

    Arguments
    - `select`: keep only these columns (Vector of Symbols/Strings/Ints or a Regex).
    - `drop`  : drop these columns (Vector or Regex).
    - `normalizenames=true`: turn headers like `"time (s)"` into `:time_s`.

    Notes
    - `select` and `drop` follow CSV.jl’s precedence rules.
    """
    function load_fw_data_csv(path::AbstractString;
                            select=nothing, drop=nothing, normalizenames::Bool=true)
        kwargs = (; normalizenames)
        select === nothing || (kwargs = merge(kwargs, (; select=select)))
        drop   === nothing || (kwargs = merge(kwargs, (; drop=drop)))
        return CSV.read(path, DataFrame; kwargs...)
    end

# ─────────────────────────────────────────────────────────────────────────────
# 5) Collect per-folder results as a map: name => (binning, smoothing, df)
# ─────────────────────────────────────────────────────────────────────────────
    """
        collect_fw_map(parent; select=nothing, drop=nothing,
                    filename="fw_data.csv",
                    report_name="experiment_report.txt",
                    skip_missing=true,
                    sort_on=:folder)
    -> OrderedDict{String, NamedTuple{(:binning,:smoothing,:df),
        Tuple{Union{Int,Missing},Union{Float64,Missing},DataFrame}}}

    Scan `parent/<hint>*` (via `folder_read`) and return:
    `map["20250814"] => (binning=1, smoothing=0.03, df=<DataFrame>)`.

    Sorting (always ascending):
    - `sort_on = :binning`   → by (binning, smoothing, folder)
    - `sort_on = :smoothing` → by (smoothing, binning, folder)
    - `sort_on = :folder`    → by (folder, binning, smoothing)
    """
    function collect_fw_map(parent::AbstractString;
                            select=nothing, drop=nothing,
                            filename::AbstractString="fw_data.csv",
                            report_name::AbstractString="experiment_report.txt",
                            skip_missing::Bool=true,
                            sort_on::Symbol=:folder,
                            data_dir_filter::String="20250814")

        out = OrderedDict{String, NamedTuple{(:binning,:smoothing,:df),
            Tuple{Union{Int,Missing},Union{Float64,Missing},DataFrame}}}()

        # normalize the user-provided filter once (allow "20250820" or "20250820/")
        data_dir_filter_norm = data_dir_filter === nothing ? nothing : replace(strip(data_dir_filter), r"[\\/]+$" => "")

        for f in folder_read(parent)
            folder_path = joinpath(parent, f)

            report_path = joinpath(folder_path, report_name)
            !isfile(report_path) && skip_missing && (@warn "Skipping (missing report)" folder=f report=report_name; continue)

            # --------- apply Data directory filter if requested ---------
            if data_dir_filter_norm !== nothing
                rep_dir = extract_data_dir(report_path)  # "20250820" (or missing)
                if rep_dir === missing || rep_dir != data_dir_filter_norm
                    continue  # skip folders whose report doesn't match the requested data dir
                end
            end
            # ------------------------------------------------------------

            fw_path = find_fw_data_csv(folder_path; filename=filename)
            fw_path === nothing && skip_missing && (@warn "Skipping (missing fw_data CSV)" folder=f csv=filename; continue)

            binning, smoothing = isfile(report_path) ? extract_info(report_path) : (missing, missing)
            df = fw_path === nothing ? DataFrame() : load_fw_data_csv(fw_path; select=select, drop=drop)
            out[f] = (binning=binning, smoothing=smoothing, df=df)
        end

        # --- always-ascending sort with missings last ---
        kvs = collect(pairs(out))

        _kbin(b) = b === missing ? (1, 0)    : (0, b)              # (missing-flag, value)
        _ksmo(s) = (s === missing || !isfinite(s)) ? (1, 0.0) : (0, s)

        if sort_on === :binning
            sort!(kvs; by = kv -> (_kbin(kv[2].binning), _ksmo(kv[2].smoothing), kv[1]))
        elseif sort_on === :smoothing
            sort!(kvs; by = kv -> (_ksmo(kv[2].smoothing), _kbin(kv[2].binning), kv[1]))
        elseif sort_on === :folder
            sort!(kvs; by = kv -> (kv[1], _kbin(kv[2].binning), _ksmo(kv[2].smoothing)))
        else
            error("sort_on must be one of :folder, :binning, :smoothing")
        end

        return OrderedCollections.OrderedDict(kvs)
    end




end