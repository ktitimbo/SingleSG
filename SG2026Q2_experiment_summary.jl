# Kelvin Titimbo
# California Institute of Technology
# February 2026

############## EXPERIMENT ANALYSIS PREAMBLE ##############
# Headless/Windows-safe GR: set before using Plots
# if !haskey(ENV, "GKSwstype")
#     ENV["GKSwstype"] = "100"  # offscreen; avoids popup windows/crashes
# end

# Plotting backend and general appearance settings
using Plots; gr()
# Set default plot aesthetics
const IN_NOTEBOOK = isdefined(Main, :IJulia);
Plots.default(
    show=IN_NOTEBOOK, dpi=800, fontfamily="Computer Modern", 
    grid=true, minorgrid=true, framestyle=:box, widen=true,
)
using Plots.PlotMeasures
# Data I/O and numerical tools
using LinearAlgebra, Random
using Statistics, StatsBase, OrderedCollections, Interpolations
using Dierckx
# Aesthetics and output formatting
using Colors, ColorSchemes
using Printf, LaTeXStrings, PrettyTables
using CSV, DataFrames, DelimitedFiles, JLD2
# Time-stamping/logging
using Dates
using Alert
const T_START = Dates.now()
# Custom modules
include("./Modules/TheoreticalSimulation.jl");
include("./Modules/MyExperimentalAnalysis.jl");
using .MyExperimentalAnalysis;
include("./Modules/DataReading.jl");
include("./Modules/JLD2_MyTools.jl");
# Set the working directory to the current location
cd(@__DIR__) 
const BASE_PATH = raw"F:\SternGerlachExperiments"
const RUN_STAMP = Dates.format(T_START, "yyyymmddTHHMMSS");
const OUTDIR    = joinpath(@__DIR__, "EXPDATA_ANALYSIS", "smoothing_binning")
isdir(OUTDIR) || mkpath(OUTDIR);
@info "Created output directory" OUTDIR
# General setup
hostname = gethostname();
@info "Running on host" hostname=hostname
# For Plots
FIG_EXT = "png"   # could be "pdf", "svg", etc.
SAVE_FIG = false
MyExperimentalAnalysis.SAVE_FIG = SAVE_FIG;
MyExperimentalAnalysis.FIG_EXT  = FIG_EXT;
MyExperimentalAnalysis.OUTDIR   = OUTDIR;

# Previous experiment data for comparison
data_JSF = OrderedDict(
    :exp => hcat(
    [0.0200, 0.0300, 0.0500, 0.1500, 0.2000, 0.2500, 0.3500, 0.5000, 0.7500], #mA
    [0.0229, 0.0610, 0.1107, 0.3901, 0.5122, 0.6315, 0.8139, 1.1201, 1.5738]),
    :model => hcat(
    [0.0150, 0.0200, 0.0250, 0.0300, 0.0400, 0.0500, 0.0700, 0.1000, 0.1500, 0.2000, 0.2500, 0.3500, 0.5000, 0.7500], #mA
    [0.0409, 0.0566, 0.0830, 0.1015, 0.1478, 0.1758, 0.2409, 0.3203, 0.4388, 0.5433, 0.6423, 0.8394, 1.1267, 1.5288], #CQD
    [0.0179, 0.0233, 0.0409, 0.0536, 0.0883, 0.1095, 0.1713, 0.2487, 0.3697, 0.4765, 0.5786, 0.7757, 1.0655, 1.4630]) #QM
);


const DIR_LIST = [
    # "20260819",
    "20260821",
    "20260826",
    "20260827",
    "20260831",
    "20260902",
    # "20260903"
]
n_runs = length(DIR_LIST)
cols = palette(:darkrainbow, n_runs);

# --------------------------------------------------------------- configuration

"""
    SETUP

Everything file- and axis-specific, in one place.

`axes.match` are the axes shared with the convolution target — these, and only
these, are compared. `axes.exact` are hard constraints: `nz` is a bin count, so
an off-by-one is a different analysis rather than a small error, and it must
agree exactly instead of being traded off against λ0 and σw. `axes.qm` and
`axes.cqd` are the axes each model stores as `meta/<name>`; CQD's extra `ki` has
no convolution counterpart, so it is reported as a free axis, never matched.

`paths` point at the convolution study and the two profile tables.
"""
const SETUP = let matched = (:λ0, :nz, :σw)
    (axes = (match = matched,
             exact = (:nz,),
             qm    = matched,
             cqd   = (matched..., :ki)),
     paths = (conv   = joinpath(@__DIR__, "data_studies",
                                "CONV2026_20260916T121229019", "blur_conv_3.jld2"),
              qm     = joinpath(BASE_PATH, "SIMULATIONS", "2026Q2_SETUP",
                                "QM_T205_8M", "qm_screen_profiles_f1_table.jld2"),
              cqd_up = joinpath(BASE_PATH, "SIMULATIONS", "2026Q2_SETUP",
                                "CQD_T205_8M_v2",
                                "cqd_8000000_up_profiles_bykey.jld2")))
end


# =====================================================================
#  Convolution → simulation parameter matching
#
#  The convolution study fixes three parameters: smoothing λ0, z-binning nz and
#  Gaussian blur width σw. Each model (QM, CQD) was run on a discrete grid over
#  the same three, CQD additionally over ki. This module picks, for each model,
#  the grid point representing the convolution target — the exact one if it was
#  simulated, otherwise the nearest — then reduces the models to the single
#  (λ0, nz, σw) triple used by every downstream comparison.
#
#  Layout:  configuration → result type → readers → matcher → reduction → run
#  Assumes JLD2, Statistics and Printf are already loaded by the enclosing script.
# =====================================================================

# --------------------------------------------------------------- result type

"""
    ParamMatch{F}

Outcome of [`match_params`](@ref) for one model. `F` is the tuple of matched axis
names, shared by `target`, `matched`, `residual` and `relative`.

| field      | meaning                                                            |
|:-----------|:-------------------------------------------------------------------|
| `exact`    | the grid holds the target parameters, within `rtol`/`atol`         |
| `target`   | values that were searched for (the convolution study)              |
| `matched`  | grid point selected — this is what goes downstream                 |
| `residual` | `matched - target`, signed, in each axis' own units                |
| `relative` | `residual / target`, signed and dimensionless                      |
| `cost`     | weighted sum of `abs.(relative)`, the quantity minimised           |
| `free`     | per unmatched grid axis (e.g. CQD's `ki`), the values available at  |
|            | the matched point — all equally valid, none preferred              |

See also [`resolve_params`](@ref) to reduce several `ParamMatch` to one triple.
"""
struct ParamMatch{F}
    exact::Bool
    target::NamedTuple{F}
    matched::NamedTuple{F}
    residual::NamedTuple{F}
    relative::NamedTuple{F}
    cost::Float64
    free::NamedTuple
end

# Compact form, used when a ParamMatch is interpolated into a string or nested
# inside another container.
Base.show(io::IO, m::ParamMatch) =
    print(io, "ParamMatch(", m.exact ? "exact" : @sprintf("cost %.3g", m.cost),
          ", ", m.matched, ")")

# Full form, used when a ParamMatch is the value displayed at the REPL.
function Base.show(io::IO, ::MIME"text/plain", m::ParamMatch)
    println(io, m.exact ? "ParamMatch: exact grid point" :
                @sprintf("ParamMatch: nearest grid point (cost %.4g)", m.cost))
    for k in keys(m.matched)
        @printf(io, "  %-3s target = %-12.6g  matched = %-12.6g  Δ = %+-11.4g  (%+.3f %%)\n",
                k, float(m.target[k]), float(m.matched[k]),
                m.residual[k], 100 * m.relative[k])
    end
    for (k, vals) in pairs(m.free)
        @printf(io, "  %-3s free axis, %d value(s) here: %s\n", k, length(vals), vals)
    end
end

# --------------------------------------------------------------- readers

"""
    read_conv_target(path, dirs = DIR_LIST) -> NamedTuple

Target parameters from a convolution study at `path`: smoothing `λ0`, z-binning
`nz`, and the mean blur width `σw` **in mm** over the experiment dates `dirs`,
kept at full precision (rounding before matching can select the wrong neighbour).

Also returns `σw_std` and the per-date widths `σw_by_dir`. [`match_params`](@ref)
ignores both, but `σw_std` is the natural yardstick for judging whether a
non-exact match matters: a grid offset well inside the day-to-day scatter is
irrelevant, one far outside it is not.

Errors if any date in `dirs` is missing from the study, rather than silently
averaging over fewer runs than intended.
"""
function read_conv_target(path::AbstractString,
                          dirs::AbstractVector{<:AbstractString} = DIR_LIST)
    jldopen(path, "r") do file
        conv = file["convolution"]
        absent = filter(d -> !haskey(conv, d), dirs)
        isempty(absent) || error("No convolution entry for: $(join(absent, ", "))")

        # `blurrGwidth_um` is stored in µm; every simulation grid `meta/σw` is in mm.
        widths = [Float64(conv[d].blurrGwidth_um) * 1.0e-3 for d in dirs]
        (λ0        = file["meta/smoothing"],
         nz        = file["meta/zbinning"],
         σw        = mean(widths),
         σw_std    = std(widths),
         σw_by_dir = Dict(dirs .=> widths))
    end
end

"""
    read_grid_axes(path, names = SETUP.axes.match) -> NamedTuple of Vectors

Read a model's parameter axes from `meta/<name>` for each name in `names`.
Generic over the axis list, so a new axis needs only a new entry in
[`SETUP`](@ref)`.axes`.
"""
read_grid_axes(path::AbstractString, names::Tuple = SETUP.axes.match) =
    jldopen(path, "r") do file
        NamedTuple{names}(map(n -> collect(file["meta/$n"]), names))
    end

"""
    grid_points(axes) -> Vector{<:NamedTuple}

Expand parameter axes into the catalogue of points [`match_params`](@ref) searches.

!!! warning "Cartesian-grid assumption"
    Assumes the axes span a **full** Cartesian product. That is safe for the QM
    table but questionable once `ki` is involved, since a `ki` sweep is usually
    run at one or two (λ0, nz, σw) settings rather than at all of them. If the
    profile file is keyed by parameter tuple, build this vector from those keys
    instead — `match_params` takes any `Vector` of NamedTuples, and will then
    report honestly when a combination was never simulated.
"""
grid_points(axes::NamedTuple) =
    vec([NamedTuple{keys(axes)}(p) for p in Iterators.product(values(axes)...)])

# --------------------------------------------------------------- matcher

"""
    match_params(grid, target; fields, exact_fields, weights, rtol, atol) -> ParamMatch

Select the point of `grid` best representing `target`, comparing only `fields`.

Distance is **relative** to the target on each axis, so parameters in unrelated
units (mm, bin counts, smoothing lengths) contribute comparably; `weights`
rescales individual axes, e.g. `weights = (σw = 3.0,)` to prioritise blur width.

`exact_fields` are hard constraints — candidates disagreeing there are discarded
rather than traded off (default `SETUP.axes.exact`; pass `()` to let every axis
trade off). Errors if the constraints eliminate the whole grid.

Grid axes outside `fields` are unconstrained by the target, so every point
sharing the winner's matched coordinates is an equally good answer; those values
are returned in `free` instead of one being chosen arbitrarily.
"""
function match_params(grid::AbstractVector{<:NamedTuple}, target::NamedTuple;
                      fields::Tuple = SETUP.axes.match,
                      exact_fields::Tuple = SETUP.axes.exact,
                      weights::NamedTuple = NamedTuple(),
                      rtol::Real = 1.0e-8, atol::Real = 0)

    isclose(a, b) = isapprox(float(a), float(b); rtol = rtol, atol = atol)
    reldiff(k, p) = (t = float(target[k]); (float(p[k]) - t) / max(abs(t), eps()))
    cost(p)       = sum(k -> get(weights, k, 1.0) * abs(reldiff(k, p)), fields)

    # Hard constraints first. `all` over an empty tuple is true, so an empty
    # `exact_fields` leaves the grid untouched.
    pool = filter(p -> all(k -> isclose(p[k], target[k]), exact_fields), grid)
    isempty(pool) && error("No grid point satisfies the hard constraints " *
                           "$(NamedTuple{exact_fields}(target)) — relax `exact_fields`.")

    # An exact point has cost ≈ 0 and therefore wins the minimisation on its own;
    # no separate exact-match search is needed.
    best = argmin(cost, pool)

    free_axes = Tuple(setdiff(keys(best), fields))
    siblings  = filter(p -> all(k -> isclose(p[k], best[k]), fields), pool)

    ParamMatch(all(k -> isclose(best[k], target[k]), fields),
               NamedTuple{fields}(target),
               NamedTuple{fields}(best),
               NamedTuple{fields}(map(k -> float(best[k]) - float(target[k]), fields)),
               NamedTuple{fields}(map(k -> reldiff(k, best), fields)),
               cost(best),
               NamedTuple{free_axes}(map(k -> sort(unique(p[k] for p in siblings)),
                                         free_axes)))
end

"""
    match_model(path, target, axes; kwargs...) -> ParamMatch

Read a model's grid from `path` (axes `axes`) and match it against `target` on
`SETUP.axes.match`. Keyword arguments are forwarded to [`match_params`](@ref).
"""
match_model(path::AbstractString, target::NamedTuple, axes::Tuple; kwargs...) =
    match_params(grid_points(read_grid_axes(path, axes)), target;
                 fields = SETUP.axes.match, kwargs...)

# --------------------------------------------------------------- reduction

"""
    print_match_table([io], matches; fields = SETUP.axes.match)

Tabulate the convolution target beside each model's matched value and relative
offset, one row per axis, followed by any free axes. `matches` is a NamedTuple of
[`ParamMatch`](@ref), e.g. `(QM = ..., CQD = ...)`.
"""
function print_match_table(io::IO, matches::NamedTuple;
                           fields::Tuple = SETUP.axes.match)
    print(io, rpad("axis", 5), rpad("conv target", 15))
    for name in keys(matches)
        print(io, rpad(string(name), 15), rpad("Δ %", 10))
    end
    println(io)

    target = first(values(matches)).target
    for k in fields
        print(io, rpad(string(k), 5), rpad(@sprintf("%.6g", float(target[k])), 15))
        for m in values(matches)
            print(io, rpad(@sprintf("%.6g", float(m.matched[k])), 15),
                      rpad(@sprintf("%+.3f", 100 * m.relative[k]), 10))
        end
        println(io)
    end

    for (name, m) in pairs(matches), (k, vals) in pairs(m.free)
        println(io, "$name free axis $k: ", vals)
    end
    return nothing
end

print_match_table(matches::NamedTuple; kwargs...) =
    print_match_table(stdout, matches; kwargs...)

"""
    resolve_params(matches; fields, rtol, strict, verbose) -> NamedTuple

Reduce several models' matches to the **single** parameter triple used downstream.

The models must have landed on the same grid point: if QM sits at one σw and CQD
at another, any difference between their screen profiles is partly grid mismatch
rather than physics, so the comparison is not like-for-like. Disagreement
therefore throws (`strict = true`, the default) or warns and returns the first
model's values (`strict = false`).

A non-exact but unanimous match is fine — it only means the convolution target
falls between grid points — and is reported as a warning, not an error. Judge it
against `conv_target.σw_std` before accepting it.

Returns `NamedTuple{fields}` of the agreed values, and prints the comparison
table unless `verbose = false`.
"""
function resolve_params(matches::NamedTuple; fields::Tuple = SETUP.axes.match,
                        rtol::Real = 1.0e-8, strict::Bool = true,
                        verbose::Bool = true)
    isempty(matches) && error("resolve_params needs at least one ParamMatch")
    verbose && print_match_table(matches; fields = fields)

    reference = first(values(matches)).matched
    disagreeing = [k for k in fields
                   if !all(m -> isapprox(float(m.matched[k]), float(reference[k]);
                                         rtol = rtol), values(matches))]

    if !isempty(disagreeing)
        detail = join(("$k: " * join(("$name=$(m.matched[k])"
                                      for (name, m) in pairs(matches)), ", ")
                       for k in disagreeing), "; ")
        msg = "Models matched to different grid points ($detail) — profiles are " *
              "not directly comparable."
        strict ? error(msg) : @warn msg
    end

    for (name, m) in pairs(matches)
        m.exact || @warn "$name has no exact grid point for the convolution target" m.residual
    end

    return NamedTuple{fields}(map(k -> reference[k], fields))
end

# --------------------------------------------------------------- run

conv_target = read_conv_target(SETUP.paths.conv)
sim_matches = (QM  = match_model(SETUP.paths.qm,     conv_target, SETUP.axes.qm),
               CQD = match_model(SETUP.paths.cqd_up, conv_target, SETUP.axes.cqd))

"""
    SIM_PARAMS

The one parameter set every downstream QM/CQD comparison uses: `λ0`, `nz`, `σw`
agreed by both models (`resolve_params` throws if they disagree), plus `ki`, the
values CQD simulated at that grid point.
"""
const SIM_PARAMS = let p = resolve_params(sim_matches)
    (λ0 = Float64(p.λ0),
     nz = Int(p.nz),
     σw = Float64(p.σw),
     ki = sim_matches.CQD.free.ki)
end




# =====================================================================
#  SG field calibration: B0(I), B1(I) from several experiment runs
#
#  Each date in DIR_LIST sweeps SG1 current and records two independent field
#  measurements. The runs do NOT share a current grid — SG1currentInA is a
#  readback, not a commanded value — so they are combined in the function
#  domain: pool every point, average the replicates at each setpoint, fit one
#  smoothing spline per channel, and evaluate that on whatever grid is needed.
#
#  Layout: configuration → loading → pooling → noise → fitting → diagnostics →
#          sampling → uncertainty → plotting → run
#
#  Expects BASE_PATH and DIR_LIST to be defined by the enclosing script.
#  Requires: JLD2, Dierckx, Statistics, Printf, Random, Plots, LaTeXStrings
# =====================================================================
 

# --------------------------------------------------------------- configuration
 
"""
    FIT_OPTS
 
Grouping and fitting options, in one place.
 
Every fit in this section must use the same options or its diagnostics and
uncertainty band describe a different curve than the one plotted — which is why
they live here rather than at each call site. Splat `FIT_OPTS...` into
[`combine_fields`](@ref), [`bootstrap_band`](@ref) and [`bootstrap_domain`](@ref)
alike.
 
- `rtol`, `atol` — setpoint grouping tolerance, `max(atol, rtol * I)`. A purely
  absolute tolerance cannot work here: the sweep is log-spaced at low current
  (setpoints ~5e-4 A apart near zero) and linear above 0.1 A (0.05 A apart), so
  one value either splits the readback jitter at 1 A into ten knots or merges
  the whole low-current decade into one group. `rtol = 0.02` sits above the
  ~0.2 % readback jitter and well below the 5 % fractional gap at the top of the
  range; `atol = 1.5e-4` takes over near zero, where a relative tolerance
  vanishes, and is small enough to keep the log-spaced low-current setpoints
  apart.
- `s_scale` — smoothing inflation, see [`fit_channel`](@ref).
- `robust`, `zmax` — outlier rejection against the calibrated noise scale, see
  [`fit_channel_calibrated`](@ref).
 
Verify `rtol`/`atol` with [`setpoint_gaps`](@ref) after any change: `n` should be
close to the number of runs at shared setpoints, and no group should span a
decade of current.
"""
const FIT_OPTS = (rtol = 0.02, atol = 1e-4, s_scale = 25.0, robust = true, zmax = 4.0)

# --------------------------------------------------------------- loading

"""
    read_runs(dirs = DIR_LIST; b0_sign = -1) -> NamedTuple
 
Per-run current and field sweeps, kept separate: `dirs`, plus `I`, `B0`, `B1` as
vectors-of-vectors indexed like `dirs`.
 
`b0_sign = -1` flips `SG0BfieldInTesla`, whose stored sign convention is opposite
to SG1's. Applying it here, at the single point of entry, is what keeps the fit
and the plots from disagreeing; pass `b0_sign = 1` if the raw files are ever
corrected upstream.
 
Pass a subset of dates to analyse fewer runs — every function below takes the
result as an argument, so nothing else changes. Bind subsets to **distinct
names** (`runs_all`, `runs_late`, …): rebinding `runs` silently changes what
every later call sees, and a plot built from the wrong binding looks plausible.
 
Errors on ragged columns within a run, which indicates a mismatched write rather
than a short sweep.
"""
function read_runs(dirs::AbstractVector{<:AbstractString} = DIR_LIST; b0_sign::Real = -1)
    runs = map(dirs) do dir
        meta = load(joinpath(BASE_PATH, "EXPERIMENTS", dir, "data_processed.jld2"), "meta")
        r = (I  = Vector{Float64}(meta["SG1currentInA"]),
             B0 = b0_sign .* Vector{Float64}(meta["SG0BfieldInTesla"]),
             B1 = Vector{Float64}(meta["SG1BfieldInTesla"]))
        allequal(length.(values(r))) || error("$dir: ragged columns $(map(length, r))")
        r
    end
    (dirs = collect(dirs),
     I = [r.I for r in runs], B0 = [r.B0 for r in runs], B1 = [r.B1 for r in runs])
end

# --------------------------------------------------------------- pooling

"""
    pool_runs(runs) -> NamedTuple

Flatten per-run sweeps into one point cloud sorted by current: `I`, `B0`, `B1`,
and `run` (the index into `runs.dirs` each point came from).

Sorting by `I` is what makes the pooled cloud usable — setpoint grouping and the
second-difference noise estimator both assume neighbouring entries are
neighbouring currents, regardless of which run supplied them.
"""
function pool_runs(runs::NamedTuple)
    I   = reduce(vcat, runs.I)
    B0  = reduce(vcat, runs.B0)
    B1  = reduce(vcat, runs.B1)
    run = reduce(vcat, [fill(i, length(v)) for (i, v) in enumerate(runs.I)])
    p   = sortperm(I)
    (I = I[p], B0 = B0[p], B1 = B1[p], run = run[p])
end

"""
    group_setpoints(pooled; rtol = FIT_OPTS.rtol, atol = FIT_OPTS.atol) -> NamedTuple

Collapse the pooled cloud onto unique current setpoints, averaging the replicate
measurements at each. Returns `I` (group mean current), `B0`, `B1` (group means),
`n` (replicate count), `B0_sd`, `B1_sd` (within-group sample SD, `NaN` for
singletons), `runs` (contributing run indices), and `index` (which group each
*pooled row* fell into — length `length(pooled.I)`, not `length(I)`).

Consecutive points join the same group when their currents differ by less than
`max(atol, rtol * I)`; see [`FIT_OPTS`](@ref) for why the tolerance must be
relative. Runs revisit anchor setpoints (0 A, top current) many times, so
replicates are the rule rather than the exception; averaging them here means the
spline sees each setpoint once, weighted by how many measurements back it.

Grouping is single-linkage on the sorted currents, so too loose a tolerance
chains a whole sweep into one group. Check with [`setpoint_gaps`](@ref): `n`
should be close to the number of runs at shared setpoints, and `length(I)` should
match the number of distinct currents actually set, not the number of raw rows.
"""
function group_setpoints(pooled::NamedTuple;
                         rtol::Real = FIT_OPTS.rtol, atol::Real = FIT_OPTS.atol)
    tol(a, b) = max(atol, rtol * max(abs(a), abs(b)))
    breaks = [i + 1 for i in 1:length(pooled.I)-1
              if pooled.I[i+1] - pooled.I[i] > tol(pooled.I[i], pooled.I[i+1])]
    starts = [1; breaks]
    groups = [a:b for (a, b) in zip(starts, [starts[2:end] .- 1; length(pooled.I)])]

    index = similar(pooled.run)
    for (j, g) in enumerate(groups)
        index[g] .= j
    end
    sd(v) = length(v) > 1 ? std(v) : NaN
    (I     = [mean(@view pooled.I[g])  for g in groups],
     B0    = [mean(@view pooled.B0[g]) for g in groups],
     B1    = [mean(@view pooled.B1[g]) for g in groups],
     B0_sd = [sd(@view pooled.B0[g])   for g in groups],
     B1_sd = [sd(@view pooled.B1[g])   for g in groups],
     n     = [length(g) for g in groups],
     runs  = [unique(@view pooled.run[g]) for g in groups],
     index = index)
end

"""
    select_setpoints(setpoints, keep) -> NamedTuple

Restrict `setpoints` to the group indices `keep`, preserving the per-setpoint
columns and dropping `index` (whose length tracks pooled rows, not setpoints, so
it cannot be subset the same way). Used by [`fit_channel_robust`](@ref).
"""
select_setpoints(setpoints::NamedTuple, keep::AbstractVector{<:Integer}) =
    (I = setpoints.I[keep], B0 = setpoints.B0[keep], B1 = setpoints.B1[keep],
     B0_sd = setpoints.B0_sd[keep], B1_sd = setpoints.B1_sd[keep],
     n = setpoints.n[keep], runs = setpoints.runs[keep])

# --------------------------------------------------------------- noise

"""
    replicate_noise(sd, n) -> NamedTuple

Pooled within-setpoint noise from replicate scatter: `σ`, its degrees of freedom
`dof`, and `n_groups`, the number of setpoints that had replicates.

The reference noise estimate — model-free *and* curvature-free, unlike
[`noise_sigma`](@ref). Compare the two: `σ` far below the second-difference
figure means that one was absorbing curvature in `B(I)`; `σ` far above it means
replicates disagree more than neighbouring setpoints do, which points at drift
within a run rather than measurement noise.

Returns `σ = NaN` when nothing is replicated.

!!! note "Repeatability, not reproducibility"
    Replicates taken back-to-back measure short-term repeatability. Run-to-run
    reproducibility — hysteresis, re-calibration, probe repositioning — is
    larger, and it is the relevant uncertainty for a curve fitted across dates.
    `s_scale` in [`fit_channel`](@ref) and [`bootstrap_band`](@ref) exist to
    bridge that gap.
"""
function replicate_noise(sd::AbstractVector, n::AbstractVector)
    keep = findall(i -> n[i] > 1 && isfinite(sd[i]), eachindex(n))
    isempty(keep) && return (σ = NaN, dof = 0, n_groups = 0)
    dof = sum(n[i] - 1 for i in keep)
    (σ = sqrt(sum(sd[i]^2 * (n[i] - 1) for i in keep) / dof),
     dof = dof, n_groups = length(keep))
end

"""
    noise_sigma(x, y) -> Float64

Measurement noise of `y(x)` from second-difference pseudo-residuals (Gasser,
Sroka & Jennen-Steinmetz), for `x` sorted ascending and possibly unevenly spaced.

Each interior point is compared against the straight line through its two
neighbours, so a smooth trend cancels and point-to-point scatter remains. Needs
no model and cannot be deflated by overfitting, but it does absorb curvature —
so [`replicate_noise`](@ref) is preferred where replicates exist, and this is
the fallback for channels or subranges without them.
"""
function noise_sigma(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n ≥ 3 || error("need at least 3 points, got $n")
    acc = 0.0
    for i in 2:n-1
        h = x[i+1] - x[i-1]
        h > 0 || continue                       # coincident currents contribute nothing
        a, b = (x[i+1] - x[i]) / h, (x[i] - x[i-1]) / h
        ε = a * y[i-1] + b * y[i+1] - y[i]
        acc += ε^2 / (a^2 + b^2 + 1)
    end
    sqrt(acc / (n - 2))
end

# --------------------------------------------------------------- fitting

"""
    fit_channel(setpoints, channel; k = 3, σ = nothing, s_scale = FIT_OPTS.s_scale)

Smoothing-spline calibration of one field channel (`:B0` or `:B1`) against
current, fitted to grouped setpoints from [`group_setpoints`](@ref).

Each setpoint is weighted by `√n / σ`, because the mean of `n` replicates has
standard error `σ/√n` — without this the heavily-replicated anchors are
under-weighted relative to the information they carry. With those weights
Dierckx's criterion is a chi-square, so the smoothing target is
`s = s_scale * length(I)`: one unit of misfit per setpoint, scaled.

`s_scale > 1` is needed here because σ comes from back-to-back replicates and so
understates the uncertainty of a setpoint mean taken across months. With
`s_scale = 1` the spline near-interpolates, and near-coincident knots then make
it ring violently between sparse setpoints. `s_scale ≈ (σ_between/σ)²`, or
equivalently the `χ²_red` of an unsmoothed fit, is the principled value.

Returns `spline` (callable, tesla vs amp), `σ` and `σ_source` (`:replicates`,
`:second_differences` or `:given`), `σ_replicate`, `resid` and `zresid`
(setpoint residuals, absolute and σ-normalised), `χ²_red`, `n_knots`, `kept`
(the setpoint indices fitted — all of them here), and `domain` (the spline's
own valid interval, from its knots).

`χ²_red` near 1 means the curve fits as well as the noise allows; ≫ 1 means too
stiff a spline, a drifting run, or an underestimated σ. `bc = "error"` makes
evaluation outside `domain` throw rather than silently return the edge value,
which would otherwise appear as a plausible-looking plateau.
"""
function fit_channel(setpoints::NamedTuple, channel::Symbol; k::Int = 3,
                     σ::Union{Nothing,Real} = nothing,
                     s_scale::Real = FIT_OPTS.s_scale)
    I, B = setpoints.I, setpoints[channel]
    sd   = setpoints[Symbol(channel, :_sd)]
    length(I) > k + 1 || error("$(length(I)) setpoints is too few for a degree-$k spline")

    rep = replicate_noise(sd, setpoints.n)
    σ_used, source = if σ !== nothing
        Float64(σ), :given
    elseif isfinite(rep.σ) && rep.σ > 0
        rep.σ, :replicates
    else
        noise_sigma(I, B), :second_differences
    end

    w      = sqrt.(setpoints.n) ./ σ_used
    spline = Spline1D(I, B; w = w, k = k, s = s_scale * length(I), bc = "error")
    knots  = get_knots(spline)
    resid  = B .- spline.(I)
    (spline = spline, σ = σ_used, σ_source = source, σ_replicate = rep,
     resid = resid, zresid = resid .* w,
     χ²_red = sum(abs2, resid .* w) / max(length(I) - k - 1, 1),
     n_knots = length(knots),
     kept = collect(eachindex(I)),
     domain = (first(knots), last(knots)))
end

"""
    fit_channel_robust(setpoints, channel; zmax = FIT_OPTS.zmax, passes = 2, kwargs...)

Fit, flag setpoints beyond `zmax` σ, refit without them, up to `passes` times.
Same fields as [`fit_channel`](@ref), with `kept` narrowed to the surviving
setpoints and `dropped` listing the excluded ones.

A single bad reading — a probe returning ≈0, an interrupted point — otherwise
drags the spline into a large excursion that reads as structure. `zmax = 4`
retains genuine scatter and removes only what no noise model explains.

Always inspect `dropped` rather than trusting it: a rejection at a setpoint with
`n > 1` discards good measurements along with the bad one, and several rejections
in one current band mean that band has a real problem, not an outlier. A burst of
rejections right after a change to `rtol`/`atol` usually means the grouping is
fragmenting setpoints, not that the data went bad.
"""
function fit_channel_robust(setpoints::NamedTuple, channel::Symbol;
                            zmax::Real = FIT_OPTS.zmax, passes::Int = 2, kwargs...)
    keep = collect(eachindex(setpoints.I))
    fit  = fit_channel(setpoints, channel; kwargs...)
    for _ in 1:passes
        bad = findall(>(zmax), abs.(fit.zresid))
        isempty(bad) && break
        keep = deleteat!(copy(keep), sort(bad))
        fit  = fit_channel(select_setpoints(setpoints, keep), channel; kwargs...)
    end
    merge(fit, (kept = keep, dropped = setdiff(eachindex(setpoints.I), keep)))
end
 
"""
    fit_channel_calibrated(setpoints, channel; zmax, passes, σ_floor = 1.0, σ = nothing, kwargs...)
 
Fit with σ rescaled to the misfit the data actually exhibit, then reject outliers
against that scale. This is the fitter [`combine_fields`](@ref) uses.
 
The replicate σ measures back-to-back repeatability, but runs are separated by
weeks, so real setpoint-to-setpoint reproducibility is several times larger. Left
uncorrected, the weights are too tight: the spline near-interpolates and the
robust pass rejects legitimate scatter — and the severity depends on how many
runs are in the pool, so a threshold tuned on one subset misbehaves on another.
 
A first pass gives z-scores; `σ_scale = 1.4826 · median(|z|)` is their robust
spread, equal to `√χ²_red` for clean data but immune to a few bad points (a mean
misfit would be inflated by the very outliers it is meant to help detect).
Refitting with `σ_scale · σ` makes `χ²_red ≈ 1` by construction, so `zmax` means
the same thing on any subset, and the smoothing is corrected at the same time,
since `s` is expressed in units of σ.
 
`σ_floor = 1.0` forbids shrinking σ: a fit that is already smoother than the
noise should not be tightened. Passing `σ` explicitly skips calibration entirely.
 
Adds `σ_scale` to the usual fields. A `σ_scale` far above ~3, or one that differs
sharply between subsets, means a run is drifting rather than that the noise was
underestimated — check [`run_offsets`](@ref).
"""
function fit_channel_calibrated(setpoints::NamedTuple, channel::Symbol;
                                zmax::Real = FIT_OPTS.zmax, passes::Int = 2,
                                σ_floor::Real = 1.0,
                                σ::Union{Nothing,Real} = nothing, kwargs...)
    if σ !== nothing
        fit = fit_channel_robust(setpoints, channel;
                                 zmax = zmax, passes = passes, σ = σ, kwargs...)
        return merge(fit, (σ_scale = 1.0,))
    end
    probe = fit_channel(setpoints, channel; kwargs...)
    scale = max(σ_floor, 1.4826 * median(abs.(probe.zresid)))
    fit   = fit_channel_robust(setpoints, channel; zmax = zmax, passes = passes,
                               σ = scale * probe.σ, kwargs...)
    merge(fit, (σ_scale = scale, σ_replicate = probe.σ_replicate))
end

"""
    combine_fields(runs; rtol, atol, robust, kwargs...) -> NamedTuple
 
The single combined dataset. Returns the `pooled` cloud, the averaged
`setpoints`, fitted calibrations `B0` and `B1`, and three current intervals:
 
- `I_range` — covered by *every* run; includes 0 A
- `I_span`  — full extent of the pooled measurements
- `I_fit`   — where **both** splines are valid, from their knots, shrunk a few
              ulps so grid endpoints cannot fall outside by rounding
 
`I_fit` is the authoritative domain and is narrower than `I_span`: a spline's
outermost knots sit inside the first and last setpoint means, and robust fitting
may drop an extreme setpoint from one channel but not the other. Everything that
evaluates a spline clamps to it.
 
`robust = true` uses [`fit_channel_calibrated`](@ref); `false` uses the raw
[`fit_channel`](@ref) with no rescaling and no rejection. `kwargs` (`k`, `σ`,
`s_scale`, `zmax`, `passes`, `σ_floor`) go to the fitter; defaults come from
[`FIT_OPTS`](@ref).
"""
function combine_fields(runs::NamedTuple;
                        rtol::Real = FIT_OPTS.rtol, atol::Real = FIT_OPTS.atol,
                        robust::Bool = FIT_OPTS.robust, kwargs...)
    pooled    = pool_runs(runs)
    setpoints = group_setpoints(pooled; rtol = rtol, atol = atol)
 
    # fit_channel knows nothing of rejection, so strip those keywords when the
    # plain fitter is requested (FIT_OPTS carries zmax unconditionally).
    fitter, fitkw = if robust
        fit_channel_calibrated, values(kwargs)
    else
        fit_channel, Base.structdiff(values(kwargs),
                                     NamedTuple{(:zmax, :passes, :σ_floor)})
    end
    b0 = fitter(setpoints, :B0; fitkw...)
    b1 = fitter(setpoints, :B1; fitkw...)
 
    # Intersection of the two splines' own domains, with a few ulps of margin:
    # range() and logrange() need not reproduce their endpoints exactly, and
    # bc = "error" rejects a point an epsilon outside.
    dom = (nextfloat(max(b0.domain[1], b1.domain[1]), 8),
           prevfloat(min(b0.domain[2], b1.domain[2]), 8))
 
    (pooled = pooled, setpoints = setpoints, B0 = b0, B1 = b1,
     I_range = (maximum(minimum, runs.I), minimum(maximum, runs.I)),
     I_span  = (minimum(pooled.I), maximum(pooled.I)),
     I_fit   = dom)
end

# --------------------------------------------------------------- diagnostics

"""
    fit_summary(combined; channels = (:B0, :B1)) -> Nothing
 
Per channel: the noise scale and how far it was rescaled, achieved `χ²_red`,
knots against fitted setpoints, and any rejected setpoints.
 
Read three numbers. `×f` is the reproducibility factor — how much larger the real
setpoint-to-setpoint scatter is than back-to-back repeatability; 2–3 is ordinary,
much more suggests a drifting run. `χ²_red` should now sit near 1 by
construction, so a value far from it means the rescaling hit `σ_floor` or the
spline is genuinely too stiff. The knots/setpoints ratio is the overfitting
gauge: near 1 means the spline interpolates and any wiggle is tracked noise;
0.1–0.4 is healthy for a smooth monotone calibration.
"""
function fit_summary(combined::NamedTuple; channels::Tuple = (:B0, :B1))
    sp = combined.setpoints
    @printf("%d raw points → %d setpoints (max %d replicates)\n",
            length(combined.pooled.I), length(sp.I), maximum(sp.n))
    @printf("I: all runs [%.4f, %.4f], pooled [%.4f, %.4f], splines [%.4f, %.4f] A\n",
            combined.I_range..., combined.I_span..., combined.I_fit...)
    for ch in channels
        f = combined[ch]
        @printf("%s: σ = %.4g T (×%.2f of replicate %.4g T, %d groups, %d dof)  χ²_red = %.2f  knots %d/%d = %.2f  dropped %d\n",
                ch, f.σ, get(f, :σ_scale, 1.0), f.σ_replicate.σ,
                f.σ_replicate.n_groups, f.σ_replicate.dof, f.χ²_red,
                f.n_knots, length(f.kept), f.n_knots / length(f.kept),
                length(get(f, :dropped, Int[])))
        for j in get(f, :dropped, Int[])
            @printf("   dropped I = %.4f A  n = %d  %s = %.5g T\n",
                    sp.I[j], sp.n[j], ch, sp[ch][j])
        end
    end
    return nothing
end

"""
    setpoint_gaps(combined; below = 1e-2) -> Nothing
 
List consecutive setpoints closer than `below`, with their replicate counts, and
report the minimum and median gap.
 
The grouping check. A minimum gap orders of magnitude below the median means the
tolerance is fragmenting one physical setpoint into several — near-coincident
knots, hence violent oscillation between well-behaved points. Conversely a group
whose neighbours are a decade away has swallowed several real setpoints, and its
mean current matches none of them. Every gap listed should correspond to a
current actually set.
 
An alternating `n = 1, 7, 1, 7` pattern is not fragmentation: it means some runs
sampled currents the others did not. [`singleton_setpoints`](@ref) names them.
"""
function setpoint_gaps(combined::NamedTuple; below::Real = 1e-2)
    sp = combined.setpoints
    d  = diff(sp.I)
    @printf("%d setpoints, min gap %.2e A, median gap %.2e A\n",
            length(sp.I), minimum(d), median(d))
    close_pairs = findall(<(below), d)
    @printf("gaps below %.1e A: %d\n", below, length(close_pairs))
    for j in close_pairs
        @printf("  %.5f → %.5f  (Δ=%.2e)  n = %d, %d\n",
                sp.I[j], sp.I[j+1], d[j], sp.n[j], sp.n[j+1])
    end
    println("replicate counts: ", sort(unique(sp.n)))
    return nothing
end


"""
    singleton_setpoints(combined, runs; channel = :B1) -> Nothing
 
List setpoints measured by a single run, naming that run.
 
These arise when some dates used a finer sweep than others. They are legitimate
data, but they carry no replicate averaging, so they are the setpoints most
likely to be rejected — and if they cluster in one run, that run is the sole
source of a whole current range, which also caps how far a bootstrap band can
extend (see [`bootstrap_domain`](@ref)).
"""
function singleton_setpoints(combined::NamedTuple, runs::NamedTuple;
                             channel::Symbol = :B1)
    sp = combined.setpoints
    rows = findall(==(1), sp.n)
    isempty(rows) && return println("no single-run setpoints")
    @printf("%d setpoints measured by one run only:\n", length(rows))
    for j in rows
        @printf("  I = %.5f A  run %-10s %s = %.5g T\n",
                sp.I[j], runs.dirs[only(sp.runs[j])], channel, sp[channel][j])
    end
    return nothing
end
 
"""
    run_offsets(combined, channel; dirs) -> Nothing
 
Per-run mean σ-normalised residual about the shared calibration.
 
The drift test a common current grid would have given directly: if all runs
sample one curve, each run's residuals scatter about zero and `mean/SE` sits
within roughly ±2. A run several SE away has a systematically different `B(I)` —
a re-calibration, a moved probe, a different zero — and pooling it biases every
other run. Setpoints count once per contributing run; rejected ones are skipped.
 
Identical values across every run mean the per-run selection is not
discriminating — usually because grouping merged so much that every group
contains every run.
"""
function run_offsets(combined::NamedTuple, channel::Symbol;
                     dirs::AbstractVector = 1:maximum(combined.pooled.run))
    fit, sp = combined[channel], combined.setpoints
    @printf("%-10s %6s %12s %10s\n", "run", "n_pts", "mean z", "mean/SE")
    for i in eachindex(dirs)
        z = [fit.zresid[pos] for (pos, j) in enumerate(fit.kept) if i in sp.runs[j]]
        isempty(z) && continue
        @printf("%-10s %6d %12.4g %10.2f\n",
                dirs[i], length(z), mean(z), mean(z) * sqrt(length(z)))
    end
    return nothing
end


"""
    noise_model(combined, channel) -> Nothing
 
Within-setpoint SD against field magnitude, in decade bins.
 
Additive noise (SD roughly constant) justifies the constant-σ weighting used
here; multiplicative noise (SD rising with |B|) means the high-current end
dominates the fit while the low-current end is under-resolved. A relative SD near
100 % in the lowest decade means the field there is at or below the probe's noise
floor and no fit can calibrate it — cut the range instead.
"""
function noise_model(combined::NamedTuple, channel::Symbol)
    sp = combined.setpoints
    B, sd = sp[channel], sp[Symbol(channel, :_sd)]
    keep = findall(i -> sp.n[i] > 1 && isfinite(sd[i]) && B[i] != 0, eachindex(sp.n))
    isempty(keep) && return println("$channel: no replicated setpoints")
    @printf("%-12s %6s %12s %12s\n", "|B| decade", "n", "median SD", "median SD/|B|")
    for d in sort(unique(floor(Int, log10(abs(B[i]))) for i in keep))
        rows = filter(i -> floor(Int, log10(abs(B[i]))) == d, keep)
        @printf("1e%-10d %6d %12.4g %12.4g\n", d, length(rows),
                median(sd[rows]), median(sd[rows] ./ abs.(B[rows])))
    end
    return nothing
end
 
"""
    channel_difference(pooled) -> NamedTuple
 
Row-wise `B0 - B1` statistics. Both channels are recorded at the same currents
within a run, so this needs no fit and no grouping.
 
A `ratio_median` well away from 1 with a small `mean` means the probes differ by
a scale factor — two positions in the same gradient, which is expected and needs
no correction beyond keeping the channels fitted separately. A large `mean` with
`ratio_median ≈ 1` would instead indicate an additive offset.
"""
channel_difference(pooled::NamedTuple) =
    (mean = mean(pooled.B0 .- pooled.B1),
     std  = std(pooled.B0 .- pooled.B1),
     ratio_median = median(pooled.B0 ./ pooled.B1))

"""
    compare_subsets(subsets; channel = :B1, kwargs...) -> Nothing
 
Fit each named subset of runs and tabulate the fit side by side.
 
`subsets` is a NamedTuple of [`read_runs`](@ref) results, e.g.
`(all = runs_all, late = runs_late)`. Use it to ask whether a suspect date
matters: a `×f` that falls sharply when one run is removed means that run
dominates the scatter, and a domain that shrinks means it was the sole source of
some setpoints. Expect the replicate σ to *rise* on smaller subsets — fewer
replicates per setpoint, less to estimate from.
"""
function compare_subsets(subsets::NamedTuple; channel::Symbol = :B1, kwargs...)
    @printf("%-10s %5s %7s %11s %6s %8s %20s\n",
            "subset", "runs", "setpts", "σ [T]", "×f", "χ²_red", "domain [A]")
    for (name, r) in pairs(subsets)
        c = combine_fields(r; kwargs...)
        f = c[channel]
        @printf("%-10s %5d %7d %11.4g %6.2f %8.2f  [%.4f, %.4f]\n",
                name, length(r.dirs), length(c.setpoints.I), f.σ,
                get(f, :σ_scale, 1.0), f.χ²_red, c.I_fit...)
    end
    return nothing
end


"""
    fields_at(combined, I) -> NamedTuple
 
Evaluate both calibrations at the currents `I`: returns `I` (clamped), `B0`,
`B1`, and the per-point measurement noise `B0_err`, `B1_err`.
 
Currents are clamped to `combined.I_fit`, which is the only reliable way to avoid
Dierckx's "Input point out of range": the domain is set by knot positions that
neither `range` nor `logrange` is guaranteed to reproduce exactly, and robust
fitting can pull it inside `I_span` by a whole setpoint. Clamping at the single
point of evaluation means no caller has to know this — but compare the returned
`I` against what you asked for if it matters.
 
`B*_err` is single-measurement noise, not the uncertainty of the fitted curve —
use [`bootstrap_band`](@ref) for that.
"""
function fields_at(combined::NamedTuple, I::AbstractVector)
    Iv = clamp.(collect(float.(I)), combined.I_fit...)
    (I = Iv,
     B0 = combined.B0.spline.(Iv), B1 = combined.B1.spline.(Iv),
     B0_err = fill(combined.B0.σ, length(Iv)),
     B1_err = fill(combined.B1.σ, length(Iv)))
end
 
"""
    field_function(combined, channel) -> Function
 
A single-argument `B(I)` in tesla, clamped to the fitted domain, for passing to
downstream code that wants a plain callable.
 
Clamping means a current below the calibrated range returns the edge value rather
than throwing; check `combined.I_fit` if that distinction matters.
"""
field_function(combined::NamedTuple, channel::Symbol) =
    let sp = combined[channel].spline, dom = combined.I_fit
        I -> sp(clamp(float(I), dom...))
    end
 
"""
    sample_fields(combined; n = 200, scale = :log, lo = nothing, hi = nothing)
 
Both calibrations on a common current grid of `n` points, as [`fields_at`](@ref)
plus `scale`.
 
`scale = :log` spaces currents geometrically (`Base.logrange`, Julia ≥ 1.11),
`:linear` uniformly. Bounds default to `combined.I_range` — the interval every run
covers — clipped to `combined.I_fit`.
 
A log grid needs a positive lower bound: when the requested one is ≤ 0, as it is
whenever the sweep includes 0 A, it is raised to the smallest positive setpoint
and a notice is printed. The grid then no longer reaches the zero-field end; keep
`I = 0` as a separate point if you need it.
 
!!! note "Correlated points"
    The spline's resolution is roughly uniform in `I`, so a log grid oversamples
    the low-current decade: those points read one spline segment at high density
    and are strongly correlated. `B*_err` reflects per-point noise only.
"""
function sample_fields(combined::NamedTuple; n::Int = 200, scale::Symbol = :log,
                       lo::Union{Nothing,Real} = nothing,
                       hi::Union{Nothing,Real} = nothing)
    n ≥ 2 || error("need at least 2 grid points, got $n")
    lo = clamp(something(lo, combined.I_range[1]), combined.I_fit...)
    hi = clamp(something(hi, combined.I_range[2]), combined.I_fit...)
    lo < hi || error("empty current range [$lo, $hi]")
 
    I = if scale === :linear
        collect(range(lo, hi; length = n))
    elseif scale === :log
        if lo ≤ 0
            positive = filter(>(0), combined.setpoints.I)
            isempty(positive) && error("no positive currents — a log grid is impossible")
            lo = clamp(minimum(positive), combined.I_fit...)
            @info "Log grid lower bound raised to the smallest positive setpoint" lo
        end
        collect(logrange(lo, hi, n))
    else
        error("scale must be :linear or :log, got :$scale")
    end
 
    merge(fields_at(combined, I), (scale = scale,))
end
 
# --------------------------------------------------------------- uncertainty
 
"""
    bootstrap_domain(runs; n_probe = 40, level = 1.0, rng, kwargs...) -> Tuple
 
The current interval that (almost) every bootstrap resample can be fitted across,
found by fitting `n_probe` resamples and intersecting their domains.
 
A resample drawing some runs twice omits others, losing any setpoint only those
runs measured, so its spline domain is narrower than the full fit's — and a grid
built from `combined.I_fit` then falls outside it, which is why an unguarded
bootstrap skips every resample. `level = 1.0` takes the worst case; lower it to
0.95 to tolerate a few skips in exchange for a wider band.
 
`kwargs` must match the fit options used everywhere else.
"""
function bootstrap_domain(runs::NamedTuple; n_probe::Int = 40, level::Real = 1.0,
                          rng = Random.default_rng(), kwargs...)
    n = length(runs.dirs)
    los, his = Float64[], Float64[]
    for _ in 1:n_probe
        pick = rand(rng, 1:n, n)
        s = (dirs = runs.dirs[pick], I = runs.I[pick],
             B0 = runs.B0[pick], B1 = runs.B1[pick])
        try
            d = combine_fields(s; kwargs...).I_fit
            push!(los, d[1]); push!(his, d[2])
        catch err
            err isa ArgumentError || err isa ErrorException || rethrow()
        end
    end
    isempty(los) && error("no resample could be fitted — check the fit options")
    (quantile(los, level), quantile(his, 1 - level))
end
 
"""
    bootstrap_band(runs, channel; grid, n_boot = 200, rng, sanity = 0.25, kwargs...)
 
Pointwise uncertainty of a fitted calibration, by resampling **runs** with
replacement `n_boot` times, refitting, and evaluating each fit on `grid`.
 
Returns `I`, `sd` (bootstrap standard deviation per grid point), `lo`/`hi` (16th
and 84th percentiles — a ±1σ-equivalent interval needing no normality
assumption), and the counts `n_boot`, `n_rejected`, `n_skipped`.
 
Resampling whole runs is what makes this the right error bar: it captures
run-to-run reproducibility alongside measurement noise, whereas the per-point σ
sees only short-term repeatability. Expect the band to be narrow where setpoints
are dense and replicated, and to flare at the ends.
 
Two guards. Resamples whose domain does not cover `grid` are skipped — use
[`bootstrap_domain`](@ref) to choose a grid that keeps `n_skipped` near zero.
Curves leaving the measured data's envelope by more than `sanity` times its range
are rejected: they describe spline instability, not calibration uncertainty, and
would dominate the percentiles. `n_rejected` above ~10 % of `n_boot` means the
fit is too lightly smoothed — raise `s_scale` rather than `sanity`.
 
!!! warning "Pass the same options as the main fit"
    `kwargs` are forwarded to [`combine_fields`](@ref) and must match the ones
    used for the plotted curve, or the band describes a differently-smoothed
    function. Splat `FIT_OPTS...` into both.
 
!!! note "Expect an invisible ribbon"
    With seven runs and many replicates the band is often ~0.1–1 mT against a
    0.75 T curve — thinner than the plotted line. That is a well-determined
    calibration, not a failure; [`plot_band_width`](@ref) is the readable view.
"""
function bootstrap_band(runs::NamedTuple, channel::Symbol;
                        grid::AbstractVector, n_boot::Int = 200,
                        rng = Random.default_rng(), sanity::Real = 0.25,
                        kwargs...)
    n_runs = length(runs.dirs)
    all_B  = reduce(vcat, getproperty(runs, channel))
    span   = maximum(all_B) - minimum(all_B)
    bounds = (minimum(all_B) - sanity * span, maximum(all_B) + sanity * span)
 
    curves   = Vector{Vector{Float64}}()
    rejected = 0
    skipped  = 0
    for _ in 1:n_boot
        pick = rand(rng, 1:n_runs, n_runs)
        sample = (dirs = runs.dirs[pick], I = runs.I[pick],
                  B0 = runs.B0[pick], B1 = runs.B1[pick])
        try
            c = combine_fields(sample; kwargs...)
            lo, hi = c.I_fit
            if !all(x -> lo ≤ x ≤ hi, grid)
                skipped += 1
                continue
            end
            y = getproperty(c, channel).spline.(grid)
            all(v -> bounds[1] ≤ v ≤ bounds[2], y) ? push!(curves, y) : (rejected += 1)
        catch err
            err isa ArgumentError || err isa ErrorException || rethrow()
            skipped += 1
        end
    end
    length(curves) ≥ 20 || error("only $(length(curves)) usable resamples " *
                                 "($rejected unstable, $skipped skipped) — " *
                                 "narrow the grid with bootstrap_domain")
 
    col(j) = [c[j] for c in curves]
    (I = collect(grid),
     sd = [std(col(j)) for j in eachindex(grid)],
     lo = [quantile(col(j), 0.16) for j in eachindex(grid)],
     hi = [quantile(col(j), 0.84) for j in eachindex(grid)],
     n_boot = length(curves), n_rejected = rejected, n_skipped = skipped)
end
 
# --------------------------------------------------------------- plotting
 
"""
    plot_run_currents(runs; cols) -> Plots.Plot
 
One column of markers per run showing which currents that run sampled, on a log
axis. The quickest read on grid incompatibility: differing column patterns are
exactly why the runs are combined by fitting rather than by averaging rows.
 
Zero-current points fall off a log axis silently — `ylim` starts at 1e-4, so the
0 A setpoint every run records is deliberately not shown.
"""
function plot_run_currents(runs::NamedTuple;
                           cols = palette(:darkrainbow, length(runs.dirs)))
    n = length(runs.dirs)
    fig = plot(title = "Sampled SG1 currents per run", titlefontsize = 12,
               legend = false, xgrid = false, gridalpha = 0.25, gridstyle = :dot,
               minorgridalpha = 0.05, tickfontsize = 11, guidefontsize = 14)
    for i in 1:n
        scatter!(fig, fill(i, length(runs.I[i])), runs.I[i];
                 marker = (:circle, :white, 2.5),
                 markerstrokecolor = cols[i], markerstrokewidth = 1.5)
    end
    plot!(fig,
          ylim = (1e-4, 1.05), xlim = (0, n + 1),
          yaxis = (:log10, L"$I_{0} \ (\mathrm{A})$"),
          xticks = (1:n, runs.dirs),
          yticks = ([1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                    [L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
          xminorticks = false, xrotation = 75,
          bottom_margin = -2Plots.mm, left_margin = 6Plots.mm,
          right_margin = 6Plots.mm, size = (350, 720))
    return fig
end
 
"""
    calibration_panel!(fig, runs, combined; bands, cols, n_grid, xscale, yscale, legend)
 
Draw the calibration into an existing plot: raw per-run measurements (circles
`B0`, squares `B1`, one colour per run), the fitted curves, and — when `bands` is
given — their uncertainty ribbons. Rejected setpoints are ringed in red.
 
`bands` is a NamedTuple of [`bootstrap_band`](@ref) results keyed by channel; the
curves are then evaluated on the bands' own grid, so ribbon and line cannot fall
out of alignment. Without bands, a linear grid of `n_grid` points over the safe
interval is used.
 
The curve grid is linear regardless of the axis scales, so `:log10` rescales an
already-smooth curve rather than concentrating samples at one end. On a log axis
non-positive values are dropped silently, so the 0 A points vanish and the limits
are taken from the smallest positive measured value instead; on a log y-axis the
ribbon's lower edge is also clamped positive.
 
`legend` is forwarded so a multi-panel figure can show the run legend once.
"""
function calibration_panel!(fig, runs::NamedTuple, combined::NamedTuple;
                            bands = nothing,
                            cols = palette(:darkrainbow, length(runs.dirs)),
                            n_grid::Int = 400,
                            xscale::Symbol = :identity, yscale::Symbol = :identity,
                            legend = :bottomright)
    curve = if bands === nothing
        sample_fields(combined; n = n_grid, scale = :linear)
    else
        gs = [getproperty(bands, ch).I for ch in (:B0, :B1)]
        all(g -> g ≈ first(gs), gs) ||
            error("band grids differ between channels — build them from one grid")
        fields_at(combined, first(gs))
    end
 
    plot!(fig; xscale = xscale, yscale = yscale, legend = legend, legendfontsize = 7,
          gridalpha = 0.25, gridstyle = :dot, tickfontsize = 11, guidefontsize = 14,
          xlabel = L"$I_{0} \ (\mathrm{A})$", ylabel = L"$B \ (\mathrm{T})$")
    if xscale === :log10
        lo = minimum(filter(>(0), reduce(vcat, runs.I)))
        plot!(fig; xlims = (0.8 * lo, 1.2 * maximum(maximum, runs.I)))
    end
    if yscale === :log10
        allB = vcat(reduce(vcat, runs.B0), reduce(vcat, runs.B1))
        lo = minimum(filter(>(0), allB))
        plot!(fig; ylims = (0.8 * lo, 1.2 * maximum(allB)))
    end
 
    for i in eachindex(runs.dirs)
        scatter!(fig, runs.I[i], runs.B0[i]; label = runs.dirs[i],
                 marker = (:circle, :white, 2.5),
                 markerstrokecolor = cols[i], markerstrokewidth = 1.5)
        scatter!(fig, runs.I[i], runs.B1[i]; label = false,
                 marker = (:square, :white, 2.5),
                 markerstrokecolor = cols[i], markerstrokewidth = 1.5)
    end
 
    for (ch, style, lbl) in ((:B0, :solid, L"$B_0$ fit"), (:B1, :dash, L"$B_1$ fit"))
        y = getproperty(curve, ch)
        # ribbon takes (below, above) widths — distances from the curve, not
        # absolute levels. On a log y-axis the lower edge must stay positive.
        rib = if bands === nothing
            nothing
        else
            b = getproperty(bands, ch)
            below = y .- b.lo
            yscale === :log10 ? (min.(below, 0.999 .* y), b.hi .- y) :
                                (below, b.hi .- y)
        end
        plot!(fig, curve.I, y; label = lbl, lw = 1.5, color = :black,
              linestyle = style, ribbon = rib, fillalpha = 0.35,
              fillcolor = :steelblue)
    end
 
    for ch in (:B0, :B1), j in get(combined[ch], :dropped, Int[])
        scatter!(fig, [combined.setpoints.I[j]], [combined.setpoints[ch][j]];
                 label = false, marker = (:circle, :transparent, 7),
                 markerstrokecolor = :red, markerstrokewidth = 2)
    end
    return fig
end
 
"""
    plot_field_calibration(runs, combined; xscale = :identity, yscale = :identity, kwargs...)
 
Single-panel calibration figure. `kwargs` go to [`calibration_panel!`](@ref).
 
`xscale = :log10` opens up the low-current decade that a linear axis collapses
into the origin; adding `yscale = :log10` turns a proportional region into a
straight line of slope 1, which is the diagnostic for whether `B ∝ I` still holds
at small current.
"""
function plot_field_calibration(runs::NamedTuple, combined::NamedTuple;
                                xscale::Symbol = :identity,
                                yscale::Symbol = :identity, kwargs...)
    fig = plot(title = "SG field calibration", titlefontsize = 12, size = (760, 480))
    calibration_panel!(fig, runs, combined; xscale = xscale, yscale = yscale, kwargs...)
end
 
"""
    plot_field_calibration_pair(runs, combined; logy = false, size, kwargs...)
 
The calibration twice side by side: linear current on the left, log current on
the right (log-log with `logy = true`).
 
The log panel is where the low-current decade is legible and where the fit's
lower domain limit becomes visible as data points with no curve through them.
The linear panel carries the run legend; the log panel drops it and its y-label
to save width.
 
`kwargs` (`bands`, `cols`, `n_grid`) go to [`calibration_panel!`](@ref).
"""
function plot_field_calibration_pair(runs::NamedTuple, combined::NamedTuple;
                                     logy::Bool = false, size = (1100, 460),
                                     kwargs...)
    left  = plot(title = "linear", titlefontsize = 11)
    right = plot(title = logy ? "log-log" : "log current", titlefontsize = 11)
    calibration_panel!(left,  runs, combined; xscale = :identity,
                       legend = :bottomright, kwargs...)
    calibration_panel!(right, runs, combined; xscale = :log10,
                       yscale = logy ? :log10 : :identity, legend = false, kwargs...)
    plot!(right; ylabel = "")
    plot(left, right; layout = (1, 2), size = size,
         plot_title = "SG field calibration", plot_titlefontsize = 13,
         left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
end
 
"""
    plot_band_width(bands) -> Plots.Plot
 
Half-width of each channel's bootstrap band against current, in mT.
 
A band of order 0.1–1 mT is invisible against a 0.75 T calibration curve, so this
is the readable view: it shows where the fit is best constrained (dense,
replicated setpoints) and where it flares (the ends of the range).
"""
function plot_band_width(bands::NamedTuple)
    fig = plot(xlabel = L"$I_{0} \ (\mathrm{A})$", ylabel = "band half-width (mT)",
               title = "Calibration uncertainty", titlefontsize = 12,
               legend = :top, gridalpha = 0.25, gridstyle = :dot, size = (760, 320))
    for (ch, lbl) in ((:B0, L"$B_0$"), (:B1, L"$B_1$"))
        b = getproperty(bands, ch)
        plot!(fig, b.I, 1e3 .* b.sd; label = lbl, lw = 2)
    end
    return fig
end


# --------------------------------------------------------------- run
runs     = read_runs();
display(plot_run_currents(runs))

combined = combine_fields(runs; FIT_OPTS...);


# Read in this order: grouping, then fit quality, then whether the runs agree,
# then the noise model.
setpoint_gaps(combined)
singleton_setpoints(combined, runs)
fit_summary(combined)
run_offsets(combined, :B1; dirs = runs.dirs)
noise_model(combined, :B1)
run_offsets(combined, :B0; dirs = runs.dirs)
noise_model(combined, :B0)
@show channel_difference(combined.pooled)

# Uncertainty band. The grid must lie inside every resample's domain, not just
# the full fit's: a resample omitting the runs that alone measured the lowest
# currents has a higher first knot.
band_lo, band_hi = bootstrap_domain(runs; FIT_OPTS...)
band_grid = range(max(band_lo, combined.I_fit[1]),
                  min(band_hi, combined.I_fit[2]); length = 400)
@printf("band grid [%.4f, %.4f] A  (full fit [%.4f, %.4f])\n",
        first(band_grid), last(band_grid), combined.I_fit...)
 
bands = (B0 = bootstrap_band(runs, :B0; grid = band_grid, n_boot = 200, FIT_OPTS...),
         B1 = bootstrap_band(runs, :B1; grid = band_grid, n_boot = 200, FIT_OPTS...))
@printf("bootstrap: B0 %d used / %d unstable / %d skipped,  B1 %d / %d / %d\n",
        bands.B0.n_boot, bands.B0.n_rejected, bands.B0.n_skipped,
        bands.B1.n_boot, bands.B1.n_rejected, bands.B1.n_skipped)
@printf("band half-width (median): B0 %.4g T, B1 %.4g T   [σ: %.4g, %.4g T]\n",
        median(bands.B0.sd), median(bands.B1.sd), combined.B0.σ, combined.B1.σ)
 
display(plot_field_calibration_pair(runs, combined; bands = bands, logy=true))
display(plot_band_width(bands))

# Calibration sampled for downstream use: logarithmic in current.
scan = sample_fields(combined; n = 1201, scale = :log, lo = 10.0e-3, hi = 1.05)

fig_Bfield = plot(
        title = "Stern–Gerlach field estimation",
        titlefontsize = 12,
        xlabel="SG current (A)",
        ylabel="Magnetic field (T)",
        legend = :bottomright,
        xgrid=false,
        gridalpha = 0.25,
        gridstyle = :dot,
        minorgridalpha = 0.05,
        tickfontsize
        =11,
        guidefontsize=14,
    );
plot!(fig_Bfield,
    scan.I, TheoreticalSimulation.BvsI.(scan.I),
    label="SG manual",
    line=(:dash, 1, :black))
plot!(scan.I,scan.B0, 
    label=L"SG experiment: $B_{0}$",
    line=(:dot, 1.5, :gray26))
plot!(scan.I,scan.B1, 
    label=L"SG experiment: $B_{1}$",
    line=(:dashdot, 2.5, :blue))
for (idx,data_directory) in enumerate(DIR_LIST)
    scatter!(fig_Bfield,
        runs.I[idx], runs.B0[idx],
        label=data_directory,
        # yerror=dI_all[idx],
        marker = (:circle, :white, 2.5),
        markerstrokecolor = cols[idx],
        markerstrokewidth = 1.5,)
    scatter!(fig_Bfield,
        runs.I[idx], runs.B1[idx],
        # yerror=dI_all[idx],
        label=data_directory,
        marker = (:square, :white, 2.5),
        markerstrokecolor = cols[idx],
        markerstrokewidth = 1.5,)
end
plot!(fig_Bfield,
    size=(600,400),
    legend_columns =2,
    legendfontsize =8,
    foreground_color_legend= nothing,)
display(fig_Bfield)
plot!(fig_Bfield,
    yticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xlims = (5e-3,1.1),
    xscale=:log10,
    ylims=(0.9e-3,1.0),
    yscale=:log10,
)
display(fig_Bfield)











































































# EXPERIMENT
n_runs = length(DIR_LIST)
I_all  = Vector{Vector{Float64}}(undef, n_runs);
B0_all = Vector{Vector{Float64}}(undef, n_runs);
B1_all = Vector{Vector{Float64}}(undef, n_runs);
cols = palette(:darkrainbow, n_runs);

for (i, dir) in enumerate(DIR_LIST)
    d   = load(joinpath(BASE_PATH, "EXPERIMENTS", dir, "data_processed.jld2"), "meta");
    I_all[i]  = Vector{Float64}(d["SG1currentInA"]);
    B0_all[i] = -1 * Vector{Float64}(d["SG0BfieldInTesla"]); # -1 due to inconsistencies with field directions
    B1_all[i] = Vector{Float64}(d["SG1BfieldInTesla"]);
end


"""
    read_runs(dirs = DIR_LIST) -> NamedTuple

Per-run current and field sweeps, kept separate: `dirs`, plus `I`, `B0`, `B1` as
vectors-of-vectors indexed like `dirs`. Errors if a run's three columns are
ragged, which would mean a mismatched write rather than a short sweep.
"""
function read_runs(dirs::AbstractVector{<:AbstractString} = DIR_LIST)
    runs = map(dirs) do dir
        meta = load(joinpath(BASE_PATH, "EXPERIMENTS", dir, "data_processed.jld2"), "meta")
        r = (I  = Vector{Float64}(meta["SG1currentInA"]),
             B0 = Vector{Float64}(meta["SG0BfieldInTesla"]),
             B1 = Vector{Float64}(meta["SG1BfieldInTesla"]))
        allequal(length.(values(r))) ||
            error("$dir: ragged columns $(map(length, r))")
        r
    end
    (dirs = collect(dirs),
     I = [r.I for r in runs], B0 = [-r.B0 for r in runs], B1 = [r.B1 for r in runs])
end

# --------------------------------------------------------------- pooling

"""
    pool_runs(runs) -> NamedTuple

Flatten per-run sweeps into one point cloud sorted by current: `I`, `B0`, `B1`,
and `run` (the index into `runs.dirs` each point came from).

Sorting by `I` is what makes the pooled cloud usable — setpoint grouping and the
noise estimator below both assume neighbouring entries are neighbouring
currents, regardless of which run supplied them.
"""
function pool_runs(runs::NamedTuple)
    I   = reduce(vcat, runs.I)
    B0  = reduce(vcat, runs.B0)
    B1  = reduce(vcat, runs.B1)
    run = reduce(vcat, [fill(i, length(v)) for (i, v) in enumerate(runs.I)])
    p   = sortperm(I)
    (I = I[p], B0 = B0[p], B1 = B1[p], run = run[p])
end

"""
    group_setpoints(pooled; atol = 1e-4) -> NamedTuple

Collapse the pooled cloud onto unique current setpoints, averaging the replicate
measurements at each. Returns `I` (group mean current), `B0`, `B1` (group means),
`n` (replicate count), `B0_sd`, `B1_sd` (within-group sample SD, `NaN` for
singletons), `runs` (contributing run indices), and `index` (the group each
pooled row fell into).

`atol` must exceed setpoint reproducibility but stay well below the sweep step:
replicates of a nominal 0.5 A setpoint may read 0.4999 / 0.5001 and must group,
while genuinely adjacent setpoints must not. Grouping is single-linkage on the
sorted currents, so too large an `atol` chains a whole sweep into one group —
check that `n` comes out as small integers and `length(I)` matches the number of
distinct setpoints you expect.
"""
function group_setpoints(pooled::NamedTuple; atol::Real = 1e-4)
    starts = [1; findall(>(atol), diff(pooled.I)) .+ 1]
    groups = [a:b for (a, b) in zip(starts, [starts[2:end] .- 1; length(pooled.I)])]
    index  = similar(pooled.run)
    for (j, g) in enumerate(groups)
        index[g] .= j
    end
    sd(v) = length(v) > 1 ? std(v) : NaN
    (I     = [mean(@view pooled.I[g])  for g in groups],
     B0    = [mean(@view pooled.B0[g]) for g in groups],
     B1    = [mean(@view pooled.B1[g]) for g in groups],
     B0_sd = [sd(@view pooled.B0[g])   for g in groups],
     B1_sd = [sd(@view pooled.B1[g])   for g in groups],
     n     = [length(g) for g in groups],
     runs  = [unique(@view pooled.run[g]) for g in groups],
     index = index)
end

# --------------------------------------------------------------- noise

"""
    replicate_noise(sd, n) -> NamedTuple

Pooled within-setpoint noise from replicate scatter: `σ`, its degrees of freedom
`dof`, and `n_groups`, the number of setpoints that had replicates.

This is the reference noise estimate — model-free *and* curvature-free, unlike
[`noise_sigma`](@ref). Compare the two: `σ` far below the second-difference
figure means that one was picking up curvature in `B(I)`; `σ` far above it means
replicates disagree more than neighbouring setpoints do, which points at drift
within a run rather than measurement noise.

Returns `σ = NaN` when nothing is replicated.
"""
function replicate_noise(sd::AbstractVector, n::AbstractVector)
    keep = findall(i -> n[i] > 1 && isfinite(sd[i]), eachindex(n))
    isempty(keep) && return (σ = NaN, dof = 0, n_groups = 0)
    dof = sum(n[i] - 1 for i in keep)
    (σ = sqrt(sum(sd[i]^2 * (n[i] - 1) for i in keep) / dof),
     dof = dof, n_groups = length(keep))
end

"""
    noise_sigma(x, y) -> Float64

Measurement noise of `y(x)` from second-difference pseudo-residuals (Gasser,
Sroka & Jennen-Steinmetz), for `x` sorted ascending and possibly unevenly spaced.

Each interior point is compared against the straight line through its two
neighbours, so any *smooth* trend cancels and point-to-point scatter remains.
Needs no model and cannot be deflated by overfitting, but it does absorb
curvature, so [`replicate_noise`](@ref) is preferred where replicates exist.
This is the fallback for channels or subranges without them.
"""
function noise_sigma(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n ≥ 3 || error("need at least 3 points, got $n")
    acc = 0.0
    for i in 2:n-1
        h = x[i+1] - x[i-1]
        h > 0 || continue                       # coincident currents contribute nothing
        a, b = (x[i+1] - x[i]) / h, (x[i] - x[i-1]) / h
        ε = a * y[i-1] + b * y[i+1] - y[i]
        acc += ε^2 / (a^2 + b^2 + 1)
    end
    sqrt(acc / (n - 2))
end

# --------------------------------------------------------------- fitting

"""
    fit_channel(setpoints, channel; k = 3, σ = nothing) -> NamedTuple

Smoothing-spline calibration of one field channel (`:B0` or `:B1`) against
current, fitted to grouped setpoints from [`group_setpoints`](@ref).

Each setpoint is weighted by `√n / σ`, because the mean of `n` replicates has
standard error `σ/√n` — without this the heavily-replicated anchors (typically
0 A and the top current) are under-weighted relative to the information they
carry. With those weights Dierckx's criterion is a chi-square, so the smoothing
target is `s = length(I)`: one unit of misfit per setpoint, independent of how
the replicates are distributed.

Returns `spline` (callable, tesla vs amp), `σ` and `σ_source` (`:replicates`,
`:second_differences` or `:given`), `σ_replicate` (the full replicate estimate),
`resid` and `zresid` (setpoint residuals, absolute and σ-normalised), and
`χ²_red`. `χ²_red` near 1 means the curve fits as well as the noise allows; ≫ 1
means too stiff a spline, a drifting run, or an underestimated σ.

`bc = "error"` makes evaluation outside the fitted current range throw instead of
silently returning the edge value — a flat plateau at the bottom of a log-spaced
scan is otherwise easy to mistake for real data.
"""
function fit_channel(setpoints::NamedTuple, channel::Symbol; k::Int = 3,
                     σ::Union{Nothing,Real} = nothing)
    I, B = setpoints.I, setpoints[channel]
    sd   = setpoints[Symbol(channel, :_sd)]
    length(I) > k + 1 || error("$(length(I)) setpoints is too few for a degree-$k spline")

    rep = replicate_noise(sd, setpoints.n)
    σ_used, source = if σ !== nothing
        Float64(σ), :given
    elseif isfinite(rep.σ) && rep.σ > 0
        rep.σ, :replicates
    else
        noise_sigma(I, B), :second_differences
    end

    w      = sqrt.(setpoints.n) ./ σ_used
    spline = Spline1D(I, B; w = w, k = k, s = float(length(I)), bc = "error")
    resid  = B .- spline.(I)
    (spline = spline, σ = σ_used, σ_source = source, σ_replicate = rep,
     resid = resid, zresid = resid .* w,
     χ²_red = sum(abs2, resid .* w) / max(length(I) - k - 1, 1))
end

"""
    combine_fields(runs; atol = 1e-4, kwargs...) -> NamedTuple

The single combined dataset. Returns the `pooled` cloud, the averaged
`setpoints`, fitted calibrations `B0` and `B1`, `I_range` (the interval every run
covers, where the fit is supported by all of them) and `I_span` (the full pooled
extent). `kwargs` go to [`fit_channel`](@ref).

Stay inside `I_range` for anything quantitative; `I_span` is the hard limit
beyond which the spline now throws.
"""
function combine_fields(runs::NamedTuple; atol::Real = 1e-4, kwargs...)
    pooled    = pool_runs(runs)
    setpoints = group_setpoints(pooled; atol = atol)
    lo, hi    = maximum(minimum, runs.I), minimum(maximum, runs.I)
    (pooled = pooled, setpoints = setpoints,
     B0 = fit_channel(setpoints, :B0; kwargs...),
     B1 = fit_channel(setpoints, :B1; kwargs...),
     I_range = (lo, hi), I_span = (minimum(pooled.I), maximum(pooled.I)))
end

# --------------------------------------------------------------- diagnostics

"""
    run_offsets(combined, channel; dirs) -> Nothing

Per-run mean σ-normalised residual about the shared calibration.

The drift test a common current grid would have given directly: if all runs
sample one calibration curve, each run's residuals scatter about zero and
`mean/SE` sits within roughly ±2. A run several SE away has a systematically
different `B(I)` — a re-calibration, a moved probe, a different zero — and
pooling it biases every other run. Setpoints are attributed to a run when that
run contributed to them, so shared setpoints count for each contributor.
"""
function run_offsets(combined::NamedTuple, channel::Symbol;
                     dirs::AbstractVector = 1:maximum(combined.pooled.run))
    fit, sp = combined[channel], combined.setpoints
    @printf("%-10s %6s %12s %10s\n", "run", "n_pts", "mean z", "mean/SE")
    for i in eachindex(dirs)
        rows = findall(r -> i in r, sp.runs)
        isempty(rows) && continue
        z = fit.zresid[rows]
        @printf("%-10s %6d %12.4g %10.2f\n",
                dirs[i], length(z), mean(z), mean(z) * sqrt(length(z)))
    end
    return nothing
end

"""
    noise_model(combined, channel) -> Nothing

Tabulate within-setpoint SD against field magnitude, in decade bins.

Answers whether the noise is additive (SD roughly constant) or multiplicative
(SD rising with |B|). Additive justifies the constant-σ weighting used above;
multiplicative means the high-current end dominates the fit and the low-current
end is under-resolved — the case for fitting `log B` against `log I` over the
positive-current subset instead.
"""
function noise_model(combined::NamedTuple, channel::Symbol)
    sp = combined.setpoints
    B, sd = sp[channel], sp[Symbol(channel, :_sd)]
    keep = findall(i -> sp.n[i] > 1 && isfinite(sd[i]) && B[i] != 0, eachindex(sp.n))
    isempty(keep) && return println("$channel: no replicated setpoints")
    @printf("%-12s %6s %12s %12s\n", "|B| decade", "n", "median SD", "median SD/|B|")
    for d in sort(unique(floor(Int, log10(abs(B[i]))) for i in keep))
        rows = filter(i -> floor(Int, log10(abs(B[i]))) == d, keep)
        @printf("1e%-10d %6d %12.4g %12.4g\n", d, length(rows),
                median(sd[rows]), median(sd[rows] ./ abs.(B[rows])))
    end
    return nothing
end

"""
    channel_difference(pooled) -> NamedTuple

Row-wise `B0 - B1` statistics. Both channels are recorded at the same currents
within a run, so this needs no fit and no grouping.

If the probes see the same field, `mean ≈ 0` and `std ≈ √(σ₀² + σ₁²)` — an
independent check on the replicate noise estimates. A large, stable `mean`
instead means the channels differ by a genuine offset or scale factor and must
be fitted separately, never averaged together.
"""
channel_difference(pooled::NamedTuple) =
    (mean = mean(pooled.B0 .- pooled.B1),
     std  = std(pooled.B0 .- pooled.B1),
     ratio_median = median(pooled.B0 ./ pooled.B1))

# --------------------------------------------------------------- sampling

"""
    sample_fields(combined; n = 200, scale = :log, lo = nothing, hi = nothing)

Evaluate both calibrations on one current grid: `I`, `B0`, `B1`, and the
per-point standard errors `B0_err`, `B1_err`.

`scale = :log` spaces currents geometrically (via `Base.logrange`), `:linear`
uniformly. Bounds default to `combined.I_range` and are clipped to `I_span`,
since the spline throws outside its data. A log grid needs a positive lower
bound: when the requested one is ≤ 0 — as it is whenever the sweep includes
0 A — it is raised to the smallest positive measured current and a notice is
printed. The grid then no longer reaches the zero-field end; keep `I = 0` as a
separate point if you need it.

!!! note "Correlated points"
    The spline's resolution is roughly uniform in `I`, so a log grid oversamples
    the low-current decade: points there are reading one spline segment at high
    density and are strongly correlated. `B*_err` reflects per-point noise only,
    not that correlation.
"""
function sample_fields(combined::NamedTuple; n::Int = 200, scale::Symbol = :log,
                       lo::Union{Nothing,Real} = nothing,
                       hi::Union{Nothing,Real} = nothing)
    n ≥ 2 || error("need at least 2 grid points, got $n")
    span = combined.I_span
    lo = clamp(something(lo, combined.I_range[1]), span...)
    hi = clamp(something(hi, combined.I_range[2]), span...)
    lo < hi || error("empty current range [$lo, $hi]")

    I = if scale === :linear
        collect(range(lo, hi; length = n))
    elseif scale === :log
        if lo ≤ 0
            positive = filter(>(0), combined.pooled.I)
            isempty(positive) && error("no positive currents — a log grid is impossible")
            lo = minimum(positive)
            @info "Log grid lower bound raised to the smallest positive measured current" lo
        end
        collect(logrange(lo, hi, n))      # Base.logrange, Julia ≥ 1.11
    else
        error("scale must be :linear or :log, got :$scale")
    end

    (I = I, scale = scale,
     B0 = combined.B0.spline.(I), B1 = combined.B1.spline.(I),
     B0_err = fill(combined.B0.σ, n), B1_err = fill(combined.B1.σ, n))
end

# --------------------------------------------------------------- run

runs     = read_runs()
combined = combine_fields(runs)

for ch in (:B0, :B1)
    f = combined[ch]
    @printf("%s: σ = %.4g T (%s, %d replicated setpoints, %d dof), χ²_red = %.2f\n",
            ch, f.σ, f.σ_source, f.σ_replicate.n_groups, f.σ_replicate.dof, f.χ²_red)
end
@printf("%d raw points → %d setpoints; I ∈ [%.4f, %.4f] A (all runs)\n",
        length(combined.pooled.I), length(combined.setpoints.I), combined.I_range...)

run_offsets(combined, :B1; dirs = DIR_LIST)
noise_model(combined, :B1)

scan = sample_fields(combined; n = 200, scale = :log)

fig_Is = plot(
        title = "Stern–Gerlach Currents Sampled",
        titlefontsize = 12,
        legend = :bottomright,
        xgrid=false,
        gridalpha = 0.25,
        gridstyle = :dot,
        minorgridalpha = 0.05,
        tickfontsize=11,
        guidefontsize=14,
    );
for (idx,data_directory) in enumerate(DIR_LIST)
    scatter!(fig_Is,
        idx .* ones(length(I_all[idx])), 
        I_all[idx],
        # yerror=dI_all[idx],
        label=false,
        marker = (:circle, :white, 2.5),
        markerstrokecolor = cols[idx],
        markerstrokewidth = 1.5,)
end
plot!(fig_Is,
    ylim = (1e-4,1.05),
    xlim=(-1,n_runs+2),
    yaxis = (:log10, L"$I_{0} \ (\mathrm{A})$"),
    xticks = (1:n_runs, DIR_LIST),
    yticks = ([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xminorticks = false,
    xrotation=75,
    bottom_margin=-2mm,
    left_margin = 6mm,
    right_margin = 6mm,
    size=(350,720)
)
display(fig_Is)

fig_Bfield = plot(
        title = "Stern–Gerlach Currents Sampled",
        titlefontsize = 12,
        legend = :bottomright,
        xgrid=false,
        gridalpha = 0.25,
        gridstyle = :dot,
        minorgridalpha = 0.05,
        tickfontsize=11,
        guidefontsize=14,
    );
for (idx,data_directory) in enumerate(DIR_LIST)
    scatter!(fig_Bfield,
        I_all[idx], B0_all[idx],
        label=data_directory,
        # yerror=dI_all[idx],
        # label=false,
        marker = (:circle, :white, 2.5),
        markerstrokecolor = cols[idx],
        markerstrokewidth = 1.5,)
    scatter!(fig_Bfield,
        I_all[idx], B1_all[idx],
        # yerror=dI_all[idx],
        label=false,
        marker = (:square, :white, 2.5),
        markerstrokecolor = cols[idx],
        markerstrokewidth = 1.5,)
end
display(fig_Bfield)
plot!(fig_Bfield,
    scan.I, scan.B0)
plot!(fig_Bfield,
    scan.I, scan.B1)



plot!(fig_Bfield,
    xlims = (1e-5,1.1),
    ylims = (1e-6,1),
    xscale=:log10,
    yscale=:log10,
    # yaxis = (:log10, L"$I_{0} \ (\mathrm{A})$"),
    # xticks = (1:n_runs, DIR_LIST),
    xticks = ([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xminorticks = false,
    # xrotation=75,
    bottom_margin=-2mm,
    left_margin = 6mm,
    right_margin = 6mm,
    # size=(350,720)
)
display(fig_Bfield)











JLD2_MyTools.show_exp_summary(joinpath(BASE_PATH, "EXPERIMENTS", dir, "data_processed.jld2"), dir)

d   = load(joinpath(BASE_PATH, "EXPERIMENTS", dir, "data_processed.jld2"), "meta")

d["F1ProcessedImages"]

# Quantum mechanics
data_qm_path = joinpath(BASE_PATH,"SIMULATIONS",
                "2026Q2_SETUP","CQD_T205_8M_v2",
                "cqd_8000000_up_profiles_bykey.jld2");

JLD2_MyTools.list_keys_jld_qm(data_qm_path)

σw_sim_qm =  jldopen(data_qm_path,"r") do file
    file["meta/σw"]
    file["meta/λ0"]
    ile["meta/nz"]
end

JLD2_MyTools.summarize_meta_qm_jld2(data_qm_path)


nz_fix, σ_fix, λ0_fix = (2,0.250,0.01);
data_qm_path = joinpath(@__DIR__,"simulation_data","QM_T200_8M","qm_screen_profiles_f1_table.jld2");
chosen_qm = jldopen(data_qm_path,"r") do file
    file[JLD2_MyTools.make_keypath_qm(nz_fix,σ_fix,λ0_fix)]
end
Ic_qm     = [chosen_qm[i][:Icoil] for i in eachindex(chosen_qm)][2:end];
zm_qm     = [chosen_qm[i][:z_max_smooth_spline_mm] for i in eachindex(chosen_qm)][2:end];

data_directories = [
    # "20250814", "20250820", "20250825","20250919","20251002","20251003","20251006",
    # "20251109",
    # "20260211", "20260213", 
    "20260220", "20260225", "20260226am","20260226pm","20260227", "20260303", "20260306r1", "20260306r2"
];

n_runs = length(data_directories)
I_all  = Vector{Vector{Float64}}(undef, n_runs);
dI_all = Vector{Vector{Float64}}(undef, n_runs);
cols = palette(:darkrainbow, n_runs);

for (i, dir) in enumerate(data_directories)
    d   = load(joinpath(@__DIR__, "EXPERIMENTS", dir, "data_processed.jld2"), "data");
    I_all[i]  = Vector{Float64}(d[:Currents]);
    dI_all[i] = Vector{Float64}(d[:CurrentsError]);
end

fig_Is = plot(
        title = "Coil Currents",
        legend = :bottomright,
        xgrid=false,
        gridalpha = 0.25,
        gridstyle = :dot,
        minorgridalpha = 0.05,
        tickfontsize=11,
        guidefontsize=14,
    );
for (idx,data_directory) in enumerate(data_directories)
    scatter!(fig_Is,
        idx .* ones(length(I_all[idx])), 
        I_all[idx],
        yerror=dI_all[idx],
        label=false,
        marker = (:circle, :white, 2.5),
        markerstrokecolor = cols[idx],
        markerstrokewidth = 1.5,)
end
plot!(fig_Is,
    ylim = (1e-5,1.05),
    xlim=(-1,n_runs+2),
    yaxis = (:log10, L"$I_{0} \ (\mathrm{A})$"),
    xticks = (1:n_runs, data_directories),
    yticks = ([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
    xminorticks = false,
    xrotation=75,
    bottom_margin=-2mm,
    left_margin = 6mm,
    size=(350,720)
)
display(fig_Is)
saveplot(fig_Is, "currents_sampled")

sel = [:Icoil_A, :Icoil_error_A, :F1_z_centroid_mm, :F1_z_centroid_se_mm]; 
for data_directory in data_directories
    # Data Directory
    # data_directory = "20250814" ;

    magnification_factor = mag_factor(data_directory) ;

    parent_folder = joinpath(@__DIR__, "EXPDATA_ANALYSIS",data_directory);        
    m = DataReading.collect_fw_map(parent_folder; 
                                    select=sel, 
                                    filename="fw_data.csv", 
                                    report_name="experiment_report.txt", 
                                    sort_on=:binning, 
                                    data_dir_filter=data_directory);
  
    pretty_table(hcat(collect(keys(m)),
                        [v.binning   for v in values(m)],
                        [v.smoothing for v in values(m)]); 
                title = "Analysis for $(data_directory)",
                column_labels=["Run Label","Binning","Smoothing"],
                alignment=:c,
                style = TextTableStyle(
                        first_line_column_label = crayon"yellow bold",
                        table_border  = crayon"blue bold",
                        # column_label  = crayon"yellow bold",
                ),
                # border_crayon = crayon"blue bold",
                table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
                # header_crayon = crayon"yellow bold",
                equal_data_column_widths= true,
    )

    summary_path = joinpath(@__DIR__,"EXPDATA_ANALYSIS","summary",data_directory, data_directory*"_report_summary.jld2")
    Icoils = jldopen(summary_path,"r") do mfile
            abs.(mfile["meta/Currents"])
    end

    nz_list = [1,2]
    λ0_list = [0.001, 0.005, 0.01, 0.02]
    param_grid = vec(collect(Iterators.product(nz_list, λ0_list)))
    sort!(param_grid, by = x -> (x[1], x[2]))
    N_labels = length(param_grid);
    cols_k = palette(:darkrainbow, N_labels)
    
    fig=plot(title="Experimental Data : binning & spline smoothing factor",
        titlefontsize = 12)
    i = 1
    for (nz,λ0) in param_grid
        # Check experimental data
        data_exp = jldopen(summary_path,"r") do mfile
                mfile[JLD2_MyTools.make_keypath_exp(data_directory,nz,λ0)]
        end
        
        ic = Icoils
        δic = data_exp[:ErrorCurrentsPhys]
        zf1 = data_exp[:fw_F1_peak_pos][1] / magnification_factor[1]
        δzf1 = abs.(zf1) .* sqrt.( (data_exp[:fw_F1_peak_pos][2] ./ data_exp[:fw_F1_peak_pos][1]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )

        plot!(fig,
        ic, zf1, 
        xerror = δic,
        yerror = δzf1,
        label="n=$(nz) | λ=$(λ0)", 
        color=cols_k[i],
        marker=(:circle,cols_k[i],2),
        markerstrokewidth = 1,
        markerstrokecolor=cols_k[i]
        )

        chosen_qm_i  = jldopen(data_qm_path,"r") do file
                            file[JLD2_MyTools.make_keypath_qm(nz,σ_fix,λ0)]
        end       
        Ic_qm_i      = [chosen_qm_i[i][:Icoil] for i in eachindex(chosen_qm_i)]
        zm_qm_i      = [chosen_qm_i[i][:z_max_smooth_spline_mm] for i in eachindex(chosen_qm_i)]
        if nz == 1
            qm_color = :grey28
        elseif nz == 2
            qm_color = :grey42
        elseif nz ==4
            qm_color = :grey56
        else
            qm_color = :grey70
        end
        plot!(Ic_qm_i,zm_qm_i,
            label=false,
            line=(qm_color,:dash,1.5)
        )
        i+=1
    end
    display(fig)
    plot!(fig,Ic_qm,zm_qm, label=L"QM: 
        $(n_{z},\sigma,λ_{0})=(%$(nz_fix),%$(Int(1000*σ_fix))\mathrm{\mu m},%$(λ0_fix))$", line=(:solid,:black,2), marker=(:square,:grey66,2))
    plot!(fig,
        xlabel="Current (A)",
        ylabel=L"$z_{F_{1}}$ (mm)",
        xaxis=:log10,
        yaxis=:log10,
        xticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        yticks = ([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-6}", L"10^{-5}", L"10^{-4}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
        xlims=(1e-4,1.2),
        ylims=(1e-4,5),
        size=(1050,600),
        legend=:outerright,
        legend_columns=1,
        legendfontsize=8,
        foreground_color_legend = nothing,
        left_margin=5mm,
        bottom_margin=3mm,
        legend_title = data_directory,
    )
    saveplot(fig,"bin_vs_smoothing_$(data_directory)")
    display(fig)
    println("\n")
end

println("Experiment analysis finished!\n\n")

#########################################################################################
# Choose a particular configuration for comparison purposes 
#########################################################################################

# desired values
selected_bin = nz_fix
selected_spl = λ0_fix

cols = palette(:darkrainbow, n_runs)

# ---------- common axis + style ----------
xticks_vals = 10.0 .^ (-6:-1); xticks_vals = vcat(xticks_vals, 1.0)
yticks_vals = 10.0 .^ (-6:-1); yticks_vals = vcat(yticks_vals, 1.0)
xtick_labels = [L"10^{%$k}" for k in -6:-1]; xtick_labels = vcat(xtick_labels, L"10^{0}")
ytick_labels = [L"10^{%$k}" for k in -6:-1]; ytick_labels = vcat(ytick_labels, L"10^{0}")

fig1 = plot(
    xlabel = "Current (A)",
    ylabel = L"$z_{F_{1}}$ (mm)",
    xaxis  = :log10,
    yaxis  = :log10,
    xticks = (xticks_vals, xtick_labels),
    yticks = (yticks_vals, ytick_labels),
    xlims  = (1e-3, 1.2),
    ylims  = (1e-4, 3.0),
    legend = :outerright,
    legend_title = L"$n=%$(selected_bin)$ & $\lambda_{0}=%$(selected_spl)$",
    size   = (900, 420),
    left_margin = 4mm,
    bottom_margin = 3mm,
)
for (idx,data_directory) in enumerate(data_directories)
    magnification_factor = mag_factor(data_directory) ;

    summary_path = joinpath(@__DIR__,"EXPDATA_ANALYSIS","summary",data_directory, data_directory*"_report_summary.jld2")

    Icoils = jldopen(summary_path,"r") do mfile
            mfile["meta/Currents"]
    end

    # Check experimental data
    data_exp = jldopen(summary_path,"r") do mfile
            mfile[JLD2_MyTools.make_keypath_exp(data_directory,selected_bin,selected_spl)]
    end

    ic = Icoils
    δic = data_exp[:ErrorCurrentsPhys]
    zf1 = data_exp[:fw_F1_peak_pos][1] / magnification_factor[1]
    δzf1 = zf1 .* sqrt.( (data_exp[:fw_F1_peak_pos][2] ./ data_exp[:fw_F1_peak_pos][1]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )

    # guard for log10 axes: filter out non-positive values
    ic   = ifelse.(ic .> 0, ic, missing)
    zf1  = ifelse.(zf1 .> 0, zf1, missing)
    plot!(fig1, ic, zf1;
        xerror = δic,
        yerror = δzf1,
        label = "Experiment $(data_directory)",
        marker = (:circle,cols[idx],3),
        markerstrokewidth = 1,
        markerstrokecolor = cols[idx],
        line = (:solid,cols[idx],1) # pure markers; change to :solid if you want lines
    )
    display(fig1)
end
plot!(fig1, # ---------- Alexander's data ----------
    data_JSF[:exp][:, 1],
    data_JSF[:exp][:, 2],
    label = "Alexander's data",
    line = (:dash, :green, 2),
)
plot!(fig1, Ic_qm, zm_qm, label=L"QM $(\sigma_{w}=%$(Int(1000*σ_fix))\mathrm{\mu m})$", line=(:black,2))
display(fig1)
saveplot(fig1, "bin_vs_smoothing_log")   # use explicit extension; pdf/png/svg as you like


fig2 = plot(
    xlabel = "Current (A)",
    ylabel = L"$z_{F_{1}}$ (mm)",
    xlims  = (1e-3, 1.1),
    ylims  = (1e-4, 2.0),
    legend = :outerright,
    legend_title = L"$n=%$(selected_bin)$ & $\lambda_{0}=%$(selected_spl)$",
    size   = (900, 420),
    left_margin = 4mm,
    bottom_margin = 3mm,
)
for (idx,data_directory) in enumerate(data_directories)
    magnification_factor = mag_factor(data_directory) ;

    summary_path = joinpath(@__DIR__,"EXPDATA_ANALYSIS","summary",data_directory, data_directory*"_report_summary.jld2")

    Icoils = jldopen(summary_path,"r") do mfile
            mfile["meta/Currents"]
    end

    # Check experimental data
    data_exp = jldopen(summary_path,"r") do mfile
            mfile[JLD2_MyTools.make_keypath_exp(data_directory,selected_bin,selected_spl)]
    end

    ic = Icoils
    δic = data_exp[:ErrorCurrentsPhys]
    zf1 = data_exp[:fw_F1_peak_pos][1] / magnification_factor[1]
    δzf1 = abs.(zf1) .* sqrt.( (data_exp[:fw_F1_peak_pos][2] ./ data_exp[:fw_F1_peak_pos][1]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )

    # guard for log10 axes: filter out non-positive values
    ic   = ifelse.(ic .> 0, ic, missing)
    zf1  = ifelse.(zf1 .> 0, zf1, missing)
    plot!(fig2, ic, zf1;
        xerror = δic,
        yerror = δzf1,
        label = "Experiment $(data_directory)",
        marker = (:circle,cols[idx],3),
        markerstrokewidth = 1,
        markerstrokecolor = cols[idx],
        line = (:solid,cols[idx],1) # pure markers; change to :solid if you want lines
    )
    display(fig2)
end
plot!(fig2, # ---------- Alexander's data ----------
    data_JSF[:exp][:, 1],
    data_JSF[:exp][:, 2],
    label = "Alexander's data",
    line = (:dash, :green, 2),
)
plot!(fig2, Ic_qm, zm_qm, label=L"QM $(\sigma_{w}=%$(Int(1000*σ_fix))\mathrm{\mu m})$", line=(:black,2))
display(fig2)
saveplot(fig2, "bin_vs_smoothing_lin")   # use explicit extension; pdf/png/svg as you like

println("\nComparison of differente experiments finished!\n\n")

#######################################################################################################################
######################################### AVERAGING ###################################################################
#######################################################################################################################
Ics = Vector{Vector{Float64}}(undef, n_runs);
tol_grouping = 0.05
for (i, dir) in enumerate(data_directories)
    data = load(joinpath(@__DIR__, "EXPERIMENTS", dir, "data_processed.jld2"), "data")
    Ics[i] = data[:Currents]
end
clusters = MyExperimentalAnalysis.cluster_by_tolerance(Ics; tol=tol_grouping);
for s in clusters.summary
    println("Value group ≈ $(@sprintf("%1.3f", s.mean_val)) ± $(round(s.std_val;sigdigits=1)) \t appears in datasets: ", s.datasets)
end
Ic_grouped  = round.([clusters.summary[i].mean_val for i in 1:length(clusters.summary)]; digits=3)
δIc_grouped = round.([clusters.summary[i].std_val for i in 1:length(clusters.summary)]; sigdigits=1)

magnification_factor_ith        =  [mag_factor(d)[1] for d in data_directories]
magnification_factor_error_ith  =  [mag_factor(d)[2] for d in data_directories]
"""
    average_on_grid_mc(xsets, ysets;
                       σxsets=nothing, σysets=nothing,
                       xq=:union, B=400, outside=:mask, rel_x=false,
                       rng=Random.default_rng()) -> (xq_vec, μ, σ)

Monte-Carlo average of multiple noisy curves onto a common 1D grid, propagating
uncertainties in both the x- and y-coordinates.

Each input dataset `i` is a pair `(xsets[i], ysets[i])`. For each Monte-Carlo replicate,
the function perturbs `x` and/or `y` according to the provided uncertainties, interpolates
the perturbed curve onto a shared query grid `xq_vec` using linear gridded interpolation,
and then averages across datasets at each grid point (ignoring missing values). The output
mean `μ` and standard deviation `σ` are computed pointwise across the `B` replicates.

# Arguments
- `xsets::AbstractVector{<:AbstractVector}`: Collection of x-vectors, one per dataset.
- `ysets::AbstractVector{<:AbstractVector}`: Collection of y-vectors, one per dataset.
  Must satisfy `length(xsets) == length(ysets)` and each pair must have matching lengths.

# Keyword Arguments
- `σxsets::Union{Nothing,AbstractVector}=nothing`:
  Per-dataset vectors of 1σ uncertainties for `x`. If `nothing`, `x` is not perturbed.
  Each `σxsets[i]` must match `length(xsets[i])`.
- `σysets::Union{Nothing,AbstractVector}=nothing`:
  Per-dataset vectors of 1σ uncertainties for `y`. If `nothing`, `y` is not perturbed.
  Each `σysets[i]` must match `length(ysets[i])`.
- `rel_x::Bool=false`:
  If `true`, interpret `σxsets[i]` as *relative* uncertainties so that `Δx = σx .* x`.
  If `false`, interpret `σxsets[i]` as absolute uncertainties.
- `xq::Union{Symbol,AbstractVector}=:union`:
  Query grid specification. If `:union`, uses `sort!(unique(vcat(xsets...)))`.
  Otherwise, uses `collect(xq)` as the query grid.
- `B::Integer=400`:
  Number of Monte-Carlo replicates.
- `outside::Symbol=:mask`:
  Policy for evaluating outside each dataset's x-range:
  - `:mask`  → return `NaN` outside `[minimum(x), maximum(x)]` (excluded from averages)
  - `:linear` → linear extrapolation
  - `:flat`   → constant (flat) extrapolation
- `rng::AbstractRNG=Random.default_rng()`:
  Random number generator used for the perturbations.

# Returns
- `xq_vec::Vector{Float64}`: The common query grid.
- `μ::Vector{Float64}`: Pointwise Monte-Carlo mean on `xq_vec`.
- `σ::Vector{Float64}`: Pointwise Monte-Carlo standard deviation (sample std, `corrected=true`)
  on `xq_vec`. Entries may be `NaN` where no dataset covered that grid point (under `:mask`).

# Notes
- Within each replicate, each dataset is interpolated with `Gridded(Linear())` after sorting by `x`.
- When `outside == :mask`, points with no coverage across all datasets remain `NaN` in the output.
- This routine performs an *unweighted* mean across datasets at each grid point; if you need weighting
  (e.g. by `σy`), modify the combine step accordingly.
"""
function average_on_grid_mc(xsets, ysets;
                            σxsets=nothing, σysets=nothing,
                            xq=:union, B::Int=400, outside::Symbol=:mask, rel_x::Bool=false,
                            rng = Random.default_rng())

    @assert length(xsets) == length(ysets) "xsets and ysets must have the same number of datasets"
    nset = length(xsets)
    
    @assert outside in (:mask, :linear, :flat) "outside must be :mask, :linear, or :flat"
    @assert B > 0 "B must be positive"

    # Build common grid
    xq_vec = xq === :union ? sort!(unique(vcat(map(collect, xsets)...))) : collect(xq)
    m = length(xq_vec)

    preds = Matrix{Float64}(undef, B, m)
    fill!(preds, NaN)

    # small helper: eval with chosen outside policy
    function eval_on_grid(xb, yb, xq)
        p = sortperm(xb); xb = xb[p]; yb = yb[p]
        itp = Interpolations.interpolate((xb,), yb, Gridded(Interpolations.Linear()))
        ext = outside === :linear ? Interpolations.extrapolate(itp, Line()) :
              outside === :flat   ? Interpolations.extrapolate(itp, Flat()) :
                                    Interpolations.extrapolate(itp, Throw())
        vals = similar(xq, Float64); fill!(vals, NaN)
        if outside === :mask
            mask = (xq .>= first(xb)) .& (xq .<= last(xb))
            vals[mask] = itp.(xq[mask])  # safe since on-grid
        else
            vals .= ext.(xq)
        end
        return vals
    end

    # Monte-Carlo
    for b in 1:B
        # gather each set’s curve on xq for this replicate
        curves = Vector{Vector{Float64}}(undef, nset)
        for i in 1:nset
            x = collect(xsets[i])
            y = collect(ysets[i])

            # jitter x
            if σxsets === nothing
                xb = x
            else
                σx = σxsets[i]
                dx = rel_x ? (σx .* x) : σx                      # abs σ from relative if requested
                xb = x .+ randn(rng, length(x)) .* dx
            end

            # jitter y
            if σysets === nothing
                yb = y
            else
                σy = σysets[i]
                yb = y .+ randn(rng, length(y)) .* σy
            end

            curves[i] = eval_on_grid(xb, yb, xq_vec)
        end

        # combine across sets at each xq (ignore NaNs)
        for j in 1:m
            s = 0.0; k = 0
            @inbounds for i in 1:nset
                v = curves[i][j]
                if !isnan(v); s += v; k += 1; end
            end
            preds[b, j] = k == 0 ? NaN : (s / k)
        end
    end

    # MC mean & std at each xq (ignore NaNs if some points had no coverage)
    μ  = similar(xq_vec, Float64)
    σ  = similar(xq_vec, Float64)
    for j in 1:m
        col = @view preds[:, j]
        vals = [v for v in col if !isnan(v)]
        if isempty(vals)
            μ[j] = NaN; σ[j] = NaN
        else
            μ[j] = mean(vals)
            σ[j] = std(vals; corrected=true)
        end
    end
    return xq_vec, μ, σ
end

# helper: first index where column > threshold (skips missings; falls back to 1)
@inline function first_gt_idx(df::DataFrame, col::Symbol, thr::Real)
    v = df[!, col]
    idx = findfirst(x -> !ismissing(x) && x >= thr, v)
    return idx === nothing ? 1 : idx
end

tables = Vector{DataFrame}(undef, n_runs)
for (idx,data_directory) in enumerate(data_directories)
    magnification_factor = mag_factor(data_directory) ;

    summary_path = joinpath(@__DIR__,"EXPDATA_ANALYSIS","summary",data_directory, data_directory*"_report_summary.jld2")

    Icoils = jldopen(summary_path,"r") do mfile
            mfile["meta/Currents"]
    end

    # Check experimental data
    data_exp = jldopen(summary_path,"r") do mfile
            mfile[JLD2_MyTools.make_keypath_exp(data_directory,selected_bin,selected_spl)]
    end

    ic = Icoils
    δic = data_exp[:ErrorCurrentsPhys]
    zf1 = data_exp[:fw_F1_peak_pos][1] / magnification_factor[1]
    δzf1 = abs.(zf1) .* sqrt.( (data_exp[:fw_F1_peak_pos][2] ./ data_exp[:fw_F1_peak_pos][1]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )
    zf2 = data_exp[:fw_F2_peak_pos][1] / magnification_factor[1]
    δzf2 = abs.(zf2) .* sqrt.( (data_exp[:fw_F2_peak_pos][2] ./ data_exp[:fw_F2_peak_pos][1]).^2 .+ (magnification_factor[2] ./ magnification_factor[1]).^2 )

    tables[idx] = DataFrame(hcat(ic,δic,zf1,δzf1,zf2,δzf2),[:x,:sx,:y1,:sy1,:y2,:sy2])
end

threshold = 0.000 # lower cut-off for experimental currents
CURRENT_ROW_START = [first_gt_idx(t, :x, threshold) for t in tables]

xsets  = [ t[i:end, :x]  for (t,i) in zip(tables, CURRENT_ROW_START)]
y1sets = [ t[i:end, :y1] for (t,i) in zip(tables, CURRENT_ROW_START)]
y2sets = [ t[i:end, :y2] for (t,i) in zip(tables, CURRENT_ROW_START)]
σxsets = [ t[i:end, :sx] for (t,i) in zip(tables, CURRENT_ROW_START)]
σy1sets = [ t[i:end, :sy1] for (t, i) in zip(tables, CURRENT_ROW_START)]
σy2sets = [ t[i:end, :sy2] for (t, i) in zip(tables, CURRENT_ROW_START)]

# pick a log-spaced grid across the overall x-range (nice for decades-wide currents)
i_sampled_length = 20001
xlo = maximum([minimum(first.(xsets)),1e-9])
xhi = maximum([maximum(last.(xsets)),1.000])
xq  = exp10.(range(log10(xlo), log10(xhi), length=i_sampled_length))

xi1, μ1, σ1 = average_on_grid_mc(xsets, y1sets; σxsets=σxsets, σysets=σy1sets,
                              xq=:union, B=500, outside=:mask, rel_x=true)

# xq, μ, σ_xy = average_on_grid_mc(xsets, y1sets; σxsets=σxsets, σy1sets=σy1sets)
# _,  _, σ_y  = average_on_grid_mc(xsets, y1sets; σxsets=nothing,   σy1sets=σy1sets)
# _,  _, σ_x  = average_on_grid_mc(xsets, y1sets; σxsets=σxsets,    σy1sets=nothing)
## If x and y errors are independent, typically:
# σ_quad = sqrt.(σ_x.^2 .+ σ_y.^2)  # should be close to σ_xy
# hcat(σ_xy, σ_quad )

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
    xlims = (1e-3,1.0),
    ylims = (1e-3, 2),
    legend=:bottomright,
)
for i=1:n_runs
    xs = tables[i][CURRENT_ROW_START[i]:end,:x]
    ys = tables[i][CURRENT_ROW_START[i]:end,:y1]
    scatter!(fig,xs,ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
end
plot!(fig, Ic_qm, zm_qm, label="QM", line=(:red,:dash,2))
plot!(fig, xi1, μ1; 
    ribbon=σ1,
    # yerror=σ1,
    label=false,
)
plot!(fig,
    xscale=:log10,
    yscale=:log10, 
    title = "Interpolation MC",
    color=:black,
)
display(fig)
saveplot(fig, "MC_interpolation")


# using Dierckx
# spl = Spline1D(m_sets[1][runs[1]][3][!,"Icoil_A"], m_sets[1][runs[1]][3][!,"F1_z_centroid_mm"]; k=3, s=0.5, bc="extrapolate")   # k=3 cubic; s=0 exact interpolate, s>0 smoothing

using BSplineKit
# i_sampled_length = 2*i_sampled_length
# i_xx = round.(range(threshold,1.000,length=i_sampled_length); digits=5)
i_xx0 = unique(round.(sort(union(xq,Ic_grouped)); digits=9))
i_sampled_length = length(i_xx0)

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
    xlims = (10e-3,1.0),
    ylims = (8e-3, 2),
)
z_final = zeros(n_runs,i_sampled_length)
cols = palette(:darkrainbow, n_runs);
for i=1:n_runs
    xs = tables[i][CURRENT_ROW_START[i]:end,:x]
    ys = tables[i][CURRENT_ROW_START[i]:end,:y1]
    spl = BSplineKit.extrapolate(BSplineKit.interpolate(xs,ys, BSplineKit.BSplineOrder(4),BSplineKit.Natural()),BSplineKit.Linear())
    z_final[i,:] = spl.(i_xx0)
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx0,spl.(i_xx0),
        label=false,
        line=(cols[i],1))
end
plot!(fig, Ic_qm, zm_qm, label="QM", line=(:red,:dash,2))
display(fig)
plot!(fig,
title="Interpolation: cubic splines",
xaxis=:log10, 
yaxis=:log10,
xticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([1e-3, 1e-2, 1e-1, 1.0], [ L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
)
zf1 = vec(mean(z_final, dims=1))
δzf1 = vec(std(z_final; dims=1, corrected=true)/sqrt(n_runs))
plot!(fig, i_xx0, zf1,
    ribbon = δzf1,
    fillalpha=0.40, 
    fillcolor=:gray36, 
    label="Mean",
    line=(:dash,:black,:2))
display(fig)
saveplot(fig, "interpolation")

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
)
z_final_fit = zeros(n_runs,i_sampled_length)
cols = palette(:darkrainbow, n_runs)
for i=1:n_runs
    xs = tables[i][CURRENT_ROW_START[i]:end,:x]
    ys = tables[i][CURRENT_ROW_START[i]:end,:y1]
    δys = tables[i][CURRENT_ROW_START[i]:end,:sy1]
    spl = BSplineKit.extrapolate(BSplineKit.fit(BSplineKit.BSplineOrder(4),xs,ys, 0.002, BSplineKit.Natural(); weights=1 ./ δys.^2),BSplineKit.Smooth())
    z_final_fit[i,:] = spl.(i_xx0)
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx0,spl.(i_xx0),
        label=false,
        line=(cols[i],1))
end
plot!(fig, Ic_qm, zm_qm, label="QM", line=(:red,:dash,2))
display(fig)
plot!(fig,
title = "Fit smoothing cubic spline",
xaxis=:log10, 
yaxis=:log10,
xticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlims = (10e-3,1.0),
ylims = (8e-3, 2),
)
display(fig)
zf1_fit = vec(mean(z_final_fit, dims=1))
δzf1_fit = vec(std(z_final_fit; dims=1, corrected=true)/sqrt(n_runs))
plot!(fig,i_xx0, zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:gray36, 
    label="Mean",
    line=(:dash,:black,:2))
display(fig)
saveplot(fig, "smoothing_interpolation")


z2_final_fit = zeros(n_runs,i_sampled_length)
for i=1:n_runs
    xs = tables[i][CURRENT_ROW_START[i]:end,:x]
    ys = tables[i][CURRENT_ROW_START[i]:end,:y2]
    δys = tables[i][CURRENT_ROW_START[i]:end,:sy2]
    spl = BSplineKit.extrapolate(BSplineKit.fit(BSplineKit.BSplineOrder(4),xs,ys, 0.002, BSplineKit.Natural(); weights=1 ./ δys.^2),BSplineKit.Smooth())
    z2_final_fit[i,:] = spl.(i_xx0)
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx0,spl.(i_xx0),
        label=false,
        line=(cols[i],1))
end
zf2_fit = vec(mean(z2_final_fit, dims=1))
δzf2_fit = vec(std(z2_final_fit; dims=1, corrected=true)/sqrt(n_runs))
fig_c = plot(i_xx0, zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:gray36, 
    label="Mean F1",
    line=(:dash,:black,1))
plot!(fig_c,
    i_xx0, zf2_fit,
    ribbon = δzf2_fit,
    fillalpha=0.40, 
    fillcolor=:gray36, 
    label="Mean F2",
    line=(:dash,:black,1)
)
plot!(fig_c,
    xlabel="Current (A)",
    ylabel="Centered Peak position (mm)",
    # xlims=(0,0.020)
    )
display(fig_c)
saveplot(fig_c, "fit_interpol_centroid")


Ic_around_0 = filter(v -> v <= 0.010, i_xx0)
ni_0  = length(Ic_around_0)
δi, eδi, m, b0, i0, σd0, ishift = curr_error_physical(
                i_xx0, 0.001*i_xx0,
                zf1_fit, zf2_fit;
                δz1 = δzf1_fit,
                δz2 = δzf2_fit,
                use_mismatch = false,
                nfit = ni_0, order = 2,
                weight = :gaussian, h = nothing
            );
@info "Error computed $(round(eδi,sigdigits=1))mA"
@info "Current shift $(round(1000*ishift; sigdigits=3))mA"
@info "Channel disagreement at Ic=$(i_xx0[i0])A is $(round(1000*abs.((zf1_fit[i0] + zf2_fit[i0]) / 2 );sigdigits=3))μm"
@info "Channel error measured at Ic=$(i_xx0[i0])A is $(round(1000*abs.( 0.5 * sqrt( δzf1_fit[i0]^2 + δzf2_fit[i0]^2 ) ); sigdigits=3))μm"

fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
)
z_final_fit = zeros(n_runs,i_sampled_length)
cols = palette(:darkrainbow, n_runs)
for i=1:n_runs
    xs = tables[i][CURRENT_ROW_START[i]:end,:x]
    ys = tables[i][CURRENT_ROW_START[i]:end,:y1]
    δys = tables[i][CURRENT_ROW_START[i]:end,:sy1]
    spl = BSplineKit.extrapolate(BSplineKit.fit(BSplineKit.BSplineOrder(4),xs,ys, 0.002, BSplineKit.Natural(); weights=1 ./ δys.^2),BSplineKit.Smooth())
    z_final_fit[i,:] = spl.(i_xx0)
    scatter!(fig,xs, ys,
        label=data_directories[i],
        marker=(:circle, :white,3),
        markerstrokecolor=cols[i],
        markerstrokewidth=1,
        )
    plot!(fig,i_xx0,spl.(i_xx0),
        label=false,
        line=(cols[i],1))
end
plot!(fig, Ic_qm, zm_qm, label="QM", line=(:red,:dash,2))
display(fig)
plot!(fig,
title = "Fit smoothing cubic spline",
xaxis=:log10, 
yaxis=:log10,
xticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlims = (10e-3,1.0),
ylims = (8e-3, 2),
)
idx = collect(1:200:length(i_xx0))
if idx[end] != length(i_xx0)
    push!(idx, length(i_xx0))
end
plot!(fig,i_xx0[idx], zf1_fit[idx],
    xerror = δi[idx] ./ 2,
    yerror = δzf1_fit[idx],
    marker=(:square,1,:black),
    fillalpha=0.40, 
    fillcolor=:gray36, 
    label="Mean",
    line=(:solid,:black,2))
display(fig)
saveplot(fig, "smoothing_interpolation_err")


fig = plot(
    xlabel="Current (A)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
)
plot!(fig,i_xx0, zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:green, 
    label="Average: smoothing cubic spline",
    line=(:dot,:green,:2))
plot!(fig, i_xx0, zf1,
    ribbon = δzf1,
    fillalpha=0.40, 
    fillcolor=:dodgerblue, 
    label="Average: interpolation cubic spline",
    line=(:dash,:dodgerblue,:2))
plot!(fig, xi1, μ1; 
    ribbon=σ1, 
    label="Interpolation MC",
    color=:orangered2
)
plot!(fig, Ic_qm, zm_qm, label=L"QM $(n_{z},\sigma,λ_{0})=(%$(nz_fix),%$(Int(1000*σ_fix))\mathrm{\mu m},%$(λ0_fix))$", line=(:red,:dash,2))
plot!(fig,
xaxis=:log10, 
yaxis=:log10,
xticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
yticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlims = (15e-3,1.0),
ylims = (1e-3, 2),
legend=:bottomright,
)
display(fig)
saveplot(fig, "inter_vs_mc_vs_fit")



include("./Modules/TheoreticalSimulation.jl");
fig = plot(
    xlabel="Magnetic field gradient  (T/m)",
    ylabel=L"$F_{1} : z_{\mathrm{peak}}$ (mm)",
)
plot!(fig,TheoreticalSimulation.GvsI(i_xx0), zf1_fit,
    ribbon = δzf1_fit,
    fillalpha=0.40, 
    fillcolor=:green, 
    label="Average: smoothing cubic spline",
    line=(:dot,:green,:2))
plot!(fig, TheoreticalSimulation.GvsI(i_xx0), zf1,
    ribbon = δzf1,
    fillalpha=0.40, 
    fillcolor=:dodgerblue, 
    label="Average: interpolation cubic spline",
    line=(:dash,:dodgerblue,:2))
plot!(fig, TheoreticalSimulation.GvsI(xi1), μ1; 
    ribbon=σ1, 
    label="Interpolation MC",
    color=:orangered2
)
plot!(fig, TheoreticalSimulation.GvsI.(Ic_qm), zm_qm, label=L"QM $(n_{z},\sigma,λ_{0})=(%$(nz_fix),%$(Int(1000*σ_fix))\mathrm{\mu m},%$(λ0_fix))$", line=(:red,:dash,2))
plot!(fig,
xaxis=:log10, 
yaxis=:log10,
xticks = ([ 1.0, 10, 100, 1000], [L"10^{0}", L"10^{1}", L"10^{2}", L"10^{3}"]),
yticks = ([ 1e-3, 1e-2, 1e-1, 1.0], [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
xlims = (3,400,),
ylims = (8e-3, 2),
legend=:bottomright,
)
display(fig)
saveplot(fig, "g_inter_vs_mc_vs_fit")


jldsave(joinpath(OUTDIR,"data_averaged_$(selected_bin).jld2"), 
    data=OrderedDict(
        :nz_bin         => selected_bin,
        :σw_um          => round(1000*σ_fix; sigdigits=6),
        :λ0_spl         => selected_spl,
        :dir            => data_directories,
        :mag_factor     => hcat(magnification_factor_ith ,magnification_factor_error_ith),
        :tol_grouping   => tol_grouping,
        :Ic_grouped     => hcat(Ic_grouped , δIc_grouped),
        # Interpolated data => Mean
        :i_interp       => i_xx0,
        :z_interp       => zf1,
        :δz_interp      => δzf1,
        # Fitting data => Mean
        :i_smooth       => i_xx0,
        :δi_smooth      => δi ./ 2,
        :z_smooth       => zf1_fit,
        :δz_smooth      => δzf1_fit,
        :z2_smooth      => zf2_fit,
        :δz2_smooth     => δzf2_fit,
        # MonteCarlo sampling => Mean
        :i_mc           => xi1,
        :z_mc           => μ1,
        :δz_mc          => σ1
    )
)

T_END = Dates.now()
T_RUN = Dates.canonicalize(T_END-T_START)
println("\nEXPERIMENTS ANALYSIS FINISHED! $(T_RUN)")
alert("EXPERIMENTS ANALYSIS FINISHED!")


# using Optim

# zQM_itpl = BSplineKit.extrapolate(BSplineKit.interpolate(Ic_qm, zm_qm, BSplineKit.BSplineOrder(4),BSplineKit.Natural()),BSplineKit.Linear())
# # index cutoff
# idx = 8



# # -------------------------------------------------------------
# # 1. Scaling model:   z_scaled = X/s + r
# # -------------------------------------------------------------
# scale_model(X, r, s) = @. X/s + r 

# # -------------------------------------------------------------
# # 2. Log-error function with positivity constraints
# # -------------------------------------------------------------
# function log_error(X::Vector, Y::Vector)
#     function f(x)
#         r, s = x
#         s <= 0 && return Inf

#         vals = scale_model(X, r, s)
#         any(vals .<= 0) && return Inf  # log safety

#         diff = log10.(Y) .- log10.(vals)
#         return sum(diff .^ 2)
#     end
#     return f
# end

# # -------------------------------------------------------------
# # 3. Fit (r, s) using Nelder–Mead
# # -------------------------------------------------------------
# function fit_rs(X::Vector, Y::Vector; x0=[0.0, 1.0])
#     f = log_error(X, Y)
#     res = optimize(f, x0, NelderMead())
#     return Optim.minimizer(res) 
# end

# # -------------------------------------------------------------
# # Plot (QM vs Experiment)
# # -------------------------------------------------------------
# function plotting_qm_fixed(X::Vector,Y::Vector; idx::Integer = 1, title::String = "title" , yscale::Symbol = :identity)
#     z0_fit, m_fit = fit_rs(X, Y; x0=[0.0, 1.0])

#     X_scaled = scale_model(X, z0_fit, m_fit)

#     fig1 = plot(Ic_qm, zm_qm,
#         label="Model: QM",
#         line=(:dash,:blue,2));
#     plot!(i_xx0[idx:end], X_scaled,
#         label="Experiment: Scaled ($(@sprintf("%2.2f",m_fit*mean(magnification_factor_ith))), $(@sprintf("%2.2f",1000*z0_fit))μm)",
#         line=(:solid,0.75,2,:red)
#     );
#     plot!(
#         title=title,
#         xaxis = (L"$I_{c} \ (\mathrm{A})$",
#                 (10e-3,1),
#                 ([1e-3, 1e-2, 1e-1, 1.0], 
#                     [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
#                 :log10,),
#         yaxis=(L"$z_{\mathrm{max}} \ (\mathrm{mm})$",yscale),
#         legend=:bottomright,
#     );
#     display(fig1)

#     fig2 = plot(i_xx0[idx:end], 100 .*( Y ./ X  .- 1),
#         label="Experiment : original",
#         line=(:solid,:red,2)
#     );
#     plot!(i_xx0[idx:end], 100*(Y ./ X_scaled .- 1),
#         label  = "Experiment : scaled",
#         line=(:solid,:dodgerblue4,2),
#         ylabel = "Relative Error (%)",
#         xaxis = (L"$I_{c} \ (\mathrm{A})$",
#                 (10e-3,1),
#                 ([1e-3, 1e-2, 1e-1, 1.0], 
#                     [L"10^{-3}", L"10^{-2}", L"10^{-1}", L"10^{0}"]),
#                 :log10,),
#     );
#     hline!([0], line=(:dash,:black,1), label=nothing)

#     fig = plot(fig1,fig2,
#         layout=(2,1),
#         size=(800,500),
#         left_margin=3mm,)
#     display(fig)

#     return (m_fit = m_fit, z0_fit = z0_fit, fig=fig)
# end

# # plotting_qm_fixed(zf1_fit[idx:end],zQM_itpl.(i_xx0[idx:end]); idx=idx, title="Spline fitting", yscale=:log10)
# # plotting_qm_fixed(zf1_fit[idx:end],zQM_itpl.(i_xx0[idx:end]); idx=idx, title="Spline fitting", yscale=:identity)

# # plotting_qm_fixed(zf1[idx:end],zQM_itpl.(i_xx0[idx:end]); idx=idx, title="Spline interpolation", yscale=:log10)
# # plotting_qm_fixed(zf1[idx:end],zQM_itpl.(i_xx0[idx:end]); idx=idx, title="Spline interpolation", yscale=:identity)

# # ss = load(joinpath(@__DIR__,"20250820","data_processed.jld2"))

# # ss["data"]
# # ss["data"][:Currents]
# # size(ss["data"][:F1ProcessedImages])
# # ss["data"][:F1ProcessedImages]


# jldopen(joinpath(@__DIR__,"EXPDATA_ANALYSIS","summary","20260225","20260225_report_summary.jld2"),"r") do file
#     println(file["meta/Currents"])
#     file[JLD2_MyTools.make_keypath_exp("20260225",2,0.10)]
# end

# f["meta"]
# f[JLD2_MyTools.make_keypath_exp("20260211",2,0.01)]