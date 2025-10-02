# make_module_doc.jl
using Markdown, Dates

# 1) Load your code (adjust the path if needed)
include(joinpath(@__DIR__, "TheoreticalSimulation.jl"))
using .TheoreticalSimulation  # module name as in your file

# 2) Collect the public API (exported names)
exports = sort(names(TheoreticalSimulation, all=false))

# 3) Dump a Markdown file with signatures + docstrings
function dump_module_docs(mod::Module, syms::Vector{Symbol}; out=joinpath(@__DIR__,"TheoreticalSimulation_API.md"))
    open(out, "w") do f
        println(f, "# TheoreticalSimulation — API\n")
        println(f, "_Generated: ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"), "_\n")

        for s in syms
            # skip internal compiler gensyms etc.
            isdefined(mod, s) || continue
            obj = getfield(mod, s)

            # Header with a simple kind tag
            kind = obj isa Function ? "function" :
                   obj isa DataType ? "type" : "binding"
            println(f, "## `", s, "` (", kind, ")\n")

            # List methods if it's a function
            if obj isa Function
                for m in methods(obj)
                    println(f, "- `", m, "`")
                end
                println(f)
            end

            # Docstring (if present)
            docs = Base.Docs.doc(obj)
            if docs === nothing
                println(f, "_No docstring._\n")
            else
                println(f, sprint(Markdown.plain, docs), "\n")
            end
        end
    end
    return abspath(out)
end

md = dump_module_docs(TheoreticalSimulation, exports)
println("Wrote ", md)