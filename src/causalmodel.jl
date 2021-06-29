using LightGraphs, NamedTupleTools, Omega

import LightGraphs:
    AbstractGraph, nv, ne,
    edges, add_edge!, rem_edge!,
    has_edge, inneighbors, outneighbors,
    neighbors, all_neighbors,
    indegree, outdegree, add_vertex!

import LightGraphs.SimpleGraphs: SimpleDiGraph, fadj, badj

import Base.eltype

export CausalModel, add_vertex, add_vertex!, mechanism, 
    add_endo_variable!, add_exo_variable!, CausalVar

export nv, ne, edges, add_edge!, rem_edge!, 
    has_edge, inneighbors, outneighbors,
    neighbors, all_neighbors,
    indegree, outdegree, eltype

# :name -> variable_name and :func -> function or distribution
const Variable = @NamedTuple{name::Symbol, func}

"Causal model with `dag` representing the model and `scm` holds the name of the variables and the SCM"
struct CausalModel{T} <: AbstractGraph{T}
    dag::SimpleDiGraph{T}  # Representing the DAG of the causal model
    scm::Dict{T,Variable}  # Name and function (or distribution) of each variable
end

function CausalModel(x)
    T = Base.eltype(x)
    g = SimpleDiGraph(x)
    scm = Dict{T,Variable}()
    return CausalModel(g, scm)
end

CausalModel() = CausalModel(SimpleDiGraph())

eltype(g::CausalModel) = Base.eltype(g.dag)
nv(g::CausalModel) = nv(g.dag)
ne(g::CausalModel) = ne(g.dag)

edges(g::CausalModel) = edges(g.dag)
has_edge(g::CausalModel, x...) = has_edge(g.dag, x...)
inneighbors(g::CausalModel, v::Integer) = inneighbors(g.dag, v)
outneighbors(g::CausalModel, v::Integer) = outneighbors(g.dag, v)
neighbors(g::CausalModel, v::Integer) = outneighbors(g, v)
all_neighbors(g::CausalModel, v::Integer) = vcat(inneighbors(g, v), outneighbors(g, v))

function add_vertex(g::CausalModel, prop)
    m_fadj = vcat(fadj(g.dag), [Array{Int64,1}(undef, 0)])
    m_badj = vcat(badj(g.dag), [Array{Int64,1}(undef, 0)])
    m = SimpleDiGraph(ne(g), m_fadj, m_badj)
    scm = g.scm
    scm[nv(g) + 1] = Variable(prop)
    return CausalModel(m, scm)
end

function add_vertex!(g::CausalModel, prop)
    t = add_vertex!(g.dag)
    g.scm[nv(g)] = Variable(prop)
    return t
end

add_edge!(g::CausalModel, x...) = add_edge!(g.dag, x...)
rem_edge!(g::CausalModel, x...) = rem_edge!(g.dag, x...)

indegree(g::CausalModel) = indegree(g.dag)
outdegree(g::CausalModel) = outdegree(g.dag)

struct CausalVar
    model::CausalModel
    varname::Symbol
end

function (v::CausalVar)(ω::AbstractΩ)
    for i in 1:nv(v.model)
        if mechanism(v.model, i).name == v.varname
            parents = inneighbors(v.model, i)
            isexo = Bool(length(parents) == 0)
            func = mechanism(v.model, i)[:func]
            if !(isexo)
                p = [CausalVar(v.model, mechanism(v.model, parent).name)(ω) for parent in parents]
                if length(func) != 1
                    return func[1](func[2], p...)
                else
                    return func[1](p...)
                end  
            else
                return func(ω)
            end
        end
    end
end

function vertex(c::CausalVar)
    for n in 1:nv(c.model)
        if mechanism(c.model, n)[:name] == c.varname
            return n
        end
    end
    print("The vertex does not exist")
    throw(error())
end

function add_exo_variable!(m::CausalModel, name::Symbol, dist)
    add_vertex!(m, (name, dist))
    return CausalVar(m, name)
end

function add_endo_variable!(m::CausalModel, name::Symbol, func, parents::CausalVar...)
    # var = ω -> func(map(parents, ω)...)
    add_vertex!(m, (name, (func,)))
    for p in parents
        add_edge!(m, vertex(p) => nv(m))
    end
    return CausalVar(m, name)
end

function add_endo_variable!(m::CausalModel, name::Symbol, func, num::Number, parents::CausalVar...)
    # var = ω -> func(num..., map(parents, ω)...)
    add_vertex!(m, (name, (func, num)))
    for p in parents
        add_edge!(m, vertex(p) => nv(m))
    end
    return CausalVar(m, name)
end

"""
    mechanism(g, v)
 Returns the distribution of the vertex `v` in the causal model `g`

    mechanism(g)
 Returns the causal mechanism of the model `g`
"""
mechanism(g::CausalModel, v::Integer) = g.scm[v]
mechanism(g::CausalModel) = [mechanism(g, v) for v in 1:nv(g)]