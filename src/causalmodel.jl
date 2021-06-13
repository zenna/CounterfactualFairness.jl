using LightGraphs, NamedTupleTools

import LightGraphs:
    AbstractGraph, nv, ne,
    edges, add_edge!, rem_edge!,
    has_edge, inneighbors, outneighbors,
    neighbors, all_neighbors,
    indegree, outdegree

import LightGraphs.SimpleGraphs: SimpleDiGraph, fadj, badj

import Base.eltype

export CausalModel, add_vertex, mechanism

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

add_edge!(g::CausalModel, x...) = add_edge!(g.dag, x...)
rem_edge!(g::CausalModel, x...) = rem_edge!(g.dag, x...)

indegree(g::CausalModel) = indegree(g.dag)
outdegree(g::CausalModel) = outdegree(g.dag)

"""
    mechanism(g, v)
 Returns the distribution of the vertex `v` in the causal model `g`

    mechanism(g)
 Returns the causal mechanism of the model `g`
"""
mechanism(g::CausalModel, v::Integer) = g.scm[v]
mechanism(g::CausalModel) = [mechanism(g, v) for v in 1:nv(g)]