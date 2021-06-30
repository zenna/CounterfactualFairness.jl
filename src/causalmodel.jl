using LightGraphs, NamedTupleTools, Omega

import LightGraphs:
    AbstractGraph, nv, ne,
    edges, add_edge!, rem_edge!,
    has_edge, inneighbors, outneighbors,
    neighbors, all_neighbors,
    indegree, outdegree, add_vertex!, is_directed

import LightGraphs.SimpleGraphs: SimpleDiGraph, fadj, badj

import Base.eltype

export CausalModel, variable, variables,
    add_endo_variable!, add_exo_variable!, CausalVar

export nv, ne, edges, rem_edge!, 
    has_edge, inneighbors, outneighbors,
    neighbors, all_neighbors,
    indegree, outdegree, eltype

# Endogenous or exogenous variable of structural causal model 
# Corresponds to structural equations of the form :name = func(...),
# where in endogenous variables func is applied to parent variables
# and in exogenous variables func is applied to context/ω
const Variable = @NamedTuple{name::Symbol, func}

"Causal model with `dag` representing the model and `scm` holds the name of the variables and the SCM"
struct CausalModel{T} <: AbstractGraph{T}
    dag::SimpleDiGraph{T}  # Representing the DAG of the causal model
    scm::Dict{T,Variable}  # Name and function (or distribution) of each variable
end

CausalModel() = CausalModel(SimpleDiGraph(), Dict{Int64, Variable}())

eltype(g::CausalModel) = Base.eltype(g.dag)
nv(g::CausalModel) = nv(g.dag)
ne(g::CausalModel) = ne(g.dag)

edges(g::CausalModel) = edges(g.dag)
has_edge(g::CausalModel, x...) = has_edge(g.dag, x...)
inneighbors(g::CausalModel, v::Integer) = inneighbors(g.dag, v)
outneighbors(g::CausalModel, v::Integer) = outneighbors(g.dag, v)
neighbors(g::CausalModel, v::Integer) = outneighbors(g, v)
all_neighbors(g::CausalModel, v::Integer) = vcat(inneighbors(g, v), outneighbors(g, v))

function add_vertex!(g::CausalModel, prop)
    t = add_vertex!(g.dag)
    g.scm[nv(g)] = Variable(prop)
    return t
end

add_edge!(g::CausalModel, x...) = add_edge!(g.dag, x...)
rem_edge!(g::CausalModel, x...) = rem_edge!(g.dag, x...)

indegree(g::CausalModel) = indegree(g.dag)
outdegree(g::CausalModel) = outdegree(g.dag)

"""
Defines the variable `varname` in the model
"""
struct CausalVar
    model::CausalModel
    varname::Symbol
end

function (v::CausalVar)(ω::AbstractΩ)
    # @show topological_order = topological_sort_by_dfs(v.model.dag)
    cm = []
    for i in 1:nv(v.model)
        # parents = map(p -> getindex(topological_order, p), inneighbors(v.model, i))
        parents = inneighbors(v.model, i)
        isexo = Bool(length(parents) == 0)
        func = variable(v.model, i).func
        if !(isexo)
            p = [cm[parent] for parent in parents]
            if length(func) != 1
                cm = vcat(cm, func[1](func[2], p...))
            else
                cm = vcat(cm, func[1](p...))
            end  
        else
            cm = vcat(cm, func(ω))
        end
        if variable(v.model, i).name == v.varname
            return cm[i]
        end
    end
end

function vertex(c::CausalVar)
    for n in 1:nv(c.model)
        if variable(c.model, n)[:name] == c.varname
            return n
        end
    end
    throw(error("The vertex does not exist"))
end

function (g::CausalModel)(ω::AbstractΩ)
    # order = topological_sort_by_dfs(g.dag)
    cm = Float64[]
    for i in 1:nv(g)
        # parents = map(p -> getindex(order, p), inneighbors(g, i))
        parents = inneighbors(g, i)
        isexo = Bool(length(parents) == 0)
        func = variable(g, i).func
        if !(isexo)
            p = [cm[parent] for parent in parents]
            if length(func) != 1
                cm = vcat(cm, func[1](func[2], p...))
            else
                cm = vcat(cm, func[1](p...))
            end  
        else
            cm = vcat(cm, func(ω))
        end
    end
    return cm
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
    `variable(g, v)`
 Returns the `Variable` of the vertex `v` in the causal model `g`
"""
variable(g::CausalModel, v::Integer) = g.scm[v]

"""
    `variables(g)`
 Returns all the `Variable`s in the model `g`
"""
variables(g::CausalModel) = [variable(g, v) for v in 1:nv(g)]