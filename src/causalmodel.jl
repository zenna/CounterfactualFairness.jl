using LightGraphs, NamedTupleTools, Omega, Distributions

import LightGraphs:
    AbstractGraph, nv, ne,
    edges, add_edge!, rem_edge!,
    has_edge, inneighbors, outneighbors,
    neighbors, all_neighbors,
    indegree, outdegree, add_vertex!, is_directed

import LightGraphs.SimpleGraphs: SimpleDiGraph, fadj, badj

import Base.eltype

export CausalModel, variable, variables, dag,
    add_endo_variable!, add_exo_variable!, CausalVar

export nv, ne, edges, rem_edge!, 
    has_edge, inneighbors, outneighbors,
    neighbors, all_neighbors,
    indegree, outdegree, eltype

# Endogenous or exogenous variable of structural causal model 
# Corresponds to structural equations of the form :name = func(...),
# where in endogenous variables func is applied to parent variables
# and in exogenous variables func is applied to context/ω
const Variable = @NamedTuple{name::Symbol, func::Union{Tuple{Core.Function}, Tuple{Core.Function, <:Real}, Member{<:Sampleable, Int64}}}

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
Defines the variable `varname` in the causal model as a functor of ω.
"""
struct CausalVar{T}
    model::CausalModel{T}
    varname::Symbol
end

function (v::CausalVar)(ω::AbstractΩ)
    # @show topological_order = topological_sort_by_dfs(v.model.dag)
    cm = Float64[]
    for i in 1:nv(v.model)
        # parents = map(p -> getindex(topological_order, p), inneighbors(v.model, i))
        parents = inneighbors(v.model, i)
        func = variable(v.model, i).func
        isexo = Bool(length(parents) == 0) # To check if the variable is exogenous (or endogenous)
        if !isexo
            if length(func) != 1
                cm = [cm; (func[1]::Function)(func[2], cm[parents]...)]
            else
                if isa(func[1](cm[parents]...), Member)
                    cm = [cm; func[1](cm[parents]...)(ω)]
                else
                    cm = [cm; func[1](cm[parents]...)]
                end
            end
        else
            if isa(func, Member)
                cm = [cm; func(ω)]
            else
                cm = [cm; func[2]]
            end
        end
        if variable(v.model, i).name == v.varname
            return cm[i]
        end
    end
    error("The variable does not exist in the model")
end

function vertex(c::CausalVar)
    for n in 1:nv(c.model)
        if variable(c.model, n).name == c.varname
            return n
        end
    end
    error("The vertex does not exist")
end

function (g::CausalModel)(ω::AbstractΩ; return_type = Vector)
    # order = topological_sort_by_dfs(g.dag)
    cm = Float64[]
    names = Symbol[]
    for i in 1:nv(g)
        # parents = map(p -> getindex(order, p), inneighbors(g, i))
        parents = inneighbors(g, i)
        isexo = Bool(length(parents) == 0)
        func = variable(g, i).func
        if !(isexo)
            if length(func) != 1
                cm = [cm; (func[1]::Function)(func[2], cm[parents]...)]
            else
                if isa(func[1](cm[parents]...), Member)
                    cm = [cm; func[1](cm[parents]...)(ω)]
                else
                    cm = [cm; (func[1]::Function)(cm[parents]...)]
                end
            end  
        else
            if isa(func, Member)
                cm = [cm; func(ω)]
            else
                cm::Vector{Float64} = [cm; func[2]]
            end
        end
        names::Vector{Symbol} = [names; variable(g, i).name]
    end
    if return_type == NamedTuple
        return namedtuple(names, cm)
    else
        return cm
    end
end

"""
    `add_exo_variable!(m::CausalModel, name::Symbol, dist)`

To add an exogenous variable to a causal model.

### Inputs -
- The causal model where the exogenous variable must be added
- The name of the exogenous variable (as type Symbol)
- The distribution of the variable as random variable in `Omega`

### Returns -
- The `CausalVar` of the variable
"""
function add_exo_variable!(m::CausalModel, name::Symbol, dist)
    add_vertex!(m, (name, dist))
    return CausalVar(m, name)
end

"""
    `add_endo_variable!(m::CausalModel, name::Symbol, func, parents...)`

To add an endogenous variable to a causal model.

### Inputs -
- The causal model where the endogenous variable must be added
- The name of the endogenous variable (as type Symbol)
- Real numbers or ω (if present) must be entered first,
 followed by the `CausalVar` of parent variables 

### Returns -
- The `CausalVar` of the variable
"""
function add_endo_variable!(m::CausalModel, name::Symbol, func, parents::CausalVar...)
    # var = ω -> func(map(parents, ω)...)
    add_vertex!(m, (name, (func,)))
    for p in parents
        add_edge!(m, vertex(p) => nv(m))
    end
    return CausalVar(m, name)
end

function add_endo_variable!(m::CausalModel, name::Symbol, func, num::Union{Number, AbstractΩ}, parents::CausalVar...)
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

"Returns the DAG of the causal model"
dag(g::CausalModel) = g.dag