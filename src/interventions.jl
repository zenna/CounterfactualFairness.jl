using Omega, Random

export Context, Intervention, DifferentiableIntervention, apply_context, apply_intervention

"Maps every exogenous variable to its value"
struct Context{N <: Real}
    c::Dict{Symbol,N}
end

function Context(s::AbstractVector{Symbol}, v::AbstractVector{T} where T <: Real)
   return Context(Dict(zip(s, v)))
end

eltype(ctx::Context) = eltype(values(ctx.c))

"""
Intervention on a variable in the model,
    which makes `apply_intervention` differentiable
-`x` : Values of all variables in the model
-`β` : Denotes if each value in `x` is affected (1) or not (0)
-`l` : Cumulative Lengths of each variable in the model
"""
struct DifferentiableIntervention{T1 <: Real,T2 <: Real}
    x::Vector{T1}
    β::Vector{T2}
    l::Vector{Int64}
end

DifferentiableIntervention(X, x, model::CausalModel, context::Context) = 
    DifferentiableIntervention(X, [x], model, context)

"""
`Intervention(X, x, model::CausalModel, context::Context)`
# Inputs
- `X`: name of variable
- `x`: value to fix `X` to
- `model`: Causal Model in which to apply intervention
- `context`
# Returns - struct Intervention with 
-`x` denoting values of variables in `model` after applying model to `context`,
-`β` denoting 0 (not affected by the intervention) 
    or 1 (affected by the intervention) for each value in `x`,
-`l` denoting cummulative lengths of each variable in the model.  
"""
function DifferentiableIntervention(X, x::AbstractArray, model::CausalModel, context::Context)
    m = apply_context(model, context)
    l = Int64[]
    var = Float64[]  # Array of all values including intervention
    for n in 1:length(m)
        l = vcat(l, length(m[n]))
        var::Array{Float64,1} = (mechanism(model, n).name == X) ? vcat(var, x) : vcat(var, m[n])
    end
    b = Int64[]   # the mask
    int = 0   # to identify the vertex that is being intervened
    for n in 1:nv(model)
        p = inneighbors(model, n)
        val = Bool[b[l] for l in p]
        # s is true is any of the parent vetices have the value 1 in the mask
        s::Bool = (length(val) == 0) ? false : |(val...)
        if (mechanism(model, n).name == X)
            b = vcat(b, 1)
            int = n
        elseif (s == 1)
            b = vcat(b, 1)
        else
            b = vcat(b, 0)
        end
    end
    b[int] = 0   # setting the value 0 in the position of intervention in the mask (since it needn't be changed)
        b1 = Float64[]
    for i in 1:length(l)
    b1 = Bool(b[i]) ? vcat(b1, ones(l[i])) : vcat(b1, zeros(l[i]))
    end
    for i in 2:length(l)
        l[i] = l[i] + l[i - 1]
    end  # l now is array of cummulative lengths
    return DifferentiableIntervention(var, b1, l)
end

eltype(i::DifferentiableIntervention) = eltype(i.x)

struct Intervention{T <: Real}
    X::Symbol
    x::T
end

eltype(i::Intervention) = eltype(i.x)

"""
`apply_context(model::CausalModel, context::Context)`
Returns - a vector `v` where each value is a vector that the respected variable takes after applying model to `context`.
"""
function apply_context(model::CausalModel, context::Context, rng::AbstractRNG)
    m = []
    for key in keys(context.c)
        push!(m, context.c[key]) # Assuming the values in Context are all arrays
    end
    for n in 1:nv(model)
        parents = inneighbors(model, n)
        p = [m[parent] for parent in parents] # p contains values of parent variables
        isexo = (length(p) == 0)
        if !(isexo)
            v = mechanism(model, n)[:func]
            for p in parents
                v = replace(v, mechanism(model, p)[:func] => m[p])
    end
            v = rand(rng, v)
            push!(m, v)
        end
    end
    return m
end

apply_context(model::CausalModel, context::Context) = 
    apply_context(model, context, Random.GLOBAL_RNG)

"""
'apply_intervention(model::CausalModel, i::Intervention)'
Returns - (vector of) values of each variable in `model` after applying the intervention.
"""
function apply_intervention(model::CausalModel, i::DifferentiableIntervention, rng::AbstractRNG)
    m = []
    for n in 1:nv(model)
        parents::Array{Int64,1} = inneighbors(model, n)
        p = [m[p] for p in parents]
        isexo::Bool = (length(p) == 0)
        elts = !(n == 1) ? i.x[(i.l[n - 1] + 1):i.l[n]] : i.x[1:i.l[1]] # value of the nth variable
        if isexo
            value = elts
        else
            v = mechanism(model, n).func
            for p in parents
                v = replace(v, mechanism(model, p).func => m[p])
            end
            value = rand(rng, v)
        end
        new_val = value .* i.β[i.l[n]] .+ elts .* (1 - i.β[i.l[n]]) # applying the intervention to the variable
        m = vcat(m, new_val)
    end
    return m
end

apply_intervention(model::CausalModel, i::DifferentiableIntervention) = 
    apply_intervention(model, i, Random.GLOBAL_RNG)

function apply_intervention(model::CausalModel, i::Intervention)
    m = CausalModel()
    for n in 1:nv(model)
        v = mechanism(model, n).func
        if mechanism(model, n)[:name] == i.X
            m = add_vertex(m, (i.X, replace(v, v => i.x)))
            continue
        end
        parents::Array{Int64,1} = inneighbors(model, n)
        if length(parents) == 0
            m = add_vertex(m, (mechanism(model, n)[:name], v))
            continue
        end
        for p in parents
            v = replace(v, mechanism(model, p).func => mechanism(model, p).func)
        end
        m = add_vertex(m, (mechanism(model, n)[:name], v))
    end
    c = collect(edges(model))
    for e in 1:length(c)
        add_edge!(m, c[e])
    end
    for n in 1:nv(model)
        if mechanism(model, n)[:name] == i.X
            for p in inneighbors(m, n)
                rem_edge!(m, p, n)
            end
            return m
        end
    end
end