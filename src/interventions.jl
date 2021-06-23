using Omega, Random
import Omega:intervene

export Context, Intervention, DifferentiableIntervention, 
    apply_context, apply_intervention, intervene

"Maps every exogenous variable to its value"
struct Context{N <: Real}
    c::Dict{Symbol,N}
end

function Context(s::AbstractVector{Symbol}, v::AbstractVector{T} where T <: Real)
   return Context(Dict(zip(s, v)))
end

Context(c::NamedTuple) = Context(convert(Dict, c))

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
    `DifferentiableIntervention(X, x, model::CausalModel, context::Context)`

# Inputs
- `X`: name of variable
- `x`: value to fix `X` to
- `model`: Causal Model in which to apply intervention
- `context`
# Returns - struct DifferentiableIntervention with 
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

Returns vector of values the variables in the model take when applied to the context.
"""
function apply_context(model::CausalModel, context::Context)
    m = Float64[]
    for key in keys(context.c)
        push!(m, context.c[key])
    end
    for n in 1:nv(model)
        parents = inneighbors(model, n)
        isexo = Bool(length(parents) == 0)
        if !(isexo)
            p = [m[parent] for parent in parents] # p contains values of parent variables
            v = mechanism(model, n)[:func]
            if length(v) != 1
                push!(m, v[1](v[2], p...))
            else
                push!(m, v[1](p...))
            end
        end
    end
    return m
end

apply_context(m::CausalModel, c::NamedTuple) = apply_context(m, Context(c))

function (g::CausalModel)(ω::AbstractΩ)
    cm = Float64[]
    for i in 1:nv(g)
        push!(cm, CausalVar(g, mechanism(g, i)[:name])(ω))
    end
    return cm
end

"""
    `apply_intervention(model::CausalModel, i::DifferentiableIntervention)`

# Returns - (vector of) values of each variable in `model` after applying the intervention.

    `apply_intervention(model, i::Intervention)`

Returns the interevened causal model `modelᵢ`
"""
function apply_intervention(model::CausalModel, i::DifferentiableIntervention, ω::AbstractΩ)
    m = []
    for n in 1:nv(model)
        parents::Array{Int64,1} = inneighbors(model, n)
        p = [m[p] for p in parents]
        isexo::Bool = (length(p) == 0)
        elts = !(n == 1) ? i.x[(i.l[n - 1] + 1):i.l[n]] : i.x[1:i.l[1]] # value of the nth variable
        if isexo
            value = elts
        else
            v = CausalVar(model, mechanism(model, n).name)
            for parent in parents
                p_ = CausalVar(model, mechanism(model, parent).name)
                v = intervene(v, p_ => m[parent])
            end
            value = randsample(ω -> v(ω))
        end
        new_val = value .* i.β[i.l[n]] .+ elts .* (1 - i.β[i.l[n]]) # applying the intervention to the variable
        m = vcat(m, new_val)
    end
    return m
end

apply_intervention(model::CausalModel, i::DifferentiableIntervention) = 
    apply_intervention(model, i, defω())

function apply_intervention(model, intervention::Intervention)
    m = deepcopy(model)
    int_X = identity # Initializing for if an endogenous variable is before intervened variable
    for n in 1:nv(model)
        if intervention.X == mechanism(model, n).name
            m.scm[n] = Variable((intervention.X, ω -> intervention.x))
            for p in inneighbors(m, n)
                rem_edge!(m, p, n)
            end
            return m
        end
    end
end

<<<<<<< HEAD
struct CausalVar
    model::CausalModel
    varname::Symbol
end

function intervene(x, p::Pair{CausalVar, Y}) where Y
    var = identity
    for i in 1:nv(p.first.model)
        if mechanism(p.first.model, i).name == p.first.varname
            var = mechanism(p.first.model, i).func
            break
        end
=======
function intervene(v::CausalVar, intervention::Intervention)
    if v.varname == intervention.X
        return ω -> intervention.x
>>>>>>> arw-week1
    end
    m = v.model
    for i in 1:nv(m)
        if mechanism(m, i).name == v.varname
            parents = inneighbors(m, i)
            isexo = Bool(length(parents) == 0)
            func = mechanism(m, i)[:func]
            if !(isexo)
                p_int = ω -> [intervene(CausalVar(m, mechanism(m, parent).name), intervention)(ω) for parent in parents]
                if length(func) != 1
                    return ω -> func[1](func[2], p_int(ω)...)
                else
                    return ω -> func[1](p_int(ω)...)
                end  
            else
                return func
            end
        end
    end

end

function intervene(v::CausalVar, p::Pair{CausalVar, Y}) where Y

    return intervene(v, Intervention(p.first.varname, p.second))
end