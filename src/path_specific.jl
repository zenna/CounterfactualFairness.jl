using Omega, Random
using LightGraphs:SimpleEdge

import Omega:intervene

export apply_ps_intervention, BlockedEdges, PS_Intervention

"Vector of all blocked edges"
struct BlockedEdges{T}
    edges::Vector{Edge{T}}
end

"""
    `(B::BlockedEdges)(g::CausalModel)`

 Returns boolean vector: `true` if the vertex must take the alternate value and 
 `false` if the variable must take the value of the intervention
"""
function (B::BlockedEdges)(g::CausalModel)
    blocked_vertices = zeros(Bool, nv(g))
    for e in B.edges
        blocked_vertices[dst(e)] = true
    end
    for i in 1:nv(g)
        parents = inneighbors(g, i)
        if any(map(p -> getindex(blocked_vertices, p), parents))
            blocked_vertices[i] = true
        end
    end
    return blocked_vertices
end

"""
 Path-Specific intervention for value change of a variable
 from `x₁` to `x₂` along causal path (with `be` edges blocked)
"""
struct PS_Intervention{I <: Interventions} <: Interventions
    be::BlockedEdges
    x₁::I
    x₂::I
end

"""
    `apply_ps_intervention(model::CausalModel, x₁::Intervention, blocked_edges::BlockedEdges, x₂::Intervention, ω::AbstractΩ)`

 Apply the model to a path-specific intervention when the 
 value of a variable is changed from x₁ to x₂ (x₁ is the refernce value, x₂ is the required intervention).
 ## Inputs -
 - model : Causal model
 - x₁ : The refernce intervention
 - blocked_edges : Edges that are blocked specified as the type `BlockedEdges`
 - x₂ : The intervention that must be applied to the causal model
 - ω

 ## Returns -
 Vector of values the variables in the model take after the applying the path-specific intervention
"""
function apply_ps_intervention(model::CausalModel, x₁::Intervention, blocked_edges::BlockedEdges, x₂::Intervention, ω::AbstractΩ)
    intervened = apply_intervention(model, x₁)(ω)
    blocked_vertices = blocked_edges(model)
    for n in 1:nv(model)
        if !(blocked_vertices[n])
            if x₂.X == variable(model, n).name
                intervened[n] = x₂.x
            else
                parents = inneighbors(model, n)
                v = variable(model, n).func
                if !(length(parents) == 0)
                    if length(v) == 1
                        if isa(v[1](intervened[parents]...), Member)
                            intervened[n] = v[1](intervened[parents]...)(ω)
                        else
                            intervened[n] = v[1](intervened[parents]...)
                        end
                    else
                        intervened[n] = v[1](v[2], intervened[parents]...)
                    end
                end
            end
        end
    end
    return intervened
end

apply_ps_intervention(model::CausalModel, x₁::Intervention, blocked_edges::BlockedEdges, x₂::Intervention) = 
    ω ->  apply_ps_intervention(model, x₁, blocked_edges, x₂, ω)

function apply_ps_intervention(model::CausalModel, x₁::DifferentiableIntervention, blocked_edges::BlockedEdges, x₂::DifferentiableIntervention, ω::AbstractΩ = defω())
    int_x₁ = apply_intervention(model, x₁)
    blocked_vertices = blocked_edges(model)
    intervened = Float64[]
    for n in 1:nv(model)
        parents = inneighbors(model, n)
        v = variable(model, n).func
        if !(length(parents) == 0)
            if length(v) == 1
                if isa(v[1](intervened[parents]...), Member)
                    elem = v[1](intervened[parents]...)(ω)
                else
                    elem = v[1](intervened[parents]...)
                end
            else
                elem = v[1](v[2], intervened[parents]...)
            end
            elem = x₂.β[n]*elem .+ (1 .- x₂.β[n])*x₂.x[n]
        else
            elem = x₂.x[n]
        end
        intervened = [intervened; (1 - blocked_vertices[n])*elem .+ blocked_vertices[n]*int_x₁[n]]
    end
    return intervened
end

"Apply the model to a path specific intervention given"
apply_ps_intervention(model::CausalModel, psint::PS_Intervention) = 
            apply_ps_intervention(model, psint.x₁, psint.be, psint.x₂)