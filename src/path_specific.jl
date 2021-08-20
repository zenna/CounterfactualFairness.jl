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
            intervened[n] = intervene(CausalVar(model, variable(model, n).name), x₂)(ω)
        end
    end
    return intervened
end

apply_ps_intervention(model::CausalModel, x₁::Intervention, blocked_edges::BlockedEdges, x₂::Intervention) = 
    ω ->  apply_ps_intervention(model, x₁, blocked_edges, x₂, ω)

function apply_ps_intervention(model::CausalModel, x₁::DifferentiableIntervention, blocked_edges::BlockedEdges, x₂::DifferentiableIntervention)
    blocked_vertices = blocked_edges(model)
    int_model_x₁ = apply_intervention(model, x₁)
    int_model_x₂ = apply_intervention(model, x₂)
    ps_int = []
    for e in 1:length(blocked_vertices)
        a = (1 - blocked_vertices[e]) * int_model_x₁[e] + blocked_vertices[e] * int_model_x₂[e]
        ps_int = vcat(ps_int, a)
    end
    return ps_int
end

"Apply the model to a path specific intervention given"
apply_ps_intervention(model::CausalModel, psint::PS_Intervention) = 
            apply_ps_intervention(model, psint.x₁, psint.be, psint.x₂)