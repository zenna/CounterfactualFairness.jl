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
        blocked_vertices[src(e)] = true
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

function apply_ps_intervention(model::CausalModel, x₁::Intervention, blocked_edges::BlockedEdges, x₂::Intervention, ω::AbstractΩ)
    blocked_vertices = blocked_edges(model)
    int_model_x₁::Vector{Float64} = apply_intervention(model, x₁)(ω)
    int_model_x₂::Vector{Float64} = apply_intervention(model, x₂)(ω)
    for e in 1:length(blocked_vertices)
        if blocked_vertices[e]
            int_model_x₁[e] = int_model_x₂[e]
        end
    end
    return int_model_x₁
end

apply_ps_intervention(model::CausalModel, x₁::Intervention, blocked_edges::BlockedEdges, x₂::Intervention) = 
    ω ->  apply_ps_intervention(model, x₁, blocked_edges, x₂, ω)

function apply_ps_intervention(model::CausalModel, x₁::DifferentiableIntervention, blocked_edges::BlockedEdges, x₂::DifferentiableIntervention)
    blocked_vertices = blocked_edges(model)
    int_model_x₁ = apply_intervention(model, x₁)
    int_model_x₂ = apply_intervention(model, x₂)
    ps_int = []
    length(blocked_vertices)
    for e in 1:length(blocked_vertices)
        a = (1 - blocked_vertices[e]) * int_model_x₁[e] + blocked_vertices[e] * int_model_x₂[e]
        ps_int = vcat(ps_int, a)
    end
    return ps_int
end

"Apply the model to a path specific intervention given"
apply_ps_intervention(model::CausalModel, psint::PS_Intervention) = 
            apply_ps_intervention(model, psint.x₁, psint.be, psint.x₂)