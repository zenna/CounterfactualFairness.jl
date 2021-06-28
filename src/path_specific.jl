using Omega, Random
import Omega:intervene, Interventions

export apply_ps_intervention, BlockedEdges

function apply_ps_intervention(model::CausalModel, x₁::Intervention, blocked_edges::AbstractVector{Bool}, x₂::Intervention, ω::AbstractΩ)
    int_model_x₁ = apply_intervention(model, x₁)(ω)
    int_model_x₂ = apply_intervention(model, x₂)(ω)
    for e in 1:length(blocked_edges)
        if blocked_edges[e]
            int_model_x₁[e] = int_model_x₂[e]
        end
    end
    return int_model_x₁
end

apply_ps_intervention(model::CausalModel, x₁::Intervention, blocked_edges::AbstractVector{Bool}, x₂::Intervention) = 
    ω ->  apply_ps_intervention(model, x₁, blocked_edges, x₂, ω)

function apply_ps_intervention(model::CausalModel, x₁::DifferentiableIntervention, blocked_edges::AbstractVector{Bool}, x₂::DifferentiableIntervention, ω::AbstractΩ)
    int_model_x₁ = apply_intervention(model, x₁, ω)
    int_model_x₂ = apply_intervention(model, x₂, ω)
    ps_int = []
    for e in 1:length(blocked_edges)
        a = (1 - blocked_edges[e])*int_model_x₁[e] + blocked_edges[e]*int_model_x₂[e]
        ps_int = vcat(ps_int, a)
    end
    return ps_int
end

apply_ps_intervention(model::CausalModel, x₁::DifferentiableIntervention, blocked_edges::AbstractVector{Bool}, x₂::DifferentiableIntervention) = 
    apply_ps_intervention(model, x₁, blocked_edges, x₂, defω())