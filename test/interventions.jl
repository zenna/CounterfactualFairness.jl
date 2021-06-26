using CounterfactualFairness, Test
using Omega, ForwardDiff, Distributions

g = CausalModel()
U₁ = add_exo_variable!(g, :U₁, 1 ~ Normal(27, 5))
U₂ = add_exo_variable!(g, :U₂, 2 ~ Normal(27, 5))
U₃ = add_exo_variable!(g, :U₃, 3 ~ Normal(27, 5))
X = add_endo_variable!(g, :X, identity, U₁)
Ya = add_endo_variable!(g, :Ya, *, 4, X)
Yb = add_endo_variable!(g, :Yb, +, Ya, U₂)
Z = add_endo_variable!(g, :Z, /, X, U₃)

context = Context((U₁ = 1.23, U₂ = 15, U₃ = 1.451))
output = apply_context(g, context)
i = DifferentiableIntervention(:X, 15, g, context)
apply_intervention(g, i)

function loss(xβ)
    n = length(xβ)
    x = xβ[1:div(n, 2)]
    β = xβ[div(n, 2) + 1:end]
    intervention_ = DifferentiableIntervention(x, β, i.l)
    output = apply_intervention(g, intervention_)
    loss_ = sum(output)
    return loss_
end
@test typeof(ForwardDiff.gradient(loss, vcat(i.x, i.β))) == Array{Float64,1}

i = CounterfactualFairness.Intervention(:X, 15)
m = apply_intervention(g, i)
@test typeof(m) == CausalModel{Int64}
@test indegree(m, 4) == 0
m(ω)
ω = defω()
@test CausalVar(m, mechanism(m, 4).name)(ω) != CausalVar(g, mechanism(m, 4).name)(ω)
@test CausalVar(m, mechanism(m, 7).name)(ω) != CausalVar(g, mechanism(m, 7).name)(ω)