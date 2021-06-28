using CounterfactualFairness, Test
using Omega, LightGraphs, ForwardDiff, Distributions

g = CausalModel()
U₁ = add_exo_variable!(g, :U₁, 1 ~ Normal(27, 5))
# U₂ = add_exo_variable!(g, :U₂, 2 ~ Normal(27, 5))
# U₃ = add_exo_variable!(g, :U₃, 3 ~ Normal(27, 5))
X = add_endo_variable!(g, :X, identity, U₁)
Ya = add_endo_variable!(g, :Ya, *, 4, X)
Yb = add_endo_variable!(g, :Yb, +, 20, Ya)
Z = add_endo_variable!(g, :Z, *,1/3, Yb)

blocked_edges = [false, false, true, true, true]
x₁ = CounterfactualFairness.Intervention(:X, 15)
x₂ = CounterfactualFairness.Intervention(:X, 20)
@test all(isapprox.(randsample(apply_ps_intervention(g, x₁, blocked_edges, x₂))[2:end], [15.0, 80.0, 100.0, 100/3], atol = 0.0001))

ctx = Context((U₁ = 25, ))
x₁ = DifferentiableIntervention(:X, 15, g, ctx)
x₂ = DifferentiableIntervention(:X, 20, g, ctx)
@test all(isapprox.(randsample(ω -> apply_ps_intervention(g, x₁, blocked_edges, x₂, ω))[2:end], [15.0, 80.0, 100.0, 100/3], atol = 0.001))

function loss(xβ)
    n = length(xβ)
    x = xβ[1:div(n, 2)]
    β = xβ[div(n, 2) + 1:end]
    intervention_ = DifferentiableIntervention(x, β, x₁.l)
    output = apply_ps_intervention(g, intervention_, blocked_edges, x₂)
    loss_ = sum(output)
    return loss_
end
@test typeof(ForwardDiff.gradient(loss, vcat(x₁.x, x₁.β))) == Array{Float64,1}