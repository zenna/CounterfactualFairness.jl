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

blocked_edges = BlockedEdges([Edge(3 => 4)])
x₁ = CounterfactualFairness.Intervention(:X, 15)
x₂ = CounterfactualFairness.Intervention(:X, 20)
# X must be 20.0 after applying the path specific intervention. Ya takes the value accordingly.
# The blocked edge is the one from Ya to Yb, 
# so the values of Yb and Z will be as though the value of X were 15.0.
@test all(isapprox.(randsample(apply_ps_intervention(g, x₁, blocked_edges, x₂))[2:end], [20.0, 20.0*4, 15.0*4 + 20.0, 80/3], atol = 0.0001))
psint = PS_Intervention(blocked_edges, x₁, x₂)
p = apply_ps_intervention(g, psint)
@test randsample(p)[2:end] == randsample(apply_ps_intervention(g, x₁, blocked_edges, x₂))[2:end]
@inferred Vector{Float64} p(defω())

ctx = Context((U₁ = 25, ))
x₁ = DifferentiableIntervention(:X, 15, g, ctx)
x₂ = DifferentiableIntervention(:X, 20, g, ctx)
psint = PS_Intervention(blocked_edges, x₁, x₂)
p = apply_ps_intervention(g, psint)
@test p == apply_ps_intervention(g, x₁, blocked_edges, x₂)
@test all(isapprox.(randsample(ω -> apply_ps_intervention(g, x₁, blocked_edges, x₂, ω)), [25.0, 20.0, 80.0, 80.0, 80/3], atol = 0.001))

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