using Test, CounterfactualFairness
using Omega, Distributions

g = CausalModel()
U₁ = add_exo_variable!(g, :U₁, 1 ~ Normal(27, 5))
U₂ = add_exo_variable!(g, :U₂, 2 ~ Normal(27, 5))
U₃ = add_exo_variable!(g, :U₃, 3 ~ Normal(27, 5))
X = add_endo_variable!(g, :X, identity, U₁)
Ya = add_endo_variable!(g, :Ya, *, 4, X)
Yb = add_endo_variable!(g, :Yb, +, Ya, U₂)
Z = add_endo_variable!(g, :Z, *, 1/3, Yb)

c = ω -> counterfactual(:Z, (Ya = 60.0, Yb = 85.0), CounterfactualFairness.Intervention(:X, 15.0), g, ω)
@test isapprox(randsample(ω -> c(ω)), 28.33, atol = 0.1) # 85/3 = 28.33
@test isNonDesc(g, (:U₁, :U₃, :X, :Ya), (:U₂,)) == true
@test isNonDesc(g, (:U₁, :U₃, :X, :Ya, :Yb, :Z), (:U₂,)) == false


blocked_edges = BlockedEdges([Edge(3 => 4)])
x₁ = CounterfactualFairness.Intervention(:X, 15)
x₂ = CounterfactualFairness.Intervention(:X, 20)
psint = PS_Intervention(blocked_edges, x₁, x₂)
@test isapprox(counterfactual(:Z, (Ya = 60.0, Yb = 85.0), psint, g, defω()), 33.33, atol = 0.1)