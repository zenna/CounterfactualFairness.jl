using Test, CounterfactualFairness
using Omega, Distributions, LightGraphs

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

m = CausalModel()
A = add_exo_variable!(m, :A, 1 ~ Normal(0, 5))
B = add_endo_variable!(m, :B, identity, A)
C = add_endo_variable!(m, :C, identity, A)
D = add_endo_variable!(m, :D, +, A, B, C)
blocked_edges = BlockedEdges([Edge(1 => 2)])
x₁ = CounterfactualFairness.Intervention(:A, 1)
x₂ = CounterfactualFairness.Intervention(:A, 0)
psint = PS_Intervention(blocked_edges, x₁, x₂)
@test isapprox(counterfactual(:D, (B = 1, C = 0), x₁, m, defω()), 3.0, atol = 0.1)