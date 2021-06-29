using CounterfactualFairness, Test
using Omega, Distributions

# U₁ = 1 ~ Normal(27, 5)
# U₂ = 1 ~ Normal(2, 4)
# U₃ = 1 ~ Normal(1, 2)
# X = U₁
# Y(ω) = 4 * X(ω) + U₂(ω)
# Z(ω) = X(ω) / 10 + U₃(ω)
# g = CausalModel()
# g = add_vertex(g, (:Temp, X))
# g = add_vertex(g, (:IceCreamSales, Y))
# g = add_vertex(g, (:Crime, Z))

# @test add_edge!(g, 1 => 2)
# @test add_edge!(g, 1 => 3)

# @test mechanism(g) == [NamedTuple{(:name, :func)}((:Temp, X)),
#         NamedTuple{(:name, :func)}((:IceCreamSales, Y)), NamedTuple{(:name, :func)}((:Crime, Z))]


g = CausalModel()
U₁ = add_exo_variable!(g, :U₁, 1 ~ Normal(27, 5))
U₂ = add_exo_variable!(g, :U₂, 2 ~ Normal(27, 5))
U₃ = add_exo_variable!(g, :U₃, 3 ~ Normal(27, 5))
X = add_endo_variable!(g, :X, identity, U₁)
Ya = add_endo_variable!(g, :Ya, *, 4, X)
Yb = add_endo_variable!(g, :Yb, +, Ya, U₂)
Z = add_endo_variable!(g, :Z, /, X, U₃)

@test typeof(X) == CausalVar
output = apply_context(g, (U₁ = 1.23, U₂ = 15, U₃ = 1.451))

ω = defω()
output2 = g(ω)
@test typeof(output) == typeof(output2)

@test typeof(X(ω)) == Float64
@test X(ω) == output2[4]

Z_intervene = intervene(Z, X => 12.30)
Z_intervene(ω)
@test Z_intervene(ω) != Z(ω)