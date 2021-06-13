using CounterfactualFairness, Test
using Omega

U₁ = normal(27, 5)
U₂ = normal(2, 4)
U₃ = normal(1, 2)
X = U₁
Y = 4 * X + U₂
Z = X / 10 + U₃
g = CausalModel()
g = add_vertex(g, (:Temp, U₁))
g = add_vertex(g, (:IceCreamSales, U₂))
g = add_vertex(g, (:Crime, U₃))

@test add_edge!(g, 1 => 2)
@test add_edge!(g, 1 => 3)

@test mechanism(g) == [NamedTuple{(:name, :func)}((:Temp, U₁)),
        NamedTuple{(:name, :func)}((:IceCreamSales, U₂)), NamedTuple{(:name, :func)}((:Crime, U₃))]