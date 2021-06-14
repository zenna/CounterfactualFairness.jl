using CounterfactualFairness, Test
using Omega

U₁ = normal(27, 5)
U₂ = normal(2, 4)
U₃ = normal(1, 2)
X = U₁
Y = 4 * X + U₂
Z = X / 10 + U₃
g = CausalModel()
g = add_vertex(g, (:Temp, X))
g = add_vertex(g, (:IceCreamSales, Y))
g = add_vertex(g, (:Crime, Z))

@test add_edge!(g, 1 => 2)
@test add_edge!(g, 1 => 3)

@test mechanism(g) == [NamedTuple{(:name, :func)}((:Temp, X)),
        NamedTuple{(:name, :func)}((:IceCreamSales, Y)), NamedTuple{(:name, :func)}((:Crime, Z))]