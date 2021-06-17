using CounterfactualFairness, Test
using Omega, Distributions

U₁ = 1 ~ Normal(27, 5)
U₂ = 1 ~ Normal(2, 4)
U₃ = 1 ~ Normal(1, 2)
X = U₁
Y(ω) = 4 * X(ω) + U₂
Z(ω) = X(ω) / 10 + U₃
g = CausalModel()
g = add_vertex(g, (:Temp, X))
g = add_vertex(g, (:IceCreamSales, Y))
g = add_vertex(g, (:Crime, Z))

@test add_edge!(g, 1 => 2)
@test add_edge!(g, 1 => 3)

@test mechanism(g) == [NamedTuple{(:name, :func)}((:Temp, X)),
        NamedTuple{(:name, :func)}((:IceCreamSales, Y)), NamedTuple{(:name, :func)}((:Crime, Z))]