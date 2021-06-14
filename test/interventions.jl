using CounterfactualFairness, Test
using Omega, ForwardDiff

U₁ = normal(5, 10)
U₂ = normal(1000, 200)
U₃ = normal(1, 2)
X = U₁
Y = X / 3 + U₂
Z = X / 16 + U₃
g = CausalModel()
g = add_vertex(g, (:Fund, X))
g = add_vertex(g, (:SAT, Y))
g = add_vertex(g, (:ColApp, Z))

add_edge!(g, 1 => 2)
add_edge!(g, 2 => 3)

context = Context([:Fund], [3])
i = DifferentiableIntervention(:SAT, 1560, g, context)
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

i = Intervention(:SAT, 1560)
m = apply_intervention(g, i)
@test typeof(m) == CausalModel{Int64}