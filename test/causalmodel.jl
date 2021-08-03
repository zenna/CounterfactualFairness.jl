using CounterfactualFairness, Test
using Omega, Distributions

g = CausalModel()
U₁ = add_exo_variable!(g, :U₁, 1 ~ Normal(27, 5))
U₂ = add_exo_variable!(g, :U₂, 2 ~ Normal(27, 5))
U₃ = add_exo_variable!(g, :U₃, 3 ~ Normal(27, 5))
X = add_endo_variable!(g, :X, identity, U₁)
Ya = add_endo_variable!(g, :Ya, *, 4, X)
Yb = add_endo_variable!(g, :Yb, +, Ya, U₂)
Z = add_endo_variable!(g, :Z, /, X, U₃)

@test typeof(X) == CausalVar{Int64}
c = Context((U₁ = 1.23, U₂ = 15., U₃ = 1.451))
@inferred Vector{Float64} apply_context(g, c, return_type = Vector) # or return_type = Array
@inferred NamedTuple apply_context(g, c) # Default return type is NamedTuple
output = apply_context(g, (U₁ = 1.23, U₂ = 15., U₃ = 1.451))

ω = defω()
@inferred Vector{Float64} g(ω)
output2 = g(ω, return_type = NamedTuple) # Default value of return_type is Vector
@test typeof(output) == typeof(output2)

@test typeof(X(ω)) == Float64
@test Ya(ω) == output2.Ya