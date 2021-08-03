using CounterfactualFairness, Test
using Omega, Flux, DataFrames

ω = defω()

toy = @load_synthetic
X = (CausalVar(toy, :X1), CausalVar(toy, :X2), CausalVar(toy, :X3), CausalVar(toy, :X4))
U = (CausalVar(toy, :U₁), CausalVar(toy, :U₂), CausalVar(toy, :U₃), CausalVar(toy, :U₄), CausalVar(toy, :U₅))
A = CausalVar(toy, :A)
Y = CausalVar(toy, :Y)

pred = Chain(Dense(4, 3), Dense(3, 2), Dense(2, 1))
adv = Chain(Dense(5, 3), Dense(3, 2, relu))
l(x, y) = Flux.Losses.logitbinarycrossentropy(x, y)

x, y, a = Vector([randsample(ω -> k(ω)) for k in X]), randsample(ω -> Y(ω)), randsample(ω -> A(ω))

@inferred cond_latent_var(U, X .=> x, A, a, ω)
u = cond_latent_var(U, X .=> x, A, a, ω)
@test length(u) == 5
@inferred P(u, ω, adv)
@inferred ℒ(toy, X, x, y, A, a, U, adv, pred, l, ω)
@inferred Tuple gradient(x -> ℒ(toy, X, x, y, A, a, U, adv, pred, l, ω), ones(4))