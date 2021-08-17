using MLJ, Flux, CounterfactualFairness, Omega
using DataFrames

toy = @load_synthetic
pred = Chain(Dense(4, 3), Dense(3, 2), Dense(2, 1))
adv = Chain(Dense(5, 3), Dense(3, 2, relu))

n = 500
l(x, y) = Flux.Losses.logitbinarycrossentropy(x, y)
X = (CausalVar(toy, :X1), CausalVar(toy, :X2), CausalVar(toy, :X3), CausalVar(toy, :X4))
U = (CausalVar(toy, :U₁), CausalVar(toy, :U₂), CausalVar(toy, :U₃), CausalVar(toy, :U₄), CausalVar(toy, :U₅))
A = CausalVar(toy, :A)
Y = CausalVar(toy, :Y)

df = DataFrame(
            X1 = randsample(ω -> X[1](ω), n),
            X2 = randsample(ω -> X[2](ω), n), 
            X3 = randsample(ω -> X[3](ω), n), 
            X4 = randsample(ω -> X[4](ω), n),
            A = randsample(ω -> A(ω), n),
            Y = randsample(ω -> Y(ω), n)
        )

model = AdversarialWrapper(cm = toy, grp = :A, latent = [:U₁, :U₂, :U₃, :U₄, :U₅], 
                    observed = [:X1, :X2, :X3, :X4], predictor = pred, adversary = adv, loss = l, iters = 1)

mach = machine(model, df[!, Not(:Y)], df[!, :Y])
fit!(mach)
predict(mach, df[1:3, Not(:Y)])