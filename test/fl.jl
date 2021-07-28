using CounterfactualFairness, Test
using Omega, Flux, Distributions
using DataFrames, CSV
using Distances, Base.Threads
using Bijectors

# df = CSV.read("law_data.csv", DataFrame)
# df = df[!, [2, 3, 4, 5, 7]]
# df[!, :race] = [(df[i, :race] == "White") ? 1 : 2 for i in 1:length(df[!, :race])]

# decision_model = Dense(1, 1)
# params = Flux.params(decision_model)

# model = CausalModel()
# R = add_exo_variable!(model, :race, 1 ~ Bernoulli(0.75))
# S = add_exo_variable!(model, :sex, 2 ~ Bernoulli(0.5))
# K = add_exo_variable!(model, :K, 3 ~ Normal(0, 1))

# R1 = add_endo_variable!(model, :R1, *, 4.0, R)
# S1 = add_endo_variable!(model, :S1, *, 1.5, S)
# R2 = add_endo_variable!(model, :R2, *, 6.0, R)
# S2 = add_endo_variable!(model, :S3, *, 0.5, S)
# R3 = add_endo_variable!(model, :R3, *, 6.0, R)
# S3 = add_endo_variable!(model, :S3, *,  0.5, S)

# GPA = add_endo_variable!(model, :UGPA, +, K, R1, S1)
# # GPA = add_endo_variable!(model, :GPA, θ -> 4 ~ Normal(θ, 0.1), GPA1)
# LSAT = add_endo_variable!(model, :LSAT, +, K, R2, S2)
# # LSAT = add_endo_variable!(model, :LSAT, θ -> 5 ~ Normal(θ, 0.1), LSAT1)
# # FYA = add_endo_variable!(model, :FYA, decision_model, K)

# loss(x, y) = Flux.Losses.mae(decision_model(x), y)
# opt = Flux.Descent(0.01)

# FairLearning!(df, model, :ZFYA, (:race, :sex), (:K, ), loss, opt, params, defω())

############################################################################

cm = CausalModel()
U₁ = add_exo_variable!(cm, :U₁, 1 ~ Normal(0, 1))
U₂ = add_exo_variable!(cm, :U₂, 2 ~ Normal(0.5, 2))
U₃ = add_exo_variable!(cm, :U₃, 3 ~ Normal(1, sqrt(2)))
U₄ = add_exo_variable!(cm, :U₄, 4 ~ Normal(1.5, sqrt(3)))
U₅ = add_exo_variable!(cm, :U₅, 5 ~ Normal(2, sqrt(2)))
A = add_exo_variable!(cm, :A, 6 ~ Normal(45, sqrt(5)))

A1 = add_endo_variable!(cm, :A1, *, 0.1, A)
A2 = add_endo_variable!(cm, :A2, *, 5, A)
A3 = add_endo_variable!(cm, :A3, *, 7, A)
U₂1 = add_endo_variable!(cm, :U₂1, x -> x^2, U₂)
U₃1 = add_endo_variable!(cm, :U₃1, *, 0.1, U₃)
X1_ = add_endo_variable!(cm, :X1_, +, 7, A1, U₁, U₂, U₃)
X2_ = add_endo_variable!(cm, :X2_, +, 80, A, U₂1)
X3_ = add_endo_variable!(cm, :X3_, +, 200, A2, U₃1)
X4_ = add_endo_variable!(cm, :X4_, +, 1000, A2, U₄, U₅)

X1 = add_endo_variable!(cm, :X1, θ -> 7 ~ Normal(θ, 1), X1_) # 16
X2 = add_endo_variable!(cm, :X2, θ -> 8 ~ Normal(θ, sqrt(10)), X2_)
X3 = add_endo_variable!(cm, :X3, θ -> 9 ~ Normal(θ, sqrt(20)), X3_)
X4 = add_endo_variable!(cm, :X4, θ -> 10 ~ Normal(θ, sqrt(1000)), X4_)
# X = add_endo_variable!(cm, :X, vcat, X1, X2, X3, X4)

Y1 = add_endo_variable!(cm, :Y1, +, U₁, U₂, U₃, U₄, U₅)
Y2 = add_endo_variable!(cm, :Y2, *, 20, Y1)
Y3 = add_endo_variable!(cm, :Y3, +, Y2, A3)
Y4 = add_endo_variable!(cm, :Y4, *, 2, Y3)
Y = add_endo_variable!(cm, :Y, θ -> 11 ~ Normal(θ, sqrt(0.1)), Y4)

pred = Chain(Dense(4, 3), Dense(3, 2), Dense(2, 1))
adv = Chain(Dense(5, 3), Dense(3, 2, relu))
ps_pred = Flux.params(pred)
ps_adv = Flux.params(adv)
opt = Flux.Optimise.ADAM()
λ = 0.1

l(x, y) = Flux.Losses.logitbinarycrossentropy(x, y)
c = (X1, X2, X3, X4)
U = (U₁, U₂, U₃, U₄, U₅)
obs = DataFrame(X1 = randsample(ω -> X1(ω), 100), X2 = randsample(ω -> X2(ω), 100), X3 = randsample(ω -> X3(ω), 100), X4 = randsample(ω -> X4(ω), 100), A = randsample(ω -> A(ω), 100), Y = randsample(ω -> Y(ω), 100))
train!(obs, cm, A, U, c, pred, adv, defω(), λ, l, opt)