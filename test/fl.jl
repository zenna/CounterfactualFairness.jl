using CounterfactualFairness, Test
using Omega, Flux, Distributions
using DataFrames, CSV

# df = CSV.read("law_data.csv", DataFrame)
decision_model = Dense(1, 1)
params = Flux.params(decision_model)

model = CausalModel()
R = add_exo_variable!(model, :R, 1 ~ Bernoulli(0.75))
S = add_exo_variable!(model, :S, 2 ~ Bernoulli(0.5))
K = add_exo_variable!(model, :K, 3 ~ Normal(0, 1))

R1 = add_endo_variable!(model, :R1, *, 4.0, R)
S1 = add_endo_variable!(model, :S1, *, 1.5, S)
R2 = add_endo_variable!(model, :R2, *, 6.0, R)
S2 = add_endo_variable!(model, :S3, *, 0.5, S)
R3 = add_endo_variable!(model, :R3, *, 6.0, R)
S3 = add_endo_variable!(model, :S3, *,  0.5, S)

GPA = add_endo_variable!(model, :GPA, +, K, R1, S1)
# GPA = add_endo_variable!(model, :GPA, θ -> 4 ~ Normal(θ, 0.1), GPA1)
LSAT = add_endo_variable!(model, :LSAT, +, K, R2, S2)
# LSAT = add_endo_variable!(model, :LSAT, θ -> 5 ~ Normal(θ, 0.1), LSAT1)
# FYA = add_endo_variable!(model, :FYA, decision_model, K)

loss(x, y) = Flux.Losses.mae(decision_model(x), y)
opt = Flux.Descent(0.01)

# FairLearning!(df, model, :FYA, (:R, :S), (:K, ), loss, opt, params, defω()) #Too many allocations