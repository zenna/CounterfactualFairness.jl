using CounterfactualFairness, Test
using Omega, Flux, Distributions
using DataFrames, CSV, ProgressMeter
using UnicodePlots

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
toy = @load_synthetic

pred = Chain(Dense(4, 3), Dense(3, 2), Dense(2, 1))
adv = Chain(Dense(5, 3), Dense(3, 2, relu))
opt_pred = Flux.Optimise.ADAM(0.001)
opt_adv = Flux.Optimise.ADAM()
λ = 0.1

n = 500
l(x, y) = Flux.Losses.logitbinarycrossentropy(x, y)
X = (CausalVar(toy, :X1), CausalVar(toy, :X2), CausalVar(toy, :X3), CausalVar(toy, :X4))
U = (CausalVar(toy, :U₁), CausalVar(toy, :U₂), CausalVar(toy, :U₃), CausalVar(toy, :U₄), CausalVar(toy, :U₅))
A = CausalVar(toy, :A)
Y = CausalVar(toy, :Y)
df = DataFrame(X1 = randsample(ω -> X[1](ω), n), X2 = randsample(ω -> X[2](ω), n), X3 = randsample(ω -> X[3](ω), n), X4 = randsample(ω -> X[4](ω), n), A = randsample(ω -> A(ω), n), Y = randsample(ω -> Y(ω), n))
df_x, df_y, df_a = Vector.(eachrow(df[!, Not([:A, :Y])])), df[!, :Y], df[!, :A]
train = Flux.Data.DataLoader((df_x, df_y, df_a))

losses = Float64[]
p = Progress(n, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:green)
for i in 1:10
    sleep(0.1)
    push!(losses, train!(train, toy, A, U, X, pred, adv, defω(), λ, l, opt_pred, opt_adv)...)
    ProgressMeter.next!(p)
end