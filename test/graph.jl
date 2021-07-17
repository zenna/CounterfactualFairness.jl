using CounterfactualFairness, Test, Distributions
using DataFrames, CSV

path = joinpath(pwd(), "data", "adult_binary.csv")
df = CSV.read(path, DataFrame)
df[!, :] = Float64.(df[!, :])
adult = prob_causal_graph(df)

@test typeof(adult) == CausalModel{Int64}