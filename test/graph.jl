using CounterfactualFairness, Test, Distributions
using DataFrames, CSV

df = CSV.read("adult_binary.csv", DataFrame)
df[!, 1] = Float64.(df[!, :])
adult = prob_causal_graph(df)

@test typeof(adult) == CausalModel{Int64}