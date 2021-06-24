using CounterfactualFairness, Test
using DataFrames, CSV

df = CSV.read("student-mat.csv", DataFrame)
# df = df[!, [3, 4, 5, 7]]
# df[!, 1] = Float64.(df[!, 1])
m = prob_causal_graph(df, p = 0.025)