using CounterfactualFairness, Test, Distributions
using DataFrames, CSV

R = Float64.(rand(Bernoulli(0.75), 1000))
S = Float64.(rand(Bernoulli(0.5), 1000))
GPA = 4 * R .+ 1.5 * S .+ 80 * randn(1000)
LSAT = 5 * R .+ 0.5 * S .+ 23 * randn(1000)
FYA = 3 * R .+ 2 * S .+ 15 * randn(1000)

df = DataFrame(R=R, S=S, GPA=GPA, LSAT=LSAT, FYA=FYA)
# df = CSV.read("law_data.csv", DataFrame)
# df = df[!, [3, 4, 5, 7]]
# df[!, 1] = Float64.(df[!, 1])
# m = prob_causal_graph(df, p=0.025)