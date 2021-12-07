using CounterfactualFairness, Test
using Distributions, Omega, DataFrames
using Flux

m = @load_law_school
n = 2
test = DataFrame(Sex = randsample(ω -> CausalVar(m, :Sex)(ω), n), 
    Race = randsample(ω -> CausalVar(m, :Race)(ω), n),
    GPA = randsample(ω -> CausalVar(m, :GPA)(ω), n),
    LSAT = randsample(ω -> CausalVar(m, :LSAT)(ω), n),
    FYA = randsample(ω -> CausalVar(m, :FYA)(ω), n)
)
predictor = Dense(4, 1, σ)
Y = :FYA
A = :Sex
a′ = 0
cf_effect(test, predictor, Y, A, a′, m)