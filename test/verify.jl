using CounterfactualFairness, Test
using Distributions, Omega, DataFrames

m = @load_law_school
test = DataFrame(Sex = randsample(ω -> CausalVar(m, :Sex)(ω), 1000), 
    Race = randsample(ω -> CausalVar(m, :Race)(ω), 1000),
    GPA = randsample(ω -> CausalVar(m, :GPA)(ω), 1000),
    LSAT = randsample(ω -> CausalVar(m, :LSAT)(ω), 1000))
Y = CausalVar(m, :FYA)
X = (:GPA, :LSAT, :Race, )
A = :Sex
a′ = 0
m.scm[15] = (name = :FYA, func = (identity, ))
rem_edge!(m, 14, 15)
add_edge!(m, 8, 15)
(K::CausalVar)(x) = 2.3*x[2]  # Should be changed, (CV::CausalVar)(x) must be the predictor
(K::CausalVar)(x...) = (K::CausalVar)(x)
verify(test, Y, X, A, a′)