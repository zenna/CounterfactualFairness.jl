using Test, CounterfactualFairness
using Omega, Distributions

x1 = 1 ~ Normal(0, 1)
x2 = 2 ~ Normal(0, 1)
x3 = 3 ~ Normal(0, 1)
a1 = 4 ~ Normal(0, 1)
a2 = 5 ~ Normal(0, 1)
Y′ = (:x1, :x2, :x3)
A = (:a1, :a2)
m = CausalModel()
m = add_vertex(m, (:x1, x1))
m = add_vertex(m, (:x2, x2))
m = add_vertex(m, (:x3, x3))
m = add_vertex(m, (:a1, a1))
m = add_vertex(m, (:a2, a2))
add_edge!(m, 4 => 1)
add_edge!(m, 2 => 3)

# :x1 is a descendant of :a1, so Y′ may or may not be counterfactually fair
@test isNonDesc(m, Y′, A) == false
# Once we remove :x1 from Y′, Y′ passes the sufficiency test 
@test isNonDesc(m, (:x2, :x3), A) == true