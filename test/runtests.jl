using CounterfactualFairness, Test

@testset "Causal Model" begin
    include("causalmodel.jl")
end

@testset "Interventions" begin
    include("interventions.jl")
end