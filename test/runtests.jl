using CounterfactualFairness, Test

@testset "Causal Model" begin
    include("causalmodel.jl")
end

@testset "Interventions" begin
    include("interventions.jl")
end

@testset "Counterfactuals" begin
    include("counterfactuals.jl")
end

@testset "Path Specific Effects" begin
    include("ps.jl")
end

@testset "Gaussian Causal Graph" begin
    include("graph.jl")
end

@testset "FairLearning" begin
    include("fl.jl")
end