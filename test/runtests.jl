using CounterfactualFairness, Test, OmegaCore

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

@testset "FairLearning" begin
    include("train.jl")
end

@testset "MLJ AdversarialWrapper" begin
    include("MLJ/adv.jl")
end