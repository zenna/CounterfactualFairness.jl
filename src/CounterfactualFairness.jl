module CounterfactualFairness

using OmegaCore
include("causalmodel.jl")
include("interventions.jl")
include("path_specific.jl")
include("counterfactuals.jl")
include("prob_causal_graph.jl")
include("FairLearning.jl")
include("examples.jl")
include("verification.jl")

# Include the files in the folder MLJ
include("MLJ/include_mlj.jl")
end