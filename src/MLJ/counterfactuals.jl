using MLJBase, Omega, CausalInference
const MMI = MLJBase.MLJModelInterface

"""
    CounterfactualWrapper

"""
struct CounterfactualWrapper{I<:Interventions} <: Unsupervised
    test::Function # Conditional independence tests for PC algorithm
    p::Float64 # Parameter for PCA
    cf::Symbol # The group whose counterfactual is to be found
    interventions::I # Intervention or PS_Intervention
end

function CounterfactualWrapper(; test = gausscitest, p = 0.1, cf::Symbol = :class, interventions::Interventions = nothing)
    model =  CounterfactualWrapper(test, p, cf, interventions)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.clean!(model::CounterfactualWrapper)
    warning = ""
	model.test ∈ [typeof(gausscitest), typeof(cmitest)] || (warning *= "Conditional independence tests for PC algorithm must be either `gausscitest` or `cmitest`\n")
    # PCA requires the dataframe to have float values
	# Constraint on p (?)
    return warning
end

# fitresult is of type CausalModel
function MMI.fit(model::CounterfactualWrapper, verbosity::Int, X)
	fitresult = prob_causal_graph(X, p = model.p, test = model.test)
	return fitresult, nothing, nothing
end

# Returns counterfactuals for each row in Xnew
function MMI.transform(model::CounterfactualWrapper, fitresult, Xnew)
	ŷ = []
    for r in eachrow(Xnew)
        c = ω -> counterfactual(model.cf, convert(NamedTuple, r[Not(model.cf)]), model.interventions, fitresult, ω)
        ŷ = [ŷ; randsample(ω -> c(ω))]
    end
    Xnew[!, :Ŷ] = ŷ
	return Xnew[!, :Ŷ]
end