using MLJBase, Omega, MLJModels, MLJFlux

export AdversarialWrapper

"""
    AdversarialWrapper

 As given in the paper - [Adversarial Learning for Counterfactual Fairness](https://arxiv.org/pdf/2008.13122.pdf),
 it is an algorithm that enables the simulation of counterfactual samples used for
 training the target fair model, the goal being to produce similar outcomes
 for every alternate version of any individual. It relies on an adversarial neural learning
 approach.
"""
struct AdversarialWrapper{M<:Chain, N<:Chain} <: Supervised
    cm::CausalModel
	grp::Symbol
    latent::Vector{Symbol}
    observed::Vector{Symbol}
	predictor::M
	adversary::N
    loss
    ω::AbstractΩ
    num_cf::Int64
    λ::Float64
    iters::Int64
    lr::Float64
end

function AdversarialWrapper(;
    cm::CausalModel = nothing,
	grp::Symbol = :class,
    latent::Array{Symbol} = [],
    observed::Array{Symbol} = [],
    predictor = nothing,
    adversary = nothing,
    loss,
    ω::AbstractΩ = defω(),
    num_cf::Int64 = 100,
    λ::Float64 = 0.1,
    iters::Int64 = 10,
    lr::Float64 = 10^(-3))

    # predictor = NeuralNetworkRegressor(loss = loss, optimiser = ADAM(lr))

    # build = MLJFlux.@builder Chain(Dense(length(latent), 2, relu))
    # adversary = MultitargetNeuralNetworkRegressor(builder = build, loss = loss, optimiser = ADAM(10*model.lr))

    model = AdversarialWrapper(cm, grp, latent, observed, predictor, adversary, loss, ω, num_cf, λ, iters, lr)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.clean!(model::AdversarialWrapper)
	warning = ""
    model.predictor !== nothing || (warning *= "No predictor specified in model\n")
    model.adversary !== nothing || (warning *= "No adversary specified in model\n")
    return warning
end

function MMI.fit(model::AdversarialWrapper, verbosity::Int, X, y)
	sensitive = X[!, model.grp]
	# TODO: Check if the sensitive attribute is continuous or discrete
    obs = Vector.(eachrow(X[!, Not(model.grp)]))
    data = Flux.DataLoader((obs, y, sensitive))

    O = Tuple([CausalVar(model.cm, name) for name in model.observed])
    A = CausalVar(model.cm, model.grp)
    U = Tuple([CausalVar(model.cm, name) for name in model.latent])

    train!(data, model.cm, A, U, O, model.predictor, model.adversary, model.ω, model.λ, model.loss, ADAM(model.lr), ADAM(10*model.lr))

    fitresult = model.predictor
	return fitresult, nothing, nothing
end

function MMI.predict(model::AdversarialWrapper, fitresult, Xnew)
	return reduce(vcat, fitresult.(Vector.(eachrow(Xnew[!, Not(model.grp)]))))
end