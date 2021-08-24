using Omega, Flux, DataFrames, Distances
using NamedTupleTools
using Flux: @adjoint
ignore(f) = f()
@adjoint ignore(f) = f(), _ -> nothing

export FairLearning!, train!, cond_latent_var, ℒ, P

# Assuming Y, the node to be predicted, is in the model as Normal(decision model, 1)
" FairLearning algorithm from the Counterfactual Fairness paper by Kusner et al"
function FairLearning!(df, model::CausalModel, Y, sensitive::Tuple, prior::Tuple, loss, opt, ps, ω)    
    # Sampling from posterior (randsample(K |ᶜ Evidence; alg = MH))...MCMC Sampling [Line: 2]
    # u -> result from above, x -> all non-descendants of A (sensitive)
    u = []
    ep = 10
    for K in prior
        for n in 1:nv(model)
            if variable(model, n).name == K
                pr = variable(model, n).func
                desc = outneighbors(model, n)
                for i in 1:div(size(df, 1), ep)
                    for p in desc
                        evidence = df[!, variable(model, p).name]
                        par_f = CausalVar(model, variable(model, p).name)
                        for e in evidence
                            pr = pr |ᶜ (par_f(ω) ==ₛ e)
                        end
                        u = vcat(u, randsample(ω -> pr(ω), ep; alg = MH))
                    end
                end
                break
            end
        end
    end
    for name in names(df[!, Not(Y)])
        if isNonDesc(model, (Symbol(name), ), sensitive)
            u = hcat(u, df[!, name])
        end
    end
    data = zip(u, df[!, Y])
    Flux.train!(loss, ps, data, opt)
end

function cond_latent_var(U::Tuple, X::Vector{Pair{CausalVar{Int64}, Float64}}, S::CausalVar, s::Real, ω::AbstractΩ)
    ret = Float64[]
    for u in U
        om = ω
        temp = u(om)
        for x in X
            cond!(om, isapprox(x.first(om), x.second, atol = 0.01))
        end
        cond!(om, isapprox(S(om), s, atol = 0.01))
        ret::Vector{Float64} = [ret; temp]
    end
    return ret
end

function P(u, ω, adv)
    μ, σ = adv(u)
    v = (1 ~ Normal(μ, σ^2))(ω)
    Flux.sigmoid(v)
end

function ℒ(cm::CausalModel, X::Tuple, x, y::Float64, A::CausalVar, a::Float64, U::Tuple, adv, pred, loss, ω::AbstractΩ; λ = 0.1, epochs = 100)
    ans = Float64[]
    d = ignore() do
        X .=> x
    end
    u = cond_latent_var(U, d, A, a, ω)
    exo_names = Symbol[]
    for n in U
        exo_names::Vector{Symbol} = [exo_names; n.varname]
    end
    exo_names = [exo_names; A.varname]
    c1 = ignore() do
        Context(exo_names, vcat(u, a))
    end
    for i in 1:epochs
        ac1 = apply_context(cm, c1, ω)
        x̃ = ignore() do 
            collect(values(NamedTupleTools.select(ac1, map(x -> x.varname, X))))
        end
        a′ = P(u, ω, adv)
        c = ignore() do 
            Context(exo_names, vcat(u, a′))
        end
        ac2 = apply_context(cm, c, ω)
        x′ = ignore() do 
            collect(values(NamedTupleTools.select(ac2, map(x -> x.varname, X))))
        end
        ans::Vector{Float64} = [ans; msd(pred(x̃), pred(x′))]
    end
    return abs(mean(loss(pred(x), y)) + λ * mean(ans))
end

function train!(data_loader, cm::CausalModel, A::CausalVar, U::Tuple, X::Tuple, pred, adv, ω, λ, l, opt_pred, opt_adv; cfs = 100)
    ps_pred = ignore() do 
        Flux.params(pred)
    end
    ps_adv = ignore() do
        Flux.params(adv)
    end
    losses = Float64[]

    for (x, y, a) in data_loader
        pred_g = gradient(() -> ℒ(cm, X, x[1], y[1], A, a[1], U, adv, pred, l, ω, λ = λ, epochs = cfs), ps_pred)
        adv_g = gradient(() -> -ℒ(cm, X, x[1], y[1], A, a[1], U, adv, pred, l, ω, λ = λ, epochs = cfs), ps_adv)
        Flux.update!(opt_pred, ps_pred, pred_g)
        Flux.update!(opt_adv, ps_adv, adv_g)
        push!(losses, mean(l(pred(x[1]), y[1])))
    end
    return losses
end