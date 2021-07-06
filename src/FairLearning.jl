using Omega, Flux, DataFrames

export FairLearning!

# Assuming Y, the node to be predicted, is in the model as Normal(decision model, 1)
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