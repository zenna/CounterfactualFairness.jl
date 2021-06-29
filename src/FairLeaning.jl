using Omega, Flux

# Assuming Y, the node to be predicted, is in the model as Normal(decision model, 1)
function FairLearning!(df, model::CausalModel, decision, sensitive::Tuple, prior::Tuple, loss, opt, params, ω)    
    # Sampling from posterior (randsample(K |ᶜ Evidence; alg = MH))...MCMC Sampling [Line: 2]
    # u -> result from above, x -> all non-descendants of A (sensitive)
    u = []
    for K in prior
        evidence = df[K]
        for n in 1:nv(model)
            if mechanism(model, n)[name] == K
                pr = mechanism(model, n)[func]
                break
            end
        end
        u = hcat(u, randsample( pr |ᶜ pw(pr(ω) ==ₛ evidence), length(evidence); alg = MH))
    end
    for name in names(df)
        if isNonDesc(model, (name, ), A)
            u = hcat(u, df[name])
        end
    end
    for i in rows(df)
        g = gradient(() -> loss(decision(u[i]), df[Y][i]), params)
        Flux.update(opt, ps, g)
    end
end