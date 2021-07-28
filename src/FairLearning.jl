using Omega, Flux, DataFrames, Distances
using Bijectors
using Flux: @adjoint
ignore(f) = f()
@adjoint ignore(f) = f(), _ -> nothing

export FairLearning!, train!, cond_latent_var

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


# Crrently dependent on data in fl.jl (2nd part)
function cond_latent_var(U::Tuple, X::Dict{<:CausalVar, <:Real}, S::CausalVar, s, ω)
    ret = Float64[]
    for u in U
        om = ω
        temp = u(om)
        for x in keys(X)
            cond!(om, isapprox(x(om), X[x], atol = 0.01))
        end
        cond!(om, isapprox( S(om), s, atol = 0.01))
        ret = vcat(ret, temp)
    end
    return ret
end

function train!(df, cm, A, U, c, pred, adv, ω, λ, l, opt)
    df_x, df_y, df_a = Vector.(eachrow(df[!, Not([:Y, :A])])), df[!, :Y], df[!, :A]
    df_d = Dict{CausalVar{Int64}, Real}[]
    for i in 1:length(df_x)
        d = Dict{CausalVar{Int64}, Real}()
        for k in 1:length(df_x[1]) d[c[k]] = df_x[i][k] end
        push!(df_d, d)
    end
        
    # function P(u, ω)
    #   μ, σ = adv(u)
    #   v = (1 ~ Normal(μ, σ^2))(ω)
    #   Flux.sigmoid(v)
    # end
    function P(u)
        μ, σ = adv(u)
        rand(LogitNormal(μ, exp(σ)))
    end

    function loss(x, y, a, d)
        ans = Float64[]
        for i in 1:100
            u = cond_latent_var(U, d, A, a, ω)
            c = ignore() do
                Context([:U₁, :U₂, :U₃, :U₄, :U₅, :A], vcat(u, a))
            end
            x̃ = apply_context(cm, c)[16:19]
            a′ = P(u)
            c = ignore() do 
                Context([:U₁, :U₂, :U₃, :U₄, :U₅, :A], vcat(u, a′)) 
            end
            x′ = apply_context(cm, c)[16:19]
            ans = vcat(ans, msd(pred(x̃), pred(x′)))
        end
        return mean(l(pred(x), y)) + λ * mean(ans)
    end

    ps_pred = Flux.params(pred)
    ps_adv = Flux.params(adv)
    losses = Float64[]

    for i in 1:size(df, 1)
        x, y, a = df_x[i], df_y[i], df_a[i]
        d = df_d[i]
        loss_i(x, y, a) = loss(x, y, a, d)
        pred_g = gradient(() -> loss_i(x, y, a), ps_pred)
        adv_g = gradient(() -> -loss_i(x, y, a), ps_adv)
        Flux.update!(opt, ps_pred, pred_g)
        Flux.update!(opt, ps_adv, adv_g)
        push!(losses, mean(l(pred(x), y)))
    end
    return losses
end