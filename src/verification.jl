using Distributions, Omega
using ForwardDiff

export verify, cf_effect, ps_effect

# CF = 1/m Σ (E(Δ(h(X) - h(X <- a′))))
function verify(test, Y::CausalVar, X, A::Symbol, a′, a, effect = cf_effect; ϵ=0.01, epochs = 10)
    m = effect(test, Y, X, A, a′, a; epochs)
    @show m
    if m > ϵ
        return false
    else
        return true
    end
end

function cf_effect(test, Y::CausalVar, X, A::Symbol, a′, a; epochs = 10)
    m = 0
    function cf(ω, x)
        Y_int_a′ = intervene(Y, Intervention(A, a′))(ω)
        for k in X
            for n in 1:nv(Y.model)
                if variable(Y.model, n).name == k
                    v = CausalVar(Y.model, k)(ω)
                    cond!(ω, isapprox(v, x[k], atol = 0.1))
                    break
                end
            end
        end
        cond!(ω, isapprox(CausalVar(Y.model, A)(ω), x[A], atol = 0.1))
        Y_int_a′
    end
    for x in eachrow(test)
        y = Y(x...)
        cfs = []
        for i in 1:epochs
            push!(cfs, randsample(ω -> cf(ω, x)))
        end
        m += mean(abs.(y .- cfs))
    end
    m /= size(test, 1)
    return m
end

function ps_effect(test, Y::CausalVar, X, A::Symbol, a′, a; epochs = 10)
    m = 0
    function cf(ω, x₁, x₂)
        int_model = apply_ps_intervention(g, x₁, blocked_edges, x₂, ω)
        Y_int_a′ = CausalVar(int_model, Y)
        for k in X
            for n in 1:nv(Y.model)
                if variable(Y.model, n).name == k
                    v = CausalVar(Y.model, k)(ω)
                    cond!(ω, isapprox(v, x[k], atol = 0.1))
                    break
                end
            end
        end
        cond!(ω, isapprox(CausalVar(Y.model, A)(ω), x[A], atol = 0.1))
        Y_int_a′
    end
    for x in eachrow(test)
        y = Y(x...)
        cfs = []
        for i in 1:epochs
            push!(cfs, randsample(ω -> cf(ω, a′, a)))
        end
        m += mean(abs.(y .- cfs))
    end
    m /= size(test, 1)
    return m
end