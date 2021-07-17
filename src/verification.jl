using Distributions, Omega
using ForwardDiff

export verify

# CF = 1/m Σ (E(Δ(h(X) - h(X <- a′))))
function verify(test, Y::CausalVar, X, A::Symbol, a′; ϵ=0.01, epochs = 10)
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
    @show m
    if m > ϵ
        return false
    else
        return true
    end
end