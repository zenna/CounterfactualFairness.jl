using Distributions, Omega, CausalInference

export verify, cf_effect, ps_effect

# CF = 1/m Σ (E(Δ(h(X) - h(X <- a′))))
"To verify counterfactual (or path-specific) fairness"
function verify(test, Y::CausalVar, X, A::Symbol, a′, a, effect=cf_effect; ϵ=0.05, epochs=10)
    m = effect(test, Y, X, A, a′, a; epochs)
    if m > ϵ
        return false
    else
        return true
    end
end

"Computing counterfactual effect of given data (dataframe)"
function cf_effect(df, predictor, Y::Symbol, A::Symbol, a′, model; epochs=10)
    m = 0.
    i = CounterfactualFairness.Intervention(A, a′)
    for x in eachrow(df)
        cfs = []
        for _ in 1:epochs
            push!(cfs, randsample(counterfactual(Y, NamedTuple(x[Not([Y, A])]), i, model)))
        end
        m += mean(abs.(predictor(x[Y]) .- cfs))
    end
    m /= size(test, 1)
    return m
end

"Computing path-specific effect of given data (dataframe)"
function ps_effect(df, predictor, Y::Symbol, X, A::Symbol, a′, a, blocked_edges, model; epochs=10)
    blocked_graph = DiGraph(nv(model))
    for e in blocked_edges.edges
        add_edge!(blocked_graph, e)
    end
    @assert has_recanting_witness(model.dag, 1, nv(model), blocked_graph) "The graph has recanting witness"
    m = 0.
    i₁ = CounterfactualFairness.Intervention(A, a′)
    i₂ = CounterfactualFairness.Intervention(A, a)
    i = PS_Intervention(blocked_edges, i₁, i₂)
    for x in eachrow(df)
        obs = x[Not([Y, A])]
        y = predictor(x)
        cfs = []
        for _ in 1:epochs
            push!(cfs, randsample(counterfactual(Y, NamedTuple(zip(X, obs)), i, model)))
        end
        m += mean(abs.(predictor(x[Y]) .- cfs))
    end
    m /= size(test, 1)
    return m
end