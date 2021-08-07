using Omega, LightGraphs, Base.Threads
import Omega: intervene

export isNonDesc, counterfactual

# Level 1:
"""
    `isNonDesc(m::CausalModel, Y::Tuple, A::Tuple)`

To check for the sufficient condition given in the paper, Counterfactual Fairness:
 "Lemma 1. Let G be the causal graph of the given model (U, V, F). Then Y will be counterfactually
    fair if it is a function of the non-descendants of A."

# Inputs -
- m : Causal model of the dataset
- Y : Tuple of variables used for prediction
- A : Tuple Protected attributes

## Returns true if the condition is satisfied and false if it isn't.
"""
function isNonDesc(m::CausalModel, Y::Tuple, A::Tuple)
    if length(intersect(Y, A)) != 0
        return false
    end
    for n in 1:nv(m)
        for a in A
            if variable(m, n).name == a
                for d in outneighbors(m, n)
                    des = variable(m, d).name
                    if des ∈ Y
                        return false
                    end
                    if !(isNonDesc(m, filter(m -> m ∉ (des, ), Y), (des, )))
                        return false
                    end
                end
            end
        end
    end
    return true
end

# Y is a linear regression of X and A (observable variables)
# In the paper, the authors have written the following:
# Y ~ Normal(* linear combination of X and A *, 1)
"""
    `counterfactual(Y::Symbol, V::NamedTuple, i::Intervention, model::CausalModel, ω::AbstractΩ)`

# Inputs -
- Y : Name of the Variable that is being predicted
- V : NamedTuple of obvservable variables and their values
- i : Intervention to be performed
- model : Causal model
- ω

# Returns the counterfactual P(Yₐ = y | V = v), where Yₐ is the intervened distribution
"""
function counterfactual(Y::Symbol, V::NamedTuple, i::Intervention, model::CausalModel, ω::AbstractΩ)
    Y_ = CausalVar(model, Y)
    X = CausalVar(model, i.X)
    Y′ = intervene(Y_, X => i.x)(ω)
    @threads for k in keys(V)
        for n in 1:nv(model)
            if variable(model, n).name == k
                var = CausalVar(model, variable(model, n).name)
                cond!(ω, isapprox(var(ω), V[k], atol = 0.01))
                break
            end
        end
    end
    return Y′
end

# test remaining
function counterfactual(Y::Symbol, V::NamedTuple, i::PS_Intervention, model::CausalModel, ω::AbstractΩ)
    Y′ = apply_ps_intervention(model, i)(ω)[sum([Y == variable(model, i).name ? i : 0 for i in 1:nv(model)])]
    @threads for k in keys(V)
        for n in 1:nv(model)
            if variable(model, n).name == k
                var = CausalVar(model, variable(model, n).name)
                cond!(ω, isapprox(var(ω), V[k], atol = 0.01))
                break
            end
        end
    end
    return Y′
end

counterfactual(Y::Symbol, V::NamedTuple, i::Interventions, model::CausalModel) = 
        ω -> counterfactual(Y, V, i, model, ω)