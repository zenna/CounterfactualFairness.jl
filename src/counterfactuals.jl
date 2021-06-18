using Omega, LightGraphs

export isNonDesc

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
            if mechanism(m, n).name == a
                for d in outneighbors(m, n)
                    if mechanism(m, d).name ∈ Y
                        return false
                    end
                end
            end
        end
    end
    return true
end