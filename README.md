# CounterfactualFairness.jl

`CounterfactualFairness.jl` is a julia package to implement causality-based tools for checking and forcing fairness into machine learning models.

It enables the user to do the following -
- Produce a fully-specified causal model from a dataset.
- Build a causal graph manually.
- Apply the causal model to a context.
- Compute interventions and counterfactuals.
- Train a model to be counterfactually fair.
- Verify counterfactual fairness.

# Installation

```
(@v1.6) pkg> add https://github.com/zenna/CounterfactualFairness.jl
```

# Usage

To start using the package,
```
julia> using CounterfactualFairness
```