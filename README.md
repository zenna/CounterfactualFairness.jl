# CounterfactualFairness.jl

`CounterfactualFairness.jl` is a julia package to implement causality-based tools for checking and forcing fairness into machine learning models.

It enables the user to do the following -
- Produce a fully-specified causal model from a dataset.
- Build a causal graph by adding the exogenous and endogenous variables.
- Compute interventions and counterfactuals.
- Train a model to be counterfactually fair.
- Compute counterfactual effect and verify counterfactual fairness.

The package also provides methods to compute counterfactuals and train models as given in the MLJ framework.

For an introduction to the package, see [here](https://nextjournal.com/archanarw/counterfactualfairnessjl)

# Installation

```julia
(@v1.6) pkg> add https://github.com/zenna/CounterfactualFairness.jl#arw-week2
```

# Usage

To start using the package,
```julia
julia> using CounterfactualFairness
```