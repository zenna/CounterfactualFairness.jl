using Omega, Distributions, CSV, DataFrames

export @load_law_school, @load_adult, @load_synthetic

"""
Macro to create probabilistic causal model of law school data,
used in the paper, [Counterfactual Fairness](https://papers.nips.cc/paper/2017/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html).
- Sensitive attributes : :Sex, :Race
- Observed variables : :GPA, :LSAT
- Exogenous variables : :Knowledge
"""
macro load_law_school()
    quote
        law_school_cm = CausalModel()

        # Exogenous Variables
        Sex = add_exo_variable!(law_school_cm, :Sex, 1 ~ Bernoulli(0.6))
        Race  = add_exo_variable!(law_school_cm, :Race, 2 ~ Bernoulli(0.75))
        Knowledge = add_exo_variable!(law_school_cm, :Knowledge, 3 ~ Normal(0, 1))

        # Endogenous variables
        R_GPA = add_endo_variable!(law_school_cm, :R_GPA, *, 2.1, Race)
        S_GPA = add_endo_variable!(law_school_cm, :S_GPA, *, 3.3, Sex)
        R_LSAT = add_endo_variable!(law_school_cm, :R_LSAT, *, 5.8, Race)
        S_LSAT = add_endo_variable!(law_school_cm, :S_LSAT, *, 0.7, Sex)
        R_FYA = add_endo_variable!(law_school_cm, :R_FYA, *, 2.3, Race)
        S_FYA = add_endo_variable!(law_school_cm, :S_FYA, *,  1, Sex)
        GPA′ = add_endo_variable!(law_school_cm, :GPA′, +, Knowledge, R_GPA, S_GPA)
        GPA = add_endo_variable!(law_school_cm, :GPA, θ -> 4 ~ Normal(θ, 0.1), GPA′)
        LSAT′ = add_endo_variable!(law_school_cm, :LSAT′, +, Knowledge, R_LSAT, S_LSAT)
        LSAT = add_endo_variable!(law_school_cm, :LSAT, θ -> 5 ~ Normal(θ, 0.1), LSAT′)
        FYA′ = add_endo_variable!(law_school_cm, :FYA′, +, Knowledge, R_FYA, S_FYA)
        FYA = add_endo_variable!(law_school_cm, :FYA, θ -> 6 ~ Normal(θ, 1), FYA′)

        law_school_cm
    end
end

"""
Macro to create probabilistic causal model of [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult),
the attributes of which are as selected in paper - [Counterfactual Fairness: Unidentification, Bound and Algorithm](https://www.ijcai.org/proceedings/2019/199).
The dataset used is from [here](https://github.com/yongkaiwu/Counterfactual-Fairness/tree/master/data/adult).
"""
macro load_adult()
    quote
        path = joinpath(pwd(), "data", "adult_binary.csv")
        df = CSV.read(path, DataFrame)
        df = float.(df[!, :])
        adult = prob_causal_graph(df)
        adult
    end
end

# macro load_COMPAS()
#     quote
        
#     end
# end

"""
Macro to create the probabilistic causal model for 
the toy dataset used in the paper - 
[Adversarial Learning for Counterfactual Fairness](https://arxiv.org/pdf/2008.13122.pdf)
- Sensitive attributes : :A
- Observed variables : :X1, :X2, :X3, :X4
- Exogenous variables : :U₁, :U₂, :U₃, :U₄, :U₅
"""
macro load_synthetic()
    toy = CausalModel()
    U₁ = add_exo_variable!(toy, :U₁, 1 ~ Normal(0, 1))
    U₂ = add_exo_variable!(toy, :U₂, 2 ~ Normal(0.5, 2))
    U₃ = add_exo_variable!(toy, :U₃, 3 ~ Normal(1, sqrt(2)))
    U₄ = add_exo_variable!(toy, :U₄, 4 ~ Normal(1.5, sqrt(3)))
    U₅ = add_exo_variable!(toy, :U₅, 5 ~ Normal(2, sqrt(2)))
    A = add_exo_variable!(toy, :A, 6 ~ Normal(45, sqrt(5)))

    A1 = add_endo_variable!(toy, :A1, *, 0.1, A)
    A2 = add_endo_variable!(toy, :A2, *, 5, A)
    A3 = add_endo_variable!(toy, :A3, *, 7, A)
    U₂1 = add_endo_variable!(toy, :U₂1, x -> x^2, U₂)
    U₃1 = add_endo_variable!(toy, :U₃1, *, 0.1, U₃)
    X1_ = add_endo_variable!(toy, :X1_, +, 7, A1, U₁, U₂, U₃)
    X2_ = add_endo_variable!(toy, :X2_, +, 80, A, U₂1)
    X3_ = add_endo_variable!(toy, :X3_, +, 200, A2, U₃1)
    X4_ = add_endo_variable!(toy, :X4_, +, 1000, A2, U₄, U₅)

    X1 = add_endo_variable!(toy, :X1, θ -> 7 ~ Normal(θ, 1), X1_) # 16
    X2 = add_endo_variable!(toy, :X2, θ -> 8 ~ Normal(θ, sqrt(10)), X2_)
    X3 = add_endo_variable!(toy, :X3, θ -> 9 ~ Normal(θ, sqrt(20)), X3_)
    X4 = add_endo_variable!(toy, :X4, θ -> 10 ~ Normal(θ, sqrt(1000)), X4_)
    # X = add_endo_variable!(toy, :X, vcat, X1, X2, X3, X4)

    Y1 = add_endo_variable!(toy, :Y1, +, U₁, U₂, U₃, U₄, U₅)
    Y2 = add_endo_variable!(toy, :Y2, *, 20, Y1)
    Y3 = add_endo_variable!(toy, :Y3, +, Y2, A3)
    Y4 = add_endo_variable!(toy, :Y4, *, 2, Y3)
    Y = add_endo_variable!(toy, :Y, θ -> 11 ~ Normal(θ, sqrt(0.1)), Y4)
    return toy
end