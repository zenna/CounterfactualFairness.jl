using Omega, Distributions, CSV, DataFrames

export @load_law_school, @load_adult

"""
Macro to create probabilistic causal model of law school data,
used in the paper, [Counterfactual Fairness](https://papers.nips.cc/paper/2017/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html).
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
        df = CSV.read("adult_binary.csv", DataFrame)
        df = float.(df[!, :])
        adult = prob_causal_graph(df)
        adult
    end
end

# macro load_COMPAS()
#     quote
        
#     end
# end