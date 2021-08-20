using CausalInference, Distributions, LightGraphs, LinearAlgebra, Omega

export prob_causal_graph, empirical_mechanism
# Y = aX + b + c*u, u ~ N(0, 1)
"""
    `empirical_mechanism((; μ=..., Σ=...), A, B)`
 Emprical mechanism of A given B
"""
function empirical_mechanism(P, A, B)
    Z = P.Σ[A,B] * inv(P.Σ[B,B])
    # (; μ = P.μ[A] + Z*(xB - P.μ[B]), Σ = P.Σ[A,A] - Z*P.Σ[B,A])
    (; a=Z, b=P.μ[A] + Z * ( - P.μ[B]), c=P.Σ[A,A] - Z * P.Σ[B,A])
end

"Returns the fully specified causal graph assuming Gausain mechanism for each variable"
function prob_causal_graph(df, cg)
    if any(Bool.(Matrix(adjacency_matrix(cg, dir=:in)) .& Matrix(adjacency_matrix(cg, dir=:out))))
        println("There are undirected edges in the graph, the graph is: ", cg)
        throw(error())
    end
    order = topological_sort_by_dfs(cg)
    cm = CausalModel()
    id = 1
    μ = vec(mean(Matrix(df), dims=1))
    Σ = cov(Matrix(df))
    for node in order
        if length(inneighbors(cg, node)) == 0
            add_exo_variable!(cm, Symbol(names(df)[node]), id ~ Normal(μ[node], sqrt(Σ[node, node])))
        else
            B = [node]
            A = inneighbors(cg, node)
            a, b, c = empirical_mechanism((μ = μ, Σ = Σ), A, B)
            pa = []
            pc = []
            for i in 1:length(a)
                par = CausalVar(cm, variable(cm, i).name)
                ex = add_exo_variable!(cm, Symbol("U" * string(i)), id ~ Normal(0, 1))
                push!(pc, add_endo_variable!(cm, Symbol("U′" * string(i)), *, c[i], ex))
                push!(pa, add_endo_variable!(cm, Symbol(string(names(df)[node]) * string(i)), *, a[i], par))
                id = id + 1
            end
            add_endo_variable!(cm, Symbol(names(df)[node]), +, sum(b), vcat(pa, pc)...)
        end
    end
    return cm
end

prob_causal_graph(df; p = 0.01, test = gausscitest) =
    prob_causal_graph(df, pcalg(df, p, test))