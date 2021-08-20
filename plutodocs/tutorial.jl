### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ ed58b9b9-3596-4ffd-8bbf-9c08dafb7de5
begin
	using Pkg
	Pkg.rm("CounterfactualFairness")
	Pkg.add(url = "https://github.com/zenna/Omega.jl#102cc01d1f7dbb4a4caad822746ced6fa5c7164b")
	Pkg.add("Colors")
	using Omega, LightGraphs, Distributions
end

# ╔═╡ 4f7d94b6-7678-46fc-9483-830075d4e5c8
begin
	Pkg.add(url = "https://github.com/zenna/CounterfactualFairness.jl#arw-week2")
	# Pkg.add(url = joinpath(dirname(dirname(pwd())), "CounterfactualFairness.jl"))
	using CounterfactualFairness
end

# ╔═╡ f52c13e9-3655-4f77-ac0a-73a7123775ac
using GraphPlot, Colors

# ╔═╡ 5ed8d5b9-a644-469d-866e-ad26a891fd37
using CSV, DataFrames

# ╔═╡ 377999f1-a44e-4143-844c-6501cb4fd810
md"## CounterfactualFairness.jl"

# ╔═╡ a891e3a8-8f58-47da-a8bc-0161eea2c012
md"##### Why care about fairness in ML?"

# ╔═╡ 8cabacf2-b9bb-4841-ad35-81ba79b1f834
md"Machine learning models are being used to assess loan and job applications, in bail, sentencing and parole decisions, and in an increasing number of impactful decisions.  Unfortunately, due to both bias in the training data and training methods, machine learning models can unfairly discriminate against individuals or groups. While there are many statistical methods to alleviate unfairness, there’s a growing awareness that any account of fairness must take causality into account."

# ╔═╡ 86a1b061-1627-465f-98bb-cf5d0113deb6
md"##### What is counterfactual fairness?"

# ╔═╡ 122a2281-d239-477a-b64f-068a475304f9
md"Counterfactual fairness is a causality-based tool to mitigate bias in machine learning tools. Counterfactuals constitute the third layer of Pearl's Causal Ladder, the first two layers of which are - Association (seeing) and Interventions (doing)."

# ╔═╡ db89f6f2-0dc2-42e2-9c53-7d49cbbbfe07
md"Association is simply describing the observational data through joint and conditional probability distributions. Interventions include fixing the values of particular variables and describing how it influences the causal model. Counterfactuals describe the data in retrospection. It describes questions of the form \"What happens if I had chosen B, given that I chose A?\"."

# ╔═╡ 9ed19ff6-e5cd-448e-94cf-944b6f22b5f2
md"Counterfactual fairness is a definition of fairness that considers a model fair if it predicts the same outcome for a particular individual or group in the real world and in the counterfactual world where they belong to a different demographic."

# ╔═╡ e7d419f8-234a-4929-8021-ad63b818aa5a
md"##### Importing the required packages:"

# ╔═╡ 1303c96e-21bb-40e2-8b35-2f7d7392e041
md"#### Building a causal model from data:"

# ╔═╡ 1fac533d-eb71-4fdf-9e7d-ccde66bfc811
md"Load dataset:"

# ╔═╡ 52a6bc3c-7e53-4a6f-bcbf-ea1788e8c7d8
begin
	df = CSV.read(joinpath(dirname(pwd()), "data", "adult_binary.csv"), DataFrame)
	df = float.(df[!, :])
end

# ╔═╡ 9aa3d013-7623-40d4-afd8-91debfc5c36c
CounterfactualFairness.prob_causal_graph(df)

# ╔═╡ cf5e3b1c-958e-4725-8be0-0ae78fb8d9c7
md"#### Load a causal model:"

# ╔═╡ 4c2a6516-bc5f-401e-b594-6bf0f715c9db
c = @load_law_school

# ╔═╡ b4f7bc22-cdf0-4a33-9bcd-b6cd952f24c9
gplot(c.dag, nodelabel = ([variable(c, i).name for i in 1:nv(c)]), nodefillc = colorant"blue", edgestrokec = colorant"black", layout = stressmajorize_layout, NODESIZE = 0.4/sqrt(nv(c)))

# ╔═╡ 44e5432f-3ec5-48a2-a146-c58fa1145c0a
md"#### Building a causal model by enterring the variables manually:"

# ╔═╡ 88a12369-c15e-4f85-a9a3-821b10a66fbf
md"Create an empty causal graph:"

# ╔═╡ 0f315dec-b0fb-4220-bb24-6a227140884c
g = CausalModel()

# ╔═╡ b280b353-835f-446a-a2e2-4f9071d27b1c
md"Add exogenous variables (latent variables):"

# ╔═╡ c381da1e-fa0e-4e28-96d5-3b1620a2ea6f
begin
	U₁ = add_exo_variable!(g, :U₁, 1 ~ Normal(24, 8))
	U₂ = add_exo_variable!(g, :U₂, 1 ~ Normal(15, 3))
	U₃ = add_exo_variable!(g, :U₃, 1 ~ Normal(2, 1))
end

# ╔═╡ 926bdd6c-66ad-4cae-bdc0-979fa6efa891
md"Add endogenous (observable) variables:"

# ╔═╡ ce575db6-8164-4486-89eb-aad8a70f319c
begin
	Temp = add_endo_variable!(g, :Temp, identity, U₁)
	IceCreamSales = add_endo_variable!(g, :IceCreamSales, *, Temp, U₂)
	Crime = add_endo_variable!(g, :Crime, /, Temp, U₃)
end

# ╔═╡ 29f2bc9b-95ed-459b-8d4f-bc3feb583319
gplot(g.dag, nodelabel = ([variable(g, i).name for i in 1:nv(g)]), nodefillc = colorant"red", edgestrokec = colorant"black", layout = stressmajorize_layout, NODESIZE = 0.4/sqrt(nv(g)))

# ╔═╡ 18ccc45f-c2bf-492b-88dc-3a02f18d5b79
md"To apply a context (values of latent variables) to the model, we may use `apply_context` or use g(ω) where ω is a random variable from `Omega`."

# ╔═╡ 5c168c05-0bde-47e8-ab7c-e0d90a0bc9ca
apply_context(g, (U₁ = 26.2, U₂ = 14.8, U₃ = 2.))

# ╔═╡ 22713c16-8680-48d0-bf9d-40be5de5675d
g(defω())

# ╔═╡ cb9b42f1-a07a-4c95-828a-7982337ea3e6
md"### Interventions:"

# ╔═╡ 3824f163-f4fe-4ff8-bce0-271d68d2468c
md"In the above causal model, we could set the value of temperature to 24 and modify the entire model accordingly (by removing all incloning edges to temperature since it now has a fixed value) as given below -"

# ╔═╡ 464b9613-6d74-4fb4-93a5-0cb3828bb94f
i = CounterfactualFairness.Intervention(:Temp, 24.)

# ╔═╡ d0fa2d8e-f72c-44fd-bbe8-c89b77c381e4
m = apply_intervention(g, i)

# ╔═╡ 565835f6-a27c-455e-9e1a-6ab97a17baa1
gplot(m.dag, nodelabel = ([variable(m, i).name for i in 1:nv(m)]), nodefillc = colorant"green", edgestrokec = colorant"black", layout = stressmajorize_layout, NODESIZE = 0.4/sqrt(nv(m)))

# ╔═╡ abf50dc6-83fe-422a-b417-da5ad8375444
md"### Counterfactuals"

# ╔═╡ 5baa385b-a46b-4407-b5c2-2ad16b7ddf78
md"$P(Y = y | do(X = x), V = v)$

where V contain observed observed variables."

# ╔═╡ bb4da83e-375f-4bfa-ae7d-a31c16fbb545
md"To obtain counterfactuals, if the observation values are: $(Temp = 26.0,  IceCreamSales = 340., Crime = 1.)$, we condition Y on the observed values and the intervention $do(Temp = 24.0)$. "

# ╔═╡ 24d29a5b-26d1-402d-bc8c-647ac39d7230
count = ω -> counterfactual(:Crime, (IceCreamSales = 340., Crime = 1.), i, g, ω)

# ╔═╡ 9435ea98-573f-4744-8363-9a34b7080682
randsample(ω -> count(ω))

# ╔═╡ b875e2c6-bb9d-4426-9a76-67abdb78cab1
# isNonDesc, path-specific(?), training (?), verification

# ╔═╡ Cell order:
# ╟─377999f1-a44e-4143-844c-6501cb4fd810
# ╟─a891e3a8-8f58-47da-a8bc-0161eea2c012
# ╟─8cabacf2-b9bb-4841-ad35-81ba79b1f834
# ╟─86a1b061-1627-465f-98bb-cf5d0113deb6
# ╟─122a2281-d239-477a-b64f-068a475304f9
# ╟─db89f6f2-0dc2-42e2-9c53-7d49cbbbfe07
# ╟─9ed19ff6-e5cd-448e-94cf-944b6f22b5f2
# ╟─e7d419f8-234a-4929-8021-ad63b818aa5a
# ╠═ed58b9b9-3596-4ffd-8bbf-9c08dafb7de5
# ╠═f52c13e9-3655-4f77-ac0a-73a7123775ac
# ╠═4f7d94b6-7678-46fc-9483-830075d4e5c8
# ╟─1303c96e-21bb-40e2-8b35-2f7d7392e041
# ╟─1fac533d-eb71-4fdf-9e7d-ccde66bfc811
# ╠═5ed8d5b9-a644-469d-866e-ad26a891fd37
# ╠═52a6bc3c-7e53-4a6f-bcbf-ea1788e8c7d8
# ╠═9aa3d013-7623-40d4-afd8-91debfc5c36c
# ╟─cf5e3b1c-958e-4725-8be0-0ae78fb8d9c7
# ╠═4c2a6516-bc5f-401e-b594-6bf0f715c9db
# ╠═b4f7bc22-cdf0-4a33-9bcd-b6cd952f24c9
# ╟─44e5432f-3ec5-48a2-a146-c58fa1145c0a
# ╟─88a12369-c15e-4f85-a9a3-821b10a66fbf
# ╠═0f315dec-b0fb-4220-bb24-6a227140884c
# ╟─b280b353-835f-446a-a2e2-4f9071d27b1c
# ╠═c381da1e-fa0e-4e28-96d5-3b1620a2ea6f
# ╟─926bdd6c-66ad-4cae-bdc0-979fa6efa891
# ╠═ce575db6-8164-4486-89eb-aad8a70f319c
# ╠═29f2bc9b-95ed-459b-8d4f-bc3feb583319
# ╟─18ccc45f-c2bf-492b-88dc-3a02f18d5b79
# ╠═5c168c05-0bde-47e8-ab7c-e0d90a0bc9ca
# ╠═22713c16-8680-48d0-bf9d-40be5de5675d
# ╟─cb9b42f1-a07a-4c95-828a-7982337ea3e6
# ╟─3824f163-f4fe-4ff8-bce0-271d68d2468c
# ╠═464b9613-6d74-4fb4-93a5-0cb3828bb94f
# ╠═d0fa2d8e-f72c-44fd-bbe8-c89b77c381e4
# ╠═565835f6-a27c-455e-9e1a-6ab97a17baa1
# ╟─abf50dc6-83fe-422a-b417-da5ad8375444
# ╟─5baa385b-a46b-4407-b5c2-2ad16b7ddf78
# ╟─bb4da83e-375f-4bfa-ae7d-a31c16fbb545
# ╠═24d29a5b-26d1-402d-bc8c-647ac39d7230
# ╠═9435ea98-573f-4744-8363-9a34b7080682
# ╠═b875e2c6-bb9d-4426-9a76-67abdb78cab1
