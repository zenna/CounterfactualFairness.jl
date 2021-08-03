### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ ecde16f7-f53c-48d8-b9fb-830081ecfe06
begin
	using Pkg
	Pkg.add(url = "https://github.com/zenna/Omega.jl#8254e97bea3d8434d20369904977968a716ea047")
end

# ╔═╡ e25855c0-611b-4a41-8c70-4a0d546d269a
Pkg.add(url = dirname(pwd()))

# ╔═╡ ed58b9b9-3596-4ffd-8bbf-9c08dafb7de5
using Omega, LightGraphs, Distributions

# ╔═╡ 4f7d94b6-7678-46fc-9483-830075d4e5c8
using CounterfactualFairness

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

# ╔═╡ 44e5432f-3ec5-48a2-a146-c58fa1145c0a
md"### Building a causal model:"

# ╔═╡ 8faac207-f411-455e-b65a-13c9da49ac1b
begin
	g = CausalModel()
	U₁ = add_exo_variable!(g, :U₁, 1 ~ Normal(24, 8))
	U₂ = add_exo_variable!(g, :U₂, 1 ~ Normal(15, 3))
	U₃ = add_exo_variable!(g, :U₃, 1 ~ Normal(2, 1))
	Temp = add_vertex(g, :Temp, identity, U₁)
	IceCreamSales = add_endo_variable!(g, :IceCreamSales, *, Temp, U₂)
	Crime = add_endo_variable!(g, :Crime, /, Temp, U₃)
end

# ╔═╡ c381da1e-fa0e-4e28-96d5-3b1620a2ea6f


# ╔═╡ Cell order:
# ╟─377999f1-a44e-4143-844c-6501cb4fd810
# ╟─a891e3a8-8f58-47da-a8bc-0161eea2c012
# ╟─8cabacf2-b9bb-4841-ad35-81ba79b1f834
# ╟─86a1b061-1627-465f-98bb-cf5d0113deb6
# ╟─122a2281-d239-477a-b64f-068a475304f9
# ╟─db89f6f2-0dc2-42e2-9c53-7d49cbbbfe07
# ╟─9ed19ff6-e5cd-448e-94cf-944b6f22b5f2
# ╟─e7d419f8-234a-4929-8021-ad63b818aa5a
# ╠═ecde16f7-f53c-48d8-b9fb-830081ecfe06
# ╠═ed58b9b9-3596-4ffd-8bbf-9c08dafb7de5
# ╠═e25855c0-611b-4a41-8c70-4a0d546d269a
# ╠═4f7d94b6-7678-46fc-9483-830075d4e5c8
# ╟─44e5432f-3ec5-48a2-a146-c58fa1145c0a
# ╠═8faac207-f411-455e-b65a-13c9da49ac1b
# ╠═c381da1e-fa0e-4e28-96d5-3b1620a2ea6f
