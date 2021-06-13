### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 2fff61a0-cc83-11eb-2ddc-0f07685dd90b
begin
	using CounterfactualFairness, Test
	using Omega
end

# ╔═╡ c4c5775d-9d24-409a-8296-992a7628f508
U₁ = normal(27, 5)

# ╔═╡ 3016d627-e26f-4718-bcb9-7cf46c6246c8
U₂ = normal(2, 4)

# ╔═╡ 242a662e-a992-4e4c-b5bf-17d35ce1fccf
U₃ = normal(1, 2)

# ╔═╡ ad904494-2d77-43e3-b4b9-ddc0bc2e2097
begin
	X = U₁
	Y = 4 * X + U₂
	Z = X / 10 + U₃
end

# ╔═╡ 8faac207-f411-455e-b65a-13c9da49ac1b
begin
	g = CausalModel()
	g = add_vertex(g, (:Temp, U₁))
	g = add_vertex(g, (:IceCreamSales, U₂))
	g = add_vertex(g, (:Crime, U₃))
end

# ╔═╡ c381da1e-fa0e-4e28-96d5-3b1620a2ea6f


# ╔═╡ Cell order:
# ╠═2fff61a0-cc83-11eb-2ddc-0f07685dd90b
# ╠═c4c5775d-9d24-409a-8296-992a7628f508
# ╠═3016d627-e26f-4718-bcb9-7cf46c6246c8
# ╠═242a662e-a992-4e4c-b5bf-17d35ce1fccf
# ╠═ad904494-2d77-43e3-b4b9-ddc0bc2e2097
# ╠═8faac207-f411-455e-b65a-13c9da49ac1b
# ╠═c381da1e-fa0e-4e28-96d5-3b1620a2ea6f
