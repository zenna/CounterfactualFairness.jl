using Documenter, CounterfactualFairness
makedocs(
    modules = [CounterfactualFairness],
    pages = [
        "Home" => "index.md",
        "Functions" => "functions.md"
    ]
    sitename = "CounterfactualFairness.jl"
)

deploydocs(
    repo = "github.com/zenna/CounterfactualFairness.jl.git",
)