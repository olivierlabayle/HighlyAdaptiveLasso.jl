using HighlyAdaptiveLasso
using Documenter

DocMeta.setdocmeta!(HighlyAdaptiveLasso, :DocTestSetup, :(using HighlyAdaptiveLasso); recursive=true)

makedocs(;
    modules=[HighlyAdaptiveLasso],
    authors="Olivier Labayle",
    repo="https://github.com/olivierlabayle/HighlyAdaptiveLasso.jl/blob/{commit}{path}#{line}",
    sitename="HighlyAdaptiveLasso.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://olivierlabayle.github.io/HighlyAdaptiveLasso.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/olivierlabayle/HighlyAdaptiveLasso.jl",
)
