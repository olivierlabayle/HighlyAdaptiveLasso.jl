# HighlyAdaptiveLasso

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://olivierlabayle.github.io/HighlyAdaptiveLasso.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://olivierlabayle.github.io/HighlyAdaptiveLasso.jl/dev)
[![Build Status](https://github.com/olivierlabayle/HighlyAdaptiveLasso.jl/workflows/CI/badge.svg)](https://github.com/olivierlabayle/HighlyAdaptiveLasso.jl/actions)
[![Coverage](https://codecov.io/gh/olivierlabayle/HighlyAdaptiveLasso.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/olivierlabayle/HighlyAdaptiveLasso.jl)


A MLJ wrapper to the R [HAL](https://github.com/tlverse/hal9001) package.


## Installation


The project relies on [RCall](https://juliainterop.github.io/RCall.jl/stable/installation/), you should make sure the `R_HOME` environment 
variable is correctly set as described at the previous address. 
I suggest **not** using the `ENV["R_HOME"] = "*"` which will default to using [Conda.jl](https://github.com/JuliaPy/Conda.jl) as I got into trouble with it.

Then simply run:

```julia
add "https://github.com/olivierlabayle/HighlyAdaptiveLasso.jl"
```

## Usage

```julia

using HighlyAdaptiveLasso, MLJ, Random

X = randn(100, 4)
y = randn(100)

model = HAL()
mach = machine(HAL, X, y)

evaluate!(mach)

```