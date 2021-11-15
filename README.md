# HighlyAdaptiveLasso

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://olivierlabayle.github.io/HighlyAdaptiveLasso.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://olivierlabayle.github.io/HighlyAdaptiveLasso.jl/dev)
[![Build Status](https://github.com/olivierlabayle/HighlyAdaptiveLasso.jl/workflows/CI/badge.svg)](https://github.com/olivierlabayle/HighlyAdaptiveLasso.jl/actions)
[![Coverage](https://codecov.io/gh/olivierlabayle/HighlyAdaptiveLasso.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/olivierlabayle/HighlyAdaptiveLasso.jl)


A MLJ wrapper to the R [HAL](https://github.com/tlverse/hal9001) package.


## Installation


The project relies on [RCall](https://juliainterop.github.io/RCall.jl/stable/installation/), if you don't want to fallback on a Conda installation, you should make sure the `R_HOME` environment variable is correctly set. For known issues regarding RCall compatibility look into the RCall intallation section.

You will also need to install the original [HAL package](https://github.com/tlverse/hal9001) in your R environment.

This wrapper can then be installed via:

```julia
add HighlyAdaptiveLasso
```

## Usage

```julia

using HighlyAdaptiveLasso, MLJ, Random

X = randn(100, 4)
y = randn(100)

hal = HALRegressor()
mach = machine(hal, X, y)

evaluate!(mach)

```
