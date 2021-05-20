module HighlyAdaptiveLasso

using MLJModelInterface
using MLJModelInterface: @mlj_model
using RCall


@mlj_model mutable struct HAL <: MLJModelInterface.Deterministic
end


function MLJModelInterface.fit(m::HAL, verbose::Int, X, y)

    fitresult = R"fit_hal(X = x, Y = y)"
    cache = nothing
    report = NamedTuple()
    return (fitresult, cache, report)
end

end
