module HighlyAdaptiveLasso

using MLJModelInterface
using MLJModelInterface: @mlj_model
using RCall


@mlj_model mutable struct HALRegressor <: MLJModelInterface.Deterministic
end


function MLJModelInterface.fit(m::HALRegressor, verbose::Int, X, y)

    fitresult = R"fit_hal(X = $X, Y = $y)"
    cache = nothing
    report = NamedTuple()
    return (fitresult, cache, report)
end


function MLJModelInterface.fitted_params(m::HALRegressor, fitresult)
    return rcopy(fitresult)
end


function MLJModelInterface.predict(m::HALRegressor, fitresult, Xnew)
    ypred = R"predict($fitresult, new_data = $Xnew)"
    return rcopy(ypred)
end


end
