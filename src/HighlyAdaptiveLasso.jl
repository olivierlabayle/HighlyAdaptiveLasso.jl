module HighlyAdaptiveLasso

using MLJModelInterface
using MLJModelInterface: @mlj_model
using RCall


################################
########## Structure ###########
################################


@mlj_model mutable struct HALRegressor <: MLJModelInterface.Deterministic
end



################################
########### Methods ############
################################


function MLJModelInterface.fit(m::HALRegressor, verbose::Int, X, y)
    R"library(hal9001)"
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


################################
########### METADATA ###########
################################


const ALL_MODELS = Union{HALRegressor}


MLJModelInterface.metadata_pkg.(ALL_MODELS,
    name       = "HighlyAdaptiveLasso",
    uuid       = "c5dac772-1445-43c4-b698-9440de7877f6",
    url        = "https://github.com/olivierlabayle/HighlyAdaptiveLasso.jl",
    julia      = true,
    license    = "MIT",
    is_wrapper = false,
)


MLJModelInterface.metadata_model(HALRegressor,
    input_scitype    = MLJModelInterface.Table(MLJModelInterface.Continuous),
    target_scitype   = AbstractVector{MLJModelInterface.Continuous},
    output_scitype   = MLJModelInterface.Table(MLJModelInterface.Continuous),
    supports_weights = false,
    descr            = "The Highly Adaptive Lasso",
	load_path        = "HighlyAdaptiveLasso.HALRegressor"
    )


################################
########### Exports ############
################################

export HALRegressor


end
