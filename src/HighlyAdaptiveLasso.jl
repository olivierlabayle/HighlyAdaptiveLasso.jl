module HighlyAdaptiveLasso

using MLJModelInterface
using MLJModelInterface: @mlj_model
using RCall


################################
########## Structure ###########
################################


"""
    HALRegressor()

Wraps the HAL implementation from [hal9001](https://github.com/tlverse/hal9001/).
Refer to the original repository for more details on parameters.
"""
@mlj_model mutable struct HALRegressor <: MLJModelInterface.Deterministic
    X_unpenalized = nothing
    max_degree::Int = 3::(_ >= 0)
    fit_type::String = "glmnet"::(_ in ("glmnet", "lassi"))
    n_folds::Int = 10::(_ >= 0)
    use_min::Bool = true
    family::String = "gaussian"::(_ in ("gaussian", "binomial", "cox"))
    return_lasso::Bool = true
    return_x_basis::Bool = false
    lambda = nothing
    id = nothing
    offset = nothing
    cv_select::Bool = true
    yolo::Bool = true
end



################################
########### Methods ############
################################


function MLJModelInterface.fit(m::HALRegressor, verbose::Int, X, y)
    R"library(hal9001)"
    fitresult = R"""
                fit_hal(X = $X, Y = $y,
                        X_unpenalized = $(m.X_unpenalized),
                        max_degree = $(m.max_degree),
                        fit_type = $(m.fit_type),
                        n_folds = $(m.n_folds),
                        use_min = $(m.use_min),
                        family = $(m.family),
                        return_lasso = $(m.return_lasso),
                        return_x_basis = $(m.return_x_basis),
                        lambda = $(m.lambda),
                        id = $(m.id),
                        offset = $(m.offset),
                        cv_select = $(m.cv_select),
                        yolo = $(m.yolo))
                """
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
