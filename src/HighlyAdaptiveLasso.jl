module HighlyAdaptiveLasso

using MLJModelInterface
using MLJModelInterface: @mlj_model
using RCall
using DataFrames


################################
########## Structure ###########
################################

DESCR = 
"""
    HALRegressor/HALClassifier(;formula=nothing, 
                 max_degree=nothing,
                 smoothness_orders=nothing,
                 num_knots=nothing,
                 reduce_basis=nothing,
                 lambda=nothing,
                 cv_select=nothing,
                 n_folds=nothing)

An interface respecting the MMI for a deterministic/probabilistic regression/classification model that wraps the 
HAL implementation from [hal9001](https://github.com/tlverse/hal9001/)m version 0.4.1.

# Arguments

Only some hyper parameters are exposed for now. Refer to the original repository for more details on possible values.

- formula: A String, eg ~h(x1) + h(x2, x1) + h(x3)
- max_degree: Integer
- smoothness_orders: Integer
- num_knots: Integer, Vector, ...
- reduce_basis: Float64
- lambda: Float64 or Vector of such if running in cross validation mode
- cv_select: Bool
- n_folds: Integer

"""

"$DESCR"
@mlj_model mutable struct HALRegressor <: MLJModelInterface.Deterministic
    formula::Union{String, Nothing} = nothing
    max_degree::Union{Int, Nothing} = nothing
    smoothness_orders::Union{Int, Nothing} = nothing
    num_knots = nothing
    reduce_basis::Union{Float64, Nothing} = nothing
    family::String = "gaussian"
    lambda = nothing

    cv_select::Union{Bool, Nothing} = nothing
    n_folds::Union{Int, Nothing} = nothing
end


"$DESCR"
@mlj_model mutable struct HALClassifier <: MLJModelInterface.Probabilistic
    formula = nothing
    max_degree::Union{Int, Nothing} = nothing
    smoothness_orders::Union{Int, Nothing} = nothing
    num_knots = nothing
    reduce_basis::Union{Float64, Nothing} = nothing
    family::String = "binomial"
    lambda = nothing

    cv_select::Union{Bool, Nothing} = nothing
    n_folds::Union{Int, Nothing} = nothing
end


HAL = Union{HALRegressor, HALClassifier}


make_fitresult(m::HAL, fitresult, y) = (fitresult, y[1])

adapt_y(m::HALRegressor, y) = y
adapt_y(m::HALClassifier, y) = int(y) .- 1

"""
    fit(m::HAL, verbosity::Int, Xt, y)

"""
function MLJModelInterface.fit(m::HAL, verbosity::Int, Xt, yt)
    R"library(hal9001)"

    X = DataFrame(Xt)
    @rput X

    y = adapt_y(m, yt)
    @rput y

    # I can't make dynamical assignement of variables work
    # So I rely on this ugly iteraction over variables
    # to build the string for RCall
    fitstring = "fit_hal(X, y, family = '$(m.family)'"
    if m.formula !== nothing
        R"formula <- $(m.formula)"
        fitstring *= ", formula = formula"
    end
    if m.max_degree !== nothing
        R"max_degree <- $(m.max_degree)"
        fitstring *= ", max_degree = max_degree"
    end
    if m.smoothness_orders !== nothing
        R"smoothness_orders <- $(m.smoothness_orders)"
        fitstring *= ", smoothness_orders = smoothness_orders"
    end
    if m.num_knots !== nothing
        R"num_knots <- $(m.num_knots)"
        fitstring *= ", num_knots = num_knots"
    end
    if m.reduce_basis !== nothing
        R"reduce_basis <- $(m.reduce_basis)"
        fitstring *= ", reduce_basis = reduce_basis"
    end
    if m.lambda !== nothing
        R"lambda <- $(m.lambda)"
        fitstring *= ", lambda = lambda"
    end
    # Fit control string
    if m.cv_select !== nothing || m.n_folds !== nothing
        fitstring *= ", fit_control = list("
        if m.cv_select !== nothing
            R"cv_select <- $(m.cv_select)"
            fitstring *= "cv_select = cv_select, "
        end
        if m.n_folds !== nothing
            R"n_folds <- $(m.n_folds)"
            fitstring *= "n_folds = n_folds, "
        end
        fitstring = fitstring[1:end - 2] * ")"
    end
    fitstring *= ")"
    # Rcall

    fitresult = reval(fitstring)
 
    cache = nothing
    report = fitstring
    return (make_fitresult(m, fitresult, yt), cache, report)
end


function MLJModelInterface.fitted_params(m::HALRegressor, fitresult)
    return rcopy(fitresult)
end


function MLJModelInterface.predict(m::HALClassifier, (fitresult, decode), Xnew)
    Xnew = DataFrame(Xnew)
    @rput Xnew
    ypred = R"predict($fitresult, new_data = $Xnew)"
    μ = rcopy(ypred)
    return UnivariateFinite(classes(decode), μ, augment=true)
end

function MLJModelInterface.predict(m::HALRegressor, (fitresult, _), Xnew)
    Xnew = DataFrame(Xnew)
    @rput Xnew
    ypred = R"predict($fitresult, new_data = $Xnew)"
    return rcopy(ypred)
end

################################
########### METADATA ###########
################################


MLJModelInterface.metadata_pkg.((HALRegressor, HALClassifier),
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
    supports_weights = false,
    descr            = DESCR,
	load_path        = "HighlyAdaptiveLasso.HALRegressor"
)


MLJModelInterface.metadata_model(HALClassifier,
    input_scitype    = MLJModelInterface.Table(MLJModelInterface.Continuous),
    target_scitype   = AbstractVector{<:Finite},
    supports_weights = false,
    descr            = DESCR,
	load_path        = "HighlyAdaptiveLasso.HALClassifier"
)

################################
########### Exports ############
################################

export HALRegressor, HALClassifier
export fit, fitted_params, predict


end
