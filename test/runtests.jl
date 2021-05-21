using HighlyAdaptiveLasso
using Test
using Random
using MLJBase


@testset "HighlyAdaptiveLasso.jl" begin
    Random.seed!(1234)
    # Those are sanity checks as the original package is already well tested
    n = 100
    p = 3
    X = randn(n, p)
    y = X[:, 1] .* 3sin.(X[:, 2]) + randn(n)

    model = HALRegressor()
    mach = machine(model, X, y)
    fit!(mach)
    fp = fitted_params(mach)
    # Checking keys associated with fitted params
    # From the warning, some are lost, but it's only for end user inspection
    @test Set(keys(fp)) == Set([:copy_map,
                                :x_basis,
                                :lambda_star,
                                :col_lists,
                                :glmnet_lasso,
                                :call,
                                :times,
                                :hal_lasso,
                                :coefs,
                                :basis_list,
                                :family,
                                :unpenalized_covariates,
                                :reduce_basis])

    # Checking basic prediction is better than the mean
    ypred = predict(mach, X)
    mse = mean((y - ypred).^2)
    avg_mse = mean((y .- mean(y)).^2)
    @test mse < avg_mse
    
    # Checking evaluation
    res = evaluate!(mach, resampling=CV(), measure=rmse, verbosity=0)
end
