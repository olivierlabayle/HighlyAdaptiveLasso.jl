using HighlyAdaptiveLasso
using Test
using Random
using MLJBase
using RCall


@testset "Test HALRegressor" begin
    # Those are sanity checks as the original package is already well tested
    n = 100
    p = 3
    X = randn(n, p)
    y = X[:, 1] .* 3sin.(X[:, 2]) + randn(n)
    Xt = MLJBase.table(X)
    # With cross validation config
    model = HALRegressor(cv_select=true, n_folds=3)
    mach = machine(model, Xt, y)
    fit!(mach)
    @test mach.state == 1
    @test predict(mach, Xt) isa Vector{Float64}

    @rget n_folds
    @test n_folds == 3

    @test mach.report ==
        "fit_hal(X = X, Y = y, family='gaussian', fit_control = list(cv_select=cv_select, n_folds=n_folds))"


    # Test specifying a lambda
    model = HALRegressor(cv_select=false, lambda=30, max_degree=2, num_knots=[100, 50, 20], smoothness_orders=2)
    mach = machine(model, Xt, y)
    fit!(mach)
    @test mach.state == 1
    @test predict(mach, Xt) isa Vector{Float64}

    @rget max_degree 
    @rget num_knots
    @rget smoothness_orders
    @rget cv_select

    @test cv_select == false
    @test max_degree == 2
    @test smoothness_orders == 2
    @test num_knots == [100, 50, 20]

    @test mach.report ==
        "fit_hal(X = X, Y = y, family='gaussian', max_degree=max_degree, smoothness_orders=smoothness_orders, "*
        "num_knots=num_knots, lambda=lambda, fit_control = list(cv_select=cv_select))"

    # Test specifying a formula
    formula_ = "~h(x1) + h(x2, x1) + h(x3)"
    model = HALRegressor(formula=formula_, lambda=30, cv_select=false)
    mach = machine(model, Xt, y)
    fit!(mach)
    @test mach.state == 1
    @test predict(mach, Xt) isa Vector{Float64}

    @rget formula
    @test formula == formula_

    @test mach.report ==
        "fit_hal(X = X, Y = y, family='gaussian', formula=formula, lambda=lambda, fit_control = list(cv_select=cv_select))"

end


@testset "Test HALClassifier" begin
    # Those are sanity checks as the original package is already well tested
    n = 100
    p = 3
    X = randn(n, p)
    y = categorical(rand(["toto", "tata"], n))
    Xt = MLJBase.table(X)
    # With cross validation config
    model = HALClassifier()
    mach = machine(model, Xt, y)
    fit!(mach)
    @test mach.state == 1
    @test predict(mach, Xt) isa UnivariateFiniteVector

    @test mach.report == "fit_hal(X = X, Y = y, family='binomial')"

end