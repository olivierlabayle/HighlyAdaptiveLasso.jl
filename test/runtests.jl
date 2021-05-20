using HighlyAdaptiveLasso
using Test
using RCall

@testset "HighlyAdaptiveLasso.jl" begin

    @test R"library(hal9001)" isa RObject
    
end
