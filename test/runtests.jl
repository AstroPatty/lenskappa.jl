using lenskappa
using Test
using CSV
using DataFrames

test_data = CSV.read("./test_data.csv", DataFrame)

@testset "LensKappa.jl" begin
    @testset "Fit" begin
        @testset "GEVFit" begin
            f = fit(test_data[!,"kappa"], 100, :GEV)
            @test isapprox(f.μ, -0.029, atol=1e-3)
            @test isapprox(f.β, -3.342, atol=1e-3)
            @test isapprox(f.ξ, 0.079, atol=1e-3)
        end

        @testset "GumbelFit" begin
            f = fit(test_data[!,"kappa"], 100, :Gumbel)
            @test isapprox(f.μ, -0.030, atol=1e-3)
            @test isapprox(f.β, -3.345, atol=1e-3)
        end

    end
    # Write your tests here.
end
