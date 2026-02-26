"""
Test the refactored separate adjustment cost system.
"""
using Test
using UncertaintyInvestment
using Random
using Statistics

@testset "Separate Adjustment Costs" begin

    # Small grid for fast tests
    params = ModelParameters(
        numerical = NumericalSettings(n_K=15, n_D=5, n_sigma=3, max_iter=100, tol_vfi=1e-4)
    )

    @testset "Old API backward compatibility" begin
        Random.seed!(12345)
        ac = ConvexAdjustmentCost(phi=2.0)
        sol = solve_model(params; ac=ac, verbose=false)
        @test sol.convergence.converged
        @test sol.ac_begin isa ConvexAdjustmentCost
        @test sol.ac_mid_year isa ConvexAdjustmentCost
        @test sol.ac_begin.phi == 2.0
        @test sol.ac_mid_year.phi == 2.0
    end

    @testset "Old API == New API when same cost" begin
        Random.seed!(12345)
        ac = ConvexAdjustmentCost(phi=2.0)
        sol_old = solve_model(params; ac=ac, verbose=false)

        Random.seed!(12345)
        sol_new = solve_model(params; ac_begin=ac, ac_mid_year=ac, verbose=false)

        @test maximum(abs.(sol_old.V .- sol_new.V)) < 1e-10
        @test maximum(abs.(sol_old.I_policy .- sol_new.I_policy)) < 1e-10
    end

    @testset "Error on conflicting API usage" begin
        ac = ConvexAdjustmentCost(phi=2.0)
        @test_throws ErrorException solve_model(params; ac=ac, ac_begin=ac, verbose=false)
    end

    @testset "Different costs at each stage" begin
        ac_begin = ConvexAdjustmentCost(phi=3.0)
        ac_mid_year = ConvexAdjustmentCost(phi=1.0)
        sol = solve_model(params; ac_begin=ac_begin, ac_mid_year=ac_mid_year, verbose=false)
        @test sol.convergence.converged
        @test sol.ac_begin.phi == 3.0
        @test sol.ac_mid_year.phi == 1.0
    end

    @testset "Mixed cost types" begin
        ac_begin = FixedAdjustmentCost(F=0.1)
        ac_mid_year = ConvexAdjustmentCost(phi=1.0)
        sol = solve_model(params; ac_begin=ac_begin, ac_mid_year=ac_mid_year, verbose=false)
        @test sol.convergence.converged
        @test sol.ac_begin isa FixedAdjustmentCost
        @test sol.ac_mid_year isa ConvexAdjustmentCost
    end

    @testset "No cost begin, cost at mid-year" begin
        ac_begin = NoAdjustmentCost()
        ac_mid_year = FixedAdjustmentCost(F=0.1)
        sol = solve_model(params; ac_begin=ac_begin, ac_mid_year=ac_mid_year, verbose=false)
        @test sol.convergence.converged
    end

    @testset "Default is NoAdjustmentCost for both" begin
        sol = solve_model(params; verbose=false)
        @test sol.convergence.converged
        @test sol.ac_begin isa NoAdjustmentCost
        @test sol.ac_mid_year isa NoAdjustmentCost
    end

    @testset "Save/load round-trip" begin
        ac_begin = FixedAdjustmentCost(F=0.1)
        ac_mid_year = ConvexAdjustmentCost(phi=2.0)
        sol = solve_model(params; ac_begin=ac_begin, ac_mid_year=ac_mid_year, verbose=false)

        mkpath("output/test")
        save_solution("output/test/separate_costs.jld2", sol)
        sol_loaded = load_solution("output/test/separate_costs.jld2")

        @test sol_loaded.ac_begin isa FixedAdjustmentCost
        @test sol_loaded.ac_mid_year isa ConvexAdjustmentCost
        @test sol_loaded.ac_begin.F == 0.1
        @test sol_loaded.ac_mid_year.phi == 2.0
        @test maximum(abs.(sol.V .- sol_loaded.V)) < 1e-10

        rm("output/test/separate_costs.jld2")
    end
end
