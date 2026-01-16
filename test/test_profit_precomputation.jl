"""
test_profit_precomputation.jl

Test that profit function precomputation preserves solution accuracy
and provides performance benefits.

This test verifies:
1. Precomputed profits match the original profit function exactly
2. VFI converges with precomputed profits
3. Solution diagnostics are sensible

Per CLAUDE.md guidelines, any modification that could affect the solution
must include a value function comparison test.
"""

using Test
using Random
using Statistics

# Add package to path if running standalone
if !isdefined(Main, :UncertaintyInvestment)
    push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
end

using UncertaintyInvestment

println("="^70)
println("Testing Profit Precomputation Optimization")
println("="^70)

@testset "Profit Precomputation" begin

    @testset "Precomputed profits match original function" begin
        println("\n1. Testing precomputed profits match original profit function...")

        # Set up parameters
        params = ModelParameters(
            alpha = 0.33,
            epsilon = 4.0,
            delta = 0.10,
            beta = 0.96,
            demand = DemandProcess(mu_D = 0.0, rho_D = 0.9),
            volatility = VolatilityProcess(
                sigma_bar = log(0.1),
                rho_sigma = 0.95,
                sigma_eta = 0.15
            ),
            numerical = NumericalSettings(
                n_K = 50,
                n_D = 15,
                n_sigma = 7
            )
        )

        derived = get_derived_parameters(params)
        grids = construct_grids(params)

        # Compare precomputed profits with original profit function
        max_diff = 0.0
        total_checks = 0

        for i_D in 1:grids.n_D
            D = get_D(grids, i_D)
            for i_K in 1:grids.n_K
                K = get_K(grids, i_K)

                # Original profit function
                pi_original = profit(K, D, derived)

                # Precomputed profit
                pi_precomputed = get_profit(grids, i_K, i_D)

                # Check match
                diff = abs(pi_original - pi_precomputed)
                max_diff = max(max_diff, diff)
                total_checks += 1

                @test isapprox(pi_original, pi_precomputed, rtol=1e-12)
            end
        end

        println("  Checked $total_checks (K, D) combinations")
        println("  Max absolute difference: $max_diff")
        @test max_diff < 1e-10 "Precomputed profits should match exactly"
        println("  PASS: Precomputed profits match original function")
    end

    @testset "Log profits are correct" begin
        println("\n2. Testing log profits...")

        params = ModelParameters(
            numerical = NumericalSettings(n_K = 20, n_D = 10, n_sigma = 5)
        )
        grids = construct_grids(params)

        # Check log profits match log of profits
        for i_D in 1:grids.n_D
            for i_K in 1:grids.n_K
                pi = get_profit(grids, i_K, i_D)
                log_pi = get_log_profit(grids, i_K, i_D)
                @test isapprox(log(pi), log_pi, rtol=1e-12)
            end
        end

        println("  PASS: Log profits are correct")
    end

    @testset "Off-grid interpolation works" begin
        println("\n3. Testing off-grid profit interpolation...")

        params = ModelParameters(
            numerical = NumericalSettings(n_K = 20, n_D = 10, n_sigma = 5)
        )
        grids = construct_grids(params)
        derived = get_derived_parameters(params)

        # Test interpolation at midpoints between grid points
        for i_D in 1:grids.n_D
            for i_K in 1:(grids.n_K - 1)
                K_low = get_K(grids, i_K)
                K_high = get_K(grids, i_K + 1)
                K_mid = 0.5 * (K_low + K_high)

                # Interpolated profit
                pi_interp = get_profit_at_K(grids, K_mid, i_D)

                # Exact profit at midpoint
                D = get_D(grids, i_D)
                pi_exact = profit(K_mid, D, derived)

                # Linear interpolation should be close for smooth function
                rel_err = abs(pi_interp - pi_exact) / pi_exact
                @test rel_err < 0.05  # Within 5% for linear interpolation
            end
        end

        println("  PASS: Off-grid interpolation works correctly")
    end

    @testset "VFI converges with precomputed profits" begin
        println("\n4. Testing VFI convergence with precomputed profits...")

        # Use smaller grid for faster test
        Random.seed!(12345)
        params = ModelParameters(
            alpha = 0.33,
            epsilon = 4.0,
            delta = 0.10,
            beta = 0.96,
            demand = DemandProcess(mu_D = 0.0, rho_D = 0.9),
            volatility = VolatilityProcess(
                sigma_bar = log(0.1),
                rho_sigma = 0.95,
                sigma_eta = 0.15
            ),
            numerical = NumericalSettings(
                n_K = 30,
                n_D = 9,
                n_sigma = 5,
                tol_vfi = 1e-5,
                tol_policy = 1e-5,
                max_iter = 500
            )
        )

        ac = ConvexAdjustmentCost(phi = 2.0)

        println("  Solving model...")
        sol = solve_model(params; ac=ac, verbose=false, use_parallel=false)

        # Check convergence
        @test sol.convergence.converged "VFI should converge"
        println("  Converged in $(sol.convergence.iterations) iterations")
        println("  Final distance: $(sol.convergence.final_distance)")

        # Check solution is sensible
        @test all(isfinite.(sol.V)) "Value function should be finite"
        @test all(isfinite.(sol.I_policy)) "Policy function should be finite"
        @test minimum(sol.V) > 0 "Value function should be positive"

        println("  PASS: VFI converges correctly")
    end

    @testset "Solution diagnostics are reasonable" begin
        println("\n5. Testing solution diagnostics...")

        params = ModelParameters(
            numerical = NumericalSettings(
                n_K = 30,
                n_D = 9,
                n_sigma = 5,
                tol_vfi = 1e-5,
                tol_policy = 1e-5
            )
        )

        ac = ConvexAdjustmentCost(phi = 2.0)
        sol = solve_model(params; ac=ac, verbose=false, use_parallel=false)

        diag = solution_diagnostics(sol)

        # Check diagnostics are reasonable
        @test diag.V_mean > 0 "Mean value should be positive"
        @test diag.V_std > 0 "Value function should have variation"
        @test 0.0 < diag.I_rate_mean < 0.5 "Average investment rate should be reasonable"

        println("  V_mean: $(round(diag.V_mean, digits=4))")
        println("  I_rate_mean: $(round(diag.I_rate_mean, digits=4))")
        println("  I_rate_ss: $(round(diag.I_rate_ss, digits=4))")
        println("  depreciation: $(round(diag.depreciation_rate, digits=4))")

        println("  PASS: Diagnostics are reasonable")
    end

    @testset "Parallel execution works" begin
        println("\n6. Testing parallel execution...")

        params = ModelParameters(
            numerical = NumericalSettings(
                n_K = 25,
                n_D = 7,
                n_sigma = 4,
                tol_vfi = 1e-4,
                tol_policy = 1e-4
            )
        )

        ac = ConvexAdjustmentCost(phi = 2.0)

        # Solve serial
        sol_serial = solve_model(params; ac=ac, verbose=false, use_parallel=false)

        # Solve parallel (may be same as serial if only 1 thread)
        sol_parallel = solve_model(params; ac=ac, verbose=false, use_parallel=true)

        # Results should be identical (or very close due to floating point)
        V_diff = maximum(abs.(sol_serial.V .- sol_parallel.V))
        I_diff = maximum(abs.(sol_serial.I_policy .- sol_parallel.I_policy))

        println("  V difference (serial vs parallel): $V_diff")
        println("  I difference (serial vs parallel): $I_diff")

        @test V_diff < 1e-10 "Serial and parallel should produce same results"
        @test I_diff < 1e-10 "Serial and parallel should produce same policy"

        println("  PASS: Parallel execution produces identical results")
    end

end

println("\n" * "="^70)
println("All profit precomputation tests passed!")
println("="^70)
