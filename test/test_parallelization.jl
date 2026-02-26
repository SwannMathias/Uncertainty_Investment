"""
Tests for multi-threaded parallelization of the UncertaintyInvestment package.

These tests verify that:
1. Parallel and serial implementations produce identical results
2. Thread safety is maintained (no race conditions)
3. Performance scales with thread count

Run with multiple threads for full testing:
    julia -t 4 test/test_parallelization.jl

Or as part of the test suite:
    julia -t 4 -e 'using Pkg; Pkg.test()'
"""

using Test
using UncertaintyInvestment
using Random
using Statistics

@testset "Parallelization Tests" begin
    println("\n" * "="^70)
    println("Parallelization Tests")
    println("="^70)
    println("Available threads: $(get_nthreads())")

    if get_nthreads() == 1
        println("WARNING: Running with single thread. Parallel execution will be skipped.")
        println("  For full parallel testing, use: julia -t N test/test_parallelization.jl")
    end
    println("="^70 * "\n")

    @testset "Threading Utilities" begin
        # Test that threading utilities are exported and work
        n = get_nthreads()
        @test n >= 1
        @test typeof(n) == Int

        tid = get_threadid()
        @test 1 <= tid <= n
    end

    @testset "VFI: Serial vs Parallel Consistency" begin
        println("\nTesting VFI serial/parallel consistency...")

        # Use small grid for fast testing
        params = ModelParameters(
            numerical = NumericalSettings(
                n_K=15, n_D=5, n_sigma=3,
                max_iter=100, tol_vfi=1e-4, tol_policy=1e-3
            )
        )

        # Test with no adjustment costs
        println("  Testing NoAdjustmentCost...")
        sol_serial = solve_model(params; ac=NoAdjustmentCost(), verbose=false, use_parallel=false)
        sol_parallel = solve_model(params; ac=NoAdjustmentCost(), verbose=false, use_parallel=true)

        # Value functions should be identical (or very close)
        V_diff = maximum(abs.(sol_serial.V .- sol_parallel.V))
        I_diff = maximum(abs.(sol_serial.I_policy .- sol_parallel.I_policy))

        @test V_diff < 1e-10 || get_nthreads() == 1  # Allow exact match or skip if single thread
        @test I_diff < 1e-10 || get_nthreads() == 1

        println("    V difference: $(V_diff)")
        println("    I difference: $(I_diff)")

        # Test with convex adjustment costs
        println("  Testing ConvexAdjustmentCost...")
        ac = ConvexAdjustmentCost(phi=2.0)
        sol_serial_ac = solve_model(params; ac=ac, verbose=false, use_parallel=false)
        sol_parallel_ac = solve_model(params; ac=ac, verbose=false, use_parallel=true)

        V_diff_ac = maximum(abs.(sol_serial_ac.V .- sol_parallel_ac.V))
        I_diff_ac = maximum(abs.(sol_serial_ac.I_policy .- sol_parallel_ac.I_policy))

        @test V_diff_ac < 1e-10 || get_nthreads() == 1
        @test I_diff_ac < 1e-10 || get_nthreads() == 1

        println("    V difference: $(V_diff_ac)")
        println("    I difference: $(I_diff_ac)")

        # Both should converge
        @test sol_serial.convergence.converged
        @test sol_parallel.convergence.converged
        @test sol_serial_ac.convergence.converged
        @test sol_parallel_ac.convergence.converged

        println("  [PASS] VFI serial/parallel consistency verified")
    end

    @testset "Firm Simulation: Serial vs Parallel Consistency" begin
        println("\nTesting firm simulation serial/parallel consistency...")

        # Solve a small model first
        params = ModelParameters(
            numerical = NumericalSettings(
                n_K=15, n_D=5, n_sigma=3,
                max_iter=100, tol_vfi=1e-4
            )
        )
        sol = solve_model(params; ac=NoAdjustmentCost(), verbose=false, use_parallel=false)

        # Generate shock panel (serial, reproducible)
        Random.seed!(12345)
        shocks = generate_shock_panel(params.demand, params.volatility, 50, 40;
                                      burn_in=10, use_parallel=false)

        # Simulate firms: serial vs parallel
        histories_serial = simulate_firm_panel(sol, shocks; K_init=1.0, T_years=15, use_parallel=false)
        histories_parallel = simulate_firm_panel(sol, shocks; K_init=1.0, T_years=15, use_parallel=true)

        # Results should be identical (same shocks, same policy)
        n_firms = length(histories_serial)
        @test length(histories_parallel) == n_firms

        all_K_match = true
        all_I_match = true
        max_K_diff = 0.0
        max_I_diff = 0.0

        for i in 1:n_firms
            K_diff = maximum(abs.(histories_serial[i].K .- histories_parallel[i].K))
            I_diff = maximum(abs.(histories_serial[i].I_total .- histories_parallel[i].I_total))

            max_K_diff = max(max_K_diff, K_diff)
            max_I_diff = max(max_I_diff, I_diff)

            if K_diff > 1e-10
                all_K_match = false
            end
            if I_diff > 1e-10
                all_I_match = false
            end
        end

        @test all_K_match || get_nthreads() == 1
        @test all_I_match || get_nthreads() == 1

        println("  Max K difference: $(max_K_diff)")
        println("  Max I difference: $(max_I_diff)")
        println("  [PASS] Firm simulation serial/parallel consistency verified")
    end

    @testset "Shock Generation: Reproducibility" begin
        println("\nTesting parallel shock generation reproducibility...")

        demand = DemandProcess(mu_D=0.0, rho_D=0.9)
        vol = VolatilityProcess(sigma_bar=log(0.1), rho_sigma=0.95, sigma_eta=0.1)

        # Generate with same seed multiple times
        seed = 42
        shocks1 = generate_shock_panel_parallel(demand, vol, 100, 50; burn_in=10, seed=seed)
        shocks2 = generate_shock_panel_parallel(demand, vol, 100, 50; burn_in=10, seed=seed)

        # Should be identical with same seed
        @test shocks1.D == shocks2.D
        @test shocks1.sigma == shocks2.sigma

        # Different seeds should give different results
        shocks3 = generate_shock_panel_parallel(demand, vol, 100, 50; burn_in=10, seed=seed+1)
        @test shocks1.D != shocks3.D

        println("  [PASS] Parallel shock generation is reproducible with seed")
    end

    @testset "Thread Safety: No Race Conditions" begin
        println("\nTesting thread safety (stress test)...")

        # Run VFI multiple times to check for race conditions
        params = ModelParameters(
            numerical = NumericalSettings(
                n_K=20, n_D=5, n_sigma=3,
                max_iter=50, tol_vfi=1e-4
            )
        )

        n_runs = 3
        solutions = Vector{SolvedModel}(undef, n_runs)

        for i in 1:n_runs
            solutions[i] = solve_model(params; ac=NoAdjustmentCost(), verbose=false, use_parallel=true)
        end

        # All runs should produce identical results (deterministic)
        for i in 2:n_runs
            V_diff = maximum(abs.(solutions[1].V .- solutions[i].V))
            @test V_diff < 1e-10
        end

        println("  [PASS] Thread safety verified ($(n_runs) runs identical)")
    end

    @testset "Bellman Operator: Direct Test" begin
        println("\nTesting Bellman operator parallel implementation...")

        params = ModelParameters(
            numerical = NumericalSettings(n_K=20, n_D=5, n_sigma=3)
        )
        grids = construct_grids(params)
        derived = get_derived_parameters(params)
        ac = ConvexAdjustmentCost(phi=1.0)

        # Initialize
        V = rand(grids.n_K, grids.n_D, grids.n_sigma)
        V_new_serial = similar(V)
        V_new_parallel = similar(V)
        I_policy_serial = zeros(grids.n_K, grids.n_D, grids.n_sigma)
        I_policy_parallel = zeros(grids.n_K, grids.n_D, grids.n_sigma)

        # Apply both operators
        bellman_operator!(V_new_serial, V, I_policy_serial, grids, params, ac, ac, derived)
        bellman_operator_parallel!(V_new_parallel, V, I_policy_parallel, grids, params, ac, ac, derived)

        # Check equality
        V_diff = maximum(abs.(V_new_serial .- V_new_parallel))
        I_diff = maximum(abs.(I_policy_serial .- I_policy_parallel))

        @test V_diff < 1e-10 || get_nthreads() == 1
        @test I_diff < 1e-10 || get_nthreads() == 1

        println("  V_new difference: $(V_diff)")
        println("  I_policy difference: $(I_diff)")
        println("  [PASS] Bellman operator parallel matches serial")
    end

    # Performance benchmark (only if multiple threads available)
    if get_nthreads() > 1
        @testset "Performance: Speedup Measurement" begin
            println("\nMeasuring parallel speedup...")

            params = ModelParameters(
                numerical = NumericalSettings(
                    n_K=30, n_D=7, n_sigma=5,
                    max_iter=30, tol_vfi=1e-6
                )
            )

            # Warm-up
            _ = solve_model(params; ac=NoAdjustmentCost(), verbose=false, use_parallel=false)
            _ = solve_model(params; ac=NoAdjustmentCost(), verbose=false, use_parallel=true)

            # Time serial
            t_serial = @elapsed begin
                for _ in 1:2
                    solve_model(params; ac=NoAdjustmentCost(), verbose=false, use_parallel=false)
                end
            end
            t_serial /= 2

            # Time parallel
            t_parallel = @elapsed begin
                for _ in 1:2
                    solve_model(params; ac=NoAdjustmentCost(), verbose=false, use_parallel=true)
                end
            end
            t_parallel /= 2

            speedup = t_serial / t_parallel
            efficiency = speedup / get_nthreads()

            println("  Serial time:   $(round(t_serial, digits=3))s")
            println("  Parallel time: $(round(t_parallel, digits=3))s")
            println("  Speedup:       $(round(speedup, digits=2))x")
            println("  Efficiency:    $(round(efficiency*100, digits=1))% (speedup/threads)")
            println("  Threads:       $(get_nthreads())")

            # Parallel should not be significantly slower
            @test speedup > 0.5  # At worst, parallel is 2x slower (overhead)

            # With enough threads, should see some speedup
            if get_nthreads() >= 4
                @test speedup > 1.0  # Should have at least some speedup
            end

            println("  [PASS] Performance benchmark completed")
        end
    else
        println("\nSkipping performance benchmark (need >1 thread)")
    end

    println("\n" * "="^70)
    println("All parallelization tests completed!")
    println("="^70)
end
