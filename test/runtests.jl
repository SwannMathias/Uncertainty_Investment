using Test
using UncertaintyInvestment
using Random

println("Running tests with $(get_nthreads()) thread(s)")
if get_nthreads() == 1
    println("  Tip: Use 'julia -t N' for parallel testing")
end

@testset "UncertaintyInvestment.jl" begin
    @testset "Parameters" begin
        # Test parameter construction
        params = ModelParameters(alpha=0.33, epsilon=4.0, delta=0.10, beta=0.96)
        @test params.alpha == 0.33
        @test params.epsilon == 4.0

        # Test derived parameters
        derived = get_derived_parameters(params)
        @test derived.gamma > 0.0
        @test derived.gamma < 1.0
        @test derived.h > 0.0
        @test derived.K_ss > 0.0

        # Test parameter validation
        @test validate_parameters(params)
    end

    @testset "Profit Function" begin
        params = ModelParameters()
        derived = get_derived_parameters(params)

        K = 1.0
        D = 1.0

        # Test profit positivity
        pi = profit(K, D, derived)
        @test pi > 0.0

        # Test MPK positivity
        mpk = marginal_product_capital(K, D, derived)
        @test mpk > 0.0

        # Test concavity
        d2pi = profit_second_derivative_K(K, D, derived)
        @test d2pi < 0.0

        # Test elasticities
        epsilon_K = profit_elasticity_K(K, D, derived)
        epsilon_D = profit_elasticity_D(K, D, derived)
        @test isapprox(epsilon_K, 1 - derived.gamma, rtol=1e-6)
        @test isapprox(epsilon_D, derived.gamma, rtol=1e-6)

        # Run full property check
        @test check_profit_properties(derived)
    end

    @testset "Adjustment Costs" begin
        # No adjustment cost
        ac = NoAdjustmentCost()
        @test compute_cost(ac, 0.1, 0.0, 1.0) == 0.0
        @test !has_fixed_cost(ac)
        @test is_differentiable(ac)

        # Convex cost
        ac = ConvexAdjustmentCost(phi=2.0)
        cost = compute_cost(ac, 0.1, 0.0, 1.0)
        @test cost > 0.0
        @test !has_fixed_cost(ac)

        # Fixed cost
        ac = FixedAdjustmentCost(F=0.1)
        @test compute_cost(ac, 0.0, 0.0, 1.0) == 0.0
        @test compute_cost(ac, 0.1, 0.0, 1.0) == 0.1
        @test has_fixed_cost(ac)

        # Composite cost
        ac = CompositeAdjustmentCost(
            FixedAdjustmentCost(F=0.1),
            ConvexAdjustmentCost(phi=1.0)
        )
        @test has_fixed_cost(ac)
    end

    @testset "Stochastic Processes" begin
        # Rouwenhorst
        n = 7
        rho = 0.9
        sigma = 0.1
        mu = 0.0

        grid, Pi = rouwenhorst(n, rho, sigma; mu=mu)
        @test length(grid) == n
        @test size(Pi) == (n, n)
        @test is_valid_probability_matrix(Pi)

        # Verify moments
        moments = verify_discretization(grid, Pi, rho, sigma; mu=mu)
        @test moments.mean_error < 0.1
        @test moments.std_error < 0.1
        @test moments.autocorr_error < 0.1

        # Tauchen
        grid_t, Pi_t = tauchen(n, rho, sigma; mu=mu)
        @test length(grid_t) == n
        @test is_valid_probability_matrix(Pi_t)
    end

    @testset "SV Discretization" begin
        demand = DemandProcess(mu_D=0.0, rho_D=0.9)
        vol = VolatilityProcess(sigma_bar=log(0.1), rho_sigma=0.95, sigma_eta=0.1)

        sv = discretize_sv_process(demand, vol, 10, 5)
        @test sv.n_D == 10
        @test sv.n_sigma == 5
        @test length(sv.D_grid) == 10
        @test length(sv.sigma_grid) == 5
        @test is_valid_probability_matrix(sv.Pi_joint)
    end

    @testset "Grid Construction" begin
        params = ModelParameters(
            numerical = NumericalSettings(n_K=50, n_D=10, n_sigma=5)
        )

        grids = construct_grids(params)
        @test grids.n_K == 50
        @test grids.n_D == 10
        @test grids.n_sigma == 5
        @test length(grids.K_grid) == 50
        @test grids.K_min < grids.K_max
        @test is_monotonic_increasing(grids.K_grid)
    end

    @testset "Model Solution (Small)" begin
        # Small grid for fast testing
        params = ModelParameters(
            numerical = NumericalSettings(
                n_K=20, n_D=7, n_sigma=3,
                max_iter=100, tol_vfi=1e-4
            )
        )

        # Solve without adjustment costs
        sol = solve_model(params; ac=NoAdjustmentCost(), verbose=false)
        @test sol.convergence.converged
        @test all(isfinite.(sol.V))
        @test all(isfinite.(sol.I_policy))

        # Check solution properties
        @test minimum(sol.V) < maximum(sol.V)  # Value varies across states

        # Solve with convex costs
        sol_ac = solve_model(params; ac=ConvexAdjustmentCost(phi=1.0), verbose=false)
        @test sol_ac.convergence.converged
        @test all(isfinite.(sol_ac.V))
    end

    @testset "Shock Simulation" begin
        demand = DemandProcess(mu_D=0.0, rho_D=0.9)
        vol = VolatilityProcess(sigma_bar=log(0.1), rho_sigma=0.95, sigma_eta=0.1)

        # Generate shock panel
        shocks = generate_shock_panel(demand, vol, 10, 20; burn_in=10)
        @test shocks.n_firms == 10
        @test shocks.T == 20
        @test size(shocks.D) == (10, 20)
        @test size(shocks.sigma) == (10, 20)
        @test all(isfinite.(shocks.D))
        @test all(isfinite.(shocks.sigma))
    end

    @testset "Numerical Utilities" begin
        # Golden section search
        f(x) = -(x - 2)^2 + 3  # Max at x=2
        x_opt, f_opt = maximize_univariate(f, 0.0, 4.0; method=:golden)
        @test isapprox(x_opt, 2.0, atol=1e-5)

        # Convergence check
        x_new = [1.0, 2.0, 3.0]
        x_old = [1.01, 2.01, 3.01]
        @test check_convergence(x_new, x_old; atol=1e-1)
        @test !check_convergence(x_new, x_old; atol=1e-5)
    end

    @testset "Parallelization (Basic)" begin
        # Basic parallel tests (full tests in test_parallelization.jl)
        @test get_nthreads() >= 1
        @test 1 <= get_threadid() <= get_nthreads()

        # Test that parallel flag is accepted
        params = ModelParameters(
            numerical = NumericalSettings(n_K=10, n_D=3, n_sigma=3, max_iter=20, tol_vfi=1e-3)
        )
        sol = solve_model(params; ac=NoAdjustmentCost(), verbose=false, use_parallel=true)
        @test sol.convergence.converged
        @test sol.convergence.threads_used >= 1
    end
end

# Include comprehensive parallelization tests
include("test_parallelization.jl")

println("\n All tests passed!")
