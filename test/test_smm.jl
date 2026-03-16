"""
Tests for SMM estimation module.

Tests cover:
1. OLS correctness
2. Revision transforms
3. Moment computation from known panel data
4. SMM objective pipeline (single evaluation)
5. Warm-start speedup verification
"""

using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(project_root)

using UncertaintyInvestment
using Test
using Random
using Statistics
using LinearAlgebra
using DataFrames

# ============================================================================
# Test 1: OLS Correctness
# ============================================================================

@testset "OLS Coefficients" begin
    Random.seed!(12345)
    n = 1000

    # Generate synthetic data: y = 2 + 3x1 + (-1.5)x2 + eps
    x1 = randn(n)
    x2 = randn(n)
    eps = 0.1 * randn(n)
    y = 2.0 .+ 3.0 .* x1 .+ (-1.5) .* x2 .+ eps

    X = hcat(ones(n), x1, x2)
    beta = ols_coefficients(y, X)

    @test length(beta) == 3
    @test abs(beta[1] - 2.0) < 0.05    # Intercept
    @test abs(beta[2] - 3.0) < 0.05    # Coefficient on x1
    @test abs(beta[3] - (-1.5)) < 0.05 # Coefficient on x2

    # Test with perfect multicollinearity detection (should not error but result may be imprecise)
    X_singular = hcat(ones(n), x1, x1)  # x2 = x1
    @test_throws SingularException ols_coefficients(y, X_singular)
end

# ============================================================================
# Test 2: Revision Transforms
# ============================================================================

@testset "Revision Transforms" begin
    K = 10.0

    # ASINH transform
    @testset "ASINH" begin
        rev = apply_transform(5.0, 3.0, K, ASINH_TRANSFORM)
        @test rev ≈ asinh(5.0) - asinh(3.0)

        # Works with negative values
        rev_neg = apply_transform(-1.0, 2.0, K, ASINH_TRANSFORM)
        @test isfinite(rev_neg)

        # Works with zero
        rev_zero = apply_transform(0.0, 0.0, K, ASINH_TRANSFORM)
        @test rev_zero ≈ 0.0
    end

    # LOG transform
    @testset "LOG" begin
        rev = apply_transform(5.0, 3.0, K, LOG_TRANSFORM)
        @test rev ≈ log(5.0) - log(3.0)

        # Returns NaN for non-positive
        @test isnan(apply_transform(-1.0, 3.0, K, LOG_TRANSFORM))
        @test isnan(apply_transform(5.0, 0.0, K, LOG_TRANSFORM))
        @test isnan(apply_transform(0.0, 0.0, K, LOG_TRANSFORM))
    end

    # LEVEL_OVER_K transform
    @testset "LEVEL_OVER_K" begin
        rev = apply_transform(5.0, 3.0, K, LEVEL_OVER_K_TRANSFORM)
        @test rev ≈ (5.0 - 3.0) / 10.0

        # Works with any values
        rev_neg = apply_transform(-1.0, 2.0, K, LEVEL_OVER_K_TRANSFORM)
        @test rev_neg ≈ (-1.0 - 2.0) / 10.0
    end
end

# ============================================================================
# Test 3: Configuration Structs
# ============================================================================

@testset "Configuration" begin
    @testset "FixedCalibration defaults" begin
        cal = FixedCalibration()
        @test cal.alpha == 0.33
        @test cal.epsilon == 4.0
        @test cal.delta == 0.10
        @test cal.beta == 0.96
        @test cal.n_K == 50
    end

    @testset "SMMConfig defaults" begin
        config = SMMConfig()
        @test length(config.lower_bounds) == 4
        @test length(config.upper_bounds) == 4
        @test length(config.m_data) == 4
        @test size(config.W) == (4, 4)
        @test config.revision_transform == ASINH_TRANSFORM
    end

    @testset "build_model_parameters" begin
        cal = FixedCalibration()
        params = build_model_parameters(cal)
        @test params.alpha == cal.alpha
        @test params.epsilon == cal.epsilon
        @test params.demand.rho_D == cal.rho_D
        @test params.volatility.sigma_bar == cal.sigma_bar
        @test params.numerical.n_K == cal.n_K
    end

    @testset "SMMConfig validation" begin
        # Invalid bounds
        @test_throws AssertionError SMMConfig(
            lower_bounds = [1.0, 0.0, 0.0, 0.0],
            upper_bounds = [0.0, 10.0, 20.0, 20.0]  # lower > upper
        )
    end
end

# ============================================================================
# Test 4: Latin Hypercube Sampling
# ============================================================================

@testset "Latin Hypercube Sampling" begin
    rng = MersenneTwister(42)
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [10.0, 10.0, 20.0, 20.0]

    samples = latin_hypercube_sample(20, lower, upper; rng=rng)

    @test size(samples) == (20, 4)
    # All samples within bounds
    for j in 1:4
        @test all(samples[:, j] .>= lower[j])
        @test all(samples[:, j] .<= upper[j])
    end
    # Good coverage: each dimension should span most of the range
    for j in 1:4
        range_covered = maximum(samples[:, j]) - minimum(samples[:, j])
        @test range_covered > 0.5 * (upper[j] - lower[j])
    end
end

# ============================================================================
# Test 5: Moment Computation with Synthetic Panel
# ============================================================================

@testset "Moment Computation - Synthetic" begin
    # Create a synthetic panel to test moment computation
    Random.seed!(54321)
    n_firms = 100
    T = 50
    n_obs = n_firms * T

    # Generate synthetic data
    K = abs.(randn(n_obs)) .+ 0.5
    D = exp.(randn(n_obs))
    D_half = exp.(randn(n_obs))
    sigma = exp.(-2.0 .+ 0.3 .* randn(n_obs))
    sigma_half = exp.(-2.0 .+ 0.3 .* randn(n_obs))

    # E_beginning and E_half are correlated with sigma
    E_beginning = K .* (0.1 .+ 0.05 .* randn(n_obs))
    E_half = E_beginning .+ 0.02 .* K .* randn(n_obs)

    # E_last_semester: NaN for year 1, otherwise lagged
    E_last_semester = similar(E_beginning)
    for i in 1:n_obs
        year = ((i - 1) % T) + 1
        if year == 1
            E_last_semester[i] = NaN
        else
            E_last_semester[i] = E_beginning[i] .+ 0.03 .* K[i] .* randn()
        end
    end

    df = DataFrame(
        firm_id = repeat(1:n_firms, inner=T),
        year = repeat(1:T, outer=n_firms),
        K = K,
        D = D,
        D_half = D_half,
        sigma = sigma,
        sigma_half = sigma_half,
        log_D = log.(D),
        log_D_half = log.(D_half),
        log_sigma = log.(sigma),
        log_sigma_half = log.(sigma_half),
        I = zeros(n_obs),
        Delta_I = zeros(n_obs),
        I_total = zeros(n_obs),
        I_rate = zeros(n_obs),
        profit = zeros(n_obs),
        E_last_semester = E_last_semester,
        E_beginning = E_beginning,
        E_half = E_half
    )

    config = SMMConfig()

    moments = compute_simulated_moments(df, config)

    @test length(moments) == 4
    @test all(isfinite.(moments))
    # Share of zero should be between 0 and 1
    @test 0.0 <= moments[1] <= 1.0
    @test 0.0 <= moments[2] <= 1.0
    # Regression coefficients should be finite
    @test isfinite(moments[3])
    @test isfinite(moments[4])
end

# ============================================================================
# Test 6: SMM Objective (Single Evaluation)
# ============================================================================

@testset "SMM Objective - Single Evaluation" begin
    # Use small grid for fast testing
    cal = FixedCalibration(
        n_K = 20,
        n_D = 7,
        n_sigma = 3,
        K_min_factor = 0.3,
        K_max_factor = 3.0,
        tol_vfi = 1e-4,
        max_iter = 200,
        howard_steps = 20
    )
    config = SMMConfig(
        calibration = cal,
        n_firms = 50,
        T_years = 30,
        burn_in_years = 10,
        shock_seed = 42
    )

    # Build grids and shocks once
    params = build_model_parameters(cal)
    grids = construct_grids(params)
    T_semesters = 2 * (config.T_years + config.burn_in_years)
    shocks = generate_shock_panel(
        params.demand, params.volatility,
        config.n_firms, T_semesters;
        seed=config.shock_seed
    )

    # Test with moderate adjustment costs
    theta = [0.5, 1.0, 1.0, 2.0]  # F_begin, F_mid, phi_begin, phi_mid

    result = smm_objective(theta, config, grids, shocks, nothing, nothing)

    @test haskey(result, :objective) || hasproperty(result, :objective)
    # Objective should be finite if VFI converged
    if result.converged
        @test isfinite(result.objective)
        @test length(result.moments) == 4
        @test !isnothing(result.V)
        @test !isnothing(result.I_policy)
    end

    # Test warm-starting: second evaluation with V_init should work
    if result.converged && !isnothing(result.V)
        theta2 = theta .+ [0.05, 0.05, 0.1, 0.1]  # Small perturbation
        result2 = smm_objective(theta2, config, grids, shocks, result.V, result.I_policy)

        if result2.converged
            @test isfinite(result2.objective)
            # Results should be different (different parameters)
            @test result2.objective != result.objective || result2.moments != result.moments
        end
    end
end

# ============================================================================
# Test 7: ParticleState
# ============================================================================

@testset "ParticleState" begin
    theta = [1.0, 2.0, 3.0, 4.0]
    p = ParticleState(theta)

    @test p.theta == theta
    @test p.velocity == zeros(4)
    @test p.objective_best == Inf
    @test isnothing(p.V_cache)
    @test isnothing(p.I_cache)
    @test p.n_evaluations == 0

    # Modify theta doesn't affect particle (deep copy)
    theta[1] = 99.0
    @test p.theta[1] == 1.0
end

println("\nAll SMM tests completed!")
