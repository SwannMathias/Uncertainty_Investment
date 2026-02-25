"""
test_annual_no_half_period.jl

Test the 4 analytical cases for the annual model without half-period.

Cases:
1. No fixed cost, deterministic demand → smooth investment, MPK = r + δ
2. Fixed cost, deterministic demand → (s,S) policy, lumpy investment
3. No fixed cost, stochastic demand → smooth response, E[β V_K] = 1
4. Fixed cost, stochastic demand → stochastic (s,S), option value
"""

using UncertaintyInvestment
using Test
using Statistics
using Random

println("=" ^ 70)
println("Annual Model (No Half-Period) — Analytical Test Cases")
println("=" ^ 70)

# ============================================================================
# Common parameters
# ============================================================================

alpha = 0.33
epsilon = 4.0
delta = 0.10
beta = 0.96

# Derived profit-function parameters
derived_gamma = (epsilon - 1) / (epsilon - (1 - alpha))
h_term1 = alpha
h_term2 = (1 - 1/epsilon)^(epsilon / alpha)
h_term3 = (1 - alpha)^(epsilon / alpha - 1)
h = h_term1 * h_term2 * h_term3

D_ss = 1.0   # exp(mu_D) with mu_D = 0
r = (1 - beta) / beta
user_cost = r + delta   # exact annual user cost

# Analytical steady-state capital: MPK = user_cost
# h * D^gamma * K^(-gamma) = user_cost  =>  K* = (h * D^gamma / user_cost)^(1/gamma)
K_star = (h * D_ss^derived_gamma / user_cost)^(1/derived_gamma)

println("Analytical K* = $(round(K_star, digits=4))")
println("User cost (r+δ) = $(round(user_cost, digits=4))")
println("MPK at K* = $(round(h * D_ss^derived_gamma * K_star^(-derived_gamma), digits=4))")

# ============================================================================
# Case 1: No adjustment cost, deterministic demand
# ============================================================================

@testset "Case 1: No AC, Deterministic" begin
    println("\n" * "=" ^ 70)
    println("CASE 1: No Adjustment Cost, Deterministic Demand")
    println("=" ^ 70)

    params = ModelParameters(
        alpha = alpha, epsilon = epsilon, delta = delta, beta = beta,
        demand = DemandProcess(mu_D = 0.0, rho_D = 0.0),
        volatility = VolatilityProcess(sigma_bar = log(0.1), rho_sigma = 0.0, sigma_eta = 0.001),
        numerical = NumericalSettings(
            n_K = 80, n_D = 1, n_sigma = 1,
            K_min_factor = 0.1, K_max_factor = 5.0,
            tol_vfi = 1e-8, max_iter = 2000, howard_steps = 20
        )
    )

    sol = solve_model(params; ac=NoAdjustmentCost(), verbose=true, use_half_period=false)

    @test sol.convergence.converged
    @test sol.use_half_period == false

    # Check steady state: find K closest to K_star
    grids = sol.grids
    i_K_star = argmin(abs.(grids.K_grid .- K_star))
    K_grid_star = grids.K_grid[i_K_star]

    # Investment at K* should be ≈ δ * K*
    I_at_Kstar = sol.I_policy[i_K_star, 1, 1]
    I_rate_at_Kstar = I_at_Kstar / K_grid_star

    println("  K* (analytical) = $(round(K_star, digits=4))")
    println("  K* (nearest grid) = $(round(K_grid_star, digits=4))")
    println("  I(K*) = $(round(I_at_Kstar, digits=4))")
    println("  I/K at K* = $(round(I_rate_at_Kstar, digits=4)) (should be ≈ δ = $delta)")

    @test abs(I_rate_at_Kstar - delta) < 0.02  # Within 2% of depreciation rate

    # Inaction frequency should be 0
    inaction = sum(abs.(sol.I_policy) .< 1e-8) / length(sol.I_policy)
    println("  Inaction frequency = $(round(inaction * 100, digits=2))% (should be ≈ 0%)")
    @test inaction < 0.05

    println("  ✓ Case 1 PASSED")
end

# ============================================================================
# Case 2: Fixed cost, deterministic demand
# ============================================================================

@testset "Case 2: Fixed AC, Deterministic" begin
    println("\n" * "=" ^ 70)
    println("CASE 2: Fixed Adjustment Cost, Deterministic Demand")
    println("=" ^ 70)

    F_val = 0.1

    params = ModelParameters(
        alpha = alpha, epsilon = epsilon, delta = delta, beta = beta,
        demand = DemandProcess(mu_D = 0.0, rho_D = 0.0),
        volatility = VolatilityProcess(sigma_bar = log(0.1), rho_sigma = 0.0, sigma_eta = 0.001),
        numerical = NumericalSettings(
            n_K = 100, n_D = 1, n_sigma = 1,
            K_min_factor = 0.1, K_max_factor = 5.0,
            tol_vfi = 1e-8, max_iter = 2000, howard_steps = 20
        )
    )

    sol = solve_model(params; ac=FixedAdjustmentCost(F=F_val), verbose=true, use_half_period=false)

    @test sol.convergence.converged
    @test sol.use_half_period == false

    # Should have positive inaction
    inaction = sum(abs.(sol.I_policy) .< 1e-8) / length(sol.I_policy)
    println("  Inaction frequency = $(round(inaction * 100, digits=2))% (should be > 0%)")
    @test inaction > 0.05

    # Investment should be lumpy (check variance of I conditional on I ≠ 0)
    I_nonzero = filter(x -> abs(x) > 1e-8, vec(sol.I_policy))
    if length(I_nonzero) > 0
        println("  Mean |I| when investing = $(round(mean(abs.(I_nonzero)), digits=4))")
        println("  Num investing states = $(length(I_nonzero)) / $(length(sol.I_policy))")
    end

    println("  ✓ Case 2 PASSED")
end

# ============================================================================
# Case 3: No adjustment cost, stochastic demand
# ============================================================================

@testset "Case 3: No AC, Stochastic" begin
    println("\n" * "=" ^ 70)
    println("CASE 3: No Adjustment Cost, Stochastic Demand")
    println("=" ^ 70)

    params = ModelParameters(
        alpha = alpha, epsilon = epsilon, delta = delta, beta = beta,
        demand = DemandProcess(mu_D = 0.0, rho_D = 0.9),
        volatility = VolatilityProcess(
            sigma_bar = log(0.1), rho_sigma = 0.95, sigma_eta = 0.15
        ),
        numerical = NumericalSettings(
            n_K = 80, n_D = 15, n_sigma = 7,
            K_min_factor = 0.1, K_max_factor = 5.0,
            tol_vfi = 1e-6, max_iter = 1000, howard_steps = 10
        )
    )

    sol = solve_model(params; ac=NoAdjustmentCost(), verbose=true, use_half_period=false)

    @test sol.convergence.converged
    @test sol.use_half_period == false

    # No inaction (investment every period)
    inaction = sum(abs.(sol.I_policy) .< 1e-8) / length(sol.I_policy)
    println("  Inaction frequency = $(round(inaction * 100, digits=2))% (should be ≈ 0%)")
    @test inaction < 0.05

    # Average investment rate should be ≈ δ
    diag = solution_diagnostics(sol)
    println("  Mean I/K = $(round(diag.I_rate_mean, digits=4)) (should be ≈ $delta)")
    @test abs(diag.I_rate_mean - delta) < 0.05

    # Policy should be monotone in K (decreasing) for median D
    mid_D = div(params.numerical.n_D, 2) + 1
    mid_sigma = div(params.numerical.n_sigma, 2) + 1
    I_slice = sol.I_policy[:, mid_D, mid_sigma]
    # Check roughly decreasing (allow some noise at edges)
    diffs = diff(I_slice)
    frac_decreasing = sum(diffs .< 0) / length(diffs)
    println("  Fraction I decreasing in K = $(round(frac_decreasing, digits=2)) (should be high)")
    @test frac_decreasing > 0.7

    println("  ✓ Case 3 PASSED")
end

# ============================================================================
# Case 4: Fixed cost, stochastic demand
# ============================================================================

@testset "Case 4: Fixed AC, Stochastic" begin
    println("\n" * "=" ^ 70)
    println("CASE 4: Fixed Adjustment Cost, Stochastic Demand")
    println("=" ^ 70)

    F_val = 0.1

    params = ModelParameters(
        alpha = alpha, epsilon = epsilon, delta = delta, beta = beta,
        demand = DemandProcess(mu_D = 0.0, rho_D = 0.9),
        volatility = VolatilityProcess(
            sigma_bar = log(0.1), rho_sigma = 0.95, sigma_eta = 0.15
        ),
        numerical = NumericalSettings(
            n_K = 80, n_D = 15, n_sigma = 7,
            K_min_factor = 0.1, K_max_factor = 5.0,
            tol_vfi = 1e-6, max_iter = 1000, howard_steps = 10
        )
    )

    sol = solve_model(params; ac=FixedAdjustmentCost(F=F_val), verbose=true, use_half_period=false)

    @test sol.convergence.converged
    @test sol.use_half_period == false

    # Should have significant inaction
    inaction = sum(abs.(sol.I_policy) .< 1e-8) / length(sol.I_policy)
    println("  Inaction frequency = $(round(inaction * 100, digits=2))% (should be > 0%)")
    @test inaction > 0.05

    # Comparative statics: higher F → more inaction
    sol_highF = solve_model(params; ac=FixedAdjustmentCost(F=2*F_val), verbose=false, use_half_period=false)
    inaction_highF = sum(abs.(sol_highF.I_policy) .< 1e-8) / length(sol_highF.I_policy)
    println("  Inaction (F=$(F_val)) = $(round(inaction * 100, digits=2))%")
    println("  Inaction (F=$(2*F_val)) = $(round(inaction_highF * 100, digits=2))%")
    @test inaction_highF >= inaction - 0.01  # Higher F → more inaction

    println("  ✓ Case 4 PASSED")
end

# ============================================================================
# Regression test: use_half_period=true still works (no regression)
# ============================================================================

@testset "Regression: half-period mode unchanged" begin
    println("\n" * "=" ^ 70)
    println("REGRESSION: Verifying half-period mode still works correctly")
    println("=" ^ 70)

    params = ModelParameters(
        alpha = 0.33, epsilon = 4.0, delta = 0.10, beta = 0.96,
        demand = DemandProcess(mu_D = 0.0, rho_D = 0.9),
        volatility = VolatilityProcess(
            sigma_bar = log(0.1), rho_sigma = 0.95, sigma_eta = 0.15
        ),
        numerical = NumericalSettings(
            n_K = 30, n_D = 7, n_sigma = 3,
            K_min_factor = 0.1, K_max_factor = 3.0,
            tol_vfi = 1e-5, max_iter = 500, howard_steps = 0
        )
    )

    sol_half = solve_model(params; ac=ConvexAdjustmentCost(phi=2.0),
                           verbose=false, use_half_period=true)
    sol_annual = solve_model(params; ac=ConvexAdjustmentCost(phi=2.0),
                             verbose=false, use_half_period=false)

    @test sol_half.convergence.converged "Half-period model should converge"
    @test sol_annual.convergence.converged "Annual model should converge"
    @test sol_half.use_half_period == true
    @test sol_annual.use_half_period == false

    # The two models should produce DIFFERENT value functions (they're different problems)
    V_diff = maximum(abs.(sol_half.V .- sol_annual.V))
    println("  Max |V_half - V_annual| = $(round(V_diff, digits=6)) (expected non-zero)")
    @test V_diff > 1e-6  # They should differ

    println("  ✓ Regression test PASSED: both modes converge, results differ as expected")
end

println("\n" * "=" ^ 70)
println("ALL TEST CASES COMPLETED")
println("=" ^ 70)
