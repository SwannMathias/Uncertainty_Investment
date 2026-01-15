"""
Benchmark multi-scale grid refinement against standard VFI.

This script:
1. Solves model with both methods
2. Simulates firm panel with each solution
3. Saves results to CSV
4. Reports performance comparison
"""

using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(project_root)

using UncertaintyInvestment
using Random
using Printf
using CSV
using DataFrames
using Statistics

println("="^70)
println("Multi-Scale Grid Refinement Benchmark")
println("="^70)
println("Threads available: $(get_nthreads())")

# ============================================================================
# Parameters
# ============================================================================

println("\n1. Setting up parameters...")

params = ModelParameters(
    # Technology
    alpha = 0.33,
    epsilon = 4.0,
    delta = 0.10,
    beta = 0.96,

    # Demand process
    demand = DemandProcess(
        mu_D = 0.0,
        rho_D = 0.9
    ),

    # Volatility process
    volatility = VolatilityProcess(
        sigma_bar = log(0.1),
        rho_sigma = 0.95,
        sigma_eta = 0.15,
        rho_epsilon_eta = 0.0
    ),

    # Numerical settings - use moderately fine grid for clear comparison
    numerical = NumericalSettings(
        n_K = 80,
        n_D = 12,
        n_sigma = 6,
        K_min_factor = 0.1,
        K_max_factor = 3.0,
        tol_vfi = 1e-6,
        max_iter = 1000,
        howard_steps = 10
    )
)

validate_parameters(params)
print_parameters(params)

# No adjustment costs for clearest comparison
ac = NoAdjustmentCost()

println("\n" * "="^70)
println("Grid size: $(params.numerical.n_K) × $(params.numerical.n_D) × $(params.numerical.n_sigma) = $(params.numerical.n_K * params.numerical.n_D * params.numerical.n_sigma) states")
println("Adjustment costs: $(describe_adjustment_cost(ac))")
println("="^70)

# ============================================================================
# Method 1: Standard VFI (Cold Start)
# ============================================================================

println("\n" * "="^70)
println("METHOD 1: Standard VFI (Cold Start)")
println("="^70)

t_start_standard = time()
sol_standard = solve_model(params; ac=ac, verbose=true, use_parallel=true, use_multiscale=false)
t_standard = time() - t_start_standard

println("\n✓ Standard VFI complete")
println("  Time: $(format_time(t_standard))")
println("  Iterations: $(sol_standard.convergence.iterations)")

# Save solution
mkpath("output/benchmark")
save_solution("output/benchmark/solution_standard.jld2", sol_standard)

# ============================================================================
# Method 2: Multi-Scale VFI
# ============================================================================

println("\n" * "="^70)
println("METHOD 2: Multi-Scale VFI")
println("="^70)

t_start_multiscale = time()
sol_multiscale = solve_model(params; ac=ac, verbose=true, use_parallel=true, use_multiscale=true)
t_multiscale = time() - t_start_multiscale

println("\n✓ Multi-scale VFI complete")
println("  Time: $(format_time(t_multiscale))")
println("  Iterations: $(sol_multiscale.convergence.iterations)")

# Save solution
save_solution("output/benchmark/solution_multiscale.jld2", sol_multiscale)

# ============================================================================
# Compare Solutions
# ============================================================================

println("\n" * "="^70)
println("SOLUTION COMPARISON")
println("="^70)

# Value function difference
V_diff = maximum(abs.(sol_standard.V .- sol_multiscale.V))
V_mean = mean(abs.(sol_standard.V))
V_rel_diff = V_diff / V_mean * 100

println("\nValue Function:")
println("  Max absolute difference: $(format_number(V_diff))")
println("  Relative difference: $(format_number(V_rel_diff, digits=4))%")

# Policy function difference
I_diff = maximum(abs.(sol_standard.I_policy .- sol_multiscale.I_policy))
I_mean = mean(abs.(sol_standard.I_policy))
I_rel_diff = I_mean > 0 ? I_diff / I_mean * 100 : 0.0

println("\nPolicy Function:")
println("  Max absolute difference: $(format_number(I_diff))")
if I_mean > 0
    println("  Relative difference: $(format_number(I_rel_diff, digits=4))%")
end

# Solutions should be nearly identical
if V_diff < 1e-3 && I_diff < 1e-3
    println("\n✓ Solutions are equivalent (differences < 1e-3)")
else
    @warn "Solutions differ more than expected"
end

# ============================================================================
# Performance Summary
# ============================================================================

println("\n" * "="^70)
println("PERFORMANCE SUMMARY")
println("="^70)

speedup = t_standard / t_multiscale
time_saved = t_standard - t_multiscale

println(@sprintf("%-25s %15s %15s", "Method", "Time", "Iterations"))
println("-"^70)
println(@sprintf("%-25s %15s %15d", "Standard VFI", format_time(t_standard), sol_standard.convergence.iterations))
println(@sprintf("%-25s %15s %15d", "Multi-Scale VFI", format_time(t_multiscale), sol_multiscale.convergence.iterations))
println("-"^70)
println(@sprintf("%-25s %15s", "Speedup", "$(round(speedup, digits=2))x"))
println(@sprintf("%-25s %15s", "Time Saved", format_time(time_saved)))
println("="^70)

# ============================================================================
# Simulate Firm Panels
# ============================================================================

println("\n" * "="^70)
println("SIMULATING FIRM PANELS")
println("="^70)

# Generate common shock panel
Random.seed!(12345)
n_firms = 500
T_years = 30

println("\nGenerating shocks for $n_firms firms over $T_years years...")
shocks = generate_shock_panel(
    params.demand,
    params.volatility,
    n_firms,
    2 * T_years;  # Convert years to semesters
    burn_in = 20,
    use_parallel = true,
    seed = 12345
)

print_shock_statistics(shocks)

# Simulate with standard solution
println("\nSimulating firms with STANDARD solution...")
t_start_sim1 = time()
histories_standard = simulate_firm_panel(
    sol_standard,
    shocks;
    K_init = 1.0,
    T_years = T_years,
    use_parallel = true,
    verbose = false
)
t_sim1 = time() - t_start_sim1
println("  Simulation time: $(format_time(t_sim1))")

# Simulate with multiscale solution
println("\nSimulating firms with MULTISCALE solution...")
t_start_sim2 = time()
histories_multiscale = simulate_firm_panel(
    sol_multiscale,
    shocks;
    K_init = 1.0,
    T_years = T_years,
    use_parallel = true,
    verbose = false
)
t_sim2 = time() - t_start_sim2
println("  Simulation time: $(format_time(t_sim2))")

# Construct panels
panel_standard = construct_estimation_panel(histories_standard)
panel_multiscale = construct_estimation_panel(histories_multiscale)

println("\nPanel summaries:")
print_panel_summary(panel_standard)

# ============================================================================
# Save Results
# ============================================================================

println("\n" * "="^70)
println("SAVING RESULTS")
println("="^70)

# Save panels
CSV.write("output/benchmark/panel_standard.csv", panel_standard.df)
CSV.write("output/benchmark/panel_multiscale.csv", panel_multiscale.df)

println("✓ Panels saved:")
println("  - output/benchmark/panel_standard.csv")
println("  - output/benchmark/panel_multiscale.csv")

# Save summary statistics
summary_df = DataFrame(
    method = ["Standard", "Multiscale"],
    vfi_time = [t_standard, t_multiscale],
    vfi_iterations = [sol_standard.convergence.iterations, sol_multiscale.convergence.iterations],
    sim_time = [t_sim1, t_sim2],
    V_max_diff = [V_diff, V_diff],
    I_max_diff = [I_diff, I_diff],
    speedup = [1.0, speedup]
)

CSV.write("output/benchmark/summary_statistics.csv", summary_df)
println("  - output/benchmark/summary_statistics.csv")

# ============================================================================
# Final Report
# ============================================================================

println("\n" * "="^70)
println("BENCHMARK COMPLETE")
println("="^70)
println("\nKey Results:")
println("  • Multi-scale is $(round(speedup, digits=2))x faster")
println("  • Solutions differ by < $(format_number(max(V_diff, I_diff)))")
println("  • Time saved: $(format_time(time_saved))")
println("\nNext steps:")
println("  1. Run: python scripts/compare_panels.py")
println("  2. This will generate comparison plots and statistics")
println("="^70)
