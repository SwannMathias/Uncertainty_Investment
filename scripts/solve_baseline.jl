"""
Solve baseline investment model with and without adjustment costs.

This script demonstrates:
1. Parameter specification
2. Model solution
3. Solution diagnostics
4. Saving results
"""

using Pkg
# Activate the project environment (locate project root from script location)
project_root = dirname(@__DIR__)
Pkg.activate(project_root)

# Install dependencies if not already installed
if !isfile(joinpath(project_root, "Manifest.toml"))
    println("Installing package dependencies...")
    Pkg.instantiate()
end

using UncertaintyInvestment
using Random
using Printf

println("="^70)
println("Uncertainty Investment Model - Baseline Solution")
println("="^70)
println("Threads available: $(get_nthreads())")
if get_nthreads() == 1
    println("  Tip: Use 'julia -t N' for N threads to enable parallelization")
end

# ============================================================================
# 1. Define Parameters
# ============================================================================

println("\n1. Setting up model parameters...")

params = ModelParameters(
    # Technology
    alpha = 0.33,        # Capital share
    epsilon = 4.0,         # Demand elasticity
    delta = 0.10,        # Annual depreciation
    beta = 0.96,        # Annual discount factor

    # Demand process (semester frequency)
    demand = DemandProcess(
        mu_D = 0.0,   # Long-run mean of log demand
        rho_D = 0.9    # Persistence
    ),

    # Volatility process (semester frequency)
    volatility = VolatilityProcess(
        sigma_bar = log(0.1),   # Long-run mean of log volatility
        rho_sigma = 0.95,     # Persistence
        sigma_eta = 0.15,     # Volatility of volatility
        rho_epsilon_eta = 0.0      # Correlation with demand shocks
    ),

    # Numerical settings
    numerical = NumericalSettings(
        n_K = 100,           # Capital grid points
        n_D = 15,            # Demand states
        n_sigma = 7,             # Volatility states
        K_min_factor = 0.1,
        K_max_factor = 3.0,
        tol_vfi = 1e-6,
        max_iter = 1000,
        howard_steps = 10    # Acceleration
    )
)

# Validate and print
validate_parameters(params)
print_parameters(params)

# ============================================================================
# 2. Solve Baseline Model (No Adjustment Costs)
# ============================================================================

println("\n2. Solving baseline model (no adjustment costs)...")

sol_baseline = solve_model(params; ac=NoAdjustmentCost(), verbose=true)

# Save solution
mkpath("output/solutions")
save_solution("output/solutions/baseline.jld2", sol_baseline)

# ============================================================================
# 3. Solve with Convex Adjustment Costs
# ============================================================================

println("\n3. Solving model with convex adjustment costs...")

ac_convex = ConvexAdjustmentCost(phi = 2.0)
sol_convex = solve_model(params; ac=ac_convex, verbose=true)

save_solution("output/solutions/convex_ac.jld2", sol_convex)

# ============================================================================
# 4. Solve with Fixed Adjustment Costs
# ============================================================================

println("\n4. Solving model with fixed adjustment costs...")

ac_fixed = FixedAdjustmentCost(F = 0.1)
sol_fixed = solve_model(params; ac=ac_fixed, verbose=true)

save_solution("output/solutions/fixed_ac.jld2", sol_fixed)

# ============================================================================
# 5. Solve with Composite Costs
# ============================================================================

println("\n5. Solving model with composite adjustment costs...")

ac_composite = CompositeAdjustmentCost(
    FixedAdjustmentCost(F = 0.05),
    ConvexAdjustmentCost(phi = 1.0)
)
sol_composite = solve_model(params; ac=ac_composite, verbose=true)

save_solution("output/solutions/composite_ac.jld2", sol_composite)

# ============================================================================
# 6. Compare Solutions
# ============================================================================

println("\n6. Comparing solutions...")

solutions = [
    ("Baseline", sol_baseline),
    ("Convex AC", sol_convex),
    ("Fixed AC", sol_fixed),
    ("Composite AC", sol_composite)
]

println("\n" * "="^70)
println("Solution Comparison")
println("="^70)
println(@sprintf("%-15s %15s %15s %15s", "Model", "Avg I/K", "Inaction %", "Avg Value"))
println("-"^70)

for (name, sol) in solutions
    diag = solution_diagnostics(sol)
    inaction_pct = diag.inaction_frequency * 100

    println(@sprintf("%-15s %15.4f %15.2f %15.2f",
                    name,
                    diag.I_rate_mean,
                    inaction_pct,
                    diag.V_mean))
end

println("="^70)

# ============================================================================
# 7. Export Results
# ============================================================================

println("\n7. Exporting results to CSV...")

for (name, sol) in solutions
    dirname = replace(lowercase(name), " " => "_")
    export_to_csv(sol, "output/solutions/$dirname/")
end

println("\n✓ All solutions complete!")
println("\nResults saved in:")
println("  - output/solutions/baseline.jld2")
println("  - output/solutions/convex_ac.jld2")
println("  - output/solutions/fixed_ac.jld2")
println("  - output/solutions/composite_ac.jld2")
println("\nCSV exports in output/solutions/*/")


