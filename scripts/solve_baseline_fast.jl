"""
Fast testing version of baseline model solver.

This script uses reduced grid sizes for quick testing and debugging.
Typical runtime: < 1 minute instead of several minutes.

# Parallelization
This script supports multi-threaded execution. To enable:
    julia -t 8 scripts/solve_baseline_fast.jl

Or set the environment variable before starting Julia:
    export JULIA_NUM_THREADS=8
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
using Printf

println("="^70)
println("Uncertainty Investment Model - FAST TESTING MODE")
println("="^70)
println("Threads available: $(get_nthreads())")
if get_nthreads() == 1
    println("  Tip: Use 'julia -t N' for N threads")
end

# ============================================================================
# 1. Define Parameters (FAST VERSION)
# ============================================================================

println("\n1. Setting up model parameters (fast testing mode)...")

params = ModelParameters(
    # Technology
    alpha = 0.33,        # Capital share
    epsilon = 4.0,       # Demand elasticity
    delta = 0.10,        # Annual depreciation
    beta = 0.96,         # Annual discount factor

    # Demand process (semester frequency)
    demand = DemandProcess(
        mu_D = 0.0,      # Long-run mean of log demand
        rho_D = 0.9      # Persistence
    ),

    # Volatility process (semester frequency)
    volatility = VolatilityProcess(
        sigma_bar = log(0.1),     # Long-run mean of log volatility
        rho_sigma = 0.95,         # Persistence
        sigma_eta = 0.15,         # Volatility of volatility
        rho_epsilon_eta = 0.0     # Correlation with demand shocks
    ),

    # Numerical settings - REDUCED FOR FAST TESTING
    numerical = NumericalSettings(
        n_K = 20,              # Capital grid points (was 100)
        n_D = 5,               # Demand states (was 15)
        n_sigma = 3,           # Volatility states (was 7)
        K_min_factor = 0.1,
        K_max_factor = 3.0,
        tol_vfi = 1e-4,        # Relaxed tolerance (was 1e-6)
        max_iter = 200,        # Fewer iterations (was 1000)
        howard_steps = 5       # Fewer acceleration steps (was 10)
    )
)

# Validate and print
validate_parameters(params)
print_parameters(params)

println("\n" * "="^70)
println("FAST TEST MODE - State space: $(params.numerical.n_K) × $(params.numerical.n_D) × $(params.numerical.n_sigma) = $(params.numerical.n_K * params.numerical.n_D * params.numerical.n_sigma)")
println("Compare to full: 100 × 15 × 7 = 10,500 states")
println("Speedup: ~$(round(10500 / (params.numerical.n_K * params.numerical.n_D * params.numerical.n_sigma), digits=1))x faster")
println("="^70)

# ============================================================================
# 2. Solve Baseline Model (No Adjustment Costs)
# ============================================================================

println("\n2. Solving baseline model (no adjustment costs)...")

sol_baseline = solve_model(params; ac=NoAdjustmentCost(), verbose=true)

# Save solution
mkpath("output/solutions_fast")
save_solution("output/solutions_fast/baseline_fast.jld2", sol_baseline)

println("\n✓ Fast test complete!")
println("\nResults saved in:")
println("  - output/solutions_fast/baseline_fast.jld2")
println("\nNote: This is a FAST TEST VERSION with reduced grid resolution.")
println("For production runs, use solve_baseline.jl with full parameters.")
