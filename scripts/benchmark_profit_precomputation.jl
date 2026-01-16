"""
benchmark_profit_precomputation.jl

Benchmark script to measure the performance improvement from profit precomputation.

This script compares:
1. Direct profit function calls vs. precomputed lookups
2. VFI timing with the optimized implementation

Usage:
    julia -t 8 scripts/benchmark_profit_precomputation.jl

Expected results:
- Profit lookup: ~50-100x faster than function call
- VFI: Measurable improvement in total solve time
"""

using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(project_root)

using UncertaintyInvestment
using Printf
using Random
using Statistics

println("="^70)
println("Profit Precomputation Benchmark")
println("="^70)
println("Threads available: $(get_nthreads())")

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
        n_K = 80,
        n_D = 15,
        n_sigma = 7,
        tol_vfi = 1e-6,
        tol_policy = 1e-6
    )
)

derived = get_derived_parameters(params)
grids = construct_grids(params)

# ============================================================================
# Benchmark 1: Profit lookup vs. function call
# ============================================================================
println("\n" * "-"^70)
println("Benchmark 1: Profit Lookup vs. Function Call")
println("-"^70)

n_iterations = 100_000

# Warmup
for _ in 1:1000
    _ = get_profit(grids, 1, 1)
    _ = profit(get_K(grids, 1), get_D(grids, 1), derived)
end

# Benchmark precomputed lookup
t_lookup = @elapsed begin
    sum_lookup = 0.0
    for _ in 1:n_iterations
        i_K = rand(1:grids.n_K)
        i_D = rand(1:grids.n_D)
        sum_lookup += get_profit(grids, i_K, i_D)
    end
end

# Benchmark function call
t_function = @elapsed begin
    sum_function = 0.0
    for _ in 1:n_iterations
        i_K = rand(1:grids.n_K)
        i_D = rand(1:grids.n_D)
        K = get_K(grids, i_K)
        D = get_D(grids, i_D)
        sum_function += profit(K, D, derived)
    end
end

speedup = t_function / t_lookup

println(@sprintf("  Iterations:     %d", n_iterations))
println(@sprintf("  Lookup time:    %.4f seconds (%.2f ns/call)", t_lookup, t_lookup / n_iterations * 1e9))
println(@sprintf("  Function time:  %.4f seconds (%.2f ns/call)", t_function, t_function / n_iterations * 1e9))
println(@sprintf("  Speedup:        %.1fx", speedup))

# ============================================================================
# Benchmark 2: VFI Solve Time
# ============================================================================
println("\n" * "-"^70)
println("Benchmark 2: VFI Solve Time")
println("-"^70)

ac = ConvexAdjustmentCost(phi = 2.0)

# Warmup run (smaller grid)
params_small = ModelParameters(
    numerical = NumericalSettings(n_K = 20, n_D = 7, n_sigma = 3, tol_vfi = 1e-4)
)
_ = solve_model(params_small; ac=ac, verbose=false, use_parallel=false)

# Serial benchmark
println("\nSerial execution:")
t_serial = @elapsed begin
    sol_serial = solve_model(params; ac=ac, verbose=false, use_parallel=false)
end
println(@sprintf("  Time:       %.2f seconds", t_serial))
println(@sprintf("  Iterations: %d", sol_serial.convergence.iterations))

# Parallel benchmark
n_threads = get_nthreads()
if n_threads > 1
    println("\nParallel execution ($n_threads threads):")
    t_parallel = @elapsed begin
        sol_parallel = solve_model(params; ac=ac, verbose=false, use_parallel=true)
    end
    println(@sprintf("  Time:       %.2f seconds", t_parallel))
    println(@sprintf("  Iterations: %d", sol_parallel.convergence.iterations))
    println(@sprintf("  Speedup:    %.2fx", t_serial / t_parallel))

    # Verify same solution
    V_diff = maximum(abs.(sol_serial.V .- sol_parallel.V))
    println(@sprintf("  V diff:     %.2e", V_diff))
else
    println("\nNote: Run with multiple threads for parallel benchmark")
    println("  julia -t 8 scripts/benchmark_profit_precomputation.jl")
end

# ============================================================================
# Benchmark 3: Multi-Scale Solver
# ============================================================================
println("\n" * "-"^70)
println("Benchmark 3: Multi-Scale Solver Comparison")
println("-"^70)

println("\nStandard VFI:")
t_standard = @elapsed begin
    sol_standard = solve_model(params; ac=ac, verbose=false, use_parallel=true, use_multiscale=false)
end
println(@sprintf("  Time:       %.2f seconds", t_standard))
println(@sprintf("  Iterations: %d", sol_standard.convergence.iterations))

println("\nMulti-Scale VFI:")
t_multiscale = @elapsed begin
    sol_multiscale = solve_model(params; ac=ac, verbose=false, use_parallel=true, use_multiscale=true)
end
println(@sprintf("  Time:       %.2f seconds", t_multiscale))
println(@sprintf("  Iterations: %d", sol_multiscale.convergence.iterations))
println(@sprintf("  Speedup:    %.2fx", t_standard / t_multiscale))

# Verify same solution
V_diff_ms = maximum(abs.(sol_standard.V .- sol_multiscale.V))
println(@sprintf("  V diff:     %.2e", V_diff_ms))

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^70)
println("Summary")
println("="^70)
println(@sprintf("  Profit lookup speedup:     %.1fx faster than function call", speedup))
println(@sprintf("  Grid size:                 %d x %d x %d = %d states",
    grids.n_K, grids.n_D, grids.n_sigma, grids.n_K * grids.n_D * grids.n_sigma))
println(@sprintf("  VFI convergence:           %d iterations", sol_serial.convergence.iterations))
println(@sprintf("  Final V distance:          %.2e", sol_serial.convergence.final_distance))
println("="^70)
