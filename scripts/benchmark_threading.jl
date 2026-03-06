"""
    benchmark_threading.jl

Benchmark parallel scaling of UncertaintyInvestment.jl components.

Run with different thread counts to measure scaling:
    julia -t 1  scripts/benchmark_threading.jl
    julia -t 2  scripts/benchmark_threading.jl
    julia -t 4  scripts/benchmark_threading.jl
    julia -t 8  scripts/benchmark_threading.jl

Results are saved to output/benchmarks/benchmark_results_<nthreads>t.csv
"""

using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(project_root)

using UncertaintyInvestment
using Random
using Printf
using DataFrames
using CSV

# ============================================================================
# Setup
# ============================================================================

n_threads = get_nthreads()
println("="^60)
println("Threading Benchmark — $n_threads thread(s)")
println("="^60)

outdir = joinpath(project_root, "output", "benchmarks")
mkpath(outdir)

# ============================================================================
# Problem definition
# ============================================================================

Random.seed!(12345)

params = ModelParameters(
    alpha = 0.33,
    epsilon = 4.0,
    delta = 0.10,
    beta = 0.96,
    demand = DemandProcess(mu_D = log(500), rho_D = 0.5),
    volatility = VolatilityProcess(
        sigma_bar = log(0.1),
        rho_sigma = 0.00001,
        sigma_eta = 0.00001,
        rho_epsilon_eta = 0.0
    ),
    numerical = NumericalSettings(
        n_K = 50,
        n_D = 50,
        n_sigma = 5,
        K_min_factor = 0.1,
        K_max_factor = 10.0,
        tol_vfi = 1e-4,
        max_iter = 200,
        howard_steps = 0
    )
)

ac_begin = FixedAdjustmentCost(F = 0.5)
ac_mid_year = FixedAdjustmentCost(F = 0.5)

# Pre-generate shocks outside the benchmark loop
shocks = generate_shock_panel(params.demand, params.volatility, 1000, 200;
                              use_parallel = false)

# ============================================================================
# Benchmark runner
# ============================================================================

function benchmark_scaling(params, ac_begin, ac_mid_year, shocks; n_runs = 3)
    results = DataFrame(
        component    = String[],
        use_parallel = Bool[],
        run          = Int[],
        time_seconds = Float64[],
        n_threads    = Int[]
    )

    nt = get_nthreads()

    # ---- Warm-up (JIT compilation) ----
    println("\nWarm-up run (JIT)...")
    _ = solve_model(params; ac_begin = ac_begin, ac_mid_year = ac_mid_year,
                    verbose = false, use_parallel = false, use_multiscale = false)
    _ = solve_model(params; ac_begin = ac_begin, ac_mid_year = ac_mid_year,
                    verbose = false, use_parallel = true, use_multiscale = false)

    # ---- VFI ----
    println("\n" * "="^60)
    println("Benchmarking VFI")
    println("="^60)

    for use_parallel in [false, true]
        label = use_parallel ? "parallel" : "serial"
        println("\nVFI — $label:")
        for run in 1:n_runs
            GC.gc()
            t = @elapsed begin
                solve_model(params;
                            ac_begin = ac_begin, ac_mid_year = ac_mid_year,
                            verbose = false, use_parallel = use_parallel,
                            use_multiscale = false)
            end
            push!(results, ("VFI", use_parallel, run, t, nt))
            println("  Run $run: $(@sprintf("%.3f", t)) s")
        end
    end

    # Solve once for the simulation benchmark
    sol = solve_model(params; ac_begin = ac_begin, ac_mid_year = ac_mid_year,
                      verbose = false, use_parallel = true, use_multiscale = false)

    # ---- Firm simulation ----
    println("\n" * "="^60)
    println("Benchmarking Firm Simulation (1000 firms, 100 years)")
    println("="^60)

    # Warm-up simulation
    _ = simulate_firm_panel(sol, shocks; K_init = 1.0, T_years = 100, use_parallel = false)
    _ = simulate_firm_panel(sol, shocks; K_init = 1.0, T_years = 100, use_parallel = true)

    for use_parallel in [false, true]
        label = use_parallel ? "parallel" : "serial"
        println("\nSimulation — $label:")
        for run in 1:n_runs
            GC.gc()
            t = @elapsed begin
                simulate_firm_panel(sol, shocks;
                                    K_init = 1.0, T_years = 100,
                                    use_parallel = use_parallel)
            end
            push!(results, ("Simulation", use_parallel, run, t, nt))
            println("  Run $run: $(@sprintf("%.3f", t)) s")
        end
    end

    # ---- Shock generation ----
    println("\n" * "="^60)
    println("Benchmarking Shock Generation (1000 firms, 200 semesters)")
    println("="^60)

    # Warm-up
    _ = generate_shock_panel(params.demand, params.volatility, 1000, 200;
                             use_parallel = false)
    _ = generate_shock_panel(params.demand, params.volatility, 1000, 200;
                             use_parallel = true, seed = 12345)

    for use_parallel in [false, true]
        label = use_parallel ? "parallel" : "serial"
        println("\nShock generation — $label:")
        for run in 1:n_runs
            GC.gc()
            t = @elapsed begin
                generate_shock_panel(params.demand, params.volatility, 1000, 200;
                                     use_parallel = use_parallel, seed = 12345)
            end
            push!(results, ("Shock generation", use_parallel, run, t, nt))
            println("  Run $run: $(@sprintf("%.3f", t)) s")
        end
    end

    return results
end

# ============================================================================
# Run
# ============================================================================

results = benchmark_scaling(params, ac_begin, ac_mid_year, shocks; n_runs = 3)

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^60)
println("SUMMARY  ($n_threads threads)")
println("="^60)

summary_df = combine(groupby(results, [:component, :use_parallel])) do df
    (mean_time = mean(df.time_seconds),
     std_time  = std(df.time_seconds),
     min_time  = minimum(df.time_seconds))
end

println("\n" * "-"^70)
println(@sprintf("%-20s %10s %10s %10s %10s", "Component", "Serial(s)", "Parallel(s)", "Speedup", "Efficiency"))
println("-"^70)

for component in ["VFI", "Simulation", "Shock generation"]
    rows_s = summary_df[(summary_df.component .== component) .& (summary_df.use_parallel .== false), :]
    rows_p = summary_df[(summary_df.component .== component) .& (summary_df.use_parallel .== true), :]
    serial_time   = rows_s.mean_time[1]
    parallel_time = rows_p.mean_time[1]
    speedup    = serial_time / parallel_time
    efficiency = speedup / n_threads * 100

    println(@sprintf("%-20s %10.3f %10.3f %10.2fx %9.1f%%",
                     component, serial_time, parallel_time, speedup, efficiency))
end

println("-"^70)
println("Threads: $n_threads")
println("\nEfficiency = Speedup / Threads × 100%")
println("  100%  = perfect linear scaling")
println("  <100% = sublinear (typical: overhead, memory bandwidth)")
println("  >100% = superlinear (rare, usually cache effects)")

# ============================================================================
# Save
# ============================================================================

outfile = joinpath(outdir, "benchmark_results_$(n_threads)t.csv")
CSV.write(outfile, results)
println("\nResults saved to: $outfile")