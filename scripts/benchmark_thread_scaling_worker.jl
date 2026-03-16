"""
    benchmark_worker.jl

Worker script launched by benchmark_thread_scaling.jl with a specific thread count.
Solves the model n_runs times and prints CSV-formatted timing results to stdout.

Usage (called automatically by launcher):
    julia -t N benchmark_worker.jl <n_runs> <outfile>
"""

using Pkg
project_root = ARGS[length(ARGS) >= 3 ? 3 : end]  # last arg or fallback
if isdir(joinpath(project_root, "src"))
    Pkg.activate(project_root)
end

using UncertaintyInvestment
using Random
using LinearAlgebra
using Printf
using Statistics

# ─── Parse arguments ──────────────────────────────────────────────────────────
n_runs   = parse(Int, ARGS[1])
outfile  = ARGS[2]
# project_root already parsed above

# ─── Suppress BLAS threading (critical for accurate thread scaling) ─────────
BLAS.set_num_threads(1)

n_threads = Threads.nthreads()

# ─── Problem definition (must match across all thread counts) ──────────────
Random.seed!(12345)

sigma_bar_1 = log(0.1)

params = ModelParameters(
    alpha   = 0.33,
    epsilon = 4.0,
    delta   = 0.10,
    beta    = 0.96,
    demand  = DemandProcess(mu_D = log(500), rho_D = 0.5),
    volatility = VolatilityProcess(
        sigma_bar       = sigma_bar_1,
        rho_sigma       = 0.1,
        sigma_eta       = 0.1,
        rho_epsilon_eta = 0.0
    ),
    numerical = NumericalSettings(
        n_K          = 50,
        n_D          = 50,
        n_sigma      = 5,
        K_min_factor = 0.1,
        K_max_factor = 10.0,
        tol_vfi      = 1e-4,
        max_iter     = 200,
        howard_steps = 0
    )
)

ac_begin    = FixedAdjustmentCost(F = 0.5)
ac_mid_year = FixedAdjustmentCost(F = 0.5)

# Pre-generate shocks for simulation benchmark
shocks = generate_shock_panel(params.demand, params.volatility, 500, 200;
                              use_parallel=false)

# ─── Warm-up (JIT compilation) ─────────────────────────────────────────────
println(stderr, "  [threads=$n_threads] JIT warm-up...")
_ = solve_model(params; ac_begin=ac_begin, ac_mid_year=ac_mid_year,
                verbose=false, use_parallel=(n_threads > 1), use_multiscale=false)

# ─── Benchmark VFI ─────────────────────────────────────────────────────────
println(stderr, "  [threads=$n_threads] Benchmarking VFI ($n_runs runs)...")
vfi_times = Float64[]
for run in 1:n_runs
    GC.gc()
    t = @elapsed begin
        solve_model(params; ac_begin=ac_begin, ac_mid_year=ac_mid_year,
                    verbose=false, use_parallel=(n_threads > 1), use_multiscale=false)
    end
    push!(vfi_times, t)
    println(stderr, "    Run $run: $(@sprintf("%.3f", t)) s")
end

# ─── Benchmark Simulation ─────────────────────────────────────────────────
println(stderr, "  [threads=$n_threads] Benchmarking Simulation ($n_runs runs)...")
sol = solve_model(params; ac_begin=ac_begin, ac_mid_year=ac_mid_year,
                  verbose=false, use_parallel=(n_threads > 1), use_multiscale=false)

# Warm-up simulation
_ = simulate_firm_panel(sol, shocks; K_init=1.0, T_years=50,
                        use_parallel=(n_threads > 1))

sim_times = Float64[]
for run in 1:n_runs
    GC.gc()
    t = @elapsed begin
        simulate_firm_panel(sol, shocks; K_init=1.0, T_years=50,
                            use_parallel=(n_threads > 1))
    end
    push!(sim_times, t)
    println(stderr, "    Run $run: $(@sprintf("%.3f", t)) s")
end

# ─── Write results ─────────────────────────────────────────────────────────
open(outfile, "w") do io
    println(io, "n_threads,component,run,time_seconds")
    for (run, t) in enumerate(vfi_times)
        println(io, "$n_threads,VFI,$run,$t")
    end
    for (run, t) in enumerate(sim_times)
        println(io, "$n_threads,Simulation,$run,$t")
    end
end

println(stderr, "  [threads=$n_threads] Done. Results → $outfile")
