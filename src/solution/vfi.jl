"""
Value function iteration (VFI) solver for the dynamic investment problem.

Main function: solve_model(params; ac, verbose, use_parallel) -> SolvedModel

# Parallelization

This module supports multi-threaded parallelization of value function iteration.
To enable parallel execution:

1. Start Julia with multiple threads:
   ```bash
   julia -t 8  # Use 8 threads
   # Or set environment variable before starting Julia:
   export JULIA_NUM_THREADS=8
   ```

2. Pass `use_parallel=true` to `solve_model()` or `value_function_iteration()`

The parallelization is applied to the Bellman operator, which distributes
the state space optimization across available threads. Each thread independently
solves the dynamic programming problem for its assigned states.

Thread safety is ensured by:
- Read-only access to the current value function V
- Independent write locations for V_new and I_policy per state
- No shared mutable state between threads

Performance notes:
- Near-linear speedup for large state spaces (n_states > 1000)
- Threading overhead may reduce benefit for small problems
- Recommended: n_threads ≤ number of physical CPU cores
"""

using ProgressMeter
using Printf
using Base.Threads: @threads, nthreads, threadid

# Re-export threading utilities for user convenience
"""
    get_nthreads() -> Int

Return the number of threads available for parallel execution.
This wraps `Threads.nthreads()` for convenience.

To increase the number of threads, start Julia with:
    julia -t N
or set JULIA_NUM_THREADS=N before starting Julia.
"""
get_nthreads() = nthreads()

"""
    get_threadid() -> Int

Return the ID of the current thread (1 to nthreads()).
This wraps `Threads.threadid()` for convenience.
"""
get_threadid() = threadid()

"""
    SolvedModel

Container for solved model including value and policy functions.
"""
struct SolvedModel
    params::ModelParameters
    grids::StateGrids
    ac::AbstractAdjustmentCost
    V::Array{Float64,3}           # V[i_K, i_D, i_sigma]
    I_policy::Array{Float64,3}    # I[i_K, i_D, i_sigma]
    Delta_I_policy::Union{Nothing,Array{Float64,5}}  # Delta_I[i_K, i_D, i_sigma, i_D_half, i_sigma_half] (optional)
    convergence::NamedTuple
end

"""
    value_function_iteration(grids::StateGrids, params::ModelParameters,
                            ac::AbstractAdjustmentCost;
                            V_init=nothing, verbose=true, use_parallel=true) -> SolvedModel

Solve the model using value function iteration.

# Arguments
- `grids`: StateGrids object
- `params`: ModelParameters
- `ac`: AbstractAdjustmentCost specification
- `V_init`: Initial guess for value function (default: zeros)
- `verbose`: Print iteration progress
- `use_parallel`: Use multi-threaded parallel execution (default: true)

# Parallelization
When `use_parallel=true` and Julia is started with multiple threads
(e.g., `julia -t 8`), the Bellman operator will parallelize over the
state space. Each thread independently solves the optimization problem
for its assigned states.

To check/set the number of threads:
- `get_nthreads()` returns current thread count
- Start Julia with `julia -t N` or set `JULIA_NUM_THREADS=N`

# Returns
- SolvedModel object with solution
"""
function value_function_iteration(grids::StateGrids, params::ModelParameters,
                                 ac::AbstractAdjustmentCost;
                                 V_init=nothing, verbose=true, use_parallel=true)
    derived = get_derived_parameters(params)

    # Determine if we should use parallel execution
    n_threads = nthreads()
    use_parallel_actual = use_parallel && n_threads > 1

    # Initialize value and policy functions
    if isnothing(V_init)
        V = zeros(grids.n_K, grids.n_D, grids.n_sigma)
        # Better initial guess: static profit
        for i_sigma in 1:grids.n_sigma
            for i_D in 1:grids.n_D
                D = get_D(grids, i_D)
                for i_K in 1:grids.n_K
                    K = get_K(grids, i_K)
                    # Approximate value as discounted stream of current profit
                    pi = profit(K, D, derived)
                    V[i_K, i_D, i_sigma] = pi / (1 - params.beta)
                end
            end
        end
    else
        V = copy(V_init)
    end

    V_new = similar(V)
    I_policy = zeros(grids.n_K, grids.n_D, grids.n_sigma)
    I_policy_old = zeros(grids.n_K, grids.n_D, grids.n_sigma)

    # VFI iteration
    iter = 0
    dist = Inf
    dist_policy = 0.0
    start_time = time()

    if verbose
        println("\n" * "="^70)
        println("Value Function Iteration")
        println("="^70)
        println("Grid size: $(grids.n_K) × $(grids.n_D) × $(grids.n_sigma) = $(grids.n_K * grids.n_D * grids.n_sigma)")
        println("Adjustment cost: $(describe_adjustment_cost(ac))")
        if use_parallel_actual
            println("Parallelization: ENABLED ($n_threads threads)")
        else
            if use_parallel && n_threads == 1
                println("Parallelization: DISABLED (only 1 thread available)")
                println("  Tip: Start Julia with 'julia -t N' for N threads")
            else
                println("Parallelization: DISABLED (use_parallel=false)")
            end
        end
        println("="^70)
        println(@sprintf("%-8s %-15s %-15s %-15s", "Iter", "V Distance", "Policy Distance", "Time"))
        println("-"^70)
    end

    while iter < params.numerical.max_iter
        iter += 1

        # Apply Bellman operator (parallel or serial)
        if use_parallel_actual
            bellman_operator_parallel!(V_new, V, I_policy, grids, params, ac, derived)
        else
            bellman_operator!(V_new, V, I_policy, grids, params, ac, derived)
        end

        # Check convergence
        dist = maximum(abs.(V_new .- V))

        # Compare with previous policy
        if iter > 1
            dist_policy = maximum(abs.(I_policy .- I_policy_old))
        end

        # Store current policy for next iteration
        I_policy_old .= I_policy

        # Print progress
        if verbose && (iter % 10 == 0 || iter == 1)
            elapsed = time() - start_time
            println(@sprintf("%-8d %-15.2e %-15.2e %-15.2f", iter, dist, dist_policy, elapsed))
        end

        # Check convergence
        if dist < params.numerical.tol_vfi && dist_policy < params.numerical.tol_policy
            if verbose
                elapsed = time() - start_time
                println("-"^70)
                println("✓ Converged in $iter iterations ($(format_time(elapsed)))")
                println("  Final distance: $(dist)")
                println("  Policy distance: $(dist_policy)")
                if use_parallel_actual
                    println("  Threads used: $n_threads")
                end
                println("="^70)
            end
            break
        end

        # Update value function
        V .= V_new

        # Howard improvement steps (parallel or serial)
        if params.numerical.howard_steps > 0 && iter % 20 == 0
            if use_parallel_actual
                howard_improvement_step_parallel!(V, I_policy, grids, params, ac, derived,
                                                  params.numerical.howard_steps)
            else
                howard_improvement_step!(V, I_policy, grids, params, ac, derived,
                                        params.numerical.howard_steps)
            end
        end

        if iter == params.numerical.max_iter
            @warn "VFI did not converge after $iter iterations. Distance: $dist"
        end
    end

    # Convergence information
    convergence = (
        converged = (dist < params.numerical.tol_vfi),
        iterations = iter,
        final_distance = dist,
        final_policy_distance = dist_policy,
        elapsed_time = time() - start_time,
        threads_used = use_parallel_actual ? n_threads : 1
    )

    return SolvedModel(params, grids, ac, V, I_policy, nothing, convergence)
end

"""
    solve_model(params::ModelParameters; ac=NoAdjustmentCost(), verbose=true, use_parallel=true) -> SolvedModel

High-level interface to solve the model.

# Arguments
- `params`: ModelParameters
- `ac`: AbstractAdjustmentCost (default: no adjustment costs)
- `verbose`: Print progress
- `use_parallel`: Use multi-threaded parallel execution (default: true)

# Parallelization
The solver automatically uses all available threads when `use_parallel=true`.
To set the number of threads, start Julia with:
    julia -t N
or set the environment variable before starting Julia:
    export JULIA_NUM_THREADS=N

# Returns
- SolvedModel object

# Example
```julia
params = ModelParameters(alpha=0.33, epsilon=4.0)
sol = solve_model(params; ac=ConvexAdjustmentCost(phi=2.0))

# Disable parallelization (for debugging or comparison)
sol_serial = solve_model(params; ac=ConvexAdjustmentCost(phi=2.0), use_parallel=false)
```
"""
function solve_model(params::ModelParameters; ac=NoAdjustmentCost(), verbose=true, use_parallel=true)
    # Validate parameters
    if !validate_parameters(params)
        error("Invalid parameter configuration")
    end

    # Construct grids
    if verbose
        println("\nConstructing state space grids...")
    end
    grids = construct_grids(params)

    if verbose
        print_grid_info(grids)
    end

    # Solve using VFI
    sol = value_function_iteration(grids, params, ac; verbose=verbose, use_parallel=use_parallel)

    # Diagnostics
    if verbose
        diag = solution_diagnostics(sol)
        print_solution_diagnostics(diag)
    end

    return sol
end

"""
    solution_diagnostics(sol::SolvedModel) -> NamedTuple

Compute diagnostic statistics for the solution.
"""
function solution_diagnostics(sol::SolvedModel)
    grids = sol.grids
    derived = get_derived_parameters(sol.params)

    # Value function statistics
    V_mean = mean(sol.V)
    V_std = std(sol.V)
    V_min = minimum(sol.V)
    V_max = maximum(sol.V)

    # Policy function statistics
    I_mean = mean(sol.I_policy)
    I_std = std(sol.I_policy)
    I_min = minimum(sol.I_policy)
    I_max = maximum(sol.I_policy)

    # Investment rate (I/K) statistics
    I_rate = zeros(grids.n_K, grids.n_D, grids.n_sigma)
    for i_sigma in 1:grids.n_sigma
        for i_D in 1:grids.n_D
            for i_K in 1:grids.n_K
                K = get_K(grids, i_K)
                I_rate[i_K, i_D, i_sigma] = sol.I_policy[i_K, i_D, i_sigma] / K
            end
        end
    end

    I_rate_mean = mean(I_rate)
    I_rate_std = std(I_rate)

    # Inaction rate (for models with adjustment costs)
    inaction_freq = sum(abs.(sol.I_policy) .< 1e-6) / length(sol.I_policy)

    # Steady-state investment rate (should approximately equal depreciation)
    # At K = K_ss, average investment rate
    i_K_ss = argmin(abs.(grids.K_grid .- derived.K_ss))
    I_rate_ss = mean(I_rate[i_K_ss, :, :])

    return (
        V_mean = V_mean,
        V_std = V_std,
        V_range = (V_min, V_max),
        I_mean = I_mean,
        I_std = I_std,
        I_range = (I_min, I_max),
        I_rate_mean = I_rate_mean,
        I_rate_std = I_rate_std,
        I_rate_ss = I_rate_ss,
        inaction_frequency = inaction_freq,
        depreciation_rate = derived.delta_semester
    )
end

"""
    print_solution_diagnostics(diag::NamedTuple)

Print formatted solution diagnostics.
"""
function print_solution_diagnostics(diag::NamedTuple)
    println("\n" * "="^70)
    println("Solution Diagnostics")
    println("="^70)

    println("\nValue Function:")
    println("  Mean: $(format_number(diag.V_mean))")
    println("  Std Dev: $(format_number(diag.V_std))")
    println("  Range: [$(format_number(diag.V_range[1])), $(format_number(diag.V_range[2]))]")

    println("\nInvestment Policy:")
    println("  Mean: $(format_number(diag.I_mean))")
    println("  Std Dev: $(format_number(diag.I_std))")
    println("  Range: [$(format_number(diag.I_range[1])), $(format_number(diag.I_range[2]))]")

    println("\nInvestment Rate (I/K):")
    println("  Mean: $(format_number(diag.I_rate_mean, digits=4))")
    println("  Std Dev: $(format_number(diag.I_rate_std, digits=4))")
    println("  At K_ss: $(format_number(diag.I_rate_ss, digits=4))")
    println("  Depreciation: $(format_number(diag.depreciation_rate, digits=4))")

    if diag.inaction_frequency > 0.01
        println("\nAdjustment Costs:")
        println("  Inaction frequency: $(format_number(diag.inaction_frequency * 100, digits=2))%")
    end

    println("="^70)
end

"""
    evaluate_value(sol::SolvedModel, K::Float64, D::Float64, sigma::Float64) -> Float64

Evaluate value function at arbitrary (K, D, sigma) using interpolation.

# Arguments
- `sol`: SolvedModel
- `K`: Capital level
- `D`: Demand level (not log)
- `sigma`: Volatility level (not log)

# Returns
- Interpolated value
"""
function evaluate_value(sol::SolvedModel, K::Float64, D::Float64, sigma::Float64)
    # Find D and sigma indices (nearest neighbor for discrete states)
    log_D = log(D)
    log_sigma = log(sigma)

    i_D = argmin(abs.(sol.grids.sv.D_grid .- log_D))
    i_sigma = argmin(abs.(sol.grids.sv.sigma_grid .- log_sigma))

    # Interpolate on K
    return interpolate_value(sol.grids, sol.V, K, i_D, i_sigma)
end

"""
    evaluate_policy(sol::SolvedModel, K::Float64, D::Float64, sigma::Float64) -> Float64

Evaluate policy function at arbitrary (K, D, sigma) using interpolation.
"""
function evaluate_policy(sol::SolvedModel, K::Float64, D::Float64, sigma::Float64)
    log_D = log(D)
    log_sigma = log(sigma)

    i_D = argmin(abs.(sol.grids.sv.D_grid .- log_D))
    i_sigma = argmin(abs.(sol.grids.sv.sigma_grid .- log_sigma))

    return interpolate_policy(sol.grids, sol.I_policy, K, i_D, i_sigma)
end

"""
    compute_stationary_distribution(sol::SolvedModel; tol=1e-6, max_iter=10000) -> Array{Float64,3}

Compute stationary distribution of (K, D, sigma) under optimal policy.

This is computationally intensive and not typically needed for estimation.
"""
function compute_stationary_distribution(sol::SolvedModel; tol=1e-6, max_iter=10000)
    # This would require simulating the distribution forward
    # For now, return a placeholder
    @warn "Stationary distribution computation not yet implemented"
    return ones(sol.grids.n_K, sol.grids.n_D, sol.grids.n_sigma) ./ (sol.grids.n_K * sol.grids.n_D * sol.grids.n_sigma)
end
