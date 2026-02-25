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
    use_half_period::Bool         # Whether the nested half-period problem was used
end

"""
    value_function_iteration(grids::StateGrids, params::ModelParameters,
                            ac::AbstractAdjustmentCost;
                            V_init=nothing, verbose=true, use_parallel=true,
                            use_half_period=true) -> SolvedModel

Solve the model using value function iteration.

# Arguments
- `grids`: StateGrids object
- `params`: ModelParameters
- `ac`: AbstractAdjustmentCost specification
- `V_init`: Initial guess for value function (default: zeros)
- `verbose`: Print iteration progress
- `use_parallel`: Use multi-threaded parallel execution (default: true)
- `use_half_period`: Use nested half-period Bellman (default: true). When false,
  solves the simpler annual model without mid-year investment revision.

# Returns
- SolvedModel object with solution
"""
function value_function_iteration(grids::StateGrids, params::ModelParameters,
                                 ac::AbstractAdjustmentCost;
                                 V_init=nothing, verbose=true, use_parallel=true,
                                 use_half_period=true)
    derived = get_derived_parameters(params)

    # Determine if we should use parallel execution
    n_threads = nthreads()
    use_parallel_actual = use_parallel && n_threads > 1

    # Initialize value and policy functions
    if isnothing(V_init)
        V = zeros(grids.n_K, grids.n_D, grids.n_sigma)
        # Better initial guess: static profit (using precomputed profits)
        for i_sigma in 1:grids.n_sigma
            for i_D in 1:grids.n_D
                for i_K in 1:grids.n_K
                    # Approximate value as discounted stream of current profit
                    # Use precomputed profit for efficiency
                    pi = get_profit(grids, i_K, i_D)
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
        println("Model mode: $(use_half_period ? "nested half-period" : "annual (no half-period)")")
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

        # Apply Bellman operator (parallel or serial, half-period or annual)
        if use_half_period
            if use_parallel_actual
                bellman_operator_parallel!(V_new, V, I_policy, grids, params, ac, derived)
            else
                bellman_operator!(V_new, V, I_policy, grids, params, ac, derived)
            end
        else
            if use_parallel_actual
                bellman_operator_annual_parallel!(V_new, V, I_policy, grids, params, ac, derived)
            else
                bellman_operator_annual!(V_new, V, I_policy, grids, params, ac, derived)
            end
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

        # Howard improvement steps (parallel or serial, half-period or annual)
        if params.numerical.howard_steps > 0 && iter % 20 == 0
            if use_half_period
                if use_parallel_actual
                    howard_improvement_step_parallel!(V, I_policy, grids, params, ac, derived,
                                                      params.numerical.howard_steps)
                else
                    howard_improvement_step!(V, I_policy, grids, params, ac, derived,
                                            params.numerical.howard_steps)
                end
            else
                if use_parallel_actual
                    howard_improvement_step_annual_parallel!(V, I_policy, grids, params, ac, derived,
                                                             params.numerical.howard_steps)
                else
                    howard_improvement_step_annual!(V, I_policy, grids, params, ac, derived,
                                                   params.numerical.howard_steps)
                end
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
        threads_used = use_parallel_actual ? n_threads : 1,
        use_half_period = use_half_period
    )

    return SolvedModel(params, grids, ac, V, I_policy, nothing, convergence, use_half_period)
end

"""
    solve_model(params::ModelParameters; ac=NoAdjustmentCost(), verbose=true, use_parallel=true,
                use_multiscale=false, use_half_period=true) -> SolvedModel

High-level interface to solve the model.

# Arguments
- `params`: ModelParameters
- `ac`: AbstractAdjustmentCost (default: no adjustment costs)
- `verbose`: Print progress
- `use_parallel`: Use multi-threaded parallel execution (default: true)
- `use_multiscale`: Use coarse-to-fine grid refinement for faster convergence (default: false)
- `use_half_period`: Use nested half-period Bellman (default: true). When false,
  solves the simpler annual model:
  V(K,D,σ) = max_I { π(K,D) - C(I,K) + β E[V(K',D',σ')|D,σ] }
  using annual β and δ and Pi_year for expectations.

# Returns
- SolvedModel object

# Example
```julia
params = ModelParameters(alpha=0.33, epsilon=4.0)
sol = solve_model(params; ac=ConvexAdjustmentCost(phi=2.0))

# Annual model without half-period
sol_annual = solve_model(params; ac=ConvexAdjustmentCost(phi=2.0), use_half_period=false)
```
"""
function solve_model(params::ModelParameters; ac=NoAdjustmentCost(), verbose=true, use_parallel=true,
                     use_multiscale=false, use_half_period=true)
    # Validate parameters
    if !validate_parameters(params)
        error("Invalid parameter configuration")
    end

    # Use multi-scale solver if requested
    if use_multiscale
        return solve_model_multiscale(params; ac=ac, verbose=verbose, use_parallel=use_parallel,
                                      use_half_period=use_half_period)
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
    sol = value_function_iteration(grids, params, ac; verbose=verbose, use_parallel=use_parallel,
                                   use_half_period=use_half_period)

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
    edge_min_count = 0
    edge_max_count = 0
    for i_sigma in 1:grids.n_sigma
        for i_D in 1:grids.n_D
            for i_K in 1:grids.n_K
                K = get_K(grids, i_K)
                I_rate[i_K, i_D, i_sigma] = sol.I_policy[i_K, i_D, i_sigma] / K
                if i_K == 1
                    edge_min_count += 1
                elseif i_K == grids.n_K
                    edge_max_count += 1
                end
            end
        end
    end

    I_rate_mean = mean(I_rate)
    I_rate_std = std(I_rate)

    total_states = grids.n_K * grids.n_D * grids.n_sigma
    edge_min_share = edge_min_count / total_states
    edge_max_share = edge_max_count / total_states

    # Inaction rate (for models with adjustment costs)
    inaction_freq = sum(abs.(sol.I_policy) .< 1e-6) / length(sol.I_policy)

    # Steady-state investment rate (should approximately equal depreciation)
    # At K = K_ss, average investment rate
    i_K_ss = argmin(abs.(grids.K_grid .- derived.K_ss))
    I_rate_ss = mean(I_rate[i_K_ss, :, :])

    # Report the depreciation rate relevant to the solution mode
    dep_rate = sol.use_half_period ? derived.delta_semester : sol.params.delta

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
        depreciation_rate = dep_rate,
        K_edge_min_share = edge_min_share,
        K_edge_max_share = edge_max_share
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

    println("\nCapital Grid Edges (current K):")
    println("  At K_min: $(format_number(diag.K_edge_min_share * 100, digits=2))%")
    println("  At K_max: $(format_number(diag.K_edge_max_share * 100, digits=2))%")

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

# ============================================================================
# Multi-Scale Grid Refinement
# ============================================================================

"""
    interpolate_value_function(V_coarse::Array{Float64,3},
                               grids_coarse::StateGrids,
                               grids_fine::StateGrids) -> Array{Float64,3}

Interpolate value function from coarse grid to fine grid.

Uses linear interpolation in K (continuous) and nearest neighbor in D, σ (discrete).

# Arguments
- `V_coarse`: Value function on coarse grid, dimensions [n_K_coarse, n_D_coarse, n_sigma_coarse]
- `grids_coarse`: StateGrids for coarse grid
- `grids_fine`: StateGrids for fine grid

# Returns
- `V_fine`: Value function interpolated to fine grid, dimensions [n_K_fine, n_D_fine, n_sigma_fine]

# Algorithm
For each point on the fine grid:
1. Find nearest D and σ indices on coarse grid (discrete states)
2. Linearly interpolate in K dimension (continuous state)
"""
function interpolate_value_function(
    V_coarse::Array{Float64,3},
    grids_coarse::StateGrids,
    grids_fine::StateGrids
) :: Array{Float64,3}

    V_fine = zeros(grids_fine.n_K, grids_fine.n_D, grids_fine.n_sigma)

    for i_sigma_fine in 1:grids_fine.n_sigma
        for i_D_fine in 1:grids_fine.n_D
            # Find nearest coarse grid indices for D and sigma (discrete states)
            log_D_fine = get_log_D(grids_fine, i_D_fine)
            log_sigma_fine = get_log_sigma(grids_fine, i_sigma_fine)

            i_D_coarse = argmin(abs.(grids_coarse.sv.D_grid .- log_D_fine))
            i_sigma_coarse = argmin(abs.(grids_coarse.sv.sigma_grid .- log_sigma_fine))

            # Extract coarse values at this (D, sigma) combination
            V_coarse_slice = V_coarse[:, i_D_coarse, i_sigma_coarse]

            # Interpolate in K dimension (continuous)
            for i_K_fine in 1:grids_fine.n_K
                K_fine = get_K(grids_fine, i_K_fine)
                V_fine[i_K_fine, i_D_fine, i_sigma_fine] = linear_interp_1d(
                    grids_coarse.K_grid,
                    V_coarse_slice,
                    K_fine
                )
            end
        end
    end

    return V_fine
end

"""
    solve_model_multiscale(params::ModelParameters;
                          ac=NoAdjustmentCost(),
                          verbose=true,
                          use_parallel=true) -> SolvedModel

Solve model using coarse-to-fine grid refinement for faster convergence.

# Algorithm
1. Solve on coarse grid (grid sizes ÷ 2)
2. Interpolate solution to fine grid
3. Use as warm start for fine grid VFI

Typically 3-5x faster than direct fine grid solution.

# Arguments
- `params`: ModelParameters with target (fine) grid settings
- `ac`: AbstractAdjustmentCost (default: no adjustment costs)
- `verbose`: Print progress information
- `use_parallel`: Use multi-threaded parallel execution

# Returns
- SolvedModel object with solution on fine grid

# Example
```julia
params = ModelParameters(
    numerical = NumericalSettings(n_K = 80, n_D = 12, n_sigma = 6)
)
sol = solve_model_multiscale(params; ac=ConvexAdjustmentCost(phi=2.0))
```
"""
function solve_model_multiscale(
    params::ModelParameters;
    ac::AbstractAdjustmentCost = NoAdjustmentCost(),
    verbose::Bool = true,
    use_parallel::Bool = true,
    use_half_period::Bool = true
) :: SolvedModel

    if verbose
        println("\n" * "="^70)
        println("Multi-Scale VFI Solver")
        println("="^70)
        println("Target grid: $(params.numerical.n_K) × $(params.numerical.n_D) × $(params.numerical.n_sigma)")
    end

    # Step 1: Create coarse parameters
    params_coarse = ModelParameters(
        alpha = params.alpha,
        epsilon = params.epsilon,
        delta = params.delta,
        beta = params.beta,
        demand = params.demand,
        volatility = params.volatility,
        numerical = NumericalSettings(
            n_K = max(10, params.numerical.n_K ÷ 2),
            n_D = max(5, params.numerical.n_D ÷ 2),
            n_sigma = max(3, params.numerical.n_sigma ÷ 2),
            K_min_factor = params.numerical.K_min_factor,
            K_max_factor = params.numerical.K_max_factor,
            tol_vfi = 1e-4,  # Relaxed tolerance for coarse grid
            tol_policy = 1e-4,
            max_iter = params.numerical.max_iter,
            howard_steps = params.numerical.howard_steps,
            interp_method = params.numerical.interp_method
        )
    )

    if verbose
        println("Coarse grid: $(params_coarse.numerical.n_K) × $(params_coarse.numerical.n_D) × $(params_coarse.numerical.n_sigma)")
    end

    # Step 2: Solve coarse grid
    if verbose
        println("\nSTEP 1: Solving on coarse grid...")
    end

    grids_coarse = construct_grids(params_coarse)
    sol_coarse = value_function_iteration(
        grids_coarse, params_coarse, ac;
        V_init = nothing,
        verbose = verbose,
        use_parallel = use_parallel,
        use_half_period = use_half_period
    )

    coarse_time = sol_coarse.convergence.elapsed_time
    coarse_iters = sol_coarse.convergence.iterations

    if verbose
        println("Coarse grid converged in $coarse_iters iterations ($(format_time(coarse_time)))")
    end

    # Step 3: Interpolate to fine grid
    if verbose
        println("\nSTEP 2: Interpolating to fine grid...")
    end

    grids_fine = construct_grids(params)
    V_init_fine = interpolate_value_function(sol_coarse.V, grids_coarse, grids_fine)

    if verbose
        println("Interpolation complete")
        println("  Value range: [$(format_number(minimum(V_init_fine))), $(format_number(maximum(V_init_fine)))]")
    end

    # Step 4: Solve fine grid with warm start
    if verbose
        println("\nSTEP 3: Solving on fine grid with warm start...")
    end

    sol_fine = value_function_iteration(
        grids_fine, params, ac;
        V_init = V_init_fine,
        verbose = verbose,
        use_parallel = use_parallel,
        use_half_period = use_half_period
    )

    fine_time = sol_fine.convergence.elapsed_time
    fine_iters = sol_fine.convergence.iterations

    if verbose
        total_time = coarse_time + fine_time
        println("\n" * "="^70)
        println("Multi-Scale Summary:")
        println("  Coarse: $coarse_iters iterations ($(format_time(coarse_time)))")
        println("  Fine: $fine_iters iterations ($(format_time(fine_time)))")
        println("  Total: $(format_time(total_time))")
        println("="^70)
    end

    return sol_fine
end
