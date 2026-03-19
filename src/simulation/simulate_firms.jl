"""
Simulate firm decisions using solved policy functions.

# Parallelization

This module supports multi-threaded parallelization of firm simulation.
Firms are simulated independently in parallel, with each thread handling
a subset of firms.

To enable parallel execution:
1. Start Julia with multiple threads: `julia -t 8`
2. Use `simulate_firm_panel_parallel()` or pass `use_parallel=true`

Thread safety:
- Each firm's simulation is completely independent
- Thread-local RNG streams ensure reproducibility
- The solved model is read-only (shared safely across threads)
"""

using Base.Threads: @threads, nthreads, threadid
using Random: MersenneTwister

"""
    FirmHistory

Container for simulated firm history.
"""
struct FirmHistory
    T::Int                          # Number of years
    K::Vector{Float64}              # Capital stock
    D::Vector{Float64}              # Demand (first semester)
    D_half::Vector{Float64}         # Demand (second semester)
    sigma::Vector{Float64}              # Volatility (first semester)
    sigma_half::Vector{Float64}         # Volatility (second semester)
    I::Vector{Float64}              # Initial investment
    Delta_I::Vector{Float64}             # Investment revision
    I_total::Vector{Float64}        # Total investment (I + Delta_I)
    profit::Vector{Float64}         # Annual profit
    E_last_semester::Vector{Float64}  # E[I_total_t | K_t, D_{t-1/2}, σ_{t-1/2}]
    E_beginning::Vector{Float64}      # E[I_total_t | K_t, D_t, σ_t]
    E_half::Vector{Float64}           # E[I_total_t | info at t+1/2] = I_t + ΔI_t
end

"""
    simulate_firm(sol::SolvedModel, D_path::Vector{Float64}, sigma_path::Vector{Float64},
                  K_init::Float64; T_years::Int) -> FirmHistory

Simulate single firm given shock paths.

# Arguments
- `sol`: SolvedModel object
- `D_path`: Demand path in native space (semesters)
- `sigma_path`: Volatility path in native space (semesters)
- `K_init`: Initial capital stock
- `T_years`: Number of years to simulate

# Returns
- FirmHistory object
"""
function simulate_firm(sol::SolvedModel, D_path::Vector{Float64}, sigma_path::Vector{Float64},
                      K_init::Float64; T_years::Int, compute_plans::Bool=true)
    @assert length(D_path) >= 2 * T_years "Need at least 2*T_years semesters"
    @assert length(sigma_path) >= 2 * T_years "Need at least 2*T_years semesters"

    derived = get_derived_parameters(sol.params)
    grids = sol.grids

    # Precompute expected value cache (fixed during simulation since sol.V doesn't change)
    EV_cache = zeros(grids.n_K, grids.n_states)
    precompute_expectation_cache!(EV_cache, sol.V, grids; horizon=:semester)

    # Allocate storage
    K = zeros(T_years + 1)
    D_first = zeros(T_years)
    D_second = zeros(T_years)
    sigma_first = zeros(T_years)
    sigma_second = zeros(T_years)
    I_initial = zeros(T_years)
    Delta_I = zeros(T_years)
    I_tot = zeros(T_years)
    profits = zeros(T_years)
    E_last_semester = fill(NaN, T_years)
    E_beginning = zeros(T_years)
    E_half = zeros(T_years)

    # Track previous mid-year state indices for E_last_semester computation
    prev_i_D_half = 0
    prev_i_sigma_half = 0

    # Check if Delta_I_policy is available as a 3D array for expectation computation
    has_delta_policy_3d = !isnothing(sol.Delta_I_policy) && ndims(sol.Delta_I_policy) == 3

    # Initial capital
    K[1] = K_init

    # Simulate year by year
    for year in 1:T_years
        # Semester indices
        sem1 = 2 * (year - 1) + 1
        sem2 = 2 * year

        # Current state
        K_current = K[year]
        D_native = D_path[sem1]       # Value in process's native space
        sigma_native = sigma_path[sem1]
        D_level = grids.sv.D_space == :log ? exp(D_native) : D_native
        sigma_level = grids.sv.sigma_space == :log ? exp(sigma_native) : sigma_native

        # Store first semester states
        D_first[year] = D_level
        sigma_first[year] = sigma_level

        # Find nearest grid points (compare in native space)
        i_D = argmin(abs.(grids.sv.D_grid .- D_native))
        i_sigma = argmin(abs.(grids.sv.sigma_grid .- sigma_native))

        # Interpolate policy function for initial investment
        I = interpolate_policy(grids, sol.I_policy, K_current, i_D, i_sigma)
        I_initial[year] = I

        # Capital after initial investment (before revision)
        K_prime = (1 - derived.delta_semester) * K_current + I

        # Mid-year shocks
        D_half_native = D_path[sem2]
        sigma_half_native = sigma_path[sem2]
        D_half_level = grids.sv.D_space == :log ? exp(D_half_native) : D_half_native
        sigma_half_level = grids.sv.sigma_space == :log ? exp(sigma_half_native) : sigma_half_native

        # Store second semester states
        D_second[year] = D_half_level
        sigma_second[year] = sigma_half_level

        # Find nearest grid points for mid-year states (compare in native space)
        i_D_half = argmin(abs.(grids.sv.D_grid .- D_half_native))
        i_sigma_half = argmin(abs.(grids.sv.sigma_grid .- sigma_half_native))

        # Stage-1 policy is now solved directly in VFI; interpolate it at (K_prime, D_half, sigma_half)
        if has_delta_policy_3d
            Delta_I_opt = interpolate_policy(grids, sol.Delta_I_policy, K_prime, i_D_half, i_sigma_half)
        else
            i_K = argmin(abs.(grids.K_grid .- K_current))
            Delta_I_opt, _ = solve_midyear_problem(
                K_prime, i_D_half, i_sigma_half, i_K, K_current, I,
                sol.V, grids, sol.params, sol.ac_mid_year, derived, EV_cache
            )
        end
        Delta_I[year] = Delta_I_opt

        # Total investment
        I_tot[year] = I + Delta_I_opt

        # --- Conditional expectations of I_total_t ---
        if compute_plans
            # E_half: E[I_total_t | info at t+1/2] = I_t + ΔI_t (both decisions made)
            E_half[year] = I + Delta_I_opt

            # E_beginning: E[I_total_t | K_t, D_t, σ_t]
            # I_t is chosen; integrate ΔI*(K'_t, D_half, σ_half) over mid-year states
            if has_delta_policy_3d
                i_state_beg = get_joint_state_index(grids, i_D, i_sigma)
                E_delta_I_beg = 0.0
                for i_sh in 1:grids.n_states
                    prob = grids.Pi_semester[i_state_beg, i_sh]
                    if prob < 1e-15
                        continue
                    end
                    i_D_h, i_sigma_h = get_D_sigma_indices(grids, i_sh)
                    delta_I_star = interpolate_policy(grids, sol.Delta_I_policy, K_prime, i_D_h, i_sigma_h)
                    E_delta_I_beg += prob * delta_I_star
                end
                E_beginning[year] = I + E_delta_I_beg
            else
                # Fallback: without 3D Delta_I_policy, use realized value
                E_beginning[year] = I + Delta_I_opt
            end

            # E_last_semester: E[I_total_t | K_t, D_{t-1/2}, σ_{t-1/2}]
            # Double expectation: outer over (D_t, σ_t), inner over (D_half, σ_half)
            if year == 1
                E_last_semester[year] = NaN  # No t-1/2 information available
            elseif has_delta_policy_3d
                i_state_prev_half = get_joint_state_index(grids, prev_i_D_half, prev_i_sigma_half)
                E_last = 0.0
                for i_st in 1:grids.n_states
                    prob_outer = grids.Pi_semester[i_state_prev_half, i_st]
                    if prob_outer < 1e-15
                        continue
                    end
                    i_D_t, i_sigma_t = get_D_sigma_indices(grids, i_st)
                    # I*(K_t, D_t, σ_t) under this realization
                    I_star = interpolate_policy(grids, sol.I_policy, K_current, i_D_t, i_sigma_t)
                    K_prime_t = (1 - derived.delta_semester) * K_current + I_star
                    # Inner sum: E[ΔI | D_t, σ_t]
                    E_delta_I_inner = 0.0
                    for i_sh in 1:grids.n_states
                        prob_inner = grids.Pi_semester[i_st, i_sh]
                        if prob_inner < 1e-15
                            continue
                        end
                        i_D_h, i_sigma_h = get_D_sigma_indices(grids, i_sh)
                        delta_I_star = interpolate_policy(grids, sol.Delta_I_policy, K_prime_t, i_D_h, i_sigma_h)
                        E_delta_I_inner += prob_inner * delta_I_star
                    end
                    E_last += prob_outer * (I_star + E_delta_I_inner)
                end
                E_last_semester[year] = E_last
            else
                E_last_semester[year] = NaN
            end
        end  # compute_plans

        # Store previous mid-year indices for next year's E_last_semester
        prev_i_D_half = i_D_half
        prev_i_sigma_half = i_sigma_half

        # Next period capital (apply second semester depreciation)
        K[year + 1] = (1 - derived.delta_semester) * K_prime + Delta_I_opt

        # Annual profit
        pi1 = profit(K_current, D_level, derived)
        pi2 = profit(K_current, D_half_level, derived)
        profits[year] = pi1 + pi2
    end

    return FirmHistory(T_years, K[1:end-1], D_first, D_second, sigma_first, sigma_second,
                      I_initial, Delta_I, I_tot, profits,
                      E_last_semester, E_beginning, E_half)
end

"""
    simulate_firm_panel(sol::SolvedModel, shocks::ShockPanel;
                       K_init::Union{Float64, Nothing}=nothing,
                       T_years::Int=50) -> Vector{FirmHistory}

Simulate panel of firms using shock panel.

# Arguments
- `sol`: SolvedModel object
- `shocks`: ShockPanel object
- `K_init`: Initial capital for firms. If `Float64`, all firms start with this value.
  If `nothing` (default), each firm is randomly assigned a capital level from the grid.
- `T_years`: Number of years to simulate per firm
- `use_parallel`: Whether to use multi-threaded parallelization (default: `true`)
- `verbose`: Print progress information (default: `false`)

# Returns
- Vector of FirmHistory objects
"""
function simulate_firm_panel(sol::SolvedModel, shocks::ShockPanel;
                            K_init::Union{Float64, Nothing}=nothing, T_years::Int=50,
                            use_parallel::Bool=true, verbose::Bool=false,
                            compute_plans::Bool=true)
    @assert shocks.T >= 2 * T_years "Shock panel too short for requested simulation length"

    n_firms = shocks.n_firms
    grids = sol.grids

    # Generate initial capital for each firm
    K_inits = Vector{Float64}(undef, n_firms)
    if K_init isa Float64
        fill!(K_inits, K_init)
    else
        # Randomly assign each firm an initial capital from the grid.
        # Use a deterministic seed for reproducibility.
        rng = MersenneTwister(42)
        for i in 1:n_firms
            idx = rand(rng, 1:grids.n_K)
            K_inits[i] = grids.K_grid[idx]
        end
    end

    n_threads = nthreads()
    use_parallel_actual = use_parallel && n_threads > 1

    if verbose
        if use_parallel_actual
            println("Simulating $n_firms firms with $n_threads threads...")
        else
            println("Simulating $n_firms firms (serial)...")
        end
    end

    histories = Vector{FirmHistory}(undef, n_firms)

    if use_parallel_actual
        # Parallel simulation across firms
        @threads for i in 1:n_firms
            D_path = shocks.D[i, :]
            sigma_path = shocks.sigma[i, :]
            histories[i] = simulate_firm(sol, D_path, sigma_path, K_inits[i]; T_years=T_years, compute_plans=compute_plans)
        end
    else
        # Serial simulation
        for i in 1:n_firms
            D_path = shocks.D[i, :]
            sigma_path = shocks.sigma[i, :]
            histories[i] = simulate_firm(sol, D_path, sigma_path, K_inits[i]; T_years=T_years, compute_plans=compute_plans)
        end
    end

    return histories
end

"""
    simulate_firm_panel_parallel(sol::SolvedModel, shocks::ShockPanel;
                                 K_init::Union{Float64, Nothing}=nothing,
                                 T_years::Int=50,
                                 verbose::Bool=false) -> Vector{FirmHistory}

Explicit parallel version of firm panel simulation.

This is equivalent to `simulate_firm_panel(...; use_parallel=true)` but
provides a more explicit interface for parallel execution.

# Parallelization Strategy
- Firms are distributed across threads using `@threads`
- Each thread simulates its assigned firms completely independently
- The solved model is shared (read-only) across all threads
- Shock paths are pre-computed and read-only

# Thread Safety
- `sol::SolvedModel` is immutable and shared safely
- `shocks::ShockPanel` is read-only
- Each element `histories[i]` is written by exactly one thread

# Reproducibility
Results are deterministic because:
- Shock paths are pre-generated (no RNG during simulation loop)
- When `K_init=nothing`, initial capitals are drawn using a seeded RNG (seed=42)
- Each firm's trajectory depends only on its shock path and initial capital
- No race conditions in output array

# Performance Notes
- Near-linear speedup for large firm panels (>100 firms)
- Minimal overhead: only array indexing is parallelized
- Memory: O(n_firms × T_years) for output storage

# Arguments
- `sol`: SolvedModel object with value and policy functions
- `shocks`: ShockPanel with pre-generated shock paths
- `K_init`: Initial capital for firms. If `Float64`, all firms start with this value.
  If `nothing` (default), each firm is randomly assigned a capital level from the grid.
- `T_years`: Number of years to simulate per firm
- `verbose`: Print progress information

# Returns
- Vector of FirmHistory objects, one per firm

# Example
```julia
# Generate shocks
shocks = generate_shock_panel(params.demand, params.volatility, 1000, 120)

# Parallel simulation (using all available threads)
histories = simulate_firm_panel_parallel(sol, shocks; T_years=50)
```
"""
function simulate_firm_panel_parallel(sol::SolvedModel, shocks::ShockPanel;
                                      K_init::Union{Float64, Nothing}=nothing, T_years::Int=50,
                                      verbose::Bool=false, compute_plans::Bool=true)
    return simulate_firm_panel(sol, shocks; K_init=K_init, T_years=T_years,
                               use_parallel=true, verbose=verbose, compute_plans=compute_plans)
end
