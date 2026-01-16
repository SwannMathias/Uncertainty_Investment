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
end

"""
    simulate_firm(sol::SolvedModel, D_path::Vector{Float64}, sigma_path::Vector{Float64},
                  K_init::Float64; T_years::Int) -> FirmHistory

Simulate single firm given shock paths.

# Arguments
- `sol`: SolvedModel object
- `D_path`: Log demand path (semesters)
- `sigma_path`: Log volatility path (semesters)
- `K_init`: Initial capital stock
- `T_years`: Number of years to simulate

# Returns
- FirmHistory object
"""
function simulate_firm(sol::SolvedModel, D_path::Vector{Float64}, sigma_path::Vector{Float64},
                      K_init::Float64; T_years::Int)
    @assert length(D_path) >= 2 * T_years "Need at least 2*T_years semesters"
    @assert length(sigma_path) >= 2 * T_years "Need at least 2*T_years semesters"

    derived = get_derived_parameters(sol.params)
    grids = sol.grids

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

    # Initial capital
    K[1] = K_init

    # Simulate year by year
    for year in 1:T_years
        # Semester indices
        sem1 = 2 * (year - 1) + 1
        sem2 = 2 * year

        # Current state
        K_current = K[year]
        log_D = D_path[sem1]
        log_sigma = sigma_path[sem1]
        D_level = exp(log_D)
        sigma_level = exp(log_sigma)

        # Store first semester states
        D_first[year] = D_level
        sigma_first[year] = sigma_level

        # Find nearest grid points for (D, sigma)
        i_D = argmin(abs.(grids.sv.D_grid .- log_D))
        i_sigma = argmin(abs.(grids.sv.sigma_grid .- log_sigma))

        # Interpolate policy function for initial investment
        I = interpolate_policy(grids, sol.I_policy, K_current, i_D, i_sigma)
        I_initial[year] = I

        # Capital after initial investment (before revision)
        K_prime = (1 - derived.delta_semester) * K_current + I

        # Mid-year shocks
        log_D_half = D_path[sem2]
        log_sigma_half = sigma_path[sem2]
        D_half_level = exp(log_D_half)
        sigma_half_level = exp(log_sigma_half)

        # Store second semester states
        D_second[year] = D_half_level
        sigma_second[year] = sigma_half_level

        # Find nearest grid points for mid-year states
        i_D_half = argmin(abs.(grids.sv.D_grid .- log_D_half))
        i_sigma_half = argmin(abs.(grids.sv.sigma_grid .- log_sigma_half))

        # Find nearest K grid index for precomputed profit lookup
        i_K = argmin(abs.(grids.K_grid .- K_current))

        # Solve mid-year problem for investment revision
        # This requires solving the optimization problem
        Delta_I_opt, _ = solve_midyear_problem(
            K_prime, i_D_half, i_sigma_half, i_K, K_current, I,
            sol.V, grids, sol.params, sol.ac, derived
        )
        Delta_I[year] = Delta_I_opt

        # Total investment
        I_tot[year] = I + Delta_I_opt

        # Next period capital
        K[year + 1] = K_prime + Delta_I_opt

        # Annual profit
        pi1 = profit(K_current, D_level, derived)
        pi2 = profit(K_current, D_half_level, derived)
        profits[year] = pi1 + pi2
    end

    return FirmHistory(T_years, K[1:end-1], D_first, D_second, sigma_first, sigma_second,
                      I_initial, Delta_I, I_tot, profits)
end

"""
    simulate_firm_panel(sol::SolvedModel, shocks::ShockPanel;
                       K_init::Float64=1.0, T_years::Int=50) -> Vector{FirmHistory}

Simulate panel of firms using shock panel.

# Arguments
- `sol`: SolvedModel object
- `shocks`: ShockPanel object
- `K_init`: Initial capital for all firms (or can be randomized)
- `T_years`: Number of years to simulate per firm

# Returns
- Vector of FirmHistory objects
"""
function simulate_firm_panel(sol::SolvedModel, shocks::ShockPanel;
                            K_init::Float64=1.0, T_years::Int=50,
                            use_parallel::Bool=true, verbose::Bool=false)
    @assert shocks.T >= 2 * T_years "Shock panel too short for requested simulation length"

    n_threads = nthreads()
    use_parallel_actual = use_parallel && n_threads > 1

    if verbose
        if use_parallel_actual
            println("Simulating $(shocks.n_firms) firms with $n_threads threads...")
        else
            println("Simulating $(shocks.n_firms) firms (serial)...")
        end
    end

    histories = Vector{FirmHistory}(undef, shocks.n_firms)

    if use_parallel_actual
        # Parallel simulation across firms
        @threads for i in 1:shocks.n_firms
            D_path = shocks.D[i, :]
            sigma_path = shocks.sigma[i, :]
            histories[i] = simulate_firm(sol, D_path, sigma_path, K_init; T_years=T_years)
        end
    else
        # Serial simulation
        for i in 1:shocks.n_firms
            D_path = shocks.D[i, :]
            sigma_path = shocks.sigma[i, :]
            histories[i] = simulate_firm(sol, D_path, sigma_path, K_init; T_years=T_years)
        end
    end

    return histories
end

"""
    simulate_firm_panel_parallel(sol::SolvedModel, shocks::ShockPanel;
                                 K_init::Float64=1.0, T_years::Int=50,
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
- Shock paths are pre-generated (no RNG during simulation)
- Each firm's trajectory depends only on its shock path
- No race conditions in output array

# Performance Notes
- Near-linear speedup for large firm panels (>100 firms)
- Minimal overhead: only array indexing is parallelized
- Memory: O(n_firms × T_years) for output storage

# Arguments
- `sol`: SolvedModel object with value and policy functions
- `shocks`: ShockPanel with pre-generated shock paths
- `K_init`: Initial capital for all firms (default: 1.0)
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
                                      K_init::Float64=1.0, T_years::Int=50,
                                      verbose::Bool=false)
    return simulate_firm_panel(sol, shocks; K_init=K_init, T_years=T_years,
                               use_parallel=true, verbose=verbose)
end
