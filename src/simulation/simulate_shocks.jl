"""
Simulate stochastic shock paths for demand and volatility processes.

# Parallelization

This module supports multi-threaded parallelization of shock panel generation.
Each thread generates shocks for a subset of firms using independent RNG streams.

To enable parallel execution:
1. Start Julia with multiple threads: `julia -t 8`
2. Use `generate_shock_panel_parallel()` or pass `use_parallel=true`

# Reproducibility

When using parallel shock generation with a seed:
- The master RNG is used to generate per-firm seeds
- Each firm gets a deterministic, unique seed based on the master seed
- Results are reproducible regardless of thread count
- Seeds are computed as: seed_base + firm_index

This ensures that firm i always gets the same shock sequence,
regardless of which thread simulates it.
"""

using Random
using Random: MersenneTwister
using Distributions
using Base.Threads: @threads, nthreads, threadid

"""
    ShockPanel

Container for simulated shock panel (demand and volatility paths for multiple firms).

The `D` and `sigma` fields store values in the process's native space
(as indicated by `D_space` and `sigma_space`). The `D_level` and `sigma_level`
fields always store values in levels.
"""
struct ShockPanel
    n_firms::Int
    T::Int                        # Length in semesters
    D::Matrix{Float64}            # Demand in native space [firm, semester]
    sigma::Matrix{Float64}            # Volatility in native space [firm, semester]
    D_level::Matrix{Float64}      # Demand level [firm, semester]
    sigma_level::Matrix{Float64}      # Volatility level [firm, semester]
    D_space::Symbol               # :log or :level
    sigma_space::Symbol           # :log or :level
end

# Backward-compatible constructor (defaults to :log space)
function ShockPanel(n_firms::Int, T::Int, D::Matrix{Float64}, sigma::Matrix{Float64},
                    D_level::Matrix{Float64}, sigma_level::Matrix{Float64})
    ShockPanel(n_firms, T, D, sigma, D_level, sigma_level, :log, :log)
end

"""
    simulate_ar1_path(rho::Float64, sigma::Float64, mu::Float64, T::Int;
                      x0=nothing, rng=Random.GLOBAL_RNG) -> Vector{Float64}

Simulate AR(1) process: x' = mu(1-rho) + rhox + sigmaepsilon.

# Arguments
- `rho`: Persistence
- `sigma`: Standard deviation of innovations
- `mu`: Long-run mean
- `T`: Length of path
- `x0`: Initial value (default: draw from stationary distribution)
- `rng`: Random number generator

# Returns
- Vector of length T with simulated path
"""
function simulate_ar1_path(rho::Float64, sigma::Float64, mu::Float64, T::Int;
                          x0=nothing, rng=Random.GLOBAL_RNG)
    @assert 0.0 <= rho < 1.0 "rho must be in [0, 1)"
    @assert sigma > 0.0 "sigma must be positive"
    @assert T > 0 "T must be positive"

    # Initialize
    x = zeros(T)

    if isnothing(x0)
        # Draw from stationary distribution
        sigma_x = sigma / sqrt(1 - rho^2)
        x[1] = mu + sigma_x * randn(rng)
    else
        x[1] = x0
    end

    # Simulate
    for t in 2:T
        epsilon = randn(rng)
        x[t] = mu * (1 - rho) + rho * x[t-1] + sigma * epsilon
    end

    return x
end

"""
    simulate_sv_path(demand::DemandProcess, vol::VolatilityProcess, T::Int;
                     D0=nothing, sigma0=nothing, rng=Random.GLOBAL_RNG) -> (Vector, Vector)

Simulate paths for demand and stochastic volatility jointly.

Paths are simulated in the process's native space (`:log` or `:level`).

# Arguments
- `demand`: DemandProcess parameters
- `vol`: Volatility specification (VolatilityProcess or TwoStateVolatility)
- `T`: Length of path (in semesters)
- `D0`: Initial value in native space (default: stationary mean)
- `sigma0`: Initial value in native space (default: stationary mean)
- `rng`: Random number generator

# Returns
- `D_path`: Demand path in native space (length T)
- `sigma_path`: Volatility path in native space (length T)
"""
function simulate_sv_path(demand::DemandProcess, vol::VolatilityProcess, T::Int;
                         D0=nothing, sigma0=nothing, rng=Random.GLOBAL_RNG)
    # Initialize
    D_path = zeros(T)
    sigma_path = zeros(T)

    # Initial values (in native space)
    D_path[1] = isnothing(D0) ? demand.mu_D : D0
    sigma_path[1] = isnothing(sigma0) ? vol.sigma_bar : sigma0

    # Correlation structure
    if abs(vol.rho_epsilon_eta) > 1e-10
        Sigma = [1.0 vol.rho_epsilon_eta; vol.rho_epsilon_eta 1.0]
        mvn = MvNormal(zeros(2), Sigma)
    else
        mvn = nothing
    end

    # Simulate
    for t in 2:T
        # Get volatility level (demand innovation std dev)
        if vol.process_space == :log
            sigma_current = exp(sigma_path[t-1])
        else
            sigma_current = max(sigma_path[t-1], 1e-10)
        end

        if isnothing(mvn)
            epsilon_D = randn(rng)
            epsilon_sigma = randn(rng)
        else
            shocks = rand(rng, mvn)
            epsilon_D = shocks[1]
            epsilon_sigma = shocks[2]
        end

        # Update demand in its native space
        D_path[t] = demand.mu_D * (1 - demand.rho_D) + demand.rho_D * D_path[t-1] + sigma_current * epsilon_D

        # Update volatility in its native space
        sigma_path[t] = vol.sigma_bar * (1 - vol.rho_sigma) + vol.rho_sigma * sigma_path[t-1] + vol.sigma_eta * epsilon_sigma
    end

    return D_path, sigma_path
end

"""
    simulate_sv_path(demand::DemandProcess, vol::TwoStateVolatility, T::Int;
                     D0=nothing, sigma0=nothing, rng=Random.GLOBAL_RNG) -> (Vector, Vector)

Simulate demand path with two-state Markov switching volatility.

# Returns
- `D_path`: Demand path in native space (length T)
- `sigma_path`: Volatility path in native space (length T)
"""
function simulate_sv_path(demand::DemandProcess, vol::TwoStateVolatility, T::Int;
                         D0=nothing, sigma0=nothing, rng=Random.GLOBAL_RNG)
    D_path = zeros(T)
    sigma_path = zeros(T)

    # Initial demand value
    D_path[1] = isnothing(D0) ? demand.mu_D : D0

    # Initial volatility state (default: state 1)
    if isnothing(sigma0)
        sigma_state = 1
    else
        # Find closest state
        sigma_state = argmin(abs.(vol.sigma_levels .- sigma0))
    end
    sigma_path[1] = vol.sigma_levels[sigma_state]

    # Get volatility levels for demand innovations
    if vol.process_space == :log
        sigma_levels_for_demand = exp.(vol.sigma_levels)
    else
        sigma_levels_for_demand = vol.sigma_levels
    end

    for t in 2:T
        # Transition volatility state
        u = rand(rng)
        sigma_state = u < vol.Pi_sigma[sigma_state, 1] ? 1 : 2
        sigma_path[t] = vol.sigma_levels[sigma_state]

        # Current demand innovation std dev
        sigma_current = max(sigma_levels_for_demand[sigma_state], 1e-10)

        # Update demand in its native space
        epsilon_D = randn(rng)
        D_path[t] = demand.mu_D * (1 - demand.rho_D) + demand.rho_D * D_path[t-1] + sigma_current * epsilon_D
    end

    return D_path, sigma_path
end

"""
    generate_shock_panel(demand::DemandProcess, vol::AbstractVolatilitySpec,
                        n_firms::Int, T::Int;
                        burn_in::Int=100, rng=Random.GLOBAL_RNG,
                        use_parallel::Bool=false, seed::Union{Int,Nothing}=nothing) -> ShockPanel

Generate panel of shock paths for multiple firms.

# Arguments
- `demand`: DemandProcess parameters
- `vol`: Volatility specification (VolatilityProcess or TwoStateVolatility)
- `n_firms`: Number of firms
- `T`: Length of each path (in semesters)
- `burn_in`: Number of initial periods to discard
- `rng`: Random number generator (used in serial mode)
- `use_parallel`: Enable parallel generation (default: false for backward compatibility)
- `seed`: Master seed for parallel generation (only used if use_parallel=true)

# Parallelization
When `use_parallel=true` and multiple threads are available:
- Each firm gets a unique RNG seeded as: seed_base + firm_index
- This ensures reproducibility regardless of thread count
- The `rng` argument is ignored in parallel mode

# Returns
- ShockPanel object
"""
function generate_shock_panel(demand::DemandProcess, vol::AbstractVolatilitySpec,
                             n_firms::Int, T::Int;
                             burn_in::Int=100, rng=Random.GLOBAL_RNG,
                             use_parallel::Bool=false, seed::Union{Int,Nothing}=nothing)
    # Use parallel version if requested and threads available
    if use_parallel && nthreads() > 1
        return generate_shock_panel_parallel(demand, vol, n_firms, T;
                                             burn_in=burn_in, seed=seed)
    end

    # Serial implementation
    @assert n_firms > 0 "n_firms must be positive"
    @assert T > 0 "T must be positive"
    @assert burn_in >= 0 "burn_in must be non-negative"

    # Determine process spaces
    D_space = demand.process_space
    sigma_space = vol isa VolatilityProcess ? vol.process_space :
                  vol isa TwoStateVolatility ? vol.process_space : :log

    # Allocate storage
    D_native = zeros(n_firms, T)
    sigma_native = zeros(n_firms, T)

    # Simulate each firm
    for i in 1:n_firms
        D_full, sigma_full = simulate_sv_path(demand, vol, T + burn_in; rng=rng)
        D_native[i, :] = D_full[(burn_in+1):end]
        sigma_native[i, :] = sigma_full[(burn_in+1):end]
    end

    # Convert to levels
    D_level = D_space == :log ? exp.(D_native) : D_native
    sigma_level = sigma_space == :log ? exp.(sigma_native) : sigma_native

    return ShockPanel(n_firms, T, D_native, sigma_native, D_level, sigma_level, D_space, sigma_space)
end

"""
    get_firm_shocks(panel::ShockPanel, firm_id::Int) -> (Vector, Vector)

Extract shock paths for a single firm.

# Returns
- `D_path`: Log demand path
- `sigma_path`: Log volatility path
"""
function get_firm_shocks(panel::ShockPanel, firm_id::Int)
    @assert 1 <= firm_id <= panel.n_firms "firm_id out of range"
    return panel.D[firm_id, :], panel.sigma[firm_id, :]
end

"""
    get_firm_shocks_level(panel::ShockPanel, firm_id::Int) -> (Vector, Vector)

Extract shock paths for a single firm (in levels).
"""
function get_firm_shocks_level(panel::ShockPanel, firm_id::Int)
    @assert 1 <= firm_id <= panel.n_firms "firm_id out of range"
    return panel.D_level[firm_id, :], panel.sigma_level[firm_id, :]
end

"""
    shock_statistics(panel::ShockPanel) -> NamedTuple

Compute summary statistics for shock panel.
"""
function shock_statistics(panel::ShockPanel)
    return (
        n_firms = panel.n_firms,
        T = panel.T,
        # Log demand statistics
        D_log_mean = mean(panel.D),
        D_log_std = std(panel.D),
        D_log_min = minimum(panel.D),
        D_log_max = maximum(panel.D),
        # Log volatility statistics
        sigma_log_mean = mean(panel.sigma),
        sigma_log_std = std(panel.sigma),
        sigma_log_min = minimum(panel.sigma),
        sigma_log_max = maximum(panel.sigma),
        # Level statistics
        D_level_mean = mean(panel.D_level),
        D_level_std = std(panel.D_level),
        sigma_level_mean = mean(panel.sigma_level),
        sigma_level_std = std(panel.sigma_level)
    )
end

"""
    print_shock_statistics(panel::ShockPanel)

Print formatted shock panel statistics.
"""
function print_shock_statistics(panel::ShockPanel)
    stats = shock_statistics(panel)

    println("\n" * "="^70)
    println("Shock Panel Statistics")
    println("="^70)
    println("Number of firms: $(stats.n_firms)")
    println("Time periods: $(stats.T) semesters")

    println("\nLog Demand:")
    println("  Mean: $(format_number(stats.D_log_mean))")
    println("  Std Dev: $(format_number(stats.D_log_std))")
    println("  Range: [$(format_number(stats.D_log_min)), $(format_number(stats.D_log_max))]")

    println("\nLog Volatility:")
    println("  Mean: $(format_number(stats.sigma_log_mean))")
    println("  Std Dev: $(format_number(stats.sigma_log_std))")
    println("  Range: [$(format_number(stats.sigma_log_min)), $(format_number(stats.sigma_log_max))]")

    println("\nDemand Level:")
    println("  Mean: $(format_number(stats.D_level_mean))")
    println("  Std Dev: $(format_number(stats.D_level_std))")

    println("\nVolatility Level:")
    println("  Mean: $(format_number(stats.sigma_level_mean))")
    println("  Std Dev: $(format_number(stats.sigma_level_std))")

    println("="^70)
end

# ============================================================================
# Parallel Shock Generation
# ============================================================================

"""
    generate_shock_panel_parallel(demand::DemandProcess, vol::AbstractVolatilitySpec,
                                  n_firms::Int, T::Int;
                                  burn_in::Int=100, seed::Union{Int,Nothing}=nothing) -> ShockPanel

Generate panel of shock paths using parallel execution with thread-safe RNGs.

# Parallelization Strategy
- Each firm gets a unique RNG seeded deterministically from the master seed
- Firms are distributed across threads using @threads
- Each thread simulates its assigned firms independently
- Results are reproducible regardless of thread count

# Reproducibility
When a seed is provided:
- Firm i always receives seed = seed_base + i
- This ensures identical results whether using 1 or N threads
- Without a seed, behavior is non-deterministic

# Thread Safety
- Each firm uses its own MersenneTwister RNG instance
- No shared mutable state between threads
- Output arrays are written to independent locations

# Arguments
- `demand`: DemandProcess parameters
- `vol`: Volatility specification (VolatilityProcess or TwoStateVolatility)
- `n_firms`: Number of firms
- `T`: Length of each path (in semesters)
- `burn_in`: Number of initial periods to discard
- `seed`: Master random seed (optional, for reproducibility)

# Returns
- ShockPanel object

# Example
```julia
# Reproducible parallel shock generation
shocks = generate_shock_panel_parallel(
    params.demand, params.volatility, 1000, 120;
    seed=12345
)

# Results are identical regardless of thread count
```
"""
function generate_shock_panel_parallel(demand::DemandProcess, vol::AbstractVolatilitySpec,
                                       n_firms::Int, T::Int;
                                       burn_in::Int=100, seed::Union{Int,Nothing}=nothing)
    @assert n_firms > 0 "n_firms must be positive"
    @assert T > 0 "T must be positive"
    @assert burn_in >= 0 "burn_in must be non-negative"

    # Determine process spaces
    D_space = demand.process_space
    sigma_space = vol isa VolatilityProcess ? vol.process_space :
                  vol isa TwoStateVolatility ? vol.process_space : :log

    # Determine seed base (use current time if no seed provided)
    seed_base = isnothing(seed) ? abs(rand(Int)) : seed

    # Allocate storage
    D_native = zeros(n_firms, T)
    sigma_native = zeros(n_firms, T)

    n_threads = nthreads()

    # Parallel generation with per-firm RNGs
    @threads for i in 1:n_firms
        firm_rng = MersenneTwister(seed_base + i)
        D_full, sigma_full = simulate_sv_path(demand, vol, T + burn_in; rng=firm_rng)
        D_native[i, :] = D_full[(burn_in+1):end]
        sigma_native[i, :] = sigma_full[(burn_in+1):end]
    end

    # Convert to levels
    D_level = D_space == :log ? exp.(D_native) : D_native
    sigma_level = sigma_space == :log ? exp.(sigma_native) : sigma_native

    return ShockPanel(n_firms, T, D_native, sigma_native, D_level, sigma_level, D_space, sigma_space)
end

