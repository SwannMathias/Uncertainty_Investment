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
"""
struct ShockPanel
    n_firms::Int
    T::Int                        # Length in semesters
    D::Matrix{Float64}            # Log demand [firm, semester]
    sigma::Matrix{Float64}            # Log volatility [firm, semester]
    D_level::Matrix{Float64}      # Demand level [firm, semester]
    sigma_level::Matrix{Float64}      # Volatility level [firm, semester]
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

# Arguments
- `demand`: DemandProcess parameters
- `vol`: VolatilityProcess parameters
- `T`: Length of path (in semesters)
- `D0`: Initial log demand (default: stationary mean)
- `sigma0`: Initial log volatility (default: stationary mean)
- `rng`: Random number generator

# Returns
- `D_path`: Log demand path (length T)
- `sigma_path`: Log volatility path (length T)
"""
function simulate_sv_path(demand::DemandProcess, vol::VolatilityProcess, T::Int;
                         D0=nothing, sigma0=nothing, rng=Random.GLOBAL_RNG)
    # Initialize
    D_path = zeros(T)
    sigma_path = zeros(T)

    # Initial values
    if isnothing(D0)
        D_path[1] = demand.mu_D
    else
        D_path[1] = D0
    end

    if isnothing(sigma0)
        sigma_path[1] = vol.sigma_bar
    else
        sigma_path[1] = sigma0
    end

    # Correlation structure
    if abs(vol.rho_epsilon_eta) > 1e-10
        # Correlated shocks
        Sigma = [1.0 vol.rho_epsilon_eta; vol.rho_epsilon_eta 1.0]
        mvn = MvNormal(zeros(2), Sigma)
    else
        mvn = nothing
    end

    # Simulate
    for t in 2:T
        # Current volatility level (for demand innovation)
        sigma_current = exp(sigma_path[t-1])

        if isnothing(mvn)
            # Independent shocks
            epsilon_D = randn(rng)
            epsilon_sigma = randn(rng)
        else
            # Correlated shocks
            shocks = rand(rng, mvn)
            epsilon_D = shocks[1]
            epsilon_sigma = shocks[2]
        end

        # Update demand (volatility affects demand innovation)
        D_path[t] = demand.mu_D * (1 - demand.rho_D) + demand.rho_D * D_path[t-1] + sigma_current * epsilon_D

        # Update volatility
        sigma_path[t] = vol.sigma_bar * (1 - vol.rho_sigma) + vol.rho_sigma * sigma_path[t-1] + vol.sigma_eta * epsilon_sigma
    end

    return D_path, sigma_path
end

"""
    generate_shock_panel(demand::DemandProcess, vol::VolatilityProcess,
                        n_firms::Int, T::Int;
                        burn_in::Int=100, rng=Random.GLOBAL_RNG,
                        use_parallel::Bool=false, seed::Union{Int,Nothing}=nothing) -> ShockPanel

Generate panel of shock paths for multiple firms.

# Arguments
- `demand`: DemandProcess parameters
- `vol`: VolatilityProcess parameters
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
function generate_shock_panel(demand::DemandProcess, vol::VolatilityProcess,
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

    # Allocate storage
    D_log = zeros(n_firms, T)
    sigma_log = zeros(n_firms, T)

    # Simulate each firm
    for i in 1:n_firms
        # Simulate with burn-in
        D_full, sigma_full = simulate_sv_path(demand, vol, T + burn_in; rng=rng)

        # Keep post-burn-in periods
        D_log[i, :] = D_full[(burn_in+1):end]
        sigma_log[i, :] = sigma_full[(burn_in+1):end]
    end

    # Convert to levels
    D_level = exp.(D_log)
    sigma_level = exp.(sigma_log)

    return ShockPanel(n_firms, T, D_log, sigma_log, D_level, sigma_level)
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
    generate_shock_panel_parallel(demand::DemandProcess, vol::VolatilityProcess,
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
- `vol`: VolatilityProcess parameters
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
function generate_shock_panel_parallel(demand::DemandProcess, vol::VolatilityProcess,
                                       n_firms::Int, T::Int;
                                       burn_in::Int=100, seed::Union{Int,Nothing}=nothing)
    @assert n_firms > 0 "n_firms must be positive"
    @assert T > 0 "T must be positive"
    @assert burn_in >= 0 "burn_in must be non-negative"

    # Determine seed base (use current time if no seed provided)
    seed_base = isnothing(seed) ? abs(rand(Int)) : seed

    # Allocate storage
    D_log = zeros(n_firms, T)
    sigma_log = zeros(n_firms, T)

    n_threads = nthreads()

    # Parallel generation with per-firm RNGs
    @threads for i in 1:n_firms
        # Create a unique, deterministic RNG for this firm
        # This ensures reproducibility regardless of thread assignment
        firm_rng = MersenneTwister(seed_base + i)

        # Simulate with burn-in using firm-specific RNG
        D_full, sigma_full = simulate_sv_path(demand, vol, T + burn_in; rng=firm_rng)

        # Keep post-burn-in periods
        D_log[i, :] = D_full[(burn_in+1):end]
        sigma_log[i, :] = sigma_full[(burn_in+1):end]
    end

    # Convert to levels
    D_level = exp.(D_log)
    sigma_level = exp.(sigma_log)

    return ShockPanel(n_firms, T, D_log, sigma_log, D_level, sigma_level)
end

