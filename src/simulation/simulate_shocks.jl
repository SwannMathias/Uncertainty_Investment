"""
Simulate stochastic shock paths for demand and volatility processes.
"""

using Random
using Distributions

"""
    ShockPanel

Container for simulated shock panel (demand and volatility paths for multiple firms).
"""
struct ShockPanel
    n_firms::Int
    T::Int                        # Length in semesters
    D::Matrix{Float64}            # Log demand [firm, semester]
    σ::Matrix{Float64}            # Log volatility [firm, semester]
    D_level::Matrix{Float64}      # Demand level [firm, semester]
    sigma_level::Matrix{Float64}      # Volatility level [firm, semester]
end

"""
    simulate_ar1_path(ρ::Float64, σ::Float64, μ::Float64, T::Int;
                      x0=nothing, rng=Random.GLOBAL_RNG) -> Vector{Float64}

Simulate AR(1) process: x' = μ(1-ρ) + ρx + σε.

# Arguments
- `ρ`: Persistence
- `σ`: Standard deviation of innovations
- `μ`: Long-run mean
- `T`: Length of path
- `x0`: Initial value (default: draw from stationary distribution)
- `rng`: Random number generator

# Returns
- Vector of length T with simulated path
"""
function simulate_ar1_path(ρ::Float64, σ::Float64, μ::Float64, T::Int;
                          x0=nothing, rng=Random.GLOBAL_RNG)
    @assert 0.0 <= ρ < 1.0 "ρ must be in [0, 1)"
    @assert σ > 0.0 "σ must be positive"
    @assert T > 0 "T must be positive"

    # Initialize
    x = zeros(T)

    if isnothing(x0)
        # Draw from stationary distribution
        sigma_x = σ / sqrt(1 - ρ^2)
        x[1] = μ + sigma_x * randn(rng)
    else
        x[1] = x0
    end

    # Simulate
    for t in 2:T
        ε = randn(rng)
        x[t] = μ * (1 - ρ) + ρ * x[t-1] + σ * ε
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
        Σ = [1.0 vol.rho_epsilon_eta; vol.rho_epsilon_eta 1.0]
        mvn = MvNormal(zeros(2), Σ)
    else
        mvn = nothing
    end

    # Simulate
    for t in 2:T
        # Current volatility level (for demand innovation)
        sigma_current = exp(sigma_path[t-1])

        if isnothing(mvn)
            # Independent shocks
            ε_D = randn(rng)
            ε_sigma = randn(rng)
        else
            # Correlated shocks
            shocks = rand(rng, mvn)
            ε_D = shocks[1]
            ε_sigma = shocks[2]
        end

        # Update demand (volatility affects demand innovation)
        D_path[t] = demand.mu_D * (1 - demand.rho_D) + demand.rho_D * D_path[t-1] + sigma_current * ε_D

        # Update volatility
        sigma_path[t] = vol.sigma_bar * (1 - vol.rho_sigma) + vol.rho_sigma * sigma_path[t-1] + vol.sigma_eta * ε_sigma
    end

    return D_path, sigma_path
end

"""
    generate_shock_panel(demand::DemandProcess, vol::VolatilityProcess,
                        n_firms::Int, T::Int;
                        burn_in::Int=100, rng=Random.GLOBAL_RNG) -> ShockPanel

Generate panel of shock paths for multiple firms.

# Arguments
- `demand`: DemandProcess parameters
- `vol`: VolatilityProcess parameters
- `n_firms`: Number of firms
- `T`: Length of each path (in semesters)
- `burn_in`: Number of initial periods to discard
- `rng`: Random number generator

# Returns
- ShockPanel object
"""
function generate_shock_panel(demand::DemandProcess, vol::VolatilityProcess,
                             n_firms::Int, T::Int;
                             burn_in::Int=100, rng=Random.GLOBAL_RNG)
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
    return panel.D[firm_id, :], panel.σ[firm_id, :]
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
        sigma_log_mean = mean(panel.σ),
        sigma_log_std = std(panel.σ),
        sigma_log_min = minimum(panel.σ),
        sigma_log_max = maximum(panel.σ),
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
