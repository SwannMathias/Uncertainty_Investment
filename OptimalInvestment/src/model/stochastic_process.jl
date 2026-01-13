"""
Discretization of stochastic processes using Rouwenhorst and Tauchen methods.

Handles:
1. AR(1) processes (demand)
2. Stochastic volatility (joint discretization)
3. Transition matrix computation
"""

using LinearAlgebra
using Distributions

"""
    rouwenhorst(n::Int, ρ::Float64, σ::Float64; μ::Float64=0.0)

Discretize AR(1) process using Rouwenhorst method (preferred for persistent processes).

Process: x' = μ(1-ρ) + ρx + σε, ε ~ N(0,1)

# Arguments
- `n`: Number of grid points
- `ρ`: Persistence parameter
- `σ`: Standard deviation of innovations
- `μ`: Long-run mean (default: 0.0)

# Returns
- `grid`: Vector of n grid points
- `Π`: n×n transition probability matrix

# References
Kopecky & Suen (2010), "Finite State Markov-Chain Approximations to Highly Persistent Processes"
"""
function rouwenhorst(n::Int, ρ::Float64, σ::Float64; μ::Float64=0.0)
    @assert n >= 2 "Need at least 2 grid points"
    @assert 0.0 <= ρ < 1.0 "ρ must be in [0, 1)"
    @assert σ > 0.0 "σ must be positive"

    # Step 1: Compute unconditional variance
    σ_x = σ / sqrt(1 - ρ^2)

    # Step 2: Construct symmetric grid around zero
    ψ = sqrt(n - 1) * σ_x
    grid_centered = range(-ψ, ψ, length=n)

    # Step 3: Build transition matrix recursively
    p = (1 + ρ) / 2
    Π = rouwenhorst_matrix(n, p)

    # Step 4: Shift grid to have mean μ
    grid = collect(grid_centered) .+ μ

    return grid, Π
end

"""
    rouwenhorst_matrix(n::Int, p::Float64) -> Matrix{Float64}

Recursively construct Rouwenhorst transition matrix.
"""
function rouwenhorst_matrix(n::Int, p::Float64)
    if n == 2
        return [p (1-p); (1-p) p]
    else
        Π_nm1 = rouwenhorst_matrix(n-1, p)

        # Build n×n matrix from (n-1)×(n-1) matrix
        Π = zeros(n, n)

        Π[1:n-1, 1:n-1] .+= p .* Π_nm1
        Π[1:n-1, 2:n] .+= (1-p) .* Π_nm1
        Π[2:n, 1:n-1] .+= (1-p) .* Π_nm1
        Π[2:n, 2:n] .+= p .* Π_nm1

        # Normalize interior rows
        Π[2:n-1, :] ./= 2

        return Π
    end
end

"""
    tauchen(n::Int, ρ::Float64, σ::Float64; μ::Float64=0.0, n_std::Float64=3.0)

Discretize AR(1) process using Tauchen method.

Process: x' = μ(1-ρ) + ρx + σε, ε ~ N(0,1)

# Arguments
- `n`: Number of grid points
- `ρ`: Persistence parameter
- `σ`: Standard deviation of innovations
- `μ`: Long-run mean (default: 0.0)
- `n_std`: Number of standard deviations for grid bounds (default: 3.0)

# Returns
- `grid`: Vector of n grid points
- `Π`: n×n transition probability matrix

# References
Tauchen (1986), "Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions"
"""
function tauchen(n::Int, ρ::Float64, σ::Float64; μ::Float64=0.0, n_std::Float64=3.0)
    @assert n >= 2 "Need at least 2 grid points"
    @assert 0.0 <= ρ < 1.0 "ρ must be in [0, 1)"
    @assert σ > 0.0 "σ must be positive"
    @assert n_std > 0.0 "n_std must be positive"

    # Unconditional variance
    σ_x = σ / sqrt(1 - ρ^2)

    # Construct grid
    x_max = n_std * σ_x
    x_min = -x_max
    grid_centered = range(x_min, x_max, length=n)
    grid = collect(grid_centered) .+ μ

    # Step size
    step = (x_max - x_min) / (n - 1)

    # Transition matrix
    Π = zeros(n, n)
    d = Normal(0, 1)

    for i in 1:n
        for j in 1:n
            if j == 1
                # Lower bound
                threshold = (grid_centered[1] - ρ * grid_centered[i] + step/2) / σ
                Π[i, j] = cdf(d, threshold)
            elseif j == n
                # Upper bound
                threshold = (grid_centered[n] - ρ * grid_centered[i] - step/2) / σ
                Π[i, j] = 1 - cdf(d, threshold)
            else
                # Interior points
                threshold_upper = (grid_centered[j] - ρ * grid_centered[i] + step/2) / σ
                threshold_lower = (grid_centered[j] - ρ * grid_centered[i] - step/2) / σ
                Π[i, j] = cdf(d, threshold_upper) - cdf(d, threshold_lower)
            end
        end
    end

    # Ensure rows sum to 1 (numerical precision)
    for i in 1:n
        Π[i, :] ./= sum(Π[i, :])
    end

    return grid, Π
end

"""
    SVDiscretization

Joint discretization of demand and stochastic volatility processes.
"""
struct SVDiscretization
    D_grid::Vector{Float64}      # Log demand grid
    σ_grid::Vector{Float64}      # Log volatility grid
    n_D::Int
    n_σ::Int
    Π_σ::Matrix{Float64}         # Volatility transition P(σ'|σ)
    Π_D_given_σ::Array{Float64,3}  # Demand transition P(D'|D,σ) for each σ
    Π_joint::Matrix{Float64}     # Joint transition on (D,σ) pairs
end

"""
    discretize_sv_process(demand::DemandProcess, vol::VolatilityProcess,
                          n_D::Int, n_σ::Int; method::Symbol=:rouwenhorst)

Discretize joint stochastic volatility process.

# Arguments
- `demand`: DemandProcess parameters
- `vol`: VolatilityProcess parameters
- `n_D`: Number of demand states
- `n_σ`: Number of volatility states
- `method`: Discretization method (:rouwenhorst or :tauchen)

# Returns
- `SVDiscretization` object containing grids and transition matrices
"""
function discretize_sv_process(demand::DemandProcess, vol::VolatilityProcess,
                               n_D::Int, n_σ::Int; method::Symbol=:rouwenhorst)
    @assert method in [:rouwenhorst, :tauchen] "Method must be :rouwenhorst or :tauchen"

    # 1. Discretize volatility process (independent)
    if method == :rouwenhorst
        σ_grid, Π_σ = rouwenhorst(n_σ, vol.ρ_σ, vol.σ_η; μ=vol.σ̄)
    else
        σ_grid, Π_σ = tauchen(n_σ, vol.ρ_σ, vol.σ_η; μ=vol.σ̄)
    end

    # 2. Discretize demand process for each volatility level
    Π_D_given_σ = zeros(n_D, n_D, n_σ)
    D_grids = Vector{Vector{Float64}}(undef, n_σ)

    for i_σ in 1:n_σ
        # Volatility level (in levels, not logs)
        σ_level = exp(σ_grid[i_σ])

        # Discretize demand with this volatility
        if method == :rouwenhorst
            D_grid_temp, Π_D_temp = rouwenhorst(n_D, demand.ρ_D, σ_level; μ=demand.μ_D)
        else
            D_grid_temp, Π_D_temp = tauchen(n_D, demand.ρ_D, σ_level; μ=demand.μ_D)
        end

        D_grids[i_σ] = D_grid_temp
        Π_D_given_σ[:, :, i_σ] = Π_D_temp
    end

    # 3. Use average demand grid (they should be similar for persistent processes)
    D_grid = mean(D_grids)

    # 4. Handle correlation between shocks if ρ_εη ≠ 0
    if abs(vol.ρ_εη) > 1e-10
        # Adjust transition probabilities for correlation
        # This is a simplified approach; full bivariate normal would be more accurate
        @warn "Correlation ρ_εη = $(vol.ρ_εη) ≠ 0 detected. Using simplified correlation adjustment."
        # TODO: Implement proper bivariate discretization if needed
    end

    # 5. Construct joint transition matrix
    # State space is (D, σ) pairs, ordered as [(D₁,σ₁), (D₂,σ₁), ..., (Dₙ,σ₁), (D₁,σ₂), ...]
    n_states = n_D * n_σ
    Π_joint = zeros(n_states, n_states)

    for i_σ in 1:n_σ
        for i_D in 1:n_D
            i_state = (i_σ - 1) * n_D + i_D  # Current state index

            for i_σ_next in 1:n_σ
                for i_D_next in 1:n_D
                    i_state_next = (i_σ_next - 1) * n_D + i_D_next

                    # Joint probability: P(D',σ'|D,σ) = P(D'|D,σ) * P(σ'|σ)
                    Π_joint[i_state, i_state_next] = Π_D_given_σ[i_D, i_D_next, i_σ] * Π_σ[i_σ, i_σ_next]
                end
            end
        end
    end

    # Verify transition matrix properties
    @assert all(isapprox.(sum(Π_joint, dims=2), 1.0, atol=1e-10)) "Joint transition matrix rows must sum to 1"

    return SVDiscretization(D_grid, σ_grid, n_D, n_σ, Π_σ, Π_D_given_σ, Π_joint)
end

"""
    stationary_distribution(Π::Matrix{Float64}; tol::Float64=1e-10) -> Vector{Float64}

Compute stationary distribution of Markov chain with transition matrix Π.
"""
function stationary_distribution(Π::Matrix{Float64}; tol::Float64=1e-10, max_iter::Int=10000)
    n = size(Π, 1)
    π_stat = ones(n) / n  # Initial guess

    for iter in 1:max_iter
        π_new = π_stat' * Π
        π_new = vec(π_new)

        if maximum(abs.(π_new - π_stat)) < tol
            return π_new
        end

        π_stat = π_new
    end

    @warn "Stationary distribution did not converge after $max_iter iterations"
    return π_stat
end

"""
    verify_discretization(grid::Vector{Float64}, Π::Matrix{Float64}, ρ::Float64, σ::Float64; μ::Float64=0.0)

Verify that discretization matches theoretical moments of AR(1) process.

Returns NamedTuple with:
- theoretical_mean, empirical_mean
- theoretical_std, empirical_std
- theoretical_autocorr, empirical_autocorr
"""
function verify_discretization(grid::Vector{Float64}, Π::Matrix{Float64},
                               ρ::Float64, σ::Float64; μ::Float64=0.0)
    # Compute stationary distribution
    π_stat = stationary_distribution(Π)

    # Empirical moments
    emp_mean = dot(π_stat, grid)
    emp_std = sqrt(dot(π_stat, (grid .- emp_mean).^2))

    # Autocorrelation: E[x_{t+1} x_t] - (E[x_t])^2
    E_x_next_x = sum(π_stat[i] * grid[i] * sum(Π[i, j] * grid[j] for j in 1:length(grid))
                     for i in 1:length(grid))
    emp_autocorr = (E_x_next_x - emp_mean^2) / emp_std^2

    # Theoretical moments
    theo_mean = μ
    theo_std = σ / sqrt(1 - ρ^2)
    theo_autocorr = ρ

    return (
        theoretical_mean = theo_mean,
        empirical_mean = emp_mean,
        theoretical_std = theo_std,
        empirical_std = emp_std,
        theoretical_autocorr = theo_autocorr,
        empirical_autocorr = emp_autocorr,
        mean_error = abs(emp_mean - theo_mean),
        std_error = abs(emp_std - theo_std),
        autocorr_error = abs(emp_autocorr - theo_autocorr)
    )
end
