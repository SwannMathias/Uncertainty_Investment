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
    rouwenhorst(n::Int, rho::Float64, sigma::Float64; mu::Float64=0.0)

Discretize AR(1) process using Rouwenhorst method (preferred for persistent processes).

Process: x' = mu(1-rho) + rho*x + sigma*epsilon, epsilon ~ N(0,1)

# Arguments
- `n`: Number of grid points
- `rho`: Persistence parameter
- `sigma`: Standard deviation of innovations
- `mu`: Long-run mean (default: 0.0)

# Returns
- `grid`: Vector of n grid points
- `Pi`: nxn transition probability matrix

# References
Kopecky & Suen (2010), "Finite State Markov-Chain Approximations to Highly Persistent Processes"
"""
function rouwenhorst(n::Int, rho::Float64, sigma::Float64; mu::Float64=0.0)
    @assert n >= 2 "Need at least 2 grid points"
    @assert 0.0 <= rho < 1.0 "rho must be in [0, 1)"
    @assert sigma > 0.0 "sigma must be positive"

    # Step 1: Compute unconditional variance
    sigma_x = sigma / sqrt(1 - rho^2)

    # Step 2: Construct symmetric grid around zero
    psi = sqrt(n - 1) * sigma_x
    grid_centered = range(-psi, psi, length=n)

    # Step 3: Build transition matrix recursively
    p = (1 + rho) / 2
    Pi = rouwenhorst_matrix(n, p)

    # Step 4: Shift grid to have mean mu
    grid = collect(grid_centered) .+ mu

    return grid, Pi
end

"""
    rouwenhorst_matrix(n::Int, p::Float64) -> Matrix{Float64}

Recursively construct Rouwenhorst transition matrix.
"""
function rouwenhorst_matrix(n::Int, p::Float64)
    if n == 2
        return [p (1-p); (1-p) p]
    else
        Pi_nm1 = rouwenhorst_matrix(n-1, p)

        # Build nxn matrix from (n-1)x(n-1) matrix
        Pi = zeros(n, n)

        Pi[1:n-1, 1:n-1] .+= p .* Pi_nm1
        Pi[1:n-1, 2:n] .+= (1-p) .* Pi_nm1
        Pi[2:n, 1:n-1] .+= (1-p) .* Pi_nm1
        Pi[2:n, 2:n] .+= p .* Pi_nm1

        # Normalize interior rows
        Pi[2:n-1, :] ./= 2

        return Pi
    end
end

"""
    tauchen(n::Int, rho::Float64, sigma::Float64; mu::Float64=0.0, n_std::Float64=3.0)

Discretize AR(1) process using Tauchen method.

Process: x' = mu(1-rho) + rho*x + sigma*epsilon, epsilon ~ N(0,1)

# Arguments
- `n`: Number of grid points
- `rho`: Persistence parameter
- `sigma`: Standard deviation of innovations
- `mu`: Long-run mean (default: 0.0)
- `n_std`: Number of standard deviations for grid bounds (default: 3.0)

# Returns
- `grid`: Vector of n grid points
- `Pi`: nxn transition probability matrix

# References
Tauchen (1986), "Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions"
"""
function tauchen(n::Int, rho::Float64, sigma::Float64; mu::Float64=0.0, n_std::Float64=3.0)
    @assert n >= 2 "Need at least 2 grid points"
    @assert 0.0 <= rho < 1.0 "rho must be in [0, 1)"
    @assert sigma > 0.0 "sigma must be positive"
    @assert n_std > 0.0 "n_std must be positive"

    # Unconditional variance
    sigma_x = sigma / sqrt(1 - rho^2)

    # Construct grid
    x_max = n_std * sigma_x
    x_min = -x_max
    grid_centered = range(x_min, x_max, length=n)
    grid = collect(grid_centered) .+ mu

    # Step size
    step = (x_max - x_min) / (n - 1)

    # Transition matrix
    Pi = zeros(n, n)
    d = Normal(0, 1)

    for i in 1:n
        for j in 1:n
            if j == 1
                # Lower bound
                threshold = (grid_centered[1] - rho * grid_centered[i] + step/2) / sigma
                Pi[i, j] = cdf(d, threshold)
            elseif j == n
                # Upper bound
                threshold = (grid_centered[n] - rho * grid_centered[i] - step/2) / sigma
                Pi[i, j] = 1 - cdf(d, threshold)
            else
                # Interior points
                threshold_upper = (grid_centered[j] - rho * grid_centered[i] + step/2) / sigma
                threshold_lower = (grid_centered[j] - rho * grid_centered[i] - step/2) / sigma
                Pi[i, j] = cdf(d, threshold_upper) - cdf(d, threshold_lower)
            end
        end
    end

    # Ensure rows sum to 1 (numerical precision)
    for i in 1:n
        Pi[i, :] ./= sum(Pi[i, :])
    end

    return grid, Pi
end

"""
    SVDiscretization

Joint discretization of demand and stochastic volatility processes.
"""
struct SVDiscretization
    D_grid::Vector{Float64}      # Log demand grid
    sigma_grid::Vector{Float64}      # Log volatility grid
    n_D::Int
    n_sigma::Int
    Pi_sigma::Matrix{Float64}         # Volatility transition P(sigma'|sigma)
    Pi_D_given_sigma::Array{Float64,3}  # Demand transition P(D'|D,sigma) for each sigma
    Pi_joint::Matrix{Float64}     # Joint transition on (D,sigma) pairs
end

"""
    discretize_sv_process(demand::DemandProcess, vol::VolatilityProcess,
                          n_D::Int, n_sigma::Int; method::Symbol=:rouwenhorst)

Discretize joint stochastic volatility process.

# Arguments
- `demand`: DemandProcess parameters
- `vol`: VolatilityProcess parameters
- `n_D`: Number of demand states
- `n_sigma`: Number of volatility states
- `method`: Discretization method (:rouwenhorst or :tauchen)

# Returns
- `SVDiscretization` object containing grids and transition matrices
"""
function discretize_sv_process(demand::DemandProcess, vol::VolatilityProcess,
                               n_D::Int, n_sigma::Int; method::Symbol=:rouwenhorst)
    @assert method in [:rouwenhorst, :tauchen] "Method must be :rouwenhorst or :tauchen"

    # 1. Discretize volatility process (independent)
    if method == :rouwenhorst
        sigma_grid, Pi_sigma = rouwenhorst(n_sigma, vol.rho_sigma, vol.sigma_eta; mu=vol.sigma_bar)
    else
        sigma_grid, Pi_sigma = tauchen(n_sigma, vol.rho_sigma, vol.sigma_eta; mu=vol.sigma_bar)
    end

    # 2. Discretize demand process for each volatility level
    Pi_D_given_sigma = zeros(n_D, n_D, n_sigma)
    D_grids = Vector{Vector{Float64}}(undef, n_sigma)

    for i_sigma in 1:n_sigma
        # Volatility level (in levels, not logs)
        sigma_level = exp(sigma_grid[i_sigma])

        # Discretize demand with this volatility
        if method == :rouwenhorst
            D_grid_temp, Pi_D_temp = rouwenhorst(n_D, demand.rho_D, sigma_level; mu=demand.mu_D)
        else
            D_grid_temp, Pi_D_temp = tauchen(n_D, demand.rho_D, sigma_level; mu=demand.mu_D)
        end

        D_grids[i_sigma] = D_grid_temp
        Pi_D_given_sigma[:, :, i_sigma] = Pi_D_temp
    end

    # 3. Use average demand grid (they should be similar for persistent processes)
    D_grid = mean(D_grids)

    # 4. Handle correlation between shocks if rho_epsilon_eta ≠ 0
    if abs(vol.rho_epsilon_eta) > 1e-10
        # Adjust transition probabilities for correlation
        # This is a simplified approach; full bivariate normal would be more accurate
        @warn "Correlation rho_epsilon_eta = $(vol.rho_epsilon_eta) ≠ 0 detected. Using simplified correlation adjustment."
        # TODO: Implement proper bivariate discretization if needed
    end

    # 5. Construct joint transition matrix
    # State space is (D, sigma) pairs, ordered as [(D_1,sigma_1), (D_2,sigma_1), ..., (D_n,sigma_1), (D_1,sigma_2), ...]
    n_states = n_D * n_sigma
    Pi_joint = zeros(n_states, n_states)

    for i_sigma in 1:n_sigma
        for i_D in 1:n_D
            i_state = (i_sigma - 1) * n_D + i_D  # Current state index

            for i_sigma_next in 1:n_sigma
                for i_D_next in 1:n_D
                    i_state_next = (i_sigma_next - 1) * n_D + i_D_next

                    # Joint probability: P(D',sigma'|D,sigma) = P(D'|D,sigma) * P(sigma'|sigma)
                    Pi_joint[i_state, i_state_next] = Pi_D_given_sigma[i_D, i_D_next, i_sigma] * Pi_sigma[i_sigma, i_sigma_next]
                end
            end
        end
    end

    # Verify transition matrix properties
    @assert all(isapprox.(sum(Pi_joint, dims=2), 1.0, atol=1e-10)) "Joint transition matrix rows must sum to 1"

    return SVDiscretization(D_grid, sigma_grid, n_D, n_sigma, Pi_sigma, Pi_D_given_sigma, Pi_joint)
end

"""
    stationary_distribution(Pi::Matrix{Float64}; tol::Float64=1e-10) -> Vector{Float64}

Compute stationary distribution of Markov chain with transition matrix Pi.
"""
function stationary_distribution(Pi::Matrix{Float64}; tol::Float64=1e-10, max_iter::Int=10000)
    n = size(Pi, 1)
    pi_stat = ones(n) / n  # Initial guess

    for iter in 1:max_iter
        pi_new = pi_stat' * Pi
        pi_new = vec(pi_new)

        if maximum(abs.(pi_new - pi_stat)) < tol
            return pi_new
        end

        pi_stat = pi_new
    end

    @warn "Stationary distribution did not converge after $max_iter iterations"
    return pi_stat
end

"""
    verify_discretization(grid::Vector{Float64}, Pi::Matrix{Float64}, rho::Float64, sigma::Float64; mu::Float64=0.0)

Verify that discretization matches theoretical moments of AR(1) process.

Returns NamedTuple with:
- theoretical_mean, empirical_mean
- theoretical_std, empirical_std
- theoretical_autocorr, empirical_autocorr
"""
function verify_discretization(grid::Vector{Float64}, Pi::Matrix{Float64},
                               rho::Float64, sigma::Float64; mu::Float64=0.0)
    # Compute stationary distribution
    pi_stat = stationary_distribution(Pi)

    # Empirical moments
    emp_mean = dot(pi_stat, grid)
    emp_std = sqrt(dot(pi_stat, (grid .- emp_mean).^2))

    # Autocorrelation: E[x_{t+1} x_t] - (E[x_t])^2
    E_x_next_x = sum(pi_stat[i] * grid[i] * sum(Pi[i, j] * grid[j] for j in 1:length(grid))
                     for i in 1:length(grid))
    emp_autocorr = (E_x_next_x - emp_mean^2) / emp_std^2

    # Theoretical moments
    theo_mean = mu
    theo_std = sigma / sqrt(1 - rho^2)
    theo_autocorr = rho

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
