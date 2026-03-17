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

Grid values are stored in the process's native space (`:log` or `:level`),
as indicated by `D_space` and `sigma_space`. Use `get_D_levels(sv)` and
`get_sigma_levels(sv)` to obtain values in levels for profit computation.
"""
struct SVDiscretization
    D_grid::Vector{Float64}      # Demand grid (in D_space)
    sigma_grid::Vector{Float64}      # Volatility grid (in sigma_space)
    D_space::Symbol              # :log or :level — space of D_grid values
    sigma_space::Symbol          # :log or :level — space of sigma_grid values
    n_D::Int
    n_sigma::Int
    Pi_sigma::Matrix{Float64}         # Volatility transition P(sigma'|sigma)
    Pi_D_given_sigma::Array{Float64,3}  # Demand transition P(D'|D,sigma) for each sigma
    Pi_joint::Matrix{Float64}     # Joint transition on (D,sigma) pairs
end

"""
    get_D_levels(sv::SVDiscretization) -> Vector{Float64}

Get demand grid values in levels (for profit computation).
Applies exp() when D_space is :log, returns grid directly when :level.
"""
get_D_levels(sv::SVDiscretization) = sv.D_space == :log ? exp.(sv.D_grid) : sv.D_grid

"""
    get_sigma_levels(sv::SVDiscretization) -> Vector{Float64}

Get volatility grid values in levels (demand innovation std devs).
Applies exp() when sigma_space is :log, returns grid directly when :level.
"""
get_sigma_levels(sv::SVDiscretization) = sv.sigma_space == :log ? exp.(sv.sigma_grid) : sv.sigma_grid

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

    discretize_fn = method == :rouwenhorst ? rouwenhorst : tauchen

    # 1. Discretize volatility process (independent)
    sigma_grid, Pi_sigma = discretize_fn(n_sigma, vol.rho_sigma, vol.sigma_eta; mu=vol.sigma_bar)

    # 2. Discretize demand process for each volatility level
    Pi_D_given_sigma = zeros(n_D, n_D, n_sigma)
    D_grids = Vector{Vector{Float64}}(undef, n_sigma)

    for i_sigma in 1:n_sigma
        # Get volatility level (demand innovation std dev)
        if vol.process_space == :log
            sigma_level = exp(sigma_grid[i_sigma])
        else
            sigma_level = max(sigma_grid[i_sigma], 1e-10)
        end

        # Discretize demand with this volatility
        D_grid_temp, Pi_D_temp = discretize_fn(n_D, demand.rho_D, sigma_level; mu=demand.mu_D)

        D_grids[i_sigma] = D_grid_temp
        Pi_D_given_sigma[:, :, i_sigma] = Pi_D_temp
    end

    # 3. Use average demand grid (they should be similar for persistent processes)
    D_grid = mean(D_grids)

    # 4. Handle correlation between shocks if rho_epsilon_eta ≠ 0
    if abs(vol.rho_epsilon_eta) > 1e-10
        @warn "Correlation rho_epsilon_eta = $(vol.rho_epsilon_eta) ≠ 0 detected. Using simplified correlation adjustment."
        # TODO: Implement proper bivariate discretization if needed
    end

    # 5. Construct joint transition matrix
    Pi_joint = _build_joint_transition(Pi_D_given_sigma, Pi_sigma, n_D, n_sigma)

    return SVDiscretization(D_grid, sigma_grid, demand.process_space, vol.process_space,
                           n_D, n_sigma, Pi_sigma, Pi_D_given_sigma, Pi_joint)
end

"""
    discretize_sv_process(demand::DemandProcess, vol::TwoStateVolatility,
                          n_D::Int, n_sigma::Int; method::Symbol=:rouwenhorst)

Discretize demand process with two-state Markov switching volatility.
The `n_sigma` argument is ignored (forced to 2).
"""
function discretize_sv_process(demand::DemandProcess, vol::TwoStateVolatility,
                               n_D::Int, n_sigma::Int; method::Symbol=:rouwenhorst)
    @assert method in [:rouwenhorst, :tauchen] "Method must be :rouwenhorst or :tauchen"

    if n_sigma != 2
        @warn "TwoStateVolatility forces n_sigma=2 (was $n_sigma in NumericalSettings)"
    end

    discretize_fn = method == :rouwenhorst ? rouwenhorst : tauchen

    # Volatility grid and transitions are given directly
    sigma_grid = vol.sigma_levels
    Pi_sigma = vol.Pi_sigma
    actual_n_sigma = 2

    # Get volatility levels for demand discretization
    if vol.process_space == :log
        sigma_levels_for_demand = exp.(vol.sigma_levels)
    else
        sigma_levels_for_demand = vol.sigma_levels
    end

    # Discretize demand for each volatility state
    Pi_D_given_sigma = zeros(n_D, n_D, actual_n_sigma)
    D_grids = Vector{Vector{Float64}}(undef, actual_n_sigma)

    for i_sigma in 1:actual_n_sigma
        sigma_level = max(sigma_levels_for_demand[i_sigma], 1e-10)
        D_grid_temp, Pi_D_temp = discretize_fn(n_D, demand.rho_D, sigma_level; mu=demand.mu_D)
        D_grids[i_sigma] = D_grid_temp
        Pi_D_given_sigma[:, :, i_sigma] = Pi_D_temp
    end

    # Average demand grid across volatility states
    D_grid = mean(D_grids)

    # Construct joint transition matrix
    Pi_joint = _build_joint_transition(Pi_D_given_sigma, Pi_sigma, n_D, actual_n_sigma)

    return SVDiscretization(D_grid, sigma_grid, demand.process_space, vol.process_space,
                           n_D, actual_n_sigma, Pi_sigma, Pi_D_given_sigma, Pi_joint)
end

"""
    _build_joint_transition(Pi_D_given_sigma, Pi_sigma, n_D, n_sigma) -> Matrix{Float64}

Build joint transition matrix P(D',σ'|D,σ) = P(D'|D,σ) × P(σ'|σ).
State ordering: [(D_1,σ_1), (D_2,σ_1), ..., (D_{n_D},σ_1), (D_1,σ_2), ...].
"""
function _build_joint_transition(Pi_D_given_sigma::Array{Float64,3},
                                 Pi_sigma::Matrix{Float64},
                                 n_D::Int, n_sigma::Int)
    n_states = n_D * n_sigma
    Pi_joint = zeros(n_states, n_states)

    for i_sigma in 1:n_sigma
        for i_D in 1:n_D
            i_state = (i_sigma - 1) * n_D + i_D

            for i_sigma_next in 1:n_sigma
                for i_D_next in 1:n_D
                    i_state_next = (i_sigma_next - 1) * n_D + i_D_next
                    Pi_joint[i_state, i_state_next] = Pi_D_given_sigma[i_D, i_D_next, i_sigma] * Pi_sigma[i_sigma, i_sigma_next]
                end
            end
        end
    end

    @assert all(isapprox.(sum(Pi_joint, dims=2), 1.0, atol=1e-10)) "Joint transition matrix rows must sum to 1"
    return Pi_joint
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
