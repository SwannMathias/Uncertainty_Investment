"""
State space construction combining capital grid and stochastic processes.

StateGrids structure contains:
- Capital grid (deterministic, continuous state)
- Demand and volatility grids (discrete, stochastic states)
- Transition matrices for expectations
"""

"""
    StateGrids

Complete state space for the dynamic programming problem.

State variables:
- K: Capital (continuous, discretized)
- D: Demand (stochastic, discrete)
- sigma: Volatility (stochastic, discrete)
"""
struct StateGrids
    # Capital grid
    K_grid::Vector{Float64}
    n_K::Int
    K_min::Float64
    K_max::Float64

    # Stochastic processes
    sv::SVDiscretization
    n_D::Int
    n_sigma::Int

    # Transition matrices
    Pi_semester::Matrix{Float64}  # One-semester transition
    Pi_year::Matrix{Float64}      # Full-year transition (two semesters)

    # Total state space size
    n_states::Int
end

"""
    construct_grids(params::ModelParameters) -> StateGrids

Construct complete state space grids from model parameters.

# Arguments
- `params`: ModelParameters struct

# Returns
- StateGrids object containing all discretized state spaces
"""
function construct_grids(params::ModelParameters)
    derived = get_derived_parameters(params)

    # 1. Capital grid (logarithmically spaced for better resolution near K_ss)
    K_min = params.numerical.K_min_factor * derived.K_ss
    K_max = params.numerical.K_max_factor * derived.K_ss
    n_K = params.numerical.n_K

    # Use log-spacing for better resolution at low K
    K_grid_log = range(log(K_min), log(K_max), length=n_K)
    K_grid = exp.(K_grid_log)

    # 2. Discretize stochastic processes
    sv = discretize_sv_process(params.demand, params.volatility,
                               params.numerical.n_D, params.numerical.n_sigma)

    # 3. Transition matrices
    Pi_semester = sv.Pi_joint
    Pi_year = Pi_semester * Pi_semester  # Two semesters = one year

    # 4. Verify Pi_year is still a valid transition matrix
    @assert all(isapprox.(sum(Pi_year, dims=2), 1.0, atol=1e-10)) "Year transition matrix rows must sum to 1"

    n_states = sv.n_D * sv.n_sigma

    return StateGrids(
        K_grid, n_K, K_min, K_max,
        sv, params.numerical.n_D, params.numerical.n_sigma,
        Pi_semester, Pi_year,
        n_states
    )
end

# =============================================================================
# Accessor functions
# =============================================================================

"""
    get_K(grids::StateGrids, i_K::Int) -> Float64

Get capital level at index i_K.
"""
get_K(grids::StateGrids, i_K::Int) = grids.K_grid[i_K]

"""
    get_log_D(grids::StateGrids, i_D::Int) -> Float64

Get log demand at index i_D.
"""
get_log_D(grids::StateGrids, i_D::Int) = grids.sv.D_grid[i_D]

"""
    get_D(grids::StateGrids, i_D::Int) -> Float64

Get demand level (not log) at index i_D.
"""
get_D(grids::StateGrids, i_D::Int) = exp(grids.sv.D_grid[i_D])

"""
    get_log_sigma(grids::StateGrids, i_sigma::Int) -> Float64

Get log volatility at index i_sigma.
"""
get_log_sigma(grids::StateGrids, i_sigma::Int) = grids.sv.sigma_grid[i_sigma]

"""
    get_sigma(grids::StateGrids, i_sigma::Int) -> Float64

Get volatility level (not log) at index i_sigma.
"""
get_sigma(grids::StateGrids, i_sigma::Int) = exp(grids.sv.sigma_grid[i_sigma])

"""
    get_joint_state_index(grids::StateGrids, i_D::Int, i_sigma::Int) -> Int

Convert (D, sigma) indices to joint state index.
State ordering: [(D_1,sigma_1), (D_2,sigma_1), ..., (D_n_D,sigma_1), (D_1,sigma_2), ...]
"""
get_joint_state_index(grids::StateGrids, i_D::Int, i_sigma::Int) = (i_sigma - 1) * grids.n_D + i_D

"""
    get_D_sigma_indices(grids::StateGrids, i_state::Int) -> (Int, Int)

Convert joint state index to (i_D, i_sigma) indices.
"""
function get_D_sigma_indices(grids::StateGrids, i_state::Int)
    i_sigma = div(i_state - 1, grids.n_D) + 1
    i_D = mod(i_state - 1, grids.n_D) + 1
    return i_D, i_sigma
end

# =============================================================================
# Interpolation helpers
# =============================================================================

"""
    find_K_bracket(grids::StateGrids, K::Float64) -> (Int, Int, Float64)

Find bracket indices and interpolation weight for capital K.

Returns:
- i_low: Lower bracket index
- i_high: Upper bracket index
- weight: Weight on upper bracket (0 if K ≤ K_grid[1], 1 if K ≥ K_grid[end])
"""
function find_K_bracket(grids::StateGrids, K::Float64)
    # Handle boundary cases
    if K <= grids.K_grid[1]
        return 1, 1, 0.0
    elseif K >= grids.K_grid[end]
        return grids.n_K, grids.n_K, 1.0
    end

    # Binary search for bracket
    i_low = searchsortedlast(grids.K_grid, K)
    i_high = i_low + 1

    # Linear interpolation weight
    K_low = grids.K_grid[i_low]
    K_high = grids.K_grid[i_high]
    weight = (K - K_low) / (K_high - K_low)

    return i_low, i_high, weight
end

"""
    interpolate_value(grids::StateGrids, V::Array{Float64,3},
                      K::Float64, i_D::Int, i_sigma::Int) -> Float64

Interpolate value function at (K, D, sigma) using linear interpolation in K.

# Arguments
- `grids`: StateGrids
- `V`: Value function array V[i_K, i_D, i_sigma]
- `K`: Capital level (not on grid)
- `i_D`: Demand state index
- `i_sigma`: Volatility state index

# Returns
- Interpolated value
"""
function interpolate_value(grids::StateGrids, V::Array{Float64,3},
                          K::Float64, i_D::Int, i_sigma::Int)
    i_low, i_high, weight = find_K_bracket(grids, K)

    if i_low == i_high
        return V[i_low, i_D, i_sigma]
    else
        V_low = V[i_low, i_D, i_sigma]
        V_high = V[i_high, i_D, i_sigma]
        return (1 - weight) * V_low + weight * V_high
    end
end

"""
    interpolate_policy(grids::StateGrids, I_policy::Array{Float64,3},
                       K::Float64, i_D::Int, i_sigma::Int) -> Float64

Interpolate policy function at (K, D, sigma) using linear interpolation in K.
"""
function interpolate_policy(grids::StateGrids, I_policy::Array{Float64,3},
                           K::Float64, i_D::Int, i_sigma::Int)
    i_low, i_high, weight = find_K_bracket(grids, K)

    if i_low == i_high
        return I_policy[i_low, i_D, i_sigma]
    else
        I_low = I_policy[i_low, i_D, i_sigma]
        I_high = I_policy[i_high, i_D, i_sigma]
        return (1 - weight) * I_low + weight * I_high
    end
end

# =============================================================================
# Expectation computation
# =============================================================================

"""
    compute_expectation(grids::StateGrids, V::Array{Float64,3},
                        i_D::Int, i_sigma::Int; horizon::Symbol=:semester) -> Vector{Float64}

Compute expected value over stochastic states for each capital level.

E[V(K', D', sigma') | D, sigma]

# Arguments
- `grids`: StateGrids
- `V`: Value function array V[i_K, i_D, i_sigma]
- `i_D`: Current demand state
- `i_sigma`: Current volatility state
- `horizon`: :semester or :year (uses Pi_semester or Pi_year)

# Returns
- Vector of length n_K containing E[V(K', D', sigma') | D, sigma] for each K'
"""
function compute_expectation(grids::StateGrids, V::Array{Float64,3},
                            i_D::Int, i_sigma::Int; horizon::Symbol=:semester)
    @assert horizon in [:semester, :year] "horizon must be :semester or :year"

    Pi = horizon == :semester ? grids.Pi_semester : grids.Pi_year
    i_state = get_joint_state_index(grids, i_D, i_sigma)

    EV = zeros(grids.n_K)

    for i_K_next in 1:grids.n_K
        ev = 0.0
        for i_state_next in 1:grids.n_states
            i_D_next, i_sigma_next = get_D_sigma_indices(grids, i_state_next)
            ev += Pi[i_state, i_state_next] * V[i_K_next, i_D_next, i_sigma_next]
        end
        EV[i_K_next] = ev
    end

    return EV
end

"""
    compute_conditional_expectation(grids::StateGrids, V::Array{Float64,3},
                                    i_D::Int, i_sigma::Int, i_K_next::Int;
                                    horizon::Symbol=:semester) -> Float64

Compute expected value for a specific next-period capital level.

E[V(K'[i_K_next], D', sigma') | D, sigma]
"""
function compute_conditional_expectation(grids::StateGrids, V::Array{Float64,3},
                                        i_D::Int, i_sigma::Int, i_K_next::Int;
                                        horizon::Symbol=:semester)
    @assert horizon in [:semester, :year] "horizon must be :semester or :year"

    Pi = horizon == :semester ? grids.Pi_semester : grids.Pi_year
    i_state = get_joint_state_index(grids, i_D, i_sigma)

    ev = 0.0
    for i_state_next in 1:grids.n_states
        i_D_next, i_sigma_next = get_D_sigma_indices(grids, i_state_next)
        ev += Pi[i_state, i_state_next] * V[i_K_next, i_D_next, i_sigma_next]
    end

    return ev
end

# =============================================================================
# Grid diagnostics
# =============================================================================

"""
    print_grid_info(grids::StateGrids)

Print summary information about the state space.
"""
function print_grid_info(grids::StateGrids)
    println("=" ^ 60)
    println("State Space Information")
    println("=" ^ 60)

    println("\nCapital Grid:")
    println("  Points: $(grids.n_K)")
    println("  Range: [$(round(grids.K_min, digits=4)), $(round(grids.K_max, digits=4))]")
    println("  K[1] = $(round(grids.K_grid[1], digits=4))")
    println("  K[end] = $(round(grids.K_grid[end], digits=4))")

    println("\nDemand Grid (log):")
    println("  Points: $(grids.n_D)")
    println("  Range: [$(round(grids.sv.D_grid[1], digits=4)), $(round(grids.sv.D_grid[end], digits=4))]")

    println("\nDemand Grid (level):")
    D_levels = exp.(grids.sv.D_grid)
    println("  Range: [$(round(D_levels[1], digits=4)), $(round(D_levels[end], digits=4))]")

    println("\nVolatility Grid (log):")
    println("  Points: $(grids.n_sigma)")
    println("  Range: [$(round(grids.sv.sigma_grid[1], digits=4)), $(round(grids.sv.sigma_grid[end], digits=4))]")

    println("\nVolatility Grid (level):")
    sigma_levels = exp.(grids.sv.sigma_grid)
    println("  Range: [$(round(sigma_levels[1], digits=4)), $(round(sigma_levels[end], digits=4))]")

    println("\nTotal State Space:")
    println("  K × D × sigma = $(grids.n_K) × $(grids.n_D) × $(grids.n_sigma) = $(grids.n_K * grids.n_D * grids.n_sigma)")
    println("  Joint (D,sigma) states: $(grids.n_states)")

    println("=" ^ 60)
end
