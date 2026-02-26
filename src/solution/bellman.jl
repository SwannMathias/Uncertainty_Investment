"""
Bellman operators for the nested dynamic programming problem.

Timeline within year t:
1. Beginning of year: Observe (K_t, D_t, sigma_t), choose I_t
2. Mid-year: Observe (D_{t+1/2}, sigma_{t+1/2}), choose Delta_I_t
3. End of year: K_{t+1} = (1-delta)K_t + I_t + Delta_I_t

Value functions:
- V(K, D, sigma): Beginning-of-year value
- W(K', D, sigma): Mid-year continuation value (after initial investment)

Performance optimization:
- Profit function values are precomputed for all (K, D) grid points
- Uses get_profit(grids, i_K, i_D) for O(1) lookups instead of function calls
- For off-grid K, uses get_profit_at_K(grids, K, i_D) with interpolation
"""

"""
    solve_midyear_problem(K_prime::Float64, i_D_half::Int, i_sigma_half::Int,
                          i_K::Int, K_current::Float64, I_initial::Float64,
                          V::Array{Float64,3}, grids::StateGrids,
                          params::ModelParameters, ac::AbstractAdjustmentCost,
                          derived::DerivedParameters) -> (Float64, Float64)

Solve mid-year problem: choose Delta_I to maximize expected value.

Given:
- K_prime: Capital after initial investment I (before Delta_I)
- (D_half, sigma_half): Mid-year realizations
- i_K: Capital grid index (for precomputed profit lookup)
- K_current: Beginning-of-year capital value
- I_initial: Initial investment chosen at beginning of year

Choose Delta_I to maximize:
    pi(K_current, D_half) - C_2(Delta_I, K_current) + beta_semester * E[V(K'', D', sigma') | D_half, sigma_half]

where K'' = K_prime + Delta_I.

# Returns
- Delta_I_opt: Optimal investment revision
- value: Maximized value

# Performance
Uses precomputed profits via get_profit(grids, i_K, i_D_half) for O(1) lookup.
"""
function solve_midyear_problem(K_prime::Float64, i_D_half::Int, i_sigma_half::Int,
                               i_K::Int, K_current::Float64, I_initial::Float64,
                               V::Array{Float64,3}, grids::StateGrids,
                               params::ModelParameters, ac::AbstractAdjustmentCost,
                               derived::DerivedParameters)
    # Mid-year profit (operating on current capital) - use precomputed value
    pi_half = get_profit(grids, i_K, i_D_half)

    # Expected value over next year's states
    EV = compute_expectation(grids, V, i_D_half, i_sigma_half; horizon=:semester)

    # Objective function: maximize over Delta_I
    function obj_Delta_I(Delta_I)
        K_double_prime = K_prime + Delta_I

        # Check feasibility
        if K_double_prime < grids.K_min
            return -Inf
        end
        if K_double_prime > grids.K_max
            return -1e10  # Penalty but not -Inf
        end

        # Adjustment cost (mid-year component)
        # For SeparateConvexCost, only charge the revision cost (phi_2) here,
        # since the initial cost (phi_1) was already charged in the beginning-of-year problem
        if ac isa SeparateConvexCost
            cost = 0.5 * ac.phi_2 * (Delta_I / K_current)^2 * K_current
        else
            # For other cost types, compute the full cost
            cost = compute_cost(ac, I_initial, Delta_I, K_current)
        end

        # Interpolate expected value
        EV_interp = linear_interp_1d(grids.K_grid, EV, K_double_prime)

        # Total value
        return -cost + derived.beta_semester * EV_interp
    end

    # Determine search bounds for Delta_I
    # Capital constraint: K_min <= K_prime + Delta_I <= K_max
    Delta_I_min = grids.K_min - K_prime
    Delta_I_max = grids.K_max - K_prime

    # Additional constraint: K'' must be positive
    Delta_I_min = max(Delta_I_min, -K_prime + 1e-6)

    # If no adjustment costs, analytical solution from FOC
    if ac isa NoAdjustmentCost
        # FOC: beta * ∂EV/∂K = 0 => Choose K'' to maximize EV
        # This is equivalent to choosing Delta_I to maximize EV(K'')
        # Use simple grid search
        EV_on_grid = derived.beta_semester .* EV
        i_K_opt = argmax(EV_on_grid)
        K_opt = grids.K_grid[i_K_opt]
        Delta_I_opt = K_opt - K_prime

        # Ensure within bounds
        Delta_I_opt = clamp(Delta_I_opt, Delta_I_min, Delta_I_max)

        value = pi_half + obj_Delta_I(Delta_I_opt)
        return Delta_I_opt, value
    end

    # With adjustment costs, need to optimize
    if has_fixed_cost(ac)
        # Discrete choice: adjust or not
        # Option 1: No adjustment (Delta_I = 0)
        value_no_adjust = pi_half + obj_Delta_I(0.0)

        # Option 2: Adjust optimally
        if Delta_I_min < -1e-10 || Delta_I_max > 1e-10  # Can actually adjust
            Delta_I_opt_adjust, val_adjust = maximize_univariate(obj_Delta_I, Delta_I_min, Delta_I_max; tol=1e-6)
            value_adjust = pi_half + val_adjust

            if value_adjust > value_no_adjust
                return Delta_I_opt_adjust, value_adjust
            else
                return 0.0, value_no_adjust
            end
        else
            return 0.0, value_no_adjust
        end
    else
        # Continuous optimization
        Delta_I_opt, val = maximize_univariate(obj_Delta_I, Delta_I_min, Delta_I_max; tol=1e-6)
        value = pi_half + val
        return Delta_I_opt, value
    end
end

"""
    compute_midyear_continuation(K_prime::Float64, i_D::Int, i_sigma::Int,
                                  i_K::Int, K_current::Float64, I_initial::Float64,
                                  V::Array{Float64,3}, grids::StateGrids,
                                  params::ModelParameters, ac::AbstractAdjustmentCost,
                                  derived::DerivedParameters) -> Float64

Compute W(K', D, sigma): expected value of mid-year problem.

W(K', D, sigma) = E_{D_half, sigma_half | D, sigma}[max_Delta_I {...}]

# Arguments
- i_K: Capital grid index for precomputed profit lookup

# Returns
- Expected mid-year continuation value
"""
function compute_midyear_continuation(K_prime::Float64, i_D::Int, i_sigma::Int,
                                      i_K::Int, K_current::Float64, I_initial::Float64,
                                      V::Array{Float64,3}, grids::StateGrids,
                                      params::ModelParameters, ac::AbstractAdjustmentCost,
                                      derived::DerivedParameters)
    # Get transition probability from (D, sigma) to (D_half, sigma_half)
    i_state = get_joint_state_index(grids, i_D, i_sigma)

    W_value = 0.0

    # Expectation over mid-year states
    for i_state_half in 1:grids.n_states
        i_D_half, i_sigma_half = get_D_sigma_indices(grids, i_state_half)

        # Solve mid-year problem for this realization (using precomputed profits)
        Delta_I_opt, value_half = solve_midyear_problem(
            K_prime, i_D_half, i_sigma_half, i_K, K_current, I_initial,
            V, grids, params, ac, derived
        )

        # Weight by probability
        prob = grids.Pi_semester[i_state, i_state_half]
        W_value += prob * value_half
    end

    return W_value
end

"""
    solve_beginning_year_problem(i_K::Int, i_D::Int, i_sigma::Int,
                                  V::Array{Float64,3}, grids::StateGrids,
                                  params::ModelParameters, ac::AbstractAdjustmentCost,
                                  derived::DerivedParameters) -> (Float64, Float64)

Solve beginning-of-year problem: choose I to maximize value.

V(K, D, sigma) = max_I { pi(K, D) - C_1(I, K) + E[W(K', D, sigma) | D, sigma] }

where K' = (1-delta)K + I.

# Returns
- I_opt: Optimal initial investment
- V_value: Maximized value

# Performance
Uses precomputed profits via get_profit(grids, i_K, i_D) for O(1) lookup.
"""
function solve_beginning_year_problem(i_K::Int, i_D::Int, i_sigma::Int,
                                      V::Array{Float64,3}, grids::StateGrids,
                                      params::ModelParameters, ac::AbstractAdjustmentCost,
                                      derived::DerivedParameters)
    # Current state
    K = get_K(grids, i_K)

    # First-semester profit - use precomputed value
    pi_first = get_profit(grids, i_K, i_D)

    # Objective function: maximize over I
    function obj_I(I)
        K_prime = (1 - derived.delta_semester) * K + I

        # Check feasibility
        if K_prime < grids.K_min
            return -Inf
        end
        if K_prime > grids.K_max
            return -1e10
        end

        # Initial adjustment cost
        # Note: For most cost types, this depends on I only (Delta_I = 0 at this stage)
        # Exception: ConvexAdjustmentCost depends on total I + Delta_I,
        # but here we're just making initial decision, so we use a placeholder
        # The full cost will be computed in mid-year problem

        # For separate costs, only charge C_1(I, K)
        if ac isa SeparateConvexCost
            cost = 0.5 * ac.phi_1 * (I / K)^2 * K
        elseif ac isa NoAdjustmentCost
            cost = 0.0
        elseif ac isa ConvexAdjustmentCost
            # For standard convex, we need to anticipate Delta_I
            # Simplified: assume Delta_I = 0 for now (will be optimized in mid-year)
            cost = 0.5 * ac.phi * (I / K)^2 * K
        else
            # For other types, compute cost assuming Delta_I = 0
            cost = compute_cost(ac, I, 0.0, K)
        end

        # Continuation value (pass i_K for precomputed profit access)
        W_value = compute_midyear_continuation(
            K_prime, i_D, i_sigma, i_K, K, I, V, grids, params, ac, derived
        )

        return -cost + W_value
    end

    # Determine search bounds for I
    # Constraint: K_min <= (1-delta)K + I <= K_max
    I_min = grids.K_min - (1 - derived.delta_semester) * K
    I_max = grids.K_max - (1 - derived.delta_semester) * K

    # Ensure K_prime > 0
    I_min = max(I_min, -(1 - derived.delta_semester) * K + 1e-6)

    # Optimize
    if ac isa NoAdjustmentCost && !has_fixed_cost(ac)
        # Can use coarser search for faster convergence
        I_opt, val = maximize_univariate(obj_I, I_min, I_max; method=:brent, tol=1e-5)
    elseif has_fixed_cost(ac)
        # Discrete choice
        value_no_invest = pi_first + obj_I(0.0)

        if I_min < -1e-10 || I_max > 1e-10
            I_opt_invest, val_invest = maximize_univariate(obj_I, I_min, I_max; tol=1e-6)
            value_invest = pi_first + val_invest

            if value_invest > value_no_invest
                I_opt = I_opt_invest
                val = val_invest
            else
                I_opt = 0.0
                val = value_no_invest - pi_first
            end
        else
            I_opt = 0.0
            val = value_no_invest - pi_first
        end
    else
        I_opt, val = maximize_univariate(obj_I, I_min, I_max; tol=1e-6)
    end

    V_value = pi_first + val

    return I_opt, V_value
end

"""
    bellman_operator!(V_new::Array{Float64,3}, V::Array{Float64,3},
                      I_policy::Array{Float64,3}, grids::StateGrids,
                      params::ModelParameters, ac::AbstractAdjustmentCost,
                      derived::DerivedParameters) -> Nothing

Apply Bellman operator: V_new = T(V).

Updates V_new and I_policy in-place.

For each state (K, D, sigma):
1. Solve beginning-of-year problem to get I_opt and V_new
2. Store optimal policy I_policy[i_K, i_D, i_sigma] = I_opt
3. Store value V_new[i_K, i_D, i_sigma] = V_value
"""
function bellman_operator!(V_new::Array{Float64,3}, V::Array{Float64,3},
                          I_policy::Array{Float64,3}, grids::StateGrids,
                          params::ModelParameters, ac::AbstractAdjustmentCost,
                          derived::DerivedParameters)
    # Loop over all states
    for i_sigma in 1:grids.n_sigma
        for i_D in 1:grids.n_D
            for i_K in 1:grids.n_K
                I_opt, V_value = solve_beginning_year_problem(
                    i_K, i_D, i_sigma, V, grids, params, ac, derived
                )

                V_new[i_K, i_D, i_sigma] = V_value
                I_policy[i_K, i_D, i_sigma] = I_opt
            end
        end
    end

    return nothing
end

"""
    bellman_operator_no_ac!(V_new::Array{Float64,3}, V::Array{Float64,3},
                           I_policy::Array{Float64,3}, grids::StateGrids,
                           params::ModelParameters, derived::DerivedParameters) -> Nothing

Simplified Bellman operator for no adjustment costs case.

This is more efficient than the general case.
"""
function bellman_operator_no_ac!(V_new::Array{Float64,3}, V::Array{Float64,3},
                                I_policy::Array{Float64,3}, grids::StateGrids,
                                params::ModelParameters, derived::DerivedParameters)
    ac = NoAdjustmentCost()
    bellman_operator!(V_new, V, I_policy, grids, params, ac, derived)
    return nothing
end

"""
    howard_improvement_step!(V::Array{Float64,3}, I_policy::Array{Float64,3},
                            grids::StateGrids, params::ModelParameters,
                            ac::AbstractAdjustmentCost, derived::DerivedParameters,
                            n_steps::Int) -> Nothing

Perform Howard improvement (policy iteration) steps.

Given fixed policy I_policy, update value function by iterating:
V^{k+1} = pi + Gamma(I_policy) V^k

where Gamma is the transition operator under policy I_policy.

This accelerates VFI convergence.
"""
function howard_improvement_step!(V::Array{Float64,3}, I_policy::Array{Float64,3},
                                 grids::StateGrids, params::ModelParameters,
                                 ac::AbstractAdjustmentCost, derived::DerivedParameters,
                                 n_steps::Int)
    V_temp = copy(V)

    for step in 1:n_steps
        for i_sigma in 1:grids.n_sigma
            for i_D in 1:grids.n_D
                for i_K in 1:grids.n_K
                    K = get_K(grids, i_K)

                    # Fixed policy
                    I = I_policy[i_K, i_D, i_sigma]

                    # First-semester profit - use precomputed value
                    pi_first = get_profit(grids, i_K, i_D)

                    # Capital after initial investment
                    K_prime = (1 - derived.delta_semester) * K + I

                    # Initial cost
                    if ac isa SeparateConvexCost
                        cost_I = 0.5 * ac.phi_1 * (I / K)^2 * K
                    else
                        cost_I = compute_cost(ac, I, 0.0, K)
                    end

                    # Mid-year continuation (using current V, pass i_K for precomputed profits)
                    W_val = compute_midyear_continuation(
                        K_prime, i_D, i_sigma, i_K, K, I, V_temp, grids, params, ac, derived
                    )

                    # Update value
                    V[i_K, i_D, i_sigma] = pi_first - cost_I + W_val
                end
            end
        end

        V_temp .= V
    end

    return nothing
end

# ============================================================================
# Parallelized Bellman Operators
# ============================================================================

"""
    bellman_operator_parallel!(V_new::Array{Float64,3}, V::Array{Float64,3},
                               I_policy::Array{Float64,3}, grids::StateGrids,
                               params::ModelParameters, ac::AbstractAdjustmentCost,
                               derived::DerivedParameters) -> Nothing

Parallel version of the Bellman operator using multi-threading.

Parallelization strategy:
- Flatten the 3D state space into 1D indices
- Distribute work across threads using @threads
- Each thread independently solves optimization problems for its assigned states
- No race conditions since each (i_K, i_D, i_sigma) writes to a unique location

Thread safety:
- V (input) is read-only during the operator
- V_new and I_policy have independent write locations per state
- All temporary allocations happen on the stack within each thread

Performance notes:
- Near-linear speedup expected up to number of physical cores
- Overhead is minimal for large state spaces (>1000 states)
- For small state spaces, serial may be faster due to threading overhead
"""
function bellman_operator_parallel!(V_new::Array{Float64,3}, V::Array{Float64,3},
                                    I_policy::Array{Float64,3}, grids::StateGrids,
                                    params::ModelParameters, ac::AbstractAdjustmentCost,
                                    derived::DerivedParameters)
    n_K = grids.n_K
    n_D = grids.n_D
    n_sigma = grids.n_sigma
    n_total = n_K * n_D * n_sigma

    # Parallelize over flattened state space
    # Each thread handles a subset of states independently
    @threads for idx in 1:n_total
        # Convert linear index to 3D indices (column-major order)
        # Julia arrays are column-major: [i_K, i_D, i_sigma]
        i_K = ((idx - 1) % n_K) + 1
        temp = (idx - 1) ÷ n_K
        i_D = (temp % n_D) + 1
        i_sigma = (temp ÷ n_D) + 1

        # Solve optimization for this state
        I_opt, V_value = solve_beginning_year_problem(
            i_K, i_D, i_sigma, V, grids, params, ac, derived
        )

        # Store results (no race condition - each idx writes to unique location)
        V_new[i_K, i_D, i_sigma] = V_value
        I_policy[i_K, i_D, i_sigma] = I_opt
    end

    return nothing
end

"""
    howard_improvement_step_parallel!(V::Array{Float64,3}, I_policy::Array{Float64,3},
                                      grids::StateGrids, params::ModelParameters,
                                      ac::AbstractAdjustmentCost, derived::DerivedParameters,
                                      n_steps::Int) -> Nothing

Parallel version of Howard improvement (policy iteration) steps.

Each step updates the value function for all states in parallel,
given the fixed policy I_policy.

Thread safety:
- V_temp is read-only during each step
- V has independent write locations per state
- Synchronization happens between steps (implicit barrier at end of @threads)
"""
function howard_improvement_step_parallel!(V::Array{Float64,3}, I_policy::Array{Float64,3},
                                           grids::StateGrids, params::ModelParameters,
                                           ac::AbstractAdjustmentCost, derived::DerivedParameters,
                                           n_steps::Int)
    n_K = grids.n_K
    n_D = grids.n_D
    n_sigma = grids.n_sigma
    n_total = n_K * n_D * n_sigma

    V_temp = copy(V)

    for step in 1:n_steps
        # Parallelize over flattened state space
        @threads for idx in 1:n_total
            # Convert linear index to 3D indices
            i_K = ((idx - 1) % n_K) + 1
            temp = (idx - 1) ÷ n_K
            i_D = (temp % n_D) + 1
            i_sigma = (temp ÷ n_D) + 1

            K = get_K(grids, i_K)

            # Fixed policy
            I = I_policy[i_K, i_D, i_sigma]

            # First-semester profit - use precomputed value
            pi_first = get_profit(grids, i_K, i_D)

            # Capital after initial investment
            K_prime = (1 - derived.delta_semester) * K + I

            # Initial cost
            if ac isa SeparateConvexCost
                cost_I = 0.5 * ac.phi_1 * (I / K)^2 * K
            else
                cost_I = compute_cost(ac, I, 0.0, K)
            end

            # Mid-year continuation (using V_temp which is read-only this step, pass i_K for precomputed profits)
            W_val = compute_midyear_continuation(
                K_prime, i_D, i_sigma, i_K, K, I, V_temp, grids, params, ac, derived
            )

            # Update value (no race condition)
            V[i_K, i_D, i_sigma] = pi_first - cost_I + W_val
        end

        # Synchronization point: copy updated V to V_temp for next step
        V_temp .= V
    end

    return nothing
end
