"""
Bellman operators for the nested dynamic programming problem.

Timeline within year t:
1. Beginning of year: Observe (K_t, D_t, σ_t), choose I_t
2. Mid-year: Observe (D_{t+1/2}, σ_{t+1/2}), choose ΔI_t
3. End of year: K_{t+1} = (1-δ)K_t + I_t + ΔI_t

Value functions:
- V(K, D, σ): Beginning-of-year value
- W(K', D, σ): Mid-year continuation value (after initial investment)
"""

"""
    solve_midyear_problem(K_prime::Float64, i_D_half::Int, i_σ_half::Int,
                          K_current::Float64, I_initial::Float64,
                          V::Array{Float64,3}, grids::StateGrids,
                          params::ModelParameters, ac::AbstractAdjustmentCost,
                          derived::DerivedParameters) -> (Float64, Float64)

Solve mid-year problem: choose ΔI to maximize expected value.

Given:
- K_prime: Capital after initial investment I (before ΔI)
- (D_half, σ_half): Mid-year realizations
- K_current: Beginning-of-year capital
- I_initial: Initial investment chosen at beginning of year

Choose ΔI to maximize:
    π(K_current, D_half) - C_2(ΔI, K_current) + β_semester * E[V(K'', D', σ') | D_half, σ_half]

where K'' = K_prime + ΔI.

# Returns
- ΔI_opt: Optimal investment revision
- value: Maximized value
"""
function solve_midyear_problem(K_prime::Float64, i_D_half::Int, i_σ_half::Int,
                               K_current::Float64, I_initial::Float64,
                               V::Array{Float64,3}, grids::StateGrids,
                               params::ModelParameters, ac::AbstractAdjustmentCost,
                               derived::DerivedParameters)
    # Get mid-year demand level
    D_half = get_D(grids, i_D_half)

    # Mid-year profit (operating on current capital)
    π_half = profit(K_current, D_half, derived)

    # Expected value over next year's states
    EV = compute_expectation(grids, V, i_D_half, i_σ_half; horizon=:semester)

    # Objective function: maximize over ΔI
    # Note: We need to account for BOTH adjustment costs if using SeparateConvexCost
    function obj_ΔI(ΔI)
        K_double_prime = K_prime + ΔI

        # Check feasibility
        if K_double_prime < grids.K_min
            return -Inf
        end
        if K_double_prime > grids.K_max
            return -1e10  # Penalty but not -Inf
        end

        # Adjustment cost (mid-year component)
        # For most cost types, we need total investment I + ΔI
        cost = compute_cost(ac, I_initial, ΔI, K_current)

        # Interpolate expected value
        EV_interp = linear_interp_1d(grids.K_grid, EV, K_double_prime)

        # Total value
        return -cost + derived.β_semester * EV_interp
    end

    # Determine search bounds for ΔI
    # Capital constraint: K_min <= K_prime + ΔI <= K_max
    ΔI_min = grids.K_min - K_prime
    ΔI_max = grids.K_max - K_prime

    # Additional constraint: K'' must be positive
    ΔI_min = max(ΔI_min, -K_prime + 1e-6)

    # If no adjustment costs, analytical solution from FOC
    if ac isa NoAdjustmentCost
        # FOC: β * ∂EV/∂K = 0 => Choose K'' to maximize EV
        # This is equivalent to choosing ΔI to maximize EV(K'')
        # Use simple grid search
        EV_on_grid = derived.β_semester .* EV
        i_K_opt = argmax(EV_on_grid)
        K_opt = grids.K_grid[i_K_opt]
        ΔI_opt = K_opt - K_prime

        # Ensure within bounds
        ΔI_opt = clamp(ΔI_opt, ΔI_min, ΔI_max)

        value = π_half + obj_ΔI(ΔI_opt)
        return ΔI_opt, value
    end

    # With adjustment costs, need to optimize
    if has_fixed_cost(ac)
        # Discrete choice: adjust or not
        # Option 1: No adjustment (ΔI = 0)
        value_no_adjust = π_half + obj_ΔI(0.0)

        # Option 2: Adjust optimally
        if ΔI_min < -1e-10 || ΔI_max > 1e-10  # Can actually adjust
            ΔI_opt_adjust, val_adjust = maximize_univariate(obj_ΔI, ΔI_min, ΔI_max; tol=1e-6)
            value_adjust = π_half + val_adjust

            if value_adjust > value_no_adjust
                return ΔI_opt_adjust, value_adjust
            else
                return 0.0, value_no_adjust
            end
        else
            return 0.0, value_no_adjust
        end
    else
        # Continuous optimization
        ΔI_opt, val = maximize_univariate(obj_ΔI, ΔI_min, ΔI_max; tol=1e-6)
        value = π_half + val
        return ΔI_opt, value
    end
end

"""
    compute_midyear_continuation(K_prime::Float64, i_D::Int, i_σ::Int,
                                  K_current::Float64, I_initial::Float64,
                                  V::Array{Float64,3}, grids::StateGrids,
                                  params::ModelParameters, ac::AbstractAdjustmentCost,
                                  derived::DerivedParameters) -> Float64

Compute W(K', D, σ): expected value of mid-year problem.

W(K', D, σ) = E_{D_half, σ_half | D, σ}[max_ΔI {...}]

# Returns
- Expected mid-year continuation value
"""
function compute_midyear_continuation(K_prime::Float64, i_D::Int, i_σ::Int,
                                      K_current::Float64, I_initial::Float64,
                                      V::Array{Float64,3}, grids::StateGrids,
                                      params::ModelParameters, ac::AbstractAdjustmentCost,
                                      derived::DerivedParameters)
    # Get transition probability from (D, σ) to (D_half, σ_half)
    i_state = get_joint_state_index(grids, i_D, i_σ)

    W_value = 0.0

    # Expectation over mid-year states
    for i_state_half in 1:grids.n_states
        i_D_half, i_σ_half = get_D_σ_indices(grids, i_state_half)

        # Solve mid-year problem for this realization
        ΔI_opt, value_half = solve_midyear_problem(
            K_prime, i_D_half, i_σ_half, K_current, I_initial,
            V, grids, params, ac, derived
        )

        # Weight by probability
        prob = grids.Π_semester[i_state, i_state_half]
        W_value += prob * value_half
    end

    return W_value
end

"""
    solve_beginning_year_problem(i_K::Int, i_D::Int, i_σ::Int,
                                  V::Array{Float64,3}, grids::StateGrids,
                                  params::ModelParameters, ac::AbstractAdjustmentCost,
                                  derived::DerivedParameters) -> (Float64, Float64)

Solve beginning-of-year problem: choose I to maximize value.

V(K, D, σ) = max_I { π(K, D) - C_1(I, K) + E[W(K', D, σ) | D, σ] }

where K' = (1-δ)K + I.

# Returns
- I_opt: Optimal initial investment
- V_value: Maximized value
"""
function solve_beginning_year_problem(i_K::Int, i_D::Int, i_σ::Int,
                                      V::Array{Float64,3}, grids::StateGrids,
                                      params::ModelParameters, ac::AbstractAdjustmentCost,
                                      derived::DerivedParameters)
    # Current state
    K = get_K(grids, i_K)
    D = get_D(grids, i_D)

    # First-semester profit
    π_first = profit(K, D, derived)

    # Objective function: maximize over I
    function obj_I(I)
        K_prime = (1 - derived.δ_semester) * K + I

        # Check feasibility
        if K_prime < grids.K_min
            return -Inf
        end
        if K_prime > grids.K_max
            return -1e10
        end

        # Initial adjustment cost
        # Note: For most cost types, this depends on I only (ΔI = 0 at this stage)
        # Exception: ConvexAdjustmentCost depends on total I + ΔI,
        # but here we're just making initial decision, so we use a placeholder
        # The full cost will be computed in mid-year problem

        # For separate costs, only charge C_1(I, K)
        if ac isa SeparateConvexCost
            cost = 0.5 * ac.ϕ₁ * (I / K)^2 * K
        elseif ac isa NoAdjustmentCost
            cost = 0.0
        elseif ac isa ConvexAdjustmentCost
            # For standard convex, we need to anticipate ΔI
            # Simplified: assume ΔI = 0 for now (will be optimized in mid-year)
            cost = 0.5 * ac.ϕ * (I / K)^2 * K
        else
            # For other types, compute cost assuming ΔI = 0
            cost = compute_cost(ac, I, 0.0, K)
        end

        # Continuation value
        W_value = compute_midyear_continuation(
            K_prime, i_D, i_σ, K, I, V, grids, params, ac, derived
        )

        return -cost + W_value
    end

    # Determine search bounds for I
    # Constraint: K_min <= (1-δ)K + I <= K_max
    I_min = grids.K_min - (1 - derived.δ_semester) * K
    I_max = grids.K_max - (1 - derived.δ_semester) * K

    # Ensure K_prime > 0
    I_min = max(I_min, -(1 - derived.δ_semester) * K + 1e-6)

    # Optimize
    if ac isa NoAdjustmentCost && !has_fixed_cost(ac)
        # Can use coarser search for faster convergence
        I_opt, val = maximize_univariate(obj_I, I_min, I_max; method=:brent, tol=1e-5)
    elseif has_fixed_cost(ac)
        # Discrete choice
        value_no_invest = π_first + obj_I(0.0)

        if I_min < -1e-10 || I_max > 1e-10
            I_opt_invest, val_invest = maximize_univariate(obj_I, I_min, I_max; tol=1e-6)
            value_invest = π_first + val_invest

            if value_invest > value_no_invest
                I_opt = I_opt_invest
                val = val_invest
            else
                I_opt = 0.0
                val = value_no_invest - π_first
            end
        else
            I_opt = 0.0
            val = value_no_invest - π_first
        end
    else
        I_opt, val = maximize_univariate(obj_I, I_min, I_max; tol=1e-6)
    end

    V_value = π_first + val

    return I_opt, V_value
end

"""
    bellman_operator!(V_new::Array{Float64,3}, V::Array{Float64,3},
                      I_policy::Array{Float64,3}, grids::StateGrids,
                      params::ModelParameters, ac::AbstractAdjustmentCost,
                      derived::DerivedParameters) -> Nothing

Apply Bellman operator: V_new = T(V).

Updates V_new and I_policy in-place.

For each state (K, D, σ):
1. Solve beginning-of-year problem to get I_opt and V_new
2. Store optimal policy I_policy[i_K, i_D, i_σ] = I_opt
3. Store value V_new[i_K, i_D, i_σ] = V_value
"""
function bellman_operator!(V_new::Array{Float64,3}, V::Array{Float64,3},
                          I_policy::Array{Float64,3}, grids::StateGrids,
                          params::ModelParameters, ac::AbstractAdjustmentCost,
                          derived::DerivedParameters)
    # Loop over all states
    for i_σ in 1:grids.n_σ
        for i_D in 1:grids.n_D
            for i_K in 1:grids.n_K
                I_opt, V_value = solve_beginning_year_problem(
                    i_K, i_D, i_σ, V, grids, params, ac, derived
                )

                V_new[i_K, i_D, i_σ] = V_value
                I_policy[i_K, i_D, i_σ] = I_opt
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
V^{k+1} = π + Γ(I_policy) V^k

where Γ is the transition operator under policy I_policy.

This accelerates VFI convergence.
"""
function howard_improvement_step!(V::Array{Float64,3}, I_policy::Array{Float64,3},
                                 grids::StateGrids, params::ModelParameters,
                                 ac::AbstractAdjustmentCost, derived::DerivedParameters,
                                 n_steps::Int)
    V_temp = copy(V)

    for step in 1:n_steps
        for i_σ in 1:grids.n_σ
            for i_D in 1:grids.n_D
                for i_K in 1:grids.n_K
                    K = get_K(grids, i_K)
                    D = get_D(grids, i_D)

                    # Fixed policy
                    I = I_policy[i_K, i_D, i_σ]

                    # First-semester profit
                    π_first = profit(K, D, derived)

                    # Capital after initial investment
                    K_prime = (1 - derived.δ_semester) * K + I

                    # Initial cost
                    if ac isa SeparateConvexCost
                        cost_I = 0.5 * ac.ϕ₁ * (I / K)^2 * K
                    else
                        cost_I = compute_cost(ac, I, 0.0, K)
                    end

                    # Mid-year continuation (using current V)
                    W_val = compute_midyear_continuation(
                        K_prime, i_D, i_σ, K, I, V_temp, grids, params, ac, derived
                    )

                    # Update value
                    V[i_K, i_D, i_σ] = π_first - cost_I + W_val
                end
            end
        end

        V_temp .= V
    end

    return nothing
end
