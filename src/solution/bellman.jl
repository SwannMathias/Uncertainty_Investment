"""
Bellman operators for the two-stage (semester-frequency) dynamic program.

State in both stages: (K, D, sigma)
- Stage 0 (beginning of year): choose I
- Stage 1 (mid-year after signal): choose Delta_I

Value functions:
- V0(K, D, sigma): beginning-of-year value
- V1(K, D, sigma): mid-year continuation value
"""

@inline function precompute_expectation_cache!(
    EV_cache::AbstractMatrix{Float64},
    V::Array{Float64,3},
    grids::StateGrids;
    horizon::Symbol=:semester
)
    @assert horizon in [:semester, :year] "horizon must be :semester or :year"
    Pi = horizon == :semester ? grids.Pi_semester : grids.Pi_year
    V_mat = reshape(V, grids.n_K, grids.n_states)
    mul!(EV_cache, V_mat, transpose(Pi))
    return nothing
end

@inline function _best_adjustment_choice(obj, x_min::Float64, x_max::Float64)
    if x_min >= x_max
        return x_min, obj(x_min)
    end
    x_opt, v_opt = maximize_univariate(obj, x_min, x_max; tol=1e-6)
    return x_opt, v_opt
end

function _maximize_with_inaction(obj, lower::Float64, upper::Float64)
    v0 = obj(0.0)
    x_best, v_best = 0.0, v0

    if lower < -1e-10
        x_neg, v_neg = _best_adjustment_choice(obj, lower, min(-1e-10, upper))
        if v_neg > v_best
            x_best, v_best = x_neg, v_neg
        end
    end

    if upper > 1e-10
        x_pos, v_pos = _best_adjustment_choice(obj, max(1e-10, lower), upper)
        if v_pos > v_best
            x_best, v_best = x_pos, v_pos
        end
    end

    return x_best, v_best
end

function solve_midyear_problem(K_stage1::Float64, i_D::Int, i_sigma::Int,
                               i_K::Int, K_current::Float64, I_initial::Float64,
                               V0::Array{Float64,3}, grids::StateGrids,
                               params::ModelParameters, ac_mid_year::AbstractAdjustmentCost,
                               derived::DerivedParameters,
                               EV1_to_0::AbstractMatrix{Float64})
    i_state = get_joint_state_index(grids, i_D, i_sigma)
    EV = @view EV1_to_0[:, i_state]

    # No mid-year depreciation: capital stays at K_stage1 until end of year
    function obj_Delta_I(Delta_I)
        K_next = K_stage1 + Delta_I
        if K_next < grids.K_min
            return -Inf
        end
        if K_next > grids.K_max
            return -1e10
        end
        cost = compute_cost(ac_mid_year, 0.0, Delta_I, K_current)
        return -cost + params.beta * linear_interp_1d(grids.K_grid, EV, K_next)
    end

    Delta_I_min = max(grids.K_min - K_stage1, -K_stage1 + 1e-6)
    Delta_I_max = grids.K_max - K_stage1

    if ac_mid_year isa NoAdjustmentCost
        i_opt = argmax(EV)
        Delta = clamp(grids.K_grid[i_opt] - K_stage1, Delta_I_min, Delta_I_max)
        return Delta, obj_Delta_I(Delta)
    elseif has_fixed_cost(ac_mid_year)
        return _maximize_with_inaction(obj_Delta_I, Delta_I_min, Delta_I_max)
    else
        return _best_adjustment_choice(obj_Delta_I, Delta_I_min, Delta_I_max)
    end
end

function compute_midyear_continuation(K_stage1::Float64, i_D::Int, i_sigma::Int,
                                      i_K::Int, K_current::Float64, I_initial::Float64,
                                      V0::Array{Float64,3}, grids::StateGrids,
                                      params::ModelParameters, ac_mid_year::AbstractAdjustmentCost,
                                      derived::DerivedParameters,
                                      EV1_to_0::AbstractMatrix{Float64})
    _, value = solve_midyear_problem(K_stage1, i_D, i_sigma, i_K, K_current, I_initial,
                                     V0, grids, params, ac_mid_year, derived, EV1_to_0)
    return value
end

function solve_beginning_year_problem(i_K::Int, i_D::Int, i_sigma::Int,
                                      V1::Array{Float64,3}, grids::StateGrids,
                                      params::ModelParameters,
                                      ac_begin::AbstractAdjustmentCost,
                                      ac_mid_year::AbstractAdjustmentCost,
                                      derived::DerivedParameters,
                                      EV0_to_1::AbstractMatrix{Float64})
    K = get_K(grids, i_K)
    pi_first = get_profit(grids, i_K, i_D)
    i_state = get_joint_state_index(grids, i_D, i_sigma)
    EV = @view EV0_to_1[:, i_state]

    # Expected mid-year profit E[π(K, D_half) | D, σ] — constant w.r.t. I
    probs_mid = @view grids.Pi_semester[i_state, :]
    expected_pi_mid = 0.0
    for i_state_half in 1:grids.n_states
        i_D_half, _ = get_D_sigma_indices(grids, i_state_half)
        expected_pi_mid += probs_mid[i_state_half] * get_profit(grids, i_K, i_D_half)
    end

    function obj_I(I)
        K_stage1 = (1 - derived.delta_annual) * K + I
        if K_stage1 < grids.K_min
            return -Inf
        end
        if K_stage1 > grids.K_max
            return -1e10
        end
        cost = compute_cost(ac_begin, I, 0.0, K)
        return pi_first - cost + expected_pi_mid + linear_interp_1d(grids.K_grid, EV, K_stage1)
    end

    I_min = max(grids.K_min - (1 - derived.delta_annual) * K,
                -(1 - derived.delta_annual) * K + 1e-6)
    I_max = grids.K_max - (1 - derived.delta_annual) * K

    if ac_begin isa NoAdjustmentCost
        i_opt = argmax(EV)
        I = clamp(grids.K_grid[i_opt] - (1 - derived.delta_annual) * K, I_min, I_max)
        return I, obj_I(I)
    elseif has_fixed_cost(ac_begin)
        return _maximize_with_inaction(obj_I, I_min, I_max)
    else
        return _best_adjustment_choice(obj_I, I_min, I_max)
    end
end

function update_stage1!(V1_new::Array{Float64,3}, Delta_I_policy::Array{Float64,3},
                        V0::Array{Float64,3}, grids::StateGrids, params::ModelParameters,
                        ac_mid_year::AbstractAdjustmentCost, derived::DerivedParameters,
                        EV1_to_0::AbstractMatrix{Float64})
    for i_sigma in 1:grids.n_sigma, i_D in 1:grids.n_D, i_K in 1:grids.n_K
        K = get_K(grids, i_K)
        ΔI_opt, V_val = solve_midyear_problem(K, i_D, i_sigma, i_K, K, 0.0,
                                              V0, grids, params, ac_mid_year, derived, EV1_to_0)
        Delta_I_policy[i_K, i_D, i_sigma] = ΔI_opt
        V1_new[i_K, i_D, i_sigma] = V_val
    end
    return nothing
end

function update_stage0!(V0_new::Array{Float64,3}, I_policy::Array{Float64,3},
                        V1::Array{Float64,3}, grids::StateGrids, params::ModelParameters,
                        ac_begin::AbstractAdjustmentCost, ac_mid_year::AbstractAdjustmentCost,
                        derived::DerivedParameters, EV0_to_1::AbstractMatrix{Float64})
    for i_sigma in 1:grids.n_sigma, i_D in 1:grids.n_D, i_K in 1:grids.n_K
        I_opt, V_val = solve_beginning_year_problem(i_K, i_D, i_sigma, V1, grids,
                                                    params, ac_begin, ac_mid_year,
                                                    derived, EV0_to_1)
        I_policy[i_K, i_D, i_sigma] = I_opt
        V0_new[i_K, i_D, i_sigma] = V_val
    end
    return nothing
end

function bellman_operator!(V0_new::Array{Float64,3}, V0::Array{Float64,3},
                           I_policy::Array{Float64,3}, grids::StateGrids,
                           params::ModelParameters,
                           ac_begin::AbstractAdjustmentCost,
                           ac_mid_year::AbstractAdjustmentCost,
                           derived::DerivedParameters,
                           EV1_to_0::AbstractMatrix{Float64},
                           EV0_to_1::AbstractMatrix{Float64},
                           V1::Array{Float64,3}, V1_new::Array{Float64,3},
                           Delta_I_policy::Array{Float64,3})
    precompute_expectation_cache!(EV1_to_0, V0, grids; horizon=:semester)
    update_stage1!(V1_new, Delta_I_policy, V0, grids, params, ac_mid_year, derived, EV1_to_0)
    precompute_expectation_cache!(EV0_to_1, V1_new, grids; horizon=:semester)
    update_stage0!(V0_new, I_policy, V1_new, grids, params, ac_begin, ac_mid_year, derived, EV0_to_1)
    return nothing
end

function bellman_operator!(V_new::Array{Float64,3}, V::Array{Float64,3},
                           I_policy::Array{Float64,3}, grids::StateGrids,
                           params::ModelParameters,
                           ac_begin::AbstractAdjustmentCost,
                           ac_mid_year::AbstractAdjustmentCost,
                           derived::DerivedParameters)
    EV1_to_0 = zeros(grids.n_K, grids.n_states)
    EV0_to_1 = zeros(grids.n_K, grids.n_states)
    V1 = copy(V)
    V1_new = similar(V)
    Delta_I_policy = zeros(size(V))
    bellman_operator!(V_new, V, I_policy, grids, params, ac_begin, ac_mid_year, derived,
                      EV1_to_0, EV0_to_1, V1, V1_new, Delta_I_policy)
    return nothing
end

function howard_improvement_step!(V0::Array{Float64,3}, V1::Array{Float64,3},
                                  I_policy::Array{Float64,3}, Delta_I_policy::Array{Float64,3},
                                  grids::StateGrids, params::ModelParameters,
                                  ac_begin::AbstractAdjustmentCost,
                                  ac_mid_year::AbstractAdjustmentCost,
                                  derived::DerivedParameters,
                                  n_steps::Int)
    EV1_to_0 = zeros(grids.n_K, grids.n_states)
    EV0_to_1 = zeros(grids.n_K, grids.n_states)

    for _ in 1:n_steps
        precompute_expectation_cache!(EV1_to_0, V0, grids; horizon=:semester)
        for i_sigma in 1:grids.n_sigma, i_D in 1:grids.n_D, i_K in 1:grids.n_K
            K = get_K(grids, i_K)
            ΔI = Delta_I_policy[i_K, i_D, i_sigma]
            i_state = get_joint_state_index(grids, i_D, i_sigma)
            EV = @view EV1_to_0[:, i_state]
            K_next = K + ΔI  # No mid-year depreciation
            V1[i_K, i_D, i_sigma] = -compute_cost(ac_mid_year, 0.0, ΔI, K) +
                                    params.beta * linear_interp_1d(grids.K_grid, EV, K_next)
        end

        precompute_expectation_cache!(EV0_to_1, V1, grids; horizon=:semester)
        for i_sigma in 1:grids.n_sigma, i_D in 1:grids.n_D, i_K in 1:grids.n_K
            K = get_K(grids, i_K)
            I = I_policy[i_K, i_D, i_sigma]
            i_state = get_joint_state_index(grids, i_D, i_sigma)
            EV = @view EV0_to_1[:, i_state]
            K_stage1 = (1 - derived.delta_annual) * K + I
            # Expected mid-year profit E[π(K, D_half) | D, σ]
            probs_mid = @view grids.Pi_semester[i_state, :]
            expected_pi_mid = 0.0
            for i_state_half in 1:grids.n_states
                i_D_half, _ = get_D_sigma_indices(grids, i_state_half)
                expected_pi_mid += probs_mid[i_state_half] * get_profit(grids, i_K, i_D_half)
            end
            V0[i_K, i_D, i_sigma] = get_profit(grids, i_K, i_D) - compute_cost(ac_begin, I, 0.0, K) +
                                    expected_pi_mid + linear_interp_1d(grids.K_grid, EV, K_stage1)
        end
    end
    return nothing
end

function bellman_operator_parallel!(V0_new::Array{Float64,3}, V0::Array{Float64,3},
                                    I_policy::Array{Float64,3}, grids::StateGrids,
                                    params::ModelParameters,
                                    ac_begin::AbstractAdjustmentCost,
                                    ac_mid_year::AbstractAdjustmentCost,
                                    derived::DerivedParameters,
                                    EV1_to_0::AbstractMatrix{Float64},
                                    EV0_to_1::AbstractMatrix{Float64},
                                    V1::Array{Float64,3}, V1_new::Array{Float64,3},
                                    Delta_I_policy::Array{Float64,3})
    precompute_expectation_cache!(EV1_to_0, V0, grids; horizon=:semester)
    n_total = grids.n_K * grids.n_D * grids.n_sigma
    @threads for idx in 1:n_total
        i_K = ((idx - 1) % grids.n_K) + 1
        temp = (idx - 1) ÷ grids.n_K
        i_D = (temp % grids.n_D) + 1
        i_sigma = (temp ÷ grids.n_D) + 1
        K = get_K(grids, i_K)
        ΔI_opt, V_val = solve_midyear_problem(K, i_D, i_sigma, i_K, K, 0.0,
                                              V0, grids, params, ac_mid_year, derived, EV1_to_0)
        Delta_I_policy[i_K, i_D, i_sigma] = ΔI_opt
        V1_new[i_K, i_D, i_sigma] = V_val
    end

    precompute_expectation_cache!(EV0_to_1, V1_new, grids; horizon=:semester)
    @threads for idx in 1:n_total
        i_K = ((idx - 1) % grids.n_K) + 1
        temp = (idx - 1) ÷ grids.n_K
        i_D = (temp % grids.n_D) + 1
        i_sigma = (temp ÷ grids.n_D) + 1
        I_opt, V_val = solve_beginning_year_problem(i_K, i_D, i_sigma, V1_new, grids,
                                                    params, ac_begin, ac_mid_year,
                                                    derived, EV0_to_1)
        I_policy[i_K, i_D, i_sigma] = I_opt
        V0_new[i_K, i_D, i_sigma] = V_val
    end
    return nothing
end

function bellman_operator_parallel!(V_new::Array{Float64,3}, V::Array{Float64,3},
                                    I_policy::Array{Float64,3}, grids::StateGrids,
                                    params::ModelParameters,
                                    ac_begin::AbstractAdjustmentCost,
                                    ac_mid_year::AbstractAdjustmentCost,
                                    derived::DerivedParameters)
    EV1_to_0 = zeros(grids.n_K, grids.n_states)
    EV0_to_1 = zeros(grids.n_K, grids.n_states)
    V1 = copy(V)
    V1_new = similar(V)
    Delta_I_policy = zeros(size(V))
    bellman_operator_parallel!(V_new, V, I_policy, grids, params, ac_begin, ac_mid_year, derived,
                               EV1_to_0, EV0_to_1, V1, V1_new, Delta_I_policy)
    return nothing
end

function howard_improvement_step_parallel!(V0::Array{Float64,3}, V1::Array{Float64,3},
                                           I_policy::Array{Float64,3}, Delta_I_policy::Array{Float64,3},
                                           grids::StateGrids, params::ModelParameters,
                                           ac_begin::AbstractAdjustmentCost,
                                           ac_mid_year::AbstractAdjustmentCost,
                                           derived::DerivedParameters,
                                           n_steps::Int)
    howard_improvement_step!(V0, V1, I_policy, Delta_I_policy,
                             grids, params, ac_begin, ac_mid_year, derived, n_steps)
end

# ============================================================================
# Full Howard Acceleration (fix both I and ΔI policies)
# ============================================================================

"""
    record_midyear_policy!(Delta_I_mid, I_policy, V, grids, params, ac_mid_year, derived, EV_cache)

Record the optimal mid-year investment revision ΔI* for every state and every
mid-year realization, given fixed beginning-of-year policy I_policy and current
value function V.

After each Bellman step, call this function once (cost ≈ 1 old Howard step).
The stored Delta_I_mid array is then used by howard_full_step! for cheap
policy-evaluation steps with no optimizers.

# Arguments
- `Delta_I_mid`: (n_K, n_D, n_sigma, n_states) output array — modified in place
- `I_policy`: (n_K, n_D, n_sigma) fixed beginning-of-year policy
- `V`: (n_K, n_D, n_sigma) current value function
- `EV_cache`: (n_K, n_states) precomputed expectations E[V(K,·)|state]
"""
function record_midyear_policy!(
    Delta_I_mid::Array{Float64,4},
    I_policy::Array{Float64,3},
    V::Array{Float64,3},
    grids::StateGrids,
    params::ModelParameters,
    ac_mid_year::AbstractAdjustmentCost,
    derived::DerivedParameters,
    EV_cache::AbstractMatrix{Float64}
)
    n_K = grids.n_K
    n_D = grids.n_D
    n_sigma = grids.n_sigma
    n_states = grids.n_states

    for i_sigma in 1:n_sigma
        for i_D in 1:n_D
            for i_K in 1:n_K
                K = get_K(grids, i_K)
                I_opt = I_policy[i_K, i_D, i_sigma]
                K_prime = (1.0 - derived.delta_annual) * K + I_opt

                for i_state_half in 1:n_states
                    i_D_half, i_sigma_half = get_D_sigma_indices(grids, i_state_half)

                    Delta_I_opt, _ = solve_midyear_problem(
                        K_prime, i_D_half, i_sigma_half,
                        i_K, K, I_opt,
                        V, grids, params,
                        ac_mid_year, derived, EV_cache
                    )

                    Delta_I_mid[i_K, i_D, i_sigma, i_state_half] = Delta_I_opt
                end
            end
        end
    end
    return nothing
end

"""
    record_midyear_policy_parallel!(Delta_I_mid, I_policy, V, grids, params, ac_mid_year, derived, EV_cache)

Parallel version of record_midyear_policy! using multi-threading.
"""
function record_midyear_policy_parallel!(
    Delta_I_mid::Array{Float64,4},
    I_policy::Array{Float64,3},
    V::Array{Float64,3},
    grids::StateGrids,
    params::ModelParameters,
    ac_mid_year::AbstractAdjustmentCost,
    derived::DerivedParameters,
    EV_cache::AbstractMatrix{Float64}
)
    n_K = grids.n_K
    n_D = grids.n_D
    n_sigma = grids.n_sigma
    n_states = grids.n_states
    total_states = n_K * n_D * n_sigma

    @threads for idx in 1:total_states
        # Convert linear index to 3D indices (column-major, matches bellman_operator_parallel!)
        i_K = ((idx - 1) % n_K) + 1
        temp = (idx - 1) ÷ n_K
        i_D = (temp % n_D) + 1
        i_sigma = (temp ÷ n_D) + 1

        K = get_K(grids, i_K)
        I_opt = I_policy[i_K, i_D, i_sigma]
        K_prime = (1.0 - derived.delta_annual) * K + I_opt

        for i_state_half in 1:n_states
            i_D_half, i_sigma_half = get_D_sigma_indices(grids, i_state_half)

            Delta_I_opt, _ = solve_midyear_problem(
                K_prime, i_D_half, i_sigma_half,
                i_K, K, I_opt,
                V, grids, params,
                ac_mid_year, derived, EV_cache
            )

            Delta_I_mid[i_K, i_D, i_sigma, i_state_half] = Delta_I_opt
        end
    end
    return nothing
end

"""
    howard_full_step!(V, I_policy, Delta_I_mid, grids, params, ac_begin, ac_mid_year, derived, EV_cache)

Cheap Howard policy-evaluation step using both fixed policies (I and ΔI).

Unlike howard_improvement_step! which re-runs the ΔI optimizer for every
mid-year realization, this function uses prerecorded Delta_I_mid values so
each step is purely a lookup + interpolation with no optimizers.

Cost per step: ~105 interpolations per state (vs ~105 optimizations before).

# Arguments
- `V`: (n_K, n_D, n_sigma) value function — modified in place
- `I_policy`: (n_K, n_D, n_sigma) fixed beginning-of-year policy
- `Delta_I_mid`: (n_K, n_D, n_sigma, n_states) fixed mid-year policy
- `EV_cache`: (n_K, n_states) precomputed expectations (must reflect current V)
"""
function howard_full_step!(
    V::Array{Float64,3},
    I_policy::Array{Float64,3},
    Delta_I_mid::Array{Float64,4},
    grids::StateGrids,
    params::ModelParameters,
    ac_begin::AbstractAdjustmentCost,
    ac_mid_year::AbstractAdjustmentCost,
    derived::DerivedParameters,
    EV_cache::AbstractMatrix{Float64}
)
    n_K = grids.n_K
    n_D = grids.n_D
    n_sigma = grids.n_sigma
    n_states = grids.n_states
    beta = params.beta

    for i_sigma in 1:n_sigma
        for i_D in 1:n_D
            i_state = get_joint_state_index(grids, i_D, i_sigma)

            for i_K in 1:n_K
                K = get_K(grids, i_K)
                I_opt = I_policy[i_K, i_D, i_sigma]
                K_prime = (1.0 - derived.delta_annual) * K + I_opt

                pi_first = get_profit(grids, i_K, i_D)
                cost_I = compute_cost(ac_begin, I_opt, 0.0, K)

                W_value = 0.0
                for i_state_half in 1:n_states
                    prob = grids.Pi_semester[i_state, i_state_half]
                    if prob < 1e-15
                        continue
                    end

                    i_D_half, i_sigma_half = get_D_sigma_indices(grids, i_state_half)
                    Delta_I = Delta_I_mid[i_K, i_D, i_sigma, i_state_half]
                    K_double_prime = K_prime + Delta_I

                    # Mid-year profit uses beginning-of-year capital index (matches solve_midyear_problem)
                    pi_half = get_profit(grids, i_K, i_D_half)
                    # Mid-year cost: (0.0, Delta_I, K_current) matches solve_midyear_problem
                    cost_delta_I = compute_cost(ac_mid_year, 0.0, Delta_I, K)

                    EV_vec = @view EV_cache[:, i_state_half]
                    K_dp_clamped = clamp(K_double_prime, grids.K_min, grids.K_max)
                    EV_interp = linear_interp_1d(grids.K_grid, EV_vec, K_dp_clamped)

                    value_half = pi_half - cost_delta_I + beta * EV_interp
                    W_value += prob * value_half
                end

                V[i_K, i_D, i_sigma] = pi_first - cost_I + W_value
            end
        end
    end
    return nothing
end

"""
    howard_full_step_parallel!(V, I_policy, Delta_I_mid, grids, params, ac_begin, ac_mid_year, derived, EV_cache)

Parallel version of howard_full_step! using multi-threading.

Thread safety: each thread writes to a distinct V[i_K, i_D, i_sigma] element.
All reads (I_policy, Delta_I_mid, EV_cache, grids) are shared and read-only.
"""
function howard_full_step_parallel!(
    V::Array{Float64,3},
    I_policy::Array{Float64,3},
    Delta_I_mid::Array{Float64,4},
    grids::StateGrids,
    params::ModelParameters,
    ac_begin::AbstractAdjustmentCost,
    ac_mid_year::AbstractAdjustmentCost,
    derived::DerivedParameters,
    EV_cache::AbstractMatrix{Float64}
)
    n_K = grids.n_K
    n_D = grids.n_D
    n_sigma = grids.n_sigma
    n_states = grids.n_states
    beta = params.beta
    total_states = n_K * n_D * n_sigma

    @threads for idx in 1:total_states
        # Convert linear index to 3D indices (column-major, matches bellman_operator_parallel!)
        i_K = ((idx - 1) % n_K) + 1
        temp = (idx - 1) ÷ n_K
        i_D = (temp % n_D) + 1
        i_sigma = (temp ÷ n_D) + 1

        K = get_K(grids, i_K)
        I_opt = I_policy[i_K, i_D, i_sigma]
        K_prime = (1.0 - derived.delta_annual) * K + I_opt

        pi_first = get_profit(grids, i_K, i_D)
        cost_I = compute_cost(ac_begin, I_opt, 0.0, K)

        i_state = get_joint_state_index(grids, i_D, i_sigma)

        W_value = 0.0
        for i_state_half in 1:n_states
            prob = grids.Pi_semester[i_state, i_state_half]
            if prob < 1e-15
                continue
            end

            i_D_half, i_sigma_half = get_D_sigma_indices(grids, i_state_half)
            Delta_I = Delta_I_mid[i_K, i_D, i_sigma, i_state_half]
            K_double_prime = K_prime + Delta_I

            pi_half = get_profit(grids, i_K, i_D_half)
            cost_delta_I = compute_cost(ac_mid_year, 0.0, Delta_I, K)

            EV_vec = @view EV_cache[:, i_state_half]
            K_dp_clamped = clamp(K_double_prime, grids.K_min, grids.K_max)
            EV_interp = linear_interp_1d(grids.K_grid, EV_vec, K_dp_clamped)

            value_half = pi_half - cost_delta_I + beta * EV_interp
            W_value += prob * value_half
        end

        V[i_K, i_D, i_sigma] = pi_first - cost_I + W_value
    end
    return nothing
end
