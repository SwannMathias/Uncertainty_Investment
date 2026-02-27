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

    function obj_Delta_I(Delta_I)
        K_next = K_stage1 + Delta_I
        if K_next < grids.K_min
            return -Inf
        end
        if K_next > grids.K_max
            return -1e10
        end
        cost = compute_cost(ac_mid_year, 0.0, Delta_I, K_current)
        return -cost + linear_interp_1d(grids.K_grid, EV, K_next)
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

    function obj_I(I)
        K_stage1 = (1 - derived.delta_semester) * K + I
        if K_stage1 < grids.K_min
            return -Inf
        end
        if K_stage1 > grids.K_max
            return -1e10
        end
        cost = compute_cost(ac_begin, I, 0.0, K)
        return pi_first - cost + linear_interp_1d(grids.K_grid, EV, K_stage1)
    end

    I_min = max(grids.K_min - (1 - derived.delta_semester) * K,
                -(1 - derived.delta_semester) * K + 1e-6)
    I_max = grids.K_max - (1 - derived.delta_semester) * K

    if ac_begin isa NoAdjustmentCost
        i_opt = argmax(EV)
        I = clamp(grids.K_grid[i_opt] - (1 - derived.delta_semester) * K, I_min, I_max)
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
            V1[i_K, i_D, i_sigma] = -compute_cost(ac_mid_year, 0.0, ΔI, K) +
                                    linear_interp_1d(grids.K_grid, EV, K + ΔI)
        end

        precompute_expectation_cache!(EV0_to_1, V1, grids; horizon=:semester)
        for i_sigma in 1:grids.n_sigma, i_D in 1:grids.n_D, i_K in 1:grids.n_K
            K = get_K(grids, i_K)
            I = I_policy[i_K, i_D, i_sigma]
            i_state = get_joint_state_index(grids, i_D, i_sigma)
            EV = @view EV0_to_1[:, i_state]
            K_stage1 = (1 - derived.delta_semester) * K + I
            V0[i_K, i_D, i_sigma] = get_profit(grids, i_K, i_D) - compute_cost(ac_begin, I, 0.0, K) +
                                    linear_interp_1d(grids.K_grid, EV, K_stage1)
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
