using Test
using UncertaintyInvestment

function nested_stage1_value(K_stage1, i_D, i_sigma, i_K, K_current, V0, grids, params, ac_mid, derived, EV1_to_0)
    i_state = get_joint_state_index(grids, i_D, i_sigma)
    EV = @view EV1_to_0[:, i_state]
    # No mid-year depreciation: capital stays at K_stage1 until end of year
    obj(ΔI) = begin
        K_next = K_stage1 + ΔI
        if K_next < grids.K_min || K_next > grids.K_max
            return -1e12
        end
        -compute_cost(ac_mid, 0.0, ΔI, K_current) + params.beta * linear_interp_1d(grids.K_grid, EV, K_next)
    end
    Δmin = max(grids.K_min - K_stage1, -K_stage1 + 1e-6)
    Δmax = grids.K_max - K_stage1
    Δ, v = maximize_univariate(obj, Δmin, Δmax; tol=1e-6)
    return Δ, v
end

function nested_stage0_update!(V0_new, I_policy, V0, grids, params, ac_begin, ac_mid, derived)
    EV1_to_0 = zeros(grids.n_K, grids.n_states)
    precompute_expectation_cache!(EV1_to_0, V0, grids; horizon=:semester)

    for i_sigma in 1:grids.n_sigma, i_D in 1:grids.n_D, i_K in 1:grids.n_K
        K = get_K(grids, i_K)
        pi_first = get_profit(grids, i_K, i_D)
        i_state = get_joint_state_index(grids, i_D, i_sigma)
        probs = @view grids.Pi_semester[i_state, :]

        function obj_I(I)
            K_stage1 = (1 - derived.delta_annual) * K + I
            if K_stage1 < grids.K_min || K_stage1 > grids.K_max
                return -1e12
            end
            cont = 0.0
            for i_state_half in 1:grids.n_states
                i_D_half, i_sigma_half = get_D_sigma_indices(grids, i_state_half)
                _, v_half = nested_stage1_value(K_stage1, i_D_half, i_sigma_half, i_K, K, V0,
                                                grids, params, ac_mid, derived, EV1_to_0)
                pi_mid = get_profit(grids, i_K, i_D_half)
                cont += probs[i_state_half] * (pi_mid + v_half)
            end
            return pi_first - compute_cost(ac_begin, I, 0.0, K) + cont
        end

        Imin = max(grids.K_min - (1 - derived.delta_annual) * K,
                   -(1 - derived.delta_annual) * K + 1e-6)
        Imax = grids.K_max - (1 - derived.delta_annual) * K
        I_opt, V_opt = maximize_univariate(obj_I, Imin, Imax; tol=1e-6)
        I_policy[i_K, i_D, i_sigma] = I_opt
        V0_new[i_K, i_D, i_sigma] = V_opt
    end
end

@testset "Two-stage refactor vs nested baseline (small grid)" begin
    params = ModelParameters(numerical=NumericalSettings(n_K=10, n_D=4, n_sigma=3, max_iter=8, tol_vfi=1e-5))
    grids = construct_grids(params)
    derived = get_derived_parameters(params)
    ac_begin = ConvexAdjustmentCost(phi=0.8)
    ac_mid = ConvexAdjustmentCost(phi=1.2)

    V_nested = zeros(grids.n_K, grids.n_D, grids.n_sigma)
    V_twostage = copy(V_nested)
    V1_twostage = copy(V_nested)
    Vn_new = similar(V_nested)
    V0_new = similar(V_nested)
    V1_new = similar(V_nested)
    I_nested = zeros(size(V_nested))
    I_two = zeros(size(V_nested))
    Δ_two = zeros(size(V_nested))
    EV1_to_0 = zeros(grids.n_K, grids.n_states)
    EV0_to_1 = zeros(grids.n_K, grids.n_states)

    for _ in 1:4
        nested_stage0_update!(Vn_new, I_nested, V_nested, grids, params, ac_begin, ac_mid, derived)
        V_nested .= Vn_new

        bellman_operator!(V0_new, V_twostage, I_two, grids, params, ac_begin, ac_mid, derived,
                          EV1_to_0, EV0_to_1, V1_twostage, V1_new, Δ_two)
        V_twostage .= V0_new
        V1_twostage .= V1_new
    end

    nested_stage0_update!(Vn_new, I_nested, V_nested, grids, params, ac_begin, ac_mid, derived)
    bellman_operator!(V0_new, V_twostage, I_two, grids, params, ac_begin, ac_mid, derived,
                      EV1_to_0, EV0_to_1, V1_twostage, V1_new, Δ_two)

    residual_nested = maximum(abs.(Vn_new .- V_nested))
    residual_twostage = maximum(abs.(V0_new .- V_twostage))
    @test abs(residual_nested - residual_twostage) < 5e-2

    points = [(2,1,1), (5,2,2), (8,4,3)]
    for (iK,iD,iS) in points
        @test abs(I_nested[iK,iD,iS] - I_two[iK,iD,iS]) < 5e-2
    end

    for iD in 1:grids.n_D, iS in 1:grids.n_sigma
        dI = diff(I_two[:, iD, iS])
        @test sum(dI .< -1e-3) <= 2
    end
end
