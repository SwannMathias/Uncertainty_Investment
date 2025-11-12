
# ============================================================================
# SOLVER
# ============================================================================

"""
    solve_model(model; max_iterations, tolerance, fixed_cost, verbose)

Solve the dynamic investment model via value function iteration.

# Returns
Named tuple with value and policy functions for each quarter.
"""
function solve_model(
    model::InvestmentModel;
    max_iterations=1000,
    tolerance=1e-6,
    fixed_cost=0.0,
    verbose=true
)
    # Dimensions
    num_K = length(model.capital_grid)
    num_I_cum = length(model.cumulative_investment_grid)
    num_D = length(model.demand_shock_grid)
    num_vol = length(model.volatility_grid)
    
    # Precompute transition matrices (MAJOR SPEEDUP)
    verbose && println("  Precomputing transition matrices...")
    vol_within_transition = precompute_volatility_transitions(model, 0.25)
    vol_between_transition = precompute_volatility_transitions(model, 0.25)
    demand_transition = precompute_demand_transitions(model, 1.0)
    
    # Initialize arrays
    V_january  = zeros(num_K, num_I_cum, num_D, num_vol)
    V_april    = similar(V_january)
    V_august   = similar(V_january)
    V_october  = similar(V_january)
    
    policy_january = zeros(num_K, num_D, num_vol)
    policy_april   = zeros(num_K, num_I_cum, num_D, num_vol)
    policy_august  = similar(policy_april)
    policy_october = similar(policy_april)
    
    V_january_new = similar(V_january)
    
    # Indices for parallel computation
    idx_full = collect(CartesianIndices((num_K, num_I_cum, num_D, num_vol)))
    idx_january = collect(CartesianIndices((num_K, num_D, num_vol)))
    
    verbose && println("  State space: K=$num_K × I_cum=$num_I_cum × D=$num_D × σ=$num_vol")
    verbose && println("  Starting iteration...")
    
    iteration, difference = 0, Inf
    
    while iteration < max_iterations && difference > tolerance
        iteration += 1
        
        # ===== OCTOBER =====
        results_october = pmap(idx_full) do idx
            k, i, d, v = Tuple(idx)
            value_october(k, i, d, v, model, V_january, vol_between_transition, 
                         demand_transition, fixed_cost)
        end
        V_october .= reshape(first.(results_october), size(V_october))
        policy_october .= reshape(last.(results_october), size(policy_october))
        
        # ===== AUGUST =====
        results_august = pmap(idx_full) do idx
            k, i, d, v = Tuple(idx)
            value_august(k, i, d, v, model, V_october, vol_within_transition, fixed_cost)
        end
        V_august .= reshape(first.(results_august), size(V_august))
        policy_august .= reshape(last.(results_august), size(policy_august))
        
        # ===== APRIL =====
        results_april = pmap(idx_full) do idx
            k, i, d, v = Tuple(idx)
            value_april(k, i, d, v, model, V_august, vol_within_transition, fixed_cost)
        end
        V_april .= reshape(first.(results_april), size(V_april))
        policy_april .= reshape(last.(results_april), size(policy_april))
        
        # ===== JANUARY =====
        results_january = pmap(idx_january) do idx
            k, d, v = Tuple(idx)
            value_january(k, 1, d, v, model, V_april, vol_within_transition, fixed_cost)
        end
        V_january_new[:, 1, :, :] .= reshape(first.(results_january), size(V_january[:, 1, :, :]))
        policy_january .= reshape(last.(results_january), size(policy_january))
        
        # Check convergence
        difference = maximum(abs.(V_january_new .- V_january))
        V_january .= V_january_new
        
        verbose && (iteration % 10 == 0) && println("  Iter $iteration: diff = $(round(difference, sigdigits=4))")
    end
    
    verbose && println("  ✓ Converged in $iteration iterations")
    
    return (
        V_january=V_january, V_april=V_april, V_august=V_august, V_october=V_october,
        policy_january=policy_january, policy_april=policy_april,
        policy_august=policy_august, policy_october=policy_october
    )
end
