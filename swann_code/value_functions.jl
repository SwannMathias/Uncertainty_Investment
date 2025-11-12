
# ============================================================================
# VALUE FUNCTIONS (OPTIMIZED)
# ============================================================================

"""
    value_october(...)

Q4 value function: Last investment decision of the year.
State: (K_t, cumulative_I, X_t, σ_{t,Q4})
"""
function value_october(
    capital_index::Int,
    cumulative_investment_index::Int,
    demand_shock_index::Int,
    volatility_index::Int,
    model::InvestmentModel,
    value_january_next::Array{Float64, 4},
    vol_transition_matrix::Matrix{Float64},
    demand_transition_array::Array{Float64, 3},
    fixed_cost::Float64
)
    capital = model.capital_grid[capital_index]
    cumulative_investment = model.cumulative_investment_grid[cumulative_investment_index]
    demand_shock = model.demand_shock_grid[demand_shock_index]
    
    max_value = -Inf
    best_investment = 0.0
    
    # Precompute volatility and demand transitions for this state
    vol_probs = vol_transition_matrix[volatility_index, :]
    demand_probs_matrix = demand_transition_array[demand_shock_index, :, :]
    
    for investment_october in model.investment_grid
        total_investment = cumulative_investment + investment_october
        next_capital = capital_transition(capital, total_investment, model)
        
        # Bounds checking
        next_capital < model.capital_grid[1] && continue
        next_capital > model.capital_grid[end] && continue
        
        # Current period payoff
        current_payoff = -investment_cost(investment_october, model, fixed_cost=fixed_cost)
        !isfinite(current_payoff) && continue
        
        # Expected continuation value (vectorized)
        # E[V_{t+1}] = Σ_σ P(σ_{t+1}|σ_t) Σ_X P(X_{t+1}|X_t,σ_{t+1}) V(K_{t+1}, X_{t+1}, σ_{t+1})
        expected_continuation = 0.0
        
        for (next_vol_idx, vol_prob) in enumerate(vol_probs)
            vol_prob < 1e-10 && continue
            
            demand_probs = demand_probs_matrix[next_vol_idx, :]
            
            # Vectorized interpolation over demand shocks
            value_slice = value_january_next[:, 1, :, next_vol_idx]
            value_interp = LinearInterpolation(model.capital_grid, value_slice[:, 1])
            interpolated_value = value_interp(next_capital)
            
            # Weighted sum over demand shocks
            for (next_shock_idx, shock_prob) in enumerate(demand_probs)
                shock_prob < 1e-10 && continue
                
                value_interp = LinearInterpolation(
                    model.capital_grid,
                    value_january_next[:, 1, next_shock_idx, next_vol_idx]
                )
                expected_continuation += vol_prob * shock_prob * value_interp(next_capital)
            end
        end
        
        total_value = current_payoff + model.discount_factor * expected_continuation
        
        if total_value > max_value
            max_value = total_value
            best_investment = investment_october
        end
    end
    
    return max_value, best_investment
end

"""
    value_august(...)

Q3 value function: State (K_t, cumulative_I, X_t, σ_{t,Q3})
"""
function value_august(
    capital_index::Int,
    cumulative_investment_index::Int,
    demand_shock_index::Int,
    volatility_index::Int,
    model::InvestmentModel,
    value_october_array::Array{Float64, 4},
    vol_transition_matrix::Matrix{Float64},
    fixed_cost::Float64
)
    capital = model.capital_grid[capital_index]
    cumulative_investment = model.cumulative_investment_grid[cumulative_investment_index]
    
    max_value = -Inf
    best_investment = 0.0
    
    vol_probs = vol_transition_matrix[volatility_index, :]
    
    for investment_august in model.investment_grid
        new_cumulative_investment = cumulative_investment + investment_august
        
        # Bounds checking
        new_cumulative_investment < model.cumulative_investment_grid[1] && continue
        new_cumulative_investment > model.cumulative_investment_grid[end] && continue
        
        current_payoff = -investment_cost(investment_august, model, fixed_cost=fixed_cost)
        !isfinite(current_payoff) && continue
        
        # Expected October value
        expected_october = 0.0
        for (next_vol_idx, vol_prob) in enumerate(vol_probs)
            vol_prob < 1e-10 && continue
            
            value_interp = LinearInterpolation(
                model.cumulative_investment_grid,
                value_october_array[capital_index, :, demand_shock_index, next_vol_idx]
            )
            expected_october += vol_prob * value_interp(new_cumulative_investment)
        end
        
        total_value = current_payoff + model.discount_factor^(1/4) * expected_october
        
        if total_value > max_value
            max_value = total_value
            best_investment = investment_august
        end
    end
    
    return max_value, best_investment
end

"""
    value_april(...)

Q2 value function: State (K_t, cumulative_I, X_t, σ_{t,Q2})
"""
function value_april(
    capital_index::Int,
    cumulative_investment_index::Int,
    demand_shock_index::Int,
    volatility_index::Int,
    model::InvestmentModel,
    value_august_array::Array{Float64, 4},
    vol_transition_matrix::Matrix{Float64},
    fixed_cost::Float64
)
    capital = model.capital_grid[capital_index]
    cumulative_investment = model.cumulative_investment_grid[cumulative_investment_index]
    
    max_value = -Inf
    best_investment = 0.0
    
    vol_probs = vol_transition_matrix[volatility_index, :]
    
    for investment_april in model.investment_grid
        new_cumulative_investment = cumulative_investment + investment_april
        
        new_cumulative_investment < model.cumulative_investment_grid[1] && continue
        new_cumulative_investment > model.cumulative_investment_grid[end] && continue
        
        current_payoff = -investment_cost(investment_april, model, fixed_cost=fixed_cost)
        !isfinite(current_payoff) && continue
        
        expected_august = 0.0
        for (next_vol_idx, vol_prob) in enumerate(vol_probs)
            vol_prob < 1e-10 && continue
            
            value_interp = LinearInterpolation(
                model.cumulative_investment_grid,
                value_august_array[capital_index, :, demand_shock_index, next_vol_idx]
            )
            expected_august += vol_prob * value_interp(new_cumulative_investment)
        end
        
        total_value = current_payoff + model.discount_factor^(1/4) * expected_august
        
        if total_value > max_value
            max_value = total_value
            best_investment = investment_april
        end
    end
    
    return max_value, best_investment
end

"""
    value_january(...)

Q1 value function: State (K_t, X_t, σ_{t,Q1})
Firm observes new demand shock and makes first investment decision.
"""
function value_january(
    capital_index::Int,
    cumulative_investment_index::Int,
    demand_shock_index::Int,
    volatility_index::Int,
    model::InvestmentModel,
    value_april_array::Array{Float64, 4},
    vol_transition_matrix::Matrix{Float64},
    fixed_cost::Float64
)
    capital = model.capital_grid[capital_index]
    cumulative_investment = model.cumulative_investment_grid[cumulative_investment_index]
    demand_shock = model.demand_shock_grid[demand_shock_index]
    
    max_value = -Inf
    best_investment = 0.0
    
    vol_probs = vol_transition_matrix[volatility_index, :]
    
    for investment_january in model.investment_grid
        new_cumulative_investment = cumulative_investment + investment_january
        
        new_cumulative_investment < model.cumulative_investment_grid[1] && continue
        new_cumulative_investment > model.cumulative_investment_grid[end] && continue
        
        # Include profit in January (when demand is observed)
        current_payoff = profit(capital, demand_shock, model) - 
                        investment_cost(investment_january, model, fixed_cost=fixed_cost)
        !isfinite(current_payoff) && continue
        
        expected_april = 0.0
        for (next_vol_idx, vol_prob) in enumerate(vol_probs)
            vol_prob < 1e-10 && continue
            
            value_interp = LinearInterpolation(
                model.cumulative_investment_grid,
                value_april_array[capital_index, :, demand_shock_index, next_vol_idx]
            )
            expected_april += vol_prob * value_interp(new_cumulative_investment)
        end
        
        total_value = current_payoff + model.discount_factor^(1/4) * expected_april
        
        if total_value > max_value
            max_value = total_value
            best_investment = investment_january
        end
    end
    
    return max_value, best_investment
end
