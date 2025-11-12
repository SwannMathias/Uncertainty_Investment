
# ============================================================================
# MODEL STRUCTURE
# ============================================================================

struct InvestmentModel
    # Profit function parameters (Abel & Eberly 1996)
    h::Float64                # Scale parameter
    gamma::Float64            # Demand elasticity (0 < gamma < 1)
    
    # Demand process (GBM with stochastic volatility)
    mu_X::Float64             # Drift of X_t
    sigma_X_mean::Float64     # Mean volatility
    rho_sigma::Float64        # Volatility persistence (AR(1))
    sigma_v::Float64          # Volatility of volatility
    
    # Investment and capital parameters
    discount_factor::Float64
    depreciation_rate::Float64
    investment_cost_param::Float64
    
    # State space grids
    capital_grid::Vector{Float64}
    demand_shock_grid::Vector{Float64}
    volatility_grid::Vector{Float64}
    investment_grid::Vector{Float64}
    cumulative_investment_grid::Vector{Float64}
end

"""
    create_model(; kwargs...)

Initialize InvestmentModel with specified parameters.

# Key Parameters
- `h`: Profit scale parameter
- `gamma`: Demand elasticity
- `mu_X`: Demand drift
- `sigma_X_mean`: Mean volatility
- `discount_factor`: Quarterly discount factor
- `depreciation_rate`: Annual depreciation rate
"""
function create_model(;
    h = 1.0,
    gamma = 0.5,
    mu_X = 0.01,
    sigma_X_mean = 0.1,
    rho_sigma = 0.9,
    sigma_v = 0.01,
    discount_factor = 0.96,
    depreciation_rate = 0.1,
    investment_cost_param = 0.5,
    num_capital_points = 60,
    num_demand_points = 15,
    num_volatility_points = 10,
    num_investment_points = 40,
    num_cumulative_investment_points = 60
)
    capital_grid = range(0.1, 5.0, length=num_capital_points) |> collect
    demand_shock_grid = range(0.2, 3.0, length=num_demand_points) |> collect
    volatility_grid = range(0.02, 0.3, length=num_volatility_points) |> collect
    investment_grid = range(-0.5, 1.0, length=num_investment_points) |> collect
    cumulative_investment_grid = range(-1.0, 3.0, length=num_cumulative_investment_points) |> collect
    
    return InvestmentModel(
        h, gamma, mu_X, sigma_X_mean, rho_sigma, sigma_v,
        discount_factor, depreciation_rate, investment_cost_param,
        capital_grid, demand_shock_grid, volatility_grid, 
        investment_grid, cumulative_investment_grid
    )
end

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

"""
    profit(capital, demand_shock, model)

Compute profit π(K,X) = (h/(1-γ)) * X^γ * K^(1-γ)
"""
function profit(capital::Float64, demand_shock::Float64, model::InvestmentModel)
    capital <= 0 && return -Inf
    demand_shock <= 0 && return -Inf
    return (model.h / (1 - model.gamma)) * (demand_shock^model.gamma) * (capital^(1 - model.gamma))
end

"""
    investment_cost(investment, model; fixed_cost)

Compute adjustment cost C(I) = (c/2)*I² + F*𝟙(I≠0)
"""
function investment_cost(investment::Float64, model::InvestmentModel; fixed_cost=0.0)
    abs(investment) < 1e-6 && return 0.0
    return model.investment_cost_param * investment^2 / 2 + fixed_cost
end

"""
    capital_transition(capital, cumulative_investment, model)

Capital law of motion: K_{t+1} = (1-δ)K_t + I_cumulative
"""
function capital_transition(capital::Float64, cumulative_investment::Float64, model::InvestmentModel)
    return (1 - model.depreciation_rate) * capital + cumulative_investment
end

# ============================================================================
# TRANSITION PROBABILITIES (OPTIMIZED)
# ============================================================================

"""
    precompute_volatility_transitions(model, dt)

Precompute volatility transition matrices for efficiency.
Returns: matrix where entry [i,j] = P(σ_j | σ_i)
"""
function precompute_volatility_transitions(model::InvestmentModel, dt::Float64=0.25)
    num_vol = length(model.volatility_grid)
    transition_matrix = zeros(num_vol, num_vol)
    
    for (i, vol_current) in enumerate(model.volatility_grid)
        vol_mean = model.rho_sigma * vol_current + (1 - model.rho_sigma) * model.sigma_X_mean
        vol_std = model.sigma_v * sqrt(dt)
        vol_dist = Normal(vol_mean, vol_std)
        
        # Vectorized probability computation
        probs = pdf.(vol_dist, model.volatility_grid)
        probs ./= sum(probs)
        transition_matrix[i, :] = probs
    end
    
    return transition_matrix
end

"""
    precompute_demand_transitions(model)

Precompute demand shock transition probabilities.
Returns: 3D array [current_shock, next_vol, next_shock]
"""
function precompute_demand_transitions(model::InvestmentModel, dt::Float64=1.0)
    num_D = length(model.demand_shock_grid)
    num_vol = length(model.volatility_grid)
    
    transition_array = zeros(num_D, num_vol, num_D)
    
    for (i, current_shock) in enumerate(model.demand_shock_grid)
        for (j, next_volatility) in enumerate(model.volatility_grid)
            drift = model.mu_X - 0.5 * next_volatility^2
            log_mean = log(current_shock) + drift * dt
            log_std = next_volatility * sqrt(dt)
            log_dist = Normal(log_mean, log_std)
            
            # Vectorized probability computation with Jacobian
            probs = zeros(num_D)
            for (k, X_next) in enumerate(model.demand_shock_grid)
                if X_next > 0
                    probs[k] = pdf(log_dist, log(X_next)) / X_next
                end
            end
            probs ./= sum(probs)
            transition_array[i, j, :] = probs
        end
    end
    
    return transition_array
end
