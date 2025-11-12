using Plots, Statistics, Distributions, Interpolations
using Distributed

# Control maximum number of parallel workers
max_workers = 100
current_workers = nworkers()

if current_workers < max_workers
    workers_to_add = max_workers - current_workers
    addprocs(workers_to_add)
    println("Added $workers_to_add workers. Total workers: $(nworkers())")
else
    println("Already have $current_workers workers (max: $max_workers)")
end

# Load packages on all workers
@everywhere using Statistics, Distributions, Interpolations

# Model parameters with intra-year periods
@everywhere struct InvestmentModel
    # Profit function parameters (Abel & Eberly 1996)
    h::Float64                # Scale parameter in profit function
    gamma::Float64            # Demand elasticity (0 < gamma < 1)

    # Demand process parameters (GBM with stochastic volatility)
    mu_X::Float64             # Drift rate of X_t
    sigma_X_mean::Float64     # Mean volatility level
    rho_sigma::Float64        # Persistence of volatility (AR(1) coefficient)
    sigma_v::Float64          # Volatility of volatility

    # Investment and capital parameters
    discount_factor::Float64
    depreciation_rate::Float64
    investment_cost_param::Float64

    # Grids for numerical computation
    capital_grid::Vector{Float64}
    demand_shock_grid::Vector{Float64}      # Grid for X_t (demand shock)
    volatility_grid::Vector{Float64}        # Grid for σ_t
    investment_grid::Vector{Float64}
    cumulative_investment_grid::Vector{Float64}
end

# Initialize model with default parameters
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
    # Capital grid
    capital_min, capital_max = 0.1, 5.0
    capital_grid = range(capital_min, capital_max, length=num_capital_points) |> collect

    # Demand shock grid (X_t in GBM)
    demand_shock_min, demand_shock_max = 0.2, 3.0
    demand_shock_grid = range(demand_shock_min, demand_shock_max, length=num_demand_points) |> collect

    # Volatility grid (σ_t)
    volatility_min, volatility_max = 0.02, 0.3
    volatility_grid = range(volatility_min, volatility_max, length=num_volatility_points) |> collect

    # Investment grid
    investment_min, investment_max = -0.5, 1.0
    investment_grid = range(investment_min, investment_max, length=num_investment_points) |> collect

    # Cumulative investment grid (within year)
    cumulative_investment_min, cumulative_investment_max = -1.0, 3.0
    cumulative_investment_grid = range(cumulative_investment_min, cumulative_investment_max, 
                                      length=num_cumulative_investment_points) |> collect

    return InvestmentModel(
        h, gamma, mu_X, sigma_X_mean, rho_sigma, sigma_v,
        discount_factor, depreciation_rate, investment_cost_param,
        capital_grid, demand_shock_grid, volatility_grid, 
        investment_grid, cumulative_investment_grid
    )
end

# Profit function π(K,X) from Abel & Eberly (1996)
@everywhere function profit(capital::Float64, demand_shock::Float64, model::InvestmentModel)
    if capital <= 0 || demand_shock <= 0
        return -Inf
    end
    return (model.h / (1 - model.gamma)) * (demand_shock^model.gamma) * (capital^(1 - model.gamma))
end

# Investment adjustment cost C(I): convex with possible fixed cost
@everywhere function investment_cost(investment::Float64, model::InvestmentModel; fixed_cost=0.0)
    if abs(investment) < 1e-6 && fixed_cost > 0
        return 0.0
    else
        quadratic_cost = model.investment_cost_param * investment^2 / 2 # On pourrait différencier les fonctions dans l'année. 
        fixed_cost_component = fixed_cost * (abs(investment) > 1e-6 ? 1.0 : 0.0)
        return quadratic_cost + fixed_cost_component
    end
end

# Capital transition: K_{t+1} = (1-δ)K_t + I_cumulative_t
@everywhere function capital_transition(capital::Float64, cumulative_investment::Float64, model::InvestmentModel)
    return (1 - model.depreciation_rate) * capital + cumulative_investment
end

# Volatility evolves continuously within year via AR(1) with small innovations
@everywhere function volatility_within_year_probabilities(
    volatility_current_quarter::Float64, 
    model::InvestmentModel,
    dt::Float64 = 0.25  # One quarter time step
)
    # Volatility evolves continuously: σ_{t,q+1} = ρ*σ_{t,q} + (1-ρ)*σ̄ + ε
    # where ε ~ N(0, σ_v*√dt) for within-year transitions
    vol_mean = model.rho_sigma * volatility_current_quarter + 
               (1 - model.rho_sigma) * model.sigma_X_mean
    vol_std = model.sigma_v * sqrt(dt)  # Scaled by time step
    vol_dist = Normal(vol_mean, vol_std)
    
    probs = [pdf(vol_dist, v) for v in model.volatility_grid]
    probs ./= sum(probs)
    return probs
end

# Volatility transition between years (from Q4 of year t to Q1 of year t+1)
@everywhere function volatility_between_year_probabilities(
    volatility_q4::Float64, 
    model::InvestmentModel,
    dt::Float64 = 0.25  # Q4 to next Q1
)
    # Same AR(1) process but this is the transition that matters for next year's X
    vol_mean = model.rho_sigma * volatility_q4 + 
               (1 - model.rho_sigma) * model.sigma_X_mean
    vol_std = model.sigma_v * sqrt(dt)
    vol_dist = Normal(vol_mean, vol_std)
    
    probs = [pdf(vol_dist, v) for v in model.volatility_grid]
    probs ./= sum(probs)
    return probs
end

# X_t is only observed in January
# Given σ_{t+1}, we can forecast the distribution of X_{t+1}
# X_{t+1} = X_t * exp((μ - σ²_{t+1}/2)*Δt + σ_{t+1}*√Δt*Z)
# where Δt = 1 year between January observations
@everywhere function demand_shock_annual_transition_probabilities(
    current_shock::Float64, 
    volatility_next_year::Float64, 
    model::InvestmentModel,
    dt::Float64 = 1.0  # One full year between X observations
)
    # GBM for annual transition
    drift = model.mu_X - 0.5 * volatility_next_year^2
    log_mean = log(current_shock) + drift * dt
    log_std = volatility_next_year * sqrt(dt)
    
    # Distribution of log(X_{t+1})
    log_dist = Normal(log_mean, log_std)
    
    # Compute probabilities for grid points
    probs = zeros(length(model.demand_shock_grid))
    for (i, X_next) in enumerate(model.demand_shock_grid)
        if X_next > 0
            probs[i] = pdf(log_dist, log(X_next)) / X_next  # Jacobian adjustment
        end
    end
    
    # Normalize
    probs ./= sum(probs)
    return probs
end

#=
TIMELINE OF A YEAR t:

JANUARY (Q1) - Start of year t:
- Firm sells output: observes realized X_t (demand shock at year t)
- Observes current volatility σ_{t,Q1}
- Has capital K_t from previous year
- Chooses I_january based on (K_t, X_t, σ_{t,Q1})

APRIL (Q2):
- Observes σ_{t,Q2} (volatility evolves continuously via AR(1))
- Updates forecast of σ_{t+1} based on more recent σ observation
- This improves forecast of X_{t+1} distribution
- Chooses I_april based on (K_t, cumulative_I, σ_{t,Q2})
- Note: X_t is NOT observed again (only observed in January)

AUGUST (Q3):
- Observes σ_{t,Q3} (even more recent volatility)
- Further refines forecast of σ_{t+1}
- Better prediction of X_{t+1}
- Chooses I_august based on (K_t, cumulative_I, σ_{t,Q3})

OCTOBER (Q4):
- Observes σ_{t,Q4} (most recent volatility before year-end)
- Best forecast of σ_{t+1} via AR(1): E[σ_{t+1}|σ_{t,Q4}] has lowest uncertainty
- Most precise prediction of X_{t+1} distribution
- Chooses I_october based on (K_t, cumulative_I, σ_{t,Q4})

END OF YEAR:
- K_{t+1} = (1-δ)K_t + Σ I_quarters becomes effective
- σ_{t+1} realized from AR(1): σ_{t+1} = ρ*σ_{t,Q4} + (1-ρ)*σ̄ + ε_{t+1}
- X_{t+1} realized from GBM given σ_{t+1}

KEY INSIGHTS: 
- X_t is observed ONLY in January (annual sales/revenue realization)
- σ_t is observed CONTINUOUSLY throughout the year (market/demand volatility)
- Later quarters have progressively better information about σ_{t+1}
- Better σ_{t+1} forecast → better X_{t+1} distribution forecast → better investment decision
- This creates OPTION VALUE of waiting: defer investment to later quarters when uncertainty about future demand is lower
=#

# OCTOBER (Q4): Last investment decision of the year
# State: (K_t, cumulative_I, σ_{t,Q4})
# Note: X_t was observed in January, not observed again
@everywhere function value_october(
    capital_index::Int,
    cumulative_investment_index::Int,
    demand_shock_index::Int,  # This is X_t observed in January
    volatility_index::Int,     # This is σ_{t,Q4} observed now
    model::InvestmentModel,
    value_january_next::Array{Float64, 4},
    volatility_between_year_probs::Vector{Float64},  # P(σ_{t+1}|σ_{t,Q4})
    fixed_cost::Float64
)
    capital = model.capital_grid[capital_index]
    cumulative_investment = model.cumulative_investment_grid[cumulative_investment_index]
    demand_shock = model.demand_shock_grid[demand_shock_index]  # X_t from January
    volatility_q4 = model.volatility_grid[volatility_index]     # Current σ_{t,Q4}
    
    max_value = -Inf
    best_investment = 0.0
    
    for investment_october in model.investment_grid
        total_investment = cumulative_investment + investment_october
        next_capital = capital_transition(capital, total_investment, model)
        
        if next_capital < model.capital_grid[1] || next_capital > model.capital_grid[end]
            continue
        end
        
        # Current payoff in October (still using K_t and X_t from January)
        #current_payoff = profit(capital, demand_shock, model) - 
        #                investment_cost(investment_october, model, fixed_cost=fixed_cost)
        current_payoff = -investment_cost(investment_october, model, fixed_cost=fixed_cost)
        if !isfinite(current_payoff)
            continue
        end
        
        # Expected continuation value for next January
        # Key: forecast X_{t+1} based on X_t and σ_{t+1}
        expected_continuation = 0.0
        for (next_vol_idx, next_volatility) in enumerate(model.volatility_grid)
            vol_prob = volatility_between_year_probs[next_vol_idx]  # P(σ_{t+1}|σ_{t,Q4})
            
            # Given σ_{t+1}, compute P(X_{t+1}|X_t, σ_{t+1})
            next_demand_probs = demand_shock_annual_transition_probabilities(
                demand_shock, next_volatility, model, 1.0  # 1 year transition
            )
            
            for (next_shock_idx, next_shock) in enumerate(model.demand_shock_grid)
                shock_prob = next_demand_probs[next_shock_idx]
                
                # Interpolate next year's value at next January
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

# AUGUST (Q3): Choose I_august
# State: (K_t, cumulative_I, X_t, σ_{t,Q3})
@everywhere function value_august(
    capital_index::Int,
    cumulative_investment_index::Int,
    demand_shock_index::Int,  # X_t from January
    volatility_index::Int,     # σ_{t,Q3} observed now
    model::InvestmentModel,
    value_october_array::Array{Float64, 4},
    volatility_within_year_probs::Vector{Float64},  # P(σ_{t,Q4}|σ_{t,Q3})
    fixed_cost::Float64
)
    capital = model.capital_grid[capital_index]
    cumulative_investment = model.cumulative_investment_grid[cumulative_investment_index]
    demand_shock = model.demand_shock_grid[demand_shock_index]  # X_t from January
    volatility_q3 = model.volatility_grid[volatility_index]
    
    max_value = -Inf
    best_investment = 0.0
    
    for investment_august in model.investment_grid
        new_cumulative_investment = cumulative_investment + investment_august
        
        if new_cumulative_investment < model.cumulative_investment_grid[1] || 
           new_cumulative_investment > model.cumulative_investment_grid[end]
            continue
        end
        
        # Current payoff
        #current_payoff = profit(capital, demand_shock, model) - 
        #                investment_cost(investment_august, model, fixed_cost=fixed_cost)
        current_payoff = -investment_cost(investment_august,model, fixed_cost = fixed_cost)
        
        if !isfinite(current_payoff)
            continue
        end
        
        # Expected value in October
        # Volatility evolves from Q3 to Q4, but X_t stays the same (not observed again)
        expected_october = 0.0
        for (next_vol_idx, next_volatility_q4) in enumerate(model.volatility_grid)
            vol_prob = volatility_within_year_probs[next_vol_idx]  # P(σ_{t,Q4}|σ_{t,Q3})
            
            # Interpolate October value (same X_t, updated σ)
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

# APRIL (Q2): Choose I_april
# State: (K_t, cumulative_I, X_t, σ_{t,Q2})
@everywhere function value_april(
    capital_index::Int,
    cumulative_investment_index::Int,
    demand_shock_index::Int,  # X_t from January
    volatility_index::Int,     # σ_{t,Q2} observed now
    model::InvestmentModel,
    value_august_array::Array{Float64, 4},
    volatility_within_year_probs::Vector{Float64},  # P(σ_{t,Q3}|σ_{t,Q2})
    fixed_cost::Float64
)
    capital = model.capital_grid[capital_index]
    cumulative_investment = model.cumulative_investment_grid[cumulative_investment_index]
    demand_shock = model.demand_shock_grid[demand_shock_index]  # X_t from January
    volatility_q2 = model.volatility_grid[volatility_index]
    
    max_value = -Inf
    best_investment = 0.0
    
    for investment_april in model.investment_grid
        new_cumulative_investment = cumulative_investment + investment_april
        
        if new_cumulative_investment < model.cumulative_investment_grid[1] || 
           new_cumulative_investment > model.cumulative_investment_grid[end]
            continue
        end
        
        # Current payoff
        #current_payoff = profit(capital, demand_shock, model) - 
        #                investment_cost(investment_april, model, fixed_cost=fixed_cost)
        current_payoff = -investment_cost(investment_april, model, fixed_cost=fixed_cost)
        
        if !isfinite(current_payoff)
            continue
        end
        
        # Expected value in August
        expected_august = 0.0
        for (next_vol_idx, next_volatility_q3) in enumerate(model.volatility_grid)
            vol_prob = volatility_within_year_probs[next_vol_idx]  # P(σ_{t,Q3}|σ_{t,Q2})
            
            # Interpolate August value (same X_t, updated σ)
            value_interp = LinearInterpolation(
                model.cumulative_investment_grid,
                value_august_array[capital_index, :, demand_shock_index, next_vol_idx]
            )
        end

        total_value = current_payoff + model.discount_factor^(1/4) * expected_august
        
        if total_value > max_value
            max_value = total_value
            best_investment = investment_april
        end
    end
    
    return max_value, best_investment
end
# January (Q1): Choose I_january
# State: (K_t, cumulative_I, X_t, σ_{t})
@everywhere function value_january(
    capital_index::Int,
    cumulative_investment_index::Int,
    demand_shock_index::Int,  # X_t from January
    volatility_index::Int,     # σ_{t} observed now
    model::InvestmentModel,
    value_april_array::Array{Float64, 4},
    volatility_within_year_probs::Vector{Float64},  # P(σ_{t,Q2}|σ_{t,Q1})
    fixed_cost::Float64
)
    capital = model.capital_grid[capital_index]
    cumulative_investment = model.cumulative_investment_grid[cumulative_investment_index]
    demand_shock = model.demand_shock_grid[demand_shock_index]  # X_t from January
    volatility_q2 = model.volatility_grid[volatility_index]
    
    max_value = -Inf
    best_investment = 0.0
    
    for investment_january in model.investment_grid
        new_cumulative_investment = cumulative_investment + investment_january
        
        if new_cumulative_investment < model.cumulative_investment_grid[1] || 
           new_cumulative_investment > model.cumulative_investment_grid[end]
            continue
        end
        
        # Current payoff
        current_payoff = profit(capital, demand_shock, model) - 
                        investment_cost(investment_january, model, fixed_cost=fixed_cost)
        
        if !isfinite(current_payoff)
            continue
        end
        
        # Expected value in April
        expected_april = 0.0
        for (next_vol_idx, next_volatility_q3) in enumerate(model.volatility_grid)
            vol_prob = volatility_within_year_probs[next_vol_idx]  # P(σ_{t,Q2}|σ_{t,Q1})
            
            # Interpolate April value (same X_t, updated σ)
            value_interp = LinearInterpolation(
                model.cumulative_investment_grid,
                value_april_array[capital_index, :, demand_shock_index, next_vol_idx]
            )
        end
        total_value = current_payoff + model.discount_factor^(1/4) * expected_april
        
        if total_value > max_value
            max_value = total_value
            best_investment = investment_january
        end
    end
    
    return max_value, best_investment
end

# Solve the full model via backward induction within year, value iteration across years
@everywhere function solve_model(
    model::InvestmentModel;
    max_iterations=1000,
    tolerance=1e-6,
    fixed_cost=0.0,
    verbose=true
)
    num_K = length(model.capital_grid)
    num_I_cum = length(model.cumulative_investment_grid)
    num_D = length(model.demand_shock_grid)
    num_vol = length(model.volatility_grid)

    # Initialize value and policy arrays
    V_january  = zeros(num_K, num_I_cum, num_D, num_vol)
    V_april    = similar(V_january)
    V_august   = similar(V_january)
    V_october  = similar(V_january)

    policy_january = zeros(num_K, num_D, num_vol)
    policy_april   = zeros(num_K, num_I_cum, num_D, num_vol)
    policy_august  = similar(policy_april)
    policy_october = similar(policy_april)

    V_january_new = similar(V_january) # Needed to store the next period values in October

    iteration, difference = 0, Inf

    # Precompute index tuples for parallel mapping
    idx_full = collect(CartesianIndices((num_K, num_I_cum, num_D, num_vol)))
    idx_january = collect(CartesianIndices((num_K, num_D, num_vol)))

    if verbose
        println("Starting parallel value function iteration with $(nworkers()) workers...")
        println("State space: K=$num_K × I_cum=$num_I_cum × D=$num_D × σ=$num_vol")
    end

    while iteration < max_iterations && difference > tolerance
        iteration += 1

        # === OCTOBER ===
        if verbose && iteration % 1 == 0
            println("  Iteration $iteration: Solving October...")
        end

        results_october = pmap(idx_full) do idx
            k, i, d, v = Tuple(idx)
            vol_q4 = model.volatility_grid[v]
            vol_between = volatility_between_year_probabilities(vol_q4, model, 0.25)
            value_october(k, i, d, v, model, V_january, vol_between, fixed_cost)
        end

        V_october .= reshape(first.(results_october), size(V_october))
        policy_october .= reshape(last.(results_october), size(policy_october))

        # === AUGUST ===
        if verbose && iteration % 1 == 0
            println("  Iteration $iteration: Solving August...")
        end

        results_august = pmap(idx_full) do idx
            k, i, d, v = Tuple(idx)
            vol_q3 = model.volatility_grid[v]
            vol_within = volatility_within_year_probabilities(vol_q3, model, 0.25)
            value_august(k, i, d, v, model, V_october, vol_within, fixed_cost)
        end

        V_august .= reshape(first.(results_august), size(V_august))
        policy_august .= reshape(last.(results_august), size(policy_august))

        # === APRIL ===
        if verbose && iteration % 10 == 0
            println("  Iteration $iteration: Solving April...")
        end

        results_april = pmap(idx_full) do idx
            k, i, d, v = Tuple(idx)
            vol_q2 = model.volatility_grid[v]
            vol_within = volatility_within_year_probabilities(vol_q2, model, 0.25)
            value_april(k, i, d, v, model, V_august, vol_within, fixed_cost)
        end

        V_april .= reshape(first.(results_april), size(V_april))
        policy_april .= reshape(last.(results_april), size(policy_april))

        # === JANUARY ===
        if verbose && iteration % 10 == 0
            println("  Iteration $iteration: Solving January...")
        end

        results_january = pmap(idx_january) do idx
            k, d, v = Tuple(idx)
            vol_q1 = model.volatility_grid[v]
            vol_within = volatility_within_year_probabilities(vol_q1, model, 0.25)
            value_january(k, 1, d, v, model, V_april, vol_within, fixed_cost)
        end

        V_january_new[:, 1, :, :] .= reshape(first.(results_january), size(V_january[:, 1, :, :]))
        policy_january .= reshape(last.(results_january), size(policy_january))

        difference = maximum(abs.(V_january_new .- V_january))
        V_january .= V_january_new

        if verbose && iteration % 10 == 0
            println("  Iteration $iteration: difference = $(round(difference, sigdigits=4))")
        end
    end

    if verbose
        println("Converged in $iteration iterations (diff = $(round(difference, sigdigits=4)))")
    end

    return (
        V_january=V_january,
        V_april=V_april,
        V_august=V_august,
        V_october=V_october,
        policy_january=policy_january,
        policy_april=policy_april,
        policy_august=policy_august,
        policy_october=policy_october
    )
end


# Main execution
println("="^60)
println("Optimal Investment with Intra-Year Adjustments")
println("="^60)
println("Number of workers: $(nworkers())")

# Create model
println("\n1. Creating model...")
model = create_model()

# Solve model
println("\n2. Solving model with quarterly investment decisions...")
println("   Timeline: January → April → August → October → Next January")
@time solution = solve_model(model, fixed_cost=0.0, max_iterations=50)

println("\n" * "="^60)
println("Solution computed!")
println("="^60)
println("\nPolicy functions available for:")
println("  - January (start of year)")
println("  - April (after first demand observation)")
println("  - August (after second demand observation)")
println("  - October (after third demand observation)")
println("\nKey insight: Firms can adjust investment based on")
println("observed demand, which reveals information about volatility.")


results = Dict{Float64, Any}()
results[0] = solution

"""
    compute_investment_plan(model, policies, weights)

Compute expected investment plans across states and revisions.

# Arguments
- `model`: InvestmentModel instance
- `policies`: Dict with keys :january, :april, :august, :october (each is an array of optimal I decisions)
- `weights`: optional weighting scheme over (K, D, vol) states (default = uniform)

# Returns
- Dict with fields :plan (quarterly expected plans) and :revision (change in plan)
"""
function compute_investment_plan(model, policies; weights=nothing)
    num_K = length(model.capital_grid)
    num_D = length(model.demand_shock_grid)
    num_vol = length(model.volatility_grid)

    if weights === nothing
        weights = ones(num_K, num_D, num_vol)
        weights ./= sum(weights)
    end

    # Compute expectation of investment (policy) at each quarter
    plan = Dict(
        :january => sum(policies[:january] .* weights),
        :april   => sum(mean(policies[:april], dims=2)[:, 1, :, :] .* weights),  # avg over I_cum dim
        :august  => sum(mean(policies[:august], dims=2)[:, 1, :, :] .* weights),
        :october => sum(mean(policies[:october], dims=2)[:, 1, :, :] .* weights)
    )

    # Compute revision (Δ plan quarter-to-quarter)
    revision = Dict(
        :Q1_to_Q2 => plan[:april] - plan[:january],
        :Q2_to_Q3 => plan[:august] - plan[:april],
        :Q3_to_Q4 => plan[:october] - plan[:august]
    )

    return Dict(:plan => plan, :revision => revision)
end

"""
    simulate_capital_dynamics(model, policies, initial_state, periods)

Simulate quarterly capital evolution within one year.
"""
function simulate_capital_dynamics(model, policies;
    initial_state = (round(Int, length(model.capital_grid)/2),
                     round(Int, length(model.demand_shock_grid)/2),
                     round(Int, length(model.volatility_grid)/2)),
    periods = [:january, :april, :august, :october]
)
    k_idx, d_idx, v_idx = initial_state
    capital_path = Float64[]
    investment_path = Float64[]

    current_K = model.capital_grid[k_idx]
    push!(capital_path, current_K)

    for q in periods
        policy = policies[q]
        if q == :january
            I = policy[k_idx, d_idx, v_idx]
        else
            I = policy[k_idx, 1, d_idx, v_idx]
        end
        push!(investment_path, I)
        # Simple capital accumulation: K_{t+1} = (1-δ)K_t + I
        δ = model.depreciation_rate
        current_K = (1 - δ) * current_K + I
        push!(capital_path, current_K)
    end

    return (capital_path=capital_path, investment_path=investment_path)
end

for (fc, sol) in results
    policies = Dict(
        :january => sol.policy_january,
        :april   => sol.policy_april,
        :august  => sol.policy_august,
        :october => sol.policy_october
    )

    sim = simulate_capital_dynamics(model, policies)
    quarters = ["Jan", "Apr", "Aug", "Oct", "Next Jan"]

    plot(quarters, sim.capital_path, label="Fixed cost = $fc",
         xlabel="Quarter", ylabel="Capital stock", lw=2, legend=:bottomright)
end
title!("Capital evolution across quarters")

inv_summary = Dict{Float64, Any}()

for (fc, sol) in results
    policies = Dict(
        :january => sol.policy_january,
        :april   => sol.policy_april,
        :august  => sol.policy_august,
        :october => sol.policy_october
    )
    inv_summary[fc] = compute_investment_plan(model, policies)
end

# Plot expected plan
fixed_cost_grid =[0]
plans = [inv_summary[fc][:plan] for fc in [fixed_cost_grid]]
quarters = ["Jan", "Apr", "Aug", "Oct"]

plot()
for (i, fc) in enumerate(fixed_cost_grid)
    vals = [plans[i][q] for q in [:january, :april, :august, :october]]
    plot!(quarters, vals, lw=2, label="Fixed cost = $fc")
end
xlabel!("Quarter")
ylabel!("Expected investment plan")
title!("Investment plan evolution with fixed cost")
display(current())

# Plot revisions
plot()
for (i, fc) in enumerate(fixed_cost_grid)
    rev = inv_summary[fc][:revision]
    vals = [rev[:Q1_to_Q2], rev[:Q2_to_Q3], rev[:Q3_to_Q4]]
    plot!(["Q1→Q2","Q2→Q3","Q3→Q4"], vals, lw=2, label="Fixed cost = $fc")
end
xlabel!("Revision interval")
ylabel!("Change in expected investment")
title!("Investment plan revisions")
display(current())

