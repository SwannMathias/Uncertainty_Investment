# comment test.

using Plots, Statistics, Distributions, Interpolations
using Distributed

# ps aux | grep '[j]ulia' | awk '{print $2}' | xargs kill -9
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

# Model parameters
@everywhere struct InvestmentModel
    discount_factor::Float64
    depreciation_rate::Float64
    capital_share::Float64
    demand_elasticity::Float64
    investment_cost_param::Float64
    mean_demand::Float64
    std_demand::Float64
    demand_grid::Vector{Float64}
    capital_grid::Vector{Float64}
    investment_grid::Vector{Float64}
end

# Initialize model with default parameters
function create_model(;
    discount_factor = 0.96,
    depreciation_rate = 0.1,
    capital_share = 0.65,
    demand_elasticity = 0.5,
    investment_cost_param = 0.5,
    mean_demand = 1.0,
    std_demand = 0.2,
    num_capital_points = 100,
    num_demand_points = 15,
    num_investment_points = 100
)
    # Create grids
    capital_min, capital_max = 0.1, 5.0
    capital_grid = range(capital_min, capital_max, length=num_capital_points) |> collect
    
    demand_min = max(0.1, mean_demand - 3*std_demand)
    demand_max = mean_demand + 3*std_demand
    demand_grid = range(demand_min, demand_max, length=num_demand_points) |> collect
    
    investment_min, investment_max = -0.5, 1.5
    investment_grid = range(investment_min, investment_max, length=num_investment_points) |> collect
    
    return InvestmentModel(
        discount_factor, 
        depreciation_rate, 
        capital_share, 
        demand_elasticity, 
        investment_cost_param, 
        mean_demand, 
        std_demand, 
        demand_grid, 
        capital_grid, 
        investment_grid
    )
end

# Profit function π(K,D): concave in both arguments
@everywhere function profit(capital::Float64, demand::Float64, model::InvestmentModel)
    return demand * capital^model.capital_share - model.demand_elasticity * capital^2
end

# Investment adjustment cost C(I): convex with possible fixed cost
@everywhere function investment_cost(investment::Float64, model::InvestmentModel; fixed_cost=0.0)
    if abs(investment) < 1e-6 && fixed_cost > 0
        return 0.0  # No cost if no investment
    else
        quadratic_cost = model.investment_cost_param * investment^2 / 2
        fixed_cost_component = fixed_cost * (abs(investment) > 1e-6 ? 1.0 : 0.0)
        return quadratic_cost + fixed_cost_component
    end
end

# Transition for capital: K' = (1-δ)K + I
@everywhere function capital_transition(capital::Float64, investment::Float64, model::InvestmentModel)
    return (1 - model.depreciation_rate) * capital + investment
end

# Demand transition (AR(1) process with shock)
function demand_shock(demand::Float64, model::InvestmentModel, persistence=0.8)
    distribution = Normal(0, model.std_demand)
    shock = rand(distribution)
    next_demand = persistence * demand + (1 - persistence) * model.mean_demand + shock
    return next_demand
end

# Compute value for a single state point (parallelizable unit)
@everywhere function compute_state_value(
    capital_index::Int, 
    demand_index::Int, 
    model::InvestmentModel, 
    value_function::Matrix{Float64}, 
    demand_probabilities::Vector{Float64}, 
    fixed_cost::Float64
)
    capital = model.capital_grid[capital_index]
    demand = model.demand_grid[demand_index]
    
    max_value = -Inf
    best_investment = 0.0
    
    # Optimize over investment choices
    for investment in model.investment_grid
        next_capital = capital_transition(capital, investment, model)
        
        # Check if next_capital is within bounds
        if next_capital < model.capital_grid[1] || next_capital > model.capital_grid[end]
            continue
        end
        
        # Current period payoff: π(K,D) - C(I)
        current_payoff = profit(capital, demand, model) - investment_cost(investment, model, fixed_cost=fixed_cost)
        
        # Expected continuation value: E[V(K', D')]
        expected_continuation = 0.0
        for (next_demand_index, next_demand) in enumerate(model.demand_grid)
            # Interpolate value at next_capital
            value_interpolated = linear_interpolation(
                model.capital_grid, 
                value_function[:, next_demand_index], 
                extrapolation_bc=Line()
            )
            expected_continuation += demand_probabilities[next_demand_index] * value_interpolated(next_capital)
        end
        
        # Total value: current payoff + β * E[V(K', D')]
        total_value = current_payoff + model.discount_factor * expected_continuation
        
        if total_value > max_value
            max_value = total_value
            best_investment = investment
        end
    end
    
    return max_value, best_investment
end

# Value function iteration using pmap
function solve_model(
    model::InvestmentModel; 
    max_iterations=1000, 
    tolerance=1e-6, 
    fixed_cost=0.0,
    verbose=true
)
    
    num_capital_points = length(model.capital_grid)
    num_demand_points = length(model.demand_grid)
    
    # Initialize value function and policy
    value_function = zeros(num_capital_points, num_demand_points)
    value_function_new = zeros(num_capital_points, num_demand_points)
    policy_investment = zeros(num_capital_points, num_demand_points)
    
    # Discrete approximation of demand distribution
    demand_distribution = Normal(model.mean_demand, model.std_demand)
    demand_probabilities = [pdf(demand_distribution, demand) for demand in model.demand_grid]
    demand_probabilities ./= sum(demand_probabilities)  # Normalize
    
    iteration = 0
    difference = Inf
    
    if verbose
        println("Starting parallel value function iteration with $(nworkers()) workers...")
    end
    
    while iteration < max_iterations && difference > tolerance
        iteration += 1
        
        # Create all state indices for parallelization
        state_indices = [(capital_idx, demand_idx) 
                        for capital_idx in 1:num_capital_points, 
                            demand_idx in 1:num_demand_points]
        
        # PARALLEL MAP: Distribute computation across workers
        results = pmap(state_indices) do (capital_idx, demand_idx)
            compute_state_value(
                capital_idx, 
                demand_idx, 
                model, 
                value_function, 
                demand_probabilities, 
                fixed_cost
            )
        end
        
        # Unpack results into matrices
        for (linear_index, (capital_idx, demand_idx)) in enumerate(state_indices)
            value_function_new[capital_idx, demand_idx] = results[linear_index][1]
            policy_investment[capital_idx, demand_idx] = results[linear_index][2]
        end
        
        # Compute convergence metric
        difference = maximum(abs.(value_function_new - value_function))
        value_function .= value_function_new
        
        if verbose && iteration % 50 == 0
            println("Iteration $iteration: difference = $difference")
        end
    end
    
    if verbose
        println("Converged in $iteration iterations")
    end
    
    return value_function, policy_investment
end

# Simulate the economy
function simulate(
    model::InvestmentModel, 
    policy_investment, 
    num_periods=200; 
    initial_capital=1.0, 
    initial_demand=nothing
)
    if isnothing(initial_demand)
        initial_demand = model.mean_demand
    end
    
    capital_path = zeros(num_periods)
    demand_path = zeros(num_periods)
    investment_path = zeros(num_periods)
    profit_path = zeros(num_periods)
    
    capital_path[1] = initial_capital
    demand_path[1] = initial_demand
    
    for time in 1:(num_periods-1)
        # Interpolate policy function
        policy_interpolated = linear_interpolation(
            (model.capital_grid, model.demand_grid), 
            policy_investment, 
            extrapolation_bc=Line()
        )
        investment_path[time] = policy_interpolated(capital_path[time], demand_path[time])
        
        # Compute profit
        profit_path[time] = profit(capital_path[time], demand_path[time], model)
        
        # Transition to next period
        capital_path[time+1] = capital_transition(capital_path[time], investment_path[time], model)
        demand_path[time+1] = max(0.1, demand_shock(demand_path[time], model))
    end
    
    profit_path[num_periods] = profit(capital_path[num_periods], demand_path[num_periods], model)
    
    return capital_path, demand_path, investment_path, profit_path
end

# Main execution
println("="^60)
println("Optimal Investment Under Uncertainty - Parallel Version")
println("="^60)
println("Number of workers: $(nworkers())")

# Create model
println("\n1. Creating model...")
model = create_model(std_demand=0.2)

# Solve model without fixed costs
println("\n2. Solving without fixed costs...")
@time value_no_fc, policy_no_fc = solve_model(model, fixed_cost=0.0)

# Solve model with fixed costs
println("\n3. Solving with fixed costs...")
@time value_with_fc, policy_with_fc = solve_model(model, fixed_cost=0.05, verbose=false)

# Simulate paths
println("\n4. Simulating economy...")
capital_no_fc, demand_no_fc, investment_no_fc, profit_no_fc = simulate(model, policy_no_fc, 200)
capital_with_fc, demand_with_fc, investment_with_fc, profit_with_fc = simulate(model, policy_with_fc, 200)

# Plotting
println("\n5. Creating plots...")

# Policy function plots
plot1 = plot(
    model.capital_grid, 
    policy_no_fc[:, 8], 
    label="No fixed cost", 
    lw=2,
    xlabel="Capital (K)", 
    ylabel="Investment (I)",
    title="Investment Policy Function",
    legend=:topright
)
plot!(
    plot1, 
    model.capital_grid, 
    policy_with_fc[:, 8], 
    label="With fixed costs", 
    lw=2, 
    linestyle=:dash
)

# Capital path simulation
plot2 = plot(
    capital_no_fc, 
    label="No fixed cost", 
    lw=2,
    xlabel="Time", 
    ylabel="Capital",
    title="Capital Path", 
    legend=:bottomright
)
plot!(
    plot2, 
    capital_with_fc, 
    label="With fixed cost", 
    lw=2, 
    linestyle=:dash
)

# Investment path simulation
plot3 = plot(
    investment_no_fc, 
    label="No fixed cost", 
    lw=2,
    xlabel="Time", 
    ylabel="Investment",
    title="Investment Path", 
    legend=:topright
)
plot!(
    plot3, 
    investment_with_fc, 
    label="With fixed cost", 
    lw=2, 
    linestyle=:dash
)

# Demand path
plot4 = plot(
    demand_no_fc, 
    label="Demand shocks", 
    lw=2, 
    color=:red,
    xlabel="Time", 
    ylabel="Demand",
    title="Demand Path", 
    legend=:topright
)

# Combine plots
plot(plot1, plot2, plot3, plot4, layout=(2,2), size=(1000, 800))

println("\n" * "="^60)
println("Simulation complete!")
println("="^60)
println("\nKey Statistics:")
println("  Average Capital (no FC): $(round(mean(capital_no_fc), digits=3))")
println("  Average Capital (with FC): $(round(mean(capital_with_fc), digits=3))")
println("  Investment volatility (no FC): $(round(std(investment_no_fc), digits=3))")
println("  Investment volatility (with FC): $(round(std(investment_with_fc), digits=3))")
println("  Inaction frequency (with FC): $(round(100*mean(abs.(investment_with_fc) .< 0.01), digits=1))%")