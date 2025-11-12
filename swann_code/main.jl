
using Distributed
# ============================================================================
# PARALLELIZATION SETUP
# ============================================================================

"""
    setup_workers(max_workers::Int=100)

Initialize parallel workers for computation.
"""
function setup_workers(max_workers::Int=100)
    current_workers = nworkers()
    
    if current_workers < max_workers
        workers_to_add = max_workers - current_workers
        addprocs(workers_to_add)
        println("Added $workers_to_add workers. Total workers: $(nworkers())")
    else
        println("Already have $current_workers workers (max: $max_workers)")
    end
    
end


setup_workers(100)
@everywhere using Plots, Statistics, Distributions, Interpolations
@everywhere include("models.jl")
@everywhere include("solver.jl")
@everywhere include("value_functions.jl")
@everywhere include("expectations.jl")
@everywhere include("vizualisation.jl")


# ============================================================================
# TESTING ENVIRONMENT
# ============================================================================

"""
    run_fixed_cost_experiment(model; fixed_cost_grid, max_iterations, verbose)

Run comprehensive testing across multiple fixed cost levels.

# Arguments
- `fixed_cost_grid`: Vector of fixed costs to test
- `max_iterations`: Max iterations for value function iteration
- `verbose`: Print progress

# Returns
Dict mapping fixed_cost → solution
"""
function run_fixed_cost_experiment(
    model::InvestmentModel;
    fixed_cost_grid::Vector{Float64} = [0.0, 0.02, 0.05, 0.1, 0.15],
    max_iterations::Int = 50,
    tolerance::Float64 = 1e-6,
    verbose::Bool = true
)
    results = Dict{Float64, Any}()
    
    verbose && println("\n" * "="^60)
    verbose && println("FIXED COST EXPERIMENT")
    verbose && println("="^60)
    verbose && println("Testing $(length(fixed_cost_grid)) fixed cost levels")
    verbose && println("Fixed costs: $fixed_cost_grid")
    
    for (i, fc) in enumerate(fixed_cost_grid)
        verbose && println("\n[$(i)/$(length(fixed_cost_grid))] Solving for fixed cost = $fc")
        
        t_start = time()
        solution = solve_model(model, fixed_cost=fc, max_iterations=max_iterations, 
                              tolerance=tolerance, verbose=verbose)
        t_elapsed = time() - t_start
        
        results[fc] = solution
        verbose && println("  ✓ Completed in $(round(t_elapsed, digits=2))s")
    end
    
    verbose && println("\n" * "="^60)
    verbose && println("EXPERIMENT COMPLETE")
    verbose && println("="^60)
    
    return results
end

"""
    generate_summary_statistics(results_dict, model)

Generate comprehensive summary statistics from experiment results.

# Returns
DataFrame with summary statistics for each fixed cost level
"""
function generate_summary_statistics(results_dict::Dict, model::InvestmentModel)
    fixed_costs = sort(collect(keys(results_dict)))
    
    summary = []
    
    for fc in fixed_costs
        sol = results_dict[fc]
        policies = Dict(
            :january => sol.policy_january,
            :april => sol.policy_april,
            :august => sol.policy_august,
            :october => sol.policy_october
        )
        
        # Compute expectations
        exp_data = compute_investment_expectations(model, policies)
        
        # Quarterly averages
        avg_I_jan = mean(sol.policy_january)
        avg_I_apr = mean(sol.policy_april)
        avg_I_aug = mean(sol.policy_august)
        avg_I_oct = mean(sol.policy_october)
        
        # Inaction rates (fraction of states with |I| < 1e-6)
        inaction_jan = mean(abs.(sol.policy_january) .< 1e-6)
        inaction_apr = mean(abs.(sol.policy_april) .< 1e-6)
        inaction_aug = mean(abs.(sol.policy_august) .< 1e-6)
        inaction_oct = mean(abs.(sol.policy_october) .< 1e-6)
        
        # Expected annual investment from January perspective
        exp_annual = exp_data[:expectations][:january]
        
        # Total revision magnitude
        revisions = exp_data[:revisions]
        total_revision = abs(revisions[:jan_to_apr]) + abs(revisions[:apr_to_aug]) + 
                        abs(revisions[:aug_to_oct])
        
        push!(summary, Dict(
            "Fixed Cost" => fc,
            "Exp Annual Investment" => round(exp_annual, digits=4),
            "Avg I (Jan)" => round(avg_I_jan, digits=4),
            "Avg I (Apr)" => round(avg_I_apr, digits=4),
            "Avg I (Aug)" => round(avg_I_aug, digits=4),
            "Avg I (Oct)" => round(avg_I_oct, digits=4),
            "Inaction Rate (Jan)" => round(inaction_jan, digits=3),
            "Inaction Rate (Oct)" => round(inaction_oct, digits=3),
            "Total Revision" => round(total_revision, digits=4)
        ))
    end
    
    return summary
end

"""
    create_comprehensive_report(results_dict, model; save_plots)

Generate complete analysis report with all visualizations.

# Arguments
- `results_dict`: Experiment results
- `model`: InvestmentModel instance
- `save_plots`: If true, save plots to files

# Returns
Tuple of (summary_statistics, plots)
"""
function create_comprehensive_report(
    results_dict::Dict, 
    model::InvestmentModel;
    save_plots::Bool = false
)
    println("\n" * "="^60)
    println("GENERATING COMPREHENSIVE REPORT")
    println("="^60)
    
    # Summary statistics
    println("\n1. Computing summary statistics...")
    summary = generate_summary_statistics(results_dict, model)
    
    println("\nSummary Statistics:")
    println("-" * "-"^60)
    for stat in summary
        println("Fixed Cost: $(stat["Fixed Cost"])")
        for (key, val) in stat
            key == "Fixed Cost" && continue
            println("  $key: $val")
        end
        println()
    end
    
    # Create visualizations
    println("2. Creating visualizations...")
    
    p_expectations = plot_investment_expectations(results_dict, model)
    println("  ✓ Investment expectations plot")
    
    p_quarterly = plot_quarterly_investment_plans(results_dict, model)
    println("  ✓ Quarterly investment plans plot")
    
    # Create inaction rate comparison
    fixed_costs = sort(collect(keys(results_dict)))
    inaction_jan = [mean(abs.(results_dict[fc].policy_january) .< 1e-6) for fc in fixed_costs]
    inaction_oct = [mean(abs.(results_dict[fc].policy_october) .< 1e-6) for fc in fixed_costs]
    
    p_inaction = plot(fixed_costs, [inaction_jan inaction_oct], 
                     label=["January" "October"],
                     marker=[:circle :square], lw=2,
                     xlabel="Fixed Cost", ylabel="Inaction Rate",
                     title="Investment Inaction by Fixed Cost",
                     legend=:topleft, size=(800, 500))
    println("  ✓ Inaction rate plot")
    
    # Combine all plots
    p_combined = plot(p_expectations, p_quarterly, p_inaction, 
                     layout=(3,1), size=(1000, 1400))
    
    if save_plots
        println("\n3. Saving plots...")
        savefig(p_expectations, "investment_expectations.png")
        savefig(p_quarterly, "quarterly_investment.png")
        savefig(p_inaction, "inaction_rates.png")
        savefig(p_combined, "combined_report.png")
        println("  ✓ Plots saved")
    end
    
    println("\n" * "="^60)
    println("REPORT GENERATION COMPLETE")
    println("="^60)
    
    return (summary=summary, plots=(expectations=p_expectations, 
                                   quarterly=p_quarterly, 
                                   inaction=p_inaction,
                                   combined=p_combined))
end

# ============================================================================
# SIMULATION UTILITIES
# ============================================================================

"""
    simulate_investment_path(model, solution, initial_state, num_years)

Simulate investment decisions over multiple years with stochastic shocks.

# Returns
Dict with paths for capital, investment, demand shocks, and volatility
"""
function simulate_investment_path(
    model::InvestmentModel,
    solution,
    initial_state::Tuple{Int, Int, Int},
    num_years::Int = 5
)
    k_idx, d_idx, v_idx = initial_state
    
    # Initialize paths
    capital_path = Float64[]
    investment_path_jan = Float64[]
    investment_path_apr = Float64[]
    investment_path_aug = Float64[]
    investment_path_oct = Float64[]
    demand_path = Float64[]
    volatility_path = Float64[]
    
    current_K = model.capital_grid[k_idx]
    current_D = model.demand_shock_grid[d_idx]
    current_vol = model.volatility_grid[v_idx]
    
    for year in 1:num_years
        # JANUARY: Observe demand, choose investment
        push!(capital_path, current_K)
        push!(demand_path, current_D)
        push!(volatility_path, current_vol)
        
        k_idx = argmin(abs.(model.capital_grid .- current_K))
        d_idx = argmin(abs.(model.demand_shock_grid .- current_D))
        v_idx = argmin(abs.(model.volatility_grid .- current_vol))
        
        I_jan = solution.policy_january[k_idx, d_idx, v_idx]
        push!(investment_path_jan, I_jan)
        
        # APRIL: Choose investment
        cum_I = I_jan
        cum_idx = argmin(abs.(model.cumulative_investment_grid .- cum_I))
        I_apr = solution.policy_april[k_idx, cum_idx, d_idx, v_idx]
        push!(investment_path_apr, I_apr)
        
        # AUGUST: Choose investment
        cum_I += I_apr
        cum_idx = argmin(abs.(model.cumulative_investment_grid .- cum_I))
        I_aug = solution.policy_august[k_idx, cum_idx, d_idx, v_idx]
        push!(investment_path_aug, I_aug)
        
        # OCTOBER: Choose investment
        cum_I += I_aug
        cum_idx = argmin(abs.(model.cumulative_investment_grid .- cum_I))
        I_oct = solution.policy_october[k_idx, cum_idx, d_idx, v_idx]
        push!(investment_path_oct, I_oct)
        
        # Update capital for next year
        total_I = I_jan + I_apr + I_aug + I_oct
        current_K = capital_transition(current_K, total_I, model)
        
        # Simulate next period's demand and volatility
        # Volatility evolution
        vol_mean = model.rho_sigma * current_vol + (1 - model.rho_sigma) * model.sigma_X_mean
        vol_std = model.sigma_v * sqrt(0.25)
        current_vol = max(model.volatility_grid[1], 
                         min(model.volatility_grid[end], 
                             rand(Normal(vol_mean, vol_std))))
        
        # Demand evolution
        drift = model.mu_X - 0.5 * current_vol^2
        log_D_next = log(current_D) + drift + current_vol * randn()
        current_D = max(model.demand_shock_grid[1],
                       min(model.demand_shock_grid[end], exp(log_D_next)))
    end
    
    return Dict(
        :capital => capital_path,
        :investment_jan => investment_path_jan,
        :investment_apr => investment_path_apr,
        :investment_aug => investment_path_aug,
        :investment_oct => investment_path_oct,
        :demand => demand_path,
        :volatility => volatility_path
    )
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

"""
    main(; setup_parallel, fixed_cost_grid, max_iterations)

Main entry point for running the investment model analysis.
"""
function main(;
    setup_parallel::Bool = true,
    max_workers::Int = 100,
    fixed_cost_grid::Vector{Float64} = [0.0, 0.02, 0.05, 0.1, 0.15],
    max_iterations::Int = 50,
    tolerance::Float64 = 1e-6,
    generate_report::Bool = true,
    save_plots::Bool = false
)
    println("\n" * "="^70)
    println("OPTIMAL INVESTMENT WITH QUARTERLY INFORMATION UPDATES")
    println("="^70)
    
    # Setup parallel workers
    if setup_parallel
        println("\n→ Setting up parallel computation...")
        
    end
    
    # Create model
    println("\n→ Creating model...")
    model = create_model()
    println("  ✓ Model created with:")
    println("    • Capital grid: $(length(model.capital_grid)) points")
    println("    • Demand grid: $(length(model.demand_shock_grid)) points")
    println("    • Volatility grid: $(length(model.volatility_grid)) points")
    
    # Run experiment
    println("\n→ Running fixed cost experiment...")
    results = run_fixed_cost_experiment(
        model,
        fixed_cost_grid=fixed_cost_grid,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=true
    )
    
    # Generate report
    if generate_report
        report = create_comprehensive_report(results, model, save_plots=save_plots)
        display(report.plots.combined)
        return results, report
    end
    
    return results
end

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

println("""
╔════════════════════════════════════════════════════════════════════╗
║                    INVESTMENT MODEL - READY                        ║
╚════════════════════════════════════════════════════════════════════╝

QUICK START:
------------

1. Run full experiment:
   results, report = main(
       fixed_cost_grid = [0.0, 0.02, 0.05, 0.1],
       max_iterations = 50
   )

2. Single fixed cost:
   model = create_model()
   setup_workers(100)
   solution = solve_model(model, fixed_cost=0.05, max_iterations=50)

3. Analyze expectations:
   policies = Dict(
       :january => solution.policy_january,
       :april => solution.policy_april,
       :august => solution.policy_august,
       :october => solution.policy_october
   )
   exp = compute_investment_expectations(model, policies)
   println("Expected annual investment (January): ", exp[:expectations][:january])
   println("Revision (Jan→Apr): ", exp[:revisions][:jan_to_apr])

4. Custom experiment:
   results = run_fixed_cost_experiment(
       model,
       fixed_cost_grid = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
       max_iterations = 100
   )
   report = create_comprehensive_report(results, model, save_plots=true)

KEY FEATURES:
-------------
✓ Optimized parallel computation with precomputed transitions
✓ Expected total annual investment from each quarter's perspective
✓ Investment expectation revisions as information arrives
✓ Comprehensive testing environment for fixed cost sensitivity
✓ Clear visualization of expectations and revisions
✓ Summary statistics and inaction rate analysis

TIMELINE:
---------
JANUARY  : Observe X_t, σ_{t,Q1} → Choose I_jan → Earn profit
APRIL    : Observe σ_{t,Q2}      → Choose I_apr (refine forecast)
AUGUST   : Observe σ_{t,Q3}      → Choose I_aug (better forecast)
OCTOBER  : Observe σ_{t,Q4}      → Choose I_oct (best forecast)
YEAR END : K_{t+1} = (1-δ)K_t + ΣI, observe X_{t+1}, σ_{t+1,Q1}

""")

main()