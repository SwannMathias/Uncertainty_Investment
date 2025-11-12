
# ============================================================================
# VISUALIZATION
# ============================================================================

"""
    plot_investment_expectations(results_dict, model)

Create comprehensive visualization of investment expectations across fixed costs.

# Arguments
- `results_dict`: Dict mapping fixed_cost → solution
- `model`: InvestmentModel instance
"""
function plot_investment_expectations(results_dict::Dict, model::InvestmentModel)
    fixed_costs = sort(collect(keys(results_dict)))
    
    # Compute expectations for each fixed cost
    all_expectations = Dict()
    for fc in fixed_costs
        sol = results_dict[fc]
        policies = Dict(
            :january => sol.policy_january,
            :april => sol.policy_april,
            :august => sol.policy_august,
            :october => sol.policy_october
        )
        all_expectations[fc] = compute_investment_expectations(model, policies)
    end
    
    # Create comprehensive plot
    p1 = plot(title="Expected Total Annual Investment", xlabel="Quarter", 
              ylabel="Expected Annual Investment", legend=:topright, size=(800, 500))
    
    quarters = ["Jan", "Apr", "Aug", "Oct"]
    quarter_syms = [:january, :april, :august, :october]
    
    for fc in fixed_costs
        exp = all_expectations[fc][:expectations]
        vals = [exp[q] for q in quarter_syms]
        plot!(p1, quarters, vals, marker=:circle, lw=2, label="Fixed cost = $fc")
    end
    
    # Plot revisions
    p2 = plot(title="Expectation Revisions", xlabel="Revision Period", 
              ylabel="Change in Expected Annual Investment", legend=:topright, size=(800, 500))
    
    revision_periods = ["Jan→Apr", "Apr→Aug", "Aug→Oct"]
    revision_syms = [:jan_to_apr, :apr_to_aug, :aug_to_oct]
    
    for fc in fixed_costs
        rev = all_expectations[fc][:revisions]
        vals = [rev[r] for r in revision_syms]
        plot!(p2, revision_periods, vals, marker=:square, lw=2, label="Fixed cost = $fc")
    end
    hline!(p2, [0], linestyle=:dash, color=:black, label="")
    
    return plot(p1, p2, layout=(2,1), size=(800, 800))
end

"""
    plot_quarterly_investment_plans(results_dict, model)

Visualize quarterly investment decisions (not cumulative) across fixed costs.
"""
function plot_quarterly_investment_plans(results_dict::Dict, model::InvestmentModel)
    fixed_costs = sort(collect(keys(results_dict)))
    
    p = plot(title="Average Quarterly Investment by Period", 
             xlabel="Quarter", ylabel="Investment", legend=:topright, size=(800, 500))
    
    quarters = ["Jan", "Apr", "Aug", "Oct"]
    
    for fc in fixed_costs
        sol = results_dict[fc]
        
        # Compute average investment in each quarter
        I_jan = mean(sol.policy_january)
        I_apr = mean(sol.policy_april)
        I_aug = mean(sol.policy_august)
        I_oct = mean(sol.policy_october)
        
        vals = [I_jan, I_apr, I_aug, I_oct]
        plot!(p, quarters, vals, marker=:circle, lw=2, label="Fixed cost = $fc")
    end
    
    return p
end