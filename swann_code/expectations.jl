
# ============================================================================
# EXPECTATION COMPUTATION
# ============================================================================

"""
    compute_expected_annual_investment(model, policies, quarter; weights)

Compute expected TOTAL annual investment E[I_jan + I_apr + I_aug + I_oct]
as perceived at a given quarter.

This captures the firm's expectation of total year investment based on 
information available at that quarter.

# Arguments
- `quarter`: :january, :april, :august, or :october
- `weights`: optional state probabilities (default uniform)

# Returns
Expected total annual investment
"""
function compute_expected_annual_investment(
    model::InvestmentModel,
    policies::Dict,
    quarter::Symbol;
    weights=nothing
)
# Swann: This function is not correct. It is important to take the expected path according to the likelihood of cumulative investment.
    num_K = length(model.capital_grid)
    num_D = length(model.demand_shock_grid)
    num_vol = length(model.volatility_grid)
    
    if weights === nothing
        weights = ones(num_K, num_D, num_vol)
        weights ./= sum(weights)
    end
    
    # Extract relevant policy
    if quarter == :january
        # January: expect I_jan (realized) + E[I_apr + I_aug + I_oct | info at Jan]
        policy_jan = policies[:january]
        expected_I_jan = sum(policy_jan .* weights)
        
        # For future quarters, take expectation over cumulative investment dimension
        policy_apr = mean(policies[:april], dims=2)[:, 1, :, :]
        policy_aug = mean(policies[:august], dims=2)[:, 1, :, :]
        policy_oct = mean(policies[:october], dims=2)[:, 1, :, :]
        
        expected_I_apr = sum(policy_apr .* weights)
        expected_I_aug = sum(policy_aug .* weights)
        expected_I_oct = sum(policy_oct .* weights)
        
        return expected_I_jan + expected_I_apr + expected_I_aug + expected_I_oct
        
    elseif quarter == :april
        # April: I_jan already realized, expect I_apr (current) + E[I_aug + I_oct]
        policy_apr = mean(policies[:april], dims=2)[:, 1, :, :]
        policy_aug = mean(policies[:august], dims=2)[:, 1, :, :]
        policy_oct = mean(policies[:october], dims=2)[:, 1, :, :]
        
        expected_I_apr = sum(policy_apr .* weights)
        expected_I_aug = sum(policy_aug .* weights)
        expected_I_oct = sum(policy_oct .* weights)
        
        # Note: We should ideally condition on realized I_jan, but for simplicity
        # we compute unconditional expectation
        return expected_I_apr + expected_I_aug + expected_I_oct
        
    elseif quarter == :august
        policy_aug = mean(policies[:august], dims=2)[:, 1, :, :]
        policy_oct = mean(policies[:october], dims=2)[:, 1, :, :]
        
        expected_I_aug = sum(policy_aug .* weights)
        expected_I_oct = sum(policy_oct .* weights)
        
        return expected_I_aug + expected_I_oct
        
    elseif quarter == :october
        policy_oct = mean(policies[:october], dims=2)[:, 1, :, :]
        return sum(policy_oct .* weights)
    end
end

"""
    compute_investment_expectations(model, policies; weights)

Compute expected total annual investment at each quarter and revisions.

# Returns
Dict with:
- `:expectations`: Expected total annual investment from each quarter's perspective
- `:revisions`: Quarter-to-quarter changes in expectations
"""
function compute_investment_expectations(model::InvestmentModel, policies::Dict; weights=nothing)
    quarters = [:january, :april, :august, :october]
    
    expectations = Dict{Symbol, Float64}()
    for q in quarters
        expectations[q] = compute_expected_annual_investment(model, policies, q, weights=weights)
    end
    
    revisions = Dict(
        :jan_to_apr => expectations[:april] - expectations[:january],
        :apr_to_aug => expectations[:august] - expectations[:april],
        :aug_to_oct => expectations[:october] - expectations[:august]
    )
    
    return Dict(:expectations => expectations, :revisions => revisions)
end
