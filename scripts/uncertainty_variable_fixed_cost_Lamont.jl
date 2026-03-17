# I am going to test simple scenario
using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(project_root)
using UncertaintyInvestment
using Printf
using NPZ
using Distributed
using DataFrames
using Parquet2

using Random

project_root = dirname(@__DIR__)
outdir = joinpath(project_root,
                  "output",
                  "simulations_uncertainty_variable_fixed_cost_Lamont")

mkpath(outdir)

## 1. Uncertainty, variable fixed cost for Lamont type regression
# Set seed for reproducibility
Random.seed!(12345)

sigma_bar_1 = log(0.1)
sigma_1 = exp(sigma_bar_1)

params = ModelParameters(
    # Technology
    alpha = 0.33,        # Capital share
    epsilon = 4.0,       # Demand elasticity
    delta = 0.10,        # Annual depreciation
    beta = 0.96,         # Annual discount factor

    # Stochastic processes
    demand = DemandProcess(
        mu_D = log(500),
        rho_D = 0.5          # Persistent demand (semester frequency)
    ),
    volatility = VolatilityProcess(
        sigma_bar = sigma_bar_1,  # ~10% demand volatility per semester
        rho_sigma = 0.00001,      # Persistent volatility regime
        sigma_eta = 0.00001,      # Moderate stochastic volatility
        rho_epsilon_eta = 0.0
    ),
    
    numerical = NumericalSettings(
        n_K = 50,
        n_D = 50,
        n_sigma = 5,
        K_min_factor = 0.1,
        K_max_factor = 10.0,
        tol_vfi = 1e-4,
        max_iter = 200,
        howard_steps = 0
    )
)

ac_begin = FixedAdjustmentCost(F = 1)
range_F_mid_year = range(0.5, stop=1.5, length=10)

test = true

NPZ.npzwrite(joinpath(outdir, "grid_K.npy"), construct_grids(params).K_grid)

if test
    # Initialize empty DataFrame to collect all panels
    all_panels_df = DataFrame()
    
    n_firms_per_scenario = 10
    T_years = 100
    
    for F_mid_year in range_F_mid_year
        println("Solving for F_mid_year = $F_mid_year")
        
        ac_mid_year = FixedAdjustmentCost(F = F_mid_year)
        
        sol = solve_model(params; ac_begin=ac_begin, ac_mid_year=ac_mid_year, 
                          verbose=true, use_parallel=true, use_multiscale=true)

        # Generate shock panel
        shocks = generate_shock_panel(
            params.demand,
            params.volatility,
            n_firms_per_scenario,  # Number of firms
            200                     # Number of semesters
        )
        
        histories = simulate_firm_panel(
            sol,
            shocks;
            K_init = 1.0,
            T_years = T_years
        )
        
        panel = construct_estimation_panel(histories)
        
        # Add the fixed cost columns to this panel's DataFrame
        panel.df[!, :F_mid_year] .= F_mid_year
        panel.df[!, :F_begin] .= ac_begin.F
        
        # Append to the combined DataFrame
        append!(all_panels_df, panel.df)
        
        # Optionally save individual policy functions
        F_str = @sprintf("%.2f", F_mid_year)
        NPZ.npzwrite(joinpath(outdir, "I_policy_F$(F_str).npy"), sol.I_policy)
        NPZ.npzwrite(joinpath(outdir, "Delta_I_policy_F$(F_str).npy"), sol.Delta_I_policy)
    end
    
    # Create a FirmPanel from the concatenated DataFrame
    n_scenarios = length(range_F_mid_year)
    total_firms = n_firms_per_scenario * n_scenarios
    all_panels = FirmPanel(all_panels_df, total_firms, T_years)
    
    # Save using the proper function
    save_simulation(joinpath(outdir, "panel_all_fixed_costs.parquet"), all_panels)
    
    println("Saved combined panel with $(nrow(all_panels_df)) observations")
    println("Fixed cost values: $(unique(all_panels_df.F_mid_year))")
end