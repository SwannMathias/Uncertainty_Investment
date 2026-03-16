# I am going to test simple scenario
using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(project_root)
using UncertaintyInvestment
using Printf
using NPZ
using Distributed
using DataFrames
using CSV

using Random

project_root = dirname(@__DIR__)
outdir = joinpath(project_root,
                  "output",
                  "simulations_revision_regression_convex_adj")

mkpath(outdir)

## 1. Uncertainty, variable fixed cost for Lamont type regression
# Set seed for reproducibility
Random.seed!(12345)

sigma_bar_1 = log(0.1)
sigma_1 = exp(sigma_bar_1)


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
        rho_sigma = 0.1,      # Persistent volatility regime
        sigma_eta = 0.1,      # Moderate stochastic volatility
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

ac_mid_year = FixedAdjustmentCost(F = 100)
range_phi_begin = range(0, stop=5, length=2)

test = true

NPZ.npzwrite(joinpath(outdir,"grid_K.npy"),construct_grids(params).K_grid)
NPZ.npzwrite(joinpath(outdir, "grid_D.npy"), construct_grids(params).sv.D_grid)
NPZ.npzwrite(joinpath(outdir,"profits.npy"),construct_grids(params).precomputed_profits.profits)

if test
    # Initialize empty DataFrame to collect all panels
    all_panels_df = DataFrame()
    
    n_firms_per_scenario = 100
    T_years = 1000
    
    for phi in range_phi_begin
        println("Solving for phi = $phi")
        
        ac_begin = ConvexAdjustmentCost(phi = phi)
        
        sol = solve_model(params; ac_begin=ac_begin, ac_mid_year=ac_mid_year, 
                          verbose=true, use_parallel=true, use_multiscale=true)

        # Generate shock panel
        shocks = generate_shock_panel(
            params.demand,
            params.volatility,
            n_firms_per_scenario,  # Number of firms
            2000                     # Number of semesters
        )
        
        histories = simulate_firm_panel(
            sol,
            shocks;
            K_init = nothing,
            T_years = T_years
        )
        
        panel = construct_estimation_panel(histories)
        
        # Add the fixed cost columns to this panel's DataFrame
        panel.df[!, :phi] .= phi
        
        # Append to the combined DataFrame
        append!(all_panels_df, panel.df)
        
        # Optionally save individual policy functions
        phi = @sprintf("%.2f", phi)
        NPZ.npzwrite(joinpath(outdir, "V_$(phi).npy"), sol.V)
        NPZ.npzwrite(joinpath(outdir, "V_stage1$(phi).npy"), sol.V_stage1)
        NPZ.npzwrite(joinpath(outdir, "I_policy_F$(phi).npy"), sol.I_policy)
        NPZ.npzwrite(joinpath(outdir, "Delta_I_policy_F$(phi).npy"), sol.Delta_I_policy)
    end
    
    # Create a FirmPanel from the concatenated DataFrame
    n_scenarios = length(range_phi_begin)
    total_firms = n_firms_per_scenario * n_scenarios
    all_panels = FirmPanel(all_panels_df, total_firms, T_years)
    
    # Save using the proper function
    save_simulation(joinpath(outdir, "panel_all.csv"), all_panels)
    
    println("Saved combined panel with $(nrow(all_panels_df)) observations")
    println("Phi values: $(unique(all_panels_df.phi))")
end