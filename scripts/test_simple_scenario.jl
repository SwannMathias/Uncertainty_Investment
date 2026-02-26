

# I am going to test simple scenario
using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(project_root)
using UncertaintyInvestment
using Printf

using Distributed
addprocs(70)
using Random

## 1. No uncertainty, Fixed cost
# Increasing the fixed cost should decrease the frequency of investment. 
# Set seed for reproducibility
Random.seed!(12345)

params = ModelParameters(
    # Technology
    alpha = 0.33,        # Capital share
    epsilon = 4.0,       # Demand elasticity
    delta = 0.10,        # Annual depreciation
    beta = 0.96,         # Annual discount factor

    
    # DETERMINISTIC: Zero volatility
    demand = DemandProcess(
        mu_D = log(500),
        rho_D = 0.0    # Near unit root, but stable
    ),
    
    volatility = VolatilityProcess(
        sigma_bar = log(0.00001),      # Essentially zero volatility
        rho_sigma = 0.0,
        sigma_eta = 0.000000001,           # No volatility of volatility
        rho_epsilon_eta = 0.0
    ),
    
    numerical = NumericalSettings(
        n_K = 100,
        n_D = 5,
        n_sigma = 3,    # Can reduce to 1 since σ ≈ 0
        K_min_factor = 0.1,
        K_max_factor = 10.0,
        tol_vfi = 1e-4,
        max_iter = 200,
        howard_steps = 5
    )
)

ac_begin = FixedAdjustmentCost(F = 0.1)
ac_mid_year= FixedAdjustmentCost(F = 0.1)
sol_scenario1 = solve_model(params; ac_begin=ac_begin,ac_mid_year = ac_mid_year, verbose=true,use_parallel=true, use_multiscale=true)



ac_begin = FixedAdjustmentCost(F = 1)
ac_mid_year= FixedAdjustmentCost(F = 1)
sol_scenario2 = solve_model(params; ac_begin=ac_begin,ac_mid_year = ac_mid_year, verbose=true,use_parallel=true, use_multiscale=true)

# Generate shock panel
shocks = generate_shock_panel(
    params.demand ,
    params.volatility,
    1000,  # Number of firms
    200   # Number of semesters
)
histories = simulate_firm_panel(
    sol_scenario1,
    shocks;
    K_init = 1.,
    T_years = 100
)
panel = construct_estimation_panel(histories)
save_simulation("output/simulations/panel_data_s11.csv", panel)

histories = simulate_firm_panel(
    sol_scenario2,
    shocks;
    K_init = 1.,
    T_years = 100
)
panel = construct_estimation_panel(histories)
save_simulation("output/simulations/panel_data_s12.csv", panel)
## 2. No uncertainty, Different fixed cost
# Increasing the fixed cost should decrease the frequency of investment. 
# Set seed for reproducibility
Random.seed!(12345)

ac_begin = FixedAdjustmentCost(F = 0.1)
ac_mid_year= FixedAdjustmentCost(F = 0.1)
sol_scenario1 = solve_model(params; ac_begin=ac_begin,ac_mid_year = ac_mid_year, verbose=true,use_parallel=true, use_multiscale=true)

ac_begin = FixedAdjustmentCost(F = 0.1)
ac_mid_year= FixedAdjustmentCost(F = 1)
sol_scenario2 = solve_model(params; ac_begin=ac_begin,ac_mid_year = ac_mid_year, verbose=true,use_parallel=true, use_multiscale=true)

# Generate shock panel
shocks = generate_shock_panel(
    params.demand ,
    params.volatility,
    1000,  # Number of firms
    200   # Number of semesters
)
histories = simulate_firm_panel(
    sol_scenario1,
    shocks;
    K_init = 1.,
    T_years = 100
)
panel = construct_estimation_panel(histories)
save_simulation("output/simulations/panel_data_s21.csv", panel)

histories = simulate_firm_panel(
    sol_scenario2,
    shocks;
    K_init = 1.,
    T_years = 100
)
panel = construct_estimation_panel(histories)
save_simulation("output/simulations/panel_data_s22.csv", panel)

