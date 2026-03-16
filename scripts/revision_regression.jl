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
                  "simulations_revision_regression")

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

ac_begin = FixedAdjustmentCost(F = 0)
ac_mid_year = FixedAdjustmentCost(F = 100)


NPZ.npzwrite(joinpath(outdir,"grid_K.npy"),construct_grids(params).K_grid)
NPZ.npzwrite(joinpath(outdir, "grid_D.npy"), construct_grids(params).sv.D_grid)
#NPZ.npzwrite(joinpath(outdir,"grid_sigma.npy"),construct_grids(params).sigma_grid)
sol_scenario1 = solve_model(params; ac_begin=ac_begin,ac_mid_year = ac_mid_year, verbose=true,use_parallel=true, use_multiscale=true)

# Generate shock panel
shocks = generate_shock_panel(
    params.demand ,
    params.volatility,
    1000,  # Number of firms
    2000   # Number of semesters
)
histories = simulate_firm_panel(
    sol_scenario1,
    shocks;
    K_init = nothing,
    T_years = 1000
)
panel = construct_estimation_panel(histories)
save_simulation(joinpath(outdir,"panel_data_s11.csv"), panel)
NPZ.npzwrite(joinpath(outdir,"I_policy_s11.npy"),sol_scenario1.I_policy)
NPZ.npzwrite(joinpath(outdir,"Delta_I_policy_s11.npy"),sol_scenario1.Delta_I_policy)
