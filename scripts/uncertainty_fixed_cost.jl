

# I am going to test simple scenario
using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(project_root)
using UncertaintyInvestment
using Printf
using NPZ
using Distributed

using Random

project_root = dirname(@__DIR__)
outdir = joinpath(project_root,
                  "output",
                  "simulations_uncertainty_fixed_cost")

mkpath(outdir)

## 1. No uncertainty, Fixed cost
# Increasing the fixed cost should decrease the frequency of investment. 
# Set seed for reproducibility
Random.seed!(12345)


"""
    delta_mu(sigma_1, sigma_2)

Return the change in the log-mean (Δμ) required to keep E[D] constant
when volatility changes from sigma_1 to sigma_2 in a log-normal process.

Both sigmas must be the *level* standard deviations of log demand.
"""
function delta_mu(sigma_1::Real, sigma_2::Real)
    return -0.5 * (sigma_2^2 - sigma_1^2)
end

sigma_bar_1 = log(0.1)
sigma_bar_2 = 2*log(0.1) 

sigma_1 = exp(sigma_bar_1)
sigma_2 = exp(sigma_bar_2)

mu_adjustment = delta_mu(sigma_1, sigma_2)

ac_begin = FixedAdjustmentCost(F = 0.5)
ac_mid_year= FixedAdjustmentCost(F = 1000)

test = true

if test

    

    params = ModelParameters(
        # Technology
        alpha = 0.33,        # Capital share
        epsilon = 4.0,       # Demand elasticity
        delta = 0.10,        # Annual depreciation
        beta = 0.96,         # Annual discount factor

        
        # DETERMINISTIC: Zero volatility
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
            n_sigma = 5,    # Can reduce to 1 since σ ≈ 0
            K_min_factor = 0.1,
            K_max_factor = 10.0,
            tol_vfi = 1e-4,
            max_iter = 200,
            howard_steps = 0
        )
    )
    NPZ.npzwrite("output/simulations_uncertainty_fixed_cost/grid_K.npy",construct_grids(params).K_grid)
    sol_scenario1 = solve_model(params; ac_begin=ac_begin,ac_mid_year = ac_mid_year, verbose=true,use_parallel=true, use_multiscale=true)

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
    save_simulation("output/simulations_uncertainty_fixed_cost/panel_data_s11.parquet", panel)
    NPZ.npzwrite("output/simulations_uncertainty_fixed_cost/I_policy_s11.npy",sol_scenario1.I_policy)
    NPZ.npzwrite("output/simulations_uncertainty_fixed_cost/Delta_I_policy_s11.npy",sol_scenario1.Delta_I_policy)

    params = ModelParameters(
        # Technology
        alpha = 0.33,        # Capital share
        epsilon = 4.0,       # Demand elasticity
        delta = 0.10,        # Annual depreciation
        beta = 0.96,         # Annual discount factor

        
        # DETERMINISTIC: Zero volatility
        demand = DemandProcess(
                mu_D = log(500) ,
                rho_D = 0.5           # Persistent demand (semester frequency)
            ),
        volatility = VolatilityProcess(
                sigma_bar = sigma_bar_2,  # ~10% demand volatility per semester
                rho_sigma = 0.00001,      # Persistent volatility regime
                sigma_eta = 0.00001,      # Moderate stochastic volatility
                rho_epsilon_eta = 0.0
            ),
        
        numerical = NumericalSettings(
            n_K = 50,
            n_D = 50,
            n_sigma = 5,    # Can reduce to 1 since σ ≈ 0
            K_min_factor = 0.1,
            K_max_factor = 10.0,
            tol_vfi = 1e-4,
            max_iter = 200,
            howard_steps = 0
        )
    )
    sol_scenario1 = solve_model(params; ac_begin=ac_begin,ac_mid_year = ac_mid_year, verbose=true,use_parallel=true, use_multiscale=true)

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
    save_simulation("output/simulations_uncertainty_fixed_cost/panel_data_s12.parquet", panel)
    NPZ.npzwrite("output/simulations_uncertainty_fixed_cost/I_policy_s12.npy",sol_scenario1.I_policy)
    NPZ.npzwrite("output/simulations_uncertainty_fixed_cost/Delta_I_policy_s12.npy",sol_scenario1.Delta_I_policy)

end
