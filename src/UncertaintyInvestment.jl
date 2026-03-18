"""
    UncertaintyInvestment

Julia package for solving and estimating dynamic investment models with stochastic volatility.

# Main Features
- Flexible adjustment cost specifications
- Stochastic volatility with intra-period information arrival
- Value function iteration solver
- Firm simulation and panel construction
- GMM estimation via indirect inference

# Quick Start
```julia
using UncertaintyInvestment

# Define parameters
params = ModelParameters(alpha=0.33, epsilon=4.0, delta=0.10, beta=0.96)

# Solve model
sol = solve_model(params; ac=ConvexAdjustmentCost(phi=2.0))

# Simulate firms
shocks = generate_shock_panel(params.demand, params.volatility, 1000, 120)
histories = simulate_firm_panel(sol, shocks; K_init=1.0, T_years=50)
panel = construct_estimation_panel(histories)
```
"""
module UncertaintyInvestment

# External dependencies
using Parameters
using LinearAlgebra
using Statistics
using Random
using Distributions
using Interpolations
using Optim
using NLsolve
using Parquet2
using DataFrames
using JLD2
using ProgressMeter
using Printf
using FixedEffectModels
using StatsModels

# Threading support (Julia Base)
using Base.Threads: @threads, nthreads, threadid

# ============================================================================
# Model primitives
# ============================================================================

include("model/parameters.jl")
export DemandProcess, AbstractVolatilitySpec, VolatilityProcess, TwoStateVolatility
export NumericalSettings, ModelParameters
export DerivedParameters, get_derived_parameters
export validate_parameters, print_parameters

include("model/primitives.jl")
export profit, marginal_product_capital, profit_derivative_K, profit_derivative_D
export profit_second_derivative_K, annual_profit, optimal_capital_static
export profit_elasticity_K, profit_elasticity_D, check_profit_properties

include("model/adjustment_costs.jl")
export AbstractAdjustmentCost
export NoAdjustmentCost, ConvexAdjustmentCost
export FixedAdjustmentCost
export CompositeAdjustmentCost
export ConvexCrossStageAdjustmentCost
export compute_cost, marginal_cost_I, marginal_cost_Delta_I
export has_fixed_cost, is_differentiable, describe_adjustment_cost
export total_adjustment_cost

include("model/stochastic_process.jl")
export rouwenhorst, tauchen, SVDiscretization
export discretize_sv_process, stationary_distribution, verify_discretization
export get_D_levels, get_sigma_levels

include("model/grids.jl")
export StateGrids, PrecomputedProfits, construct_grids, precompute_profits
export get_K, get_D, get_sigma, get_log_D, get_log_sigma
export get_joint_state_index, get_D_sigma_indices
export get_profit, get_log_profit, get_profit_at_K, get_profit_vector  # Precomputed profit accessors
export find_K_bracket, interpolate_value, interpolate_policy
export compute_expectation, compute_conditional_expectation
export print_grid_info

# ============================================================================
# Solution methods
# ============================================================================

include("solution/interpolation.jl")
export linear_interp_1d, create_interpolant_1d, create_interpolant_3d
export find_bracket, bilinear_interp, interpolate_on_K
export derivative_fd, derivative_cd, gradient_fd

include("solution/bellman.jl")
export solve_midyear_problem, compute_midyear_continuation
export solve_beginning_year_problem, bellman_operator!
export howard_improvement_step!
export bellman_operator_parallel!, howard_improvement_step_parallel!

include("solution/vfi.jl")
export SolvedModel, value_function_iteration, solve_model
export solve_model_multiscale, interpolate_value_function  # Multi-scale grid refinement
export solution_diagnostics, print_solution_diagnostics
export evaluate_value, evaluate_policy, compute_stationary_distribution
export get_nthreads, get_threadid  # Threading utilities

# ============================================================================
# Simulation
# ============================================================================

include("simulation/simulate_shocks.jl")
export ShockPanel, simulate_ar1_path, simulate_sv_path
export generate_shock_panel, generate_shock_panel_parallel
export get_firm_shocks, get_firm_shocks_level
export shock_statistics, print_shock_statistics

include("simulation/simulate_firms.jl")
export FirmHistory, simulate_firm, simulate_firm_panel
export simulate_firm_panel_parallel

include("simulation/panel.jl")
export FirmPanel, construct_estimation_panel
export panel_summary_statistics, print_panel_summary

include("model/irf_shocks.jl")
export generate_irf_panels, year_to_semester

# ============================================================================
# Estimation
# ============================================================================

include("estimation/types.jl")
export EstimationResult

include("estimation/estimation_spec.jl")
export EstimationSpec, AbstractMoment, ShareZeroMoment, RegressionCoefficientMoment
export CostParameterMapping, build_adjustment_costs
export composite_spec, convex_only_spec, fixed_only_spec
export build_estimation_spec, COMPOSITE_PARAM_DEFS, COMPOSITE_PARAM_ORDER
export n_params, n_moments, moment_names

include("estimation/smm_config.jl")
export SMMConfig, FixedCalibration, RevisionTransform
export LOG_TRANSFORM, LEVEL_OVER_K_TRANSFORM, ASINH_TRANSFORM
export build_model_parameters
export get_lower_bounds, get_upper_bounds, get_param_names

include("estimation/moments.jl")
export compute_simulated_moments, compute_revision_panel, prepare_regression_df
export apply_transform, ols_coefficients

include("estimation/smm_objective.jl")
export smm_objective, ParticleState

include("estimation/pso.jl")
export PSOConfig, PSOResult, pso_optimize
export latin_hypercube_sample
export save_pso_checkpoint, load_pso_checkpoint, resume_pso

include("estimation/run_estimation.jl")
export run_smm_estimation

# ============================================================================
# Utilities
# ============================================================================

include("utils/numerical.jl")
export golden_section_search, maximize_univariate, minimize_univariate
export log_sum_exp, softmax, check_convergence, check_convergence_policy
export relative_difference, safe_log, safe_exp, clamp_to_range
export linspace, logspace, grid_K_optimal, bisection
export is_monotonic_increasing, is_valid_probability_matrix
export condition_number, format_time, format_number

include("utils/io.jl")
export save_solution, load_solution
export export_policy_to_parquet, export_value_function_to_parquet
export save_simulation, load_simulation
export save_estimation_results, load_estimation_results
export export_to_parquet, create_output_directories

# ============================================================================
# Package info
# ============================================================================

"""
    version()

Return package version.
"""
version() = "0.1.0"

"""
    cite()

Return citation information.
"""
function cite()
    println("""
    UncertaintyInvestment.jl

    A Julia package for solving and estimating dynamic investment models
    with stochastic volatility and adjustment costs.

    If you use this package in your research, please cite:
    [Your citation here]
    """)
end

end # module UncertaintyInvestment
