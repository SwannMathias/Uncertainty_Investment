"""
    simulate_economy_smm.jl

Monte Carlo validation of the SMM estimation pipeline.

Workflow:
    1. Set stochastic process parameters (demand, volatility)
    2. Choose "true" adjustment cost structure (fixed + convex)
    3. Solve the model with true parameters
    4. Store the solution for external analysis
    5. Simulate an economy and compute "empirical" moments
    6. Configure SMM: select which parameters to estimate vs. hold fixed
    7. Run PSO/SMM to recover the true parameters

Usage:
    julia -t auto scripts/simulate_economy_smm.jl
"""

using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(project_root)

using UncertaintyInvestment
using Random
using Printf
using DataFrames
using Statistics
using StatsModels: @formula
using LinearAlgebra: I as eye

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Stochastic Process & Economic Parameters
# ═══════════════════════════════════════════════════════════════════════════════
# Modify these to set the data-generating process (DGP).

calibration = FixedCalibration(
    # Technology
    alpha   = 0.33,      # Capital share in production
    epsilon = 4.0,       # Demand elasticity
    delta   = 0.10,      # Annual depreciation rate
    beta    = 0.96,      # Annual discount factor

    # Demand process (semester frequency)
    mu_D  = 0.0,         # Long-run mean of log demand
    rho_D = 0.5,         # Demand persistence

    # Volatility process (semester frequency, continuous AR(1))
    sigma_bar       = log(0.1),   # Long-run mean of log volatility (~10%)
    rho_sigma       = 0.1,        # Volatility persistence
    sigma_eta       = 0.1,        # Volatility of volatility
    rho_epsilon_eta = 0.0,        # Correlation between demand and vol shocks

    # Numerical settings for VFI
    n_K          = 50,
    n_D          = 15,
    n_sigma      = 7,
    K_min_factor = 0.1,
    K_max_factor = 3.0,
    tol_vfi      = 1e-6,
    max_iter     = 1000,
    howard_steps = 50,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: True Adjustment Cost Structure
# ═══════════════════════════════════════════════════════════════════════════════
# These are the "true" parameter values that SMM will attempt to recover.
# The cost structure is composite: FixedAdjustmentCost + ConvexAdjustmentCost
# at each stage (beginning-of-year and mid-year).

true_cost_params = Dict{Symbol,Float64}(
    :F_begin   => 0.5,    # Fixed cost at beginning of year
    :F_mid     => 0.3,    # Fixed cost at mid-year
    :phi_begin => 2.0,    # Convex cost parameter at beginning of year
    :phi_mid   => 1.5,    # Convex cost parameter at mid-year
)

# Build adjustment costs from true parameters using the composite spec
true_theta = [true_cost_params[p] for p in [:F_begin, :F_mid, :phi_begin, :phi_mid]]
true_spec = composite_spec()
ac_begin_true, ac_mid_true = build_adjustment_costs(true_theta, true_spec)

println("="^70)
println("MONTE CARLO SMM VALIDATION")
println("="^70)
println("\nTrue adjustment cost parameters:")
for (k, v) in sort(collect(true_cost_params), by=first)
    @printf("  %-12s = %.4f\n", k, v)
end
println()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Solve the Model
# ═══════════════════════════════════════════════════════════════════════════════

params = build_model_parameters(calibration)

println("Solving model with true parameters...")
sol = solve_model(params;
    ac_begin    = ac_begin_true,
    ac_mid_year = ac_mid_true,
    verbose     = true,
    use_parallel = true,
)

if sol.convergence.converged
    @printf("VFI converged in %d iterations (distance = %.2e)\n\n",
            sol.convergence.iterations, sol.convergence.final_distance)
else
    @warn "VFI did NOT converge — results may be unreliable"
end

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Store Solution for External Analysis
# ═══════════════════════════════════════════════════════════════════════════════

output_dir = joinpath(project_root, "output", "monte_carlo")
mkpath(output_dir)

# Save full solution (JLD2) — can be reloaded with load_solution()
save_solution(joinpath(output_dir, "true_solution.jld2"), sol)

# Export policy functions and value function to Parquet for Python/R analysis
export_to_parquet(sol, joinpath(output_dir, "true_model"))

println()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Simulate Economy & Compute "Empirical" Moments
# ═══════════════════════════════════════════════════════════════════════════════

# Simulation settings (shared with SMM estimation for consistency)
sim_n_firms       = 1000
sim_T_years       = 50
sim_burn_in_years = 30
sim_shock_seed    = 42
sim_transform     = ASINH_TRANSFORM
sim_zero_threshold = 1e-4

# Generate shocks
T_semesters = 2 * (sim_T_years + sim_burn_in_years)
shocks = generate_shock_panel(
    params.demand, params.volatility,
    sim_n_firms, T_semesters;
    seed = sim_shock_seed, use_parallel = true,
)

# Simulate firms
println("Simulating $(sim_n_firms) firms for $(sim_T_years + sim_burn_in_years) years...")
histories = simulate_firm_panel(sol, shocks;
    K_init       = nothing,
    T_years      = sim_T_years + sim_burn_in_years,
    use_parallel = true,
    verbose      = false,
)

# Build estimation panel and discard burn-in
panel = construct_estimation_panel(histories)
df_post_burnin = panel.df[panel.df.year .> sim_burn_in_years, :]

# Save simulated panel
save_simulation(joinpath(output_dir, "simulated_panel.parquet"), panel)

# Compute "empirical" moments using a temporary SMMConfig with the composite spec
# (all 4 moments: share_zero_begin, share_zero_mid, coef_begin, coef_mid)
moment_config = SMMConfig(
    calibration        = calibration,
    estimation_spec    = composite_spec(),
    m_data             = zeros(4),   # placeholder — not used for moment computation
    n_firms            = sim_n_firms,
    T_years            = sim_T_years,
    burn_in_years      = sim_burn_in_years,
    shock_seed         = sim_shock_seed,
    revision_transform = sim_transform,
    zero_threshold     = sim_zero_threshold,
)

m_data = compute_simulated_moments(df_post_burnin, moment_config)
moment_labels = moment_names(composite_spec())

println("\n" * "="^70)
println("COMPUTED \"EMPIRICAL\" MOMENTS (from true model)")
println("="^70)
for (name, val) in zip(moment_labels, m_data)
    @printf("  %-24s = %+.6f\n", name, val)
end
println()

if any(isnan.(m_data))
    error("Some moments are NaN — check simulation output. " *
          "The SMM estimation cannot proceed with NaN targets.")
end

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SMM Configuration — Select Parameters to Estimate
# ═══════════════════════════════════════════════════════════════════════════════
# Choose estimation_mode:
#   :composite   — estimate all 4 parameters (F_begin, F_mid, phi_begin, phi_mid)
#   :convex_only — fix F at true values, estimate phi_begin and phi_mid
#   :fixed_only  — fix phi at true values, estimate F_begin and F_mid
#   :custom      — manually specify estimated_params, fixed_params, and moments below

estimation_mode = :composite

if estimation_mode == :composite
    # Estimate all 4 parameters, match all 4 moments
    estimated_params = Dict{Symbol,Tuple{Float64,Float64}}(
        :F_begin   => (0.0, 10.0),
        :F_mid     => (0.0, 10.0),
        :phi_begin => (0.0, 20.0),
        :phi_mid   => (0.0, 20.0),
    )
    fixed_params = Dict{Symbol,Float64}()
    moments = AbstractMoment[
        ShareZeroMoment(:begin, "share_zero_begin"),
        ShareZeroMoment(:mid, "share_zero_mid"),
        RegressionCoefficientMoment(:begin,
            @formula(revision_begin ~ log_sigma + log_K + log_D),
            :log_sigma, "coef_begin_sigma"),
        RegressionCoefficientMoment(:mid,
            @formula(revision_mid ~ log_sigma_half + log_K + log_D),
            :log_sigma_half, "coef_mid_sigma"),
    ]
    m_target = m_data  # all 4 moments

elseif estimation_mode == :convex_only
    # Fix F at true values, estimate phi only
    estimated_params = Dict{Symbol,Tuple{Float64,Float64}}(
        :phi_begin => (0.0, 20.0),
        :phi_mid   => (0.0, 20.0),
    )
    fixed_params = Dict{Symbol,Float64}(
        :F_begin => true_cost_params[:F_begin],
        :F_mid   => true_cost_params[:F_mid],
    )
    moments = AbstractMoment[
        RegressionCoefficientMoment(:begin,
            @formula(revision_begin ~ log_sigma + log_K + log_D),
            :log_sigma, "coef_begin_sigma"),
        RegressionCoefficientMoment(:mid,
            @formula(revision_mid ~ log_sigma_half + log_K + log_D),
            :log_sigma_half, "coef_mid_sigma"),
    ]
    # Select matching moments from m_data (indices 3 and 4 in composite spec)
    m_target = m_data[3:4]

elseif estimation_mode == :fixed_only
    # Fix phi at true values, estimate F only
    estimated_params = Dict{Symbol,Tuple{Float64,Float64}}(
        :F_begin => (0.0, 10.0),
        :F_mid   => (0.0, 10.0),
    )
    fixed_params = Dict{Symbol,Float64}(
        :phi_begin => true_cost_params[:phi_begin],
        :phi_mid   => true_cost_params[:phi_mid],
    )
    moments = AbstractMoment[
        ShareZeroMoment(:begin, "share_zero_begin"),
        ShareZeroMoment(:mid, "share_zero_mid"),
    ]
    # Select matching moments from m_data (indices 1 and 2 in composite spec)
    m_target = m_data[1:2]

elseif estimation_mode == :custom
    # ─── Customize here ───
    estimated_params = Dict{Symbol,Tuple{Float64,Float64}}(
        :phi_begin => (0.0, 20.0),
    )
    fixed_params = Dict{Symbol,Float64}(
        :F_begin   => true_cost_params[:F_begin],
        :F_mid     => true_cost_params[:F_mid],
        :phi_mid   => true_cost_params[:phi_mid],
    )
    moments = AbstractMoment[
        RegressionCoefficientMoment(:begin,
            @formula(revision_begin ~ log_sigma + log_K + log_D),
            :log_sigma, "coef_begin_sigma"),
    ]
    m_target = m_data[3:3]
    # ─── End customization ───

else
    error("Unknown estimation_mode: $estimation_mode. " *
          "Choose :composite, :convex_only, :fixed_only, or :custom.")
end

# Build SMMConfig
smm_config = SMMConfig(
    calibration        = calibration,
    fixed_params       = fixed_params,
    estimated_params   = estimated_params,
    moments            = moments,
    m_data             = m_target,
    W                  = Matrix{Float64}(eye, length(m_target), length(m_target)),
    n_firms            = sim_n_firms,
    T_years            = sim_T_years,
    burn_in_years      = sim_burn_in_years,
    shock_seed         = sim_shock_seed,
    revision_transform = sim_transform,
    zero_threshold     = sim_zero_threshold,
)

println("SMM estimation mode: $estimation_mode")
println("Parameters to estimate: $(collect(keys(estimated_params)))")
println("Fixed parameters: $(collect(fixed_params))")
println("Target moments: $m_target")
println()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Run SMM via PSO
# ═══════════════════════════════════════════════════════════════════════════════

pso_config = PSOConfig(
    n_particles         = 20,
    threads_per_particle = 1,
    max_iterations      = 100,
    w_inertia           = 0.7,
    c_cognitive         = 1.5,
    c_social            = 1.5,
    reassign_every      = 20,
    reassign_fraction   = 0.1,
    tol_objective       = 1e-8,
    patience            = 20,
    verbose             = true,
    save_history        = true,
    checkpoint_every    = 10,
    output_dir          = joinpath(output_dir, "estimation"),
)

# Reuse the grids from the true model solve (same calibration)
grids = sol.grids

# Run estimation
result = run_smm_estimation(smm_config, pso_config; grids = grids)

# ═══════════════════════════════════════════════════════════════════════════════
# Results Summary: Estimated vs. True Parameters
# ═══════════════════════════════════════════════════════════════════════════════

param_names_est = get_param_names(smm_config)

println("\n" * "="^70)
println("ESTIMATION RESULTS")
println("="^70)
println("\nParameter Recovery:")
@printf("  %-16s  %10s  %10s  %10s\n", "Parameter", "True", "Estimated", "Error")
@printf("  %-16s  %10s  %10s  %10s\n", "-"^16, "-"^10, "-"^10, "-"^10)
for (i, pname) in enumerate(param_names_est)
    true_val = true_cost_params[pname]
    est_val  = result.theta_best[i]
    err      = est_val - true_val
    @printf("  %-16s  %10.4f  %10.4f  %+10.4f\n", pname, true_val, est_val, err)
end

println("\nMoment Fit:")
mnames = moment_names(smm_config.estimation_spec)
@printf("  %-24s  %10s  %10s  %10s\n", "Moment", "Target", "Simulated", "Diff")
@printf("  %-24s  %10s  %10s  %10s\n", "-"^24, "-"^10, "-"^10, "-"^10)
for i in eachindex(mnames)
    @printf("  %-24s  %+10.6f  %+10.6f  %+10.6f\n",
            mnames[i], result.moments_data[i],
            result.moments_best[i],
            result.moments_best[i] - result.moments_data[i])
end

@printf("\nSMM Objective: %.2e\n", result.objective_best)
@printf("PSO Iterations: %d  |  Evaluations: %d  |  Converged: %s\n",
        result.n_iterations, result.n_evaluations,
        result.converged ? "yes" : "no")
@printf("Elapsed time: %.1f seconds\n", result.elapsed_time)
println("="^70)
