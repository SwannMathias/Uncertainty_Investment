"""
Top-level entry point for SMM estimation.

Provides `run_smm_estimation` which orchestrates the full estimation pipeline:
grid construction, shock generation, PSO optimization, and result packaging.
"""

"""
    run_smm_estimation(config::SMMConfig, pso_config::PSOConfig;
                       grids=nothing, shocks=nothing) -> PSOResult

Run full SMM estimation pipeline.

# Pipeline
1. Build ModelParameters from FixedCalibration
2. Construct grids (or use provided)
3. Generate shocks (or use provided)
4. Run PSO optimization
5. Save final results

# Arguments
- `config`: SMM estimation configuration (calibration, bounds, targets, simulation settings)
- `pso_config`: PSO optimizer configuration (particles, iterations, hyperparameters)
- `grids`: Pre-constructed StateGrids (constructed from calibration if nothing)
- `shocks`: Pre-generated ShockPanel (generated with shock_seed if nothing)

# Returns
- `PSOResult` with best parameters, moments, convergence info, and history

# Example
```julia
# Default 4-parameter composite spec
config = SMMConfig(
    calibration = FixedCalibration(rho_D=0.5, sigma_bar=log(0.1)),
    m_data = [0.35, 0.50, -0.15, 0.10],
    n_firms = 1000, T_years = 50, burn_in_years = 30
)

# Convex-only 2-parameter spec
config_convex = SMMConfig(
    estimation_spec = convex_only_spec(),
    m_data = [-0.15, 0.10],
    n_firms = 1000, T_years = 50, burn_in_years = 30
)

pso_config = PSOConfig(n_particles=10, max_iterations=50, verbose=true)
result = run_smm_estimation(config, pso_config)
```
"""
function run_smm_estimation(config::SMMConfig, pso_config::PSOConfig;
                            grids::Union{Nothing, StateGrids}=nothing,
                            shocks::Union{Nothing, ShockPanel}=nothing)
    spec = config.estimation_spec
    np = n_params(spec)
    nm = n_moments(spec)

    if pso_config.verbose
        println("="^70)
        println("SMM-PSO Estimation")
        println("="^70)
        println("\nFixed Calibration:")
        @printf("  alpha=%.2f, epsilon=%.1f, delta=%.2f, beta=%.2f\n",
                config.calibration.alpha, config.calibration.epsilon,
                config.calibration.delta, config.calibration.beta)
        @printf("  rho_D=%.2f, sigma_bar=%.4f, rho_sigma=%.2f, sigma_eta=%.2f\n",
                config.calibration.rho_D, config.calibration.sigma_bar,
                config.calibration.rho_sigma, config.calibration.sigma_eta)
        println("\nEstimation Specification:")
        @printf("  Parameters (%d): %s\n", np, join(string.(spec.param_names), ", "))
        for (i, pname) in enumerate(spec.param_names)
            @printf("    %-16s in [%.1f, %.1f]\n", string(pname),
                    spec.lower_bounds[i], spec.upper_bounds[i])
        end
        mnames = moment_names(spec)
        @printf("  Moments (%d): %s\n", nm, join(mnames, ", "))
        @printf("  Targets: [%s]\n", join([@sprintf("%.4f", m) for m in config.m_data], ", "))
        @printf("  Transform: %s\n", string(config.revision_transform))
        @printf("  Simulation: %d firms x %d years (burn-in: %d)\n",
                config.n_firms, config.T_years, config.burn_in_years)
        println("\nPSO Setup:")
        @printf("  Particles: %d, Max iterations: %d\n",
                pso_config.n_particles, pso_config.max_iterations)
        @printf("  Inertia=%.2f, Cognitive=%.2f, Social=%.2f\n",
                pso_config.w_inertia, pso_config.c_cognitive, pso_config.c_social)
        if pso_config.reassign_every > 0
            @printf("  Reassignment: every %d iters, %.0f%% of particles\n",
                    pso_config.reassign_every, pso_config.reassign_fraction * 100)
        end
        println("="^70)
        println()
    end

    # 1. Build model parameters and grids
    params = build_model_parameters(config.calibration)

    if isnothing(grids)
        if pso_config.verbose
            @info "Constructing state grids..."
        end
        grids = construct_grids(params)
    end

    # 2. Generate shocks (same for all evaluations)
    if isnothing(shocks)
        if pso_config.verbose
            @info "Generating shock panel (seed=$(config.shock_seed))..."
        end
        T_semesters = 2 * (config.T_years + config.burn_in_years)
        shocks = generate_shock_panel(
            params.demand, params.volatility,
            config.n_firms, T_semesters;
            seed=config.shock_seed, use_parallel=true
        )
    end

    # 3. Run PSO
    result = pso_optimize(config, pso_config; grids=grids, shocks=shocks)

    # 4. Save final results
    _save_final_results(result, config, pso_config)

    return result
end

"""
    _save_final_results(result, config, pso_config)

Save estimation results to the output directory.
"""
function _save_final_results(result::PSOResult, config::SMMConfig, pso_config::PSOConfig)
    spec = config.estimation_spec
    np = n_params(spec)
    nm = n_moments(spec)
    mnames = moment_names(spec)

    mkpath(pso_config.output_dir)

    # Save main results to JLD2
    results_file = joinpath(pso_config.output_dir, "smm_results.jld2")
    try
        jldsave(results_file;
                theta_best=result.theta_best,
                objective_best=result.objective_best,
                moments_best=result.moments_best,
                moments_data=result.moments_data,
                n_iterations=result.n_iterations,
                n_evaluations=result.n_evaluations,
                converged=result.converged,
                elapsed_time=result.elapsed_time,
                param_names=string.(spec.param_names),
                moment_names=mnames,
                lower_bounds=spec.lower_bounds,
                upper_bounds=spec.upper_bounds)

        # Save history if available
        if !isnothing(result.history_objective)
            jldsave(joinpath(pso_config.output_dir, "pso_history.jld2");
                    history_theta=result.history_theta,
                    history_objective=result.history_objective)
        end
    catch e
        @warn "Failed to save results: $e"
    end

    # Save summary to text file
    summary_file = joinpath(pso_config.output_dir, "estimation_summary.txt")
    try
        open(summary_file, "w") do io
            println(io, "SMM-PSO Estimation Results")
            println(io, "="^50)
            println(io, "")
            println(io, "Estimated Parameters:")
            for (i, pname) in enumerate(spec.param_names)
                @printf(io, "  %-16s = %.6f\n", string(pname), result.theta_best[i])
            end
            println(io, "")
            println(io, "Objective: $(result.objective_best)")
            println(io, "")
            println(io, "Moments (simulated vs data):")
            for i in 1:nm
                @printf(io, "  %-20s sim=%.6f  data=%.6f  diff=%.6f\n",
                        mnames[i],
                        result.moments_best[i], result.moments_data[i],
                        result.moments_best[i] - result.moments_data[i])
            end
            println(io, "")
            @printf(io, "Iterations: %d\n", result.n_iterations)
            @printf(io, "Evaluations: %d\n", result.n_evaluations)
            @printf(io, "Converged: %s\n", result.converged ? "yes" : "no")
            @printf(io, "Elapsed time: %.1f seconds\n", result.elapsed_time)
        end
    catch e
        @warn "Failed to save summary: $e"
    end
end
