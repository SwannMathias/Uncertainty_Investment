"""
SMM objective function: maps parameter vector theta to objective value Q(theta).

# Pipeline
theta -> adjustment costs (via EstimationSpec) -> ModelParameters -> VFI (with warm start) ->
simulate firms -> construct panel -> compute moments -> Q(theta)

# Economic context
Each evaluation solves the full dynamic programming problem for a given
set of adjustment cost parameters, then simulates firms and computes
moments that can be compared to empirical targets.
"""

"""
    ParticleState

Mutable state for a single PSO particle, including warm-start cache.

# Fields
- `theta`: Current position in parameter space
- `velocity`: PSO velocity vector
- `theta_best`: Best position found by this particle
- `objective_best`: Best objective value found
- `V_cache`: Cached value function from last evaluation (for warm-starting VFI)
- `I_cache`: Cached investment policy from last evaluation
- `n_evaluations`: Number of objective evaluations performed
- `last_objective`: Most recent objective value
- `converged_last`: Whether VFI converged on the last evaluation

# Warm-starting
When a particle moves from theta_old to theta_new, its cached V_cache and
I_cache are passed as V_init to value_function_iteration. For small PSO
steps, this reduces VFI iterations from ~100-200 to ~5-20.
"""
mutable struct ParticleState
    theta::Vector{Float64}
    velocity::Vector{Float64}
    theta_best::Vector{Float64}
    objective_best::Float64
    V_cache::Union{Nothing, Array{Float64,3}}
    I_cache::Union{Nothing, Array{Float64,3}}
    n_evaluations::Int
    last_objective::Float64
    converged_last::Bool
end

"""
    ParticleState(theta::Vector{Float64})

Initialize a particle at position theta with zero velocity and no cache.
Dimension-agnostic: velocity size matches theta length.
"""
function ParticleState(theta::Vector{Float64})
    n = length(theta)
    return ParticleState(
        copy(theta),
        zeros(n),
        copy(theta),
        Inf,
        nothing,
        nothing,
        0,
        Inf,
        false
    )
end

"""
    smm_objective(theta, config, grids, shocks, V_init, I_init)

Evaluate the SMM objective for a single parameter vector.

# Arguments
- `theta`: Parameter vector (length = n_params(config.estimation_spec))
- `config`: SMMConfig with calibration, simulation settings, and empirical targets
- `grids`: Pre-constructed StateGrids (shared across evaluations)
- `shocks`: Pre-generated ShockPanel (identical across all evaluations)
- `V_init`: Initial value function guess for warm-starting VFI (or nothing)
- `I_init`: Initial policy guess (unused by VFI, kept for diagnostics)

# Returns
Named tuple: (objective, moments, V, I_policy, converged)
- `objective`: Q(theta) = (m_sim - m_data)' W (m_sim - m_data), or Inf if VFI fails
- `moments`: Vector of simulated moments (length = n_moments)
- `V`: Value function array (for warm-starting next evaluation)
- `I_policy`: Investment policy array
- `converged`: Whether VFI converged

# Economic interpretation
Each evaluation:
1. Constructs stage-specific adjustment costs from theta via EstimationSpec
2. Solves the nested Bellman equation via VFI
3. Simulates a panel of firms under the solved policy
4. Computes moments from the simulated panel
5. Returns the SMM distance to empirical targets
"""
function smm_objective(theta::Vector{Float64}, config::SMMConfig,
                       grids::StateGrids, shocks::ShockPanel,
                       V_init::Union{Nothing, Array{Float64,3}},
                       I_init::Union{Nothing, Array{Float64,3}})
    spec = config.estimation_spec
    nm = n_moments(spec)

    # 1. Construct adjustment costs from theta via EstimationSpec
    ac_begin, ac_mid = build_adjustment_costs(theta, spec)

    # 2. Build ModelParameters from calibration
    params = build_model_parameters(config.calibration)

    # 3. Solve model via VFI with warm start
    sol = try
        value_function_iteration(grids, params, ac_begin, ac_mid;
                                 V_init=V_init,
                                 verbose=false,
                                 use_parallel=true)
    catch e
        @warn "VFI failed for theta=$theta: $e"
        return (objective=Inf,
                moments=fill(NaN, nm),
                V=V_init,
                I_policy=I_init,
                converged=false)
    end

    # 4. Check convergence — penalize non-convergent solutions
    if !sol.convergence.converged
        @warn "VFI did not converge for theta=$theta " *
              "($(sol.convergence.iterations) iterations, " *
              "dist=$(sol.convergence.final_distance))"
        return (objective=Inf,
                moments=fill(NaN, nm),
                V=sol.V,
                I_policy=sol.I_policy,
                converged=false)
    end

    # 5. Simulate firms using pre-generated shocks
    T_total = config.T_years + config.burn_in_years
    histories = try
        simulate_firm_panel(sol, shocks;
                           K_init=nothing,
                           T_years=T_total,
                           use_parallel=true,
                           verbose=false)
    catch e
        @warn "Simulation failed for theta=$theta: $e"
        return (objective=Inf,
                moments=fill(NaN, nm),
                V=sol.V,
                I_policy=sol.I_policy,
                converged=true)
    end

    # 6. Construct panel and discard burn-in
    panel = construct_estimation_panel(histories)
    df = panel.df[panel.df.year .> config.burn_in_years, :]

    # 7. Compute simulated moments
    m_sim = compute_simulated_moments(df, config)

    # 8. Handle degenerate moments (NaN from failed regressions)
    if any(isnan.(m_sim))
        return (objective=Inf,
                moments=m_sim,
                V=sol.V,
                I_policy=sol.I_policy,
                converged=true)
    end

    # 9. Compute SMM objective: Q(theta) = (m_sim - m_data)' W (m_sim - m_data)
    diff = m_sim - config.m_data
    Q = diff' * config.W * diff

    return (objective=Q,
            moments=m_sim,
            V=sol.V,
            I_policy=sol.I_policy,
            converged=true)
end
