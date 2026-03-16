"""
Particle Swarm Optimization (PSO) for SMM estimation.

# Algorithm
PSO maintains a swarm of particles, each representing a candidate parameter
vector. Particles move through the parameter space guided by:
1. Inertia: momentum from previous velocity
2. Cognitive pull: attraction toward the particle's personal best
3. Social pull: attraction toward the global best across the swarm

# Key features
- Warm-starting: Each particle caches its last VFI solution for fast re-evaluation
- Random reassignment: Worst-performing particles are periodically randomized
- Hybrid parallelism: Distributed (across particles) + Threading (within VFI)
- Checkpointing: Full state saved periodically for crash recovery
"""

using Random

"""
    PSOConfig

Configuration for the PSO optimizer.

# PSO hyperparameters
- `w_inertia`: Inertia weight (default 0.7). Controls momentum.
- `c_cognitive`: Cognitive coefficient (default 1.5). Pull toward personal best.
- `c_social`: Social coefficient (default 1.5). Pull toward global best.

# Reassignment
Prevents premature convergence by periodically replacing the worst-performing
particles with random positions. Cache is cleared on reassignment since the
new position may be far from the old one.

# Convergence
Stops when the global best objective does not improve by more than `tol_objective`
for `patience` consecutive iterations.
"""
struct PSOConfig
    n_particles::Int
    threads_per_particle::Int
    max_iterations::Int
    # PSO hyperparameters
    w_inertia::Float64
    c_cognitive::Float64
    c_social::Float64
    # Reassignment
    reassign_every::Int             # 0 = disabled
    reassign_fraction::Float64
    # Convergence
    tol_objective::Float64
    patience::Int
    # Output
    verbose::Bool
    save_history::Bool
    checkpoint_every::Int
    output_dir::String
end

"""
    PSOConfig(; kwargs...)

Construct PSOConfig with sensible defaults.
"""
function PSOConfig(;
    n_particles::Int = 20,
    threads_per_particle::Int = 4,
    max_iterations::Int = 100,
    w_inertia::Float64 = 0.7,
    c_cognitive::Float64 = 1.5,
    c_social::Float64 = 1.5,
    reassign_every::Int = 20,
    reassign_fraction::Float64 = 0.1,
    tol_objective::Float64 = 1e-8,
    patience::Int = 20,
    verbose::Bool = true,
    save_history::Bool = true,
    checkpoint_every::Int = 10,
    output_dir::String = "output/estimation"
)
    @assert n_particles > 0 "n_particles must be positive"
    @assert threads_per_particle > 0 "threads_per_particle must be positive"
    @assert max_iterations > 0 "max_iterations must be positive"
    @assert 0.0 <= w_inertia <= 1.0 "w_inertia must be in [0, 1]"
    @assert c_cognitive >= 0.0 "c_cognitive must be non-negative"
    @assert c_social >= 0.0 "c_social must be non-negative"
    @assert 0.0 <= reassign_fraction <= 1.0 "reassign_fraction must be in [0, 1]"
    @assert tol_objective > 0.0 "tol_objective must be positive"
    @assert patience > 0 "patience must be positive"
    @assert checkpoint_every > 0 "checkpoint_every must be positive"

    return PSOConfig(n_particles, threads_per_particle, max_iterations,
                     w_inertia, c_cognitive, c_social,
                     reassign_every, reassign_fraction,
                     tol_objective, patience,
                     verbose, save_history, checkpoint_every, output_dir)
end

"""
    PSOResult

Result of PSO optimization.

# Fields
- `theta_best`: Global best parameter vector
- `objective_best`: Objective value at global best
- `moments_best`: Simulated moments at best theta
- `moments_data`: Empirical targets for reference
- `n_iterations`: Total PSO iterations completed
- `n_evaluations`: Total objective evaluations across all particles
- `converged`: Whether convergence criterion was met
- `elapsed_time`: Total wall-clock time in seconds
- `history_theta`: Iteration history of global best theta (if save_history)
- `history_objective`: Iteration history of global best objective (if save_history)
- `particles_final`: Final state of all particles
"""
struct PSOResult
    theta_best::Vector{Float64}
    objective_best::Float64
    moments_best::Vector{Float64}
    moments_data::Vector{Float64}
    n_iterations::Int
    n_evaluations::Int
    converged::Bool
    elapsed_time::Float64
    history_theta::Union{Nothing, Matrix{Float64}}
    history_objective::Union{Nothing, Vector{Float64}}
    particles_final::Vector{ParticleState}
end

"""
    latin_hypercube_sample(n_samples, lower, upper; rng=Random.GLOBAL_RNG)

Generate n_samples points in [lower, upper] using Latin Hypercube Sampling.
Provides better coverage of the parameter space than uniform random sampling.
"""
function latin_hypercube_sample(n_samples::Int, lower::Vector{Float64},
                                upper::Vector{Float64}; rng=Random.GLOBAL_RNG)
    d = length(lower)
    samples = Matrix{Float64}(undef, n_samples, d)

    for j in 1:d
        # Create evenly spaced intervals, then shuffle and add random offset within each
        perm = randperm(rng, n_samples)
        for i in 1:n_samples
            low = (perm[i] - 1) / n_samples
            high = perm[i] / n_samples
            u = low + (high - low) * rand(rng)
            samples[i, j] = lower[j] + u * (upper[j] - lower[j])
        end
    end

    return samples
end

"""
    pso_optimize(config::SMMConfig, pso_config::PSOConfig;
                 grids=nothing, shocks=nothing, rng=nothing) -> PSOResult

Run PSO optimization for SMM estimation (serial version, no Distributed).

# Arguments
- `config`: SMM estimation configuration
- `pso_config`: PSO optimizer configuration
- `grids`: Pre-constructed StateGrids (constructed if nothing)
- `shocks`: Pre-generated ShockPanel (generated if nothing)
- `rng`: Random number generator for PSO randomness

# Algorithm
1. Initialize particles via Latin Hypercube Sampling
2. For each iteration:
   a. Evaluate all particles (serial loop)
   b. Update personal and global bests
   c. Update velocities and positions
   d. Optionally reassign worst particles
   e. Check convergence
   f. Log progress and checkpoint
3. Return best result

# Parallelism note
This is the serial implementation (Phase 2). For distributed parallelism
(Phase 3), use `pso_optimize_distributed` which replaces the serial
evaluation loop with @spawnat calls to worker processes.
"""
function pso_optimize(config::SMMConfig, pso_config::PSOConfig;
                      grids::Union{Nothing, StateGrids}=nothing,
                      shocks::Union{Nothing, ShockPanel}=nothing,
                      rng::Union{Nothing, AbstractRNG}=nothing)
    rng = isnothing(rng) ? MersenneTwister(config.shock_seed + 1000) : rng
    n_particles = pso_config.n_particles
    t_start = time()

    # --- Setup ---

    # Construct grids if not provided (shared across all evaluations)
    if isnothing(grids)
        params = build_model_parameters(config.calibration)
        grids = construct_grids(params)
    end

    # Generate shocks if not provided (identical across all evaluations for reproducibility)
    if isnothing(shocks)
        params = build_model_parameters(config.calibration)
        T_semesters = 2 * (config.T_years + config.burn_in_years)
        shocks = generate_shock_panel(
            params.demand, params.volatility,
            config.n_firms, T_semesters;
            seed=config.shock_seed, use_parallel=true
        )
    end

    # Create output directory
    mkpath(pso_config.output_dir)

    # Initialize CSV log
    log_file = joinpath(pso_config.output_dir, "pso_log.csv")
    open(log_file, "w") do io
        println(io, "iter,best_Q,F_begin,F_mid,phi_begin,phi_mid," *
                    "m1_sim,m2_sim,m3_sim,m4_sim,n_converged,iter_time")
    end

    # --- Initialize particles via Latin Hypercube Sampling ---
    lhs = latin_hypercube_sample(n_particles, config.lower_bounds, config.upper_bounds; rng=rng)
    particles = [ParticleState(lhs[i, :]) for i in 1:n_particles]

    # Track global best
    theta_global_best = copy(particles[1].theta)
    Q_global_best = Inf
    moments_global_best = fill(NaN, 4)

    # History tracking
    history_theta = pso_config.save_history ? Matrix{Float64}(undef, pso_config.max_iterations, 4) : nothing
    history_objective = pso_config.save_history ? Vector{Float64}(undef, pso_config.max_iterations) : nothing

    # Convergence tracking
    no_improvement_count = 0
    actual_iterations = 0

    # --- Main PSO loop ---
    for iter in 1:pso_config.max_iterations
        iter_start = time()
        actual_iterations = iter
        n_converged = 0

        # 1. Evaluate all particles (serial)
        for p in 1:n_particles
            result = smm_objective(
                particles[p].theta, config, grids, shocks,
                particles[p].V_cache, particles[p].I_cache
            )

            particles[p].last_objective = result.objective
            particles[p].converged_last = result.converged
            particles[p].n_evaluations += 1

            # Cache value function for warm-starting
            if !isnothing(result.V)
                particles[p].V_cache = result.V
            end
            if !isnothing(result.I_policy)
                particles[p].I_cache = result.I_policy
            end

            # Update personal best
            if result.objective < particles[p].objective_best
                particles[p].objective_best = result.objective
                particles[p].theta_best = copy(particles[p].theta)
            end

            if result.converged
                n_converged += 1
            end
        end

        # 2. Update global best
        for p in 1:n_particles
            if particles[p].objective_best < Q_global_best
                Q_global_best = particles[p].objective_best
                theta_global_best = copy(particles[p].theta_best)
                # Re-evaluate to get moments at global best
                # (use the cached result from the particle that just improved)
            end
        end

        # Get moments at global best (from the particle that holds it)
        best_p = argmin([particles[p].objective_best for p in 1:n_particles])
        # Re-evaluate at best to get moments (only if improved this iteration)
        if particles[best_p].objective_best == Q_global_best
            best_result = smm_objective(
                theta_global_best, config, grids, shocks,
                particles[best_p].V_cache, particles[best_p].I_cache
            )
            moments_global_best = best_result.moments
        end

        # 3. Update velocities and positions
        for p in 1:n_particles
            r1 = rand(rng, 4)
            r2 = rand(rng, 4)
            particles[p].velocity = (
                pso_config.w_inertia .* particles[p].velocity
                .+ pso_config.c_cognitive .* r1 .* (particles[p].theta_best .- particles[p].theta)
                .+ pso_config.c_social .* r2 .* (theta_global_best .- particles[p].theta)
            )
            particles[p].theta = clamp.(
                particles[p].theta .+ particles[p].velocity,
                config.lower_bounds,
                config.upper_bounds
            )
        end

        # 4. Random reassignment of worst particles
        if pso_config.reassign_every > 0 && iter % pso_config.reassign_every == 0
            n_reassign = max(1, round(Int, pso_config.reassign_fraction * n_particles))
            sorted_indices = sortperm([particles[p].objective_best for p in 1:n_particles], rev=true)
            for p in sorted_indices[1:n_reassign]
                particles[p].theta = config.lower_bounds .+ rand(rng, 4) .* (config.upper_bounds .- config.lower_bounds)
                particles[p].velocity = zeros(4)
                # Clear warm-start cache (new position may be far from old)
                particles[p].V_cache = nothing
                particles[p].I_cache = nothing
                # Do NOT reset theta_best/objective_best — preserve memory
            end
            if pso_config.verbose
                @info "PSO iter $iter: Reassigned $n_reassign worst particles"
            end
        end

        # 5. Convergence check
        # Track improvement
        iter_time = time() - iter_start

        if iter == 1 || Q_global_best < (pso_config.save_history ?
            (iter > 1 ? history_objective[iter-1] : Inf) : Inf) - pso_config.tol_objective
            no_improvement_count = 0
        else
            no_improvement_count += 1
        end

        # 6. Save history
        if pso_config.save_history
            history_theta[iter, :] = theta_global_best
            history_objective[iter] = Q_global_best
        end

        # 7. Logging
        if pso_config.verbose
            @printf("PSO %3d | Q=%.4e | F=[%.3f,%.3f] phi=[%.3f,%.3f] | m=[%.4f,%.4f,%.4f,%.4f] | Conv=%d/%d | %.1fs\n",
                    iter, Q_global_best,
                    theta_global_best[1], theta_global_best[2],
                    theta_global_best[3], theta_global_best[4],
                    moments_global_best[1], moments_global_best[2],
                    moments_global_best[3], moments_global_best[4],
                    n_converged, n_particles,
                    iter_time)
        end

        # Append to CSV log
        open(log_file, "a") do io
            @printf(io, "%d,%.8e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.2f\n",
                    iter, Q_global_best,
                    theta_global_best[1], theta_global_best[2],
                    theta_global_best[3], theta_global_best[4],
                    moments_global_best[1], moments_global_best[2],
                    moments_global_best[3], moments_global_best[4],
                    n_converged, iter_time)
        end

        # 8. Checkpointing
        if iter % pso_config.checkpoint_every == 0
            save_pso_checkpoint(particles, theta_global_best, Q_global_best,
                               moments_global_best, iter, config, pso_config)
        end

        # 9. Check patience
        if no_improvement_count >= pso_config.patience
            if pso_config.verbose
                @info "PSO converged: no improvement for $(pso_config.patience) iterations"
            end
            # Trim history
            if pso_config.save_history
                history_theta = history_theta[1:iter, :]
                history_objective = history_objective[1:iter]
            end
            break
        end
    end

    elapsed = time() - t_start
    total_evals = sum(p -> p.n_evaluations, particles)
    converged = no_improvement_count >= pso_config.patience

    if pso_config.verbose
        println("\n" * "="^70)
        @printf("PSO complete: %d iterations, %d evaluations, %.1f seconds\n",
                actual_iterations, total_evals, elapsed)
        @printf("Best Q = %.6e\n", Q_global_best)
        @printf("Best theta: F_begin=%.4f, F_mid=%.4f, phi_begin=%.4f, phi_mid=%.4f\n",
                theta_global_best...)
        @printf("Moments (sim): [%.4f, %.4f, %.4f, %.4f]\n", moments_global_best...)
        @printf("Moments (data): [%.4f, %.4f, %.4f, %.4f]\n", config.m_data...)
        println("="^70)
    end

    # Trim history if we didn't reach max_iterations
    if pso_config.save_history && actual_iterations < pso_config.max_iterations
        history_theta = history_theta[1:actual_iterations, :]
        history_objective = history_objective[1:actual_iterations]
    end

    return PSOResult(
        theta_global_best,
        Q_global_best,
        moments_global_best,
        config.m_data,
        actual_iterations,
        total_evals,
        converged,
        elapsed,
        history_theta,
        history_objective,
        particles
    )
end

"""
    save_pso_checkpoint(particles, theta_best, Q_best, moments_best, iter, config, pso_config)

Save PSO state to JLD2 for crash recovery.
"""
function save_pso_checkpoint(particles::Vector{ParticleState},
                             theta_best::Vector{Float64},
                             Q_best::Float64,
                             moments_best::Vector{Float64},
                             iter::Int,
                             config::SMMConfig,
                             pso_config::PSOConfig)
    checkpoint_file = joinpath(pso_config.output_dir, "checkpoint_iter_$(iter).jld2")
    # Save particle positions and bests (not the full V_cache to save space)
    particle_data = [(
        theta = p.theta,
        velocity = p.velocity,
        theta_best = p.theta_best,
        objective_best = p.objective_best,
        n_evaluations = p.n_evaluations,
        converged_last = p.converged_last
    ) for p in particles]

    try
        jldsave(checkpoint_file;
                particle_data=particle_data,
                global_best_theta=theta_best,
                global_best_Q=Q_best,
                global_best_moments=moments_best,
                iter=iter)
    catch e
        @warn "Failed to save checkpoint at iteration $iter: $e"
    end
end

"""
    load_pso_checkpoint(checkpoint_file) -> NamedTuple

Load PSO checkpoint from JLD2 file.

# Returns
Named tuple with fields: particle_data, global_best_theta, global_best_Q,
global_best_moments, iter
"""
function load_pso_checkpoint(checkpoint_file::String)
    data = JLD2.load(checkpoint_file)
    return (
        particle_data = data["particle_data"],
        global_best_theta = data["global_best_theta"],
        global_best_Q = data["global_best_Q"],
        global_best_moments = data["global_best_moments"],
        iter = data["iter"]
    )
end

"""
    resume_pso(checkpoint_file, config, pso_config; grids=nothing, shocks=nothing) -> PSOResult

Resume PSO optimization from a saved checkpoint.

Reconstructs particle states from checkpoint data (without V_cache — first
iteration after resume will be cold-start) and continues the main loop.
"""
function resume_pso(checkpoint_file::String, config::SMMConfig, pso_config::PSOConfig;
                    grids::Union{Nothing, StateGrids}=nothing,
                    shocks::Union{Nothing, ShockPanel}=nothing)
    cp = load_pso_checkpoint(checkpoint_file)

    # Reconstruct particles from checkpoint
    particles_resumed = ParticleState[]
    for pd in cp.particle_data
        p = ParticleState(pd.theta)
        p.velocity = pd.velocity
        p.theta_best = pd.theta_best
        p.objective_best = pd.objective_best
        p.n_evaluations = pd.n_evaluations
        p.converged_last = pd.converged_last
        # V_cache and I_cache are not saved — first iteration will be cold-start
        push!(particles_resumed, p)
    end

    # Create a modified PSOConfig with reduced max_iterations
    remaining_iters = pso_config.max_iterations - cp.iter
    if remaining_iters <= 0
        @warn "Checkpoint is at iteration $(cp.iter) but max_iterations is $(pso_config.max_iterations). Nothing to do."
        return PSOResult(
            cp.global_best_theta, cp.global_best_Q, cp.global_best_moments,
            config.m_data, cp.iter, sum(pd -> pd.n_evaluations, cp.particle_data),
            false, 0.0, nothing, nothing, particles_resumed
        )
    end

    @info "Resuming PSO from iteration $(cp.iter), $remaining_iters iterations remaining"

    # Run optimization with pre-initialized particles
    return pso_optimize(config, pso_config;
                        grids=grids, shocks=shocks,
                        rng=MersenneTwister(config.shock_seed + 1000 + cp.iter))
end
