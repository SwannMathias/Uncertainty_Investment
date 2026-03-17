"""
CLI entry point for SMM-PSO estimation.

Usage:
    julia -t 1 scripts/run_smm.jl [options]

For distributed parallelism (Phase 3), workers are spawned internally
with threads_per_particle threads each. The master process runs single-threaded.

Options (via command-line arguments):
    --n_particles N          Number of PSO particles (default: 20)
    --threads_per_particle N Threads per worker (default: 4)
    --max_iterations N       Maximum PSO iterations (default: 100)
    --m_data "a,b,c,d"       Empirical moment targets (comma-separated, must match spec)
    --output DIR             Output directory (default: output/estimation/run_001)
    --config FILE            Load config from TOML file (overrides other options)
    --resume FILE            Resume from checkpoint file
    --seed N                 Shock generation seed (default: 42)
    --transform MODE         Revision transform: asinh, log, level_over_K (default: asinh)
    --cost_spec SPEC         Cost specification: composite, convex_only, fixed_only (default: composite)

Example (default 4-parameter composite):
    julia -t 1 scripts/run_smm.jl \\
        --n_particles 20 \\
        --threads_per_particle 8 \\
        --max_iterations 100 \\
        --m_data "0.35,0.50,-0.15,0.10" \\
        --output output/estimation/run_001/

Example (2-parameter convex-only):
    julia -t 1 scripts/run_smm.jl \\
        --cost_spec convex_only \\
        --m_data "-0.15,0.10" \\
        --max_iterations 50 \\
        --output output/estimation/convex_run/
"""

using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(project_root)

using UncertaintyInvestment
using Printf
using Random

# ============================================================================
# Argument parsing
# ============================================================================

function parse_args(args)
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            key = args[i][3:end]
            if i + 1 <= length(args) && !startswith(args[i+1], "--")
                opts[key] = args[i+1]
                i += 2
            else
                opts[key] = "true"
                i += 1
            end
        else
            i += 1
        end
    end
    return opts
end

function parse_transform(s::String)
    s_lower = lowercase(s)
    if s_lower == "asinh"
        return ASINH_TRANSFORM
    elseif s_lower == "log"
        return LOG_TRANSFORM
    elseif s_lower == "level_over_k"
        return LEVEL_OVER_K_TRANSFORM
    else
        error("Unknown transform: $s. Use: asinh, log, or level_over_k")
    end
end

function parse_cost_spec(s::String)
    s_lower = lowercase(s)
    if s_lower == "composite"
        return composite_spec()
    elseif s_lower == "convex_only"
        return convex_only_spec()
    elseif s_lower == "fixed_only"
        return fixed_only_spec()
    else
        error("Unknown cost_spec: $s. Use: composite, convex_only, or fixed_only")
    end
end

# ============================================================================
# Main
# ============================================================================

function main()
    opts = parse_args(ARGS)

    # Parse cost specification first (determines expected dimensions)
    spec = parse_cost_spec(get(opts, "cost_spec", "composite"))
    np = n_params(spec)
    nm = n_moments(spec)

    # Parse options with defaults
    n_particles = parse(Int, get(opts, "n_particles", "20"))
    threads_per_particle = parse(Int, get(opts, "threads_per_particle", "4"))
    max_iterations = parse(Int, get(opts, "max_iterations", "100"))
    output_dir = get(opts, "output", "output/estimation/run_001")
    shock_seed = parse(Int, get(opts, "seed", "42"))
    transform = parse_transform(get(opts, "transform", "asinh"))

    # Parse empirical targets
    m_data = if haskey(opts, "m_data")
        parse.(Float64, split(opts["m_data"], ","))
    else
        nothing  # Will use SMMConfig defaults
    end

    if !isnothing(m_data)
        @assert length(m_data) == nm "m_data must have exactly $nm values for $(get(opts, "cost_spec", "composite")) spec, got $(length(m_data))"
    end

    # Build configuration
    config = SMMConfig(
        calibration = FixedCalibration(),
        estimation_spec = spec,
        m_data = m_data,
        shock_seed = shock_seed,
        revision_transform = transform
    )

    pso_config = PSOConfig(
        n_particles = n_particles,
        threads_per_particle = threads_per_particle,
        max_iterations = max_iterations,
        verbose = true,
        save_history = true,
        output_dir = output_dir
    )

    # Check for resume
    if haskey(opts, "resume")
        @info "Resuming from checkpoint: $(opts["resume"])"
        result = resume_pso(opts["resume"], config, pso_config)
    else
        result = run_smm_estimation(config, pso_config)
    end

    # Print final results
    println("\n" * "="^70)
    println("ESTIMATION COMPLETE")
    println("="^70)
    @printf("Best parameters:\n")
    for (i, pname) in enumerate(spec.param_names)
        @printf("  %-16s = %.6f\n", string(pname), result.theta_best[i])
    end
    @printf("\nObjective: %.8e\n", result.objective_best)
    @printf("Converged: %s\n", result.converged ? "yes" : "no")
    @printf("Time: %.1f seconds\n", result.elapsed_time)
    println("Results saved to: $output_dir")
    println("="^70)

    return result
end

main()
