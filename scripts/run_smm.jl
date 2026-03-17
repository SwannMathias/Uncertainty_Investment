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
    --m_data "a,b,..."       Empirical moment targets (comma-separated, must match moments)
    --output DIR             Output directory (default: output/estimation/run_001)
    --resume FILE            Resume from checkpoint file
    --seed N                 Shock generation seed (default: 42)
    --transform MODE         Revision transform: asinh, log, level_over_K (default: asinh)
    --fixed_params "F_begin=0.5,F_mid=0.5"       Parameters held constant
    --estimated_params "phi_begin=0:20,phi_mid=0:20"  Parameters to estimate (name=lower:upper)

If neither --fixed_params nor --estimated_params is given, all 4 parameters are estimated
(backward compatible with original specification).

Example (default 4-parameter composite):
    julia -t 1 scripts/run_smm.jl \\
        --n_particles 20 \\
        --max_iterations 100 \\
        --m_data "0.35,0.50,-0.15,0.10" \\
        --output output/estimation/run_001/

Example (convex only with fixed costs held constant):
    julia -t 1 scripts/run_smm.jl \\
        --fixed_params "F_begin=0.5,F_mid=0.5" \\
        --estimated_params "phi_begin=0:20,phi_mid=0:20" \\
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
using StatsModels: @formula

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

"""
Parse "F_begin=0.5,F_mid=0.5" into Dict(:F_begin => 0.5, :F_mid => 0.5)
"""
function parse_fixed_params(s::String)
    result = Dict{Symbol,Float64}()
    for pair in split(s, ",")
        k, v = split(strip(pair), "=")
        result[Symbol(strip(k))] = parse(Float64, strip(v))
    end
    return result
end

"""
Parse "phi_begin=0:20,phi_mid=0:20" into Dict(:phi_begin => (0.0, 20.0), ...)
"""
function parse_estimated_params(s::String)
    result = Dict{Symbol,Tuple{Float64,Float64}}()
    for pair in split(s, ",")
        k, bounds_str = split(strip(pair), "=")
        lb, ub = split(strip(bounds_str), ":")
        result[Symbol(strip(k))] = (parse(Float64, strip(lb)), parse(Float64, strip(ub)))
    end
    return result
end

"""
Build default moments for estimated parameters.

- If any F parameter is estimated, include ShareZeroMoment for its stage
- If any phi parameter is estimated, include RegressionCoefficientMoment for its stage
"""
function default_moments_for_params(estimated_params::Dict{Symbol,Tuple{Float64,Float64}})
    moments = AbstractMoment[]

    # Share-of-zero moments for fixed cost parameters
    if haskey(estimated_params, :F_begin)
        push!(moments, ShareZeroMoment(:begin, "share_zero_begin"))
    end
    if haskey(estimated_params, :F_mid)
        push!(moments, ShareZeroMoment(:mid, "share_zero_mid"))
    end

    # Regression moments for convex cost parameters
    if haskey(estimated_params, :phi_begin)
        push!(moments, RegressionCoefficientMoment(
            :begin,
            @formula(revision_begin ~ log_sigma + log_K + log_D),
            :log_sigma,
            "coef_begin_sigma"
        ))
    end
    if haskey(estimated_params, :phi_mid)
        push!(moments, RegressionCoefficientMoment(
            :mid,
            @formula(revision_mid ~ log_sigma_half + log_K + log_D),
            :log_sigma_half,
            "coef_mid_sigma"
        ))
    end

    return moments
end

# ============================================================================
# Main
# ============================================================================

function main()
    opts = parse_args(ARGS)

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

    # Determine estimation spec from dict-based interface or default
    if haskey(opts, "estimated_params")
        ep = parse_estimated_params(opts["estimated_params"])
        fp = haskey(opts, "fixed_params") ? parse_fixed_params(opts["fixed_params"]) : Dict{Symbol,Float64}()
        moments = default_moments_for_params(ep)

        nm = length(moments)
        if !isnothing(m_data)
            @assert length(m_data) == nm "m_data must have $nm values, got $(length(m_data))"
        end

        config = SMMConfig(
            calibration = FixedCalibration(),
            fixed_params = fp,
            estimated_params = ep,
            moments = moments,
            m_data = m_data,
            shock_seed = shock_seed,
            revision_transform = transform
        )
    else
        # Default: all 4 parameters estimated
        if !isnothing(m_data)
            @assert length(m_data) == 4 "Default spec has 4 moments, got $(length(m_data)) m_data values"
        end

        config = SMMConfig(
            calibration = FixedCalibration(),
            m_data = m_data,
            shock_seed = shock_seed,
            revision_transform = transform
        )
    end

    spec = config.estimation_spec

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
