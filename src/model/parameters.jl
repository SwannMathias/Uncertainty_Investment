"""
    DemandProcess

Autoregressive demand process parameters at semester frequency.

When `process_space = :log` (default):
    log D_{s+½} = μ_D(1-ρ_D) + ρ_D log D_s + σ_s ε_{s+½}

When `process_space = :level`:
    D_{s+½} = μ_D(1-ρ_D) + ρ_D D_s + σ_s ε_{s+½}

In both cases, μ_D is the long-run mean of the variable in the chosen space.
"""
@with_kw struct DemandProcess
    mu_D::Float64 = 0.0       # Long-run mean (of log demand if :log, of demand level if :level)
    rho_D::Float64 = 0.9       # Persistence (semester)
    process_space::Symbol = :log  # :log or :level

    function DemandProcess(mu_D, rho_D, process_space)
        @assert 0.0 <= rho_D < 1.0 "rho_D must be in [0, 1)"
        @assert process_space in (:log, :level) "process_space must be :log or :level"
        new(mu_D, rho_D, process_space)
    end
end

"""
    AbstractVolatilitySpec

Abstract supertype for volatility process specifications.
Concrete subtypes: `VolatilityProcess` (continuous AR(1)), `TwoStateVolatility` (Markov switching).
"""
abstract type AbstractVolatilitySpec end

"""
    VolatilityProcess <: AbstractVolatilitySpec

Stochastic volatility process parameters at semester frequency.

When `process_space = :log` (default):
    log σ_{s+½} = σ̄(1-ρ_σ) + ρ_σ log σ_s + σ_η η_{s+½}

When `process_space = :level`:
    σ_{s+½} = σ̄(1-ρ_σ) + ρ_σ σ_s + σ_η η_{s+½}
"""
@with_kw struct VolatilityProcess <: AbstractVolatilitySpec
    sigma_bar::Float64 = log(0.1)    # Long-run mean (of log vol if :log, of vol level if :level)
    rho_sigma::Float64 = 0.95      # Persistence (semester)
    sigma_eta::Float64 = 0.1       # Volatility of volatility
    rho_epsilon_eta::Float64 = 0.0      # Correlation between demand and volatility shocks
    process_space::Symbol = :log    # :log or :level

    function VolatilityProcess(sigma_bar, rho_sigma, sigma_eta, rho_epsilon_eta, process_space)
        @assert 0.0 <= rho_sigma < 1.0 "rho_sigma must be in [0, 1)"
        @assert sigma_eta > 0.0 "sigma_eta must be positive"
        @assert -1.0 <= rho_epsilon_eta <= 1.0 "rho_epsilon_eta must be in [-1, 1]"
        @assert process_space in (:log, :level) "process_space must be :log or :level"
        new(sigma_bar, rho_sigma, sigma_eta, rho_epsilon_eta, process_space)
    end
end

"""
    TwoStateVolatility <: AbstractVolatilitySpec

Two-state Markov switching volatility process.

Volatility alternates between two states with transition matrix Pi_sigma.
The `sigma_levels` are interpreted according to `process_space`:
- `:level` (default): values are demand innovation standard deviations directly
- `:log`: values are log-volatilities (exponentiated to get std devs)
"""
@with_kw struct TwoStateVolatility <: AbstractVolatilitySpec
    sigma_levels::Vector{Float64}      # Two volatility values [σ_low, σ_high]
    Pi_sigma::Matrix{Float64}          # 2×2 transition matrix
    process_space::Symbol = :level     # :log or :level

    function TwoStateVolatility(sigma_levels, Pi_sigma, process_space)
        @assert length(sigma_levels) == 2 "sigma_levels must have exactly 2 elements"
        @assert size(Pi_sigma) == (2, 2) "Pi_sigma must be a 2×2 matrix"
        @assert all(isapprox.(sum(Pi_sigma, dims=2), 1.0, atol=1e-10)) "Pi_sigma rows must sum to 1"
        @assert all(Pi_sigma .>= 0.0) "Pi_sigma entries must be non-negative"
        @assert process_space in (:log, :level) "process_space must be :log or :level"
        new(sigma_levels, Pi_sigma, process_space)
    end
end

"""
    NumericalSettings

Numerical solution parameters including grid sizes and convergence tolerances.
"""
@with_kw struct NumericalSettings
    # Grid sizes
    n_K::Int = 100           # Capital grid points
    n_D::Int = 15            # Demand states
    n_sigma::Int = 7             # Volatility states

    # Grid bounds (relative to steady state)
    K_min_factor::Float64 = 0.1   # K_min = K_ss * factor
    K_max_factor::Float64 = 3.0   # K_max = K_ss * factor

    # Convergence tolerances
    tol_vfi::Float64 = 1e-6       # Value function tolerance
    tol_policy::Float64 = 1e-6    # Policy function tolerance
    max_iter::Int = 1000          # Maximum VFI iterations

    # Acceleration
    howard_steps::Int = 50        # Howard improvement steps (0 = disabled)

    # Interpolation
    interp_method::Symbol = :linear  # :linear or :cubic

    function NumericalSettings(n_K, n_D, n_sigma, K_min_factor, K_max_factor,
                               tol_vfi, tol_policy, max_iter, howard_steps, interp_method)
        @assert n_K > 0 "n_K must be positive"
        @assert n_D > 0 "n_D must be positive"
        @assert n_sigma > 0 "n_sigma must be positive"
        @assert 0.0 < K_min_factor < K_max_factor "Invalid K bounds"
        @assert tol_vfi > 0.0 "tol_vfi must be positive"
        @assert tol_policy > 0.0 "tol_policy must be positive"
        @assert max_iter > 0 "max_iter must be positive"
        @assert howard_steps >= 0 "howard_steps must be non-negative"
        @assert interp_method in [:linear, :cubic] "Invalid interpolation method"
        new(n_K, n_D, n_sigma, K_min_factor, K_max_factor, tol_vfi, tol_policy,
            max_iter, howard_steps, interp_method)
    end
end

"""
    ModelParameters

Main parameter structure containing all model primitives.
"""
@with_kw struct ModelParameters
    # Technology
    alpha::Float64 = 0.33        # Capital share in production
    epsilon::Float64 = 4.0         # Demand elasticity (must be > 1)
    delta::Float64 = 0.10        # Annual depreciation rate

    # Preferences
    beta::Float64 = 0.96        # Annual discount factor

    # Stochastic processes
    demand::DemandProcess = DemandProcess()
    volatility::AbstractVolatilitySpec = VolatilityProcess()

    # Numerical settings
    numerical::NumericalSettings = NumericalSettings()

    function ModelParameters(alpha, epsilon, delta, beta, demand, volatility, numerical)
        @assert 0.0 < alpha < 1.0 "alpha must be in (0, 1)"
        @assert epsilon > 1.0 "epsilon must be > 1 for positive profits"
        @assert 0.0 < delta < 1.0 "delta must be in (0, 1)"
        @assert 0.0 < beta < 1.0 "beta must be in (0, 1)"
        new(alpha, epsilon, delta, beta, demand, volatility, numerical)
    end
end

"""
    DerivedParameters

Derived parameters computed from ModelParameters.
These are calculated once and reused throughout the solution.
"""
struct DerivedParameters
    gamma::Float64               # Profit function demand exponent
    h::Float64               # Profit function scale parameter
    delta_semester::Float64      # Semester depreciation rate
    beta_semester::Float64      # Semester discount factor
    K_ss::Float64            # Steady-state capital (no uncertainty)
end

"""
    get_derived_parameters(p::ModelParameters) -> DerivedParameters

Compute derived parameters from primitive parameters.

# Formulas
- gamma = (epsilon - 1) / (epsilon - (1 - alpha))
- h = alpha * (1 - 1/epsilon)^(epsilon/alpha) * (1 - alpha)^(epsilon/alpha - 1)
- delta_semester = 1 - (1 - delta)^(1/2)
- beta_semester = beta^(1/2)
- K_ss computed from first-order condition: MPK = delta/beta
"""
function get_derived_parameters(p::ModelParameters)
    # Profit function exponents from iso-elastic demand and Cobb-Douglas technology
    gamma = (p.epsilon - 1) / (p.epsilon - (1 - p.alpha))

    # Scale parameter h
    term1 = p.alpha
    term2 = (1 - 1/p.epsilon)^(p.epsilon / p.alpha)
    term3 = (1 - p.alpha)^(p.epsilon / p.alpha - 1)
    h = term1 * term2 * term3

    # Convert annual rates to semester rates
    delta_semester = 1 - (1 - p.delta)^(1/2)
    beta_semester = p.beta^(1/2)

    # Steady-state capital (deterministic case)
    # From FOC: MPK = delta/beta => h * D_ss^gamma * K_ss^(-gamma) = delta/beta
    # Note: The (1-gamma) terms cancel when taking derivative of profit function:
    #   pi(K,D) = (h/(1-gamma)) * D^gamma * K^(1-gamma)
    #   dpi/dK = (h/(1-gamma)) * (1-gamma) * D^gamma * K^(-gamma) = h * D^gamma * K^(-gamma)
    # With D_ss = exp(mu_D) for :log space, or D_ss = mu_D for :level space
    D_ss = p.demand.process_space == :log ? exp(p.demand.mu_D) : p.demand.mu_D
    user_cost = p.delta  + (1/p.beta-1)  # r = (1/beta - 1)

    # MPK = h * D^gamma * K^(-gamma)
    # K_ss = [h * D_ss^gamma / user_cost]^(1/gamma)
    K_ss = (h * D_ss^gamma / user_cost)^(1/gamma)

    return DerivedParameters(gamma, h, delta_semester, beta_semester, K_ss)
end

"""
    validate_parameters(p::ModelParameters) -> Bool

Perform additional validation checks on parameter combinations.
"""
function validate_parameters(p::ModelParameters)
    # Check that profit function exponent is well-defined
    derived = get_derived_parameters(p)

    if derived.gamma <= 0.0 || derived.gamma >= 1.0
        @warn "Profit function exponent gamma = $(derived.gamma) is outside (0,1). This may cause numerical issues."
        return false
    end

    if derived.K_ss <= 0.0
        @warn "Steady-state capital K_ss = $(derived.K_ss) is non-positive."
        return false
    end

    # Check grid bounds make sense
    K_min = p.numerical.K_min_factor * derived.K_ss
    K_max = p.numerical.K_max_factor * derived.K_ss

    if K_min >= K_max
        @warn "Invalid capital grid bounds: K_min = $K_min >= K_max = $K_max"
        return false
    end

    return true
end

"""
    print_parameters(p::ModelParameters)

Print a formatted summary of model parameters.
"""
function print_parameters(p::ModelParameters)
    derived = get_derived_parameters(p)

    println("=" ^ 60)
    println("Model Parameters")
    println("=" ^ 60)
    println("\nTechnology:")
    println("  alpha (capital share)      = $(p.alpha)")
    println("  epsilon (demand elasticity)  = $(p.epsilon)")
    println("  delta (annual depreciation) = $(p.delta)")
    println("  beta (annual discount)    = $(p.beta)")

    println("\nDemand Process (semester frequency):")
    space_label_D = p.demand.process_space == :log ? "log demand" : "demand level"
    println("  mu_D (mean $space_label_D)  = $(p.demand.mu_D)")
    println("  rho_D (persistence)      = $(p.demand.rho_D)")
    println("  process_space            = $(p.demand.process_space)")

    println("\nVolatility Process (semester frequency):")
    if p.volatility isa VolatilityProcess
        space_label_vol = p.volatility.process_space == :log ? "log vol" : "vol level"
        println("  Type: Continuous AR(1)")
        println("  sigma_bar (mean $space_label_vol)       = $(p.volatility.sigma_bar)")
        println("  rho_sigma (persistence)      = $(p.volatility.rho_sigma)")
        println("  sigma_eta (vol of vol)       = $(p.volatility.sigma_eta)")
        println("  rho_epsilon_eta (correlation)     = $(p.volatility.rho_epsilon_eta)")
        println("  process_space                = $(p.volatility.process_space)")
    elseif p.volatility isa TwoStateVolatility
        space_label_vol = p.volatility.process_space == :log ? "log" : "level"
        println("  Type: Two-state Markov switching")
        println("  sigma_levels ($space_label_vol)  = $(p.volatility.sigma_levels)")
        println("  Pi_sigma                     = $(p.volatility.Pi_sigma)")
        println("  process_space                = $(p.volatility.process_space)")
    end

    println("\nDerived Parameters:")
    println("  gamma (profit exponent)    = $(round(derived.gamma, digits=4))")
    println("  h (scale parameter)    = $(round(derived.h, digits=4))")
    println("  delta_semester             = $(round(derived.delta_semester, digits=4))")
    println("  beta_semester             = $(round(derived.beta_semester, digits=4))")
    println("  K_ss (steady state)    = $(round(derived.K_ss, digits=4))")

    println("\nNumerical Settings:")
    println("  n_K x n_D x n_sigma        = $(p.numerical.n_K) x $(p.numerical.n_D) x $(p.numerical.n_sigma)")
    println("  K_min factor           = $(p.numerical.K_min_factor)")
    println("  K_max factor           = $(p.numerical.K_max_factor)")
    println("  VFI tolerance          = $(p.numerical.tol_vfi)")
    println("  Max iterations         = $(p.numerical.max_iter)")
    println("  Howard steps           = $(p.numerical.howard_steps)")
    println("=" ^ 60)
end
