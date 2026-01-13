"""
    DemandProcess

Autoregressive demand process parameters at semester frequency.
Log demand evolves as: log D_{s+1/2} = μ_D(1-ρ_D) + ρ_D log D_s + σ_s ε_{s+1/2}
"""
@with_kw struct DemandProcess
    μ_D::Float64 = 0.0       # Long-run mean of log demand
    ρ_D::Float64 = 0.9       # Persistence (semester)

    function DemandProcess(μ_D, ρ_D)
        @assert 0.0 <= ρ_D < 1.0 "ρ_D must be in [0, 1)"
        new(μ_D, ρ_D)
    end
end

"""
    VolatilityProcess

Stochastic volatility process parameters at semester frequency.
Log volatility evolves as: log σ_{s+1/2} = σ̄(1-ρ_σ) + ρ_σ log σ_s + σ_η η_{s+1/2}
"""
@with_kw struct VolatilityProcess
    σ̄::Float64 = log(0.1)    # Long-run mean of log volatility
    ρ_σ::Float64 = 0.95      # Persistence (semester)
    σ_η::Float64 = 0.1       # Volatility of volatility
    ρ_εη::Float64 = 0.0      # Correlation between demand and volatility shocks

    function VolatilityProcess(σ̄, ρ_σ, σ_η, ρ_εη)
        @assert 0.0 <= ρ_σ < 1.0 "ρ_σ must be in [0, 1)"
        @assert σ_η > 0.0 "σ_η must be positive"
        @assert -1.0 <= ρ_εη <= 1.0 "ρ_εη must be in [-1, 1]"
        new(σ̄, ρ_σ, σ_η, ρ_εη)
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
    n_σ::Int = 7             # Volatility states

    # Grid bounds (relative to steady state)
    K_min_factor::Float64 = 0.1   # K_min = K_ss * factor
    K_max_factor::Float64 = 3.0   # K_max = K_ss * factor

    # Convergence tolerances
    tol_vfi::Float64 = 1e-6       # Value function tolerance
    tol_policy::Float64 = 1e-6    # Policy function tolerance
    max_iter::Int = 1000          # Maximum VFI iterations

    # Acceleration
    howard_steps::Int = 0         # Howard improvement steps (0 = disabled)

    # Interpolation
    interp_method::Symbol = :linear  # :linear or :cubic

    function NumericalSettings(n_K, n_D, n_σ, K_min_factor, K_max_factor,
                               tol_vfi, tol_policy, max_iter, howard_steps, interp_method)
        @assert n_K > 0 "n_K must be positive"
        @assert n_D > 0 "n_D must be positive"
        @assert n_σ > 0 "n_σ must be positive"
        @assert 0.0 < K_min_factor < K_max_factor "Invalid K bounds"
        @assert tol_vfi > 0.0 "tol_vfi must be positive"
        @assert tol_policy > 0.0 "tol_policy must be positive"
        @assert max_iter > 0 "max_iter must be positive"
        @assert howard_steps >= 0 "howard_steps must be non-negative"
        @assert interp_method in [:linear, :cubic] "Invalid interpolation method"
        new(n_K, n_D, n_σ, K_min_factor, K_max_factor, tol_vfi, tol_policy,
            max_iter, howard_steps, interp_method)
    end
end

"""
    ModelParameters

Main parameter structure containing all model primitives.
"""
@with_kw struct ModelParameters
    # Technology
    α::Float64 = 0.33        # Capital share in production
    ε::Float64 = 4.0         # Demand elasticity (must be > 1)
    δ::Float64 = 0.10        # Annual depreciation rate

    # Preferences
    β::Float64 = 0.96        # Annual discount factor

    # Stochastic processes
    demand::DemandProcess = DemandProcess()
    volatility::VolatilityProcess = VolatilityProcess()

    # Numerical settings
    numerical::NumericalSettings = NumericalSettings()

    function ModelParameters(α, ε, δ, β, demand, volatility, numerical)
        @assert 0.0 < α < 1.0 "α must be in (0, 1)"
        @assert ε > 1.0 "ε must be > 1 for positive profits"
        @assert 0.0 < δ < 1.0 "δ must be in (0, 1)"
        @assert 0.0 < β < 1.0 "β must be in (0, 1)"
        new(α, ε, δ, β, demand, volatility, numerical)
    end
end

"""
    DerivedParameters

Derived parameters computed from ModelParameters.
These are calculated once and reused throughout the solution.
"""
struct DerivedParameters
    γ::Float64               # Profit function demand exponent
    h::Float64               # Profit function scale parameter
    δ_semester::Float64      # Semester depreciation rate
    β_semester::Float64      # Semester discount factor
    K_ss::Float64            # Steady-state capital (no uncertainty)
end

"""
    get_derived_parameters(p::ModelParameters) -> DerivedParameters

Compute derived parameters from primitive parameters.

# Formulas
- γ = (ε - 1) / (ε - (1 - α))
- h = α * (1 - 1/ε)^(ε/α) * (1 - α)^(ε/α - 1)
- δ_semester = 1 - (1 - δ)^(1/2)
- β_semester = β^(1/2)
- K_ss computed from first-order condition: MPK = δ/β
"""
function get_derived_parameters(p::ModelParameters)
    # Profit function exponents from iso-elastic demand and Cobb-Douglas technology
    γ = (p.ε - 1) / (p.ε - (1 - p.α))

    # Scale parameter h
    term1 = p.α
    term2 = (1 - 1/p.ε)^(p.ε / p.α)
    term3 = (1 - p.α)^(p.ε / p.α - 1)
    h = term1 * term2 * term3

    # Convert annual rates to semester rates
    δ_semester = 1 - (1 - p.δ)^(1/2)
    β_semester = p.β^(1/2)

    # Steady-state capital (deterministic case)
    # From FOC: MPK = δ/β => (1-γ) * h * D_ss^γ * K_ss^(-γ) = δ/β
    # With D_ss = exp(μ_D), solve for K_ss
    D_ss = exp(p.demand.μ_D)
    user_cost = p.δ / p.β

    # MPK = (1 - γ) * h * D^γ * K^(-γ)
    # K_ss = [(1 - γ) * h * D_ss^γ / user_cost]^(1/γ)
    K_ss = ((1 - γ) * h * D_ss^γ / user_cost)^(1/γ)

    return DerivedParameters(γ, h, δ_semester, β_semester, K_ss)
end

"""
    validate_parameters(p::ModelParameters) -> Bool

Perform additional validation checks on parameter combinations.
"""
function validate_parameters(p::ModelParameters)
    # Check that profit function exponent is well-defined
    derived = get_derived_parameters(p)

    if derived.γ <= 0.0 || derived.γ >= 1.0
        @warn "Profit function exponent γ = $(derived.γ) is outside (0,1). This may cause numerical issues."
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
    println("  α (capital share)      = $(p.α)")
    println("  ε (demand elasticity)  = $(p.ε)")
    println("  δ (annual depreciation) = $(p.δ)")
    println("  β (annual discount)    = $(p.β)")

    println("\nDemand Process (semester frequency):")
    println("  μ_D (mean log demand)  = $(p.demand.μ_D)")
    println("  ρ_D (persistence)      = $(p.demand.ρ_D)")

    println("\nVolatility Process (semester frequency):")
    println("  σ̄ (mean log vol)       = $(p.volatility.σ̄)")
    println("  ρ_σ (persistence)      = $(p.volatility.ρ_σ)")
    println("  σ_η (vol of vol)       = $(p.volatility.σ_η)")
    println("  ρ_εη (correlation)     = $(p.volatility.ρ_εη)")

    println("\nDerived Parameters:")
    println("  γ (profit exponent)    = $(round(derived.γ, digits=4))")
    println("  h (scale parameter)    = $(round(derived.h, digits=4))")
    println("  δ_semester             = $(round(derived.δ_semester, digits=4))")
    println("  β_semester             = $(round(derived.β_semester, digits=4))")
    println("  K_ss (steady state)    = $(round(derived.K_ss, digits=4))")

    println("\nNumerical Settings:")
    println("  n_K × n_D × n_σ        = $(p.numerical.n_K) × $(p.numerical.n_D) × $(p.numerical.n_σ)")
    println("  K_min factor           = $(p.numerical.K_min_factor)")
    println("  K_max factor           = $(p.numerical.K_max_factor)")
    println("  VFI tolerance          = $(p.numerical.tol_vfi)")
    println("  Max iterations         = $(p.numerical.max_iter)")
    println("  Howard steps           = $(p.numerical.howard_steps)")
    println("=" ^ 60)
end
