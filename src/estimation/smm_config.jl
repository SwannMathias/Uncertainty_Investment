"""
Configuration structs for Simulated Method of Moments (SMM) estimation.

# Economic context
The estimation targets 4 structural parameters governing adjustment costs
in a two-stage investment model: fixed and convex costs at both the
beginning-of-year and mid-year stages.

# Parameters estimated
- `F_begin`: Fixed adjustment cost, beginning of year (Stage 0)
- `F_mid`: Fixed adjustment cost, mid-year (Stage 1)
- `phi_begin`: Convex adjustment cost, beginning of year (Stage 0)
- `phi_mid`: Convex adjustment cost, mid-year (Stage 1)
"""

"""
    RevisionTransform

Transform applied to investment expectations before computing revisions.

- `LOG_TRANSFORM`: log(E_new) - log(E_old). Drops observations where either E <= 0.
- `LEVEL_OVER_K_TRANSFORM`: (E_new - E_old) / K. No domain restriction.
- `ASINH_TRANSFORM`: asinh(E_new) - asinh(E_old). No domain restriction. Default.
"""
@enum RevisionTransform LOG_TRANSFORM LEVEL_OVER_K_TRANSFORM ASINH_TRANSFORM

"""
    FixedCalibration

Non-estimated model parameters held constant during SMM estimation.

# Fields
- Economic parameters: `alpha`, `epsilon`, `delta`, `beta`
- Demand process: `rho_D`, `mu_D`
- Volatility process: `sigma_bar`, `rho_sigma`, `sigma_eta`, `rho_epsilon_eta`
- Numerical settings for VFI: `n_K`, `n_D`, `n_sigma`, grid bounds, tolerances
"""
struct FixedCalibration
    # Economic parameters
    alpha::Float64
    epsilon::Float64
    delta::Float64
    beta::Float64

    # Demand process (semester frequency)
    rho_D::Float64
    mu_D::Float64

    # Volatility process (semester frequency)
    sigma_bar::Float64
    rho_sigma::Float64
    sigma_eta::Float64
    rho_epsilon_eta::Float64

    # Numerical settings for VFI
    n_K::Int
    n_D::Int
    n_sigma::Int
    K_min_factor::Float64
    K_max_factor::Float64
    tol_vfi::Float64
    max_iter::Int
    howard_steps::Int
end

"""
    FixedCalibration(; kwargs...)

Construct FixedCalibration with sensible defaults matching project conventions.
"""
function FixedCalibration(;
    alpha::Float64 = 0.33,
    epsilon::Float64 = 4.0,
    delta::Float64 = 0.10,
    beta::Float64 = 0.96,
    rho_D::Float64 = 0.5,
    mu_D::Float64 = 0.0,
    sigma_bar::Float64 = log(0.1),
    rho_sigma::Float64 = 0.1,
    sigma_eta::Float64 = 0.1,
    rho_epsilon_eta::Float64 = 0.0,
    n_K::Int = 50,
    n_D::Int = 15,
    n_sigma::Int = 7,
    K_min_factor::Float64 = 0.1,
    K_max_factor::Float64 = 3.0,
    tol_vfi::Float64 = 1e-6,
    max_iter::Int = 1000,
    howard_steps::Int = 50
)
    return FixedCalibration(
        alpha, epsilon, delta, beta,
        rho_D, mu_D,
        sigma_bar, rho_sigma, sigma_eta, rho_epsilon_eta,
        n_K, n_D, n_sigma, K_min_factor, K_max_factor,
        tol_vfi, max_iter, howard_steps
    )
end

"""
    build_model_parameters(cal::FixedCalibration) -> ModelParameters

Construct ModelParameters from a FixedCalibration struct.
"""
function build_model_parameters(cal::FixedCalibration)
    return ModelParameters(
        alpha = cal.alpha,
        epsilon = cal.epsilon,
        delta = cal.delta,
        beta = cal.beta,
        demand = DemandProcess(
            mu_D = cal.mu_D,
            rho_D = cal.rho_D
        ),
        volatility = VolatilityProcess(
            sigma_bar = cal.sigma_bar,
            rho_sigma = cal.rho_sigma,
            sigma_eta = cal.sigma_eta,
            rho_epsilon_eta = cal.rho_epsilon_eta
        ),
        numerical = NumericalSettings(
            n_K = cal.n_K,
            n_D = cal.n_D,
            n_sigma = cal.n_sigma,
            K_min_factor = cal.K_min_factor,
            K_max_factor = cal.K_max_factor,
            tol_vfi = cal.tol_vfi,
            max_iter = cal.max_iter,
            howard_steps = cal.howard_steps
        )
    )
end

"""
    SMMConfig

Configuration for SMM estimation.

# Fields
- `calibration`: Fixed (non-estimated) model parameters
- `lower_bounds`: Lower bounds for [F_begin, F_mid, phi_begin, phi_mid]
- `upper_bounds`: Upper bounds for [F_begin, F_mid, phi_begin, phi_mid]
- `m_data`: Empirical moment targets [share_zero_begin, share_zero_mid, coef_begin, coef_mid]
- `W`: Weighting matrix (4x4)
- `n_firms`: Number of firms to simulate
- `T_years`: Years per firm (post burn-in)
- `burn_in_years`: Years to discard from beginning of simulation
- `shock_seed`: Seed for shock generation (ensures identical shocks across evaluations)
- `revision_transform`: Transform applied before computing revisions
- `zero_threshold`: Threshold for share-of-zero moments
"""
struct SMMConfig
    # Fixed parameters
    calibration::FixedCalibration

    # Estimation bounds: [F_begin, F_mid, phi_begin, phi_mid]
    lower_bounds::Vector{Float64}   # length 4
    upper_bounds::Vector{Float64}   # length 4

    # Empirical targets: [share_zero_begin, share_zero_mid, coef_begin, coef_mid]
    m_data::Vector{Float64}         # length 4

    # Weighting matrix (4x4)
    W::Matrix{Float64}

    # Simulation settings
    n_firms::Int                    # Number of firms to simulate
    T_years::Int                    # Years per firm (post burn-in)
    burn_in_years::Int              # Years to discard
    shock_seed::Int                 # Seed for shock generation

    # Revision transform
    revision_transform::RevisionTransform

    # Zero threshold for share-of-zero moments
    zero_threshold::Float64

    function SMMConfig(calibration, lower_bounds, upper_bounds, m_data, W,
                       n_firms, T_years, burn_in_years, shock_seed,
                       revision_transform, zero_threshold)
        @assert length(lower_bounds) == 4 "lower_bounds must have length 4"
        @assert length(upper_bounds) == 4 "upper_bounds must have length 4"
        @assert all(lower_bounds .<= upper_bounds) "lower_bounds must be <= upper_bounds"
        @assert length(m_data) == 4 "m_data must have length 4"
        @assert size(W) == (4, 4) "W must be 4x4"
        @assert n_firms > 0 "n_firms must be positive"
        @assert T_years > 0 "T_years must be positive"
        @assert burn_in_years >= 0 "burn_in_years must be non-negative"
        @assert zero_threshold > 0.0 "zero_threshold must be positive"
        new(calibration, lower_bounds, upper_bounds, m_data, W,
            n_firms, T_years, burn_in_years, shock_seed,
            revision_transform, zero_threshold)
    end
end

"""
    SMMConfig(; kwargs...)

Construct SMMConfig with defaults.
"""
function SMMConfig(;
    calibration::FixedCalibration = FixedCalibration(),
    lower_bounds::Vector{Float64} = [0.0, 0.0, 0.0, 0.0],
    upper_bounds::Vector{Float64} = [10.0, 10.0, 20.0, 20.0],
    m_data::Vector{Float64} = [0.35, 0.50, -0.15, 0.10],
    W::Matrix{Float64} = Matrix{Float64}(I, 4, 4),
    n_firms::Int = 1000,
    T_years::Int = 50,
    burn_in_years::Int = 30,
    shock_seed::Int = 42,
    revision_transform::RevisionTransform = ASINH_TRANSFORM,
    zero_threshold::Float64 = 1e-4
)
    return SMMConfig(calibration, lower_bounds, upper_bounds, m_data, W,
                     n_firms, T_years, burn_in_years, shock_seed,
                     revision_transform, zero_threshold)
end
