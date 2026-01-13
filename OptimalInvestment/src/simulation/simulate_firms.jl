"""
Simulate firm decisions using solved policy functions.
"""

"""
    FirmHistory

Container for simulated firm history.
"""
struct FirmHistory
    T::Int                          # Number of years
    K::Vector{Float64}              # Capital stock
    D::Vector{Float64}              # Demand (first semester)
    D_half::Vector{Float64}         # Demand (second semester)
    σ::Vector{Float64}              # Volatility (first semester)
    σ_half::Vector{Float64}         # Volatility (second semester)
    I::Vector{Float64}              # Initial investment
    ΔI::Vector{Float64}             # Investment revision
    I_total::Vector{Float64}        # Total investment (I + ΔI)
    profit::Vector{Float64}         # Annual profit
end

"""
    simulate_firm(sol::SolvedModel, D_path::Vector{Float64}, σ_path::Vector{Float64},
                  K_init::Float64; T_years::Int) -> FirmHistory

Simulate single firm given shock paths.

# Arguments
- `sol`: SolvedModel object
- `D_path`: Log demand path (semesters)
- `σ_path`: Log volatility path (semesters)
- `K_init`: Initial capital stock
- `T_years`: Number of years to simulate

# Returns
- FirmHistory object
"""
function simulate_firm(sol::SolvedModel, D_path::Vector{Float64}, σ_path::Vector{Float64},
                      K_init::Float64; T_years::Int)
    @assert length(D_path) >= 2 * T_years "Need at least 2*T_years semesters"
    @assert length(σ_path) >= 2 * T_years "Need at least 2*T_years semesters"

    derived = get_derived_parameters(sol.params)
    grids = sol.grids

    # Allocate storage
    K = zeros(T_years + 1)
    D_first = zeros(T_years)
    D_second = zeros(T_years)
    σ_first = zeros(T_years)
    σ_second = zeros(T_years)
    I_initial = zeros(T_years)
    ΔI = zeros(T_years)
    I_tot = zeros(T_years)
    profits = zeros(T_years)

    # Initial capital
    K[1] = K_init

    # Simulate year by year
    for year in 1:T_years
        # Semester indices
        sem1 = 2 * (year - 1) + 1
        sem2 = 2 * year

        # Current state
        K_current = K[year]
        log_D = D_path[sem1]
        log_σ = σ_path[sem1]
        D_level = exp(log_D)
        σ_level = exp(log_σ)

        # Store first semester states
        D_first[year] = D_level
        σ_first[year] = σ_level

        # Find nearest grid points for (D, σ)
        i_D = argmin(abs.(grids.sv.D_grid .- log_D))
        i_σ = argmin(abs.(grids.sv.σ_grid .- log_σ))

        # Interpolate policy function for initial investment
        I = interpolate_policy(grids, sol.I_policy, K_current, i_D, i_σ)
        I_initial[year] = I

        # Capital after initial investment (before revision)
        K_prime = (1 - derived.δ_semester) * K_current + I

        # Mid-year shocks
        log_D_half = D_path[sem2]
        log_σ_half = σ_path[sem2]
        D_half_level = exp(log_D_half)
        σ_half_level = exp(log_σ_half)

        # Store second semester states
        D_second[year] = D_half_level
        σ_second[year] = σ_half_level

        # Find nearest grid points for mid-year states
        i_D_half = argmin(abs.(grids.sv.D_grid .- log_D_half))
        i_σ_half = argmin(abs.(grids.sv.σ_grid .- log_σ_half))

        # Solve mid-year problem for investment revision
        # This requires solving the optimization problem
        ΔI_opt, _ = solve_midyear_problem(
            K_prime, i_D_half, i_σ_half, K_current, I,
            sol.V, grids, sol.params, sol.ac, derived
        )
        ΔI[year] = ΔI_opt

        # Total investment
        I_tot[year] = I + ΔI_opt

        # Next period capital
        K[year + 1] = K_prime + ΔI_opt

        # Annual profit
        π1 = profit(K_current, D_level, derived)
        π2 = profit(K_current, D_half_level, derived)
        profits[year] = π1 + π2
    end

    return FirmHistory(T_years, K[1:end-1], D_first, D_second, σ_first, σ_second,
                      I_initial, ΔI, I_tot, profits)
end

"""
    simulate_firm_panel(sol::SolvedModel, shocks::ShockPanel;
                       K_init::Float64=1.0, T_years::Int=50) -> Vector{FirmHistory}

Simulate panel of firms using shock panel.

# Arguments
- `sol`: SolvedModel object
- `shocks`: ShockPanel object
- `K_init`: Initial capital for all firms (or can be randomized)
- `T_years`: Number of years to simulate per firm

# Returns
- Vector of FirmHistory objects
"""
function simulate_firm_panel(sol::SolvedModel, shocks::ShockPanel;
                            K_init::Float64=1.0, T_years::Int=50)
    @assert shocks.T >= 2 * T_years "Shock panel too short for requested simulation length"

    histories = Vector{FirmHistory}(undef, shocks.n_firms)

    for i in 1:shocks.n_firms
        D_path = shocks.D[i, :]
        σ_path = shocks.σ[i, :]

        histories[i] = simulate_firm(sol, D_path, σ_path, K_init; T_years=T_years)
    end

    return histories
end
