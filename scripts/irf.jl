# Impulse Response Function tests with flexible shock/volatility specifications
using Pkg
project_root = dirname(@__DIR__)
Pkg.activate(project_root)
using UncertaintyInvestment
using Printf
using NPZ
using DataFrames
using Parquet2
using Random


project_root = dirname(@__DIR__)
outdir = joinpath(project_root, "output", "simulations_irf")
mkpath(outdir)

Random.seed!(12345)

# ============================================================================
# Configuration: choose demand space, volatility type, and cost structure
# ============================================================================

# --- Demand process ---
demand_space = :log

demand = DemandProcess(
    mu_D = demand_space == :log ? log(500) : 500.0,
    rho_D = 0.5,
    process_space = demand_space
)

# --- Volatility process ---
volatility_type = :two_state

# For :continuous — AR(1) in chosen space
vol_space = :log

volatility_continuous = VolatilityProcess(
    sigma_bar = vol_space == :log ? log(0.1) : 0.1,
    rho_sigma = 0.1,
    sigma_eta = 0.1,
    rho_epsilon_eta = 0.0,
    process_space = vol_space
)

# For :two_state — Markov switching
vol_two_state_space = :level

volatility_two_state = TwoStateVolatility(
    sigma_levels = [0.05, 0.20],
    Pi_sigma = [0.95 0.05; 0.10 0.90],
    process_space = vol_two_state_space
)

volatility = volatility_type == :continuous ? volatility_continuous : volatility_two_state

# --- Numerical settings ---
n_sigma_grid = volatility_type == :two_state ? 2 : 5

numerical = NumericalSettings(
    n_K = 50,
    n_D = 50,
    n_sigma = n_sigma_grid,
    K_min_factor = 0.1,
    K_max_factor = 10.0,
    tol_vfi = 1e-4,
    max_iter = 200,
    howard_steps = 0
)

# --- Model parameters ---
params = ModelParameters(
    alpha = 0.33,
    epsilon = 4.0,
    delta = 0.10,
    beta = 0.96,
    demand = demand,
    volatility = volatility,
    numerical = numerical
)

# --- Adjustment costs ---
ac_begin = FixedAdjustmentCost(F = 1.0)
ac_mid_year = FixedAdjustmentCost(F = 100.0)

# --- IRF settings ---
n_firms      = 100
T_years      = 500
T_semesters  = 2 * T_years
T_shock_year = 300      # shock hits at the beginning of year 50
seed         = 12345

# ============================================================================
# Print configuration summary
# ============================================================================

println("="^70)
println("IRF Test Configuration")
println("="^70)
println("Demand: process_space = :$(demand.process_space), mu_D = $(demand.mu_D), rho_D = $(demand.rho_D)")
if volatility isa VolatilityProcess
    println("Volatility: continuous AR(1), process_space = :$(volatility.process_space)")
    println("  sigma_bar = $(volatility.sigma_bar), rho_sigma = $(volatility.rho_sigma), sigma_eta = $(volatility.sigma_eta)")
elseif volatility isa TwoStateVolatility
    println("Volatility: two-state Markov, process_space = :$(volatility.process_space)")
    println("  sigma_levels = $(volatility.sigma_levels)")
    println("  Pi_sigma = $(volatility.Pi_sigma)")
end
println("Grid: n_K=$(numerical.n_K), n_D=$(numerical.n_D), n_sigma=$(numerical.n_sigma)")
println("IRF: shock at beginning of year $T_shock_year, $n_firms firms, $T_years years")
println("="^70)

# ============================================================================
# Solve the model
# ============================================================================

grids = construct_grids(params)
NPZ.npzwrite(joinpath(outdir, "grid_K.npy"), grids.K_grid)
NPZ.npzwrite(joinpath(outdir, "grid_D.npy"), grids.sv.D_grid)
NPZ.npzwrite(joinpath(outdir, "grid_sigma.npy"), grids.sv.sigma_grid)

sol = solve_model(params;
    ac_begin = ac_begin,
    ac_mid_year = ac_mid_year,
    verbose = true,
    use_parallel = true,
    use_multiscale = true
)

NPZ.npzwrite(joinpath(outdir, "V.npy"), sol.V)
NPZ.npzwrite(joinpath(outdir, "V_stage1.npy"), sol.V_stage1)
NPZ.npzwrite(joinpath(outdir, "I_policy.npy"), sol.I_policy)
NPZ.npzwrite(joinpath(outdir, "Delta_I_policy.npy"), sol.Delta_I_policy)

# ============================================================================
# Generate IRF shock panels (Bloom 2009 protocol)
# ============================================================================

shock_sem = year_to_semester(T_shock_year; stage = :begin)

println("\nGenerating IRF panels (Bloom 2009 protocol)...")
println("  Shock semester: $shock_sem (beginning of year $T_shock_year)")

if volatility isa TwoStateVolatility
    println("  Forcing σ to σ_high = $(volatility.sigma_levels[2]) at shock date")
    panels = generate_irf_panels(
        params.demand,
        params.volatility,
        n_firms,
        T_semesters;
        shock_semester = shock_sem,
        seed = seed,
        burn_in = 100
    )
elseif volatility isa VolatilityProcess
    # For continuous AR(1): shock to 2× the long-run mean level
    if vol_space == :log
        sigma_shock = log(2.0 * exp(volatility.sigma_bar))
    else
        sigma_shock = 2.0 * volatility.sigma_bar
    end
    println("  Forcing σ to $(sigma_shock) ($vol_space space) at shock date")
    panels = generate_irf_panels(
        params.demand,
        params.volatility,
        n_firms,
        T_semesters;
        shock_semester = shock_sem,
        sigma_shock_value = sigma_shock,
        seed = seed,
        burn_in = 100
    )
end

# Sanity checks
ctrl_sig = panels.control.sigma
trt_sig  = panels.treatment.sigma

# Before shock: paths should be identical
@assert ctrl_sig[:, 1:shock_sem-1] == trt_sig[:, 1:shock_sem-1] "Paths should match before shock"
println("  ✓ Control and treatment paths identical before shock")

# At shock: all treatment firms at σ_high
if volatility isa TwoStateVolatility
    @assert all(trt_sig[:, shock_sem] .== volatility.sigma_levels[2]) "Treatment σ should be σ_high at shock"
end
println("  ✓ Treatment σ forced at shock semester")

# After shock: paths diverge
n_diverged = sum(ctrl_sig[:, shock_sem] .!= trt_sig[:, shock_sem])
println("  ✓ $n_diverged / $n_firms firms have different σ at shock date (rest were already at σ_high)")

# ============================================================================
# Simulate firms under both scenarios
# ============================================================================

println("\nSimulating control panel ($n_firms firms, $T_years years)...")
hist_ctrl = simulate_firm_panel(sol, panels.control;
    K_init = nothing, T_years = T_years, use_parallel = true)
panel_ctrl = construct_estimation_panel(hist_ctrl)

println("Simulating treatment panel ($n_firms firms, $T_years years)...")
hist_trt = simulate_firm_panel(sol, panels.treatment;
    K_init = nothing, T_years = T_years, use_parallel = true)
panel_trt = construct_estimation_panel(hist_trt)

# ============================================================================
# Save everything
# ============================================================================

save_simulation(joinpath(outdir, "panel_control.parquet"), panel_ctrl)
save_simulation(joinpath(outdir, "panel_treatment.parquet"), panel_trt)

# Save raw σ paths for plotting the IRF on σ itself
NPZ.npzwrite(joinpath(outdir, "sigma_control.npy"),  panels.control.sigma_level)
NPZ.npzwrite(joinpath(outdir, "sigma_treatment.npy"), panels.treatment.sigma_level)
NPZ.npzwrite(joinpath(outdir, "D_control.npy"),  panels.control.D_level)
NPZ.npzwrite(joinpath(outdir, "D_treatment.npy"), panels.treatment.D_level)

println("\n" * "="^70)
println("IRF outputs saved to: $outdir")
println("  panel_control.parquet    — firm panel under baseline shocks")
println("  panel_treatment.parquet  — firm panel with σ shock at year $T_shock_year")
println("  sigma_control.npy        — σ_level paths, shape (n_firms, T_semesters)")
println("  sigma_treatment.npy      — σ_level paths with impulse")
println("  D_control.npy            — D_level paths (control)")
println("  D_treatment.npy          — D_level paths (treatment)")
println("="^70)
println("\nTo compute the IRF:")
println("  IRF(t) = mean(Y_treatment(t)) - mean(Y_control(t))")
println("  for Y ∈ {I_total, I_rate, K, profit, ...}")