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

# ============================================================================
# Global toggles
# ============================================================================

# Set to true to compute conditional expectation columns (E_last_semester,
# E_beginning, E_half). These are slow due to O(n_states^2) loops per firm-year.
COMPUTE_PLANS = false

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
    sigma_levels = [0.05, 1],
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
ac_begin = CompositeAdjustmentCost(FixedAdjustmentCost(F=1.0), ConvexAdjustmentCost(phi=1))
ac_mid_year = FixedAdjustmentCost(F = 100.0)

# --- IRF settings ---
n_firms      = 1000
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
# Generate IRF shock panels (Bloom 2009 protocol) for all treatments
# ============================================================================

shock_sem = year_to_semester(T_shock_year; stage = :begin)

println("\nGenerating IRF panels (Bloom 2009 protocol)...")
println("  Shock semester: $shock_sem (beginning of year $T_shock_year)")
println("  COMPUTE_PLANS: $COMPUTE_PLANS")

# Precompute sigma_shock for continuous volatility (used across all treatments)
sigma_shock = NaN
if volatility isa VolatilityProcess
    if vol_space == :log
        sigma_shock = log(2.0 * exp(volatility.sigma_bar))
    else
        sigma_shock = 2.0 * volatility.sigma_bar
    end
end

# --- Loop over mean-preserving correction treatments ---
treatments = [:none, :static, :dynamic]
all_panels = DataFrame[]

for trt in treatments
    println("\n" * "-"^70)
    println("Treatment: $trt")
    println("-"^70)

    if volatility isa TwoStateVolatility
        println("  Forcing σ to σ_high = $(volatility.sigma_levels[2]) at shock date")
        panels = generate_irf_panels(
            params.demand,
            params.volatility,
            n_firms,
            T_semesters;
            shock_semester = shock_sem,
            seed = seed,
            burn_in = 100,
            mean_preserving = trt
        )
    elseif volatility isa VolatilityProcess
        println("  Forcing σ to $(sigma_shock) ($vol_space space) at shock date")
        panels = generate_irf_panels(
            params.demand,
            params.volatility,
            n_firms,
            T_semesters;
            shock_semester = shock_sem,
            sigma_shock_value = sigma_shock,
            seed = seed,
            burn_in = 100,
            mean_preserving = trt
        )
    end

    # Sanity checks
    ctrl_sig = panels.control.sigma
    trt_sig  = panels.treatment.sigma

    @assert ctrl_sig[:, 1:shock_sem-1] == trt_sig[:, 1:shock_sem-1] "Paths should match before shock"
    println("  Control and treatment paths identical before shock")

    if volatility isa TwoStateVolatility
        @assert all(trt_sig[:, shock_sem] .== volatility.sigma_levels[2]) "Treatment σ should be σ_high at shock"
    end
    println("  Treatment σ forced at shock semester")

    n_diverged = sum(ctrl_sig[:, shock_sem] .!= trt_sig[:, shock_sem])
    println("  $n_diverged / $n_firms firms have different σ at shock date")

    # Simulate control and treatment panels
    println("  Simulating control panel...")
    hist_ctrl = simulate_firm_panel(sol, panels.control;
        K_init = nothing, T_years = T_years, use_parallel = true, compute_plans = COMPUTE_PLANS)
    panel_ctrl = construct_estimation_panel(hist_ctrl)
    panel_ctrl.df[!, :treatment] .= String(trt)
    panel_ctrl.df[!, :group] .= "control"

    println("  Simulating treatment panel...")
    hist_trt = simulate_firm_panel(sol, panels.treatment;
        K_init = nothing, T_years = T_years, use_parallel = true, compute_plans = COMPUTE_PLANS)
    panel_trt = construct_estimation_panel(hist_trt)
    panel_trt.df[!, :treatment] .= String(trt)
    panel_trt.df[!, :group] .= "treatment"

    push!(all_panels, panel_ctrl.df)
    push!(all_panels, panel_trt.df)

    # Save raw σ and D paths per treatment
    NPZ.npzwrite(joinpath(outdir, "sigma_control_$(trt).npy"),   panels.control.sigma_level)
    NPZ.npzwrite(joinpath(outdir, "sigma_treatment_$(trt).npy"), panels.treatment.sigma_level)
    NPZ.npzwrite(joinpath(outdir, "D_control_$(trt).npy"),       panels.control.D_level)
    NPZ.npzwrite(joinpath(outdir, "D_treatment_$(trt).npy"),     panels.treatment.D_level)
end

# ============================================================================
# Stack and save combined panel
# ============================================================================

combined_df = vcat(all_panels...)
combined_panel = FirmPanel(combined_df, n_firms, T_years)
save_simulation(joinpath(outdir, "panel.parquet"), combined_panel)

println("\n" * "="^70)
println("IRF outputs saved to: $outdir")
println("  panel_combined.parquet   — all treatments (none/static/dynamic) × control/treatment")
println("  sigma_*_<trt>.npy        — σ_level paths per treatment, shape (n_firms, T_semesters)")
println("  D_*_<trt>.npy            — D_level paths per treatment")
println("="^70)
println("\nTo compute the IRF:")
println("  IRF(t) = mean(Y_treatment(t)) - mean(Y_control(t))")
println("  for Y ∈ {I_total, I_rate, K, profit, ...}")
println("  Filter by 'treatment' column for specific correction type")