"""
    irf_shocks.jl

Generate matched (control, treatment) shock panels for impulse response analysis,
following Bloom (2009, Econometrica) Section 4.

Methodology:
    1. Pre-draw all standardised innovations (shared between control and treatment)
    2. Let firms evolve freely during burn-in, reaching the ergodic distribution
       over (K, D, σ) — including the possibility that some firms are already at σ_H
    3. At `shock_semester`:
       - Control: σ evolves normally (no intervention)
       - Treatment: σ is forced to σ_high
    4. After the shock, both paths evolve using the same draws but starting from
       different σ states, so paths diverge and gradually reconverge

Key feature (Bloom 2009, p. 651): "the rise [in average σ] is less than 100%
because some of the 25,000 simulations already had σ_t = σ_H when the shock
occurred." This attenuation is by design — the IRF captures the average effect
of forcing σ_H across the ergodic distribution, including firms for which the
shock is redundant.

Semester convention (matching simulate_firm):
    Beginning of year t → semester 2*(t-1) + 1
    Mid-year of year t  → semester 2*t
"""

using Random: MersenneTwister

# ============================================================================
# Two-state Markov switching volatility
# ============================================================================

"""
    generate_irf_panels(demand::DemandProcess, vol::TwoStateVolatility,
                        n_firms, T_semesters;
                        shock_semester, seed=42, burn_in=100)

Generate matched control/treatment shock panels with two-state Markov volatility.

# Protocol (Bloom 2009)
1. All firms evolve freely during burn-in, reaching the ergodic distribution.
   At the shock date, some fraction of firms will already be in σ_high — this
   is correct and expected.
2. At `shock_semester`:
   - **Control**: σ evolves normally via Markov transition (no intervention)
   - **Treatment**: σ is forced to state 2 (σ_high)
3. After `shock_semester`: both paths evolve with normal Markov transitions
   using the same uniform draws, but from different current states.

# Arguments
- `demand`: DemandProcess
- `vol`: TwoStateVolatility (sigma_levels[1] = σ_low, sigma_levels[2] = σ_high)
- `n_firms`: number of firms
- `T_semesters`: length of each path (post burn-in)
- `shock_semester`: semester index (1-based, post burn-in) at which to impose the impulse
- `seed`: master seed for reproducibility
- `burn_in`: periods to discard before the panel starts
- `mean_preserving`: if `true` and demand is in log space, apply Jensen's inequality
  correction `-σ²_t/2` to the log-demand drift each period, ensuring E[D_level] is
  unaffected by volatility changes. Default: `false`.

# Returns
NamedTuple `(control=ShockPanel, treatment=ShockPanel)`
"""
function generate_irf_panels(
    demand::DemandProcess,
    vol::TwoStateVolatility,
    n_firms::Int,
    T_semesters::Int;
    shock_semester::Int,
    seed::Int = 42,
    burn_in::Int = 100,
    mean_preserving::Bool = false
)
    @assert 1 <= shock_semester <= T_semesters "shock_semester must be in [1, $T_semesters]"

    if mean_preserving && demand.process_space != :log
        @warn "mean_preserving=true has no effect when demand.process_space = :$(demand.process_space). " *
              "Jensen's inequality correction only applies in log space."
    end

    # Volatility levels for demand innovation scaling
    if vol.process_space == :log
        sig_levels_for_D = exp.(vol.sigma_levels)
    else
        sig_levels_for_D = vol.sigma_levels
    end

    T_total = T_semesters + burn_in
    D_space     = demand.process_space
    sigma_space = vol.process_space

    D_ctrl   = zeros(n_firms, T_semesters)
    sig_ctrl = zeros(n_firms, T_semesters)
    D_trt    = zeros(n_firms, T_semesters)
    sig_trt  = zeros(n_firms, T_semesters)

    apply_jensen = mean_preserving && D_space == :log
    n_already_high = 0  # diagnostic counter

    for i in 1:n_firms
        rng = MersenneTwister(seed + i)

        # Pre-draw all randomness for the full path
        u_markov = rand(rng, T_total)       # Uniform draws for Markov transitions
        eps_D    = randn(rng, T_total)       # Standardised demand innovations

        # --- Simulate both paths ---
        Dc = zeros(T_total);  Dt = zeros(T_total)
        sc = zeros(T_total);  st = zeros(T_total)

        # Initial Markov state (draw from ergodic or just start at 1, burn-in will mix)
        state_c = 1;  state_t = 1
        sc[1] = vol.sigma_levels[state_c]
        st[1] = vol.sigma_levels[state_t]
        Dc[1] = demand.mu_D
        Dt[1] = demand.mu_D

        for t in 2:T_total
            t_post = t - burn_in   # Post burn-in index

            # ---- Volatility transitions ----
            # Control: always normal Markov transition
            state_c = u_markov[t] < vol.Pi_sigma[state_c, 1] ? 1 : 2

            # Treatment: force σ_high at shock_semester, otherwise normal
            if t_post == shock_semester
                state_t = 2   # Force to σ_high
            else
                state_t = u_markov[t] < vol.Pi_sigma[state_t, 1] ? 1 : 2
            end

            sc[t] = vol.sigma_levels[state_c]
            st[t] = vol.sigma_levels[state_t]

            # ---- Demand (innovations scale with current σ) ----
            sig_c = max(sig_levels_for_D[state_c], 1e-10)
            sig_t = max(sig_levels_for_D[state_t], 1e-10)

            # Jensen's inequality correction: -σ²/2 ensures E[exp(σε - σ²/2)] = 1
            jc_c = apply_jensen ? -0.5 * sig_c^2 : 0.0
            jc_t = apply_jensen ? -0.5 * sig_t^2 : 0.0

            Dc[t] = demand.mu_D * (1 - demand.rho_D) + demand.rho_D * Dc[t-1] + jc_c + sig_c * eps_D[t]
            Dt[t] = demand.mu_D * (1 - demand.rho_D) + demand.rho_D * Dt[t-1] + jc_t + sig_t * eps_D[t]
        end

        # Track how many firms were already at σ_high just before the shock
        shock_t_abs = burn_in + shock_semester
        if shock_t_abs >= 2
            # Check control state at shock_semester (what the firm would have been without intervention)
            prev_state = sc[shock_t_abs] == vol.sigma_levels[2]
            if prev_state
                n_already_high += 1
            end
        end

        # Store post burn-in
        D_ctrl[i, :]   = Dc[(burn_in+1):end]
        sig_ctrl[i, :] = sc[(burn_in+1):end]
        D_trt[i, :]    = Dt[(burn_in+1):end]
        sig_trt[i, :]  = st[(burn_in+1):end]
    end

    # Report diagnostic (mirrors Bloom's observation)
    frac_already = n_already_high / n_firms
    @info "IRF diagnostic: $(n_already_high)/$(n_firms) firms ($(round(frac_already*100, digits=1))%) " *
          "were already at σ_high at the shock date"

    control   = _build_shock_panel(n_firms, T_semesters, D_ctrl, sig_ctrl, D_space, sigma_space)
    treatment = _build_shock_panel(n_firms, T_semesters, D_trt,  sig_trt,  D_space, sigma_space)

    return (control = control, treatment = treatment)
end

# ============================================================================
# Continuous AR(1) volatility
# ============================================================================

"""
    generate_irf_panels(demand::DemandProcess, vol::VolatilityProcess,
                        n_firms, T_semesters;
                        shock_semester, sigma_shock_value,
                        seed=42, burn_in=100)

Generate matched control/treatment shock panels with continuous AR(1) volatility.

At `shock_semester`, treatment σ is set to `sigma_shock_value` (in native space).
Control σ evolves normally. Before the shock, both paths are identical.

# Arguments
- `sigma_shock_value`: forced σ value **in native space** (log if vol.process_space == :log).
  E.g., for a VolatilityProcess with process_space = :log and you want σ_level = 0.20,
  pass `log(0.20)`.
- `mean_preserving`: if `true` and demand is in log space, apply Jensen's inequality
  correction `-σ²_t/2` to the log-demand drift each period, ensuring E[D_level] is
  unaffected by volatility changes. Default: `false`.
"""
function generate_irf_panels(
    demand::DemandProcess,
    vol::VolatilityProcess,
    n_firms::Int,
    T_semesters::Int;
    shock_semester::Int,
    sigma_shock_value::Float64,
    seed::Int = 42,
    burn_in::Int = 100,
    mean_preserving::Bool = false
)
    @assert 1 <= shock_semester <= T_semesters "shock_semester must be in [1, $T_semesters]"

    if mean_preserving && demand.process_space != :log
        @warn "mean_preserving=true has no effect when demand.process_space = :$(demand.process_space). " *
              "Jensen's inequality correction only applies in log space."
    end

    T_total = T_semesters + burn_in
    D_space     = demand.process_space
    sigma_space = vol.process_space
    apply_jensen = mean_preserving && D_space == :log

    D_ctrl   = zeros(n_firms, T_semesters)
    sig_ctrl = zeros(n_firms, T_semesters)
    D_trt    = zeros(n_firms, T_semesters)
    sig_trt  = zeros(n_firms, T_semesters)

    for i in 1:n_firms
        rng = MersenneTwister(seed + i)

        eps_D     = randn(rng, T_total)
        eps_sigma = randn(rng, T_total)

        # Apply correlation
        if abs(vol.rho_epsilon_eta) > 1e-10
            rho = vol.rho_epsilon_eta
            eps_sigma .= rho .* eps_D .+ sqrt(1 - rho^2) .* eps_sigma
        end

        Dc = zeros(T_total);  sc = zeros(T_total)
        Dt = zeros(T_total);  st = zeros(T_total)

        Dc[1] = demand.mu_D;   sc[1] = vol.sigma_bar
        Dt[1] = demand.mu_D;   st[1] = vol.sigma_bar

        for t in 2:T_total
            t_post = t - burn_in

            # ---- Volatility ----
            # Control: always normal AR(1)
            sc[t] = vol.sigma_bar * (1 - vol.rho_sigma) + vol.rho_sigma * sc[t-1] + vol.sigma_eta * eps_sigma[t]

            # Treatment: force at shock_semester, otherwise normal AR(1)
            if t_post == shock_semester
                st[t] = sigma_shock_value
            else
                st[t] = vol.sigma_bar * (1 - vol.rho_sigma) + vol.rho_sigma * st[t-1] + vol.sigma_eta * eps_sigma[t]
            end

            # ---- Demand ----
            sig_level_c = sigma_space == :log ? exp(sc[t]) : max(sc[t], 1e-10)
            sig_level_t = sigma_space == :log ? exp(st[t]) : max(st[t], 1e-10)

            # Jensen's inequality correction: -σ²/2 ensures E[exp(σε - σ²/2)] = 1
            jc_c = apply_jensen ? -0.5 * sig_level_c^2 : 0.0
            jc_t = apply_jensen ? -0.5 * sig_level_t^2 : 0.0

            Dc[t] = demand.mu_D * (1 - demand.rho_D) + demand.rho_D * Dc[t-1] + jc_c + sig_level_c * eps_D[t]
            Dt[t] = demand.mu_D * (1 - demand.rho_D) + demand.rho_D * Dt[t-1] + jc_t + sig_level_t * eps_D[t]
        end

        D_ctrl[i, :]   = Dc[(burn_in+1):end]
        sig_ctrl[i, :] = sc[(burn_in+1):end]
        D_trt[i, :]    = Dt[(burn_in+1):end]
        sig_trt[i, :]  = st[(burn_in+1):end]
    end

    control   = _build_shock_panel(n_firms, T_semesters, D_ctrl, sig_ctrl, D_space, sigma_space)
    treatment = _build_shock_panel(n_firms, T_semesters, D_trt,  sig_trt,  D_space, sigma_space)

    return (control = control, treatment = treatment)
end

# ============================================================================
# Helper
# ============================================================================

function _build_shock_panel(n_firms, T, D_native, sigma_native, D_space, sigma_space)
    D_level     = D_space     == :log ? exp.(D_native)     : copy(D_native)
    sigma_level = sigma_space == :log ? exp.(sigma_native) : copy(sigma_native)
    return ShockPanel(n_firms, T, D_native, sigma_native, D_level, sigma_level, D_space, sigma_space)
end

# ============================================================================
# Convenience: shock_year → shock_semester mapping
# ============================================================================

"""
    year_to_semester(year::Int; stage::Symbol=:begin) -> Int

Convert a year index to the corresponding semester index used in ShockPanel.

- `stage = :begin` → beginning of year (semester 2*(year-1) + 1)
- `stage = :mid`   → mid-year          (semester 2*year)
"""
function year_to_semester(year::Int; stage::Symbol = :begin)
    @assert year >= 1 "year must be >= 1"
    @assert stage in (:begin, :mid) "stage must be :begin or :mid"
    return stage == :begin ? 2 * (year - 1) + 1 : 2 * year
end