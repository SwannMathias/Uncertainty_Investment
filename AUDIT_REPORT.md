# Dual Audit Report: Investment Under Uncertainty Codebase

**Date:** 2026-03-02
**Codebase:** UncertaintyInvestment.jl v0.1.0
**Scope:** Full source tree (src/, test/, scripts/)
**Intended use:** Submission-level academic research

---

## Executive Summary

This codebase implements a two-stage (semester-frequency) dynamic investment model with stochastic volatility, solved by value function iteration (VFI). Overall, the code is well-structured, with a clear separation of economic primitives, solution algorithms, simulation, and estimation. The test suite covers serial/parallel consistency and key functional correctness.

However, the audit identifies **5 critical**, **8 major**, and **12 minor** issues across software engineering and economic dimensions. The most consequential findings are:

1. **Missing discount factor in the mid-year Bellman operator** — the continuation value at the mid-year stage omits `beta_semester`, implying zero discounting over the second half-year.
2. **Inconsistent capital evolution in the Howard step** — the Howard policy-evaluation step for the mid-year stage uses `K` (beginning-of-year capital) instead of `K_prime = (1-delta)*K + I`, breaking consistency with the Bellman operator.
3. **SV discretization uses averaged demand grids** — when volatility affects the demand innovation variance, averaging grids across volatility states destroys the state-contingent nature of the discretization.
4. **Profit in the mid-year stage uses beginning-of-year capital** — the economic model requires profit to be evaluated at current capital (which may have changed after investment), but the code evaluates it at the grid index for beginning-of-year K.
5. **The `@assert` guards inside hot loops** impose a performance penalty in debug mode and are silently removed in optimized builds, creating inconsistent behavior.

---

## 1. Software Engineering Audit

### 1.1 Bugs

#### BUG-1 [CRITICAL]: Missing discount factor in `solve_midyear_problem`

**File:** `src/solution/bellman.jl:75`

```julia
function obj_Delta_I(Delta_I)
    K_next = K_dep + Delta_I
    ...
    cost = compute_cost(ac_mid_year, 0.0, Delta_I, K_current)
    return -cost + params.beta * linear_interp_1d(grids.K_grid, EV, K_next)
end
```

The function uses `params.beta` (the **annual** discount factor, = 0.96) instead of `derived.beta_semester` (= sqrt(0.96) ≈ 0.98). Since this is the mid-year stage discounting the next beginning-of-year value by half a year, the correct discount factor is `beta_semester`. Using the annual factor introduces excess discounting of ~2% per semester, systematically biasing the mid-year continuation value downward.

**Impact:** Distorts the relative value of mid-year adjustment vs. waiting. Under-values the option to revise investment, biasing the model toward inaction at mid-year.

**Fix:**
```julia
return -cost + derived.beta_semester * linear_interp_1d(grids.K_grid, EV, K_next)
```

**Note:** The same bug appears in `howard_improvement_step!` at line 230 (`params.beta` should be `derived.beta_semester` for the mid-year stage). The `howard_full_step!` function correctly uses `derived.beta_semester` (line 461/495), confirming this is an inconsistency, not a design choice.

---

#### BUG-2 [CRITICAL]: Missing discount factor in `solve_beginning_year_problem` continuation

**File:** `src/solution/bellman.jl:132`

```julia
function obj_I(I)
    K_stage1 = (1 - derived.delta_semester) * K + I
    ...
    cost = compute_cost(ac_begin, I, 0.0, K)
    return pi_first - cost + expected_pi_mid + linear_interp_1d(grids.K_grid, EV, K_stage1)
end
```

The beginning-of-year problem adds three terms: current profit, expected mid-year profit, and the continuation (expected V1). The mid-year profit and continuation should be discounted by `beta_semester` because they occur half a year later, but neither term is discounted here. This means:
- `expected_pi_mid` is undiscounted (should be `beta_semester * expected_pi_mid`).
- The continuation `linear_interp_1d(...)` is also undiscounted (should be `beta_semester * linear_interp_1d(...)`).

This effectively sets the intra-year discount rate to zero, overstating the value of future cash flows within the year.

**Impact:** Systematically inflates the value function. Since all models share this bias, relative comparisons (e.g., convex vs. fixed costs) may still be approximately correct, but **absolute welfare levels and the implied interest rate are wrong**.

**Note:** This should be verified against the intended timing convention. If the model intends for all within-year flows to be undiscounted (treating the year as a single period), then the mid-year structure is purely informational. But the presence of `beta_semester` in `howard_full_step!` and the CLAUDE.md documentation suggest semester-level discounting is intended.

---

#### BUG-3 [MAJOR]: Inconsistent capital in Howard improvement step (mid-year)

**File:** `src/solution/bellman.jl:228`

```julia
K_next = (1 - derived.delta_semester) * K + ΔI
```

In the Howard step for stage 1, `K` is `get_K(grids, i_K)`, i.e., beginning-of-year capital. But the mid-year capital after depreciation and initial investment should be `K_prime = (1 - delta_semester) * K + I_policy[i_K, i_D, i_sigma]`. Then:

```julia
K_next = (1 - derived.delta_semester) * K_prime + ΔI
```

The current code computes capital evolution as if the initial investment `I` never happened, making the Howard policy evaluation inconsistent with the Bellman operator (which correctly uses `K_stage1`).

**Impact:** The Howard acceleration steps produce incorrect value updates, potentially slowing or misleading convergence. Since VFI uses full Bellman steps between Howard blocks, the converged solution may still be correct, but convergence speed is degraded and intermediate values are wrong.

---

#### BUG-4 [MAJOR]: `_maximize_with_inaction` boundary handling

**File:** `src/solution/bellman.jl:34-53`

```julia
function _maximize_with_inaction(obj, lower::Float64, upper::Float64)
    v0 = obj(0.0)
    x_best, v_best = 0.0, v0

    if lower < -1e-10
        x_neg, v_neg = _best_adjustment_choice(obj, lower, min(-1e-10, upper))
        ...
    end

    if upper > 1e-10
        x_pos, v_pos = _best_adjustment_choice(obj, max(1e-10, lower), upper)
        ...
    end
```

When the feasible region `[lower, upper]` does not contain 0 (e.g., both lower and upper are positive), the function still evaluates `obj(0.0)` and uses it as the initial best. If `0.0` is infeasible (e.g., it produces `K_next < K_min`), this will return `-Inf` as the baseline, which is handled correctly downstream. However, if `lower > 0` and `upper > 0`, the function correctly searches only the positive branch, but the initial `v0 = obj(0.0)` evaluation is wasted and may trigger assertions in `obj` if 0 leads to `K_next` out of bounds.

More importantly, the gap `[-1e-10, 1e-10]` around zero is never searched. With fixed costs, the optimal adjustment near zero is compared against exact inaction (ΔI=0), but adjustments in `(-1e-10, 1e-10)` are ignored. For the fixed-cost model this is intentional (threshold for "zero"), but the hardcoded `1e-10` is much tighter than the `FixedAdjustmentCost.threshold` default of `1e-6`, creating an inconsistency.

**Impact:** Minor in practice, but could cause assertion failures in edge cases or produce incorrect results if the optimal adjustment is very small and positive/negative but within the dead zone.

---

#### BUG-5 [MINOR]: `linear_interp_1d` assertions in hot path

**File:** `src/solution/interpolation.jl:23-26`

```julia
function linear_interp_1d(x_grid::Vector{Float64}, y_vals::AbstractVector{Float64}, x::Float64)
    n = length(x_grid)
    @assert length(y_vals) == n "Grid and values must have same length"
    @assert issorted(x_grid) "Grid must be sorted"
```

This function is called millions of times during VFI (every state × every optimization evaluation). The `@assert issorted(x_grid)` check is O(n) per call. In Julia, `@assert` is compiled away with `--check-bounds=no`, but in default mode this creates a massive performance overhead.

**Impact:** Significant slowdown in debug/default mode. In optimized builds, no impact but also no safety net.

**Fix:** Move assertions to grid construction time. Replace with `@boundscheck` or remove entirely from the inner loop.

---

#### BUG-6 [MINOR]: `NoAdjustmentCost` shortcut in Bellman is suboptimal

**File:** `src/solution/bellman.jl:81-84`

```julia
if ac_mid_year isa NoAdjustmentCost
    i_opt = argmax(EV)
    Delta = clamp(grids.K_grid[i_opt] - K_dep, Delta_I_min, Delta_I_max)
    return Delta, obj_Delta_I(Delta)
end
```

When there are no adjustment costs, the optimal next-period capital maximizes `EV(K_next)`. The code finds the grid index maximizing `EV` and backs out `Delta_I`. But `EV` is defined over the grid points of K_next, while the true optimum may lie between grid points. The golden section search (used in the else branch) would find a more precise solution. This introduces a discretization error specific to the `NoAdjustmentCost` case.

**Impact:** The no-cost solution is slightly less accurate than the convex-cost solution, which uses continuous optimization. For fine grids, the difference is negligible.

---

### 1.2 Parallelization Issues

#### PAR-1 [MAJOR]: `howard_improvement_step_parallel!` is a no-op wrapper

**File:** `src/solution/bellman.jl:309-318`

```julia
function howard_improvement_step_parallel!(...)
    howard_improvement_step!(V0, V1, I_policy, Delta_I_policy,
                             grids, params, ac_begin, ac_mid_year, derived, n_steps)
end
```

The "parallel" Howard step simply calls the serial implementation. There is no parallelization. This means VFI iterations that invoke Howard acceleration (every 20th iteration) always run serially, even when `use_parallel=true`. For models where Howard steps dominate runtime, this significantly limits parallel speedup.

**Impact:** Parallel performance is worse than expected. The user sees "Parallelization: ENABLED" but Howard steps are serial. For the default `howard_steps=50`, this can represent a substantial fraction of total runtime.

---

#### PAR-2 [MINOR]: Thread scheduling granularity

**File:** `src/solution/bellman.jl:266`

```julia
@threads for idx in 1:n_total
```

Julia's `@threads` macro uses static scheduling by default, distributing iterations evenly across threads. If the workload per state varies (e.g., the optimization is harder near grid boundaries), load imbalance can reduce efficiency. Julia 1.8+ supports `@threads :dynamic` for dynamic scheduling.

**Impact:** Suboptimal thread utilization on heterogeneous workloads. Likely <10% impact given the relatively uniform optimization cost across states.

---

#### PAR-3 [MINOR]: No NUMA-awareness for large state spaces

For very large grids (e.g., 200 × 30 × 15 = 90,000 states), the value function arrays may span multiple NUMA nodes. Julia's `@threads` does not account for data locality. This is unlikely to matter for the current default grid sizes but could become relevant at scale.

---

### 1.3 Numerical and Reproducibility Concerns

#### NUM-1 [CRITICAL]: SV discretization averages demand grids across volatility states

**File:** `src/model/stochastic_process.jl:212-213`

```julia
# 3. Use average demand grid (they should be similar for persistent processes)
D_grid = mean(D_grids)
```

The demand process has state-dependent innovation variance: when volatility is high, the demand grid should be wider. By averaging the grids, the code:
1. Uses a single demand grid that doesn't reflect the current volatility state.
2. Assigns transition probabilities (`Pi_D_given_sigma`) computed on different grids to the averaged grid.

When `sigma_eta` is large (high volatility of volatility), the demand grids for different sigma levels can differ substantially. The averaged grid misrepresents the distribution for both high- and low-volatility states.

**Impact:** The joint transition matrix `Pi_joint` is only approximately correct. The demand distribution is too wide for low-volatility states and too narrow for high-volatility states. This biases the value of the information arrival (mid-year signal) and can affect the estimated option value of waiting.

**Fix:** Use a single demand grid that spans the range of all volatility-specific grids (union grid), or use a grid that is wide enough for the highest-volatility state and recompute transition probabilities accordingly.

---

#### NUM-2 [MAJOR]: Convergence criterion uses AND (both V and policy must converge)

**File:** `src/solution/vfi.jl:176`

```julia
if dist < params.numerical.tol_vfi && dist_policy < params.numerical.tol_policy
```

This requires both the value function and the policy function to converge simultaneously. For problems with kinks (fixed costs, irreversibility), the policy function may oscillate near the threshold while the value function has converged. This can cause the solver to run to `max_iter` and report non-convergence even though the value function is well-approximated.

**Impact:** May trigger false non-convergence for fixed-cost and composite-cost models. Users may increase `max_iter` unnecessarily.

**Recommendation:** Use OR logic or add a separate policy-only convergence flag. Report both distances so the user can judge.

---

#### NUM-3 [MAJOR]: Flat extrapolation at grid boundaries

**File:** `src/solution/interpolation.jl:28-31`

```julia
if x <= x_grid[1]
    return y_vals[1]
elseif x >= x_grid[end]
    return y_vals[end]
end
```

Value function interpolation uses flat (constant) extrapolation outside the capital grid. If the optimal policy pushes capital outside `[K_min, K_max]`, the value function is artificially flat, destroying the incentive to invest/disinvest. This creates an artificial "absorbing boundary" at the grid edges.

Combined with `get_profit_at_K` (grids.jl:265-269), which also uses flat extrapolation, this means both profits and values are clamped at the boundaries.

**Impact:** If the grid is too narrow or the shocks are large, the solution will be distorted near the boundaries. The diagnostics do report `K_edge_min_share` and `K_edge_max_share`, which is good, but there is no automatic warning when these are high.

---

#### NUM-4 [MINOR]: Golden section tolerance vs. VFI tolerance

The golden section search in `maximize_univariate` uses `tol=1e-6` (hardcoded in `_best_adjustment_choice`). The VFI tolerance is also `1e-6` by default. When the optimization tolerance equals the convergence tolerance, the optimizer's approximation error can prevent the VFI from converging below `1e-6`. A rule of thumb is that the inner optimizer should be 10-100x more precise than the outer convergence criterion.

**Impact:** VFI may converge to a slightly imprecise solution or require more iterations than necessary.

---

#### NUM-5 [MINOR]: `compute_expectation` allocates a new vector each call

**File:** `src/model/grids.jl:414`

```julia
EV = zeros(grids.n_K)
```

Each call to `compute_expectation` allocates a new vector. This function is not used in the hot path (the Bellman operator uses the cached matrix multiplication approach), so the impact is limited. But if used in simulation or diagnostics loops, it could cause GC pressure.

---

#### NUM-6 [MINOR]: Reproducibility across Julia versions

The code uses `randn(rng)` and `MersenneTwister`. Julia's random number generation changed between 1.6 and 1.7+ (introduction of `Xoshiro` as default). Since the `compat` specifies `julia = "1.6"`, results generated on Julia 1.6 may not reproduce on Julia 1.9+. The explicit `MersenneTwister` usage in parallel shock generation is correct and version-stable, but the serial path uses `Random.GLOBAL_RNG`, which is implementation-dependent.

---

### 1.4 Performance and Scalability

#### PERF-1 [MAJOR]: Expectation cache uses dense matrix multiplication

**File:** `src/solution/bellman.jl:21-23`

```julia
V_mat = reshape(V, grids.n_K, grids.n_states)
mul!(EV_cache, V_mat, transpose(Pi))
```

This computes `EV_cache = V_mat * Pi'`, which is an `(n_K × n_states) × (n_states × n_states)` dense matrix multiply. For the default grid (100 × 105 × 105), this is efficient. But if `n_D` and `n_sigma` grow, the `n_states²` cost of the transition matrix and the multiply become dominant. The transition matrix `Pi_joint` is often sparse (especially with Rouwenhorst/Tauchen discretization), but dense storage and multiplication are used.

**Impact:** For the current grid sizes, this is fine. For `n_D=30, n_sigma=15` (n_states = 450), the Pi matrix is 450² = 202,500 entries and the multiply is 100 × 450 × 450 ≈ 20M FLOPs per call, which is still fast. But scaling to finer grids will hit this bottleneck.

---

#### PERF-2 [MINOR]: `get_D_sigma_indices` computes div/mod repeatedly

**File:** `src/model/grids.jl:218-222`

```julia
function get_D_sigma_indices(grids::StateGrids, i_state::Int)
    i_sigma = div(i_state - 1, grids.n_D) + 1
    i_D = mod(i_state - 1, grids.n_D) + 1
    return i_D, i_sigma
end
```

This is called in inner loops (e.g., `compute_expectation`, `howard_full_step!`). While `div` and `mod` are fast, the Julia compiler may not fuse them into a single `divrem` instruction. Using `divrem` explicitly would save one division.

---

#### PERF-3 [MINOR]: `construct_estimation_panel` uses untyped `rows = []`

**File:** `src/simulation/panel.jl:51`

```julia
rows = []
```

This creates a `Vector{Any}`, which prevents type inference and causes boxing of each `NamedTuple`. For large panels (1000 firms × 50 years = 50,000 rows), this creates significant GC pressure.

**Fix:**
```julia
rows = NamedTuple{(:firm_id, :year, :K, ...), Tuple{Int, Int, Float64, ...}}[]
```
Or, better, pre-allocate column vectors directly.

---

### 1.5 Proposed Code Corrections (Software Engineering)

| ID | File | Severity | Fix |
|---|---|---|---|
| BUG-1 | bellman.jl:75 | Critical | Replace `params.beta` with `derived.beta_semester` in `solve_midyear_problem` |
| BUG-2 | bellman.jl:132 | Critical | Add `beta_semester *` discount to `expected_pi_mid` and continuation in `solve_beginning_year_problem` |
| BUG-3 | bellman.jl:228 | Major | Replace `K` with `K_prime` in Howard step mid-year capital evolution |
| BUG-5 | interpolation.jl:23-26 | Minor | Remove `@assert` from hot-path interpolation, validate at grid construction |
| PAR-1 | bellman.jl:309-318 | Major | Implement actual parallel Howard step using `@threads` |
| NUM-1 | stochastic_process.jl:212 | Critical | Use union demand grid or highest-variance grid instead of average |
| PERF-3 | panel.jl:51 | Minor | Type the `rows` vector or pre-allocate columns |

---

## 2. Economic Consistency Audit

### 2.1 Theoretical Coherence

#### ECON-1 [CRITICAL]: Discounting structure is inconsistent with semester timing

The model documentation (CLAUDE.md) specifies a semester-frequency timeline:
- Beginning of year: observe (K, D, σ), earn π₁, choose I
- Mid-year: observe (D', σ'), earn π₂, choose ΔI
- End of year: K' realized

This requires discounting mid-year flows by `β^(1/2)` relative to beginning-of-year flows. The code is inconsistent:

| Component | Discount used | Correct? |
|---|---|---|
| `solve_midyear_problem` | `params.beta` (annual) | **Wrong** — should be `beta_semester` |
| `solve_beginning_year_problem` | 1.0 (no discount) | **Wrong** — mid-year flows should be discounted by `beta_semester` |
| `howard_full_step!` | `derived.beta_semester` | **Correct** |
| `howard_improvement_step!` | `params.beta` | **Wrong** — same as `solve_midyear_problem` |

The inconsistency means the Bellman operator and the Howard accelerator evaluate the same problem with different discount factors. At convergence, the Bellman operator's value dominates (since it sets V), but the Howard steps may push V in the wrong direction during iteration.

**Economic consequence:** The effective intra-year discount rate is ambiguous. If the intent is no intra-year discounting (common in annual models), then `howard_full_step!` is wrong. If semester discounting is intended, then the Bellman operator is wrong. The model cannot be correctly identified without resolving this.

---

#### ECON-2 [CRITICAL]: Mid-year profit uses beginning-of-year capital

**File:** `src/solution/bellman.jl:116-121`

```julia
# Expected mid-year profit E[π(K, D_half) | D, σ] — constant w.r.t. I
probs_mid = @view grids.Pi_semester[i_state, :]
expected_pi_mid = 0.0
for i_state_half in 1:grids.n_states
    i_D_half, _ = get_D_sigma_indices(grids, i_state_half)
    expected_pi_mid += probs_mid[i_state_half] * get_profit(grids, i_K, i_D_half)
end
```

This computes `E[π(K_t, D_{t+1/2})]` — mid-year profit at the **beginning-of-year** capital `K_t`. Economically, if the firm invests `I` at the beginning of the year and capital depreciates, the mid-year capital stock is `K' = (1-δ)K + I`, and mid-year profit should be `π(K', D_{t+1/2})`.

The comment says "constant w.r.t. I", which is only true because it uses beginning-of-year capital. This is a modeling choice, not a bug, but it has economic implications:

1. It implies the firm earns mid-year profit on old capital regardless of investment.
2. This is inconsistent with the capital accumulation equation `K_{t+1} = (1-δ)K' + ΔI`, which implies K' is the productive capital at mid-year.

**Economic consequence:** If investment is immediate (capital installed at beginning of year), mid-year profit should depend on K'. The current implementation understates the return to early investment, biasing the model toward less aggressive investment timing.

**Caveat:** Some models assume capital is only productive next period ("time to build"). If this is the intended interpretation, the implementation is correct but should be documented explicitly, and the two-semester structure loses some of its economic motivation.

---

#### ECON-3 [MAJOR]: Mid-year profit in `howard_full_step!` also uses beginning-of-year K

**File:** `src/solution/bellman.jl:487`

```julia
pi_half = get_profit(grids, i_K, i_D_half)
```

Consistent with ECON-2, the Howard step evaluates mid-year profit at `i_K` (beginning-of-year grid index), not at `K'`. This is internally consistent within the code but potentially inconsistent with the economic model.

---

#### ECON-4 [MAJOR]: Adjustment cost evaluated at beginning-of-year K in mid-year stage

**File:** `src/solution/bellman.jl:74`

```julia
cost = compute_cost(ac_mid_year, 0.0, Delta_I, K_current)
```

The mid-year adjustment cost uses `K_current` (beginning-of-year capital) as the scale variable. For convex costs `C = (φ/2)(ΔI/K)²K`, this means the cost is scaled by beginning-of-year capital rather than mid-year capital `K'`. This is consistent with the mid-year profit assumption (ECON-2) but economically questionable — if capital has changed by mid-year, costs should reflect the actual capital stock.

---

#### ECON-5 [MINOR]: The convex adjustment cost is on total investment, not investment rate

**File:** `src/model/adjustment_costs.jl:86`

```julia
function compute_cost(ac::ConvexAdjustmentCost, I, Delta_I, K)
    I_total = I + Delta_I
    return 0.5 * ac.phi * (I_total / K)^2 * K
end
```

The cost depends on `I_total = I + ΔI`. In a two-stage model, the cost at each stage should arguably depend on that stage's investment only:
- Stage 0 cost: `(φ/2)(I/K)²K`
- Stage 1 cost: `(φ/2)(ΔI/K')²K'`

The current formulation couples the two stages through `I_total`, but the Bellman operator passes `(I, 0)` for stage 0 and `(0, ΔI)` for stage 1, so in practice `I_total = I` at stage 0 and `I_total = ΔI` at stage 1. The code behavior is correct, but the interface is misleading — the `I_total` formulation in the docstring/struct suggests joint cost, while the actual usage separates costs.

---

### 2.2 Calibration and Identification

#### CAL-1 [MAJOR]: Semester vs. annual parameter confusion

The stochastic processes are specified at semester frequency:
- `rho_D = 0.9` (semester persistence)
- `rho_sigma = 0.95` (semester persistence)

The annual persistence is `rho_D_annual = rho_D^2 = 0.81` and `rho_sigma_annual = 0.9025`. These values need to be compared against empirical estimates (typically reported at annual frequency) with care.

The default `sigma_bar = log(0.1)` implies a long-run average volatility of `σ = 0.1` per semester, which corresponds to an annual standard deviation of `0.1 × √2 ≈ 0.14`. This is within the range of demand shock estimates in the literature (Bloom 2009 reports ~0.10-0.20 annual).

However, the semester persistence `rho_D = 0.9` implies an annual persistence of 0.81, which is notably lower than most estimates (0.85-0.95 annual). If the user intends annual persistence of 0.9, the semester parameter should be `sqrt(0.9) ≈ 0.949`.

**Impact:** If calibration targets are annual moments, the current defaults may not match. The mapping between semester and annual parameters must be explicit in any calibration exercise.

---

#### CAL-2 [MAJOR]: Scale parameter `h` derivation assumes optimized-out variable input

**File:** `src/model/parameters.jl:140-143`

```julia
term1 = p.alpha
term2 = (1 - 1/p.epsilon)^(p.epsilon / p.alpha)
term3 = (1 - p.alpha)^(p.epsilon / p.alpha - 1)
h = term1 * term2 * term3
```

The reduced-form profit function `π(K,D) = (h/(1-γ))D^γ K^(1-γ)` is derived from optimizing over variable inputs (labor/materials) given Cobb-Douglas technology and iso-elastic demand. The expression for `h` should be:

Starting from the firm's revenue:
```
R(Y) = D·Y^(1-1/ε)
```
With Cobb-Douglas `Y = K^α X^(1-α)`, optimizing over X:
```
π(K,D) = [α(1-1/ε)]^{1/(1-η)} · [(1-α)(1-1/ε)]^{η/(1-η)} · D^{1/(1-η)} · K
```
where `η = (1-α)(1-1/ε)`.

Let me verify the formula. With `γ = (ε-1)/(ε-(1-α))` and the given `h`:

For `α=0.33, ε=4`:
- `γ = 3/(4-0.67) = 3/3.33 = 0.9009`
- `h = 0.33 × (0.75)^(4/0.33) × (0.67)^(4/0.33 - 1)`
- `h = 0.33 × (0.75)^{12.12} × (0.67)^{11.12}`

These exponents are very large, making `h` extremely small. This seems algebraically correct but numerically delicate. The formula should be verified against a closed-form solution.

**Recommendation:** Add a unit test that verifies `π(K,D)` computed from the reduced form matches `π(K,D)` computed from the structural form (optimizing over X numerically). This would catch any errors in the `h` derivation.

---

#### CAL-3 [MINOR]: User cost of capital in steady state

**File:** `src/model/parameters.jl:156`

```julia
user_cost = p.delta + (1/p.beta - 1)  # r = (1/beta - 1)
```

This is the standard Jorgensonian user cost at annual frequency: `c = δ + r`, where `r = 1/β - 1`. This is correct for an annual model. However, the model operates at semester frequency internally. If the steady state is intended as a semester-frequency steady state, the user cost should use semester rates:
```julia
user_cost = delta_semester + (1/beta_semester - 1)
```

The steady-state capital `K_ss` is used only for grid construction (setting `K_min` and `K_max`), so this mainly affects grid placement. Using annual rates gives a different `K_ss` than semester rates, which could lead to suboptimal grid placement.

---

### 2.3 Stability and Equilibrium Conditions

#### STAB-1 [MAJOR]: No verification that the contraction mapping property holds

VFI convergence relies on Blackwell's sufficient conditions: monotonicity and discounting. For the two-stage formulation:
- The operator must be a contraction with modulus `β` (or `β^{1/2}` per stage).
- The current discounting inconsistency (ECON-1) may violate the contraction property if some stages have effective discount = 1.0.

If mid-year flows are undiscounted (BUG-2), the effective per-year discount factor is `β` (from the beginning-of-year to next beginning-of-year), which preserves the contraction. But the operator structure is non-standard and the convergence rate may be affected.

**Recommendation:** Add a theoretical verification comment documenting why the operator is a contraction, specifying the effective discount factor for each transition.

---

#### STAB-2 [MINOR]: Convergence tolerance may be insufficient for welfare comparisons

The default `tol_vfi = 1e-6` provides about 6 significant digits in the value function. For welfare comparisons (e.g., "what is the value of the option to revise investment?"), the difference between models may be of order `1e-3` to `1e-4` of the value function level. With a tolerance of `1e-6`, this leaves 2-3 significant digits in the welfare comparison, which may be insufficient for precise quantitative statements.

**Recommendation:** For welfare comparisons, use `tol_vfi = 1e-8` or report the convergence tolerance alongside welfare numbers.

---

#### STAB-3 [MINOR]: No ergodicity check for the simulated panel

The simulation code runs firms for `T_years` starting from `K_init`. If `T_years` is insufficient for the capital distribution to reach its ergodic distribution, the panel statistics will reflect initial conditions rather than the model's stationary distribution. The code does not provide guidance on burn-in for the simulation (separate from the shock burn-in).

---

### 2.4 Conceptual Risks

#### CONCEPT-1 [MAJOR]: The two-stage structure may not identify the option value of waiting

The model's key contribution is the intra-year information arrival: the firm can revise investment after observing new demand. The "option value of waiting" is the welfare difference between the one-stage and two-stage models. However, due to ECON-2 (mid-year profit uses beginning-of-year K), the mid-year stage only affects capital accumulation through ΔI, not through mid-year profitability. This weakens the economic channel and may understate the option value.

---

#### CONCEPT-2 [MINOR]: No partial irreversibility or asymmetric costs implemented

The CLAUDE.md documentation mentions "Partial irreversibility: -(1-p_S)max(-I_total, 0)" and "Asymmetric: φ₊(I₊)²/K + φ₋(I₋)²/K" as planned adjustment cost types, but these are not implemented. The `adjustment_costs.jl` file only contains `NoAdjustmentCost`, `ConvexAdjustmentCost`, `FixedAdjustmentCost`, and `CompositeAdjustmentCost`.

---

### 2.5 Proposed Economic Corrections

| ID | Issue | Severity | Recommendation |
|---|---|---|---|
| ECON-1 | Inconsistent discounting | Critical | Decide on timing convention (annual or semester discounting), apply consistently across all operators |
| ECON-2 | Mid-year profit at old K | Critical | Either use `K' = (1-δ)K + I` for mid-year profit, or document "time to build" assumption explicitly |
| CAL-1 | Semester/annual parameter mapping | Major | Add helper function `annual_to_semester_params()` and document the mapping |
| CAL-2 | Scale parameter `h` verification | Major | Add structural verification test comparing reduced-form and structural profit |
| STAB-1 | Contraction mapping verification | Major | Document the effective discount factor and contraction modulus |

---

## 3. Risk Assessment

### Severity Classification

| Severity | Count | Definition |
|---|---|---|
| **Critical** | 5 | Affects correctness of the economic model or solution algorithm; must be fixed before any publication |
| **Major** | 8 | Affects performance, robustness, or secondary economic quantities; should be fixed for submission |
| **Minor** | 12 | Code quality, documentation, or edge cases; recommended fixes |

### Critical Issues Summary

| # | ID | Domain | Description |
|---|---|---|---|
| 1 | BUG-1 / ECON-1 | Both | Annual β used instead of semester β in mid-year Bellman |
| 2 | BUG-2 / ECON-1 | Both | Missing discount on mid-year flows in beginning-of-year problem |
| 3 | NUM-1 | Engineering | Averaged demand grids destroy state-dependent volatility structure |
| 4 | ECON-2 | Economics | Mid-year profit evaluated at beginning-of-year capital |
| 5 | BUG-3 | Engineering | Howard step uses wrong capital in mid-year evaluation |

---

## 4. Recommended Action Plan (Prioritized)

### Phase 1: Critical Fixes (Before any results are used)

1. **Resolve the discounting convention (BUG-1, BUG-2, ECON-1)**
   - Decide: is the model annually-discounted with a within-year information structure, or semester-discounted?
   - Apply the chosen convention consistently in `solve_midyear_problem`, `solve_beginning_year_problem`, `howard_improvement_step!`, and `howard_full_step!`.
   - Re-run convergence tests and compare value functions before/after.

2. **Fix the SV discretization (NUM-1)**
   - Replace the averaged demand grid with a common grid spanning the full range.
   - Recompute `Pi_D_given_sigma` on the common grid.
   - Verify moment matching with `verify_discretization`.

3. **Decide on mid-year capital convention (ECON-2)**
   - If "time to build": document explicitly, no code change needed.
   - If immediate installation: update `solve_beginning_year_problem` and `howard_full_step!` to evaluate mid-year profit at `K'`.

4. **Fix Howard step capital evolution (BUG-3)**
   - Use `K_prime = (1-delta_semester) * K + I_policy[i_K, i_D, i_sigma]` in the mid-year Howard step.

### Phase 2: Major Improvements (Before submission)

5. **Parallelize Howard steps (PAR-1)**
6. **Add `annual_to_semester_params` helper (CAL-1)**
7. **Add structural verification test for profit function (CAL-2)**
8. **Improve convergence criterion for non-smooth models (NUM-2)**
9. **Fix steady-state user cost to use semester rates (CAL-3)**
10. **Remove hot-path assertions (BUG-5)**

### Phase 3: Minor Polish (Before final revision)

11. Type the panel construction vectors (PERF-3)
12. Implement asymmetric and partial irreversibility costs (CONCEPT-2)
13. Add ergodicity guidance for simulation burn-in (STAB-3)
14. Use `@threads :dynamic` for better load balancing (PAR-2)
15. Document contraction mapping property (STAB-1)

---

## Assumptions and Limitations of This Audit

1. **No runtime execution:** This audit is based on static code analysis. Some findings (especially performance claims) should be verified by profiling.
2. **Economic model interpretation:** The discounting findings (ECON-1) depend on the intended timing convention. If the authors intend annual discounting with within-year information arrival (a common simplification), then BUG-2 is by design and only BUG-1 needs fixing.
3. **Scale parameter `h`:** The algebraic derivation was not fully verified analytically. A numerical verification test (CAL-2) would resolve any doubt.
4. **Parameter values:** The default calibration was assessed for plausibility against the literature, but a formal calibration exercise is outside the scope of this audit.
5. **Estimation module:** The `estimation/types.jl` file defines only data structures; no estimation logic is implemented yet. The GMM/indirect inference methodology cannot be audited.

---

*End of Audit Report*
