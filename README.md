# UncertaintyInvestment.jl
 
A comprehensive Julia package for solving and estimating dynamic investment models with stochastic volatility and flexible adjustment costs.
 
## Features
 
- **Flexible Economic Model**: Iso-elastic demand + Cobb-Douglas production with stochastic volatility
- **Adjustment Cost Menu**: 6 different specifications (convex, fixed, asymmetric, partial irreversibility, composite)
- **Intra-Period Information**: Mid-year information arrival allowing investment revisions
- **Efficient Solution**: Value function iteration with Howard acceleration
- **Multi-Core Parallelization**: Thread-based parallel execution for VFI and simulation
- **Simulation**: Generate firm panels from solved models
- **GMM Estimation**: Indirect inference via auxiliary regressions (planned)
- **SMM-PSO Estimation**: Simulated Method of Moments with Particle Swarm Optimization
 
## Installation
 
### Quick Start (Recommended)
 
Simply run the main calibration script—it will handle setup automatically:
 
```bash
julia run_calibration.jl
```
 
On first run, this will automatically install all dependencies.
 
### Manual Setup
 
If you want to use the package interactively:
 
```julia
# From the repository directory
using Pkg
Pkg.activate(".")
Pkg.instantiate()
 
# Load the package
using UncertaintyInvestment
```
 
## Quick Start
 
### 1. Define Parameters
 
```julia
using UncertaintyInvestment
 
# Create model parameters
params = ModelParameters(
    alpha = 0.33,        # Capital share
    epsilon = 4.0,         # Demand elasticity
    delta = 0.10,        # Annual depreciation
    beta = 0.96,        # Annual discount factor
    demand = DemandProcess(mu_D = 0.0, rho_D = 0.9),
    volatility = VolatilityProcess(sigma_bar = log(0.1), rho_sigma = 0.95, sigma_eta = 0.1),
    numerical = NumericalSettings(n_K = 100, n_D = 15, n_sigma = 7)
)
 
# Print parameter summary
print_parameters(params)
```
 
### 2. Solve Model
 
```julia
# Baseline: No adjustment costs
sol_baseline = solve_model(params; ac = NoAdjustmentCost(), verbose = true)
 
# With convex adjustment costs
ac = ConvexAdjustmentCost(phi = 2.0)
sol_ac = solve_model(params; ac = ac, verbose = true)
 
# With fixed costs
ac_fixed = FixedAdjustmentCost(F = 0.1)
sol_fixed = solve_model(params; ac = ac_fixed)
 
# Composite costs
ac_composite = CompositeAdjustmentCost(
    FixedAdjustmentCost(F = 0.05),
    ConvexAdjustmentCost(phi = 1.0)
)
sol_comp = solve_model(params; ac = ac_composite)
```
 
### 3. Save Solution
 
```julia
using JLD2
 
# Save to file
save_solution("output/solutions/baseline.jld2", sol_baseline)
 
# Load from file
sol_loaded = load_solution("output/solutions/baseline.jld2")
 
# Export to CSV for analysis
export_to_csv(sol_ac, "output/solutions/with_ac/")
```
 
### 4. Simulate Firms
 
```julia
using Random
 
# Set seed for reproducibility
Random.seed!(12345)
 
# Generate shock panel
shocks = generate_shock_panel(
    params.demand,
    params.volatility,
    1000,  # Number of firms
    120    # Number of semesters
)
 
# Print shock statistics
print_shock_statistics(shocks)
 
# Simulate firm panel
histories = simulate_firm_panel(
    sol_ac,
    shocks;
    K_init = 1.0,
    T_years = 50
)
 
# Construct estimation panel
panel = construct_estimation_panel(histories)
print_panel_summary(panel)
 
# Save simulation
save_simulation("output/simulations/panel_data.csv", panel)
```
 
### 5. Analyze Results
 
```julia
# Evaluate value and policy at arbitrary points
K = 1.0
D = 1.0
sigma = 0.1
 
V_val = evaluate_value(sol_ac, K, D, sigma)
I_opt = evaluate_policy(sol_ac, K, D, sigma)
 
println("At (K=$K, D=$D, sigma=$sigma):")
println("  Value: $V_val")
println("  Optimal investment: $I_opt")
```
 
## Multi-Scale Grid Refinement (Performance Optimization)
 
The package includes an optional **multi-scale grid refinement** optimization that speeds up model solution by 3-5x with no loss in accuracy.
 
### How It Works
 
Multi-scale VFI uses a coarse-to-fine approach:
 
1. **Solve on coarse grid** (grid sizes / 2) with relaxed tolerance
2. **Interpolate** solution to fine grid
3. **Refine** on fine grid (converges in ~10-20 iterations instead of ~100-500)
 
This exploits the smoothness of value functions to dramatically reduce computation time.
 
### Usage
 
```julia
# Standard VFI (default)
sol = solve_model(params; ac=ConvexAdjustmentCost(phi=2.0))
 
# Multi-scale VFI (3-5x faster)
sol = solve_model(params; ac=ConvexAdjustmentCost(phi=2.0), use_multiscale=true)
```
 
Both methods produce identical results (differences < 1e-4).
 
### When to Use
 
**Highly recommended when:**
- Fine grids (n_K > 50)
- Repeated solves (GMM estimation)
- Repeated solves (SMM estimation)
- Smooth value functions (no/convex adjustment costs)
 
**Less beneficial when:**
- Small grids (n_K < 30)
- Fixed costs (discontinuities reduce interpolation accuracy)
- Single solve with fast convergence
 
### Benchmark
 
Run the comprehensive benchmark:
 
```bash
# Solve model with both methods, simulate panels, compare
julia scripts/benchmark_multiscale.jl
 
# Generate comparison plots
python scripts/compare_panels.py
```
 
**Expected results:**
- Speedup: 3-5x on typical grids (80 x 12 x 6)
- Solution accuracy: identical (max difference < 1e-4)
- Panel simulations: statistically indistinguishable
 
### Implementation Details
 
The multi-scale solver:
- Uses linear interpolation in capital (continuous)
- Nearest neighbor for demand/volatility (discrete)
- Maintains exact same model specification
- Fully compatible with all adjustment cost types
- Works with both serial and parallel execution
 
## SMM Estimation via PSO
 
The package estimates structural adjustment cost parameters by matching simulated moments to empirical targets using **Simulated Method of Moments (SMM)** with a **Particle Swarm Optimization (PSO)** search algorithm.
 
### Key Features
 
- **Flexible parameter specification**: Declare which parameters are estimated (with bounds) vs held fixed (with values) via a dict-based interface
- **Warm-starting VFI**: Each PSO particle caches its value function, so repeated model solves start from the previous solution
- **Latin Hypercube initialization**: Better coverage of the parameter space than uniform random sampling
- **Checkpointing and resume**: Full PSO state saved periodically; estimation can be resumed from any checkpoint
- **Dimension-agnostic**: Works with any subset of the 4 composite cost parameters `{F_begin, F_mid, phi_begin, phi_mid}`
 
### Julia API
 
```julia
using UncertaintyInvestment
using StatsModels: @formula
 
# Estimate all 4 parameters (composite fixed + convex costs)
config = SMMConfig(
    calibration = FixedCalibration(),
    m_data = [0.35, 0.50, -0.15, 0.10],
    shock_seed = 42,
)
 
pso_config = PSOConfig(
    n_particles = 20,
    max_iterations = 100,
    verbose = true,
    output_dir = "output/estimation/run_001",
)
 
result = run_smm_estimation(config, pso_config)
 
println("Best parameters: ", result.theta_best)
println("Objective: ", result.objective_best)
println("Converged: ", result.converged)
```
 
### Dict-Based Parameter Specification
 
The cost structure is always composite (FixedAdjustmentCost + ConvexAdjustmentCost at each stage). Use `fixed_params` and `estimated_params` to control which parameters are optimized:
 
```julia
# Estimate convex costs only, hold fixed costs constant
config = SMMConfig(
    fixed_params = Dict(:F_begin => 0.5, :F_mid => 0.5),
    estimated_params = Dict(:phi_begin => (0.0, 20.0), :phi_mid => (0.0, 20.0)),
    moments = [
        RegressionCoefficientMoment(:begin,
            @formula(revision_begin ~ log_sigma + log_K + log_D),
            :log_sigma, "coef_begin_sigma"),
        RegressionCoefficientMoment(:mid,
            @formula(revision_mid ~ log_sigma_half + log_K + log_D),
            :log_sigma_half, "coef_mid_sigma"),
    ],
    m_data = [-0.15, 0.10],
)
```
 
Available parameters and their mapping:
 
| Parameter | Stage | Cost Type | Field |
|-----------|-------|-----------|-------|
| `:F_begin` | `:begin` | `FixedAdjustmentCost` | `:F` |
| `:F_mid` | `:mid` | `FixedAdjustmentCost` | `:F` |
| `:phi_begin` | `:begin` | `ConvexAdjustmentCost` | `:phi` |
| `:phi_mid` | `:mid` | `ConvexAdjustmentCost` | `:phi` |
 
Available moment types:
- `ShareZeroMoment(stage, name)` — fraction of firms with near-zero investment at a given stage
- `RegressionCoefficientMoment(stage, formula, coef_name, name)` — OLS coefficient from a fixest-style `@formula`
 
### CLI Usage
 
```bash
# Default: estimate all 4 parameters
julia -t 1 scripts/run_smm.jl \
    --n_particles 20 \
    --max_iterations 100 \
    --m_data "0.35,0.50,-0.15,0.10" \
    --output output/estimation/run_001/
 
# Convex costs only with fixed costs held constant
julia -t 1 scripts/run_smm.jl \
    --fixed_params "F_begin=0.5,F_mid=0.5" \
    --estimated_params "phi_begin=0:20,phi_mid=0:20" \
    --m_data "-0.15,0.10" \
    --max_iterations 50 \
    --output output/estimation/convex_run/
 
# Resume from checkpoint
julia -t 1 scripts/run_smm.jl \
    --resume output/estimation/run_001/checkpoint_iter_50.jld2 \
    --output output/estimation/run_001/
```
 
### PSO Hyperparameters
 
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_particles` | 20 | Number of swarm particles |
| `max_iterations` | 100 | Maximum PSO iterations |
| `w_inertia` | 0.7 | Inertia weight (momentum) |
| `c_cognitive` | 1.5 | Pull toward personal best |
| `c_social` | 1.5 | Pull toward global best |
| `reassign_every` | 20 | Re-randomize worst particles every N iterations (0 = disabled) |
| `reassign_fraction` | 0.1 | Fraction of particles to reassign |
| `tol_objective` | 1e-8 | Convergence tolerance |
| `patience` | 20 | Stop after N iterations without improvement |
| `checkpoint_every` | 10 | Save checkpoint every N iterations |
 
### Output Files
 
After estimation, the output directory contains:
 
| File | Contents |
|------|----------|
| `smm_results.jld2` | Best parameters, moments, convergence info |
| `pso_history.jld2` | Full iteration history (theta and objective per iteration) |
| `pso_log.log` | CSV log of each iteration |
| `estimation_summary.txt` | Human-readable results summary |
| `checkpoint_iter_N.jld2` | PSO state snapshots for crash recovery |
 
## Model Specification
 
### Timeline Within Year t
 
```
Year t
├── Beginning (t)
│   ├── Observe: (K_t, D_t, sigma_t)
│   └── Choose: Initial investment I_t
│
├── Mid-year (t + 1/2)
│   ├── Observe: (D_{t+1/2}, sigma_{t+1/2})
│   └── Choose: Investment revision Delta_I_t
│
└── End of year
    └── Capital: K_{t+1} = (1-delta)K_t + I_t + Delta_I_t
```
 
### Stochastic Processes (Semester Frequency)
 
**Demand:**
```
log D_{s+1/2} = mu_D(1-rho_D) + rho_D log D_s + sigma_s epsilon_{s+1/2}
```
 
**Volatility:**
```
log sigma_{s+1/2} = sigma_bar(1-rho_sigma) + rho_sigma log sigma_s + sigma_eta eta_{s+1/2}
```
 
### Profit Function
 
```
pi(K, D) = (h/(1-gamma)) D^gamma K^(1-gamma)
```
 
where:
- gamma = (epsilon-1)/(epsilon-(1-alpha))
- h = alpha(1-1/epsilon)^(epsilon/alpha) (1-alpha)^(epsilon/alpha-1)
 
### Bellman Equations
 
**Beginning of year:**
```
V(K, D, sigma) = max_I { pi(K,D) - C_1(I,K) + E[W(K', D, sigma) | D, sigma] }
```
 
**Mid-year:**
```
W(K', D, sigma) = E{ max_Delta_I { pi(K,D_1/2) - C_2(Delta_I,K) + beta E[V(K'', D', sigma') | D_1/2, sigma_1/2] }}
```
 
## Adjustment Cost Specifications
 
| Type | Formula | Parameters |
|------|---------|------------|
| **None** | 0 | — |
| **Convex** | (phi/2)(I_total/K)² K | phi |
| **Separate** | (phi_1/2)(I/K)² K + (phi_2/2)(Delta_I/K)² K | phi_1, phi_2 |
| **Fixed** | F · 1{I_total ≠ 0} | F |
| **Asymmetric** | phi_plus(I_plus)²/K + phi_minus(I_minus)²/K | phi_plus, phi_minus |
| **Partial Irreversibility** | -(1-p_S) max(-I_total, 0) | p_S in [0,1] |
| **Composite** | Sum of above | varies |
 
## Examples
 
See the `scripts/` directory for complete examples:
 
- `solve_baseline.jl`: Solve baseline model with various adjustment costs
- `run_simulation.jl`: Generate simulated data (coming soon)
- `comparative_statics.jl`: Parameter sensitivity analysis (coming soon)
 
### Running Scripts
 
The easiest way to run scripts is using the main runner:
 
```bash
# Run the baseline calibration (solves model with all adjustment cost types)
julia run_calibration.jl
 
# Or explicitly specify the script name
julia run_calibration.jl solve_baseline
```
 
The runner script will automatically:
1. Activate the project environment
2. Install dependencies if needed (first time only)
3. Execute the requested script
 
Alternatively, run scripts directly:
 
```bash
# From the repository root
julia scripts/solve_baseline.jl
```
 
The scripts will create output in the `output/` directory:
- `output/solutions/`: Saved model solutions (.jld2 files)
- `output/solutions/*/`: CSV exports of policy functions and value functions
 
## Project Structure
 
```
Uncertainty_Investment/
├── src/
│   ├── UncertaintyInvestment.jl    # Main module
│   ├── model/                   # Economic primitives
│   │   ├── parameters.jl
│   │   ├── primitives.jl
│   │   ├── adjustment_costs.jl
│   │   ├── stochastic_process.jl
│   │   └── grids.jl
│   ├── solution/                # Solution algorithms
│   │   ├── bellman.jl
│   │   ├── vfi.jl
│   │   └── interpolation.jl
│   ├── simulation/              # Simulation tools
│   │   ├── simulate_shocks.jl
│   │   ├── simulate_firms.jl
│   │   └── panel.jl
│   ├── estimation/              # SMM-PSO estimation
│   │   ├── types.jl
│   │   ├── estimation_spec.jl
│   │   ├── smm_config.jl
│   │   ├── moments.jl
│   │   ├── smm_objective.jl
│   │   ├── pso.jl
│   │   └── run_estimation.jl
│   └── utils/                   # Utilities
│       ├── numerical.jl
│       └── io.jl
├── test/                        # Test suite
├── scripts/                     # Example scripts
├── output/                      # Results directory
└── Project.toml                 # Dependencies
```
 
## Multi-Core Parallelization
 
The package supports multi-threaded parallel execution for both VFI solution and firm simulation, providing significant speedups on multi-core systems.
 
### Enabling Parallelization
 
To use multiple threads, start Julia with the `-t` flag:
 
```bash
# Use 8 threads
julia -t 8 scripts/solve_baseline.jl
 
# Use all available CPU cores
julia -t auto scripts/solve_baseline.jl
```
 
Alternatively, set the environment variable before starting Julia:
 
```bash
export JULIA_NUM_THREADS=8
julia scripts/solve_baseline.jl
```
 
### Parallelized Operations
 
| Operation | Function | Parallelization Strategy |
|-----------|----------|--------------------------|
| **VFI Bellman Operator** | `solve_model()` | State space distributed across threads |
| **Howard Improvement** | `solve_model()` | State space distributed across threads |
| **Firm Simulation** | `simulate_firm_panel()` | Firms distributed across threads |
| **Shock Generation** | `generate_shock_panel()` | Firms with thread-safe RNGs |
 
### Usage Examples
 
```julia
using UncertaintyInvestment
 
# Check available threads
println("Threads: $(get_nthreads())")
 
# Solve model with parallelization (enabled by default)
sol = solve_model(params; ac=ConvexAdjustmentCost(phi=2.0), use_parallel=true)
 
# Disable parallelization for debugging
sol_serial = solve_model(params; use_parallel=false)
 
# Parallel firm simulation
histories = simulate_firm_panel(sol, shocks; use_parallel=true)
 
# Parallel shock generation with reproducible seed
shocks = generate_shock_panel(params.demand, params.volatility, 1000, 120;
                              use_parallel=true, seed=12345)
```
 
### Reproducibility
 
Parallel execution maintains full reproducibility:
 
- **VFI**: Deterministic (no RNG involved)
- **Firm Simulation**: Deterministic when shock paths are pre-generated
- **Shock Generation**: Reproducible when using `seed` parameter
 
```julia
# Reproducible parallel shock generation
shocks1 = generate_shock_panel_parallel(demand, vol, 1000, 120; seed=42)
shocks2 = generate_shock_panel_parallel(demand, vol, 1000, 120; seed=42)
@assert shocks1.D == shocks2.D  # Always true, regardless of thread count
```
 
### Performance Expectations
 
Typical speedups on an 8-core machine:
 
| Operation | Speedup | Notes |
|-----------|---------|-------|
| VFI Solution | 4-6x | Near-linear for large state spaces |
| Firm Simulation | 6-8x | Near-linear scaling |
| Shock Generation | 5-7x | I/O bound for small panels |
 
Speedup depends on:
- Number of physical CPU cores (not hyperthreads)
- State space size (larger = better scaling)
- Problem complexity (more computation per state = better)
 
### Thread Safety
 
All parallel operations are thread-safe:
 
- **Read-only sharing**: Value functions, grids, and parameters shared safely
- **Independent writes**: Each thread writes to unique array locations
- **No locks needed**: No synchronization overhead within iterations
- **Per-firm RNGs**: Shock generation uses independent RNG streams
 
### Limitations
 
1. **Memory**: Each thread may allocate temporary arrays
2. **Overhead**: For very small problems, serial may be faster
3. **BLAS**: Julia's BLAS operations may also use threads; consider `BLAS.set_num_threads(1)` if contention occurs
 
### Verifying Parallel Execution
 
```julia
# Check if parallelization is active
sol = solve_model(params; verbose=true)
# Output will show: "Parallelization: ENABLED (8 threads)"
 
# Check threads used in solution
println("Threads used: $(sol.convergence.threads_used)")
```
 
## Impulse Response Functions (IRF)

The package implements the Bloom (2009, Econometrica) IRF protocol for analyzing the effect of uncertainty shocks on firm investment. See `scripts/irf.jl`.

### Mean-Preserving Spread Correction (Jensen's Inequality)

When demand evolves in log space, a volatility shock mechanically affects the expected *level* of demand via Jensen's inequality: `E[exp(log D)] != exp(E[log D])` when variance changes. To isolate the pure uncertainty channel from this first-moment effect, the IRF generator supports three correction modes via the `mean_preserving` argument:

| Mode | Description |
|------|-------------|
| `:none` | No correction. The raw IRF captures both the uncertainty effect and the mechanical first-moment effect of the variance shock. |
| `:static` | Apply the differential correction at the shock semester only. Corrects the one-time drift shift caused by the variance shock at impact. |
| `:dynamic` | Apply the differential correction every semester from the shock onward. Tracks the evolving variance gap as treatment and control paths reconverge. |

**Correction formula** (applied to treatment path only):

```
jc_t = -0.5 * (sigma_treatment^2 - sigma_control^2)
```

This *differential* formulation corrects only the **excess** variance introduced by the shock, not the baseline variance shared by both control and treatment. The control path never receives any correction (`jc_c = 0`).

**Key properties:**
- Before the shock: `sigma_treatment == sigma_control`, so `jc_t = 0` (no correction needed)
- At/after the shock: correction = `-0.5 * (sigma_trt^2 - sigma_ctrl^2)`, removing the mechanical drift effect
- As paths reconverge: the correction naturally decays to zero (for `:dynamic` mode)
- Only active when `demand.process_space == :log`

**Usage:**

```julia
# No correction (default)
panels = generate_irf_panels(demand, vol, n_firms, T; shock_semester=s, mean_preserving=:none)

# Static correction (shock semester only)
panels = generate_irf_panels(demand, vol, n_firms, T; shock_semester=s, mean_preserving=:static)

# Dynamic correction (every semester from shock onward)
panels = generate_irf_panels(demand, vol, n_firms, T; shock_semester=s, mean_preserving=:dynamic)
```

The `scripts/irf.jl` script loops over all three treatments and produces a combined panel (`panel_combined.parquet`) with a `treatment` column identifying the correction mode and a `group` column distinguishing control from treatment.

## Performance Tips
 
1. **Grid Size**: Start with smaller grids (n_K=50, n_D=10, n_sigma=5) for testing
2. **Howard Acceleration**: Use `howard_steps=10` for faster convergence
3. **Parallelization**: Use multiple threads for production runs (`julia -t auto`)
4. **Initial Guess**: Provide `V_init` when solving similar models
5. **Thread Count**: Use physical cores, not hyperthreads, for best efficiency
 
## Citation
 
If you use this package in your research, please cite:
 
```bibtex
@software{optimalinvestment2024,
  title = {UncertaintyInvestment.jl: Dynamic Investment Models with Stochastic Volatility},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/UncertaintyInvestment.jl}
}
```
 
## License
 
MIT License - see LICENSE file for details.
 
## Contributing
 
Contributions welcome! Please open an issue or pull request.
 
## Authors
 
- Your Name (@yourusername)
 
## Acknowledgments
 
This package implements models from the literature on dynamic investment under uncertainty, particularly building on:
 
- Abel & Eberly (1994, 1996): Optimal investment with adjustment costs
- Bloom (2009): Impact of uncertainty on investment
- Cooper & Haltiwanger (2006): Discrete investment choices
 
## Status
 
**Version 0.2.0** - Core functionality complete with parallelization:
- ✅ Model solution (VFI)
- ✅ Simulation
- ✅ Adjustment cost menu
- ✅ Stochastic volatility
- ✅ Multi-core parallelization (VFI + Simulation)
- 🚧 GMM estimation (in progress)
- ✅ SMM-PSO estimation
- ✅ Comprehensive tests
 
## Support
 
For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com
