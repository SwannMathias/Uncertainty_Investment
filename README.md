# UncertaintyInvestment.jl

A comprehensive Julia package for solving and estimating dynamic investment models with stochastic volatility and flexible adjustment costs.

## Features

- **Flexible Economic Model**: Iso-elastic demand + Cobb-Douglas production with stochastic volatility
- **Adjustment Cost Menu**: 6 different specifications (convex, fixed, asymmetric, partial irreversibility, composite)
- **Intra-Period Information**: Mid-year information arrival allowing investment revisions
- **Efficient Solution**: Value function iteration with Howard acceleration
- **Simulation**: Generate firm panels from solved models
- **GMM Estimation**: Indirect inference via auxiliary regressions (planned)

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
│   └── utils/                   # Utilities
│       ├── numerical.jl
│       └── io.jl
├── test/                        # Test suite
├── scripts/                     # Example scripts
├── output/                      # Results directory
└── Project.toml                 # Dependencies
```

## Performance Tips

1. **Grid Size**: Start with smaller grids (n_K=50, n_D=10, n_sigma=5) for testing
2. **Howard Acceleration**: Use `howard_steps=10` for faster convergence
3. **Parallel Simulation**: Firms are independent—use `@threads` for large panels
4. **Initial Guess**: Provide `V_init` when solving similar models

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

**Version 0.1.0** - Core functionality complete:
- ✅ Model solution (VFI)
- ✅ Simulation
- ✅ Adjustment cost menu
- ✅ Stochastic volatility
- 🚧 GMM estimation (in progress)
- 🚧 Comprehensive tests (in progress)

## Support

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com
