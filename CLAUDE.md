# Investment Under Uncertainty - Project Documentation

## Project Overview

This is a Julia-based computational economics project for PhD research on optimal investment under uncertainty. The project implements, solves, and estimates dynamic models of firm investment decisions with stochastic volatility and flexible adjustment costs.

- **Collaborators**: Two economics PhD students
- **Language**: Julia (core model), Python (analysis/visualization)
- **Version Control**: Git

## Economic Model

### Firm Problem

Firms operate under monopolistic competition and face a nested dynamic programming problem with intra-period information arrival:

### Timeline (Annual Frequency)

```
Year t
├── Beginning (t)
│   ├── State: (Kₜ, Dₜ, σₜ)
│   ├── Flow profit: π(Kₜ, Dₜ)
│   └── Decision: Initial investment Iₜ
│
├── Mid-year (t + ½)
│   ├── NEW information: (Dₜ₊₁/₂, σₜ₊₁/₂)
│   ├── Flow profit: π(Kₜ, Dₜ₊₁/₂)
│   └── Decision: Investment revision ΔIₜ
│
└── End of year
    └── Capital evolution: Kₜ₊₁ = (1-δ)Kₜ + Iₜ + ΔIₜ
```

### Key Features

**Technology**
- Iso-elastic demand: Pₜ(Yₜ) = Dₜ Yₜ^(-1/ε)
- Cobb-Douglas production: Yₜ = Kₜ^α Xₜ^(1-α)
- Reduced-form profit: π(K,D) = h·D^γ·K^(1-γ)

**Stochastic Processes (semester frequency)**
- Demand: log Dₛ₊₁/₂ = μ_D(1-ρ_D) + ρ_D log Dₛ + σₛ εₛ₊₁/₂
- Volatility: log σₛ₊₁/₂ = σ̄(1-ρ_σ) + ρ_σ log σₛ + σ_η ηₛ₊₁/₂

**Adjustment Costs (flexible menu)**
- None
- Convex: (φ/2)(I_total/K)²K
- Fixed: F·𝟙{I_total ≠ 0}
- Asymmetric: φ₊(I₊)²/K + φ₋(I₋)²/K
- Partial irreversibility: -(1-p_S)max(-I_total, 0)
- Composite: combinations of above

## Computational Implementation

### Solution Algorithm

Value Function Iteration (VFI) with nested Bellman operators:

1. **Beginning-of-year problem:**
   ```
   V(K, D, σ) = max_I { π(K,D) - C₁(I,K) + E[W(K', D, σ) | D, σ] }
   ```

2. **Mid-year problem:**
   ```
   W(K', D, σ) = E_{D',σ'} [ max_ΔI { π(K,D') - C₂(ΔI,K) + β E[V(K'', D'', σ'') | D', σ'] } ]
   ```

### Performance Optimizations

- **Multi-threaded parallelization**
  - State space distributed across threads
  - Near-linear speedup on multi-core systems
  - Thread-safe implementation

- **Multi-scale grid refinement (optional)**
  - Solve on coarse grid → interpolate → refine on fine grid
  - 3-5x speedup with identical results

- **Howard acceleration**
  - Policy iteration steps to accelerate convergence

- **Profit precomputation**
  - Pre-compute π(K, D) for all grid points before VFI
  - Array lookups instead of function calls during iteration
  - Significant speedup for repeated profit evaluations

### Key Data Structures

```julia
# Model solution
SolvedModel {
    params::ModelParameters
    grids::StateGrids
    V::Array{Float64,3}        # V[i_K, i_D, i_σ]
    I_policy::Array{Float64,3} # I[i_K, i_D, i_σ]
    convergence::NamedTuple
}

# State space
StateGrids {
    K_grid::Vector{Float64}     # Capital (continuous, discretized)
    sv::SVDiscretization        # Demand & volatility (discrete)
    Pi_semester::Matrix{Float64}
    Pi_year::Matrix{Float64}
}
```

## Project Structure

```
UncertaintyInvestment.jl/
├── src/
│   ├── UncertaintyInvestment.jl     # Main module
│   ├── model/                        # Economic primitives
│   │   ├── parameters.jl
│   │   ├── primitives.jl             # Profit function
│   │   ├── adjustment_costs.jl       # Cost specifications
│   │   ├── stochastic_process.jl     # Discretization
│   │   └── grids.jl
│   ├── solution/                     # Solution algorithms
│   │   ├── bellman.jl                # Nested DP operators
│   │   ├── vfi.jl                    # VFI solver
│   │   └── interpolation.jl
│   ├── simulation/                   # Simulation tools
│   │   ├── simulate_shocks.jl
│   │   ├── simulate_firms.jl
│   │   └── panel.jl
│   ├── estimation/                   # GMM estimation
│   │   └── types.jl
│   └── utils/                        # Utilities
│       ├── numerical.jl
│       └── io.jl
├── scripts/                          # Executable scripts
│   ├── solve_baseline.jl
│   ├── benchmark_multiscale.jl
│   └── compare_panels.py
├── test/
│   ├── runtests.jl
│   └── test_parallelization.jl
└── output/                           # Results
```

## Making Changes to the Code

### Critical Testing Requirement

⚠️ **ANY MODIFICATION TO THE CODE MUST INCLUDE A VALUE FUNCTION COMPARISON TEST**

Before making changes:
1. Solve the model and save the value function
2. Make your modifications
3. Re-solve the model
4. Compare value functions before/after

### Required Test Template

When modifying code that could affect the model solution, create a test file like this:

```julia
"""
test_modification_[description].jl

Test that [description of change] preserves solution accuracy.
"""

using UncertaintyInvestment
using Test
using Random

println("Testing modification: [description]")
println("="^70)

# Set reproducible parameters
Random.seed!(12345)
params = ModelParameters(
    alpha = 0.33,
    epsilon = 4.0,
    delta = 0.10,
    beta = 0.96,
    demand = DemandProcess(mu_D = 0.0, rho_D = 0.9),
    volatility = VolatilityProcess(
        sigma_bar = log(0.1),
        rho_sigma = 0.95,
        sigma_eta = 0.15
    ),
    numerical = NumericalSettings(
        n_K = 50,
        n_D = 15,
        n_sigma = 7
    )
)

ac = ConvexAdjustmentCost(phi = 2.0)

# Solve with ORIGINAL code (before modification)
println("\n1. Solving with ORIGINAL implementation...")
sol_original = solve_model(params; ac=ac, verbose=true)

# Solve with MODIFIED code (after modification)
println("\n2. Solving with MODIFIED implementation...")
sol_modified = solve_model(params; ac=ac, verbose=true)

# Compare value functions
println("\n3. Comparing value functions...")
V_diff = maximum(abs.(sol_original.V .- sol_modified.V))
V_mean = mean(abs.(sol_original.V))
V_rel_diff = V_diff / V_mean * 100

println("\nValue Function Comparison:")
println("  Max absolute difference: $(V_diff)")
println("  Mean |V|: $(V_mean)")
println("  Relative difference: $(V_rel_diff)%")

# Compare policy functions
I_diff = maximum(abs.(sol_original.I_policy .- sol_modified.I_policy))
I_mean = mean(abs.(sol_original.I_policy))
I_rel_diff = I_mean > 0 ? I_diff / I_mean * 100 : 0.0

println("\nPolicy Function Comparison:")
println("  Max absolute difference: $(I_diff)")
println("  Mean |I|: $(I_mean)")
if I_mean > 0
    println("  Relative difference: $(I_rel_diff)%")
end

# PASS/FAIL criteria
tolerance = 1e-4  # Adjust based on modification type

@testset "Value Function Preservation" begin
    @test V_diff < tolerance "Value function changed by $(V_diff), exceeds tolerance $(tolerance)"
    @test I_diff < tolerance "Policy function changed by $(I_diff), exceeds tolerance $(tolerance)"
end

println("\n" * "="^70)
if V_diff < tolerance && I_diff < tolerance
    println("✓ TEST PASSED: Solution preserved within tolerance")
    println("  Modification is safe to commit")
else
    println("✗ TEST FAILED: Solution changed beyond tolerance")
    println("  Investigate before committing")
end
println("="^70)
```

### When to Run Comparison Tests

**Always required** for modifications to:
- `src/solution/bellman.jl` (Bellman operators)
- `src/solution/vfi.jl` (VFI algorithm)
- `src/model/primitives.jl` (Profit function)
- `src/model/adjustment_costs.jl` (Cost calculations)
- `src/model/stochastic_process.jl` (Discretization)
- `src/model/grids.jl` (State space construction)

**Recommended** for modifications to:
- `src/solution/interpolation.jl`
- `src/model/parameters.jl`
- `src/utils/numerical.jl`

**Not required** for:
- Documentation changes
- Test file modifications
- Utility functions that don't affect solution
- Output/IO functions

### Acceptable Differences

Different modification types have different tolerance levels:
- **Bug fixes** → Expect differences, document them
- **Performance optimizations** → Should have V_diff < 1e-6
- **Numerical precision changes** → Should have V_diff < 1e-4
- **Algorithm changes** → Compare with benchmark, document if different

### Example Workflow

```bash
# 1. Create feature branch
git checkout -b feature/improve-convergence

# 2. Solve baseline (before changes)
julia scripts/solve_baseline.jl
cp output/solutions/baseline.jld2 output/solutions/baseline_before.jld2

# 3. Make your modifications
# ... edit files ...

# 4. Create comparison test
julia test_modification_convergence.jl

# 5. If test passes, commit
git add .
git commit -m "Improve convergence with [method]

- Modified: [files]
- Value function preserved within 1e-6
- Speedup: [X]x faster"

# 6. Push and create PR
git push origin feature/improve-convergence
```

## Common Modification Patterns

### 1. Adding New Adjustment Cost

```julia
# 1. Define cost type in adjustment_costs.jl
struct NewCost <: AbstractAdjustmentCost
    param::Float64
end

# 2. Implement interface
compute_cost(ac::NewCost, I, ΔI, K) = ...
marginal_cost_I(ac::NewCost, I, ΔI, K) = ...
# ... etc

# 3. Test with comparison
ac_new = NewCost(param=1.0)
sol_new = solve_model(params; ac=ac_new)
sol_convex = solve_model(params; ac=ConvexAdjustmentCost(phi=1.0))
# Compare if they should be equivalent
```

### 2. Improving VFI Convergence

```julia
# 1. Save baseline
sol_before = solve_model(params; ac=ac, verbose=true)

# 2. Modify vfi.jl
# ... changes to value_function_iteration() ...

# 3. Test
sol_after = solve_model(params; ac=ac, verbose=true)
@test maximum(abs.(sol_before.V .- sol_after.V)) < 1e-6

# 4. Document convergence improvement
println("Iterations before: $(sol_before.convergence.iterations)")
println("Iterations after: $(sol_after.convergence.iterations)")
```

### 3. Changing Grid Construction

```julia
# Grid changes WILL affect value function
# Document the change and verify convergence still works

grids_old = construct_grids(params)
sol_old = solve_model(params)

# Modify grids.jl
grids_new = construct_grids(params)
sol_new = solve_model(params)

# These WILL differ, but solution should still converge
@test sol_old.convergence.converged
@test sol_new.convergence.converged
# Document differences in commit message
```

## Development Guidelines

### Code Style
- Follow existing conventions (see Project Instructions)
- Document all public functions
- Add type annotations where helpful
- Use descriptive variable names

### Performance
- Profile before optimizing
- Maintain multi-threading support
- Test serial vs parallel consistency

### Git Workflow
- Feature branches for new work
- Clear commit messages
- Reference issues in commits
- Run tests before pushing

### Documentation
- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features

## Quick Reference

### Solve a Model

```julia
using UncertaintyInvestment

params = ModelParameters(alpha=0.33, epsilon=4.0)
sol = solve_model(params; ac=ConvexAdjustmentCost(phi=2.0))
```

### Compare Solutions

```julia
sol1 = load_solution("output/solutions/before.jld2")
sol2 = load_solution("output/solutions/after.jld2")

V_diff = maximum(abs.(sol1.V .- sol2.V))
println("Max difference: $V_diff")
```

### Run All Tests

```bash
julia -t 8 --project=. -e 'using Pkg; Pkg.test()'
```

### Benchmark

```bash
julia -t 8 scripts/benchmark_multiscale.jl
```

## Support

For questions or issues:
1. Check existing tests in `test/`
2. Review `README.md` for usage examples
3. Examine `scripts/` for complete workflows
4. Open issue if needed

---

**Remember: Value function preservation is not optional. It's the core output of the model and must be verified for any code change that could affect the solution algorithm.**
