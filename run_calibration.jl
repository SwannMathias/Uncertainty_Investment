#!/usr/bin/env julia
"""
Main runner script for the Uncertainty Investment model.

Usage:
    julia run_calibration.jl [script_name]

Examples:
    julia run_calibration.jl                    # Run solve_baseline.jl
    julia run_calibration.jl solve_baseline     # Same as above

Available scripts:
    - solve_baseline: Solve baseline model with various adjustment costs
"""

using Pkg

# Get script directory
const PROJECT_ROOT = @__DIR__

# Activate project
println("Activating project at $PROJECT_ROOT")
Pkg.activate(PROJECT_ROOT)

# Install dependencies if needed
if !isfile(joinpath(PROJECT_ROOT, "Manifest.toml"))
    println("Installing dependencies (first time setup)...")
    Pkg.instantiate()
    println("✓ Dependencies installed")
end

# Determine which script to run
script_name = length(ARGS) >= 1 ? ARGS[1] : "solve_baseline"

# Remove .jl extension if provided
script_name = replace(script_name, r"\.jl$" => "")

# Build script path
script_path = joinpath(PROJECT_ROOT, "scripts", "$(script_name).jl")

if !isfile(script_path)
    println("ERROR: Script not found: $script_path")
    println("\nAvailable scripts:")
    for file in readdir(joinpath(PROJECT_ROOT, "scripts"))
        if endswith(file, ".jl")
            println("  - $(replace(file, ".jl" => ""))")
        end
    end
    exit(1)
end

# Run the script
println("Running script: $script_path\n")
include(script_path)
