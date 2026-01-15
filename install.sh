#!/bin/bash
#
# Install all dependencies for the Uncertainty Investment project
#
# Usage: ./install.sh
#
# This script will:
#   1. Install Julia packages from Project.toml/Manifest.toml
#   2. Install Python packages required for comparison scripts
#

set -e

echo "========================================"
echo "Installing Uncertainty Investment Dependencies"
echo "========================================"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Julia Dependencies
# ============================================================================

echo ""
echo "[1/2] Installing Julia packages..."

if command -v julia &> /dev/null; then
    julia --project="$SCRIPT_DIR" -e '
        using Pkg
        println("  Activating project environment...")
        Pkg.activate(".")
        println("  Installing/updating packages from Project.toml...")
        Pkg.instantiate()
        println("  Precompiling packages...")
        Pkg.precompile()
        println("  Julia packages installed successfully!")
    '
else
    echo "  WARNING: Julia not found in PATH. Please install Julia from https://julialang.org/"
    echo "  Skipping Julia package installation."
fi

# ============================================================================
# Python Dependencies (optional, for comparison scripts)
# ============================================================================

echo ""
echo "[2/2] Installing Python packages..."

if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo "  WARNING: pip not found. Skipping Python package installation."
    PIP_CMD=""
fi

if [ -n "$PIP_CMD" ]; then
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        echo "  Installing packages from requirements.txt via $PIP_CMD..."
        $PIP_CMD install --quiet --upgrade -r "$SCRIPT_DIR/requirements.txt"
        echo "  Python packages installed successfully!"
    else
        echo "  WARNING: requirements.txt not found. Skipping Python installation."
    fi
fi

# ============================================================================
# Done
# ============================================================================

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "To verify Julia installation, run:"
echo "  julia --project=. -e 'using UncertaintyInvestment'"
echo ""
echo "To run the main model:"
echo "  julia --project=. run_calibration.jl"
echo ""
