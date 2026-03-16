#!/bin/bash
#
# benchmark_thread_scaling.sh
#
# Benchmarks UncertaintyInvestment.jl solve speed across different thread counts.
# Spawns separate Julia processes with -t N for each thread count.
#
# Usage:
#   chmod +x benchmark_thread_scaling.sh
#   ./benchmark_thread_scaling.sh [PROJECT_ROOT]
#
# The script will:
#   1. Run the model solver with 1, 2, 4, 8, 16, 32, 64, 128 threads
#   2. Collect timing data into output/benchmarks/thread_scaling/
#   3. Call a Python script to generate plots
#
# Adapt THREAD_COUNTS below to your machine.
#

set -e

# ─── Configuration ─────────────────────────────────────────────────────────
PROJECT_ROOT="${1:-.}"                  # Default: current directory
N_RUNS=3                                # Repetitions per thread count
WORKER_SCRIPT="scripts/benchmark_thread_scaling_worker.jl"

# Detect available CPU cores
if command -v nproc &> /dev/null; then
    MAX_CORES=$(nproc)
elif command -v sysctl &> /dev/null; then
    MAX_CORES=$(sysctl -n hw.ncpu)
else
    MAX_CORES=128
fi

echo "Detected $MAX_CORES CPU cores"

# Build thread count list: 1, 2, 4, ..., up to MAX_CORES
THREAD_COUNTS=()
t=1
while [ $t -le $MAX_CORES ]; do
    THREAD_COUNTS+=($t)
    t=$((t * 2))
done
# Always include MAX_CORES if not already in list
last=${THREAD_COUNTS[-1]}
if [ "$last" -ne "$MAX_CORES" ]; then
    THREAD_COUNTS+=($MAX_CORES)
fi

echo "Thread counts to benchmark: ${THREAD_COUNTS[*]}"

# ─── Output directory ──────────────────────────────────────────────────────
OUTDIR="$PROJECT_ROOT/output/benchmarks/thread_scaling"
mkdir -p "$OUTDIR"

# ─── Run benchmarks ───────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Thread Scaling Benchmark"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Runs per config: $N_RUNS"
echo "Output: $OUTDIR"
echo "========================================"

for N in "${THREAD_COUNTS[@]}"; do
    echo ""
    echo "──────────────────────────────────────"
    echo "  Threads: $N"
    echo "──────────────────────────────────────"

    RESULT_FILE="$OUTDIR/results_${N}t.csv"

    julia -t "$N" --project="$PROJECT_ROOT" \
        "$PROJECT_ROOT/$WORKER_SCRIPT" \
        "$N_RUNS" "$RESULT_FILE" "$PROJECT_ROOT"

    echo "  → Saved: $RESULT_FILE"
done

# ─── Merge results ─────────────────────────────────────────────────────────
MERGED="$OUTDIR/all_results.csv"
echo "n_threads,component,run,time_seconds" > "$MERGED"
for N in "${THREAD_COUNTS[@]}"; do
    tail -n +2 "$OUTDIR/results_${N}t.csv" >> "$MERGED"
done
echo ""
echo "Merged results: $MERGED"

# ─── Generate plot ─────────────────────────────────────────────────────────
echo ""
echo "Generating plots..."
if command -v python3 &> /dev/null; then
    python3 "$PROJECT_ROOT/scripts/plot_thread_scaling.py" "$OUTDIR"
elif command -v python &> /dev/null; then
    python "$PROJECT_ROOT/scripts/plot_thread_scaling.py" "$OUTDIR"
else
    echo "Python not found. Skipping plot generation."
    echo "Run manually: python scripts/plot_thread_scaling.py $OUTDIR"
fi

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "========================================"
