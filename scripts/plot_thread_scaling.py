"""
plot_thread_scaling.py

Generates thread scaling plots from benchmark results.

Usage:
    python scripts/plot_thread_scaling.py output/benchmarks/thread_scaling/

Produces:
    - thread_scaling.pdf: Speed and efficiency vs thread count
    - optimal_allocation.pdf: Marginal speedup curve for PSO allocation
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ─── Load data ─────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python plot_thread_scaling.py <results_dir>")
    sys.exit(1)

results_dir = sys.argv[1]
merged_file = os.path.join(results_dir, "all_results.parquet")

if not os.path.exists(merged_file):
    print(f"Error: {merged_file} not found")
    sys.exit(1)

df = pd.read_parquet(merged_file)
print(f"Loaded {len(df)} observations")
print(f"Thread counts: {sorted(df.n_threads.unique())}")
print(f"Components: {df.component.unique()}")

# ─── Compute summary statistics ────────────────────────────────────────────

summary = (
    df.groupby(["n_threads", "component"])["time_seconds"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)

# Get serial (1-thread) baseline for each component
baselines = summary[summary.n_threads == 1].set_index("component")["mean"].to_dict()

# Compute speedup and efficiency
summary["speedup"] = summary.apply(
    lambda row: baselines.get(row["component"], row["mean"]) / row["mean"],
    axis=1,
)
summary["efficiency"] = summary["speedup"] / summary["n_threads"]

# ─── Print summary table ──────────────────────────────────────────────────

print("\n" + "=" * 80)
print(f"{'Component':<15} {'Threads':>8} {'Mean(s)':>10} {'Speedup':>10} {'Efficiency':>12}")
print("-" * 80)
for _, row in summary.iterrows():
    print(
        f"{row['component']:<15} {int(row['n_threads']):>8} "
        f"{row['mean']:>10.3f} {row['speedup']:>10.2f}x "
        f"{row['efficiency'] * 100:>10.1f}%"
    )
print("=" * 80)

# ─── Figure 1: Speed and efficiency vs threads ────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

components = sorted(summary.component.unique())
colors = {"VFI": "#2563eb", "Simulation": "#dc2626"}

# Panel A: Raw time
ax = axes[0]
for comp in components:
    sub = summary[summary.component == comp].sort_values("n_threads")
    ax.plot(sub.n_threads, sub["mean"], "o-", color=colors.get(comp, "gray"),
            label=comp, linewidth=2, markersize=6)
    ax.fill_between(sub.n_threads,
                    sub["mean"] - sub["std"],
                    sub["mean"] + sub["std"],
                    alpha=0.15, color=colors.get(comp, "gray"))

ax.set_xlabel("Number of threads", fontsize=12)
ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("(a) Solve time", fontsize=13, fontweight="bold")
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel B: Speedup
ax = axes[1]
thread_range = sorted(summary.n_threads.unique())
ax.plot(thread_range, thread_range, "--", color="gray", alpha=0.5,
        label="Linear (ideal)", linewidth=1)

for comp in components:
    sub = summary[summary.component == comp].sort_values("n_threads")
    ax.plot(sub.n_threads, sub.speedup, "o-", color=colors.get(comp, "gray"),
            label=comp, linewidth=2, markersize=6)

ax.set_xlabel("Number of threads", fontsize=12)
ax.set_ylabel("Speedup (T₁ / Tₙ)", fontsize=12)
ax.set_title("(b) Speedup", fontsize=13, fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel C: Efficiency
ax = axes[2]
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

for comp in components:
    sub = summary[summary.component == comp].sort_values("n_threads")
    ax.plot(sub.n_threads, sub.efficiency, "o-", color=colors.get(comp, "gray"),
            label=comp, linewidth=2, markersize=6)

ax.set_xlabel("Number of threads", fontsize=12)
ax.set_ylabel("Efficiency (Speedup / Threads)", fontsize=12)
ax.set_title("(c) Parallel efficiency", fontsize=13, fontweight="bold")
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
outpath = os.path.join(results_dir, "thread_scaling.pdf")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nSaved: {outpath}")
plt.close()

# ─── Figure 2: Marginal speedup & PSO allocation guide ────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Focus on VFI (the bottleneck in SMM)
vfi = summary[summary.component == "VFI"].sort_values("n_threads").copy()

if len(vfi) >= 2:
    # Panel A: Marginal speedup — how much does each additional doubling help?
    ax = axes[0]
    vfi = vfi.reset_index(drop=True)

    # Compute marginal speedup: ratio of speedup gain from doubling
    marginal_speedup = []
    thread_labels = []
    for i in range(1, len(vfi)):
        t_prev = vfi.loc[i - 1, "n_threads"]
        t_curr = vfi.loc[i, "n_threads"]
        s_prev = vfi.loc[i - 1, "speedup"]
        s_curr = vfi.loc[i, "speedup"]
        # Marginal: how much additional speedup per additional thread
        additional_threads = t_curr - t_prev
        additional_speedup = s_curr - s_prev
        marginal = additional_speedup / additional_threads if additional_threads > 0 else 0
        marginal_speedup.append(marginal)
        thread_labels.append(f"{int(t_prev)}→{int(t_curr)}")

    ax.bar(range(len(marginal_speedup)), marginal_speedup,
           color="#2563eb", alpha=0.7, edgecolor="#1e40af")
    ax.set_xticks(range(len(thread_labels)))
    ax.set_xticklabels(thread_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Thread doubling", fontsize=12)
    ax.set_ylabel("Marginal speedup per thread", fontsize=12)
    ax.set_title("(a) Diminishing returns in VFI threading", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Optimal allocation for PSO
    # Given C total cores, how to split between particles (Distributed) and
    # threads per particle?
    ax = axes[1]

    # Interpolate the speedup curve
    thread_vals = vfi.n_threads.values.astype(float)
    speedup_vals = vfi.speedup.values

    # For a range of total cores C, compute throughput = n_particles × speedup(C/n_particles)
    C_total = int(thread_vals.max())
    if C_total < 4:
        C_total = 128  # fallback

    # Possible allocations: threads_per_particle must be in our measured set
    possible_threads = thread_vals[thread_vals >= 1]
    particles_options = []
    throughput_options = []

    for tpp in possible_threads:
        n_particles = int(C_total // tpp)
        if n_particles < 1:
            continue
        # Find speedup at this thread count (interpolate)
        sp = np.interp(tpp, thread_vals, speedup_vals)
        # Throughput: number of model evaluations per unit time
        # Each particle evaluates at rate sp/T1, and we have n_particles
        throughput = n_particles * sp
        particles_options.append((int(tpp), n_particles, throughput))
        throughput_options.append(throughput)

    if particles_options:
        tpp_list = [p[0] for p in particles_options]
        n_part_list = [p[1] for p in particles_options]
        tp_list = [p[2] for p in particles_options]

        # Normalize throughput
        max_tp = max(tp_list)
        tp_normalized = [t / max_tp for t in tp_list]

        bars = ax.bar(range(len(tpp_list)), tp_normalized,
                      color="#059669", alpha=0.7, edgecolor="#047857")

        # Highlight optimal
        best_idx = tp_normalized.index(1.0)
        bars[best_idx].set_color("#dc2626")
        bars[best_idx].set_edgecolor("#991b1b")

        labels = [f"{t}t×{n}p" for t, n in zip(tpp_list, n_part_list)]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Allocation (threads × particles)", fontsize=12)
        ax.set_ylabel("Relative throughput", fontsize=12)
        ax.set_title(
            f"(b) PSO allocation ({C_total} total cores)",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Print recommendation
        best = particles_options[best_idx]
        print(f"\n{'=' * 60}")
        print(f"PSO ALLOCATION RECOMMENDATION ({C_total} cores)")
        print(f"{'=' * 60}")
        print(f"  Optimal: {best[0]} threads/particle × {best[1]} particles")
        print(f"  Throughput: {best[2]:.1f} (relative model evals/time)")
        print(f"{'=' * 60}")

plt.tight_layout()
outpath = os.path.join(results_dir, "optimal_allocation.pdf")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
plt.close()

print("\nDone!")
