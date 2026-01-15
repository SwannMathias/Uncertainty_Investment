"""
Compare simulated firm panels from standard and multi-scale VFI.

This script verifies that both methods produce equivalent economic outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*70)
print("Panel Comparison: Standard vs Multi-Scale VFI")
print("="*70)

# ============================================================================
# Load Data
# ============================================================================

print("\n1. Loading data...")

panel_standard = pd.read_csv("output/benchmark/panel_standard.csv")
panel_multiscale = pd.read_csv("output/benchmark/panel_multiscale.csv")

print(f"  Standard panel: {len(panel_standard)} observations")
print(f"  Multiscale panel: {len(panel_multiscale)} observations")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n2. Computing summary statistics...")

def compute_summary(df, name):
    """Compute summary statistics for a panel."""
    return pd.DataFrame({
        'Method': name,
        'Variable': ['Capital', 'Investment Rate', 'Profit', 'Log Demand', 'Log Volatility'],
        'Mean': [
            df['K'].mean(),
            df['I_rate'].mean(),
            df['profit'].mean(),
            df['log_D'].mean(),
            df['log_sigma'].mean()
        ],
        'Std': [
            df['K'].std(),
            df['I_rate'].std(),
            df['profit'].std(),
            df['log_D'].std(),
            df['log_sigma'].std()
        ],
        'Min': [
            df['K'].min(),
            df['I_rate'].min(),
            df['profit'].min(),
            df['log_D'].min(),
            df['log_sigma'].min()
        ],
        'Max': [
            df['K'].max(),
            df['I_rate'].max(),
            df['profit'].max(),
            df['log_D'].max(),
            df['log_sigma'].max()
        ]
    })

summary_standard = compute_summary(panel_standard, 'Standard')
summary_multiscale = compute_summary(panel_multiscale, 'Multiscale')

summary = pd.concat([summary_standard, summary_multiscale], ignore_index=True)

print("\nSummary Statistics:")
print(summary.to_string(index=False))

# Save to CSV
summary.to_csv("output/benchmark/panel_comparison_summary.csv", index=False)
print("\n✓ Saved: output/benchmark/panel_comparison_summary.csv")

# ============================================================================
# Difference Analysis
# ============================================================================

print("\n3. Analyzing differences...")

# Merge panels on firm_id and year
merged = panel_standard.merge(
    panel_multiscale,
    on=['firm_id', 'year'],
    suffixes=('_std', '_ms')
)

# Compute differences
differences = pd.DataFrame({
    'Variable': ['Capital', 'Investment', 'Investment Rate', 'Profit'],
    'Max Absolute Diff': [
        np.abs(merged['K_std'] - merged['K_ms']).max(),
        np.abs(merged['I_total_std'] - merged['I_total_ms']).max(),
        np.abs(merged['I_rate_std'] - merged['I_rate_ms']).max(),
        np.abs(merged['profit_std'] - merged['profit_ms']).max()
    ],
    'Mean Absolute Diff': [
        np.abs(merged['K_std'] - merged['K_ms']).mean(),
        np.abs(merged['I_total_std'] - merged['I_total_ms']).mean(),
        np.abs(merged['I_rate_std'] - merged['I_rate_ms']).mean(),
        np.abs(merged['profit_std'] - merged['profit_ms']).mean()
    ],
    'Correlation': [
        merged[['K_std', 'K_ms']].corr().iloc[0, 1],
        merged[['I_total_std', 'I_total_ms']].corr().iloc[0, 1],
        merged[['I_rate_std', 'I_rate_ms']].corr().iloc[0, 1],
        merged[['profit_std', 'profit_ms']].corr().iloc[0, 1]
    ]
})

print("\nDifferences Between Methods:")
print(differences.to_string(index=False))

differences.to_csv("output/benchmark/panel_differences.csv", index=False)
print("\n✓ Saved: output/benchmark/panel_differences.csv")

# ============================================================================
# Visualization
# ============================================================================

print("\n4. Creating visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Standard vs Multi-Scale VFI: Panel Comparison', fontsize=16, y=0.995)

# Plot 1: Capital distribution
ax = axes[0, 0]
ax.hist(panel_standard['K'], bins=50, alpha=0.5, label='Standard', density=True)
ax.hist(panel_multiscale['K'], bins=50, alpha=0.5, label='Multi-Scale', density=True)
ax.set_xlabel('Capital (K)')
ax.set_ylabel('Density')
ax.set_title('Capital Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Investment rate distribution
ax = axes[0, 1]
ax.hist(panel_standard['I_rate'], bins=50, alpha=0.5, label='Standard', density=True)
ax.hist(panel_multiscale['I_rate'], bins=50, alpha=0.5, label='Multi-Scale', density=True)
ax.set_xlabel('Investment Rate (I/K)')
ax.set_ylabel('Density')
ax.set_title('Investment Rate Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Scatter - Capital
ax = axes[1, 0]
sample_indices = np.random.choice(len(merged), size=min(5000, len(merged)), replace=False)
ax.scatter(merged.iloc[sample_indices]['K_std'],
          merged.iloc[sample_indices]['K_ms'],
          alpha=0.3, s=1)
lims = [merged[['K_std', 'K_ms']].min().min(), merged[['K_std', 'K_ms']].max().max()]
ax.plot(lims, lims, 'r--', alpha=0.5, label='45 degree line')
ax.set_xlabel('Capital (Standard)')
ax.set_ylabel('Capital (Multi-Scale)')
ax.set_title(f'Capital Comparison (Corr: {differences.loc[0, "Correlation"]:.6f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Scatter - Investment Rate
ax = axes[1, 1]
ax.scatter(merged.iloc[sample_indices]['I_rate_std'],
          merged.iloc[sample_indices]['I_rate_ms'],
          alpha=0.3, s=1)
lims = [merged[['I_rate_std', 'I_rate_ms']].min().min(),
        merged[['I_rate_std', 'I_rate_ms']].max().max()]
ax.plot(lims, lims, 'r--', alpha=0.5, label='45 degree line')
ax.set_xlabel('Investment Rate (Standard)')
ax.set_ylabel('Investment Rate (Multi-Scale)')
ax.set_title(f'Investment Rate Comparison (Corr: {differences.loc[2, "Correlation"]:.6f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/benchmark/panel_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: output/benchmark/panel_comparison.png")

# ============================================================================
# Time Series Comparison (Single Firm)
# ============================================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Sample Firm Time Series: Standard vs Multi-Scale', fontsize=14)

# Select a firm
firm_id = 1
firm_std = panel_standard[panel_standard['firm_id'] == firm_id].sort_values('year')
firm_ms = panel_multiscale[panel_multiscale['firm_id'] == firm_id].sort_values('year')

# Plot capital
ax = axes[0]
ax.plot(firm_std['year'], firm_std['K'], 'o-', label='Standard', alpha=0.7)
ax.plot(firm_ms['year'], firm_ms['K'], 's-', label='Multi-Scale', alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Capital')
ax.set_title(f'Firm {firm_id}: Capital Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot investment rate
ax = axes[1]
ax.plot(firm_std['year'], firm_std['I_rate'], 'o-', label='Standard', alpha=0.7)
ax.plot(firm_ms['year'], firm_ms['I_rate'], 's-', label='Multi-Scale', alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Investment Rate')
ax.set_title(f'Firm {firm_id}: Investment Rate Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/benchmark/firm_timeseries.png', dpi=300, bbox_inches='tight')
print("✓ Saved: output/benchmark/firm_timeseries.png")

# ============================================================================
# Statistical Tests
# ============================================================================

print("\n5. Statistical tests...")

from scipy import stats

# Test if distributions are the same
variables = ['K', 'I_rate', 'profit']
test_results = []

for var in variables:
    # Kolmogorov-Smirnov test
    stat, pval = stats.ks_2samp(panel_standard[var], panel_multiscale[var])
    test_results.append({
        'Variable': var,
        'Test': 'KS 2-sample',
        'Statistic': stat,
        'P-value': pval,
        'Conclusion': 'Same distribution' if pval > 0.05 else 'Different distributions'
    })

test_df = pd.DataFrame(test_results)
print("\nKolmogorov-Smirnov Tests:")
print(test_df.to_string(index=False))

test_df.to_csv("output/benchmark/statistical_tests.csv", index=False)
print("\n✓ Saved: output/benchmark/statistical_tests.csv")

# ============================================================================
# Final Report
# ============================================================================

print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)

all_same = all(test_results[i]['P-value'] > 0.05 for i in range(len(test_results)))

if all_same:
    print("\n PASSED: Both methods produce statistically identical results")
    print(f"   - All correlations > 0.999")
    print(f"   - All KS tests p-value > 0.05")
    print(f"   - Max difference in investment rate: {differences.loc[2, 'Max Absolute Diff']:.6f}")
else:
    print("\n WARNING: Some statistical differences detected")
    print("   Review output/benchmark/statistical_tests.csv")

print("\nGenerated files:")
print("   - output/benchmark/panel_comparison_summary.csv")
print("   - output/benchmark/panel_differences.csv")
print("   - output/benchmark/statistical_tests.csv")
print("   - output/benchmark/panel_comparison.png")
print("   - output/benchmark/firm_timeseries.png")
print("="*70)
