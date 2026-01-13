"""
Construct estimation panels from simulated firm histories.
"""

using DataFrames

"""
    FirmPanel

Container for firm panel data in DataFrame format.
"""
struct FirmPanel
    df::DataFrame
    n_firms::Int
    T::Int
end

"""
    construct_estimation_panel(histories::Vector{FirmHistory}) -> FirmPanel

Convert simulated firm histories into panel DataFrame.

# Arguments
- `histories`: Vector of FirmHistory objects

# Returns
- FirmPanel with DataFrame containing:
  - firm_id: Firm identifier
  - year: Year
  - K: Capital stock
  - D: First semester demand
  - D_half: Second semester demand
  - sigma: First semester volatility
  - sigma_half: Second semester volatility
  - I: Initial investment
  - Delta_I: Investment revision
  - I_total: Total investment
  - I_rate: Investment rate (I_total / K)
  - profit: Annual profit
  - log_D, log_sigma: Log states
"""
function construct_estimation_panel(histories::Vector{FirmHistory})
    n_firms = length(histories)
    @assert n_firms > 0 "Need at least one firm"

    T = histories[1].T
    @assert all(h.T == T for h in histories) "All firms must have same length"

    # Build rows
    rows = []

    for (firm_id, hist) in enumerate(histories)
        for year in 1:hist.T
            push!(rows, (
                firm_id = firm_id,
                year = year,
                K = hist.K[year],
                D = hist.D[year],
                D_half = hist.D_half[year],
                sigma = hist.sigma[year],
                sigma_half = hist.sigma_half[year],
                log_D = log(hist.D[year]),
                log_D_half = log(hist.D_half[year]),
                log_sigma = log(hist.sigma[year]),
                log_sigma_half = log(hist.sigma_half[year]),
                I = hist.I[year],
                Delta_I = hist.Delta_I[year],
                I_total = hist.I_total[year],
                I_rate = hist.I_total[year] / hist.K[year],
                profit = hist.profit[year]
            ))
        end
    end

    df = DataFrame(rows)

    return FirmPanel(df, n_firms, T)
end

"""
    panel_summary_statistics(panel::FirmPanel) -> NamedTuple

Compute summary statistics for firm panel.
"""
function panel_summary_statistics(panel::FirmPanel)
    df = panel.df

    return (
        n_firms = panel.n_firms,
        T = panel.T,
        n_obs = nrow(df),
        # Capital
        K_mean = mean(df.K),
        K_std = std(df.K),
        # Investment rate
        I_rate_mean = mean(df.I_rate),
        I_rate_std = std(df.I_rate),
        I_rate_median = median(df.I_rate),
        # Investment
        I_total_mean = mean(df.I_total),
        I_total_std = std(df.I_total),
        # Inaction rate
        inaction_rate = sum(abs.(df.I_total) .< 1e-6) / nrow(df),
        # Demand
        D_mean = mean(df.D),
        D_std = std(df.D),
        # Volatility
        sigma_mean = mean(df.sigma),
        sigma_std = std(df.sigma),
        # Profit
        profit_mean = mean(df.profit),
        profit_std = std(df.profit)
    )
end

"""
    print_panel_summary(panel::FirmPanel)

Print formatted panel summary statistics.
"""
function print_panel_summary(panel::FirmPanel)
    stats = panel_summary_statistics(panel)

    println("\n" * "="^70)
    println("Panel Summary Statistics")
    println("="^70)
    println("Firms: $(stats.n_firms)")
    println("Years per firm: $(stats.T)")
    println("Total observations: $(stats.n_obs)")

    println("\nCapital:")
    println("  Mean: $(format_number(stats.K_mean))")
    println("  Std Dev: $(format_number(stats.K_std))")

    println("\nInvestment Rate (I/K):")
    println("  Mean: $(format_number(stats.I_rate_mean, digits=4))")
    println("  Median: $(format_number(stats.I_rate_median, digits=4))")
    println("  Std Dev: $(format_number(stats.I_rate_std, digits=4))")

    if stats.inaction_rate > 0.01
        println("  Inaction rate: $(format_number(stats.inaction_rate * 100, digits=2))%")
    end

    println("\nDemand:")
    println("  Mean: $(format_number(stats.D_mean))")
    println("  Std Dev: $(format_number(stats.D_std))")

    println("\nVolatility:")
    println("  Mean: $(format_number(stats.sigma_mean, digits=4))")
    println("  Std Dev: $(format_number(stats.sigma_std, digits=4))")

    println("\nProfit:")
    println("  Mean: $(format_number(stats.profit_mean))")
    println("  Std Dev: $(format_number(stats.profit_std))")

    println("="^70)
end
