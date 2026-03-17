"""
Moment computation for SMM estimation.

# Economic context
The estimation matches simulated moments to empirical targets. Available moments:
1. Share of zero investment revisions (extensive margin — identifies fixed costs)
2. OLS regression coefficients (intensive margin — identifies convex costs)

Moments are declared via EstimationSpec and computed dynamically from the
simulated firm panel. Regression moments use FixedEffectModels.jl (Julia's
equivalent of R's fixest) for clear, modifiable formula specifications.
"""

using FixedEffectModels: reg, FixedEffectModel
using StatsModels: @formula, FormulaTerm, coefnames

"""
    apply_transform(E_new, E_old, K, transform::RevisionTransform) -> Float64

Apply the configured transform to compute an investment revision.

# Arguments
- `E_new`: Updated investment expectation (after new information)
- `E_old`: Previous investment expectation (before new information)
- `K`: Current capital stock (used only for LEVEL_OVER_K_TRANSFORM)
- `transform`: Which transform to apply

# Returns
- Transformed revision. Returns NaN if LOG_TRANSFORM and either expectation is non-positive.

# Economic interpretation
Revisions measure how new information changes the firm's investment plan.
The transform choice affects scale and domain:
- LOG_TRANSFORM: log-difference, natural for proportional changes
- LEVEL_OVER_K_TRANSFORM: level difference normalized by capital
- ASINH_TRANSFORM: inverse hyperbolic sine difference, robust to zeros/negatives
"""
function apply_transform(E_new::Float64, E_old::Float64, K::Float64,
                         transform::RevisionTransform)
    if transform == LOG_TRANSFORM
        if E_new <= 0.0 || E_old <= 0.0
            return NaN
        end
        return log(E_new) - log(E_old)
    elseif transform == LEVEL_OVER_K_TRANSFORM
        return (E_new - E_old) / K
    elseif transform == ASINH_TRANSFORM
        return asinh(E_new) - asinh(E_old)
    end
end

"""
    ols_coefficients(y::Vector{Float64}, X::Matrix{Float64}) -> Vector{Float64}

Compute OLS coefficients: beta = (X'X)^{-1} X'y.

X should include a constant column. No external package dependency.
Kept for backward compatibility and testing (cross-validation with reg()).
"""
function ols_coefficients(y::Vector{Float64}, X::Matrix{Float64})
    return (X' * X) \ (X' * y)
end

"""
    compute_revision_panel(df::DataFrame, config::SMMConfig) -> DataFrame

Add revision columns to the panel DataFrame.

# Columns added
- `revision_begin`: Transform of beginning-of-year revision
- `revision_mid`: Transform of mid-year revision

# Economic interpretation
- `revision_begin` = transform(E_beginning) - transform(E_last_semester)
  Measures how the firm updates its investment plan between mid-year of t-1
  and beginning of year t, as new demand/volatility information arrives.
- `revision_mid` = transform(E_half) - transform(E_beginning)
  Measures how the firm updates its plan between beginning and mid-year of t.
"""
function compute_revision_panel(df::DataFrame, config::SMMConfig)
    n = nrow(df)
    revision_begin = Vector{Float64}(undef, n)
    revision_mid = Vector{Float64}(undef, n)

    for i in 1:n
        revision_begin[i] = apply_transform(
            df.E_beginning[i], df.E_last_semester[i], df.K[i],
            config.revision_transform
        )
        revision_mid[i] = apply_transform(
            df.E_half[i], df.E_beginning[i], df.K[i],
            config.revision_transform
        )
    end

    result = copy(df)
    result.revision_begin = revision_begin
    result.revision_mid = revision_mid
    return result
end

"""
    prepare_regression_df(df::DataFrame, config::SMMConfig) -> DataFrame

Prepare a DataFrame for regression-based moment computation.

Adds computed columns needed by regression formulas:
- `revision_begin`, `revision_mid` (from transforms)
- `log_K` (log of capital, not in raw panel)

Filters out invalid observations (year 1 for begin, NaN revisions).
"""
function prepare_regression_df(df::DataFrame, config::SMMConfig)
    # Add revision columns
    result = compute_revision_panel(df, config)

    # Add log_K column (not in raw panel — panel has K but not log_K)
    result.log_K = log.(result.K)

    return result
end

"""
    compute_simulated_moments(panel_df::DataFrame, config::SMMConfig) -> Vector{Float64}

Compute simulated moments from a firm panel according to the EstimationSpec.

Iterates over `config.estimation_spec.moments` and computes each moment:
- `ShareZeroMoment`: share of observations with near-zero revisions
- `RegressionCoefficientMoment`: OLS coefficient via FixedEffectModels.jl `reg()`

# Returns
- Vector of length `n_moments(config.estimation_spec)`. Returns NaN for failed moments.

# Implementation notes
- Year 1 observations are dropped for beginning-of-year moments (E_last_semester is NaN).
- Rows with NaN revisions (from log transform of non-positive values) are dropped.
- Requires ≥10 valid observations per moment to proceed.
"""
function compute_simulated_moments(panel_df::DataFrame, config::SMMConfig)
    spec = config.estimation_spec
    nm = n_moments(spec)

    # Prepare DataFrame with revision columns and log_K
    df = prepare_regression_df(panel_df, config)

    # Pre-compute validity masks
    valid_begin = .!isnan.(df.revision_begin) .& .!isnan.(df.E_last_semester)
    valid_mid = .!isnan.(df.revision_mid)

    n_total = nrow(df)
    n_valid_begin = sum(valid_begin)
    n_valid_mid = sum(valid_mid)

    # Warn if too many observations dropped by log transform
    if config.revision_transform == LOG_TRANSFORM
        if n_valid_begin < n_total / 2
            @warn "Log transform dropped >50% of beginning-of-year observations " *
                  "($(n_total - n_valid_begin)/$n_total). Parameter region may produce " *
                  "mostly negative expected investment."
        end
        if n_valid_mid < n_total / 2
            @warn "Log transform dropped >50% of mid-year observations " *
                  "($(n_total - n_valid_mid)/$n_total)."
        end
    end

    # Compute each moment
    moments = Vector{Float64}(undef, nm)
    for (i, m) in enumerate(spec.moments)
        moments[i] = _compute_single_moment(m, df, valid_begin, valid_mid, config)
    end

    return moments
end

"""
    _compute_single_moment(m::ShareZeroMoment, df, valid_begin, valid_mid, config) -> Float64

Compute share of near-zero revisions for the given stage.
"""
function _compute_single_moment(m::ShareZeroMoment, df::DataFrame,
                                 valid_begin::BitVector, valid_mid::BitVector,
                                 config::SMMConfig)
    if m.stage == :begin
        valid = valid_begin
        revision_col = df.revision_begin
    else
        valid = valid_mid
        revision_col = df.revision_mid
    end

    n_valid = sum(valid)
    if n_valid < 10
        @warn "Too few valid observations for $(m.name) ($n_valid). Returning NaN."
        return NaN
    end

    return mean(abs.(revision_col[valid]) .< config.zero_threshold)
end

"""
    _compute_single_moment(m::RegressionCoefficientMoment, df, valid_begin, valid_mid, config) -> Float64

Compute an OLS regression coefficient using FixedEffectModels.jl `reg()`.

The regression is estimated on the stage-appropriate subset of observations.
The coefficient named by `m.coef_name` is extracted from the results.
"""
function _compute_single_moment(m::RegressionCoefficientMoment, df::DataFrame,
                                 valid_begin::BitVector, valid_mid::BitVector,
                                 config::SMMConfig)
    # Select stage-appropriate observations
    valid = m.stage == :begin ? valid_begin : valid_mid
    n_valid = sum(valid)

    if n_valid < 10
        @warn "Too few valid observations for $(m.name) ($n_valid). Returning NaN."
        return NaN
    end

    # Subset DataFrame for regression
    df_reg = df[valid, :]

    # Check for zero variance in dependent variable
    dep_var_name = m.formula.lhs.sym
    if std(df_reg[!, dep_var_name]) < 1e-15
        return NaN
    end

    # Run regression using FixedEffectModels.jl
    result = try
        reg(df_reg, m.formula)
    catch e
        @warn "Regression failed for $(m.name): $e"
        return NaN
    end

    # Extract the coefficient of interest
    coef_idx = _find_coef_index(result, m.coef_name)
    if isnothing(coef_idx)
        @warn "Coefficient $(m.coef_name) not found in regression for $(m.name). " *
              "Available: $(coefnames(result))"
        return NaN
    end

    coef_val = coef(result)[coef_idx]
    return isfinite(coef_val) ? coef_val : NaN
end

"""
    _find_coef_index(result::FixedEffectModel, name::Symbol) -> Union{Int, Nothing}

Find the index of a named coefficient in a FixedEffectModels regression result.
"""
function _find_coef_index(result::FixedEffectModel, name::Symbol)
    names = coefnames(result)
    name_str = string(name)
    idx = findfirst(==(name_str), names)
    return idx
end

"""
    _safe_ols_coef(y, X, coef_index) -> Float64

Run OLS and return the specified coefficient. Returns NaN if regression fails
(e.g., singular X'X due to zero-variance regressors).

Kept for backward compatibility and testing.
"""
function _safe_ols_coef(y::Vector{Float64}, X::Matrix{Float64}, coef_index::Int)
    try
        # Check for zero variance in dependent variable
        if std(y) < 1e-15
            return NaN
        end
        beta = ols_coefficients(y, X)
        result = beta[coef_index]
        return isfinite(result) ? result : NaN
    catch e
        if e isa SingularException || e isa LAPACKException
            return NaN
        end
        rethrow(e)
    end
end
