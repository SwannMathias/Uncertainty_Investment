"""
Moment computation for SMM estimation.

# Economic context
The estimation matches 4 simulated moments to empirical targets:
1. Share of zero beginning-of-year investment revisions
2. Share of zero mid-year investment revisions
3. OLS coefficient of beginning-of-year revision on log uncertainty
4. OLS coefficient of mid-year revision on log uncertainty

Investment revisions measure how firms update their investment plans
as new information arrives within the year.
"""

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
For the 4-regressor case in moment computation, this is trivially fast.
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
    compute_simulated_moments(panel_df::DataFrame, config::SMMConfig) -> Vector{Float64}

Compute 4 simulated moments from a firm panel.

# Moments
1. Share of zero beginning-of-year revisions
2. Share of zero mid-year revisions
3. OLS coefficient of beginning-of-year revision on log(sigma)
4. OLS coefficient of mid-year revision on log(sigma_half)

# Returns
- 4-element vector of simulated moments. Returns fill(NaN, 4) if regressions fail.

# Implementation notes
- Year 1 observations are dropped (E_last_semester is NaN).
- Rows with NaN revisions (from log transform of non-positive values) are dropped.
- If >50% of observations are dropped by the log transform, a warning is logged.
"""
function compute_simulated_moments(panel_df::DataFrame, config::SMMConfig)
    # Step 1: Compute revisions
    df = compute_revision_panel(panel_df, config)

    # Step 2: Filter — drop year 1 (E_last_semester is NaN) and NaN revisions
    valid_begin = .!isnan.(df.revision_begin) .& .!isnan.(df.E_last_semester)
    valid_mid = .!isnan.(df.revision_mid)

    n_total = nrow(df)
    n_valid_begin = sum(valid_begin)
    n_valid_mid = sum(valid_mid)

    # Warn if too many observations dropped (suggests bad parameter region)
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

    # Check we have enough observations for regressions
    if n_valid_begin < 10 || n_valid_mid < 10
        @warn "Too few valid observations for moment computation " *
              "(begin: $n_valid_begin, mid: $n_valid_mid). Returning NaN."
        return fill(NaN, 4)
    end

    # Step 3: Share-of-zero moments
    m1 = mean(abs.(df.revision_begin[valid_begin]) .< config.zero_threshold)
    m2 = mean(abs.(df.revision_mid[valid_mid]) .< config.zero_threshold)

    # Step 4: Regression — beginning-of-year revision on log(sigma)
    # revision_begin_{i,t} = alpha + beta_1 log(sigma_t) + beta_2 log(K_t) + beta_3 log(D_t) + eps
    y_begin = df.revision_begin[valid_begin]
    X_begin = hcat(
        ones(n_valid_begin),
        df.log_sigma[valid_begin],
        log.(df.K[valid_begin]),
        df.log_D[valid_begin]
    )

    m3 = _safe_ols_coef(y_begin, X_begin, 2)

    # Step 5: Regression — mid-year revision on log(sigma_half)
    # revision_mid_{i,t} = alpha + beta_1 log(sigma_{t+1/2}) + beta_2 log(K_t) + beta_3 log(D_t) + eps
    y_mid = df.revision_mid[valid_mid]
    X_mid = hcat(
        ones(n_valid_mid),
        df.log_sigma_half[valid_mid],
        log.(df.K[valid_mid]),
        df.log_D[valid_mid]
    )

    m4 = _safe_ols_coef(y_mid, X_mid, 2)

    return [m1, m2, m3, m4]
end

"""
    _safe_ols_coef(y, X, coef_index) -> Float64

Run OLS and return the specified coefficient. Returns NaN if regression fails
(e.g., singular X'X due to zero-variance regressors).
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
