"""
Numerical utility functions for optimization, root-finding, and numerical stability.
"""

using LinearAlgebra
using Optim

"""
    golden_section_search(f, a, b; tol=1e-6, max_iter=1000) -> (x_opt, f_opt)

Find minimum of univariate function f on interval [a, b] using golden section search.

# Arguments
- `f`: Function to minimize
- `a`: Lower bound
- `b`: Upper bound
- `tol`: Tolerance for convergence
- `max_iter`: Maximum iterations

# Returns
- `x_opt`: Optimal point
- `f_opt`: Function value at optimum
"""
function golden_section_search(f, a, b; tol=1e-6, max_iter=1000)
    phi = (1 + sqrt(5)) / 2  # Golden ratio
    resphi = 2 - phi

    # Initial points
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    for iter in 1:max_iter
        if f1 < f2
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)
        end

        if abs(b - a) < tol
            x_opt = (a + b) / 2
            return x_opt, f(x_opt)
        end
    end

    @warn "Golden section search did not converge"
    x_opt = (a + b) / 2
    return x_opt, f(x_opt)
end

"""
    maximize_univariate(f, a, b; method=:golden, tol=1e-6) -> (x_opt, f_opt)

Maximize univariate function f on interval [a, b].

# Arguments
- `f`: Function to maximize
- `a`: Lower bound
- `b`: Upper bound
- `method`: :golden (golden section) or :brent (Brent's method via Optim.jl)
- `tol`: Tolerance

# Returns
- `x_opt`: Optimal point
- `f_opt`: Function value at optimum
"""
function maximize_univariate(f, a, b; method=:golden, tol=1e-6)
    # Maximize f is equivalent to minimize -f
    if method == :golden
        x_opt, neg_f_opt = golden_section_search(x -> -f(x), a, b; tol=tol)
        return x_opt, -neg_f_opt
    elseif method == :brent
        result = optimize(x -> -f(x), a, b, Brent(); abs_tol=tol)
        return Optim.minimizer(result), -Optim.minimum(result)
    else
        error("Unknown method: $method")
    end
end

"""
    minimize_univariate(f, a, b; method=:golden, tol=1e-6) -> (x_opt, f_opt)

Minimize univariate function f on interval [a, b].
"""
function minimize_univariate(f, a, b; method=:golden, tol=1e-6)
    if method == :golden
        return golden_section_search(f, a, b; tol=tol)
    elseif method == :brent
        result = optimize(f, a, b, Brent(); abs_tol=tol)
        return Optim.minimizer(result), Optim.minimum(result)
    else
        error("Unknown method: $method")
    end
end

"""
    log_sum_exp(x::Vector{Float64}) -> Float64

Compute log(sum(exp.(x))) in numerically stable way.
"""
function log_sum_exp(x::Vector{Float64})
    x_max = maximum(x)
    return x_max + log(sum(exp.(x .- x_max)))
end

"""
    softmax(x::Vector{Float64}) -> Vector{Float64}

Compute softmax in numerically stable way.
"""
function softmax(x::Vector{Float64})
    x_shifted = x .- maximum(x)
    exp_x = exp.(x_shifted)
    return exp_x ./ sum(exp_x)
end

"""
    check_convergence(x_new, x_old; atol=1e-6, rtol=1e-6) -> Bool

Check convergence using both absolute and relative tolerance.

Convergence if: |x_new - x_old| < atol + rtol * |x_old|
"""
function check_convergence(x_new, x_old; atol=1e-6, rtol=1e-6)
    diff = maximum(abs.(x_new .- x_old))
    threshold = atol + rtol * maximum(abs.(x_old))
    return diff < threshold
end

"""
    check_convergence_policy(policy_new, policy_old; tol=1e-6) -> Bool

Check policy function convergence.
"""
function check_convergence_policy(policy_new, policy_old; tol=1e-6)
    return maximum(abs.(policy_new .- policy_old)) < tol
end

"""
    relative_difference(x_new, x_old) -> Float64

Compute maximum relative difference: max|(x_new - x_old) / x_old|.
"""
function relative_difference(x_new, x_old)
    # Avoid division by zero
    denominator = max.(abs.(x_old), 1e-10)
    return maximum(abs.((x_new .- x_old) ./ denominator))
end

"""
    safe_log(x::Float64; min_val=1e-300) -> Float64

Compute log(x) with protection against log(0).
"""
safe_log(x::Float64; min_val=1e-300) = log(max(x, min_val))

"""
    safe_exp(x::Float64; max_val=700.0) -> Float64

Compute exp(x) with protection against overflow.
"""
safe_exp(x::Float64; max_val=700.0) = exp(min(x, max_val))

"""
    clamp_to_range(x::Float64, lower::Float64, upper::Float64) -> Float64

Clamp x to [lower, upper].
"""
clamp_to_range(x::Float64, lower::Float64, upper::Float64) = clamp(x, lower, upper)

"""
    linspace(start::Float64, stop::Float64, n::Int) -> Vector{Float64}

Create linearly spaced vector (Julia 1.x compatible).
"""
linspace(start::Float64, stop::Float64, n::Int) = collect(range(start, stop, length=n))

"""
    logspace(start::Float64, stop::Float64, n::Int) -> Vector{Float64}

Create logarithmically spaced vector.
"""
function logspace(start::Float64, stop::Float64, n::Int)
    return exp.(linspace(log(start), log(stop), n))
end

"""
    grid_K_optimal(K_min, K_max, n_K; curvature=1.0) -> Vector{Float64}

Create capital grid with flexible spacing.

- curvature = 1.0: Linear spacing
- curvature > 1.0: More points near K_min
- curvature < 1.0: More points near K_max
"""
function grid_K_optimal(K_min, K_max, n_K; curvature=1.0)
    if curvature ≈ 1.0
        return linspace(K_min, K_max, n_K)
    else
        t = linspace(0.0, 1.0, n_K)
        return K_min .+ (K_max - K_min) .* t.^curvature
    end
end

"""
    bisection(f, a, b; tol=1e-6, max_iter=100) -> Float64

Find root of f using bisection method.

# Arguments
- `f`: Function with f(a) and f(b) having opposite signs
- `a`: Lower bound
- `b`: Upper bound
- `tol`: Tolerance
- `max_iter`: Maximum iterations

# Returns
- Root x such that f(x) ≈ 0
"""
function bisection(f, a, b; tol=1e-6, max_iter=100)
    fa = f(a)
    fb = f(b)

    @assert fa * fb < 0 "f(a) and f(b) must have opposite signs"

    for iter in 1:max_iter
        c = (a + b) / 2
        fc = f(c)

        if abs(fc) < tol || abs(b - a) < tol
            return c
        end

        if fa * fc < 0
            b = c
            fb = fc
        else
            a = c
            fa = fc
        end
    end

    @warn "Bisection did not converge"
    return (a + b) / 2
end

"""
    is_monotonic_increasing(x::Vector{Float64}; strict=false) -> Bool

Check if vector is monotonically increasing.
"""
function is_monotonic_increasing(x::Vector{Float64}; strict=false)
    if strict
        return all(x[i] < x[i+1] for i in 1:length(x)-1)
    else
        return all(x[i] <= x[i+1] for i in 1:length(x)-1)
    end
end

"""
    is_valid_probability_matrix(P::Matrix{Float64}; tol=1e-10) -> Bool

Check if P is a valid transition probability matrix.
"""
function is_valid_probability_matrix(P::Matrix{Float64}; tol=1e-10)
    n, m = size(P)

    # Must be square
    if n != m
        return false
    end

    # All entries non-negative
    if any(P .< -tol)
        return false
    end

    # Rows sum to 1
    row_sums = sum(P, dims=2)
    if !all(isapprox.(row_sums, 1.0, atol=tol))
        return false
    end

    return true
end

"""
    condition_number(A::Matrix{Float64}) -> Float64

Compute condition number of matrix A.
"""
condition_number(A::Matrix{Float64}) = cond(A)

"""
    format_time(seconds::Float64) -> String

Format elapsed time in human-readable format.
"""
function format_time(seconds::Float64)
    if seconds < 60
        return @sprintf("%.2f seconds", seconds)
    elseif seconds < 3600
        minutes = floor(seconds / 60)
        secs = seconds - 60 * minutes
        return @sprintf("%.0f min %.0f sec", minutes, secs)
    else
        hours = floor(seconds / 3600)
        minutes = floor((seconds - 3600 * hours) / 60)
        return @sprintf("%.0f hr %.0f min", hours, minutes)
    end
end

"""
    format_number(x::Float64; digits=4) -> String

Format number for display.
"""
function format_number(x::Float64; digits=4)
    if abs(x) < 1e-10
        return "0.0"
    elseif abs(x) >= 1e4 || abs(x) < 1e-3
        # Use Printf.Format for runtime format string construction
        fmt = Printf.Format("%.$(digits)e")
        return Printf.format(fmt, x)
    else
        # Use Printf.Format for runtime format string construction
        fmt = Printf.Format("%.$(digits)f")
        return Printf.format(fmt, x)
    end
end
