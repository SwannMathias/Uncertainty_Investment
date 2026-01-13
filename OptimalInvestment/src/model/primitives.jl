"""
Primitive economic functions: profit, marginal products, etc.

The profit function is derived from iso-elastic demand and Cobb-Douglas production:
    π(K, D) = (h / (1-γ)) * D^γ * K^(1-γ)

where γ and h are derived from structural parameters α and ε.
"""

"""
    profit(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute per-semester operating profit.

π(K, D) = (h / (1-γ)) * D^γ * K^(1-γ)

# Arguments
- `K`: Capital stock
- `D`: Demand level (not log demand)
- `derived`: DerivedParameters containing γ and h

# Returns
- Operating profit (> 0 for K, D > 0)
"""
function profit(K::Float64, D::Float64, derived::DerivedParameters)
    @assert K > 0.0 "Capital must be positive"
    @assert D > 0.0 "Demand must be positive"

    γ = derived.γ
    h = derived.h

    return (h / (1 - γ)) * D^γ * K^(1 - γ)
end

"""
    marginal_product_capital(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute marginal product of capital: ∂π/∂K.

MPK = h * D^γ * K^(-γ)

# Arguments
- `K`: Capital stock
- `D`: Demand level
- `derived`: DerivedParameters

# Returns
- Marginal product of capital (> 0)
"""
function marginal_product_capital(K::Float64, D::Float64, derived::DerivedParameters)
    @assert K > 0.0 "Capital must be positive"
    @assert D > 0.0 "Demand must be positive"

    γ = derived.γ
    h = derived.h

    return h * D^γ * K^(-γ)
end

"""
    profit_derivative_K(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Alias for marginal_product_capital. Returns ∂π/∂K.
"""
profit_derivative_K(K::Float64, D::Float64, derived::DerivedParameters) =
    marginal_product_capital(K, D, derived)

"""
    profit_derivative_D(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute marginal product of demand: ∂π/∂D.

∂π/∂D = (h * γ / (1-γ)) * D^(γ-1) * K^(1-γ)

# Arguments
- `K`: Capital stock
- `D`: Demand level
- `derived`: DerivedParameters

# Returns
- Marginal effect of demand on profit (> 0)
"""
function profit_derivative_D(K::Float64, D::Float64, derived::DerivedParameters)
    @assert K > 0.0 "Capital must be positive"
    @assert D > 0.0 "Demand must be positive"

    γ = derived.γ
    h = derived.h

    return (h * γ / (1 - γ)) * D^(γ - 1) * K^(1 - γ)
end

"""
    profit_second_derivative_K(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute second derivative of profit w.r.t. capital: ∂²π/∂K².

∂²π/∂K² = -γ * h * D^γ * K^(-γ-1) < 0

This is negative, confirming concavity in K.

# Arguments
- `K`: Capital stock
- `D`: Demand level
- `derived`: DerivedParameters

# Returns
- Second derivative (< 0, diminishing returns)
"""
function profit_second_derivative_K(K::Float64, D::Float64, derived::DerivedParameters)
    @assert K > 0.0 "Capital must be positive"
    @assert D > 0.0 "Demand must be positive"

    γ = derived.γ
    h = derived.h

    return -γ * h * D^γ * K^(-γ - 1)
end

"""
    annual_profit(K::Float64, D_first::Float64, D_second::Float64,
                  derived::DerivedParameters;
                  aggregation::Symbol=:sum) -> Float64

Compute annual profit from two semesters.

# Arguments
- `K`: Capital stock (assumed constant within year)
- `D_first`: First semester demand
- `D_second`: Second semester demand
- `derived`: DerivedParameters
- `aggregation`: How to aggregate semesters (:sum, :mean, :geometric_mean)

# Returns
- Annual profit
"""
function annual_profit(K::Float64, D_first::Float64, D_second::Float64,
                       derived::DerivedParameters;
                       aggregation::Symbol=:sum)
    π1 = profit(K, D_first, derived)
    π2 = profit(K, D_second, derived)

    if aggregation == :sum
        return π1 + π2
    elseif aggregation == :mean
        return (π1 + π2) / 2
    elseif aggregation == :geometric_mean
        return sqrt(π1 * π2)
    else
        error("Unknown aggregation method: $aggregation")
    end
end

"""
    optimal_capital_static(D::Float64, user_cost::Float64, derived::DerivedParameters) -> Float64

Compute optimal capital in static problem: max_K π(K,D) - user_cost * K.

From FOC: MPK = user_cost
=> h * D^γ * K^(-γ) = user_cost
=> K* = (h * D^γ / user_cost)^(1/γ)

# Arguments
- `D`: Demand level
- `user_cost`: User cost of capital (typically δ/β in steady state)
- `derived`: DerivedParameters

# Returns
- Optimal capital stock
"""
function optimal_capital_static(D::Float64, user_cost::Float64, derived::DerivedParameters)
    @assert D > 0.0 "Demand must be positive"
    @assert user_cost > 0.0 "User cost must be positive"

    γ = derived.γ
    h = derived.h

    return (h * D^γ / user_cost)^(1 / γ)
end

"""
    profit_elasticity_K(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute elasticity of profit w.r.t. capital: (K/π) * (∂π/∂K).

# Returns
- Elasticity (should equal 1 - γ)
"""
function profit_elasticity_K(K::Float64, D::Float64, derived::DerivedParameters)
    π = profit(K, D, derived)
    mpk = marginal_product_capital(K, D, derived)
    return (K / π) * mpk
end

"""
    profit_elasticity_D(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute elasticity of profit w.r.t. demand: (D/π) * (∂π/∂D).

# Returns
- Elasticity (should equal γ)
"""
function profit_elasticity_D(K::Float64, D::Float64, derived::DerivedParameters)
    π = profit(K, D, derived)
    mpd = profit_derivative_D(K, D, derived)
    return (D / π) * mpd
end

"""
    check_profit_properties(derived::DerivedParameters; K_test=1.0, D_test=1.0) -> Bool

Verify analytical properties of profit function:
1. Positive for K, D > 0
2. Increasing in K and D
3. Concave in K
4. Correct elasticities

# Returns
- true if all checks pass
"""
function check_profit_properties(derived::DerivedParameters; K_test=1.0, D_test=1.0)
    all_pass = true

    # 1. Positivity
    π = profit(K_test, D_test, derived)
    if π <= 0
        @warn "Profit is not positive: π = $π"
        all_pass = false
    end

    # 2. Increasing in K
    mpk = marginal_product_capital(K_test, D_test, derived)
    if mpk <= 0
        @warn "MPK is not positive: MPK = $mpk"
        all_pass = false
    end

    # 3. Concave in K
    d2π_dK2 = profit_second_derivative_K(K_test, D_test, derived)
    if d2π_dK2 >= 0
        @warn "Profit is not concave in K: ∂²π/∂K² = $d2π_dK2"
        all_pass = false
    end

    # 4. Elasticity w.r.t. K should equal 1 - γ
    ε_K = profit_elasticity_K(K_test, D_test, derived)
    expected_ε_K = 1 - derived.γ
    if !isapprox(ε_K, expected_ε_K, rtol=1e-6)
        @warn "Capital elasticity incorrect: got $ε_K, expected $expected_ε_K"
        all_pass = false
    end

    # 5. Elasticity w.r.t. D should equal γ
    ε_D = profit_elasticity_D(K_test, D_test, derived)
    expected_ε_D = derived.γ
    if !isapprox(ε_D, expected_ε_D, rtol=1e-6)
        @warn "Demand elasticity incorrect: got $ε_D, expected $expected_ε_D"
        all_pass = false
    end

    if all_pass
        println("✓ All profit function properties verified")
    end

    return all_pass
end
