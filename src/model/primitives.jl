"""
Primitive economic functions: profit, marginal products, etc.

The profit function is derived from iso-elastic demand and Cobb-Douglas production:
    pi(K, D) = (h / (1-gamma)) * D^gamma * K^(1-gamma)

where gamma and h are derived from structural parameters alpha and epsilon.
"""

"""
    profit(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute per-semester operating profit.

pi(K, D) = (h / (1-gamma)) * D^gamma * K^(1-gamma)

# Arguments
- `K`: Capital stock
- `D`: Demand level (not log demand)
- `derived`: DerivedParameters containing gamma and h

# Returns
- Operating profit (> 0 for K, D > 0)
"""
function profit(K::Float64, D::Float64, derived::DerivedParameters)
    @assert K > 0.0 "Capital must be positive"
    @assert D > 0.0 "Demand must be positive"

    gamma = derived.gamma
    h = derived.h

    return (h / (1 - gamma)) * D^gamma * K^(1 - gamma)
end

"""
    marginal_product_capital(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute marginal product of capital: ∂pi/∂K.

MPK = h * D^gamma * K^(-gamma)

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

    gamma = derived.gamma
    h = derived.h

    return h * D^gamma * K^(-gamma)
end

"""
    profit_derivative_K(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Alias for marginal_product_capital. Returns ∂pi/∂K.
"""
profit_derivative_K(K::Float64, D::Float64, derived::DerivedParameters) =
    marginal_product_capital(K, D, derived)

"""
    profit_derivative_D(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute marginal product of demand: ∂pi/∂D.

∂pi/∂D = (h * gamma / (1-gamma)) * D^(gamma-1) * K^(1-gamma)

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

    gamma = derived.gamma
    h = derived.h

    return (h * gamma / (1 - gamma)) * D^(gamma - 1) * K^(1 - gamma)
end

"""
    profit_second_derivative_K(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute second derivative of profit w.r.t. capital: ∂²pi/∂K².

∂²pi/∂K² = -gamma * h * D^gamma * K^(-gamma-1) < 0

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

    gamma = derived.gamma
    h = derived.h

    return -gamma * h * D^gamma * K^(-gamma - 1)
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
    pi1 = profit(K, D_first, derived)
    pi2 = profit(K, D_second, derived)

    if aggregation == :sum
        return pi1 + pi2
    elseif aggregation == :mean
        return (pi1 + pi2) / 2
    elseif aggregation == :geometric_mean
        return sqrt(pi1 * pi2)
    else
        error("Unknown aggregation method: $aggregation")
    end
end

"""
    optimal_capital_static(D::Float64, user_cost::Float64, derived::DerivedParameters) -> Float64

Compute optimal capital in static problem: max_K pi(K,D) - user_cost * K.

From FOC: MPK = user_cost
=> h * D^gamma * K^(-gamma) = user_cost
=> K* = (h * D^gamma / user_cost)^(1/gamma)

# Arguments
- `D`: Demand level
- `user_cost`: User cost of capital (typically delta/beta in steady state)
- `derived`: DerivedParameters

# Returns
- Optimal capital stock
"""
function optimal_capital_static(D::Float64, user_cost::Float64, derived::DerivedParameters)
    @assert D > 0.0 "Demand must be positive"
    @assert user_cost > 0.0 "User cost must be positive"

    gamma = derived.gamma
    h = derived.h

    return (h * D^gamma / user_cost)^(1 / gamma)
end

"""
    profit_elasticity_K(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute elasticity of profit w.r.t. capital: (K/pi) * (∂pi/∂K).

# Returns
- Elasticity (should equal 1 - gamma)
"""
function profit_elasticity_K(K::Float64, D::Float64, derived::DerivedParameters)
    pi = profit(K, D, derived)
    mpk = marginal_product_capital(K, D, derived)
    return (K / pi) * mpk
end

"""
    profit_elasticity_D(K::Float64, D::Float64, derived::DerivedParameters) -> Float64

Compute elasticity of profit w.r.t. demand: (D/pi) * (∂pi/∂D).

# Returns
- Elasticity (should equal gamma)
"""
function profit_elasticity_D(K::Float64, D::Float64, derived::DerivedParameters)
    pi = profit(K, D, derived)
    mpd = profit_derivative_D(K, D, derived)
    return (D / pi) * mpd
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
    pi = profit(K_test, D_test, derived)
    if pi <= 0
        @warn "Profit is not positive: pi = $pi"
        all_pass = false
    end

    # 2. Increasing in K
    mpk = marginal_product_capital(K_test, D_test, derived)
    if mpk <= 0
        @warn "MPK is not positive: MPK = $mpk"
        all_pass = false
    end

    # 3. Concave in K
    d2pi_dK2 = profit_second_derivative_K(K_test, D_test, derived)
    if d2pi_dK2 >= 0
        @warn "Profit is not concave in K: ∂²pi/∂K² = $d2pi_dK2"
        all_pass = false
    end

    # 4. Elasticity w.r.t. K should equal 1 - gamma
    epsilon_K = profit_elasticity_K(K_test, D_test, derived)
    expected_epsilon_K = 1 - derived.gamma
    if !isapprox(epsilon_K, expected_epsilon_K, rtol=1e-6)
        @warn "Capital elasticity incorrect: got $epsilon_K, expected $expected_epsilon_K"
        all_pass = false
    end

    # 5. Elasticity w.r.t. D should equal gamma
    epsilon_D = profit_elasticity_D(K_test, D_test, derived)
    expected_epsilon_D = derived.gamma
    if !isapprox(epsilon_D, expected_epsilon_D, rtol=1e-6)
        @warn "Demand elasticity incorrect: got $epsilon_D, expected $expected_epsilon_D"
        all_pass = false
    end

    if all_pass
        println("✓ All profit function properties verified")
    end

    return all_pass
end
