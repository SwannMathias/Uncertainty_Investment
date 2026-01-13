"""
Flexible menu of adjustment cost specifications.

All adjustment cost types implement the AbstractAdjustmentCost interface:
- compute_cost(ac, I, ΔI, K): Total adjustment cost
- marginal_cost_I(ac, I, ΔI, K): ∂C/∂I
- marginal_cost_ΔI(ac, I, ΔI, K): ∂C/∂ΔI
- has_fixed_cost(ac): Whether cost function has discontinuity
- is_differentiable(ac): Whether cost function is differentiable everywhere
"""

abstract type AbstractAdjustmentCost end

# =============================================================================
# Interface functions (all types must implement)
# =============================================================================

"""
    compute_cost(ac::AbstractAdjustmentCost, I, ΔI, K) -> Float64

Compute total adjustment cost for investment I (initial) and ΔI (revision).
"""
function compute_cost end

"""
    marginal_cost_I(ac::AbstractAdjustmentCost, I, ΔI, K) -> Float64

Compute marginal adjustment cost w.r.t. initial investment I: ∂C/∂I.
"""
function marginal_cost_I end

"""
    marginal_cost_ΔI(ac::AbstractAdjustmentCost, I, ΔI, K) -> Float64

Compute marginal adjustment cost w.r.t. investment revision ΔI: ∂C/∂ΔI.
"""
function marginal_cost_ΔI end

"""
    has_fixed_cost(ac::AbstractAdjustmentCost) -> Bool

Returns true if adjustment cost has a fixed component (discontinuity at zero).
"""
function has_fixed_cost end

"""
    is_differentiable(ac::AbstractAdjustmentCost) -> Bool

Returns true if cost function is differentiable everywhere.
"""
function is_differentiable end

# =============================================================================
# 1. No Adjustment Cost
# =============================================================================

struct NoAdjustmentCost <: AbstractAdjustmentCost end

compute_cost(::NoAdjustmentCost, I, ΔI, K) = 0.0
marginal_cost_I(::NoAdjustmentCost, I, ΔI, K) = 0.0
marginal_cost_ΔI(::NoAdjustmentCost, I, ΔI, K) = 0.0
has_fixed_cost(::NoAdjustmentCost) = false
is_differentiable(::NoAdjustmentCost) = true

# =============================================================================
# 2. Convex Adjustment Cost (Standard)
# =============================================================================

"""
    ConvexAdjustmentCost

Standard quadratic adjustment cost on total investment:
C(I, ΔI, K) = (ϕ/2) * ((I + ΔI) / K)^2 * K
"""
@with_kw struct ConvexAdjustmentCost <: AbstractAdjustmentCost
    ϕ::Float64 = 1.0

    function ConvexAdjustmentCost(ϕ)
        @assert ϕ >= 0.0 "ϕ must be non-negative"
        new(ϕ)
    end
end

function compute_cost(ac::ConvexAdjustmentCost, I, ΔI, K)
    I_total = I + ΔI
    return 0.5 * ac.ϕ * (I_total / K)^2 * K
end

function marginal_cost_I(ac::ConvexAdjustmentCost, I, ΔI, K)
    I_total = I + ΔI
    return ac.ϕ * (I_total / K)
end

function marginal_cost_ΔI(ac::ConvexAdjustmentCost, I, ΔI, K)
    I_total = I + ΔI
    return ac.ϕ * (I_total / K)
end

has_fixed_cost(::ConvexAdjustmentCost) = false
is_differentiable(::ConvexAdjustmentCost) = true

# =============================================================================
# 3. Separate Convex Costs (Initial vs Revision)
# =============================================================================

"""
    SeparateConvexCost

Separate quadratic costs for initial investment and revision:
C(I, ΔI, K) = (ϕ₁/2) * (I/K)^2 * K + (ϕ₂/2) * (ΔI/K)^2 * K
"""
@with_kw struct SeparateConvexCost <: AbstractAdjustmentCost
    ϕ₁::Float64 = 1.0   # Initial investment cost
    ϕ₂::Float64 = 1.0   # Revision cost

    function SeparateConvexCost(ϕ₁, ϕ₂)
        @assert ϕ₁ >= 0.0 "ϕ₁ must be non-negative"
        @assert ϕ₂ >= 0.0 "ϕ₂ must be non-negative"
        new(ϕ₁, ϕ₂)
    end
end

function compute_cost(ac::SeparateConvexCost, I, ΔI, K)
    cost_I = 0.5 * ac.ϕ₁ * (I / K)^2 * K
    cost_ΔI = 0.5 * ac.ϕ₂ * (ΔI / K)^2 * K
    return cost_I + cost_ΔI
end

function marginal_cost_I(ac::SeparateConvexCost, I, ΔI, K)
    return ac.ϕ₁ * (I / K)
end

function marginal_cost_ΔI(ac::SeparateConvexCost, I, ΔI, K)
    return ac.ϕ₂ * (ΔI / K)
end

has_fixed_cost(::SeparateConvexCost) = false
is_differentiable(::SeparateConvexCost) = true

# =============================================================================
# 4. Fixed Adjustment Cost
# =============================================================================

"""
    FixedAdjustmentCost

Fixed cost paid whenever total investment is non-zero:
C(I, ΔI, K) = F * 𝟙{I + ΔI ≠ 0}
"""
@with_kw struct FixedAdjustmentCost <: AbstractAdjustmentCost
    F::Float64 = 0.1
    threshold::Float64 = 1e-6  # Threshold for "zero" investment

    function FixedAdjustmentCost(F, threshold)
        @assert F >= 0.0 "F must be non-negative"
        @assert threshold > 0.0 "threshold must be positive"
        new(F, threshold)
    end
end

function compute_cost(ac::FixedAdjustmentCost, I, ΔI, K)
    I_total = I + ΔI
    return abs(I_total) > ac.threshold ? ac.F : 0.0
end

function marginal_cost_I(ac::FixedAdjustmentCost, I, ΔI, K)
    # Marginal cost is zero except at discontinuity
    return 0.0
end

function marginal_cost_ΔI(ac::FixedAdjustmentCost, I, ΔI, K)
    return 0.0
end

has_fixed_cost(::FixedAdjustmentCost) = true
is_differentiable(::FixedAdjustmentCost) = false

# =============================================================================
# 5. Asymmetric Adjustment Cost
# =============================================================================

"""
    AsymmetricAdjustmentCost

Different convex costs for positive vs negative net investment:
C(I, ΔI, K) = ϕ⁺ * (I_total^+)^2 / K + ϕ⁻ * (I_total^-)^2 / K

where I_total^+ = max(I_total, 0) and I_total^- = max(-I_total, 0).
"""
@with_kw struct AsymmetricAdjustmentCost <: AbstractAdjustmentCost
    ϕ_plus::Float64 = 1.0   # Cost for expansion
    ϕ_minus::Float64 = 2.0  # Cost for contraction (typically higher)

    function AsymmetricAdjustmentCost(ϕ_plus, ϕ_minus)
        @assert ϕ_plus >= 0.0 "ϕ_plus must be non-negative"
        @assert ϕ_minus >= 0.0 "ϕ_minus must be non-negative"
        new(ϕ_plus, ϕ_minus)
    end
end

function compute_cost(ac::AsymmetricAdjustmentCost, I, ΔI, K)
    I_total = I + ΔI

    if I_total > 0
        return ac.ϕ_plus * I_total^2 / K
    else
        return ac.ϕ_minus * I_total^2 / K
    end
end

function marginal_cost_I(ac::AsymmetricAdjustmentCost, I, ΔI, K)
    I_total = I + ΔI

    if I_total > 0
        return 2 * ac.ϕ_plus * I_total / K
    else
        return 2 * ac.ϕ_minus * I_total / K
    end
end

function marginal_cost_ΔI(ac::AsymmetricAdjustmentCost, I, ΔI, K)
    # Same as marginal_cost_I since both affect I_total
    return marginal_cost_I(ac, I, ΔI, K)
end

has_fixed_cost(::AsymmetricAdjustmentCost) = false
is_differentiable(::AsymmetricAdjustmentCost) = false  # Kink at zero

# =============================================================================
# 6. Partial Irreversibility
# =============================================================================

"""
    PartialIrreversibility

Capital can be sold but at fraction p_S < 1 of purchase price:
C(I, ΔI, K) = -(1 - p_S) * max(-(I + ΔI), 0)

This creates an asymmetry: selling capital is costly.
"""
@with_kw struct PartialIrreversibility <: AbstractAdjustmentCost
    p_S::Float64 = 0.8  # Resale price as fraction of purchase price

    function PartialIrreversibility(p_S)
        @assert 0.0 <= p_S <= 1.0 "p_S must be in [0, 1]"
        new(p_S)
    end
end

function compute_cost(ac::PartialIrreversibility, I, ΔI, K)
    I_total = I + ΔI

    if I_total < 0
        # Selling: lose (1 - p_S) fraction
        return -(1 - ac.p_S) * I_total
    else
        return 0.0
    end
end

function marginal_cost_I(ac::PartialIrreversibility, I, ΔI, K)
    I_total = I + ΔI

    if I_total < 0
        return -(1 - ac.p_S)
    else
        return 0.0
    end
end

function marginal_cost_ΔI(ac::PartialIrreversibility, I, ΔI, K)
    return marginal_cost_I(ac, I, ΔI, K)
end

has_fixed_cost(::PartialIrreversibility) = false
is_differentiable(::PartialIrreversibility) = false  # Kink at zero

# =============================================================================
# 7. Composite Adjustment Cost
# =============================================================================

"""
    CompositeAdjustmentCost

Sum of multiple adjustment cost components.
Example: Fixed cost + Convex cost
"""
struct CompositeAdjustmentCost <: AbstractAdjustmentCost
    components::Vector{AbstractAdjustmentCost}

    function CompositeAdjustmentCost(components::Vector{<:AbstractAdjustmentCost})
        @assert length(components) > 0 "Must have at least one component"
        new(components)
    end
end

# Convenience constructor
CompositeAdjustmentCost(components::AbstractAdjustmentCost...) =
    CompositeAdjustmentCost(collect(components))

function compute_cost(ac::CompositeAdjustmentCost, I, ΔI, K)
    return sum(compute_cost(c, I, ΔI, K) for c in ac.components)
end

function marginal_cost_I(ac::CompositeAdjustmentCost, I, ΔI, K)
    return sum(marginal_cost_I(c, I, ΔI, K) for c in ac.components)
end

function marginal_cost_ΔI(ac::CompositeAdjustmentCost, I, ΔI, K)
    return sum(marginal_cost_ΔI(c, I, ΔI, K) for c in ac.components)
end

function has_fixed_cost(ac::CompositeAdjustmentCost)
    return any(has_fixed_cost(c) for c in ac.components)
end

function is_differentiable(ac::CompositeAdjustmentCost)
    return all(is_differentiable(c) for c in ac.components)
end

# =============================================================================
# Utility functions
# =============================================================================

"""
    describe_adjustment_cost(ac::AbstractAdjustmentCost) -> String

Return a human-readable description of the adjustment cost specification.
"""
function describe_adjustment_cost(ac::AbstractAdjustmentCost)
    if ac isa NoAdjustmentCost
        return "No adjustment costs"
    elseif ac isa ConvexAdjustmentCost
        return "Convex: ($(ac.ϕ)/2) * (I_total/K)²"
    elseif ac isa SeparateConvexCost
        return "Separate convex: ϕ₁=$(ac.ϕ₁) (initial), ϕ₂=$(ac.ϕ₂) (revision)"
    elseif ac isa FixedAdjustmentCost
        return "Fixed cost: F=$(ac.F)"
    elseif ac isa AsymmetricAdjustmentCost
        return "Asymmetric: ϕ⁺=$(ac.ϕ_plus), ϕ⁻=$(ac.ϕ_minus)"
    elseif ac isa PartialIrreversibility
        return "Partial irreversibility: resale price=$(ac.p_S)"
    elseif ac isa CompositeAdjustmentCost
        desc = "Composite: " * join([describe_adjustment_cost(c) for c in ac.components], " + ")
        return desc
    else
        return "Custom adjustment cost"
    end
end

"""
    total_adjustment_cost(ac, I, ΔI, K) -> Float64

Alias for compute_cost for readability.
"""
total_adjustment_cost(ac, I, ΔI, K) = compute_cost(ac, I, ΔI, K)
