"""
Flexible menu of adjustment cost specifications.

All adjustment cost types implement the AbstractAdjustmentCost interface:
- compute_cost(ac, I, Delta_I, K): Total adjustment cost
- marginal_cost_I(ac, I, Delta_I, K): ∂C/∂I
- marginal_cost_Delta_I(ac, I, Delta_I, K): ∂C/∂Delta_I
- has_fixed_cost(ac): Whether cost function has discontinuity
- is_differentiable(ac): Whether cost function is differentiable everywhere
"""

abstract type AbstractAdjustmentCost end

# =============================================================================
# Interface functions (all types must implement)
# =============================================================================

"""
    compute_cost(ac::AbstractAdjustmentCost, I, Delta_I, K) -> Float64

Compute total adjustment cost for investment I (initial) and Delta_I (revision).
"""
function compute_cost end

"""
    marginal_cost_I(ac::AbstractAdjustmentCost, I, Delta_I, K) -> Float64

Compute marginal adjustment cost w.r.t. initial investment I: ∂C/∂I.
"""
function marginal_cost_I end

"""
    marginal_cost_Delta_I(ac::AbstractAdjustmentCost, I, Delta_I, K) -> Float64

Compute marginal adjustment cost w.r.t. investment revision Delta_I: ∂C/∂Delta_I.
"""
function marginal_cost_Delta_I end

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

compute_cost(::NoAdjustmentCost, I, Delta_I, K) = 0.0
marginal_cost_I(::NoAdjustmentCost, I, Delta_I, K) = 0.0
marginal_cost_Delta_I(::NoAdjustmentCost, I, Delta_I, K) = 0.0
has_fixed_cost(::NoAdjustmentCost) = false
is_differentiable(::NoAdjustmentCost) = true

# =============================================================================
# 2. Convex Adjustment Cost (Standard)
# =============================================================================

"""
    ConvexAdjustmentCost

Standard quadratic adjustment cost on total investment:
C(I, Delta_I, K) = (phi/2) * ((I + Delta_I) / K)^2 * K
"""
@with_kw struct ConvexAdjustmentCost <: AbstractAdjustmentCost
    phi::Float64 = 1.0

    function ConvexAdjustmentCost(phi)
        @assert phi >= 0.0 "phi must be non-negative"
        new(phi)
    end
end

function compute_cost(ac::ConvexAdjustmentCost, I, Delta_I, K)
    I_total = I + Delta_I
    return 0.5 * ac.phi * (I_total / K)^2 * K
end

function marginal_cost_I(ac::ConvexAdjustmentCost, I, Delta_I, K)
    I_total = I + Delta_I
    return ac.phi * (I_total / K)
end

function marginal_cost_Delta_I(ac::ConvexAdjustmentCost, I, Delta_I, K)
    I_total = I + Delta_I
    return ac.phi * (I_total / K)
end

has_fixed_cost(::ConvexAdjustmentCost) = false
is_differentiable(::ConvexAdjustmentCost) = true

# =============================================================================
# 3. Fixed Adjustment Cost
# =============================================================================

"""
    FixedAdjustmentCost

Fixed cost paid whenever total investment is non-zero:
C(I, Delta_I, K) = F * 𝟙{I + Delta_I ≠ 0}
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

function compute_cost(ac::FixedAdjustmentCost, I, Delta_I, K)
    I_total = I + Delta_I
    return abs(I_total) > ac.threshold ? ac.F : 0.0
end

function marginal_cost_I(ac::FixedAdjustmentCost, I, Delta_I, K)
    # Marginal cost is zero except at discontinuity
    return 0.0
end

function marginal_cost_Delta_I(ac::FixedAdjustmentCost, I, Delta_I, K)
    return 0.0
end

has_fixed_cost(::FixedAdjustmentCost) = true
is_differentiable(::FixedAdjustmentCost) = false

# =============================================================================
# 4. Composite Adjustment Cost
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

function compute_cost(ac::CompositeAdjustmentCost, I, Delta_I, K)
    return sum(compute_cost(c, I, Delta_I, K) for c in ac.components)
end

function marginal_cost_I(ac::CompositeAdjustmentCost, I, Delta_I, K)
    return sum(marginal_cost_I(c, I, Delta_I, K) for c in ac.components)
end

function marginal_cost_Delta_I(ac::CompositeAdjustmentCost, I, Delta_I, K)
    return sum(marginal_cost_Delta_I(c, I, Delta_I, K) for c in ac.components)
end

function has_fixed_cost(ac::CompositeAdjustmentCost)
    return any(has_fixed_cost(c) for c in ac.components)
end

function is_differentiable(ac::CompositeAdjustmentCost)
    return all(is_differentiable(c) for c in ac.components)
end

# =============================================================================
# 5. Convex Adjustment Cost with Cross-Stage Dependency
# =============================================================================

"""
    ConvexCrossStageAdjustmentCost

Convex adjustment cost where the mid-year cost depends on both ΔI and the
beginning-of-year investment I (passed via the `I` argument).

At beginning of year (stage 0, called with Delta_I=0):
    C(I, 0, K) = (phi_begin/2) * (I/K)^2 * K

At mid-year (stage 1, called with I=0, but I_begin passed separately):
    C(0, ΔI, K) = (phi_mid/2) * (ΔI/K)^2 * K + phi_cross * |I_begin * ΔI| / K

The cross term `phi_cross * |I_begin * ΔI| / K` penalises mid-year revisions
that are large relative to the initial investment, capturing the idea that
revising a large initial commitment is costly.

To use the cross term, the caller must pass `I_begin` via the `I` argument
at mid-year instead of 0. See the Bellman operator for the calling convention.
"""
@with_kw struct ConvexCrossStageAdjustmentCost <: AbstractAdjustmentCost
    phi_begin::Float64 = 1.0    # Convex cost parameter for beginning-of-year
    phi_mid::Float64 = 1.0      # Convex cost parameter for mid-year ΔI
    phi_cross::Float64 = 0.0    # Cross-term coupling I and ΔI

    function ConvexCrossStageAdjustmentCost(phi_begin, phi_mid, phi_cross)
        @assert phi_begin >= 0.0 "phi_begin must be non-negative"
        @assert phi_mid >= 0.0 "phi_mid must be non-negative"
        @assert phi_cross >= 0.0 "phi_cross must be non-negative"
        new(phi_begin, phi_mid, phi_cross)
    end
end

function compute_cost(ac::ConvexCrossStageAdjustmentCost, I, Delta_I, K)
    # Stage 0 cost: convex in I
    cost_begin = 0.5 * ac.phi_begin * (I / K)^2 * K
    # Stage 1 cost: convex in ΔI + cross-term coupling I and ΔI
    cost_mid = 0.5 * ac.phi_mid * (Delta_I / K)^2 * K
    cost_cross = ac.phi_cross * abs(I * Delta_I) / K
    return cost_begin + cost_mid + cost_cross
end

function marginal_cost_I(ac::ConvexCrossStageAdjustmentCost, I, Delta_I, K)
    mc_begin = ac.phi_begin * (I / K)
    mc_cross = ac.phi_cross * abs(Delta_I) * sign(I) / K
    return mc_begin + mc_cross
end

function marginal_cost_Delta_I(ac::ConvexCrossStageAdjustmentCost, I, Delta_I, K)
    mc_mid = ac.phi_mid * (Delta_I / K)
    mc_cross = ac.phi_cross * abs(I) * sign(Delta_I) / K
    return mc_mid + mc_cross
end

has_fixed_cost(::ConvexCrossStageAdjustmentCost) = false
is_differentiable(::ConvexCrossStageAdjustmentCost) = true

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
        return "Convex: ($(ac.phi)/2) * (I_total/K)²"
    elseif ac isa FixedAdjustmentCost
        return "Fixed cost: F=$(ac.F)"
    elseif ac isa ConvexCrossStageAdjustmentCost
        return "ConvexCrossStage: phi_begin=$(ac.phi_begin), phi_mid=$(ac.phi_mid), phi_cross=$(ac.phi_cross)"
    elseif ac isa CompositeAdjustmentCost
        desc = "Composite: " * join([describe_adjustment_cost(c) for c in ac.components], " + ")
        return desc
    else
        return "Custom adjustment cost"
    end
end

"""
    total_adjustment_cost(ac, I, Delta_I, K) -> Float64

Alias for compute_cost for readability.
"""
total_adjustment_cost(ac, I, Delta_I, K) = compute_cost(ac, I, Delta_I, K)
