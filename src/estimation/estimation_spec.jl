"""
Flexible estimation specification for SMM.

Defines what parameters to estimate, how they map to adjustment costs,
and which moments to compute from simulated data.

# Economic context
The estimation framework can target different adjustment cost structures:
- Composite (fixed + convex at both stages): 4 parameters, 4 moments
- Convex only: 2 parameters (phi_begin, phi_mid), 2 regression moments
- Fixed only: 2 parameters (F_begin, F_mid), 2 share-of-zero moments
- Custom combinations via user-defined EstimationSpec

# Design
Each EstimationSpec bundles:
1. Parameter names, bounds, and mappings to adjustment cost constructors
2. Moment definitions (share-of-zero or regression coefficient via @formula)
3. Optional fixed (non-estimated) cost components at each stage
"""

using StatsModels: FormulaTerm, @formula, Term, term, ConstantTerm

# =============================================================================
# Moment types
# =============================================================================

"""
    AbstractMoment

Base type for moment specifications. Each subtype defines one moment
to be computed from the simulated firm panel.
"""
abstract type AbstractMoment end

"""
    ShareZeroMoment <: AbstractMoment

Moment: share of observations with near-zero investment revisions.

# Fields
- `stage`: `:begin` (beginning-of-year revision) or `:mid` (mid-year revision)
- `name`: Human-readable label for output/logging

# Economic interpretation
Measures the extensive margin of investment adjustment. A high share of zeros
indicates substantial fixed costs that induce inaction regions.
"""
struct ShareZeroMoment <: AbstractMoment
    stage::Symbol
    name::String

    function ShareZeroMoment(stage::Symbol, name::String)
        @assert stage in (:begin, :mid) "stage must be :begin or :mid"
        new(stage, name)
    end
end

"""
    RegressionCoefficientMoment <: AbstractMoment

Moment: OLS regression coefficient from a fixest-style formula.

Uses FixedEffectModels.jl `reg()` to estimate the regression, then extracts
the coefficient on the variable of interest.

# Fields
- `stage`: `:begin` or `:mid` — determines which observations are used
- `formula`: StatsModels `FormulaTerm` (e.g., `@formula(revision_begin ~ log_sigma + log_K + log_D)`)
- `coef_name`: Symbol naming the coefficient to extract (e.g., `:log_sigma`)
- `name`: Human-readable label for output/logging

# Economic interpretation
Regression coefficients capture the intensive margin response of investment
revisions to state variables (uncertainty, capital, demand). The coefficient
on log(σ) measures how strongly firms revise investment in response to
uncertainty changes.

# Example
```julia
RegressionCoefficientMoment(
    :begin,
    @formula(revision_begin ~ log_sigma + log_K + log_D),
    :log_sigma,
    "coef_begin_sigma"
)
```
"""
struct RegressionCoefficientMoment <: AbstractMoment
    stage::Symbol
    formula::FormulaTerm
    coef_name::Symbol
    name::String

    function RegressionCoefficientMoment(stage::Symbol, formula::FormulaTerm,
                                         coef_name::Symbol, name::String)
        @assert stage in (:begin, :mid) "stage must be :begin or :mid"
        new(stage, formula, coef_name, name)
    end
end

"""
    moment_name(m::AbstractMoment) -> String

Return the human-readable name of a moment.
"""
moment_name(m::AbstractMoment) = m.name

# =============================================================================
# Parameter-to-cost mapping
# =============================================================================

"""
    CostParameterMapping

Maps an estimated parameter in theta to a field of an adjustment cost constructor.

# Fields
- `param_name`: Name in the theta vector (e.g., `:phi_begin`)
- `stage`: `:begin` or `:mid` — which stage this cost applies to
- `cost_type`: Type of adjustment cost (e.g., `ConvexAdjustmentCost`)
- `field_name`: Field in the cost constructor to set (e.g., `:phi`)
- `defaults`: Default values for non-estimated fields of this cost type

# Example
```julia
CostParameterMapping(:phi_begin, :begin, ConvexAdjustmentCost, :phi, Dict{Symbol,Any}())
```
maps theta[i] (named :phi_begin) to `ConvexAdjustmentCost(phi=theta[i])` at stage :begin.
"""
struct CostParameterMapping
    param_name::Symbol
    stage::Symbol
    cost_type::Type{<:AbstractAdjustmentCost}
    field_name::Symbol
    defaults::Dict{Symbol, Any}

    function CostParameterMapping(param_name::Symbol, stage::Symbol,
                                   cost_type::Type{<:AbstractAdjustmentCost},
                                   field_name::Symbol,
                                   defaults::Dict{Symbol, Any}=Dict{Symbol,Any}())
        @assert stage in (:begin, :mid) "stage must be :begin or :mid"
        new(param_name, stage, cost_type, field_name, defaults)
    end
end

# =============================================================================
# EstimationSpec
# =============================================================================

"""
    EstimationSpec

Complete specification of what to estimate and how.

# Fields
- `param_names`: Ordered names for theta elements
- `param_mappings`: How each theta element maps to a cost constructor field
- `lower_bounds`, `upper_bounds`: Parameter bounds for optimization
- `moments`: Which moments to compute from simulated data
- `fixed_ac_begin`, `fixed_ac_mid`: Non-estimated cost components (e.g., a fixed
  NoAdjustmentCost, or a ConvexAdjustmentCost with known phi)

# Identification requirement
`length(moments) >= length(param_names)` — at least as many moments as parameters.

# Example
```julia
spec = convex_only_spec()  # 2 params (phi_begin, phi_mid), 2 regression moments
config = SMMConfig(estimation_spec=spec, m_data=[-0.15, 0.10])
```
"""
struct EstimationSpec
    param_names::Vector{Symbol}
    param_mappings::Vector{CostParameterMapping}
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    moments::Vector{AbstractMoment}
    fixed_ac_begin::Union{Nothing, AbstractAdjustmentCost}
    fixed_ac_mid::Union{Nothing, AbstractAdjustmentCost}

    function EstimationSpec(param_names, param_mappings, lower_bounds, upper_bounds,
                            moments, fixed_ac_begin, fixed_ac_mid)
        np = length(param_names)
        nm = length(moments)
        @assert length(param_mappings) == np "param_mappings must match param_names length"
        @assert length(lower_bounds) == np "lower_bounds must match param_names length"
        @assert length(upper_bounds) == np "upper_bounds must match param_names length"
        @assert all(lower_bounds .<= upper_bounds) "lower_bounds must be <= upper_bounds"
        @assert nm >= np "Need at least as many moments ($nm) as parameters ($np) for identification"
        @assert all(pm.param_name == pn for (pm, pn) in zip(param_mappings, param_names)) "param_mappings names must match param_names"
        new(param_names, param_mappings, lower_bounds, upper_bounds,
            moments, fixed_ac_begin, fixed_ac_mid)
    end
end

"""
    n_params(spec::EstimationSpec) -> Int

Number of parameters to estimate.
"""
n_params(spec::EstimationSpec) = length(spec.param_names)

"""
    n_moments(spec::EstimationSpec) -> Int

Number of moments to match.
"""
n_moments(spec::EstimationSpec) = length(spec.moments)

"""
    moment_names(spec::EstimationSpec) -> Vector{String}

Human-readable names of all moments in the spec.
"""
moment_names(spec::EstimationSpec) = [moment_name(m) for m in spec.moments]

# =============================================================================
# build_adjustment_costs: theta -> (ac_begin, ac_mid)
# =============================================================================

"""
    build_adjustment_costs(theta::Vector{Float64}, spec::EstimationSpec)
        -> (ac_begin::AbstractAdjustmentCost, ac_mid::AbstractAdjustmentCost)

Construct stage-specific adjustment costs from parameter vector theta.

Groups parameter mappings by stage, constructs cost instances with the
appropriate field values, and wraps in CompositeAdjustmentCost if needed.
Fixed (non-estimated) cost components are included if specified.

# Economic interpretation
This is the structural mapping from reduced-form parameter vector theta
to the economic objects (adjustment costs) used in the Bellman operators.
"""
function build_adjustment_costs(theta::Vector{Float64}, spec::EstimationSpec)
    @assert length(theta) == n_params(spec) "theta length ($(length(theta))) must match n_params ($(n_params(spec)))"

    # Group mappings by (stage, cost_type) to handle multiple fields of the same cost type
    # Key: (stage, cost_type) -> Dict{field_name => value}
    cost_fields = Dict{Tuple{Symbol, DataType}, Dict{Symbol, Any}}()

    for (i, mapping) in enumerate(spec.param_mappings)
        key = (mapping.stage, mapping.cost_type)
        if !haskey(cost_fields, key)
            cost_fields[key] = copy(mapping.defaults)
        end
        cost_fields[key][mapping.field_name] = theta[i]
    end

    # Construct cost instances for each stage
    begin_costs = AbstractAdjustmentCost[]
    mid_costs = AbstractAdjustmentCost[]

    for ((stage, cost_type), fields) in cost_fields
        ac = _construct_cost(cost_type, fields)
        if stage == :begin
            push!(begin_costs, ac)
        else
            push!(mid_costs, ac)
        end
    end

    # Add fixed (non-estimated) components
    if !isnothing(spec.fixed_ac_begin)
        push!(begin_costs, spec.fixed_ac_begin)
    end
    if !isnothing(spec.fixed_ac_mid)
        push!(mid_costs, spec.fixed_ac_mid)
    end

    # Wrap in CompositeAdjustmentCost or use NoAdjustmentCost
    ac_begin = _wrap_costs(begin_costs)
    ac_mid = _wrap_costs(mid_costs)

    return (ac_begin, ac_mid)
end

"""
    _construct_cost(cost_type, fields) -> AbstractAdjustmentCost

Construct a cost instance from a type and a dictionary of field values.
Uses keyword constructor when available.
"""
function _construct_cost(cost_type::Type{<:AbstractAdjustmentCost}, fields::Dict{Symbol, Any})
    if cost_type == ConvexAdjustmentCost
        return ConvexAdjustmentCost(phi=get(fields, :phi, 1.0))
    elseif cost_type == FixedAdjustmentCost
        return FixedAdjustmentCost(
            F=get(fields, :F, 0.1),
            threshold=get(fields, :threshold, 1e-6)
        )
    elseif cost_type == ConvexCrossStageAdjustmentCost
        return ConvexCrossStageAdjustmentCost(
            phi_begin=get(fields, :phi_begin, 1.0),
            phi_mid=get(fields, :phi_mid, 1.0),
            phi_cross=get(fields, :phi_cross, 0.0)
        )
    else
        error("Unknown cost type: $cost_type. Add a constructor case in _construct_cost.")
    end
end

"""
    _wrap_costs(costs) -> AbstractAdjustmentCost

Wrap a vector of cost components into a single cost object.
"""
function _wrap_costs(costs::Vector{AbstractAdjustmentCost})
    if isempty(costs)
        return NoAdjustmentCost()
    elseif length(costs) == 1
        return costs[1]
    else
        return CompositeAdjustmentCost(costs)
    end
end

# =============================================================================
# Convenience constructors for common specifications
# =============================================================================

"""
    composite_spec(; F_begin_bounds=(0.0, 10.0), F_mid_bounds=(0.0, 10.0),
                     phi_begin_bounds=(0.0, 20.0), phi_mid_bounds=(0.0, 20.0))

Default 4-parameter specification: fixed + convex costs at both stages.

Parameters: [F_begin, F_mid, phi_begin, phi_mid]
Moments: [share_zero_begin, share_zero_mid, coef_begin, coef_mid]

This reproduces the original hardcoded SMM specification.
"""
function composite_spec(;
    F_begin_bounds::Tuple{Float64,Float64} = (0.0, 10.0),
    F_mid_bounds::Tuple{Float64,Float64} = (0.0, 10.0),
    phi_begin_bounds::Tuple{Float64,Float64} = (0.0, 20.0),
    phi_mid_bounds::Tuple{Float64,Float64} = (0.0, 20.0)
)
    param_names = [:F_begin, :F_mid, :phi_begin, :phi_mid]

    param_mappings = [
        CostParameterMapping(:F_begin, :begin, FixedAdjustmentCost, :F),
        CostParameterMapping(:F_mid, :mid, FixedAdjustmentCost, :F),
        CostParameterMapping(:phi_begin, :begin, ConvexAdjustmentCost, :phi),
        CostParameterMapping(:phi_mid, :mid, ConvexAdjustmentCost, :phi),
    ]

    lower_bounds = [F_begin_bounds[1], F_mid_bounds[1], phi_begin_bounds[1], phi_mid_bounds[1]]
    upper_bounds = [F_begin_bounds[2], F_mid_bounds[2], phi_begin_bounds[2], phi_mid_bounds[2]]

    moments = AbstractMoment[
        ShareZeroMoment(:begin, "share_zero_begin"),
        ShareZeroMoment(:mid, "share_zero_mid"),
        RegressionCoefficientMoment(
            :begin,
            @formula(revision_begin ~ log_sigma + log_K + log_D),
            :log_sigma,
            "coef_begin_sigma"
        ),
        RegressionCoefficientMoment(
            :mid,
            @formula(revision_mid ~ log_sigma_half + log_K + log_D),
            :log_sigma_half,
            "coef_mid_sigma"
        ),
    ]

    return EstimationSpec(param_names, param_mappings, lower_bounds, upper_bounds,
                          moments, nothing, nothing)
end

"""
    convex_only_spec(; phi_begin_bounds=(0.0, 20.0), phi_mid_bounds=(0.0, 20.0))

2-parameter specification: convex costs only at both stages.

Parameters: [phi_begin, phi_mid]
Moments: [coef_begin, coef_mid] (regression coefficients on log uncertainty)

Use when assuming no fixed adjustment costs (F=0).
"""
function convex_only_spec(;
    phi_begin_bounds::Tuple{Float64,Float64} = (0.0, 20.0),
    phi_mid_bounds::Tuple{Float64,Float64} = (0.0, 20.0)
)
    param_names = [:phi_begin, :phi_mid]

    param_mappings = [
        CostParameterMapping(:phi_begin, :begin, ConvexAdjustmentCost, :phi),
        CostParameterMapping(:phi_mid, :mid, ConvexAdjustmentCost, :phi),
    ]

    lower_bounds = [phi_begin_bounds[1], phi_mid_bounds[1]]
    upper_bounds = [phi_begin_bounds[2], phi_mid_bounds[2]]

    moments = AbstractMoment[
        RegressionCoefficientMoment(
            :begin,
            @formula(revision_begin ~ log_sigma + log_K + log_D),
            :log_sigma,
            "coef_begin_sigma"
        ),
        RegressionCoefficientMoment(
            :mid,
            @formula(revision_mid ~ log_sigma_half + log_K + log_D),
            :log_sigma_half,
            "coef_mid_sigma"
        ),
    ]

    return EstimationSpec(param_names, param_mappings, lower_bounds, upper_bounds,
                          moments, nothing, nothing)
end

"""
    fixed_only_spec(; F_begin_bounds=(0.0, 10.0), F_mid_bounds=(0.0, 10.0))

2-parameter specification: fixed costs only at both stages.

Parameters: [F_begin, F_mid]
Moments: [share_zero_begin, share_zero_mid]

Use when assuming no convex adjustment costs (phi=0).
"""
function fixed_only_spec(;
    F_begin_bounds::Tuple{Float64,Float64} = (0.0, 10.0),
    F_mid_bounds::Tuple{Float64,Float64} = (0.0, 10.0)
)
    param_names = [:F_begin, :F_mid]

    param_mappings = [
        CostParameterMapping(:F_begin, :begin, FixedAdjustmentCost, :F),
        CostParameterMapping(:F_mid, :mid, FixedAdjustmentCost, :F),
    ]

    lower_bounds = [F_begin_bounds[1], F_mid_bounds[1]]
    upper_bounds = [F_begin_bounds[2], F_mid_bounds[2]]

    moments = AbstractMoment[
        ShareZeroMoment(:begin, "share_zero_begin"),
        ShareZeroMoment(:mid, "share_zero_mid"),
    ]

    return EstimationSpec(param_names, param_mappings, lower_bounds, upper_bounds,
                          moments, nothing, nothing)
end
