"""
Estimation-related data structures.
"""

"""
    EstimationResult

Results from GMM estimation via indirect inference.

# Fields
- `theta_hat::Vector{Float64}`: Estimated parameter values
- `se::Vector{Float64}`: Standard errors of parameter estimates
- `param_names::Vector{String}`: Names of estimated parameters
- `objective_value::Float64`: Value of GMM objective function at optimum
- `convergence::Bool`: Whether estimation converged successfully
- `iterations::Int`: Number of optimization iterations
- `beta_sim::Vector{Float64}`: Simulated moments/coefficients
- `beta_data::Vector{Float64}`: Data moments/coefficients
- `W::Matrix{Float64}`: Weighting matrix used in GMM
"""
struct EstimationResult
    theta_hat::Vector{Float64}
    se::Vector{Float64}
    param_names::Vector{String}
    objective_value::Float64
    convergence::Bool
    iterations::Int
    beta_sim::Vector{Float64}
    beta_data::Vector{Float64}
    W::Matrix{Float64}
end
