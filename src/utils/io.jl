"""
Input/output utilities for saving and loading model solutions and results.
"""

using JLD2
using CSV
using DataFrames

"""
    save_solution(filename::String, sol::SolvedModel)

Save solved model to JLD2 file.

# Arguments
- `filename`: Output file path (should end in .jld2)
- `sol`: SolvedModel object
"""
function save_solution(filename::String, sol::SolvedModel)
    # Ensure directory exists
    dir = dirname(filename)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    # Save to JLD2
    jldsave(filename;
            params = sol.params,
            grids = sol.grids,
            ac = sol.ac,
            V = sol.V,
            I_policy = sol.I_policy,
            Delta_I_policy = sol.Delta_I_policy,
            convergence = sol.convergence)

    println("Solution saved to: $filename")
end

"""
    load_solution(filename::String) -> SolvedModel

Load solved model from JLD2 file.

# Arguments
- `filename`: Input file path

# Returns
- SolvedModel object
"""
function load_solution(filename::String)
    @assert isfile(filename) "File not found: $filename"

    # Load from JLD2
    data = load(filename)

    return SolvedModel(
        data["params"],
        data["grids"],
        data["ac"],
        data["V"],
        data["I_policy"],
        data["Delta_I_policy"],
        data["convergence"]
    )
end

"""
    export_policy_to_csv(sol::SolvedModel, filename::String; subset_K=nothing)

Export policy functions to CSV for analysis.

# Arguments
- `sol`: SolvedModel object
- `filename`: Output CSV file path
- `subset_K`: Optional vector of K indices to export (default: all)
"""
function export_policy_to_csv(sol::SolvedModel, filename::String; subset_K=nothing)
    # Ensure directory exists
    dir = dirname(filename)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    grids = sol.grids

    # Determine which K indices to export
    K_indices = isnothing(subset_K) ? (1:grids.n_K) : subset_K

    # Build DataFrame
    rows = []

    for i_K in K_indices
        K = get_K(grids, i_K)

        for i_sigma in 1:grids.n_sigma
            sigma_level = get_sigma(grids, i_sigma)
            log_sigma = get_log_sigma(grids, i_sigma)

            for i_D in 1:grids.n_D
                D_level = get_D(grids, i_D)
                log_D = get_log_D(grids, i_D)

                I_initial = sol.I_policy[i_K, i_D, i_sigma]
                V_value = sol.V[i_K, i_D, i_sigma]

                push!(rows, (
                    K = K,
                    D = D_level,
                    sigma = sigma_level,
                    log_D = log_D,
                    log_sigma = log_sigma,
                    I_initial = I_initial,
                    V = V_value
                ))
            end
        end
    end

    df = DataFrame(rows)
    CSV.write(filename, df)

    println("Policy functions exported to: $filename")
end

"""
    export_value_function_to_csv(sol::SolvedModel, filename::String)

Export value function to CSV.
"""
function export_value_function_to_csv(sol::SolvedModel, filename::String)
    # Same as export_policy_to_csv but focuses on value function
    export_policy_to_csv(sol, filename)
end

"""
    save_simulation(filename::String, panel::FirmPanel)

Save simulated firm panel to CSV.

# Arguments
- `filename`: Output CSV file path
- `panel`: FirmPanel object
"""
function save_simulation(filename::String, panel::FirmPanel)
    # Ensure directory exists
    dir = dirname(filename)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    CSV.write(filename, panel.df)
    println("Simulation saved to: $filename")
end

"""
    load_simulation(filename::String) -> DataFrame

Load simulated firm panel from CSV.

# Arguments
- `filename`: Input CSV file path

# Returns
- DataFrame with simulation data
"""
function load_simulation(filename::String)
    @assert isfile(filename) "File not found: $filename"
    return CSV.read(filename, DataFrame)
end

"""
    save_estimation_results(filename::String, result::EstimationResult)

Save GMM estimation results to CSV and JLD2.

# Arguments
- `filename`: Base output file path (without extension)
- `result`: EstimationResult object

Creates two files:
- filename_summary.csv: Parameter estimates with standard errors
- filename_full.jld2: Full estimation results
"""
function save_estimation_results(filename::String, result::EstimationResult)
    # Ensure directory exists
    dir = dirname(filename)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    # Save summary to CSV
    summary_file = filename * "_summary.csv"
    df_summary = DataFrame(
        parameter = result.param_names,
        estimate = result.theta_hat,
        std_error = result.se,
        t_stat = result.theta_hat ./ result.se
    )
    CSV.write(summary_file, df_summary)

    # Save full results to JLD2
    full_file = filename * "_full.jld2"
    jldsave(full_file;
            theta_hat = result.theta_hat,
            se = result.se,
            param_names = result.param_names,
            objective_value = result.objective_value,
            convergence = result.convergence,
            iterations = result.iterations,
            beta_sim = result.beta_sim,
            beta_data = result.beta_data,
            W = result.W)

    println("Estimation results saved to:")
    println("  - $summary_file")
    println("  - $full_file")
end

"""
    load_estimation_results(filename::String) -> EstimationResult

Load GMM estimation results from JLD2 file.

# Arguments
- `filename`: Input file path (JLD2 file)

# Returns
- EstimationResult object
"""
function load_estimation_results(filename::String)
    @assert isfile(filename) "File not found: $filename"

    data = load(filename)

    return EstimationResult(
        data["theta_hat"],
        data["se"],
        data["param_names"],
        data["objective_value"],
        data["convergence"],
        data["iterations"],
        data["beta_sim"],
        data["beta_data"],
        data["W"]
    )
end

"""
    export_to_csv(sol::SolvedModel, output_dir::String)

Export all model outputs to CSV files in specified directory.

Creates multiple files:
- policy_functions.csv: Investment policies
- value_function.csv: Value function
- grids.csv: Grid information

# Arguments
- `sol`: SolvedModel object
- `output_dir`: Output directory path
"""
function export_to_csv(sol::SolvedModel, output_dir::String)
    # Create output directory
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    # Export policy functions
    policy_file = joinpath(output_dir, "policy_functions.csv")
    export_policy_to_csv(sol, policy_file)

    # Export grids info
    grids_file = joinpath(output_dir, "grids.csv")
    df_grids = DataFrame(
        i_K = 1:sol.grids.n_K,
        K = sol.grids.K_grid
    )
    CSV.write(grids_file, df_grids)

    # Export demand/volatility grids
    dv_file = joinpath(output_dir, "demand_volatility_grids.csv")
    df_dv = DataFrame(
        i_D = repeat(1:sol.grids.n_D, outer=sol.grids.n_sigma),
        i_sigma = repeat(1:sol.grids.n_sigma, inner=sol.grids.n_D),
        log_D = repeat(sol.grids.sv.D_grid, outer=sol.grids.n_sigma),
        log_sigma = repeat(sol.grids.sv.sigma_grid, inner=sol.grids.n_D),
        D = exp.(repeat(sol.grids.sv.D_grid, outer=sol.grids.n_sigma)),
        sigma = exp.(repeat(sol.grids.sv.sigma_grid, inner=sol.grids.n_D))
    )
    CSV.write(dv_file, df_dv)

    println("All outputs exported to: $output_dir")
end

"""
    create_output_directories(base_dir::String="output")

Create standard output directory structure.
"""
function create_output_directories(base_dir::String="output")
    dirs = [
        joinpath(base_dir, "solutions"),
        joinpath(base_dir, "simulations"),
        joinpath(base_dir, "estimates"),
        joinpath(base_dir, "figures")
    ]

    for dir in dirs
        if !isdir(dir)
            mkpath(dir)
            println("Created directory: $dir")
        end
    end
end
