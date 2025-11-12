using Plots, Statistics, Distributions, Interpolations
using Distributed

# ============================================================================
# PARALLELIZATION SETUP
# ============================================================================


function setup_workers(max_workers::Int=100)
    """
        setup_workers(max_workers::Int=100)

        Initialize parallel workers for computation.
    """
    current_workers = nworkers()
    
    if current_workers < max_workers
        workers_to_add = max_workers - current_workers
        addprocs(workers_to_add)
        println("Added $workers_to_add workers. Total workers: $(nworkers())")
    else
        println("Already have $current_workers workers (max: $max_workers)")
    end
    
end
